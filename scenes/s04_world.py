"""s04 — Flagartha's Geomantic Command Center: a pentagonal basalt chamber (columnar-basalt walls, coffered
pentagonal vault), the pentagonal descent shaft above it, a polished inlaid floor with the summoning seal,
five consoles, the pentagon map-table and its pedestal.  World units: cm, Y up, floor Y=0.
"""
import math

import cairo
import cv2
import numpy as np

from vx.canvas import col
from vx.config import FONT_MONO
from scenes.s04_geo import ngon, pent, poly_screen, path_pts, on_screen

RC = 5600.0            # chamber circumradius at the floor
YW = 2600.0            # wall top
RS = 1800.0            # shaft circumradius
YV = 5600.0            # shaft mouth (vault apex)
SHAFT_TOP = 17500.0
RT = 112.0             # map table circumradius
YT = 96.0              # table-top height
PED_H = 14.0           # pedestal height above the table top
PED_Y = YT + PED_H     # socket height (the hypermeme Flag's pole base)
CONSOLE_R = 800.0
SEAL_C = np.array([330.0, 0.0, 2100.0])
SEAL_R = 430.0

BASALT = np.array(col("#2E3750"), np.float32)
BASALT_D = np.array(col("#262B38"), np.float32)
AMB = np.array([0.03, 0.045, 0.11], np.float32)
FOG = np.array(col("#0E1528"), np.float32)
WARM = np.array(col("#FFA64A"), np.float32)
INLAY = np.array(col("#C08A45"), np.float32)


# ------------------------------------------------------------------ lighting
class Light:
    __slots__ = ("pos", "col", "power", "rad")

    def __init__(self, pos, c, power, rad):
        self.pos = np.asarray(pos, np.float32)
        self.col = np.asarray(c, np.float32)
        self.power = float(power)
        self.rad = float(rad)


def shade(P, Nn, albedo, lights, cam, fog_d=16000.0, fog_max=0.7, amb=AMB):
    """per-vertex colour: P (n,3), Nn (n,3) or (3,), albedo (3,)"""
    P = np.asarray(P, np.float32)
    Nn = np.broadcast_to(np.asarray(Nn, np.float32), P.shape)
    c = np.broadcast_to(amb * albedo * 3.0, P.shape).copy()
    for L in lights:
        d = L.pos - P
        dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
        ndl = np.clip((Nn * d).sum(1, keepdims=True) / dist, 0, 1)
        att = L.power / (1.0 + (dist / L.rad) ** 2)
        c += albedo * L.col * ndl * att
    dc = np.linalg.norm(P - cam.pos.astype(np.float32), axis=1, keepdims=True)
    fa = fog_max * (1 - np.exp(-dc / fog_d))
    return np.clip(c * (1 - fa) + FOG * fa, 0, 1)


def fog_amount(P, cam, fog_d=16000.0, fog_max=0.7):
    dc = float(np.linalg.norm(np.asarray(P) - cam.pos))
    return fog_max * (1 - math.exp(-dc / fog_d))


# ------------------------------------------------------------------ geometry (seeded)
def build(seed=5):
    rng = np.random.default_rng(seed)
    V = pent(RC, 0.0)                 # floor vertices
    walls = []
    for k in range(5):
        a, b = V[k], V[(k + 1) % 5]
        d = b - a
        L = float(np.linalg.norm(d))
        t_hat = d / L
        n_in = -np.array([a[0] + b[0], 0, a[2] + b[2]])
        n_in /= np.linalg.norm(n_in)
        ncol = 24
        cols = []
        for j in range(ncol):
            # a basalt column seen from inside: two facets meeting at a ridge (prism front), strong light rhythm
            u0, u1 = j / ncol, (j + 1) / ncol
            p0 = a + d * u0
            p1 = a + d * u1
            dep = rng.uniform(-50, 50)
            ridge = rng.uniform(55, 95)
            um = rng.uniform(0.35, 0.65)
            pm = a + d * (u0 + (u1 - u0) * um) + n_in * (dep + ridge)
            p0 = p0 + n_in * dep
            p1 = p1 + n_in * dep
            top = YW + rng.uniform(-80, 180)
            tone = rng.uniform(0.8, 1.15)
            for q0, q1 in ((p0, pm), (pm, p1)):
                e = q1 - q0
                nrm = np.array([-e[2], 0.0, e[0]])
                nrm /= np.linalg.norm(nrm) + 1e-9
                if np.dot(nrm, n_in) < 0:
                    nrm = -nrm
                cols.append(dict(P=np.array([[q0[0], 0, q0[2]], [q1[0], 0, q1[2]], [q1[0], top, q1[2]],
                                             [q0[0], top, q0[2]]]), n=nrm, tone=tone))
        walls.append(dict(a=a, b=b, n=n_in, cols=cols))
    # vault: frustum from (RC, YW) to (RS, YV) — panels
    vault = []
    Vt = pent(RC, YW)
    Vs = pent(RS, YV)
    rows, ncols = 5, 7
    for k in range(5):
        A0, A1 = Vt[k], Vt[(k + 1) % 5]
        B0, B1 = Vs[k], Vs[(k + 1) % 5]
        e1 = A1 - A0
        e2 = B0 - A0
        nrm = np.cross(e2, e1)
        nrm = nrm / np.linalg.norm(nrm)
        if nrm[1] > 0:
            nrm = -nrm
        for r_ in range(rows):
            for c_ in range(ncols):
                def pt(u, v):
                    lo = A0 + (A1 - A0) * u
                    hi = B0 + (B1 - B0) * u
                    return lo + (hi - lo) * v
                u0, u1 = c_ / ncols, (c_ + 1) / ncols
                v0, v1 = r_ / rows, (r_ + 1) / rows
                P = np.array([pt(u0, v0), pt(u1, v0), pt(u1, v1), pt(u0, v1)])
                vault.append(dict(P=P, n=nrm, tone=rng.uniform(0.85, 1.1), inset=(r_ + c_) % 2))
    # shaft: faces from YV to SHAFT_TOP
    shaft = []
    Vb = pent(RS, YV)
    for k in range(5):
        a, b = Vb[k], Vb[(k + 1) % 5]
        n_in = -np.array([a[0] + b[0], 0, a[2] + b[2]])
        n_in /= np.linalg.norm(n_in)
        ys = np.arange(YV, SHAFT_TOP, 700.0)
        for i, y in enumerate(ys):
            nc = 5
            for j in range(nc):
                p0 = a + (b - a) * (j / nc)
                p1 = a + (b - a) * ((j + 1) / nc)
                dep = rng.uniform(-25, 25)
                P = np.array([[p0[0], y, p0[2]], [p1[0], y, p1[2]], [p1[0], y + 700, p1[2]], [p0[0], y + 700, p0[2]]])
                P[:, [0, 2]] += n_in[[0, 2]] * dep
                shaft.append(dict(P=P, n=n_in, tone=rng.uniform(0.8, 1.15), ring=(i % 2 == 0)))
    consoles = []
    for k in range(5):
        th = k * 2 * math.pi / 5 + math.pi / 5
        consoles.append(dict(th=th, pos=np.array([CONSOLE_R * math.sin(th), 0.0, CONSOLE_R * math.cos(th)]),
                             seed=int(rng.integers(0, 1000))))
    return dict(walls=walls, vault=vault, shaft=shaft, consoles=consoles)


# ------------------------------------------------------------------ floor texture (baked)
FN = 2048


def floor_texture(cache_dir):
    path = cache_dir / "floor_tex_v7.npz"
    if path.exists():
        return np.load(path)["t"].astype(np.float32) / 255.0
    n = FN
    k = n / (2 * RC)

    def uv(X, Z):
        return n / 2 + X * k, n / 2 - Z * k

    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    X = (xx + 0.5 - n / 2) / k
    Z = -(yy + 0.5 - n / 2) / k
    r = np.hypot(X, Z)
    rng = np.random.default_rng(3)
    nz = cv2.resize(cv2.GaussianBlur(rng.random((n // 16, n // 16)).astype(np.float32), (0, 0), 2), (n, n))
    base = BASALT_D * (0.55 + 0.25 * nz[..., None])
    # light pool under the Hypermind + table glow (baked, warm)
    pool = np.exp(-(r / 2200.0) ** 2)[..., None]
    base = base + WARM * 0.10 * pool + np.array(col("#5AA7B8"), np.float32) * 0.06 * np.exp(-(r / 420.0) ** 2)[..., None]
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, n, n)
    ctx = cairo.Context(surf)
    ctx.set_line_join(cairo.LINE_JOIN_MITER)

    def pent_path(R_, rot=math.pi, cx=0.0, cz=0.0):
        P = ngon(5, R_, 0, rot, cx, cz)
        ctx.move_to(*uv(P[0, 0], P[0, 2]))
        for p in P[1:]:
            ctx.line_to(*uv(p[0], p[2]))
        ctx.close_path()

    # slab seams: concentric pentagons + radial lines
    for R_, w_, a_ in ((420, 3.5, 0.8), (1000, 3, 0.6), (1800, 2.5, 0.5), (2900, 3.5, 0.6), (4000, 2.5, 0.45), (5200, 4, 0.5)):
        pent_path(R_)
        ctx.set_source_rgba(*INLAY, a_)
        ctx.set_line_width(w_)
        ctx.stroke()
    V = pent(5200, 0)
    for p in V:
        ctx.move_to(*uv(0, 0))
        ctx.line_to(*uv(p[0], p[2]))
        ctx.set_source_rgba(*INLAY, 0.35)
        ctx.set_line_width(2.2)
        ctx.stroke()
    # great pentagram (lore: proper configurations resemble overlapping pentagrams)
    V = pent(2900, 0)
    for i in range(5):
        a, b = V[i], V[(i + 2) % 5]
        ctx.move_to(*uv(a[0], a[2]))
        ctx.line_to(*uv(b[0], b[2]))
        ctx.set_source_rgba(*INLAY, 0.45)
        ctx.set_line_width(2.8)
        ctx.stroke()
    # seal surround
    cx, cz = SEAL_C[0], SEAL_C[2]
    for R_, w_, a_ in ((SEAL_R + 40, 5, 0.9), (SEAL_R + 95, 2.5, 0.6)):
        ctx.arc(*uv(cx, cz), R_ * k, 0, 2 * math.pi)
        ctx.set_source_rgba(*INLAY, a_)
        ctx.set_line_width(w_)
        ctx.stroke()
    # tick ring around the seal
    for i in range(40):
        a = i / 40 * 2 * math.pi
        r0, r1 = (SEAL_R + 50) * k, (SEAL_R + 85) * k
        u0, v0 = uv(cx, cz)
        ctx.move_to(u0 + r0 * math.sin(a), v0 + r0 * math.cos(a))
        ctx.line_to(u0 + r1 * math.sin(a), v0 + r1 * math.cos(a))
        ctx.set_line_width(3)
        ctx.stroke()
    surf.flush()
    buf = np.ndarray((n, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :n].astype(np.float32) / 255
    lines = buf[..., [2, 1, 0]]
    la = buf[..., 3:4]
    img = base * (1 - la) + lines * (0.55 + 0.45 * pool)
    # outside the pentagon: transparent
    Vf = pent(RC, 0)
    mask = np.zeros((n, n), np.uint8)
    cv2.fillPoly(mask, [np.array([uv(p[0], p[2]) for p in Vf], np.int32)], 255)
    a = (mask.astype(np.float32) / 255)[..., None]
    out = np.concatenate([img * a, a], -1).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=(np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
    return out


# ------------------------------------------------------------------ plane texture warp
def plane_H(cam, Y, half, n, s, cx0=0.0, cz0=0.0):
    """3x3 homography: texture px (square, n px covering [cx0-half, cx0+half] x [cz0-half, cz0+half],
    +Z up in the texture) -> output pixels at render scale s"""
    k = 2 * half / n
    A = np.array([[k, 0, cx0 - n / 2 * k], [0, -k, cz0 + n / 2 * k], [0, 0, 1]], np.float64)
    R = cam.R
    M = np.stack([R[:, 0], R[:, 2], R[:, 1] * Y - R @ cam.pos], 1)
    K = np.array([[cam.f, 0, cam.cx], [0, -cam.f, cam.cy], [0, 0, 1]], np.float64)
    S = np.diag([s, s, 1.0])
    return S @ K @ M @ A


def front_mask(cam, Y, w, h, s):
    """per-pixel mask: 1 where the view ray hits the plane Y in front of the camera"""
    xs = (np.arange(w, dtype=np.float32) + 0.5) / s
    ys = (np.arange(h, dtype=np.float32) + 0.5) / s
    a = (xs - cam.cx) / cam.f
    b = -(ys - cam.cy) / cam.f
    R = cam.R
    # world direction y-component: d_y = R[0,1]*a + R[1,1]*b + R[2,1]
    dy = R[0, 1] * a[None, :] + R[1, 1] * b[:, None] + R[2, 1]
    t = (Y - cam.pos[1]) / np.where(np.abs(dy) < 1e-6, 1e-6, dy)
    return (t > 0).astype(np.float32)[..., None]


_MIPS = {}


def _mip(tex, lvl):
    key = (id(tex), lvl)
    m = _MIPS.get(key)
    if m is None:
        m = tex if lvl == 0 else cv2.pyrDown(_mip(tex, lvl - 1))
        _MIPS[key] = m
    return m


def _mip_level(Hm, n):
    c0 = Hm @ np.array([n / 2, n / 2, 1.0])
    c1 = Hm @ np.array([n / 2 + 1, n / 2, 1.0])
    c2 = Hm @ np.array([n / 2, n / 2 + 1, 1.0])
    lvl = 0
    if c0[2] > 0 and c1[2] > 0 and c2[2] > 0:
        p0 = c0[:2] / c0[2]
        fp = min(np.hypot(*(c1[:2] / c1[2] - p0)), np.hypot(*(c2[:2] / c2[2] - p0)))
        while fp < 0.5 and lvl < 3 and (n >> lvl) > 256:
            fp *= 2
            lvl += 1
    return lvl


def warp_plane(tex, cam, Y, half, fc, cx0=0.0, cz0=0.0, mip=True, roi=False):
    """project a square texture lying on plane Y into the frame.
    roi=False -> full frame (h,w,C) premultiplied, masked to the half-space in front of the camera.
    roi=True  -> (img, x0, y0) cropped to the projected square's bounding box (pixels), or None if not fully
                 in front of the camera / off-screen (caller falls back)."""
    n = tex.shape[0]
    Hm = plane_H(cam, Y, half, n, fc.s, cx0, cz0)
    lvl = _mip_level(Hm, n) if mip else 0
    src = _mip(tex, lvl)
    if lvl:
        sc = n / src.shape[0]
        Hm = Hm @ np.diag([sc, sc, 1.0])
    if roi:
        m = src.shape[0]
        C = Hm @ np.array([[0, 0, 1], [m, 0, 1], [m, m, 1], [0, m, 1]], np.float64).T
        if (C[2] <= 1e-6).any():
            return None
        xs, ys = C[0] / C[2], C[1] / C[2]
        x0, y0 = max(0, int(math.floor(xs.min())) - 1), max(0, int(math.floor(ys.min())) - 1)
        x1, y1 = min(fc.w, int(math.ceil(xs.max())) + 1), min(fc.h, int(math.ceil(ys.max())) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        Tm = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
        out = cv2.warpPerspective(src, Tm @ Hm, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if out.ndim == 2:
            out = out[..., None]
        return out, x0, y0
    out = cv2.warpPerspective(src, Hm, (fc.w, fc.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
    if out.ndim == 2:
        out = out[..., None]
    return out * front_mask(cam, Y, fc.w, fc.h, fc.s)


# ------------------------------------------------------------------ drawing: shell
def draw_shell(ctx, gctx, cam, geo, lights, T, shaft_first):
    """walls, vault, shaft. Painter: shaft first when the camera is below the mouth."""
    def draw_shaft():
        for q in geo["shaft"]:
            P = q["P"]
            if np.dot(cam.pos - P[0], q["n"]) <= 0:
                continue
            pts = poly_screen(cam, P)
            if pts is None or not on_screen(pts, 50):
                continue
            cs = shade(P, q["n"], BASALT * q["tone"], lights, cam)
            _grad_fill(ctx, cam, P, pts, cs)
        # pentagonal lamp rings every 1400 in the shaft (thin glowing strips on every wall)
        for y in np.arange(YV + 350, SHAFT_TOP, 1400.0):
            Vb = pent(RS * 0.99, y)
            Vt = pent(RS * 0.99, y + 36)
            fa = fog_amount(np.array([0, y, 0]), cam)
            hot = 0.55 + 0.45 * math.exp(-(y - YV) / 5000.0)
            for k in range(5):
                P = np.array([Vb[k], Vb[(k + 1) % 5], Vt[(k + 1) % 5], Vt[k]])
                n_in = -np.array([P[0, 0] + P[1, 0], 0, P[0, 2] + P[1, 2]])
                if np.dot(cam.pos - P[0], n_in) <= 0:
                    continue
                pts = poly_screen(cam, P)
                if pts is None or not on_screen(pts, 50):
                    continue
                path_pts(ctx, pts)
                ctx.set_source_rgba(*np.clip(col("#FFD9A0"), 0, 1), 0.95 * hot * (1 - fa))
                ctx.fill()
                path_pts(gctx, pts)
                gctx.set_source_rgba(*WARM, 0.8 * hot * (1 - fa))
                gctx.set_line_width(6)
                gctx.stroke_preserve()
                gctx.fill()

    def light_well():
        """volumetric warm haze filling the shaft mouth, seen from above (the chamber's light pouring up)"""
        ring = pent(RS, YV)
        pts = poly_screen(cam, ring)
        if pts is None:
            return
        c = cam.proj(np.array([0.0, YV, 0.0]))
        R_ = float(np.max(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])))
        for c_, a_ in ((ctx, 0.15), (gctx, 0.16)):
            g = cairo.RadialGradient(float(c[0]), float(c[1]), 0, float(c[0]), float(c[1]), R_)
            g.add_color_stop_rgba(0, *col("#FFE2B0"), a_)
            g.add_color_stop_rgba(0.6, *WARM, a_ * 0.7)
            g.add_color_stop_rgba(1, *WARM, a_ * 0.25)
            path_pts(c_, pts)
            c_.set_source(g)
            c_.fill()

    if shaft_first:
        draw_shaft()
    # walls
    for w in geo["walls"]:
        for c_ in w["cols"]:
            P = c_["P"]
            if np.dot(cam.pos - P[0], c_["n"]) <= 0:
                continue
            pts = poly_screen(cam, P)
            if pts is None or not on_screen(pts, 30):
                continue
            cs = shade(P, c_["n"], BASALT * c_["tone"], lights, cam)
            _grad_fill(ctx, cam, P, pts, cs)
    # wall inlay bands (gold strips near the floor and at the top): true 3D strips, glow outline
    for yb, hh, aa in ((300.0, 16.0, 0.8), (2230.0, 24.0, 0.9)):
        ring = pent(RC * 0.985, yb)
        for k in range(5):
            a_, b_ = ring[k], ring[(k + 1) % 5]
            m = (a_ + b_) / 2
            if np.dot(cam.pos - m, -np.array([m[0], 0, m[2]])) <= 0:
                continue
            P = np.array([a_, b_, b_ + [0, hh, 0], a_ + [0, hh, 0]])
            pts = poly_screen(cam, P)
            if pts is None or not on_screen(pts, 30):
                continue
            path_pts(ctx, pts)
            ctx.set_source_rgba(*np.clip(INLAY * 1.25, 0, 1), aa * (1 - fog_amount(m, cam)))
            ctx.fill()
            path_pts(gctx, pts)
            gctx.set_source_rgba(*WARM, 0.3)
            gctx.set_line_width(3.0)
            gctx.stroke_preserve()
            gctx.fill()
    # vault
    for q in geo["vault"]:
        P = q["P"]
        if np.dot(cam.pos - P[0], q["n"]) <= 0:
            continue
        pts = poly_screen(cam, P)
        if pts is None or not on_screen(pts, 30):
            continue
        cs = shade(P, q["n"], BASALT * q["tone"] * (0.9 if q["inset"] else 1.0), lights, cam)
        _grad_fill(ctx, cam, P, pts, cs)
    # vault ribs (gold strips)
    Vt = pent(RC, YW)
    Vs = pent(RS, YV)
    for k in range(5):
        a_, b_ = Vt[k], Vs[k]
        t_ = np.array([-a_[2], 0.0, a_[0]])
        t_ /= np.linalg.norm(t_)
        u_ = np.array([-b_[2], 0.0, b_[0]])
        u_ /= np.linalg.norm(u_)
        P = np.array([a_ - t_ * 28, a_ + t_ * 28, b_ + u_ * 18, b_ - u_ * 18]) + np.array([0, -4.0, 0])
        pts = poly_screen(cam, P)
        if pts is None or not on_screen(pts, 30):
            continue
        path_pts(ctx, pts)
        ctx.set_source_rgba(*np.clip(INLAY * 1.1, 0, 1), 0.85)
        ctx.fill()
    # mouth ring (flat gold collar)
    inner = pent(RS, YV - 2)
    outer = pent(RS + 90, YV - 2)
    for k in range(5):
        P = np.array([inner[k], inner[(k + 1) % 5], outer[(k + 1) % 5], outer[k]])
        pts = poly_screen(cam, P)
        if pts is None or not on_screen(pts, 30):
            continue
        path_pts(ctx, pts)
        ctx.set_source_rgba(*np.clip(INLAY * 1.3, 0, 1), 0.9)
        ctx.fill()
    if not shaft_first:
        light_well()
        draw_shaft()


def _grad_fill(ctx, cam, P, pts, cs):
    lum = cs @ np.array([0.3, 0.55, 0.15], np.float32)
    i0, i1 = int(np.argmin(lum)), int(np.argmax(lum))
    a = cam.proj(P[i0])
    b = cam.proj(P[i1])
    path_pts(ctx, pts)
    if a[2] > cam.near and b[2] > cam.near and lum[i1] - lum[i0] > 0.004:
        g = cairo.LinearGradient(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
        g.add_color_stop_rgb(0, *cs[i0])
        g.add_color_stop_rgb(1, *cs[i1])
        ctx.set_source(g)
    else:
        ctx.set_source_rgb(*cs.mean(0))
    ctx.fill_preserve()
    ctx.set_line_width(0.9)
    ctx.stroke()


# ------------------------------------------------------------------ boxes (consoles, table, pedestal)
def beam_faces(p0, p1, w, h):
    """box beam from p0 to p1 (world), width w (horizontal), height h (vertical)"""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    L = float(np.linalg.norm(d))
    ex = d / L
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    def p(a, b, c):
        return p0 + ex * a + ey * b + ez * c
    hw, hh = w / 2, h / 2
    return [
        (np.array([p(0, -hh, -hw), p(L, -hh, -hw), p(L, hh, -hw), p(0, hh, -hw)]), -ez),
        (np.array([p(0, -hh, hw), p(L, -hh, hw), p(L, hh, hw), p(0, hh, hw)]), ez),
        (np.array([p(0, hh, -hw), p(L, hh, -hw), p(L, hh, hw), p(0, hh, hw)]), ey),
        (np.array([p(0, -hh, -hw), p(L, -hh, -hw), p(L, -hh, hw), p(0, -hh, hw)]), -ey),
    ]


def draw_truss(ctx, gctx, cam, lights, hub_y):
    """the Hypermind's suspension hub: a pentagonal ring with five beams anchored in the shaft walls"""
    Vb = pent(RS * 0.97, hub_y)
    items = []
    for k in range(5):
        m = (Vb[k] + Vb[(k + 1) % 5]) / 2
        hub = m / np.linalg.norm(m[[0, 2]]) * 420
        hub[1] = hub_y
        for P, n in beam_faces(hub, m, 130, 190):
            items.append((float(cam.to_cam(P.mean(0))[2]), P, n))
    fs, top = prism_faces(460, hub_y - 160, hub_y + 160)
    for P, n in fs:
        items.append((float(cam.to_cam(P.mean(0))[2]), P, n))
    bot = ngon(5, 460, hub_y - 160, math.pi)
    items.append((float(cam.to_cam(bot.mean(0))[2]) + 1, bot[::-1], np.array([0, -1.0, 0])))
    items.append((float(cam.to_cam(top.mean(0))[2]) + 1, top, np.array([0, 1.0, 0])))
    items.sort(key=lambda x: -x[0])
    for _, P, n in items:
        if np.dot(cam.pos - P[0], n) <= 0:
            continue
        pts = poly_screen(cam, P)
        if pts is None or not on_screen(pts, 50):
            continue
        cs = shade(P, n, np.array(col("#6A4A22")), lights, cam)
        _grad_fill(ctx, cam, P, pts, cs)
    # gold rim of the hub (catches the light from below)
    pts = poly_screen(cam, bot)
    if pts is not None:
        z = float(np.mean(np.maximum(cam.to_cam(bot)[:, 2], cam.near)))
        path_pts(ctx, pts)
        ctx.set_source_rgba(*col("#FFC878"), 0.9)
        ctx.set_line_width(max(1, 16 * cam.f / z))
        ctx.stroke()


def box_faces(c, hw, hd, h, th, y0=0.0):
    """oriented box: centre c (x,z), half width hw (tangential), half depth hd (radial), height h,
    rotated by th about Y. -> list of (P(4,3), n)"""
    ct, st_ = math.cos(th), math.sin(th)
    ex = np.array([ct, 0, -st_])      # local x
    ez = np.array([st_, 0, ct])       # local z (outward radial)
    base = np.array([c[0], y0, c[2]])
    def p(x, y, z):
        return base + ex * x + ez * z + np.array([0, y, 0])
    fs = [
        (np.array([p(-hw, 0, -hd), p(hw, 0, -hd), p(hw, h, -hd), p(-hw, h, -hd)]), -ez),
        (np.array([p(hw, 0, hd), p(-hw, 0, hd), p(-hw, h, hd), p(hw, h, hd)]), ez),
        (np.array([p(hw, 0, -hd), p(hw, 0, hd), p(hw, h, hd), p(hw, h, -hd)]), ex),
        (np.array([p(-hw, 0, hd), p(-hw, 0, -hd), p(-hw, h, -hd), p(-hw, h, hd)]), -ex),
        (np.array([p(-hw, h, -hd), p(hw, h, -hd), p(hw, h, hd), p(-hw, h, hd)]), np.array([0, 1.0, 0])),
    ]
    return fs


def draw_faces(ctx, cam, faces, albedo, lights, emissive=None):
    for P, n in faces:
        if np.dot(cam.pos - P[0], n) <= 0:
            continue
        pts = poly_screen(cam, P)
        if pts is None:
            continue
        cs = shade(P, n, albedo, lights, cam)
        _grad_fill(ctx, cam, P, pts, cs)


def prism_faces(r, y0, y1, rot=math.pi, cx=0.0, cz=0.0, n=5):
    lo = ngon(n, r, y0, rot, cx, cz)
    hi = ngon(n, r, y1, rot, cx, cz)
    fs = []
    for k in range(n):
        P = np.array([lo[k], lo[(k + 1) % n], hi[(k + 1) % n], hi[k]])
        m = (lo[k] + lo[(k + 1) % n]) / 2
        nn = np.array([m[0] - cx, 0, m[2] - cz])
        nn /= np.linalg.norm(nn)
        fs.append((P, nn))
    return fs, hi

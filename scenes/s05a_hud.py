"""s05a K02 (96.08-98.42): 'Method: recursive balkanization.'
The Commander's wrist projector throws a holographic globe (dot-matrix Earth from a real land mask). Target
regions lock on and balkanize recursively: a true hierarchical spherical Voronoi (each cell splits into
children seeded inside it), rendered per pixel so old borders persist while new ones crack in."""
import math

import numpy as np

from vx import W, H, Canvas, set_color, mix, circle, poly, clamp, lerp, smoothstep, ease_out, ease_in_out, \
    ease_out_back, hash01, blur, text, FONT_MONO, ASSETS
from scenes.s05a_fx import HOLO, GOLD, GOLD_HOT, lin_grad, rad_grad, rimlit, radial_img, streak, comp, grid_xy, motes
from scenes import s05a_tunnel as tun

CHILDREN = 5
LEVELS = 5


def _unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)], -1)


def _rand_sphere(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def on_land(P, land):
    """bilinear-interpolated land test (smooth coastlines from the 1-degree mask)"""
    lat = np.degrees(np.arcsin(np.clip(P[:, 1], -1, 1)))
    lon = np.degrees(np.arctan2(P[:, 0], P[:, 2]))
    fi = np.clip(lat + 89.5, 0, 178.999)
    fj = (lon + 179.5) % 360.0
    i0 = fi.astype(np.int64)
    j0 = fj.astype(np.int64) % 360
    j1 = (j0 + 1) % 360
    a, b = fi - i0, fj - np.floor(fj)
    L = land.astype(np.float32) if land.dtype == bool else land
    v = (L[i0, j0] * (1 - a) * (1 - b) + L[i0, j1] * (1 - a) * b + L[i0 + 1, j0] * a * (1 - b)
         + L[i0 + 1, j1] * a * b)
    return v > 0.5


def build_hierarchy(center, radius_deg, n1, seed, land):
    """seeds[level] (n_level, 3); level 1 = n1 seeds on land inside a cap, then CHILDREN per parent sampled
    inside it. Region = cap ∩ land, so the cells follow real coastlines."""
    rng = np.random.default_rng(seed)
    c = _unit(*center)
    cosr = math.cos(math.radians(radius_deg))
    pts = _rand_sphere(rng, 600000)
    pts = pts[pts @ c > cosr]
    pts = pts[on_land(pts, land)]
    lab = np.zeros(len(pts), np.int64)
    # level 1: farthest-point-ish sampling for even cells
    idx = [int(rng.integers(len(pts)))]
    d = np.full(len(pts), -2.0)
    for _ in range(n1 - 1):
        d = np.maximum(d, pts @ pts[idx[-1]])
        cand = np.argsort(d)[:max(1, len(pts) // 30)]
        idx.append(int(rng.choice(cand)))
    seeds = [pts[idx]]
    lab = np.argmax(pts @ seeds[0].T, 1)
    for lv in range(2, LEVELS + 1):
        par = seeds[-1]
        ch = np.zeros((len(par), CHILDREN, 3))
        for p in range(len(par)):
            inside = np.nonzero(lab == p)[0]
            if len(inside) < CHILDREN:
                inside = np.argsort(-(pts @ par[p]))[:CHILDREN * 4]
            pick = rng.choice(inside, CHILDREN, replace=False)
            ch[p] = pts[pick]
        seeds.append(ch.reshape(-1, 3))
        # relabel samples at the new level (nearest child of their parent)
        cc = ch[lab]                                   # (n, CHILDREN, 3)
        k = np.argmax(np.einsum("nk,nck->nc", pts, cc), 1)
        lab = lab * CHILDREN + k
    return dict(c=c, cosr=cosr, seeds=seeds, land=land)


def labels_all(P, hier, depth):
    """hierarchical labels for levels 0..depth in one pass. P (n,3) -> list of (n,) int arrays (-1 outside),
    plus the parent seed (n,3) of every point at level `depth` (for the crack-in reveal)."""
    inside = (P @ hier["c"] > hier["cosr"]) & on_land(P, hier["land"])
    out = []
    lab0 = np.where(inside, 0, -1)
    out.append(lab0)
    if depth <= 0:
        return out, None
    Q = P[inside]
    l = np.argmax(Q @ hier["seeds"][0].T, 1)
    parent_seed = np.broadcast_to(hier["c"], Q.shape)
    full = np.full(len(P), -1, np.int64)
    full[inside] = l
    out.append(full)
    for lv in range(2, depth + 1):
        parent_seed = hier["seeds"][lv - 2][l]
        ch = hier["seeds"][lv - 1].reshape(-1, CHILDREN, 3)[l]
        k = np.argmax(np.einsum("nk,nck->nc", Q, ch), 1)
        l = l * CHILDREN + k
        full = np.full(len(P), -1, np.int64)
        full[inside] = l
        out.append(full)
    ps = np.zeros((len(P), 3))
    ps[inside] = parent_seed
    return out, ps


def setup(S, st):
    raw = np.load(ASSETS / "s05a" / "land_1deg.npy")
    land = np.unpackbits(raw)[:180 * 360].reshape(180, 360).astype(bool)
    # dot matrix every 2 degrees (lat -80..80)
    lats, lons = [], []
    for la in range(-78, 82, 2):
        step = 2.0 / max(0.25, math.cos(math.radians(la)))
        lo = -180.0
        while lo < 180.0:
            i, j = int(la + 90), int((lo + 180) % 360)
            if land[min(179, i), j]:
                lats.append(la)
                lons.append(lo)
            lo += step
    st["hud_dots"] = _unit(np.array(lats, float), np.array(lons, float))
    # three memeplex regions balkanize, staggered: the Nation (Europe/Near East), the Faith, the Philosophy
    tl = S.tl
    t_rec = tl.word("K02", "recursive")[0]
    t_bal = tl.word("K02", "balkanization")[0]
    st["hud_regions"] = [
        dict(h=build_hierarchy((50, 14), 24, 6, 11, land), t0=t_rec, dt=0.29),
        dict(h=build_hierarchy((30, 58), 19, 5, 12, land), t0=t_bal + 0.22, dt=0.26),
        dict(h=build_hierarchy((2, 21), 27, 6, 13, land), t0=t_bal + 0.45, dt=0.24),
    ]
    st["hud_t"] = dict(method=tl.word("K02", "Method")[0], rec=t_rec, bal=t_bal,
                       end=tl.line("K02")["end"])
    st["commander_hud"] = tun.Character("commander", seed=1, goggles_down=False)


def region_depth(reg, T):
    if T < reg["t0"]:
        return -1, 0.0
    k = (T - reg["t0"]) / reg["dt"]
    d = min(LEVELS, int(k) + 1)
    return d, k - int(k)


def _view_matrix(lat0, lon0):
    a, b = math.radians(lon0), math.radians(lat0)
    Ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return Rx @ Ry          # world -> view (view z toward the viewer)


def render_globe(fc, st, T, gx, gy, R, lat0, lon0, build=1.0):
    """additive hologram image (h,w,3) of the globe + balkanizing regions"""
    s = fc.s
    M = _view_matrix(lat0, lon0)
    cv = Canvas(fc)
    ctx = cv.ctx
    # graticule
    ctx.set_line_width(1.2)
    for la in range(-60, 90, 30):
        pts = _unit(np.full(73, la), np.linspace(-180, 180, 73)) @ M.T
        _poly_front(ctx, pts, gx, gy, R, (*HOLO, 0.22))
    for lo in range(-180, 180, 30):
        pts = _unit(np.linspace(-88, 88, 60), np.full(60, lo)) @ M.T
        _poly_front(ctx, pts, gx, gy, R, (*HOLO, 0.22))
    # limb
    set_color(ctx, HOLO, 0.55)
    ctx.set_line_width(2.4)
    circle(ctx, gx, gy, R)
    ctx.stroke()
    # land dots, brightness by facing
    V = st["hud_dots"] @ M.T
    vis = V[:, 2] > 0.02
    Vv = V[vis]
    for (x, y, z) in Vv:
        a = 0.25 + 0.75 * z
        set_color(ctx, HOLO, a * 0.9)
        circle(ctx, gx + R * x, gy - R * y, 2.1 + 1.2 * z)
        ctx.fill()
    img = cv.rgba()[..., :3]
    # ---- balkanization: per-pixel hierarchical Voronoi on the front hemisphere (full resolution: 1 px borders)
    ds = 1
    x0, y0 = int((gx - R) * s) // ds, int((gy - R) * s) // ds
    n = int(2 * R * s) // ds + 2
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    vx_ = ((xs + x0) * ds / s - gx) / R
    vy_ = -((ys + y0) * ds / s - gy) / R
    rr = vx_ ** 2 + vy_ ** 2
    disk = rr < 1.0
    vz = np.sqrt(np.clip(1 - rr, 0, 1))
    Pv = np.stack([vx_[disk], vy_[disk], vz[disk]], -1)
    Pw = Pv @ M                                  # view -> world
    edge_img = np.zeros((n, n), np.float32)
    fill_img = np.zeros((n, n), np.float32)
    flash_img = np.zeros((n, n), np.float32)
    WG = (1.0, 0.9, 0.65, 0.42, 0.2, 0.0)
    for reg in st["hud_regions"]:
        d, frac = region_depth(reg, T)
        if d < 0:
            continue
        h = reg["h"]
        labs, pseed = labels_all(Pw, h, d)
        reveal = None
        if pseed is not None and frac < 1.0:
            # new borders crack outward from each parent's seed
            cell_r = math.radians(24.0) / math.sqrt(6 * CHILDREN ** max(0, d - 2))
            ang = np.arccos(np.clip(np.einsum("nk,nk->n", Pw, pseed), -1, 1))
            rv = np.zeros((n, n), np.float32)
            rv[disk] = np.clip((ease_out(min(1.0, frac * 2.2)) * cell_r * 1.6 - ang) / (cell_r * 0.25), 0, 1)
            reveal = rv
        for lv, l in enumerate(labs):
            lab = np.full((n, n), -1, np.int64)
            lab[disk] = l
            e = ((lab[:, 1:] != lab[:, :-1])[:-1, :] | (lab[1:, :] != lab[:-1, :])[:, :-1])
            e = np.pad(e, ((0, 1), (0, 1))).astype(np.float32)
            wgt = WG[min(lv, len(WG) - 1)]
            if lv == d and reveal is not None:
                e = e * reveal
            edge_img = np.maximum(edge_img, e * wgt)
            if lv == d:
                inside = (lab >= 0).astype(np.float32)
                hsh = ((lab * 2654435761) % 1000) / 1000.0
                fill_img = np.maximum(fill_img, inside * (0.05 + 0.15 * hsh))
                flash_img = np.maximum(flash_img, inside * math.exp(-frac * 7) * 0.22)
    edge_img *= disk * (0.35 + 0.65 * vz)
    import cv2
    edge_img = edge_img * 0.85 + cv2.GaussianBlur(edge_img, (0, 0), 1.0) * 0.5
    # (full-resolution edges; resize below is identity at ds=1)
    E = cv2.resize(edge_img, (n * ds, n * ds), interpolation=cv2.INTER_LINEAR)
    Fi = cv2.resize(fill_img * disk * vz, (n * ds, n * ds), interpolation=cv2.INTER_LINEAR)
    Fl = cv2.resize(flash_img * disk, (n * ds, n * ds), interpolation=cv2.INTER_LINEAR)
    layer = np.zeros((fc.h, fc.w), np.float32)
    lay_f = np.zeros((fc.h, fc.w), np.float32)
    lay_l = np.zeros((fc.h, fc.w), np.float32)
    X0, Y0 = x0 * ds, y0 * ds
    xa, ya = max(0, X0), max(0, Y0)
    xb, yb = min(fc.w, X0 + n * ds), min(fc.h, Y0 + n * ds)
    if xb > xa and yb > ya:
        layer[ya:yb, xa:xb] = E[ya - Y0:yb - Y0, xa - X0:xb - X0]
        lay_f[ya:yb, xa:xb] = Fi[ya - Y0:yb - Y0, xa - X0:xb - X0]
        lay_l[ya:yb, xa:xb] = Fl[ya - Y0:yb - Y0, xa - X0:xb - X0]
    white = np.array((0.92, 1.0, 1.0), np.float32)
    img = img + layer[..., None] * white * 1.7 + lay_f[..., None] * np.asarray(HOLO, np.float32) * 0.8 \
        + lay_l[..., None] * white * 0.6
    # scan-build (the hologram materialises bottom -> top) and scanlines
    xx, yy = grid_xy(fc)
    if build < 1.0:
        edge = gy + R - 2 * R * build
        img = img * (yy > edge)[..., None] + np.exp(-((yy - edge) / 6.0) ** 2)[..., None] * \
            np.asarray(HOLO, np.float32) * ((np.abs(xx - gx) < R * 1.05))[..., None] * 1.5
    scan = 0.78 + 0.22 * np.sin(yy * (math.pi / 3.0) + T * 40)
    img = img * scan[..., None]
    return img


def _poly_front(ctx, pts, gx, gy, R, rgba):
    ctx.set_source_rgba(*rgba)
    drawing = False
    for x, y, z in pts:
        if z > 0:
            if not drawing:
                ctx.move_to(gx + R * x, gy - R * y)
                drawing = True
            else:
                ctx.line_to(gx + R * x, gy - R * y)
        else:
            drawing = False
    ctx.stroke()


def hud_text(ctx, T, st, gx, gy, R):
    tt = st["hud_t"]
    total = 0
    for reg in st["hud_regions"]:
        d, _ = region_depth(reg, T)
        if d > 0:
            total += 6 * CHILDREN ** (d - 1)
    depth = max((region_depth(r, T)[0] for r in st["hud_regions"]), default=0)
    x = gx + R + 50
    set_color(ctx, HOLO, 0.8)
    ctx.set_line_width(1.5)
    ctx.move_to(x - 30, gy - R * 0.75)
    ctx.line_to(x - 30, gy + R * 0.75)
    ctx.stroke()
    rows = [("SCHISM ACTUAL", 1.0), ("OPORD 7 / ALL TEAMS", 0.7), ("", 0),
            ("METHOD  RECURSIVE", 0.9 if T > tt["rec"] else 0.35),
            (f"DEPTH   {max(0, depth):d}", 0.9), (f"CELLS   {total:,}", 0.9),
            ("TARGETS MEMEPLEX x3", 0.7)]
    for i, (s_, a) in enumerate(rows):
        if s_:
            text(ctx, s_, x, gy - R * 0.62 + i * 32, 20, HOLO, alpha=a, font=FONT_MONO, bold=True)
    # the armed line, typed on then blinking
    msg = "WEAPONS OF MASS ENLIGHTENMENT: ARMED"
    t0 = tt["bal"] + 0.35
    if T > t0:
        k = int((T - t0) / 0.018)
        s_ = msg[:min(len(msg), k)]
        blink = 1.0 if (T - t0) < 0.7 or int(T * 6) % 2 == 0 else 0.35
        text(ctx, s_, gx, gy + R + 78, 26, (0.95, 1.0, 1.0), alpha=blink, font=FONT_MONO, bold=True, align="center",
             tracking=0.08)
    # lock-on brackets over the first region
    t_lock = tt["method"]
    if T > t_lock - 0.05:
        k = ease_out_back(clamp((T - t_lock + 0.05) / 0.22))
        cx_, cy_, sz = gx - R * 0.18, gy - R * 0.55, lerp(R * 1.1, R * 0.42, k)
        set_color(ctx, (0.95, 1.0, 1.0), 0.9)
        ctx.set_line_width(3)
        for sx_, sy_ in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            ctx.move_to(cx_ + sx_ * sz, cy_ + sy_ * sz * 0.8 - sy_ * 26)
            ctx.line_to(cx_ + sx_ * sz, cy_ + sy_ * sz * 0.8)
            ctx.line_to(cx_ + sx_ * sz - sx_ * 26, cy_ + sy_ * sz * 0.8)
            ctx.stroke()


def hud(fc, st, u):
    """over-the-shoulder profile: the Commander (left, lit cyan by his own hologram) and the balkanizing globe"""
    T = fc.T
    push = ease_in_out(clamp(u / 2.4)) * 0.06
    gx, gy, R = 1210 - 40 * push, 470, 300 * (1 + push)
    # background: the tunnel behind, deep out of focus
    cam = tun.Cam(x=1.2, y=1.5, z=-1.0, f=900, hor=600, cx=560)
    bgc = Canvas(fc)
    mcx, mcy = tun.draw_tunnel(bgc.ctx, cam, T)
    bg = blur(bgc.rgb(), 14 * fc.s) * 0.55
    bg += radial_img(fc, mcx, mcy, 300, GOLD, 1.6) * 0.45
    img = bg
    mc = Canvas(fc)
    motes(mc.ctx, T, 60, 17, (0, 60, 1920, 1020), lambda x, y: 0.6, rmin=3, rmax=14, alpha=0.5, speed=0.4)
    img = comp(img, mc.rgba())
    # the Commander in profile at left, arm raised with the wrist projector
    ch = st["commander_hud"]
    hC = 1180
    fx, fy = 500 + 20 * push, 1520
    mouth = fc.tl.mouth("COMMANDER", T)
    wrist = (840, 800)
    kw = dict(turn=0.85, hand_l=wrist, shape_l="open", goggles_down=False,
              tint=((0.02, 0.05, 0.08), 0.45), rim=(mix(HOLO, (1, 1, 1), 0.3), 1.3, (1.0, -0.15)), light=(1.0, -0.1))
    lay = Canvas(fc)
    a = ch.draw(lay.ctx, fx, fy, hC, "talk", T, mouth=mouth, facing=1, expr="determined", look=(0.8, -0.3), **kw)
    rgba = lay.rgba()
    # cyan key from the hologram (right) and a thin warm kiss from the tunnel light behind (left)
    back = rimlit(rgba, fc, fill=0.0, rim=GOLD_HOT, strength=0.6, width=4, light=(-1.0, -0.3), dir_k=1.0)
    rgba[..., :3] = rgba[..., :3] * np.asarray((0.85, 1.0, 1.08), np.float32) + back[..., :3]
    img = comp(img, rgba)
    # wrist emitter + projection cone
    wx, wy = a.get("hand_l", wrist)
    build = smoothstep(0.0, 0.28, u + 0.06)
    cone = Canvas(fc)
    c = cone.ctx
    poly(c, [(wx - 6, wy - 10), (gx - R * 0.95, gy + R * 0.3), (gx + R * 0.2, gy + R * 1.0), (wx + 8, wy + 4)])
    c.set_source(lin_grad(wx, wy, gx, gy + R * 0.5, [(0, (*HOLO, 0.45 * build)), (1, (*HOLO, 0.03 * build))]))
    c.fill()
    c.set_source(rad_grad(wx, wy, 0, 40, [(0, (1, 1, 1, 0.9)), (0.3, (*HOLO, 0.7)), (1, (*HOLO, 0))]))
    circle(c, wx, wy, 40)
    c.fill()
    img = img + cone.rgba()[..., :3]
    # the globe + HUD, additive hologram with a little flicker and chromatic split
    lon0 = -10 + 16 * (T - 96.0)
    g = render_globe(fc, st, T, gx, gy, R, 22, lon0, build)
    tc = Canvas(fc)
    hud_text(tc.ctx, T, st, gx, gy, R)
    g = g + tc.rgba()[..., :3] * build
    fl = 0.9 + 0.1 * math.sin(T * 53) * math.sin(T * 17)
    import cv2
    M = np.float32([[1, 0, 2.0 * fc.s], [0, 1, 0]])
    gr = g.copy()
    gr[..., 0] = cv2.warpAffine(np.ascontiguousarray(g[..., 0]), M, (fc.w, fc.h))
    img = img + gr * fl + blur(g, 10 * fc.s) * 0.5
    # cyan spill on the Commander's face/visor
    img += radial_img(fc, gx - R * 0.9, gy + 40, 420, HOLO, 2.2) * 0.12 * build
    img += streak(img, fc, 0.95, 300, 0.2, tint=(0.7, 0.95, 1.0))
    return img


SHOTS = {"hud": hud}

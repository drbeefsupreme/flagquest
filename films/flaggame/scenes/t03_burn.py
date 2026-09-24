"""t03 the Burn at Alchemy IX, engraved (owner T03Genesis).

Grey world painted for vx.ink.engrave: a per-pixel mud plain (numba, same ground plane Y=0 and camera as
the op-art so one can dissolve into the other), a Dore sky (precomputed billow texture, swept by the
storm), the wall of camps on the horizon, and the crowd of muddy hippies (vx.chars 'hippie', ink render)
placed in 3D and projected. Stink lines rise from every head and are swept into the vortex.
"""
import math

import cairo
import cv2
import numpy as np
from numba import njit

from vx import *
from vx.chars import draw_crowd
from scenes.t03_cam import Cam
from scenes import t03_opart as OP

# ------------------------------------------------------------------ the crowd-shot stage
# the camera comes down out of the genesis orbit to eye level, looking along FWD (horizontal), RIGHT to the right
YAW_C = 2.12
FWD = np.array([-math.sin(YAW_C), 0.0, -math.cos(YAW_C)])
RIGHT = np.array([math.cos(YAW_C), 0.0, -math.sin(YAW_C)])
UP = np.array([0.0, 1.0, 0.0])


def stage(s, d, y=0.0):
    """stage coords: s lateral (right +), d depth beyond the flag ring (away from camera +)"""
    return s * RIGHT + d * FWD + y * UP


# ------------------------------------------------------------------ camera through the whole flashback

def world_cam(T):
    """camera for the op-art genesis and the engraved Burn (continuous through the dissolve)"""
    tau = OP.tau_of(T)
    if T <= 54.6:
        return OP.genesis_cam(tau)
    g = OP.genesis_cam(OP.tau_of(54.6))
    ge = (g.elev, g.dist, g.yaw, float(g.target[1]))
    ce = _crowd_params(T)
    u = smootherstep(54.6, 56.5, T)
    # descend fast, level off gently (a crane coming down to eye level)
    ue = 1.0 - (1.0 - smoothstep(54.6, 56.5, T)) ** 2
    elev = ge[0] + (ce[0] - ge[0]) * ue
    dist = ge[1] + (ce[1] - ge[1]) * u
    yaw = ge[2] + (ce[2] - ge[2]) * u
    ty = ge[3] + (ce[3] - ge[3]) * u
    tilt = ce[4] * smoothstep(56.4, 56.6, T)
    cx = OP.CX_GEN + (960.0 - OP.CX_GEN) * u
    return _cam(elev, dist, yaw, ty, tilt, cx)


def _crowd_params(T):
    """crowd camera (a little above the heads, Dore's vantage): slow push in; tilts up with the stench"""
    p = smoothstep(55.5, 59.4, T)
    elev = math.radians(5.8 - 0.6 * p)
    dist = 8.6 - 0.45 * p
    yaw = YAW_C + 0.05 * (p - 0.5)
    ty = 0.95
    tilt = math.radians(10.0) * smootherstep(58.2, 59.34, T) ** 1.5
    return (elev, dist, yaw, ty, tilt)


def _cam(elev, dist, yaw, ty, tilt=0.0, cx=960.0):
    c = Cam(yaw=yaw, elev=elev, dist=dist, f=OP.F_GEN, target=(0.0, ty, 0.0), cx=cx)
    if tilt:
        # pitch the view up around the camera position (look at the sky) without moving the camera
        C = c.C
        fwd = c.fwd * math.cos(tilt) + np.array([0.0, 1.0, 0.0]) * math.sin(tilt) * 1.0
        fwd /= np.linalg.norm(fwd)
        c = Cam.look(C, C + fwd * dist, f=OP.F_GEN, cx=cx)
    return c


# ------------------------------------------------------------------ mud (numba, per pixel)
@njit(cache=True, fastmath=True, inline="always")
def _h2(ix, iy, seed):
    n = ix * 374761393 + iy * 668265263 + seed * 144269504
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return (n & 0xFFFFFF) / 16777216.0


@njit(cache=True, fastmath=True)
def _vn(x, y, seed):
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    ix = int(ix)
    iy = int(iy)
    a = _h2(ix, iy, seed)
    b = _h2(ix + 1, iy, seed)
    c = _h2(ix, iy + 1, seed)
    d = _h2(ix + 1, iy + 1, seed)
    return (a + (b - a) * ux) * (1 - uy) + (c + (d - c) * ux) * uy


@njit(cache=True, fastmath=True)
def _fbm(x, y, seed, octv, foot):
    s = 0.0
    a = 0.5
    f = 1.0
    nrm = 0.0
    for o in range(octv):
        # drop octaves finer than the pixel footprint (no shimmering)
        k = 1.0 - min(1.0, max(0.0, (foot * f - 0.25) / 0.5))
        s += a * k * (_vn(x * f, y * f, seed + o * 17) - 0.5)
        nrm += a
        a *= 0.5
        f *= 2.03
    return s / nrm


@njit(cache=True, fastmath=True)
def _mud(tone, mask, cp, s, sp, fwdx, fwdz):
    h, w = tone.shape
    Cx, Cy, Cz = cp[0], cp[1], cp[2]
    rx, ry, rz = cp[3], cp[4], cp[5]
    dx_, dy_, dz_ = cp[6], cp[7], cp[8]
    fx, fy, fz = cp[9], cp[10], cp[11]
    f = cp[12]
    ccx = cp[13]
    ccy = cp[14]
    wet = sp[0]
    T = sp[1]
    rain = sp[2]
    fog0 = sp[3]
    fog1 = sp[4]
    for i in range(h):
        for j in range(w):
            sx = (j + 0.5) / s
            sy = (i + 0.5) / s
            kx = (sx - ccx) / f
            ky = (sy - ccy) / f
            Dx = fx + rx * kx + dx_ * ky
            Dy = fy + ry * kx + dy_ * ky
            Dz = fz + rz * kx + dz_ * ky
            if Dy >= -1e-5:
                mask[i, j] = 0.0
                continue
            t = -Cy / Dy
            X = Cx + t * Dx
            Z = Cz + t * Dz
            dist = t * math.sqrt(Dx * Dx + Dy * Dy + Dz * Dz)
            foot = dist / (f * s) / max(0.05, -Dy / math.sqrt(Dx * Dx + Dy * Dy + Dz * Dz))
            # depth / lateral coordinates on the stage (ruts run across the view, the crowd's path)
            q = X * fwdx + Z * fwdz
            l = -X * fwdz + Z * fwdx
            n1 = _fbm(X * 0.22, Z * 0.22, 3, 3, foot * 0.22)
            n2 = _fbm(X * 1.3, Z * 1.3, 11, 4, foot * 1.3)
            n3 = _fbm(X * 4.0 + 7.0, Z * 4.0, 23, 3, foot * 4.0)
            v = 0.76 + 0.30 * n1 + 0.22 * n2 + 0.12 * n3
            # trampled ruts along the crowd's path: dark furrows with lit crests
            rq = q * 1.7 + 1.1 * n1 + 0.6 * n2
            fr = rq - math.floor(rq)
            rut = math.exp(-((fr - 0.5) * 7.0) ** 2)
            crest = math.exp(-((fr - 0.72) * 9.0) ** 2)
            kd = 1.0 - min(1.0, foot * 1.2)
            v += (-0.26 * rut + 0.10 * crest) * kd
            # footprints: small dark ovals in the churn
            fu = l * 2.6 + 0.3 * n2
            fv = q * 4.4
            cu = math.floor(fu)
            cv = math.floor(fv)
            hh = _h2(int(cu), int(cv), 77)
            if hh < 0.35 and kd > 0.05:
                ou = fu - cu - 0.5 - (hh - 0.17) * 0.8
                ov = fv - cv - 0.5
                e = ou * ou * 4.0 + ov * ov * 9.0
                v -= 0.22 * kd * max(0.0, 1.0 - e * 4.0)
            # puddles (grow with the rain) mirror the pale sky; dark wet rim
            pz = n1 + 0.55 * n2 + 0.20 * wet
            thr = 0.13 - 0.10 * wet
            edge = foot * 1.5 + 0.02
            inner = min(1.0, max(0.0, (pz - thr) / edge + 0.5))
            rim = math.exp(-((pz - thr) / (0.035 + edge)) ** 2)
            pv = 0.95 + 0.06 * n3 * kd
            if rain > 0.0:
                # rain rings on the puddles
                rr = _h2(int(math.floor(X * 3.0)), int(math.floor(Z * 3.0)), int(T * 6.0))
                if rr < 0.25 * rain:
                    px = X * 3.0 - math.floor(X * 3.0) - 0.5
                    pz2 = Z * 3.0 - math.floor(Z * 3.0) - 0.5
                    ring = math.sqrt(px * px + pz2 * pz2)
                    ph = (T * 6.0 - math.floor(T * 6.0))
                    pv -= 0.35 * kd * math.exp(-((ring - ph * 0.45) * 30.0) ** 2) * (1.0 - ph)
            v = v * (1.0 - inner) + pv * inner - 0.18 * rim * (1.0 - inner) * kd
            v -= 0.12 * wet * (1.0 - inner)
            # aerial perspective: Dore's distance goes pale
            fg = min(1.0, max(0.0, (dist - fog0) / (fog1 - fog0)))
            fg = fg * fg * (3 - 2 * fg)
            v = v * (1.0 - fg) + 0.92 * fg
            tone[i, j] = min(0.98, max(0.04, v))
            mask[i, j] = 1.0


def mud_tone(fc, cam, T, wet=0.0, rain=0.0, fog0=9.0, fog1=48.0):
    tone = np.ones((fc.h, fc.w), np.float32)
    mask = np.zeros((fc.h, fc.w), np.float32)
    _mud(tone, mask, cam.params(), float(fc.s), np.array([wet, T, rain, fog0, fog1], np.float64),
         float(FWD[0]), float(FWD[2]))
    return tone, mask


# ------------------------------------------------------------------ Dore sky
def make_sky(S, W_=4096, H_=1536, version="v5"):
    """Dore cumulus panorama (azimuth x altitude): banks of lit billows with flat, shadowed bases,
    squashed toward the horizon; bottom row = horizon. Seamless in x."""
    p = S.cache / f"t03_sky_{version}.npy"
    if p.exists():
        return np.load(p)
    import cairo as _c
    surf = _c.ImageSurface(_c.FORMAT_ARGB32, W_, H_)
    ctx = _c.Context(surf)
    g = _c.LinearGradient(0, 0, 0, H_)
    g.add_color_stop_rgb(0, 0.80, 0.80, 0.80)
    g.add_color_stop_rgb(1, 0.95, 0.95, 0.95)
    ctx.set_source(g)
    ctx.paint()
    rng = np.random.default_rng(1234)
    banks = []
    for i in range(190):
        a = rng.random() ** 0.8                           # altitude fraction (0 horizon .. 1 top)
        banks.append((rng.uniform(0, W_), a))
    banks.sort(key=lambda b: b[1])                        # far (low) banks first; overhead banks occlude them
    for bx, a in banks:
        base_y = H_ * (1.0 - a) - 10
        sq = 0.22 + 0.78 * a                             # perspective squash toward the horizon
        wb = rng.uniform(260, 800) * (0.45 + 0.55 * a)
        npf = int(rng.integers(12, 28))
        dark = rng.uniform(0.0, 0.35)
        puffs = []
        for k in range(npf):
            u = rng.uniform(-1, 1)
            r = wb * 0.22 * (1.0 - 0.6 * abs(u)) * rng.uniform(0.6, 1.2)
            puffs.append((bx + u * wb * 0.5, base_y - r * sq * rng.uniform(0.2, 1.1), r))
        puffs.sort(key=lambda q: q[1])
        for (px, py, r) in puffs:
            for dx in (-W_, 0, W_):
                x = px + dx
                if x + r < 0 or x - r > W_:
                    continue
                ctx.save()
                ctx.translate(x, py)
                ctx.scale(1.0, sq)
                gg = _c.RadialGradient(-r * 0.35, -r * 0.45, r * 0.05, 0, 0, r)
                hi = 0.97 - 0.1 * dark
                lo = 0.30 - 0.1 * dark
                gg.add_color_stop_rgb(0, hi, hi, hi)
                gg.add_color_stop_rgb(0.55, 0.72 - 0.2 * dark, 0.72 - 0.2 * dark, 0.72 - 0.2 * dark)
                gg.add_color_stop_rgb(1, lo, lo, lo)
                ctx.set_source(gg)
                ctx.arc(0, 0, r, 0, 2 * math.pi)
                ctx.fill()
                ctx.restore()
    surf.flush()
    a = np.ndarray((H_, W_, 4), np.uint8, surf.get_data())
    sky = (a[..., 1].astype(np.float32) / 255.0)
    sky = cv2.GaussianBlur(sky, (0, 0), 1.2)
    np.save(p, sky)
    return sky


def sky_tone(fc, cam, sky, T, dark=0.0, swirl=0.0, swirl_c=(960.0, 120.0)):
    """screen-space sky tone from the panorama; dark 0..1 = the storm coming; swirl = rotate the clouds
    around swirl_c (the vortex forming overhead)"""
    h, w = fc.h, fc.w
    Hs, Ws = sky.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs = (xs + 0.5) / fc.s
    ys = (ys + 0.5) / fc.s
    if swirl:
        dx = xs - swirl_c[0]
        dy = (ys - swirl_c[1]) * 1.8
        r = np.sqrt(dx * dx + dy * dy) + 1.0
        a = swirl * np.exp(-r / 700.0) * 2.2
        ca, sa = np.cos(a), np.sin(a)
        xs = swirl_c[0] + dx * ca - dy * sa
        ys = swirl_c[1] + (dx * sa + dy * ca) / 1.8
    # direction per pixel -> azimuth / altitude
    kx = (xs - cam.cx) / cam.f
    ky = (ys - cam.cy) / cam.f
    Dx = cam.fwd[0] + cam.right[0] * kx + cam.down[0] * ky
    Dy = cam.fwd[1] + cam.right[1] * kx + cam.down[1] * ky
    Dz = cam.fwd[2] + cam.right[2] * kx + cam.down[2] * ky
    az = np.arctan2(Dx, Dz)
    alt = np.arctan2(Dy, np.sqrt(Dx * Dx + Dz * Dz))
    mu = ((az / (2 * math.pi) + 0.5 + T * 0.004) % 1.0) * Ws
    mv = np.clip(Hs - 1 - alt / (math.pi * 0.5) * Hs * 1.6, 0, Hs - 1)
    tone = cv2.remap(sky, mu.astype(np.float32), mv.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    # pale near the horizon, heavy overhead; darker as the storm gathers
    hz = np.clip(alt / 0.35, 0, 1)
    tone = tone * hz + 0.95 * (1 - hz)
    tone = tone * (1.0 - 0.45 * dark * hz)
    return np.clip(tone, 0.05, 0.97).astype(np.float32)


# ------------------------------------------------------------------ the wall of camps
def make_camps(seed=5):
    rng = np.random.default_rng(seed)
    out = []
    s = -46.0
    while s < 46.0:
        kind = rng.choice(["dome", "geo", "shade", "rv", "tent", "yurt", "tower", "shade"],
                          p=[0.16, 0.12, 0.2, 0.14, 0.14, 0.08, 0.06, 0.10])
        size = {"dome": 1.6, "geo": 2.6, "shade": 3.2, "rv": 3.6, "tent": 1.8, "yurt": 2.4, "tower": 1.0}[kind]
        size *= rng.uniform(0.8, 1.25)
        d = rng.uniform(19.0, 27.0)
        out.append(dict(kind=str(kind), s=s + size * 0.5, d=d, size=size, tone=float(rng.uniform(0.35, 0.7)),
                        seed=int(rng.integers(1 << 20))))
        s += size * rng.uniform(0.55, 1.0)
    out.sort(key=lambda c: -c["d"])
    return out


def draw_camps(ctx, cam, camps, T, wind=0.0):
    for c in camps:
        P = stage(c["s"], c["d"])
        sx, sy, z = cam.project(P)
        if z <= 0.5:
            continue
        k = cam.f / z
        if sx < -400 or sx > 2320:
            continue
        fog = min(1.0, max(0.0, (z - 12.0) / 30.0))
        tone = c["tone"] * (1 - fog * 0.55) + 0.86 * fog * 0.55
        ctx.save()
        ctx.translate(sx, sy)
        ctx.scale(k, k)
        _camp(ctx, c, tone, T, wind)
        ctx.restore()


def _g(ctx, v, a=1.0):
    ctx.set_source_rgba(v, v, v, a)


def _camp(ctx, c, tone, T, wind):
    kind, sz = c["kind"], c["size"]
    rng = np.random.default_rng(c["seed"])
    lw = 0.03
    ctx.set_line_width(lw)
    if kind == "dome":
        ctx.new_path()
        ctx.save()
        ctx.scale(sz * 0.5, sz * 0.36)
        ctx.arc(0, 0, 1, math.pi, 2 * math.pi)
        ctx.restore()
        ctx.close_path()
        ctx.set_source(lin_grad(-sz * 0.5, -sz * 0.4, sz * 0.5, 0, [(0, (tone + 0.2,) * 3), (1, (tone - 0.15,) * 3)]))
        ctx.fill_preserve()
        _g(ctx, 0.05)
        ctx.stroke()
        for kk in (-0.25, 0.25):
            ctx.move_to(-sz * 0.5, 0)
            ctx.curve_to(-sz * 0.3, -sz * 0.4, sz * kk, -sz * 0.4, sz * 0.5, 0)
        ctx.stroke()
        _g(ctx, 0.12)
        ctx.rectangle(-sz * 0.08, -sz * 0.18, sz * 0.16, sz * 0.18)
        ctx.fill()
    elif kind == "geo":
        R = sz * 0.5
        pts = []
        for i in range(9):
            a = math.pi + math.pi * i / 8
            pts.append((R * math.cos(a), R * 0.9 * math.sin(a)))
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.set_source(lin_grad(-R, -R, R, 0, [(0, (tone + 0.25,) * 3), (1, (tone - 0.2,) * 3)]))
        ctx.fill_preserve()
        _g(ctx, 0.05)
        ctx.stroke()
        # geodesic struts
        for ring in (0.35, 0.7):
            for i in range(9):
                a = math.pi + math.pi * i / 8
                ctx.move_to(0, -R * 0.9)
                ctx.line_to(R * math.cos(a), R * 0.9 * math.sin(a))
            ctx.move_to(-R, -R * ring * 0.6)
            ctx.line_to(R, -R * ring * 0.6)
        ctx.set_line_width(lw * 0.6)
        ctx.stroke()
    elif kind == "shade":
        w_, h_ = sz, sz * 0.62
        sag = 0.12 * sz * (1 + 0.3 * math.sin(T * 2.0 + c["seed"]) * wind)
        _g(ctx, 0.1)
        for px in (-w_ * 0.5, -w_ * 0.1, w_ * 0.5):
            ctx.rectangle(px - 0.03, -h_, 0.06, h_)
        ctx.fill()
        ctx.move_to(-w_ * 0.55, -h_)
        ctx.curve_to(-w_ * 0.2, -h_ + sag, w_ * 0.2, -h_ + sag, w_ * 0.55, -h_)
        ctx.line_to(w_ * 0.5, -h_ + 0.18 * sz)
        ctx.curve_to(w_ * 0.2, -h_ + 0.18 * sz + sag, -w_ * 0.2, -h_ + 0.18 * sz + sag, -w_ * 0.5, -h_ + 0.18 * sz)
        ctx.close_path()
        _g(ctx, tone - 0.1)
        ctx.fill_preserve()
        _g(ctx, 0.05)
        ctx.stroke()
        # shadow under the shade
        ctx.rectangle(-w_ * 0.5, -h_ * 0.55, w_, h_ * 0.55)
        _g(ctx, 0.25, 0.5)
        ctx.fill()
    elif kind == "rv":
        w_, h_ = sz, sz * 0.36
        rrect(ctx, -w_ * 0.5, -h_ - 0.12, w_, h_, 0.12)
        ctx.set_source(lin_grad(0, -h_, 0, 0, [(0, (tone + 0.22,) * 3), (1, (tone - 0.1,) * 3)]))
        ctx.fill_preserve()
        _g(ctx, 0.05)
        ctx.stroke()
        _g(ctx, 0.15)
        for i in range(3):
            ctx.rectangle(-w_ * 0.4 + i * w_ * 0.22, -h_ * 0.85, w_ * 0.14, h_ * 0.3)
        ctx.fill()
        ctx.rectangle(-w_ * 0.5, -h_ * 0.45, w_, 0.05)
        ctx.fill()
        for px in (-w_ * 0.3, w_ * 0.3):
            circle(ctx, px, -0.1, 0.13)
        _g(ctx, 0.05)
        ctx.fill()
    elif kind == "tent":
        w_, h_ = sz, sz * 0.55
        ctx.move_to(-w_ * 0.5, 0)
        ctx.line_to(-w_ * 0.1, -h_)
        ctx.line_to(w_ * 0.2, -h_ * 0.95)
        ctx.line_to(w_ * 0.5, 0)
        ctx.close_path()
        ctx.set_source(lin_grad(-w_ * 0.5, 0, w_ * 0.5, 0, [(0, (tone + 0.2,) * 3), (1, (tone - 0.2,) * 3)]))
        ctx.fill_preserve()
        _g(ctx, 0.05)
        ctx.stroke()
        ctx.move_to(-w_ * 0.1, -h_)
        ctx.line_to(0.0, 0)
        ctx.stroke()
    elif kind == "yurt":
        w_, h_ = sz, sz * 0.32
        ctx.rectangle(-w_ * 0.5, -h_, w_, h_)
        ctx.set_source(lin_grad(-w_ * 0.5, 0, w_ * 0.5, 0, [(0, (tone + 0.15,) * 3), (1, (tone - 0.2,) * 3)]))
        ctx.fill_preserve()
        _g(ctx, 0.05)
        ctx.stroke()
        ctx.move_to(-w_ * 0.55, -h_)
        ctx.line_to(0, -h_ - sz * 0.25)
        ctx.line_to(w_ * 0.55, -h_)
        ctx.close_path()
        _g(ctx, tone - 0.18)
        ctx.fill_preserve()
        _g(ctx, 0.05)
        ctx.stroke()
    elif kind == "tower":
        h_ = sz * 5.0
        _g(ctx, 0.08)
        ctx.set_line_width(0.05)
        ctx.move_to(-0.35, 0)
        ctx.line_to(-0.08, -h_)
        ctx.move_to(0.35, 0)
        ctx.line_to(0.08, -h_)
        for i in range(1, 6):
            y0 = -h_ * i / 6
            ctx.move_to(-0.35 + 0.27 * i / 6, y0)
            ctx.line_to(0.35 - 0.27 * i / 6, y0 - h_ / 6)
        ctx.stroke()
        # a lamp / sound cannon on top
        ctx.rectangle(-0.3, -h_ - 0.3, 0.6, 0.3)
        ctx.fill()


# ------------------------------------------------------------------ the crowd
HIPPIE_H = 1.42        # world units (pole = 1.9 = 96")


def make_crowd(seed=11, n=470):
    rng = np.random.default_rng(seed)
    ppl = []
    tries = 0
    while len(ppl) < n and tries < 20000:
        tries += 1
        d = float(rng.uniform(-5.0, 20.0))
        s0 = float(rng.uniform(-30.0, 30.0))
        # keep the flag ring clear (the crowd flows around it)
        if abs(d) < 1.5:
            continue
        v = float(rng.uniform(0.3, 0.7)) * (1 if rng.random() < 0.82 else -1)
        ppl.append(dict(s0=s0, d=d, v=v, seed=int(rng.integers(1 << 20)), h=HIPPIE_H * float(rng.uniform(0.86, 1.1)),
                        phase=float(rng.uniform(0, 10)), idle=bool(rng.random() < 0.14),
                        look_up=float(rng.uniform(0.0, 0.5))))
    # the heroes of the crowd shot: big, oblivious, cheerful, framing the ring (medium shot, Chick-style)
    for s0, d, face, sd, hh, walk in ((-1.02, -6.25, 1, 101, 1.46, False), (1.18, -5.95, -1, 202, 1.40, False),
                                      (-2.6, -3.6, 1, 303, 1.44, True), (2.9, -3.0, -1, 404, 1.36, True)):
        ppl.append(dict(s0=s0, d=d, v=0.18 * face if walk else 0.0, seed=sd, h=hh, phase=sd * 0.01,
                        idle=not walk, look_up=0.0, hero=True, facing=face, expr="cheerful"))
    return ppl


def crowd_people(cam, crowd, T, t_local, wet=0.0, mud=0.7, heroes=False):
    """screen-space draw_crowd dicts, back-to-front, each with world depth + head for the stink lines"""
    out = []
    for p in crowd:
        hero = p.get("hero", False)
        if hero and not heroes:
            continue
        s = p["s0"] + (0.0 if p["idle"] else p["v"] * (T - 56.0))
        if not hero:
            # wrap people across the stage so the flow never empties
            s = (s + 30.0) % 60.0 - 30.0
        P = stage(s, p["d"])
        sx, sy, z = cam.project(P)
        if z < 0.8:
            continue
        k = cam.f / z
        hpx = p["h"] * k
        if sx < -hpx or sx > 1920 + hpx or hpx < 6:
            continue
        pose = "stand" if p["idle"] else "walk"
        spd = (abs(p["v"]) / 0.43) ** 0.91 if p["v"] else 1.0
        facing = p.get("facing", 1 if p["v"] >= 0 else -1)
        out.append(dict(x=float(sx), y=float(sy), h=float(hpx), kind="hippie", seed=p["seed"], pose=pose,
                        facing=facing, t_off=p["phase"], speed=spd, expr=p.get("expr", "neutral"),
                        render="ink", mud=mud, wet=wet > 0.5, depth=float(z), look=(0.0, -p["look_up"])))
    out.sort(key=lambda q: -q["depth"])
    return out


def draw_people(ctx, people, t):
    """draw_crowd wrapper; returns anchors aligned with people"""
    return draw_crowd(ctx, people, t)

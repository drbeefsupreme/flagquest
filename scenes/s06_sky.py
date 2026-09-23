"""s06 helper: the violet-black storm (ray-cast cloud deck + swirling fog sea) and procedural lightning."""
import math

import cairo
import cv2
import numpy as np

from vx import col, hash01

SEA_Y = -470.0
CLOUD_Y = 8200.0


def tileable_noise(n=256, beta=2.3, seed=0):
    """periodic fbm-like noise via a power-law spectrum -> [0,1]"""
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(n)[None, :]
    fy = np.fft.fftfreq(n)[:, None]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    amp = f ** (-beta / 2)
    amp[0, 0] = 0
    ph = rng.uniform(0, 2 * np.pi, (n, n))
    z = np.real(np.fft.ifft2(amp * np.exp(1j * ph)))
    z = (z - z.min()) / (z.max() - z.min())
    return z.astype(np.float32)


def make_textures(seed=5):
    return dict(cloud=tileable_noise(256, 2.6, seed), detail=tileable_noise(256, 1.9, seed + 1),
                sea=tileable_noise(256, 2.4, seed + 2))


def _sample(tex, u, v):
    """bilinear wrap sampling; u, v in texels (float32 arrays)"""
    return cv2.remap(tex, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def sky(fc, cam, t, tex, vortex_xz, base_xz, flash=0.0, flash_dir=None, eg_glow=0.0, eg_dir=None, res=0.5):
    """render the storm sky + fog sea through `cam` -> float32 (fc.h, fc.w, 3)"""
    hh, ww = max(8, int(fc.h * res)), max(8, int(fc.w * res))
    ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float32)
    xs = (xs + 0.5) / ww * 1920.0
    ys = (ys + 0.5) / hh * 1080.0
    D = cam.ray_dirs(xs, ys).astype(np.float32)
    dy = D[..., 1]
    out = np.zeros((hh, ww, 3), np.float32)
    up = dy > 0.0
    px, py, pz = cam.pos

    # ---------------- upper storm deck
    horizon = np.array([0.17, 0.12, 0.25], np.float32)
    zenith = np.array([0.025, 0.022, 0.055], np.float32)
    e = np.clip(dy, 0, 1)[..., None]
    grad = horizon * (1 - e) ** 3 + zenith * (1 - (1 - e) ** 3)
    tt = (CLOUD_Y - py) / np.maximum(dy, 0.02)
    X = px + D[..., 0] * tt - vortex_xz[0]
    Z = pz + D[..., 2] * tt - vortex_xz[1]
    rho = np.sqrt(X * X + Z * Z) + 1.0
    phi = np.arctan2(Z, X)
    rot = t * 0.33 * (5000.0 / (rho + 2500.0)) + 1.1 * np.log(rho / 3000.0 + 1.0)
    ang = phi + rot
    sc = 1.0 / 70.0
    U = rho * np.cos(ang) * sc
    V = rho * np.sin(ang) * sc
    rn = np.sqrt(U * U + V * V) + 1e-3
    oU, oV = -U / rn * 2.5, -V / rn * 2.5          # one step toward the eye of the vortex
    c1 = _sample(tex["cloud"], U, V)
    c1o = _sample(tex["cloud"], U + oU, V + oV)
    c2 = _sample(tex["detail"], U * 1.9 + t * 2.0, V * 1.9)
    d = c1 * 0.9 + c2 * 0.16
    do = c1o * 0.9 + c2 * 0.16
    dens = np.clip((d - 0.42) / 0.3, 0, 1)
    dens = dens * dens * (3 - 2 * dens)
    lit = np.clip((d - do) * 7.0, 0, 1)[..., None]
    far = np.clip(tt / 110000.0, 0, 1)
    dens = dens * (1 - far * 0.8)
    cloud_dark = np.array([0.045, 0.04, 0.085], np.float32)
    cloud_lit = np.array([0.40, 0.34, 0.60], np.float32)
    vort = np.exp(-rho / 11000.0)[..., None]
    upper = grad * (1 - dens[..., None] * 0.92) + dens[..., None] * (cloud_dark + cloud_lit * lit * (0.25 + 0.9 * vort))
    # the eye of the vortex glows cold (Egregore's screens light the clouds)
    upper += np.array([0.20, 0.30, 0.45], np.float32) * (np.exp(-rho / 3800.0) * eg_glow)[..., None] * \
        (0.4 + dens[..., None])

    # ---------------- lower fog sea (s05b's swirl lives here)
    tl = (SEA_Y - py) / np.minimum(dy, -0.02)
    X2 = px + D[..., 0] * tl - base_xz[0]
    Z2 = pz + D[..., 2] * tl - base_xz[1]
    r2 = np.sqrt(X2 * X2 + Z2 * Z2) + 1.0
    p2 = np.arctan2(Z2, X2)
    w = 0.5 * np.sqrt(960.0 / np.maximum(r2, 200.0))
    a2 = p2 - w * t * 0.6 - 1.4 * np.log(r2 / 900.0 + 1.0)
    sc2 = 1.0 / 26.0
    s1 = _sample(tex["sea"], r2 * np.cos(a2) * sc2, r2 * np.sin(a2) * sc2)
    s2 = _sample(tex["detail"], r2 * np.cos(a2) * sc2 * 2.3, r2 * np.sin(a2) * sc2 * 2.3)
    wisp = np.clip((s1 * 0.8 + s2 * 0.4 - 0.52) * 2.6, 0, 1)[..., None]
    dusk1 = np.array(col("dusk1"), np.float32)
    night = np.array(col("night"), np.float32)
    dusk2 = np.array(col("dusk2"), np.float32)
    rr = np.clip(r2 / 3600.0, 0, 1)[..., None]
    sea = dusk1 * (1 - rr) + night * rr
    sea = sea * (1 - wisp * 0.75) + dusk2 * wisp * 0.9 * (1 - rr * 0.6)
    core = np.exp(-(r2 / 1100.0) ** 2)[..., None]
    sea += np.array([0.42, 0.38, 0.60], np.float32) * core * 0.55
    dist = np.clip(tl / 45000.0, 0, 1)[..., None]
    sea = sea * (1 - dist) + horizon * 0.8 * dist

    out = np.where(up[..., None], upper, sea)
    # horizon haze band
    hz = np.exp(-(dy / 0.05) ** 2)[..., None]
    out = out * (1 - hz * 0.6) + np.array([0.16, 0.12, 0.24], np.float32) * hz * 0.6
    if flash > 0 and flash_dir is not None:
        fd = np.asarray(flash_dir, np.float32)
        fd = fd / np.linalg.norm(fd)
        cosang = D @ fd
        fl = np.clip((cosang - 0.2) / 0.8, 0, 1) ** 3
        out += (np.array([0.55, 0.55, 0.8], np.float32) * (fl[..., None] * (0.6 + dens[..., None] * 0.8 * up[..., None])
                                                          + 0.06)) * flash
    out = np.clip(out, 0, 2)
    if res != 1.0:
        out = cv2.resize(out, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)
    return out


# ------------------------------------------------------------------------------------------ lightning


def bolt(seed, x0, y0, x1, y1, rough=0.22, depth=7, branches=True):
    """midpoint-displacement lightning -> list of (points array, width_factor)"""
    rng = np.random.default_rng(seed)
    paths = []

    def build(a, b, d, rough):
        pts = [a, b]
        for _ in range(d):
            new = [pts[0]]
            for p, q in zip(pts[:-1], pts[1:]):
                mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
                L = math.hypot(q[0] - p[0], q[1] - p[1])
                nx, ny = -(q[1] - p[1]) / (L + 1e-9), (q[0] - p[0]) / (L + 1e-9)
                o = rng.normal(0, rough) * L
                new += [(mx + nx * o, my + ny * o), q]
            pts = new
        return np.array(pts)

    main = build((x0, y0), (x1, y1), depth, rough)
    paths.append((main, 1.0))
    if branches:
        for _ in range(int(rng.integers(3, 7))):
            k = int(rng.integers(len(main) // 6, len(main) - 4))
            p = main[k]
            L = math.hypot(x1 - x0, y1 - y0) * rng.uniform(0.15, 0.4)
            a = math.atan2(y1 - y0, x1 - x0) + rng.choice([-1, 1]) * rng.uniform(0.3, 0.9)
            q = (p[0] + math.cos(a) * L, p[1] + math.sin(a) * L)
            paths.append((build(tuple(p), q, depth - 2, rough * 1.1), 0.45))
    return paths


def flash_env(t, strikes):
    """total flash intensity at time t from strikes [(t0, strength, ...)] with 3 return strokes"""
    f = 0.0
    for s in strikes:
        t0, k = s[0], s[1]
        for j, (dt, a) in enumerate(((0.0, 1.0), (0.07, 0.7), (0.16, 0.85), (0.27, 0.35))):
            d = t - (t0 + dt)
            if d >= 0:
                f += k * a * math.exp(-d / 0.045)
    return f


def bolt_alpha(t, t0):
    d = t - t0
    if d < 0 or d > 0.45:
        return 0.0
    a = 0.0
    for dt, amp in ((0.0, 1.0), (0.07, 0.8), (0.16, 0.9), (0.27, 0.4)):
        e = d - dt
        if e >= 0:
            a = max(a, amp * math.exp(-e / 0.05))
    return a


def draw_bolt(ctx, gctx, paths, alpha, width=3.0, glow_col=(0.62, 0.55, 1.0)):
    if alpha <= 0.01:
        return
    for pts, wf in paths:
        n = len(pts)
        for cc, lw, a, colr in ((gctx, width * 7 * wf, 0.55, glow_col), (ctx, width * 2.2 * wf, 0.45, glow_col),
                                (ctx, width * wf, 1.0, (0.97, 0.97, 1.0))):
            if cc is None:
                continue
            cc.set_line_width(max(0.6, lw))
            cc.set_source_rgba(*colr, min(1, a * alpha))
            cc.move_to(*pts[0])
            for p in pts[1:]:
                cc.line_to(*p)
            cc.stroke()


def cloud_layers(fc, cam, t, tex, planes=((1450.0, 260.0), (2650.0, 300.0)), dmax=2400.0, flash=0.0, res=0.25,
                 k=1.0):
    """thin cloud decks the crane flies through: premultiplied RGBA overlay (h, w, 4) in front of the tower.
    planes = ((height, thickness), ...); only cloud nearer than dmax is drawn (it sits in front of the tower)."""
    hh, ww = max(8, int(fc.h * res)), max(8, int(fc.w * res))
    ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float32)
    xs = (xs + 0.5) / ww * 1920.0
    ys = (ys + 0.5) / hh * 1080.0
    D = cam.ray_dirs(xs, ys).astype(np.float32)
    dy = D[..., 1]
    px, py, pz = cam.pos
    a_tot = np.zeros((hh, ww), np.float32)
    for (Yc, th) in planes:
        # distance along the ray inside the slab [Yc - th/2, Yc + th/2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (Yc - th / 2 - py) / dy
            t1 = (Yc + th / 2 - py) / dy
        tn = np.minimum(t0, t1)
        tf = np.maximum(t0, t1)
        inside = abs(py - Yc) < th / 2
        tn = np.where(inside, 0.0, tn)
        tn = np.nan_to_num(tn, nan=1e9, posinf=1e9, neginf=-1e9)
        tf = np.nan_to_num(tf, nan=1e9, posinf=1e9, neginf=-1e9)
        tf = np.minimum(tf, dmax)
        seglen = np.clip(tf - np.maximum(tn, 0.0), 0, None)
        tm = np.clip((np.maximum(tn, 0) + tf) * 0.5, 0, dmax)
        X = px + D[..., 0] * tm
        Z = pz + D[..., 2] * tm
        n1 = _sample(tex["cloud"], X / 55.0 + t * 1.5, Z / 55.0)
        n2 = _sample(tex["detail"], X / 18.0, Z / 18.0 + t * 2.0)
        dens = np.clip((n1 * 0.8 + n2 * 0.35 - 0.56) * 3.2, 0, 1)
        a = 1 - np.exp(-seglen / 900.0 * dens * 1.6)
        a_tot = 1 - (1 - a_tot) * (1 - a)
    a_tot = cv2.GaussianBlur(a_tot, (0, 0), 1.2)
    a_tot = cv2.resize(a_tot, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)[..., None] * 0.6 * k
    col_ = np.array([0.14, 0.13, 0.24], np.float32) + np.array([0.35, 0.35, 0.5], np.float32) * flash
    out = np.empty((fc.h, fc.w, 4), np.float32)
    out[..., :3] = col_ * a_tot
    out[..., 3:] = a_tot
    return out

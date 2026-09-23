"""s05a shared effects: rim-lit silhouettes, screen-space god rays, anamorphic streaks, dust motes,
cached noise tiles, lens helpers. Everything works in design units; pixel work uses fc.s."""
import math

import cairo
import cv2
import numpy as np

from vx import W, H, C, col, blur, hash01, fbm2, set_color, circle

# ----------------------------------------------------------------- palette local to s05a
GOLD = (1.0, 0.72, 0.30)        # tunnel backlight: amber-gold (warmer/duller than flag yellow)
GOLD_HOT = (1.0, 0.86, 0.60)
NVG = (0.55, 1.0, 0.45)         # night-vision phosphor green
HOLO = (0.42, 0.95, 1.0)        # hologram cyan
BASALT = (0.075, 0.085, 0.13)
STEEL = (0.20, 0.22, 0.27)
FLAGY = np.array(C["flag"], np.float32)
FLAGGLOW = np.array(C["flag_glow"], np.float32)

_cache = {}


def _stops(g, stops):
    for s in stops:
        c = s[1]
        a = s[2] if len(s) > 2 else (c[3] if not isinstance(c, str) and len(c) == 4 else 1.0)
        r, gg, b = col(c)
        g.add_color_stop_rgba(s[0], r, gg, b, a)
    return g


def lin_grad(x0, y0, x1, y1, stops):
    """like vx.lin_grad but also accepts (t, (r,g,b,a)) stops"""
    return _stops(cairo.LinearGradient(x0, y0, x1, y1), stops)


def rad_grad(cx, cy, r0, r1, stops, fx=None, fy=None):
    return _stops(cairo.RadialGradient(fx if fx is not None else cx, fy if fy is not None else cy, r0, cx, cy, r1), stops)


def noise_tile(seed=0, size=512, octaves=5, base=6.0):
    """tileable-ish fbm noise tile in [0,1] (float32), cached per process"""
    k = ("tile", seed, size, octaves, base)
    if k not in _cache:
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float64) / size * base
        n = fbm2(xs, ys, seed=seed, octaves=octaves)
        n = (n - n.min()) / (n.max() - n.min() + 1e-9)
        _cache[k] = n.astype(np.float32)
    return _cache[k]


def noise_layer(fc, seed=0, scale=1.0, ox=0.0, oy=0.0, rot=0.0, octaves=5):
    """full-frame noise (h,w) sampled from a cached tile; scale = design px per tile px"""
    tile = noise_tile(seed, octaves=octaves)
    s = fc.s * scale
    c, sn = math.cos(rot), math.sin(rot)
    # dst pixel -> tile coords
    M = np.array([[c / s, sn / s, ox], [-sn / s, c / s, oy]], np.float32)
    return cv2.warpAffine(tile, M, (fc.w, fc.h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_WRAP)


def shift(a, dx, dy):
    """shift an image by pixels (dx, dy), edge-replicate"""
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def rimlit(rgba, fc, fill=0.18, rim=GOLD, strength=1.4, width=6.0, light=(0.0, -1.0), dir_k=0.6, tint=None):
    """Backlit look for a premultiplied character layer: body darkened to `fill`, edges catch a rim light.
    width in design px; light = 2D direction TOWARDS the light (screen space) to bias the rim."""
    A = rgba[..., 3]
    wpx = max(0.8, width * fc.s)
    inner = np.clip(A - blur(A, wpx), 0, 1) * 2.2
    lx, ly = light
    sh = shift(A, -lx * wpx * 1.6, -ly * wpx * 1.6)
    dirr = np.clip(A - sh, 0, 1)
    edge = np.clip(inner * (1 - dir_k) + (inner * 0.6 + dirr) * dir_k, 0, 1.2)
    out = np.empty_like(rgba)
    body = rgba[..., :3] * fill
    if tint is not None:
        body = body * np.asarray(tint, np.float32)
    out[..., :3] = body + edge[..., None] * np.asarray(rim, np.float32) * strength * A[..., None]
    out[..., 3] = A
    return out


def zoom_about(img, f, cx, cy):
    M = np.array([[f, 0, cx * (1 - f)], [0, f, cy * (1 - f)]], np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def god_rays(light, fc, cx, cy, length=0.6, passes=6, decay=0.94, down=4):
    """screen-space volumetric light scattering: `light` (h,w,3) emission/occlusion image,
    (cx, cy) light centre in design units -> rays image (h,w,3)"""
    h, w = light.shape[:2]
    sw, sh = max(8, w // down), max(8, h // down)
    acc = cv2.resize(light, (sw, sh), interpolation=cv2.INTER_AREA)
    px, py = cx * fc.s * sw / w, cy * fc.s * sh / h
    for i in range(passes):
        f = 1.0 - length * 2.0 ** (i - passes)
        acc = 0.5 * (acc + zoom_about(acc, f, px, py) * decay)
    return cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)


def streak(img, fc, thresh=0.8, length=260.0, k=0.5, tint=(1.0, 0.85, 0.65)):
    """anamorphic horizontal lens streak from bright areas"""
    h, w = img.shape[:2]
    L = img.max(axis=2) if img.ndim == 3 else img
    b = np.maximum(L - thresh, 0) / (1 - thresh)
    sw, sh = max(8, w // 4), max(8, h // 4)
    sm = cv2.resize(b, (sw, sh), interpolation=cv2.INTER_AREA)
    sig = max(1.0, length * fc.s / 4)
    s1 = cv2.GaussianBlur(sm, (0, 0), sigmaX=sig, sigmaY=0.6)
    s2 = cv2.GaussianBlur(sm, (0, 0), sigmaX=sig * 3, sigmaY=0.8)
    s = cv2.resize(s1 * 0.6 + s2 * 0.4, (w, h), interpolation=cv2.INTER_LINEAR)
    return s[..., None] * np.asarray(tint, np.float32) * k


def motes(ctx, T, n, seed, box, lit_fn=None, rmin=1.0, rmax=5.0, color=GOLD_HOT, drift=(8.0, -5.0), alpha=0.8,
          speed=1.0):
    """drifting dust motes. box = (x0, y0, x1, y1). lit_fn(x, y) -> 0..1 brightness. Near motes (big) are soft."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    for i in range(n):
        u, v, z = hash01(i, seed), hash01(i, seed + 1), hash01(i, seed + 2)
        r = rmin + (rmax - rmin) * z ** 3
        ph = hash01(i, seed + 3) * 6.28
        t = T * speed
        x = x0 + ((u * bw + drift[0] * t * (0.4 + z) + 14 * math.sin(t * 0.7 + ph)) % bw)
        y = y0 + ((v * bh + drift[1] * t * (0.4 + z) + 10 * math.sin(t * 0.5 + ph * 1.3)) % bh)
        lit = lit_fn(x, y) if lit_fn else 1.0
        tw = 0.6 + 0.4 * math.sin(t * (1.5 + z * 2) + ph)
        a = alpha * lit * tw * (0.35 + 0.65 * (1 - z * 0.5))
        if a < 0.01:
            continue
        if r > 2.5:     # soft bokeh disc
            g = cairo.RadialGradient(x, y, 0, x, y, r)
            g.add_color_stop_rgba(0, *color, a * 0.55)
            g.add_color_stop_rgba(0.75, *color, a * 0.35)
            g.add_color_stop_rgba(1, *color, 0)
            ctx.set_source(g)
        else:
            set_color(ctx, color, a)
        circle(ctx, x, y, r)
        ctx.fill()


def speed_lines(ctx, cx, cy, n, seed, r0, r1, color=(1, 1, 1), alpha=0.3, width=2.0, ang0=0.0, spread=math.tau):
    for i in range(n):
        a = ang0 + (hash01(i, seed) - 0.5) * spread
        rr0 = r0 + hash01(i, seed + 1) * (r1 - r0) * 0.5
        rr1 = rr0 + (r1 - r0) * (0.3 + 0.7 * hash01(i, seed + 2))
        ctx.set_line_width(width * (0.4 + hash01(i, seed + 3)))
        set_color(ctx, color, alpha * (0.3 + 0.7 * hash01(i, seed + 4)))
        ctx.move_to(cx + math.cos(a) * rr0, cy + math.sin(a) * rr0)
        ctx.line_to(cx + math.cos(a) * rr1, cy + math.sin(a) * rr1)
        ctx.stroke()


def radial_img(fc, cx, cy, r, color, power=2.0, aspect=1.0):
    """additive radial falloff image (h,w,3)"""
    k = ("grid", fc.h, fc.w)
    if k not in _cache:
        yy, xx = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
        _cache[k] = (xx / fc.s, yy / fc.s)
    xx, yy = _cache[k]
    d = np.sqrt(((xx - cx) / aspect) ** 2 + (yy - cy) ** 2) / r
    m = 1.0 / (1.0 + d ** power)
    return m[..., None] * np.asarray(col(color), np.float32)


def grid_xy(fc):
    k = ("grid", fc.h, fc.w)
    if k not in _cache:
        yy, xx = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
        _cache[k] = (xx / fc.s, yy / fc.s)
    return _cache[k]


def vgrad(fc, stops):
    from vx import gradient_v
    return gradient_v(fc, stops).astype(np.float32)


def comp(dst, rgba):
    """premultiplied over"""
    return rgba[..., :3] + dst * (1.0 - rgba[..., 3:4])


def to_premul(rgb, a):
    return np.concatenate([rgb * a[..., None], a[..., None]], -1).astype(np.float32)

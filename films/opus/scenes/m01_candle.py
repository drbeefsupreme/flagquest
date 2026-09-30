"""m01 - the pilgrim's candle: a slim ivory beeswax taper and a painted, living flame; the match that lights it.

Pure functions of time (frames render out of order).  Everything is drawn in design units on a float32 sRGB
image (h, w, 3) at pixel scale s:

    candle(img, s, x, y, h, t, lit=1.0, seed=0, light=1.0) -> img
        x, y  = top of the wick (design px); h = flame height (design px, ~60-120);
        lit   = 0..1 flame size (0 = out: a smouldering wick if 0 < ember); light = 0..1 how much the flame
                lights its own wax (the glow at the taper's top).
    flame_light(t, seed=0) -> intensity multiplier of the flame (flicker) for lighting the room
    match(img, s, x, y, h, dt, seed=0) -> img        the match: dt = seconds since the strike (flare at 0)
    snuff(t_since) -> (lit, smoke): a snuffed flame: shrinks, a thread of smoke rises (for the ending)
    smoke(img, s, x, y, h, dt, seed=0) -> img        the thread of smoke after a snuff

M10 (the ending) uses the same candle: draw it with candle(...) and animate lit with snuff().
"""
import math

import cairo
import cv2
import numpy as np

from vx.canvas import Canvas, blur, surface_from_array
from vx.ease import fbm1, noise1, clamp, smoothstep


def flame_light(t, seed=0):
    """flicker of the flame's light output (1 = steady)"""
    return 1.0 + 0.10 * fbm1(t * 2.3, seed + 3, 3) + 0.05 * noise1(t * 9.7, seed + 4) + 0.03 * noise1(t * 17.0, seed + 5)


def _flame_params(t, seed):
    hk = 1.0 + 0.09 * fbm1(t * 2.1, seed + 11, 3) + 0.04 * noise1(t * 11.0, seed + 12)
    sway = 0.10 * fbm1(t * 0.9, seed + 21, 3) + 0.05 * noise1(t * 6.5, seed + 22)
    wk = 1.0 + 0.05 * noise1(t * 7.3, seed + 31)
    return hk, sway, wk


def _flame_field(Wp, Hp, s, h, t, seed, lit):
    """emissive flame image (Hp, Wp, 3) linear, around a local origin at the wick top (Wp/2, Hp*0.78)"""
    hk, sway, wk = _flame_params(t, seed)
    H = h * hk * lit
    Wd = h * 0.26 * wk * (0.55 + 0.45 * lit)
    yy, xx = np.mgrid[0:Hp, 0:Wp].astype(np.float32)
    ox, oy = Wp * 0.5, Hp * 0.78
    X = (xx + 0.5 - ox) / s
    Y = (oy - (yy + 0.5)) / s            # up = +
    v = Y / max(H, 1e-3)                 # 0 at the wick top, 1 at the tip
    # the flame leans and sways more toward its tip
    X = X - sway * h * np.clip(v, 0, 1.2) ** 2
    # teardrop radius profile: round bottom, pointed top; the base sits slightly below the wick top
    vb = v + 0.16
    vc = np.clip(vb / 1.16, 0, 1)
    prof = np.where(vb > 0, np.clip(np.sin(vc * math.pi), 0, 1) ** 0.62 * (1 - 0.55 * vc), 0)
    rr = np.abs(X) / (Wd * np.maximum(prof, 1e-3))
    inside = np.clip(1.0 - rr, 0, 1) * (vb > 0) * (vb < 1.16)
    env = np.clip(inside * 1.6, 0, 1) ** 1.4
    # colour zones: blue root, white-yellow core, orange-gold envelope, red-dim tip
    core_v = np.exp(-((v - 0.28) / 0.26) ** 2)
    core = np.clip(1.0 - rr * 1.9, 0, 1) * core_v
    blue = np.exp(-((v + 0.06) / 0.07) ** 2) * np.clip(1.0 - np.abs(X) / (Wd * 0.95), 0, 1) * (1 - core * 0.8)
    tipdim = np.clip((v - 0.55) / 0.5, 0, 1)
    R = env * (1.9 - 0.9 * tipdim) + core * 2.4
    G = env * (1.0 - 0.55 * tipdim) + core * 2.0
    B = env * (0.25 - 0.2 * tipdim) + core * 1.1
    R = R * (1 - 0.6 * blue) + blue * 0.25
    G = G * (1 - 0.6 * blue) + blue * 0.35
    B = B * (1 - 0.6 * blue) + blue * 1.1
    return np.dstack([R, G, B]).astype(np.float32), H, Wd


def _taper(ctx, x, y, h, glow, light):
    """the wax taper below the wick top (x, y); h = flame height. Beeswax ivory, warm lit top."""
    tw = h * 0.36                  # half width of the taper
    top = y + h * 0.10
    bot = y + h * 6.0
    # body: cylindrical shading, lit from its own flame above
    g = cairo.LinearGradient(x - tw, 0, x + tw, 0)
    base = (0.44, 0.37, 0.27)
    for k, a in ((0.0, 0.35), (0.22, 0.75), (0.45, 1.0), (0.7, 0.8), (1.0, 0.38)):
        g.add_color_stop_rgb(k, base[0] * a * light, base[1] * a * light, base[2] * a * light)
    ctx.set_source(g)
    ctx.move_to(x - tw, bot)
    ctx.line_to(x - tw, top + tw * 0.25)
    ctx.curve_to(x - tw, top - tw * 0.05, x + tw, top - tw * 0.05, x + tw, top + tw * 0.25)
    ctx.line_to(x + tw, bot)
    ctx.close_path()
    ctx.fill()
    # the top of the wax glows (translucent beeswax around the melt pool)
    rg = cairo.RadialGradient(x, top + tw * 0.2, 0, x, top + tw * 0.2, tw * 3.2)
    rg.add_color_stop_rgba(0.0, 1.0, 0.78, 0.45, 0.85 * glow)
    rg.add_color_stop_rgba(0.35, 0.95, 0.62, 0.30, 0.45 * glow)
    rg.add_color_stop_rgba(1.0, 0.6, 0.35, 0.15, 0.0)
    ctx.save()
    ctx.rectangle(x - tw, top - tw, 2 * tw, tw * 4.5)
    ctx.clip()
    ctx.set_source(rg)
    ctx.paint()
    ctx.restore()
    # melt-pool rim highlight
    ctx.set_source_rgba(1.0, 0.86, 0.6, 0.55 * glow)
    ctx.set_line_width(max(0.8, h * 0.025))
    ctx.move_to(x - tw * 0.92, top + tw * 0.22)
    ctx.curve_to(x - tw * 0.6, top + tw * 0.42, x + tw * 0.6, top + tw * 0.42, x + tw * 0.92, top + tw * 0.22)
    ctx.stroke()
    # a wax drip down the front
    ctx.set_source_rgba(0.62 * light + 0.25 * glow, 0.52 * light + 0.16 * glow, 0.38 * light + 0.06 * glow, 0.9)
    dx = x + tw * 0.35
    ctx.move_to(dx - tw * 0.13, top + tw * 0.3)
    ctx.curve_to(dx - tw * 0.16, top + tw * 1.4, dx - tw * 0.12, top + tw * 2.2, dx, top + tw * 2.4)
    ctx.curve_to(dx + tw * 0.12, top + tw * 2.2, dx + tw * 0.14, top + tw * 1.3, dx + tw * 0.13, top + tw * 0.3)
    ctx.close_path()
    ctx.fill()


def _wick(ctx, x, y, h, t, lit, ember):
    tw = h * 0.36
    top = y + h * 0.10
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_width(max(0.7, h * 0.035))
    # the wick curls a little in the flame
    ctx.set_source_rgba(0.05, 0.035, 0.03, 1.0)
    ctx.move_to(x, top + tw * 0.2)
    ctx.curve_to(x, y + h * 0.02, x + h * 0.01, y - h * 0.06, x + h * 0.035, y - h * 0.10)
    ctx.stroke()
    if ember > 0:
        ctx.set_source_rgba(1.0, 0.35, 0.08, ember)
        ctx.arc(x + h * 0.035, y - h * 0.10, max(0.8, h * 0.022), 0, 2 * math.pi)
        ctx.fill()


def candle(img, s, x, y, h, t, lit=1.0, seed=0, light=1.0, ember=0.0, halo=1.0):
    """draw the candle + flame on img (float32 sRGB (H, W, 3), pixel scale s) with its top of wick at (x, y)."""
    Hh, Ww = img.shape[:2]
    # taper + wick (cairo, on a small canvas region)
    pad = h * 3.5
    x0, y0 = int(max(0, (x - pad) * s)), int(max(0, (y - h * 2.2) * s))
    x1, y1 = int(min(Ww, (x + pad) * s)), int(min(Hh, (y + h * 6.5) * s))
    if x1 <= x0 or y1 <= y0:
        return img
    cw, ch = x1 - x0, y1 - y0
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, cw, ch)
    ctx = cairo.Context(surf)
    ctx.scale(s, s)
    ctx.translate(-x0 / s, -y0 / s)
    _taper(ctx, x, y, h, glow=lit * min(1.0, light), light=(0.10 + 0.9 * lit) * light)
    _wick(ctx, x, y, h, t, lit, ember)
    surf.flush()
    buf = np.ndarray((ch, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :cw]
    a = buf[..., 3:4].astype(np.float32) / 255.0
    rgb = buf[..., 2::-1].astype(np.float32) / 255.0
    reg = img[y0:y1, x0:x1]
    reg[:] = rgb + reg * (1.0 - a)
    if lit <= 0.01:
        return img
    # flame (emissive, additive) + halo
    fh = h * 1.6
    fw = h * 1.2
    fx0, fy0 = int(max(0, (x - fw) * s)), int(max(0, (y - fh) * s))
    fx1, fy1 = int(min(Ww, (x + fw) * s)), int(min(Hh, (y + h * 0.45) * s))
    if fx1 > fx0 and fy1 > fy0:
        Wp, Hp = int(2 * fw * s), int((fh + h * 0.45) * s)
        F, H, Wd = _flame_field(Wp, Hp, s, h, t, seed, lit)
        # local origin of the field -> image coords
        lx0 = int(round((x - fw) * s))
        ly0 = int(round((y - fh) * s)) + int(round(Hp * 0.78)) - int(round(fh * s))
        ly0 = int(round(y * s - Hp * 0.78))
        ax0, ay0 = max(0, lx0), max(0, ly0)
        ax1, ay1 = min(Ww, lx0 + Wp), min(Hh, ly0 + Hp)
        if ax1 > ax0 and ay1 > ay0:
            sub = F[ay0 - ly0:ay1 - ly0, ax0 - lx0:ax1 - lx0]
            soft = cv2.GaussianBlur(sub, (0, 0), max(0.6, 0.012 * h * s))
            reg = img[ay0:ay1, ax0:ax1]
            # a flame is light: screen it on (keeps the core white-hot without harsh clipping)
            reg[:] = 1.0 - (1.0 - reg) * np.exp(-soft * 1.25)
    if halo > 0:
        # soft warm halo around the flame (the air glowing)
        R = h * 3.2
        hx0, hy0 = int(max(0, (x - R) * s)), int(max(0, (y - h * 0.5 - R) * s))
        hx1, hy1 = int(min(Ww, (x + R) * s)), int(min(Hh, (y - h * 0.5 + R) * s))
        if hx1 > hx0 and hy1 > hy0:
            yy, xx = np.mgrid[hy0:hy1, hx0:hx1].astype(np.float32)
            d = np.hypot((xx + 0.5) / s - x, ((yy + 0.5) / s - (y - h * 0.45)) * 0.8) / h
            k = halo * lit * flame_light(t, seed) * (0.20 * np.exp(-d * 1.4) + 0.10 * np.exp(-d * 0.5))
            reg = img[hy0:hy1, hx0:hx1]
            reg[..., 0] += k * 1.0
            reg[..., 1] += k * 0.62
            reg[..., 2] += k * 0.28
    return img


def match(img, s, x, y, h, dt, seed=0):
    """the match that lights the candle. dt = seconds since the strike flare (negative: the scrape).
    The match head flares at the wick (x, y), sparks fly, its small flame lingers, then the match withdraws
    down-right out of view. Returns img."""
    if dt < -0.12 or dt > 1.2:
        return img
    Hh, Ww = img.shape[:2]
    rng = np.random.default_rng(seed + 77)
    # match position: held at the wick, then withdrawn
    wd = smoothstep(0.42, 1.05, dt)
    mx = x + h * 0.25 + wd * h * 5.0
    my = y - h * 0.15 + wd * wd * h * 5.5
    layer = np.zeros((Hh, Ww, 3), np.float32)
    # scrape sparks before the flare, flare burst, then the match's own small flame
    if dt < 0:
        k = 1.0 - (-dt / 0.12)
        _dot(layer, s, mx + h * 0.4 * (1 + dt * 8), my + h * 0.1, h * 0.06, (2.0 * k, 1.2 * k, 0.5 * k))
    else:
        fl = math.exp(-dt / 0.07)
        _dot(layer, s, mx, my, h * (0.10 + 0.24 * (1 - math.exp(-dt / 0.03))), (6.5 * fl, 4.0 * fl, 1.5 * fl))
        # sparks
        n = 16
        ang = rng.uniform(-math.pi, 0.2, n)
        sp = rng.uniform(1.5, 5.0, n) * h
        life = rng.uniform(0.12, 0.45, n)
        for i in range(n):
            if dt > life[i]:
                continue
            px = x + h * 0.25 + math.cos(ang[i]) * sp[i] * dt
            py = y - h * 0.15 + math.sin(ang[i]) * sp[i] * dt + 0.5 * 9.0 * h * dt * dt
            k = (1 - dt / life[i]) ** 1.5
            _dot(layer, s, px, py, h * 0.035, (3.0 * k, 1.8 * k, 0.6 * k))
        # the match flame (small, bluish base)
        if dt > 0.05:
            ml = min(1.0, (dt - 0.05) / 0.12) * (1.0 - smoothstep(0.95, 1.2, dt))
            _dot(layer, s, mx, my - h * 0.18, h * 0.22 * ml, (2.2 * ml, 1.3 * ml, 0.45 * ml), elong=1.9)
    # the match stick (dark, thin) - visible only against the flare light
    if dt > 0.0:
        # the stick is seen only by its own light: bright in the flare, fading as it withdraws
        _stick(img, s, mx, my, h, (0.35 + 0.65 * math.exp(-dt / 0.25)) * (1.0 - smoothstep(0.8, 1.15, dt)))
    g = cv2.GaussianBlur(layer, (0, 0), max(0.7, h * 0.03 * s))
    img[:] = 1.0 - (1.0 - img) * np.exp(-g)
    return img


def _dot(layer, s, x, y, r, c, elong=1.0):
    if r <= 0:
        return
    Hh, Ww = layer.shape[:2]
    R = r * 2.5
    x0, y0 = int(max(0, (x - R) * s)), int(max(0, (y - R * elong) * s))
    x1, y1 = int(min(Ww, (x + R) * s + 1)), int(min(Hh, (y + R) * s + 1))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dy = (yy + 0.5) / s - y
    dy = np.where(dy < 0, dy / elong, dy)
    d2 = (((xx + 0.5) / s - x) ** 2 + dy ** 2) / (r * r)
    k = np.exp(-d2 * 1.6)[..., None]
    layer[y0:y1, x0:x1] += k * np.array(c, np.float32)


def _stick(img, s, mx, my, h, a):
    if a <= 0:
        return
    Hh, Ww = img.shape[:2]
    x0, y0 = mx, my
    x1, y1 = mx + h * 1.5, my + h * 1.25
    xa, ya = int(max(0, min(x0, x1) * s - 3)), int(max(0, min(y0, y1) * s - 3))
    xb, yb = int(min(Ww, max(x0, x1) * s + 3)), int(min(Hh, max(y0, y1) * s + 3))
    if xb <= xa or yb <= ya:
        return
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    px, py = (xx + 0.5) / s, (yy + 0.5) / s
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    u = np.clip(((px - x0) * vx + (py - y0) * vy) / L2, 0, 1)
    d = np.hypot(px - (x0 + u * vx), py - (y0 + u * vy))
    w = h * 0.045
    k = np.clip((w - d) * s + 0.5, 0, 1) * a
    reg = img[ya:yb, xa:xb]
    # charred tip near the head, pale wood behind (lit faintly by the flare)
    col = np.where((u < 0.12)[..., None], np.array([0.05, 0.04, 0.035], np.float32),
                   np.array([0.30, 0.22, 0.14], np.float32))
    reg[:] = reg * (1 - k[..., None]) + col * k[..., None]


def snuff(t_since):
    """(lit, ember) for a flame snuffed t_since seconds ago (<0 = still burning)."""
    if t_since <= 0:
        return 1.0, 0.0
    lit = max(0.0, 1.0 - t_since / 0.12)
    ember = math.exp(-max(0.0, t_since - 0.1) / 0.9) if t_since > 0.05 else 0.0
    return lit, ember


def smoke(img, s, x, y, h, dt, seed=0):
    """a thin thread of smoke rising from a snuffed wick (dt seconds after the snuff)."""
    if dt <= 0 or dt > 6.0:
        return img
    Hh, Ww = img.shape[:2]
    n = 90
    k = np.arange(n) / n
    rise = h * 7.0 * k
    t_ = dt - k * 1.8
    vis = t_ > 0
    wob = np.array([fbm1(dt * 0.6 + kk * 3.0, seed + 5, 3) for kk in k]) * h * (0.2 + 1.8 * k)
    px = x + h * 0.035 + wob
    py = y - h * 0.12 - rise
    a = np.clip(1 - dt / 6.0, 0, 1) * (1 - k) ** 1.5 * vis * 0.35
    layer = np.zeros((Hh, Ww), np.float32)
    for i in range(n):
        if a[i] <= 0.005:
            continue
        xi, yi = int(px[i] * s), int(py[i] * s)
        if 0 <= xi < Ww and 0 <= yi < Hh:
            layer[yi, xi] += a[i]
    layer = cv2.GaussianBlur(layer, (0, 0), max(0.8, h * 0.05 * s))
    img += layer[..., None] * np.array([0.55, 0.52, 0.5], np.float32) * 3.0
    return img

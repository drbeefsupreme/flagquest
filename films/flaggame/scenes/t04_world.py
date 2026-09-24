"""t04 — the grey Doré world (painted in cairo greys, then printed by vx.ink.engrave).

Everything here is NEUTRAL GREY (tone only): the engraving turns tone into hatching; the Flags are never
drawn here (they live on the spot plate). Helpers are pure functions of their arguments (+ cached textures).
"""
import math

import cairo
import cv2
import numpy as np
from numba import njit
from vx import (Canvas, W, H, set_color, lin_grad, rad_grad, ellipse, circle, poly, smooth, hash01, noise1,
                fbm1, clamp, lerp, smoothstep)
from vx.canvas import surface_from_array

G = lambda v: (v, v, v)  # noqa: E731   grey tone -> rgb


# ============================================================ numpy textures
def _vnoise(shape, cells, rng):
    """smooth value noise on a (h, w) grid with `cells` lattice cells across the width"""
    h, w = shape
    ch = max(2, int(round(cells * h / w)))
    g = rng.random((ch + 3, cells + 3)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def fbm_tex(shape, seed, base=4, octaves=6, gain=0.55):
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32)
    a, tot = 1.0, 0.0
    for o in range(octaves):
        out += a * _vnoise(shape, base * 2 ** o, rng)
        tot += a
        a *= gain
    return out / tot


def cloud_tex(shape, seed, light=(0.5, 0.35), cover=0.52, dark=0.16, bright=0.97):
    """Doré storm clouds: billowing masses lit from `light` (fractions of the texture), bright rims, dark
    bellies, a clearing around the light. Returns tone (h, w) float32 0..1."""
    h, w = shape
    f = fbm_tex(shape, seed, base=3, octaves=6)
    wx = fbm_tex(shape, seed + 1, base=2, octaves=4) - 0.5
    wy = fbm_tex(shape, seed + 2, base=2, octaves=4) - 0.5
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = (xs + wx * w * 0.18).astype(np.float32)
    my = (ys + wy * h * 0.30).astype(np.float32)
    f = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # billows: horizontal stretch + a clearing around the light
    lx, ly = light[0] * w, light[1] * h
    r = np.sqrt(((xs - lx) / (w * 0.42)) ** 2 + ((ys - ly) / (h * 0.55)) ** 2)
    dens = np.clip((f - (cover - 0.30 * np.exp(-r * r * 2.2) * 1.0)) * 4.0, 0, 1)
    dens = dens * dens * (3 - 2 * dens)
    hgt = cv2.GaussianBlur(dens, (0, 0), w * 0.012)
    gy, gx = np.gradient(hgt)
    # normal-ish shading toward the light
    dx, dy = lx - xs, ly - ys
    dn = np.sqrt(dx * dx + dy * dy) + 1e-3
    lit = np.clip(0.5 + (gx * dx / dn + gy * dy / dn) * w * 0.9, 0, 1)
    rim = np.clip(np.abs(np.gradient(dens)[0]) * h * 0.06, 0, 1)
    glow = np.exp(-r * r * 1.6)
    sky = 0.42 + 0.55 * glow                        # the open sky: dark far, bright near the light
    cloud = dark + (0.62 - dark) * lit + 0.30 * glow * lit
    tone = sky * (1 - dens) + cloud * dens
    tone = np.clip(tone + rim * 0.25 * glow, 0, 1)
    # small torn wisps
    wisp = fbm_tex(shape, seed + 7, base=18, octaves=3)
    tone = np.clip(tone + (wisp - 0.5) * 0.10, 0, 1)
    return np.clip(tone * (bright - 0.0), 0, 1).astype(np.float32)


def paint_tone(ctx, tone, x, y, w, h, alpha=1.0, filt=cairo.FILTER_BILINEAR):
    """paint a tone (h, w) or rgb array stretched over the design rect (x, y, w, h) under the ctx transform"""
    arr = tone if tone.ndim == 3 else np.repeat(tone[..., None], 3, axis=2)
    surf = surface_from_array(arr)
    ih, iw = arr.shape[:2]
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(w / iw, h / ih)
    pat = cairo.SurfacePattern(surf)
    pat.set_filter(filt)
    pat.set_extend(cairo.EXTEND_PAD)
    ctx.set_source(pat)
    ctx.rectangle(0, 0, iw, ih)
    ctx.clip()
    ctx.paint_with_alpha(alpha)
    ctx.restore()


# ============================================================ storm sky (Doré billows)
def make_billows(seed, light, *, n_clusters=26, rx=1300, ry=620, ring0=0.22, size=1.0, ybias=0.0):
    """deterministic cloud billows around a light opening: each cluster = a few big flattened bases plus a
    cauliflower of smaller billows along its light-facing crown. Returns list of (x, y, r, dd, rnd, amp)."""
    rng = np.random.default_rng(seed)
    lx, ly = light
    out = []
    for c in range(n_clusters):
        a = rng.uniform(0, 2 * math.pi)
        d = ring0 + rng.random() ** 0.7 * (1.1 - ring0)
        cx = lx + math.cos(a) * rx * d
        cy = ly + math.sin(a) * ry * d + ybias
        big = (110 + 220 * d * rng.uniform(0.7, 1.3)) * size
        ux, uy = (lx - cx), (ly - cy)
        un = math.hypot(ux, uy) + 1e-6
        ux, uy = ux / un, uy / un
        for b in range(3):                                   # bases
            out.append((cx + rng.normal(0, big * 0.8), cy + rng.normal(0, big * 0.15), big * rng.uniform(0.8, 1.3),
                        d, rng.random(), 0.9))
        nb = int(10 + 16 * rng.random())
        for b in range(nb):                                  # crown billows, biased toward the light & up
            ang = rng.normal(0, 1.2)
            dirx = ux * math.cos(ang) - uy * math.sin(ang)
            diry = ux * math.sin(ang) + uy * math.cos(ang)
            diry = diry * 0.5 - 0.5
            rad = big * rng.uniform(0.55, 1.05)
            r = big * rng.uniform(0.18, 0.5) * (1.0 - 0.25 * b / nb)
            out.append((cx + dirx * rad * 1.3, cy + diry * rad * 0.55, r, d, rng.random(), 0.75 + 0.4 * rng.random()))
    out.sort(key=lambda b: -b[3])
    return out


@njit(cache=True, fastmath=True)
def _splat(D, xs, ys, rs, amp, sx, sy, ox, oy):
    """sum smooth kernels (1 - d^2/r^2)^2 * amp into D; positions in design px, grid scale sx (grid px per
    design px), offset (ox, oy) design px of grid (0, 0)"""
    h, w = D.shape
    for k in range(xs.shape[0]):
        cx = (xs[k] - ox) * sx
        cy = (ys[k] - oy) * sy
        r = rs[k] * sx
        ry = r * 0.62
        i0 = max(0, int(cy - ry) - 1)
        i1 = min(h, int(cy + ry) + 2)
        j0 = max(0, int(cx - r) - 1)
        j1 = min(w, int(cx + r) + 2)
        for i in range(i0, i1):
            dy = (i - cy) / ry
            for j in range(j0, j1):
                dx = (j - cx) / r
                q = 1.0 - (dx * dx + dy * dy)
                if q > 0:
                    D[i, j] += amp[k] * q * q


def storm_sky(fc, billows, light, t, cam_off=(0.0, 0.0), *, res=0.5, drift=(7.0, -1.2), dark=0.10,
              light_tone=0.99, cover=0.30, breathe=1.0, rim=0.45, zoom=1.0, far=0.42, noise=None):
    """Doré storm sky as a tone image (fc.h, fc.w): merged billowing cloud masses (metaballs) shaded from the
    bright opening at `light` (design px, screen space). cam_off shifts the clouds (parallax)."""
    gw, gh = int(W * res * 0.5) * 2, int(H * res * 0.5) * 2
    D = np.zeros((gh, gw), np.float32)
    b = billows
    xs = (b[:, 0] + drift[0] * t * (0.5 + b[:, 4]) - cam_off[0] - W / 2) * zoom + W / 2
    ys = (b[:, 1] + drift[1] * t * (0.5 + b[:, 4]) - cam_off[1] - H / 2) * zoom + H / 2
    rs = b[:, 2] * zoom * (1.0 + 0.05 * breathe * np.sin(t * 0.33 + b[:, 4] * 6.28))
    _splat(D, xs.astype(np.float32), ys.astype(np.float32), rs.astype(np.float32), b[:, 5].astype(np.float32),
           np.float32(gw / W), np.float32(gh / H), np.float32(0), np.float32(0))
    if noise is not None:
        D += (cv2.resize(noise, (gw, gh)) - 0.5) * 0.35 * np.clip(D * 3, 0, 1)
    Hh = cv2.GaussianBlur(D, (0, 0), 1.5)
    gy, gx = np.gradient(Hh)
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    lx, ly = light[0] * gw / W, light[1] * gh / H
    vx, vy = lx - xx, ly - yy
    vn = np.sqrt(vx * vx + vy * vy) + 1e-3
    k = 60.0 * res
    nz = 1.0 / np.sqrt(1 + (gx * k) ** 2 + (gy * k) ** 2)
    ndl = (-gx * k * vx / vn - gy * k * vy / vn) * nz + nz * 0.15
    lit = np.clip(ndl, 0, 1)
    rr = vn / (gw * 0.5)
    glow = np.exp(-rr * rr * 2.2)
    cov = np.clip((D - cover) / 0.18, 0, 1)
    cov = cov * cov * (3 - 2 * cov)
    inner = np.clip((D - cover) / 1.1, 0, 1)
    cloud = dark + (0.25 + 0.65 * glow) * lit ** 1.4 * (1 - 0.55 * inner) + 0.10 * (1 - inner) * glow
    edge = np.clip(1 - np.abs(cov - 0.5) * 2, 0, 1) * (0.3 + 0.7 * glow)   # silver lining
    sky = far + (light_tone - far) * glow ** 0.6
    tone = sky * (1 - cov) + cloud * cov + rim * edge * np.clip(lit * 1.5 + 0.2, 0, 1)
    tone = cv2.resize(np.clip(tone, 0, 1).astype(np.float32), (fc.w, fc.h), interpolation=cv2.INTER_CUBIC)
    return np.clip(tone, 0, 1)


def billow_array(billows):
    """list from make_billows -> float32 array (n, 6): x, y, r, dd, rnd, amp"""
    return np.array(billows, np.float32)


# ============================================================ light
def glory(ctx, cx, cy, r0, r1, t, *, n=42, strength=1.0, seed=3, lines=True, spread=1.0):
    """engraved radiance: soft white bloom + white wedge beams (slowly turning) + fine dark ray lines
    beyond the halo, like a Doré 'glory'."""
    ctx.save()
    ctx.set_source(rad_grad(cx, cy, 0, r1 * 0.9, [(0, (1, 1, 1), 0.95 * strength), (0.25, (1, 1, 1), 0.55 * strength),
                                                   (1, (1, 1, 1), 0.0)]))
    ctx.arc(cx, cy, r1, 0, 2 * math.pi)
    ctx.fill()
    for k in range(n):
        h = hash01(k, seed)
        a = (k + 0.5 * h) / n * 2 * math.pi * spread + t * 0.035 * (1 if k % 2 else -0.6)
        wdt = (0.010 + 0.030 * hash01(k, seed + 1)) * (1.0 + 0.3 * math.sin(t * 0.8 + k))
        L = r1 * (0.55 + 0.45 * hash01(k, seed + 2))
        ctx.move_to(cx + math.cos(a) * r0 * 0.3, cy + math.sin(a) * r0 * 0.3)
        ctx.line_to(cx + math.cos(a - wdt) * L, cy + math.sin(a - wdt) * L)
        ctx.line_to(cx + math.cos(a + wdt) * L, cy + math.sin(a + wdt) * L)
        ctx.close_path()
        g = rad_grad(cx, cy, r0 * 0.3, L, [(0, (1, 1, 1), 0.55 * strength), (1, (1, 1, 1), 0.0)])
        ctx.set_source(g)
        ctx.fill()
    if lines:
        # engraver's fine radial lines, fading in from the halo edge
        m = n * 4
        ctx.set_line_width(1.3)
        for k in range(m):
            a = k / m * 2 * math.pi * spread + t * 0.01
            a0 = r0 * (1.0 + 0.25 * hash01(k, seed + 5))
            a1 = r1 * (0.8 + 0.35 * hash01(k, seed + 6))
            g = lin_grad(cx + math.cos(a) * a0, cy + math.sin(a) * a0, cx + math.cos(a) * a1, cy + math.sin(a) * a1,
                         [(0, (0.25, 0.25, 0.25), 0.0), (0.3, (0.25, 0.25, 0.25), 0.35 * strength),
                          (1, (0.25, 0.25, 0.25), 0.0)])
            ctx.set_source(g)
            ctx.move_to(cx + math.cos(a) * a0, cy + math.sin(a) * a0)
            ctx.line_to(cx + math.cos(a) * a1, cy + math.sin(a) * a1)
            ctx.stroke()
    ctx.restore()


def light_pool(ctx, cx, cy, rx, ry, a=0.6):
    """lighten a region (white radial wash)"""
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(rx, ry)
    ctx.set_source(rad_grad(0, 0, 0, 1, [(0, (1, 1, 1), a), (0.5, (1, 1, 1), a * 0.45), (1, (1, 1, 1), 0)]))
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()


def shadow_pool(ctx, cx, cy, rx, ry, a=0.6, tone=0.05):
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(rx, ry)
    ctx.set_source(rad_grad(0, 0, 0, 1, [(0, G(tone), a), (0.6, G(tone), a * 0.5), (1, G(tone), 0)]))
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()


# ============================================================ candlesticks
# Revelation's lampstand as a tall ornate church candlestick: (height fraction from the ground, half-width in
# units of the total height). Knops, a baluster, a drip pan, a tall candle.
_STICK = [(0.000, 0.105), (0.018, 0.105), (0.030, 0.085), (0.050, 0.060), (0.075, 0.030), (0.085, 0.024),
          (0.098, 0.040), (0.112, 0.024), (0.30, 0.013), (0.315, 0.026), (0.335, 0.030), (0.352, 0.016),
          (0.52, 0.011), (0.535, 0.024), (0.555, 0.028), (0.572, 0.014), (0.70, 0.011), (0.715, 0.020),
          (0.735, 0.024), (0.748, 0.013), (0.775, 0.013), (0.790, 0.030), (0.800, 0.058), (0.812, 0.060),
          (0.822, 0.022)]
CANDLE_TOP = 0.965


def candlestick(ctx, x, y, h, t, seed=0, *, light_dx=0.0, flame=1.0, tone=0.55, wet=True):
    """tall ornate candlestick standing at ground point (x, y), total height h (to the flame tip).
    light_dx: -1..1 which side the key light comes from (the Flag Maker's radiance)."""
    pts_r = [(x + r * h, y - f * h) for f, r in _STICK]
    pts_l = [(x - r * h, y - f * h) for f, r in reversed(_STICK)]
    ctx.save()
    # contact shadow / reflection in the mud
    ellipse(ctx, x, y + h * 0.004, h * 0.13, h * 0.018)
    set_color(ctx, G(0.12), 0.7)
    ctx.fill()
    ctx.move_to(*pts_r[0])
    for p in pts_r[1:]:
        ctx.line_to(*p)
    for p in pts_l:
        ctx.line_to(*p)
    ctx.close_path()
    wmax = h * 0.11
    s = 0.5 + 0.5 * clamp(light_dx, -1, 1)
    g = lin_grad(x - wmax, 0, x + wmax, 0,
                 [(0, G(tone * 0.25)), (0.30 + 0.25 * s, G(min(0.97, tone + 0.42))), (0.42 + 0.25 * s, G(tone)),
                  (1, G(tone * 0.18))])
    ctx.set_source(g)
    ctx.fill_preserve()
    set_color(ctx, G(0.02))
    ctx.set_line_width(max(1.2, h * 0.0045))
    ctx.stroke()
    # knop highlights (engraved specular ticks)
    ctx.set_line_width(max(1.0, h * 0.003))
    for f, r in _STICK:
        if r > 0.022:
            set_color(ctx, (1, 1, 1), 0.75)
            ctx.move_to(x - r * h * (0.1 - 0.6 * (s - 0.5)), y - f * h)
            ctx.line_to(x - r * h * (0.1 - 0.6 * (s - 0.5)), y - f * h - h * 0.008)
            ctx.stroke()
    # candle
    cw = h * 0.018
    c0, c1 = y - 0.822 * h, y - CANDLE_TOP * h
    ctx.rectangle(x - cw, c1, cw * 2, c0 - c1)
    ctx.set_source(lin_grad(x - cw, 0, x + cw, 0, [(0, G(0.55)), (0.4, G(0.98)), (1, G(0.62))]))
    ctx.fill_preserve()
    set_color(ctx, G(0.05))
    ctx.set_line_width(max(1.0, h * 0.003))
    ctx.stroke()
    # wax drips
    for k in range(3):
        dx = (hash01(k, seed) - 0.5) * cw * 1.6
        L = h * (0.01 + 0.03 * hash01(k, seed + 9))
        ctx.move_to(x + dx, c1 + h * 0.004)
        ctx.line_to(x + dx, c1 + L)
        set_color(ctx, G(0.4), 0.8)
        ctx.set_line_width(max(1.0, h * 0.004))
        ctx.stroke()
    ctx.restore()
    if flame > 0:
        draw_flame(ctx, x, c1, h * 0.075 * flame, t, seed)
    return (x, c1 - h * 0.05)


def draw_flame(ctx, x, y, fh, t, seed=0):
    """a white-hot candle flame (paper white core, soft white halo that lifts the hatching)"""
    ctx.save()
    light_pool(ctx, x, y - fh * 0.5, fh * 2.8, fh * 2.8, 0.85)
    sway = noise1(t * 3.1 + seed * 7.3, seed) * 0.35 + noise1(t * 11.0, seed + 3) * 0.12
    stretch = 1.0 + 0.18 * noise1(t * 7.7 + seed, seed + 5)
    tipx, tipy = x + sway * fh * 0.6, y - fh * 1.25 * stretch
    ctx.move_to(x, y + fh * 0.05)
    ctx.curve_to(x - fh * 0.42, y - fh * 0.1, x - fh * 0.30, y - fh * 0.62, tipx, tipy)
    ctx.curve_to(x + fh * 0.30, y - fh * 0.62, x + fh * 0.42, y - fh * 0.1, x, y + fh * 0.05)
    ctx.close_path()
    set_color(ctx, (1, 1, 1))
    ctx.fill_preserve()
    set_color(ctx, G(0.25), 0.8)
    ctx.set_line_width(max(0.8, fh * 0.05))
    ctx.stroke()
    # wick
    ctx.move_to(x, y + fh * 0.02)
    ctx.line_to(x + sway * fh * 0.1, y - fh * 0.22)
    set_color(ctx, G(0.1))
    ctx.set_line_width(max(0.8, fh * 0.06))
    ctx.stroke()
    ctx.restore()


# ============================================================ ground
def mud_ground(ctx, y0, y1, x0=-400, x1=W + 400, *, tone0=0.46, tone1=0.30, seed=11, t=0.0):
    """wet churned mud from the horizon y0 to the bottom y1: tone gradient + ruts (horizontal-ish dark strokes)
    + bright wet highlights"""
    ctx.save()
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.set_source(lin_grad(0, y0, 0, y1, [(0, G(tone0)), (0.35, G((tone0 + tone1) / 2 + 0.03)), (1, G(tone1))]))
    ctx.fill()
    n = 160
    for k in range(n):
        u = hash01(k, seed)
        yy = y0 + (y1 - y0) * u ** 1.6
        depth = (yy - y0) / max(1, (y1 - y0))
        xx = x0 + (x1 - x0) * hash01(k, seed + 1)
        L = (30 + 220 * hash01(k, seed + 2)) * (0.2 + depth)
        th = (1.5 + 6 * hash01(k, seed + 3)) * (0.2 + depth)
        tone = 0.06 + 0.18 * hash01(k, seed + 4)
        ctx.move_to(xx - L / 2, yy)
        ctx.curve_to(xx - L / 6, yy - th, xx + L / 6, yy + th, xx + L / 2, yy)
        set_color(ctx, G(tone), 0.55)
        ctx.set_line_width(th)
        ctx.stroke()
        if hash01(k, seed + 5) < 0.45:     # wet shine on the lip of the rut
            ctx.move_to(xx - L * 0.35, yy - th * 0.9)
            ctx.curve_to(xx - L / 8, yy - th * 1.6, xx + L / 8, yy - th * 0.4, xx + L * 0.3, yy - th * 0.9)
            set_color(ctx, (1, 1, 1), 0.55)
            ctx.set_line_width(max(1.0, th * 0.35))
            ctx.stroke()
    ctx.restore()


def puddle(ctx, cx, cy, rx, ry, t, *, seed=0, sky=0.9, ripples=6):
    """a puddle mirroring the bright sky, rain ripples expanding in it"""
    ctx.save()
    pts = []
    for k in range(18):
        a = k / 18 * 2 * math.pi
        rr = 1.0 + 0.22 * noise1(k * 0.9 + seed, seed)
        pts.append((cx + math.cos(a) * rx * rr, cy + math.sin(a) * ry * rr))
    smooth(ctx, pts, close=True)
    ctx.set_source(lin_grad(0, cy - ry, 0, cy + ry, [(0, G(sky * 0.75)), (0.6, G(sky)), (1, G(sky * 0.8))]))
    ctx.fill_preserve()
    set_color(ctx, G(0.08))
    ctx.set_line_width(max(1.0, ry * 0.08))
    ctx.stroke_preserve()
    ctx.clip()
    ctx.set_line_width(max(0.8, ry * 0.035))
    for k in range(ripples):
        period = 0.55 + 0.4 * hash01(k, seed + 3)
        ph = (t / period + hash01(k, seed + 4)) % 1.0
        cyc = int(t / period + hash01(k, seed + 4))
        px = cx + (hash01(cyc * 31 + k, seed + 5) - 0.5) * rx * 1.6
        py = cy + (hash01(cyc * 31 + k, seed + 6) - 0.5) * ry * 1.2
        rr = ph * rx * 0.35
        ellipse(ctx, px, py, rr, rr * ry / rx)
        set_color(ctx, G(0.15), 0.8 * (1 - ph))
        ctx.stroke()
    ctx.restore()


# ============================================================ rain (tone-adaptive)
def rain_layer(fc, draw_fn):
    """render a rain mask: draw_fn(ctx) paints streaks in white on a transparent canvas -> (h, w) coverage"""
    cv = Canvas(fc)
    draw_fn(cv.ctx)
    return cv.rgba()[..., 3]


def apply_rain(rgb, cov, k_dark=0.85, k_light=0.40):
    """engraved rain: over dark tone a streak lifts toward paper (white gaps in the hatching); over light tone it
    darkens a little (thin engraved lines) so the downpour reads everywhere."""
    L = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    up = cov * k_dark * (0.97 - L)
    dn = cov * k_light * np.clip((L - 0.62) / 0.38, 0, 1) * 0.55
    d = np.where(L < 0.66, up, -dn)
    return np.clip(rgb + d[..., None], 0, 1)


def fallback_rain(ctx, T, *, rect=(0, 0, W, H), density=1.0, slant=0.18, seed=0, alpha=1.0):
    """stand-in with t03_rain.draw_rain's signature (used until scenes/t03_rain lands)"""
    x0, y0, w, h = rect
    n = int(900 * density * (w * h) / (W * H))
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for k in range(n):
        layer = hash01(k, seed + 1)
        spd = 1500 + 1700 * layer
        L = 18 + 60 * layer
        x = x0 + (hash01(k, seed + 2) * (w + h * slant)) - h * slant * 0.5
        yy = (hash01(k, seed + 3) * (h + L) + T * spd) % (h + L) - L
        xx = x + (yy + y0) * slant * 0.0 + (T * spd * slant) % (w + 200) * 0 + yy * slant
        ctx.move_to(xx, y0 + yy)
        ctx.line_to(xx + L * slant, y0 + yy + L)
        set_color(ctx, (1, 1, 1), alpha * (0.35 + 0.65 * layer))
        ctx.set_line_width(0.8 + 1.8 * layer)
        ctx.stroke()
    ctx.restore()


def get_rain():
    try:
        from scenes import t03_rain
        return t03_rain.draw_rain, getattr(t03_rain, "splashes", None)
    except Exception:
        return fallback_rain, None

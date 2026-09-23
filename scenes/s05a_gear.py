"""s05a gear-up montage (90.0-91.2): five macro close-ups, ~6 frames each, rendered with true sub-frame
motion blur (the runner calls each shot at several shutter sub-times and averages).

Every shot fn(fc, st, u) -> float32 (h,w,3); u = seconds since the shot's first frame."""
import math

import cairo
import cv2
import numpy as np

from vx import (W, H, C, Canvas, col, set_color, mix, shade, rrect, circle, ellipse, poly, smooth, lin_grad,
                rad_grad, clamp, lerp, smoothstep, ease_in, ease_out, ease_in_out, ease_out_back, spring, hash01,
                shake, blur)
from vx.flag import Flag, draw_pole
from scenes.s05a_fx import (GOLD, GOLD_HOT, NVG, noise_layer, radial_img, grid_xy, speed_lines, streak, comp,
                            lin_grad, rad_grad)

FR = 1.0 / 24


def _amber_sweep(fc, img, u, dur, ang=-0.55, width=170, k=0.38, c=(1.0, 0.69, 0.19)):
    """a single amber-gold light band sweeping across the frame (echo of s04's klaxon)"""
    xx, yy = grid_xy(fc)
    p = lerp(-700, 2600, u / dur)
    d = xx * math.cos(ang) + yy * math.sin(ang) + 900 * math.sin(ang) * 0 - p * math.cos(ang)
    band = np.exp(-(d / width) ** 2)
    return img + band[..., None] * np.asarray(c, np.float32) * k


# =========================================================================================== VELCRO
def velcro(fc, st, u):
    """macro cross-section: the hook strap peels off the loop panel; backlight blazes through the gap and
    silhouettes the hooks, loops and the threads that stretch and snap at the peel front."""
    cv = Canvas(fc, bg=(0.02, 0.022, 0.03))
    ctx = cv.ctx
    p = ease_out(clamp((u + 0.06) / 0.30), 1.5)
    front = lerp(900, 1330, p)                   # peel front x (moves right)
    ang = math.radians(lerp(17, 25, p))          # opening angle of the lifted strap
    base = 770.0                                 # loop-panel surface
    thick = 380.0
    sx, sy = shake(u, 4.0, 60, 5)
    zz = 1.75 + 0.1 * p
    ctx.translate(1000 + sx, 700 + sy)
    ctx.rotate(-0.04)
    ctx.scale(zz, zz)
    ctx.translate(-(lerp(900, front, 0.25) - 250), -(base - 100))
    tn = math.tan(ang)

    def under(x):                                # strap underside y at x
        return base - max(0.0, front - x) * tn

    # --- the light behind the gap
    ctx.move_to(-400, base + 30)
    ctx.line_to(front + 40, base + 30)
    ctx.line_to(-400, under(-400) - 20)
    ctx.close_path()
    ctx.set_source(rad_grad(front - 460, base - 110, 10, 1400, [(0, "#FFE6CC"), (0.1, "#FFAE5C"), (0.3, "#D8682A"),
                                                                (0.65, "#6A3414"), (1, "#1E1008")]))
    ctx.fill()
    # --- loops rising from the panel (silhouettes); near the front some are caught by hooks and stretched
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for i in range(230):
        x = -400 + hash01(i, 11) * 2500
        w = 30 + 70 * hash01(i, 12)
        h = 40 + 110 * hash01(i, 13) ** 1.4
        lean = (hash01(i, 14) - 0.5) * 0.8
        top = None
        if front - 330 < x < front - 12 and hash01(i, 15) < 0.6:
            tgt = under(x) + 70
            stretch = base - tgt
            if stretch < 150 + 130 * hash01(i, 16):
                top = tgt
        ctx.set_line_width(5.5 + 3 * hash01(i, 17))
        c = 0.015 + 0.04 * hash01(i, 18)
        ctx.set_source_rgb(c, c * 1.05, c * 1.35)
        if top is not None:
            ctx.move_to(x - w / 2, base + 4)
            ctx.curve_to(x - w * 0.25, base - (base - top) * 0.55, x - 6, top + 12, x, top)
            ctx.curve_to(x + 6, top + 12, x + w * 0.25, base - (base - top) * 0.55, x + w / 2, base + 4)
        else:
            ctx.move_to(x - w / 2, base + 4)
            ctx.curve_to(x - w / 2 + lean * h, base - h * 1.25, x + w / 2 + lean * h, base - h * 1.25, x + w / 2,
                         base + 4)
        ctx.stroke()
    # snapped thread ends whipping up into the light
    for i in range(46):
        x0 = -400 + hash01(i, 21) * 2500
        if x0 > front - 280:
            continue
        age = (front - 280 - x0) / 800.0
        x = x0 - age * 160 * hash01(i, 22)
        y = base - 90 - age * 300 * (0.3 + hash01(i, 23))
        if y < under(x) + 20 or y > base - 20:
            continue
        L = 34 + 50 * hash01(i, 24)
        a = hash01(i, 25) * 6.28
        ctx.set_line_width(4.0)
        ctx.set_source_rgb(0.02, 0.02, 0.03)
        ctx.move_to(x, y)
        ctx.rel_curve_to(math.cos(a) * L, math.sin(a) * L, math.cos(a + 2) * L, math.sin(a + 2) * L,
                         math.cos(a + 3) * L * 0.5, math.sin(a + 3) * L * 0.5)
        ctx.stroke()

    def strap_body(c, x0, x1):
        c.rectangle(x0, -thick, x1 - x0, thick)
        c.set_source(lin_grad(0, -thick, 0, 0, [(0, "#2E323C"), (0.06, "#16181E"), (0.85, "#0B0C10"), (1, "#050507")]))
        c.fill()
        c.set_line_width(2.0)
        for k in range(1, 30):
            set_color(c, "#252932" if k % 2 else "#060708", 0.9)
            c.move_to(x0, -k * thick / 30)
            c.line_to(x1, -k * thick / 30)
            c.stroke()
        c.set_dash([22, 14])
        set_color(c, "#6A7282", 0.8)
        c.set_line_width(4)
        c.move_to(x0, -thick + 40)
        c.line_to(x1, -thick + 40)
        c.stroke()
        c.set_dash([])
        set_color(c, GOLD, 0.4)
        c.set_line_width(4)
        c.move_to(x0, -thick + 1)
        c.line_to(x1, -thick + 1)
        c.stroke()

    # --- the attached part of the strap (flat, right of the front)
    ctx.save()
    ctx.translate(0, base)
    strap_body(ctx, front - 2, 2400)
    ctx.restore()
    # --- the lifted part with its hooks (rotated about the peel front)
    ctx.save()
    ctx.translate(front, base)
    ctx.rotate(ang)
    for i in range(120):
        lx = -2800 + hash01(i, 31) * 2800
        if lx > -30:
            continue
        L = 55 + 55 * hash01(i, 32)
        cw = 16 + 12 * hash01(i, 33)
        d = 1 if hash01(i, 34) > 0.5 else -1
        ctx.set_line_width(6.5)
        ctx.set_source_rgb(0.02, 0.022, 0.035)
        ctx.move_to(lx, 0)
        ctx.line_to(lx, L)
        ctx.curve_to(lx, L + cw * 1.3, lx + d * cw * 1.6, L + cw * 1.3, lx + d * cw * 1.5, L + cw * 0.15)
        ctx.stroke()
    strap_body(ctx, -2800, 0)
    ctx.set_line_width(4)
    ctx.move_to(-2800, -1.5)
    ctx.line_to(-4, -1.5)
    ctx.set_source(lin_grad(-1400, 0, 0, 0, [(0, (*GOLD, 0.2)), (0.85, (*GOLD_HOT, 0.8)), (1, (1, 1, 1, 0.9))]))
    ctx.stroke()
    ctx.restore()
    # --- the loop panel / vest below
    ctx.rectangle(-400, base, 2800, 700)
    ctx.set_source(lin_grad(0, base, 0, 1100, [(0, "#101217"), (0.4, "#1A1D24"), (1, "#0A0B0E")]))
    ctx.fill()
    ctx.set_line_width(2.0)
    for k in range(1, 24):
        set_color(ctx, "#20242C" if k % 2 else "#07080A", 0.9)
        ctx.move_to(-400, base + 10 + k * 14)
        ctx.line_to(2400, base + 10 + k * 14)
        ctx.stroke()
    ctx.set_dash([22, 14])
    set_color(ctx, "#5E6676", 0.7)
    ctx.set_line_width(4)
    ctx.move_to(-400, base + 48)
    ctx.line_to(2400, base + 48)
    ctx.stroke()
    ctx.set_dash([])
    ctx.rectangle(-400, base, front + 400, 12)
    ctx.set_source(lin_grad(-400, 0, front, 0, [(0, (*GOLD, 0.03)), (0.75, (*GOLD, 0.22)), (1, (*GOLD_HOT, 0.0))]))
    ctx.fill()
    # --- lint lit in the beam
    for i in range(50):
        age = u + hash01(i, 41) * 0.3
        x = front - 400 + hash01(i, 42) * 700 - age * (300 + 700 * hash01(i, 43))
        y = base - 60 - age * (200 + 500 * hash01(i, 44))
        inside = under(x) + 6 < y < base
        set_color(ctx, GOLD_HOT if inside else (0.3, 0.32, 0.37), 0.8 if inside else 0.5)
        circle(ctx, x, y, 2 + 3.5 * hash01(i, 45))
        ctx.fill()
    img = cv.rgb()
    img += radial_img(fc, 560, 600, 480, GOLD, 2.0) * 0.25
    img += streak(img, fc, 0.9, 420, 0.3)
    return img


# =========================================================================================== TACKS


def _cloth_verts(u, impacts, x0, y0, cw, chh, tack, nx=44, ny=36):
    """flat cloth lying on the bench (top view) with soft wrinkles; each hammer impact sends a ripple ring
    through it from the tack. z = height above the bench (toward camera)."""
    xs = np.linspace(0, cw, nx)[None, :]
    ys = np.linspace(0, chh, ny)[:, None]
    X = x0 + xs + 0 * ys
    Y = y0 + ys + 0 * xs
    Z = (14 * np.sin(xs * 0.011 + ys * 0.004 + 0.7) * (xs / cw)
         + 9 * np.sin(xs * 0.006 - ys * 0.013 + 2.0) + 6 * np.sin(ys * 0.021 + xs * 0.002)
         + 22 * np.exp(-((ys - chh * 0.7) / 90.0) ** 2) * np.clip(xs / cw * 1.4, 0, 1))
    r = np.hypot(X - tack[0], Y - tack[1])
    for ti in impacts:
        a = u - ti
        if a <= 0:
            continue
        front = 1400 * a
        Z = Z + 26 * np.exp(-a * 7) * np.sin((r - front) * 0.03) * np.exp(-((r - front) / 170.0) ** 2)
        X = X + 5 * np.exp(-a * 9) * np.exp(-((r - front) / 170.0) ** 2) * (X - tack[0]) / (r + 1)
    return np.stack([X, Y, Z], -1)


def tacks(fc, st, u):
    """top-down on the armoury bench: the flag lies flat beside its pole; a hammer drives a carpet tack through
    the hoist into the wood (twice). The impacts ripple through the cloth; the hammer's shadow sells height."""
    from vx.flag import draw_cloth
    s1, s2 = 1 * FR, 4 * FR                # hammer strikes (shot frames 1 and 4)
    last = s2 if u >= s2 else (s1 if u >= s1 else None)
    imp = (u - last) if last is not None else 1.0
    sx, sy = shake(u, 12 * math.exp(-imp * 28) if last is not None else 0, 45, 7)
    cv = Canvas(fc, bg=(0.03, 0.032, 0.045))
    ctx = cv.ctx
    ctx.translate(960 + sx, 560 + sy)
    ctx.rotate(-0.42)
    zz = 1.0 + 0.05 * clamp(u / 0.3)
    ctx.scale(zz, zz)
    # bench: dark steel-blue top under a warm lamp pool
    ctx.rectangle(-1600, -1400, 3200, 2800)
    ctx.set_source(rad_grad(-120, -60, 20, 1500, [(0, "#3A3440"), (0.35, "#1C1C26"), (1, "#07080C")]))
    ctx.fill()
    set_color(ctx, (0, 0, 0), 0.35)
    ctx.set_line_width(3)
    for k in range(-8, 9):
        ctx.move_to(-1600, k * 170 + 40)
        ctx.line_to(1600, k * 170 + 40)
        ctx.stroke()
    px = -300.0                            # pole centre line (runs along local y)
    pw = 96.0
    tack = (px - 30.0, 30.0)               # tacks go into the bare wood beside the hoist: nothing ever on the cloth
    # cast shadows of pole + cloth on the bench (lamp upper-left -> shadows fall down-right)
    set_color(ctx, (0, 0, 0), 0.45)
    ctx.rectangle(px - pw / 2 + 22, -1400, pw, 2800)
    ctx.fill()
    ctx.rectangle(px + 22, -560 + 22, 1400, 1150)
    ctx.fill()
    draw_pole(ctx, px, 1400, px, -1400, pw, light=(-0.5, -0.5, 0.7))
    V = _cloth_verts(u, (s1, s2), px, -560, 1400, 1150, tack)
    draw_cloth(ctx, V, light=(-0.7, -0.45, 0.5))
    # tacks: one already flush, the hero tack being driven
    depth = 0.0 if u < s1 else (0.55 if u < s2 else 1.0)
    hgt = 34 * (1 - depth)
    ctx.save()                             # everything below is clipped to the bench/pole side of the hoist
    ctx.rectangle(-1700, -1500, (px - 3) + 1700, 3000)
    ctx.clip()
    for (tx, ty, hh) in ((px - 30.0, -400.0, 0.0), (tack[0], tack[1], hgt)):
        set_color(ctx, (0, 0, 0), 0.5)
        circle(ctx, tx + 6 + hh * 0.6, ty + 6 + hh * 0.9, 21)
        ctx.fill()
        if hh > 1:   # shank visible between head and cloth
            set_color(ctx, "#5E6573")
            poly(ctx, [(tx - 4, ty), (tx + 4, ty), (tx + 4 - hh * 0.25, ty - hh * 0.4), (tx - 4 - hh * 0.25, ty - hh * 0.4)])
            ctx.fill()
        hx_, hy_ = tx - hh * 0.25, ty - hh * 0.4
        circle(ctx, hx_, hy_, 21 + hh * 0.08)
        ctx.set_source(rad_grad(hx_ - 7, hy_ - 7, 1, 24, [(0, "#F4F7FC"), (0.45, "#A9B1BF"), (1, "#3F4550")]))
        ctx.fill()

    # hammer: height above the bench (0 = contact); scale + shadow offset from height
    def height(uu):
        if uu < s1:
            return 380 * (1 - ease_in(clamp((uu + 1.6 * FR) / (1.6 * FR + s1)), 2.4))
        if uu < s2:
            k = (uu - s1) / (s2 - s1)
            return 300 * math.sin(math.pi * clamp(k)) ** 0.85
        return 330 * ease_out(clamp((uu - s2) / 0.14))
    h = height(u)
    sc = 0.78 * (1.0 + h * 0.0016)
    hx, hy = tack[0] - hgt * 0.25 - h * 0.5, tack[1] - hgt * 0.4 - h * 0.35

    def hammer_shape(c, k):
        c.save()
        c.translate(hx, hy)
        c.rotate(math.pi - 0.55)
        c.scale(k, k)
        rrect(c, 20, -34, 900, 68, 30)          # handle toward lower right
        rrect(c, -58, -150, 116, 300, 16)       # head (T-bar), striking face under the camera's view
        c.move_to(-40, -150)                    # claw
        c.curve_to(-50, -230, -20, -300, 30, -330)
        c.line_to(40, -300)
        c.curve_to(10, -270, 20, -220, 30, -150)
        c.close_path()
        c.restore()

    ctx.save()
    ctx.translate(-h * 0.2, h * 1.1)
    hammer_shape(ctx, 0.78)
    set_color(ctx, (0, 0, 0), 0.55 * clamp(1 - h / 900))
    ctx.fill()
    ctx.restore()
    ctx.save()
    hammer_shape(ctx, sc)
    ctx.set_source(lin_grad(hx - 150, hy - 150, hx + 150, hy + 150, [(0, "#9AA3B2"), (0.3, "#4A505C"), (1, "#15171C")]))
    ctx.fill()
    ctx.restore()
    ctx.save()
    ctx.translate(hx, hy)
    ctx.rotate(math.pi - 0.55)
    ctx.scale(sc, sc)
    rrect(ctx, -58, -150, 116, 300, 16)
    ctx.set_source(lin_grad(-58, 0, 58, 0, [(0, "#C9D0DB"), (0.35, "#8A93A2"), (1, "#2A2E36")]))
    ctx.fill()
    rrect(ctx, 20, -34, 900, 68, 30)
    ctx.set_source(lin_grad(0, -34, 0, 34, [(0, "#4A4E58"), (0.3, "#1A1B20"), (1, "#060607")]))
    ctx.fill()
    set_color(ctx, (1, 1, 1), 0.35)
    ctx.set_line_width(3)
    ctx.move_to(-52, -140)
    ctx.line_to(-52, 140)
    ctx.stroke()
    ctx.restore()
    # impact: white-hot spark star + wood/cloth dust
    if last is not None and imp < 0.12:
        e = math.exp(-imp * 30)
        speed_lines(ctx, hx, hy, 26, 41 + int(last * 100), 120, 420 * (0.4 + imp * 7), (1, 0.97, 0.9), 0.75 * e, 3)
        ctx.set_source(rad_grad(hx, hy, 0, 70, [(0, (1, 1, 1, 0.9 * e)), (0.35, (1, 0.97, 0.9, 0.45 * e)),
                                                (1, (1, 0.9, 0.8, 0))]))
        circle(ctx, hx, hy, 70)
        ctx.fill()
        for i in range(30):
            a = hash01(i, 51 + int(last * 99)) * math.tau
            r = (140 + 420 * hash01(i, 52)) * (1 - math.exp(-imp * 20))
            set_color(ctx, mix("pole", "#FFFFFF", hash01(i, 53) * 0.5), 0.9 * e ** 0.4)
            circle(ctx, hx + math.cos(a) * r, hy + math.sin(a) * r, 2 + 3 * hash01(i, 54))
            ctx.fill()
    ctx.restore()
    img = cv.rgb()
    img = _amber_sweep(fc, img, u - 0.04, 0.3, ang=-0.4, width=160, k=0.16)
    return img


# =========================================================================================== POLES
def _tele_pole(c, u, clicks, seg, wd, dl=0.0, spark=False, hand=False):
    """one telescoping wooden pole along local +x; sections shoot out on each click (with overshoot)"""
    ext = []
    for k, tc in enumerate(clicks):
        tc = tc + dl
        e = ease_out(clamp((u - tc + FR) / (1.0 * FR)), 3)
        if u > tc:
            e += 0.07 * math.sin(clamp((u - tc) / 0.09) * math.pi) * math.exp(-(u - tc) * 18)
        ext.append(e)
    # section k (k>=1) slides out of section k-1 by ext[k-1]*seg
    x = 0.0
    xs = [(0.0, seg)]
    for k in range(3):
        L = seg * 0.92 * ext[k]
        if L < 2:
            break
        xs.append((xs[-1][1] - 40, xs[-1][1] - 40 + L + 40))
    for k, (a, b) in reversed(list(enumerate(xs))):
        w = wd * (1 - 0.12 * k)
        draw_pole(c, a, 0, b, 0, w)
        c.save()
        c.rectangle(a, -w / 2, b - a, w)
        c.clip()
        for g in range(6):
            yy = -w / 2 + (g + 0.5) * w / 6
            set_color(c, "pole_shadow", 0.3)
            c.set_line_width(max(0.8, w * 0.025))
            c.move_to(a, yy)
            for xx in range(int(a), int(b) + 40, 40):
                c.line_to(xx, yy + w * 0.035 * math.sin(xx * 0.013 + g * 2.1 + k * 3))
            c.stroke()
        c.set_source(lin_grad(0, -w / 2, 0, w / 2, [(0, (*GOLD_HOT, 0.75)), (0.12, (*GOLD, 0.0)), (0.7, (0, 0, 0, 0.0)),
                                                    (1, (0, 0, 0, 0.55))]))
        c.rectangle(a, -w / 2, b - a, w)
        c.fill()
        # the lip where the next section emerges: a dark ring
        c.rectangle(b - 10, -w / 2, 10, w)
        set_color(c, (0.05, 0.03, 0.02), 0.55)
        c.fill()
        c.restore()
    if spark:
        lastk = max([k for k, tc in enumerate(clicks) if u >= tc + dl], default=None)
        if lastk is not None and u - clicks[lastk] - dl < 0.1:
            imp = u - clicks[lastk] - dl
            e = math.exp(-imp * 28)
            jx = xs[min(lastk, len(xs) - 1)][1] - 40
            speed_lines(c, jx, 0, 30, 70 + lastk, wd * 0.45, wd * 1.9, (1, 0.96, 0.88), 0.9 * e, 3.5)
            c.set_source(rad_grad(jx, 0, 0, wd * 0.45, [(0, (1, 1, 1, 0.9 * e)), (0.4, (1, 0.92, 0.75, 0.45 * e)),
                                                          (1, (1, 0.8, 0.5, 0))]))
            circle(c, jx, 0, wd * 0.45)
            c.fill()
            for i in range(18):   # wood dust flicked off the joint
                a = hash01(i, 90 + lastk) * math.tau
                r = wd * (0.4 + 3.0 * hash01(i, 91 + lastk)) * (1 - math.exp(-imp * 25))
                set_color(c, mix("pole", GOLD_HOT, hash01(i, 92)), 0.9 * e ** 0.5)
                circle(c, jx + math.cos(a) * r, math.sin(a) * r * 0.7, 2 + 3 * hash01(i, 93))
                c.fill()
    if hand:   # gloved fist wrapped around the base section
        hx = seg * 0.35
        for k in range(4):
            fx = hx + k * wd * 0.52
            rrect(c, fx, -wd * 0.78, wd * 0.5, wd * 1.56, wd * 0.24)
            c.set_source(lin_grad(fx, -wd, fx + wd * 0.2, wd, [(0, "#50555F"), (0.25, "#24262C"), (1, "#060607")]))
            c.fill()
            set_color(c, GOLD_HOT, 0.7)
            c.set_line_width(3)
            c.move_to(fx + wd * 0.06, -wd * 0.74)
            c.curve_to(fx + wd * 0.15, -wd * 0.82, fx + wd * 0.35, -wd * 0.82, fx + wd * 0.44, -wd * 0.74)
            c.stroke()
        rrect(c, hx - 170, -wd * 0.72, 200, wd * 1.44, wd * 0.3)
        c.set_source(lin_grad(0, -wd * 0.72, 0, wd * 0.72, [(0, "#2C2F37"), (0.3, "#101115"), (1, "#050506")]))
        c.fill()
        rrect(c, hx - 60, -wd * 0.66, wd * 1.5, wd * 0.36, wd * 0.18)
        c.set_source(lin_grad(0, -wd * 0.66, 0, -wd * 0.3, [(0, "#5A606B"), (0.5, "#1E2026"), (1, "#0B0B0E")]))
        c.fill()


def poles(fc, st, u):
    clicks = [0.0, 2 * FR, 4 * FR]
    lastc = max([cc for cc in clicks if u >= cc], default=None)
    imp = u - lastc if lastc is not None else 1.0
    sx, sy = shake(u, 8 * math.exp(-imp * 35), 50, 11)
    # backdrop: blue-black, gold haze from the upper right, a few soft bokeh
    bg = Canvas(fc)
    b = bg.ctx
    b.rectangle(0, 0, 1920, 1080)
    b.set_source(rad_grad(1650, 120, 20, 1700, [(0, "#8A5A24"), (0.18, "#3A2618"), (0.5, "#12121C"), (1, "#040509")]))
    b.fill()
    for i in range(9):
        x = 900 + hash01(i, 61) * 1000
        y = 120 + hash01(i, 62) * 600
        r = 18 + 40 * hash01(i, 63)
        cc = GOLD if hash01(i, 64) > 0.25 else (0.5, 0.65, 1.0)
        b.set_source(rad_grad(x, y, 0, r, [(0, (*cc, 0.12)), (0.8, (*cc, 0.09)), (1, (*cc, 0))]))
        circle(b, x, y, r)
        b.fill()
    img = blur(bg.rgb(), 3 * fc.s)
    ang = -0.36
    for pi_, (x0, y0, wd, seg, dl, bl, sp) in enumerate([
            (-60, 470, 40, 380, 1.4 * FR, 5.0, False),
            (120, 760, 66, 430, 0.7 * FR, 2.5, False),
            (-260, 1200, 170, 720, 0.0, 0.0, True)]):
        L = Canvas(fc)
        c = L.ctx
        c.translate(x0 + sx, y0 + sy)
        c.rotate(ang)
        _tele_pole(c, u, clicks, seg, wd, dl, spark=sp, hand=(pi_ == 2))
        rgba = L.rgba()
        if bl > 0:
            rgba = blur(rgba, bl * fc.s) * np.float32(0.85)
        img = comp(img, rgba)
    img += radial_img(fc, 1750, 60, 500, GOLD, 2.0) * 0.22
    img += streak(img, fc, 0.85, 300, 0.2)
    return img


# =========================================================================================== NVG
def _rotx(p, h, th):
    c, s = math.cos(th), math.sin(th)
    y, z = p[1] - h[1], p[2] - h[2]
    return (p[0], h[1] + y * c - z * s, h[2] + y * s + z * c)


def _roty(p, th):
    c, s = math.cos(th), math.sin(th)
    return (p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c)


def nvg(fc, st, u):
    """near-frontal head in deep shadow, backlit gold; quad-tube NVG swings down on its hinge and the four
    lenses ignite phosphor green (the only colour in the frame)."""
    lock = 2 * FR                                    # shot frame 2 -> global 90.875
    imp = u - lock
    sx, sy = shake(u, 7 * math.exp(-imp * 30) if imp > 0 else 0, 50, 17)
    push = 1.0 + 0.06 * clamp((u + 0.04) / 0.25)
    cx, cy = 960 + sx, 520 + sy
    R = 250 * push
    yaw = -0.22
    if u < lock:
        th = -1.8 * (1 - ease_in(clamp((u + FR) / (lock + FR)), 2.2))
    else:
        th = 0.12 * math.exp(-imp * 22) * math.sin(imp * 55)
    on = 0.0
    if imp > 0:
        fl = [1, 0.2, 1, 0.55, 1]
        on = fl[min(len(fl) - 1, int(imp / 0.017))] * smoothstep(0, 0.008, imp)
    F, D = 5.0, 6.0

    def P(p):
        q = _roty(p, yaw)
        k = F / (D - q[2]) * D / F
        return (cx + q[0] * k * R, cy + q[1] * k * R, q[2])

    cv = Canvas(fc)
    ctx = cv.ctx
    # --- backdrop: backlight haze behind the head (upper right), dim tunnel ribs
    ctx.rectangle(-40, -40, 2000, 1160)
    ctx.set_source(rad_grad(cx + 330, cy - 250, 20, 1500, [(0, "#FFCE8A"), (0.08, "#D98C3A"), (0.3, "#4A2C18"),
                                                           (0.65, "#12121C"), (1, "#05060A")]))
    ctx.fill()
    s = R / 250.0

    def T(x, y):
        return (cx + x * s, cy + y * s)

    def path(pts):
        smooth(ctx, [T(*p) for p in pts], close=True)

    # --- shoulders: dark vest over the ochre robe
    path([(-760, 900), (-520, 470), (-250, 400), (0, 420), (240, 400), (520, 460), (760, 900)])
    ctx.set_source(lin_grad(*T(500, 400), *T(-400, 800), [(0, "#2A2C34"), (0.3, "#111217"), (1, "#050506")]))
    ctx.fill()
    path([(-250, 405), (-120, 470), (0, 480), (120, 470), (250, 405), (140, 520), (0, 560), (-140, 520)])
    set_color(ctx, mix("robe_shadow", (0, 0, 0), 0.45))
    ctx.fill()
    # neck
    path([(-120, 250), (110, 250), (130, 430), (0, 470), (-130, 430)])
    set_color(ctx, "#0C0908")
    ctx.fill()
    # --- face (near-frontal, in deep shadow)
    face = [(-205, -60), (-218, 90), (-195, 230), (-120, 330), (-30, 365), (60, 345), (150, 250), (185, 100),
            (178, -60)]
    path(face)
    ctx.set_source(lin_grad(*T(200, -50), *T(-200, 300), [(0, "#2A1A13"), (0.3, "#130D0A"), (1, "#070505")]))
    ctx.fill()
    # nose shadow + mouth (barely readable until the lenses light them)
    set_color(ctx, (0, 0, 0), 0.55)
    path([(-40, 60), (-10, 140), (10, 175), (-40, 185), (-75, 172), (-60, 130)])
    ctx.fill()
    ctx.set_line_width(9 * s)
    set_color(ctx, (0.01, 0.005, 0.005), 0.9)
    ctx.move_to(*T(-95, 250))
    ctx.curve_to(*T(-55, 242), *T(0, 244), *T(35, 250))
    ctx.stroke()
    # phosphor-green light modelling the face from just above (cheekbones, nose, lips, chin)
    if on > 0:
        for (hx_, hy_, rx_, ry_, a_) in ((-110, 125, 120, 70, 0.16), (90, 125, 100, 60, 0.1), (-22, 150, 40, 50, 0.2),
                                         (-35, 225, 90, 30, 0.12), (-30, 305, 100, 45, 0.08)):
            ctx.save()
            ctx.translate(*T(hx_, hy_))
            ctx.scale(rx_ * s, ry_ * s)
            ctx.set_source(rad_grad(0, 0, 0, 1, [(0, (*NVG, a_ * on)), (0.5, (*NVG, a_ * on * 0.4)), (1, (*NVG, 0))]))
            circle(ctx, 0, 0, 1)
            ctx.fill()
            ctx.restore()
    # chin strap
    set_color(ctx, "#0A0B0E")
    ctx.set_line_width(26 * s)
    ctx.move_to(*T(-205, 20))
    ctx.curve_to(*T(-200, 200), *T(-120, 330), *T(-30, 355))
    ctx.curve_to(*T(60, 340), *T(150, 250), *T(178, 20))
    ctx.stroke()
    # ear cups (comms)
    for side in (-1, 1):
        ex = -222 if side < 0 else 196
        ellipse(ctx, *T(ex, 60), 58 * s, 105 * s)
        ctx.set_source(lin_grad(*T(ex + 50, -40), *T(ex - 50, 160), [(0, "#3C4049"), (0.4, "#16181D"), (1, "#060607")]))
        ctx.fill()
    # --- helmet: high-cut dome with rails and a front shroud
    path([(-265, 20), (-300, -160), (-230, -330), (-60, -410), (120, -405), (270, -320), (320, -150), (280, 20),
          (215, -10), (200, -70), (100, -95), (-40, -100), (-180, -85), (-215, -10)])
    ctx.set_source(lin_grad(*T(260, -380), *T(-240, 0), [(0, "#555C69"), (0.28, "#2A2E36"), (0.7, "#14161B"),
                                                          (1, "#0B0C0F")]))
    ctx.fill()
    for side in (-1, 1):   # side rails
        x0 = -285 if side < 0 else 250
        rrect(ctx, *T(x0, -170), 36 * s, 150 * s, 10 * s)
        set_color(ctx, "#0E0F13")
        ctx.fill()
    # rim light on the helmet crown and right side (backlight upper-right)
    ctx.set_line_width(7 * s)
    ctx.move_to(*T(-120, -402))
    ctx.curve_to(*T(40, -415), *T(200, -380), *T(275, -310))
    ctx.curve_to(*T(320, -240), *T(330, -120), *T(282, 18))
    ctx.set_source(lin_grad(*T(-120, -400), *T(280, 0), [(0, (*GOLD, 0.0)), (0.3, (*GOLD_HOT, 0.9)), (1, (*GOLD, 0.6))]))
    ctx.stroke()
    ctx.set_line_width(5 * s)
    set_color(ctx, GOLD, 0.55)
    ctx.move_to(*T(185, 110))
    ctx.curve_to(*T(175, 200), *T(140, 270), *T(70, 340))
    ctx.stroke()
    ctx.move_to(*T(250, 405))
    ctx.curve_to(*T(420, 420), *T(560, 520), *T(700, 800))
    ctx.stroke()
    # front shroud plate
    rrect(ctx, *T(-70, -300), 140 * s, 110 * s, 18 * s)
    ctx.set_source(lin_grad(*T(-70, -300), *T(70, -190), [(0, "#4A505B"), (1, "#121318")]))
    ctx.fill()
    # --- quad-tube NVG in 3D, hinged at the shroud
    hinge = (0.0, -1.0, 0.72)
    tubes = []
    for dx, yw in ((-0.66, -0.16), (-0.24, -0.03), (0.24, 0.03), (0.66, 0.16)):
        a0 = (dx, 0.06, 0.95)
        dv = (math.sin(yw), 0.0, math.cos(yw))
        Ln = 0.55 if abs(dx) < 0.4 else 0.45
        tubes.append((a0, (a0[0] + dv[0] * Ln, a0[1], a0[2] + dv[2] * Ln), 0.19 if abs(dx) < 0.4 else 0.17))
    arm = [(-0.14, -1.0, 0.72), (0.14, -1.0, 0.72), (0.14, 0.0, 1.0), (-0.14, 0.0, 1.0)]
    bridge = [(-0.85, -0.1, 0.95), (0.85, -0.1, 0.95), (0.85, 0.22, 1.02), (-0.85, 0.22, 1.02)]
    for quad, colr in ((arm, "#1E2127"), (bridge, "#16181D")):
        poly(ctx, [P(_rotx(p, hinge, th))[:2] for p in quad])
        set_color(ctx, colr)
        ctx.fill()

    def ring(cen, dv, r, n=32):
        ax = np.array(dv)
        e1 = np.cross(ax, np.array((0.0, 1.0, 0.0)))
        e1 /= np.linalg.norm(e1) + 1e-9
        e2 = np.cross(ax, e1)
        return [tuple(np.array(cen) + r * (math.cos(a) * e1 + math.sin(a) * e2))
                for a in np.linspace(0, math.tau, n, endpoint=False)]

    lenses = []
    for a0, a1, r in sorted(tubes, key=lambda tb: _roty(_rotx(tb[1], hinge, th), yaw)[2]):
        c0, c1 = _rotx(a0, hinge, th), _rotx(a1, hinge, th)
        dv = np.array(c1) - np.array(c0)
        dv /= np.linalg.norm(dv)
        r0 = [P(p)[:2] for p in ring(c0, dv, r * 1.06)]
        r1 = [P(p)[:2] for p in ring(c1, dv, r)]
        hull = cv2.convexHull(np.array(r0 + r1, np.float32)).reshape(-1, 2)
        poly(ctx, [tuple(p) for p in hull])
        e0 = P(c0)
        ctx.set_source(lin_grad(e0[0] + 90, e0[1] - 90, e0[0] - 90, e0[1] + 90,
                                [(0, "#4E5460"), (0.35, "#1E2127"), (1, "#08090B")]))
        ctx.fill()
        poly(ctx, r1)
        set_color(ctx, "#08090B")
        ctx.fill()
        # rim light on the upper-right lip of each tube
        set_color(ctx, GOLD_HOT, 0.75)
        ctx.set_line_width(3.5 * s)
        n1 = len(r1)
        arc_pts = [r1[(k) % n1] for k in range(int(n1 * 0.6), int(n1 * 0.95))]
        ctx.move_to(*arc_pts[0])
        for p_ in arc_pts[1:]:
            ctx.line_to(*p_)
        ctx.stroke()
        lr = [P(p)[:2] for p in ring(c1, dv, r * 0.8)]
        lenses.append(lr)
        lcx = sum(p_[0] for p_ in lr) / len(lr)
        lcy = sum(p_[1] for p_ in lr) / len(lr)
        poly(ctx, lr)
        g = rad_grad(lcx - 10 * s, lcy - 10 * s, 1, r * R * 0.95,
                     [(0, mix((0.10, 0.14, 0.13), (0.93, 1.0, 0.9), on)), (0.35, mix((0.05, 0.08, 0.08), NVG, on)),
                      (1, mix((0.02, 0.03, 0.03), (0.08, 0.42, 0.12), on))])
        ctx.set_source(g)
        ctx.fill()
        # glass glint
        set_color(ctx, (0.8, 0.9, 1.0), 0.25 + 0.3 * (1 - on))
        ellipse(ctx, lcx - r * R * 0.3, lcy - r * R * 0.35, r * R * 0.18, r * R * 0.1, -0.6)
        ctx.fill()
    img = cv.rgb()
    if on > 0:
        m = Canvas(fc)
        for lr in lenses:
            poly(m.ctx, lr)
            set_color(m.ctx, NVG, 1)
            m.ctx.fill()
        ml = m.rgba()[..., :3]
        img += blur(ml, 14 * fc.s) * 0.9 * on + blur(ml, 60 * fc.s) * 0.7 * on
        # green spill on nose/lips/cheeks under the tubes
        spill = radial_img(fc, cx - 20 * s, cy + 150 * s, 230 * s, NVG, 2.4)
        img += spill * 0.12 * on
        img += streak(np.maximum(ml, 0) * 1.2, fc, 0.5, 520, 0.35 * on, tint=NVG)
    img += streak(img, fc, 0.9, 300, 0.2)
    return img


# =========================================================================================== BOOTS
def _boot(ctx, x, y, s, rim=1.0):
    """profile tactical boot (toe to the right), (x, y) = heel bottom; s = scale"""
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    # lug sole
    ctx.move_to(-10, 0)
    ctx.line_to(430, 0)
    ctx.curve_to(470, 0, 480, -30, 470, -48)
    ctx.line_to(-15, -48)
    ctx.close_path()
    set_color(ctx, "#0B0B0D")
    ctx.fill()
    for k in range(9):
        set_color(ctx, "#1E1F24")
        ctx.rectangle(10 + k * 48, -8, 26, 8)
        ctx.fill()
    # upper
    ctx.move_to(-5, -48)
    ctx.line_to(-12, -330)
    ctx.curve_to(40, -345, 110, -345, 150, -330)
    ctx.curve_to(170, -230, 220, -170, 330, -140)
    ctx.curve_to(430, -115, 470, -80, 465, -48)
    ctx.close_path()
    ctx.set_source(lin_grad(0, -340, 300, 0, [(0, "#2B2C33"), (0.5, "#18191E"), (1, "#0C0C0F")]))
    ctx.fill()
    # laces
    set_color(ctx, "#4A4E58")
    ctx.set_line_width(7)
    for k in range(6):
        t = k / 5
        px, py = lerp(150, 300, t) + 5, lerp(-320, -150, t ** 0.8)
        ctx.move_to(px - 26, py - 8)
        ctx.line_to(px + 26, py + 8)
        ctx.stroke()
    # rim light along the back/top edge
    ctx.set_line_width(6)
    set_color(ctx, GOLD_HOT, 0.9 * rim)
    ctx.move_to(-8, -60)
    ctx.line_to(-13, -328)
    ctx.curve_to(40, -343, 100, -343, 148, -330)
    ctx.stroke()
    ctx.restore()


def boots(fc, st, u):
    stomp = 1 * FR                                   # local frame 25 -> 91.042
    cv = Canvas(fc)
    ctx = cv.ctx
    imp = u - stomp
    sx, sy = shake(u, 12 * math.exp(-imp * 26) if imp > 0 else 0, 45, 23)
    vpx, vpy = 1180, 560
    hor = 560
    # --- background: tunnel end blazing gold, basalt walls
    bg = Canvas(fc)
    b = bg.ctx
    b.translate(sx * 0.3, sy * 0.3)
    b.rectangle(-40, -40, 2000, 1160)
    b.set_source(rad_grad(vpx, vpy - 60, 10, 1200, [(0, "#FFD9A0"), (0.06, "#E09A40"), (0.25, "#4A3020"), (0.6, "#14131C"),
                                                     (1, "#07080C")]))
    b.fill()
    # far legs of the team walking (silhouettes) against the light
    for i in range(4):
        lx = vpx - 420 + i * 230 + 30 * math.sin(i * 2.1)
        ph = u * 3 + i * 1.3
        for leg in (0, 1):
            sw = math.sin(ph + leg * math.pi) * 22
            b.move_to(lx + leg * 40, 150)
            b.line_to(lx + leg * 40 + sw, vpy + 20)
            b.set_line_width(38)
            set_color(b, "#0B0B10")
            b.stroke()
            b.rectangle(lx + leg * 40 + sw - 22, vpy + 8, 60, 22)
            b.fill()
        b.rectangle(lx - 40, 100, 140, 200)
        set_color(b, "#0E0E14")
        b.fill()
    bimg = blur(bg.rgb(), 5 * fc.s)
    # --- floor: wet stone plates in perspective
    fl = Canvas(fc)
    f = fl.ctx
    f.translate(sx, sy)
    f.move_to(-40, hor)
    f.line_to(1960, hor)
    f.line_to(1960, 1120)
    f.line_to(-40, 1120)
    f.close_path()
    f.set_source(lin_grad(0, hor, 0, 1080, [(0, "#26201C"), (0.25, "#15141A"), (1, "#07070A")]))
    f.fill()
    set_color(f, (0, 0, 0), 0.6)
    f.set_line_width(3)
    for k in range(-14, 15):
        f.move_to(vpx, vpy)
        f.line_to(vpx + k * 260, 1200)
        f.stroke()
    for k in range(1, 12):
        yy = hor + (1080 - hor) * (k / 11) ** 2.2
        f.move_to(-40, yy)
        f.line_to(1960, yy)
        f.stroke()
    fimg = fl.rgba()
    img = comp(bimg, fimg)
    # glossy reflection of the light on the wet floor
    xx, yy = grid_xy(fc)
    dy_ = np.maximum(yy - hor, 0)
    refl = np.exp(-((xx - vpx) / (70 + dy_ * 0.35)) ** 2) * np.clip(dy_ / 30, 0, 1) * np.exp(-dy_ / 420)
    img += refl[..., None] * np.array(GOLD, np.float32) * 0.9
    # --- the hero boot slamming down (left), robe hem above
    down = ease_in(clamp((u + 1.2 * FR) / (1.2 * FR + stomp)), 2)
    by = lerp(740, 930, down) + (6 * math.sin(imp * 80) * math.exp(-imp * 30) if imp > 0 else 0)
    bc = Canvas(fc)
    c = bc.ctx
    c.translate(sx, sy)
    # dust ring (backlit)
    if imp > 0:
        for i in range(120):
            side = 1 if hash01(i, 81) > 0.5 else -1
            sp = 300 + 1100 * hash01(i, 82)
            a = (0.05 + 0.5 * hash01(i, 83))
            dx = side * sp * imp * math.cos(a) * (1.0 if side > 0 else 0.8)
            dy = -sp * imp * math.sin(a) * 0.9 + 600 * imp * imp
            x = 520 + side * (180 + 60 * hash01(i, 84)) + dx
            y = 930 + dy
            r = 3 + 16 * hash01(i, 85) * (0.5 + imp * 5)
            lit = clamp(1 - abs(x - vpx) / 1500)
            c.set_source(rad_grad(x, y, 0, r, [(0, (*mix((0.55, 0.5, 0.45), GOLD_HOT, lit), 0.55)),
                                               (1, (*GOLD, 0))]))
            circle(c, x, y, r)
            c.fill()
    # pant leg + robe hem
    lx0 = 360
    c.rectangle(lx0 + 20, by - 900, 230, 600)
    set_color(c, "#131419")
    c.fill()
    hem = by - 360
    sway = 20 * math.sin(imp * 18) * math.exp(-imp * 8) if imp > 0 else -15
    c.move_to(lx0 - 140, -50)
    c.line_to(lx0 + 420, -50)
    c.curve_to(lx0 + 430, hem - 200, lx0 + 450 + sway, hem - 60, lx0 + 470 + sway, hem)
    for k in range(6):
        c.line_to(lx0 + 470 + sway - (k + 0.5) * 105, hem + (18 if k % 2 == 0 else -6))
    c.curve_to(lx0 - 150, hem - 100, lx0 - 150, hem - 300, lx0 - 140, -50)
    c.close_path()
    c.set_source(lin_grad(lx0 + 400, hem - 500, lx0 - 100, hem, [(0, "robe_light"), (0.35, "robe"), (1, "robe_shadow")]))
    c.fill()
    set_color(c, "robe_shadow", 0.6)
    c.set_line_width(12)
    for k in range(3):
        c.move_to(lx0 + 40 + k * 110, -40)
        c.curve_to(lx0 + 60 + k * 110, hem - 300, lx0 + 30 + k * 120 + sway, hem - 120, lx0 + 20 + k * 125 + sway, hem - 10)
        c.stroke()
    set_color(c, GOLD_HOT, 0.8)
    c.set_line_width(5)
    c.move_to(lx0 + 420, -50)
    c.curve_to(lx0 + 430, hem - 200, lx0 + 450 + sway, hem - 60, lx0 + 470 + sway, hem)
    c.stroke()
    _boot(c, lx0, by, 1.05)
    brgba = bc.rgba()
    if imp < 0:
        brgba = blur(brgba, 4 * fc.s)
    img = comp(img, brgba)
    # wet reflection of the boot
    if True:
        refl = np.flipud(brgba)
        off = int(round((2 * by - 1080) * fc.s))
        M = np.array([[1, 0, 0], [0, 1, off]], np.float32)
        refl = cv2.warpAffine(refl, M, (fc.w, fc.h))
        refl = blur(refl, 10 * fc.s)
        mask = np.clip((yy - by) / 12, 0, 1)[..., None] * 0.35
        img = img * (1 - refl[..., 3:4] * mask) + refl[..., :3] * mask * 0.8
    # water droplets thrown at impact
    if imp > 0:
        dr = Canvas(fc)
        d = dr.ctx
        for i in range(40):
            side = 1 if hash01(i, 91) > 0.4 else -1
            vx = side * (300 + 900 * hash01(i, 92))
            vy = -(500 + 900 * hash01(i, 93))
            x = 520 + side * 200 + vx * imp
            y = 925 + vy * imp + 2200 * imp * imp
            L = 10 + 20 * hash01(i, 94)
            d.set_line_width(2 + 3 * hash01(i, 95))
            set_color(d, GOLD_HOT, 0.8)
            d.move_to(x, y)
            d.rel_line_to(-vx * 0.02, -(vy + 4400 * imp) * 0.02)
            d.stroke()
        img = comp(img, dr.rgba())
    img += streak(img, fc, 0.8, 380, 0.35)
    return img


SHOTS = {"velcro": velcro, "tacks": tacks, "poles": poles, "nvg": nvg, "boots": boots}

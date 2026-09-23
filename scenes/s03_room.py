"""s03 room: the flooded living room (world units = wide-shot design units).

Every draw_* takes a cairo ctx already carrying the camera transform. Albedo layers are drawn in
flat-ish base colours (the scene multiplies them by a light map driven by the broadcast);
*_emit functions draw self-luminous things (added after lighting).
"""
import math

import cairo
import numpy as np

from vx import (set_color, rrect, circle, ellipse, poly, smooth, lin_grad, rad_grad, text, mix, shade, col, clamp,
                lerp, seg, ease_in_out, ease_out, ease_in, smoothstep, pulse, fbm1, noise1, hash01)
from scenes import s03_layout as lay

WL = lay.WALL_WL


def bob(T, seed, amp=4.0, ramp=0.02, freq=1.0):
    """buoyant bob of a floating object -> (dy, rot)"""
    return (amp * (math.sin(T * 1.05 * freq + seed * 1.7) * 0.7 + 0.3 * math.sin(T * 2.3 * freq + seed)),
            ramp * math.sin(T * 0.9 * freq + seed * 2.3) + ramp * 0.4 * math.sin(T * 1.9 * freq + seed * 0.7))


def wl_at(x, y, T, amp=1.2, seed=0):
    return y + amp * math.sin(x * 0.02 + T * 2.1 + seed) + amp * 0.5 * math.sin(x * 0.047 - T * 1.3)


def clip_above(ctx, y, T=0.0, amp=1.2, seed=0, x0=-800.0, x1=2800.0):
    """clip everything below a (gently rippling) world waterline y"""
    ctx.move_to(x0, -3000)
    ctx.line_to(x1, -3000)
    n = 90
    for i in range(n + 1):
        x = x1 - i * (x1 - x0) / n
        ctx.line_to(x, wl_at(x, y, T, amp, seed))
    ctx.close_path()
    ctx.clip()


def draw_meniscus(ctx, x0, x1, y, T, amp=1.2, seed=0, lift=4.0, drop=5.0, alpha=1.0):
    """the creamy lip where the slop clings to whatever pierces it"""
    n = max(4, int((x1 - x0) / 12))
    top, bot = [], []
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        e = min(1.0, min(i, n - i) / 2.0)
        yw = wl_at(x, y, T, amp, seed)
        top.append((x, yw - lift * (0.35 + 0.65 * e) - 1.2 * math.sin(x * 0.11 + T * 1.7 + seed)))
        bot.append((x, yw + drop * (0.5 + 0.5 * e)))
    ctx.move_to(x0 - lift, wl_at(x0, y, T, amp, seed) + 1)
    for p in top:
        ctx.line_to(*p)
    ctx.line_to(x1 + lift, wl_at(x1, y, T, amp, seed) + 1)
    for p in reversed(bot):
        ctx.line_to(*p)
    ctx.close_path()
    ctx.set_source(lin_grad(0, y - lift, 0, y + drop, [(0, "#F4EBD8", 0.95 * alpha), (0.45, "#D6CAB0", 0.9 * alpha),
                                                       (1, "#B9AD95", 0.0)]))
    ctx.fill()


# ================================================================ the back wall
FRAMES = [
    # x, y (centre), w, h, rot, photo kind, drip seed
    (618.0, 188.0, 132.0, 168.0, -0.05, "family", 1),
    (620.0, 390.0, 96.0, 76.0, 0.07, "dog", 2),
    (1520.0, 150.0, 196.0, 150.0, 0.035, "wedding", 3),
    (1732.0, 330.0, 118.0, 148.0, -0.08, "baby", 4),
    (1478.0, 352.0, 104.0, 84.0, 0.02, "grad", 5),
]

WINDOW = (136.0, 96.0, 344.0, 380.0)       # x, y, w, h of the glass area


def draw_wall(ctx, T):
    # wallpaper
    ctx.rectangle(-600, -400, 3200, WL + 400 + 40)
    ctx.set_source(lin_grad(0, -200, 0, WL, [(0, "#556F78"), (0.6, "#5E7B80"), (1, "#4F666B")]))
    ctx.fill()
    # damask motif (low contrast)
    set_color(ctx, "#4B646B", 0.55)
    for iy in range(-4, 11):
        for ix in range(-6, 38):
            x = ix * 64 + (32 if iy % 2 else 0)
            y = iy * 64
            ctx.move_to(x, y - 13)
            ctx.curve_to(x + 9, y - 5, x + 9, y + 5, x, y + 13)
            ctx.curve_to(x - 9, y + 5, x - 9, y - 5, x, y - 13)
            ctx.close_path()
            ctx.fill()
            circle(ctx, x + 32, y + 32 - 64 * 0, 2.2)
            ctx.fill()
    set_color(ctx, "#6C8A8E", 0.12)
    for ix in range(-10, 60):
        ctx.rectangle(ix * 32, -400, 3, WL + 400)
        ctx.fill()
    # crown moulding
    ctx.rectangle(-600, -60, 3200, 76)
    ctx.set_source(lin_grad(0, -60, 0, 16, [(0, "#6E7380"), (0.7, "#9CA1AC"), (1, "#5A5F6B")]))
    ctx.fill()
    # slop tide line (the wave reached this high) and the stain below it
    ctx.move_to(-600, 300)
    for i in range(0, 61):
        x = -600 + i * 55
        ctx.line_to(x, 300 + 14 * math.sin(x * 0.011) + 9 * math.sin(x * 0.037 + 1) + 5 * noise1(x * 0.05, 3))
    ctx.line_to(2600, WL + 20)
    ctx.line_to(-600, WL + 20)
    ctx.close_path()
    ctx.set_source(lin_grad(0, 290, 0, WL, [(0, "#8A7E70", 0.62), (0.08, "#7A7166", 0.34), (1, "#6F675E", 0.5)]))
    ctx.fill()
    # drip streaks running down from the tide line
    for k in range(70):
        x = -500 + k * 43 + 17 * hash01(k, 8)
        y0 = 300 + 14 * math.sin(x * 0.011) + 9 * math.sin(x * 0.037 + 1)
        L = 30 + 150 * hash01(k, 9) ** 2
        w = 2 + 5 * hash01(k, 10)
        ctx.move_to(x - w / 2, y0)
        ctx.line_to(x + w / 2, y0)
        ctx.line_to(x + w * 0.3, y0 + L)
        ctx.arc(x, y0 + L, w * 0.35, 0, math.pi)
        ctx.close_path()
        set_color(ctx, "#8C8274", 0.45)
        ctx.fill()
    # dado rail + drowned wainscot
    ctx.rectangle(-600, 512, 3200, WL + 60 - 512)
    ctx.set_source(lin_grad(0, 512, 0, WL + 60, [(0, "#5B4035"), (1, "#3C2A23")]))
    ctx.fill()
    for ix in range(-6, 30):
        rrect(ctx, ix * 120 + 14, 540, 92, 90, 4)
        set_color(ctx, "#332420", 0.5)
        ctx.set_line_width(3)
        ctx.stroke()
    ctx.rectangle(-600, 504, 3200, 14)
    ctx.set_source(lin_grad(0, 504, 0, 518, [(0, "#8C6A56"), (1, "#4E372C")]))
    ctx.fill()
    # the wainscot is soaked just above the flood line; a creamy film clings at the line itself
    ctx.rectangle(-600, WL - 34, 3200, 40)
    ctx.set_source(lin_grad(0, WL - 34, 0, WL + 2, [(0, "#1A1210", 0.0), (0.8, "#1A1210", 0.55), (1, "#C9BEA6", 0.6)]))
    ctx.fill()
    draw_window(ctx, T)
    for fr in FRAMES:
        draw_frame(ctx, T, *fr)
    draw_melting_clock(ctx, T)


def draw_window(ctx, T):
    x, y, w, h = WINDOW
    # recess shadow + frame
    rrect(ctx, x - 22, y - 22, w + 44, h + 44, 4)
    set_color(ctx, "#AEB3BD")
    ctx.fill()
    rrect(ctx, x - 22, y - 22, w + 44, h + 44, 4)
    ctx.set_source(lin_grad(0, y - 22, 0, y + h + 22, [(0, "#C4C8D0", 0.0), (1, "#6E7380", 0.6)]))
    ctx.fill()
    ctx.rectangle(x, y, w, h)
    set_color(ctx, "#000000")      # glass (exterior is emissive)
    ctx.fill()
    # sill
    ctx.rectangle(x - 40, y + h + 14, w + 80, 16)
    ctx.set_source(lin_grad(0, y + h + 14, 0, y + h + 30, [(0, "#D2D5DC"), (1, "#8A8F9A")]))
    ctx.fill()


def draw_window_mullions(ctx):
    x, y, w, h = WINDOW
    set_color(ctx, "#B9BDC6")
    ctx.rectangle(x + w / 2 - 6, y, 12, h)
    ctx.fill()
    ctx.rectangle(x, y + h * 0.42 - 6, w, 12)
    ctx.fill()


def draw_exterior_emit(ctx, T):
    """night outside: moon, a slop ocean to the horizon, drowned roofs, a scatter of lonely feed-glows"""
    x, y, w, h = WINDOW
    ctx.save()
    ctx.rectangle(x, y, w, h)
    ctx.clip()
    ctx.rectangle(x, y, w, h)
    ctx.set_source(lin_grad(0, y, 0, y + h, [(0, "#070B1E"), (0.55, "#161B3C"), (0.78, "#2E2A4E"), (1, "#1A1830")]))
    ctx.fill()
    # moon halo + disc
    mx, my = x + 96, y + 78
    ctx.rectangle(x, y, w, h)
    ctx.set_source(rad_grad(mx, my, 0, 160, [(0, "#AFC4E8", 0.55), (0.3, "#5A6EA8", 0.18), (1, "#1A2040", 0.0)]))
    ctx.fill()
    circle(ctx, mx, my, 27)
    ctx.set_source(rad_grad(mx - 6, my - 6, 2, 30, [(0, "#F4F7FF"), (1, "#C9D4EA")]))
    ctx.fill()
    set_color(ctx, "#AEB9D2", 0.5)
    circle(ctx, mx + 7, my + 4, 5)
    ctx.fill()
    circle(ctx, mx - 8, my + 9, 3)
    ctx.fill()
    # thin cloud bands
    for k in range(4):
        cy = y + 60 + k * 34
        ox = (T * 6 + k * 90) % 500 - 120
        ellipse(ctx, x + ox, cy, 140, 6)
        set_color(ctx, "#2A3160", 0.55)
        ctx.fill()
    # slop sea horizon
    hy = y + h * 0.66
    ctx.rectangle(x, hy, w, h)
    ctx.set_source(lin_grad(0, hy, 0, y + h, [(0, "#4A4260"), (1, "#231F33")]))
    ctx.fill()
    # moon glitter path on the slop
    for k in range(26):
        yy = hy + 3 + k * 4.5
        ww = 4 + k * 0.9 + 3 * math.sin(T * 3 + k * 1.7)
        ctx.rectangle(mx - ww / 2 + 3 * math.sin(T * 2.2 + k), yy, ww, 1.6)
        set_color(ctx, "#C5D0EC", 0.5 * (1 - k / 26))
        ctx.fill()
    # drowned rooftops / towers
    set_color(ctx, "#0F1022")
    for k, (bx, bw, bh) in enumerate(((x + 12, 40, 22), (x + 70, 22, 48), (x + 160, 56, 16), (x + 230, 30, 60),
                                      (x + 272, 50, 26), (x + 318, 30, 12))):
        ctx.rectangle(bx, hy - bh, bw, bh + 2)
        ctx.fill()
        if k == 3:
            poly(ctx, [(bx, hy - bh), (bx + bw / 2, hy - bh - 22), (bx + bw, hy - bh)])
            ctx.fill()
    # lonely feed glows: a different screen for every face (never yellow)
    for k in range(9):
        gx = x + 20 + (k * 71.3) % (w - 40)
        gy = hy - 6 + 30 * hash01(k, 3) + (k % 3) * 6
        c = ("#00E5FF", "#FF2E88", "#5CFF9D", "#A57CFF", "#00E5FF", "#FF6BD5", "#6BC8FF", "#FF2E88", "#5CFF9D")[k]
        a = 0.55 + 0.45 * math.sin(T * (2 + hash01(k, 4) * 3) + k)
        ctx.rectangle(x, y, w, h)
        ctx.set_source(rad_grad(gx, gy, 0, 16, [(0, c, 0.6 * a), (1, c, 0.0)]))
        ctx.fill()
        circle(ctx, gx, gy, 1.6)
        set_color(ctx, c, 0.95)
        ctx.fill()
    ctx.restore()


def draw_curtains(ctx, T):
    x, y, w, h = WINDOW
    for side in (-1, 1):
        cx = x - 18 if side < 0 else x + w + 18
        top = y - 44
        bot = WL + 30
        inner = cx + side * -1 * (70 + 6 * math.sin(T * 0.5 + side))
        outer = cx + side * 52
        # tie-back pinch at ~62% height
        ty = y + h * 0.62
        pts = [(outer, top), (inner + side * 30, top), (inner + side * 2, ty - 90), (cx + side * 12 - side * 8, ty),
               (inner - side * 16, bot), (outer + side * 14, bot)]
        ctx.move_to(*pts[0])
        ctx.line_to(*pts[1])
        ctx.curve_to(inner + side * 20, top + 120, inner - side * 4, ty - 60, pts[3][0], pts[3][1])
        ctx.curve_to(pts[3][0] - side * 18, ty + 60, inner - side * 26, bot - 60, pts[4][0], pts[4][1])
        ctx.line_to(*pts[5])
        ctx.close_path()
        g = cairo.LinearGradient(min(inner, outer) - 30, 0, max(inner, outer) + 30, 0)
        for i in range(9):
            u = i / 8
            k = 0.78 + 0.22 * math.sin(i * 2.6 + side)
            c = shade("#7B5E78", k)
            g.add_color_stop_rgb(u, *c)
        ctx.set_source(g)
        ctx.fill_preserve()
        # slop-soaked hem
        ctx.set_source(lin_grad(0, ty + 40, 0, bot, [(0, "#8C8274", 0.0), (0.6, "#8C8274", 0.55), (1, "#9A8F7E", 0.85)]))
        ctx.fill()
        # tie-back
        ellipse(ctx, cx + side * 12 - side * 8, ty, 30, 8, 0.1 * side)
        set_color(ctx, "#5A3F52")
        ctx.fill()
    # rod
    ctx.rectangle(x - 90, y - 52, w + 180, 9)
    set_color(ctx, "#6A6E78")
    ctx.fill()
    for ex in (x - 94, x + w + 94):
        circle(ctx, ex, y - 48, 9)
        ctx.fill()


PHOTO_BG = {"family": ("#A9C3D9", "#7F9DB8"), "dog": ("#C9D8B6", "#9FB58E"), "wedding": ("#E3C6CF", "#C99DAC"),
            "baby": ("#CFE0EC", "#A9C5DA"), "grad": ("#C8C2E0", "#9E97C3")}


def _bust(ctx, x, y, r, c):
    set_color(ctx, c)
    circle(ctx, x, y, r)
    ctx.fill()
    ctx.move_to(x - r * 1.9, y + r * 3.2)
    ctx.curve_to(x - r * 1.8, y + r * 1.3, x + r * 1.8, y + r * 1.3, x + r * 1.9, y + r * 3.2)
    ctx.close_path()
    ctx.fill()


def draw_frame(ctx, T, x, y, w, h, rot, kind, seed):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot + 0.004 * math.sin(T * 0.7 + seed))
    # shadow on wall
    rrect(ctx, -w / 2 + 6, -h / 2 + 9, w, h, 3)
    set_color(ctx, "#26343A", 0.45)
    ctx.fill()
    # frame
    rrect(ctx, -w / 2, -h / 2, w, h, 3)
    ctx.set_source(lin_grad(-w / 2, -h / 2, w / 2, h / 2, [(0, "#6A4A38"), (0.5, "#3E2A20"), (1, "#2A1C16")]))
    ctx.fill()
    fw = min(w, h) * 0.1
    pw, ph = w - 2 * fw, h - 2 * fw
    ctx.rectangle(-pw / 2, -ph / 2, pw, ph)
    set_color(ctx, "#DCD6CC")
    ctx.fill()
    mw = min(pw, ph) * 0.08
    iw, ih = pw - 2 * mw, ph - 2 * mw
    ctx.rectangle(-iw / 2, -ih / 2, iw, ih)
    c0, c1 = PHOTO_BG[kind]
    ctx.set_source(lin_grad(0, -ih / 2, 0, ih / 2, [(0, c0), (1, c1)]))
    ctx.fill()
    ctx.save()
    ctx.rectangle(-iw / 2, -ih / 2, iw, ih)
    ctx.clip()
    sil = "#4A5566"
    if kind == "family":
        _bust(ctx, -iw * 0.2, -ih * 0.08, iw * 0.12, "#556070")
        _bust(ctx, iw * 0.2, -ih * 0.12, iw * 0.13, "#4A5566")
        _bust(ctx, 0, ih * 0.14, iw * 0.09, "#66728A")
    elif kind == "dog":
        set_color(ctx, "#6A5A4E")
        ellipse(ctx, 0, ih * 0.12, iw * 0.24, ih * 0.22)
        ctx.fill()
        circle(ctx, iw * 0.14, -ih * 0.14, iw * 0.14)
        ctx.fill()
        ellipse(ctx, iw * 0.02, -ih * 0.22, iw * 0.05, ih * 0.14, 0.4)
        ctx.fill()
    elif kind == "wedding":
        _bust(ctx, -iw * 0.14, -ih * 0.06, iw * 0.085, "#3E4656")
        _bust(ctx, iw * 0.14, -ih * 0.04, iw * 0.08, "#F2EEF0")
        set_color(ctx, "#F6F2F4", 0.9)
        poly(ctx, [(iw * 0.14, -ih * 0.26), (iw * 0.32, ih * 0.5), (-iw * 0.02, ih * 0.5)])
        ctx.fill()
    elif kind == "baby":
        circle(ctx, 0, -ih * 0.02, iw * 0.2)
        set_color(ctx, "#E6C2B0")
        ctx.fill()
        set_color(ctx, "#7C8FB0")
        ellipse(ctx, 0, ih * 0.4, iw * 0.4, ih * 0.22)
        ctx.fill()
    elif kind == "grad":
        _bust(ctx, 0, -ih * 0.02, iw * 0.13, sil)
        set_color(ctx, "#222833")
        poly(ctx, [(-iw * 0.24, -ih * 0.2), (0, -ih * 0.3), (iw * 0.24, -ih * 0.2), (0, -ih * 0.12)])
        ctx.fill()
    ctx.restore()
    # glass sheen
    ctx.rectangle(-pw / 2, -ph / 2, pw, ph)
    ctx.set_source(lin_grad(-pw / 2, -ph / 2, pw / 2, ph / 2, [(0, "#FFFFFF", 0.18), (0.35, "#FFFFFF", 0.0),
                                                              (0.6, "#FFFFFF", 0.06), (1, "#FFFFFF", 0.0)]))
    ctx.fill()
    # slop dripping over the top of the frame and down the glass
    draw_drips(ctx, T, -w / 2, -h / 2, w, h, seed)
    ctx.restore()


def drip_state(T, k, seed):
    """a slop drip's current length and detached drop (per drip k): period, phase -> (len, drop_y or None)"""
    per = 2.2 + 2.6 * hash01(k, seed * 13 + 1)
    ph = hash01(k, seed * 13 + 2) * per
    u = ((T + ph) % per) / per
    L = 0.15 + 0.85 * u ** 1.6
    drop = None
    if u < 0.25:
        drop = u / 0.25
    return L, drop


def draw_drips(ctx, T, x, y, w, h, seed, thick=1.0):
    """a glossy gob of slop sagging over an edge (x..x+w at height y), with drips creeping down"""
    t = 7.0 * thick
    # the gob: a lumpy overhang with rounded bumps
    ctx.move_to(x - 6, y + t)
    ctx.line_to(x - 6, y - t * 0.3)
    n = 7
    for i in range(n):
        x0 = x - 6 + i * (w + 12) / n
        x1 = x - 6 + (i + 1) * (w + 12) / n
        bh = t * (0.8 + 0.9 * hash01(i, seed * 7))
        ctx.curve_to(x0 + (x1 - x0) * 0.2, y - bh, x0 + (x1 - x0) * 0.8, y - bh, x1, y - t * 0.3)
    ctx.line_to(x + w + 6, y + t)
    ctx.close_path()
    ctx.set_source(lin_grad(0, y - 2 * t, 0, y + t, [(0, "#EDE3D0"), (0.5, "#D2C6AE"), (1, "#AFA38C")]))
    ctx.fill()
    nd = 3 + int(3 * hash01(seed, 99))
    for k in range(nd):
        dx = x + 8 + (w - 16) * (k + 0.5 * hash01(k, seed)) / nd
        wd = (5 + 6 * hash01(k, seed + 5)) * thick
        Lmax = h * (0.35 + 0.6 * hash01(k, seed + 6))
        L, drop = drip_state(T, k, seed)
        ln = t + Lmax * L
        br = wd * 0.62
        ctx.move_to(dx - wd * 0.75, y + t - 1)
        ctx.curve_to(dx - wd * 0.3, y + t + ln * 0.3, dx - wd * 0.22, y + ln - br * 1.6, dx - br, y + ln)
        ctx.arc_negative(dx, y + ln, br, math.pi, 0)
        ctx.curve_to(dx + wd * 0.22, y + ln - br * 1.6, dx + wd * 0.3, y + t + ln * 0.3, dx + wd * 0.75, y + t - 1)
        ctx.close_path()
        ctx.set_source(lin_grad(dx - wd, 0, dx + wd, 0, [(0, "#A99D86"), (0.35, "#DDD2BC"), (1, "#9E927B")]))
        ctx.fill()
        # specular pip on the bulb + a streak down the neck
        set_color(ctx, "#FFF8EC", 0.7)
        ellipse(ctx, dx - br * 0.35, y + ln - br * 0.2, br * 0.18, br * 0.3)
        ctx.fill()
        ctx.set_line_width(max(0.6, wd * 0.12))
        set_color(ctx, "#FFF8EC", 0.3)
        ctx.move_to(dx - wd * 0.2, y + t + 2)
        ctx.line_to(dx - wd * 0.14, y + ln - br * 1.4)
        ctx.stroke()
        # the drop that just let go
        if drop is not None:
            dy_ = y + ln + 6 + drop * drop * 60
            ellipse(ctx, dx, dy_, br * 0.55, br * (0.7 + 0.3 * drop))
            set_color(ctx, "#D6CBB4", 1 - drop)
            ctx.fill()


def draw_melting_clock(ctx, T):
    """a wall clock sagging like it was caught in the wave (a nod to the slop's melting clocks)"""
    cx, cy = 1268.0, 96.0
    ctx.save()
    ctx.translate(cx, cy)
    ctx.move_to(-44, -8)
    ctx.curve_to(-46, -52, 46, -52, 44, -8)
    ctx.curve_to(46, 30, 30, 34, 26, 58)
    ctx.curve_to(22, 84, 6, 86, 4, 64)
    ctx.curve_to(0, 40, -40, 40, -44, -8)
    ctx.close_path()
    ctx.set_source(lin_grad(-40, -40, 40, 60, [(0, "#E4E0D8"), (1, "#A8A298")]))
    ctx.fill_preserve()
    ctx.set_line_width(5)
    set_color(ctx, "#4A4E5A")
    ctx.stroke()
    set_color(ctx, "#2A2C34")
    for k in range(12):
        a = k * math.pi / 6
        r0 = 30 + (6 if a > 0.3 and a < 2.4 else 0) * math.sin(a)
        ctx.rectangle(math.cos(a) * r0 - 1.5, math.sin(a) * r0 * (1.0 + 0.35 * max(0, math.sin(a))) - 1.5 - 4, 3, 3)
        ctx.fill()
    ctx.set_line_width(3)
    ctx.move_to(0, -6)
    ctx.line_to(12, -22)
    ctx.stroke()
    ctx.set_line_width(2)
    ctx.move_to(0, -6)
    ctx.curve_to(4, 10, 2, 24, -4, 36)
    ctx.stroke()
    ctx.restore()


# ================================================================ floating furniture
LAMP_WL = 684.0          # waterline of the lamp's depth
LAMP_BASE = (1812.0, LAMP_WL)
LAMP_LEN = 330.0
LAMP_ANG = -0.62         # radians from vertical (leaning left)


def lamp_geom(T):
    dy, r = bob(T, 3.0, 3.0, 0.03, 0.8)
    ang = LAMP_ANG + r
    bx, by = LAMP_BASE[0], LAMP_BASE[1] + dy + 30
    tx, ty = bx + math.sin(ang) * LAMP_LEN, by - math.cos(ang) * LAMP_LEN
    return bx, by, tx, ty, ang


def lamp_power(T):
    tm = lay.times()
    # the lamp gutters out with the TV at the end (the room goes dark)
    p = 1.0 - 0.1 * max(0.0, noise1(T * 9.0, 4)) * (T > 52)
    p *= 1.0 - seg(T, tm["off"] + 0.02, tm["off"] + 0.1, ease_in)
    return p


def draw_lamp(ctx, T):
    bx, by, tx, ty, ang = lamp_geom(T)
    ctx.save()
    clip_above(ctx, LAMP_WL, T, seed=3)
    ctx.set_line_width(9)
    ctx.set_source(lin_grad(bx - 5, 0, bx + 5, 0, [(0, "#6E7078"), (0.5, "#B6B9C2"), (1, "#5A5C64")]))
    ctx.move_to(bx, by)
    ctx.line_to(tx, ty)
    ctx.stroke()
    ctx.restore()
    # where the pole pierces the slop
    ct = math.cos(ang)
    if abs(ct) > 1e-3:
        tpar = (by - LAMP_WL) / ct
        xw = bx + math.sin(ang) * tpar
        draw_meniscus(ctx, xw - 11, xw + 11, LAMP_WL, T, seed=3, lift=3, drop=4)
    # shade (a tilted drum), bulb below
    ctx.save()
    ctx.translate(tx, ty)
    ctx.rotate(ang)
    poly(ctx, [(-58, -64), (58, -64), (84, 22), (-84, 22)])
    ctx.set_source(lin_grad(-84, 0, 84, 0, [(0, "#9C6A62"), (0.4, "#D49A88"), (1, "#8A5A54")]))
    ctx.fill()
    ellipse(ctx, 0, 22, 84, 11)
    set_color(ctx, "#6A4440")
    ctx.fill()
    ctx.restore()


def draw_lamp_emit(ctx, T):
    bx, by, tx, ty, ang = lamp_geom(T)
    p = lamp_power(T)
    if p <= 0.01:
        return
    ctx.save()
    ctx.translate(tx, ty)
    ctx.rotate(ang)
    # translucent shade glow
    poly(ctx, [(-58, -64), (58, -64), (84, 22), (-84, 22)])
    ctx.set_source(rad_grad(0, 0, 4, 110, [(0, "#FFB08C", 0.95 * p), (0.6, "#E07A62", 0.55 * p), (1, "#8A3A34", 0.2 * p)]))
    ctx.fill()
    # hot rim at the bottom opening + bulb
    ellipse(ctx, 0, 22, 80, 9)
    set_color(ctx, "#FFD8C4", 0.8 * p)
    ctx.fill()
    circle(ctx, 0, 10, 18)
    ctx.set_source(rad_grad(0, 10, 0, 30, [(0, "#FFF6EE", p), (1, "#FFB08C", 0.0)]))
    ctx.fill()
    # light spilling out of the openings
    poly(ctx, [(-80, 22), (80, 22), (190, 260), (-190, 260)])
    ctx.set_source(lin_grad(0, 22, 0, 260, [(0, "#FF9A7A", 0.2 * p), (1, "#FF9A7A", 0.0)]))
    ctx.fill()
    poly(ctx, [(-56, -64), (56, -64), (120, -200), (-120, -200)])
    ctx.set_source(lin_grad(0, -64, 0, -200, [(0, "#FF9A7A", 0.14 * p), (1, "#FF9A7A", 0.0)]))
    ctx.fill()
    ctx.restore()


CHAIR_WL = 842.0
CHAIR_X = 330.0


def chair_pose(T):
    dy, r = bob(T, 1.0, 5.0, 0.025, 0.9)
    tm = lay.times()
    # the slam's slosh reaches the chair a beat later
    d = T - tm["slam"] - 0.45
    if d > 0:
        dy += 7 * math.sin(d * 7) * math.exp(-d * 2.2)
        r += 0.02 * math.sin(d * 6 + 1) * math.exp(-d * 2.0)
    return CHAIR_X, CHAIR_WL - 40 + dy, -0.13 + r


def draw_chair(ctx, T, cat=True):
    x, y, rot = chair_pose(T)
    ctx.save()
    clip_above(ctx, CHAIR_WL, T, amp=1.8, seed=1)
    ctx.translate(x, y)
    ctx.rotate(rot)
    velvet = "#7E5A84"
    # back (wingback), seen 3/4 from behind-left, facing the TV (right)
    ctx.move_to(-190, 40)
    ctx.curve_to(-205, -120, -170, -290, -60, -300)
    ctx.curve_to(10, -305, 40, -250, 30, -170)
    ctx.line_to(20, 40)
    ctx.close_path()
    ctx.set_source(lin_grad(-200, -300, 40, 40, [(0, shade(velvet, 1.18)), (0.5, velvet), (1, shade(velvet, 0.6))]))
    ctx.fill()
    # tufting on the back
    for iy in range(4):
        for ix in range(3):
            bx = -150 + ix * 55 + (iy % 2) * 27
            by_ = -230 + iy * 55
            circle(ctx, bx, by_, 4)
            set_color(ctx, shade(velvet, 0.55))
            ctx.fill()
            set_color(ctx, shade(velvet, 0.8), 0.6)
            ctx.set_line_width(2)
            ctx.move_to(bx - 20, by_ - 18)
            ctx.line_to(bx, by_)
            ctx.line_to(bx + 22, by_ - 20)
            ctx.stroke()
    # wing
    ctx.move_to(10, -250)
    ctx.curve_to(90, -250, 120, -170, 110, -80)
    ctx.line_to(30, -60)
    ctx.curve_to(30, -130, 40, -200, 10, -250)
    ctx.close_path()
    ctx.set_source(lin_grad(0, -250, 120, -60, [(0, shade(velvet, 1.05)), (1, shade(velvet, 0.7))]))
    ctx.fill()
    # seat cushion + arm
    rrect(ctx, -170, -40, 300, 70, 26)
    ctx.set_source(lin_grad(0, -40, 0, 30, [(0, shade(velvet, 1.25)), (1, shade(velvet, 0.75))]))
    ctx.fill()
    rrect(ctx, 60, -120, 150, 70, 34)
    ctx.set_source(lin_grad(0, -120, 0, -50, [(0, shade(velvet, 1.3)), (1, shade(velvet, 0.8))]))
    ctx.fill()
    ctx.move_to(70, -60)
    ctx.line_to(205, -80)
    ctx.line_to(215, 60)
    ctx.line_to(60, 60)
    ctx.close_path()
    ctx.set_source(lin_grad(60, 0, 215, 0, [(0, shade(velvet, 0.9)), (1, shade(velvet, 0.62))]))
    ctx.fill()
    # skirt
    ctx.rectangle(-190, 20, 410, 80)
    set_color(ctx, shade(velvet, 0.5))
    ctx.fill()
    # slop soaking the lower upholstery
    ctx.rectangle(-220, -10, 460, 120)
    ctx.set_source(lin_grad(0, -10, 0, 60, [(0, "#8C8274", 0.0), (1, "#9C917E", 0.8)]))
    ctx.fill()
    # a TV remote left on the seat
    ctx.save()
    ctx.translate(-80, -44)
    ctx.rotate(0.12)
    rrect(ctx, -34, -7, 68, 14, 5)
    set_color(ctx, "#202127")
    ctx.fill()
    for k in range(5):
        circle(ctx, -22 + k * 11, 0, 2.2)
        set_color(ctx, "#5A5C68")
        ctx.fill()
    circle(ctx, 26, 0, 3)
    set_color(ctx, "#A83A3A")
    ctx.fill()
    ctx.restore()
    if cat:
        draw_cat(ctx, T, 136, -118)
    ctx.restore()
    draw_meniscus(ctx, x - 212, x + 232, CHAIR_WL, T, amp=1.8, seed=1, lift=6, drop=8)


def cat_state(T):
    tm = lay.times()
    d = T - tm["slam"]
    startle = 0.0
    if d > 0:
        startle = math.exp(-d * 2.0) * (1 - math.exp(-d * 30))
    return startle


def draw_cat(ctx, T, x, y):
    """the household's lone survivor, sitting on the armrest, watching the President"""
    st = cat_state(T)
    ctx.save()
    ctx.translate(x, y - 22 * st)
    fur = "#2B2B33"
    # tail curling down over the arm
    sw = math.sin(T * 1.7) * 0.5 + 0.3 * math.sin(T * 3.1)
    ctx.set_line_width(11)
    set_color(ctx, fur)
    ctx.move_to(-30, -6)
    ctx.curve_to(-60, 10, -70 + 12 * sw, 40, -52 + 16 * sw, 70 - 10 * st)
    ctx.stroke()
    # body (sitting, 3/4 from behind, facing right)
    ctx.move_to(-40, 0)
    ctx.curve_to(-52, -50, -30, -92, 4, -96)
    ctx.curve_to(30, -96, 40, -60, 36, 0)
    ctx.close_path()
    set_color(ctx, fur)
    ctx.fill()
    # head
    hx, hy = 16 + 4 * st, -112 - 6 * st
    ear = 0.35 * st
    circle(ctx, hx, hy, 24)
    ctx.fill()
    for ex, s in ((hx - 14, -1), (hx + 12, 1)):
        poly(ctx, [(ex - 9, hy - 14), (ex + s * 6 + s * ear * 10, hy - 40 + ear * 16), (ex + 10, hy - 16)])
        ctx.fill()
    ctx.restore()


def draw_cat_emit(ctx, T, cam_to_chair):
    """eyeshine: tapetum catching the TV (only the near eye is visible from our angle)"""
    x, y, rot = chair_pose(T)
    st = cat_state(T)
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    ctx.translate(136, -118 - 22 * st)
    hx, hy = 16 + 4 * st, -112 - 6 * st
    r = 4.2 + 2.0 * st
    ellipse(ctx, hx + 20, hy - 2, r * 0.6, r)
    ctx.set_source(rad_grad(hx + 20, hy - 2, 0, r * 2.5, [(0, "#D8FFF6", 0.95), (0.4, "#5FE0D0", 0.6), (1, "#1A6060", 0.0)]))
    ctx.fill()
    ctx.restore()


# ================================================================ the TV
def draw_tv(ctx, T, led=1.0):
    """cabinet + bezel in TV-local coords (caller: translate to tube centre, rotate by roll).
    The tube face is composited by the scene (s03_crt) on top of the bezel hole."""
    # cabinet body
    rrect(ctx, lay.CAB_L, lay.CAB_T, lay.CAB_R - lay.CAB_L, lay.CAB_B - lay.CAB_T, 22)
    ctx.set_source(lin_grad(0, lay.CAB_T, 0, lay.CAB_B, [(0, "#7A4E36"), (0.3, "#63402C"), (1, "#3E281C")]))
    ctx.fill()
    # walnut grain
    ctx.save()
    rrect(ctx, lay.CAB_L, lay.CAB_T, lay.CAB_R - lay.CAB_L, lay.CAB_B - lay.CAB_T, 22)
    ctx.clip()
    for k in range(38):
        yy = lay.CAB_T + 8 + k * 18
        ctx.move_to(lay.CAB_L, yy)
        for i in range(1, 17):
            xx = lay.CAB_L + i * 40
            ctx.line_to(xx, yy + 5 * math.sin(xx * 0.013 + k * 0.9) + 3 * math.sin(xx * 0.04 + k))
        ctx.set_line_width(1.4 + 1.6 * hash01(k, 21))
        set_color(ctx, "#3A2418", 0.35)
        ctx.stroke()
    # knots
    for (kx, ky) in ((300, 130), (-180, 205), (200, -170)):
        ellipse(ctx, kx, ky, 18, 6)
        set_color(ctx, "#3A2418", 0.35)
        ctx.set_line_width(2)
        ctx.stroke()
    ctx.restore()
    # top bevel highlight
    rrect(ctx, lay.CAB_L + 6, lay.CAB_T + 3, lay.CAB_R - lay.CAB_L - 12, 12, 6)
    set_color(ctx, "#A07458", 0.55)
    ctx.fill()
    # bezel around the tube
    rrect(ctx, -240, -186, 480, 372, 34)
    ctx.set_source(lin_grad(0, -186, 0, 186, [(0, "#2A2A32"), (1, "#141418")]))
    ctx.fill()
    rrect(ctx, -228, -174, 456, 348, 28)
    ctx.set_line_width(4)
    ctx.set_source(lin_grad(-228, -174, 228, 174, [(0, "#D8DCE4"), (0.5, "#7A7F8A"), (1, "#B8BCC6")]))
    ctx.stroke()
    # tube recess (dark) -- the CRT face covers it
    _superellipse(ctx, 0, 0, lay.TUBE_HW + 7, lay.TUBE_HH + 7)
    set_color(ctx, "#08080A")
    ctx.fill()
    # control column
    px = 300.0
    rrect(ctx, 250, -186, 100, 372, 12)
    ctx.set_source(lin_grad(250, 0, 350, 0, [(0, "#2B2B31"), (0.5, "#3A3A42"), (1, "#222228")]))
    ctx.fill()
    for k, (ky, r) in enumerate(((-130, 30), (-50, 26))):
        circle(ctx, px, ky, r + 5)
        set_color(ctx, "#15151A")
        ctx.fill()
        ang = (0.6 if k == 0 else -1.2) + 0.02 * math.sin(T)
        # ridged knob
        for j in range(24):
            a = ang + j * math.pi / 12
            ctx.move_to(px + math.cos(a) * r * 0.6, ky + math.sin(a) * r * 0.6)
            ctx.line_to(px + math.cos(a) * r, ky + math.sin(a) * r)
        ctx.set_line_width(3)
        set_color(ctx, "#4A4A54")
        ctx.stroke()
        circle(ctx, px, ky, r * 0.62)
        ctx.set_source(rad_grad(px - r * 0.2, ky - r * 0.2, 1, r * 0.7, [(0, "#E8ECF2"), (1, "#7A7F8A")]))
        ctx.fill()
        ctx.set_line_width(3)
        set_color(ctx, "#2A2A30")
        ctx.move_to(px, ky)
        ctx.line_to(px + math.cos(ang) * r * 0.55, ky + math.sin(ang) * r * 0.55)
        ctx.stroke()
    # channel numbers around the dial
    for j in range(8):
        a = -math.pi * 0.9 + j * math.pi * 0.26
        text(ctx, str(2 + j), px + math.cos(a) * 44, -130 + math.sin(a) * 44 + 3, 9, "#9A9EA8", align="center",
             bold=True)
    # push buttons + LED
    for j in range(3):
        rrect(ctx, 270 + j * 22, 14, 16, 22, 3)
        set_color(ctx, "#B8BCC6")
        ctx.fill()
    circle(ctx, px, 58, 5)
    set_color(ctx, "#3A1010")
    ctx.fill()
    # speaker grille
    for j in range(8):
        rrect(ctx, 266, 80 + j * 12, 68, 5, 2.5)
        set_color(ctx, "#121216")
        ctx.fill()
    # brand badge
    rrect(ctx, -40, 194, 80, 18, 4)
    ctx.set_source(lin_grad(-40, 0, 40, 0, [(0, "#8A8F9A"), (0.5, "#E8ECF2"), (1, "#8A8F9A")]))
    ctx.fill()
    text(ctx, "NOOTRON", 0, 207, 11, "#2A2C34", font="C059", bold=True, align="center", tracking=0.18)
    # a gob of slop sagging over the cabinet's top edge, dripping down the wood
    draw_drips(ctx, T, lay.CAB_L + 20, lay.CAB_T + 2, 250, 42, 17, thick=1.25)
    # wet band just above the waterline
    ctx.rectangle(lay.CAB_L, lay.TV_WL_LOCAL - 40, lay.CAB_R - lay.CAB_L, 60)
    ctx.set_source(lin_grad(0, lay.TV_WL_LOCAL - 40, 0, lay.TV_WL_LOCAL + 5,
                            [(0, "#9C917E", 0.0), (1, "#A89C86", 0.85)]))
    ctx.fill()


def draw_tv_emit(ctx, T, led=1.0):
    if led > 0:
        circle(ctx, 300, 58, 4.5)
        ctx.set_source(rad_grad(300, 58, 0, 16, [(0, "#FF5A4A", led), (0.35, "#FF2A1A", 0.6 * led), (1, "#FF2A1A", 0.0)]))
        ctx.arc(300, 58, 16, 0, 2 * math.pi)
        ctx.fill()


def _superellipse(ctx, cx, cy, a, b, p=5.0, n=96):
    for i in range(n + 1):
        t = i / n * 2 * math.pi
        c, s = math.cos(t), math.sin(t)
        x = cx + a * math.copysign(abs(c) ** (2 / p), c)
        y = cy + b * math.copysign(abs(s) ** (2 / p), s)
        if i == 0:
            ctx.move_to(x, y)
        else:
            ctx.line_to(x, y)
    ctx.close_path()


def draw_antenna(ctx, T):
    """rabbit ears (TV-local), one tip crowned with a wad of foil"""
    tm = lay.times()
    d = T - tm["slam"]
    wob = 0.0
    if d > 0:
        wob = 0.12 * math.sin(d * 18) * math.exp(-d * 3.5)
    base = (60.0, lay.CAB_T - 4)
    ellipse(ctx, base[0], base[1] - 6, 34, 16)
    ctx.set_source(lin_grad(0, base[1] - 22, 0, base[1] + 10, [(0, "#4A4A52"), (1, "#18181C")]))
    ctx.fill()
    for k, (a0, L) in enumerate(((-0.62, 320.0), (0.5, 300.0))):
        a = a0 + wob * (1 if k else -0.8) + 0.02 * math.sin(T * 1.3 + k)
        tx, ty = base[0] + math.sin(a) * L, base[1] - 16 - math.cos(a) * L
        ctx.set_line_width(7)
        ctx.set_source(lin_grad(base[0], base[1], tx, ty, [(0, "#7A7E88"), (1, "#C8CCD6")]))
        ctx.move_to(base[0], base[1] - 16)
        ctx.line_to(tx, ty)
        ctx.stroke()
        # telescoping joints + a specular line
        for j in (0.4, 0.7):
            circle(ctx, base[0] + math.sin(a) * L * j, base[1] - 16 - math.cos(a) * L * j, 5)
            set_color(ctx, "#A8ACB6")
            ctx.fill()
        ctx.set_line_width(2)
        set_color(ctx, "#FFFFFF", 0.7)
        ctx.move_to(base[0] - 2, base[1] - 20)
        ctx.line_to(tx - 2, ty)
        ctx.stroke()
        if k == 0:
            # crumpled foil
            for j in range(9):
                aa = j * 0.7
                rr = 11 + 5 * hash01(j, 50)
                ctx.line_to(tx + math.cos(aa) * rr, ty + math.sin(aa) * rr)
            ctx.close_path()
            ctx.set_source(rad_grad(tx - 4, ty - 4, 1, 16, [(0, "#F2F4F8"), (0.6, "#A8AEB8"), (1, "#6A707A")]))
            ctx.fill()
        else:
            circle(ctx, tx, ty, 5)
            set_color(ctx, "#D8DCE4")
            ctx.fill()


# ================================================================ debris
DEBRIS = [
    # kind, x, waterline y, scale, seed, rot
    ("pizza", 690.0, 905.0, 1.15, 1, -0.2),
    ("card", 1180.0, 792.0, 0.9, 2, 0.15),
    ("mag", 1480.0, 868.0, 1.1, 3, 0.3),
    ("phone", 1650.0, 980.0, 1.3, 4, -0.4),
    ("card", 1860.0, 760.0, 0.75, 5, -0.2),
    ("plant", 90.0, 1000.0, 1.5, 6, 0.1),
    ("can", 960.0, 1010.0, 1.4, 7, 0.9),
    ("card", 420.0, 1050.0, 1.4, 8, -0.1),
    ("duck_no", 0, 0, 0, 0, 0),
]


def debris_list():
    return [d for d in DEBRIS if d[0] != "duck_no"]


def draw_debris_item(ctx, T, kind, x, wl, sc, seed, rot):
    dy, r = bob(T, seed, 2.5 * sc, 0.05, 1.2)
    ctx.save()
    ctx.translate(x, wl + dy)
    ctx.rotate(r * 0.5)
    ctx.scale(sc, sc)
    if kind == "pizza":
        # open box: base (flat, sunk) + lid tilted up
        poly(ctx, [(-70, -4), (70, -4), (86, 6), (-86, 6)])
        set_color(ctx, "#9C8468")
        ctx.fill()
        poly(ctx, [(-70, -4), (70, -4), (60, -80), (-60, -84)])
        ctx.set_source(lin_grad(0, -84, 0, -4, [(0, "#B09678"), (1, "#8A7458")]))
        ctx.fill()
        # the pizzeria's logo on the lid: a red roundel with a chef's moustache
        circle(ctx, -4, -44, 24)
        set_color(ctx, "#B8322C")
        ctx.fill()
        circle(ctx, -4, -44, 19)
        set_color(ctx, "#E9E2D6")
        ctx.fill()
        ctx.set_line_width(4)
        set_color(ctx, "#B8322C")
        ctx.move_to(-16, -42)
        ctx.curve_to(-12, -50, -6, -46, -4, -43)
        ctx.curve_to(-2, -46, 4, -50, 8, -42)
        ctx.stroke()
        text(ctx, "PIZZA", -4, -12, 12, "#B8322C", bold=True, align="center", tracking=0.1)
        # slop soaked into the corner of the lid
        poly(ctx, [(40, -4), (70, -4), (64, -46), (48, -30)])
        set_color(ctx, "#B5AA92", 0.8)
        ctx.fill()
    elif kind == "card":
        ctx.rotate(rot)
        rrect(ctx, -46, -26, 92, 30, 6)
        set_color(ctx, "#E9E6EE")
        ctx.fill()
        rrect(ctx, -40, -21, 30, 20, 3)
        set_color(ctx, "#B8C8D8")
        ctx.fill()
        for k in range(2):
            ctx.rectangle(-4, -18 + k * 8, 40 - k * 12, 3)
            set_color(ctx, "#9AA0AC")
            ctx.fill()
        circle(ctx, 42, -24, 8)
        set_color(ctx, "#E0304A")
        ctx.fill()
        text(ctx, str(9 + seed * 13), 42, -21, 9, "#FFFFFF", bold=True, align="center")
    elif kind == "mag":
        ctx.rotate(rot * 0.3)
        poly(ctx, [(-60, -4), (60, -10), (66, 4), (-56, 8)])
        ctx.set_source(lin_grad(-60, 0, 66, 0, [(0, "#D86A9A"), (1, "#8A5AB8")]))
        ctx.fill()
        text(ctx, "SLOP", -40, 2, 12, "#FFFFFF", bold=True)
    elif kind == "phone":
        ctx.rotate(rot * 0.2)
        poly(ctx, [(-34, -6), (30, -12), (40, 2), (-26, 8)])
        set_color(ctx, "#15161B")
        ctx.fill()
    elif kind == "plant":
        rrect(ctx, -30, -30, 60, 36, 6)
        set_color(ctx, "#A0503A")
        ctx.fill()
        for k in range(9):
            a = -1.4 + k * 0.35
            ctx.move_to(0, -28)
            ctx.curve_to(math.sin(a) * 40, -60, math.sin(a) * 70, -80 + 20 * abs(a), math.sin(a) * 90, -40 + 30 * abs(a))
            ctx.set_line_width(5)
            set_color(ctx, "#3F6E5A")
            ctx.stroke()
    elif kind == "can":
        ctx.rotate(1.3)
        rrect(ctx, -12, -26, 24, 52, 6)
        ctx.set_source(lin_grad(-12, 0, 12, 0, [(0, "#8A1E2E"), (0.4, "#D84A5A"), (1, "#6A1420")]))
        ctx.fill()
    ctx.restore()


def draw_debris(ctx, T):
    for d in debris_list():
        ctx.save()
        clip_above(ctx, d[2], T, amp=1.0, seed=d[4])
        draw_debris_item(ctx, T, *d)
        ctx.restore()
        hw = DEBRIS_HW.get(d[0], 40) * d[3]
        draw_meniscus(ctx, d[1] - hw, d[1] + hw, d[2], T, amp=1.0, seed=d[4], lift=2.5 * d[3], drop=3.5 * d[3],
                      alpha=0.85)


DEBRIS_HW = dict(pizza=88, card=48, mag=64, phone=38, plant=34, can=30)


def draw_floaters(ctx, T):
    """bits of the slop wave still bobbing in the flood: hearts, likes, badges, spinners, grins (never yellow)"""
    kinds = ("heart", "badge", "like", "spin", "grin", "heart", "badge", "blob")
    for k in range(34):
        u = hash01(k, 71)
        wl = WL + 14 + (1080 - WL) * (hash01(k, 72) ** 1.35)
        depth = (wl - lay.EYE_Y) / (WL - lay.EYE_Y)          # 1 at the wall, bigger nearer
        sc = 0.55 * depth
        x = -200 + 2300 * u + T * (6 + 8 * hash01(k, 73)) * depth
        if 700 < x < 1420 and wl < 700:
            continue     # keep the TV's reflection clean
        dy, r = bob(T, k * 1.3, 1.5 * sc, 0.12, 1.4)
        kind = kinds[k % len(kinds)]
        ctx.save()
        clip_above(ctx, wl, T, amp=0.8, seed=k, x0=x - 60 * sc, x1=x + 60 * sc)
        ctx.translate(x, wl + dy + 4 * sc)
        ctx.rotate(r)
        ctx.scale(sc, sc)
        if kind == "heart":
            ctx.move_to(0, 6)
            ctx.curve_to(-22, -8, -12, -26, 0, -14)
            ctx.curve_to(12, -26, 22, -8, 0, 6)
            ctx.close_path()
            ctx.set_source(lin_grad(0, -26, 0, 6, [(0, "#FF8AB0"), (1, "#C8406E")]))
            ctx.fill()
        elif kind == "badge":
            circle(ctx, 0, -8, 12)
            set_color(ctx, "#E0304A")
            ctx.fill()
            text(ctx, str(3 + k * 7 % 97), 0, -4, 11, "#FFFFFF", bold=True, align="center")
        elif kind == "like":
            rrect(ctx, -12, -18, 24, 22, 5)
            set_color(ctx, "#7FA8E0")
            ctx.fill()
            rrect(ctx, -4, -30, 9, 16, 4)
            ctx.fill()
        elif kind == "spin":
            ctx.set_line_width(4)
            for j in range(8):
                a = j * math.pi / 4 + T * 3
                ctx.move_to(math.cos(a) * 6, -10 + math.sin(a) * 6)
                ctx.line_to(math.cos(a) * 12, -10 + math.sin(a) * 12)
                set_color(ctx, "#D8DCE6", 0.25 + 0.75 * j / 8)
                ctx.stroke()
        elif kind == "grin":
            circle(ctx, 0, -6, 14)
            set_color(ctx, "#F0B8C8")
            ctx.fill()
            ctx.set_line_width(2.5)
            set_color(ctx, "#5A3040")
            ctx.arc(0, -8, 8, 0.3, math.pi - 0.3)
            ctx.stroke()
            circle(ctx, -5, -12, 1.8)
            circle(ctx, 5, -12, 1.8)
            ctx.fill()
        else:
            ellipse(ctx, 0, 0, 20, 10)
            ctx.set_source(lin_grad(0, -10, 0, 10, [(0, "#EFE4D0"), (1, "#B8AC94")]))
            ctx.fill()
        ctx.restore()
        draw_meniscus(ctx, x - 16 * sc, x + 16 * sc, wl, T, amp=0.8, seed=k, lift=1.5 * sc, drop=2.5 * sc, alpha=0.7)


def draw_debris_emit(ctx, T):
    for kind, x, wl, sc, seed, rot in debris_list():
        if kind != "phone":
            continue
        dy, r = bob(T, seed, 2.5 * sc, 0.05, 1.2)
        ctx.save()
        ctx.translate(x, wl + dy)
        ctx.rotate(r * 0.5 + rot * 0.2)
        ctx.scale(sc, sc)
        poly(ctx, [(-30, -5), (26, -10), (35, 1), (-22, 6)])
        a = 0.8 + 0.2 * math.sin(T * 4)
        set_color(ctx, "#B07CFF", a)
        ctx.fill()
        ctx.restore()


def draw_splash(ctx, T, t0, x0, x1, wl, n=18, seed=5, power=1.0):
    """slop kicked up where the TV slams down into the flood: a brief crown + ballistic gobs"""
    d = T - t0
    if d < 0 or d > 1.2:
        return
    # crown sheet along the front of the cabinet
    if d < 0.4:
        hgt = 34 * power * math.sin(math.pi * min(1.0, d / 0.4)) * (1 - d / 0.4) ** 0.3
        m = 24
        ctx.move_to(x0 - 20, wl + 4)
        for i in range(m + 1):
            x = x0 - 20 + (x1 - x0 + 40) * i / m
            spike = 0.55 + 0.45 * math.sin(i * 2.3 + seed) ** 2
            ctx.line_to(x, wl - hgt * spike)
        ctx.line_to(x1 + 20, wl + 4)
        ctx.close_path()
        ctx.set_source(lin_grad(0, wl - hgt, 0, wl + 4, [(0, "#F2E9D8", 0.85), (1, "#BFB39A", 0.95)]))
        ctx.fill()
    g = 1500.0
    for k in range(n):
        u = hash01(k, seed * 31)
        x = x0 + (x1 - x0) * u
        vx = (u - 0.5) * 380 + 40 * (hash01(k, seed * 31 + 1) - 0.5)
        vy = -(170 + 190 * hash01(k, seed * 31 + 2)) * power
        dd = d - 0.04 * hash01(k, seed * 31 + 3)
        if dd <= 0:
            continue
        px = x + vx * dd
        py = wl + vy * dd + 0.5 * g * dd * dd
        if py > wl + 2:
            continue
        r = 2.5 + 5.5 * hash01(k, seed * 31 + 4)
        sp = math.hypot(vx, vy + g * dd)
        ang = math.atan2(vy + g * dd, vx)
        ctx.save()
        ctx.translate(px, py)
        ctx.rotate(ang)
        ellipse(ctx, 0, 0, r * (1 + sp / 900), r)
        ctx.restore()
        ctx.set_source(rad_grad(px - r * 0.3, py - r * 0.4, 0.5, r * 1.6, [(0, "#FFF8EA"), (0.5, "#D9CDB4"), (1, "#A99D86")]))
        ctx.fill()


MOTES = [(hash01(k, 301), hash01(k, 302), hash01(k, 303)) for k in range(70)]


def draw_motes_emit(ctx, T, tv_xy, tv_rgb):
    """dust motes drifting through the TV's light"""
    tx, ty = tv_xy
    for k, (a, b, c) in enumerate(MOTES):
        x = tx - 520 + 1040 * a + 30 * noise1(T * 0.2 + k, 7) + 9 * T * (c - 0.5)
        y = ty - 330 + 520 * b + 24 * noise1(T * 0.17 + k * 3, 9) - 4 * T * c
        d2 = ((x - tx) ** 2 + ((y - ty) * 1.3) ** 2) / (360.0 ** 2)
        f = 1.0 / (1.0 + d2 * 3.0)
        if f < 0.08:
            continue
        tw = 0.6 + 0.4 * math.sin(T * (1.3 + 2 * c) + k)
        r = 1.1 + 1.6 * c
        col_ = tuple(min(1.0, v * f * tw * 0.9) for v in tv_rgb)
        circle(ctx, x, y, r * 2.2)
        ctx.set_source(rad_grad(x, y, 0, r * 2.2, [(0, col_), (1, (0, 0, 0))]))
        ctx.fill()

"""The anime gag: Summer as a shoujo-manga close-up (S04 'Huh? What you are saying is... its so weird....').
Bewildered, sparkly, sweat drop, blush hatching — never lewd.

manga_summer(ctx, x, y, k, *, t, mouth, look, blink, sparkle)  3/4 view facing right, head units like
    t02_cast (eye line y=0, crown -1, chin +1); (x, y) = head centre.
sparkles(ctx, pts, t)  twinkling 4-point 'kira' stars (white with ink keylines).
Owner: T02Wrong."""
import math

from .t02_brush import stroke, grey, WHITE, dots, poly_fill, path_curve, clip_curve, lock, catmull
from .t02_cast import begin, fill_curve, ellipse_path

M_FACE = [(0.44, -0.50), (0.60, -0.22), (0.63, 0.08), (0.60, 0.34), (0.50, 0.58), (0.36, 0.76), (0.24, 0.84),
          (0.10, 0.82), (-0.16, 0.66), (-0.38, 0.40), (-0.48, 0.05), (-0.50, -0.30), (-0.30, -0.62)]


def star4(ctx, x, y, r, rot=0.0, thin=0.14, key=True, lw=1.0):
    pts = []
    for i in range(8):
        a = rot + i * math.pi / 4
        rr = r if i % 2 == 0 else r * thin
        pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr))
    ctx.save()
    ctx.new_path()
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.set_source_rgb(1, 1, 1)
    if key:
        ctx.fill_preserve()
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(lw)
        ctx.stroke()
    else:
        ctx.fill()
    ctx.restore()


def sparkles(ctx, pts, t, lw=1.0):
    """pts = [(x, y, r, phase)]: stars twinkle (scale pulse) and slowly turn"""
    for i, (x, y, r, ph) in enumerate(pts):
        s = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * 5.0 + ph * 6.28))
        star4(ctx, x, y, r * s, rot=0.2 * math.sin(t + ph), lw=lw)


def _manga_eye(ctx, cx, cy, w, h, look, blink, lw, t, far=False):
    """huge glossy shoujo eye facing right"""
    op = max(0.0, 1.0 - blink)
    top = h * 0.55 * op
    bot = h * 0.50 * op
    up = [(cx - w * 0.50, cy - top * 0.35), (cx - w * 0.15, cy - top * 1.0), (cx + w * 0.25, cy - top * 1.02),
          (cx + w * 0.52, cy - top * 0.55)]
    lo = [(cx + w * 0.50, cy - top * 0.3 + h * 0.05), (cx + w * 0.30, cy + bot * 0.85), (cx - w * 0.10, cy + bot),
          (cx - w * 0.42, cy + bot * 0.55)]
    if op > 0.05:
        ctx.save()
        outline = catmull(up, False, 8).tolist() + catmull(lo, False, 8).tolist()
        ctx.new_path()
        ctx.move_to(*outline[0])
        for p in outline[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.clip()
        # huge iris: dark top, lighter bottom (screentone gradient), big pupil
        ix = cx + w * (0.06 + 0.12 * look[0])
        iy = cy + h * (0.05 + 0.10 * look[1])
        irx, iry = w * 0.36, h * 0.62
        ellipse_path(ctx, ix, iy, irx, iry)
        ctx.set_source_rgb(0.35, 0.35, 0.35)
        ctx.fill()
        ellipse_path(ctx, ix, iy + iry * 0.45, irx * 0.85, iry * 0.5)
        ctx.set_source_rgb(0.68, 0.68, 0.68)
        ctx.fill()
        ellipse_path(ctx, ix, iy - iry * 0.25, irx * 0.95, iry * 0.55)
        ctx.set_source_rgb(0.05, 0.05, 0.05)
        ctx.fill()
        ellipse_path(ctx, ix, iy, irx * 0.42, iry * 0.46)
        ctx.set_source_rgb(0, 0, 0)
        ctx.fill()
        ellipse_path(ctx, ix, iy, irx, iry)
        ctx.set_line_width(lw * 1.2)
        ctx.stroke()
        # highlights: big oval, small circle, a sparkle, wobbling (the 'uruuru' tearful gleam)
        wob = 0.02 * math.sin(t * 13)
        ellipse_path(ctx, ix - irx * 0.35 + wob, iy - iry * 0.42, irx * 0.34, iry * 0.26, -0.4)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill()
        ellipse_path(ctx, ix + irx * 0.38, iy + iry * 0.40 + wob, irx * 0.14, iry * 0.10)
        ctx.fill()
        star4(ctx, ix + irx * 0.25, iy - iry * 0.55, irx * 0.30 * (0.7 + 0.3 * math.sin(t * 7)), key=False)
        # glossy tear film along the lower lid
        stroke(ctx, [(cx - w * 0.3, cy + bot * 0.78), (cx + w * 0.3, cy + bot * 0.70)], lw * 1.2, color=WHITE,
               taper=(0.4, 0.4))
        ctx.restore()
    # heavy upper lash line with flicked lashes, thin lower line
    stroke(ctx, up, lw * 3.2, taper=(0.1, 0.15))
    if op > 0.05:
        ex, ey = up[-1]
        for i, (dx, dy) in enumerate(((0.22, -0.18), (0.28, -0.04), (0.20, 0.08))):
            stroke(ctx, [(ex - w * 0.05, ey + h * 0.05 * i), (ex + dx * w, ey + dy * h)], lw * 1.3, taper=(0.05, 0.95),
                   n=3)
        stroke(ctx, lo[1:], lw * 0.8, taper=(0.4, 0.3))
        for i in range(3):
            u = 0.25 + i * 0.22
            p = lo[1][0] + (lo[3][0] - lo[1][0]) * u, lo[1][1] + (lo[3][1] - lo[1][1]) * u + h * 0.03
            stroke(ctx, [p, (p[0] - w * 0.03, p[1] + h * 0.12)], lw * 0.6, taper=(0.1, 0.9))


def manga_summer(ctx, x, y, k, *, t=0.0, mouth=0.0, look=(0.45, -0.05), blink=0.0, tilt=0.0, lw=None, sweat=1.0,
                 blush=1.0, flip=False):
    lw = (lw if lw is not None else 0.016 * k) / k
    begin(ctx, x, y, k, tilt, flip)
    # --- back hair + pigtail (glossy, spiky)
    back = [(-0.40, 0.70), (-0.80, 0.30), (-0.98, -0.25), (-0.82, -0.78), (-0.40, -1.10), (0.10, -1.18),
            (0.52, -0.98), (0.72, -0.60), (0.66, -0.30), (0.3, -0.3)]
    fill_curve(ctx, back, grey(0.84))
    rx, ry = -0.86, 0.05
    sw = 0.04 * math.sin(t * 2.2)
    for i in range(5):
        o = (i - 2) * 0.07
        lock(ctx, [(rx + o * 0.4, ry), (rx - 0.22 + o, ry + 0.40), (rx - 0.30 + sw + o * 1.3, ry + 0.90),
                   (rx - 0.12 + sw + o * 0.6, ry + 1.45 - abs(o) * 1.5)], 0.14, line=lw * 1.1, dark=grey(0.8),
             shadow_side=-1, root=0.1, tip=0.35)
    ctx.save()
    clip_curve(ctx, back, 10)
    fill_curve(ctx, [(-1.1, -0.5), (-0.62, -0.55), (-0.60, 0.0), (-0.40, 0.6), (-1.1, 0.7)], grey(0.60), 6)
    # the glossy 'angel ring' highlight band: white zigzag across the crown
    band = []
    for i in range(13):
        a = math.radians(-160 + i * 12)
        r = 0.80 + (0.07 if i % 2 else -0.02)
        band.append((-0.12 + math.cos(a) * r, -0.20 + math.sin(a) * r))
    for i in range(12, -1, -1):
        a = math.radians(-160 + i * 12)
        r = 0.66 + (0.05 if i % 2 else -0.03)
        band.append((-0.12 + math.cos(a) * r, -0.20 + math.sin(a) * r))
    poly_fill(ctx, band, grey(1.0))
    ctx.restore()
    stroke(ctx, back[1:9], lw * 2.0, taper=(0.1, 0.2))
    # hair ties: a big ribbon bow (manga!)
    for sx_, a in ((-1, 0.5), (1, -0.4)):
        bow = [(rx, ry), (rx + sx_ * 0.18, ry - 0.14), (rx + sx_ * 0.20, ry + 0.12)]
        poly_fill(ctx, bow, grey(0.2))
        stroke(ctx, bow + [bow[0]], lw, taper=(0, 0))
    dots(ctx, [(rx, ry)], 0.045, grey(0.1))
    # --- neck + collar
    neck = [(-0.20, 0.60), (-0.26, 1.40), (0.24, 1.42), (0.20, 0.80)]
    fill_curve(ctx, neck, WHITE, 4)
    ctx.save()
    clip_curve(ctx, neck, 4)
    fill_curve(ctx, [(-0.4, 0.5), (0.4, 0.8), (0.4, 1.0), (-0.4, 1.02)], grey(0.78), 6)
    ctx.restore()
    stroke(ctx, [(-0.21, 0.62), (-0.26, 1.40)], lw * 1.3, taper=(0.1, 0.3))
    stroke(ctx, [(0.20, 0.82), (0.24, 1.40)], lw * 1.1, taper=(0.1, 0.3))
    fur = []
    for i in range(29):
        a = i / 28
        xx = -1.1 + 2.2 * a
        yy = 1.30 + 0.12 * (2 * a - 1) ** 2 - (0.08 if i % 2 else 0.0)
        fur.append((xx, yy))
    fur += [(1.1, 1.7), (-1.1, 1.7)]
    poly_fill(ctx, fur, grey(0.9))
    stroke(ctx, fur[:29], lw * 1.2, taper=(0.1, 0.1))
    # --- face: round, small chin
    fill_curve(ctx, M_FACE, WHITE)
    ctx.save()
    clip_curve(ctx, M_FACE, 10)
    fill_curve(ctx, [(0.40, 0.1), (0.52, 0.30), (0.40, 0.58), (0.20, 0.9), (0.7, 0.9), (0.7, 0.1)], grey(0.86), 6)
    if blush > 0:
        # manga blush: a soft tone patch + diagonal hatching across the cheeks and nose
        for bx, by, bw, nl in ((0.06, 0.50, 0.22, 4), (0.50, 0.47, 0.09, 2)):
            ellipse_path(ctx, bx, by, bw, bw * 0.40)
            ctx.set_source_rgba(0.82, 0.82, 0.82, blush)
            ctx.fill()
            for j in range(nl):
                hx = bx - bw * 0.55 + j * bw * 1.1 / max(1, nl - 1)
                stroke(ctx, [(hx - 0.025, by + 0.035), (hx + 0.025, by - 0.035)], lw * 1.1, taper=(0.3, 0.3),
                       alpha=blush)
    ctx.restore()
    stroke(ctx, M_FACE[:9], lw * 1.6, taper=(0.1, 0.3))
    # --- eyes (huge), brows raised in bewilderment
    _manga_eye(ctx, 0.05, 0.14, 0.50, 0.56, look, blink, lw, t)
    _manga_eye(ctx, 0.47, 0.13, 0.24, 0.50, look, blink, lw, t, far=True)
    stroke(ctx, [(-0.12, -0.30), (0.06, -0.40), (0.26, -0.37)], lw * 1.1, taper=(0.3, 0.5))
    stroke(ctx, [(0.40, -0.37), (0.48, -0.41), (0.57, -0.36)], lw * 0.9, taper=(0.4, 0.4))
    # tiny nose tick + small mouth (lip-synced 'huh?')
    stroke(ctx, [(0.47, 0.44), (0.49, 0.47), (0.46, 0.48)], lw * 0.9, taper=(0.3, 0.3))
    mo = max(0.0, mouth)
    if mo < 0.08:
        stroke(ctx, [(0.29, 0.64), (0.34, 0.655), (0.38, 0.64)], lw * 1.2, taper=(0.3, 0.3))
    else:
        mp = [(0.28, 0.63), (0.335, 0.62 - 0.01 * mo), (0.385, 0.63), (0.37, 0.64 + 0.09 * mo), (0.30, 0.645 + 0.10 * mo)]
        ctx.save()
        ctx.new_path()
        path_curve(ctx, mp, True, 6)
        ctx.set_source_rgb(0.12, 0.12, 0.12)
        ctx.fill_preserve()
        ctx.clip()
        ellipse_path(ctx, 0.335, 0.66 + 0.08 * mo, 0.035, 0.03 * mo + 0.008)
        ctx.set_source_rgb(0.6, 0.6, 0.6)
        ctx.fill()
        ctx.restore()
        ctx.save()
        ctx.new_path()
        path_curve(ctx, mp, True, 6)
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(lw * 1.1)
        ctx.stroke()
        ctx.restore()
    # --- bangs: spiky manga locks with white gloss cuts
    tips = [(-0.30, -0.16), (-0.10, -0.08), (0.10, -0.14), (0.30, -0.06), (0.48, -0.16), (0.64, -0.26)]
    for i, (bx, by) in enumerate(tips):
        base = -0.05 + (bx - 0.15) * 0.4
        s_ = 0.03 * math.sin(t * 2.5 + i)
        lock(ctx, [(base, -1.05), (base + (bx - base) * 0.45 - 0.04, -0.62), (bx - 0.03 + s_, -0.28), (bx + s_, by - 0.08)],
             0.24 if i % 2 == 0 else 0.20, line=lw * 1.3, dark=None, shadow_side=1, root=0.05, tip=0.45)
    for i in range(4):
        gx = -0.1 + i * 0.2
        poly_fill(ctx, [(gx, -0.80), (gx + 0.05, -0.80), (gx + 0.03, -0.60)], WHITE)
    # --- the classic giant sweat drop
    if sweat > 0:
        sx, sy, r = -0.62, -0.62 + 0.05 * math.sin(t * 2.0), 0.16 * sweat
        pts = [(sx, sy - r * 1.8), (sx + r * 0.8, sy + r * 0.2), (sx, sy + r), (sx - r * 0.8, sy + r * 0.2)]
        ctx.save()
        ctx.new_path()
        path_curve(ctx, pts, True, 8)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(lw * 1.6)
        ctx.stroke()
        ctx.restore()
        stroke(ctx, [(sx - r * 0.4, sy - r * 0.2), (sx - r * 0.35, sy + r * 0.45)], lw * 1.4, taper=(0.3, 0.3))
    ctx.restore()

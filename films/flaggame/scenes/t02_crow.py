"""Dr. Beelzebub Crow, playa evangelist — Chick-tract close-up art (grey tones -> halftone; black = ink).

crow(ctx, x, y, k, **p)       head-and-shoulders, 3/4 view facing LEFT by default (flip=True);
                              head units as t02_cast (eye line y=0, crown -1, chin +1), (x, y) = head centre.
point_arm(ctx, x, y, k, ...)  his pointing arm (coat sleeve + hand, index finger out) in the same head units,
                              from shoulder to hand; draw after the bust.
Owner: T02Wrong."""
import math

import numpy as np

from .t02_brush import stroke, shape, hatch, dots, clip_curve, path_curve, grey, WHITE, poly_fill
from .t02_cast import begin, fill_curve, eye, mouth, LIGHT

SKIN = grey(1.0)
SKIN_SH = grey(0.82)
SKIN_SH2 = grey(0.62)
BEARD = grey(0.48)
COAT = grey(0.0)
FUR = grey(0.66)

C_FACE = [(0.56, -0.50), (0.68, -0.26), (0.72, -0.10), (0.66, 0.04), (0.72, 0.16), (0.80, 0.30), (0.86, 0.40),
          (0.80, 0.47), (0.70, 0.48), (0.72, 0.58), (0.72, 0.76), (0.64, 0.96), (0.40, 1.08), (0.08, 1.04),
          (-0.28, 0.86), (-0.48, 0.58), (-0.52, 0.20), (-0.52, -0.20), (-0.30, -0.56)]
C_CONTOUR = C_FACE[1:15]
PINS = [  # (x, y, r, style, tone) on the cap's front panel, head units
    (0.10, -1.02, 0.085, "ring", 0.3), (0.30, -0.95, 0.07, "star", 0.9), (0.48, -0.84, 0.075, "dot", 0.55),
    (-0.08, -0.90, 0.07, "smile", 0.85), (0.14, -0.80, 0.06, "bars", 0.45), (0.36, -0.72, 0.065, "ring", 0.75),
    (0.56, -0.64, 0.06, "star", 0.35), (-0.20, -0.74, 0.08, "spiral", 0.6), (-0.02, -0.66, 0.055, "dot", 0.2),
    (0.20, -0.62, 0.06, "smile", 0.6), (-0.30, -0.98, 0.06, "bars", 0.8), (0.42, -1.04, 0.05, "dot", 0.9),
    (-0.14, -1.12, 0.055, "ring", 0.5), (0.64, -0.76, 0.045, "dot", 0.4),
]


def _pin(ctx, x, y, r, style, tone, lw, glint=0.0):
    ctx.save()
    ctx.new_path()
    ctx.arc(x, y, r, 0, 2 * math.pi)
    ctx.set_source_rgb(tone, tone, tone)
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(lw * 0.8)
    ctx.stroke()
    ink = (0, 0, 0) if tone > 0.5 else (1, 1, 1)
    ctx.set_source_rgb(*ink)
    ctx.set_line_width(lw * 0.5)
    if style == "ring":
        ctx.new_path()
        ctx.arc(x, y, r * 0.6, 0, 2 * math.pi)
        ctx.stroke()
        ctx.new_path()
        ctx.arc(x, y, r * 0.22, 0, 2 * math.pi)
        ctx.fill()
    elif style == "star":
        ctx.new_path()
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            rr = r * (0.7 if i % 2 == 0 else 0.3)
            (ctx.move_to if i == 0 else ctx.line_to)(x + math.cos(a) * rr, y + math.sin(a) * rr)
        ctx.close_path()
        ctx.fill()
    elif style == "dot":
        ctx.new_path()
        ctx.arc(x - r * 0.25, y - r * 0.25, r * 0.28, 0, 2 * math.pi)
        ctx.fill()
    elif style == "smile":
        dots(ctx, [(x - r * 0.3, y - r * 0.2), (x + r * 0.3, y - r * 0.2)], r * 0.12, ink)
        ctx.new_path()
        ctx.arc(x, y + r * 0.05, r * 0.45, 0.3, math.pi - 0.3)
        ctx.stroke()
    elif style == "bars":
        for i in range(3):
            yy = y - r * 0.4 + i * r * 0.4
            ctx.move_to(x - r * 0.55, yy)
            ctx.line_to(x + r * 0.55, yy)
        ctx.stroke()
    elif style == "spiral":
        ctx.new_path()
        for i in range(40):
            a = i * 0.45
            rr = r * 0.75 * i / 40
            (ctx.move_to if i == 0 else ctx.line_to)(x + math.cos(a) * rr, y + math.sin(a) * rr)
        ctx.stroke()
    ctx.restore()
    if glint > 0.02:
        _sparkle(ctx, x - r * 0.35, y - r * 0.35, r * 1.6 * glint, lw)


def _sparkle(ctx, x, y, s, lw):
    """a white 4-point star with an ink keyline (reads on any tone)"""
    pts = []
    for i in range(8):
        a = i * math.pi / 4
        rr = s if i % 2 == 0 else s * 0.16
        pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr))
    ctx.save()
    ctx.new_path()
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(lw * 0.45)
    ctx.stroke()
    ctx.restore()


def crow(ctx, x, y, k, *, t=0.0, tilt=0.0, look=(0.3, 0.0), mouth_open=0.0, mshape="talk", brow=-0.6,
         eye_open=1.0, pupil=0.9, lw=None, flip=True, body=True, glint=0.0, grin=0.0, lens=0.62, lens_glint=1.0,
         sweat=0, glasses_down=0.0, sx=1.0):
    """brow: -1 fervent frown .. 0 .. 1 raised. glint: 0..1 pin sparkle strength (animated with t).
    lens: tint of the aviators (0 clear .. 1 opaque). glasses_down: 0..1 slides the aviators down the nose
    (conspiratorial peek over the rims)."""
    lw = (lw if lw is not None else 0.02 * k) / k
    begin(ctx, x, y, k, tilt, flip, sx)
    if body:
        _coat(ctx, t, lw)
    # hair at the back of the head / nape (grizzled, collar length)
    nape = [(-0.30, -0.30), (-0.84, -0.28), (-0.96, 0.20), (-0.86, 0.70), (-0.40, 0.84), (-0.30, 0.30)]
    fill_curve(ctx, nape, grey(0.86))
    for i in range(12):
        a = i / 11
        x0 = -0.86 + a * 0.46
        stroke(ctx, [(x0, -0.24), (x0 - 0.06, 0.2), (x0 + 0.02 - 0.1 * a, 0.74)], lw * (0.9 if i % 3 else 0.5),
               taper=(0.3, 0.5), color=WHITE if i % 4 == 0 else (0, 0, 0))
    stroke(ctx, nape[1:5], lw * 1.4, taper=(0.1, 0.4))
    # neck
    neck = [(-0.36, 0.60), (-0.46, 1.40), (0.46, 1.44), (0.40, 0.92)]
    fill_curve(ctx, neck, SKIN, 4)
    ctx.save()
    clip_curve(ctx, neck, 4)
    fill_curve(ctx, [(-0.6, 0.5), (0.2, 1.0), (0.5, 0.9), (0.5, 1.12), (0.0, 1.2), (-0.6, 1.0)], SKIN_SH, 6)
    ctx.restore()
    stroke(ctx, [(0.40, 0.95), (0.44, 1.40)], lw * 1.2, taper=(0.1, 0.4))
    stroke(ctx, [(-0.36, 0.64), (-0.44, 1.0), (-0.46, 1.40)], lw * 1.3, taper=(0.1, 0.4))
    # face
    fill_curve(ctx, C_FACE, SKIN)
    ctx.save()
    clip_curve(ctx, C_FACE, 10)
    fill_curve(ctx, [(0.66, -0.2), (0.64, 0.06), (0.68, 0.22), (0.62, 0.5), (0.62, 0.8), (0.56, 1.0), (1.0, 1.0),
                     (1.0, -0.2)], SKIN_SH, 8)
    fill_curve(ctx, [(-0.6, 0.3), (-0.38, 0.56), (-0.12, 0.84), (0.1, 1.1), (-0.6, 1.1)], SKIN_SH, 6)
    # cap brim shadow: a thin band under the bill
    fill_curve(ctx, [(-0.6, -0.6), (0.9, -0.6), (0.9, -0.14), (0.3, -0.18), (-0.6, -0.22)], SKIN_SH, 6)
    # stubble along the jaw
    rng = np.random.default_rng(5)
    st = [(rng.uniform(-0.40, 0.30), rng.uniform(0.50, 0.98)) for _ in range(90)]
    st = [p for p in st if p[1] > 0.55 + 0.5 * max(0, -p[0] - 0.1)]
    dots(ctx, st, 0.0065)
    ctx.restore()
    stroke(ctx, C_CONTOUR, lw * 1.7, taper=(0.1, 0.3), light=LIGHT, lk=0.5)
    # ear + grey sideburn
    ear = [(-0.40, 0.00), (-0.30, -0.05), (-0.25, 0.10), (-0.27, 0.28), (-0.34, 0.34), (-0.42, 0.24)]
    shape(ctx, ear, SKIN, line=lw * 1.2)
    stroke(ctx, [(-0.36, 0.04), (-0.29, 0.10), (-0.31, 0.24)], lw * 0.9, taper=(0.3, 0.3))
    stroke(ctx, [(-0.34, 0.12), (-0.36, 0.20)], lw * 1.4, taper=(0.3, 0.3))
    sb = [(-0.16, -0.40), (-0.02, -0.40), (-0.08, 0.10), (-0.14, 0.30), (-0.18, -0.1)]
    fill_curve(ctx, sb, grey(0.72))
    for i in range(6):
        stroke(ctx, [(-0.14 + i * 0.02, -0.38), (-0.12 + i * 0.012, 0.0), (-0.15 + i * 0.01, 0.22)], lw * 0.6,
               taper=(0.3, 0.5))
    # wrinkles: nasolabial folds, crow's feet
    stroke(ctx, [(0.62, 0.44), (0.54, 0.58), (0.46, 0.76)], lw * 1.3, taper=(0.2, 0.6))
    stroke(ctx, [(0.26, 0.36), (0.30, 0.52), (0.24, 0.72)], lw * 1.0, taper=(0.4, 0.6))
    stroke(ctx, [(0.08, 0.30), (0.20, 0.36), (0.32, 0.34)], lw * 0.7, taper=(0.4, 0.4))
    for i in range(3):
        stroke(ctx, [(-0.06, -0.02 + i * 0.06), (-0.14, -0.04 + i * 0.08)], lw * 0.5, taper=(0.2, 0.7))
    # brows (bushy, grizzled) — fervent: inner ends pulled down
    b = brow
    for (x0, x1, yb, wmul) in ((0.00, 0.40, -0.20, 1.0), (0.48, 0.70, -0.20, 0.7)):
        pts = [(x0, yb - 0.02 + 0.03 * b), ((x0 + x1) / 2, yb - 0.06 - 0.05 * b), (x1, yb + 0.02 - 0.10 * b)]
        if x0 > 0.4:
            pts = [(x0, yb + 0.03 - 0.10 * b), ((x0 + x1) / 2, yb - 0.06 - 0.04 * b), (x1, yb - 0.01 + 0.02 * b)]
        stroke(ctx, pts, lw * 3.4 * wmul, taper=(0.3, 0.4), color=grey(0.35))
        for j in range(7):
            u = j / 6
            px = pts[0][0] + (pts[-1][0] - pts[0][0]) * u
            py = pts[0][1] + (pts[-1][1] - pts[0][1]) * u - 0.03 * math.sin(u * math.pi)
            stroke(ctx, [(px - 0.03, py + 0.025), (px + 0.03, py - 0.03)], lw * 0.7, taper=(0.2, 0.6),
                   color=WHITE if j % 3 == 1 else (0, 0, 0))
    # eyes (behind the lenses)
    gd = glasses_down
    eye(ctx, 0.20, 0.03, 0.28, 0.14, openk=eye_open, look=look, pupil=pupil, lw=lw, lash=0.0, iris_c=0.35)
    eye(ctx, 0.585, 0.02, 0.15, 0.13, openk=eye_open, look=look, pupil=pupil, lw=lw, lash=0.0, iris_c=0.35)
    # nose (big, a bit bulbous)
    stroke(ctx, [(0.58, 0.10), (0.63, 0.24), (0.66, 0.34)], lw * 0.8, taper=(0.6, 0.3))
    stroke(ctx, [(0.80, 0.44), (0.72, 0.47), (0.64, 0.45), (0.60, 0.40)], lw * 1.3, taper=(0.2, 0.4))
    shape(ctx, [(0.63, 0.40), (0.60, 0.43), (0.64, 0.45)], grey(0.1))
    # moustache + goatee (Van Dyke), grizzled
    _beard(ctx, t, lw, mouth_open, mshape, grin)
    # aviators
    _aviators(ctx, lw, lens, lens_glint, gd, t)
    # trucker cap
    ctx.save()
    ctx.translate(0.0, 0.24)
    ctx.scale(1.06, 1.0)
    _cap(ctx, t, lw, glint)
    ctx.restore()
    for i in range(int(sweat)):
        from .t02_cast import drop
        sx_, sy_ = [(0.86, -0.20), (-0.50, -0.30)][i % 2]
        drop(ctx, sx_, sy_ + 0.02 * math.sin(t * 3 + i), 0.07, lw)
    ctx.restore()


def _beard(ctx, t, lw, mo, mshape, grin):
    # goatee + chin
    goat = [(0.34, 0.74), (0.56, 0.70), (0.70, 0.78), (0.64, 1.00), (0.50, 1.20), (0.40, 1.26), (0.30, 1.12),
            (0.22, 0.92)]
    fill_curve(ctx, goat, BEARD)
    ctx.save()
    clip_curve(ctx, goat, 8)
    fill_curve(ctx, [(0.2, 1.0), (0.7, 0.95), (0.6, 1.3), (0.2, 1.3)], grey(0.5), 6)
    rng = np.random.default_rng(9)
    for i in range(46):
        px = rng.uniform(0.24, 0.68)
        py = rng.uniform(0.74, 1.22)
        ln = 0.06 + rng.random() * 0.06
        stroke(ctx, [(px, py), (px - 0.01 + (0.45 - px) * 0.2, py + ln)], lw * (0.7 if i % 3 else 1.0),
               taper=(0.2, 0.7), color=WHITE if i % 3 == 0 else (0, 0, 0))
    ctx.restore()
    stroke(ctx, goat[1:7], lw * 1.2, taper=(0.2, 0.3))
    # mouth (opens inside the beard)
    mouth(ctx, 0.50, 0.65, 0.40, mo, lw=lw, shape_=mshape, smile=0.2 + grin, far_k=0.7, lips=0.0)
    # moustache
    ms = [(0.30, 0.64), (0.44, 0.53), (0.62, 0.50), (0.76, 0.56), (0.80, 0.66), (0.70, 0.62), (0.56, 0.60),
          (0.42, 0.63)]
    lift = 0.05 * mo
    ms = [(px, py - lift * (0.5 if 0.4 < px < 0.7 else 0.2)) for px, py in ms]
    fill_curve(ctx, ms, BEARD)
    for i in range(12):
        u = i / 11
        px = 0.32 + u * 0.46
        py = 0.56 + 0.02 * math.sin(u * 3) - lift * 0.4
        stroke(ctx, [(px, py - 0.02), (px + 0.01 + (u - 0.5) * 0.06, py + 0.07)], lw * 0.8, taper=(0.2, 0.7),
               color=WHITE if i % 3 == 0 else (0, 0, 0))
    stroke(ctx, ms[:5], lw * 1.0, taper=(0.3, 0.3))


def _aviators(ctx, lw, lens, glint, down, t):
    dy = down * 0.22
    lenses = [
        [(0.02, -0.08 + dy), (0.20, -0.12 + dy), (0.40, -0.08 + dy), (0.40, 0.10 + dy), (0.30, 0.25 + dy),
         (0.12, 0.24 + dy), (0.02, 0.10 + dy)],
        [(0.50, -0.09 + dy), (0.62, -0.12 + dy), (0.72, -0.08 + dy), (0.73, 0.10 + dy), (0.68, 0.23 + dy),
         (0.56, 0.22 + dy), (0.50, 0.08 + dy)],
    ]
    for L in lenses:
        ctx.save()
        ctx.new_path()
        path_curve(ctx, L, True, 8)
        ctx.set_source_rgba(0.32, 0.32, 0.32, lens * 0.85)
        ctx.fill_preserve()
        ctx.clip()
        # darker gradient band at the top
        fill_curve(ctx, [(-0.1, -0.2 + dy), (0.9, -0.2 + dy), (0.9, 0.02 + dy), (-0.1, 0.06 + dy)], grey(0.2), 4,
                   alpha=lens * 0.7)
        if glint > 0:
            x0 = L[0][0]
            w = L[3][0] - x0
            gp = 0.35 + 0.25 * math.sin(t * 0.7)
            poly_fill(ctx, [(x0 + w * (gp - 0.10), 0.30 + dy), (x0 + w * (gp + 0.08), 0.30 + dy),
                            (x0 + w * (gp + 0.38), -0.2 + dy), (x0 + w * (gp + 0.20), -0.2 + dy)], WHITE, glint * 0.95)
            poly_fill(ctx, [(x0 + w * (gp + 0.14), 0.30 + dy), (x0 + w * (gp + 0.19), 0.30 + dy),
                            (x0 + w * (gp + 0.49), -0.2 + dy), (x0 + w * (gp + 0.44), -0.2 + dy)], WHITE, glint * 0.95)
        ctx.restore()
        stroke(ctx, L + [L[0]], lw * 1.0, taper=(0, 0))
    # double bridge + temple arm
    stroke(ctx, [(0.40, -0.07 + dy), (0.45, -0.10 + dy), (0.50, -0.08 + dy)], lw * 1.0, taper=(0, 0))
    stroke(ctx, [(0.39, 0.03 + dy), (0.45, 0.0 + dy), (0.51, 0.03 + dy)], lw * 0.8, taper=(0, 0))
    stroke(ctx, [(0.02, -0.05 + dy), (-0.18, -0.06 + dy * 0.5), (-0.30, -0.02)], lw * 1.1, taper=(0, 0.3))


def _cap(ctx, t, lw, glint):
    # mesh back
    mesh = [(-0.30, -0.56), (-0.86, -0.46), (-0.98, -0.82), (-0.76, -1.14), (-0.30, -1.22), (-0.10, -0.90)]
    fill_curve(ctx, mesh, grey(0.85))
    ctx.save()
    clip_curve(ctx, mesh, 8)
    hatch(ctx, (-1.0, -1.3, -0.1, -0.4), 0.8, 0.045, lw * 0.5, seed=1, jitter=0.0, lenvar=0.0)
    hatch(ctx, (-1.0, -1.3, -0.1, -0.4), -0.8, 0.045, lw * 0.5, seed=2, jitter=0.0, lenvar=0.0)
    ctx.restore()
    stroke(ctx, mesh[:5], lw * 1.4, taper=(0.1, 0.1))
    # tall foam front panel
    front = [(-0.30, -0.54), (-0.36, -0.95), (-0.10, -1.24), (0.30, -1.20), (0.60, -1.00), (0.74, -0.56),
             (0.30, -0.50)]
    fill_curve(ctx, front, grey(0.92))
    ctx.save()
    clip_curve(ctx, front, 8)
    fill_curve(ctx, [(-0.5, -1.3), (-0.1, -1.3), (-0.2, -0.8), (-0.3, -0.4), (-0.5, -0.4)], grey(0.72), 6)
    ctx.restore()
    stroke(ctx, front[:6], lw * 1.6, taper=(0.1, 0.1), light=LIGHT)
    # patch (stitched)
    patch = [(-0.30, -0.64), (-0.08, -0.66), (-0.06, -0.52), (-0.30, -0.50)]
    poly_fill(ctx, patch, grey(0.35))
    ctx.save()
    ctx.set_source_rgb(1, 1, 1)
    ctx.set_line_width(lw * 0.5)
    ctx.set_dash([0.02, 0.015])
    ctx.move_to(-0.28, -0.62)
    ctx.line_to(-0.09, -0.63)
    ctx.line_to(-0.08, -0.53)
    ctx.line_to(-0.28, -0.52)
    ctx.close_path()
    ctx.stroke()
    ctx.restore()
    # pins
    for i, (px, py, r, st, tone) in enumerate(PINS):
        g = 0.0
        if glint > 0:
            ph = (t * 0.9 + i * 0.37) % 1.0
            g = glint * max(0.0, 1 - abs(ph - 0.5) * 6) if i % 3 == 0 else 0.0
        _pin(ctx, px, py, r, st, tone, lw, g)
    # button on top
    shape(ctx, [(0.00, -1.27), (0.08, -1.26), (0.08, -1.21), (0.0, -1.21)], grey(0.5), line=lw * 0.8)
    # bill: curved brim toward the face side, underside in shadow
    bill = [(-0.30, -0.52), (0.30, -0.50), (0.78, -0.56), (1.14, -0.44), (1.18, -0.36), (0.80, -0.38),
            (0.30, -0.38), (-0.20, -0.44)]
    fill_curve(ctx, bill, grey(0.55))
    ctx.save()
    clip_curve(ctx, bill, 8)
    fill_curve(ctx, [(-0.4, -0.44), (1.3, -0.46), (1.3, -0.3), (-0.4, -0.3)], grey(0.12), 4)
    stroke(ctx, [(-0.1, -0.49), (0.6, -0.52), (1.06, -0.44)], lw * 0.5, taper=(0.2, 0.2), color=WHITE)
    ctx.restore()
    stroke(ctx, bill + [bill[0]], lw * 1.5, taper=(0, 0))


def _coat(ctx, t, lw):
    coat = [(-1.7, 2.8), (-1.55, 1.70), (-0.90, 1.30), (0.0, 1.32), (0.90, 1.28), (1.55, 1.62), (1.75, 2.8)]
    fill_curve(ctx, coat, COAT, 6)
    ctx.save()
    clip_curve(ctx, coat, 6)
    # rim light + folds in white brush
    stroke(ctx, [(-1.45, 1.8), (-1.5, 2.2), (-1.55, 2.8)], lw * 2.2, color=grey(0.8), taper=(0.2, 0.2))
    for fx in (-0.9, -0.5, 0.6, 1.1):
        stroke(ctx, [(fx, 1.7), (fx + 0.05, 2.1), (fx + 0.02, 2.8)], lw * 1.0, color=grey(0.85), taper=(0.3, 0.6))
    # lapel edge + a sheaf of tracts poking out of the breast pocket
    stroke(ctx, [(0.30, 1.50), (0.45, 2.0), (0.50, 2.8)], lw * 1.4, color=(0, 0, 0), taper=(0.1, 0.1))
    poly_fill(ctx, [(0.62, 2.12), (1.08, 2.10), (1.08, 2.8), (0.62, 2.8)], grey(0.15))
    stroke(ctx, [(0.62, 2.12), (1.08, 2.10)], lw * 1.0, color=grey(0.6), taper=(0, 0))
    ctx.restore()
    stroke(ctx, coat, lw * 1.8, taper=(0, 0))
    # shaggy fur collar
    rng = np.random.default_rng(21)
    n = 46
    top, bot = [], []
    for i in range(n + 1):
        a = i / n
        xx = -1.30 + 2.6 * a
        base = 1.30 + 0.16 * (2 * a - 1) ** 2
        top.append((xx, base - 0.10 - 0.10 * rng.random() * (1 if i % 2 else 0.3)))
        bot.append((xx, base + 0.34 + 0.10 * rng.random() * (1 if i % 2 else 0.3) - 0.08 * math.sin(a * math.pi)))
    ctx.save()
    ctx.new_path()
    ctx.move_to(*top[0])
    for p in top[1:]:
        ctx.line_to(*p)
    for p in bot[::-1]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.set_source_rgb(*FUR)
    ctx.fill_preserve()
    ctx.clip()
    fill_curve(ctx, [(-1.4, 1.5), (1.4, 1.5), (1.4, 1.9), (-1.4, 1.9)], grey(0.45), 4)
    for i in range(90):
        px = rng.uniform(-1.3, 1.3)
        py = rng.uniform(1.18, 1.8)
        stroke(ctx, [(px, py), (px + rng.normal() * 0.03, py + 0.12)], lw * (0.9 if i % 2 else 0.6), taper=(0.1, 0.9),
               color=WHITE if i % 4 == 0 else (0, 0, 0))
    ctx.restore()
    for i in range(0, n + 1, 2):
        x0, y0 = top[i]
        stroke(ctx, [(x0, y0), (x0 + 0.02, y0 + 0.10)], lw * 0.9, taper=(0.1, 0.9))
        x1, y1 = bot[i]
        stroke(ctx, [(x1, y1 - 0.12), (x1 - 0.01, y1)], lw * 0.9, taper=(0.1, 0.9))


def _limb(ctx, p0, p1, w0, w1, lw, color=None, folds=3):
    ax, ay = p0
    bx, by = p1
    L = math.hypot(bx - ax, by - ay) or 1e-6
    ca, sa = (bx - ax) / L, (by - ay) / L
    nx, ny = -sa, ca
    poly = [(ax + nx * w0 / 2, ay + ny * w0 / 2), (bx + nx * w1 / 2, by + ny * w1 / 2),
            (bx - nx * w1 / 2, by - ny * w1 / 2), (ax - nx * w0 / 2, ay - ny * w0 / 2)]
    poly_fill(ctx, poly, color if color is not None else COAT)
    stroke(ctx, [poly[0], poly[1]], lw * 1.8, taper=(0.05, 0.05))
    stroke(ctx, [poly[3], poly[2]], lw * 1.8, taper=(0.05, 0.05))
    for i in range(folds):
        f = (i + 1) / (folds + 1)
        px, py = ax + (bx - ax) * f, ay + (by - ay) * f
        ww = (w0 + (w1 - w0) * f) * 0.5
        stroke(ctx, [(px + nx * ww * 0.85, py + ny * ww * 0.85), (px + ca * 0.06, py + sa * 0.06),
                     (px - nx * ww * 0.2, py - ny * ww * 0.2)], lw * 0.9, color=grey(0.75), taper=(0.2, 0.6))
    return ca, sa, nx, ny


def point_arm(ctx, x, y, k, *, sh=(0.30, 1.75), hand=(2.1, 1.0), lw=None, flip=True, t=0.0, shake=0.0, tilt=0.0,
              bend=1.0, finger="point"):
    """the Chick-tract witness's arm, in crow() head units (same x, y, k, flip as the bust): a coat sleeve in two
    segments (2-bone IK, elbow bending down/out), fur cuff, and a hand with the index finger out along the
    forearm. sh = shoulder, hand = wrist target. Returns the fingertip in the caller's space."""
    lw = (lw if lw is not None else 0.02 * k) / k
    begin(ctx, x, y, k, tilt, flip)
    hx, hy = hand
    hx += shake * 0.02 * math.sin(t * 40)
    hy += shake * 0.02 * math.cos(t * 37)
    l1, l2 = 1.05, 1.00
    dx, dy = hx - sh[0], hy - sh[1]
    d = min(math.hypot(dx, dy), (l1 + l2) * 0.999)
    base = math.atan2(dy, dx)
    cosA = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    A = math.acos(max(-1.0, min(1.0, cosA)))
    ea = base + A * bend                       # elbow swings to the lower side of the reach
    ex, ey = sh[0] + math.cos(ea) * l1, sh[1] + math.sin(ea) * l1
    wx_, wy_ = ex + (hx - ex) * l2 / max(1e-6, math.hypot(hx - ex, hy - ey)), ey + (hy - ey) * l2 / max(1e-6, math.hypot(hx - ex, hy - ey))
    _limb(ctx, sh, (ex, ey), 0.60, 0.46, lw, folds=2)
    ctx.save()
    ctx.new_path()
    ctx.arc(ex, ey, 0.23, 0, 2 * math.pi)
    ctx.set_source_rgb(*COAT)
    ctx.fill()
    ctx.restore()
    ca, sa, nx, ny = _limb(ctx, (ex, ey), (wx_, wy_), 0.46, 0.34, lw, folds=3)
    hx, hy = wx_ + ca * 0.10, wy_ + sa * 0.10
    # fur cuff
    cf = [(wx_ + nx * 0.22 - ca * 0.06, wy_ + ny * 0.22 - sa * 0.06), (wx_ + nx * 0.22 + ca * 0.10, wy_ + ny * 0.22 + sa * 0.10),
          (wx_ - nx * 0.22 + ca * 0.10, wy_ - ny * 0.22 + sa * 0.10), (wx_ - nx * 0.22 - ca * 0.06, wy_ - ny * 0.22 - sa * 0.06)]
    poly_fill(ctx, cf, FUR)
    for i in range(9):
        u = i / 8 - 0.5
        px, py = wx_ + nx * 0.44 * u, wy_ + ny * 0.44 * u
        stroke(ctx, [(px - ca * 0.06, py - sa * 0.06), (px + ca * 0.12, py + sa * 0.12)], lw * 0.8, taper=(0.1, 0.9))

    def P(u, v):
        return (hx + ca * u + nx * v, hy + sa * u + ny * v)
    fist = [P(-0.06, -0.18), P(0.12, -0.20), P(0.21, -0.12), P(0.22, 0.12), P(0.10, 0.21), P(-0.08, 0.19)]
    shape(ctx, fist, SKIN, line=lw * 1.5)
    ctx.save()
    clip_curve(ctx, fist, 6)
    fill_curve(ctx, [P(-0.2, 0.06), P(0.35, 0.06), P(0.35, 0.35), P(-0.2, 0.35)], SKIN_SH, 4)
    ctx.restore()
    if finger == "point":
        fg = [P(0.14, -0.18), P(0.52, -0.17), P(0.58, -0.12), P(0.54, -0.06), P(0.16, -0.05)]
        shape(ctx, fg, SKIN, line=lw * 1.4)
        stroke(ctx, [P(0.47, -0.15), P(0.49, -0.09)], lw * 0.7, taper=(0.2, 0.2))
        stroke(ctx, [P(0.30, -0.16), P(0.31, -0.07)], lw * 0.6, taper=(0.3, 0.3))
    for v in (0.02, 0.10):
        stroke(ctx, [P(0.15, v), P(0.21, v + 0.02)], lw * 0.9, taper=(0.2, 0.4))
    thumb = [P(-0.02, -0.15), P(0.16, -0.08), P(0.21, -0.01), P(0.12, 0.02), P(-0.02, -0.02)]
    shape(ctx, thumb, SKIN, line=lw * 1.2)
    ctx.restore()
    tip = P(0.58, -0.12)
    sgn = -1 if flip else 1
    return (x + (tip[0] * math.cos(tilt) - tip[1] * math.sin(tilt)) * k * sgn,
            y + (tip[0] * math.sin(tilt) + tip[1] * math.cos(tilt)) * k)

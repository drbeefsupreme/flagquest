"""t02 close-up cast, drawn as Chick-tract brush art (grey tones -> halftone via vx.ink; black = solid ink).

Head units: eye line y = 0, crown y = -1, chin y = +1 (a head is 2 units tall); x right, y down.
summer(ctx, x, y, k, **p)   Summer head-and-shoulders, 3/4 view facing right; (x, y) = head centre (eye line),
                            k = design units per head unit.
crow(ctx, x, y, k, **p)     Dr. Beelzebub Crow, 3/4 view facing left (see t02_crow.py)
Owner: T02Wrong."""
import math

import numpy as np

from .t02_brush import (stroke, shape, hatch, dots, clip_curve, path_curve, grey, WHITE, catmull, poly_fill,
                        lock)

SKIN = grey(1.0)
SKIN_SH = grey(0.80)
SKIN_SH2 = grey(0.62)
HAIR = grey(1.0)
HAIR_SH = grey(0.78)
HAIR_SH2 = grey(0.55)
LIGHT = (-0.55, -0.8)


def begin(ctx, x, y, k, tilt=0.0, flip=False, sx=1.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(tilt)
    ctx.scale((-k if flip else k) * sx, k)


def fill_curve(ctx, pts, c, n=10, alpha=1.0):
    ctx.new_path()
    path_curve(ctx, pts, True, n)
    ctx.set_source_rgba(c[0], c[1], c[2], alpha)
    ctx.fill()


def ellipse_path(ctx, x, y, rx, ry, rot=0.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    ctx.scale(rx, ry)
    ctx.new_sub_path()
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()


# ============================================================ shared features
def eye(ctx, cx, cy, w, h, *, openk=1.0, look=(0.0, 0.0), pupil=1.0, lw=0.02, lash=1.0, lid=0.0, far=False,
        iris_c=0.42, shine=1.0, lower=True):
    """comic eye (facing right). openk: 0 shut .. 1 normal .. 1.6 wide-open shock.
    look: gaze (-1..1, -1..1). pupil: 0.3 pin-prick .. 1 normal. lid: 0..1 heavy-lidded (sardonic/sleepy)."""
    o = max(0.0, openk)
    top = h * (0.55 * o - 0.25 * lid)
    bot = h * (0.38 * min(o, 1.3))
    tilt = -h * 0.06
    up = [(cx - w * 0.52, cy + h * 0.02), (cx - w * 0.22, cy - top * 0.95), (cx + w * 0.15, cy - top * 1.02 + tilt),
          (cx + w * 0.50, cy - top * 0.2 + tilt)]
    lo = [(cx + w * 0.50, cy - top * 0.2 + tilt), (cx + w * 0.18, cy + bot * 0.95), (cx - w * 0.22, cy + bot * 0.9),
          (cx - w * 0.52, cy + h * 0.02)]
    if o > 0.06:
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
        ir = h * 0.40
        ix = cx + w * (0.04 + 0.26 * look[0])
        iy = cy + h * (0.02 + 0.22 * look[1])
        ctx.new_path()
        ctx.arc(ix, iy, ir, 0, 2 * math.pi)
        ctx.set_source_rgb(iris_c, iris_c, iris_c)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(lw * 0.3)
        for a in range(14):
            an = a * math.pi / 7 + 0.2
            ctx.move_to(ix + math.cos(an) * ir * 0.5, iy + math.sin(an) * ir * 0.5)
            ctx.line_to(ix + math.cos(an) * ir * 0.92, iy + math.sin(an) * ir * 0.92)
        ctx.stroke()
        ctx.new_path()
        ctx.arc(ix, iy, ir, 0, 2 * math.pi)
        ctx.set_line_width(lw * 0.55)
        ctx.stroke()
        ctx.new_path()
        ctx.arc(ix, iy, ir * 0.5 * pupil, 0, 2 * math.pi)
        ctx.fill()
        # the upper lid casts a soft shadow onto the eyeball
        ctx.new_path()
        path_curve(ctx, up + [(cx + w * 0.6, cy - top * 0.2 + h * 0.3), (cx - w * 0.6, cy + h * 0.32)], True, 6)
        ctx.set_source_rgba(0, 0, 0, 0.0)
        ctx.new_path()
        if shine > 0:
            ctx.set_source_rgba(1, 1, 1, shine)
            ctx.new_path()
            ctx.arc(ix - ir * 0.32, iy - ir * 0.36, ir * 0.22, 0, 2 * math.pi)
            ctx.fill()
        ctx.restore()
        if lower:
            stroke(ctx, lo, lw * 0.5, taper=(0.55, 0.15))
    stroke(ctx, up, lw * (1.3 + 0.8 * lash), taper=(0.12, 0.3))
    if lash > 0:
        ex, ey = up[-1]
        for i, (dx, dy) in enumerate(((0.16, -0.14), (0.22, -0.04), (0.18, 0.06))):
            stroke(ctx, [(ex - w * 0.07, ey + h * 0.05 * i), (ex + dx * w * 0.5, ey + dy * h * 1.1 - h * 0.04)],
                   lw * 0.8 * lash, taper=(0.05, 0.95), n=3)
    if o > 0.25:
        cr = [(cx - w * 0.38, cy - top * 0.9 - h * 0.18), (cx + w * 0.02, cy - top * 1.08 - h * 0.22),
              (cx + w * 0.36, cy - top * 0.55 - h * 0.18)]
        stroke(ctx, cr, lw * 0.45, taper=(0.5, 0.5))


def mouth(ctx, cx, cy, w, openk, *, lw=0.02, shape_="talk", smile=0.0, far_k=0.65, lips=0.0, teeth=True,
          tongue=True):
    """mouth facing right, centred (cx, cy). w = full width. shape_: talk | o | shout | grin.
    far_k shortens the far half (3/4 view). lips > 0 draws shaded lips (Summer)."""
    nl = w * 0.5
    fr = w * 0.5 * far_k
    if shape_ == "o":
        nl *= 0.6
        fr *= 0.6
    oh = w * 0.42 * openk
    if shape_ == "shout":
        oh = w * 0.62 * max(openk, 0.35)
    if shape_ == "grin":
        oh = w * 0.28 * max(openk, 0.3)
    cu = w * 0.08 * smile       # corners up
    if openk < 0.07 and shape_ != "grin":
        up = [(cx - nl, cy - cu), (cx - nl * 0.4, cy - w * 0.03), (cx + fr * 0.1, cy - w * 0.01),
              (cx + fr, cy - cu * 0.8)]
        if lips:
            fill_curve(ctx, [(cx - nl * 0.85, cy - cu * 0.5), (cx - nl * 0.3, cy - w * 0.12), (cx + fr * 0.05, cy - w * 0.09),
                             (cx + fr * 0.4, cy - w * 0.12), (cx + fr * 0.9, cy - cu * 0.6), (cx + fr * 0.2, cy + w * 0.02)],
                       grey(0.72), 6)
            fill_curve(ctx, [(cx - nl * 0.7, cy + w * 0.02), (cx + fr * 0.7, cy + w * 0.02), (cx + fr * 0.3, cy + w * 0.17),
                             (cx - nl * 0.4, cy + w * 0.16)], grey(0.80), 6)
        stroke(ctx, up, lw * 1.0, taper=(0.35, 0.35))
        stroke(ctx, [(cx - nl * 0.3, cy + w * 0.2), (cx + fr * 0.4, cy + w * 0.19)], lw * 0.55, taper=(0.4, 0.4))
        return
    top_y = cy - oh * 0.22 - w * 0.02
    bot_y = cy + oh * 0.85
    pts = [(cx - nl, cy - cu), (cx - nl * 0.45, top_y), (cx + fr * 0.35, top_y - w * 0.015), (cx + fr, cy - cu * 0.8),
           (cx + fr * 0.4, bot_y), (cx - nl * 0.4, bot_y - oh * 0.04)]
    if lips:
        # upper lip band + lower lip
        fill_curve(ctx, [(cx - nl * 1.02, cy - cu), (cx - nl * 0.4, top_y - w * 0.10), (cx + fr * 0.0, top_y - w * 0.07),
                         (cx + fr * 0.4, top_y - w * 0.10), (cx + fr * 1.02, cy - cu * 0.8), (cx, top_y + 0.01)],
                   grey(0.72), 6)
        fill_curve(ctx, [(cx - nl * 0.6, bot_y - oh * 0.02), (cx + fr * 0.6, bot_y - oh * 0.02),
                         (cx + fr * 0.3, bot_y + w * 0.14), (cx - nl * 0.3, bot_y + w * 0.14)], grey(0.80), 6)
    ctx.save()
    ctx.new_path()
    path_curve(ctx, pts, True, 8)
    ctx.set_source_rgb(0, 0, 0)
    ctx.fill_preserve()
    ctx.clip()
    if teeth and openk > 0.12:
        ctx.set_source_rgb(1, 1, 1)
        ctx.new_path()
        ctx.rectangle(cx - nl - 1, top_y - 1, nl + fr + 2, oh * 0.24 + 1)
        ctx.fill()
    if tongue and openk > 0.3:
        ctx.set_source_rgb(0.6, 0.6, 0.6)
        ellipse_path(ctx, cx - w * 0.02, bot_y + oh * 0.12, w * 0.24, oh * 0.35)
        ctx.fill()
    ctx.restore()
    stroke(ctx, pts[:4], lw * 0.9, taper=(0.3, 0.3))
    stroke(ctx, [(cx - nl * 0.5, bot_y + w * 0.12), (cx, bot_y + w * 0.15), (cx + fr * 0.5, bot_y + w * 0.11)],
           lw * 0.6, taper=(0.4, 0.4))


def drop(ctx, x, y, r, lw, rot=0.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    pts = [(0, -r * 1.7), (r * 0.72, r * 0.1), (0, r), (-r * 0.72, r * 0.1)]
    shape(ctx, pts, WHITE, line=lw * 1.2)
    stroke(ctx, [(-r * 0.38, -r * 0.05), (-r * 0.28, r * 0.5)], lw * 0.8, taper=(0.3, 0.3))
    ctx.restore()


# ============================================================ SUMMER
S_FACE = [(0.50, -0.66), (0.60, -0.36), (0.615, -0.14), (0.575, 0.00), (0.625, 0.15), (0.60, 0.34), (0.555, 0.52),
          (0.50, 0.68), (0.42, 0.84), (0.28, 0.94), (0.10, 0.92), (-0.14, 0.80), (-0.34, 0.58), (-0.42, 0.28),
          (-0.46, -0.10), (-0.30, -0.66)]
S_CONTOUR = S_FACE[1:13]
S_HAIR = [(0.66, -0.36), (0.72, -0.70), (0.52, -1.02), (0.10, -1.16), (-0.40, -1.10), (-0.80, -0.82),
          (-0.98, -0.36), (-0.92, 0.04), (-0.66, 0.16), (-0.44, 0.10), (-0.30, -0.10)]


def summer(ctx, x, y, k, *, t=0.0, tilt=0.0, look=(0.5, 0.0), eye_open=1.0, pupil=1.0, brow=0.0, mouth_open=0.0,
           mshape="talk", smile=0.35, sweat=0, blush=0.0, lw=None, body=True, lid=0.0, flip=False, hair_wind=0.0,
           wet=0.0, shade=1.0, tears=0.0):
    """Summer, 3/4 facing right. brow: -1 frown .. 0 .. 1 raised (worry / shock). lw = base line width
    in design units (default 0.02*k). tears 0..1: streams from both eyes. wet 0..1: rain-plastered hair + drips."""
    lw = (lw if lw is not None else 0.02 * k) / k
    begin(ctx, x, y, k, tilt, flip)
    if body:
        _summer_body(ctx, t, lw)
    _summer_back_hair(ctx, t, lw, hair_wind, wet)
    _summer_far_pigtail(ctx, t, lw, hair_wind)
    # neck (short, soft)
    neck = [(-0.24, 0.56), (-0.30, 1.30), (0.34, 1.32), (0.30, 0.84)]
    fill_curve(ctx, neck, SKIN, 4)
    if shade:
        ctx.save()
        clip_curve(ctx, neck, 4)
        # the jaw's cast shadow on the neck
        fill_curve(ctx, [(-0.5, 0.5), (0.1, 0.86), (0.36, 0.84), (0.4, 1.0), (0.0, 1.06), (-0.5, 0.96)], SKIN_SH, 6)
        ctx.restore()
    stroke(ctx, [(-0.25, 0.60), (-0.27, 0.95), (-0.31, 1.30)], lw * 1.3, taper=(0.1, 0.3))
    stroke(ctx, [(0.30, 0.86), (0.30, 1.1), (0.34, 1.30)], lw * 1.1, taper=(0.1, 0.4))
    # face
    fill_curve(ctx, S_FACE, SKIN)
    if shade:
        ctx.save()
        clip_curve(ctx, S_FACE, 10)
        # a crisp form shadow along the far cheek (the plane turning away from the key light)
        fill_curve(ctx, [(0.585, -0.06), (0.56, 0.06), (0.585, 0.18), (0.53, 0.38), (0.47, 0.60), (0.38, 0.80),
                         (0.30, 0.95), (0.8, 1.0), (0.8, -0.06)], SKIN_SH, 8)
        # under the nose, under the lower lip
        fill_curve(ctx, [(0.40, 0.40), (0.50, 0.37), (0.56, 0.40), (0.48, 0.44)], SKIN_SH, 6)
        fill_curve(ctx, [(0.28, 0.76), (0.44, 0.74), (0.38, 0.80)], SKIN_SH, 6)
        if blush > 0:
            ctx.set_source_rgba(0.6, 0.6, 0.6, blush)
            ellipse_path(ctx, 0.14, 0.30, 0.18, 0.07)
            ellipse_path(ctx, 0.55, 0.28, 0.05, 0.06)
            ctx.fill()
            for j in range(6):
                hx = 0.02 + j * 0.045
                stroke(ctx, [(hx, 0.34), (hx + 0.05, 0.26)], lw * 0.7, taper=(0.2, 0.2), alpha=blush)
        ctx.restore()
    stroke(ctx, S_CONTOUR, lw * 1.45, taper=(0.12, 0.35), light=LIGHT, lk=0.55)
    # ear + little hoop
    ear = [(-0.36, -0.02), (-0.24, -0.06), (-0.19, 0.10), (-0.22, 0.28), (-0.29, 0.34), (-0.38, 0.26)]
    shape(ctx, ear, SKIN, line=lw * 1.0)
    stroke(ctx, [(-0.31, 0.03), (-0.25, 0.12), (-0.28, 0.24)], lw * 0.7, taper=(0.3, 0.3))
    ctx.save()
    ctx.set_source_rgb(0, 0, 0)
    ctx.new_path()
    ctx.arc(-0.30, 0.37, 0.045, 0.3, math.pi * 2.3)
    ctx.set_line_width(lw * 1.0)
    ctx.stroke()
    ctx.restore()
    # eyes + brows
    b = brow
    eh = 0.20 * (1 + 0.15 * max(0.0, b))
    eye(ctx, 0.15, 0.02, 0.35, eh, openk=eye_open, look=look, pupil=pupil, lw=lw, lid=lid, iris_c=0.30)
    eye(ctx, 0.49, 0.01, 0.175, eh * 0.95, openk=eye_open, look=look, pupil=pupil, lw=lw, lid=lid, far=True, lash=0.8,
        iris_c=0.30)
    by = -0.20 - 0.10 * max(0, b) - 0.02 * min(0, b)
    stroke(ctx, [(-0.02, by + 0.05 + 0.02 * b), (0.15, by - 0.03 - 0.05 * b), (0.33, by + 0.02 + 0.07 * b)],
           lw * 1.8, taper=(0.25, 0.7))
    stroke(ctx, [(0.43, by + 0.03 + 0.07 * b), (0.51, by - 0.02 - 0.02 * b), (0.59, by + 0.04 - 0.01 * b)],
           lw * 1.3, taper=(0.5, 0.5))
    # nose: a soft bridge accent + an upturned tip and nostril
    stroke(ctx, [(0.42, 0.10), (0.47, 0.22), (0.52, 0.30)], lw * 0.55, taper=(0.7, 0.3))
    stroke(ctx, [(0.585, 0.30), (0.565, 0.355), (0.50, 0.37), (0.46, 0.345)], lw * 1.15, taper=(0.3, 0.5))
    stroke(ctx, [(0.49, 0.33), (0.51, 0.355)], lw * 0.9, taper=(0.2, 0.2))
    # freckles across nose and cheeks
    rng = np.random.default_rng(7)
    fr = [(0.20 + rng.normal() * 0.07, 0.22 + rng.normal() * 0.03) for _ in range(10)]
    fr += [(0.55 + rng.normal() * 0.02, 0.21 + rng.normal() * 0.025) for _ in range(4)]
    fr += [(0.43 + rng.normal() * 0.025, 0.21 + rng.normal() * 0.02) for _ in range(5)]
    dots(ctx, fr, 0.0095)
    # mouth (full lips)
    mouth(ctx, 0.35, 0.60, 0.34, mouth_open, lw=lw, shape_=mshape, smile=smile, lips=1.0)
    _summer_fringe(ctx, t, lw, hair_wind, wet)
    if tears > 0:
        for ex, ey, s_ in ((0.10, 0.12, 1.0), (0.50, 0.10, 0.7)):
            L = 0.15 + 0.55 * tears
            pts = [(ex, ey), (ex + 0.02 * s_, ey + L * 0.4), (ex - 0.01, ey + L * 0.8), (ex + 0.01, ey + L)]
            stroke(ctx, pts, lw * 3.2 * s_, color=WHITE, taper=(0.1, 0.1))
            stroke(ctx, [(p[0] - 0.022 * s_, p[1]) for p in pts], lw * 0.7, taper=(0.1, 0.3))
            stroke(ctx, [(p[0] + 0.022 * s_, p[1]) for p in pts], lw * 0.7, taper=(0.1, 0.3))
            drop(ctx, ex + 0.01, ey + L + 0.05 + 0.03 * math.sin(t * 5), 0.035 * s_, lw)
    for i in range(int(sweat)):
        sx, sy, r = [(0.72, -0.40, 0.075), (-0.54, -0.40, 0.06), (0.74, -0.02, 0.05), (-0.62, 0.05, 0.05)][i % 4]
        drop(ctx, sx, sy + 0.02 * math.sin(t * 3 + i), r, lw, rot=0.25 if sx > 0 else -0.25)
    if wet > 0:
        rng = np.random.default_rng(int(t * 6) + 11)
        for i in range(int(2 + 3 * wet)):
            px = rng.uniform(-0.7, 0.6)
            py = rng.uniform(-0.3, 0.9)
            drop(ctx, px, py, 0.025, lw)
    ctx.restore()


def _summer_back_hair(ctx, t, lw, wind, wet=0.0):
    rx, ry = -0.84, 0.02
    # the near pigtail (tied behind the ear, falls behind the shoulder)
    sw = math.sin(t * 1.4) * 0.03 + wind * 0.1
    tail = [(rx + 0.10, ry - 0.10), (rx - 0.14, ry + 0.22), (rx - 0.32 + sw, ry + 0.66), (rx - 0.28 + sw, ry + 1.06),
            (rx - 0.08 + sw * 1.4, ry + 1.40), (rx - 0.03 + sw, ry + 1.06), (rx + 0.06 + sw, ry + 0.64),
            (rx + 0.16, ry + 0.24)]
    fill_curve(ctx, tail, HAIR)
    ctx.save()
    clip_curve(ctx, tail, 10)
    fill_curve(ctx, [(rx - 0.6, ry), (rx - 0.16, ry + 0.25), (rx - 0.20 + sw, ry + 0.8), (rx - 0.07 + sw, ry + 1.6),
                     (rx - 0.6, ry + 1.6)], HAIR_SH, 6)
    for i, o in enumerate((-0.07, -0.01, 0.05, 0.10)):
        stroke(ctx, [(rx + 0.05 + o * 0.5, ry + 0.02), (rx - 0.13 + o, ry + 0.40), (rx - 0.17 + sw + o * 0.8, ry + 0.85),
                     (rx - 0.07 + sw + o * 0.3, ry + 1.28)], lw * (0.75 if i % 2 else 0.5), taper=(0.2, 0.5))
    stroke(ctx, [(rx - 0.03, ry + 0.25), (rx - 0.22 + sw, ry + 0.72), (rx - 0.18 + sw, ry + 1.05)], lw * 2.4,
           taper=(0.3, 0.7))
    ctx.restore()
    stroke(ctx, tail[:6], lw * 1.6, taper=(0.05, 0.3), light=LIGHT)
    stroke(ctx, tail[5:], lw * 0.9, taper=(0.1, 0.6))
    # the head of hair
    fill_curve(ctx, S_HAIR, HAIR)
    ctx.save()
    clip_curve(ctx, S_HAIR, 10)
    # one shadow mass behind the ear toward the nape (+ a darker core), bare paper elsewhere (blonde)
    fill_curve(ctx, [(-1.1, -0.40), (-0.72, -0.46), (-0.60, -0.10), (-0.44, 0.30), (-1.1, 0.4)], HAIR_SH, 6)
    fill_curve(ctx, [(-1.1, -0.10), (-0.84, -0.18), (-0.70, 0.08), (-0.60, 0.3), (-1.1, 0.3)], HAIR_SH2, 6)
    # strands sweeping from the parting over the skull to the tie, broken around a highlight band
    for i in range(8):
        a0 = math.radians(-44 - i * 15)
        R0 = 0.94 - 0.04 * (i % 2)
        pts = []
        for j in range(6):
            u = j / 5
            an = a0 + (math.atan2(ry + 0.22, rx + 0.08) - a0) * u
            rad = R0 * (1 - 0.22 * u * u)
            pts.append((-0.08 + math.cos(an) * rad, -0.22 + math.sin(an) * rad))
        pts[-1] = (rx + 0.08, ry - 0.04)
        stroke(ctx, pts[:3], lw * (0.9 if i % 2 else 0.6), taper=(0.5, 0.5))
        stroke(ctx, pts[3:], lw * (1.0 if i % 2 else 0.7), taper=(0.6, 0.1))
        if i in (1, 4, 6):
            stroke(ctx, pts[3:], lw * 2.8, taper=(0.8, 0.1))
    ctx.restore()
    # contour inked in broken, tapered strokes (no helmet outline)
    stroke(ctx, S_HAIR[0:4], lw * 1.3, taper=(0.3, 0.3), light=LIGHT)
    stroke(ctx, S_HAIR[3:7], lw * 1.7, taper=(0.15, 0.2), light=LIGHT)
    stroke(ctx, S_HAIR[6:9], lw * 2.0, taper=(0.15, 0.4), light=LIGHT)
    # hair tie
    shape(ctx, [(rx - 0.06, ry - 0.12), (rx + 0.10, ry - 0.10), (rx + 0.10, ry + 0.08), (rx - 0.07, ry + 0.06)],
          grey(0.12), line=lw * 0.9)


def _summer_far_pigtail(ctx, t, lw, wind):
    """the far pigtail: tied behind the head, it drapes forward over the far shoulder (in front of the collar)"""
    sw = math.sin(t * 1.3 + 1.0) * 0.025 + wind * 0.08
    rx, ry = 0.36, 0.70
    tail = [(rx - 0.10, ry - 0.30), (rx + 0.14, ry + 0.05), (rx + 0.30 + sw, ry + 0.52), (rx + 0.30 + sw, ry + 0.92),
            (rx + 0.16 + sw * 1.3, ry + 1.22), (rx + 0.10 + sw, ry + 0.90), (rx + 0.02 + sw, ry + 0.50),
            (rx - 0.14, ry + 0.10)]
    fill_curve(ctx, tail, HAIR)
    ctx.save()
    clip_curve(ctx, tail, 10)
    fill_curve(ctx, [(rx + 0.6, ry), (rx + 0.20, ry + 0.2), (rx + 0.22 + sw, ry + 0.8), (rx + 0.12 + sw, ry + 1.4),
                     (rx + 0.6, ry + 1.4)], HAIR_SH, 6)
    for i, o in enumerate((-0.06, 0.0, 0.05, 0.10)):
        stroke(ctx, [(rx - 0.02 + o * 0.5, ry - 0.1), (rx + 0.12 + o, ry + 0.32), (rx + 0.20 + sw + o * 0.7, ry + 0.74),
                     (rx + 0.16 + sw + o * 0.3, ry + 1.10)], lw * (0.75 if i % 2 else 0.5), taper=(0.2, 0.5))
    stroke(ctx, [(rx + 0.22, ry + 0.30), (rx + 0.30 + sw, ry + 0.70), (rx + 0.26 + sw, ry + 1.0)], lw * 2.2,
           taper=(0.3, 0.7))
    ctx.restore()
    stroke(ctx, tail[1:6], lw * 1.5, taper=(0.05, 0.3), light=LIGHT)
    stroke(ctx, tail[5:], lw * 0.9, taper=(0.1, 0.6))
    # the band
    bx, by = rx + 0.12, ry + 0.16
    shape(ctx, [(bx - 0.13, by - 0.03), (bx + 0.09, by - 0.10), (bx + 0.12, by + 0.02), (bx - 0.10, by + 0.09)],
          grey(0.12), line=lw * 0.9)


def _summer_fringe(ctx, t, lw, wind, wet=0.0):
    """side-swept bangs: four chunky locks from the parting, tips curling toward the face"""
    locks = [  # root (on the parting), mid, tip
        ((0.08, -1.04), (-0.12, -0.70), (-0.20, -0.32)),
        ((0.16, -1.04), (0.06, -0.66), (0.06, -0.24)),
        ((0.26, -1.02), (0.30, -0.64), (0.32, -0.22)),
        ((0.36, -0.98), (0.54, -0.64), (0.60, -0.28)),
        ((0.44, -0.92), (0.70, -0.62), (0.70, -0.42)),
    ]
    widths = (0.22, 0.28, 0.28, 0.26, 0.18)
    for i, ((r0, r1), (m0, m1), (t0, t1)) in enumerate(locks):
        sw = wind * 0.04 * math.sin(t * 3 + i * 1.7)
        curl = 0.06
        pts = [(r0, r1), (m0, m1), (t0 - curl * 0.3 + sw, t1 - 0.12), (t0 + curl + sw, t1)]
        lock(ctx, pts, widths[i], line=lw * 1.05, dark=HAIR_SH if i in (1, 3) else None, shadow_side=1, root=0.08,
             tip=0.55)
        # one inner strand per lock
        stroke(ctx, [(r0 + 0.02, r1 + 0.04), (m0 + 0.03, m1), (t0 + curl * 0.4 + sw, t1 - 0.08)], lw * 0.5,
               taper=(0.3, 0.7))
    # thin shadow the fringe throws on the forehead
    ctx.save()
    ctx.set_source_rgb(*SKIN_SH)
    for (_, _, (t0, t1)) in locks[:4]:
        ellipse_path(ctx, t0 + 0.05, t1 + 0.05, 0.07, 0.025)
    ctx.fill()
    ctx.restore()


def _summer_body(ctx, t, lw):
    jacket = [(-1.45, 2.5), (-1.35, 1.62), (-0.80, 1.28), (0.0, 1.30), (0.80, 1.26), (1.40, 1.60), (1.50, 2.5)]
    fill_curve(ctx, jacket, grey(0.62), 6)
    ctx.save()
    clip_curve(ctx, jacket, 6)
    fill_curve(ctx, [(-1.6, 1.4), (-0.95, 1.4), (-0.75, 2.6), (-1.6, 2.6)], grey(0.30), 4)
    hatch(ctx, (0.35, 1.45, 1.6, 2.6), 1.15, 0.05, lw * 0.6, seed=3, frac=0.7)
    for sx in (-0.55, 0.85):
        stroke(ctx, [(sx, 1.55), (sx + 0.05, 1.95), (sx + 0.02, 2.5)], lw * 0.8, taper=(0.3, 0.3))
    ctx.restore()
    stroke(ctx, jacket, lw * 1.6, taper=(0, 0))
    stroke(ctx, [(0.08, 1.55), (0.14, 2.5)], lw * 1.3, taper=(0.1, 0.1))
    # faux-fur collar: white, tufted, with one soft shadow band
    n = 40
    rng = np.random.default_rng(11)
    top = []
    for i in range(n + 1):
        a = i / n
        xx = -1.05 + 2.1 * a
        yy = 1.26 + 0.10 * (2 * a - 1) ** 2 - 0.02
        top.append((xx, yy - (0.06 + 0.07 * rng.random()) * (1 if i % 2 else 0.2)))
    bot = [(1.10, 1.62), (0.5, 1.76), (0.0, 1.78), (-0.5, 1.76), (-1.10, 1.62)]
    poly_fill(ctx, top + bot, grey(1.0))
    ctx.save()
    ctx.new_path()
    ctx.move_to(*top[0])
    for p in top[1:] + bot:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.clip()
    fill_curve(ctx, [(-1.2, 1.58), (1.2, 1.58), (1.2, 1.9), (-1.2, 1.9)], grey(0.80), 4)
    ctx.restore()
    for i in range(1, n, 1):
        x0, y0 = top[i]
        ln = 0.10 + 0.08 * rng.random()
        stroke(ctx, [(x0, y0), (x0 + 0.015, y0 + ln * 0.6), (x0 - 0.01, y0 + ln)], lw * 0.65, taper=(0.1, 0.9))
    for i in range(18):
        a = i / 17
        x0 = -1.0 + 2.0 * a
        y0 = 1.65 + 0.1 * (1 - (2 * a - 1) ** 2)
        stroke(ctx, [(x0, y0), (x0 + 0.02, y0 + 0.08)], lw * 0.6, taper=(0.1, 0.9))
    # goggles hanging round the neck
    for gx in (-0.16, 0.14):
        gy = 1.46
        lens = [(gx - 0.12, gy), (gx, gy - 0.06), (gx + 0.12, gy), (gx + 0.115, gy + 0.11), (gx, gy + 0.15),
                (gx - 0.12, gy + 0.10)]
        shape(ctx, lens, grey(0.30), line=lw * 1.5)
        stroke(ctx, [(gx - 0.07, gy - 0.01), (gx + 0.03, gy - 0.03)], lw * 1.2, color=WHITE, taper=(0.3, 0.3))
    stroke(ctx, [(-0.04, 1.50), (0.02, 1.50)], lw * 2.0, taper=(0, 0))
    stroke(ctx, [(-0.28, 1.50), (-0.52, 1.40), (-0.40, 1.26)], lw * 1.6, taper=(0.1, 0.4), color=grey(0.1))
    stroke(ctx, [(0.26, 1.50), (0.46, 1.40), (0.36, 1.27)], lw * 1.6, taper=(0.1, 0.4), color=grey(0.1))

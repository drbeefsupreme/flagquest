"""t03 the forecast (owner T03Genesis): 61.65 the boombox in the mud (F01, match-cut out of the hurricane's
eye into its speaker cone), 62.95 the downpour, 64.63 the Ranger's handheld radio (G01) -> t04 at 67.46."""
import math

import cairo
import numpy as np

from vx import *
from vx import ink, comic
from vx.flag import draw_flag
from vx.chars import Character
from scenes.t03_cam import Cam
from scenes import t03_burn as BU
from scenes import t03_props as PR
from scenes import t03_rain as RN

T_BOOM = 61.65
T_RAIN = 62.951
T_RANGER = 64.63
RAIN_SLANT = 0.18

# ------------------------------------------------------------------ the boombox in the mud
BB_UNITS = 0.95                       # boombox width in world units (comic-sized)
BB_U = BB_UNITS / PR.BB_W             # world units per boombox unit
BB_BASE = BU.stage(0.25, -3.4)        # on the ground
BB_TILT = math.radians(-7.0)
BB_SINK = 0.14                        # fraction of the body height under the mud


def _bb_centre():
    return BB_BASE + BU.UP * (PR.BB_H * BB_U * (0.5 - BB_SINK))


def _bb_local_to_world(x, y):
    c, s = math.cos(BB_TILT), math.sin(BB_TILT)
    xr, yr = x * c - y * s, x * s + y * c
    return _bb_centre() + BU.RIGHT * (xr * BB_U) + BU.UP * (-yr * BB_U)


def rain_amount(T):
    return smoothstep(T_RAIN - 0.05, T_RAIN + 0.3, T)


def boom_cam(T):
    spk = _bb_local_to_world(*PR.SPK_R)
    Bc = _bb_centre()
    u = 1.0 - (1.0 - smoothstep(T_BOOM, T_BOOM + 1.15, T)) ** 3       # fast pull-back out of the cone
    d0, d1 = 0.48, 1.85
    dist = d0 + (d1 - d0) * u
    tgt = spk * (1 - u) + (Bc + BU.UP * 0.22 + BU.RIGHT * (-0.02)) * u
    drift = smoothstep(T_BOOM + 1.0, T_RANGER, T)
    C = tgt - BU.FWD * dist + BU.UP * (0.02 + 0.06 * u + 0.03 * drift) + BU.RIGHT * (-0.22 * u - 0.1 * drift)
    return Cam.look(C, tgt, f=1500.0)


def _people_behind(cam, st, T, zmin, wet):
    ppl = BU.crowd_people(cam, st["crowd"], T, 0.0, wet=wet, mud=1.0)
    return [p for p in ppl if p["depth"] > zmin]


def _compose(fc, bg, mid, flags, front, fx, focus=None):
    """ruled background -> mid layer (form-following) -> flag plate -> front objects -> rain/splash layer
    (knocked out of the cloth: nothing is ever drawn over a flag). focus=(x, y): Dore light pool on the subject."""
    lm = 1.0
    if focus is not None:
        ys, xs = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
        dx = xs / fc.s - focus[0]
        dy = ys / fc.s - focus[1]
        pool = np.exp(-((dx / 900.0) ** 2 + (dy / 620.0) ** 2))
        lm = (0.45 + 0.55 * pool ** 0.8)[..., None].astype(np.float32)
    img = ink.engrave(fc, bg.rgb() * lm, angle=0.0)
    m = mid.rgba()
    m[..., :3] *= lm
    img = over(img, ink.print_layer(fc, m, style="engrave"))
    img = ink.spot(img, flags)
    img = over(img, ink.print_layer(fc, front.rgba(), style="engrave"))
    fxl = ink.print_layer(fc, fx.rgba(), style="engrave")
    return over(img, fxl * (1.0 - flags[..., 3:4]))


def _stray_flag(fc, cam, P, T, rain, seed, ang, side):
    """one of the six, dropped by the storm and stuck askew in the mud: wet, heavy, plain"""
    fl = Canvas(fc)
    fx, fy, fz = cam.project(P)
    if fz > 0.5:
        sc = cam.f / fz
        draw_flag(fl.ctx, fx, fy, pole=1.9 * sc, t=T, wind=0.12 + 0.2 * (1 - rain), side=side, ang=ang, seed=seed,
                  light=(-0.3, -0.8, 0.5))
    return fl.rgba(), fz


def _world_bg(fc, cam, st, T, wet, rain, fog0=6.0, fog1=30.0):
    tone, gmask = BU.mud_tone(fc, cam, T, wet=wet, rain=rain, fog0=fog0, fog1=fog1)
    sky = BU.sky_tone(fc, cam, st["sky"], T, dark=1.0)
    world = gmask * tone + (1 - gmask) * sky
    bg = Canvas(fc)
    bg.paint_array(np.repeat(world[..., None], 3, axis=2), 0, 0)
    BU.draw_camps(bg.ctx, cam, st["camps"], T, wind=1.0)
    return bg


def boombox_frame(fc, st):
    T = fc.T
    cam = boom_cam(T)
    rain = rain_amount(T)
    wet = rain
    bg = _world_bg(fc, cam, st, T, 0.4 + 0.6 * wet, rain)
    RN.draw_rain(bg.ctx, T, density=0.6 * rain, slant=RAIN_SLANT, seed=3, speed=1.0, width=1.2)
    flags, fz = _stray_flag(fc, cam, BOOM_FLAG, T, rain, 77, -0.42, -1)
    bz = float(cam.project(_bb_centre())[2])
    ppl = _people_behind(cam, st, T, bz + 2.6, wet)
    mid, front, fx = Canvas(fc), Canvas(fc), Canvas(fc)
    BU.draw_people(mid.ctx, [p for p in ppl if p["depth"] > fz], T)
    BU.draw_people(front.ctx, [p for p in ppl if p["depth"] <= fz], T)
    # the boombox, sunk in the mud: clipped at the mud line, then a heap of mud over its base
    ctx = front.ctx
    c = cam.project(_bb_centre())
    k = cam.f / c[2] * BB_U
    gl = cam.project(BB_BASE)
    ctx.save()
    ctx.rectangle(-2000, -2000, 6000, gl[1] + 2000)
    ctx.clip()
    ctx.translate(c[0], c[1])
    ctx.scale(k, k)
    ctx.rotate(BB_TILT)
    lv = fc.tl.mouth("FORECAST", T)
    PR.boombox(ctx, T, level=lv, wet=wet, mud=0.8)
    ctx.restore()
    _mud_lip(ctx, gl[0], gl[1], k * PR.BB_W * 0.62, k, T)
    # rain in front + crowns on the mud
    RN.splashes(fx.ctx, T, gl[1] - 30 * k, fc.H + 40, density=rain * 0.9, seed=4, size=1.3)
    RN.draw_rain(fx.ctx, T, density=rain * 0.35, slant=RAIN_SLANT, seed=9, speed=1.25, width=1.5, length=1.6)
    img = _compose(fc, bg, mid, flags, front, fx, focus=(c[0], c[1]))
    top = Canvas(fc)
    sp = cam.project(_bb_local_to_world(*PR.SPK_L))
    comic.say(top.ctx, fc, "F01", 700, 190, w=760, tail=(sp[0], sp[1] - 40),
              avoid=lambda TT: stray_box(boom_cam(TT), BOOM_FLAG))
    return img, flags, top


def _mud_lip(ctx, x, y, halfw, k, T, seed=5):
    """the mud swallowing the base of something: a heaped, glistening mound that fades into the ground"""
    rng = np.random.default_rng(seed)
    ctx.save()
    H = halfw * 0.16
    # heaped lumps, soft-edged so they melt into the mud plain
    for i in range(14):
        u = (i + 0.5) / 14
        cx = x - halfw * 1.25 + 2.5 * halfw * u + rng.uniform(-0.04, 0.04) * halfw
        rx = halfw * rng.uniform(0.2, 0.34) * (1.2 - 0.5 * abs(u - 0.5))
        ry = H * rng.uniform(0.7, 1.2) * (1.25 - abs(u - 0.5))
        cy = y + ry * 0.35
        g = cairo.RadialGradient(cx - rx * 0.2, cy - ry * 0.5, 0, cx, cy, rx)
        v = rng.uniform(0.3, 0.45)
        g.add_color_stop_rgba(0, v, v, v, 1.0)
        g.add_color_stop_rgba(0.7, v + 0.05, v + 0.05, v + 0.05, 0.9)
        g.add_color_stop_rgba(1, v + 0.1, v + 0.1, v + 0.1, 0.0)
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(1.0, ry / rx)
        ctx.arc(0, 0, rx, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source(g)
        ctx.fill()
    # the contact line: a black crease where object meets mud, and a wet white glint on the crest
    pts = []
    for i in range(25):
        u = i / 24
        pts.append((x - halfw * 1.05 + 2.1 * halfw * u, y - H * (0.35 + 0.25 * math.sin(u * 8.0 + seed)
                                                             + 0.15 * math.sin(u * 21.0))))
    ctx.set_line_cap(1)
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(max(1.5, H * 0.12))
    smooth(ctx, pts)
    ctx.stroke()
    ctx.restore()


# ------------------------------------------------------------------ the Ranger's radio (G01) -> t04
RANGER = Character("hippie", 41, render="ink", hat="cap", mud=0.25)
R_POS = BU.stage(1.05, -4.2)
R_H = 1.5                              # standing height (units)
R_SINK = 0.2                           # shins in the mud (units)


def ranger_cam(T):
    """medium shot on the Ranger; a slow crane back and up over the rainy crowd for the cut to t04"""
    u = smootherstep(66.4, 67.46, T) ** 1.2
    drift = smoothstep(T_RANGER, 66.6, T)
    tgt = R_POS + BU.UP * (0.95 - 0.1 * u) + BU.RIGHT * (-0.42 - 0.12 * drift + 0.3 * u) + BU.FWD * (1.2 * u)
    C = R_POS - BU.FWD * (2.35 - 0.12 * drift + 1.4 * u) + BU.UP * (1.05 + 0.45 * u) + BU.RIGHT * (-0.35 - 0.5 * u)
    return Cam.look(C, tgt, f=1500.0)


def _radio_hand(a, h):
    """the radio held up at the mouth"""
    mx, my = a["mouth"]
    return (mx - h * 0.075, my + h * 0.075)


RANGER_FLAG = BU.stage(-3.6, 6.5)
BOOM_FLAG = BU.stage(-2.2, 3.4)


def stray_box(cam, P):
    """conservative screen box of a stray flag (pole 1.9 units standing at P, tilted <= 0.45 rad)"""
    fx, fy, fz = cam.project(P)
    if fz <= 0.5:
        return []
    L = 1.9 * cam.f / fz
    return [(fx - 0.8 * L, fy - 1.12 * L, fx + 0.8 * L, fy - 0.35 * L)]


def ranger_frame(fc, st):
    T = fc.T
    cam = ranger_cam(T)
    rain = 1.0
    bg = _world_bg(fc, cam, st, T, 0.55, rain, fog0=6.0, fog1=34.0)
    RN.draw_rain(bg.ctx, T, density=0.6 * rain, slant=RAIN_SLANT, seed=3, width=1.2)
    flags, fz = _stray_flag(fc, cam, RANGER_FLAG, T, 0.0, 81, 0.38, 1)
    rz = float(cam.project(R_POS)[2])
    ppl = [p for p in BU.crowd_people(cam, st["crowd"], T, 0.0, wet=1.0, mud=1.0) if p["depth"] > rz + 1.2]
    mid, front, fx = Canvas(fc), Canvas(fc), Canvas(fc)
    BU.draw_people(mid.ctx, [p for p in ppl if p["depth"] > fz], T)
    BU.draw_people(front.ctx, [p for p in ppl if p["depth"] <= fz], T)
    ctx = front.ctx
    # the Ranger, shin-deep: drawn from a ground point below the mud, clipped at the mud line
    gx, gy, gz = cam.project(R_POS)
    k = cam.f / gz
    h = R_H * k
    sink = R_SINK * k
    mouth = fc.tl.mouth("RANGER", T)
    w0, w1 = fc.tl.word("G01", "dirt")
    down = smoothstep(w0 - 0.25, w0 + 0.2, T) * (1 - smoothstep(67.1, 67.45, T) * 0.5)
    look = (0.25 - 0.1 * down, -0.1 + 0.95 * down)
    expr = {"bewildered": 0.6 + 0.4 * down, "neutral": 0.4 - 0.4 * down}
    pose_kw = dict(facing=-1, turn=0.35)
    a0 = RANGER.anchors(gx, gy + sink, h, "stand", T, **pose_kw)
    hand = _radio_hand(a0, h)
    ctx.save()
    ctx.rectangle(-4000, -4000, 9000, gy - 0.02 * k + 4000)
    ctx.clip()
    a = RANGER.draw(ctx, gx, gy + sink, h, "stand", T, mouth=mouth, look=look, expr=expr, hand_r=hand,
                    shape_r="grip", **pose_kw)
    ctx.restore()
    # the radio in his fist
    hx, hy = a["hand_r"]
    rs = h * 0.0005
    ctx.save()
    ctx.translate(hx, hy - h * 0.04)
    ctx.rotate(0.12)
    ctx.scale(rs, rs)
    PR.radio(ctx, T, level=mouth)
    ctx.restore()
    _mud_lip(ctx, gx, gy + 0.03 * k, h * 0.22, k / 900.0 * 1.4, T)
    RN.splashes(fx.ctx, T, gy - 0.2 * k, fc.H + 40, density=0.9, seed=6, size=1.2)
    RN.draw_rain(fx.ctx, T, density=0.35, slant=RAIN_SLANT, seed=11, speed=1.25, width=1.5, length=1.6)
    img = _compose(fc, bg, mid, flags, front, fx, focus=(gx, gy - 0.6 * h))
    top = Canvas(fc)
    comic.say(top.ctx, fc, "G01", 640, 210, w=820, tail=(hx - 0.3 * rs * 100, hy - h * 0.12),
              avoid=lambda TT: stray_box(ranger_cam(TT), RANGER_FLAG))
    return img, flags, top

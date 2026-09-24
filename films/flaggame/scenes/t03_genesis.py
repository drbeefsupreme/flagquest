"""t03 — In the Beginning (43.29-67.46)  · owner T03Genesis

Signature: a moving op-art illusion genesis, then a hurricane literally made of stink lines.

43.29 genesis   the black void breathes open into an infinite op-art hexagon floor; six poles sprout at the
                vertices of the central hexagon (furled), orbiting camera. P07 caption.
45.55 "FLAGS"   the six cloths burst open around the ring (real cloth sims, radial breath).
48.66 "Yellow"  the flags glow.  49.42 S05 from the present: the flashback FREEZES (story time stops),
                off-panel balloons; 52.30 "ABSENCE" time runs again; the floor begins to swirl.
54.6-56.5       the camera cranes down to eye level while the op-art floor dissolves into the engraved
                mud of Alchemy IX: the crowd of muddy hippies, the wall of camps, a Dore sky. P09 caption.
57.34 stench    stink lines rise off every head; the flags' breath turns into a vortex.
59.02-59.34     the poles are torn out of the mud; 59.34 hurricane (top-down chart), 61.65 boombox,
62.95 rain, 64.63 the Ranger's radio -> t04 at 67.46.
"""
import math

import cv2
import numpy as np

from vx import *
from vx import ink, comic
from scenes import t03_opart as OP
from scenes import t03_flags as FL
from scenes import t03_burn as BU
from scenes import t03_storm as SM
from scenes import t03_forecast as FC
from scenes import t03_sound as SD

POST = dict(bloom=0.0, grain=0.0, vignette=0.12)

T_DISSOLVE = (54.75, 56.45)
T_HURRICANE = SM.T_IN


def setup(S):
    tracks = FL.setup_flags(S)
    SD.export(S, tracks, S.tl)
    return {"flags": tracks, "sky": BU.make_sky(S), "camps": BU.make_camps(), "crowd": BU.make_crowd(),
            "stinks": SM.make_stinks()}


# ------------------------------------------------------------------ helpers
def _flag_layer(fc, geoms, glow=0.0):
    lay = Canvas(fc)
    for g in geoms:
        FL.draw_geom(lay.ctx, g, glow=glow)
    return lay.rgba()


def _glow_yellow(T):
    """the flags' own soft glow on the word 'Yellow' (P07 48.66)"""
    return 0.85 * pulse(T, 48.95, 0.45) + 0.25 * smoothstep(48.6, 49.2, T) * (1 - smoothstep(49.6, 51.0, T))


_KEY = {}


def _dissolve_key(fc):
    """static per-pixel reveal order: horizon first, ragged blotted edge (cached per resolution)"""
    k = _KEY.get((fc.h, fc.w))
    if k is None:
        h, w = fc.h, fc.w
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        xs /= fc.s
        ys /= fc.s
        n = value_noise2(xs * 0.006, ys * 0.006, 41) * 0.6 + value_noise2(xs * 0.025, ys * 0.025, 42) * 0.25
        k = (0.62 * (ys / 1080.0) + 0.38 * (n * 0.5 + 0.5)).astype(np.float32)
        _KEY[(fc.h, fc.w)] = k
    return k


def _mask_dissolve(fc, T):
    """0 = op-art, 1 = engraving. Ink floods in from the horizon down with a ragged, blotted edge."""
    p = (T - T_DISSOLVE[0]) / (T_DISSOLVE[1] - T_DISSOLVE[0])
    if p <= 0:
        return None
    if p >= 1:
        return 1.0
    e = p * 1.3 - 0.15
    return np.clip((e - _dissolve_key(fc)) / 0.05 + 0.5, 0, 1)


# ------------------------------------------------------------------ stink lines
def _swirl(x, y, vc, amt):
    """rotate (x, y) about the vortex centre; amt ~ radians at 400 px, stronger near the eye; slight inflow"""
    dx, dy = x - vc[0], (y - vc[1]) * 1.7
    r = math.hypot(dx, dy) + 1e-3
    ang = amt * (0.6 + 500.0 / (r + 250.0))
    ca, sa = math.cos(ang), math.sin(ang)
    pull = 1.0 / (1.0 + 0.12 * abs(amt))
    return vc[0] + (dx * ca - dy * sa) * pull, vc[1] + (dx * sa + dy * ca) * pull / 1.7


def _wavy(ctx, x0, y0, L, amp, ph, n, dx=0.0, dy=-1.0):
    """a wavy stink line starting at (x0, y0) running L px along the unit direction (dx, dy)"""
    px, py = -dy, dx
    pts = []
    for i in range(n + 1):
        u = i / n
        o = amp * (0.5 + u) * math.sin(2 * math.pi * (u * 1.6 - ph))
        pts.append((x0 + dx * L * u + px * o, y0 + dy * L * u + py * o))
    smooth(ctx, pts)


VC = (1000.0, 30.0)                    # where the stench's vortex forms on screen (the sky swirls about it too)


STINK_T = 57.339


def _stink(ctx, anchors, people, T, vort=0.0, vc=VC):
    """comic stink lines standing over every head + wisps peeling off and rising (black ink);
    vort 0..1: the wisps accelerate and spiral up into the vortex"""
    if T < STINK_T:
        return
    ctx.set_line_cap(1)
    ctx.set_source_rgba(0, 0, 0, 1)
    for a, p in zip(anchors, people):
        hx, hy = a.get("top", a.get("head"))
        hr = p["h"] * 0.09                   # emanata unit: comic scale, not anatomy
        sd = p["seed"]
        t0 = STINK_T + (sd % 97) / 97 * 0.32
        if T < t0:
            continue
        small = p["h"] < 70
        grow = ease_out_back(min(1.0, (T - t0) / 0.3))
        w0 = max(1.4, hr * 0.15)
        # the classic three wavy lines, the wave crawling upward
        ctx.set_line_width(w0)
        for j in ((0,) if small else (-1, 0, 1)):
            L = hr * (2.6 + 0.5 * ((sd >> (3 + j)) % 5) / 5) * grow * (1 - 0.15 * abs(j))
            _wavy(ctx, hx + j * hr * 0.8, hy - hr * 0.3 - abs(j) * hr * 0.15, L, hr * 0.22,
                  T * 1.7 + j * 0.3 + sd * 0.013, 12)
            ctx.stroke()
        # wisps: peel off every ~0.7 s and rise, faster and swirling as the hurricane forms
        per = 0.7 if not small else 1.3
        life = 2.4
        k0 = math.floor((T - t0) / per)
        for kk in range(int(k0), int(k0) - int(life / per) - 1, -1):
            tb = t0 + kk * per + 0.35
            age = T - tb
            if age < 0 or age > life:
                continue
            ua = age / life
            acc = 1.0 + 4.0 * vort
            rise = hr * (3.0 + 6.0 * age + 9.0 * age * age * acc)
            L = hr * (2.2 + 1.5 * vort)
            ctx.set_line_width(w0 * (1.0 - ua) ** 0.7)
            jj = ((sd >> 5) + kk) % 3 - 1
            x0 = hx + jj * hr * 0.6 + hr * 0.8 * math.sin(age * 2.1 + sd)
            y0 = hy - rise
            ddx, ddy = 0.0, -1.0
            if vort > 0:
                # carried round the vortex: the wisp's base swirls, the wisp turns into the flow
                x0, y0 = _swirl(x0, y0, vc, vort * (0.4 + 2.6 * ua))
                rx, ry = x0 - vc[0], (y0 - vc[1]) * 1.7
                rr = math.hypot(rx, ry) + 1e-3
                tx, ty = -ry / rr, rx / rr / 1.7
                w_ = min(1.0, vort * (0.5 + ua))
                ddx, ddy = (1 - w_) * 0.0 + w_ * tx, (1 - w_) * -1.0 + w_ * ty
                nrm = math.hypot(ddx, ddy) + 1e-6
                ddx, ddy = ddx / nrm, ddy / nrm
            _wavy(ctx, x0, y0, L, hr * 0.25, T * 1.3 + kk * 0.37, 12 if not small else 8, ddx, ddy)
            ctx.stroke()


# ------------------------------------------------------------------ the engraved Burn (crowd shot)
def burn_frame(fc, st, T, cam, flags):
    """engraved Burn with the flag layer `flags` spotted in; returns rgb.
    Background (sky, mud, camps) is ruled with horizontal burin lines like Dore's plates; people are
    printed on their own layers so their hatching follows form."""
    wet = 0.0
    dark = smoothstep(57.6, 59.3, T)
    tone, gmask = BU.mud_tone(fc, cam, T, wet=wet)
    swirl = 1.4 * smoothstep(58.0, 59.4, T) ** 1.4
    sky = BU.sky_tone(fc, cam, st["sky"], T, dark=dark, swirl=swirl, swirl_c=VC)
    world = gmask * tone + (1 - gmask) * sky
    wc = Canvas(fc)
    wc.paint_array(np.repeat(world[..., None], 3, axis=2), 0, 0)
    BU.draw_camps(wc.ctx, cam, st["camps"], T, wind=dark)
    # chiaroscuro: a pool of light on the six Flags, the crowd sinking into shadow toward the edges
    lm = _light_map(fc, cam, dark)
    img = ink.engrave(fc, wc.rgb() * lm, angle=0.0)
    ring_z = float(cam.project(np.zeros(3))[2])
    ppl = BU.crowd_people(cam, st["crowd"], T, fc.t, wet=wet, heroes=True)
    back = [p for p in ppl if p["depth"] >= ring_z - 1.2]
    front = [p for p in ppl if p["depth"] < ring_z - 1.2]
    vort = smoothstep(58.25, 59.34, T)
    bl = Canvas(fc)
    ab = BU.draw_people(bl.ctx, back, T)
    _stink(bl.ctx, ab, back, T, vort=vort)
    img = over(img, ink.print_layer(fc, _shade(bl.rgba(), lm), style="engrave"))
    img = ink.spot(img, flags)
    if front:
        fcv = Canvas(fc)
        af = BU.draw_people(fcv.ctx, front, T)
        img = over(img, ink.print_layer(fc, _shade(fcv.rgba(), lm), style="engrave"))
        sc = Canvas(fc)
        _stink(sc.ctx, af, front, T, vort=vort)
        img = over(img, sc.rgba() * (1.0 - flags[..., 3:4]))
    return img


def _light_map(fc, cam, dark=0.0):
    """(h, w, 1) multiplier: bright on the flag ring, falling into shadow toward the frame edges"""
    rx, ry, _ = cam.project(np.array([0.0, 1.0, 0.0]))
    ys, xs = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
    xs = xs / fc.s - rx
    ys = ys / fc.s - ry
    pool = np.exp(-((xs / 820.0) ** 2 + (ys / 560.0) ** 2))
    lm = (0.40 + 0.60 * pool ** 0.8) * (1.0 - 0.22 * dark * (1.0 - pool))
    return lm[..., None].astype(np.float32)


def _shade(rgba, lm):
    out = rgba.copy()
    out[..., :3] *= lm
    return out


# ------------------------------------------------------------------ genesis (+ dissolve into the Burn)
def genesis(fc, st):
    T = fc.T
    tau = OP.tau_of(T)
    cam = BU.world_cam(T)
    geoms = FL.all_geoms(st["flags"], cam, T)
    m = _mask_dissolve(fc, T)
    flags = _flag_layer(fc, geoms, glow=_glow_yellow(T) if m is None else 0.0)
    if m is None or not np.isscalar(m):
        cov = OP.opart_cov(fc, cam, OP.params(T, tau))
        art = OP.to_rgb(cov)
    if m is None:
        img = ink.spot(art, flags)
    else:
        burn = burn_frame(fc, st, T, cam, flags)
        if np.isscalar(m):
            img = burn
        else:
            img = burn * m[..., None] + ink.spot(art, flags) * (1 - m[..., None])
    k = _panel_k(T)
    if k < 1.0:
        # the flashback freezes into a printed panel on the tract page: the present talks from outside it
        img = _panelize(fc, img, k)
        flags = _panelize(fc, flags, k, rgba=True)
    top = Canvas(fc)
    avoid = _avoid_fn(st)
    # the ring sits right of centre (lens shift): the left third carries the lettering
    comic.say(top.ctx, fc, "P07", 48, 44, w=560, avoid=avoid)
    # interruptions from the present: off-panel balloons, tails running off the left edge
    comic.say(top.ctx, fc, "S05", 300, 905, w=440, tail=(-70, 1010), avoid=avoid)
    comic.say(top.ctx, fc, "P08", 320, 470, w=520, tail=(-70, 380), tokens=(0, 9), avoid=avoid)
    comic.say(top.ctx, fc, "P08", 48, 1036, kind="footnote", tokens=(9, None), anchor="bl", avoid=avoid)
    comic.say(top.ctx, fc, "P09", 48, 44, w=980, hold=0.25, avoid=avoid)
    return over(img, comic.protect(top.rgba(), flags))


PAGE = np.array([0x2B, 0x2B, 0x2D], np.float32) / 255.0


def _panel_k(T):
    """scale of the frozen flashback panel (1 = full frame)"""
    return 1.0 - 0.085 * smootherstep(0.0, 1.0, OP.pause_amount(T))


def _panelize(fc, img, k, rgba=False):
    M = np.array([[k, 0.0, 960.0 * (1 - k)], [0.0, k, 540.0 * (1 - k)]], np.float32)
    Mp = M.copy()
    Mp[:, 2] *= fc.s
    if rgba:
        return cv2.warpAffine(img, Mp, (fc.w, fc.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out = cv2.warpAffine(img, Mp, (fc.w, fc.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=tuple(float(v) for v in PAGE))
    c = Canvas(fc)
    ctx = c.ctx
    x0, y0 = 960.0 * (1 - k), 540.0 * (1 - k)
    ctx.rectangle(x0, y0, 1920.0 * k, 1080.0 * k)
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(7.0)
    ctx.stroke()
    return over(out, c.rgba())


_AV = {}


def _avoid_fn(st):
    """avoid(T) -> cloth boxes of every flag on screen at global time T (the lettering is placed against
    the flags' whole path over the line's lifetime); memoised per frame"""
    def f(TT):
        key = int(round(TT * 24))
        v = _AV.get(key)
        if v is None:
            T = key / 24.0
            if T >= SM.T_IN:
                z, cx, cy = SM.view(T)
                v = SM.flag_boxes(T, z, cx, cy)
            else:
                gs = FL.all_geoms(st["flags"], BU.world_cam(T), T)
                k = _panel_k(T)
                v = [(960 + (b[0] - 970) * k, 540 + (b[1] - 550) * k, 960 + (b[2] - 950) * k, 540 + (b[3] - 530) * k)
                     for b in (g["cloth_bbox"] for g in gs)]
            _AV[key] = v
        return v
    return f


# ------------------------------------------------------------------ the hurricane of stink (top-down chart)
def storm(fc, st):
    T = fc.T
    img, flags = SM.render_storm(fc, T, st["sky"], st["stinks"], ink)
    img = ink.spot(img, flags)
    top = Canvas(fc)
    z, cx, cy = SM.view(T)
    comic.say(top.ctx, fc, "P09", 48, 44, w=980, hold=0.25, avoid=_avoid_fn(st))
    return over(img, comic.protect(top.rgba(), flags))


def render(fc, st):
    T = fc.T
    if T < T_HURRICANE:
        return genesis(fc, st)
    if T < FC.T_BOOM:
        return storm(fc, st)
    if T < FC.T_RANGER:
        img, flags, top = FC.boombox_frame(fc, st)
    else:
        img, flags, top = FC.ranger_frame(fc, st)
    return over(img, comic.protect(top.rgba(), flags))


def post(fc, st):
    """per-frame film-look overrides (also shadows the vx.post module that `from vx import *` drags in)"""
    return {}

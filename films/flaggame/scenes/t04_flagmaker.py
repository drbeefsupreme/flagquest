"""t04 — THE FLAG MAKER (67.46–105.67). Owner: T04FlagMaker.

An animated Gustave-Doré engraving in the rain, then a crowd whose incomprehension becomes a wall of type
that drowns even the Devil.

Shots (global s):
  A walk      67.46  the crowd parts; the Flag Maker walks among seven candlesticks (P10, M01)
  B rejected  74.62  hooded rain-poncho Pharisees, over his shoulder (P11, X01 'Shut up!', P12 + FLAGS 4:20)
  C explain   82.55  he explains to the spotted crowd (M02); 87.15 flags_flood: his words flip to FLAGS,
                     the crowd's balloons fill with FLAGS FLAGS FLAGS on the chant beats
  D fez       89.70  a man in a fez preaches it to a vast engraved multitude
  E flood     92.00  M03; the balloons overflow, the rain turns to words, the words pile up (rigid bodies) around
                     his Flag — they cannot touch it — and at 95.72 flood_peak snap into a typeset wall of FLAGS
                     reflowed live around the cloth
  F devil     97.02  the wall scrolls away: the Devil on his throne with a lyre (V01); his words keep getting
                     stamped FLAGS; 102.42 devil_glitch V02 — possessed by the word; the wall closes. Hard cut.
Compositing: grey world (cairo) -> vx.ink.engrave (page-locked to the camera) -> vx.ink.spot(flag layer) with
foreground occluders cut out -> lettering/kinetic type (masked off the flag) -> POST (no grain, no bloom).
"""
import inspect
import json
import math

import cairo
import cv2
import numpy as np

from vx import *
from vx import ink, comic, foley
from vx.config import CACHE
from vx.flag import Flag, FlagTrack, draw_flag
from vx.chars import Character, draw_crowd

from scenes import t04_world as tw
from scenes import t04_type as ty

POST = dict(bloom=0.0, grain=0.0, vignette=0.14)


def post(fc, state):
    """per-frame film look: no vignette on the white FLAGS walls (they are the page itself)"""
    if T_PEAK <= fc.T < T_SCROLL0 or fc.T >= 104.6:
        return dict(vignette=0.0)
    return None


# ---------------------------------------------------------------- shot table (global seconds)
T_A, T_B, T_C, T_D, T_E, T_F = 67.4583, 74.62, 82.55, 89.70, 92.00, 97.02
T_END = 105.6667
T_FLOOD, T_PEAK, T_GLITCH = 87.153, 95.722, 102.415
T_SCROLL0, T_SCROLL1 = 97.02, 97.30
REF_H = 500.0                           # reference figure height for local-frame flag sims
PAPERW = (1.0, 1.0, 1.0)
INKC = (0.04, 0.04, 0.045)

_ENG_PARAMS = set(inspect.signature(ink.engrave).parameters)


def engrave(fc, grey, cam=None, **kw):
    if cam is not None and "space" in _ENG_PARAMS:
        kw.update(space="page", origin=(cam.x - W / 2 / cam.zoom, cam.y - H / 2 / cam.zoom), zoom=cam.zoom)
    kw = {k: v for k, v in kw.items() if k in _ENG_PARAMS}
    return ink.engrave(fc, grey, **kw)


def G(v):
    return (v, v, v)


# ---------------------------------------------------------------- chant grid
def chant_beats(tl):
    p = CACHE / "foley" / "chant_grid.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            d = (d.get("beats") or d.get("t")) if isinstance(d, dict) else d
            return [float(x) for x in d]
        except Exception:
            pass
    c = tl.d["chant"]
    t0, tp, t1 = c["start"], c["peak"], c["end"]
    out, t, ph = [], t0, 0.0
    dt = 1 / 240
    out.append(t0)
    while t < t1:
        bpm = 126 + 12 * min(1.0, (t - t0) / (tp - t0))
        ph += bpm / 60 * dt
        t += dt
        if ph >= 1.0:
            ph -= 1.0
            out.append(round(t, 3))
    return out


def beat_env(T, beats, decay=0.18):
    """1 on each chant beat, decaying"""
    e = 0.0
    for b in beats:
        if b <= T:
            e = max(e, math.exp(-(T - b) / decay))
    return e


def beats_before(T, beats):
    return sum(1 for b in beats if b <= T)


# ---------------------------------------------------------------- perspective for shot A
HY, VPX, GY, FX = 452.0, 990.0, 610.0, 900.0


def proj(X, Z):
    return VPX + X * FX / Z, HY + GY / Z, 1.0 / Z


PH = 640.0                              # person height at Z = 1


# ================================================================ setup
def setup(S):
    tl = S.tl
    st = {"beats": chant_beats(tl)}
    rng = np.random.default_rng(404)
    st["fm"] = Character("flagmaker", seed=7, render="ink")
    st["fez"] = Character("hippie", seed=21, render="ink", hat="fez", prop="megaphone")
    st["pharisees"] = [Character("pharisee", seed=40 + k, render="ink", wet=True, robe="#202024") for k in range(8)]
    # ---- A: sky, crowd, candlesticks
    st["billA"] = tw.billow_array(tw.make_billows(4, (1000, 330), n_clusters=44, rx=1400, ry=560, ring0=0.32,
                                                  size=0.9))
    st["noise"] = tw.fbm_tex((270, 480), 9, base=6, octaves=5)
    crowd = []
    for side in (-1, 1):
        for k in range(78):
            Z = 0.95 + rng.random() ** 1.3 * 9.5
            X = side * (0.92 + rng.random() ** 1.3 * 2.9 * max(1.0, Z / 2.2))
            crowd.append(dict(X=X, Z=Z, seed=int(rng.integers(0, 9999)), side=side,
                              t_off=float(rng.uniform(0, 5)), sway=float(rng.uniform(0.5, 1.5)),
                              mud=float(rng.uniform(0.5, 1.0))))
        for k in range(2):                                   # repoussoir: big dark shoulders at the corners
            crowd.append(dict(X=side * (1.05 + 0.35 * k), Z=0.74 + 0.06 * k, seed=900 + k + (side > 0) * 10,
                              side=side, t_off=float(k), sway=1.0, mud=1.0))
    st["crowdA"] = crowd
    st["sticks"] = [(-0.80, 1.55), (0.80, 1.95), (-0.80, 2.45), (0.80, 3.05), (-0.80, 3.85), (0.80, 4.9),
                    (-0.80, 6.3)]
    # ---- flag sims
    st["tracks"] = {}
    _sim_walk(S, st)
    _sim_rejected(S, st)
    _sim_explain(S, st)
    _sim_flood(S, st)
    # ---- C crowd (explain), D multitude
    st["crowdC"] = _crowd_explain(rng)
    st["multitude"] = _multitude(np.random.default_rng(77))
    # ---- flood pile (rigid bodies)
    _setup_flood(S, st)
    _export(S, st)
    return st


# ---------------------------------------------------------------- Flag Maker motion (shot A)
T_STOP = 70.0          # he arrives
T_PLANT = 71.73        # "FLAGS": he slams the Flag into the mud (impact)
T_RAISE0, T_RAISE1 = 69.45, 70.0     # from the shoulder to upright in his hand as he stops
T_WIND = T_PLANT - 0.34              # wind-up before the thrust
POLE_RATIO = 1.39      # 96" pole / 69" man
HELD_A = (-0.42 * REF_H, -0.33 * REF_H, -0.03)     # local frame: held upright at his side (screen left)
PLANTED_A = (-0.42 * REF_H, 3.0, -0.035)          # ...then planted in the mud
HAND_FRAC_HELD, HAND_FRAC_PLANTED = 0.16, 0.36


def fm_walk_state(T):
    """(X, Z, walking 0..1) of the Flag Maker in shot A"""
    u = T - T_A
    stop = T_STOP - T_A
    p = clamp(u / stop)
    e = 1 - (1 - p) ** 1.5
    Z = lerp(2.7, 1.28, e)
    X = lerp(-0.06, 0.0, e)
    walking = 1.0 - smoothstep(stop - 0.35, stop + 0.15, u)
    return X, Z, walking


def _carry_pose(T):
    X, Z, walking = fm_walk_state(T)
    return {"walk": walking, "stand": 1 - walking, "shoulder_flag": 1.0}


def _carry_pole(ch, T):
    a = ch.anchors(0.0, 0.0, REF_H, _carry_pose(T), T, facing=0, turn=0.12, render="ink")
    return a.get("pole", (-60.0, -260.0, -0.5)), a


def pole_A(st, T):
    """the Flag's pole in the Flag Maker's local frame: on his shoulder while he walks, swung upright into his
    hand as he stops, then — on 'FLAGS' — a wind-up and an accelerating thrust into the mud, and a quiver"""
    ch = st["fm"]
    if T < T_RAISE0:
        return _carry_pole(ch, T)[0]
    if T < T_RAISE1:
        P0 = st.get("_pole_raise")
        if P0 is None:
            P0 = st["_pole_raise"] = _carry_pole(ch, T_RAISE0)[0]
        u = ease_in_out(clamp((T - T_RAISE0) / (T_RAISE1 - T_RAISE0)), 2)
        return tuple(lerp(a, b, u) for a, b in zip(P0, HELD_A))
    bx0, by0, a0 = HELD_A
    bx1, by1, a1 = PLANTED_A
    if T < T_WIND:
        return (bx0, by0 + 0.01 * REF_H * math.sin((T - T_RAISE1) * 3.0), a0)
    if T < T_PLANT:
        u = clamp((T - T_WIND) / (T_PLANT - T_WIND))
        lift = 0.10 * REF_H * math.sin(math.pi * min(1.0, u / 0.55)) if u < 0.55 else 0.0
        e = ease_in(clamp((u - 0.35) / 0.65), 3)
        return (lerp(bx0, bx1, u), lerp(by0, by1, e) - lift, lerp(a0, a1, u))
    tt = T - T_PLANT
    return (bx1, by1, a1 + 0.045 * math.exp(-tt * 7) * math.sin(tt * 38))


def fm_pose_A(T, st):
    """(X, Z, pose dict, kw): walk with the Flag on his shoulder, stop with it upright in his hand, raise the
    other hand (M01), slam the Flag into the mud on 'FLAGS' (his hand rides the pole), re-grip, preach"""
    X, Z, walking = fm_walk_state(T)
    if T < T_RAISE0:
        return X, Z, _carry_pose(T), {}
    ch = st["fm"]
    f0 = st.get("_hand_frac0")
    if f0 is None:
        P0, a0 = _carry_pole(ch, T_RAISE0)
        hx, hy = a0["hand_r"]
        L = REF_H * POLE_RATIO
        f0 = ((hx - P0[0]) * math.sin(P0[2]) - (hy - P0[1]) * math.cos(P0[2])) / L
        f0 = st["_hand_frac0"] = float(clamp(f0, 0.05, 0.9))
    carry = 1.0 - smoothstep(T_RAISE0, T_RAISE1, T)
    frac = lerp(f0, HAND_FRAC_HELD, 1 - carry)
    frac = lerp(frac, HAND_FRAC_PLANTED, smoothstep(T_PLANT + 0.25, T_PLANT + 0.6, T))
    pole = pole_A(st, T)
    L = REF_H * POLE_RATIO
    hand = (pole[0] + math.sin(pole[2]) * L * frac, pole[1] - math.cos(pole[2]) * L * frac)
    pose = {"walk": walking * carry, "stand": 1.0 - walking * carry}
    if carry > 0:
        pose["shoulder_flag"] = carry
    pr = smoothstep(70.1, 70.5, T) * (1 - smoothstep(T_WIND - 0.1, T_WIND + 0.1, T)) + smoothstep(72.5, 72.95, T)
    if pr > 0:
        pose["preach"] = pr
    crouch = math.exp(-((T - T_PLANT + 0.03) / 0.16) ** 2)       # he leans into the thrust
    kw = dict(hand_r=hand, shape_r="grip", lean=0.12 * crouch, pole_ang=pole[2])
    return X, Z, pose, kw


def _sim_walk(S, st):
    n = int(round((T_B - S.start) * S.fps)) + 2
    fl = Flag(pole=REF_H * POLE_RATIO, seed=41, turbulence=0.8, flutter=0.7)
    st["tracks"]["A"] = fl.simulate(n, lambda i: pole_A(st, S.start + i / S.fps),
                                    lambda i: (190 + 60 * math.sin(i / 37.0), 40, -30), warmup=60)


# ---------------------------------------------------------------- B: Pharisees
PHAR = [  # x, feet_y, h, facing, turn
    (860, 1560, 1060, -1, 0.35), (1090, 1500, 980, -1, 0.25), (1310, 1580, 1100, -1, 0.3), (1530, 1510, 990, -1, 0.2),
    (1740, 1600, 1080, -1, 0.35), (980, 1420, 820, -1, 0.3), (1420, 1420, 820, -1, 0.3), (1880, 1440, 860, -1, 0.3)]
FM_B = (250.0, 1330.0, 900.0)           # Flag Maker from behind, foreground left (feet, height)
POLE_B = (470.0, 1330.0, 0.04)          # his Flag planted beside him


def _sim_rejected(S, st):
    n = int(round((T_C - T_B) * S.fps)) + 2
    L = FM_B[2] * POLE_RATIO

    def pole_fn(i):
        T = T_B + i / S.fps
        su = T - 76.98
        jolt = 0.02 * math.exp(-max(0.0, su) * 5) * math.sin(max(0.0, su) * 30) if su > 0 else 0.0
        return (POLE_B[0], POLE_B[1], POLE_B[2] + jolt)

    fl = Flag(pole=L, seed=42, turbulence=0.9, flutter=0.8)
    st["tracks"]["B"] = fl.simulate(n, pole_fn, lambda i: (230 + 80 * math.sin(i / 29.0), 30, 0), warmup=60)


# ---------------------------------------------------------------- C: explain
FM_C = (300.0, 1075.0, 700.0)
POSE_C = {"hold_flag": 1.0, "preach": 1.0}


def _sim_explain(S, st):
    ch = st["fm"]
    n = int(round((T_D - T_C) * S.fps)) + 2
    L = REF_H * POLE_RATIO

    def pole_fn(i):
        T = T_C + i / S.fps
        a = ch.anchors(0.0, 0.0, REF_H, POSE_C, T, facing=1, turn=0.35, render="ink", pole_len=L)
        return a.get("pole", (30.0, -250.0, 0.0))

    fl = Flag(pole=L, seed=43, turbulence=0.9, flutter=0.8)
    st["tracks"]["C"] = fl.simulate(n, pole_fn, lambda i: (260 + 90 * math.sin(i / 31.0), 30, 0), warmup=60)


def _crowd_explain(rng):
    """the spotted crowd of page 25 panel 1: three overlapping rows, eye level ~ y 470"""
    out = []
    rows = [(930, 480, 13, 640), (1000, 540, 11, 700), (1085, 620, 9, 760)]
    for r, (fy, h, cnt, x0) in enumerate(rows):
        xs = np.linspace(x0, 2250, cnt) + rng.normal(0, 26, cnt)
        for x in xs:
            out.append(dict(x=float(x), y=float(fy + rng.normal(0, 6)), h=float(h * rng.uniform(0.9, 1.08)),
                            seed=int(rng.integers(0, 9999)), t_off=float(rng.uniform(0, 5)), row=r,
                            mud=float(rng.uniform(0.6, 1.0))))
    out.sort(key=lambda p: p["y"])
    return out


# ---------------------------------------------------------------- D: multitude (sea of heads)
def _multitude(rng):
    heads = []
    for k in range(5200):
        v = rng.random() ** 1.7                  # 0 far .. 1 near
        y = 395 + v * 700
        s = 3.0 + v * 42.0
        x = rng.uniform(-200, W + 200)
        heads.append((x, y, s, rng.random(), rng.random()))
    heads.sort(key=lambda h: h[1])
    return np.array(heads, np.float32)


# ---------------------------------------------------------------- E: flood
FM_E = (690.0, 1150.0, 800.0)           # feet (just below the frame), height
POLE_E = (905.0, 1150.0, 0.035)         # his Flag planted beside him: cloth upper right, alone in the wall


def fm_E_pose(T):
    return {"stand": 1.0, "preach": 0.8 * smoothstep(92.0, 92.6, T)}


def pole_E(T):
    bx, by, ang = POLE_E
    return (bx, by - scroll_offset(T), ang)             # the page scroll carries the flag up with the wall


def fm_E_kw(T):
    L = FM_E[2] * POLE_RATIO
    bx, by, ang = POLE_E
    return dict(hand_r=(bx + math.sin(ang) * L * 0.46, by - math.cos(ang) * L * 0.46), shape_r="grip")


def _sim_flood(S, st):
    n = int(round((T_SCROLL1 + 0.25 - T_E) * S.fps)) + 2
    fl = Flag(pole=FM_E[2] * POLE_RATIO, seed=44, turbulence=1.0, flutter=0.9)
    st["tracks"]["E"] = fl.simulate(n, lambda i: pole_E(T_E + i / S.fps),
                                    lambda i: (320 + 120 * math.sin(i / 23.0), 20, 0), warmup=60)


def scroll_offset(T):
    """page scroll between the wall and the Devil panel (design px, positive = content moved up)"""
    return H * ease_in_out(clamp((T - T_SCROLL0) / (T_SCROLL1 - T_SCROLL0)), 3)


# ================================================================ render
def render(fc, st):
    T = fc.T
    if T < T_B:
        return shot_walk(fc, st, T)
    if T < T_C:
        return shot_rejected(fc, st, T)
    if T < T_D:
        return shot_explain(fc, st, T)
    if T < T_E:
        return shot_fez(fc, st, T)
    if T < T_SCROLL0:
        return shot_flood(fc, st, T)
    return shot_devil(fc, st, T)


# ---------------------------------------------------------------- composition helpers
def finish(fc, grey, flags, front=None, cam=None, rain=None, eng_kw=None):
    """grey world (+front occluders) -> engrave -> spot(flags minus occluders)"""
    if front is not None:
        fr = front.rgba()
        grey = over(grey, fr)
        occ = fr[..., 3:4]
    else:
        occ = None
    if rain is not None:
        grey = tw.apply_rain(grey, rain)
    img = engrave(fc, grey, cam, **(eng_kw or {}))
    fl = flags.rgba() if flags is not None else None
    if fl is not None:
        if occ is not None:
            fl = fl * (1.0 - occ)
        img = ink.spot(img, fl)
    return img, fl


def letter(fc, img, draw_fn, flag_rgba=None, margin=16.0):
    """lettering canvas composited over img, masked off the flag"""
    top = Canvas(fc)
    draw_fn(top.ctx)
    rgba = top.rgba()
    if flag_rgba is not None:
        rgba = ty.mask_out(rgba, ty.flag_block_mask(flag_rgba[..., 3], fc.s, margin))
    return over(img, rgba)


def cloth_bbox(track, i, xf=None, pad=10.0):
    """screen bbox of the cloth at frame i; xf(x, y) -> screen (x, y)"""
    V, _ = track._frame(i)
    xs, ys = V[..., 0].ravel(), V[..., 1].ravel()
    pts = [(float(xs.min()), float(ys.min())), (float(xs.max()), float(ys.max()))]
    if xf is not None:
        pts = [xf(*p) for p in pts]
    (x0, y0), (x1, y1) = pts
    return (min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad)


def rain_mask(fc, T, *, density=1.0, seed=0, alpha=1.0, hole=None, slant=0.18):
    draw_rain, _ = tw.get_rain()
    cv = Canvas(fc)
    draw_rain(cv.ctx, T, rect=(0, 0, W, H), density=density, slant=slant, seed=seed, alpha=alpha)
    m = cv.rgba()[..., 3]
    if hole is not None:
        hx, hy, r0, r1 = hole
        m = m * (1.0 - radial_mask(fc, hx, hy, r0, r1)[..., 0])
    return m


def mouth(fc, who):
    return fc.tl.mouth(who, fc.T)


# ================================================================ SHOT A — the walk among the candlesticks
T_A2 = 70.05          # cut to a medium-wide on the Flag Maker as he stops ("By FINDING...")


def cam_A(T):
    if T < T_A2:
        e = ease_in_out_sine(clamp((T - T_A) / (T_A2 - T_A)))
        return Camera(x=lerp(960, 975, e), y=lerp(540, 560, e), zoom=lerp(1.0, 1.06, e))
    e = ease_in_out_sine(clamp((T - T_A2) / (T_B - T_A2)))
    return Camera(x=lerp(905, 918, e), y=lerp(452, 440, e), zoom=lerp(1.10, 1.17, e))


def fm_draw_kw_A(T, st):
    return fm_pose_A(T, st)


def shot_walk(fc, st, T):
    cam = cam_A(T)
    i = (T - S_start(fc)) * fc.fps
    ch = st["fm"]
    X, Z, pose, kw = fm_draw_kw_A(T, st)
    fx, fy, k = proj(X, Z)
    fh = PH * 1.08 * k
    kk = fh / REF_H
    track = st["tracks"]["A"]
    world = Canvas(fc, bg=G(0.6))
    ctx = world.ctx
    # ---- sky (screen space, parallax 0.25)
    ox, oy = (cam.x - 960) * 0.25, (cam.y - 540) * 0.25
    sky = tw.storm_sky(fc, st["billA"], (1000 - ox, 330 - oy), T - T_A, noise=st["noise"], dark=0.2, far=0.34,
                       rim=0.55, zoom=1 + (cam.zoom - 1) * 0.25)
    world.paint_array(np.repeat(sky[..., None], 3, axis=2))
    ctx.save()
    cam.apply(ctx)
    head = (fx, fy - fh * 0.9)
    # glory behind the Flag Maker (engraved rays through the rain)
    tw.glory(ctx, head[0], head[1] - 20 * k, 170 * k, 1500, T - T_A, n=46, strength=0.9)
    _camps(ctx, T)
    tw.mud_ground(ctx, HY, 1300, t=T, tone0=0.34, tone1=0.14)
    _aisle(ctx, T)
    tw.light_pool(ctx, fx, fy, 420 * k * 1.6, 90 * k * 1.6, 0.55)
    # ---- depth-sorted: crowd, candlesticks, Flag Maker
    items = []
    for p in st["crowdA"]:
        items.append((p["Z"], "p", p))
    for j, (sx, sz) in enumerate(st["sticks"]):
        items.append((sz, "c", (j, sx, sz)))
    items.append((Z, "fm", None))
    items.sort(key=lambda it: -it[0])
    front = Canvas(fc)
    fctx = front.ctx
    fctx.save()
    cam.apply(fctx)
    anc = None
    m = mouth(fc, "FLAGMAKER")
    for zz, kind, obj in items:
        tgt = ctx if zz >= Z else fctx
        if kind == "p":
            _crowd_person_A(tgt, obj, T, (fx, fy - fh * 0.8), Z)
        elif kind == "c":
            j, sx, sz = obj
            cx, cy, ck = proj(sx, sz)
            tw.candlestick(tgt, cx, cy, PH * 1.32 * ck, T, seed=j, light_dx=(fx - cx) / 500)
        else:
            for c_, layer in ((ctx, "back"), (fctx, "front")):
                c_.save()
                c_.translate(fx, fy)
                c_.scale(kk, kk)
                a = ch.draw(c_, 0, 0, REF_H, pose, T, mouth=m, facing=0, turn=0.12, expr="serene", render="ink",
                            layer=layer, look=(0.0, 0.1), **kw)
                c_.restore()
                anc = anc or a
    fctx.restore()
    ctx.restore()
    # ---- flag plate (the only colour)
    flags = Canvas(fc)
    c2 = flags.ctx
    c2.save()
    cam.apply(c2)
    c2.translate(fx, fy)
    c2.scale(kk, kk)
    track.draw(c2, i, light=(-0.2, -0.8, 0.55), glow=0.12)
    c2.restore()
    hs = cam.to_screen(*head)
    rain = rain_mask(fc, T, density=1.0, seed=4, hole=(hs[0], hs[1] + 140 * k * cam.zoom, 90, 380 * k * 2))
    img, fl = finish(fc, world.rgb(), flags, front, cam, rain)
    ms = cam.to_screen(*anc_mouth(anc, fx, fy, kk))
    cb = cloth_bbox(track, i, lambda x, y: cam.to_screen(fx + x * kk, fy + y * kk))

    def cloth_at(TT):
        cm = cam_A(TT)
        X_, Z_, _, _ = fm_draw_kw_A(TT, st)
        fx_, fy_, k_ = proj(X_, Z_)
        kk_ = PH * 1.08 * k_ / REF_H
        return cloth_bbox(track, (TT - S_start(fc)) * fc.fps, lambda x, y: cm.to_screen(fx_ + x * kk_, fy_ + y * kk_))
    av = lambda TT: [cloth_at(min(TT, T_B - 0.05))]  # noqa: E731

    def lettering(c):
        comic.say(c, fc, "P10", 40, 880, w=700)
        comic.say(c, fc, "M01", 1500, 330, w=640, tail=(ms[0] + 14, ms[1] - 6), avoid=av)
    return letter(fc, img, lettering, fl)


def anc_mouth(anc, fx, fy, kk):
    mx, my = anc["mouth"]
    return fx + mx * kk, fy + my * kk


def S_start(fc):
    return fc.S.start


def rimmed(ctx, person, T, light_xy, *, tone=0.16, rim=1.0, rim_w=None, ch=None):
    """a Doré crowd figure: a dark mass with a hard white rim on the side facing the light (drawn as a white
    silhouette nudged toward the light, then the dark silhouette over it). With ch=Character, `person` holds
    x, y, h, pose + any Character.draw kwargs (hand_r, lean, ...) and is drawn directly."""
    x, y, h = person["x"], person["y"], person["h"]
    dx, dy = light_xy[0] - x, light_xy[1] - (y - h * 0.7)
    dn = math.hypot(dx, dy) + 1e-6
    w = rim_w if rim_w is not None else max(1.4, h * 0.011)

    def one(px, py, sil):
        if ch is None:
            return draw_crowd(ctx, [dict(person, x=px, y=py, silhouette=sil)], T)[0]
        kw = {k: v for k, v in person.items() if k not in ("x", "y", "h", "pose", "kind", "seed")}
        return ch.draw(ctx, px, py, h, person.get("pose", "stand"), T, silhouette=sil, **kw)
    if rim > 0:
        one(x + dx / dn * w, y + dy / dn * w * 0.6, (rim, rim, rim))
    return one(x, y, (tone, tone, tone))


def _crowd_person_A(ctx, p, T, fm_head, fm_Z):
    """a hippie on the aisle bank, a dark rim-lit mass; those in his way step aside as he approaches"""
    X = p["X"]
    if abs(X) < 1.3:
        near = smoothstep(1.5, 0.3, p["Z"] - fm_Z) if p["Z"] < 4.0 else 0.0
        X = math.copysign(abs(X) * lerp(0.6, 1.0, near) + 0.14 * near, X)
    x, y, k = proj(X, p["Z"])
    h = PH * k * (0.9 + 0.2 * hash01(p["seed"], 1))
    facing = -1 if p["side"] > 0 else 1
    lookx = clamp((fm_head[0] - x) / 400, -1, 1)
    tone = lerp(0.07, 0.40, clamp((p["Z"] - 1.0) / 9.0) ** 0.7)
    person = dict(x=x, y=y, h=h, kind="hippie", seed=p["seed"], pose="stand", facing=facing,
                  turn=0.25 + 0.5 * hash01(p["seed"], 2), look=(lookx, -0.1), expr="bewildered",
                  t_off=p["t_off"], mud=p["mud"], wet=True, render="ink")
    rimmed(ctx, person, T, fm_head, tone=tone, rim=0.97 if p["Z"] < 5 else 0.75)


def _camps(ctx, T):
    """distant camps against the glowing horizon: shade structures, domes, tents (low-contrast silhouettes)"""
    ctx.save()
    rng = np.random.default_rng(12)
    x = -300.0
    set_color(ctx, G(0.38))
    while x < W + 300:
        kind = rng.integers(0, 4)
        w = rng.uniform(60, 180)
        hgt = rng.uniform(18, 60)
        if kind == 0:     # dome
            ctx.arc(x + w / 2, HY, w / 2, math.pi, 2 * math.pi)
        elif kind == 1:   # shade structure (posts + sagging tarp)
            ctx.rectangle(x, HY - hgt, 4, hgt)
            ctx.rectangle(x + w - 4, HY - hgt, 4, hgt)
            ctx.move_to(x - 6, HY - hgt)
            ctx.curve_to(x + w * 0.3, HY - hgt + 10, x + w * 0.7, HY - hgt + 10, x + w + 6, HY - hgt)
            ctx.line_to(x + w + 6, HY - hgt - 5)
            ctx.line_to(x - 6, HY - hgt - 5)
        elif kind == 2:   # tent
            ctx.move_to(x, HY)
            ctx.line_to(x + w / 2, HY - hgt)
            ctx.line_to(x + w, HY)
        else:             # box truck / trailer
            ctx.rectangle(x, HY - hgt * 0.8, w, hgt * 0.8)
        ctx.fill()
        x += w * rng.uniform(0.7, 1.2)
    # fog band at the horizon
    ctx.rectangle(-400, HY - 60, W + 800, 90)
    ctx.set_source(lin_grad(0, HY - 60, 0, HY + 30, [(0, (1, 1, 1), 0.0), (0.6, (1, 1, 1), 0.45), (1, (1, 1, 1), 0.0)]))
    ctx.fill()
    ctx.restore()


def _aisle(ctx, T):
    """the trampled muddy aisle, lighter, with sky-bright puddles"""
    ctx.save()
    x0, y0, _ = proj(-0.55, 12)
    x1, y1, _ = proj(0.55, 12)
    x2, y2, _ = proj(0.62, 0.9)
    x3, y3, _ = proj(-0.62, 0.9)
    ctx.move_to(x0, y0)
    ctx.line_to(x1, y1)
    ctx.line_to(x2, y2)
    ctx.line_to(x3, y3)
    ctx.close_path()
    ctx.set_source(lin_grad(0, y0, 0, y2, [(0, (1, 1, 1), 0.35), (0.5, (1, 1, 1), 0.18), (1, (1, 1, 1), 0.05)]))
    ctx.fill()
    for j, (X, Z, rx) in enumerate([(-0.15, 1.3, 0.30), (0.2, 2.1, 0.22), (-0.1, 3.2, 0.25), (0.12, 5.0, 0.2),
                                     (-0.3, 1.9, 0.12)]):
        px, py, k = proj(X, Z)
        tw.puddle(ctx, px, py, rx * FX * k, rx * FX * k * 0.12 * (1 + 0.6 / Z), T, seed=j, sky=0.93)
    ctx.restore()


# ================================================================ SHOT B — the Pharisees
def cam_B(T):
    u = clamp((T - T_B) / (T_C - T_B))
    zx = lerp(1.0, 1.07, ease_in_out_sine(u))
    sx, sy = (0, 0)
    su = T - 76.98
    if 0 <= su < 0.6:
        a = math.exp(-su * 7) * 14
        sx, sy = shake(T, amp=a, freq=18, seed=5)
    return Camera(x=1000 + sx, y=560 + sy, zoom=zx)


def phar_state(T, j):
    """lean/expr/mouth phase for Pharisee j"""
    su = T - 76.98
    lurch = 0.0
    if su > -0.1:
        lurch = math.exp(-max(0.0, su - 0.15) * 1.6) * smoothstep(-0.1, 0.12, su)
    return lurch


def shot_rejected(fc, st, T):
    cam = cam_B(T)
    world = Canvas(fc, bg=G(0.5))
    ctx = world.ctx
    ox = (cam.x - 1000) * 0.3
    sky = tw.storm_sky(fc, st["billA"], (1500 - ox, 120), T - T_A + 3.0, noise=st["noise"], dark=0.18, far=0.34,
                       rim=0.35)
    flash = math.exp(-max(0.0, T - 77.0) * 6) if T >= 76.98 else 0.0
    if flash > 0:
        sky = np.clip(sky + flash * 0.35, 0, 1)
    world.paint_array(np.repeat(sky[..., None], 3, axis=2))
    ctx.save()
    cam.apply(ctx)
    # camp wall behind them (plywood + tarps), dark
    ctx.rectangle(-200, 520, W + 400, 700)
    ctx.set_source(lin_grad(0, 520, 0, 900, [(0, G(0.30)), (1, G(0.18))]))
    ctx.fill()
    set_color(ctx, G(0.1))
    ctx.set_line_width(3)
    for xx in range(-200, W + 400, 160):
        ctx.move_to(xx, 520)
        ctx.line_to(xx, 900)
        ctx.stroke()
    lurch = phar_state(T, 0)
    ph = st["pharisees"]
    order = sorted(range(len(PHAR)), key=lambda j: PHAR[j][1])
    speaking = mouth(fc, "PHARISEES")
    mouths = {}
    for j in order:
        x, fy, h, facing, turn = PHAR[j]
        ex = {"stern": 1.0 - lurch, "furious": lurch} if lurch > 0.01 else "stern"
        lean = 0.08 + 0.22 * lurch * (0.8 + 0.4 * hash01(j, 3))
        m = speaking * (0.8 + 0.4 * hash01(j, 4)) if 76.9 < T < 77.9 else 0.0
        a = ph[j].draw(ctx, x - lurch * 30, fy, h, "stand", T + j * 0.7, mouth=m, facing=facing, turn=turn,
                       expr=ex, lean=lean, render="ink", wet=True, look=(-1, -0.1))
        mouths[j] = a["mouth"]
    ctx.restore()
    # foreground: the Flag Maker from behind, a solid black silhouette (repoussoir), hand on his planted Flag
    front = Canvas(fc)
    fctx = front.ctx
    fctx.save()
    cam.apply(fctx)
    bx, by, bh = FM_B
    L = bh * POLE_RATIO
    hand = (POLE_B[0] + math.sin(POLE_B[2]) * L * 0.5, POLE_B[1] - math.cos(POLE_B[2]) * L * 0.5)
    rimmed(fctx, dict(x=bx, y=by, h=bh, pose="stand", facing=1, turn=2.0, render="ink", hand_r=hand, shape_r="grip",
                      pole_ang=POLE_B[2]), T, (1100, 150), tone=0.04, rim=0.9, rim_w=4.0, ch=st["fm"])
    fctx.restore()
    flags = Canvas(fc)
    c2 = flags.ctx
    c2.save()
    cam.apply(c2)
    tr = st["tracks"]["B"]
    i = (T - T_B) * fc.fps
    tr.draw(c2, i, light=(-0.3, -0.8, 0.5), glow=0.1)
    c2.restore()
    rain = rain_mask(fc, T, density=0.75, seed=5) * 0.6
    img, fl = finish(fc, world.rgb(), flags, front, cam, rain)
    cb = cloth_bbox(tr, i, cam.to_screen)
    mx, my = cam.to_screen(*mouths[2])

    def lettering(c):
        comic.say(c, fc, "P11", 900, 30, w=980)
        comic.say(c, fc, "X01", 1330, 300, tail=(mx, my), avoid=[cb], size=66)
        _p12(c, fc, T)
    return letter(fc, img, lettering, fl)


def _p12(c, fc, T):
    """P12 caption (tokens 0..8) + its synced citation box 'FLAGS 4:20' (page 24: bottom-right)"""
    comic.say(c, fc, "P12", 470, 905, w=980, tokens=(0, 9), hold=0.9)
    comic.say(c, fc, "P12", W - 60, H - 48, kind="footnote", tokens=(9, None), anchor="br", size=40, hold=0.9)


def flagify_u(T, t0, t1):
    """how much of a balloon has turned into FLAGS (eased; 0 before t0)"""
    return ease_in_out(clamp((T - t0) / (t1 - t0)), 2)


# ================================================================ SHOT C — explaining; the words turn to FLAGS
def cam_C(T):
    u = clamp((T - T_C) / (T_D - T_C))
    e = ease_in_out_sine(u)
    return Camera(x=lerp(940, 1000, e), y=lerp(545, 530, e), zoom=lerp(1.0, 1.07, e))


def shot_explain(fc, st, T):
    cam = cam_C(T)
    world = Canvas(fc, bg=G(0.55))
    ctx = world.ctx
    fx, fy, fh = FM_C
    kk = fh / REF_H
    L = REF_H * POLE_RATIO
    sky = tw.storm_sky(fc, st["billA"], (330 - (cam.x - 940) * 0.3, 140), T - T_A + 6.0, noise=st["noise"],
                       dark=0.22, far=0.38, rim=0.5)
    world.paint_array(np.repeat(sky[..., None], 3, axis=2))
    ctx.save()
    cam.apply(ctx)
    _camp_wall(ctx, T)
    tw.mud_ground(ctx, 860, 1300, t=T, seed=21, tone0=0.40, tone1=0.2)
    beats = st["beats"]
    tw.glory(ctx, fx + 20, fy - fh * 0.92, 90, 1100, T, n=40, strength=0.85, lines=True)
    tw.light_pool(ctx, fx + 200, fy, 700, 120, 0.5)
    # the spotted crowd (facing him, lit by his radiance from the left), stink lines
    heads = []
    for p in st["crowdC"]:
        dist = clamp((p["x"] - fx) / 1700)
        person = dict(x=p["x"], y=p["y"], h=p["h"], kind="hippie", seed=p["seed"], pose="stand", facing=-1,
                      turn=0.3 + 0.3 * hash01(p["seed"], 3), look=(-1, -0.05), expr="bewildered",
                      t_off=p["t_off"], mud=p["mud"], wet=True, render="ink", light=(-0.9, -0.4),
                      mouth=_chant_mouth(T, beats, p["seed"]))
        if p["row"] == 0:
            a = rimmed(ctx, person, T, (fx, fy - fh), tone=lerp(0.18, 0.30, dist), rim=0.95)
        else:
            a = draw_crowd(ctx, [person], T)[0]
        heads.append((p, a))
    for p, a in heads:
        if hash01(p["seed"], 31) < 0.45:
            hx, hy = a["head"]
            comic.emanata(ctx, hx, hy - p["h"] * 0.12, p["h"] * 0.13, "stink", t=T + p["t_off"])
    ctx.save()
    ctx.translate(fx, fy)
    ctx.scale(kk, kk)
    m = mouth(fc, "FLAGMAKER")
    anc = st["fm"].draw(ctx, 0, 0, REF_H, POSE_C, T, mouth=m, facing=1, turn=0.35, expr="serene", render="ink",
                        layer="back", pole_len=L)
    ctx.restore()
    ctx.restore()
    front = Canvas(fc)
    fctx = front.ctx
    fctx.save()
    cam.apply(fctx)
    fctx.translate(fx, fy)
    fctx.scale(kk, kk)
    st["fm"].draw(fctx, 0, 0, REF_H, POSE_C, T, mouth=m, facing=1, turn=0.35, expr="serene", render="ink",
                  layer="front", pole_len=L)
    fctx.restore()
    flags = Canvas(fc)
    c2 = flags.ctx
    c2.save()
    cam.apply(c2)
    c2.translate(fx, fy)
    c2.scale(kk, kk)
    tr = st["tracks"]["C"]
    i = (T - T_C) * fc.fps
    tr.draw(c2, i, light=(-0.3, -0.8, 0.5), glow=0.12)
    c2.restore()
    rain = rain_mask(fc, T, density=1.0, seed=6)
    img, fl = finish(fc, world.rgb(), flags, front, cam, rain)
    ms = cam.to_screen(fx + anc["mouth"][0] * kk, fy + anc["mouth"][1] * kk)
    head_s = [(cam.to_screen(*a["head"]), p) for p, a in heads]

    def cloth_at(TT):
        cm = cam_C(TT)
        return cloth_bbox(tr, (TT - T_C) * fc.fps, lambda x, y: cm.to_screen(fx + x * kk, fy + y * kk))
    av = lambda TT: [cloth_at(min(TT, T_D - 0.05))]  # noqa: E731

    def lettering(c):
        comic.say(c, fc, "M02", 1210, 185, w=980, tail=(ms[0] + 10, ms[1] - 10), avoid=av,
                  flagify=flagify_u(T, T_FLOOD, 91.9))
        _crowd_balloons(c, fc, T, beats, head_s, av(T))
    return letter(fc, img, lettering, fl)


def _chant_mouth(T, beats, seed):
    if T < T_FLOOD:
        return 0.0
    return 0.7 * beat_env(T + 0.03 * hash01(seed, 9), beats, 0.2)


def _crowd_balloons(c, fc, T, beats, head_s, avoid=None):
    """from flags_flood: on each chant beat another hippie's balloon pops, and every open balloon gains
    another FLAGS on each beat — the crowd's balloons fill with FLAGS FLAGS FLAGS"""
    if T < T_FLOOD - 0.05:
        return
    nb = beats_before(T, beats)
    cand = [hp for hp in head_s if 620 < hp[1]["x"] < 1830]
    cand.sort(key=lambda hp: hash01(hp[1]["seed"], 17))
    for k in range(min(nb, len(cand))):
        (hx, hy), p = cand[k]
        tb = beats[k]
        age = T - tb
        pop = ease_out_back(clamp(age / 0.16), 2.2)
        grown = nb - k
        nwords = min(9, 1 + grown)
        txt = " ".join([ty.WORD] * nwords)
        bx = clamp(hx + (hash01(k, 3) - 0.5) * 160, 170, W - 170)
        by = clamp(hy - 120 - 90 * hash01(k, 4) - 30 * (k % 3), 380, 640)
        sz = 27 * pop
        if sz < 3:
            continue
        comic.balloon(c, txt, bx, by, 330, tail=(hx, hy - 30), kind="balloon", size=sz, avoid=avoid)


def _camp_wall(ctx, T):
    """a wall of camps behind the crowd (page 25 panel 1): plywood/brick with tarps, lit from the left"""
    ctx.save()
    ctx.rectangle(-300, 330, W + 600, 520)
    ctx.set_source(lin_grad(-300, 0, W + 300, 0, [(0, G(0.72)), (0.4, G(0.55)), (1, G(0.40))]))
    ctx.fill()
    set_color(ctx, G(0.2))
    ctx.set_line_width(2.2)
    for r in range(0, 16):
        y = 340 + r * 30
        ctx.move_to(-300, y)
        ctx.line_to(W + 300, y)
        ctx.stroke()
        off = (r % 2) * 55
        for xx in range(-300 + off, W + 300, 110):
            ctx.move_to(xx, y)
            ctx.line_to(xx, y + 30)
            ctx.stroke()
    ctx.rectangle(-300, 322, W + 600, 14)
    set_color(ctx, G(0.12))
    ctx.fill()
    ctx.restore()


# ================================================================ SHOT D — the man in the fez
def cam_D(T):
    u = clamp((T - T_D) / (T_E - T_D))
    return Camera(x=lerp(930, 970, u), y=lerp(560, 530, u), zoom=lerp(1.0, 1.09, ease_in_out_sine(u)))


FEZ = (760.0, 1700.0, 1300.0)


def shot_fez(fc, st, T):
    """page 25 panel 2: a man in a fez preaches it through a megaphone to a vast engraved multitude"""
    cam = cam_D(T)
    beats = st["beats"]
    world = Canvas(fc, bg=G(0.6))
    ctx = world.ctx
    ctx.save()
    cam.apply(ctx)
    ctx.rectangle(-300, -300, W + 600, 720)
    ctx.set_source(rad_grad(1150, 390, 0, 1400, [(0, G(0.98)), (0.25, G(0.72)), (1, G(0.26))]))
    ctx.fill()
    tw.glory(ctx, 1150, 390, 60, 1300, T, n=44, strength=0.75)
    _draw_multitude(ctx, st["multitude"], T, beats)
    _curtains(ctx, T)
    fez = st["fez"]
    env = beat_env(T, beats, 0.25)
    fx, fy, fh = FEZ
    raise_ = 0.5 + 0.5 * env
    person = dict(x=fx, y=fy, h=fh, pose={"stand": 1.0, "megaphone": 1.0}, facing=1, turn=1.75, render="ink",
                  mouth=0.6 * env, lean=-0.04 - 0.06 * env,
                  hand_l=(fx - 330 - 30 * raise_, fy - fh * (0.92 + 0.06 * raise_)), shape_l="fist")
    anc = rimmed(ctx, person, T, (1150, 390), tone=0.03, rim=0.98, rim_w=9.0, ch=fez)
    ctx.restore()
    rain = rain_mask(fc, T, density=0.8, seed=7)
    img, _ = finish(fc, world.rgb(), None, None, cam, rain)
    nb = beats_before(T, beats)
    b0 = beats_before(T_D, beats)
    meg = cam.to_screen(*anc.get("prop", anc["hand_r"]))

    def lettering(c):
        k = nb - b0
        top_n = min(20, 4 + 4 * k)
        bot_n = min(15, 3 * k)
        comic.say(c, fc, "M02", 330, 150, w=560, kind="offpanel", tail=(-60, 120), flagify=flagify_u(T, T_FLOOD, 91.9),
                  size=34)
        comic.balloon(c, " ".join([ty.WORD] * top_n), 1330, 190, 880, tail=(meg[0] + 20, meg[1] - 10),
                      kind="balloon", size=40)
        if bot_n:
            comic.balloon(c, " ".join([ty.WORD] * bot_n), 1380, 860, 820, tail=(1180, 610), kind="balloon",
                          size=38)
    return letter(fc, img, lettering, None)


def _draw_multitude(ctx, heads, T, beats):
    """a vast engraved multitude: thousands of heads receding to the horizon, bobbing to the chant"""
    env = beat_env(T, beats, 0.3)
    ctx.save()
    for (x, y, s, r1, r2) in heads:
        bob = env * s * 0.25 * (0.5 + r1)
        yy = y - bob
        tone = 0.15 + 0.5 * r2
        ellipse(ctx, x, yy, s * 0.42, s * 0.5)
        set_color(ctx, G(tone))
        ctx.fill()
        if s > 8:
            ellipse(ctx, x - s * 0.12, yy - s * 0.15, s * 0.14, s * 0.12)
            set_color(ctx, G(min(1.0, tone + 0.4)))
            ctx.fill()
            ctx.rectangle(x - s * 0.55, yy + s * 0.4, s * 1.1, s * 0.9)
            set_color(ctx, G(tone * 0.6))
            ctx.fill()
    ctx.restore()


def _curtains(ctx, T):
    """heavy theatre drapery framing the top corners (Doré swags)"""
    ctx.save()
    for side in (-1, 1):
        x0 = 0 if side < 0 else W
        for f in range(9):
            u = f / 8
            xa = x0 - side * (40 + 260 * u)
            ctx.move_to(x0 - side * 0, -40)
            ctx.line_to(xa, -40)
            ctx.curve_to(xa - side * 30, 200 + 60 * u, xa - side * 20, 380 - 100 * u, x0 - side * 0, 520 - 200 * u)
            ctx.close_path()
            ctx.set_source(lin_grad(xa - 30, 0, xa + 30, 0, [(0, G(0.12)), (0.5, G(0.45 if f % 2 else 0.25)),
                                                           (1, G(0.10))]))
            ctx.fill()
    ctx.restore()


# ================================================================ SHOT E — the flood of FLAGS
def cam_E(T):
    return Camera(x=960, y=540, zoom=1.0 + 0.05 * ease_in_out_sine(clamp((T - T_E) / (T_PEAK - T_E))))


def shot_flood(fc, st, T):
    cam = cam_E(min(T, T_PEAK))
    beats = st["beats"]
    fx, fy, fh = FM_E
    tr = st["tracks"]["E"]
    i = (T - T_E) * fc.fps
    peak = T >= T_PEAK
    kw = fm_E_kw(T)
    m = mouth(fc, "FLAGMAKER")
    world = Canvas(fc, bg=(1, 1, 1))
    ctx = world.ctx
    front = Canvas(fc)
    if T < T_PEAK + 0.3:
        sky = tw.storm_sky(fc, st["billA"], (760, 250), T - T_A, noise=st["noise"], dark=0.25, far=0.45, rim=0.5)
        world.paint_array(np.repeat(sky[..., None], 3, axis=2))
        ctx.save()
        cam.apply(ctx)
        tw.glory(ctx, fx + 10, fy - fh * 0.9, 110, 1300, T, n=48, strength=0.9)
        anc = st["fm"].draw(ctx, fx, fy, fh, fm_E_pose(T), T, mouth=m, facing=1, turn=0.3, expr="serene",
                            render="ink", layer="back", **kw)
        ctx.restore()
        fctx = front.ctx
        fctx.save()
        cam.apply(fctx)
        st["fm"].draw(fctx, fx, fy, fh, fm_E_pose(T), T, mouth=m, facing=1, turn=0.3, expr="serene",
                      render="ink", layer="front", **kw)
        fctx.restore()
    else:
        anc = st["fm"].anchors(fx, fy, fh, fm_E_pose(T), T, facing=1, turn=0.3, render="ink", **kw)
    flags = Canvas(fc)
    c2 = flags.ctx
    c2.save()
    cam.apply(c2)
    tr.draw(c2, i, light=(-0.25, -0.8, 0.55), glow=0.25 if peak else 0.12)
    c2.restore()
    rain = rain_mask(fc, T, density=1.0 * (1 - smoothstep(93.2, 94.6, T)), seed=8) if T < T_PEAK else None
    img, fl = finish(fc, world.rgb(), flags, front if T < T_PEAK + 0.3 else None, cam, rain)
    ms = cam.to_screen(*anc["mouth"])
    cb = cloth_bbox(tr, i, cam.to_screen)
    block = ty.flag_block_mask(fl[..., 3], fc.s, 18.0)

    def lettering(c):
        if T < T_PEAK:
            comic.say(c, fc, "M03", 470, 190, w=700, tail=(ms[0] - 10, ms[1] - 10), avoid=[cb],
                      flagify=flagify_u(T, 92.5, 95.5), hold=0.0)
            _edge_balloons(c, fc, T, beats)
        _flood_words(c, fc, st, T, block)
    top = Canvas(fc)
    lettering(top.ctx)
    rgba = ty.mask_out(top.rgba(), block)
    return over(img, rgba)


EDGE_BAL = [(-1, 0.30), (1, 0.62), (0, 0.12), (-1, 0.72), (1, 0.18), (0, 0.9), (-1, 0.5), (1, 0.85), (0, 0.55),
            (-1, 0.1), (1, 0.4), (0, 0.3)]


def _edge_balloons(c, fc, T, beats):
    """balloons popping in from off-screen speakers on every beat of the chant (tails from the frame edges)"""
    b0 = beats_before(T_E, beats)
    nb = beats_before(T, beats) - b0
    for k in range(min(nb, len(EDGE_BAL))):
        side, v = EDGE_BAL[k]
        tb = beats[b0 + k]
        age = T - tb
        pop = ease_out_back(clamp(age / 0.15), 2.0)
        n = 2 + min(8, int(age / 0.25))
        if side == -1:
            bx, by, tail = 260, 160 + v * 760, (-60, 160 + v * 760 + 60)
        elif side == 1:
            bx, by, tail = W - 260, 160 + v * 760, (W + 60, 160 + v * 760 + 60)
        else:
            bx, by, tail = 200 + v * 1500, 960, (200 + v * 1500 + 50, H + 80)
        sz = 30 * pop
        if sz > 3:
            comic.balloon(c, " ".join([ty.WORD] * n), bx, by, 420, tail=tail, kind="balloon", size=sz)


def _setup_flood(S, st):
    """spawn the falling/spilling FLAGS words and simulate the pile around the flag cloth"""
    beats = st["beats"]
    rng = np.random.default_rng(95)
    f0 = int(round((T_E - S.start) * S.fps))
    n = int(round((T_PEAK - T_E) * S.fps)) + 2
    spawns = []
    # word rain: the downpour turns into FLAGS, accelerating to the peak
    for k in range(760):
        u = rng.random() ** 0.55
        t = 0.6 * S.fps + u * (n - 0.6 * S.fps - 6)
        size = rng.uniform(24, 62) * (0.8 + 0.4 * u)
        x = rng.uniform(20, W - 20)
        spawns.append(dict(t=float(t), x=float(x), y=float(-60 - rng.uniform(0, 120)), vx=float(180 + rng.normal(0, 40)),
                           vy=float(900 + rng.uniform(0, 500)), a=float(rng.normal(0, 0.25)),
                           av=float(rng.normal(0, 1.5)), size=float(size)))
    tr = st["tracks"]["E"]

    def obstacles(i):
        cam = cam_E(T_E + i / S.fps)
        V, _ = tr._frame(i)
        pts = V[::3, ::3, :2].reshape(-1, 2)
        r = 0.085 * tr.meta["pole"] * cam.zoom
        out = [(*map(float, cam.to_screen(x, y)), float(r)) for x, y in pts]
        # the pole (a chain of circles)
        bx, by, ang = tr.poles[min(int(i), tr.n - 1)]
        L = tr.meta["pole"]
        for q in np.linspace(0.0, 1.0, 12):
            px, py = cam.to_screen(bx + math.sin(ang) * L * q, by - math.cos(ang) * L * q)
            out.append((float(px), float(py), 16.0))
        return out

    key = dict(v=4, n=n, m=len(spawns), s=spawns[:3], fm=FM_E, pole=float(tr.meta["pole"]), pe=POLE_E,
               v0=float(tr.verts[0, 0, 0, 0]), v1=float(tr.verts[-1, -1, -1, 1]))
    st["pile"] = ty.simulate_pile(spawns, n, S.fps, obstacles, cache=S.cache, key=key)
    st["pile_f0"] = f0


WALL_SIZE = 46.0
SNAP_DUR = 0.32


def _snap_assignment(st, slots):
    """pile word -> wall slot (greedy nearest, computed once at the peak so the crystallisation is local)"""
    a = st.get("_snap")
    if a is not None:
        return a
    P = st["pile"]
    pos = P["pos"][-1]
    alive = np.nonzero(~np.isnan(pos[:, 0]))[0]
    src = pos[alive, :2].astype(np.float64)
    used = np.zeros(len(alive), bool)
    pairs = []
    order = sorted(range(len(slots)), key=lambda j: (slots[j][1], slots[j][0]))
    for j in order:
        sx, sy = slots[j][0], slots[j][1]
        d = (src[:, 0] - sx) ** 2 * 0.35 + (src[:, 1] - sy) ** 2
        d[used] = np.inf
        k = int(np.argmin(d))
        if np.isfinite(d[k]):
            used[k] = True
            pairs.append((j, int(alive[k])))
    st["_snap"] = (pairs, [int(alive[k]) for k in np.nonzero(~used)[0]])
    return st["_snap"]


def _flood_words(c, fc, st, T, block):
    """the pile of FLAGS words; at the peak they crystallise into a typeset wall that REFLOWS live around the
    flag (justified text-wrap around the moving cloth); the wall pulses on the chant, then scrolls away"""
    P = st["pile"]
    i = int(round((min(T, T_PEAK) - T_E) * fc.fps))
    i = max(0, min(P["pos"].shape[0] - 1, i))
    pos = P["pos"][i]
    size = P["size"]
    alive = ~np.isnan(pos[:, 0])
    if T < T_PEAK:
        words = [(pos[k, 0], pos[k, 1], pos[k, 2], size[k], 1.0, ty.WORD) for k in np.nonzero(alive)[0]]
        ty.draw_words(c, words)
        return
    # ---- the wall
    u = T - T_PEAK
    off = scroll_offset(T)
    beats = st["beats"]
    pulse = beat_env(T, beats, 0.12) if T < fc.tl.d["chant"]["end"] else 0.0
    blk = block
    if off > 0.5:
        d = int(round(off * fc.s))
        blk = np.zeros_like(block)
        blk[d:] = block[:block.shape[0] - d]
    c.save()
    c.translate(0, -off)
    if u < 0.3:
        set_color(c, ink.PAPER, clamp(u / 0.18))
        c.rectangle(-10, -10, W + 20, H + 20)
        c.fill()
    snap = clamp(u / SNAP_DUR)
    e = ease_out_back(snap, 1.3)
    if snap < 1:
        slots0, _ = ty.wall_slots((0, 0, W, H), WALL_SIZE, _peak_block(st, fc, block), fc.s)
        pairs, spare = _snap_assignment(st, slots0)
        words = []
        for j, k in pairs:
            sx, sy = slots0[j][0], slots0[j][1]
            px, py, pa = pos[k]
            words.append((lerp(px, sx, e), lerp(py, sy, e), pa * (1 - e), lerp(size[k], WALL_SIZE, e), 1.0, ty.WORD))
        for k in spare:                                   # the surplus is squeezed out
            px, py, pa = pos[k]
            words.append((px, py, pa, size[k] * (1 - snap), 1.0 - snap, ty.WORD))
        ty.draw_words(c, words, keyline=0.10 * (1 - snap))
    else:
        slots, _ = ty.wall_slots((0, 0, W, H), WALL_SIZE, blk, fc.s)
        sz = WALL_SIZE * (1.0 + 0.06 * pulse)
        ty.draw_words(c, [(x, y, 0.0, sz, 1.0, ty.WORD) for x, y, r, cc in slots], keyline=0.0)
    c.restore()


def _peak_block(st, fc, block):
    """the flag's block mask exactly at the peak frame (deterministic in every worker)"""
    b = st.get("_peak_block")
    if b is None or b.shape != block.shape:
        fl = Canvas(fc)
        fl.ctx.save()
        cam_E(T_PEAK).apply(fl.ctx)
        st["tracks"]["E"].draw(fl.ctx, (T_PEAK - T_E) * fc.fps)
        fl.ctx.restore()
        b = ty.flag_block_mask(fl.rgba()[..., 3], fc.s, 18.0)
        st["_peak_block"] = b
    return b


# ================================================================ SHOT F — the Devil
def shot_devil(fc, st, T):
    return shot_devil_impl(fc, st, T)


def shot_devil_impl(fc, st, T):
    from scenes import t04_devil
    return t04_devil.render(fc, st, T)


# ================================================================ sound exports
def _export(S, st):
    ev = []
    ch = st["fm"]
    for (t, which) in ch.footsteps(0.0, 2.7, "walk", 1.0):
        T = S.start + t
        X, Z, _ = fm_walk_state(T)
        x, _, k = proj(X, Z)
        ev.append(dict(t=T, kind="step", strength=0.35 + 0.3 * k, pan=foley.screen_pan(x), desc="Flag Maker barefoot in mud"))
    ev.append(dict(t=T_PLANT, kind="thunk", strength=1.0, pan=-0.1, desc="the Flag Maker slams his Flag pole into the mud on FLAGS (mud splash)"))
    ev.append(dict(t=T_A2, kind="cut", strength=0.3, pan=0.0, desc="cut to medium-wide on the Flag Maker"))
    ev.append(dict(t=76.98, kind="impact", strength=1.0, pan=0.2, desc="SHUT UP! the Pharisees lurch forward, camera jolt"))
    ev.append(dict(t=77.0, kind="thunder", strength=0.7, pan=0.3, desc="lightning flash behind the Pharisees"))
    ws = S.tl.words("P12")
    ev.append(dict(t=ws[-3]["s"] - 0.05, kind="whoosh", strength=0.5, pan=0.6, desc="FLAGS 4:20 citation box slides in"))
    beats = st["beats"]
    for k, b in enumerate(beats):
        if b < T_D:
            ev.append(dict(t=b, kind="balloon", strength=0.5, pan=0.3, desc="crowd balloon pops: FLAGS FLAGS"))
        elif b < T_E:
            ev.append(dict(t=b, kind="balloon", strength=0.6, pan=0.0, desc="fez preacher / multitude balloon row"))
        elif b < T_PEAK:
            ev.append(dict(t=b, kind="balloon", strength=0.7, pan=(-1) ** k * 0.7, desc="edge balloon pops"))
    ev.append(dict(t=T_E + 0.6, kind="type_rain", strength=0.6, pan=0.0,
                   desc="words start raining (FLAGS blocks falling and piling up until 95.72)"))
    ev.append(dict(t=T_PEAK, kind="snap", strength=1.0, pan=0.0, desc="FLOOD PEAK: the word pile snaps into a wall"))
    ev.append(dict(t=T_SCROLL0, kind="whoosh", strength=0.8, pan=0.0, desc="page scroll to the Devil"))
    try:
        from scenes import t04_devil
        ev += t04_devil.events(S, st)
    except Exception:
        pass
    foley.export_events(S, "hits", ev)
    tA = st["tracks"]["A"]
    tA.export_foley(S, "flag_walk", f0=0, pan=0.0)
    st["tracks"]["B"].export_foley(S, "flag_rejected", f0=int(round((T_B - S.start) * S.fps)), pan=-0.4)
    st["tracks"]["C"].export_foley(S, "flag_explain", f0=int(round((T_C - S.start) * S.fps)), pan=-0.65)
    st["tracks"]["E"].export_foley(S, "flag_flood", f0=int(round((T_E - S.start) * S.fps)))

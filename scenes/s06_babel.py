"""s06 — BABEL (125.5–139.0).  Owner: Scene06.

Signature: a vertical crane up a procedurally generated, genuinely 3D Tower of Babel assembled — block
by block, on camera — from every fragment of the shattered memeplexes, babbling in ~40 real writing
systems, crowned by LORD EGREGORE (a hooded mass of static screens).

Shots (global s):
  A 125.50–130.00  s05b's galaxy swirl (top-down, CCW about (960,620), lavender core) lifts into a helix:
                   every shard/card flies a helix and is squared into a brick of the tower; the camera
                   pitches from straight down to level and cranes up (orbiting fragments, cloud decks) as
                   the tiers build and their crowds pop up shouting in ~40 scripts; Lord Egregore lowers
                   out of the storm, lightning strikes his antenna (128.98) and his screens power on.
  B 130.00–133.60  rubble-ledge two-shot in a gale: Lutie (L03) shouts clutching her hood, then throws her
                   arms wide; Crocus calm beside his planted flag-staff.
  C 133.60–136.00  Crocus MCU, lightning on "Good" (133.78, 134.12), nod + tiny smile; the Flag steady, glowing.
  D 136.00–138.50  babel_crack: cracks race up, Egregore glitches, blocks tear loose, the top topples, shake;
                   Lutie & Crocus silhouettes, the Flag the only warm light (no chromatic split across it).
  E 138.50–139.00  `silence`: pure black.
Helpers: s06_tower (3D camera + tower model), s06_draw (tower renderer), s06_egregore, s06_sky (storm,
lightning, cloud decks), s06_glyphs (PIL+raqm glyph atlas), s06_fg (ledge, ruins, gale).
"""
import math
from pathlib import Path

import cairo
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from vx import *
from vx import foley
from vx.flag import Flag
from vx.chars import Character
import vx.chars as _chars

from . import s06_glyphs as G
from . import s06_sky as SKY
from .s06_tower import Tower, Cam3, Orbiters, smooth_arr
from .s06_draw import draw_tower, Light
from .s06_egregore import Egregore
from . import s06_fg as FG

POST = dict(bloom=0.55, bloom_thresh=0.68, bloom_radius=24.0, grain=0.035, vignette=0.34, contrast=1.04)

T0 = 125.5
TB, TC, TD, TE = 130.0, 133.6, 136.0, 138.5
CRACK = 136.0

# lightning strikes: (global t, strength/closeness, shot-specific target key, seed)
STRIKES = [
    (126.85, 0.45, "far_l", 11),
    (128.15, 0.6, "far_r", 12),
    (128.98, 1.0, "antenna", 13),
    (130.02, 0.75, "bg_l", 14),
    (131.55, 0.5, "bg_r", 15),
    (133.78, 1.0, "crocus", 16),
    (134.12, 0.75, "crocus", 24),
    (135.05, 0.45, "bg_r", 17),
    (136.0, 1.0, "antenna", 18),
    (136.55, 0.8, "bg_l", 19),
    (137.12, 0.9, "antenna2", 20),
    (137.62, 0.85, "bg_r", 21),
    (138.05, 1.0, "antenna", 22),
    (138.33, 0.9, "bg_l", 23),
]


# ============================================================================================ camera A
_KA = np.array([
    # t,   tgtY,  dist,  pitch(deg), yaw,  cy
    (0.0, -300, 2900, -90.0, 0.00, 620),
    (0.45, -285, 2870, -86.0, 0.02, 616),
    (1.2, -40, 2700, -40.0, 0.15, 580),
    (1.8, 350, 2600, -8.0, 0.30, 545),
    (2.5, 1150, 2450, 0.0, 0.50, 540),
    (3.3, 2350, 2380, 4.0, 0.80, 540),
    (3.85, 3050, 2900, 10.0, 0.98, 540),
    (4.3, 4450, 3850, 20.0, 1.08, 540),
    (4.5, 4700, 4000, 22.0, 1.10, 540),
])
_PA = PchipInterpolator(_KA[:, 0], _KA[:, 1:], axis=0)
FA = 950.0


def cam_A(t, tw=None):
    tt = min(max(t, 0.0), 4.5)
    ty, dist, pitch, yaw, cy = _PA(tt)
    ax, az = (0.0, 0.0) if tw is None else tw.axis_y(min(ty, tw.top))
    k = smoothstep(0.6, 2.0, tt)
    return Cam3.orbit((float(ax) * k, ty, float(az) * k), dist, yaw, math.radians(pitch), f=FA, cy=cy)


# ============================================================================================ far cameras
def cam_B(T):
    u = seg(T, TB, TC, ease_in_out_sine)
    return Cam3.orbit((380.0, 2900.0 + 80 * u, 0.0), 10600 - 500 * u, 0.06 + 0.02 * u, math.radians(11.0 + 0.6 * u),
                      f=1250.0, cx=940, cy=500)


def cam_C(T):
    u = seg(T, TC, TD, ease_in_out_sine)
    return Cam3.orbit((420.0, 3100.0, 0.0), 11200 - 500 * u, -0.10 - 0.02 * u, math.radians(14.5 + 0.8 * u),
                      f=1250.0, cx=500, cy=520)


def cam_D(T):
    u = seg(T, TD, TE, ease_in_out)
    return Cam3.orbit((380.0, 2750.0 + 150 * u, 0.0), 9400 - 900 * u, 0.14, math.radians(13.0 + 1.5 * u),
                      f=1250.0, cx=1010, cy=520)


# ============================================================================================ setup
def setup(S):
    tw = Tower(seed=6)
    # tier build windows follow the crane: each tier finishes just as the camera reaches it
    ts = np.linspace(0, 4.5, 2000)
    tys = _PA(ts)[:, 0]
    win = []
    prev = 0.0
    for k in range(tw.NT):
        if k == 0:
            a, b = 0.45, 1.3
        elif k == 1:
            a, b = 0.7, 1.55
        elif k == 2:
            a, b = 0.95, 1.85
        else:
            tk = float(np.interp(tw.ym[k] - 250, tys, ts))
            a = max(prev + 0.18, tk - 0.75)
            b = a + 0.62
        win.append((a, b))
        prev = a
    tw.schedule(win)
    st = dict(tw=tw, eg=Egregore(), orb=Orbiters(), atlas=G.get(S), tex=SKY.make_textures(), win=win,
              detach=crack_detach(tw),
              lutie=Character("lutie", seed=1, hood="up"), crocus=Character("crocus", seed=2),
              cracks=make_cracks(tw))
    st["flag"] = sim_flag(S, st)
    export_sound(S, st)
    return st


def crack_detach(tw):
    """global time at which each block is shaken loose (1e9 = holds)"""
    rng = np.random.default_rng(99)
    front_t = CRACK + 0.95 * ((tw.b_ym - tw.y0[0]) / tw.total) ** 1.4
    td = front_t + 0.12 + rng.exponential(0.8, tw.nb)
    td -= 0.45 * (tw.b_tier / (tw.NT - 1))       # the screens & slop up top go first
    return np.where(rng.random(tw.nb) < 0.6, td, 1e9)


def make_cracks(tw):
    """crack polylines in (theta, y) tower-surface space"""
    rng = np.random.default_rng(7)
    cracks = []
    for j in range(18):
        th = rng.uniform(-math.pi, math.pi)
        y = tw.y0[0] + rng.uniform(0, 500)
        pts = [(th, y)]
        while y < tw.top - 20:
            y += rng.uniform(40, 110)
            th += rng.normal(0, 0.05)
            pts.append((th, y))
            if rng.random() < 0.1:
                b = [(th, y)]
                bth, by = th, y
                for _ in range(int(rng.integers(3, 9))):
                    by += rng.uniform(30, 90)
                    bth += rng.normal(0, 0.09) + rng.choice([-1, 1]) * 0.03
                    b.append((bth, by))
                cracks.append((np.array(b), 0.5))
        cracks.append((np.array(pts), 1.0))
    return cracks


def topple(T):
    """extra lean of the upper tower as Babel starts to fall (rad) and pivot height"""
    return 0.2 * seg(T, 137.2, 138.5, ease_in), 1300.0


def make_pre(T):
    a, py = topple(T)
    if a <= 0:
        return None
    px = 300.0
    ca, sa = math.cos(a), math.sin(a)

    def pre(P):
        P = np.array(P, np.float64, copy=True)
        dx, dy = P[..., 0] - px, P[..., 1] - py
        w = np.clip(dy / 600.0, 0, 1)            # soft hinge
        x2 = px + dx * ca + dy * sa
        y2 = py - dx * sa + dy * ca
        P[..., 0] = P[..., 0] * (1 - w) + x2 * w
        P[..., 1] = P[..., 1] * (1 - w) + y2 * w
        return P
    return pre


# ============================================================================================ the Flag
CANON_H = 600.0
CANON_POLE = 840.0          # 1.4 x Crocus's height (96" furring pole)
POLE_DX = 0.2 * CANON_H     # planted beside him like a staff, on his front side


def _pole_ang(T):
    return 0.03 + 0.006 * math.sin(T * 0.8)


def sim_flag(S, st):
    import hashlib
    from vx.flag import FlagTrack
    import vx.flag as _vf
    vxd = Path(_vf.__file__).parent
    src = b"".join(open(f, "rb").read() for f in sorted(vxd.glob("flag*.py")))
    key = hashlib.md5(src + repr((TB, CANON_POLE, POLE_DX, "staff2")).encode()).hexdigest()[:10]
    p = S.cache / f"crocus_flag_{key}.npz"
    f0 = int(round((TB - T0) * S.fps))
    n = S.n - f0
    try:
        tr = FlagTrack.load(p)
        if tr.n == n:
            return tr
    except Exception:
        pass

    def pole_fn(i):
        return (0.0, 0.0, _pole_ang(TB + i / S.fps))

    def wind_fn(i):
        T = TB + i / S.fps
        return (840.0 * (1.0 + 0.06 * math.sin(T * 1.3)), -30.0)

    tr = Flag(pole=CANON_POLE, seed=21, side=1, turbulence=0.5, flutter=0.85).simulate(n, pole_fn, wind_fn)
    tr.save(p)
    return tr


# ============================================================================================ sound export
def export_sound(S, st):
    tw = st["tw"]
    ev = []
    for k in range(tw.NT):
        m = tw.b_tier == k
        tas = np.sort(tw.ta[m])
        for q in range(6):
            tq = tas[int(q / 6 * (len(tas) - 1))]
            ev.append(dict(t=T0 + tq, kind="stack", strength=0.55 - 0.03 * k, pan=math.sin(q * 1.7) * 0.5,
                           desc=f"tier {k} fragments slam into place (stone/marble/phones)"))
        ev.append(dict(t=T0 + tw.tier_done[k], kind="thunk", strength=0.6 - 0.03 * k, pan=0.0,
                       desc=f"tier {k} complete; its crowd pops up shouting"))
    ev.append(dict(t=T0 + 0.05, kind="whoosh", strength=0.8, pan=0.0, desc="galaxy swirl lifts into a helix"))
    ev.append(dict(t=T0 + 2.2, kind="whoosh", strength=0.9, pan=0.0, desc="crane rushes up the tower"))
    ev.append(dict(t=T0 + 2.4, kind="descend", strength=0.9, pan=0.0,
                   desc="Lord Egregore lowers out of the storm onto the tower top (sub-bass swell 127.9-129.0)"))
    ev.append(dict(t=T0 + 3.5, kind="power_on", strength=1.0, pan=0.0,
                   desc="lightning wakes Egregore: screens power on (CRT thump cascade 129.0-129.85, hem to hood)"))
    for (t, k, key, sd) in STRIKES:
        pan = -0.5 if key.endswith("_l") else (0.5 if key.endswith("_r") else 0.0)
        if key == "crocus":
            pan = -0.4
        ev.append(dict(t=t, kind="lightning", strength=k, pan=pan, desc=f"lightning strike ({key}); thunder"))
    for q in range(12):
        tq = CRACK + 0.95 * (q / 11) ** 1.4
        ev.append(dict(t=tq, kind="crack", strength=1.0 - 0.03 * q, pan=math.sin(q * 2.1) * 0.4,
                       desc="crack races up the tower"))
    for tq in (136.1, 136.45, 136.9, 137.3, 137.75, 138.1, 138.3):
        ev.append(dict(t=tq, kind="glitch", strength=0.8, pan=0.1, desc="Egregore's screens glitch burst"))
    for tq in (136.7, 137.05, 137.4, 137.7, 137.95, 138.15, 138.3, 138.42):
        ev.append(dict(t=tq, kind="collapse", strength=0.9, pan=math.sin(tq * 5) * 0.5,
                       desc="blocks tear loose and fall / debris impacts"))
    ev.append(dict(t=137.2, kind="groan", strength=1.0, pan=0.2, desc="the upper tower starts to topple"))
    ev.append(dict(t=TE, kind="cut", strength=1.0, pan=0.0, desc="HARD CUT TO BLACK + SILENCE"))
    foley.export_events(S, "hits", ev)
    f0 = int(round((TB - T0) * S.fps))
    pan = np.zeros(S.n, np.float32)
    near = np.zeros(S.n, np.float32)        # closeness to camera per shot (two-shot / MCU / far silhouette)
    for f in range(S.n):
        T = T0 + f / S.fps
        pan[f] = 0.45 if T < TC else (0.3 if T < TD else -0.62)
        near[f] = 0.0 if T < TB else (0.7 if T < TC else (1.0 if T < TD else (0.35 if T < TE else 0.0)))
    tr = st["flag"]
    tr.export_foley(S, "crocus_flag", pan=pan, f0=f0, gain=0.8)
    # fold the per-shot closeness into the energy so the flutter sits right in each shot (silent after the cut)
    p = CACHE / "foley" / f"{S.id}_crocus_flag.npz"
    z = dict(np.load(p))
    ev = [e for e in z["events"].tolist() if e[0] < TE]
    for e in ev:
        e[1] *= float(near[min(S.n - 1, int(round((e[0] - T0) * S.fps)))])
    foley.export_track(S, "crocus_flag", z["energy"] * near, pan, gain=0.8, kind="flag", events=ev,
                       flap_hz=z["flap_hz"], snap=z["snap"] * near, closeness=near)


# ============================================================================================ render
def post(fc, st):
    T = fc.T
    if T >= TE - 1e-6:
        return dict(fade=1.0, grain=0.0, bloom=0.0, vignette=0.0, ca=0.0)
    d = {}
    if T >= TD:
        u = seg(T, TD, TE, ease_in)
        d["contrast"] = 1.06 + 0.1 * u     # (chromatic split is applied in-scene, never across the Flag)
    return d


def rgb_split(img, px, fc):
    """radial chromatic aberration on the world layers only (the Flag is composited afterwards)"""
    if px <= 0.2:
        return img
    h, w = img.shape[:2]
    a = px * fc.s / (w / 2)
    out = img.copy()
    for ch, k in ((0, 1 + a), (2, 1 - a)):
        M = np.array([[k, 0, w / 2 * (1 - k)], [0, k, h / 2 * (1 - k)]], np.float32)
        out[..., ch] = cv2.warpAffine(np.ascontiguousarray(img[..., ch]), M, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
    return out


def render(fc, st):
    T = fc.T
    if T >= TE - 1e-6:
        return np.zeros((fc.h, fc.w, 3), np.float32)
    if T < TB:
        return shot_A(fc, st)
    return shot_far(fc, st, "B" if T < TC else ("C" if T < TD else "D"))


def flash_at(T, lo=None, hi=None):
    return SKY.flash_env(T, [s for s in STRIKES if (lo is None or s[0] >= lo) and (hi is None or s[0] < hi)])


def _compose(fc, sky, main, glow, glow_k=1.0):
    img = over(sky, main.rgba())
    g = glow.rgba()[..., :3]
    s = fc.s
    return img + blur(g, 5 * s) * 0.8 * glow_k + blur(g, 26 * s) * 0.9 * glow_k


# -------------------------------------------------------------------------------------------- shot A
def shot_A(fc, st):
    tw, eg = st["tw"], st["eg"]
    t = fc.T - T0
    cam = cam_A(t, tw)
    sx, sy = shake(t, 3.0, 0.8, 5)
    cam.cx += sx
    cam.cy += sy
    fl = flash_at(fc.T, None, TB)
    eg_on = seg(t, 3.5, 4.35, ease_in_out)
    topc = np.array([tw.ax[-1], tw.top, tw.az[-1]])
    def _behind(ts, side):
        """a strike placed behind the tower as seen by the crane at strike time (always in frame)"""
        c = cam_A(ts - T0, tw)
        p = c.pos + c.fw * 14000 + c.rt * side * 5200
        return np.array([p[0], min(p[1], 2500.0) - 1500.0, p[2]])
    strike_pos = {"far_l": _behind(126.85, -1), "far_r": _behind(128.15, 1),
                  "antenna": topc + np.array([200, 2600, 0])}
    fpos, fk = None, 0.0
    for (ts, k, key, sd) in STRIKES:
        if ts > fc.T or ts < T0 or ts >= TB:
            continue
        a = SKY.flash_env(fc.T, [(ts, k)])
        if a > fk:
            fk, fpos = a, strike_pos.get(key, strike_pos["far_l"])
    sky = SKY.sky(fc, cam, t, st["tex"], (topc[0], topc[2]), (0.0, 0.0), flash=fl,
                  flash_dir=None if fpos is None else fpos - cam.pos, eg_glow=eg_on)
    # s05b's swirl is lit from its lavender core: bright shards while we still look straight down
    boost = 1.0 - seg(t, 0.5, 1.7)
    light = Light(fog_col=(0.10, 0.09, 0.19), eg_pos=topc + np.array([0, 1500, 0]), eg_k=0.25 + 0.9 * eg_on,
                  flash_pos=fpos, flash_k=fk * 0.9 + 0.4 * boost, fog_near=2500 + 3000 * boost, fog_far=14000,
                  sea_fog=0.85 * (1 - boost))
    if boost > 0:
        light.flash_pos = np.array([0.0, 6000.0, 0.0]) if fpos is None or fk < 0.05 else fpos
    main, glow, bolts = Canvas(fc), Canvas(fc), Canvas(fc)
    ctx, gctx = main.ctx, glow.ctx
    for (ts, k, key, sd) in STRIKES:
        a = SKY.bolt_alpha(fc.T, ts)
        if a <= 0 or ts >= TB:
            continue
        p1 = strike_pos.get(key)
        p0 = p1 + np.array([-900 + 500 * (sd % 3), 9000, 600])
        x0, y0, z0 = cam.proj(p0[None])
        x1, y1, z1 = cam.proj(p1[None])
        if z0[0] < 50 or z1[0] < 50:
            continue
        SKY.draw_bolt(bolts.ctx, gctx, SKY.bolt(sd, x0[0], y0[0], x1[0], y1[0]), a * k, width=2.6)
    sky = over(sky, bolts.rgba())

    e_in = seg(t, 2.4, 3.5, ease_out)     # Lord Egregore lowers himself out of the storm onto the tower

    def eg_draw():
        if e_in <= 0:
            return
        x, y, z = cam.proj((topc + np.array([0.0, 2400.0 * (1 - e_in), 0.0]))[None])
        if z[0] < 100:
            return
        eg.draw(ctx, gctx, x[0], y[0], cam.f / z[0], fc.T, on=eg_on, flash=fk, fog=0.1 + 0.85 * (1 - e_in),
                fog_col=(0.1, 0.09, 0.19), rot=cam.roll)

    draw_tower(ctx, gctx, tw, cam, t, light, atlas=st["atlas"], eg_draw=eg_draw, orb=st["orb"])
    img = _compose(fc, sky, main, glow)
    # the crane punches up through thin cloud decks wrapped around the tower
    ck = smoothstep(1.6, 2.0, t) * (1 - smoothstep(3.3, 3.7, t))
    if ck > 0:
        img = over(img, SKY.cloud_layers(fc, cam, t, st["tex"], flash=fl, k=ck))
    rain = Canvas(fc)
    FG.weather(rain.ctx, None, fc.T, 0.55 * smoothstep(0.8, 2.0, t), None, bubbles=False)
    img = over(img, rain.rgba())
    # vertical motion blur while the crane rushes (90-degree shutter)
    c2 = cam_A(t + 1.0 / 24, tw)
    ref = np.array([[tw.ax[5], tw.ym[5], 0.0]])
    vy = abs(c2.proj(ref)[1][0] - cam.proj(ref)[1][0])
    L = int(min(14, vy * 0.25) * fc.s)
    if L >= 3:
        img = cv2.filter2D(img, -1, np.ones((L, 1), np.float32) / L, borderType=cv2.BORDER_REPLICATE)
    img = img * (1 + 0.18 * fl)
    return np.clip(img, 0, 1).astype(np.float32)


# -------------------------------------------------------------------------------------------- shots B/C/D
def shot_far(fc, st, which):
    tw, eg = st["tw"], st["eg"]
    T = fc.T
    t = T - T0
    cam = {"B": cam_B, "C": cam_C, "D": cam_D}[which](T)
    chaos = seg(T, TD, TE, ease_in) if which == "D" else 0.0
    if which == "D":
        amp = 3 + 22 * chaos + 14 * pulse(T, CRACK + 0.03, 0.12)
        dx, dy = shake(T, amp, 11.0, 3)
        cam = Cam3(cam.pos, cam.yaw, cam.pitch, 0.010 * chaos * math.sin(T * 23), cam.f, cam.cx + dx, cam.cy + dy)
        cam.pre = make_pre(T)
    else:
        dx, dy = shake(T, 2.0, 1.2, 4)
        cam.cx += dx
        cam.cy += dy
    topc = np.array([tw.ax[-1], tw.top, tw.az[-1]])
    x_top, y_top, z_top = cam.proj(topc[None])
    s_eg = cam.f / z_top[0]
    eg_rot = cam.roll + (topple(T)[0] if which == "D" else 0.0)
    tips = eg.antenna_tips()
    fl = flash_at(T, TB, None)
    fk, fpos = 0.0, None
    main, glow, bolts = Canvas(fc), Canvas(fc), Canvas(fc)
    ctx, gctx = main.ctx, glow.ctx
    for (ts, k, key, sd) in STRIKES:
        if ts < TB or ts > T + 1e-6:
            continue
        a = SKY.bolt_alpha(T, ts)
        e = SKY.flash_env(T, [(ts, k)])
        if key.startswith("antenna"):
            lx, ly = tips[0 if key == "antenna" else 4]
            ca, sa = math.cos(eg_rot), math.sin(eg_rot)
            x1 = x_top[0] + (lx * ca - ly * sa) * s_eg
            y1 = y_top[0] + (lx * sa + ly * ca) * s_eg
            x0, y0 = x1 - 160 + (sd % 5) * 70, -40
            wpos = topc + np.array([lx, -ly + 800, -600])
        elif key == "bg_l":
            x0, y0, x1, y1 = 180 + 50 * (sd % 4), -30, 300 + 40 * (sd % 3), 700
            wpos = cam.pos + cam.fw * 18000 - cam.rt * 14000 + np.array([0, 3000, 0])
        elif key == "bg_r":
            x0, y0, x1, y1 = 1660 - 50 * (sd % 4), -30, 1540, 660
            wpos = cam.pos + cam.fw * 18000 + cam.rt * 14000 + np.array([0, 3000, 0])
        else:  # "crocus": a huge bolt splits the sky behind the tower
            x0, y0, x1, y1 = x_top[0] + 260, -40, x_top[0] + 520, 760
            wpos = cam.pos + cam.fw * 2500 - cam.rt * 2500 + np.array([0, 2500, 0])
        if e > fk:
            fk, fpos = e, wpos
        if a > 0:
            SKY.draw_bolt(bolts.ctx, gctx, SKY.bolt(sd, x0, y0, x1, y1, depth=8), a * k, width=2.4)
    sky = SKY.sky(fc, cam, t, st["tex"], (topc[0], topc[2]), (0.0, 0.0), flash=fl,
                  flash_dir=None if fpos is None else fpos - cam.pos, eg_glow=1.0)
    sky = over(sky, bolts.rgba())
    # ruins of the old memeplexes standing in the fog sea (mid-ground)
    rz = Canvas(fc)
    hz = cam.cy + cam.f * math.tan(cam.pitch)
    if which == "B":
        FG.ruins(rz.ctx, T, fk, hz + 60, scale=0.9)
    elif which == "D":
        FG.ruins(rz.ctx, T, fk, hz + 70, scale=0.75, dx=260)
    sky = over(sky, rz.rgba())
    light = Light(fog_col=(0.11, 0.10, 0.20), eg_pos=topc + np.array([0, 1500, 0]), eg_k=1.1, flash_pos=fpos,
                  flash_k=fk * 0.9, fog_near=6000, fog_far=30000, sea_fog=0.9)
    glitch = 0.0
    collapse = None
    if which == "D":
        glitch = clamp(seg(T, CRACK + 0.03, CRACK + 0.35) * 0.45 + chaos * 0.65)
        collapse = dict(detach_t=st["detach"] - T0)

    def eg_draw():
        if which == "D":
            draw_cracks(ctx, gctx, tw, cam, st["cracks"], T)
        eg.draw(ctx, gctx, x_top[0], y_top[0], s_eg, T, on=1.0, glitch=glitch, flash=fk, fog=0.06,
                fog_col=(0.11, 0.10, 0.20), rot=eg_rot, lod=0.7)

    draw_tower(ctx, gctx, tw, cam, t, light, atlas=st["atlas"], collapse=collapse, glitch=glitch,
               eg_draw=eg_draw, lod=1.0, bub_min=2.6, bub_alpha=0.42, bub_frac=0.3)
    img = _compose(fc, sky, main, glow)
    if which == "D":
        img = rgb_split(img, 1.5 + 9.0 * chaos + 6.0 * pulse(T, CRACK + 0.05, 0.1), fc)
    fg, fglow = Canvas(fc), Canvas(fc)
    foreground(fc, st, which, fg.ctx, fglow.ctx, fk, chaos)
    img = over(img, fg.rgba())
    gg = fglow.rgba()[..., :3]
    img = img + blur(gg, 7 * fc.s) * 0.55 + blur(gg, 45 * fc.s) * 0.75
    img = img * (1 + 0.16 * fl)
    return np.clip(img, 0, 1).astype(np.float32)


def draw_cracks(ctx, gctx, tw, cam, cracks, T):
    if T < CRACK:
        return
    front = tw.y0[0] + tw.total * clamp((T - CRACK) / 0.95) ** (1 / 1.4) + 60
    for pts, wf in cracks:
        m = pts[:, 1] <= front
        if m.sum() < 2:
            continue
        p = pts[m]
        th, y = p[:, 0], p[:, 1]
        k = np.clip(np.searchsorted(tw.y1, y), 0, tw.NT - 1)
        r = tw.r[k] + 14
        P = np.stack([tw.ax[k] + r * np.cos(th), y, tw.az[k] + r * np.sin(th)], -1)
        nrm = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], -1)
        visf = (nrm * (cam.pos[None, :] - P)).sum(1) > 0
        x, yy, z = cam.proj(P)
        s = cam.f / np.maximum(z, 1)
        for i in range(len(p) - 1):
            if not (visf[i] and visf[i + 1]):
                continue
            age = clamp((T - CRACK) * 2.0 - (y[i] - tw.y0[0]) / tw.total * 0.5)
            w = max(0.7, 9 * s[i] * wf) * (0.5 + 0.7 * age)
            for cc, lw, colr, a in ((gctx, w * 3.2, (0.55, 0.4, 0.9), 0.6), (ctx, w * 1.6, (0.7, 0.6, 1.0), 0.85),
                                    (ctx, w * 0.6, (1.0, 1.0, 1.0), 1.0)):
                cc.set_line_width(lw)
                cc.set_source_rgba(*colr, a)
                cc.move_to(x[i], yy[i])
                cc.line_to(x[i + 1], yy[i + 1])
                cc.stroke()


# ============================================================================================ foreground
def P(pose):
    """pose dict filtered to the poses the current rig knows (robust across rig versions)"""
    if isinstance(pose, dict):
        d = {k: v for k, v in pose.items() if k in _chars.POSES}
        return d or "stand"
    return pose if pose in _chars.POSES else "stand"


STORM_TINT = (0.07, 0.07, 0.16)
RIM_COL = (0.66, 0.76, 1.0)


def _shout(m):
    """a shouted line keeps the jaw open between syllables"""
    return min(1.0, 0.45 + 0.6 * m)


def flag_bbox(st, T, fi, x, y, h, facing, pad=40.0):
    """screen bbox of the Flag cloth (nothing with glyphs may come near it)"""
    tr = st["flag"]
    k = h / CANON_H
    bx = x + facing * POLE_DX * k
    V = tr.verts[int(max(0, min(tr.n - 1, round(fi))))][..., :2]
    xs, ys = bx + V[..., 0] * k, y + V[..., 1] * k
    return (float(xs.min()) - pad, float(ys.min()) - pad, float(xs.max()) + pad, float(ys.max()) + pad)


def _crocus_with_flag(ctx, gctx, st, T, fi, x, y, h, facing, mouth, expr, look, tint, rim, silhouette=None,
                      glow_k=1.0, light=(-0.75, -0.45, 0.3), nod=0.0):
    """Crocus with his planted flag-staff: cloth behind him, pole in his front hand."""
    cro, tr = st["crocus"], st["flag"]
    k = h / CANON_H
    ang = _pole_ang(T)
    bx = x + facing * POLE_DX * k
    grip = (bx + math.sin(ang) * 0.6 * h, y - 0.6 * h)
    kw = dict(mouth=mouth, expr=expr, facing=facing, look=look, wind=1.4, hand_r=grip, shape_r="grip", nod=nod)
    if tint is not None:
        kw["tint"] = tint
    if rim is not None:
        kw["rim"] = rim
    if silhouette is not None:
        kw["silhouette"] = silhouette
    fi = max(0.0, min(tr.n - 1.0, fi))
    # cloth (+ its warm halo) behind him
    for c, a, g in ((ctx, 1.0, 0.12), (gctx, 0.24 * glow_k, 0.0)):
        c.save()
        c.translate(bx, y)
        c.scale(k, k)
        tr.draw(c, fi, alpha=a, glow=g, light=light, pole=False)
        c.restore()
    cro.draw(ctx, x, y, h, P("hold_flag"), T, layer="back", **kw)
    ctx.save()
    ctx.translate(bx, y)
    ctx.scale(k, k)
    tr.draw(ctx, fi, light=light, cloth=False)
    ctx.restore()
    cro.draw(ctx, x, y, h, P("hold_flag"), T, layer="front", **kw)


def foreground(fc, st, which, ctx, gctx, flash, chaos):
    """weather & debris first (behind everyone: nothing may ever cross the Flag), then ledge, then people"""
    T = fc.T
    fi = (T - TB) * fc.fps
    lut = st["lutie"]
    atlas = st["atlas"]
    # slow push-ins on the foreground (the background camera pushes too, so the two layers part)
    if which == "B":
        z, (px, py) = 1.0 + 0.045 * seg(T, TB, TC, ease_in_out_sine), (900.0, 640.0)
    elif which == "C":
        z, (px, py) = 1.0 + 0.07 * seg(T, TC, TD, ease_in_out_sine), (1120.0, 420.0)
    else:
        z, (px, py) = 1.0, (0.0, 0.0)
    for c in (ctx, gctx):
        c.translate(px, py)
        c.scale(z, z)
        c.translate(-px, -py)
    if which == "B":
        FG.weather(ctx, gctx, T, 1.0, atlas, avoid=[flag_bbox(st, T, fi, 1360, 990, 590, -1)])
        FG.ground(ctx, gctx, T, 868, flash)
        FG.contact_shadow(ctx, 560, 1004, 150)
        FG.contact_shadow(ctx, 1330, 992, 190)
        tint = (STORM_TINT, clamp(0.5 - 0.35 * flash))
        _crocus_with_flag(ctx, gctx, st, T, fi, 1360, 990, 590, -1, 0.0, "serene", (-0.5, -0.1), tint,
                          (RIM_COL, 0.35 + 0.7 * flash, (-1.0, -0.5)))
        m = fc.tl.mouth("LUTIE", T)
        sp = any(w["s"] - 0.04 <= T <= w["e"] + 0.02 for w in fc.tl.words("L03"))
        lean = 0.14 * seg(T, 130.1, 130.4) * (1 - seg(T, 133.3, 133.6))
        ex = {"worried": 0.6, "surprised": 0.4}
        # "Master!" clutching her hood in the gale -> arms thrown wide on "Nobody understands anybody anymore!"
        u = seg(T, 130.98, 131.32) * (1 - seg(T, 133.25, 133.6))
        pose = P({"shout": 0.7 + 0.3 * u, "hold_hood": max(0.001, 1.0 - u)})
        lut.draw(ctx, 560, 1004, 505, pose, T, mouth=_shout(m) if sp else 0.0,
                 expr=ex, facing=1, look=(0.7, -0.15), wind=2.0, lean=lean, tint=tint,
                 rim=(RIM_COL, 0.35 + 0.7 * flash, (1.0, -0.5)))
    elif which == "C":
        xc, yc, hc = 1190.0, 1500.0, 1060.0
        FG.weather(ctx, gctx, T, 1.1, atlas, depth_scale=1.3, avoid=[flag_bbox(st, T, fi, xc, yc, hc, -1)])
        tint = (STORM_TINT, clamp(0.45 - 0.42 * flash))
        m = fc.tl.mouth("CROCUS", T)
        smile = seg(T, 134.0, 134.5)
        ex = {"serene": 1.0 - 0.5 * smile, "smile": 0.5 * smile}
        nod = -0.1 * pulse(T, 134.0, 0.14) - 0.05 * pulse(T, 135.4, 0.18)
        _crocus_with_flag(ctx, gctx, st, T, fi, xc, yc, hc, -1, m, ex, (-0.6, -0.35), tint,
                          (RIM_COL, 0.4 + 1.2 * flash, (-1.0, -0.35)), nod=nod)
    elif which == "D":
        FG.weather(ctx, gctx, T, 1.0 + chaos, atlas, avoid=[flag_bbox(st, T, fi, 350, 1014, 240, 1)])
        flying_debris(ctx, gctx, T, chaos, avoid=flag_bbox(st, T, fi, 350, 1014, 240, 1))
        FG.ground(ctx, gctx, T, 1000, flash, dark=True, n=60)
        sil = (0.025, 0.022, 0.04)
        rim = (RIM_COL, 0.6 + flash, (1.0, -0.6))
        lut.draw(ctx, 190, 1018, 205, P({"hold_hood": 1.0, "look_up": 0.7}), T, facing=1, wind=2.0, silhouette=sil,
                 rim=rim, look=(0.8, -0.8))
        _crocus_with_flag(ctx, gctx, st, T, fi, 350, 1014, 240, 1, 0.0, "serene", (0.8, -0.7), None, rim,
                          silhouette=sil, glow_k=3.0)


def flying_debris(ctx, gctx, T, chaos, avoid=None):
    """big fragments torn off the tower hurtling past the camera (2D, motion-streaked)"""
    if T < 136.6:
        return
    for j in range(int(3 + 9 * chaos)):
        t0 = 136.6 + hash01(j, 40) * 1.7
        if T < t0:
            continue
        u = (T - t0) / 0.9
        if u > 1.2:
            continue
        x0 = 700 + 900 * hash01(j, 41)
        y0 = 150 + 400 * hash01(j, 42)
        vx = (hash01(j, 43) - 0.35) * 2600
        vy = -300 + 1800 * u
        x = x0 + vx * u
        y = y0 + vy * u
        s = (40 + 110 * hash01(j, 44)) * (1 + 1.6 * u)
        a = T * (4 + 5 * hash01(j, 45)) + j
        if avoid is not None and avoid[0] - s < x < avoid[2] + s and avoid[1] - s < y < avoid[3] + s:
            continue
        c = [(0.20, 0.18, 0.25), (0.34, 0.34, 0.40), (0.14, 0.22, 0.24), (0.30, 0.26, 0.30), (0.05, 0.05, 0.07)][j % 5]
        for q in (2, 1, 0):
            aa = 0.22 * (3 - q) / 3
            ctx.save()
            ctx.translate(x - vx * 0.014 * q, y - vy * 0.014 * q)
            ctx.rotate(a - q * 0.05)
            ctx.scale(1, 0.4 + 0.6 * abs(math.cos(a)))
            rrect(ctx, -s / 2, -s * 0.4, s, s * 0.8, s * 0.06)
            ctx.set_source_rgba(*c, 1.0 if q == 0 else aa)
            ctx.fill_preserve()
            if q == 0:
                ctx.set_line_width(max(1.0, s * 0.05))
                ctx.set_source_rgba(0.6, 0.66, 0.95, 0.6)
                ctx.stroke()
                if j % 5 == 4:
                    rrect(ctx, -s * 0.4, -s * 0.32, s * 0.8, s * 0.64, s * 0.04)
                    ctx.set_source_rgba(0.62, 0.86, 1.0, 0.8)
                    ctx.fill()
            ctx.new_path()
            ctx.restore()

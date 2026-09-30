"""m01 - INCIPIT: THE TESSERACT (0.0-22.0).  Owner: M01.

The narthex vault at night, a Galla Placidia dome of lapis and gold stars, seen from its centre by a pilgrim's
candle.  0.8 a match strikes; the candle's warm pool wakes the gold.  2.0 the cantor sings and INCIPIT · OPVS ·
SCHISMATICVM sets itself, syllable by syllable, over its red sinopia in the gold band at the dome's foot.  4.9 we
look up; the rings of stars turn (N01).  8.4 at the zenith one tessera flashes and unfolds square -> cube ->
Tesseract, which turns in 4D (inside out); every star in the vault is its shadow and turns with it (N02).  14.8
("rotated") a Flag rotates into existence out of it: the first thing in the film not made of tiles; the
Tesseract's tiles scatter and circle it; 18.8 ("woven without seam") hold on the cloth.  19.6 the circling tiles
set themselves as OPVS SCHISMATICVM in the great band below it; the gloss in small white.  21.0-22.0 tilt down
out of the vault (hand-off to m02: lapis + gold stars moving up at HANDOFF_V px/s).

Helpers: m01_vault (3D dome renderer), m01_tesseract (4D), m01_titulus (stroke-font tituli), m01_candle.
"""
import json
import math

import cairo
import cv2
import numpy as np

from vx import *
from vx import mosaic as mz
from vx.flag import Flag, draw_pole, FlagTrack
from vx.foley import export_events, screen_pan

from scenes import m01_vault as mv
from scenes import m01_tesseract as mt
from scenes import m01_titulus as tt
from scenes import m01_candle as mc

POST = dict(bloom=0.42, bloom_thresh=0.64, bloom_radius=24.0, vignette=0.36, grain=0.0)

R_ = math.radians
KTEX = 935.0                  # band texture units per radian (= design px at f=935)
F_REF = 1060.0                # focal length of the zenith view (design px): zenith-plane px <-> radians

# ---------------------------------------------------------------- the dome's programme (colatitude from zenith)
G_CLIP = R_(16.3)             # the clipeus: the Tesseract's medallion, then the Flag's (imago clipeata)
SPHERES = [R_(x) for x in (18.0, 24.0, 30.5, 37.5, 45.0, 53.0, 61.5, 70.0, 76.6)]   # 8 turning spheres
G_BAND0, G_BAND1 = R_(78.2), R_(84.6)     # INCIPIT band (gold) at the dome's foot
G_LOW = R_(90.0)
TITLE_H1, TITLE_H2 = 116.0, 31.0           # title / gloss register heights (texture units)
TITLE_HW = (TITLE_H1 + TITLE_H2) / 2 / KTEX
TITLE_G = R_(16.9) + TITLE_HW  # the great band (a great circle, straight on screen), its top just below the clipeus
TITLE_S = R_(46.0)            # its half length along the circle
LAPIS_TINT = np.array([0.62, 1.45, 2.50])  # push the smalti toward Galla Placidia's saturated ultramarine
TILE = 0.0118                 # ground tessera (rad)

# ---------------------------------------------------------------- timing (global = local: m01 starts at 0)
T_STRIKE = 0.80
T_TILT_UP = (4.85, 8.30)
T_TESS = 8.40
T_FLAG = 14.78
T_TITLE = 19.60
T_TILT_DN = (21.0, 22.0)
PITCH0 = R_(12.0)
PITCH_END = R_(36.8)
F_HOLD = 1225.0              # the slow push onto the cloth (15.3 -> 19.3)
F_TITLE = 1100.0             # framing of the Flag + title (19.4 -> 21.0)
F_END = 1200.0               # focal length at the hand-off (tiles ~14 px)

CANTOR_PLAN = [("IN", 2.02), ("CI", 2.27), ("PIT", 2.52), ("O", 2.79), ("PVS", 3.03), ("SCHIS", 3.29),
               ("MA", 3.55), ("TI", 3.96), ("CVM", 4.20)]
INCIPIT = "INCIPIT·OPVS·SCHISMATICVM"
TITLE = "OPVS SCHISMATICVM"
GLOSS = "THE UNBABELING · SET IN GLASS AND GOLD"

L = mv.L
CANDLE_POS = np.array([0.0, 0.60, -0.10])        # where the candle's light comes from (world, dome radius 1)
FLAME_POS = np.array([0.0, 0.175, -0.022])        # where the flame is seen (in front of the pilgrim)
POLE = 600.0                                      # the Flag's pole (zenith px)
POLE_BASE = (-112.0, 298.0)                       # pole base in zenith px (x right, y down): the clipeus's foot
MANDORLA_C = (0.0, 0.0)                           # centre of the glory of circling tesserae (the clipeus)


# ================================================================ small helpers
def zdir(X, Y):
    """zenith-plane px (x right/east, y down/north, at F_REF) -> unit world direction"""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    v = np.stack([X / F_REF, Y / F_REF, np.ones_like(X)], -1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ease_speed(t, a, b, ramp):
    """distance fraction for a move over [a, b] that ramps smoothly to constant speed (C1) and ends at speed"""
    u = t - a
    T = b - a
    if u <= 0:
        return 0.0, 0.0
    vmax = 1.0 / (T - ramp / 2)
    if u < ramp:
        x = u / ramp
        return vmax * ramp * (x ** 3 - 0.5 * x ** 4), vmax * (3 * x * x - 2 * x ** 3)
    return vmax * ramp * 0.5 + vmax * (u - ramp), vmax


# ================================================================ camera
def camera(t):
    if t < T_TILT_UP[0]:
        k = smoothstep(0.0, T_TILT_UP[0], t)
        pitch = PITCH0 + R_(1.4) * k
        f = 935.0 + 14.0 * k
        yaw = R_(-1.0) + R_(0.8) * k
    elif t < T_TILT_UP[1]:
        u = (t - T_TILT_UP[0]) / (T_TILT_UP[1] - T_TILT_UP[0])
        e = ease_in_out_sine(u)
        e = e * e * (3 - 2 * e) * 0.4 + e * 0.6
        pitch = PITCH0 + R_(1.4) + (R_(90.0) - PITCH0 - R_(1.4)) * e
        f = 949.0 + (1020.0 - 949.0) * e
        yaw = R_(-0.2) * (1 - e)
    elif t < T_TILT_DN[0]:
        pitch = R_(90.0)
        f = 1020.0 + (F_REF - 1020.0) * smoothstep(T_TILT_UP[1], 14.6, t)
        f += (F_HOLD - F_REF) * smoothstep(15.3, 19.1, t)
        f += (F_TITLE - F_HOLD) * smoothstep(19.15, 19.75, t)
        yaw = 0.0
    else:
        d, _ = ease_speed(t, T_TILT_DN[0], T_TILT_DN[1], 0.30)
        pitch = R_(90.0) - (R_(90.0) - PITCH_END) * d
        f = F_TITLE + (F_END - F_TITLE) * smoothstep(T_TILT_DN[0], T_TILT_DN[1], t)
        yaw = 0.0
    # a gentle lens shift: centre the cloth for the hold, then frame the Flag over its (large) title card
    dy = 168.0 * smoothstep(15.3, 18.6, t) * (1.0 - smoothstep(19.15, 19.75, t))
    dy -= 40.0 * smoothstep(19.15, 19.75, t) * (1.0 - smoothstep(21.0, 21.9, t))
    return mv.Cam(yaw=yaw, pitch=pitch, f=f, cy=540.0 + dy)


def tilt_speed_px(t):
    """screen-space vertical speed of the content (design px/s) from the camera's pitch rate"""
    dt = 1.0 / 96
    c0, c1 = camera(t - dt), camera(t + dt)
    return (c1.pitch - c0.pitch) / (2 * dt) * c0.f


# ================================================================ the dome
def _lapis(zone):
    lap = np.array([L("lapis"), L("lapis_d"), L("night_vault"), L("lapis_l")])
    probs = {"clip": [0.20, 0.45, 0.35, 0.0], "sep": [0.10, 0.35, 0.55, 0.0],
             "vault": [0.62, 0.28, 0.06, 0.04]}[zone]

    def fn(g, th, rng):
        n = len(g)
        p = rng.choice(4, n, p=probs)
        c = lap[p] * (1.0 + rng.normal(0, 0.05, (n, 1))) * (LAPIS_TINT if zone != "sep" else LAPIS_TINT * 0.9)
        c[:, 0] *= 1.0 + rng.normal(0, 0.06, n)
        return c
    return fn


def _flat(key, var=0.05):
    base = L(key)
    return lambda g, th, rng: np.tile(base, (len(g), 1)) * (1 + rng.normal(0, var, (len(g), 1)))


def build_rings():
    z = []
    tl = TILE
    z.append(dict(g0=0.0, g1=G_CLIP - 3 * tl * 0.9, tile=tl * 0.9, colour_fn=_lapis("clip"), grp=0))
    # the clipeus rim: one gold course and a graded blue 'rainbow' toward the sky
    for k, key in enumerate(("gold", "lapis_l", "lapis")):
        z.append(dict(g0=G_CLIP - (3 - k) * tl * 0.9, g1=G_CLIP - (2 - k) * tl * 0.9, tile=tl * 0.9,
                      colour_fn=_flat(key), gold_fn=(lambda g, th: np.ones(len(g))) if key == "gold" else None,
                      grp=0, tilt=0.20 if key == "gold" else 0.13))
    z.append(dict(g0=G_CLIP, g1=SPHERES[0], tile=tl, colour_fn=_lapis("clip"), grp=0))
    for i in range(len(SPHERES) - 1):
        a, b = SPHERES[i], SPHERES[i + 1]
        z.append(dict(g0=a, g1=a + tl, tile=tl, colour_fn=_lapis("sep"), grp=1 + i))
        z.append(dict(g0=a + tl, g1=b, tile=tl, colour_fn=_lapis("vault"), grp=1 + i))
    z.append(dict(g0=SPHERES[-1], g1=G_BAND0, tile=tl, colour_fn=_lapis("vault"), grp=0))
    z.append(dict(g0=G_BAND1, g1=G_LOW, tile=tl * 0.9, colour_fn=_lapis("sep"), grp=0))
    return mv.Rings(z, seed=11)


def build_bands(inc_times, title_times, gloss_times):
    B = mv.Bands()
    rng = np.random.default_rng(4)
    # ---------------------------------------------------------- INCIPIT: a small circle at the dome's foot
    gmid = 0.5 * (G_BAND0 + G_BAND1)
    sgm = math.sin(gmid)
    Wb = mv.TAU * sgm * KTEX
    Hb = (G_BAND1 - G_BAND0) * KTEX
    inc = tt.Band([(INCIPIT, Wb / 2)], width=Wb, height=Hb, letter_h=Hb * 0.56, tile=7.2, letter_tile=4.7,
                  ground="gold", ink="lapis_d", rule="porphyry", pearl="marble", seed=21)
    T = inc.tiles
    tset = np.where(T["letter"], inc_times(inc), -1e9)
    bed = np.tile(L("sinopia") * 0.26 + L("plaster") * 0.06, (inc.n, 1))
    tilt = rng.normal(0, 0.13, (inc.n, 2))
    gi = T["gold"] > 0
    tilt[gi] = rng.normal(0, 0.17, (int(gi.sum()), 2))
    th0 = R_(90.0) + (Wb / 2) / (sgm * KTEX)
    inc_geom = (G_BAND0, th0, sgm)
    B.add(mv.BAND_RING, [G_BAND0, G_BAND1, th0, sgm, -1.0], np.stack([T["x"], T["y"]], 1), T["ang"], T["hu"],
          mv.lin(T["rgb"]), T["gold"], tilt, t_set=tset + 0.12, bed=bed, K=KTEX, extent=(Wb, Hb), wrap=Wb, seed=5,
          hv=T["hv"])
    # ---------------------------------------------------------- the great band (title + gloss): a great circle
    n = np.array([0.0, math.cos(TITLE_G), -math.sin(TITLE_G)])       # plane normal (+ = away from zenith)
    a = np.array([0.0, math.sin(TITLE_G), math.cos(TITLE_G)])        # its highest point
    b = np.array([1.0, 0.0, 0.0])                                    # east
    Wt = 2 * TITLE_S * KTEX
    Ht = 2 * TITLE_HW * KTEX
    ttl = tt.Band([(TITLE, Wt / 2)], width=Wt, height=TITLE_H1, letter_h=TITLE_H1 * 0.62, tile=8.6,
                  letter_tile=6.6, ground="lapis_d", ink="gold", rule="gold", pearl="marble", seed=33,
                  tracking=0.16)
    gl = tt.Band([(GLOSS, Wt / 2)], width=Wt, height=TITLE_H2, letter_h=TITLE_H2 * 0.56, tile=4.4,
                 letter_tile=2.35, ground="lapis_d", ink="marble", border=False, seed=34, tracking=0.20)
    T1, T2 = ttl.tiles, gl.tiles
    # the gloss in small WHITE smalti (a cool white that stays white under the warm candle)
    T2["rgb"][T2["letter"]] = np.array([0.80, 0.88, 1.0], np.float32)
    xs = np.concatenate([T1["x"], T2["x"]])
    ys = np.concatenate([T1["y"], T2["y"] + TITLE_H1])
    ang = np.concatenate([T1["ang"], T2["ang"]])
    hu = np.concatenate([T1["hu"], T2["hu"]])
    hv = np.concatenate([T1["hv"], T2["hv"]])
    rgb = np.vstack([T1["rgb"], T2["rgb"]])
    gold = np.concatenate([T1["gold"], T2["gold"]])
    ts1 = np.where(T1["letter"], title_times(ttl), -1e9)
    ts2 = np.where(T2["letter"], gloss_times(gl), -1e9)
    tset = np.concatenate([ts1, ts2])
    bed = np.tile(L("lapis_d") * 0.7, (len(xs), 1))
    bed[:len(ts1)][T1["letter"]] = L("sinopia") * 0.085 + L("lapis_d") * 0.62
    tilt = rng.normal(0, 0.14, (len(xs), 2))
    gi = gold > 0
    tilt[gi] = rng.normal(0, 0.18, (int(gi.sum()), 2))
    B.add(mv.BAND_GREAT, [*n, *a, *b, -TITLE_S, TITLE_S, TITLE_HW], np.stack([xs, ys], 1), ang, hu, mv.lin(rgb),
          gold, tilt, t_set=tset + 0.10, bed=bed, K=KTEX, extent=(Wt, Ht), wrap=0.0, seed=6, hv=hv)
    title_geom = (n, a, b)
    # ---------------------------------------------------------- the springing cornice on the wall
    Ww = mv.TAU * KTEX
    Hw = 0.052 * KTEX
    cor = tt.Band([], width=Ww, height=Hw, tile=6.5, ground="lapis_d", rule="gold", pearl="marble", seed=41)
    T3 = cor.tiles
    tilt = rng.normal(0, 0.15, (cor.n, 2))
    B.add(mv.BAND_WALL, [0.0, -0.052, R_(90.0) + math.pi, -1.0], np.stack([T3["x"], T3["y"]], 1), T3["ang"],
          T3["hu"], mv.lin(T3["rgb"]), T3["gold"], tilt, K=KTEX, extent=(Ww, Hw), wrap=Ww, seed=7, hv=T3["hv"])
    geo = dict(inc=inc_geom, title=title_geom, Wb=Wb, Hb=Hb, Wt=Wt)
    return B, inc, ttl, gl, geo, (ts1, ts2)


def band_ring_world(geo, x, y):
    g0, th0, sgm = geo["inc"]
    g = g0 + np.asarray(y) / KTEX
    th = th0 - np.asarray(x) / (sgm * KTEX)
    P = mv.sph(g, th)
    ex = np.stack([np.sin(th), -np.cos(th), np.zeros_like(th)], -1)          # -e_theta (x runs east)
    ey = np.stack([np.cos(g) * np.cos(th), np.cos(g) * np.sin(th), -np.sin(g)], -1)   # e_gamma (down)
    return P, ex, ey


def band_great_world(geo, x, y):
    n, a, b = geo["title"]
    s = np.asarray(x) / KTEX - TITLE_S
    c = np.asarray(y) / KTEX - TITLE_HW
    cs, ss = np.cos(s)[..., None], np.sin(s)[..., None]
    base = a * cs + b * ss
    P = np.cos(c)[..., None] * base + np.sin(c)[..., None] * n
    ex = -a * ss + b * cs
    ey = np.broadcast_to(n, P.shape).copy()
    return P, ex, ey


# ================================================================ timing helpers
def _cantor():
    """syllable onsets of A01 (the Vocalist's final file if present, else the planned onsets)."""
    p = CACHE / "foley" / "cantor_syllables.json"
    sy = list(CANTOR_PLAN)
    try:
        d = json.loads(p.read_text())
        inc = d.get("incipit") or []
        if len(inc) == len(CANTOR_PLAN):
            sy = [(s0[0], float(e["t"])) for s0, e in zip(CANTOR_PLAN, inc)]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return sy


def incipit_windows(band):
    """glyph index -> (t0, t1): each syllable's letters set across the syllable (the consonant leads)."""
    sy = _cantor()
    out = {}
    gi = 0
    text = band.text
    ends = [sy[i + 1][1] for i in range(len(sy) - 1)] + [sy[-1][1] + 0.40]
    for (syl, t0), t1 in zip(sy, ends):
        a = t0 - 0.07
        b = a + min(0.30, 0.85 * (t1 - t0))
        k = len(syl)
        for j in range(k):
            while gi < len(text) and text[gi] in "· ":
                gi += 1
            if gi >= len(text):
                break
            out[gi] = (a + (b - a) * j / k, a + (b - a) * (j + 1) / k)
            gi += 1
        if gi < len(text) and text[gi] == "·":
            out[gi] = (b + 0.02, b + 0.06)
            gi += 1
    return out


def title_windows(band):
    out = {}
    n = len(band.text)
    for gi, ch in enumerate(band.text):
        if ch == " ":
            continue
        t0 = T_TITLE + 0.36 * gi / n          # all letters land 19.60 -> 19.94 (legible from ~20.05)
        out[gi] = (t0, t0 + 0.12)
    return out


def gloss_windows(band):
    out = {}
    n = len(band.text)
    for gi, ch in enumerate(band.text):
        if ch == " ":
            continue
        t0 = 20.06 + 0.30 * gi / n
        out[gi] = (t0, t0 + 0.05)
    return out


# ================================================================ the Tesseract's choreography
TESS_SCALE = 108.0            # zenith px per unit of the tesseract's 3D projection
STAR_PE = 2                   # tiles per edge in a star (a tiny shadow of the Tesseract)
V3_REST = mt.rot3(1, 0.42) @ mt.rot3(0, -0.36)


def tess_angles(t):
    """XW angle a (the inside-out turn), YW angle b, and the 3D yaw of the view."""
    if t < 9.6:
        return 0.0, 0.0, 0.0
    # angular speed (rad/s): eases in, steady, then builds toward the Flag
    def w(u):
        return R_(62.0) * smoothstep(9.6, 10.5, u) * (1.0 + 1.35 * smoothstep(12.9, 14.5, u))
    # integrate numerically (cheap: ~0.1 s resolution with Simpson on a fine grid)
    ts = np.linspace(9.6, t, max(3, int((t - 9.6) * 60) | 1))
    ws = np.array([w(u) for u in ts])
    a = float(np.trapezoid(ws, ts)) if hasattr(np, "trapezoid") else float(np.trapz(ws, ts))
    return a, 0.38 * a, 0.22 * (t - 9.6)


def tess_pose(t):
    """-> dict(R, zs, ws, V3, alpha) for the big Tesseract at time t (None before it exists / after it is gone)."""
    if t < T_TESS or t > T_FLAG + 0.02:
        return None
    zs = smoothstep(8.78, 9.22, t)
    ws = smoothstep(9.08, 9.58, t)
    a, b, yaw = tess_angles(t)
    R = mt.rot(1, 3, b) @ mt.rot(0, 3, a)
    k = smootherstep(8.70, 9.25, t)
    V3 = mt.rot3(1, yaw) @ (np.eye(3) * (1 - k) + V3_REST * k)
    # orthonormalise the blend
    U, _, Vt = np.linalg.svd(V3)
    V3 = U @ Vt
    # the slice: just before the Flag the turning cube flattens into our plane
    zs *= 1.0 - 0.8 * smoothstep(14.45, 14.76, t)
    return dict(R=R, zs=zs, ws=ws, V3=V3, a=a)


def big_R(t):
    """the big Tesseract's 4D rotation (frozen after it gives itself up at the Flag)"""
    a, b, _ = tess_angles(min(t, T_FLAG))
    return mt.rot(1, 3, b) @ mt.rot(0, 3, a)


def star_R(t, ring):
    """4D rotation of the stars of sphere `ring`: the Petrie star (8-fold) before the Tesseract; a ripple from
    the zenith turns them into its square shadow; they turn with it; after the Flag they settle back."""
    din = T_TESS + 0.05 + 0.085 * ring
    dout = T_FLAG + 0.45 + 0.20 * ring
    w_in = smoothstep(din, din + 0.45, t)
    w_out = smoothstep(dout, dout + 1.0, t)
    if w_in <= 0.0:
        return mt.PETRIE
    Rb = big_R(t)
    R = mt.slerp4(mt.PETRIE, Rb, w_in) if w_in < 1.0 else Rb
    if w_out > 0.0:
        R = mt.slerp4(R, mt.PETRIE, w_out)
    return R


# ================================================================ setup
def setup(S):
    tl = S.tl
    rings = build_rings()
    B, inc, ttl, gl, geo, (ts1, ts2) = build_bands(lambda b: b.set_times(incipit_windows(b), seed=1),
                                                   lambda b: b.set_times(title_windows(b), seed=2),
                                                   lambda b: b.set_times(gloss_windows(b), seed=3))
    vault = mv.Vault(rings, B)
    st = dict(vault=vault, inc=inc, ttl=ttl, gl=gl, rings=rings, geo=geo, ts1=ts1, ts2=ts2)
    st["tiling"] = mt.Tiling(per_edge=22, seed=5)
    # ---------------------------------------------------------- stars on the turning spheres
    rng = np.random.default_rng(8)
    sc, sg, sph_i, srad = [], [], [], []
    for i in range(len(SPHERES) - 1):
        gm = 0.5 * (SPHERES[i] + SPHERES[i + 1])
        width = SPHERES[i + 1] - SPHERES[i]
        nst = max(6, int(round(mv.TAU * math.sin(gm) / (width * 1.12))))
        off = rng.uniform(0, mv.TAU)
        for k in range(nst):
            sc.append(off + k * mv.TAU / nst)
            sg.append(gm + rng.normal(0, 0.025) * width)
            sph_i.append(i + 1)
            big = (k % 2 == 0)
            srad.append(width * (0.33 if big else 0.24) * (1 + rng.normal(0, 0.04)))
    st["stars"] = dict(th=np.array(sc), g=np.array(sg), grp=np.array(sph_i), r=np.array(srad))
    ns = len(sc)
    npt = 16 + 32 * STAR_PE
    st["stars"]["tilt"] = rng.normal(0, 0.17, (ns, npt, 2))
    st["star_scale"] = 1.0 / 1.85          # raw Petrie shadow radius -> 1
    # ---------------------------------------------------------- the Flag
    st["flag"] = _flag_track(S)
    st["inc_tset"] = inc.set_times(incipit_windows(inc), seed=1)
    # ---------------------------------------------------------- the circling tiles -> the title
    st["swarm"] = _swarm(st)
    _export_sound(S, st)
    return st


# ================================================================ the Flag
FLAG_F0 = int(math.floor((T_FLAG - 0.25) * 24))     # scene-local frame of the flag track's frame 0


def _flag_track(S):
    path = S.cache / "flag_v7.npz"
    if path.exists():
        return FlagTrack.load(path)
    n = S.n - FLAG_F0
    fl = Flag(pole=POLE, seed=7, side=1)

    def wind(i):
        t = (FLAG_F0 + i) / 24.0
        g = 1.0 + 0.16 * math.sin(t * 1.3) + 0.08 * math.sin(t * 3.1 + 1.0)
        return (880.0 * g, -40.0, 60.0 * math.sin(t * 0.7))

    def furl(i):
        t = (FLAG_F0 + i) / 24.0
        return 1.0 - smoothstep(T_FLAG - 0.02, T_FLAG + 0.32, t)

    tr = fl.simulate(n, lambda i: (0.0, 0.0, 0.0), wind, warmup=24, furl_fn=furl)
    tr.save(path)
    return tr


def flag_xform(t, cam):
    """screen placement of the flag's local frame (pole base at 0,0; zenith px, y down):
    (x0, y0, (ux, uy) local +x on screen, (vx, vy) local +y on screen), and sx = the cloth's 'rotation into
    existence' about the pole (0 = edge-on .. 1 = whole, with a little overshoot)."""
    P = zdir(POLE_BASE[0], POLE_BASE[1])
    P2 = zdir(POLE_BASE[0], POLE_BASE[1] - 50.0)
    P3 = zdir(POLE_BASE[0] + 50.0, POLE_BASE[1])
    x, y, _, _ = cam.project(P)
    xu, yu, _, _ = cam.project(P2)
    xr, yr, _, _ = cam.project(P3)
    ux, uy = (xr - x) / 50.0, (yr - y) / 50.0          # local +x
    vx, vy = -(xu - x) / 50.0, -(yu - y) / 50.0        # local +y (down)
    u = (t - T_FLAG) / 0.46
    sx = 0.0 if u <= 0 else ease_out_back(min(1.0, u), 1.3)
    return float(x), float(y), (float(ux), float(uy), float(vx), float(vy)), max(sx, 0.0)


# ================================================================ the swarm: tesseract tiles -> title
def _swarm(st):
    """per flying tile: its start (the tesseract tile it was at T_FLAG), its orbit, and its title slot."""
    T_ = tess_tiles(st, T_FLAG, tess_pose(T_FLAG))
    x0, y0, kind = T_["x"], T_["y"], T_["kind"]
    n = len(x0)
    rng = np.random.default_rng(12)
    sw = dict(x0=x0, y0=y0, kind=kind, n=n)
    # after the burst the tiles are drawn into a MANDORLA round the Flag: three almond tracks of gold,
    # flowing in alternate directions (the glory of an icon, made of the Tesseract's own tesserae)
    trk = rng.integers(0, 3, n)
    sw["trk"] = trk
    sw["ax"] = np.array([352.0, 388.0, 424.0])[trk] + rng.normal(0, 4, n)
    sw["ay"] = np.array([338.0, 370.0, 402.0])[trk] + rng.normal(0, 4, n)
    sw["w"] = np.array([0.34, -0.25, 0.19])[trk] * (1 + rng.normal(0, 0.04, n))
    sw["ph"] = np.arctan2((y0 - MANDORLA_C[1]) / 1.3, x0 - MANDORLA_C[0]) + rng.normal(0, 0.05, n)
    dirn = np.stack([x0 - MANDORLA_C[0], y0 - MANDORLA_C[1]], 1)
    dirn /= np.linalg.norm(dirn, axis=1, keepdims=True) + 1e-9
    sw["burst"] = dirn * rng.uniform(140, 330, n)[:, None]
    sw["spin"] = rng.normal(0, 2.0, n)
    sw["tilt"] = rng.normal(0, 0.18, (n, 2))
    sw["m"] = 0
    sw["pick"] = np.zeros(0, int)
    # title letter slots (zenith px), in setting order
    ttl = st["ttl"]
    T1 = ttl.tiles
    let = np.nonzero(T1["letter"])[0]
    ts = st["ts1"][let]
    order = np.argsort(ts)
    let, ts = let[order], ts[order]
    P, _, _ = band_great_world(st["geo"], T1["x"][let], T1["y"][let])
    sx = F_REF * P[:, 0] / P[:, 2]
    sy = F_REF * P[:, 1] / P[:, 2]
    m = min(n, len(let))
    # the tiles lowest in their orbit when the title calls go; left ones to the left letters
    ox, oy, _ = _orbit(T_TITLE - 0.3, sw)
    cand = np.argsort(-oy + rng.normal(0, 40, n))[:m]
    cand = cand[np.argsort(ox[cand])]
    slot_order = np.argsort(sx[:m], kind="stable")
    pick = np.empty(m, int)
    pick[slot_order] = cand
    sw.update(m=m, pick=pick, slot_x=sx[:m], slot_y=sy[:m], t_land=ts[:m], ang_land=T1["ang"][let][:m],
              hu=T1["hu"][let][:m], hv=T1["hv"][let][:m])
    return sw


def _orbit(t, sw):
    """burst outward from the Tesseract, then drawn into the mandorla's flowing tracks"""
    u = max(0.0, t - T_FLAG)
    kb = 1.0 - math.exp(-u / 0.22)
    bx = sw["x0"] + sw["burst"][:, 0] * kb
    by = sw["y0"] + sw["burst"][:, 1] * kb
    ph = sw["ph"] + sw["w"] * u
    tx = MANDORLA_C[0] + sw["ax"] * np.cos(ph)
    ty = MANDORLA_C[1] + sw["ay"] * np.sin(ph)
    w = smoothstep(0.18, 1.25, u)
    x = bx * (1 - w) + tx * w
    y = by * (1 - w) + ty * w
    return x, y, w


def swarm_state(t, sw):
    """positions (zenith px) + angle + visibility of the flying tiles at t >= T_FLAG"""
    n = sw["n"]
    u = t - T_FLAG
    x, y, k = _orbit(t, sw)
    ang = sw["spin"] * u
    vis = np.ones(n, bool)
    land = np.zeros(n)
    # flight to the title slots: a swooping curve that dips below the band and rises into place
    idx = sw["pick"]
    if len(idx):
        tl = sw["t_land"]
        dur = 0.60
        f = np.clip((t - (tl - dur)) / dur, 0, 1)
        fe = f * f * (3 - 2 * f)
        xs0 = x[idx]
        ys0 = y[idx]
        cx = 0.5 * (xs0 + sw["slot_x"])
        cy = np.maximum(ys0, sw["slot_y"]) + 90.0
        bx = (1 - fe) ** 2 * xs0 + 2 * (1 - fe) * fe * cx + fe ** 2 * sw["slot_x"]
        by = (1 - fe) ** 2 * ys0 + 2 * (1 - fe) * fe * cy + fe ** 2 * sw["slot_y"]
        x[idx] = np.where(f > 0, bx, xs0)
        y[idx] = np.where(f > 0, by, ys0)
        ang[idx] = np.where(f > 0, ang[idx] * (1 - fe) + sw["ang_land"] * fe, ang[idx])
        vis[idx] = t < tl
        land[idx] = f
    # tiles without a slot fade into the vault as the title sets
    rest = np.ones(n, bool)
    rest[idx] = False
    fade = 1.0 - smoothstep(T_TITLE - 0.1, T_TITLE + 0.9, t)
    alpha = np.where(rest, fade, 1.0)
    return x, y, ang, vis & (alpha > 0.01), alpha, land


# ================================================================ sound
def _export_sound(S, st):
    ev = [dict(t=T_STRIKE - 0.10, kind="candle_strike", strength=0.5, pan=0.05, desc="match scrape"),
          dict(t=T_STRIKE, kind="candle_strike", strength=1.0, pan=0.05, desc="match flares (strike)"),
          dict(t=T_STRIKE, kind="strike", strength=1.0, pan=0.0, desc="the strike: candle ignition"),
          dict(t=T_STRIKE + 0.28, kind="candle", strength=0.6, pan=0.0,
               desc="candle wick catches; steady flame flicker from here (to 22.0)")]
    # INCIPIT letters: one tile_set per letter (+ 'set' for the composer)
    inc = st["inc"]
    tset = inc.set_times(incipit_windows(inc), seed=1)
    T = inc.tiles
    for gi, ch in enumerate(inc.text):
        m = (T["glyph"] == gi)
        if not m.any():
            continue
        t0 = float(tset[m].min())
        t1 = float(tset[m].max())
        xg = inc.glyph_x[gi] - inc.W / 2
        pan = float(np.clip(xg / 700.0, -1, 1))
        ev.append(dict(t=t0, kind="tile_set", strength=0.7 if ch != "·" else 0.4, pan=pan,
                       desc=f"INCIPIT letter {ch} sets (to {t1:.2f})"))
        ev.append(dict(t=t0, kind="set", strength=0.7, pan=pan, desc=f"INCIPIT letter {ch}"))
    ev.append(dict(t=T_TILT_UP[0], kind="camera", strength=0.5, pan=0.0,
                   desc=f"tilt up into the dome (to {T_TILT_UP[1]:.2f}); rings of stars start turning ~5.6"))
    ev.append(dict(t=T_TESS, kind="tesseract", strength=1.0, pan=0.0,
                   desc="one gold tessera flashes at the zenith; unfolds square->cube->tesseract (to 9.6)"))
    ev.append(dict(t=T_TESS + 0.05, kind="star_shadow", strength=0.8, pan=0.0,
                   desc="ripple: every star in the vault collapses into a tiny square (its shadow) (to 9.0)"))
    ev.append(dict(t=8.78, kind="whoosh", strength=0.5, pan=0.0, desc="square extrudes into a cube (to 9.22)"))
    ev.append(dict(t=9.08, kind="whoosh", strength=0.7, pan=0.0, desc="cube splits into the tesseract (to 9.58)"))
    # quarter turns + inside-out crossings of the XW rotation
    ts = np.arange(9.6, T_FLAG, 1 / 96)
    aa = np.array([tess_angles(x)[0] for x in ts])
    for k in range(1, 12):
        for target, kind in ((R_(45 + 90 * (k - 1)), "inside_out"), (R_(90 * k), "turn")):
            hit = np.nonzero((aa[:-1] < target) & (aa[1:] >= target))[0]
            if len(hit):
                ev.append(dict(t=float(ts[hit[0]]), kind=kind, strength=0.8, pan=0.0,
                               desc=("inner cube passes through the outer (4D inside-out)" if kind == "inside_out"
                                     else f"quarter turn {k} completes (aligned)")))
    ev.append(dict(t=14.45, kind="whoosh", strength=0.8, pan=0.0, desc="the tesseract flattens into our plane (to 14.78)"))
    ev.append(dict(t=T_FLAG, kind="emerge", strength=1.0, pan=-0.05, desc="the Flag's first cloth frame"))
    ev.append(dict(t=T_FLAG, kind="unfurl", strength=1.0, pan=-0.05,
                   desc="the Flag rotates into existence and unfurls (to 15.4)"))
    ev.append(dict(t=T_FLAG + 0.02, kind="shatter", strength=0.6, pan=0.0,
                   desc="the tesseract's tiles scatter outward and circle the Flag (to 19.3)"))
    ev.append(dict(t=18.79, kind="hold", strength=0.3, pan=0.0, desc="'woven without seam': hold on the cloth"))
    ev.append(dict(t=T_TITLE, kind="title", strength=1.0, pan=0.0, desc="the title sets (to 20.4)"))
    ttl = st["ttl"]
    tt1 = st["ts1"]
    T1 = ttl.tiles
    for gi, ch in enumerate(ttl.text):
        m = T1["glyph"] == gi
        if not m.any():
            continue
        xg = ttl.glyph_x[gi] - ttl.W / 2
        pan_ = float(np.clip(xg / 700, -1, 1))
        ev.append(dict(t=float(tt1[m].min()), kind="tile_set", strength=0.8, pan=pan_,
                       desc=f"title letter {ch} lands (to {float(tt1[m].max()):.2f})"))
        ev.append(dict(t=float(tt1[m].min()), kind="set", strength=0.8, pan=pan_, desc=f"title letter {ch}"))
    ev.append(dict(t=20.06, kind="tile_set", strength=0.35, pan=0.0, desc="the white gloss sets, small (to 20.4)"))
    ev.append(dict(t=T_TILT_DN[0], kind="camera", strength=0.7, pan=0.0, desc="tilt down (to 22.0)"))
    export_events(S, "sfx", ev)
    try:
        st["flag"].export_foley(S, "flag", f0=FLAG_F0)
    except Exception:
        pass


# ================================================================ lights
def lights(t):
    Ls = []
    cand = 0.0
    if t >= T_STRIKE:
        u = t - T_STRIKE
        grow = smoothstep(0.12, 1.1, u)
        flare = 2.2 * math.exp(-u / 0.08) + 0.5 * math.exp(-u / 0.35)
        cand = (0.22 + 0.78 * grow) * mc.flame_light(t) + flare
    elif t > T_STRIKE - 0.12:
        cand = 0.12 * (1 - (T_STRIKE - t) / 0.12)
    if cand > 0:
        Ls.append([*CANDLE_POS, 1.0, 0.72, 0.44, 0.34 * cand, 0.02])
    # the Tesseract's own light at the zenith
    if T_TESS <= t < 17.5:
        fl = 1.2 * math.exp(-(t - T_TESS) / 0.25)
        steady = 0.30 * smoothstep(T_TESS, 9.6, t) * (1 + 0.7 * smoothstep(12.9, 14.6, t))
        after = 1.0 - smoothstep(T_FLAG, 16.8, t)
        k = (fl + steady) * after
        if t >= T_FLAG:
            k += 0.5 * math.exp(-(t - T_FLAG) / 0.30)
        Ls.append([0.0, 0.0, 0.72, 1.0, 0.74, 0.40, 0.20 * k, 0.03])
    # the Flag's presence: a warm yellow glow from where it stands
    if t >= T_FLAG:
        k = smoothstep(T_FLAG, T_FLAG + 0.5, t) * 0.5 + 1.2 * math.exp(-(t - T_FLAG) / 0.35)
        P = zdir(POLE_BASE[0] + 100, POLE_BASE[1] - 420) * 0.78
        Ls.append([*P, 1.0, 0.86, 0.36, 0.12 * k, 0.03])
    return Ls


def prm(t, rv_ang=4.0, expo=1.5):
    gd, gm, gl = L("gold_d"), L("gold"), L("gold_l")
    mo = L("mortar")
    fill = smoothstep(3.8, 6.2, t)            # the eye opens to the blue night of the vault
    return [0.0016 + 0.010 * fill, 0.0020 + 0.030 * fill, 0.0052 + 0.100 * fill,       # ambient (night blue)
            expo,                         # exposure
            0.12,                         # grout (fraction of the tile half-size)
            72.0, 2.2,                    # gold shininess, sharp glint strength
            *mo, *gd, *gm, *gl,
            *(CANDLE_POS / np.linalg.norm(CANDLE_POS)), rv_ang, 0.8,    # reveal around the candle direction
            3.2,                          # white point
            0.55,                         # bevel
            0.30,                         # marble slab height (wall z units)
            1.18]                         # saturation


def exposure(t):
    # the eye adapts: a touch brighter looking up into the far vault
    return 1.45 + 0.52 * smoothstep(T_TILT_UP[0], T_TILT_UP[1], t) - 0.1 * smoothstep(T_TILT_DN[0], 22.0, t)


def reveal(t):
    """angular radius of the candle's pool on the vault: the band wakes first, the vault fades up slowly"""
    if t < T_STRIKE:
        return -0.5
    u = t - T_STRIKE
    return 0.12 + 0.62 * smoothstep(0.0, 3.2, u) + 2.6 * smoothstep(2.2, 7.5, u) ** 1.5


# ================================================================ sphere rotation (N01: the vault itself turns)
SPIN = [0.050, -0.040, 0.032, -0.026, 0.021, -0.017, 0.014, -0.011]       # rad/s per sphere (inner fastest)


def sphere_rot(t):
    # turning starts during the look up (5.6) and gently calms after the Flag appears
    def prof(u):
        return smoothstep(5.0, 7.0, u) * (1.0 - 0.6 * smoothstep(15.0, 17.5, u))
    ts = np.linspace(4.5, max(4.5, t), max(3, int(max(0.0, t - 4.5) * 12) | 1))
    I = float(np.trapezoid([prof(u) for u in ts], ts)) if t > 4.5 else 0.0
    rot = np.zeros(1 + len(SPIN))
    rot[1:] = np.array(SPIN) * I * 2.2
    return rot


# ================================================================ stars (every star is a shadow of the Tesseract)
def in_bands(P, geo):
    """bool (N,): world points covered by the INCIPIT band or the great band (stars hide under them)"""
    g = np.arccos(np.clip(P[:, 2], -1, 1))
    hit = (g > G_BAND0 - 0.004) & (g < G_BAND1 + 0.004)
    n, a, b = geo["title"]
    c = P @ n
    s = np.arctan2(P @ b, P @ a)
    hit |= (np.abs(c) < math.sin(TITLE_HW + 0.004)) & (np.abs(s) < TITLE_S)
    return hit


def star_frames(st, t, cam, rot, lights_, pr):
    """screen tiles for all stars at t -> (x, y, ang, hu, hv, rgb)"""
    S_ = st["stars"]
    ns = len(S_["th"])
    th = S_["th"] + rot[S_["grp"]]
    C = mv.sph(S_["g"], th)
    # cull stars far outside the view first
    zc = C @ cam.F
    keep = np.nonzero(zc > 0.25)[0]
    if not len(keep):
        return None
    C = C[keep]
    ring = S_["grp"][keep] - 1
    # shape per sphere: the orthographic shadow of the 4D pose (x -> east, y -> north on the tangent plane)
    npt = 16 + 32 * STAR_PE
    shapes = {}
    for rg in np.unique(ring):
        xy, ang, kind, dep = mt.wire_shape(star_R(t, int(rg)), per_edge=STAR_PE, ortho=True)
        shapes[int(rg)] = (xy * st["star_scale"], ang, kind)
    xy = np.stack([shapes[int(r)][0] for r in ring])            # (m, npt, 2)
    ang_l = np.stack([shapes[int(r)][1] for r in ring])
    kind = shapes[int(ring[0])][2]
    E = np.array([1.0, 0.0, 0.0]) - C * C[:, :1]
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    Nn = np.array([0.0, 1.0, 0.0]) - C * C[:, 1:2]
    Nn = Nn - E * np.sum(Nn * E, 1, keepdims=True)
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
    r = S_["r"][keep][:, None, None]
    off = xy * r
    Pt = C[:, None, :] + off[..., :1] * E[:, None, :] + off[..., 1:2] * Nn[:, None, :]
    Pt /= np.linalg.norm(Pt, axis=-1, keepdims=True)
    Pf = Pt.reshape(-1, 3)
    ca, sa = np.cos(ang_l), np.sin(ang_l)
    exf = (ca[..., None] * E[:, None, :] + sa[..., None] * Nn[:, None, :]).reshape(-1, 3)
    sz = np.where(kind == 0, 0.20, 0.15)[None, :] * S_["r"][keep][:, None]
    hu = (sz * 0.5).reshape(-1)
    x, y, ang, hu_s, hv_s, vis = mv.screen_tiles(cam, Pf, exf, None, hu, hu)
    on = vis & (x > -30) & (x < W + 30) & (y > -30) & (y < H + 30)
    on &= ~in_bands(Pf, st["geo"])
    if not on.any():
        return None
    tilt = S_["tilt"][keep].reshape(-1, 2)[on]
    # the ripple flashes as it passes each sphere; vertex jewels a touch brighter
    rip = np.exp(-((t - (T_TESS + 0.12 + 0.085 * ring)) / 0.09) ** 2) * 2.2 + \
        np.exp(-((t - (T_FLAG + 0.9 + 0.20 * ring)) / 0.2) ** 2) * 0.6
    gain = (np.repeat(1.0 + rip, npt) * np.tile(np.where(kind == 0, 1.15, 1.0), len(ring)))[on]
    n_on = int(on.sum())
    cols = mv.shade_tiles(Pf[on], np.tile(L("gold"), (n_on, 1)), np.ones(n_on), tilt, lights_, pr, gain=gain)
    return x[on], y[on], ang[on], hu_s[on], hv_s[on], cols


# ================================================================ the Tesseract's tiles (big one)
def tess_tiles(st, t, pose):
    """the big Tesseract laid in courses of gold (w=+1 cell), silver (w=-1 cell) and alternating gold/silver
    (the edges running in the 4th direction), pearls at the vertices. Zenith px, back to front.
    -> dict(x, y, ang, lng, wid, dep, kind, idx) (kind 0 gold, 1 silver, 3 pearl)"""
    til = st["tiling"]
    sl = til.slots(pose["R"], pose["zs"], pose["ws"], V3=pose["V3"], scale=TESS_SCALE)
    vx_, vdep, vk = mt.Tiling.vertices(pose["R"], pose["zs"], pose["ws"], V3=pose["V3"], scale=TESS_SCALE)
    k = np.clip(sl["k"], 0.45, 2.4)
    wid = 5.6 * k
    lng = np.clip(sl["along"] * 2 * 0.88, wid * 0.22, wid * 1.6)
    # two courses side by side along the cube edges, one along the 4th-direction edges
    nx, ny = -np.sin(sl["ang"]), np.cos(sl["ang"])
    two = ~til.wedge
    offs = [(-0.55, two), (0.55, two), (0.0, ~two)]
    X, Y, A, LN, WD, DP, KD, IX = [], [], [], [], [], [], [], []
    for c, m in offs:
        X.append(sl["x"][m] + c * wid[m] * nx[m])
        Y.append(sl["y"][m] + c * wid[m] * ny[m])
        A.append(sl["ang"][m])
        LN.append(lng[m])
        WD.append(wid[m] * (0.92 if c else 1.05))
        DP.append(sl["depth"][m] + 0.001 * c)
        kd = sl["kind"][m].copy()
        if c > 0:
            kd = np.where(kd == 0, 0, kd)
        KD.append(kd)
        IX.append(np.nonzero(m)[0])
    X.append(vx_[:, 0])
    Y.append(vx_[:, 1])
    A.append(np.full(16, math.pi / 4))
    vw = 11.0 * np.clip(vk, 0.5, 2.2)
    LN.append(vw)
    WD.append(vw)
    DP.append(vdep + 0.02)
    KD.append(np.full(16, 3))
    IX.append(len(til.X) + np.arange(16))
    d = dict(x=np.concatenate(X), y=np.concatenate(Y), ang=np.concatenate(A), lng=np.concatenate(LN),
             wid=np.concatenate(WD), dep=np.concatenate(DP), kind=np.concatenate(KD), idx=np.concatenate(IX))
    o = np.argsort(d["dep"], kind="stable")
    return {k_: v[o] for k_, v in d.items()}


def tess_frames(st, t, cam, lights_, pr):
    """screen tiles of the big Tesseract (8.4..14.8) and of its scattered tiles afterwards"""
    out = []
    if T_TESS <= t < T_TESS + 0.30:
        # the first tessera: one gold square flashing at the zenith, growing into the square's outline
        u = (t - T_TESS) / 0.30
        size = 30.0 + (2 * TESS_SCALE * 1.2 - 30.0) * smoothstep(0.35, 1.0, u)
        a = 1.0 - smoothstep(0.55, 1.0, u)
        P = zdir(np.array([0.0]), np.array([0.0]))
        ex = np.array([[1.0, 0.0, 0.0]])
        x, y, ang, hu, hv, vis = mv.screen_tiles(cam, P, ex, None, np.array([size / 2 / F_REF]),
                                                 np.array([size / 2 / F_REF]))
        g = 1.0 + 3.0 * math.exp(-(t - T_TESS) / 0.07)
        col = mv.shade_tiles(P, L("gold")[None], np.ones(1), np.array([[0.05, -0.08]]), lights_, pr,
                             gain=np.array([g]), emit=np.array([L("gold_l") * 0.22 * g]))
        out.append((x, y, ang, hu, hv, col, np.array([a]), np.array([0.0])))
    pose = tess_pose(t)
    if pose is not None and t >= T_TESS + 0.14:
        T_ = tess_tiles(st, t, pose)
        til = st["tiling"]
        fcol = np.concatenate([til.f, np.full(16, 0.5)])[T_["idx"]]
        al = np.clip((t - (T_TESS + 0.14 + 0.16 * fcol)) / 0.08, 0, 1)
        P = zdir(T_["x"], T_["y"])
        ex = np.stack([np.cos(T_["ang"]), np.sin(T_["ang"]), np.zeros(len(P))], -1)
        x, y, ang, hu, hv, vis = mv.screen_tiles(cam, P, ex, None, T_["lng"] / 2 / F_REF, T_["wid"] / 2 / F_REF)
        kind = T_["kind"]
        base = np.where(kind[:, None] == 1, L("silver"), np.where(kind[:, None] == 3, L("marble"), L("gold")))
        gold = np.where(kind == 1, 0.0, np.where(kind == 3, 0.0, 1.0))
        dep = T_["dep"]
        dk = np.clip(0.50 + 0.50 * (dep - dep.min()) / max(1e-6, float(np.ptp(dep))), 0.5, 1.0)
        glow = 0.30 + 0.9 * smoothstep(12.9, 14.7, t)
        emit = base * (0.07 * glow) * dk[:, None] + (L("gold") * 0.10 * glow)[None] * (kind != 1)[:, None]
        tilt = np.concatenate([til.tilt, np.zeros((16, 2))])[T_["idx"]] * 0.16
        col = mv.shade_tiles(P, base * dk[:, None], gold, tilt, lights_, pr, emit=emit)
        # silver reads as cool bright metal: lift it a touch
        col = np.where(kind[:, None] == 1, np.clip(col * 1.15 + 0.03, 0, 1), col)
        out.append((x, y, ang, hu, hv, col, al, np.full(len(x), 0.35)))
    swarm = []
    if t >= T_FLAG:
        sw = st["swarm"]
        xs, ys, ang_l, vis, alpha, land = swarm_state(t, sw)
        if vis.any():
            idx = np.nonzero(vis)[0]
            P = zdir(xs[idx], ys[idx])
            ex = np.stack([np.cos(ang_l[idx]), np.sin(ang_l[idx]), np.zeros(len(idx))], -1)
            kind = sw["kind"][idx]
            wid = np.where(kind == 3, 9.0, 7.4)
            x, y, ang, hu, hv, _ = mv.screen_tiles(cam, P, ex, None, wid / 2 / F_REF, wid / 2 / F_REF)
            base = np.where(kind[:, None] == 3, L("marble"), np.where(kind[:, None] == 1, L("silver"), L("gold")))
            gold = np.where(kind == 0, 1.0, 0.0)
            fl = 1.0 + 0.9 * math.exp(-(t - T_FLAG) / 0.25)
            col = mv.shade_tiles(P, base, gold, sw["tilt"][idx], lights_, pr, gain=np.full(len(idx), fl),
                                 emit=(L("gold") * 0.05)[None] * np.ones((len(idx), 1)))
            swarm.append((x, y, ang, hu, hv, col, alpha[idx], np.zeros(len(idx))))
    return out, swarm


# ================================================================ letter pops (INCIPIT, title, gloss)
def pop_frames(st, t, cam, lights_, pr):
    out = []
    geo = st["geo"]
    # INCIPIT: tiles pressed into the bed with a glint
    inc = st["inc"]
    T = inc.tiles
    if 1.8 < t < 5.0:
        ts = st["inc_tset"]
        vis, scale, flash = tt.pop(t, ts, 0.12)
        idx = np.nonzero(vis)[0]
        if len(idx):
            P, exd, eyd = band_ring_world(geo, T["x"][idx], T["y"][idx])
            a = T["ang"][idx]
            ex = exd * np.cos(a)[:, None] + eyd * np.sin(a)[:, None]
            x, y, ang, hu, hv, _ = mv.screen_tiles(cam, P, ex, None, T["hu"][idx] / KTEX * scale[idx],
                                                   T["hv"][idx] / KTEX * scale[idx])
            col = mv.shade_tiles(P, mv.lin(T["rgb"][idx]), T["gold"][idx], np.zeros((len(idx), 2)), lights_, pr,
                                 emit=(L("gold_l") * 0.9)[None] * flash[idx, None] ** 2)
            out.append((x, y, ang, hu, hv, col, np.ones(len(idx)), np.full(len(idx), 0.6)))
    # title / gloss: a flash where each flying tile lands, and the gloss tiles popping
    if T_TITLE < t < 21.2:
        ttl, gl = st["ttl"], st["gl"]
        for band, ts, yoff, emitk in ((ttl, st["ts1"], 0.0, 1.3), (gl, st["ts2"], TITLE_H1, 0.6)):
            Tb = band.tiles
            vis, scale, flash = tt.pop(t, ts, 0.10)
            idx = np.nonzero(vis)[0]
            if not len(idx):
                continue
            P, exd, eyd = band_great_world(geo, Tb["x"][idx], Tb["y"][idx] + yoff)
            a = Tb["ang"][idx]
            ex = exd * np.cos(a)[:, None] + eyd * np.sin(a)[:, None]
            sc = scale[idx] if band is gl else 1.0 + 0.25 * flash[idx]
            x, y, ang, hu, hv, _ = mv.screen_tiles(cam, P, ex, None, Tb["hu"][idx] / KTEX * sc,
                                                   Tb["hv"][idx] / KTEX * sc)
            col = mv.shade_tiles(P, mv.lin(Tb["rgb"][idx]), Tb["gold"][idx], np.zeros((len(idx), 2)), lights_, pr,
                                 emit=(L("gold_l") * emitk)[None] * flash[idx, None] ** 2)
            out.append((x, y, ang, hu, hv, col, np.ones(len(idx)), np.full(len(idx), 0.5)))
    return out


# ================================================================ render
def _splat_all(img, fc, groups, rim=1.0):
    for x, y, ang, hu, hv, col, al, seat in groups:
        mv.splat(img, fc.s, x, y, ang, hu, hv, col, alpha=al, rim=rim, light=(0.0, 1.0), seat=seat)


def render(fc, st):
    t = fc.T
    cam = camera(t)
    vault = st["vault"]
    if "inc_tset" not in st:                 # (set in setup; kept for safety)
        st["inc_tset"] = st["inc"].set_times(incipit_windows(st["inc"]), seed=1)
    Ls = lights(t)
    pr = prm(t, reveal(t), exposure(t))
    rot = sphere_rot(t)
    lin_img = vault.render(fc, cam, t, Ls, rot, pr)
    img = mv.to_srgb_inplace(lin_img)
    # loose / moving tesserae
    sf = star_frames(st, t, cam, rot, Ls, pr)
    if sf is not None:
        x, y, ang, hu, hv, col = sf
        mv.splat(img, fc.s, x, y, ang, hu, hv, col, rim=0.8, light=(0.0, 1.0), seat=0.5)
    _splat_all(img, fc, pop_frames(st, t, cam, Ls, pr))
    tf, swarm = tess_frames(st, t, cam, Ls, pr)
    if tf and T_TESS <= t < T_FLAG + 0.12:
        # the Tesseract is luminous: a warm-white glow round its tiles (never flag-yellow)
        glow = np.zeros_like(img)
        _splat_all(glow, fc, [(g[0], g[1], g[2], g[3] * 1.6, g[4] * 1.6, g[5], g[6], np.zeros(len(g[0])))
                              for g in tf])
        k = 0.35 + 0.7 * smoothstep(12.9, 14.75, t) + 0.35 * math.exp(-abs(t - T_FLAG) / 0.15)
        gl_ = blur(glow, 18 * fc.s) * 0.6 * k + blur(glow, 60 * fc.s) * 0.5 * k
        lum = gl_.mean(axis=2, keepdims=True)
        img += gl_ * 0.45 + lum * np.array([1.0, 0.93, 0.80], np.float32) * 0.55
    _splat_all(img, fc, tf)
    _splat_all(img, fc, swarm)
    # the candle (screen space, projected from its place in front of the pilgrim)
    if t >= T_STRIKE - 0.15:
        x, y, z, vis = cam.project(FLAME_POS)
        if vis and -300 < y < 1500:
            u = t - T_STRIKE
            lit = 0.0 if u < 0.16 else min(1.0, 0.2 + 0.8 * smoothstep(0.16, 1.0, u))
            # the wax is seen only by the flare and then by its own flame
            seen = 0.0 if u < 0 else min(1.0, lit + 1.6 * math.exp(-u / 0.12))
            h = 100.0 * cam.f / 935.0
            mc.candle(img, fc.s, float(x), float(y), h, t, lit=lit, light=seen)
            if u < 1.25:
                mc.match(img, fc.s, float(x), float(y), h, u)
    # the tilt's motion smear (a 180-degree shutter), before the Flag: the cloth is never smeared
    v = abs(tilt_speed_px(t))
    if v > 60:
        L_ = int(round(v / 48.0 * fc.s))
        if L_ >= 2:
            img = cv2.blur(img, (1, L_ | 1))
    # THE FLAG: whole cloth, composited last
    if t >= T_FLAG - 0.16:
        img = _draw_flag(img, fc, st, t, cam)
    return img


def _draw_flag(img, fc, st, t, cam):
    x, y, (ux, uy, vx, vy), sx = flag_xform(t, cam)
    if y > H + 800 or y < -900:
        return img
    cv = Canvas(fc)
    ctx = cv.ctx
    k = smoothstep(T_FLAG + 1.0, T_FLAG + 3.5, t)
    # the Flag is lit from the front (the candle below us + the Tesseract's afterglow): the brightest yellow
    light = tuple(a * (1 - k) + b * k for a, b in zip((-0.35, -0.50, 0.70), (-0.20, 0.35, 0.80)))
    M_pole = cairo.Matrix(ux, uy, vx, vy, x, y)
    if t < T_FLAG:
        # the pole materialises out of the slice, from its middle outward
        g = smoothstep(T_FLAG - 0.16, T_FLAG, t)
        mid = -POLE * 0.55
        ctx.save()
        ctx.transform(M_pole)
        draw_pole(ctx, 0.0, mid + POLE * 0.55 * g, 0.0, mid - POLE * 0.45 * g, width=POLE * 0.0167,
                  alpha=g, light=light)
        ctx.restore()
        return cv.over(img)
    tr = st["flag"]
    i = (t * 24.0) - FLAG_F0
    glow = 0.10 + 0.30 * math.exp(-(t - T_FLAG) / 0.8)
    if abs(sx - 1.0) < 1e-3:
        ctx.save()
        ctx.transform(M_pole)
        tr.draw(ctx, i, pole=True, cloth=True, light=light, glow=glow)
        ctx.restore()
        return cv.over(img)
    # the cloth rotates into existence about the pole (the pole is already whole)
    ctx.save()
    ctx.transform(M_pole)
    tr.draw(ctx, i, pole=True, cloth=False, light=light)
    ctx.restore()
    if sx > 0.01:
        ctx.save()
        ctx.transform(cairo.Matrix(ux * sx, uy * sx, vx, vy, x, y))
        tr.draw(ctx, i, pole=False, cloth=True, light=light, glow=glow)
        ctx.restore()
    return cv.over(img)


def post(fc, st):
    """per-frame look overrides (the module-level POST holds the rest)"""
    t = fc.T
    # a whisper more bloom when the Flag arrives
    return dict(bloom=0.42 + 0.25 * math.exp(-max(0.0, t - T_FLAG) / 0.6) * (t >= T_FLAG))

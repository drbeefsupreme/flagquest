"""m04 - FLAGARTHA: THE HYPERMIND AND THE DISPUTATION (62.5-95.75). Owner M04.

The apse at night. From m03's gold point an eye opens (hypermind_wake 62.9); pull back to the Hypermind, an
ophanim-polycandelon in a Sinai-style mandorla: gimballed gold wheels full of eyes around a great eye, tiers of
pierced gold discs with glass lamps and coiled lines (a dilution refrigerator as a Byzantine chandelier) on chains,
six seraphic wings. H01: the wings unfold, the wheels begin to turn in 3D, and the wings ignite one per syllable of
"Six ca-nons in-gest-ed" (CANON I..VI along their arms); the eyes open. H02: the wheels whirl and LOCK face-on at
hypermeme 74.2 (the machine becomes a flat icon again); the great eye's pupil opens into light and a plain Flag (real
vx.flag cloth, the only thing not made of tesserae, lighting the gold as it comes) descends to hover over the donors
Lutie and Crocus (L01/C01). 81.2 videtur: the wall re-sets around the Flag in a split-flap wave into a disputation
in the form of the Summa (m04_disp.py). Hard cut at 95.75.

Stopgap mosaic tech: m04_tiles.py (camera-projected layouts + point-light tesserae); cartoons: m04_art.py.
"""
import math

import cairo
import numpy as np

from vx import *
from vx import mosaic as mz
from vx.flag import Flag, FlagTrack
from vx.foley import export_events

from scenes import m04_tiles as mt
from scenes import m04_art as art
from scenes import m04_disp as dsp

POST = dict(bloom=0.35, bloom_thresh=0.78, grain=0.0, vignette=0.3)
VERSION = "m04v3"

# ------------------------------------------------------------------ timing (global seconds)
T0, T1 = 62.5, 95.75
WAKE = 62.9
IGN = [66.03, 66.46, 66.88, 67.30, 67.76, 68.22]       # "Six ca-nons in-gest-ed"
EVERY, TRUE_ = 68.93, 71.11
HYPERMEME, LOCK = 72.43, 74.2
TURN0 = 64.7                                            # the wheels leave the face-on rosette
EMERGE, HOVER = 74.45, 76.35
VIDETVR = 81.2
FLIP0 = 80.98                                           # split-flap wave starts (front leaves the Flag)
F81 = (1050.0, 1190.0, 1.28)                            # camera framing held through the re-set (world_H)
POLE = 400.0
HOVER_TOP = 1124.0                                      # pole top when hovering (world_H)
Z0 = 7.4                                                # opening zoom on the great eye


def post(fc, st):
    """per-frame look: the first frames match m03's last one (bloom 0.30 @0.76, no vignette), then the basilica"""
    k = smoothstep(62.5, 64.0, fc.T)
    return dict(bloom=lerp(0.30, 0.35, k), bloom_thresh=lerp(0.76, 0.78, k), bloom_radius=lerp(20.0, 22.0, k),
                vignette=lerp(0.0, 0.3, k))


# ------------------------------------------------------------------ camera
def camera_H(T):
    """(cx, cy, zoom, rot) over world_H"""
    if T < WAKE:
        z = Z0 - 0.15 * (T - T0) / (WAKE - T0)
        c = (art.CX, art.CY)
    elif T < 63.3:
        z = Z0 - 0.15 - 0.45 * smoothstep(WAKE, 63.3, T)
        c = (art.CX, art.CY)
    elif T < 67.4:
        e = ease_in_out(clamp((T - 63.3) / (67.4 - 63.3)), 2.2)
        z = math.exp(lerp(math.log(Z0 - 0.6), 0.0, e))
        c = (art.CX, lerp(art.CY, 640.0, e))
    elif T < 72.4:
        e = seg(T, 67.4, 72.4, ease_in_out_sine)
        z = lerp(1.0, 1.04, e)
        c = (art.CX, lerp(640.0, 655.0, e))
    elif T < LOCK:
        e = seg(T, 72.4, LOCK, ease_in_out_sine)
        z = lerp(1.04, 1.1, e)
        c = (art.CX, lerp(655.0, 690.0, e))
    elif T < 76.5:
        e = seg(T, LOCK + 0.2, 76.5, ease_in_out_sine)
        z = lerp(1.1, 1.0, e)
        c = (art.CX, lerp(690.0, 1030.0, e))
    else:
        e = seg(T, 76.5, 80.4, ease_in_out_sine)
        z = lerp(1.0, F81[2], e)
        c = (art.CX, lerp(1030.0, F81[1], e))
    rot = 0.0035 * math.sin(T * 0.31) * smoothstep(63.5, 66.0, T) * (1 - smoothstep(79.5, 80.4, T))
    cx, cy = c
    if T >= LOCK:
        k = math.exp(-(T - LOCK) * 5.0)
        dx, dy = shake(T, 7.0 * k, 11.0, seed=4)
        cx, cy = cx + dx / z, cy + dy / z
    return cx, cy, z, rot


# ------------------------------------------------------------------ the wheels
def _base_speed(T, w1, w2, whirl):
    if T < TURN0 or T >= LOCK:
        return 0.0
    w = w1 * smoothstep(TURN0, 66.6, T) + (w2 - w1) * smoothstep(EVERY, TRUE_, T)
    w += (whirl - w2) * smoothstep(HYPERMEME - 0.1, HYPERMEME + 0.45, T)
    w -= (whirl - 1.4) * smoothstep(73.0, 74.1, T)
    return w


SPEEDS = [(0.42, 0.85, 4.2), (0.55, 1.1, 5.4), (0.7, 1.35, 6.4), (0.15, 0.3, 1.2), (-0.2, -0.36, -1.5),
          (0.25, 0.42, 1.8)]


def build_angles():
    """angle tables on a 1 ms grid, integrated BACKWARDS from the lock so every axis lands exactly on its locked
    pose at 74.2; each profile is scaled so the wheels also START face-on (a,b,c multiples of 2 pi, spins
    multiples of one eye step): rosette -> turning gimbal -> the same rosette, locked."""
    grid = np.arange(T0 - 0.1, LOCK + 1e-9, 0.001)
    tabs, ends = [], []
    for k, (w1, w2, wh) in enumerate(SPEEDS):
        w = np.array([_base_speed(T, w1, w2, wh) for T in grid])
        tot = float(np.sum(0.5 * (w[1:] + w[:-1])) * 0.001)
        step = 2 * math.pi if k < 3 else 2 * math.pi / art.RINGS[k - 3][2]
        n = max(1, round(abs(tot) / step))
        sc = n * step / max(1e-9, abs(tot))
        w *= sc
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * 0.001)])
        tabs.append(-(cum[-1] - cum))
        ends.append(_base_speed(LOCK - 0.002, w1, w2, wh) * sc)
    return grid, np.array(tabs), np.array(ends)


def angles_at(st, T):
    grid, tabs, ends = st["ang"]
    if T <= LOCK:
        i = (T - grid[0]) / 0.001
        i0 = int(np.clip(math.floor(i), 0, len(grid) - 2))
        f = min(1.0, max(0.0, i - i0))
        return tabs[:, i0] * (1 - f) + tabs[:, i0 + 1] * f
    tau = T - LOCK
    wv = 2 * math.pi * 4.5
    return ends / wv * math.exp(-tau * 7.0) * np.sin(wv * tau)


def ring_rots(st, T):
    a, b, c, p0, p1, p2 = angles_at(st, T)
    R0 = mt.rot3(0, a, 0)
    R1 = R0 @ mt.rot3(b, 0, 0)
    R2 = R1 @ mt.rot3(0, c, 0)
    return [R0 @ mt.rot3(0, 0, p0), R1 @ mt.rot3(0, 0, p1), R2 @ mt.rot3(0, 0, p2)]


def ring_speed(st, T):
    """0..1 for the sound team"""
    a0 = angles_at(st, T - 0.02)
    a1 = angles_at(st, T + 0.02)
    return float(np.clip(np.abs(a1[:3] - a0[:3]).sum() / 0.04 / 14.0, 0, 1))


# ------------------------------------------------------------------ wings
def _wing_unfold_t(j, side):
    return 64.25 + 0.16 * [2, 2, 0, 0, 4, 4][j] + (0.05 if side < 0 else 0.0)


def wing_state(T):
    """per wing: (angle_deg, ignition 0..1, front 0..1.3, unfold 0..1)"""
    out = {}
    for j, (name, side, off, a_open, a_fold, L, Wd) in enumerate(art.WINGS):
        k = art.IGN_ORDER.index(name)
        t_un = _wing_unfold_t(j, side)
        u = ease_out_back(clamp((T - t_un) / 1.55), 1.05) if T > t_un else 0.0
        ang = lerp(a_fold, a_open, u)
        ti = IGN[k]
        br = 1.4 * math.sin(T * 1.9 + j) * smoothstep(65.8, 67.0, T)
        flap = 4.0 * math.exp(-max(0.0, T - ti) * 4.0) * math.sin(max(0.0, T - ti) * 14.0) if T > ti else 0.0
        ang += side * (br + flap) * (-1 if name.startswith("lo") else 1)
        front = clamp((T - ti) / 0.42, 0.0, 1.3) if T > ti else 0.0
        ign = smoothstep(ti, ti + 0.42, T)
        out[name] = (ang, ign, front, u)
    return out


def wing_matrix(name, ang_deg):
    spec = next(w for w in art.WINGS if w[0] == name)
    _, side, off, _, _, L, Wd = spec
    ox, oy = art.wing_origin(L, Wd)
    Mr = mt.sim(art.CX + off[0], art.CY + off[1], math.radians(ang_deg), 1.0, mirror=side < 0)
    return mt.compose(Mr, mt.affine(1, 0, 0, 1, -ox, -oy))


# ------------------------------------------------------------------ eyes
RING_EYES_T = [64.55, 64.05, 63.5]


def eyes_state(T, k):
    """ring k: per-eye (open, look)"""
    ne = art.RINGS[k][2]
    opens, looks = [], []
    for i in range(ne):
        h = hash01(i * 7 + k * 101, 3)
        t1 = RING_EYES_T[k] + i * 0.045 + 0.1 * h
        o = 0.6 * smoothstep(t1, t1 + 0.25, T)
        t2 = EVERY + 0.12 * k + (i / ne) * 1.1
        o += 0.4 * smoothstep(t2, t2 + 0.3, T)
        o *= _blink(T, i + 17 * k)
        opens.append(o)
        looks.append(_look(T, i + 31 * k))
    return opens, looks


def _blink(T, i):
    """occasional blinks + the unison blinks (lock, after 'always going to be a Flag')"""
    v = 1.0
    ph = hash01(i, 9) * 5.0
    per = 3.2 + 2.0 * hash01(i, 5)
    x = ((T + ph) % per) / per
    if T > 64.8:
        v = min(v, 1.0 - 0.95 * math.exp(-((x - 0.5) * per / 0.07) ** 2))
    for tb in (LOCK + 0.02, 80.55):
        v = min(v, 1.0 - 0.97 * pulse(T, tb + 0.07, 0.06))
    return v


def _look(T, i):
    """wandering gaze -> the camera on 'true' -> down at the Flag once it descends"""
    lx = 0.8 * fbm1(T * 0.35 + i * 3.1, i)
    ly = 0.7 * fbm1(T * 0.3 + i * 1.7, i + 50)
    k = smoothstep(TRUE_ - 0.05, TRUE_ + 0.15, T)
    return (lx * (1 - k), ly * (1 - k))


def _look_at_flag(T, st, k):
    """after the lock the rings are face-on: every eye looks toward the Flag (in its own rotated frame)"""
    ne = art.RINGS[k][2]
    rm = 0.5 * (art.RINGS[k][0] + art.RINGS[k][1])
    fx, fy = st["flag"].cloth_center(max(0, min(st["flag"].n - 1, int(round((T - T0) * 24)))))
    out = []
    for i in range(ne):
        a = -math.pi / 2 + i * 2 * math.pi / ne
        ex, ey = art.CX + rm * math.cos(a), art.CY + rm * math.sin(a)
        dx, dy = fx - ex, fy - ey
        rot = a + math.pi / 2
        # into the eye frame (x along rot)
        u = dx * math.cos(rot) + dy * math.sin(rot)
        v = -dx * math.sin(rot) + dy * math.cos(rot)
        n = math.hypot(u, v) + 1e-6
        out.append((u / n, v / n))
    return out


# ------------------------------------------------------------------ the Flag
def flag_pole(T):
    """pole base (x, y) world_H and lean"""
    top0 = art.CY - 6.0
    if T <= EMERGE:
        top = top0
    else:
        e = ease_in_out(clamp((T - EMERGE) / (HOVER - EMERGE)), 2.4)
        top = lerp(top0, HOVER_TOP, e)
    bob = 7.0 * math.sin((T - HOVER) * 1.6) * smoothstep(HOVER - 0.4, HOVER + 1.2, T)
    ang = 0.018 * math.sin(T * 0.9 + 1.0) * smoothstep(EMERGE, HOVER + 1.0, T)
    return art.CX, top + POLE + bob, ang


def build_flag(S):
    path = S.cache / f"flag_{VERSION}.npz"
    if path.exists():
        return FlagTrack.load(path)
    fps = S.fps

    def pole_fn(i):
        return flag_pole(S.start + i / fps)

    def wind_fn(i):
        T = S.start + i / fps
        g = 0.5 + 0.5 * math.sin(T * 0.7)
        return (470.0 + 120.0 * g, -30.0, 25.0 * math.sin(T * 0.5))
    tr = Flag(pole=POLE, seed=41, side=1).simulate(S.n, pole_fn, wind_fn, fps=fps, warmup=48)
    tr.save(path)
    return tr


# ------------------------------------------------------------------ layouts (vx.mosaic v1 caches them on disk)
def _flow(painter, extent, s, tile, fine_tile, seed, alpha=False, gold=True, ground=None, **kw):
    g = mt.guide(painter, extent, s)
    return mz.Layout.flow(g["rgb"], tile=tile, seed=seed, extent=extent, fine=g["fine"] > 0.5, fine_tile=fine_tile,
                          gold=(g["gold"] > 0.5) if gold else None, alpha=g["alpha"] if alpha else None,
                          ground=ground(g) if ground else None, **kw)


def _guide_state():
    return dict(t=70.0, rays=1.0, caption_n=99, caption_flash=None, caption_gold=0.0, donors=None,
                lamp_on=lambda k, i: 1.0, eye_open=1.0, eye_look=(0.0, 0.0), core_glow=1.0, portal=0.0)


def build_layouts(S):
    gs = _guide_state()
    L = {}
    L["wall"] = _flow(art.wall_painter(dict(gs, donors=donors_painter(76.0, None))), art.WH, 0.75, 12.0, 3.8, 3,
                      ground=lambda g: (g["gold"] > 0.5) & (g["fine"] < 0.5))
    L["stack"] = _flow(art.stack_painter(gs), art.WH, 0.75, 6.0, 4.0, 5, alpha=True)
    for k in range(3):
        ne = art.RINGS[k][2]
        L[f"ring{k}"] = _flow(art.ring_painter(k, [1.0] * ne, [(0.0, 0.0)] * ne), art.ring_extent(k), 1.0, 6.0, 3.6,
                              20 + k, alpha=True)
    for j, (name, side, off, a_open, a_fold, Lw, Wd) in enumerate(art.WINGS):
        k = art.IGN_ORDER.index(name)
        L[name] = _flow(art.wing_painter(Lw, Wd, side < 0, 1.0, 1.3, 70.0, "CANON " + art.ROMAN[k], 1.0),
                        art.wing_extent(Lw, Wd), 0.6, 8.0, 4.2, 40 + k, alpha=True, gold=False)
    return L


# ------------------------------------------------------------------ donors (vx.icon)
_FIG = {}


def figure(kind):
    f = _FIG.get(kind)
    if f is None:
        from vx import icon
        f = _FIG[kind] = icon.Figure(kind)
    return f


def donors_painter(T, tl):
    """Lutie and Crocus as small donor figures on the meadow, looking up at the Flag (lip-synced)"""
    def paint(P):
        for who, (x, y, h) in art.DONORS.items():
            spk = "LUTIE" if who == "lutie" else "CROCUS"
            m = tl.mouth(spk, T) if tl is not None else 0.0
            look = (0.35, -0.8) if who == "lutie" else (-0.35, -0.8)
            expr = "awe" if who == "lutie" else "serene"
            P.figure(figure(who), x, y, h, pose="look_up" if who == "lutie" else "stand", t=T, mouth=m, look=look,
                     expr=expr, facing=0.35 if who == "lutie" else -0.35, tile=3.8, fine_all=True)
    return paint

# ------------------------------------------------------------------ foley / music events
def export_all(S, st):
    ev = [dict(t=WAKE, kind="eye", strength=1.0, pan=0.0, desc="the great eye opens where m03's gold point was")]
    for k in (2, 1, 0):
        ne = art.RINGS[k][2]
        tb = RING_EYES_T[k]
        for i in range(ne):
            ev.append(dict(t=tb + i * 0.045 + 0.1 * hash01(i * 7 + k * 101, 3), kind="eye", strength=0.22,
                           pan=0.0, desc=f"ring {k} eye {i} half-opens"))
    for j, (name, side, off, a_open, a_fold, L, Wd) in enumerate(art.WINGS):
        t_un = _wing_unfold_t(j, side)
        ev.append(dict(t=t_un, kind="wing_unfold", strength=0.8 if name.startswith("mid") else 0.6,
                       pan=0.6 * side, desc=f"wing {name} unfolds (to {t_un + 1.4:.2f})"))
    for k, name in enumerate(art.IGN_ORDER):
        side = -1 if name.endswith("_l") else 1
        ev.append(dict(t=IGN[k], kind="ignite", strength=0.7 + 0.05 * k, pan=0.55 * side,
                       desc=f"CANON {art.ROMAN[k]}: wing {name} ignites root->tip (to {IGN[k] + 0.42:.2f})"))
    ev.append(dict(t=EVERY, kind="eye", strength=0.6, pan=0.0, desc="all ring eyes open fully in a wave (to 70.4)"))
    ev.append(dict(t=TRUE_, kind="eye", strength=0.7, pan=0.0, desc="'true': every eye turns to the camera"))
    # the wheels: one event per 0.5 s with the current speed (strength) while they turn
    T = TURN0
    while T < LOCK:
        ev.append(dict(t=T, kind="wheels", strength=round(ring_speed(st, T + 0.25), 3), pan=0.0,
                       desc=f"the wheels turn (to {min(LOCK, T + 0.5):.2f})"))
        T += 0.5
    ev.append(dict(t=LOCK, kind="lock", strength=1.0, pan=0.0,
                   desc="the wheels LOCK face-on with a boom; a shock ring of glints runs across the gold"))
    ev.append(dict(t=LOCK + 0.02, kind="eye", strength=0.5, pan=0.0, desc="all eyes blink at the lock"))
    ev.append(dict(t=EMERGE - 0.3, kind="portal", strength=0.6, pan=0.0, desc="the great pupil opens into light"))
    ev.append(dict(t=EMERGE, kind="descend", strength=0.8, pan=0.05,
                   desc=f"the Flag materialises out of the pupil and descends (to {HOVER:.2f})"))
    ev.append(dict(t=80.55, kind="eye", strength=0.4, pan=0.0, desc="all eyes blink in unison (deadpan)"))
    ev += st.get("disp_sfx", [])
    export_events(S, "sfx", ev)
    mus = [dict(t=WAKE, kind="eye", strength=1.0, desc="first eye opens")]
    for k in range(6):
        mus.append(dict(t=IGN[k], kind="wing", strength=0.7 + 0.05 * k, desc=f"CANON {art.ROMAN[k]} ignites"))
    mus += [dict(t=LOCK, kind="lock", strength=1.0, desc="wheels lock"),
            dict(t=EMERGE, kind="descend", strength=0.8, desc="Flag descent starts"),
            dict(t=HOVER, kind="descend", strength=0.4, desc="Flag descent lands (hovers)")]
    mus += st.get("disp_music", [])
    export_events(S, "music", mus)


# ------------------------------------------------------------------ setup
def setup(S):
    st = {}
    st["lay"] = build_layouts(S)
    st["ang"] = build_angles()
    st["flag"] = build_flag(S)
    st["flag"].export_foley(S, "flag")
    build_disp(S, st)
    export_all(S, st)
    return st


# ------------------------------------------------------------------ render: the Hypermind panel
def _mouth(spk, T):
    return get_timeline().mouth(spk, T)


def _lamp_on(T, k, i):
    t0 = 63.75 + 0.5 * k + i * 0.09
    return smoothstep(t0, t0 + 0.18, T)


def _lights_H(T, st, ws):
    """world_H lights: (x, y, z, rgb, intensity, radius)"""
    Ls = []
    wake = smoothstep(WAKE, WAKE + 0.6, T)
    voice = _mouth("HYPERMIND", T)
    if T < WAKE + 0.5:
        k = smoothstep(T0 - 0.05, WAKE, T) * (1.0 - smoothstep(WAKE + 0.1, WAKE + 0.5, T))
        Ls.append((art.CX, art.CY, 10.0, (1.0, 0.9, 0.7), 1.1 * k, 12.0))
    core = 0.9 * wake + 0.3 * voice + 0.35 * smoothstep(EVERY, TRUE_, T) + 1.0 * pulse(T, LOCK + 0.05, 0.12)
    Ls.append((art.CX, art.CY, 190.0, (1.0, 0.92, 0.8), core, 520.0))
    for k in range(3):
        y, rx, ry, nl = art.PLATES[k]
        on = sum(_lamp_on(T, k, i) for i in range(nl)) / max(1, nl)
        for fx in (-0.8, 0.0, 0.8):
            Ls.append((art.CX + fx * rx, y + 40, 130.0, (1.0, 0.8, 0.55), 0.32 * on, 300.0))
    for name, (ang, ign, front, u) in ws.items():
        if ign <= 0:
            continue
        spec = next(w for w in art.WINGS if w[0] == name)
        L = spec[5]
        a = math.radians(ang)
        d = 0.5 * L
        x = art.CX + spec[2][0] + math.cos(a) * d
        y = art.CY + spec[2][1] + math.sin(a) * d
        k = art.IGN_ORDER.index(name)
        flare = 1.0 + 1.4 * math.exp(-max(0.0, T - IGN[k]) * 3.0)
        Ls.append((x, y, 170.0, (1.0, 0.66, 0.42), 0.42 * ign * flare, 420.0))
    if LOCK <= T < LOCK + 0.9:
        r = 150 + (T - LOCK) * 1500.0
        k = (1 - (T - LOCK) / 0.9) ** 1.5
        for i in range(14):
            a = i * 2 * math.pi / 14
            Ls.append((art.CX + math.cos(a) * r, art.CY + math.sin(a) * r * 0.95, 70.0, (1.0, 0.9, 0.72),
                       1.3 * k, 170.0))
    if T > EMERGE:
        x, yb, _ = flag_pole(T)
        a = smoothstep(EMERGE, EMERGE + 0.6, T)
        Ls.append((x + 75, yb - POLE + 60, 150.0, (1.0, 0.94, 0.74), 1.25 * a, 400.0))
        # the Flag's light falls on the donors' upturned faces
        for who, (dx, dy, dh) in art.DONORS.items():
            Ls.append((dx + (60 if dx < art.CX else -60), dy - dh * 0.85, 190.0, (1.0, 0.93, 0.76),
                       0.55 * smoothstep(EMERGE + 0.5, HOVER, T), 300.0))
    return Ls


def _caption(T):
    """HIC · HYPERMENS · VEXILLVM · COMPILAT sets itself during H02 ('Hypermeme compiled')"""
    s = "HIC · HYPERMENS · VEXILLVM · COMPILAT"
    t0, t1 = 72.45, 74.15
    n = int(clamp((T - t0) / (t1 - t0)) * len(s) + 0.5) if T > t0 else 0

    def flash(k):
        tk = t0 + (k + 0.5) / len(s) * (t1 - t0)
        return clamp(1.0 - (T - tk) / 0.35) if T >= tk else 0.0
    return n, flash


AMB = 0.02            # ambient (vx.mosaic: x warm tint, linear)
EMIT_K = 2.2          # self-lit glass gain (eyes, flames, fire, rays)
CART_S = 0.8          # cartoon pixels per design unit (tiles are single-coloured: ~4 px per tile is plenty)
WALL_CLIP = (-600.0, -800.0, art.WH[0] + 600.0, art.WH[1] + 800.0)


def _cart_s(fc):
    return CART_S if fc.s >= 0.99 else 0.55


def render_wall(fc, lay, painter, M, Ls, amb=AMB, env=1.0):
    d = mt.arrays(painter, mt.screen_window(M, 30.0, WALL_CLIP), _cart_s(fc))
    return mz.render(fc, d["rgb"], lay, crect=d["crect"], view=M, light=mt.lights(Ls), ambient=amb,
                     emit=d["emit"] * EMIT_K, gold=d["gold"], env=env)


def render_layer(fc, lay, painter, crect, M, Ls_chart, gold=False, amb=AMB, s=None, env=1.0):
    """a layer in its own chart (crect of it painted) placed on screen by M -> premultiplied rgba"""
    d = mt.arrays(painter, crect, s or _cart_s(fc))
    return mz.render(fc, d["rgb"], lay, crect=d["crect"], view=M, light=mt.lights(Ls_chart), ambient=amb,
                     emit=d["emit"] * EMIT_K, gold=d["gold"] if gold else None, return_alpha=True, env=env)


def render_ring(fc, st, k, painter, R3, M, Ls, env=1.0):
    """wheel k: a flat disc of tiles oriented R3 about the great eye, projected orthographically.
    Returns (far, near) premultiplied rgba (the halves behind / in front of the axis)."""
    ex, ey = art.ring_extent(k)
    A = M[:, :2] @ R3[:2, :2]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-3 * mt.scale_of(M) ** 2:          # exactly edge-on: keep the affine invertible
        A = A + np.array([[0.0, 0.0], [0.0, 0.032 * mt.scale_of(M)]])
    cx, cy = mt.apply(M, art.CX, art.CY)
    b = np.array([cx, cy]) - A @ np.array([ex / 2, ey / 2])
    Mr = np.hstack([A, b[:, None]])
    Lr = mt.lights_to_disc(Ls, (art.CX, art.CY), R3, (ex / 2, ey / 2))
    rgba = render_layer(fc, st["lay"][f"ring{k}"], painter, (0.0, 0.0, ex, ey), Mr, Lr, s=1.0 if fc.s > 0.99 else 0.6,
                        env=env)
    # depth (world units, + toward the viewer) of every screen point on the disc plane, relative to its centre
    gv = R3[2, :2] @ np.linalg.inv(A)
    pts = np.array([mt.apply(Mr, u, v) for u, v in ((0, 0), (ex, 0), (0, ey), (ex, ey))]) * fc.s
    x0, y0 = np.clip(np.floor(pts.min(0)).astype(int) - 2, 0, None)
    x1 = min(fc.w, int(np.ceil(pts[:, 0].max())) + 2)
    y1 = min(fc.h, int(np.ceil(pts[:, 1].max())) + 2)
    far = np.zeros_like(rgba)
    if x1 > x0 and y1 > y0:
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dep = gv[0] * ((xx + 0.5) / fc.s - cx) + gv[1] * ((yy + 0.5) / fc.s - cy)
        wf = np.clip(0.5 - dep * 2.0, 0, 1)[..., None]
        far[y0:y1, x0:x1] = rgba[y0:y1, x0:x1] * wf
    near = rgba - far
    return far, near


def render_hyper(fc, st, T):
    cx, cy, z, rot = camera_H(T)
    M = mt.cam_matrix(cx, cy, z, rot)
    ws = wing_state(T)
    Ls = _lights_H(T, st, ws)
    lay = st["lay"]
    tl = get_timeline()
    env = 0.03 + 0.75 * smoothstep(WAKE, 65.5, T)

    # 1. the wall: gold ground, mandorla, rays, rim inscription, meadow + donors
    n_cap, fl_cap = _caption(T)
    wst = dict(t=T, rays=smoothstep(64.0, 68.5, T), caption_n=n_cap, caption_flash=fl_cap,
               donors=donors_painter(T, tl))
    img = render_wall(fc, lay["wall"], art.wall_painter(wst), M, Ls, env=env)

    # 2. wings (behind the wheels)
    if T > 63.6:
        for name in ("mid_r", "mid_l", "up_r", "up_l", "lo_r", "lo_l"):
            ang, ign, front, u = ws[name]
            spec = next(w for w in art.WINGS if w[0] == name)
            _, side, off_, _, _, Lw, Wd = spec
            k = art.IGN_ORDER.index(name)
            Mwc = wing_matrix(name, ang)
            ex, ey = art.wing_extent(Lw, Wd)
            pw = art.wing_painter(Lw, Wd, side < 0, ign, front, T, "CANON " + art.ROMAN[k], 1.0, seed=k)
            rgba = render_layer(fc, lay[name], pw, (0.0, 0.0, ex, ey), mt.compose(M, Mwc),
                                mt.lights_to_chart(Ls, Mwc), gold=True, s=0.6 if fc.s > 0.99 else 0.4, env=env)
            mt.over(img, rgba)

    # 3. wheels: far halves (outer first)
    Rk = ring_rots(st, T)
    halves = []
    heat = smoothstep(EVERY, TRUE_, T)
    for k in range(3):
        if T < 63.1:
            halves.append(None)
            continue
        opens, looks = eyes_state(T, k)
        if T > EMERGE - 0.2:
            fl = _look_at_flag(T, st, k)
            kk = smoothstep(EMERGE - 0.2, EMERGE + 0.4, T)
            looks = [(a[0] * (1 - kk) + b[0] * kk, a[1] * (1 - kk) + b[1] * kk) for a, b in zip(looks, fl)]
        pr = art.ring_painter(k, opens, looks, glow=0.25 + 0.3 * heat)
        halves.append(render_ring(fc, st, k, pr, Rk[k], M, Ls, env=env))
    for hv in halves:
        if hv is not None:
            mt.over(img, hv[0])

    # 4. the polycandelon stack + the great eye
    eo = smoothstep(WAKE, WAKE + 0.38, T)
    eo *= _blink(T, 999) if T < EMERGE - 0.4 else 1.0
    portal = smoothstep(EMERGE - 0.3, EMERGE + 0.1, T) * (1 - 0.5 * smoothstep(HOVER, HOVER + 1.5, T))
    look_e = (0.0, 0.9 * smoothstep(EMERGE, HOVER, T))
    sst = dict(t=T, lamp_on=lambda k, i: _lamp_on(T, k, i), eye_open=eo, eye_look=look_e,
               core_glow=smoothstep(WAKE, 64.5, T) * (0.75 + 0.25 * _mouth("HYPERMIND", T)), portal=portal)
    win = mt.screen_window(M, 20.0, (art.CX - 380, -80.0, art.CX + 380, art.CY + art.CORE_R + 30))
    if win[2] > win[0] and win[3] > win[1]:
        rgba = render_layer(fc, lay["stack"], art.stack_painter(sst), win, M, Ls, s=max(_cart_s(fc), 0.9), env=env)
        mt.over(img, rgba)

    # 5. wheels: near halves (inner first)
    for hv in halves[::-1]:
        if hv is not None:
            mt.over(img, hv[1])

    # 6. m03's gold point (hand-off) -> the great eye's catchlight
    if T < WAKE + 0.5:
        k = 1.0 - smoothstep(WAKE - 0.05, WAKE + 0.5, T)
        _gold_point(img, fc, 960.0, 540.0, k)
    return img, M


def _gold_point(img, fc, x, y, k):
    s = fc.s
    r = int(110 * s) + 2
    x0, y0 = int(round(x * s)) - r, int(round(y * s)) - r
    yy, xx = np.mgrid[0:2 * r, 0:2 * r].astype(np.float32)
    d = np.hypot(xx + 0.5 - r, yy + 0.5 - r) / s
    core = np.exp(-(d / 5.0) ** 2)[..., None] * np.array(art.hexc("#FFF6DA"), np.float32)
    halo = np.exp(-(d / 22.0) ** 2)[..., None] * np.array(art.S["gold_l"], np.float32) * 0.5
    halo2 = np.exp(-(d / 60.0) ** 2)[..., None] * np.array(art.S["gold"], np.float32) * 0.22
    spr = (core + halo + halo2) * k
    ya, xa = max(0, y0), max(0, x0)
    yb, xb = min(img.shape[0], y0 + 2 * r), min(img.shape[1], x0 + 2 * r)
    img[ya:yb, xa:xb] += spr[ya - y0:yb - y0, xa - x0:xb - x0]


def draw_flag(fc, st, img, T, Mw):
    """the Flag, LAST, whole cloth, drawn through the world->screen transform Mw"""
    if T < EMERGE:
        return img
    tr = st["flag"]
    i = min(tr.n - 1, max(0, fc.f))
    cv = Canvas(fc)
    ctx = cv.ctx
    ctx.save()
    ctx.transform(cairo.Matrix(Mw[0, 0], Mw[1, 0], Mw[0, 1], Mw[1, 1], Mw[0, 2], Mw[1, 2]))
    # THE ONE RULE: the cloth is always fully opaque (a translucent flag would show tiles through it).
    # It is revealed by a disc of light growing out of the pupil, so it passes through the portal whole.
    r = smoothstep(EMERGE, EMERGE + 0.55, T)
    if r < 1.0:
        ctx.arc(art.CX, art.CY, 24.0 + (POLE + 260.0) * ease_in_out(r, 2.0), 0.0, 2.0 * math.pi)
        ctx.clip()
    glow = 0.2 + 0.8 * (1.0 - smoothstep(EMERGE + 0.2, HOVER, T))
    tr.draw(ctx, i, alpha=1.0, glow=glow, light=(-0.35, -0.75, 0.5))
    ctx.restore()
    return cv.over(img)


# ================================================================== part II: the disputation (m04_disp.py)
def M_H81():
    return mt.cam_matrix(F81[0], F81[1], F81[2], 0.0)


def flag_to_D():
    """world_H -> world_D chart (the screen at 81.2, shifted by (OX, OY))"""
    return mt.compose(mt.affine(1, 0, 0, 1, dsp.OX, dsp.OY), M_H81())


def build_disp(S, st):
    tl = S.tl
    times = dsp.letter_times(tl)
    gtimes = dsp.glyph_times(times)
    guide_times = {k: [0.0] * len(v) for k, v in times.items()}
    gst = dict(T=95.0, times=guide_times, figs=dsp.figures_painter(90.0, tl, figure, guide=True),
               icon=dsp.icon_painter(90.0))
    s = 0.6
    g = mt.guide(dsp.wall_painter(gst), dsp.WD, s)
    # glyph labels -> lay.grp
    lw, lh = g["rgb"].shape[1], g["rgb"].shape[0]
    sf = cairo.ImageSurface(cairo.FORMAT_ARGB32, lw, lh)
    ctx = cairo.Context(sf)
    ctx.scale(s, s)
    ctx.translate(dsp.OX, dsp.OY)
    dsp.draw_labels(ctx)
    sf.flush()
    buf = np.ndarray((lh, sf.get_stride() // 4, 4), np.uint8, sf.get_data())[:, :lw]
    labels = buf[..., 2].astype(np.int32) - 1
    ground = (g["gold"] > 0.5) & (g["fine"] < 0.5)
    lay = mz.Layout.flow(g["rgb"], tile=10.0, seed=7, extent=dsp.WD, fine=g["fine"] > 0.5, fine_tile=4.4,
                         gold=(g["gold"] > 0.5) & (labels < 0), ground=ground, groups=labels)
    n = len(lay)
    D = dict(lay=lay, times=times)
    grp = lay.grp if hasattr(lay, "grp") else np.full(n, -1)
    D["t_letter"] = np.where(grp >= 0, gtimes[np.clip(grp, 0, len(gtimes) - 1)], -np.inf)
    sx, sy = lay.xy[:, 0] - dsp.OX, lay.xy[:, 1] - dsp.OY
    D["frag"] = dsp.frag_of(sx, sy)
    D["screen"] = (D["frag"] >= 0).astype(np.float32)
    ox, oy, R = dsp.ORB_R
    D["gold_orb"] = np.hypot(sx - ox, sy - oy) <= R + 4
    D["rise_t"] = 91.35 + (oy + R - sy) / 420.0
    fx0, fy0 = mt.apply(flag_to_D(), art.CX + 70.0, HOVER_TOP + 60.0)
    D["t_wave"] = mz.fx.wave_times(lay, (fx0, fy0), dsp.FLIP_SPEED, t0=dsp.FLIP0, jitter=0.02, seed=4)
    # the smiling slop world: a layer that cracks on "must" and falls on "fall"
    ls = dsp.smile_layout()
    so = dsp.smile_origin()
    D["smile_wave"] = mz.fx.wave_times(ls.xy + np.array(so, np.float32), (fx0, fy0), dsp.FLIP_SPEED, t0=dsp.FLIP0,
                                       jitter=0.02, seed=5)
    Rr = dsp.ORB_R[2]
    c0 = Rr + dsp.SMILE_PAD
    impact = (c0 + Rr * 0.4, c0 - Rr * 0.48)
    t_must = tl.word("C02", "must")[0]
    t_fall = tl.word("C02", "fall")[0]
    D["smile_lay"] = ls
    D["crack"] = mz.fx.Crack(ls, start=impact, direction=2.3, length=430.0, t0=t_must, speed=650.0, open=3.0,
                             band=22.0, branch=0.4, seed=4, max_branches=7)
    floor_local = dsp.FLOOR_Y + dsp.OY - dsp.smile_origin()[1]
    D["fall"] = mz.fx.Fall(ls, t_detach=mz.fx.wave_times(ls, impact, 380.0, t0=t_fall + 0.05, jitter=0.05, seed=3),
                           floor=floor_local, g=2600.0, bounce=0.3, seed=2, depth=(20.0, 160.0))
    D["t_fall"] = t_fall
    st["D"] = D
    ev = [dict(t=dsp.FLIP0, kind="flip_wave", strength=0.9, pan=0.0,
               desc="videtur: the whole wall re-sets around the Flag in a split-flap wave (to 81.85)")]
    for key, word in (("v1", "VIDETVR"), ("v2", "QVOD NON"), ("pr", "PRAETEREA"), ("s1", "SED"), ("s2", "CONTRA"),
                      ("r", "RESPONDEO DICENDVM")):
        ts = times[key]
        x = dsp.TIT[key].x
        ev.append(dict(t=ts[0], kind="tile_set", strength=0.6 if key != "pr" else 0.35,
                       pan=float(np.clip((x - 960) / 960, -1, 1)),
                       desc=f"titulus {word} sets itself letter by letter (to {ts[-1] + dsp.FLIP_DUR:.2f})"))
    ev += D["crack"].events(T0=0.0, desc="the too-perfect smiling slop-world cracks like a phone screen")
    ev.append(dict(t=t_fall, kind="fall", strength=0.85, pan=0.35,
                   desc=f"the smiling slop-world falls away tile by tile (to {t_fall + 1.3:.2f})"))
    ev += D["fall"].events(T0=0.0, desc="fallen slop tiles hit the floor", kind="tiles")
    t_rise = tl.word("C02", "rise")[0]
    ev.append(dict(t=t_rise, kind="glint", strength=0.7, pan=0.35, desc="'rise': the true world in gold catches the light"))
    t_sum = tl.word("C03", "summon")[0]
    ev.append(dict(t=t_sum - 0.25, kind="rustle", strength=0.6, pan=0.6,
                   desc="Crocus raises the Byzantine blessing hand (to %.2f)" % (t_sum + 0.2)))
    ev.append(dict(t=95.4, kind="flash", strength=0.8, pan=0.0, desc="gold flash into the hard cut (to 95.75)"))
    st["disp_sfx"] = ev
    st["disp_music"] = [dict(t=dsp.FLIP0, kind="flip_wave", strength=0.9, desc="wall re-sets (videtur)"),
                        dict(t=t_must, kind="crack", strength=0.7, desc="the smiling world cracks"),
                        dict(t=t_fall, kind="crack", strength=1.0, desc="the smiling world falls"),
                        dict(t=t_rise, kind="glint", strength=0.8, desc="gold glint on 'rise'"),
                        dict(t=dsp.RESP, kind="bless", strength=0.6, desc="RESPONDEO: Crocus lifts the blessing hand"),
                        dict(t=t_sum - 0.25, kind="bless", strength=1.0, desc="'summon': the blessing hand raised")]


def camera_D(T):
    """(cx, cy, zoom) over the disputation chart"""
    OX, OY = dsp.OX, dsp.OY
    keys = [(81.9, (960, 540, 1.0)), (86.4, (700, 610, 1.18)), (87.35, (1250, 612, 1.14)),
            (92.3, (1296, 640, 1.24)), (93.3, (960, 560, 1.0))]
    cx, cy, z = keys[0][1]
    for (t0, a), (t1, b) in zip(keys[:-1], keys[1:]):
        if T >= t0:
            e = seg(T, t0, t1, ease_in_out_sine)
            cx, cy, z = lerp(a[0], b[0], e), lerp(a[1], b[1], e), lerp(a[2], b[2], e)
    cx, cy = cx + OX, cy + OY
    hw, hh = W / 2 / z + 2.0, H / 2 / z + 2.0
    cx = min(max(cx, hw), dsp.WD[0] - hw)
    cy = min(max(cy, hh), dsp.WD[1] - hh)
    return cx, cy, z


T_PUSH = 93.95


def camera_D_final(T):
    """the last push: Crocus's blessing hand glides to screen (960, 380) as the zoom grows to 2.3"""
    cx, cy, z = camera_D(T)
    if T < T_PUSH:
        return cx, cy, z
    hx, hy = dsp.crocus_hand(T, figure)
    e = ease_in_out(clamp((T - T_PUSH) / (dsp.END - 1.0 / 24 - T_PUSH)), 2.6)
    z1 = lerp(z, 2.3, e)
    sx0 = 960 + (hx - cx) * z
    sy0 = 540 + (hy - cy) * z
    sx, sy = lerp(sx0, 960.0, e), lerp(sy0, 380.0, e)
    return hx - (sx - 960) / z1, hy - (sy - 540) / z1, z1


def _lights_D(T, st, flag_xy):
    OX, OY = dsp.OX, dsp.OY
    Ls = [(400 + OX, 140 + OY, 800.0, (1.0, 0.88, 0.7), 0.5, 1600.0),
          (960 + OX, 262 + OY, 120.0, (1.0, 0.9, 0.75), 0.45, 320.0),
          (flag_xy[0], flag_xy[1], 150.0, (1.0, 0.94, 0.74), 1.0, 460.0)]
    tr_ = get_timeline().word("C02", "rise")[0]
    if T > tr_ - 0.3:
        ox, oy, R = dsp.ORB_R
        u = clamp((T - (tr_ - 0.3)) / 1.1)
        Ls.append((ox + OX, oy + R - 2.2 * R * u + OY, 110.0, (1.0, 0.9, 0.7), 1.6 * math.sin(math.pi * u) + 0.4,
                   260.0))
    if T > T_PUSH:
        hx, hy = dsp.crocus_hand(T, figure)
        k = smoothstep(T_PUSH, dsp.END, T)
        Ls.append((hx, hy, 90.0, (1.0, 0.88, 0.62), 2.5 * k * k, 380.0))
    lights = mt.lights(Ls)
    lights += mz.candles(T, [(300 + OX, 880 + OY, 240.0), (1560 + OX, 880 + OY, 240.0)], power=0.8, radius=560.0)
    return lights


_A_CACHE = {}


def frozen_A(fc, st):
    """the Hypermind panel at the start of the re-set (the wave flips it away): computed once per process"""
    key = (fc.w, fc.h)
    if key not in _A_CACHE:
        img, M = render_hyper(fc, st, dsp.FLIP0)
        _A_CACHE.clear()
        _A_CACHE[key] = np.clip(img, 0, 1)
    return _A_CACHE[key]


def render_disp(fc, st, T):
    D = st["D"]
    lay = D["lay"]
    tl = get_timeline()
    cx, cy, z = camera_D_final(T)
    M = mt.cam_matrix(cx, cy, z)
    Mf = mt.compose(M, flag_to_D())
    fi = min(st["flag"].n - 1, max(0, fc.f))
    fxy = mt.apply(flag_to_D(), *st["flag"].cloth_center(fi))
    lights = _lights_D(T, st, fxy)
    wst = dict(T=T, times=D["times"], figs=dsp.figures_painter(T, tl, figure), icon=dsp.icon_painter(T))
    d = mt.arrays(dsp.wall_painter(wst), mt.screen_window(M, 30.0, (0.0, 0.0) + dsp.WD), _cart_s(fc))
    n = len(lay)
    # per-tile motion: the shattered fragments drift / jolt; letters flip in; the gold world glints on 'rise'
    off = np.zeros((n, 2), np.float32)
    fo = dsp.frag_offsets(T, tl)
    m = D["frag"] >= 0
    off[m] = fo[D["frag"][m]]
    gain = np.ones(n, np.float32)
    gain[D["gold_orb"]] += 2.2 * np.exp(-((T - D["rise_t"][D["gold_orb"]]) / 0.16) ** 2)
    fl_ = mz.fx.flip(T, D["t_letter"], dsp.FLIP_DUR, size=lay.size)
    poses = [dict(rot=fl_["rot"], lift=fl_["lift"])]
    kw = {}
    rgb = None
    if T < dsp.FLIP0 + 1.4:
        A = frozen_A(fc, st)
        fw = mz.fx.flip(T, D["t_wave"], dsp.FLIP_DUR, size=lay.size)
        poses.append(dict(rot=fw["rot"], lift=fw["lift"]))
        B = mz.sample(d["rgb"], lay, crect=d["crect"])
        sx, sy = mt.apply(M, lay.xy[:, 0], lay.xy[:, 1])
        ix = np.clip((np.asarray(sx) * fc.s).astype(int), 0, fc.w - 1)
        iy = np.clip((np.asarray(sy) * fc.s).astype(int), 0, fc.h - 1)
        Acol = A[iy, ix]
        rgb = np.where(fw["new"][:, None], B, Acol).astype(np.float32)
        kw = dict(present=T >= D["t_wave"], bg=A, holes="bg")
    pose = mz.fx.combine(*poses)
    img = mz.render(fc, d["rgb"], lay, crect=d["crect"], view=M, light=lights, ambient=0.05,
                    emit=d["emit"] * EMIT_K, gold=d["gold"], screen=D["screen"], offset=off, gain=gain, rgb=rgb,
                    env=0.8, **pose, **kw)
    # the smiling slop-world: cracks on "must", falls on "fall"
    t_fall = D["t_fall"]
    if T >= dsp.FLIP0:
        crack = D["crack"]
        ox, oy = dsp.smile_origin()
        Ms = mt.compose(M, mt.affine(1, 0, 0, 1, ox, oy))
        ex, ey = dsp.smile_extent()
        smile = 1.0 + 0.6 * smoothstep(88.3, 89.3, T)
        ds = mt.arrays(dsp.smile_painter(T, crack.paths(T), smile), (0.0, 0.0, ex, ey), 1.0 if fc.s > 0.99 else 0.6)
        poses_s = [crack.pose(T), D["fall"].pose(T)]
        present = None
        if T < dsp.FLIP0 + 1.4:
            # the smiling world is part of the new picture: its tiles flip in when the wave reaches them
            fw = mz.fx.flip(T, D["smile_wave"], dsp.FLIP_DUR, size=D["smile_lay"].size)
            poses_s.append(dict(rot=fw["rot"], lift=fw["lift"]))
            present = fw["new"]
        sp = mz.fx.combine(*poses_s)
        if present is not None:
            sp["present"] = present
        Lsm = [mz.Light(pos=(l.p[0] - ox, l.p[1] - oy, l.p[2]), color=l.color, power=l.power, radius=l.radius)
               for l in lights if not l.directional]
        dying = 1.0 - 0.85 * smoothstep(t_fall + 0.2, t_fall + 1.4, T)
        rgba = mz.render(fc, ds["rgb"], D["smile_lay"], crect=ds["crect"], view=Ms, light=Lsm, ambient=0.06,
                         emit=ds["emit"] * (EMIT_K * dying), return_alpha=True, env=0.8, holes="bg", **sp)
        mt.over(img, rgba)
    # the last 0.35 s: a gold flash on the blessing hand, peaking on the final frame (m05 opens on a gold glint);
    # it lights the mosaic, then the Flag is composited over it (nothing is drawn over the cloth)
    if T > 95.38:
        k = smoothstep(95.38, dsp.END - 1.0 / 24, T)
        hx, hy = mt.apply(M, *dsp.crocus_hand(T, figure))
        _flash(img, fc, hx, hy, k)
    img = draw_flag(fc, st, img, T, Mf)
    return img


def _flash(img, fc, x, y, k):
    """a gold glint on the blessing hand: a white-gold core with four long rays and a warm bloom (never flag
    yellow: the colours stay pale and the frame is lifted, not clipped into saturation)"""
    s = fc.s
    yy, xx = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
    dx = (xx + 0.5) / s - x
    dy = (yy + 0.5) / s - y
    d = np.hypot(dx, dy)
    core = np.exp(-(d / (14 + 70 * k)) ** 2)
    rays = (np.exp(-(dy / (2.5 + 5 * k)) ** 2) * np.exp(-(np.abs(dx) / (120 + 700 * k)))
            + np.exp(-(dx / (2.5 + 5 * k)) ** 2) * np.exp(-(np.abs(dy) / (90 + 420 * k))))
    bloom = np.exp(-(d / (260 + 700 * k)) ** 2)
    wc = np.array((1.0, 0.98, 0.93), np.float32)
    gc = np.array((0.98, 0.86, 0.62), np.float32)
    lum = img.mean(axis=2, keepdims=True)
    img += (core * 1.4 * k)[..., None] * wc + (rays * 0.9 * k)[..., None] * wc + (bloom * 0.45 * k * k)[..., None] * gc
    img += (lum * 0.3 * k * k) * gc


def render(fc, st):
    T = fc.T
    if T < dsp.FLIP0:
        img, M = render_hyper(fc, st, T)
        img = draw_flag(fc, st, img, T, M)
    else:
        img = render_disp(fc, st, T)
    return np.clip(img, 0, 1.5)

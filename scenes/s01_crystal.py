"""s01 — Rotating Into Existence (0.0–21.0).

THE CRYSTAL is the 24-cell, turned with true 4D rotations (XW/YW/ZW planes) and centrally projected
4D -> 3D -> 2D (scenes/s01_geom.py). A 2-plane slices it; the exact cross-section polygon morphs as the
Crystal rotates and settles — mathematically — into the square great-section, which becomes the plain yellow
Flag. A pole draws itself in light and turns to wood; at 14.2 the cloth becomes a real cloth sim and snaps
open; the Flag is reflected endlessly in the Crystal's mirror facets (true planar reflections, two bounces).
16.0 the IVG title page slams down (scenes/s01_title.py); 20.3 it slides up, revealing flat concrete_l.
"""
import math

import cairo
import cv2
import numpy as np

from vx import *
from vx import foley
from vx.flag import Flag, draw_cloth
from scenes.s01_geom import V24, E24, T24, N24, D24, rot4, rot3, Proj, section, resample, hull2, ORDER
from scenes import s01_title as title

POST = dict(bloom=0.0, grain=0.032, vignette=0.30, ca=0.0)

# ------------------------------------------------------------------ timing (global seconds; s01 starts at 0)
TL = get_timeline()
T_N01 = TL.line("N01")["start"]            # 2.6  crystal condenses
T_N02 = TL.line("N02")["start"]            # 5.9  rotation in W accelerates
T_N03 = TL.line("N03")["start"]            # 10.3
T_ROT = TL.cue("flag_rotates_in")          # 12.0 the decisive 4D turn
T_ROTATED = TL.word("N03", "rotated")      # (12.06, 12.58)
T_SNAP = TL.cue("flag_snap_open")          # 14.2
T_SLAM = TL.cue("title_slam")              # 16.0
T_END = TL.scene("s01")["end"]             # 21.0
# glass notes of the Composer's Flag theme (12.00 12.49 12.73 13.22 13.47 13.96 -> bloom 14.20)
T_SETTLE = 12.49                           # section lands on the square
T_RECT = 12.73                             # square -> Flag rectangle complete
T_POLE0, T_POLE1 = 13.22, 13.47            # pole of light draws; then becomes wood
T_WOOD1 = 13.80
T_CLOTH = 13.96                            # the cloth becomes real cloth (sim starts)
T_SUCK = 15.30                             # reverse-suck into the title slam
T_DROP = T_SLAM - 0.26                     # page starts falling
T_SLIDE0, T_SLIDE1 = 20.30, 20.90          # page slides up (fully gone by frame 502 = 20.917)

CX, CY = 960.0, 520.0
D4 = 2.6                                   # 4D eye distance (strong nesting = the inside-out look)
UNIT = 200.0                               # design px per 4D unit at the settle (square side = 400 px)
CLOTH_W, CLOTH_H, POLE_L = 480.0, 400.0, 1280.0                # canonical 36"x30" on 96"
FLAG_X0, FLAG_Y0 = CX - CLOTH_W / 2, CY - CLOTH_H / 2          # pole top = cloth top-left (hoist)
POLE_BASE = (FLAG_X0, FLAG_Y0 + POLE_L)

GOLD = (1.0, 0.76, 0.36)
WHITE_GOLD = (1.0, 0.93, 0.74)
NEAR_C = np.array([1.0, 0.84, 0.52])
FAR_C = np.array([0.40, 0.48, 1.0])
MID_C = np.array([0.86, 0.86, 0.94])


def depth_col(d):
    """far (cool violet-blue) -> mid (silver) -> near (warm gold): never through pink"""
    if d < 0.5:
        return FAR_C + (MID_C - FAR_C) * (d / 0.5)
    return MID_C + (NEAR_C - MID_C) * ((d - 0.5) / 0.5)

_CACHE = {}


def _cell_edges(ci):
    vs = set(np.nonzero(np.abs(V24 @ N24[ci] - D24[ci]) < 1e-9)[0])
    return np.array([a in vs and b in vs for a, b in E24])


# two tracked cells (centred on +X and -X): as the Crystal turns in XW they swap inside <-> outside through
# each other; lit warm and cool so the eye can follow the impossible turn
EDGE_TRACK = np.zeros(96, int)
EDGE_TRACK[_cell_edges(1)] = 1
EDGE_TRACK[_cell_edges(0)] = 2
TRACK_C = {1: np.array([1.0, 0.66, 0.22]), 2: np.array([0.45, 0.80, 1.0])}


# ------------------------------------------------------------------ 4D choreography
def _speeds(t):
    """angular velocities (rad/s) per rotation plane before the decisive turn. The W planes dominate: those
    are the turns no 3D object can make (the Crystal passes inside-out through itself); the 3D planes are
    kept to a slow drift so the eye is not fooled by ordinary tumbling."""
    acc = smoothstep(T_N02, T_N02 + 2.6, t) * (1 - smoothstep(9.3, 10.9, t))
    zacc = smoothstep(T_N02 + 0.8, T_N02 + 3.1, t) * (1 - smoothstep(9.4, 11.0, t))
    return {"xw": 0.20 + 1.10 * acc, "yw": 0.06 + 0.30 * acc, "zw": 0.03 + 0.45 * zacc,
            "xy": 0.03, "yz": 0.05 + 0.05 * acc, "xz": 0.015}


# where each plane's raw angle should sit (mod pi/2) at T_ROT, so the decisive turn has a chosen size
_TURN = {"xw": math.pi / 2 + 0.22, "yw": 0.55, "zw": 0.85, "xy": 0.30, "yz": 0.42, "xz": 0.25}


def _turn_ease(p):
    p = clamp(p)
    return ease_out_back(ease_in(p, 1.5), 1.25)


def _build_angles():
    """integrate the speed profiles; offset each plane so that at T_ROT its angle sits exactly _TURN short of a
    multiple of pi/2 (the target) — the decisive turn then lands the Crystal on a symmetry of the 24-cell"""
    dt = 1 / 480
    ts = np.arange(0, T_SETTLE + dt, dt)
    A = {k: np.zeros(len(ts)) for k in ORDER}
    for i in range(1, len(ts)):
        sp = _speeds(ts[i - 1])
        for k in ORDER:
            A[k][i] = A[k][i - 1] + sp[k] * dt
    q = math.pi / 2
    off, target = {}, {}
    for k in ORDER:
        a_rot = float(np.interp(T_ROT, ts, A[k]))
        off[k] = (-(a_rot + _TURN[k])) % q
        target[k] = round((a_rot + off[k] + _TURN[k]) / q) * q
    return ts, A, off, target


def rot_at(st, t):
    """the Crystal's 4D orientation R(t) (points p = R v)"""
    ts, A, off, target = st["ang"]
    if t < T_ROT:
        return rot4({k: float(np.interp(t, ts, A[k])) + off[k] for k in ORDER})
    if t < T_SETTLE:
        f = _turn_ease((t - T_ROT) / (T_SETTLE - T_ROT))
        out = {}
        for k in ORDER:
            raw = float(np.interp(t, ts, A[k])) + off[k]
            end_raw = float(np.interp(T_SETTLE, ts, A[k])) + off[k]
            out[k] = raw + (target[k] - end_raw) * f
        return rot4(out)
    # canonical pose (a symmetry of the 24-cell), then turning in the world ZW plane (inside-out breathing) and
    # XY plane (spin): both keep the projection's 4-fold mirror symmetry, which the kaleidoscope of reflections
    # shares. The gust kicks the ZW turn.
    r = max(0.0, t - T_RECT)
    ramp = r - (1 - math.exp(-r / 0.8)) * 0.8       # integral of (1 - e^-r/0.8)
    kick = 1.1 * seg(t, T_SNAP, T_SNAP + 1.5, lambda x: ease_out(x, 3))
    return rot4({"zw": 0.30 * ramp + kick, "xy": spin_at(t)}) @ rot4(target)


def spin_at(t):
    r = max(0.0, t - T_RECT)
    return 0.07 * (r - (1 - math.exp(-r / 0.8)) * 0.8)


def view_at(t):
    """3D camera orbit (yaw, pitch) around the projected Crystal"""
    orbit = 0.40 * seg(t, T_N02, 10.3, ease_in_out_sine) + 0.30 * seg(t, 10.2, 11.2, ease_in_out_sine)
    pitch = 0.07 * math.sin(t * 0.23) + 0.30 * seg(t, 10.2, 11.2, ease_in_out_sine)
    yaw = 0.06 * math.sin(t * 0.17 + 0.4) + orbit
    if t >= T_ROT:
        k = 1 - _turn_ease((t - T_ROT) / (T_SETTLE - T_ROT)) if t < T_SETTLE else 0.0
        yaw, pitch = yaw * k, pitch * k
    return yaw, pitch


def unit_at(t):
    grow = 0.45 * seg(t, T_RECT + 0.05, T_CLOTH, ease_in_out) + 0.25 * seg(t, T_SNAP, T_SNAP + 0.7, ease_out)
    return UNIT * (1 + grow) * (1 + 0.12 * seg(t, T_SUCK, T_SLAM, lambda x: ease_in(x, 3)))


def cam_at(t):
    """2D camera (zoom about the Crystal); exactly 1.0 from the settle so the Flag lands at its design size"""
    z = 1.05 + 0.06 * seg(t, 1.0, T_N02, ease_in_out_sine) + 0.16 * seg(t, T_N02, 9.0, ease_in_out_sine)
    z -= 0.15 * seg(t, 9.4, 11.0, ease_in_out_sine) + 0.12 * seg(t, T_ROT, T_SETTLE, ease_in_out_sine)
    z += 0.03 * seg(t, T_SNAP, T_SUCK, ease_in_out_sine) + 0.20 * seg(t, T_SUCK, T_SLAM, lambda x: ease_in(x, 2.5))
    x = CX + 14 * math.sin(t * 0.21)
    y = CY + 10 * math.sin(t * 0.17 + 1.0)
    if t >= T_SETTLE:
        x, y = CX + (x - CX) * 0.3, CY + (y - CY) * 0.3
    dx, dy = 0.0, 0.0
    if T_SNAP <= t < T_SNAP + 0.8:
        a = 7 * math.exp(-(t - T_SNAP) * 6)
        dx, dy = shake(t, a, 13, 5)
    return Camera(x + dx, y + dy, z)


def growth(st, t):
    """per-vertex condensation (0 -> 1, overshoot), per-edge draw fraction, global crystal intensity"""
    ap = st["v_appear"]
    g = np.array([ease_out_back(clamp((t - a) / 1.1), 1.4) for a in ap])
    e_start = np.maximum(ap[E24[:, 0]], ap[E24[:, 1]]) + 0.15
    frac = np.clip((t - e_start) / 0.55, 0, 1)
    frac = frac * frac * (3 - 2 * frac)
    return g, frac


def intensity(t):
    on = seg(t, T_N01 + 0.15, T_N01 + 2.0, ease_in_out)
    dim = 1 - 0.30 * seg(t, T_SETTLE, T_RECT + 0.3)            # steps back to let the Flag lead
    dim += 0.12 * pulse(t, T_SNAP + 0.05, 0.18)                  # the gust flash
    return on * dim


# ------------------------------------------------------------------ setup
def setup(S):
    rng = np.random.default_rng(101)
    st = {"ang": _build_angles()}
    # vertices appear in a wave ordered by their canonical W then angle -> the Crystal unfolds from its core
    order = np.argsort(V24[:, 3] * 3 + np.arctan2(V24[:, 1], V24[:, 0]) * 0.2 + rng.uniform(0, 0.4, 24))
    ap = np.zeros(24)
    ap[order] = T_N01 + 0.35 + np.linspace(0, 1.75, 24) ** 1.15
    st["v_appear"] = ap
    # stars
    n = 1500
    st["stars"] = dict(x=rng.uniform(-200, W + 200, n), y=rng.uniform(-150, H + 150, n),
                       b=rng.uniform(0, 1, n) ** 5.5, d=rng.uniform(0.15, 1.0, n),
                       t_on=0.15 + 2.3 * rng.uniform(0, 1, n) ** 0.8, ph=rng.uniform(0, 6.3, n),
                       w=rng.uniform(0.6, 2.4, n), c=rng.integers(0, 3, n))
    # golden dust motes in a parallax volume
    m = 320
    st["dust"] = dict(x=rng.uniform(-1300, 1300, m), y=rng.uniform(-800, 800, m), z=rng.uniform(0.3, 2.6, m) ** 1.4,
                      vx=rng.normal(4, 7, m), vy=rng.normal(-9, 5, m), ph=rng.uniform(0, 6.3, m),
                      r=rng.uniform(0.7, 1.6, m), t_on=rng.uniform(0.0, 2.0, m))
    st["flag"] = _build_flag(S)
    st["hits"] = _export_foley(S, st)
    return st


# ------------------------------------------------------------------ the Flag
FLAG_LIGHT = (-0.35, -0.6, 0.72)


def _build_flag(S):
    """real cloth sim starting (warmup=0) from the flat, fully extended rectangle the section became"""
    f_c = int(round(T_CLOTH * S.fps))
    f_end = int(round(T_SLAM * S.fps))
    n = f_end - f_c + 1
    fl = Flag(width=CLOTH_W, height=CLOTH_H, pole=POLE_L, seed=11)

    def pole_fn(i):
        return (POLE_BASE[0], POLE_BASE[1], 0.0)

    def wind_fn(i):
        t = T_CLOTH + i / S.fps
        if t < T_SNAP:
            return (30.0, 0.0)
        g = 1350 * math.exp(-(t - T_SNAP) / 0.40) + 560 + 90 * math.sin((t - T_SNAP) * 2.3)
        return (g, -30.0)

    tr = fl.simulate(n, pole_fn, wind_fn, fps=S.fps, warmup=0)
    return dict(track=tr, f0=f_c)


def draw_hero(ctx, st, t, fcf, wood):
    """the hero Flag. Before the sim starts it is the geometric section rectangle, already cloth-shaded by
    vx.flag, with a faint living shimmer that dies away exactly as the real cloth takes over (frame 0 = flat)"""
    tr, i = flag_frame(st, fcf)
    glow = 0.25 + 0.55 * (1 - seg(t, T_RECT, T_CLOTH + 0.4))
    if t < T_CLOTH:
        V = tr.verts[0].astype(np.float64).copy()
        ny, nx = V.shape[:2]
        u = np.linspace(0, 1, nx)[None, :]
        v = np.linspace(0, 1, ny)[:, None]
        amp = 10.0 * seg(t, T_RECT - 0.1, T_RECT + 0.5) * (1 - seg(t, T_CLOTH - 0.5, T_CLOTH, ease_in_out))
        tau = t - T_RECT
        V[..., 2] += amp * u * (0.55 * np.sin(np.pi * u) + 0.45 * np.sin(2 * np.pi * (1.1 * u - 0.7 * tau) + 1.6 * v))
        draw_cloth(ctx, V, glow=glow, light=FLAG_LIGHT)
    else:
        tr.draw(ctx, i, pole=False, cloth=True, light=FLAG_LIGHT, glow=glow)
    _draw_partial_wood(ctx, tr, i, wood)


def flag_frame(st, fc_f):
    """(track, index) of the Flag at local frame fc_f (before the sim starts: its flat frame 0)"""
    fd = st["flag"]
    i = fc_f - fd["f0"]
    return fd["track"], int(min(max(i, 0), fd["track"].n - 1))


# ------------------------------------------------------------------ drawing: void
def draw_stars(ctx, st, t, yaw, pitch, lit):
    s = st["stars"]
    tw = 0.72 + 0.28 * np.sin(t * s["w"] * 2.2 + s["ph"])
    a = np.clip((t - s["t_on"]) / 0.9, 0, 1) * (0.25 + 0.75 * s["b"]) * tw
    a *= 1 - 0.35 * lit
    xs = s["x"] - yaw * 160 * s["d"]
    ys = s["y"] + pitch * 120 * s["d"]
    cols = [(0.86, 0.9, 1.0), (0.7, 0.8, 1.0), (1.0, 0.9, 0.78)]
    for k in range(3):
        m = (s["c"] == k) & (a > 0.01)
        c = cols[k]
        for x, y, al, b in zip(xs[m], ys[m], a[m], s["b"][m]):
            r = 0.55 + 1.5 * b
            ctx.set_source_rgba(*c, al)
            ctx.arc(x, y, r, 0, 6.2832)
            ctx.fill()
            if b > 0.8:
                ctx.set_source_rgba(*c, al * 0.22)
                ctx.set_line_width(0.6)
                L = 3 + 9 * b
                ctx.move_to(x - L, y)
                ctx.line_to(x + L, y)
                ctx.move_to(x, y - L)
                ctx.line_to(x, y + L)
                ctx.stroke()


def dust_pos(st, t):
    d = st["dust"]
    gust = 0.0
    if t > T_SNAP:
        tau = t - T_SNAP
        gust = 1500 * (1 - math.exp(-tau * 1.6)) + 260 * tau
    x = d["x"] + d["vx"] * t + gust * (0.5 + 0.5 * d["z"]) + 18 * np.sin(t * 0.3 + d["ph"])
    y = d["y"] + d["vy"] * t + 14 * np.cos(t * 0.25 + d["ph"] * 1.3) - (gust * 0.12 if t > T_SNAP else 0)
    x = (x + 1300) % 2600 - 1300
    y = (y + 800) % 1600 - 800
    return x, y


def draw_dust(ctx, st, t, lit, cam):
    d = st["dust"]
    x, y = dust_pos(st, t)
    z = d["z"]
    fl = 0.6 + 0.4 * np.sin(t * 1.7 + d["ph"] * 3)
    a0 = np.clip((t - d["t_on"]) / 1.5, 0, 1) * fl
    for xi, yi, zi, ai, ri in zip(x, y, z, a0, d["r"]):
        # parallax: nearer motes move more with the camera zoom
        px = CX + xi * (0.6 + 0.4 * zi) * (1 + (cam.zoom - 1) * zi * 0.6)
        py = CY + yi * (0.6 + 0.4 * zi) * (1 + (cam.zoom - 1) * zi * 0.6)
        dist = math.hypot(px - CX, py - CY)
        glowk = 0.6 + 0.6 * lit * math.exp(-(dist / 700) ** 2)
        if zi > 2.3:        # out-of-focus bokeh
            r = 9 + 10 * (zi - 2.3)
            al = ai * 0.075 * glowk
            g = cairo.RadialGradient(px, py, 0, px, py, r)
            g.add_color_stop_rgba(0, *GOLD, al)
            g.add_color_stop_rgba(0.7, *GOLD, al * 0.8)
            g.add_color_stop_rgba(1, *GOLD, 0)
            ctx.set_source(g)
            ctx.arc(px, py, r, 0, 6.2832)
            ctx.fill()
        else:
            r = ri * (0.5 + 0.6 * zi)
            al = min(1.0, ai * (0.4 + 0.7 * zi) * glowk)
            ctx.set_source_rgba(1.0, 0.84, 0.5, al * 0.28)
            ctx.arc(px, py, r * 3, 0, 6.2832)
            ctx.fill()
            ctx.set_source_rgba(1.0, 0.9, 0.66, al)
            ctx.arc(px, py, r, 0, 6.2832)
            ctx.fill()


def nebula(fc):
    k = ("neb", fc.w, fc.h)
    if k not in _CACHE:
        w, h = 320, 180
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        n1 = fbm2(xs / 60.0, ys / 60.0, seed=3, octaves=5)
        n2 = fbm2(xs / 34.0 + 11, ys / 34.0 - 7, seed=9, octaves=4)
        r = np.hypot((xs - w / 2) / w * 1.6, (ys - h * 0.48) / h)
        fall = np.exp(-(r / 0.55) ** 2)
        a = np.clip(n1 * 0.9 + 0.35, 0, 1) ** 2 * (0.35 + 0.65 * fall)
        b = np.clip(n2 * 0.9 + 0.1, 0, 1) ** 2 * fall
        img = (a[..., None] * np.array(col("dusk1")) * 0.9 + b[..., None] * np.array(col("dusk2")) * 0.55
               + fall[..., None] * np.array([0.05, 0.035, 0.02]))
        img = cv2.resize(img.astype(np.float32), (fc.w, fc.h), interpolation=cv2.INTER_CUBIC)
        _CACHE[k] = np.clip(img, 0, 1)
    return _CACHE[k]


# ------------------------------------------------------------------ drawing: the Crystal
def crystal_geom(st, t):
    R = rot_at(st, t)
    yaw, pitch = view_at(t)
    pr = Proj(CX, CY, unit_at(t), d4=D4, d3=7.0, view=rot3(yaw, pitch, 0.0))
    g, frac = growth(st, t)
    V = (V24 * g[:, None]) @ R.T
    return R, pr, V, g, frac, (yaw, pitch)


def _depth(Pw, Qz):
    near4 = (Pw + 1.4142) / 2.8284
    near3 = (Qz + 1.9) / 3.8
    return np.clip(0.62 * near4 + 0.38 * near3, 0, 1)


def draw_crystal(ctx, st, t, R, pr, V, g, frac, I, sheet=None, front=None, sub=5, dsel=(0.0, 1.01)):
    """additive luminous edges; sheet=(z0, strength) lights edges near the slicing plane;
    front: None = all, True = only segments in front of view z=0 (over the Flag), False = only behind;
    dsel: depth band (0 far .. 1 near) drawn — lets far edges go to a defocused layer"""
    if I <= 0.002:
        return
    s = np.linspace(0, 1, sub + 1)
    A = V[E24[:, 0]]
    B = V[E24[:, 1]]
    P = A[:, None, :] * (1 - s[None, :, None]) + B[:, None, :] * s[None, :, None]     # (96, sub+1, 4)
    flat = P.reshape(-1, 4)
    S2, Q, k4, k3 = pr.full(flat)
    P3, _ = pr.to3(flat)
    S2 = S2.reshape(96, sub + 1, 2)
    dep = _depth(flat[:, 3], Q[:, 2]).reshape(96, sub + 1)
    zq = Q[:, 2].reshape(96, sub + 1)
    boost = np.zeros_like(dep)
    if sheet is not None and sheet[1] > 0:
        dz = P3[:, 2].reshape(96, sub + 1) - sheet[0]
        boost = sheet[1] * np.exp(-(dz / 0.10) ** 2)
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    track_k = seg(t, T_N02 - 0.5, T_N02 + 1.0) * (1 - seg(t, 10.6, 11.4))
    EDGE_K = 1.0 + (EDGE_TRACK == 1) * 1.1 * track_k + (EDGE_TRACK == 2) * 0.5 * track_k \
        - (EDGE_TRACK == 0) * 0.40 * track_k
    for e in range(96):
        fr = frac[e]
        if fr <= 0:
            continue
        for j in range(sub):
            s0, s1 = s[j], s[j + 1]
            if s0 >= fr:
                break
            if front is not None:
                zm = 0.5 * (zq[e, j] + zq[e, j + 1])
                if (zm > 0.0) != front:
                    continue
            dm = 0.5 * (dep[e, j] + dep[e, j + 1])
            if not (dsel[0] <= dm < dsel[1]):
                continue
            x0, y0 = S2[e, j]
            if s1 > fr:
                u = (fr - s0) / (s1 - s0)
                x1, y1 = S2[e, j] + (S2[e, j + 1] - S2[e, j]) * u
            else:
                x1, y1 = S2[e, j + 1]
            d = 0.5 * (dep[e, j] + dep[e, j + 1])
            b = 0.5 * (boost[e, j] + boost[e, j + 1])
            c = depth_col(d ** 0.8)
            al = (0.04 + 0.66 * d ** 2.3) * I * EDGE_K[e]
            if EDGE_TRACK[e] > 0:
                c = c + (TRACK_C[EDGE_TRACK[e]] - c) * 0.85 * track_k
                al = max(al, (0.22 + 0.3 * d) * I * track_k)
            wdt = 0.7 + 3.0 * d ** 1.6
            hot = np.array([1.0, 0.97, 0.9])
            # wide saturated halo (gold near, violet far) -> the light has colour; thin hot core
            halo = c * np.array([1.0, 0.94, 0.84])
            ctx.set_source_rgba(*(halo + (hot - halo) * min(1.0, b) * 0.5), min(1.0, al * 0.16 + b * 0.14))
            ctx.set_line_width(wdt * 5.5 + b * 7)
            ctx.move_to(x0, y0)
            ctx.line_to(x1, y1)
            ctx.stroke()
            core = c * (0.45 + 0.35 * (EDGE_TRACK[e] > 0) * track_k) + (0.55 - 0.35 * (EDGE_TRACK[e] > 0) * track_k)
            ctx.set_source_rgba(*(core + (hot - core) * min(1.0, b)), min(1.0, al + b * 0.45))
            ctx.set_line_width(wdt * 0.8 + b * 1.2)
            ctx.move_to(x0, y0)
            ctx.line_to(x1, y1)
            ctx.stroke()
    ctx.restore()


def draw_aura(ctx, pr, V, I, t):
    """the Crystal's light scattered in the void: warm heart, violet fringe, breathing with the 4D turn"""
    if I <= 0.002:
        return
    S2, _, k4, _ = pr.full(V)
    r = float(np.percentile(np.hypot(S2[:, 0] - CX, S2[:, 1] - CY), 80)) + 40
    br = 0.85 + 0.15 * float(np.clip(k4.max() - 1.2, 0, 1))
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    g = cairo.RadialGradient(CX, CY, 0, CX, CY, r * 2.1)
    g.add_color_stop_rgba(0.0, 1.0, 0.86, 0.60, 0.18 * I * br)
    g.add_color_stop_rgba(0.35, 0.80, 0.62, 0.45, 0.08 * I * br)
    g.add_color_stop_rgba(0.7, 0.30, 0.22, 0.55, 0.045 * I)
    g.add_color_stop_rgba(1.0, 0.10, 0.08, 0.30, 0.0)
    ctx.set_source(g)
    ctx.arc(CX, CY, r * 2.1, 0, 6.2832)
    ctx.fill()
    ctx.restore()


def draw_facets(ctx, pr, V, g, I, t):
    """translucent glass: each triangle carries a gradient sheen (bright toward its nearest vertex), stronger
    at grazing angles (fresnel), gold when near in W, violet when far"""
    if I <= 0.002:
        return
    S2, Q, k4, k3 = pr.full(V)
    dep = _depth(V[:, 3], Q[:, 2])
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    for (a, b, c) in T24:
        vis = min(g[a], g[b], g[c])
        if vis <= 0.01:
            continue
        n = np.cross(Q[b] - Q[a], Q[c] - Q[a])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        fres = 1 - abs(n[2]) / nn
        tri = (a, b, c)
        dd = dep[list(tri)]
        hi = tri[int(np.argmax(dd))]
        lo = tri[int(np.argmin(dd))]
        d = float(dd.mean())
        al = (0.018 + 0.085 * fres ** 2.2) * (0.25 + 0.75 * d) * I * clamp(vis)
        cc = MID_C + (NEAR_C - MID_C) * d
        gr = cairo.LinearGradient(*S2[hi], *S2[lo])
        gr.add_color_stop_rgba(0, *cc, al)
        gr.add_color_stop_rgba(1, *(MID_C * 0.7), al * 0.15)
        ctx.set_source(gr)
        ctx.move_to(*S2[a])
        ctx.line_to(*S2[b])
        ctx.line_to(*S2[c])
        ctx.close_path()
        ctx.fill()
    ctx.restore()


def draw_vertices(ctx, pr, V, g, I, t, Vprev=None):
    S2, Q, k4, k3 = pr.full(V)
    dep = _depth(V[:, 3], Q[:, 2])
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    if Vprev is not None:            # sparks flying out of the kindling light leave streaks
        P2, _, _, _ = pr.full(Vprev)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for i in range(24):
            if not (0.01 < g[i] < 0.995):
                continue
            x0, y0 = P2[i]
            x1, y1 = S2[i]
            if math.hypot(x1 - x0, y1 - y0) < 2:
                continue
            lg = cairo.LinearGradient(x0, y0, x1, y1)
            lg.add_color_stop_rgba(0, *GOLD, 0)
            lg.add_color_stop_rgba(1, *WHITE_GOLD, 0.8 * I)
            ctx.set_source(lg)
            ctx.set_line_width(2.4)
            ctx.move_to(x0, y0)
            ctx.line_to(x1, y1)
            ctx.stroke()
    for i in range(24):
        if g[i] <= 0.01:
            continue
        d = dep[i]
        tw = 0.75 + 0.25 * math.sin(t * 3.1 + i * 1.7)
        al = (0.15 + 0.85 * d ** 2) * I * tw * min(1.0, g[i] * 1.5)
        x, y = S2[i]
        r = 5 + 11 * d
        gr = cairo.RadialGradient(x, y, 0, x, y, r * 2.2)
        gr.add_color_stop_rgba(0, 1, 0.97, 0.88, al)
        gr.add_color_stop_rgba(0.25, *WHITE_GOLD, al * 0.5)
        gr.add_color_stop_rgba(1, *GOLD, 0)
        ctx.set_source(gr)
        ctx.arc(x, y, r * 2.2, 0, 6.2832)
        ctx.fill()
        if d > 0.55:
            L = (10 + 34 * (d - 0.55)) * tw
            ctx.set_line_width(1.1)
            for ang in (0.35, 0.35 + math.pi / 2):
                dx, dy = math.cos(ang) * L, math.sin(ang) * L
                lg = cairo.LinearGradient(x - dx, y - dy, x + dx, y + dy)
                lg.add_color_stop_rgba(0, *WHITE_GOLD, 0)
                lg.add_color_stop_rgba(0.5, *WHITE_GOLD, al * 0.8)
                lg.add_color_stop_rgba(1, *WHITE_GOLD, 0)
                ctx.set_source(lg)
                ctx.move_to(x - dx, y - dy)
                ctx.line_to(x + dx, y + dy)
                ctx.stroke()
    ctx.restore()


def draw_ley(ctx, pr, V, t, I):
    """faint ley lines radiating from the Crystal's heart along its projected vertices"""
    if I <= 0.002:
        return
    S2, Q, _, _ = pr.full(V)
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.set_line_width(1.0)
    for i in range(24):
        dx, dy = S2[i, 0] - CX, S2[i, 1] - CY
        L = math.hypot(dx, dy)
        if L < 5:
            continue
        ux, uy = dx / L, dy / L
        far = 1500
        lg = cairo.LinearGradient(CX, CY, CX + ux * far, CY + uy * far)
        lg.add_color_stop_rgba(0, *GOLD, 0)
        lg.add_color_stop_rgba(min(0.9, L / far), *GOLD, 0.10 * I)
        lg.add_color_stop_rgba(1, *GOLD, 0)
        ctx.set_source(lg)
        ctx.move_to(CX + ux * L * 0.6, CY + uy * L * 0.6)
        ctx.line_to(CX + ux * far, CY + uy * far)
        ctx.stroke()
    # a few slow concentric resonance rings
    for k in range(3):
        ph = (t * 0.12 + k / 3) % 1
        r = 260 + ph * 900
        ctx.set_source_rgba(*GOLD, 0.035 * I * math.sin(ph * math.pi))
        ctx.arc(CX, CY, r, 0, 6.2832)
        ctx.stroke()
    ctx.restore()


def draw_kindle(ctx, t):
    """the point of light the Crystal condenses from"""
    a = seg(t, T_N01 - 0.05, T_N01 + 0.5, ease_out) * (1 - seg(t, T_N01 + 1.6, T_N01 + 3.4, ease_in_out))
    if a <= 0:
        return
    br = 0.8 + 0.2 * math.sin(t * 9)
    r = 18 + 60 * seg(t, T_N01, T_N01 + 1.8, ease_out)
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    g = cairo.RadialGradient(CX, CY, 0, CX, CY, r)
    g.add_color_stop_rgba(0, 1, 1, 0.95, a * br)
    g.add_color_stop_rgba(0.08, *WHITE_GOLD, a * 0.8)
    g.add_color_stop_rgba(0.3, *GOLD, a * 0.18)
    g.add_color_stop_rgba(1, *GOLD, 0)
    ctx.set_source(g)
    ctx.arc(CX, CY, r, 0, 6.2832)
    ctx.fill()
    # anamorphic streak
    ctx.save()
    ctx.translate(CX, CY)
    ctx.scale(1, 0.012)
    g = cairo.RadialGradient(0, 0, 0, 0, 0, 700)
    g.add_color_stop_rgba(0, 1, 0.95, 0.85, 0.55 * a)
    g.add_color_stop_rgba(1, 0.6, 0.7, 1.0, 0)
    ctx.set_source(g)
    ctx.arc(0, 0, 700, 0, 6.2832)
    ctx.fill()
    ctx.restore()
    ctx.restore()


# ------------------------------------------------------------------ the slicing plane and its section
def sheet_state(t):
    """(z0, sheet alpha) — the 2-plane {z = z0, w = 0} sweeps in from the front and stops at the centre"""
    a = seg(t, 10.85, 11.25, ease_in_out) * (1 - seg(t, T_ROT + 0.02, T_ROT + 0.30, ease_in_out))
    z0 = 1.75 * (1 - seg(t, 10.95, T_ROT, lambda x: ease_out(x, 2.2)))
    return z0, a


def draw_sheet(ctx, pr, z0, a):
    if a <= 0.002:
        return
    n = 10
    A = 2.2
    us = np.linspace(-A, A, n + 1)
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    m = cairo.MeshPattern()
    grid = np.array([[(x, y, z0) for x in us] for y in us]).reshape(-1, 3)
    S2, k = pr.to2(pr.view3(grid))
    S2 = S2.reshape(n + 1, n + 1, 2)
    for j in range(n):
        for i in range(n):
            m.begin_patch()
            corners = ((j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i))
            m.move_to(*S2[corners[0]])
            for cc in corners[1:]:
                m.line_to(*S2[cc])
            for q, (jj, ii) in enumerate(corners):
                r = math.hypot(us[ii], us[jj]) / A
                al = a * 0.045 * max(0.0, 1 - r) ** 1.6
                m.set_corner_color_rgba(q, 1.0, 0.86, 0.55, al)
            m.end_patch()
    ctx.set_source(m)
    ctx.paint()
    # the plane's rim: a thin blade of light
    rim = np.array([(-A, -A, z0), (A, -A, z0), (A, A, z0), (-A, A, z0)])
    R2, _ = pr.to2(pr.view3(rim))
    ctx.set_line_width(1.6)
    ctx.set_source_rgba(1.0, 0.9, 0.7, 0.35 * a)
    ctx.move_to(*R2[0])
    for q in R2[1:]:
        ctx.line_to(*q)
    ctx.close_path()
    ctx.stroke()
    # the lattice ruled on the plane (reads its perspective)
    ctx.set_line_width(1.0)
    for k_ in np.linspace(-A, A, 17):
        for horiz in (True, False):
            pts = np.array([(x, k_, z0) if horiz else (k_, x, z0) for x in np.linspace(-A, A, 24)])
            P2, _ = pr.to2(pr.view3(pts))
            for q in range(len(pts) - 1):
                r = math.hypot(*pts[q, :2]) / A
                al = a * 0.16 * max(0.0, 1 - r * 0.8) ** 1.3
                if al < 0.004:
                    continue
                ctx.set_source_rgba(1.0, 0.9, 0.65, al)
                ctx.move_to(*P2[q])
                ctx.line_to(*P2[q + 1])
                ctx.stroke()
    ctx.restore()


RECT_PTS = resample([(-1.2, -1.0), (1.2, -1.0), (1.2, 1.0), (-1.2, 1.0)], 96)


def section_screen(st, t, R, pr, z0):
    """screen polygon of the true section (plus morph to the Flag rectangle after the settle)"""
    if t < T_SETTLE:
        poly = section(R, z0, 0.0)
        if not poly:
            return None, 0.0
        P = resample(poly, 96)
    else:
        m = seg(t, T_SETTLE, T_RECT, lambda x: ease_out_back(x, 1.1))
        P = resample([(-1, -1), (1, -1), (1, 1), (-1, 1)], 96) * (1 - m) + RECT_PTS * m
    pts = np.column_stack([P, np.full(len(P), z0)])
    S2, _ = pr.to2(pr.view3(pts))
    Pc, Pn = P - P.mean(0), np.roll(P, -1, 0) - P.mean(0)
    return S2, abs((Pc[:, 0] * Pn[:, 1] - Pc[:, 1] * Pn[:, 0]).sum()) * 0.5


def draw_section(ctx, S2, a_line, a_fill, t):
    if S2 is None:
        return
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.new_path()
    ctx.move_to(*S2[0])
    for p in S2[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    if a_fill > 0:
        ctx.set_source_rgba(1.0, 0.78, 0.25, a_fill)
        ctx.fill_preserve()
    if a_line > 0:
        ctx.set_source_rgba(1.0, 0.86, 0.5, 0.25 * a_line)
        ctx.set_line_width(12)
        ctx.stroke_preserve()
        ctx.set_source_rgba(1.0, 0.97, 0.88, a_line)
        ctx.set_line_width(2.6)
        ctx.stroke_preserve()
    ctx.new_path()
    ctx.restore()


# ------------------------------------------------------------------ the pole of light
def draw_pole_light(ctx, t):
    p = seg(t, T_POLE0, T_POLE1, lambda x: ease_in_out(x, 2))
    fade = 1 - seg(t, T_POLE1 + 0.05, T_WOOD1 + 0.35)
    if p <= 0 or fade <= 0:
        return
    x = FLAG_X0
    y0 = FLAG_Y0
    y1 = y0 + (POLE_L + 40) * p
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    for wdt, al in ((46, 0.10), (16, 0.30), (5, 0.8), (2.2, 1.0)):
        ctx.set_source_rgba(1.0, 0.9, 0.62, al * fade)
        ctx.set_line_width(wdt)
        ctx.move_to(x, y0)
        ctx.line_to(x, y1)
        ctx.stroke()
    if p < 1:       # the drawing spark
        g = cairo.RadialGradient(x, y1, 0, x, y1, 110)
        g.add_color_stop_rgba(0, 1, 1, 0.95, 1.0 * fade)
        g.add_color_stop_rgba(0.12, *WHITE_GOLD, 0.6 * fade)
        g.add_color_stop_rgba(1, *GOLD, 0)
        ctx.set_source(g)
        ctx.arc(x, y1, 110, 0, 6.2832)
        ctx.fill()
    ctx.restore()


def wood_reveal(t):
    """0..1 fraction of the pole (from the top) that has become wood"""
    return seg(t, T_POLE1, T_WOOD1, lambda x: ease_in_out(x, 2))


# ------------------------------------------------------------------ reflections ("A sign of itself")
REF_BOX = (FLAG_X0 - 60, FLAG_Y0 - 150, FLAG_X0 + CLOTH_W + 200, FLAG_Y0 + CLOTH_H * 2.3)
REF_C = (FLAG_X0 + CLOTH_W * 0.5, FLAG_Y0 + CLOTH_H * 0.5 + 70)     # image point placed at a cell's heart


def flag_layer(fc, tr, i, wood):
    """offscreen image of the whole Flag (cloth + upper pole, fading down) for the Crystal's reflections"""
    bx0, by0, bx1, by1 = REF_BOX
    s = fc.s
    w, h = int((bx1 - bx0) * s) + 1, int((by1 - by0) * s) + 1
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    c = cairo.Context(surf)
    c.scale(s, s)
    c.translate(-bx0, -by0)
    c.push_group()
    tr.draw(c, i, pole=wood > 0.999, cloth=False, light=FLAG_LIGHT)
    if wood <= 0.999:
        _draw_partial_wood(c, tr, i, wood)
    c.pop_group_to_source()
    m = cairo.LinearGradient(0, FLAG_Y0 + CLOTH_H * 0.9, 0, by1)
    m.add_color_stop_rgba(0, 0, 0, 0, 1)
    m.add_color_stop_rgba(1, 0, 0, 0, 0)
    c.mask(m)
    tr.draw(c, i, pole=False, cloth=True, light=FLAG_LIGHT)
    return surf


def _draw_partial_wood(ctx, tr, i, wood):
    if wood <= 0:
        return
    ctx.save()
    ctx.rectangle(FLAG_X0 - 30, FLAG_Y0 - 30, 60, 30 + POLE_L * wood)
    ctx.clip()
    tr.draw(ctx, i, pole=True, cloth=False, light=FLAG_LIGHT)
    ctx.restore()


def _incircle(tri):
    a = np.linalg.norm(tri[1] - tri[2])
    b = np.linalg.norm(tri[0] - tri[2])
    c = np.linalg.norm(tri[0] - tri[1])
    per = a + b + c
    if per < 1e-6:
        return tri.mean(0), 0.0
    ctr = (a * tri[0] + b * tri[1] + c * tri[2]) / per
    s_ = per / 2
    area = max(0.0, s_ * (s_ - a) * (s_ - b) * (s_ - c)) ** 0.5
    return ctr, area / s_


def draw_reflections(ctx, fc, st, R, pr, V, surf, a_ref, t):
    """'A sign of itself': the Crystal's facets on either side of the Flag are facing mirrors, so the Flag is
    reflected back and forth between them — two rows of whole images, alternately mirrored (each bounce
    reverses it), each bounce smaller and dimmer as it recedes into the Crystal toward a vanishing point:
    endlessly. Only the plain Flag appears, only inside the Crystal. The rows run outward from the snap and
    sway with the Crystal's turning."""
    if a_ref <= 0.003:
        return
    s = fc.s
    bx0, by0 = REF_BOX[0], REF_BOX[1]
    hc = np.array(REF_C)
    S2v, _, _, _ = pr.full(V)
    hull = hull2(S2v)
    pat = cairo.SurfacePattern(surf)
    pat.set_filter(cairo.FILTER_GOOD)
    ctx.save()
    ctx.new_path()
    ctx.move_to(*hull[0])
    for q_ in hull[1:]:
        ctx.line_to(*q_)
    ctx.close_path()
    ctx.clip()
    grow = unit_at(t) / (UNIT * 1.7)
    sway = 0.6 * spin_at(t) + 0.04 * math.sin(t * 0.9)
    jobs = []
    for side in (1, -1):
        ang = sway + (0.0 if side > 0 else math.pi) + side * 0.10
        vp = np.array([CX, CY]) + 560 * grow * np.array([math.cos(ang), math.sin(ang) - 0.22])
        rho = 0.70
        for k in range(1, 14):
            sc = rho ** k
            if sc * CLOTH_W < 5:
                break
            t0 = T_SNAP - 0.04 + 0.05 * k + (0.02 if side < 0 else 0.0)
            ap = seg(t, t0, t0 + 0.25, ease_out)
            if ap <= 0:
                break
            flip = -1.0 if k % 2 else 1.0
            L2 = np.array([[flip, 0.0], [0.0, 1.0]]) * sc
            ctr = vp + (hc - vp) * sc
            jobs.append((k, L2, ctr, a_ref * ap * 0.80 * 0.86 ** (k - 1)))
    jobs.sort(key=lambda j: -j[0])
    for k, L2, ctr, al in jobs:
        A = L2 / s
        o = ctr + L2 @ (np.array([bx0, by0]) - hc)
        mat = cairo.Matrix(A[0, 0], A[1, 0], A[0, 1], A[1, 1], o[0], o[1])
        try:
            mat.invert()
        except cairo.Error:
            continue
        pat.set_matrix(mat)
        ctx.set_source(pat)
        ctx.paint_with_alpha(al)
    ctx.restore()


# ------------------------------------------------------------------ bloom (own, so the page never blooms)
def bloom(img, fc, k=0.5, thr=0.62, radius=24):
    bright = np.maximum(img - thr, 0.0) * (1.0 / (1 - thr))
    r = radius * fc.s
    b = blur(bright, r * 0.35) * 0.45 + blur(bright, r) * 0.35 + blur(bright, r * 3.0) * 0.30
    return img + b * k


LIGHT_GAIN = 2.1


def tonemap(x):
    """soft-saturating light (additive layers only): colour holds in halos, cores burn to white"""
    return 1.0 - np.exp(-x)


def shutter_samples(st, t, shutter=1 / 48):
    """sub-frame sample times across a 180-degree shutter, more when the Crystal turns fast"""
    d = np.linalg.norm(rot_at(st, t) - rot_at(st, t - shutter))
    if t < T_N01 or d < 0.03:
        return [t]
    n = int(min(6, math.ceil(d / 0.025)))
    return [t - shutter * k / (n - 1) for k in range(n)]


def god_rays(light, cx, cy, n=14, step=0.045, decay=0.90):
    """light shafts: zoom-blur of the light layer away from the Crystal's heart (quarter resolution)"""
    h, w = light.shape[:2]
    sw, sh = max(8, w // 4), max(8, h // 4)
    small = cv2.resize(light, (sw, sh), interpolation=cv2.INTER_AREA)
    small = np.maximum(small - 0.35, 0)
    cxs, cys = cx * sw / w, cy * sh / h
    acc = np.zeros_like(small)
    wsum = 0.0
    for i in range(n):
        k = 1.0 / (1.0 + i * step)          # sample closer to the centre -> streaks outward
        M = np.array([[k, 0, cxs * (1 - k)], [0, k, cys * (1 - k)]], np.float32)
        wgt = decay ** i
        acc += cv2.warpAffine(small, np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2].astype(np.float32), (sw, sh),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * wgt
        wsum += wgt
    acc = cv2.GaussianBlur(acc / wsum, (0, 0), 1.2)
    return cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ render
def render_crystal(fc, st, t):
    R, pr, V, g, frac, (yaw, pitch) = crystal_geom(st, t)
    I = intensity(t)
    cam = cam_at(t)
    img = nebula(fc) * (0.10 + 0.9 * I)
    # --- void: stars + dust (behind)
    cv = Canvas(fc)
    ctx = cv.ctx
    draw_stars(ctx, st, t, yaw, pitch, I)
    img = cv.over(img)

    # --- the Crystal's light: a defocused far layer (deep in W / Z) and a crisp near layer
    z0, sa = sheet_state(t)
    has_flag = t >= T_RECT - 0.12
    Ic = I * (0.6 if has_flag else 1.0 - 0.35 * seg(t, 10.9, 11.4) * (1 - seg(t, T_SETTLE, T_RECT)))
    FAR = (0.0, 0.40)
    Lf = Canvas(fc)
    fcx = Lf.ctx
    fcx.save()
    cam.apply(fcx)
    draw_aura(fcx, pr, V, I, t)
    draw_facets(fcx, pr, V, g, I, t)
    # 180-degree shutter: fast 4D turns are drawn as several sub-frame samples (no strobing)
    samples = shutter_samples(st, t)
    for ts_ in samples:
        Rq, prq, Vq, gq, fq, _ = crystal_geom(st, ts_) if ts_ != t else (R, pr, V, g, frac, None)
        draw_crystal(fcx, st, ts_, Rq, prq, Vq, gq, fq, Ic / len(samples), sheet=(z0, sa * 0.45),
                     front=False if has_flag else None, dsel=FAR)
    fcx.restore()
    L = Canvas(fc)
    lc = L.ctx
    lc.save()
    cam.apply(lc)
    draw_kindle(lc, t)
    ley = I * seg(t, T_N02, T_N02 + 1.5) * (1 - seg(t, 10.3, 11.5))
    draw_ley(lc, pr, V, t, ley)
    for ts_ in samples:
        Rq, prq, Vq, gq, fq, _ = crystal_geom(st, ts_) if ts_ != t else (R, pr, V, g, frac, None)
        draw_crystal(lc, st, ts_, Rq, prq, Vq, gq, fq, Ic / len(samples), sheet=(z0, sa * 0.45),
                     front=False if has_flag else None, dsel=(FAR[1], 1.01))
    Vprev = None
    if T_N01 < t < T_N02:
        gp, _ = growth(st, t - 0.12)
        Vprev = (V24 * gp[:, None]) @ rot_at(st, t - 0.12).T
    draw_vertices(lc, pr, V, g, Ic, t, Vprev)
    draw_sheet(lc, pr, z0, sa)
    # the section: bright line + gold fill, handing over to the Flag's own yellow
    if t >= 11.0 and t < T_RECT + 0.5:
        S2, area = section_screen(st, t, R, pr, z0)
        a_line = seg(t, 11.0, 11.2) * (1 - seg(t, T_SETTLE + 0.05, T_RECT + 0.12))
        a_fill = (0.07 + 0.20 * seg(t, T_ROT, T_SETTLE)) * (1 - seg(t, T_RECT - 0.12, T_RECT + 0.3))
        a_fill *= seg(t, 11.0, 11.3)
        draw_section(lc, S2, a_line, a_fill, t)
    lc.restore()
    light = L.rgba()[..., :3] * LIGHT_GAIN + blur(Lf.rgba()[..., :3], 2.2 * fc.s) * LIGHT_GAIN
    lg = blur(light, 2.5 * fc.s) * 0.5 + blur(light, 14 * fc.s) * 0.5 + blur(light, 55 * fc.s) * 0.45
    sx, sy = cam.to_screen(CX, CY)
    rays = god_rays(light, sx * fc.s, sy * fc.s) * (0.55 * I)
    img = img + tonemap(light + lg + rays)

    # snap flash lights the void and the Crystal; bloom them now, BEFORE the Flag goes on, so the cloth keeps
    # exactly its palette shading (bloom on top would push the yellow toward lemon)
    fl = pulse(t, T_SNAP + 0.03, 0.10) * 0.9 + pulse(t, T_SNAP + 0.1, 0.35) * 0.35
    if fl > 0.003:
        img = img + np.array([1.0, 0.95, 0.84], np.float32) * fl * 0.30 * radial_mask(fc, CX, CY, 100, 1300)
    k_bloom = 0.35 + 0.45 * fl + 0.4 * seg(t, T_SUCK, T_SLAM, ease_in)
    img = bloom(np.clip(img, 0, 4), fc, k=k_bloom, thr=0.7)

    # --- the Flag and its reflections
    if has_flag:
        tr, i = flag_frame(st, fc.f)
        wood = wood_reveal(t)
        a_flag = seg(t, T_RECT - 0.12, T_RECT + 0.2)
        F = Canvas(fc)
        fx = F.ctx
        fx.save()
        cam.apply(fx)
        a_ref = seg(t, T_SNAP - 0.15, T_SNAP + 0.8, ease_in_out) * (1 + 0.35 * seg(t, T_SUCK, T_SLAM, ease_in))
        if a_ref > 0:
            surf = flag_layer(fc, tr, i, wood)
            draw_reflections(fx, fc, st, R, pr, V, surf, a_ref * a_flag, t)
        fx.push_group()
        draw_hero(fx, st, t, fc.f, wood)
        fx.pop_group_to_source()
        fx.paint_with_alpha(a_flag)
        draw_pole_light(fx, t)
        fx.restore()
        fl_rgba = F.rgba()
        # the Flag glows faintly in the dark: a warm halo around (never over) the cloth
        halo = blur(fl_rgba[..., :3], 18 * fc.s) * 0.45 + blur(fl_rgba[..., :3], 60 * fc.s) * 0.30
        halo *= (0.8 + 2.2 * fl) * (1 - fl_rgba[..., 3:4])
        img = over(img + halo, fl_rgba)

    # --- golden dust drifting through the void (it passes behind the Flag: nothing ever sits on the cloth)
    D = Canvas(fc)
    draw_dust(D.ctx, st, t, I, cam)
    dust = D.rgba()[..., :3]
    if has_flag:
        dust *= 1 - fl_rgba[..., 3:4]
    return img + dust


def page_y(t):
    """page top edge offset (design px): falls in at the slam, slides up at the end"""
    if t < T_SLAM:
        u = clamp((t - T_DROP) / (T_SLAM - T_DROP))
        return -H * (1 - u * u)
    if t < T_SLIDE0:
        tau = t - T_SLAM
        return -9 * math.exp(-tau * 11) * abs(math.sin(tau * 26))
    if t < T_SLIDE0 + 0.12:
        return 16 * math.sin(math.pi * 0.5 * (t - T_SLIDE0) / 0.12)
    u = clamp((t - T_SLIDE0 - 0.12) / (T_SLIDE1 - T_SLIDE0 - 0.12))
    return 16 * (1 - u) - (H + 160) * u ** 2.4


def _paper(fc):
    k = ("paper", fc.w, fc.h)
    if k not in _CACHE:
        arr = title.paper_texture(fc.s)
        surf = surface_from_array(arr)
        pat = cairo.SurfacePattern(surf)
        pat.set_matrix(cairo.Matrix(xx=fc.s, yy=fc.s))
        _CACHE[k] = (pat, surf)
    return _CACHE[k][0]


def render_page(fc, st, t, under):
    y = page_y(t)
    if y <= -H - 150:
        return under
    P = Canvas(fc)
    pc = P.ctx
    # shadow the page casts on what's beneath
    sh_a = 0.55 if t < T_SLAM else 0.35
    ybot = y + H
    g = cairo.LinearGradient(0, ybot, 0, ybot + 90)
    g.add_color_stop_rgba(0, 0, 0, 0, sh_a)
    g.add_color_stop_rgba(1, 0, 0, 0, 0)
    pc.rectangle(0, ybot, W, 90)
    pc.set_source(g)
    pc.fill()
    pc.save()
    shx, shy = (0.0, 0.0)
    if T_SLAM <= t < T_SLAM + 0.6:
        shx, shy = shake(t, 5 * math.exp(-(t - T_SLAM) * 7), 16, 21)
    pc.translate(shx, y + shy)
    pc.rectangle(-80, -80, W + 160, H + 80)
    pc.clip()
    title.draw_page(pc, t, T_SLAM, paper=_paper(fc))
    pc.restore()
    lay = P.rgba()
    # motion blur along the page's travel (shutter 180 deg)
    vel = (page_y(t + 1 / 48) - page_y(t - 1 / 48)) * 24
    k = int(abs(vel) / 24 * 0.5 * fc.s)
    if k >= 3:
        lay = cv2.blur(lay, (1, k))
    return over(under, lay)


def render(fc, st):
    t = fc.T
    if t >= T_SLAM and t < T_SLIDE0:
        under = np.zeros((fc.h, fc.w, 3), np.float32)
    elif t >= T_SLIDE0:
        under = solid(fc, "concrete_l")
    else:
        under = render_crystal(fc, st, t)
    if t >= T_DROP:
        under = render_page(fc, st, t, under)
    return np.clip(under, 0, 1).astype(np.float32)


def post(fc, st):
    t = fc.T
    if t >= T_DROP:
        return dict(vignette=0.30 + (0.25 - 0.30) * seg(t, T_DROP, T_SLAM), grain=0.03, ca=0.0)
    return dict(ca=0.8 * seg(t, T_N01, T_N01 + 2))


# ------------------------------------------------------------------ sound exports
def _export_foley(S, st):
    hits = [
        dict(t=T_N01, kind="swell", strength=0.5, pan=0.0, desc="a point of light kindles in the void"),
        dict(t=float(st["v_appear"].min()), kind="shimmer", strength=0.5, pan=0.0, desc="the Crystal begins to condense (vertices unfold)"),
        dict(t=float(st["v_appear"].max()) + 0.7, kind="chime", strength=0.6, pan=0.0, desc="the Crystal is complete (last edge ignites)"),
        dict(t=T_N02, kind="whoosh", strength=0.45, pan=0.0, desc="4D rotation in W accelerates (long rising turn, peaks ~8.6)"),
        dict(t=10.85, kind="sweep", strength=0.45, pan=0.3, desc="the glowing hyperplane appears and sweeps in (to 12.0)"),
        dict(t=11.35, kind="chime", strength=0.5, pan=0.0, desc="the plane first cuts the Crystal: a section polygon ignites"),
        dict(t=T_ROT, kind="whoosh", strength=1.0, pan=0.0, desc="flag_rotates_in: the decisive 4D turn, section morphs"),
        dict(t=T_SETTLE, kind="click", strength=0.7, pan=0.0, desc="the section locks onto the square"),
        dict(t=T_RECT, kind="swell", strength=0.7, pan=0.0, desc="square becomes the plain yellow Flag cloth"),
        dict(t=T_POLE0, kind="sweep", strength=0.55, pan=-0.2, desc="the pole draws itself as a line of light (downward, to 13.47)"),
        dict(t=T_POLE1, kind="thunk", strength=0.45, pan=-0.2, desc="the light becomes wood (to 13.8)"),
        dict(t=T_CLOTH, kind="rustle", strength=0.35, pan=0.0, desc="the geometric rectangle becomes real cloth and sags"),
        dict(t=T_SNAP, kind="snap", strength=1.0, pan=0.1, desc="flag_snap_open: gust, the Flag SNAPS open, bloom flash"),
        dict(t=T_SNAP + 0.1, kind="shimmer", strength=0.6, pan=0.0, desc="endless reflections of the Flag bloom in the facets"),
        dict(t=T_SUCK, kind="whoosh", strength=0.5, pan=0.0, desc="reverse-suck: everything draws inward toward the title"),
        dict(t=T_DROP, kind="whoosh", strength=0.6, pan=0.0, desc="the title page falls from above"),
        dict(t=T_SLAM, kind="slam", strength=1.0, pan=0.0, desc="title_slam: the IVG page lands full-frame"),
        dict(t=T_SLAM + 0.02, kind="scribble", strength=0.3, pan=-0.3, desc="thin white grid lines draw in, cascading (to ~16.9)"),
        dict(t=T_SLAM + title.DE_T, kind="swipe", strength=0.35, pan=-0.4, desc="'Die Entbabelung' wipes in left-to-right (0.55 s)"),
        dict(t=T_SLAM + 1.10, kind="tick", strength=0.15, pan=-0.5, desc="small type fades up line by line (to ~18.4)"),
        dict(t=T_SLIDE0, kind="rustle", strength=0.35, pan=0.0, desc="page dips (anticipation)"),
        dict(t=T_SLIDE0 + 0.12, kind="whoosh", strength=0.85, pan=0.0, desc="page slides UP out of frame (to 20.9), revealing concrete"),
        dict(t=T_SLIDE1, kind="settle", strength=0.2, pan=0.0, desc="page gone: flat concrete_l fog (s02 opens here at 21.0)"),
    ]
    for land, x in title.letter_landings(T_SLAM):
        hits.append(dict(t=land, kind="thunk", strength=0.75, pan=foley.screen_pan(x),
                         desc="title letter lands (THE UNBABELING)"))
    foley.export_events(S, "hits", hits)
    # hero flag: physics-derived flutter energy, snaps, flap rate (the sim ends at the slam: the page hides it)
    fd = st["flag"]
    fd["track"].export_foley(S, "hero_flag", f0=fd["f0"], gain=0.9)
    return hits

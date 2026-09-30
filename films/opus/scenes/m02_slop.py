"""m02 - THE GREAT MACHINES AND THE SLOP (22.0-48.5). Owner: M02.

One continuous camera move over one colossal nave mosaic, HVMANITAS: a crowned Byzantine face whose crown is the
three great machines of meaning (NATIONES a walled city, FIDES a domed temple, PHILOSOPHIAE the philosophers under a
tree - gilded cogs of tesserae turning inside them, tiny processions walking through them) and whose face, collar and
robe are a card stunt of thousands of tiny people, each wearing the colour of the picture where they stand.

    22.0  continue m01's tilt down (900 px/s) from the starry lapis, past the jewelled border, onto the dark crown
    23.0  N04: the machines sleep in candlelight; 'Nations.' 'Faiths.' 'Philosophies.' - each wakes on its word
          (light floods it, cogs clunk into motion, processions walk, its titulus sets letter by letter)
    29.8  N05: pull back - the machines are the crown of one vast icon of Humanity made of tiny people
    34.4  'Then came the slop.' ONE tessera - the card held by the tiny person who is the pupil of the left eye -
          flips into a glowing pastel phone; the camera is drawn into it: a scrolling feed, a face lit from below
    36.3  slop_crash: the camera whips back as a wave of split-flap flips washes across the wall; every person's
          card becomes a phone and lights their face with its own feed; the machines grind to a halt
    41.0  N07: wave upon wave; 'a different feed for every face'; 'until no two tesserae showed the same world' -
          every tile its own screen, the picture unrecoverable
    47.0  the camera dives into ONE slop tessera; its feed melts into pastel static that fills the frame by 48.30
          (hand-off to m03: scenes/m02_static.pastel_static, same T)
Render: python -m vx.render m02 --at 1,4.5,9   (local seconds)
"""
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import cairo
import cv2
import numpy as np
from scipy.spatial import cKDTree

from vx import *
from vx import mosaic as mz
from vx.foley import export_events, export_track, screen_pan

from scenes import m02_art as A
from scenes import m02_wall as MW
from scenes.m02_static import pastel_static

POST = dict(bloom=0.38, bloom_thresh=0.74, grain=0.018, vignette=0.30)

T_CUT_IN = 22.0
T_END = 48.5
T_STATIC = 48.30          # full-frame static from here (M03 hand-off)
V0_TILT = 1310.0          # m01's content velocity at the cut (design px/s, upward, constant in m01)
Z0 = 3.0                  # lapis tiles 4 wu -> 12 px (m01's ground tiles at the cut)

SLOP = [col(x) for x in ("#E8C8D0", "#B8E0E8", "#C9C0A8", "#9FD8FF", "#D8C8F0", "#C8F0D8", "#F0D8C0",
                         "#E0B8E8", "#A8E8E0")]
GLITCH = [col("#FF2E88"), col("#00E5FF")]


# ============================================================ timing (all from the locked timeline)
def _times(tl):
    t = dict(
        wake=[tl.word("N04", "Nations")[0], tl.word("N04", "Faiths")[0], tl.word("N04", "Philosophies")[0]],
        pull0=tl.line("N05")["start"],
        whole=tl.word("N05", "whole")[0],
        picture=tl.word("N05", "picture")[0],
        slop=tl.cue("slop"),
        crash=tl.cue("slop_crash"),
        wave2=tl.word("N07", "Wave")[0],
        wave3=tl.word("N07", "wave", 1)[0],
        face=tl.word("N07", "face")[0],
        different=tl.word("N07", "different")[0],
        until=tl.word("N07", "until")[0],
        world=tl.word("N07", "world")[0],
    )
    return t


# ============================================================ camera
def _herm(p0, p1, m0, m1, s):
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1


def _zoom_move(c0, z0, c1, z1, a, b=None):
    """log-zoom from (c0, z0) to (c1, z1) at eased progress a; zoom-in keeps the target converging on the centre,
    zoom-out lets the start centre drift out to its final place (no overshoot either way)."""
    b = a if b is None else b
    z = z0 * (z1 / z0) ** a
    c0 = np.asarray(c0, float)
    c1 = np.asarray(c1, float)
    if z1 >= z0:
        s0 = (c1 - c0) * z0
        c = c1 - s0 * (1 - b) / z
    else:
        s1 = (c0 - c1) * z1
        c = c0 - s1 * b / z
    return float(c[0]), float(c[1]), float(z)


class Cam:
    def __init__(self, tt, eye_xy, dive_xy, dive_z):
        self.t = tt
        self.eye = eye_xy
        self.dive = dive_xy
        self.dive_z = dive_z

    def __call__(self, T):
        t = self.t
        # A: tilt down from the lapis (continuing m01 at 900 px/s), settle on the dormant crown, slow truck
        if T < t["pull0"]:
            dur = 2.8
            s = clamp((T - T_CUT_IN) / dur)
            y = _herm(-300.0, 178.0, (V0_TILT / Z0) * dur, 0.0, s) if T < T_CUT_IN + dur else 178.0
            y += 20.0 * smoothstep(T_CUT_IN + dur, t["pull0"], T)
            x = 860.0 + 125.0 * ease_in_out_sine(clamp((T - T_CUT_IN) / (t["pull0"] - T_CUT_IN)))
            if T < 24.6:
                z = Z0 + 0.3 * ease_in_out_sine(clamp((T - T_CUT_IN) / 2.6))
            else:
                z = 3.3 - 0.45 * ease_in_out_sine(clamp((T - 24.6) / (t["wake"][0] - 24.6)))
            # each machine wakes with a clunk: a tiny push and settle
            for tw in t["wake"]:
                if T > tw:
                    u = T - tw
                    z *= 1.0 + 0.016 * (1.0 - math.exp(-u / 0.05)) * math.exp(-u / 0.28)
            return x, y, z
        # C: pull back to the whole picture
        P1 = t["picture"] + 0.25
        if T < P1:
            a = ease_in_out(clamp((T - t["pull0"]) / (P1 - t["pull0"])), 2.4)
            return _zoom_move((985.0, 198.0), 2.85, (960.0, 540.0), 1.0, a)
        # D: hold, leaning very slightly toward the left eye
        if T < t["slop"]:
            a = ease_in_out_sine(clamp((T - P1) / (t["slop"] - P1)))
            return _zoom_move((960.0, 540.0), 1.0, (940.0, 532.0), 1.06, a)
        # E: drawn into the first screen
        E1 = t["crash"] - 0.12
        if T < t["crash"]:
            a = ease_in_out(clamp((T - t["slop"] - 0.05) / (E1 - t["slop"] - 0.05)), 2.2)
            return _zoom_move((940.0, 532.0), 1.06, self.eye, 11.0, a, ease_out(a, 1.6))
        # F: the crash - whip back out as the wave spreads
        F1 = t["crash"] + 1.9
        if T < F1:
            a = ease_out(clamp((T - t["crash"]) / (F1 - t["crash"])), 3.2)
            return _zoom_move(self.eye, 11.0, (960.0, 546.0), 1.12, a)
        # G: drift while the waves come
        G1 = t["different"] - 0.1
        if T < G1:
            a = ease_in_out_sine(clamp((T - F1) / (G1 - F1)))
            return _zoom_move((960.0, 546.0), 1.12, (968.0, 548.0), 1.17, a)
        # H: 'a different feed for every face' - lean into the face
        H1 = t["until"] + 0.1
        if T < H1:
            a = ease_in_out(clamp((T - G1) / (H1 - G1)), 2.0)
            return _zoom_move((968.0, 548.0), 1.17, (1000.0, 590.0), 2.3, a)
        # I: back out over the dissolving picture
        I1 = 47.0
        if T < I1:
            a = ease_in_out(clamp((T - H1) / (I1 - H1)), 2.0)
            return _zoom_move((1000.0, 590.0), 2.3, (968.0, 560.0), 1.3, a)
        # J: dive into one tessera; full stop by T_STATIC
        a = ease_in_out(clamp((T - I1) / (T_STATIC - I1)), 2.6)
        return _zoom_move((968.0, 560.0), 1.3, self.dive, self.dive_z, a, ease_out(clamp(a * 1.25), 2.0))

    def vel(self, T, dt=1 / 48):
        a = self(T - dt)
        b = self(T + dt)
        return ((b[0] - a[0]) / (2 * dt), (b[1] - a[1]) / (2 * dt), (b[2] - a[2]) / (2 * dt))


def world_to_screen(cam, x, y):
    return (np.asarray(x) - cam[0]) * cam[2] + 960.0, (np.asarray(y) - cam[1]) * cam[2] + 540.0


# ============================================================ building the wall (cached)
def _src_hash():
    h = hashlib.sha1()
    for f in ("m02_art.py", "m02_wall.py"):
        h.update((Path(__file__).parent / f).read_bytes())
    h.update(b"v3")
    return h.hexdigest()[:10]


def _build_wall(S):
    key = _src_hash()
    path = S.cache / f"wall_{key}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    M = A.build()
    ts = M["ts"].copy()
    ts[ts < 1.2] = 0.0
    xy, ang, sz = MW.flow(M["edges"], ts, A.K, A.X0, A.Y0, seed=7)
    crown_a, pend_a, bcov = M["crown_a"], M["pend_a"], M["bust_cov"].astype(np.uint8)
    n_static = len(xy)
    parts = [(xy, ang, sz, np.zeros(len(xy), np.int32), np.ones(len(xy), np.float32))]
    # the bust: every tiny person is a glyph of ~14 tesserae (head, hair, shoulders, body, hands, the card)
    cells = M["cells"]
    keep = M["cell_keep"]
    G = A.glyph_tiles(cells, keep, M["cell_cols"])
    ok = (A.sample(bcov, G["xy"]) > 0) & (A.sample(crown_a, G["xy"]) < 100) & (A.sample(pend_a, G["xy"]) < 100)
    G = {k: v[ok] for k, v in G.items()}
    ng = len(G["xy"])
    parts.append((G["xy"], G["ang"], G["size"], np.zeros(ng, np.int32), G["aspect"]))
    for k, d in enumerate(A.DISCS):
        dxy, dang, dsz = MW.rings(d["c"][0], d["c"][1], 0.0, d["r"] - 0.9, 1.9 if d["r"] > 12 else 1.6, seed=k,
                                  phase=0.0)
        parts.append((dxy, dang, dsz, np.full(len(dxy), k + 1, np.int32), np.ones(len(dxy), np.float32)))
    xy = np.vstack([p[0] for p in parts]).astype(np.float32)
    ang = np.concatenate([p[1] for p in parts]).astype(np.float32)
    sz = np.concatenate([p[2] for p in parts]).astype(np.float32)
    grp = np.concatenate([p[3] for p in parts]).astype(np.int32)
    aspect = np.concatenate([p[4] for p in parts]).astype(np.float32)
    is_g = np.zeros(len(xy), bool)
    is_g[n_static:n_static + ng] = True
    base = A.sample(M["rgb"], xy).astype(np.float32) / 255.0
    base[is_g] = G["rgb"]
    gold = A.sample(M["gold"], xy).astype(np.float32) / 255.0
    crown = A.sample(crown_a, xy) > 100
    tit = A.sample(M["tit_a"], xy) > 110
    pend = A.sample(pend_a, xy) > 100
    lab = A.sample(M["lab_a"], xy) > 100
    bust = is_g
    # gold leaf in the crown only where the machine art is gold-coloured
    gcol = np.array(A.GOLDC, np.float32)
    isgold = np.abs(base - gcol).sum(1) < 0.12
    gold = np.where(crown | pend | bust, isgold.astype(np.float32) * crown, gold)
    gold[lab | tit] = 0.0
    ci = np.full(len(xy), -1, np.int32)
    part = np.full(len(xy), -1, np.int8)
    ci[is_g] = G["ci"]
    part[is_g] = G["part"]
    out = dict(xy=xy, ang=ang, sz=sz, grp=grp, aspect=aspect, base=base, gold=gold.astype(np.float32),
               crown=crown, tit=tit, pend=pend, lab=lab, bust=bust, ci=ci, part=part, cells=cells, keep=keep,
               cell_cols=M["cell_cols"].astype(np.float32))
    np.savez(path, **out)
    for old in S.cache.glob("wall_*.npz"):
        if old != path:
            old.unlink()
    return out


# ============================================================ per-group precompute
def _polar(xy, c):
    d = xy - np.asarray(c, np.float32)
    return np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0])


def _disc_colours(W, st):
    """rest-frame colours of every cog tile (gold spokes/rim/teeth, dark openwork, jewelled rims)."""
    cols = W["base"].copy()
    gold = W["gold"].copy()
    ink = np.array(mz.SC["ink"])
    dark = np.array(mz.SC["lapis_d"]) * 0.7
    for k, d in enumerate(A.DISCS):
        m = W["grp"] == k + 1
        r, ph = _polar(W["xy"][m], d["c"])
        R = d["r"]
        rn = r / R
        c = np.zeros((m.sum(), 3), np.float32)
        g = np.zeros(m.sum(), np.float32)
        spoke = np.cos(d["spokes"] * ph) > 0.55
        tooth = np.cos(d["teeth"] * ph) > -0.1
        is_gold = (rn < 0.26) | ((rn >= 0.26) & (rn < 0.70) & spoke) | ((rn >= 0.70) & (rn < 0.84)) | \
                  ((rn >= 0.84) & tooth)
        c[:] = dark
        g[is_gold] = 1.0
        c[is_gold] = mz.SC["gold"]
        c[rn < 0.10] = ink
        g[rn < 0.10] = 0
        if d["style"] == "crown":
            gem = (rn >= 0.70) & (rn < 0.84) & (np.cos(d["teeth"] * ph) > 0.7)
            gi = np.nonzero(gem)[0]
            pal = np.array([mz.SC["porphyry_l"], mz.SC["malachite"], mz.SC["lapis_l"]])
            c[gi] = pal[(np.floor((ph[gi] + math.pi) / (2 * math.pi) * d["teeth"]).astype(int)) % 3]
            g[gi] = 0
            pearl = (rn >= 0.9) & tooth & (np.cos(d["teeth"] * ph) > 0.8)
            c[pearl] = mz.SC["marble"]
            g[pearl] = 0
        cols[m] = c
        gold[m] = g
    return cols, gold


def _proc_setup(W):
    """tiles of each procession band -> (idx, u, v) + the strip (vx.icon figures) and its flame/face masks."""
    out = []
    xy = W["xy"]
    for pi, P in enumerate(A.PROCS):
        Hb = P["h"] * A.PROC_TOP
        m = (xy[:, 0] >= P["x0"]) & (xy[:, 0] <= P["x1"]) & (xy[:, 1] >= P["y"] - Hb) & (xy[:, 1] < P["y"]) & \
            (W["grp"] == 0) & ~W["tit"] & ~W["bust"] & ~W["pend"]
        if P["gap"] is not None:
            m &= ~((xy[:, 0] > P["gap"][0]) & (xy[:, 0] < P["gap"][1]) & (xy[:, 1] > P["y"] - 26))
        idx = np.nonzero(m)[0]
        strip, masks = A.procession_strip(P, seed=pi + 1)
        out.append(dict(idx=idx, u=(xy[idx, 0] - P["x0"]).astype(np.float32),
                        v=(xy[idx, 1] - (P["y"] - Hb)).astype(np.float32), strip=strip, masks=masks, P=P,
                        cx=0.5 * (P["x0"] + P["x1"]), cy=P["y"] - Hb / 2))
    return out


def _integrate(ts, rate_fn):
    """cumulative integral of rate over the time grid ts (for cog angles / walk offsets as pure functions of T)."""
    r = np.array([rate_fn(t) for t in ts])
    return np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(ts))])


# ============================================================ setup
def setup(S):
    tl = S.tl
    tt = _times(tl)
    W = _build_wall(S)
    wall = MW.Wall(W["xy"], W["ang"], W["sz"], W["grp"], [(d["c"][0], d["c"][1], d["r"]) for d in A.DISCS],
                   cs=1.6, seed=3, aspect=W["aspect"])
    n = wall.n
    xy = W["xy"]
    disc_cols, disc_gold = _disc_colours(W, None)
    base = disc_cols
    gold = disc_gold
    # the upper lapis in m01's mix (lapis 62 / lapis_d 28 / night_vault 6 / lapis_l 4 %), darker courses every 50 wu
    lap = np.nonzero((xy[:, 1] < 17.0) & (gold < 0.5) & (W["grp"] == 0))[0]
    hh = np.random.default_rng(17).random(len(lap))
    pal = np.array([mz.SC["lapis"], mz.SC["lapis_d"], mz.SC["night_vault"], mz.SC["lapis_l"]])
    pick = np.searchsorted(np.array([0.62, 0.90, 0.96, 1.0]), hh)
    lc = pal[np.clip(pick, 0, 3)]
    ring = np.hypot(xy[lap, 0] - 960.0, xy[lap, 1] + 2300.0) - 1800.0
    lc[(ring % 50.0) < 3.6] = mz.SC["lapis_d"]
    base[lap] = np.clip(lc * np.array([0.85, 1.42, 1.78], np.float32), 0, 1)   # m01's saturated ultramarine

    # machines: which tiles belong to which machine (crown columns); wake times
    mach = np.full(n, -1, np.int8)
    for m_, (xa, xb) in enumerate(A.MACHINE_X):
        mm = (xy[:, 0] >= xa) & (xy[:, 0] < xb) & (xy[:, 1] < A.BAND_Y[1]) & (xy[:, 1] > 60) & \
             (W["crown"] | (W["grp"] > 0))
        mach[mm] = m_
    for k, d in enumerate(A.DISCS):
        mach[W["grp"] == k + 1] = d["m"]

    # the first screen: the card of the tiny person who is the pupil of the left eye
    cells = W["cells"]
    ex, ey = A.EYES[0]
    card_tiles = np.nonzero(W["part"] == 3)[0]
    cc = W["ci"][card_tiles]
    dd = (cells[cc, 0] - (ex - 4)) ** 2 + (cells[cc, 1] - (ey + 1)) ** 2
    first = int(card_tiles[np.argmin(dd)])
    first_cell = int(W["ci"][first])
    eye_xy = (float(xy[first, 0]), float(xy[first, 1] - 3.0))

    # ---------------------------------------------------------------- the slop schedule
    rng = np.random.default_rng(99)
    h = wall.hash
    part = W["part"]
    ox, oy = xy[first]
    dist1 = np.hypot(xy[:, 0] - ox, xy[:, 1] - oy)
    jit = (rng.random(n) - 0.5) * 0.22
    # the crash wave: every person's card becomes a phone; a sparse scatter of other tiles flips too
    p1 = np.full(n, 0.13, np.float32)
    p1[part == 3] = 1.0
    p1[(part == 2) | (part == 4) | (part == 5)] = 0.0
    p1[part == 1] = 0.10
    V1 = 950.0
    t1 = np.where(h < p1, tt["crash"] + dist1 / V1 + jit, np.inf).astype(np.float32)
    t1[first] = tt["slop"]
    # wave 2 (from the top-left of the crown) and wave 3 (from the lower right): the picture degrades
    d2 = np.hypot(xy[:, 0] - 640, xy[:, 1] - 90)
    d3 = np.hypot(xy[:, 0] - 1760, xy[:, 1] - 1000)
    h2 = rng.random(n)
    h3 = rng.random(n)
    t2 = np.where(h2 < 0.2, tt["wave2"] + d2 / 850.0 + (rng.random(n) - 0.5) * 0.2, np.inf).astype(np.float32)
    t3 = np.where(h3 < 0.22, tt["wave3"] + d3 / 850.0 + (rng.random(n) - 0.5) * 0.2, np.inf).astype(np.float32)
    faces_ = (part == 2) | (part == 4) | (part == 5)
    t2[faces_] = np.inf
    t3[faces_] = np.inf
    # 'a different feed for every face': every phone keeps flicking to new feeds, each on its own clock
    is_card = part == 3
    t7 = np.where(is_card, tt["wave2"] + 0.6 + rng.random(n) * 1.4, np.inf).astype(np.float32)
    t8 = np.where(is_card, tt["different"] + rng.random(n) * 1.3, np.inf).astype(np.float32)
    # dissolution: every tile, heads included, flips before 'world' ... and keeps changing its own feed
    span = tt["world"] - tt["until"] - 0.15
    t4 = (tt["until"] + (rng.random(n) ** 0.8) * span).astype(np.float32)
    t5 = (t4 + 0.35 + rng.random(n) * 1.2).astype(np.float32)
    t6 = (t5 + 0.3 + rng.random(n) * 1.0).astype(np.float32)
    # the flips of each tile, sorted; each flip shows a new feed
    F = np.sort(np.stack([t1, t2, t3, t7, t8, t4, t5, t6], 1), axis=1)
    F[F > 48.2] = np.inf
    # first flip time per tile; heads get lit by their person's card
    tfirst = F[:, 0].copy()
    ncell = len(cells)
    cell_card_t = np.full(ncell, np.inf, np.float32)
    ct = W["ci"][card_tiles]
    np.minimum.at(cell_card_t, ct, tfirst[card_tiles])
    cell_col = rng.integers(0, len(SLOP), ncell)
    # the dive target: the phone of a tiny person near the frame centre at 47.0
    cam47 = (968.0, 560.0)
    dv = np.hypot(xy[card_tiles, 0] - cam47[0], xy[card_tiles, 1] - cam47[1])
    dive = int(card_tiles[np.argmin(dv)])
    hs = 0.5 * float(W["sz"][dive])
    dive_z = 1010.0 / (hs * 0.8)
    F[dive, F[dive] > 46.6] = np.inf                  # its feed stays put while the camera dives in

    cam = Cam(tt, eye_xy, (float(xy[dive, 0]), float(xy[dive, 1])), dive_z)

    # cog angles and walk offsets as pure functions of T (integrated on a grid)
    grid = np.linspace(S.start - 1, S.end + 1, int((S.dur + 2) * 240))
    stop_t = []
    for m_ in range(3):
        c = A.MACHINE_C[m_]
        stop_t.append(tt["crash"] + math.hypot(c[0] - ox, c[1] - oy) / V1)
    stop_t.append(tt["crash"] + 1.1)       # corner processions

    def run(m_, t):
        tw = tt["wake"][m_] if m_ < 3 else tt["pull0"] + 1.2
        up = smoothstep(tw, tw + 0.55, t)
        kick = 0.35 * math.exp(-max(0.0, t - tw - 0.18) / 0.25) * (t > tw + 0.05)
        down = 1.0 - smoothstep(stop_t[m_], stop_t[m_] + 0.9, t)
        return (up + kick) * down

    disc_ang = np.stack([_integrate(grid, lambda t, d=d: d["w"] * run(d["m"], t)) for d in A.DISCS])
    mach_run = np.stack([_integrate(grid, lambda t, m_=m_: run(m_, t)) for m_ in range(4)])

    procs = _proc_setup(W)
    for pr in procs:
        pr["t_slop"] = tt["crash"] + math.hypot(pr["cx"] - ox, pr["cy"] - oy) / V1

    # dome / globe / sundial / tree / letter tiles
    dcx, dcy, dr = A.DOME
    rr = np.hypot(xy[:, 0] - dcx, xy[:, 1] - dcy)
    dome = np.nonzero((rr < dr - 1.5) & (xy[:, 1] < dcy - 1.0) & (W["grp"] == 0))[0]
    lat = np.arcsin(np.clip((dcy - xy[dome, 1]) / dr, 0, 1))
    lon = np.arcsin(np.clip((xy[dome, 0] - dcx) / (dr * np.maximum(np.cos(lat), 0.08)), -1, 1))
    gx, gy, gr = A.GLOBE
    gl_i = np.nonzero((np.hypot(xy[:, 0] - gx, xy[:, 1] - gy) < gr) & (W["grp"] == 0))[0]
    sx, sy, sr = A.SUNDIAL
    sd_i = np.nonzero((np.hypot(xy[:, 0] - sx, xy[:, 1] - sy) < sr - 0.8) & (xy[:, 1] > sy) & (W["grp"] == 0))[0]
    tx, ty, tr = A.TREE
    leaf = np.nonzero((np.hypot((xy[:, 0] - tx) / 1.2, (xy[:, 1] - ty) / 0.9) < tr + 8) & (W["grp"] == 0) &
                      (base[:, 1] > base[:, 0] * 1.05))[0]
    letters = A.titulus_letters()
    tit_i = np.nonzero(W["tit"])[0]
    tit_t = np.full(len(tit_i), np.inf, np.float32)
    for wi, li, xa, xb in letters:
        m = (xy[tit_i, 0] >= xa - 1.0) & (xy[tit_i, 0] < xb + 1.0) & \
            (np.abs(xy[tit_i, 0] - A.TITULI[wi][1]) < 140)
        tit_t[m] = np.minimum(tit_t[m], tt["wake"][wi] + 0.12 + 0.055 * li)

    # per-tile static helpers
    lum = base @ np.array([0.2126, 0.7152, 0.0722], np.float32)

    tile_col = rng.integers(0, len(SLOP) * 4, (n, 8)).astype(np.int16)
    glitch = rng.random((n, 8)) < 0.12
    tile_col[card_tiles, 0] = cell_col[W["ci"][card_tiles]]     # the phone's first feed lights its own face
    glitch[card_tiles, 0] = False
    st = dict(tt=tt, wall=wall, W=W, base=base, gold=gold, mach=mach, cam=cam, grid=grid, disc_ang=disc_ang,
              mach_run=mach_run, procs=procs, dome=dome, dome_lat=lat.astype(np.float32),
              dome_lon=lon.astype(np.float32), globe=gl_i, sundial=sd_i, leaf=leaf, tit_i=tit_i, tit_t=tit_t,
              F=F, tfirst=tfirst, cell_card_t=cell_card_t, cell_col=cell_col, first=first, first_cell=first_cell,
              dive=dive, lum=lum, stop_t=stop_t, eye_xy=eye_xy, ncell=ncell, tile_col=tile_col, glitch=glitch,
              flick=rng.random(n).astype(np.float32), ph_speed=(0.6 + rng.random(n) * 1.8).astype(np.float32))
    _export_foley(S, st)
    return st


# ============================================================ foley / music sync
def _export_foley(S, st):
    tt, cam = st["tt"], st["cam"]
    ev_sfx, ev_mus = [], []
    names = ["nations: the crowned walled city wakes, its gatehouse gear train clunks into motion",
             "faiths: the domed temple wakes, dome turns, saints process to the door",
             "philosophies: the philosophers' globe spins, cogs turn in the tree"]
    for m_ in range(3):
        c = A.MACHINE_C[m_]
        x, _ = world_to_screen(cam(tt["wake"][m_]), c[0], c[1])
        pan = screen_pan(float(x))
        ev_sfx.append(dict(t=tt["wake"][m_], kind="machine", strength=0.8, pan=pan, desc=names[m_]))
        ev_sfx.append(dict(t=tt["wake"][m_] + 0.05, kind="gears", strength=0.5, pan=pan,
                           desc=f"gilded cogs turning (to {st['stop_t'][m_] + 0.9:.2f}), grinding to a halt as the "
                                f"slop wave reaches them at {st['stop_t'][m_]:.2f}"))
        ev_mus.append(dict(t=tt["wake"][m_], kind="reveal", strength=0.8, pan=pan,
                           desc=["NATIONES", "FIDES", "PHILOSOPHIAE"][m_]))
    ev_mus.append(dict(t=tt["pull0"], kind="pullback", strength=0.6, pan=0.0, desc="pull back begins (to 32.63)"))
    ev_mus.append(dict(t=tt["picture"], kind="pullback", strength=1.0, pan=0.0,
                       desc="'whole picture': the vast icon of Humanity fully revealed"))
    fx, _ = world_to_screen(cam(tt["slop"]), *st["eye_xy"])
    ev_sfx.append(dict(t=tt["slop"], kind="tile_flip", strength=0.7, pan=screen_pan(float(fx)),
                       desc="ONE tessera (the pupil of the colossal eye) flips into a glowing pastel phone screen"))
    ev_mus.append(dict(t=tt["slop"], kind="flip", strength=0.7, pan=screen_pan(float(fx)), desc="first slop tessera"))
    ev_sfx.append(dict(t=tt["crash"], kind="whoosh", strength=0.8, pan=0.0,
                       desc="camera whips back out of the eye as the slop wave hits (to 38.2)"))
    for tw, nm, org in ((tt["crash"], "crash wave from the left eye", st["eye_xy"]),
                        (tt["wave2"], "second wave from the top-left", (640, 90)),
                        (tt["wave3"], "third wave from the lower right", (1760, 1000))):
        x, _ = world_to_screen(cam(tw), org[0], org[1])
        ev_sfx.append(dict(t=tw, kind="flip_wave", strength=1.0 if tw == tt["crash"] else 0.8,
                           pan=screen_pan(float(x)), desc=f"{nm}: split-flap flips sweep the wall (to {tw + 1.6:.2f})"))
        ev_mus.append(dict(t=tw, kind="wave", strength=1.0 if tw == tt["crash"] else 0.8, pan=screen_pan(float(x)),
                           desc=nm))
    ev_sfx.append(dict(t=tt["until"], kind="noise", strength=0.9, pan=0.0,
                       desc=f"every tile independent: the picture dissolves into noise (to {tt['world']:.2f})"))
    ev_mus.append(dict(t=tt["until"], kind="wave", strength=0.9, pan=0.0, desc="total dissolution"))
    ev_sfx.append(dict(t=47.0, kind="static", strength=0.8, pan=0.0,
                       desc="dive into one slop tessera; its feed melts into full-frame pastel static (to 48.5)"))
    ev_mus.append(dict(t=47.0, kind="push", strength=0.7, pan=0.0, desc="push into one tessera begins"))
    ev_mus.append(dict(t=T_STATIC, kind="push", strength=1.0, pan=0.0, desc="full-frame static (hand-off to m03)"))
    export_events(S, "sfx", ev_sfx)
    export_events(S, "music", ev_mus)
    # per-frame flip counts from the actual schedule (each flip = a split-flap click)
    F = st["F"]
    fr = np.arange(S.n)
    T = S.start + fr / S.fps
    flat = F[np.isfinite(F)]
    fl_i = np.floor((flat - S.start) * S.fps).astype(np.int64)
    ok = (fl_i >= 0) & (fl_i < S.n)
    counts = np.bincount(fl_i[ok], minlength=S.n).astype(np.float32)
    # pan: mean screen x of the tiles flipping in each frame
    xy = st["W"]["xy"]
    rows, cols_ = np.nonzero(np.isfinite(F))
    tf = F[rows, cols_]
    fi = np.floor((tf - S.start) * S.fps).astype(np.int64)
    ok = (fi >= 0) & (fi < S.n)
    rows, fi = rows[ok], fi[ok]
    panx = np.zeros(S.n, np.float32)
    for f in np.unique(fi):
        sel = rows[fi == f]
        c = cam(S.start + f / S.fps)
        x, _ = world_to_screen(c, xy[sel, 0], xy[sel, 1])
        panx[f] = float(np.clip((np.clip(x, -200, 2120).mean() - 960) / 960, -1, 1))
    energy = counts / max(1.0, counts.max())
    export_track(S, "flips", energy, pan=panx, kind="tiles", flips=counts)


# ============================================================ per-frame
def _lookup(grid, arr, T):
    i = np.searchsorted(grid, T)
    i = int(np.clip(i, 1, len(grid) - 1))
    a = (T - grid[i - 1]) / (grid[i] - grid[i - 1])
    return arr[..., i - 1] * (1 - a) + arr[..., i] * a


def _candle_screen(T, cam):
    """the pilgrim's candle pool in screen design units: below the frame during the tilt, then in the frame."""
    rise = smoothstep(22.0, 25.2, T)
    x = 900 + 70 * math.sin(T * 0.31) + 40 * math.sin(T * 0.83)
    y = lerp(1450.0, 760.0, rise) + 30 * math.sin(T * 0.47)
    r = lerp(900.0, 560.0, rise) + 220.0 * smoothstep(26.3, 29.5, T)
    return x, y, r


def _wake_k(tw, T):
    """a machine's own light: 0 asleep .. ~0.85 awake, with a flare at the moment it wakes."""
    if T <= tw - 0.05:
        return 0.0, 0.0
    on = smoothstep(tw - 0.05, tw + 0.4, T) * 0.85
    flare = math.exp(-max(0.0, T - tw) / 0.3) * (T > tw) * min(1.0, (T - tw) / 0.04 if T > tw else 0.0)
    return on, flare


def _light(fc, st, cam, T):
    """per-tile light 0..~1.4 and warm tint weight."""
    W, tt = st["W"], st["tt"]
    xy = W["xy"]
    sx, sy = world_to_screen(cam, xy[:, 0], xy[:, 1])
    cxs, cys, cr = _candle_screen(T, cam)
    fl = 1.0 + 0.035 * math.sin(T * 14.5) + 0.03 * math.sin(T * 23.1 + 1.3) + 0.025 * math.sin(T * 9.7)
    d2 = ((sx - cxs) ** 2 + (sy - cys) ** 2) / (cr * cr)
    pool = np.exp(-d2 * 1.6).astype(np.float32) * fl
    amb = 0.1 + 0.04 * smoothstep(23.0, 26.0, T) + 0.45 * (1.0 - smoothstep(22.2, 23.6, T))
    L = amb + 0.85 * pool
    # the machines wake: each floods with its own light (and spills gold onto the wall round it)
    for m_ in range(3):
        on, flare = _wake_k(tt["wake"][m_], T)
        if on > 0 or flare > 0:
            sel = st["mach"] == m_
            L[sel] = np.maximum(L[sel], 0.18 + on + 0.6 * flare)
            c = A.MACHINE_C[m_]
            spill = np.exp(-(((xy[:, 0] - c[0]) / 150.0) ** 2 + ((xy[:, 1] - c[1]) / 120.0) ** 2))
            L = np.maximum(L, (0.45 * on + 0.35 * flare) * spill.astype(np.float32))
    # the dawn of the whole picture: light spreads out from the crown as the camera pulls back
    t0 = tt["pull0"] - 0.2
    if T > t0:
        R = 1600.0 * ease_in_out(clamp((T - t0) / (tt["picture"] + 0.2 - t0)), 1.6)
        dc = np.hypot(xy[:, 0] - 960, xy[:, 1] - 230)
        dawn = np.clip((R - dc) / 260.0, 0, 1) * 0.95
        L = np.maximum(L, dawn + 0.12 * pool)
    # after the crash, the warm light drains; screens take over
    drain = smoothstep(tt["crash"], tt["crash"] + 3.0, T) * 0.16 + smoothstep(tt["wave2"], tt["world"], T) * 0.46
    L = L * (1.0 - drain)
    # 'Then came the slop.': the basilica holds its breath - the wall dims round the one glowing phone
    k = smoothstep(tt["slop"] - 0.2, tt["slop"] + 0.8, T) * (1.0 - smoothstep(tt["crash"], tt["crash"] + 0.6, T))
    if k > 0:
        ex, ey = st["eye_xy"]
        dd = ((xy[:, 0] - ex) ** 2 + ((xy[:, 1] - ey) * 0.8) ** 2) / (16.0 ** 2)
        L = L * (1.0 - 0.55 * k) + (0.6 * k) * np.exp(-dd).astype(np.float32) * L
    L = L * (1.0 - 0.08 * smoothstep(tt["crash"], tt["crash"] + 0.5, T))
    return L.astype(np.float32), pool


def _gold_col(gl):
    gd, g, glt = mz.SC["gold_d"], mz.SC["gold"], mz.SC["gold_l"]
    gl = gl[:, None]
    return gd + (g - gd) * np.clip(gl * 1.15, 0, 1) + (glt - g) * np.clip(gl - 0.85, 0, 1) * 1.3


def render(fc, st):
    T = fc.T
    tt = st["tt"]
    W = st["W"]
    wall = st["wall"]
    n = wall.n
    xy = W["xy"]
    cam = st["cam"](T)

    col_ = st["base"].copy()
    gold = st["gold"].copy()
    tilt = wall.tilt

    # ---------------------------------------------------------------- the machines' animation (re-colouring)
    run = _lookup(st["grid"], st["mach_run"], T)          # per machine, integrated 'running' time
    # processions (gliding icon figures); candle flames and faces of the corner processions are remembered
    flames, faces = [], []
    for pr in st["procs"]:
        P = pr["P"]
        m_ = P["m"]
        off = run[m_] * P["v"]
        s_ = pr["strip"]
        L = P["L"]
        u = (pr["u"] - off) % L
        px = np.clip((u * A.K).astype(np.int32), 0, s_.shape[1] - 1)
        py = np.clip((pr["v"] * A.K).astype(np.int32), 0, s_.shape[0] - 1)
        c = s_[py, px]
        on = c[:, 3] > 0.5
        idx = pr["idx"][on]
        col_[idx] = c[on, :3]
        gold[idx] = 0.0
        mk = pr["masks"][py, px]
        fl = pr["idx"][on & (mk[:, 0] > 0.5)]
        fa = pr["idx"][on & (mk[:, 1] > 0.5) & (mk[:, 0] < 0.5)]
        if len(fl):
            flames.append((fl, pr))
        if len(fa):
            faces.append((fa, pr))
    # dome ribs turning
    lon = st["dome_lon"] + run[1] * 0.45
    rib = np.cos(lon * 12.0) > 0.88
    ci_ = st["dome"]
    col_[ci_] = np.where(rib[:, None], mz.SC["gold_d"] * 0.9, col_[ci_])
    gold[ci_] = np.where(rib, 0.25, 1.0)
    # the celestial globe (a tilted sphere spinning)
    gx, gy, gr = A.GLOBE
    gi = st["globe"]
    X = (xy[gi, 0] - gx) / gr
    Y = (xy[gi, 1] - gy) / gr
    Z = np.sqrt(np.clip(1 - X * X - Y * Y, 0, 1))
    a = 0.41
    Y2 = Y * math.cos(a) - Z * math.sin(a)
    Z2 = Y * math.sin(a) + Z * math.cos(a)
    lo = np.arctan2(X, Z2) + run[2] * 1.4
    la = np.arcsin(np.clip(-Y2, -1, 1))
    gcol = np.tile(mz.SC["lapis"], (len(gi), 1)) * (0.55 + 0.45 * Z[:, None])
    line = (np.abs(np.sin(lo * 3)) < 0.16) | (np.abs(la) < 0.08) | (np.abs(np.abs(la) - 0.41) < 0.06)
    gcol[line] = mz.SC["gold"] * (0.7 + 0.3 * Z[line, None])
    ecl = np.abs(la - 0.4 * np.sin(lo)) < 0.1
    gcol[ecl] = mz.SC["porphyry_l"]
    col_[gi] = gcol
    gold[gi] = line.astype(np.float32)
    # the sundial's shadow sweeping
    sx_, sy_, sr_ = A.SUNDIAL
    si = st["sundial"]
    ang = 0.4 + (run[2] * 0.5) % 2.3
    dx_, dy_ = xy[si, 0] - sx_, xy[si, 1] - sy_
    dist_line = np.abs(dx_ * math.sin(ang) - dy_ * math.cos(ang))
    sh = (dist_line < 1.4) & ((dx_ * math.cos(ang) + dy_ * math.sin(ang)) > 0)
    col_[si[sh]] = mz.SC["ink"]
    # leaves breathing in a wind that only blows while the machine runs
    li = st["leaf"]
    wk = smoothstep(tt["wake"][2], tt["wake"][2] + 0.5, T)
    col_[li] *= (1.0 + 0.14 * wk * np.sin(T * 2.7 + xy[li, 0] * 0.21 + xy[li, 1] * 0.13))[:, None]
    # tituli set letter by letter on their words
    ti = st["tit_i"]
    k = np.clip((T - st["tit_t"]) / 0.12, 0, 1)
    pop = np.exp(-np.clip(T - st["tit_t"] - 0.12, 0, None) / 0.3) * (k > 0)
    col_[ti] = col_[ti] * (1 - k[:, None]) + mz.SC["marble"] * k[:, None]
    gold[ti] = 0.0

    # ---------------------------------------------------------------- light, gold, glint
    L, pool = _light(fc, st, cam, T)
    lx, ly, li_ = mz.candle(T, seed=2, flicker=0.1)
    # rotating cogs: their glass tilts turn with them
    theta = _lookup(st["grid"], st["disc_ang"], T)
    tl_ = tilt
    if len(A.DISCS):
        tl_ = tilt.copy()
        g = W["grp"]
        for k_, th in enumerate(theta):
            m = g == k_ + 1
            c, s = math.cos(th), math.sin(th)
            t0 = tilt[m]
            tl_[m, 0] = t0[:, 0] * c - t0[:, 1] * s
            tl_[m, 1] = t0[:, 0] * s + t0[:, 1] * c
    tdot = tl_[:, 0] * lx + tl_[:, 1] * ly
    facing = np.clip(0.58 + tdot * 0.2, 0, 1.5)
    gl = facing * L * li_
    gcol = _gold_col(gl)
    warm = np.clip(pool, 0, 1)[:, None] * np.array([0.0, -0.05, -0.14], np.float32)
    lit = col_ * (0.92 + 0.14 * facing[:, None]) * (L[:, None] * (1 + warm))
    col_ = lit * (1 - gold[:, None]) + gcol * gold[:, None]
    spec = np.clip(tdot * 0.35 - 0.41, 0, None)
    glint = spec * spec * 12.0 * L * li_ * (0.35 + 1.6 * gold)
    for m_ in range(3):
        _, flare = _wake_k(tt["wake"][m_], T)
        if flare > 0.01:
            sel = st["mach"] == m_
            glint[sel] = glint[sel] * (1.0 + 4.0 * flare) + 0.25 * flare * gold[sel] * (wall.hash[sel] > 0.8)
    if len(ti):
        col_[ti] += (pop * 0.9)[:, None] * mz.SC["gold_l"]
    col_ *= (1.0 + 0.04 * wall.jit[:, :1])

    # ---------------------------------------------------------------- the slop
    col_pre = col_.copy() if tt["until"] < T < 47.2 else None
    vs = np.ones(n, np.float32)
    em = np.zeros(n, np.float32)
    ph = np.zeros(n, np.float32)
    if T >= tt["slop"] - 0.01:
        F = st["F"]
        FD = 0.2
        nflip = (F <= T).sum(1)                       # flips completed or started
        flipping = nflip > 0
        idx = np.nonzero(flipping)[0]
        k_last = nflip[idx] - 1
        t_last = F[idx, k_last]
        p = np.clip((T - t_last) / FD, 0, 1)
        vs[idx] = np.abs(np.cos(math.pi * p))
        after = p >= 0.5
        # the screen colour of each flip (a different feed every time)
        ci = st["tile_col"][idx, k_last] % len(SLOP)
        scr = np.array(SLOP, np.float32)[ci]
        gm = st["glitch"][idx, k_last]
        if gm.any():
            gpick = (st["tile_col"][idx[gm], k_last[gm]] // len(SLOP)) % 2
            scr[gm] = np.array(GLITCH, np.float32)[gpick]
        # the previous state (old colour or previous screen) shows during the first half of the flip
        prev_scr = k_last > 0
        pk = np.maximum(k_last - 1, 0)
        pc = np.array(SLOP, np.float32)[st["tile_col"][idx, pk] % len(SLOP)]
        fl = st["flick"][idx]
        bright = 0.9 + 0.2 * np.sin(T * (5 + 9 * fl) + fl * 40) * np.sin(T * (1.3 + 2 * fl))
        bright *= 1.0 + 0.3 * smoothstep(tt["until"], tt["world"], T) * np.sin(T * 23 + fl * 90)
        flash = 1.0 + 0.8 * np.exp(-np.clip(T - t_last - FD * 0.5, 0, None) / 0.07)
        newc = scr * (0.8 * bright * flash)[:, None]
        oldc = np.where(prev_scr[:, None], pc * 0.8, col_[idx])
        edge = (0.45 + 0.55 * vs[idx])[:, None]
        col_[idx] = np.where(after[:, None], newc, oldc) * edge
        em[idx] = np.where(after | prev_scr, 1.0, 0.0)
        glint[idx] *= np.where(after, 0.0, 1.0)
        ph[idx] = (T * st["ph_speed"][idx] + fl * 7.0).astype(np.float32)
        # the tiny people: each face lit by the feed its own phone shows right now
        part = W["part"]
        ci_t = W["ci"]
        cellc = np.array(SLOP, np.float32)[st["cell_col"]]
        csel = part[idx] == 3
        if csel.any():
            cidx = ci_t[idx[csel]]
            cur = np.where(after[csel][:, None], scr[csel] * bright[csel, None],
                           np.where(prev_scr[csel][:, None], pc[csel], cellc[cidx]))
            cellc[cidx] = cur
        bi = np.nonzero((part >= 0) & (em == 0))[0]
        if len(bi):
            tc = st["cell_card_t"][ci_t[bi]]
            lk = np.clip((T - tc - 0.1) / 0.5, 0, 1)
            on = lk > 0
            if on.any():
                b2 = bi[on]
                cc = cellc[ci_t[b2]]
                w_ = lk[on]
                pt = part[b2]
                kk = np.select([pt == 2, pt == 5, pt == 4, pt == 1], [0.82, 0.7, 0.25, 0.1], 0.0)[:, None] * w_[:, None]
                col_[b2] = col_[b2] * (1 - kk) + cc * kk * 0.95
        # the first screen burns brighter while it is alone
        f0 = st["first"]
        if T < tt["crash"] + 0.4:
            col_[f0] *= 1.3
        # the dissolution: the old picture flashes back for a frame or two (it is fighting), then is lost;
        # corrupt bands of tiles tear across the wall
        if tt["until"] < T < 47.2:
            ghost = max(math.exp(-((T - g) / 0.035) ** 2) for g in (45.22, 45.71, 46.13, 46.52))
            ghost *= 0.62 * (1.0 - smoothstep(45.0, 46.9, T) * 0.7)
            if ghost > 0.02:
                col_ = col_ * (1 - ghost) + col_pre * ghost
            Fi = int(round(T * 24))
            rb = np.random.default_rng(Fi * 7 + 1)
            for _ in range(int(rb.integers(0, 4))):
                yb = cam[1] + (rb.random() - 0.5) * 1080 / cam[2]
                hb = (6 + rb.random() * 34) / max(1.0, cam[2] ** 0.5)
                sel = np.nonzero((np.abs(xy[:, 1] - yb) < hb) & (wall.hash > 0.25))[0]
                gc = np.array(GLITCH, np.float32)[int(rb.integers(0, 2))]
                col_[sel] = col_[sel] * 0.25 + gc * 0.85
                em[sel] = 1.0
    # the candles of the processions: warm flames until the wave reaches them, then each becomes a phone
    for fl, pr in flames:
        ts_ = pr["t_slop"]
        k = float(np.clip((T - ts_) / 0.18, 0, 1))
        fk = 1.0 + 0.18 * np.sin(T * 13.0 + xy[fl, 0] * 0.7) + 0.1 * np.sin(T * 21.0 + xy[fl, 1])
        warm_c = np.array([1.35, 1.02, 0.55], np.float32) * fk[:, None]
        if T < ts_:
            col_[fl] = warm_c
            glint[fl] = 0.0
        else:
            scr = np.array(SLOP, np.float32)[(np.floor(xy[fl, 0] / 40.0).astype(int) + 3) % len(SLOP)]
            vs[fl] = abs(math.cos(math.pi * k))
            col_[fl] = scr * 1.15 if k >= 0.5 else warm_c
            em[fl] = 1.0 if k >= 0.5 else 0.0
            ph[fl] = T * 1.3
    for fa, pr in faces:
        ts_ = pr["t_slop"]
        if T > ts_:
            w_ = min(1.0, (T - ts_ - 0.1) / 0.5)
            if w_ > 0:
                scr = np.array(SLOP, np.float32)[(np.floor(xy[fa, 0] / 40.0).astype(int) + 3) % len(SLOP)]
                col_[fa] = col_[fa] * (1 - 0.6 * w_) + scr * 0.75 * w_
    # ---------------------------------------------------------------- set the tiles
    # far out the tesserae are a few pixels wide: supersample 2x2 so their edges don't crawl as the camera drifts
    ss = 2 if cam[2] < 3.2 else 1
    if ss > 1:
        fc2 = dataclasses.replace(fc, s=fc.s * ss, w=fc.w * ss, h=fc.h * ss)
        img2, emis2 = wall.render(fc2, cam, col_, L, glint, vs=vs, em=em, ph=ph, theta=theta, ldir=(lx, ly),
                                  lod_scale=1.0 / ss)
        img = cv2.resize(img2, (fc.w, fc.h), interpolation=cv2.INTER_AREA)
        emis = cv2.resize(emis2, ((fc.w + 3) // 4, (fc.h + 3) // 4), interpolation=cv2.INTER_AREA)
    else:
        img, emis = wall.render(fc, cam, col_, L, glint, vs=vs, em=em, ph=ph, theta=theta, ldir=(lx, ly))
    # screen glow spills onto the wall
    if T >= tt["slop"] - 0.01:
        e = cv2.GaussianBlur(emis, (0, 0), max(1.0, 1.5 * fc.s * max(1.0, cam[2] ** 0.5)))
        e2 = cv2.GaussianBlur(emis, (0, 0), max(2.0, 6.0 * fc.s * max(1.0, cam[2] ** 0.5)))
        glow = cv2.resize(e * 0.3 + e2 * 0.4, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)
        img = img + glow * (0.5 + 0.25 * smoothstep(tt["crash"], tt["crash"] + 2, T))
    # ---------------------------------------------------------------- the dive: feed -> static
    if T >= 47.0:
        img = _dive(fc, st, cam, img, T)
    # motion blur of the tilt (matches m01's V/48)
    if T < 23.6:
        vx_, vy_, _ = st["cam"].vel(T)
        L_px = abs(vy_ * cam[2]) / 48.0 * fc.s
        if L_px > 1.5:
            k_ = int(round(L_px)) | 1
            img = cv2.blur(img, (1, k_))
    return np.clip(img, 0, 1.5).astype(np.float32)


def _feed_card(ctx, x, y, w, h, seed, T):
    """one post of the feed: avatar + handle bars, a pastel 'generated' picture, reactions. No readable text."""
    rng = np.random.default_rng(seed)
    P9 = SLOP
    rrect(ctx, x, y, w, h, w * 0.05)
    set_color(ctx, (0.96, 0.95, 0.97))
    ctx.fill()
    circle(ctx, x + w * 0.09, y + h * 0.075, w * 0.045)
    set_color(ctx, P9[(seed + 3) % 9])
    ctx.fill()
    for j in range(2):
        rrect(ctx, x + w * 0.17, y + h * (0.05 + j * 0.04), w * (0.42 - j * 0.18), h * 0.022, h * 0.011)
        set_color(ctx, (0.62, 0.6, 0.66) if j == 0 else (0.78, 0.76, 0.8))
        ctx.fill()
    # the picture: a pastel sky, soft hills, a sun with a smile (the uncanny cheer of slop)
    ix, iy, iw, ih = x + w * 0.04, y + h * 0.15, w * 0.92, h * 0.6
    ctx.save()
    rrect(ctx, ix, iy, iw, ih, w * 0.03)
    ctx.clip()
    g = lin_grad(ix, iy, ix, iy + ih, [(0, P9[(seed + 1) % 9]), (0.6, P9[(seed + 4) % 9]), (1, P9[(seed + 7) % 9])])
    ctx.set_source(g)
    ctx.paint()
    kind = seed % 3
    for j in range(3):
        ctx.new_path()
        hy = iy + ih * (0.62 + 0.12 * j)
        ctx.move_to(ix - 5, iy + ih + 5)
        ctx.line_to(ix - 5, hy)
        ctx.curve_to(ix + iw * 0.3, hy - ih * rng.uniform(0.1, 0.3), ix + iw * 0.6, hy + ih * 0.1, ix + iw + 5, hy)
        ctx.line_to(ix + iw + 5, iy + ih + 5)
        ctx.close_path()
        set_color(ctx, shade(P9[(seed + 5 + j) % 9], 0.92 - 0.06 * j))
        ctx.fill()
    sx_, sy_, sr_ = ix + iw * rng.uniform(0.3, 0.7), iy + ih * 0.36, iw * 0.16
    circle(ctx, sx_, sy_, sr_)
    set_color(ctx, (1.0, 0.86, 0.7) if kind else (1.0, 0.72, 0.84))
    ctx.fill()
    for side in (-1, 1):
        circle(ctx, sx_ + side * sr_ * 0.35, sy_ - sr_ * 0.15, sr_ * 0.1)
        set_color(ctx, (0.25, 0.2, 0.3))
        ctx.fill()
    ctx.new_path()
    ctx.arc(sx_, sy_ + sr_ * 0.05, sr_ * 0.55, 0.15 * math.pi, 0.85 * math.pi)
    set_color(ctx, (0.25, 0.2, 0.3))
    ctx.set_line_width(sr_ * 0.1)
    ctx.stroke()
    ctx.restore()
    # reactions: a heart (glitch magenta), counters, share
    hx, hy = x + w * 0.1, y + h * 0.84
    ctx.new_path()
    ctx.move_to(hx, hy + w * 0.03)
    ctx.curve_to(hx - w * 0.06, hy - w * 0.01, hx - w * 0.02, hy - w * 0.05, hx, hy - w * 0.015)
    ctx.curve_to(hx + w * 0.02, hy - w * 0.05, hx + w * 0.06, hy - w * 0.01, hx, hy + w * 0.03)
    set_color(ctx, GLITCH[0])
    ctx.fill()
    for j in range(3):
        rrect(ctx, x + w * (0.2 + j * 0.2), hy - h * 0.012, w * 0.1, h * 0.024, h * 0.012)
        set_color(ctx, (0.7, 0.68, 0.74))
        ctx.fill()
    rrect(ctx, x + w * 0.06, y + h * 0.9, w * 0.7, h * 0.02, h * 0.01)
    set_color(ctx, (0.8, 0.78, 0.82))
    ctx.fill()


def _dive(fc, st, cam, img, T):
    """inside the target tessera: a phone feed scrolling, melting into pastel static that fills the frame."""
    if T >= T_STATIC - 1e-6:
        return pastel_static(fc.w, fc.h, fc.s, T)
    W = st["W"]
    d = st["dive"]
    x, y = W["xy"][d]
    hs = 0.5 * float(W["sz"][d])
    asp = float(W["aspect"][d])
    z = cam[2]
    sx, sy = world_to_screen(cam, x, y)
    hw = hs * z * 0.8
    hh = hw * asp
    if hw < 5:
        return img
    stat_k = smoothstep(47.72, T_STATIC - 0.03, T)
    cv = Canvas(fc)
    ctx = cv.ctx
    ctx.save()
    ctx.translate(sx, sy)
    rrect(ctx, -hw, -hh, 2 * hw, 2 * hh, hw * 0.08)
    ctx.clip()
    set_color(ctx, (0.93, 0.91, 0.95))
    ctx.paint()
    # the feed: posts scrolling faster and faster (the glass is a whole phone now)
    cw = 2 * hw * 0.9
    chh = cw * 1.25
    u = T - 47.0
    scroll = (u * 1.6 + u ** 3 * 2.2) * chh
    step = chh * 1.06
    y0 = -hh + 0.05 * cw - (scroll % step)
    k = int(scroll // step)
    for j in range(-1, int(2 * hh / step) + 3):
        _feed_card(ctx, -cw / 2, y0 + j * step, cw, chh, k + j + 11, T)
    ctx.restore()
    lay = cv.rgba()
    a = lay[..., 3:4]
    out = img * (1 - a) + lay[..., :3] * 0.92
    if stat_k > 0:
        stat = pastel_static(fc.w, fc.h, fc.s, T)
        k2 = (stat_k * a[..., 0])[..., None]
        out = out * (1 - k2) + stat * k2
    return out


def post(fc, st):
    T = fc.T
    tt = st["tt"]
    d = {}
    if T >= tt["crash"]:
        k = smoothstep(tt["crash"], tt["crash"] + 1.5, T)
        d.update(bloom=0.38 + 0.2 * k, bloom_thresh=0.74 - 0.08 * k,
                 ca=1.2 * k + 1.8 * smoothstep(tt["until"], tt["world"], T))
    if T >= 47.6:
        # hand-off frames: pure static, no film look (m03's frame 0 matches)
        k = smoothstep(47.6, T_STATIC, T)
        d.update(bloom=d.get("bloom", 0.38) * (1 - k), vignette=0.30 * (1 - k), grain=0.018 * (1 - k),
                 ca=d.get("ca", 0.0) * (1 - k))
    return d

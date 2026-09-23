"""s05b - Recursive Balkanization (100.5-125.5).  owner: Scene05b

NATION (100.5-107.8)  scenes/s05b_nation.py (+ s05b_mapk.py numba fractal-map kernel)
FAITH  (107.8-114.6)  scenes/s05b_faith.py
PHILO  (114.6-125.5)  scenes/s05b_philo.py (+ s05b_head.py sculpted marble head): shatter, crackpots in the void,
                      K07 galaxy pull-back with the Commander -> storm swirl on dusk1 (hand-off to s06)
Section changes are "balkanizing" cuts: the old frame itself cracks into Voronoi shards that fly/fall away.
"""
import math

import numpy as np

from vx import *
from vx import foley
from scenes.s05b_common import T0, lg, shard_cells, draw_shards
from scenes.s05b_nation import Nation
from scenes.s05b_faith import Faith
from scenes.s05b_philo import Philo, T_GAL

POST = dict(bloom=0.4, grain=0.03, vignette=0.25)

X_NF = (107.32, 107.78)      # nation -> faith shatter
X_FP = (113.80, 114.14)      # faith -> philosophy shatter


def setup(S):
    st = {"nation": Nation(S), "faith": Faith(S), "philo": Philo(S)}
    st["cells_nf"] = shard_cells(51, 30)
    st["cells_fp"] = shard_cells(77, 24)
    ev = st["nation"].events() + st["faith"].events() + st["philo"].events()
    ev += [dict(t=X_NF[0], kind="shatter", strength=0.9, pan=0.0,
                desc="the whole image of the Nation cracks into shards that fly past camera -> temple"),
           dict(t=X_FP[0], kind="shatter", strength=0.8, pan=0.0,
                desc="the temple image cracks into shards that fly past camera -> agora")]
    foley.export_events(S, "hits", ev)
    _export_flags(S, st)
    return st


def _export_flag(S, name, tr, f0, gain, fn):
    """flag flutter for Foley in scene frames: fn(T) -> (visibility 0..1, screen x). Silent while unseen."""
    n = S.n
    e, sn, fh, pv = (np.zeros(n, np.float32) for _ in range(4))
    for k in range(n):
        T = S.start + k / S.fps
        v, x = fn(T)
        pv[k] = np.clip((x - W / 2) / (W / 2), -1, 1)
        i = k - f0
        if 0 <= i < tr.n and v > 0:
            e[k] = tr.energy[i] * v
            sn[k] = tr.snap[i] * (v > 0.4)
            fh[k] = tr.flap_hz[i]
    ev = [[S.start + k / S.fps, float(sn[k]), float(pv[k])] for k in np.nonzero(sn > 0)[0]]
    foley.export_track(S, name, e, pv, gain=gain, kind="flag", events=ev, flap_hz=fh, snap=sn)


def _onscreen(x, y, size, margin=0.0):
    inside = -margin < x < W + margin and -size * 1.2 < y < H + size
    return float(inside) * clamp(size / 140.0, 0.15, 1.0)


def _export_flags(S, st):
    na, fa, ph = st["nation"], st["faith"], st["philo"]

    def f_nation(T):
        t = T - S.start
        cam = na.camera(t)
        x, y = na.proj(cam, 0.0, 0.0)
        size = 0.52 * cam[2]
        v = _onscreen(x, y, size, 200) if 100.6 <= T < X_NF[0] + 0.2 else 0.0
        return v, float(x)

    from scenes.s05b_faith import cam_at, FLAG_XZ

    def f_faith(T):
        c = cam_at(T)
        (x, y), z = c.proj(np.array([FLAG_XZ[0], 1.2, FLAG_XZ[1]]))
        size = 2.44 * c.scale(z)
        v = _onscreen(x, y, size) if X_NF[1] - 0.05 <= T < X_FP[0] + 0.2 else 0.0
        return v, float(x)

    def f_crown(T):
        i = int(round((T - ph.flag_T0) * 24))
        if T < 114.27 or i < 0 or i >= ph.flag.n:
            return 0.0, W / 2
        bx, by, _ = ph._pole_fn(i)
        _, z, _ = ph.frag_state(ph.frags[ph.crown_i], T)
        x, y, eff = ph.to_screen(ph.camera(T), (bx, by), z)
        return _onscreen(x, y, ph.flag_pole * eff) * (1 - smoothstep(124.2, 124.9, T)), float(x)

    def f_cmd(T):
        x, y, s = ph.cmd_xy(T)
        v = 1.0 if T_GAL <= T and y < H + ph.cmd_h * s and x > -ph.cmd_h * 0.8 * s else 0.0
        return v, float(x + 150 * s)

    _export_flag(S, "nation_flag", na.flag, 0, 0.5, f_nation)
    _export_flag(S, "temple_flag", fa.flag, fa.flag_f0, 0.45, f_faith)
    _export_flag(S, "crown_flag", ph.flag, int(round((ph.flag_T0 - S.start) * S.fps)), 0.4, f_crown)
    _export_flag(S, "commander_flag", ph.cmd_flag, int(round((ph.cmd_T0 - S.start) * S.fps)), 0.8, f_cmd)


def _shatter(fc, st, old_key, new_key, cells, span, seed, origin):
    p = (fc.T - span[0]) / (span[1] - span[0])
    new = st[new_key].render(fc)
    old = st[old_key].render(fc)
    cv = Canvas(fc)
    cv.paint_array(new)
    draw_shards(cv.ctx, old, cells, p, fc.s, seed=seed, origin=origin)
    return cv.rgb()


def render(fc, st):
    T = fc.T
    if T < X_NF[0]:
        return st["nation"].render(fc)
    if T < X_NF[1]:
        return _shatter(fc, st, "nation", "faith", st["cells_nf"], X_NF, 5, (960, 520))
    if T < X_FP[0]:
        return st["faith"].render(fc)
    if T < X_FP[1]:
        return _shatter(fc, st, "faith", "philo", st["cells_fp"], X_FP, 9, (1300, 480))
    return st["philo"].render(fc)


def post(fc, st):
    t = fc.T - T0
    k = smoothstep(0.0, 0.35, t)
    d = dict(bloom=0.4 * k, vignette=0.25 * k, grain=0.012 + 0.018 * k)
    for a, b in (X_NF, X_FP):
        if a - 0.05 <= fc.T <= b + 0.1:
            d["ca"] = 3.0 * math.sin(math.pi * clamp((fc.T - a) / (b - a)))
    if 114.59 <= fc.T <= 115.1:
        d["ca"] = 4.0 * math.exp(-(fc.T - 114.6) * 6)
    # the hand-off frame to s06 wants a clean storm: settle the look over the last second
    return d

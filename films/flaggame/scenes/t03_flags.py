"""t03 the six primordial Flags (owner T03Genesis).

Six real cloth sims (vx.flag.Flag.simulate), one per vertex of the central hexagon. Each sim runs in its
own vertical plane (sim +x = radially outward, sim z = tangential), in flashback STORY time (tau, 24 fps),
and is then placed in the 3D world and projected through the t03 perspective camera every frame, so the
cloth is a true 3D object seen by an orbiting camera (lighting follows the world light).
Wind: a gentle radial breath (genesis) -> a tangential vortex (the stench) -> gale; at the hurricane the
poles are torn out of the mud and fly up into the storm. Cloth drawn ONLY through vx.flag.
"""
import math

import numpy as np

from vx.flag import Flag, FlagTrack, draw_cloth, draw_pole, POLE_W
from scenes.t03_opart import tau_of, R_CELL

FPS = 24
N_FLAGS = 6
POLE_W_UNITS = 1.9                    # pole length in world units (hex cell radius = 1)
SIM_POLE = 420.0
K = POLE_W_UNITS / SIM_POLE           # world units per sim px
TAU_END = 16.5
SPROUT_T = 43.33                      # global: poles sprout (genesis)
UNFURL_T = 45.55                      # global: "FLAGS" - the cloths burst open around the ring
RIP_T = 58.78                         # global: the first pole torn out of the mud (hurricane 59.34)
LIGHT_W = np.array([-0.55, 0.72, 0.42])
VERSION = "v4"


def alpha_k(k):
    return math.radians(60.0 * k)


def base_world(k):
    a = alpha_k(k)
    return np.array([R_CELL * math.cos(a), 0.0, R_CELL * math.sin(a)])


def frame_axes(k):
    a = alpha_k(k)
    ax = np.array([math.cos(a), 0.0, math.sin(a)])           # sim +x: radially outward
    bz = np.array([-math.sin(a), 0.0, math.cos(a)])          # sim +z: tangential (a x up)
    return ax, bz


def _tau(T):
    return tau_of(T)


def unfurl_tau(k):
    return _tau(UNFURL_T) + 0.11 * k


def rip_tau(k):
    order = (0, 3, 1, 4, 2, 5)
    return _tau(RIP_T) + 0.075 * order.index(k)


def _wind(k, i):
    """sim-frame wind (x radial, y down, z tangential) in design px/s on the canonical pole"""
    tau = i / FPS
    T_st = _tau(57.34)
    T_hu = _tau(59.34)
    rad = 330.0 + 60.0 * math.sin(tau * 0.7 + k)
    # stench: the breath turns into a vortex around the hexagon (tangential), rising to a gale
    v = min(1.0, max(0.0, (tau - T_st + 0.4) / 2.2))
    v = v * v * (3 - 2 * v)
    g = min(1.0, max(0.0, (tau - T_hu + 0.9) / 0.9))
    tang = 700.0 * v + 900.0 * g
    radial = rad * (1.0 - 0.75 * v) - 250.0 * g
    return (radial, -140.0 * g, tang)


def _pole(k, i):
    tau = i / FPS
    tr = rip_tau(k)
    if tau <= tr:
        return (0.0, 0.0, 0.0)
    s = tau - tr
    # torn out: up and outward, accelerating, tumbling
    up = 900.0 * s * s + 260.0 * s
    out = 520.0 * s * s
    ang = (1.5 + 0.3 * (k % 3)) * s * s * (1 if k % 2 == 0 else -1)
    return (out, -up, ang)


def _furl(k, i):
    tau = i / FPS
    u = (tau - unfurl_tau(k)) / 0.38
    return 1.0 if u <= 0 else max(0.0, 1.0 - u)


def setup_flags(S):
    n = int(TAU_END * FPS) + 1
    tracks = []
    for k in range(N_FLAGS):
        p = S.cache / f"t03_flag{k}_{VERSION}.npz"
        if p.exists():
            tracks.append(FlagTrack.load(p))
            continue
        fl = Flag(pole=SIM_POLE, seed=31 + k * 7, side=1, turbulence=0.8, flutter=0.9)
        tr = fl.simulate(n, lambda i, k=k: _pole(k, i), lambda i, k=k: _wind(k, i), fps=FPS, substeps=10,
                         warmup=48, furl_fn=lambda i, k=k: _furl(k, i))
        tr.save(p)
        tracks.append(tr)
    return tracks


def sprout(k, T):
    """0..1 pop of flag k (poles shoot up out of the op-art at genesis)"""
    t0 = SPROUT_T + 0.09 * k
    u = (T - t0) / 0.55
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    c1 = 1.70158 * 1.3
    q = u - 1.0
    return 1.0 + (c1 + 1) * q ** 3 + c1 * q * q


def _to_world(k, P, pop=1.0):
    """sim-frame points (..., 3) [x, y(down), z] -> world"""
    ax, bz = frame_axes(k)
    b0 = base_world(k)
    x = P[..., 0:1] * K * pop
    y = -P[..., 1:2] * K * pop
    z = P[..., 2:3] * K * pop
    return b0 + x * ax + y * np.array([0.0, 1.0, 0.0]) + z * bz


def flag_geom(tracks, k, cam, T):
    """screen geometry of flag k at global T: dict(verts, base, top, width, depth, light) or None"""
    pop = sprout(k, T)
    if pop <= 0.0:
        return None
    tr = tracks[k]
    i = min(max(_tau(T) * FPS, 0.0), tr.n - 1.0)
    V, (px, py, pa) = tr._frame(i)
    Lp = tr.meta["pole"]
    pb = np.array([px, py, 0.0])
    pt = np.array([px + math.sin(pa) * Lp, py - math.cos(pa) * Lp, 0.0])
    Wv = _to_world(k, V.astype(np.float64), pop)
    Wb = _to_world(k, pb, pop)
    Wt = _to_world(k, pt, pop)
    sv = cam.project(Wv)
    sb = cam.project(Wb)
    st = cam.project(Wt)
    if sb[2] <= 0.05 or st[2] <= 0.05:
        return None
    sc = cam.f / sb[2]
    zc = (sb[2] - sv[..., 2]) * sc                # toward the camera, in screen px
    VW = np.empty_like(sv)
    VW[..., 0] = sv[..., 0] - 0.06 * zc           # cancel vx.flag's oblique VIEW offset: exact projection
    VW[..., 1] = sv[..., 1] + 0.24 * zc
    VW[..., 2] = zc
    pw = max(1.2, POLE_W * POLE_W_UNITS * sc * pop * 1.15)
    x0, y0 = float(VW[..., 0].min()), float(VW[..., 1].min())
    x1, y1 = float(VW[..., 0].max()), float(VW[..., 1].max())
    return dict(k=k, verts=VW, base=(float(sb[0]), float(sb[1])), top=(float(st[0]), float(st[1])), width=pw,
                depth=float(sb[2]), light=cam.light_view(LIGHT_W), pop=pop,
                bbox=(min(x0, sb[0], st[0]), min(y0, sb[1], st[1]), max(x1, sb[0], st[0]), max(y1, sb[1], st[1])),
                cloth_bbox=(x0, y0, x1, y1))


def draw_geom(ctx, g, glow=0.0, alpha=1.0):
    bx, by = g["base"]
    tx, ty = g["top"]
    draw_pole(ctx, bx, by, tx, ty, g["width"], alpha, g["light"])
    seg = (tx + (bx - tx) * 0.75, ty + (by - ty) * 0.75, tx, ty, g["width"])
    draw_cloth(ctx, g["verts"], alpha, glow, g["light"], pole_seg=seg)


def all_geoms(tracks, cam, T):
    gs = [flag_geom(tracks, k, cam, T) for k in range(N_FLAGS)]
    gs = [g for g in gs if g is not None]
    gs.sort(key=lambda g: -g["depth"])
    return gs

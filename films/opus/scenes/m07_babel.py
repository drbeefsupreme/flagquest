"""m07 - BABEL (131.5-146.0). Owner: M07.

The Babel wall of the basilica (after the Monreale / San Marco Babel mosaics and the Pantocrator conchs), seen by
candlelight, one wall 3000 x 1500 design units with a nave floor in front of it (see m07_ruin for the world).

    131.50 babel_rise  m06's glittering dust (cache/m06/handoff_dust.npz, or a matching fake) swirls on, is caught by
                       a vortex round the tower's axis and spirals up; every speck that reaches its socket becomes a
                       tessera: the Tower of Babel sets itself tier by tier (7 tiers, 132.2-133.9), its bricks
                       glowing phone screens; the camera cranes up with the building front.
    134.00 egregore    the vortex tops out in the conch: LORD EGREGORE sets himself - the static face flares on,
                       the halo of screens spins like a loading wheel, the arms spread to the hands, the billboards
                       hung from the sleeves light up; the camera looks up, then pulls back to the whole programme.
    136.00 L03         Lutie (alarm) + Crocus (serene, blessing) stand tiny on the nave floor at the tower's foot, by
                       the votive candle; slow push toward them through C04 (139.40).
    141.80 collapse    the tower's top hinges over to the right and breaks up; Egregore's static drains away (the
                       face empties top-down, billboards collapse to a line, the halo stops); then the ENTIRE mosaic
                       sheds off the wall - sheets peel, break into ~60k tumbling tesserae, pour onto the floor,
                       bounce, skid and heap (deterministic physics: m07_phys / m07_ruin); screens die mid-air.
    145.00 silence     locked camera: bare lime plaster and the red sinopia (the master drew the colossus WITH a
                       face), the heaps, one candle. Nothing moves but the flame. Hand-off to m08 at 146.0.

Render: python -m vx.render m07 --at 0.5,2.6,5,8.5,10.6,11.5,12.5,14 (stills) | --scale 0.5 --step 2 --workers 4
Foley: films/opus/cache/foley/m07_sfx_events.json + m07_avalanche.npz (nothing after 145.0).
"""
import math
from pathlib import Path

import cairo
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from vx import *
from vx import mosaic as mz
from vx import foley
from vx import icon

from . import m07_art as A
from . import m07_phys as P
from . import m07_ruin as R
from . import m07_workers as WK

POST = dict(bloom=0.42, bloom_thresh=0.66, bloom_radius=26.0, vignette=0.32, grain=0.016)

T0 = 131.5
T_EGR = 134.0
T_L03 = 136.0
T_C04 = 139.4
T_COL = R.T_COLLAPSE
T_SIL = 145.0
T_END = 146.0
TIER_T = [132.25 + 0.27 * k for k in range(7)]         # each tier completes
T_CROWN = 134.05

# ---------------------------------------------------------------- camera keys (global T, cx, cy, zoom, eye height)
CAM_KEYS = [
    (131.50, 1430.0, 1150.0, 1.55),     # m06's dust over the tower's (future) foot
    (132.10, 1440.0, 1110.0, 1.47),
    (133.00, 1475.0, 900.0, 1.22),      # crane up with the building front
    (133.80, 1500.0, 700.0, 1.05),
    (134.40, 1500.0, 566.0, 0.965),     # egregore: look up into the conch (top of frame = top of wall)
    (135.10, 1500.0, 600.0, 0.90),
    (136.00, 1500.0, 862.0, 0.665),     # the whole programme; Lutie + Crocus tiny at the foot
    (137.10, 1492.0, 880.0, 0.68),
    (139.40, 1380.0, 1150.0, 1.02),     # push toward the pair through L03 ...
    (141.60, 1345.0, 1255.0, 1.30),     # ... C04, close on them under the tower's base
    (141.84, 1345.0, 1256.0, 1.29),
    (142.42, 1500.0, 866.0, 0.665),     # the fall: recoil to the whole wall
    (143.60, 1560.0, 905.0, 0.70),
    (144.85, R.CAM_END["cx"], R.CAM_END["cy"], R.CAM_END["zoom"]),     # settle on the ruin; locked
    (146.00, R.CAM_END["cx"], R.CAM_END["cy"], R.CAM_END["zoom"]),
]
_KT = np.array([k[0] for k in CAM_KEYS])
_KX = PchipInterpolator(_KT, [k[1] for k in CAM_KEYS])
_KY = PchipInterpolator(_KT, [k[2] for k in CAM_KEYS])
_KZ = PchipInterpolator(_KT, np.log([k[3] for k in CAM_KEYS]))

# figures on the nave floor, in front of the tower's foot
FIG_Z = 240.0
LUTIE = dict(x=1250.0, h=205.0)
CROCUS = dict(x=1400.0, h=235.0)


def sst(a, b, x):
    """vectorised smoothstep"""
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def cam_at(T):
    T = min(max(T, _KT[0]), _KT[-1])
    cx, cy, z = float(_KX(T)), float(_KY(T)), math.exp(float(_KZ(T)))
    sh = (0.0, 0.0)
    if 141.84 < T < 144.4:
        # the wall's fall shakes the floor (and the eye): hard at the topple impact, fading
        k = pulse(T, 142.45, 0.35) * 1.0 + pulse(T, 143.1, 0.6) * 0.55 + 0.25 * smoothstep(141.84, 142.0, T) * (
            1 - smoothstep(143.6, 144.4, T))
        a = 7.0 * k / z
        sh = (a * fbm1(T * 11.0, 3, 3), a * fbm1(T * 13.0, 9, 3))
    return R.camera(cx, cy, z, he=R.CAM_END["he"], shake=sh), (cx, cy, z)


# ================================================================== setup
def setup(S):
    r = R.load_ruin(S.cache)
    lay = r.lay
    n = len(lay)
    st = dict(r=r, n=n)
    # ---------------------------------------------------------------- set times of what the dust builds
    zone = r.zone
    ts = np.full(n, -1.0)
    rng = np.random.default_rng(71)
    tw = A.tiers()
    for k, (xc, hw, yb, yt) in enumerate(tw):
        m = (zone == A.Z_TIER0 + k) | ((zone == A.Z_LADDER) & (r.obj == max(0, k - 1)))
        f = np.clip((yb - lay.xy[m, 1]) / (yb - yt + 20), 0, 1)
        ts[m] = TIER_T[k] - 0.32 * (1 - f) + rng.normal(0, 0.03, m.sum())
    for zz in (A.Z_CROWN, A.Z_CRANE):
        m = zone == zz
        ts[m] = T_CROWN + rng.uniform(-0.12, 0.08, m.sum())
    eg = np.isin(zone, A.EGREGORE_ZONES)
    hx, hy = A.HALO_C
    d = np.hypot(lay.xy[:, 0] - hx, lay.xy[:, 1] - hy)
    ts_e = T_EGR + 0.02 + 0.30 * np.clip(d / 260.0, 0, 1)
    arm = np.isin(zone, [A.Z_ARM_L, A.Z_ARM_R, A.Z_SLEEVE_L, A.Z_SLEEVE_R, A.Z_HAND_L, A.Z_HAND_R])
    ts_e = np.where(arm, T_EGR + 0.22 + 0.85 * np.clip((np.abs(lay.xy[:, 0] - A.AX) - 180) / 720, 0, 1), ts_e)
    cab = zone == A.Z_CABLE
    ts_e = np.where(cab, T_EGR + 1.10 + 0.45 * np.clip((lay.xy[:, 1] - 420) / 900, 0, 1), ts_e)
    ts[eg] = ts_e[eg] + rng.normal(0, 0.025, eg.sum())
    st["ts"] = ts
    built = ts > 0
    st["built"] = built
    # ---------------------------------------------------------------- per-tile animation attributes
    st["emat"] = r.emat
    st["hash"] = mz._hash_uniform(np.arange(n), 5)
    # billboards: local (u, v) inside each board for the CRT switch-off
    boards = r.emat == A.M_BOARD
    bkey = zone * 256 + r.obj
    buv = np.zeros((n, 2), np.float32)
    boff = np.zeros(n, np.float32)
    bph = np.zeros(n, np.float32)
    for key in np.unique(bkey[boards]):
        m = boards & (bkey == key)
        x0, y0 = lay.xy[m].min(0)
        x1, y1 = lay.xy[m].max(0)
        buv[m, 0] = (lay.xy[m, 0] - (x0 + x1) / 2) / max(1.0, (x1 - x0) / 2)
        buv[m, 1] = (lay.xy[m, 1] - (y0 + y1) / 2) / max(1.0, (y1 - y0) / 2)
        boff[m] = 141.95 + 0.6 * rng.uniform()
        bph[m] = rng.uniform(0, 10)
    st["buv"], st["boff"], st["bph"] = buv, boff, bph
    # screens keep glowing as they fall (a rain of dying phones), then flash and go dark - mid-air or on the heap
    f = r.fall
    st["t_die"] = np.minimum(f.td + rng.uniform(0.35, 1.4, n), 144.45 - rng.uniform(0, 0.3, n))
    st["rgb_lin"] = mz.srgb_to_lin(r.rgb)
    # the cities' windows come on after the tower has risen (in m06's dark they are dead glass)
    city = np.isin(zone, [A.Z_CITY_L, A.Z_CITY_R]) & (r.emat == A.M_SCREEN)
    t_city = np.full(n, -1.0)
    t_city[city] = 132.9 + 1.5 * rng.uniform(0, 1, city.sum()) ** 0.7
    st["t_city"] = t_city
    # ---------------------------------------------------------------- the dust (m06 hand-off) and its flights
    st["dust"] = _dust_setup(S, st, r)
    # ---------------------------------------------------------------- the two figures (vx.icon, tessellated)
    st["figs"] = _figs_setup(S)
    st["candle"] = R.candle_tiles()
    st["workers"] = WK.Workers(r, TIER_T, T_CROWN)
    # ---------------------------------------------------------------- bed: where the setting bed hides the plaster
    st["bed"] = _bed_chart(S, r, ts)
    _export_foley(S, st)
    _export_handoff(S, st)
    return st


# ---------------------------------------------------------------- dust
def _dust_setup(S, st, r):
    lay = r.lay
    cam0, _ = cam_at(T0)
    p = Path(S.cache).parent / "m06" / "handoff_dust.npz"
    rng = np.random.default_rng(606)
    if p.exists():
        z = np.load(p)
        sxy = z["xy"].astype(np.float64)
        svel = z["vel"].astype(np.float64)
        ssz = z["size"].astype(np.float64)
        srgb = np.clip(z["rgb"].astype(np.float32), 0, 1)
        src = "m06"
    else:
        # a stand-in that matches M06's description: specks swirling CCW (up on the right) round (960, 560)
        nd = 26000
        rr = 700 * np.sqrt(rng.uniform(0.002, 1, nd)) ** 1.15
        th = rng.uniform(0, 2 * np.pi, nd)
        sxy = np.stack([960 + rr * np.cos(th) * 1.25, 560 + rr * np.sin(th) * 0.8], 1)
        om = -(0.35 + 1.3 / (1 + rr / 160))
        svel = np.stack([-(sxy[:, 1] - 560) * om * 1.25 / 0.8, (sxy[:, 0] - 960) * om * 0.8 / 1.25], 1)
        svel[:, 1] -= 12
        ssz = rng.uniform(1.2, 4.0, nd)
        pal = np.array([A.rgb(c) for c in ("gold", "gold_l", "lapis_l", "porphyry_l", "verdigris", "marble",
                                            "lapis", "gold")], np.float32)
        srgb = pal[rng.integers(0, len(pal), nd)]
        src = "fake"
    nd = len(sxy)
    # place each speck in 3D so that it projects to its m06 screen position at 131.5
    zd = rng.uniform(90, 420, nd)
    D = cam0.ray(sxy)
    tt = (zd - cam0.eye[2]) / np.minimum(D[:, 2], -1e-6)
    P0 = cam0.eye + D * tt[:, None]
    dist = np.linalg.norm(P0 - cam0.eye, axis=1)
    kpx = cam0.F / np.maximum((P0 - cam0.eye) @ cam0.f, 1.0)             # screen px per world unit at that depth
    size_w = ssz / kpx
    # swirl continuation: rotation about the swirl centre (in each speck's depth plane)
    C = cam0.eye + cam0.ray(np.array([[960.0, 560.0]]))[0][None, :] * (
        (zd - cam0.eye[2]) / min(cam0.ray(np.array([[960.0, 560.0]]))[0][2], -1e-6))[:, None]
    rel = P0[:, :2] - C[:, :2]
    vw = svel / kpx[:, None]
    r2 = np.maximum((rel ** 2).sum(1), 25.0)
    omega = (rel[:, 0] * vw[:, 1] - rel[:, 1] * vw[:, 0]) / r2
    drift = vw - np.stack([-rel[:, 1] * omega, rel[:, 0] * omega], 1)
    # targets: every tile the dust builds gets one speck (the rest spiral up and fade into the vortex)
    built = np.nonzero(st["built"])[0]
    order_t = built[np.argsort(st["ts"][built])]
    # specks nearer the vortex axis / lower go first
    key = np.abs(P0[:, 0] - A.AX) * 0.6 + (A.FLOOR - P0[:, 1]) * 0.4 + rng.uniform(0, 250, nd)
    order_s = np.argsort(key)
    na = min(len(order_t), int(nd * 0.8))
    tgt = np.full(nd, -1)
    tgt[order_s[:na]] = order_t[:na]
    ts = np.where(tgt >= 0, st["ts"][np.maximum(tgt, 0)], 0.0)
    dur = np.where(tgt >= 0, rng.uniform(0.75, 1.35, nd), 0.0)
    tf = np.maximum(ts - dur, T0 + 0.12 + rng.uniform(0, 0.25, nd))
    # free specks: caught later, spiral up the column and fade
    free = tgt < 0
    tf[free] = rng.uniform(T0 + 0.5, 133.6, free.sum())
    ts[free] = tf[free] + rng.uniform(1.2, 2.2, free.sum())
    return dict(P0=P0, C=C, omega=omega, drift=drift, size=size_w, rgb=srgb, tgt=tgt, ts=ts, tf=tf, zd=zd,
                spin=rng.normal(0, 1, (nd, 3)), spin_a=rng.uniform(3, 12, nd), src=src, n=nd,
                gold=np.array([c[0] > 0.6 and c[2] < 0.35 for c in srgb]))


AXIS_XZ = (A.AX, 170.0)


def _dust_at(T, st):
    """loose-tile arrays of the specks alive at T (world)"""
    d = st["dust"]
    r = st["r"]
    lay = r.lay
    live = T < d["ts"]
    if not live.any():
        return None
    i = np.nonzero(live)[0]
    tau = T - T0
    # swirl position at T (before the flight) and at the flight start
    def swirl(ii, tt):
        a = d["omega"][ii] * (tt - T0)
        c, s = np.cos(a), np.sin(a)
        rel = d["P0"][ii, :2] - d["C"][ii, :2]
        p = np.stack([d["C"][ii, 0] + c * rel[:, 0] - s * rel[:, 1], d["C"][ii, 1] + s * rel[:, 0] + c * rel[:, 1]],
                     1) + d["drift"][ii] * (tt - T0)[:, None]
        return np.c_[p, d["P0"][ii, 2]]
    tf = d["tf"][i]
    pos = swirl(i, np.minimum(T, tf) + 0 * tf)
    fl = T > tf
    u = np.zeros(len(i))
    if fl.any():
        j = i[fl]
        u[fl] = np.clip((T - d["tf"][j]) / (d["ts"][j] - d["tf"][j]), 0, 1)
        s = u[fl] * u[fl] * (3 - 2 * u[fl])
        p0 = pos[fl]
        ax, az = AXIS_XZ
        r0 = np.hypot(p0[:, 0] - ax, p0[:, 2] - az)
        ph0 = np.arctan2(p0[:, 2] - az, p0[:, 0] - ax)
        tg = d["tgt"][j]
        has = tg >= 0
        # target: the socket on the wall (or, for free specks, high up the column)
        tx = np.where(has, lay.xy[np.maximum(tg, 0), 0], ax + 60 * np.sin(j * 1.7))
        ty = np.where(has, lay.xy[np.maximum(tg, 0), 1], 380.0 + 200 * np.sin(j * 0.9))
        tz = np.where(has, 0.0, az)
        r1 = np.hypot(tx - ax, tz - az)
        ph1 = np.arctan2(tz - az, tx - ax)
        dph = -(np.mod(ph0 - ph1, 2 * np.pi) + 2 * np.pi)
        ph = ph0 + dph * s
        rr = (r0 + (r1 - r0) * s) * (1 - 0.55 * np.sin(np.pi * s) ** 1.4)
        yy = p0[:, 1] + (ty - p0[:, 1]) * (s ** 1.3)
        pos[fl] = np.stack([ax + rr * np.cos(ph), yy, az + rr * np.sin(ph)], 1)
    tg = d["tgt"][i]
    has = tg >= 0
    # size + colour grow from speck to tessera near arrival
    g = np.clip(u, 0, 1) ** 3
    size_t = np.where(has, lay.size[np.maximum(tg, 0)], d["size"][i])
    size = d["size"][i] + (size_t - d["size"][i]) * g
    col = d["rgb"][i] + (np.where(has[:, None], r.rgb[np.maximum(tg, 0)], d["rgb"][i]) - d["rgb"][i]) * (g[:, None]
                                                                                                         ** 0.6)
    ang = np.where(has, lay.ang[np.maximum(tg, 0)], 0.0)
    rot = mz.axis_angle(d["spin"][i] / np.linalg.norm(d["spin"][i], axis=1, keepdims=True) *
                        (d["spin_a"][i] * (T - T0) * (1 - u) ** 1.5)[:, None])
    # polygons: a speck is a small square; near arrival it takes its tessera's shape
    poly, nv = mz._square_polys(pos[:, :2].astype(np.float32), ang.astype(np.float32),
                                (size * 0.88).astype(np.float32))
    near = has & (u > 0.7)
    if near.any():
        k = np.nonzero(near)[0]
        t_ = tg[k]
        rel = lay.poly[t_] - lay.xy[t_][:, None, :]
        sc = (size[k] / lay.size[t_])[:, None, None]
        poly[k] = (pos[k, None, :2] + rel * sc).astype(np.float32)
        nv = nv.copy()
        nv[k] = lay.nv[t_]
    # free specks fade as they rise into the column
    alpha = np.where(has, 1.0, np.clip(1.3 - u * 1.3, 0, 1)) * np.clip((T - T0) * 20 + 1, 0, 1)
    mat = np.where(d["gold"][i], mz.GOLD, mz.GLASS)
    # m06's dust glitters on its own in the dark: each speck twinkles, fading as it becomes a tessera
    F = int(round(T * 24))
    tw_ = mz._hash_uniform(i * 7 + (F // 2) * 104729, 13)
    sp = 0.22 + 1.8 * (tw_ > 0.86) + 0.5 * (tw_ > 0.6)
    # at the cut the specks show exactly m06's displayed colours (their rgb includes m06's glitter): emitted, unlit;
    # that fades as the vortex light takes over, and our own twinkle ramps in over 0.3 s after 131.5
    k_cut = 1.0 - smoothstep(T0, 132.3, T)
    ramp = smoothstep(T0, T0 + 0.3, T)
    emit = mz.srgb_to_lin(np.clip(d["rgb"][i], 0, 1)) * (k_cut * (1 - g) + ramp * sp * (1 - g) * (1 - 0.6 * u)
                                                         )[:, None]
    return dict(xy=pos[:, :2], z=np.maximum(pos[:, 2], 0.6), poly=poly, nv=nv, size=size, ang=ang, rgb=col,
                mat=mat, rot=rot, alpha=alpha, emit=emit.astype(np.float32), ids=100000 + i)


# ---------------------------------------------------------------- figures
FIG_EXT = (500.0, 420.0)
FIG_FEET = {"lutie": (170.0, 405.0), "crocus": (320.0, 405.0)}


def _fig_draw(poses):
    lut, cro = icon.Figure("lutie", tile=6.0), icon.Figure("crocus", tile=6.0)

    def draw(ctx, layer):
        for fig, key, h in ((lut, "lutie", LUTIE["h"]), (cro, "crocus", CROCUS["h"])):
            kw = dict(poses[key])
            fx, fy = FIG_FEET[key]
            fig.draw(ctx, fx, fy, h, layer=layer, tile=6.0, **kw)
    return draw


def _figs_setup(S):
    # the layout covers the union of every pose the two take
    poses = [dict(lutie=dict(pose="stand"), crocus=dict(pose="stand")),
             dict(lutie=dict(pose="alarm", expr="worried"), crocus=dict(pose="bless", expr="serene")),
             dict(lutie=dict(pose="recoil", expr="worried"), crocus=dict(pose="bless", expr="serene", nod=0.1))]
    cs = [icon.cartoon(_fig_draw(p), s=1.0, extent=FIG_EXT) for p in poses]
    alpha = np.max([c["alpha"] for c in cs], axis=0)
    fine = np.max([c["fine"] for c in cs], axis=0)
    gold = cs[1]["gold"]
    rgb = cs[1]["rgb"] / np.maximum(cs[1]["alpha"][..., None], 1e-3)
    lay = mz.Layout.flow(np.clip(rgb, 0, 1), tile=6.0, seed=31, extent=FIG_EXT, alpha=alpha, fine=fine > 0.5,
                         fine_tile=3.0, gold=gold)
    return dict(lay=lay)


def _figs_at(T, st, fc):
    """per-frame poses -> loose tiles in world (standing on the floor at FIG_Z)"""
    tl = fc.tl
    lay = st["figs"]["lay"]
    ml = tl.mouth("LUTIE", T)
    mc = tl.mouth("CROCUS", T)
    alarm = smoothstep(135.6, 136.1, T) * (1 - smoothstep(139.0, 139.6, T))
    recoil = smoothstep(141.85, 142.3, T)
    if recoil > 0.5:
        lp = dict(pose="recoil", expr="worried", look=(0.2, -0.6))
    elif alarm > 0.5:
        lp = dict(pose="alarm", expr="worried", look=(0.35, -0.5))
    else:
        lp = dict(pose="stand", expr="worried" if T > 135.0 else "awe", look=(0.4, -0.7))
    bless = smoothstep(139.2, 139.6, T) * (1 - smoothstep(141.4, 141.9, T))
    cp = dict(pose="bless" if bless > 0.5 else "stand", expr="serene", nod=0.08 * pulse(T, 140.0, 0.35),
              look=(-0.5, 0.0) if T < 139.3 else (-0.25, -0.2))
    lp.update(mouth=ml, t=T)
    cp.update(mouth=mc, t=T)
    c = icon.cartoon(_fig_draw(dict(lutie=lp, crocus=cp)), s=1.0, extent=FIG_EXT, layers=("color",))
    a = c["alpha"]
    rgb = c["rgb"] / np.maximum(a[..., None], 1e-3)
    col = mz.sample(np.clip(rgb, 0, 1), lay)
    al = mz.sample_mask(a, lay)
    # local -> world: the figure plane stands on the floor at FIG_Z, feet midpoint between the two
    ox = (LUTIE["x"] - FIG_FEET["lutie"][0])
    oy = A.FLOOR - FIG_FEET["lutie"][1]
    xy = lay.xy + np.array([ox, oy], np.float32)
    poly = lay.poly + np.array([ox, oy], np.float32)
    return dict(xy=xy, z=np.full(len(lay), FIG_Z), poly=poly, nv=lay.nv, size=lay.size, ang=lay.ang, rgb=col,
                mat=lay.mat, present=(al > 0.5).astype(np.float32), tilt=lay.tilt, jit=lay.jit, rnd=lay.rnd,
                ids=200000 + np.arange(len(lay)))


# ---------------------------------------------------------------- bed (setting mortar hides the sinopia)
def _bed_chart(S, r, ts):
    """chart image (k=0.5): time the setting bed appears (dust-built parts) and falls (with its sheet)."""
    p = Path(S.cache) / f"bed2_{R.VERSION}.npz"
    if p.exists():
        z = np.load(p)
        return dict(t_on=z["t_on"], t_off=z["t_off"])
    from scipy.spatial import cKDTree
    k = 0.5
    h, w = int(A.WY * k), int(A.WX * k)
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.stack([(xx.ravel() + 0.5) / k, (yy.ravel() + 0.5) / k], 1)
    tree = cKDTree(r.lay.xy)
    dist, idx = tree.query(pts, workers=1)
    t_on = np.full((h, w), -1.0, np.float32)          # the setting bed is laid before the dust arrives
    t_off = (r.fall.tb[idx] + 0.02).reshape(h, w).astype(np.float32)
    t_on = cv2.GaussianBlur(t_on, (0, 0), 2.0)
    np.savez_compressed(p, t_on=t_on, t_off=t_off)
    return dict(t_on=t_on, t_off=t_off)


# ================================================================== lights
def lights_at(T, st):
    """the light of each moment. The first light is the key for glints: the Egregore's glare while he burns,
    else the tower, else the candle."""
    L = []
    e = st["_static"]
    flare = pulse(T, 142.3, 0.16)            # his last flare as the static drains out of him
    if e > 0.02 or flare > 0.02:
        L.append(mz.Light(pos=(A.AX, 330.0, 620.0), color=(0.84, 0.9, 1.0), power=0.9 * e + 2.6 * flare,
                          radius=950.0 + 900 * flare))
        L.append(mz.Light(pos=(A.AX - 720, 460.0, 260.0), color=(1.0, 0.42, 0.72), power=0.22 * e, radius=480.0))
        L.append(mz.Light(pos=(A.AX + 720, 460.0, 260.0), color=(0.42, 0.88, 1.0), power=0.22 * e, radius=480.0))
    # the tower's glow: its screens (dying as they fall - the falling screens still light the air)
    lit = st["_tower_lit"]
    # as the screens fall their light falls with them and thins out
    dy = 0.0
    fall_dim = 1.0
    if T > 142.2:
        k = smoothstep(142.3, 143.4, T)
        fall_dim = 1.0 - 0.8 * k
    for yk, c in ((A.H(230), (0.78, 0.88, 1.0)), (A.H(520), (1.0, 0.82, 0.92)), (A.H(800), (0.82, 1.0, 0.95))):
        L.append(mz.Light(pos=(A.AX + 30, yk + dy, 450.0), color=c, power=0.62 * lit * fall_dim, radius=560.0))
    # the vortex's cold light during the rise
    v = smoothstep(131.85, 132.6, T) * (1 - smoothstep(133.7, 134.5, T))
    if v > 0:
        L.append(mz.Light(pos=(A.AX, A.H(600), 380.0), color=(0.72, 0.84, 1.0), power=0.55 * v, radius=650.0))
    # the votive candle (the one light that survives); the avalanche's wind makes it gutter
    L.append(R.candle_light(T, 1.0, _gust(T)))
    amb = 0.003 + 0.072 * smoothstep(131.95, 134.5, T) - 0.07 * smoothstep(142.3, 143.6, T)
    return L, max(0.012, amb)


def _gust(T):
    return 0.7 * math.sin(T * 6.3) * smoothstep(142.2, 142.6, T) * (1 - smoothstep(143.8, 144.7, T))


# ================================================================== per-frame tile state
def _static_level(T):
    """Egregore's static: flares on at 134.0, drains away 141.8-142.65"""
    on = smoothstep(T_EGR - 0.02, T_EGR + 0.12, T)
    return on * (1 - smoothstep(142.35, 142.7, T))


def _drain_y(T):
    """the drain level (chart y): above it the static is gone"""
    return 60.0 + 520.0 * smoothstep(141.82, 142.62, T) ** 1.2


def _wall_state(T, st, fc):
    r = st["r"]
    lay = r.lay
    n = st["n"]
    emat = st["emat"]
    zone = r.zone
    F = int(round(T * 24))
    rgb = r.rgb.copy()
    mat = r.mat.copy()
    emit = np.zeros((n, 3), np.float32)
    gain = np.ones(n, np.float32)
    # ---- presence: dust-built tiles appear when their speck arrives; everything leaves at its detach
    ts = st["ts"]
    present = ~st["built"] | (T >= ts)
    # arrival flash
    arr = st["built"] & (T >= ts) & (T < ts + 0.35)
    gain[arr] += 2.4 * np.exp(-(T - ts[arr]) / 0.07)
    h = st["hash"]
    # ---- the cities' windows: dead glass until they come on (a flicker), after the tower has risen
    tc = st["t_city"]
    off = (tc > 0) & (T < tc)
    if off.any():
        mat[off] = mz.GLASS
        rgb[off] = r.rgb_dead[off]
    blink = (tc > 0) & (T >= tc) & (T < tc + 0.12)
    if blink.any():
        gain[blink] += 3.0 * (mz._hash_uniform(np.nonzero(blink)[0] * 3 + int(T * 48), 4) > 0.4)
    # ---- static (face, halo disc, hands, clavi): TV snow, rolling bar, draining away top-down
    stat = emat == A.M_STATIC
    lev = _static_level(T)
    si = np.nonzero(stat)[0]
    if lev > 0 or (T > T_EGR and T < 142.8):
        u = mz._hash_uniform(si * 131 + F * 7919, 3)
        y = lay.xy[si, 1]
        roll = 0.35 * np.exp(-((np.mod(y - 160.0 * T, 420.0) - 210.0) / 30.0) ** 2)
        dy = _drain_y(T)
        keep = sst(dy - 26, dy + 6, y)
        band = 1.5 * np.exp(-((y - dy - 8) / 22.0) ** 2) * (T > 141.82)
        on = smoothstep(T_EGR - 0.02, T_EGR + 0.12, T)
        L = (0.12 + 0.88 * u ** 1.6 + roll) * on * np.where(T > 141.82, keep * (1 + band), 1.0)
        L *= 1.0 + 2.0 * pulse(T, T_EGR + 0.02, 0.05)        # the flare when he switches on
        disc = zone[si] == A.Z_DISC
        L = np.where(disc, L * 0.55, L)
        emit[si] = (L[:, None] * np.array([0.78, 0.82, 0.95], np.float32)) * 1.35
        rgb[si] = np.array(A.rgb("ink")) * 0.7 + np.array(A.rgb("slate")) * 0.3
        mat[si] = mz.GLASS
    else:
        rgb[si] = np.array(A.rgb("ink")) * 0.7 + np.array(A.rgb("slate")) * 0.3
    # ---- the halo of screens: a loading wheel, spinning, stopping when he dies
    hal = np.nonzero(emat == A.M_HALO)[0]
    if len(hal):
        seg = r.obj[hal].astype(np.float32)
        spin_t = min(T, 142.0) - T_EGR
        ph = np.mod(seg / 28.0 - spin_t * 0.85, 1.0)
        b = 0.22 + 1.0 * np.exp(-ph * 7.0)
        b *= _static_level(T) ** 0.7 * (1 - smoothstep(141.9, 142.25, T) * (seg > -1))
        emit[hal] = st["rgb_lin"][hal] * (b[:, None] * 1.3)
        rgb[hal] = rgb[hal] * 0.25
    # ---- billboards: flicker, change their ads, and collapse to a line (CRT off) in the drain
    bi = np.nonzero(emat == A.M_BOARD)[0]
    if len(bi):
        on = sst(T_EGR + 0.8, T_EGR + 1.05, T + 0.6 * (st["bph"][bi] / 10.0))
        swap = np.floor((T + st["bph"][bi]) / 1.6)
        pal = np.array([A.rgb(c) for c in ("glitch_m", "slop_cyan", "slop_flesh", "glitch_c", "screen_glow",
                                           "slop_mauve")], np.float32)
        ci = (r.obj[bi] + swap.astype(int)) % len(pal)
        base = mz.srgb_to_lin(pal[ci])
        # a white 'logo' blob in each ad
        uv = st["buv"][bi]
        blob = np.exp(-((uv[:, 0] - 0.35 * np.sin(swap)) ** 2 + (uv[:, 1] + 0.1) ** 2) * 5.0)
        colr = base * (1 - 0.8 * blob[:, None]) + 0.9 * blob[:, None]
        fl = 0.85 + 0.15 * np.sin(T * 23.0 + st["bph"][bi] * 5)
        # CRT off
        to = st["boff"][bi]
        k = np.clip((T - to) / 0.14, 0, 1)
        vis = np.abs(uv[:, 1]) < (1 - k) * 1.05 + 0.08 * (k < 1)
        line = (k > 0) & (k < 1) & (np.abs(uv[:, 1]) < 0.18)
        e = on * fl * vis * (T < to + 0.14)
        e = e + 2.5 * line * (1 - k)
        emit[bi] = colr * (e[:, None] * 1.5)
        rgb[bi] = rgb[bi] * 0.2
    # ---- dim screen specks in the conch ground
    di = np.nonzero(emat == A.M_DIM)[0]
    if len(di):
        tw_ = 0.3 + 0.7 * (mz._hash_uniform(di * 17 + np.floor(T * 3.0).astype(int) * 131, 9) > 0.6)
        emit[di] = st["rgb_lin"][di] * (0.22 * tw_ * smoothstep(T_EGR, T_EGR + 0.8, T) *
                                        (1 - smoothstep(141.9, 142.4, T)))[:, None]
    # ---- the treadwheel: its spokes turn (re-colouring)
    wi = np.nonzero(emat == A.M_WHEEL)[0]
    if len(wi):
        wx, wy, wr = A.CRANE["wheel"]
        a = np.arctan2(lay.xy[wi, 1] - wy, lay.xy[wi, 0] - wx) - min(T, T_COL) * 2.2
        rad = np.hypot(lay.xy[wi, 1] - wy, lay.xy[wi, 0] - wx)
        spoke = (np.abs(np.sin(a * 3)) < 0.28) | (rad > wr * 0.78) | (rad < 6)
        rgb[wi] = np.where(spoke[:, None], np.array(A.rgb("ochre_red")), np.array(A.rgb("bone")))
    # ---- the collapse: loose tiles and dying screens
    idx, pos, quat = r.fall.at(T)
    screen = np.isin(emat, [A.M_SCREEN, A.M_FEED])
    td_ = st["t_die"]
    dying = screen & (T >= r.fall.td) & (T < td_)
    flash = dying & (T > td_ - 0.07)
    if flash.any():
        fi = np.nonzero(flash)[0]
        mat[fi] = mz.GLASS
        g = np.where(mz._hash_uniform(fi, 11) > 0.5, 1, 0)
        emit[fi] = np.where(g[:, None] > 0, np.array([1.0, 0.03, 0.25]), np.array([0.0, 0.78, 1.0])) * 1.8
    dead = (np.isin(emat, A.EMISSIVE) & (T >= td_)) | (stat & (T > 142.7))
    if dead.any():
        mat[dead & (mat == mz.SCREEN)] = mz.GLASS
        rgb[dead] = r.rgb_dead[dead]
        emit[dead] = 0
    # the tower's light: its living screens
    tower = st["_tower_mask"] if "_tower_mask" in st else None
    return dict(present=present, rgb=rgb, mat=mat, emit=emit, gain=gain, loose=(idx, pos, quat))


def _tower_lit(T, st):
    r = st["r"]
    if "_tower_mask" not in st:
        st["_tower_mask"] = np.isin(r.zone, list(range(A.Z_TIER0, A.Z_TIER0 + 7)) + [A.Z_CROWN]) & np.isin(
            st["emat"], [A.M_SCREEN, A.M_FEED])
    m = st["_tower_mask"]
    set_ = (T >= st["ts"][m]).mean()
    alive = (T < st["t_die"][m]).mean()
    return float(set_ * alive)


# ================================================================== render
def render(fc, st):
    T = fc.T
    cam, (cx, cy, zoom) = cam_at(T)
    r = st["r"]
    lay = r.lay
    n = st["n"]
    st["_tower_lit"] = _tower_lit(T, st)
    st["_static"] = _static_level(T)
    lights, amb = lights_at(T, st)
    ws = _wall_state(T, st, fc)
    # ------------------------------------------------ background: bed / bare plaster + sinopia / floor
    bg = _backdrop(fc, cam, T, st, lights, amb)
    # ------------------------------------------------ one bundle: wall tiles (bedded or loose) + extras (loose)
    idx, pos, quat = ws["loose"]
    xy = lay.xy.copy()
    poly = lay.poly.copy()
    z = np.zeros(n, np.float32)
    rot = np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy()
    if len(idx):
        d = (pos[:, :2] - lay.xy[idx]).astype(np.float32)
        xy[idx] = pos[:, :2]
        poly[idx] = lay.poly[idx] + d[:, None, :]
        z[idx] = np.maximum(pos[:, 2], 0.3)
        qrel = P.q_mul(quat, R._q_conj(r.fall.q_home[idx]))
        rot[idx] = P.q_matrix(qrel).astype(np.float32)
    present = ws["present"].astype(np.float32)
    wall = dict(xy=xy, poly=poly, nv=lay.nv, size=lay.size, ang=lay.ang, rgb=ws["rgb"], mat=ws["mat"], z=z,
                rot=rot, emit=ws["emit"], gain=ws["gain"], present=present, tilt=lay.tilt, jit=lay.jit,
                rnd=lay.rnd, ids=np.arange(n))
    extras = []
    if T < 135.7:
        dd = _dust_at(T, st)
        if dd is not None:
            extras.append(dd)
    if T < 144.9:
        extras.extend(st["workers"].at(T, st["_static"]))
    extras.append(dict(st["candle"]))
    # depth layers: everything behind the pair / the pair (a bedded panel on its own plane) / everything in front
    zcut = FIG_Z - 4.0
    front_w = z >= zcut
    wall_back = dict(wall)
    wall_back["present"] = np.where(front_w, 0.0, present).astype(np.float32)
    back = [wall_back] + [_sub(e, np.asarray(e.get("z", np.zeros(len(e["xy"])))) < zcut) for e in extras]
    front = [_sub(wall, front_w)] + [_sub(e, np.asarray(e.get("z", np.zeros(len(e["xy"])))) >= zcut)
                                     for e in extras]
    env = 0.08 * smoothstep(131.9, 132.6, T) + 0.55 * smoothstep(131.9, 134.0, T) * (
        1 - smoothstep(143.4, 144.5, T))
    look = dict(view=cam, light=lights, ambient=amb, t=T, env=env,
                sparkle=1.35 + 0.9 * smoothstep(142.0, 142.4, T))
    img = mz.render_tiles(fc, _bundle([b for b in back if len(b["xy"])]), loose=False, bg=bg, holes="bg", **look)
    if T < 145.2:
        fg = _figs_at(T, st, fc)
        # in the silence they stand beyond the candle's reach: the heap between them and the flame shades them
        fg["gain"] = np.full(len(fg["xy"]), 1.0 - 0.94 * smoothstep(143.2, 144.8, T), np.float32)
        fz = FIG_Z
        fg["z"] = np.zeros(len(fg["xy"]), np.float32)
        lay_f = _bundle([fg])
        panel = mz.render_tiles(fc, lay_f, loose=False, surface=mz.Plane(origin=(0.0, 0.0, fz)), holes="bg",
                                return_alpha=True, **look)
        img = panel[..., :3] + img * (1 - panel[..., 3:4])
    fr = [f_ for f_ in front if len(f_["xy"])]
    if fr:
        img = mz.render_tiles(fc, _bundle(fr), loose=True, bg=img, **look)
    # ------------------------------------------------ the flame (painted light)
    top = Canvas(fc)
    R.draw_flame(top.ctx, cam, T, fc.s, gust=_gust(T))
    img = top.over(img)
    return img


def _sub(part, m):
    """rows m of a part dict (every per-tile array)"""
    m = np.asarray(m, bool)
    n = len(part["xy"])
    return {k: (v[m] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in part.items()}


def _bundle(parts):
    """merge per-part tile dicts into one mz.Tiles (missing per-part arrays get defaults)"""
    keys = ("xy", "poly", "nv", "size", "ang", "rgb", "mat", "z", "rot", "emit", "gain", "alpha", "present",
            "tilt", "jit", "rnd", "ids")
    out = {}
    for k in keys:
        arrs = []
        for p in parts:
            m = len(p["xy"])
            v = p.get(k)
            if v is None:
                if k == "nv":
                    v = np.full(m, 4, np.int8)
                elif k == "rot":
                    v = np.broadcast_to(np.eye(3, dtype=np.float32), (m, 3, 3))
                elif k == "emit":
                    v = np.zeros((m, 3), np.float32)
                elif k in ("gain", "alpha", "present"):
                    v = np.ones(m, np.float32)
                elif k == "z":
                    v = np.full(m, 0.6, np.float32)
                elif k == "mat":
                    v = np.zeros(m, np.uint8)
                elif k == "ang":
                    v = np.zeros(m, np.float32)
                elif k in ("tilt",):
                    v = mz._hash_normals(np.asarray(p["ids"]), 0)[:, :2]
                elif k == "jit":
                    v = mz._hash_normals(np.asarray(p["ids"]), 0)[:, 2:5]
                elif k == "rnd":
                    v = mz._hash_uniform(np.asarray(p["ids"]), 0)
                elif k == "ids":
                    v = np.arange(m) + 300000
            v = np.asarray(v)
            if k == "poly":
                v = v.astype(np.float32)
                if v.shape[1] < mz.KMAX:
                    v = np.concatenate([v, np.repeat(v[:, -1:], mz.KMAX - v.shape[1], axis=1)], 1)
            if k == "rot" and v.ndim == 2:
                v = mz.axis_angle(v)
            arrs.append(np.broadcast_to(v, (m,) + v.shape[1:]) if v.ndim else np.full(m, v))
        out[k] = np.concatenate(arrs, 0)
    return mz.Tiles(out["xy"], ang=out["ang"], size=out["size"], rgb=out["rgb"], mat=out["mat"], z=out["z"],
                    rot=out["rot"], alpha=out["alpha"], gain=out["gain"], emit=out["emit"], present=out["present"],
                    poly=out["poly"], nv=out["nv"], tilt=out["tilt"], jit=out["jit"], rnd=out["rnd"], ids=out["ids"])


def _backdrop(fc, cam, T, st, lights, amb):
    """bare plaster + sinopia where no bed; the dark setting bed where tiles are (or are about to be) set"""
    r = st["r"]
    bed = st["bed"]
    k = 0.5

    def paint_bed(fc_, cam_, img):
        on = bed["t_on"]
        has_bed = ((on < 0) | (T >= on)) & (T < bed["t_off"])
        mort = mz.srgb_to_lin(np.array(A.rgb("mortar"), np.float32)) * 0.22
        m = has_bed.astype(np.float32)[..., None]
        return img * (1 - m) + mort * m
    return r.backdrop.render(fc, cam, lights, amb, extra=paint_bed)


# ================================================================== post
def post(fc, st):
    T = fc.T
    # the silence: a touch darker, no bloom pumping; the grain stays (it is the film, not the world)
    return dict(bloom=0.42 * (1 - 0.5 * smoothstep(144.0, 145.0, T)))


# ================================================================== foley
def _export_foley(S, st):
    r = st["r"]
    f = r.fall
    ev = []
    for k, t in enumerate(TIER_T):
        xc = A.tiers()[k][0]
        ev.append(dict(t=t, kind="tier", strength=0.55 + 0.06 * k, pan=0.0,
                       desc=f"tier {k + 1} of the tower sets itself (the spiral of dust lands)"))
    ev.append(dict(t=T_CROWN, kind="tier", strength=0.6, pan=0.05, desc="the unfinished crown + crane set"))
    ev.append(dict(t=T_EGR, kind="egregore", strength=1.0, pan=0.0,
                   desc="Lord Egregore sets himself in the conch: the static face flares on"))
    ev.append(dict(t=T_EGR + 1.4, kind="egregore_lit", strength=0.8, pan=0.0,
                   desc="all billboards and the loading-wheel halo lit (to 141.8)"))
    ev.append(dict(t=T_COL, kind="topple", strength=1.0, pan=0.15,
                   desc="the tower cracks above tier 3 and its top hinges over to the right (to 142.5)"))
    ev.append(dict(t=141.82, kind="static_drain", strength=0.9, pan=0.0,
                   desc="Egregore's static drains away top-down, billboards collapse to a line (to 142.65)"))
    tt = f.tl[f.tower_top]
    ev.append(dict(t=float(np.percentile(tt, 20)), kind="tower_impact", strength=1.0, pan=0.35,
                   desc="the tower's top hits the floor"))
    ev.append(dict(t=float(np.percentile(f.td, 3)), kind="avalanche", strength=1.0, pan=0.0,
                   desc="the whole mosaic begins to fall off the wall (start)"))
    ev.append(dict(t=float(f.tr.max()), kind="avalanche_end", strength=0.2, pan=0.0,
                   desc="the last tessera settles (end; silence from 145.0)"))
    last = int(np.argmax(f.tl))
    ev.append(dict(t=float(f.tl[last]), kind="last_tile", strength=0.35,
                   pan=foley.screen_pan(960 + (f.pl[last, 0] - R.CAM_END["cx"]) * R.CAM_END["zoom"]),
                   desc="the very last tessera lands, alone"))
    ev = [e for e in ev if e["t"] < T_SIL]
    foley.export_events(S, "sfx", ev)
    # per-frame avalanche track: fraction moving, detaching / landing / bouncing counts, screen pan of the mass
    nfr = S.n
    Tg = S.start + np.arange(nfr) / S.fps
    moving = np.zeros(nfr, np.float32)
    fall_ = np.zeros(nfr, np.float32)
    land = np.zeros(nfr, np.float32)
    bounce = np.zeros(nfr, np.float32)
    pan = np.zeros(nfr, np.float32)
    fb = np.clip(((f.td - S.start) * S.fps).astype(int), 0, nfr - 1)
    np.add.at(fall_, fb, 1)
    lb = np.clip(((f.tl - S.start) * S.fps).astype(int), 0, nfr - 1)
    np.add.at(land, lb, 1)
    for j in range(1, 4):
        tb_ = f.tl + (f.tr - f.tl) * (j / (f.nhop + 1))
        m = f.nhop >= j
        bb = np.clip(((tb_[m] - S.start) * S.fps).astype(int), 0, nfr - 1)
        np.add.at(bounce, bb, 1)
    for i, T in enumerate(Tg):
        if T < T_COL - 0.05 or T >= T_SIL:
            continue
        mv = (T >= f.td) & (T < f.tr)
        moving[i] = mv.mean()
        if mv.any():
            cam, _ = cam_at(T)
            xs = f.home[mv, 0]
            sx = cam.project3(np.c_[xs, np.full(mv.sum(), 800.0), np.zeros(mv.sum())])[0][:, 0]
            pan[i] = float(np.clip((sx.mean() - 960) / 960, -1, 1))
    after = Tg >= T_SIL
    for a in (moving, fall_, land, bounce):
        a[after] = 0
    foley.export_track(S, "avalanche", energy=moving, pan=pan, kind="tiles", fall=fall_, land=land, bounce=bounce)


# ================================================================== hand-off to m08 (146.0)
def _export_handoff(S, st):
    """cache/m07/handoff.npz: everything m08 needs to continue from m07's last frame (see m07_ruin docstring)."""
    r = st["r"]
    f = r.fall
    ce = R.CAM_END
    cam = R.cam_end()
    feet = {k: (LUTIE["x"] if k == "lutie" else CROCUS["x"], A.FLOOR, FIG_Z) for k in ("lutie", "crocus")}
    scr = {k: cam.project3(np.array([v]))[0][0] for k, v in feet.items()}
    flame = cam.project3(np.array([R.CANDLE_FLAME]))[0][0]
    np.savez_compressed(
        Path(S.cache) / "handoff.npz",
        cam_eye=cam.eye, cam_target=cam.target, cam_fov=R.FOV, cam_at=cam.at, cam_params=np.array(
            [ce["cx"], ce["cy"], ce["zoom"], ce["he"]]),
        candle_xz=np.array(R.CANDLE_XZ), candle_flame=np.array(R.CANDLE_FLAME), candle_flame_screen=flame,
        heap_pos=f.pr.astype(np.float32), heap_rot=r.rest_rot.astype(np.float32), heap_rgb=r.rgb_dead,
        heap_mat=np.where(r.mat == mz.SCREEN, mz.GLASS, r.mat).astype(np.uint8), heap_size=r.lay.size,
        heap_poly_home=r.lay.poly, heap_nv=r.lay.nv, home_xy=r.lay.xy, heightfield=f.heightfield,
        heightfield_grid=np.array([P.HX0, P.HD, 0.0, P.HD]), fig_z=FIG_Z,
        lutie_world=np.array(feet["lutie"]), crocus_world=np.array(feet["crocus"]),
        lutie_screen=scr["lutie"], crocus_screen=scr["crocus"], lutie_h=LUTIE["h"], crocus_h=CROCUS["h"],
        t_last_rest=float(f.tr.max()))

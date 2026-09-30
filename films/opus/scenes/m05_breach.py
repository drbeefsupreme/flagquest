"""m05 - THE SCHISMMANCERS (95.75-106.5).  IN · HOC · SIGNO · SCINDES  ("in this sign you shall split")

A frieze of eleven Byzantine warrior saints on gold - the Schismmancers, a breaching team stacked along the wall like
the martyrs' procession of Sant'Apollinare Nuovo - and at its centre Schism Actual (hierarchic scale, radio headset,
a T-O map orb like Michael the Archistrategos). The icon is a window; they break it.

  95.75  hard cut (graphic match on m04's gold flash): the frieze by candlelight, NVG goggles up, furled flags on
         collapsed telescoping lances. Titulus IN·HOC·SIGNO·SCINDES; each saint named (SCS·FISSOR, SCS·RVPTOR ...).
  96.6   K01 (radio). Rings of glint run through the gold from his headset with his voice; he drops his goggles on
         "Actual"; on "to all teams" the ring reaches the saints and the goggles drop in a ripple outward (clicks),
         the quad tubes power up phosphor green.  98.7 "Target:" all eyes flare; slow push-in to the Archistrategos.
  101.6  K02 cut wide - the starry heaven above the frieze. "Method:" the lances swing into a fan; "re-cur-sive":
         three clacks, three telescoping sections; "balkanization": the flags unfurl from the centre outward, real
         cloth snapping open in a wind that splits at the centre (Constantine's vision, above the titulus).
  103.9  K03 "Breach! Breach! Breach!": three punch-ins (the saints brace, then charge).
  105.2  cut wide: dead calm, the flags fall limp; light leaks through every mortar joint of the wall.
  105.8  BREACH: the wall of the mosaic blasts toward camera - every tessera of the frieze becomes the storm - and
         the whole cloth flags come through it; flag-yellow light floods the frame (flat flag_glow by 106.5 -> m06).
"""
import hashlib
import math
import pickle

import cairo
import numpy as np
from scipy.spatial import cKDTree

from vx import *
from vx import mosaic as mz
from vx import foley
from vx.flag import Flag, draw_pole, draw_cloth
from vx.icon import Figure
from scenes import m05_wall as wall


POST = dict(bloom=0.42, bloom_thresh=0.7, bloom_radius=22.0, grain=0.0, vignette=0.38, contrast=1.04)   # no grain: nothing over the cloth

SC = 0.8                     # cartoon scale (tiles are single-coloured: ~4 px per fine tile)
FG = np.array(C["flag_glow"], np.float32)
NVG_CORE = np.array(col("#D9FFC8"), np.float32)
NVG_GLOW = np.array(col("#43FF6E"), np.float32)
NVG_GLOW_LIN = NVG_GLOW ** 2.2
FG_LIN = FG ** 2.2
BREACH_PT = (960.0, 470.0)   # wall point the blast starts from (Schism Actual's breastplate)
L_SIM = 576.0                # flag sim pole length (canonical 36"x30" proportions for a 216 x 180 cloth)
CLOTH_W, CLOTH_H = 216.0, 180.0
SECTION = 150.0              # each telescoping section extends this far
POLE_W = (12.0, 10.5, 9.0, 7.5)


# ============================================================ timing (local seconds, all from the locked timeline)
def _times(S):
    tl = S.tl
    L = lambda T: T - S.start

    def w(lid, k, n=0):
        a, b = tl.word(lid, k, n)
        return L(a), L(b)

    T = dict(k01=L(tl.line("K01")["start"]), actual=w("K01", "Actual")[0], all=w("K01", "all")[0],
             teams=w("K01", "teams"), target=w("K01", "Target")[0], earth=w("K01", "Earth"),
             k02=L(tl.line("K02")["start"]), method=w("K02", "Method")[0], rec=w("K02", "recursive"),
             balk=w("K02", "balkanization"), b=[w("K03", "Breach", k) for k in range(3)],
             breach=L(tl.cue("breach")), end=S.dur)
    T["calm"] = T["b"][2][1]                                   # after the third "Breach!" dies away
    r0, r1 = T["rec"]
    T["clack"] = [r0 + 0.02 + k * (r1 - r0) / 3.0 for k in range(3)]   # re - cur - sive
    T["gog"] = [T["actual"] - 0.02 if i == wall.CENTRE else T["all"] + 0.12 + 0.16 * (abs(i - wall.CENTRE) - 1)
                for i in range(wall.N_SAINTS)]
    T["unfurl"] = [T["balk"][0] + 0.04 + 0.075 * abs(i - wall.CENTRE) for i in range(wall.N_SAINTS)]
    return T


def _goggles(t, td):
    """0 up -> 1 down; the mount snaps down in 0.11 s and bounces"""
    tau = t - td
    if tau <= 0:
        return 0.0
    if tau < 0.11:
        return (tau / 0.11) ** 2
    return 1.0 - 0.07 * math.exp(-(tau - 0.11) * 22) * abs(math.sin((tau - 0.11) * 45))


def _power(t, td, i, T):
    """phosphor emission 0..~1.8 of saint i's tubes"""
    tau = t - td - 0.11
    if tau <= 0:
        return 0.0
    p = 1.0 - math.exp(-tau / 0.12)
    if tau < 0.35:
        p *= 0.6 + 0.4 * (0.5 + 0.5 * noise1(t * 45 + i * 3.1, 7))
    p *= 0.93 + 0.07 * noise1(t * 9 + i, 11)
    p *= 1 + 0.9 * pulse(t, T["target"] + 0.12, 0.13)
    for b0, b1 in T["b"]:
        p *= 1 + 0.5 * pulse(t, b0 + 0.06, 0.08)
    return p


def _pose(t, T):
    b = T["b"]
    return "rest" if t < b[0][0] else "brace" if t < b[2][0] else "charge"


def _icon_pose(i, key):
    """scene pose key -> vx.icon pose name (Schism Actual presents the orb while at rest)"""
    if key == "rest":
        return "command" if i == wall.CENTRE else "hold_pole"
    return key


# ============================================================ figures
def _fig_params(i, t, fc_T, T, tl):
    actual = i == wall.CENTRE
    return dict(pose=_icon_pose(i, _pose(t, T)), goggles=_goggles(t, T["gog"][i]), t=t,
                mouth=tl.mouth("COMMANDER", fc_T) if actual else 0.0, pole_ang=_fan(i, t, T),
                wind=0.0 if t < T["b"][2][0] else min(1.0, (t - T["b"][2][0]) * 4))


def _anchors(fig, x, h, pose, goggles=0.0):
    return fig.anchors(x, wall.GROUND, h, pose=pose, goggles=goggles, pole_ang=0.0)


def _fig_order():
    """outer saints first, Schism Actual last (on top)"""
    return sorted(range(wall.N_SAINTS), key=lambda i: -abs(i - wall.CENTRE))


# ============================================================ poles (telescoping lances), pure functions of t
def _ext(tau):
    """one telescoping section shooting out: 0 -> 1 with a hard stop overshoot"""
    if tau <= 0:
        return 0.0
    return 1.0 - math.exp(-tau * 40.0) * math.cos(tau * 55.0)


REST_SLANT = -0.07          # at rest the lances lean in parallel, like the guards' spears in the Justinian panel


def _fan(i, t, T):
    """lance angle (0 = vertical): parallel slant at rest; on "Method:" they swing out into a fan split at the centre"""
    return lerp(REST_SLANT, (i - wall.CENTRE) * 0.05, ease_out_back(clamp((t - T["method"]) / 0.32), 1.2))


def _pole_geom(i, t, T, grips):
    """(butt, u, tip, joints[4]) in wall coords for saint i at local time t.
    grips: {pose: (gx, gy, ang0), 'L0': collapsed length above the grip}; joints = ends of base tube + 3 sections."""
    gx, gy, a0 = grips[_pose(t, T)]
    ang = _fan(i, t, T)
    ux, uy = math.sin(ang), -math.cos(ang)
    below = wall.GROUND - 6 - gy
    butt = (gx - ux * below, gy - uy * below)
    js = [grips["L0"]]
    for c in range(3):
        jit = 0.012 * (hash01(i * 3 + c, 21) - 0.5)
        js.append(js[-1] + SECTION * _ext(t - T["clack"][c] - jit))
    joints = [(gx + ux * L, gy + uy * L) for L in js]
    return butt, (ux, uy), joints[-1], joints


def _wind(i, t, T):
    k = i - wall.CENTRE
    side = 1.0 if k >= 0 else -1.0
    tu = T["unfurl"][i]
    if k == 0:
        base = np.array([260.0, -420.0, 520.0])
    else:
        base = np.array([side * 820.0, -70.0, 60.0])
    a = smoothstep(tu - 0.05, tu + 0.12, t)
    wv = np.array([side * 200.0, 0.0, 0.0]) * (1 - a) + base * a * (1 + 0.25 * pulse(t, tu + 0.1, 0.1))
    calm = smoothstep(T["calm"] - 0.15, T["calm"] + 0.25, t)
    wv = wv * (1 - calm) + np.array([side * 25.0, 0.0, 0.0]) * calm
    if t >= T["breach"]:
        bx, by = wall.saints()[i][2] - BREACH_PT[0], -300.0 - BREACH_PT[1]
        r = math.hypot(bx, by) + 1e-6
        blast = min(1.0, (t - T["breach"]) * 8)
        wv = wv * (1 - blast) + np.array([bx / r * 1300.0, by / r * 900.0, 1500.0]) * blast
    return tuple(wv)


def _simulate_flags(S, T, grips_all):
    """eleven cloth sims (cached on disk, keyed by every pole pose / wind / furl value they are driven by)"""
    poles, winds, furls = [], [], []
    for i in range(wall.N_SAINTS):
        p, w, u = [], [], []
        for f in range(S.n):
            t = f / S.fps
            _, (ux, uy), tip, _ = _pole_geom(i, t, T, grips_all[i])
            p.append((tip[0] - ux * L_SIM, tip[1] - uy * L_SIM, math.atan2(ux, -uy)))
            w.append(_wind(i, t, T))
            u.append(1.0 - smoothstep(T["unfurl"][i], T["unfurl"][i] + 0.28, t))
        poles.append(np.array(p))
        winds.append(np.array(w))
        furls.append(np.array(u))
    sig = np.round(np.concatenate([np.concatenate(poles).ravel(), np.concatenate(winds).ravel(),
                                   np.concatenate(furls).ravel()]), 2)
    key = f"flags_{S.n}_{hashlib.md5(sig.tobytes() + repr((CLOTH_W, CLOTH_H, L_SIM)).encode()).hexdigest()[:14]}"
    path = S.cache / f"{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    tracks = []
    for i in range(wall.N_SAINTS):
        fl = Flag(width=CLOTH_W, height=CLOTH_H, pole=L_SIM, pole_w=POLE_W[3], side=1 if i >= wall.CENTRE else -1,
                  seed=500 + i * 13, turbulence=1.1, flutter=1.0)
        tracks.append(fl.simulate(S.n, lambda f, i=i: tuple(poles[i][f]), lambda f, i=i: tuple(winds[i][f]),
                                  fps=S.fps, furl_fn=lambda f, i=i: float(furls[i][f]), warmup=24))
    for old in S.cache.glob("flags_*.pkl"):
        old.unlink()
    path.write_bytes(pickle.dumps(tracks))
    return tracks


# ============================================================ camera
def _camera(t, T):
    """(cx, cy, zoom, rot, shake_dx, shake_dy): wall point at screen centre, zoom, roll, screen shake (design px)"""
    b = T["b"]
    sx = sy = 0.0
    if t < T["k02"]:                                             # A: the frieze, then the push to the Archistrategos
        z = 1.0 + 0.05 * seg(t, 0.0, T["target"], ease_in_out) \
            + 0.95 * seg(t, T["target"] - 0.2, T["k02"] - 0.05, ease_in_out)
        cy = lerp(lerp(540.0, 525.0, seg(t, 0, T["target"])), 420.0, seg(t, T["target"] - 0.2, T["k02"] - 0.05))
        cx = 960.0
    elif t < b[0][0]:                                            # B: the heaven opens above the frieze
        u = seg(t, T["k02"], b[0][0], ease_in_out_sine)
        z = lerp(0.76, 0.8, u)
        cx, cy = 960.0, lerp(318.0, 334.0, u)
    elif t < b[1][0]:                                            # C1..C3: three punch-ins
        z, cx, cy = 1.55, 960.0, T["head"]["brace"][1] + 190.0
    elif t < b[2][0]:
        z, (cx, cy) = 2.7, (T["head"]["brace"][0], T["head"]["brace"][1] + 55.0)
    elif t < T["calm"]:
        z, (cx, cy) = 4.6, (T["head"]["charge"][0], T["head"]["charge"][1] + 12.0)
    else:                                                        # D: the lit wall, the blast
        u = seg(t, T["calm"], T["breach"], ease_in)
        z = lerp(0.8, 0.86, u) + 0.1 * ease_out(clamp((t - T["breach"]) / 0.7))
        cx, cy = 960.0, lerp(360.0, 395.0, u)
    for k, (b0, b1) in enumerate(b):                              # each "Breach!" jolts the camera
        if t >= b0:
            a = 9.0 * math.exp(-(t - b0) * 9.0)
            dx, dy = shake(t, a, 13.0, 40 + k)
            sx += dx
            sy += dy
    if t >= T["calm"]:
        dx, dy = shake(t, 1.6 * smoothstep(T["calm"], T["breach"], t), 17.0, 77)
        sx, sy = sx + dx, sy + dy
    if t >= T["breach"]:
        a = 26.0 * math.exp(-(t - T["breach"]) * 5.0)
        dx, dy = shake(t, a, 11.0, 91)
        sx, sy = sx + dx, sy + dy
    return cx, cy, z, 0.0, sx, sy


def _apply_cam(ctx, cam):
    """cairo ctx (already scaled by fc.s) -> draw in WALL coords"""
    cx, cy, z, rot, sx, sy = cam
    ctx.translate(960.0 + sx, 540.0 + sy)
    ctx.rotate(rot)
    ctx.scale(z, z)
    ctx.translate(-cx, -cy)


def _to_screen(cam, x, y):
    cx, cy, z, rot, sx, sy = cam
    dx, dy = (x - cx) * z, (y - cy) * z
    c, s = math.cos(rot), math.sin(rot)
    return 960.0 + sx + dx * c - dy * s, 540.0 + sy + dx * s + dy * c


# ============================================================ setup
def _paint_wall_layers(scale, figs, saints, layers=("color", "gold", "fine"), halos=None):
    """static wall (+ figures at rest if figs, for the layout guide) at `scale` -> dict of float32 images
    ('color' rgb, masks as alpha; with figs also 'figs' = the figures' coverage)"""
    w, h = int(round(wall.WALL_W * scale)), int(round(wall.WALL_H * scale))
    out = {}

    def grab(surf):
        surf.flush()
        return np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]

    def figures(ctx, layer):
        for i in _fig_order():
            kind, seed, x, hh = saints[i]
            a = figs[i].draw(ctx, x, wall.GROUND, hh, pose=_icon_pose(i, "rest"), goggles=0.0, pole_ang=0.0,
                             layer=layer)
            if kind == "schism_actual" and a.get("orb"):
                wall.paint_orb(ctx, *a["orb"], layer)

    for layer in layers:
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surf)
        ctx.scale(scale, scale)
        ctx.translate(-wall.WX0, -wall.WY0)
        wall.paint_static(ctx, layer, halos)
        if figs is not None:
            figures(ctx, layer)
        buf = grab(surf)
        if layer == "color":
            out[layer] = buf[..., [2, 1, 0]].astype(np.float32) / 255.0
        else:
            out[layer] = buf[..., 3].astype(np.float32) / 255.0
    if figs is not None:
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surf)
        ctx.scale(scale, scale)
        ctx.translate(-wall.WX0, -wall.WY0)
        figures(ctx, "color")
        out["figs"] = grab(surf)[..., 3].astype(np.float32) / 255.0
    return out


def setup(S):
    T = _times(S)
    saints = wall.saints()
    # Schism Actual: imperial purple chlamys, white tunic under the lamellar, pearled halo, broader build
    figs = [Figure(kind, seed=seed, **(dict(cloak="tyrian", robe="white", halo_style="gold_pearl", build=1.1)
                                        if kind == "schism_actual" else {}))
            for (kind, seed, x, h) in saints]
    # grips per pose (the poles pivot in the fists)
    grips_all, halos = [], []
    for i, (kind, seed, x, h) in enumerate(saints):
        g = {}
        for key in ("rest", "brace", "charge"):
            a = _anchors(figs[i], x, h, _icon_pose(i, key))
            gx, gy = a["grip"]
            g[key] = (float(gx), float(gy), float(a["pole"][2]))
            if i == wall.CENTRE:
                T.setdefault("head", {})[key] = tuple(float(v) for v in a["head"])
        a = _anchors(figs[i], x, h, _icon_pose(i, "rest"))
        halos.append(tuple(float(v) for v in a["halo"]))
        hx, hy, hr = halos[-1]
        g["L0"] = g["rest"][1] - (hy - hr - 12.0 - CLOTH_H)  # collapsed lance: the furled roll sits above the halo
        grips_all.append(g)
    # the fixed tile layout of the whole wall: andamento follows the saints at rest (cached by vx.mosaic)
    guide = _paint_wall_layers(0.5, figs, saints, halos=halos)
    lay = mz.Layout.flow(guide["color"], tile=12.0, seed=5, extent=(wall.WALL_W, wall.WALL_H),
                         fine=guide["fine"] > 0.5, fine_tile=5.0, gold=guide["gold"] > 0.5,
                         ground=guide["figs"] < 0.5, ground_style="rows", echo=2)
    static = _paint_wall_layers(SC, None, saints, layers=("color", "gold"), halos=halos)
    tracks = _simulate_flags(S, T, grips_all)
    txy = lay.xy + np.array([wall.WX0, wall.WY0], np.float32)
    st = dict(T=T, saints=saints, figs=figs, grips=grips_all, lay=lay, static=static, tracks=tracks, txy=txy,
              tl=S.tl, S_start=S.start, kd=cKDTree(lay.xy))
    # static gold fraction per tile (the radio rings run through the gold ground only)
    gm = static["gold"]
    ix = np.clip((lay.xy[:, 0] * SC).astype(int), 0, gm.shape[1] - 1)
    iy = np.clip((lay.xy[:, 1] * SC).astype(int), 0, gm.shape[0] - 1)
    st["gold_frac"] = gm[iy, ix].astype(np.float32)
    # Schism Actual's voice envelope (local seconds, 200 Hz) for the radio rings
    st["env_t"] = np.arange(-1.0, S.dur + 0.5, 1 / 200.0)
    st["env"] = np.array([S.tl.mouth("COMMANDER", S.start + x) for x in st["env_t"]], np.float32)
    _storm_setup(st)
    _cracks_setup(st)
    st["storm_rgb"], st["storm_gold"] = _storm_colours(st)
    fb = min(S.n - 1, int(round(T["breach"] * S.fps)))
    st["cloth0"] = [tr.cloth_center(fb) for tr in tracks]
    _export_foley(S, st)
    return st


# ============================================================ per-frame cartoon (static wall + figure sprites)
_SPR = {}


def _sprite(st, i, prm, s, part="all"):
    """figure i rendered at scale s over its bbox: (x0, y0 wall coords, rgb premult, alpha, gold, anchors)"""
    kind, seed, x, h = st["saints"][i]
    key = (i, part, prm["pose"], round(prm["goggles"], 2), round(prm["mouth"], 2), round(prm["wind"], 1),
           round(prm["pole_ang"], 3), s)
    r = _SPR.get(key)
    if r is not None:
        return r
    fig = st["figs"][i]
    kw = dict(pose=prm["pose"], t=prm["t"], goggles=prm["goggles"], wind=prm["wind"], pole_ang=prm["pole_ang"])
    a = fig.anchors(x, wall.GROUND, h, **kw)
    m = 40 + 60 * prm["wind"]
    bx0, by0, bx1, by1 = a["bbox"]
    x0, y0 = math.floor(bx0 - m), math.floor(by0 - m)
    w, hh = int(math.ceil((bx1 + m - x0) * s)), int(math.ceil((by1 + m - y0) * s))
    out = []
    for layer in ("color", "gold"):
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, hh)
        ctx = cairo.Context(surf)
        ctx.scale(s, s)
        ctx.translate(-x0, -y0)
        fig.draw(ctx, x, wall.GROUND, h, mouth=prm["mouth"], layer=layer, part=part, **kw)
        if kind == "schism_actual" and a.get("orb") and part != "front":
            wall.paint_orb(ctx, *a["orb"], layer)
        surf.flush()
        buf = np.ndarray((hh, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
        if layer == "color":
            out.append(buf[..., [2, 1, 0]].astype(np.float32) / 255.0)
        out.append(buf[..., 3].astype(np.float32) / 255.0)
    r = (x0, y0, out[0], out[1], out[2], a)
    if len(_SPR) > 300:
        _SPR.clear()
    _SPR[key] = r
    return r


def _paste(dst, x0, y0, src, a=None):
    """composite a wall-space sprite (premultiplied, alpha a) into a full-wall image at scale SC (clipped)"""
    px, py = int(round((x0 - wall.WX0) * SC)), int(round((y0 - wall.WY0) * SC))
    hh, ww = src.shape[:2]
    H_, W_ = dst.shape[:2]
    sx0, sy0 = max(0, -px), max(0, -py)
    sx1, sy1 = min(ww, W_ - px), min(hh, H_ - py)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    d = dst[py + sy0:py + sy1, px + sx0:px + sx1]
    s_ = src[sy0:sy1, sx0:sx1]
    if a is None:
        d[...] = np.maximum(d, s_)
    else:
        aa = a[sy0:sy1, sx0:sx1]
        d[...] = s_ + d * (1 - (aa[..., None] if d.ndim == 3 else aa))


def _cartoon(fc, st, t=None, T_glob=None):
    """full-wall cartoon (rgb, gold mask) at scale SC + per-saint (params, anchors) for this frame"""
    t = fc.t if t is None else t
    T_glob = fc.T if T_glob is None else T_glob
    rgb = st["static"]["color"].copy()
    gold = st["static"]["gold"].copy()
    prms, anchors = [None] * wall.N_SAINTS, [None] * wall.N_SAINTS
    for i in _fig_order():
        prm = _fig_params(i, t, T_glob, st["T"], st["tl"])
        x0, y0, frgb, fa, fg, a = _sprite(st, i, prm, SC)
        prms[i], anchors[i] = prm, a
        _paste(rgb, x0, y0, frgb, fa)
        _paste(gold, x0, y0, fg, fa)
    return rgb, gold, prms, anchors


# ============================================================ storm (the wall blasting toward camera)
def _storm_setup(st):
    """the wall fractures from the breach point outward along an irregular front, in chunks (Voronoi cells of
    ~70 units) that break away together and tumble toward the camera; the nearest fragments fly past the lens."""
    T = st["T"]
    xy = st["txy"]
    n = len(xy)
    rng = np.random.default_rng(55)
    d = np.hypot(xy[:, 0] - BREACH_PT[0], (xy[:, 1] - BREACH_PT[1]) * 1.15)
    nc = int(wall.WALL_W * wall.WALL_H / 70.0 ** 2)
    cells = rng.uniform((wall.WX0, wall.WY0), (wall.WX1, wall.WY1), (nc, 2))
    _, cid = cKDTree(cells).query(xy)
    cr = np.random.default_rng(56)
    c_delay, c_speed = cr.uniform(0, 1, nc), cr.uniform(0.7, 1.35, nc)
    c_dir = cr.normal(0, 1, (nc, 2))
    c_axis = cr.normal(0, 1, (nc, 3))
    front = 0.5 + 0.5 * value_noise2(xy[:, 0] / 230.0, xy[:, 1] / 230.0, seed=4)
    st["t_rel"] = (T["breach"] + d / 3000.0 + 0.07 * front + 0.05 * c_delay[cid] + rng.uniform(0, 0.012, n)
                   ).astype(np.float32)
    near = np.exp(-d / 520.0)
    st["vz"] = ((1100 + 3400 * near) * c_speed[cid] * rng.uniform(0.85, 1.15, n)).astype(np.float32)
    rad = np.stack([xy[:, 0] - BREACH_PT[0], xy[:, 1] - BREACH_PT[1]], 1) / (d[:, None] + 1e-3)
    dirs = rad + 0.35 * c_dir[cid]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-6
    vr = (250 + 700 * near) * rng.uniform(0.6, 1.2, n)
    st["vxy"] = (dirs * vr[:, None] - np.stack([np.zeros(n), rng.uniform(0, 200, n)], 1)).astype(np.float32)
    ax = c_axis[cid] + 0.4 * rng.normal(0, 1, (n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    st["spin"] = (ax * rng.uniform(5, 22, (n, 1))).astype(np.float32)
    st["d_breach"] = d.astype(np.float32)
    st["tremble"] = rng.uniform(0, 2 * np.pi, (n, 2)).astype(np.float32)
    st["tremble_w"] = rng.uniform(35, 75, n).astype(np.float32)


def _storm_colours(st):
    """the colours of every tessera at the instant of the blast (the frieze in its charge)"""
    T = st["T"]
    rgb, gold, _, _ = _cartoon(None, st, T["breach"], st["S_start"] + T["breach"])
    return mz.sample(rgb, st["lay"]), mz.sample_mask(gold, st["lay"])


CAM_DIST = 1500.0


def _mz_view(cam):
    """the vx.mosaic v1 camera for this frame (layout coords; perspective twin of the 2D view)"""
    cx, cy, z, rot, sx, sy = cam
    return mz.Camera.frontal(cx - wall.WX0, cy - wall.WY0, zoom=z, dist=CAM_DIST, at=(960.0 + sx, 540.0 + sy))


def _storm_pos(st, idx, t):
    tau = t - st["t_rel"][idx]
    lxy = st["lay"].xy[idx]
    v = st["vxy"][idx]
    z = st["vz"][idx] * tau + 900.0 * tau * tau
    x = lxy[:, 0] + v[:, 0] * tau
    y = lxy[:, 1] + v[:, 1] * tau + 450.0 * tau * tau
    return np.stack([x, y, z], 1), tau


def _storm(fc, st, img, cam, t):
    """released tesserae tumbling toward and past the camera, silhouetted against the light behind the wall"""
    T = st["T"]
    if t < T["breach"]:
        return img
    idx = np.nonzero(st["t_rel"] <= t)[0]
    if len(idx) == 0:
        return img
    P, tau = _storm_pos(st, idx, t)
    keep = P[:, 2] < CAM_DIST * 0.965
    idx, P, tau = idx[keep], P[keep], tau[keep]
    view = _mz_view(cam)
    scr, _ = view.project3(P)
    P0, _ = _storm_pos(st, idx, t - 1.0 / 48.0)
    scr0, _ = view.project3(np.where(P0[:, 2:3] > 0, P0, P))
    size_px = st["lay"].size[idx] * cam[2] * CAM_DIST / np.maximum(CAM_DIST - P[:, 2], 1.0)
    on = ((scr[:, 0] > -size_px) & (scr[:, 0] < W + size_px) & (scr[:, 1] > -size_px) & (scr[:, 1] < H + size_px))
    idx, P, tau, scr, scr0 = idx[on], P[on], tau[on], scr[on], scr0[on]
    if len(idx) == 0:
        return img
    rgb, gold = st["storm_rgb"], st["storm_gold"]
    tiles = mz.Tiles(P[:, :2], ang=st["lay"].ang[idx], size=st["lay"].size[idx], rgb=rgb[idx], gold=gold[idx],
                     z=P[:, 2], rot=st["spin"][idx] * tau[:, None], stretch=(0.45 * (scr - scr0)).astype(np.float32),
                     ids=idx)
    back = [mz.Light(dir=(0.0, 0.25, -1.0), color=(1.0, 0.93, 0.72), power=1.6)] + _look(fc)["light"]
    rgba = mz.render_tiles(fc, tiles, view=view, loose=True, return_alpha=True, light=back, ambient=0.07)
    return over(img, rgba)


def _cracks_setup(st):
    """a branching crack network radiating from the breach point (jagged polylines sampled every ~3 units);
    per tile: distance to the nearest crack, the crack's arc length there (the front reaches it when s_max > s)
    and the unit vector from the crack to the tile (the side it is shoved to when the crack opens)."""
    rng = np.random.default_rng(77)
    pts, ss = [], []

    def grow(x, y, ang, s, length, depth):
        while length > 0:
            step = rng.uniform(14.0, 30.0)
            ang += rng.normal(0.0, 0.28)
            nx, ny = x + math.cos(ang) * step, y + math.sin(ang) * step
            m = max(2, int(step / 3.0))
            for j in range(m):
                a = j / m
                pts.append((x + (nx - x) * a, y + (ny - y) * a))
                ss.append(s + step * a)
            s, length, x, y = s + step, length - step, nx, ny
            if depth < 2 and rng.random() < 0.07:
                grow(x, y, ang + rng.choice((-1.0, 1.0)) * rng.uniform(0.45, 1.2), s,
                     length * rng.uniform(0.3, 0.7), depth + 1)

    n0 = 11
    for k in range(n0):
        grow(BREACH_PT[0], BREACH_PT[1], 2 * math.pi * k / n0 + rng.normal(0, 0.18), 0.0, rng.uniform(900, 2000), 0)
    P, S_ = np.array(pts, np.float32), np.array(ss, np.float32)
    d, j = cKDTree(P).query(st["txy"])
    nv = st["txy"] - P[j]
    nv /= np.linalg.norm(nv, axis=1, keepdims=True) + 1e-6
    st["crack_d"], st["crack_s"], st["crack_n"] = d.astype(np.float32), S_[j], nv.astype(np.float32)


def _crack_front(t, T):
    """how far (arc length, wall units) the cracks have run: a jump on each "Breach!", a creep in the calm"""
    s = 0.0
    for (b0, _), inc in zip(T["b"], (110.0, 150.0, 220.0)):
        s += inc * ease_out(clamp((t - b0) / 0.08))
    return s + 1700.0 * ease_in(clamp((t - T["calm"]) / (T["breach"] - T["calm"])), 2)


# ============================================================ per-tile effects on the wall (vx.mosaic v1)
def _tile_fx(fc, st, t, prms, anchors):
    T = st["T"]
    n = len(st["lay"])
    out = {}
    # radio: rings of glint run through the gold ground from the headset, carrying the voice envelope
    hx, hy = anchors[wall.CENTRE]["mouth"]
    d = np.hypot(st["txy"][:, 0] - hx, st["txy"][:, 1] - hy)
    ts = t - d / 1400.0
    env = np.interp(ts, st["env_t"], st["env"], left=0.0, right=0.0)
    ring = (0.5 + 0.5 * np.cos(2 * np.pi * (d / 95.0 - t * 14.7))) ** 3
    gain = 1.0 + 0.9 * env * ring * np.exp(-d / 1100.0) * st["gold_frac"]
    # dead calm before the blast: the wall darkens as the light gathers behind it
    calm = smoothstep(T["calm"], T["breach"], t)
    out["gain"] = (gain * (1.0 - 0.3 * calm)).astype(np.float32)
    # phosphor-green lens glass
    emit = np.zeros((n, 3), np.float32)
    for i in range(wall.N_SAINTS):
        p = _power(t, T["gog"][i], i, T)
        if p <= 0.01:
            continue
        for (lx, ly, lr) in anchors[i]["lenses"]:
            ids = st["kd"].query_ball_point((lx - wall.WX0, ly - wall.WY0), lr * 1.1)
            if ids:
                emit[ids] = NVG_GLOW_LIN * (1.6 * p)
    out["emit"] = emit
    if t >= T["b"][0][0]:
        # the wall cracks from Schism Actual's breastplate: light blazes through the joints along the cracks
        s_max = _crack_front(t, T)
        opened = st["crack_s"] < s_max
        u = clamp((t - T["calm"]) / (T["breach"] - T["calm"]))
        age = np.clip((s_max - st["crack_s"]) / 250.0, 0.0, 1.0)
        dist = st["crack_d"]
        core = np.exp(-dist / 7.0) * (0.55 + 1.2 * u) + np.exp(-dist / 34.0) * (0.06 + 0.3 * u)
        lit = np.where(opened, core * (0.3 + 0.7 * age), 0.0)
        lit = lit * (0.85 + 0.15 * np.sin(t * 57.0 + st["tremble"][:, 0]))
        lit += 0.8 * np.exp(-st["d_breach"] / 150.0) * u * u
        if t >= T["breach"]:
            lit = np.maximum(lit, 0.35)
        out["joint_emit"] = (FG_LIN[None, :] * lit[:, None]).astype(np.float32)
        # the cracks open (tiles shoved apart across them); in the calm the whole wall trembles
        opn = np.where(opened, 1.6 * np.exp(-dist / 18.0) * (0.4 + 0.6 * age), 0.0)
        off = st["crack_n"] * opn[:, None]
        if t >= T["calm"] - 0.05:
            amp = 0.6 * smoothstep(0.0, 1.0, u) + 1.3 * smoothstep(0.75, 1.0, u)
            ph, w = st["tremble"], st["tremble_w"]
            off = off + amp * np.stack([np.sin(t * w + ph[:, 0]), np.cos(t * w * 1.13 + ph[:, 1])], 1)
        out["offset"] = off.astype(np.float32)
    if t >= T["breach"]:
        out["present"] = np.where(st["t_rel"] > t, 1, -1).astype(np.int8)     # -1: bare, the light shows through
    return out


def _fist_tiles(st, prms, anchors):
    """tiles of the fists that hold the poles (re-set over the poles so the hands grip them)"""
    ids = []
    for i in range(wall.N_SAINTS):
        x0, y0, frgb, fa, fg, a = _sprite(st, i, prms[i], SC, part="front")
        for g in anchors[i].get("grips", [anchors[i]["grip"]]):
            cand = st["kd"].query_ball_point((g[0] - wall.WX0, g[1] - wall.WY0), 60.0)
            for j in cand:
                px = int((st["txy"][j, 0] - x0) * SC)
                py = int((st["txy"][j, 1] - y0) * SC)
                if 0 <= py < fa.shape[0] and 0 <= px < fa.shape[1] and fa[py, px] > 0.5:
                    ids.append(j)
    m = np.zeros(len(st["lay"]), np.float32)
    if ids:
        m[np.array(ids)] = 1.0
    return m


# ============================================================ render
def render(fc, st):
    T = st["T"]
    t = fc.t
    cam = _camera(t, T)
    rgb, gold, prms, anchors = _cartoon(fc, st)
    img, fx = _render_wall(fc, st, rgb, gold, cam, t, prms, anchors)
    if t < T["breach"] + 0.1:
        img = _lens_glow(fc, st, img, cam, anchors)
    top = Canvas(fc)
    _draw_poles(top.ctx, st, t, cam)
    img = top.over(img)
    if t < T["breach"]:
        # the fists are re-set over the poles (a layout of just their tesserae, with their own bedding)
        idx = np.nonzero(_fist_tiles(st, prms, anchors) > 0.5)[0]
        if len(idx):
            kw = {k: v[idx] for k, v in fx.items() if k in ("gain", "emit", "offset", "joint_emit")}
            layer = mz.render(fc, rgb, st["lay"].subset(idx), gold=gold, view=_mz_view(cam), return_alpha=True,
                              **_look(fc), **kw)
            img = over(img, layer)
    # after the blast: the light front swallows the wall and the lances' shafts; the storm flies in front of it
    img = _flood(fc, st, img, cam, t)
    img = _storm(fc, st, img, cam, t)
    cl = Canvas(fc)
    _draw_cloth(cl.ctx, st, t, fc.f, cam)
    img = cl.over(img)
    # hand-off to m06: flat flag_glow on the last frame
    wf = smoothstep(T["end"] - 3.5 / fc.fps, T["end"] - 1.0 / fc.fps, t)
    img = img * (1 - wf) + FG * wf
    return np.clip(img, 0, 1).astype(np.float32)


def _look(fc):
    """the basilica at night: one candle below Schism Actual makes a warm pool, two weaker ones at the flanks;
    little ambient, the gold wakes only where the flames reach it (world = layout chart coords + z)"""
    L = lambda x, y, z: (x - wall.WX0, y - wall.WY0, z)
    lights = (mz.candles(fc.T, [L(960, 1040, 230)], power=4.2, radius=360, flicker_amt=0.3, sway=6.0, seed=5) +
              mz.candles(fc.T, [L(380, 1080, 240), L(1540, 1080, 240)], power=1.1, radius=300, flicker_amt=0.3,
                         sway=5.0, seed=9))
    return dict(light=lights, ambient=0.06, env=1.35)


def _render_wall(fc, st, rgb, gold, cam, t, prms, anchors):
    fx = _tile_fx(fc, st, t, prms, anchors)
    kw = dict(fx)
    if t >= st["T"]["breach"]:
        kw.update(bg=_light_bg(fc, st, cam, t), holes="bg")
    return mz.render(fc, rgb, st["lay"], gold=gold, view=_mz_view(cam), **_look(fc), **kw), fx


def _light_bg(fc, st, cam, t):
    """the flag-yellow light behind the wall, seen through the breach (screen image)"""
    T = st["T"]
    bx, by = _to_screen(cam, *BREACH_PT)
    ys = (np.arange(fc.h, dtype=np.float32) + 0.5) / fc.s - by
    xs = (np.arange(fc.w, dtype=np.float32) + 0.5) / fc.s - bx
    r2 = xs[None, :] ** 2 + ys[:, None] ** 2
    k = clamp((t - T["breach"]) / (T["end"] - T["breach"]))
    inten = 0.9 + 0.35 * np.exp(-r2 / (300.0 + 1500.0 * k) ** 2)
    return np.clip(FG[None, None, :] * inten[..., None], 0, 1).astype(np.float32)


def _flood(fc, st, img, cam, t):
    """after the blast a front of flag-yellow light races out from the breach (behind the storm and the flags)"""
    T = st["T"]
    if t < T["breach"]:
        return img
    tau = t - T["breach"]
    span = T["end"] - 1.0 / fc.fps - T["breach"]
    R = 1600.0 * clamp(tau / span) ** 1.7
    bx, by = _to_screen(cam, *BREACH_PT)
    ys = (np.arange(fc.h, dtype=np.float32) + 0.5) / fc.s - by
    xs = (np.arange(fc.w, dtype=np.float32) + 0.5) / fc.s - bx
    r = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)
    core = np.clip((R - r) / 300.0 + 0.3, 0.0, 1.0)
    halo = np.exp(-np.maximum(r - R, 0.0) / 220.0) * min(1.0, tau / 0.1) * 0.3
    a = np.clip(core + halo * (1 - core), 0.0, 1.0)[..., None]
    flash = 0.25 * math.exp(-tau / 0.05)
    return img * (1 - a) + FG * a + FG * flash


def _lens_glow(fc, st, img, cam, anchors):
    T = st["T"]
    z = cam[2]
    g = Canvas(fc)
    ctx = g.ctx
    any_on = False
    for i in range(wall.N_SAINTS):
        p = _power(fc.t, T["gog"][i], i, T)
        if p <= 0.01:
            continue
        any_on = True
        q = min(1.0, p)
        for (lx, ly, lr) in anchors[i]["lenses"]:
            sx, sy = _to_screen(cam, lx, ly)
            r = lr * z
            ctx.set_source(rad_grad(sx, sy, 0, r * 3.2, [(0, NVG_GLOW, 0.42 * q), (0.3, NVG_GLOW, 0.14 * q),
                                                        (1, NVG_GLOW, 0.0)]))
            ctx.arc(sx, sy, r * 3.2, 0, 2 * math.pi)
            ctx.fill()
            ctx.set_source(rad_grad(sx, sy, 0, r * 1.05, [(0, NVG_CORE, min(1, 0.9 * p)),
                                                         (0.7, NVG_GLOW, min(1, 0.8 * p)), (1, NVG_GLOW, 0.0)]))
            ctx.arc(sx, sy, r * 1.05, 0, 2 * math.pi)
            ctx.fill()
    if not any_on:
        return img
    return img + g.rgba()[..., :3] * 1.25


def _approach(st, i, t):
    """after the blast each flag (carried by its unseen saint) charges through the breach toward and past the
    camera: perspective scale (1 = on the wall plane)"""
    T = st["T"]
    tau = t - T["breach"] - 0.012 * abs(i - wall.CENTRE) - 0.08
    if tau <= 0:
        return 1.0
    z = 900.0 * tau + 3000.0 * tau * tau                       # design units toward the camera
    return 1.0 / max(0.05, 1.0 - z / CAM_DIST)


def _flag_ctx(ctx, st, i, t, cam):
    """push the per-flag transform (approach in screen space, then camera). The bearers lower their lances as they
    charge, so each flag grows about a point high above it: it swells and sweeps DOWN across the frame, a little
    outward, and out past the lens."""
    ctx.save()
    f = _approach(st, i, t)
    if f != 1.0:
        bx, _ = _to_screen(cam, *BREACH_PT)
        px, py = _to_screen(cam, *st["cloth0"][i])
        cx, cy = bx + (px - bx) * 0.55, py - 520.0
        ctx.translate(cx, cy)
        ctx.scale(f, f)
        ctx.translate(-cx, -cy)
    _apply_cam(ctx, cam)


def _draw_poles(ctx, st, t, cam):
    """the telescoping shafts (the top section is drawn with its cloth). After the blast the shafts are burnt out
    by the glare within ~0.2 s: only the flags and their top sections charge on."""
    T = st["T"]
    lt = (-0.45, -0.8, 0.4)
    al = 1.0 - smoothstep(T["breach"] + 0.02, T["breach"] + 0.2, t)
    if al <= 0.0:
        return
    for i in _fig_order():
        _flag_ctx(ctx, st, i, t, cam)
        butt, (ux, uy), tip, joints = _pole_geom(i, t, T, st["grips"][i])
        # base tube from the butt to its mouth, then each inner section out of the previous one
        draw_pole(ctx, butt[0], butt[1], joints[0][0], joints[0][1], POLE_W[0], alpha=al, light=lt)
        for c in range(3):
            a, b = joints[c], joints[c + 1]
            if math.hypot(b[0] - a[0], b[1] - a[1]) < 0.5:
                continue
            draw_pole(ctx, a[0] - ux * 6, a[1] - uy * 6, b[0], b[1], POLE_W[c + 1], alpha=al, light=lt)
        for c in range(3):                                   # wooden collars at the joints
            a = joints[c]
            draw_pole(ctx, a[0] - ux * 7, a[1] - uy * 7, a[0] + ux * 2, a[1] + uy * 2, POLE_W[c] + 3.0, alpha=al,
                      light=lt)
        ctx.restore()


def _draw_cloth(ctx, st, t, f, cam):
    T = st["T"]
    lt = (-0.45, -0.8, 0.4)
    back = smoothstep(T["calm"], T["breach"], t)
    if back > 0:
        lt = (-0.45 * (1 - back), -0.8 * (1 - back) + 0.2 * back, 0.4 * (1 - back) - 1.0 * back)
    for i in _fig_order():
        if _approach(st, i, t) > 12:
            continue
        _flag_ctx(ctx, st, i, t, cam)
        tr = st["tracks"][i]
        V = tr.verts[min(f, tr.n - 1)]
        _, (ux, uy), tip, joints = _pole_geom(i, t, T, st["grips"][i])
        a = joints[2]
        draw_cloth(ctx, V, light=lt, glow=0.35 * back, pole_seg=(a[0], a[1], tip[0], tip[1], POLE_W[3]))
        ctx.restore()


def post(fc, st):
    T = st["T"]
    u = smoothstep(T["breach"] + 0.3, T["end"] - 1.0 / fc.fps, fc.t)
    return dict(vignette=0.32 * (1 - u), grain=0.0,
                bloom=(0.42 + 0.3 * pulse(fc.t, T["breach"], 0.2)) * (1 - u))


# ============================================================ sound: every hit exported for Foley-2 / Composer-2
def _export_foley(S, st):
    T = st["T"]
    G = lambda t: S.start + t
    saints = st["saints"]
    pan = foley.screen_pan
    ev = [dict(t=G(0.0), kind="cut", strength=0.7, pan=0.0, desc="hard cut onto the frieze of warrior saints"),
          dict(t=G(T["k01"] - 0.06), kind="radio", strength=0.6, pan=0.0, desc="radio key-up / squelch: Schism Actual")]
    for i, (kind, seed, x, h) in enumerate(saints):
        ev.append(dict(t=G(T["gog"][i] + 0.11), kind="goggles", strength=0.9 if i == wall.CENTRE else 0.6,
                       pan=pan(x), desc=f"NVG mount snaps down + tube whine, saint {i} (to {G(T['gog'][i] + 0.5):.2f})"))
    ev.append(dict(t=G(min(T["gog"])), kind="goggles_ripple", strength=0.5, pan=0.0,
                   desc=f"goggle ripple from the centre outward (to {G(max(T['gog']) + 0.11):.2f})"))
    ev.append(dict(t=G(T["target"] + 0.1), kind="nvg_flare", strength=0.5, pan=0.0, desc="all NVG eyes flare on 'Target'"))
    ev.append(dict(t=G(T["method"]), kind="armour", strength=0.8, pan=0.0,
                   desc=f"eleven lamellar cuirasses: the lances swing into a fan (to {G(T['method'] + 0.35):.2f})"))
    for c, tc in enumerate(T["clack"]):
        for i, (kind, seed, x, h) in enumerate(saints):
            jit = 0.012 * (hash01(i * 3 + c, 21) - 0.5)
            ev.append(dict(t=G(tc + jit + 0.045), kind="telescope", strength=0.75 + 0.1 * c, pan=pan(x),
                           desc=f"telescoping section {c + 1}/3 locks (clack), pole {i}"))
    for i, (kind, seed, x, h) in enumerate(saints):
        ev.append(dict(t=G(T["unfurl"][i]), kind="unfurl", strength=0.8, pan=pan(x),
                       desc=f"flag {i} unrolls and snaps open (to {G(T['unfurl'][i] + 0.35):.2f})"))
    for k, (b0, b1) in enumerate(T["b"]):
        ev.append(dict(t=G(b0 + 0.02), kind="step", strength=0.7 + 0.1 * k, pan=0.0,
                       desc=f"'Breach!' {k + 1}: eleven armoured saints stamp forward as one + lamellar rattle"))
        ev.append(dict(t=G(b0), kind="punch_in", strength=0.6, pan=0.0, desc=f"hard punch-in {k + 1}/3"))
    ev.append(dict(t=G(T["b"][2][0]), kind="charge", strength=0.8, pan=0.0,
                   desc=f"the charge: armour, boots, cloaks (to {G(T['breach']):.2f})"))
    ev.append(dict(t=G(T["calm"]), kind="calm", strength=0.6, pan=0.0,
                   desc=f"dead calm: flags fall limp; light leaks through every mortar joint, the wall creaks and ticks "
                        f"(to {G(T['breach']):.2f})"))
    ev.append(dict(t=G(T["breach"] - 0.25), kind="inhale", strength=0.7, pan=0.0,
                   desc=f"pre-blast suck-back (to {G(T['breach']):.2f})"))
    ev.append(dict(t=G(T["breach"]), kind="breach", strength=1.0, pan=0.0, desc="BREACH: the mosaic wall blasts toward camera"))
    ev.append(dict(t=G(T["breach"]), kind="tile_storm", strength=1.0, pan=0.0,
                   desc=f"tens of thousands of glass and gold tesserae fly at and past the camera (to {G(T['end']):.2f})"))
    ev.append(dict(t=G(T["breach"] + 0.05), kind="flags_whoosh", strength=0.9, pan=0.0,
                   desc=f"eleven flags fly over the camera (to {G(T['breach'] + 0.6):.2f})"))
    ev.append(dict(t=G(T["breach"] + 0.35), kind="glow", strength=1.0, pan=0.0,
                   desc=f"flag-yellow light floods the frame (flat by {G(T['end']):.2f})"))
    foley.export_events(S, "sfx", ev)
    for i, tr in enumerate(st["tracks"]):
        tr.export_foley(S, f"flag_{i:02d}")

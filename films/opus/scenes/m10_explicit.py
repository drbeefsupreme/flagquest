"""m10 - THE STICKY FLAG: EXPLICIT (187.5-212.0), the ending. Owner M10.

The apse of the Basilica of the Unbabeling at dawn. Out of m09's white-gold flash: a meadow of weld (the luteolin
herb) at the foot of the conch, the rising sun behind a real Flag, and Lutie and Master Crocus as a small donor
pair (square nimbi: the living). L04 Lutie offers him a small Flag (real cloth); C06 it is sticky - shown only by
behaviour: the pole will not leave her hand, the cloth glues itself to his sleeve (m10_cloth: vx.flag's solver
+ a sticky sleeve). L05 / C07 the koan, deadpan, the gold twinkles. N11 the camera pulls back through the arch
into the nave: the whole basilica (m10_world, m10_art), its congregation on every wall. 202.9 Glorious: every
figure raises a Flag at once (tiles flip to the raised pose, real cloth pops out everywhere). 204.6 EXPLICIT sets
itself tile by tile in the conch's gold band to the cantor's syllables (M01's titulus Band: the same letters as
INCIPIT); the donor inscription sets itself word by word to the sung "Vexillo gratias". 209.5 end card; the
pilgrim's candle (M01's candle) is snuffed, the tesserae go dark tile by tile, the Flags remain, then fade -
the sticky one last. Black by 212.0.

Files: m10_world (geometry, charts, camera, low-res caster for visibility), m10_art (cartoons), m10_pair (the
donor pair's choreography + the sticky Flag), m10_cloth (sticky cloth solver).
"""
import json
import math

import cairo
import cv2
import numpy as np

from vx import foley, icon
from vx import mosaic as mz
from vx.canvas import Canvas, set_color
from vx.config import AUDIO, C as PC, CACHE, W, H
from vx.ease import clamp, ease_in_out, smoothstep, fbm1
from vx.flag import Flag, FlagTrack, draw_cloth, draw_flag_field, draw_pole, POLE_W

from scenes import m01_candle as Cn
from scenes import m01_titulus as Ti
from scenes import m10_art as A
from scenes import m10_pair as P
from scenes import m10_world as Wd

POST = dict(bloom=0.34, bloom_thresh=0.74, bloom_radius=24.0, grain=0.0, vignette=0.3, saturation=1.02)
POP_MIN = 0.09                 # vx.flag's pop ramps alpha below 1/12: a Flag is shown only at pop >= this (opaque)
VERSION = "m10-v1"

TL = P.TL
T0, T_END = 187.5, 212.0
T_PULL = TL.line("N11")["start"]               # 199.6 the pull-back
T_GLOR = TL.cue("glorious")                    # 202.9
T_EXPL = TL.cue("explicit")                    # 204.6
T_GRAT = TL.cue("gratias")                     # 207.6
T_CARD = TL.cue("end_card")                    # 209.5
T_SNUFF = 210.45                               # the snuffer closes on the flame
T_WAVE = T_SNUFF + 0.08                        # the tesserae go dark from the band outward
T_FLAGS_OUT = (211.08, 211.46)                 # in the dark the congregation lowers its Flags, all at once
T_STICKY_OUT = (211.74, 211.9)                 # ... the sticky one lingers, then pops out
T_LAST = 211.42                                # the EXPLICIT band keeps the last light until here
T_INTO, T_BAND = 203.5, 206.3                  # after Glorious the camera flies through the arch to the band
EXPO = dict(floor=0.5, conch=1.0, hemi=0.9, arch=0.78, left=0.66, right=0.5, ceil=0.36, soffit=0.72, jambL=0.62,
            jambR=0.55)
T_TWINKLE = P.T_STICKY2[1] + 0.06              # the deadpan: gold twinkles after "This one is sticky."
R, AX, SY, U = Wd.R, Wd.AX, Wd.SPRING_Y, Wd.U

EXPLICIT_TEXT = "EXPLICIT·OPVS·SCHISMATICVM"
DONOR_TEXT = "HOC·OPVS·TESSELLAVIT·MACHINA·ANNO·SVRVEII·MMXXVI"
EXPL_SYL = [(0, 2, 204.62), (2, 5, 204.97), (5, 8, 205.22), (8, 10, 205.495), (10, 13, 205.745), (13, 19, 205.995),
            (19, 21, 206.245), (21, 23, 206.645), (23, 26, 206.895)]     # (glyph i0, i1, onset) Ex-pli-cit ...
GRATIAS_SYL = [207.60, 207.92, 208.22, 208.60, 208.94, 209.18]         # Ve-xil-lo gra-ti-as (vowel onsets)
DONOR_WORDS = [(0, 8), (8, 20), (20, 28), (28, 33), (33, 41), (41, 47)]  # HOC·OPVS· | TESSELLAVIT· | ... per syllable


def _gratias():
    """the choir's "Vexillo gratias" syllable onsets (the voice department's grid; fallback: GRATIAS_SYL)"""
    try:
        g = json.loads((AUDIO / "vocal" / "hymn_grid.json").read_text())["gratias"]
        on = [float(x["t"]) for x in g]
        if len(on) == len(GRATIAS_SYL):
            return on
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return GRATIAS_SYL


def _syllables():
    """cantor syllable onsets from the voice department (fallback: the values above)"""
    try:
        d = json.loads((CACHE / "foley" / "cantor_syllables.json").read_text())["explicit"]
        on = [s["t"] for s in d]
        if len(on) == len(EXPL_SYL):
            return [(a, b, t) for (a, b, _), t in zip(EXPL_SYL, on)]
    except (OSError, KeyError, ValueError):
        pass
    return EXPL_SYL


# ============================================================ surfaces
def _surfaces():
    return {
        "floor": dict(sid=Wd.S_FLOOR, surface=mz.Plane(Wd.PLANES[Wd.S_FLOOR][0], (1, 0, 0), (0, 0, 1)), sc=0.13, tile=30.0),
        "conch": dict(sid=Wd.S_CONCH, surface=mz.Conch(R, AX, SY), sc=0.75, tile=11.0),
        "hemi": dict(sid=Wd.S_HEMI, surface=mz.Hemicycle(R, AX, 0.0), sc=0.34, tile=13.0),
        "arch": dict(sid=Wd.S_ARCH, surface=mz.Plane(*Wd.PLANES[Wd.S_ARCH]), sc=0.25, tile=16.0),
        "left": dict(sid=Wd.S_LEFT, surface=mz.Plane(*Wd.PLANES[Wd.S_LEFT]), sc=0.17, tile=24.0),
        "right": dict(sid=Wd.S_RIGHT, surface=mz.Plane(*Wd.PLANES[Wd.S_RIGHT]), sc=0.17, tile=24.0),
        "ceil": dict(sid=Wd.S_CEIL, surface=mz.Plane((Wd.XR, Wd.CEIL_Y, Wd.ARCH_Z), (-1, 0, 0), (0, 0, 1)), sc=0.12,
                     tile=34.0),
        "soffit": dict(sid=Wd.S_SOFFIT, surface=mz.Vault(R, AX, SY, z0=Wd.ARCH_Z), sc=0.5, tile=13.0,
                       ext=(math.pi * R, Wd.ARCH_Z)),
        "jambL": dict(sid=Wd.S_JAMBL, surface=mz.Plane((AX - R, SY, Wd.ARCH_Z), (0, 0, -1), (0, 1, 0)), sc=0.5,
                      tile=13.0, ext=(Wd.ARCH_Z, Wd.HEMI_H)),
        "jambR": dict(sid=Wd.S_JAMBR, surface=mz.Plane((AX + R, SY, 0.0), (0, 0, 1), (0, 1, 0)), sc=0.5,
                      tile=13.0, ext=(Wd.ARCH_Z, Wd.HEMI_H)),
    }


ORDER = ["floor", "conch", "hemi", "soffit", "jambL", "jambR", "arch", "left", "right", "ceil"]
# the nave (seen only from afar or at a slant) is baked once into textures and projected per frame by the
# m10_world ray caster; the apse and the arch are set live every frame (tiles glint as the camera moves)
BAKED = {"floor": 0.16, "ceil": 0.14, "left": 0.36, "right": 0.36}     # texture px per chart unit
LIVE = ["conch", "hemi", "soffit", "jambL", "jambR", "arch"]
CULLED = ("conch", "hemi", "arch")                                      # per-time-bin visible tile subsets
BIN = 0.5


def _figs(name):
    if name == "conch":
        return [f for f in A.conch_figures() if f["role"] not in ("lutie", "crocus")]
    if name == "hemi":
        return A.hemi_figures()
    if name == "arch":
        return A.arch_figures()
    if name in ("left", "right"):
        return A.nave_figures(-1 if name == "left" else 1)
    return []


def _ornament(name, sc):
    if name == "conch":
        return lambda ctx: A.conch_ornament(ctx, sc)
    if name == "hemi":
        return A.hemi_ornament
    if name == "arch":
        return A.arch_ornament
    if name in ("left", "right"):
        return lambda ctx: A.nave_ornament(ctx, -1 if name == "left" else 1)
    if name == "floor":
        return A.floor_cartoon_paint
    if name == "soffit":
        return A.soffit_ornament
    if name in ("jambL", "jambR"):
        return A.jamb_ornament
    return lambda ctx: _ceiling(ctx)


def _ceiling(ctx):
    ctx.save()
    ctx.translate(Wd.XR - Wd.XL, 0)
    ctx.scale(-1, 1)
    A.ceiling_cartoon_paint(ctx)
    ctx.restore()


PAIR_SC = 3.0              # px per chart unit of the pair's own cartoon/layout
PAIR_TILE = 5.4 / 1.92     # the pair in opus vermiculatum: 5.4 physical units (k ~ 1.92 there), faces finer still
PAIR_UP = 3.0              # the pair's layout is built at 3x (mosaic joints/minimum sizes are in design units)


def _pair_layout():
    """the donor pair's own fine layout (their motion envelope over the whole choreography, in the conch chart).
    -> (layout shifted into conch chart units with the conch extent, envelope mask function for other scales)"""
    x0, y0, x1, y1 = P.bbox()
    ew, eh = x1 - x0, y1 - y0

    def env_draw(ctx, layer):
        ctx.translate(-x0, -y0)
        for T in np.arange(T0, T_END, 0.3):
            P.draw(ctx, float(T), layer="color")
    ce = icon.cartoon(env_draw, s=PAIR_SC, extent=(ew, eh), layers=("color", "fine"))
    env = ce["alpha"] > 0.3
    k = max(3, int(round(P.CH * 0.035 * PAIR_SC)) | 1)
    env = cv2.dilate(env.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    # faces and hands wherever they go during the scene: laid in the finest tiles
    kf = max(3, int(round(P.CH * 0.012 * PAIR_SC)) | 1)
    fine_all = cv2.dilate((ce["fine"] > 0.3).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kf, kf))) > 0
    bg = A.conch_crop(x0, y0, x1, y1, PAIR_SC)
    Tref = T_PULL + 0.4

    def draw(ctx, layer):
        if layer == "color":
            from vx.canvas import paint_array
            paint_array(ctx, bg, 0.0, 0.0, PAIR_SC)
        ctx.translate(-x0, -y0)
        P.draw(ctx, Tref, layer=layer)
    c = icon.cartoon(draw, s=PAIR_SC, extent=(ew, eh), layers=A.LAYERS)
    figa = icon.cartoon(lambda ctx, layer: (ctx.translate(-x0, -y0), P.draw(ctx, Tref, layer=layer)),
                        s=PAIR_SC, extent=(ew, eh), layers=("color",))["alpha"]
    g = A.gold_mask(c["rgb"])
    up = PAIR_UP
    lay = mz.Layout.flow(c["rgb"], tile=PAIR_TILE * up, extent=(ew * up, eh * up), fine=fine_all | (c["fine"] > 0.5),
                         fine_tile=PAIR_TILE * up * 0.62, gold=np.maximum(g, c["gold"]), silver=c["silver"],
                         pearl=c["pearl"], edges=c["edges"] > 0.5, ground=(g * (figa < 0.5)).astype(np.float32),
                         alpha=env.astype(np.float32), seed=77)
    lay = lay.copy()
    off = np.array([x0, y0], np.float32)
    lay.xy = lay.xy / up + off
    lay.poly = lay.poly / up + off
    lay.size = lay.size / up
    if lay.area is not None:
        lay.area = lay.area / (up * up)
    lay.extent = Wd.EXTENT[Wd.S_CONCH]
    lay.extent = (int(round(lay.extent[0])), int(round(lay.extent[1])))

    def env_at(sc):
        """the envelope rasterised over the whole conch chart at scale sc"""
        w, h = int(round(2 * R * sc)), int(round(R * sc))
        out = np.zeros((h, w), bool)
        M = np.array([[sc / PAIR_SC, 0, x0 * sc], [0, sc / PAIR_SC, y0 * sc]], np.float32)
        warped = cv2.warpAffine(env.astype(np.uint8) * 255, M, (w, h), flags=cv2.INTER_LINEAR) > 127
        out |= warped
        return out
    return lay, env_at, bg, (x0, y0, x1, y1)


def _hemi_density(u, v):
    """opus sectile and window glass are cut in bigger pieces than the smalti"""
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    d = np.ones(np.broadcast(u, v).shape)
    d = np.where(v > SY + 2330.0, 0.42, d)
    for phi in A.HEMI_WINDOWS:
        uc = Wd.HEMI_U0 + R * phi
        d = np.where((np.abs(u - uc) < 205.0) & (v > SY + 600.0) & (v < SY + 1940.0), 0.5, d)
    return d


def _arch_density(u, v):
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    return np.where(v > Wd.EXTENT[Wd.S_ARCH][1] - 1700.0, 0.42, np.ones(np.broadcast(u, v).shape))


DENSITY = {"hemi": _hemi_density, "arch": _arch_density}


def _build_surface(name, cfg, S):
    """cartoons (rest + raise), layout, per-tile colours/materials, figure anchors for one surface"""
    sid = cfg["sid"]
    ext = cfg.get("ext") or Wd.EXTENT[sid]
    sc, tile = cfg["sc"], cfg["tile"]
    figs = _figs(name)
    orn = _ornament(name, sc)
    extra = None
    rest = A.chart(ext, sc, ornament=orn, figs=figs, pose="rest", tile=tile, extra=extra)
    alpha = rest["alpha"] > 0.5
    fine = rest["fine"] > 0.5
    pair = None
    if name == "conch":
        band = A.conch_band_mask(sc)
        alpha &= ~(band & _band_core(sc))
        pair = _pair_layout()
        alpha &= ~pair[1](sc)
    kw = dict(tile=tile, extent=ext, fine=fine, gold=rest["goldall"], silver=rest.get("silver"),
              pearl=rest.get("pearl"), ground=rest["ground"], edges=rest["edges"] > 0.5, alpha=alpha.astype(np.float32),
              seed=11 + sid)
    if name == "conch":
        kw.update(density=cfg["surface"].k, ground_style="fan", fan_center=A.chart_of(0.0, A.SUN_H)[:2])
    elif name in DENSITY:
        kw.update(density=DENSITY[name])
    lay = mz.Layout.flow(rest["rgb"], **kw)
    rgbA = mz.sample(rest["rgb"], lay)
    out = dict(anchorsA=rest["anchors"], figs=figs, ext=ext)
    if pair is not None:
        play, _, pbg, prect = pair
        n0 = len(lay)
        rgbP = mz.sample(pbg, play, crect=prect)
        lay = lay + play
        rgbA = np.vstack([rgbA, rgbP])
        out["pair"] = dict(idx=np.arange(n0, len(lay)), lay=play, bg=pbg, rect=prect, sc=PAIR_SC)
    out.update(lay=lay, rgbA=rgbA)
    out["mat"] = lay.mat.copy()
    if figs:
        up = A.chart(ext, sc, ornament=orn, figs=figs, pose="raise", tile=tile, extra=extra,
                     layers=("color", "gold", "silver"))
        out["rgbB"] = mz.sample(up["rgb"], lay)
        out["anchorsB"] = up["anchors"]
        matB = lay.mat.copy()
        gB = mz.sample_mask(up["goldall"], lay)
        sB = mz.sample_mask(up["silver"], lay) if "silver" in up else np.zeros(len(lay))
        matB[:] = np.where(gB >= 0.5, mz.GOLD, np.where(sB >= 0.5, mz.SILVER, np.where(lay.mat == mz.STONE, mz.STONE, 0)))
        out["matB"] = matB.astype(np.uint8)
        changed = np.abs(out["rgbA"] - out["rgbB"]).max(1) > 0.04
        out["flip"] = np.nonzero(changed)[0]
    # glowing window glass (dawn light through the panes)
    glass = A.glass_mask(rest["rgb"]) * alpha
    out["glass"] = mz.sample_mask(glass, lay)
    # world positions of the tiles (for the lights-out wave) + random per tile
    P3 = cfg["surface"].embed(lay.xy.astype(np.float64))
    out["world"] = P3.astype(np.float32)
    return out


def _band_core(sc):
    """the central part of the conch's gold band that M01's titulus Band tiles (the rest stays conch layout)"""
    w, h = int(round(2 * R * sc)), int(round(R * sc))
    jj, ii = np.meshgrid((np.arange(w) + 0.5) / sc, (np.arange(h) + 0.5) / sc)
    a, b = jj - AX, ii - SY
    t = 2 * R * R / (a * a + b * b + R * R)
    return np.abs(t * a) < BAND_HALF


# ============================================================ the tituli (M01's Band, the same letters as INCIPIT)
BAND_HALF = 0.8 * R          # front-view half width of the EXPLICIT band's own tiles


def _front_to_chart(X, Y):
    dx, dy = X - AX, Y - SY
    dz = -np.sqrt(np.maximum(R * R - dx * dx - dy * dy, 1.0))
    s = R / (R - dz)
    return AX + dx * s, SY + dy * s


def explicit_band():
    """EXPLICIT band tiles in the conch chart: polys, colours, gold, per-tile set times (syllable-synced)"""
    wband = 2 * BAND_HALF
    band = Ti.Band([(EXPLICIT_TEXT, wband / 2)], wband, height=A.H_BAND, letter_h=0.56 * A.H_BAND, tile=11.0,
                   letter_tile=6.0, ground="gold", ink="lapis_d", seed=4)
    tl = band.tiles
    x, y, a, hu, hv = tl["x"], tl["y"], tl["ang"], tl["hu"], tl["hv"]
    c, s = np.cos(a), np.sin(a)
    corners = []
    for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        X = (AX - BAND_HALF) + x + c * du * hu - s * dv * hv
        Y = (SY - A.H_BAND) + y + s * du * hu + c * dv * hv
        corners.append(np.stack(_front_to_chart(X, Y), -1))
    poly = np.stack(corners, 1).astype(np.float32)
    cx, cy = _front_to_chart((AX - BAND_HALF) + x, (SY - A.H_BAND) + y)
    wins = {}
    for i0, i1, t in _syllables():
        n = i1 - i0
        for k, g in enumerate(range(i0, i1)):
            wins[g] = (t + k * 0.055, t + k * 0.055 + 0.16)
    t_set = band.set_times(wins, seed=5)
    return dict(xy=np.stack([cx, cy], 1).astype(np.float32), poly=poly, rgb=tl["rgb"], gold=tl["gold"],
                letter=tl["letter"], t_set=t_set, last=True,
                ghost=(0.62 * mz.SC["gold"] + 0.38 * mz.SC["sinopia"]).astype(np.float32))


def donor_band():
    """the donor inscription in the hemicycle's lapis band (isometric chart): gold letters, word by word"""
    L = math.pi * R * 0.62
    band = Ti.Band([(DONOR_TEXT, L / 2)], L, height=A.HEMI_DONOR_V[1] - 45.0, letter_h=118.0, tile=12.0,
                   letter_tile=6.5, ground="lapis_d", ink="gold", rule="gold", seed=8)
    tl = band.tiles
    u0 = Wd.HEMI_U0 - L / 2
    v0 = SY + 45.0
    x, y, a, hu, hv = tl["x"], tl["y"], tl["ang"], tl["hu"], tl["hv"]
    c, s = np.cos(a), np.sin(a)
    corners = [np.stack([u0 + x + c * du * hu - s * dv * hv, v0 + y + s * du * hu + c * dv * hv], -1)
               for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    wins = {}
    for (i0, i1), t in zip(DONOR_WORDS, _gratias()):
        for k, g in enumerate(range(i0, i1)):
            wins[g] = (t + k * 0.03, t + k * 0.03 + 0.12)
    t_set = band.set_times(wins, seed=9)
    return dict(xy=np.stack([u0 + x, v0 + y], 1).astype(np.float32), poly=np.stack(corners, 1).astype(np.float32),
                rgb=tl["rgb"], gold=tl["gold"], letter=tl["letter"], t_set=t_set, u_span=(u0, u0 + L),
                ghost=(0.85 * mz.SC["lapis_d"] + 0.15 * mz.SC["sinopia"]).astype(np.float32))


# ============================================================ camera path
def _orbit(target, eye, fov):
    t = np.asarray(target, float)
    e = np.asarray(eye, float)
    d = e - t
    return dict(t=t, d=np.log(np.linalg.norm(d)), yaw=math.atan2(d[0], d[2]), pitch=math.asin(d[1] / np.linalg.norm(d)),
                tf=math.log(math.tan(math.radians(fov) / 2)))


def _cam(o):
    dist = math.exp(o["d"])
    cp = math.cos(o["pitch"])
    d = np.array([math.sin(o["yaw"]) * cp, math.sin(o["pitch"]), math.cos(o["yaw"]) * cp]) * dist
    return Wd.Cam(o["t"] + d, o["t"], 2 * math.degrees(math.atan(math.exp(o["tf"]))))


def _mix(o0, o1, u):
    return {k: o0[k] + (o1[k] - o0[k]) * u for k in o0}


def _pair_center(T):
    """world point between the pair's heads and the Flag (the close-up's centre of interest)"""
    x = 0.5 * (P.LX + P.CX) + 0.04 * P.CH
    y = 0.5 * (P.LY + P.CY) - 0.64 * P.CH
    return np.array(Wd.conch_to_world(x, y), float)


def camera_keys():
    pc = _pair_center(T0)
    k = {}
    # close-up on the donor pair: from the nave axis, a little below their heads
    eye_cu = np.array([AX - 0.45 * (AX - pc[0]), pc[1] + 40.0, 700.0])
    k["cu0"] = _orbit(pc + np.array([0, -40.0, 0]), eye_cu + np.array([0, -60.0, 260.0]), 27.5)
    k["cu"] = _orbit(pc, eye_cu, 26.0)
    k["cu2"] = _orbit(pc + np.array([10.0, -8.0, 0]), eye_cu + np.array([20.0, 0, -90.0]), 25.2)
    # C07 deadpan: a slow push toward the Master
    cr = np.array(Wd.conch_to_world(P.CX, P.CY - 0.7 * P.CH), float)
    k["deadpan"] = _orbit(0.55 * pc + 0.45 * cr, eye_cu + np.array([60.0, -10.0, -380.0]), 22.8)
    # the whole basilica, from a pilgrim's height: the floor leads to the apse, the walls tower
    k["wide"] = _orbit((AX, Wd.FLOOR_Y - 8.7 * U, 0.0), (AX, Wd.FLOOR_Y - 3.6 * U, Wd.ARCH_Z + 19.5 * U), 66.0)
    k["wide2"] = _orbit((AX, Wd.FLOOR_Y - 8.8 * U, 0.0), (AX, Wd.FLOOR_Y - 3.6 * U, Wd.ARCH_Z + 21.0 * U), 66.0)
    # into the apse, to the band where EXPLICIT sets itself
    k["band"] = _orbit((AX, SY - 0.12 * R, -0.85 * R), (AX, SY + 0.35 * U, Wd.ARCH_Z + 3.0 * U), 50.0)
    # the end card: the conch's sun and meadow, EXPLICIT, the donor band, the windows; the candle below
    k["end"] = _orbit((AX, SY + 0.35 * U, -0.8 * R), (AX, SY + 1.0 * U, Wd.ARCH_Z + 5.6 * U), 54.0)
    return k


def camera(T, K):
    if T < 188.6:
        o = _mix(K["cu0"], K["cu"], ease_in_out((T - T0) / 1.1, 2))
    elif T < P.T_THIS2 - 0.3:
        o = _mix(K["cu"], K["cu2"], ease_in_out((T - 188.6) / (P.T_THIS2 - 0.3 - 188.6), 2))
    elif T < T_PULL:
        o = _mix(K["cu2"], K["deadpan"], ease_in_out((T - P.T_THIS2 + 0.3) / (T_PULL - P.T_THIS2 + 0.3), 2))
    elif T < T_GLOR:
        u = (T - T_PULL) / (T_GLOR - T_PULL)
        u = ease_in_out(u, 3) * 0.985 + 0.015 * u
        o = _mix(K["deadpan"], K["wide"], u)
    elif T < T_INTO:
        o = _mix(K["wide"], K["wide2"], 0.3 * ease_in_out((T - T_GLOR) / (T_INTO - T_GLOR), 2))
    elif T < T_BAND:
        base = _mix(K["wide"], K["wide2"], 0.3)
        o = _mix(base, K["band"], ease_in_out((T - T_INTO) / (T_BAND - T_INTO), 2.4))
    elif T < T_CARD:
        o = _mix(K["band"], K["end"], ease_in_out((T - T_BAND) / (T_CARD - T_BAND), 2))
    else:
        o = K["end"]
    return _cam(o)


# ============================================================ flags
def _flag_world(tr, i, surface, z0=0.0):
    """FlagTrack in chart units (x, y chart; z toward the camera, chart units) -> world vertices (ny, nx, 3)"""
    V = tr.verts[i]
    bx, by, ba = (float(v) for v in tr.poles[i])
    uv = V[..., :2].reshape(-1, 2).astype(np.float64)
    Pw, T1, T2, N = surface.frame(uv)
    k = surface.k(uv[:, 0], uv[:, 1])
    z = (V[..., 2].reshape(-1).astype(np.float64) + z0) * k
    Pw = Pw + N * z[:, None]
    return Pw.reshape(V.shape[0], V.shape[1], 3), (bx, by, ba)


def draw_track(ctx, tr, i, surface, cam, glow=0.0, light=None, z0=0.0, pole=True, cloth=True, scale=1.0):
    """draw a chart-space FlagTrack through the 3D camera: pole (projected) then the cloth (projected verts).
    pole=False, cloth=True: only the cloth (depth-sorted against the pole drawn earlier). scale < 1 shrinks the
    whole Flag toward its pole base (how a Flag leaves: always opaque)."""
    if scale < POP_MIN:
        return
    Vw, (bx, by, ba) = _flag_world(tr, i, surface, z0)
    L = tr.meta["pole"]
    top = (bx + math.sin(ba) * L, by - math.cos(ba) * L)
    ends = np.array([[bx, by], top], np.float64)
    Pw, _, _, N = surface.frame(ends)
    k = surface.k(ends[:, 0], ends[:, 1])
    Pw = Pw + N * (z0 * k)[:, None]
    sp, dep = cam.project(Pw)
    if np.any(dep < 1.0):
        return
    sc = cam.focal / max(float(dep.mean()), 1.0)
    pw = max(0.6, tr.meta.get("pole_w", 3.0) * float(k.mean()) * sc * max(scale, 0.35))
    b0 = sp[0].copy()
    sp = b0 + (sp - b0) * scale
    if pole:
        draw_pole(ctx, sp[0, 0], sp[0, 1], sp[1, 0], sp[1, 1], pw, 1.0, light)
    if not cloth:
        return
    sl = min(1.0, (tr.meta["width"] + tr.meta["height"]) * 1.05 / L)
    seg = (sp[1, 0] + (sp[0, 0] - sp[1, 0]) * sl, sp[1, 1] + (sp[0, 1] - sp[1, 1]) * sl, sp[1, 0], sp[1, 1], pw)
    s2, d2 = cam.project(Vw.reshape(-1, 3))
    zc = -(d2 - float(dep.mean())) * sc
    s2 = b0 + (s2 - b0) * scale
    V = np.concatenate([s2, (zc * scale)[:, None]], 1).reshape(Vw.shape)
    draw_cloth(ctx, V, 1.0, glow, light, pole_seg=seg)


def central_flag(n):
    """the one Flag planted in the heap of tesserae before the rising sun (chart units of the conch)"""
    bx, by = A.flag_base_chart()
    k = float(Wd.conch_k(bx, by))
    pole = 2.2 * U / k
    fl = Flag(pole=pole, seed=31)
    tr = fl.simulate(n, lambda i: (bx, by, 0.02 * math.sin(i / 37.0)),
                     lambda i: (330.0 + 90.0 * math.sin(i / 53.0), 0.0), warmup=72)
    return tr


def congregation_flags(built):
    """every figure's raised pole (Glorious): world base, world tip direction, world pole length, seed"""
    base, tip, seeds, surf = [], [], [], []
    for name, b in built.items():
        if "anchorsB" not in b:
            continue
        S = b["surface"]
        for f, an in zip(b["figs"], b["anchorsB"]):
            if an is None or "pole" not in an:
                continue
            px, py, pa = (float(v) for v in an["pole"])
            k = float(np.asarray(S.k(np.array([px]), np.array([py]))).reshape(-1)[0])
            Lc = 1.05 * f["h"]                                    # pole length in chart units
            p0 = np.array([px, py])
            p1 = p0 + Lc * np.array([math.sin(pa), -math.cos(pa)])
            Pw = S.embed(np.stack([p0, p1]).astype(np.float64))
            base.append(Pw[0])
            tip.append(Pw[1])
            seeds.append(f["seed"] * 7 + len(seeds))
            surf.append(name)
    # the Flag on the empty throne above the arch
    tu, tv = A.throne_seat_chart()
    Sa = built["arch"]["surface"]
    Pw = Sa.embed(np.array([[tu, tv], [tu, tv - 330.0]], np.float64))
    base.append(Pw[0])
    tip.append(Pw[1])
    seeds.append(999)
    surf.append("arch")
    base, tip = np.array(base), np.array(tip)
    rng = np.random.default_rng(12)
    t_pop = T_GLOR - 0.03 + rng.uniform(0, 0.09, len(base))
    t_pop[-1] = T_GLOR + 0.05
    return dict(base=base, tip=tip, seeds=np.array(seeds), surf=surf, t_pop=t_pop)


def _pop(p):
    """Flags appear/leave by scale (vx.flag pop), never by transparency: below POP_MIN a Flag is simply not there"""
    p = np.asarray(p, np.float64)
    return np.where(p < POP_MIN, 0.0, p)


def lowered(T):
    """1 while the Flags are held up; they are lowered together in the dark after the snuff"""
    return 1.0 - smoothstep(T_FLAGS_OUT[0], T_FLAGS_OUT[1], T)


def draw_congregation_flags(ctx, cf, cam, T):
    pop = _pop(np.minimum(np.clip((T - cf["t_pop"]) / 0.55, 0, 1), lowered(T)))
    m = pop > 0
    if not m.any():
        return
    b, _ = cam.project(cf["base"][m])
    t, dt = cam.project(cf["tip"][m])
    _, db = cam.project(cf["base"][m])
    ok = (db > 50) & (dt > 50)
    vec = t - b
    L = np.hypot(vec[:, 0], vec[:, 1])
    ang = np.arctan2(vec[:, 0], -vec[:, 1])
    ok &= (L > 1.5) & (b[:, 0] > -300) & (b[:, 0] < W + 300) & (b[:, 1] > -300) & (b[:, 1] < H + 400)
    if not ok.any():
        return
    seeds = cf["seeds"][m][ok]
    wind = 0.55 + 0.25 * np.sin(seeds * 1.7)
    draw_flag_field(ctx, b[ok, 0], b[ok, 1], L[ok], T - T0, seeds=seeds, wind=wind, ang=ang[ok],
                    pop=pop[m][ok], light=(-0.35, -0.8, 0.5), lean=0.05)


# ============================================================ light
def dawn(T):
    """the dawn key light (directional, from the south-east clerestory) + ambient; the sun keeps rising"""
    rise = smoothstep(T0, T_GLOR, T)
    flash = 0.0
    if T > T_GLOR - 0.05:
        flash = 0.42 * math.exp(-max(0.0, T - T_GLOR) / 0.9) * smoothstep(T_GLOR - 0.05, T_GLOR + 0.08, T)
    power = (1.0 + 0.35 * rise + flash) * (1.0 - 0.45 * smoothstep(T_EXPL, T_BAND + 0.6, T))
    d = (0.42 - 0.12 * rise, -0.62 - 0.1 * rise, 0.72)
    key = mz.Light(dir=d, color=(1.0, 0.84, 0.64), power=power)
    fill = mz.Light(dir=(-0.5, -0.3, 0.8), color=(0.62, 0.7, 0.95), power=0.22)
    return [key, fill], 0.3 + 0.06 * rise


def candle_light(T, cam):
    """the pilgrim's candle, held just below the eye: it wakes the gold of the apse; snuffed at T_SNUFF"""
    lit, _ = Cn.snuff(T - T_SNUFF)
    k = smoothstep(T_EXPL + 0.2, T_BAND, T) * lit * Cn.flame_light(T)
    if k <= 0.0:
        return None
    pos = cam.eye + cam.f * 420.0 + cam.d * 260.0
    return mz.Light(pos=tuple(pos), color=(1.0, 0.72, 0.42), power=2.6 * k, radius=2600.0)


def expo(name, T):
    """per-surface exposure: the nave in shadow, the apse in the dawn; the close-up a touch brighter"""
    e = EXPO[name] * house(T)
    if name == "conch":
        e *= 1.0 + 0.16 * (1.0 - smoothstep(T_PULL, T_PULL + 1.5, T))
    return e


def house(T):
    """whatever light is left in the basilica after the wave (mortar, beds, windows) goes with it"""
    return 1.0 - smoothstep(211.5, 211.78, T)


def lights_out(world, T, seed=0, t0=None, speed=4200.0, origin=None, inward=False):
    """per-tile light gain for the end: the tesserae go dark tile by tile, from the candle's band outward
    (inward=True: from the far ends toward the origin)"""
    t0 = T_WAVE if t0 is None else t0
    if T < t0:
        return None
    origin = np.array([AX, SY + 150.0, -0.9 * R]) if origin is None else np.asarray(origin, float)
    d = np.linalg.norm(world - origin, axis=1)
    if inward:
        d = d.max() - d
    rnd = ((np.arange(len(world)) * 2654435761 + seed * 97) % 1000003) / 1000003.0
    t_off = t0 + d / speed + 0.14 * rnd
    return (1.0 - np.clip((T - t_off) / 0.2, 0, 1)).astype(np.float32)


# ============================================================ baked nave surfaces
def _bake(S, name, b, s_tex, sets=None, bands=()):
    """the surface's chart set in tesserae once (flat, lit by the dawn in its own frame) -> {'A': Tex, ['B': Tex],
    'aux': Tex (gold leaf, per-tile phase: the baked gold still twinkles)}. sets: {which: (rgb, mat)} (default: the
    surface's rest/raise colours); bands: titulus bands (ghost state) to set into the texture too."""
    import hashlib
    if sets is None:
        sets = {"A": (b["rgbA"], b["mat"])}
        if "rgbB" in b:
            sets["B"] = (b["rgbB"], b["matB"])
    h = hashlib.sha1()
    for part in (VERSION, name, str(len(b["lay"])), str(s_tex), str(len(bands))):
        h.update(part.encode())
    for w_ in sorted(sets):
        h.update(np.ascontiguousarray(sets[w_][0][::5]).tobytes())
    path = S.cache / f"bake_{name}_{h.hexdigest()[:16]}.npz"
    if path.exists():
        z = np.load(path)
        outs = {k: z[k] for k in z.files}
    else:
        surf = b["surface"]
        frame = getattr(surf, "ex", None)
        lights, amb = dawn(T_GLOR - 0.3)
        if frame is not None:
            loc = lambda d: (float(np.dot(d, surf.ex)), float(np.dot(d, surf.ey)), float(np.dot(d, surf.nrm)))
            Ls = [mz.Light(dir=loc(np.asarray(L.p, float)), color=tuple(L.color), power=L.power) for L in lights]
        else:
            # curved surfaces: the key light as it falls on the middle of the surface
            ew, eh = b["ext"]
            uv = np.array([[ew * 0.5, eh * 0.75]])
            _, T1, T2, N = surf.frame(uv)
            loc = lambda d: (float(d @ T1[0]), float(d @ T2[0]), float(d @ N[0]))
            Ls = [mz.Light(dir=loc(np.asarray(L.p, float)), color=tuple(L.color), power=L.power) for L in lights]

        class F:
            s = s_tex
            T = T_GLOR
            w = int(round(b["ext"][0] * s_tex))
            h = int(round(b["ext"][1] * s_tex))
        outs = {}
        for w_, (rgb, mat) in sets.items():
            img = mz.render(F, None, b["lay"], rgb=rgb, mat=mat, light=Ls, ambient=amb, s=s_tex, t=T_GLOR)
            for band in bands:
                tl = mz.Tiles(band["xy"], poly=band["poly"], nv=4, rgb=np.where(band["letter"][:, None], band["ghost"],
                              band["rgb"]), gold=np.where(band["letter"], 0.0, band["gold"]), seed=3)
                img = mz.render_tiles(F, tl, view=mz.View.affine(np.array([[1.0, 0, 0], [0, 1.0, 0]])), light=Ls,
                                      ambient=amb, bg=img, s=s_tex, size_px=(F.w, F.h), t=T_GLOR)
            outs[w_] = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        lab = mz.raster(b["lay"], s_tex)[0]
        L = np.where(lab >= 0, lab, 0)
        gold = (sets["A"][1] == mz.GOLD).astype(np.float32)
        ph = b["lay"].rnd
        aux = np.zeros(lab.shape + (3,), np.uint8)
        aux[..., 0] = (gold[L] * (lab >= 0) * 255).astype(np.uint8)
        aux[..., 1] = (ph[L] * 255).astype(np.uint8)
        outs["aux"] = aux
        np.savez(path, **outs)
    return {k: Wd.Tex(v, 1.0 / s_tex) for k, v in outs.items()}


BAKE_SID = {"floor": Wd.S_FLOOR, "ceil": Wd.S_CEIL, "left": Wd.S_LEFT, "right": Wd.S_RIGHT}
LIVE_SID = {"conch": Wd.S_CONCH, "hemi": Wd.S_HEMI, "arch": Wd.S_ARCH, "soffit": Wd.S_SOFFIT, "jambL": Wd.S_JAMBL,
            "jambR": Wd.S_JAMBR}
BAKE_ALL = {"conch": 0.7, "hemi": 0.5, "arch": 0.45, "soffit": 0.5, "jambL": 0.5, "jambR": 0.5}
FULLBAKE = (201.25, 204.5)       # the wide segment: every surface from its baked texture


def _baked_frame(fc, st, cam, T, full=False):
    """baked surfaces through the camera: one full-resolution ray cast, trilinear mip sampling per surface.
    full=False: the nave only; True: every surface (the wide segment)"""
    sid, u, v, fp = Wd.cast(cam, fc.w, fc.h, fc.s)
    img = np.zeros((fc.h, fc.w, 3), np.float32)
    lights, _ = dawn(T)
    ref, _ = dawn(T_GLOR - 0.3)
    pw = lights[0].power / ref[0].power
    flash = 0.0
    if T >= T_GLOR:
        flash = 0.5 * math.exp(-(T - T_GLOR) / 0.16)
    srcs = dict(BAKE_SID)
    if full:
        srcs.update(LIVE_SID)
    for name, sv in srcs.items():
        m = sid == sv
        if not m.any():
            continue
        tex = st["bake"][name]
        which = "B" if (T >= T_GLOR + 0.02 and "B" in tex) else "A"
        col = tex[which].sample(u[m], v[m], fp[m])[:, :3]
        k = expo(name, T) * pw * (1.0 + (flash if "B" in tex else 0.4 * flash))
        ax = tex["aux"].sample(u[m], v[m], fp[m])
        # the gold still twinkles: each baked gold tile on its own phase
        k = k * (1.0 + 0.38 * ax[:, 0] * np.sin(2 * math.pi * (ax[:, 1] * 3.0 + T * 0.55)))
        img[m] = col * k[:, None] if np.ndim(k) else col * k
    return img, sid


def _cull_bins(st, K):
    """per 0.5 s bin and per culled surface: the tiles the camera can see during that bin (+ a margin)"""
    out = {}
    nb = int(math.ceil((T_END - T0) / BIN))
    cs = 90.0
    for name in CULLED:
        b = st["built"][name]
        sv = b["sid"]
        ew, eh = b["ext"]
        gw, gh = int(ew / cs) + 1, int(eh / cs) + 1
        ci = np.clip((b["lay"].xy[:, 0] / cs).astype(int), 0, gw - 1)
        cj = np.clip((b["lay"].xy[:, 1] / cs).astype(int), 0, gh - 1)
        bins = []
        for k in range(nb):
            grid = np.zeros((gh, gw), np.uint8)
            for T in np.linspace(T0 + k * BIN, T0 + (k + 1) * BIN, 4):
                cam = camera(min(T, T_END - 1e-3), K)
                sid, u, v, _ = Wd.cast(cam, 192, 108, 0.1)
                m = sid == sv
                if m.any():
                    gi = np.clip((u[m] / cs).astype(int), 0, gw - 1)
                    gj = np.clip((v[m] / cs).astype(int), 0, gh - 1)
                    grid[gj, gi] = 1
            if not grid.any():
                bins.append(np.zeros(0, np.int64))
                continue
            grid = cv2.dilate(grid, np.ones((5, 5), np.uint8))
            keep = grid[cj, ci] > 0
            bins.append(None if keep.mean() > 0.85 else np.nonzero(keep)[0])
        out[name] = bins
    return out


_SUB = {}


def _subset(st, name, T):
    """(layout, idx) for this frame: the pre-culled tiles of the current time bin (cached per process)"""
    b = st["built"][name]
    bins = st["cull"].get(name)
    if bins is None:
        return b["lay"], None
    k = min(len(bins) - 1, max(0, int((T - T0) / BIN)))
    idx = bins[k]
    if idx is None:
        return b["lay"], None
    key = (name, k)
    lay = _SUB.get(key)
    if lay is None:
        if len(_SUB) > 8:
            _SUB.clear()
        lay = b["lay"].subset(idx)
        _SUB[key] = lay
    return lay, idx


# ============================================================ setup
def setup(S):
    n = S.n
    built = {}
    for name, cfg in _surfaces().items():
        b = _build_surface(name, cfg, S)
        b.update(surface=cfg["surface"], sid=cfg["sid"])
        built[name] = b
    # the donor pair: their own fine tiles, re-coloured per frame from a repainted chart window
    pair = built["conch"]["pair"]
    # twinkles: a few gold ground tiles above the Master's head
    cl = built["conch"]
    gold_i = np.nonzero(cl["lay"].mat == mz.GOLD)[0]
    hx, hy = P.CX, P.CY - 1.25 * P.CH
    dd = np.hypot(cl["lay"].xy[gold_i, 0] - hx, cl["lay"].xy[gold_i, 1] - hy)
    tw = gold_i[np.argsort(dd)[[3, 17, 41, 77, 120]]]
    # flags
    fpath = S.cache / f"flags_{VERSION}_{n}.npz"
    if fpath.exists():
        z = np.load(fpath, allow_pickle=True)
        sticky = FlagTrack(z["sv"], z["sp"], z["se"], z["ss"], eval(str(z["sm"][0])), z["sf"])
        info = dict(glued=z["glued"], touch=z["touch"])
        central = FlagTrack(z["cv"], z["cp"], z["ce"], z["cs"], eval(str(z["cm"][0])), z["cf"])
    else:
        sticky, info = P.simulate(n)
        central = central_flag(n)
        np.savez_compressed(fpath, sv=sticky.verts, sp=sticky.poles, se=sticky.energy, ss=sticky.snap,
                            sm=np.array([repr(sticky.meta)]), sf=sticky.flap_hz, glued=info["glued"],
                            touch=info["touch"], cv=central.verts, cp=central.poles, ce=central.energy,
                            cs=central.snap, cm=np.array([repr(central.meta)]), cf=central.flap_hz)
    st = dict(built=built, pair=pair, twinkle=tw, sticky=sticky, sticky_info=info, central=central,
              cflags=congregation_flags(built), K=camera_keys(), expl=explicit_band(), donor=donor_band())
    st["bake"] = {name: _bake(S, name, built[name], s_tex) for name, s_tex in BAKED.items()}
    # the wide segment: the apse and the arch baked too (the pair as they are then; the bands still unset)
    for name, s_tex in BAKE_ALL.items():
        b = built[name]
        bands = ()
        if name == "conch":
            sets = {}
            for w_, tb in (("A", FULLBAKE[0] + 0.5), ("B", T_GLOR + 0.7)):
                rgb = b["rgbA"].copy()
                mat = b["mat"].copy()
                prgb, _ = _pair_rgb(st, tb)
                rgb[pair["idx"]] = prgb
                mat[pair["idx"]] = np.where(A.is_gold(prgb), mz.GOLD, 0).astype(np.uint8)
                if w_ == "B" and "rgbB" in b:
                    rgb[b["flip"]] = b["rgbB"][b["flip"]]
                    mat[b["flip"]] = b["matB"][b["flip"]]
                sets[w_] = (rgb, mat)
            bands = (st["expl"],)
        else:
            sets = None
            if name == "hemi":
                bands = (st["donor"],)
        st["bake"][name] = _bake(S, name, b, s_tex, sets=sets, bands=bands)
    st["cull"] = _cull_bins(st, st["K"])
    _export_foley(S, st)
    return st


# ============================================================ render helpers
def _pair_rgb(st, T, part="all"):
    """per-frame colours of the conch tiles under the donor pair (they move by their tiles re-colouring)"""
    pr = st["pair"]
    x0, y0, x1, y1 = pr["rect"]
    sc = pr["sc"]
    img = pr["bg"].copy() if part == "all" else np.zeros_like(pr["bg"])
    surf = mz_surface_from(img)
    ctx = cairo.Context(surf)
    ctx.scale(sc, sc)
    ctx.translate(-x0, -y0)
    P.draw(ctx, T, layer="color", part=part)
    surf.flush()
    h, w = img.shape[:2]
    b = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
    out = b[..., [2, 1, 0, 3]].astype(np.float32) / 255.0
    rgb = mz.sample(out, pr["lay"], crect=(x0, y0, x1, y1), mode="medoid")
    if part == "all":
        return rgb, None
    a = mz.sample_mask(out[..., 3], pr["lay"], crect=(x0, y0, x1, y1))
    return rgb, a


T_FISTS = T_PULL + 1.6        # after this the pair is too small for the fists pass


def _fists(fc, st, T, cam, lights, amb, img):
    """Lutie's fists (vx.icon part='front') as wall tesserae over the pole she grips"""
    pr = st["pair"]
    rgb, a = _pair_rgb(st, T, part="front")
    idx = np.nonzero(a >= 0.5)[0]
    if len(idx) == 0:
        return img
    return mz.render(fc, None, pr["lay"].subset(idx), rgb=rgb[idx], light=lights, ambient=amb, view=cam.mosaic(),
                     surface=st["built"]["conch"]["surface"], bg=img, t=T, exposure=expo("conch", T))


def mz_surface_from(img):
    from vx.canvas import surface_from_array
    return surface_from_array(img)


def _visible(cam):
    """surface ids seen by the camera (a coarse ray cast; the thin intrados rides along with the arch)"""
    sid, _, _, _ = Wd.cast(cam, 192, 108, 0.1)
    v = set(np.unique(sid).tolist())
    if Wd.S_ARCH in v or Wd.S_HEMI in v:
        v |= {Wd.S_SOFFIT, Wd.S_JAMBL, Wd.S_JAMBR}
    return v


def _surface_render(fc, st, name, cam, T, lights, amb, bg=None):
    b = st["built"][name]
    lay = b["lay"]
    rgb = b["rgbA"]
    mat = b["mat"]
    gain = None
    if name == "conch":
        rgb = rgb.copy()
        mat = mat.copy()
        prgb, _ = _pair_rgb(st, T)
        pi = st["pair"]["idx"]
        rgb[pi] = prgb
        # the pair moves over its tiles: gold leaf wherever this frame's cartoon is gold (ground, nimbus edges)
        mat[pi] = np.where(A.is_gold(prgb), mz.GOLD, 0).astype(np.uint8)
    if "rgbB" in b and T >= T_GLOR - 0.05:
        rng_t = (b["world"][:, 0] * 0.00013 + b["world"][:, 1] * 0.00007) % 0.06
        t_flip = T_GLOR - 0.02 + rng_t[b["flip"]]
        done = T >= t_flip
        if done.any():
            rgb = rgb.copy()
            mat = mat.copy()
            fi = b["flip"][done]
            rgb[fi] = b["rgbB"][fi]
            mat[fi] = b["matB"][fi]
        # a flash as they set
        fl = np.exp(-np.clip(T - t_flip, 0, None) / 0.12) * (T >= t_flip)
        gain = np.ones(len(lay), np.float32)
        gain[b["flip"]] += 1.4 * fl
    if name == "conch" and T_TWINKLE - 0.1 < T < T_TWINKLE + 1.4:
        gain = np.ones(len(lay), np.float32) if gain is None else gain
        for k, ti in enumerate(st["twinkle"]):
            tk = T_TWINKLE + 0.2 * k
            gain[ti] += 5.0 * math.exp(-((T - tk) / 0.07) ** 2)
    lo = lights_out(b["world"], T, seed=b["sid"])
    if lo is not None:
        gain = lo if gain is None else gain * lo
    emit = None
    if b["glass"].max() > 0.5:
        g = (b["glass"] >= 0.5).astype(np.float32)
        k = (0.32 + 0.22 * smoothstep(T0, T_GLOR, T)) * g * (1.0 if lo is None else lo)
        emit = mz.srgb_to_lin(rgb) * k[:, None]
    sub, idx = _subset(st, name, T)
    if idx is not None:
        if len(idx) == 0:
            return bg
        rgb, mat = rgb[idx], mat[idx]
        gain = None if gain is None else gain[idx]
        emit = None if emit is None else emit[idx]
    return mz.render(fc, None, sub, rgb=rgb, mat=mat, light=lights, ambient=amb, view=cam.mosaic(),
                     surface=b["surface"], gain=gain, emit=emit, bg=bg, t=T, exposure=expo(name, T))


def _band_render(fc, st, band, surface, cam, T, lights, amb, bg=None):
    """a titulus band (tiles setting themselves) on a surface"""
    t_set = band["t_set"]
    vis, scale, flash = Ti.pop(T, t_set)
    letter = band["letter"]
    arriving = vis & letter
    # before its moment a letter's tiles are laid in the ground's colour (its courses show only as a ghost in the
    # andamento); then each is pressed in, large and bright, in the ink
    unset = letter & (T < t_set)
    rgb = band["rgb"].copy()
    gold = band["gold"].copy()
    rgb[unset] = band["ghost"]
    gold[unset] = 0.0
    poly = band["poly"].copy()
    if arriving.any():
        c = band["xy"][arriving][:, None, :]
        poly[arriving] = c + (poly[arriving] - c) * scale[arriving][:, None, None]
    gain = np.ones(len(t_set), np.float32) + 2.2 * flash * arriving - 0.3 * unset
    Pb = surface.embed(band["xy"].astype(np.float64))
    if band.get("last"):
        # the EXPLICIT band keeps the last light: it closes in from both ends onto the middle of the word
        lo = lights_out(Pb, T, seed=7, t0=T_LAST - 0.6, speed=4000.0, origin=Pb.mean(0), inward=True)
    else:
        lo = lights_out(Pb, T, seed=7)
    if lo is not None:
        gain = gain * lo
    lift = np.where(arriving, 6.0 * (scale - 1.0) / 0.55, 0.0).astype(np.float32)
    tiles = mz.Tiles(band["xy"], poly=poly, nv=4, rgb=rgb, gold=gold, gain=gain,
                     lift=lift if arriving.any() else None, seed=3)
    return mz.render_tiles(fc, tiles, view=cam.mosaic(), surface=surface, light=lights, ambient=amb,
                           bg=bg, t=T, exposure=house(T))


# ============================================================ render
def render(fc, st):
    T = fc.T
    cam = camera(T, st["K"])
    lights, amb = dawn(T)
    cl = candle_light(T, cam)
    if cl is not None:
        lights = lights + [cl]
    vis = _visible(cam)
    B = st["built"]
    full = FULLBAKE[0] <= T < FULLBAKE[1]
    if full or vis & set(BAKE_SID.values()):
        img, _ = _baked_frame(fc, st, cam, T, full=full)
    else:
        img = np.zeros((fc.h, fc.w, 3), np.float32)
    for name in ([] if full else LIVE):
        if B[name]["sid"] not in vis:
            continue
        img = _surface_render(fc, st, name, cam, T, lights, amb, bg=img)
        if name == "conch":
            img = _band_render(fc, st, st["expl"], B["conch"]["surface"], cam, T, lights, amb, bg=img)
        if name == "hemi":
            img = _band_render(fc, st, st["donor"], B["hemi"]["surface"], cam, T, lights, amb, bg=img)
    img = _god_rays(fc, img, cam, T, st)
    if T_TWINKLE - 0.1 < T < T_TWINKLE + 1.3:
        img = _twinkles(fc, img, cam, T, st)
    # the pilgrim's candle and the snuffer (drawn under the Flags: nothing is ever drawn over cloth)
    if T > T_CARD - 0.8:
        img = _candle(fc, img, T)
    # ---------------------------------------------------------------- Flags LAST (never tessellated)
    i = fc.f
    Sc = B["conch"]["surface"]
    sa = 1.0 - smoothstep(*T_STICKY_OUT, T)          # the sticky Flag's scale as it finally goes
    SL = (-0.4, -0.7, 0.6)
    if sa > 0 and T < T_FISTS:
        # the sticky Flag's pole, then Lutie's fists closing over it (tesserae), then all the cloth
        cv = Canvas(fc)
        draw_track(cv.ctx, st["sticky"], i, Sc, cam, z0=18.0, light=SL, cloth=False, scale=sa)
        img = cv.over(img)
        img = _fists(fc, st, T, cam, lights, amb, img)
    cv = Canvas(fc)
    ctx = cv.ctx
    fa = lowered(T)
    draw_track(ctx, st["central"], i, Sc, cam, z0=30.0, light=(-0.3, -0.75, 0.55), scale=fa)
    draw_congregation_flags(ctx, st["cflags"], cam, T)
    if sa > 0:
        # alone in the dark it keeps its own soft light
        gl = 0.7 * smoothstep(T_FLAGS_OUT[0], T_FLAGS_OUT[1], T)
        draw_track(ctx, st["sticky"], i, Sc, cam, z0=18.0, light=SL, pole=T >= T_FISTS, glow=gl, scale=sa)
    img = cv.over(img)
    # ---------------------------------------------------------------- the white-gold hand-off from m09
    if T < 188.45:
        img = _dissolve(fc, img, T, cam)
    return np.clip(img, 0, 1)


def _twinkles(fc, img, cam, T, st):
    """the deadpan's punctuation: single gold tesserae above the Master catch the light, one after another"""
    Sc = st["built"]["conch"]["surface"]
    lay = st["built"]["conch"]["lay"]
    cv = Canvas(fc)
    ctx = cv.ctx
    for k, ti in enumerate(st["twinkle"]):
        a = math.exp(-((T - (T_TWINKLE + 0.2 * k)) / 0.085) ** 2)
        if a < 0.02:
            continue
        P3 = Sc.embed(lay.xy[ti:ti + 1].astype(np.float64))
        sp, dep = cam.project(P3)
        if dep[0] < 1:
            continue
        x, y = float(sp[0, 0]), float(sp[0, 1])
        r = 34.0 * a
        for ang, ln in ((0.0, 1.0), (math.pi / 2, 1.0), (math.pi / 4, 0.45), (-math.pi / 4, 0.45)):
            dx, dy = math.cos(ang) * r * ln, math.sin(ang) * r * ln
            g = cairo.LinearGradient(x - dx, y - dy, x + dx, y + dy)
            g.add_color_stop_rgba(0.0, 1.0, 0.95, 0.8, 0.0)
            g.add_color_stop_rgba(0.5, 1.0, 0.97, 0.88, 0.95 * a)
            g.add_color_stop_rgba(1.0, 1.0, 0.95, 0.8, 0.0)
            ctx.set_source(g)
            ctx.set_line_width(2.2)
            ctx.move_to(x - dx, y - dy)
            ctx.line_to(x + dx, y + dy)
            ctx.stroke()
        rg = cairo.RadialGradient(x, y, 0, x, y, r * 0.35)
        rg.add_color_stop_rgba(0.0, 1.0, 0.98, 0.9, 0.9 * a)
        rg.add_color_stop_rgba(1.0, 1.0, 0.9, 0.7, 0.0)
        ctx.set_source(rg)
        ctx.arc(x, y, r * 0.35, 0, 2 * math.pi)
        ctx.fill()
    return img + cv.rgba()[..., :3]


def _god_rays(fc, img, cam, T, st):
    """dawn shafts from the south clerestory windows across the nave (additive, soft)"""
    amt = smoothstep(200.1, 201.8, T) * (1.0 - smoothstep(T_SNUFF - 0.2, T_SNUFF + 0.9, T))
    if T > T_GLOR - 0.05:
        amt *= 1.0 + 0.9 * math.exp(-max(0.0, T - T_GLOR) / 1.1)
    if amt <= 0.01:
        return img
    lights, _ = dawn(T)
    d = np.asarray(lights[0].p, float)
    d = d / np.linalg.norm(d)
    trav = -d
    Sr = st["built"]["right"]["surface"]
    cv = Canvas(s=fc.s * 0.25)
    ctx = cv.ctx
    for k, u in enumerate(A.CLERE_WIN):
        q = np.array([[u - 200, 880], [u + 200, 880], [u + 200, 2180], [u - 200, 2180]], float)
        Pw = Sr.embed(q)
        Lb = (Wd.FLOOR_Y - Pw[:, 1]) / max(trav[1], 1e-3)
        Pf = Pw + trav[None, :] * Lb[:, None] * 0.92
        a, da = cam.project(Pw)
        b, db = cam.project(Pf)
        if (da < 30).any() or (db < 30).any():
            continue
        pts = np.vstack([a, b]).astype(np.float32)
        hull = cv2.convexHull(pts)[:, 0, :]
        ca, cb = a.mean(0), b.mean(0)
        g = cairo.LinearGradient(ca[0], ca[1], cb[0], cb[1])
        wv = 0.8 + 0.2 * math.sin(k * 2.3 + T * 0.4)
        g.add_color_stop_rgba(0.0, 1.0, 0.86, 0.6, 0.5 * wv)
        g.add_color_stop_rgba(0.35, 1.0, 0.82, 0.55, 0.26 * wv)
        g.add_color_stop_rgba(1.0, 1.0, 0.8, 0.5, 0.0)
        ctx.new_path()
        ctx.move_to(*hull[0])
        for p in hull[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.set_source(g)
        ctx.fill()
    small = cv2.GaussianBlur(cv.rgba()[..., :3], (0, 0), 2.5 * fc.s)
    beams = cv2.resize(small, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)
    return img + beams * (0.42 * amt)


def _dissolve(fc, img, T, cam):
    """the white-gold flash condenses into the rising sun: the glow lingers longest where the sun is"""
    su = A.chart_of(0.0, A.SUN_H)
    sp, _ = cam.project(np.array(Wd.conch_to_world(su[0], su[1]), float))
    yy, xx = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
    d = np.hypot(xx / fc.s - sp[0], yy / fc.s - sp[1]) / 2200.0
    p = (T - T0) / 0.9
    a = np.clip(1.0 - (p * 1.35 - 0.35 * (1.0 - np.clip(d, 0, 1))), 0, 1) ** 1.6
    g = np.array(PC["flag_glow"], np.float32)
    return img * (1 - a[..., None]) + g * a[..., None]


CANDLE = dict(x=W * 0.5, y=H * 0.845, h=86.0)


def _candle(fc, img, T):
    """end card: the pilgrim lifts the candle into view, a brass snuffer comes down on it at T_SNUFF"""
    rise = ease_in_out(clamp((T - (T_CARD - 0.8)) / 1.2), 2)
    y = CANDLE["y"] + (1.0 - rise) * 260.0
    lit, ember = Cn.snuff(T - T_SNUFF)
    # the flame's warm pool on the lower frame
    if lit > 0:
        yy, xx = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
        r = np.hypot(xx / fc.s - CANDLE["x"], (yy / fc.s - y) * 1.3) / 520.0
        pool = np.exp(-r * r * 2.2) * 0.1 * lit * Cn.flame_light(T) * rise
        img = img * (1 + pool[..., None] * np.array([1.2, 0.95, 0.6], np.float32)) + \
            pool[..., None] * np.array([0.35, 0.22, 0.08], np.float32)
    img = Cn.candle(img, fc.s, CANDLE["x"], y, CANDLE["h"], T, lit=lit, ember=ember,
                    light=0.9 * max(lit, 0.2 * ember) * house(T))
    if T > T_SNUFF:
        sm = Cn.smoke(img.copy(), fc.s, CANDLE["x"], y, CANDLE["h"], T - T_SNUFF, seed=4)
        img = img + (sm - img) * house(T)
    img = _snuffer(fc, img, T, y)
    return img


def _snuffer(fc, img, T, y_wick):
    """a bronze candle extinguisher: a bell hanging from the end of an acolyte's long pole, raised from below the
    frame (lower right), lowered over the flame at T_SNUFF, lifted away"""
    t_in, t_hov, t_up, t_out = T_SNUFF - 0.75, T_SNUFF - 0.22, T_SNUFF + 0.4, T_SNUFF + 1.15
    if not (t_in < T < t_out):
        return img
    fx, top = CANDLE["x"], y_wick - CANDLE["h"] * 1.05          # over the flame: the bell's mouth at the wick
    p_off, p_hov, p_on = (fx + 620.0, y_wick + 420.0), (fx + 40.0, top - 70.0), (fx, y_wick + 6.0)
    if T < t_hov:
        u = ease_in_out((T - t_in) / (t_hov - t_in), 2)
        bx, by = p_off[0] + (p_hov[0] - p_off[0]) * u, p_off[1] + (p_hov[1] - p_off[1]) * u
    elif T < T_SNUFF:
        u = ease_in_out((T - t_hov) / (T_SNUFF - t_hov), 2)
        bx, by = p_hov[0] + (p_on[0] - p_hov[0]) * u, p_hov[1] + (p_on[1] - p_hov[1]) * u
    elif T < t_up:
        bx, by = p_on
    else:
        u = ease_in_out((T - t_up) / (t_out - t_up), 2)
        bx, by = p_on[0] + (p_off[0] - p_on[0]) * u, p_on[1] + (p_off[1] - p_on[1]) * u
    cv = Canvas(fc)
    ctx = cv.ctx
    stops = ((0.0, (0.26, 0.17, 0.08)), (0.4, (0.62, 0.45, 0.24)), (0.55, (0.78, 0.6, 0.36)), (1.0, (0.22, 0.14, 0.06)))
    # the pole comes up from the lower right to a hook above the bell
    hx, hy = bx + 22.0, by - 88.0
    g = cairo.LinearGradient(hx - 6, hy, hx + 6, hy)
    for k, c in stops:
        g.add_color_stop_rgb(k, *c)
    ctx.set_source(g)
    ctx.set_line_width(10)
    ctx.move_to(hx + 700.0, hy + 1300.0)
    ctx.line_to(hx, hy)
    ctx.stroke()
    ctx.set_line_width(4)
    ctx.move_to(hx, hy)
    ctx.curve_to(hx - 6, hy - 18, bx - 16, hy - 14, bx, by - 70)
    ctx.stroke()
    # the bell: a cone, mouth down, with a rolled lip
    g2 = cairo.LinearGradient(bx - 34, by, bx + 34, by)
    for k, c in stops:
        g2.add_color_stop_rgb(k, *c)
    ctx.move_to(bx - 9, by - 70)
    ctx.curve_to(bx - 13, by - 42, bx - 28, by - 14, bx - 34, by)
    ctx.line_to(bx + 34, by)
    ctx.curve_to(bx + 28, by - 14, bx + 13, by - 42, bx + 9, by - 70)
    ctx.close_path()
    ctx.set_source(g2)
    ctx.fill()
    ctx.set_source_rgb(0.72, 0.56, 0.34)
    ctx.set_line_width(4)
    ctx.move_to(bx - 34, by)
    ctx.line_to(bx + 34, by)
    ctx.stroke()
    return cv.over(img)


def post(fc, st):
    T = fc.T
    p = {}
    if T < 188.5:
        p["vignette"] = 0.3 * smoothstep(T0, 188.5, T)
    if T > T_GLOR - 0.05:
        p["bloom"] = 0.34 + 0.2 * math.exp(-max(0.0, T - T_GLOR) / 0.8)
    if T > T_STICKY_OUT[1]:
        p["fade"] = smoothstep(T_STICKY_OUT[1], T_STICKY_OUT[1] + 0.05, T)       # after the last Flag has gone
    return p


# ============================================================ foley
def _export_foley(S, st):
    tr = st["sticky"]
    tr.export_foley(S, "flag")
    st["central"].export_foley(S, "flag_central", gain=0.6)
    info = st["sticky_info"]
    ev = []
    ev.append(dict(t=P.T_MASTER, kind="offer", strength=0.6, pan=-0.2, desc="Lutie offers the small Flag"))
    touch = np.nonzero(info["touch"] > 0)[0]
    if len(touch):
        t1 = T0 + touch[0] / S.fps
        ev.append(dict(t=t1, kind="sticky", strength=0.8, pan=-0.05, desc="touch: the cloth meets his sleeve"))
        ev.append(dict(t=t1 + 0.05, kind="cling", strength=0.8, pan=-0.05,
                       desc="cling: the cloth glues itself to Crocus's sleeve (it never lets go)"))
    ev.append(dict(t=P.T_ITIS + 0.1, kind="sticky", strength=0.6, pan=0.0,
                   desc="stretch: he lifts his arm, the cloth comes with it"))
    for k in range(3):
        ev.append(dict(t=P.T_STICKY1[0] + 0.12 + k * 0.29, kind="sticky", strength=0.45, pan=0.0,
                       desc="shake: the cloth jiggles on the sleeve, still stuck"))
    ev.append(dict(t=P.T_STICKY1[1] + 0.1, kind="cling", strength=0.6, pan=-0.2,
                   desc="the pole will not leave Lutie's hand: she lets go and shakes it (to %.2f)" % (P.T_BUT)))
    for k in range(5):
        ev.append(dict(t=T_TWINKLE + 0.2 * k, kind="twinkle", strength=0.5 - 0.06 * k, pan=0.1 + 0.05 * k,
                       desc="deadpan: a gold tessera twinkles"))
    ev.append(dict(t=T_PULL, kind="whoosh", strength=0.5, pan=0.0,
                   desc="the camera pulls back through the arch into the nave (to %.1f)" % T_GLOR))
    ev.append(dict(t=T_GLOR, kind="raise", strength=1.0, pan=0.0,
                   desc="Glorious: every figure raises a Flag at once (%d real Flags pop, tiles flip; to %.1f)"
                   % (len(st["cflags"]["base"]), T_GLOR + 0.6)))
    for k in range(10):
        ev.append(dict(t=T_GLOR + 0.02 + k * 0.05, kind="flag_whoosh", strength=0.9 - 0.07 * k,
                       pan=float(np.clip(math.sin(k * 2.1), -1, 1)) * 0.8, desc="crowd of Flags springing open"))
    for band, nm, pan0 in ((st["expl"], "EXPLICIT", 0.0), (st["donor"], "donor inscription", 0.0)):
        ts = band["t_set"][band["letter"]]
        xs = band["xy"][band["letter"], 0]
        order = np.argsort(ts)
        for j in order[:: max(1, len(order) // 90)]:
            ev.append(dict(t=float(ts[j]), kind="tile_set", strength=0.35, pan=float(np.clip((xs[j] - AX) / R, -1, 1)) * 0.6,
                           desc=f"{nm}: a letter tessera sets itself"))
    ev.append(dict(t=T_SNUFF - 0.55, kind="whoosh", strength=0.2, pan=0.4, desc="the brass snuffer comes down"))
    ev.append(dict(t=T_SNUFF, kind="snuff", strength=0.9, pan=0.0, desc="the candle is snuffed"))
    ev.append(dict(t=T_WAVE, kind="lights_out", strength=0.5, pan=0.0,
                   desc="the tesserae go dark tile by tile from the band outward (to %.2f)" % (T_WAVE + 1.35)))
    ev.append(dict(t=T_STICKY_OUT[0], kind="sticky", strength=0.3, pan=-0.2,
                   desc="the sticky Flag lingers alone after the others are lowered, then pops out (gone %.2f)"
                   % T_STICKY_OUT[1]))
    ev.append(dict(t=T_FLAGS_OUT[0], kind="lower", strength=0.5, pan=0.0,
                   desc="in the dark every Flag is lowered at once, except the sticky one (to %.2f)" % T_FLAGS_OUT[1]))
    ev.append(dict(t=T_END, kind="black", strength=0.0, pan=0.0, desc="black"))
    foley.export_events(S, "sfx", ev)

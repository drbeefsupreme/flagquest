"""m07 ruin - the Babel wall, its collapse and the bare wall it leaves. Owner: M07. m08 imports this READ-ONLY.

WORLD (vx.mosaic v1 convention): design units, X right, Y DOWN, Z toward the viewer.
    The wall is the plane z = 0; its chart (x, y) spans [0, WX] x [0, WY] = [0, 3000] x [0, 1500].
    The nave floor is the plane y = FLOOR = 1500 (the wall base), extending toward the camera (z > 0).
    Height above the floor = FLOOR - y.

CAMERA
    camera(cx, cy, zoom, he=700, shake=(0, 0)) -> mz.Camera: a frontal pinhole (no keystone), horizontal fov FOV=50,
        eye at height `he` above the floor, distance F/zoom from the wall; lens shift puts the wall point (cx, cy)
        at the screen centre and scales the wall by `zoom` (screen units per wall unit at z=0).
    CAM_END = dict(cx, cy, zoom, he): m07's locked camera 145.0-146.0 (floor line at screen y=880).
    For pitch/yaw moves build your own mz.Camera(eye, target, fov=FOV, at=...) - everything here is plain 3D.

THE RUIN AT 146.0 (load_ruin(cache) -> Ruin; heavy build once, cached in films/opus/cache/m07/)
    ruin.lay           the wall's mz.Layout (chart units; every tessera that fell)
    ruin.rgb (N,3)     sRGB colour of each tessera (screens and static already dead = dark glass)
    ruin.mat (N,)      mosaic material codes (GLASS / GOLD / STONE)
    ruin.rest_pos (N,3), ruin.rest_rot (N,3,3)   final pose in the heap (world, rot = rotation of the tile about its
                       centre relative to its wall pose, as mz.render(rot=)/Tiles(rot=) expect with its home poly)
    ruin.tiles(sel=None) -> mz.Tiles of the heap (world xy, z, rot, poly moved with the tile) for render_tiles
    ruin.fall          the m07_phys.Fall (full trajectories: ruin.fall.at(T) -> loose, pos, quat)
    ruin.backdrop      Backdrop: bare lime plaster + the red sinopia of the old programme + the stone floor
        .render(fc, cam, lights, ambient, extra=None) -> lit sRGB background (fc.h, fc.w, 3); extra(fc, cam, img_chart)
        is a hook to paint your own layer onto the plaster chart image (a new sinopia drawing itself) before warping.
    CANDLE_XZ, CANDLE_FLAME (world flame base), candle_light(T, power=1.0) -> mz.Light (flickering, swaying)
    candle_tiles(cache) -> mz.Tiles of the bronze stand + beeswax taper (it is set in tesserae too)
    draw_flame(ctx, cam, T, s, gust=0.0) -> paints the flame (cairo, design units) and returns its screen xy
    lights_silence(T) -> ([mz.Light], ambient) the rig of the silence (one candle, near-black)
    render_silence(fc, T, cam=None) -> the exact look of m07's last second (for checks / the m08 hand-off)
"""
import math
from pathlib import Path

import cairo
import cv2
import numpy as np

from vx import mosaic as mz
from vx.config import W, H, CACHE

from . import m07_art as A
from . import m07_phys as P

WX, WY, FLOOR, AX = A.WX, A.WY, A.FLOOR, A.AX
FOV = 50.0
F = (W / 2) / math.tan(math.radians(FOV) / 2)
VERSION = "m07-ruin-v10"
CACHE_DIR = CACHE / "m07"

CAM_END = dict(cx=1874.0, cy=1160.0, zoom=1.0, he=420.0)

# the votive candle on its bronze stand, before the wall, right of the tower (world units)
CANDLE_XZ = (2097.0, 420.0)
STAND_H = 150.0
TAPER_H = 30.0
CANDLE_FLAME = (CANDLE_XZ[0], FLOOR - STAND_H - TAPER_H, CANDLE_XZ[1])


def camera(cx, cy, zoom, he=700.0, shake=(0.0, 0.0)):
    D = F / zoom
    ey = FLOOR - he
    ex = cx + shake[0]
    eye = (ex, ey + shake[1], D)
    tgt = (ex, ey + shake[1], 0.0)
    at = (W / 2 - zoom * shake[0] * 0.0, H / 2 - zoom * (cy - ey))
    return mz.Camera(eye, tgt, fov=FOV, at=at)


def cam_end():
    return camera(**CAM_END)


# ============================================================ light
def candle_light(T, power=1.0, gust=0.0):
    """the votive candle as an mz.Light: a small warm pool (fast falloff), flickering; gust bends + tears it"""
    x, y, z = CANDLE_FLAME
    return mz.candles(T, [(x + 7 * gust, y - 12, z)], color=(1.0, 0.68, 0.38), power=1.25 * power,
                      radius=125.0, flicker_amt=0.07 + 0.3 * abs(gust), sway=1.5 + 8 * abs(gust), seed=7)[0]


def lights_silence(T):
    return [candle_light(T, 1.0)], 0.012


# ============================================================ backdrop: plaster + sinopia + floor
class Backdrop:
    """The bare wall (lime plaster with the red sinopia) and the nave floor, lit per frame at low resolution."""
    K = 0.5                        # chart px per world unit of the sinopia image

    def __init__(self, cache_dir=CACHE_DIR):
        p = Path(cache_dir) / "backdrop_v3.npz"
        if p.exists():
            z = np.load(p)
            self.sin, self.floor = z["sin"], z["floor"]
        else:
            self.sin = A.paint_sinopia(self.K).astype(np.float32)
            self.floor = _paint_floor().astype(np.float32)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, sin=self.sin, floor=self.floor)
        self.sin_lin = mz.srgb_to_lin(self.sin)
        self.floor_lin = mz.srgb_to_lin(self.floor)

    def render(self, fc, cam, lights, ambient, extra=None, exposure=1.0):
        """lit sRGB background (fc.h, fc.w, 3): the wall above the floor line, the floor below it."""
        s = fc.s
        w, h = fc.w, fc.h
        wall_img = self.sin_lin
        if extra is not None:
            wall_img = extra(fc, cam, wall_img.copy())
        # chart (px of the sinopia image) -> screen px homography
        Hw = _plane_homography(cam, lambda u, v: np.stack([u, v, np.zeros_like(u)], -1), self.K, s)
        wall = cv2.warpPerspective(wall_img, Hw, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # floor texture: x in [FX0, FX0 + FW], z in [0, FZ] at FK px/unit
        Hf = _plane_homography(cam, lambda u, v: np.stack([u / FK + FX0, np.full_like(u, FLOOR), v / FK], -1), 1.0, s,
                               pre=True)
        flo = cv2.warpPerspective(self.floor_lin, Hf, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # which pixels see the floor: below the projected floor line
        yl = cam.project3(np.array([[AX, FLOOR, 0.0]]))[0][0, 1] * s
        yy = np.arange(h, dtype=np.float32)[:, None]
        m_floor = np.clip(yy - yl + 0.5, 0, 1)
        img = wall * (1 - m_floor[..., None]) + flo * (0.5 * m_floor[..., None])
        # lighting on a coarse grid of world points
        irr = _irradiance_grid(fc, cam, lights, ambient, yl)
        img = img * irr * exposure
        return mz.lin_to_srgb(np.clip(img, 0, None)).astype(np.float32)


FX0, FW, FZ, FK = -600.0, 4200.0, 2600.0, 0.25


def _paint_floor(seed=5):
    """the nave floor in (x, z): big worn stone slabs + a Cosmati guilloche band along the wall base"""
    rng = np.random.default_rng(seed)
    w, h = int(FW * FK), int(FZ * FK)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    c = cairo.Context(surf)
    c.scale(FK, FK)
    c.set_source_rgb(0.20, 0.18, 0.16)
    c.paint()
    zs = 150.0
    r = 0
    z = 90.0
    while z < FZ:
        hz = rng.uniform(160, 240)
        x = -FX0 * 0 + rng.uniform(-200, 0)
        while x < FW:
            wd = rng.uniform(220, 380)
            base = rng.uniform(0.2, 0.3)
            tint = np.array([1.0, 0.95, 0.86]) * base * rng.uniform(0.9, 1.08, 3)
            c.set_source_rgb(*tint)
            c.rectangle(x + 3, z + 3, wd - 6, hz - 6)
            c.fill()
            x += wd
        z += hz
        r += 1
    # Cosmati band along the wall: porphyry roundels in a marble guilloche
    c.set_source_rgb(0.42, 0.39, 0.35)
    c.rectangle(0, 8, FW, zs * 0.5)
    c.fill()
    for i in range(int(FW // 70) + 1):
        cx = i * 70 + 35
        c.set_source_rgb(0.38, 0.12, 0.13) if i % 2 else c.set_source_rgb(0.16, 0.30, 0.22)
        c.arc(cx, 8 + zs * 0.25, 22, 0, 2 * math.pi)
        c.fill()
    surf.flush()
    img = A._read_rgb(surf, w, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    from vx.ease import fbm2
    n = fbm2(xx / 30.0, yy / 30.0, seed=seed, octaves=4).astype(np.float32)
    img *= (1.0 + 0.12 * n)[..., None]
    return np.clip(img, 0, 1)


def _plane_homography(cam, embed, k, s, pre=False):
    """homography: plane image px -> screen px. embed(u, v) maps image px (pre=True) or chart units (/k) to world."""
    src = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], np.float64)
    if pre:
        Pw = embed(src[:, 0], src[:, 1])
    else:
        Pw = embed(src[:, 0] / k, src[:, 1] / k)
    scr, dep = cam.project3(Pw)
    return cv2.getPerspectiveTransform(src.astype(np.float32), (scr * s).astype(np.float32))


def _irradiance_grid(fc, cam, lights, ambient, yl, step=12):
    """per-pixel linear irradiance (h, w, 3) from point/directional lights on the wall (above the floor line) and
    the floor (below), computed on a coarse grid and upsampled."""
    s = fc.s
    gw, gh = fc.w // step + 2, fc.h // step + 2
    gx, gy = np.meshgrid(np.arange(gw) * step / s, np.arange(gh) * step / s)
    D = cam.ray(np.stack([gx, gy], -1))
    o = cam.eye
    # wall hit (z = 0) and floor hit (y = FLOOR)
    tw = -o[2] / np.minimum(D[..., 2], -1e-6)
    Pw = o + D * tw[..., None]
    tf = (FLOOR - o[1]) / np.maximum(D[..., 1], 1e-6)
    Pf = o + D * tf[..., None]
    on_floor = (Pw[..., 1] > FLOOR) & (D[..., 1] > 0)
    Pt = np.where(on_floor[..., None], Pf, Pw)
    Nn = np.where(on_floor[..., None], np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    amb = np.asarray(ambient, np.float32)
    acc = np.zeros(Pt.shape[:2] + (3,), np.float32) + (amb * mz.srgb_to_lin(np.array((1.0, 0.9, 0.78),
                                                                                      np.float32)) if amb.ndim == 0
                                                        else mz.srgb_to_lin(amb))
    for L in lights:
        if L.directional:
            ld = L.p / np.linalg.norm(L.p)
            lam = np.clip((Nn * ld).sum(-1), 0, 1)
            acc += (lam[..., None] * L.lin()).astype(np.float32)
        else:
            d = L.p - Pt
            dist = np.linalg.norm(d, axis=-1)
            lam = np.clip((Nn * d).sum(-1) / np.maximum(dist, 1e-6), 0, 1) * 0.7 + 0.3
            e = L.power / (1.0 + (dist / L.radius) ** 2)
            # the dark polished floor takes little diffuse light from the high, far lights (the tower, the
            # colossus); a near low flame (radius < 200: the candle) lights it fully
            if L.radius >= 200.0:
                e = np.where(on_floor, e * 0.28, e)
            acc += (lam * e)[..., None] * (mz.srgb_to_lin(L.color) * 1.0)
    return cv2.resize(acc, (fc.w + 2 * step, fc.h + 2 * step), interpolation=cv2.INTER_LINEAR)[:fc.h, :fc.w]


# ============================================================ the candle (stand + taper are tesserae)
_CANDLE = {}


def candle_tiles():
    """mz.Tiles (world) of the bronze stand + beeswax taper, standing on the floor at CANDLE_XZ"""
    if "t" in _CANDLE:
        return _CANDLE["t"]
    ext = (80, 190)
    k = 4.0
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(ext[0] * k), int(ext[1] * k))
    c = cairo.Context(surf)
    c.scale(k, k)
    bronze = A.rgb("#6B4A22")
    bronze_l = A.rgb("#A07A3A")
    wax = A.rgb("#E8D6A8")
    cx, base = 40.0, 188.0
    c.set_source_rgb(*bronze)
    c.move_to(cx - 30, base)                      # tripod foot
    c.line_to(cx - 8, base - 22)
    c.line_to(cx + 8, base - 22)
    c.line_to(cx + 30, base)
    c.close_path()
    c.fill()
    c.rectangle(cx - 4, base - STAND_H + 10, 8, STAND_H - 20)   # shaft
    c.fill()
    c.set_source_rgb(*bronze_l)
    for yk in (base - 40, base - 80, base - 112):                # knops
        c.save()
        c.translate(cx, yk)
        c.scale(9, 5)
        c.arc(0, 0, 1, 0, 2 * math.pi)
        c.restore()
        c.fill()
    c.set_source_rgb(*bronze)
    c.rectangle(cx - 17, base - STAND_H - 2, 34, 8)              # drip pan
    c.fill()
    c.set_source_rgb(*wax)
    c.rectangle(cx - 5, base - STAND_H - TAPER_H, 10, TAPER_H)   # the taper
    c.fill()
    c.set_source_rgb(*A.rgb("#F4E9CC"))
    c.rectangle(cx - 5, base - STAND_H - TAPER_H, 4, TAPER_H)
    c.fill()
    surf.flush()
    a = A._read_bgra(surf, surf.get_width(), surf.get_height()).astype(np.float32) / 255.0
    rgba = np.dstack([a[..., 2], a[..., 1], a[..., 0], a[..., 3]])
    alpha = rgba[..., 3]
    rgb = rgba[..., :3] / np.maximum(alpha[..., None], 1e-4)
    lay = mz.Layout.flow(rgb, tile=5.0, seed=21, extent=ext, alpha=alpha, fine=alpha > 0.5, fine_tile=3.6)
    col = mz.sample(rgb, lay)
    wx = CANDLE_XZ[0] + (lay.xy[:, 0] - cx)
    wy = FLOOR + (lay.xy[:, 1] - base)
    poly = lay.poly.copy()
    poly[..., 0] += CANDLE_XZ[0] - cx
    poly[..., 1] += FLOOR - base
    mat = np.where(np.abs(lay.xy[:, 1] - (base - STAND_H - TAPER_H / 2)) < TAPER_H / 2 + 1, mz.STONE, mz.GOLD)
    t = dict(xy=np.stack([wx, wy], 1), poly=poly, nv=lay.nv, size=lay.size, ang=lay.ang, rgb=col, mat=mat,
             tilt=lay.tilt, jit=lay.jit, rnd=lay.rnd, z=np.full(len(lay), CANDLE_XZ[1]))
    _CANDLE["t"] = t
    return t


def draw_flame(ctx, cam, T, s=1.0, gust=0.0, power=1.0, alpha=1.0):
    """a living candle flame at CANDLE_FLAME, drawn with cairo (design units); returns its screen (x, y) base and
    its on-screen scale. gust: -1..1 (the avalanche's wind bends and tears it)."""
    (sx, sy), = cam.project3(np.array([CANDLE_FLAME]))[0]
    dep = cam.project3(np.array([CANDLE_FLAME]))[1][0]
    k = cam.F / max(dep, 1.0)
    fl = mz.flicker(T, 7, 0.10 + 0.25 * abs(gust))
    hgt = 24.0 * k * (0.9 + 0.1 * fl) * power
    wid = 6.2 * k
    lean = 0.25 * math.sin(T * 1.7) * 0.2 + gust * 0.9 + 0.04 * math.sin(T * 9.1)
    ctx.save()
    ctx.translate(sx, sy)
    # halo
    g = cairo.RadialGradient(0, -hgt * 0.45, 0, 0, -hgt * 0.45, hgt * 3.2)
    g.add_color_stop_rgba(0, 1.0, 0.80, 0.50, 0.30 * alpha * power)
    g.add_color_stop_rgba(0.35, 1.0, 0.62, 0.30, 0.10 * alpha * power)
    g.add_color_stop_rgba(1, 1.0, 0.55, 0.25, 0.0)
    ctx.set_source(g)
    ctx.arc(0, -hgt * 0.45, hgt * 3.2, 0, 2 * math.pi)
    ctx.fill()
    # body: a teardrop leaning with the draught
    tipx = lean * hgt
    for (sc, col, a) in ((1.0, (1.0, 0.56, 0.18), 0.85), (0.72, (1.0, 0.80, 0.42), 0.95), (0.42, (1.0, 0.97, 0.86), 1.0)):
        ctx.new_path()
        ctx.move_to(0, 0)
        ctx.curve_to(-wid * sc * 1.25, -hgt * 0.10, -wid * sc * 0.9, -hgt * 0.55 * sc, tipx * sc, -hgt * sc)
        ctx.curve_to(wid * sc * 0.9, -hgt * 0.55 * sc, wid * sc * 1.25, -hgt * 0.10, 0, 0)
        ctx.close_path()
        ctx.set_source_rgba(*col, a * alpha)
        ctx.fill()
    # the blue root
    ctx.set_source_rgba(0.35, 0.45, 1.0, 0.55 * alpha)
    ctx.save()
    ctx.translate(0, -hgt * 0.06)
    ctx.scale(wid * 0.8, hgt * 0.1)
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()
    ctx.fill()
    # wick
    ctx.set_source_rgba(0.1, 0.07, 0.05, alpha)
    ctx.set_line_width(max(0.8, 0.9 * k))
    ctx.move_to(0, 2 * k)
    ctx.line_to(tipx * 0.08, -hgt * 0.16)
    ctx.stroke()
    ctx.restore()
    return (sx, sy), k


# ============================================================ the wall: build + collapse (cached)
class Ruin:
    pass


_RUIN = {}


def load_ruin(cache_dir=CACHE_DIR):
    """build (or load) the wall layout, its per-tile attributes and the whole collapse. Deterministic."""
    key = str(cache_dir)
    if key in _RUIN:
        return _RUIN[key]
    r = _build(Path(cache_dir))
    _RUIN[key] = r
    return r


def _build(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    k = 0.5
    p = A.paint_wall(k=k)
    cart = p.colour()
    emat_img, zone_img, obj_img = p.idmap()
    fine = p.fine()
    tfine = p.text_fine()
    bands = p.bands()
    gold = emat_img == A.M_GOLD
    ground = np.isin(zone_img, [A.Z_SKY, A.Z_ARCH_GROUND, A.Z_SPANDREL])
    # the programme in opus tessellatum/vermiculatum; the three inscriptions laid letter by letter in finer smalti
    la = mz.Layout.flow(cart, tile=12, seed=11, extent=(WX, WY), fine=fine & ~bands, fine_tile=5.0, gold=gold,
                        ground=ground, ground_style="echo", echo=2)
    lb = mz.Layout.flow(cart, tile=7.0, seed=12, extent=(WX, WY), fine=tfine, fine_tile=3.0, gold=gold,
                        alpha=bands.astype(np.float32))
    inb = lambda lay_: bands[np.clip((lay_.xy[:, 1] * k).astype(int), 0, bands.shape[0] - 1),
                             np.clip((lay_.xy[:, 0] * k).astype(int), 0, bands.shape[1] - 1)]
    lay = la.subset(np.nonzero(~inb(la))[0]) + lb.subset(np.nonzero(inb(lb))[0])
    n = len(lay)
    fn = cache_dir / f"wall_{VERSION}_{n}.npz"
    ix = np.clip((lay.xy[:, 0] * k).astype(int), 0, emat_img.shape[1] - 1)
    iy = np.clip((lay.xy[:, 1] * k).astype(int), 0, emat_img.shape[0] - 1)
    r = Ruin()
    r.lay = lay
    r.cart = cart
    r.k = k
    r.emat = emat_img[iy, ix].astype(np.int32)
    r.zone = zone_img[iy, ix].astype(np.int32)
    r.obj = obj_img[iy, ix].astype(np.int32)
    if fn.exists():
        z = np.load(fn)
        r.rgb = z["rgb"]
        arrays = {kk: z[kk] for kk in z.files if kk.startswith("f_")}
    else:
        r.rgb = mz.sample(cart, lay).astype(np.float32)
        arrays = None
    r.mat = lay.mat.copy()
    r.mat[np.isin(r.emat, [A.M_SCREEN, A.M_FEED])] = mz.SCREEN
    r.mat[r.emat == A.M_GOLD] = mz.GOLD
    r.fall = _collapse(r, arrays)
    if arrays is None:
        np.savez_compressed(fn, rgb=r.rgb, **{"f_" + a: getattr(r.fall, a) for a in _FALL_ARRAYS})
    # the dead colours: screens, static, billboards die to dark glass before they land
    r.rgb_dead = _dead_colours(r)
    r.backdrop = Backdrop(cache_dir)
    r.rest_pos = r.fall.pr
    r.rest_rot = P.q_matrix(P.q_mul(r.fall.qr, _q_conj(r.fall.q_home)))
    r.tiles = lambda sel=None: _heap_tiles(r, sel)
    return r


_FALL_ARRAYS = ("td", "tb", "pb", "qb", "vb", "tl", "pl", "vl", "pr", "ql", "qr", "hop", "nhop", "tr", "e",
                "spin_axis", "spin_rate", "chunk", "piv", "axis", "alpha", "tsh", "sag", "heightfield")


def _q_conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _dead_colours(r):
    dead = r.rgb.copy()
    em = np.isin(r.emat, [A.M_SCREEN, A.M_FEED, A.M_STATIC, A.M_BOARD, A.M_HALO, A.M_DIM])
    rng = np.random.default_rng(3)
    base = np.array(A.rgb("ink")) * 0.6 + np.array(A.rgb("slate")) * 0.4
    dead[em] = base * rng.uniform(0.6, 1.1, (em.sum(), 1)) + rng.normal(0, 0.012, (em.sum(), 3))
    return np.clip(dead, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- the collapse schedule
T_COLLAPSE = 141.8
T_UPPER = 142.2           # the conch (after the static has drained) starts to shed
T_LOWER = 142.45          # the lower field
T_BASE = 143.35           # ground band + dado
T_LAST = 144.86           # the last tessera must have landed + settled by here


def _collapse(r, arrays=None):
    lay = r.lay
    n = len(lay)
    rng = np.random.default_rng(1419)
    xy = lay.xy.astype(np.float64)
    zone = r.zone
    tw = A.tiers()
    # -------- chunks: the tower in blocks per tier; everything else in Voronoi sheets
    tower_top = np.isin(zone, [A.Z_TIER0 + 3, A.Z_TIER0 + 4, A.Z_TIER0 + 5, A.Z_TIER0 + 6, A.Z_CROWN, A.Z_CRANE])
    lad = zone == A.Z_LADDER
    tower_top |= lad & (r.obj >= 2)
    tower_low = np.isin(zone, [A.Z_TIER0, A.Z_TIER0 + 1, A.Z_TIER0 + 2]) | (lad & (r.obj < 2))
    from scipy.spatial import cKDTree
    seeds = np.stack([rng.uniform(-50, WX + 50, 900), rng.uniform(-30, WY + 30, 900)], 1)
    # relax the seeds a little (more even sheets)
    for _ in range(2):
        tree = cKDTree(seeds)
        _, lab = tree.query(xy)
        for j in range(len(seeds)):
            m = lab == j
            if m.any():
                seeds[j] = xy[m].mean(0)
    tree = cKDTree(seeds)
    _, chunk = tree.query(xy)
    chunk = chunk.astype(np.int64)
    C0 = len(seeds)
    # tower blocks: tier k, block by x (70 units)
    tk = np.full(n, -1)
    for kk in range(7):
        tk[zone == A.Z_TIER0 + kk] = kk
    tk[zone == A.Z_CROWN] = 7
    tk[zone == A.Z_CRANE] = 8
    tk[lad] = np.minimum(r.obj[lad] + 1, 6)
    blk = np.floor((xy[:, 0] - 900) / 70.0).astype(int)
    tw_m = tk >= 0
    tchunk = C0 + tk * 40 + np.clip(blk, 0, 39)
    chunk = np.where(tw_m, tchunk, chunk)
    # renumber
    uniq, chunk = np.unique(chunk, return_inverse=True)
    C = len(uniq)
    cen = np.zeros((C, 2))
    np.add.at(cen, chunk, xy)
    cnt = np.bincount(chunk, minlength=C).astype(float)
    cen /= cnt[:, None]
    ymin = np.full(C, 1e9)
    np.minimum.at(ymin, chunk, xy[:, 1])
    ymax = np.full(C, -1e9)
    np.maximum.at(ymax, chunk, xy[:, 1])
    # -------- per-chunk motion: sheets peel (top-hinged, the bottom swings out, sagging)
    piv = np.stack([cen[:, 0], ymin, np.zeros(C)], 1)
    axis = np.stack([np.ones(C), rng.normal(0, 0.25, C), rng.normal(0, 0.25, C)], 1)
    alpha = rng.uniform(2.0, 6.0, C)
    tsh = rng.uniform(0.10, 0.28, C)
    sag = np.full(C, 0.6)
    ctop = np.zeros(C, bool)
    ctop[np.unique(chunk[tower_top])] = True
    clow = np.zeros(C, bool)
    clow[np.unique(chunk[tower_low])] = True
    # the tower's top hinges over to the right about the crack above tier 2 (one rigid body, pieces let go)
    xc3, hw3, yb3, yt3 = tw[3]
    hinge = np.array([xc3 + hw3 + A.SIDE, yb3 + 4.0, 0.0])
    piv[ctop] = hinge
    axis[ctop] = [0.0, 0.0, 1.0]
    alpha[ctop] = 2.1
    sag[ctop] = 0.0
    # higher pieces let go first (they are thrown furthest)
    tsh[ctop] = 0.60 + 0.28 * np.clip((cen[ctop, 1] - A.CROWN[3]) / (yb3 - A.CROWN[3]), 0, 1) + rng.uniform(0, 0.05,
                                                                                                          ctop.sum())
    # the lower tiers crumble in place, sagging, toppling slightly outward
    piv[clow, 1] = ymax[clow]
    alpha[clow] = -rng.uniform(2.0, 6.0, clow.sum())
    tsh[clow] = rng.uniform(0.12, 0.3, clow.sum())
    sag[clow] = 0.75
    # -------- detach times
    td_c = np.zeros(C)
    ycen = cen[:, 1]
    xcen = cen[:, 0]
    upper = ycen < A.TITULUS_Y[0]
    td_c[upper] = T_UPPER + 0.48 * (ycen[upper] / A.TITULUS_Y[0]) ** 0.8 + rng.normal(0, 0.035, upper.sum())
    lower = ~upper
    f = np.clip((ycen - A.TITULUS_Y[0]) / (A.H(110.0) - A.TITULUS_Y[0]), 0, 1)
    td_c[lower] = T_LOWER + 0.72 * f[lower] + 0.22 * np.abs(xcen[lower] - 1800) / 1800 + rng.normal(0, 0.04,
                                                                                                       lower.sum())
    base = ycen > A.H(112.0)
    td_c[base] = T_BASE + 0.35 * rng.uniform(0, 1, base.sum()) + 0.1 * np.abs(xcen[base] - 1800) / 1800
    td_c[ctop] = T_COLLAPSE
    td_c[clow] = 142.35 + 0.12 * (ymax[clow] - A.H(385)) / 300 + rng.uniform(0, 0.08, clow.sum())
    td = td_c[chunk].copy()
    # precursors: tiles on sheet edges dribble a little before their sheet
    pre = rng.uniform(0, 1, n) < 0.06
    td[pre] -= rng.uniform(0.05, 0.3, pre.sum())
    td[pre & (td < T_COLLAPSE + 0.3)] = np.maximum(td[pre & (td < T_COLLAPSE + 0.3)], T_COLLAPSE + 0.3)
    td[tower_top] = T_COLLAPSE
    # stragglers: a few hang on and let go late (their fall must end before the silence)
    h = FLOOR - xy[:, 1]
    tfall = np.sqrt(2 * np.maximum(h, 1) / P.G)
    strag = (rng.uniform(0, 1, n) < 0.012) & ~tower_top
    late = np.minimum(td + rng.uniform(0.25, 0.9, n), T_LAST - 0.45 - tfall)
    td = np.where(strag, np.maximum(td, late), td)
    td = np.minimum(td, T_LAST - 0.45 - tfall)
    td = np.maximum(td, np.where(tower_top, T_COLLAPSE, T_COLLAPSE + 0.3))
    # -------- velocities at breakup + tumble
    v_kick = np.stack([rng.normal(0, 18, n), rng.normal(-6, 14, n), rng.uniform(4, 30, n)], 1)
    v_kick[tower_top, 2] += rng.uniform(10, 45, tower_top.sum())
    # a few are thrown hard and rattle far across the floor
    fly = rng.uniform(0, 1, n) < 0.07
    v_kick[fly, 2] += rng.uniform(60, 160, fly.sum())
    v_kick[fly, 0] += rng.normal(0, 60, fly.sum())
    spin_ax = rng.normal(0, 1, (n, 3))
    spin = spin_ax / np.linalg.norm(spin_ax, axis=1, keepdims=True) * rng.uniform(6, 26, n)[:, None]
    obst = np.array([[CANDLE_XZ[0], CANDLE_XZ[1], 18.0]])
    # the toppling top's pieces crash through the tiers below: they lose most of their swing
    damp = np.where(tower_top, 0.42, 1.0)
    if arrays is not None:
        f_ = P.Fall(xy, lay.ang, lay.size, td, chunk, piv, axis, alpha, tsh, v_kick, spin, FLOOR, seed=5,
                    skid=0.09, obst=obst, sag=sag, arrays=arrays, damp=damp)
    else:
        # the silence is sacred: every tessera must be at rest before T_LAST. Pull late ones earlier and rebuild.
        for it in range(4):
            f_ = P.Fall(xy, lay.ang, lay.size, td, chunk, piv, axis, alpha, tsh, v_kick, spin, FLOOR, seed=5,
                        skid=0.09, obst=obst, sag=sag, damp=damp)
            over = f_.tr - T_LAST
            if over.max() <= 0:
                break
            m = over > 0
            td[m] = np.maximum(td[m] - over[m] - 0.04, np.where(tower_top[m], T_COLLAPSE, T_COLLAPSE + 0.3))
        if (f_.tr > T_LAST).any():
            m = f_.tr > T_LAST                    # last resort: those few settle faster
            f_.tr[m] = np.maximum(f_.tl[m] + 0.03, T_LAST)
            f_.tl[m] = np.minimum(f_.tl[m], f_.tr[m] - 0.03)
    f_.tower_top = tower_top
    f_.tower_low = tower_low
    f_.hinge = hinge
    return f_


def _heap_tiles(r, sel=None):
    """mz.Tiles of the fallen tesserae at rest (world)."""
    f = r.fall
    idx = np.arange(f.n) if sel is None else np.asarray(sel)
    lay = r.lay
    pos = f.pr[idx]
    d = pos[:, :2] - lay.xy[idx]
    return mz.Tiles(pos[:, :2], ang=lay.ang[idx], size=lay.size[idx], rgb=r.rgb_dead[idx],
                    mat=np.where(r.mat[idx] == mz.SCREEN, mz.GLASS, r.mat[idx]), z=pos[:, 2],
                    rot=r.rest_rot[idx], poly=lay.poly[idx] + d[:, None, :].astype(np.float32), nv=lay.nv[idx],
                    tilt=lay.tilt[idx], jit=lay.jit[idx], rnd=lay.rnd[idx], ids=idx)

"""THE FLAG. Plain yellow cloth on an untreated wooden pole. NO markings, NO outlines, NO borders,
NO text, NO emblems, NO other colours on the cloth: only smooth yellow shading from light.
Every flag in the film is drawn through this module (the rule is enforced here: the rasterisers can
only output interpolations of the palette yellows flag_deep..flag_glow).

Real cloth physics (vx/flag_sim.py, numba): 3D particle grid pinned along the pole, small-step XPBD
(structural + shear + bend + long-range attachments), gravity, per-triangle aerodynamics from the
relative wind, convected turbulence + gusts, a travelling-wave flutter pressure, adaptive substeps for
fast poles (>2000 px/s), rigid carry-over on teleports/cuts, optional furl (roll around the pole).
Internally SI: every pole is a real 96" furring pole with a 36"x30" cloth, whatever its size on screen.
Rendering (vx/flag_raster.py): lighting from 3D normals on a Catmull-Rom-refined mesh (wrap diffuse,
soft sheen, backlit translucency, fold occlusion, same colour both sides), z-buffered folds (and pole),
supersampled AA silhouette, no outline. Pole = smooth light wood with soft cylindrical shading.
Perf (1 core): 700-frame hero sim ~2 s; track.draw ~2 ms (pole 300) / 4 ms (600) / 20 ms (2000, close-up);
draw_flag ~30 us (<12 px cloth), 0.15-0.9 ms (medium), draw_flag_field ~20-150 us per flag.

Public API
----------
Flag(width=None, height=None, pole=420, nx=22, ny=15, side=1, seed=0, pole_w=None,
     turbulence=1.0, flutter=1.0, stiffness=1.0)
    width/height default to the canonical 0.375*pole x 0.3125*pole (36"x30" on 96"); height is along the pole.
    side = +1 cloth starts to the right of the pole, -1 left (the wind decides afterwards).
    .simulate(n, pole_fn, wind_fn, fps=24, substeps=10, warmup=48, furl_fn=None) -> FlagTrack
        pole_fn(i) -> (x, y, ang)  pole BASE (bottom) in world/design coords, ang radians (0 = straight up,
                                    positive = leaning clockwise / to the right). Called for i in range(n).
        wind_fn(i) -> (wx, wy) or (wx, wy, wz)   mean wind in design px/s at frame i, calibrated on the
                                    canonical 420 px pole (breeze 300, strong 700, gale 1200) so the same
                                    number means the same weather at any flag size. wz (+ = toward camera)
                                    optional. Turbulence and gusts are added (deterministic per seed).
        furl_fn(i) -> 0..1         optional: 1 = cloth rolled tight around the top of the pole, 0 = free.
                                    Ramp 1 -> 0 over ~0.2-0.5 s and the wind really unrolls it.
        warmup: frames simulated (pole at pole_fn(0), wind_fn(0)) before frame 0 so frame 0 is settled.
        substeps: base substeps per frame (x3 internal XPBD small steps, more automatically for fast poles).
        warmup=0: frame 0 is the flat, fully extended rectangle at rest (perpendicular to the pole).
        Pole motion is interpolated smoothly across substeps; >0.6*pole jumps in one frame are treated as
        cuts (the cloth is carried rigidly). Plants with a jolt, waves, carries, raises: all fine.
FlagTrack(verts, poles, energy=None, snap=None, meta=None, flap_hz=None)   (constructible: you may build/concat)
    .n, .verts (n, ny, nx, 3) design px (z toward camera), .poles (n, 3)
    .energy (n,) 0..~1 flutter intensity (max over substeps of mean cloth speed rel. to the pole; 1 ~ gale)
    .snap (n,) 0..1 sparse snap peaks (trailing-edge whip / tension release), .flap_hz (n,) flap rate (Hz)
    .draw(ctx, i, alpha=1.0, glow=0.0, pole=True, cloth=True, light=(-0.45, -0.8, 0.4))
        i may be fractional (slow motion: interpolates frames). light = direction TO the light
        (x right, y down, z toward camera): z < 0 = backlit (translucent glow). glow > 0 = soft halo.
        Transform-safe: honours any ctx transform (camera, billboards, --scale).
    .pole_top(i) -> (x, y)
    .cloth_center(i) -> (x, y)
    .save(path) / FlagTrack.load(path)
    .export_foley(S, name, pan=None, f0=0, gain=1.0)  -> vx.foley.export_track with energy, pan
        (from the cloth's screen x if None), snap events, flap_hz and snap arrays. f0 = scene-local frame
        of track frame 0.
draw_flag(ctx, x, y, pole=420, t=0.0, wind=1.0, side=1, ang=0.0, seed=0, alpha=1.0,
          cloth_w=None, cloth_h=None, glow=0.0, pop=1.0, lean=0.0, light=None)
    crowd/background flag (x, y = pole base) from a bank of real simulated seamless loops
    (4 variants x 5 wind strengths), random phase per seed, LOD by device size (tiny: 7-point polygon,
    medium: batch kernel, large: full hero renderer).
    wind: 0 calm (hangs), 0.3 breeze, 0.6 fresh, 1.0 strong (flies out), 1.5 gale; negative = other side.
    pop 0..1: appear animation (pole grows, cloth springs open with overshoot). lean: amplitude (rad)
    of a gentle per-seed sway of the pole (held flags bobbing). light as in FlagTrack.draw.
draw_flag_field(ctx, xs, ys, pole, t, seeds=None, wind=1.0, side=1, ang=0.0, pop=None, alpha=1.0,
                light=None, glow=0.0, bright=None, sort=True, lean=0.0, cloth_w=None, cloth_h=None)
    thousands of flags in one call (numba, AA, back-to-front by ys). Every argument after ctx except
    light/sort may be a scalar or a per-instance array (incl. glow). bright: per-instance lighting lift
    (-1..1, e.g. sunlit rims). ~20-150 us/flag depending on size; each call is one layer paint.
draw_furled(ctx, x, y, pole=420, ang=0.0, seed=0, alpha=1.0, light=None, glow=0.0, cloth_w=None, cloth_h=None)
    a flag with its cloth rolled around the top of the pole (shoulders, marching, crowds).
draw_cloth(ctx, verts, alpha=1.0, glow=0.0, light=None, lift=0.0, pole_seg=None)
    shade + draw any cloth grid verts (ny, nx, 3) design px through the rule-enforcing renderer.
    pole_seg=(x0, y0, x1, y1, width) depth-sorts a pole segment with the cloth (cloth behind the pole hides).
draw_pole(ctx, x0, y0, x1, y1, width=7.0, alpha=1.0, light=None)
cloth_color(k) -> (r, g, b)   k in [-1, 2]: -1 deep, -0.5 shadow, 0 flag, 1 light, 2 glow. Only yellows.
"""
import fcntl
import math
import os

import cairo
import cv2
import numpy as np

from .config import C, CACHE
from . import flag_sim as _sim
from . import flag_raster as _ras

PAL5 = np.array([C["flag_deep"], C["flag_shadow"], C["flag"], C["flag_light"], C["flag_glow"]], np.float64)
FLAG_Y = np.array(C["flag"])
FLAG_S = np.array(C["flag_shadow"])
FLAG_D = np.array(C["flag_deep"])
FLAG_L = np.array(C["flag_light"])
FLAG_G = np.array(C["flag_glow"])
DEFAULT_LIGHT = (-0.45, -0.8, 0.4)
CLOTH_W, CLOTH_H = 0.375, 0.3125     # 36" x 30" on a 96" pole
POLE_W = 0.0167                      # 1.6" furring strip on 96"
_EMPTY2 = np.zeros((0, 0))
# cloth/air model (see vx/flag_sim.py). flutter = travelling-wave pressure gain, fcoef = flap rate f = fcoef*U/L,
# lam = wavelength in cloth widths, span = phase shift top->bottom (diagonal crests); shear/bend = compliances.
AERO = dict(cl=0.25, cd=1.0, cf=0.05, flutter=1.4, fcoef=0.42, lam=0.6, span=1.8, shear=3.0, bend=0.8, iters=3, comp=0.0)
# oblique view of the cloth's depth: screen += z * VIEW (we look at flags slightly from below, so crests toward the
# camera rise a little: the top/bottom hems show the waves). Hoist (z=0) is unaffected.
VIEW = (0.06, -0.24)


def cloth_color(k):
    """k in [-1, 2]: -1 deep shadow, -0.5 shadow, 0 base yellow, +1 highlight, +2 glow. Only yellows."""
    k = float(min(max(k, -1.0), 2.0))
    ks = (-1.0, -0.5, 0.0, 1.0, 2.0)
    for i in range(4):
        if k <= ks[i + 1]:
            f = (k - ks[i]) / (ks[i + 1] - ks[i])
            return tuple(PAL5[i] + (PAL5[i + 1] - PAL5[i]) * f)
    return tuple(PAL5[4])


def _norm_light(light):
    L = np.asarray(DEFAULT_LIGHT if light is None else light, np.float64)
    return L / (np.linalg.norm(L) + 1e-12)


def _shade_params(L, lift=0.0):
    back = float(np.clip((0.35 - L[2]) / 0.85, 0.0, 1.0))
    return np.array([-1.0 - 0.1 * back, 1.75 - 0.5 * back, 0.4, 0.35 + 1.65 * back, 0.55, 12.0, 0.9, lift],
                    np.float64)


# ============================================================ pole
def _pole_frame(x0, y0, x1, y1, width, light):
    """half-width normal (+n side = gradient offset 0) and highlight position across the pole"""
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) + 1e-9
    nx, ny = -dy / L * width / 2, dx / L * width / 2
    Ld = _norm_light(light)
    facing = (nx * Ld[0] + ny * Ld[1]) / (width / 2 + 1e-9)
    hl = 0.5 - 0.28 * float(np.clip(facing * 1.6, -1, 1))
    return nx, ny, hl


def draw_pole(ctx, x0, y0, x1, y1, width=7.0, alpha=1.0, light=None):
    """smooth untreated light wood, soft cylindrical shading. (x0,y0) base -> (x1,y1) top."""
    nx, ny, hl = _pole_frame(x0, y0, x1, y1, width, light)
    g = cairo.LinearGradient(x0 + nx, y0 + ny, x0 - nx, y0 - ny)
    ps, pl = C["pole_shadow"], C["pole"]
    lite = tuple(p + (b - p) * 0.35 for p, b in zip(pl, C["bone"]))
    g.add_color_stop_rgba(0.0, *ps, alpha)
    g.add_color_stop_rgba(max(0.05, hl - 0.3), *pl, alpha)
    g.add_color_stop_rgba(hl, *lite, alpha)
    g.add_color_stop_rgba(min(0.95, hl + 0.3), *pl, alpha)
    g.add_color_stop_rgba(1.0, *ps, alpha)
    ctx.new_path()
    ctx.move_to(x0 + nx, y0 + ny)
    ctx.line_to(x1 + nx, y1 + ny)
    ctx.line_to(x1 - nx, y1 - ny)
    ctx.line_to(x0 - nx, y0 - ny)
    ctx.close_path()
    ctx.set_source(g)
    ctx.fill()


# ============================================================ mesh refinement
_UPM = {}


def _up_matrix(n, r):
    key = (n, r)
    M = _UPM.get(key)
    if M is None:
        m = (n - 1) * r + 1
        M = np.zeros((m, n))
        for q in range(m):
            s = q / r
            i1 = min(int(s), n - 2)
            t = s - i1
            idx = (i1 - 1, i1, i1 + 1, i1 + 2)
            t2, t3 = t * t, t * t * t
            wts = (0.5 * (-t + 2 * t2 - t3), 0.5 * (2 - 5 * t2 + 3 * t3), 0.5 * (t + 4 * t2 - 3 * t3), 0.5 * (-t2 + t3))
            for ii, wt in zip(idx, wts):
                if ii < 0:        # linear extrapolation p[-1] = 2 p0 - p1
                    M[q, 0] += 2 * wt
                    M[q, 1] -= wt
                elif ii > n - 1:
                    M[q, n - 1] += 2 * wt
                    M[q, n - 2] -= wt
                else:
                    M[q, ii] += wt
        _UPM[key] = M
    return M


def _refine(A, ry, rx):
    """A (ny, nx, ...) -> Catmull-Rom refined grid"""
    if rx == 1 and ry == 1:
        return A
    My = _up_matrix(A.shape[0], ry)
    Mx = _up_matrix(A.shape[1], rx)
    if A.ndim == 2:
        return My @ A @ Mx.T
    return np.einsum("ab,bcd,ec->aed", My, A, Mx, optimize=True)


def _normals(P):
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n


def _cavity(V):
    """fold occlusion on the (coarse) sim grid: 1 in valleys seen from the camera"""
    Pp = np.pad(V, ((1, 1), (1, 1), (0, 0)), mode="edge")
    lap = 0.25 * (Pp[:-2, 1:-1] + Pp[2:, 1:-1] + Pp[1:-1, :-2] + Pp[1:-1, 2:]) - V
    n = _normals(V)
    n *= np.where(n[..., 2:3] < 0, -1.0, 1.0)
    sp = (np.linalg.norm(V[:, -1] - V[:, 0], axis=-1).mean() + 1e-6) / max(1, V.shape[1] - 1)
    c = np.sum(lap * n, -1) / (0.35 * sp)
    return np.clip(c, 0.0, 1.0) ** 0.8


# ============================================================ cloth rendering
def _clip_rect(ctx):
    ctx.save()
    ctx.identity_matrix()
    try:
        x0, y0, x1, y1 = ctx.clip_extents()
    finally:
        ctx.restore()
    return int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))


def draw_cloth(ctx, verts, alpha=1.0, glow=0.0, light=None, lift=0.0, pole_seg=None, _cav=True):
    """Render a cloth grid (ny, nx, 3) in design/world px through the plain-yellow renderer.
    pole_seg = (x0, y0, x1, y1, width): a pole segment (top part) depth-sorted with the cloth, so cloth that
    swings behind the pole is hidden by it (draw the full pole with draw_pole first)."""
    V = np.asarray(verts, np.float64)
    ny, nx = V.shape[:2]
    xx, yx, xy, yy, x0, y0 = ctx.get_matrix()
    det = abs(xx * yy - xy * yx)
    ds = math.sqrt(det) + 1e-12
    # size on the device -> refinement / supersampling
    ext = V[..., :2].reshape(-1, 2)
    span = float(np.max(ext.max(0) - ext.min(0))) * ds
    if span < 0.5 or alpha <= 0.0:
        return
    r = 1 if span < 40 else 2 if span < 160 else 3 if span < 500 else 4
    cav = _cavity(V) if _cav else np.zeros((ny, nx))
    P = _refine(V, r, r)
    Nn = _normals(P)
    cavf = _refine(cav, r, r)
    vwx, vwy = VIEW
    Xo = P[..., 0] + vwx * P[..., 2]
    Yo = P[..., 1] + vwy * P[..., 2]
    D = np.empty_like(P)
    D[..., 0] = xx * Xo + xy * Yo + x0
    D[..., 1] = yx * Xo + yy * Yo + y0
    D[..., 2] = P[..., 2] * ds
    margin = 2
    if glow > 0:
        sig = max(2.0, 0.07 * span)
        margin = int(3 * sig) + 2
    pq = np.full((4, 2), np.nan)
    prad, hl = 0.0, 0.5
    if pole_seg is not None:
        px0, py0, px1, py1, pwid = pole_seg
        pnx, pny, hl = _pole_frame(px0, py0, px1, py1, pwid, light)
        cw = np.array([[px0 + pnx, py0 + pny], [px0 - pnx, py0 - pny], [px1 - pnx, py1 - pny], [px1 + pnx, py1 + pny]])
        pq[:, 0] = xx * cw[:, 0] + xy * cw[:, 1] + x0
        pq[:, 1] = yx * cw[:, 0] + yy * cw[:, 1] + y0
        prad = 0.5 * pwid * ds
    allx = np.concatenate([D[..., 0].ravel(), pq[:, 0]]) if pole_seg is not None else D[..., 0]
    ally = np.concatenate([D[..., 1].ravel(), pq[:, 1]]) if pole_seg is not None else D[..., 1]
    bx0 = int(math.floor(allx.min())) - margin
    by0 = int(math.floor(ally.min())) - margin
    bx1 = int(math.ceil(allx.max())) + margin
    by1 = int(math.ceil(ally.max())) + margin
    cx0, cy0, cx1, cy1 = _clip_rect(ctx)
    bx0, by0 = max(bx0, cx0), max(by0, cy0)
    bx1, by1 = min(bx1, cx1), min(by1, cy1)
    w, h = bx1 - bx0, by1 - by0
    if w <= 0 or h <= 0:
        return
    area = w * h
    ss = 4 if area < 40_000 else 3 if area < 250_000 else 2 if area < 2_000_000 else 1
    L = _norm_light(light)
    prm = _shade_params(L, lift)
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    buf = np.zeros((h, stride // 4, 4), np.uint8)
    if r >= 2:
        Kv = np.empty(P.shape[:2])
        _ras.shade_grid(Nn, cavf, L, prm, Kv)
    else:
        Kv = _EMPTY2
    cov = np.zeros((h, w), np.float32)
    _ras.raster_cloth(D, Nn, cavf, float(bx0), float(by0), w, h, ss, L, prm, PAL5, buf, pq, prad, hl, _PPAL, Kv, cov)
    ctx.save()
    ctx.identity_matrix()
    if glow > 0:
        hal = cv2.GaussianBlur(cov, (0, 0), sig) * min(1.0, glow) * 0.85 * alpha
        hb = np.zeros_like(buf)
        gc = FLAG_G * 0.6 + FLAG_Y * 0.4
        for c, ch in ((0, 2), (1, 1), (2, 0)):
            hb[:, :w, c] = np.clip(hal * gc[ch] * 255.0, 0, 255).astype(np.uint8)
        hb[:, :w, 3] = np.clip(hal * 255.0, 0, 255).astype(np.uint8)
        hs = cairo.ImageSurface.create_for_data(memoryview(hb), cairo.FORMAT_ARGB32, w, h, stride)
        ctx.set_operator(cairo.OPERATOR_ADD)
        ctx.set_source_surface(hs, bx0, by0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_ARGB32, w, h, stride)
    ctx.set_source_surface(surf, bx0, by0)
    if alpha >= 1.0:
        ctx.paint()
    else:
        ctx.paint_with_alpha(alpha)
    ctx.restore()
    surf.finish()


# ============================================================ track
class FlagTrack:
    def __init__(self, verts, poles, energy=None, snap=None, meta=None, flap_hz=None):
        self.verts = np.asarray(verts, np.float32)
        self.poles = np.asarray(poles, np.float32)
        self.n = len(self.verts)
        z = np.zeros(self.n, np.float32)
        self.energy = z.copy() if energy is None else np.asarray(energy, np.float32)
        self.snap = z.copy() if snap is None else np.asarray(snap, np.float32)
        self.flap_hz = z.copy() if flap_hz is None else np.asarray(flap_hz, np.float32)
        self.meta = dict(meta or {})
        self.meta.setdefault("pole", 420.0)
        self.meta.setdefault("pole_w", self.meta["pole"] * POLE_W)
        self.meta.setdefault("side", 1)

    def _frame(self, i):
        if isinstance(i, (int, np.integer)) or float(i).is_integer():
            k = int(min(max(int(i), 0), self.n - 1))
            return self.verts[k], self.poles[k]
        i = min(max(float(i), 0.0), self.n - 1.0)
        k = int(math.floor(i))
        k1 = min(k + 1, self.n - 1)
        f = i - k
        return (self.verts[k] * (1 - f) + self.verts[k1] * f, self.poles[k] * (1 - f) + self.poles[k1] * f)

    def pole_top(self, i):
        _, (x, y, a) = self._frame(i)
        L = self.meta["pole"]
        return float(x + math.sin(a) * L), float(y - math.cos(a) * L)

    def cloth_center(self, i):
        V, _ = self._frame(i)
        c = V[..., :2].reshape(-1, 2).mean(0)
        return float(c[0]), float(c[1])

    def draw(self, ctx, i, alpha=1.0, glow=0.0, pole=True, cloth=True, light=DEFAULT_LIGHT):
        V, (x, y, a) = self._frame(i)
        L = self.meta["pole"]
        tx, ty = x + math.sin(a) * L, y - math.cos(a) * L
        pw = self.meta.get("pole_w", 7.0)
        if pole:
            draw_pole(ctx, x, y, tx, ty, pw, alpha, light)
        if cloth:
            seg = None
            if pole:
                sl = min(1.0, (self.meta.get("width", CLOTH_W * L) + self.meta.get("height", CLOTH_H * L)) * 1.05 / L)
                seg = (tx + (x - tx) * sl, ty + (y - ty) * sl, tx, ty, pw)
            draw_cloth(ctx, V, alpha, glow, light, pole_seg=seg)

    def save(self, path):
        np.savez_compressed(path, verts=self.verts, poles=self.poles, energy=self.energy, snap=self.snap,
                            flap_hz=self.flap_hz, meta=np.array([repr(self.meta)]))

    @staticmethod
    def load(path):
        z = np.load(path)
        fh = z["flap_hz"] if "flap_hz" in z.files else None
        return FlagTrack(z["verts"], z["poles"], z["energy"], z["snap"], eval(str(z["meta"][0])), fh)

    def export_foley(self, S, name, pan=None, f0=0, gain=1.0):
        """write cache/foley/<scene>_<name>.npz for the sound team (energy, pan, snap events, flap_hz, snap)."""
        from . import foley
        n = S.n
        e = np.zeros(n, np.float32)
        sn = np.zeros(n, np.float32)
        fh = np.zeros(n, np.float32)
        lo, hi = max(0, f0), min(n, f0 + self.n)
        if hi > lo:
            e[lo:hi] = self.energy[lo - f0:hi - f0]
            sn[lo:hi] = self.snap[lo - f0:hi - f0]
            fh[lo:hi] = self.flap_hz[lo - f0:hi - f0]
        if pan is None:
            cx = self.verts[..., 0].mean(axis=(1, 2))
            pv = np.clip((cx - 960.0) / 960.0, -1, 1).astype(np.float32)
            pan = np.zeros(n, np.float32)
            if hi > lo:
                pan[lo:hi] = pv[lo - f0:hi - f0]
                pan[:lo] = pan[lo]
                pan[hi:] = pan[hi - 1]
        panv = np.broadcast_to(np.asarray(pan, np.float32), (n,))
        ev = [[S.start + k / S.fps, float(sn[k]), float(panv[k])] for k in np.nonzero(sn > 0)[0]]
        foley.export_track(S, name, e, panv, gain=gain, kind="flag", events=ev, flap_hz=fh, snap=sn)


def _postprocess_audio(raw_e, tension, verts, fly, fps):
    """physics -> audio descriptors. raw_e: max-over-substeps mean cloth speed rel. to the pole (m/s);
    tension: max-over-substeps mean stretch strain; verts (n,ny,nx,3) (m); fly: fly-edge speed rel. to pole (m/s)."""
    n = len(raw_e)
    energy = (1.0 - np.exp(-raw_e / 2.2)).astype(np.float32)
    snap = np.zeros(n, np.float32)
    flap = np.zeros(n, np.float32)
    if n < 3:
        return energy, snap, flap
    # snaps: sudden fly-edge deceleration (the whip cracking) + tension spikes above the local baseline
    k = max(3, int(fps * 0.5))
    pad = np.pad(tension, (k, k), mode="edge")
    base = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(n)])
    ex = np.maximum(0.0, tension - 1.5 * base - 0.006)
    dfly = np.maximum(0.0, -np.diff(fly, prepend=fly[:1]))
    sraw = np.maximum(ex / 0.03, dfly / 6.0)
    for i in range(n):
        lo, hi = max(0, i - 3), min(n, i + 4)
        if sraw[i] > 1.0 and sraw[i] >= sraw[lo:hi].max():
            snap[i] = min(1.0, 0.35 + 0.65 * (sraw[i] - 1.0) / 2.0)
    # flap rate: fly-edge excursion normal to the cloth (relative to the cloth centre), zero crossings with
    # hysteresis, 0.75 s window
    ny, nx = verts.shape[1:3]
    mid, jm = nx // 2, ny // 2
    tu = verts[:, jm, -1] - verts[:, jm, 0]
    tv = verts[:, -1, 0] - verts[:, 0, 0]
    nn = np.cross(tu, tv)
    nn /= np.linalg.norm(nn, axis=-1, keepdims=True) + 1e-9
    d = np.sum((verts[:, jm, -1] - verts[:, jm, mid]) * nn, -1)
    win = max(3, int(fps * 0.75))
    dm = d - np.convolve(np.pad(d, (win, win), mode="edge"), np.ones(2 * win + 1) / (2 * win + 1), mode="same")[win:-win]
    h = 0.012 * float(np.linalg.norm(tu, axis=-1).mean() + 1e-6)
    cross = np.zeros(n)
    state = 0
    for i in range(n):
        s = 1 if dm[i] > h else -1 if dm[i] < -h else 0
        if s != 0 and state != 0 and s != state:
            cross[i] = 1.0
        if s != 0:
            state = s
    cs = np.convolve(cross, np.ones(win) / win, mode="same")
    flap = (cs * fps / 2.0).astype(np.float32)
    return energy, snap, flap


class Flag:
    def __init__(self, width=None, height=None, pole=420.0, nx=22, ny=15, side=1, seed=0, pole_w=None,
                 turbulence=1.0, flutter=1.0, stiffness=1.0):
        self.pole = float(pole)
        self.width = float(width) if width else CLOTH_W * self.pole
        self.height = float(height) if height else CLOTH_H * self.pole
        self.nx, self.ny = int(nx), int(ny)
        self.side = 1 if side >= 0 else -1
        self.seed = int(seed)
        self.pole_w = float(pole_w) if pole_w else max(1.5, POLE_W * self.pole)
        self.turbulence, self.flutter, self.stiffness = float(turbulence), float(flutter), float(stiffness)

    def simulate(self, n, pole_fn, wind_fn, fps=24, substeps=10, warmup=48, furl_fn=None):
        n = int(n)
        warm = int(max(0, warmup))
        T = warm + n
        mpp = _sim.POLE_M / self.pole
        poses_px = np.array([pole_fn(i) for i in range(n)], np.float64).reshape(n, 3)
        winds_px = np.zeros((n, 3))
        for i in range(n):
            wv = tuple(wind_fn(i))
            winds_px[i, :len(wv)] = wv[:3]
        furl = np.zeros(n) if furl_fn is None else np.clip(np.array([float(furl_fn(i)) for i in range(n)]), 0, 1)
        poses = np.empty((T, 3))
        winds = np.empty((T, 3))
        furls = np.empty(T)
        poses[:warm] = poses_px[0]
        poses[warm:] = poses_px
        winds[:warm] = winds_px[0]
        winds[warm:] = winds_px
        furls[:warm] = furl[0]
        furls[warm:] = furl
        poses_m = poses.copy()
        poses_m[:, :2] *= mpp
        winds_m = winds * _sim.WIND_MPP
        # gusts: slow multiplicative envelope (deterministic per seed)
        tt = np.arange(T) / fps
        g = np.array([_sim.noise1(t * 0.35, self.seed + 5) + 0.5 * _sim.noise1(t * 0.9, self.seed + 6) for t in tt])
        winds_m *= (1.0 + 0.32 * self.turbulence * g)[:, None]
        # cuts: huge jumps in one frame
        cuts = np.zeros(T, np.bool_)
        dpos = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))
        dang = np.abs(np.diff(poses[:, 2]))
        cuts[1:] = (dpos > 0.6 * self.pole) | (dang > 1.2)
        Wc = self.width * mpp
        Hc = self.height * mpp
        Lp = _sim.POLE_M
        A, B, R, Ty, TA, TM, TB = _sim.build_constraints(self.nx, self.ny, Wc, Hc)
        out_x = np.zeros((n, self.ny, self.nx, 3))
        out_e = np.zeros(n)
        out_t = np.zeros(n)
        out_c = np.zeros(n)
        out_f = np.zeros(n)
        st = self.stiffness
        A_ = AERO
        prm = np.array([self.turbulence, A_["flutter"] * self.flutter, A_["cl"], A_["cd"], A_["cf"],
                        A_["fcoef"], A_["lam"], A_["span"], A_["comp"]], np.float64)
        _sim.simulate(self.nx, self.ny, Wc, Hc, Lp, poses_m, cuts, winds_m, furls, warm, float(fps),
                      int(substeps) * 3, A_["iters"], self.seed, prm,
                      A_["shear"] / st, A_["bend"] / st, float(self.side), self.pole_w * mpp * 0.5, A, B, R, Ty,
                      TA, TM, TB,
                      out_x, out_e, out_t, out_c, out_f)
        verts = (out_x / mpp).astype(np.float32)
        energy, snap, flap = _postprocess_audio(out_e, out_t, out_x, out_f, fps)
        meta = {"pole": self.pole, "pole_w": self.pole_w, "side": self.side, "width": self.width,
                "height": self.height, "seed": self.seed}
        return FlagTrack(verts, poses_px.astype(np.float32), energy, snap, meta, flap)


# ============================================================ crowd bank
BANK_VER = 7
BANK_WINDS = np.array([0.0, 0.3, 0.6, 1.0, 1.5])
BANK_PX = np.array([0.0, 260.0, 480.0, 760.0, 1200.0])
BANK_VAR = 4
BANK_L = 144
BANK_X = 24
BANK_NX, BANK_NY = 14, 10
_BANK = {}


def _build_bank():
    Lp = 420.0
    out = np.zeros((len(BANK_WINDS), BANK_VAR, BANK_L, BANK_NY, BANK_NX, 3), np.float32)
    for wi, wpx in enumerate(BANK_PX):
        for vi in range(BANK_VAR):
            fl = Flag(pole=Lp, nx=BANK_NX, ny=BANK_NY, seed=1000 + 17 * vi + wi, side=1)
            tr = fl.simulate(BANK_L + BANK_X, lambda i: (0.0, Lp, 0.0),
                             lambda i, w=wpx: (w, 0.0, 0.0), warmup=72)
            V = tr.verts.astype(np.float64)
            V[..., 1] -= 0.0      # top of pole is at (0, 0)
            V /= Lp
            loop = V[:BANK_L].copy()
            for j in range(BANK_X):
                f = j / BANK_X
                loop[j] = (1 - f) * V[BANK_L + j] + f * V[j]
            out[wi, vi] = loop
    return out


def _bank():
    b = _BANK.get("v")
    if b is not None:
        return b
    d = CACHE / "flagbank"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"bank_v{BANK_VER}_{BANK_NX}x{BANK_NY}.npy"
    if not path.exists():
        with open(d / "bank.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not path.exists():
                arr = _build_bank()
                tmp = d / f".tmp_{os.getpid()}.npy"
                np.save(tmp, arr)
                os.replace(tmp, path)
            fcntl.flock(lk, fcntl.LOCK_UN)
    arr = np.load(path)
    _BANK["v"] = arr
    return arr


def _bank_idx(wind, seed, t, fps=24.0):
    w = min(max(abs(float(wind)), 0.0), BANK_WINDS[-1])
    wi = min(max(int(np.searchsorted(BANK_WINDS, w)) - 1, 0), len(BANK_WINDS) - 2)
    fw = (w - BANK_WINDS[wi]) / (BANK_WINDS[wi + 1] - BANK_WINDS[wi])
    vi = int(seed) % BANK_VAR
    off = (int(seed) * 2654435761 % 1000003) / 1000003.0 * BANK_L
    fr = (t * fps + off) % BANK_L
    k0 = int(fr)
    return wi, fw, vi, k0, (k0 + 1) % BANK_L, fr - k0


def _bank_blend(A, wind, seed, t):
    wi, fw, vi, k0, k1, ff = _bank_idx(wind, seed, t)
    a = A[wi, vi, k0] * (1 - ff) + A[wi, vi, k1] * ff
    if fw > 1e-3:
        b = A[wi + 1, vi, k0] * (1 - ff) + A[wi + 1, vi, k1] * ff
        a = a * (1 - fw) + b * fw
    return a


def _bank_mesh(wind, seed, t, fps=24.0):
    """normalised cloth (ny, nx, 3), top of pole at origin, pole up, cloth toward +x"""
    return _bank_blend(_bank(), wind, seed, t)


def _bank_small():
    """7-point outlines + mean (camera-facing) normals of every bank frame, for tiny flags"""
    s = _BANK.get("s")
    if s is None:
        B = _bank().astype(np.float64)
        ny, nx = B.shape[3:5]
        jm, im = ny // 2, nx // 2
        idx = [(0, 0), (0, im), (0, nx - 1), (jm, nx - 1), (ny - 1, nx - 1), (ny - 1, im), (ny - 1, 0)]
        O = np.stack([B[..., j, i, :] for j, i in idx], -2)
        n = np.cross(np.gradient(B, axis=-2), np.gradient(B, axis=-3))
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        n *= np.where(n[..., 2:3] < 0, -1.0, 1.0)
        N = n.mean(axis=(-3, -2))
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
        s = (O, N)
        _BANK["s"] = s
    return s


_POLE_RGB = C["pole"]


def _draw_tiny(ctx, x, y, pole, t, wind, side, ang, seed, alpha, cloth_w, cloth_h, pop, lean, light, ds):
    """a few device pixels: 7-point outline, one shade, hairline pole. ~30 us."""
    O, N = _bank_small()
    sc = _ease_back(pop) if pop < 1.0 else 1.0
    if lean:
        ph = (int(seed) * 0.6180339887) % 1.0
        ang = ang + lean * (0.7 * math.sin(2 * math.pi * (t * 0.33 + ph)) + 0.3 * math.sin(2 * math.pi * (t * 0.71 + 2 * ph)))
    Lp = pole * min(1.0, max(0.0, pop * 1.6)) if pop < 1.0 else pole
    sd = (1 if side >= 0 else -1) * (-1 if wind < 0 else 1)
    P = _bank_blend(O, wind, seed, t)
    n = _bank_blend(N, wind, seed, t)
    ca, sa = math.cos(ang), math.sin(ang)
    tx, ty = x + sa * Lp, y - ca * Lp
    s = pole * sc
    fx = sd * s * (cloth_w / (CLOTH_W * pole) if cloth_w else 1.0)
    fy = s * (cloth_h / (CLOTH_H * pole) if cloth_h else 1.0)
    lx = P[:, 0] * fx
    ly = P[:, 1] * fy
    lz = P[:, 2] * s
    X = tx + lx * ca - ly * sa + VIEW[0] * lz
    Y = ty + lx * sa + ly * ca + VIEW[1] * lz
    # pole: hairline of solid wood colour
    ctx.new_path()
    ctx.move_to(x, y)
    ctx.line_to(tx, ty)
    ctx.set_line_width(max(0.7 / ds, POLE_W * pole))
    ctx.set_source_rgba(_POLE_RGB[0], _POLE_RGB[1], _POLE_RGB[2], alpha)
    ctx.stroke()
    # one shade from the mean normal
    nx_, ny_, nz_ = sd * n[0], n[1], n[2]
    nx2, ny2 = nx_ * ca - ny_ * sa, nx_ * sa + ny_ * ca
    L = _norm_light(light)
    prm = _shade_params(L)
    ndl = nx2 * L[0] + ny2 * L[1] + nz_ * L[2]
    diff = max(0.0, (ndl + prm[2]) / (1 + prm[2]))
    tr = max(0.0, -ndl)
    k = prm[0] + prm[1] * diff + prm[3] * tr * math.sqrt(tr)
    ctx.move_to(X[0], Y[0])
    for i in range(1, 7):
        ctx.line_to(X[i], Y[i])
    ctx.close_path()
    r, g, b = cloth_color(k)
    ctx.set_source_rgba(r, g, b, alpha)
    ctx.fill()


def _ease_back(p):
    p = min(max(p, 0.0), 1.0)
    c1 = 1.70158 * 1.2
    q = p - 1.0
    return 1.0 + (c1 + 1.0) * q ** 3 + c1 * q ** 2


def _flag_world(x, y, pole, t, wind, side, ang, seed, cloth_w, cloth_h, pop, lean):
    """world-space cloth mesh + pole geometry for a crowd flag"""
    sc = _ease_back(pop) if pop < 1.0 else 1.0
    if lean:
        ph = (int(seed) * 0.6180339887) % 1.0
        ang = ang + lean * (0.7 * math.sin(2 * math.pi * (t * 0.33 + ph)) + 0.3 * math.sin(2 * math.pi * (t * 0.71 + 2 * ph)))
    Lp = pole * min(1.0, max(0.0, pop * 1.6)) if pop < 1.0 else pole
    sdir = (1 if side >= 0 else -1) * (-1 if wind < 0 else 1)
    M = _bank_mesh(wind, seed, t)
    lx = M[..., 0] * sdir * (cloth_w / (CLOTH_W * pole) if cloth_w else 1.0)
    ly = M[..., 1] * (cloth_h / (CLOTH_H * pole) if cloth_h else 1.0)
    lz = M[..., 2]
    ca, sa = math.cos(ang), math.sin(ang)
    tx, ty = x + sa * Lp, y - ca * Lp
    s = pole * sc
    V = np.empty(M.shape)
    V[..., 0] = tx + (lx * ca - ly * sa) * s
    V[..., 1] = ty + (lx * sa + ly * ca) * s
    V[..., 2] = lz * s
    return V, (x, y, tx, ty)


def draw_flag(ctx, x, y, pole=420, t=0.0, wind=1.0, side=1, ang=0.0, seed=0, alpha=1.0,
              cloth_w=None, cloth_h=None, glow=0.0, pop=1.0, lean=0.0, light=None):
    if pop <= 0.0 or alpha <= 0.0:
        return
    xx, yx, xy, yy, _, _ = ctx.get_matrix()
    ds = math.sqrt(abs(xx * yy - xy * yx))
    if CLOTH_W * pole * ds < 12.0 and glow <= 0.0:
        _draw_tiny(ctx, x, y, float(pole), float(t), float(wind), side, float(ang), seed,
                   alpha * (min(1.0, pop * 12.0) if pop < 1.0 else 1.0), cloth_w, cloth_h, float(pop), float(lean),
                   light, ds)
        return
    a = alpha * (min(1.0, pop * 12.0) if pop < 1.0 else 1.0)
    if CLOTH_W * pole * ds < 150:
        # small/medium: the batch kernel (Gouraud on the bank mesh, AA, pole included)
        draw_flag_field(ctx, x, y, pole, t, seed, wind, side, ang, pop, alpha, light, glow, None, False, lean,
                        cloth_w, cloth_h)
        return
    V, (bx, by, tx, ty) = _flag_world(x, y, float(pole), float(t), float(wind), side, float(ang), seed,
                                      cloth_w, cloth_h, float(pop), float(lean))
    pw = max(0.7 / ds, POLE_W * pole)
    draw_pole(ctx, bx, by, tx, ty, pw, a, light)
    sl = min(1.0, 0.72 * pole / max(1e-6, math.hypot(tx - bx, ty - by)))
    draw_cloth(ctx, V, a, glow, light, pole_seg=(tx + (bx - tx) * sl, ty + (by - ty) * sl, tx, ty, pw))


def _ease_back_v(p):
    p = np.clip(p, 0.0, 1.0)
    c1 = 1.70158 * 1.2
    q = p - 1.0
    return 1.0 + (c1 + 1.0) * q ** 3 + c1 * q ** 2


_PPAL = np.array([C["pole_shadow"], C["pole"], tuple(p + (b - p) * 0.35 for p, b in zip(C["pole"], C["bone"]))])


def draw_flag_field(ctx, xs, ys, pole, t, seeds=None, wind=1.0, side=1, ang=0.0, pop=None, alpha=1.0,
                    light=None, glow=0.0, bright=None, sort=True, lean=0.0, cloth_w=None, cloth_h=None):
    """Thousands of crowd flags in one numba pass (anti-aliased, back-to-front by ys when sort=True).
    Any argument after ctx may be a scalar or a per-instance array. Plain yellow cloth only."""
    xs = np.atleast_1d(np.asarray(xs, np.float64))
    n = len(xs)
    if n == 0:
        return
    bc = lambda a, dt=np.float64: np.ascontiguousarray(np.broadcast_to(np.asarray(a, dt), (n,)))
    ys = bc(ys)
    pole = bc(pole)
    seeds = np.arange(n, dtype=np.int64) if seeds is None else bc(seeds, np.int64)
    wind = bc(wind)
    side = bc(side)
    ang = bc(ang).copy()
    pop = np.ones(n) if pop is None else bc(pop)
    alpha = bc(alpha).copy()
    tt = bc(t)
    lift = np.zeros(n) if bright is None else bc(bright) * 0.9
    lean = bc(lean)
    order = np.argsort(ys, kind="stable") if sort else np.arange(n)
    B = _bank()
    Lf = B.shape[2]
    w = np.clip(np.abs(wind), 0.0, BANK_WINDS[-1])
    wi = np.clip(np.searchsorted(BANK_WINDS, w) - 1, 0, len(BANK_WINDS) - 2).astype(np.int64)
    fw = (w - BANK_WINDS[wi]) / (BANK_WINDS[wi + 1] - BANK_WINDS[wi])
    vi = (seeds % BANK_VAR).astype(np.int64)
    off = ((seeds * 2654435761) % 1000003) / 1000003.0 * Lf
    fr = np.mod(tt * 24.0 + off, Lf)
    k0 = np.floor(fr).astype(np.int64) % Lf
    k1 = (k0 + 1) % Lf
    ff = fr - np.floor(fr)
    sdir = np.where(side >= 0, 1.0, -1.0) * np.where(wind < 0, -1.0, 1.0)
    if np.any(lean != 0):
        ph = (seeds * 0.6180339887) % 1.0
        ang += lean * (0.7 * np.sin(2 * np.pi * (tt * 0.33 + ph)) + 0.3 * np.sin(2 * np.pi * (tt * 0.71 + 2 * ph)))
    sc = np.where(pop < 1.0, _ease_back_v(pop), 1.0)
    Lp = pole * np.where(pop < 1.0, np.clip(pop * 1.6, 0.0, 1.0), 1.0)
    alpha *= np.where(pop < 1.0, np.clip(pop * 12.0, 0.0, 1.0), 1.0)
    alpha[pop <= 0.0] = 0.0
    s = pole * sc
    M = np.array(tuple(ctx.get_matrix()), np.float64)
    xx, yx, xy, yy, x0, y0 = M
    ds = math.sqrt(abs(xx * yy - xy * yx))
    pw = np.maximum(0.7 / ds, POLE_W * pole)
    sxm = np.ones(n) if cloth_w is None else bc(cloth_w) / (CLOTH_W * pole)
    sym = np.ones(n) if cloth_h is None else bc(cloth_h) / (CLOTH_H * pole)
    # device bounds of everything (conservative, handles lean and camera rotation) ∩ clip
    sn = np.abs(np.sin(ang))
    hw = (0.5 * np.maximum(1.0, sxm) + sn) * pole
    ytop = ys - 1.4 * np.maximum(1.0, sym) * pole
    ybot = ys + (0.05 + 0.6 * sn) * pole
    cx = np.stack([xs - hw, xs + hw, xs - hw, xs + hw], 1)
    cy = np.stack([ytop, ytop, ybot, ybot], 1)
    dx = xx * cx + xy * cy + x0
    dy = yx * cx + yy * cy + y0
    vis = alpha > 0
    if not np.any(vis):
        return
    c0, c1_, c2, c3 = _clip_rect(ctx)
    bx0 = max(c0, int(math.floor(dx[vis].min())))
    by0 = max(c1_, int(math.floor(dy[vis].min())))
    bx1 = min(c2, int(math.ceil(dx[vis].max())))
    by1 = min(c3, int(math.ceil(dy[vis].max())))
    W_, H_ = bx1 - bx0, by1 - by0
    if W_ <= 0 or H_ <= 0:
        return
    L = _norm_light(light)
    prm = _shade_params(L)
    gl = bc(glow).astype(np.float64)
    buf = np.zeros((H_, W_, 5), np.float32)
    o = order
    _ras.raster_field(B, wi[o], fw[o], vi[o], k0[o], k1[o], ff[o], sdir[o], s[o], Lp[o], ang[o], xs[o], ys[o],
                      alpha[o], lift[o], pw[o], M, np.array(VIEW, np.float64), L, prm, PAL5, _PPAL,
                      buf, float(bx0), float(by0), gl[o], np.ascontiguousarray(sxm[o]), np.ascontiguousarray(sym[o]))
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, W_)
    out = np.zeros((H_, stride // 4, 4), np.uint8)
    _ras.to_bgra(buf, out)
    ctx.save()
    ctx.identity_matrix()
    if np.any(gl > 0):
        a = np.ascontiguousarray(buf[..., 4])
        sig = float(np.clip(np.median(0.375 * s[vis] * ds) * 0.35, 1.5, 40.0))
        hal = np.minimum(cv2.GaussianBlur(a, (0, 0), sig) * 0.6, 1.0)
        hb = np.zeros_like(out)
        gc = FLAG_G * 0.6 + FLAG_Y * 0.4
        for c, ch in ((0, 2), (1, 1), (2, 0)):
            hb[:, :W_, c] = np.clip(hal * gc[ch] * 255.0, 0, 255).astype(np.uint8)
        hb[:, :W_, 3] = np.clip(hal * 255.0, 0, 255).astype(np.uint8)
        hs = cairo.ImageSurface.create_for_data(memoryview(hb), cairo.FORMAT_ARGB32, W_, H_, stride)
        ctx.set_operator(cairo.OPERATOR_ADD)
        ctx.set_source_surface(hs, bx0, by0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
    surf = cairo.ImageSurface.create_for_data(memoryview(out), cairo.FORMAT_ARGB32, W_, H_, stride)
    ctx.set_source_surface(surf, bx0, by0)
    ctx.paint()
    ctx.restore()
    surf.finish()


def _furl_mesh(pole, cloth_w=None, cloth_h=None, side=1, nxs=40, nys=10, seed=0):
    """static furled cloth in local coords (top of pole at origin, pole up), design px"""
    Wc = (cloth_w or CLOTH_W * pole)
    Hc = (cloth_h or CLOTH_H * pole)
    rp = max(1.5, POLE_W * pole) * 0.5
    mpp = _sim.POLE_M / pole
    V = np.zeros((nys, nxs, 3))
    for j in range(nys):
        hy = (0.006 * _sim.POLE_M + Hc * mpp * j / (nys - 1))
        for i in range(nxs):
            V[j, i] = _sim.furl_target(j, i, nxs, nys, Wc * mpp, Hc * mpp, 0.0, hy, 0.0, 0.0, float(side), rp * mpp)
    V /= mpp
    # a little per-seed irregularity
    rng = np.random.default_rng(seed)
    V[..., 0] *= 1.0 + 0.06 * rng.standard_normal()
    return V


_FURL = {}


def draw_furled(ctx, x, y, pole=420, ang=0.0, seed=0, alpha=1.0, light=None, glow=0.0, cloth_w=None, cloth_h=None):
    pole = float(pole)
    key = (round(pole, 1), cloth_w, cloth_h, seed % 4)
    M = _FURL.get(key)
    if M is None:
        M = _furl_mesh(pole, cloth_w, cloth_h, 1, seed=seed % 4)
        _FURL[key] = M
    ca, sa = math.cos(ang), math.sin(ang)
    tx, ty = x + sa * pole, y - ca * pole
    V = np.empty_like(M)
    V[..., 0] = tx + M[..., 0] * ca - M[..., 1] * sa
    V[..., 1] = ty + M[..., 0] * sa + M[..., 1] * ca
    V[..., 2] = M[..., 2]
    draw_pole(ctx, x, y, tx, ty, max(1.5, POLE_W * pole), alpha, light)
    draw_cloth(ctx, V, alpha, glow, light, pole_seg=(tx + (x - tx) * 0.4, ty + (y - ty) * 0.4, tx, ty,
                                                     max(1.5, POLE_W * pole)))

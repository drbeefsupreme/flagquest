"""vx.mosaic_sinopia - bare lime plaster with the red-ochre SINOPIA (the preparatory drawing under a mosaic), what
shows where the tesserae AND their setting bed have fallen (owner: Tessellator). Re-exported by vx.mosaic.

    sin = mz.Sinopia(extent=(W, H), cartoon=cart_rgb, seed=3)      # setup: wall-space image (cached per process)
    bg  = sin.view(fc, view, light=L)                             # per frame: lit + seen through the view
    img = mz.render(fc, cart, lay, present=codes, bg=bg, view=view, light=L)
          # present codes: 1 tile, 0 socket (setting bed with the tile imprint), -1 bare (bg = plaster + sinopia)

The drawing is derived from the cartoon's contours, drawn as the fresco painter would: a free, wobbly brush line of
red ochre (varying width, dry-brush breaks), fainter searching lines (pentimenti) beside it, soft ochre washes
in the shadows and a snapped-cord squaring grid; the plaster is warm lime (mottled, trowel marks, pits,
hairline cracks).
"""
import math

import cv2
import numpy as np

from .config import W, H
from .canvas import col


def _noise(h, w, cell, rng, interp=cv2.INTER_CUBIC):
    gh, gw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
    n = rng.standard_normal((gh, gw)).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=interp)


def _features(img, thr=13):
    q = (np.clip(img[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
    lab = cv2.cvtColor(q, cv2.COLOR_RGB2LAB).astype(np.int16)
    F = np.zeros(q.shape[:2], np.float32)
    dx = np.abs(np.diff(lab, axis=1)).sum(-1)
    dy = np.abs(np.diff(lab, axis=0)).sum(-1)
    F[:, :-1] = np.maximum(F[:, :-1], (dx > thr).astype(np.float32))
    F[:-1, :] = np.maximum(F[:-1, :], (dy > thr).astype(np.float32))
    return F


def _single_lines(F, s, thin_units=9.0):
    """one brush line per contour: strokes thinner than thin_units (two parallel colour boundaries) are replaced
    by their middle line; other boundaries are kept."""
    from scipy import ndimage
    D = cv2.distanceTransform((~F).astype(np.uint8), cv2.DIST_L2, 3)
    n, lab = cv2.connectedComponents((~F).astype(np.uint8), connectivity=4)
    Dmax = ndimage.maximum(D, lab, index=np.arange(n)).astype(np.float32)
    Dm = Dmax[lab]
    thin = (Dm < thin_units * s) & ~F
    ridge = thin & (D >= 0.62 * Dm)
    near_thin = cv2.dilate(thin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return (ridge | (F & ~near_thin)).astype(np.float32)


def _warp(m, amp_px, cell_px, rng):
    h, w = m.shape
    dx = _noise(h, w, cell_px, rng) * amp_px
    dy = _noise(h, w, cell_px, rng) * amp_px
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(m, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class Sinopia:
    """extent: chart size (design units); cartoon: the mosaic's cartoon (rgb, any resolution covering extent) or
    None; edges: optional extra line mask; s: px per design unit of the stored image; grid: squaring grid spacing
    (0 = none); wash: strength of shadow washes; lines: line strength; plaster/ink: colours."""

    def __init__(self, extent=(W, H), cartoon=None, edges=None, s=0.5, seed=0, grid=150.0, wash=0.35, lines=1.0,
                 plaster="#D3CCBE", ink="#A0442C", cracks=5, line_w=3.2):
        self.extent = (int(extent[0]), int(extent[1]))
        self.s = float(s)
        h, w = int(round(extent[1] * s)), int(round(extent[0] * s))
        rng = np.random.default_rng(seed)
        pl = np.array(col(plaster), np.float32)
        ink = np.array(col(ink), np.float32)
        # ---- plaster
        mott = _noise(h, w, 160 * s, rng) * 0.022 + _noise(h, w, 40 * s, rng) * 0.012
        trow = cv2.GaussianBlur(_noise(h, w, 6 * s, rng, cv2.INTER_LINEAR), (0, 0), sigmaX=22 * s, sigmaY=3 * s)
        grain = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.6) * 0.018
        v = 1.0 + mott + trow * 0.035 + grain
        img = pl[None, None, :] * v[..., None]
        img[..., 2] *= 1.0 - 0.012 * _noise(h, w, 200 * s, rng)         # warm / cool patches of lime
        # pits
        npits = int(w * h / (900.0 * max(s, 0.1) ** 0 * 60))
        py = rng.integers(0, h, npits)
        px = rng.integers(0, w, npits)
        pits = np.zeros((h, w), np.float32)
        pits[py, px] = rng.uniform(0.3, 1.0, npits)
        pits = cv2.GaussianBlur(pits, (0, 0), max(0.6, 0.9 * s)) * (2.5 + 2.5 * s)
        img *= (1.0 - 0.35 * np.clip(pits, 0, 1))[..., None]
        # ---- drawing
        a = np.zeros((h, w), np.float32)
        if cartoon is not None or edges is not None:
            F = np.zeros((h, w), np.float32)
            if cartoon is not None:
                c = np.asarray(cartoon, np.float32)[..., :3]
                c = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA if c.shape[1] > w else cv2.INTER_LINEAR)
                F = np.maximum(F, _single_lines(_features(c) > 0, s))
            else:
                c = None
            if edges is not None:
                e = np.asarray(edges, np.float32)
                if e.ndim == 3:
                    e = e[..., -1]
                F = np.maximum(F, cv2.resize(e, (w, h), interpolation=cv2.INTER_AREA))
            lw = max(line_w * s, 0.7)
            main = _warp(F, 2.2 * s * 1.6, 70 * s, rng)
            main = cv2.GaussianBlur(main, (0, 0), lw * 0.45)
            main = np.clip(main * 3.0, 0, 1)
            var = np.clip(0.55 + 0.45 * _noise(h, w, 40 * s, rng), 0.15, 1.0)
            dry = np.clip(0.75 + 0.35 * cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0),
                                                         max(0.5, 0.7 * s)), 0.2, 1.0)
            a = np.clip(main * var * dry * 1.25 * lines, 0, 0.95)
            pent = _warp(F, 7.0 * s, 110 * s, rng)
            pent = np.clip(cv2.GaussianBlur(pent, (0, 0), lw * 0.35) * 2.0, 0, 1)
            a = np.maximum(a, pent * 0.28 * lines * np.clip(0.4 + 0.6 * _noise(h, w, 60 * s, rng), 0, 1))
            if c is not None and wash > 0:
                L = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
                dark = np.clip((0.55 - L) / 0.55, 0, 1)
                dark = cv2.GaussianBlur(dark, (0, 0), 9 * s)
                strokes = np.clip(0.5 + 0.5 * cv2.GaussianBlur(_noise(h, w, 5 * s, rng, cv2.INTER_LINEAR), (0, 0),
                                                               sigmaX=2 * s, sigmaY=9 * s), 0, 1)
                a = np.maximum(a, dark * strokes * wash * 0.55)
        if grid and grid > 0:
            gl = np.zeros((h, w), np.float32)
            for x in np.arange(grid, extent[0], grid):
                xs = (x + np.cumsum(rng.normal(0, 0.25, h)) * 0.1) * s
                for yy in range(h):
                    xi = int(xs[yy])
                    if 0 <= xi < w:
                        gl[yy, xi] = 1.0
            for y in np.arange(grid, extent[1], grid):
                ys_ = (y + np.cumsum(rng.normal(0, 0.25, w)) * 0.1) * s
                for xx in range(w):
                    yi = int(ys_[xx])
                    if 0 <= yi < h:
                        gl[yi, xx] = 1.0
            gl = cv2.GaussianBlur(gl, (0, 0), max(0.5, 0.6 * s)) * 1.6
            a = np.maximum(a, np.clip(gl, 0, 1) * 0.22 * np.clip(0.6 + 0.5 * _noise(h, w, 80 * s, rng), 0.2, 1))
        inkv = ink[None, None, :] * (0.92 + 0.08 * _noise(h, w, 30 * s, rng))[..., None]
        img = img * (1 - a[..., None]) + (img * inkv / np.maximum(pl, 1e-3)) * a[..., None] * 0.35 + \
            inkv * a[..., None] * 0.65
        # hairline cracks in the plaster
        cr = np.zeros((h, w), np.uint8)
        for _ in range(int(cracks)):
            p = np.array([rng.uniform(0, w), rng.uniform(0, h)])
            d = rng.uniform(0, 2 * math.pi)
            pts = [p.copy()]
            for _k in range(int(rng.uniform(20, 80))):
                d += rng.normal(0, 0.35)
                p = p + 6 * s * np.array([math.cos(d), math.sin(d)])
                pts.append(p.copy())
            cv2.polylines(cr, [np.array(pts, np.int32)], False, 255, 1, cv2.LINE_AA)
        img *= (1.0 - 0.18 * cr.astype(np.float32)[..., None] / 255.0)
        self.image = np.clip(img, 0, 1).astype(np.float32)
        self.alpha = a

    # ---------------------------------------------------------------- viewing
    def view(self, fc, view=None, surface=None, light=None, ambient=None, lit=True):
        """the plaster + sinopia as seen through `view` (None: the extent at fc.s), optionally lit by `light`
        (matte: ambient + diffuse, same rig as render())."""
        from . import mosaic as mz
        v = mz._as_view(view)
        if v is None:
            img = cv2.resize(self.image, (int(round(self.extent[0] * fc.s)), int(round(self.extent[1] * fc.s))),
                             interpolation=cv2.INTER_LINEAR)
            img = _fit(img, fc.h, fc.w)
        elif v.kind in ("affine", "homog"):
            img = v.warp(fc, self.image, self.s, border=cv2.BORDER_REPLICATE)
        else:
            img = remap_chart(fc, self.image, self.s, v, surface if surface is not None else mz.Plane())
        if not lit:
            return img
        E = irradiance(fc, v, surface, light, ambient)
        lin = mz.srgb_to_lin(img) * E
        return mz.lin_to_srgb(np.clip(lin, 0, 1))


def _fit(img, h, w):
    out = np.zeros((h, w, 3), np.float32)
    hh, ww = min(h, img.shape[0]), min(w, img.shape[1])
    out[:hh, :ww] = img[:hh, :ww]
    return out


_MAPC = {}


def remap_chart(fc, img, s_img, cam, surface, step=4, fill=0.0):
    """a chart-space image seen through a Camera on a curved surface (per-pixel ray casting at 1/step res)."""
    key = (id(cam), tuple(np.round(cam.eye, 3)), tuple(np.round(cam.f, 5)), round(cam.F, 3), id(surface), fc.w, fc.h)
    m = _MAPC.get(key)
    if m is None:
        gw, gh = fc.w // step + 2, fc.h // step + 2
        gx, gy = np.meshgrid(np.arange(gw) * step, np.arange(gh) * step)
        D = cam.ray(np.stack([gx / fc.s, gy / fc.s], -1).astype(np.float64))
        P, N, ok = surface.intersect(cam.eye, D)
        uv = surface.to_chart(P)
        mx = (uv[..., 0] * s_img).astype(np.float32)
        my = (uv[..., 1] * s_img).astype(np.float32)
        mx[~ok] = -1e4
        my[~ok] = -1e4
        mx = cv2.resize(mx, (gw * step, gh * step), interpolation=cv2.INTER_LINEAR)[:fc.h, :fc.w]
        my = cv2.resize(my, (gw * step, gh * step), interpolation=cv2.INTER_LINEAR)[:fc.h, :fc.w]
        m = (mx, my)
        if len(_MAPC) > 8:
            _MAPC.pop(next(iter(_MAPC)))
        _MAPC[key] = m
    return cv2.remap(img, m[0], m[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def irradiance(fc, view=None, surface=None, light=None, ambient=None, step=24):
    """(fc.h, fc.w, 1|3) linear irradiance of a matte wall under the render() light rig (for plaster, sinopia,
    painted backgrounds)."""
    from . import mosaic as mz
    v = mz._as_view(view)
    Ls, amb = mz._lights(light, v, ambient)
    if v is None:
        v = mz.View()
    if v.kind in ("affine", "homog"):
        gx, gy, P, N = v.world_grid(fc, step)
        ok = np.ones(P.shape[:2], bool)
    else:
        gx, gy, P, N, ok = v.world_grid(fc, surface or mz.Plane(), step)
    E = np.broadcast_to(amb, P.shape[:2] + (3,)).astype(np.float64).copy()
    for L in Ls:
        c = L.lin().astype(np.float64)
        if L.directional:
            d = L.p / np.linalg.norm(L.p)
            ndl = np.clip((N @ d + 0.12) / 1.12, 0, None)
            E += ndl[..., None] * c
        else:
            dv = L.p - P
            d2 = (dv ** 2).sum(-1)
            dn = dv / np.sqrt(np.maximum(d2, 1e-9))[..., None]
            ndl = np.clip(((N * dn).sum(-1) + 0.12) / 1.12, 0, None)
            E += (ndl / (1 + d2 / L.radius ** 2))[..., None] * c
    E[~ok] = amb
    E = cv2.resize(E.astype(np.float32), (gx.shape[1] * step, gx.shape[0] * step), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(E[:fc.h, :fc.w])

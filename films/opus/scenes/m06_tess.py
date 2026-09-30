"""m06 tile engine: MOVING tesserae for Recursive Balkanization (scene-specific; owner M06).

Everything in m06 is a tile that moves: map fragments drifting apart (hierarchical rigid pieces), tiles
splitting into four as the camera zooms (scale-free quadtree LOD), a tessera growing to fill the frame
(dives), quadtree dust. Tiles are given in SCREEN design units and rendered at fc.s with the vx.mosaic
look (Voronoi-filled joints up to 1.18 half-size, bevel, gold ramp + glint, mortar bed).
This is a thin adapter over the raster: `draw` takes the same per-tile data as mz.Tiles (v1 contract:
xy, ang, size, rgb, gold, tilt, jit, alpha, joint glow), so it can be swapped for mz.render_tiles.

    draw(fc, img, xy, ang, size, rgb, gold=, tilt=, jit=, alpha=, joint=, light=)   # in place over img
    kids = lod_expand(*lod_levels(...), ...)               # quadtree children for big-on-screen tiles
    pyr = Pyramid(rgba, x0, y0, ppu); rgb, gold = pyr.sample(x, y, size)
    sh = Shatter(...)                                      # hierarchical rigid fragments, pure f(t)
"""

import math

import cv2
import numpy as np
from numba import njit

from vx import mosaic as mz
from vx.config import W, H

GOLD_D, GOLD, GOLD_L = mz.SC["gold_d"], mz.SC["gold"], mz.SC["gold_l"]
MORTAR = mz.SC["mortar"]
SPLIT_PX = 17.0          # a tile bigger than this on screen (design px) splits into four
TABLE_M = 4096


# ============================================================ material (vx.mosaic v0 formulas)
def material(rgb, gold, tilt, jit, light, var=0.06, sparkle=1.0):
    """per-tile colour + glint exactly as mz.render shades them (tilt already rotated to screen)."""
    lx, ly, li = light
    tc = np.clip(rgb * (1.0 + var * jit[:, :1]) + var * 0.35 * jit, 0, 1)
    t = tilt * 0.35
    facing = np.clip(0.55 + t[:, 0] * lx + t[:, 1] * ly, 0, 1.5)
    glint = np.clip(facing - 0.7, 0, None) ** 2 * 2.0 * sparkle * li
    if gold is not None:
        g = np.asarray(gold, np.float32)[:, None]
        gl = np.clip(facing * li, 0, 1.6)[:, None]
        gcol = GOLD_D + (GOLD - GOLD_D) * np.clip(gl * 1.2, 0, 1) + (GOLD_L - GOLD) * np.clip(gl - 0.8, 0, 1) * 1.4
        tc = tc * (1 - g) + gcol * g
        glint = glint * (1 + 2.5 * g[:, 0])
    tc = np.clip(tc * (0.9 + 0.2 * facing[:, None]), 0, 1)
    return tc.astype(np.float32), glint.astype(np.float32)


# ============================================================ the raster kernel
@njit(cache=True, fastmath=True)
def _paint(img, cx, cy, ca, sa, hs, asp, grout, col, glint, alpha, emit, lx, ly, mr, mg, mb, bed):
    h, w = img.shape[0], img.shape[1]
    n = cx.shape[0]
    X0, Y0, X1, Y1 = w, h, 0, 0
    for i in range(n):
        H = hs[i]
        if H <= 0.05 or alpha[i] <= 0.004:
            continue
        e = bed * H * (abs(ca[i]) + abs(sa[i])) + 1.0
        x0 = int(cx[i] - e)
        x1 = int(cx[i] + e) + 1
        y0 = int(cy[i] - e)
        y1 = int(cy[i] + e) + 1
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            continue
        X0 = min(X0, max(0, x0))
        Y0 = min(Y0, max(0, y0))
        X1 = max(X1, min(w, x1))
        Y1 = max(Y1, min(h, y1))
    if X1 <= X0 or Y1 <= Y0:
        return
    bw = X1 - X0
    bh = Y1 - Y0
    best = np.full((bh, bw), 9.0, np.float32)
    second = np.full((bh, bw), 9.0, np.float32)
    lab = np.full((bh, bw), -1, np.int32)
    for i in range(n):
        H = hs[i]
        if H <= 0.05 or alpha[i] <= 0.004:
            continue
        c = ca[i]
        s = sa[i]
        e = bed * H * (abs(c) + abs(s)) + 1.0
        x0 = max(X0, int(cx[i] - e))
        x1 = min(X1, int(cx[i] + e) + 1)
        y0 = max(Y0, int(cy[i] - e))
        y1 = min(Y1, int(cy[i] + e) + 1)
        inv = 1.0 / H
        iasp = 1.0 / max(asp[i], 0.02)
        for py in range(y0, y1):
            fy = py + 0.5 - cy[i]
            for px in range(x0, x1):
                fx = px + 0.5 - cx[i]
                u = fx * c + fy * s
                v = -fx * s + fy * c
                d = max(abs(u), abs(v) * iasp) * inv
                if d < bed:
                    qy = py - Y0
                    qx = px - X0
                    b = best[qy, qx]
                    if d < b:
                        second[qy, qx] = b
                        best[qy, qx] = d
                        lab[qy, qx] = i
                    elif d < second[qy, qx]:
                        second[qy, qx] = d
    for qy in range(bh):
        for qx in range(bw):
            i = lab[qy, qx]
            if i < 0:
                continue
            H = hs[i]
            b = best[qy, qx]
            y = Y0 + qy
            x = X0 + qx
            if b >= 1.18:
                # the setting bed between loosely laid tiles and around a fragment's broken edge
                a = alpha[i] * min(max((bed - b) * H + 0.3, 0.0), 1.0)
                img[y, x, 0] = img[y, x, 0] * (1.0 - a) + (mr * 0.8 + emit[i, 0]) * a
                img[y, x, 1] = img[y, x, 1] * (1.0 - a) + (mg * 0.8 + emit[i, 1]) * a
                img[y, x, 2] = img[y, x, 2] * (1.0 - a) + (mb * 0.8 + emit[i, 2]) * a
                continue
            m = min((1.18 - b) * H, (second[qy, qx] - b) * H * 0.5)
            e = m - 0.5 * grout[i]
            fx = X0 + qx + 0.5 - cx[i]
            fy = Y0 + qy + 0.5 - cy[i]
            u = (fx * ca[i] + fy * sa[i]) / H
            v = (-fx * sa[i] + fy * ca[i]) / (H * max(asp[i], 0.02))
            rim = e / (0.25 * H + 0.4)
            rim = min(max(rim, 0.0), 1.0)
            shade = 0.72 + 0.28 * rim
            lit = -(u * lx + v * ly) * 0.8
            lit = min(max(lit, 0.0), 1.0) * (1.0 - rim)
            gl = glint[i] * (0.35 + 0.65 * lit) * 0.9
            k = min(max(e + 0.5, 0.0), 1.0)
            r = (col[i, 0] * shade + gl) * k + (mr + emit[i, 0]) * (1.0 - k)
            g = (col[i, 1] * shade + gl) * k + (mg + emit[i, 1]) * (1.0 - k)
            bb = (col[i, 2] * shade + gl) * k + (mb + emit[i, 2]) * (1.0 - k)
            a = alpha[i]
            img[y, x, 0] = img[y, x, 0] * (1.0 - a) + r * a
            img[y, x, 1] = img[y, x, 1] * (1.0 - a) + g * a
            img[y, x, 2] = img[y, x, 2] * (1.0 - a) + bb * a


USE_V1 = True          # route draw() through vx.mosaic v1 render_tiles (the film's shared material look)


def draw(fc, img, xy, ang, size, rgb, gold=None, tilt=None, jit=None, alpha=None, joint=None, light=None,
         grout=None, mortar=None, var=0.06, sparkle=1.0, bed=1.55, aspect=None, ids=None, gain=None):
    """render tiles (SCREEN design units) over img (fc.h, fc.w, 3) float32, in place, with the mosaic look.
    rgb (N,3) smalti colour, gold (N,) 0..1 gold leaf, tilt (N,2)/jit (N,3) per-tile glass randomness (already
    rotated with the tile), alpha (N,), joint (N,3) light glowing in the mortar around those tiles (cracks),
    aspect (N,) 0..1 squash of the tile's v axis (a tile tumbling / flipping)."""
    n = len(xy)
    if n == 0:
        return img
    light = light if light is not None else (0.3, -0.5, 1.0)
    tilt = np.zeros((n, 2), np.float32) if tilt is None else tilt
    jit = np.zeros((n, 3), np.float32) if jit is None else jit
    if USE_V1:
        rot = None
        if aspect is not None:
            rot = np.zeros((n, 3), np.float32)
            rot[:, 0] = np.arccos(np.clip(np.asarray(aspect, np.float32), 0, 1))
        Tl = mz.Tiles(np.asarray(xy, np.float32), np.asarray(ang, np.float32), np.asarray(size, np.float32),
                      np.clip(np.asarray(rgb, np.float32), 0, 1), gold=gold, alpha=alpha, tilt=tilt, jit=jit,
                      ids=ids, rot=rot, gain=gain)
        out = mz.render_tiles(fc, Tl, light=light, bg=img, joint_emit=joint, loose=aspect is not None,
                              holes="bg")
        img[...] = out
        return img
    col, glint = material(np.asarray(rgb, np.float32), gold, tilt, jit, light, var, sparkle)
    s = fc.s
    xy = np.asarray(xy, np.float64)
    ang = np.asarray(ang, np.float64)
    cx = (xy[:, 0] * s).astype(np.float32)
    cy = (xy[:, 1] * s).astype(np.float32)
    ca = np.cos(ang).astype(np.float32)
    sa = np.sin(ang).astype(np.float32)
    hs = (np.asarray(size, np.float32) * 0.5 * s).astype(np.float32)
    if grout is None:
        g = np.maximum(0.55 * s, 0.108 * 2 * hs).astype(np.float32)
    else:
        g = (np.broadcast_to(np.asarray(grout, np.float32), (n,)) * s).astype(np.float32)
    a = np.ones(n, np.float32) if alpha is None else np.broadcast_to(np.asarray(alpha, np.float32), (n,)).astype(np.float32)
    em = np.zeros((n, 3), np.float32) if joint is None else np.ascontiguousarray(joint, np.float32)
    lx, ly = light[0], light[1]
    mc = (MORTAR if mortar is None else np.asarray(mortar, np.float32)) * 0.85
    asp = np.ones(n, np.float32) if aspect is None else np.broadcast_to(np.asarray(aspect, np.float32), (n,)).astype(np.float32)
    _paint(img, cx, cy, ca, sa, hs, asp, g, col, glint, np.ascontiguousarray(a), em, float(lx), float(ly),
           float(mc[0]), float(mc[1]), float(mc[2]), float(max(bed, 1.18)))
    return img


# ============================================================ loose specks (dust), soft shadows
@njit(cache=True, fastmath=True)
def _specks(img, x, y, r, col, a):
    """tiny square specks with coverage AA (r = half size px, may be < 1)."""
    h, w = img.shape[0], img.shape[1]
    for i in range(x.shape[0]):
        R = max(r[i], 0.35)
        cov_k = min(1.0, (r[i] * 2.0) ** 2) / max(1e-6, (2.0 * R) ** 2) if r[i] < 0.5 else 1.0
        x0 = int(x[i] - R)
        x1 = int(x[i] + R) + 1
        y0 = int(y[i] - R)
        y1 = int(y[i] + R) + 1
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            continue
        for py in range(max(0, y0), min(h, y1 + 1)):
            oy = min(py + 1.0, y[i] + R) - max(float(py), y[i] - R)
            if oy <= 0:
                continue
            for px in range(max(0, x0), min(w, x1 + 1)):
                ox = min(px + 1.0, x[i] + R) - max(float(px), x[i] - R)
                if ox <= 0:
                    continue
                k = min(1.0, ox * oy) * a[i] * cov_k
                img[py, px, 0] = img[py, px, 0] * (1 - k) + col[i, 0] * k
                img[py, px, 1] = img[py, px, 1] * (1 - k) + col[i, 1] * k
                img[py, px, 2] = img[py, px, 2] * (1 - k) + col[i, 2] * k


def specks(fc, img, xy, size, col, alpha=None):
    s = fc.s
    n = len(xy)
    if n == 0:
        return img
    a = np.ones(n, np.float32) if alpha is None else np.broadcast_to(np.asarray(alpha, np.float32), (n,)).astype(np.float32)
    _specks(img, (xy[:, 0] * s).astype(np.float32), (xy[:, 1] * s).astype(np.float32),
            (np.asarray(size, np.float32) * 0.5 * s).astype(np.float32), np.ascontiguousarray(col, np.float32), a)
    return img


@njit(cache=True, fastmath=True)
def _cover(buf, x, y, r, a):
    h, w = buf.shape
    for i in range(x.shape[0]):
        x0 = max(0, int(x[i] - r[i]))
        x1 = min(w, int(x[i] + r[i]) + 1)
        y0 = max(0, int(y[i] - r[i]))
        y1 = min(h, int(y[i] + r[i]) + 1)
        for py in range(y0, y1):
            for px in range(x0, x1):
                buf[py, px] = max(buf[py, px], a[i])


def shadow(fc, img, xy, size, off, sigma, k, alpha=None, down=4):
    """soft drop shadow of tiles (screen design units) multiplied into img: off=(dx,dy) design px."""
    if len(xy) == 0:
        return img
    s = fc.s / down
    h, w = int(math.ceil(fc.h / down)), int(math.ceil(fc.w / down))
    buf = np.zeros((h, w), np.float32)
    a = np.ones(len(xy), np.float32) if alpha is None else np.asarray(alpha, np.float32)
    _cover(buf, ((xy[:, 0] + off[0]) * s).astype(np.float32), ((xy[:, 1] + off[1]) * s).astype(np.float32),
           (np.asarray(size, np.float32) * 0.62 * s).astype(np.float32), a)
    buf = cv2.GaussianBlur(buf, (0, 0), max(0.6, sigma * s))
    buf = cv2.resize(buf, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)
    img *= (1.0 - k * np.clip(buf, 0, 1))[..., None]
    return img


# ============================================================ quadtree LOD
def _child_tables(lmax):
    """unit offsets (in parent sizes) of the 4^L children at every depth: prefix (all but last) and last."""
    pre, last = [None], [None]
    for L in range(1, lmax + 1):
        k = np.arange(4 ** L)
        p = np.zeros((len(k), 2))
        q = np.zeros((len(k), 2))
        for j in range(L):
            d = (k >> (2 * (L - 1 - j))) & 3        # j = 0 is the first (outermost) split
            sx = (d & 1) * 2 - 1
            sy = (d >> 1) * 2 - 1
            o = np.stack([sx, sy], 1) / 2.0 ** (j + 2)
            if j < L - 1:
                p += o
            else:
                q += o
        pre.append(p)
        last.append(q)
    return pre, last


_TABLES = _child_tables(6)
_RNG = np.random.default_rng(606)
TILT_T = _RNG.normal(0, 1, (TABLE_M, 2)).astype(np.float32)
JIT_T = _RNG.normal(0, 1, (TABLE_M, 3)).astype(np.float32)
ANG_T = _RNG.normal(0, 1, TABLE_M).astype(np.float32)


def lod_levels(size_px, lam_extra=0.0, delta=0.0, lmax=5, split_px=SPLIT_PX, width=0.22):
    """per tile: (level, progress of the latest split 0..1). size_px = on-screen size (design px)."""
    lam = np.log2(np.maximum(size_px, 1e-6) / split_px) + 1.0 + lam_extra - delta
    L = np.clip(np.floor(lam), 0, lmax).astype(np.int64)
    p = np.where(L > 0, np.clip((lam - L) / width, 0, 1), 1.0)
    p = np.where(lam >= lmax + 1, 1.0, p)
    return L, p.astype(np.float32)


def lod_expand(level, prog, sxy, sang, ssize, hxy, hang, hsize, uid, sep=0.10):
    """expand tiles into their quadtree children.
    level/prog: from lod_levels. s* = screen centre/angle/size, h* = home (local) centre/angle/size, uid (N,) int
    stable tile ids. Returns dict with screen xy/ang/size, home xy/size, parent index, level, prog, child uid."""
    out = {k: [] for k in ("sxy", "sang", "ssize", "hxy", "hsize", "par", "lev", "prog", "cid")}
    pre, last = _TABLES
    for L in np.unique(level):
        idx = np.nonzero(level == L)[0]
        if L == 0:
            out["sxy"].append(sxy[idx]); out["sang"].append(sang[idx]); out["ssize"].append(ssize[idx])
            out["hxy"].append(hxy[idx]); out["hsize"].append(hsize[idx]); out["par"].append(idx)
            out["lev"].append(np.zeros(len(idx), np.int64)); out["prog"].append(np.ones(len(idx), np.float32))
            out["cid"].append(uid[idx].astype(np.int64))
            continue
        m = 4 ** L
        pr = prog[idx]
        k = np.arange(m)
        # offsets in parent-size units; the latest split opens up with progress (sep = extra separation)
        opn = (1.0 + sep * pr)[:, None, None]
        off = pre[L][None] + last[L][None] * opn                       # (n, m, 2)
        ca, sa = np.cos(sang[idx]), np.sin(sang[idx])
        o_s = off * ssize[idx][:, None, None]
        sx = sxy[idx][:, None, 0] + ca[:, None] * o_s[..., 0] - sa[:, None] * o_s[..., 1]
        sy = sxy[idx][:, None, 1] + sa[:, None] * o_s[..., 0] + ca[:, None] * o_s[..., 1]
        hc, hs_ = np.cos(hang[idx]), np.sin(hang[idx])
        o_h = (pre[L][None] + last[L][None]) * hsize[idx][:, None, None]
        hx = hxy[idx][:, None, 0] + hc[:, None] * o_h[..., 0] - hs_[:, None] * o_h[..., 1]
        hy = hxy[idx][:, None, 1] + hs_[:, None] * o_h[..., 0] + hc[:, None] * o_h[..., 1]
        cid = (uid[idx].astype(np.int64)[:, None] * 4099 + k[None] * 7 + L * 131)
        jj = cid % TABLE_M
        ja = ANG_T[jj] * 0.07 * np.minimum(1.0, pr)[:, None]
        # hand-laid, not a pixel grid: every child sits a little off its quadrant and varies in size
        cs = (ssize[idx] / 2 ** L)[:, None] * np.minimum(1.0, pr)[:, None]
        sx = sx + JIT_T[jj, 0] * 0.07 * cs
        sy = sy + JIT_T[jj, 1] * 0.07 * cs
        n = len(idx)
        out["sxy"].append(np.stack([sx.ravel(), sy.ravel()], 1))
        out["sang"].append((sang[idx][:, None] + ja).ravel())
        out["ssize"].append((np.repeat(ssize[idx] / 2 ** L, m) * np.repeat(1.0 - 0.04 * pr, m)
                             * (1.0 + 0.05 * JIT_T[jj, 2].ravel())))
        out["hxy"].append(np.stack([hx.ravel(), hy.ravel()], 1))
        out["hsize"].append(np.repeat(hsize[idx] / 2 ** L, m))
        out["par"].append(np.repeat(idx, m))
        out["lev"].append(np.full(n * m, L, np.int64))
        out["prog"].append(np.repeat(pr, m))
        out["cid"].append(cid.ravel())
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


def child_material(cid, ptilt, pjit):
    """tilt / jit for children: parent's, perturbed deterministically per child id."""
    j = cid % TABLE_M
    return (0.55 * ptilt + 0.8 * TILT_T[j]).astype(np.float32), (0.5 * pjit + 0.85 * JIT_T[j]).astype(np.float32)


# ============================================================ static cartoon sampler
class Pyramid:
    """sample a static rgba cartoon (rgb + gold in alpha slot) at world points with tile-size-aware blur.
    Stored as uint8 (memory is shared by forked render workers)."""

    def __init__(self, rgba, x0, y0, ppu):
        self.x0, self.y0, self.ppu = float(x0), float(y0), float(ppu)
        a = np.asarray(rgba)
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
        lv = [np.ascontiguousarray(a)]
        while min(lv[-1].shape[:2]) > 16 and len(lv) < 9:
            lv.append(cv2.pyrDown(lv[-1]))
        self.lv = lv

    def sample(self, x, y, size):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        out = np.zeros((len(x), 4), np.float32)
        l = np.clip(np.floor(np.log2(np.maximum(np.asarray(size) * self.ppu * 0.5, 1.0))), 0, len(self.lv) - 1).astype(int)
        for L in np.unique(l):
            idx = np.nonzero(l == L)[0]
            im = self.lv[L]
            k = self.ppu / 2 ** L
            u = (x[idx] - self.x0) * k - 0.5
            v = (y[idx] - self.y0) * k - 0.5
            h, w = im.shape[:2]
            u = np.clip(u, 0, w - 1.001)
            v = np.clip(v, 0, h - 1.001)
            i0 = np.floor(u).astype(int)
            j0 = np.floor(v).astype(int)
            fu = (u - i0)[:, None]
            fv = (v - j0)[:, None]
            a = im[j0, i0] * (1 - fu) + im[j0, i0 + 1] * fu
            b = im[j0 + 1, i0] * (1 - fu) + im[j0 + 1, i0 + 1] * fu
            out[idx] = a * (1 - fv) + b * fv
        out *= 1.0 / 255.0
        return out[:, :3], out[:, 3]


def sample_image(img, x, y, ppu=1.0, x0=0.0, y0=0.0):
    """bilinear sample of a (h,w,C) image at local points (units -> px via ppu)."""
    h, w = img.shape[:2]
    u = np.clip((np.asarray(x) - x0) * ppu - 0.5, 0, w - 1.001)
    v = np.clip((np.asarray(y) - y0) * ppu - 0.5, 0, h - 1.001)
    i0 = np.floor(u).astype(int)
    j0 = np.floor(v).astype(int)
    fu = (u - i0)
    fv = (v - j0)
    if img.ndim == 3:
        fu = fu[:, None]
        fv = fv[:, None]
    a = img[j0, i0] * (1 - fu) + img[j0, i0 + 1] * fu
    b = img[j0 + 1, i0] * (1 - fu) + img[j0 + 1, i0 + 1] * fu
    return a * (1 - fv) + b * fv


# ============================================================ hierarchical rigid fragments
def ease_jolt(tau, t_jolt=0.22, share=0.45, t_slow=2.6):
    """0 before a split; a violent jolt to `share` of the gap in t_jolt, then a slow drift to 1."""
    tau = np.maximum(tau, 0.0)
    a = 1.0 - (1.0 - np.clip(tau / t_jolt, 0, 1)) ** 3
    b = 1.0 - np.exp(-tau / t_slow)
    return share * a + (1 - share) * b


class Shatter:
    """Rigid fragments in a hierarchy of generations. For each generation g (1..G): node arrays
    parent[g] (index into gen g-1 nodes; gen 0 = one root), cen[g] (n,2) home centroid, t0[g] (n,) split time,
    dvec[g] (n,2) home-units displacement at full drift, spin[g] (n,) radians at full drift, and tiles' node ids
    tnode[g] (N,). node_mats(t) -> per-node 2x3 affines (home -> world), cumulative through the generations.
    After the jolt the pieces keep drifting apart slowly (lin = fraction of dvec per second)."""

    def __init__(self, parent, cen, t0, dvec, spin, tnode, jolt=(0.22, 0.45, 2.6), lin=0.06):
        self.parent, self.cen, self.t0, self.dvec, self.spin, self.tnode = parent, cen, t0, dvec, spin, tnode
        self.G = len(cen) - 1
        self.jolt = jolt
        self.lin = lin

    def drift(self, g, t):
        tau = t - self.t0[g]
        return ease_jolt(tau, *self.jolt) + self.lin * np.maximum(tau - 0.3, 0.0)

    def node_mats(self, t, upto=None):
        """cumulative 2x3 affine for every node of every generation at time t."""
        mats = [np.array([[[1.0, 0, 0], [0, 1.0, 0]]])]
        G = self.G if upto is None else upto
        for g in range(1, G + 1):
            f = self.drift(g, t)
            th = self.spin[g] * f
            d = self.dvec[g] * f[:, None]
            c, s = np.cos(th), np.sin(th)
            cx, cy = self.cen[g][:, 0], self.cen[g][:, 1]
            # rotate about centroid then translate: p' = R(p - c) + c + d
            A = np.zeros((len(th), 2, 3))
            A[:, 0, 0], A[:, 0, 1] = c, -s
            A[:, 1, 0], A[:, 1, 1] = s, c
            A[:, 0, 2] = cx - (c * cx - s * cy) + d[:, 0]
            A[:, 1, 2] = cy - (s * cx + c * cy) + d[:, 1]
            P = mats[-1][self.parent[g]]
            mats.append(compose(P, A))
        return mats

    def tile_mats(self, t):
        m = self.node_mats(t)
        return m[-1][self.tnode[self.G]]


def compose(P, A):
    """P o A for stacks of 2x3 affines."""
    out = np.empty_like(A)
    out[:, :, :2] = P[:, :, :2] @ A[:, :, :2]
    out[:, :, 2] = (P[:, :, :2] @ A[:, :, 2:3])[..., 0] + P[:, :, 2]
    return out


def apply(M, p):
    """per-point 2x3 affines (N,2,3) applied to points (N,2)."""
    return np.stack([M[:, 0, 0] * p[:, 0] + M[:, 0, 1] * p[:, 1] + M[:, 0, 2],
                     M[:, 1, 0] * p[:, 0] + M[:, 1, 1] * p[:, 1] + M[:, 1, 2]], 1)


def mat_angle(M):
    return np.arctan2(M[..., 1, 0], M[..., 0, 0])


# ============================================================ splitting helpers (setup time)
def noise_field(seed, scale):
    """smooth 2D noise function f(x, y) -> [-1, 1] (vectorised)."""
    from vx import value_noise2

    def f(x, y):
        return value_noise2(np.asarray(x) / scale, np.asarray(y) / scale, seed) * 0.7 + \
            value_noise2(np.asarray(x) / (scale * 0.37), np.asarray(y) / (scale * 0.37), seed + 11) * 0.3
    return f


def split_points(xy, k, seed, rough=0.35):
    """partition points into k jagged pieces: farthest-point seeds + noise-warped nearest seed."""
    n = len(xy)
    if n <= 1 or k <= 1:
        return np.zeros(n, np.int64)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    first = rng.integers(n)
    seeds = [first]
    d = np.linalg.norm(xy - xy[first], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d * rng.uniform(0.75, 1.0, n)))
        seeds.append(j)
        d = np.minimum(d, np.linalg.norm(xy - xy[j], axis=1))
    sp = xy[seeds]
    ext = max(1e-6, float(np.ptp(xy[:, 0]) + np.ptp(xy[:, 1])) / 2)
    D = np.linalg.norm(xy[:, None, :] - sp[None], axis=2)
    for j in range(k):
        f = noise_field(seed * 31 + j, ext * 0.35)
        D[:, j] += rough * ext * 0.5 * f(xy[:, 0], xy[:, 1])
    return np.argmin(D, axis=1)


# ============================================================ cameras + rendering a tile set through them
def view_matrix(cx, cy, zoom, rot=0.0, at=(960.0, 540.0)):
    """world -> screen (design units): world point (cx, cy) lands on `at`, scaled by zoom, rotated by rot."""
    c, s = math.cos(rot) * zoom, math.sin(rot) * zoom
    return np.array([[c, -s, at[0] - (c * cx - s * cy)], [s, c, at[1] - (s * cx + c * cy)]], np.float64)


def mat_scale(M):
    return np.sqrt(np.abs(M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]))


def rot2(v, th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1).astype(np.float32)


def render_set(fc, img, V, hxy, hang, hsize, tilt, jit, uid, colour_fn, M=None, light=None, alpha=None, joint=None,
               lam_extra=0.0, delta=None, lmax=5, margin=30.0, bed=2.1, cull=True, keep=None, extra=None, gain=None,
               batch=None):
    """Render a tile set: tiles at home/local coords (hxy, hang, hsize) -> world via per-tile 2x3 M (or identity)
    -> screen via V (2x3). Culls, splits big tiles into quadtree children (colours from colour_fn(local_xy,
    local_size, parent_idx) -> (rgb, gold)), rotates glass tilt with the tiles, draws. Returns the drawn dict.
    batch=list: append the prepared screen tiles instead of drawing (draw non-overlapping sets in ONE call with
    flush(); every render_tiles call costs ~60 ms of full-frame colour conversion)."""
    n = len(hxy)
    if n == 0:
        return None
    if keep is not None:
        sel = np.nonzero(keep)[0]
    else:
        sel = np.arange(n)
    if M is None:
        w = hxy[sel]
        wang = hang[sel]
        wsc = np.ones(len(sel))
    else:
        Ms = M[sel] if len(M) == n else M
        w = apply(Ms, hxy[sel])
        wang = hang[sel] + mat_angle(Ms)
        wsc = mat_scale(Ms)
    zoom = math.hypot(V[0, 0], V[1, 0])
    vrot = math.atan2(V[1, 0], V[0, 0])
    sxy = np.stack([V[0, 0] * w[:, 0] + V[0, 1] * w[:, 1] + V[0, 2], V[1, 0] * w[:, 0] + V[1, 1] * w[:, 1] + V[1, 2]], 1)
    ssize = hsize[sel] * wsc * zoom
    if cull:
        m = ssize * 1.2 + margin
        vis = (sxy[:, 0] > -m) & (sxy[:, 0] < W + m) & (sxy[:, 1] > -m) & (sxy[:, 1] < H + m) & (ssize > 0.15)
        if alpha is not None:
            vis &= np.asarray(alpha)[sel] > 0.004
        sel, sxy, ssize, wang = sel[vis], sxy[vis], ssize[vis], wang[vis]
    if len(sel) == 0:
        return None
    sang = wang + vrot
    th = sang - hang[sel]
    d = 0.0 if delta is None else delta[sel]
    le = lam_extra if np.isscalar(lam_extra) else np.asarray(lam_extra)[sel]
    L, p = lod_levels(ssize, le, d, lmax)
    kids = lod_expand(L, p, sxy, sang, ssize, hxy[sel], hang[sel], hsize[sel], uid[sel])
    par = kids["par"]
    rgb, gold = colour_fn(kids["hxy"], kids["hsize"], sel[par])
    ptilt = rot2(tilt[sel], th)
    tl, jt = ptilt[par], jit[sel][par]
    ch = kids["lev"] > 0
    if ch.any():
        tl2, jt2 = child_material(kids["cid"][ch], tl[ch], jt[ch])
        tl = tl.copy()
        jt = jt.copy()
        tl[ch], jt[ch] = tl2, jt2
    a = None if alpha is None else np.asarray(alpha, np.float32)[sel][par]
    jn = None if joint is None else np.asarray(joint, np.float32)[sel][par]
    if extra is not None:
        rgb, gold, a, jn = extra(kids, sel, rgb, gold, a, jn)
    gn = None if gain is None else np.asarray(gain, np.float32)[sel][par]
    if batch is not None:
        batch.append(dict(xy=kids["sxy"], ang=kids["sang"], size=kids["ssize"], rgb=rgb,
                          gold=np.zeros(len(rgb), np.float32) if gold is None else gold, tilt=tl, jit=jt,
                          alpha=np.ones(len(rgb), np.float32) if a is None else a,
                          joint=np.zeros((len(rgb), 3), np.float32) if jn is None else jn, ids=kids["cid"],
                          gain=np.ones(len(rgb), np.float32) if gn is None else gn))
        return kids
    draw(fc, img, kids["sxy"], kids["sang"], kids["ssize"], rgb, gold, tl, jt, alpha=a, joint=jn, light=light, bed=bed,
         ids=kids["cid"], gain=gn)
    kids["sel"] = sel
    return kids


def flush(fc, img, batch, light, bed=1.6):
    """draw every prepared set of a batch in one render call (sets must not overlap much)."""
    batch = [b for b in batch if b is not None and len(b["xy"])]
    if not batch:
        return img
    cat = {k: np.concatenate([b[k] for b in batch]) for k in batch[0]}
    return draw(fc, img, cat["xy"], cat["ang"], cat["size"], cat["rgb"], cat["gold"], cat["tilt"], cat["jit"],
                alpha=cat["alpha"], joint=cat["joint"], light=light, bed=bed, ids=cat["ids"], gain=cat["gain"])


# ============================================================ sprites: small mosaic pictures with their own layout
def _quant(v):
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, (tuple, list)):
        return tuple(_quant(x) for x in v)
    return v


def blink(T, seed=0, period=3.1, dur=0.13):
    """True during a brief blink (deterministic per seed)."""
    ph = (T + 1.7 * seed) % period
    return ph < dur


class Sprite:
    """A small mosaic picture (figure, prop, titulus) laid in its own fixed tile layout, in LOCAL px units
    (canvas w x h, ppu = cartoon px per local unit). draw_fn(ctx, layer, **state) paints the cartoon
    (layer 'color' | 'gold' | 'fine'); `poses` = list of state dicts whose union silhouette gets tiles.
    Static sprites keep one set of tile colours; dynamic ones re-colour per frame via colours(**state)."""

    def __init__(self, w, h, draw_fn, tile=14.0, fine_tile=8.0, seed=0, poses=({},), ppu=1.0, anchor=(0.0, 0.0),
                 min_alpha=0.45):
        from vx.canvas import Canvas
        w, h = float(round(w)), float(round(h))              # layouts round their extent to whole units
        self.w, self.h, self.ppu, self.draw_fn, self.anchor = w, h, ppu, draw_fn, anchor

        class _F:
            s = ppu
        cw, chh = int(round(w * ppu)), int(round(h * ppu))
        acc = np.zeros((chh, cw, 4), np.float32)
        fine = np.zeros((chh, cw), np.float32)
        for st in poses:
            cv = Canvas(s=ppu, w=cw, h=chh)
            draw_fn(cv.ctx, "color", **st)
            acc = np.maximum(acc, cv.rgba())
            fv = Canvas(s=ppu, w=cw, h=chh)
            draw_fn(fv.ctx, "fine", **st)
            fine = np.maximum(fine, fv.rgba()[..., 3])
        guide = acc[..., :3] + (1 - acc[..., 3:4]) * np.array([0.2, 0.2, 0.25], np.float32)
        lay = mz.Layout.flow(guide, tile=tile, seed=seed, extent=(w, h), fine=fine > 0.5, fine_tile=fine_tile)
        lab = mz.raster(lay, ppu)[0]
        _, cov, _ = mz.tile_colours(acc[..., :3], lab, len(lay), alpha=acc[..., 3])
        inside = cov > min_alpha * 0.6
        lay = lay.keep(lambda x, y: inside)
        self.lay = lay
        self.lab = mz.raster(lay, ppu)[0]
        self.n = len(lay)
        self.xy = lay.xy.astype(np.float64)
        self.ang = lay.ang.astype(np.float64)
        self.size = lay.size.astype(np.float64)
        self.tilt, self.jit = lay.tilt, lay.jit
        self.uid = np.arange(self.n, dtype=np.int64) + seed * 100003
        self.min_alpha = min_alpha
        self._cache = {}
        self.static = self.colours(**(poses[0] if poses else {}))

    def canvases(self, **state):
        from vx.canvas import Canvas
        cw, chh = int(round(self.w * self.ppu)), int(round(self.h * self.ppu))
        cv = Canvas(s=self.ppu, w=cw, h=chh)
        self.draw_fn(cv.ctx, "color", **state)
        gv = Canvas(s=self.ppu, w=cw, h=chh)
        self.draw_fn(gv.ctx, "gold", **state)
        return cv.rgba(), gv.rgba()[..., 3]

    def colours(self, **state):
        """-> dict(rgb (n,3), gold (n,), alpha (n,), pyr) for the tiles + a sampler for quadtree children.
        Cached per quantised state (render workers get contiguous frames: idle figures cost nothing)."""
        key = tuple(sorted((k, _quant(v)) for k, v in state.items()))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        qstate = {k: _quant(v) for k, v in state.items()}
        rgba, gold = self.canvases(**qstate)
        a = rgba[..., 3]
        rgb = rgba[..., :3] / np.maximum(a[..., None], 1e-4)
        tc, ta, _ = mz.tile_colours(rgb, self.lab, self.n, alpha=a)
        tg, _, _ = mz.tile_colours(np.dstack([gold, gold, gold]), self.lab, self.n)
        pyr = Pyramid(np.dstack([rgb, gold]), 0.0, 0.0, self.ppu)
        out = dict(rgb=tc.astype(np.float32), gold=tg[:, 0].astype(np.float32),
                   alpha=np.clip((ta - self.min_alpha * 0.6) / 0.25, 0, 1).astype(np.float32), pyr=pyr)
        if len(self._cache) > 48:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = out
        return out

    def colour_fn(self, col):
        """colour_fn for render_set: level-0 tiles use the tile means, children sample the pyramid."""
        def f(lxy, lsize, pidx):
            rgb = col["rgb"][pidx].copy()
            gold = col["gold"][pidx].copy()
            ch = lsize < self.size[pidx] * 0.99
            if ch.any():
                r2, g2 = col["pyr"].sample(lxy[ch, 0], lxy[ch, 1], lsize[ch])
                rgb[ch] = r2
                gold[ch] = g2
            return rgb, gold
        return f

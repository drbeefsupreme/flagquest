"""vx.mosaic_lay - ANDAMENTO: fixed tesserae layouts (owner: Tessellator). Internal; use vx.mosaic.Layout.

flow_layout = how a Byzantine mosaicist sets a panel:
  1. contour lines of the cartoon (colour boundaries, explicit edges, mask borders) are the features;
  2. a distance field from them, normalised by the local tile size (fine tiles in faces/hands/letters, or a
     surface density so tesserae keep a constant physical size on domes/conches) gives the COURSES: level lines
     at 0.5, 1.5, 2.5 ... tile sizes. Course 0 hugs every contour on both sides (the dark outline course of a
     figure, the halo course around it in the ground), 1, 2, ... echo outward like ripples;
  3. thin regions (strokes narrower than a tile) get one course of narrower tiles down their middle;
  4. the ground beyond `echo` courses is set in straight rows (or fan/rings, or keeps echoing);
  5. each course is cut into tesserae of slightly varying length with hand-cut irregular corners; where courses
     collide (medial axes, ground meeting halo) tiles are nipped to fit (vx.mosaic_geo.place_batch);
  6. remaining gaps are filled with smaller pieces.
"""
import hashlib
import math
import os

import cv2
import numpy as np
from scipy import ndimage

from .config import W, H, CACHE
from . import mosaic_geo as G

KMAX = G.KMAX
VERSION = 5


def _joint(t):
    return np.clip(0.09 * np.asarray(t, np.float64), 0.45, 2.2)


class _Placer:
    def __init__(self, extent, tmax, cap):
        self.cell = 1.7 * tmax + 3.0
        ncx = int(extent[0] / self.cell) + 4
        ncy = int(extent[1] / self.cell) + 4
        self.grid = np.array([-2 * self.cell, -2 * self.cell, self.cell, ncx, ncy], np.float64)
        self.cap = int(cap)
        self.polys = np.zeros((self.cap, KMAX, 2), np.float64)
        self.nvs = np.zeros(self.cap, np.int32)
        self.cen = np.zeros((self.cap, 2), np.float64)
        self.rad = np.zeros(self.cap, np.float64)
        self.info = np.zeros((self.cap, 6), np.float64)
        self.head = -np.ones(ncx * ncy, np.int64)
        self.nxt = -np.ones(self.cap, np.int64)
        self.cnt = np.zeros(1, np.int64)
        self.rmax = np.zeros(1, np.float64)

    def _grow(self):
        n = self.cap * 2
        for k in ("polys", "nvs", "cen", "rad", "info"):
            a = getattr(self, k)
            b = np.zeros((n,) + a.shape[1:], a.dtype)
            b[:self.cap] = a
            setattr(self, k, b)
        nx = -np.ones(n, np.int64)
        nx[:self.cap] = self.nxt
        self.nxt = nx
        self.cap = n

    def place(self, C, info, minfrac=0.42, minw=0.28, joint=1.0):
        if len(C) == 0:
            return 0
        while self.cnt[0] + len(C) >= self.cap:
            self._grow()
        cn = np.full(len(C), C.shape[1], np.int32)
        CC = np.zeros((len(C), KMAX + 4, 2), np.float64)
        CC[:, :C.shape[1]] = C
        return G.place_batch(CC, cn, info.astype(np.float64), float(minfrac), float(minw), float(joint),
                             self.polys, self.nvs, self.cen, self.rad, self.info, self.head, self.nxt, self.cnt,
                             self.grid, self.rmax)

    @property
    def n(self):
        return int(self.cnt[0])


# ============================================================ chains
def _resample(p, closed, step):
    q = np.vstack([p, p[:1]]) if closed else p
    seg = np.sqrt(((q[1:] - q[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return None
    n = max(int(total / step), 4)
    t = np.arange(n) * (total / n) if closed else np.linspace(0, total, n)
    return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)


def _normals(p, closed):
    if closed:
        d = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    else:
        d = np.gradient(p, axis=0)
    L = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    d = d / L
    return np.stack([-d[:, 1], d[:, 0]], 1)


def _chain_tiles(pl, pts, closed, Tf, Hf, kw, level, chain, jscale, kind, lenvar, irr, seed, step):
    """pts (M,2) design units (already smoothed). Resample, cut into quads, place."""
    p = _resample(pts, closed, step)
    if p is None or len(p) < 4:
        return 0
    nrm = _normals(p, closed)
    ix = np.clip((p[:, 0] * kw).astype(np.int32), 0, Tf.shape[1] - 1)
    iy = np.clip((p[:, 1] * kw).astype(np.int32), 0, Tf.shape[0] - 1)
    Tp = Tf[iy, ix].astype(np.float64)
    Hp = Hf[iy, ix].astype(np.float64)
    total = float(np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1)).sum())
    cap = int(total / max(0.4 * Tp.min(), 0.3)) + 8
    out = np.zeros((cap, 4, 2), np.float64)
    oinfo = np.zeros((cap, 6), np.float64)
    c = G.chain_quads(p.astype(np.float64), nrm.astype(np.float64), Tp, Hp, bool(closed), float(jscale),
                      float(lenvar), float(irr), int(seed) % (2 ** 31 - 1), out, oinfo, float(level), float(chain),
                      float(kind))
    return pl.place(out[:c], oinfo[:c])


def _contour_chains(mask, phi, c, kw, smooth_pts):
    """level curves phi == c as lists of (points design units, closed)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    gh, gw = mask.shape
    chains = []
    for cnt in cs:
        q = cnt[:, 0, :].astype(np.float64)
        if len(q) < 4:
            continue
        border = (q[:, 0] <= 0) | (q[:, 1] <= 0) | (q[:, 0] >= gw - 1) | (q[:, 1] >= gh - 1)
        if border.all():
            continue
        if border.any():
            # split into open runs of non-border points
            idx = np.nonzero(~border)[0]
            k = np.nonzero(border)[0][0]
            order = np.roll(np.arange(len(q)), -k)
            b = border[order]
            runs = []
            cur = []
            for j, bb in zip(order, b):
                if bb:
                    if len(cur) >= 4:
                        runs.append(cur)
                    cur = []
                else:
                    cur.append(j)
            if len(cur) >= 4:
                runs.append(cur)
            for r in runs:
                chains.append((q[r], False))
        else:
            chains.append((q, True))
    out = []
    if not chains:
        return out
    # sub-pixel refinement: newton steps toward phi == c (bilinear)
    allp = np.vstack([ch[0] for ch in chains])
    gy, gx = np.gradient(phi)
    for _ in range(2):
        coords = [allp[:, 1], allp[:, 0]]
        f = ndimage.map_coordinates(phi, coords, order=1, mode="nearest")
        ax = ndimage.map_coordinates(gx, coords, order=1, mode="nearest")
        ay = ndimage.map_coordinates(gy, coords, order=1, mode="nearest")
        g2 = ax * ax + ay * ay
        stp = np.where(g2 > 1e-8, (c - f) / np.maximum(g2, 1e-8), 0.0)
        stp = np.clip(stp, -1.2 / np.sqrt(np.maximum(g2, 1e-8)), 1.2 / np.sqrt(np.maximum(g2, 1e-8)))
        allp[:, 0] += stp * ax
        allp[:, 1] += stp * ay
    i = 0
    for q, closed in chains:
        n = len(q)
        p = allp[i:i + n]
        i += n
        sp = smooth_pts
        if sp > 0.3:
            mode = "wrap" if closed else "nearest"
            p = np.stack([ndimage.gaussian_filter1d(p[:, 0], sp, mode=mode),
                          ndimage.gaussian_filter1d(p[:, 1], sp, mode=mode)], 1)
        out.append(((p + 0.5) / kw, closed))
    return out


def _chain_len(p, closed):
    d = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1)).sum()
    if closed:
        d += math.hypot(*(p[0] - p[-1]))
    return d


# ============================================================ simple builders
def _finish(pl, extent, seed, drop_outside=True, from_layout=None):
    from .mosaic import Layout
    n = pl.n
    P = pl.polys[:n].astype(np.float32)
    nv = pl.nvs[:n].astype(np.int8)
    info = pl.info[:n]
    cen = pl.cen[:n].astype(np.float32)
    keep = np.ones(n, bool)
    if drop_outside:
        keep = (cen[:, 0] >= -1) & (cen[:, 0] <= extent[0] + 1) & (cen[:, 1] >= -1) & (cen[:, 1] <= extent[1] + 1)
    kind = info[:, 5]
    lev = np.where(kind == 0, info[:, 2], np.where(kind == 1, -1, -2)).astype(np.int16)
    lay = Layout(cen[keep], info[keep, 1].astype(np.float32), info[keep, 0].astype(np.float32), seed, extent,
                 poly=P[keep], nv=nv[keep], lev=lev[keep], chain=info[keep, 3].astype(np.int32))
    return lay


def rows_layout(tile=14.0, rect=None, ang=0.0, jitter=0.12, seed=0, extent=(W, H), brick=True):
    x0, y0, w, h = rect or (0, 0, extent[0], extent[1])
    cx, cy = x0 + w / 2, y0 + h / 2
    R = math.hypot(w, h) / 2 + tile
    ca, sa = math.cos(ang), math.sin(ang)
    pl = _Placer(extent, tile * 1.3, (w * h) / (tile * tile) * 1.3 + 200)
    rng = np.random.default_rng(seed)
    nrow = int(2 * R / tile) + 1
    Hh = tile - _joint(tile)
    for j in range(nrow):
        yl = -R + (j + 0.5) * tile
        xs = np.arange(-R, R, 0.25 * tile)
        p = np.stack([cx + ca * xs - sa * yl, cy + sa * xs + ca * yl], 1)
        inside = (p[:, 0] > x0 - tile) & (p[:, 0] < x0 + w + tile) & (p[:, 1] > y0 - tile) & (p[:, 1] < y0 + h + tile)
        if inside.sum() < 4:
            continue
        p = p[inside]
        Tp = np.full(len(p), float(tile))
        Hp = np.full(len(p), float(Hh))
        nrm = np.broadcast_to(np.array([-sa, ca]), p.shape).copy()
        cap = int(len(p) * 0.25 / 0.4) + 8
        out = np.zeros((cap, 4, 2))
        oinfo = np.zeros((cap, 6))
        c = G.chain_quads(p, nrm, Tp, Hp, False, 1.0, 0.18, max(jitter, 0.0) * 0.35, int(rng.integers(1 << 30)),
                          out, oinfo, -1.0, float(j), 1.0)
        pl.place(out[:c], oinfo[:c])
    lay = _finish(pl, extent, seed)
    k = (lay.xy[:, 0] >= x0) & (lay.xy[:, 0] <= x0 + w) & (lay.xy[:, 1] >= y0) & (lay.xy[:, 1] <= y0 + h)
    return lay.subset(k)


def rings_layout(cx, cy, r0, r1, tile=12.0, seed=0, extent=(W, H)):
    pl = _Placer(extent, tile * 1.3, math.pi * (r1 * r1 - r0 * r0) / (tile * tile) * 1.3 + 200)
    rng = np.random.default_rng(seed)
    r = max(r0 + tile * 0.5, tile * 0.5)
    j = 0
    Hh = tile - _joint(tile)
    while r <= r1:
        n = max(int(2 * math.pi * r / (0.2 * tile)), 12)
        th = np.arange(n) * 2 * math.pi / n + rng.uniform(0, 2 * math.pi)
        p = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1)
        nrm = np.stack([np.cos(th), np.sin(th)], 1)
        Tp = np.full(n, float(tile))
        Hp = np.full(n, float(Hh))
        cap = int(2 * math.pi * r / (0.4 * tile)) + 8
        out = np.zeros((cap, 4, 2))
        oinfo = np.zeros((cap, 6))
        c = G.chain_quads(p, nrm, Tp, Hp, True, 1.0, 0.16, 0.04, int(rng.integers(1 << 30)), out, oinfo, float(j),
                          float(j), 0.0)
        pl.place(out[:c], oinfo[:c])
        r += tile
        j += 1
    return _finish(pl, extent, seed, drop_outside=False)


# ============================================================ flow (andamento)
def _to_float_img(a):
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    return a.astype(np.float32)


def _mask_at(m, gw, gh):
    if m is None:
        return None
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[..., -1] if m.shape[2] in (2, 4) else m[..., 0]
    m = m.astype(np.float32)
    if m.shape != (gh, gw):
        m = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA if m.shape[1] > gw else cv2.INTER_LINEAR)
    return m >= 0.5


def _mask_border(m):
    er = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
    return m & (er == 0)


def _colour_features(img, thr=13):
    q = (np.clip(img[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
    lab = cv2.cvtColor(q, cv2.COLOR_RGB2LAB).astype(np.int16)
    F = np.zeros(q.shape[:2], bool)
    dx = np.abs(np.diff(lab, axis=1)).sum(-1)
    dy = np.abs(np.diff(lab, axis=0)).sum(-1)
    F[:, :-1] |= dx > thr
    F[:-1, :] |= dy > thr
    return F


def _hash_inputs(*arrs, **params):
    h = hashlib.sha1()
    h.update(str(VERSION).encode())
    for a in arrs:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode() + str(a.dtype).encode())
        h.update(a.tobytes())
    h.update(repr(sorted(params.items())).encode())
    return h.hexdigest()[:20]


def flow_layout(guide, tile=12.0, seed=0, extent=None, edges=None, fine=None, fine_tile=None, rows=True,
                gold=None, silver=None, pearl=None, stone=None, ground=None, ground_style="rows", echo=2,
                ground_angle=0.0, fan_center=None, density=None, alpha=None, lenvar=0.22, irregular=0.05,
                joint=1.0, groups=None, cache=True):
    from .mosaic import Layout, GOLD, SILVER, PEARL, STONE
    if isinstance(guide, dict):          # an icon.cartoon() dict
        d = guide
        guide = d.get("rgb", d.get("rgba"))
        extent = extent or d.get("extent")
        fine = fine if fine is not None else d.get("fine")
        edges = edges if edges is not None else d.get("edges")
        gold = gold if gold is not None else d.get("gold")
        silver = silver if silver is not None else d.get("silver")
        pearl = pearl if pearl is not None else d.get("pearl")
        ground = ground if ground is not None else d.get("ground")
        # NOTE: the dict's alpha is NOT applied automatically (a cartoon painted over a bg must be tiled
        # everywhere); pass alpha=d['alpha'] explicitly to lay tiles only inside the figures (moving layers).
    g = np.asarray(guide)
    extent = tuple(extent) if extent is not None else (W, H)
    fine_tile = fine_tile or tile * 0.55
    key = None
    if cache:
        dens_sig = None
        if density is not None:
            ys, xs = np.meshgrid(np.linspace(0, extent[1], 17), np.linspace(0, extent[0], 17), indexing="ij")
            dens_sig = np.asarray(density(xs, ys), np.float32)
        key = _hash_inputs(g if g.size < 4e6 else g[::2, ::2], None if edges is None else np.asarray(edges),
                           None if fine is None else np.asarray(fine), None if gold is None else np.asarray(gold),
                           None if silver is None else np.asarray(silver), None if pearl is None else np.asarray(pearl),
                           None if stone is None else np.asarray(stone), None if ground is None else np.asarray(ground),
                           None if alpha is None else np.asarray(alpha), None if groups is None else np.asarray(groups),
                           dens_sig, tile=tile, seed=seed, extent=extent, fine_tile=fine_tile, rows=rows,
                           style=ground_style, echo=echo, gang=ground_angle, fan=fan_center, lenvar=lenvar,
                           irr=irregular, joint=joint)
        path = CACHE / "mosaic" / f"flow_{key}.npz"
        if path.exists():
            try:
                return Layout.load(path)
            except Exception:
                pass
    tmin = min(tile, fine_tile) if fine is not None else tile
    if density is not None:
        ys, xs = np.meshgrid(np.linspace(0, extent[1], 33), np.linspace(0, extent[0], 33), indexing="ij")
        kmax_d = float(np.max(density(xs, ys)))
        tmin = min(tmin, tile / max(kmax_d, 1e-3))
    kw = float(np.clip(5.5 / tmin, 0.25, 2.0))
    gw, gh = int(round(extent[0] * kw)), int(round(extent[1] * kw))
    # ---- guide at working resolution
    labels = None
    if g.ndim == 2 and np.issubdtype(g.dtype, np.integer):
        labels = cv2.resize(g.astype(np.int32), (gw, gh), interpolation=cv2.INTER_NEAREST)
        img = None
    else:
        img = _to_float_img(g)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, 2)
        if img.shape[2] == 4 and alpha is None:
            alpha = img[..., 3]
            img = img[..., :3] / np.maximum(img[..., 3:4], 1e-4)
        img = img[..., :3]
        interp = cv2.INTER_AREA if img.shape[1] > gw else cv2.INTER_LINEAR
        img = cv2.resize(img, (gw, gh), interpolation=interp)
    valid = _mask_at(alpha, gw, gh) if alpha is not None else np.ones((gh, gw), bool)
    fine_m = _mask_at(fine, gw, gh)
    ground_m = _mask_at(ground, gw, gh)
    # ---- features
    if labels is not None:
        F = np.zeros((gh, gw), bool)
        F[:, :-1] |= labels[:, 1:] != labels[:, :-1]
        F[:-1, :] |= labels[1:, :] != labels[:-1, :]
    else:
        F = _colour_features(img)
    if edges is not None:
        em = np.asarray(edges)
        if em.ndim == 3:
            em = em[..., -1]
        em = cv2.resize(em.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) > 0.25
        if em.mean() > 0.3:
            import warnings
            warnings.warn(f"Layout.flow: edges= covers {em.mean():.0%} of the extent - it should be thin contour lines "
                          "(inverted mask?)", stacklevel=3)
        F |= em
    for m in (fine_m, ground_m, _mask_at(gold, gw, gh), _mask_at(silver, gw, gh)):
        if m is not None and m.any() and not m.all():
            F |= _mask_border(m)
    if not valid.all():
        F |= _mask_border(valid) | ~valid
    # ---- tile size field (design units)
    Tf = np.full((gh, gw), float(tile), np.float32)
    if fine_m is not None:
        Tf[fine_m] = float(fine_tile)
    if density is not None:
        ys, xs = np.mgrid[0:gh, 0:gw].astype(np.float32)
        k = np.asarray(density((xs + 0.5) / kw, (ys + 0.5) / kw), np.float32)
        Tf = Tf / np.maximum(k, 1e-3)
    # ---- distance + normalised field
    D = cv2.distanceTransform((~F).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / kw
    nlab, lab = cv2.connectedComponents((~F).astype(np.uint8), connectivity=4)
    Dmax = ndimage.maximum(D, lab, index=np.arange(nlab)).astype(np.float32)
    Dm = Dmax[lab]
    phi = (D / Tf).astype(np.float32)
    ratio = Dm / Tf
    thin = (ratio < 0.6) & ~F
    phi[thin] = (D[thin] / np.maximum(1.6 * Dm[thin], 1e-4)).astype(np.float32)
    Jf = _joint(Tf) * joint
    Hf = (Tf - Jf).astype(np.float32)
    Hf[thin] = np.clip(2 * Dm[thin] - Jf[thin], 0.3 * Tf[thin], Tf[thin] - Jf[thin])
    phi[~valid] = 0
    phi_full = phi.copy()
    if ground_m is not None and ground_style != "echo":
        gm = ground_m & valid
        phi[gm] = np.minimum(phi[gm], float(echo))
    # ---- place courses
    area_tiles = float((valid / (Tf * Tf)).sum()) / (kw * kw)
    pl = _Placer(extent, float(Tf.max()) * 1.25, area_tiles * 1.35 + 1000)
    kmax = int(math.ceil(float(phi.max())))
    rs = np.random.default_rng(seed)
    step = max(0.12 * float(Tf.min()), 0.2)
    chain_id = 0
    irr = irregular
    for kk in range(kmax):
        c = kk + 0.5
        mask = phi >= c
        if not mask.any():
            break
        chains = _contour_chains(mask, phi, c, kw, smooth_pts=max(0.8, 0.3 * tmin * kw))
        chains.sort(key=lambda ch: -_chain_len(ch[0], ch[1]))
        for p, closed in chains:
            if _chain_len(p, closed) < 0.6 * tmin:
                continue
            _chain_tiles(pl, p, closed, Tf, Hf, kw, kk, chain_id, joint, 0, lenvar, irr, rs.integers(1 << 30), step)
            chain_id += 1
    # ---- ground style
    if ground_m is not None and ground_style in ("rows", "fan"):
        zone = ground_m & valid & (phi_full >= echo - 0.05)
        if zone.any():
            zone_d = cv2.dilate(zone.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            tg = float(np.median(Tf[zone]))
            if ground_style == "rows":
                a = ground_angle
                ca, sa = math.cos(a), math.sin(a)
                cx, cy = extent[0] / 2, extent[1] / 2
                R = math.hypot(*extent) / 2 + tg
                xs = np.arange(-R, R, step)
                for j in range(int(2 * R / tg) + 1):
                    yl = -R + (j + 0.5) * tg
                    p = np.stack([cx + ca * xs - sa * yl, cy + sa * xs + ca * yl], 1)
                    _runs(pl, p, zone_d, kw, Tf, Hf, j, chain_id, joint, lenvar, irr, rs, step, a)
                    chain_id += 1
            else:
                fx, fy = fan_center if fan_center is not None else (extent[0] / 2, extent[1])
                rmaxf = max(math.hypot(fx - x, fy - y) for x in (0, extent[0]) for y in (0, extent[1]))
                j = 0
                r = 0.5 * tg
                while r < rmaxf:
                    n = max(int(2 * math.pi * r / step), 16)
                    th = np.arange(n) * 2 * math.pi / n
                    p = np.stack([fx + r * np.cos(th), fy + r * np.sin(th)], 1)
                    _runs(pl, p, zone_d, kw, Tf, Hf, j, chain_id, joint, lenvar, irr, rs, step, None, closed_ok=True)
                    chain_id += 1
                    r += tg
                    j += 1
    # ---- gap fill
    _fill_gaps(pl, valid, Tf, phi_full, kw, joint, rs, rounds=3)
    lay = _finish(pl, extent, seed)
    # ---- materials / groups
    from .mosaic import sample_mask
    for code, m in ((GOLD, gold), (SILVER, silver), (PEARL, pearl), (STONE, stone)):
        if m is not None:
            lay.mat[sample_mask(m, lay, extent=extent) >= 0.5] = code
    if groups is not None:
        gr = np.asarray(groups)
        iy = np.clip((lay.xy[:, 1] * gr.shape[0] / extent[1]).astype(int), 0, gr.shape[0] - 1)
        ix = np.clip((lay.xy[:, 0] * gr.shape[1] / extent[0]).astype(int), 0, gr.shape[1] - 1)
        lay.grp = gr[iy, ix].astype(np.int32)
    if key is not None:
        try:
            (CACHE / "mosaic").mkdir(parents=True, exist_ok=True)
            tmp = CACHE / "mosaic" / f"flow_{key}.{os.getpid()}.tmp.npz"
            lay.save(tmp)
            tmp.replace(CACHE / "mosaic" / f"flow_{key}.npz")
        except OSError:
            pass
    return lay


def _runs(pl, p, zone, kw, Tf, Hf, j, chain_id, joint, lenvar, irr, rs, step, ang, closed_ok=False):
    gh, gw = zone.shape
    ix = (p[:, 0] * kw).astype(np.int64)
    iy = (p[:, 1] * kw).astype(np.int64)
    inb = (ix >= 0) & (iy >= 0) & (ix < gw) & (iy < gh)
    ok = np.zeros(len(p), bool)
    ok[inb] = zone[iy[inb], ix[inb]]
    if not ok.any():
        return
    if closed_ok and ok.all():
        _chain_tiles(pl, p, True, Tf, Hf, kw, -1, chain_id, joint, 1, lenvar, irr, rs.integers(1 << 30), step)
        return
    d = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    for a, b in zip(starts, ends):
        # extend each run a little so the end tiles reach into the joint with the halo course (then get nipped)
        a2 = max(a - 3, 0)
        b2 = min(b + 3, len(p))
        if b2 - a2 < 4:
            continue
        _chain_tiles(pl, p[a2:b2], False, Tf, Hf, kw, -1, chain_id, joint, 1, lenvar, irr,
                     rs.integers(1 << 30), step)


def _fill_gaps(pl, valid, Tf, phi, kw, joint, rs, rounds=4):
    gh, gw = valid.shape
    gy, gx = np.gradient(phi)
    for r in range(rounds):
        cover = np.zeros((gh, gw), np.uint8)
        n = pl.n
        if n:
            G.raster_cover(cover, pl.polys, pl.nvs, n, kw, 0.5)
        free = (cover == 0) & valid
        if not free.any():
            break
        dt = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / kw
        mx = ndimage.maximum_filter(dt, size=3)
        cand = (dt >= mx - 1e-6) & (dt >= 0.11 * Tf) & free
        # large empty areas: also seed a lattice so they are paved in one round, not only along their medial axes
        big = dt >= 0.9 * Tf
        if big.any():
            step = max(1, int(round(float(np.median(Tf[big])) * kw)))
            lat = np.zeros_like(big)
            lat[::step, ::step] = True
            cand |= big & lat
        ys, xs = np.nonzero(cand)
        if len(ys) == 0:
            break
        order = np.argsort(-dt[ys, xs], kind="stable")
        ys, xs = ys[order], xs[order]
        T = Tf[ys, xs].astype(np.float64)
        side = np.minimum(2.0 * dt[ys, xs] - 0.2, T - _joint(T))
        ok = side >= (0.3 if r < 2 else 0.18) * T          # later rounds: small chips for the last gaps
        ys, xs, T, side = ys[ok], xs[ok], T[ok], side[ok]
        if len(ys) == 0:
            break
        a = np.arctan2(gy[ys, xs], gx[ys, xs]) + np.pi / 2
        a += rs.normal(0, 0.05, len(a))
        cx = (xs + 0.5) / kw
        cy = (ys + 0.5) / kw
        ca, sa = np.cos(a), np.sin(a)
        h = side / 2
        loc = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
        C = np.empty((len(ys), 4, 2))
        for k in range(4):
            lx, ly = loc[k, 0] * h, loc[k, 1] * h
            C[:, k, 0] = cx + ca * lx - sa * ly
            C[:, k, 1] = cy + sa * lx + ca * ly
        C += rs.normal(0, 0.03, C.shape) * T[:, None, None]
        info = np.stack([T, a, np.full(len(T), -2.0), np.full(len(T), -1.0), np.full(len(T), joint),
                         np.full(len(T), 2.0)], 1)
        pl.place(C, info, minfrac=0.5 if r < 2 else 0.35, minw=0.25 if r < 2 else 0.14)

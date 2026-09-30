"""m02 wall engine (owner M02): one colossal world-space mosaic seen through a moving camera.

The nave wall is a fixed set of tesserae in WORLD units (the wide view of the whole picture frames world
[0,1920]x[0,1080] at zoom 1). Every frame, a numba kernel maps each screen pixel into the world through the
camera (centre cx, cy, zoom z), finds the tessera under it in a uniform grid, and shades it (bevel, glint on the
lit rim, mortar joints with analytic anti-aliasing, split-flap foreshortening for flipping tiles, glowing
'screen' tiles with a scrolling feed). Tiles never move between frames except:
  * rotating groups (gilded cogs): every tile of group k turns rigidly about a disc centre; the kernel rotates
    the query point instead of the tiles (the disc is exclusive to its group);
  * flipping tiles: VS (0..1) foreshortens a tile about its own horizontal axis.
Colours are recomputed per frame outside the kernel (tiles re-colour = figures move, light wakes, slop).
"""
import math

import cv2
import numpy as np
from numba import njit

FILL = 1.27          # a tile claims pixels out to FILL half-sizes (max-norm) before the mortar bed shows


# ============================================================ layout builders
@njit(cache=True)
def _place(dist, gx, gy, ts, k, x0, y0, seed, rows, smax):
    """Place square tesserae in courses parallel to the edges encoded by `dist` (distance to the nearest edge,
    world units). ts: tile size (world units) per guide pixel, <=0 = no tiles. k: guide px per world unit;
    (x0, y0): world position of the guide's top-left corner. Returns xy, ang, size (world)."""
    gh, gw = dist.shape
    np.random.seed(seed)
    cnt = 0
    for y in range(gh):
        for x in range(gw):
            s = ts[y, x]
            if s > 0:
                st = max(1, int(s * k / 4.0))
                if x % st == 0 and y % st == 0:
                    cnt += 1
    cyy = np.empty(cnt, np.int32)
    cxx = np.empty(cnt, np.int32)
    i = 0
    for y in range(gh):
        for x in range(gw):
            s = ts[y, x]
            if s > 0:
                st = max(1, int(s * k / 4.0))
                if x % st == 0 and y % st == 0:
                    cyy[i] = y
                    cxx[i] = x
                    i += 1
    order = np.random.permutation(cnt)
    cell = smax * 1.02
    ncx = int(gw / k / cell) + 3
    ncy = int(gh / k / cell) + 3
    head = -np.ones(ncx * ncy, np.int64)
    nxt = -np.ones(cnt, np.int64)
    oxy = np.zeros((cnt, 2), np.float32)
    oang = np.zeros(cnt, np.float32)
    osz = np.zeros(cnt, np.float32)
    m = 0
    for oi in range(cnt):
        y = cyy[order[oi]]
        x = cxx[order[oi]]
        s = ts[y, x]
        d = dist[y, x]
        if rows:
            f = d / s - math.floor(d / s)
            if abs(f - 0.5) > 0.2 and d > 0.3 * s:
                continue
        px = (x + 0.5) / k + (np.random.random() - 0.5) * 0.5 / k
        py = (y + 0.5) / k + (np.random.random() - 0.5) * 0.5 / k
        ccx = int(px / cell) + 1
        ccy = int(py / cell) + 1
        ok = True
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                qx, qy = ccx + dx, ccy + dy
                if qx < 0 or qy < 0 or qx >= ncx or qy >= ncy:
                    continue
                j = head[qy * ncx + qx]
                while j >= 0:
                    ddx = oxy[j, 0] - px
                    ddy = oxy[j, 1] - py
                    lim = 0.5 * (osz[j] + s) * 0.97
                    if ddx * ddx + ddy * ddy < lim * lim:
                        ok = False
                        break
                    j = nxt[j]
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        oxy[m, 0] = px
        oxy[m, 1] = py
        g0 = gx[y, x]
        g1 = gy[y, x]
        oang[m] = math.atan2(g1, g0) if (g0 != 0.0 or g1 != 0.0) else 0.0
        osz[m] = s * (0.9 + 0.1 * np.random.random())
        c = ccy * ncx + ccx
        nxt[m] = head[c]
        head[c] = m
        m += 1
    out = oxy[:m].copy()
    out[:, 0] += x0
    out[:, 1] += y0
    return out, oang[:m].copy(), osz[:m].copy()


def flow(edges, ts, k, x0, y0, seed=0, rows=True, blur=0.25):
    """andamento layout: tiles in courses echoing `edges` (bool guide image) with per-pixel tile size ts."""
    e = np.asarray(edges, bool)
    dist = cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 5) / k
    smin = float(np.min(ts[ts > 0])) if np.any(ts > 0) else 1.0
    dist = cv2.GaussianBlur(dist, (0, 0), max(0.7, blur * smin * k))
    gy, gx = np.gradient(dist)
    smax = float(ts.max())
    return _place(dist.astype(np.float32), gx.astype(np.float32), gy.astype(np.float32),
                  ts.astype(np.float32), float(k), float(x0), float(y0), int(seed), bool(rows), smax)


def rings(cx, cy, r0, r1, tile, seed=0, phase=None):
    """concentric courses (cogs, halos, domes). Returns xy, ang, size."""
    rng = np.random.default_rng(seed)
    pts, angs, szs = [], [], []
    r = max(r0, tile * 0.5)
    while r <= r1 + 1e-6:
        n = max(6, int(round(2 * math.pi * r / tile)))
        th = np.arange(n) * 2 * math.pi / n + (rng.uniform(0, 2 * math.pi) if phase is None else phase)
        pts.append(np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1))
        angs.append(th + math.pi / 2)
        szs.append(np.full(n, min(tile, 2 * math.pi * r / n) * 0.96))
        r += tile
    return (np.vstack(pts).astype(np.float32), np.concatenate(angs).astype(np.float32),
            np.concatenate(szs).astype(np.float32))


# ============================================================ grid
@njit(cache=True)
def _grid_build(X, Y, HS, gx0, gy0, cs, ncx, ncy):
    n = X.shape[0]
    counts = np.zeros(ncx * ncy + 1, np.int64)
    for i in range(n):
        r = HS[i] * FILL * 1.4143 * 1.05 + 0.02
        ix0 = max(0, int((X[i] - r - gx0) / cs))
        ix1 = min(ncx - 1, int((X[i] + r - gx0) / cs))
        iy0 = max(0, int((Y[i] - r - gy0) / cs))
        iy1 = min(ncy - 1, int((Y[i] + r - gy0) / cs))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                counts[iy * ncx + ix + 1] += 1
    for c in range(ncx * ncy):
        counts[c + 1] += counts[c]
    items = np.empty(counts[ncx * ncy], np.int32)
    fill = counts[:ncx * ncy].copy()
    for i in range(n):
        r = HS[i] * FILL * 1.4143 * 1.05 + 0.02
        ix0 = max(0, int((X[i] - r - gx0) / cs))
        ix1 = min(ncx - 1, int((X[i] + r - gx0) / cs))
        iy0 = max(0, int((Y[i] - r - gy0) / cs))
        iy1 = min(ncy - 1, int((Y[i] + r - gy0) / cs))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                c = iy * ncx + ix
                items[fill[c]] = i
                fill[c] += 1
    return counts, items


@njit(cache=True)
def _disc_cells(dcx, dcy, dr, gx0, gy0, cs, ncx, ncy):
    c0 = -np.ones(ncx * ncy, np.int32)
    c1 = -np.ones(ncx * ncy, np.int32)
    for k in range(dcx.shape[0]):
        r = dr[k] + cs * 1.5
        ix0 = max(0, int((dcx[k] - r - gx0) / cs))
        ix1 = min(ncx - 1, int((dcx[k] + r - gx0) / cs))
        iy0 = max(0, int((dcy[k] - r - gy0) / cs))
        iy1 = min(ncy - 1, int((dcy[k] + r - gy0) / cs))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                c = iy * ncx + ix
                if c0[c] < 0:
                    c0[c] = k
                elif c1[c] < 0:
                    c1[c] = k
    return c0, c1


# ============================================================ the camera kernel
@njit(cache=True, fastmath=True)
def _render(h, w, s, cx, cy, z, gx0, gy0, cs, ncx, ncy, cstart, citems,
            X, Y, CA, SA, HS, VS, GRP, COL, LT, GL, EM, PH,
            dcx, dcy, dr, dca, dsa, cd0, cd1, lx, ly, mr, mg, mb, bgr, bgg, bgb, out, emis, lods):
    inv = 1.0 / (s * z)                      # world units per pixel
    lodk = 1.0 / lods                         # LOD is judged at the output pixel size (supersampling)
    eh, ew = emis.shape[0], emis.shape[1]
    for py in range(h):
        wy0 = ((py + 0.5) / s - 540.0) / z + cy
        for px in range(w):
            wx = ((px + 0.5) / s - 960.0) / z + cx
            wy = wy0
            fx = (wx - gx0) / cs
            fy = (wy - gy0) / cs
            if fx < 0 or fy < 0 or fx >= ncx or fy >= ncy:
                out[py, px, 0] = bgr
                out[py, px, 1] = bgg
                out[py, px, 2] = bgb
                continue
            c = int(fy) * ncx + int(fx)
            grp = 0
            llx = lx
            lly = ly
            for kk in range(2):
                k = cd0[c] if kk == 0 else cd1[c]
                if k < 0 or grp > 0:
                    continue
                ex = wx - dcx[k]
                ey = wy - dcy[k]
                if ex * ex + ey * ey < dr[k] * dr[k]:
                    grp = k + 1
                    ca = dca[k]
                    sa = dsa[k]
                    # rest-frame query point = rotate by -theta
                    wx = dcx[k] + ex * ca + ey * sa
                    wy = dcy[k] - ex * sa + ey * ca
                    llx = lx * ca + ly * sa
                    lly = -lx * sa + ly * ca
            if grp > 0:
                fx = (wx - gx0) / cs
                fy = (wy - gy0) / cs
                if fx < 0 or fy < 0 or fx >= ncx or fy >= ncy:
                    out[py, px, 0] = bgr
                    out[py, px, 1] = bgg
                    out[py, px, 2] = bgb
                    continue
                c = int(fy) * ncx + int(fx)
            best = 1e9
            second = 1e9
            bi = -1
            bu = 0.0
            bv = 0.0
            for q in range(cstart[c], cstart[c + 1]):
                j = citems[q]
                if GRP[j] != grp:
                    continue
                ex = wx - X[j]
                ey = wy - Y[j]
                u = ex * CA[j] + ey * SA[j]
                v = -ex * SA[j] + ey * CA[j]
                hj = HS[j]
                vj = VS[j]
                au = abs(u) / hj
                av = abs(v) / (hj * vj)
                d = au if au > av else av
                if d < best:
                    second = best
                    best = d
                    bi = j
                    bu = u / hj
                    bv = v / (hj * vj)
                elif d < second:
                    second = d
            if bi < 0:
                out[py, px, 0] = mr * 0.5
                out[py, px, 1] = mg * 0.5
                out[py, px, 2] = mb * 0.5
                continue
            lt = LT[bi]
            hsb = HS[bi]
            tpx = 2.0 * hsb / inv / lodk                      # tile size on screen (output px)
            lod = (5.0 - tpx) / 3.5
            lod = 0.0 if lod < 0.0 else (1.0 if lod > 1.0 else lod)
            m0 = FILL - best
            m1 = (second - best) * 0.5
            m = (m0 if m0 < m1 else m1) * hsb / inv          # px to the joint
            gpx = 0.17 * hsb / inv                            # joint width in px
            cov = m - gpx * 0.5 + 0.5
            cov = 0.0 if cov < 0.0 else (1.0 if cov > 1.0 else cov)
            cov = cov + (1.0 - cov) * lod * 0.8               # far away the joints close up (no darkening)
            r = COL[bi, 0]
            g = COL[bi, 1]
            b = COL[bi, 2]
            e = EM[bi]
            bev = m / (0.2 * hsb / inv + 0.6)
            rim = 0.0 if bev < 0.0 else (1.0 if bev > 1.0 else bev)
            if e > 0.0:
                # a glowing screen: dark bezel, the cards of a feed scrolling through it
                yv = bv * 0.5 + 0.5
                qq = yv * 2.2 + PH[bi]
                fr = qq - math.floor(qq)
                card = 1.0
                if fr < 0.10:
                    card = 0.42                               # gap between posts
                elif fr < 0.62:
                    if abs(bu) < 0.66:
                        # the post's picture: a pastel blob world
                        ib = math.sin(bu * 5.0 + qq * 7.0) * math.sin(fr * 13.0 + bu * 3.0)
                        card = 1.18 + 0.16 * ib
                    else:
                        card = 0.86
                else:
                    # text lines under the picture
                    tl = (fr - 0.62) * 14.0
                    card = 0.72 if (tl - math.floor(tl) < 0.45 and bu < 0.3 - 0.4 * math.floor(tl) * 0.3) \
                        else 0.95
                bez = 0.2 + 0.8 * (1.0 if rim > 0.55 else rim / 0.55)
                k2 = (1.0 - e) + e * card * bez
                sh = (0.72 + 0.28 * rim) * (1.0 - e) + e
                r = r * sh * k2
                g = g * sh * k2
                b = b * sh * k2
                if cov > 0.0:
                    ey2 = min(eh - 1, py >> 2)
                    ex2 = min(ew - 1, px >> 2)
                    wgt = e * cov * 0.0625
                    emis[ey2, ex2, 0] += r * wgt
                    emis[ey2, ex2, 1] += g * wgt
                    emis[ey2, ex2, 2] += b * wgt
            else:
                sh = 0.72 + 0.28 * rim
                sh = 1.0 - (1.0 - sh) * (1.0 - 0.75 * lod)
                lit = -(bu * llx + bv * lly) * 0.8
                lit = (0.0 if lit < 0.0 else (1.0 if lit > 1.0 else lit)) * (1.0 - rim)
                gl = GL[bi] * (0.35 + 0.65 * lit) * 0.9
                r = r * sh + gl
                g = g * sh + gl * 0.93
                b = b * sh + gl * 0.8
            mm = lt * 0.85
            out[py, px, 0] = mr * mm * (1.0 - cov) + r * cov
            out[py, px, 1] = mg * mm * (1.0 - cov) + g * cov
            out[py, px, 2] = mb * mm * (1.0 - cov) + b * cov


class Wall:
    """Tile set + grid + rotating discs. Arrays are world units (float32)."""

    def __init__(self, xy, ang, size, grp, discs, cs=1.6, seed=0, aspect=None):
        self.xy = np.ascontiguousarray(xy, np.float32)
        self.ang = np.ascontiguousarray(ang, np.float32)
        self.size = np.ascontiguousarray(size, np.float32)
        self.grp = np.ascontiguousarray(grp, np.int32)
        self.n = len(self.xy)
        self.X = np.ascontiguousarray(self.xy[:, 0])
        self.Y = np.ascontiguousarray(self.xy[:, 1])
        self.CA = np.cos(self.ang).astype(np.float32)
        self.SA = np.sin(self.ang).astype(np.float32)
        self.HS = (0.5 * self.size).astype(np.float32)
        self.AS = np.ones(self.n, np.float32) if aspect is None else np.ascontiguousarray(aspect, np.float32)
        lo = self.xy.min(0) - 8
        hi = self.xy.max(0) + 8
        self.gx0, self.gy0 = float(lo[0]), float(lo[1])
        self.cs = float(cs)
        self.ncx = int((hi[0] - lo[0]) / cs) + 1
        self.ncy = int((hi[1] - lo[1]) / cs) + 1
        self.cstart, self.citems = _grid_build(self.X, self.Y, (self.HS * np.maximum(self.AS, 1.0)).astype(
            np.float32), self.gx0, self.gy0, self.cs, self.ncx, self.ncy)
        d = np.asarray(discs, np.float32).reshape(-1, 3)      # (cx, cy, r) per rotating group k (grp = k+1)
        if len(d) == 0:
            d = np.array([[-1e6, -1e6, 1.0]], np.float32)
        self.dcx, self.dcy, self.dr = (np.ascontiguousarray(d[:, i]) for i in range(3))
        self.cd0, self.cd1 = _disc_cells(self.dcx, self.dcy, self.dr, self.gx0, self.gy0, self.cs,
                                         self.ncx, self.ncy)
        rng = np.random.default_rng(seed + 7919)
        self.tilt = rng.normal(0, 1, (self.n, 2)).astype(np.float32)
        self.jit = rng.normal(0, 1, (self.n, 3)).astype(np.float32)
        self.hash = rng.random(self.n).astype(np.float32)

    def render(self, fc, cam, col, light, glint, vs=None, em=None, ph=None, theta=None, ldir=(0.3, -0.5),
               mortar=(0.29, 0.26, 0.22), bg=(0.02, 0.02, 0.03), lod_scale=1.0):
        """cam = (cx, cy, z). col (N,3) lit tile colours; light (N,) for the joints; glint (N,) specular.
        vs (N,) flip foreshortening; em (N,) screen emission 0..1; ph (N,) feed scroll phase;
        theta (K,) rotation of each disc. Returns (rgb (h,w,3), emission (h/4,w/4,3))."""
        n = self.n
        vs = self.AS.copy() if vs is None else np.maximum(np.asarray(vs, np.float32), 0.04) * self.AS
        em = np.zeros(n, np.float32) if em is None else np.asarray(em, np.float32)
        ph = np.zeros(n, np.float32) if ph is None else np.asarray(ph, np.float32)
        K = len(self.dcx)
        th = np.zeros(K, np.float32) if theta is None else np.asarray(theta, np.float32)
        out = np.empty((fc.h, fc.w, 3), np.float32)
        emis = np.zeros(((fc.h + 3) // 4, (fc.w + 3) // 4, 3), np.float32)
        _render(fc.h, fc.w, float(fc.s), float(cam[0]), float(cam[1]), float(cam[2]),
                self.gx0, self.gy0, self.cs, self.ncx, self.ncy, self.cstart, self.citems,
                self.X, self.Y, self.CA, self.SA, self.HS, vs, self.grp,
                np.ascontiguousarray(col, np.float32), np.ascontiguousarray(light, np.float32),
                np.ascontiguousarray(glint, np.float32), em, ph,
                self.dcx, self.dcy, self.dr, np.cos(th).astype(np.float32), np.sin(th).astype(np.float32),
                self.cd0, self.cd1, float(ldir[0]), float(ldir[1]),
                float(mortar[0]), float(mortar[1]), float(mortar[2]), float(bg[0]), float(bg[1]), float(bg[2]),
                out, emis, float(lod_scale))
        return out, emis

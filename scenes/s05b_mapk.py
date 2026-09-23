"""s05b NATION: numba kernels for the scale-free, recursively balkanized archipelago map.

The partition is the intersection of a stack of domain-warped Worley (Voronoi) levels, each level `q` times
finer and born later than the previous one. Every pixel walks the stack, so the zoom is exact at any
magnification (no textures, no tiles). Channels between cells open as each generation's crack front passes,
turning borders into seas: an archipelago of archipelagos.

Camera: yaw `rot` in the ground plane, then the plane is squashed vertically by kt (tilt):
    q = R(rot)(p - cam);  screen(design) = (960 + q.x*zpx, 540 + q.y*zpx*kt)
"""
import math

import numpy as np
from numba import njit, prange

JIT = 0.84          # seed jitter inside the Worley grid
WARP_A = 0.30       # domain warp amplitude (fraction of the level's cell size)


@njit(cache=True, inline="always")
def _h(ix, iy, seed):
    n = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFF) / 16777216.0


@njit(cache=True, inline="always")
def _vn(x, y, seed):
    xi = math.floor(x)
    yi = math.floor(y)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    ix = int(xi)
    iy = int(yi)
    a = _h(ix, iy, seed)
    b = _h(ix + 1, iy, seed)
    c = _h(ix, iy + 1, seed)
    d = _h(ix + 1, iy + 1, seed)
    return ((a + (b - a) * u) * (1.0 - v) + (c + (d - c) * u) * v) * 2.0 - 1.0


@njit(cache=True, inline="always")
def _sstep(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, inline="always")
def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@njit(cache=True, inline="always")
def _seed_xy(ci, cj, sd):
    n = (ci * 374761393 + cj * 668265263 + sd * 1442695041) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    n = n ^ (n >> 16)
    return (ci + 0.5 + ((n & 0xFFF) / 4096.0 - 0.5) * JIT,
            cj + 0.5 + (((n >> 12) & 0xFFF) / 4096.0 - 0.5) * JIT)


@njit(cache=True)
def level_cell(px, py, s, cs, sn, seed):
    """Worley cell of one level at world point p.
    returns (ci, cj, e = world distance to the cell's edge, seed x, seed y (world, warp ignored))"""
    qx = cs * px - sn * py
    qy = sn * px + cs * py
    gx0 = qx / s
    gy0 = qy / s
    gx = gx0 + WARP_A * _vn(gx0 * 0.9 + 11.3, gy0 * 0.9 + 3.1, seed) + 0.08 * _vn(gx0 * 2.9, gy0 * 2.9, seed + 5)
    gy = gy0 + WARP_A * _vn(gx0 * 0.9 - 7.7, gy0 * 0.9 + 9.9, seed + 1) + 0.08 * _vn(gx0 * 2.9 + 4.4, gy0 * 2.9 - 1.3,
                                                                                  seed + 6)
    ix0 = int(math.floor(gx))
    iy0 = int(math.floor(gy))
    bd = 1e18
    bi = 0
    bj = 0
    bx = 0.0
    by = 0.0
    for dj in range(-1, 2):
        for di in range(-1, 2):
            ci = ix0 + di
            cj = iy0 + dj
            sx, sy = _seed_xy(ci, cj, seed)
            d = (gx - sx) * (gx - sx) + (gy - sy) * (gy - sy)
            if d < bd:
                bd = d
                bi = ci
                bj = cj
                bx = sx
                by = sy
    e = 1e18
    for dj in range(-1, 2):
        for di in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            ci = bi + di
            cj = bj + dj
            sx, sy = _seed_xy(ci, cj, seed)
            nx = sx - bx
            ny = sy - by
            ln = math.sqrt(nx * nx + ny * ny) + 1e-9
            dd = -((gx - (bx + sx) * 0.5) * nx + (gy - (by + sy) * 0.5) * ny) / ln
            if dd < e:
                e = dd
    wx = (cs * bx + sn * by) * s
    wy = (-sn * bx + cs * by) * s
    return bi, bj, e * s, wx, wy


@njit(cache=True)
def coast_sd(px, py, R0):
    r = math.sqrt(px * px + py * py)
    n = (0.62 * _vn(px * 1.3 + 3.0, py * 1.3 - 2.0, 901) + 0.25 * _vn(px * 3.1, py * 3.1, 902)
         + 0.1 * _vn(px * 7.3, py * 7.3, 903) + 0.04 * _vn(px * 17.0, py * 17.0, 904))
    return R0 * (1.0 + 0.24 * n) - r


@njit(cache=True)
def land(px, py, t, g0, NG, LS, LC, LSN, LSEED, LT, LGAP, LCR, PAL, st_sd, st_tint, st_par, st_pid, R0, fp, wexit):
    """signed land distance (world; >0 land) + tint rgb + crack glow.
    Levels < g0 are pre-resolved for the frame (see resolve()). fp = world size of a pixel (LOD);
    wexit = stop early once this deep in the water (world units)."""
    if g0 == 0:
        sd = coast_sd(px, py, R0)
    else:
        sd = st_sd
    tr = st_tint[0]
    tg = st_tint[1]
    tb = st_tint[2]
    parx = st_par[0]
    pary = st_par[1]
    pars = st_par[2]
    pid = st_pid
    crack = 0.0
    npal = PAL.shape[0]
    for g in range(g0, NG):
        if sd < -wexit:
            break
        Tg = LT[g]
        if t < Tg:
            break
        s = LS[g]
        if s < 3.0 * fp:
            break
        lod = _sstep(3.0 * fp, 16.0 * fp, s)
        bi, bj, e, wx, wy = level_cell(px, py, s, LC[g], LSN[g], LSEED[g])
        dpx = px - parx
        dpy = py - pary
        dpar = math.sqrt(dpx * dpx + dpy * dpy) / pars
        tl = t - (Tg + _h(pid, g, 99) * 0.10 + dpar * LCR[g])
        if tl <= 0.0:
            break
        # political tint: each new cell drifts to its own cool tone
        pi = int(_h(bi, bj, LSEED[g] + 41) * npal)
        if pi >= npal:
            pi = npal - 1
        a = 0.6 * _sstep(0.0, 0.3, tl) * lod
        tr += (PAL[pi, 0] - tr) * a
        tg += (PAL[pi, 1] - tg) * a
        tb += (PAL[pi, 2] - tb) * a
        op = _clamp01((tl - 0.03) / 0.55)
        op = 1.0 - (1.0 - op) * (1.0 - op) * (1.0 - op)
        w = LGAP[g] * op * lod
        sd_prev = sd
        sdg = e - 0.5 * w
        if sdg < sd:
            sd = sdg
        k = math.exp(-tl * 3.0) * lod * (1.0 - 0.8 * op) * _sstep(-fp, 2.0 * fp, sd_prev)
        if k > 0.02:
            lw = max(0.0035 * s, 1.0 * fp) + 0.3 * w
            cg = k * math.exp(-(e / lw) ** 2)
            tip = math.exp(-(tl / 0.05) ** 2)
            cg = cg * (1.0 + 1.5 * tip)
            if cg > crack:
                crack = cg
        parx = wx
        pary = wy
        pars = s
        pid = (bi * 7919 + bj * 104729 + g * 31) & 0x7FFFFFFF
    return sd, tr, tg, tb, crack


@njit(cache=True)
def resolve(cx, cy, rview, t, NG, LS, LC, LSN, LSEED, LT, LGAP, LCR, PAL, base_tint, R0):
    """Per frame: find the leading levels whose borders cannot reach the view (disc of radius rview around c)
    and whose tint has settled there; fold them into constants. -> (g0, st_sd, st_tint, st_par, st_pid)"""
    st_tint = base_tint.copy()
    st_par = np.array([0.0, 0.0, 1.0])
    pid = 0
    if coast_sd(cx, cy, R0) < rview * 1.6 + 0.02:
        return 0, 0.0, st_tint, st_par, 0
    npal = PAL.shape[0]
    g0 = 0
    for g in range(NG):
        Tg = LT[g]
        s = LS[g]
        bi, bj, e, wx, wy = level_cell(cx, cy, s, LC[g], LSN[g], LSEED[g])
        dpx = cx - st_par[0]
        dpy = cy - st_par[1]
        dpar = (math.sqrt(dpx * dpx + dpy * dpy) + rview) / st_par[2]
        tl = t - (Tg + 0.1 + dpar * LCR[g])       # latest crack arrival anywhere in view
        if tl < 0.35:
            break
        if e - 0.5 * LGAP[g] < rview * 1.5:
            break
        pi = int(_h(bi, bj, LSEED[g] + 41) * npal)
        if pi >= npal:
            pi = npal - 1
        for c in range(3):
            st_tint[c] += (PAL[pi, c] - st_tint[c]) * 0.6
        st_par[0] = wx
        st_par[1] = wy
        st_par[2] = s
        pid = (bi * 7919 + bj * 104729 + g * 31) & 0x7FFFFFFF
        g0 = g + 1
    return g0, 1e9, st_tint, st_par, pid


@njit(cache=True)
def terrain(px, py, zpx):
    """scale-free fBm height: octaves windowed by their on-screen wavelength (px)"""
    lam = 2.0
    hsum = 0.0
    wsum = 0.0
    o = 0
    while o < 60:
        L = lam * zpx
        if L < 8.0:
            break
        if L < 2400.0:
            amp = (L / 900.0) ** 1.15
            w = _sstep(2400.0, 1200.0, L) * _sstep(8.0, 20.0, L)
            f = 1.0 / lam
            hsum += amp * w * _vn(px * f + o * 13.1, py * f - o * 7.7, 300 + o)
            wsum += amp * w
        lam *= 0.45
        o += 1
    return hsum / (wsum + 1e-9) if wsum > 0 else 0.0


@njit(cache=True)
def rivers(px, py, zpx):
    """scale-free meandering rivers: zero-contours of warped noise; returns 0..1 coverage"""
    lam = 2.4
    best = 0.0
    o = 0
    while o < 40:
        L = lam * zpx
        if L < 500.0:
            break
        if L < 7000.0:
            f = 1.0 / lam
            wx = px * f + 0.6 * _vn(px * f * 2.3, py * f * 2.3, 500 + o)
            wy = py * f + 0.6 * _vn(px * f * 2.3 + 5.2, py * f * 2.3, 520 + o)
            n = _vn(wx, wy, 540 + o) + 0.35
            wpx = 1.0 + 1.6 * _sstep(700.0, 5000.0, L)       # line width in px
            dist = abs(n) * L * 0.2                            # ~px distance to the contour
            c = _sstep(wpx + 0.8, wpx - 0.8, dist)
            c *= _sstep(500.0, 1000.0, L) * _sstep(7000.0, 4000.0, L)
            if c > best:
                best = c
        lam *= 0.5
        o += 1
    return best


@njit(parallel=True, cache=True)
def render_map(out, cover, buf, s_px, camx, camy, zpx, kt, rot, t, depth, g0, NG, LS, LC, LSN, LSEED, LT, LGAP, LCR,
               PAL, st_sd, st_tint, st_par, st_pid, R0, water_deep, water_shal, foam, crack_col, relief, river_k):
    """out (h,w,3) float32 sRGB; cover (h,w) land coverage of top faces; buf (h,w,10) scratch."""
    h, w = cover.shape
    cr = math.cos(rot)
    sr = math.sin(rot)
    fpw = 1.0 / (zpx * s_px)                  # world units per pixel (x)
    fpy = 1.0 / (zpx * kt * s_px)
    fp = 0.5 * (fpw + fpy)
    dwy = depth / (zpx * kt)                  # slab depth in world units (screen-down direction)
    dnx = sr
    dny = cr
    wexit = 70.0 * fp
    # ------------------------------------------------ pass 1: geometry
    for yy in prange(h):
        for xx in range(w):
            sx = (xx + 0.5) / s_px - 960.0
            sy = (yy + 0.5) / s_px - 540.0
            qx = sx / zpx
            qy = sy / (zpx * kt)
            px = camx + cr * qx + sr * qy
            py = camy - sr * qx + cr * qy
            sd, tr, tg, tb, crack = land(px, py, t, g0, NG, LS, LC, LSN, LSEED, LT, LGAP, LCR, PAL, st_sd, st_tint,
                                         st_par, st_pid, R0, fp, wexit)
            cov = _sstep(-fp, fp, sd)
            wr = water_deep[0]
            wg = water_deep[1]
            wb = water_deep[2]
            if cov < 1.0:
                dpx = -sd / fp                                    # px out into the water
                shal = math.exp(-max(dpx, 0.0) / 9.0)
                wr = water_deep[0] + (water_shal[0] - water_deep[0]) * shal
                wg = water_deep[1] + (water_shal[1] - water_deep[1]) * shal
                wb = water_deep[2] + (water_shal[2] - water_deep[2]) * shal
                # bathymetric contour rings (cartographic)
                ring = 0.0
                for rr in (9.0, 20.0, 36.0):
                    ring += math.exp(-((dpx - rr) / 0.8) ** 2) * (0.13 - rr * 0.002)
                wr += ring * 0.9
                wg += ring
                wb += ring * 1.05
                # drifting glints on the water (screen-space, so every scale sparkles the same)
                gl = _vn(sx * 0.045 + t * 0.7, sy * 0.13 - t * 0.25, 91) + 0.4 * _vn(sx * 0.11 - t, sy * 0.3, 92)
                gl = max(gl - 0.62, 0.0) * 1.8
                wr += 0.14 * gl
                wg += 0.16 * gl
                wb += 0.2 * gl
                if -sd < dwy + depth * 1.8 * fp + 2.0 * fp:
                    # slab side (cliff) visible below the top face, and its shadow on the water
                    cl = 0.0
                    cfr = 0.0
                    ctr = 0.0
                    ctg = 0.0
                    ctb = 0.0
                    sd2 = -1e9
                    for k in range(1, 3):
                        fr = k * 0.5
                        sdk, a1, a2, a3, c_ = land(px - dnx * dwy * fr, py - dny * dwy * fr, t, g0, NG, LS, LC, LSN,
                                                   LSEED, LT, LGAP, LCR, PAL, st_sd, st_tint, st_par, st_pid, R0, fp,
                                                   wexit)
                        ck = _sstep(-fp, fp, sdk)
                        if ck > cl:
                            cl = ck
                            cfr = fr
                            ctr = a1
                            ctg = a2
                            ctb = a3
                        if k == 2:
                            sd2 = sdk
                    shd = _sstep(-depth * 1.8 * fp, 0.0, sd2) * 0.5
                    wr *= (1.0 - shd)
                    wg *= (1.0 - shd)
                    wb *= (1.0 - shd * 0.8)
                    fo = math.exp(-((dpx - 0.8) / 1.1) ** 2) * 0.5
                    wr += (foam[0] - wr) * fo
                    wg += (foam[1] - wg) * fo
                    wb += (foam[2] - wb) * fo
                    if cl > 0.0:
                        kd = 0.62 - 0.26 * cfr
                        wr += (ctr * kd - wr) * cl
                        wg += (ctg * kd - wg) * cl
                        wb += (ctb * (kd + 0.06) - wb) * cl
            hh = 0.0
            rv = 0.0
            fo_ = 0.0
            if sd > -3.0 * fp:
                hh = terrain(px, py, zpx)
                # woodland patches: one noise octave kept ~300 px on screen (blend of the two nearest octaves)
                lf = math.log2(320.0 / zpx)
                o0 = math.floor(lf)
                fr_ = lf - o0
                l0 = 2.0 ** o0
                n0 = _vn(px / l0 + 3.3, py / l0 - 1.7, 700 + int(o0) & 0xFFFF)
                n1 = _vn(px / (l0 * 2.0) + 3.3, py / (l0 * 2.0) - 1.7, 700 + int(o0 + 1) & 0xFFFF)
                fo_ = _sstep(0.12, 0.42, n0 * (1.0 - fr_) + n1 * fr_) * (1.0 - _sstep(0.1, 0.5, hh))
                rv = rivers(px, py, zpx) * _sstep(0.0, 5.0 * fp, sd) * river_k if river_k > 0.0 else 0.0
            buf[yy, xx, 0] = wr
            buf[yy, xx, 1] = wg
            buf[yy, xx, 2] = wb
            buf[yy, xx, 3] = tr
            buf[yy, xx, 4] = tg
            buf[yy, xx, 5] = tb
            buf[yy, xx, 6] = hh
            buf[yy, xx, 7] = crack
            buf[yy, xx, 8] = rv
            buf[yy, xx, 9] = fo_
            cover[yy, xx] = cov
    # ------------------------------------------------ pass 2: relief shading + compose
    lx = -0.55
    ly = -0.65
    lz = 0.52
    for yy in prange(h):
        y0 = max(yy - 1, 0)
        y1 = min(yy + 1, h - 1)
        for xx in range(w):
            cov = cover[yy, xx]
            r = buf[yy, xx, 0]
            g_ = buf[yy, xx, 1]
            b = buf[yy, xx, 2]
            if cov > 0.0:
                x0 = max(xx - 1, 0)
                x1 = min(xx + 1, w - 1)
                hh = buf[yy, xx, 6]
                gx = (buf[yy, x1, 6] - buf[yy, x0, 6]) * relief / s_px
                gy = (buf[y1, xx, 6] - buf[y0, xx, 6]) * relief / s_px
                nz = 1.0 / math.sqrt(gx * gx + gy * gy + 1.0)
                shade = (-gx * lx - gy * ly + lz) * nz          # ~0.5 flat
                sh = 0.62 + 0.75 * shade
                elev = 0.5 + 0.5 * hh
                base = 0.7 + 0.36 * elev
                fo_ = buf[yy, xx, 9]
                lr = buf[yy, xx, 3] * base * sh * (1.0 - 0.22 * fo_)
                lg = buf[yy, xx, 4] * base * sh * (1.0 - 0.12 * fo_)
                lb = buf[yy, xx, 5] * base * (sh * 0.94 + 0.05) * (1.0 - 0.14 * fo_)
                hi = _sstep(0.22, 0.7, hh) * 0.45
                lr += (0.86 - lr) * hi
                lg += (0.89 - lg) * hi
                lb += (0.94 - lb) * hi
                # faint elevation contour lines
                gm = math.sqrt(gx * gx + gy * gy) / relief * s_px + 1e-6
                fc = hh * 7.0
                dc = abs(fc - math.floor(fc + 0.5)) / (gm * 7.0 + 1e-6)     # px to nearest contour
                cl = math.exp(-(dc / 0.7) ** 2) * 0.1
                lr *= 1.0 - cl
                lg *= 1.0 - cl
                lb *= 1.0 - cl * 0.7
                rv = buf[yy, xx, 8]
                if rv > 0.0:
                    lr += (water_shal[0] * 0.85 - lr) * rv
                    lg += (water_shal[1] * 0.9 - lg) * rv
                    lb += (water_shal[2] * 1.0 - lb) * rv
                r = r * (1.0 - cov) + lr * cov
                g_ = g_ * (1.0 - cov) + lg * cov
                b = b * (1.0 - cov) + lb * cov
            crack = buf[yy, xx, 7]
            if crack > 0.003:
                ck = crack if crack < 1.0 else 1.0
                r += (crack_col[0] - r) * ck + 0.12 * crack
                g_ += (crack_col[1] - g_) * ck + 0.12 * crack
                b += (crack_col[2] - b) * ck + 0.15 * crack
            out[yy, xx, 0] = r
            out[yy, xx, 1] = g_
            out[yy, xx, 2] = b


@njit(cache=True)
def _unwarp(sx, sy, seed):
    """grid point g with warp(g) = (sx, sy)  (fixed-point iteration)"""
    gx = sx
    gy = sy
    for _ in range(6):
        dx = WARP_A * _vn(gx * 0.9 + 11.3, gy * 0.9 + 3.1, seed) + 0.08 * _vn(gx * 2.9, gy * 2.9, seed + 5)
        dy = WARP_A * _vn(gx * 0.9 - 7.7, gy * 0.9 + 9.9, seed + 1) + 0.08 * _vn(gx * 2.9 + 4.4, gy * 2.9 - 1.3,
                                                                              seed + 6)
        gx = sx - dx
        gy = sy - dy
    return gx, gy


@njit(cache=True)
def seeds_in_rect(x0, y0, x1, y1, s, cs, sn, seed):
    """cells of one level whose centre (seed pulled back through the warp) lies in the world AABB.
    returns (k,4): ci, cj, wx, wy"""
    xs = np.array([x0, x1, x0, x1])
    ys = np.array([y0, y0, y1, y1])
    gxs = (cs * xs - sn * ys) / s
    gys = (sn * xs + cs * ys) / s
    i0 = int(math.floor(gxs.min())) - 1
    i1 = int(math.floor(gxs.max())) + 1
    j0 = int(math.floor(gys.min())) - 1
    j1 = int(math.floor(gys.max())) + 1
    n = (i1 - i0 + 1) * (j1 - j0 + 1)
    out = np.empty((n, 4))
    k = 0
    for cj in range(j0, j1 + 1):
        for ci in range(i0, i1 + 1):
            sx, sy = _seed_xy(ci, cj, seed)
            gx, gy = _unwarp(sx, sy, seed)
            wx = (cs * gx + sn * gy) * s
            wy = (-sn * gx + cs * gy) * s
            if x0 <= wx <= x1 and y0 <= wy <= y1:
                out[k, 0] = ci
                out[k, 1] = cj
                out[k, 2] = wx
                out[k, 3] = wy
                k += 1
    return out[:k]


@njit(cache=True, parallel=True)
def land_points(xs, ys, t, NG, LS, LC, LSN, LSEED, LT, LGAP, LCR, PAL, R0, glev, fp, base_tint):
    """sd + tint + the cell index of level `glev` at arbitrary world points (for layout in setup/overlays)."""
    n = xs.shape[0]
    sd = np.empty(n)
    tint = np.empty((n, 3))
    ids = np.empty((n, 2), np.int64)
    st_tint = base_tint.copy()
    st_par = np.array([0.0, 0.0, 1.0])
    for i in prange(n):
        v = land(xs[i], ys[i], t, 0, NG, LS, LC, LSN, LSEED, LT, LGAP, LCR, PAL, 0.0, st_tint, st_par, 0, R0, fp,
                 1e9)
        sd[i] = v[0]
        tint[i, 0] = v[1]
        tint[i, 1] = v[2]
        tint[i, 2] = v[3]
        bi, bj, e, wx, wy = level_cell(xs[i], ys[i], LS[glev], LC[glev], LSN[glev], LSEED[glev])
        ids[i, 0] = bi
        ids[i, 1] = bj
    return sd, tint, ids

"""Numba rasterisers for vx.flag (FlagSmith): plain-yellow cloth, per-pixel lighting, z-buffer, AA.

The only colours these kernels can produce are interpolations between the palette yellows
(flag_deep -> flag_shadow -> flag -> flag_light -> flag_glow) and, for poles, the pole woods.
"""
import math

import numpy as np
from numba import njit

# palette ramp: k = -1 deep, -0.5 shadow, 0 flag, 1 light, 2 glow  (sRGB 0..1, filled by vx.flag)
RAMP_K = np.array([-1.0, -0.5, 0.0, 1.0, 2.0])


@njit(cache=True, inline="always")
def ramp(k, pal, out3):
    if k <= -1.0:
        for c in range(3):
            out3[c] = pal[0, c]
        return
    if k >= 2.0:
        for c in range(3):
            out3[c] = pal[4, c]
        return
    if k < -0.5:
        i = 0
        f = (k + 1.0) / 0.5
    elif k < 0.0:
        i = 1
        f = (k + 0.5) / 0.5
    elif k < 1.0:
        i = 2
        f = k
    else:
        i = 3
        f = k - 1.0
    for c in range(3):
        out3[c] = pal[i, c] + (pal[i + 1, c] - pal[i, c]) * f


@njit(cache=True, inline="always")
def shade(nx, ny, nz, cav, L, prm):
    """n (unit, any side), cavity 0..1 -> k on the palette ramp. prm: k0, kd, wrap, kt, ks, shin, kc, lift"""
    if nz < 0.0:
        nx = -nx
        ny = -ny
        nz = -nz
    ndl = nx * L[0] + ny * L[1] + nz * L[2]
    wrap = prm[2]
    diff = (ndl + wrap) / (1.0 + wrap)
    if diff < 0.0:
        diff = 0.0
    tr = -ndl
    if tr < 0.0:
        tr = 0.0
    # half vector with viewer (0,0,1)
    hx = L[0]
    hy = L[1]
    hz = L[2] + 1.0
    hl = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
    ndh = (nx * hx + ny * hy + nz * hz) / hl
    if ndh < 0.0:
        ndh = 0.0
    spec = ndh ** prm[5]
    # sheen: grazing brightening typical of satin-ish nylon
    sheen = (1.0 - nz) ** 3
    k = prm[0] + prm[1] * diff + prm[3] * tr * math.sqrt(tr) + prm[4] * spec + 0.25 * prm[4] * sheen * diff
    k -= prm[6] * cav
    k += prm[7]
    return k


@njit(cache=True, fastmath=True)
def shade_grid(Nn, cav, L, prm, out):
    for j in range(Nn.shape[0]):
        for i in range(Nn.shape[1]):
            out[j, i] = shade(Nn[j, i, 0], Nn[j, i, 1], Nn[j, i, 2], cav[j, i], L, prm)
    return 0


@njit(cache=True, fastmath=True)
def raster_cloth(P, Nn, cav, ox, oy, w, h, ss, L, prm, pal, out, pq, prad, hl, ppal, Kv, cov):
    """P (Ny,Nx,3) device px (x,y,z toward viewer); Nn world normals; cav (Ny,Nx);
    tile origin (ox,oy) device px, size (w,h); ss supersampling; out uint8 (h, stride/4, 4) BGRA premult.
    pq (4,2): device quad of the pole's top segment (l=-1,+1,+1,-1 corners; NaN = none) z-buffered as a cylinder
    of radius prad (device px) so cloth passing behind the pole is hidden; hl = highlight position 0..1."""
    Ny, Nx = P.shape[0], P.shape[1]
    SW = w * ss
    SH = h * ss
    zb = np.full((SH, SW), -1e30, np.float32)
    kb = np.zeros((SH, SW), np.float32)
    tb = np.zeros((SH, SW), np.uint8)
    if not math.isnan(pq[0, 0]):
        for tri in range(2):
            if tri == 0:
                a_, b_, c_ = 0, 1, 2
                la, lb, lc = -1.0, 1.0, 1.0
            else:
                a_, b_, c_ = 0, 2, 3
                la, lb, lc = -1.0, 1.0, -1.0
            ax = (pq[a_, 0] - ox) * ss - 0.5
            ay = (pq[a_, 1] - oy) * ss - 0.5
            bx = (pq[b_, 0] - ox) * ss - 0.5
            by = (pq[b_, 1] - oy) * ss - 0.5
            cx = (pq[c_, 0] - ox) * ss - 0.5
            cy = (pq[c_, 1] - oy) * ss - 0.5
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-9:
                continue
            ia_ = 1.0 / area
            x0 = max(int(math.floor(min(ax, bx, cx))), 0)
            x1 = min(int(math.ceil(max(ax, bx, cx))), SW - 1)
            y0 = max(int(math.floor(min(ay, by, cy))), 0)
            y1 = min(int(math.ceil(max(ay, by, cy))), SH - 1)
            for sy in range(y0, y1 + 1):
                fy = float(sy)
                for sx in range(x0, x1 + 1):
                    fx = float(sx)
                    w0 = ((bx - fx) * (cy - fy) - (by - fy) * (cx - fx)) * ia_
                    w1 = ((cx - fx) * (ay - fy) - (cy - fy) * (ax - fx)) * ia_
                    w2 = 1.0 - w0 - w1
                    if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4:
                        continue
                    l = w0 * la + w1 * lb + w2 * lc
                    z = prad * math.sqrt(max(0.0, 1.0 - l * l))
                    if z > zb[sy, sx]:
                        zb[sy, sx] = z
                        kb[sy, sx] = l
                        tb[sy, sx] = 2
    p0 = max(0.05, hl - 0.3)
    p2 = min(0.95, hl + 0.3)
    gour = Kv.shape[0] == Ny and Kv.shape[1] == Nx
    for j in range(Ny - 1):
        for i in range(Nx - 1):
            for tri in range(2):
                if tri == 0:
                    ja, ia, jb, ib, jc, ic = j, i, j, i + 1, j + 1, i + 1
                else:
                    ja, ia, jb, ib, jc, ic = j, i, j + 1, i + 1, j + 1, i
                ax = (P[ja, ia, 0] - ox) * ss - 0.5
                ay = (P[ja, ia, 1] - oy) * ss - 0.5
                bx = (P[jb, ib, 0] - ox) * ss - 0.5
                by = (P[jb, ib, 1] - oy) * ss - 0.5
                cx = (P[jc, ic, 0] - ox) * ss - 0.5
                cy = (P[jc, ic, 1] - oy) * ss - 0.5
                area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                if abs(area) < 1e-9:
                    continue
                ia_ = 1.0 / area
                x0 = int(math.floor(min(ax, bx, cx)))
                x1 = int(math.ceil(max(ax, bx, cx)))
                y0 = int(math.floor(min(ay, by, cy)))
                y1 = int(math.ceil(max(ay, by, cy)))
                if x0 < 0:
                    x0 = 0
                if y0 < 0:
                    y0 = 0
                if x1 > SW - 1:
                    x1 = SW - 1
                if y1 > SH - 1:
                    y1 = SH - 1
                if x0 > x1 or y0 > y1:
                    continue
                za = P[ja, ia, 2]
                zb_ = P[jb, ib, 2]
                zc = P[jc, ic, 2]
                eps = -1e-4
                for sy in range(y0, y1 + 1):
                    fy = float(sy)
                    for sx in range(x0, x1 + 1):
                        fx = float(sx)
                        w0 = ((bx - fx) * (cy - fy) - (by - fy) * (cx - fx)) * ia_
                        w1 = ((cx - fx) * (ay - fy) - (cy - fy) * (ax - fx)) * ia_
                        w2 = 1.0 - w0 - w1
                        if w0 < eps or w1 < eps or w2 < eps:
                            continue
                        z = w0 * za + w1 * zb_ + w2 * zc
                        if z <= zb[sy, sx]:
                            continue
                        zb[sy, sx] = z
                        tb[sy, sx] = 1
                        if gour:
                            kb[sy, sx] = w0 * Kv[ja, ia] + w1 * Kv[jb, ib] + w2 * Kv[jc, ic]
                            continue
                        nx_ = w0 * Nn[ja, ia, 0] + w1 * Nn[jb, ib, 0] + w2 * Nn[jc, ic, 0]
                        ny_ = w0 * Nn[ja, ia, 1] + w1 * Nn[jb, ib, 1] + w2 * Nn[jc, ic, 1]
                        nz_ = w0 * Nn[ja, ia, 2] + w1 * Nn[jb, ib, 2] + w2 * Nn[jc, ic, 2]
                        nl = math.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_) + 1e-12
                        cv = w0 * cav[ja, ia] + w1 * cav[jb, ib] + w2 * cav[jc, ic]
                        kb[sy, sx] = shade(nx_ / nl, ny_ / nl, nz_ / nl, cv, L, prm)
    # resolve
    col = np.zeros(3)
    n2 = ss * ss
    for y in range(h):
        for x in range(w):
            cnt = 0
            ccnt = 0
            r = 0.0
            g = 0.0
            b = 0.0
            for sy in range(y * ss, y * ss + ss):
                for sx in range(x * ss, x * ss + ss):
                    t_ = tb[sy, sx]
                    if t_ == 0:
                        continue
                    cnt += 1
                    if t_ == 1:
                        ccnt += 1
                        ramp(kb[sy, sx], pal, col)
                    else:
                        pp = 0.5 * (kb[sy, sx] + 1.0)
                        if pp < p0:
                            f = pp / p0
                            for c in range(3):
                                col[c] = ppal[0, c] + (ppal[1, c] - ppal[0, c]) * f
                        elif pp < hl:
                            f = (pp - p0) / max(1e-6, hl - p0)
                            for c in range(3):
                                col[c] = ppal[1, c] + (ppal[2, c] - ppal[1, c]) * f
                        elif pp < p2:
                            f = (pp - hl) / max(1e-6, p2 - hl)
                            for c in range(3):
                                col[c] = ppal[2, c] + (ppal[1, c] - ppal[2, c]) * f
                        else:
                            f = (pp - p2) / max(1e-6, 1.0 - p2)
                            for c in range(3):
                                col[c] = ppal[1, c] + (ppal[0, c] - ppal[1, c]) * min(1.0, f)
                    r += col[0]
                    g += col[1]
                    b += col[2]
            if cnt == 0:
                continue
            a = cnt / n2
            cov[y, x] = ccnt / n2
            out[y, x, 0] = int(b / n2 * 255.0 + 0.5)
            out[y, x, 1] = int(g / n2 * 255.0 + 0.5)
            out[y, x, 2] = int(r / n2 * 255.0 + 0.5)
            out[y, x, 3] = int(a * 255.0 + 0.5)
    return 0


@njit(cache=True, fastmath=True, inline="always")
def _tri(zb, kb, tb, SW, SH, ax, ay, bx, by, cx, cy, za, zb_, zc, ka, kb_, kc, typ):
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(area) < 1e-9:
        return
    ia_ = 1.0 / area
    x0 = int(math.floor(min(ax, bx, cx)))
    x1 = int(math.ceil(max(ax, bx, cx)))
    y0 = int(math.floor(min(ay, by, cy)))
    y1 = int(math.ceil(max(ay, by, cy)))
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x1 > SW - 1:
        x1 = SW - 1
    if y1 > SH - 1:
        y1 = SH - 1
    for sy in range(y0, y1 + 1):
        fy = float(sy)
        for sx in range(x0, x1 + 1):
            fx = float(sx)
            w0 = ((bx - fx) * (cy - fy) - (by - fy) * (cx - fx)) * ia_
            w1 = ((cx - fx) * (ay - fy) - (cy - fy) * (ax - fx)) * ia_
            w2 = 1.0 - w0 - w1
            if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4:
                continue
            z = w0 * za + w1 * zb_ + w2 * zc
            if z <= zb[sy, sx]:
                continue
            zb[sy, sx] = z
            kb[sy, sx] = w0 * ka + w1 * kb_ + w2 * kc
            tb[sy, sx] = typ


@njit(cache=True, fastmath=True)
def raster_field(B, wi, fw, vi, k0, k1, ff, sdir, s, Lp, ang, bx, by, alpha, lift, pw,
                 M, view, L, prm, pal, ppal, buf, ox, oy, glow, sxm, sym):
    """Batch crowd flags. B bank (Wn,Vn,Lf,ny,nx,3) normalised; per-instance arrays (already painter-sorted);
    M = ctx matrix (xx,yx,xy,yy,x0,y0); buf float32 (H,W,5) premultiplied RGBA + glow weight at device (ox,oy)."""
    ny = B.shape[3]
    nx = B.shape[4]
    H = buf.shape[0]
    W = buf.shape[1]
    xx, yx, xy, yy, x0, y0 = M[0], M[1], M[2], M[3], M[4], M[5]
    ds = math.sqrt(abs(xx * yy - xy * yx))
    col = np.zeros(3)
    P = np.zeros((ny, nx, 3))      # world
    D = np.zeros((ny, nx, 3))      # device
    K = np.zeros((ny, nx))
    ci = np.zeros(nx, np.int64)
    ri = np.zeros(ny, np.int64)
    for q in range(wi.shape[0]):
        a_q = alpha[q]
        if a_q <= 0.0 or s[q] <= 0.0:
            continue
        cwd = 0.375 * s[q] * ds * sxm[q]
        if cwd < 5.0:
            nc, nr = 3, 2
        elif cwd < 18.0:
            nc, nr = 5, 3
        elif cwd < 60.0:
            nc, nr = 8, 5
        else:
            nc, nr = nx, ny
        for i in range(nc):
            ci[i] = int(round(i * (nx - 1) / (nc - 1)))
        for j in range(nr):
            ri[j] = int(round(j * (ny - 1) / (nr - 1)))
        ca = math.cos(ang[q])
        sa = math.sin(ang[q])
        tx = bx[q] + sa * Lp[q]
        ty = by[q] - ca * Lp[q]
        w_ = wi[q]
        v_ = vi[q]
        f0 = k0[q]
        f1 = k1[q]
        a0 = (1.0 - ff[q]) * (1.0 - fw[q])
        a1 = ff[q] * (1.0 - fw[q])
        a2 = (1.0 - ff[q]) * fw[q]
        a3 = ff[q] * fw[q]
        w2 = min(w_ + 1, B.shape[0] - 1)
        sc = s[q]
        sd = sdir[q] * sxm[q]
        scy = sc * sym[q]
        dminx = 1e30
        dminy = 1e30
        dmaxx = -1e30
        dmaxy = -1e30
        for jj in range(nr):
            j = ri[jj]
            for ii in range(nc):
                i = ci[ii]
                lx = (a0 * B[w_, v_, f0, j, i, 0] + a1 * B[w_, v_, f1, j, i, 0]
                      + a2 * B[w2, v_, f0, j, i, 0] + a3 * B[w2, v_, f1, j, i, 0]) * sd * sc
                ly = (a0 * B[w_, v_, f0, j, i, 1] + a1 * B[w_, v_, f1, j, i, 1]
                      + a2 * B[w2, v_, f0, j, i, 1] + a3 * B[w2, v_, f1, j, i, 1]) * scy
                lz = (a0 * B[w_, v_, f0, j, i, 2] + a1 * B[w_, v_, f1, j, i, 2]
                      + a2 * B[w2, v_, f0, j, i, 2] + a3 * B[w2, v_, f1, j, i, 2]) * sc
                wx = tx + lx * ca - ly * sa
                wy = ty + lx * sa + ly * ca
                P[jj, ii, 0] = wx
                P[jj, ii, 1] = wy
                P[jj, ii, 2] = lz
                Xo = wx + view[0] * lz
                Yo = wy + view[1] * lz
                dx_ = xx * Xo + xy * Yo + x0 - ox
                dy_ = yx * Xo + yy * Yo + y0 - oy
                D[jj, ii, 0] = dx_
                D[jj, ii, 1] = dy_
                D[jj, ii, 2] = lz * ds
                dminx = min(dminx, dx_)
                dminy = min(dminy, dy_)
                dmaxx = max(dmaxx, dx_)
                dmaxy = max(dmaxy, dy_)
        # per-vertex lighting
        for jj in range(nr):
            j0 = max(jj - 1, 0)
            j1 = min(jj + 1, nr - 1)
            for ii in range(nc):
                i0 = max(ii - 1, 0)
                i1 = min(ii + 1, nc - 1)
                ux = P[jj, i1, 0] - P[jj, i0, 0]
                uy = P[jj, i1, 1] - P[jj, i0, 1]
                uz = P[jj, i1, 2] - P[jj, i0, 2]
                vx_ = P[j1, ii, 0] - P[j0, ii, 0]
                vy_ = P[j1, ii, 1] - P[j0, ii, 1]
                vz_ = P[j1, ii, 2] - P[j0, ii, 2]
                nx_ = uy * vz_ - uz * vy_
                ny_ = uz * vx_ - ux * vz_
                nz_ = ux * vy_ - uy * vx_
                nl = math.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_) + 1e-12
                K[jj, ii] = shade(nx_ / nl, ny_ / nl, nz_ / nl, 0.0, L, prm) + lift[q]
        # pole quad (device)
        pdx = ca * pw[q] * 0.5
        pdy = sa * pw[q] * 0.5
        pb = np.empty((4, 2))
        pts = ((bx[q] - pdx, by[q] - pdy), (bx[q] + pdx, by[q] + pdy), (tx + pdx, ty + pdy), (tx - pdx, ty - pdy))
        for m in range(4):
            pb[m, 0] = xx * pts[m][0] + xy * pts[m][1] + x0 - ox
            pb[m, 1] = yx * pts[m][0] + yy * pts[m][1] + y0 - oy
            dminx = min(dminx, pb[m, 0])
            dminy = min(dminy, pb[m, 1])
            dmaxx = max(dmaxx, pb[m, 0])
            dmaxy = max(dmaxy, pb[m, 1])
        tx0 = max(int(math.floor(dminx)) - 1, 0)
        ty0 = max(int(math.floor(dminy)) - 1, 0)
        tx1 = min(int(math.ceil(dmaxx)) + 1, W)
        ty1 = min(int(math.ceil(dmaxy)) + 1, H)
        tw = tx1 - tx0
        th = ty1 - ty0
        if tw <= 0 or th <= 0:
            continue
        area = tw * th
        ss = 4 if area < 3000 else 3 if area < 30000 else 2
        SW = tw * ss
        SH = th * ss
        zb = np.full((SH, SW), -1e30, np.float32)
        kb = np.zeros((SH, SW), np.float32)
        tb = np.zeros((SH, SW), np.uint8)
        offx = tx0
        offy = ty0
        # pole (behind the cloth): attribute = lateral coordinate -1..1
        _tri(zb, kb, tb, SW, SH,
             (pb[0, 0] - offx) * ss - 0.5, (pb[0, 1] - offy) * ss - 0.5,
             (pb[1, 0] - offx) * ss - 0.5, (pb[1, 1] - offy) * ss - 0.5,
             (pb[2, 0] - offx) * ss - 0.5, (pb[2, 1] - offy) * ss - 0.5, -1e9, -1e9, -1e9, -1.0, 1.0, 1.0, 2)
        _tri(zb, kb, tb, SW, SH,
             (pb[0, 0] - offx) * ss - 0.5, (pb[0, 1] - offy) * ss - 0.5,
             (pb[2, 0] - offx) * ss - 0.5, (pb[2, 1] - offy) * ss - 0.5,
             (pb[3, 0] - offx) * ss - 0.5, (pb[3, 1] - offy) * ss - 0.5, -1e9, -1e9, -1e9, -1.0, 1.0, -1.0, 2)
        for jj in range(nr - 1):
            for ii in range(nc - 1):
                for tri in range(2):
                    if tri == 0:
                        ja, ia, jb, ib, jc, ic = jj, ii, jj, ii + 1, jj + 1, ii + 1
                    else:
                        ja, ia, jb, ib, jc, ic = jj, ii, jj + 1, ii + 1, jj + 1, ii
                    _tri(zb, kb, tb, SW, SH,
                         (D[ja, ia, 0] - offx) * ss - 0.5, (D[ja, ia, 1] - offy) * ss - 0.5,
                         (D[jb, ib, 0] - offx) * ss - 0.5, (D[jb, ib, 1] - offy) * ss - 0.5,
                         (D[jc, ic, 0] - offx) * ss - 0.5, (D[jc, ic, 1] - offy) * ss - 0.5,
                         D[ja, ia, 2], D[jb, ib, 2], D[jc, ic, 2], K[ja, ia], K[jb, ib], K[jc, ic], 1)
        # resolve + composite over
        n2 = ss * ss
        for y in range(th):
            for x in range(tw):
                cnt = 0
                r = 0.0
                g = 0.0
                b = 0.0
                for sy in range(y * ss, y * ss + ss):
                    for sx in range(x * ss, x * ss + ss):
                        t_ = tb[sy, sx]
                        if t_ == 0:
                            continue
                        cnt += 1
                        if t_ == 1:
                            ramp(kb[sy, sx], pal, col)
                        else:
                            l = kb[sy, sx]
                            # soft cylindrical wood: shadow at the rims, light a little left of centre
                            e = abs(l + 0.25)
                            if e < 0.5:
                                f = e / 0.5
                                for c in range(3):
                                    col[c] = ppal[2, c] + (ppal[1, c] - ppal[2, c]) * f
                            else:
                                f = min(1.0, (e - 0.5) / 0.75)
                                for c in range(3):
                                    col[c] = ppal[1, c] + (ppal[0, c] - ppal[1, c]) * f
                        r += col[0]
                        g += col[1]
                        b += col[2]
                if cnt == 0:
                    continue
                cov = cnt / n2 * a_q
                inv = a_q / n2
                yy_ = y + ty0
                xx_ = x + tx0
                keep = 1.0 - cov
                buf[yy_, xx_, 0] = r * inv + buf[yy_, xx_, 0] * keep
                buf[yy_, xx_, 1] = g * inv + buf[yy_, xx_, 1] * keep
                buf[yy_, xx_, 2] = b * inv + buf[yy_, xx_, 2] * keep
                buf[yy_, xx_, 3] = cov + buf[yy_, xx_, 3] * keep
                buf[yy_, xx_, 4] = cov * glow[q] + buf[yy_, xx_, 4] * keep
    return 0


@njit(cache=True)
def to_bgra(buf, out):
    H, W = buf.shape[0], buf.shape[1]
    for y in range(H):
        for x in range(W):
            a = buf[y, x, 3]
            if a <= 0.0:
                continue
            out[y, x, 0] = min(255, int(buf[y, x, 2] * 255.0 + 0.5))
            out[y, x, 1] = min(255, int(buf[y, x, 1] * 255.0 + 0.5))
            out[y, x, 2] = min(255, int(buf[y, x, 0] * 255.0 + 0.5))
            out[y, x, 3] = min(255, int(a * 255.0 + 0.5))
    return 0

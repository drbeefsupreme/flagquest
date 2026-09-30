"""vx.mosaic_ras - numba raster kernels for tesserae (owner: Tessellator). Internal; use vx.mosaic.

Every tile is a convex polygon already projected to SCREEN PIXELS. One pass per tile over its bounding box:
signed distance to the polygon (max over edge half-planes) gives exact AA coverage, the bevel profile, and the
distance to the nearest tile for the mortar joints (bed). Shading inputs are per tile (computed in numpy in linear
light); the kernel only adds the per-pixel bevel / dome / rim-glint / screen-feed terms.
"""
import math

import numpy as np
from numba import njit

BIG = np.float32(1e9)

# kinds
K_TILE, K_IMPRINT, K_SCREEN = 0, 1, 2

_LUT_N = 4096
_x = np.linspace(0.0, 1.0, _LUT_N, dtype=np.float64)
ENC = np.where(_x <= 0.0031308, 12.92 * _x, 1.055 * np.power(_x, 1 / 2.4) - 0.055).astype(np.float32)
DEC = np.where(_x <= 0.04045, _x / 12.92, np.power((_x + 0.055) / 1.055, 2.4)).astype(np.float32)


def srgb_to_lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, np.power((np.maximum(c, 0) + 0.055) / 1.055, 2.4)).astype(np.float32)


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, np.float32), 0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


@njit(cache=True, fastmath=True)
def _lut(lut, v):
    if v <= 0.0:
        return lut[0]
    if v >= 1.0:
        return lut[lut.shape[0] - 1]
    f = v * (lut.shape[0] - 1)
    i = int(f)
    u = f - i
    return lut[i] + (lut[i + 1] - lut[i]) * u


@njit(cache=True, fastmath=True)
def _hash(a, b):
    h = (a * 374761393 + b * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / 16777216.0


# slop palette (linear): flesh, cyan, beige, mauve, magenta, electric cyan, pale blue
_FEED = np.array([[0.80, 0.58, 0.63], [0.48, 0.75, 0.81], [0.58, 0.53, 0.40], [0.26, 0.21, 0.27],
                  [1.00, 0.03, 0.25], [0.00, 0.78, 1.00], [0.35, 0.69, 1.00], [0.95, 0.80, 0.55]], np.float32)


@njit(cache=True, fastmath=True)
def _feed(u, v, t, seed, out):
    """a tiny glowing phone feed; u,v in [-1,1] (v down). Writes linear rgb (emissive) into out."""
    au = abs(u)
    av = abs(v)
    # rounded bezel
    bx = max(au - 0.80, 0.0)
    by = max(av - 0.86, 0.0)
    if au > 0.94 or av > 0.97 or bx * bx + by * by > 0.012:
        out[0] = 0.012
        out[1] = 0.012
        out[2] = 0.016
        return
    sx = (u + 0.84) / 1.68
    sy = (v + 0.90) / 1.80
    if sy < 0.07:  # status bar
        out[0] = 0.05
        out[1] = 0.05
        out[2] = 0.07
        return
    sid = int(seed * 1000003.0)
    speed = 0.35 + 0.9 * _hash(sid, 7)
    pos = sy * 2.1 + t * speed + 10.0 * _hash(sid, 3)
    k = int(math.floor(pos))
    f = pos - k
    hk = _hash(sid, k)
    if f < 0.64:
        ci = int(hk * 8.0) % 8
        # soft blob in the image (a face / product / meme)
        cx = 0.5 + 0.25 * (_hash(sid + 1, k) - 0.5)
        cy = 0.32
        dx = sx - cx
        dy = (f - cy) * 1.4
        blob = max(0.0, 1.0 - (dx * dx + dy * dy) * 9.0)
        cj = (ci + 3) % 8
        for c in range(3):
            out[c] = (_FEED[ci, c] * (1.0 - blob) + _FEED[cj, c] * blob) * 1.25
        return
    ll = 0.35 + 0.55 * _hash(sid + 2, k)
    if (0.70 < f < 0.76 and sx < ll + 0.3) or (0.82 < f < 0.87 and sx < ll):
        out[0] = 0.30
        out[1] = 0.32
        out[2] = 0.36
        return
    out[0] = 0.92
    out[1] = 0.95
    out[2] = 1.00


@njit(cache=True, fastmath=True)
def wall_pass(acc, cov, bedd, bedi, beda, lab, V, nv, kind, base, spec, rim, ldir, bevw, bevk, domek, cen, hs, axu,
              alpha, inset, bedx, seed, tfeed, write_lab):
    """Bedded wall tiles. acc (h,w,3) linear premult colour, cov (h,w), bedd (h,w) px distance to nearest tile
    (BIG = no bed), bedi (h,w) index of that tile, lab (h,w) tile index where coverage >= 0.5 (if write_lab)."""
    h, w = cov.shape
    M = V.shape[0]
    enx = np.empty(16, np.float32)
    eny = np.empty(16, np.float32)
    ec = np.empty(16, np.float32)
    fc = np.empty(3, np.float32)
    for m in range(M):
        n = nv[m]
        if n < 3:
            continue
        xmin = 1e30
        xmax = -1e30
        ymin = 1e30
        ymax = -1e30
        k = 0
        for e in range(n):
            j = e + 1 if e + 1 < n else 0
            ex = V[m, j, 0] - V[m, e, 0]
            ey = V[m, j, 1] - V[m, e, 1]
            L = math.sqrt(ex * ex + ey * ey)
            x = V[m, e, 0]
            y = V[m, e, 1]
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            if L < 1e-6:
                continue
            enx[k] = ey / L
            eny[k] = -ex / L
            ec[k] = enx[k] * x + eny[k] * y
            k += 1
        if k < 3:
            continue
        bx = bedx[m]
        x0 = max(int(xmin - bx - 1.0), 0)
        x1 = min(int(xmax + bx + 1.0), w - 1)
        y0 = max(int(ymin - bx - 1.0), 0)
        y1 = min(int(ymax + bx + 1.0), h - 1)
        if x0 > x1 or y0 > y1:
            continue
        kd = kind[m]
        cx = cen[m, 0]
        cy = cen[m, 1]
        ihs = 1.0 / max(hs[m], 0.5)
        lx = ldir[m, 0]
        ly = ldir[m, 1]
        bw = bevw[m]
        bk = bevk[m]
        dk = domek[m]
        am = alpha[m]
        ins = inset[m]
        rk = 0.0
        if hs[m] > 7.0 and kd != 2:
            rk = 0.035 * min((hs[m] - 7.0) / 25.0, 1.0)
        sd = seed[m] * 6.2831853
        r1 = 3.0 + 4.0 * math.sin(sd * 3.1)
        r2 = 3.0 + 4.0 * math.cos(sd * 5.3)
        r3 = sd * 7.0
        for py in range(y0, y1 + 1):
            fy = py + 0.5
            for px in range(x0, x1 + 1):
                fx = px + 0.5
                dmax = -1e30
                eb = 0
                for e in range(k):
                    d = enx[e] * fx + eny[e] * fy - ec[e]
                    if d > dmax:
                        dmax = d
                        eb = e
                din = -dmax - ins
                if din < bx:
                    dd = -din if din < 0.0 else 0.0
                    if dd < bedd[py, px]:
                        bedd[py, px] = dd
                        bedi[py, px] = m
                    ab = bx - dd + 0.5
                    if ab > beda[py, px]:
                        beda[py, px] = ab if ab < 1.0 else 1.0
                if din <= -0.5:
                    continue
                a = din + 0.5
                if a > 1.0:
                    a = 1.0
                a *= am
                free = 1.0 - cov[py, px]
                if free <= 0.0:
                    continue
                if a > free:
                    a = free
                dx = fx - cx
                dy = fy - cy
                if kd == 2:
                    u = (dx * axu[m, 0] + dy * axu[m, 1]) * ihs
                    v = (-dx * axu[m, 1] + dy * axu[m, 0]) * ihs
                    _feed(u, v, tfeed, seed[m], fc)
                    r = fc[0] * base[m, 0]
                    g = fc[1] * base[m, 1]
                    b = fc[2] * base[m, 2]
                else:
                    gl = (dx * lx + dy * ly) * ihs
                    sh = 1.0 + dk * gl
                    if rk > 0.0:   # conchoidal fracture ripples on the glass face (close-ups only)
                        uu = (dx * axu[m, 0] + dy * axu[m, 1]) * ihs
                        vv = (-dx * axu[m, 1] + dy * axu[m, 0]) * ihs
                        sh += rk * math.sin(r1 * uu + r2 * vv + r3) + 0.5 * rk * math.sin(r2 * uu * 1.7 - r1 * vv + r3 * 2.0)
                    rf = 0.0
                    if din < bw:
                        tb = 1.0 - din / bw
                        tb2 = tb * tb
                        lit = enx[eb] * lx + eny[eb] * ly
                        if kd == 1:
                            sh = 0.84 - 0.42 * tb2 * max(lit, 0.0) + 0.22 * tb2 * max(-lit, 0.0)
                        else:
                            sh += bk * tb2 * (0.9 * lit - 0.3)
                            if lit > 0.0:
                                rf = tb2 * tb * lit * lit
                    r = base[m, 0] * sh + spec[m, 0] + rim[m, 0] * rf
                    g = base[m, 1] * sh + spec[m, 1] + rim[m, 1] * rf
                    b = base[m, 2] * sh + spec[m, 2] + rim[m, 2] * rf
                acc[py, px, 0] += r * a
                acc[py, px, 1] += g * a
                acc[py, px, 2] += b * a
                cov[py, px] += a
                if write_lab and a >= 0.5:
                    lab[py, px] = m


@njit(cache=True, fastmath=True)
def composite(out, acc, cov, bedd, bedi, beda, bg, mortar, mirr, memit, aow, aolo, has_bg):
    """linear frame = tiles + mortar (lit by the nearest tile's irradiance, darker in narrow joints) + bg.
    The outer rim of the bed is antialiased against bg (beda = bed coverage written by wall_pass)."""
    h, w = cov.shape
    for y in range(h):
        for x in range(w):
            a = cov[y, x]
            if a > 1.0:
                a = 1.0
            r = acc[y, x, 0]
            g = acc[y, x, 1]
            b = acc[y, x, 2]
            if a < 1.0:
                k = 1.0 - a
                d = bedd[y, x]
                ab = 0.0
                if d < 1e8:
                    i = bedi[y, x]
                    ab = beda[y, x]
                    t = d / aow
                    if t > 1.0:
                        t = 1.0
                    ao_lo = aolo[i]
                    ao = ao_lo + (1.0 - ao_lo) * t * t * (3.0 - 2.0 * t)
                    r += k * ab * (mortar[0] * mirr[i, 0] * ao + memit[i, 0])
                    g += k * ab * (mortar[1] * mirr[i, 1] * ao + memit[i, 1])
                    b += k * ab * (mortar[2] * mirr[i, 2] * ao + memit[i, 2])
                if has_bg and ab < 1.0:
                    kb = k * (1.0 - ab)
                    r += kb * bg[y, x, 0]
                    g += kb * bg[y, x, 1]
                    b += kb * bg[y, x, 2]
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b


@njit(cache=True, fastmath=True)
def alpha_of(outa, cov, beda):
    h, w = cov.shape
    for y in range(h):
        for x in range(w):
            a = min(cov[y, x], 1.0)
            ab = beda[y, x]
            if ab > 0.0:
                a = a + (1.0 - a) * ab
            outa[y, x] = a


@njit(cache=True, fastmath=True)
def shadow_pass(frame, V, nv, off, soft, dark):
    """multiplicative soft shadows of lifted tiles: polygon V[m] shifted by off[m] (px), edge softness soft[m]
    px, max darkening dark[m] (0..1)."""
    h, w = frame.shape[:2]
    enx = np.empty(16, np.float32)
    eny = np.empty(16, np.float32)
    ec = np.empty(16, np.float32)
    for m in range(V.shape[0]):
        n = nv[m]
        if n < 3 or dark[m] <= 0.003:
            continue
        ox = off[m, 0]
        oy = off[m, 1]
        xmin = 1e30
        xmax = -1e30
        ymin = 1e30
        ymax = -1e30
        k = 0
        for e in range(n):
            j = e + 1 if e + 1 < n else 0
            x = V[m, e, 0] + ox
            y = V[m, e, 1] + oy
            ex = V[m, j, 0] - V[m, e, 0]
            ey = V[m, j, 1] - V[m, e, 1]
            L = math.sqrt(ex * ex + ey * ey)
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            if L < 1e-6:
                continue
            enx[k] = ey / L
            eny[k] = -ex / L
            ec[k] = enx[k] * x + eny[k] * y
            k += 1
        if k < 3:
            continue
        s = max(soft[m], 0.5)
        x0 = max(int(xmin - s - 1), 0)
        x1 = min(int(xmax + s + 1), w - 1)
        y0 = max(int(ymin - s - 1), 0)
        y1 = min(int(ymax + s + 1), h - 1)
        dk = dark[m]
        for py in range(y0, y1 + 1):
            fy = py + 0.5
            for px in range(x0, x1 + 1):
                fx = px + 0.5
                dmax = -1e30
                for e in range(k):
                    d = enx[e] * fx + eny[e] * fy - ec[e]
                    if d > dmax:
                        dmax = d
                t = 0.5 - dmax / (2.0 * s)
                if t <= 0.0:
                    continue
                if t > 1.0:
                    t = 1.0
                f = 1.0 - dk * t
                frame[py, px, 0] *= f
                frame[py, px, 1] *= f
                frame[py, px, 2] *= f


@njit(cache=True, fastmath=True)
def loose_pass(frame, fa, V, nv, order, kind, base, spec, rim, ldir, bevw, bevk, domek, cen, hs, axu, alpha, seed,
               tfeed, use_fa):
    """loose tiles composited OVER the (linear) frame in the given order (back to front). fa (h,w) alpha
    accumulates coverage if use_fa (for rgba layers)."""
    h, w = frame.shape[:2]
    enx = np.empty(16, np.float32)
    eny = np.empty(16, np.float32)
    ec = np.empty(16, np.float32)
    fc = np.empty(3, np.float32)
    for oi in range(order.shape[0]):
        m = order[oi]
        n = nv[m]
        if n < 3 or alpha[m] <= 0.002:
            continue
        xmin = 1e30
        xmax = -1e30
        ymin = 1e30
        ymax = -1e30
        k = 0
        for e in range(n):
            j = e + 1 if e + 1 < n else 0
            x = V[m, e, 0]
            y = V[m, e, 1]
            ex = V[m, j, 0] - x
            ey = V[m, j, 1] - y
            L = math.sqrt(ex * ex + ey * ey)
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            if L < 1e-6:
                continue
            enx[k] = ey / L
            eny[k] = -ex / L
            ec[k] = enx[k] * x + eny[k] * y
            k += 1
        if k < 3:
            continue
        x0 = max(int(xmin - 1.0), 0)
        x1 = min(int(xmax + 1.0), w - 1)
        y0 = max(int(ymin - 1.0), 0)
        y1 = min(int(ymax + 1.0), h - 1)
        if x0 > x1 or y0 > y1:
            continue
        kd = kind[m]
        cx = cen[m, 0]
        cy = cen[m, 1]
        ihs = 1.0 / max(hs[m], 0.5)
        lx = ldir[m, 0]
        ly = ldir[m, 1]
        bw = bevw[m]
        bk = bevk[m]
        dk = domek[m]
        am = alpha[m]
        rk = 0.0
        if hs[m] > 7.0 and kd != 2:
            rk = 0.035 * min((hs[m] - 7.0) / 25.0, 1.0)
        sd = seed[m] * 6.2831853
        r1 = 3.0 + 4.0 * math.sin(sd * 3.1)
        r2 = 3.0 + 4.0 * math.cos(sd * 5.3)
        r3 = sd * 7.0
        for py in range(y0, y1 + 1):
            fy = py + 0.5
            for px in range(x0, x1 + 1):
                fx = px + 0.5
                dmax = -1e30
                eb = 0
                for e in range(k):
                    d = enx[e] * fx + eny[e] * fy - ec[e]
                    if d > dmax:
                        dmax = d
                        eb = e
                din = -dmax
                if din <= -0.5:
                    continue
                a = din + 0.5
                if a > 1.0:
                    a = 1.0
                a *= am
                dx = fx - cx
                dy = fy - cy
                if kd == 2:
                    u = (dx * axu[m, 0] + dy * axu[m, 1]) * ihs
                    v = (-dx * axu[m, 1] + dy * axu[m, 0]) * ihs
                    _feed(u, v, tfeed, seed[m], fc)
                    r = fc[0] * base[m, 0]
                    g = fc[1] * base[m, 1]
                    b = fc[2] * base[m, 2]
                else:
                    gl = (dx * lx + dy * ly) * ihs
                    sh = 1.0 + dk * gl
                    if rk > 0.0:
                        uu = (dx * axu[m, 0] + dy * axu[m, 1]) * ihs
                        vv = (-dx * axu[m, 1] + dy * axu[m, 0]) * ihs
                        sh += rk * math.sin(r1 * uu + r2 * vv + r3) + 0.5 * rk * math.sin(r2 * uu * 1.7 - r1 * vv + r3 * 2.0)
                    rf = 0.0
                    if din < bw:
                        tb = 1.0 - din / bw
                        tb2 = tb * tb
                        lit = enx[eb] * lx + eny[eb] * ly
                        sh += bk * tb2 * (0.9 * lit - 0.3)
                        if lit > 0.0:
                            rf = tb2 * tb * lit * lit
                    r = base[m, 0] * sh + spec[m, 0] + rim[m, 0] * rf
                    g = base[m, 1] * sh + spec[m, 1] + rim[m, 1] * rf
                    b = base[m, 2] * sh + spec[m, 2] + rim[m, 2] * rf
                ia = 1.0 - a
                frame[py, px, 0] = frame[py, px, 0] * ia + r * a
                frame[py, px, 1] = frame[py, px, 1] * ia + g * a
                frame[py, px, 2] = frame[py, px, 2] * ia + b * a
                if use_fa:
                    fa[py, px] = fa[py, px] * ia + a


@njit(cache=True, fastmath=True)
def encode(out, frame, exposure, hdr):
    """linear -> sRGB float32 (in place into out). hdr: keep values > 1 (encoded with the same curve)."""
    h, w = frame.shape[:2]
    for y in range(h):
        for x in range(w):
            for c in range(3):
                v = frame[y, x, c] * exposure
                if v <= 0.0:
                    out[y, x, c] = 0.0
                elif v < 1.0:
                    out[y, x, c] = ENC[int(v * 4095.0 + 0.5)]
                elif hdr:
                    out[y, x, c] = 1.055 * v ** (1.0 / 2.4) - 0.055
                else:
                    out[y, x, c] = 1.0


@njit(cache=True, fastmath=True)
def decode(out, img):
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            for c in range(3):
                out[y, x, c] = _lut(DEC, img[y, x, c])


# ============================================================ per-tile lighting (linear light)
# materials
M_GLASS, M_GOLD, M_SILVER, M_STONE, M_PEARL, M_SCREEN, M_BACK, M_BED = 0, 1, 2, 3, 4, 5, 6, 7


@njit(cache=True, fastmath=True)
def shade_tiles(P, N, eye, alb, mat, gain, emit, rnd,
                Lp, Lc, Lr, Ld, amb, envd, envc, envp, envb, sparkle,
                base, spec, rim, irr, ltan):
    """P, N (M,3) world centre + tilted unit normal; eye (3,); alb (M,3) linear albedo; mat (M,) codes;
    lights: Lp (K,3) position or direction (Ld[k]=1 directional, pointing TO the light), Lc (K,3) linear colour*power,
    Lr (K,) falloff radius; amb (3,); env lobes envd (J,3) unit dirs, envc (J,3), envp (J,), envb (3,).
    Outputs (M,3): base (body colour incl. metal reflection), spec (flat additive), rim (bevel glint colour),
    irr (irradiance, for the mortar), ltan (tangential direction of the dominant light, scaled by strength)."""
    M = P.shape[0]
    K = Lp.shape[0]
    J = envd.shape[0]
    for i in range(M):
        nx = N[i, 0]
        ny = N[i, 1]
        nz = N[i, 2]
        vx = eye[0] - P[i, 0]
        vy = eye[1] - P[i, 1]
        vz = eye[2] - P[i, 2]
        vl = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9
        vx /= vl
        vy /= vl
        vz /= vl
        ndv = nx * vx + ny * vy + nz * vz
        if ndv < 0.0:
            ndv = 0.0
        ir = amb[0]
        ig = amb[1]
        ib = amb[2]
        s1r = 0.0  # sharp lobe
        s1g = 0.0
        s1b = 0.0
        s2r = 0.0  # metal lobe
        s2g = 0.0
        s2b = 0.0
        s3r = 0.0  # broad rim lobe
        s3g = 0.0
        s3b = 0.0
        s4r = 0.0  # leaf glint (metals)
        s4g = 0.0
        s4b = 0.0
        best = -1.0
        bx = 0.0
        by = 0.0
        bz = 0.0
        m = mat[i]
        for k in range(K):
            if Ld[k] > 0.5:
                lx = Lp[k, 0]
                ly = Lp[k, 1]
                lz = Lp[k, 2]
                ll = math.sqrt(lx * lx + ly * ly + lz * lz) + 1e-9
                lx /= ll
                ly /= ll
                lz /= ll
                att = 1.0
            else:
                lx = Lp[k, 0] - P[i, 0]
                ly = Lp[k, 1] - P[i, 1]
                lz = Lp[k, 2] - P[i, 2]
                d2 = lx * lx + ly * ly + lz * lz
                ll = math.sqrt(d2) + 1e-9
                lx /= ll
                ly /= ll
                lz /= ll
                att = 1.0 / (1.0 + d2 / (Lr[k] * Lr[k]))
            er = Lc[k, 0] * att
            eg = Lc[k, 1] * att
            eb = Lc[k, 2] * att
            ndl = nx * lx + ny * ly + nz * lz
            wrap = (ndl + 0.12) / 1.12
            if wrap < 0.0:
                wrap = 0.0
            ir += er * wrap
            ig += eg * wrap
            ib += eb * wrap
            hx = lx + vx
            hy = ly + vy
            hz = lz + vz
            hl = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
            ndh = (nx * hx + ny * hy + nz * hz) / hl
            if ndh < 0.0:
                ndh = 0.0
            if ndl > 0.0:
                p4 = 0.0
                if m == 1 or m == 2:
                    p1 = ndh ** 1500
                    p2 = ndh ** 36
                    p3 = ndh ** 6
                    p4 = ndh ** 280
                elif m == 4:
                    p1 = ndh ** 60
                    p2 = 0.0
                    p3 = ndh ** 8
                else:
                    p1 = ndh ** 380
                    p2 = 0.0
                    p3 = ndh ** 10
                s1r += er * p1
                s1g += eg * p1
                s1b += eb * p1
                s2r += er * p2
                s2g += eg * p2
                s2b += eb * p2
                s3r += er * p3 * ndl
                s3g += eg * p3 * ndl
                s3b += eb * p3 * ndl
                s4r += er * p4
                s4g += eg * p4
                s4b += eb * p4
                st = (er + eg + eb) * ndl
                if st > best:
                    best = st
                    # tangential component of L
                    tx = lx - ndl * nx
                    ty = ly - ndl * ny
                    tz = lz - ndl * nz
                    bx = tx
                    by = ty
                    bz = tz
        irr[i, 0] = ir
        irr[i, 1] = ig
        irr[i, 2] = ib
        ltan[i, 0] = bx
        ltan[i, 1] = by
        ltan[i, 2] = bz
        ar = alb[i, 0]
        ag = alb[i, 1]
        ab = alb[i, 2]
        gn = gain[i]
        if m == 1 or m == 2 or m == 4:
            # reflection of the environment
            rx = 2.0 * ndv * nx - vx
            ry = 2.0 * ndv * ny - vy
            rz = 2.0 * ndv * nz - vz
            er_ = envb[0]
            eg_ = envb[1]
            eb_ = envb[2]
            for j in range(J):
                c = rx * envd[j, 0] + ry * envd[j, 1] + rz * envd[j, 2]
                if c > 0.0:
                    f = c ** envp[j]
                    er_ += envc[j, 0] * f
                    eg_ += envc[j, 1] * f
                    eb_ += envc[j, 2] * f
            if m == 2:   # silver: a cooler, more neutral reflection
                mean = (er_ + eg_ + eb_) / 3.0
                er_ = mean + 0.4 * (er_ - mean)
                eg_ = mean + 0.4 * (eg_ - mean)
                eb_ = mean + 0.4 * (eb_ - mean)
            if m == 4:
                hue = 6.2831853 * (rnd[i] + 1.3 * ndv)
                ar = ar * (0.88 + 0.12 * math.cos(hue))
                ag = ag * (0.88 + 0.12 * math.cos(hue + 2.094))
                ab = ab * (0.88 + 0.12 * math.cos(hue + 4.189))
                base[i, 0] = gn * ar * (0.55 * ir + 0.8 * er_ + 0.6 * s3r) + emit[i, 0]
                base[i, 1] = gn * ag * (0.55 * ig + 0.8 * eg_ + 0.6 * s3g) + emit[i, 1]
                base[i, 2] = gn * ab * (0.55 * ib + 0.8 * eb_ + 0.6 * s3b) + emit[i, 2]
                spec[i, 0] = gn * 0.5 * sparkle * s1r
                spec[i, 1] = gn * 0.5 * sparkle * s1g
                spec[i, 2] = gn * 0.5 * sparkle * s1b
                rim[i, 0] = gn * 0.4 * s3r
                rim[i, 1] = gn * 0.4 * s3g
                rim[i, 2] = gn * 0.4 * s3b
            else:
                kg = 0.9
                kl = 3.2 * sparkle
                base[i, 0] = gn * ar * (0.08 * ir + er_ + kg * s2r + kl * s4r) + emit[i, 0]
                base[i, 1] = gn * ag * (0.08 * ig + eg_ + kg * s2g + kl * s4g) + emit[i, 1]
                base[i, 2] = gn * ab * (0.08 * ib + eb_ + kg * s2b + kl * s4b) + emit[i, 2]
                spec[i, 0] = gn * 0.9 * sparkle * s1r
                spec[i, 1] = gn * 0.9 * sparkle * s1g
                spec[i, 2] = gn * 0.9 * sparkle * s1b
                rim[i, 0] = gn * ar * 1.1 * s3r
                rim[i, 1] = gn * ag * 1.1 * s3g
                rim[i, 2] = gn * ab * 1.1 * s3b
        elif m == 5:
            base[i, 0] = gn * ar + emit[i, 0]
            base[i, 1] = gn * ag + emit[i, 1]
            base[i, 2] = gn * ab + emit[i, 2]
            spec[i, 0] = 0.0
            spec[i, 1] = 0.0
            spec[i, 2] = 0.0
            rim[i, 0] = 0.0
            rim[i, 1] = 0.0
            rim[i, 2] = 0.0
        else:
            kd = 1.0
            ks = 0.9
            kr = 0.42
            if m == 3:     # stone: matte
                ks = 0.0
                kr = 0.12
            elif m == 6:   # back of a tessera: rough
                ks = 0.0
                kr = 0.1
            elif m == 7:   # setting bed
                ks = 0.0
                kr = 0.0
            fres = 0.35 + 0.65 * (1.0 - ndv) ** 5
            base[i, 0] = gn * ar * ir * kd + emit[i, 0]
            base[i, 1] = gn * ag * ir * kd + emit[i, 1]
            base[i, 2] = gn * ab * ir * kd + emit[i, 2]
            spec[i, 0] = gn * ks * fres * sparkle * s1r
            spec[i, 1] = gn * ks * fres * sparkle * s1g
            spec[i, 2] = gn * ks * fres * sparkle * s1b
            rim[i, 0] = gn * kr * s3r
            rim[i, 1] = gn * kr * s3g
            rim[i, 2] = gn * kr * s3b


# ============================================================ tile albedo (sRGB cartoon colour -> linear albedo)
@njit(cache=True, fastmath=True)
def _s2l(c):
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


@njit(cache=True, fastmath=True)
def albedo(rgb, mat, jit, rnd, idx, melts, var, gold, silver, pearl, back, bed, gold_ref, silver_ref, out):
    """per visible tile: smalti melts (Oklab quantisation with dithered lightness) + batch variation; metals take
    the cartoon's value only; sockets take the bed colour."""
    qL = 0.05 * melts
    qc = 0.02 * melts
    for m in range(idx.shape[0]):
        i = idx[m]
        mt = mat[i]
        r = _s2l(min(max(rgb[i, 0], 0.0), 1.0))
        g = _s2l(min(max(rgb[i, 1], 0.0), 1.0))
        b = _s2l(min(max(rgb[i, 2], 0.0), 1.0))
        if mt == 7:   # setting bed (socket)
            k = 0.88 + 0.07 * jit[i, 0]
            out[m, 0] = bed[0] * k
            out[m, 1] = bed[1] * k
            out[m, 2] = bed[2] * k
            continue
        if mt == 1 or mt == 2 or mt == 4:
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if mt == 1:
                k = min(max(lum / gold_ref, 0.35), 1.6) ** 0.7
                v = min(max(1.0 + 0.10 * jit[i, 0], 0.7), 1.25)
                bb = min(max(1.0 + 0.25 * jit[i, 1], 0.5), 1.6)
                if rnd[i] < 0.035:     # reversed / lost-leaf tesserae
                    f = 0.8 + 0.4 * rnd[i] / 0.035
                    out[m, 0] = 0.20 * f
                    out[m, 1] = 0.16 * f
                    out[m, 2] = 0.07 * f
                else:
                    out[m, 0] = gold[0] * k * v
                    out[m, 1] = gold[1] * k * v
                    out[m, 2] = gold[2] * k * v * bb
            elif mt == 2:
                k = min(max(lum / silver_ref, 0.35), 1.5) ** 0.7
                v = min(max(1.0 + 0.08 * jit[i, 0], 0.75), 1.2)
                out[m, 0] = silver[0] * k * v
                out[m, 1] = silver[1] * k * v
                out[m, 2] = silver[2] * k * v
            else:
                v = min(max(0.9 + 0.08 * jit[i, 0], 0.7), 1.1)
                out[m, 0] = pearl[0] * v
                out[m, 1] = pearl[1] * v
                out[m, 2] = pearl[2] * v
            continue
        # Oklab
        l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1.0 / 3.0)
        m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1.0 / 3.0)
        s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1.0 / 3.0)
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        if melts > 0.0:
            L = qL * math.floor(L / qL + 0.5 + (rnd[i] - 0.5) * 0.85)
            A = qc * math.floor(A / qc + 0.5)
            B = qc * math.floor(B / qc + 0.5)
        L += var * 0.42 * jit[i, 0]
        A += var * 0.10 * jit[i, 1]
        B += var * 0.10 * jit[i, 2]
        L = min(max(L, 0.0), 1.0)
        l_ = L + 0.3963377774 * A + 0.2158037573 * B
        m_ = L - 0.1055613458 * A - 0.0638541728 * B
        s_ = L - 0.0894841775 * A - 1.2914855480 * B
        l3 = l_ * l_ * l_
        m3 = m_ * m_ * m_
        s3 = s_ * s_ * s_
        r = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
        g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
        b = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3
        if mt == 6:   # the back of a tessera: its own glass dirtied with mortar
            r = r * 0.62 + back[0] * 0.38
            g = g * 0.62 + back[1] * 0.38
            b = b * 0.62 + back[2] * 0.38
        out[m, 0] = max(r, 0.0)
        out[m, 1] = max(g, 0.0)
        out[m, 2] = max(b, 0.0)

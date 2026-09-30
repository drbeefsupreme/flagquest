"""m09 helper: the dome of Flagistan as an exact hyperbolic {8,3} tiling (Poincare disk = the dome seen from the floor).

Geometry. A Byzantine dome is a hemisphere; a camera standing at the "south pole" of the dome's sphere (floor level
under the centre) sees it through a central projection that IS the stereographic projection: the hemisphere model
of the hyperbolic plane seen from there is exactly the Poincare disk. So every camera orientation at that point is a
3x3 homography pixel -> w (the plane z = 1 of the dome's sphere); |w| < 1 is the dome, |w| > 1 the drum, pendentives
and arches (architecture).

Tiling. {8,3}: regular octagons, three at every vertex. Each octagon is a little Pentecost dome: an oculus at its
centre, 8 rays, 8 peoples standing on its 8 edges (feet on the edge, head toward the oculus), small round windows
(lamps) at the vertices. The orientation-preserving triangle group D+(2,8,3) acts simply transitively on the 8 "sectors" of every
tile, so one sector texture (the fundamental domain: centre O, edge midpoint M on the positive real axis, vertices
V, V' at angles +-pi/8) tiles the whole plane without mirroring anybody. The fold: rotate into sector 0, and while
the point lies beyond edge 0, apply the half-turn about M (H0(z) = (c z - 1)/(z - c)); track the inverse map G
(sector 0 -> the copy) as an SU(1,1)-normalised 2x2 complex matrix.

Variants. Every copy gets its "people" from (sector index inside its tile + a hash of the tile centre) mod 8, so each
tile shows all eight peoples, in a different order per tile (the enumeration for the flags uses the same function).

Kernel. render_frame(): per pixel homography -> dome or architecture; dome: zoom, camera Moebius, fold, mosaic
sheet sample (tessera id, bevel, per-tessera colour/gold/tilt/ray/emission, candle glints rotated into the copy's
own frame, LOD from the exact conformal footprint), beat rings, ray fronts, per-tessera dissolve into light.
splat_tiles(): the loose tesserae of the dissolve (rotated bevelled squares, anti-aliased).
"""
import math

import numpy as np
from numba import njit

P, Q = 8, 3
AL = math.pi / P                                            # half sector angle (22.5 deg)
OM = math.acosh(math.cos(math.pi / Q) / math.sin(math.pi / P))   # centre -> edge midpoint (hyperbolic)
OV = math.acosh(1.0 / (math.tan(math.pi / P) * math.tan(math.pi / Q)))   # centre -> vertex
XM = math.tanh(OM / 2)                                      # Euclidean radius of the edge midpoint (0.3646)
RV = math.tanh(OV / 2)                                      # Euclidean radius of a vertex (0.4065)
CC = (XM + 1 / XM) / 2                                      # edge 0 geodesic: circle centre (CC, 0), radius RR
RR = (1 / XM - XM) / 2
STEP = 2 * OM                                               # centre -> neighbouring centre (one tile step)
NVAR = 8                                                    # peoples (variants)


# ------------------------------------------------------------------ Moebius helpers (python side, SU(1,1) pairs)
def mob_translate(d, theta=0.0):
    """hyperbolic translation by distance d along the geodesic through 0 in direction theta (0 -> tanh(d/2) e^it)."""
    e = complex(math.cos(theta), math.sin(theta))
    return complex(math.cosh(d / 2), 0.0), math.sinh(d / 2) * e


def mob_rotate(phi):
    return complex(math.cos(phi / 2), math.sin(phi / 2)), 0j


def mob_compose(m1, m2):
    """m1 o m2 (apply m2 first)"""
    a1, b1 = m1
    a2, b2 = m2
    return a1 * a2 + b1 * b2.conjugate(), a1 * b2 + b1 * a2.conjugate()


def mob_inverse(m):
    a, b = m
    return a.conjugate(), -b


def mob_apply(m, z):
    a, b = m
    return (a * z + b) / (b.conjugate() * z + a.conjugate())


def mob_apply_v(m, z):
    """vectorised (numpy complex array)"""
    a, b = m
    return (a * z + b) / (np.conj(b) * z + np.conj(a))


# ------------------------------------------------------------------ shared numba helpers
@njit(cache=True)
def _hash2(ix, iy):
    h = (ix * 374761393 + iy * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


@njit(cache=True)
def copy_info(Ar, Ai, Br, Bi, Cr, Ci, Dr, Di):
    """(variant, seed, tile_hash) of the sector copy G = [[A, B], [C, D]] (det 1). Identical for the per-pixel fold
    and the python enumeration (both call this)."""
    dd = Dr * Dr + Di * Di
    cx = (Br * Dr + Bi * Di) / dd                  # tile centre G(0) = B / D
    cy = (Bi * Dr - Br * Di) / dd
    nn = 1.0 - (cx * cx + cy * cy)
    if nn < 1e-12:
        nn = 1e-12
    hX = 2.0 * cx / nn
    hY = 2.0 * cy / nn
    th = _hash2(int(round(hX * 7.31)) + 100000, int(round(hY * 7.31)) + 100000)
    phi = -2.0 * math.atan2(Di, Dr)                # rotation of the copy in its tile's canonical frame
    j = int(math.floor(phi / (2.0 * math.pi / 8.0) + 0.5)) % 8
    if j < 0:
        j += 8
    var = (j + th % 8) % 8
    seed = (th * 8 + j) & 0x7FFFFFFF
    return var, seed, th


@njit(cache=True)
def hash01f(x, y):
    """continuous hash in [0,1) of a real 2-vector (stable under tiny numerical noise)"""
    v = math.sin(x * 12.9898 + y * 78.233) * 43758.5453
    return v - math.floor(v)


@njit(cache=True, fastmath=True)
def fold(x, y):
    """world point (x, y) in the disk -> (zx, zy) in sector 0 of the central tile, the matrix G (8 floats, det 1)
    with world = G(zeta), and the number of half-turns."""
    al2 = 2.0 * math.pi / 8.0
    c = _C_CC                      # module globals (frozen by numba at compile time)
    r = _C_RR
    ir = 1.0 / r
    Ar, Ai, Br, Bi, Cr, Ci, Dr, Di = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
    nh = 0
    for it in range(80):
        ang = math.atan2(y, x)
        k = int(math.floor(ang / al2 + 0.5))
        if k != 0:
            phi = k * al2
            cp = math.cos(phi)
            sp = math.sin(phi)
            x, y = x * cp + y * sp, y * cp - x * sp
            ch = math.cos(0.5 * phi)
            sh = math.sin(0.5 * phi)
            # G <- G o diag(e^{i phi/2}, e^{-i phi/2})
            Ar, Ai = Ar * ch - Ai * sh, Ar * sh + Ai * ch
            Cr, Ci = Cr * ch - Ci * sh, Cr * sh + Ci * ch
            Br, Bi = Br * ch + Bi * sh, Bi * ch - Br * sh
            Dr, Di = Dr * ch + Di * sh, Di * ch - Dr * sh
        dx = x - c
        d2 = dx * dx + y * y
        if d2 < r * r:
            # zeta <- H0(zeta) = (c z - 1)/(z - c)
            nr = c * x - 1.0
            ni = c * y
            q = dx * dx + y * y
            x = (nr * dx + ni * y) / q
            y = (ni * dx - nr * y) / q
            # G <- G o H0n, H0n = (-i/r) [[c, -1], [1, -c]]
            pAr = Ar * c + Br
            pAi = Ai * c + Bi
            pBr = -Ar - Br * c
            pBi = -Ai - Bi * c
            pCr = Cr * c + Dr
            pCi = Ci * c + Di
            pDr = -Cr - Dr * c
            pDi = -Ci - Di * c
            # multiply by -i/r: (u + iv)(-i) = v - iu
            Ar, Ai = pAi * ir, -pAr * ir
            Br, Bi = pBi * ir, -pBr * ir
            Cr, Ci = pCi * ir, -pCr * ir
            Dr, Di = pDi * ir, -pDr * ir
            nh += 1
        else:
            break
    return x, y, Ar, Ai, Br, Bi, Cr, Ci, Dr, Di, nh


_C_CC = CC
_C_RR = RR


# ------------------------------------------------------------------ enumeration (flags, particles)
def _mat_rot(phi):
    return np.array([[complex(math.cos(phi / 2), math.sin(phi / 2)), 0], [0, complex(math.cos(phi / 2), -math.sin(phi / 2))]],
                    complex)


def _mat_h0():
    return np.array([[CC, -1.0], [1.0, -CC]], complex) * (-1j / RR)


def enumerate_sectors(rho_max=9.5, centre=0j):
    """BFS over tiles within hyperbolic distance rho_max of `centre` (world). Returns an (N, 2, 2) complex array
    of sector copies G (sector 0 -> copy), det 1."""
    H0 = _mat_h0()
    R = [_mat_rot(2 * math.pi * k / P) for k in range(P)]
    tc = math.tanh(rho_max / 2)

    def dist_ok(z):
        u = (z - centre) / (1 - np.conj(centre) * z)
        return abs(u) < tc

    def key(z):
        n = 1 - abs(z) ** 2
        return (int(round(2 * z.real / n * 7.31)), int(round(2 * z.imag / n * 7.31)))

    I = np.eye(2, dtype=complex)
    seen = {key(0j)}
    tiles = [I]
    front = [I]
    while front:
        nf = []
        for G in front:
            for k in range(P):
                N = G @ R[k] @ H0
                cz = N[0, 1] / N[1, 1]
                if not dist_ok(cz):
                    continue
                kk = key(cz)
                if kk in seen:
                    continue
                seen.add(kk)
                N = N / np.sqrt(np.linalg.det(N))
                tiles.append(N)
                nf.append(N)
        front = nf
    T = np.array(tiles)
    out = np.empty((len(T) * P, 2, 2), complex)
    for j in range(P):
        out[j::P] = T @ R[j]
    return out


def sector_table(Gs):
    """per copy: (variant, seed) arrays computed with the shared copy_info"""
    n = len(Gs)
    var = np.empty(n, np.int64)
    seed = np.empty(n, np.int64)
    for i in range(n):
        A, B, C, D = Gs[i, 0, 0], Gs[i, 0, 1], Gs[i, 1, 0], Gs[i, 1, 1]
        v, s, _ = copy_info(A.real, A.imag, B.real, B.imag, C.real, C.imag, D.real, D.imag)
        var[i] = v
        seed[i] = s
    return var, seed


# ------------------------------------------------------------------ the per-pixel kernel
@njit(cache=True, fastmath=True)
def _bilin5(atl, k, lv, lvl, cx, cy, out):
    """bilinear sample of mip level lv (row in lvl: x0, y0, w, h) of atlas atl[k] at level-0 texel coords (cx, cy)"""
    sc = 1.0 / (1 << lv)
    x0 = lvl[lv, 0]
    y0 = lvl[lv, 1]
    w = lvl[lv, 2]
    h = lvl[lv, 3]
    fx = cx * sc - 0.5
    fy = cy * sc - 0.5
    if fx < 0.0:
        fx = 0.0
    if fy < 0.0:
        fy = 0.0
    if fx > w - 1.001:
        fx = w - 1.001
    if fy > h - 1.001:
        fy = h - 1.001
    ix = int(fx)
    iy = int(fy)
    tx = fx - ix
    ty = fy - iy
    for ch in range(5):
        a = atl[k, y0 + iy, x0 + ix, ch]
        b = atl[k, y0 + iy, x0 + ix + 1, ch]
        c = atl[k, y0 + iy + 1, x0 + ix, ch]
        d = atl[k, y0 + iy + 1, x0 + ix + 1, ch]
        out[ch] = ((a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty) * (1.0 / 255.0)


@njit(cache=True, fastmath=True)
def _gold(gold3, gl):
    gk = gl * 1.2
    if gk > 1.0:
        gk = 1.0
    gk2 = gl - 0.8
    if gk2 < 0.0:
        gk2 = 0.0
    if gk2 > 1.0:
        gk2 = 1.0
    r = gold3[0, 0] + (gold3[1, 0] - gold3[0, 0]) * gk + (gold3[2, 0] - gold3[1, 0]) * gk2 * 1.4
    g = gold3[0, 1] + (gold3[1, 1] - gold3[0, 1]) * gk + (gold3[2, 1] - gold3[1, 1]) * gk2 * 1.4
    b = gold3[0, 2] + (gold3[1, 2] - gold3[0, 2]) * gk + (gold3[2, 2] - gold3[1, 2]) * gk2 * 1.4
    return r, g, b


@njit(cache=True, fastmath=True)
def _shade(k, cx, cy, fpt, lab, rim, uu, vv, atl, lvl, nlev,
           Tcol, Tgold, Tgt, Ttilt, Tray, Temit, Trnd, Tgrp,
           lxr, lyr, li, illum, sparkle, glint_gain, ray_front, E, gold3, mortar3, t_now, tdis, dis_on,
           jit, flare, tmp):
    """colour of sheet k at level-0 texel (cx, cy) with footprint fpt (texels per pixel).
    Returns (r, g, b, alpha) - alpha < 1 where tesserae have dissolved (the light shows through)."""
    H = lab.shape[1]
    Wd = lab.shape[2]
    lodw = 0.0
    lr = 0.0
    lg = 0.0
    lb = 0.0
    la = 1.0
    if fpt > 1.25:
        # ---------------- LOD colour (mip): averages of tile colours, gold fraction, emission
        lf = math.log2(fpt) - 0.35
        if lf < 1.0:
            lf = 1.0
        if lf > nlev - 1.0:
            lf = nlev - 1.0
        l0 = int(lf)
        l1 = l0 + 1
        if l1 > nlev - 1:
            l1 = nlev - 1
        tt = lf - l0
        _bilin5(atl, k, l0, lvl, cx, cy, tmp)
        a0, a1, a2, a3, a4 = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]
        _bilin5(atl, k, l1, lvl, cx, cy, tmp)
        m0 = a0 + (tmp[0] - a0) * tt
        m1 = a1 + (tmp[1] - a1) * tt
        m2 = a2 + (tmp[2] - a2) * tt
        m3 = a3 + (tmp[3] - a3) * tt
        m4 = a4 + (tmp[4] - a4) * tt
        gcr, gcg, gcb = _gold(gold3, 0.74 * li)
        sp = (0.06 * sparkle * li * glint_gain + flare * 0.45) * m3
        lr = (m0 + m3 * gcr) * illum + sp + m4 * E[1]
        lg = (m1 + m3 * gcg) * illum + sp * 0.9 + m4 * E[1] * 0.96
        lb = (m2 + m3 * gcb) * illum + sp * 0.6 + m4 * E[1] * 0.88
        if dis_on and t_now > tdis:
            la = 1.0 - (t_now - tdis) / 0.7
            if la < 0.0:
                la = 0.0
        lodw = (math.log2(fpt) - 0.35) / 0.9
        if lodw > 1.0:
            lodw = 1.0
        if lodw < 0.0:
            lodw = 0.0
        if lodw >= 1.0:
            return lr, lg, lb, la
    # ---------------- level 0: this pixel's tessera
    ix = int(cx)
    iy = int(cy)
    if ix < 0:
        ix = 0
    if iy < 0:
        iy = 0
    if ix > Wd - 1:
        ix = Wd - 1
    if iy > H - 1:
        iy = H - 1
    i = lab[k, iy, ix]
    rr = 0.0
    rg = 0.0
    rb = 0.0
    alpha = 1.0
    if i < 0:
        rr = mortar3[0] * 0.85 * illum
        rg = mortar3[1] * 0.85 * illum
        rb = mortar3[2] * 0.85 * illum
        if dis_on and t_now > tdis:
            alpha = 1.0 - (t_now - tdis) / 0.25
            if alpha < 0.0:
                alpha = 0.0
    else:
        rim_v = rim[k, iy, ix] * (1.0 / 255.0)
        u = uu[k, iy, ix] * (1.0 / 127.0)
        v = vv[k, iy, ix] * (1.0 / 127.0)
        tx = Ttilt[i, 0] + 0.5 * (jit - 0.5)
        ty = Ttilt[i, 1] + 0.5 * (hash01f(jit * 7.0, Trnd[i]) - 0.5)
        facing = 0.66 + 0.40 * (tx * lxr + ty * lyr)
        if facing < 0.0:
            facing = 0.0
        if facing > 1.5:
            facing = 1.5
        g = Tgold[i]
        gl = facing * li
        if gl > 1.6:
            gl = 1.6
        gcr, gcg, gcb = _gold(gold3, gl)
        br = Tcol[i, 0] * (1.0 - g) + gcr * g * Tgt[i, 0]
        bg = Tcol[i, 1] * (1.0 - g) + gcg * g * Tgt[i, 1]
        bb = Tcol[i, 2] * (1.0 - g) + gcb * g * Tgt[i, 2]
        fk = 0.9 + 0.2 * facing
        gx = facing - 0.82
        if gx < 0.0:
            gx = 0.0
        glint = gx * gx * 2.0 * sparkle * li * (1.0 + 2.5 * g) * glint_gain + flare * (0.3 + 1.2 * g) * (0.4 + facing)
        shade = 0.72 + 0.28 * rim_v
        lit = -(u * lxr + v * lyr) * 0.8
        if lit < 0.0:
            lit = 0.0
        if lit > 1.0:
            lit = 1.0
        lit *= (1.0 - rim_v)
        il = illum
        # rays: silver-white glass that lights up as the front passes
        ry = Tray[i]
        if ry >= 0.0:
            q = (ray_front - ry) / 0.08
            if q > 0.0:
                on = q if q < 1.0 else 1.0
                fr = math.exp(-(q - 0.6) * (q - 0.6) * 2.0)
                il = il + on * 0.8 + fr * 0.9
                glint += fr * 0.8
        gw = glint * (0.35 + 0.65 * lit) * 0.9
        rr = br * fk * shade * il + gw
        rg = bg * fk * shade * il + gw * 0.92
        rb = bb * fk * shade * il + gw * 0.72
        em = Temit[i]
        if em > 0.0:
            # glass lit from behind: at night only its light shows
            gi = Tgrp[i]
            e = em * E[gi]
            if gi <= 2:
                e += em * flare * 1.8          # lamps and oculi ignite as a ring of light passes
            rr = rr * 0.22 + Tcol[i, 0] * e * shade
            rg = rg * 0.22 + Tcol[i, 1] * e * shade
            rb = rb * 0.22 + Tcol[i, 2] * e * shade
        if dis_on:
            td = tdis + 0.35 * hash01f(Trnd[i] * 3.1 + jit, jit * 5.7)
            if t_now > td:
                pgr = (t_now - td) / 0.3
                m = u if u > 0 else -u
                mv = v if v > 0 else -v
                if mv > m:
                    m = mv
                if m > 1.0 - pgr:
                    alpha = 0.0
                else:
                    # a tessera flares as it lets go
                    rr += 0.6 * pgr
                    rg += 0.55 * pgr
                    rb += 0.4 * pgr
    if lodw > 0.0:
        rr = rr * (1.0 - lodw) + lr * lodw
        rg = rg * (1.0 - lodw) + lg * lodw
        rb = rb * (1.0 - lodw) + lb * lodw
        alpha = alpha * (1.0 - lodw) + la * lodw
    return rr, rg, rb, alpha


@njit(cache=True, fastmath=True)
def render_frame(out, Hm, prm, ma_r, ma_i, mb_r, mb_i, zref_r, zref_i,
                 s_lab, s_rim, s_u, s_v, s_atl, s_lvl,
                 a_lab, a_rim, a_u, a_v, a_atl, a_lvl,
                 Tcol, Tgold, Tgt, Ttilt, Tray, Temit, Trnd, Tgrp,
                 E, rings, gold3, mortar3, bgcol, avg_dome):
    """out (h, w, 3) float32. prm (float64):
      0 zoom Z            1 roll (rad)         2 li (light intensity)  3 lx  4 ly (screen light dir)
      5 sparkle           6 illum base         7 illum rim gain        8 illum centre gain
      9 ray_front        10 t_now             11 dis_dome_t0          12 dis_dome_rate (s per hyperbolic unit)
     13 dis_arch_t0      14 dis_arch_rate (s per w unit, from the outside in)  15 dis_arch_rmax
     16 sector x0        17 sector y0         18 sector res (texels / zeta unit)
     19 arch res         20 arch cx (texel of w=0)   21 arch cy     22 bg intensity   23 bg radius
     24 glint gain       25 dome rim glow     26 arch illum        27 dissolve on (0/1)
    rings (nr, 3): [rho, strength, width] light rings travelling outward (hyperbolic radius from the view centre).
    E (16,) emission level per emit group. gold3 (3,3) gold_d/gold/gold_l. bgcol (2,3) light core/rim colours.
    avg_dome (4,): average dome colour rgb + average emission (for sub-pixel tiles)."""
    h, w = out.shape[0], out.shape[1]
    Z = prm[0]
    roll = prm[1]
    li = prm[2]
    lx = prm[3]
    ly = prm[4]
    sparkle = prm[5]
    ib = prm[6]
    irim = prm[7]
    icen = prm[8]
    ray_front = prm[9]
    t_now = prm[10]
    dd_t0 = prm[11]
    dd_rate = prm[12]
    da_t0 = prm[13]
    da_rate = prm[14]
    da_rmax = prm[15]
    sx0 = prm[16]
    sy0 = prm[17]
    sres = prm[18]
    ares = prm[19]
    acx = prm[20]
    acy = prm[21]
    bgi = prm[22]
    bgr = prm[23]
    ggain = prm[24]
    rimglow = prm[25]
    aill = prm[26]
    dis_on = prm[27] > 0.5
    nr = rings.shape[0]
    snl = s_lvl.shape[0]
    anl = a_lvl.shape[0]
    tmp = np.zeros(5)
    cr_ = math.cos(roll)
    sr_ = math.sin(roll)
    # light direction in w-plane coordinates (undo the camera roll)
    lxw = lx * cr_ - ly * sr_
    lyw = lx * sr_ + ly * cr_
    for py in range(h):
        for px in range(w):
            X = px + 0.5
            Y = py + 0.5
            hw_ = Hm[2, 0] * X + Hm[2, 1] * Y + Hm[2, 2]
            if hw_ <= 1e-9:
                out[py, px, 0] = mortar3[0] * 0.3
                out[py, px, 1] = mortar3[1] * 0.3
                out[py, px, 2] = mortar3[2] * 0.3
                continue
            u = (Hm[0, 0] * X + Hm[0, 1] * Y + Hm[0, 2]) / hw_
            v = (Hm[1, 0] * X + Hm[1, 1] * Y + Hm[1, 2]) / hw_
            j00 = (Hm[0, 0] - u * Hm[2, 0]) / hw_
            j01 = (Hm[0, 1] - u * Hm[2, 1]) / hw_
            j10 = (Hm[1, 0] - v * Hm[2, 0]) / hw_
            j11 = (Hm[1, 1] - v * Hm[2, 1]) / hw_
            fp = math.sqrt(abs(j00 * j11 - j01 * j10)) + 1e-12      # w units per pixel
            rr2 = u * u + v * v
            rs = math.sqrt(rr2)
            # the light behind the mosaic (shows where tesserae have gone)
            gb = math.exp(-rr2 / (bgr * bgr))
            bgR = (bgcol[1, 0] + (bgcol[0, 0] - bgcol[1, 0]) * gb) * bgi
            bgG = (bgcol[1, 1] + (bgcol[0, 1] - bgcol[1, 1]) * gb) * bgi
            bgB = (bgcol[1, 2] + (bgcol[0, 2] - bgcol[1, 2]) * gb) * bgi
            cov = (1.0 - rs) / fp + 0.5          # dome coverage (anti-aliased rim)
            if cov > 1.0:
                cov = 1.0
            if cov < 0.0:
                cov = 0.0
            dr = 0.0
            dg = 0.0
            db = 0.0
            ar = 0.0
            ag = 0.0
            ab = 0.0
            if cov > 0.0:
                # ------------------------------------------------ the dome
                wx = u * Z
                wy = v * Z
                wr2 = wx * wx + wy * wy
                if wr2 > 0.9999999:
                    sc_ = math.sqrt(0.9999999 / wr2)
                    wx *= sc_
                    wy *= sc_
                    wr2 = 0.9999999
                rho_pix = 2.0 * math.atanh(math.sqrt(wr2))
                il = ib * (1.0 + irim * math.exp(-(1.0 - rs) * 9.0) + icen * math.exp(-rr2 * 30.0))
                flare = 0.0
                for ri in range(nr):
                    q = (rho_pix - rings[ri, 0]) / rings[ri, 2]
                    if q > -3.0 and q < 3.0:
                        flare += rings[ri, 1] * math.exp(-q * q)
                il += flare * 0.55
                # world point: z = (a w + b)/(conj(b) w + conj(a))
                nr_ = ma_r * wx - ma_i * wy + mb_r
                ni_ = ma_r * wy + ma_i * wx + mb_i
                dr_ = mb_r * wx + mb_i * wy + ma_r
                di_ = mb_r * wy - mb_i * wx - ma_i
                q2 = dr_ * dr_ + di_ * di_
                zx = (nr_ * dr_ + ni_ * di_) / q2
                zy = (ni_ * dr_ - nr_ * di_) / q2
                tdis = 1e9
                if dis_on:
                    nxr = zx - zref_r
                    nxi = zy - zref_i
                    dxr = 1.0 - (zref_r * zx + zref_i * zy)
                    dxi = -(zref_r * zy - zref_i * zx)
                    qq = (nxr * nxr + nxi * nxi) / (dxr * dxr + dxi * dxi)
                    if qq > 0.9999999:
                        qq = 0.9999999
                    tdis = dd_t0 + dd_rate * 2.0 * math.atanh(math.sqrt(qq))
                scale_h = (1.0 - wr2) / (Z * fp)          # ~ 2 x pixels per hyperbolic unit
                da = 1.0
                if scale_h < 2.2:
                    # tiles far below a pixel: the average of all peoples
                    dr = avg_dome[0] * il + avg_dome[3] * E[1] * 0.3 + flare * 0.2
                    dg = avg_dome[1] * il + avg_dome[3] * E[1] * 0.28 + flare * 0.18
                    db = avg_dome[2] * il + avg_dome[3] * E[1] * 0.25 + flare * 0.12
                    if dis_on and t_now > tdis:
                        da = 1.0 - (t_now - tdis) / 0.7
                        if da < 0.0:
                            da = 0.0
                else:
                    fx, fy, Ar, Ai, Br, Bi, Cr, Ci, Dr, Di, nh = fold(zx, zy)
                    var, seed, th = copy_info(Ar, Ai, Br, Bi, Cr, Ci, Dr, Di)
                    jit = ((seed * 2654435761) % 1000003) / 1000003.0
                    # rotation of this copy on screen: arg d(w_v)/d(zeta) = -2 arg(e21 zeta + e22), e = M^-1 G,
                    # M^-1 = [[conj a, -b], [-conj b, a]]
                    e21r = -(mb_r * Ar + mb_i * Ai) + (ma_r * Cr - ma_i * Ci)
                    e21i = -(mb_r * Ai - mb_i * Ar) + (ma_r * Ci + ma_i * Cr)
                    e22r = -(mb_r * Br + mb_i * Bi) + (ma_r * Dr - ma_i * Di)
                    e22i = -(mb_r * Bi - mb_i * Br) + (ma_r * Di + ma_i * Dr)
                    qr = e21r * fx - e21i * fy + e22r
                    qi = e21r * fy + e21i * fx + e22i
                    ang = -2.0 * math.atan2(qi, qr)
                    ca = math.cos(ang)
                    sa = math.sin(ang)
                    lxr = lxw * ca + lyw * sa
                    lyr = -lxw * sa + lyw * ca
                    fz2 = fx * fx + fy * fy
                    fpt = Z * fp * (1.0 - fz2) / (1.0 - wr2) * sres        # texels per pixel
                    tcx = (fx - sx0) * sres
                    tcy = (fy - sy0) * sres
                    dr, dg, db, da = _shade(var, tcx, tcy, fpt, s_lab, s_rim, s_u, s_v, s_atl, s_lvl, snl,
                                            Tcol, Tgold, Tgt, Ttilt, Tray, Temit, Trnd, Tgrp,
                                            lxr, lyr, li, il, sparkle, ggain, ray_front, E, gold3, mortar3,
                                            t_now, tdis, dis_on, jit, flare, tmp)
                    if scale_h < 4.0:
                        kk = (scale_h - 2.2) / 1.8
                        dr = dr * kk + (avg_dome[0] * il) * (1 - kk)
                        dg = dg * kk + (avg_dome[1] * il) * (1 - kk)
                        db = db * kk + (avg_dome[2] * il) * (1 - kk)
                # the circle at infinity glows faintly
                rg_ = rimglow * math.exp(-(1.0 - rs) / (0.004 + 2.0 * fp))
                dr = (dr + rg_) * da + bgR * (1.0 - da)
                dg = (dg + rg_ * 0.9) * da + bgG * (1.0 - da)
                db = (db + rg_ * 0.7) * da + bgB * (1.0 - da)
            if cov < 1.0:
                # ------------------------------------------------ architecture (the w plane itself)
                tcx = acx + u * ares
                tcy = acy + v * ares
                fpt = fp * ares
                jit = 0.5
                tdis = 1e9
                if dis_on:
                    tdis = da_t0 + da_rate * (da_rmax - rs)
                ar, ag, ab, aa = _shade(0, tcx, tcy, fpt, a_lab, a_rim, a_u, a_v, a_atl, a_lvl, anl,
                                        Tcol, Tgold, Tgt, Ttilt, Tray, Temit, Trnd, Tgrp,
                                        lxw, lyw, li, aill, sparkle, ggain, -1.0, E, gold3, mortar3,
                                        t_now, tdis, dis_on, jit, 0.0, tmp)
                ar = ar * aa + bgR * (1.0 - aa)
                ag = ag * aa + bgG * (1.0 - aa)
                ab = ab * aa + bgB * (1.0 - aa)
            out[py, px, 0] = dr * cov + ar * (1.0 - cov)
            out[py, px, 1] = dg * cov + ag * (1.0 - cov)
            out[py, px, 2] = db * cov + ab * (1.0 - cov)


# ------------------------------------------------------------------ loose tesserae (the dissolve)
@njit(cache=True, fastmath=True)
def splat_tiles(buf, xs, ys, hs, ang, cols, alpha):
    """draw rotated square tesserae (pixel coords, half-size hs px) into premultiplied rgba buf (h, w, 4), in
    order, anti-aliased, with a darker bevel rim."""
    h, w = buf.shape[0], buf.shape[1]
    n = xs.shape[0]
    for i in range(n):
        a = alpha[i]
        if a <= 0.003:
            continue
        s = hs[i]
        c = math.cos(ang[i])
        sn = math.sin(ang[i])
        rad = s * 1.42 + 1.0
        x0 = int(xs[i] - rad)
        x1 = int(xs[i] + rad) + 1
        y0 = int(ys[i] - rad)
        y1 = int(ys[i] + rad) + 1
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > w:
            x1 = w
        if y1 > h:
            y1 = h
        r, g, b = cols[i, 0], cols[i, 1], cols[i, 2]
        for py in range(y0, y1):
            for px in range(x0, x1):
                dx = px + 0.5 - xs[i]
                dy = py + 0.5 - ys[i]
                u = dx * c + dy * sn
                v = -dx * sn + dy * c
                au = u if u > 0 else -u
                av = v if v > 0 else -v
                d = (au if au > av else av) - s
                cov = 0.5 - d
                if cov <= 0.0:
                    continue
                if cov > 1.0:
                    cov = 1.0
                rim = -d / (0.3 * s + 0.6)
                if rim > 1.0:
                    rim = 1.0
                if rim < 0.0:
                    rim = 0.0
                k = (0.7 + 0.3 * rim) * cov * a
                ia = 1.0 - cov * a
                buf[py, px, 0] = r * k + buf[py, px, 0] * ia
                buf[py, px, 1] = g * k + buf[py, px, 1] * ia
                buf[py, px, 2] = b * k + buf[py, px, 2] * ia
                buf[py, px, 3] = cov * a + buf[py, px, 3] * ia

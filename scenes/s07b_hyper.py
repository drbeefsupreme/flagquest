"""FLAGISTAN: the exact hyperbolic {5,4} tiling in the Poincare disk (s07b helper).

Per-pixel reflection-group folding (numba): every pixel is mapped screen -> disk (3x3 homography, so the
disk can be seen top-down or as a tilted floor in perspective) -> world disk (Mobius automorphism
z -> (a z + b) / (conj(b) z + conj(a))) and then folded by the {5,4} Coxeter group into the fundamental
triangle (O, M, V) while tracking the inverse of the folding map as an (anti-)Mobius matrix. That yields,
exactly and per pixel: the tile centre (colour + ring flashes), the facet direction (pentagonal pyramid
shading), the hyperbolic distance to the tile edges / facet creases / vertices (analytic anti-aliasing
in screen pixels via the conformal factor and the homography Jacobian).

Vertices of the tiling (where the plain yellow flags stand) are enumerated once by BFS over the group.
"""
import math

import numpy as np
from numba import njit, prange

P, Q = 5, 4
ALPHA = math.pi / P                     # angle at the tile centre
BETA = math.pi / Q                      # angle at a vertex
OM = math.acosh(math.cos(BETA) / math.sin(ALPHA))          # centre -> edge midpoint (hyperbolic)
OV = math.acosh(1.0 / (math.tan(ALPHA) * math.tan(BETA)))  # centre -> vertex
XM = math.tanh(OM / 2)                  # Euclidean radius of the edge midpoint
RV = math.tanh(OV / 2)                  # Euclidean radius of a vertex
CC = (XM + 1 / XM) / 2                  # edge geodesic: circle centre (CC, 0), radius RR (orthogonal to |z|=1)
RR = (1 / XM - XM) / 2
EDGE = 2 * math.acosh(math.cos(ALPHA) / math.sin(BETA))    # hyperbolic edge length


# ------------------------------------------------------------------ Mobius helpers (python side)
def mob_translate(d, theta=0.0):
    """hyperbolic translation by distance d along the geodesic through 0 in direction theta.
    returns (a, b) of z -> (a z + b)/(conj(b) z + conj(a))"""
    t = math.tanh(d / 2)
    e = complex(math.cos(theta), math.sin(theta))
    n = 1 / math.sqrt(1 - t * t)
    return complex(n, 0), n * t * e


def mob_rotate(phi):
    return complex(math.cos(phi / 2), math.sin(phi / 2)), 0j


def mob_compose(m1, m2):
    """m1 o m2 (apply m2 first)"""
    a1, b1 = m1
    a2, b2 = m2
    a = a1 * a2 + b1 * b2.conjugate()
    b = a1 * b2 + b1 * a2.conjugate()
    return a, b


def mob_inverse(m):
    a, b = m
    return a.conjugate(), -b


def mob_apply(m, z):
    a, b = m
    return (a * z + b) / (b.conjugate() * z + a.conjugate())


def mob_about(point, m):
    """conjugate m so that it acts about `point` instead of 0 (e.g. rotate about a vertex)"""
    T = (1 + 0j, point)                   # 0 -> point
    return mob_compose(T, mob_compose(m, mob_inverse(T)))


def mob_log_lerp(m, s):
    """fractional power m^s of an automorphism (for smooth easing of a Mobius motion).
    Uses the SU(1,1) matrix logarithm via eigen-decomposition."""
    a, b = m
    M = np.array([[a, b], [b.conjugate(), a.conjugate()]], complex)
    w, V = np.linalg.eig(M)
    Ms = V @ np.diag(w ** s) @ np.linalg.inv(V)
    # renormalise to SU(1,1) form
    det = np.linalg.det(Ms)
    Ms /= np.sqrt(det)
    return complex(Ms[0, 0]), complex(Ms[0, 1])


# ------------------------------------------------------------------ enumeration of tiles / vertices
def _refl_edge_mat(k):
    """reflection across edge k of the central pentagon, as (matrix, conj=True) acting z -> M conj(z)"""
    # R_k = rot(2k a) o InvC o rot(-2k a); InvC(z) = (c conj z - 1)/(conj z - c)
    e = complex(math.cos(2 * k * ALPHA), math.sin(2 * k * ALPHA))
    Rot = np.array([[e, 0], [0, 1]], complex)
    Rinv = np.array([[e.conjugate(), 0], [0, 1]], complex)
    Ic = np.array([[CC, -1], [1, -CC]], complex) / RR
    # rot o Ic(conj(.)) o rot^-1 : Ic acts on conj(rot^-1 z) = conj(Rinv) conj(z)
    return Rot @ Ic @ np.conj(Rinv)


def enumerate_tiling(rho_max=8.0):
    """BFS over tiles. returns centres (N,) complex, verts (Nv,) complex (world disk coords)."""
    Rk = [_refl_edge_mat(k) for k in range(P)]
    I = np.eye(2, dtype=complex)

    def key(z):
        n = 1 - abs(z) ** 2
        X, Y = 2 * z.real / n, 2 * z.imag / n
        return (int(round(X * 7.31)), int(round(Y * 7.31)))

    def app(M, cf, z):
        zz = z.conjugate() if cf else z
        return (M[0, 0] * zz + M[0, 1]) / (M[1, 0] * zz + M[1, 1])

    tmax = math.tanh(rho_max / 2)
    seen = {key(0j): True}
    front = [(I, False)]
    centres = [0j]
    tiles = [(I, False)]
    while front:
        nf = []
        for M, cf in front:
            for R in Rk:
                # g o R_k  (R_k anti-holomorphic)
                N = M @ (np.conj(R) if cf else R)
                ncf = not cf
                cz = app(N, ncf, 0j)
                if abs(cz) > tmax:
                    continue
                k = key(cz)
                if k in seen:
                    continue
                seen[k] = True
                N = N / np.sqrt(np.linalg.det(N) + 0j)
                centres.append(cz)
                tiles.append((N, ncf))
                nf.append((N, ncf))
        front = nf
    vseen = {}
    verts = []
    V0 = [RV * complex(math.cos((2 * j + 1) * ALPHA), math.sin((2 * j + 1) * ALPHA)) for j in range(P)]
    for M, cf in tiles:
        for v in V0:
            z = app(M, cf, v)
            if abs(z) >= 0.99999:
                continue
            k = key(z)
            if k in vseen:
                continue
            vseen[k] = True
            verts.append(z)
    return np.array(centres), np.array(verts)


# ------------------------------------------------------------------ the per-pixel kernel
@njit(cache=True, fastmath=True)
def _hash2(ix, iy):
    h = (ix * 374761393 + iy * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


@njit(parallel=True, cache=True, fastmath=True)
def render_disk(out, Hm, ma, mb, pal, avg_col, seam_col, prm, rings, lightv):
    """out: (h, w, 4) float32 premultiplied RGBA (written).
    Hm: 3x3 float64 pixel(x+.5, y+.5, 1) -> disk (u, v) homogeneous.
    ma, mb: complex camera Mobius screen-disk -> world-disk.
    pal: (K, 3) tile colours. avg_col, seam_col: (3,).
    prm: float64 params:
      0 seam_hw (hyperbolic half width)   1 seam_glow (strength)      2 seam_glow_w (hyperbolic)
      3 facet_slope                       4 reveal_rho (tiles with centre rho < reveal visible) 5 reveal_soft
      6 vertex_glow strength              7 inner_glow strength       8 time
      9 rim_glow                         10 tile_flash_width (rho)   11 brightness
    rings: (nr, 2) [rho, strength] tile flashes travelling outward from screen centre.
    lightv: (3,) light direction (disk plane x, y, up z), normalised.
    """
    h, w = out.shape[0], out.shape[1]
    al = math.pi / 5.0
    c = 1.7989074399478668
    r = 1.49534878122122
    xm = 0.30355865872664684
    rv = 0.3979754267847908
    ca, sa = math.cos(al), math.sin(al)
    e2r, e2i = math.cos(2 * al), math.sin(2 * al)
    vx0, vy0 = rv * ca, rv * sa
    K = pal.shape[0]
    seam_hw, seam_glow, seam_gw = prm[0], prm[1], prm[2]
    slope = prm[3]
    rev, rev_s = prm[4], prm[5]
    vglow, iglow = prm[6], prm[7]
    rim = prm[9]
    fw = prm[10]
    bright = prm[11]
    sls, slc = math.sin(slope), math.cos(slope)
    lx, ly, lz = lightv[0], lightv[1], lightv[2]
    flat = lz
    nr = rings.shape[0]
    mar, mai = ma.real, ma.imag
    mbr, mbi = mb.real, mb.imag
    for py in prange(h):
        for px in range(w):
            X = px + 0.5
            Y = py + 0.5
            hw_ = Hm[2, 0] * X + Hm[2, 1] * Y + Hm[2, 2]
            if hw_ <= 1e-9:
                out[py, px, 0] = 0.0
                out[py, px, 1] = 0.0
                out[py, px, 2] = 0.0
                out[py, px, 3] = 0.0
                continue
            u = (Hm[0, 0] * X + Hm[0, 1] * Y + Hm[0, 2]) / hw_
            v = (Hm[1, 0] * X + Hm[1, 1] * Y + Hm[1, 2]) / hw_
            # footprint (disk units per pixel) from the homography Jacobian
            j00 = (Hm[0, 0] - u * Hm[2, 0]) / hw_
            j01 = (Hm[0, 1] - u * Hm[2, 1]) / hw_
            j10 = (Hm[1, 0] - v * Hm[2, 0]) / hw_
            j11 = (Hm[1, 1] - v * Hm[2, 1]) / hw_
            fp = math.sqrt(abs(j00 * j11 - j01 * j10)) + 1e-12
            rr2 = u * u + v * v
            rs = math.sqrt(rr2)
            # disk coverage (anti-aliased rim)
            cov = (1.0 - rs) / fp + 0.5
            if cov <= 0.0:
                out[py, px, 0] = 0.0
                out[py, px, 1] = 0.0
                out[py, px, 2] = 0.0
                out[py, px, 3] = 0.0
                continue
            if cov > 1.0:
                cov = 1.0
            lam = (1.0 - rr2) * 0.5 / fp          # pixels per hyperbolic unit here
            if lam < 0:
                lam = 0.0
            tile_px = lam * 1.25
            rim_k = rim * math.exp(-(1.0 - rs) / (0.02 + 3.0 * fp))
            # LOD: sub-pixel tiles -> average colour
            lod = (tile_px - 2.0) / 5.0
            if lod < 0.0:
                lod = 0.0
            if lod > 1.0:
                lod = 1.0
            cr = avg_col[0]
            cg = avg_col[1]
            cb = avg_col[2]
            rsq = rs if rs < 0.9999999 else 0.9999999
            talpha = (rev - 2.0 * math.atanh(rsq)) / rev_s
            if talpha < 0.0:
                talpha = 0.0
            if talpha > 1.0:
                talpha = 1.0
            talpha_pix = talpha
            if lod > 0.0:
                # ---- world point
                # w = (a z + b)/(conj(b) z + conj(a))
                nr_ = mar * u - mai * v + mbr
                ni_ = mar * v + mai * u + mbi
                dr_ = mbr * u + mbi * v + mar
                di_ = mbr * v - mbi * u - mai
                dd = dr_ * dr_ + di_ * di_
                x = (nr_ * dr_ + ni_ * di_) / dd
                y = (ni_ * dr_ - nr_ * di_) / dd
                # inverse-fold tracking matrix  [[A, B], [C2, D]] (complex as pairs), flag f
                Ar, Ai, Br, Bi, Cr, Ci, Dr, Di = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
                f = 0
                for it in range(64):
                    ang = math.atan2(y, x)
                    k = math.floor(ang / (2 * al))
                    if k != 0:
                        psi = k * 2 * al
                        cp, sp = math.cos(psi), math.sin(psi)
                        # z <- z * e^{-i psi}
                        x, y = x * cp + y * sp, y * cp - x * sp
                        # Inv <- Inv o rot(+psi) ; N = diag(e^{i psi}, 1); if f: conj(N)
                        if f == 1:
                            sp = -sp
                        Ar, Ai = Ar * cp - Ai * sp, Ar * sp + Ai * cp
                        Cr, Ci = Cr * cp - Ci * sp, Cr * sp + Ci * cp
                    # reflect across the line at angle al if above it
                    if x * sa - y * ca < 0.0:
                        # z <- e^{2i al} conj(z)
                        x, y = e2r * x + e2i * y, e2i * x - e2r * y
                        er, ei = e2r, e2i
                        if f == 1:
                            ei = -ei
                        Ar, Ai = Ar * er - Ai * ei, Ar * ei + Ai * er
                        Cr, Ci = Cr * er - Ci * ei, Cr * ei + Ci * er
                        f = 1 - f
                    dx = x - c
                    d2 = dx * dx + y * y
                    if d2 < r * r:
                        # invert in the edge circle: z <- c + r^2 (z - c)/|z - c|^2
                        s = r * r / d2
                        x = c + dx * s
                        y = y * s
                        # Inv <- Inv o N, N = [[c, -1], [1, -c]] / r (real)
                        nAr = (Ar * c + Br) / r
                        nAi = (Ai * c + Bi) / r
                        nBr = (-Ar - Br * c) / r
                        nBi = (-Ai - Bi * c) / r
                        nCr = (Cr * c + Dr) / r
                        nCi = (Ci * c + Di) / r
                        nDr = (-Cr - Dr * c) / r
                        nDi = (-Ci - Di * c) / r
                        Ar, Ai, Br, Bi, Cr, Ci, Dr, Di = nAr, nAi, nBr, nBi, nCr, nCi, nDr, nDi
                        f = 1 - f
                    else:
                        break
                # ---- tile centre in world = Inv(0) = B/D
                dd = Dr * Dr + Di * Di
                cwx = (Br * Dr + Bi * Di) / dd
                cwy = (Bi * Dr - Br * Di) / dd
                # ---- to screen disk: z = (conj(a) w - b)/(-conj(b) w + a)
                nr_ = mar * cwx + mai * cwy - mbr
                ni_ = mar * cwy - mai * cwx - mbi
                dr_ = -(mbr * cwx + mbi * cwy) + mar
                di_ = -(mbr * cwy - mbi * cwx) + mai
                dd = dr_ * dr_ + di_ * di_
                csx = (nr_ * dr_ + ni_ * di_) / dd
                csy = (ni_ * dr_ - nr_ * di_) / dd
                # facet midpoints: Inv(xm) and Inv(conj?(xm e^{2i al}))
                # Inv(q) = (A q* + B)/(C q* + D)
                qr, qi = xm, 0.0
                nr_ = Ar * qr - Ai * qi + Br
                ni_ = Ar * qi + Ai * qr + Bi
                dr_ = Cr * qr - Ci * qi + Dr
                di_ = Cr * qi + Ci * qr + Di
                dd = dr_ * dr_ + di_ * di_
                m1x = (nr_ * dr_ + ni_ * di_) / dd
                m1y = (ni_ * dr_ - nr_ * di_) / dd
                qr, qi = xm * e2r, xm * e2i
                if f == 1:
                    qi = -qi
                nr_ = Ar * qr - Ai * qi + Br
                ni_ = Ar * qi + Ai * qr + Bi
                dr_ = Cr * qr - Ci * qi + Dr
                di_ = Cr * qi + Ci * qr + Di
                dd = dr_ * dr_ + di_ * di_
                m2x = (nr_ * dr_ + ni_ * di_) / dd
                m2y = (ni_ * dr_ - nr_ * di_) / dd
                # map facet midpoints to screen
                nr_ = mar * m1x + mai * m1y - mbr
                ni_ = mar * m1y - mai * m1x - mbi
                dr_ = -(mbr * m1x + mbi * m1y) + mar
                di_ = -(mbr * m1y - mbi * m1x) + mai
                dd = dr_ * dr_ + di_ * di_
                s1x = (nr_ * dr_ + ni_ * di_) / dd - csx
                s1y = (ni_ * dr_ - nr_ * di_) / dd - csy
                nr_ = mar * m2x + mai * m2y - mbr
                ni_ = mar * m2y - mai * m2x - mbi
                dr_ = -(mbr * m2x + mbi * m2y) + mar
                di_ = -(mbr * m2y - mbi * m2x) + mai
                dd = dr_ * dr_ + di_ * di_
                s2x = (nr_ * dr_ + ni_ * di_) / dd - csx
                s2y = (ni_ * dr_ - nr_ * di_) / dd - csy
                n1 = math.sqrt(s1x * s1x + s1y * s1y) + 1e-12
                n2 = math.sqrt(s2x * s2x + s2y * s2y) + 1e-12
                sh1 = (s1x / n1 * sls * lx + s1y / n1 * sls * ly + slc * lz) / flat
                sh2 = (s2x / n2 * sls * lx + s2y / n2 * sls * ly + slc * lz) / flat
                # ---- distances in the folded frame -> pixels
                zf2 = x * x + y * y
                conv = (1.0 - rr2) / (1.0 - zf2) / fp       # folded Euclid -> screen px
                hconv = 2.0 / (1.0 - zf2)                   # folded Euclid -> hyperbolic
                d_crease = (x * sa - y * ca) * conv          # to facet crease (centre->vertex line)
                de = math.sqrt((x - c) * (x - c) + y * y) - r
                d_edge_h = de * hconv                        # hyperbolic distance to tile edge
                d_edge_px = de * conv
                dvx, dvy = x - vx0, y - vy0
                d_vert_h = math.sqrt(dvx * dvx + dvy * dvy) * hconv
                rho_f = 2.0 * math.atanh(math.sqrt(zf2) if zf2 < 0.999999 else 0.9999995)
                # ---- tile colour from world centre (hyperboloid coords hash)
                nn = 1.0 - (cwx * cwx + cwy * cwy)
                hX = 2.0 * cwx / nn
                hY = 2.0 * cwy / nn
                hh = _hash2(int(round(hX * 7.31)) + 100000, int(round(hY * 7.31)) + 100000)
                ci = hh % K
                jit = ((hh >> 8) & 255) / 255.0 - 0.5
                br = pal[ci, 0] * (1.0 + 0.12 * jit)
                bg = pal[ci, 1] * (1.0 + 0.12 * jit)
                bb = pal[ci, 2] * (1.0 + 0.12 * jit)
                # tile centre hyperbolic distance from screen centre
                cs2 = csx * csx + csy * csy
                if cs2 > 0.99999999:
                    cs2 = 0.99999999
                rho_c = 2.0 * math.atanh(math.sqrt(cs2))
                # reveal
                talpha = (rev - rho_c) / rev_s
                if talpha < 0.0:
                    talpha = 0.0
                if talpha > 1.0:
                    talpha = 1.0
                talpha = talpha * lod + talpha_pix * (1.0 - lod)
                # ring flashes (whole tiles light up as the hyperbolic ring passes their centre)
                flash = 0.0
                for ri in range(nr):
                    dq = (rho_c - rings[ri, 0]) / fw
                    flash += rings[ri, 1] * math.exp(-dq * dq)
                flash = 1.0 - math.exp(-flash)
                # facet shading with anti-aliased crease
                wnb = 0.5 - d_crease
                if wnb < 0.0:
                    wnb = 0.0
                if wnb > 0.5:
                    wnb = 0.5
                shd = sh1 * (1.0 - wnb) + sh2 * wnb
                inner = math.exp(-rho_f * 1.6)
                kk = shd * (0.82 + iglow * inner) * bright
                cr = br * kk + flash * (0.25 * br + 0.42 * prm[12])
                cg = bg * kk + flash * (0.25 * bg + 0.42 * prm[13])
                cb = bb * kk + flash * (0.25 * bb + 0.42 * prm[14])
                # seam glow + crisp seam
                gl = seam_glow * math.exp(-d_edge_h / seam_gw) * (1.0 + 1.2 * flash)
                cr += seam_col[0] * gl
                cg += seam_col[1] * gl
                cb += seam_col[2] * gl
                swp = seam_hw * lam
                scov = swp + 0.5 - d_edge_px
                if scov < 0.0:
                    scov = 0.0
                if scov > 1.0:
                    scov = 1.0
                thin = 2.0 * swp
                if thin > 1.0:
                    thin = 1.0
                scov *= thin
                sb = 1.0 + 0.5 * flash
                cr = cr * (1.0 - scov) + seam_col[0] * sb * scov
                cg = cg * (1.0 - scov) + seam_col[1] * sb * scov
                cb = cb * (1.0 - scov) + seam_col[2] * sb * scov
                # vertex sockets
                vg = vglow * math.exp(-d_vert_h * d_vert_h / 0.012) * (1.0 + flash)
                cr += vg * 1.0
                cg += vg * 0.95
                cb += vg * 0.9
                # LOD blend
                cr = cr * lod + avg_col[0] * (1.0 - lod)
                cg = cg * lod + avg_col[1] * (1.0 - lod)
                cb = cb * lod + avg_col[2] * (1.0 - lod)
            # rim glow
            cr += seam_col[0] * rim_k
            cg += seam_col[1] * rim_k
            cb += seam_col[2] * rim_k
            a = cov * talpha
            out[py, px, 0] = cr * a
            out[py, px, 1] = cg * a
            out[py, px, 2] = cb * a
            out[py, px, 3] = a

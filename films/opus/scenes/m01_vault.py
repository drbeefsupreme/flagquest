"""m01 - the narthex vault: a real 3D starry dome (Galla Placidia) set in tesserae, lit by one candle.

The camera stands at the centre of the dome's sphere (radius 1).  Every screen pixel casts a ray that hits the
dome (z >= 0) or the drum wall below the springing (a cylinder of radius 1).  The dome is laid in concentric ring
courses of tesserae around the zenith (each ring may turn on its own: the celestial spheres); bands (the INCIPIT
band at the dome's base, the great gold frame round the zenith) are flat tile layouts mapped along small circles,
great circles (which the central camera always sees as straight lines) or the wall.  Shading: a flickering candle
point light (+ optional second light), gold-leaf glints from each tile's own tilt, bevelled rims, mortar joints
with analytic anti-aliasing, sRGB <-> linear.  Loose / moving tesserae (stars, the Tesseract, flying tiles) are
splatted afterwards with the same look (splat()).

Scene-specific (m01); vx.mosaic is the film's flat-wall engine, this is its dome cousin.
"""
import math

import numpy as np
from numba import njit

from vx.mosaic import SC

TAU = 2.0 * math.pi


def lin(c):
    """sRGB (0..1) -> linear"""
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def L(name):
    """palette key (vx.mosaic.SMALTI) -> linear rgb"""
    return lin(SC[name])


# ============================================================ camera
class Cam:
    """camera at the sphere centre. yaw 0 = looking north (+y), pitch 0 = horizon, 90 = zenith. f in design px."""

    def __init__(self, yaw=0.0, pitch=0.0, roll=0.0, f=935.0, cx=960.0, cy=540.0):
        self.yaw, self.pitch, self.roll, self.f, self.cx, self.cy = yaw, pitch, roll, f, cx, cy
        cp, sp = math.cos(pitch), math.sin(pitch)
        cyw, syw = math.cos(yaw), math.sin(yaw)
        F = np.array([cp * syw, cp * cyw, sp])
        R = np.array([cyw, -syw, 0.0])
        U = np.cross(R, F)
        if roll:
            cr, sr = math.cos(roll), math.sin(roll)
            R, U = R * cr + U * sr, -R * sr + U * cr
        self.F, self.R, self.U = F, R, U

    def pack(self, s):
        return np.array([*self.F, *self.R, *self.U, self.f * s, self.cx * s, self.cy * s], np.float64)

    def project(self, P):
        """world directions/points (..., 3) -> screen design (x, y), depth (P.F), visible mask"""
        P = np.asarray(P, np.float64)
        z = P @ self.F
        zz = np.where(z > 1e-6, z, 1e-6)
        x = self.cx + self.f * (P @ self.R) / zz
        y = self.cy - self.f * (P @ self.U) / zz
        return x, y, z, z > 1e-3

    def dir(self, x, y):
        """screen design (x, y) -> unit world direction"""
        v = self.F + (x - self.cx) / self.f * self.R - (y - self.cy) / self.f * self.U
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def sph(g, th):
    """colatitude g (0 = zenith), azimuth th (0 = +x, pi/2 = +y = north) -> unit vector"""
    sg = np.sin(g)
    return np.stack([sg * np.cos(th), sg * np.sin(th), np.cos(g)], -1)


# ============================================================ ring courses
class Rings:
    """concentric courses around the zenith. zones: list of dict(g0, g1, tile, colour_fn, gold_fn, grp).
    colour_fn(g, th, rng) -> (n,3) linear colours; gold_fn(g, th) -> (n,) 0/1."""

    def __init__(self, zones, seed=0, gmax=math.pi / 2):
        rng = np.random.default_rng(seed)
        rg, rows, starts, ns = [], [], [], []
        tiles = []
        count = 0
        for z in zones:
            g = z["g0"]
            tile = z["tile"]
            while g < z["g1"] - 1e-9:
                dg = min(tile, z["g1"] - g)
                gc = g + dg / 2
                circ = TAU * math.sin(gc)
                n = max(6, int(round(circ / tile)))
                dth = TAU / n
                off = rng.uniform(0, dth)
                th = (np.arange(n) + 0.5) * dth + off
                hs_c = 0.5 * dg                     # half size across the ring
                hs_a = 0.5 * circ / n               # half size along
                base = np.minimum(hs_c, hs_a)
                hs = base * (0.92 + rng.uniform(-0.07, 0.04, n))
                hv = np.minimum(base * (0.92 + rng.uniform(-0.07, 0.04, n)), hs_c * 0.97)
                ja = rng.normal(0, 0.06, n) * hs_a  # along jitter (radians of arc)
                jc = rng.normal(0, 0.05, n) * hs_c
                ang = rng.normal(0, 0.07, n)
                col = z["colour_fn"](np.full(n, gc), th, rng)
                gold = z["gold_fn"](np.full(n, gc), th) if z.get("gold_fn") else np.zeros(n)
                tilt = rng.normal(0, z.get("tilt", 0.13), (n, 2))
                arr = np.zeros((n, 13), np.float32)
                arr[:, 0], arr[:, 1] = ja, jc
                arr[:, 2], arr[:, 3] = np.cos(ang), np.sin(ang)
                arr[:, 4] = hs
                arr[:, 5:8] = col
                arr[:, 8] = gold
                arr[:, 9:11] = tilt
                arr[:, 11] = rng.normal(0, 1, n)     # value jitter seed
                arr[:, 12] = hv
                tiles.append(arr)
                rows.append([gc, 0.5 * dg, off, dth, z.get("grp", 0)])
                starts.append(count)
                ns.append(n)
                count += n
                g += dg
        self.rows = np.array(rows, np.float64)            # (K, 5): g, hw, off, dth, grp
        self.ri = np.stack([np.array(ns), np.array(starts)], 1).astype(np.int64)   # (K, 2): n, start
        self.tiles = np.vstack(tiles).astype(np.float32)  # (N, 12)
        M = 16384
        self.gmax = gmax
        gs = (np.arange(M) + 0.5) / M * gmax
        edges = self.rows[:, 0] + self.rows[:, 1]
        self.lut = np.clip(np.searchsorted(edges, gs), 0, len(self.rows) - 1).astype(np.int32)
        self.ngrp = int(self.rows[:, 4].max()) + 1

    def rot_per_ring(self, grp_rot):
        return np.asarray(grp_rot, np.float64)[self.rows[:, 4].astype(int)]

    def centres(self, k):
        """world unit vectors of ring k's tile centres (unrotated)"""
        g, hw, off, dth, _ = self.rows[k]
        n = self.ri[k, 0]
        return sph(np.full(n, g), (np.arange(n) + 0.5) * dth + off)


# ============================================================ flat bands
BAND_RING, BAND_GREAT, BAND_WALL = 0, 1, 2


class Bands:
    """flat tile layouts mapped onto the dome/wall.  Tiles in texture units (K units per radian of arc).
    tile columns: x, y, cos, sin, hu, r, g, b, gold, tiltx, tilty, t_set, bed_r, bed_g, bed_b, jit, hv"""

    def __init__(self):
        self.desc, self.desci, self.tiles, self.heads, self.nexts = [], [], [], [], []
        self.ntile = 0
        self.nhead = 0

    def add(self, kind, geom, xy, ang, hs, col, gold, tilt, t_set=None, bed=None, K=935.0, extent=None, wrap=0.0,
            seed=0, hv=None):
        """geom: RING (g0, g1, th0, sg_mid, dir) | GREAT (n(3), a(3), b(3), s0, s1, hw) | WALL (z0, z1, th0, dir).
        extent = (width, height) texture units; wrap = width for azimuthal wrap (0 = none).
        hs = half size along the tile's own u axis, hv (optional) across."""
        n = len(xy)
        rng = np.random.default_rng(seed)
        arr = np.zeros((n, 17), np.float32)
        arr[:, 0:2] = xy
        arr[:, 2], arr[:, 3] = np.cos(ang), np.sin(ang)
        arr[:, 4] = hs
        arr[:, 5:8] = col
        arr[:, 8] = gold
        arr[:, 9:11] = tilt
        arr[:, 11] = -1e9 if t_set is None else t_set
        arr[:, 12:15] = 0.0 if bed is None else bed
        arr[:, 15] = rng.normal(0, 1, n)
        arr[:, 16] = hs if hv is None else hv
        cs = float(max(np.max(arr[:, 4]), np.max(arr[:, 16])) * 2.0 * 1.25 + 1e-3)
        if wrap > 0:
            gw = max(1, int(wrap // cs))
            cs = wrap / gw               # whole cells round the circle so the wrap is exact
        else:
            gw = int(extent[0] / cs) + 2
        gh = int(extent[1] / cs) + 2
        head = -np.ones(gw * gh, np.int32)
        nxt = -np.ones(n, np.int32)
        cx = np.clip((xy[:, 0] / cs).astype(int), 0, gw - 1)
        cy = np.clip((xy[:, 1] / cs).astype(int), 0, gh - 1)
        for i in range(n):
            c = cy[i] * gw + cx[i]
            nxt[i] = head[c]
            head[c] = i
        # offset to global indices
        nxt = np.where(nxt >= 0, nxt + self.ntile, -1).astype(np.int32)
        head = np.where(head >= 0, head + self.ntile, -1).astype(np.int32)
        d = np.zeros(20, np.float64)
        di = np.zeros(8, np.int64)
        di[0] = kind
        di[1] = self.nhead
        di[2] = gw
        di[3] = gh
        d[0] = K
        d[1] = cs
        d[2] = wrap
        d[3], d[4] = extent
        g = np.asarray(geom, np.float64).ravel()
        d[5:5 + len(g)] = g
        self.desc.append(d)
        self.desci.append(di)
        self.tiles.append(arr)
        self.heads.append(head)
        self.nexts.append(nxt)
        self.ntile += n
        self.nhead += gw * gh
        return len(self.desc) - 1

    def pack(self):
        if not self.desc:
            z = np.zeros((1, 17), np.float32)
            return (np.zeros((0, 20)), np.zeros((0, 8), np.int64), z, -np.ones(1, np.int32), -np.ones(1, np.int32))
        return (np.array(self.desc), np.array(self.desci), np.vstack(self.tiles).astype(np.float32),
                np.concatenate(self.heads).astype(np.int32), np.concatenate(self.nexts).astype(np.int32))


# ============================================================ the per-pixel kernel
@njit(cache=True, fastmath=True, inline="always")
def _cheb(du, dv, c, s, hu, hv):
    u = du * c + dv * s
    v = -du * s + dv * c
    return max(abs(u) / hu, abs(v) / hv), u / hu, v / hv


@njit(cache=True, fastmath=True)
def _hash2(ix, iy, seed):
    n = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


@njit(cache=True, fastmath=True)
def _vnoise(x, y, seed):
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    a = _hash2(int(ix), int(iy), seed)
    b = _hash2(int(ix) + 1, int(iy), seed)
    c = _hash2(int(ix), int(iy) + 1, seed)
    d = _hash2(int(ix) + 1, int(iy) + 1, seed)
    return (a + (b - a) * ux) * (1 - uy) + (c + (d - c) * ux) * uy


@njit(cache=True, fastmath=True)
def _render(out, cam, rows, ri, lut, gmax, rtiles, rrot, bdesc, bdesci, btiles, bhead, bnext,
            lights, nl, prm, t):
    h, w, _ = out.shape
    Fx, Fy, Fz = cam[0], cam[1], cam[2]
    Rx, Ry, Rz = cam[3], cam[4], cam[5]
    Ux, Uy, Uz = cam[6], cam[7], cam[8]
    fpx, cxp, cyp = cam[9], cam[10], cam[11]
    K = rows.shape[0]
    M = lut.shape[0]
    NB = bdesc.shape[0]
    amb_r, amb_g, amb_b = prm[0], prm[1], prm[2]
    expo = prm[3]
    gfrac = prm[4]
    shin_g = prm[5]
    spec_g = prm[6]
    mort_r, mort_g, mort_b = prm[7], prm[8], prm[9]
    gd_r, gd_g, gd_b = prm[10], prm[11], prm[12]      # gold dark (linear)
    gm_r, gm_g, gm_b = prm[13], prm[14], prm[15]      # gold mid
    gl_r, gl_g, gl_b = prm[16], prm[17], prm[18]      # gold light
    rv_x, rv_y, rv_z, rv_ang, rv_soft = prm[19], prm[20], prm[21], prm[22], prm[23]
    white = prm[24]
    bevel_k = prm[25]
    wall_z1 = prm[26]
    sat = prm[27]
    for py in range(h):
        for px in range(w):
            xi = (px + 0.5 - cxp) / fpx
            yi = -(py + 0.5 - cyp) / fpx
            dx = Fx + xi * Rx + yi * Ux
            dy = Fy + xi * Ry + yi * Uy
            dz = Fz + xi * Rz + yi * Uz
            n2 = dx * dx + dy * dy + dz * dz
            inv = 1.0 / math.sqrt(n2)
            dx *= inv
            dy *= inv
            dz *= inv
            pa = 1.0 / (fpx * n2)              # pixel footprint (radians) on the unit sphere
            # ---------------------------------------------------------- surface hit
            on_wall = dz < 0.0
            if on_wall:
                rho = math.sqrt(dx * dx + dy * dy) + 1e-9
                Px, Py, Pz = dx / rho, dy / rho, dz / rho
                Nx, Ny, Nz = -Px, -Py, 0.0
                pa = pa / (rho * rho)
            else:
                Px, Py, Pz = dx, dy, dz
                Nx, Ny, Nz = -dx, -dy, -dz
            th = math.atan2(Py, Px)
            g = math.acos(min(1.0, max(-1.0, Pz))) if not on_wall else 0.5 * math.pi
            # tangent frame: T1 azimuthal (east-ish), T2 = N x T1
            if on_wall:
                T1x, T1y, T1z = -Py, Px, 0.0
                T2x, T2y, T2z = 0.0, 0.0, 1.0
            else:
                sx = math.sqrt(Px * Px + Py * Py)
                if sx > 1e-6:
                    T1x, T1y, T1z = -Py / sx, Px / sx, 0.0
                else:
                    T1x, T1y, T1z = 1.0, 0.0, 0.0
                T2x = Ny * T1z - Nz * T1y
                T2y = Nz * T1x - Nx * T1z
                T2z = Nx * T1y - Ny * T1x
            # ---------------------------------------------------------- find the tessera
            found = 0          # 0 none, 1 ring tile, 2 band tile, 3 marble
            bi = -1
            best = 1e9
            second = 1e9
            bu = 0.0
            bv = 0.0
            bhs = 1.0
            bca = 1.0
            bsa = 0.0
            ktex = 1.0
            mort_edge = 1e9    # distance (rad) to a hard zone boundary
            # bands first (they lie over the rings)
            for b in range(NB):
                kind = bdesci[b, 0]
                Kb = bdesc[b, 0]
                xb = 0.0
                yb = 0.0
                inside = False
                if kind == 0 and not on_wall:
                    g0 = bdesc[b, 5]
                    g1 = bdesc[b, 6]
                    if g >= g0 and g < g1:
                        rel = (th - bdesc[b, 7]) * bdesc[b, 9]
                        rel = rel - TAU * math.floor(rel / TAU)
                        xb = rel * bdesc[b, 8] * Kb
                        yb = (g - g0) * Kb
                        inside = True
                        mort_edge = min(g - g0, g1 - g)
                elif kind == 1 and not on_wall:
                    c = Px * bdesc[b, 5] + Py * bdesc[b, 6] + Pz * bdesc[b, 7]
                    hw = bdesc[b, 16]
                    if abs(c) < math.sin(hw):
                        sa_ = Px * bdesc[b, 11] + Py * bdesc[b, 12] + Pz * bdesc[b, 13]
                        ca_ = Px * bdesc[b, 8] + Py * bdesc[b, 9] + Pz * bdesc[b, 10]
                        sl = math.atan2(sa_, ca_)
                        if sl >= bdesc[b, 14] and sl < bdesc[b, 15]:
                            cc = math.asin(c)
                            xb = (sl - bdesc[b, 14]) * Kb
                            yb = (cc + hw) * Kb
                            inside = True
                            mort_edge = min(hw - abs(cc), min(sl - bdesc[b, 14], bdesc[b, 15] - sl))
                elif kind == 2 and on_wall:
                    z0 = bdesc[b, 5]
                    z1 = bdesc[b, 6]
                    if Pz <= z0 and Pz > z1:
                        rel = (th - bdesc[b, 7]) * bdesc[b, 8]
                        rel = rel - TAU * math.floor(rel / TAU)
                        xb = rel * Kb
                        yb = (z0 - Pz) * Kb
                        inside = True
                        mort_edge = min(z0 - Pz, Pz - z1)
                if inside:
                    cs = bdesc[b, 1]
                    wrapw = bdesc[b, 2]
                    goff = bdesci[b, 1]
                    gw = bdesci[b, 2]
                    gh = bdesci[b, 3]
                    gxc = int(xb / cs)
                    gyc = int(yb / cs)
                    for oy in range(-1, 2):
                        gy = gyc + oy
                        if gy < 0 or gy >= gh:
                            continue
                        for ox in range(-1, 2):
                            gx = gxc + ox
                            if wrapw > 0:
                                if gx < 0:
                                    gx += gw
                                elif gx >= gw:
                                    gx -= gw
                            if gx < 0 or gx >= gw:
                                continue
                            j = bhead[goff + gy * gw + gx]
                            while j >= 0:
                                ddx = xb - btiles[j, 0]
                                if wrapw > 0:
                                    if ddx > 0.5 * wrapw:
                                        ddx -= wrapw
                                    elif ddx < -0.5 * wrapw:
                                        ddx += wrapw
                                ddy = yb - btiles[j, 1]
                                hs = btiles[j, 4]
                                hv_ = btiles[j, 16]
                                hm = max(hs, hv_)
                                if abs(ddx) < 2.2 * hm and abs(ddy) < 2.2 * hm:
                                    d, uu, vv = _cheb(ddx, ddy, btiles[j, 2], btiles[j, 3], hs, hv_)
                                    if d < best:
                                        second = best
                                        best = d
                                        bi = j
                                        bu = uu
                                        bv = vv
                                        bhs = min(hs, hv_)
                                        bca = btiles[j, 2]
                                        bsa = btiles[j, 3]
                                    elif d < second:
                                        second = d
                                j = bnext[j]
                    found = 2
                    ktex = Kb
                    break
            if found == 0 and not on_wall:
                sg = math.sin(g)
                li = int(g / gmax * M)
                if li >= M:
                    li = M - 1
                k0 = lut[li]
                for k in range(k0 - 1, k0 + 2):
                    if k < 0 or k >= K:
                        continue
                    gk = rows[k, 0]
                    dg = g - gk
                    hwk = rows[k, 1]
                    if abs(dg) > hwk * 2.5:
                        continue
                    n = ri[k, 0]
                    st = ri[k, 1]
                    dth = rows[k, 3]
                    rel = th - rows[k, 2] - rrot[k]
                    rel = rel - TAU * math.floor(rel / TAU)
                    j0 = int(rel / dth)
                    for dj in range(-1, 2):
                        j = j0 + dj
                        if j < 0:
                            j += n
                        elif j >= n:
                            j -= n
                        idx = st + j
                        thc = (j + 0.5) * dth
                        dd = rel - thc
                        if dd > math.pi:
                            dd -= TAU
                        elif dd < -math.pi:
                            dd += TAU
                        du = dd * sg - rtiles[idx, 0]
                        dv = dg - rtiles[idx, 1]
                        hs = rtiles[idx, 4]
                        d, uu, vv = _cheb(du, dv, rtiles[idx, 2], rtiles[idx, 3], hs, rtiles[idx, 12])
                        if d < best:
                            second = best
                            best = d
                            bi = idx
                            bu = uu
                            bv = vv
                            bhs = min(hs, rtiles[idx, 12])
                            bca = rtiles[idx, 2]
                            bsa = rtiles[idx, 3]
                        elif d < second:
                            second = d
                if bi >= 0:
                    found = 1
            if found == 0 and on_wall:
                found = 3
            # ---------------------------------------------------------- material
            cr = 0.0
            cg = 0.0
            cb = 0.0
            gold = 0.0
            tx = 0.0
            ty = 0.0
            flat = False          # unset bed: matte plaster, no joints
            vj = 0.0
            cov = 0.0
            if found == 1:
                cr = rtiles[bi, 5]
                cg = rtiles[bi, 6]
                cb = rtiles[bi, 7]
                gold = rtiles[bi, 8]
                tx = rtiles[bi, 9]
                ty = rtiles[bi, 10]
                vj = rtiles[bi, 11]
            elif found == 2 and bi >= 0:
                if t < btiles[bi, 11]:
                    cr = btiles[bi, 12]
                    cg = btiles[bi, 13]
                    cb = btiles[bi, 14]
                    flat = True
                else:
                    cr = btiles[bi, 5]
                    cg = btiles[bi, 6]
                    cb = btiles[bi, 7]
                    gold = btiles[bi, 8]
                    tx = btiles[bi, 9]
                    ty = btiles[bi, 10]
                    vj = btiles[bi, 15]
            elif found == 3:
                # opus sectile: marble revetment slabs (book-matched veining)
                zt = -Pz
                sw = 0.42
                shh = wall_z1
                sxi = math.floor(th / sw)
                fxs = th / sw - sxi
                if int(sxi) % 2 == 1:
                    fxs = 1.0 - fxs
                syi = math.floor(zt / shh)
                fys = zt / shh - syi
                vn = _vnoise(fxs * 3.0 + syi * 7.1, fys * 2.2, 11) * 0.65 + _vnoise(fxs * 9.0, fys * 7.0, 12) * 0.35
                vein = abs(math.sin((fxs * 5.0 + fys * 2.0 + vn * 3.2) * 2.2))
                vein = 1.0 - min(1.0, vein * 5.0)
                vein2 = abs(math.sin((fxs * 11.0 - fys * 4.0 + vn * 6.0) * 1.7))
                vein2 = 1.0 - min(1.0, vein2 * 9.0)
                # dark verde antico: green-black with pale veins
                cr = 0.008 + 0.006 * vn + 0.030 * vein + 0.014 * vein2
                cg = 0.012 + 0.008 * vn + 0.034 * vein + 0.014 * vein2
                cb = 0.010 + 0.006 * vn + 0.028 * vein + 0.013 * vein2
                jd = min(min(fxs, 1.0 - fxs) * sw, min(fys, 1.0 - fys) * shh)
                cov = min(1.0, max(0.0, (jd - 0.0015) / pa + 0.5))
                tx = 0.0
                ty = 0.0
            if found == 1 or (found == 2 and bi >= 0):
                # distance to the mortar: own outline + bisector with the nearest neighbour
                m = min((1.16 - best) * bhs, (second - best) * bhs * 0.5)
                if found == 2:
                    m = m / ktex
                    bhs_r = bhs / ktex
                else:
                    bhs_r = bhs
                m = min(m, mort_edge)
                gw_ = gfrac * bhs_r
                cov = min(1.0, max(0.0, (m - gw_) / pa + 0.5))
                if flat:
                    cov = 1.0
                    m = 1.0
            # ---------------------------------------------------------- tile normal (tilt + bevel)
            nx = Nx + tx * T1x + ty * T2x
            ny = Ny + tx * T1y + ty * T2y
            nz = Nz + tx * T1z + ty * T2z
            rimf = 1.0
            if (found == 1 or found == 2) and not flat:
                # the tile's own axes in the tangent plane
                E1x = T1x * bca + T2x * bsa
                E1y = T1y * bca + T2y * bsa
                E1z = T1z * bca + T2z * bsa
                E2x = -T1x * bsa + T2x * bca
                E2y = -T1y * bsa + T2y * bca
                E2z = -T1z * bsa + T2z * bca
                au = abs(bu)
                av = abs(bv)
                edge = 1.0 - max(au, av)                 # 0 at the rim, 1 at the centre
                rimf = min(1.0, edge / 0.28)
                bev = (1.0 - rimf) * bevel_k
                if au > av:
                    sgn = 1.0 if bu > 0 else -1.0
                    nx += bev * sgn * E1x
                    ny += bev * sgn * E1y
                    nz += bev * sgn * E1z
                else:
                    sgn = 1.0 if bv > 0 else -1.0
                    nx += bev * sgn * E2x
                    ny += bev * sgn * E2y
                    nz += bev * sgn * E2z
            nn = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            nx *= nn
            ny *= nn
            nz *= nn
            # view vector (to the camera at the origin)
            vx_ = -Px
            vy_ = -Py
            vz_ = -Pz
            vn_ = 1.0 / math.sqrt(vx_ * vx_ + vy_ * vy_ + vz_ * vz_)
            vx_ *= vn_
            vy_ *= vn_
            vz_ *= vn_
            # reveal ("the light wakes the tiles"): angular front around a direction
            rdot = Px * rv_x + Py * rv_y + Pz * rv_z
            rdot = rdot / math.sqrt(Px * Px + Py * Py + Pz * Pz)
            ra = math.acos(min(1.0, max(-1.0, rdot)))
            rvv = 1.0 - min(1.0, max(0.0, (ra - rv_ang) / rv_soft + 0.5))
            rvv = rvv * rvv * (3 - 2 * rvv)
            # ---------------------------------------------------------- light
            dr = 0.0
            dg_ = 0.0
            db = 0.0
            sr = 0.0
            sgc = 0.0
            sbc = 0.0
            mdr = 0.0
            mdg = 0.0
            mdb = 0.0
            shr = 0.0
            shg = 0.0
            shb = 0.0
            for li_ in range(nl):
                lx = lights[li_, 0] - Px
                ly = lights[li_, 1] - Py
                lz = lights[li_, 2] - Pz
                ld2 = lx * lx + ly * ly + lz * lz
                ld = math.sqrt(ld2)
                lx /= ld
                ly /= ld
                lz /= ld
                att = lights[li_, 6] / (ld2 + lights[li_, 7])
                ndl = nx * lx + ny * ly + nz * lz
                ndl0 = Nx * lx + Ny * ly + Nz * lz
                dif = max(0.0, 0.75 * ndl + 0.25 * ndl0)
                hx = lx + vx_
                hy = ly + vy_
                hz = lz + vz_
                hn = 1.0 / math.sqrt(hx * hx + hy * hy + hz * hz + 1e-12)
                ndh = max(0.0, (nx * hx + ny * hy + nz * hz) * hn)
                lr = lights[li_, 3] * att
                lg = lights[li_, 4] * att
                lb = lights[li_, 5] * att
                dr += dif * lr
                dg_ += dif * lg
                db += dif * lb
                md = max(0.0, ndl0)
                mdr += md * lr
                mdg += md * lg
                mdb += md * lb
                broad = ndh ** 6
                sharp = ndh ** shin_g
                sr += broad * lr
                sgc += broad * lg
                sbc += broad * lb
                shr += sharp * lr
                shg += sharp * lg
                shb += sharp * lb
            dr = dr * rvv + amb_r
            dg_ = dg_ * rvv + amb_g
            db = db * rvv + amb_b
            sr *= rvv
            sgc *= rvv
            sbc *= rvv
            shr *= rvv
            shg *= rvv
            shb *= rvv
            mdr = mdr * rvv + amb_r
            mdg = mdg * rvv + amb_g
            mdb = mdb * rvv + amb_b
            # value jitter
            jk = 1.0 + 0.07 * vj
            # glass smalti: diffuse body + a small glassy glint
            glr = cr * dr + 0.035 * shr
            glg = cg * dg_ + 0.035 * shg
            glb = cb * db + 0.035 * shb
            if gold > 0.0:
                # gold leaf under glass: dark body + broad sheen + sharp glint of the flame
                gor = gd_r * (0.35 * dr + 0.02) + gm_r * sr * 0.9 + gl_r * shr * spec_g
                gog = gd_g * (0.35 * dg_ + 0.02) + gm_g * sgc * 0.9 + gl_g * shg * spec_g
                gob = gd_b * (0.35 * db + 0.02) + gm_b * sbc * 0.9 + gl_b * shb * spec_g
                tr = (glr * (1 - gold) + gor * gold) * jk
                tg = (glg * (1 - gold) + gog * gold) * jk
                tb = (glb * (1 - gold) + gob * gold) * jk
            else:
                tr = glr * jk
                tg = glg * jk
                tb = glb * jk
            if found != 3 and not flat:
                shade_ = 0.78 + 0.22 * rimf
                tr *= shade_
                tg *= shade_
                tb *= shade_
            # mortar
            mr = mort_r * mdr * 0.30
            mg = mort_g * mdg * 0.30
            mb = mort_b * mdb * 0.30
            if found == 0:
                cov = 0.0
            rr = tr * cov + mr * (1.0 - cov)
            gg = tg * cov + mg * (1.0 - cov)
            bb = tb * cov + mb * (1.0 - cov)
            # saturation (lighting-based desaturation of the dark) + exposure + soft shoulder
            lum = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
            rr = lum + (rr - lum) * sat
            gg = lum + (gg - lum) * sat
            bb = lum + (bb - lum) * sat
            rr *= expo
            gg *= expo
            bb *= expo
            rr = rr * (1.0 + rr / (white * white)) / (1.0 + rr)
            gg = gg * (1.0 + gg / (white * white)) / (1.0 + gg)
            bb = bb * (1.0 + bb / (white * white)) / (1.0 + bb)
            out[py, px, 0] = rr
            out[py, px, 1] = gg
            out[py, px, 2] = bb


def to_srgb_inplace(img):
    np.clip(img, 0, 1, out=img)
    m = img <= 0.0031308
    hi = 1.055 * np.power(img, 1 / 2.4, dtype=np.float32) - 0.055
    return np.where(m, 12.92 * img, hi).astype(np.float32)


# ============================================================ loose tiles (splat)
@njit(cache=True, fastmath=True)
def _splat(img, s, xs, ys, cs, sn, hu, hv, cols, alpha, rim, lxs, lys, glint, seat):
    """xs, ys, hu, hv in design px; cols (N,3) final sRGB tile colour; alpha (N,); rim light dir (lxs, lys);
    glint (N,) extra rim sparkle; seat (N,) darkening of the joint ring around the tile (0..1)."""
    h, w, _ = img.shape
    for i in range(xs.shape[0]):
        a = alpha[i]
        if a <= 0.003:
            continue
        x = xs[i] * s
        y = ys[i] * s
        u0 = hu[i] * s
        v0 = hv[i] * s
        if u0 < 0.15 or v0 < 0.15:
            continue
        c = cs[i]
        sn_ = sn[i]
        ext = (abs(c) * u0 + abs(sn_) * v0, abs(sn_) * u0 + abs(c) * v0)
        rr = rim * s + 1.0
        x0 = int(math.floor(x - ext[0] - rr - 1))
        x1 = int(math.ceil(x + ext[0] + rr + 1))
        y0 = int(math.floor(y - ext[1] - rr - 1))
        y1 = int(math.ceil(y + ext[1] + rr + 1))
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w - 1, x1)
        y1 = min(h - 1, y1)
        cr, cg, cb = cols[i, 0], cols[i, 1], cols[i, 2]
        gl = glint[i]
        st = seat[i] * a
        for py in range(y0, y1 + 1):
            fy = py + 0.5 - y
            for px in range(x0, x1 + 1):
                fx = px + 0.5 - x
                u = fx * c + fy * sn_
                v = -fx * sn_ + fy * c
                du = abs(u) - u0
                dv = abs(v) - v0
                dout = max(du, dv)          # >0 outside (px)
                if dout > rr:
                    continue
                cov = min(1.0, max(0.0, 0.5 - dout))
                if st > 0.0:
                    sc = min(1.0, max(0.0, (rr - dout) / rr)) * st * (1.0 - cov)
                    k = 1.0 - 0.55 * sc
                    img[py, px, 0] *= k
                    img[py, px, 1] *= k
                    img[py, px, 2] *= k
                if cov <= 0.0:
                    continue
                # bevel: darken towards the rim, light the rim facing the light
                un = u / u0
                vn = v / v0
                e = 1.0 - max(abs(un), abs(vn))
                rimf = min(1.0, e / 0.3)
                if abs(un) > abs(vn):
                    ox = 1.0 if un > 0 else -1.0
                    oy = 0.0
                else:
                    ox = 0.0
                    oy = 1.0 if vn > 0 else -1.0
                # outward direction in screen space
                wx = ox * c - oy * sn_
                wy = ox * sn_ + oy * c
                lit = max(0.0, wx * lxs + wy * lys) * (1.0 - rimf)
                k = (0.8 + 0.2 * rimf)
                r_ = cr * k + gl * lit
                g_ = cg * k + gl * lit
                b_ = cb * k + gl * lit * 0.85
                aa = cov * a
                img[py, px, 0] = img[py, px, 0] * (1 - aa) + r_ * aa
                img[py, px, 1] = img[py, px, 1] * (1 - aa) + g_ * aa
                img[py, px, 2] = img[py, px, 2] * (1 - aa) + b_ * aa


def splat(img, s, xs, ys, ang, hu, hv, cols, alpha=None, rim=1.2, light=(0.0, 1.0), glint=None, seat=None):
    n = len(xs)
    if n == 0:
        return img
    f32 = lambda a: np.ascontiguousarray(np.broadcast_to(np.asarray(a, np.float32), (n,)))
    ln = math.hypot(*light) + 1e-9
    _splat(img, float(s), f32(xs), f32(ys), f32(np.cos(ang)), f32(np.sin(ang)), f32(hu), f32(hv),
           np.ascontiguousarray(np.asarray(cols, np.float32).reshape(n, 3)),
           f32(1.0 if alpha is None else alpha), float(rim), light[0] / ln, light[1] / ln,
           f32(0.0 if glint is None else glint), f32(0.6 if seat is None else seat))
    return img


# ============================================================ the vault object
class Vault:
    def __init__(self, rings, bands):
        self.rings = rings
        self.bands = bands
        self.bp = bands.pack()

    def render(self, fc, cam, t, lights, ring_rot, prm):
        """-> linear-light, tone-shouldered float32 (h, w, 3) (call to_srgb_inplace for display)"""
        out = np.empty((fc.h, fc.w, 3), np.float32)
        L_ = np.zeros((4, 8), np.float64)
        for i, l in enumerate(lights[:4]):
            L_[i, :len(l)] = l
        rr = self.rings.rot_per_ring(ring_rot)
        bd, bdi, bt, bh, bn = self.bp
        _render(out, cam.pack(fc.s), self.rings.rows, self.rings.ri, self.rings.lut, float(self.rings.gmax),
                self.rings.tiles, rr, bd, bdi, bt, bh, bn, L_, len(lights), np.asarray(prm, np.float64), float(t))
        return out


def tangent_frame(P):
    """(N,3) unit points -> inward normal N, T1 (azimuthal), T2 = N x T1 (as the kernel)"""
    P = np.asarray(P, np.float64)
    N = -P
    sx = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
    T1 = np.where(sx[:, None] > 1e-6, np.stack([-P[:, 1], P[:, 0], np.zeros(len(P))], 1) / np.maximum(sx, 1e-9)[:, None],
                  np.array([1.0, 0.0, 0.0]))
    T2 = np.cross(N, T1)
    return N, T1, T2


def shade_tiles(P, col, gold, tilt, lights, prm, emit=None, gain=None, view=None):
    """per-tile lighting with the kernel's model (no bevel: the splat adds it). P (N,3) world points of the
    tiles (on/near the unit dome), col (N,3) linear albedo, gold (N,), tilt (N,2). -> sRGB (N,3)"""
    P = np.asarray(P, np.float64)
    n = len(P)
    if n == 0:
        return np.zeros((0, 3), np.float32)
    p = np.asarray(prm, np.float64)
    amb = p[0:3]
    expo, shin, spec_g = p[3], p[5], p[6]
    gd, gm, gl = p[10:13], p[13:16], p[16:19]
    rv_dir, rv_ang, rv_soft = p[19:22], p[22], p[23]
    white, sat = p[24], p[27]
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    N, T1, T2 = tangent_frame(Pn)
    tl = np.asarray(tilt, np.float64).reshape(n, 2)
    nrm = N + tl[:, :1] * T1 + tl[:, 1:2] * T2
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    V = -P if view is None else np.asarray(view, np.float64) - P
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    ra = np.arccos(np.clip(Pn @ rv_dir, -1, 1))
    rvv = 1.0 - np.clip((ra - rv_ang) / rv_soft + 0.5, 0, 1)
    rvv = rvv * rvv * (3 - 2 * rvv)
    dif = np.zeros((n, 3))
    br = np.zeros((n, 3))
    sh = np.zeros((n, 3))
    for l in lights:
        l = np.asarray(l, np.float64)
        Lv = l[:3] - P
        d2 = np.sum(Lv * Lv, 1)
        Ld = Lv / np.sqrt(d2)[:, None]
        att = l[6] / (d2 + l[7])
        ndl = np.sum(nrm * Ld, 1)
        ndl0 = np.sum(N * Ld, 1)
        dd = np.maximum(0, 0.75 * ndl + 0.25 * ndl0)
        Hh = Ld + V
        Hh /= np.linalg.norm(Hh, axis=1, keepdims=True) + 1e-12
        ndh = np.maximum(0, np.sum(nrm * Hh, 1))
        lc = l[3:6][None, :] * att[:, None]
        dif += dd[:, None] * lc
        br += (ndh ** 6)[:, None] * lc
        sh += (ndh ** shin)[:, None] * lc
    dif = dif * rvv[:, None] + amb
    br *= rvv[:, None]
    sh *= rvv[:, None]
    col = np.asarray(col, np.float64).reshape(n, 3)
    g = np.asarray(gold, np.float64).reshape(n)[:, None]
    glass = col * dif + 0.03 * sh
    metal = gd * (0.35 * dif + 0.02) + gm * br * 0.9 + gl * sh * spec_g
    out = glass * (1 - g) + metal * g
    if gain is not None:
        out = out * np.asarray(gain, np.float64).reshape(n, 1)
    if emit is not None:
        out = out + np.asarray(emit, np.float64).reshape(n, 3)
    lum = out @ np.array([0.2126, 0.7152, 0.0722])
    out = lum[:, None] + (out - lum[:, None]) * sat
    out = np.maximum(out * expo, 0)
    out = out * (1 + out / (white * white)) / (1 + out)
    return srgb(np.clip(out, 0, 1)).astype(np.float32)


def screen_tiles(cam, P, ex, ey, hu_rad, hv_rad):
    """project tiles centred at world points P (N,3) with local axes ex, ey (N,3 world tangent vectors) and
    half sizes in radians -> screen x, y, angle, hu, hv (design px), visible mask"""
    x, y, z, vis = cam.project(P)
    zz = np.maximum(z, 1e-3)
    # screen direction of the tile's local x axis
    x2, y2, _, _ = cam.project(P + ex * 1e-3)
    ang = np.arctan2(y2 - y, x2 - x)
    # gnomonic magnification ~ 1/cos^1.5 of the off-axis angle (radial 1/cos^2, tangential 1/cos)
    k = cam.f / zz ** 1.5
    return x, y, ang, hu_rad * k, hv_rad * k, vis

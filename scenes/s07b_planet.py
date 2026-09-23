"""The rise from the plain to orbit (s07b helper): a per-pixel ray-traced planet whose surface is the
balkanized plain, with recursive borders at every scale (plates -> nations-of-one -> regions -> continents),
a flag in every country, the dawn terminator, atmosphere limb and the flag-lights of the night side.

World: km. The planet (radius R) touches the origin O = Lutie's hill; centre (0, 0, -R); x = east,
y = north, z = up. Map coordinates are azimuthal-equidistant around O (exactly the flat plain near O).
Borders are straight Voronoi (like s05b's balkanization) on jittered grids, cell size S0 * 3^k.
"""
import math

import numpy as np
from numba import njit, prange

R_PLANET = 6371.0
S0 = 0.0055          # finest plate size (km) = s07a's dried-slop plates
RATIO = 3.0          # recursive balkanization factor between border levels
NLEV = 13


@njit(cache=True, fastmath=True)
def _h(ix, iy, k):
    h = (ix * 374761393 + iy * 668265263 + k * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return h ^ (h >> 16)


def _h_np(ix, iy, k):
    """numpy mirror of _h (int64 arrays)"""
    h = (ix * 374761393 + iy * 668265263 + k * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return h ^ (h >> 16)


@njit(cache=True, fastmath=True)
def seed_of(ix, iy, k):
    """jittered seed of grid cell (ix, iy) at level k, in cell units"""
    h = _h(ix, iy, k + 17)
    jx = ((h & 1023) / 1023.0) * 0.8 + 0.1
    jy = (((h >> 10) & 1023) / 1023.0) * 0.8 + 0.1
    return ix + jx, iy + jy, h


@njit(cache=True, fastmath=True)
def _cell(px, py, k):
    """returns (distance to the nearest border in cell units, F1 distance, hash of nearest cell)"""
    ix = math.floor(px)
    iy = math.floor(py)
    d1 = 1e9
    d2 = 1e9
    ax = 0.0
    ay = 0.0
    bx = 0.0
    by = 0.0
    hh = 0
    for j in range(-1, 2):
        for i in range(-1, 2):
            sx, sy, h = seed_of(int(ix) + i, int(iy) + j, k)
            dx = sx - px
            dy = sy - py
            d = dx * dx + dy * dy
            if d < d1:
                d2 = d1
                bx, by = ax, ay
                d1 = d
                ax, ay = sx, sy
                hh = h
            elif d < d2:
                d2 = d
                bx, by = sx, sy
    mx = (ax + bx) * 0.5
    my = (ay + by) * 0.5
    nx = bx - ax
    ny = by - ay
    nl = math.sqrt(nx * nx + ny * ny) + 1e-12
    bd = ((mx - px) * nx + (my - py) * ny) / nl
    return bd, math.sqrt(d1), hh


@njit(parallel=True, cache=True, fastmath=True)
def render_planet(out, cam, prm, pal, cols):
    """out (h, w, 3) float32.
    cam: [px, py, pz, rx, ry, rz, ux, uy, uz, fx, fy, fz, tan_half_hfov]  (km, orthonormal basis)
    prm: 0-2 sun dir (unit, towards the sun)  3 time  4 front_km (flags exist within)
         5 ring_km 6 ring_strength 7 flag_glint_gain 8 crack_gain 9 night_lights 10 flatten (0 sphere-lit .. 1 flat)
         11 star_gain 12 border_glow
    pal: (8, 3) country tones.
    cols: [crack, rose_lip, haze_day, haze_night, flag_gold, space, atm_day, atm_blue, border_day, border_night]
    """
    h, w = out.shape[0], out.shape[1]
    R = 6371.0
    Px, Py, Pz = cam[0], cam[1], cam[2] + R        # relative to planet centre
    rx, ry, rz = cam[3], cam[4], cam[5]
    ux, uy, uz = cam[6], cam[7], cam[8]
    fx, fy, fz = cam[9], cam[10], cam[11]
    th = cam[12]
    aspect = h / w
    Lx, Ly, Lz = prm[0], prm[1], prm[2]
    front = prm[4]
    ring_km, ring_s = prm[5], prm[6]
    fgain, bgain = prm[7], prm[8]
    nlight = prm[9]
    flat = prm[10]
    starg = prm[11]
    bglow = prm[12]
    pa = 2.0 * th / w                              # radians per pixel
    PP = Px * Px + Py * Py + Pz * Pz
    for yy in prange(h):
        for xx in range(w):
            sx = (2.0 * (xx + 0.5) / w - 1.0) * th
            sy = (1.0 - 2.0 * (yy + 0.5) / h) * th * aspect
            dx = fx + sx * rx + sy * ux
            dy = fy + sx * ry + sy * uy
            dz = fz + sx * rz + sy * uz
            dl = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= dl
            dy /= dl
            dz /= dl
            b = Px * dx + Py * dy + Pz * dz
            c = PP - R * R
            disc = b * b - c
            tca = -b
            if tca < 0:
                tca = 0.0
            cx_ = Px + dx * tca
            cy_ = Py + dy * tca
            cz_ = Pz + dz * tca
            dmin = math.sqrt(cx_ * cx_ + cy_ * cy_ + cz_ * cz_)
            if disc > 0.0 and b < 0.0:
                t = c / (-b + math.sqrt(disc))
                hx = Px + dx * t
                hy = Py + dy * t
                hz = Pz + dz * t
                nxx = hx / R
                nyy = hy / R
                nzz = hz / R
                cosv = -(dx * nxx + dy * nyy + dz * nzz)
                if cosv < 0.02:
                    cosv = 0.02
                fp = t * pa / math.sqrt(cosv)                 # km per pixel
                rho = math.sqrt(hx * hx + hy * hy)
                theta = math.atan2(rho, hz)
                s = R * theta
                if rho > 1e-12:
                    mx = s * hx / rho
                    my = s * hy / rho
                else:
                    mx = 0.0
                    my = 0.0
                # ---- recursive borders: each level is a political map; the level whose countries are ~70 px on
                # screen gives the fill (cool political-map tones), plate cracks at the finest levels (s07a),
                # luminous borders above, a glint at every capital too small for a drawn flag
                fr_ = 0.0
                fg_ = 0.0
                fb_ = 0.0
                fw_ = 0.0
                crack = 0.0
                edge = 0.0
                bgl = 0.0
                dots = 0.0
                wash = 0.03
                for k in range(13):
                    sk = 0.0055 * 3.0 ** k
                    cp = sk / fp                               # cell size in pixels
                    if cp < 1.2:
                        continue
                    if cp > 2500.0:
                        break
                    bd, f1, hh = _cell(mx / sk, my / sk, k)
                    lc = math.log(cp / 70.0)
                    wf = math.exp(-lc * lc)
                    ci = hh % 8
                    jb = 1.0 + 0.18 * (((hh >> 12) & 255) / 255.0 - 0.5)
                    fr_ += pal[ci, 0] * jb * wf
                    fg_ += pal[ci, 1] * jb * wf
                    fb_ += pal[ci, 2] * jb * wf
                    fw_ += wf
                    dpx = bd * cp
                    fade = (cp - 3.0) / 10.0
                    if fade > 0.0:
                        if fade > 1.0:
                            fade = 1.0
                        if k <= 1:
                            # dried-slop plate cracks (s07a): dark, with a rose-lit lip
                            wpx = 0.5 + 0.02 * cp
                            if wpx > 2.5:
                                wpx = 2.5
                            a = wpx + 0.5 - dpx
                            if a > 0.0:
                                if a > 1.0:
                                    a = 1.0
                                crack = max(crack, a * fade * bgain)
                            qq = (dpx - wpx - 1.0) / 1.2
                            if qq < 3.0:
                                edge = max(edge, math.exp(-qq * qq) * fade * 0.6)
                        else:
                            rank = math.log(cp / 8.0) / 3.3
                            if rank < 0.0:
                                rank = 0.0
                            if rank > 1.0:
                                rank = 1.0
                            wpx = 0.45 + 0.9 * rank
                            a = wpx + 0.5 - dpx
                            if a < 0.0:
                                a = 0.0
                            if a > 1.0:
                                a = 1.0
                            hw = 1.0 + 3.0 * rank
                            halo = 0.0
                            if dpx < 5.0 * hw:
                                halo = math.exp(-dpx / hw)
                            ints = fade * (0.28 + 0.72 * rank)
                            bgl = max(bgl, ints * (0.85 * a + 0.35 * halo))
                    if 4.0 < cp < 40.0:
                        dd = f1 * cp
                        rad = 0.45 + 0.018 * cp
                        if dd < 3.0 * rad:
                            wlev = (cp - 4.0) / 6.0
                            if wlev > 1.0:
                                wlev = 1.0
                            if cp > 22.0:
                                wlev *= (40.0 - cp) / 18.0
                            dots += wlev * math.exp(-(dd * dd) / (rad * rad)) * (0.7 + 0.6 * ((hh >> 3) & 1))
                    elif cp <= 4.0:
                        wash += 0.02 * (cp - 1.2) / 2.8
                # ---- colour
                if fw_ > 1e-6:
                    br = fr_ / fw_
                    bg = fg_ / fw_
                    bb = fb_ / fw_
                else:
                    br = pal[0, 0]
                    bg = pal[0, 1]
                    bb = pal[0, 2]
                ndl = nxx * Lx + nyy * Ly + nzz * Lz
                day = (ndl + 0.08) / 0.22
                if day < 0.0:
                    day = 0.0
                if day > 1.0:
                    day = 1.0
                lit = 0.16 + 0.84 * day * (0.6 + 0.4 * math.sqrt(max(ndl, 0.0)))
                lit = lit * (1 - flat) + flat
                br *= lit
                bg *= lit
                bb *= lit
                br = br * (1 - crack) + cols[0, 0] * crack
                bg = bg * (1 - crack) + cols[0, 1] * crack
                bb = bb * (1 - crack) + cols[0, 2] * crack
                ek = edge * (0.25 + 0.75 * day)
                br += cols[1, 0] * ek * 0.4
                bg += cols[1, 1] * ek * 0.4
                bb += cols[1, 2] * ek * 0.4
                # ring of light travelling across the map (phrase accent)
                dm = math.sqrt(mx * mx + my * my)
                rg = 0.0
                if ring_s > 0.0:
                    rq = (dm - ring_km) / (0.12 * ring_km + 1e-6)
                    rg = ring_s * math.exp(-rq * rq)
                    if dm < ring_km:
                        rg += ring_s * 0.25 * math.exp(-(ring_km - dm) / (0.4 * ring_km + 1e-6))
                # luminous borders: dawn-rose on the day side, cool blue on the night side
                bk = bgl * (1.0 + 2.5 * rg) * bglow
                br += (cols[8, 0] * day + cols[9, 0] * (1 - day)) * bk
                bg += (cols[8, 1] * day + cols[9, 1] * (1 - day)) * bk
                bb += (cols[8, 2] * day + cols[9, 2] * (1 - day)) * bk
                br += 0.10 * rg
                bg += 0.09 * rg
                bb += 0.12 * rg
                # flags: gold glints within the spreading front (night side glows brighter)
                fk = (front - dm) / (0.08 * front + 0.01)
                if fk < 0.0:
                    fk = 0.0
                if fk > 1.0:
                    fk = 1.0
                gd = (dots * (1.0 + 1.5 * rg) + wash) * fgain * fk * (0.75 + nlight * (1.0 - day))
                br += cols[4, 0] * gd
                bg += cols[4, 1] * gd
                bb += cols[4, 2] * gd
                # atmospheric haze (airmass)
                am = 1.0 - math.exp(-0.05 / cosv * (1.0 - flat))
                hz_r = cols[2, 0] * day + cols[3, 0] * (1 - day)
                hz_g = cols[2, 1] * day + cols[3, 1] * (1 - day)
                hz_b = cols[2, 2] * day + cols[3, 2] * (1 - day)
                br = br * (1 - am) + hz_r * am
                bg = bg * (1 - am) + hz_g * am
                bb = bb * (1 - am) + hz_b * am
                # planet rim anti-alias: blend with space when the ray grazes
                edgek = (R - dmin) / (t * pa) if t * pa > 0 else 10.0
                ak = edgek + 0.5
                if ak > 1.0:
                    ak = 1.0
                out[yy, xx, 0] = br * ak + cols[5, 0] * (1 - ak)
                out[yy, xx, 1] = bg * ak + cols[5, 1] * (1 - ak)
                out[yy, xx, 2] = bb * ak + cols[5, 2] * (1 - ak)
            else:
                br = cols[5, 0]
                bg = cols[5, 1]
                bb = cols[5, 2]
                if starg > 0.0:
                    sxq = math.atan2(dy, dx) * 700.0
                    syq = math.asin(dz) * 700.0
                    ix = math.floor(sxq)
                    iy = math.floor(syq)
                    hs = _h(int(ix), int(iy), 555)
                    if (hs & 255) < 2:
                        jx = ((hs >> 8) & 255) / 255.0
                        jy = ((hs >> 16) & 255) / 255.0
                        ddx = (sxq - ix - jx) * 2.0
                        ddy = (syq - iy - jy) * 2.0
                        sb = math.exp(-(ddx * ddx + ddy * ddy)) * starg * (0.4 + 0.6 * ((hs >> 24) & 255) / 255.0)
                        br += sb * 0.9
                        bg += sb * 0.92
                        bb += sb * 1.0
                out[yy, xx, 0] = br
                out[yy, xx, 1] = bg
                out[yy, xx, 2] = bb
            # atmosphere glow around the limb (both on and off the disc)
            alt = dmin - R
            if b < 0.0:
                ah = 90.0
                ag = math.exp(-abs(alt) / ah) if alt > 0 else math.exp(alt / 25.0)
                if alt > 0 or disc > 0:
                    cxn = cx_ / (dmin + 1e-9)
                    cyn = cy_ / (dmin + 1e-9)
                    czn = cz_ / (dmin + 1e-9)
                    sd = cxn * Lx + cyn * Ly + czn * Lz
                    sdd = (sd + 0.25) / 0.6
                    if sdd < 0.0:
                        sdd = 0.0
                    if sdd > 1.0:
                        sdd = 1.0
                    fs = dx * Lx + dy * Ly + dz * Lz
                    fs = max(fs, 0.0) ** 6
                    k = ag * (0.25 + 0.75 * sdd) * (1.0 - flat)
                    out[yy, xx, 0] += (cols[6, 0] * sdd + cols[7, 0] * (1 - sdd)) * k * (0.8 + 1.5 * fs)
                    out[yy, xx, 1] += (cols[6, 1] * sdd + cols[7, 1] * (1 - sdd)) * k * (0.8 + 1.5 * fs)
                    out[yy, xx, 2] += (cols[6, 2] * sdd + cols[7, 2] * (1 - sdd)) * k * (0.8 + 1.5 * fs)


def map_to_world(mx, my):
    """azimuthal-equidistant map coords (km) -> 3D surface point relative to origin O"""
    R = R_PLANET
    s = np.hypot(mx, my)
    th = s / R
    safe = np.maximum(s, 1e-12)
    ex = np.where(s > 1e-12, mx / safe, 0.0)
    ey = np.where(s > 1e-12, my / safe, 0.0)
    rr = R * np.sin(th)
    return np.stack([rr * ex, rr * ey, R * np.cos(th) - R], -1)


def world_to_map(p):
    """3D point (relative to O) -> map coords (km)"""
    R = R_PLANET
    hx, hy, hz = p[..., 0], p[..., 1], p[..., 2] + R
    rho = np.hypot(hx, hy)
    s = R * np.arctan2(rho, hz)
    safe = np.maximum(rho, 1e-12)
    return np.where(rho > 1e-12, s * hx / safe, 0.0), np.where(rho > 1e-12, s * hy / safe, 0.0)


def seeds_in(level, x0, x1, y0, y1, max_n=120000):
    """map-space seeds (km) + hashes of level `level` cells in the bbox (vectorised mirror of seed_of)"""
    sk = S0 * RATIO ** level
    i0, i1 = int(math.floor(x0 / sk)) - 1, int(math.ceil(x1 / sk)) + 1
    j0, j1 = int(math.floor(y0 / sk)) - 1, int(math.ceil(y1 / sk)) + 1
    if (i1 - i0) * (j1 - j0) > max_n or i1 <= i0 or j1 <= j0:
        return np.zeros((0, 2)), np.zeros(0, np.int64)
    I, J = np.meshgrid(np.arange(i0, i1, dtype=np.int64), np.arange(j0, j1, dtype=np.int64))
    I = I.ravel()
    J = J.ravel()
    h = _h_np(I, J, np.int64(level + 17))
    jx = ((h & 1023) / 1023.0) * 0.8 + 0.1
    jy = (((h >> 10) & 1023) / 1023.0) * 0.8 + 0.1
    return np.stack([(I + jx) * sk, (J + jy) * sk], -1), h

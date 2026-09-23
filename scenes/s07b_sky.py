"""Heaven for s07b: a golden sky over an endless sea of clouds (per-pixel ray-plane, numba), god rays
(screen-space volumetric scattering from the real occluders), and the Gateless Gate (a torii of pure light).

World units: the Flagistan disk has radius 1 and lies at z = 0 in the cloud sea; x east, y north, z up.
"""
import math

import numpy as np
from numba import njit, prange

import cv2


@njit(cache=True, fastmath=True)
def _h(ix, iy, k):
    h = (ix * 374761393 + iy * 668265263 + k * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return h ^ (h >> 16)


@njit(cache=True, fastmath=True)
def _vnd(x, y, k):
    """value noise in [-1,1] with analytic gradient"""
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    dux = 6 * fx * (1 - fx)
    duy = 6 * fy * (1 - fy)
    i0 = int(ix)
    j0 = int(iy)
    a = (_h(i0, j0, k) & 65535) / 32767.5 - 1.0
    b = (_h(i0 + 1, j0, k) & 65535) / 32767.5 - 1.0
    c = (_h(i0, j0 + 1, k) & 65535) / 32767.5 - 1.0
    d = (_h(i0 + 1, j0 + 1, k) & 65535) / 32767.5 - 1.0
    v = a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
    gx = dux * ((b - a) + (a - b - c + d) * uy)
    gy = duy * ((c - a) + (a - b - c + d) * ux)
    return v, gx, gy


@njit(parallel=True, cache=True, fastmath=True)
def render_heaven(out, cam, prm, cols):
    """out (h, w, 3) float32.
    cam: [px py pz rx ry rz ux uy uz fx fy fz tan_half_hfov]
    prm: 0-2 sun dir (unit)  3 time  4 cloud_scale  5 cloud_z  6 cover  7 drift_x 8 drift_y
         9 disk_hole (radius of a clear hole round the Flagistan disk, 0 = none)  10 sun_glow  11 flash
    cols: [zenith, mid, horizon, sun, cloud_lit, cloud_shadow, cloud_rim, haze]
    """
    h, w = out.shape[0], out.shape[1]
    Px, Py, Pz = cam[0], cam[1], cam[2]
    rx, ry, rz = cam[3], cam[4], cam[5]
    ux, uy, uz = cam[6], cam[7], cam[8]
    fx, fy, fz = cam[9], cam[10], cam[11]
    th = cam[12]
    aspect = h / w
    Lx, Ly, Lz = prm[0], prm[1], prm[2]
    tm = prm[3]
    csc = prm[4]
    cz = prm[5]
    cover = prm[6]
    drx, dry = prm[7], prm[8]
    hole = prm[9]
    sglow = prm[10]
    flash = prm[11]
    pa = 2.0 * th / w
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
            # ---- sky
            el = dz
            e = el
            if e < 0.0:
                e = 0.0
            # gradient horizon -> mid -> zenith
            k1 = e / 0.28
            if k1 > 1.0:
                k1 = 1.0
            k2 = (e - 0.28) / 0.72
            if k2 < 0.0:
                k2 = 0.0
            if k2 > 1.0:
                k2 = 1.0
            k1 = k1 ** 0.7
            sr = cols[2, 0] + (cols[1, 0] - cols[2, 0]) * k1
            sg = cols[2, 1] + (cols[1, 1] - cols[2, 1]) * k1
            sb = cols[2, 2] + (cols[1, 2] - cols[2, 2]) * k1
            sr = sr + (cols[0, 0] - sr) * k2
            sg = sg + (cols[0, 1] - sg) * k2
            sb = sb + (cols[0, 2] - sb) * k2
            # sun glow (mie-ish)
            cs = dx * Lx + dy * Ly + dz * Lz
            if cs < -1.0:
                cs = -1.0
            g1 = math.exp((cs - 1.0) * 1400.0)
            g0 = math.exp((cs - 1.0) * 160.0)
            g2 = math.exp((cs - 1.0) * 10.0)
            g3 = math.exp((cs - 1.0) * 1.5)
            sun = sglow * (2.2 * g1 + 0.3 * g0 + 0.16 * g2 + 0.05 * g3) + 0.3 * math.exp((cs - 1.0) * 4.0)
            # horizon brightening band
            hb = math.exp(-abs(el) / 0.035) * 0.35
            sr += cols[3, 0] * (sun + hb * g3)
            sg += cols[3, 1] * (sun + hb * g3)
            sb += cols[3, 2] * (sun + hb * g3)
            # faint high cirrus streaks
            if dz > 0.02:
                t = (6.0 - Pz) / dz
                if t > 0:
                    qx = (Px + dx * t) * 0.03 + tm * 0.004
                    qy = (Py + dy * t) * 0.012
                    v1, _a, _b = _vnd(qx * 3.0, qy * 9.0, 7)
                    v2, _a, _b = _vnd(qx * 7.0, qy * 21.0, 8)
                    ci = (v1 * 0.7 + v2 * 0.3) - 0.15
                    if ci > 0:
                        ci = ci * 0.9 * (1.0 - math.exp(-dz * 12.0)) * math.exp(-t * 0.004)
                        cr_ = cols[4, 0] * (0.6 + 1.2 * g2)
                        cg_ = cols[4, 1] * (0.6 + 1.2 * g2)
                        cb_ = cols[4, 2] * (0.6 + 1.2 * g2)
                        sr = sr * (1 - ci) + cr_ * ci
                        sg = sg * (1 - ci) + cg_ * ci
                        sb = sb * (1 - ci) + cb_ * ci
            r_ = sr
            g_ = sg
            b_ = sb
            if dz < 0.0 and cover < -5.0:
                # luminous mist below the horizon (no cloud sea): horizon glow -> deep rose-violet below
                kd = 0.75 * (-dz / 0.6) ** 0.6
                if kd > 0.75:
                    kd = 0.75
                r_ = r_ * (1 - kd) + (cols[5, 0] * 0.8 + cols[2, 0] * 0.2) * kd
                g_ = g_ * (1 - kd) + (cols[5, 1] * 0.8 + cols[2, 1] * 0.2) * kd
                b_ = b_ * (1 - kd) + (cols[5, 2] * 0.8 + cols[2, 2] * 0.2) * kd
            # ---- cloud sea (below the horizon)
            if dz < -1e-4 and Pz > cz and cover > -5.0:
                t = (cz - Pz) / dz
                X = Px + dx * t
                Y = Py + dy * t
                fp = t * pa / math.sqrt(max(-dz, 0.02))         # world units per pixel
                qx = X * csc + drx
                qy = Y * csc + dry
                # fbm with gradient, octaves faded by footprint
                v = 0.0
                gx = 0.0
                gy = 0.0
                amp = 0.5
                fr = 1.0
                norm = 0.0
                for o in range(6):
                    cell_px = 1.0 / (fr * csc * fp)
                    wo = (cell_px - 1.5) / 3.0
                    if wo <= 0.0:
                        break
                    if wo > 1.0:
                        wo = 1.0
                    n, ngx, ngy = _vnd(qx * fr + o * 17.3, qy * fr - o * 9.1, 30 + o)
                    v += n * amp * wo
                    gx += ngx * amp * fr * wo
                    gy += ngy * amp * fr * wo
                    norm += amp
                    amp *= 0.5
                    fr *= 2.07
                dens = v * 0.9 + cover
                # puffy height field -> normal
                hgt = 0.18
                nx_ = -gx * hgt * csc
                ny_ = -gy * hgt * csc
                nz_ = 1.0
                nl = math.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_)
                nx_ /= nl
                ny_ /= nl
                nz_ /= nl
                ndl = nx_ * Lx + ny_ * Ly + nz_ * Lz
                lit = ndl * 2.2 + 0.25
                if lit < 0.0:
                    lit = 0.0
                if lit > 1.4:
                    lit = 1.4
                # silver lining toward the sun
                back = max(0.0, cs) ** 8 * 0.6
                cr = cols[5, 0] + (cols[4, 0] - cols[5, 0]) * lit + cols[6, 0] * back
                cg = cols[5, 1] + (cols[4, 1] - cols[5, 1]) * lit + cols[6, 1] * back
                cb = cols[5, 2] + (cols[4, 2] - cols[5, 2]) * lit + cols[6, 2] * back
                # gaps between cloud heaps: deeper violet
                gap = 0.5 - dens * 1.6
                if gap < 0.0:
                    gap = 0.0
                if gap > 1.0:
                    gap = 1.0
                cr = cr * (1 - 0.45 * gap) + cols[5, 0] * 0.45 * gap * 0.7
                cg = cg * (1 - 0.45 * gap) + cols[5, 1] * 0.45 * gap * 0.7
                cb = cb * (1 - 0.45 * gap) + cols[5, 2] * 0.45 * gap * 0.7
                # clear hole round the Flagistan disk
                if hole > 0.0:
                    rr = math.sqrt(X * X + Y * Y)
                    hk = (rr - hole) / (0.35 * hole)
                    if hk < 0.0:
                        hk = 0.0
                    if hk > 1.0:
                        hk = 1.0
                    cr = cr * hk + cols[5, 0] * 0.35 * (1 - hk)
                    cg = cg * hk + cols[5, 1] * 0.35 * (1 - hk)
                    cb = cb * hk + cols[5, 2] * 0.35 * (1 - hk)
                # aerial perspective toward the horizon
                hz = 1.0 - math.exp(-t * 0.018)
                hr = cols[7, 0] + cols[3, 0] * 0.4 * g2
                hg = cols[7, 1] + cols[3, 1] * 0.4 * g2
                hbb = cols[7, 2] + cols[3, 2] * 0.4 * g2
                cr = cr * (1 - hz) + hr * hz
                cg = cg * (1 - hz) + hg * hz
                cb = cb * (1 - hz) + hbb * hz
                # soft blend to sky at the very horizon
                hk2 = -dz / 0.004
                if hk2 > 1.0:
                    hk2 = 1.0
                r_ = cr * hk2 + sr * (1 - hk2)
                g_ = cg * hk2 + sg * (1 - hk2)
                b_ = cb * hk2 + sb * (1 - hk2)
            out[yy, xx, 0] = r_ + flash
            out[yy, xx, 1] = g_ + flash
            out[yy, xx, 2] = b_ + flash


@njit(cache=True, fastmath=True)
def splat_puffs(img, xs, ys, rs, vx, vy, cl, cs, cr, alpha, soft):
    """composite soft, sun-lit cumulus puffs (already sorted far -> near) into img (h, w, 3) in place.
    xs, ys, rs in pixels; (vx, vy) unit screen direction toward the sun; cl/cs/cr (N, 3) lit/shadow/rim colours."""
    h, w = img.shape[0], img.shape[1]
    for i in range(xs.shape[0]):
        x = xs[i]
        y = ys[i]
        r = rs[i]
        if r < 0.5:
            continue
        x0 = int(max(0.0, math.floor(x - r)))
        x1 = int(min(w - 1.0, math.ceil(x + r)))
        y0 = int(max(0.0, math.floor(y - r)))
        y1 = int(min(h - 1.0, math.ceil(y + r)))
        if x1 < x0 or y1 < y0:
            continue
        ir = 1.0 / r
        sf = soft + 1.2 * ir
        for py in range(y0, y1 + 1):
            dy = (py + 0.5 - y) * ir
            for px in range(x0, x1 + 1):
                dx = (px + 0.5 - x) * ir
                d2 = dx * dx + dy * dy
                if d2 >= 1.0:
                    continue
                d = math.sqrt(d2)
                a = (1.0 - d) / sf
                if a > 1.0:
                    a = 1.0
                a *= alpha[i]
                # position along the sun direction (+1 = sun side), bulged like a sphere
                s = dx * vx[i] + dy * vy[i]
                nz = math.sqrt(1.0 - d2)
                t = 0.5 - 0.5 * (s * 0.85 + 0.35 * nz - 0.1)
                if t < 0.0:
                    t = 0.0
                if t > 1.0:
                    t = 1.0
                t = t * t * (3 - 2 * t)
                rim = 0.0
                if s > 0.0:
                    rim = s * d2 * d2 * 1.2
                    if rim > 1.0:
                        rim = 1.0
                c0 = cl[i, 0] + (cs[i, 0] - cl[i, 0]) * t
                c1 = cl[i, 1] + (cs[i, 1] - cl[i, 1]) * t
                c2 = cl[i, 2] + (cs[i, 2] - cl[i, 2]) * t
                c0 = c0 + (cr[i, 0] - c0) * rim
                c1 = c1 + (cr[i, 1] - c1) * rim
                c2 = c2 + (cr[i, 2] - c2) * rim
                img[py, px, 0] = img[py, px, 0] * (1.0 - a) + c0 * a
                img[py, px, 1] = img[py, px, 1] * (1.0 - a) + c1 * a
                img[py, px, 2] = img[py, px, 2] * (1.0 - a) + c2 * a


def god_rays(src, sx, sy, passes=6, step=0.035, decay=0.985):
    """screen-space volumetric light scattering: log-step radial blur of `src` (h, w, 3) along rays toward the
    light at pixel (sx, sy). Pass i averages with a copy sampled (1 - step*2^i) of the way toward the light,
    so the smear length doubles per pass; `decay` attenuates the far samples."""
    h, w = src.shape[:2]
    acc = src
    for i in range(passes):
        k = max(0.0, 1.0 - step * (2 ** i))
        M = np.array([[k, 0, sx * (1 - k)], [0, k, sy * (1 - k)]], np.float32)
        warped = cv2.warpAffine(acc, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
        acc = 0.5 * (acc + warped * (decay ** (2 ** i)))
    return acc

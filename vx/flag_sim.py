"""Cloth solver for vx.flag (FlagSmith). Numba XPBD cloth with aerodynamics.

Internal units are SI: the pole is always a real 96" (2.44 m) furring pole, the cloth 36"x30",
gravity 9.81 m/s^2, air 1.2 kg/m^3, survey-flag nylon ~0.11 kg/m^2. Design-pixel inputs are converted
with mpp = 2.44 / pole_px so a close-up hero flag and a distant one behave like the same real flag.

Model per substep (small-step XPBD, Macklin et al. 2019):
  * aerodynamic force per triangle from the relative wind (quasi-steady plate model: normal force
    ~ |v| v_n + |v_n| v_n, skin friction ~ |v_t| v_t), clamped so it can never overshoot the flow
  * convected turbulence (Taylor frozen field, 2 octaves of value noise advected by the mean wind)
  * a travelling-wave "vortex street" pressure (Strouhal-like flap frequency ~ U/L) that drives the
    classic flag flutter; the real cloth dynamics (inertia, inextensibility, gravity) do the rest
  * structural (stiff in tension) + soft shear distance constraints, isometric (second-difference) bending,
    long-range attachments to the hoist (inextensible)
  * hoist column pinned to the pole, following a Catmull-Rom path between frames (adaptive substeps
    for fast poles, rigid carry-over on camera cuts / teleports)
  * optional furl: soft attraction to a spiral roll around the top of the pole
"""
import math

import numpy as np
from numba import njit

RHO = 1.2
SIGMA = 0.11
GRAV = 9.81
POLE_M = 2.4384                 # 96 inches
WIND_MPP = POLE_M / 420.0       # wind px/s -> m/s (calibrated on the canonical 420 px pole)

CL = 1.6        # linear (lift-like) normal-force coefficient
CD = 1.1        # quadratic normal drag
CF = 0.06       # skin friction


@njit(cache=True, inline="always")
def _h(ix, iy, iz, s):
    n = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (s * 2654435761)
    n = n & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    n = n ^ (n >> 16)
    return (n & 0xFFFFFF) / 8388607.5 - 1.0


@njit(cache=True)
def vnoise3(x, y, z, s):
    fx = math.floor(x)
    fy = math.floor(y)
    fz = math.floor(z)
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    tx = x - fx
    ty = y - fy
    tz = z - fz
    tx = tx * tx * (3.0 - 2.0 * tx)
    ty = ty * ty * (3.0 - 2.0 * ty)
    tz = tz * tz * (3.0 - 2.0 * tz)
    a = _h(ix, iy, iz, s) + (_h(ix + 1, iy, iz, s) - _h(ix, iy, iz, s)) * tx
    b = _h(ix, iy + 1, iz, s) + (_h(ix + 1, iy + 1, iz, s) - _h(ix, iy + 1, iz, s)) * tx
    c = _h(ix, iy, iz + 1, s) + (_h(ix + 1, iy, iz + 1, s) - _h(ix, iy, iz + 1, s)) * tx
    d = _h(ix, iy + 1, iz + 1, s) + (_h(ix + 1, iy + 1, iz + 1, s) - _h(ix, iy + 1, iz + 1, s)) * tx
    e = a + (b - a) * ty
    f = c + (d - c) * ty
    return e + (f - e) * tz


@njit(cache=True)
def noise1(t, s):
    return vnoise3(t, 0.37, 0.61, s)


@njit(cache=True)
def _cr(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)


@njit(cache=True)
def _hoist(bx, by, a, Lp, Hc, ny, out):
    sa = math.sin(a)
    ca = math.cos(a)
    tx = bx + sa * Lp
    ty = by - ca * Lp
    top = 0.006 * Lp
    for j in range(ny):
        d = top + Hc * j / (ny - 1)
        out[j, 0] = tx - sa * d
        out[j, 1] = ty + ca * d
        out[j, 2] = 0.0


@njit(cache=True)
def furl_target(j, i, nx, ny, Wc, Hc, hx, hy, hz, a, side, rp):
    """spiral roll of the cloth around the pole axis; (hx,hy,hz) = hoist point of row j"""
    u = i / (nx - 1)
    s = u * Wc
    r0 = rp * 1.15
    b = 0.0135 / (2.0 * math.pi)
    th = (-r0 + math.sqrt(r0 * r0 + 2.0 * b * s)) / b
    r = r0 + b * th
    # rows slide down a little and the roll bulges in the middle (a real furl is lumpy)
    v = j / (ny - 1)
    r *= 1.0 + 0.22 * math.sin(math.pi * v) * (0.4 + 0.6 * u)
    along = u * Hc * 0.06
    sa = math.sin(a)
    ca = math.cos(a)
    e1x = ca * side
    e1y = sa * side
    c = math.cos(th)
    sn = math.sin(th)
    x = hx - sa * along + r * c * e1x
    y = hy + ca * along + r * c * e1y
    z = hz + r * sn
    return x, y, z


@njit(cache=True, fastmath=True)
def simulate(nx, ny, Wc, Hc, Lp, poses, cuts, winds, furl, warm, fps, base_sub, iters, seed,
             prm, s_shear, s_bend, side, rp, ca_, cb_, crest, ctype, ta_, tm_, tb_,
             out_x, out_e, out_t, out_c, out_fly):
    """prm = [turbulence, flutter, CL, CD, CF, flap_coef, wavelength(u), span_phase, compression_softness]"""
    turb = prm[0]
    flutter = prm[1]
    cl = prm[2]
    cd = prm[3]
    cf = prm[4]
    fcoef = prm[5]
    lamu = prm[6]
    spanph = prm[7]
    T = poses.shape[0]
    N = nx * ny
    NC = ca_.shape[0]
    NB = tm_.shape[0]
    lamv = np.zeros((NB, 3))
    dxr = Wc / (nx - 1)
    dyr = Hc / (ny - 1)
    spacing = min(dxr, dyr)
    m = SIGMA * Wc * Hc / N
    winv = 1.0 / m
    dt_ref = 1.0 / 240.0
    comp = np.zeros(4)
    comp[1] = s_shear * winv * dt_ref * dt_ref
    comp[2] = s_bend * winv * dt_ref * dt_ref
    comp[3] = prm[8] * winv * dt_ref * dt_ref      # structural constraints under compression (cloth buckles softly)

    x = np.zeros((N, 3))
    v = np.zeros((N, 3))
    p = np.zeros((N, 3))
    acc = np.zeros((N, 3))
    wv = np.zeros((N, 3))
    w = np.full(N, winv)
    lam = np.zeros(NC)
    uu = np.zeros(N)
    vv = np.zeros(N)
    hs = np.zeros((ny, 3))
    hp = np.zeros((ny, 3))
    for j in range(ny):
        w[j * nx] = 0.0
        for i in range(nx):
            uu[j * nx + i] = i / (nx - 1)
            vv[j * nx + i] = j / (ny - 1)

    # initial: flat rectangle, perpendicular to the pole, on `side`
    bx, by, a = poses[0, 0], poses[0, 1], poses[0, 2]
    _hoist(bx, by, a, Lp, Hc, ny, hs)
    sa = math.sin(a)
    cs = math.cos(a)
    for j in range(ny):
        for i in range(nx):
            q = j * nx + i
            d = uu[q] * Wc * side
            x[q, 0] = hs[j, 0] + cs * d
            x[q, 1] = hs[j, 1] + sa * d
            x[q, 2] = 1e-4 * _h(i, j, 7, seed) * (i > 0)
            if furl[0] > 0.0:
                fx, fy, fz = furl_target(j, i, nx, ny, Wc, Hc, hs[j, 0], hs[j, 1], hs[j, 2], a, side, rp)
                f = furl[0]
                x[q, 0] += (fx - x[q, 0]) * f
                x[q, 1] += (fy - x[q, 1]) * f
                x[q, 2] += (fz - x[q, 2]) * f
    for j in range(ny):
        for k in range(3):
            hp[j, k] = hs[j, k]
    if warm == 0:
        for j in range(ny):
            for i in range(nx):
                for k in range(3):
                    out_x[0, j, i, k] = x[j * nx + i, k]

    phase = 0.5 * (_h(seed, 3, 5, 11) + 1.0)
    fly = (ny // 2) * nx + nx - 1
    last_sign = 0.0

    for kf in range(1, T):
        # ---- rigid carry-over on cuts (teleports): move the cloth with the pole, keep its shape
        if cuts[kf]:
            ox, oy, oa = poses[kf - 1, 0], poses[kf - 1, 1], poses[kf - 1, 2]
            nbx, nby, na = poses[kf, 0], poses[kf, 1], poses[kf, 2]
            otx = ox + math.sin(oa) * Lp
            oty = oy - math.cos(oa) * Lp
            ntx = nbx + math.sin(na) * Lp
            nty = nby - math.cos(na) * Lp
            da = na - oa
            c_ = math.cos(da)
            s_ = math.sin(da)
            for q in range(N):
                rx = x[q, 0] - otx
                ry = x[q, 1] - oty
                x[q, 0] = ntx + c_ * rx - s_ * ry
                x[q, 1] = nty + s_ * rx + c_ * ry
                vx_ = v[q, 0]
                vy_ = v[q, 1]
                v[q, 0] = c_ * vx_ - s_ * vy_
                v[q, 1] = s_ * vx_ + c_ * vy_
            _hoist(nbx, nby, na, Lp, Hc, ny, hp)

        # ---- adaptive substeps
        k0 = max(kf - 2, 0)
        k3 = min(kf + 1, T - 1)
        p0x = poses[kf - 1, 0] + math.sin(poses[kf - 1, 2]) * Lp
        p0y = poses[kf - 1, 1] - math.cos(poses[kf - 1, 2]) * Lp
        p1x = poses[kf, 0] + math.sin(poses[kf, 2]) * Lp
        p1y = poses[kf, 1] - math.cos(poses[kf, 2]) * Lp
        disp = 0.0 if cuts[kf] else math.hypot(p1x - p0x, p1y - p0y)
        disp = max(disp, 0.0 if cuts[kf] else math.hypot(poses[kf, 0] - poses[kf - 1, 0],
                                                          poses[kf, 1] - poses[kf - 1, 1]))
        wm = math.sqrt(winds[kf, 0] ** 2 + winds[kf, 1] ** 2 + winds[kf, 2] ** 2)
        ns = base_sub
        ns = max(ns, int(math.ceil(disp / (0.3 * spacing))))
        ns = max(ns, int(math.ceil((wm + disp * fps) / fps / (0.5 * spacing))))
        ns = min(ns, 200)
        dt = 1.0 / (fps * ns)
        damp = 0.2
        if kf < warm * 0.5:
            damp = 3.0
        dfac = math.exp(-damp * dt)

        emax = 0.0
        tmax = 0.0
        crossings = 0.0
        # frame normal for flap-rate measurement
        mid = (ny // 2) * nx
        tux = x[mid + nx - 1, 0] - x[mid, 0]
        tuy = x[mid + nx - 1, 1] - x[mid, 1]
        tuz = x[mid + nx - 1, 2] - x[mid, 2]
        tvx = x[(ny - 1) * nx, 0] - x[0, 0]
        tvy = x[(ny - 1) * nx, 1] - x[0, 1]
        tvz = x[(ny - 1) * nx, 2] - x[0, 2]
        fnx = tuy * tvz - tuz * tvy
        fny = tuz * tvx - tux * tvz
        fnz = tux * tvy - tuy * tvx
        fl = math.sqrt(fnx * fnx + fny * fny + fnz * fnz) + 1e-12
        fnx /= fl
        fny /= fl
        fnz /= fl

        for s in range(ns):
            al = (s + 1.0) / ns
            if cuts[kf]:
                bx, by, a = poses[kf, 0], poses[kf, 1], poses[kf, 2]
            else:
                bx = _cr(poses[k0, 0], poses[kf - 1, 0], poses[kf, 0], poses[k3, 0], al)
                by = _cr(poses[k0, 1], poses[kf - 1, 1], poses[kf, 1], poses[k3, 1], al)
                a = _cr(poses[k0, 2], poses[kf - 1, 2], poses[kf, 2], poses[k3, 2], al)
            _hoist(bx, by, a, Lp, Hc, ny, hs)
            vhx = 0.0
            vhy = 0.0
            vhz = 0.0
            for j in range(ny):
                vhx += (hs[j, 0] - hp[j, 0]) / dt
                vhy += (hs[j, 1] - hp[j, 1]) / dt
                vhz += (hs[j, 2] - hp[j, 2]) / dt
            vhx /= ny
            vhy /= ny
            vhz /= ny
            t = (kf - 1 + al) / fps
            wmx = winds[kf - 1, 0] + (winds[kf, 0] - winds[kf - 1, 0]) * al
            wmy = winds[kf - 1, 1] + (winds[kf, 1] - winds[kf - 1, 1]) * al
            wmz = winds[kf - 1, 2] + (winds[kf, 2] - winds[kf - 1, 2]) * al
            wmag = math.sqrt(wmx * wmx + wmy * wmy + wmz * wmz)
            fr = furl[kf - 1] + (furl[kf] - furl[kf - 1]) * al

            # ---- wind at particles: mean + convected turbulence
            sig = turb * (0.16 * wmag + 0.18) / 0.42
            for q in range(N):
                px = (x[q, 0] - wmx * t) / 1.3
                py = (x[q, 1] - wmy * t + 0.7 * x[q, 2]) / 1.3
                pt = t * 0.45
                n0 = vnoise3(px, py, pt, seed * 7 + 1) + 0.5 * vnoise3(px * 3.1 + 17.0, py * 3.1, pt * 2.3, seed * 7 + 2)
                n1 = vnoise3(px + 31.0, py, pt, seed * 7 + 3) + 0.5 * vnoise3(px * 3.1 + 5.0, py * 3.1 + 9.0, pt * 2.3, seed * 7 + 4)
                n2 = vnoise3(px, py + 57.0, pt, seed * 7 + 5) + 0.5 * vnoise3(px * 3.1 + 3.0, py * 3.1 + 1.0, pt * 2.3, seed * 7 + 6)
                wv[q, 0] = wmx + sig * n0
                wv[q, 1] = wmy + sig * n1 * 0.7
                wv[q, 2] = wmz + sig * n2 * 0.9
                acc[q, 0] = 0.0
                acc[q, 1] = 0.0
                acc[q, 2] = 0.0

            # ---- flutter phase (Strouhal-like: f ~ 0.85 U / L)
            urx = wmx - vhx
            ury = wmy - vhy
            urz = wmz - vhz
            ubar = math.sqrt(urx * urx + ury * ury + urz * urz)
            ff = fcoef * ubar / Wc * (1.0 + 0.3 * noise1(t * 0.5, seed + 91))
            ff = min(ff, 4.5)
            phase += ff * dt
            fmod = flutter * (0.65 + 0.45 * noise1(t * 0.8 + 3.0, seed + 17))
            boost = 1.5 * min(1.0, ubar / 2.0) ** 2

            # ---- aerodynamics per triangle
            for j in range(ny - 1):
                for i in range(nx - 1):
                    for tri in range(2):
                        ia = j * nx + i
                        if tri == 0:
                            ib = ia + 1
                            ic = ia + nx + 1
                        else:
                            ib = ia + nx + 1
                            ic = ia + nx
                        e1x = x[ib, 0] - x[ia, 0]
                        e1y = x[ib, 1] - x[ia, 1]
                        e1z = x[ib, 2] - x[ia, 2]
                        e2x = x[ic, 0] - x[ia, 0]
                        e2y = x[ic, 1] - x[ia, 1]
                        e2z = x[ic, 2] - x[ia, 2]
                        nxx = e1y * e2z - e1z * e2y
                        nyy = e1z * e2x - e1x * e2z
                        nzz = e1x * e2y - e1y * e2x
                        ln = math.sqrt(nxx * nxx + nyy * nyy + nzz * nzz)
                        if ln < 1e-12:
                            continue
                        area = 0.5 * ln
                        nxx /= ln
                        nyy /= ln
                        nzz /= ln
                        rx = (wv[ia, 0] + wv[ib, 0] + wv[ic, 0] - v[ia, 0] - v[ib, 0] - v[ic, 0]) / 3.0
                        ry = (wv[ia, 1] + wv[ib, 1] + wv[ic, 1] - v[ia, 1] - v[ib, 1] - v[ic, 1]) / 3.0
                        rz = (wv[ia, 2] + wv[ib, 2] + wv[ic, 2] - v[ia, 2] - v[ib, 2] - v[ic, 2]) / 3.0
                        vn = rx * nxx + ry * nyy + rz * nzz
                        tx_ = rx - vn * nxx
                        ty_ = ry - vn * nyy
                        tz_ = rz - vn * nzz
                        vm = math.sqrt(rx * rx + ry * ry + rz * rz)
                        vtm = math.sqrt(tx_ * tx_ + ty_ * ty_ + tz_ * tz_)
                        fn = 0.5 * RHO * area * (cl * vm * vn + cd * abs(vn) * vn)
                        # travelling-wave flutter pressure
                        uc = (uu[ia] + uu[ib] + uu[ic]) / 3.0
                        vc = (vv[ia] + vv[ib] + vv[ic]) / 3.0
                        ph = 2.0 * math.pi * (phase - uc / lamu) + spanph * vc
                        fn += fmod * RHO * vtm * (vtm + boost) * area * math.sqrt(uc) * math.sin(ph)
                        ft = 0.5 * RHO * area * cf * vtm
                        fx = (fn * nxx + ft * tx_) * winv / 3.0
                        fy = (fn * nyy + ft * ty_) * winv / 3.0
                        fz = (fn * nzz + ft * tz_) * winv / 3.0
                        acc[ia, 0] += fx
                        acc[ia, 1] += fy
                        acc[ia, 2] += fz
                        acc[ib, 0] += fx
                        acc[ib, 1] += fy
                        acc[ib, 2] += fz
                        acc[ic, 0] += fx
                        acc[ic, 1] += fy
                        acc[ic, 2] += fz

            # ---- integrate (aero clamped: it can accelerate cloth at most up to the flow)
            for q in range(N):
                if w[q] == 0.0:
                    continue
                dvx = acc[q, 0] * dt
                dvy = acc[q, 1] * dt
                dvz = acc[q, 2] * dt
                dm = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
                rx = wv[q, 0] - v[q, 0]
                ry = wv[q, 1] - v[q, 1]
                rz = wv[q, 2] - v[q, 2]
                lim = math.sqrt(rx * rx + ry * ry + rz * rz) * 0.9 + 0.05
                if dm > lim:
                    sc = lim / dm
                    dvx *= sc
                    dvy *= sc
                    dvz *= sc
                v[q, 0] = (v[q, 0] + dvx) * dfac
                v[q, 1] = (v[q, 1] + dvy + GRAV * dt) * dfac
                v[q, 2] = (v[q, 2] + dvz) * dfac
                p[q, 0] = x[q, 0] + v[q, 0] * dt
                p[q, 1] = x[q, 1] + v[q, 1] * dt
                p[q, 2] = x[q, 2] + v[q, 2] * dt
            for j in range(ny):
                q = j * nx
                p[q, 0] = hs[j, 0]
                p[q, 1] = hs[j, 1]
                p[q, 2] = hs[j, 2]

            # ---- constraints
            for c in range(NC):
                lam[c] = 0.0
            for c in range(NB):
                lamv[c, 0] = 0.0
                lamv[c, 1] = 0.0
                lamv[c, 2] = 0.0
            atb = comp[2] / (dt * dt)
            strain = 0.0
            nstr = 0
            for it in range(iters):
                for c in range(NC):
                    ia = ca_[c]
                    ib = cb_[c]
                    wsum = w[ia] + w[ib]
                    if wsum == 0.0:
                        continue
                    dx = p[ib, 0] - p[ia, 0]
                    dy = p[ib, 1] - p[ia, 1]
                    dz = p[ib, 2] - p[ia, 2]
                    L = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if L < 1e-12:
                        continue
                    C = L - crest[c]
                    ty = ctype[c]
                    if it == 0 and ty == 0:
                        if C > 0.0:
                            strain += C / crest[c]
                        nstr += 1
                    if ty == 0 and C < 0.0:
                        at = comp[3] / (dt * dt)
                    else:
                        at = comp[ty] / (dt * dt)
                    dl = (-C - at * lam[c]) / (wsum + at)
                    lam[c] += dl
                    cx = dl * dx / L
                    cy = dl * dy / L
                    cz = dl * dz / L
                    p[ia, 0] -= w[ia] * cx
                    p[ia, 1] -= w[ia] * cy
                    p[ia, 2] -= w[ia] * cz
                    p[ib, 0] += w[ib] * cx
                    p[ib, 1] += w[ib] * cy
                    p[ib, 2] += w[ib] * cz
                # isometric bending: second differences -> 0 (linear in curvature, kills grid-scale crumples)
                for c in range(NB):
                    ia = ta_[c]
                    im = tm_[c]
                    ib = tb_[c]
                    wsum = w[im] + 0.25 * (w[ia] + w[ib])
                    if wsum == 0.0:
                        continue
                    den = 1.0 / (wsum + atb)
                    for k in range(3):
                        Ck = p[im, k] - 0.5 * (p[ia, k] + p[ib, k])
                        dl = (-Ck - atb * lamv[c, k]) * den
                        lamv[c, k] += dl
                        p[im, k] += w[im] * dl
                        p[ia, k] -= 0.5 * w[ia] * dl
                        p[ib, k] -= 0.5 * w[ib] * dl
                # long-range attachments: never farther from the hoist than the cloth allows
                for j in range(ny):
                    for i in range(1, nx):
                        q = j * nx + i
                        dx = p[q, 0] - hs[j, 0]
                        dy = p[q, 1] - hs[j, 1]
                        dz = p[q, 2] - hs[j, 2]
                        L = math.sqrt(dx * dx + dy * dy + dz * dz)
                        md = uu[q] * Wc * 1.01
                        if L > md:
                            sc = md / L
                            p[q, 0] = hs[j, 0] + dx * sc
                            p[q, 1] = hs[j, 1] + dy * sc
                            p[q, 2] = hs[j, 2] + dz * sc

            # ---- furl (roll around the pole top)
            if fr > 0.0:
                fw = min(1.0, fr * fr * (1.35 - 0.35 * fr) + 0.0)
                for j in range(ny):
                    for i in range(1, nx):
                        q = j * nx + i
                        fx, fy, fz = furl_target(j, i, nx, ny, Wc, Hc, hs[j, 0], hs[j, 1], hs[j, 2], a, side, rp)
                        p[q, 0] += (fx - p[q, 0]) * fw
                        p[q, 1] += (fy - p[q, 1]) * fw
                        p[q, 2] += (fz - p[q, 2]) * fw

            # ---- velocities, stats
            esum = 0.0
            for q in range(N):
                v[q, 0] = (p[q, 0] - x[q, 0]) / dt
                v[q, 1] = (p[q, 1] - x[q, 1]) / dt
                v[q, 2] = (p[q, 2] - x[q, 2]) / dt
                x[q, 0] = p[q, 0]
                x[q, 1] = p[q, 1]
                x[q, 2] = p[q, 2]
                if w[q] > 0.0:
                    ex = v[q, 0] - vhx
                    ey = v[q, 1] - vhy
                    ez = v[q, 2] - vhz
                    esum += math.sqrt(ex * ex + ey * ey + ez * ez)
            esum /= (N - ny)
            if esum > emax:
                emax = esum
            if nstr > 0:
                st = strain / nstr
                if st > tmax:
                    tmax = st
            fd = (v[fly, 0] - vhx) * fnx + (v[fly, 1] - vhy) * fny + (v[fly, 2] - vhz) * fnz
            if abs(fd) > 0.08:
                sg = 1.0 if fd > 0 else -1.0
                if last_sign != 0.0 and sg != last_sign:
                    crossings += 1.0
                last_sign = sg
            for j in range(ny):
                for k in range(3):
                    hp[j, k] = hs[j, k]

        if kf >= warm:
            o = kf - warm
            for j in range(ny):
                for i in range(nx):
                    for k in range(3):
                        out_x[o, j, i, k] = x[j * nx + i, k]
            out_e[o] = emax
            out_t[o] = tmax
            out_c[o] = crossings
            ex = v[fly, 0] - vhx
            ey = v[fly, 1] - vhy
            ez = v[fly, 2] - vhz
            out_fly[o] = math.sqrt(ex * ex + ey * ey + ez * ez)
    return 0


def build_constraints(nx, ny, W, H):
    dx = W / (nx - 1)
    dy = H / (ny - 1)
    A, B, R, T = [], [], [], []

    def add(j0, i0, j1, i1, r, t):
        A.append(j0 * nx + i0)
        B.append(j1 * nx + i1)
        R.append(r)
        T.append(t)

    dd = math.hypot(dx, dy)
    TA, TM, TB = [], [], []
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx:
                add(j, i, j, i + 1, dx, 0)
            if j + 1 < ny:
                add(j, i, j + 1, i, dy, 0)
            if i + 1 < nx and j + 1 < ny:
                add(j, i, j + 1, i + 1, dd, 1)
                add(j, i + 1, j + 1, i, dd, 1)
            if i + 2 < nx:
                TA.append(j * nx + i)
                TM.append(j * nx + i + 1)
                TB.append(j * nx + i + 2)
            if j + 2 < ny:
                TA.append(j * nx + i)
                TM.append((j + 1) * nx + i)
                TB.append((j + 2) * nx + i)
    I = lambda a: np.array(a, np.int64)
    return (I(A), I(B), np.array(R, np.float64), I(T), I(TA), I(TM), I(TB))

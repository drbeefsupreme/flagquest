"""m10 - THE STICKY FLAG: the vx.flag cloth model plus a sticky sleeve (scene-local; owner M10).

Stickiness is shown ONLY by behaviour: the cloth that touches Master Crocus's sleeve stays glued to it and
follows the arm. Same solver, same aerodynamics, same renderer as every other Flag in the film (the returned
FlagTrack is drawn with FlagTrack.draw / vx.flag.draw_cloth, the plain-yellow rule-enforcing renderer).

simulate_sticky(flag, n, pole_fn, wind_fn, arm_fn, stick_fn, arm_r=18.0, fps=24, substeps=10, warmup=48)
    flag       vx.flag.Flag (size, seed, side, stiffness ... as usual)
    pole_fn(i) -> (x, y, ang)        pole base, as Flag.simulate
    wind_fn(i) -> (wx, wy[, wz])     mean wind, as Flag.simulate
    arm_fn(i)  -> (ax, ay, az, bx, by, bz) or None   the forearm as a capsule (elbow -> cuff), design px,
                                     z toward the camera; None = the arm is away (no collision)
    stick_fn(i) -> 0..1              > 0.5: cloth touching the sleeve now sticks to it (glue never lets go)
    -> (FlagTrack, info) with info = dict(glued=(n,) glued particle count, touch=(n,) new contacts per frame)

The sleeve pushes cloth to its FRONT (toward the camera): the Flag is composited last, over the figure, so the
cloth must physically lie in front of the arm it clings to.

The kernel is vx.flag_sim.simulate (FlagSmith) with the capsule collision + glue added inside the constraint
loop; keep the physics identical so this Flag moves like every other Flag in the film.
"""
import math

import numpy as np
from numba import njit

from vx import flag as _flag
from vx import flag_sim as _sim
from vx.flag_sim import GRAV, RHO, SIGMA, _cr, _hoist, _h, furl_target, noise1, vnoise3


@njit(cache=True, fastmath=True)
def _simulate_sticky(nx, ny, Wc, Hc, Lp, poses, cuts, winds, furl, warm, fps, base_sub, iters, seed,
                     prm, s_shear, s_bend, side, rp, ca_, cb_, crest, ctype, ta_, tm_, tb_,
                     arms, arm_r, stick,
                     out_x, out_e, out_t, out_c, out_fly, out_glued, out_touch):
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
    comp[3] = prm[8] * winv * dt_ref * dt_ref

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
    glued = np.zeros(N, np.bool_)
    gs = np.zeros(N)            # glue: parameter along the forearm (0 elbow .. 1 cuff)
    g1 = np.zeros(N)            # glue: offset along the arm's front axis
    g2 = np.zeros(N)            # glue: offset along the arm's side axis
    for j in range(ny):
        w[j * nx] = 0.0
        for i in range(nx):
            uu[j * nx + i] = i / (nx - 1)
            vv[j * nx + i] = j / (ny - 1)

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
    rcol = arm_r * 1.02

    for kf in range(1, T):
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

        k0 = max(kf - 2, 0)
        k3 = min(kf + 1, T - 1)
        p0x = poses[kf - 1, 0] + math.sin(poses[kf - 1, 2]) * Lp
        p0y = poses[kf - 1, 1] - math.cos(poses[kf - 1, 2]) * Lp
        p1x = poses[kf, 0] + math.sin(poses[kf, 2]) * Lp
        p1y = poses[kf, 1] - math.cos(poses[kf, 2]) * Lp
        disp = 0.0 if cuts[kf] else math.hypot(p1x - p0x, p1y - p0y)
        disp = max(disp, 0.0 if cuts[kf] else math.hypot(poses[kf, 0] - poses[kf - 1, 0],
                                                          poses[kf, 1] - poses[kf - 1, 1]))
        arm_on = arms[kf, 0] > 0.5 and arms[kf - 1, 0] > 0.5
        if arm_on:
            for e in range(2):
                o = 1 + 3 * e
                dd = math.sqrt((arms[kf, o] - arms[kf - 1, o]) ** 2 + (arms[kf, o + 1] - arms[kf - 1, o + 1]) ** 2
                               + (arms[kf, o + 2] - arms[kf - 1, o + 2]) ** 2)
                disp = max(disp, dd)
        wm = math.sqrt(winds[kf, 0] ** 2 + winds[kf, 1] ** 2 + winds[kf, 2] ** 2)
        ns = base_sub
        ns = max(ns, int(math.ceil(disp / (0.3 * spacing))))
        if arm_on:
            ns = max(ns, int(math.ceil(disp / (0.25 * arm_r))))
        ns = max(ns, int(math.ceil((wm + disp * fps) / fps / (0.5 * spacing))))
        ns = min(ns, 240)
        dt = 1.0 / (fps * ns)
        damp = 0.2
        if kf < warm * 0.5:
            damp = 3.0
        dfac = math.exp(-damp * dt)
        sticky = stick[kf] > 0.5

        emax = 0.0
        tmax = 0.0
        crossings = 0.0
        touch = 0.0
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
            # ---- the sleeve (capsule elbow A -> cuff B) and its frame: d along, e1 front (toward camera), e2 side
            Ax = 0.0
            Ay = 0.0
            Az = 0.0
            dlx = 0.0
            dly = 0.0
            dlz = 0.0
            e1x = 0.0
            e1y = 0.0
            e1z = 1.0
            e2x = 0.0
            e2y = 0.0
            e2z = 0.0
            La = 0.0
            if arm_on:
                Ax = _cr(arms[k0, 1], arms[kf - 1, 1], arms[kf, 1], arms[k3, 1], al)
                Ay = _cr(arms[k0, 2], arms[kf - 1, 2], arms[kf, 2], arms[k3, 2], al)
                Az = _cr(arms[k0, 3], arms[kf - 1, 3], arms[kf, 3], arms[k3, 3], al)
                Bx = _cr(arms[k0, 4], arms[kf - 1, 4], arms[kf, 4], arms[k3, 4], al)
                By = _cr(arms[k0, 5], arms[kf - 1, 5], arms[kf, 5], arms[k3, 5], al)
                Bz = _cr(arms[k0, 6], arms[kf - 1, 6], arms[kf, 6], arms[k3, 6], al)
                dlx = Bx - Ax
                dly = By - Ay
                dlz = Bz - Az
                La = math.sqrt(dlx * dlx + dly * dly + dlz * dlz) + 1e-9
                dlx /= La
                dly /= La
                dlz /= La
                # front axis: +z made perpendicular to the forearm
                dz = dlz
                e1x = -dz * dlx
                e1y = -dz * dly
                e1z = 1.0 - dz * dlz
                le = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z) + 1e-12
                e1x /= le
                e1y /= le
                e1z /= le
                e2x = dly * e1z - dlz * e1y
                e2y = dlz * e1x - dlx * e1z
                e2z = dlx * e1y - dly * e1x
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

            urx = wmx - vhx
            ury = wmy - vhy
            urz = wmz - vhz
            ubar = math.sqrt(urx * urx + ury * ury + urz * urz)
            ff = fcoef * ubar / Wc * (1.0 + 0.3 * noise1(t * 0.5, seed + 91))
            ff = min(ff, 4.5)
            phase += ff * dt
            fmod = flutter * (0.65 + 0.45 * noise1(t * 0.8 + 3.0, seed + 17))
            boost = 1.5 * min(1.0, ubar / 2.0) ** 2

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
                        e1x_ = x[ib, 0] - x[ia, 0]
                        e1y_ = x[ib, 1] - x[ia, 1]
                        e1z_ = x[ib, 2] - x[ia, 2]
                        e2x_ = x[ic, 0] - x[ia, 0]
                        e2y_ = x[ic, 1] - x[ia, 1]
                        e2z_ = x[ic, 2] - x[ia, 2]
                        nxx = e1y_ * e2z_ - e1z_ * e2y_
                        nyy = e1z_ * e2x_ - e1x_ * e2z_
                        nzz = e1x_ * e2y_ - e1y_ * e2x_
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
                # ---- the sleeve: glued cloth follows it; free cloth is pushed to its front and may stick
                if arm_on:
                    for q in range(N):
                        if w[q] == 0.0:
                            continue
                        if glued[q]:
                            ox_ = Ax + dlx * gs[q] * La + e1x * g1[q] + e2x * g2[q]
                            oy_ = Ay + dly * gs[q] * La + e1y * g1[q] + e2y * g2[q]
                            oz_ = Az + dlz * gs[q] * La + e1z * g1[q] + e2z * g2[q]
                            p[q, 0] = ox_
                            p[q, 1] = oy_
                            p[q, 2] = oz_
                            continue
                        rx = p[q, 0] - Ax
                        ry = p[q, 1] - Ay
                        rz = p[q, 2] - Az
                        sp = (rx * dlx + ry * dly + rz * dlz) / La
                        if sp < 0.0 or sp > 1.0:
                            continue
                        cx = Ax + dlx * sp * La
                        cy = Ay + dly * sp * La
                        cz = Az + dlz * sp * La
                        ox_ = p[q, 0] - cx
                        oy_ = p[q, 1] - cy
                        oz_ = p[q, 2] - cz
                        o1 = ox_ * e1x + oy_ * e1y + oz_ * e1z
                        o2 = ox_ * e2x + oy_ * e2y + oz_ * e2z
                        rr = math.sqrt(o1 * o1 + o2 * o2)
                        if rr >= rcol:
                            continue
                        # inside the sleeve: out through the FRONT half (o1 >= 0) along the radial direction
                        if o1 < 0.0:
                            o1 = -o1
                        if rr < 1e-9:
                            o1 = 1.0
                            o2 = 0.0
                            rr = 1.0
                        nr = math.sqrt(o1 * o1 + o2 * o2)
                        o1 = o1 / nr * rcol
                        o2 = o2 / nr * rcol
                        p[q, 0] = cx + e1x * o1 + e2x * o2
                        p[q, 1] = cy + e1y * o1 + e2y * o2
                        p[q, 2] = cz + e1z * o1 + e2z * o2
                        if sticky:
                            glued[q] = True
                            gs[q] = sp
                            g1[q] = o1
                            g2[q] = o2
                            touch += 1.0

            if fr > 0.0:
                fw = min(1.0, fr * fr * (1.35 - 0.35 * fr) + 0.0)
                for j in range(ny):
                    for i in range(1, nx):
                        q = j * nx + i
                        fx, fy, fz = furl_target(j, i, nx, ny, Wc, Hc, hs[j, 0], hs[j, 1], hs[j, 2], a, side, rp)
                        p[q, 0] += (fx - p[q, 0]) * fw
                        p[q, 1] += (fy - p[q, 1]) * fw
                        p[q, 2] += (fz - p[q, 2]) * fw

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
            ng = 0.0
            for q in range(N):
                if glued[q]:
                    ng += 1.0
            out_glued[o] = ng
            out_touch[o] = touch
    return 0


def simulate_sticky(flag, n, pole_fn, wind_fn, arm_fn, stick_fn, arm_r=18.0, fps=24, substeps=10, warmup=48):
    """see module docstring. Mirrors vx.flag.Flag.simulate (gusts, cuts, units) + the sticky sleeve."""
    n = int(n)
    warm = int(max(0, warmup))
    T = warm + n
    mpp = _sim.POLE_M / flag.pole
    poses_px = np.array([pole_fn(i) for i in range(n)], np.float64).reshape(n, 3)
    winds_px = np.zeros((n, 3))
    arms_px = np.zeros((n, 7))
    stick = np.zeros(n)
    for i in range(n):
        wv = tuple(wind_fn(i))
        winds_px[i, :len(wv)] = wv[:3]
        a = arm_fn(i)
        if a is not None:
            arms_px[i, 0] = 1.0
            arms_px[i, 1:] = a
        stick[i] = float(stick_fn(i))
    poses = np.empty((T, 3))
    winds = np.empty((T, 3))
    arms = np.empty((T, 7))
    sticks = np.empty(T)
    poses[:warm] = poses_px[0]
    poses[warm:] = poses_px
    winds[:warm] = winds_px[0]
    winds[warm:] = winds_px
    arms[:warm] = arms_px[0]
    arms[warm:] = arms_px
    sticks[:warm] = 0.0
    sticks[warm:] = stick
    poses_m = poses.copy()
    poses_m[:, :2] *= mpp
    arms_m = arms.copy()
    arms_m[:, 1:] *= mpp
    winds_m = winds * _sim.WIND_MPP
    tt = np.arange(T) / fps
    g = np.array([_sim.noise1(t * 0.35, flag.seed + 5) + 0.5 * _sim.noise1(t * 0.9, flag.seed + 6) for t in tt])
    winds_m *= (1.0 + 0.32 * flag.turbulence * g)[:, None]
    cuts = np.zeros(T, np.bool_)
    dpos = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))
    dang = np.abs(np.diff(poses[:, 2]))
    cuts[1:] = (dpos > 0.6 * flag.pole) | (dang > 1.2)
    Wc = flag.width * mpp
    Hc = flag.height * mpp
    A, B, R, Ty, TA, TM, TB = _sim.build_constraints(flag.nx, flag.ny, Wc, Hc)
    out_x = np.zeros((n, flag.ny, flag.nx, 3))
    out_e = np.zeros(n)
    out_t = np.zeros(n)
    out_c = np.zeros(n)
    out_f = np.zeros(n)
    out_g = np.zeros(n)
    out_h = np.zeros(n)
    st = flag.stiffness
    A_ = _flag.AERO
    prm = np.array([flag.turbulence, A_["flutter"] * flag.flutter, A_["cl"], A_["cd"], A_["cf"],
                    A_["fcoef"], A_["lam"], A_["span"], A_["comp"]], np.float64)
    furls = np.zeros(T)
    _simulate_sticky(flag.nx, flag.ny, Wc, Hc, _sim.POLE_M, poses_m, cuts, winds_m, furls, warm, float(fps),
                     int(substeps) * 3, A_["iters"], flag.seed, prm,
                     A_["shear"] / st, A_["bend"] / st, float(flag.side), flag.pole_w * mpp * 0.5, A, B, R, Ty,
                     TA, TM, TB, arms_m, float(arm_r) * mpp, sticks,
                     out_x, out_e, out_t, out_c, out_f, out_g, out_h)
    verts = (out_x / mpp).astype(np.float32)
    energy, snap, flap = _flag._postprocess_audio(out_e, out_t, out_x, out_f, fps)
    meta = {"pole": flag.pole, "pole_w": flag.pole_w, "side": flag.side, "width": flag.width,
            "height": flag.height, "seed": flag.seed}
    return _flag.FlagTrack(verts, poses_px.astype(np.float32), energy, snap, meta, flap), \
        dict(glued=out_g.astype(np.float32), touch=out_h.astype(np.float32))

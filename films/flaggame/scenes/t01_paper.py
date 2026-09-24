"""t01 — a real bending-paper simulation of a saddle-stitched booklet (the tract).  Owner: T01Tract.

Physics (cgs units: cm, g, s), XPBD with small steps (numba):
  * L leaves (each = half of a folded sheet), each a NU x NV particle grid; column 0 of every leaf is the
    SHARED spine (the fold + two staples): all leaves hinge on the same line.
  * in-plane: hard stretch + shear distance constraints (paper does not stretch)
  * bending: discrete-Laplacian constraints along rows/columns (curvature) with a compliance = stiffness;
    WET paper is ~4x softer than dry (it droops); the cover stock is stiffer.
  * crease memory at the fold: each sheet (leaf k + leaf L-1-k) wants to stay folded near its crease angle.
  * leaf ordering: a booklet's leaves can never pass through each other: at every (row, column) the angles
    of consecutive leaves around the spine axis are kept ordered with a paper-thickness gap.
  * ground: contact with the mud heightfield (stack-aware offsets), sticky wet-mud friction.
  * air: per-triangle pressure drag from the relative wind, only on EXPOSED faces (a face covered by the
    next leaf or lying on the mud feels no wind), + a boundary-layer wind profile and gust updrafts.
Output: per frame particle grids -> BookTrack (leaf grids (L, NV, NU, 3)), per-frame paper energy for foley.
"""
import math

import numpy as np
from numba import njit

PAGE_W_CM = 12.0          # across (spine -> free edge)
PAGE_H_CM = 12.0 * 1298.0 / 1000.0   # along the spine (page aspect of t01_page)
NU, NV = 13, 13
LEAVES = 6
THICK = 0.045             # layer spacing used by contacts (a bit more than real paper)
DX_CM = PAGE_W_CM / (NU - 1)


def crease_rest(angle):
    """|x_a + x_c - 2 x_spine| of a sheet folded to `angle` (0 = closed flat, pi = open flat)"""
    return 2 * DX_CM * math.cos(angle / 2)


CREST_CLOSED = crease_rest(0.35)


def pid(k, i, j):
    if i == 0:
        return j
    return NV + k * (NU - 1) * NV + (i - 1) * NV + j


N_PART = NV + LEAVES * (NU - 1) * NV


def _build_constraints():
    dx = PAGE_W_CM / (NU - 1)
    dy = PAGE_H_CM / (NV - 1)
    dist = []      # (a, b, rest, kind) kind 0 stretch, 1 shear
    bend = []      # (a, b, c, leaf) with b the middle: C = |xa + xc - 2 xb|
    for k in range(LEAVES):
        for i in range(NU):
            for j in range(NV):
                if i + 1 < NU:
                    dist.append((pid(k, i, j), pid(k, i + 1, j), dx, 0, k))
                if j + 1 < NV and i > 0:
                    dist.append((pid(k, i, j), pid(k, i, j + 1), dy, 0, k))
                if i + 1 < NU and j + 1 < NV:
                    dd = math.hypot(dx, dy)
                    dist.append((pid(k, i, j), pid(k, i + 1, j + 1), dd, 1, k))
                    dist.append((pid(k, i + 1, j), pid(k, i, j + 1), dd, 1, k))
                if 0 < i < NU - 1:
                    bend.append((pid(k, i - 1, j), pid(k, i, j), pid(k, i + 1, j), k))
                if 0 < j < NV - 1 and i > 0:
                    bend.append((pid(k, i, j - 1), pid(k, i, j), pid(k, i, j + 1), k))
    # the spine itself: stiff line
    for j in range(NV - 1):
        dist.append((j, j + 1, dy, 0, -1))
    for j in range(1, NV - 1):
        bend.append((j - 1, j, j + 1, -1))
    D = np.array([d[:3] for d in dist], np.float64)
    Dk = np.array([d[3] for d in dist], np.int64)
    Dl = np.array([d[4] for d in dist], np.int64)
    B = np.array([b[:3] for b in bend], np.int64)
    Bl = np.array([b[3] for b in bend], np.int64)
    # crease: (leaf k col 1, spine j, leaf L-1-k col 1)
    cr = []
    for k in range(LEAVES // 2):
        for j in range(NV):
            cr.append((pid(k, 1, j), j, pid(LEAVES - 1 - k, 1, j), k))
    C = np.array(cr, np.int64)
    return D, Dk, Dl, B, Bl, C


@njit(cache=True)
def _rot(v, a, ang):
    c = math.cos(ang)
    s = math.sin(ang)
    cx = a[1] * v[2] - a[2] * v[1]
    cy = a[2] * v[0] - a[0] * v[2]
    cz = a[0] * v[1] - a[1] * v[0]
    d = a[0] * v[0] + a[1] * v[1] + a[2] * v[2]
    return (v[0] * c + cx * s + a[0] * d * (1 - c), v[1] * c + cy * s + a[1] * d * (1 - c),
            v[2] * c + cz * s + a[2] * d * (1 - c))


@njit(cache=True)
def _ground(G, gx0, gy0, gdx, gdy, x, y):
    n = G.shape[0]
    u = (x - gx0) / gdx
    v = (y - gy0) / gdy
    if u < 0:
        u = 0.0
    if v < 0:
        v = 0.0
    if u > n - 1.001:
        u = n - 1.001
    if v > n - 1.001:
        v = n - 1.001
    i = int(u)
    j = int(v)
    fu = u - i
    fv = v - j
    return (G[j, i] * (1 - fu) * (1 - fv) + G[j, i + 1] * fu * (1 - fv) + G[j + 1, i] * (1 - fu) * fv
            + G[j + 1, i + 1] * fu * fv)


@njit(cache=True)
def _vnoise(x, seed):
    i = math.floor(x)
    f = x - i
    u = f * f * (3 - 2 * f)
    a = ((int(i) * 374761393 + seed * 668265263) & 0xFFFFFF) / 16777215.0
    b = ((int(i + 1) * 374761393 + seed * 668265263) & 0xFFFFFF) / 16777215.0
    a = (a * 2654435761.0) % 1.0
    b = (b * 2654435761.0) % 1.0
    return (a + (b - a) * u) * 2 - 1


@njit(cache=True)
def _step(X, Xp, V, invm, D, Dk, Dl, B, Bl, C, crest, comp_s, comp_b, comp_c, G, gx0, gy0, gdx, gdy,
          wind, updraft, upx, lift_gain, adh, KT, PIN, pin_w, TURN, t, dt, nu, nv, nl, thick, fric, rank, E):
    n = X.shape[0]
    # ------------------------------------------------ external forces: gravity + aero (per triangle)
    F = np.zeros((n, 3))
    rho = 0.0012
    # upwind extreme of the booklet (suction lift peaks at the upwind edge, where the flow separates)
    xup = -1e9
    for p in range(n):
        v = X[p, 0] if wind[0] < 0 else -X[p, 0]
        if v > xup:
            xup = v
    for k in range(nl):
        for i in range(nu - 1):
            for j in range(nv - 1):
                a = j if i == 0 else nv + k * (nu - 1) * nv + (i - 1) * nv + j
                b = nv + k * (nu - 1) * nv + i * nv + j
                c = (j + 1) if i == 0 else nv + k * (nu - 1) * nv + (i - 1) * nv + j + 1
                d = nv + k * (nu - 1) * nv + i * nv + j + 1
                for tri in range(2):
                    if tri == 0:
                        p0, p1, p2 = a, b, c
                    else:
                        p0, p1, p2 = b, d, c
                    e1x = X[p1, 0] - X[p0, 0]
                    e1y = X[p1, 1] - X[p0, 1]
                    e1z = X[p1, 2] - X[p0, 2]
                    e2x = X[p2, 0] - X[p0, 0]
                    e2y = X[p2, 1] - X[p0, 1]
                    e2z = X[p2, 2] - X[p0, 2]
                    nx = e1y * e2z - e1z * e2y
                    ny = e1z * e2x - e1x * e2z
                    nz = e1x * e2y - e1y * e2x
                    ln = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
                    area = 0.5 * ln
                    nx /= ln
                    ny /= ln
                    nz /= ln
                    cx = (X[p0, 0] + X[p1, 0] + X[p2, 0]) / 3
                    cy = (X[p0, 1] + X[p1, 1] + X[p2, 1]) / 3
                    cz = (X[p0, 2] + X[p1, 2] + X[p2, 2]) / 3
                    hg = cz - _ground(G, gx0, gy0, gdx, gdy, cx, cy)
                    if hg < 0:
                        hg = 0.0
                    # boundary layer: the wind is weak right at the mud, full a few cm up
                    prof = 0.30 + 0.70 * min(1.0, hg / 3.0)
                    # turbulence
                    tu = _vnoise(t * 3.1 + cx * 0.05, 11) * 0.28 + _vnoise(t * 7.3 + cy * 0.09, 12) * 0.14
                    wx = wind[0] * prof * (1 + tu)
                    wy = wind[1] * prof * (1 + tu * 0.5) + wind[0] * 0.15 * _vnoise(t * 2.3 + cx * 0.04, 13)
                    wz = wind[2] * prof
                    # gust updraft: air deflected up over the upwind edge of whatever lies on the mud
                    if updraft > 0:
                        side = (cx - upx) * (1.0 if wind[0] < 0 else -1.0)
                        wz += updraft * math.exp(-side * side / 30.0) * (0.6 + 0.4 * min(1.0, hg / 2.0))
                    vx = (V[p0, 0] + V[p1, 0] + V[p2, 0]) / 3
                    vy = (V[p0, 1] + V[p1, 1] + V[p2, 1]) / 3
                    vz = (V[p0, 2] + V[p1, 2] + V[p2, 2]) / 3
                    rx, ry, rz = wx - vx, wy - vy, wz - vz
                    vn = rx * nx + ry * ny + rz * nz
                    # exposure of the face the air pushes on: vn > 0 pushes the BACK face (-n side)
                    ex = 1.0
                    if vn > 0:
                        ex = E[k, 1, i, j]
                        if nz > 0.2:
                            ex *= min(1.0, hg / 1.2)
                    else:
                        ex = E[k, 0, i, j]
                        if nz < -0.2:
                            ex *= min(1.0, hg / 1.2)
                    f = rho * 1.2 * vn * abs(vn) * area * ex * lift_gain[k]
                    # skin friction (tangential)
                    tx = rx - vn * nx
                    ty = ry - vn * ny
                    tz = rz - vn * nz
                    fs = rho * 0.02 * area * ex * lift_gain[k]
                    # suction lift on the exposed UPPER face when wind sweeps across it (Bernoulli)
                    su = 0.0
                    if abs(nz) > 0.3:
                        eu = E[k, 0, i, j] if nz > 0 else E[k, 1, i, j]
                        dd = xup - (cx if wind[0] < 0 else -cx)
                        U2 = rx * rx + ry * ry
                        su = rho * 0.55 * U2 * area * eu * math.exp(-dd / 4.5) * abs(nz) * lift_gain[k] * math.exp(-hg / 1.5)
                        if nz < 0:
                            su = -su
                    f += su
                    for p in (p0, p1, p2):
                        F[p, 0] += (f * nx + fs * tx * abs(tx)) / 3
                        F[p, 1] += (f * ny + fs * ty * abs(ty)) / 3
                        F[p, 2] += (f * nz + fs * tz * abs(tz)) / 3
    # the director's breath: a signed turning acceleration per leaf (air scooping under a free edge),
    # tangential around the spine axis, growing toward the free edge
    for k in range(nl):
        if TURN[k] == 0.0:
            continue
        for j in range(nv):
            j0 = j - 1 if j > 0 else 0
            j1 = j + 1 if j < nv - 1 else nv - 1
            ax = X[j1, 0] - X[j0, 0]
            ay = X[j1, 1] - X[j0, 1]
            az = X[j1, 2] - X[j0, 2]
            la = math.sqrt(ax * ax + ay * ay + az * az) + 1e-12
            ax /= la
            ay /= la
            az /= la
            for i in range(1, nu):
                p = nv + k * (nu - 1) * nv + (i - 1) * nv + j
                rx = X[p, 0] - X[j, 0]
                ry = X[p, 1] - X[j, 1]
                rz = X[p, 2] - X[j, 2]
                d = rx * ax + ry * ay + rz * az
                rx -= d * ax
                ry -= d * ay
                rz -= d * az
                # tangent = -(axis x r) normalised
                tx = -(ay * rz - az * ry)
                ty = -(az * rx - ax * rz)
                tz = -(ax * ry - ay * rx)
                lt = math.sqrt(tx * tx + ty * ty + tz * tz) + 1e-12
                wgt = (i / (nu - 1.0)) ** 1.5
                if invm[p] > 0:
                    F[p, 0] += TURN[k] * wgt * tx / lt / invm[p]
                    F[p, 1] += TURN[k] * wgt * ty / lt / invm[p]
                    F[p, 2] += TURN[k] * wgt * tz / lt / invm[p]
    for p in range(n):
        if invm[p] > 0:
            V[p, 0] += dt * F[p, 0] * invm[p]
            V[p, 1] += dt * F[p, 1] * invm[p]
            V[p, 2] += dt * (F[p, 2] * invm[p] - 981.0)
            # air damping
            V[p, 0] *= 1 - 0.6 * dt
            V[p, 1] *= 1 - 0.6 * dt
            V[p, 2] *= 1 - 0.6 * dt
        Xp[p, 0] = X[p, 0]
        Xp[p, 1] = X[p, 1]
        Xp[p, 2] = X[p, 2]
        if p < nv:
            # the stapled spine is driven kinematically (the director's rigid path)
            X[p, 0] = KT[p, 0]
            X[p, 1] = KT[p, 1]
            X[p, 2] = KT[p, 2]
        else:
            X[p, 0] += dt * V[p, 0]
            X[p, 1] += dt * V[p, 1]
            X[p, 2] += dt * V[p, 2]
    dt2 = dt * dt
    # ------------------------------------------------ distance constraints (hard stretch, shear)
    for q in range(D.shape[0]):
        a = int(D[q, 0])
        b = int(D[q, 1])
        rest = D[q, 2]
        dxv = X[b, 0] - X[a, 0]
        dyv = X[b, 1] - X[a, 1]
        dzv = X[b, 2] - X[a, 2]
        L = math.sqrt(dxv * dxv + dyv * dyv + dzv * dzv) + 1e-12
        Cv = L - rest
        w = invm[a] + invm[b]
        if w == 0:
            continue
        al = comp_s[Dk[q]] / dt2
        lam = -Cv / (w + al)
        gx, gy, gz = dxv / L, dyv / L, dzv / L
        X[a, 0] -= invm[a] * lam * gx
        X[a, 1] -= invm[a] * lam * gy
        X[a, 2] -= invm[a] * lam * gz
        X[b, 0] += invm[b] * lam * gx
        X[b, 1] += invm[b] * lam * gy
        X[b, 2] += invm[b] * lam * gz
    # ------------------------------------------------ bending (discrete Laplacian, rest = flat)
    for q in range(B.shape[0]):
        a = B[q, 0]
        b = B[q, 1]
        c = B[q, 2]
        lx = X[a, 0] + X[c, 0] - 2 * X[b, 0]
        ly = X[a, 1] + X[c, 1] - 2 * X[b, 1]
        lz = X[a, 2] + X[c, 2] - 2 * X[b, 2]
        L = math.sqrt(lx * lx + ly * ly + lz * lz)
        if L < 1e-9:
            continue
        gx, gy, gz = lx / L, ly / L, lz / L
        w = invm[a] + invm[c] + 4 * invm[b]
        if w == 0:
            continue
        lf = Bl[q]
        al = comp_b[lf + 1] / dt2
        lam = -L / (w + al)
        X[a, 0] += invm[a] * lam * gx
        X[a, 1] += invm[a] * lam * gy
        X[a, 2] += invm[a] * lam * gz
        X[c, 0] += invm[c] * lam * gx
        X[c, 1] += invm[c] * lam * gy
        X[c, 2] += invm[c] * lam * gz
        X[b, 0] -= 2 * invm[b] * lam * gx
        X[b, 1] -= 2 * invm[b] * lam * gy
        X[b, 2] -= 2 * invm[b] * lam * gz
    # ------------------------------------------------ crease memory at the fold
    for q in range(C.shape[0]):
        a = C[q, 0]
        b = C[q, 1]
        c = C[q, 2]
        sheet = C[q, 3]
        lx = X[a, 0] + X[c, 0] - 2 * X[b, 0]
        ly = X[a, 1] + X[c, 1] - 2 * X[b, 1]
        lz = X[a, 2] + X[c, 2] - 2 * X[b, 2]
        L = math.sqrt(lx * lx + ly * ly + lz * lz)
        if L < 1e-9:
            continue
        Cv = L - crest[sheet]
        gx, gy, gz = lx / L, ly / L, lz / L
        w = invm[a] + invm[c] + 4 * invm[b]
        if w == 0:
            continue
        al = comp_c / dt2
        lam = -Cv / (w + al)
        X[a, 0] += invm[a] * lam * gx
        X[a, 1] += invm[a] * lam * gy
        X[a, 2] += invm[a] * lam * gz
        X[c, 0] += invm[c] * lam * gx
        X[c, 1] += invm[c] * lam * gy
        X[c, 2] += invm[c] * lam * gz
        X[b, 0] -= 2 * invm[b] * lam * gx
        X[b, 1] -= 2 * invm[b] * lam * gy
        X[b, 2] -= 2 * invm[b] * lam * gz
    # ------------------------------------------------ leaf ordering around the spine (no interpenetration)
    for j in range(nv):
        j0 = j - 1 if j > 0 else 0
        j1 = j + 1 if j < nv - 1 else nv - 1
        ax = X[j1, 0] - X[j0, 0]
        ay = X[j1, 1] - X[j0, 1]
        az = X[j1, 2] - X[j0, 2]
        la = math.sqrt(ax * ax + ay * ay + az * az) + 1e-12
        axis = (ax / la, ay / la, az / la)
        sx, sy, sz = X[j, 0], X[j, 1], X[j, 2]
        for i in range(1, nu):
            for k in range(nl - 1):
                pa = nv + k * (nu - 1) * nv + (i - 1) * nv + j
                pb = nv + (k + 1) * (nu - 1) * nv + (i - 1) * nv + j
                ra = (X[pa, 0] - sx, X[pa, 1] - sy, X[pa, 2] - sz)
                rb = (X[pb, 0] - sx, X[pb, 1] - sy, X[pb, 2] - sz)
                # project out the axis component
                da = ra[0] * axis[0] + ra[1] * axis[1] + ra[2] * axis[2]
                db = rb[0] * axis[0] + rb[1] * axis[1] + rb[2] * axis[2]
                pax = (ra[0] - da * axis[0], ra[1] - da * axis[1], ra[2] - da * axis[2])
                pbx = (rb[0] - db * axis[0], rb[1] - db * axis[1], rb[2] - db * axis[2])
                na = math.sqrt(pax[0] ** 2 + pax[1] ** 2 + pax[2] ** 2) + 1e-9
                nb = math.sqrt(pbx[0] ** 2 + pbx[1] ** 2 + pbx[2] ** 2) + 1e-9
                crx = pbx[1] * pax[2] - pbx[2] * pax[1]
                cry = pbx[2] * pax[0] - pbx[0] * pax[2]
                crz = pbx[0] * pax[1] - pbx[1] * pax[0]
                sn = crx * axis[0] + cry * axis[1] + crz * axis[2]
                cs = pbx[0] * pax[0] + pbx[1] * pax[1] + pbx[2] * pax[2]
                delta = -math.atan2(sn, cs)          # >0 when leaf k lies "above" leaf k+1
                r = 0.5 * (na + nb)
                eps = thick / max(r, 0.3)
                corr = 0.0
                if delta < eps and delta > -0.6:
                    corr = min(eps - delta, 0.02)
                elif adh[k] > 0 and delta < eps + 0.35:
                    # wet pages cling together (surface tension): pull back to the contact gap
                    corr = -min((delta - eps) * adh[k] * 0.05, 0.0012)
                if corr != 0.0:
                    wa = invm[pa]
                    wb = invm[pb]
                    if wa + wb == 0:
                        continue
                    # rotate a by -corr*wa/(wa+wb), b by +corr*wb/(wa+wb)  (sign: increasing delta)
                    ang_a = -corr * wa / (wa + wb)
                    ang_b = corr * wb / (wa + wb)
                    va = _rot(ra, axis, ang_a)
                    vb = _rot(rb, axis, ang_b)
                    X[pa, 0], X[pa, 1], X[pa, 2] = sx + va[0], sy + va[1], sz + va[2]
                    X[pb, 0], X[pb, 1], X[pb, 2] = sx + vb[0], sy + vb[1], sz + vb[2]
    # ------------------------------------------------ the director's hand: the free edge held on the mud (flip hinge)
    if pin_w > 0:
        for k in range(nl):
            for j in range(nv):
                p = nv + k * (nu - 1) * nv + (nu - 2) * nv + j
                tz = PIN[j, 2] + rank[p] * thick
                X[p, 0] += (PIN[j, 0] - X[p, 0]) * pin_w
                X[p, 1] += (PIN[j, 1] - X[p, 1]) * pin_w
                X[p, 2] += (tz - X[p, 2]) * pin_w
    # ------------------------------------------------ ground contact (stack-aware) + sticky friction
    for p in range(nv, n):
        g = _ground(G, gx0, gy0, gdx, gdy, X[p, 0], X[p, 1]) + 0.03 + rank[p] * thick
        if X[p, 2] < g:
            X[p, 2] = g
            # friction: undo most tangential motion of this substep
            mx = X[p, 0] - Xp[p, 0]
            my = X[p, 1] - Xp[p, 1]
            X[p, 0] -= mx * fric
            X[p, 1] -= my * fric
    # ------------------------------------------------ velocities
    for p in range(n):
        V[p, 0] = (X[p, 0] - Xp[p, 0]) / dt
        V[p, 1] = (X[p, 1] - Xp[p, 1]) / dt
        V[p, 2] = (X[p, 2] - Xp[p, 2]) / dt
        sp = math.sqrt(V[p, 0] ** 2 + V[p, 1] ** 2 + V[p, 2] ** 2)
        if sp > 900.0:
            V[p, 0] *= 900.0 / sp
            V[p, 1] *= 900.0 / sp
            V[p, 2] *= 900.0 / sp


@njit(cache=True)
def _aux(X, nu, nv, nl, rank, E, thick):
    """stack rank (how many leaves lie right below each particle) and per-face exposure gaps"""
    for k in range(nl):
        for i in range(nu):
            for j in range(nv):
                p = j if i == 0 else nv + k * (nu - 1) * nv + (i - 1) * nv + j
                cnt = 0
                gup = 99.0
                gdn = 99.0
                for k2 in range(nl):
                    if k2 == k:
                        continue
                    q = j if i == 0 else nv + k2 * (nu - 1) * nv + (i - 1) * nv + j
                    dx = X[q, 0] - X[p, 0]
                    dy = X[q, 1] - X[p, 1]
                    dz = X[q, 2] - X[p, 2]
                    if dx * dx + dy * dy < 0.8 * 0.8:
                        if dz < 0:
                            cnt += 1
                    d = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if k2 == k - 1:
                        gup = d
                    if k2 == k + 1:
                        gdn = d
                if i > 0:
                    rank[p] = cnt
                # exposure: 0 = front face (towards leaf k-1), 1 = back face (towards leaf k+1)
                if i < nu - 1 and j < nv - 1:
                    e0 = min(1.0, max(0.0, (gup - 0.15) / 1.2))
                    e1 = min(1.0, max(0.0, (gdn - 0.15) / 1.2))
                    E[k, 0, i, j] = e0
                    E[k, 1, i, j] = e1


class BookTrack:
    """leaf grids per frame: verts (n, L, NV, NU, 3) cm; energy (n,) paper motion for foley"""

    def __init__(self, verts, t0, fps, energy, events):
        self.verts = verts
        self.t0 = t0
        self.fps = fps
        self.energy = energy
        self.events = events

    def at(self, T):
        f = (T - self.t0) * self.fps
        i = int(math.floor(f))
        a = f - i
        n = len(self.verts)
        i0 = min(max(i, 0), n - 1)
        i1 = min(max(i + 1, 0), n - 1)
        return self.verts[i0] * (1 - a) + self.verts[i1] * a


def leaf_grids(X):
    """particle array -> (L, NV, NU, 3)"""
    out = np.empty((LEAVES, NV, NU, 3))
    for k in range(LEAVES):
        out[k, :, 0] = X[:NV]
        blk = X[NV + k * (NU - 1) * NV: NV + (k + 1) * (NU - 1) * NV].reshape(NU - 1, NV, 3)
        out[k, :, 1:] = blk.transpose(1, 0, 2)
    return out


def initial_face_down(center, ground_fn):
    """closed booklet lying face-down (back cover up), spine on the +x side, page top toward +y"""
    cx, cy = center
    X = np.zeros((N_PART, 3))
    dx = PAGE_W_CM / (NU - 1)
    dy = PAGE_H_CM / (NV - 1)
    xs = cx + PAGE_W_CM / 2      # spine on the right
    y0 = cy - PAGE_H_CM / 2
    for k in range(LEAVES):
        for i in range(NU):
            for j in range(NV):
                x = xs - i * dx
                y = y0 + j * dy
                z = ground_fn(x, y) + 0.03 + k * THICK
                X[pid(k, i, j)] = (x, y, z)
    return X


def simulate(ground, t_start, t_end, fps=24, substeps=40, controls=None, center=None, warmup=1.0, seed=3, init=None):
    """controls(T) -> dict(wind=(wx,wy,wz), updraft=float, upx=float, lift=(L,), crest=(L//2,), wet=0..1)"""
    D, Dk, Dl, B, Bl, C = _build_constraints()
    gx0, gy0 = center[0] - 45.0, center[1] - 45.0
    Gn = 361
    Ggrid, rect = ground.grid(gx0, gy0, gx0 + 90.0, gy0 + 90.0, Gn)
    gdx = rect[2] / (Gn - 1)
    gdy = rect[3] / (Gn - 1)
    X = init() if init is not None else initial_face_down(center, lambda x, y: float(ground.height(x, y)))
    Xp = X.copy()
    V = np.zeros_like(X)
    # masses: area density wet paper ~0.016 g/cm^2, cover stock heavier
    cell = (PAGE_W_CM / (NU - 1)) * (PAGE_H_CM / (NV - 1))
    invm = np.empty(N_PART)
    for k in range(LEAVES):
        dens = 0.030 if k in (0, LEAVES - 1) else 0.018
        for i in range(1, NU):
            for j in range(NV):
                w = 0.5 if (i == NU - 1) else 1.0
                w *= 0.5 if j in (0, NV - 1) else 1.0
                invm[pid(k, i, j)] = 1.0 / (dens * cell * w)
    for j in range(NV):
        invm[j] = 0.0                                     # kinematic spine
    comp_s = np.array([0.0, 1e-7])
    rank = np.zeros(N_PART, np.int64)
    E = np.ones((LEAVES, 2, NU, NV))
    dt = 1.0 / (fps * substeps)
    n_frames = int(round((t_end - t_start) * fps))
    out = np.empty((n_frames, LEAVES, NV, NU, 3))
    energy = np.zeros(n_frames)
    nw = int(round(warmup * fps * substeps))
    total = nw + n_frames * substeps
    fr = 0
    for s in range(total):
        T = t_start + (s - nw) / (fps * substeps)
        c = controls(max(T, t_start), X)
        wet = c.get("wet", 1.0)
        # bending compliance per leaf class (index 0 = spine line, 1.. = leaves): wet paper is softer
        cb = np.empty(LEAVES + 1)
        cb[0] = 1e-6
        for k in range(LEAVES):
            base = 0.0022 if k in (0, LEAVES - 1) else 0.0045
            cb[k + 1] = base * (1 + 3.0 * wet)
        if s % substeps == 0:
            _aux(X, NU, NV, LEAVES, rank, E, THICK)
        crest = np.asarray(c.get("crest", [CREST_CLOSED] * (LEAVES // 2)), np.float64)
        lift = np.asarray(c.get("lift", [1.0] * LEAVES), np.float64)
        p0, p1 = c["spine"]
        if "pin" in c:
            q0, q1 = c["pin"]
            PIN = np.asarray(q0, np.float64)[None] + (np.asarray(q1, np.float64) - np.asarray(q0, np.float64))[None] * \
                np.linspace(0, 1, NV)[:, None]
        else:
            PIN = np.zeros((NV, 3))
        KT = np.asarray(p0, np.float64)[None] + (np.asarray(p1, np.float64) - np.asarray(p0, np.float64))[None] * \
            np.linspace(0, 1, NV)[:, None]
        _step(X, Xp, V, invm, D, Dk, Dl, B, Bl, C, crest, comp_s, cb, float(c.get("crease_comp", 0.004)), Ggrid,
              gx0, gy0, gdx, gdy, np.asarray(c["wind"], np.float64), float(c.get("updraft", 0.0)),
              float(c.get("upx", center[0])), lift, np.asarray(c.get("adh", [1.0] * (LEAVES - 1)), np.float64), KT, PIN, float(c.get("pin_w", 0.0)),
              np.asarray(c.get("turn", [0.0] * LEAVES), np.float64), T, dt, NU, NV, LEAVES, THICK, float(c.get("fric", 0.25)), rank, E)
        if s >= nw and (s - nw) % substeps == substeps - 1 and fr < n_frames:
            out[fr] = leaf_grids(X)
            energy[fr] = float(np.sqrt((V[NV:] ** 2).sum(1)).mean())
            fr += 1
    return BookTrack(out, t_start, fps, energy, [])

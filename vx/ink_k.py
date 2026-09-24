"""Numba kernels of the print shop (vx.ink).  Owner: Printer.

One fused per-pixel kernel does: line-art separation (solid black kept crisp, AA edges un-mixed from
the tone underneath), the tone screen (AM dots, engraved line layers, or a morph between two screens),
ink spread + edge roughness, ink mottle / pinholes, and the paper stock. Everything is sampled in
PAGE coordinates (design px on the page, see ink._page_map) so screens/textures can ride a moving page.
"""
import math

import numpy as np
from numba import njit, prange

TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------- parameter slots (P array)
MODE_A, MODE_B, MIX, USE_MIXMAP = 0, 1, 2, 3
CX, OX, OY, INVS = 4, 5, 6, 7
CELL, CA, SA = 8, 9, 10
LO_D, HI_D, LO_L, HI_L = 11, 12, 13, 14
GAIN, ROUGH, SPREAD, LROUGH = 15, 16, 17, 18
LGAIN = 19
K = 20
N1X, N1Y, N2X, N2Y, N3X, N3Y = 21, 22, 23, 24, 25, 26
GCELL, WIG, TAPER, NLAY = 27, 28, 29, 30
R0, R1, PM = 31, 32, 33
PR, PG, PB, IR, IG, IB = 35, 36, 37, 38, 39, 40
NOX, NOY = 41, 42
MOTT, PIN = 43, 44
A1, A2 = 45, 46
USE_PH1, USE_PH2, USE_PH3, USE_PHM = 47, 48, 49, 50
RMIN, DASH, USE_PSI = 51, 52, 53
PAPER_ON = 54
GOX, GOY = 55, 56
MA00, MA01, MA10, MA11 = 57, 58, 59, 60
NP = 64

DOTS, LINES = 0, 1


@njit(fastmath=True, cache=True)
def _wrap(tile, x, y):
    """bilinear sample of a periodic power-of-two tile at texel coords (x, y)"""
    n0 = tile.shape[0]
    n1 = tile.shape[1]
    xf = math.floor(x)
    yf = math.floor(y)
    fx = x - xf
    fy = y - yf
    x0 = int(xf) & (n1 - 1)
    y0 = int(yf) & (n0 - 1)
    x1 = (x0 + 1) & (n1 - 1)
    y1 = (y0 + 1) & (n0 - 1)
    a = tile[y0, x0] + (tile[y0, x1] - tile[y0, x0]) * fx
    b = tile[y1, x0] + (tile[y1, x1] - tile[y1, x0]) * fx
    return a + (b - a) * fy


@njit(fastmath=True, cache=True)
def _grid(g, L, gx, gy):
    """clamped bilinear sample of layer L of a coarse (nL, gh, gw) grid; cell centres at integers"""
    gh = g.shape[1]
    gw = g.shape[2]
    if gx < 0.0:
        gx = 0.0
    elif gx > gw - 1.0:
        gx = gw - 1.0
    if gy < 0.0:
        gy = 0.0
    elif gy > gh - 1.0:
        gy = gh - 1.0
    x0 = int(gx)
    y0 = int(gy)
    x1 = x0 + 1 if x0 + 1 < gw else x0
    y1 = y0 + 1 if y0 + 1 < gh else y0
    fx = gx - x0
    fy = gy - y0
    a = g[L, y0, x0] + (g[L, y0, x1] - g[L, y0, x0]) * fx
    b = g[L, y1, x0] + (g[L, y1, x1] - g[L, y1, x0]) * fx
    return a + (b - a) * fy


@njit(fastmath=True, cache=True)
def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@njit(fastmath=True, cache=True)
def _box(r, x):
    """exact 1-px box-filtered coverage of a line of half-width r (px) at distance x >= 0 (px)"""
    if r <= 0.0:
        return 0.0
    hi = x + 0.5
    if hi > r:
        hi = r
    lo = x - 0.5
    if lo < -r:
        lo = -r
    c = hi - lo
    return 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)


@njit(fastmath=True, cache=True)
def _hash(n):
    v = math.sin(n * 12.9898 + 78.233) * 43758.5453
    return v - math.floor(v)


@njit(fastmath=True, cache=True)
def _dot_T(px, py, cell, ca, sa, cx, lutA, lutG):
    """calibrated threshold T (area of ink at tone d is {T < d}) of the round-dot screen + |grad T| per px"""
    u = (px * ca + py * sa) / cell
    v = (py * ca - px * sa) / cell
    tu = TWO_PI * u
    tv = TWO_PI * v
    f = 0.5 * (math.cos(tu) + math.cos(tv))
    su = math.sin(tu)
    sv = math.sin(tv)
    gf = math.pi * math.sqrt(su * su + sv * sv) / cell * cx
    NL = lutA.shape[0]
    t = (f + 1.0) * 0.5 * (NL - 1)
    i0 = int(t)
    if i0 > NL - 2:
        i0 = NL - 2
    if i0 < 0:
        i0 = 0
    fr = t - i0
    T = lutA[i0] + (lutA[i0 + 1] - lutA[i0]) * fr
    G = (lutG[i0] + (lutG[i0 + 1] - lutG[i0]) * fr) * gf
    if G < 1e-4:
        G = 1e-4
    return T, G


@njit(fastmath=True, cache=True)
def _phase(L, px, py, i, j, gx, gy, P, psi, ph, phm, wg):
    """engraving phase (in line periods) of layer L at page point (px, py) / pixel (i, j);
    wg = the burin's slow wiggle at this point (shared by the layers, weighted per layer)"""
    if L == 0:
        nx = P[N1X]
        ny = P[N1Y]
        ww = 1.0
    elif L == 1:
        nx = P[N2X]
        ny = P[N2Y]
        ww = -0.7
    else:
        nx = P[N3X]
        ny = P[N3Y]
        ww = 0.5
    phi = P[K] * (px * nx + py * ny) + ww * wg
    if P[USE_PSI] > 0.5:
        phi += _grid(psi, L, gx, gy)
    if P[USE_PH1 + L] > 0.5:
        pv = ph[i, j]
        if P[USE_PHM] > 0.5:
            m = phm[i, j]
            phi = phi + (pv - phi) * m
        else:
            phi = pv
    return phi


@njit(fastmath=True, cache=True)
def _ihash(n):
    """cheap integer hash of a line index -> [0, 1)"""
    k = (int(n) * 2654435761) & 0xFFFFFFFF
    k = ((k >> 15) ^ k) * 2246822519 & 0xFFFFFFFF
    return ((k >> 13) & 0xFFFF) / 65536.0


@njit(fastmath=True, cache=True)
def _lines_cov(dm, px, py, i, j, gx, gy, nf, wg, P, psi, ph1, ph2, ph3, phm, nzm):
    """engraving coverage, width form: up to 3 line layers (hatch + 2 crosshatch levels)"""
    kpx = P[K] * P[CX]
    Ppx = 1.0 / kpx                    # line period in output pixels
    a1 = P[A1]
    a2 = P[A2]
    nlay = int(P[NLAY])
    rmin = P[RMIN]
    edge = P[LGAIN] + P[ROUGH] * nf
    keep = 1.0
    # ---- layer 1: the form-following hatch; the burin's pressure swells and thins it, and in the
    #      light it breaks into tapered flicks (long irregular dashes) instead of grey hairlines
    w1 = dm if dm < a1 else a1
    phi = _phase(0, px, py, i, j, gx, gy, P, psi, ph1, phm, wg)
    fl = math.floor(phi)
    f = abs(phi - fl - 0.5) * 2.0
    x = f * 0.5 * Ppx
    along = py * P[N1X] - px * P[N1Y]
    tn = _wrap(nzm, along / P[DASH] + 97.0 * _ihash(fl), fl * 1.618)
    r = w1 * 0.5 * Ppx * (1.0 + P[TAPER] * tn)
    if r < rmin:
        duty = r / rmin if r > 0.0 else 0.0
        e = duty - 0.5 + 1.1 * tn          # > 0 inside a flick
        if e > 0.0:
            c1 = _box(rmin * min(1.0, e * 5.0) + edge * duty, x)
        else:
            c1 = 0.0
    else:
        c1 = _box(r + edge, x)
    keep *= 1.0 - c1
    # ---- layer 2: first crosshatch
    if nlay >= 2 and dm > a1:
        w2 = 1.0 - (1.0 - dm) / (1.0 - a1)
        if w2 > a2:
            w2 = a2
        phi = _phase(1, px, py, i, j, gx, gy, P, psi, ph2, phm, wg)
        f = abs(phi - math.floor(phi) - 0.5) * 2.0
        c2 = _box(w2 * 0.5 * Ppx + edge, f * 0.5 * Ppx)
        keep *= 1.0 - c2
        # ---- layer 3: second crosshatch for the deepest shadow
        d2 = 1.0 - (1.0 - a1) * (1.0 - a2)
        if nlay >= 3 and dm > d2:
            w3 = 1.0 - (1.0 - dm) / ((1.0 - a1) * (1.0 - a2))
            phi = _phase(2, px, py, i, j, gx, gy, P, psi, ph3, phm, wg)
            f = abs(phi - math.floor(phi) - 0.5) * 2.0
            c3 = _box(w3 * 0.5 * Ppx + edge, f * 0.5 * Ppx)
            keep *= 1.0 - c3
    return 1.0 - keep


@njit(fastmath=True, cache=True)
def _lines_T(px, py, i, j, gx, gy, wg, P, psi, ph1, ph2, ph3, phm):
    """engraving as a threshold function (for morphing screens): black iff T < tone"""
    a1 = P[A1]
    a2 = P[A2]
    d2 = 1.0 - (1.0 - a1) * (1.0 - a2)
    g = 2.0 * P[K] * P[CX]
    phi = _phase(0, px, py, i, j, gx, gy, P, psi, ph1, phm, wg)
    f = abs(phi - math.floor(phi) - 0.5) * 2.0
    if f < a1:
        T = f
        G = g
    else:
        T = a1 + (f - a1) * 6.0
        G = 6.0 * g
    nlay = int(P[NLAY])
    if nlay >= 2:
        phi = _phase(1, px, py, i, j, gx, gy, P, psi, ph2, phm, wg)
        f = abs(phi - math.floor(phi) - 0.5) * 2.0
        if f < a2:
            T2 = a1 + (1.0 - a1) * f
            G2 = (1.0 - a1) * g
        else:
            T2 = d2 + (f - a2) * 6.0 * (1.0 - a1)
            G2 = 6.0 * (1.0 - a1) * g
        if T2 < T:
            T = T2
            G = G2
        if nlay >= 3:
            phi = _phase(2, px, py, i, j, gx, gy, P, psi, ph3, phm, wg)
            f = abs(phi - math.floor(phi) - 0.5) * 2.0
            T3 = d2 + (1.0 - d2) * f
            if T3 < T:
                T = T3
                G = (1.0 - d2) * g
    return T, G


# ---------------------------------------------------------------- phase-field solver (coarse grid)
@njit(fastmath=True, cache=True)
def _apply(x, lam, ih2, out):
    """out = (D^T D) x / h^2 + lam * x on a grid with free (Neumann) borders"""
    gh, gw = x.shape
    for i in range(gh):
        for j in range(gw):
            c = x[i, j]
            s = 0.0
            if i > 0:
                s += c - x[i - 1, j]
            if i < gh - 1:
                s += c - x[i + 1, j]
            if j > 0:
                s += c - x[i, j - 1]
            if j < gw - 1:
                s += c - x[i, j + 1]
            out[i, j] = s * ih2 + lam[i, j] * c


@njit(fastmath=True, cache=True)
def cg_solve(rhs, lam, x, h, iters, tol):
    """Jacobi-preconditioned conjugate gradient for ((D^T D)/h^2 + diag(lam)) x = rhs; x = warm start"""
    gh, gw = rhs.shape
    ih2 = 1.0 / (h * h)
    dg = np.empty((gh, gw))
    for i in range(gh):
        for j in range(gw):
            nb = (1 if i > 0 else 0) + (1 if i < gh - 1 else 0) + (1 if j > 0 else 0) + (1 if j < gw - 1 else 0)
            dg[i, j] = 1.0 / (nb * ih2 + lam[i, j])
    r = np.empty((gh, gw))
    _apply(x, lam, ih2, r)
    bn = 0.0
    for i in range(gh):
        for j in range(gw):
            r[i, j] = rhs[i, j] - r[i, j]
            bn += rhs[i, j] * rhs[i, j]
    if bn <= 0.0:
        bn = 1e-30
    z = r * dg
    p = z.copy()
    rz = np.sum(r * z)
    Ap = np.empty((gh, gw))
    for it in range(iters):
        _apply(p, lam, ih2, Ap)
        pAp = np.sum(p * Ap)
        if pAp <= 0.0:
            break
        a = rz / pAp
        rr = 0.0
        for i in range(gh):
            for j in range(gw):
                x[i, j] += a * p[i, j]
                r[i, j] -= a * Ap[i, j]
                rr += r[i, j] * r[i, j]
        if rr < tol * tol * bn:
            break
        rzn = 0.0
        for i in range(gh):
            for j in range(gw):
                z[i, j] = r[i, j] * dg[i, j]
                rzn += r[i, j] * z[i, j]
        b = rzn / rz
        rz = rzn
        for i in range(gh):
            for j in range(gw):
                p[i, j] = z[i, j] + b * p[i, j]
    return x


@njit(parallel=True, fastmath=True, cache=True)
def k_print(L, LMIN, LMAX, out, P, lutA, lutG, psi, ph1, ph2, ph3, phm, mixmap, nzf, nzm, pap0, pap1):
    """L: luminance (h, w) f32 (1 = paper); LMIN / LMAX: its 3x3 local min / max; out: (h, w, 3) f32"""
    h = L.shape[0]
    w = L.shape[1]
    modeA = int(P[MODE_A])
    modeB = int(P[MODE_B])
    usemix = P[MIX] > 0.0 or P[USE_MIXMAP] > 0.5
    lines = modeA == LINES or (usemix and modeB == LINES)
    cx = P[CX]
    ox = P[OX]
    oy = P[OY]
    invs = P[INVS]
    gcell = P[GCELL]
    two_paper = P[PM] > 1e-4
    for i in prange(h):
        yp = i + 0.5
        bx = P[MA01] * yp + ox
        by = P[MA11] * yp + oy
        gy = (yp * invs - P[GOY]) / gcell - 0.5
        for j in range(w):
            xp = j + 0.5
            px = P[MA00] * xp + bx
            py = P[MA10] * xp + by
            gx = ((j + 0.5) * invs - P[GOX]) / gcell - 0.5
            sv = 1.0 - L[i, j]
            mx = 1.0 - LMIN[i, j]
            nf = _wrap(nzf, px + P[NOX], py + P[NOY])
            wg = P[WIG] * _wrap(nzm, px * 0.09, py * 0.09) if lines else 0.0
            # ---- line art: solid ink and its AA edges are un-mixed from the tone underneath
            c = 0.0
            d = sv
            if mx > 0.8:
                wl = (mx - 0.8) / 0.14
                wl = 1.0 if wl > 1.0 else wl
                wl = wl * wl * (3.0 - 2.0 * wl)
                bg = 1.0 - LMAX[i, j]
                if bg > 0.995:
                    cl = 1.0
                else:
                    inv = 1.0 / (1.0 - bg)
                    cl = (sv - bg) * inv
                    cd = (mx - bg) * inv
                    cl = cl + (P[SPREAD] + P[LROUGH] * nf) * (cd - cl)
                    cl = _clamp01(cl)
                c = wl * cl
                d = sv + (bg - sv) * wl
            # ---- tone screen
            if usemix:
                m = mixmap[i, j] if P[USE_MIXMAP] > 0.5 else P[MIX]
                if modeA == DOTS:
                    TA, GA = _dot_T(px, py, P[CELL], P[CA], P[SA], cx, lutA, lutG)
                    dA = (d - P[LO_D]) / (P[HI_D] - P[LO_D])
                else:
                    TA, GA = _lines_T(px, py, i, j, gx, gy, wg, P, psi, ph1, ph2, ph3, phm)
                    dA = (d - P[LO_L]) / (P[HI_L] - P[LO_L])
                if modeB == DOTS:
                    TB, GB = _dot_T(px, py, P[CELL], P[CA], P[SA], cx, lutA, lutG)
                    dB = (d - P[LO_D]) / (P[HI_D] - P[LO_D])
                else:
                    TB, GB = _lines_T(px, py, i, j, gx, gy, wg, P, psi, ph1, ph2, ph3, phm)
                    dB = (d - P[LO_L]) / (P[HI_L] - P[LO_L])
                dm = dA + (dB - dA) * m
                if dm <= 0.0:
                    ct = 0.0
                elif dm >= 1.0:
                    ct = 1.0
                else:
                    T = TA + (TB - TA) * m
                    G = GA + (GB - GA) * m
                    ct = _clamp01((dm - T) / G + 0.5 + P[GAIN] * min(1.0, dm * 6.0) + P[ROUGH] * nf)
                # near either end of the morph, hand over to the native engraving (flicks, taper, box AA)
                # so arriving at / leaving the engraving never pops
                if modeB == LINES and m > 0.8:
                    k = (m - 0.8) / 0.2
                    cn = 0.0 if dB <= 0.0 else (1.0 if dB >= 1.0 else _lines_cov(
                        dB, px, py, i, j, gx, gy, nf, wg, P, psi, ph1, ph2, ph3, phm, nzm))
                    ct = ct + (cn - ct) * k
                elif modeA == LINES and m < 0.2:
                    k = (0.2 - m) / 0.2
                    cn = 0.0 if dA <= 0.0 else (1.0 if dA >= 1.0 else _lines_cov(
                        dA, px, py, i, j, gx, gy, nf, wg, P, psi, ph1, ph2, ph3, phm, nzm))
                    ct = ct + (cn - ct) * k
            elif modeA == DOTS:
                dm = (d - P[LO_D]) / (P[HI_D] - P[LO_D])
                if dm <= 0.0:
                    ct = 0.0
                elif dm >= 1.0:
                    ct = 1.0
                else:
                    T, G = _dot_T(px, py, P[CELL], P[CA], P[SA], cx, lutA, lutG)
                    ct = _clamp01((dm - T) / G + 0.5 + P[GAIN] * min(1.0, dm * 6.0) + P[ROUGH] * nf)
            else:
                dm = (d - P[LO_L]) / (P[HI_L] - P[LO_L])
                if dm <= 0.0:
                    ct = 0.0
                elif dm >= 1.0:
                    ct = 1.0
                else:
                    ct = _lines_cov(dm, px, py, i, j, gx, gy, nf, wg, P, psi, ph1, ph2, ph3, phm, nzm)
            cov = c + (1.0 - c) * ct
            # ---- ink character: starved specks in heavy ink
            if P[PIN] > 0.0 and cov > 0.5:
                n2 = _wrap(nzf, px * 0.5 + 97.0, py * 0.5 + 331.0)
                sp = _clamp01((n2 - 0.45) * 3.0)
                cov *= 1.0 - P[PIN] * sp * sp
            # ---- paper stock + ink film
            if P[PAPER_ON] > 0.5:
                tex = _wrap(pap0, px * P[R0], py * P[R0])
                if two_paper:
                    tex += (_wrap(pap1, px * P[R1], py * P[R1]) - tex) * P[PM]
                im = 1.0 + P[MOTT] * _wrap(nzm, px * 0.03 + 5.0, py * 0.03 + 17.0) if cov > 0.0 else 1.0
            else:
                tex = 1.0
                im = 1.0
            kp = 1.0 - cov
            ki = cov * im
            out[i, j, 0] = P[PR] * tex * kp + P[IR] * ki
            out[i, j, 1] = P[PG] * tex * kp + P[IG] * ki
            out[i, j, 2] = P[PB] * tex * kp + P[IB] * ki

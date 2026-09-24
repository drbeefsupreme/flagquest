"""t03 op-art genesis field (owner T03Genesis).

An infinite black/white Vasarely/Riley hexagon tessellation on the ground plane Y=0, rendered per pixel
(numba) through the t03 perspective camera with an exact box-filtered square-wave (anisotropic, from the
screen->ground Jacobian) + 2x2 supersampling, so it stays crisp from the flags' feet to the horizon.
Three cell families (3-colouring of the hex lattice): concentric hex rings, pinwheel chevrons, zigzag
nested triangles. The stripes breathe (duty cycle wave travelling outward), the chevrons rotate, a global
circular grating XORs over everything (Riley moire), and a swirl can twist the whole floor (the storm).
NO colour anywhere: ink coverage only.

handoff(fc, T) -> float32 rgb    exact op-art frame at global time T (no flags). T02Wrong imports it for
                                 the 43.29 hand-off; valid for any T (T < 43.29 = almost solid black).
tau_of(T)                        flashback story time (s since 43.29; pauses while the present interrupts)
genesis_cam(tau) -> Cam          the genesis camera path
opart_cov(fc, cam, prm) -> (h,w) ink coverage;  params(T, tau, **over) -> prm
"""
import math

import numpy as np
from numba import njit
from scipy.interpolate import PchipInterpolator

from vx import ink
from scenes.t03_cam import Cam

T0 = 43.29166666666667          # scene start (genesis)
SQ3 = math.sqrt(3.0)

# ------------------------------------------------------------------ story time (the flashback pauses)
PAUSE_A = (49.28, 49.50)        # decelerate to a stop as Summer interrupts from the present (S05 49.42)
PAUSE_B = (52.14, 52.62)        # "...merely the ABSENCE" (52.30): the flashback starts again


def _speed(T):
    return 1.0 - _ss(PAUSE_A[0], PAUSE_A[1], T) + _ss(PAUSE_B[0], PAUSE_B[1], T)


def _ss(a, b, x):
    t = np.clip((np.asarray(x) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


_TT = np.arange(T0 - 3.0, 70.0, 0.0005)
_SP = _speed(_TT)
_TAU = np.concatenate([[0.0], np.cumsum(0.5 * (_SP[1:] + _SP[:-1]) * 0.0005)])
_TAU -= np.interp(T0, _TT, _TAU)


def tau_of(T):
    """flashback story time: T - 43.29 except during the S05 pause (about 2.9 s of frozen time)"""
    if T <= _TT[0]:
        return float(T - T0)
    if T >= _TT[-1]:
        return float(_TAU[-1] + (T - _TT[-1]))
    return float(np.interp(T, _TT, _TAU))


def T_of_tau(tau):
    """inverse of tau_of (global time at which story time reaches tau)"""
    return float(np.interp(tau, _TAU, _TT))


def pause_amount(T):
    """0 running .. 1 fully frozen"""
    return float(1.0 - _speed(T))


# ------------------------------------------------------------------ genesis camera path
_K_TAU = [-2.0, 0.0, 2.3, 6.1, 9.45, 12.0]
_K_ELEV = [80.0, 76.0, 47.0, 29.0, 17.0, 14.0]
_K_DIST = [4.2, 4.5, 5.7, 6.1, 5.9, 5.6]
_K_YAW = [-0.25, 0.0, 0.42, 1.30, 1.95, 2.20]
_K_TY = [0.05, 0.10, 0.62, 0.88, 0.90, 0.90]
_I_ELEV = PchipInterpolator(_K_TAU, _K_ELEV)
_I_DIST = PchipInterpolator(_K_TAU, _K_DIST)
_I_YAW = PchipInterpolator(_K_TAU, _K_YAW)
_I_TY = PchipInterpolator(_K_TAU, _K_TY)
F_GEN = 1500.0
CX_GEN = 1190.0                 # lens shift: the ring sits right of centre, the left third is for lettering


def genesis_cam(tau):
    t = min(max(tau, _K_TAU[0]), _K_TAU[-1])
    return Cam(yaw=float(_I_YAW(t)), elev=math.radians(float(_I_ELEV(t))), dist=float(_I_DIST(t)), f=F_GEN,
               target=(0.0, float(_I_TY(t)), 0.0), cx=CX_GEN)


# ------------------------------------------------------------------ pattern parameters
R_CELL = 1.0                    # hex circumradius (world units); the six flags stand on the central cell's vertices


def params(T, tau=None, **over):
    """pattern parameter vector at global time T"""
    if tau is None:
        tau = tau_of(T)
    opening = _ss(T0 - 0.05, T0 + 1.05, T)                      # the void breathes open into the pattern
    p = dict(R=R_CELL, tau=tau, open=float(opening), amp=0.30, moire=0.0, far0=4.5, far1=12.0,
             twist=0.0, twist_r=3.0, rotB=0.55, wave=0.0, grout=0.07)
    # Riley moire rises after the flags open, peaks on "Red, Blue and Yellow", eases off for the pause
    p["moire"] = float(0.38 * _ss(45.8, 47.4, T) * (1.0 - 0.6 * _ss(49.6, 51.0, T)) + 0.25 * _ss(52.4, 54.0, T))
    p["wave"] = float(0.18 * _ss(46.0, 48.0, T))
    # the floor starts to swirl (foreshadowing the hurricane) before the dissolve into the engraving
    p["twist"] = float(1.6 * _ss(53.2, 56.2, T) ** 2)
    p.update(over)
    return p


_KEYS = ("R", "tau", "open", "amp", "moire", "far0", "far1", "twist", "twist_r", "rotB", "wave", "grout")


def _vec(p):
    return np.array([p[k] for k in _KEYS], np.float64)


@njit(cache=True, fastmath=True, inline="always")
def _F(x, d):
    fl = math.floor(x)
    fr = x - fl
    return fl * d + (fr if fr < d else d)


@njit(cache=True, fastmath=True, inline="always")
def _box(g, gam, d):
    """coverage of a square wave (ink where frac(g) < d) box-filtered over width gam (stripe units)"""
    if gam < 1e-4:
        fr = g - math.floor(g)
        return 1.0 if fr < d else 0.0
    if gam > 4.0:
        return d
    return (_F(g + 0.5 * gam, d) - _F(g - 0.5 * gam, d)) / gam


@njit(cache=True, fastmath=True)
def _pattern(u, v, ju_x, jv_x, ju_y, jv_y, dist, pr):
    R = pr[0]
    tau = pr[1]
    opening = pr[2]
    amp = pr[3]
    moire = pr[4]
    far0 = pr[5]
    far1 = pr[6]
    twist = pr[7]
    twr = pr[8]
    rotB = pr[9]
    wave = pr[10]
    r = math.sqrt(u * u + v * v)
    # --- swirl: rotate by an angle that falls off with radius (hurricane foreshadowing)
    if twist != 0.0:
        a = twist * twr / (r + twr) * (1.0 + 0.25 * math.sin(r * 0.9 - tau * 2.0))
        ca = math.cos(a)
        sa = math.sin(a)
        u, v = u * ca - v * sa, u * sa + v * ca
        ju_x, jv_x = ju_x * ca - jv_x * sa, ju_x * sa + jv_x * ca
        ju_y, jv_y = ju_y * ca - jv_y * sa, ju_y * sa + jv_y * ca
    # --- hex cell (flat-top), cube rounding
    qf = (2.0 / 3.0 * u) / R
    rf = (-1.0 / 3.0 * u + SQ3 / 3.0 * v) / R
    sf = -qf - rf
    qi = round(qf)
    ri = round(rf)
    si = round(sf)
    dq = abs(qi - qf)
    dr = abs(ri - rf)
    ds = abs(si - sf)
    if dq > dr and dq > ds:
        qi = -ri - si
    elif dr > ds:
        ri = -qi - si
    cxh = R * 1.5 * qi
    cyh = R * SQ3 * (ri + qi * 0.5)
    lx = u - cxh
    ly = v - cyh
    # hex distance (0 centre .. 1 edge) and its gradient
    ap = R * SQ3 * 0.5
    p0 = lx * 0.8660254 + ly * 0.5
    p1 = ly
    p2 = -lx * 0.8660254 + ly * 0.5
    m = abs(p0)
    gx = 0.8660254 if p0 >= 0 else -0.8660254
    gy = 0.5 if p0 >= 0 else -0.5
    if abs(p1) > m:
        m = abs(p1)
        gx = 0.0
        gy = 1.0 if p1 >= 0 else -1.0
    if abs(p2) > m:
        m = abs(p2)
        gx = -0.8660254 if p2 >= 0 else 0.8660254
        gy = 0.5 if p2 >= 0 else -0.5
    hd = m / ap
    gx /= ap
    gy /= ap
    ang = math.atan2(ly, lx)
    if ang < 0:
        ang += 2 * math.pi
    k = int(ang / (math.pi / 3.0)) % 6
    typ = int(qi - ri) % 3
    if typ < 0:
        typ += 3
    cring = max(abs(qi), max(abs(ri), abs(qi + ri)))
    hsh = math.sin(qi * 12.9898 + ri * 78.233) * 43758.5453
    hsh = hsh - math.floor(hsh)
    rho = r / R
    # --- breathing duty cycle: a wave of thick/thin stripes rolling outward from the flags
    d = 0.5 + amp * math.sin(2 * math.pi * (0.2 * rho - 0.42 * tau) + hsh * 1.3)
    if cring == 0:
        d = 0.5 + 0.22 * math.sin(2 * math.pi * (0.55 * tau) + 1.0)
    # distance: the black thins away into the white void (stays pure b/w, never a grey mush)
    fade = 1.0
    if dist > far0:
        fade = max(0.0, 1.0 - (dist - far0) / (far1 - far0))
        fade = fade * fade * (3 - 2 * fade)
    d = d * (0.35 + 0.65 * fade) * (1.0 if fade > 0 else 0.0)
    if d < 0.06:
        d = 0.06 * fade
    if d > 0.93:
        d = 0.93
    d = d * opening + 0.955 * (1.0 - opening)
    # --- the three families
    if typ == 0 or cring == 0:
        n = 3.0 if cring == 0 else 3.5
        g = hd * n - 0.35 * tau * (1.0 if cring == 0 else -1.0) + 0.02
        gu = gx * n
        gv = gy * n
    elif typ == 1:
        if k % 2 == 0:
            th = (60.0 * k + 90.0) * math.pi / 180.0
        else:
            th = (60.0 * k - 30.0) * math.pi / 180.0
        th += rotB * math.sin(0.45 * tau + hsh * 6.283)
        mx = math.cos(th)
        my = math.sin(th)
        nb = 5.0 / R
        g = (lx * mx + ly * my) * nb + 0.25 * k + wave * math.sin(2 * math.pi * (hd * 1.5 - 0.6 * tau))
        gu = mx * nb
        gv = my * nb
    else:
        n = 6.0
        g = hd * n + 0.5 * (k % 2) - 0.25 * tau
        gu = gx * n
        gv = gy * n
    gsx = gu * ju_x + gv * jv_x
    gsy = gu * ju_y + gv * jv_y
    gam = math.sqrt(gsx * gsx + gsy * gsy)
    c = _box(g, gam, d)
    # --- global circular grating XORed into a band of tiles rolling outward: the Riley moire (stays pure b/w)
    if moire > 0.0 and opening > 0.5:
        band = (cring - tau * 1.1) % 5.0
        if band < 0:
            band += 5.0
        if band < moire * 4.0 and cring > 0:
            n2 = 4.2 / R
            g2 = r * n2 - 0.8 * tau
            if r > 1e-6:
                ru = u / r
                rv = v / r
            else:
                ru = 1.0
                rv = 0.0
            a2x = n2 * (ru * ju_x + rv * jv_x)
            a2y = n2 * (ru * ju_y + rv * jv_y)
            c2 = _box(g2, math.sqrt(a2x * a2x + a2y * a2y), 0.5) * fade
            c = c + c2 - 2.0 * c * c2
    # --- white grout between the cells: the tessellation reads as separate tiles
    gw = pr[11] * opening
    if gw > 0.0:
        ghx = gx * ju_x + gy * jv_x
        ghy = gx * ju_y + gy * jv_y
        gh = math.sqrt(ghx * ghx + ghy * ghy) + 1e-4
        wht = (hd - (1.0 - gw)) / gh + 0.5
        if wht > 0.0:
            c *= 1.0 - (wht if wht < 1.0 else 1.0) * fade
    return c


@njit(cache=True, fastmath=True)
def _render(cov, cp, s, pr, ss):
    h, w = cov.shape
    Cx, Cy, Cz = cp[0], cp[1], cp[2]
    rx, ry, rz = cp[3], cp[4], cp[5]
    dx_, dy_, dz_ = cp[6], cp[7], cp[8]
    fx, fy, fz = cp[9], cp[10], cp[11]
    f = cp[12]
    ccx = cp[13]
    ccy = cp[14]
    inv = 1.0 / (f * s)
    n = ss * ss
    for i in range(h):
        for j in range(w):
            acc = 0.0
            for a in range(ss):
                for b in range(ss):
                    sx = (j + (b + 0.5) / ss) / s
                    sy = (i + (a + 0.5) / ss) / s
                    kx = (sx - ccx) / f
                    ky = (sy - ccy) / f
                    Dx = fx + rx * kx + dx_ * ky
                    Dy = fy + ry * kx + dy_ * ky
                    Dz = fz + rz * kx + dz_ * ky
                    if Dy >= -1e-5:
                        continue
                    t = -Cy / Dy
                    X = Cx + t * Dx
                    Z = Cz + t * Dz
                    # d(ground)/d(device px) for x and y, scaled to the sub-sample footprint
                    kk = t * inv / ss
                    jux = kk * (rx - Dx * ry / Dy)
                    jvx = kk * (rz - Dz * ry / Dy)
                    juy = kk * (dx_ - Dx * dy_ / Dy)
                    jvy = kk * (dz_ - Dz * dy_ / Dy)
                    dist = t * math.sqrt(Dx * Dx + Dy * Dy + Dz * Dz)
                    acc += _pattern(X, Z, jux, jvx, juy, jvy, dist, pr)
            cov[i, j] = acc / n


def opart_cov(fc, cam, prm, ss=2):
    cov = np.zeros((fc.h, fc.w), np.float32)
    _render(cov, cam.params(), float(fc.s), _vec(prm), int(ss))
    return cov


def to_rgb(cov):
    paper = np.array(ink.PAPER, np.float32)
    inkc = np.array(ink.INK, np.float32)
    return paper * (1.0 - cov[..., None]) + inkc * cov[..., None]


def handoff(fc, T):
    """the exact t03 op-art frame at global time T (no flags) - for T02Wrong's ripple dissolve"""
    tau = tau_of(T)
    return to_rgb(opart_cov(fc, genesis_cam(tau), params(T, tau))).astype(np.float32)

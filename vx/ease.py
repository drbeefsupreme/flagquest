"""Easing, interpolation, deterministic noise. Pure functions of time -> frames are independent."""
import math

import numpy as np


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_c(c0, c1, t):
    """lerp two colour tuples"""
    return tuple(a + (b - a) * t for a, b in zip(c0, c1))


def remap(x, a, b, c=0.0, d=1.0, clip=True):
    t = (x - a) / (b - a) if b != a else 0.0
    if clip:
        t = clamp(t)
    return c + (d - c) * t


def smoothstep(a, b, x):
    t = clamp((x - a) / (b - a)) if b != a else float(x >= a)
    return t * t * (3 - 2 * t)


def smootherstep(a, b, x):
    t = clamp((x - a) / (b - a)) if b != a else float(x >= a)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_in(t, p=3):
    return clamp(t) ** p


def ease_out(t, p=3):
    return 1 - (1 - clamp(t)) ** p


def ease_in_out(t, p=3):
    t = clamp(t)
    return 0.5 * (2 * t) ** p if t < 0.5 else 1 - 0.5 * (2 - 2 * t) ** p


def ease_in_out_sine(t):
    return 0.5 - 0.5 * math.cos(math.pi * clamp(t))


def ease_out_back(t, s=1.70158):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def ease_out_elastic(t, period=0.3):
    t = clamp(t)
    if t in (0.0, 1.0):
        return t
    return 2 ** (-10 * t) * math.sin((t - period / 4) * (2 * math.pi) / period) + 1


def seg(t, t0, t1, fn=ease_in_out):
    """eased 0..1 progress of t across [t0, t1]"""
    return fn(clamp((t - t0) / (t1 - t0))) if t1 != t0 else float(t >= t0)


def spring(t, freq=2.2, damp=4.0):
    """0 -> 1 damped spring settle; t in seconds since trigger"""
    if t <= 0:
        return 0.0
    return 1 - math.exp(-damp * t) * math.cos(2 * math.pi * freq * t)


def pulse(t, t0, width):
    """smooth bump centred at t0"""
    x = (t - t0) / max(width, 1e-6)
    return math.exp(-x * x)


# ---------------------------------------------------------------- noise
def hash01(i, seed=0):
    """deterministic pseudo random in [0,1) for integer i"""
    x = (int(i) * 374761393 + int(seed) * 668265263) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFFFF) / float(0x1000000)


def noise1(x, seed=0):
    """smooth 1D value noise in [-1,1]"""
    i = math.floor(x)
    f = x - i
    u = f * f * (3 - 2 * f)
    a = hash01(i, seed) * 2 - 1
    b = hash01(i + 1, seed) * 2 - 1
    return a + (b - a) * u


def fbm1(x, seed=0, octaves=4, lac=2.0, gain=0.5):
    s, a, fr, norm = 0.0, 1.0, 1.0, 0.0
    for o in range(octaves):
        s += a * noise1(x * fr, seed + o * 17)
        norm += a
        a *= gain
        fr *= lac
    return s / norm


def shake(t, amp=4.0, freq=9.0, seed=0):
    """camera shake offset (dx, dy) in design px"""
    return (amp * fbm1(t * freq, seed, 3), amp * fbm1(t * freq, seed + 99, 3))


def value_noise2(xs, ys, seed=0):
    """vectorised 2D value noise in [-1,1] for numpy arrays"""
    xi = np.floor(xs).astype(np.int64)
    yi = np.floor(ys).astype(np.int64)
    xf = xs - xi
    yf = ys - yi
    u = xf * xf * (3 - 2 * xf)
    v = yf * yf * (3 - 2 * yf)

    def h(ix, iy):
        n = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
        n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
        return ((n ^ (n >> 16)) & 0xFFFF) / 32767.5 - 1.0

    a, b = h(xi, yi), h(xi + 1, yi)
    c, d = h(xi, yi + 1), h(xi + 1, yi + 1)
    return (a + (b - a) * u) * (1 - v) + (c + (d - c) * u) * v


def fbm2(xs, ys, seed=0, octaves=4, lac=2.0, gain=0.5):
    s = np.zeros_like(xs, dtype=np.float64)
    a, fr, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        s += a * value_noise2(xs * fr, ys * fr, seed + o * 31)
        norm += a
        a *= gain
        fr *= lac
    return s / norm

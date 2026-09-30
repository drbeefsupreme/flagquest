"""Pastel slop static - the full-frame noise m02 ends on and m03 opens on (hand-off @48.5).

    from scenes.m02_static import pastel_static
    img = pastel_static(fc.w, fc.h, fc.s, fc.T)      # float32 (h, w, 3) sRGB

A pure function of (w, h, s, T): the same global time gives the same frame in both scenes. Grey-pastel TV snow
(luminance ~0.55-0.85, grain ~2 design px) tinted by slowly drifting slop_flesh / slop_cyan / slop_mauve
blotches, with a faint rolling bar and a few magenta/cyan scan-line glitches.
"""
import math

import cv2
import numpy as np

_PASTELS = np.array([
    [0.910, 0.784, 0.816],   # slop_flesh  #E8C8D0
    [0.722, 0.878, 0.910],   # slop_cyan   #B8E0E8
    [0.788, 0.753, 0.659],   # slop_beige  #C9C0A8
    [0.690, 0.640, 0.720],   # slop_mauve, lifted
], np.float32)
_GLITCH = np.array([[1.0, 0.18, 0.53], [0.0, 0.90, 1.0]], np.float32)   # glitch_m, glitch_c


def _field(key, gw, gh):
    """a small deterministic random field for integer key."""
    return np.random.default_rng(key * 7919 + 101).random((gh, gw, len(_PASTELS)), np.float32)


def pastel_static(w, h, s, T):
    F = int(round(T * 24))
    rng = np.random.default_rng(F * 104729 + 12345)
    g = max(1, int(round(2 * s)))                      # grain block in pixels (~2 design px)
    gw, gh = -(-w // g), -(-h // g)
    lum = rng.random((gh, gw), np.float32)
    lum = 0.55 + 0.30 * (lum * lum * (3 - 2 * lum))    # soft-clipped snow
    chroma = (rng.random((gh, gw, 3), np.float32) - 0.5) * 0.10
    snow = lum[..., None] + chroma
    if g > 1:
        snow = np.repeat(np.repeat(snow, g, 0), g, 1)
    snow = snow[:h, :w]
    # slowly drifting pastel blotches (keyframes 6 per second, smoothly interpolated)
    u = T * 6.0
    k0 = math.floor(u)
    a = u - k0
    a = a * a * (3 - 2 * a)
    f = _field(k0, 12, 7) * (1 - a) + _field(k0 + 1, 12, 7) * a
    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
    f = np.maximum(f, 0) ** 3
    f /= f.sum(axis=2, keepdims=True) + 1e-6
    tint = f @ _PASTELS
    tint /= (tint @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None] + 1e-6
    img = snow * (0.55 + 0.45 * tint)
    # rolling bar: a brighter band drifting down the frame
    yy = (np.arange(h, dtype=np.float32) / max(1, h))
    bar = np.exp(-((yy - ((T * 0.37) % 1.3 - 0.15)) / 0.06) ** 2) * 0.10
    img += bar[:, None, None]
    # a few scan-line glitches in magenta / cyan with a small horizontal shift
    for i in range(3):
        y0 = int(rng.random() * h)
        th = max(1, int((1 + rng.random() * 5) * s))
        sh = int((rng.random() - 0.5) * 40 * s)
        c = _GLITCH[i % 2]
        band = img[y0:y0 + th]
        if band.size:
            img[y0:y0 + th] = np.roll(band, sh, axis=1) * 0.75 + c * 0.25
    return np.clip(img, 0, 1).astype(np.float32)

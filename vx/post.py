"""Unified 'film look' applied by the runner to every frame of every scene.
Scenes set module-level POST = {...} and/or define post(fc, state) -> dict of per-frame overrides."""
import cv2
import numpy as np

from .canvas import blur, col

DEFAULT = dict(
    exposure=1.0,
    contrast=1.0,
    saturation=1.0,
    tint=None, tint_k=0.0,          # multiply-tint colour and strength
    bloom=0.35, bloom_thresh=0.72, bloom_radius=22.0,   # radius in design px
    ca=0.0,                         # chromatic aberration (design px at frame edge)
    vignette=0.25,
    grain=0.03,
    fade=0.0, fade_color=(0, 0, 0),  # 1.0 = fully faded to fade_color
    letterbox=0.0,                  # fraction of frame height covered by bars (e.g. 0.12)
)

_cache = {}


def _vig(h, w):
    k = ("vig", h, w)
    if k not in _cache:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / w - 0.5) * 2
        ny = (yy / h - 0.5) * 2
        r2 = (nx * nx * 0.85 + ny * ny) / 1.85
        _cache[k] = r2[..., None]
    return _cache[k]


def _noise(h, w):
    k = ("grain", h, w)
    if k not in _cache:
        rng = np.random.default_rng(1234)
        n = rng.standard_normal((h + 64, w + 64)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 0.6)
        n /= n.std()
        _cache[k] = n
    return _cache[k]


def finish(img, fc, **over):
    p = dict(DEFAULT)
    p.update(over)
    x = img.astype(np.float32, copy=True)
    h, w = x.shape[:2]
    s = fc.s
    if p["exposure"] != 1.0:
        x *= p["exposure"]
    if p["bloom"] > 0:
        thr = p["bloom_thresh"]
        bright = np.maximum(x - thr, 0.0) * (1.0 / max(1e-3, 1 - thr))
        r = p["bloom_radius"] * s
        b = blur(bright, r * 0.35) * 0.45 + blur(bright, r) * 0.35 + blur(bright, r * 3.0) * 0.30
        x += b * p["bloom"]
    if p["saturation"] != 1.0 or p["contrast"] != 1.0:
        L = x[..., 0:1] * 0.2126 + x[..., 1:2] * 0.7152 + x[..., 2:3] * 0.0722
        if p["saturation"] != 1.0:
            x = L + (x - L) * p["saturation"]
        if p["contrast"] != 1.0:
            x = (x - 0.5) * p["contrast"] + 0.5
    if p["tint"] is not None and p["tint_k"] > 0:
        t = np.array(col(p["tint"]), np.float32)
        x = x * (1 - p["tint_k"]) + x * t * p["tint_k"]
    if p["ca"] > 0:
        a = p["ca"] * s / (w / 2)
        cx, cy = w / 2, h / 2
        for ch, k in ((0, 1 + a), (2, 1 - a)):
            M = np.array([[k, 0, cx * (1 - k)], [0, k, cy * (1 - k)]], np.float32)
            x[..., ch] = cv2.warpAffine(np.ascontiguousarray(x[..., ch]), M, (w, h),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if p["vignette"] > 0:
        x *= np.clip(1.0 - p["vignette"] * _vig(h, w), 0, 1)
    if p["fade"] > 0:
        fcol = np.array(col(p["fade_color"]), np.float32)
        x = x * (1 - p["fade"]) + fcol * p["fade"]
    if p["letterbox"] > 0:
        bh = int(round(h * p["letterbox"] / 2))
        if bh > 0:
            x[:bh] = 0
            x[h - bh:] = 0
    if p["grain"] > 0:
        n = _noise(h, w)
        rng = np.random.default_rng(fc.F * 7919 + 17)
        ox, oy = rng.integers(0, 64, 2)
        g = n[oy:oy + h, ox:ox + w][..., None]
        L = np.clip(x.mean(axis=2, keepdims=True), 0, 1)
        x += g * p["grain"] * (0.35 + 0.65 * (1 - np.abs(2 * L - 1)))
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)

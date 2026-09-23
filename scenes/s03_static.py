"""Shared hand-off static (s02 -> s03).

static_frame(T, w, h, s) -> float32 (h, w, 3), pre-post: full-frame analog TV snow seen through the
picture tube at exactly s03's opening framing (scanlines, aperture grille, faceplate curvature).
Deterministic in T. Scene02 composites this over its last frames; s03 frame 0 (T=47.5) is this same
image (identical analog/tube parameters), so the cut is invisible.
Film look to use on those frames (= s03 POST at 47.5): bloom=0.42, bloom_thresh=0.74, bloom_radius=20,
vignette=0.3, grain=0.03, ca=0.
"""
import numpy as np

from vx.ease import seg, ease_in_out, ease_out
from scenes import s03_crt as crt
from scenes import s03_layout as lay

LOCK = 48.02


def signal_px(hw_px):
    """signal image width in pixels for a tube of half-width hw_px"""
    return int(np.clip(2.0 * hw_px * 1.12, 320, 2000))


def signal_h(wpx):
    return int(round(wpx * 0.75))


def tuning(T):
    """analog chain while the receiver tunes in (T <= T0: pure snow). Shared with s03 so the cut matches."""
    t0 = lay.T0
    u = seg(T, t0, LOCK, lambda x: x)
    p = dict(snow=1.0, roll=0.0, jitter=0.25, tear=0.0, tear_y=0.35, chroma=0.0)
    if T < LOCK:
        p["snow"] = 1.0 - ease_in_out(seg(T, t0 + 0.04, LOCK - 0.06, lambda x: x), 2)
        # the picture rolls twice and locks; starts at a whole period so pure snow shows no blanking bar
        p["roll"] = 2.0 * (1 - ease_out(u, 2.2))
        p["jitter"] = 0.25 + 3.5 * (1 - u)
        p["tear"] = 0.4 * (1 - u)
        p["tear_y"] = 0.2 + 0.5 * u
    else:
        p["snow"] = 0.0
    p["chroma"] = seg(T, LOCK - 0.2, LOCK + 0.25, ease_out)
    p["snow"] = max(p["snow"], 0.08 * (1 - seg(T, LOCK, LOCK + 0.5, ease_out)))
    return p


def glass_params(T, lamp_power, off=61.0):
    """room light seen in the faceplate + reflection strength"""
    amb = np.array([0.15, 0.17, 0.27], np.float32) * (1.0 - 0.75 * seg(T, off, off + 0.2, lambda x: x * x))
    lamp_rgb = np.array([1.05, 0.55, 0.42], np.float32)
    rl = amb + lamp_rgb * 0.15 * lamp_power
    return amb, tuple(float(v) for v in rl), 0.10 + 0.08 * lamp_power


def static_frame(T, w, h, s):
    T0 = lay.T0
    cx, cy, hw, hh, rot = lay.tube_geom(T0, s)
    F = int(round(T * 24))
    p = tuning(T0)
    wpx = signal_px(hw)
    hpx = signal_h(wpx)
    sig = crt.snow_signal(F, wpx, hpx)
    sig = crt.ntsc(sig, wpx / crt.SIG_W, p["chroma"])
    _, room_light, refl = glass_params(T0, 1.0)
    out = np.zeros((h, w, 3), np.float32)
    r = crt.crt_render(sig, cx, cy, hw, hh, rot, w, h, F, roll=p["roll"], jitter=p["jitter"], tear=p["tear"],
                       tear_y=p["tear_y"], room_light=room_light, refl=refl, s=s)
    if r is not None:
        x0, y0, pt = r
        reg = out[y0:y0 + pt.shape[0], x0:x0 + pt.shape[1]]
        reg[...] = pt[..., :3] + reg * (1 - pt[..., 3:4])
    return out

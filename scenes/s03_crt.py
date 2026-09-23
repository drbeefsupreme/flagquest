"""s03 CRT simulation: a broadcast signal -> the image on a curved-glass picture tube.

Physically motivated: signal-space snow (per scanline, bandwidth-limited), vertical-hold roll with the
blanking bar, per-line sync jitter / tearing, barrel geometry of the convex faceplate, gaussian beam
scanlines whose width grows with brightness, Trinitron aperture grille (with its two damper wires),
raster edge falloff, halation inside the glass, a rolling camera/refresh beat bar, reflections on the
glass, and the switch-off collapse (vertical deflection dies -> line, horizontal dies -> dot).
Every fine pattern is anti-aliased from its on-screen pitch, so it resolves when close and melts to
an even glow when far (no moire).
"""
import math

import cv2
import numpy as np

NL = 240          # visible scanlines per field
NT = 380          # aperture grille RGB triads across the raster
NC = 330          # horizontal resolution of analog snow (TV lines)
SIG_W, SIG_H = 800, 600


# ---------------------------------------------------------------- analog snow
def snow_cells(F, seed=0):
    """(NL, NC) float32 luminance of receiver noise for global frame F"""
    rng = np.random.default_rng(F * 7331 + 11 + seed * 101)
    n = rng.standard_normal((NL, NC + 8)).astype(np.float32)
    n = cv2.GaussianBlur(n, (5, 1), 0.85)[:, 4:-4]
    n /= 0.62
    v = 0.44 + 0.21 * n
    v += 0.05 * rng.standard_normal((NL, 1)).astype(np.float32)           # per-line AGC wobble
    sp = rng.random((NL, NC)) > 0.994
    v += sp * 0.45                                                           # bright impulse specks
    y = np.arange(NL, dtype=np.float32)[:, None] / NL
    ph = (F * 0.071) % 1.0
    v += 0.06 * np.sin(2 * math.pi * (y * 1.3 - ph))                        # slow rolling band
    return np.clip(v, 0.0, 1.0)


def snow_signal(F, wpx, hpx, seed=0):
    """snow resized into a (hpx, wpx, 3) signal image (rows quantised to scanlines)"""
    c = snow_cells(F, seed)
    row = cv2.resize(c, (wpx, NL), interpolation=cv2.INTER_LINEAR)
    idx = np.minimum((np.arange(hpx) * NL) // hpx, NL - 1)
    g = row[idx]
    img = np.empty((hpx, wpx, 3), np.float32)
    img[..., 0] = g * 0.97
    img[..., 1] = g * 0.99
    img[..., 2] = g * 1.03
    return img


def ntsc(sig, s_units_px, chroma=1.0):
    """bandwidth-limit the signal: luma soft, chroma bleeds horizontally; chroma<1 desaturates"""
    k = max(1, int(round(s_units_px * 0.9)))
    Y = sig[..., 0] * 0.299 + sig[..., 1] * 0.587 + sig[..., 2] * 0.114
    Ys = cv2.blur(Y, (k | 1, 1))
    ch = sig - Y[..., None]
    kc = max(1, int(round(s_units_px * 4.5)))
    ch = cv2.blur(ch, (kc | 1, 1))
    return Ys[..., None] + ch * chroma


# ---------------------------------------------------------------- the tube
_line_rng = {}


def _line_noise(F, seed=0):
    k = (F, seed)
    if k not in _line_rng:
        rng = np.random.default_rng(F * 977 + seed)
        _line_rng.clear()
        _line_rng[k] = rng.standard_normal(NL + 1).astype(np.float32)
    return _line_rng[k]


def crt_render(sig, cx, cy, hw, hh, rot, fw, fh, F, *, k=0.085, ov=0.925, roll=0.0, jitter=0.0,
               tear=0.0, tear_y=0.3, csx=1.0, csy=1.0, bright=1.0, halation=0.17, beat=0.035,
               glass=(0.050, 0.058, 0.060), refl=0.10, room_light=(0.2, 0.2, 0.25), lamp_refl=0.0, s=1.0):
    """Render the tube face. sig: (hs, ws, 3) float signal image (SIG_W x SIG_H units).
    cx, cy, hw, hh in frame pixels (centre + half size of the glass opening), rot radians.
    Returns (x0, y0, rgba) premultiplied patch, or None if off-frame."""
    c, s = math.cos(rot), math.sin(rot)
    ex = abs(hw * c) + abs(hh * s)
    ey = abs(hw * s) + abs(hh * c)
    x0, x1 = max(0, int(cx - ex - 2)), min(fw, int(cx + ex + 3))
    y0, y1 = max(0, int(cy - ey - 2)), min(fh, int(cy + ey + 3))
    if x1 <= x0 or y1 <= y0:
        return None
    hs, ws = sig.shape[:2]
    dx = (np.arange(x0, x1, dtype=np.float32) + 0.5 - cx)[None, :]
    dy = (np.arange(y0, y1, dtype=np.float32) + 0.5 - cy)[:, None]
    nx = (dx * c + dy * s) * (1.0 / hw)
    ny = (-dx * s + dy * c) * (1.0 / hh)
    ax, ay = np.abs(nx), np.abs(ny)
    # glass opening: superellipse (rounded, slightly bulged rectangle)
    f = (ax ** 5 + ay ** 5) ** 0.2
    aa = 1.3 / min(hw, hh)
    mask = np.clip((1.0 - f) / aa + 0.5, 0.0, 1.0)
    r2 = nx * nx + ny * ny
    g = (1.0 + k * r2) * ov
    u = nx * g
    v = ny * g
    # switch-off collapse (deflection dies)
    u = u / csx
    v = v / csy
    # vertical hold roll with the blanking bar
    vbi = 0.14
    if roll != 0.0:
        v = np.mod(v + 1.0 + roll * (2.0 + vbi), 2.0 + vbi) - 1.0
    # scanline structure
    l = (v * 0.5 + 0.5) * NL
    li = np.floor(l)
    frac = l - li - 0.5
    lpp = NL * ov / (2.0 * hh)                          # scanlines per pixel
    # the fine structure fades as the tube fills the frame: up close it would read as a pattern printed
    # on the (plain!) flags, so it is only present at full strength once the camera pulls back
    near = float(np.clip((hh / max(s, 1e-3) - 280.0) / 260.0, 0.0, 1.0))
    fine = 1.0 - 0.9 * near
    amt_s = float(np.clip(1.0 - (lpp - 0.2) / 0.32, 0.0, 1.0)) * fine
    # per-line horizontal sync jitter + tear
    lidx = np.clip(li, 0, NL).astype(np.int32)
    if jitter > 0 or tear > 0:
        ln = _line_noise(F)
        du = jitter * ln[lidx] * (2.0 / SIG_W)
        if tear > 0:
            ty = (v * 0.5 + 0.5)
            band = np.exp(-((ty - tear_y) / 0.09) ** 2)
            du = du + tear * band * (0.35 + 0.65 * np.sin(ty * 40.0 + F * 1.7) ** 2) * 0.22
        u = u + du
    vs = v
    if amt_s > 0:
        vc = (li + 0.5) * (2.0 / NL) - 1.0
        vs = v + (vc - v) * (amt_s * 0.85)
    mx = (u * 0.5 + 0.5) * ws - 0.5
    my = (vs * 0.5 + 0.5) * hs - 0.5
    img = cv2.remap(sig, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # raster extent (beyond it the phosphor is dark)
    eu = 0.9 / max(hw * csx, 1.0)
    ev = 0.9 / max(hh * csy, 1.0)
    rmask = np.clip((1.0 - np.abs(u)) / max(eu, 0.006), 0, 1) * np.clip((1.0 - np.abs(v)) / max(ev, 0.006), 0, 1)
    if roll != 0.0:
        rmask = rmask * (v <= 1.0)
    lum = img[..., 0] * 0.3 + img[..., 1] * 0.59 + img[..., 2] * 0.11
    # gaussian beam: brighter lines are fatter
    mod = None
    if amt_s > 0:
        sig_b = 0.19 + 0.2 * np.clip(lum, 0, 1)
        prof = np.exp(-(frac * frac) / (2 * sig_b * sig_b)) / (sig_b * 2.5066)
        mod = 1.0 + amt_s * (prof - 1.0)
    # aperture grille
    tpp = NT * ov / (2.0 * hw)
    amt_m = float(np.clip(1.0 - (tpp - 0.17) / 0.26, 0.0, 1.0)) * fine
    base = rmask * (1.0 - 0.2 * r2) * bright
    if beat > 0:
        base = base * (1.0 + beat * np.exp(-(((ny * 0.5 + 0.5) - ((F * 0.023) % 1.3 - 0.15)) / 0.12) ** 2))
    if mod is not None:
        base = base * mod
    if amt_m > 0:
        q = (u * 0.5 + 0.5) * NT
        q = q - np.floor(q)
        out = np.empty_like(img)
        for ch, cen in enumerate((1 / 6, 3 / 6, 5 / 6)):
            d = q - cen
            d = d - np.round(d)
            gch = np.exp(-(d * d) * (1.0 / (2 * 0.115 * 0.115)))
            m = 0.28 + 0.72 * gch * (1.0 / 0.2883)
            out[..., ch] = img[..., ch] * base * (1.0 + amt_m * (m - 1.0))
        # Trinitron damper wires
        for wy in (-0.34, 0.34):
            out *= (1.0 - 0.3 * amt_m * np.exp(-((ny - wy) * hh / 1.1) ** 2))[..., None]
        img = out
    else:
        img = img * base[..., None]
    if csy < 1.0:
        # energy of the collapsing raster concentrates: the line/dot burns white
        boost = min(9.0, (1.0 / max(csy, 1e-3)) ** 0.45 * (1.0 / max(csx, 1e-3)) ** 0.35)
        wt = float(np.clip(1.0 - csy * 3.0, 0, 1))
        img = img * boost + (rmask * wt * boost * 0.55)[..., None]
    # halation: light scattering inside the thick faceplate glass
    if halation > 0:
        sg = max(1.5, hw * 0.035)
        small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        hal = cv2.GaussianBlur(small, (0, 0), sg * 0.25)
        hal = cv2.resize(hal, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        img = img + hal * halation
    # the glass itself: dark grey-green, lit a little by the room, convex highlights
    rl = np.array(room_light, np.float32)
    gl = np.array(glass, np.float32) * (0.55 + 0.45 * rl / max(1e-3, rl.max()))
    edge = np.clip((1.0 - f) / 0.07, 0.0, 1.0)            # inner shadow cast by the bezel lip
    gshade = (0.75 + 0.25 * (-ny * 0.5 + 0.5)) * (0.35 + 0.65 * edge)
    col = img * (0.45 + 0.55 * edge)[..., None] + gl * gshade[..., None]
    if refl > 0:
        # soft reflections on the dome: window glow top-left, lamp streak, a broad sheen
        hl = np.exp(-(((nx + 0.52) / 0.36) ** 2 + ((ny + 0.58) / 0.22) ** 2))
        streak = np.exp(-(((nx - ny * 0.35 - 0.52) / 0.05) ** 2)) * np.clip(1 - ((ny + 0.2) / 0.75) ** 2, 0, 1)
        sheen = np.clip(1.0 - r2 * 0.6, 0, 1) * (0.5 - 0.5 * ny)
        rv = (hl * 0.55 + streak * 0.18 + sheen * 0.10) * refl
        col = col + (rv[..., None] * (0.55 + 0.45 * rl / max(1e-3, rl.max())))
    if lamp_refl > 0:
        # the floating lamp, off to the right, mirrored small and warm in the convex faceplate
        lb = np.exp(-(((nx - 0.78) / 0.11) ** 2 + ((ny + 0.05) / 0.16) ** 2))
        lh = np.exp(-(((nx - 0.78) / 0.3) ** 2 + ((ny + 0.05) / 0.4) ** 2))
        col = col + ((lb * 0.5 + lh * 0.12) * lamp_refl)[..., None] * np.array([1.0, 0.6, 0.48], np.float32)
    rgba = np.empty((y1 - y0, x1 - x0, 4), np.float32)
    rgba[..., :3] = col * mask[..., None]
    rgba[..., 3] = mask
    return x0, y0, rgba


def raster_mean(sig):
    """average emitted colour of a signal (drives the room light)"""
    small = cv2.resize(sig, (16, 12), interpolation=cv2.INTER_AREA)
    return small.reshape(-1, 3).mean(0)

"""THE PRINT SHOP — the printed look of THE FLAG GAME? (film 2).  Owner: Printer.
(STUB v0 by Main: simple but working halftone / line-screen; Printer replaces the internals and keeps
every public signature below.)

The whole world of the film is BLACK INK ON WHITE PAPER. Scenes paint their world in GRAYSCALE with
cairo (any gray value; pure black strokes stay solid ink; white stays paper) and hand the image to a
print style. FLAGS ARE NEVER PRINTED: flags are drawn on their own layer with vx.flag and laid on top
as the "spot yellow plate" with spot() — plain yellow, unscreened, never halftoned, never hatched.

Images: float32 numpy, (h, w, 3) RGB or (h, w) tone in [0, 1]; luminance is used as tone
(0 = solid ink, 1 = bare paper). Layers with coverage are premultiplied (h, w, 4) RGBA.

API
---
tract(fc, img, *, lpi=None, angle=45.0, paper=True, space="screen", origin=(0, 0), zoom=1.0) -> rgb
    Chick-tract comic print: grays become an AM halftone dot screen, blacks stay solid, whites are paper.
    lpi = dots per 1080 design px (default ~72 -> big, visible comic dots). space="page" locks the dot
    screen to a page moving under the camera: the screen origin/zoom follow `origin` (design px) and `zoom`.
engrave(fc, img, *, angle=None, spacing=None, cross=True, paper=True) -> rgb
    Gustave-Dore wood engraving: tones become parallel lines whose width grows with darkness,
    cross-hatching in deep shadows. angle: scalar radians or per-pixel (h, w) radians map (optional).
manga(fc, img, **kw) -> rgb            fine screentone look (the anime gag)
print_layer(fc, rgba, style="tract", **kw) -> rgba   print a premultiplied layer, keeping its coverage,
    so scenes can stack: paper <- back print <- spot(flags) <- front print (hands/foreground over a pole).
paper(fc, **kw) -> rgb                 the paper stock (neutral white; NEVER yellowish)
spot(dst_rgb, flag_rgba) -> rgb        lay the flag layer (premultiplied RGBA of a Canvas that contains
    ONLY vx.flag drawings) over the print. The only place yellow enters the film.
PAPER, INK                             the two print colours (RGB tuples)
"""
import math

import numpy as np

from .canvas import blur

PAPER = (0.945, 0.941, 0.925)   # neutral uncoated white (keep it neutral: yellow belongs to the Flags)
INK = (0.075, 0.071, 0.078)     # warm-free black


def _tone(img):
    if img.ndim == 2:
        return img.astype(np.float32)
    return (img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722).astype(np.float32)


def _grid(fc, space="screen", origin=(0.0, 0.0), zoom=1.0):
    h, w = fc.h, fc.w
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs /= fc.s
    ys /= fc.s
    if space == "page":
        xs = xs / zoom + origin[0]
        ys = ys / zoom + origin[1]
    return xs, ys


def _mix(ink_cov, paper_rgb):
    ink = np.array(INK, np.float32)
    return paper_rgb * (1 - ink_cov[..., None]) + ink * ink_cov[..., None]


def paper(fc, **kw):
    img = np.empty((fc.h, fc.w, 3), np.float32)
    img[:] = PAPER
    return img


def tract(fc, img, *, lpi=None, angle=45.0, paper=True, space="screen", origin=(0.0, 0.0), zoom=1.0):
    t = np.clip(_tone(img), 0, 1)
    dark = 1.0 - t
    cell = 1080.0 / (lpi or 72.0)                       # design px per halftone cell
    xs, ys = _grid(fc, space, origin, zoom)
    a = math.radians(angle)
    u = (xs * math.cos(a) + ys * math.sin(a)) / cell
    v = (-xs * math.sin(a) + ys * math.cos(a)) / cell
    du = u - np.floor(u) - 0.5
    dv = v - np.floor(v) - 0.5
    d = np.sqrt(du * du + dv * dv)                     # distance to cell centre, in cells
    r = np.sqrt(np.clip(dark, 0, 1) / math.pi)         # dot radius giving the right ink area
    aa = 0.75 / (cell * fc.s)                          # ~1 px antialias, in cells
    cov = np.clip((r - d) / aa + 0.5, 0, 1)
    cov = np.where(dark > 0.92, 1.0, np.where(dark < 0.04, 0.0, cov)).astype(np.float32)
    return _mix(cov, np.broadcast_to(np.array(PAPER, np.float32), cov.shape + (3,)).copy())


def engrave(fc, img, *, angle=None, spacing=None, cross=True, paper=True):
    t = np.clip(_tone(img), 0, 1)
    dark = 1.0 - t
    sp = spacing or 7.0                                  # design px between engraved lines
    xs, ys = _grid(fc)
    a = np.float32(angle if angle is not None and np.ndim(angle) == 0 else math.radians(30.0))
    if angle is not None and np.ndim(angle) == 2:
        a = angle.astype(np.float32)
    ph = (xs * np.cos(a) + ys * np.sin(a)) / sp
    f = np.abs(ph - np.floor(ph) - 0.5) * 2            # 0 on the line centre .. 1 between lines
    aa = 1.0 / (sp * fc.s)
    cov = np.clip((np.clip(dark, 0, 1) * 1.05 - f) / aa + 0.5, 0, 1)
    if cross:
        ph2 = (xs * np.cos(a + 1.1) + ys * np.sin(a + 1.1)) / sp
        f2 = np.abs(ph2 - np.floor(ph2) - 0.5) * 2
        c2 = np.clip(((dark - 0.55) * 2.0 - f2) / aa + 0.5, 0, 1)
        cov = np.maximum(cov, c2)
    cov = np.where(dark > 0.95, 1.0, cov).astype(np.float32)
    return _mix(cov, np.broadcast_to(np.array(PAPER, np.float32), cov.shape + (3,)).copy())


def manga(fc, img, **kw):
    kw.setdefault("lpi", 120.0)
    return tract(fc, img, **kw)


_STYLES = {"tract": tract, "engrave": engrave, "manga": manga}


def print_layer(fc, rgba, style="tract", **kw):
    a = np.clip(rgba[..., 3:4], 0, 1)
    rgb = np.where(a > 1e-4, rgba[..., :3] / np.maximum(a, 1e-4), 1.0)   # un-premultiply over paper-white
    out = _STYLES[style](fc, rgb, **kw)
    return np.concatenate([out * a, a], axis=-1).astype(np.float32)


def spot(dst_rgb, flag_rgba):
    return flag_rgba[..., :3] + dst_rgb * (1.0 - flag_rgba[..., 3:4])

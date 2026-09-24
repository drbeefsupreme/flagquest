"""t05 shot A — the white card (105.667 -> 108.75): 'BUT THERE WERE SOME THAT LISTENED...' (P13, page 25
panel 4). A hard cut out of t04's wall of FLAGS into silence: stark paper, heavy black caps inked word by word
with the voice. The wall of FLAGS from the previous page still shows through the thin tract paper (mirrored,
a ghost), fading. Then the engraving prints itself back in (the next shot renders with print_k: its lines
swell out of the white) while the title lifts off."""
import cv2
import numpy as np

from vx import ink, comic
from vx.canvas import Canvas
from vx.ease import clamp, smoothstep, ease_out

T_CUT = 105.6667
T_PRINT0 = 108.44            # the engraving starts printing back in
T_PRINT1 = 108.78


def print_k(T):
    return smoothstep(T_PRINT0, T_PRINT1, T)


def zoom_at(T):
    return 1.0 + 0.05 * ease_out(clamp((T - T_CUT) / 3.1), 2)


_GHOST = {}


def _ghost(fc):
    """the reverse page's FLAGS wall showing through the paper: mirrored, soft, very faint"""
    key = (fc.w, fc.h)
    g = _GHOST.get(key)
    if g is None:
        cv = Canvas(fc, bg=(1, 1, 1))
        ctx = cv.ctx
        ctx.save()
        ctx.translate(1920, 0)
        ctx.scale(-1, 1)
        comic.flags_flood(ctx, (-40, -30, 2000, 1140), 1.0, t=0.0, size=64, seed=3)
        ctx.restore()
        a = cv.rgb()[..., 0]
        g = cv2.GaussianBlur(1.0 - a, (0, 0), max(0.8, 2.2 * fc.s)).astype(np.float32)
        _GHOST[key] = g
    return g


def render(fc, T):
    img = np.empty((fc.h, fc.w, 3), np.float32)
    img[:] = ink.PAPER
    # the show-through ghost (paper translucency: not ink, never screened)
    k = 0.075 * (1 - smoothstep(T_CUT, T_CUT + 1.05, T))
    img *= (1.0 - k * _ghost(fc))[..., None]
    return img


def title(ctx, fc, T):
    """the lettering, with the slow push-in; lifts off (scales up + fades) while the engraving prints in"""
    z = zoom_at(T) + 0.25 * smoothstep(T_PRINT0 - 0.05, T_PRINT1, T) ** 2
    ctx.save()
    ctx.translate(960, 540)
    ctx.scale(z, z)
    ctx.translate(-960, -540)
    bb = comic.say(ctx, fc, "P13", 960, 535, kind="title", w=1760, size=122, hold=0.12,
                   alpha=1.0 - smoothstep(T_PRINT0, T_PRINT1 - 0.06, T))
    ctx.restore()
    return bb

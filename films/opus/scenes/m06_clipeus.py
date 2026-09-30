"""m06 the Commander's voice: SCHISMA ACTVALIS as an imago clipeata - a jewelled roundel with the warrior
saint's bust (NVG lenses glowing phosphor green, radio headset, the T-O orb of the world in his hand) that sets
itself tile by tile in the upper left corner for each order (K04-K07) and lip-syncs to the radio voice."""
import math

import cv2
import numpy as np

import cairo

from vx import Canvas, set_color, circle
from vx import icon
from vx import mosaic as mz
from vx.canvas import col

from scenes import m06_tess as tx


def arc_text_bottom(ctx, s, cx, cy, rad, size, colr, font="P052"):
    """letters along the BOTTOM of a circle, reading left to right, tops toward the centre."""
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    adv = [ctx.text_extents(ch).x_advance + 0.14 * size for ch in s]
    span = (sum(adv) - 0.14 * size) / rad
    a = math.pi / 2 + span / 2                      # start at the left (angle decreases to the right)
    set_color(ctx, colr)
    for ch, ad in zip(s, adv):
        e = ctx.text_extents(ch)
        am = a - (e.x_advance / 2) / rad
        ctx.save()
        ctx.translate(cx + rad * math.cos(am), cy + rad * math.sin(am))
        ctx.rotate(am - math.pi / 2)
        ctx.move_to(-e.x_advance / 2, size * 0.36)
        ctx.show_text(ch)
        ctx.restore()
        a -= ad / rad

S = mz.SMALTI
SZ = 300.0
C = SZ / 2
R_IN, R_OUT = 112.0, 146.0
POS = (178.0, 188.0)                  # screen design px of the roundel's centre
LINES = ("K04", "K05", "K06", "K07")
LENS_CORE = np.array(col("#D9FFC8"), np.float32)
LENS_GLOW = np.array(col("#43FF6E"), np.float32)


class Clipeus:
    def __init__(self, tl):
        self.tl = tl
        self.win = [(tl.line(k)["start"], tl.line(k)["end"]) for k in LINES]
        fig = icon.Figure("schism_actual")
        self.fig = fig

        def draw(ctx, layer, mouth=0.0, blink=False):
            colr = layer == "color"
            wht = (1, 1, 1)
            if layer in ("color", "gold"):
                set_color(ctx, S["lapis_d"] if colr else (0, 0, 0))
                circle(ctx, C, C, R_IN)
                ctx.fill()
            ctx.save()
            circle(ctx, C, C, R_IN - 3)
            ctx.clip()
            fig.bust(ctx, C, C + 30, 292, mouth=mouth, goggles=1.0, layer=layer, blink=blink, tile=5.0)
            ctx.restore()
            # the jewelled ring: pearls above, the name below
            if layer in ("color", "gold"):
                set_color(ctx, S["gold"] if colr else wht)
                ctx.set_line_width(R_OUT - R_IN)
                circle(ctx, C, C, (R_IN + R_OUT) / 2)
                ctx.stroke()
            if colr:
                set_color(ctx, S["ink"])
                ctx.set_line_width(3)
                circle(ctx, C, C, R_IN)
                ctx.stroke()
                circle(ctx, C, C, R_OUT)
                ctx.stroke()
                set_color(ctx, S["lapis_d"])
                ctx.set_line_width(R_OUT - R_IN - 6)
                ctx.arc(C, C, (R_IN + R_OUT) / 2, math.radians(20), math.radians(160))
                ctx.stroke()
            if layer in ("color", "fine"):
                for k in range(15):
                    a = math.radians(196 + k * 148 / 14)
                    set_color(ctx, S["marble"] if colr else wht)
                    circle(ctx, C + (R_IN + R_OUT) / 2 * math.cos(a), C + (R_IN + R_OUT) / 2 * math.sin(a), 5.5)
                    ctx.fill()
            if layer in ("color", "gold", "fine"):
                arc_text_bottom(ctx, "SCHISMA · ACTVALIS", C, C, (R_IN + R_OUT) / 2 - 1, 20,
                                S["gold_l"] if colr else wht)
        self.spr = tx.Sprite(SZ, SZ, draw, tile=5.2, fine_tile=2.8, seed=77, ppu=1.0, poses=[dict(mouth=0.0),
                                                                                          dict(mouth=1.0)],
                             anchor=(C, C))
        r = np.hypot(self.spr.xy[:, 0] - C, self.spr.xy[:, 1] - C)
        self.r = r / R_OUT
        a = fig.bust(Canvas(s=1.0, w=int(SZ), h=int(SZ)).ctx, C, C + 30, 292, mouth=0.0, goggles=1.0, layer="color")
        self.lenses = a.get("lenses", [])

    def presence(self, T):
        """0..1: the roundel setting itself (in) and lifting away (out) around each order."""
        p = 0.0
        for s, e in self.win:
            if s - 0.3 <= T <= e + 0.7:
                p = max(p, min(np.clip((T - (s - 0.3)) / 0.3, 0, 1), np.clip((e + 0.7 - T) / 0.3, 0, 1)))
        return p

    def render(self, fc, img, T, light):
        p = self.presence(T)
        if p <= 0:
            return img
        m = round(self.tl.mouth("COMMANDER", T) / 0.05) * 0.05
        talking = self.tl.speaking("COMMANDER", T, 0.05) is not None
        colr = self.spr.colours(mouth=m, blink=tx.blink(T, 30) and not talking)
        # tiles set from the centre outward (and leave the same way)
        a = np.clip((p * 1.35 - self.r) / 0.35, 0, 1).astype(np.float32) * colr["alpha"]
        sp = self.spr
        M = np.array([[1.0, 0, POS[0] - C], [0, 1.0, POS[1] - C]])
        # a dark disc behind it so the roundel reads over any panel
        x1, y1 = int(min(fc.w, (POS[0] + R_OUT + 40) * fc.s)), int(min(fc.h, (POS[1] + R_OUT + 40) * fc.s))
        yy, xx = np.ogrid[:y1, :x1]
        d = np.hypot(xx / fc.s - POS[0], yy / fc.s - POS[1])
        sh = np.clip((R_OUT + 16 - d) / 22.0, 0, 1) * 0.55 * p
        img[:y1, :x1] *= (1 - sh)[..., None]
        tx.render_set(fc, img, tx.view_matrix(960.0, 540.0, 1.0), sp.xy, sp.ang, sp.size, sp.tilt, sp.jit, sp.uid,
                      sp.colour_fn(colr), M=np.repeat(M[None], sp.n, 0), light=light, alpha=a, lmax=0, bed=1.3,
                      margin=5.0)
        # the lenses glow phosphor green (brighter while he speaks)
        if self.lenses and p > 0.6:
            k = (0.7 + 0.5 * m) * (p - 0.6) / 0.4
            for (lx, ly, lr) in self.lenses:
                x, y = (lx + POS[0] - C) * fc.s, (ly + POS[1] - C) * fc.s
                r = max(1.0, lr * fc.s)
                x0, x1 = int(max(0, x - 6 * r)), int(min(fc.w, x + 6 * r + 1))
                y0, y1 = int(max(0, y - 6 * r)), int(min(fc.h, y + 6 * r + 1))
                gy, gx = np.mgrid[y0:y1, x0:x1]
                dd = np.hypot(gx - x, gy - y) / r
                glow = np.exp(-(dd / 2.2) ** 2)[..., None] * LENS_GLOW * 0.9 + \
                    np.exp(-(dd / 0.8) ** 2)[..., None] * LENS_CORE * 0.9
                img[y0:y1, x0:x1] += glow * k
        return img


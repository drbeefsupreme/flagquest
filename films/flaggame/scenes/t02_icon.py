"""The icon: Dr. Beelzebub Crow as a Byzantine saint (ref page 23, panel 3) — Chick-tract print of an icon.

Icon(rect)                              rect = the icon's outer frame (page units)
  .draw(ctx, T, *, gleam, radiance, mouth)   the painting in grey tones; gleam 0..1 = position of the silver-leaf
                                          gleam sweep (None = none); radiance 0..1 = the 'ESSENCE' glory
  .flag_hand(ctx, T)                      the fist on the pole (print it in FRONT of the flag plate)
  .pole_base / .pole_len / .pole_ang      the flag the saint raises (for the spot-plate simulation)
Silver/grey leaf only (NOT gold), inverse perspective (the Gospel book's page blocks and the little basilica splay
outward on both sides), an engraved guilloche frame, the inscription O AGIOS KORAX ('Saint Crow'), and the tract
itself as the Gospel book.
Owner: T02Wrong."""
import math

import cairo
import numpy as np

from .t02_brush import stroke, grey, WHITE, dots, poly_fill, path_curve, clip_curve
from .t02_cast import fill_curve


def _ring_text(ctx, text, x, y, size, font="C059", spacing=1.0, weight=cairo.FONT_WEIGHT_BOLD):
    ctx.save()
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, weight)
    ctx.set_font_size(size)
    ext = ctx.text_extents(text)
    ctx.move_to(x - ext.x_advance / 2, y)
    ctx.text_path(text)
    ctx.set_source_rgb(0, 0, 0)
    ctx.fill()
    ctx.restore()


def guilloche_band(ctx, x0, y0, x1, y1, *, period=22.0, lines=5, lw=0.9, phase=0.0, vertical=False):
    """interlaced sine ribbons along a band (banknote engraving), black on the current fill"""
    ctx.save()
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.clip()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(lw)
    if not vertical:
        L, a0, a1 = x1 - x0, y0, y1
    else:
        L, a0, a1 = y1 - y0, x0, x1
    mid, amp = (a0 + a1) / 2, (a1 - a0) * 0.42
    for k in range(lines):
        ph = phase + k * 2 * math.pi / lines
        ctx.new_path()
        n = int(L / 2) + 2
        for i in range(n):
            s = i * L / (n - 1)
            v = mid + amp * math.sin(s / period * 2 * math.pi + ph) * (0.55 + 0.45 * math.cos(s / period * 0.5 + ph * 0.5))
            p = (x0 + s, v) if not vertical else (v, y0 + s)
            (ctx.move_to if i == 0 else ctx.line_to)(*p)
        ctx.stroke()
    ctx.restore()


def rosette(ctx, cx, cy, r0, r1, *, n=18, lobes=7, lw=0.6, phase=0.0, color=(0, 0, 0)):
    """guilloche rosette: n rotated epitrochoid-ish loops between r0 and r1"""
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(lw)
    for j in range(n):
        off = j * 2 * math.pi / n + phase
        ctx.new_path()
        m = 220
        for i in range(m + 1):
            a = i / m * 2 * math.pi
            r = r0 + (r1 - r0) * (0.5 + 0.5 * math.sin(lobes * a + off))
            (ctx.move_to if i == 0 else ctx.line_to)(cx + math.cos(a) * r, cy + math.sin(a) * r)
        ctx.stroke()
    ctx.restore()


class Icon:
    def __init__(self, rect):
        x, y, w, h = rect
        self.rect = rect
        self.fw = 22.0                                       # frame band width
        self.inner = (x + self.fw, y + self.fw, w - 2 * self.fw, h - 2 * self.fw)
        ix, iy, iw, ih = self.inner
        self.cx = ix + iw * 0.42
        self.s = 1.06                                        # saint scale (face drawn in local units)
        self.face = (self.cx, iy + 128)
        self.halo = (self.cx, iy + 128 - 30 * self.s, 78.0)
        # the flag: held in his left hand (viewer's right), raised up to the top right of the icon
        self.pole_base = (ix + iw * 0.885, y + h + 30)
        self.pole_len = 360.0
        self.pole_ang = 0.035
        hy = iy + ih * 0.52
        bx, by = self.pole_base
        self.hand = (bx + math.sin(self.pole_ang) * (by - hy) / math.cos(self.pole_ang), hy)

    # ---------------------------------------------------------------- pieces
    def leaf(self, ctx, T, gleam, radiance):
        ix, iy, iw, ih = self.inner
        ctx.save()
        ctx.rectangle(ix, iy, iw, ih)
        ctx.clip()
        base = 0.80 + 0.14 * radiance
        ctx.set_source_rgb(base, base, base)
        ctx.paint()
        # punched tooling: fine concentric arcs around the halo (tooled silver leaf)
        hx, hy, hr = self.halo
        ctx.set_source_rgba(0, 0, 0, 0.55 * (1 - radiance))
        ctx.set_line_width(0.5)
        for k in range(18):
            ctx.new_path()
            ctx.arc(hx, hy, hr + 18 + k * 13, 0, 2 * math.pi)
            ctx.stroke()
        if gleam is not None:
            # a diagonal specular sweep across the metal
            gx = ix - 150 + (iw + 300) * gleam
            for wdt, a in ((120, 0.25), (55, 0.55), (18, 0.9)):
                ctx.new_path()
                ctx.move_to(gx - wdt, iy + ih + 20)
                ctx.line_to(gx + wdt, iy + ih + 20)
                ctx.line_to(gx + wdt + 170, iy - 20)
                ctx.line_to(gx - wdt + 170, iy - 20)
                ctx.close_path()
                ctx.set_source_rgba(1, 1, 1, a)
                ctx.fill()
        if radiance > 0:
            # engraved glory: rays burst from behind the saint's head
            n = 72
            R = 900
            for i in range(n):
                a = i / n * 2 * math.pi + T * 0.04
                wid = math.pi / n * (0.28 + 0.12 * math.sin(i * 1.7))
                ctx.new_path()
                ctx.move_to(hx + math.cos(a) * hr * 0.9, hy + math.sin(a) * hr * 0.9)
                ctx.line_to(hx + math.cos(a - wid) * R, hy + math.sin(a - wid) * R)
                ctx.line_to(hx + math.cos(a + wid) * R, hy + math.sin(a + wid) * R)
                ctx.close_path()
                ctx.set_source_rgba(0, 0, 0, 0.85 * radiance)
                ctx.fill()
            # the glow core (white)
            g = cairo.RadialGradient(hx, hy, hr * 0.8, hx, hy, hr * (2.4 + 1.2 * radiance))
            g.add_color_stop_rgba(0, 1, 1, 1, radiance)
            g.add_color_stop_rgba(1, 1, 1, 1, 0)
            ctx.set_source(g)
            ctx.paint()
        ctx.restore()

    def inscription(self, ctx):
        ix, iy, iw, ih = self.inner
        hx, hy, hr = self.halo
        # O AGIOS | KORAX split around the halo, like IC | XC (Greek: 'Saint Crow')
        _ring_text(ctx, "Ο ΑΓΙΟΣ", ix + (hx - hr - ix) / 2, iy + 58, 19)
        _ring_text(ctx, "ΚΟΡΑΞ", ix + (hx - hr - ix) / 2, iy + 82, 19)
        stroke(ctx, [(ix + 22, iy + 66), (hx - hr - 22, iy + 66)], 1.2, taper=(0.2, 0.2))

    def architecture_bg(self, ctx):
        """a little basilica top-left, in inverse perspective (behind the saint)"""
        ix, iy, iw, ih = self.inner
        cx0, cy0 = ix + 8, iy + ih * 0.46
        w0, w1 = 50, 70           # near (front) width < far width
        roof = [(cx0 + (w1 - w0) / 2, cy0), (cx0 + (w1 - w0) / 2 + w0, cy0), (cx0 + w1 + 6, cy0 - 18), (cx0 - 6, cy0 - 18)]
        poly_fill(ctx, roof, grey(0.45))
        stroke(ctx, roof + [roof[0]], 1.3, taper=(0, 0))
        body = [(cx0 + (w1 - w0) / 2, cy0), (cx0 + (w1 - w0) / 2 + w0, cy0), (cx0 + (w1 - w0) / 2 + w0, cy0 + 70),
                (cx0 + (w1 - w0) / 2, cy0 + 70)]
        poly_fill(ctx, body, grey(0.9))
        stroke(ctx, body + [body[0]], 1.3, taper=(0, 0))
        for k in range(3):
            wx = cx0 + (w1 - w0) / 2 + 9 + k * 12
            ctx.save()
            ctx.new_path()
            ctx.arc(wx + 3, cy0 + 14, 3, math.pi, 0)
            ctx.line_to(wx + 6, cy0 + 30)
            ctx.line_to(wx, cy0 + 30)
            ctx.close_path()
            ctx.set_source_rgb(0.15, 0.15, 0.15)
            ctx.fill()
            ctx.restore()
        door = [(cx0 + w1 / 2 - 7, cy0 + 70), (cx0 + w1 / 2 - 7, cy0 + 52), (cx0 + w1 / 2 + 7, cy0 + 52), (cx0 + w1 / 2 + 7, cy0 + 70)]
        poly_fill(ctx, door, grey(0.3))
        for k in range(4):
            stroke(ctx, [(cx0 + (w1 - w0) / 2, cy0 + 38 + k * 8), (cx0 + (w1 - w0) / 2 + w0, cy0 + 38 + k * 8)], 0.6,
                   taper=(0, 0))
        ctx.save()
        ctx.new_path()
        ctx.arc(cx0 + w1 / 2, cy0 - 18, 16, math.pi, 0)
        ctx.close_path()
        ctx.set_source_rgb(0.6, 0.6, 0.6)
        ctx.fill_preserve()
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(1.3)
        ctx.stroke()
        ctx.restore()

    def saint(self, ctx, T, radiance, mouth=0.0):
        ix, iy, iw, ih = self.inner
        cx = self.cx
        hx, hy, hr = self.halo
        fx, fy = self.face
        # halo: silver leaf disc, incised rings, punched rim, guilloche rosette
        ctx.save()
        ctx.new_path()
        ctx.arc(hx, hy, hr, 0, 2 * math.pi)
        v = 0.86 + 0.1 * radiance
        ctx.set_source_rgb(v, v, v)
        ctx.fill()
        rosette(ctx, hx, hy, hr * 0.55, hr * 0.92, n=14, lobes=9, lw=0.45, phase=T * 0.08)
        for rr, lw in ((hr, 2.2), (hr * 0.93, 0.8), (hr * 0.52, 0.8)):
            ctx.new_path()
            ctx.arc(hx, hy, rr, 0, 2 * math.pi)
            ctx.set_source_rgb(0, 0, 0)
            ctx.set_line_width(lw)
            ctx.stroke()
        dots(ctx, [(hx + math.cos(a) * hr * 0.965, hy + math.sin(a) * hr * 0.965) for a in np.linspace(0, 2 * math.pi, 64, False)], 1.1)
        ctx.restore()
        s = self.s
        ctx.save()
        ctx.translate(fx, fy)
        ctx.scale(s, s)
        bot = (iy + ih + 10 - fy) / s
        cx = fx = fy = 0.0
        # body: phelonion (dark cape) with chrysography (fine highlight lines), omophorion (white stole)
        sh_y = fy + 62
        cape = [(cx - 128, bot), (cx - 118, sh_y + 30), (cx - 70, sh_y), (cx + 70, sh_y), (cx + 118, sh_y + 30),
                (cx + 128, bot)]
        poly_fill(ctx, cape, grey(0.30))
        ctx.save()
        ctx.new_path()
        ctx.move_to(*cape[0])
        for p in cape[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.clip()
        # chrysography: rays of fine white lines fanning from the shoulders (grey leaf, not gold)
        for side in (-1, 1):
            ox, oy = cx + side * 60, sh_y + 6
            for i in range(26):
                a = math.pi / 2 + side * (0.15 + i * 0.045)
                stroke(ctx, [(ox, oy), (ox + math.cos(a) * 260, oy + math.sin(a) * 260)], 0.9, taper=(0.3, 0.3),
                       color=grey(0.85))
        # tunic under the cape
        poly_fill(ctx, [(cx - 34, sh_y + 20), (cx + 34, sh_y + 20), (cx + 48, bot), (cx - 48, bot)],
                  grey(0.7))
        for i in range(7):
            xx = cx - 30 + i * 10
            stroke(ctx, [(xx, sh_y + 24), (xx + (xx - cx) * 0.3, bot)], 0.8, taper=(0.1, 0.1))
        ctx.restore()
        stroke(ctx, cape[1:5], 2.2, taper=(0, 0))
        # omophorion: a white stole with circles (no crosses, no flags: only plain geometry)
        stole = [(cx - 76, sh_y + 4), (cx + 76, sh_y + 4), (cx + 22, sh_y + 60), (cx + 22, bot),
                 (cx - 22, bot), (cx - 22, sh_y + 60)]
        poly_fill(ctx, stole, grey(0.98))
        stroke(ctx, stole + [stole[0]], 1.6, taper=(0, 0))
        for k in range(5):
            yy = sh_y + 75 + k * 32
            if yy < (bot - 10):
                ctx.save()
                ctx.new_path()
                ctx.arc(cx, yy, 9, 0, 2 * math.pi)
                ctx.set_source_rgb(0, 0, 0)
                ctx.set_line_width(1.6)
                ctx.stroke()
                ctx.restore()
                dots(ctx, [(cx, yy)], 3.0)
        for side in (-1, 1):
            dots(ctx, [(cx + side * 48, sh_y + 20), (cx + side * 30, sh_y + 38)], 3.2)
        # neck + face (icon style: long straight nose, almond eyes, highlights as fine white hatching)
        poly_fill(ctx, [(fx - 16, fy + 30), (fx + 16, fy + 30), (fx + 18, sh_y + 6), (fx - 18, sh_y + 6)], grey(0.72))
        face = [(fx, fy - 46), (fx + 28, fy - 36), (fx + 34, fy - 6), (fx + 28, fy + 26), (fx + 12, fy + 44),
                (fx, fy + 48), (fx - 12, fy + 44), (fx - 28, fy + 26), (fx - 34, fy - 6), (fx - 28, fy - 36)]
        fill_curve(ctx, face, grey(0.74))
        ctx.save()
        clip_curve(ctx, face, 8)
        for i in range(9):   # the painter's highlight 'hatching' (sankir + ochre highlights)
            yy = fy - 30 + i * 5
            stroke(ctx, [(fx - 22, yy), (fx - 8, yy - 3)], 0.8, color=WHITE, taper=(0.2, 0.2))
            stroke(ctx, [(fx + 8, yy - 3), (fx + 22, yy)], 0.8, color=WHITE, taper=(0.2, 0.2))
        ctx.restore()
        stroke(ctx, face + [face[0]], 1.8, taper=(0, 0))
        # long straight icon nose
        stroke(ctx, [(fx - 3, fy - 16), (fx - 4, fy + 8), (fx - 7, fy + 14)], 1.2, taper=(0.3, 0.2))
        stroke(ctx, [(fx - 7, fy + 14), (fx, fy + 17), (fx + 7, fy + 14)], 1.1, taper=(0.2, 0.2))
        # AVIATORS (even in heaven)
        for side in (-1, 1):
            L = [(fx + side * 4, fy - 12), (fx + side * 16, fy - 15), (fx + side * 29, fy - 12), (fx + side * 28, fy),
                 (fx + side * 20, fy + 7), (fx + side * 8, fy + 5)]
            ctx.save()
            ctx.new_path()
            path_curve(ctx, L, True, 6)
            ctx.set_source_rgb(0.28, 0.28, 0.28)
            ctx.fill_preserve()
            ctx.set_source_rgb(0, 0, 0)
            ctx.set_line_width(1.2)
            ctx.stroke()
            ctx.restore()
            stroke(ctx, [(fx + side * 10, fy + 2), (fx + side * 22, fy - 11)], 1.4, color=WHITE, taper=(0.3, 0.3))
        stroke(ctx, [(fx - 5, fy - 12), (fx, fy - 14), (fx + 5, fy - 12)], 1.0, taper=(0, 0))
        # stylised Van Dyke: moustache + pointed goatee of parallel wavy lines
        must = [(fx - 16, fy + 24), (fx, fy + 20), (fx + 16, fy + 24), (fx + 10, fy + 27), (fx, fy + 24), (fx - 10, fy + 27)]
        fill_curve(ctx, must, grey(0.5))
        goat = [(fx - 12, fy + 32), (fx + 12, fy + 32), (fx + 8, fy + 50), (fx, fy + 64), (fx - 8, fy + 50)]
        fill_curve(ctx, goat, grey(0.5))
        for k in range(5):
            xx = fx - 8 + k * 4
            stroke(ctx, [(xx, fy + 33), (xx + (fx - xx) * 0.5, fy + 50), (fx + (xx - fx) * 0.15, fy + 62)], 0.7,
                   taper=(0.2, 0.5))
        # small solemn mouth (opens a touch with the voice)
        stroke(ctx, [(fx - 7, fy + 29), (fx, fy + 29 + 4 * mouth), (fx + 7, fy + 29)], 1.3 + 2 * mouth, taper=(0.3, 0.3))
        # ears
        for side in (-1, 1):
            ear = [(fx + side * 33, fy - 10), (fx + side * 39, fy - 6), (fx + side * 38, fy + 10), (fx + side * 32, fy + 12)]
            fill_curve(ctx, ear, grey(0.74))
            stroke(ctx, ear, 1.2, taper=(0.1, 0.1))
        ctx.save()
        ctx.translate(0, -30)
        ctx.scale(1.0, 0.78)
        ctx.translate(0, 30)
        # the MITRE (tall, onion-domed, pearls, jewels — and the pins from his trucker cap)
        mt = [(fx - 40, fy - 30), (fx - 44, fy - 60), (fx - 34, fy - 96), (fx - 12, fy - 118), (fx, fy - 128),
              (fx + 12, fy - 118), (fx + 34, fy - 96), (fx + 44, fy - 60), (fx + 40, fy - 30)]
        fill_curve(ctx, mt, grey(0.88))
        ctx.save()
        clip_curve(ctx, mt, 8)
        poly_fill(ctx, [(fx - 60, fy - 44), (fx + 60, fy - 44), (fx + 60, fy - 30), (fx - 60, fy - 30)], grey(0.35))
        for k in range(-3, 4):
            stroke(ctx, [(fx + k * 12, fy - 44), (fx + k * 7, fy - 128)], 0.8, taper=(0, 0))
        ctx.restore()
        stroke(ctx, mt, 2.0, taper=(0, 0))
        dots(ctx, [(fx + k * 9, fy - 37) for k in range(-4, 5)], 2.4, color=WHITE)
        for (px, py, r, tone) in ((fx - 18, fy - 70, 7, 0.3), (fx + 16, fy - 76, 6, 0.6), (fx, fy - 98, 7, 0.2),
                                  (fx - 26, fy - 52, 5, 0.5), (fx + 25, fy - 55, 5.5, 0.35), (fx + 2, fy - 64, 5, 0.9)):
            ctx.save()
            ctx.new_path()
            ctx.arc(px, py, r, 0, 2 * math.pi)
            ctx.set_source_rgb(tone, tone, tone)
            ctx.fill_preserve()
            ctx.set_source_rgb(0, 0, 0)
            ctx.set_line_width(1.1)
            ctx.stroke()
            ctx.restore()
            dots(ctx, [(px - r * 0.3, py - r * 0.3)], r * 0.25, color=WHITE)
        # a little cross-less orb on top
        ctx.save()
        ctx.new_path()
        ctx.arc(fx, fy - 133, 5, 0, 2 * math.pi)
        ctx.set_source_rgb(0.2, 0.2, 0.2)
        ctx.fill()
        ctx.restore()
        ctx.restore()
        ctx.restore()

    def hands(self, ctx, T):
        """the book-hand (viewer's left) holding the tract as the Gospel book, in INVERSE perspective (both
        page-block edges visible, widening away from us); the flag hand is printed in front()"""
        fx, fy = self.face
        s = self.s
        ctx.save()
        ctx.translate(fx, fy)
        ctx.scale(s, s)
        bx, by = -104.0, 70.0
        w, h = 56.0, 72.0
        # page blocks on both sides, splaying outward (inverse perspective)
        lb = [(bx, by), (bx - 9, by - 5), (bx - 9, by + h + 5), (bx, by + h)]
        rb = [(bx + w, by), (bx + w + 9, by - 5), (bx + w + 9, by + h + 5), (bx + w, by + h)]
        tb = [(bx, by), (bx + w, by), (bx + w + 9, by - 5), (bx - 9, by - 5)]
        for q, v in ((lb, 0.9), (rb, 0.9), (tb, 0.95)):
            poly_fill(ctx, q, grey(v))
            stroke(ctx, q + [q[0]], 1.2, taper=(0, 0))
        for k in range(4):
            stroke(ctx, [(bx - 2 - k * 2, by + 2), (bx - 2 - k * 2, by + h - 2)], 0.5, taper=(0, 0))
            stroke(ctx, [(bx + w + 2 + k * 2, by + 2), (bx + w + 2 + k * 2, by + h - 2)], 0.5, taper=(0, 0))
        book = [(bx, by), (bx + w, by), (bx + w, by + h), (bx, by + h)]
        poly_fill(ctx, book, grey(0.98))
        stroke(ctx, book + [book[0]], 1.8, taper=(0, 0))
        dots(ctx, [(bx + 5, by + 5), (bx + w - 5, by + 5), (bx + 5, by + h - 5), (bx + w - 5, by + h - 5)], 2.2)
        ctx.save()
        ctx.translate(bx + w / 2, by + 24)
        ctx.select_font_face("Archivo Black", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        for i, (t, sz) in enumerate((("THE", 8.5), ("FLAG", 11.5), ("GAME?", 11.0))):
            ctx.set_font_size(sz)
            e = ctx.text_extents(t)
            ctx.move_to(-e.x_advance / 2, i * 13)
            ctx.set_source_rgb(0, 0, 0)
            ctx.show_text(t)
        ctx.restore()
        # veiled hand under the book (icons veil the hand holding the Gospel)
        veil = [(bx - 14, by + h - 12), (bx + w + 12, by + h - 18), (bx + w + 16, by + h + 26), (bx - 10, by + h + 34)]
        poly_fill(ctx, veil, grey(0.55))
        for k in range(6):
            stroke(ctx, [(bx - 8 + k * 13, by + h - 12), (bx - 4 + k * 13, by + h + 30)], 0.8, color=WHITE, taper=(0.2, 0.2))
        stroke(ctx, veil + [veil[0]], 1.5, taper=(0, 0))
        ctx.restore()

    def flag_hand(self, ctx, T):
        """the raised hand gripping the pole (printed in FRONT of the pole)"""
        hx, hy = self.hand
        a = self.pole_ang
        ctx.save()
        ctx.translate(hx, hy)
        ctx.rotate(a)
        ctx.scale(self.s, self.s)
        fist = [(-13, -14), (10, -16), (15, 0), (11, 16), (-12, 16), (-16, 0)]
        fill_curve(ctx, fist, grey(0.74), 6)
        stroke(ctx, fist + [fist[0]], 1.8, taper=(0, 0))
        for k in range(3):
            stroke(ctx, [(-10, -8 + k * 8), (6, -9 + k * 8)], 1.0, taper=(0.1, 0.4))
        ctx.restore()

    def sleeve(self, ctx, T):
        """the raised arm: a wide icon sleeve hanging from the forearm, shoulder -> flag hand (behind the pole)"""
        hx, hy = self.hand
        cx = self.cx
        fx, fy = self.face
        s = self.s
        sx, sy = cx + 70 * s, fy + 70 * s          # shoulder
        ex, ey = sx + 38 * s, sy + 8 * s            # elbow (low, the forearm rises to the hand)
        upper = [(sx - 18 * s, sy - 12 * s), (ex + 10 * s, ey - 16 * s), (ex + 14 * s, ey + 14 * s), (sx - 6 * s, sy + 26 * s)]
        poly_fill(ctx, upper, grey(0.30))
        stroke(ctx, upper + [upper[0]], 2.0, taper=(0, 0))
        # forearm (tunic) up to the hand + the wide hanging sleeve of the phelonion
        fore = [(ex - 10 * s, ey - 8 * s), (hx - 9 * s, hy + 6 * s), (hx + 9 * s, hy + 8 * s), (ex + 14 * s, ey + 6 * s)]
        poly_fill(ctx, fore, grey(0.72))
        for i in range(6):
            u = (i + 0.5) / 6
            px, py = ex + (hx - ex) * u, ey + (hy - ey) * u
            stroke(ctx, [(px - 8 * s, py + 1 * s), (px + 8 * s, py + 3 * s)], 0.8, taper=(0.1, 0.1))
        stroke(ctx, fore + [fore[0]], 1.8, taper=(0, 0))
        cuff_y = hy + 34 * s
        drape = [(hx - 12 * s, hy + 18 * s), (hx + 14 * s, hy + 20 * s), (hx + 26 * s, cuff_y + 50 * s),
                 (hx + 6 * s, cuff_y + 64 * s), (ex + 16 * s, ey + 30 * s), (ex, ey + 6 * s)]
        poly_fill(ctx, drape, grey(0.30))
        for i in range(9):
            u = i / 8
            px = hx - 6 * s + (ex - hx) * 0.3 * u
            stroke(ctx, [(px + 4 * s, hy + 24 * s + u * 10 * s), (px + 18 * s - u * 6 * s, cuff_y + 50 * s - u * 20 * s)],
                   0.8, color=grey(0.85), taper=(0.2, 0.2))
        stroke(ctx, drape + [drape[0]], 2.0, taper=(0, 0))

    def frame(self, ctx, T):
        x, y, w, h = self.rect
        fw = self.fw
        ctx.save()
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.rectangle(x, y, w, h)
        ctx.rectangle(*self.inner)
        ctx.set_source_rgb(0.62, 0.62, 0.62)
        ctx.fill()
        ctx.restore()
        ph = T * 0.25
        guilloche_band(ctx, x, y + 3, x + w, y + fw - 3, phase=ph)
        guilloche_band(ctx, x, y + h - fw + 3, x + w, y + h - 3, phase=-ph)
        guilloche_band(ctx, x + 3, y, x + fw - 3, y + h, phase=ph, vertical=True)
        guilloche_band(ctx, x + w - fw + 3, y, x + w - 3, y + h, phase=-ph, vertical=True)
        for (px, py) in ((x + fw / 2, y + fw / 2), (x + w - fw / 2, y + fw / 2), (x + fw / 2, y + h - fw / 2),
                         (x + w - fw / 2, y + h - fw / 2)):
            ctx.save()
            ctx.new_path()
            ctx.arc(px, py, fw * 0.48, 0, 2 * math.pi)
            ctx.set_source_rgb(0.9, 0.9, 0.9)
            ctx.fill()
            ctx.restore()
            rosette(ctx, px, py, fw * 0.12, fw * 0.45, n=8, lobes=5, lw=0.5, phase=ph)
        for r, lw in (((x, y, w, h), 2.4), ((x + 3, y + 3, w - 6, h - 6), 0.8), (self.inner, 2.0),
                      ((self.inner[0] - 3, self.inner[1] - 3, self.inner[2] + 6, self.inner[3] + 6), 0.8)):
            ctx.save()
            ctx.rectangle(*r)
            ctx.set_source_rgb(0, 0, 0)
            ctx.set_line_width(lw)
            ctx.stroke()
            ctx.restore()

    # ---------------------------------------------------------------- whole painting
    def draw(self, ctx, T, *, gleam=None, radiance=0.0, mouth=0.0):
        self.leaf(ctx, T, gleam, radiance)
        self.inscription(ctx)
        self.architecture_bg(ctx)
        self.saint(ctx, T, radiance, mouth)
        self.sleeve(ctx, T)
        self.hands(ctx, T)
        self.frame(ctx, T)

"""t07 montage styles, part 2: Lascaux, Egyptian relief, Bayeux Tapestry, ukiyo-e.  Owner: T07Raised.
(see t07_styles for the interface; all greyscale, the Flag is the only colour)"""
import math

import cairo
import cv2
import numpy as np

from vx.ease import clamp
from scenes.t07_cells import (grey, P, curve, limb, pole_geom, line_cov, cov_rgb, grey_rgb, tone_of, noise, fbm,
                              posterize, halftone_cov)
from scenes.t07_styles import Style, raise_ang, ctext, font, _lg, _rg, FONT_JP, FONT_EGY, FONT_SERIF
from scenes import t07_figure as F


def emboss(tone, kk, depth, light=(-0.7, -0.7), blur_units=4.0):
    """relief shading from a height map (tone as height)"""
    s = max(0.6, blur_units * kk)
    h = cv2.GaussianBlur(tone.astype(np.float32), (0, 0), s)
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return (gx * light[0] + gy * light[1]) * depth / max(kk, 0.05)


# =========================================================================== 5. LASCAUX
class Lascaux(Style):
    name = "lascaux"
    wind = 0.7
    bg = 0.55

    def layout(self, g, v):
        a = raise_ang(v["t"], -0.7, 0.12, 0.34, v.get("still"))
        base = (70.0, 905.0)
        ln = 520.0
        top, at = pole_geom(base[0], base[1], a, ln)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln, at=at, top=top)

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        # the rock: warm-free greys, torch pool of light
        ctx.set_source(_rg(0, 520, 0, 1100, [(0, 0.84), (0.55, 0.66), (1, 0.3)]))
        ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        # pigment: charcoal (0.1) and the grey "ochre" (0.45)
        # the great aurochs, left
        ctx.save(); ctx.translate(-470, 420); ctx.scale(1.25, 1.25)
        bull = [(-225, 0), (-220, -40), (-150, -70), (-40, -95), (40, -132), (110, -112), (170, -72), (212, -42),
                (232, 0), (206, 20), (160, 12), (122, 42), (102, 72), (96, 150), (80, 152), (70, 84), (-96, 74),
                (-126, 150), (-146, 150), (-156, 70), (-205, 42), (-238, 64), (-232, 20)]
        grey(ctx, 0.5, 0.7); curve(ctx, bull, close=True, tension=0.55); ctx.fill()
        grey(ctx, 0.1, 0.92)
        curve(ctx, [(40, -132), (110, -112), (170, -72), (212, -42), (232, 0), (206, 20), (160, 12), (120, 0),
                    (70, -40), (20, -80)], close=True, tension=0.55)
        ctx.fill()
        ctx.set_line_width(10)
        curve(ctx, bull, close=True, tension=0.55); ctx.stroke()
        ctx.set_line_width(7)
        ctx.move_to(180, -100); ctx.curve_to(170, -170, 230, -200, 280, -190)     # horns
        ctx.move_to(200, -95); ctx.curve_to(210, -150, 250, -170, 290, -160)
        ctx.stroke()
        ctx.restore()
        # little horses (right) with the spotted pattern
        for (hx, hy, s) in ((470, 700, 0.8), (640, 560, 0.62)):
            ctx.save(); ctx.translate(hx, hy); ctx.scale(s, s)
            horse = [(-150, 0), (-120, -60), (-40, -80), (60, -70), (120, -110), (170, -120), (190, -90), (150, -40),
                     (130, 20), (110, 100), (95, 100), (90, 30), (-80, 30), (-100, 100), (-115, 100), (-130, 20)]
            grey(ctx, 0.45, 0.6); curve(ctx, horse, close=True, tension=0.6); ctx.fill()
            grey(ctx, 0.1, 0.9); ctx.set_line_width(8); curve(ctx, horse, close=True, tension=0.6); ctx.stroke()
            for i in range(9):
                ctx.new_sub_path(); ctx.arc(-100 + i * 26, -40 + (i % 3) * 18, 9, 0, 2 * math.pi)
            ctx.fill()
            ctx.restore()
        # hand stencils: sprayed pigment around a hand
        for (x, y, r) in ((-150, 250, 0.9), (720, 240, 0.8), (-760, 800, 0.7)):
            ctx.set_source(_rg(x, y, 0, 110 * r, [(0, 0.12, 0.85), (0.6, 0.15, 0.6), (1, 0.15, 0.0)]))
            ctx.arc(x, y, 110 * r, 0, 2 * math.pi); ctx.fill()
            ctx.save(); ctx.translate(x, y); ctx.scale(r, r)
            grey(ctx, 0.74)
            limb(ctx, (0, 60), (0, 0), 70, 76); ctx.fill()
            for i, (a, l) in enumerate(((-0.5, 70), (-0.2, 85), (0.05, 90), (0.3, 82), (0.95, 60))):
                limb(ctx, (math.sin(a) * 20, -10), (math.sin(a) * (20 + l), -10 - math.cos(a) * l), 20, 15); ctx.fill()
            ctx.restore()
        # the hunter raising the Flag: stick figure (the Lascaux shaft-scene man, bird head)
        at = L["at"]
        hand1, hand2 = at(230), at(330)
        J = F.pose((-30, 760), 300, lean=0.05, hand_f=hand2, hand_b=hand1, foot_f=(20, 900), foot_b=(-90, 900))
        grey(ctx, 0.08, 0.95); ctx.set_line_width(11)
        for s_, e_ in (("hip", "neck"), ("shf", "elf"), ("elf", "haf"), ("shb", "elb"), ("elb", "hab"),
                       ("hpf", "knf"), ("knf", "ftf"), ("hpb", "knb"), ("knb", "ftb"), ("shf", "shb")):
            ctx.move_to(*J[s_]); ctx.line_to(*J[e_])
        ctx.stroke()
        hx, hy = J["head"]
        P(ctx, [(hx - 16, hy + 8), (hx + 6, hy - 22), (hx + 34, hy - 10), (hx + 10, hy + 4)]); ctx.fill()
        # a spear and a bison-dart sign
        ctx.set_line_width(6)
        ctx.move_to(-300, 880); ctx.line_to(-120, 640); ctx.stroke()
        for i in range(5):
            ctx.move_to(-620 + i * 30, 120); ctx.line_to(-620 + i * 30, 170)
        ctx.stroke()

    def prn(self, rgba, g, v, L):
        tone = tone_of(rgba, self.bg)
        X, Y = g.XY()
        kk = g.kk
        rock = fbm(X, Y, 140.0, 61, 4)
        fine = noise(X, Y, 4.0, 62)
        base_rock = 0.62 + 0.1 * rock
        # pigment = departure from the plain rock we painted; the rock relief shows through it
        pig = np.clip(1 - tone / np.maximum(base_rock, 0.2), 0, 1)
        bump = emboss(rock * 0.9 + fbm(X, Y, 40.0, 65, 3) * 0.25 + fine * 0.05, kk, 1.6 * min(1.0, 0.25 + kk),
                      blur_units=5.0)
        cracks = np.clip(1 - np.abs(fbm(X, Y, 220.0, 63, 3)) / 0.012, 0, 1) * 0.3
        cracks = cracks * (noise(X, Y, 300.0, 64) > 0.15)
        t = tone + bump * 0.18 - cracks + fine * 0.03
        t = t - pig * np.clip(fine * 0.5 + 0.2, 0, 1) * 0.06                  # dry-brush breakup
        return grey_rgb(np.clip(t, 0, 1))


# =========================================================================== 6. EGYPTIAN RELIEF
_GLYPHS = "𓊹𓇳𓋹𓂀𓆣𓊪𓏏𓇯𓅓𓁹𓎛𓂋𓈖𓊹𓆑𓃀𓇋𓏲𓄿𓊹𓋴𓅱𓉔𓊹"


class Egypt(Style):
    name = "egypt"
    wind = 0.8
    bg = 0.8

    def layout(self, g, v):
        a = raise_ang(v["t"], -0.4, 0.0, 0.3, v.get("still"))
        base = (40.0, 880.0)
        ln = 620.0
        top, at = pole_geom(base[0], base[1], a, ln)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln, at=at, top=top)

    def _figure(self, ctx, x, y, s, g_, hand=None, crown=True, fan=False):
        """canonical Egyptian figure striding right: profile head, frontal eye and shoulders, kilt"""
        ctx.save(); ctx.translate(x, y); ctx.scale(s, s)
        grey(ctx, g_)
        # legs (profile, striding) + feet flat
        limb(ctx, (-12, -300), (-60, -20), 34, 22); ctx.fill()
        limb(ctx, (12, -300), (60, -20), 34, 22); ctx.fill()
        P(ctx, [(-80, -26), (-26, -26), (-22, 0), (-86, 0)]); ctx.fill()
        P(ctx, [(46, -26), (104, -26), (108, 0), (44, 0)]); ctx.fill()
        # kilt (shendyt) with the front panel
        P(ctx, [(-40, -440), (40, -440), (70, -290), (-50, -290)]); ctx.fill()
        grey(ctx, min(1.0, g_ + 0.12))
        P(ctx, [(10, -430), (40, -430), (80, -300), (30, -300)]); ctx.fill()
        grey(ctx, g_)
        # torso: frontal shoulders tapering to the waist
        P(ctx, [(-70, -620), (70, -620), (40, -440), (-40, -440)]); ctx.fill()
        # neck + head in profile
        P(ctx, [(-14, -660), (14, -660), (16, -612), (-16, -612)]); ctx.fill()
        curve(ctx, [(-30, -700), (-10, -730), (20, -728), (34, -700), (40, -676), (30, -660), (0, -652), (-24, -664)],
              close=True); ctx.fill()
        # nemes headdress / or wig
        grey(ctx, max(0.0, g_ - 0.18))
        if crown:
            P(ctx, [(-44, -730), (20, -744), (36, -712), (0, -700), (-10, -640), (-56, -600), (-40, -690)]); ctx.fill()
        else:
            curve(ctx, [(-40, -720), (10, -742), (30, -712), (-4, -700), (-14, -646), (-44, -650)], close=True); ctx.fill()
        # the frontal eye (big almond) + kohl line
        grey(ctx, 0.95)
        ctx.save(); ctx.translate(18, -694); ctx.scale(1, 0.5); ctx.arc(0, 0, 9, 0, 2 * math.pi); ctx.restore(); ctx.fill()
        grey(ctx, 0.12); ctx.arc(20, -694, 3.5, 0, 2 * math.pi); ctx.fill()
        ctx.set_line_width(2.5); ctx.move_to(8, -696); ctx.line_to(36, -697); ctx.stroke()
        # arms
        grey(ctx, g_)
        if hand is not None:
            limb(ctx, (50, -610), (hand[0] / s - x / s, hand[1] / s - y / s), 26, 20); ctx.fill()
        else:
            limb(ctx, (50, -610), (70, -420), 26, 20); ctx.fill()
        limb(ctx, (-50, -610), (-70, -430), 26, 20); ctx.fill()
        if fan:
            ctx.set_line_width(6); ctx.move_to(-70, -430); ctx.line_to(-110, -760); ctx.stroke()
            curve(ctx, [(-110, -760), (-170, -820), (-110, -870), (-50, -820)], close=True); ctx.fill()
        ctx.restore()

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        grey(ctx, 0.8); ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        # registers
        grey(ctx, 0.62)
        for y in (110, 128, 905, 923):
            ctx.rectangle(-hw, y, 2 * hw, 8)
        ctx.fill()
        # a row of stars (the ceiling band) above
        for i in range(int(2 * hw / 60) + 2):
            x = -hw + i * 60
            ctext(ctx, "✶", x, 80, "DejaVu Sans", 34, g=0.62)
        # hieroglyph columns (the 𓊹 'netjer' glyph is literally a flag on a pole)
        cols = [x for x in range(int(-hw) + 40, int(hw) - 20, 70) if abs(x - 40) > 380]
        for j, x in enumerate(cols):
            grey(ctx, 0.66); ctx.rectangle(x - 34, 150, 2, 740); ctx.fill()
            for i in range(9):
                ch_ = _GLYPHS[(i * 7 + j * 3) % len(_GLYPHS)]
                ctext(ctx, ch_, x, 220 + i * 80, FONT_EGY, 62, g=0.6)
        # cartouche above the pharaoh
        grey(ctx, 0.62); ctx.set_line_width(6)
        cx, cy = -200, 190
        ctx.save(); ctx.translate(cx, cy + 90); ctx.scale(1, 2.2); ctx.arc(0, 0, 44, 0, 2 * math.pi); ctx.restore()
        ctx.stroke()
        ctx.rectangle(cx - 50, cy + 190, 100, 8); ctx.fill()
        for i, ch_ in enumerate("𓊹𓇳𓂋"):
            ctext(ctx, ch_, cx, cy + 40 + i * 58, FONT_EGY, 52, g=0.6)
        # the procession: two attendants and the pharaoh holding the Flag's pole
        at = L["at"]
        self._figure(ctx, -540, 900, 0.95, 0.66, crown=False, fan=True)
        self._figure(ctx, -360, 900, 0.95, 0.66, crown=False, fan=True)
        self._figure(ctx, -160, 900, 1.12, 0.64, hand=at(430))

    def front(self, ctx, g, v, L):
        x, y = L["at"](430)
        grey(ctx, 0.64)
        ctx.new_sub_path(); ctx.arc(x - 4, y, 17, 0, 2 * math.pi); ctx.fill()

    def prn(self, rgba, g, v, L):
        tone = tone_of(rgba, self.bg)
        X, Y = g.XY()
        kk = g.kk
        # sunk relief: carved edges catch the light, limestone grain, weathering, a ghost of old paint
        b = emboss(tone, kk, 3.2, light=(-0.6, -0.8), blur_units=2.5)
        stone = 0.8 + fbm(X, Y, 90.0, 71, 4) * 0.05 + noise(X, Y, 3.0, 72) * 0.025
        paint = np.clip((0.8 - tone) * 0.9, 0, 1)
        t = stone - paint + np.clip(b, -0.45, 0.45)
        pits = np.clip(noise(X, Y, 5.0, 73) - 0.9, 0, 1) * 1.2
        return grey_rgb(np.clip(t - pits, 0, 1))


# =========================================================================== 7. BAYEUX TAPESTRY
class Bayeux(Style):
    name = "bayeux"
    wind = 0.9
    bg = 0.88

    def layout(self, g, v):
        a = raise_ang(v["t"], -0.9, 0.1, 0.3, v.get("still"))
        base = (40.0, 820.0)
        ln = 560.0
        top, at = pole_geom(base[0], base[1], a, ln)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln, at=at, top=top)

    def _soldier(self, ctx, x, y, s, hand=None, spear=True, gm=0.45):
        J = F.pose((x, y - 0.5 * 360 * s), 360 * s, lean=0.05, hand_f=hand, foot_f=(x + 40 * s, y),
                   foot_b=(x - 40 * s, y))
        # hauberk (mail tunic) as a stiff bell shape, legs with cross-gartered stripes
        F.draw(ctx, J, gm, gm, k=0.9, head=False)
        grey(ctx, 0.3)
        hip, neck = J["hip"], J["neck"]
        P(ctx, [(neck[0] - 26 * s, neck[1]), (neck[0] + 26 * s, neck[1]), (hip[0] + 48 * s, hip[1] + 70 * s),
                (hip[0] - 48 * s, hip[1] + 70 * s)]); ctx.fill()
        hx, hy = J["head"]
        grey(ctx, 0.7); ctx.arc(hx, hy, 22 * s, 0, 2 * math.pi); ctx.fill()
        grey(ctx, 0.25)       # conical helmet with nasal
        P(ctx, [(hx - 26 * s, hy - 4 * s), (hx + 4 * s, hy - 52 * s), (hx + 28 * s, hy - 4 * s)]); ctx.fill()
        ctx.rectangle(hx + 12 * s, hy - 6 * s, 5 * s, 20 * s); ctx.fill()
        # kite shield on the back arm
        grey(ctx, 0.55)
        sx, sy = J["hab"]
        P(ctx, [(sx - 34 * s, sy - 70 * s), (sx + 34 * s, sy - 70 * s), (sx + 30 * s, sy), (sx, sy + 90 * s),
                (sx - 30 * s, sy)]); ctx.fill()
        grey(ctx, 0.25); ctx.set_line_width(5 * s)
        P(ctx, [(sx - 34 * s, sy - 70 * s), (sx + 34 * s, sy - 70 * s), (sx + 30 * s, sy), (sx, sy + 90 * s),
                (sx - 30 * s, sy)]); ctx.stroke()
        if spear:
            ctx.move_to(J["haf"][0] - 20 * s, J["haf"][1] + 140 * s); ctx.line_to(J["haf"][0] + 30 * s, J["haf"][1] - 240 * s)
            ctx.stroke()
        return J

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        grey(ctx, 0.88); ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        # borders: top and bottom bands with diagonal bars and little beasts
        grey(ctx, 0.3); ctx.set_line_width(5)
        for y in (150, 850):
            ctx.move_to(-hw, y); ctx.line_to(hw, y)
        ctx.stroke()
        for (y0, y1) in ((0, 150), (850, 1000)):
            for i in range(int(2 * hw / 150) + 2):
                x = -hw + i * 150
                grey(ctx, 0.45); ctx.set_line_width(9)
                ctx.move_to(x, y1); ctx.line_to(x + 60, y0); ctx.stroke()
                # a little beast (bird / lion) between the bars
                bx, by = x + 95, (y0 + y1) / 2
                grey(ctx, 0.5 if i % 2 else 0.35)
                if i % 2:
                    curve(ctx, [(bx - 26, by + 10), (bx - 8, by - 20), (bx + 20, by - 18), (bx + 30, by - 30),
                                (bx + 26, by - 6), (bx + 10, by + 16)], close=True); ctx.fill()
                else:
                    curve(ctx, [(bx - 34, by + 20), (bx - 30, by - 14), (bx + 10, by - 18), (bx + 30, by - 34),
                                (bx + 36, by - 10), (bx + 22, by + 20)], close=True); ctx.fill()
        # the inscription
        words = "HIC ✠ VEXILLVM ✠ ERIGITVR ✠ ET ✠ ITERVM ✠"
        ctext(ctx, words, 0, 235, FONT_SERIF, 70, g=0.3, align=0.5)
        # a stylised tree with interlace (scene divider)
        for tx in (-560, 560):
            grey(ctx, 0.5); ctx.set_line_width(12)
            ctx.move_to(tx, 850); ctx.line_to(tx, 420); ctx.stroke()
            for k in range(3):
                ctx.move_to(tx, 700 - k * 90)
                ctx.curve_to(tx - 90, 640 - k * 90, tx - 100, 560 - k * 90, tx - 20, 520 - k * 90)
                ctx.move_to(tx, 700 - k * 90)
                ctx.curve_to(tx + 90, 640 - k * 90, tx + 100, 560 - k * 90, tx + 20, 520 - k * 90)
            ctx.stroke()
        # a horse (stylised, striped legs)
        ctx.save(); ctx.translate(-330, 690); ctx.scale(1.0, 1.0)
        grey(ctx, 0.62)
        horse = [(-150, 0), (-120, -70), (-30, -90), (70, -80), (120, -140), (170, -150), (190, -120), (150, -60),
                 (140, 0), (120, 150), (100, 150), (90, 20), (-80, 20), (-90, 150), (-110, 150), (-130, 20)]
        curve(ctx, horse, close=True, tension=0.5); ctx.fill()
        grey(ctx, 0.3); ctx.set_line_width(7); curve(ctx, horse, close=True, tension=0.5); ctx.stroke()
        ctx.restore()
        # soldiers; the standard-bearer raises the pole
        at = L["at"]
        self._soldier(ctx, 250, 845, 1.0)
        self._soldier(ctx, 400, 845, 0.95, gm=0.5)
        self._soldier(ctx, -60, 845, 1.08, hand=at(300), spear=False, gm=0.4)

    def prn(self, rgba, g, v, L):
        tone = tone_of(rgba, self.bg)
        X, Y = g.XY()
        kk = g.kk
        # linen + wool embroidery: laid-and-couched fills (parallel stitches), stem-stitch outlines, weave
        lv = posterize(tone, [0.28, 0.45, 0.62, 0.88])
        fill_mask = (lv < 0.8).astype(np.float32)
        ang = 0.9 + np.floor((fbm(X, Y, 200.0, 81, 2) + 1) * 2.5) * 0.45
        lines = line_cov(np.full_like(tone, 0.45), X, Y, 6.0, ang, kk, min_px=1.8)
        stitch = noise(X * 0.7, Y * 0.7, 3.0, 82) * 0.06
        weave = (np.sin(X * 2.4) * np.sin(Y * 2.4)) * (0.03 if kk > 0.5 else 0.0)
        t = lv + fill_mask * (lines * -0.12 + stitch) + weave + fbm(X, Y, 60.0, 83, 2) * 0.03
        return grey_rgb(np.clip(t, 0, 1))


# =========================================================================== 8. UKIYO-E
class Ukiyoe(Style):
    name = "ukiyoe"
    wind = 1.2
    bg = 0.9

    def layout(self, g, v):
        a = raise_ang(v["t"], -0.8, 0.22, 0.32, v.get("still"))
        bob = 0.0 if v.get("still") else 8 * math.sin(v["T"] * 3.0)
        base = (170.0, 760.0 + bob)
        ln = 520.0
        top, at = pole_geom(base[0], base[1], a, ln)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln, at=at, top=top, bob=bob)

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        # washi + bokashi sky band
        ctx.set_source(_lg(0, Y0, 0, 600, [(0, 0.42), (0.35, 0.78), (1, 0.92)]))
        ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        # Fuji, small, in the trough
        grey(ctx, 0.52); P(ctx, [(-40, 700), (110, 560), (150, 548), (190, 560), (340, 700)]); ctx.fill()
        grey(ctx, 0.97); P(ctx, [(78, 590), (110, 560), (150, 548), (190, 560), (222, 590), (190, 580), (150, 600),
                               (115, 578)]); ctx.fill()
        # the sea behind
        grey(ctx, 0.6); ctx.rectangle(-hw, 700, 2 * hw, 400); ctx.fill()
        # THE WAVE: a curling body rising from the left, a hollow under the lip, claws of foam
        body = [(-hw - 60, 1100), (-hw - 60, 640), (-780, 450), (-640, 300), (-480, 205), (-330, 168), (-200, 182),
                (-100, 230), (-40, 300), (-44, 360), (-96, 336), (-160, 322), (-222, 356), (-256, 448), (-236, 590),
                (-130, 760), (60, 880), (300, 950), (hw + 60, 975), (hw + 60, 1100)]
        grey(ctx, 0.3); curve(ctx, body, close=True, tension=0.5); ctx.fill()
        # the lit face of the wave (lighter band) and its parallel keyblock lines
        face = [(-hw - 60, 1100), (-hw - 60, 760), (-720, 560), (-560, 420), (-420, 360), (-330, 380), (-300, 470),
                (-250, 620), (-120, 790), (80, 910), (hw + 60, 1010), (hw + 60, 1100)]
        grey(ctx, 0.58); curve(ctx, face, close=True, tension=0.5); ctx.fill()
        grey(ctx, 0.05); ctx.set_line_width(3.5)
        for k in range(1, 8):
            f = k / 8.0
            pts = [(-hw - 60, 760 - 120 * f), (-720, 560 - 110 * f), (-560, 420 - 110 * f), (-420, 360 - 120 * f),
                   (-330, 380 - 150 * f)]
            curve(ctx, pts, close=False, tension=0.5); ctx.stroke()
        ctx.set_line_width(6)
        curve(ctx, body[1:-1], close=False, tension=0.5); ctx.stroke()
        # the foam: a white fringe along the crest breaking into claws at the lip
        grey(ctx, 0.98)
        crest = [(-780, 450), (-640, 300), (-480, 205), (-330, 168), (-200, 182), (-100, 230), (-40, 300), (-44, 360)]
        ctx.set_line_width(46)
        curve(ctx, crest, close=False, tension=0.5); ctx.stroke()
        claws = []
        for i2 in range(len(crest) - 1):
            (x0, y0), (x1, y1) = crest[i2], crest[i2 + 1]
            for j2 in range(3):
                u2 = (j2 + 0.5) / 3.0
                claws.append((x0 + (x1 - x0) * u2, y0 + (y1 - y0) * u2, math.atan2(y1 - y0, x1 - x0)))
        for (x, y, a0) in claws:
            nx, ny = -math.sin(a0), math.cos(a0)          # toward the inside (down)
            tx, ty = math.cos(a0), math.sin(a0)
            for c in (-1, 0, 1):
                bx, by2 = x + tx * c * 16 + nx * 14, y + ty * c * 16 + ny * 14
                grey(ctx, 0.98)
                ctx.move_to(bx - tx * 7, by2 - ty * 7)
                ctx.curve_to(bx + nx * 30, by2 + ny * 30, bx + nx * 36 + tx * 22, by2 + ny * 36 + ty * 22,
                             bx + nx * 18 + tx * 26, by2 + ny * 18 + ty * 26)
                ctx.line_to(bx + tx * 7, by2 + ty * 7)
                ctx.close_path(); ctx.fill()
        grey(ctx, 0.05); ctx.set_line_width(3)
        for (x, y, a0) in claws[::2]:
            ctx.new_sub_path(); ctx.arc(x - math.sin(a0) * -6, y - math.cos(a0) * 6, 12, 0, 2 * math.pi); ctx.stroke()
        # spray flecks over the hollow
        grey(ctx, 0.98)
        rng = np.random.default_rng(91)
        for i2 in range(90):
            x = rng.uniform(-560, 60); y = rng.uniform(110, 330)
            ctx.new_sub_path(); ctx.arc(x, y, rng.uniform(3, 9), 0, 2 * math.pi)
        ctx.fill()
        # the little wave echoing Fuji (foreground right)
        grey(ctx, 0.3)
        curve(ctx, [(260, 1000), (380, 900), (470, 860), (540, 880), (560, 930), (620, 1000)], close=True); ctx.fill()
        grey(ctx, 0.98)
        curve(ctx, [(380, 900), (470, 860), (540, 880), (520, 895), (470, 885), (400, 910)], close=True); ctx.fill()
        # the boat, rowers bent low, one standing raising the Flag
        bob = L["bob"]
        by = 790 + bob
        grey(ctx, 0.82)
        P(ctx, [(-40, by - 20), (420, by - 50), (440, by - 36), (380, by + 10), (-10, by + 20)]); ctx.fill()
        grey(ctx, 0.05); ctx.set_line_width(5)
        P(ctx, [(-40, by - 20), (420, by - 50), (440, by - 36), (380, by + 10), (-10, by + 20)]); ctx.stroke()
        for i in range(5):
            rx = 40 + i * 70
            grey(ctx, 0.2)
            ctx.new_sub_path(); ctx.arc(rx, by - 50 - i * 2, 16, 0, 2 * math.pi); ctx.fill()
            grey(ctx, 0.85)
            curve(ctx, [(rx - 26, by - 30), (rx - 10, by - 60), (rx + 26, by - 44), (rx + 20, by - 24)], close=True)
            ctx.fill()
        at = L["at"]
        J = F.pose((150, by - 170), 250, lean=0.0, hand_f=at(260), hand_b=at(190), foot_f=(175, by - 34),
                   foot_b=(125, by - 34))
        F.draw(ctx, J, 0.2, 0.85, k=1.0)
        hx, hy = J["head"]
        grey(ctx, 0.05); ctx.new_sub_path(); ctx.arc(hx - 2, hy - 8, 15, math.pi, 2 * math.pi); ctx.fill()
        # cartouche (top-left) and the signature (right)
        grey(ctx, 0.92); ctx.rectangle(-hw + 40, 40, 90, 420); ctx.fill()
        grey(ctx, 0.05); ctx.set_line_width(4); ctx.rectangle(-hw + 40, 40, 90, 420); ctx.stroke()
        for i, ch_ in enumerate("旗揚げ百景"):
            ctext(ctx, ch_, -hw + 85, 110 + i * 78, FONT_JP, 64, g=0.05)
        for i, ch_ in enumerate("旗斎筆"):
            ctext(ctx, ch_, hw - 70, 180 + i * 56, FONT_JP, 46, g=0.1)
        grey(ctx, 0.3); ctx.rectangle(hw - 96, 360, 52, 52); ctx.fill()

    def prn(self, rgba, g, v, L):
        tone = tone_of(rgba, self.bg)
        X, Y = g.XY()
        kk = g.kk
        # woodblock: flat blocks, wood grain in the flats, soft misregistration of the grey block, keyblock black
        key = tone < 0.12
        flats = posterize(tone, [0.38, 0.6, 0.78, 0.92, 0.98])
        grain = noise(X * 0.08, Y * 1.0, 3.0, 101) * 0.035 + noise(X * 0.2, Y * 2.2, 2.0, 102) * 0.02
        sh = int(round(2.5 * kk))
        if sh > 0:
            flats = np.roll(flats, (sh, -sh), axis=(0, 1))
        # keep the bokashi gradient in the sky (not posterised)
        sky = np.clip((520 - Y) / 200.0, 0, 1)
        flats = flats * (1 - sky) + tone * sky
        t = np.where(key, 0.06, flats + grain * (tone > 0.12))
        paper = fbm(X, Y, 18.0, 103, 2) * 0.015
        return grey_rgb(np.clip(t + paper, 0, 1))


STYLES2 = [Lascaux, Egypt, Bayeux, Ukiyoe]

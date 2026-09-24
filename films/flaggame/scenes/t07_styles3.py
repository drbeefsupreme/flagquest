"""t07 montage styles, part 3: 8-bit FlagHack, medieval woodcut, Seurat pointillism.  Owner: T07Raised."""
import math

import cairo
import cv2
import numpy as np

from vx.ease import clamp, hash01
from scenes.t07_cells import (grey, P, curve, limb, pole_geom, line_cov, cov_rgb, grey_rgb, tone_of, noise, fbm,
                              posterize, surf_rgba)
from scenes.t07_styles import Style, raise_ang, ctext, font, _lg, _rg, FONT_SERIF
from scenes import t07_figure as F

FONT_PIX = "BigBlueTerm437 Nerd Font"

# a 16x16 hero sprite: 0 transparent, 1 black, 2 dark, 3 light, 4 white
_SPRITE = [
    "......1111......",
    ".....122221.....",
    "....12222221....",
    "....11111111....",
    "....13331331....",
    "...133313331....",
    "...133333331....",
    "....1333331.....",
    "...11222211.....",
    "..1322222231....",
    "..1332222331....",
    "..1111111111....",
    "...12211221.....",
    "...12211221.....",
    "..1111..1111....",
    ".11111..11111...",
]
_SPR_G = {"1": 0.05, "2": 0.38, "3": 0.72, "4": 0.97}


# =========================================================================== 9. FLAGHACK (the Flagazine's NetHack)
_MAP = [
    "                                                          ",
    "   -----------                 ------------------         ",
    "   |.........|                 |................|         ",
    "   |....$....+#######          |.......F........|         ",
    "   |.........|      #          |................|         ",
    "   |...d.....|      ###########+.......@........|         ",
    "   |.........|                 |................|         ",
    "   ----.------                 |.....>....)....|          ",
    "       #                       ------------------         ",
    "       ##########                                         ",
    "                                                          ",
]
_MSGS = ["You raise the Flag.  You feel glorious!", "You move the Flag.  The rain stops.",
         "The Flag is raised again!  You feel a strange sense of deja vu.",
         "You find a Flag.  It is plain yellow.  There are no markings upon it."]


class FlagHack(Style):
    """the Flagazine's own roguelike (HACK THE FLAGHACK): '@' plants a plain yellow Flag in the dungeon"""
    name = "flaghack"
    wind = 0.8
    bg = 0.05
    TW, TH = 29.0, 52.0      # terminal cell in composition units

    def _tile(self, c, r):
        return (-29.0 * 29 + c * self.TW, 200.0 + r * self.TH)

    def layout(self, g, v):
        a = raise_ang(v["t"], -1.3, 0.04, 0.3, v.get("still"))
        r, c = next((ri, row.index("@")) for ri, row in enumerate(_MAP) if "@" in row)
        x, y = self._tile(c + 1.2, r)
        base = (x, y + self.TH * 0.2)
        ln = 380.0
        top, at = pole_geom(base[0], base[1], a, ln)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln, top=top)

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        grey(ctx, 0.05); ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        font(ctx, FONT_PIX, 44)
        k = int(v.get("k", 1))
        for r, row in enumerate(_MAP):
            for c, ch in enumerate(row):
                if ch == " ":
                    continue
                x, y = self._tile(c, r)
                gv = 0.97 if ch in "@" else (0.8 if ch in "-|+" else (0.55 if ch in ".#" else 0.9))
                grey(ctx, gv)
                ctx.move_to(x, y + self.TH * 0.72)
                ctx.show_text(ch)
        ctx.new_path()
        # the message line and the status lines (tty green is NOT allowed: grey phosphor)
        msg = _MSGS[k % len(_MSGS)]
        x0 = max(g.vis()[0], -870.0) + 30
        ctext(ctx, msg, x0, 214, FONT_PIX, 36, g=0.97, align=0.0)
        ctext(ctx, "Crow the Flag-Bearer  St:18 Dx:14 Co:16 In:9 Wi:18 Ch:69 Lawful", x0, 860, FONT_PIX, 30,
              g=0.85, align=0.0)
        ctext(ctx, f"Dlvl:1 $:0 HP:69(69) Pw:7(7) AC:4 Xp:1/{k} T:{k * 100}", x0, 910, FONT_PIX, 30, g=0.85,
              align=0.0)

    def prn(self, rgba, g, v, L):
        tone = tone_of(rgba, self.bg)
        X, Y = g.XY()
        kk = g.kk
        # a CRT terminal: soft phosphor bloom, faint scanlines
        if kk > 0.3:
            b = cv2.GaussianBlur(tone.astype(np.float32), (0, 0), 3.0 * kk)
            tone = np.maximum(tone, b * 0.9)
            tone = tone * (0.86 + 0.14 * (0.5 + 0.5 * np.cos(Y * (2 * math.pi / 6.0))))
        return grey_rgb(np.clip(tone, 0, 1))


# =========================================================================== 10. MEDIEVAL WOODCUT
class Woodcut(Style):
    name = "woodcut"
    wind = 0.9
    bg = 1.0

    def layout(self, g, v):
        a = raise_ang(v["t"], -0.6, 0.06, 0.32, v.get("still"))
        base = (-40.0, 860.0)
        ln = 560.0
        top, at = pole_geom(base[0], base[1], a, ln)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln, at=at, top=top)

    def _c(self, ctx, tone, ang):
        """encode tone (R, B) + hatch angle (G) for the woodcut print"""
        ctx.set_source_rgb(tone, (ang % math.pi) / math.pi, tone)

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        c = self._c
        c(ctx, 1.0, 0.0); ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        # sky: light horizontal hatching near the horizon
        ctx.set_source(cairo.LinearGradient(0, 300, 0, 620))
        c(ctx, 0.9, math.pi / 2); ctx.rectangle(-hw, 380, 2 * hw, 260); ctx.fill()
        # the sun with a face, wavy rays (top right)
        sx, sy = 560, 190
        for i in range(16):
            a0 = i * math.pi / 8
            c(ctx, 0.0, 0)
            ctx.set_line_width(7)
            ctx.move_to(sx + 90 * math.cos(a0), sy + 90 * math.sin(a0))
            for k in range(1, 5):
                r = 90 + k * 18
                off = (0.12 if k % 2 else -0.12)
                ctx.line_to(sx + r * math.cos(a0 + off), sy + r * math.sin(a0 + off))
            ctx.stroke()
        c(ctx, 1.0, 0); ctx.arc(sx, sy, 80, 0, 2 * math.pi); ctx.fill()
        c(ctx, 0.0, 0); ctx.set_line_width(8); ctx.arc(sx, sy, 80, 0, 2 * math.pi); ctx.stroke()
        ctx.set_line_width(6)
        for ex in (-26, 26):
            ctx.arc(sx + ex, sy - 14, 10, math.pi, 2 * math.pi); ctx.stroke()
        ctx.arc(sx, sy + 10, 36, 0.3, math.pi - 0.3); ctx.stroke()
        ctx.move_to(sx, sy - 10); ctx.line_to(sx - 8, sy + 14); ctx.line_to(sx + 4, sy + 14); ctx.stroke()
        # curly clouds
        for (x, y) in ((-460, 250), (180, 130)):
            c(ctx, 0.0, 0); ctx.set_line_width(7)
            for k in range(4):
                ctx.new_sub_path(); ctx.arc(x + k * 55, y, 34, math.pi, 2 * math.pi * 0.98)
            ctx.stroke()
        # the walled town on the far hill
        c(ctx, 0.62, 0.35); curve(ctx, [(-hw, 640), (-500, 560), (-250, 540), (0, 600), (300, 640), (hw, 600),
                                         (hw, 1000), (-hw, 1000)], close=True); ctx.fill()
        for (x, w, h) in ((-560, 60, 170), (-470, 110, 110), (-380, 50, 200), (-320, 130, 120), (-200, 70, 160)):
            c(ctx, 1.0, 0); ctx.rectangle(x, 560 - h, w, h); ctx.fill()
            c(ctx, 0.55, 1.2); ctx.rectangle(x + w * 0.55, 560 - h, w * 0.45, h); ctx.fill()
            c(ctx, 0.0, 0); ctx.set_line_width(6); ctx.rectangle(x, 560 - h, w, h); ctx.stroke()
            P(ctx, [(x - 8, 560 - h), (x + w / 2, 560 - h - w * 0.9), (x + w + 8, 560 - h)]); ctx.fill()
            for k in range(2):
                ctx.rectangle(x + w * 0.3, 560 - h + 30 + k * 50, 10, 22)
            ctx.fill()
        # the near hill (the pilgrim's hillock), strongly hatched
        c(ctx, 0.5, -0.3)
        curve(ctx, [(-hw, 900), (-400, 850), (-150, 820), (100, 860), (400, 900), (hw, 880), (hw, 1100), (-hw, 1100)],
              close=True); ctx.fill()
        c(ctx, 0.0, 0); ctx.set_line_width(8)
        curve(ctx, [(-hw, 900), (-400, 850), (-150, 820), (100, 860), (400, 900), (hw, 880)], close=False)
        ctx.stroke()
        # tufts / stones
        ctx.set_line_width(5)
        for i in range(14):
            x = -hw + 60 + i * 130
            y = 900 + (i % 3) * 25
            ctx.move_to(x, y); ctx.line_to(x - 12, y - 30); ctx.move_to(x, y); ctx.line_to(x + 12, y - 34)
        ctx.stroke()
        # stylised trees
        for tx in (-700, 400, 700):
            c(ctx, 0.0, 0); ctx.set_line_width(10); ctx.move_to(tx, 880); ctx.line_to(tx, 700); ctx.stroke()
            c(ctx, 0.45, 0.9); ctx.arc(tx, 640, 80, 0, 2 * math.pi); ctx.fill()
            c(ctx, 0.0, 0); ctx.set_line_width(7); ctx.arc(tx, 640, 80, 0, 2 * math.pi); ctx.stroke()
        # the pilgrim: hooded robe, planting the pole
        at = L["at"]
        J = F.pose((-150, 660), 380, lean=0.2, hand_f=at(330), hand_b=at(240), foot_f=(-90, 860), foot_b=(-230, 860))
        robe = [(J["neck"][0] - 50, J["neck"][1] - 10), (J["neck"][0] + 40, J["neck"][1]), (J["hip"][0] + 60, J["hip"][1]),
                (J["ftf"][0] + 30, 860), (J["ftb"][0] - 50, 860), (J["hip"][0] - 70, J["hip"][1] + 20)]
        c(ctx, 0.62, 1.1); curve(ctx, robe, close=True, tension=0.5); ctx.fill()
        c(ctx, 0.0, 0); ctx.set_line_width(8); curve(ctx, robe, close=True, tension=0.5); ctx.stroke()
        # fold lines
        ctx.set_line_width(5)
        for k in range(3):
            ctx.move_to(J["hip"][0] - 30 + k * 30, J["hip"][1] + 20); ctx.line_to(J["hip"][0] - 60 + k * 40, 850)
        ctx.stroke()
        # arms (sleeves) to the pole
        for s_, e_, h_ in (("shb", "elb", "hab"), ("shf", "elf", "haf")):
            c(ctx, 0.75, 1.1); limb(ctx, J[s_], J[e_], 60, 50); ctx.fill(); limb(ctx, J[e_], J[h_], 50, 36); ctx.fill()
            c(ctx, 0.0, 0); ctx.set_line_width(6)
            limb(ctx, J[s_], J[e_], 60, 50); ctx.stroke(); limb(ctx, J[e_], J[h_], 50, 36); ctx.stroke()
        # hooded head
        hx, hy = J["head"]
        c(ctx, 0.35, 0.6); curve(ctx, [(hx - 40, hy + 30), (hx - 36, hy - 30), (hx, hy - 50), (hx + 36, hy - 24),
                                       (hx + 34, hy + 34)], close=True); ctx.fill()
        c(ctx, 1.0, 0); ctx.arc(hx + 12, hy + 4, 20, 0, 2 * math.pi); ctx.fill()
        c(ctx, 0.0, 0); ctx.set_line_width(6)
        curve(ctx, [(hx - 40, hy + 30), (hx - 36, hy - 30), (hx, hy - 50), (hx + 36, hy - 24), (hx + 34, hy + 34)],
              close=True); ctx.stroke()
        ctx.set_line_width(4); ctx.move_to(hx + 20, hy); ctx.line_to(hx + 30, hy + 2); ctx.stroke()
        # the frame line
        x0, y0, x1, y1 = g.vis()
        c(ctx, 0.0, 0); ctx.set_line_width(16)
        ctx.rectangle(x0 + 22, y0 + 22, x1 - x0 - 44, y1 - y0 - 44); ctx.stroke()

    def prn(self, rgba, g, v, L):
        a = rgba[..., 3]
        R = rgba[..., 0] + (1 - a)
        Gc = rgba[..., 1] + (1 - a) * 0
        X, Y = g.XY()
        kk = g.kk
        ang = Gc * math.pi
        dark = np.clip(1 - R, 0, 1)
        cov = line_cov(dark * 1.1, X, Y, 11.0, ang, kk, min_px=2.2, gamma=0.8)
        cov = np.where(R < 0.1, 1.0, cov)
        # the block's texture: ink skips on the grain, a gouge nick here and there
        skip = np.clip(noise(X * 0.15, Y * 1.2, 4.0, 111) - 0.55, 0, 1) * 1.5
        cov = np.clip(cov - skip * (cov < 0.999), 0, 1)
        return cov_rgb(cov)


# =========================================================================== 11. SEURAT
class Seurat(Style):
    name = "seurat"
    wind = 0.6
    bg = 0.85

    def layout(self, g, v):
        a = raise_ang(v["t"], -0.5, 0.03, 0.34, v.get("still"))
        base = (70.0, 840.0)
        ln = 560.0
        top, at = pole_geom(base[0], base[1], a, ln)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln, at=at, top=top)

    def _lady(self, ctx, x, y, s, gv, parasol=True, hand=None):
        """Grande-Jatte lady in profile (facing left if s < 0): bustle, tiny hat, parasol"""
        ctx.save(); ctx.translate(x, y); ctx.scale(s, abs(s))
        grey(ctx, gv)
        curve(ctx, [(-40, 0), (-44, -150), (-30, -300), (-20, -380), (10, -390), (24, -320), (60, -280), (96, -240),
                    (80, -150), (60, 0)], close=True, tension=0.5)
        ctx.fill()
        curve(ctx, [(-14, -380), (-18, -440), (0, -470), (20, -460), (22, -410), (10, -380)], close=True); ctx.fill()
        ctx.new_sub_path(); ctx.arc(0, -490, 30, 0, 2 * math.pi); ctx.fill()
        P(ctx, [(-24, -512), (26, -512), (18, -540), (-14, -540)]); ctx.fill()
        if hand is None:
            limb(ctx, (-10, -420), (-60, -330), 20, 16); ctx.fill()
        if parasol:
            ctx.set_line_width(5); ctx.move_to(-60, -330); ctx.line_to(-80, -560); ctx.stroke()
            ctx.save(); ctx.translate(-80, -560); ctx.scale(1, 0.45)
            ctx.arc(0, 0, 110, math.pi, 2 * math.pi); ctx.close_path(); ctx.restore(); ctx.fill()
        ctx.restore()

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        # sky + the river (light), far bank, sails
        grey(ctx, 0.97); ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        grey(ctx, 0.9); ctx.rectangle(-hw, 330, 2 * hw, 120); ctx.fill()
        grey(ctx, 0.72); ctx.rectangle(-hw, 300, 2 * hw, 40); ctx.fill()
        for (x, h) in ((-560, 110), (-300, 80), (380, 95)):
            grey(ctx, 0.97); P(ctx, [(x, 380), (x, 380 - h), (x + h * 0.5, 380)]); ctx.fill()
        # the lawn: sunlit and shadowed bands
        grey(ctx, 0.86); ctx.rectangle(-hw, 440, 2 * hw, 700); ctx.fill()
        grey(ctx, 0.5)
        curve(ctx, [(-hw, 600), (-300, 640), (200, 700), (hw, 660), (hw, 1100), (-hw, 1100)], close=True); ctx.fill()
        grey(ctx, 0.36)
        curve(ctx, [(-hw, 860), (-200, 900), (300, 930), (hw, 900), (hw, 1100), (-hw, 1100)], close=True); ctx.fill()
        # tree trunks + foliage mass at the top
        for tx, w in ((-760, 44), (-470, 30), (620, 50), (790, 36)):
            grey(ctx, 0.25); ctx.rectangle(tx - w / 2, -100, w, 800); ctx.fill()
        grey(ctx, 0.4)
        for (x, y, r) in ((-700, 40, 220), (-420, -20, 180), (700, 30, 240), (-180, -120, 200)):
            ctx.new_sub_path(); ctx.arc(x, y, r, 0, 2 * math.pi)
        ctx.fill()
        # strollers: top-hat gentleman, the parasol lady, a sitting pair; the lady with the Flag
        grey(ctx, 0.15)
        J = F.pose((-330, 590), 420, facing=1)
        F.draw(ctx, J, 0.12, 0.15)
        hx, hy = J["head"]
        ctx.rectangle(hx - 22, hy - 80, 44, 60); ctx.rectangle(hx - 34, hy - 26, 68, 10); ctx.fill()
        ctx.set_line_width(5); ctx.move_to(J["haf"][0], J["haf"][1]); ctx.line_to(J["haf"][0] + 40, 800); ctx.stroke()
        self._lady(ctx, -180, 800, 0.95, 0.2)
        self._lady(ctx, 500, 700, -0.6, 0.35)
        grey(ctx, 0.35)
        ctx.new_sub_path(); ctx.arc(330, 640, 18, 0, 2 * math.pi); ctx.fill()
        curve(ctx, [(300, 700), (320, 660), (350, 660), (380, 720), (300, 720)], close=True); ctx.fill()
        # the flag lady (centre), holding the pole
        at = L["at"]
        self._lady(ctx, -10, 860, 1.05, 0.18, parasol=False, hand=at(300))
        grey(ctx, 0.18)
        limb(ctx, (0, 860 - 420 * 1.05), at(300), 22, 16); ctx.fill()
        # a monkey on a leash (Seurat's)
        grey(ctx, 0.2)
        curve(ctx, [(-120, 850), (-110, 820), (-80, 815), (-60, 840), (-70, 856)], close=True); ctx.fill()
        ctx.set_line_width(4); ctx.move_to(-80, 830); ctx.curve_to(-100, 800, -140, 760, -150, 700); ctx.stroke()

    def prn(self, rgba, g, v, L):
        tone = tone_of(rgba, self.bg)
        X, Y = g.XY()
        kk = g.kk
        cell = max(15.0, 3.6 / max(kk, 1e-3))
        t_s = cv2.GaussianBlur(tone.astype(np.float32), (0, 0), max(0.5, cell * kk * 0.35))
        out = np.full(tone.shape, 0.96, np.float32)
        # divisionism: three offset layers of round dabs; each dab = the local tone pushed to a light or a
        # dark neighbour (optical mixing), 6 paint greys
        for layer, (ox, oy, rr) in enumerate(((0.0, 0.0, 0.78), (0.5, 0.5, 0.7), (0.25, 0.75, 0.55))):
            u = X / cell + ox
            w = Y / cell + oy
            iu, iw = np.floor(u), np.floor(w)
            h1 = np.sin(iu * 12.9898 + iw * 78.233 + layer * 3.1) * 43758.5453
            h1 = h1 - np.floor(h1)
            h2 = np.sin(iu * 39.346 + iw * 11.135 + layer * 1.7) * 24634.6345
            h2 = h2 - np.floor(h2)
            du = u - iu - 0.5 - (h1 - 0.5) * 0.3
            dv = w - iw - 0.5 - (h2 - 0.5) * 0.3
            d = np.sqrt(du * du + dv * dv)
            gv = np.clip(t_s + (h1 - 0.5) * 0.22, 0.04, 0.98)
            gv = np.round(gv * 5) / 5.0 * 0.94 + 0.04
            aa = 0.9 / max(cell * kk, 1e-3)
            m = np.clip((rr * 0.5 - d) / aa + 0.5, 0, 1)
            out = out * (1 - m) + gv * m
        return grey_rgb(out)


STYLES3 = [FlagHack, Woodcut, Seurat]

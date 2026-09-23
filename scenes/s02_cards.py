"""s02 content-card atlas: the 'slop' is made of these. Pre-rendered once (setup) into mip-mapped,
pre-shaded cairo sprites so thousands of them can be blitted per frame.

Designs are drawn in a 100 x 125 unit box (portrait card). NO YELLOW anywhere (thumbs are blue,
emoji are flesh-pink, badges red): yellow belongs to Flags only.
"""
import math

import cairo
import cv2
import numpy as np

from vx.canvas import col, set_color, rrect, circle, ellipse, poly, smooth, text, lin_grad, rad_grad, shade, mix
from vx.config import FONT_SANS

CW, CH = 100.0, 125.0
MIP_H = (192, 96, 48, 24, 12)        # sprite heights in pixels
SHADES = 4                          # 0 = lit ... 3 = deep shadow
SHADOW = col("#3A3350")             # deep mauve-violet the curl falls into

PINK = "#F4B6C8"
FLESH = "#E8C8D0"
CYAN = "#B8E0E8"
MAUVE = "#8A7F8F"
BEIGE = "#D9D0BC"
WHITE = "#FBF8F6"
RED = "#FF3B4E"
BLUE = "#3D7BF2"
MAG = "#FF2E88"
TEAL = "#2FC6D6"
INK = "#2A2530"
SKIN = "#F1C7B4"


# ------------------------------------------------------------------ primitives
def _card(ctx, bg, r=10, border=True):
    rrect(ctx, 2, 2, CW - 4, CH - 4, r)
    set_color(ctx, bg)
    ctx.fill_preserve()
    if border:
        set_color(ctx, shade(bg, 0.72))
        ctx.set_line_width(2.2)
        ctx.stroke()
    else:
        ctx.new_path()


def _lines(ctx, x, y, w, n, gap=9, c="#C9C3CC", lw=5):
    ctx.set_line_width(lw)
    set_color(ctx, c)
    for i in range(n):
        ww = w * (0.55 + 0.45 * ((i * 37 % 10) / 10.0)) if i == n - 1 else w * (0.85 + 0.15 * ((i * 53 % 7) / 7))
        ctx.move_to(x, y + i * gap)
        ctx.line_to(x + ww, y + i * gap)
        ctx.stroke()


def _heart(ctx, x, y, r):
    ctx.move_to(x, y + r * 0.95)
    ctx.curve_to(x - r * 1.6, y - r * 0.1, x - r * 0.9, y - r * 1.25, x, y - r * 0.45)
    ctx.curve_to(x + r * 0.9, y - r * 1.25, x + r * 1.6, y - r * 0.1, x, y + r * 0.95)
    ctx.close_path()


def _uncanny_face(ctx, cx, cy, r, skin, rnd, teeth=True, pupils=0.14):
    # head
    set_color(ctx, shade(skin, 0.8))
    circle(ctx, cx, cy + r * 0.04, r)
    ctx.fill()
    ctx.set_source(rad_grad(cx - r * 0.3, cy - r * 0.35, r * 0.1, r * 1.2, [(0, shade(skin, 1.25)), (1, skin)]))
    circle(ctx, cx, cy, r * 0.97)
    ctx.fill()
    # eyes (slightly mismatched: uncanny)
    for sx, dy, sc in ((-0.36, 0.0, 1.0), (0.36, -0.035 * rnd.uniform(0.5, 1.5), 1.12)):
        set_color(ctx, WHITE)
        ellipse(ctx, cx + sx * r, cy - r * 0.2 + dy * r, r * 0.21 * sc, r * 0.25 * sc)
        ctx.fill()
        set_color(ctx, INK)
        circle(ctx, cx + sx * r + r * 0.03, cy - r * 0.2 + dy * r, r * pupils * sc)
        ctx.fill()
        set_color(ctx, WHITE)
        circle(ctx, cx + sx * r - r * 0.02, cy - r * 0.26 + dy * r, r * 0.04)
        ctx.fill()
    # enormous grin
    ctx.move_to(cx - r * 0.66, cy + r * 0.18)
    ctx.curve_to(cx - r * 0.4, cy + r * 0.8, cx + r * 0.4, cy + r * 0.8, cx + r * 0.66, cy + r * 0.18)
    ctx.curve_to(cx + r * 0.3, cy + r * 0.34, cx - r * 0.3, cy + r * 0.34, cx - r * 0.66, cy + r * 0.18)
    ctx.close_path()
    set_color(ctx, "#5A1F2E")
    ctx.fill_preserve()
    if teeth:
        ctx.save()
        ctx.clip()
        set_color(ctx, WHITE)
        n = 11
        for i in range(n):
            x0 = cx - r * 0.66 + i * (r * 1.32 / n)
            rrect(ctx, x0 + r * 0.01, cy + r * 0.12, r * 1.32 / n - r * 0.025, r * 0.26, r * 0.03)
            ctx.fill()
            rrect(ctx, x0 + r * 0.01, cy + r * 0.52, r * 1.32 / n - r * 0.025, r * 0.3, r * 0.03)
            ctx.fill()
        ctx.restore()
    else:
        ctx.new_path()
    # blush
    set_color(ctx, "#FF7FA6", 0.35)
    ellipse(ctx, cx - r * 0.62, cy + r * 0.12, r * 0.16, r * 0.09)
    ctx.fill()
    ellipse(ctx, cx + r * 0.62, cy + r * 0.12, r * 0.16, r * 0.09)
    ctx.fill()


def _hand6(ctx, cx, cy, r, skin):
    """a waving palm with SIX fingers"""
    set_color(ctx, shade(skin, 0.75))
    ctx.set_line_width(r * 0.05)
    ang = [-1.05, -0.6, -0.22, 0.12, 0.45, 0.8]
    lens = [0.62, 0.9, 1.0, 0.98, 0.88, 0.72]
    for a, L in zip(ang, lens):
        x1, y1 = cx + math.sin(a) * r * L * 1.05, cy - r * 0.2 - math.cos(a) * r * L * 1.05
        ctx.save()
        ctx.translate(cx + math.sin(a) * r * 0.35, cy - r * 0.2 - math.cos(a) * r * 0.35)
        ctx.rotate(a)
        rrect(ctx, -r * 0.11, -r * L * 0.72, r * 0.22, r * L * 0.78, r * 0.11)
        set_color(ctx, skin)
        ctx.fill_preserve()
        set_color(ctx, shade(skin, 0.72))
        ctx.stroke()
        # knuckle creases
        ctx.move_to(-r * 0.06, -r * L * 0.35)
        ctx.line_to(r * 0.06, -r * L * 0.35)
        ctx.stroke()
        ctx.restore()
    # palm
    ellipse(ctx, cx, cy + r * 0.12, r * 0.5, r * 0.55)
    set_color(ctx, skin)
    ctx.fill_preserve()
    set_color(ctx, shade(skin, 0.72))
    ctx.stroke()
    # thumb on the wrong side too
    ctx.save()
    ctx.translate(cx + r * 0.42, cy + r * 0.25)
    ctx.rotate(1.2)
    rrect(ctx, -r * 0.1, -r * 0.5, r * 0.2, r * 0.55, r * 0.1)
    set_color(ctx, skin)
    ctx.fill_preserve()
    set_color(ctx, shade(skin, 0.72))
    ctx.stroke()
    ctx.restore()
    set_color(ctx, shade(skin, 0.8))
    ctx.move_to(cx - r * 0.2, cy + r * 0.3)
    ctx.curve_to(cx - r * 0.05, cy + r * 0.05, cx + r * 0.15, cy + r * 0.15, cx + r * 0.25, cy + r * 0.0)
    ctx.stroke()


def _thumb(ctx, cx, cy, r, bg=BLUE):
    circle(ctx, cx, cy, r)
    ctx.set_source(lin_grad(cx, cy - r, cx, cy + r, [(0, shade(bg, 1.25)), (1, shade(bg, 0.85))]))
    ctx.fill()
    set_color(ctx, WHITE)
    # the thumb
    rrect(ctx, cx - r * 0.5, cy - r * 0.05, r * 0.24, r * 0.55, r * 0.05)
    ctx.fill()
    ctx.move_to(cx - r * 0.2, cy - r * 0.05)
    ctx.line_to(cx - r * 0.02, cy - r * 0.55)
    ctx.curve_to(cx + r * 0.06, cy - r * 0.72, cx + r * 0.2, cy - r * 0.6, cx + r * 0.12, cy - r * 0.35)
    ctx.line_to(cx + r * 0.08, cy - r * 0.12)
    ctx.line_to(cx + r * 0.42, cy - r * 0.12)
    ctx.curve_to(cx + r * 0.58, cy - r * 0.1, cx + r * 0.56, cy + r * 0.12, cx + r * 0.44, cy + r * 0.14)
    ctx.curve_to(cx + r * 0.56, cy + r * 0.2, cx + r * 0.5, cy + r * 0.4, cx + r * 0.38, cy + r * 0.4)
    ctx.curve_to(cx + r * 0.46, cy + r * 0.52, cx + r * 0.36, cy + r * 0.6, cx + r * 0.25, cy + r * 0.58)
    ctx.line_to(cx - r * 0.2, cy + r * 0.52)
    ctx.close_path()
    ctx.fill()


def _badge(ctx, cx, cy, r, label):
    circle(ctx, cx, cy, r)
    ctx.set_source(rad_grad(cx - r * 0.3, cy - r * 0.4, r * 0.1, r * 1.1, [(0, "#FF7A86"), (1, RED)]))
    ctx.fill_preserve()
    set_color(ctx, WHITE)
    ctx.set_line_width(r * 0.13)
    ctx.stroke()
    text(ctx, label, cx, cy, r * (1.05 if len(label) <= 2 else 0.78), WHITE, bold=True, align="center",
         valign="middle")


def _spinner(ctx, cx, cy, r, c=MAUVE):
    for i in range(10):
        a = i / 10 * 2 * math.pi
        set_color(ctx, c, 0.15 + 0.85 * (i / 9))
        circle(ctx, cx + math.cos(a) * r, cy + math.sin(a) * r, r * (0.12 + 0.08 * i / 9))
        ctx.fill()


def _melting_clock(ctx, cx, cy, r):
    # dripping face
    ctx.move_to(cx - r, cy)
    ctx.arc(cx, cy, r, math.pi, 2 * math.pi)
    ctx.curve_to(cx + r, cy + r * 0.5, cx + r * 0.7, cy + r * 0.6, cx + r * 0.62, cy + r * 1.3)
    ctx.curve_to(cx + r * 0.6, cy + r * 1.6, cx + r * 0.4, cy + r * 1.6, cx + r * 0.38, cy + r * 1.25)
    ctx.curve_to(cx + r * 0.3, cy + r * 0.8, cx + r * 0.1, cy + r * 0.85, cx, cy + r * 0.95)
    ctx.curve_to(cx - r * 0.2, cy + r * 1.05, cx - r * 0.3, cy + r * 1.9, cx - r * 0.45, cy + r * 1.85)
    ctx.curve_to(cx - r * 0.6, cy + r * 1.8, cx - r * 0.5, cy + r * 1.1, cx - r * 0.7, cy + r * 0.85)
    ctx.curve_to(cx - r * 0.9, cy + r * 0.7, cx - r, cy + r * 0.4, cx - r, cy)
    ctx.close_path()
    set_color(ctx, WHITE)
    ctx.fill_preserve()
    set_color(ctx, "#8C8298")
    ctx.set_line_width(r * 0.1)
    ctx.stroke()
    # ticks
    ctx.set_line_width(r * 0.06)
    for i in range(12):
        a = i / 12 * 2 * math.pi
        k = 1 + 0.25 * max(0, math.sin(a)) ** 2
        ctx.move_to(cx + math.cos(a) * r * 0.78, cy + math.sin(a) * r * 0.78 * k)
        ctx.line_to(cx + math.cos(a) * r * 0.9, cy + math.sin(a) * r * 0.9 * k)
        ctx.stroke()
    set_color(ctx, INK)
    ctx.set_line_width(r * 0.09)
    ctx.move_to(cx, cy)
    ctx.line_to(cx + r * 0.1, cy - r * 0.6)
    ctx.stroke()
    ctx.move_to(cx, cy)
    ctx.curve_to(cx + r * 0.3, cy + r * 0.1, cx + r * 0.45, cy + r * 0.4, cx + r * 0.4, cy + r * 0.9)
    ctx.stroke()


def _captcha(ctx, rnd):
    _card(ctx, WHITE, 6)
    rrect(ctx, 8, 8, 84, 20, 3)
    set_color(ctx, "#4A86E8")
    ctx.fill()
    _lines(ctx, 13, 15, 50, 2, gap=7, c="#D8E6FF", lw=3.5)
    pal = ["#9FB4C7", "#B7C3B0", "#C9B3BF", "#8FA3A8", "#BFC7D6", "#A8B8A0"]
    for i in range(3):
        for j in range(3):
            x, y = 9 + i * 28, 32 + j * 22
            set_color(ctx, pal[(i * 3 + j + rnd.integers(6)) % 6])
            ctx.rectangle(x, y, 26, 20)
            ctx.fill()
            # blurry 'object' in the tile
            set_color(ctx, shade(pal[(i + j) % 6], 0.7))
            ellipse(ctx, x + 13 + rnd.uniform(-5, 5), y + 12, rnd.uniform(4, 9), rnd.uniform(3, 7))
            ctx.fill()
            if rnd.random() < 0.3:
                set_color(ctx, "#4A86E8")
                circle(ctx, x + 6, y + 6, 4)
                ctx.fill()
    ctx.set_line_width(2.5)
    set_color(ctx, "#8C8C99")
    rrect(ctx, 10, 101, 14, 14, 2)
    ctx.stroke()
    _lines(ctx, 30, 108, 50, 1, c="#9A9AA6", lw=4)


def _listicle(ctx, rnd, head, sub):
    _card(ctx, WHITE, 6)
    # thumbnail with a smeary uncanny image
    rrect(ctx, 7, 7, 86, 52, 4)
    ctx.set_source(lin_grad(0, 7, 0, 59, [(0, rnd.choice([CYAN, PINK, "#C8B8E8"])), (1, rnd.choice([FLESH, MAUVE, "#9FC9C0"]))]))
    ctx.fill()
    ctx.save()
    rrect(ctx, 7, 7, 86, 52, 4)
    ctx.clip()
    _uncanny_face(ctx, 50 + rnd.uniform(-12, 12), 40, 22, SKIN, rnd)
    set_color(ctx, RED)
    ctx.set_line_width(3.5)
    circle(ctx, 72, 24, 11)
    ctx.stroke()
    ctx.move_to(88, 8)
    ctx.line_to(76, 18)
    ctx.stroke()
    ctx.restore()
    text(ctx, head, 8, 76, 15, "#1B1A22", bold=True, tracking=-0.03)
    text(ctx, sub, 8, 93, 11.5, MAG, bold=True)
    _lines(ctx, 8, 104, 80, 2, gap=8, c="#CFC9D2", lw=4)


def _video(ctx, rnd):
    _card(ctx, "#221E2C", 6)
    rrect(ctx, 6, 6, 88, 66, 4)
    ctx.set_source(lin_grad(0, 6, 0, 72, [(0, "#6E5E86"), (1, "#C98FA8")]))
    ctx.fill()
    ctx.save()
    rrect(ctx, 6, 6, 88, 66, 4)
    ctx.clip()
    _uncanny_face(ctx, 34, 44, 24, SKIN, rnd, teeth=True)
    ctx.restore()
    set_color(ctx, WHITE, 0.92)
    circle(ctx, 68, 39, 13)
    ctx.fill()
    set_color(ctx, "#221E2C")
    poly(ctx, [(64, 32), (64, 46), (75, 39)])
    ctx.fill()
    set_color(ctx, "#55506A")
    ctx.rectangle(6, 76, 88, 4)
    ctx.fill()
    set_color(ctx, RED)
    ctx.rectangle(6, 76, 88 * rnd.uniform(0.3, 0.9), 4)
    ctx.fill()
    _lines(ctx, 8, 92, 78, 3, gap=9, c="#8A8499", lw=4)


def _chat(ctx, rnd):
    _card(ctx, rnd.choice([CYAN, "#D8CCF0", PINK]), 14)
    rrect(ctx, 12, 30, 76, 44, 20)
    set_color(ctx, WHITE)
    ctx.fill()
    poly(ctx, [(22, 70), (18, 86), (36, 72)])
    ctx.fill()
    for i in range(3):
        set_color(ctx, MAUVE, 0.45 + 0.25 * i)
        circle(ctx, 34 + i * 16, 52, 6)
        ctx.fill()
    _lines(ctx, 14, 100, 70, 2, gap=9, c=shade(MAUVE, 1.2), lw=4)


def _comment(ctx, rnd):
    _card(ctx, WHITE, 8)
    for k in range(3):
        y = 12 + k * 38
        set_color(ctx, rnd.choice([PINK, CYAN, MAUVE, "#C8B8E8", FLESH]))
        circle(ctx, 19, y + 10, 9)
        ctx.fill()
        _lines(ctx, 33, y + 5, 55, 2, gap=9, c="#B9B3C2", lw=4)
        set_color(ctx, RED)
        _heart(ctx, 80, y + 26, 3.5)
        ctx.fill()


def _ad(ctx, rnd, label):
    _card(ctx, rnd.choice(["#2B2440", "#3A2F5B"]), 8)
    rrect(ctx, 8, 8, 84, 60, 4)
    ctx.set_source(rad_grad(50, 38, 4, 60, [(0, "#FFD6E4"), (0.5, MAG), (1, "#6A1E5A")]))
    ctx.fill()
    set_color(ctx, WHITE)
    for i in range(7):
        a = i / 7 * 2 * math.pi
        ctx.move_to(50, 38)
        ctx.line_to(50 + math.cos(a) * 40, 38 + math.sin(a) * 40)
    ctx.set_line_width(2)
    ctx.stroke()
    text(ctx, "%d%%" % rnd.integers(70, 99), 50, 46, 22, WHITE, bold=True, align="center")
    rrect(ctx, 14, 80, 72, 28, 14)
    set_color(ctx, TEAL)
    ctx.fill()
    text(ctx, label, 50, 99, 13, WHITE, bold=True, align="center")


def _influencer(ctx, rnd):
    _card(ctx, rnd.choice([CYAN, "#C8B8E8", PINK]), 8)
    ctx.save()
    rrect(ctx, 2, 2, CW - 4, CH - 4, 10)
    ctx.clip()
    # hair blob
    set_color(ctx, rnd.choice(["#5E4A66", "#8A6B7E", "#3D3550", "#B79BB0"]))
    ellipse(ctx, 50, 58, 36, 44)
    ctx.fill()
    # neck + shoulders
    set_color(ctx, shade(SKIN, 0.85))
    ctx.rectangle(42, 80, 16, 20)
    ctx.fill()
    set_color(ctx, rnd.choice([WHITE, "#2B2440", MAG]))
    ellipse(ctx, 50, 128, 46, 30)
    ctx.fill()
    # too-perfect face
    ctx.set_source(rad_grad(44, 50, 3, 34, [(0, "#FFE6DC"), (1, SKIN)]))
    ellipse(ctx, 50, 60, 24, 30)
    ctx.fill()
    for sx in (-9, 9):
        set_color(ctx, WHITE)
        ellipse(ctx, 50 + sx, 56, 6.5, 4.2)
        ctx.fill()
        set_color(ctx, "#4B6F8A")
        circle(ctx, 50 + sx, 56, 3.2)
        ctx.fill()
        set_color(ctx, INK)
        circle(ctx, 50 + sx, 56, 1.6)
        ctx.fill()
    set_color(ctx, "#D9868F")
    ctx.move_to(40, 72)
    ctx.curve_to(45, 80, 55, 80, 60, 72)
    ctx.curve_to(55, 75, 45, 75, 40, 72)
    ctx.fill()
    set_color(ctx, WHITE, 0.9)
    ctx.rectangle(43, 73, 14, 2.5)
    ctx.fill()
    set_color(ctx, WHITE, 0.55)
    ellipse(ctx, 40, 46, 5, 3)
    ctx.fill()
    ctx.restore()
    # verified tick badge
    set_color(ctx, BLUE)
    circle(ctx, 84, 16, 8)
    ctx.fill()
    set_color(ctx, WHITE)
    ctx.set_line_width(2.4)
    ctx.move_to(80, 16)
    ctx.line_to(83, 19.5)
    ctx.line_to(88.5, 12.5)
    ctx.stroke()


def _eye(ctx, rnd):
    _card(ctx, rnd.choice([FLESH, PINK]), 12)
    ctx.move_to(12, 62)
    ctx.curve_to(35, 30, 65, 30, 88, 62)
    ctx.curve_to(65, 94, 35, 94, 12, 62)
    ctx.close_path()
    set_color(ctx, WHITE)
    ctx.fill()
    ctx.set_source(rad_grad(50, 62, 2, 20, [(0, INK), (0.35, INK), (0.4, "#5FA8C8"), (1, "#2E5E78")]))
    circle(ctx, 50, 62, 19)
    ctx.fill()
    set_color(ctx, WHITE)
    circle(ctx, 43, 55, 5)
    ctx.fill()
    set_color(ctx, "#7A4E5E")
    ctx.set_line_width(3)
    for i in range(7):
        x = 18 + i * 11
        ctx.move_to(x, 40 - 8 * math.sin(i / 6 * math.pi))
        ctx.line_to(x - 3, 28 - 12 * math.sin(i / 6 * math.pi))
        ctx.stroke()


def _mouth(ctx, rnd):
    _card(ctx, FLESH, 10)
    _uncanny_face(ctx, 50, 66, 40, rnd.choice([SKIN, "#F2B8C6", "#E6C2D6"]), rnd)


def _share(ctx, rnd):
    circle(ctx, 50, 62, 46)
    ctx.set_source(lin_grad(0, 16, 0, 108, [(0, "#FF6FAE"), (1, MAG)]))
    ctx.fill()
    set_color(ctx, WHITE)
    for (x, y) in ((34, 62), (66, 42), (66, 82)):
        circle(ctx, x, y, 8.5)
        ctx.fill()
    ctx.set_line_width(5)
    ctx.move_to(34, 62)
    ctx.line_to(66, 42)
    ctx.move_to(34, 62)
    ctx.line_to(66, 82)
    ctx.stroke()


def _likes(ctx, rnd):
    rrect(ctx, 4, 38, 92, 46, 23)
    set_color(ctx, WHITE)
    ctx.fill_preserve()
    set_color(ctx, "#D8D0DA")
    ctx.set_line_width(2)
    ctx.stroke()
    set_color(ctx, RED)
    _heart(ctx, 26, 62, 11)
    ctx.fill()
    text(ctx, rnd.choice(["1.2M", "88K", "4.1M", "999K"]), 64, 62, 17, "#2A2530", bold=True, align="center",
         valign="middle")


def _heart_icon(ctx, rnd):
    c = rnd.choice([RED, "#FF5C8A", MAG, "#FF8FB1"])
    _heart(ctx, 50, 64, 40)
    ctx.set_source(rad_grad(36, 44, 4, 70, [(0, shade(c, 1.45)), (0.5, c), (1, shade(c, 0.7))]))
    ctx.fill()
    set_color(ctx, WHITE, 0.7)
    ellipse(ctx, 34, 46, 9, 5, -0.6)
    ctx.fill()


def _progress(ctx, rnd):
    _card(ctx, "#2B2440", 8)
    _spinner(ctx, 50, 44, 20, CYAN)
    rrect(ctx, 12, 82, 76, 10, 5)
    set_color(ctx, "#4A4260")
    ctx.fill()
    rrect(ctx, 12, 82, 76 * 0.99, 10, 5)
    ctx.set_source(lin_grad(12, 0, 88, 0, [(0, TEAL), (1, MAG)]))
    ctx.fill()
    text(ctx, "99%", 50, 110, 14, CYAN, bold=True, align="center")


def _landscape(ctx, rnd):
    """an 'AI-generated' dreamscape: two moons, a melting house, impossible stairs"""
    _card(ctx, WHITE, 6)
    rrect(ctx, 6, 6, 88, 88, 4)
    ctx.set_source(lin_grad(0, 6, 0, 94, [(0, "#8FA8E8"), (0.55, "#F2B6D0"), (1, "#C8E8E0")]))
    ctx.fill()
    ctx.save()
    rrect(ctx, 6, 6, 88, 88, 4)
    ctx.clip()
    set_color(ctx, WHITE, 0.95)
    circle(ctx, 30, 28, 9)
    ctx.fill()
    circle(ctx, 70, 22, 6)
    ctx.fill()
    set_color(ctx, "#7C9A8E")
    ctx.move_to(6, 70)
    ctx.curve_to(30, 52, 50, 80, 94, 58)
    ctx.line_to(94, 94)
    ctx.line_to(6, 94)
    ctx.fill()
    set_color(ctx, "#E8DAF0")
    poly(ctx, [(44, 70), (44, 52), (54, 42), (64, 52), (64, 72)])
    ctx.fill()
    set_color(ctx, "#6E5E86")
    rrect(ctx, 51, 60, 7, 12, 1)
    ctx.fill()
    ctx.restore()
    _lines(ctx, 8, 106, 80, 2, gap=8, c="#B9B3C2", lw=4)


PALETTES = [PINK, CYAN, FLESH, "#C8B8E8", "#BFE3D4", WHITE, MAUVE]


def _design_list():
    D = []
    D.append(lambda c, r: (_card(c, r.choice([PINK, FLESH, "#C8B8E8"]), 12), _uncanny_face(c, 50, 62, 38, r.choice([SKIN, "#F2B8C6"]), r)))
    D.append(lambda c, r: (_card(c, r.choice([CYAN, "#BFE3D4"]), 12), _uncanny_face(c, 50, 62, 38, "#F2B8C6", r, pupils=0.07)))
    D.append(lambda c, r: (_card(c, r.choice([CYAN, "#C8B8E8", "#BFE3D4"]), 12), _hand6(c, 50, 70, 40, SKIN)))
    D.append(lambda c, r: (_card(c, r.choice([PINK, WHITE]), 12), _hand6(c, 50, 70, 40, "#E9B8A6")))
    D.append(lambda c, r: _listicle(c, r, "10 SHOCKING", "FACTS ABOUT YOU"))
    D.append(lambda c, r: _listicle(c, r, "YOU WON'T", "BELIEVE #7"))
    D.append(lambda c, r: _listicle(c, r, "DOCTORS", "HATE THIS ONE"))
    D.append(lambda c, r: _listicle(c, r, "TOP 10", "REASONS WHY"))
    D.append(lambda c, r: _thumb(c, 50, 62, 44))
    D.append(lambda c, r: _thumb(c, 50, 62, 44, "#5A8CFF"))
    D.append(_heart_icon)
    D.append(_heart_icon)
    D.append(lambda c, r: _badge(c, 50, 62, 40, "99+"))
    D.append(lambda c, r: _badge(c, 50, 62, 40, str(r.integers(2, 9))))
    D.append(lambda c, r: _badge(c, 50, 62, 40, str(r.integers(11, 99))))
    D.append(lambda c, r: (_card(c, WHITE, 12), _spinner(c, 50, 62, 26)))
    D.append(_captcha)
    D.append(lambda c, r: (_card(c, r.choice([CYAN, "#C8B8E8"]), 12), _melting_clock(c, 50, 46, 30)))
    D.append(_video)
    D.append(_chat)
    D.append(_comment)
    D.append(lambda c, r: _ad(c, r, "BUY NOW"))
    D.append(lambda c, r: _ad(c, r, "CLICK HERE"))
    D.append(_influencer)
    D.append(_influencer)
    D.append(_eye)
    D.append(_mouth)
    D.append(_share)
    D.append(_likes)
    D.append(_progress)
    D.append(_landscape)
    D.append(lambda c, r: (_card(c, WHITE, 12), _melting_clock(c, 50, 46, 30)))
    return D


# icons that are free-floating shapes (bubbles/foam uses them too)
FOAM_KINDS = (10, 11, 12, 13, 14, 8, 9, 27, 28)


def _render_design(fn, seed, px_h):
    s = px_h / CH
    w = int(math.ceil(CW * s))
    h = int(math.ceil(CH * s))
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surf)
    ctx.scale(s, s)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    fn(ctx, np.random.default_rng(seed))
    surf.flush()
    buf = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w].copy()
    return buf  # BGRA premultiplied uint8


_KEEP = []   # cairo surfaces created from numpy buffers must keep their buffers alive


def _to_surface(bgra):
    h, w = bgra.shape[:2]
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    buf = np.zeros((h, stride // 4, 4), np.uint8)
    buf[:, :w] = bgra
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_ARGB32, w, h, stride)
    _KEEP.append(buf)
    return surf


class Atlas:
    """sprites[design][shade][mip] -> (cairo.SurfacePattern, px_h)"""

    def __init__(self, n_variants=2, seed=7):
        designs = _design_list()
        self.n = len(designs) * n_variants
        self.sprites = []
        shadow = np.array([SHADOW[2], SHADOW[1], SHADOW[0]], np.float32)  # BGR order
        for vi in range(n_variants):
            for di, fn in enumerate(designs):
                base = _render_design(fn, seed + di * 31 + vi * 977, MIP_H[0]).astype(np.float32)
                shades = []
                for sh in range(SHADES):
                    k = sh / (SHADES - 1)
                    img = base.copy()
                    a = img[..., 3:4] / 255.0
                    # mix colour towards the shadow tone (premultiplied)
                    img[..., :3] = img[..., :3] * (1 - 0.72 * k) + shadow * 255.0 * a * 0.72 * k * 0.9
                    mips = []
                    for mh in MIP_H:
                        if mh == MIP_H[0]:
                            m = img
                        else:
                            ww = max(1, int(round(img.shape[1] * mh / MIP_H[0])))
                            m = cv2.resize(img, (ww, mh), interpolation=cv2.INTER_AREA)
                        surf = _to_surface(np.clip(m + 0.5, 0, 255).astype(np.uint8))
                        sw, shh = surf.get_width(), surf.get_height()
                        pat = cairo.SurfacePattern(surf)
                        pat.set_filter(cairo.FILTER_BILINEAR)
                        pat.set_matrix(cairo.Matrix(1, 0, 0, 1, sw / 2, shh / 2))
                        _KEEP.append(surf)
                        mips.append((pat, sw, shh))
                    shades.append(mips)
                self.sprites.append(shades)
        self.foam = [i for i in range(self.n) if (i % len(designs)) in FOAM_KINDS]

    def draw(self, ctx, idx, x, y, size, ang=0.0, sx=1.0, shade_k=0.0, alpha=1.0, px_scale=1.0):
        """draw sprite idx centred at (x,y) with height `size` (design units). shade_k 0..1 lit->shadow"""
        px = size * px_scale
        if px < 0.8:
            return
        mips = self.sprites[idx % self.n][min(SHADES - 1, max(0, int(shade_k * (SHADES - 1) + 0.5)))]
        mi = 0
        while mi + 1 < len(mips) and mips[mi + 1][2] >= px:
            mi += 1
        pat, w, h = mips[mi]
        k = size / h
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(ang)
        ctx.scale(k * sx if abs(sx) > 1e-3 else 1e-3, k)
        ctx.set_source(pat)
        ctx.rectangle(-w / 2, -h / 2, w, h)
        if alpha >= 0.999:
            ctx.fill()
        else:
            ctx.clip()
            ctx.paint_with_alpha(alpha)
        ctx.restore()

"""m05 wall: the cartoon (mosaicist's full-size design) of the Schismmancers' wall, in WALL coordinates.

Wall coordinates = design units of the zoom-1 frieze framing (the frieze fills 1920x1080 when the camera sits at
(960, 540), zoom 1); the wall extends beyond it (starry upper register for the flags, margins for camera moves).
Layout coordinates = wall - (WX0, WY0).

Registers (top -> bottom): starry lapis heaven (the flags unfurl here, Constantine's vision), pearl border,
titulus band IN · HOC · SIGNO · SCINDES, pearl/porphyry border, gold ground with the frieze of eleven warrior saints
(Schism Actual at the centre, hierarchic scale, his name titulus above his halo), meadow, jewelled + meander border.
"""
import math

import cairo
import numpy as np

from vx import *
from vx import mosaic as mz

SM = mz.SMALTI

WX0, WY0, WX1, WY1 = -420.0, -660.0, 2340.0, 1140.0
WALL_W, WALL_H = int(WX1 - WX0), int(WY1 - WY0)

Y_SKY1 = -14.0          # heaven ends
Y_TIT0, Y_TIT1 = 14.0, 132.0
Y_GOLD0, Y_GOLD1 = 150.0, 925.0
Y_EARTH0, Y_EARTH1 = 905.0, 990.0
GROUND = 958.0

N_SAINTS = 11
CENTRE = 5
SPACING = 186.0
H_SAINT = 600.0
H_ACTUAL = 640.0
TITVLVS = "IN HOC SIGNO SCINDES"


def saints():
    """[(kind, seed, x, h)] left to right"""
    out = []
    for i in range(N_SAINTS):
        x = 960.0 + (i - CENTRE) * SPACING
        if i == CENTRE:
            out.append(("schism_actual", 100, x, H_ACTUAL))
        else:
            out.append(("schismmancer", 11 + i * 7, x, H_SAINT))
    return out


# ------------------------------------------------------------------ helpers
def _fill(ctx, layer, c=None, on=False):
    """color: colour c; mask layers: white where `on`, cleared (transparent) otherwise"""
    if layer == "color":
        set_color(ctx, c)
        ctx.fill()
    elif on:
        ctx.set_source_rgba(1, 1, 1, 1)
        ctx.fill()
    else:
        ctx.save()
        ctx.set_operator(cairo.OPERATOR_CLEAR)
        ctx.fill()
        ctx.restore()


def _star(ctx, x, y, r0, r1, n=8, rot=0.0):
    pts = []
    for k in range(2 * n):
        r = r0 if k % 2 == 0 else r1
        a = rot + k * math.pi / n
        pts.append((x + r * math.cos(a), y + r * math.sin(a)))
    poly(ctx, pts)


def _letters(ctx, text, x, y, size, layer, colour, gold, font="roman", sep=None, tracking=0.14):
    """a Latin titulus in Roman square capitals (vx.mosaic Titulus: Cinzel, interpuncts, [ABC] overlines);
    size = cap height, y = baseline. 'fine' layer -> letters + their halo course; 'gold' -> gold leaf letters."""
    T = mz.Titulus(text, x, y, size, color=colour, font=font, sep=sep, tracking=tracking)
    if layer == "color":
        T.draw(ctx)
    elif layer == "fine":
        T.draw_fine(ctx)
    elif gold:
        T.draw_mask(ctx)
    else:
        ctx.save()
        ctx.set_operator(cairo.OPERATOR_CLEAR)
        T.draw_mask(ctx, margin=0.1)
        ctx.restore()
    return T


# ------------------------------------------------------------------ the static wall
def paint_static(ctx, layer, halos=None):
    """layer 'color' | 'gold' | 'fine' (masks: white = on, on a transparent canvas)"""
    rgb = layer == "color"
    # heaven: night lapis with gold stars (Galla Placidia)
    ctx.rectangle(WX0, WY0, WALL_W, Y_SKY1 - WY0)
    if rgb:
        ctx.set_source(lin_grad(0, WY0, 0, Y_SKY1, [(0, SM["night_vault"]), (0.7, SM["lapis_d"]), (1, SM["lapis"])]))
        ctx.fill()
    else:
        _fill(ctx, layer, None, False)
    k = 0
    for row, y in enumerate(np.arange(WY0 + 50, Y_SKY1 - 30, 96.0)):
        for x in np.arange(WX0 + 30 + (48 if row % 2 else 0), WX1, 96.0):
            r = 13 + 4 * hash01(k, 5)
            _star(ctx, x, y, r, r * 0.42, 8, hash01(k, 9) * 0.4)
            _fill(ctx, layer, SM["star"], layer == "gold")
            k += 1
    # pearl border (heaven / titulus)
    ctx.rectangle(WX0, Y_SKY1, WALL_W, Y_TIT0 - Y_SKY1)
    _fill(ctx, layer, SM["porphyry"], False)
    for x in np.arange(WX0 + 7, WX1, 14.0):
        circle(ctx, x, (Y_SKY1 + Y_TIT0) / 2, 4.6)
        _fill(ctx, layer, SM["marble"], layer == "fine")
    # titulus band
    ctx.rectangle(WX0, Y_TIT0, WALL_W, Y_TIT1 - Y_TIT0)
    if rgb:
        ctx.set_source(lin_grad(0, Y_TIT0, 0, Y_TIT1, [(0, SM["lapis_d"]), (0.5, SM["lapis"]), (1, SM["lapis_d"])]))
        ctx.fill()
    else:
        _fill(ctx, layer, None, False)
    _letters(ctx, TITVLVS, 960.0, Y_TIT1 - 30, 60, layer, SM["gold_l"], True, sep="·", tracking=0.2)
    # border titulus / gold: porphyry line + pearls
    ctx.rectangle(WX0, Y_TIT1, WALL_W, Y_GOLD0 - Y_TIT1)
    _fill(ctx, layer, SM["porphyry"], False)
    for x in np.arange(WX0 + 5, WX1, 10.0):
        circle(ctx, x, (Y_TIT1 + Y_GOLD0) / 2, 3.3)
        _fill(ctx, layer, SM["marble"], layer == "fine")
    # gold ground
    ctx.rectangle(WX0, Y_GOLD0, WALL_W, Y_GOLD1 - Y_GOLD0)
    _fill(ctx, layer, SM["gold"], layer == "gold")
    # Schism Actual's name titulus (dark red on gold), two lines above his halo
    if halos is not None:
        hx, hy, hr = halos[CENTRE]
        _letters(ctx, "SCHISMA\nACTVALIS", hx, hy - hr - 14 - 40, 28, layer, SM["porphyry"], False,
                 font="roman_black", tracking=0.06)
    # meadow
    ctx.rectangle(WX0, Y_EARTH0, WALL_W, Y_EARTH1 - Y_EARTH0)
    if rgb:
        ctx.set_source(lin_grad(0, Y_EARTH0, 0, Y_EARTH1, [(0, SM["verdigris"]), (0.35, SM["malachite"]),
                                                            (1, "#123A2C")]))
        ctx.fill()
    else:
        _fill(ctx, layer, None, False)
    k = 0
    for x in np.arange(WX0 + 10, WX1, 23.0):
        y = Y_EARTH0 + 18 + 55 * hash01(k, 3)
        c = SM["marble"] if hash01(k, 4) < 0.5 else SM["porphyry_l"]
        circle(ctx, x + 8 * hash01(k, 7), y, 3.2)
        _fill(ctx, layer, c, False)
        k += 1
    # lower border: white line, jewelled band, white line, meander on porphyry
    y0 = Y_EARTH1
    ctx.rectangle(WX0, y0, WALL_W, 6)
    _fill(ctx, layer, SM["marble"], False)
    ctx.rectangle(WX0, y0 + 6, WALL_W, 64)
    _fill(ctx, layer, SM["lapis_d"], False)
    k = 0
    x = WX0 + 20
    while x < WX1:
        if k % 2 == 0:
            ellipse(ctx, x, y0 + 38, 17, 22)
            _fill(ctx, layer, SM["gold"], layer == "gold")
            ellipse(ctx, x, y0 + 38, 11.5, 16)
            _fill(ctx, layer, SM["porphyry_l"], False)
        else:
            rrect(ctx, x - 16, y0 + 18, 32, 40, 3)
            _fill(ctx, layer, SM["gold"], layer == "gold")
            rrect(ctx, x - 10.5, y0 + 23.5, 21, 29, 2)
            _fill(ctx, layer, SM["malachite"], False)
        for dy in (-9, 9):
            circle(ctx, x + 32, y0 + 38 + dy, 5.2)
            _fill(ctx, layer, SM["marble"], False)
        x += 64
        k += 1
    ctx.rectangle(WX0, y0 + 70, WALL_W, 6)
    _fill(ctx, layer, SM["marble"], False)
    ctx.rectangle(WX0, y0 + 76, WALL_W, WY1 - y0 - 76)
    _fill(ctx, layer, SM["porphyry"], False)
    if rgb:
        u = 12.0
        ctx.new_path()
        x = WX0
        yb = y0 + 76 + 12 + 3 * u
        first = True
        while x < WX1:
            for (a, b) in ((0, 0), (0, 3), (2, 3), (2, 1), (1, 1), (1, 0), (4, 0)):
                (ctx.move_to if first else ctx.line_to)(x + a * u, yb - b * u)
                first = False
            x += 4 * u
        ctx.save()
        ctx.set_line_width(0.55 * u)
        ctx.set_line_join(cairo.LINE_JOIN_MITER)
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)
        set_color(ctx, SM["bone"])
        ctx.stroke()
        ctx.restore()


# ------------------------------------------------------------------ the orb (T-O map of the world)
def paint_orb(ctx, x, y, r, layer):
    """the sphaira of the Archistrategos, painted over vx.icon's orb (enlarged so 'every memeplex on Earth' reads)"""
    y -= 0.35 * r
    r *= 1.55
    circle(ctx, x, y, r)
    _fill(ctx, layer, SM["gold"], layer == "gold")
    ri = r * 0.86
    circle(ctx, x, y, ri)
    _fill(ctx, layer, SM["lapis_l"], layer == "fine")
    rl = ri * 0.8
    # Asia (upper half), Europe (lower left), Africa (lower right)
    for a0, a1, c in ((math.pi, 2 * math.pi, SM["terracotta"]), (math.pi / 2, math.pi, SM["verdigris"]),
                      (0, math.pi / 2, SM["bone"])):
        ctx.move_to(x, y)
        ctx.arc(x, y, rl, a0, a1)
        ctx.close_path()
        _fill(ctx, layer, c, layer == "fine")
    # the T: Mediterranean (stem) and Don / Nile (bar)
    ctx.rectangle(x - rl, y - 0.06 * r, 2 * rl, 0.12 * r)
    _fill(ctx, layer, SM["lapis_l"], layer == "fine")
    ctx.rectangle(x - 0.06 * r, y, 0.12 * r, rl)
    _fill(ctx, layer, SM["lapis_l"], layer == "fine")

"""t07: the changing Burns behind the lone flag-bearer, drawn in grey for the engraver (Gustave Doré).
Owner: T07Raised.  Coordinates: WALK-SCREEN units (the frame as seen during the walk: his feet at (1150, 1012),
578 px tall, horizon ~y 690). Each layer scrolls to the right as he walks left; `sc` = this layer's scroll
(q = screen_x - sc is the layer coordinate of what is at screen x)."""
import math

import cairo
import numpy as np

from vx.ease import clamp, hash01, noise1, fbm1, lerp
from scenes.t07_cells import grey, P, curve, limb

LAYERS = [("sky", 0.03), ("far", 0.12), ("mid", 0.35), ("ground", 1.0)]
FEET = (1150.0, 1012.0)
HOR = 690.0


def _lg(x0, y0, x1, y1, stops):
    g = cairo.LinearGradient(x0, y0, x1, y1)
    for s in stops:
        g.add_color_stop_rgba(s[0], s[1], s[1], s[1], s[2] if len(s) > 2 else 1.0)
    return g


def _rg(cx, cy, r0, r1, stops):
    g = cairo.RadialGradient(cx, cy, r0, cx, cy, r1)
    for s in stops:
        g.add_color_stop_rgba(s[0], s[1], s[1], s[1], s[2] if len(s) > 2 else 1.0)
    return g


def _vis(sc):
    """layer-x range that can be on screen (generous: zoom-out cells, the push-in)"""
    return -sc - 1400, -sc + 3400


# ====================================================================== skies
def _cloud(ctx, x, y, w, h, g_body, g_lit, seed, lit_dir=(-0.4, -1.0)):
    rng = np.random.default_rng(seed)
    puffs = []
    for i in range(14):
        px = x + rng.uniform(-0.5, 0.5) * w
        py = y + rng.uniform(-0.25, 0.3) * h * (1 - abs(px - x) / w)
        r = rng.uniform(0.18, 0.34) * h
        puffs.append((px, py, r))
    grey(ctx, g_body)
    for px, py, r in puffs:
        ctx.new_sub_path(); ctx.arc(px, py, r, 0, 2 * math.pi)
    ctx.fill()
    grey(ctx, g_lit)
    for px, py, r in puffs:
        ctx.new_sub_path(); ctx.arc(px + lit_dir[0] * r * 0.25, py + lit_dir[1] * r * 0.25, r * 0.8, 0, 2 * math.pi)
    ctx.fill()


def _sky_day(ctx, sc, top=0.72, hor=0.99, sun=(420, 230), seed=1, rays=True):
    x0, x1 = _vis(sc)
    ctx.set_source(_lg(0, -400, 0, HOR + 40, [(0, top), (0.75, hor * 0.97), (1, hor)]))
    ctx.rectangle(x0, -1400, x1 - x0, 3200); ctx.fill()
    # radiance behind the clouds
    ctx.set_source(_rg(sun[0], sun[1], 0, 900, [(0, 1.0, 0.95), (0.4, 0.95, 0.5), (1, 0.9, 0.0)]))
    ctx.rectangle(x0, -1400, x1 - x0, 3200); ctx.fill()
    if rays:
        for i in range(16):
            a = -2.6 + i * 0.12 + 0.03 * math.sin(i * 7.1)
            ctx.move_to(sun[0], sun[1])
            ctx.line_to(sun[0] + 2600 * math.cos(a - 0.018), sun[1] + 2600 * math.sin(a - 0.018) + 1500)
            ctx.line_to(sun[0] + 2600 * math.cos(a + 0.018), sun[1] + 2600 * math.sin(a + 0.018) + 1500)
            ctx.close_path()
        ctx.set_source(_rg(sun[0], sun[1], 0, 1400, [(0, 1.0, 0.35), (1, 1.0, 0.0)]))
        ctx.fill()
    rng = np.random.default_rng(seed)
    for i in range(9):
        cx = rng.uniform(-900, 2800)
        cy = rng.uniform(40, 460)
        w = rng.uniform(380, 820)
        _cloud(ctx, cx, cy, w, w * 0.45, 0.66 + 0.12 * (cy / 460), 0.97, seed * 31 + i)


def _sky_night(ctx, sc, moon=(1500, 200)):
    x0, x1 = _vis(sc)
    ctx.set_source(_lg(0, -400, 0, HOR + 40, [(0, 0.1), (1, 0.38)]))
    ctx.rectangle(x0, -1400, x1 - x0, 3200); ctx.fill()
    ctx.set_source(_rg(moon[0], moon[1], 0, 700, [(0, 0.75, 0.8), (1, 0.4, 0.0)]))
    ctx.rectangle(x0, -1400, x1 - x0, 3200); ctx.fill()
    grey(ctx, 0.99)
    ctx.arc(moon[0], moon[1], 58, 0, 2 * math.pi); ctx.fill()
    rng = np.random.default_rng(5)
    for i in range(160):
        x = rng.uniform(-900, 2900); y = rng.uniform(-300, HOR - 60)
        ctx.new_sub_path(); ctx.arc(x, y, rng.uniform(1.2, 3.0), 0, 2 * math.pi)
    ctx.fill()
    for i in range(5):
        x = -500 + i * 700 + 200 * math.sin(i * 3.3)
        _cloud(ctx, x, 180 + (i % 3) * 110, 700, 220, 0.2, 0.34, 70 + i, lit_dir=(0.6, -0.8))


def _sky_grey(ctx, sc):
    x0, x1 = _vis(sc)
    ctx.set_source(_lg(0, -400, 0, HOR + 40, [(0, 0.74), (1, 0.97)]))
    ctx.rectangle(x0, -1400, x1 - x0, 3200); ctx.fill()
    for i in range(7):
        x = -700 + i * 560
        _cloud(ctx, x, 150 + (i % 2) * 120, 780, 260, 0.8, 0.9, 90 + i)


# ====================================================================== far ranges
def _ridge(ctx, sc, y0, amp, g, seed, lit=None, step=30.0, sharp=1.0):
    x0, x1 = _vis(sc)
    xs = np.arange(x0 - step, x1 + step, step)
    ys = [y0 - amp * (0.5 + 0.5 * fbm1(x / 520.0, seed, 4)) ** sharp for x in xs]
    ctx.move_to(xs[0], 2000)
    for x, y in zip(xs, ys):
        ctx.line_to(x, y)
    ctx.line_to(xs[-1], 2000)
    ctx.close_path()
    grey(ctx, g)
    ctx.fill()
    if lit is not None:
        # sunlit faces: a lighter band just under the crest
        ctx.move_to(xs[0], ys[0])
        for x, y in zip(xs, ys):
            ctx.line_to(x, y)
        for x, y in zip(xs[::-1], ys[::-1]):
            ctx.line_to(x + 24, y + amp * 0.35)
        ctx.close_path()
        grey(ctx, lit)
        ctx.fill()
    return xs, ys


def _alps(ctx, sc):
    rng = np.random.default_rng(12)
    x0, x1 = _vis(sc)
    for layer, (gb, gl, y0, hmax) in enumerate(((0.8, 0.96, 640, 430), (0.62, 0.92, 670, 330))):
        x = x0 - 200
        while x < x1 + 200:
            w = rng.uniform(260, 520)
            h = rng.uniform(0.5, 1.0) * hmax
            px = x + w * rng.uniform(0.35, 0.65)
            grey(ctx, gb)
            P(ctx, [(x, y0 + 40), (px - 20, y0 - h + 30), (px, y0 - h), (px + 26, y0 - h + 40), (x + w, y0 + 40)])
            ctx.fill()
            grey(ctx, gl)            # snowy lit face
            P(ctx, [(px, y0 - h), (px - 20, y0 - h + 30), (px - w * 0.18, y0 - h * 0.55), (px - w * 0.05, y0 - h * 0.62),
                    (px + 8, y0 - h * 0.7)])
            ctx.fill()
            x += w * 0.7


# ====================================================================== pieces of the Burns
def _tent(ctx, x, y, w, g, kind=0):
    grey(ctx, g)
    if kind == 0:     # dome
        ctx.save(); ctx.translate(x, y); ctx.scale(1, 0.62)
        ctx.arc(0, 0, w / 2, math.pi, 2 * math.pi); ctx.close_path(); ctx.restore(); ctx.fill()
    elif kind == 1:   # bell tent
        P(ctx, [(x - w / 2, y), (x - w * 0.3, y - w * 0.35), (x, y - w * 0.8), (x + w * 0.3, y - w * 0.35), (x + w / 2, y)])
        ctx.fill()
    else:             # shade structure
        ctx.rectangle(x - w / 2, y - w * 0.5, w, w * 0.08); ctx.fill()
        for k in (-0.45, 0.45):
            ctx.rectangle(x + k * w - 3, y - w * 0.5, 6, w * 0.5); ctx.fill()


def _effigy(ctx, x, y, H, g, lattice=True, arms_up=True, g_light=None):
    """a lattice-wood figure (the Man), feet at (x, y), height H"""
    s = H / 1000.0
    ctx.save(); ctx.translate(x, y); ctx.scale(s, s)
    grey(ctx, g)
    # plinth
    P(ctx, [(-260, 0), (260, 0), (200, -220), (-200, -220)]); ctx.fill()
    # legs, torso, arms, head
    for a, b, w in (((-70, -220), (-40, -560), 46), ((70, -220), (40, -560), 46), ((0, -560), (0, -800), 90),
                    ((-40, -780), (-180, -1000) if arms_up else (-160, -600), 34),
                    ((40, -780), (180, -1000) if arms_up else (160, -600), 34)):
        limb(ctx, a, b, w, w * 0.85); ctx.fill()
    P(ctx, [(0, -890), (-50, -830), (0, -770), (50, -830)]); ctx.fill()
    if lattice:
        grey(ctx, g_light if g_light is not None else min(1, g + 0.35))
        ctx.set_line_width(5)
        for i in range(-3, 4):
            ctx.move_to(-200 + i * 40, -220); ctx.line_to(-160 + i * 40, 0)
        for yy in range(-560, -220, 40):
            ctx.move_to(-80, yy); ctx.line_to(80, yy)
        ctx.stroke()
    ctx.restore()


def _pine(ctx, x, y, H, g, seed):
    rng = np.random.default_rng(seed)
    grey(ctx, g)
    ctx.rectangle(x - H * 0.02, y - H * 0.25, H * 0.04, H * 0.25); ctx.fill()
    n = 7
    for i in range(n):
        yy = y - H * 0.18 - i * H * 0.12
        w = H * 0.26 * (1 - i / n) + rng.uniform(-6, 6)
        P(ctx, [(x - w, yy), (x, yy - H * 0.2), (x + w, yy)]); ctx.fill()


def _trunk(ctx, x, top, bot, w, g, g_lit=None, seed=0):
    grey(ctx, g)
    P(ctx, [(x - w * 0.5, bot), (x - w * 0.38, top), (x + w * 0.38, top), (x + w * 0.55, bot)]); ctx.fill()
    # roots
    for k in (-1, 1):
        P(ctx, [(x + k * w * 0.4, bot - 60), (x + k * w * 1.3, bot + 10), (x + k * w * 0.2, bot + 10)]); ctx.fill()
    if g_lit is not None:
        grey(ctx, g_lit)
        P(ctx, [(x - w * 0.46, bot), (x - w * 0.36, top), (x - w * 0.18, top), (x - w * 0.22, bot)]); ctx.fill()


# ====================================================================== the stations, per layer
def draw_layer(ctx, station, layer, T, sc):
    f = _STATIONS[station]
    f(ctx, layer, T, sc)


def _playa(ctx, layer, T, sc):
    if layer == "sky":
        _sky_day(ctx, sc, top=0.7, hor=0.99, sun=(560, 250), seed=3)
    elif layer == "far":
        _ridge(ctx, sc, HOR - 20, 170, 0.76, 11, lit=0.88)
        _ridge(ctx, sc, HOR + 5, 70, 0.66, 12)
    elif layer == "mid":
        # the Burn city on the horizon: domes, bell tents, shade structures, the Man on his plinth
        x0, x1 = _vis(sc)
        rng = np.random.default_rng(21)
        for i in range(90):
            x = rng.uniform(x0, x1)
            _tent(ctx, x, HOR + 14 + rng.uniform(0, 14), rng.uniform(18, 44), 0.42 + rng.uniform(0, 0.2), i % 3)
        _effigy(ctx, 520 - sc * 0 - 0, HOR + 18, 300, 0.3)
        # a dust devil
        ctx.set_source(_lg(0, HOR - 300, 0, HOR + 20, [(0, 0.85, 0.0), (1, 0.72, 0.8)]))
        P(ctx, [(-260, HOR + 20), (-300, HOR - 300), (-150, HOR - 300), (-230, HOR + 20)]); ctx.fill()
    elif layer == "ground":
        x0, x1 = _vis(sc)
        ctx.set_source(_lg(0, HOR, 0, 1100, [(0, 0.95), (0.6, 0.8), (1, 0.42)]))
        ctx.rectangle(x0, HOR, x1 - x0, 900); ctx.fill()
        # cracked playa: polygons receding in perspective
        rng = np.random.default_rng(7)
        grey(ctx, 0.6)
        ctx.set_line_width(2.5)
        for row in range(14):
            y = HOR + 8 + (row ** 1.7) * 3.2
            step = 40 + row * 18
            xx = x0 - (x0 % step)
            while xx < x1:
                ctx.move_to(xx, y); ctx.line_to(xx + step * rng.uniform(0.5, 0.9), y + rng.uniform(-3, 3) * (1 + row * 0.2))
                ctx.move_to(xx, y); ctx.line_to(xx + rng.uniform(-8, 8), y + (row + 1) * 5)
                xx += step
        ctx.stroke()
        # his long shadow (sun behind, low, to the left): falls to the right
        grey(ctx, 0.5)
        fx = FEET[0] - sc
        P(ctx, [(fx - 30, FEET[1] - 4), (fx + 40, FEET[1] + 6), (fx + 560, FEET[1] + 60), (fx + 520, FEET[1] + 40)])
        ctx.fill()


def _forest(ctx, layer, T, sc):
    if layer == "sky":
        _sky_day(ctx, sc, top=0.6, hor=0.97, sun=(900, 120), seed=8)
    elif layer == "far":
        x0, x1 = _vis(sc)
        for i in range(int((x1 - x0) / 90) + 2):
            x = x0 + i * 90 + 30 * math.sin(i * 2.3)
            _trunk(ctx, x, -600, HOR + 30, 26 + 10 * math.sin(i * 1.7), 0.8)
    elif layer == "mid":
        x0, x1 = _vis(sc)
        # a wooden temple in the clearing, lanterns strung between the trunks
        tx = 900 - 2.0 * 0
        grey(ctx, 0.4)
        for k in range(4):
            w = 300 - k * 60
            y = HOR + 20 - k * 90
            P(ctx, [(tx - w, y), (tx - w * 0.7, y - 70), (tx + w * 0.7, y - 70), (tx + w, y)]); ctx.fill()
            ctx.rectangle(tx - w * 0.55, y - 2, w * 1.1, 24)
        ctx.fill()
        grey(ctx, 0.97)
        for k in range(4):
            ctx.rectangle(900 - 30, HOR - 40 - k * 90, 60, 30)
        ctx.fill()
        for i in range(int((x1 - x0) / 330) + 2):
            x = x0 + i * 330 + 90 * math.sin(i * 3.7)
            _trunk(ctx, x, -800, HOR + 60, 70, 0.42, g_lit=0.62)
            grey(ctx, 0.3); ctx.set_line_width(3)
            ctx.move_to(x, 260); ctx.curve_to(x + 110, 330, x + 220, 330, x + 330, 262); ctx.stroke()
            grey(ctx, 0.98)
            for k in range(4):
                lx = x + 60 + k * 70
                ctx.new_sub_path(); ctx.arc(lx, 300 + 20 * math.sin(k * 1.4) + 14, 9, 0, 2 * math.pi)
            ctx.fill()
    elif layer == "ground":
        x0, x1 = _vis(sc)
        ctx.set_source(_lg(0, HOR, 0, 1100, [(0, 0.55), (1, 0.28)]))
        ctx.rectangle(x0, HOR, x1 - x0, 900); ctx.fill()
        # light shafts on the floor
        grey(ctx, 0.78)
        for i in range(int((x1 - x0) / 700) + 2):
            x = x0 + i * 700
            ctx.save(); ctx.translate(x + 200, HOR + 170); ctx.scale(3.2, 0.5); ctx.arc(0, 0, 60, 0, 2 * math.pi)
            ctx.restore(); ctx.fill()
        # ferns and roots
        grey(ctx, 0.18); ctx.set_line_width(4)
        rng = np.random.default_rng(31)
        for i in range(int((x1 - x0) / 60)):
            x = x0 + i * 60 + rng.uniform(0, 40)
            y = HOR + 150 + rng.uniform(0, 250)
            for k in range(5):
                a = -math.pi / 2 + (k - 2) * 0.4
                ctx.move_to(x, y); ctx.curve_to(x + math.cos(a) * 30, y + math.sin(a) * 30, x + math.cos(a) * 60,
                                                 y + math.sin(a) * 40, x + math.cos(a) * 80, y + math.sin(a) * 20)
        ctx.stroke()


def _lake(ctx, layer, T, sc):
    if layer == "sky":
        _sky_day(ctx, sc, top=0.66, hor=0.99, sun=(700, 180), seed=13)
    elif layer == "far":
        _alps(ctx, sc)
    elif layer == "mid":
        x0, x1 = _vis(sc)
        # the lake (light, shimmering) and the far shore with a tall tower effigy + its reflection
        grey(ctx, 0.88); ctx.rectangle(x0, HOR - 20, x1 - x0, 140); ctx.fill()
        grey(ctx, 0.5); ctx.set_line_width(2)
        for i in range(12):
            y = HOR - 10 + i * 11
            ctx.move_to(x0, y); ctx.line_to(x1, y)
        ctx.stroke()
        grey(ctx, 0.3)
        P(ctx, [(x0, HOR - 16), (x1, HOR - 16), (x1, HOR - 30), (x0, HOR - 30)]); ctx.fill()
        tx = 640
        grey(ctx, 0.25)
        P(ctx, [(tx - 60, HOR - 20), (tx - 20, HOR - 380), (tx + 20, HOR - 380), (tx + 60, HOR - 20)]); ctx.fill()
        P(ctx, [(tx - 50, HOR - 380), (tx, HOR - 450), (tx + 50, HOR - 380)]); ctx.fill()
        grey(ctx, 0.62)
        P(ctx, [(tx - 60, HOR - 16), (tx - 22, HOR + 110), (tx + 22, HOR + 110), (tx + 60, HOR - 16)]); ctx.fill()
        rng = np.random.default_rng(41)
        for i in range(30):
            _tent(ctx, rng.uniform(x0, x1), HOR - 18, rng.uniform(14, 30), 0.3, i % 2)
    elif layer == "ground":
        x0, x1 = _vis(sc)
        ctx.set_source(_lg(0, HOR + 100, 0, 1100, [(0, 0.72), (1, 0.45)]))
        curve(ctx, [(x0, HOR + 130), (x0 + 600, HOR + 110), (x1, HOR + 140), (x1, 1800), (x0, 1800)], close=True)
        ctx.fill()
        rng = np.random.default_rng(43)
        for i in range(int((x1 - x0) / 25)):
            x = rng.uniform(x0, x1); y = rng.uniform(HOR + 150, 1100)
            r = 3 + (y - HOR) * 0.03 * rng.uniform(0.5, 1.2)
            grey(ctx, 0.3); ctx.new_sub_path(); ctx.arc(x, y, r, 0, 2 * math.pi); ctx.fill()
            grey(ctx, 0.9); ctx.new_sub_path(); ctx.arc(x - r * 0.3, y - r * 0.3, r * 0.45, 0, 2 * math.pi); ctx.fill()


NOFIRE_Q = None


def _nofire(ctx, layer, T, sc):
    if layer == "sky":
        _sky_night(ctx, sc, moon=(720, 190))
    elif layer == "far":
        _ridge(ctx, sc, HOR - 10, 120, 0.2, 51)
    elif layer == "mid":
        # the effigy that never burned: passes right behind him at 'fire' (155.45)
        from scenes.t07_walker import walked, WALKCAM
        q = FEET[0] - 330 - walked(155.451) * WALKCAM[0] * 0.35
        _effigy(ctx, q, HOR + 50, 660, 0.06, g_light=0.55)
        # moonlight rim on the effigy's right side
        # the crowd sitting in a ring, waiting; a sign on a post: NO FIRE
        rng = np.random.default_rng(61)
        for i in range(26):
            x = q + rng.uniform(-620, 620)
            y = HOR + 30 + rng.uniform(0, 30)
            grey(ctx, 0.1)
            ctx.new_sub_path(); ctx.arc(x, y - 44, 13, 0, 2 * math.pi); ctx.fill()
            P(ctx, [(x - 22, y), (x - 16, y - 34), (x + 16, y - 34), (x + 24, y)]); ctx.fill()
        sx = q - 420
        grey(ctx, 0.15); ctx.rectangle(sx - 5, HOR - 140, 10, 170); ctx.fill()
        grey(ctx, 0.9); ctx.rectangle(sx - 110, HOR - 200, 220, 80); ctx.fill()
        grey(ctx, 0.1); ctx.set_line_width(5); ctx.rectangle(sx - 110, HOR - 200, 220, 80); ctx.stroke()
        ctx.select_font_face("Archivo Black"); ctx.set_font_size(40)
        e = ctx.text_extents("NO FIRE")
        ctx.move_to(sx - e.x_advance / 2, HOR - 145); ctx.show_text("NO FIRE"); ctx.new_path()
        # a bucket brigade's buckets, cold
        for k in range(5):
            bx = q - 300 + k * 140
            grey(ctx, 0.35); P(ctx, [(bx - 14, HOR + 70), (bx - 10, HOR + 44), (bx + 10, HOR + 44), (bx + 14, HOR + 70)])
            ctx.fill()
    elif layer == "ground":
        x0, x1 = _vis(sc)
        ctx.set_source(_lg(0, HOR, 0, 1100, [(0, 0.3), (1, 0.14)]))
        ctx.rectangle(x0, HOR, x1 - x0, 900); ctx.fill()
        # moonlit ruts
        grey(ctx, 0.5); ctx.set_line_width(3)
        for i in range(8):
            y = HOR + 40 + i * i * 6
            ctx.move_to(x0, y); ctx.line_to(x1, y + 10)
        ctx.stroke()


def _years(ctx, layer, T, sc):
    from scenes.t07_walker import walked, WALKCAM, WALK0, _ch, spec
    leaves = 157.25 < T < 158.15
    if layer == "sky":
        _sky_grey(ctx, sc)
    elif layer == "far":
        _ridge(ctx, sc, HOR - 5, 90, 0.72, 71, lit=0.85)
        # a line of bare trees
        x0, x1 = _vis(sc)
        grey(ctx, 0.5); ctx.set_line_width(3)
        for i in range(int((x1 - x0) / 110) + 1):
            x = x0 + i * 110
            ctx.move_to(x, HOR); ctx.line_to(x, HOR - 70)
            for k in range(4):
                ctx.move_to(x, HOR - 30 - k * 10); ctx.line_to(x + (15 + k * 3) * (1 if k % 2 else -1), HOR - 50 - k * 12)
        ctx.stroke()
    elif layer == "mid":
        x0, x1 = _vis(sc)
        # a sagging fence and a lone bare oak; the tear-off calendar on its post
        grey(ctx, 0.35); ctx.set_line_width(5)
        for i in range(int((x1 - x0) / 140) + 1):
            x = x0 + i * 140
            ctx.move_to(x, HOR + 60); ctx.line_to(x + 4, HOR - 10)
        ctx.move_to(x0, HOR + 10); ctx.line_to(x1, HOR + 20)
        ctx.move_to(x0, HOR + 36); ctx.line_to(x1, HOR + 44)
        ctx.stroke()
        ox = 300
        grey(ctx, 0.22)
        P(ctx, [(ox - 22, HOR + 60), (ox - 12, HOR - 220), (ox + 12, HOR - 220), (ox + 24, HOR + 60)]); ctx.fill()
        ctx.set_line_width(8)
        for k in range(7):
            a = -math.pi / 2 + (k - 3) * 0.35
            ctx.move_to(ox, HOR - 200)
            ctx.curve_to(ox + math.cos(a) * 90, HOR - 200 + math.sin(a) * 90, ox + math.cos(a) * 160,
                         HOR - 220 + math.sin(a) * 150, ox + math.cos(a) * 230, HOR - 230 + math.sin(a) * 180)
        ctx.stroke()
    elif layer == "ground":
        x0, x1 = _vis(sc)
        snow = 0.95 if not leaves else 0.72
        ctx.set_source(_lg(0, HOR, 0, 1100, [(0, snow), (1, snow - 0.12)]))
        ctx.rectangle(x0, HOR, x1 - x0, 900); ctx.fill()
        # his footprints trailing behind him (they scroll away to the right)
        grey(ctx, 0.55)
        for t, side in _ch().footsteps(156.3, T, "walk"):
            q = FEET[0] - walked(t) * WALKCAM[0] + (16 if side == "near" else -16) * 0 + (8 if side == "near" else -6)
            y = FEET[1] + (6 if side == "near" else -6)
            ctx.save(); ctx.translate(q, y); ctx.scale(1, 0.28); ctx.arc(0, 0, 22, 0, 2 * math.pi); ctx.restore()
            ctx.fill()


_STATIONS = {"playa": _playa, "forest": _forest, "lake": _lake, "nofire": _nofire, "years": _years}


# ====================================================================== particles (in front of the world,
# behind the Flag's spot plate): snow, autumn leaves, the calendar pages 2020 -> 2021
def draw_particles(ctx, T, sc):
    snow = (156.3 < T < 157.35) or T > 158.05
    leaves = 157.2 < T < 158.2
    cal = 157.2 < T < 158.3
    if snow:
        a = clamp((T - 156.3) / 0.3) if T < 157.35 else clamp((T - 158.05) / 0.3)
        rng = np.random.default_rng(81)
        n = 420
        xs = rng.uniform(-400, 2400, n)
        ys = rng.uniform(-200, 1200, n)
        rs = rng.uniform(2, 7, n)
        sp = rng.uniform(60, 160, n)
        for x, y, r, v in zip(xs, ys, rs, sp):
            yy = (y + T * v) % 1400 - 200
            xx = x + 40 * math.sin(T * 1.3 + y) + T * 90 * (r / 7)
            xx = (xx - 0.0) % 2800 - 400
            ctx.new_sub_path(); ctx.arc(xx, yy, r, 0, 2 * math.pi)
        grey(ctx, 0.99, a)
        ctx.fill()
    if leaves:
        a = clamp((T - 157.2) / 0.2) * clamp((158.2 - T) / 0.2)
        rng = np.random.default_rng(83)
        for i in range(60):
            x0 = rng.uniform(-400, 2400)
            y0 = rng.uniform(-300, 1100)
            ph = rng.uniform(0, 6.28)
            yy = (y0 + T * 160) % 1400 - 200
            xx = (x0 + T * 260 + 60 * math.sin(T * 2 + ph)) % 2800 - 400
            ctx.save(); ctx.translate(xx, yy); ctx.rotate(T * 3 + ph); ctx.scale(1, 0.5 + 0.4 * math.sin(T * 5 + ph))
            curve(ctx, [(-16, 0), (0, -9), (16, 0), (0, 9)], close=True)
            ctx.restore()
            grey(ctx, 0.2 + 0.4 * ((i * 37) % 3) / 2, a)
            ctx.fill()
    if cal:
        _calendar(ctx, T)


def _calendar(ctx, T):
    """a tear-off calendar's pages fly past him: 2020 ... and on 'Burns' the year turns: 2021"""
    rng = np.random.default_rng(91)
    for i in range(14):
        t0 = 157.2 + i * 0.065
        u = (T - t0) / 0.75
        if u < 0 or u > 1:
            continue
        x = -250 + u * 2400 + rng.uniform(-80, 80)
        y = 250 + rng.uniform(-120, 380) + 120 * math.sin(u * 4 + i)
        ang = u * 5 + i
        year = "2020" if t0 < 157.5 else "2021"
        month = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "JAN", "FEB"][i]
        ctx.save(); ctx.translate(x, y); ctx.rotate(ang); ctx.scale(1, 0.4 + 0.6 * abs(math.cos(u * 6 + i)))
        grey(ctx, 0.97); ctx.rectangle(-70, -90, 140, 180); ctx.fill()
        grey(ctx, 0.1); ctx.set_line_width(4); ctx.rectangle(-70, -90, 140, 180); ctx.stroke()
        ctx.rectangle(-70, -90, 140, 34); ctx.fill()
        ctx.select_font_face("Archivo Black"); ctx.set_font_size(24)
        grey(ctx, 0.97)
        e = ctx.text_extents(year); ctx.move_to(-e.x_advance / 2, -64); ctx.show_text(year)
        grey(ctx, 0.1); ctx.set_font_size(38)
        e = ctx.text_extents(month); ctx.move_to(-e.x_advance / 2, 0); ctx.show_text(month)
        ctx.set_font_size(50)
        d = str(1 + (i * 7) % 28)
        e = ctx.text_extents(d); ctx.move_to(-e.x_advance / 2, 64); ctx.show_text(d)
        ctx.new_path()
        ctx.restore()


# ====================================================================== the foreground wipers (front layer)
def draw_wiper(ctx, kind, sx, T, outer=None):
    """a solid foreground object sweeping across (screen units; covers the seam between two Burns)"""
    ctx.save()
    if outer is not None:
        ctx.transform(outer)
    if kind in ("trunk", "flurry"):
        grey(ctx, 0.12)
        P(ctx, [(sx - 190, 1300), (sx - 150, -200), (sx + 150, -200), (sx + 200, 1300)]); ctx.fill()
        grey(ctx, 0.3)
        P(ctx, [(sx - 150, 1300), (sx - 120, -200), (sx - 60, -200), (sx - 80, 1300)]); ctx.fill()
        grey(ctx, 0.05); ctx.set_line_width(5)
        for k in range(9):
            ctx.move_to(sx - 100 + k * 30, -200); ctx.curve_to(sx - 90 + k * 30, 400, sx - 110 + k * 30, 800,
                                                             sx - 95 + k * 30, 1300)
        ctx.stroke()
    elif kind == "rock":
        grey(ctx, 0.16)
        curve(ctx, [(sx - 330, 1300), (sx - 300, 700), (sx - 160, 420), (sx + 60, 380), (sx + 250, 520),
                    (sx + 330, 900), (sx + 340, 1300)], close=True)
        ctx.fill()
        grey(ctx, 0.4)
        curve(ctx, [(sx - 160, 440), (sx + 60, 400), (sx + 180, 480), (sx + 20, 520), (sx - 120, 560)], close=True)
        ctx.fill()
    elif kind == "tent":
        grey(ctx, 0.14)
        P(ctx, [(sx - 420, 1300), (sx - 360, 700), (sx, 250), (sx + 360, 700), (sx + 420, 1300)]); ctx.fill()
        grey(ctx, 0.3); ctx.set_line_width(6)
        for k in (-2, -1, 0, 1, 2):
            ctx.move_to(sx, 250); ctx.line_to(sx + k * 180, 1300)
        ctx.stroke()
    ctx.restore()

"""m04 cartoons: the Hypermind (ophanim-polycandelon) panel. Every painter draws into a m04_tiles.Paint (colour +
material) in its own frame (world_H for the wall/stack, local frames for wings/rings); the scene tessellates them.

world_H: 2100 x 1760 design units; the great eye (centre of the wheels) at (CX, CY).
"""
import math

import cairo
import numpy as np

from vx.ease import clamp, smoothstep, lerp
from vx import mosaic as mz

S = {k: tuple(float(x) for x in v) for k, v in mz.SC.items()}


def hexc(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def mixc(a, b, t):
    t = clamp(t)
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def mulc(a, k):
    return tuple(min(1.0, x * k) for x in a)


INK = S["ink"]
WHITE = hexc("#F2EDE2")
PEARL = hexc("#EDE6D6")
VERMIL = hexc("#B8392A")
CRIMSON = hexc("#8E2231")
ORANGE = hexc("#C9652A")
EMBERG = hexc("#D99A3C")
WHITEGOLD = hexc("#FFEFD0")
GLASS_G = hexc("#4E9A78")
GLASS_A = hexc("#C9822C")
GLASS_R = hexc("#A8323A")
FLAME = hexc("#FFD6A0")
NIGHT = hexc("#0B1030")
BLACKBLUE = hexc("#070A1C")
BRONZE_D = hexc("#2E2112")
IRIS = hexc("#22357E")

WH = (2100, 1760)
CX, CY = 1050.0, 700.0
# the wheels: (r_out, r_in, eyes, eye width)
RINGS = [(394.0, 318.0, 14, 60.0), (306.0, 244.0, 11, 52.0), (226.0, 178.0, 8, 42.0)]
RING_M = 12.0                      # local canvas margin
CORE_R = 124.0
GREAT_EYE_W = 158.0
# polycandelon plates (y, rx, ry, lamps)
PLATES = [(206.0, 272.0, 40.0, 14), (298.0, 214.0, 31.0, 10), (386.0, 166.0, 24.0, 8), (468.0, 118.0, 18.0, 0)]
MAND = (CX, CY - 10.0, 660.0, 700.0)   # mandorla centre + radii
GROUND_Y = 1566.0
DONORS = {"lutie": (790.0, GROUND_Y, 296.0), "crocus": (1310.0, GROUND_Y, 308.0)}


def ring_extent(k):
    r = RINGS[k][0] + RING_M
    return (2 * r, 2 * r)


def _ellipse(cx, cy, rx, ry):
    def f(ctx):
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(rx, ry)
        ctx.new_sub_path()
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
    return f


def _circles(pts, r):
    def f(ctx):
        for (x, y) in pts:
            ctx.new_sub_path()
            ctx.arc(x, y, r, 0, 2 * math.pi)
    return f


# ============================================================ eyes
def eye(P, x, y, w, open=1.0, look=(0.0, 0.0), rot=0.0, glow=0.35, iris=None, lid_gold=True, fine=1.0,
        white=WHITE, setting=None):
    """Byzantine almond eye (x-axis = rot). open 0 = shut (a curved lash line), 1 = wide.
    setting: optional colour of a dark almond bed behind the eye (so it reads on gold)."""
    h = 0.44 * w
    o = clamp(open)
    iris = iris or IRIS

    def frame(ctx):
        ctx.translate(x, y)
        ctx.rotate(rot)

    if setting is not None:
        def bed(ctx):
            ctx.save()
            frame(ctx)
            ctx.move_to(-w * 0.62, 0)
            ctx.curve_to(-w * 0.25, -h * 1.1, w * 0.25, -h * 1.1, w * 0.62, 0)
            ctx.curve_to(w * 0.25, h * 0.95, -w * 0.25, h * 0.95, -w * 0.62, 0)
            ctx.close_path()
            ctx.restore()
        P.fill(bed, setting, fine=fine)
    up = -h * (0.18 + 1.0 * o)
    lo = h * (0.2 + 0.62 * o)
    lidc = S["gold_d"] if lid_gold else INK
    if o < 0.06:
        def lid(ctx):
            ctx.save()
            frame(ctx)
            ctx.move_to(-w / 2, 0)
            ctx.curve_to(-w * 0.2, h * 0.3, w * 0.2, h * 0.3, w / 2, 0)
            ctx.restore()
        P.stroke(lid, lidc, max(1.5, 0.1 * w), gold=0.5 if lid_gold else 0.0, fine=fine)
        return

    def almond(ctx):
        ctx.save()
        frame(ctx)
        ctx.move_to(-w / 2, 0)
        ctx.curve_to(-w * 0.22, up, w * 0.22, up, w / 2, 0)
        ctx.curve_to(w * 0.22, lo, -w * 0.22, lo, -w / 2, 0)
        ctx.close_path()
        ctx.restore()

    P.fill(almond, white, glow=glow * o, fine=fine)
    P.save()
    P.clip(almond)
    ix = look[0] * w * 0.2
    iy = look[1] * h * 0.25 + h * 0.06
    ri = 0.52 * h * (0.85 + 0.25 * o)

    def disc(r, dx=0.0, dy=0.0):
        def f(ctx):
            ctx.save()
            frame(ctx)
            ctx.new_sub_path()
            ctx.arc(ix + dx, iy + dy, r, 0, 2 * math.pi)
            ctx.restore()
        return f

    P.fill(disc(ri), iris, glow=0.12 * glow * o, fine=fine)
    P.fill(disc(ri * 0.5), BLACKBLUE, fine=fine)
    P.fill(disc(ri * 0.22, -ri * 0.34, -ri * 0.34), hexc("#FFF4D8"), glow=0.9 * o, fine=fine)
    P.restore()
    P.stroke(almond, INK, max(1.4, 0.07 * w), fine=fine)

    def upper(ctx):
        ctx.save()
        frame(ctx)
        ctx.move_to(-w * 0.53, 0)
        ctx.curve_to(-w * 0.22, up * 1.06, w * 0.22, up * 1.06, w * 0.53, 0)
        ctx.restore()
    P.stroke(upper, lidc, max(1.6, 0.095 * w), gold=0.7 if lid_gold else 0.0, fine=fine)


# ============================================================ the wheels (local frame of ring k)
def ring_painter(k, opens, looks, glow=0.3, heat=0.0):
    """ring k in its local canvas (centre in the middle): a broad gold wheel with a jewelled rim of pearls,
    full of eyes set in dark lapis almonds. opens/looks: per eye."""
    r_out, r_in, ne, ew = RINGS[k]
    ex, ey = ring_extent(k)
    cx, cy = ex / 2, ey / 2

    def ann(r1, r2):
        def f(ctx):
            ctx.new_sub_path()
            ctx.arc(cx, cy, r2, 0, 2 * math.pi)
            ctx.new_sub_path()
            ctx.arc_negative(cx, cy, r1, 2 * math.pi, 0)
        return f

    def paint(P):
        rim = 9.0
        P.fill(ann(r_in - 2.0, r_out + 2.0), INK)
        P.fill(ann(r_out - rim, r_out), CRIMSON)
        P.fill(ann(r_in, r_in + rim), CRIMSON)
        P.fill(ann(r_in + rim + 1.5, r_out - rim - 1.5), S["gold"], gold=1.0)
        for rr in (r_out - rim / 2, r_in + rim / 2):
            n = int(2 * math.pi * rr / 12.0)
            pts = [(cx + rr * math.cos(i * 2 * math.pi / n), cy + rr * math.sin(i * 2 * math.pi / n)) for i in range(n)]
            P.fill(_circles(pts, 2.6), PEARL, fine=1.0)
        rm = 0.5 * (r_in + r_out)
        # small lapis lozenges between the eyes
        loz = []
        for i in range(ne):
            a2 = -math.pi / 2 + (i + 0.5) * 2 * math.pi / ne
            loz.append((cx + rm * math.cos(a2), cy + rm * math.sin(a2), a2))

        def lozp(ctx):
            for qx, qy, a2 in loz:
                ca, sa = math.cos(a2), math.sin(a2)
                pts = [(-7, 0), (0, -5.5), (7, 0), (0, 5.5)]
                for j, (u, v) in enumerate(pts):
                    X, Y = qx + u * -sa + v * ca, qy + u * ca + v * sa
                    (ctx.move_to if j == 0 else ctx.line_to)(X, Y)
                ctx.close_path()
        P.fill(lozp, S["lapis"], fine=1.0)
        for i in range(ne):
            a = -math.pi / 2 + i * 2 * math.pi / ne
            px, py = cx + rm * math.cos(a), cy + rm * math.sin(a)
            eye(P, px, py, ew, open=opens[i], look=looks[i], rot=a + math.pi / 2, glow=glow,
                setting=S["lapis_d"])
    return paint


# ============================================================ polycandelon stack + core (world_H)
def chains_painter(P, top=-500.0):
    y0 = PLATES[0][0]
    rx = PLATES[0][1]
    links_o, links_e = [], []
    for dx in (-rx * 0.93, 0.0, rx * 0.93):
        x1, y1 = CX + dx, y0 - (6 if dx else 34)
        x0, yy0 = CX + dx * 0.05, top
        L = math.hypot(x1 - x0, y1 - yy0)
        n = int(L / 16)
        ang = math.atan2(y1 - yy0, x1 - x0)
        for i in range(n):
            f = (i + 0.5) / n
            px, py = x0 + (x1 - x0) * f, yy0 + (y1 - yy0) * f
            (links_o if i % 2 == 0 else links_e).append((px, py, ang))

    def oval(ctx):
        for px, py, ang in links_o:
            ctx.save()
            ctx.translate(px, py)
            ctx.rotate(ang)
            ctx.scale(9.0, 4.6)
            ctx.new_sub_path()
            ctx.arc(0, 0, 1, 0, 2 * math.pi)
            ctx.restore()

    def bar(ctx):
        for px, py, ang in links_e:
            ca, sa = math.cos(ang), math.sin(ang)
            ctx.move_to(px - 8.5 * ca, py - 8.5 * sa)
            ctx.line_to(px + 8.5 * ca, py + 8.5 * sa)
    P.stroke(oval, INK, 5.6)
    P.stroke(oval, S["gold"], 3.2, gold=1.0)
    P.stroke(bar, INK, 6.0)
    P.stroke(bar, S["gold_d"], 3.8, gold=0.8)
    P.fill(_ellipse(CX, y0 - 34, 17, 11), S["gold"], gold=1.0)


def plate(P, y, rx, ry, lamps, lamp_on, t):
    th = max(9.0, ry * 0.45)

    def band(ctx):
        ctx.save()
        ctx.translate(CX, y)
        ctx.scale(rx, ry)
        ctx.new_sub_path()
        ctx.arc(0, 0, 1, 0, math.pi)
        ctx.restore()
        ctx.line_to(CX - rx, y - th)
        ctx.save()
        ctx.translate(CX, y - th)
        ctx.scale(rx, ry)
        ctx.arc_negative(0, 0, 1, math.pi, 0)
        ctx.restore()
        ctx.close_path()
    P.fill(_ellipse(CX, y - th * 0.5, rx + 3, ry + 3 + th * 0.5), INK)
    P.fill(band, S["gold"], gold=1.0)
    P.fill(_ellipse(CX, y, rx, ry), S["gold_l"], gold=1.0)
    P.fill(_ellipse(CX, y, rx * 0.9, ry * 0.86), BRONZE_D)
    # openwork piercings (dark) in two rings + a gold inner rim
    for f, n, r in ((0.72, max(10, int(rx / 13)), 5.5), (0.45, max(8, int(rx / 20)), 4.5)):
        pts = [(CX + math.cos(i * 2 * math.pi / n) * rx * f, y + math.sin(i * 2 * math.pi / n) * ry * f) for i in range(n)]

        def holes(ctx, pts=pts, r=r):
            for (px, py) in pts:
                ctx.save()
                ctx.translate(px, py)
                ctx.scale(1.0, max(0.35, ry / rx * 1.6))
                ctx.new_sub_path()
                ctx.arc(0, 0, r, 0, 2 * math.pi)
                ctx.restore()
        P.fill(holes, hexc("#0E0A05"))
    P.stroke(_ellipse(CX, y, rx * 0.58, ry * 0.58), S["gold"], 3.0, gold=1.0)
    P.fill(_ellipse(CX, y, rx * 0.22, ry * 0.3), S["gold"], gold=1.0)
    n = int(rx / 9)
    pts = [(CX + math.cos(i * 2 * math.pi / n) * rx * 0.95, y + math.sin(i * 2 * math.pi / n) * ry * 0.95) for i in range(n)]
    P.fill(_circles(pts, 2.3), PEARL)
    # lamps: coloured glass cups hanging under the rim, a flame at each mouth
    for i in range(lamps):
        a = (i + 0.5) * 2 * math.pi / lamps
        lx, ly = CX + math.cos(a) * rx * 0.96, y + math.sin(a) * ry * 0.96 + 2
        on = lamp_on(i)
        glass = (GLASS_G, GLASS_A, GLASS_R)[i % 3]
        fl = 0.82 + 0.18 * math.sin(t * 13.0 + i * 1.7) * math.sin(t * 7.3 + i)

        def cup(ctx, lx=lx, ly=ly):
            ctx.move_to(lx - 8.5, ly - 1)
            ctx.curve_to(lx - 8.5, ly + 15, lx - 3, ly + 23, lx, ly + 25)
            ctx.curve_to(lx + 3, ly + 23, lx + 8.5, ly + 15, lx + 8.5, ly - 1)
            ctx.close_path()
        gcol = mixc(mulc(glass, 0.4), mixc(glass, FLAME, 0.3), on * fl)
        P.fill(cup, gcol, glow=0.75 * on * fl, fine=1.0)
        P.stroke(cup, INK, 1.8, fine=1.0)
        if on > 0.02:
            def flame(ctx, lx=lx, ly=ly, k=on * fl):
                hh = 11 * k + 3
                ctx.move_to(lx, ly - 3 - hh)
                ctx.curve_to(lx + 5, ly - 3 - hh * 0.4, lx + 4, ly - 1, lx, ly)
                ctx.curve_to(lx - 4, ly - 1, lx - 5, ly - 3 - hh * 0.4, lx, ly - 3 - hh)
            P.fill(flame, mixc(ORANGE, FLAME, 0.7), glow=1.0, fine=1.0)


def coil(P, x, y0, y1, turns=3, r=7.0, phase=0.0):
    def f(ctx):
        n = 72
        for i in range(n + 1):
            f_ = i / n
            yy = y0 + (y1 - y0) * f_
            env = math.sin(math.pi * f_) ** 0.6
            xx = x + r * env * math.sin(2 * math.pi * turns * f_ + phase)
            yy += 0.45 * r * env * math.cos(2 * math.pi * turns * f_ + phase)
            (ctx.move_to if i == 0 else ctx.line_to)(xx, yy)
    P.stroke(f, INK, 5.4)
    P.stroke(f, S["gold"], 3.0, gold=1.0)


def stack_painter(st):
    """st: dict(t, lamp_on(plate, i)->0..1, eye_open, eye_look, core_glow, portal)"""
    t = st["t"]

    def paint(P):
        chains_painter(P)
        y_top = PLATES[0][0] - 36
        y_bot = CY - CORE_R + 10
        P.fill(lambda ctx: ctx.rectangle(CX - 15, y_top, 30, y_bot - y_top), INK)
        P.fill(lambda ctx: ctx.rectangle(CX - 12, y_top, 24, y_bot - y_top), S["gold_d"], gold=0.9)
        P.fill(lambda ctx: ctx.rectangle(CX - 6, y_top, 6, y_bot - y_top), S["gold_l"], gold=1.0)
        for k in range(len(PLATES) - 1):
            ya = PLATES[k][0]
            yb, rxb = PLATES[k + 1][0], PLATES[k + 1][1]
            for fx in (-0.86, 0.86):
                x = CX + fx * rxb
                P.stroke(lambda ctx, x=x: (ctx.move_to(x, ya + 4), ctx.line_to(x, yb)), INK, 7.0)
                P.stroke(lambda ctx, x=x: (ctx.move_to(x, ya + 4), ctx.line_to(x, yb)), S["gold"], 4.2, gold=1.0)
            for j, fx in enumerate((-0.6, -0.28, 0.28, 0.6)):
                coil(P, CX + fx * rxb, ya + 6, yb - 3, turns=2 + (j % 2), r=6 + 2 * (j % 2), phase=j * 1.3)
        for k in range(len(PLATES) - 1, -1, -1):
            y, rx, ry, nl = PLATES[k]
            plate(P, y, rx, ry, nl, lambda i, k=k: st["lamp_on"](k, i), t)
        y3 = PLATES[-1][0]
        for j, fx in enumerate((-0.5, 0.5)):
            coil(P, CX + fx * 64, y3 + 6, CY - CORE_R + 10, turns=3, r=6, phase=j)
        core_painter(P, st)
    return paint


def core_painter(P, st):
    o = st["eye_open"]
    g = st["core_glow"]
    R = CORE_R
    P.fill(_ellipse(CX, CY, R + 5, R + 5), INK)
    P.fill(_ellipse(CX, CY, R, R), S["gold"], gold=1.0)
    n = 44
    pts = [(CX + math.cos(i * 2 * math.pi / n) * (R - 7), CY + math.sin(i * 2 * math.pi / n) * (R - 7)) for i in range(n)]
    P.fill(_circles(pts, 3.0), PEARL, fine=1.0)
    P.fill(_ellipse(CX, CY, R - 15, R - 15), INK)
    field = mixc(NIGHT, hexc("#1E3180"), g)
    P.fill(_ellipse(CX, CY, R - 18, R - 18), field, glow=0.25 * g)
    rays = []
    for i in range(20):
        a = i * 2 * math.pi / 20
        r0, r1 = R * 0.66, R - 21
        rays.append([(CX + math.cos(a - 0.045) * r0, CY + math.sin(a - 0.045) * r0),
                     (CX + math.cos(a) * r1, CY + math.sin(a) * r1),
                     (CX + math.cos(a + 0.045) * r0, CY + math.sin(a + 0.045) * r0)])

    def rayp(ctx):
        for tri in rays:
            ctx.move_to(*tri[0])
            ctx.line_to(*tri[1])
            ctx.line_to(*tri[2])
            ctx.close_path()
    P.fill(rayp, mixc(S["gold_d"], S["gold_l"], g), gold=0.7, glow=0.2 * g, fine=1.0)
    portal = st.get("portal", 0.0)
    eye(P, CX, CY, GREAT_EYE_W, open=o, look=st["eye_look"], glow=0.3 + 0.35 * g,
        iris=mixc(IRIS, hexc("#3353B8"), g), lid_gold=True, setting=BLACKBLUE)
    if portal > 0:
        rp = 12 + 22 * portal
        P.fill(_ellipse(CX, CY + 4, rp, rp * 0.92), mixc(WHITEGOLD, WHITE, 0.4), glow=1.0, fine=1.0)


# ============================================================ the wall behind (world_H)
def curved_text(P, s, cx, cy, rx, ry, a_mid, size, rgb, n_show=None, glow=0.0, gold=0.0, flash=None,
                font="VX Cinzel", tracking=0.14):
    """letters along the bottom of an ellipse, reading left -> right, tops toward the centre"""
    c = P.c
    c.save()
    c.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    c.set_font_size(size)
    adv = [c.text_extents(ch).x_advance + tracking * size for ch in s]
    c.restore()
    total = sum(adv) - tracking * size
    r_mean = 0.5 * (rx + ry)
    span = total / r_mean
    a = a_mid + span / 2
    n_show = len(s) if n_show is None else n_show
    for k, (ch, ad) in enumerate(zip(s, adv)):
        am = a - (ad / 2) / r_mean
        if k < n_show and ch.strip():
            px, py = cx + rx * math.cos(am), cy + ry * math.sin(am)
            rot = am - math.pi / 2
            fk = flash(k) if flash else 0.0

            def path(ctx, ch=ch, px=px, py=py, rot=rot):
                ctx.save()
                ctx.translate(px, py)
                ctx.rotate(rot)
                ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                ctx.set_font_size(size)
                e = ctx.text_extents(ch)
                ctx.move_to(-e.x_advance / 2, size * 0.36)
                ctx.text_path(ch)
                ctx.restore()
            P.fill(path, mixc(rgb, WHITEGOLD, fk), glow=glow + fk, gold=gold, fine=1.0)
        a -= ad / r_mean


_STARS = None


def _stars():
    global _STARS
    if _STARS is None:
        rng = np.random.default_rng(11)
        mx, my, mrx, mry = MAND
        pts = []
        while len(pts) < 150:
            a = rng.uniform(0, 2 * math.pi)
            f = math.sqrt(rng.uniform(0.25, 0.95))
            sx, sy = mx + math.cos(a) * mrx * f, my + math.sin(a) * mry * f
            if math.hypot(sx - CX, sy - CY) < 420:
                continue
            pts.append((sx, sy, rng.uniform(2.5, 4.5)))
        _STARS = pts
    return _STARS


def wall_painter(st):
    """st: dict(t, rays 0..1, caption_n, caption_flash, donors(P) or None)"""
    def paint(P):
        W_, H_ = WH
        P.fill(lambda ctx: ctx.rectangle(-600, -800, W_ + 1200, H_ + 1600), S["gold"], gold=1.0)
        # the lower register (earth): a dark lapis band behind the donors
        P.fill(lambda ctx: ctx.rectangle(-600, 1250, W_ + 1200, GROUND_Y - 1250), S["lapis_d"])
        P.fill(lambda ctx: ctx.rectangle(-600, 1242, W_ + 1200, 10), INK)
        mx, my, mrx, mry = MAND
        bands = [(1.0, hexc("#131D4E")), (0.915, S["lapis"]), (0.83, hexc("#17245C")), (0.72, S["lapis_d"]),
                 (0.6, hexc("#0B1130")), (0.47, BLACKBLUE)]
        P.fill(_ellipse(mx, my, mrx + 12, mry + 12), INK)
        for f, colr in bands:
            P.fill(_ellipse(mx, my, mrx * f, mry * f), colr)
        for f in (0.995, 0.918):
            P.stroke(_ellipse(mx, my, mrx * f, mry * f), S["gold"], 5.0, gold=1.0)
        stars = _stars()

        def starp(ctx):
            for sx, sy, rr in stars:
                ctx.move_to(sx - rr, sy)
                ctx.line_to(sx, sy - rr)
                ctx.line_to(sx + rr, sy)
                ctx.line_to(sx, sy + rr)
                ctx.close_path()
        P.fill(starp, S["star"], gold=0.4, fine=1.0)
        # rays of the transfiguration
        ra = st.get("rays", 0.0)
        beams = []
        for i in range(12):
            a = -math.pi / 2 + (i + 0.5) * 2 * math.pi / 12
            r0 = 420.0
            beams.append([(mx + math.cos(a - 0.016) * r0, my + math.sin(a - 0.016) * r0),
                          (mx + math.cos(a) * mrx * 0.9, my + math.sin(a) * mry * 0.9),
                          (mx + math.cos(a + 0.016) * r0, my + math.sin(a + 0.016) * r0)])

        def beamp(ctx):
            for tri in beams:
                ctx.move_to(*tri[0])
                ctx.line_to(*tri[1])
                ctx.line_to(*tri[2])
                ctx.close_path()
        P.fill(beamp, mixc(hexc("#4A5F9E"), hexc("#C9D6F0"), ra), glow=0.12 * ra, fine=1.0)
        curved_text(P, "HIC · HYPERMENS · VEXILLVM · COMPILAT", mx, my, mrx * 0.9575, mry * 0.9575, math.pi / 2, 50,
                    S["gold_l"], n_show=st.get("caption_n", 0), flash=st.get("caption_flash"), glow=0.2,
                    gold=st.get("caption_gold", 1.0), tracking=0.1)
        # floor: a meadow band on a porphyry base
        gy = GROUND_Y
        P.fill(lambda ctx: ctx.rectangle(-600, gy - 6, W_ + 1200, 72), S["malachite"])
        P.fill(lambda ctx: ctx.rectangle(-600, gy + 66, W_ + 1200, 12), INK)
        P.fill(lambda ctx: ctx.rectangle(-600, gy + 78, W_ + 1200, 60), S["porphyry"])
        rng = np.random.default_rng(5)
        fl = [[], [], [], []]
        stems = []
        for i in range(80):
            fx = rng.uniform(-200, W_ + 200)
            fy = gy + rng.uniform(10, 58)
            fl[i % 4].append((fx, fy))
            stems.append((fx, fy))
        P.stroke(lambda ctx: [(ctx.move_to(x, y + 4), ctx.line_to(x + 2, y + 16)) for x, y in stems],
                 S["verdigris"], 2.4)
        for j, colr in enumerate([S["marble"], hexc("#D9A441"), S["porphyry_l"], S["bone"]]):
            P.fill(_circles(fl[j], 4.6), colr, fine=1.0)
        if st.get("donors"):
            st["donors"](P)
    return paint


# ============================================================ wings (local frame: root at (0,0), axis +x)
WINGS = [
    # name, side (+1 right / -1 left), root offset from (CX,CY), open angle, folded angle, length, width
    ("mid_r", 1, (70.0, -30.0), -8.0, 76.0, 930.0, 180.0),
    ("mid_l", -1, (-70.0, -30.0), -172.0, 104.0, 930.0, 180.0),
    ("up_r", 1, (36.0, -110.0), -63.0, -95.0, 720.0, 160.0),
    ("up_l", -1, (-36.0, -110.0), -117.0, -85.0, 720.0, 160.0),
    ("lo_r", 1, (40.0, 90.0), 54.0, 97.0, 640.0, 150.0),
    ("lo_l", -1, (-40.0, 90.0), 126.0, 83.0, 640.0, 150.0),
]
# ignition order (canon I..VI)
IGN_ORDER = ["up_l", "up_r", "mid_l", "mid_r", "lo_l", "lo_r"]
ROMAN = ["I", "II", "III", "IV", "V", "VI"]
WING_PAD = 30.0


def wing_extent(L, Wd):
    return (L + 2 * WING_PAD, 2 * Wd + 2 * WING_PAD)


def wing_origin(L, Wd):
    return (WING_PAD, Wd + WING_PAD)


def wing_painter(L, Wd, mirror, ign, front, t, label, eyes_open, seed=0):
    """a seraph's wing: jewelled arm (coverts in scales, the CANON titulus along it) and long primaries with
    peacock eyes. ign 0..1 settled fire; front 0..1.3 = the ignition front along the wing (root -> tip)."""
    ox, oy = wing_origin(L, Wd)
    F = 7
    arm_len = 0.5 * L
    feathers = []
    for k in range(F):
        u = k / (F - 1)
        bx = (0.1 + 0.36 * u) * L
        by = -Wd * 0.2 + Wd * 0.05 * u
        th = math.radians(lerp(24.0, 2.0, u))
        tip_x = lerp(0.62, 1.0, u ** 0.9) * L
        ln = (tip_x - bx) / math.cos(th)
        wd = Wd * lerp(0.36, 0.3, u)
        feathers.append((bx, by, th, ln, wd, u))

    def lit(x):
        if front <= 0:
            return 0.0
        return clamp((front * L - x) / (0.1 * L) + 0.5)

    def flash(x):
        if front <= 0 or front >= 1.3:
            return 0.0
        d = (x - front * L) / (0.08 * L)
        return math.exp(-d * d)

    def paint(P):
        P.push(np.array([[1.0, 0.0, ox], [0.0, 1.0, oy]]))
        for (bx, by, th, ln, wd, u) in feathers:
            ca, sa = math.cos(th), math.sin(th)
            ang = math.atan2(sa, ca)

            def outline(ctx, bx=bx, by=by, ang=ang, ln=ln, wd=wd, k=1.0):
                ctx.save()
                ctx.translate(bx, by)
                ctx.rotate(ang)
                ctx.move_to(0, -wd * 0.5 * k)
                ctx.curve_to(ln * 0.5, -wd * 0.58 * k, ln * 0.92, -wd * 0.46 * k, ln, 0)
                ctx.curve_to(ln * 0.9, wd * 0.5 * k, ln * 0.5, wd * 0.6 * k, 0, wd * 0.5 * k)
                ctx.close_path()
                ctx.restore()
            xm = bx + ca * ln * 0.6
            h = lit(xm)
            fx = flash(xm)
            flick = 0.92 + 0.08 * math.sin(t * 8.0 + u * 5.0 + seed)
            dorm = mixc(S["lapis_d"], S["lapis"], 0.25 + 0.3 * u)
            body = mixc(dorm, mulc(mixc(CRIMSON, VERMIL, 0.5 + 0.4 * u), flick), h)
            P.fill(outline, INK)
            P.fill(lambda ctx, o=outline: o(ctx, k=0.8), mixc(body, WHITEGOLD, 0.8 * fx), glow=0.1 * h + 0.9 * fx)
            # a vermilion core stripe + gold edge (gold leaf that glitters once lit)
            P.fill(lambda ctx, o=outline: o(ctx, k=0.34), mixc(mixc(S["lapis"], S["slate"], 0.3),
                                                              mixc(ORANGE, EMBERG, u), h), glow=0.12 * h + 0.8 * fx)

            def shaft(ctx, bx=bx, by=by, ca=ca, sa=sa, ln=ln):
                ctx.move_to(bx + ca * 10, by + sa * 10)
                ctx.line_to(bx + ca * ln * 0.94, by + sa * ln * 0.94)
            P.stroke(shaft, mixc(S["silver"], S["gold_l"], h), 2.6, gold=h)
            ex_, ey_ = bx + ca * ln * 0.8, by + sa * ln * 0.8
            eye(P, ex_, ey_, wd * 0.62, open=eyes_open * clamp(h * 1.2), rot=ang, glow=0.35,
                iris=mixc(IRIS, S["porphyry"], 0.3), lid_gold=True, setting=mixc(S["lapis_d"], CRIMSON, h))

        def arm(ctx):
            ctx.move_to(-12, -Wd * 0.02)
            ctx.curve_to(arm_len * 0.15, -Wd * 0.72, arm_len * 0.7, -Wd * 0.66, arm_len * 1.08, -Wd * 0.3)
            ctx.curve_to(arm_len * 0.95, -Wd * 0.05, arm_len * 0.55, Wd * 0.1, arm_len * 0.2, Wd * 0.14)
            ctx.curve_to(arm_len * 0.06, Wd * 0.14, -12, Wd * 0.1, -12, -Wd * 0.02)
            ctx.close_path()
        P.fill(arm, INK)
        P.save()
        P.clip(arm)
        P.fill(lambda ctx: ctx.rectangle(-30, -Wd, L, 2 * Wd), mixc(S["lapis_d"], CRIMSON, lit(arm_len * 0.4)))
        # scales (imbricated), alternately gold leaf and vermilion once lit
        rs = Wd * 0.13
        for r in range(4):
            y = -Wd * 0.48 + r * Wd * 0.17
            n = int(arm_len / (2 * rs)) + 2
            gpts, vpts, dpts = [], [], []
            for i in range(n):
                x = rs * (0.3 + 2 * i) + (r % 2) * rs
                if (i + r) % 2 == 0:
                    gpts.append((x, y))
                else:
                    vpts.append((x, y))
            hm = lit(arm_len * 0.35)

            def scales(pts):
                def f(ctx):
                    for (x, y_) in pts:
                        ctx.new_sub_path()
                        ctx.arc(x, y_, rs, 0, math.pi)
                        ctx.close_path()
                return f
            P.fill(scales(gpts), mixc(mixc(S["lapis"], S["slate"], 0.4), S["gold"], hm), gold=hm)
            P.fill(scales(vpts), mixc(S["lapis_d"], VERMIL, hm), glow=0.08 * hm)
            P.stroke(scales(gpts + vpts), mixc(S["silver"], S["gold_l"], hm), 1.8, gold=0.5 * hm)
        # titulus band along the arm: CANON I..VI (skipped for the tiny icon)
        if label:
            hl = lit(arm_len * 0.5)
            fl = flash(arm_len * 0.5)
            for ctx in P.all:
                ctx.save()
                ctx.translate(arm_len * 0.52, -Wd * 0.27)
                ctx.rotate(math.radians(3.0))
                if mirror:
                    ctx.scale(-1, 1)
            bw = 34 + 27.0 * len(label)
            P.fill(lambda ctx: ctx.rectangle(-bw / 2, -31, bw, 60), mixc(INK, hexc("#2A0A10"), hl))
            P.stroke(lambda ctx: ctx.rectangle(-bw / 2, -31, bw, 60), mixc(S["slate"], S["gold"], hl), 3.2,
                     gold=hl)
            tcol = mixc(mixc(S["slate"], S["silver"], 0.5), WHITEGOLD, hl)
            P.text(label, 0, 15.5, 44, tcol, glow=0.5 * hl + fl, fine=1.0, tracking=0.06, font="VX Cinzel Black")
            P.restore()
        P.restore()
        P.stroke(arm, mixc(S["slate"], S["gold"], lit(arm_len * 0.3)), 3.0, gold=lit(arm_len * 0.3))
        P.pop()
    return paint


# ============================================================ the Hypermind as a small flat icon
def seraph_icon(P, cx, cy, sc, look_down=1.0, wake=1.0, glow=0.35):
    """the locked Hypermind as a flat seraph for a small roundel: six crimson wings with gold feather lines and
    eyes, three gold wheels full of eyes, the great eye. Everything in fine tiles."""
    for (name, side, off, a_open, a_fold, L, Wd) in WINGS:
        a = math.radians(a_open)
        rx, ry = cx + sc * off[0], cy + sc * off[1]
        Ls, ws = sc * L * 0.93, sc * Wd * 0.75
        ca, sa = math.cos(a), math.sin(a)
        nx, ny = -sa * side, ca * side                  # the trailing (feather) side
        if name.startswith("lo"):
            nx, ny = -nx, -ny

        def pt(u, v):
            return (rx + ca * u + nx * v, ry + sa * u + ny * v)

        def wing(ctx):
            ctx.move_to(*pt(0, -ws * 0.2))
            ctx.curve_to(*pt(Ls * 0.3, -ws * 0.62), *pt(Ls * 0.75, -ws * 0.45), *pt(Ls, 0))
            for k in range(4):
                u0 = Ls * (1 - 0.16 * k)
                u1 = Ls * (1 - 0.16 * (k + 1))
                ctx.curve_to(*pt(u0 - Ls * 0.04, ws * (0.2 + 0.1 * k)), *pt(u1 + Ls * 0.02, ws * (0.35 + 0.1 * k)),
                             *pt(u1, ws * (0.18 + 0.08 * k)))
            ctx.curve_to(*pt(Ls * 0.2, ws * 0.5), *pt(Ls * 0.05, ws * 0.3), *pt(0, ws * 0.2))
            ctx.close_path()
        P.fill(wing, INK, fine=1.0)
        P.save()
        P.clip(wing)
        P.fill(wing, CRIMSON, glow=0.1, fine=1.0)
        for k in range(4):
            v = ws * (-0.3 + 0.2 * k)
            P.stroke(lambda ctx, v=v: (ctx.move_to(*pt(Ls * 0.1, v)), ctx.line_to(*pt(Ls * 0.95, v * 0.4))),
                     S["gold_l"], max(1.6, sc * 7), gold=1.0, fine=1.0)
        P.restore()
        P.stroke(wing, S["gold"], max(1.5, sc * 6), gold=1.0, fine=1.0)
        for u in (0.45, 0.72):
            ex, ey = pt(Ls * u, ws * 0.08)
            eye(P, ex, ey, max(9.0, sc * 40), open=wake, rot=a, glow=glow, lid_gold=True)
    for k, (r_out, r_in, ne, ew) in enumerate(RINGS):
        rm = sc * 0.5 * (r_out + r_in)
        bw = sc * (r_out - r_in)
        P.stroke(lambda ctx, rm=rm: (ctx.new_sub_path(), ctx.arc(cx, cy, rm, 0, 2 * math.pi)), INK, bw + 3,
                 fine=1.0)
        P.stroke(lambda ctx, rm=rm: (ctx.new_sub_path(), ctx.arc(cx, cy, rm, 0, 2 * math.pi)), S["gold"], bw - 1,
                 gold=1.0, fine=1.0)
        for i in range(ne):
            a = -math.pi / 2 + i * 2 * math.pi / ne
            eye(P, cx + rm * math.cos(a), cy + rm * math.sin(a), sc * ew * 0.95, open=wake,
                look=(0.0, 0.0), rot=a + math.pi / 2, glow=glow, setting=S["lapis_d"])
    R = sc * CORE_R
    P.fill(_ellipse(cx, cy, R + 3, R + 3), INK, fine=1.0)
    P.fill(_ellipse(cx, cy, R, R), S["gold"], gold=1.0, fine=1.0)
    P.fill(_ellipse(cx, cy, R * 0.82, R * 0.82), hexc("#1E3180"), glow=0.2, fine=1.0)
    eye(P, cx, cy, sc * GREAT_EYE_W, open=wake, look=(0.0, 0.9 * look_down), glow=glow + 0.2, lid_gold=True,
        setting=BLACKBLUE)

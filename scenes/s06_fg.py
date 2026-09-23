"""s06 helper: foreground of the ledge shots — the rubble ledge (made of the same fragments as the
tower), mid-ground ruins of the old memeplexes in fog, and the gale (rain, torn babble, scraps)."""
import math

import cairo
import numpy as np

from vx import W, H, hash01, lin_grad, rad_grad, rrect, mix

from . import s06_glyphs as G


def _jag(rng, x0, x1, y, n, amp):
    xs = np.linspace(x0, x1, n)
    ys = y + rng.normal(0, amp, n)
    return xs, ys


def ruins(ctx, T, flash, horizon, fog=(0.13, 0.11, 0.22), seed=3, scale=1.0, dx=0.0):
    """silhouettes of the broken memeplexes standing in the fog sea: a toppled marble head, a cracked dome,
    column stumps, a half-buried giant crown. Drawn in fog colour with cold rims."""
    rng = np.random.default_rng(seed)
    fr, fg_, fb = fog
    rim = (0.55, 0.6, 0.9, 0.35 + 0.5 * flash)
    items = [("dome", 250, 1.0), ("stump", 520, 0.7), ("head", 1540, 1.1), ("crown", 1790, 0.8),
             ("stump", 1300, 0.55), ("stump", 120, 0.5)]
    for kind, x, s in items:
        x = x * 1.0 + dx
        s *= scale
        k = 0.32 + 0.18 * hash01(int(x), 2)
        c = (fr * k, fg_ * k, fb * k * 1.1)
        ctx.save()
        ctx.translate(x, horizon)
        ctx.scale(s, s)
        ctx.new_path()
        if kind == "dome":
            ctx.move_to(-150, 0)
            ctx.line_to(-150, -70)
            ctx.arc(0, -70, 150, math.pi, 1.62 * math.pi)
            ctx.line_to(40, -150)
            ctx.line_to(20, -110)
            ctx.line_to(75, -120)
            ctx.line_to(150, -70)
            ctx.line_to(150, 0)
            ctx.close_path()
            for q in range(5):
                ctx.rectangle(-140 + q * 60, -70, 18, 70)
        elif kind == "stump":
            ctx.move_to(-28, 0)
            ctx.line_to(-28, -150)
            ctx.line_to(-8, -170)
            ctx.line_to(6, -150)
            ctx.line_to(28, -185)
            ctx.line_to(28, 0)
            ctx.close_path()
        elif kind == "head":
            # a colossal marble philosopher head lying on its side (no symbols)
            ctx.move_to(-190, 0)
            ctx.curve_to(-230, -80, -170, -210, -40, -230)
            ctx.curve_to(90, -240, 190, -160, 200, -60)
            ctx.line_to(170, -40)
            ctx.line_to(190, -10)
            ctx.line_to(160, 0)
            ctx.close_path()
        elif kind == "crown":
            pts = [(-130, 0), (-140, -120), (-80, -60), (-40, -160), (0, -70), (40, -150), (60, -95), (80, -40),
                   (130, 0)]
            ctx.move_to(*pts[0])
            for p in pts[1:]:
                ctx.line_to(*p)
            ctx.close_path()
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
        g = cairo.LinearGradient(0, -240, 0, 0)
        g.add_color_stop_rgb(0, c[0] * 1.6 + 0.05 * flash, c[1] * 1.6 + 0.05 * flash, c[2] * 1.5 + 0.08 * flash)
        g.add_color_stop_rgb(1, c[0] * 0.7, c[1] * 0.7, c[2] * 0.8)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_line_width(1.6 / s)
        ctx.set_source_rgba(rim[0], rim[1], rim[2], rim[3] * 0.5)
        ctx.stroke()
        ctx.restore()


def ledge(ctx, gctx, T, y, flash, seed=31, dark=False, props=True, x_lo=-60, x_hi=W + 60):
    """the rubble ledge: a thick broken slab, lit top face, dark front face, cracks and fragments"""
    rng = np.random.default_rng(seed)
    xs, ys = _jag(rng, x_lo, x_hi, y, 26, 7)
    ys += np.linspace(-10, 14, len(xs))
    # front face
    ctx.move_to(x_lo, H + 50)
    for a, b in zip(xs, ys):
        ctx.line_to(a, b + 26)
    ctx.line_to(x_hi, H + 50)
    ctx.close_path()
    base = (0.045, 0.042, 0.07) if not dark else (0.02, 0.02, 0.035)
    ctx.set_source(lin_grad(0, y, 0, H, [(0, mix(base, (0.2, 0.2, 0.32), 0.3 + 0.3 * flash)), (0.25, base),
                                         (1, (0.008, 0.008, 0.015))]))
    ctx.fill()
    # top face (thin lit band seen slightly from above)
    ctx.move_to(xs[0], ys[0])
    for a, b in zip(xs[1:], ys[1:]):
        ctx.line_to(a, b)
    for a, b in zip(xs[::-1], ys[::-1]):
        ctx.line_to(a, b + 26 + 4 * math.sin(a * 0.05))
    ctx.close_path()
    top = mix((0.14, 0.14, 0.22), (0.55, 0.58, 0.8), 0.15 + 0.6 * flash) if not dark else (0.05, 0.05, 0.08)
    ctx.set_source_rgb(*top)
    ctx.fill()
    ctx.set_line_width(2.0)
    ctx.move_to(xs[0], ys[0])
    for a, b in zip(xs[1:], ys[1:]):
        ctx.line_to(a, b)
    ctx.set_source_rgba(0.62, 0.68, 0.95, 0.35 + 0.55 * flash)
    ctx.stroke()
    # cracks and strata in the front face
    ctx.set_line_width(1.6)
    ctx.set_source_rgba(0.0, 0.0, 0.02, 0.8)
    for q in range(14):
        x = x_lo + (x_hi - x_lo) * hash01(q, seed)
        yy = y + 30
        ctx.move_to(x, yy)
        for _ in range(5):
            x += rng.normal(0, 18)
            yy += rng.uniform(20, 50)
            ctx.line_to(x, yy)
        ctx.stroke()
    if not props:
        return
    # fragments on the ledge: a fallen marble drum, a map shard, a cracked phone, a silver crown
    def drum(x, yb, s, rot):
        ctx.save()
        ctx.translate(x, yb)
        ctx.rotate(rot)
        ctx.scale(s, s)
        g = lin_grad(0, -70, 0, 0, [(0, (0.55, 0.55, 0.62)), (0.5, (0.72, 0.72, 0.78)), (1, (0.3, 0.3, 0.38))])
        rrect(ctx, -70, -70, 140, 70, 6)
        ctx.set_source(g)
        ctx.fill()
        ctx.set_source_rgba(0.25, 0.25, 0.32, 0.9)
        ctx.set_line_width(3)
        for i in range(6):
            ctx.move_to(-58 + i * 23, -66)
            ctx.line_to(-58 + i * 23, -4)
        ctx.stroke()
        ctx.restore()

    k = 0.35 + 0.65 * flash
    drum(1640, y + 14, 0.95, -0.04)
    # map shard slab leaning
    ctx.save()
    ctx.translate(120, y + 16)
    ctx.rotate(-0.35)
    ctx.move_to(0, 0)
    ctx.line_to(20, -120)
    ctx.line_to(120, -140)
    ctx.line_to(150, -30)
    ctx.close_path()
    ctx.set_source_rgb(0.16 + 0.2 * k, 0.26 + 0.2 * k, 0.28 + 0.2 * k)
    ctx.fill()
    ctx.set_source_rgba(0.6, 0.78, 0.9, 0.7)
    ctx.set_line_width(3)
    ctx.move_to(30, -100)
    ctx.curve_to(60, -80, 70, -60, 120, -50)
    ctx.stroke()
    ctx.restore()
    # cracked phone, still glowing, face up
    ctx.save()
    ctx.translate(820, y + 12)
    ctx.rotate(0.08)
    ctx.scale(1, 0.42)
    rrect(ctx, -44, -80, 88, 150, 12)
    ctx.set_source_rgb(0.05, 0.05, 0.07)
    ctx.fill()
    fl = 0.6 + 0.4 * math.sin(T * 23) ** 2
    rrect(ctx, -37, -72, 74, 134, 7)
    ctx.set_source_rgba(0.62 * fl, 0.86 * fl, 1.0 * fl, 1)
    ctx.fill()
    ctx.set_source_rgba(0.02, 0.02, 0.04, 1)
    ctx.set_line_width(2.4)
    ctx.move_to(-30, -60)
    ctx.line_to(4, -12)
    ctx.line_to(-10, 20)
    ctx.line_to(28, 55)
    ctx.move_to(4, -12)
    ctx.line_to(30, -30)
    ctx.stroke()
    ctx.restore()
    if gctx is not None:
        gctx.save()
        gctx.translate(820, y + 12)
        gctx.scale(1, 0.42)
        rrect(gctx, -37, -72, 74, 134, 7)
        gctx.set_source_rgba(0.5, 0.75, 1.0, 0.6 * fl)
        gctx.fill()
        gctx.restore()
    # a tiny silver crown, fallen
    ctx.save()
    ctx.translate(1020, y + 18)
    ctx.rotate(0.5)
    pts = [(-22, 0), (-24, -22), (-12, -10), (0, -28), (12, -10), (24, -22), (22, 0)]
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.set_source_rgb(0.5 + 0.3 * k, 0.52 + 0.3 * k, 0.6 + 0.3 * k)
    ctx.fill()
    ctx.restore()


def _hit(avoid, x0, y0, x1, y1):
    for (a, b, c, d) in avoid or ():
        if x1 > a and x0 < c and y1 > b and y0 < d:
            return True
    return False


def weather(ctx, gctx, T, k, atlas, seed=1, bubbles=True, depth_scale=1.0, avoid=None):
    """gale: slanted rain, wind-borne scraps and torn babble bubbles streaming left -> right"""
    fr = int(T * 24)
    rng = np.random.default_rng(fr * 7 + seed)
    n = int(170 * k)
    xs = rng.uniform(-200, W, n)
    ys = rng.uniform(-100, H, n)
    L = rng.uniform(40, 110, n)
    ctx.set_line_width(1.2)
    for x, y, l in zip(xs, ys, L):
        ctx.move_to(x, y)
        ctx.line_to(x + l, y + l * 0.3)
    ctx.set_source_rgba(0.72, 0.76, 0.95, 0.13)
    ctx.stroke()
    # scraps (paper, card shards) tumbling
    for j in range(int(26 * k)):
        sp = 700 + 900 * hash01(j, 11)
        x = (hash01(j, 12) * 2600 + T * sp) % 2600 - 350
        y = 60 + hash01(j, 13) * 950 + 70 * math.sin(T * 2.3 + j) + (T * 60 * hash01(j, 14)) % 200
        s = 4 + 12 * hash01(j, 15)
        a = T * (3 + 6 * hash01(j, 16)) + j
        if _hit(avoid, x - s, y - s, x + s, y + s):
            continue
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(a)
        ctx.scale(1, abs(math.cos(a * 1.3)) + 0.15)
        ctx.rectangle(-s, -s * 0.7, 2 * s, 1.4 * s)
        c = [(0.8, 0.8, 0.85), (0.72, 0.88, 0.91), (0.91, 0.78, 0.82), (0.45, 0.47, 0.55)][j % 4]
        ctx.set_source_rgba(*c, 0.75)
        ctx.fill()
        ctx.restore()
    if not bubbles or atlas is None:
        return
    nm = len(atlas["masks"])
    for j in range(int(10 * k)):
        sp = (900 + 700 * hash01(j, 1)) * depth_scale
        x = (hash01(j, 2) * 3000 + T * sp) % 3000 - 500
        y = 80 + hash01(j, 3) * 760 + 60 * math.sin(T * 2 + j)
        hb = (20 + 44 * hash01(j, 4)) * depth_scale
        gi = int(hash01(j + int(T * 1.5 + hash01(j, 9) * 5), 5) * nm) % nm
        m = atlas["masks"][gi]
        th = hb * 0.52
        tw_ = min(th * m.shape[1] / m.shape[0], hb * 5)
        th = tw_ * m.shape[0] / m.shape[1]
        bw = tw_ + hb * 0.5
        if _hit(avoid, x - bw * 0.6, y - bw * 0.35, x + bw * 0.6, y + bw * 0.35):
            continue
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(math.sin(T * 3 + j) * 0.35)
        sq = 1 - 0.25 * abs(math.sin(T * 5 + j))
        ctx.scale(1, sq)
        a = 0.5
        rrect(ctx, -bw / 2, -hb / 2, bw, hb, hb / 2)
        ctx.set_source_rgba(0.9, 0.9, 0.9, a)
        ctx.fill()
        G.draw_strip(ctx, atlas, gi, 0, 0, th, (0.08, 0.08, 0.12), a)
        ctx.restore()


def ground(ctx, gctx, T, y_far, flash, seed=8, n=190, dark=False):
    """the top of the rubble ledge seen from standing height: a plane from its broken far edge (y_far)
    down past the frame bottom, strewn with fragments of the old memeplexes (some phones still glowing)."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-80, W + 80, 40)
    ys = y_far + rng.normal(0, 4, len(xs)) + 6 * np.sin(xs * 0.01)
    ctx.move_to(-80, H + 80)
    for a, b in zip(xs, ys):
        ctx.line_to(a, b)
    ctx.line_to(W + 80, H + 80)
    ctx.close_path()
    if dark:
        ctx.set_source_rgb(0.018, 0.017, 0.03)
        ctx.fill()
    else:
        g = cairo.LinearGradient(0, y_far, 0, H)
        g.add_color_stop_rgb(0, 0.11 + 0.12 * flash, 0.10 + 0.12 * flash, 0.19 + 0.16 * flash)
        g.add_color_stop_rgb(0.35, 0.06, 0.055, 0.1)
        g.add_color_stop_rgb(1, 0.02, 0.018, 0.035)
        ctx.set_source(g)
        ctx.fill()
    # broken far edge catches the storm light
    ctx.set_line_width(2.0)
    ctx.move_to(xs[0], ys[0])
    for a, b in zip(xs[1:], ys[1:]):
        ctx.line_to(a, b)
    ctx.set_source_rgba(0.55, 0.6, 0.9, (0.22 + 0.5 * flash) * (0.4 if dark else 1.0))
    ctx.stroke()
    cols = [(0.20, 0.18, 0.26), (0.30, 0.30, 0.36), (0.16, 0.24, 0.26), (0.42, 0.41, 0.44), (0.26, 0.2, 0.26)]
    d = np.sort(rng.random(n)) ** 1.0
    for i in range(n):
        di = d[i]
        y = y_far + 8 + (H + 40 - y_far) * di ** 1.35
        x = rng.uniform(-60, W + 60)
        s = 5 + 46 * di ** 1.4
        c = np.array(cols[int(rng.integers(len(cols)))])
        fogk = (1 - di) * 0.55
        c = c * (1 - fogk) + np.array([0.12, 0.11, 0.2]) * fogk
        if dark:
            c = c * 0.25
        rot = rng.uniform(-0.6, 0.6)
        sq = 0.35 + 0.25 * rng.random()
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(rot * 0.3)
        ctx.scale(1, sq)
        kind = rng.random()
        if kind < 0.55:
            k = rng.integers(4, 7)
            ang = np.sort(rng.uniform(0, 2 * math.pi, k))
            rr = s * rng.uniform(0.6, 1.0, k)
            ctx.move_to(rr[0] * math.cos(ang[0]), rr[0] * math.sin(ang[0]))
            for a_, r_ in zip(ang[1:], rr[1:]):
                ctx.line_to(r_ * math.cos(a_), r_ * math.sin(a_))
            ctx.close_path()
        else:
            ctx.rectangle(-s, -s * 0.7, 2 * s, 1.4 * s)
        ctx.set_source_rgb(*c)
        ctx.fill()
        # lit top edge
        ctx.set_source_rgba(0.6, 0.64, 0.9, (0.18 + 0.4 * flash) * (0.3 if dark else 1.0))
        ctx.rectangle(-s * 0.8, -s * 0.72, 1.6 * s, max(1.0, s * 0.12))
        ctx.fill()
        ctx.restore()
        if not dark and rng.random() < 0.07:
            fl = 0.5 + 0.5 * math.sin(T * (9 + 7 * rng.random()) + i) ** 2
            w = s * 0.7
            ctx.save()
            ctx.translate(x + s * 0.2, y - s * 0.1)
            ctx.scale(1, sq)
            rrect(ctx, -w / 2, -w * 0.8, w, w * 1.6, w * 0.15)
            ctx.set_source_rgba(0.62 * fl, 0.86 * fl, 1.0 * fl, 0.9)
            ctx.fill()
            ctx.restore()
            if gctx is not None:
                gctx.set_source_rgba(0.5, 0.75, 1.0, 0.5 * fl)
                gctx.arc(x + s * 0.2, y - s * 0.1, w, 0, 2 * math.pi)
                gctx.fill()


def contact_shadow(ctx, x, y, w, a=0.55):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(1, 0.18)
    g = cairo.RadialGradient(0, 0, 0, 0, 0, w)
    g.add_color_stop_rgba(0, 0, 0, 0.02, a)
    g.add_color_stop_rgba(1, 0, 0, 0.02, 0)
    ctx.set_source(g)
    ctx.arc(0, 0, w, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()

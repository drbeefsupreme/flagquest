"""t08 camp inking kit: grey-world drawing of the rainy camp in Chick-tract style (sky, storm clouds,
lightning, pipe-frame shade structure with a sagging tarp, lantern, far tents, churned mud, puddles, ink
rain, splashes, drips, camp chair, the Amen sunbeam).  All functions draw in PANEL-LOCAL page units on a
ctx that already carries the page camera + panel translation.  Tones: 0 = solid ink, 1 = bare paper;
the print (vx.ink.tract) turns greys into halftone dots.  Owner: T08Blaspheme.

Rain / drips / splashes are pure functions of global time T; `stop` = global time the rain stops (drops
already falling finish their fall, no new ones spawn).
"""
import math

import cairo
import numpy as np

from vx.ease import clamp, smoothstep, hash01, noise1
from scenes.t02_brush import stroke, shape, catmull, path_curve

INK = (0.0, 0.0, 0.0)
PAPER = (1.0, 1.0, 1.0)


def g(v):
    return (v, v, v)


def fill(ctx, c, a=1.0):
    ctx.set_source_rgba(c[0], c[1], c[2], a)
    ctx.fill()


def curve_fill(ctx, pts, c, n=8, a=1.0, closed=True):
    ctx.new_path()
    path_curve(ctx, pts, closed, n)
    fill(ctx, c, a)


# ============================================================ sky
def sky(ctx, w, h, T, *, top=0.50, bottom=0.74, horizon=200.0, bw=0.0, flash=0.0):
    """storm sky gradient (top dark -> lighter at the horizon)."""
    if bw > 0.5:
        v = 1.0 if flash > 0.5 else 0.06
        ctx.rectangle(-10, -10, w + 20, horizon + 20)
        fill(ctx, g(v))
        return
    gr = cairo.LinearGradient(0, 0, 0, horizon)
    gr.add_color_stop_rgb(0.0, top, top, top)
    gr.add_color_stop_rgb(1.0, bottom, bottom, bottom)
    ctx.rectangle(-10, -10, w + 20, horizon + 20)
    ctx.set_source(gr)
    ctx.fill()


def _cloud_pts(cx, cy, rw, rh, seed, n=11):
    rng = np.random.default_rng(seed)
    pts = []
    for k in range(n):
        a = math.pi * (1 + k / (n - 1))            # upper half: pi .. 2pi
        r = 1.0 + 0.22 * rng.random()
        pts.append((cx + math.cos(a) * rw * r, cy + math.sin(a) * rh * (0.75 + 0.5 * rng.random())))
    # flat-ish underside
    pts.append((cx + rw * 0.85, cy + rh * 0.18))
    pts.append((cx, cy + rh * 0.28))
    pts.append((cx - rw * 0.85, cy + rh * 0.2))
    return pts


def cloud_bank(ctx, clouds, T, *, lw=1.6, bw=0.0, flash=0.0, part=0.0, sun=(0.0, 0.0), part_dist=600.0,
               alpha=1.0):
    """clouds = [(cx, cy, rw, rh, tone, seed, drift)]. part 0..1 pushes clouds radially away from `sun`
    (the Amen break). Billowy bumps on top, dark belly, brushed ink contour."""
    for (cx, cy, rw, rh, tone, seed, drift) in clouds:
        x = cx + drift * T
        y = cy
        if part > 0:
            dx, dy = x - sun[0], y - sun[1]
            d = math.hypot(dx, dy) + 1e-6
            push = part * part_dist * (1.2 - min(1.0, d / 900.0))
            x += dx / d * push
            y += dy / d * push * 0.6
        pts = _cloud_pts(x, y, rw, rh, seed)
        if bw > 0.5:
            body = g(0.96) if flash > 0.5 else g(0.03)
            rim = g(0.0) if flash > 0.5 else g(1.0)
        else:
            body = g(tone)
            rim = g(min(1.0, tone + 0.28))
        curve_fill(ctx, pts, body, 8, alpha)
        # belly shadow
        if bw <= 0.5:
            ctx.save()
            ctx.new_path()
            path_curve(ctx, pts, True, 8)
            ctx.clip()
            ctx.rectangle(x - rw * 1.5, y - rh * 0.05, rw * 3, rh * 1.5)
            fill(ctx, g(max(0.0, tone - 0.16)), alpha)
            ctx.restore()
        # sunlit / lightning-lit rim along the top bumps
        top = pts[1:10]
        stroke(ctx, [(px, py + rh * 0.06) for px, py in top], rh * 0.16, taper=(0.3, 0.3), color=rim,
               alpha=0.85 * alpha)
        stroke(ctx, pts[:11], lw * 1.2, taper=(0.2, 0.2), color=(INK if (bw <= 0.5 or flash > 0.5) else rim),
               alpha=alpha)


def make_clouds(w, y0, y1, n, seed, tone=(0.36, 0.5), size=(90, 170), drift=(2.0, 5.0)):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        rw = rng.uniform(*size)
        out.append((rng.uniform(-80, w + 80), rng.uniform(y0, y1), rw, rw * rng.uniform(0.4, 0.62),
                    rng.uniform(*tone), int(rng.integers(1 << 30)), rng.uniform(*drift)))
    out.sort(key=lambda c: c[1])
    return out


# ============================================================ lightning
def bolt_paths(x0, y0, x1, y1, seed, levels=6, rough=0.30, branches=3):
    """main bolt polyline + branch polylines (midpoint displacement)."""
    rng = np.random.default_rng(seed)

    def zig(a, b, lv, r):
        pts = [a, b]
        for _ in range(lv):
            out = [pts[0]]
            for p, q in zip(pts[:-1], pts[1:]):
                mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
                L = math.hypot(q[0] - p[0], q[1] - p[1])
                nx, ny = -(q[1] - p[1]) / (L + 1e-9), (q[0] - p[0]) / (L + 1e-9)
                o = rng.normal() * L * r
                out += [(mx + nx * o, my + ny * o), q]
            pts = out
            r *= 0.62
        return pts

    main = zig((x0, y0), (x1, y1), levels, rough)
    br = []
    for k in range(branches):
        i = int(len(main) * (0.25 + 0.5 * rng.random()))
        p = main[i]
        ang = math.atan2(y1 - y0, x1 - x0) + (0.5 + 0.45 * rng.random()) * (1 if rng.random() < 0.5 else -1)
        L = math.hypot(x1 - x0, y1 - y0) * (0.18 + 0.2 * rng.random())
        br.append(zig(p, (p[0] + math.cos(ang) * L, p[1] + math.sin(ang) * L), levels - 2, rough * 1.1))
    return main, br


def _ribbon(ctx, pts, w0, w1):
    P = np.asarray(pts, np.float64)
    d = np.gradient(P, axis=0)
    L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    N = np.stack([-d[:, 1] / L, d[:, 0] / L], 1)
    s = np.linspace(0, 1, len(P))[:, None]
    wd = (w0 + (w1 - w0) * s) * 0.5
    Lp, Rp = P + N * wd, P - N * wd
    ctx.new_path()
    ctx.move_to(*Lp[0])
    for p in Lp[1:]:
        ctx.line_to(*p)
    for p in Rp[::-1]:
        ctx.line_to(*p)
    ctx.close_path()


def draw_bolt(ctx, main, branches, *, core=2.6, edge=6.0, neg=False, alpha=1.0, grow=1.0, star=True):
    """lightning: a tapered white jagged core inside a black ink edge inside a white halo (the classic
    'electric' comic bolt); neg swaps ink and paper (the flash frames). Impact star where it lands."""
    cin, cout = (INK, PAPER) if neg else (PAPER, INK)
    n = max(2, int(len(main) * clamp(grow)))
    parts = [(main[:n], 1.0)] + [(b, 0.5) for b in branches if grow >= 1.0]
    for wk, c in ((2.1, cin), (1.45, cout), (0.62, cin)):
        for pts, k in parts:
            w0 = edge * k * wk
            _ribbon(ctx, pts, w0, w0 * 0.35)
            ctx.set_source_rgba(c[0], c[1], c[2], alpha)
            ctx.fill()
    if star and grow >= 1.0:
        x, y = main[-1]
        rng = np.random.default_rng(3)
        for wk, c in ((1.0, cout), (0.55, cin)):
            ctx.new_path()
            m = 14
            for j in range(m * 2):
                a = math.pi * j / m
                r = (edge * 3.2 if j % 2 == 0 else edge * 0.9) * (0.7 + 0.6 * rng.random()) * wk
                px, py = x + math.cos(a) * r, y + math.sin(a) * r * 0.5
                (ctx.move_to if j == 0 else ctx.line_to)(px, py)
            ctx.close_path()
            ctx.set_source_rgba(c[0], c[1], c[2], alpha)
            ctx.fill()


# ============================================================ camp furniture
def tents(ctx, w, horizon, seed, *, tone=0.34, bw=0.0, n=7, scale=1.0):
    """far camp: dome tents / shade structures / an RV silhouette along the horizon."""
    rng = np.random.default_rng(seed)
    c = g(0.03) if bw > 0.5 else g(tone)
    xs = np.sort(rng.uniform(-40, w + 40, n))
    for k, x in enumerate(xs):
        s = scale * rng.uniform(0.7, 1.3)
        kind = rng.integers(0, 3)
        ctx.new_path()
        if kind == 0:      # dome tent
            ctx.save()
            ctx.translate(x, horizon + 2)
            ctx.scale(34 * s, 18 * s)
            ctx.arc(0, 0, 1, math.pi, 2 * math.pi)
            ctx.restore()
        elif kind == 1:    # shade structure (flat roof on poles)
            ww = 60 * s
            ctx.rectangle(x - ww / 2, horizon - 26 * s, ww, 6 * s)
            for px in (x - ww / 2 + 2, x + ww / 2 - 4):
                ctx.rectangle(px, horizon - 26 * s, 2.2 * s, 26 * s)
        else:              # RV
            ctx.move_to(x - 40 * s, horizon + 1)
            ctx.line_to(x - 40 * s, horizon - 20 * s)
            ctx.line_to(x + 28 * s, horizon - 20 * s)
            ctx.line_to(x + 40 * s, horizon - 10 * s)
            ctx.line_to(x + 40 * s, horizon + 1)
            ctx.close_path()
        fill(ctx, c)
    # horizon band
    ctx.rectangle(-10, horizon - 1.5, w + 20, 5)
    fill(ctx, c)


def ground(ctx, w, h, horizon, seed, T, *, near=0.56, far=0.72, bw=0.0, ruts=40, lw=1.4):
    """churned mud: tone gradient + brushed ruts/boot prints (ink strokes)."""
    if bw > 0.5:
        ctx.rectangle(-10, horizon, w + 20, h - horizon + 20)
        fill(ctx, g(0.04))
    else:
        gr = cairo.LinearGradient(0, horizon, 0, h)
        gr.add_color_stop_rgb(0.0, far, far, far)
        gr.add_color_stop_rgb(1.0, near, near, near)
        ctx.rectangle(-10, horizon, w + 20, h - horizon + 20)
        ctx.set_source(gr)
        ctx.fill()
    rng = np.random.default_rng(seed)
    col = PAPER if bw > 0.5 else INK
    for k in range(ruts):
        y = horizon + (h - horizon) * rng.random() ** 0.7
        depth = (y - horizon) / max(1.0, h - horizon)
        x = rng.uniform(-20, w + 20)
        L = (12 + 60 * rng.random()) * (0.3 + depth)
        cv = rng.normal() * 0.15
        pts = [(x - L / 2, y), (x, y + cv * L - 1.5 * depth), (x + L / 2, y + 0.4 * rng.normal())]
        stroke(ctx, pts, lw * (0.4 + 1.6 * depth), taper=(0.45, 0.45), n=4, color=col, alpha=0.9)
        if rng.random() < 0.35:     # a lighter (wet) ridge beside it
            stroke(ctx, [(px, py - 2.2 * (0.3 + depth)) for px, py in pts], lw * (0.5 + 1.2 * depth),
                   taper=(0.5, 0.5), n=4, color=g(0.9 if bw <= 0.5 else 0.04), alpha=0.8)


def puddles(ctx, items, T, *, bw=0.0, stop=1e9, sky_tone=0.8):
    """items = [(x, y, rx, ry, seed)]: reflective puddles (sky-coloured) with rain ripples."""
    for (x, y, rx, ry, seed) in items:
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(rx, ry)
        ctx.new_path()
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        fill(ctx, g(0.02) if bw > 0.5 else g(sky_tone))
        # dark near rim
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(rx, ry)
        ctx.new_path()
        ctx.arc(0, 0, 1, 0.1 * math.pi, 0.9 * math.pi)
        ctx.restore()
        ctx.set_line_width(1.6)
        ctx.set_source_rgb(*(PAPER if bw > 0.5 else INK))
        ctx.stroke()
        # ripples: ring k born at t_k, expands for 0.7 s
        for k in range(5):
            per = 0.55 + 0.5 * hash01(seed * 13 + k, 3)
            ph = hash01(seed * 7 + k, 5)
            c = math.floor(T / per + ph)
            born = (c - ph) * per
            if born > stop:
                c -= 1
                born = (c - ph) * per
            age = T - born
            if age < 0 or age > 0.7:
                continue
            u = age / 0.7
            ox = (hash01(int(c) * 31 + k, seed) - 0.5) * 1.2 * rx
            oy = (hash01(int(c) * 17 + k, seed + 1) - 0.5) * 1.0 * ry
            rr = 2 + u * min(rx, 40) * 0.5
            if abs(ox) + rr > rx * 0.95:
                ox *= 0.4
            ctx.save()
            ctx.translate(x + ox, y + oy * 0.6)
            ctx.scale(rr, rr * ry / rx)
            ctx.new_path()
            ctx.arc(0, 0, 1, 0, 2 * math.pi)
            ctx.restore()
            ctx.set_line_width(1.0)
            ctx.set_source_rgba(*(PAPER if bw > 0.5 else INK), 1 - u)
            ctx.stroke()


def shade_structure(ctx, x0, x1, top, ground_y, T, *, depth=40.0, sag=26.0, bw=0.0, lw=2.2, tarp=0.86,
                    posts="all"):
    """pipe-frame shade: four posts, a sagging tarp with a water belly. posts: 'all' | 'back' | 'front'."""
    pc = PAPER if bw > 0.5 else INK
    mid = (x0 + x1) / 2
    fy = top + depth * 0.5
    if posts in ("all", "back"):
        for px in (x0 + depth * 0.8, x1 - depth * 0.2):
            ctx.new_path()
            ctx.move_to(px, top - depth * 0.1)
            ctx.line_to(px, ground_y - depth * 0.9)
            ctx.set_line_width(lw * 0.8)
            ctx.set_source_rgb(*pc)
            ctx.stroke()
    if posts in ("all", "back", "tarp"):
        # tarp underside + front edge (sag)
        back = [(x0 + depth * 0.8, top - depth * 0.1), (mid + depth * 0.3, top + sag * 0.3), (x1 - depth * 0.2, top - depth * 0.1)]
        front = [(x1, fy), (mid, fy + sag), (x0, fy)]
        pts = back + front
        tc = g(0.03) if bw > 0.5 else g(tarp)
        curve_fill(ctx, pts, tc, 8)
        # tarp folds / underside shading
        ctx.save()
        ctx.new_path()
        path_curve(ctx, pts, True, 8)
        ctx.clip()
        for k in range(6):
            u = (k + 0.5) / 6
            xa = x0 + (x1 - x0) * u
            stroke(ctx, [(xa + 8, top), (xa + (mid - xa) * 0.15, fy + sag * (1 - abs(2 * u - 1)) * 0.9)],
                   lw * 0.7, taper=(0.3, 0.6), n=3, color=(PAPER if bw > 0.5 else g(0.3)), alpha=0.7)
        ctx.rectangle(x0 - 10, fy + sag * 0.35, x1 - x0 + 20, sag * 2)
        fill(ctx, g(0.03) if bw > 0.5 else g(tarp - 0.22))
        ctx.restore()
        stroke(ctx, front, lw * 1.3, taper=(0.05, 0.05), n=10, color=pc)
        stroke(ctx, back, lw * 0.7, taper=(0.1, 0.1), n=10, color=pc)
    if posts in ("all", "front"):
        for px in (x0, x1):
            ctx.new_path()
            ctx.move_to(px, fy)
            ctx.line_to(px, ground_y)
            ctx.set_line_width(lw * 1.2)
            ctx.set_source_rgb(*pc)
            ctx.stroke()
    return (mid, fy + sag)             # the sag low point (water drips from here)


def lantern(ctx, x, y, s, T, *, bw=0.0, glow=1.0):
    """hanging camp lantern at (x, y) = cord top; s = size. Swings a little in the wind."""
    ang = 0.08 * math.sin(T * 1.7) + 0.04 * noise1(T * 0.9, 4)
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    pc = PAPER if bw > 0.5 else INK
    ctx.new_path()
    ctx.move_to(0, 0)
    ctx.line_to(0, s * 1.2)
    ctx.set_line_width(1.0)
    ctx.set_source_rgb(*pc)
    ctx.stroke()
    if glow > 0 and bw <= 0.5:
        gr = cairo.RadialGradient(0, s * 2.0, 0, 0, s * 2.0, s * 4.5)
        gr.add_color_stop_rgba(0, 1, 1, 1, 0.9 * glow)
        gr.add_color_stop_rgba(1, 1, 1, 1, 0.0)
        ctx.set_source(gr)
        ctx.arc(0, s * 2.0, s * 4.5, 0, 2 * math.pi)
        ctx.fill()
    # cap, glass, base
    ctx.rectangle(-s * 0.45, s * 1.2, s * 0.9, s * 0.25)
    fill(ctx, INK if bw <= 0.5 else PAPER)
    ctx.rectangle(-s * 0.38, s * 1.45, s * 0.76, s * 1.1)
    fill(ctx, PAPER if bw <= 0.5 else INK)
    ctx.rectangle(-s * 0.38, s * 1.45, s * 0.76, s * 1.1)
    ctx.set_line_width(1.1)
    ctx.set_source_rgb(*pc)
    ctx.stroke()
    ctx.rectangle(-s * 0.5, s * 2.55, s * 1.0, s * 0.3)
    fill(ctx, INK if bw <= 0.5 else PAPER)
    ctx.restore()


def camp_chair(ctx, x, y, s, *, facing=1, bw=0.0, lw=1.6, tone=0.42):
    """folding camp chair; (x, y) ground point under the seat front, s = seat height (PU)."""
    pc = PAPER if bw > 0.5 else INK
    f = facing
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(f, 1)
    seat_y = -s
    # back legs + backrest
    frame = [((-0.55 * s, 0), (0.25 * s, seat_y)), ((0.15 * s, 0), (-0.6 * s, seat_y)),
             ((-0.6 * s, seat_y), (-0.75 * s, seat_y - 1.05 * s)), ((0.3 * s, seat_y), (0.4 * s, seat_y - 0.35 * s))]
    back = [(-0.6 * s, seat_y + 0.05 * s), (-0.78 * s, seat_y - 1.0 * s), (-0.45 * s, seat_y - 1.05 * s),
            (-0.28 * s, seat_y - 0.05 * s)]
    ctx.new_path()
    ctx.move_to(*back[0])
    for p in back[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    fill(ctx, g(0.03) if bw > 0.5 else g(tone))
    seat = [(-0.62 * s, seat_y), (0.3 * s, seat_y - 0.02 * s), (0.25 * s, seat_y + 0.16 * s), (-0.55 * s, seat_y + 0.2 * s)]
    ctx.new_path()
    ctx.move_to(*seat[0])
    for p in seat[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    fill(ctx, g(0.03) if bw > 0.5 else g(tone - 0.12))
    for a, b in frame:
        ctx.new_path()
        ctx.move_to(*a)
        ctx.line_to(*b)
        ctx.set_line_width(lw)
        ctx.set_source_rgb(*pc)
        ctx.stroke()
    # armrest
    stroke(ctx, [(-0.62 * s, seat_y - 0.42 * s), (0.35 * s, seat_y - 0.38 * s)], lw * 1.6, taper=(0.05, 0.05), n=2,
           color=pc)
    ctx.restore()


# ============================================================ weather
class Rain:
    """deterministic ink rain for a panel (local units). slant = dx/dy (+ falls down-right)."""

    def __init__(self, w, h, n, seed, *, slant=0.22, speed=(520.0, 760.0), length=(10.0, 26.0),
                 width=(0.7, 1.5), white=0.5):
        rng = np.random.default_rng(seed)
        self.w, self.h, self.slant = w, h, slant
        self.x0 = rng.uniform(-slant * h - 30, w + 30, n)
        self.ph = rng.random(n)
        self.v = rng.uniform(*speed, n)
        self.L = rng.uniform(*length, n)
        self.wd = rng.uniform(*width, n)
        self.white = rng.random(n) < white
        self.span = h + 2 * self.L + 40
        self.per = self.span / self.v

    def draw(self, ctx, T, *, stop=1e9, amount=1.0, bw=0.0, frozen=False):
        if amount <= 0:
            return
        c = T / self.per + self.ph
        cyc = np.floor(c)
        u = c - cyc
        born = (cyc - self.ph) * self.per
        alive = (born <= stop) & (np.arange(len(u)) < int(len(u) * amount))
        y = -self.L - 20 + u * self.span
        x = self.x0 + self.slant * y
        for white in (False, True):
            m = alive & (self.white == white)
            if not m.any():
                continue
            col = PAPER if (white or bw > 0.5) else INK
            if bw > 0.5 and not white:
                continue
            ctx.set_source_rgba(col[0], col[1], col[2], 0.95)
            for xi, yi, Li, wi in zip(x[m], y[m], self.L[m], self.wd[m]):
                ctx.move_to(xi, yi)
                ctx.line_to(xi + self.slant * Li, yi + Li)
                ctx.set_line_width(wi)
                ctx.stroke()


class Splashes:
    """rain hitting the mud: tiny ink crowns at random ground points."""

    def __init__(self, w, y0, y1, n, seed, *, size=(3.0, 7.0)):
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(0, w, n)
        self.y = rng.uniform(y0, y1, n)
        self.per = rng.uniform(0.35, 0.8, n)
        self.ph = rng.random(n)
        self.s = rng.uniform(*size, n) * (0.5 + (self.y - y0) / max(1.0, y1 - y0))
        self.seed = seed

    def draw(self, ctx, T, *, stop=1e9, bw=0.0):
        c = T / self.per + self.ph
        cyc = np.floor(c)
        born = (cyc - self.ph) * self.per
        age = T - born
        m = (age < 0.14) & (born <= stop)
        col = PAPER if bw > 0.5 else INK
        ctx.set_source_rgb(*col)
        for i in np.nonzero(m)[0]:
            u = age[i] / 0.14
            s = self.s[i] * (0.5 + u)
            x = self.x[i] + (hash01(int(cyc[i]) * 7 + i, self.seed) - 0.5) * 30
            y = self.y[i]
            ctx.set_line_width(max(0.6, 1.1 * (1 - u)))
            for k in (-2, -1, 0, 1, 2):
                a = -math.pi / 2 + k * 0.42
                ctx.move_to(x + math.cos(a) * s * 0.4, y + math.sin(a) * s * 0.3)
                ctx.line_to(x + math.cos(a) * s, y + math.sin(a) * s * 0.8)
            ctx.stroke()


def drips(ctx, points, ground_y, T, seed, *, bw=0.0, size=2.2):
    """water drops falling from fixed edge points (tarp edge, cap brim): [(x, y)]"""
    col = PAPER if bw > 0.5 else INK
    for k, (x, y) in enumerate(points):
        per = 0.55 + 0.6 * hash01(k, seed)
        ph = hash01(k, seed + 1)
        u = (T / per + ph) % 1.0
        fall = u * per                       # seconds since release
        grow = clamp(fall / 0.25)
        if fall < 0.25:                       # drop swelling on the edge
            r = size * (0.4 + 0.6 * grow)
            ctx.save()
            ctx.translate(x, y + r)
            ctx.new_path()
            ctx.move_to(0, -r * 1.6)
            ctx.curve_to(r * 0.9, -r * 0.2, r * 0.9, r, 0, r)
            ctx.curve_to(-r * 0.9, r, -r * 0.9, -r * 0.2, 0, -r * 1.6)
            ctx.restore()
            fill(ctx, PAPER if bw > 0.5 else g(0.95))
            ctx.set_line_width(0.8)
            ctx.set_source_rgb(*col)
            ctx.save()
            ctx.translate(x, y + r)
            ctx.new_path()
            ctx.move_to(0, -r * 1.6)
            ctx.curve_to(r * 0.9, -r * 0.2, r * 0.9, r, 0, r)
            ctx.curve_to(-r * 0.9, r, -r * 0.9, -r * 0.2, 0, -r * 1.6)
            ctx.restore()
            ctx.stroke()
        else:
            tf = fall - 0.25
            yy = y + size + 0.5 * 900.0 * tf * tf
            if yy < ground_y:
                ctx.new_path()
                ctx.move_to(x, yy - 5 - 30 * tf)
                ctx.line_to(x, yy)
                ctx.set_line_width(size * 0.8)
                ctx.set_source_rgb(*col)
                ctx.stroke()


# ============================================================ the Amen light
def glory(ctx, sx, sy, w, h, amount, T, *, rays=64, reach=1400.0):
    """Chick-tract heavenly light: alternating pale wedges radiating from (sx, sy) (halftone rays)."""
    if amount <= 0:
        return
    for k in range(rays):
        a0 = math.pi * (0.02 + 0.5 * k / rays) + 0.004 * math.sin(T * 0.7 + k)
        a1 = a0 + math.pi * 0.5 / rays * 0.5
        ctx.move_to(sx, sy)
        ctx.line_to(sx + math.cos(a0) * reach, sy + math.sin(a0) * reach)
        ctx.line_to(sx + math.cos(a1) * reach, sy + math.sin(a1) * reach)
        ctx.close_path()
    ctx.set_source_rgba(0.0, 0.0, 0.0, 0.10 * amount)
    ctx.fill()


def beam(ctx, sx, sy, tx, ty, half, amount, *, strength=0.75):
    """a soft wedge of light from (sx, sy) toward (tx, ty) (half-angle rad). Lightens (dots shrink)."""
    if amount <= 0:
        return
    ang = math.atan2(ty - sy, tx - sx)
    L = math.hypot(tx - sx, ty - sy) * 1.6
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_SCREEN)
    for k, (hw, a) in enumerate(((half * 1.35, 0.35), (half, 0.55), (half * 0.6, 0.6))):
        ctx.new_path()
        ctx.move_to(sx, sy)
        ctx.line_to(sx + math.cos(ang - hw) * L, sy + math.sin(ang - hw) * L)
        ctx.line_to(sx + math.cos(ang + hw) * L, sy + math.sin(ang + hw) * L)
        ctx.close_path()
        gr = cairo.RadialGradient(sx, sy, 0, sx, sy, L)
        gr.add_color_stop_rgba(0.0, 1, 1, 1, strength * a * amount)
        gr.add_color_stop_rgba(0.75, 1, 1, 1, strength * a * amount * 0.8)
        gr.add_color_stop_rgba(1.0, 1, 1, 1, 0.0)
        ctx.set_source(gr)
        ctx.fill()
    ctx.restore()


def gloom(ctx, w, h, k):
    """darken everything (multiply) by k (1 = none)."""
    if k >= 0.999:
        return
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_MULTIPLY)
    ctx.rectangle(-20, -20, w + 40, h + 40)
    ctx.set_source_rgb(k, k, k)
    ctx.fill()
    ctx.restore()


# ============================================================ storm ceiling (v2)
class Ceiling:
    """a Chick-tract storm ceiling: rows of billowing puffs (grey, darker bellies, inked scallops) filling
    the sky above y_base. .draw(..., part=0..1, sun=(x, y)) opens a clearing around the sun (the Amen)."""

    def __init__(self, w, y_base, seed, *, rows=4, dy=34.0, r=(34.0, 62.0), tone=(0.50, 0.66), drift=4.0):
        rng = np.random.default_rng(seed)
        self.puffs = []
        for row in range(rows):
            y = y_base - (rows - 1 - row) * dy
            x = -80.0 - rng.uniform(0, 60)
            while x < w + 120:
                big = rng.random() < 0.18
                rr = rng.uniform(*r) * (0.8 + 0.12 * row) * (1.45 if big else 1.0)
                t = rng.uniform(*tone) + 0.05 * row
                a0 = math.radians(rng.uniform(8, 35))
                a1 = math.radians(rng.uniform(140, 172))
                self.puffs.append((x, y + rng.normal() * dy * 0.45 + (rr * 0.25 if big else 0), rr, t, row,
                                   int(rng.integers(1 << 30)), a0, a1))
                x += rr * rng.uniform(0.9, 1.6)
        self.w, self.y_base, self.drift, self.rows, self.dy = w, y_base, drift, rows, dy
        self.top_tone = tone[0] - 0.04

    def _pos(self, p, T, part, sun, reach):
        x, y, r, t, row, seed = p[:6]
        x = x + self.drift * T * (0.6 + 0.2 * row)
        span = self.w + 240
        x = (x + 120) % span - 120
        if part > 0:
            # the clouds part like curtains: pushed sideways away from the sun's column, lifting a little
            dx = x - sun[0]
            sgn = 1.0 if dx >= 0 else -1.0
            push = part * max(0.0, reach * 1.25 - abs(dx)) * 1.3
            x += sgn * push
            y -= part * 18.0
            r *= 1.0 - 0.15 * part
        return x, y, r, t, row

    def draw(self, ctx, T, *, part=0.0, sun=(0.0, -200.0), reach=520.0, bw=0.0, flash=0.0, lw=1.5, lit=0.0,
             bolt_x=None):
        # solid backing above the first row (so the sky never shows through the back rows); the Amen
        # clearing opens as a widening band over the sun's column (the curtains part)
        ctx.save()
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        yb = self.y_base - (self.rows - 1) * self.dy
        ctx.rectangle(-20, -400, self.w + 40, 400 + yb)
        if part > 0:
            hw = reach * part * 1.1
            ctx.rectangle(sun[0] - hw, -401, 2 * hw, 402 + yb)
        fill(ctx, g(0.03) if (bw > 0.5 and flash <= 0.5) else (g(0.97) if bw > 0.5 else g(self.top_tone)))
        ctx.restore()
        for p in self.puffs:
            x, y, r, t, row = self._pos(p, T, part, sun, reach)
            a0, a1 = p[6], p[7]
            if bw > 0.5:
                body, belly, edge = (g(0.97), g(0.97), INK) if flash > 0.5 else (g(0.03), g(0.03), PAPER)
            else:
                body, belly, edge = g(t), g(max(0.0, t - 0.16)), INK
            ctx.new_path()
            ctx.arc(x, y, r, 0, 2 * math.pi)
            fill(ctx, body)
            if bw <= 0.5:
                ctx.save()
                ctx.new_path()
                ctx.arc(x, y, r, 0, 2 * math.pi)
                ctx.clip()
                ctx.new_path()
                ctx.arc(x - r * 0.18, y - r * 0.32, r * 0.95, 0, 2 * math.pi)
                ctx.rectangle(x - r * 2, y - r * 2, r * 4, r * 4)
                ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
                fill(ctx, belly)
                ctx.restore()
                if r > 44:      # a few brushed belly strokes (Chick-style form lines)
                    for j in range(3):
                        aa = math.radians(40 + 35 * j)
                        stroke(ctx, [(x + math.cos(aa) * r * 0.55, y + math.sin(aa) * r * 0.45),
                                     (x + math.cos(aa + 0.35) * r * 0.82, y + math.sin(aa + 0.35) * r * 0.72)],
                               lw * 0.9, taper=(0.4, 0.6), n=2, color=INK, alpha=0.9)
                if lit > 0:     # sun-lit upper rims near the clearing
                    dx, dy = x - sun[0], y - sun[1]
                    near = clamp(1.0 - math.hypot(dx, dy) / (reach * 1.8))
                    if near > 0:
                        stroke(ctx, [(x - r * 0.8, y - r * 0.45), (x - r * 0.3, y - r * 0.9), (x + r * 0.3, y - r * 0.92)],
                               r * 0.22, taper=(0.3, 0.5), n=4, color=PAPER, alpha=lit * near)
            # inked scallop on the lower half (the billow edge); at night only the bolt-lit rims show
            ea = 1.0
            if bw > 0.5 and flash <= 0.5:
                ea = 0.0 if bolt_x is None else clamp(1.0 - abs(x - bolt_x) / 240.0) ** 0.7
                if ea <= 0.02:
                    continue
            ctx.new_path()
            ctx.arc(x, y, r, a0, a1)
            ctx.set_line_width(lw * (1.0 + 0.3 * row))
            ctx.set_source_rgba(edge[0], edge[1], edge[2], ea)
            ctx.stroke()


def glory_lines(ctx, sx, sy, amount, *, n=90, r0=40.0, r1=1500.0, a0=0.0, a1=math.pi, lw=1.2, seed=5,
                clip_r=None):
    """Chick heavenly light: thin ink rays radiating from (sx, sy) across [a0, a1] (y down).
    clip_r: only inside that radius around the source (the cloud clearing)."""
    if amount <= 0:
        return
    rng = np.random.default_rng(seed)
    ctx.save()
    if clip_r is not None:
        ctx.new_path()
        ctx.arc(sx, sy, max(1.0, clip_r), 0, 2 * math.pi)
        ctx.clip()
    ctx.set_source_rgba(0, 0, 0, clamp(amount))
    for k in range(n):
        a = a0 + (a1 - a0) * (k + 0.5 * rng.random()) / n
        ra = r0 + rng.random() * 60
        rb = r0 + (r1 - r0) * amount * (0.55 + 0.45 * rng.random())
        ctx.move_to(sx + math.cos(a) * ra, sy + math.sin(a) * ra)
        ctx.line_to(sx + math.cos(a) * rb, sy + math.sin(a) * rb)
        ctx.set_line_width(lw * (0.6 + 0.8 * rng.random()))
        ctx.stroke()
    ctx.restore()


def spotlight(ctx, w, h, sx, sy, tx, ty, half, amount, *, dim=0.80, reach=1.0, edge_lines=True):
    """the Amen beam: everything OUTSIDE a wedge from (sx, sy) toward (tx, ty) is dimmed (multiply `dim`),
    the wedge itself is lifted toward paper white; inked edge lines bound it (Chick-tract light)."""
    if amount <= 0:
        return
    ang = math.atan2(ty - sy, tx - sx)
    L = math.hypot(tx - sx, ty - sy) * 1.5 * reach
    p0 = (sx, sy)
    pa = (sx + math.cos(ang - half) * L, sy + math.sin(ang - half) * L)
    pb = (sx + math.cos(ang + half) * L, sy + math.sin(ang + half) * L)
    # dim outside
    ctx.save()
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.rectangle(-50, -50, w + 100, h + 100)
    ctx.move_to(*p0)
    ctx.line_to(*pa)
    ctx.line_to(*pb)
    ctx.close_path()
    ctx.set_operator(cairo.OPERATOR_MULTIPLY)
    k = 1.0 - (1.0 - dim) * amount
    ctx.set_source_rgb(k, k, k)
    ctx.fill()
    ctx.restore()
    # lift inside (screen white)
    ctx.save()
    ctx.move_to(*p0)
    ctx.line_to(*pa)
    ctx.line_to(*pb)
    ctx.close_path()
    ctx.set_operator(cairo.OPERATOR_SCREEN)
    gr = cairo.LinearGradient(sx, sy, tx, ty)
    gr.add_color_stop_rgba(0.0, 1, 1, 1, 0.85 * amount)
    gr.add_color_stop_rgba(1.0, 1, 1, 1, 0.55 * amount)
    ctx.set_source(gr)
    ctx.fill()
    ctx.restore()
    if edge_lines:
        for s in (-1, 1):
            for j, off in enumerate((0.0, 0.012, 0.026)):
                a = ang + s * (half - off)
                ctx.move_to(sx + math.cos(a) * 60, sy + math.sin(a) * 60)
                ctx.line_to(sx + math.cos(a) * L * 0.8, sy + math.sin(a) * L * 0.8)
                ctx.set_line_width(2.0 - 0.5 * j)
                ctx.set_source_rgba(0, 0, 0, 1.0 if amount > 0.3 else 0.0)
                ctx.stroke()

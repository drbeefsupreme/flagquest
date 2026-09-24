"""The Seraphlag's vast wings and great halo (t05). The body is RigMaster's "seraph" (pose "fly", wings=False);
the wings are drawn here so their beat can be timed to the sound (the 'wing' hits) and so they are VAST:
each wing spans ~1.6 body heights: primaries, secondaries and three rows of coverts, every feather an inked
quill (Doré's angels; the tract's page 26 panel 1)."""
import math

import cairo
import numpy as np


def _lerp2(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _bez(p0, p1, p2, t):
    a = (1 - t) ** 2
    b = 2 * (1 - t) * t
    c = t * t
    return (a * p0[0] + b * p1[0] + c * p2[0], a * p0[1] + b * p1[1] + c * p2[1])


# the wing in its own frame (right wing, units of wing length): x out along the leading edge, y down (trailing)
_ARM = [(0.0, 0.0), (0.3, -0.07), (0.56, -0.1), (0.78, -0.06), (1.0, 0.02)]      # leading edge / bones


def _arm_pt(t, fold):
    """point on the leading edge, t 0..1 (root..tip); fold < 1 tucks the hand in"""
    pts = [(x * (1 if i < 3 else fold), y) for i, (x, y) in enumerate(_ARM)]
    seg = t * (len(pts) - 1)
    i = min(int(seg), len(pts) - 2)
    return _lerp2(pts[i], pts[i + 1], seg - i)


def _trail_pt(t, fold):
    """trailing edge (feather tips): secondaries' scalloped edge then the primaries sweeping to the tip"""
    if t < 0.55:
        u = t / 0.55
        return (0.02 + 0.52 * u, 0.3 + 0.05 * math.sin(math.pi * u))
    u = (t - 0.55) / 0.45
    p = _bez((0.54, 0.31), (1.0 * fold + 0.1, 0.34), (1.28 * fold + 0.02, 0.02), u)
    return p


def _feather(ctx, x0, y0, x1, y1, width, tone, ink_w, barbs=2):
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) + 1e-9
    nx, ny = -dy / L, dx / L
    w = width * L
    ctx.move_to(x0, y0)
    ctx.curve_to(x0 + dx * 0.25 + nx * w, y0 + dy * 0.25 + ny * w, x0 + dx * 0.8 + nx * w * 0.9,
                 y0 + dy * 0.8 + ny * w * 0.9, x1, y1)
    ctx.curve_to(x0 + dx * 0.8 - nx * w * 0.6, y0 + dy * 0.8 - ny * w * 0.6, x0 + dx * 0.25 - nx * w * 0.6,
                 y0 + dy * 0.25 - ny * w * 0.6, x0, y0)
    ctx.close_path()
    ctx.set_source_rgb(tone, tone, tone)
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(ink_w)
    ctx.stroke()
    ctx.move_to(x0, y0)
    ctx.line_to(x0 + dx * 0.93 + nx * w * 0.1, y0 + dy * 0.93 + ny * w * 0.1)
    ctx.set_line_width(ink_w * 0.7)
    ctx.stroke()
    if barbs:
        ctx.set_line_width(ink_w * 0.45)
        for k in range(barbs):
            f = 0.4 + 0.25 * k
            px, py = x0 + dx * f, y0 + dy * f
            ctx.move_to(px, py)
            ctx.line_to(px - nx * w * 0.55 + dx * 0.08, py - ny * w * 0.55 + dy * 0.08)
        ctx.stroke()


def draw_wing(ctx, sx, sy, h, side, phase, spread=1.0, tone=0.78, alpha=1.0, scale=1.0, far=False):
    """one wing. (sx, sy) root (upper back) in design px, h = body height px, side = +1 right / -1 left.
    phase: 0 = wings up, 0.5 = bottom of the downstroke."""
    c = math.cos(2 * math.pi * phase)
    down = math.sin(2 * math.pi * phase) > 0          # in the downstroke
    elev = math.radians(44 + 30 * c) * (0.35 + 0.65 * spread)
    fold = (0.9 + 0.1 * (0.5 - 0.5 * c)) if down else (0.78 + 0.22 * (0.5 - 0.5 * c))
    # the hand lags the stroke: bend the outer wing against the motion
    bend = -0.22 * math.sin(2 * math.pi * phase)
    k = h * 1.3 * scale
    ctx.save()
    ctx.translate(sx, sy)
    ctx.scale(side, 1)
    ctx.rotate(-elev)
    ctx.scale(k, k * (0.92 if far else 1.0))
    iw = (1.25 if not far else 1.0) / k
    if alpha < 1:
        ctx.push_group()

    def arm(t):
        x, y = _arm_pt(t, fold)
        if t > 0.55:                 # bend the hand
            u = (t - 0.55) / 0.45
            a = bend * u
            ox, oy = _arm_pt(0.55, fold)
            dx, dy = x - ox, y - oy
            x, y = ox + dx * math.cos(a) - dy * math.sin(a), oy + dx * math.sin(a) + dy * math.cos(a)
        return x, y

    def trail(t):
        x, y = _trail_pt(t, fold)
        if t > 0.5:
            u = (t - 0.5) / 0.5
            a = bend * u * 1.2
            ox, oy = _arm_pt(0.55, fold)
            dx, dy = x - ox, y - oy
            x, y = ox + dx * math.cos(a) - dy * math.sin(a), oy + dx * math.sin(a) + dy * math.cos(a)
        return x, y

    tb = tone - (0.08 if far else 0.0)
    # primaries (from the hand to the sweeping tip), drawn outermost first
    n = 11
    for i in range(n - 1, -1, -1):
        f = i / (n - 1)
        b = arm(0.56 + 0.42 * f)
        e = trail(0.55 + 0.45 * f)
        _feather(ctx, b[0], b[1], e[0], e[1], 0.1, tb - 0.16 - 0.05 * ((i * 7) % 3), iw)
    # secondaries along the forearm
    n = 13
    for i in range(n):
        f = i / (n - 1)
        b = arm(0.08 + 0.5 * f)
        e = trail(0.02 + 0.53 * f)
        _feather(ctx, b[0], b[1], e[0] - 0.02, e[1], 0.13, tb - 0.1 - 0.04 * ((i * 5) % 3), iw)
    # coverts: three overlapping rows, shorter towards the leading edge
    for row, (depth, wd, dt) in enumerate(((0.62, 0.17, 0.0), (0.38, 0.2, 0.05), (0.2, 0.24, 0.1))):
        n = 17 - row * 2
        for i in range(n):
            f = i / (n - 1)
            t = 0.03 + 0.8 * f * (1.0 - 0.1 * row)
            b = arm(t)
            e0 = trail(min(1.0, t * 0.95))
            e = (b[0] + (e0[0] - b[0]) * depth - 0.015, b[1] + (e0[1] - b[1]) * depth)
            _feather(ctx, b[0], b[1], e[0], e[1], wd, min(0.97, tb + dt), iw * 0.9, barbs=1)
    # the leading edge: a strong inked arm with a highlight
    pts = [arm(t) for t in np.linspace(0, 0.98, 14)]
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.set_line_width(iw * 4.0)
    ctx.set_source_rgb(0, 0, 0)
    ctx.stroke()
    ctx.move_to(*pts[0])
    for p in pts[1:9]:
        ctx.line_to(*p)
    ctx.set_line_width(iw * 1.4)
    ctx.set_source_rgb(0.98, 0.98, 0.98)
    ctx.stroke()
    if alpha < 1:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(alpha)
    ctx.restore()


def draw_wings(ctx, anchors, h, phase, spread=1.0, alpha=1.0, facing=-1):
    """both wings rooted on the upper back, the far one first"""
    nx, ny = anchors["neck"]
    sl, sr = anchors["shoulder_l"], anchors["shoulder_r"]
    y = ny + 0.06 * h
    off = 0.045 * h
    # the far wing (behind the body in a 3/4 view) is a touch smaller
    far_side = 1 if facing < 0 else -1
    draw_wing(ctx, nx + far_side * off * 0.6, y - 0.01 * h, h, far_side, (phase + 0.02) % 1.0, spread,
              alpha=alpha, scale=0.9, far=True)
    draw_wing(ctx, nx - far_side * off, y, h, -far_side, phase, spread, alpha=alpha)


def draw_halo(ctx, x, y, r, t=0.0, alpha=1.0):
    """the great halo: a disc of engraved light (concentric rings + rays), behind the head"""
    ctx.save()
    g = cairo.RadialGradient(x, y, r * 0.2, x, y, r * 1.6)
    g.add_color_stop_rgba(0, 1, 1, 1, 0.95 * alpha)
    g.add_color_stop_rgba(0.55, 0.97, 0.97, 0.97, 0.7 * alpha)
    g.add_color_stop_rgba(1, 0.9, 0.9, 0.9, 0.0)
    ctx.set_source(g)
    ctx.arc(x, y, r * 1.6, 0, 2 * math.pi)
    ctx.fill()
    ctx.set_source_rgba(0, 0, 0, alpha)
    for k, (rr, lw) in enumerate(((1.0, 2.2), (0.9, 1.0), (0.78, 0.8))):
        ctx.set_line_width(lw)
        ctx.arc(x, y, r * rr, 0, 2 * math.pi)
        ctx.stroke()
    ctx.set_line_width(0.9)
    n = 48
    for k in range(n):
        a = 2 * math.pi * k / n + 0.05 * t
        r0 = r * 1.06
        r1 = r * (1.35 + 0.22 * (k % 2))
        ctx.move_to(x + math.cos(a) * r0, y + math.sin(a) * r0)
        ctx.line_to(x + math.cos(a) * r1, y + math.sin(a) * r1)
    ctx.stroke()
    ctx.restore()

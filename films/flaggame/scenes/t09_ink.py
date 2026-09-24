"""t09 inking helpers: tapered brush strokes along Catmull-Rom curves and brushed shapes (grey fills + pure
black contours: vx.ink keeps pure black as crisp solid line art and screens the greys).  Owner: T09PassItOn."""
import math

import cairo
import numpy as np

INK = (0.0, 0.0, 0.0)


def g(v):
    return (v, v, v)


def catmull(pts, closed=False, n=8):
    P = np.asarray(pts, np.float64)
    if len(P) < 3:
        return P
    if closed:
        P = np.vstack([P[-1:], P, P[:2]])
    else:
        P = np.vstack([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    p0, p1, p2, p3 = P[:-3], P[1:-2], P[2:-1], P[3:]
    seg = 0.5 * ((2 * p1[:, None]) + (-p0 + p2)[:, None] * t + (2 * p0 - 5 * p1 + 4 * p2 - p3)[:, None] * t2
                 + (-p0 + 3 * p1 - 3 * p2 + p3)[:, None] * t3)
    out = seg.reshape(-1, 2)
    if not closed:
        out = np.vstack([out, P[-2:-1]])
    return out


def path(ctx, pts, closed=False, n=8):
    S = catmull(pts, closed, n)
    ctx.move_to(*S[0])
    for p in S[1:]:
        ctx.line_to(*p)
    if closed:
        ctx.close_path()


def stroke(ctx, pts, w, taper=(0.35, 0.35), color=INK, alpha=1.0, n=8, wmin=0.06, press=None):
    """tapered brush stroke of max width w through pts"""
    S = catmull(pts, False, n)
    if len(S) < 2:
        return
    d = np.gradient(S, axis=0)
    L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    N = np.stack([-d[:, 1] / L, d[:, 0] / L], 1)
    seg = np.hypot(*np.diff(S, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = s / max(s[-1], 1e-9)
    wid = np.full(len(S), float(w))
    a, b = taper
    if a > 0:
        x = np.clip(u / a, 0, 1)
        wid *= wmin + (1 - wmin) * (x * x * (3 - 2 * x))
    if b > 0:
        x = np.clip((1 - u) / b, 0, 1)
        wid *= wmin + (1 - wmin) * (x * x * (3 - 2 * x))
    if press is not None:
        wid *= np.array([press(x) for x in u])
    Lp = S + N * wid[:, None] * 0.5
    Rp = S - N * wid[:, None] * 0.5
    ctx.save()
    ctx.set_source_rgba(color[0], color[1], color[2], alpha)
    ctx.new_path()
    ctx.move_to(*Lp[0])
    for p in Lp[1:]:
        ctx.line_to(*p)
    for p in Rp[::-1]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.fill()
    ctx.restore()


def shape(ctx, pts, fill=None, line=0.0, n=8, alpha=1.0, closed=True):
    """smooth filled closed shape with a solid black contour of width `line`"""
    ctx.save()
    ctx.new_path()
    path(ctx, pts, closed, n)
    if fill is not None:
        ctx.set_source_rgba(fill[0], fill[1], fill[2], alpha)
        if line > 0:
            ctx.fill_preserve()
        else:
            ctx.fill()
    if line > 0:
        ctx.set_source_rgba(0, 0, 0, alpha)
        ctx.set_line_width(line)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.stroke()
    ctx.restore()


def poly(ctx, pts, fill=None, line=0.0, alpha=1.0):
    ctx.save()
    ctx.new_path()
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    if fill is not None:
        ctx.set_source_rgba(fill[0], fill[1], fill[2], alpha)
        if line > 0:
            ctx.fill_preserve()
        else:
            ctx.fill()
    if line > 0:
        ctx.set_source_rgba(0, 0, 0, alpha)
        ctx.set_line_width(line)
        ctx.stroke()
    ctx.restore()


def hatch(ctx, clip_pts, x0, y0, x1, y1, ang, gap, w, color=INK, closed=True):
    """parallel ink hatching inside a closed smooth region"""
    ctx.save()
    ctx.new_path()
    path(ctx, clip_pts, closed)
    ctx.clip()
    ca, sa = math.cos(ang), math.sin(ang)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    R = math.hypot(x1 - x0, y1 - y0) / 2 + gap
    ctx.set_source_rgb(*color)
    ctx.set_line_width(w)
    k = -R
    while k <= R:
        px, py = cx - sa * k, cy + ca * k
        ctx.move_to(px - ca * R, py - sa * R)
        ctx.line_to(px + ca * R, py + sa * R)
        k += gap
    ctx.stroke()
    ctx.restore()


def lerp(a, b, t):
    return a + (b - a) * t


def lerp2(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

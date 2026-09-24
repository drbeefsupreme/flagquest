"""t02 inking toolkit: a comic brush (variable-width tapered strokes along Catmull-Rom curves), filled
shapes with brushed contours, hatching and feathering.  Pure cairo drawing in whatever units the ctx is in.
Owner: T02Wrong."""
import math

import cairo
import numpy as np

INK = (0.0, 0.0, 0.0)
WHITE = (1.0, 1.0, 1.0)


def grey(v):
    return (v, v, v)


# ------------------------------------------------------------------ curves
def catmull(pts, closed=False, n=10):
    """Catmull-Rom (centripetal-ish uniform) resample of control points -> (m, 2) array"""
    P = np.asarray(pts, np.float64)
    if len(P) < 2:
        return P
    if closed:
        P = np.vstack([P[-1:], P, P[:2]])
    else:
        P = np.vstack([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    p0, p1, p2, p3 = P[:-3], P[1:-2], P[2:-1], P[3:]
    segs = 0.5 * ((2 * p1[:, None]) + (-p0 + p2)[:, None] * t + (2 * p0 - 5 * p1 + 4 * p2 - p3)[:, None] * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3)[:, None] * t3)
    out = segs.reshape(-1, 2)
    if not closed:
        out = np.vstack([out, P[-2:-1]])
    return out


def path_curve(ctx, pts, closed=False, n=10):
    S = catmull(pts, closed, n)
    ctx.move_to(*S[0])
    for p in S[1:]:
        ctx.line_to(*p)
    if closed:
        ctx.close_path()


def _normals(S, closed):
    if closed:
        d = np.roll(S, -1, 0) - np.roll(S, 1, 0)
    else:
        d = np.gradient(S, axis=0)
    L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    T = d / L[:, None]
    return np.stack([-T[:, 1], T[:, 0]], 1), T


def stroke(ctx, pts, w, *, taper=(0.3, 0.3), closed=False, n=10, light=None, lk=0.6, press=None,
           color=INK, alpha=1.0, wmin=0.08):
    """Brush stroke. w = max width. taper = fraction of length tapering in/out.
    light = (lx, ly) direction TO the light: the stroke thickens on the side facing away (lk strength).
    press = callable(s 0..1) -> multiplier (pressure profile)."""
    S = catmull(pts, closed, n)
    if len(S) < 2:
        return
    N, T = _normals(S, closed)
    seglen = np.hypot(*np.diff(S, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seglen)])
    total = s[-1] if s[-1] > 0 else 1.0
    u = s / total
    wid = np.full(len(S), float(w))
    if not closed:
        a, b = taper
        if a > 0:
            x = np.clip(u / a, 0, 1)
            wid *= wmin + (1 - wmin) * (x * x * (3 - 2 * x))
        if b > 0:
            x = np.clip((1 - u) / b, 0, 1)
            wid *= wmin + (1 - wmin) * (x * x * (3 - 2 * x))
    if press is not None:
        wid *= np.array([press(x) for x in u])
    if light is not None:
        lx, ly = light
        ln = math.hypot(lx, ly) or 1
        d = (N[:, 0] * lx + N[:, 1] * ly) / ln          # +1 normal faces the light
        wid *= np.clip(1 - lk * d, 0.15, 2.5)
    L = S + N * (wid[:, None] * 0.5)
    R = S - N * (wid[:, None] * 0.5)
    ctx.save()
    ctx.set_source_rgba(color[0], color[1], color[2], alpha)
    ctx.new_path()
    if closed:
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.move_to(*L[0])
        for p in L[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.move_to(*R[0])
        for p in R[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.fill()
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
        # the even-odd ring can leave slivers where the stroke folds: paint the centre line too
        ctx.set_line_width(float(np.min(wid)) * 0.9)
        path_curve(ctx, pts, True, n)
        ctx.stroke()
    else:
        ctx.move_to(*L[0])
        for p in L[1:]:
            ctx.line_to(*p)
        for p in R[::-1]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.fill()
        # round caps for blunt ends
        for i in (0, -1):
            if wid[i] > 0.6:
                ctx.arc(S[i][0], S[i][1], wid[i] * 0.5, 0, 2 * math.pi)
                ctx.fill()
    ctx.restore()


def shape(ctx, pts, fill=None, *, line=0.0, closed=True, n=10, light=None, lk=0.6, alpha=1.0,
          line_color=INK):
    """filled smooth closed shape with a brushed contour"""
    ctx.save()
    if fill is not None:
        ctx.new_path()
        path_curve(ctx, pts, closed, n)
        ctx.set_source_rgba(fill[0], fill[1], fill[2], alpha)
        ctx.fill()
    if line > 0:
        stroke(ctx, pts, line, closed=closed, n=n, light=light, lk=lk, taper=(0, 0) if closed else (0.25, 0.25),
               color=line_color, alpha=alpha)
    ctx.restore()


def poly_fill(ctx, pts, fill, alpha=1.0):
    ctx.new_path()
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.set_source_rgba(fill[0], fill[1], fill[2], alpha)
    ctx.fill()


def clip_curve(ctx, pts, n=10):
    ctx.new_path()
    path_curve(ctx, pts, True, n)
    ctx.clip()


def hatch(ctx, bbox, angle, spacing, w, *, seed=0, jitter=0.35, taper=(0.35, 0.35), color=INK, alpha=1.0,
          frac=1.0, lenvar=0.25):
    """parallel hand-inked hatch strokes filling bbox (clip first). frac: fraction of lines drawn."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    R = math.hypot(x1 - x0, y1 - y0) / 2 + spacing
    ca, sa = math.cos(angle), math.sin(angle)
    rng = np.random.default_rng(seed)
    k = -R
    while k <= R:
        if rng.random() <= frac:
            o = k + (rng.random() - 0.5) * spacing * jitter
            l0 = -R * (1 - lenvar * rng.random())
            l1 = R * (1 - lenvar * rng.random())
            px, py = cx - sa * o, cy + ca * o
            p0 = (px + ca * l0, py + sa * l0)
            p1 = (px + ca * l1, py + sa * l1)
            stroke(ctx, [p0, p1], w, taper=taper, n=2, color=color, alpha=alpha)
        k += spacing


def feather(ctx, pts, length, spacing, w, *, side=1, seed=0, color=INK, alpha=1.0, n=10, var=0.35,
            density=None):
    """short tapered strokes hanging off a curve on one side (form shading along an edge).
    density(u) -> 0..1 optional length multiplier along the curve"""
    S = catmull(pts, False, n)
    N, T = _normals(S, False)
    seglen = np.hypot(*np.diff(S, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seglen)])
    total = s[-1]
    rng = np.random.default_rng(seed)
    d = 0.0
    while d < total:
        i = int(np.searchsorted(s, d))
        i = min(i, len(S) - 1)
        u = d / total
        m = density(u) if density else 1.0
        if m > 0.05:
            L = length * m * (1 - var * rng.random())
            p = S[i]
            nn = N[i] * side
            tt = T[i]
            lean = (rng.random() - 0.5) * 0.3
            q = p + (nn + tt * lean) * L
            stroke(ctx, [p, (p + q) / 2 + tt * L * 0.05, q], w, taper=(0.05, 0.9), n=4, color=color, alpha=alpha)
        d += spacing * (0.8 + 0.4 * rng.random())


def dots(ctx, pts, r, color=INK, alpha=1.0):
    ctx.save()
    ctx.set_source_rgba(color[0], color[1], color[2], alpha)
    for x, y in pts:
        ctx.new_sub_path()
        ctx.arc(x, y, r, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()


def lock(ctx, pts, w, *, fill=WHITE, line=0.02, shadow_side=1, tip=0.9, root=0.25, n=10, dark=None,
         alpha=1.0):
    """a lock of hair: a tapered ribbon filled `fill`, inked heavier on its shadow side.
    dark = grey fill for a shadow core running along the shadow side (None = none)."""
    S = catmull(pts, False, n)
    if len(S) < 2:
        return
    N, T = _normals(S, False)
    seglen = np.hypot(*np.diff(S, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seglen)])
    u = s / max(s[-1], 1e-9)
    wid = w * np.clip(np.minimum((u + 0.02) / root if root > 0 else 1.0, 1.0), 0.3, 1.0) * \
        np.clip((1 - u) / tip, 0.0, 1.0) ** 0.8
    L = S + N * (wid[:, None] * 0.5)
    R = S - N * (wid[:, None] * 0.5)
    ctx.save()
    ctx.new_path()
    ctx.move_to(*L[0])
    for p in L[1:]:
        ctx.line_to(*p)
    for p in R[::-1]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.set_source_rgba(fill[0], fill[1], fill[2], alpha)
    ctx.fill()
    if dark is not None:
        E = S + N * (wid[:, None] * 0.5 * shadow_side)
        C = S + N * (wid[:, None] * 0.05 * shadow_side)
        ctx.new_path()
        ctx.move_to(*E[0])
        for p in E[1:]:
            ctx.line_to(*p)
        for p in C[::-1]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.set_source_rgba(dark[0], dark[1], dark[2], alpha)
        ctx.fill()
    ctx.restore()
    side = L if shadow_side > 0 else R
    other = R if shadow_side > 0 else L
    if line > 0:
        stroke(ctx, side[::3].tolist() + [side[-1].tolist()], line * 1.3, taper=(0.1, 0.6), n=3, alpha=alpha)
        stroke(ctx, other[::3].tolist() + [other[-1].tolist()], line * 0.55, taper=(0.3, 0.7), n=3, alpha=alpha)

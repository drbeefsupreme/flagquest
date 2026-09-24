"""t01 — hand-drawn tract busts of SUMMER and RAVEN (close-up panel art in brush-ink).  Owner: T01Tract.

Drawn in GREY for vx.ink.tract: pure black = brush ink (contours, Raven's hair), light greys = tone
(screened into dots), pure white = paper highlights.  Any ctx transform works (page units in t01).

summer_bust(ctx, x, y, s, *, facing=-1, mouth=0.0, expr="smile", look=(0, 0), blink=0.0, tilt=0.0, t=0.0,
            wind=0.0, hands=None)
    (x, y) = centre of the head, s = head height (crown -> chin).  facing -1 = looking screen-left (3/4).
    expr: smile | proud | laugh | shock.  mouth 0..1 lip-sync.  look = eye direction (-1..1).
raven_bust(ctx, x, y, s, *, facing=1, mouth=0.0, expr="disdain", t=0.0, wind=0.0)
    back-3/4 "lost profile": a great mass of glossy black hair, hoop earring, the edge of the face
    (lashes, lips) toward `facing`.
mouth_point(x, y, s, facing, who) -> (mx, my)   where balloon tails aim
"""
import math

import cairo

from vx import set_color

INK = (0.0, 0.0, 0.0)
PAPER = (1.0, 1.0, 1.0)
SKIN = (0.985, 0.985, 0.985)
SKIN_SH = (0.82, 0.82, 0.82)
HAIR_SH = (0.84, 0.84, 0.84)


# ------------------------------------------------------------------------------------------ brush toolkit
def _cr(pts, n=10, closed=False):
    """Catmull-Rom sample of a polyline"""
    P = list(pts)
    m = len(P)
    if m < 3:
        return P
    out = []
    rng = range(m) if closed else range(m - 1)
    for i in rng:
        p0 = P[(i - 1) % m] if closed else P[max(i - 1, 0)]
        p1 = P[i % m]
        p2 = P[(i + 1) % m] if closed else P[min(i + 1, m - 1)]
        p3 = P[(i + 2) % m] if closed else P[min(i + 2, m - 1)]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    if not closed:
        out.append(P[-1])
    return out


def brush(ctx, pts, w, *, w1=None, taper=(0.35, 0.35), color=INK, alpha=1.0, n=8):
    """variable-width brush stroke along a smooth curve (filled ribbon, tapered ends)"""
    q = _cr(pts, n)
    if len(q) < 2:
        return
    w1 = w if w1 is None else w1
    L = [0.0]
    for a, b in zip(q[:-1], q[1:]):
        L.append(L[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    tot = L[-1] or 1.0
    left, right = [], []
    for i, p in enumerate(q):
        a = q[max(i - 1, 0)]
        b = q[min(i + 1, len(q) - 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / d, dx / d
        u = L[i] / tot
        ww = w + (w1 - w) * u
        if taper[0] > 0 and u < taper[0]:
            ww *= math.sin(0.5 * math.pi * u / taper[0]) * 0.85 + 0.15
        if taper[1] > 0 and u > 1 - taper[1]:
            ww *= math.sin(0.5 * math.pi * (1 - u) / taper[1]) * 0.85 + 0.15
        left.append((p[0] + nx * ww / 2, p[1] + ny * ww / 2))
        right.append((p[0] - nx * ww / 2, p[1] - ny * ww / 2))
    ctx.new_path()
    ctx.move_to(*left[0])
    for p in left[1:]:
        ctx.line_to(*p)
    for p in reversed(right):
        ctx.line_to(*p)
    ctx.close_path()
    set_color(ctx, color, alpha)
    ctx.fill()


def shape(ctx, pts, fill=None, line=None, lw=1.0, closed=True, n=8):
    q = _cr(pts, n, closed=closed)
    ctx.new_path()
    ctx.move_to(*q[0])
    for p in q[1:]:
        ctx.line_to(*p)
    if closed:
        ctx.close_path()
    if fill is not None:
        set_color(ctx, fill)
        if line is not None:
            ctx.fill_preserve()
        else:
            ctx.fill()
    if line is not None:
        set_color(ctx, line)
        ctx.set_line_width(lw)
        ctx.stroke()


def _path(ctx, pts, closed=True, n=8):
    q = _cr(pts, n, closed=closed)
    ctx.move_to(*q[0])
    for p in q[1:]:
        ctx.line_to(*p)
    if closed:
        ctx.close_path()


def _dot(ctx, x, y, r, color=INK):
    ctx.new_path()
    ctx.arc(x, y, r, 0, 2 * math.pi)
    set_color(ctx, color)
    ctx.fill()


class _Xf:
    """local head units -> ctx: centre (x, y), size s, mirror by facing (drawings are authored facing LEFT)"""

    def __init__(self, x, y, s, facing, tilt=0.0):
        self.x, self.y, self.s, self.f, self.tilt = x, y, s, (1 if facing < 0 else -1), tilt
        self.c, self.sn = math.cos(tilt), math.sin(tilt)

    def __call__(self, p):
        px, py = p[0] * self.f, p[1]
        return (self.x + (px * self.c - py * self.sn) * self.s, self.y + (px * self.sn + py * self.c) * self.s)

    def pts(self, ps):
        return [self(p) for p in ps]

    def w(self, v):
        return v * self.s


# ------------------------------------------------------------------------------------------ SUMMER
def mouth_point(x, y, s, facing=-1, who="summer", tilt=0.0):
    X = _Xf(x, y, s, facing, tilt)
    if who == "summer":
        return X((-0.21, 0.27))
    return X((-0.40, 0.25))


def summer_bust(ctx, x, y, s, *, facing=-1, mouth=0.0, expr="smile", look=(0.0, 0.0), blink=0.0, tilt=0.0,
                t=0.0, wind=0.0, body=True):
    X = _Xf(x, y, s, facing, tilt)
    W_ = X.w
    lw = W_(0.018)
    sway = 0.012 * math.sin(t * 1.3) + wind * 0.03
    laugh = expr == "laugh"
    shock = expr == "shock"
    ctx.save()
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    # ---------------------------------------------------------------- body: jacket, fur collar, goggles
    if body:
        jacket = [(-1.00, 1.75), (-0.88, 0.95), (-0.55, 0.74), (-0.05, 0.68), (0.45, 0.70), (0.86, 0.86),
                  (1.08, 1.25), (1.16, 1.75)]
        shape(ctx, X.pts(jacket), fill=(0.50, 0.50, 0.50), line=INK, lw=lw * 1.3)
        shape(ctx, X.pts([(0.45, 0.74), (0.86, 0.88), (1.08, 1.25), (1.16, 1.75), (0.60, 1.75), (0.50, 1.15)]),
              fill=(0.30, 0.30, 0.30))
        brush(ctx, X.pts([(-0.60, 1.00), (-0.56, 1.35), (-0.58, 1.72)]), W_(0.022))
        brush(ctx, X.pts([(0.66, 0.98), (0.74, 1.35), (0.78, 1.72)]), W_(0.022))
        brush(ctx, X.pts([(0.06, 1.05), (0.09, 1.40), (0.08, 1.75)]), W_(0.016))
    # neck (behind collar)
    neck = [(-0.13, 0.36), (-0.15, 0.55), (-0.14, 0.72), (0.24, 0.72), (0.26, 0.50), (0.24, 0.24)]
    shape(ctx, X.pts(neck), fill=SKIN)
    shape(ctx, X.pts([(0.04, 0.42), (0.24, 0.26), (0.26, 0.52), (0.24, 0.72), (0.04, 0.72)]), fill=SKIN_SH)
    brush(ctx, X.pts([(-0.13, 0.38), (-0.15, 0.55), (-0.14, 0.70)]), lw * 1.1)
    brush(ctx, X.pts([(0.24, 0.26), (0.26, 0.50), (0.24, 0.70)]), lw * 1.2)
    if body:
        # faux-fur collar: a fluffy ring around the neck, ragged edge + fur flicks
        n = 30
        outer, inner = [], []
        for k in range(n + 1):
            u = k / n
            ang = math.pi * (-0.06 + 1.12 * u)
            jig = 0.035 * math.sin(k * 2.9) + 0.03 * (((k * 7919) % 7) / 7 - 0.5)
            outer.append((0.06 - math.cos(ang) * (0.74 + jig), 0.70 - math.sin(ang) * (0.24 + jig * 0.6)))
            inner.append((0.06 - math.cos(ang) * 0.34, 0.68 - math.sin(ang) * 0.07))
        pts = X.pts(outer) + X.pts(list(reversed(inner)))
        ctx.new_path()
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        set_color(ctx, PAPER)
        ctx.fill_preserve()
        set_color(ctx, INK)
        ctx.set_line_width(lw * 1.1)
        ctx.stroke()
        for k in range(34):
            u = (k + 0.5) / 34
            ang = math.pi * (-0.02 + 1.04 * u)
            r0 = 0.40 + 0.28 * (((k * 37) % 11) / 11)
            bx = 0.06 - math.cos(ang) * r0
            by = 0.70 - math.sin(ang) * (0.09 + 0.15 * r0)
            brush(ctx, X.pts([(bx, by), (bx - 0.035 * math.cos(ang + 0.4), by + 0.05)]), lw * 0.75, taper=(0.0, 0.8))
        # goggles on their strap, resting on the collar
        brush(ctx, X.pts([(-0.34, 0.78), (-0.22, 0.90), (-0.02, 0.95)]), W_(0.032))
        brush(ctx, X.pts([(0.48, 0.80), (0.40, 0.92), (0.26, 0.96)]), W_(0.032))
        for gx in (0.02, 0.24):
            gp = X((gx, 0.97))
            r = W_(0.105)
            ctx.new_path()
            ctx.arc(gp[0], gp[1], r, 0, 2 * math.pi)
            set_color(ctx, (0.62, 0.62, 0.62))
            ctx.fill_preserve()
            set_color(ctx, INK)
            ctx.set_line_width(W_(0.038))
            ctx.stroke()
            hl = X((gx - 0.03, 0.93))
            brush(ctx, [(hl[0] - W_(0.03), hl[1] + W_(0.025)), (hl[0] + W_(0.015), hl[1] - W_(0.02))], W_(0.024),
                  color=PAPER)
        brush(ctx, X.pts([(0.11, 0.97), (0.15, 0.97)]), W_(0.035))
    # ---------------------------------------------------------------- far pigtail (behind the head)
    pig_far = [(-0.22, 0.16), (-0.38, 0.36), (-0.44, 0.60), (-0.40, 0.84), (-0.50, 0.98)]
    _pigtail(ctx, X, pig_far, 0.13, lw, t, wind, phase=1.3)
    # ---------------------------------------------------------------- hair back mass
    back = [(-0.34, -0.36), (-0.22, -0.58), (0.06, -0.69), (0.36, -0.60), (0.54, -0.34), (0.56, -0.02),
            (0.48, 0.20), (0.30, 0.22), (0.14, 0.02)]
    shape(ctx, X.pts(back), fill=PAPER, line=INK, lw=lw * 1.2)
    # shadow under the hair mass (nape)
    shape(ctx, X.pts([(0.50, -0.14), (0.56, -0.02), (0.48, 0.20), (0.30, 0.22), (0.34, 0.04)]), fill=HAIR_SH)
    # ---------------------------------------------------------------- face
    tilt_j = mouth * 0.05 + (0.06 if laugh else 0.0)
    face = [(-0.30, -0.34), (-0.37, -0.16), (-0.36, -0.05), (-0.40, 0.06), (-0.37, 0.20 + tilt_j * 0.5),
            (-0.30, 0.36 + tilt_j), (-0.18, 0.45 + tilt_j), (0.02, 0.40 + tilt_j * 0.6), (0.16, 0.28), (0.22, 0.10),
            (0.20, -0.10), (0.05, -0.30)]
    shape(ctx, X.pts(face), fill=SKIN)
    # form shadow on the far side of the face (light from upper left in screen space)
    shade_pts = [(0.02, 0.40 + tilt_j * 0.6), (0.16, 0.28), (0.22, 0.10), (0.18, -0.06), (0.10, 0.10), (0.02, 0.28)]
    if X.f < 0:
        shade_pts = [(-0.18, 0.45 + tilt_j), (-0.30, 0.36 + tilt_j), (-0.37, 0.20), (-0.30, 0.18), (-0.20, 0.36)]
    shape(ctx, X.pts(shade_pts), fill=SKIN_SH)
    # contour (heavier on the jaw)
    brush(ctx, X.pts([(-0.37, -0.16), (-0.36, -0.05), (-0.40, 0.06), (-0.37, 0.20 + tilt_j * 0.5),
                      (-0.30, 0.36 + tilt_j), (-0.18, 0.45 + tilt_j)]), lw * 1.2, w1=lw * 1.8, taper=(0.3, 0.1))
    brush(ctx, X.pts([(-0.18, 0.45 + tilt_j), (0.02, 0.40 + tilt_j * 0.6), (0.16, 0.28), (0.22, 0.12)]), lw * 1.9,
          w1=lw * 1.0, taper=(0.05, 0.4))
    # ear
    ear = [(0.14, -0.02), (0.22, -0.07), (0.30, -0.02), (0.30, 0.10), (0.24, 0.20), (0.16, 0.18)]
    shape(ctx, X.pts(ear), fill=SKIN, line=INK, lw=lw)
    brush(ctx, X.pts([(0.24, 0.0), (0.21, 0.07), (0.23, 0.13)]), lw * 0.8)
    # ---------------------------------------------------------------- features
    lx, ly = look
    # brows
    bl = -0.03 if shock else 0.0
    brush(ctx, X.pts([(-0.17, -0.12 + bl), (-0.06, -0.165 + bl * 1.5), (0.06, -0.14 + bl)]), W_(0.028), taper=(0.2, 0.6))
    brush(ctx, X.pts([(-0.36, -0.13 + bl), (-0.30, -0.16 + bl * 1.5), (-0.24, -0.15 + bl)]), W_(0.022), taper=(0.5, 0.3))
    # eyes
    if laugh:
        for (cx, w_) in ((-0.05, 0.16), (-0.29, 0.09)):
            brush(ctx, X.pts([(cx - w_ / 2, -0.02), (cx, -0.06), (cx + w_ / 2, -0.02)]), W_(0.03), taper=(0.3, 0.3))
            brush(ctx, X.pts([(cx + w_ / 2, -0.02), (cx + w_ / 2 + 0.03, -0.035)]), W_(0.012))
    else:
        _eye(ctx, X, -0.05, -0.03, 0.17, 0.085 + (0.03 if shock else 0.0), lx, ly, blink, near=True)
        _eye(ctx, X, -0.295, -0.035, 0.095, 0.075 + (0.02 if shock else 0.0), lx, ly, blink, near=False)
    # nose
    brush(ctx, X.pts([(-0.24, -0.04), (-0.30, 0.06), (-0.36, 0.13), (-0.33, 0.16)]), lw * 1.0, taper=(0.6, 0.1))
    brush(ctx, X.pts([(-0.33, 0.16), (-0.27, 0.165)]), lw * 1.3, taper=(0.1, 0.5))
    brush(ctx, X.pts([(-0.24, 0.13), (-0.22, 0.16)]), lw * 0.8)
    # cheek line (smile crease) + freckles
    brush(ctx, X.pts([(-0.12, 0.19), (-0.09, 0.25), (-0.10, 0.29)]), lw * 0.7, taper=(0.4, 0.5))
    for (fx, fy) in ((-0.02, 0.10), (0.03, 0.12), (-0.07, 0.13), (0.01, 0.07), (0.07, 0.09), (-0.30, 0.09),
                     (-0.34, 0.07), (-0.26, 0.11), (-0.20, 0.06)):
        p = X((fx, fy))
        _dot(ctx, p[0], p[1], W_(0.008))
    # mouth
    _summer_mouth(ctx, X, mouth, expr, lw)
    # ---------------------------------------------------------------- hair front: hairline + bangs + strands
    # irregular pointed locks of fringe hanging to the brows
    tips = [(-0.38, -0.17), (-0.25, -0.12), (-0.12, -0.15), (0.00, -0.11), (0.11, -0.18), (0.20, -0.24)]
    roots = [(-0.41, -0.36), (-0.31, -0.42), (-0.18, -0.45), (-0.05, -0.47), (0.07, -0.47), (0.17, -0.45),
             (0.28, -0.40)]
    fr = []
    for i, (tx, ty) in enumerate(tips):
        fr.append(roots[i] if i == 0 else ((roots[i][0] + tips[i - 1][0]) * 0.5, (roots[i][1] * 0.35 + tips[i - 1][1] * 0.65) - 0.035))
        fr.append((tx + sway * 1.4, ty))
    fr.append(roots[-1])
    fr += [(0.30, -0.52), (0.10, -0.64), (-0.14, -0.62), (-0.32, -0.50)]
    pts = X.pts(_cr(fr, 8, closed=True))
    ctx.new_path()
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    set_color(ctx, PAPER)
    ctx.fill_preserve()
    set_color(ctx, INK)
    ctx.set_line_width(lw * 0.8)
    ctx.stroke()
    # a soft shadow tone under the crown of the fringe (volume), then fine strands
    shape(ctx, X.pts([(-0.36, -0.40), (-0.10, -0.49), (0.18, -0.47), (0.26, -0.40), (0.05, -0.36), (-0.25, -0.33)]),
          fill=HAIR_SH)
    for i, (tx, ty) in enumerate(tips):
        r = roots[i]
        brush(ctx, X.pts([(r[0] + 0.04, r[1] + 0.03), (tx * 0.55 + r[0] * 0.45 + 0.01 + sway, ty * 0.45 + r[1] * 0.55),
                          (tx + sway * 1.3 + 0.01, ty - 0.03)]), lw * 0.35, taper=(0.3, 0.8))
    part = (0.04, -0.66)
    for k in range(6):
        u = k / 5
        mid = (0.24 + 0.14 * u, -0.56 + 0.30 * u)
        end = (0.44, 0.14 + 0.02 * k)
        brush(ctx, X.pts([(part[0] + 0.03 * k, part[1] + 0.02 * k), mid, end]), lw * 0.35, taper=(0.1, 0.6))
    # hair tie + near pigtail
    pig_near = [(0.44, 0.16), (0.56, 0.36), (0.62, 0.62), (0.58, 0.88), (0.68, 1.04)]
    _pigtail(ctx, X, pig_near, 0.15, lw, t, wind, phase=0.0)
    tie = X((0.45, 0.16))
    ctx.new_path()
    ctx.arc(tie[0], tie[1], W_(0.045), 0, 2 * math.pi)
    set_color(ctx, INK)
    ctx.fill()
    ctx.restore()
    return {"mouth": X((-0.21, 0.27 + mouth * 0.04)), "eye": X((-0.05, -0.03)), "top": X((0.0, -0.6)),
            "chin": X((-0.18, 0.46))}


def _pigtail(ctx, X, spine, width, lw, t, wind, phase=0.0):
    sw = 0.03 * math.sin(t * 1.7 + phase) + wind * 0.08
    sp = [(px + sw * max(0.0, (py - spine[0][1])) * 1.2, py) for px, py in spine]
    left, right = [], []
    q = _cr(sp, 6)
    for i, p in enumerate(q):
        a = q[max(i - 1, 0)]
        b = q[min(i + 1, len(q) - 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / d, dx / d
        u = i / (len(q) - 1)
        ww = width * (0.55 + 0.6 * math.sin(math.pi * min(1.0, u * 1.25)))
        if u > 0.8:
            ww *= max(0.05, (1 - u) / 0.2)
        left.append((p[0] + nx * ww / 2, p[1] + ny * ww / 2))
        right.append((p[0] - nx * ww / 2, p[1] - ny * ww / 2))
    outline = left + list(reversed(right))
    pts = X.pts(outline)
    ctx.new_path()
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    set_color(ctx, PAPER)
    ctx.fill_preserve()
    set_color(ctx, INK)
    ctx.set_line_width(lw * 1.1)
    ctx.stroke()
    # shadow side + strands
    sh = X.pts(right[: len(right) * 3 // 4] + list(reversed([((l[0] + r[0]) / 2, (l[1] + r[1]) / 2)
                                                             for l, r in zip(left, right)][: len(right) * 3 // 4])))
    ctx.new_path()
    ctx.move_to(*sh[0])
    for p in sh[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    set_color(ctx, HAIR_SH)
    ctx.fill()
    for k in range(4):
        f = 0.2 + 0.2 * k
        st = [(l[0] * (1 - f) + r[0] * f, l[1] * (1 - f) + r[1] * f) for l, r in zip(left, right)]
        st = st[1: int(len(st) * (0.75 + 0.05 * k))]
        if len(st) > 2:
            brush(ctx, X.pts(st[::2]), X.w(0.006), taper=(0.3, 0.6))


def _eye(ctx, X, cx, cy, w, h, lx, ly, blink, near=True):
    W_ = X.w
    op = 1.0 - max(0.0, min(1.0, blink))
    top = [(cx - w / 2, cy + 0.01), (cx - w * 0.1, cy - h * 0.55 * (0.2 + 0.8 * op)), (cx + w / 2, cy - 0.005)]
    bot = [(cx - w / 2, cy + 0.01), (cx, cy + h * 0.45 * op + 0.004), (cx + w / 2, cy - 0.005)]
    if op > 0.08:
        # eye white + iris
        pts = X.pts(_cr(top, 6) + list(reversed(_cr(bot, 6))))
        ctx.new_path()
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        set_color(ctx, PAPER)
        ctx.fill_preserve()
        ctx.save()
        ctx.clip()
        ir = h * 0.62
        ix = cx - w * 0.12 + lx * w * 0.2 * X.f * (-1)
        iy = cy + ly * h * 0.2
        p = X((ix, iy))
        ctx.new_path()
        ctx.arc(p[0], p[1], W_(ir), 0, 2 * math.pi)
        set_color(ctx, (0.35, 0.35, 0.35))
        ctx.fill()
        ctx.new_path()
        ctx.arc(p[0], p[1], W_(ir * 0.55), 0, 2 * math.pi)
        set_color(ctx, INK)
        ctx.fill()
        hp = X((ix + 0.018, iy - 0.02))
        ctx.new_path()
        ctx.arc(hp[0], hp[1], W_(ir * 0.28), 0, 2 * math.pi)
        set_color(ctx, PAPER)
        ctx.fill()
        ctx.restore()
    # upper lid: thick lash line with flicks
    brush(ctx, X.pts(top), W_(0.028 if near else 0.022), taper=(0.25, 0.1))
    ext = 1 if near else -1
    for k in range(3 if near else 2):
        bx, by = top[-1] if near else top[0]
        brush(ctx, X.pts([(bx, by), (bx + ext * 0.025 * (k + 1) * 0.8, by - 0.018 - 0.012 * k)]), W_(0.009),
              taper=(0.0, 0.9))
    if op > 0.1:
        brush(ctx, X.pts(bot), W_(0.008), taper=(0.4, 0.4))
    # crease
    brush(ctx, X.pts([(cx - w * 0.35, cy - h * 0.8), (cx + w * 0.3, cy - h * 0.85)]), W_(0.007), taper=(0.4, 0.4))


def _summer_mouth(ctx, X, mouth, expr, lw):
    W_ = X.w
    cx, cy = -0.20, 0.27
    laugh = expr == "laugh"
    shock = expr == "shock"
    o = max(0.0, min(1.0, mouth))
    if laugh:
        o = max(o, 0.85)
    wid = 0.20 if not shock else 0.13
    open_h = 0.13 * o + (0.05 if shock else 0.0)
    smile = 0.035 if not shock else -0.01
    if open_h < 0.012:
        # closed smile
        brush(ctx, X.pts([(cx - wid / 2, cy - smile), (cx - 0.02, cy + 0.012), (cx + wid / 2, cy - smile * 1.3)]),
              lw * 1.3, taper=(0.3, 0.35))
        brush(ctx, X.pts([(cx - 0.03, cy + 0.05), (cx + 0.03, cy + 0.048)]), lw * 0.6, taper=(0.4, 0.4))
        # dimple at the near corner
        brush(ctx, X.pts([(cx + wid / 2 + 0.01, cy - smile * 1.6), (cx + wid / 2 + 0.02, cy - smile * 0.6)]),
              lw * 0.6)
        return
    up = [(cx - wid / 2, cy - smile), (cx - 0.02, cy - 0.01), (cx + wid / 2, cy - smile * 1.3)]
    lo = [(cx - wid / 2, cy - smile), (cx - 0.03 + 0.02 * o, cy + open_h), (cx + wid / 2, cy - smile * 1.3)]
    pts = X.pts(_cr(up, 6) + list(reversed(_cr(lo, 6))))
    ctx.new_path()
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    set_color(ctx, INK)
    ctx.fill_preserve()
    ctx.save()
    ctx.clip()
    # teeth (upper) + tongue
    tp = X.pts([(cx - wid / 2, cy - 0.05), (cx + wid / 2, cy - 0.05), (cx + wid / 2, cy + 0.02 * (0.5 + o)),
                (cx - wid / 2, cy + 0.02 * (0.5 + o))])
    ctx.new_path()
    ctx.move_to(*tp[0])
    for p in tp[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    set_color(ctx, PAPER)
    ctx.fill()
    tg = X((cx - 0.01, cy + open_h * 0.95))
    ctx.new_path()
    ctx.save()
    ctx.translate(*tg)
    ctx.scale(W_(wid * 0.35), W_(max(open_h * 0.45, 0.01)))
    ctx.arc(0, 0, 1, math.pi, 2 * math.pi)
    ctx.restore()
    set_color(ctx, (0.55, 0.55, 0.55))
    ctx.fill()
    ctx.restore()
    brush(ctx, X.pts(up), lw * 1.2, taper=(0.3, 0.3))
    brush(ctx, X.pts([(cx - 0.05, cy + open_h + 0.035), (cx + 0.03, cy + open_h + 0.03)]), lw * 0.7, taper=(0.4, 0.4))


# ------------------------------------------------------------------------------------------ RAVEN
def raven_bust(ctx, x, y, s, *, facing=1, mouth=0.0, expr="disdain", t=0.0, wind=0.0, blink=0.0):
    """authored facing LEFT like Summer; facing=+1 (looking right) mirrors.  Lost profile from behind."""
    X = _Xf(x, y, s, -1 if facing > 0 else 1, 0.0)
    X.f = -1 if facing > 0 else 1
    W_ = X.w
    lw = W_(0.016)
    sw = 0.02 * math.sin(t * 1.1) + wind * 0.05
    o = max(0.0, min(1.0, mouth))
    ctx.save()
    # shoulders / black top
    shape(ctx, X.pts([(-0.95, 1.9), (-0.85, 1.0), (-0.40, 0.72), (0.30, 0.70), (0.85, 0.95), (1.05, 1.9)]), fill=INK)
    # lost profile: the edge of the cheek, jaw and lips, skin
    jaw = 0.035 * o
    face = [(-0.26, -0.20), (-0.36, -0.08), (-0.38, 0.06), (-0.43, 0.14), (-0.40, 0.20 + jaw * 0.3),
            (-0.44, 0.25 + jaw * 0.6), (-0.39, 0.30 + jaw), (-0.34, 0.40 + jaw), (-0.20, 0.47 + jaw), (-0.05, 0.40),
            (0.02, 0.10), (-0.10, -0.20)]
    shape(ctx, X.pts(face), fill=SKIN)
    shape(ctx, X.pts([(-0.20, 0.47 + jaw), (-0.05, 0.40), (0.0, 0.2), (-0.14, 0.30), (-0.25, 0.42)]), fill=SKIN_SH)
    brush(ctx, X.pts([(-0.36, -0.08), (-0.38, 0.06), (-0.43, 0.14)]), lw, taper=(0.4, 0.1))
    # lips (dark lipstick): upper and lower, parting with the voice
    up = [(-0.44, 0.235 + jaw * 0.4), (-0.405, 0.225), (-0.37, 0.235)]
    lo = [(-0.445, 0.255 + jaw * 0.9), (-0.41, 0.285 + jaw * 1.1), (-0.37, 0.262 + jaw * 0.5)]
    shape(ctx, X.pts([(-0.43, 0.205), (-0.38, 0.215), (-0.37, 0.235)] + list(reversed(up))), fill=INK, line=INK, lw=lw * 0.5)
    shape(ctx, X.pts(lo + [(-0.38, 0.25 + jaw * 0.5), (-0.44, 0.25 + jaw * 0.6)]), fill=INK, line=INK, lw=lw * 0.6)
    if o > 0.15:
        brush(ctx, X.pts([(-0.445, 0.245 + jaw * 0.5), (-0.40, 0.25 + jaw * 0.7)]), lw * 0.6, color=PAPER)
    brush(ctx, X.pts([(-0.39, 0.30 + jaw), (-0.34, 0.40 + jaw), (-0.20, 0.47 + jaw), (-0.05, 0.40)]), lw * 1.5,
          taper=(0.2, 0.3))
    # the tip of the nose just beyond the cheek + lashes flicking past the contour (half-lidded disdain)
    brush(ctx, X.pts([(-0.40, 0.04), (-0.46, 0.13), (-0.43, 0.155)]), lw * 1.1, taper=(0.5, 0.2))
    lid = 0.6 + 0.4 * max(0.0, min(1.0, blink))
    for k in range(4):
        bx, by = -0.36, -0.035 + 0.012 * k * lid
        brush(ctx, X.pts([(bx, by), (bx - 0.05 - 0.01 * k, by - 0.03 + 0.015 * k)]), W_(0.011), taper=(0.0, 0.9))
    brush(ctx, X.pts([(-0.30, -0.10), (-0.37, -0.09)]), W_(0.026), taper=(0.2, 0.6))     # brow
    # ear + hoop earring
    ear = [(-0.06, 0.02), (0.03, -0.02), (0.08, 0.06), (0.06, 0.18), (-0.02, 0.22)]
    shape(ctx, X.pts(ear), fill=SKIN, line=INK, lw=lw)
    hx, hy = X((0.02, 0.36 + 0.01 * math.sin(t * 2.3)))
    ctx.new_path()
    ctx.arc(hx, hy, W_(0.15), 0, 2 * math.pi)
    set_color(ctx, INK)
    ctx.set_line_width(W_(0.03))
    ctx.stroke()
    ctx.new_path()
    ctx.arc(hx - W_(0.03), hy - W_(0.02), W_(0.15), math.pi * 1.1, math.pi * 1.45)
    set_color(ctx, PAPER)
    ctx.set_line_width(W_(0.01))
    ctx.stroke()
    # ---------------------------------------------------------------- the HAIR: a glossy black mass
    hair = [(-0.30, -0.28), (-0.20, -0.48), (0.05, -0.62), (0.40, -0.56), (0.66, -0.30), (0.78, 0.10),
            (0.86, 0.60), (0.95 + sw, 1.10), (1.02 + sw, 1.60), (0.30 + sw * 0.5, 1.70), (0.20, 1.20), (0.12, 0.70),
            (0.10, 0.30), (0.12, 0.04), (0.04, -0.12), (-0.14, -0.20)]
    shape(ctx, X.pts(hair), fill=INK)
    # a lock falling in front of the ear toward the shoulder
    # a thin wisp escaping in front of the ear
    brush(ctx, X.pts([(-0.10, -0.16), (-0.08, 0.02), (-0.11, 0.22), (-0.07, 0.40)]), W_(0.03), taper=(0.2, 0.9))
    # form shadow on the back of the cheek (the face turns away from us)
    shape(ctx, X.pts([(-0.12, -0.18), (-0.04, 0.00), (-0.02, 0.24), (-0.10, 0.38), (-0.16, 0.20), (-0.18, -0.02)]),
          fill=SKIN_SH)
    # white highlight streaks (glossy): a crown sheen band and long strands
    for k in range(7):
        u = k / 6
        a0 = (-0.18 + 0.12 * u, -0.40 - 0.10 * math.sin(u * math.pi))
        a1 = (0.10 + 0.5 * u, -0.45 + 0.05 * math.sin(u * 3.0))
        a2 = (0.25 + 0.55 * u, 0.0 + 0.15 * u)
        brush(ctx, X.pts([a0, a1, a2]), W_(0.018 + 0.012 * math.sin(k * 1.9) ** 2), color=PAPER, taper=(0.6, 0.7))
    for k in range(6):
        u = k / 5
        x0 = 0.35 + 0.45 * u
        brush(ctx, X.pts([(x0, 0.25), (x0 + 0.06 + sw, 0.7), (x0 + 0.12 + sw * 1.4, 1.2)]),
              W_(0.012 + 0.01 * (k % 2)), color=PAPER, taper=(0.7, 0.7))
    ctx.restore()
    return {"mouth": X((-0.40, 0.25 + jaw)), "top": X((0.1, -0.6))}

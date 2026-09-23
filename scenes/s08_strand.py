"""s08 helper: the sticky strand - a glistening, pale viscoelastic filament between Crocus's fingertip and
the POLE (never the cloth). Capillary thinning, catenary sag, beads-on-a-string, snap + elastic recoil.

Drawn in stage units under whatever ctx transform is active; `px` = screen px per stage unit so hairlines
stay visible (min width) at every shot size.
"""
import math

from vx import clamp, smoothstep, lerp

T_TOUCH = 176.40      # finger meets pole
T_PULL = 176.95       # starts pulling away (insert close-up)
T_SNAP = 178.90       # end of "sticky." -> the strand breaks

CORE = (0.95, 0.955, 0.94)
EDGE = (0.36, 0.40, 0.46)      # refraction: a clear filament shows dark edges against bright light
HI = (1.0, 1.0, 1.0)


def _curve(A, B, sag, n=36):
    pts = []
    for i in range(n + 1):
        s = i / n
        x = lerp(A[0], B[0], s)
        y = lerp(A[1], B[1], s) + 4 * sag * s * (1 - s)
        pts.append((x, y, s))
    return pts


def _ribbon(ctx, pts, widths):
    """closed outline of a variable-width ribbon along pts"""
    left, right = [], []
    n = len(pts)
    for i, (x, y, s) in enumerate(pts):
        x0, y0 = pts[max(0, i - 1)][:2]
        x1, y1 = pts[min(n - 1, i + 1)][:2]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) + 1e-9
        nx, ny = -dy / L, dx / L
        w = widths[i] * 0.5
        left.append((x + nx * w, y + ny * w))
        right.append((x - nx * w, y - ny * w))
    ctx.new_path()
    ctx.move_to(*left[0])
    for p in left[1:]:
        ctx.line_to(*p)
    for p in reversed(right):
        ctx.line_to(*p)
    ctx.close_path()


def _blob(ctx, x, y, r, a, px):
    r = max(r, 0.6 / px)
    ctx.new_path()
    ctx.arc(x, y, r * 1.08 + 0.7 / px, 0, 2 * math.pi)
    ctx.set_source_rgba(*EDGE, 0.45 * a)
    ctx.fill()
    ctx.arc(x, y, r, 0, 2 * math.pi)
    ctx.set_source_rgba(*CORE, 0.72 * a)
    ctx.fill()
    ctx.arc(x - r * 0.35, y - r * 0.38, r * 0.32, 0, 2 * math.pi)
    ctx.set_source_rgba(*HI, 0.95 * a)
    ctx.fill()
    ctx.arc(x + r * 0.3, y + r * 0.35, r * 0.14, 0, 2 * math.pi)
    ctx.set_source_rgba(*HI, 0.5 * a)
    ctx.fill()


def _strand(ctx, pts, wmid, wend, px, alpha, beads, T):
    n = len(pts)
    minw = 0.75 / px
    widths = []
    for (x, y, s) in pts:
        cone = math.exp(-s / 0.07) + math.exp(-(1 - s) / 0.07)
        widths.append(max(minw, wmid + (wend - wmid) * min(1.0, cone)))
    a_line = alpha * min(1.0, (wmid * px) / 0.75 + 0.35)
    # luminous halo (it catches the morning sun), translucent body with a refractive (slightly darker) rim,
    # then a bright core and a specular line
    _ribbon(ctx, pts, [w * 3.2 for w in widths])
    ctx.set_source_rgba(*HI, 0.10 * a_line)
    ctx.fill()
    _ribbon(ctx, pts, [w * 1.35 + 0.9 / px for w in widths])
    ctx.set_source_rgba(*EDGE, 0.45 * a_line)
    ctx.fill()
    _ribbon(ctx, pts, widths)
    ctx.set_source_rgba(*CORE, 0.62 * a_line)
    ctx.fill()
    ctx.new_path()
    for i, (x, y, s) in enumerate(pts):
        yy = y - widths[i] * 0.22
        (ctx.move_to if i == 0 else ctx.line_to)(x, yy)
    ctx.set_line_width(max(0.5 / px, wmid * 0.35))
    ctx.set_source_rgba(*HI, 0.85 * a_line)
    ctx.stroke()
    # travelling glints (light sliding along the wet thread)
    for g in range(2):
        sg = (T * 0.55 + g * 0.5) % 1.0
        i = int(sg * (n - 1))
        x, y, _ = pts[i]
        r = max(0.9 / px, widths[i] * 0.9)
        ctx.new_path()
        ctx.arc(x, y - widths[i] * 0.2, r, 0, 2 * math.pi)
        ctx.set_source_rgba(*HI, 0.45 * a_line * math.sin(math.pi * sg))
        ctx.fill()
    for (sb, rb) in beads:
        i = min(n - 1, max(0, int(sb * (n - 1))))
        x, y, _ = pts[i]
        _blob(ctx, x, y, rb, alpha, px)


def draw(ctx, T, A, B, px, alpha=1.0):
    """A = fingertip, B = contact point on the pole (stage units, B stays on the pole)."""
    if T < T_TOUCH - 0.02:
        return
    d = math.hypot(A[0] - B[0], A[1] - B[1])
    if T < T_SNAP:
        if d < 1.2:
            # squished contact dab between finger and pole
            _blob(ctx, (A[0] + B[0]) / 2, (A[1] + B[1]) / 2, 1.6, alpha * smoothstep(T_TOUCH - 0.02, T_TOUCH + 0.1, T), px)
            return
        stretch = d / 1.2
        age = max(0.0, T - T_PULL)
        wend = 3.6 / (1 + 0.05 * stretch)
        wmid = 4.5 / stretch ** 0.45 * math.exp(-0.35 * age) + 0.45
        sag = d * (0.08 + 0.12 * smoothstep(0.3, 2.2, age)) + 0.6
        pts = _curve(A, B, sag)
        beads = []
        kb = smoothstep(0.35, 1.0, age)
        if kb > 0:
            for j, (s0, r0) in enumerate(((0.26, 0.95), (0.41, 0.7), (0.58, 1.15), (0.74, 0.8), (0.86, 0.55))):
                drift = 0.06 * smoothstep(0.8, 2.0, age) * (0.5 - s0)          # beads creep to the low point
                beads.append((s0 + drift, r0 * kb * (0.6 + 0.4 * wmid / 0.8)))
        _strand(ctx, pts, wmid, wend, px, alpha, beads, T)
        _blob(ctx, B[0], B[1], 1.5, alpha, px)
        _blob(ctx, A[0], A[1], 1.3, alpha, px)
        return
    # ---- snapped: two tails recoil elastically (damped), leaving glistening dabs on finger and pole
    tt = T - T_SNAP
    brk = 0.55
    rec = math.exp(-9.0 * tt) * math.cos(2 * math.pi * 3.2 * tt)
    rec = max(0.0, rec)
    sag = d * 0.15 + 0.6
    pts = _curve(A, B, sag, 40)
    k = int(brk * 40)
    for side, (anchor, seg_pts) in enumerate(((A, pts[:k + 1]), (B, list(reversed(pts[k:]))))):
        m = max(2, int(len(seg_pts) * rec))
        tail = seg_pts[:m]
        if rec > 0.03 and len(tail) > 2:
            # gravity droop grows as the tail loses tension
            droop = 6.0 * (1 - rec) * (1 if side == 0 else 0.8)
            tail = [(x, y + droop * (i / (len(tail) - 1)) ** 2, s) for i, (x, y, s) in enumerate(tail)]
            _strand(ctx, tail, 0.35, 2.0, px, alpha, [(1.0, 0.9 * (1 - rec) + 0.4)], T)
    wob = 1 + 0.25 * math.exp(-6 * tt) * math.sin(tt * 38)
    _blob(ctx, A[0], A[1] + 0.8, 1.6 * wob, alpha, px)
    _blob(ctx, B[0], B[1] + 0.6, 1.8 * wob, alpha, px)

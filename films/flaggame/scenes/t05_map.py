"""t05 shot C — the map (110.05 -> 113.10): 'Move it! MOVE IT!' — an engraved 19th-century plan of the Burn at
Alchemy IX, in the rain. The Flags that were found are carried across the city (dotted trails, like a
treasure map) to the five corners of the pentagonal trash fence; the ley lines strike between them — a
pentagram around the effigy — and the circle of cloud over the city starts to open ('AND IT WORKED.').
Map space = design px at zoom 1, the effigy at (960, 540)."""
import math

import cairo
import cv2
import numpy as np

from vx import ink, comic
from vx.canvas import Canvas
from vx.config import FONT_SERIF
from vx.ease import clamp, lerp, smoothstep, ease_in_out, ease_out, ease_in, fbm2
from vx.flag import draw_flag

T0 = 110.05
T1 = 113.10
CX, CY = 960.0, 540.0
R_FENCE = 400.0
T_ARRIVE = (111.30, 111.36, 111.42, 111.47, 111.52)
T_LEY0, T_LEY1 = 111.50, 111.84
T_OPEN = 111.90


def corners():
    return [(CX + R_FENCE * math.cos(-math.pi / 2 + k * 2 * math.pi / 5),
             CY + R_FENCE * math.sin(-math.pi / 2 + k * 2 * math.pi / 5)) for k in range(5)]


def camera(T):
    """(zoom, rot, cx, cy): map point at screen centre"""
    u = clamp((T - T0) / (T1 - T0))
    z = lerp(1.18, 0.97, ease_in_out(clamp((T - T0) / 1.3), 2))
    zin = ease_in(clamp((T - 112.0) / (T1 - 112.0)), 2.2)
    z *= 1.0 + 2.6 * zin
    rot = lerp(-0.06, 0.03, ease_in_out(u, 2))
    k = ease_in_out(clamp((T - 111.4) / 0.9), 2)
    return z, rot, lerp(CX - 150, CX, k), lerp(CY - 10, CY - 20 + 30 * zin, k)


def apply_cam(ctx, T):
    z, rot, cx, cy = camera(T)
    ctx.translate(960, 540)
    ctx.rotate(rot)
    ctx.scale(z, z)
    ctx.translate(-cx, -cy)


def page_matrix(T):
    """cairo matrix: map (page) design px -> screen design px (the camera over the engraved plan)"""
    z, rot, cx, cy = camera(T)
    M = cairo.Matrix()
    M.translate(960, 540)
    M.rotate(rot)
    M.scale(z, z)
    M.translate(-cx, -cy)
    return M


def map_to_screen(T, x, y):
    z, rot, cx, cy = camera(T)
    dx, dy = (x - cx) * z, (y - cy) * z
    c, s = math.cos(rot), math.sin(rot)
    return 960 + dx * c - dy * s, 540 + dx * s + dy * c


class MapPlan:
    def __init__(self, seed=11):
        rng = np.random.default_rng(seed)
        self.rng = rng
        # camps along the arcs of a horseshoe city (open to the north, the temple beyond)
        self.arcs = [165.0, 215.0, 265.0, 315.0, 365.0]
        self.a0, self.a1 = math.radians(-45.0), math.radians(225.0)    # cairo angles: the gap is at the top
        camps = []
        for i in range(len(self.arcs) - 1):
            r0, r1 = self.arcs[i], self.arcs[i + 1]
            n = int(30 + 8 * i)
            for k in range(n):
                a = lerp(self.a0, self.a1, (k + rng.uniform(0.2, 0.8)) / n)
                if abs(math.degrees(a) % 30) < 3:     # radial streets
                    continue
                r = rng.uniform(r0 + 9, r1 - 9)
                camps.append((CX + r * math.cos(a), CY + r * math.sin(a),
                              int(rng.integers(0, 4)), rng.uniform(0.7, 1.25)))
        self.camps = camps
        cs = corners()
        # carriers: from a camp towards a corner, a wandering path (bezier through a waypoint)
        self.carriers = []
        starts = [(CX - 120, CY + 250), (CX + 230, CY + 170), (CX - 280, CY - 60), (CX + 300, CY - 120),
                  (CX + 40, CY + 330)]
        order = [2, 1, 4, 0, 3]   # which corner each start goes to
        for k, (sx, sy) in enumerate(starts):
            ex, ey = cs[order[k]]
            mx, my = (sx + ex) / 2 + rng.uniform(-90, 90), (sy + ey) / 2 + rng.uniform(-90, 90)
            self.carriers.append(dict(p0=(sx, sy), pm=(mx, my), p1=(ex, ey), seed=31 + k, t_arr=T_ARRIVE[k],
                                      t_go=T0 + 0.05 + 0.08 * k))
        # mountains around the playa (hachured hills on the map's rim)
        self.hills = []
        for k in range(46):
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(640, 900)
            self.hills.append((CX + r * math.cos(a) * 1.25, CY + r * math.sin(a) * 0.95, rng.uniform(26, 60)))
        self.rain = np.stack([rng.uniform(-300, 2300, 2600), rng.uniform(-300, 1400, 2600), rng.uniform(0.5, 1.0, 2600)], 1)

    @staticmethod
    def path(c, u):
        (x0, y0), (xm, ym), (x1, y1) = c["p0"], c["pm"], c["p1"]
        a = (1 - u) ** 2
        b = 2 * (1 - u) * u
        d = u * u
        return a * x0 + b * xm + d * x1, a * y0 + b * ym + d * y1

    def progress(self, c, T):
        return ease_in_out(clamp((T - c["t_go"]) / (c["t_arr"] - c["t_go"])), 1.6)


# ------------------------------------------------------------ drawing helpers (map space)
def _hill(ctx, x, y, s):
    ctx.save()
    ctx.set_line_width(1.1)
    ctx.set_source_rgb(0.12, 0.12, 0.12)
    ctx.move_to(x - s, y)
    ctx.curve_to(x - s * 0.5, y - s * 0.2, x - s * 0.25, y - s * 0.95, x, y - s)
    ctx.curve_to(x + s * 0.3, y - s * 0.9, x + s * 0.55, y - s * 0.2, x + s, y)
    ctx.stroke()
    # hachures on the shadow side
    for k in range(7):
        u = k / 7
        ctx.move_to(x + s * 0.05 + u * s * 0.8, y - s * (0.95 - u * 0.9))
        ctx.line_to(x + s * 0.05 + u * s * 0.75, y - s * 0.02)
    ctx.set_line_width(0.7)
    ctx.stroke()
    ctx.restore()


def _camp(ctx, x, y, kind, s):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    ctx.set_line_width(1.0)
    if kind == 0:     # tent
        ctx.move_to(-7, 3)
        ctx.line_to(0, -7)
        ctx.line_to(7, 3)
        ctx.close_path()
        ctx.set_source_rgb(0.62, 0.62, 0.62)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.05, 0.05, 0.05)
        ctx.stroke()
        ctx.move_to(0, -7)
        ctx.line_to(2, 3)
        ctx.stroke()
    elif kind == 1:   # shade structure
        ctx.rectangle(-9, -5, 18, 8)
        ctx.set_source_rgb(0.8, 0.8, 0.8)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.05, 0.05, 0.05)
        ctx.stroke()
        ctx.move_to(-9, -5)
        ctx.line_to(9, 3)
        ctx.stroke()
    elif kind == 2:   # dome
        ctx.arc(0, 3, 7, math.pi, 2 * math.pi)
        ctx.close_path()
        ctx.set_source_rgb(0.72, 0.72, 0.72)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.05, 0.05, 0.05)
        ctx.stroke()
        ctx.move_to(-5, -2)
        ctx.line_to(5, -2)
        ctx.stroke()
    else:             # van
        ctx.rectangle(-8, -4, 16, 7)
        ctx.set_source_rgb(0.45, 0.45, 0.45)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.05, 0.05, 0.05)
        ctx.stroke()
    ctx.restore()


def _effigy_icon(ctx, x, y, s, glow=0.0):
    """the effigy as an old-map pictogram (in elevation): a lattice man, arms raised, on his plinth"""
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    ctx.set_source_rgb(0.02, 0.02, 0.02)
    ctx.set_line_width(1.6)
    pts = [((-14, 0), (14, 0)), ((-14, 0), (-9, -8)), ((14, 0), (9, -8)), ((-9, -8), (9, -8)),
           ((-5, -8), (-3, -26)), ((5, -8), (3, -26)), ((-4, -26), (4, -26)), ((-4, -26), (-7, -40)),
           ((4, -26), (7, -40)), ((-7, -40), (7, -40)), ((-7, -40), (-15, -50)), ((-15, -50), (-17, -62)),
           ((7, -40), (15, -50)), ((15, -50), (17, -62))]
    for (a, b) in pts:
        ctx.move_to(*a)
        ctx.line_to(*b)
    ctx.stroke()
    ctx.set_line_width(0.8)
    for k in range(4):   # lattice X's
        yy = -8 - k * 4.5
        ctx.move_to(-5 + k * 0.5, yy)
        ctx.line_to(-3.5 + k * 0.5, yy - 4.5)
        ctx.move_to(5 - k * 0.5, yy)
        ctx.line_to(3.5 - k * 0.5, yy - 4.5)
    ctx.stroke()
    ctx.move_to(0, -41)
    ctx.line_to(-4, -47)
    ctx.line_to(0, -55)
    ctx.line_to(4, -47)
    ctx.close_path()
    ctx.set_line_width(1.4)
    ctx.stroke()
    ctx.restore()


def _carrier(ctx, x, y, t, s, facing):
    """tiny black map-figure running with its flag held up (the flag itself is drawn on the spot plate)"""
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s * facing, s)
    ph = t * 9.0
    ctx.set_source_rgb(0.02, 0.02, 0.02)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_width(2.4)
    ctx.arc(1.5, -21, 3.2, 0, 2 * math.pi)
    ctx.fill()
    ctx.move_to(1, -17)
    ctx.line_to(-0.5, -8)
    ctx.stroke()
    for sg in (1, -1):
        a = 0.6 * math.sin(ph) * sg
        ctx.move_to(-0.5, -8)
        ctx.line_to(-0.5 + 5 * math.sin(a), -8 + 8 * math.cos(a))
    ctx.stroke()
    ctx.move_to(1, -15)
    ctx.line_to(5, -19)       # arm to the pole
    ctx.move_to(0.5, -15)
    ctx.line_to(-3.5, -11 + 1.5 * math.sin(ph))
    ctx.stroke()
    ctx.restore()


def draw_world(ctx, fc, T, plan):
    """the engraved map in grey (map space via the camera)"""
    ctx.save()
    apply_cam(ctx, T)
    # the playa: pale, lightly stippled by the engraving; salt flats
    ctx.set_source_rgb(0.985, 0.985, 0.985)
    ctx.rectangle(-900, -700, 3700, 2500)
    ctx.fill()
    for (x, y, s) in plan.hills:
        _hill(ctx, x, y, s)
    # the pentagonal trash fence (dashed)
    cs = corners()
    ctx.set_source_rgb(0.05, 0.05, 0.05)
    ctx.set_line_width(3.2)
    ctx.set_dash([12, 7])
    for k in range(5):
        ctx.move_to(*cs[k])
        ctx.line_to(*cs[(k + 1) % 5])
    ctx.stroke()
    ctx.set_dash([])
    # roads: concentric arcs + radials (double-ruled like an old town plan)
    for r in plan.arcs:
        for w_, c in ((8.5, 0.0), (4.6, 0.99)):
            ctx.set_line_width(w_)
            ctx.set_source_rgb(c, c, c)
            ctx.new_sub_path()
            ctx.arc(CX, CY, r, plan.a0, plan.a1)
            ctx.stroke()
    for kdeg in range(-45, 226, 30):
        a = math.radians(kdeg)
        for w_, c in ((7.0, 0.0), (3.8, 0.99)):
            ctx.set_line_width(w_)
            ctx.set_source_rgb(c, c, c)
            ctx.move_to(CX + plan.arcs[0] * math.cos(a), CY + plan.arcs[0] * math.sin(a))
            ctx.line_to(CX + plan.arcs[-1] * math.cos(a), CY + plan.arcs[-1] * math.sin(a))
            ctx.stroke()
    for (x, y, k, s) in plan.camps:
        _camp(ctx, x, y, k, s)
    # the esplanade + the effigy's circle
    ctx.set_source_rgb(0.05, 0.05, 0.05)
    ctx.set_line_width(1.4)
    ctx.arc(CX, CY, 70, 0, 2 * math.pi)
    ctx.stroke()
    _effigy_icon(ctx, CX, CY + 22, 1.35)
    # the temple beyond the open end
    tx, ty = CX, CY - 290
    ctx.move_to(tx - 16, ty + 10)
    ctx.line_to(tx, ty - 18)
    ctx.line_to(tx + 16, ty + 10)
    ctx.close_path()
    ctx.set_source_rgb(0.55, 0.55, 0.55)
    ctx.fill_preserve()
    ctx.set_source_rgb(0.05, 0.05, 0.05)
    ctx.stroke()
    # compass rose
    _compass(ctx, CX + 610, CY + 300, 80)
    # cartouche
    _cartouche(ctx, CX + 600, CY - 330)
    # the carriers' dotted trails + the figures
    for c in plan.carriers:
        u = plan.progress(c, T)
        if u <= 0:
            continue
        n = max(2, int(60 * u))
        ctx.set_source_rgb(0.02, 0.02, 0.02)
        for k in range(n):
            x, y = plan.path(c, u * k / (n - 1))
            ctx.arc(x, y, 2.8, 0, 2 * math.pi)
            ctx.fill()
        if u < 1:
            x, y = plan.path(c, u)
            x2, y2 = plan.path(c, min(1, u + 0.02))
            _carrier(ctx, x, y, T + c["seed"], 1.9, 1 if x2 >= x else -1)
        else:
            # X marks the spot
            x, y = c["p1"]
            ctx.set_line_width(2.6)
            ctx.move_to(x - 9, y - 9)
            ctx.line_to(x + 9, y + 9)
            ctx.move_to(x + 9, y - 9)
            ctx.line_to(x - 9, y + 9)
            ctx.stroke()
    # the ley lines: the pentagram strikes between the planted Flags, then the circle
    _ley(ctx, T)
    ctx.restore()


def _ley(ctx, T):
    if T < T_LEY0:
        return
    cs = corners()
    u = clamp((T - T_LEY0) / (T_LEY1 - T_LEY0))
    star = [(cs[k], cs[(k + 2) % 5]) for k in range(5)]
    for k, (a, b) in enumerate(star):
        v = clamp(u * 1.6 - k * 0.12)
        if v <= 0:
            continue
        v = ease_out(v, 2)
        ex, ey = a[0] + (b[0] - a[0]) * v, a[1] + (b[1] - a[1]) * v
        for w_, c in ((9.0, 0.0), (5.0, 1.0)):     # engraved light: a bare white band with inked borders
            ctx.set_line_width(w_)
            ctx.set_source_rgb(c, c, c)
            ctx.move_to(*a)
            ctx.line_to(ex, ey)
            ctx.stroke()
    cu = clamp((T - (T_LEY1 - 0.12)) / 0.3)
    if cu > 0:
        for w_, c in ((8.0, 0.0), (4.4, 1.0)):
            ctx.set_line_width(w_)
            ctx.set_source_rgb(c, c, c)
            ctx.new_sub_path()
            ctx.arc(CX, CY, R_FENCE, -math.pi / 2, -math.pi / 2 + 2 * math.pi * ease_in_out(cu, 2))
            ctx.stroke()


def _compass(ctx, x, y, r):
    ctx.save()
    ctx.translate(x, y)
    ctx.set_line_width(1.2)
    ctx.set_source_rgb(0.05, 0.05, 0.05)
    ctx.arc(0, 0, r, 0, 2 * math.pi)
    ctx.stroke()
    ctx.arc(0, 0, r * 0.86, 0, 2 * math.pi)
    ctx.stroke()
    for k in range(8):
        a = k * math.pi / 4 - math.pi / 2
        L = r * (1.0 if k % 2 == 0 else 0.6)
        wv = r * 0.13
        for sd, c in ((1, 0.05), (-1, 0.95)):
            ctx.move_to(0, 0)
            ctx.line_to(L * math.cos(a), L * math.sin(a))
            ctx.line_to(wv * math.cos(a + sd * math.pi / 2), wv * math.sin(a + sd * math.pi / 2))
            ctx.close_path()
            ctx.set_source_rgb(c, c, c)
            ctx.fill_preserve()
            ctx.set_source_rgb(0.05, 0.05, 0.05)
            ctx.stroke()
    ctx.select_font_face(FONT_SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(r * 0.34)
    e = ctx.text_extents("N")
    ctx.move_to(-e.width / 2 - e.x_bearing, -r * 1.12)
    ctx.show_text("N")
    ctx.restore()


def _cartouche(ctx, x, y):
    ctx.save()
    ctx.translate(x, y)
    w, h = 400, 118
    ctx.rectangle(-w / 2, -h / 2, w, h)
    ctx.set_source_rgb(0.97, 0.97, 0.97)
    ctx.fill_preserve()
    ctx.set_source_rgb(0.05, 0.05, 0.05)
    ctx.set_line_width(3.0)
    ctx.stroke()
    ctx.rectangle(-w / 2 + 7, -h / 2 + 7, w - 14, h - 14)
    ctx.set_line_width(1.0)
    ctx.stroke()
    ctx.select_font_face(FONT_SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    for txt, sz, yy in (("ALCHEMY IX", 40, -6), ("A PLAN OF THE BURN IN THE RAIN", 15, 26)):
        ctx.set_font_size(sz)
        e = ctx.text_extents(txt)
        ctx.move_to(-e.width / 2 - e.x_bearing, yy)
        ctx.show_text(txt)
    ctx.restore()


def cloud_layer(fc, T, plan):
    """engraved storm clouds drifting over the plan, and the circle opening over the effigy.
    returns (tone (h,w), alpha (h,w)) in screen space"""
    k = 4
    hh, ww = fc.h // k + 1, fc.w // k + 1
    ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float64)
    sx = xs * k / fc.s
    sy = ys * k / fc.s
    # screen -> map
    z, rot, cx, cy = camera(T)
    c, s = math.cos(-rot), math.sin(-rot)
    dx, dy = sx - 960, sy - 540
    mx = cx + (dx * c - dy * s) / z
    my = cy + (dx * s + dy * c) / z
    drift = (T - T0) * 22.0
    n = fbm2((mx + drift) / 260.0, my / 260.0, seed=5, octaves=4)
    n2 = fbm2((mx + drift * 1.4) / 90.0, my / 90.0 + 3.0, seed=9, octaves=3)
    dens = np.clip((n + 0.02) * 3.2 + 0.3 * n2, 0, 1)
    lit = np.clip(0.5 + 9.0 * (n - fbm2((mx + drift - 14) / 260.0, (my - 20) / 260.0, seed=5, octaves=4)), 0, 1)
    tone = 0.3 + 0.62 * lit + 0.08 * n2
    # the circle of cloud opening around the effigy
    r = np.hypot(mx - CX, my - CY)
    if T > T_OPEN - 0.2:
        u = ease_out(clamp((T - (T_OPEN - 0.2)) / 1.1), 2.2)
        R = 60 + 640 * u
        ang = np.arctan2(my - CY, mx - CX)
        R_ = R * (0.86 + 0.28 * (0.5 + 0.5 * fbm2(np.cos(ang) * 2 + 5, np.sin(ang) * 2 + T * 0.3, seed=21, octaves=3)))
        m = np.clip((r - R_) / 60.0, 0, 1)
        dens = dens * m
        rim = np.clip(1 - np.abs(r - R_ - 30) / 50.0, 0, 1)
        tone = np.minimum(1.0, tone + 0.3 * rim)
    alpha = np.clip(dens * 0.92, 0, 1).astype(np.float32)
    tone = tone.astype(np.float32)
    tone = cv2.resize(tone, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(alpha, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)
    return tone, alpha


def open_amount(T):
    return ease_out(clamp((T - (T_OPEN - 0.2)) / 1.1), 2.2)


def draw_rain(ctx, fc, T, plan, bg, cloud_a=None):
    """slanted rain streaks over the whole plan (screen space), stopped inside the opened circle"""
    P = plan.rain
    x = (P[:, 0] + (T - T0) * 260.0) % 2600 - 300
    y = (P[:, 1] + (T - T0) * 900.0 * P[:, 2]) % 1700 - 300
    cxs, cys = map_to_screen(T, CX, CY)
    z = camera(T)[0]
    R = (60 + 640 * open_amount(T)) * z if T > T_OPEN - 0.2 else -1
    h, w = bg.shape[:2]
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for i in range(len(P)):
        xi, yi = x[i], y[i]
        if xi < -20 or xi > W + 20 or yi < -40 or yi > H + 20:
            continue
        if R > 0 and math.hypot(xi - cxs, yi - cys) < R:
            continue
        iy, ix = min(h - 1, max(0, int(yi * fc.s))), min(w - 1, max(0, int(xi * fc.s)))
        ca = 1.0 if cloud_a is None else float(cloud_a[iy, ix])
        if ca < 0.15:
            continue
        tb = bg[iy, ix]
        c = 0.97 if tb < 0.6 else 0.2
        ctx.set_source_rgba(c, c, c, 0.8 * min(1.0, ca * 1.5))
        ctx.set_line_width(1.1 * P[i, 2])
        ctx.move_to(xi, yi)
        ctx.line_to(xi - 7 * P[i, 2], yi - 26 * P[i, 2])
        ctx.stroke()
    ctx.restore()


W, H = 1920, 1080


def draw_flags(ctx, fc, T, plan):
    """the found Flags travelling on the map (spot plate). Returns their screen cloth bboxes."""
    boxes = []
    ctx.save()
    apply_cam(ctx, T)
    for c in plan.carriers:
        u = plan.progress(c, T)
        if u <= 0:
            x, y = c["p0"]
            lean = 0.0
        elif u < 1:
            x, y = plan.path(c, u)
            x2, _ = plan.path(c, min(1, u + 0.02))
            x += 8 * (1 if x2 >= x else -1)
            y -= 26
            lean = 0.12 * math.sin(T * 9 + c["seed"])
        else:
            x, y = c["p1"]
            lean = 0.0
        pop = 1.0 if u < 1 else 1.0
        ang = lean
        if u >= 1:
            # planted: a little jolt
            dt = T - c["t_arr"]
            ang = 0.25 * math.exp(-dt * 7) * math.sin(dt * 30)
        draw_flag(ctx, x, y, pole=120, t=T - T0 + c["seed"], wind=0.8, side=1, ang=ang, seed=c["seed"], pop=pop)
        sx, sy = map_to_screen(T, x, y - 105)
        zz = camera(T)[0]
        boxes.append((sx - 30 * zz, sy - 40 * zz, sx + 80 * zz, sy + 40 * zz))
    ctx.restore()
    return boxes


def render(fc, T, st):
    plan = st["map"]
    world = Canvas(fc, bg=(1, 1, 1))
    draw_world(world.ctx, fc, T, plan)
    g = world.rgb()[..., 0]
    tone, alpha = cloud_layer(fc, T, plan)
    # cloud shadows darken the plan under the clouds; the opened circle floods with light
    g = g * (1 - 0.35 * alpha) 
    g = g * (1 - alpha) + tone * alpha
    if T > T_OPEN - 0.2:
        cxs, cys = map_to_screen(T, CX, CY)
        z = camera(T)[0]
        R = (60 + 640 * open_amount(T)) * z
        m = radial_mask_px(fc, cxs, cys, R * 0.55, R * 1.05)
        g = np.minimum(1.0, g + 0.1 * m * open_amount(T))
    rc = Canvas(fc)
    rc.paint_array(np.repeat(g[..., None], 3, 2))
    draw_rain(rc.ctx, fc, T, plan, g, alpha)
    grey = rc.rgb()
    img = ink.engrave(fc, grey, space="page", matrix=page_matrix(T))
    flags = Canvas(fc)
    boxes = draw_flags(flags.ctx, fc, T, plan)
    img = ink.spot(img, flags.rgba())
    top = Canvas(fc)
    from scenes import t05_found as B
    B.inset(top, fc, T, st, avoid=boxes)
    comic.say(top.ctx, fc, "P14", 60, 52, w=520, avoid=boxes)
    img = top.over(img)
    return img


def radial_mask_px(fc, cx, cy, r0, r1):
    ys, xs = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
    d = np.hypot(xs / fc.s - cx, ys / fc.s - cy)
    t = np.clip((d - r0) / max(1e-3, r1 - r0), 0, 1)
    return 1 - t * t * (3 - 2 * t)

"""m10 - the cartoons (the mosaicist's full-size designs) of the apse, the arch wall, the nave and the floor (owner M10).

Every painter draws in its surface's CHART units (see m10_world) on a cairo ctx pre-scaled by the caller.
The conch is designed in two layers:
  front layer  (conch_front)  bands, sky, clouds, the rising sun, the meadow ground, the EXPLICIT band, the rim:
               painted as seen from the nave (front orthographic view, horizontal = horizontal) and warped into
               the stereographic chart (warp_front_to_chart)
  chart layer  (conch_chart)   stars, weld, flowers, rocks, the heap of tesserae, figures: painted in the chart
               directly (stereographic = conformal, so shapes stay true)
Gold leaf: gold_mask(img) finds the flat gold smalti in a finished cartoon.
"""
import math

import cairo
import cv2
import numpy as np

from vx.canvas import col, set_color, mix, circle, ellipse, poly, smooth, surface_from_array
from vx.ease import hash01
from vx import mosaic as mz

from . import m10_world as Wd

SM = mz.SMALTI
R, AX, SY, U = Wd.R, Wd.AX, Wd.SPRING_Y, Wd.U

# extra smalti for this scene (mixes of the house palette; never flag yellow)
PAL = dict(SM)
PAL.update({
    "rose": "#C47A74", "rose_l": "#E0A89A", "dawn_blue": "#6F8CC4", "sky_pale": "#B9C8DC",
    "meadow_d": "#1E4A33", "meadow": "#2F6B45", "meadow_l": "#4F8A55", "hill": "#6E9A7A", "hill_far": "#8FAFA4",
    "weld": "#A3A553", "weld_d": "#6F7A3A", "weld_l": "#C2C27A", "leaf": "#2C5A34",
    "poppy": "#A8322D", "lily": "#EDE6D6", "rock": "#7C7468", "rock_l": "#A39A8B",
    "glass": "#F2E4C4", "glass_rose": "#E8B9A0", "glass_amber": "#E6BE93", "glass_green": "#C2CDAE",
    "aisle": "#0E0D14", "aisle_l": "#1C1A24",
})
GOLDS = [np.array(col(SM[k]), np.float32) for k in ("gold", "gold_d", "gold_l")]


def C(k):
    return col(PAL[k]) if k in PAL else col(k)


def _rng(seed):
    return np.random.default_rng(seed)


# ============================================================ helpers
def gold_mask(img, tol=0.035):
    """1 where the flat cartoon colour is one of the gold smalti"""
    m = np.zeros(img.shape[:2], np.float32)
    for g in GOLDS:
        d = np.abs(img[..., :3] - g).max(-1)
        m = np.maximum(m, (d < tol).astype(np.float32))
    return m


def is_gold(rgb, tol=0.05):
    """per-tile (N,3) colours -> bool: one of the gold smalti"""
    rgb = np.asarray(rgb, np.float32)
    return np.min([np.abs(rgb - g).max(-1) for g in GOLDS], axis=0) < tol


GLASS = [np.array(col(PAL[k]), np.float32) for k in ("glass", "glass_rose", "glass_amber", "glass_green")]


def glass_mask(img, tol=0.03):
    """1 where the flat cartoon colour is window glass (these tesserae glow with the dawn behind them)"""
    m = np.zeros(img.shape[:2], np.float32)
    for g in GLASS:
        m = np.maximum(m, (np.abs(img[..., :3] - g).max(-1) < tol).astype(np.float32))
    return m


_WARP = {}


def warp_front_to_chart(front, sc):
    """front-view image (rgb/rgba, covering [0, 2R] x [0, R] at scale sc) -> the conch chart image (same size).
    Outside the half-disc -> 0."""
    h, w = front.shape[:2]
    key = (w, h, round(sc, 5))
    mp = _WARP.get(key)
    if mp is None:
        jj, ii = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        u = (jj + 0.5) / sc
        v = (ii + 0.5) / sc
        a, b = u - AX, v - SY
        rr = a * a + b * b
        t = 2 * R * R / (rr + R * R)
        X = AX + t * a
        Y = SY + t * b
        inside = (rr <= R * R * 1.0005) & (b <= 0.5 / sc)
        mx = np.where(inside, X * sc - 0.5, -10).astype(np.float32)
        my = np.where(inside, Y * sc - 0.5, -10).astype(np.float32)
        mp = (mx, my)
        _WARP[key] = mp
    return cv2.remap(front, mp[0], mp[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def front_xy(phi, h):
    """front-view point of azimuth phi (rad) at height h above the springing (world units)"""
    th = math.asin(max(-1.0, min(1.0, h / R)))
    return AX + R * math.cos(th) * math.sin(phi), SY - h


def chart_of(phi, h):
    """conch chart point (u, v) + local chart scale k (world per chart unit) for azimuth phi, height h"""
    th = math.asin(max(-1.0, min(1.0, h / R)))
    u, v = Wd.conch_dir(phi, th)
    return u, v, float(Wd.conch_k(u, v))


# ============================================================ the conch: front layer
H_BAND = 0.085 * R          # EXPLICIT band height (0.51 m)
H_MEADOW0 = H_BAND          # meadow front edge
H_HORIZON = 0.25 * R        # distant hills
SUN_H, SUN_R = 0.33 * R, 0.16 * R
H_NIGHT = 0.70 * R          # the starry crown begins
RIM_W = 0.058 * R


def _band(ctx, h0, h1, c):
    set_color(ctx, C(c))
    ctx.rectangle(0, SY - h1, 2 * R, h1 - h0)
    ctx.fill()


def _lumpy(ctx, x0, x1, yb, hgt, lumps, taper=0.18):
    """closed path: flat-ish bottom at yb, lumpy top made of circular bumps [(fx 0..1, radius frac)], tapered ends"""
    L = x1 - x0
    n = 64
    top = []
    for i in range(n + 1):
        f = i / n
        x = x0 + L * f
        env = min(1.0, f / taper, (1 - f) / taper) ** 0.6
        y = yb
        for fx, rf in lumps:
            r = rf * hgt
            dx = x - (x0 + L * fx)
            if abs(dx) < r:
                y = min(y, yb - env * (hgt * 0.35 + math.sqrt(r * r - dx * dx)))
        y = min(y, yb - env * hgt * 0.3)
        top.append((x, y))
    ctx.new_path()
    ctx.move_to(x0, yb)
    for p in top:
        ctx.line_to(*p)
    ctx.line_to(x1, yb)
    # slightly bellied bottom
    ctx.curve_to(x1 - L * 0.2, yb + hgt * 0.12, x0 + L * 0.2, yb + hgt * 0.12, x0, yb)
    ctx.close_path()


def _cloud(ctx, cx, h, length, rng, thick=0.07):
    """a Roman sunrise cloud (SS. Cosma e Damiano): a lumpy streak whose colour bands follow its top edge -
    white crest, rose, flame, red, and a blue-grey underside"""
    bands = ["marble", "rose_l", "rose", "terracotta", "porphyry_l", "porphyry", "dawn_blue", "lapis"]
    H = thick * R
    d = H * 0.16
    nl = int(rng.integers(3, 6))
    lumps = [(0.12 + 0.76 * (i + rng.random() * 0.6) / nl, 0.3 + 0.35 * rng.random()) for i in range(nl)]
    x0, x1 = cx - length / 2, cx + length / 2
    yb = SY - h
    for k, c in enumerate(bands):
        set_color(ctx, C(c))
        sh = k * d
        _lumpy(ctx, x0 + sh * 1.6, x1 - sh * 1.2, yb + sh * 0.35, H - sh * 0.65, lumps)
        ctx.fill()


def conch_front(ctx, seed=10):
    """the conch as seen from the nave (front orthographic), world units, springing midpoint at (AX, SY)"""
    rng = _rng(seed)
    ctx.save()
    ctx.new_path()
    ctx.arc(AX, SY, R, math.pi, 2 * math.pi)
    ctx.close_path()
    ctx.clip()
    # --- gold sky
    set_color(ctx, C("gold"))
    ctx.paint()
    # --- the night crown: lapis cap with a stepped dawn edge
    for h0, c in ((H_NIGHT - 0.075 * R, "sky_pale"), (H_NIGHT - 0.055 * R, "dawn_blue"), (H_NIGHT - 0.032 * R, "lapis_l"),
                  (H_NIGHT - 0.012 * R, "lapis"), (H_NIGHT + 0.03 * R, "lapis_d"), (H_NIGHT + 0.12 * R, "night_vault")):
        set_color(ctx, C(c))
        ctx.new_path()
        # scalloped lower edge: little arcs (the night lifts like a curtain)
        y = SY - h0
        n = 26
        ctx.move_to(0, y)
        for i in range(n):
            x0 = 2 * R * i / n
            x1 = 2 * R * (i + 1) / n
            ctx.curve_to(x0 + (x1 - x0) * 0.25, y + 0.012 * R, x0 + (x1 - x0) * 0.75, y + 0.012 * R, x1, y)
        ctx.line_to(2 * R, 0)
        ctx.line_to(0, 0)
        ctx.close_path()
        ctx.fill()
    # --- the rising sun: a glory of rays, dawn rings, the white disc
    sx, sy = AX, SY - SUN_H
    nr = 48
    for i in range(nr):
        a = -math.pi * (i + 0.5) / nr            # upper half only
        k = i % 4
        r1 = SUN_R * (3.4, 2.3, 2.9, 2.2)[k] * (0.95 + 0.1 * rng.random())
        wd = (0.034, 0.02, 0.026, 0.02)[k]
        set_color(ctx, C(("gold_l", "bone", "gold_l", "marble")[k]))
        ctx.new_path()
        ctx.move_to(sx + math.cos(a - wd) * SUN_R * 1.2, sy + math.sin(a - wd) * SUN_R * 1.2)
        ctx.line_to(sx + math.cos(a) * r1, sy + math.sin(a) * r1)
        ctx.line_to(sx + math.cos(a + wd) * SUN_R * 1.2, sy + math.sin(a + wd) * SUN_R * 1.2)
        ctx.close_path()
        ctx.fill()
    for rr, c in ((1.34, "gold_l"), (1.27, "rose_l"), (1.2, "bone"), (1.1, "marble"), (0.93, "bone"),
                  (0.9, "marble")):
        set_color(ctx, C(c))
        circle(ctx, sx, sy, SUN_R * rr)
        ctx.fill()
    # --- dawn clouds: banks on both sides of the glory, the lowest half behind the hills
    for side in (-1, 1):
        for (dx, h, L, th) in ((0.58, 0.575, 0.46, 0.075), (0.8, 0.47, 0.42, 0.07), (0.4, 0.44, 0.2, 0.05),
                               (0.7, 0.66, 0.3, 0.055), (0.92, 0.36, 0.3, 0.06), (0.52, 0.33, 0.22, 0.045)):
            _cloud(ctx, AX + side * dx * R, h * R, L * R, rng, thick=th)
    # --- the Tesseract (m01's first image) in a starry medallion under the crown: the alpha returns as the omega
    tesseract_medallion(ctx, AX, SY - 0.855 * R, 0.075 * R)
    # --- the meadow: distant hills, graded ground
    _band(ctx, H_MEADOW0, H_HORIZON - 0.01 * R, "meadow_l")
    set_color(ctx, C("hill_far"))
    ctx.new_path()
    ctx.move_to(0, SY - H_HORIZON + 0.02 * R)
    n = 17
    for i in range(n + 1):
        x = 2 * R * i / n
        hh = H_HORIZON + 0.018 * R * (0.5 + 0.5 * math.sin(i * 2.3 + 1.1)) + 0.012 * R * rng.random()
        ctx.curve_to(x - R / n * 0.9, SY - hh - 0.012 * R, x - R / n * 0.4, SY - hh - 0.012 * R, x, SY - hh + 0.006 * R)
    ctx.line_to(2 * R, SY - H_MEADOW0)
    ctx.line_to(0, SY - H_MEADOW0)
    ctx.close_path()
    ctx.fill()
    for h1, c in ((H_HORIZON - 0.004 * R, "hill"), (H_HORIZON - 0.03 * R, "meadow_l"), (0.175 * R, "meadow"),
                  (0.125 * R, "meadow_d")):
        set_color(ctx, C(c))
        ctx.new_path()
        n = 11
        ctx.move_to(0, SY - h1)
        for i in range(n + 1):
            x = 2 * R * i / n
            hh = h1 + 0.01 * R * math.sin(i * 1.9 + h1 * 0.01)
            ctx.curve_to(x - R / n * 0.9, SY - hh - 0.006 * R, x - R / n * 0.4, SY - hh - 0.006 * R, x, SY - hh)
        ctx.line_to(2 * R, SY)
        ctx.line_to(0, SY)
        ctx.close_path()
        ctx.fill()
    # --- the EXPLICIT band (blank lapis between gold fillets)
    _band(ctx, 0, H_BAND, "gold")
    _band(ctx, 0.01 * R, H_BAND - 0.009 * R, "lapis_d")
    # --- the rim: white line, jewelled lapis band, gold fillet
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    for r0, r1, c in ((R - 0.012 * R, R, "marble"), (R - RIM_W + 0.008 * R, R - 0.012 * R, "lapis_d"),
                      (R - RIM_W, R - RIM_W + 0.008 * R, "gold")):
        set_color(ctx, C(c))
        ctx.set_line_width(r1 - r0)
        ctx.new_path()
        ctx.arc(AX, SY, 0.5 * (r0 + r1), math.pi, 2 * math.pi)
        ctx.stroke()
    rm = R - RIM_W * 0.5 - 0.002 * R
    n = 120
    for i in range(n):
        a = math.pi + math.pi * (i + 0.5) / n
        x, y = AX + rm * math.cos(a), SY + rm * math.sin(a)
        if i % 4 == 0:
            set_color(ctx, C("porphyry_l"))
            ellipse(ctx, x, y, 0.014 * R, 0.009 * R, a + math.pi / 2)
        elif i % 4 == 2:
            set_color(ctx, C("malachite"))
            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(a + math.pi / 4)
            ctx.rectangle(-0.0075 * R, -0.0075 * R, 0.015 * R, 0.015 * R)
            ctx.restore()
        else:
            set_color(ctx, C("marble"))
            circle(ctx, x, y, 0.0055 * R)
        ctx.fill()
    ctx.restore()


# ============================================================ the conch: chart layer
def _star(ctx, x, y, r, c="gold_l", n=8):
    set_color(ctx, C(c))
    ctx.new_path()
    for i in range(2 * n):
        a = math.pi * i / n - math.pi / 2
        rr = r if i % 2 == 0 else r * 0.42
        ctx.line_to(x + rr * math.cos(a), y + rr * math.sin(a))
    ctx.close_path()
    ctx.fill()
    set_color(ctx, C("marble"))
    circle(ctx, x, y, r * 0.18)
    ctx.fill()


def stars(seed=21, n=90):
    """star positions on the night crown: list of (u, v, r_chart, id)"""
    rng = _rng(seed)
    out = []
    tries = 0
    while len(out) < n and tries < 20000:
        tries += 1
        phi = rng.uniform(-1.45, 1.45)
        h = rng.uniform(H_NIGHT + 0.03 * R, R * 0.975)
        if h / R > 0.999:
            continue
        u, v, k = chart_of(phi, h)
        fx, fy = front_xy(phi, h)
        if math.hypot(fx - AX, fy - SY) > R - RIM_W * 1.25:
            continue
        if math.hypot(fx - AX, fy - (SY - 0.855 * R)) < 0.105 * R:
            continue
        r = (0.026 + 0.012 * rng.random()) * R / k
        if any((u - q[0]) ** 2 + (v - q[1]) ** 2 < (2.6 * max(r, q[2])) ** 2 for q in out):
            continue
        out.append((u, v, r, len(out)))
    return out


def weld(ctx, x, y, hgt, rng, k=1.0, lean=0.0):
    """Reseda luteola: a slender stem, narrow wavy leaves low down, a long raceme of tiny yellow-green flowers"""
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(lean)
    s = hgt
    set_color(ctx, C("leaf"))
    for i in range(5):
        sd = -1 if i % 2 else 1
        yy = -s * (0.02 + 0.07 * i)
        ctx.new_path()
        ctx.move_to(0, yy)
        ctx.curve_to(sd * s * 0.08, yy - s * 0.03, sd * s * 0.17, yy - s * 0.01, sd * s * (0.2 + 0.03 * rng.random()),
                     yy + s * 0.03)
        ctx.curve_to(sd * s * 0.14, yy - s * 0.005, sd * s * 0.06, yy - s * 0.005, 0, yy + s * 0.025)
        ctx.close_path()
        ctx.fill()
    set_color(ctx, C("weld_d"))
    ctx.set_line_width(s * 0.028)
    ctx.move_to(0, 0)
    ctx.curve_to(s * 0.02, -s * 0.3, -s * 0.02, -s * 0.6, s * 0.01, -s * 0.97)
    ctx.stroke()
    # raceme: from 0.42 to 1.0 of the height, narrowing to the tip
    n = 16
    for i in range(n):
        f = i / (n - 1)
        yy = -s * (0.42 + 0.56 * f)
        wdt = s * (0.055 * (1 - f) + 0.018)
        for sd in (-1, 1):
            set_color(ctx, C("weld_l" if (i + (sd > 0)) % 3 == 0 else "weld"))
            circle(ctx, sd * wdt * 0.55 + s * 0.004 * math.sin(i), yy, wdt * 0.62)
            ctx.fill()
    ctx.restore()


def lily(ctx, x, y, r):
    set_color(ctx, C("leaf"))
    ctx.set_line_width(r * 0.25)
    ctx.move_to(x, y)
    ctx.line_to(x, y - r * 2.2)
    ctx.stroke()
    set_color(ctx, C("lily"))
    for a in (-0.6, 0.0, 0.6):
        ellipse(ctx, x + math.sin(a) * r * 0.7, y - r * 2.5 - math.cos(a) * r * 0.4, r * 0.35, r * 0.75, a)
        ctx.fill()


def poppy(ctx, x, y, r):
    set_color(ctx, C("leaf"))
    ctx.set_line_width(r * 0.22)
    ctx.move_to(x, y)
    ctx.line_to(x + r * 0.2, y - r * 2.4)
    ctx.stroke()
    set_color(ctx, C("poppy"))
    circle(ctx, x + r * 0.2, y - r * 2.7, r * 0.7)
    ctx.fill()
    set_color(ctx, C("ink"))
    circle(ctx, x + r * 0.2, y - r * 2.7, r * 0.2)
    ctx.fill()


def rock(ctx, x, y, r, rng):
    set_color(ctx, C("rock"))
    pts = []
    n = 7
    for i in range(n):
        a = math.pi + math.pi * i / (n - 1)
        rr = r * (0.75 + 0.35 * rng.random())
        pts.append((x + math.cos(a) * rr * 1.3, y + math.sin(a) * rr * 0.8))
    poly(ctx, pts)
    ctx.fill()
    set_color(ctx, C("rock_l"))
    ellipse(ctx, x - r * 0.3, y - r * 0.45, r * 0.45, r * 0.18, -0.2)
    ctx.fill()


def tessera_heap(ctx, x, y, w, h, rng):
    """the heap of fallen tesserae (m08) in which the Flag was planted: a mound of loose many-coloured tiles"""
    cols = ["gold", "lapis", "porphyry", "malachite", "marble", "terracotta", "verdigris", "slate", "gold_d", "bone"]
    set_color(ctx, C("mortar_l"))
    ctx.new_path()
    ctx.move_to(x - w / 2, y)
    ctx.curve_to(x - w * 0.3, y - h * 1.1, x + w * 0.3, y - h * 1.1, x + w / 2, y)
    ctx.close_path()
    ctx.fill()
    n = 140
    for i in range(n):
        f = rng.random()
        px = x + (f - 0.5) * w * 0.95
        top = h * (1 - (2 * f - 1) ** 2)
        py = y - rng.random() * top
        s = h * (0.09 + 0.05 * rng.random())
        set_color(ctx, C(cols[rng.integers(len(cols))]))
        ctx.save()
        ctx.translate(px, py)
        ctx.rotate(rng.uniform(0, math.pi))
        ctx.rectangle(-s / 2, -s / 2, s, s)
        ctx.restore()
        ctx.fill()


# meadow plants in the chart (deterministic list; figures are placed by the scene)
def meadow_items(seed=33, avoid=()):
    """[(kind, u, v, size_chart, seed)] for weld / lily / poppy / rock across the conch meadow.
    avoid: list of (phi0, phi1) azimuth ranges kept clear (figures)."""
    rng = _rng(seed)
    items = []
    for i in range(420):
        phi = rng.uniform(-1.5, 1.5)
        if any(a <= phi <= b for a, b in avoid):
            continue
        h = rng.uniform(H_MEADOW0 + 0.006 * R, H_HORIZON - 0.03 * R)
        u, v, k = chart_of(phi, h)
        a, b = u - AX, v - SY
        if a * a + b * b > (R - RIM_W * 1.2) ** 2:
            continue
        depth = (h - H_MEADOW0) / (H_HORIZON - H_MEADOW0)     # 0 front .. 1 back: smaller toward the horizon
        r = rng.random()
        if r < 0.34:
            items.append(("weld", u, v, (0.10 + 0.07 * rng.random()) * R * (1 - 0.45 * depth) / k, int(rng.integers(1 << 30))))
        elif r < 0.55:
            items.append(("lily", u, v, 0.011 * R * (1 - 0.4 * depth) / k, 0))
        elif r < 0.78:
            items.append(("poppy", u, v, 0.010 * R * (1 - 0.4 * depth) / k, 0))
        elif r < 0.86:
            items.append(("rock", u, v, 0.016 * R * (1 - 0.4 * depth) / k, int(rng.integers(1 << 30))))
    items.sort(key=lambda it: it[2])          # back (smaller v = higher = farther) first
    return items


def conch_chart(ctx, star_list, items, heap=True):
    """chart layer over the warped front layer: stars, meadow plants, the heap of tesserae at the centre"""
    for (u, v, r, i) in star_list:
        _star(ctx, u, v, r)
    for kind, u, v, sz, sd in items:
        rng = _rng(sd)
        if kind == "weld":
            weld(ctx, u, v, sz, rng, lean=rng.uniform(-0.08, 0.08))
        elif kind == "lily":
            lily(ctx, u, v, sz)
        elif kind == "poppy":
            poppy(ctx, u, v, sz)
        else:
            rock(ctx, u, v, sz, rng)
    if heap:
        u, v, k = chart_of(0.0, H_MEADOW0 + 0.02 * R)
        tessera_heap(ctx, u, v, 0.1 * R / k, 0.035 * R / k, _rng(5))


def flag_base_chart():
    """chart point where the central Flag is planted (top of the heap)"""
    u, v, k = chart_of(0.0, H_MEADOW0 + 0.02 * R)
    return u, v - 0.03 * R / k


# ============================================================ assemble the conch's colour layer
_FRONT = {}


def conch_front_image(sc, seed=10):
    """the front layer painted as seen from the nave and warped into the chart, at pixel scale sc
    (premultiplied rgba covering [0, 2R] x [0, R]); cached per scale"""
    key = (round(sc, 5), seed)
    img = _FRONT.get(key)
    if img is None:
        w, h = int(round(2 * R * sc)), int(round(R * sc))
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surf)
        ctx.scale(sc, sc)
        conch_front(ctx, seed)
        surf.flush()
        buf = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
        front = buf[..., [2, 1, 0, 3]].astype(np.float32) / 255.0
        img = warp_front_to_chart(front, sc)
        _FRONT[key] = img
    return img


PAIR_AVOID = ((-0.52, -0.2), (-0.07, 0.07), (0.24, 0.46))


def conch_ornament(ctx, sc):
    """colour layer of the conch chart (ctx pre-scaled by sc): the warped front layer, stars, weld and flowers,
    the heap of tesserae at the centre. Figures are drawn by the scene."""
    from vx.canvas import paint_array
    paint_array(ctx, conch_front_image(sc), 0.0, 0.0, sc)
    conch_chart(ctx, stars(), meadow_items(avoid=PAIR_AVOID))


def conch_band_mask(sc):
    """(h, w) bool in the conch chart at scale sc: the EXPLICIT band (front-view height < H_BAND)"""
    w, h = int(round(2 * R * sc)), int(round(R * sc))
    jj, ii = np.meshgrid((np.arange(w) + 0.5) / sc, (np.arange(h) + 0.5) / sc)
    a, b = jj - AX, ii - SY
    t = 2 * R * R / (a * a + b * b + R * R)
    return (-t * b) < H_BAND


def conch_crop(x0, y0, x1, y1, sc, seed=10):
    """the conch's colour layer for the chart rectangle (x0,y0)-(x1,y1) only, at pixel scale sc (premultiplied
    rgba): the front layer is painted just for the part of the front view this window needs, then warped."""
    w, h = int(round((x1 - x0) * sc)), int(round((y1 - y0) * sc))
    jj, ii = np.meshgrid(x0 + (np.arange(w) + 0.5) / sc, y0 + (np.arange(h) + 0.5) / sc)
    a, b = jj - AX, ii - SY
    t = 2 * R * R / (a * a + b * b + R * R)
    X, Y = AX + t * a, SY + t * b
    fx0, fx1, fy0, fy1 = X.min() - 4, X.max() + 4, Y.min() - 4, Y.max() + 4
    sf = sc * float(t.max())
    fw, fh = int(math.ceil((fx1 - fx0) * sf)), int(math.ceil((fy1 - fy0) * sf))
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, fw, fh)
    ctx = cairo.Context(surf)
    ctx.scale(sf, sf)
    ctx.translate(-fx0, -fy0)
    conch_front(ctx, seed)
    surf.flush()
    buf = np.ndarray((fh, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :fw]
    front = buf[..., [2, 1, 0, 3]].astype(np.float32) / 255.0
    inside = (a * a + b * b <= R * R) & (b <= 0.5 / sc)
    mx = np.where(inside, (X - fx0) * sf - 0.5, -10).astype(np.float32)
    my = np.where(inside, (Y - fy0) * sf - 0.5, -10).astype(np.float32)
    img = cv2.remap(front, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    csurf = surface_from_array(img)
    cctx = cairo.Context(csurf)
    cctx.scale(sc, sc)
    cctx.translate(-x0, -y0)
    conch_chart(cctx, stars(), meadow_items(avoid=PAIR_AVOID))
    csurf.flush()
    b2 = np.ndarray((h, csurf.get_stride() // 4, 4), np.uint8, csurf.get_data())[:, :w]
    return b2[..., [2, 1, 0, 3]].astype(np.float32) / 255.0


# ============================================================ small ornaments
def tesseract_medallion(ctx, x, y, r):
    """lapis roundel with a gold rim, stars, and the Tesseract (the 4-cube's classic projection) in gold and silver"""
    set_color(ctx, C("gold"))
    circle(ctx, x, y, r)
    ctx.fill()
    set_color(ctx, C("marble"))
    circle(ctx, x, y, r * 0.93)
    ctx.fill()
    set_color(ctx, C("lapis_d"))
    circle(ctx, x, y, r * 0.88)
    ctx.fill()
    for i in range(12):
        a = 2 * math.pi * i / 12 + 0.2
        _star(ctx, x + math.cos(a) * r * 0.74, y + math.sin(a) * r * 0.74, r * 0.07, n=6)
    # outer and inner cube (a slight 4D turn), connecting edges
    ro, ri = r * 0.5, r * 0.24
    rot = 0.18
    outer = [(x + ro * math.cos(rot + math.pi / 4 + k * math.pi / 2), y + ro * math.sin(rot + math.pi / 4 + k * math.pi / 2))
             for k in range(4)]
    inner = [(x + ri * math.cos(-rot * 2 + math.pi / 4 + k * math.pi / 2),
              y + ri * math.sin(-rot * 2 + math.pi / 4 + k * math.pi / 2)) for k in range(4)]
    ctx.set_line_width(r * 0.06)
    set_color(ctx, C("silver"))
    for k in range(4):
        ctx.move_to(*outer[k])
        ctx.line_to(*inner[k])
        ctx.stroke()
    set_color(ctx, C("gold_l"))
    ctx.set_line_width(r * 0.075)
    for q in (outer, inner):
        poly(ctx, q)
        ctx.stroke()


def pearl_band(ctx, x0, y0, x1, y1, w, ground="lapis_d", step=None, seed=0):
    """straight jewelled band from (x0,y0) to (x1,y1), width w: ground, gold edges, pearls and alternating gems"""
    L = math.hypot(x1 - x0, y1 - y0)
    ang = math.atan2(y1 - y0, x1 - x0)
    ctx.save()
    ctx.translate(x0, y0)
    ctx.rotate(ang)
    set_color(ctx, C("gold"))
    ctx.rectangle(0, -w / 2, L, w)
    ctx.fill()
    set_color(ctx, C(ground))
    ctx.rectangle(0, -w * 0.36, L, w * 0.72)
    ctx.fill()
    step = step or w * 0.9
    n = max(1, int(L / step))
    for i in range(n):
        cx = (i + 0.5) * L / n
        if i % 2 == 0:
            set_color(ctx, C("porphyry_l" if (i // 2) % 2 == 0 else "malachite"))
            if (i // 2) % 2 == 0:
                ellipse(ctx, cx, 0, w * 0.3, w * 0.2)
            else:
                ctx.rectangle(cx - w * 0.18, -w * 0.18, w * 0.36, w * 0.36)
        else:
            set_color(ctx, C("marble"))
            circle(ctx, cx, 0, w * 0.13)
        ctx.fill()
    ctx.restore()


def rainbow_arch(ctx, cx, cy, r0, width, a0=math.pi, a1=2 * math.pi, drop=0.0):
    """Byzantine rainbow archivolt: concentric bands; `drop` continues the bands straight down both ends"""
    bands = ["marble", "porphyry", "porphyry_l", "terracotta", "gold", "gold_l", "marble", "sky_pale", "lapis_l",
             "lapis", "lapis_d"]
    n = len(bands)
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    for k, c in enumerate(bands):
        r = r0 + width * (k + 0.5) / n
        set_color(ctx, C(c))
        ctx.set_line_width(width / n * 1.04)
        ctx.new_path()
        if drop > 0:
            ctx.move_to(cx + r * math.cos(a0), cy + r * math.sin(a0) + drop)
        ctx.arc(cx, cy, r, a0, a1)
        if drop > 0:
            ctx.line_to(cx + r * math.cos(a1), cy + r * math.sin(a1) + drop)
        ctx.stroke()


def guilloche(ctx, x0, y0, L, w, c1="porphyry", c2="malachite", ground="marble"):
    """Cosmatesque band: two interlaced strands with roundels, on white"""
    set_color(ctx, C(ground))
    ctx.rectangle(x0, y0, L, w)
    ctx.fill()
    n = max(1, int(L / w))
    for i in range(n):
        cx = x0 + (i + 0.5) * L / n
        set_color(ctx, C(c1 if i % 2 else c2))
        circle(ctx, cx, y0 + w / 2, w * 0.36)
        ctx.fill()
        set_color(ctx, C("gold"))
        circle(ctx, cx, y0 + w / 2, w * 0.14)
        ctx.fill()
    set_color(ctx, C("slate"))
    ctx.set_line_width(w * 0.06)
    for sgn in (-1, 1):
        ctx.move_to(x0, y0 + w / 2)
        for i in range(n):
            cx = x0 + (i + 0.5) * L / n
            ctx.curve_to(cx - L / n * 0.3, y0 + w / 2 + sgn * w * 0.5, cx + L / n * 0.3, y0 + w / 2 + sgn * w * 0.5,
                         x0 + (i + 1) * L / n, y0 + w / 2)
        ctx.stroke()


def marble_panel(ctx, x, y, w, h, rng, kind=0):
    """opus sectile panel: veined slab in a white frame, a porphyry or serpentine roundel"""
    set_color(ctx, C("marble"))
    ctx.rectangle(x, y, w, h)
    ctx.fill()
    slab = ["marble_d", "slate", "verdigris", "porphyry"][kind % 4]
    set_color(ctx, C(slab))
    ctx.rectangle(x + w * 0.06, y + h * 0.06, w * 0.88, h * 0.88)
    ctx.fill()
    set_color(ctx, mix(C(slab), C("marble"), 0.3))
    ctx.set_line_width(min(w, h) * 0.018)
    for i in range(4):
        yy = y + h * (0.15 + 0.2 * i + 0.05 * rng.random())
        ctx.move_to(x + w * 0.06, yy)
        ctx.curve_to(x + w * 0.35, yy - h * 0.08, x + w * 0.6, yy + h * 0.1, x + w * 0.94, yy + h * 0.02)
        ctx.stroke()
    rr = min(w, h) * 0.28
    set_color(ctx, C("marble"))
    circle(ctx, x + w / 2, y + h / 2, rr * 1.12)
    ctx.fill()
    set_color(ctx, C("porphyry" if kind % 2 == 0 else "malachite"))
    circle(ctx, x + w / 2, y + h / 2, rr)
    ctx.fill()


def arched_window(ctx, cx, top, w, h, rng, glass=True):
    """round-headed window (top = crown of the arch): rainbow border, marble jamb, transenna lattice of glass panes.
    Returns the glass outline (cx, top, w, h) so the scene can make those tiles glow."""
    r = w / 2
    rainbow_arch(ctx, cx, top + r, r + w * 0.02, w * 0.16, drop=h - r)
    set_color(ctx, C("marble"))
    ctx.new_path()
    ctx.arc(cx, top + r, r + w * 0.02, math.pi, 2 * math.pi)
    ctx.line_to(cx + r + w * 0.02, top + h)
    ctx.line_to(cx - r - w * 0.02, top + h)
    ctx.close_path()
    ctx.fill()
    if glass:
        ctx.save()
        ctx.new_path()
        ctx.arc(cx, top + r, r * 0.9, math.pi, 2 * math.pi)
        ctx.line_to(cx + r * 0.9, top + h - w * 0.05)
        ctx.line_to(cx - r * 0.9, top + h - w * 0.05)
        ctx.close_path()
        ctx.clip()
        set_color(ctx, C("slate"))
        ctx.paint()
        pane = w * 0.2
        ny, nx = int(h / pane) + 2, int(w / pane) + 2
        for j in range(ny):
            for i in range(nx):
                px = cx - w / 2 + i * pane + (pane / 2 if j % 2 else 0) - pane / 2
                py = top + j * pane
                c = ("glass", "glass_rose", "glass", "glass_amber", "glass", "glass_green")[(i * 7 + j * 3) % 6]
                set_color(ctx, C(c))
                ctx.rectangle(px + pane * 0.09, py + pane * 0.09, pane * 0.82, pane * 0.82)
                ctx.fill()
        ctx.restore()
    return (cx, top, w, h)


# ============================================================ lettering (tituli)
_FONT = None


def titulus_font():
    """Roman square capitals: the Tessellator's face if installed (Cinzel), else C059 bold"""
    global _FONT
    if _FONT is None:
        import subprocess
        fams = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True).stdout
        _FONT = next((f for f in ("Cinzel", "Trajan Pro", "Marcellus") if f in fams), "C059")
    return _FONT


def lay_letters(ctx, text, cx, base, cap, tracking=0.12, font=None):
    """measure a line of Roman capitals centred on cx with baseline `base` and cap height `cap`.
    Returns [(char, x, width)] (interpuncts are drawn as lozenges at mid height)."""
    font = font or titulus_font()
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    size = cap / 0.7
    ctx.set_font_size(size)
    xb, yb, wd, ht, xa, ya = ctx.text_extents("H")
    size = size * cap / max(ht, 1e-6)
    ctx.set_font_size(size)
    items = []
    for ch in text:
        if ch == "·":
            items.append((ch, cap * 0.55))
        elif ch == " ":
            items.append((ch, cap * 0.18))
        else:
            items.append((ch, ctx.text_extents(ch)[4]))
    tw = sum(wdt for _, wdt in items) + tracking * cap * (len(items) - 1)
    x = cx - tw / 2
    out = []
    for ch, wdt in items:
        out.append((ch, x, wdt))
        x += wdt + tracking * cap
    return out, size


def draw_letters(ctx, laid, size, base, cap, color, font=None, only=None):
    """paint laid-out letters (only: optional set of indices)"""
    font = font or titulus_font()
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    set_color(ctx, color)
    for i, (ch, x, wdt) in enumerate(laid):
        if only is not None and i not in only:
            continue
        if ch == " ":
            continue
        if ch == "·":
            d = cap * 0.13
            cxx, cyy = x + wdt / 2, base - cap * 0.5
            poly(ctx, [(cxx, cyy - d * 1.4), (cxx + d, cyy), (cxx, cyy + d * 1.4), (cxx - d, cyy)])
            ctx.fill()
            continue
        ctx.move_to(x, base)
        ctx.show_text(ch)


# ============================================================ figure placements (the congregation)
KINDS = ["citizen", "pope", "heretic", "philosopher", "crackpot", "citizen", "schismmancer", "citizen", "pope",
         "philosopher", "heretic", "citizen"]


def conch_figures():
    """figures in the conch meadow: [dict(kind, surf, u, v (feet, chart), h (chart units), facing, seed, role)]"""
    figs = []
    def put(kind, phi, hh, hgt, facing, seed, role="congregation", **kw):
        u, v, k = chart_of(phi, hh)
        figs.append(dict(kind=kind, surf="conch", u=u, v=v, h=hgt / k, facing=facing, seed=seed, role=role, phi=phi, **kw))
    # the donor pair (left of the Flag) and their counterparts, the Powers (right)
    put("lutie", -0.395, H_MEADOW0 + 0.03 * R, 1.12 * U, 0.8, 1, role="lutie")
    put("crocus", -0.285, H_MEADOW0 + 0.025 * R, 1.34 * U, 0.0, 2, role="crocus")
    put("jaguar", 0.30, H_MEADOW0 + 0.025 * R, 1.3 * U, -0.8, 3, role="power", pose="offer")
    put("commander", 0.40, H_MEADOW0 + 0.03 * R, 1.25 * U, -0.8, 4, role="power", pose="command")
    # the processions of the peoples converging on the Flag
    for i in range(6):
        put(KINDS[i], -1.36 + i * 0.13, H_MEADOW0 + 0.035 * R, (1.05 + 0.1 * ((i * 7) % 3)) * U, 0.8, 10 + i,
            pose="march", arms="offer", phase=(i * 0.37) % 1.0)
        put(KINDS[(i + 5) % len(KINDS)], 1.36 - i * 0.13, H_MEADOW0 + 0.035 * R, (1.05 + 0.1 * ((i * 5) % 3)) * U, -0.8,
            30 + i, pose="march", arms="offer", phase=(i * 0.41 + 0.2) % 1.0)
    return figs


HEMI_WINDOWS = (-1.26, -0.63, 0.0, 0.63, 1.26)


def hemi_figures():
    figs = []
    for k, phi in enumerate((-0.945, -0.315, 0.315, 0.945)):
        u = Wd.HEMI_U0 + R * phi
        figs.append(dict(kind=["philosopher", "citizen", "citizen", "philosopher"][k], surf="hemi", u=u, v=SY + 2150.0,
                         h=760.0, facing=0.0, seed=50 + k, role="congregation"))
    return figs


# arch wall chart geometry
ARCH_CU, ARCH_SV = AX - Wd.XL, SY - Wd.CEIL_Y            # opening centre u, springing v in the arch chart
ARCHIVOLT = 320.0


def _arch_clear(u, v, margin=40.0):
    """True if (u, v) on the arch wall is outside the opening + archivolt"""
    du, dv = u - ARCH_CU, v - ARCH_SV
    r = R + ARCHIVOLT + margin
    if dv >= 0:
        return abs(du) > r
    return du * du + dv * dv > r * r


def arch_figures():
    figs = []
    W_ = Wd.XR - Wd.XL
    # top register: the peoples converge on the empty throne above the arch
    for side in (-1, 1):
        for i in range(7):
            u = ARCH_CU + side * (820 + i * 330)
            if 60 < u < W_ - 60:
                figs.append(dict(kind=KINDS[(i * 3 + (side > 0)) % len(KINDS)], surf="arch", u=u, v=1480.0, h=680.0,
                                 facing=-0.8 * side, seed=70 + i + (side > 0) * 10, role="congregation"))
    # spandrel registers
    for side in (-1, 1):
        for (vb, hh, n) in ((2440.0, 600.0, 3), (3340.0, 560.0, 2)):
            for i in range(n):
                u = 150 + i * 270 if side < 0 else W_ - 150 - i * 270
                if _arch_clear(u + 120 * -side, vb - hh):
                    figs.append(dict(kind=KINDS[(i + int(vb)) % len(KINDS)], surf="arch", u=u, v=vb, h=hh,
                                     facing=-0.8 * side, seed=90 + i + int(vb) % 7 + (side > 0) * 20, role="congregation"))
    # the two voices on the piers: the Cantor (left) and the Lector (right)
    figs.append(dict(kind="cantor", surf="arch", u=240.0, v=5500.0, h=1250.0, facing=0.3, seed=120, role="voice"))
    figs.append(dict(kind="lector", surf="arch", u=W_ - 240.0, v=5500.0, h=1250.0, facing=-0.3, seed=121, role="voice"))
    return figs


NAVE_PROC_V, NAVE_PROC_H = 4150.0, 820.0
CLERE_WIN = [1600.0 * i + 1000.0 for i in range(9)]


def nave_figures(side):
    """side walls: the procession toward the apse + figures between the clerestory windows"""
    figs = []
    Lw = Wd.EXTENT[Wd.S_LEFT][0]
    surf = "left" if side < 0 else "right"
    # facing toward the apse: left wall chart u grows toward the apse (+), right wall chart u grows away (-)
    fac = 0.8 if side < 0 else -0.8
    n = 30
    for i in range(n):
        u = 260 + i * (Lw - 520) / (n - 1)
        figs.append(dict(kind=KINDS[(i * 5 + (side > 0) * 3) % len(KINDS)], surf=surf, u=u, v=NAVE_PROC_V,
                         h=NAVE_PROC_H * (0.94 + 0.08 * ((i * 7) % 3) / 2), facing=fac, seed=200 + i + (side > 0) * 50,
                         role="congregation", pose="march", arms="offer", phase=(i * 0.29) % 1.0))
    for i in range(len(CLERE_WIN) - 1):
        u = 0.5 * (CLERE_WIN[i] + CLERE_WIN[i + 1])
        figs.append(dict(kind=KINDS[(i * 2 + 1) % len(KINDS)], surf=surf, u=u, v=2380.0, h=1080.0, facing=0.0,
                         seed=300 + i + (side > 0) * 40, role="congregation"))
    return figs


# ============================================================ figures through vx.icon
ICON_KIND = {"commander": "schism_actual", "cantor": "deacon", "lector": "deacon"}
REST_POSES = ["stand", "orans", "bless", "speak", "clasp", "stand", "offer"]
LAYERS = ("color", "fine", "gold", "silver", "pearl", "edges")


def people(figs, pose="rest"):
    """figure specs -> vx.icon crowd dicts. pose 'rest' (each figure's own pose) | 'raise' (Glorious: pole high)"""
    out = []
    for f in figs:
        kind = ICON_KIND.get(f["kind"], f["kind"])
        style = {}
        if f["role"] in ("lutie", "crocus"):
            style["halo"] = "square"
        if f["kind"] == "cantor":
            style["prop"] = "scroll"
        if f["kind"] == "lector":
            style["prop"] = "codex_open"
        p = dict(kind=kind, x=float(f["u"]), y=float(f["v"]), h=float(f["h"]), seed=int(f["seed"]),
                 facing=float(f["facing"]), t_off=(f["seed"] % 17) * 0.29, style=style)
        if pose == "raise":
            p["pose"] = "raise"
            p["expr"] = "joyful"
        else:
            p["pose"] = f.get("pose") or REST_POSES[f["seed"] % len(REST_POSES)]
            for k in ("arms", "phase", "expr", "look"):
                if k in f:
                    p[k] = f[k]
        out.append(p)
    return out


def chart(extent, sc, ornament=None, figs=(), pose="rest", tile=None, t=0.0, layers=LAYERS, fine_fn=None, extra=None):
    """paint a surface chart at pixel scale sc: ornament(ctx) (colour layer only) + the vx.icon figures on every
    layer (+ extra(ctx, layer): custom-drawn figures, every layer). -> icon.cartoon dict (rgb, alpha, fine, gold,
    silver, pearl, edges ...) + 'fig' (figure coverage), 'anchors' (per figure), 'goldall' (ornament gold smalti +
    figure gold), 'ground' (the ornament's gold ground outside the figures)."""
    from vx import icon
    ppl = people(figs, pose)
    anchors = []

    def figures(ctx, layer):
        a = icon.crowd(ctx, ppl, t=t, layer=layer, tile=tile, sort=False) if ppl else []
        if extra is not None:
            extra(ctx, layer)
        return a

    def draw_all(ctx, layer):
        if layer == "color" and ornament is not None:
            ornament(ctx)
        if layer == "fine" and fine_fn is not None:
            fine_fn(ctx)
        a = figures(ctx, layer)
        if layer == "color":
            anchors[:] = a

    c = icon.cartoon(draw_all, s=sc, extent=extent, layers=layers)
    if ppl or extra is not None:
        c["fig"] = icon.cartoon(figures, s=sc, extent=extent, layers=("color",))["alpha"]
    else:
        c["fig"] = np.zeros(c["alpha"].shape, np.float32)
    c["anchors"] = anchors
    g = gold_mask(c["rgb"]) * (c["alpha"] > 0.5)
    c["goldall"] = np.maximum(g, c["gold"] if "gold" in c else 0).astype(np.float32)
    c["ground"] = (g * (c["fig"] < 0.5)).astype(np.float32)
    return c


# ============================================================ the hemicycle (apse wall)
HEMI_DONOR_V = (45.0, 250.0)


def hemi_ornament(ctx, seed=40):
    """hemicycle chart (vx.mosaic Hemicycle: u = u0 + arc length 0..pi R, v = WORLD y from the springing SY down
    to the floor): cornice, the donor band (blank: its letters are set by the scene), windows, the ground line,
    weld, the guilloche and the marble dado. Figures are drawn by the scene (hemi_figures)."""
    rng = _rng(seed)
    Wc, Hc = Wd.EXTENT[Wd.S_HEMI][0], Wd.HEMI_H
    ctx.save()
    ctx.translate(0, SY)
    set_color(ctx, C("gold"))
    ctx.rectangle(0, 0, Wc, Hc)
    ctx.fill()
    set_color(ctx, C("marble"))
    ctx.rectangle(0, 0, Wc, 45)
    ctx.fill()
    set_color(ctx, C("marble_d"))
    for i in range(int(Wc / 40)):
        ctx.rectangle(i * 40 + 6, 26, 26, 19)
    ctx.fill()
    set_color(ctx, C("lapis_d"))
    ctx.rectangle(0, 45, Wc, HEMI_DONOR_V[1] - 45)
    ctx.fill()
    set_color(ctx, C("gold"))
    ctx.rectangle(0, 52, Wc, 8)
    ctx.rectangle(0, HEMI_DONOR_V[1] - 14, Wc, 8)
    ctx.fill()
    for phi in HEMI_WINDOWS:
        arched_window(ctx, Wd.HEMI_U0 + R * phi, 560.0, 460.0, 1390.0, rng)
    set_color(ctx, C("meadow"))
    ctx.rectangle(0, 2130, Wc, 60)
    ctx.fill()
    set_color(ctx, C("meadow_d"))
    ctx.rectangle(0, 2175, Wc, 25)
    ctx.fill()
    for f in hemi_figures():
        for dx in (-0.23, 0.23):
            weld(ctx, f["u"] + dx * R * 0.55, 2150.0, 420.0 + 60 * rng.random(), rng)
    guilloche(ctx, 0, 2250, Wc, 80)
    n = 9
    for i in range(n):
        marble_panel(ctx, i * Wc / n, 2330, Wc / n, 870, rng, i)
    ctx.restore()


# ============================================================ the triumphal arch wall
ARCH_TEXT = "SINGVLI · SOLI · OMNES · SIMVL"


def hetoimasia(ctx, x, y, s):
    """the prepared, EMPTY throne (a Flag will be planted on it at Glorious): jewelled seat, cushion, footstool"""
    set_color(ctx, C("lapis_d"))
    circle(ctx, x, y - s * 0.55, s * 0.95)
    ctx.fill()
    set_color(ctx, C("gold"))
    ctx.set_line_width(s * 0.06)
    circle(ctx, x, y - s * 0.55, s * 0.95)
    ctx.stroke()
    set_color(ctx, C("gold_l"))
    ctx.rectangle(x - s * 0.42, y - s * 0.9, s * 0.84, s * 0.72)
    ctx.fill()
    set_color(ctx, C("porphyry"))
    ctx.rectangle(x - s * 0.36, y - s * 0.84, s * 0.72, s * 0.3)
    ctx.fill()
    for i in range(5):
        set_color(ctx, C("marble" if i % 2 else "malachite"))
        circle(ctx, x - s * 0.3 + i * s * 0.15, y - s * 0.4, s * 0.05)
        ctx.fill()
    set_color(ctx, C("porphyry_l"))
    ellipse(ctx, x, y - s * 0.92, s * 0.44, s * 0.1)
    ctx.fill()
    set_color(ctx, C("gold"))
    ctx.rectangle(x - s * 0.5, y - s * 0.18, s, s * 0.1)
    ctx.fill()


def throne_seat_chart():
    """arch chart point where the throne's Flag stands (on the cushion)"""
    return ARCH_CU, 1150.0 - 250.0 * 0.92


def arch_ornament(ctx, seed=60):
    """the triumphal-arch wall chart (u 0..16 m, v 0 (ceiling) .. 18 m (floor)); the opening is left transparent.
    Figures are drawn by the scene (arch_figures)."""
    rng = _rng(seed)
    Wc, Hc = Wd.EXTENT[Wd.S_ARCH]
    set_color(ctx, C("gold"))
    ctx.rectangle(0, 0, Wc, Hc)
    ctx.fill()
    # top border + the titulus band: "Each of them alone. All of them together."
    pearl_band(ctx, 0, 40, Wc, 40, 80)
    set_color(ctx, C("lapis_d"))
    ctx.rectangle(0, 90, Wc, 400)
    ctx.fill()
    laid, size = lay_letters(ctx, ARCH_TEXT, ARCH_CU, 400.0, 250.0)
    draw_letters(ctx, laid, size, 400.0, 250.0, C("gold_l"))
    pearl_band(ctx, 0, 520, Wc, 520, 60)
    # register of the peoples on a green ground line, converging on the empty throne
    set_color(ctx, C("meadow"))
    ctx.rectangle(0, 1470, Wc, 50)
    ctx.fill()
    hetoimasia(ctx, ARCH_CU, 1150.0, 250.0)
    for vb in (2440.0, 3340.0):
        set_color(ctx, C("meadow"))
        ctx.rectangle(0, vb - 10, Wc, 45)
        ctx.fill()
    # piers: marble dado under the two voices, continued under the archivolt strips
    wpier = Wc / 2 - R
    for i in range(2):
        x0 = 0 if i == 0 else Wc - wpier
        for k in range(2):
            marble_panel(ctx, x0, Hc - 1700 + k * 850, wpier, 850, rng, k + 2 * i)
    for f in arch_figures():
        if f["role"] == "congregation":
            weld(ctx, f["u"] + 150 * (1 if f["facing"] <= 0 else -1), f["v"], f["h"] * 0.55, rng)
    # the archivolt (rainbow) around the opening, down the jambs to the dado
    ctx.save()
    ctx.new_path()
    ctx.arc(ARCH_CU, ARCH_SV, R + ARCHIVOLT, math.pi, 2 * math.pi)
    ctx.line_to(ARCH_CU + R + ARCHIVOLT, Hc - 1700)
    ctx.line_to(ARCH_CU - R - ARCHIVOLT, Hc - 1700)
    ctx.close_path()
    ctx.clip()
    set_color(ctx, C("gold"))
    ctx.paint()
    ctx.restore()
    rainbow_arch(ctx, ARCH_CU, ARCH_SV, R + 10, ARCHIVOLT, drop=Hc - ARCH_SV - 1700)
    # the opening itself: transparent (no tiles; the apse is seen through it)
    ctx.save()
    ctx.new_path()
    ctx.arc(ARCH_CU, ARCH_SV, R, math.pi, 2 * math.pi)
    ctx.line_to(ARCH_CU + R, Hc)
    ctx.line_to(ARCH_CU - R, Hc)
    ctx.close_path()
    ctx.set_operator(cairo.OPERATOR_CLEAR)
    ctx.fill()
    ctx.restore()


# ============================================================ the nave side walls
def nave_ornament(ctx, side, seed=80):
    """left (side=-1) / right (+1) nave wall chart: u along the wall (left: toward the apse; right: away),
    v 0 (ceiling) .. 7200 (floor). Figures are drawn by the scene (nave_figures)."""
    rng = _rng(seed + (side > 0))
    Wc, Hc = Wd.EXTENT[Wd.S_LEFT]
    set_color(ctx, C("gold"))
    ctx.rectangle(0, 0, Wc, Hc)
    ctx.fill()
    # top frieze: lapis with starry roundels
    set_color(ctx, C("lapis_d"))
    ctx.rectangle(0, 0, Wc, 500)
    ctx.fill()
    for i in range(int(Wc / 800)):
        x = 400 + i * 800
        set_color(ctx, C("gold"))
        circle(ctx, x, 250, 170)
        ctx.fill()
        set_color(ctx, C("lapis"))
        circle(ctx, x, 250, 140)
        ctx.fill()
        _star(ctx, x, 250, 100, n=8)
    pearl_band(ctx, 0, 520, Wc, 520, 50)
    # clerestory: windows (their glass glows at dawn); figures stand between them
    for u in CLERE_WIN:
        arched_window(ctx, u, 800.0, 520.0, 1450.0, rng)
    set_color(ctx, C("meadow"))
    ctx.rectangle(0, 2370, Wc, 40)
    ctx.fill()
    guilloche(ctx, 0, 2540, Wc, 100)
    # the procession's ground and the weld between the walkers
    set_color(ctx, C("meadow"))
    ctx.rectangle(0, NAVE_PROC_V - 30, Wc, 70)
    ctx.fill()
    set_color(ctx, C("meadow_d"))
    ctx.rectangle(0, NAVE_PROC_V + 10, Wc, 30)
    ctx.fill()
    procs = [f for f in nave_figures(side) if f["v"] == NAVE_PROC_V]
    for i in range(len(procs) - 1):
        weld(ctx, 0.5 * (procs[i]["u"] + procs[i + 1]["u"]), NAVE_PROC_V, 520 + 80 * rng.random(), rng)
    guilloche(ctx, 0, 4200, Wc, 110, c1="lapis", c2="porphyry")
    arcade(ctx, 0, 4310, Wc, Hc - 4310, rng)


def arcade(ctx, x0, y0, w, h, rng, bay=1100.0):
    """the nave arcade: marble columns with gold capitals, round arches, the aisles beyond in deep shadow"""
    set_color(ctx, C("marble_d"))
    ctx.rectangle(x0, y0, w, h)
    ctx.fill()
    n = max(1, int(round(w / bay)))
    bw = w / n
    col_w = bw * 0.13
    spring = y0 + h * 0.36
    for i in range(n):
        cx = x0 + (i + 0.5) * bw
        r = (bw - col_w) / 2
        # the opening: dark aisle, a faint far wall with a small window
        ctx.new_path()
        ctx.arc(cx, spring, r, math.pi, 2 * math.pi)
        ctx.line_to(cx + r, y0 + h)
        ctx.line_to(cx - r, y0 + h)
        ctx.close_path()
        set_color(ctx, C("aisle"))
        ctx.fill()
        set_color(ctx, C("aisle_l"))
        ctx.rectangle(cx - r * 0.18, spring + r * 0.15, r * 0.36, r * 0.6)
        ctx.fill()
        # archivolt: a thin gold and lapis moulding
        ctx.set_line_width(bw * 0.035)
        set_color(ctx, C("gold"))
        ctx.new_path()
        ctx.arc(cx, spring, r + bw * 0.02, math.pi, 2 * math.pi)
        ctx.stroke()
        set_color(ctx, C("lapis"))
        ctx.set_line_width(bw * 0.02)
        ctx.new_path()
        ctx.arc(cx, spring, r + bw * 0.052, math.pi, 2 * math.pi)
        ctx.stroke()
    for i in range(n + 1):
        cx = x0 + i * bw
        # column shaft (veined marble), capital (gold basket), base
        set_color(ctx, C("marble"))
        ctx.rectangle(cx - col_w / 2, spring, col_w, y0 + h - spring)
        ctx.fill()
        set_color(ctx, C("marble_d"))
        ctx.set_line_width(col_w * 0.08)
        for k in range(3):
            ctx.move_to(cx - col_w * 0.3 + k * col_w * 0.3, spring + 40)
            ctx.curve_to(cx - col_w * 0.1 + k * col_w * 0.2, spring + h * 0.2, cx + col_w * 0.1, spring + h * 0.4,
                         cx - col_w * 0.2 + k * col_w * 0.25, y0 + h - 60)
            ctx.stroke()
        set_color(ctx, C("gold"))
        poly(ctx, [(cx - col_w * 0.85, spring - col_w * 0.9), (cx + col_w * 0.85, spring - col_w * 0.9),
                   (cx + col_w * 0.55, spring), (cx - col_w * 0.55, spring)])
        ctx.fill()
        set_color(ctx, C("slate"))
        ctx.rectangle(cx - col_w * 0.75, y0 + h - col_w * 0.6, col_w * 1.5, col_w * 0.6)
        ctx.fill()


def soffit_ornament(ctx):
    """the intrados of the triumphal arch (vx.mosaic Vault chart: u = arc length across 0..pi R, v = depth
    0..ARCH_Z): a chain of starry medallions on lapis between jewelled borders"""
    L, D = math.pi * R, Wd.ARCH_Z
    set_color(ctx, C("lapis_d"))
    ctx.rectangle(0, 0, L, D)
    ctx.fill()
    pearl_band(ctx, 0, D * 0.09, L, D * 0.09, D * 0.16)
    pearl_band(ctx, 0, D * 0.91, L, D * 0.91, D * 0.16)
    n = 15
    for i in range(n):
        x = (i + 0.5) * L / n
        set_color(ctx, C("gold"))
        circle(ctx, x, D / 2, D * 0.3)
        ctx.fill()
        set_color(ctx, C("lapis"))
        circle(ctx, x, D / 2, D * 0.25)
        ctx.fill()
        _star(ctx, x, D / 2, D * 0.17, n=8)


def jamb_ornament(ctx):
    """the jambs of the arch (depth ARCH_Z x height from the springing to the floor): the medallion chain
    continues down, then marble"""
    D, Hh = Wd.ARCH_Z, Wd.HEMI_H
    set_color(ctx, C("lapis_d"))
    ctx.rectangle(0, 0, D, Hh)
    ctx.fill()
    pearl_band(ctx, D * 0.09, 0, D * 0.09, Hh, D * 0.16)
    pearl_band(ctx, D * 0.91, 0, D * 0.91, Hh, D * 0.16)
    for i in range(4):
        y = (i + 0.5) * 420
        set_color(ctx, C("gold"))
        circle(ctx, D / 2, y, D * 0.3)
        ctx.fill()
        set_color(ctx, C("lapis"))
        circle(ctx, D / 2, y, D * 0.25)
        ctx.fill()
        _star(ctx, D / 2, y, D * 0.17, n=8)
    rng = _rng(3)
    marble_panel(ctx, 0, 1700, D, Hh - 1700, rng, 1)


# ============================================================ floor and ceiling
def floor_cartoon_paint(ctx, seed=90):
    """Cosmati pavement: chart u = x - XL, v = z + R (0 at the back of the apse)"""
    rng = _rng(seed)
    Wc, Hc = Wd.EXTENT[Wd.S_FLOOR]
    set_color(ctx, C("marble"))
    ctx.rectangle(0, 0, Wc, Hc)
    ctx.fill()
    cu = AX - Wd.XL
    # presbytery: a great porphyry roundel under the conch
    set_color(ctx, C("porphyry"))
    circle(ctx, cu, R * 0.55, 700)
    ctx.fill()
    guilloche(ctx, cu - 700, R * 0.55 + 740, 1400, 80)
    # nave: side fields of small geometric panels, a central chain of roundels
    v = R + 500.0
    while v < Hc:
        for sgn in (-1, 1):
            for k in range(3):
                x0 = cu + sgn * (1200 + k * 600) - 280
                cols = [("porphyry", "malachite"), ("lapis", "porphyry"), ("malachite", "slate")][k]
                set_color(ctx, C(cols[0]))
                ctx.rectangle(x0, v, 560, 560)
                ctx.fill()
                set_color(ctx, C(cols[1]))
                poly(ctx, [(x0 + 280, v + 40), (x0 + 520, v + 280), (x0 + 280, v + 520), (x0 + 40, v + 280)])
                ctx.fill()
                set_color(ctx, C("marble"))
                circle(ctx, x0 + 280, v + 280, 70)
                ctx.fill()
        set_color(ctx, C("porphyry" if int(v / 1400) % 2 else "malachite"))
        circle(ctx, cu, v + 280, 420)
        ctx.fill()
        set_color(ctx, C("gold"))
        ctx.set_line_width(40)
        circle(ctx, cu, v + 280, 470)
        ctx.stroke()
        v += 1400
    for sgn in (-1, 1):
        set_color(ctx, C("slate"))
        ctx.rectangle(cu + sgn * 860 - 30, R, 60, Hc - R)
        ctx.fill()


def ceiling_cartoon_paint(ctx):
    Wc, Hc = Wd.EXTENT[Wd.S_CEIL]
    set_color(ctx, C("gold_d"))
    ctx.rectangle(0, 0, Wc, Hc)
    ctx.fill()
    s = 640.0
    for j in range(int(Hc / s) + 1):
        for i in range(int(Wc / s) + 1):
            x, y = i * s, j * s
            set_color(ctx, C("lapis_d"))
            ctx.rectangle(x + 70, y + 70, s - 140, s - 140)
            ctx.fill()
            _star(ctx, x + s / 2, y + s / 2, s * 0.22, n=8)

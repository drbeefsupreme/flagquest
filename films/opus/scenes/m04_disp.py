"""m04 part II (80.98-95.75): the wall re-sets around the Flag, in a split-flap wave, into a two-register
DISPUTATION in the form of the Summa.

    QVAESTIO · VTRVM MVNDVS SIT VLTERIVS FRANGENDVS          (the question, a gold band; QVAESTIO in red)
    upper register (night): VIDETVR / QVOD NON (left) - the Hypermind as a flat seraph icon - SED / CONTRA (right)
    lower register (gold):  Lutie + the world already shattered by the slop (fragments glowing like phone screens,
                            PRAETEREA sets itself on her second argument) - the Flag hovering in the middle - a
                            too-perfect smiling slop-world (machine-perfect grid of tiny tiles) + Crocus
    bottom band:            RESPONDEO · DICENDVM

L02 (videtur 81.2) Lutie objects; C02 (sed_contra 87.0) "the false coherence must fall": the smiling world cracks like
a phone screen on "must" and falls tile by tile on "fall", revealing a humble world set in true gold that catches the
light on "rise"; C03 (respondeo 92.8) Crocus raises the Byzantine blessing hand on "summon"; the camera pushes into
it and a gold flash peaks on the last frame (hard cut at 95.75; m05 opens with a gold glint at (960, 380)).

world_D (the disputation's chart) = the screen at 80.98-81.9 shifted by (OX, OY); the Flag keeps its world_H track,
drawn through M_D o shift o M_H(81.2).
"""
import math

import cairo
import numpy as np

from vx.ease import clamp, smoothstep
from vx.canvas import smooth
from vx import mosaic as mz

from scenes import m04_tiles as mt
from scenes import m04_art as art

S = art.S
OX, OY = 140.0, 60.0                       # chart = screen-at-81.2 + (OX, OY)
WD = (2200, 1200)
FLIP0, FLIP_SPEED, FLIP_DUR = 80.98, 1450.0, 0.28
VIDETVR, SED, RESP, END = 81.2, 87.0, 92.8, 95.75

# composition (screen-at-81.2 coordinates; painters push the (OX, OY) shift)
LUT = (292.0, 978.0, 512.0)
CRO = (1540.0, 978.0, 532.0)
ORB_L = (652.0, 728.0, 146.0)              # the shattered world
ORB_R = (1296.0, 734.0, 152.0)             # the smiling slop world (layer) / the true gold world (wall) behind it
ICON = (960.0, 262.0, 0.3)                 # the Hypermind as a flat seraph: centre, scale
FLOOR_Y = 980.0
SMILE_PAD = 26.0

INK = art.INK
PASTELS = [art.hexc(h) for h in ("#E8C8D0", "#B8E0E8", "#C9C0A8", "#D9C2E8", "#BFE3C9", "#F0D6B8", "#9FD8FF")]


def chart(x, y):
    return (x + OX, y + OY)


# ============================================================ tituli (vx.mosaic Titulus, Cinzel)
def _tituli():
    T = {}
    probe_q = mz.Titulus("VTRVM MVNDVS SIT VLTERIVS FRANGENDVS", 0, 0, 38, font="roman_black", tracking=0.05)
    probe_r = mz.Titulus("QVAESTIO ·", 0, 0, 38, font="roman_black", tracking=0.05)
    gap = 22.0
    x0 = 960.0 - (probe_r.width + gap + probe_q.width) / 2
    T["q_red"] = mz.Titulus("QVAESTIO ·", x0, 80, 38, color=S["marble"], font="roman_black", tracking=0.05,
                            align="left")
    T["q"] = mz.Titulus("VTRVM MVNDVS SIT VLTERIVS FRANGENDVS", x0 + probe_r.width + gap, 80, 38,
                        color=S["gold"], font="roman_black", tracking=0.05, align="left")
    T["v1"] = mz.Titulus("VIDETVR", 326, 248, 60, color=S["gold"], tracking=0.12)
    T["v2"] = mz.Titulus("QVOD NON", 326, 340, 60, color=S["gold"], tracking=0.12)
    T["pr"] = mz.Titulus("PRAETEREA", 326, 398, 26, color=S["silver"], tracking=0.2)
    T["s1"] = mz.Titulus("SED", 1594, 248, 60, color=S["gold"], tracking=0.12)
    T["s2"] = mz.Titulus("CONTRA", 1594, 340, 60, color=S["gold"], tracking=0.12)
    T["r"] = mz.Titulus("RESPONDEO · DICENDVM", 960, 1054, 40, color=S["gold"], tracking=0.14)
    return T


TIT = _tituli()
GROUP_ORDER = ["v1", "v2", "pr", "s1", "s2", "r"]


def letter_times(tl):
    """global set time of every glyph of every animated titulus"""
    t = {}
    t["v1"] = [81.25 + 0.06 * i for i in range(TIT["v1"].n)]
    t["v2"] = [81.68 + 0.06 * i for i in range(TIT["v2"].n)]
    t["pr"] = [tl.word("L02", "The", 1)[0] + 0.04 * i for i in range(TIT["pr"].n)]
    t["s1"] = [SED + 0.07 * i for i in range(TIT["s1"].n)]
    t["s2"] = [SED + 0.26 + 0.06 * i for i in range(TIT["s2"].n)]
    t["r"] = [RESP + 0.042 * i for i in range(TIT["r"].n)]
    return t


def glyph_times(times):
    """(G,) set time per global glyph index (GROUP_ORDER)"""
    out = []
    for key in GROUP_ORDER:
        out += list(times[key])
    return np.array(out, np.float64)


def draw_titulus(P, key, T, times, flash_col=art.WHITEGOLD, gold=0.0, glow=0.0):
    """letters whose flip has passed its midpoint are in the cartoon; a freshly set letter flashes"""
    tit = TIT[key]
    ts = times.get(key)
    c = P.c
    for ctx in P.all:
        ctx.save()
        ctx.select_font_face(tit.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(tit.fsize)
    for g in tit.glyphs:
        k = g["idx"]
        if k < 0:
            continue
        fl = 0.0
        if ts is not None:
            if T < ts[k] + FLIP_DUR * 0.5:
                continue
            fl = clamp(1.0 - (T - ts[k] - FLIP_DUR * 0.5) / 0.45)
        colr = tit.dot_color if g["kind"] in ("dot", "leaf", "cross") else tit.color
        colr = art.mixc(colr, flash_col, 0.85 * fl)
        c.set_source_rgba(colr[0], colr[1], colr[2], 1.0)
        c.new_path()
        tit._glyph(c, g)
        p = c.copy_path()
        c.fill()
        for name, v in (("gold", gold * (1 - fl)), ("glow", glow + 0.9 * fl), ("fine", 1.0)):
            ctx = P.ctx.get(name)
            if ctx is None:
                continue
            ctx.new_path()
            ctx.append_path(p)
            ctx.set_source_rgba(v, v, v, 1.0)
            ctx.fill()
    for ctx in P.all:
        ctx.restore()


def draw_labels(ctx):
    """glyph labels for Layout.flow(groups=): red = (global glyph index + 1) / 255, no antialiasing"""
    ctx.save()
    ctx.set_antialias(cairo.ANTIALIAS_NONE)
    base = 0
    for key in GROUP_ORDER:
        tit = TIT[key]
        ctx.save()
        ctx.select_font_face(tit.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(tit.fsize)
        for g in tit.glyphs:
            if g["idx"] < 0:
                continue
            ctx.set_source_rgb((base + g["idx"] + 1) / 255.0, 0, 0)
            ctx.new_path()
            tit._glyph(ctx, g)
            ctx.set_line_width(0.5 * tit.size)
            ctx.stroke_preserve()
            ctx.fill()
        ctx.restore()
        base += tit.n
    ctx.restore()


# ============================================================ the shattered world (Lutie's objection)
def _frag_seeds():
    rng = np.random.default_rng(77)
    cx, cy, R = ORB_L
    pts = []
    while len(pts) < 17:
        a = rng.uniform(0, 2 * math.pi)
        r = R * math.sqrt(rng.uniform(0.0, 0.92))
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


FRAG = _frag_seeds()


def _clip_poly(poly, mx, my, nx, ny, gap):
    """keep the part of the polygon on the seed's side of the bisector (minus a half-gap)"""
    out = []
    ln = math.hypot(nx, ny) + 1e-9
    nx, ny = nx / ln, ny / ln

    def side(p):
        return (p[0] - mx) * nx + (p[1] - my) * ny + gap

    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        sa, sb = side(a), side(b)
        if sa <= 0:
            out.append(a)
        if (sa <= 0) != (sb <= 0):
            t = sa / (sa - sb)
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def _cells():
    """the Voronoi cells of the fragment seeds (polygons, with a gap between them)"""
    cx, cy, R = ORB_L
    out = []
    for i, (fx_, fy_) in enumerate(FRAG):
        poly = [(cx - R - 5, cy - R - 5), (cx + R + 5, cy - R - 5), (cx + R + 5, cy + R + 5), (cx - R - 5, cy + R + 5)]
        for j, (gx, gy) in enumerate(FRAG):
            if j != i and poly:
                poly = _clip_poly(poly, (fx_ + gx) / 2, (fy_ + gy) / 2, gx - fx_, gy - fy_, 3.5)
        out.append(poly)
    return out


CELLS = _cells()


def frag_of(x, y):
    """fragment index of points (screen-at-81.2 coords), -1 outside the orb"""
    cx, cy, R = ORB_L
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    d = (x[..., None] - FRAG[:, 0]) ** 2 + (y[..., None] - FRAG[:, 1]) ** 2
    f = np.argmin(d, axis=-1)
    return np.where(np.hypot(x - cx, y - cy) <= R + 2, f, -1)


def frag_offsets(T, tl):
    """per fragment (F,2): the pieces drift apart; they shudder on 'broken' and jolt apart on 'shattered'"""
    cx, cy, R = ORB_L
    d = FRAG - (cx, cy)
    u = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
    k = 4.0 + 1.5 * math.sin(T * 1.3)
    tb = tl.word("L02", "broken")[0]
    ts = tl.word("L02", "shattered")[0]
    if T > tb:
        k += 3.0 * math.exp(-(T - tb) * 6.0) * math.sin((T - tb) * 40.0)
    if T > ts:
        e = T - ts
        k += 7.0 * (1 - math.exp(-e * 9.0)) + 4.0 * math.exp(-e * 5.0) * math.sin(e * 22.0)
    return u * k


def shattered_orb(P, T):
    cx, cy, R = ORB_L
    P.fill(art._ellipse(cx, cy, R + 8, R + 8), art.hexc("#0C1236"))
    P.save()
    P.clip(lambda ctx: (ctx.new_sub_path(), ctx.arc(cx, cy, R, 0, 2 * math.pi)))
    for i, poly in enumerate(CELLS):
        if not poly:
            continue
        colr = PASTELS[i % len(PASTELS)]
        flick = 0.85 + 0.15 * math.sin(T * (5 + i % 4) + i)
        P.fill(lambda ctx, poly=poly: ([(ctx.move_to if j == 0 else ctx.line_to)(*p) for j, p in enumerate(poly)],
                                       ctx.close_path()), art.mulc(colr, flick), glow=0.55)
    P.restore()
    # the continents it used to show, faintly across the broken pieces
    for path in _continents(cx, cy, R):
        P.stroke(path, S["lapis_d"], 4.0, a=0.45)


def _continents(cx, cy, R):
    """three soft landmasses of an orbis terrarum (no straight lines, nothing cross-like)"""
    blobs = [((-0.05, -0.42), (0.62, 0.34), 0.0, 3), ((-0.42, 0.3), (0.3, 0.36), 0.6, 5), ((0.4, 0.3), (0.28, 0.38), -0.5, 7)]
    out = []
    for (u, v), (a, b), rot, seed in blobs:
        rng = np.random.default_rng(seed)
        k = 14
        amp = rng.uniform(0.82, 1.12, k)
        pts = []
        for i in range(k):
            t = i * 2 * math.pi / k
            x, y = a * amp[i] * math.cos(t), b * amp[i] * math.sin(t)
            xr = x * math.cos(rot) - y * math.sin(rot)
            yr = x * math.sin(rot) + y * math.cos(rot)
            pts.append((cx + R * (u + xr), cy + R * (v + yr)))
        out.append(lambda ctx, pts=pts: smooth(ctx, pts, close=True))
    return out


# ============================================================ the true world (in gold, behind the smiling one)
def gold_orb(P, T):
    cx, cy, R = ORB_R
    P.fill(art._ellipse(cx, cy, R + 7, R + 7), INK)
    P.fill(art._ellipse(cx, cy, R, R), S["gold_d"], gold=1.0)
    P.stroke(lambda ctx: (ctx.new_sub_path(), ctx.arc(cx, cy, R * 0.9, 0, 2 * math.pi)), S["gold_l"], 7.0, gold=1.0)
    for path in _continents(cx, cy, R * 0.86):
        P.fill(path, S["gold_l"], gold=1.0)
        P.stroke(path, S["gold_d"], 4.0, gold=1.0)
    rng = np.random.default_rng(9)
    pts = [(cx + R * 0.62 * math.cos(a), cy + R * 0.62 * math.sin(a)) for a in rng.uniform(0, 2 * math.pi, 9)]
    P.fill(art._circles(pts, 4.5), art.PEARL, fine=1.0)


# ============================================================ the smiling slop world (its own layer)
def smile_extent():
    R = ORB_R[2]
    return (2 * R + 2 * SMILE_PAD, 2 * R + 2 * SMILE_PAD)


def smile_origin():
    """chart (world_D) position of the smile layer's (0, 0)"""
    R = ORB_R[2]
    return (ORB_R[0] - R - SMILE_PAD + OX, ORB_R[1] - R - SMILE_PAD + OY)


def smile_painter(T, cracks=None, smile=1.0):
    R = ORB_R[2]
    c0 = R + SMILE_PAD

    def paint(P):
        g = cairo.RadialGradient(c0 - R * 0.35, c0 - R * 0.4, R * 0.05, c0, c0, R)
        for off, h in ((0.0, "#FFF1F4"), (0.35, "#F2C9D6"), (0.75, "#B9DDEB"), (1.0, "#8FB9D6")):
            r_, g_, b_ = art.hexc(h)
            g.add_color_stop_rgb(off, r_, g_, b_)
        P.fill(art._ellipse(c0, c0, R, R), g, glow=0.45)
        for sx in (-1, 1):
            P.fill(art._ellipse(c0 + sx * R * 0.3, c0 - R * 0.18, R * 0.075, R * 0.12), art.hexc("#2A2233"))
            P.fill(art._ellipse(c0 + sx * R * 0.3 - R * 0.02, c0 - R * 0.23, R * 0.025, R * 0.035), art.WHITE,
                   glow=0.6)
            P.fill(art._ellipse(c0 + sx * R * 0.5, c0 + R * 0.12, R * 0.12, R * 0.07), art.hexc("#F4A7B9"), glow=0.3)
        w = R * (0.5 + 0.06 * smile)

        def mouth(ctx):
            ctx.move_to(c0 - w, c0 + R * 0.1)
            ctx.curve_to(c0 - w * 0.6, c0 + R * (0.52 + 0.08 * smile), c0 + w * 0.6, c0 + R * (0.52 + 0.08 * smile),
                         c0 + w, c0 + R * 0.1)
            ctx.curve_to(c0 + w * 0.55, c0 + R * 0.34, c0 - w * 0.55, c0 + R * 0.34, c0 - w, c0 + R * 0.1)
            ctx.close_path()
        P.fill(mouth, art.hexc("#2A2233"))
        P.fill(lambda ctx: (ctx.save(), ctx.translate(c0, c0 + R * 0.3), ctx.scale(w * 0.62, R * 0.07),
                            ctx.new_sub_path(), ctx.arc(0, 0, 1, 0, math.pi), ctx.restore()), art.WHITE, glow=0.5)
        P.fill(art._ellipse(c0 - R * 0.42, c0 - R * 0.52, R * 0.22, R * 0.11), art.hexc("#FFFFFF"), glow=0.8)
        if cracks:
            for path in cracks:
                P.stroke(lambda ctx, path=path: [(ctx.move_to if j == 0 else ctx.line_to)(*p)
                                                 for j, p in enumerate(path)], INK, 3.4)
    return paint


def smile_layout():
    """machine-perfect: a square grid of identical tiny tiles (no andamento, no hand)"""
    ex, ey = smile_extent()
    R = ORB_R[2]
    c0 = R + SMILE_PAD
    lay = mz.Layout.grid(tile=6.5, rect=(0, 0, ex, ey), jitter=0.0, seed=5, extent=(int(ex), int(ey)), brick=False)
    return lay.keep(lambda x, y: np.hypot(x - c0, y - c0) <= R - 1.0)


# ============================================================ the Hypermind as a flat icon (upper register)
def icon_painter(T, look_down=1.0, wake=1.0, glow=0.35):
    cx, cy, sc = ICON

    def paint(P):
        art.seraph_icon(P, cx, cy, sc, look_down=look_down, wake=wake, glow=glow)
    return paint


# ============================================================ the wall
def rainbow(P, y, h, x0, x1):
    cols = [S["porphyry"], S["terracotta"], art.hexc("#E0B04A"), S["marble"], S["lapis_l"], S["lapis_d"]]
    k = h / len(cols)
    for i, cc in enumerate(cols):
        P.fill(lambda ctx, i=i: ctx.rectangle(x0, y + i * k, x1 - x0, k + 0.5), cc)


def pearls(P, y, x0, x1, step=16.0, r=3.2):
    pts = [(x, y) for x in np.arange(x0, x1, step)]
    P.fill(art._circles(pts, r), art.PEARL, fine=1.0)


def wall_painter(st):
    """st: dict(T, times, figs(P), icon(P))"""
    T = st["T"]

    def paint(P):
        P.push(mt.affine(1, 0, 0, 1, OX, OY))
        X0, X1 = -OX - 10, WD[0] - OX + 10
        # ---- top: frame + the question
        P.fill(lambda ctx: ctx.rectangle(X0, -OY - 10, X1 - X0, OY + 24), S["lapis_d"])
        pearls(P, 6.0, X0, X1)
        P.fill(lambda ctx: ctx.rectangle(X0, 14, X1 - X0, 94), S["lapis_d"])
        P.stroke(lambda ctx: (ctx.move_to(X0, 17), ctx.line_to(X1, 17)), S["gold"], 5.0, gold=1.0)
        rainbow(P, 108, 12, X0, X1)
        # ---- upper register: night, stars, the icon, VIDETVR QVOD NON / SED CONTRA
        P.fill(lambda ctx: ctx.rectangle(X0, 120, X1 - X0, 300), art.NIGHT)
        rng = np.random.default_rng(3)
        stars = [(rng.uniform(X0, X1), rng.uniform(130, 410)) for _ in range(90)]
        stars = [p for p in stars if abs(p[0] - 960) > 430 or abs(p[1] - 262) > 150]
        P.fill(lambda ctx: [(ctx.move_to(x - 4, y), ctx.line_to(x, y - 4), ctx.line_to(x + 4, y),
                             ctx.line_to(x, y + 4), ctx.close_path()) for x, y in stars], S["star"], gold=0.6,
               fine=1.0)
        st["icon"](P)
        rainbow(P, 420, 12, X0, X1)
        # ---- lower register: the gold ground, the two worlds
        P.fill(lambda ctx: ctx.rectangle(X0, 432, X1 - X0, FLOOR_Y - 432), S["gold"], gold=1.0)
        niche(P, LUT[0], 448.0, FLOOR_Y, 318.0)
        niche(P, CRO[0] - 20.0, 448.0, FLOOR_Y, 336.0)
        gold_orb(P, T)
        shattered_orb(P, T)
        # ---- floor: a meadow strip, then the bottom band
        P.fill(lambda ctx: ctx.rectangle(X0, FLOOR_Y - 4, X1 - X0, 20), S["malachite"])
        rng = np.random.default_rng(8)
        fl = [(rng.uniform(X0, X1), rng.uniform(FLOOR_Y + 1, FLOOR_Y + 12)) for _ in range(90)]
        P.fill(art._circles(fl, 3.4), S["marble"], fine=1.0)
        P.fill(lambda ctx: ctx.rectangle(X0, FLOOR_Y + 16, X1 - X0, 10), INK)
        P.fill(lambda ctx: ctx.rectangle(X0, FLOOR_Y + 26, X1 - X0, 1200 - FLOOR_Y), S["lapis_d"])
        pearls(P, 1128.0, X0, X1)
        # ---- tituli
        times = st["times"]
        draw_titulus(P, "q_red", T, {}, glow=0.15)
        draw_titulus(P, "q", T, {}, gold=1.0)
        for key in ("v1", "v2", "s1", "s2", "r"):
            draw_titulus(P, key, T, times, gold=1.0)
        draw_titulus(P, "pr", T, times, glow=0.2)
        # ---- the disputants
        st["figs"](P)
        P.pop()
    return paint


def niche(P, cx, top, bottom, w):
    """an arched niche of night-blue with gold stars and a jewelled gold frame behind a disputant"""
    r = w / 2

    def arch(ctx, g=0.0):
        ctx.move_to(cx - r + g, bottom)
        ctx.line_to(cx - r + g, top + r)
        ctx.arc(cx, top + r, r - g, math.pi, 2 * math.pi)
        ctx.line_to(cx + r - g, bottom)
        ctx.close_path()
    P.fill(arch, INK)
    P.fill(lambda ctx: arch(ctx, 6.0), S["lapis_d"])
    rng = np.random.default_rng(int(cx))
    st = []
    while len(st) < 26:
        x, y = rng.uniform(cx - r + 20, cx + r - 20), rng.uniform(top + 30, bottom - 40)
        if y < top + r and math.hypot(x - cx, y - top - r) > r - 24:
            continue
        st.append((x, y))
    P.fill(lambda ctx: [(ctx.move_to(x - 4, y), ctx.line_to(x, y - 4), ctx.line_to(x + 4, y), ctx.line_to(x, y + 4),
                         ctx.close_path()) for x, y in st], S["star"], gold=0.6, fine=1.0)
    P.stroke(lambda ctx: arch(ctx, 3.0), S["gold"], 9.0, gold=1.0)
    n = int((math.pi * r + 2 * (bottom - top - r)) / 18)
    pts = []
    for i in range(n):
        u = i / n * (math.pi * r + 2 * (bottom - top - r))
        L1 = bottom - top - r
        if u < L1:
            pts.append((cx - r + 3, bottom - u))
        elif u < L1 + math.pi * r:
            a = math.pi + (u - L1) / r
            pts.append((cx + (r - 3) * math.cos(a), top + r + (r - 3) * math.sin(a)))
        else:
            pts.append((cx + r - 3, top + r + (u - L1 - math.pi * r)))
    P.fill(art._circles(pts, 2.6), art.PEARL, fine=1.0)


# ============================================================ figures
def _blend(T, keys, fade=0.35):
    """keys: [(t_start, pose)] -> the pose at T, cross-faded over `fade` seconds ({pose: weight} blends)"""
    j = 0
    for i, (t0, p) in enumerate(keys):
        if T >= t0:
            j = i
    if j == 0:
        return keys[0][1]
    t0, p = keys[j]
    prev = keys[j - 1][1]
    k = smoothstep(t0, t0 + fade, T)
    if k >= 1.0 or prev == p:
        return p
    return {prev: 1.0 - k, p: k}


def figures_painter(T, tl, figure, guide=False):
    tb = tl.word("L02", "world")[0]
    tsl = tl.word("L02", "The", 1)[0]
    tf = tl.word("C02", "fall")[0]
    tsu = tl.word("C03", "summon")[0]
    if guide:
        lut_pose, cro_pose = "point", "bless_high"
    else:
        lut_pose = _blend(T, [(0, "object"), (tb - 0.15, "point"), (tsl - 0.1, "speak"), (86.5, "stand")])
        cro_pose = _blend(T, [(0, "stand"), (SED - 0.15, "speak"), (tf - 0.25, "point"), (tf + 0.9, "speak"),
                              (RESP - 0.2, "bless"), (tsu - 0.25, "bless_high")])
    lut_look = (0.8, -0.1) if (T < tb or T > tsl + 0.6) else (0.6, 0.35)
    cro_look = (-0.7, 0.0)
    if tf - 0.4 < T < tf + 1.5:
        cro_look = (-0.4, 0.4)
    if T > RESP:
        cro_look = (-0.25, -0.2)
    lut_expr = "worried" if T < 86.6 else {"worried": 0.4, "awe": 0.6}
    cro_expr = "serene" if T < SED else ("stern" if T < RESP else {"serene": 0.5, "stern": 0.5})

    def paint(P):
        x, y, h = LUT
        P.figure(figure("lutie"), x, y, h, pose=lut_pose, t=T, mouth=tl.mouth("LUTIE", T), look=lut_look,
                 expr=lut_expr, facing=0.45, tile=4.4, fine_all=True)
        x, y, h = CRO
        P.figure(figure("crocus"), x, y, h, pose=cro_pose, t=T, mouth=tl.mouth("CROCUS", T), look=cro_look,
                 expr=cro_expr, facing=-0.4, tile=4.4, fine_all=True)
    return paint


def crocus_hand(T, figure):
    """world_D chart position of Crocus's raised blessing hand"""
    x, y, h = CRO
    a = figure("crocus").anchors(x, y, h, "bless_high", T, -0.4)
    hx, hy = a.get("hand_active") or min((a["hand_r"], a["hand_l"]), key=lambda p: p[1])   # the blessing hand
    return hx + OX, hy + OY

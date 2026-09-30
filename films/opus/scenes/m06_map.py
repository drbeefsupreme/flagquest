"""m06 NATIONS (106.5-113.7): the round mappa mundi (orbis terrarum, a T-O map set in tesserae) floating in the
Ocean. At K04 it cracks along the T, then along every national border, then again and again (recursive), the
pieces drifting apart like islands; each last piece becomes a tiny kingdom with one crowned person on it.

World = panel design units (the 1920x1080 wall at zoom 1). The map's tiles are one fixed flow layout; fragments
are nodes of an m06_tess.Shatter; tiles split into four (quadtree LOD) whenever they grow on screen, sampling
the cartoon pyramid (2.5 px/unit) for finer detail. The Ocean is a second fixed layout with analytic colours.
"""
import math

import cairo
import cv2
import numpy as np

from vx import W, H, Canvas, set_color, circle, ellipse, poly, smooth, value_noise2, hash01, smootherstep
from vx import icon
from vx import mosaic as mz
from vx.canvas import col, text

from scenes import m06_tess as tx

C0 = np.array([960.0, 574.0])
R_LAND, R_SEA, R_RIM = 396.0, 438.0, 460.0
PPU = 2.5                                   # cartoon px per world unit
BOX = (C0[0] - R_RIM - 8, C0[1] - R_RIM - 8, 2 * R_RIM + 16, 2 * R_RIM + 16)   # x0, y0, w, h (world)
SEA_RECT = (-560.0, -420.0, 3040.0, 1920.0)  # the Ocean layout (world x0, y0, w, h)
FONT = "P052"

S = mz.SMALTI
NATIONS = [  # (continent, name, offset from C0, colour)
    (0, "SCYTHIA", (-285, -105), "#8C6A4E"), (0, "PERSIS", (-115, -225), "#B5563A"),
    (0, "BABYLON", (-55, -95), "#C98B5A"), (0, "SERES", (105, -300), "#8E3B22"),
    (0, "INDIA", (255, -175), "#A0505E"), (0, "ARABIA", (165, -70), "#D2A878"),
    (1, "SARMATIA", (-300, 70), "#6E8B5B"), (1, "GERMANIA", (-150, 95), "#2E5E47"),
    (1, "GRAECIA", (-48, 95), "#8DB39A"), (1, "GALLIA", (-255, 235), "#4F8A63"),
    (1, "ITALIA", (-85, 245), "#A7B98A"), (1, "HISPANIA", (-160, 345), "#3E7C7A"),
    (2, "AEGYPTVS", (62, 92), "#CDA66E"), (2, "LIBYA", (205, 112), "#B5714A"),
    (2, "AETHIOPIA", (315, 185), "#7A3B2E"), (2, "MAVRETANIA", (120, 300), "#D8C3A0"),
    (2, "GARAMANTES", (255, 292), "#A45E3C"),
]
CONT = ["ASIA", "EVROPA", "AFRICA"]


# ============================================================ geometry of the T-O
def bar_y(x):
    return C0[1] + 5.0 * np.sin(np.asarray(x) / 41.0) + 3.0 * np.sin(np.asarray(x) / 17.0 + 1.0)


def stem_x(y):
    return C0[0] + 7.0 * np.sin(np.asarray(y) / 37.0)


def stem_w(y):
    q = np.clip((np.asarray(y) - C0[1]) / R_LAND, 0, 1)
    return 18.0 + 30.0 * np.sin(np.pi * q) ** 0.8


def coast_r(th):
    return R_LAND + 6.0 * np.sin(th * 7.0 + 0.5) + 4.0 * np.sin(th * 13.0 + 2.0)


# ============================================================ label rasters (setup)
def build_labels():
    """per-pixel (2.5 px/unit over BOX): kind (0 land, 1 water in T, 2 ocean ring, 3 rim, -1 outside),
    continent (0 Asia, 1 Europe, 2 Africa) and nation index for every pixel of the orbis."""
    x0, y0, bw, bh = BOX
    w, h = int(bw * PPU), int(bh * PPU)
    xs = x0 + (np.arange(w) + 0.5) / PPU
    ys = y0 + (np.arange(h) + 0.5) / PPU
    X, Y = np.meshgrid(xs, ys)
    dx, dy = X - C0[0], Y - C0[1]
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    kind = np.full((h, w), -1, np.int8)
    kind[r < R_RIM] = 3
    kind[r < R_SEA] = 2
    land = r < coast_r(th)
    by = bar_y(X)
    bar = np.abs(Y - by) < 8.0
    stem = (Y > by) & (np.abs(X - stem_x(Y)) < stem_w(Y) / 2)
    water = bar | stem
    # islands (world offsets from C0, radii)
    isl = [(-52, 175, 11), (38, 250, 9), (-8, 318, 8), (12, 120, 7),          # Sicily, Crete, Sardinia, Cyprus
           (-322, 262, 17), (-352, 222, 11), (402, 30, 13), (-150, -415, 9)]   # Britannia, Hibernia, Taprobane, Thule
    for ix, iy, ir in isl:
        rr = np.hypot(dx - ix, dy - iy) * (1 + 0.15 * np.sin(np.arctan2(dy - iy, dx - ix) * 3 + ix))
        m = rr < ir
        land |= m & (r < R_SEA)
        water &= ~m
    land &= ~water
    kind[land] = 0
    kind[(r < coast_r(th)) & water] = 1
    cont = np.where(Y < by, 0, np.where(X < stem_x(Y), 1, 2)).astype(np.int8)
    # nations: noisy voronoi inside each continent
    nat = np.zeros((h, w), np.int16)
    best = np.full((h, w), 1e9)
    for j, (c, _, (ox, oy), _) in enumerate(NATIONS):
        nx = value_noise2(X / 95.0, Y / 95.0, 100 + j) * 30 + value_noise2(X / 31.0, Y / 31.0, 200 + j) * 9
        d = np.hypot(dx - ox, dy - oy) + nx
        d = np.where(cont == c, d, 1e9)
        m = d < best
        best[m] = d[m]
        nat[m] = j
    return dict(kind=kind, cont=cont, nat=nat, X=X, Y=Y, r=r, th=th, w=w, h=h)


# ============================================================ drawing the cartoon
def _tower(ctx, x, y, s, crown=True, lay="color"):
    """a crowned city: marble tower with crenellations, a door, gold crown above. (x, y) = base centre."""
    colr = lay == "color"
    wht = (1, 1, 1)
    if lay in ("color", "fine"):
        set_color(ctx, S["marble"] if colr else wht)
        ctx.rectangle(x - 0.5 * s, y - 1.1 * s, s, 1.1 * s)
        ctx.rectangle(x - 0.75 * s, y - 0.55 * s, 1.5 * s, 0.55 * s)
        for k in range(3):
            ctx.rectangle(x - 0.5 * s + k * 0.4 * s, y - 1.3 * s, 0.2 * s, 0.22 * s)
        ctx.fill()
    if colr:
        set_color(ctx, S["porphyry"])
        ctx.rectangle(x - 0.12 * s, y - 0.3 * s, 0.24 * s, 0.3 * s)
        ctx.rectangle(x - 0.08 * s, y - 0.85 * s, 0.16 * s, 0.2 * s)
        ctx.fill()
        set_color(ctx, S["ink"])
        ctx.set_line_width(0.07 * s)
        ctx.rectangle(x - 0.5 * s, y - 1.1 * s, s, 1.1 * s)
        ctx.rectangle(x - 0.75 * s, y - 0.55 * s, 1.5 * s, 0.55 * s)
        ctx.stroke()
    if crown and lay in ("color", "gold", "fine"):
        set_color(ctx, S["gold"] if colr else wht)
        cy = y - 1.38 * s
        cw = 0.46 * s
        poly(ctx, [(x - cw, cy), (x - cw, cy - 0.34 * s), (x - cw / 2, cy - 0.16 * s), (x, cy - 0.44 * s),
                   (x + cw / 2, cy - 0.16 * s), (x + cw, cy - 0.34 * s), (x + cw, cy)])
        ctx.fill()
        if colr:
            set_color(ctx, S["porphyry_l"])
            circle(ctx, x, cy - 0.09 * s, 0.07 * s)
            ctx.fill()


def _ziggurat(ctx, x, y, s, lay="color"):
    if lay not in ("color", "fine"):
        return
    for k in range(5):
        w = s * (1.0 - 0.17 * k)
        set_color(ctx, (S["terracotta"] if k % 2 == 0 else S["ochre_red"]) if lay == "color" else (1, 1, 1))
        ctx.rectangle(x - w / 2, y - (k + 1) * 0.26 * s, w, 0.26 * s)
        ctx.fill()


def _mountains(ctx, x, y, n, s, ang=0.0, tone="#4B5160"):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    for k in range(n):
        cx = (k - (n - 1) / 2) * s * 0.9
        set_color(ctx, tone)
        ctx.move_to(cx - s * 0.55, 0)
        ctx.curve_to(cx - s * 0.4, -s * 0.9, cx + s * 0.4, -s * 0.9, cx + s * 0.55, 0)
        ctx.close_path()
        ctx.fill()
        set_color(ctx, S["marble"])
        ctx.set_line_width(s * 0.12)
        ctx.move_to(cx - s * 0.25, -s * 0.45)
        ctx.curve_to(cx - s * 0.15, -s * 0.66, cx, -s * 0.7, cx + s * 0.08, -s * 0.66)
        ctx.stroke()
    ctx.restore()


def _tree(ctx, x, y, s):
    set_color(ctx, "#6B4A2E")
    ctx.rectangle(x - 0.08 * s, y - 0.5 * s, 0.16 * s, 0.5 * s)
    ctx.fill()
    set_color(ctx, S["malachite"])
    circle(ctx, x, y - 0.75 * s, 0.38 * s)
    ctx.fill()
    set_color(ctx, S["verdigris"])
    circle(ctx, x - 0.1 * s, y - 0.85 * s, 0.16 * s)
    ctx.fill()


def _river(ctx, pts, w0, w1):
    n = len(pts)
    for i in range(n - 1):
        set_color(ctx, S["lapis_l"])
        ctx.set_line_width(w0 + (w1 - w0) * i / max(1, n - 2))
        ctx.move_to(*pts[i])
        ctx.line_to(*pts[i + 1])
        ctx.stroke()


def _fish(ctx, x, y, s, flip=1):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(flip, 1)
    set_color(ctx, S["silver"])
    ellipse(ctx, 0, 0, s, 0.38 * s)
    ctx.fill()
    poly(ctx, [(-0.9 * s, 0), (-1.45 * s, -0.4 * s), (-1.45 * s, 0.4 * s)])
    ctx.fill()
    set_color(ctx, S["ink"])
    circle(ctx, 0.55 * s, -0.06 * s, 0.09 * s)
    ctx.fill()
    ctx.restore()


def _curls(ctx, x0, y0, x1, y1, step, rad, lw, tone, seed=0, clip_fn=None):
    """rows of little wave arcs (the mosaicist's sea)."""
    set_color(ctx, tone)
    ctx.set_line_width(lw)
    r = 0
    y = y0
    while y < y1:
        x = x0 + (step * 0.5 if r % 2 else 0) + hash01(r, seed) * step * 0.3
        while x < x1:
            if clip_fn is None or clip_fn(x, y):
                ctx.new_sub_path()
                ctx.arc(x, y, rad, math.pi * 1.08, math.pi * 1.92)
            x += step
        ctx.stroke()
        y += step * 0.62
        r += 1


def draw_titulus_arc(ctx, s, cx, cy, rad, a0, a1, size, colr, font=FONT):
    """letters centred on an arc (a0 -> a1 radians, measured like cairo: 0 = +x, clockwise positive)."""
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    adv = [ctx.text_extents(ch).x_advance + 0.18 * size for ch in s]
    total = sum(adv) - 0.18 * size
    span = total / rad
    a = (a0 + a1) / 2 - span / 2
    set_color(ctx, colr)
    for ch, ad in zip(s, adv):
        e = ctx.text_extents(ch)
        am = a + (e.x_advance / 2) / rad
        ctx.save()
        ctx.translate(cx + rad * math.cos(am), cy + rad * math.sin(am))
        ctx.rotate(am + math.pi / 2)
        ctx.move_to(-e.x_advance / 2, size * 0.36)
        ctx.show_text(ch)
        ctx.restore()
        a += ad / rad


class MapArt:
    """the static cartoon of the orbis (2.5 px/unit), its gold/fine masks, and region labels."""

    def __init__(self):
        L = build_labels()
        self.L = L
        h, w = L["h"], L["w"]
        rgb = np.zeros((h, w, 3), np.float32)
        kind, nat, cont = L["kind"], L["nat"], L["cont"]
        pal = np.array([col(c) for *_, c in NATIONS], np.float32)
        rgb[:] = pal[nat]
        # varied land: slight mottling so large fields don't read flat
        mot = value_noise2(L["X"] / 23.0, L["Y"] / 23.0, 7).astype(np.float32)
        rgb *= (1.0 + 0.06 * mot)[..., None]
        sea = np.array(col(S["lapis"]), np.float32)
        rgb[kind == 1] = sea
        rgb[kind == 2] = sea
        rgb[kind == 3] = col(S["gold"])
        rgb[kind == -1] = col(S["night_vault"])
        self.rgb0 = np.clip(rgb, 0, 1)
        # capitals: the most interior point of every nation
        self.capitals = []
        for j in range(len(NATIONS)):
            m = ((nat == j) & (kind == 0)).astype(np.uint8)
            dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            iy, ix = np.unravel_index(np.argmax(dt), dt.shape)
            self.capitals.append((BOX[0] + (ix + 0.5) / PPU, BOX[1] + (iy + 0.5) / PPU + 14))
        self.images = {k: self._render(k) for k in ("color", "gold", "fine")}

    def _render(self, layer):
        L = self.L
        h, w = L["h"], L["w"]
        cv = Canvas(s=PPU, w=w, h=h)
        ctx = cv.ctx
        ctx.translate(-BOX[0], -BOX[1])
        colr = layer == "color"
        if colr:
            cv.paint_array(self.rgb0, BOX[0], BOX[1])
        self.draw(ctx, layer)
        rgba = cv.rgba()
        if colr:
            return rgba[..., :3].copy()
        return rgba[..., 3].copy()

    def draw(self, ctx, layer):
        colr = layer == "color"
        wht = (1, 1, 1)
        L = self.L
        cx, cy = C0
        if colr:
            # borders between nations (from the label raster) + coastlines
            nat, kind = L["nat"], L["kind"]
            e = np.zeros(nat.shape, bool)
            e[:, 1:] |= (nat[:, 1:] != nat[:, :-1]) & (kind[:, 1:] == 0) & (kind[:, :-1] == 0)
            e[1:, :] |= (nat[1:, :] != nat[:-1, :]) & (kind[1:, :] == 0) & (kind[:-1, :] == 0)
            e = cv2.dilate(e.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            land = (kind == 0).astype(np.uint8)
            coast = (cv2.dilate(land, np.ones((5, 5), np.uint8)) > 0) & (land == 0)
            shore = (cv2.erode(land, np.ones((9, 9), np.uint8)) == 0) & (land > 0)
            ov = np.zeros(nat.shape + (4,), np.float32)
            for m, c, a in ((shore, S["bone"], 0.55), (e, "#4A1A20", 1.0), (coast, S["ink"], 1.0)):
                ov[m] = (*[v * a for v in col(c)], a)
            ctx.save()
            ctx.identity_matrix()
            ctx.set_source_surface(_surf(ov), 0, 0)
            ctx.paint()
            ctx.restore()
            # ocean ring waves + fish
            _curls(ctx, cx - R_SEA, cy - R_SEA, cx + R_SEA, cy + R_SEA, 30, 8.5, 2.6, S["silver"], 3,
                   lambda x, y: coast_r(math.atan2(y - cy, x - cx)) + 11 < math.hypot(x - cx, y - cy) < R_SEA - 9)
            for fx, fy, fs, fl in ((-250, -345, 9, 1), (330, -250, 8, -1), (380, 170, 9, -1), (-120, 405, 8, 1),
                                   (-395, -40, 7, 1)):
                _fish(ctx, cx + fx, cy + fy, fs, fl)
            # T waters: small waves
            _curls(ctx, cx - R_LAND, cy - 3, cx + R_LAND, cy + R_LAND, 22, 5.5, 1.8, S["lapis_l"], 5,
                   lambda x, y: float(abs(x - stem_x(y))) < float(stem_w(y)) / 2 - 7 and y > cy + 14)
            # mountains, rivers, trees
            for mx, my, n, s_, a_, tn in ((-210, -150, 4, 17, -0.4, "#5A3B2C"), (-40, -170, 3, 15, 0.2, "#6E4A3A"),
                                          (60, -230, 4, 16, 0.1, "#5A2B20"), (-190, 150, 4, 15, 0.3, "#23473A"),
                                          (-120, 190, 3, 14, -0.2, "#4E6A48"), (235, 250, 4, 15, 0.0, "#6B3A28"),
                                          (300, -110, 3, 16, 0.2, "#6A2E38")):
                _mountains(ctx, cx + mx, cy + my, n, s_, a_, tn)
            _river(ctx, [(cx - 170, cy - 250), (cx - 120, cy - 170), (cx - 100, cy - 90), (cx - 70, cy - 12)], 3, 7)
            _river(ctx, [(cx - 70, cy - 230), (cx - 30, cy - 160), (cx - 25, cy - 90), (cx - 20, cy - 10)], 3, 6)
            _river(ctx, [(cx + 300, cy - 280), (cx + 280, cy - 200), (cx + 330, cy - 120), (cx + 385, cy - 70)], 3, 7)
            _river(ctx, [(cx - 330, cy + 120), (cx - 230, cy + 60), (cx - 120, cy + 40), (cx - 30, cy + 60)], 3, 6)
            _river(ctx, [(cx + 120, cy + 210), (cx + 90, cy + 140), (cx + 60, cy + 60), (cx + 40, cy + 10)], 3, 8)
            for k in range(26):
                a = hash01(k, 41) * 2 * math.pi
                rr = 60 + hash01(k, 42) * 300
                x, y = cx + rr * math.cos(a), cy + rr * math.sin(a) + 20
                if y > cy + 25 and abs(x - float(stem_x(y))) > 40:
                    _tree(ctx, x, y, 12 + 5 * hash01(k, 43))
            # pyramids of Egypt
            for px_, s_ in ((cx + 95, 12), (cx + 118, 9)):
                set_color(ctx, S["bone"])
                poly(ctx, [(px_ - s_, cy + 150), (px_ + s_, cy + 150), (px_, cy + 150 - 1.3 * s_)])
                ctx.fill()
                set_color(ctx, S["marble_d"])
                poly(ctx, [(px_, cy + 150), (px_ + s_, cy + 150), (px_, cy + 150 - 1.3 * s_)])
                ctx.fill()
        # the jewelled rim of the O: gold band with pearls and gems
        if layer in ("color", "gold"):
            set_color(ctx, S["gold"] if colr else wht)
            ctx.set_line_width(R_RIM - R_SEA)
            circle(ctx, cx, cy, (R_RIM + R_SEA) / 2)
            ctx.stroke()
        if colr:
            set_color(ctx, S["ink"])
            ctx.set_line_width(2.5)
            circle(ctx, cx, cy, R_SEA)
            ctx.stroke()
            circle(ctx, cx, cy, R_RIM)
            ctx.stroke()
        if layer in ("color", "fine"):
            n = 96
            for k in range(n):
                a = 2 * math.pi * k / n
                x, y = cx + (R_RIM + R_SEA) / 2 * math.cos(a), cy + (R_RIM + R_SEA) / 2 * math.sin(a)
                if k % 2 == 0:
                    set_color(ctx, (S["marble"] if colr else wht))
                    circle(ctx, x, y, 4.2)
                else:
                    set_color(ctx, ((S["porphyry_l"], S["lapis_l"], S["malachite"])[(k // 2) % 3]) if colr else wht)
                    ctx.rectangle(x - 4.5, y - 4.5, 9, 9)
                ctx.fill()
        # Paradise in the East (top): a golden island with the tree of life
        px_, py_ = cx, cy - R_LAND + 52
        if layer in ("color", "gold"):
            set_color(ctx, S["gold"] if colr else wht)
            circle(ctx, px_, py_, 30)
            ctx.fill()
        if colr:
            set_color(ctx, S["ink"])
            ctx.set_line_width(2.5)
            circle(ctx, px_, py_, 30)
            ctx.stroke()
            _tree(ctx, px_, py_ + 17, 30)
        # Jerusalem at the navel of the world: round city, gold walls
        jx, jy = cx, cy - 36
        if layer in ("color", "gold"):
            set_color(ctx, S["gold"] if colr else wht)
            circle(ctx, jx, jy, 25)
            ctx.fill()
        if layer in ("color", "fine"):
            set_color(ctx, S["marble"] if colr else wht)
            circle(ctx, jx, jy, 17)
            ctx.fill()
            for k in range(8):
                a = k * math.pi / 4
                set_color(ctx, S["marble"] if colr else wht)
                ctx.rectangle(jx + 23 * math.cos(a) - 4, jy + 23 * math.sin(a) - 4, 8, 8)
                ctx.fill()
        if colr:
            set_color(ctx, S["porphyry"])
            circle(ctx, jx, jy, 7)
            ctx.fill()
        # capitals: crowned towers; Babylon is the Tower
        for j, (xy, (c, name, _, _)) in enumerate(zip(self.capitals, NATIONS)):
            if name == "BABYLON":
                _ziggurat(ctx, xy[0], xy[1], 44, layer)
            else:
                _tower(ctx, xy[0], xy[1], 24, True, layer)
        # continent names (tituli on the land)
        if layer in ("color", "fine", "gold"):
            for name, (ox, oy), sz in (("ASIA", (40, -150), 66), ("EVROPA", (-200, 178), 50),
                                       ("AFRICA", (205, 208), 50)):
                if colr:
                    text(ctx, name, cx + ox, cy + oy, sz, S["ink"], font=FONT, bold=True, align="center",
                         valign="middle", tracking=0.1, stroke=S["ink"], stroke_w=7.0)
                text(ctx, name, cx + ox, cy + oy, sz, S["gold_l"] if colr else wht, font=FONT, bold=True,
                     align="center", valign="middle", tracking=0.1)


def _surf(rgba_premul):
    from vx.canvas import surface_from_array
    return surface_from_array(rgba_premul)


# ============================================================ the Ocean (analytic colours)
def sea_colour(x, y):
    """deep ocean with rows of wave arcs; returns rgb (N,3) for world points."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    base = np.array(col(S["night_vault"]), np.float32)
    deep = np.array(col(S["lapis_d"]), np.float32)
    k = (0.5 + 0.5 * value_noise2(x / 260.0, y / 260.0, 9)).astype(np.float32)[:, None]
    rgb = base * (1 - k) + deep * k
    step = 86.0
    ry = step * 0.62
    r = np.floor(y / ry + 0.5)
    off = np.where(r % 2 == 1, step * 0.5, 0.0)
    cy_ = r * ry
    cx_ = np.floor((x - off) / step + 0.5) * step + off
    dx, dy = x - cx_, y - cy_
    d = np.hypot(dx, dy)
    up = dy < 1.5
    crest = (np.abs(d - 12.0) < 2.3) & up
    shade = (np.abs(d - 16.0) < 1.8) & up
    rgb = np.where(crest[:, None], np.array(col(S["silver"]), np.float32) * 0.62, rgb)
    rgb = np.where(shade[:, None], np.array(col(S["lapis"]), np.float32), rgb)
    return rgb.astype(np.float32)


def sea_guide(ppu=0.5):
    """a rendering of the sea pattern for Layout.flow (courses hug the wave arcs)."""
    x0, y0, w, h = SEA_RECT
    gw, gh = int(w * ppu), int(h * ppu)
    xs = x0 + (np.arange(gw) + 0.5) / ppu
    ys = y0 + (np.arange(gh) + 0.5) / ppu
    X, Y = np.meshgrid(xs, ys)
    return sea_colour(X.ravel(), Y.ravel()).reshape(gh, gw, 3)


# ============================================================ the panel: layouts, recursive shatter, rendering
VERSION = "map_v2"
TK = {1: 106.97, 2: 107.22, 3: 107.72, 4: 108.12, 5: 108.42}     # generations of cracking (global s)
DRIFT = {1: 70.0, 2: 36.0, 3: 24.0, 4: 18.0, 5: 14.0}
SPIN = {1: 0.025, 2: 0.06, 3: 0.1, 4: 0.14, 5: 0.18}
SPLITS = {3: (2, 3), 4: (2, 3), 5: (2, 2)}
CRACK_LIGHT = np.array([1.0, 0.86, 0.55], np.float32)
HERO_AT = C0 + np.array([-105.0, 292.0])


def _build_cache():
    from scipy.spatial import cKDTree
    art = MapArt()
    L = art.L
    # --- the map's own layout: andamento over a 1 px/unit guide of the whole panel
    colr = art.images["color"]
    x0, y0 = int(BOX[0]), int(BOX[1])
    gw = int(round(colr.shape[1] / PPU))
    small = cv2.resize(colr, (gw, gw), interpolation=cv2.INTER_AREA)
    fine_s = cv2.resize(art.images["fine"], (gw, gw), interpolation=cv2.INTER_AREA) > 0.35
    guide = np.zeros((H, W, 3), np.float32) + np.array(col(S["night_vault"]), np.float32)
    fine = np.zeros((H, W), bool)
    ys, xs = slice(max(0, y0), min(H, y0 + gw)), slice(max(0, x0), min(W, x0 + gw))
    guide[ys, xs] = small[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    fine[ys, xs] = fine_s[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    lay = mz.Layout.flow(guide, tile=8.0, seed=61, fine=fine, fine_tile=4.6)
    rr = np.hypot(lay.xy[:, 0] - C0[0], lay.xy[:, 1] - C0[1]) < R_RIM - 0.5
    lay = lay.keep(lambda x, y: rr)
    # --- the Ocean's layout (bigger world rect)
    sx0, sy0, sw, sh = SEA_RECT
    sea = mz.Layout.flow(sea_guide(0.5), tile=11.0, seed=62, extent=(int(sw), int(sh)))
    sea_xy = sea.xy + np.array([sx0, sy0], np.float32)
    # --- labels at tile centres
    def lab(img, xy):
        ix = np.clip(((xy[:, 0] - BOX[0]) * PPU).astype(int), 0, L["w"] - 1)
        iy = np.clip(((xy[:, 1] - BOX[1]) * PPU).astype(int), 0, L["h"] - 1)
        return img[iy, ix]
    kind = lab(L["kind"], lay.xy)
    nat = lab(L["nat"], lay.xy).astype(np.int64)
    land = kind == 0
    tree = cKDTree(lay.xy[land])
    _, j = tree.query(lay.xy[~land])
    nat[~land] = nat[land][j]
    cont = np.array([NATIONS[k][0] for k in nat], np.int64)
    return dict(color=np.clip(colr * 255 + 0.5, 0, 255).astype(np.uint8),
                gold=np.clip(art.images["gold"] * 255 + 0.5, 0, 255).astype(np.uint8),
                xy=lay.xy, ang=lay.ang, size=lay.size, kind=kind, nat=nat, cont=cont,
                sea_xy=sea_xy, sea_ang=sea.ang, sea_size=sea.size, capitals=np.array(art.capitals))


def _hierarchy(xy, cont, nat, seed=7):
    N = len(xy)
    tn = [np.zeros(N, np.int64), cont.copy(), nat.copy()]
    for g in (3, 4, 5):
        k0, k1 = SPLITS[g]
        prev = tn[-1]
        new = np.empty(N, np.int64)
        nid = 0
        for node in np.unique(prev):
            idx = np.nonzero(prev == node)[0]
            k = k0 + int(hash01(int(node) * 7 + g, seed) * (k1 - k0 + 1))
            k = max(1, min(k, len(idx) // 14))
            sub = tx.split_points(xy[idx].astype(np.float64), k, seed * 1000 + g * 101 + int(node))
            u, inv = np.unique(sub, return_inverse=True)
            new[idx] = nid + inv
            nid += len(u)
        tn.append(new)
    # compact every generation's ids + parents + centroids
    parent, cen = [np.zeros(1, np.int64)], [xy.mean(0, keepdims=True).astype(np.float64)]
    for g in range(1, len(tn)):
        u, inv = np.unique(tn[g], return_inverse=True)
        tn[g] = inv
        n = len(u)
        par = np.zeros(n, np.int64)
        par[tn[g]] = tn[g - 1]
        cnt = np.bincount(tn[g], minlength=n).astype(np.float64)
        c = np.stack([np.bincount(tn[g], xy[:, 0], n) / cnt, np.bincount(tn[g], xy[:, 1], n) / cnt], 1)
        parent.append(par)
        cen.append(c)
    return tn, parent, cen


# ============================================================ kingdoms of one (sprites)
KH = 400.0                       # figure height in sprite-local units (v1 layouts want tiles >= ~6 units)
SPR_W, SPR_H = 480.0, 544.0      # sprite canvas (local units)
GROUND = (240.0, 520.0)          # the feet
NVAR = 8                         # crowd variants
KING_H = 58.0                    # hero figure height (world units)
FINE_ZOOM = 7.0                  # above this zoom the heroes' islands switch to their fine andamento


def king_sprite(seed, hero=0, ppu=0.75, tile=13.0, fine_tile=8.0):
    """a crowned citizen standing on his/her kingdom (hero 1 = CITIZEN1 'declare', 2 = CITIZEN2 'one_finger')."""
    style = {1: dict(sex="m", skin="warm", beard="short"), 2: dict(sex="f", skin="light")}.get(hero, {})
    fig = icon.Figure("citizen", seed, **style)

    def draw(ctx, layer, pose="stand", t=0.0, mouth=0.0, expr="neutral", look=(0.0, 0.0), blink=False):
        fig.draw(ctx, GROUND[0], GROUND[1], KH, pose=pose, t=t, mouth=mouth, look=look, expr=expr, crown=True,
                 layer=layer, tile=tile, blink=blink)
    poses = [dict(pose="stand")]
    if hero == 1:
        poses.append(dict(pose="declare"))
    if hero == 2:
        poses.append(dict(pose="one_finger"))
    return tx.Sprite(SPR_W, SPR_H, draw, tile=tile, fine_tile=fine_tile, seed=300 + seed, poses=poses, ppu=ppu,
                     anchor=GROUND)


class NationPanel:
    def __init__(self, S):
        from scipy.spatial import cKDTree
        path = S.cache / f"m06_{VERSION}.npz"
        if path.exists():
            d = dict(np.load(path))
        else:
            d = _build_cache()
            np.savez_compressed(path, **d)
        self.d = d
        # map tiles
        lay = mz.Layout(d["xy"], d["ang"], d["size"], seed=61)
        self.xy = lay.xy.astype(np.float64)
        self.ang = lay.ang.astype(np.float64)
        self.size = lay.size.astype(np.float64)
        self.tilt, self.jit = lay.tilt, lay.jit
        self.n = len(self.xy)
        self.uid = np.arange(self.n, dtype=np.int64)
        f = tx.noise_field(5, 160.0)
        self.delta = (0.14 + 0.14 * f(self.xy[:, 0], self.xy[:, 1])).astype(np.float32)
        self.pyr = tx.Pyramid(np.dstack([d["color"], d["gold"]]), BOX[0], BOX[1], PPU)
        # the Ocean
        sea = mz.Layout(d["sea_xy"], d["sea_ang"], d["sea_size"], seed=62)
        self.sea_xy = sea.xy.astype(np.float64)
        self.sea_ang = sea.ang.astype(np.float64)
        self.sea_size = sea.size.astype(np.float64)
        self.sea_tilt, self.sea_jit = sea.tilt, sea.jit
        self.sea_uid = np.arange(len(self.sea_xy), dtype=np.int64) + 10 ** 7
        g = tx.noise_field(6, 200.0)
        self.sea_delta = (0.14 + 0.14 * g(self.sea_xy[:, 0], self.sea_xy[:, 1])).astype(np.float32)
        # hierarchy
        tn, parent, cen = _hierarchy(self.xy, d["cont"], d["nat"])
        self.tn, self.G = tn, len(tn) - 1
        leaves = tn[self.G]
        nl = leaves.max() + 1
        lc = cen[self.G]
        # the hero kingdom (CITIZEN1) and his neighbour (CITIZEN2)
        cnt = np.bincount(leaves, minlength=nl)
        leaf_cont = np.zeros(nl, np.int64)
        leaf_cont[leaves] = d["cont"]
        self.leaf_cont = leaf_cont
        ok = (cnt >= np.median(cnt)) & (leaf_cont == 1)
        dd = np.linalg.norm(lc - HERO_AT, axis=1) + np.where(ok, 0, 1e6)
        self.hero = int(np.argmin(dd))
        tree = cKDTree(self.xy)
        nb = tree.query(self.xy, k=9, distance_upper_bound=16.0)[1]
        nb[nb >= self.n] = np.arange(self.n)[:, None].repeat(9, 1)[nb >= self.n]
        self.nb = nb
        hero_tiles = leaves == self.hero
        adj = np.unique(leaves[nb[hero_tiles]].ravel())
        adj = adj[adj != self.hero]
        right = adj[lc[adj, 0] > lc[self.hero, 0] - 5] if np.any(lc[adj, 0] > lc[self.hero, 0] - 5) else adj
        self.hero2 = int(right[np.argmin(np.abs(lc[right, 1] - lc[self.hero, 1]) + 0.3 * np.abs(lc[right, 0] - lc[self.hero, 0]))])
        # split times: generation time + a stagger outward from the hero (the camera's focus)
        focus = lc[self.hero]
        t0, dvec, spin = [np.zeros(1)], [np.zeros((1, 2))], [np.zeros(1)]
        for gg in range(1, self.G + 1):
            n = len(cen[gg])
            idn = np.arange(n)
            hsh = np.array([hash01(int(i) * 13 + gg, 3) for i in idn])
            hs2 = np.array([hash01(int(i) * 17 + gg, 4) for i in idn])
            hs3 = np.array([hash01(int(i) * 19 + gg, 5) for i in idn])
            pc = cen[gg - 1][parent[gg]]
            if gg == 1:
                st = np.zeros(n)
            else:
                st = 0.16 * np.clip(np.linalg.norm(pc - focus, axis=1) / 520.0, 0, 1) + 0.03 * hsh
            t0.append(TK[gg] + st)
            v = cen[gg] - pc
            ln = np.linalg.norm(v, axis=1)
            u = np.where(ln[:, None] > 1e-6, v / np.maximum(ln, 1e-6)[:, None], 0.0)
            perp = np.stack([-u[:, 1], u[:, 0]], 1)
            mag = DRIFT[gg] * (0.75 + 0.5 * hs2)
            only = np.bincount(parent[gg], minlength=len(cen[gg - 1]))[parent[gg]] <= 1
            mag = np.where(only, 0.0, mag)
            dvec.append(u * mag[:, None] + perp * (mag * 0.35 * (hs3 * 2 - 1))[:, None])
            spin.append(np.where(only, 0.0, SPIN[gg] * (hs3 * 2 - 1)))
        self.cen = cen
        self.sh = tx.Shatter(parent, cen, t0, dvec, spin, tn, lin=0.09)
        # crack light: per generation, the border tiles and the moment their crack opens
        self.crack_t = np.full((self.G + 1, self.n), np.inf)
        for gg in range(1, self.G + 1):
            a = tn[gg]
            b = tn[gg - 1]
            border = np.any((a[nb] != a[:, None]) & (b[nb] == b[:, None]), axis=1)
            front = np.linalg.norm(self.xy - cen[gg - 1][b], axis=1) / (2600.0 if gg == 1 else 1500.0)
            ct = t0[gg][a] - 0.07 + front
            self.crack_t[gg] = np.where(border, ct, np.inf)
        # debris: a few border tiles are torn out and fly off at each crack
        rng = np.random.default_rng(9)
        bt = np.min(self.crack_t[1:], axis=0)
        frac = np.where(np.isfinite(self.crack_t[1]) | np.isfinite(self.crack_t[2]), 0.08, 0.025)
        pick = np.isfinite(bt) & (rng.random(self.n) < frac)
        self.deb = np.nonzero(pick)[0]
        self.deb_t = bt[self.deb] + 0.06
        self.deb_v = rng.normal(0, 1, (len(self.deb), 2)) * 110.0 + \
            (self.xy[self.deb] - C0) / np.maximum(np.linalg.norm(self.xy[self.deb] - C0, axis=1), 1)[:, None] * 90.0
        self.deb_v[:, 1] -= 160.0
        self.deb_w = rng.normal(0, 1, len(self.deb)) * 7.0
        self.deb_flip = rng.uniform(6, 16, len(self.deb))
        self.deb_life = rng.uniform(0.5, 1.1, len(self.deb))
        self.deb_p0 = np.zeros((len(self.deb), 2))
        self.deb_a0 = np.zeros(len(self.deb))
        for tt in np.unique(np.round(self.deb_t, 3)):
            k = np.nonzero(np.round(self.deb_t, 3) == tt)[0]
            M = self.sh.node_mats(float(tt))[-1][tn[self.G][self.deb[k]]]
            self.deb_p0[k] = tx.apply(M, self.xy[self.deb[k]])
            self.deb_a0[k] = self.ang[self.deb[k]] + tx.mat_angle(M)
        self.alive_until = np.full(self.n, np.inf)
        self.alive_until[self.deb] = self.deb_t
        # kingdoms of one: every last piece gets one crowned person standing on it
        self.tl = S.tl
        leaves = tn[self.G]
        nl = int(leaves.max()) + 1
        cnt = np.bincount(leaves, minlength=nl)
        lc = cen[self.G]
        self.k_leaf = np.nonzero(cnt >= 10)[0]
        dc = np.linalg.norm(self.xy - lc[leaves], axis=1)
        order = np.lexsort((dc, leaves))
        first = order[np.r_[0, np.nonzero(np.diff(leaves[order]))[0] + 1]]
        stand = np.zeros((nl, 2))
        stand[leaves[first]] = self.xy[first]
        self.k_h = np.clip(0.9 * np.sqrt(cnt * 51.0), 30.0, 50.0)
        self.k_h[[self.hero, self.hero2]] = KING_H
        self.k_feet = stand + np.stack([np.zeros(nl), 0.3 * self.k_h], 1)
        hk = np.array([hash01(int(i), 77) for i in range(nl)])
        self.k_var = (np.array([hash01(int(i), 78) for i in range(nl)]) * NVAR).astype(int) % NVAR
        self.k_pop = t0[self.G] + 0.12 + 0.36 * hk
        self.k_pop[self.hero] = t0[self.G][self.hero] + 0.25
        self.k_pop[self.hero2] = t0[self.G][self.hero2] + 0.3
        self.spr = [king_sprite(v) for v in range(NVAR)]
        self.hero_spr = {1: king_sprite(11, 1, ppu=2.3, tile=5.5, fine_tile=2.6),
                         2: king_sprite(12, 2, ppu=2.3, tile=5.5, fine_tile=2.6)}
        # the heroes' islands get their own fine andamento (courses echoing around the figure) for the close-ups
        # the portal: one gold tessera of CITIZEN2's crown, into which the camera dives (it holds the next picture)
        sp2 = self.hero_spr[2]
        a2 = icon.Figure("citizen", 12, sex="f", skin="light").anchors(GROUND[0], GROUND[1], KH, pose="one_finger")
        hx, hy = a2["head"]
        ty = a2["top"][1]
        cand = np.nonzero((sp2.static["gold"] > 0.5) & (sp2.xy[:, 1] < hy))[0]
        dd = np.hypot(sp2.xy[cand, 0] - hx, sp2.xy[cand, 1] - (ty + (hy - ty) * 0.3))
        self.portal_tile = int(cand[np.argmin(dd)]) if len(cand) else int(np.argmin(np.hypot(sp2.xy[:, 0] - hx, sp2.xy[:, 1] - ty)))
        near = np.linalg.norm(lc - lc[self.hero], axis=1) < 150.0
        near |= np.linalg.norm(lc - lc[self.hero2], axis=1) < 150.0
        near[[self.hero, self.hero2]] = True
        hid_of = {self.hero: 1, self.hero2: 2}
        self.fine = {int(leaf): self._fine_island(int(leaf), hid_of.get(int(leaf), 0))
                     for leaf in np.nonzero(near & (cnt >= 6))[0]}

    def _fine_island(self, leaf, hid, tile=1.3, K=8.0, ppu=4.0):
        """a proper fine layout for one island (home units), built in a x K local unit system (v1 joints)."""
        idx = np.nonzero(self.tn[self.G] == leaf)[0]
        x0, y0 = self.xy[idx].min(0) - 10
        x1, y1 = self.xy[idx].max(0) + 10
        w, h = float(x1 - x0), float(y1 - y0)
        gw, gh = int(w * ppu), int(h * ppu)
        mask = np.zeros((gh, gw), np.uint8)
        for i in idx:                                   # the broken edge keeps the coarse tiles' jagged steps
            c, s_ = math.cos(self.ang[i]), math.sin(self.ang[i])
            hs = 0.56 * self.size[i]
            q = np.array([[-hs, -hs], [hs, -hs], [hs, hs], [-hs, hs]])
            pts = (self.xy[i] + q @ np.array([[c, s_], [-s_, c]]) - [x0, y0]) * ppu
            cv2.fillConvexPoly(mask, np.round(pts).astype(np.int32), 255)
        X, Y = np.meshgrid(x0 + (np.arange(gw) + 0.5) / ppu, y0 + (np.arange(gh) + 0.5) / ppu)
        rgb, _ = self.pyr.sample(X.ravel(), Y.ravel(), np.full(X.size, 0.8))
        guide = rgb.reshape(gh, gw, 3)
        # the figure's silhouette becomes a contour: the ground courses echo around it (the Ravenna halo)
        sp = self.hero_spr[hid] if hid else self.spr[self.k_var[leaf]]
        rgba, _ = sp.canvases(pose="stand")
        k = self.k_h[leaf] / KH
        fx, fy = self.k_feet[leaf]
        # grid px -> canvas px: canvas = ((x0 + g/ppu - fx)/k + GX) * sppu
        a = sp.ppu / (k * ppu)
        Mi = np.array([[a, 0, ((x0 - fx) / k + GROUND[0]) * sp.ppu], [0, a, ((y0 - fy) / k + GROUND[1]) * sp.ppu]])
        sil = cv2.warpAffine(rgba[..., 3], Mi, (gw, gh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP) > 0.5
        edges = (cv2.dilate(sil.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~sil
        if leaf not in self.k_leaf:
            edges[:] = False
        lay = mz.Layout.flow(guide, tile=tile * K, seed=900 + hid, extent=(w * K, h * K), alpha=mask > 0,
                             edges=edges, fine=None)
        xy = lay.xy.astype(np.float64) / K + np.array([x0, y0])
        size = lay.size.astype(np.float64) / K
        rgbt, g = self.pyr.sample(xy[:, 0], xy[:, 1], size)
        return dict(xy=xy, ang=lay.ang.astype(np.float64), size=size, tilt=lay.tilt, jit=lay.jit,
                    uid=np.arange(len(xy), dtype=np.int64) + 7 * 10 ** 7 + leaf * 10 ** 4, rgb=rgbt,
                    gold=np.clip((g - 0.35) / 0.35, 0, 1))

    # ------------------------------------------------------------------ queries
    def portal(self, T):
        """(centre world, angle, side world) of the crown tessera that holds the congregation."""
        sp = self.hero_spr[2]
        j = self.portal_tile
        M = tx.compose(self.leaf_mats(T)[[self.hero2]], self.king_P([self.hero2], T))[0]
        c = M[:, :2] @ sp.xy[j] + M[:, 2]
        return c, float(sp.ang[j] + math.atan2(M[1, 0], M[0, 0])), float(sp.size[j] * math.sqrt(abs(np.linalg.det(M[:, :2]))))

    def portal_colour(self):
        c = self.hero_spr[2].static
        return c["rgb"][self.portal_tile], float(c["gold"][self.portal_tile])

    def leaf_mats(self, T):
        return self.sh.node_mats(T)[-1]

    def leaf_point(self, leaf, T, p=None):
        """world position of a home point on a leaf (default its centroid) at time T, and the leaf's rotation."""
        M = self.leaf_mats(T)[leaf][None]
        p = self.cen[self.G][leaf] if p is None else np.asarray(p, np.float64)
        return tx.apply(M, p[None])[0], float(tx.mat_angle(M)[0])

    def colour_map(self, lxy, lsize, pidx):
        rgb, g = self.pyr.sample(lxy[:, 0], lxy[:, 1], lsize)
        return rgb, np.clip((g - 0.35) / 0.35, 0, 1)

    @staticmethod
    def colour_sea(lxy, lsize, pidx):
        return sea_colour(lxy[:, 0], lxy[:, 1]), None

    # ------------------------------------------------------------------ render
    def render(self, fc, img, V, T, light, sea=True, portal=False, lam_extra=0.0):
        zoom = math.hypot(V[0, 0], V[1, 0])
        if sea:
            img[...] = mz.SC["mortar"] * 0.5
            tx.render_set(fc, img, V, self.sea_xy, self.sea_ang, self.sea_size, self.sea_tilt, self.sea_jit,
                          self.sea_uid, self.colour_sea, light=light, delta=self.sea_delta, lam_extra=lam_extra)
        mats = self.sh.node_mats(T)
        Mt = mats[-1][self.tn[self.G]]
        alive = (T < self.alive_until).astype(np.float32)
        fine_on = zoom >= FINE_ZOOM
        if fine_on:
            for leaf in self.fine:
                alive[self.tn[self.G] == leaf] = 0.0
        # the pieces float: soft shadows on the Ocean once they part
        lift = float(np.clip((T - TK[1]) / 0.35, 0, 1))
        if lift > 0:
            w = tx.apply(Mt, self.xy)
            sxy = np.stack([V[0, 0] * w[:, 0] + V[0, 1] * w[:, 1] + V[0, 2],
                            V[1, 0] * w[:, 0] + V[1, 1] * w[:, 1] + V[1, 2]], 1)
            tx.shadow(fc, img, sxy, self.size * zoom, (4.0 * zoom * lift, 7.0 * zoom * lift), 4.0 * zoom, 0.6 * lift,
                      alpha=alive)
        # crack light in the joints
        dt = T - self.crack_t[1:]
        pulse = np.where(dt > 0, np.exp(-np.maximum(dt, 0) / 0.22), np.clip(1 + dt / 0.05, 0, 1))
        glow = np.max(np.where(np.isfinite(dt), pulse, 0.0), axis=0).astype(np.float32)
        joint = glow[:, None] * CRACK_LIGHT[None] * 1.5
        tx.render_set(fc, img, V, self.xy, self.ang, self.size, self.tilt, self.jit, self.uid, self.colour_map,
                      M=Mt, light=light, alpha=alive, joint=joint, delta=self.delta, bed=1.6, lam_extra=lam_extra)
        if fine_on:
            batch = []
            for leaf, f in self.fine.items():
                M = np.repeat(mats[-1][[leaf]], len(f["xy"]), axis=0)
                rgb_, gold_ = f["rgb"], f["gold"]
                tx.render_set(fc, img, V, f["xy"], f["ang"], f["size"], f["tilt"], f["jit"], f["uid"],
                              lambda lxy, lsize, pidx, r=rgb_, g=gold_: (r[pidx], g[pidx]), M=M, light=light,
                              lmax=0, bed=1.6, batch=batch)
            tx.flush(fc, img, batch, light, bed=1.6)
        self.render_kings(fc, img, V, T, light, mats[-1], portal, lam_extra)
        # torn-out tesserae tumbling toward the camera and falling away
        tau = T - self.deb_t
        live = (tau > 0) & (tau < self.deb_life)
        if live.any():
            k = np.nonzero(live)[0]
            ta = tau[k]
            p = self.deb_p0[k] + self.deb_v[k] * ta[:, None] + np.array([0.0, 420.0]) * (ta ** 2)[:, None] * 0.5
            sxy = np.stack([V[0, 0] * p[:, 0] + V[0, 1] * p[:, 1] + V[0, 2], V[1, 0] * p[:, 0] + V[1, 1] * p[:, 1] + V[1, 2]], 1)
            grow = 1.0 + 0.7 * ta / self.deb_life[k]
            ti = self.deb[k]
            rgb, gold = self.colour_map(self.xy[ti], self.size[ti], ti)
            flip = np.cos(self.deb_flip[k] * ta)
            rgb = np.where(flip[:, None] < 0, np.array(col(S["mortar_l"]), np.float32) * 0.8, rgb)
            fade = np.clip((self.deb_life[k] - ta) / 0.25, 0, 1)
            tx.draw(fc, img, sxy, self.deb_a0[k] + self.deb_w[k] * ta + math.atan2(V[1, 0], V[0, 0]),
                    self.size[ti] * zoom * grow, rgb, gold, self.tilt[ti], self.jit[ti], alpha=fade,
                    light=light, aspect=np.abs(flip) * 0.9 + 0.1, bed=1.25)
        return mats

    def snapshot(self, T):
        """every tessera of the world at T in WORLD units: (xy, ang, size, rgb, alpha) - raw material for the dust."""
        out = [(self.sea_xy, self.sea_ang, self.sea_size, sea_colour(self.sea_xy[:, 0], self.sea_xy[:, 1]),
                np.ones(len(self.sea_xy), np.float32))]
        mats = self.sh.node_mats(T)
        Mt = mats[-1][self.tn[self.G]]
        rgb, _ = self.pyr.sample(self.xy[:, 0], self.xy[:, 1], self.size)
        out.append((tx.apply(Mt, self.xy), self.ang + tx.mat_angle(Mt), self.size, rgb,
                    (T < self.alive_until).astype(np.float32)))
        Ml = mats[-1]
        up = self.k_leaf[T > self.k_pop[self.k_leaf]]
        for leaf in up:
            if leaf == self.hero:
                sp, colr = self.hero_spr[1], self.hero_spr[1].colours(**self.hero_state(1, T))
            elif leaf == self.hero2:
                sp, colr = self.hero_spr[2], self.hero_spr[2].colours(**self.hero_state(2, T))
            else:
                sp = self.spr[self.k_var[leaf]]
                colr = sp.static
            M = np.repeat(tx.compose(Ml[[leaf]], self.king_P([leaf], T)), sp.n, axis=0)
            out.append((tx.apply(M, sp.xy), sp.ang + tx.mat_angle(M), sp.size * tx.mat_scale(M), colr["rgb"],
                        colr["alpha"]))
        cat = [np.concatenate([np.asarray(o[k]) for o in out]) for k in range(5)]
        keep = cat[4] > 0.05
        return tuple(c[keep] for c in cat)

    # ------------------------------------------------------------------ kingdoms of one
    def king_P(self, leaves, T):
        """sprite-local -> leaf-home affines (springing up out of the tiles at k_pop)."""
        leaves = np.atleast_1d(leaves)
        tau = np.clip((T - self.k_pop[leaves]) / 0.42, 0, 1)
        c1 = 1.70158 * 1.3
        q = tau - 1
        pop = np.where(tau > 0, 1 + (c1 + 1) * q ** 3 + c1 * q ** 2, 0.0)
        k = self.k_h[leaves] / KH * pop
        P = np.zeros((len(leaves), 2, 3))
        P[:, 0, 0] = k
        P[:, 1, 1] = k
        P[:, 0, 2] = self.k_feet[leaves, 0] - k * GROUND[0]
        P[:, 1, 2] = self.k_feet[leaves, 1] - k * GROUND[1]
        return P

    def hero_state(self, hid, T):
        tl = self.tl
        spk = "CITIZEN1" if hid == 1 else "CITIZEN2"
        m = round(tl.mouth(spk, T) / 0.05) * 0.05
        talking = tl.speaking(spk, T, 0.1) is not None
        common = dict(t=math.floor(T * 2) / 2, mouth=m, blink=tx.blink(T, hid) and not talking,
                      expr="shouting" if talking else "proud")
        if hid == 1:
            s = smootherstep(109.05, 109.3, T) * (1 - smootherstep(111.75, 112.05, T))
            return dict(pose="declare" if s > 0.5 else "stand", **common)
        s = smootherstep(112.3, 112.5, T)
        return dict(pose="one_finger" if s > 0.5 else "stand", **common)

    def render_kings(self, fc, img, V, T, light, Ml, portal=False, lam_extra=0.0):
        zoom = math.hypot(V[0, 0], V[1, 0])
        up = self.k_leaf[T > self.k_pop[self.k_leaf]]
        crowd = up[(up != self.hero) & (up != self.hero2)]
        if len(crowd):
            w = tx.apply(Ml[crowd], self.k_feet[crowd])
            sx = V[0, 0] * w[:, 0] + V[0, 1] * w[:, 1] + V[0, 2]
            sy = V[1, 0] * w[:, 0] + V[1, 1] * w[:, 1] + V[1, 2]
            m = self.k_h[crowd] * zoom * 1.2 + 20
            crowd = crowd[(sx > -m) & (sx < W + m) & (sy > -m * 0.2) & (sy < H + m * 1.2)]
        batch = []
        if len(crowd) and zoom * 30 > 6:
            Ms, hxy, hang, hsz, tl_, jt, uid, rgb, gold, var = [], [], [], [], [], [], [], [], [], []
            for v in range(NVAR):
                L = crowd[self.k_var[crowd] == v]
                if len(L) == 0:
                    continue
                sp = self.spr[v]
                M = tx.compose(Ml[L], self.king_P(L, T))
                Ms.append(np.repeat(M, sp.n, axis=0))
                hxy.append(np.tile(sp.xy, (len(L), 1)))
                hang.append(np.tile(sp.ang, len(L)))
                hsz.append(np.tile(sp.size, len(L)))
                tl_.append(np.tile(sp.tilt, (len(L), 1)))
                jt.append(np.tile(sp.jit, (len(L), 1)))
                uid.append((np.repeat(L, sp.n) * 100003 + np.tile(np.arange(sp.n), len(L)) + 5 * 10 ** 7).astype(np.int64))
                rgb.append(np.tile(sp.static["rgb"], (len(L), 1)))
                gold.append(np.tile(sp.static["gold"], len(L)))
                var.append(np.full(len(L) * sp.n, v))
            RGB, GOLD, VAR, HS = np.concatenate(rgb), np.concatenate(gold), np.concatenate(var), np.concatenate(hsz)
            spr = self.spr

            def cfn(lxy, lsize, pidx):
                r = RGB[pidx].copy()
                g = GOLD[pidx].copy()
                ch = np.nonzero(lsize < HS[pidx] * 0.99)[0]
                for v in np.unique(VAR[pidx[ch]]):
                    k = ch[VAR[pidx[ch]] == v]
                    r[k], g[k] = spr[v].static["pyr"].sample(lxy[k, 0], lxy[k, 1], lsize[k])
                return r, g
            tx.render_set(fc, img, V, np.concatenate(hxy), np.concatenate(hang), HS, np.concatenate(tl_),
                          np.concatenate(jt), np.concatenate(uid), cfn, M=np.concatenate(Ms), light=light, bed=1.35,
                          margin=10.0, batch=batch, lam_extra=lam_extra)
        after = []
        for hid, leaf in ((1, self.hero), (2, self.hero2)):
            if T <= self.k_pop[leaf]:
                continue
            sp = self.hero_spr[hid]
            colr = sp.colours(**self.hero_state(hid, T))
            M = np.repeat(tx.compose(Ml[[leaf]], self.king_P([leaf], T)), sp.n, axis=0)
            alpha = colr["alpha"]
            if portal and hid == 2:
                alpha = alpha.copy()
                alpha[self.portal_tile] = 0.0
            tx.render_set(fc, img, V, sp.xy, sp.ang, sp.size, sp.tilt, sp.jit, sp.uid + 9 * 10 ** 7 + hid * 10 ** 6,
                          sp.colour_fn(colr), M=M, light=light, alpha=alpha, bed=1.35, margin=10.0,
                          lmax=1 if portal else 5, batch=batch, lam_extra=lam_extra)
            if portal and hid == 2:
                # the portal tessera is drawn whole (never split): the dive looks through its glass
                j = [self.portal_tile]
                tx.render_set(fc, img, V, sp.xy[j], sp.ang[j], sp.size[j], sp.tilt[j], sp.jit[j], sp.uid[j] + 1,
                              lambda lxy, lsize, pidx: (colr["rgb"][j][pidx], colr["gold"][j][pidx]), M=M[j],
                              light=light, lmax=0, bed=1.19, margin=10.0, batch=after)
        tx.flush(fc, img, batch, light, bed=1.35)
        tx.flush(fc, img, after, light, bed=1.19)

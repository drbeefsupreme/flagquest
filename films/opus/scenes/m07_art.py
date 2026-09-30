"""m07 helper: the Babel wall programme, painted as the mosaicist's cartoon (setup-time only).

World (vx.mosaic v1 convention): design units, X right, Y DOWN, Z toward the viewer. The wall is the plane z=0;
its chart (x, y) spans [0, WX] x [0, WY]; the nave floor is the plane y = FLOOR (= WY) in front of it (z > 0).

The programme, bottom to top (heights above the floor):
    0-56     dado: VT · NON · AVDIAT · VNVSQVISQVE · VOCEM · PROXIMI · SVI (Gen 11:7) on ink
    56-110   the ground: green earth, stones, tufts
    110-330  the two walled cities of Babel (left and right), their windows lit screens
    100-893  THE TOWER OF BABEL in seven tiers (leaning a little to the right), screen-bricks, arcades of glowing
             feeds, ladders; an unfinished crown with scaffolding and a treadwheel crane
    940-996  titulus band: HIC · AEDIFICANT | TVRRIM · BABEL (the crane pierces it)
    996-1436 the conch: rainbow border + LEGIO · NOMEN · MIHI · QVIA · MVLTI · SVMVS (Mk 5:9) around LORD EGREGORE,
             the faceless anti-Pantocrator: static face in a hood of dead glass, a cross-nimbus of billboards in a
             halo of screens, arms spread over the tower, sleeves hung with billboards like wings, static hands
             dangling cables down to marionette labourers.
    spandrels: gold ground under a jewelled cornice.

Paint() gives, at k px per world unit: the colour cartoon, an id map (mat, zone, obj per pixel, no antialiasing)
and the fine-tile mask. paint_sinopia() paints the bare lime plaster with the red-ochre underdrawing of the same
programme (the master drew the colossus WITH a face; it was set faceless over it).
"""
import math

import cairo
import numpy as np

from vx import mosaic as mz
from vx.config import C

WX, WY = 3000.0, 1500.0
FLOOR = WY                     # world y of the floor plane (the wall base)
AX = 1500.0                    # the tower's axis / the Egregore's axis

SM = mz.SMALTI


def rgb(c):
    if isinstance(c, str):
        if c in SM:
            c = SM[c]
        elif c in C:
            return tuple(C[c])
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(c[:3])


def mixc(a, b, t):
    a, b = rgb(a), rgb(b)
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def H(h):
    """height above the floor -> chart y"""
    return FLOOR - h


# ------------------------------------------------------------------ materials / zones
M_NONE, M_GLASS, M_GOLD, M_SCREEN, M_FEED, M_STATIC, M_BOARD, M_HALO, M_WHEEL, M_DIM = range(10)
MAT_NAMES = ["none", "glass", "gold", "screen", "feed", "static", "board", "halo", "wheel", "dim"]
EMISSIVE = (M_SCREEN, M_FEED, M_STATIC, M_BOARD, M_HALO, M_DIM)
Z_BG = 0
Z_TIER0 = 1                    # tiers 0..6 -> zones 1..7, crown -> 8
Z_CROWN = 8
Z_CRANE = 9
Z_LADDER = 10
Z_CITY_L, Z_CITY_R = 11, 12
Z_HALO, Z_DISC, Z_FACE, Z_HOOD, Z_BODY, Z_ARM_L, Z_ARM_R, Z_HAND_L, Z_HAND_R, Z_NIMBUS, Z_SLEEVE_L, Z_SLEEVE_R, \
    Z_CABLE, Z_CHITON = range(20, 34)
EGREGORE_ZONES = tuple(range(20, 34))
Z_ARCH_GROUND = 40
Z_TITULUS = 41
Z_DADO = 42
Z_STAR = 43
Z_RAINBOW = 44
Z_SPANDREL = 45
Z_SKY = 46
Z_GROUND = 47

# ------------------------------------------------------------------ the tower
TIER_H = [150, 135, 122, 110, 100, 92, 84]
TIER_HW = [430, 372, 322, 276, 234, 196, 162]
TIER_OFF = [0, 5, 12, 10, 18, 26, 34]
TIER_WIN = [10, 9, 8, 7, 6, 5, 4]
BASE_H = 100.0
SIDE = 22.0                    # depth of the shadowed right flank


def tiers():
    """[(xc, hw, yb, yt)] chart coords, bottom tier first (yb = bottom edge y, yt = top edge y < yb)"""
    out, h = [], BASE_H
    for k in range(7):
        out.append((AX + TIER_OFF[k], float(TIER_HW[k]), H(h), H(h + TIER_H[k])))
        h += TIER_H[k]
    return out


CROWN = (AX + 40.0, 132.0, H(893.0), H(962.0))
CRANE = dict(mast=(AX + 98.0, H(893.0), H(1080.0)), boom=((AX + 98.0, H(1058.0)), (AX + 330.0, H(1118.0))),
             wheel=(AX + 4.0, H(927.0), 32.0))
TITULUS_Y = (H(1004.0), H(932.0))          # band top, bottom (chart y)
ARCH = (AX, H(1004.0), 1100.0, 432.0)      # centre x, springing y, semi-axes a (half span), b (rise)
RAINBOW = ["porphyry", "porphyry_l", "terracotta", "bone", "verdigris", "lapis_l"]    # outer -> inner
RB_W = 6.0
INSCR_W = 48.0

# ------------------------------------------------------------------ the Egregore (chart coords)
HALO_C = (AX, 292.0)
HALO_R = (150.0, 182.0)
HOOD = (AX, 302.0, 112.0, 142.0)
FACE = (AX, 296.0, 70.0, 94.0)
SHOULDER_Y = 426.0
HANDS = [(AX - 894.0, SHOULDER_Y - 34.0), (AX + 894.0, SHOULDER_Y - 34.0)]
PUPPETS = [(AX - 1000.0, 0), (AX - 890.0, 1), (AX - 780.0, 0), (AX + 790.0, 1), (AX + 900.0, 0), (AX + 1010.0, 1)]

TEXT_TITULUS = ("HIC · AEDIFICANT", "TVRRIM · BABEL")
TEXT_DADO = "VT · NON · AVDIAT · VNVSQVISQVE · VOCEM · PROXIMI · SVI"
TEXT_ARCH = "LEGIO · NOMEN · MIHI · QVIA · MVLTI · SVMVS"
FONT = "VX Cinzel Black"         # vx.mosaic_type 'roman_black' (Roman square capitals, reads best in tesserae)
try:
    from vx.mosaic_type import face as _face
    FONT = _face("roman_black")
except Exception:                 # pragma: no cover - fonts not installed
    FONT = "P052"

PASTEL = ["slop_flesh", "slop_cyan", "slop_beige", "screen_glow", "slop_mauve", "slop_flesh", "slop_cyan",
          "screen_glow"]


# ================================================================== painter
class Paint:
    """colour cartoon + id map + fine mask painted together, in chart units at k px/unit."""

    def __init__(self, k=1.0):
        self.k = k
        self.w, self.h = int(round(WX * k)), int(round(WY * k))
        self.cs = cairo.ImageSurface(cairo.FORMAT_RGB24, self.w, self.h)
        self.ids = cairo.ImageSurface(cairo.FORMAT_RGB24, self.w, self.h)
        self.fs = cairo.ImageSurface(cairo.FORMAT_A8, self.w, self.h)
        self.tfs = cairo.ImageSurface(cairo.FORMAT_A8, self.w, self.h)     # letters (opus vermiculatum)
        self.bs = cairo.ImageSurface(cairo.FORMAT_A8, self.w, self.h)      # the inscription bands
        self.c, self.i, self.f, self.tf, self.b = (cairo.Context(s) for s in (self.cs, self.ids, self.fs, self.tfs,
                                                                              self.bs))
        for x in (self.c, self.i, self.f, self.tf, self.b):
            x.scale(k, k)
            x.set_line_join(cairo.LINE_JOIN_ROUND)
            x.set_line_cap(cairo.LINE_CAP_ROUND)
        for x in (self.i, self.f, self.tf, self.b):
            x.set_antialias(cairo.ANTIALIAS_NONE)

    def band(self, path):
        """mark a region as an inscription band (laid with its own finer layout)"""
        self.b.new_path()
        path(self.b)
        self.b.set_source_rgba(0, 0, 0, 1)
        self.b.fill()

    def letters(self, path, colour, mat=M_GLASS, zone=Z_TITULUS, obj=0, margin=3.0):
        """letter glyphs: fill in colour + ids, mark the text-fine mask (grown by `margin` for the halo course)"""
        self.fill(path, colour, mat, zone, obj)
        self.tf.new_path()
        path(self.tf)
        self.tf.set_source_rgba(0, 0, 0, 1)
        self.tf.fill_preserve()
        self.tf.set_line_width(2 * margin)
        self.tf.stroke()

    def _ids(self, mat, zone, obj):
        self.i.set_source_rgb(mat / 255.0, zone / 255.0, (obj % 256) / 255.0)

    def fill(self, path, colour, mat=M_GLASS, zone=Z_BG, obj=0, fine=None, alpha=1.0, ids=True, grow=0.0):
        """path(ctx) builds the path. fine: True sets, False clears the fine mask, None leaves it.
        grow > 0 also strokes the outline that wide (thickens letters so they survive tessellation)."""
        for ctx, kind in ((self.c, "c"), (self.i, "i"), (self.f, "f")):
            if kind == "i" and not ids:
                continue
            if kind == "f" and fine is None:
                continue
            ctx.new_path()
            path(ctx)
            if kind == "c":
                ctx.set_source_rgba(*rgb(colour), alpha)
            elif kind == "i":
                self._ids(mat, zone, obj)
            else:
                ctx.set_operator(cairo.OPERATOR_SOURCE)
                ctx.set_source_rgba(0, 0, 0, 1.0 if fine else 0.0)
            if grow > 0:
                ctx.fill_preserve()
                ctx.set_line_width(grow * (2.0 if kind == "f" else 1.0))
                ctx.stroke()
            else:
                ctx.fill()

    def stroke(self, path, width, colour, mat=M_GLASS, zone=Z_BG, obj=0, fine=None, alpha=1.0, ids=True):
        for ctx, kind in ((self.c, "c"), (self.i, "i"), (self.f, "f")):
            if kind == "i" and not ids:
                continue
            if kind == "f" and fine is None:
                continue
            ctx.new_path()
            path(ctx)
            ctx.set_line_width(width)
            if kind == "c":
                ctx.set_source_rgba(*rgb(colour), alpha)
            elif kind == "i":
                self._ids(mat, zone, obj)
            else:
                ctx.set_operator(cairo.OPERATOR_SOURCE)
                ctx.set_source_rgba(0, 0, 0, 1.0 if fine else 0.0)
            ctx.stroke()

    # ---------------------------------------------------------- readback
    def colour(self):
        return _read_rgb(self.cs, self.w, self.h)

    def idmap(self):
        a = _read_bgra(self.ids, self.w, self.h)
        return a[..., 2].copy(), a[..., 1].copy(), a[..., 0].copy()      # mat, zone, obj

    def fine(self):
        return _read_a8(self.fs, self.w, self.h)

    def text_fine(self):
        return _read_a8(self.tfs, self.w, self.h)

    def bands(self):
        return _read_a8(self.bs, self.w, self.h)


def _read_a8(surf, w, h):
    surf.flush()
    st = surf.get_stride()
    return np.ndarray((h, st), np.uint8, surf.get_data())[:, :w] > 127


def _read_bgra(surf, w, h):
    surf.flush()
    st = surf.get_stride()
    return np.ndarray((h, st // 4, 4), np.uint8, surf.get_data())[:, :w].copy()


def _read_rgb(surf, w, h):
    a = _read_bgra(surf, w, h)
    return (a[..., 2::-1].astype(np.float32) / 255.0).copy()


# ================================================================== path helpers
def P_rect(x0, y0, x1, y1):
    return lambda c: c.rectangle(x0, y0, x1 - x0, y1 - y0)


def P_poly(pts):
    def f(c):
        c.move_to(*pts[0])
        for p in pts[1:]:
            c.line_to(*p)
        c.close_path()
    return f


def P_ellipse(cx, cy, rx, ry):
    def f(c):
        c.save()
        c.translate(cx, cy)
        c.scale(rx, ry)
        c.new_sub_path()
        c.arc(0, 0, 1, 0, 2 * math.pi)
        c.restore()
    return f


def P_ring(cx, cy, r0, r1, a0=0.0, a1=2 * math.pi):
    def f(c):
        c.new_sub_path()
        c.arc(cx, cy, r1, a0, a1)
        c.arc_negative(cx, cy, r0, a1, a0)
        c.close_path()
    return f


def P_line(pts):
    def f(c):
        c.move_to(*pts[0])
        for p in pts[1:]:
            c.line_to(*p)
    return f


def P_curve(pts, close=False):
    """Catmull-Rom through pts"""
    def f(c):
        q = list(pts) + ([pts[0]] if close else [])
        c.move_to(*q[0])
        n = len(q)
        for i in range(n - 1):
            p0 = q[max(0, i - 1)] if not close or i > 0 else q[-2]
            p1, p2 = q[i], q[i + 1]
            p3 = q[min(n - 1, i + 2)] if not close or i + 2 < n else q[1]
            c.curve_to(p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6,
                       p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6, p2[0], p2[1])
        if close:
            c.close_path()
    return f


def arch_pt(th, grow=0.0):
    cx, sy, a, b = ARCH
    return cx + (a + grow) * math.cos(th), sy - (b + grow) * math.sin(th)


def arch_halfwidth(y, grow=0.0):
    """half width of the (grown) arch at chart y (0 above the apex)"""
    cx, sy, a, b = ARCH
    d = (sy - y) / (b + grow)
    return (a + grow) * math.sqrt(max(0.0, 1 - d * d)) if d >= 0 else a + grow


def P_arch_band(g0, g1, n=160):
    """the band between the arch grown by g0 and by g1 (above the springing line)"""
    def f(c):
        pts = [arch_pt(math.pi * i / n, g1) for i in range(n + 1)]
        pts += [arch_pt(math.pi * (n - i) / n, g0) for i in range(n + 1)]
        c.move_to(*pts[0])
        for p in pts[1:]:
            c.line_to(*p)
        c.close_path()
    return f


def P_arch_inside(grow):
    def f(c):
        cx, sy, a, b = ARCH
        c.save()
        c.translate(cx, sy)
        c.scale(a + grow, b + grow)
        c.new_sub_path()
        c.arc_negative(0, 0, 1, 0, math.pi)
        c.close_path()
        c.restore()
    return f


def text_path_centered(c, s, x, y, size, font=None, bold=False, tracking=0.12):
    """Roman square capitals centred on x, baseline y (per-glyph advance + tracking)"""
    font = font or FONT
    c.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    c.set_font_size(size)
    adv = [c.text_extents(ch).x_advance + tracking * size for ch in s]
    wdt = sum(adv) - tracking * size
    xx = x - wdt / 2
    for ch, a in zip(s, adv):
        if ch == "·":
            c.new_sub_path()
            c.arc(xx + a / 2 - tracking * size / 2, y - size * 0.34, size * 0.1, 0, 2 * math.pi)
            c.close_path()
        else:
            c.move_to(xx, y)
            c.text_path(ch)
        xx += a
    return wdt


def text_width_(s, size, font=None, bold=False, tracking=0.12):
    font = font or FONT
    surf = cairo.ImageSurface(cairo.FORMAT_A8, 4, 4)
    c = cairo.Context(surf)
    c.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    c.set_font_size(size)
    return sum(c.text_extents(ch).x_advance + tracking * size for ch in s) - tracking * size

# ================================================================== the programme
def paint_wall(k=1.0, seed=7):
    """the full wall at rest (everything set, Egregore lit, arms spread). Returns Paint."""
    rng = np.random.default_rng(seed)
    p = Paint(k)
    _arch(p, rng)
    _spandrels(p)
    _sky(p, rng)
    _city(p, rng, -1)
    _city(p, rng, 1)
    _titulus(p)
    _ground(p, rng)
    _dado(p)
    _tower(p, rng)
    _egregore(p, rng)
    _cables(p)
    return p


def _spandrels(p):
    """gold ground outside the conch (the arch and its rainbow border are left as a hole)"""
    cx, sy, a, b = ARCH
    g = RB_W * len(RAINBOW) - 0.5

    def outside(c):
        c.rectangle(0, 0, WX, TITULUS_Y[0])
        c.save()
        c.translate(cx, sy)
        c.scale(a + g, b + g)
        c.move_to(1, 0)
        c.arc_negative(0, 0, 1, 0, math.pi)
        c.close_path()
        c.restore()
    p.fill(outside, "gold", M_GOLD, Z_SPANDREL)
    # a jewelled cornice at the very top: porphyry band, gems and pearls
    p.fill(P_rect(0, 0, WX, 24), "porphyry", M_GLASS, Z_SPANDREL)
    p.fill(P_rect(0, 24, WX, 30), "gold_d", M_GLASS, Z_SPANDREL, fine=True)
    for i in range(int(WX // 58) + 1):
        x = 18 + i * 58
        p.fill(P_ellipse(x, 12, 13, 8), "malachite" if i % 2 else "lapis_l", M_GLASS, Z_SPANDREL, fine=True)
        p.fill(P_ellipse(x + 29, 12, 3.5, 3.5), "marble", M_GLASS, Z_SPANDREL, fine=True)


def _arch(p, rng):
    cx, sy, a, b = ARCH
    # interior: an anti-glory of dead glass, concentric courses around the head (the bands below cover the spill)
    p.fill(P_arch_inside(-INSCR_W), "night_vault", M_GLASS, Z_ARCH_GROUND)
    hx, hy = HALO_C
    for j, (r, cname) in enumerate([(930, "lapis_d"), (760, "night_vault"), (600, mixc("lapis_d", "tyrian", 0.25)),
                                    (450, "night_vault"), (320, mixc("lapis_d", "ink", 0.3))]):
        p.fill(P_ellipse(hx, hy + 60, r, r), cname, M_GLASS, Z_ARCH_GROUND)
    for i in range(170):
        th = rng.uniform(0.04, math.pi - 0.04)
        rr = math.sqrt(rng.uniform(0.02, 0.92))
        x = cx + (a - INSCR_W - 14) * rr * math.cos(th)
        y = sy - (b - INSCR_W - 14) * rr * math.sin(th)
        if math.hypot(x - hx, y - hy) < HALO_R[1] + 20 or y > SHOULDER_Y - 40:
            continue
        s = rng.uniform(3.5, 5.5)
        p.fill(P_rect(x - s, y - s, x + s, y + s), rgb(PASTEL[i % len(PASTEL)]), M_DIM, Z_ARCH_GROUND, obj=i,
               fine=True)
    # rainbow border and the inscription band
    n = len(RAINBOW)
    for i, cname in enumerate(RAINBOW):
        g1 = (n - i) * RB_W
        g0 = (n - i - 1) * RB_W
        p.fill(P_arch_band(g0 + 0.2, g1 + 0.2), cname, M_GLASS, Z_RAINBOW, fine=True)
    p.fill(P_arch_band(-INSCR_W, 0.3), "ink", M_GLASS, Z_TITULUS)
    p.band(P_arch_band(-INSCR_W, 0.3))
    _arc_text(p, TEXT_ARCH, grow=-INSCR_W * 0.5, size=38.0)


def _arc_text(p, s, grow, size):
    ths = np.linspace(math.pi * 0.93, math.pi * 0.07, 2000)
    pts = np.array([arch_pt(t, grow) for t in ths])
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    total = text_width_(s, size, tracking=0.2)
    s0 = (seg[-1] - total) / 2
    surf = cairo.ImageSurface(cairo.FORMAT_A8, 4, 4)
    m = cairo.Context(surf)
    m.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    m.set_font_size(size)
    pos = s0
    for ch in s:
        adv = m.text_extents(ch).x_advance + 0.2 * size
        mid = pos + adv / 2
        j = int(np.clip(np.searchsorted(seg, mid), 1, len(seg) - 1))
        x, y = pts[j]
        dx, dy = pts[j] - pts[j - 1]
        ang = math.atan2(dy, dx)

        def glyph(c, ch=ch, x=x, y=y, ang=ang, adv=adv):
            c.save()
            c.translate(x, y)
            c.rotate(ang)
            c.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            c.set_font_size(size)
            c.move_to(-adv / 2 + 0.1 * size, size * 0.35)
            c.text_path(ch)
            c.restore()
        if ch.strip() and ch != "·":
            p.letters(glyph, "silver", M_GLASS, Z_TITULUS, margin=3.0)
        elif ch == "·":
            p.letters(P_ellipse(x, y, 4.5, 4.5), "silver", M_GLASS, Z_TITULUS, margin=3.0)
        pos += adv


def _sky(p, rng):
    y0, y1 = TITULUS_Y[1], H(110.0)
    # banded night: mosaic skies are courses of a few tones, not gradients
    bands = [(0.00, "night_vault"), (0.16, "lapis_d"), (0.40, mixc("lapis_d", "lapis", 0.45)),
             (0.64, mixc("lapis", "lapis_d", 0.2)), (0.84, mixc("lapis", "lapis_l", 0.16))]
    for i, (f, cname) in enumerate(bands):
        ya = y0 + (y1 - y0) * f
        ph = rng.uniform(0, 6)

        def band(c, ya=ya, ph=ph, first=(i == 0)):
            xs = np.linspace(-20, WX + 20, 120)
            top = ya + (0 * xs if first else 9 * np.sin(xs / 173.0 + ph) + 5 * np.sin(xs / 61.0 + 2 * ph))
            c.move_to(xs[0], y1 + 2)
            for x, y in zip(xs, top):
                c.line_to(x, y)
            c.line_to(xs[-1], y1 + 2)
            c.close_path()
        p.fill(band, cname, M_GLASS, Z_SKY)
    # stars (gold, eight-rayed, Galla Placidia) avoiding the tower, the cities and the cables
    tw = tiers()
    placed = []
    tries = 0
    while len(placed) < 58 and tries < 6000:
        tries += 1
        x = rng.uniform(40, WX - 40)
        y = rng.uniform(y0 + 30, H(360))
        inside = False
        for (xc, hw, yb, yt) in tw:
            if yt - 40 < y < yb + 10 and abs(x - xc) < hw + 60:
                inside = True
        if abs(x - CROWN[0]) < CROWN[1] + 200 and y < CROWN[2] + 20:
            inside = True
        if inside:
            continue
        if any((x - a) ** 2 + (y - b) ** 2 < 100 ** 2 for a, b in placed):
            continue
        if any(abs(x - hx) < 80 for hx, _ in HANDS):
            continue
        placed.append((x, y))
    for i, (x, y) in enumerate(placed):
        _star(p, x, y, rng.uniform(9, 14), i)


def _star(p, x, y, r, i):
    def rays(c):
        for j in range(8):
            a = j * math.pi / 4
            rr = r if j % 2 == 0 else r * 0.62
            c.move_to(x, y)
            c.line_to(x + rr * math.cos(a), y + rr * math.sin(a))
    p.stroke(rays, 3.2, "gold_l", M_GOLD, Z_STAR, obj=i % 256, fine=True)
    p.fill(P_ellipse(x, y, r * 0.3, r * 0.3), "marble", M_GLASS, Z_STAR, obj=i % 256, fine=True)


def _city(p, rng, side):
    """a walled city of Babel beside the tower: towers, domes, gables; every window a lit screen"""
    z = Z_CITY_L if side < 0 else Z_CITY_R
    x_in, x_out = AX + side * 520, AX + side * 1470
    xa, xb = min(x_in, x_out), max(x_in, x_out)
    gy = H(110.0)
    obj = 0
    # buildings behind the wall
    xs = np.sort(rng.uniform(xa + 20, xb - 60, 14))
    for i, x in enumerate(xs):
        w = rng.uniform(46, 86)
        near = 1 - abs(x - x_in) / abs(x_out - x_in)
        top = gy - rng.uniform(150, 240) - 70 * near
        body = mixc(rng.choice(["bone", "marble_d", "terracotta", "marble"]), "slate", rng.uniform(0.0, 0.35))
        p.fill(P_rect(x, top, x + w, gy), body, M_GLASS, z, obj=200)
        roof = rng.integers(3)
        if roof == 0:          # gable
            p.fill(P_poly([(x - 5, top), (x + w / 2, top - w * 0.45), (x + w + 5, top)]),
                   rng.choice(["ochre_red", "porphyry", "terracotta"]), M_GLASS, z, obj=201, fine=True)
        elif roof == 1:        # dome
            def dome(c, x=x, w=w, top=top):
                c.new_sub_path()
                c.arc(x + w / 2, top, w * 0.46, math.pi, 2 * math.pi)
                c.close_path()
            p.fill(dome, rng.choice(["silver", "slate", "verdigris"]), M_GLASS, z, obj=202, fine=True)
        else:                  # crenellated
            for j in range(int(w // 14) + 1):
                mx = x + 4 + j * 14
                if mx < x + w - 4:
                    p.fill(P_rect(mx - 4, top - 9, mx + 4, top + 1), body, M_GLASS, z, obj=203, fine=True)
        # windows: little lit screens
        for yy in np.arange(top + 16, gy - 60, 30):
            for xx in np.arange(x + 9, x + w - 12, 18):
                if rng.uniform() < 0.8:
                    p.fill(P_rect(xx, yy, xx + 8, yy + 13), rgb(PASTEL[obj % len(PASTEL)]), M_SCREEN, z,
                           obj=obj % 180, fine=True)
                    obj += 1
    # the city wall with towers and a gate
    wy = gy - 88
    p.fill(P_rect(xa, wy, xb, gy), mixc("marble_d", "terracotta", 0.35), M_GLASS, z, obj=210)

    def wall_courses(c):
        for yy in np.arange(wy + 12, gy, 12):
            c.move_to(xa, yy)
            c.line_to(xb, yy)
    p.stroke(wall_courses, 1.0, mixc("marble_d", "slate", 0.45), M_GLASS, z, obj=210, ids=False)
    for j in range(int((xb - xa) // 22) + 1):
        mx = xa + 8 + j * 22
        p.fill(P_rect(mx - 6, wy - 11, mx + 6, wy + 1), mixc("marble_d", "terracotta", 0.35), M_GLASS, z, obj=211,
               fine=True)
    for j, f in enumerate((0.06, 0.34, 0.62, 0.9)):
        tx = xa + (xb - xa) * f
        th = rng.uniform(130, 170)
        p.fill(P_rect(tx - 26, gy - th, tx + 26, gy), mixc("bone", "terracotta", 0.25), M_GLASS, z, obj=212)
        p.fill(P_poly([(tx - 32, gy - th), (tx, gy - th - 44), (tx + 32, gy - th)]), "porphyry_l", M_GLASS, z,
               obj=213, fine=True)
        p.fill(P_rect(tx - 7, gy - th + 20, tx + 7, gy - th + 40), rgb(PASTEL[j]), M_SCREEN, z, obj=190 + j,
               fine=True)
    gx = xa + (xb - xa) * 0.48
    p.fill(_arch_window(gx, gy - 70, gy, 22), "ink", M_GLASS, z, obj=214, fine=True)


def _titulus(p):
    y0, y1 = TITULUS_Y
    p.fill(P_rect(0, y0, WX, y1), "lapis_d", M_GLASS, Z_TITULUS)
    p.band(P_rect(0, y0, WX, y1))
    p.fill(P_rect(0, y0, WX, y0 + 5), "gold", M_GOLD, Z_TITULUS, fine=True)
    p.fill(P_rect(0, y1 - 5, WX, y1), "gold", M_GOLD, Z_TITULUS, fine=True)
    size = 52.0
    base = (y0 + y1) / 2 + size * 0.35
    gap = 190.0
    wl = text_width_(TEXT_TITULUS[0], size)
    wr = text_width_(TEXT_TITULUS[1], size)
    for s, xc in ((TEXT_TITULUS[0], AX - gap - wl / 2), (TEXT_TITULUS[1], AX + gap + wr / 2)):
        p.letters(lambda c, s=s, xc=xc: text_path_centered(c, s, xc, base, size), "gold", M_GOLD, Z_TITULUS)


def _ground(p, rng):
    y0, y1 = H(110.0), H(56.0)

    def earth(c):
        xs = np.linspace(-20, WX + 20, 120)
        c.move_to(xs[0], y1 + 1)
        for x in xs:
            c.line_to(x, y0 + 4 * math.sin(x / 90.0) + 3 * math.sin(x / 37.0))
        c.line_to(xs[-1], y1 + 1)
        c.close_path()
    p.fill(earth, mixc("malachite", "ink", 0.25), M_GLASS, Z_GROUND)
    p.fill(P_rect(0, y0 + 26, WX, y1), mixc("malachite", "slate", 0.35), M_GLASS, Z_GROUND)
    for i in range(80):
        x = rng.uniform(0, WX)
        y = rng.uniform(y0 + 10, y1 - 8)
        if rng.uniform() < 0.5:
            p.fill(P_ellipse(x, y, rng.uniform(6, 11), rng.uniform(4, 6)),
                   rng.choice(["terracotta", "marble_d", "slate", "ochre_red"]), M_GLASS, Z_GROUND, fine=True)
        else:
            def tuft(c, x=x, y=y):
                for j in range(3):
                    c.move_to(x + (j - 1) * 4, y + 6)
                    c.line_to(x + (j - 1) * 9, y - 10)
            p.stroke(tuft, 3.0, "verdigris", M_GLASS, Z_GROUND, fine=True)


def _dado(p):
    y0, y1 = H(56.0), FLOOR
    p.fill(P_rect(0, y0, WX, y1), "ink", M_GLASS, Z_DADO)
    p.band(P_rect(0, y0, WX, y1))
    p.fill(P_rect(0, y0, WX, y0 + 6), "porphyry_l", M_GLASS, Z_DADO, fine=True)
    p.fill(P_rect(0, y1 - 6, WX, y1), "porphyry", M_GLASS, Z_DADO, fine=True)
    size = 40.0
    p.letters(lambda c: text_path_centered(c, TEXT_DADO, AX, (y0 + y1) / 2 + size * 0.35, size, tracking=0.16),
              "bone", M_GLASS, Z_DADO)


# ------------------------------------------------------------------ the tower
def _tower(p, rng):
    tw = tiers()
    for k, (xc, hw, yb, yt) in enumerate(tw):
        z = Z_TIER0 + k
        x0, x1 = xc - hw, xc + hw
        # right flank in shadow (oblique)
        p.fill(P_poly([(x1, yb), (x1 + SIDE, yb - SIDE * 0.55), (x1 + SIDE, yt - SIDE * 0.55 + 14), (x1, yt + 14)]),
               mixc("slate", "terracotta", 0.35), M_GLASS, z, obj=250)
        # facade = screen bricks: every brick a little lit screen, laid in courses
        y = yb - 18
        r = 0
        j = 0
        while y > yt + 16:
            off = 11 if r % 2 else 0
            for xx in np.arange(x0 - off, x1, 22):
                a, b = max(x0, xx + 1), min(x1, xx + 21)
                if b - a > 3:
                    p.fill(P_rect(a, y - 10.5, b, y - 0.5), rgb(PASTEL[rng.integers(len(PASTEL))]), M_SCREEN, z,
                           obj=j % 200)
                    j += 1
            y -= 11
            r += 1
        # arcade: arched windows showing feeds, framed in dark glass, pilasters of marble between
        n = TIER_WIN[k]
        bay = 2 * hw / n
        wh = (yb - yt) * 0.52
        ww = bay * 0.46
        wy1 = yb - 26
        wy0 = wy1 - wh
        for j in range(n):
            cxw = x0 + bay * (j + 0.5)
            p.fill(P_rect(cxw - bay / 2 - 3, yt + 16, cxw - bay / 2 + 5, yb - 18), "marble_d", M_GLASS, z, obj=251)
            p.fill(_arch_window(cxw, wy0, wy1, ww / 2 + 5), "ink", M_GLASS, z, obj=252, fine=True)
            p.fill(_arch_window(cxw, wy0 + 5, wy1 - 3, ww / 2), rgb(PASTEL[(j + k) % len(PASTEL)]), M_FEED, z,
                   obj=j, fine=True)
        # cornice at the bottom (projects), gold fillet, parapet with merlons at the top
        p.fill(P_rect(x0 - 10, yb - 18, x1 + 10, yb), "bone", M_GLASS, z, obj=253)
        p.fill(P_rect(x0 - 10, yb - 13, x1 + 10, yb - 9), "gold", M_GOLD, z, obj=253, fine=True)
        p.fill(P_rect(x0 - 4, yt + 6, x1 + 4, yt + 16), "marble_d", M_GLASS, z, obj=254)
        nm = int((2 * hw) // 24)
        for j in range(nm + 1):
            mx = x0 + j * (2 * hw) / nm
            p.fill(P_rect(mx - 6, yt - 8, mx + 6, yt + 7), "marble", M_GLASS, z, obj=254, fine=True)
    for (a, b, k) in ladders():
        _ladder(p, a, b, k)
    # crown: unfinished tier with scaffolding
    xc, hw, yb, yt = CROWN
    p.fill(P_poly([(xc - hw, yb), (xc - hw, yt + 10), (xc - hw * 0.2, yt + 10), (xc - hw * 0.1, yt + 30),
                   (xc + hw * 0.35, yt + 30), (xc + hw * 0.45, yb - 34), (xc + hw, yb - 30), (xc + hw, yb)]),
           "slop_flesh", M_SCREEN, Z_CROWN, obj=0)
    p.fill(P_rect(xc - hw - 8, yb - 12, xc + hw + 8, yb), "bone", M_GLASS, Z_CROWN, obj=253)

    def scaff(c):
        for xx in np.linspace(xc - hw + 6, xc + hw - 6, 6):
            c.move_to(xx, yb)
            c.line_to(xx, yt - 18)
        for yy in (yt + 4, yt + 34):
            c.move_to(xc - hw, yy)
            c.line_to(xc + hw, yy)
        c.move_to(xc - hw, yb)
        c.line_to(xc + hw * 0.1, yt - 10)
    p.stroke(scaff, 4.0, mixc("terracotta", "ochre_red", 0.5), M_GLASS, Z_CROWN, obj=1, fine=True)
    # crane: mast, boom, stay, treadwheel (the wheel's spokes are re-coloured by rotation at render time)
    mx, my0, my1 = CRANE["mast"]
    p.stroke(P_line([(mx, my0), (mx, my1)]), 8.0, mixc("terracotta", "ink", 0.3), M_GLASS, Z_CRANE, obj=0, fine=True)
    (bx0, by0), (bx1, by1) = CRANE["boom"]
    p.stroke(P_line([(bx0, by0), (bx1, by1)]), 6.0, mixc("terracotta", "ink", 0.2), M_GLASS, Z_CRANE, obj=0,
             fine=True)
    p.stroke(P_line([(mx, my1), (bx1, by1)]), 2.5, "slate", M_GLASS, Z_CRANE, obj=0, fine=True)
    wx, wy, wr = CRANE["wheel"]
    p.fill(P_ellipse(wx, wy, wr + 4, wr + 4), mixc("terracotta", "ink", 0.35), M_GLASS, Z_CRANE, obj=0, fine=True)
    p.fill(P_ellipse(wx, wy, wr, wr), "bone", M_WHEEL, Z_CRANE, obj=1, fine=True)


def _arch_window(cx, y0, y1, r):
    def f(c):
        c.move_to(cx - r, y1)
        c.line_to(cx - r, y0 + r)
        c.arc(cx, y0 + r, r, math.pi, 2 * math.pi)
        c.line_to(cx + r, y1)
        c.close_path()
    return f


def _ladder(p, a, b, k):
    (x0, y0), (x1, y1) = a, b
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    nx, ny = -uy, ux
    wood = mixc("terracotta", "ochre_red", 0.6)

    def rails(c):
        for s in (-6, 6):
            c.move_to(x0 + nx * s, y0 + ny * s)
            c.line_to(x1 + nx * s, y1 + ny * s)
    p.stroke(rails, 3.2, wood, M_GLASS, Z_LADDER, obj=k, fine=True)

    def rungs(c):
        for d in np.arange(10, L - 4, 13):
            c.move_to(x0 + ux * d + nx * -6, y0 + uy * d + ny * -6)
            c.line_to(x0 + ux * d + nx * 6, y0 + uy * d + ny * 6)
    p.stroke(rungs, 2.4, mixc("bone", "terracotta", 0.4), M_GLASS, Z_LADDER, obj=k, fine=True)


def ladders():
    """[(foot (x,y), top (x,y), tier k)] as painted (a ladder on each terrace, alternating sides)"""
    tw = tiers()
    out = []
    for k in range(6):
        xc, hw, yb, yt = tw[k]
        xn, hn, ybn, ytn = tw[k + 1]
        side = -1 if k % 2 == 0 else 1
        out.append(((xn + side * (hn + 28), yt + 4), (xn + side * (hn - 30), ytn + 2), k))
    return out


# ------------------------------------------------------------------ Lord Egregore
def arm_path(side):
    """sleeve outline from the shoulder to the cuff: the forearm rises a little toward the hand, the sleeve hangs
    in a bell beneath it (side -1 left, +1 right)"""
    s = side
    top = [(AX + s * 196, SHOULDER_Y - 54), (AX + s * 330, SHOULDER_Y - 46), (AX + s * 500, SHOULDER_Y - 42),
           (AX + s * 660, SHOULDER_Y - 48), (AX + s * 800, SHOULDER_Y - 58), (AX + s * 846, SHOULDER_Y - 60)]
    bot = [(AX + s * 850, SHOULDER_Y - 18), (AX + s * 790, SHOULDER_Y - 10), (AX + s * 700, SHOULDER_Y + 22),
           (AX + s * 580, SHOULDER_Y + 38), (AX + s * 460, SHOULDER_Y + 28), (AX + s * 330, SHOULDER_Y + 32),
           (AX + s * 226, SHOULDER_Y + 46)]
    return top + bot


def fingers(side):
    """[(base (x,y), tip (x,y), width)] thumb first: an open hand, palm down, fingers hanging like a puppeteer's"""
    hx, hy = HANDS[0 if side < 0 else 1]
    out = []
    for f in range(5):
        k = f - 2
        bx = hx + k * 15 + side * 4
        by = hy + 14 - abs(k) * 2
        if f == 0:                              # the thumb, on the inner side, short and angled inward
            bx = hx - side * 34
            by = hy + 6
            out.append(((bx, by), (bx - side * 16, by + 22), 10.0))
            continue
        ln = 40 - abs(k) * 5
        out.append(((bx, by), (bx + side * (4 + k * 3), by + ln), 10.5 - abs(k)))
    return out


def _egregore(p, rng):
    hx, hy = HALO_C
    r0, r1 = HALO_R
    # ---- the wings of billboards under the arms (drawn first: the arms lie over their top edge)
    for side, zsl in ((-1, Z_SLEEVE_L), (1, Z_SLEEVE_R)):
        j = 0
        for row, (ya, yb, x0_, pitch, wdt) in enumerate(((SHOULDER_Y + 14, SHOULDER_Y + 50, 250, 74, 66),
                                                          (SHOULDER_Y + 50, TITULUS_Y[0] - 2, 290, 78, 70))):
            x = x0_ + (row * 30)
            while x + wdt < 840 - row * 40:
                xa, xb = AX + side * x, AX + side * (x + wdt)
                x0, x1 = min(xa, xb), max(xa, xb)
                yy0 = ya + (j % 2) * 3
                p.fill(P_rect(x0 - 3, yy0 - 3, x1 + 3, yb + 2), "ink", M_GLASS, zsl, obj=100 + j, fine=True)
                p.fill(P_rect(x0, yy0, x1, yb - 1), rgb(["glitch_m", "slop_cyan", "slop_flesh", "glitch_c",
                                                           "screen_glow", "slop_mauve"][j % 6]),
                       M_BOARD, zsl, obj=j, fine=True)
                j += 1
                x += pitch
    # ---- body: mantle over the shoulders (dead dark glass), chiton = a facade of screens
    body = [(AX - 230, SHOULDER_Y - 38), (AX - 118, SHOULDER_Y - 70), (AX + 118, SHOULDER_Y - 70),
            (AX + 230, SHOULDER_Y - 38), (AX + 330, TITULUS_Y[0]), (AX - 330, TITULUS_Y[0])]
    p.fill(P_poly(body), "tyrian", M_GLASS, Z_BODY, obj=0)
    cy0, cy1 = SHOULDER_Y - 44, TITULUS_Y[0]
    p.fill(P_rect(AX - 110, cy0, AX + 110, cy1), "ink", M_GLASS, Z_CHITON, obj=255)
    cols, rows = 4, 2
    for i in range(cols):
        for j in range(rows):
            x0 = AX - 106 + i * 212 / cols + 4
            x1 = AX - 106 + (i + 1) * 212 / cols - 4
            y0 = cy0 + 4 + j * (cy1 - cy0 - 4) / rows + 3
            y1 = cy0 + 4 + (j + 1) * (cy1 - cy0 - 4) / rows - 3
            p.fill(P_rect(x0, y0, x1, y1), rgb(PASTEL[(i + 3 * j) % len(PASTEL)]), M_FEED, Z_CHITON,
                   obj=i * rows + j, fine=True)
    for s in (-1, 1):
        for j in range(3):                                  # fold lines
            pts = [(AX + s * (206 + j * 30), SHOULDER_Y - 30 + j * 6),
                   (AX + s * (230 + j * 28), SHOULDER_Y + 20), (AX + s * (262 + j * 20), TITULUS_Y[0] - 2)]
            p.stroke(P_curve(pts), 3.4, "slate", M_GLASS, Z_BODY, obj=2, fine=True)
    # ---- arms: sleeves of dark glass, fold lines, static hands with hanging fingers
    for side, zarm, zh in ((-1, Z_ARM_L, Z_HAND_L), (1, Z_ARM_R, Z_HAND_R)):
        p.fill(P_curve(arm_path(side), close=True), mixc("tyrian", "ink", 0.3), M_GLASS, zarm, obj=0)
        p.stroke(P_curve([(AX + side * 210, SHOULDER_Y - 50), (AX + side * 500, SHOULDER_Y - 38),
                          (AX + side * 842, SHOULDER_Y - 56)]), 4.0, mixc("tyrian", "porphyry_l", 0.5), M_GLASS,
                 zarm, obj=2, fine=True)
        for j in range(5):
            xs = AX + side * (300 + j * 110)
            p.stroke(P_curve([(xs, SHOULDER_Y - 36), (xs + side * 18, SHOULDER_Y - 4), (xs + side * 40,
                                                                                         SHOULDER_Y + 24)]),
                     3.0, "slate", M_GLASS, zarm, obj=1, fine=True)
        p.stroke(P_line([(AX + side * 846, SHOULDER_Y - 58), (AX + side * 850, SHOULDER_Y - 18)]), 7.0,
                 "porphyry_l", M_GLASS, zarm, obj=3, fine=True)       # the cuff
        hx_, hy_ = HANDS[0 if side < 0 else 1]
        # a dark contour course round palm and fingers (Byzantine hands read by their outline)
        for (b, tp, wdt) in fingers(side):
            p.stroke(P_line([b, tp]), wdt + 9.0, "ink", M_GLASS, zh, obj=2, fine=True)
        p.fill(P_ellipse(hx_, hy_, 47, 29), "ink", M_GLASS, zh, obj=2, fine=True)
        for (b, tp, wdt) in fingers(side):
            p.stroke(P_line([b, tp]), wdt, "silver", M_STATIC, zh, obj=1, fine=True)
        p.fill(P_ellipse(hx_, hy_, 42, 24), "silver", M_STATIC, zh, obj=0, fine=True)
        # knuckle creases
        for f in range(4):
            kx = hx_ + (f - 1.5) * 15 + side * 4
            p.stroke(P_line([(kx, hy_ + 4), (kx + side * 2, hy_ + 14)]), 2.5, "slate", M_GLASS, zh, obj=2, fine=True)
    # ---- the halo: a ring of screens round a disc of static, the nimbus cross = three billboards
    p.fill(P_ellipse(hx, hy, r1 + 6, r1 + 6), "ink", M_GLASS, Z_HALO, obj=255)
    nseg = 28
    for i in range(nseg):
        a0 = 2 * math.pi * i / nseg + 0.012
        a1 = 2 * math.pi * (i + 1) / nseg - 0.012
        p.fill(P_ring(hx, hy, r0 + 4, r1, a0, a1), rgb(PASTEL[i % len(PASTEL)]), M_HALO, Z_HALO, obj=i, fine=True)
    p.fill(P_ellipse(hx, hy, r0, r0), "silver", M_STATIC, Z_DISC, obj=0)
    for i, (x0, y0, x1, y1) in enumerate([(hx - r0 + 6, hy - 30, hx - HOOD[2] - 4, hy + 26),
                                           (hx + HOOD[2] + 4, hy - 30, hx + r0 - 6, hy + 26),
                                           (hx - 28, hy - r0 + 8, hx + 28, HOOD[1] - HOOD[3] - 4)]):
        p.fill(P_rect(x0 - 3, y0 - 3, x1 + 3, y1 + 3), "ink", M_GLASS, Z_NIMBUS, obj=100 + i, fine=True)
        p.fill(P_rect(x0, y0, x1, y1), ["glitch_m", "glitch_c", "slop_flesh"][i], M_BOARD, Z_NIMBUS, obj=i,
               fine=True)
    # ---- hood of dead glass framing the static face
    p.fill(P_ellipse(*HOOD), "ink", M_GLASS, Z_HOOD, obj=0)
    fx, fy, frx, fry = FACE
    p.stroke(P_ellipse(fx, fy, frx + 10, fry + 10), 5.0, "slate", M_GLASS, Z_HOOD, obj=1, fine=True)
    p.fill(P_ellipse(fx, fy, frx, fry), "silver", M_STATIC, Z_FACE, obj=0, fine=True)


def cable_paths():
    """[(pts [(x,y)...], hand side, finger)] the cables from the fingertips down to the marionettes"""
    out = []
    for side in (-1, 1):
        hx_, hy_ = HANDS[0 if side < 0 else 1]
        mine = [t for t in PUPPETS if (t[0] - AX) * side > 0]
        for f in range(5):
            fx, fy = fingers(side)[f][1]
            tx = mine[f % len(mine)][0] + (f // len(mine)) * 12 - 6
            ty = H(110.0) - 50
            out.append(([(fx, fy), (fx + (tx - fx) * 0.3, fy + (ty - fy) * 0.4), (tx, ty)], side, f))
    return out


def _cables(p):
    for pts, side, f in cable_paths():
        p.stroke(P_curve(pts), 3.4, mixc("ink", "slate", 0.3), M_GLASS, Z_CABLE, obj=f + (5 if side > 0 else 0),
                 fine=True)


# ================================================================== the sinopia
def paint_sinopia(k=0.5, seed=3):
    """bare lime plaster with the red-ochre underdrawing of the whole programme -> float32 (h, w, 3) sRGB"""
    rng = np.random.default_rng(seed)
    w, h = int(round(WX * k)), int(round(WY * k))
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    c = cairo.Context(surf)
    c.scale(k, k)
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    c.set_line_join(cairo.LINE_JOIN_ROUND)
    c.set_source_rgb(*rgb("plaster"))
    c.paint()
    sin = rgb("sinopia")

    def brush(pts, wdt=3.0, a=0.75, amp=1.6):
        """a fluid brush stroke: low-frequency wander + pressure (width) swelling mid-stroke"""
        pts = np.asarray(pts, float)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        L = seg.sum()
        if L < 1:
            return
        n = max(4, int(L / 6))
        s = np.linspace(0, L, n)
        cum = np.r_[0, np.cumsum(seg)]
        q = np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1)
        ph = rng.uniform(0, 100, 2)
        wander = np.stack([np.sin(s / 57.0 + ph[0]) + 0.5 * np.sin(s / 19.0 + ph[1]),
                           np.cos(s / 61.0 + ph[1]) + 0.5 * np.sin(s / 23.0 + ph[0])], 1) * amp
        q = q + wander
        press = (0.55 + 0.45 * np.sin(np.pi * np.clip(s / L, 0, 1)) ** 0.6) * rng.uniform(0.85, 1.2)
        c.push_group()
        c.set_source_rgba(*sin, 1.0)
        for i in range(n - 1):
            c.set_line_width(wdt * press[i])
            c.move_to(*q[i])
            c.line_to(*q[i + 1])
            c.stroke()
        c.pop_group_to_source()
        c.paint_with_alpha(a)

    # squaring grid (faint cord lines)
    for x in np.arange(0, WX + 1, 150):
        brush([(x, 0), (x, WY)], 1.1, 0.16, 0.5)
    for y in np.arange(0, WY + 1, 150):
        brush([(0, y), (WX, y)], 1.1, 0.16, 0.5)
    # ruled bands of the tituli, the dado, the ground, the cornice
    for y in (TITULUS_Y[0], TITULUS_Y[1], H(56), H(110), 30):
        brush([(0, y), (WX, y)], 2.4, 0.55, 0.9)
    # the arch: compass-struck
    for g in (RB_W * len(RAINBOW), 0.0, -INSCR_W):
        brush([arch_pt(math.pi * i / 90, g) for i in range(91)], 3.0, 0.7, 1.0)
    # letter guides + sketched letters
    for (s, xc) in ((TEXT_TITULUS[0], AX - 185 - text_width_(TEXT_TITULUS[0], 34) / 2),
                    (TEXT_TITULUS[1], AX + 185 + text_width_(TEXT_TITULUS[1], 34) / 2)):
        c.save()
        c.set_source_rgba(*sin, 0.32)
        c.set_line_width(1.6)
        text_path_centered(c, s, xc, (TITULUS_Y[0] + TITULUS_Y[1]) / 2 + 12, 34)
        c.stroke()
        c.restore()
    c.save()
    c.set_source_rgba(*sin, 0.3)
    c.set_line_width(1.4)
    text_path_centered(c, TEXT_DADO, AX, (H(56) + FLOOR) / 2 + 10, 29, tracking=0.16)
    c.stroke()
    c.restore()
    # the tower: tiers, cornices, arcades
    for k, (xc, hw, yb, yt) in enumerate(tiers()):
        brush([(xc - hw, yb), (xc - hw, yt)], 3.8, 0.85)
        brush([(xc - hw, yt), (xc + hw, yt)], 3.8, 0.85)
        brush([(xc + hw, yt), (xc + hw, yb)], 3.8, 0.85)
        brush([(xc + hw, yb - 2), (xc + hw + SIDE, yb - SIDE * 0.55), (xc + hw + SIDE, yt)], 2.2, 0.5)
        brush([(xc - hw - 10, yb - 18), (xc + hw + 10, yb - 18)], 2.2, 0.6)
        n = TIER_WIN[k]
        bay = 2 * hw / n
        for j in range(n):
            cx = xc - hw + bay * (j + 0.5)
            r = bay * 0.23
            y1 = yb - 26
            y0 = y1 - (yb - yt) * 0.52
            q = [(cx - r, y1), (cx - r, y0 + r)] + [(cx + r * math.cos(a), y0 + r - r * math.sin(a))
                                                     for a in np.linspace(math.pi, 0, 9)] + [(cx + r, y1)]
            brush(q, 2.3, 0.55, 0.8)
    xc, hw, yb, yt = CROWN
    brush([(xc - hw, yb), (xc - hw, yt), (xc + hw, yt), (xc + hw, yb)], 3.0, 0.6)
    # pentimento: the master first drew the tower two tiers taller, then struck them out
    brush([(xc - hw * 0.85, yt - 5), (xc - hw * 0.8, yt - 95), (xc + hw * 0.8, yt - 95), (xc + hw * 0.85, yt - 5)],
          2.6, 0.42)
    brush([(xc - hw * 0.55, yt - 95), (xc - hw * 0.5, yt - 170), (xc + hw * 0.5, yt - 170), (xc + hw * 0.55, yt - 95)],
          2.4, 0.36)
    brush([(xc - hw * 0.9, yt - 20), (xc + hw * 0.9, yt - 160)], 2.0, 0.32)
    brush([(xc - hw * 0.9, yt - 160), (xc + hw * 0.9, yt - 20)], 2.0, 0.32)
    for (a, b, k_) in ladders():
        brush([a, b], 2.0, 0.5)
    # the cities: loose blocks
    for side in (-1, 1):
        x_in, x_out = AX + side * 520, AX + side * 1470
        xa, xb = min(x_in, x_out), max(x_in, x_out)
        gy = H(110)
        brush([(xa, gy - 88), (xb, gy - 88)], 2.6, 0.6)
        for f in (0.06, 0.34, 0.62, 0.9):
            tx = xa + (xb - xa) * f
            brush([(tx - 26, gy), (tx - 26, gy - 150), (tx, gy - 194), (tx + 26, gy - 150), (tx + 26, gy)], 2.4, 0.55)
        for x in np.linspace(xa + 40, xb - 80, 7):
            hh = rng.uniform(170, 280)
            brush([(x, gy - 88), (x, gy - hh), (x + 60, gy - hh), (x + 60, gy - 88)], 1.8, 0.4)
    # the colossus as the master drew him: a figure WITH a face (he was set faceless over it)
    hx, hy = HALO_C
    brush([(hx + HALO_R[1] * math.cos(a), hy + HALO_R[1] * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 80)],
          3.0, 0.75, 0.6)
    c.set_source_rgba(*sin, 0.9)
    c.arc(hx, hy, 3, 0, 2 * math.pi)                                                 # compass point
    c.fill()
    fx, fy, frx, fry = FACE
    brush([(fx + frx * 1.05 * math.cos(a), fy + fry * 1.02 * math.sin(a)) for a in np.linspace(-0.4, 2 * math.pi - 0.2,
                                                                                               50)], 3.4, 0.8)
    for s in (-1, 1):                                                                 # brows, closed eyes
        brush([(fx + s * 14, fy - 28), (fx + s * 36, fy - 33), (fx + s * 54, fy - 25)], 2.8, 0.75, 0.6)
        brush([(fx + s * 16, fy - 10), (fx + s * 33, fy - 6), (fx + s * 49, fy - 11)], 2.3, 0.7, 0.5)
    brush([(fx - 2, fy - 22), (fx - 6, fy + 18), (fx + 7, fy + 23)], 2.6, 0.7, 0.5)    # nose
    brush([(fx - 20, fy + 45), (fx, fy + 49), (fx + 20, fy + 45)], 2.6, 0.7, 0.5)      # mouth
    brush([(fx - 38, fy + 70), (fx, fy + 92), (fx + 38, fy + 70)], 2.2, 0.5)           # beard
    for s in (-1, 1):
        brush([(AX + s * 90, SHOULDER_Y - 64), (AX + s * 230, SHOULDER_Y - 38), (AX + s * 560, SHOULDER_Y - 26),
               (AX + s * 846, SHOULDER_Y - 24)], 3.6, 0.8)
        brush([(AX + s * 240, SHOULDER_Y + 40), (AX + s * 560, SHOULDER_Y + 26), (AX + s * 846, SHOULDER_Y + 18)],
              2.8, 0.6)
        hxx, hyy = HANDS[0 if s < 0 else 1]
        brush([(hxx - 44, hyy - 14), (hxx + 44, hyy - 20), (hxx + 46, hyy + 30), (hxx - 40, hyy + 34),
               (hxx - 44, hyy - 14)], 2.4, 0.6)
        brush([(AX + s * 330, TITULUS_Y[0]), (AX + s * 230, SHOULDER_Y - 38)], 3.0, 0.7)
    # workshop notes in the margins (colour indications), as on real sinopie
    c.set_source_rgba(*sin, 0.55)
    c.select_font_face(FONT, cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_NORMAL)
    for s, x, y, sz in (("AVRVM", 250, 300, 30), ("AVRVM", 2620, 300, 30), ("CAERVLEVM", 2380, 760, 26),
                        ("CAERVLEVM", 470, 820, 26), ("VITRVM", AX + 40, H(430), 24), ("VITRVM", AX - 360, 260, 22)):
        c.set_font_size(sz)
        c.move_to(x, y)
        c.show_text(s)
    brush([(AX, 40), (AX, FLOOR - 4)], 1.3, 0.28, 0.3)                                 # plumb line
    surf.flush()
    img = _read_rgb(surf, w, h)
    # lime plaster: trowel mottling + stains
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    from vx.ease import fbm2
    n = fbm2(xx / (60 * k), yy / (60 * k), seed=seed, octaves=4).astype(np.float32)
    n2 = fbm2(xx / (9 * k), yy / (9 * k), seed=seed + 5, octaves=2).astype(np.float32)
    img *= (1.0 + 0.07 * n + 0.035 * n2)[..., None]
    return np.clip(img, 0, 1)


# ================================================================== labourers (sprites)
WORKER_POSES = ("climb_a", "climb_b", "carry_a", "carry_b", "puppet_a", "puppet_b", "wheel_a", "wheel_b", "stand")


def draw_worker(ctx, pose, tunic, skin="flesh", brick=None, s=1.0):
    """one labourer at the ctx origin (feet), ~46 units tall, y down (Byzantine short tunic, bare legs)."""
    ctx.save()
    ctx.scale(s, s)
    sk = rgb(skin)
    tu = rgb(tunic)
    hair = rgb("ink")
    legs = [(-5, 0, -4, -18), (5, 0, 4, -18)]
    arms = [(-7, -30, -12, -18), (7, -30, 12, -18)]
    if pose == "climb_a":
        legs = [(-5, -6, -4, -18), (5, 2, 4, -18)]
        arms = [(-7, -30, -9, -44), (7, -30, 10, -38)]
    elif pose == "climb_b":
        legs = [(-5, 2, -4, -18), (5, -6, 4, -18)]
        arms = [(-7, -30, -10, -38), (7, -30, 9, -44)]
    elif pose == "carry_a":
        legs = [(-7, 0, -3, -18), (6, -1, 3, -18)]
        arms = [(-7, -30, -9, -42), (7, -30, 13, -22)]
    elif pose == "carry_b":
        legs = [(-3, -1, -3, -18), (3, 0, 3, -18)]
        arms = [(-7, -30, -9, -42), (7, -30, 12, -24)]
    elif pose == "puppet_a":
        legs = [(-8, -3, -4, -18), (6, 0, 4, -18)]
        arms = [(-7, -31, -17, -44), (7, -31, 17, -38)]
    elif pose == "puppet_b":
        legs = [(-6, 0, -4, -18), (8, -4, 4, -18)]
        arms = [(-7, -31, -17, -38), (7, -31, 17, -44)]
    elif pose in ("wheel_a", "wheel_b"):
        d = 1 if pose == "wheel_a" else -1
        legs = [(-6 * d, 0, -3, -18), (6 * d, -2, 3, -18)]
        arms = [(-7, -30, -12, -40), (7, -30, 12, -40)]
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_source_rgb(*sk)
    ctx.set_line_width(4.2)
    for x0, y0, x1, y1 in legs:
        ctx.move_to(x1, y1)
        ctx.line_to(x0, y0)
        ctx.stroke()
    ctx.set_source_rgb(*tu)
    ctx.move_to(-8, -16)
    ctx.line_to(-6, -34)
    ctx.line_to(6, -34)
    ctx.line_to(8, -16)
    ctx.close_path()
    ctx.fill()
    ctx.set_source_rgb(*[v * 0.55 for v in tu])
    ctx.rectangle(-7, -25, 14, 2.4)
    ctx.fill()
    ctx.set_source_rgb(*sk)
    ctx.set_line_width(3.6)
    for x0, y0, x1, y1 in arms:
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y1)
        ctx.stroke()
    ctx.arc(0, -40, 5.4, 0, 2 * math.pi)
    ctx.fill()
    ctx.set_source_rgb(*hair)
    ctx.arc(0, -41.5, 5.2, math.pi, 2 * math.pi)
    ctx.fill()
    if brick is not None:
        ctx.set_source_rgb(*rgb(brick))
        ctx.rectangle(-16, -54, 13, 9)
        ctx.fill()
    ctx.restore()

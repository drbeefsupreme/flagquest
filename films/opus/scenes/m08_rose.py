"""m08 helper - the NEW picture the fallen tesserae set themselves into: the Rose of the Peoples.

A gold ground; concentric courses of small arched niches around the planted Flag (the Flag is the centre and the
compass point of the new world); in every niche ONE person, alone in their own small space, beside their own real
Flag. Flags are NOT painted here (THE ONE RULE): they are drawn with vx.flag at Rose.flags, after the mosaic.
Titulus band: SINGVLI · ET · VNIVERSI ("each one and all together", the old formula of papal letters).

Geometry is in ROSE CHART units = wall design units shifted by Rose.origin (the chart's top-left on the wall).
Angles th are measured from straight up, clockwise positive (a niche's local 'up' points away from the Flag).
"""
import math

import cairo
import numpy as np

from vx import Canvas, set_color, mix, circle, poly
from vx import icon
from vx import mosaic as mz

SM = mz.SMALTI
NICHE = ["lapis_d", "night_vault", "lapis_d", "tyrian", "night_vault", "malachite", "lapis_d", "porphyry"]
POSES = [("orans", 0.5), ("look_up", 0.2), ("clasp", 0.15), ("bless", 0.15)]
EXPRS = ["serene", "joyful", "awe", "serene", "neutral"]
TITULUS = "SINGVLI · ET · VNIVERSI"
NICHE_TILE = 7.2          # tesserae in the niches (density 1.8 on the 13-unit gold ground)
_FIGS = {}


def _people(seed):
    f = _FIGS.get(seed)
    if f is None:
        f = _FIGS[seed] = icon.Figure("people", seed=seed, tile=NICHE_TILE, halo="gold")
    return f


# region codes of part_map()
GOLD, BAND, TITULUS_BAND, GLORY, UPPER = -1, -2, -3, -4, -5


class Cell:
    __slots__ = ("id", "k", "j", "th", "ri", "ro", "wi", "wo", "poly", "niche", "ground", "side", "person", "flag",
                 "origin", "rm")


class Rose:
    """centre = wall point behind the planted Flag's foot. All radii in wall units."""

    def __init__(self, centre=(1150.0, 880.0), r0=430.0, pitch=164.0, rows=5, cell_w=118.0, gap=20.0, gap_r=22.0,
                 band=70.0, margin=(1600.0, 1150.0), top=2100.0, bottom=60.0, seed=8):
        self.centre = (float(centre[0]), float(centre[1]))
        self.r0, self.pitch, self.rows, self.gap, self.gap_r = r0, pitch, rows, gap, gap_r
        self.r1 = r0 + pitch * rows - gap_r / 2           # outer edge of the outer course of niches
        self.band = band
        self.rb0 = self.r1 + gap_r * 0.6                    # jewelled band inner radius
        self.rb1 = self.rb0 + band
        # chart: wall rect [cx - margin_x, cx + margin_x] x [cy - top, cy + bottom]
        cxw, cyw = self.centre
        self.origin = (cxw - margin[0], cyw - top)
        self.ext = (int(2 * margin[0]), int(top + bottom))
        self.cx, self.cy = margin[0], top                   # centre in chart units
        # titulus band just above the rose, cornice at the chart top edge region
        self.tband = (self.cy - self.rb1 - 150.0, 104.0)    # (y0, height) chart units
        self.upper_y = self.tband[0] - 60.0                 # above: the zone under the dome (cornice, piers)
        rng = np.random.default_rng(seed)
        self.cells = []
        for k in range(rows):
            ri = r0 + pitch * k + gap_r / 2
            ro = r0 + pitch * (k + 1) - gap_r / 2
            rm = 0.5 * (ri + ro)
            n = max(3, int(round(math.pi * rm / (cell_w + gap))))
            dth = math.pi / n
            for j in range(n):
                c = Cell()
                c.id = len(self.cells)
                c.k, c.j = k, j
                c.th = -math.pi / 2 + dth * (j + 0.5)
                c.ri, c.ro, c.rm = ri, ro, rm
                c.wi = 2 * ri * math.sin(dth / 2) - gap
                c.wo = 2 * ro * math.sin(dth / 2) - gap
                c.poly = self._sector(c.th - dth / 2, c.th + dth / 2, ri, ro, gap / 2)
                c.origin = self.at(ri, c.th)
                c.side = 1 if c.th >= 0 else -1
                c.ground = NICHE[int(rng.integers(len(NICHE)))]
                H = ro - ri
                nw = c.wi - 12.0                              # niche width (fits the inner chord)
                c.niche = (nw, H - 10.0)
                ph = 0.8 * (H - 10.0)
                pose = POSES[int(rng.choice(len(POSES), p=[q[1] for q in POSES]))][0]
                c.person = dict(seed=int(rng.integers(1 << 30)), h=ph, u=-0.19 * nw * c.side, pose=pose,
                                expr=EXPRS[int(rng.integers(len(EXPRS)))])
                L = 0.9 * (H - 10.0)
                bx, by = self.local(c, 0.1 * nw * c.side, -6.0)
                c.flag = dict(x=bx, y=by, ang=c.th, side=c.side, pole=L, seed=int(rng.integers(1 << 20)))
                self.cells.append(c)
        n = len(self.cells)
        ox, oy = self.origin
        self.flags = dict(xs=np.array([c.flag["x"] + ox for c in self.cells]),     # WALL coords
                          ys=np.array([c.flag["y"] + oy for c in self.cells]),
                          ang=np.array([c.flag["ang"] for c in self.cells]),
                          side=np.array([c.flag["side"] for c in self.cells], np.float64),
                          pole=np.array([c.flag["pole"] for c in self.cells]),
                          seed=np.array([c.flag["seed"] for c in self.cells], np.int64),
                          cell=np.arange(n))

    # ---------------------------------------------------------------- geometry (chart units)
    def at(self, r, th):
        return (self.cx + r * math.sin(th), self.cy - r * math.cos(th))

    def local(self, c, u, v):
        """niche-local (u right, v down; the floor line v=0 on the inner arc) -> chart"""
        ox, oy = c.origin
        cs, sn = math.cos(c.th), math.sin(c.th)
        return (ox + u * cs - v * sn, oy + u * sn + v * cs)

    def _sector(self, a0, a1, ri, ro, g, nseg=18):
        pts = []
        for r, rev in ((ri, False), (ro, True)):
            d = math.asin(min(0.99, g / r))
            aa = np.linspace(a0 + d, a1 - d, nseg)
            if rev:
                aa = aa[::-1]
            pts += [self.at(r, a) for a in aa]
        return pts

    def to_wall(self, x, y):
        return x + self.origin[0], y + self.origin[1]

    # ---------------------------------------------------------------- painting
    def paint(self, ctx, layer="colour"):
        """layer: 'colour' (the cartoon), 'gold' (white = gold leaf), 'fine' (white = small tiles)."""
        P = _Painter(ctx, layer)
        W_, H_ = self.ext
        P.rect(0, 0, W_, H_, SM["gold"], gold=True)
        self._upper(P)
        self._glory(P)
        self._band(P)
        self._titulus(P)
        for c in self.cells:
            self._cell(P, c)

    def _upper(self, P):
        """the zone under the dome: a cornice, then gold with dark piers/windows (seen only in the tilt blur)"""
        W_ = self.ext[0]
        y1 = self.upper_y
        P.rect(0, y1 - 34, W_, 34, SM["marble_d"])
        P.rect(0, y1 - 44, W_, 10, SM["porphyry"])
        P.rect(0, y1 - 12, W_, 12, SM["ink"])
        step = 300.0
        x = (self.cx % step) - step
        while x < W_ + step:
            # arched window between piers
            ww, top = 110.0, y1 - 420.0
            P.ctx.new_path()
            P.ctx.move_to(x - ww / 2, y1 - 60)
            P.ctx.line_to(x - ww / 2, top + ww / 2)
            P.ctx.arc(x, top + ww / 2, ww / 2, math.pi, 2 * math.pi)
            P.ctx.line_to(x + ww / 2, y1 - 60)
            P.ctx.close_path()
            P.fill(SM["night_vault"])
            P.rect(x + step / 2 - 22, 0, 44, y1 - 44, SM["porphyry"])
            x += step

    def _glory(self, P):
        cx, cy = self.cx, self.cy
        R = self.r0 - self.gap_r * 0.5
        n = 36
        for i in range(0, n, 2):
            a0 = -math.pi / 2 + math.pi * i / n
            a1 = a0 + math.pi / n
            pts = [(cx, cy)] + [self.at(R * 0.95, a) for a in np.linspace(a0, a1, 6)]
            P.poly(pts, mix(SM["marble"], SM["gold_l"], 0.25))
        # rim of the glory: lapis ring with pearls
        w = 24.0
        P.arc_band(cx, cy, R - w, R, SM["lapis"])
        npearl = int(math.pi * (R - w / 2) / 24)
        for i in range(npearl + 1):
            a = -math.pi / 2 + math.pi * i / npearl
            x, y = self.at(R - w / 2, a)
            P.circle(x, y, 6.0, SM["marble"], fine=True)

    def _band(self, P):
        cx, cy = self.cx, self.cy
        r = self.rb0
        for name, f in (("porphyry", 0.15), ("porphyry_l", 0.1), ("marble", 0.1), ("lapis", 0.65)):
            P.arc_band(cx, cy, r, r + self.band * f + 0.5, SM[name])
            r += self.band * f
        rj = self.rb0 + self.band * 0.35 + self.band * 0.325
        n = int(math.pi * rj / 42)
        for i in range(n + 1):
            a = -math.pi / 2 + math.pi * i / n
            x, y = self.at(rj, a)
            if i % 2 == 0:
                P.ctx.save()
                P.ctx.translate(x, y)
                P.ctx.rotate(a)
                P.ctx.rectangle(-8, -12, 16, 24)
                P.ctx.restore()
                P.fill(SM["porphyry_l"] if (i // 2) % 2 else SM["malachite"], fine=True)
            else:
                P.circle(x, y, 6.5, SM["marble"], fine=True)

    def _titulus(self, P):
        W_ = self.ext[0]
        y0, h = self.tband
        P.rect(0, y0, W_, h, SM["lapis_d"])
        P.rect(0, y0, W_, 9, SM["porphyry"])
        P.rect(0, y0 + h - 9, W_, 9, SM["porphyry"])
        size = h * 0.5
        t = mz.Titulus(TITULUS, self.cx, y0 + h / 2 + size / 2, size, color=SM["gold_l"], font="roman",
                       tracking=0.2)
        if P.layer == "colour":
            t.draw(P.ctx)
        elif P.layer == "fine":
            t.draw_fine(P.ctx)
        else:                                   # gold leaf letters on lapis
            t.draw_mask(P.ctx)

    def density(self, x, y):
        """tile density for Layout.flow: the niches are set in tesserae ~0.55x the gold ground's, the letters of
        the titulus (and the band around them) at half size (x, y chart units, arrays)."""
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        rho = np.hypot(x - self.cx, y - self.cy)
        inner = (rho > self.r0 - 2) & (rho < self.r1 + 2) & (y < self.cy + 2)
        y0, h = self.tband
        band = (y > y0 - 4) & (y < y0 + h + 4) & (np.abs(x - self.cx) < 900.0)
        return np.where(band, 2.0, np.where(inner, 1.8, 1.0)).astype(np.float32)

    def _cell(self, P, c):
        ctx = P.ctx
        # the course: a porphyry voussoir with the arched niche set in it
        P.poly(c.poly, SM["porphyry"] if c.ground != "porphyry" else SM["tyrian"])
        ctx.save()
        ctx.translate(*c.origin)
        ctx.rotate(c.th)
        nw, nh = c.niche
        u0 = -nw / 2
        # niche: rectangle with a round arch, a marble frame line
        for pad, colour in ((0.0, SM["marble"]), (5.0, SM[c.ground])):
            ctx.new_path()
            ctx.move_to(u0 + pad, -5 - pad * 0)
            ctx.line_to(u0 + pad, -5 - nh + nw / 2)
            ctx.arc(0, -5 - nh + nw / 2, nw / 2 - pad, math.pi, 2 * math.pi)
            ctx.line_to(-u0 - pad, -5)
            ctx.close_path()
            P.fill(colour)
        # a strip of green earth underfoot
        ctx.rectangle(u0 + 5, -19, nw - 10, 14)
        P.fill(mix(SM["malachite"], SM["verdigris"], 0.3))
        self._person(P, c)
        ctx.restore()

    # ---------------------------------------------------------------- the person of the niche (vx.icon)
    def _person(self, P, c):
        p = c.person
        layer = {"colour": "color", "gold": "gold", "fine": "fine"}[P.layer]
        fig = _people(p["seed"])
        if P.layer == "fine":     # the whole figure in opus vermiculatum (fine tiles), not just face and hands
            fig.draw(P.ctx, p["u"], -9.0, p["h"], pose=p["pose"], facing=0.28 * c.side, layer="color",
                     silhouette="#ffffff", tile=NICHE_TILE, t=0.0, blink=False)
            return
        fig.draw(P.ctx, p["u"], -9.0, p["h"], pose=p["pose"], facing=0.28 * c.side, layer=layer, tile=NICHE_TILE,
                 t=0.0, blink=False, look=(0.25 * c.side, -0.2), expr=p["expr"])


class _Painter:
    """one painting routine, three layers: colour cartoon, gold mask, fine-tile mask"""

    def __init__(self, ctx, layer):
        self.ctx, self.layer = ctx, layer

    def fill(self, colour, gold=False, fine=False, outline=None):
        ctx = self.ctx
        if self.layer == "colour":
            set_color(ctx, colour)
            if outline is not None:
                ctx.fill_preserve()
                set_color(ctx, outline)
                ctx.stroke()
            else:
                ctx.fill()
        else:
            on = gold if self.layer == "gold" else fine
            if on:
                ctx.set_source_rgb(1, 1, 1)
                ctx.fill()
            elif self.layer == "gold":
                ctx.set_operator(cairo.OPERATOR_CLEAR)
                ctx.fill()
                ctx.set_operator(cairo.OPERATOR_OVER)
            else:
                ctx.new_path()

    def stroke(self, colour, gold=False, fine=False):
        ctx = self.ctx
        if self.layer == "colour":
            set_color(ctx, colour)
            ctx.stroke()
        elif self.layer == "gold" and not gold:
            ctx.set_operator(cairo.OPERATOR_CLEAR)
            ctx.stroke()
            ctx.set_operator(cairo.OPERATOR_OVER)
        elif (self.layer == "gold" and gold) or (self.layer == "fine" and fine):
            ctx.set_source_rgb(1, 1, 1)
            ctx.stroke()
        else:
            ctx.new_path()

    def rect(self, x, y, w, h, colour, **kw):
        self.ctx.new_path()
        self.ctx.rectangle(x, y, w, h)
        self.fill(colour, **kw)

    def poly(self, pts, colour, **kw):
        self.ctx.new_path()
        poly(self.ctx, pts)
        self.fill(colour, **kw)

    def circle(self, x, y, r, colour, **kw):
        self.ctx.new_path()
        circle(self.ctx, x, y, r)
        self.fill(colour, **kw)

    def arc_band(self, cx, cy, r0, r1, colour, **kw):
        ctx = self.ctx
        ctx.new_path()
        ctx.arc(cx, cy, r1, math.pi, 2 * math.pi)
        ctx.arc_negative(cx, cy, r0, 2 * math.pi, math.pi)
        ctx.close_path()
        self.fill(colour, **kw)


class Upper:
    """the zone above the rose chart (under the dome): gold pendentive ground, the great arch that carries the
    dome (lapis soffit with pearls), porphyry ribs. Seen only in the tilt up (and mostly as streaks)."""

    def __init__(self, rose, height=2500.0, extra=1600.0):
        self.ext = (int(rose.ext[0] + 2 * extra), int(height))
        self.origin = (rose.origin[0] - extra, rose.origin[1] - height)
        self.cx = rose.cx + extra

    def paint(self, ctx, layer="colour"):
        P = _Painter(ctx, layer)
        W_, H_ = self.ext
        P.rect(0, 0, W_, H_, SM["gold"], gold=True)
        # cornice under the windows' zone
        P.rect(0, H_ - 40, W_, 40, SM["marble_d"])
        P.rect(0, H_ - 52, W_, 12, SM["porphyry"])
        # the great arch carrying the dome
        cx, cy = self.cx, H_ - 60.0
        for r0, r1, colour in ((1480.0, 1640.0, SM["lapis_d"]), (1640.0, 1664.0, SM["marble"]),
                               (1456.0, 1480.0, SM["porphyry"])):
            P.arc_band(cx, cy, r0, r1, colour)
        n = 48
        for i in range(n + 1):
            a = -math.pi / 2 + math.pi * i / n
            P.circle(cx + 1560 * math.sin(a), cy - 1560 * math.cos(a), 11.0, SM["marble"], fine=True)
        # ribs rising toward the dome
        step = 420.0
        x = (cx % step) - step
        while x < W_ + step:
            P.rect(x - 18, 0, 36, H_ - 52, SM["porphyry"])
            x += step
        # the dome's lower ring band
        P.rect(0, 0, W_, 140, SM["lapis_d"])
        P.rect(0, 140, W_, 16, SM["marble"])


def canvas(rose, s, layer="colour"):
    """paint the rose chart at pixel scale s -> Canvas"""
    w, h = int(round(rose.ext[0] * s)), int(round(rose.ext[1] * s))
    cv = Canvas(s=s, w=w, h=h)
    rose.paint(cv.ctx, layer)
    return cv


def part_map(rose, s):
    """int32 (h, w) region label at scale s: niche id >= 0, else GOLD / BAND / TITULUS_BAND / GLORY / UPPER"""
    w, h = int(round(rose.ext[0] * s)), int(round(rose.ext[1] * s))
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_NONE)
    ctx.scale(s, s)

    def code(v):
        v = v + 16
        ctx.set_source_rgb(((v >> 16) & 255) / 255.0, ((v >> 8) & 255) / 255.0, (v & 255) / 255.0)

    code(GOLD)
    ctx.paint()
    ctx.rectangle(0, 0, rose.ext[0], rose.upper_y)
    code(UPPER)
    ctx.fill()
    cx, cy = rose.cx, rose.cy
    ctx.arc(cx, cy, rose.r0 - rose.gap_r * 0.5, math.pi, 2 * math.pi)
    ctx.close_path()
    code(GLORY)
    ctx.fill()
    ctx.arc(cx, cy, rose.rb1, math.pi, 2 * math.pi)
    ctx.arc_negative(cx, cy, rose.rb0, 2 * math.pi, math.pi)
    ctx.close_path()
    code(BAND)
    ctx.fill()
    y0, hb = rose.tband
    ctx.rectangle(0, y0, rose.ext[0], hb)
    code(TITULUS_BAND)
    ctx.fill()
    for c in rose.cells:
        poly(ctx, c.poly)
        code(c.id)
        ctx.fill()
    surf.flush()
    buf = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
    v = (buf[..., 2].astype(np.int32) << 16) | (buf[..., 1].astype(np.int32) << 8) | buf[..., 0].astype(np.int32)
    return v - 16

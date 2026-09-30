"""m04 helpers around vx.mosaic v1 (the renderer) and vx.icon (the figures).

* `Paint`: paired cairo surfaces over one pixel window of a chart: colour + gold / glow / fine masks, all under the
  same transform, so a painter states what each shape is made of (gold leaf, self-lit glass, fine tiles).
  `P.figure(...)` routes a vx.icon Figure through the same four layers.
* `arrays(painter, crect, s)`: paint a chart window -> the cartoon dict mz.render wants (rgb, alpha, gold, emit).
* 2x3 transforms (camera, poses), 3D orientations of the rigid wheels, lights moved into a layer's chart.
"""
import math

import cairo
import numpy as np

from vx.config import W, H
from vx import mosaic as mz


# ============================================================ 2x3 transforms
def affine(a, b, c, d, e, f):
    """2x3 [[a, b, e], [c, d, f]]"""
    return np.array([[a, b, e], [c, d, f]], np.float64)


def compose(M1, M2):
    """M1 o M2 (apply M2 first)"""
    A = M1[:, :2] @ M2[:, :2]
    return np.hstack([A, (M1[:, :2] @ M2[:, 2] + M1[:, 2])[:, None]])


def invert(M):
    Ai = np.linalg.inv(M[:, :2])
    return np.hstack([Ai, (-Ai @ M[:, 2])[:, None]])


def sim(x=0.0, y=0.0, rot=0.0, scale=1.0, mirror=False):
    """similarity: local (0,0) -> (x,y), rotated, scaled, optionally mirrored across the local x axis"""
    c, s = math.cos(rot) * scale, math.sin(rot) * scale
    my = -1.0 if mirror else 1.0
    return affine(c, -s * my, s, c * my, x, y)


def cam_matrix(cx, cy, zoom, rot=0.0):
    """world -> screen design units (world point cx,cy at the screen centre)"""
    c, s = math.cos(rot) * zoom, math.sin(rot) * zoom
    return affine(c, -s, s, c, W / 2 - (c * cx - s * cy), H / 2 - (s * cx + c * cy))


def apply(M, x, y):
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2])


def scale_of(M):
    return math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))


def screen_window(M, margin=40.0, clip=None):
    """chart rectangle (x0, y0, x1, y1) seen through M (chart -> screen), + margin (chart units)"""
    Mi = invert(M)
    pts = np.array([apply(Mi, x, y) for x, y in ((0, 0), (W, 0), (0, H), (W, H))])
    x0, y0 = pts.min(0) - margin
    x1, y1 = pts.max(0) + margin
    if clip is not None:
        x0, y0 = max(x0, clip[0]), max(y0, clip[1])
        x1, y1 = min(x1, clip[2]), min(y1, clip[3])
    return (float(x0), float(y0), float(x1), float(y1))


def rot3(ax, ay, az):
    """R = Rz(az) @ Ry(ay) @ Rx(ax) (x right, y down, z toward the viewer)"""
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ============================================================ lights
def lights(specs):
    """[(x, y, z, rgb, power, radius), ...] (world design units) -> [mz.Light]"""
    return [mz.Light(pos=(x, y, z), color=c, power=p, radius=r) for (x, y, z, c, p, r) in specs if p > 1e-4]


def lights_to_chart(specs, M_chart_to_world):
    """world lights -> the chart of a layer placed in the world by a similarity M (z scales with it)"""
    Mi = invert(M_chart_to_world)
    k = 1.0 / scale_of(M_chart_to_world)
    out = []
    for (x, y, z, c, p, r) in specs:
        u, v = apply(Mi, x, y)
        out.append((u, v, z * k, c, p, r * k))
    return out


def lights_to_disc(specs, centre_w, R3, chart_centre):
    """world lights -> the chart of a flat disc (ring) whose chart centre sits at world centre_w, oriented R3.
    Lights behind the disc light its visible face as if mirrored (both faces are set with tiles)."""
    out = []
    for (x, y, z, c, p, r) in specs:
        q = R3.T @ np.array([x - centre_w[0], y - centre_w[1], z])
        out.append((q[0] + chart_centre[0], q[1] + chart_centre[1], abs(q[2]) + 20.0, c, p, r))
    return out


# ============================================================ painting
_MASKS = ("gold", "glow", "fine")


class Paint:
    """cairo surfaces over a pixel window of a chart, in chart design units: .c colour + mask contexts
    .g (gold leaf) .e (glow, self-lit glass) .f (fine tiles), all under the same transform."""

    def __init__(self, s, px0, py0, pw, ph, masks=_MASKS):
        self.s, self.px0, self.py0, self.pw, self.ph = s, px0, py0, pw, ph
        self.surf = {}
        self.ctx = {}
        for name in ("color",) + tuple(masks):
            sf = cairo.ImageSurface(cairo.FORMAT_ARGB32, pw, ph)
            cx = cairo.Context(sf)
            cx.translate(-px0, -py0)
            cx.scale(s, s)
            cx.set_line_join(cairo.LINE_JOIN_ROUND)
            cx.set_line_cap(cairo.LINE_CAP_ROUND)
            self.surf[name] = sf
            self.ctx[name] = cx
        self.c = self.ctx["color"]
        self.all = list(self.ctx.values())

    def push(self, M):
        mat = cairo.Matrix(M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2])
        for ctx in self.all:
            ctx.save()
            ctx.transform(mat)

    def pop(self):
        for ctx in self.all:
            ctx.restore()

    def save(self):
        for ctx in self.all:
            ctx.save()

    def restore(self):
        for ctx in self.all:
            ctx.restore()

    def clip(self, path):
        for ctx in self.all:
            ctx.new_path()
            path(ctx)
            ctx.clip()

    def _masks(self, p, a, gold, glow, fine, stroke_w=None):
        for name, v in (("gold", gold), ("glow", glow), ("fine", fine)):
            ctx = self.ctx.get(name)
            if ctx is None:
                continue
            ctx.new_path()
            ctx.append_path(p)
            ctx.set_source_rgba(v, v, v, a)
            if stroke_w is None:
                ctx.fill()
            else:
                ctx.set_line_width(stroke_w)
                ctx.stroke()

    def fill(self, path, rgb, a=1.0, gold=0.0, glow=0.0, fine=0.0):
        c = self.c
        c.new_path()
        path(c)
        p = c.copy_path()
        if isinstance(rgb, cairo.Pattern):
            c.set_source(rgb)
        else:
            c.set_source_rgba(rgb[0], rgb[1], rgb[2], a)
        c.fill()
        self._masks(p, a, gold, glow, fine)

    def stroke(self, path, rgb, width, a=1.0, gold=0.0, glow=0.0, fine=0.0):
        c = self.c
        c.new_path()
        path(c)
        p = c.copy_path()
        c.set_line_width(width)
        c.set_source_rgba(rgb[0], rgb[1], rgb[2], a)
        c.stroke()
        self._masks(p, a, gold, glow, fine, stroke_w=width)

    def text(self, s, x, y, size, rgb, a=1.0, gold=0.0, glow=0.0, fine=1.0, font="VX Cinzel", bold=False,
             align="center", tracking=0.08, rot=0.0):
        """Roman capitals (caller supplies V for U). y = baseline; size = font size. Returns the width."""
        c = self.c
        c.save()
        c.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
        c.set_font_size(size)
        adv = [c.text_extents(ch).x_advance + tracking * size for ch in s]
        wdt = sum(adv) - tracking * size
        c.restore()
        x0 = -wdt / 2 if align == "center" else (-wdt if align == "right" else 0.0)

        def path(ctx):
            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(rot)
            ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL,
                                 cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(size)
            cx = x0
            for ch, ad in zip(s, adv):
                ctx.move_to(cx, 0)
                ctx.text_path(ch)
                cx += ad
            ctx.restore()
        self.fill(path, rgb, a, gold, glow, fine)
        return wdt

    def figure(self, fig, x, y, h, fine_all=False, **kw):
        """a vx.icon Figure through every layer (its masks erase under themselves, so occlusion stays exact).
        fine_all: the whole silhouette gets fine tiles (small figures, robes that must read)."""
        anchors = fig.draw(self.c, x, y, h, layer="color", **kw)
        for name, ctx in self.ctx.items():
            if name == "color":
                continue
            if name == "fine" and fine_all:
                fig.draw(ctx, x, y, h, layer="color", silhouette=(1.0, 1.0, 1.0), **kw)
                continue
            fig.draw(ctx, x, y, h, layer=name, **kw)
        return anchors

    def arrays(self):
        """float32: rgb (unpremultiplied), alpha, and each mask (0..1)"""
        out = {}
        for name, sf in self.surf.items():
            sf.flush()
            st = sf.get_stride() // 4
            buf = np.ndarray((self.ph, st, 4), np.uint8, sf.get_data())[:, :self.pw]
            if name == "color":
                a = buf[..., 3].astype(np.float32) * (1.0 / 255.0)
                rgb = buf[..., 2::-1].astype(np.float32) * (1.0 / 255.0)
                rgb /= np.maximum(a[..., None], 1e-4)
                out["rgb"] = np.clip(rgb, 0, 1)
                out["alpha"] = a
            else:
                out[name] = buf[..., 2].astype(np.float32) * (1.0 / 255.0)
        return out


def arrays(painter, crect, s, masks=("gold", "glow"), bg=None):
    """paint `painter` (chart coords) over the chart window crect=(x0,y0,x1,y1) at pixel scale s.
    Returns dict(rgb, alpha, gold, glow, [fine], emit, crect) with crect snapped to the pixel grid."""
    x0, y0, x1, y1 = crect
    px0, py0 = math.floor(x0 * s), math.floor(y0 * s)
    pw = max(1, int(math.ceil(x1 * s)) - px0)
    ph = max(1, int(math.ceil(y1 * s)) - py0)
    P = Paint(s, px0, py0, pw, ph, masks=masks)
    if bg is not None:
        P.c.save()
        P.c.identity_matrix()
        P.c.set_source_rgb(*bg)
        P.c.paint()
        P.c.restore()
    painter(P)
    d = P.arrays()
    d["crect"] = (px0 / s, py0 / s, (px0 + pw) / s, (py0 + ph) / s)
    if "glow" in d:
        d["emit"] = d["rgb"] * d["glow"][..., None]
    return d


def guide(painter, extent, s):
    """a full-extent cartoon for Layout.flow: rgb, alpha, gold, glow, fine"""
    return arrays(painter, (0.0, 0.0, float(extent[0]), float(extent[1])), s, masks=_MASKS)


def over(dst, rgba):
    """premultiplied rgba (full frame) over dst (h,w,3), in place"""
    dst *= 1.0 - rgba[..., 3:4]
    dst += rgba[..., :3]
    return dst

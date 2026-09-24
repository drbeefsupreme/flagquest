"""t07 montage framework: render one flag-raising composition ("cell") into any screen rectangle, each art
style with its own print.  Owner: T07Raised.

Content space of a cell: height = 1000 units (y 0 top .. 1000 bottom), x in [-hw, hw] with hw = 500 * aspect,
x = 0 is the cell's centre.  A style draws its grey world in content units; the framework maps it to the cell's
pixels (so the same composition works full-frame, as a 150 px thumbnail, or zoomed 13x during a dive).

Pipeline per cell (the film's compositing order):
    grey world (cairo) -> the style's print (numpy, locked to content space) -> vx.flag spot plate
    -> front objects (hands on the pole) printed with the same print -> pasted into the frame.
Flags are ONLY drawn with vx.flag on their own layer and laid over the print with vx.ink.spot.
"""
import math

import cairo
import cv2
import numpy as np

from vx import ink
from vx.canvas import set_color

PAPER = np.array(ink.PAPER, np.float32)
INKC = np.array(ink.INK, np.float32)


# ============================================================ geometry
class Geo:
    """cell geometry: rect_px = (x0, y0, w, h) float device px of the whole cell; crop = (cx, cy, cw, ch) int
    device px actually rendered (the visible part)."""

    def __init__(self, rect_px, crop):
        self.rect = rect_px
        self.crop = crop
        self.k = rect_px[3] / 1000.0                  # device px per content unit
        self.hw = 500.0 * rect_px[2] / rect_px[3]
        self.ox = rect_px[0] - crop[0]                # crop px of the cell's left edge
        self.oy = rect_px[1] - crop[1]
        self.cw, self.ch = crop[2], crop[3]
        self.dx = 0.0                                 # composition shift (content units)
        self.zoom = 1.0                               # content push-in about (zx, zy)
        self.zx, self.zy = 0.0, 500.0
        self.mirror = False
        self._xy = None

    def ext(self):
        """(half width, top y) of the visible composition area (for background fills)"""
        z = self.zoom
        top = self.zy - self.zy / z if z != 1.0 else 0.0
        return self.hw / z + abs(self.dx) + 60.0, min(-60.0, top - 60.0)

    def vis(self):
        """visible composition rectangle (x0, y0, x1, y1) of this cell (the frame of the picture)"""
        z = self.zoom
        a, b = (-self.hw - self.zx) / z + self.zx, (self.hw - self.zx) / z + self.zx
        if self.mirror:
            a, b = -b, -a
        y0, y1 = (0 - self.zy) / z + self.zy, (1000 - self.zy) / z + self.zy
        return a - self.dx, y0, b - self.dx, y1

    def apply(self, ctx):
        ctx.translate(self.ox + self.hw * self.k, self.oy)
        ctx.scale(self.k, self.k)
        if self.zoom != 1.0:
            ctx.translate(self.zx, self.zy)
            ctx.scale(self.zoom, self.zoom)
            ctx.translate(-self.zx, -self.zy)
        if self.mirror:
            ctx.scale(-1, 1)
        ctx.translate(self.dx, 0)

    @property
    def kk(self):
        """device px per composition unit (incl. zoom)"""
        return self.k * self.zoom

    def XY(self):
        """composition coordinates of every crop pixel centre (float32 arrays, shape (ch, cw))"""
        if self._xy is None:
            xs = (np.arange(self.cw, dtype=np.float32) + 0.5 - self.ox) / self.k - self.hw
            ys = (np.arange(self.ch, dtype=np.float32) + 0.5 - self.oy) / self.k
            if self.zoom != 1.0:
                xs = (xs - self.zx) / self.zoom + self.zx
                ys = (ys - self.zy) / self.zoom + self.zy
            if self.mirror:
                xs = -xs
            xs = xs - self.dx
            X = np.broadcast_to(xs[None, :], (self.ch, self.cw))
            Y = np.broadcast_to(ys[:, None], (self.ch, self.cw))
            self._xy = (X, Y)
        return self._xy


def surface(w, h):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    c = cairo.Context(s)
    c.set_line_join(cairo.LINE_JOIN_ROUND)
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    return s, c


def surf_rgba(s):
    s.flush()
    w, h = s.get_width(), s.get_height()
    buf = np.ndarray((h, s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :w]
    out = np.empty((h, w, 4), np.float32)
    out[..., 0] = buf[..., 2]
    out[..., 1] = buf[..., 1]
    out[..., 2] = buf[..., 0]
    out[..., 3] = buf[..., 3]
    out *= 1.0 / 255.0
    return out


def tone_of(rgba, bg=1.0):
    """premultiplied RGBA canvas -> tone (luminance) over a flat bg"""
    a = rgba[..., 3]
    L = rgba[..., 0] * 0.2126 + rgba[..., 1] * 0.7152 + rgba[..., 2] * 0.0722
    return L + bg * (1.0 - a)


# ============================================================ print colours
def cov_rgb(cov):
    """ink coverage (h, w) -> printed rgb"""
    c = np.clip(cov, 0, 1)[..., None]
    return PAPER * (1 - c) + INKC * c


def grey_rgb(tone):
    """continuous tone (0 ink .. 1 paper) -> neutral grey between the two print colours"""
    t = np.clip(tone, 0, 1)[..., None]
    return INKC + (PAPER - INKC) * t


# ============================================================ noise (content-locked, fast)
_TILES = {}


def _tile(seed, n=256):
    t = _TILES.get((seed, n))
    if t is None:
        t = np.random.default_rng(seed + 911).random((n, n)).astype(np.float32) * 2 - 1
        _TILES[(seed, n)] = t
    return t


def noise(X, Y, scale, seed=0, interp=cv2.INTER_CUBIC):
    """smooth value noise in ~[-1, 1] at content coords; scale = feature size in content units"""
    t = _tile(seed)
    mx = np.ascontiguousarray((X / scale) % 256.0, dtype=np.float32)
    my = np.ascontiguousarray((Y / scale) % 256.0, dtype=np.float32)
    return cv2.remap(t, mx, my, interp, borderMode=cv2.BORDER_WRAP)


def fbm(X, Y, scale, seed=0, octaves=4, gain=0.5):
    s, a, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        s = s + a * noise(X, Y, scale / (2 ** o), seed + o * 31)
        norm += a
        a *= gain
    return s / norm


# ============================================================ print screens
def halftone_cov(dark, X, Y, cell, angle_deg, kk, min_px=2.2, shape="round"):
    """AM halftone coverage. cell = screen period in composition units (clamped so a dot cell is >= min_px
    device px; below that the screen is too fine to print and it averages to a flat grey)."""
    per_px = cell * kk
    if per_px < min_px:
        return np.clip(dark, 0, 1).astype(np.float32)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    u = (X * ca + Y * sa) / cell
    v = (-X * sa + Y * ca) / cell
    du = u - np.floor(u) - 0.5
    dv = v - np.floor(v) - 0.5
    if shape == "line":
        d = np.abs(dv) * 2
        cov = np.clip((np.clip(dark, 0, 1) - d) * per_px * 0.5 + 0.5, 0, 1)
        return cov.astype(np.float32)
    d = np.sqrt(du * du + dv * dv)
    dk = np.clip(dark, 0, 1)
    r = np.sqrt(dk / math.pi)
    # beyond 50% the dots merge: switch to white holes (proper AM chain)
    r2 = np.sqrt(np.clip(1 - dk, 0, 1) / math.pi)
    aa = 0.8 / per_px
    cov_dot = np.clip((r - d) / aa + 0.5, 0, 1)
    cov_hole = 1 - np.clip((r2 - d) / aa + 0.5, 0, 1)
    cov = np.where(dk < 0.5, cov_dot, cov_hole)
    return cov.astype(np.float32)


def line_cov(dark, X, Y, spacing, angle, kk, min_px=2.0, gamma=1.0):
    """engraved/woodcut line screen: parallel lines whose width grows with darkness. angle may be an array
    (radians, line direction = perpendicular of (cos a, sin a))."""
    per_px = spacing * kk
    if per_px < min_px:
        return np.clip(dark, 0, 1).astype(np.float32)
    ph = (X * np.cos(angle) + Y * np.sin(angle)) / spacing
    f = np.abs(ph - np.floor(ph) - 0.5) * 2
    dk = np.clip(dark, 0, 1) ** gamma
    return np.clip((dk * 1.04 - f) * per_px * 0.5 + 0.5, 0, 1).astype(np.float32)


def posterize(tone, levels):
    """snap tone to the nearest of `levels` (sorted list), soft 1-px-ish edges come from the AA of the input"""
    lv = np.asarray(levels, np.float32)
    idx = np.abs(tone[..., None] - lv).argmin(-1)
    return lv[idx]


def kuwahara(img, r):
    """painterly Kuwahara filter (grey image), r = radius in px"""
    r = max(1, int(round(r)))
    k = (r + 1, r + 1)
    m = cv2.blur(img, k, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.blur(img * img, k, borderType=cv2.BORDER_REFLECT)
    v = m2 - m * m
    h, w = img.shape
    best_m = np.zeros_like(img)
    best_v = np.full_like(img, 1e9)
    o = r // 2
    for dx, dy in ((-o, -o), (o, -o), (-o, o), (o, o)):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        mm = cv2.warpAffine(m, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        vv = cv2.warpAffine(v, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        sel = vv < best_v
        best_v = np.where(sel, vv, best_v)
        best_m = np.where(sel, mm, best_m)
    return best_m


# ============================================================ drawing helpers (content units)
def grey(ctx, g, a=1.0):
    ctx.set_source_rgba(g, g, g, a)


def P(ctx, pts, close=True):
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    if close:
        ctx.close_path()


def curve(ctx, pts, close=False, tension=0.5):
    """Catmull-Rom through pts"""
    n = len(pts)
    if n < 3:
        return P(ctx, pts, close)
    ctx.move_to(*pts[0])
    rng = range(n) if close else range(n - 1)
    for i in rng:
        p0 = pts[(i - 1) % n] if (close or i > 0) else pts[i]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if (close or i + 2 < n) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) * tension / 3, p1[1] + (p2[1] - p0[1]) * tension / 3)
        c2 = (p2[0] - (p3[0] - p1[0]) * tension / 3, p2[1] - (p3[1] - p1[1]) * tension / 3)
        ctx.curve_to(*c1, *c2, *p2)
    if close:
        ctx.close_path()


def limb(ctx, a, b, wa, wb):
    """tapered capsule from a (width wa) to b (width wb)"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1e-6
    nx, ny = -dy / L, dx / L
    ang = math.atan2(dy, dx)
    ctx.new_sub_path()
    ctx.arc(a[0], a[1], wa / 2, ang + math.pi / 2, ang + 3 * math.pi / 2)
    ctx.line_to(b[0] - nx * wb / 2, b[1] - ny * wb / 2)
    ctx.arc(b[0], b[1], wb / 2, ang - math.pi / 2, ang + math.pi / 2)
    ctx.close_path()


def ik2(s, h, l1, l2, bend=1):
    """2-bone IK: shoulder s, hand target h -> elbow position (bend = +1/-1 side)"""
    dx, dy = h[0] - s[0], h[1] - s[1]
    d = math.hypot(dx, dy)
    d = min(max(d, 1e-3), l1 + l2 - 1e-3)
    a = math.atan2(dy, dx)
    cosb = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    b = math.acos(max(-1, min(1, cosb)))
    return (s[0] + l1 * math.cos(a + bend * b), s[1] + l1 * math.sin(a + bend * b))


def pole_geom(bx, by, ang, L):
    """pole base -> top, and a function to get a point at distance d from the base"""
    sx, sy = math.sin(ang), -math.cos(ang)
    return (bx + L * sx, by + L * sy), (lambda d: (bx + d * sx, by + d * sy))


def rot_pt(p, c, a):
    ca, sa = math.cos(a), math.sin(a)
    x, y = p[0] - c[0], p[1] - c[1]
    return (c[0] + x * ca - y * sa, c[1] + x * sa + y * ca)


# ============================================================ the cell renderer
_WCACHE = {}


def fit(g, L, margin=24.0):
    """zoom out / shift a composition so the whole cloth fits a narrow cell (thumbnails)"""
    cr = L.get("cloth_right", 0.0) + margin
    if cr <= g.hw:
        return
    z = max(0.6, min(1.0, g.hw / cr))
    g.zoom, g.zx, g.zy = z, 0.0, 1000.0
    g.dx = min(0.0, g.hw / z - cr)
    g._xy = None


def render_cell(frame, fc_s, rect, style, v, cache_key=None):
    """Render `style` for raising params v into frame (h, w, 3 float32, device px) at design rect
    (x0, y0, x1, y1). fc_s = device px per design px. Returns the cell's device-px rect or None if off-screen."""
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = (r * fc_s for r in rect)
    cx0, cy0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    cx1, cy1 = min(fw, int(math.ceil(x1))), min(fh, int(math.ceil(y1)))
    if cx1 - cx0 < 1 or cy1 - cy0 < 1:
        return None
    g = Geo((x0, y0, x1 - x0, y1 - y0), (cx0, cy0, cx1 - cx0, cy1 - cy0))
    if v.get("mirror"):
        g.mirror = True
    L = style.layout(g, v)
    if not v.get("nofit"):
        fit(g, L)
    key = None
    if cache_key is not None and v.get("still") and style.static_world:
        key = (cache_key, cx0, cy0, cx1, cy1, round(x0, 1), round(y0, 1), round(x1, 1))
    img = _WCACHE.get(key) if key is not None else None
    if img is None:
        s, ctx = surface(g.cw, g.ch)
        grey(ctx, style.bg)
        ctx.paint()
        g.apply(ctx)
        style.world(ctx, g, v, L)
        img = style.prn(surf_rgba(s), g, v, L).astype(np.float32)
        if style.front is not None:
            s2, c2 = surface(g.cw, g.ch)
            g.apply(c2)
            style.front(c2, g, v, L)
            fr = surf_rgba(s2)
            a = fr[..., 3:4]
            if a.max() > 0:
                un = np.where(a > 1e-4, fr[..., :3] / np.maximum(a, 1e-4), 1.0)
                un = np.concatenate([un, np.ones_like(a)], -1)
                pf = style.prn(un, g, v, L).astype(np.float32)
                img = np.concatenate([img, pf * a, a], -1)   # + printed front (premultiplied) + its alpha
        if key is not None:
            if len(_WCACHE) > 400:
                _WCACHE.clear()
            _WCACHE[key] = img
    s3, c3 = surface(g.cw, g.ch)
    g.apply(c3)
    style.flags(c3, g, v, L)
    out = ink.spot(img[..., :3], surf_rgba(s3))
    if img.shape[-1] > 3:
        out = img[..., 3:6] + out * (1 - img[..., 6:7])
    frame[cy0:cy1, cx0:cx1] = out
    return (x0, y0, x1, y1)

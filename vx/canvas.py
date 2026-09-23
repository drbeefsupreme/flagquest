"""Cairo canvas in design units (1920x1080) + numpy compositing helpers.

Images passed between helpers are float32 numpy arrays in [0,1], sRGB, shape (h, w, 3)
or premultiplied (h, w, 4). `h, w` are PIXEL sizes (= design size * fc.s).
"""
import math

import cairo
import cv2
import numpy as np

from .config import W, H, C, FONT_SANS, hex2rgb


# ============================================================ colours
def col(c):
    """accept '#rrggbb', palette key, or tuple -> (r,g,b)"""
    if isinstance(c, str):
        return C[c] if c in C else hex2rgb(c)
    return tuple(c[:3])


def set_color(ctx, c, a=1.0):
    r, g, b = col(c)
    ctx.set_source_rgba(r, g, b, a)


def mix(c0, c1, t):
    a, b = col(c0), col(c1)
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def shade(c, k):
    """k<1 darker, k>1 lighter (towards white)"""
    r, g, b = col(c)
    if k <= 1:
        return (r * k, g * k, b * k)
    t = min(k - 1, 1)
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


# ============================================================ canvas
class Canvas:
    """A transparent (or filled) cairo surface. `ctx` is pre-scaled so you draw in design units."""

    def __init__(self, fc=None, s=None, bg=None, w=None, h=None):
        self.s = s if s is not None else (fc.s if fc is not None else 1.0)
        self.w = w or int(round(W * self.s))
        self.h = h or int(round(H * self.s))
        self.surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)
        self.ctx = cairo.Context(self.surf)
        self.ctx.scale(self.s, self.s)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        if bg is not None:
            self.fill(bg)

    def fill(self, c, a=1.0):
        self.ctx.save()
        self.ctx.identity_matrix()
        set_color(self.ctx, c, a)
        self.ctx.paint()
        self.ctx.restore()

    def clear(self):
        self.ctx.save()
        self.ctx.set_operator(cairo.OPERATOR_CLEAR)
        self.ctx.paint()
        self.ctx.restore()

    def rgba(self):
        """float32 premultiplied (h,w,4) RGBA"""
        self.surf.flush()
        buf = np.ndarray((self.h, self.surf.get_stride() // 4, 4), np.uint8, self.surf.get_data())
        a = buf[:, : self.w]
        out = np.empty((self.h, self.w, 4), np.float32)
        out[..., 0] = a[..., 2]
        out[..., 1] = a[..., 1]
        out[..., 2] = a[..., 0]
        out[..., 3] = a[..., 3]
        out *= 1.0 / 255.0
        return out

    def rgb(self):
        """flattened over black -> float32 (h,w,3)"""
        return self.rgba()[..., :3].copy()

    def over(self, dst):
        """composite this canvas over numpy image dst (h,w,3) -> new array"""
        return over(dst, self.rgba())

    def paint_array(self, arr, x=0.0, y=0.0, alpha=1.0, operator=None):
        """draw a pixel-space numpy image (h,w,3|4) with its top-left at design coords (x,y)"""
        paint_array(self.ctx, arr, x, y, self.s, alpha, operator)


def surface_from_array(arr):
    """float32 (h,w,3) or premultiplied (h,w,4) -> cairo ImageSurface"""
    h, w = arr.shape[:2]
    bgra = np.empty((h, w, 4), np.uint8)
    a8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    bgra[..., 0] = a8[..., 2]
    bgra[..., 1] = a8[..., 1]
    bgra[..., 2] = a8[..., 0]
    bgra[..., 3] = a8[..., 3] if arr.shape[2] == 4 else 255
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    if stride != w * 4:
        pad = np.zeros((h, stride // 4, 4), np.uint8)
        pad[:, :w] = bgra
        bgra = pad
    surf = cairo.ImageSurface.create_for_data(memoryview(np.ascontiguousarray(bgra)), cairo.FORMAT_ARGB32, w, h, stride)
    surf._keep = bgra  # keep buffer alive
    return surf


def paint_array(ctx, arr, x, y, s, alpha=1.0, operator=None):
    surf = surface_from_array(arr)
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(1.0 / s, 1.0 / s)
    ctx.set_source_surface(surf, 0, 0)
    if operator is not None:
        ctx.set_operator(operator)
    ctx.paint_with_alpha(alpha)
    ctx.restore()


# ============================================================ numpy compositing
def over(dst, src):
    """dst (h,w,3) under premultiplied src (h,w,4)"""
    return src[..., :3] + dst * (1.0 - src[..., 3:4])


def over_alpha(dst_rgba, src_rgba):
    """premultiplied over premultiplied"""
    return src_rgba + dst_rgba * (1.0 - src_rgba[..., 3:4])


def screen(dst, src):
    return 1.0 - (1.0 - dst) * (1.0 - src)


def add(dst, src, k=1.0):
    return dst + src * k


def multiply(dst, src):
    return dst * src


def solid(fc_or_shape, c):
    if hasattr(fc_or_shape, "h"):
        h, w = fc_or_shape.h, fc_or_shape.w
    else:
        h, w = fc_or_shape
    img = np.empty((h, w, 3), np.float32)
    img[:] = col(c)
    return img


def blur(img, sigma):
    """gaussian blur, sigma in PIXELS; downsamples for big radii"""
    if sigma <= 0.3:
        return img
    if sigma > 12:
        k = sigma / 6.0
        h, w = img.shape[:2]
        sw, sh = max(1, int(w / k)), max(1, int(h / k))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 6.0)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def glow(rgba, sigma, strength=1.0):
    """return additive glow (h,w,3) from a premultiplied layer"""
    return blur(rgba[..., :3], sigma) * strength


def gradient_v(fc, stops):
    """vertical gradient image; stops [(t, colour), ...]"""
    h, w = fc.h, fc.w
    ys = np.linspace(0, 1, h, dtype=np.float32)
    ts = [s[0] for s in stops]
    cs = np.array([col(s[1]) for s in stops], np.float32)
    col_y = np.stack([np.interp(ys, ts, cs[:, k]) for k in range(3)], 1)
    return np.repeat(col_y[:, None, :], w, axis=1)


def radial_mask(fc, cx, cy, r0, r1):
    """1 inside r0 -> 0 beyond r1 (design units)"""
    s = fc.s
    yy, xx = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
    d = np.hypot(xx / s - cx, yy / s - cy)
    t = np.clip((d - r0) / max(r1 - r0, 1e-6), 0, 1)
    return (1 - t * t * (3 - 2 * t))[..., None]


def warp_affine(img, M, fc):
    """apply 2x3 affine (design units) to a pixel image"""
    s = fc.s
    S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float64)
    Si = np.linalg.inv(S)
    M3 = np.vstack([M, [0, 0, 1]])
    Mp = (S @ M3 @ Si)[:2]
    return cv2.warpAffine(img, Mp, (fc.w, fc.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ============================================================ path helpers
def rect(ctx, x, y, w, h):
    ctx.rectangle(x, y, w, h)


def rrect(ctx, x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def circle(ctx, x, y, r):
    ctx.new_sub_path()
    ctx.arc(x, y, r, 0, 2 * math.pi)
    ctx.close_path()


def ellipse(ctx, x, y, rx, ry, rot=0.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    ctx.scale(max(rx, 1e-6), max(ry, 1e-6))
    ctx.new_sub_path()
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.close_path()
    ctx.restore()


def poly(ctx, pts, close=True):
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    if close:
        ctx.close_path()


def smooth(ctx, pts, close=False, tension=0.5):
    """Catmull-Rom spline through pts as cubic beziers"""
    n = len(pts)
    if n < 3:
        poly(ctx, pts, close)
        return
    P = list(pts)
    if close:
        get = lambda i: P[i % n]
        rng = range(n)
    else:
        get = lambda i: P[max(0, min(n - 1, i))]
        rng = range(n - 1)
    ctx.move_to(*P[0])
    k = tension / 3.0 * 2
    for i in rng:
        p0, p1, p2, p3 = get(i - 1), get(i), get(i + 1), get(i + 2)
        c1 = (p1[0] + (p2[0] - p0[0]) * k / 2, p1[1] + (p2[1] - p0[1]) * k / 2)
        c2 = (p2[0] - (p3[0] - p1[0]) * k / 2, p2[1] - (p3[1] - p1[1]) * k / 2)
        ctx.curve_to(*c1, *c2, *p2)
    if close:
        ctx.close_path()


def lin_grad(x0, y0, x1, y1, stops):
    """stops: [(t, colour) or (t, colour, alpha)]"""
    g = cairo.LinearGradient(x0, y0, x1, y1)
    for s in stops:
        r, gg, b = col(s[1])
        g.add_color_stop_rgba(s[0], r, gg, b, s[2] if len(s) > 2 else 1.0)
    return g


def rad_grad(cx, cy, r0, r1, stops, fx=None, fy=None):
    g = cairo.RadialGradient(fx if fx is not None else cx, fy if fy is not None else cy, r0, cx, cy, r1)
    for s in stops:
        r, gg, b = col(s[1])
        g.add_color_stop_rgba(s[0], r, gg, b, s[2] if len(s) > 2 else 1.0)
    return g


# ============================================================ text
def _face(ctx, font, bold, italic, size):
    ctx.select_font_face(font, cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)


def text_width(ctx, s, size, font=FONT_SANS, bold=False, italic=False, tracking=0.0):
    _face(ctx, font, bold, italic, size)
    if tracking == 0:
        return ctx.text_extents(s).x_advance
    return sum(ctx.text_extents(ch).x_advance + tracking * size for ch in s) - tracking * size


def text(ctx, s, x, y, size, color="ink", alpha=1.0, font=FONT_SANS, bold=False, italic=False,
         align="left", tracking=0.0, valign="baseline", stroke=None, stroke_w=0.0):
    """draw text; align left|center|right, valign baseline|middle|top; tracking in em"""
    ctx.save()
    _face(ctx, font, bold, italic, size)
    wdt = text_width(ctx, s, size, font, bold, italic, tracking)
    if align == "center":
        x -= wdt / 2
    elif align == "right":
        x -= wdt
    fe = ctx.font_extents()
    if valign == "middle":
        y += (fe[0] - fe[1]) / 2 * 0.9
    elif valign == "top":
        y += fe[0]
    ctx.new_path()
    if tracking == 0:
        ctx.move_to(x, y)
        ctx.text_path(s)
    else:
        cx = x
        for ch in s:
            ctx.move_to(cx, y)
            ctx.text_path(ch)
            cx += ctx.text_extents(ch).x_advance + tracking * size
    if stroke is not None and stroke_w > 0:
        set_color(ctx, stroke, alpha)
        ctx.set_line_width(stroke_w)
        ctx.stroke_preserve()
    set_color(ctx, color, alpha)
    ctx.fill()
    ctx.restore()
    return wdt

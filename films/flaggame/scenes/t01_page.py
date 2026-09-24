"""THE TRACT PAGE — shared page design for the present-day tract scenes (t01, t02, t08, t09).  Owner: T01Tract.

A Chick-tract page like the Flagazine scans (ref/page_23..27.png): a portrait page of dark stock (#2B2B2D)
with three wide white panels, thin black keylines, and a page number in the outer bottom corner.
Everything is expressed in PAGE UNITS (PU): the page is 1000 x 1298 PU (US-letter aspect, like the scans).

Geometry
--------
PAGE_W, PAGE_H = 1000, 1298
MARGIN_X = 70, MARGIN_TOP = 60, MARGIN_BOTTOM = 118, GUTTER = 32
PANEL_W, PANEL_H = 860, 352                      (aspect 2.443 : 1)
PANELS = {"p1": (70, 60, 860, 352), "p2": (70, 444, 860, 352), "p3": (70, 828, 860, 352)}   (x, y, w, h)
split(rect, frac=0.5, gap=GUTTER) -> (left_rect, right_rect)      two-up rows (like page 24)
BORDER = 4.0                                     panel keyline width (PU), solid ink
PAGE_RGB = #2B2B2D (0.169, 0.169, 0.176)        the page stock outside the panels (flat, NOT halftoned)
page number: Nimbus Sans 26 PU, light grey (0.86), outer bottom corner:
    odd (right-hand) pages bottom-right (right edge x = PAGE_W - 52), even pages bottom-left (x = 52),
    baseline y = PAGE_H - 46.

Camera (page -> screen)
-----------------------
PageCam(cx, cy, zoom, rot=0.0): page point (cx, cy) at screen centre, `zoom` design px per PU.
    .apply(ctx)           push page->screen transform on a design-scaled cairo ctx (save/restore yourself)
    .to_screen(px, py) / .to_page(sx, sy)
    .ink_kw()             kwargs for vx.ink.tract so the halftone screen is LOCKED TO THE PAPER:
                          dict(space="page", origin=..., zoom=..., lpi=TRACT_LPI, angle=45)
                          (dots are TRACT_CELL = 4.6 PU; ~9.7 design px at reading zoom, bigger when closer)
frame(rect, margin=0.04, fit="contain"|"cover"|"width"|"height", bias=(0, 0)) -> PageCam
    frames a page rect on screen (reading framing = frame(PANELS["p1"]) -> zoom 2.054, panel 1767 x 723 px).
lerp_cam(a, b, t) -> PageCam       (zoom interpolated geometrically so glides feel constant-speed)

Rendering a page (the standard pipeline, one call)
--------------------------------------------------
render_page(fc, cam, panels, page_no=1, T=None, style="tract", letters=True) -> float32 rgb
    panels: {rect_or_panel_id: drawer}. A drawer is any object with optional methods (all in PAGE units;
    the ctx already carries the camera transform AND a clip to the panel rect; T = global seconds):
        world(ctx, T, rect)      grey world (white = paper, black = solid ink)
        flags(ctx, T, rect)      ONLY vx.flag drawings (the spot-yellow plate)
        front(ctx, T, rect)      foreground objects that may pass in front of a flag (hands, heads);
                                 drawn on a transparent layer, printed with ink.print_layer
        letters(ctx, fc, T, rect)   lettering (vx.comic) — NOT clipped (balloons may break borders)
    Order: world -> ink.tract(page-locked dots) -> spot(flags) -> print_layer(front) -> page stock,
    keylines, page number -> letters.  Panels outside the view are skipped.
draw_page_frame(ctx, page_no, panel_rects, border=BORDER)   the page stock with panel holes + keylines + number
"""
import math

import cairo
import numpy as np

from vx import W, H, FONT_SANS, Canvas, set_color
from vx import ink, comic

PAGE_W, PAGE_H = 1000.0, 1298.0
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM, GUTTER = 70.0, 60.0, 118.0, 32.0
PANEL_W, PANEL_H = 860.0, 352.0
PANELS = {
    "p1": (MARGIN_X, MARGIN_TOP, PANEL_W, PANEL_H),
    "p2": (MARGIN_X, MARGIN_TOP + PANEL_H + GUTTER, PANEL_W, PANEL_H),
    "p3": (MARGIN_X, MARGIN_TOP + 2 * (PANEL_H + GUTTER), PANEL_W, PANEL_H),
}
BORDER = 4.0
PAGE_HEX = "#2B2B2D"
PAGE_RGB = (0x2B / 255.0, 0x2B / 255.0, 0x2D / 255.0)
NUM_RGB = (0.86, 0.86, 0.86)
NUM_SIZE = 26.0
TRACT_CELL = 4.6                       # halftone cell in page units
TRACT_LPI = 1080.0 / TRACT_CELL        # vx.ink lpi convention: dots per 1080 (page) units
TRACT_ANGLE = 45.0
# lettering sizes in PAGE units (the lettering is printed on the page and scales with it)
LETTER = dict(balloon=15.0, offpanel=15.0, radio=15.0, shout=18.0, caption=14.0, footnote=9.5, title=30.0)


def inset(rect, d):
    x, y, w, h = rect
    return (x + d, y + d, w - 2 * d, h - 2 * d)


def split(rect, frac=0.5, gap=GUTTER):
    x, y, w, h = rect
    wl = (w - gap) * frac
    return (x, y, wl, h), (x + wl + gap, y, w - gap - wl, h)


class PageCam:
    __slots__ = ("cx", "cy", "zoom", "rot")

    def __init__(self, cx, cy, zoom, rot=0.0):
        self.cx, self.cy, self.zoom, self.rot = float(cx), float(cy), float(zoom), float(rot)

    def apply(self, ctx):
        ctx.translate(W / 2, H / 2)
        if self.rot:
            ctx.rotate(self.rot)
        ctx.scale(self.zoom, self.zoom)
        ctx.translate(-self.cx, -self.cy)

    def to_screen(self, px, py):
        dx, dy = (px - self.cx) * self.zoom, (py - self.cy) * self.zoom
        if self.rot:
            c, s = math.cos(self.rot), math.sin(self.rot)
            dx, dy = dx * c - dy * s, dx * s + dy * c
        return W / 2 + dx, H / 2 + dy

    def to_page(self, sx, sy):
        dx, dy = sx - W / 2, sy - H / 2
        if self.rot:
            c, s = math.cos(-self.rot), math.sin(-self.rot)
            dx, dy = dx * c - dy * s, dx * s + dy * c
        return self.cx + dx / self.zoom, self.cy + dy / self.zoom

    def ink_kw(self, **kw):
        d = dict(space="page", origin=(self.cx - W / 2 / self.zoom, self.cy - H / 2 / self.zoom),
                 zoom=self.zoom, lpi=TRACT_LPI, angle=TRACT_ANGLE)
        d.update(kw)
        return d

    def view_rect(self):
        """visible page rect (x0, y0, x1, y1) (ignores rot)"""
        x0, y0 = self.to_page(0, 0)
        x1, y1 = self.to_page(W, H)
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def __repr__(self):
        return f"PageCam({self.cx:.1f}, {self.cy:.1f}, {self.zoom:.4f})"


def frame(rect, margin=0.04, fit="contain", bias=(0.0, 0.0)):
    x, y, w, h = rect
    zw = W * (1 - 2 * margin) / w
    zh = H * (1 - 2 * margin) / h
    z = {"contain": min(zw, zh), "cover": max(zw, zh), "width": zw, "height": zh}[fit]
    return PageCam(x + w / 2 + bias[0] * w, y + h / 2 + bias[1] * h, z)


def lerp_cam(a, b, t):
    t = float(t)
    z = math.exp(math.log(a.zoom) + (math.log(b.zoom) - math.log(a.zoom)) * t)
    # keep the screen motion of the target point smooth under zoom: interpolate centre linearly
    return PageCam(a.cx + (b.cx - a.cx) * t, a.cy + (b.cy - a.cy) * t, z, a.rot + (b.rot - a.rot) * t)


def _rect_of(k):
    return PANELS[k] if isinstance(k, str) else tuple(k)


def draw_page_number(ctx, page_no):
    if page_no is None:
        return
    s = str(page_no)
    ctx.save()
    ctx.select_font_face(FONT_SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(NUM_SIZE)
    wdt = ctx.text_extents(s).x_advance
    x = PAGE_W - 52 - wdt if int(page_no) % 2 == 1 else 52
    ctx.move_to(x, PAGE_H - 46)
    set_color(ctx, NUM_RGB)
    ctx.show_text(s)
    ctx.restore()


def draw_page_frame(ctx, page_no, panel_rects, border=BORDER, stock=True):
    """page stock (#2B2B2D) with holes for the panels, black keylines, page number. Page units."""
    ctx.save()
    if stock:
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.rectangle(0, 0, PAGE_W, PAGE_H)
        for r in panel_rects:
            ctx.rectangle(*r)
        set_color(ctx, PAGE_RGB)
        ctx.fill()
    set_color(ctx, (0, 0, 0))
    ctx.set_line_width(border)
    ctx.set_line_join(cairo.LINE_JOIN_MITER)
    for r in panel_rects:
        ctx.rectangle(*r)
    ctx.stroke()
    ctx.restore()
    draw_page_number(ctx, page_no)


def _visible(cam, rect, pad=0.0):
    vx0, vy0, vx1, vy1 = cam.view_rect()
    x, y, w, h = rect
    return x + w >= vx0 - pad and x <= vx1 + pad and y + h >= vy0 - pad and y <= vy1 + pad


def render_page(fc, cam, panels, page_no=1, T=None, style="tract", letters=True, frame_layer=True):
    T = fc.T if T is None else T
    items = [(_rect_of(k), d) for k, d in panels.items()]
    vis = [(r, d) for r, d in items if _visible(cam, r)]

    world = Canvas(fc, bg=(1, 1, 1))
    for r, d in vis:
        if hasattr(d, "world"):
            c = world.ctx
            c.save()
            cam.apply(c)
            c.rectangle(*r)
            c.clip()
            d.world(c, T, r)
            c.restore()
    printer = {"tract": ink.tract, "engrave": ink.engrave, "manga": ink.manga}[style] if isinstance(style, str) else style
    img = printer(fc, world.rgb(), **cam.ink_kw()) if printer is ink.tract or printer is ink.manga else printer(fc, world.rgb())

    fl = None
    if any(hasattr(d, "flags") for _, d in vis):
        fl = Canvas(fc)
        for r, d in vis:
            if hasattr(d, "flags"):
                c = fl.ctx
                c.save()
                cam.apply(c)
                c.rectangle(*inset(r, BORDER / 2))   # never let the keyline touch flag cloth
                c.clip()
                d.flags(c, T, r)
                c.restore()
        img = ink.spot(img, fl.rgba())

    if any(hasattr(d, "front") for _, d in vis):
        fr = Canvas(fc)
        for r, d in vis:
            if hasattr(d, "front"):
                c = fr.ctx
                c.save()
                cam.apply(c)
                c.rectangle(*r)
                c.clip()
                d.front(c, T, r)
                c.restore()
        lay = fr.rgba()
        if lay[..., 3].max() > 0:
            front = ink.print_layer(fc, lay, "tract", **cam.ink_kw())
            img = front[..., :3] + img * (1.0 - front[..., 3:4])
    top = Canvas(fc)
    if frame_layer:
        top.ctx.save()
        cam.apply(top.ctx)
        draw_page_frame(top.ctx, page_no, [r for r, _ in items])
        top.ctx.restore()
    if letters:
        lt = Canvas(fc)
        for r, d in vis:
            if hasattr(d, "letters"):
                lt.ctx.save()
                cam.apply(lt.ctx)
                d.letters(lt.ctx, fc, T, r)
                lt.ctx.restore()
        img = top.over(img)
        la = lt.rgba()
        if fl is not None:
            la = comic.protect(la, fl.rgba())      # absolute guard: no lettering on flag cloth
        return la[..., :3] + img * (1 - la[..., 3:4])
    return top.over(img)

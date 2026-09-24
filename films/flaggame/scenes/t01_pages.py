"""t01 — the physical tract's printed pages as textures (page units -> pixels).  Owner: T01Tract.

Every page is drawn in PAGE UNITS (t01_page: 1000 x 1298 PU), printed with vx.ink.tract with the dot screen
locked to the paper, and the flags laid on as the spot plate.  A texture is float32 (h, w, 4):
rgb = the printed page (flags spotted in), a = flag-cloth coverage (the renderer keeps rain, beads, mud,
specular and shadows off every flag texel).

  back_page(ppu)                 the back cover = the LOOP FRAME's visible page (ref page 27 + IVG stamp box)
  cover(ppu, T)                  THE FLAG GAME? cover; its printed flag is ALIVE (redrawn at T)
  inside_cover(ppu)              copyright / publisher block
  page(n, ppu, T)                story pages 1..8 (page 1 = the camp/Crow/icon page read in t01/t02)
  inside_back(ppu)
Faces (leaf k front/back) -> textures: FACES[k] = (front, back).
"""
import math

import cairo
import cv2
import numpy as np

from vx import Canvas, set_color, W as DW
from vx import ink, comic
from vx.flag import draw_flag

from . import t01_page as P

PW, PH = P.PAGE_W, P.PAGE_H
FACES = [("cover", "ii"), (1, 2), (3, 4), (5, 6), (7, 8), ("iii", "back")]
COVER_FRAME = (34.0, 34.0, PW - 68.0, PH - 68.0)


class TexFC:
    """a frame-context stand-in for printing a texture: pixel grid = page units * ppu"""

    def __init__(self, ppu, F=0, T=0.0):
        self.w = int(round(PW * ppu))
        self.h = int(round(PH * ppu))
        self.s = 1.0
        self.F = F
        self.f = F
        self.T = T
        self.t = T
        self.ppu = ppu


def _canvas(tfc, bg=(1, 1, 1)):
    c = Canvas(s=tfc.ppu, w=tfc.w, h=tfc.h, bg=bg)
    return c


def _ink_kw(tfc):
    return dict(space="page", origin=(0.0, 0.0), zoom=tfc.ppu, lpi=P.TRACT_LPI, angle=P.TRACT_ANGLE)


def txt(ctx, s, x, y, size, family=None, align="left", color=(0, 0, 0), alpha=1.0):
    f = comic.face(family)
    w = f.width(s, size)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    ctx.new_path()
    f.path(ctx, s, x, y, size)
    set_color(ctx, color, alpha)
    ctx.fill()
    return w


def fit_size(s, width, family=None, size=100.0):
    f = comic.face(family)
    return size * width / max(1e-6, f.width(s, size))


def compose(tfc, world_fn, flags_fn=None, top_fn=None, stock=True, panels=(), stock_fn=None):
    """world (grey, page units) -> tract print -> spot flags -> page stock + keylines -> top layer (lettering)
    -> (h, w, 4) texture"""
    wc = _canvas(tfc)
    world_fn(wc.ctx)
    img = ink.tract(tfc, wc.rgb(), **_ink_kw(tfc))
    fa = np.zeros((tfc.h, tfc.w), np.float32)
    if flags_fn is not None:
        fl = Canvas(s=tfc.ppu, w=tfc.w, h=tfc.h)
        flags_fn(fl.ctx)
        rgba = fl.rgba()
        img = ink.spot(img, rgba)
        fa = rgba[..., 3]
    top = Canvas(s=tfc.ppu, w=tfc.w, h=tfc.h)
    if stock:
        P.draw_page_frame(top.ctx, None, list(panels)) if stock_fn is None else stock_fn(top.ctx)
    if top_fn is not None:
        top_fn(top.ctx)
    t = top.rgba()
    # lettering / stock never over a flag (absolute guard)
    if fa.max() > 0:
        keep = 1.0 - np.clip(fa * 1.5, 0, 1)[..., None]
        t = t * keep
    img = t[..., :3] + img * (1 - t[..., 3:4])
    return np.concatenate([img, fa[..., None]], -1).astype(np.float32)


# ============================================================================== THE BACK PAGE (loop frame)
LOREM = ["Lorem ipsum dolor sit amet,", "consectetur adipiscing elit, sed do", "eiusmod tempor incididunt ut labore",
         "et dolore magna aliqua. Quis ipsum", "suspendisse ultrices gravida. Risus", "commodo viverra maecenas",
         "accumsan lacus vel facilisis."]
BLASPHEME = "So never forget, FLAGS are the HEART and SOUL of the Burn itself, and speaking against them is BLASPHEME."


def _stamp_layer(tfc, cx, cy, rot, seed=7):
    """the rubber-stamp impression 'IVG' (black stamp ink, uneven, with speckle voids) -> premult rgba"""
    c = Canvas(s=tfc.ppu, w=tfc.w, h=tfc.h)
    ctx = c.ctx
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(rot)
    bw, bh = 330.0, 150.0
    set_color(ctx, (0, 0, 0))
    ctx.set_line_width(9.0)
    P_r = 16.0
    from vx import rrect
    rrect(ctx, -bw / 2, -bh / 2, bw, bh, P_r)
    ctx.stroke()
    ctx.set_line_width(3.0)
    rrect(ctx, -bw / 2 + 12, -bh / 2 + 12, bw - 24, bh - 24, P_r * 0.6)
    ctx.stroke()
    size = fit_size("IVG", bw - 120, comic.FONT_TITLE, 100)
    f = comic.face(comic.FONT_TITLE)
    capH = f.cap * size if f.cap > 0 else size * 0.72
    txt(ctx, "IVG", 0, capH / 2 - 4, size, comic.FONT_TITLE, "center")
    # small stars either side
    for sx in (-bw / 2 + 36, bw / 2 - 36):
        _star(ctx, sx, 0, 13, 5.5)
    ctx.fill()
    ctx.restore()
    rgba = c.rgba()
    a = rgba[..., 3]
    # uneven stamp ink: low-freq density + speckle voids + a dry edge on one side
    rng = np.random.default_rng(seed)
    h, w = a.shape
    lo = cv2.resize(rng.random((max(2, h // 60), max(2, w // 60))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    hi = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.3 * tfc.ppu)
    hi = (hi - hi.mean()) / (hi.std() + 1e-6)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = np.clip(((xs / tfc.ppu - cx) * math.cos(rot) + (ys / tfc.ppu - cy) * math.sin(rot)) / 200.0 + 0.5, 0, 1)
    dens = np.clip(1.25 - 0.30 * lo - 0.35 * ramp + 0.10 * hi, 0, 1)
    dens *= np.clip((hi + 1.5 + 0.8 * (1 - ramp)) * 3.0, 0, 1)
    a2 = a * dens
    out = np.zeros_like(rgba)
    out[..., :3] = np.array(ink.INK, np.float32) * a2[..., None]
    out[..., 3] = a2 * 0.96
    return out


def _star(ctx, x, y, r, ri):
    pts = []
    for k in range(10):
        ang = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else ri
        pts.append((x + math.cos(ang) * rr, y + math.sin(ang) * rr))
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()


def back_page(ppu=1.3):
    tfc = TexFC(ppu)
    box = (70.0, 60.0, 860.0, 352.0)

    def world(ctx):
        x, y, w, h = box
        # a sliver of the previous panel's art (dark) above the caption strip, like the scan
        set_color(ctx, (0.07, 0.07, 0.07))
        ctx.rectangle(x, y, w, 30)
        ctx.fill()
        rng = np.random.default_rng(27)
        for k in range(9):
            set_color(ctx, (0.25 + 0.2 * rng.random(),) * 3)
            ctx.arc(x + 60 + k * 95 + rng.random() * 30, y + 30, 14 + rng.random() * 18, math.pi, 2 * math.pi)
            ctx.fill()

    def top(ctx):
        x, y, w, h = box
        # caption strip
        sy = y + 30
        set_color(ctx, (1, 1, 1))
        ctx.rectangle(x + 5, sy, w - 10, 30)
        ctx.fill()
        set_color(ctx, (0, 0, 0))
        ctx.set_line_width(2.4)
        ctx.rectangle(x + 5, sy, w - 10, 30)
        ctx.stroke()
        fs = fit_size(BLASPHEME, w - 36, comic.FONT_LETTER, 14)
        txt(ctx, BLASPHEME, x + w / 2, sy + 21.5, min(fs, 15.0), comic.FONT_LETTER, "center")
        # lorem ipsum column (heavy condensed, centred)
        lx = x + w * 0.265
        ly = sy + 30 + 52
        for i, line in enumerate(LOREM):
            txt(ctx, line, lx, ly + i * 37.5, 27.5, comic.FONT_CONDENSED, "center")
        # the rule
        ctx.set_line_width(4.0)
        ctx.move_to(x + w * 0.535, sy + 48)
        ctx.line_to(x + w * 0.535, y + h - 22)
        ctx.stroke()
        # FLAGS FLAGS column
        fx = x + w * 0.58
        row = "FLAGS FLAGS FLAGS FLAGS"
        fsz = fit_size(row, w * 0.37, comic.FONT_LETTER, 17)
        for i in range(14):
            txt(ctx, row, fx, sy + 30 + 30 + i * 18.2, fsz, comic.FONT_LETTER)
        # "To Be Continued..." (the film loops)
        txt(ctx, "To Be Continued...", PW / 2, 700, 66, "Nimbus Sans", "center", color=(0.08, 0.08, 0.08))
        # THIS TRACT PLACED BY: box
        bx, by, bw, bh = 245.0, 930.0, 510.0, 250.0
        set_color(ctx, (1, 1, 1))
        ctx.rectangle(bx, by, bw, bh)
        ctx.fill()
        set_color(ctx, (0, 0, 0))
        ctx.set_line_width(3.0)
        ctx.rectangle(bx, by, bw, bh)
        ctx.stroke()
        txt(ctx, "THIS TRACT PLACED BY:", bx + 16, by + 30, 19, comic.FONT_LETTER)
        ctx.set_line_width(1.2)
        for k in range(3):
            yy = by + 150 + k * 36
            ctx.move_to(bx + 20, yy)
            ctx.line_to(bx + bw - 20, yy)
        set_color(ctx, (0.35, 0.35, 0.35))
        ctx.stroke()

    def stock(ctx):
        # the back cover is white cover stock inside a heavy black frame (like the front cover)
        set_color(ctx, (0, 0, 0))
        ctx.set_line_width(10.0)
        ctx.rectangle(*COVER_FRAME)
        ctx.stroke()
        ctx.set_line_width(4.0)
        ctx.rectangle(*box)
        ctx.stroke()
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.rectangle(0, 0, PW, PH)
        ctx.rectangle(*COVER_FRAME)
        set_color(ctx, P.PAGE_RGB)
        ctx.fill()

    tex = compose(tfc, world, None, top, stock=True, panels=[box], stock_fn=stock)
    st = _stamp_layer(tfc, 540.0, 1072.0, math.radians(-7.5))
    tex[..., :3] = st[..., :3] + tex[..., :3] * (1 - st[..., 3:4])
    return tex


# ============================================================================== INSIDE COVERS
def inside_cover(ppu=1.0):
    tfc = TexFC(ppu)
    box = (170.0, 820.0, 660.0, 300.0)

    def world(ctx):
        pass

    def top(ctx):
        x, y, w, h = box
        set_color(ctx, (1, 1, 1))
        ctx.rectangle(x, y, w, h)
        ctx.fill()
        set_color(ctx, (0, 0, 0))
        ctx.set_line_width(3)
        ctx.rectangle(x, y, w, h)
        ctx.stroke()
        lines = ["THE FLAG GAME?", "", "Copyright \u00a9 IVG. All rights reserved. Printed in the rain.",
                 "FLAGS are plain YELLOW with no other markings upon them.",
                 "Survey FLAGS is strictly an interactive art installation.", "",
                 "Additional copies: find a FLAG and ask."]
        yy = y + 58
        for i, l in enumerate(lines):
            if not l:
                yy += 16
                continue
            sz = 34 if i == 0 else 17
            fam = comic.FONT_TITLE if i == 0 else comic.FONT_BODY
            txt(ctx, l, x + w / 2, yy, sz, fam, "center")
            yy += 44 if i == 0 else 27

    def stock(ctx):
        set_color(ctx, P.PAGE_RGB)
        ctx.rectangle(0, 0, PW, PH)
        ctx.fill()

    return compose(tfc, world, None, top, stock=True, stock_fn=stock)


def inside_back(ppu=1.0):
    tfc = TexFC(ppu)

    def top(ctx):
        txt(ctx, "IVG", PW / 2, PH / 2 + 30, 120, comic.FONT_TITLE, "center", color=(0.20, 0.20, 0.21))

    def stock(ctx):
        set_color(ctx, P.PAGE_RGB)
        ctx.rectangle(0, 0, PW, PH)
        ctx.fill()

    return compose(tfc, lambda c: None, None, top, stock=True, stock_fn=stock)


# ============================================================================== THE COVER
COVER_FLAG = dict(x=700.0, y=1135.0, pole=640.0, wind=-1.05, seed=5)   # the passing flag (base = pole bottom)
PIC = (70.0, 440.0, 860.0, 700.0)


def _burst_path(ctx, cx, cy, rx, ry, n=22, jag=0.22, seed=3):
    rng = np.random.default_rng(seed)
    pts = []
    for k in range(n * 2):
        a = 2 * math.pi * k / (n * 2)
        r = 1.0 + (jag * (0.7 + 0.6 * rng.random()) if k % 2 == 0 else -jag * 0.35)
        pts.append((cx + math.cos(a) * rx * r, cy + math.sin(a) * ry * r))
    ctx.new_path()
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()


def cover_static(ppu=1.3):
    """the printed cover WITHOUT its flag (the flag is laid on live: cover_flag)"""
    tfc = TexFC(ppu)
    x0, y0, pw, ph = PIC

    def world(ctx):
        # picture: a Chick radial burst behind Summer
        ctx.save()
        ctx.rectangle(*PIC)
        ctx.clip()
        set_color(ctx, (0.93, 0.93, 0.93))
        ctx.paint()
        cx, cy = 380.0, 760.0
        rng = np.random.default_rng(4)
        for k in range(64):
            a0 = 2 * math.pi * k / 64 + rng.random() * 0.03
            a1 = a0 + 2 * math.pi / 64 * (0.35 + 0.3 * rng.random())
            ctx.move_to(cx, cy)
            ctx.line_to(cx + math.cos(a0) * 1400, cy + math.sin(a0) * 1400)
            ctx.line_to(cx + math.cos(a1) * 1400, cy + math.sin(a1) * 1400)
            ctx.close_path()
        set_color(ctx, (0.62, 0.62, 0.62))
        ctx.fill()
        # ground line / mud at the bottom of the picture
        set_color(ctx, (0.80, 0.80, 0.80))
        ctx.rectangle(x0, y0 + ph - 90, pw, 90)
        ctx.fill()
        # Summer, laughing her head off at the flag (the locked rig look)
        from .t01_camp import summer, place
        ch = summer()
        x, y, h = place(ch, (340.0, 720.0), 150.0, "stand", 0.4, 1, turn=0.45)
        ch.draw(ctx, x, y, h, "point", 0.4, mouth=1.0, expr="happy", facing=1, look=(1.0, -0.2), turn=0.45,
                tilt=-0.12, blink=True)
        ctx.restore()

    def top(ctx):
        # frame
        set_color(ctx, (0, 0, 0))
        ctx.set_line_width(10.0)
        ctx.rectangle(*COVER_FRAME)
        ctx.stroke()
        ctx.set_line_width(5.0)
        ctx.rectangle(*PIC)
        ctx.stroke()
        # title in a burst
        ctx.save()
        ctx.translate(500, 250)
        ctx.rotate(math.radians(-3.5))
        _burst_path(ctx, 0, 0, 430, 185, n=19, jag=0.20)
        set_color(ctx, (1, 1, 1))
        ctx.fill_preserve()
        set_color(ctx, (0, 0, 0))
        ctx.set_line_width(8.0)
        ctx.stroke()
        s1 = fit_size("THE FLAG", 600, comic.FONT_TITLE, 100)
        s2 = fit_size("GAME?", 520, comic.FONT_TITLE, 100)
        txt(ctx, "THE FLAG", 0, -12, s1, comic.FONT_TITLE, "center")
        txt(ctx, "GAME?", 0, -12 + s2 * 0.95, s2, comic.FONT_TITLE, "center")
        ctx.restore()
        # laughter
        comic.sfx(ctx, "HA HA!", 250, 560, 64, rot=-0.18)
        comic.sfx(ctx, "HA!", 590, 930, 48, rot=0.2)
        # publisher mark + price
        bx, by = 72.0, 1170.0
        set_color(ctx, (0, 0, 0))
        ctx.rectangle(bx, by, 150, 70)
        ctx.fill()
        txt(ctx, "IVG", bx + 75, by + 55, fit_size("IVG", 120, comic.FONT_TITLE, 60), comic.FONT_TITLE, "center",
            color=(1, 1, 1))
        ctx.save()
        ctx.translate(PW - 150, 1205)
        ctx.rotate(math.radians(8))
        _burst_path(ctx, 0, 0, 95, 60, n=14, jag=0.18, seed=9)
        set_color(ctx, (1, 1, 1))
        ctx.fill_preserve()
        set_color(ctx, (0, 0, 0))
        ctx.set_line_width(5)
        ctx.stroke()
        txt(ctx, "FREE", 0, 16, 46, comic.FONT_LETTER, "center")
        ctx.restore()

    def stock(ctx):
        # cover stock: white with the page's dark around the frame edge
        set_color(ctx, (1, 1, 1))
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.rectangle(0, 0, PW, PH)
        ctx.rectangle(*COVER_FRAME)
        set_color(ctx, P.PAGE_RGB)
        ctx.fill()

    return compose(tfc, world, None, top, stock=True, stock_fn=stock)


def cover_flag(ppu, T):
    """the cover's printed flag, alive: premultiplied rgba flag layer at texture size"""
    tfc = TexFC(ppu)
    fl = Canvas(s=ppu, w=tfc.w, h=tfc.h)
    ctx = fl.ctx
    ctx.save()
    ctx.rectangle(PIC[0] + 3, PIC[1] + 3, PIC[2] - 6, PIC[3] - 6)
    ctx.clip()
    f = COVER_FLAG
    draw_flag(ctx, f["x"], f["y"], pole=f["pole"], t=T, wind=f["wind"], side=-1, ang=-0.16, seed=f["seed"],
              light=(-0.45, -0.8, 0.4))
    ctx.restore()
    return fl.rgba()


def with_flag(tex, flag_rgba):
    out = tex.copy()
    out[..., :3] = ink.spot(tex[..., :3], flag_rgba)
    out[..., 3] = np.maximum(tex[..., 3], flag_rgba[..., 3])
    return out


# ============================================================================== STORY PAGES (glimpsed in the riffle)
def story_page(n, ppu=0.9, panels=None, T=6.5):
    """page n in the tract layout.  panels: {pid_or_rect: drawer} (t01_page protocol, page units);
    missing panels get a simple glimpse (tone + caption) so the booklet always looks printed."""
    tfc = TexFC(ppu, T=T)
    rects = dict(P.PANELS)
    drawers = panels or {}

    class _TFC:
        pass

    lfc = TexFC(ppu, T=T)
    lfc.tl = _tl()

    def world(ctx):
        for pid, r in rects.items():
            d = drawers.get(pid)
            ctx.save()
            ctx.rectangle(*r)
            ctx.clip()
            if d is not None and hasattr(d, "world"):
                d.world(ctx, T, r)
            else:
                _glimpse(ctx, n, pid, r)
            ctx.restore()

    def flags(ctx):
        for pid, r in rects.items():
            d = drawers.get(pid)
            ctx.save()
            ctx.rectangle(*P.inset(r, P.BORDER / 2))
            ctx.clip()
            if d is not None and hasattr(d, "flags"):
                d.flags(ctx, T, r)
            elif d is None:
                _glimpse_flags(ctx, n, pid, r)
            ctx.restore()

    def top(ctx):
        P.draw_page_frame(ctx, n, list(rects.values()))
        for pid, r in rects.items():
            d = drawers.get(pid)
            if d is not None and hasattr(d, "letters"):
                try:
                    d.letters(ctx, lfc, T, r)
                except Exception:
                    pass
            elif d is None:
                _glimpse_caption(ctx, n, pid, r)

    tex = compose(tfc, world, flags, top, stock=False)
    # front layers (heads over poles) of real drawers
    fr = Canvas(s=ppu, w=tfc.w, h=tfc.h)
    any_front = False
    for pid, r in rects.items():
        d = drawers.get(pid)
        if d is not None and hasattr(d, "front"):
            any_front = True
            fr.ctx.save()
            fr.ctx.rectangle(*r)
            fr.ctx.clip()
            d.front(fr.ctx, T, r)
            fr.ctx.restore()
    if any_front:
        lay = ink.print_layer(tfc, fr.rgba(), "tract", **_ink_kw(tfc))
        tex[..., :3] = lay[..., :3] + tex[..., :3] * (1 - lay[..., 3:4])
        # re-apply the page frame + lettering on top
        top_c = Canvas(s=ppu, w=tfc.w, h=tfc.h)
        top(top_c.ctx)
        t = top_c.rgba()
        keep = 1.0 - np.clip(tex[..., 3:4] * 1.5, 0, 1)
        t = t * keep
        tex[..., :3] = t[..., :3] + tex[..., :3] * (1 - t[..., 3:4])
    return tex


_TLD = {}


def _tl():
    if "tl" not in _TLD:
        from vx.timeline import get_timeline
        _TLD["tl"] = get_timeline()
    return _TLD["tl"]


GLIMPSE = {
    2: [("p1", "anime"), ("p2", "crow"), ("p3", "hexflags")],
    3: [("p1", "crowd"), ("p2", "storm"), ("p3", "candles")],
    4: [("p1", "shutup"), ("p2", "flagsflood"), ("p3", "devil")],
    5: [("p1", "listened"), ("p2", "effigy"), ("p3", "firmament")],
    6: [("p1", "fire"), ("p2", "deluge"), ("p3", "raised")],
    7: [("p1", "camp"), ("p2", "lightning"), ("p3", "kneel")],
    8: [("p1", "pray"), ("p2", "passiton"), ("p3", "camp")],
}
CAPTIONS = {
    "hexflags": "In the beginning there were 6 FLAGS",
    "crowd": "That year, the terrible STENCH of the hippies...",
    "candles": "The FLAG MAKER walked amongst them...",
    "listened": "BUT THERE WERE SOME THAT LISTENED...",
    "firmament": "...and the rain stopped entirely.",
    "raised": "HOWEVER, the YELLOW FLAG would be RAISED AGAIN...",
    "kneel": "What must I do?",
    "pray": "Lorem ipsum dolor sit amet...",
    "passiton": "Now go... and PASS IT ON.",
    "deluge": "and THE RAIN RETURNED.",
    "effigy": "AND IT WORKED.",
}


def _kind(n, pid):
    for p, k in GLIMPSE.get(n, []):
        if p == pid:
            return k
    return "camp"


def _glimpse(ctx, n, pid, r):
    import cairo as _c
    x, y, w, h = r
    k = _kind(n, pid)
    rng = np.random.default_rng(n * 10 + int(pid[1]))
    set_color(ctx, (0.94, 0.94, 0.94))
    ctx.rectangle(x, y, w, h)
    ctx.fill()
    if k in ("storm", "deluge", "firmament", "fire", "lightning"):
        g = _c.LinearGradient(0, y, 0, y + h)
        g.add_color_stop_rgb(0, 0.25 if k != "firmament" else 0.85, 0.25 if k != "firmament" else 0.85,
                             0.25 if k != "firmament" else 0.85)
        g.add_color_stop_rgb(1, 0.75, 0.75, 0.75)
        ctx.set_source(g)
        ctx.rectangle(x, y, w, h)
        ctx.fill()
        if k == "storm":
            cx, cy = x + w * 0.55, y + h * 0.45
            set_color(ctx, (0.05, 0.05, 0.05))
            ctx.set_line_width(3)
            for j in range(14):
                ctx.new_path()
                for q in range(60):
                    a = q * 0.18 + j * 0.45
                    rr = 8 + q * 3.2
                    ctx.line_to(cx + math.cos(a) * rr * 1.6, cy + math.sin(a) * rr * 0.6)
                ctx.stroke()
        if k == "lightning":
            set_color(ctx, (1, 1, 1))
            ctx.move_to(x + w * 0.5, y)
            for q in range(6):
                ctx.line_to(x + w * (0.45 + 0.1 * rng.random()), y + h * (q + 1) / 6)
            ctx.set_line_width(7)
            ctx.stroke()
    if k in ("crowd", "shutup", "candles", "camp", "kneel", "pray", "passiton", "anime", "crow", "devil",
             "effigy", "fire", "deluge", "raised", "hexflags", "flagsflood", "listened", "firmament"):
        pass
    # generic figures / motifs
    set_color(ctx, (0, 0, 0))
    if k in ("crowd", "shutup", "deluge"):
        for j in range(18):
            hx = x + 30 + j * (w - 60) / 17 + rng.random() * 10
            hy = y + h * (0.55 + 0.25 * rng.random())
            set_color(ctx, (0.55 + 0.35 * rng.random(),) * 3)
            ctx.new_path()
            ctx.arc(hx, hy, 18 + rng.random() * 8, 0, 2 * math.pi)
            ctx.fill()
            ctx.rectangle(hx - 22, hy + 18, 44, h)
            ctx.fill()
            set_color(ctx, (0, 0, 0))
            ctx.set_line_width(2)
            ctx.new_path()
            ctx.arc(hx, hy, 18, 0, 2 * math.pi)
            ctx.stroke()
    elif k == "hexflags":
        cx, cy = x + w * 0.5, y + h * 0.62
        for j in range(19):
            q = j % 5 - 2
            rr = j // 5 - 2
            hx = cx + q * 70 + (rr % 2) * 35
            hy = cy + rr * 30
            ctx.new_path()
            for a in range(6):
                ctx.line_to(hx + 34 * math.cos(a * math.pi / 3), hy + 16 * math.sin(a * math.pi / 3))
            ctx.close_path()
            set_color(ctx, (0, 0, 0) if (j % 3) else (1, 1, 1))
            ctx.fill_preserve()
            set_color(ctx, (0, 0, 0))
            ctx.set_line_width(2)
            ctx.stroke()
    elif k in ("effigy", "fire", "raised", "firmament"):
        ex, ey = x + w * 0.5, y + h - 10
        set_color(ctx, (0, 0, 0))
        for (ax, ay, bx, by, wd) in ((ex - 46, ey, ex - 8, ey - 150, 16), (ex + 46, ey, ex + 8, ey - 150, 16),
                                    (ex, ey - 150, ex, ey - 240, 26), (ex, ey - 212, ex - 70, ey - 292, 12),
                                    (ex, ey - 212, ex + 70, ey - 292, 12)):
            _lattice(ctx, ax, ay, bx, by, wd)
        ctx.new_path()
        ctx.arc(ex, ey - 262, 20, 0, 2 * math.pi)
        set_color(ctx, (1, 1, 1))
        ctx.fill_preserve()
        set_color(ctx, (0, 0, 0))
        ctx.set_line_width(3)
        ctx.stroke()
        if k == "fire":
            for j in range(9):
                set_color(ctx, (1, 1, 1))
                ctx.new_path()
                fx = ex + (j - 4) * 22
                ctx.move_to(fx - 16, ey)
                ctx.curve_to(fx - 20, ey - 60, fx + 5, ey - 120 - 30 * rng.random(), fx + 4, ey - 170)
                ctx.curve_to(fx + 10, ey - 90, fx + 22, ey - 50, fx + 16, ey)
                ctx.fill_preserve()
                set_color(ctx, (0, 0, 0))
                ctx.set_line_width(2)
                ctx.stroke()
        if k == "firmament":
            set_color(ctx, (0, 0, 0))
            ctx.set_line_width(2)
            for j in range(8):
                ctx.new_path()
                ctx.arc(ex, ey - 240, 120 + j * 40, math.pi, 2 * math.pi)
                ctx.stroke()
    elif k in ("devil",):
        set_color(ctx, (0.05, 0.05, 0.05))
        ctx.rectangle(x + w * 0.35, y + 40, w * 0.3, h)
        ctx.fill()
        ctx.move_to(x + w * 0.42, y + 120)
        ctx.line_to(x + w * 0.40, y + 60)
        ctx.line_to(x + w * 0.46, y + 110)
        ctx.move_to(x + w * 0.58, y + 120)
        ctx.line_to(x + w * 0.60, y + 60)
        ctx.line_to(x + w * 0.54, y + 110)
        ctx.fill()
    elif k in ("candles",):
        for j in range(7):
            cx = x + 70 + j * (w - 140) / 6
            set_color(ctx, (0, 0, 0))
            ctx.rectangle(cx - 5, y + 110, 10, h - 120)
            ctx.fill()
            set_color(ctx, (1, 1, 1))
            ctx.new_path()
            ctx.arc(cx, y + 96, 12, 0, 2 * math.pi)
            ctx.fill_preserve()
            set_color(ctx, (0, 0, 0))
            ctx.set_line_width(2)
            ctx.stroke()
    elif k == "flagsflood":
        pass
    # the cast (locked rig look) in medium shots
    for fig in CAST.get(k, []):
        ch, (fx, fy, fh), pose, facing, kw = _fig(fig, r)
        ch.draw(ctx, fx, fy, fh, pose, 0.7, facing=facing, **kw)


_GCH = {}


def _char(kind, seed, **st):
    key = (kind, seed, tuple(sorted(st.items())))
    if key not in _GCH:
        from vx.chars import Character
        _GCH[key] = Character(kind, seed, render="ink", **st)
    return _GCH[key]


# (kind, seed, style, pose, head_x_frac, head_y_frac, head_radius_frac, facing, draw kw, flag pole_len (x h) or 0)
CAST = {
    "anime": [("summer", 1, {}, "stand", 0.33, 0.50, 0.30, 1, dict(expr="bewildered", face="anime", mouth=0.5), 0)],
    "crow": [("crow", 3, {"hat": "cap"}, "point", 0.55, 0.42, 0.20, -1, dict(expr="fervent", mouth=0.6), 0)],
    "crowd": [("hippie", sd, {"mud": 0.6}, "stand", 0.12 + 0.19 * i, 0.42 + 0.05 * (i % 2), 0.10, 1 if i % 2 else -1,
               dict(expr="neutral"), 0) for i, sd in enumerate((2, 5, 8, 11, 14))],
    "candles": [("flagmaker", 1, {}, {"walk": 1.0, "shoulder_flag": 1.0}, 0.64, 0.26, 0.075, 1, dict(expr="serene"), 1.3)],
    "shutup": [("pharisee", sd, {}, "stand", 0.18 + 0.21 * i, 0.46, 0.13, -1 if i < 2 else 1,
                dict(expr="furious" if i == 1 else "stern", mouth=0.6 if i == 1 else 0.0), 0)
               for i, sd in enumerate((1, 2, 3, 4))],
    "devil": [("devil", 1, {"chair": "throne", "prop": "lyre"}, "throne", 0.5, 0.30, 0.10, 1, dict(expr="possessed"), 0)],
    "raised": [("hippie", 4, {"mud": 0.4}, "raise_flag", 0.34, 0.34, 0.08, 1, dict(expr="determined"), 1.4)],
    "kneel": [("summer", 1, {}, {"kneel": 1.0}, 0.30, 0.50, 0.16, 1, dict(expr="horror", mouth=0.8), 0),
              ("crow", 3, {"hat": "cap"}, "preach", 0.72, 0.32, 0.12, -1, dict(expr="grave"), 0)],
    "pray": [("summer", 1, {}, {"kneel": 1.0, "pray": 1.0}, 0.45, 0.45, 0.20, 1, dict(expr="beatific"), 0)],
    "passiton": [("crow", 3, {"hat": "cap"}, "offer_tract", 0.30, 0.36, 0.15, 1, dict(expr="fervent", mouth=0.3), 0),
                 ("summer", 1, {}, "stand", 0.72, 0.40, 0.15, -1, dict(expr="rapt"), 0)],
    "camp": [("raven", 1, {"chair": "camp"}, "sit", 0.26, 0.42, 0.12, 1, dict(expr="disdain"), 0),
             ("summer", 1, {"chair": "camp"}, "sit", 0.72, 0.42, 0.12, -1, dict(expr="cheerful"), 0)],
    "effigy": [("seraph", 1, {}, {"fly": 1.0, "hold_flag": 1.0}, 0.70, 0.30, 0.06, -1, dict(expr="serene"), 1.2)],
    "listened": [],
}


def _fig(fig, r):
    from .t01_camp import place
    x, y, w, h = r
    kind, seed, st, pose, hx, hy, hr, facing, kw, flag = fig
    ch = _char(kind, seed, **st)
    kw = dict(kw)
    fx, fy, fh = place(ch, (x + hx * w, y + hy * h), hr * h, pose, 0.7, facing)
    if flag:
        kw["pole_len"] = flag * fh * 0.6
    return ch, (fx, fy, fh), pose, facing, kw


def _lattice(ctx, ax, ay, bx, by, wd):
    """a lattice-lumber member (the effigy): two rails + zigzag bracing"""
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L * wd / 2, dx / L * wd / 2
    ctx.set_line_width(2.6)
    for sgn in (-1, 1):
        ctx.move_to(ax + nx * sgn, ay + ny * sgn)
        ctx.line_to(bx + nx * sgn, by + ny * sgn)
    k = max(2, int(L / (wd * 0.9)))
    for i in range(k):
        u0, u1 = i / k, (i + 1) / k
        s0 = -1 if i % 2 == 0 else 1
        ctx.move_to(ax + dx * u0 + nx * s0, ay + dy * u0 + ny * s0)
        ctx.line_to(ax + dx * u1 - nx * s0, ay + dy * u1 - ny * s0)
    ctx.stroke()


def _glimpse_flags(ctx, n, pid, r):
    x, y, w, h = r
    k = _kind(n, pid)
    for fig in CAST.get(k, []):
        if fig[-1]:
            ch, (fx, fy, fh), pose, facing, kw = _fig(fig, r)
            a = ch.anchors(fx, fy, fh, pose, 0.7, facing=facing, pole_len=kw["pole_len"])
            if "pole" not in a:
                continue
            bx, by, ang = a["pole"]
            draw_flag(ctx, bx, by, pole=kw["pole_len"], t=0.7 + n, wind=0.7 * (1 if facing < 0 else -1),
                      ang=ang, seed=n)
    if k in ("candles", "raised", "effigy"):
        return
    if k == "hexflags":
        cx, cy = x + w * 0.5, y + h * 0.62
        for a in range(6):
            fx = cx + 150 * math.cos(a * math.pi / 3 + 0.3)
            fy = cy + 60 * math.sin(a * math.pi / 3 + 0.3)
            draw_flag(ctx, fx, fy, pole=150, t=a * 0.37, wind=0.4, seed=a)
    elif k == "firmament":
        draw_flag(ctx, x + w * 0.5, y + h - 250, pole=110, t=1.0, wind=0.5, seed=3)
    elif k in ("crow",):
        draw_flag(ctx, x + w * 0.85, y + h - 10, pole=300, t=0.3, wind=0.6, seed=7, ang=-0.12)
    elif k == "fire":
        draw_flag(ctx, x + w * 0.5 + 30, y + 90, pole=90, t=0.2, wind=0.3, seed=2, ang=0.4)


def _glimpse_caption(ctx, n, pid, r):
    x, y, w, h = r
    k = _kind(n, pid)
    if k == "flagsflood":
        comic.flags_flood(ctx, (x + 4, y + 4, w - 8, h - 8), 1.0, size=30)
        return
    cap = CAPTIONS.get(k)
    if cap:
        if k == "listened":
            sz = fit_size(cap, w - 80, comic.FONT_TITLE, 40)
            txt(ctx, cap, x + w / 2, y + h / 2 + sz * 0.35, sz, comic.FONT_TITLE, "center")
            return
        comic.caption(ctx, cap, x + 14, y + 14, min(w - 28, 420), size=19)

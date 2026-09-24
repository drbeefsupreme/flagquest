"""t07 — RAISED AGAIN (142.04–159.83).  Owner: T07Raised.

142.04  smash cut on 'HOWEVER': a Soviet propaganda poster (ref page 26 panel 3); the worker heaves the Flag up
        on 'RAISED' (raise_1 143.944) — the P21 caption in the tract's condensed face, a rolling counter.
143.94  one hundred raisings: full-frame cuts through the art styles, accelerating (count 1,2,3,5,8,13,21,34)
145.14  '50': the frame becomes a tract page of fifty panels (every raising so far)
145.86  'then': four more styles (8-bit, woodcut, Seurat, blueprint), then the page again: every panel has split
        in two and the last ones pop in — the 100th lands on 'time' (raise_100 146.769) in the panel we dive into
147.9   the dive into panel #100 = the Doré engraving of the lone flag-bearer (t07_walker)
149.58  P22/P23: the walker crosses the changing Burns, the Burn without fire, the years without Burns
158.8   unmask: it is Crow; the engraving reprints into tract halftone; hand-off to t08 at 159.83
Compositing per cell: grey world -> style print -> vx.flag spot plate -> printed front objects; lettering on top.
"""
import math

import numpy as np

from vx import *
from vx import ink, comic
from vx.flag import Flag
from vx.foley import export_events
from scenes import t07_timing as TT
from scenes.t07_cells import render_cell, surface, surf_rgba
from scenes.t07_montage import STYLES, ORDER
from scenes import t07_walker as WK

POST = dict(bloom=0.0, grain=0.0, vignette=0.08)

PAGE_DARK = 0x2B / 255.0
# ---------------------------------------------------------------- the tract page grid (design units)
M, GUT = 22.0, 10.0
NC, NR = 12, 10
CW = (1920 - 2 * M - (NC - 1) * GUT) / NC
RH = (1080 - 2 * M - (NR - 1) * GUT) / NR


def post(fc, st):
    """per-frame film look (also shadows the vx.post module that `from vx import *` drags in)"""
    if fc.T >= DIVE1:
        return WK.post(fc, st["walker"])
    return {}


def _cx(c):
    return M + c * (CW + GUT)


def _ry(r):
    return M + r * (RH + GUT)


CAP = (_cx(0), _ry(0), _cx(4) + CW, _ry(1) + RH)          # caption slot: 5 cols x 2 rows (top-left)
CNT = (_cx(7), _ry(8), _cx(11) + CW, _ry(9) + RH)         # counter slot: 5 cols x 2 rows (bottom-right)
CAP_SIZE = 56
DOUBLE = []                                               # the 50 portrait panels (reading order)
for pr in range(5):
    for c in range(NC):
        if (pr == 0 and c <= 4) or (pr == 4 and c >= 7):
            continue
        DOUBLE.append((_cx(c), _ry(2 * pr), _cx(c) + CW, _ry(2 * pr + 1) + RH))
assert len(DOUBLE) == 50
CELL = {}                                                 # raising k -> its panel on the page of 100
for j, (x0, y0, x1, y1) in enumerate(DOUBLE):
    CELL[j + 1] = (x0, y0, x1, y0 + RH)
    CELL[j + 51] = (x0, y0 + RH + GUT, x1, y1)

DIVE0, DIVE1 = 147.85, 149.45                             # the dive into panel #100
FULL = (0.0, 0.0, 1920.0, 1080.0)


# ---------------------------------------------------------------- who is raised how
def _assign():
    hero = {}
    for t0, st in TT.CUTS:
        if not st.startswith("PAGE"):
            k = max(1, int(TT.count(t0 + 1e-4) + 1e-6))
            while k in hero:
                k += 1
            hero[k] = st
    style = {}
    n = len(ORDER)
    for k in range(1, 101):
        style[k] = ORDER[(k * 5 + (k - 1) // 12 * 3) % n]
    style.update(hero)
    style[100] = "dore"
    return style, hero


STYLE_OF, HERO = _assign()
MIRRORABLE = ("lascaux", "iwo", "moon", "seurat", "woodcut", "delacroix")
CUT_K = {st: k for k, st in HERO.items()}


def _v(k, T, t, still=False, pop_t=None):
    v = dict(t=t, T=T, seed=k * 7 + 3, k=k, still=still)
    if pop_t is not None:
        v["pop_t"] = pop_t
    if still and k not in HERO and hash01(k, 77) < 0.35 and STYLE_OF.get(k) in MIRRORABLE:
        v["mirror"] = True
    return v


# ---------------------------------------------------------------- setup
def setup(S):
    st = {}
    # the opener's hero flag: furled on the lowered pole, unfurled by the heave on 'RAISED'
    sov = STYLES["soviet"]
    T0 = S.start
    n = int((TT.CUTS[1][0] - T0) * S.fps) + 4

    def pole(i):
        T = T0 + i / S.fps
        L = sov.layout(None, dict(t=T - TT.R1, T=T))
        return L["pole"][0], L["pole"][1], L["pole"][2]

    def wind(i):
        T = T0 + i / S.fps
        return (760.0 if T > TT.R1 else 380.0, -30.0)

    def furl(i):
        T = T0 + i / S.fps
        return 1.0 - clamp((T - TT.R1 + 0.02) / 0.3)

    st["sov_track"] = Flag(pole=1150.0, seed=12).simulate(n, pole, wind, fps=S.fps, furl_fn=furl)
    st["walker"] = WK.setup(S)
    WK.bind(st["walker"])
    _export(S, st)
    return st


def _export(S, st):
    ev = TT.export(S) + WK.events(S, st["walker"])
    export_events(S, "hits", ev)
    try:
        st["sov_track"].export_foley(S, "soviet_flag", f0=0)
    except Exception:
        pass


# ---------------------------------------------------------------- lettering furniture
def _caption(ctx, fc, T=None):
    return comic.say(ctx, fc, "P21", CAP[0], CAP[1], w=CAP[2] - CAP[0], size=CAP_SIZE, anchor="tl", T=T)


def _counter(ctx, T, rect=CNT, alpha=1.0):
    """rolling odometer in tract caps: '37 TIME' ... '100 TIME!'"""
    c = TT.count(T)
    if c < 1:
        return
    x0, y0, x1, y1 = rect
    n = int(c + 1e-6)
    ctx.save()
    set_color(ctx, (1, 1, 1), alpha)
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.fill_preserve()
    set_color(ctx, (0, 0, 0), alpha)
    ctx.set_line_width(5)
    ctx.stroke()
    f = comic.face(comic.FONT_LETTER)
    size = 150.0
    # stamp pops on raising 1, 50 and 100
    pop = 1.0
    for kk in (1, 50, 100):
        dt = T - TT.raise_time(kk)
        if 0 <= dt < 0.35:
            pop = 1.0 + 0.35 * math.exp(-dt * 14) * math.cos(dt * 30)
    label = "TIME!" if n >= 100 else "TIME"
    lsize = 62.0
    lw = f.width(label, lsize)
    digits = str(n)
    dw = f.width("0", size)
    total = dw * len(digits) + 24 + lw
    cx = (x0 + x1) / 2 - total / 2
    base = (y0 + y1) / 2 + size * 0.36
    ctx.rectangle(x0 + 6, y0 + 6, x1 - x0 - 12, y1 - y0 - 12)
    ctx.clip()
    ctx.translate((x0 + x1) / 2, (y0 + y1) / 2)
    ctx.scale(pop, pop)
    ctx.translate(-(x0 + x1) / 2, -(y0 + y1) / 2)
    # each digit rolls up from the previous value right after its raising
    for i, ch in enumerate(digits):
        p = len(digits) - 1 - i
        unit = 10 ** p
        # the moment this digit last changed
        k_last = (n // unit) * unit
        tl = T - TT.raise_time(max(1, k_last))
        roll = 1.0 if n >= 100 else clamp(tl / 0.05)
        xx = cx + i * dw
        prev = str(((n // unit) - 1) % 10) if n >= unit else " "
        for s_, dy in ((ch, (1 - roll) * size * 1.1), (prev, -roll * size * 1.1)):
            if s_.strip() == "" or (dy < 0 and roll >= 1):
                continue
            set_color(ctx, (0, 0, 0), alpha)
            f.path(ctx, s_, xx + (dw - f.width(s_, size)) / 2, base + dy, size)
            ctx.fill()
    set_color(ctx, (0, 0, 0), alpha)
    f.path(ctx, label, cx + dw * len(digits) + 24, base, lsize)
    ctx.fill()
    ctx.restore()


# ---------------------------------------------------------------- full-frame cuts
def _cut(fc, st, T, idx, t0, style):
    k = max(1, int(TT.count(t0 + 1e-4) + 1e-6)) if style != "soviet" else 1
    k = CUT_K.get(style, k)
    img = np.empty((fc.h, fc.w, 3), np.float32)
    t = T - t0
    nxt = TT.CUTS[idx + 1][0] if idx + 1 < len(TT.CUTS) else t0 + 0.3
    u = clamp(t / max(nxt - t0, 1e-3))
    # a slow push on every cut (Ken Burns), a faster one on the opener
    if style == "soviet":
        z = 1.0 + 0.07 * ease_in_out(clamp((T - fc.S.start) / (TT.R1 - fc.S.start))) + 0.05 * ease_out(clamp(t / 0.4))
        fx, fy = 900.0, 520.0
    else:
        z = 1.02 + 0.035 * u
        fx, fy = 1150.0, 420.0
    rect = ((0 - fx) * z + fx, (0 - fy) * z + fy, (1920 - fx) * z + fx, (1080 - fy) * z + fy)
    v = _v(k, T, t + 0.05)
    if style == "soviet":
        v["t"] = T - TT.R1
        tr = st["sov_track"]
        i = (T - fc.S.start) * fc.fps
        if i <= tr.n - 1:
            v["track"], v["track_i"] = tr, i
        v["mouth"] = 0.0
    render_cell(img, fc.s, rect, STYLES[style], v)
    return img


# ---------------------------------------------------------------- the page
def _page_img(fc, T, cam=None, n_cells=50):
    """the tract page of 50 or 100 panels. cam = (fx, fy, z) zoom about a fixed page point (the dive)"""
    img = np.empty((fc.h, fc.w, 3), np.float32)
    img[:] = PAGE_DARK

    def tr(r):
        if cam is None:
            return r
        fx, fy, z = cam
        return ((r[0] - fx) * z + fx, (r[1] - fy) * z + fy, (r[2] - fx) * z + fx, (r[3] - fy) * z + fy)

    cells = []
    if n_cells == 50:
        t_page = TT.R50
        for j, r in enumerate(DOUBLE):
            k = j + 1
            d = math.hypot((r[0] + r[2]) / 2 - 1500, (r[1] + r[3]) / 2 - 950)
            cells.append((k, r, t_page + d / 11000.0))
    else:
        t_page = [t0 for t0, s in TT.CUTS if s == "PAGE100"][0]
        for k in range(1, 101):
            r = CELL[k]
            tk = TT.raise_time(k)
            if tk <= t_page:
                d = math.hypot((r[0] + r[2]) / 2 - 1040, (r[1] + r[3]) / 2 - 1010)
                tk = t_page + d / 14000.0
            cells.append((k, r, tk))
    paper = np.array(ink.PAPER, np.float32)
    for k, r, tk in cells:
        rr = tr(r)
        if rr[2] < 0 or rr[0] > 1920 or rr[3] < 0 or rr[1] > 1080:
            continue
        if T < tk:
            x0, y0 = max(0, int(rr[0] * fc.s)), max(0, int(rr[1] * fc.s))
            x1, y1 = min(fc.w, int(math.ceil(rr[2] * fc.s))), min(fc.h, int(math.ceil(rr[3] * fc.s)))
            img[y0:y1, x0:x1] = paper
            continue
        sty = STYLE_OF[k]
        if sty == "dore":
            WK.render_cell100(img, fc, rr, T, T - tk)
            continue
        v = _v(k, T, 1.0, still=True, pop_t=T - tk)
        ck = (k, n_cells) if cam is None else None
        render_cell(img, fc.s, rr, STYLES[sty], v, cache_key=ck)
    return img, tr


def _page_top(fc, T, tr, n_cells):
    """keylines, the caption panel and the counter panel (in page coordinates through tr)"""
    top = Canvas(fc)
    ctx = top.ctx
    rects = DOUBLE if n_cells == 50 else [CELL[k] for k in range(1, 101)]
    ctx.set_line_width(3.0 * (1 if tr is None else 1))
    for r in rects:
        a = tr(r)
        set_color(ctx, (0.05, 0.05, 0.05))
        ctx.rectangle(a[0] - 1.5, a[1] - 1.5, a[2] - a[0] + 3, a[3] - a[1] + 3)
        ctx.stroke()
    return top


def _lettering(fc, T, cam=None):
    top = Canvas(fc)
    ctx = top.ctx
    ctx.save()
    if cam is not None:
        fx, fy, z = cam
        ctx.translate(fx, fy)
        ctx.scale(z, z)
        ctx.translate(-fx, -fy)
    if T <= fc.tl.line("P21")["end"] + 0.5:
        _caption(ctx, fc)
    elif cam is not None:
        # the page keeps its printed caption while we dive
        comic.say(ctx, fc, "P21", CAP[0], CAP[1], w=CAP[2] - CAP[0], size=CAP_SIZE, anchor="tl",
                  T=fc.tl.line("P21")["end"], bounds=False)
    _counter(ctx, T)
    ctx.restore()
    return top


# ---------------------------------------------------------------- render
def render(fc, st):
    T = fc.T
    if T >= DIVE1:
        return WK.render(fc, st["walker"])
    cut = TT.cut_at(T)
    if cut is None:
        cut = (0, TT.CUTS[0][0], "soviet")
    if T < TT.R1:
        cut = (0, TT.CUTS[0][0], "soviet")
    idx, t0, style = cut
    cam = None
    if style == "PAGE50":
        img, tr = _page_img(fc, T, None, 50)
        top = _page_top(fc, T, tr, 50)
        img = top.over(img)
    elif style == "PAGE100":
        if T >= DIVE0:
            u = ease_in_out(clamp((T - DIVE0) / (DIVE1 - DIVE0)), 3)
            cam = _dive_cam(u)
        img, tr = _page_img(fc, T, cam, 100)
        top = _page_top(fc, T, tr, 100)
        img = top.over(img)
    else:
        img = _cut(fc, st, T, idx, t0, style)
    let = _lettering(fc, T, cam)
    return let.over(img)


def _dive_cam(u):
    """zoom about the fixed page point that carries panel #100 onto the full frame"""
    x0, y0, x1, y1 = CELL[100]
    z1 = max(1920.0 / (x1 - x0), 1080.0 / (y1 - y0))
    ccx, ccy = (x0 + x1) / 2, (y0 + y1) / 2
    # fixed point F with (C - F) * z1 + F = screen centre
    fx = (960.0 - ccx * z1) / (1 - z1)
    fy = (540.0 - ccy * z1) / (1 - z1)
    z = math.exp(math.log(z1) * u)
    return (fx, fy, z)

"""THE LETTERING DEPARTMENT — balloons, captions, footnotes, SFX lettering and panel furniture for
THE FLAG GAME? (film 2).  Owner: Letterer.
(STUB v0 by Main: working, plain; Letterer replaces the internals and keeps every public signature.)

The film has NO burned-in subtitles: every spoken line is lettered on screen, Chick-tract-parody style
(heavy sans caps + emphasis CAPS exactly as written in the display text), and the lettering is SYNCED TO
THE VOICES — each word appears the moment it is spoken (word timestamps from the timeline).
Lettering is black ink on white paper; it must NEVER overlap a Flag's cloth.

API
---
say(ctx, fc, line_id, x, y, *, w=None, tail=None, kind=None, size=None, alpha=1.0, hold=0.4, pre=0.1,
    avoid=None) -> bbox | None
    Letter timeline line `line_id` at fc.T. kind defaults to the line's meta["letter"]:
    balloon | shout | caption | radio | offpanel | footnote | title. (x, y) is the balloon centre
    (captions/footnotes: top-left corner) in design units; tail = (tx, ty) point the tail aims at.
    Visible from start - pre to end + hold; returns (x0, y0, x1, y1) or None when not visible.
balloon(ctx, text, x, y, w=None, *, tail=None, kind="balloon", reveal=None, size=38, alpha=1.0) -> bbox
    reveal = number of display tokens shown (float fades the next one in); None shows everything.
caption(ctx, text, x, y, w, *, reveal=None, size=34, alpha=1.0) -> bbox
footnote(ctx, text, x, y, *, size=20, progress=1.0, alpha=1.0) -> bbox
sfx(ctx, text, x, y, size, *, rot=0.0, t=0.0, alpha=1.0) -> bbox          onomatopoeia
burst_bg(ctx, cx, cy, rect, *, t=0.0, rays=48)                           radial action lines
emanata(ctx, x, y, r, kind="shock", *, t=0.0)                            shock | sweat | stink | sparkle
panel(ctx, rect, *, border=6.0)                                          white panel, black frame
flags_flood(ctx, rect, amount, *, t=0.0, size=34)                        'FLAGS FLAGS FLAGS' text walls
reveal_count(fc, line_id, lead=0.0) -> float                             tokens spoken by fc.T
FONT_LETTER, FONT_LETTER_ALT                                              lettering faces
"""
import math
import re

import cairo

from .canvas import set_color, rrect, ellipse, poly
from .config import FONT_SANS

FONT_LETTER = FONT_SANS          # Letterer installs the real lettering faces
FONT_LETTER_ALT = FONT_SANS
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")


def _tokens(text):
    """display tokens (whitespace split) and the index of the first spoken word inside each"""
    toks, starts, k = text.split(), [], 0
    for tk in toks:
        starts.append(k)
        k += len(_WORD.findall(tk))
    return toks, starts


def reveal_count(fc, line_id, lead=0.0):
    l = fc.tl.line(line_id)
    toks, starts = _tokens(l["text"])
    ws = fc.tl.words(line_id)
    n = 0.0
    for i, st in enumerate(starts):
        if st >= len(ws):
            t0 = ws[-1]["e"] if ws else l["start"]
        else:
            t0 = ws[st]["s"]
        if fc.T + lead >= t0:
            n = i + min(1.0, (fc.T + lead - t0) / 0.06)
    return n


def _face(ctx, size, bold=True, italic=False):
    ctx.select_font_face(FONT_LETTER, cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)


def _wrap(ctx, toks, size, maxw):
    _face(ctx, size)
    space = ctx.text_extents(" ").x_advance
    lines, cur, curw = [], [], 0.0
    for i, tk in enumerate(toks):
        tw = ctx.text_extents(tk).x_advance
        if cur and curw + space + tw > maxw:
            lines.append(cur)
            cur, curw = [], 0.0
        cur.append((i, tk, tw))
        curw += (space if len(cur) > 1 else 0) + tw
    if cur:
        lines.append(cur)
    return lines, space


def _shape(ctx, kind, x0, y0, x1, y1, tail):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    ctx.new_path()
    if kind in ("caption", "footnote", "title"):
        ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    elif kind == "radio":
        n = 26
        pts = []
        for k in range(n * 2):
            a = 2 * math.pi * k / (n * 2)
            r = 1.0 if k % 2 == 0 else 0.9
            pts.append((cx + math.cos(a) * rx * 1.08 * r, cy + math.sin(a) * ry * 1.12 * r))
        poly(ctx, pts)
    elif kind == "shout":
        n = 18
        pts = []
        for k in range(n * 2):
            a = 2 * math.pi * k / (n * 2)
            r = 1.18 if k % 2 == 0 else 0.96
            pts.append((cx + math.cos(a) * rx * 1.05 * r, cy + math.sin(a) * ry * 1.1 * r))
        poly(ctx, pts)
    else:
        rrect(ctx, x0 - rx * 0.12, y0 - ry * 0.18, (x1 - x0) + rx * 0.24, (y1 - y0) + ry * 0.36,
              min(rx, ry) * 1.1)
    if tail is not None and kind not in ("caption", "footnote", "title"):
        tx, ty = tail
        ang = math.atan2(ty - cy, tx - cx)
        bw = min(rx, ry) * 0.35
        ctx.move_to(cx + math.cos(ang + 1.5) * bw, cy + math.sin(ang + 1.5) * bw)
        ctx.line_to(tx, ty)
        ctx.line_to(cx + math.cos(ang - 1.5) * bw, cy + math.sin(ang - 1.5) * bw)
        ctx.close_path()


def balloon(ctx, text, x, y, w=None, *, tail=None, kind="balloon", reveal=None, size=38, alpha=1.0):
    toks, _ = _tokens(text)
    maxw = w or min(760.0, max(260.0, 18.0 * size * math.sqrt(max(1, len(text)) / 40.0)))
    lines, space = _wrap(ctx, toks, size, maxw)
    lh = size * 1.18
    tw = max(sum(t[2] for t in ln) + space * (len(ln) - 1) for ln in lines)
    th = lh * len(lines)
    pad = size * (0.55 if kind in ("caption", "footnote", "title") else 0.45)
    if kind in ("caption", "footnote"):
        x0, y0 = x, y
    else:
        x0, y0 = x - tw / 2 - pad, y - th / 2 - pad
    x1, y1 = x0 + tw + pad * 2, y0 + th + pad * 2
    ctx.save()
    _shape(ctx, kind, x0, y0, x1, y1, tail)
    set_color(ctx, (1, 1, 1), alpha)
    ctx.fill_preserve()
    set_color(ctx, (0, 0, 0), alpha)
    ctx.set_line_width(max(2.0, size * 0.09))
    ctx.stroke()
    _face(ctx, size)
    fe = ctx.font_extents()
    yy = y0 + pad + fe[0] * 0.92
    for ln in lines:
        lw = sum(t[2] for t in ln) + space * (len(ln) - 1)
        xx = (x0 + x1) / 2 - lw / 2 if kind != "caption" else x0 + pad
        for i, tk, tkw in ln:
            a = 1.0 if reveal is None else max(0.0, min(1.0, reveal - i))
            if a > 0:
                set_color(ctx, (0, 0, 0), alpha * a)
                ctx.move_to(xx, yy)
                ctx.show_text(tk)
            xx += tkw + space
        yy += lh
    ctx.restore()
    return (x0, y0, x1, y1)


def caption(ctx, text, x, y, w, *, reveal=None, size=34, alpha=1.0):
    return balloon(ctx, text, x, y, w, kind="caption", reveal=reveal, size=size, alpha=alpha)


def footnote(ctx, text, x, y, *, size=20, progress=1.0, alpha=1.0):
    n = int(round(len(text) * max(0.0, min(1.0, progress))))
    return balloon(ctx, text[:n] or " ", x, y, None, kind="footnote", size=size, alpha=alpha)


def say(ctx, fc, line_id, x, y, *, w=None, tail=None, kind=None, size=None, alpha=1.0, hold=0.4, pre=0.1,
        avoid=None):
    l = fc.tl.line(line_id)
    if not (l["start"] - pre <= fc.T <= l["end"] + hold):
        return None
    kind = kind or l.get("meta", {}).get("letter", "balloon")
    if kind == "offpanel":
        kind = "balloon"
    size = size or {"caption": 34, "footnote": 22, "title": 64, "shout": 46}.get(kind, 38)
    fade = min(1.0, (fc.T - (l["start"] - pre)) / 0.08, (l["end"] + hold - fc.T) / 0.12)
    rv = reveal_count(fc, line_id, lead=0.04)
    return balloon(ctx, l["text"], x, y, w, tail=tail, kind=kind, reveal=rv, size=size, alpha=alpha * fade)


def sfx(ctx, text, x, y, size, *, rot=0.0, t=0.0, alpha=1.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    _face(ctx, size)
    ext = ctx.text_extents(text)
    ctx.move_to(-ext.x_advance / 2, size * 0.35)
    ctx.text_path(text)
    set_color(ctx, (1, 1, 1), alpha)
    ctx.set_line_width(size * 0.16)
    ctx.stroke_preserve()
    set_color(ctx, (0, 0, 0), alpha)
    ctx.fill()
    ctx.restore()
    return (x - ext.x_advance / 2, y - size / 2, x + ext.x_advance / 2, y + size / 2)


def burst_bg(ctx, cx, cy, rect, *, t=0.0, rays=48):
    x0, y0, w, h = rect
    R = math.hypot(w, h)
    ctx.save()
    ctx.rectangle(x0, y0, w, h)
    ctx.clip()
    set_color(ctx, (1, 1, 1))
    ctx.paint()
    set_color(ctx, (0, 0, 0))
    for k in range(rays):
        a0 = 2 * math.pi * k / rays + t * 0.1
        a1 = a0 + math.pi / rays * 0.55
        ctx.move_to(cx, cy)
        ctx.line_to(cx + math.cos(a0) * R, cy + math.sin(a0) * R)
        ctx.line_to(cx + math.cos(a1) * R, cy + math.sin(a1) * R)
        ctx.close_path()
    ctx.fill()
    ctx.restore()


def emanata(ctx, x, y, r, kind="shock", *, t=0.0):
    ctx.save()
    set_color(ctx, (0, 0, 0))
    ctx.set_line_width(max(1.5, r * 0.06))
    if kind == "sweat":
        ellipse(ctx, x, y, r * 0.18, r * 0.28)
        set_color(ctx, (1, 1, 1))
        ctx.fill_preserve()
        set_color(ctx, (0, 0, 0))
        ctx.stroke()
    elif kind == "stink":
        for k in range(3):
            xx = x + (k - 1) * r * 0.35
            ctx.move_to(xx, y)
            for j in range(1, 9):
                ctx.line_to(xx + math.sin(j * 1.3 + t * 4 + k) * r * 0.08, y - j * r * 0.12)
            ctx.stroke()
    else:
        for k in range(7):
            a = -math.pi / 2 + (k - 3) * 0.33
            ctx.move_to(x + math.cos(a) * r, y + math.sin(a) * r)
            ctx.line_to(x + math.cos(a) * r * 1.35, y + math.sin(a) * r * 1.35)
        ctx.stroke()
    ctx.restore()


def panel(ctx, rect, *, border=6.0):
    x0, y0, w, h = rect
    ctx.save()
    ctx.rectangle(x0, y0, w, h)
    set_color(ctx, (1, 1, 1))
    ctx.fill_preserve()
    set_color(ctx, (0, 0, 0))
    ctx.set_line_width(border)
    ctx.stroke()
    ctx.restore()


def flags_flood(ctx, rect, amount, *, t=0.0, size=34):
    x0, y0, w, h = rect
    _face(ctx, size)
    word = "FLAGS "
    ww = ctx.text_extents(word).x_advance
    rows = int(h / (size * 1.1)) + 1
    per = int(w / ww) + 1
    total = rows * per
    n = int(total * max(0.0, min(1.0, amount)))
    ctx.save()
    ctx.rectangle(x0, y0, w, h)
    ctx.clip()
    set_color(ctx, (0, 0, 0))
    for k in range(n):
        r, c = divmod(k, per)
        ctx.move_to(x0 + c * ww, y0 + (r + 1) * size * 1.1)
        ctx.show_text("FLAGS")
    ctx.restore()

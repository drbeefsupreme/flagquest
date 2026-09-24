"""THE LETTERING DEPARTMENT — balloons, captions, footnotes, SFX lettering and panel furniture for
THE FLAG GAME? (film 2).  Owner: Letterer.

The film has NO burned-in subtitles: every spoken line is lettered on screen, Chick-tract-parody style
(the tract's Arial-Black look: Archivo Black; heavy condensed for the 'HOWEVER...' caption; CAPS emphasis,
typos and punctuation exactly as written in the display text), and the lettering is SYNCED TO THE VOICES:
each word is inked the moment it is spoken (timeline word timestamps; every spoken word carries `d`, its
display form), CAPS emphasis words punch (1.0 -> 1.12 -> 1.0 over 0.12 s) as they are spoken, balloons pop
in with a slight overshoot just before the first word and fade after end + hold; captions reveal phrase by
phrase; footnotes type on letter by letter. Lettering is black ink on paper white; it must NEVER overlap a
Flag's cloth: pass the cloth boxes/polygons as `avoid` (the balloon is nudged clear and anything that still
overlaps is clipped away), or knock the finished lettering layer out with protect().
All coordinates are design units (1920x1080); every line weight scales with the lettering size.

API
---
say(ctx, fc, line_id, x, y, *, w=None, tail=None, kind=None, style=None, size=None, font=None, alpha=1.0,
    hold=0.5, pre=0.12, avoid=None, bounds=None, text=None, tokens=None, mark="auto", anchor=None,
    align=None, italic=None, T=None, flagify=0.0, flag_word="FLAGS") -> bbox | None
    Letter timeline line `line_id` at fc.T (or T). kind defaults to the line's meta["letter"]:
      balloon | shout | caption | radio | offpanel | footnote | title
    style defaults to meta["style"]: None | divine (Flag Maker: radiant double border) | devil (ragged,
    dripping) | anime (wobbly, condensed).
    (x, y): balloon/title CENTRE; caption/footnote TOP-LEFT corner (override with anchor="c"|"tl"|"tr"|
    "bl"|"br"|"t"|"b"|"l"|"r"). w = max text width (captions: box width). tail = (tx, ty) the tail tip
    (the speaker's mouth; offpanel: a point on/over the frame edge — the tail runs off the frame).
    avoid = [rects (x0,y0,x1,y1) | polygons [(x,y),...] | shapely geoms] to stay clear of (flag cloth), or
    a CALLABLE T -> [shapes]: the balloon is then placed against the union over its whole lifetime (never
    jumps as a flag moves) and clipped against the current frame; e.g.
    avoid=lambda T: comic.track_avoid(track, (T - t0) * 24).
    bounds = rect (x0,y0,x1,y1) the balloon body must stay inside (default: the frame; False = none).
    Lettering under a camera/page transform (cam.apply(ctx)): coordinates are then page units — pass
    bounds=panel_rect (or False) since the default frame box no longer means the screen.
    text = override of the lettered text (e.g. with footnote marks "...WAS WRONG!!*"); words are matched
    to the voice by their letters, so extra marks are fine. tokens = (i0, i1) letters only display tokens
    [i0:i1] (e.g. P12 caption = tokens=(0, 9), its citation 'FLAGS 4:20' = kind="footnote",
    tokens=(9, None)). mark="auto" appends the line's footnote mark (P02 -> "WRONG!!*"); "" disables.
    kind="footnote" letters meta["footnote"] when present (D01 -> '** “Survey FLAGS” is strictly ...').
    italic = (i0, i1) token range set in FONT_LETTER_ALT. flagify = 0..1: that share of the words turns
    into FLAGS (seeded order; each flickers, then stamps in squeezed into its slot) — t04's flood.
    Visible from first word - pre to end + hold (+0.18 s fade). Returns the full-size (x0,y0,x1,y1).
    Per-line tract styling defaults live in SPEC (P21 condensed, P04 italic, S04 anime, V02 Anton...).
measure(ctx, fc, line_id, x, y, **same kwargs as say) -> bbox      layout only (after avoid), no drawing
measure_text(text, x, y, *, w, tail, kind, style, size, font, anchor, avoid, bounds) -> bbox
say_footnote(ctx, fc, line_id, x, y, *, t0=None, anchor="br", size=24, hold=None, T=None) -> bbox|None
    the line's meta["footnote"] as a footnote box sliding in at t0 (default: the last spoken word).
balloon(ctx, text, x, y, w=None, *, tail=None, kind="balloon", style=None, reveal=None, size=None,
        alpha=1.0, font=None, t=0.0, avoid=None, bounds=None, anchor=None, italic=None, flagify=0.0) -> bbox
    Static/manual balloon (crowd balloons...). reveal = number of display tokens shown (float fades the
    next one in); t drives animated styles (devil drips, divine glory).
caption(ctx, text, x, y, w, *, reveal=None, size=34, alpha=1.0, font=None, align="center") -> bbox
footnote(ctx, text, x, y, *, size=20, progress=1.0, alpha=1.0, anchor="tl", font=None) -> bbox
sfx(ctx, text, x, y, size, *, rot=0.0, t=None, alpha=1.0, depth=0.0, font=FONT_SFX, avoid=None) -> bbox
    onomatopoeia: black letters, white + black double outline, per-letter jitter; t = seconds since the
    hit (None = static): pops in letter by letter with overshoot and a shake. depth > 0: 3-D extrusion.
burst_bg(ctx, cx, cy, rect, *, t=0.0, rays=48, inner=None, boil=12)   Chick radial burst, boils on twos
speed_lines(ctx, rect, *, focus=None, angle=0.0, t=0.0, n=90, inner=None)
    focus=(x,y): manga focus lines converging on focus; else parallel lines at `angle` streaming with t
emanata(ctx, x, y, r, kind="shock", *, t=0.0, ang=-pi/2, seed=0)
    shock | sweat | stink | sparkle | tears | blush | anger | question | exclaim | dizzy
panel(ctx, rect, *, border=6.0, fill=True)                               white panel, black frame
Page(x, y, h, *, n=3, number=None) -> page.rect, page.panels, page.draw(ctx)   portrait tract page
frame(rect, *, pad=0.0) -> Camera ; glide(rect_a, rect_b, u, *, pad=0.0, lift=0.12) -> Camera
    camera that frames a panel (vx.camera.Camera; use cam.apply(ctx)); glide = eased panel-to-panel move
flags_flood(ctx, rect, amount, *, t=0.0, size=34, avoid=None, pulse=0.0, seed=0) -> None
    the kinetic 'FLAGS FLAGS FLAGS' wall (page 25): 0 empty -> rows drop in and stack -> overprinted
    second layer -> 1 = solid wall. Words never touch `avoid` (they reflow around the cloth).
protect(letter_rgba, flag_rgba) -> rgba      knock lettering out wherever flag cloth is (absolute guard):
    img = over(img, comic.protect(top.rgba(), flags.rgba()))
cloth_avoid(flag_rgba, s=fc.s, *, pole=16, grow=6) -> [polygons]   flag layer pixels -> cloth hulls + pole
    strips for avoid= (works for any flag drawing)
track_avoid(track, frames, *, pole=True, grow=6) -> [shapes]   FlagTrack sim -> cloth hull (+pole) over
    frames, no rendering; the natural body of a callable avoid
face(family) -> kerned face: .path(ctx, text, x_left, baseline, size) (then fill), .width(text, size)
reveal_count(fc, line_id, lead=0.0) -> float                             tokens spoken by fc.T
FONT_LETTER (Archivo Black), FONT_CAPTION, FONT_TITLE (Archivo Expanded 900), FONT_CONDENSED (Archivo
Condensed 800: HOWEVER/anime), FONT_IMPACT (Anton), FONT_SFX (Bangers), FONT_LETTER_ALT (Archivo Black
Italic), FONT_BODY (Archivo Bold). Install/rebuild: `python -m vx.comic_fonts` (OFL; assets/fonts/).
"""
import math
import re
import zlib
from functools import lru_cache

import cairo
import numpy as np

from .config import W, H, ROOT
from .camera import Camera

FONT_LETTER = "Archivo Black"
FONT_CAPTION = "Archivo Black"
FONT_TITLE = "VX Archivo Expanded"
FONT_CONDENSED = "VX Archivo Condensed"
FONT_IMPACT = "Anton"
FONT_SFX = "Bangers"
FONT_LETTER_ALT = "VX Archivo Italic"
FONT_BODY = "VX Archivo Bold"
FONT_OSWALD = "VX Oswald"
_FDIR = ROOT / "assets" / "fonts"
_FONT_FILES = {
    "Archivo Black": _FDIR / "ArchivoBlack-Regular.ttf",
    "VX Archivo Expanded": _FDIR / "VXArchivoExpanded.ttf",
    "VX Archivo Condensed": _FDIR / "VXArchivoCondensed.ttf",
    "VX Archivo Italic": _FDIR / "VXArchivoItalic.ttf",
    "VX Archivo Bold": _FDIR / "VXArchivoBold.ttf",
    "VX Oswald": _FDIR / "VXOswald.ttf",
    "Anton": _FDIR / "Anton-Regular.ttf",
    "Bangers": _FDIR / "Bangers-Regular.ttf",
}
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’]*")
_BOXED = ("caption", "footnote", "title")
DEFAULT_SIZE = {"balloon": 44, "offpanel": 44, "shout": 50, "radio": 42, "caption": 40, "footnote": 26,
                "title": 110}
# the letterer's script: per-line defaults straight from the tract scans (kwargs to say() override)
SPEC = {
    "P04": dict(italic=(3, None)),                        # 'THE LIVING ESSENCE OF THE BURN ITSELF!!!'
    "S04": dict(font=FONT_CONDENSED, size=60),            # the anime balloon (page 24)
    "P21": dict(font=FONT_CONDENSED, size=62),            # 'HOWEVER...' tall condensed caption (page 26)
    "P13": dict(font=FONT_TITLE),
    "X01": dict(italic=(0, None)),                         # 'Shut up!' (page 24)
    "V02": dict(font=FONT_IMPACT, size=70),
    "D01": dict(font=FONT_LETTER),
}

_FO = cairo.FontOptions()
_FO.set_hint_style(cairo.HINT_STYLE_NONE)
_FO.set_hint_metrics(cairo.HINT_METRICS_OFF)
_FO.set_antialias(cairo.ANTIALIAS_GRAY)


def _paper():
    from . import ink
    return ink.PAPER


def _inkc():
    from . import ink
    return ink.INK


def _clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def _smooth(x):
    x = _clamp(x)
    return x * x * (3 - 2 * x)


def _out_back(t, s=1.9):
    t = _clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def _h01(*k):
    return (zlib.crc32(repr(k).encode()) & 0xFFFFFF) / float(0x1000000)


# ================================================================== fonts / shaping
class _Face:
    """one lettering face: harfbuzz shaping (kerning) + cairo glyph outlines of the same font file"""
    _all = {}

    def __new__(cls, family):
        f = cls._all.get(family)
        if f is None:
            f = super().__new__(cls)
            f._init(family)
            cls._all[family] = f
        return f

    def _init(self, family):
        self.family = family
        self.toy = cairo.ToyFontFace(family, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        self.cache = {}
        self.hb = None
        path = _FONT_FILES.get(family)
        if path is not None and path.exists():
            try:
                import uharfbuzz as hb
                face = hb.Face(hb.Blob.from_file_path(str(path)))
                font = hb.Font(face)
                self.hb, self._hbmod, self.upem = font, hb, face.upem
                gid = font.get_nominal_glyph(ord("H"))
                ext = font.get_glyph_extents(gid)
                self.cap = ext.y_bearing / self.upem
                fe = font.get_font_extents("ltr")
                self.asc, self.desc = fe.ascender / self.upem, -fe.descender / self.upem
                # cairo must resolve the family to the same file (glyph ids are shared)
                gids, _, _ = self._shape_hb("FLAGS")
                if tuple(g.index for g in self._sf().text_to_glyphs(0, 0, "FLAGS", False)) != gids:
                    self.hb = None
            except Exception:
                self.hb = None
        if self.hb is None:
            sf = self._sf()
            e = sf.text_extents("H")
            self.cap = -e.y_bearing / 1000.0
            fe = sf.extents()
            self.asc, self.desc = fe[0] / 1000.0, fe[1] / 1000.0

    def _sf(self):
        return cairo.ScaledFont(self.toy, cairo.Matrix(1000, 0, 0, 1000), cairo.Matrix(), _FO)

    def _shape_hb(self, s):
        hb = self._hbmod
        buf = hb.Buffer()
        buf.add_str(s)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf, {"kern": True, "liga": False})
        x, gids, xs = 0, [], []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            gids.append(info.codepoint)
            xs.append((x + pos.x_offset) / self.upem)
            x += pos.x_advance
        return tuple(gids), tuple(xs), x / self.upem

    def shape(self, s):
        """-> (glyph ids, x offsets in em, advance in em)"""
        r = self.cache.get(s)
        if r is None:
            if self.hb is not None:
                r = self._shape_hb(s)
            else:
                gl = self._sf().text_to_glyphs(0, 0, s, False)
                ext = self._sf().text_extents(s)
                r = (tuple(g.index for g in gl), tuple(g.x / 1000.0 for g in gl), ext.x_advance / 1000.0)
            self.cache[s] = r
        return r

    def width(self, s, size):
        return self.shape(s)[2] * size

    def path(self, ctx, s, x, y, size):
        """append the outline of s (baseline-left at x, y) to ctx's path"""
        gids, xs, _ = self.shape(s)
        ctx.set_font_face(self.toy)
        ctx.set_font_size(size)
        ctx.set_font_options(_FO)
        ctx.glyph_path([cairo.Glyph(g, x + gx * size, y) for g, gx in zip(gids, xs)])


def face(family=None):
    return _Face(family or FONT_LETTER)


# ================================================================== words <-> voice
def _cores(tok):
    return _WORD.findall(tok)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _align(tokens, words, t0, t1):
    """per display token: (t_on, t_end) from the spoken words (matched by their display form `d`)"""
    spoken = [w for w in words if w.get("d") or any(c.isalnum() for c in w.get("w", ""))]
    out, k = [], 0
    for tok in tokens:
        cs = _cores(tok)
        if not cs:
            out.append(None)
            continue
        hit = []
        for c in cs:
            j = k
            while j < len(spoken) and j < k + 4 and _norm(spoken[j].get("d") or spoken[j]["w"]) != _norm(c):
                j += 1
            if j < len(spoken) and j < k + 4:
                k = j
            if k < len(spoken):
                hit.append(spoken[k])
                k += 1
        out.append((hit[0]["s"], hit[-1]["e"]) if hit else None)
    # tokens without a voice match: attach to the previous token (trailing marks) or the next (leading)
    n = len(out)
    for i in range(n):
        if out[i] is None:
            prev = next((out[j] for j in range(i - 1, -1, -1) if out[j] is not None), None)
            nxt = next((out[j] for j in range(i + 1, n) if out[j] is not None), None)
            if prev is not None and (i > 0 and _cores(tokens[i - 1]) or nxt is None):
                out[i] = prev
            elif nxt is not None:
                out[i] = nxt
            else:
                f = i / max(1, n)
                out[i] = (t0 + (t1 - t0) * f, t0 + (t1 - t0) * (i + 1) / max(1, n))
    for i in range(n):          # fill gaps of unmatched leading tokens
        if out[i] is None:
            out[i] = (t0, t1)
    return out


def _is_caps(tok):
    letters = [c for c in tok if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters) and _norm(tok) not in ("ii", "ix", "iv", "vi")


def _phrase_ids(tokens, times, maxlen=4):
    """caption phrases: split at punctuation and pauses, then long runs at their biggest breath gaps"""
    n = len(tokens)
    gaps = [times[i + 1][0] - times[i][1] if i + 1 < n else 0.0 for i in range(n)]
    cuts = [i for i in range(n - 1) if re.search(r"[,.;:!?…]+[\"'”’*]*$", tokens[i]) or gaps[i] > 0.25]
    segs, a = [], 0
    for c in cuts + [n - 1]:
        segs.append((a, c + 1))
        a = c + 1
    out = []
    while segs:
        a, b = segs.pop(0)
        if b - a > maxlen:
            # split at the largest gap (ties -> nearest the middle), both halves >= 2 tokens
            k = max(range(a + 1, b - 2), key=lambda i: (round(gaps[i], 2), -abs(i + 1 - (a + b) / 2)))
            segs[:0] = [(a, k + 1), (k + 1, b)]
        else:
            out.append((a, b))
    ids = [0] * n
    for p, (a, b) in enumerate(out):
        for i in range(a, b):
            ids[i] = p
    return ids


def reveal_count(fc, line_id, lead=0.0):
    l = fc.tl.line(line_id)
    toks = l["text"].split()
    tm = _align(toks, l["words"], l["start"], l["end"])
    n = 0.0
    for i, (a, _) in enumerate(tm):
        if fc.T + lead >= a:
            n = i + min(1.0, (fc.T + lead - a) / 0.06)
    return n


# ================================================================== layout
class _Layout:
    """text set into lines, centred on (0, 0); body semi-axes rx, ry (balloons) or box half size"""
    __slots__ = ("toks", "faces", "size", "lines", "lh", "cap", "rx", "ry", "hw", "hh", "p", "kind", "space")


def _prefix(ws):
    c = [0.0]
    for v in ws:
        c.append(c[-1] + v)
    return c


def _lw(cum, sp, j, i):
    return cum[i] - cum[j] + sp * (i - j - 1)


def _dp(N, n, cost, combine, nob=None):
    """split tokens 0..N into n non-empty lines. cost(k, j, i) for line k = tokens j..i.
    combine='max' (minimax) or 'sum'. nob[j]: no line may start at token j. returns (value, breaks)"""
    INF = float("inf")
    best = [[INF] * (N + 1) for _ in range(n + 1)]
    arg = [[0] * (N + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for k in range(1, n + 1):
        for i in range(k, N - (n - k) + 1):
            for j in range(k - 1, i):
                if j > 0 and nob is not None and nob[j]:
                    continue
                b = best[k - 1][j]
                if b == INF:
                    continue
                c = cost(k - 1, j, i)
                if c == INF:
                    continue
                v = max(b, c) if combine == "max" else b + c
                if v < best[k][i]:
                    best[k][i] = v
                    arg[k][i] = j
    if best[n][N] == INF:
        return INF, None
    br, i = [], N
    for k in range(n, 0, -1):
        j = arg[k][i]
        br.append((j, i))
        i = j
    return best[n][N], br[::-1]


@lru_cache(maxsize=512)
def _layout(toks, fonts, size, kind, maxw, lh_k, p):
    faces = [_Face(f) for f in fonts]
    ws = [fc.width(t, size) for fc, t in zip(faces, toks)]
    sp = faces[0].width(" ", size) * (0.92 if kind != "title" else 1.0)
    cum = _prefix(ws)
    N = len(toks)
    cap = faces[0].cap * size
    lh = size * lh_k
    L = _Layout()
    L.toks, L.faces, L.size, L.lh, L.cap, L.kind, L.p, L.space = toks, faces, size, lh, cap, kind, p, sp
    maxw = maxw or 1e9
    longest = max(ws)
    maxw = max(maxw, longest)

    fw = ("the", "a", "an", "of", "to", "and", "for", "in", "on", "at", "my", "is", "are", "that", "by", "its",
          "his", "her", "their", "with", "from", "as", "be", "i")
    bonus, malus = (0.9 * size) ** 2, (1.3 * size) ** 2
    roman = re.compile(r"^(I{1,3}|IV|V|VI{1,3}|IX|X)$")

    def brk(i):
        """cost of a line break before token i (phrase-aware, like a letterer)"""
        if i >= N:
            return 0.0
        prev, nxt = toks[i - 1], toks[i]
        c = 0.0
        if re.search(r"[,.;:!?…]+[\"'”’*]*$", prev) or nxt.startswith("..."):
            c -= bonus
        if prev.lower().strip("\"'“”") in fw:
            c += malus
        if i == N - 1 and len(_norm(nxt)) <= 3:
            c += malus * 2                     # no widowed short word on the last line
        return c

    # hard glue: never break inside a citation ('II VEXILLIANS 5', 'FLAGS 4:20', '50, then 100')
    nob = [False] * (N + 1)
    for i in range(1, N):
        nob[i] = bool(roman.match(_norm(toks[i - 1]).upper()) or re.match(r"^\d", _norm(toks[i])))

    def ys(n):
        # vertical ink extent of each line (cap top .. baseline + small descent), block centred
        tot = (n - 1) * lh + cap
        out = []
        for k in range(n):
            base = -tot / 2 + cap + k * lh
            out.append(max(abs(base - cap), abs(base + 0.16 * size)))
        return out

    if kind in _BOXED:
        # rectangle: minimal line count that fits maxw, then balanced (minimise the widest line, then
        # the sum of squared slack)
        best = None
        for n in range(1, N + 1):
            v, br = _dp(N, n, lambda k, j, i: _lw(cum, sp, j, i) if _lw(cum, sp, j, i) <= maxw + 1e-6
                        else float("inf"), "max", nob)
            if br is None:
                continue
            if maxw < 1e8 or kind == "footnote":
                best = (n, v)
                break
            # auto width: aim for a pleasant aspect (wide captions)
            aspect = (v + size) / ((n - 1) * lh + cap + size)
            score = abs(math.log(aspect / (4.2 if kind == "caption" else 5.0)))
            if best is None or score < best[2]:
                best = (n, v, score)
        n, v = best[0], best[1]
        lim = min(maxw, v * 1.06)
        _, br = _dp(N, n, lambda k, j, i: (v - _lw(cum, sp, j, i)) ** 2 + brk(i) if _lw(cum, sp, j, i) <= lim + 1e-6
                    else float("inf"), "sum", nob)
        L.lines = br
        L.hw = max(_lw(cum, sp, j, i) for j, i in br) / 2
        L.hh = ((n - 1) * lh + cap) / 2
        L.rx = L.ry = None
        return L

    # balloon: the text block must sit inside a superellipse |x/rx|^p + |y/ry|^p <= 1 (padded);
    # search line count and aspect for the smallest balloon, then fill it evenly (diamond-shaped lines)
    padx, pady = size * 0.62, size * 0.52
    best = None
    for n in range(1, min(N, 9) + 1):
        yk = ys(n)
        for a in (1.25, 1.5, 1.75, 2.0, 2.3, 2.65, 3.0, 3.5, 4.0):
            def cost(k, j, i, yk=yk, a=a):
                lw = _lw(cum, sp, j, i)
                if lw > maxw + 1e-6:
                    return float("inf")
                hx, hy = (lw / 2 + padx) / a, yk[k] + pady
                return (hx ** p + hy ** p) ** (1.0 / p)
            r, br = _dp(N, n, cost, "max", nob)
            if br is None:
                continue
            area = a * r * r * (1 + 0.6 * abs(math.log(a / 3.0)))
            if best is None or area < best[0]:
                best = (area, n, a, r, yk)
    _, n, a, r, yk = best

    def rk(k, j, i):
        hx, hy = (_lw(cum, sp, j, i) / 2 + padx) / a, yk[k] + pady
        return (hx ** p + hy ** p) ** (1.0 / p)

    def cost2(k, j, i):
        lw = _lw(cum, sp, j, i)
        if lw > maxw + 1e-6:
            return float("inf")
        hy = yk[k] + pady
        rr = rk(k, j, i)
        if rr > r * 1.06:
            return float("inf")
        room = max(0.0, r ** p - hy ** p) ** (1.0 / p) * a - padx
        return (room - lw / 2) ** 2 + brk(i) + (max(0.0, rr - r) * a * 6) ** 2

    v, br = _dp(N, n, cost2, "sum", nob)
    r = max(rk(k, j, i) for k, (j, i) in enumerate(br))
    L.lines = br
    L.rx, L.ry = a * r, r
    L.hw = max(_lw(cum, sp, j, i) for j, i in br) / 2
    L.hh = ((n - 1) * lh + cap) / 2
    return L


def _line_positions(L, align="center"):
    """-> [(token index, x_left, baseline_y, width)] relative to the text block centre"""
    out = []
    n = len(L.lines)
    tot = (n - 1) * L.lh + L.cap
    for k, (j, i) in enumerate(L.lines):
        base = -tot / 2 + L.cap + k * L.lh
        ws = [L.faces[m].width(L.toks[m], L.size) for m in range(j, i)]
        lw = sum(ws) + L.space * (i - j - 1)
        x = -lw / 2 if align == "center" else (-L.hw if align == "left" else L.hw - lw)
        for m, wm in zip(range(j, i), ws):
            out.append((m, x, base, wm))
            x += wm + L.space
    return out


# ================================================================== geometry
def _se_dense(rx, ry, p, n=720):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    x = rx * np.sign(c) * np.abs(c) ** (2.0 / p)
    y = ry * np.sign(s) * np.abs(s) ** (2.0 / p)
    return np.stack([x, y], 1)


def _resample(pts, step):
    """closed polyline -> points evenly spaced by arc length (+ outward normals)"""
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    cum = np.concatenate([[0], np.cumsum(seg)])
    per = cum[-1]
    m = max(12, int(round(per / step)))
    s = np.linspace(0, per, m, endpoint=False)
    P = np.vstack([pts, pts[:1]])
    x = np.interp(s, cum, P[:, 0])
    y = np.interp(s, cum, P[:, 1])
    q = np.stack([x, y], 1)
    tg = np.roll(q, -1, 0) - np.roll(q, 1, 0)
    nrm = np.stack([tg[:, 1], -tg[:, 0]], 1)
    nrm /= np.maximum(np.hypot(nrm[:, 0], nrm[:, 1]), 1e-9)[:, None]
    # outward: for a CCW (in y-down: clockwise-looking) ring check against the centroid
    if np.mean(np.sum((q - q.mean(0)) * nrm, 1)) < 0:
        nrm = -nrm
    return q, nrm, s / per, per


def _rho(rx, ry, p, ux, uy):
    return 1.0 / ((abs(ux) / rx) ** p + (abs(uy) / ry) ** p) ** (1.0 / p)


def _body(kind, style, rx, ry, p, em, seed, t):
    """closed outline (local coords, centre 0,0) of the balloon body"""
    rng = np.random.default_rng(seed)
    base = _se_dense(rx, ry, p)
    if kind == "shout":
        q, nrm, s, per = _resample(base, em * 0.55)
        m = len(q) // 2 * 2
        q, nrm, s = q[:m], nrm[:m], s[:m]
        spike = em * (0.42 + 0.5 * rng.random(m)) * (np.arange(m) % 2 == 0)
        spike -= em * 0.1 * (np.arange(m) % 2 == 1)
        sh = em * 0.15 * (rng.random(m) - 0.5)
        tg = np.stack([-nrm[:, 1], nrm[:, 0]], 1)
        return q + nrm * spike[:, None] + tg * sh[:, None]
    if kind == "radio":
        q, nrm, s, per = _resample(base, em * 0.2)
        m = len(q) // 2 * 2
        q, nrm = q[:m], nrm[:m]
        z = em * 0.2 * (np.arange(m) % 2 == 0) - em * 0.02
        return q + nrm * z[:, None]
    q, nrm, s, per = _resample(base, em * 0.12)
    ph = rng.random(4) * 2 * np.pi
    if style == "anime":
        disp = em * (0.16 * np.sin(2 * np.pi * (s * 5 + ph[0] / 6.28)) + 0.09 * np.sin(2 * np.pi * (s * 9) + ph[1])
                     + 0.05 * np.sin(2 * np.pi * (s * 17) + ph[2]))
    elif style == "devil":
        k = np.arange(len(q))
        jag = rng.random(len(q))
        jag = np.convolve(np.concatenate([jag[-2:], jag, jag[:2]]), np.ones(3) / 3, "same")[2:-2]
        wob = np.sin(2 * np.pi * (s * 7) + ph[0] + t * 1.3) * 0.5 + np.sin(2 * np.pi * s * 13 + ph[1]) * 0.3
        disp = em * (0.22 * jag - 0.08 + 0.1 * wob) * (1 + 0.25 * (k % 3 == 0))
    else:
        disp = em * (0.03 * np.sin(2 * np.pi * s * 3 + ph[0]) + 0.015 * np.sin(2 * np.pi * s * 5 + ph[1]))
    return q + nrm * disp[:, None]


def _bez(p0, p1, p2, n):
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    tg = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1)
    return pts, tg


def _bez3(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    tg = 3 * (1 - t) ** 2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t ** 2 * (p3 - p2)
    return pts, tg


def _ribbon(pts, tg, wid):
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
    nrm /= np.maximum(np.hypot(nrm[:, 0], nrm[:, 1]), 1e-9)[:, None]
    a = pts + nrm * (wid / 2)[:, None]
    b = pts - nrm * (wid / 2)[:, None]
    return np.vstack([a, b[::-1]])


def _tail(kind, style, rx, ry, p, em, tip, seed, t):
    """tail polygon (local coords) from inside the body to the tip, or None"""
    tip = np.asarray(tip, float)
    L = float(np.hypot(*tip))
    if L < 1e-6:
        return None
    u = tip / L
    e = _rho(rx, ry, p, u[0], u[1])
    if L <= e * 1.05:
        return None
    s0 = e * 0.55
    p0 = u * s0
    nrm = np.array([-u[1], u[0]])
    wb = min(em * 1.05, 0.55 * min(rx, ry) + em * 0.2)
    if kind == "shout":
        mid = (p0 + tip) / 2 + nrm * (L - s0) * 0.06
        pts, tg = _bez(p0, mid, tip, 12)
        wid = wb * 1.1 * (1 - np.linspace(0, 1, 12)) ** 1.0
        return _ribbon(pts, tg, wid)
    if kind == "radio":
        # lightning bolt: three hard zig-zags, stays bold until the tip
        fr = np.array([0, 0.3, 0.46, 0.68, 0.82, 1.0])
        off = np.array([0, 0.7, -0.55, 0.5, -0.3, 0]) * em
        pts = p0[None] + (tip - p0)[None] * fr[:, None] + nrm[None] * off[:, None]
        wid = wb * 0.95 * (1 - fr) ** 0.55
        tg = np.gradient(pts, axis=0)
        return _ribbon(pts, tg, wid)
    # curved tapered tail, bowed towards the balloon's vertical axis (drops out, then hooks)
    bow = (L - s0) * 0.2
    c1 = (p0 + tip) / 2 + nrm * bow
    c2 = (p0 + tip) / 2 - nrm * bow
    ctrl = c1 if abs(c1[0]) < abs(c2[0]) else c2
    if style == "devil":
        sgn = 1 if ctrl is c1 else -1
        amp = min((L - s0) * 0.16, em * 2.2)
        a1 = p0 + (tip - p0) * 0.33 + nrm * sgn * amp
        a2 = p0 + (tip - p0) * 0.66 - nrm * sgn * amp
        pts, tg = _bez3(p0, a1, a2, tip, 40)
        f = np.linspace(0, 1, 40)
        wid = np.maximum(wb * (1 - f) ** 2.2, em * 0.13)
        rib = _ribbon(pts, tg, wid)
        # spade at the tip
        d = tg[-1] / max(1e-9, np.hypot(*tg[-1]))
        nn = np.array([-d[1], d[0]])
        s_ = em * 0.55
        spade = np.array([tip + d * s_ * 0.9, tip + nn * s_ * 0.55 - d * s_ * 0.15, tip + nn * s_ * 0.12 - d * s_ * 0.05,
                          tip - d * s_ * 0.2, tip - nn * s_ * 0.12 - d * s_ * 0.05, tip - nn * s_ * 0.55 - d * s_ * 0.15])
        return [rib, spade]
    pts, tg = _bez(p0, ctrl, tip, 28)
    f = np.linspace(0, 1, 28)
    wid = wb * (1 - f) ** 1.15
    return _ribbon(pts, tg, wid)


def _drips(outline, em, seed, t):
    """devil balloon: melting drips hanging from the underside (grow slowly, drops fall)"""
    from shapely.geometry import Point
    rng = np.random.default_rng(seed + 7)
    q = outline
    ymax = q[:, 1].max()
    low = np.where(q[:, 1] > ymax - em * 1.2)[0]
    if len(low) == 0:
        return []
    geoms = []
    xs = q[low, 0]
    x0, x1 = xs.min(), xs.max()
    k = max(3, int((x1 - x0) / (em * 2.2)))
    for i in range(k):
        fx = (i + 0.5 + (rng.random() - 0.5) * 0.6) / k
        x = x0 + (x1 - x0) * fx
        j = low[np.argmin(np.abs(xs - x))]
        px, py = q[j]
        L0 = em * (0.6 + 1.4 * rng.random())
        per = 2.2 + 2.0 * rng.random()
        ph = rng.random() * per
        cyc = ((t + ph) % per) / per
        grow = L0 * (0.35 + 0.65 * _smooth(cyc / 0.8)) if cyc < 0.8 else L0 * (1 - 0.65 * _smooth((cyc - 0.8) / 0.2))
        wd = em * (0.26 + 0.2 * rng.random())
        # a drip: wide melting root, thin neck, round drop
        s = np.linspace(0, 1, 14)
        hw = wd * 1.2 * (1 - s) ** 2.2 + wd * 0.42 + wd * 0.12 * s
        ys = py - em * 0.35 + (grow + em * 0.35) * s
        prof = np.concatenate([np.stack([px - hw, ys], 1), np.stack([px + hw, ys], 1)[::-1]])
        from shapely.geometry import Polygon
        geoms.append(Polygon(prof).buffer(0))
        geoms.append(Point(px, py + grow - wd * 0.15).buffer(wd * 0.6, 12))
        if cyc > 0.8:           # the drop lets go and falls
            fall = (cyc - 0.8) / 0.2
            geoms.append(Point(px, py + L0 + em * 0.1 + fall * fall * em * 6).buffer(wd * 0.6 * (1 - fall * 0.4), 10))
    return geoms


def _geom_avoid(avoid, margin):
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    if not avoid:
        return None
    gs = []
    for a in avoid:
        if a is None:
            continue
        if hasattr(a, "geom_type"):
            gs.append(a)
        elif len(a) == 4 and not hasattr(a[0], "__len__"):
            x0, y0, x1, y1 = a
            gs.append(box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        elif len(a) >= 3:
            gs.append(Polygon(a))
    if not gs:
        return None
    g = unary_union(gs)
    return g.buffer(margin, 4) if margin > 0 else g


def _clip_out(ctx, geom):
    """clip ctx to everything EXCEPT geom (shapely)"""
    ctx.new_path()
    ctx.rectangle(-1e5, -1e5, 2e5, 2e5)
    polys = getattr(geom, "geoms", [geom])
    for g in polys:
        for ring in [g.exterior] + list(g.interiors):
            c = list(ring.coords)
            ctx.move_to(*c[0])
            for pt in c[1:]:
                ctx.line_to(*pt)
            ctx.close_path()
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.clip()
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)


def _path_rings(ctx, geom, dx=0.0, dy=0.0):
    polys = getattr(geom, "geoms", [geom])
    for g in polys:
        if g.geom_type != "Polygon":
            continue
        for ring in [g.exterior] + list(g.interiors):
            c = np.asarray(ring.coords)
            ctx.move_to(c[0, 0] + dx, c[0, 1] + dy)
            for x, y in c[1:]:
                ctx.line_to(x + dx, y + dy)
            ctx.close_path()


def _nib_stroke(ctx, w, nib=0.55, angle=-0.6, preserve=False):
    """stroke the current path with an oval pen (a lettering nib): thick/thin like hand inking"""
    ctx.save()
    ctx.rotate(angle)
    ctx.scale(1.0, nib)
    ctx.set_line_width(w)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    if preserve:
        ctx.stroke_preserve()
    else:
        ctx.stroke()
    ctx.restore()


# ================================================================== the balloon machine
class _Piece:
    """everything needed to draw one lettered element at full size"""
    __slots__ = ("L", "kind", "style", "cx", "cy", "geom", "bbox", "anchor", "lw", "align", "seed", "em",
                 "tip", "body", "tailg")


def _build(text_toks, fonts, *, x, y, w, tail, kind, style, size, anchor, avoid, bounds, align, t, seed):
    from shapely.geometry import Polygon, box
    from shapely import affinity
    from shapely.ops import unary_union
    p = {"shout": 2.2, "radio": 2.3}.get(kind, 2.45)
    lh_k = {"title": 1.04, "caption": 1.17, "footnote": 1.1}.get(kind, 1.13)
    if fonts[0] in (FONT_CONDENSED, FONT_IMPACT, FONT_OSWALD):
        lh_k *= 1.0 if kind == "title" else 0.98
    if kind in _BOXED and w is not None:
        pad = size * (0.62 if kind == "caption" else 0.4)
        maxw = w - 2 * pad if kind == "caption" else w
    else:
        maxw = w
    L = _layout(tuple(text_toks), tuple(fonts), float(size), kind, float(maxw) if maxw else None, lh_k, p)
    em = size
    pc = _Piece()
    pc.L, pc.kind, pc.style, pc.em, pc.align, pc.seed = L, kind, style, em, align, seed
    pc.lw = size * {"caption": 0.11, "footnote": 0.16, "title": 0.0, "shout": 0.15, "radio": 0.12}.get(kind, 0.12)
    if style == "divine":
        pc.lw *= 0.8
    # ---------- local geometry
    if kind in _BOXED:
        if kind == "caption":
            padx, pady = size * 0.62, size * 0.5
            hw = (w / 2) if w is not None else L.hw + padx
        elif kind == "footnote":
            padx, pady = size * 0.42, size * 0.34
            hw = L.hw + padx
        else:
            padx = pady = 0.0
            hw = L.hw
        hh = L.hh + pady
        body_local = box(-hw, -hh, hw, hh)
    else:
        outline = _body(kind, style, L.rx, L.ry, L.p, em, seed, t)
        body_local = Polygon(outline).buffer(0)
    drips = _drips(outline, em, seed, t) if (style == "devil" and kind not in _BOXED) else None
    # ---------- anchor -> centre
    anchor = anchor or ("tl" if kind in ("caption", "footnote") else "c")
    ax = {"l": 1, "r": -1}.get(anchor[-1], 0) if anchor not in ("t", "b", "c") else 0
    ay = {"t": 1, "b": -1}.get(anchor[0], 0)
    if anchor in ("l", "r"):
        ay = 0
    minx, miny, maxx, maxy = body_local.bounds
    cx = x - (minx if ax == 1 else maxx if ax == -1 else 0)
    cy = y - (miny if ay == 1 else maxy if ay == -1 else 0)

    def assemble(cx, cy, final=False):
        g = affinity.translate(body_local, cx, cy)
        if final and drips:
            g = unary_union([g] + [affinity.translate(d, cx, cy) for d in drips]).buffer(0)
        body = g
        tl = None
        if tail is not None and kind not in _BOXED:
            tp = _tail(kind, style, L.rx, L.ry, L.p, em, (tail[0] - cx, tail[1] - cy), seed, t)
            if tp is not None:
                parts = tp if isinstance(tp, list) else [tp]
                tl = unary_union([Polygon(q).buffer(0) for q in parts])
                tl = affinity.translate(tl, cx, cy)
                g = unary_union([g, tl]).buffer(0)
        return g, tl, body

    geom, _, _ = assemble(cx, cy)
    # ---------- avoid / bounds: nudge the whole balloon clear (smallest move wins)
    av = _geom_avoid(avoid, size * 0.35)
    bnd = None
    if bounds is not False:
        bx = bounds or (8, 8, W - 8, H - 8)
        bnd = box(*bx)

    def bad(g, body):
        v = 0.0
        if av is not None and g.intersects(av):
            v += g.intersection(av).area + 1.0
        if bnd is not None and not bnd.contains(body):
            v += body.difference(bnd).area + 1.0
        return v

    body_now = affinity.translate(body_local, cx, cy)
    b0 = bad(geom, body_now)
    if b0 > 0:
        best = (b0, cx, cy, geom)
        step = size * 0.5
        found = False
        for ring in range(1, 28):
            r = step * ring
            m = 8 + ring * 2
            for k in range(m):
                a = 2 * math.pi * k / m + (0.5 if ring % 2 else 0) * math.pi / m
                ncx, ncy = cx + math.cos(a) * r, cy + math.sin(a) * r * 0.8
                body_n = affinity.translate(body_local, ncx, ncy)
                if av is not None and body_n.intersects(av):
                    continue
                if bnd is not None and not bnd.contains(body_n):
                    continue
                g, _, _ = assemble(ncx, ncy)
                v = bad(g, body_n)
                if v < best[0]:
                    best = (v, ncx, ncy, g)
                    if v == 0:
                        found = True
                        break
            if found:
                break
        _, cx, cy, geom = best
    geom, tl, body = assemble(cx, cy, final=True)
    pc.cx, pc.cy, pc.geom, pc.body, pc.tailg = cx, cy, geom, body, tl
    pc.bbox = tuple(affinity.translate(body_local, cx, cy).union(tl).bounds if tl is not None
                    else affinity.translate(body_local, cx, cy).bounds)
    pc.tip = tail
    return pc, av


def _draw_piece(ctx, pc, tok_state, *, alpha=1.0, pop=1.0, pop_origin=None, t=0.0, av=None, jitter=0.0):
    """tok_state: list of (alpha, scale) per token (or None = hidden)"""
    L, kind, style, em = pc.L, pc.kind, pc.style, pc.em
    paper, inkc = _paper(), _inkc()
    ctx.save()
    if av is not None:
        _clip_out(ctx, av)
    group = alpha < 0.999
    if group:
        ctx.push_group()
    if pop != 1.0:
        ox, oy = pop_origin if pop_origin is not None else (pc.cx, pc.cy)
        ctx.translate(ox, oy)
        ctx.scale(pop, pop)
        ctx.translate(-ox, -oy)
    # ---------- body
    if kind != "title":
        g = pc.geom
        if pop != 1.0 and pc.tip is not None and pc.tailg is not None and pop > 0.05:
            # keep the tail tip pinned on the speaker while the body inflates
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
            tx, ty = pc.tip
            t2 = (ox + (tx - ox) / pop - pc.cx, oy + (ty - oy) / pop - pc.cy)
            tp = _tail(kind, style, L.rx, L.ry, L.p, em, t2, pc.seed, t)
            if tp is not None:
                parts = tp if isinstance(tp, list) else [tp]
                tl = unary_union([Polygon(q + np.array([pc.cx, pc.cy])).buffer(0) for q in parts])
                g = unary_union([pc.body, tl]).buffer(0)
        if style == "divine":
            outer = g.buffer(em * 0.3, 16)
            # glory: fine engraved rays radiating from the balloon's heart, long/short rhythm, shimmering
            body = pc.body.buffer(em * 0.3, 16)
            ring = body.exterior if body.geom_type == "Polygon" else max(body.geoms, key=lambda q: q.area).exterior
            tailz = pc.tailg.buffer(em * 0.7) if pc.tailg is not None else None
            pts = np.asarray(ring.coords)
            q, nrm, _, per = _resample(pts[:-1], em * 0.26)
            ctx.new_path()
            from shapely.geometry import Point
            cxy = np.array([pc.cx, pc.cy])
            rhythm = (1.0, 0.38, 0.62, 0.38)
            for k in range(len(q)):
                if tailz is not None and tailz.contains(Point(*q[k])):
                    continue
                d = q[k] - cxy
                d = d / max(1e-9, np.hypot(*d)) * 0.6 + nrm[k] * 0.4
                d /= max(1e-9, np.hypot(*d))
                ln = em * 1.25 * rhythm[k % 4] * (0.85 + 0.3 * _h01(k, pc.seed)) \
                    * (0.85 + 0.15 * math.sin(t * 2.2 - k * 0.45))
                tg = np.array([-d[1], d[0]])
                a = q[k] + d * em * 0.14
                wd = em * 0.028
                ctx.move_to(*(a + tg * wd))
                ctx.line_to(*(a + d * ln))
                ctx.line_to(*(a - tg * wd))
                ctx.close_path()
            ctx.set_source_rgba(*inkc, 1.0)
            ctx.fill()
            ctx.new_path()
            _path_rings(ctx, outer)
            ctx.set_source_rgba(*paper, 1.0)
            ctx.fill_preserve()
            ctx.set_source_rgba(*inkc, 1.0)
            ctx.set_line_width(pc.lw * 0.45)
            ctx.stroke()
        ctx.new_path()
        _path_rings(ctx, g)
        ctx.set_source_rgba(*paper, 1.0)
        ctx.fill_preserve()
        ctx.set_source_rgba(*inkc, 1.0)
        if kind in ("caption", "footnote"):
            ctx.set_line_width(pc.lw)
            ctx.set_line_join(cairo.LINE_JOIN_MITER)
            ctx.stroke()
        else:
            _nib_stroke(ctx, pc.lw * (1.25 if style in ("anime", "devil") else 1.0),
                        nib=0.5 if style == "anime" else 0.6)
    # ---------- lettering
    ctx.set_source_rgba(*inkc, 1.0)
    pos = _line_positions(L, pc.align)
    for m, lx, base, wm in pos:
        st = tok_state[m] if tok_state is not None else (1.0, 1.0)
        if st is None or st[0] <= 0.002:
            continue
        a, s = st[0], st[1]
        rep = st[2] if len(st) > 2 else None
        x0, y0 = pc.cx + lx, pc.cy + base
        if jitter:
            x0 += jitter * (_h01(m, int(t * 12), 1) - 0.5)
            y0 += jitter * (_h01(m, int(t * 12), 2) - 0.5)
        ctx.new_path()
        ctx.save()
        ox, oy = x0 + wm / 2, y0 - L.cap / 2
        if s != 1.0:
            ctx.translate(ox, oy)
            ctx.scale(s, s)
            ctx.translate(-ox, -oy)
        if rep is not None:
            # the word has turned into FLAGS: squeezed/stretched into the slot the word occupied
            rw = L.faces[m].width(rep, L.size)
            sx = min(1.5, max(0.38, (wm + L.space * 0.7) / max(rw, 1e-6)))
            ctx.translate(ox, oy)
            ctx.scale(sx, 1.0)
            ctx.translate(-ox, -oy)
            L.faces[m].path(ctx, rep, ox - rw / 2, y0, L.size)
        else:
            L.faces[m].path(ctx, L.toks[m], x0, y0, L.size)
        ctx.restore()
        ctx.set_source_rgba(*inkc, a)
        ctx.fill()
    if group:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(alpha)
    ctx.restore()


def _fonts_for(toks, font, italic):
    base = font or FONT_LETTER
    out = []
    i0, i1 = (italic if italic else (None, None))
    n = len(toks)
    if italic:
        i0 = i0 or 0
        i1 = n if i1 is None else (i1 if i1 >= 0 else n + i1)
    for i in range(n):
        out.append(FONT_LETTER_ALT if italic and i0 <= i < i1 and base == FONT_LETTER else base)
    return out


def balloon(ctx, text, x, y, w=None, *, tail=None, kind="balloon", style=None, reveal=None, size=None,
            alpha=1.0, font=None, t=0.0, avoid=None, bounds=None, anchor=None, align="center", italic=None,
            flagify=0.0, flag_word="FLAGS"):
    toks = text.split() or [" "]
    kind = "balloon" if kind == "offpanel" else kind
    size = size or DEFAULT_SIZE.get(kind, 38)
    fonts = _fonts_for(toks, font or (FONT_TITLE if kind == "title" else None), italic)
    seed = zlib.crc32((text + kind + str(style)).encode())
    pc, av = _build(toks, fonts, x=x, y=y, w=w, tail=tail, kind=kind, style=style, size=size, anchor=anchor,
                    avoid=avoid, bounds=bounds, align=align, t=t, seed=seed)
    st = [(1.0, 1.0) if reveal is None else (_clamp(reveal - i), 1.0) for i in range(len(toks))]
    if flagify > 0:
        st = _flagify(st, toks, flagify, flag_word, text, t)
    _draw_piece(ctx, pc, st, alpha=alpha, t=t, av=av, jitter=em_j(style, size))
    return pc.bbox


def em_j(style, size):
    return size * 0.05 if style == "devil" else 0.0


def caption(ctx, text, x, y, w, *, reveal=None, size=34, alpha=1.0, font=None, align="center", anchor=None,
            avoid=None, bounds=None):
    return balloon(ctx, text, x, y, w, kind="caption", reveal=reveal, size=size, alpha=alpha, font=font,
                   align=align, anchor=anchor, avoid=avoid, bounds=bounds)


def footnote(ctx, text, x, y, *, size=20, progress=1.0, alpha=1.0, anchor="tl", font=None, avoid=None,
             bounds=None, w=None):
    toks = text.split() or [" "]
    fonts = [font or FONT_LETTER] * len(toks)
    seed = zlib.crc32(text.encode())
    pc, av = _build(toks, fonts, x=x, y=y, w=w, tail=None, kind="footnote", style=None, size=size,
                    anchor=anchor, avoid=avoid, bounds=bounds, align="center", t=0.0, seed=seed)
    n = len(text.replace(" ", ""))
    shown = progress * n
    st, k = [], 0
    for tk in toks:
        st.append((_clamp(shown - k), 1.0) if len(tk) else None)
        k += len(tk)
    _draw_typed(ctx, pc, [(_clamp((shown - kk) / max(1, len(tk))), 1.0) for tk, kk in _cum(toks)],
                alpha=alpha, av=av)
    return pc.bbox


def _cum(toks):
    k = 0
    for tk in toks:
        yield tk, k
        k += len(tk)


def _draw_typed(ctx, pc, tok_frac, *, alpha=1.0, av=None, pop=1.0, pop_origin=None, dx=0.0):
    """footnote style: each token types on letter by letter; tok_frac = [(fraction typed, _)]"""
    L = pc.L
    full = [(1.0, 1.0) if f >= 1 else None for f, _ in tok_frac]
    ctx.save()
    ctx.translate(dx, 0)
    _draw_piece(ctx, pc, full, alpha=alpha, pop=pop, pop_origin=pop_origin, av=av)
    # partially typed token: draw its prefix
    inkc = _inkc()
    for (m, lx, base, wm) in _line_positions(L, pc.align):
        f = tok_frac[m][0]
        if 0 < f < 1:
            tk = L.toks[m]
            nch = int(math.ceil(f * len(tk)))
            ctx.new_path()
            L.faces[m].path(ctx, tk[:nch], pc.cx + lx, pc.cy + base, L.size)
            ctx.set_source_rgba(*inkc, alpha)
            ctx.fill()
    ctx.restore()


# ================================================================== voice-synced lettering
@lru_cache(maxsize=256)
def _line_plan(line_id, text, tokens, mark, fid):
    from .timeline import get_timeline
    tl = get_timeline()
    l = tl.line(line_id)
    toks = text.split()
    times = _align(toks, l["words"], l["start"], l["end"])
    i0, i1 = 0, len(toks)
    if tokens is not None:
        a, b = tokens
        i0 = 0 if a is None else (a if a >= 0 else len(toks) + a)
        i1 = len(toks) if b is None else (b if b >= 0 else len(toks) + b)
    toks, times = toks[i0:i1], times[i0:i1]
    if mark and toks:
        toks = toks[:-1] + [toks[-1] + mark]
    return tuple(toks), tuple(times), (i0, i1)


def _state_for(fc_T, kind, toks, times, phr):
    st = []
    for i, (tok, (a, e)) in enumerate(zip(toks, times)):
        on = times[phr[i]][0] if phr is not None else a
        u = (fc_T - on + 0.02) / (0.12 if phr is not None else 0.05)
        al = _clamp(u)
        if al <= 0:
            st.append(None)
            continue
        s = 1.0
        if _is_caps(tok):
            v = (fc_T - a) / 0.12
            if 0 <= v <= 1:
                s = 1.0 + 0.12 * (_smooth(v / 0.35) if v < 0.35 else 1 - _smooth((v - 0.35) / 0.65))
        st.append((al, s))
    return st


def _resolve(fc, line_id, kind, style, text, tokens, mark, font, size, italic=None):
    l = fc.tl.line(line_id)
    meta = l.get("meta", {})
    spec = SPEC.get(line_id, {})
    kind = kind or meta.get("letter", "balloon")
    style = style if style is not None else meta.get("style")
    if style not in ("divine", "devil", "anime"):
        style = None
    if text is None:
        text = meta["footnote"] if kind == "footnote" and meta.get("footnote") and tokens is None else l["text"]
    if mark == "auto":
        fn = meta.get("footnote", "")
        mark = re.match(r"^\*+", fn).group(0) if (kind != "footnote" and fn and re.match(r"^\*+", fn)
                                                  and tokens is None) else ""
    font = font or spec.get("font") or (FONT_TITLE if kind == "title" else FONT_LETTER)
    italic = italic if italic is not None else spec.get("italic")
    size = size or spec.get("size") or DEFAULT_SIZE.get(kind, 38)
    return l, kind, style, text, mark, font, size, italic


def _say_build(ctx, fc, line_id, x, y, w, tail, kind, style, size, font, avoid, bounds, text, tokens, mark,
               anchor, align, italic, T):
    l, kind, style, text, mark, font, size, italic = _resolve(fc, line_id, kind, style, text, tokens, mark,
                                                              font, size, italic)
    toks, times, _ = _line_plan(line_id, text, tuple(tokens) if tokens is not None else None, mark or "", 0)
    fonts = _fonts_for(list(toks), font, italic)
    bkind = "balloon" if kind == "offpanel" else kind
    if kind == "offpanel" and tail is None:
        bx = bounds or (0, 0, W, H)
        tail = (bx[0], y + size) if x - bx[0] < bx[2] - x else (bx[2], y + size)
    if kind == "offpanel" and tail is not None:
        # run the tail past the frame edge so the frame crops it
        tx, ty = tail
        bx = bounds or (0, 0, W, H)
        ex = min(abs(tx - bx[0]), abs(bx[2] - tx), abs(ty - bx[1]), abs(bx[3] - ty))
        if ex == abs(tx - bx[0]):
            tail = (bx[0] - size * 1.5, ty)
        elif ex == abs(bx[2] - tx):
            tail = (bx[2] + size * 1.5, ty)
        elif ex == abs(ty - bx[1]):
            tail = (tx, bx[1] - size * 1.5)
        else:
            tail = (tx, bx[3] + size * 1.5)
    seed = zlib.crc32((line_id + text).encode())
    pc, av = _build(list(toks), fonts, x=x, y=y, w=w, tail=tail, kind=bkind, style=style, size=size,
                    anchor=anchor, avoid=avoid, bounds=bounds, align=align or "center", t=T, seed=seed)
    return l, kind, style, toks, times, pc, av, size


def measure(ctx, fc, line_id, x, y, *, w=None, tail=None, kind=None, style=None, size=None, font=None,
            avoid=None, bounds=None, text=None, tokens=None, mark="auto", anchor=None, align=None, italic=None,
            T=None, hold=0.5, pre=0.12, **_):
    T = fc.T if T is None else T
    if callable(avoid):
        l, _, _, text0, mark0, _, _, _ = _resolve(fc, line_id, kind, style, text, tokens, mark, font, size, italic)
        _, times, _ = _line_plan(line_id, text0, tuple(tokens) if tokens is not None else None, mark0 or "", 0)
        t_in = min(a for a, _ in times) - pre
        t_out = max(l["end"], max(e for _, e in times)) + hold
        avoid = _avoid_span(avoid, t_in, t_out, T)[0]
    return _say_build(ctx, fc, line_id, x, y, w, tail, kind, style, size, font, avoid, bounds, text, tokens,
                      mark, anchor, align, italic, T)[5].bbox


def measure_text(text, x, y, *, w=None, tail=None, kind="balloon", style=None, size=None, font=None,
                 anchor=None, avoid=None, bounds=None):
    toks = text.split() or [" "]
    kind = "balloon" if kind == "offpanel" else kind
    size = size or DEFAULT_SIZE.get(kind, 38)
    fonts = _fonts_for(toks, font or (FONT_TITLE if kind == "title" else None), None)
    pc, _ = _build(toks, fonts, x=x, y=y, w=w, tail=tail, kind=kind, style=style, size=size, anchor=anchor,
                   avoid=avoid, bounds=bounds, align="center", t=0.0, seed=zlib.crc32(text.encode()))
    return pc.bbox


def _flagify(st, toks, amount, word, key, T):
    """turn a growing share of the (visible) words into FLAGS, in a seeded order: each word flickers
    between itself and FLAGS for a few frames, then stamps in with a punch (the crowd's incomprehension)"""
    n = len(toks)
    order = sorted(range(n), key=lambda i: _h01(key, i, "flagify"))
    v = _clamp(amount) * n
    out = list(st)
    for r, i in enumerate(order):
        d = v - r
        s = out[i]
        if d <= 0 or s is None:
            continue
        if _norm(toks[i]) == _norm(word):
            continue
        if d < 0.6 and int(T * 24) % 3 == 0:
            continue                                   # glitch frame: the old word shows through
        lead = re.match(r"^[^A-Za-z0-9]*", toks[i]).group(0)
        trail = re.search(r"[^A-Za-z0-9]*$", toks[i]).group(0)
        u = _clamp(d / 0.6)
        punch = 1.0 + 0.18 * (1 - u) * (u > 0)
        out[i] = (s[0], s[1] * punch, lead + word + trail)
    return out


def _avoid_span(avoid, t_in, t_out, T):
    """avoid may be a callable T -> [shapes]: place against the union over the element's whole lifetime
    (stable, no jumping), clip against the current frame. Returns (placement list, current list)."""
    if not callable(avoid):
        return avoid, avoid
    n = max(2, int((t_out - t_in) * 6) + 1)
    place = []
    for k in range(n):
        place.extend(avoid(t_in + (t_out - t_in) * k / (n - 1)) or [])
    return place, (avoid(T) or [])


def say(ctx, fc, line_id, x, y, *, w=None, tail=None, kind=None, style=None, size=None, font=None, alpha=1.0,
        hold=0.5, pre=0.12, avoid=None, bounds=None, text=None, tokens=None, mark="auto", anchor=None,
        align=None, italic=None, T=None, flagify=0.0, flag_word="FLAGS"):
    T = fc.T if T is None else T
    l, kind0, style0, text0, mark0, font0, size0, italic0 = _resolve(fc, line_id, kind, style, text, tokens,
                                                                      mark, font, size, italic)
    toks, times, _ = _line_plan(line_id, text0, tuple(tokens) if tokens is not None else None, mark0 or "", 0)
    if not toks:
        return None
    t_first = min(a for a, _ in times)
    t_last = max(l["end"], max(e for _, e in times))
    t_in = t_first - pre
    t_out = t_last + hold
    if T < t_in or T > t_out + 0.18:
        return None
    av_place, av_now = _avoid_span(avoid, t_in, t_out, T)
    l, kind, style, toks, times, pc, av, size = _say_build(ctx, fc, line_id, x, y, w, tail, kind, style, size,
                                                            font, av_place, bounds, text, tokens, mark, anchor,
                                                            align, italic, T)
    if callable(avoid):
        av = _geom_avoid(av_now, size * 0.12)
    # envelope: pop in with overshoot, fade out
    u = (T - t_in) / 0.24
    if kind in ("caption", "footnote"):
        pop = 1.0 + 0.04 * (1 - _smooth(u))
        a_in = _smooth(u * 2.0)
    elif kind == "title":
        pop, a_in = 1.0, 1.0
    else:
        pop = 0.45 + 0.55 * _out_back(u, 2.2)
        a_in = _clamp(u * 3.5)
    v = (T - t_out) / 0.18
    a_out = 1 - _smooth(v)
    if v > 0:
        pop *= 1 - 0.03 * _smooth(v)
    origin = None
    if pc.tip is not None and kind not in _BOXED:
        # grow out of the speaker: scale about the point where the tail leaves the body
        tx, ty = pc.tip
        dx, dy = tx - pc.cx, ty - pc.cy
        d = math.hypot(dx, dy) or 1.0
        e = _rho(pc.L.rx, pc.L.ry, pc.L.p, dx / d, dy / d)
        origin = (pc.cx + dx / d * e * 0.9, pc.cy + dy / d * e * 0.9)
    if kind == "footnote":
        # typed on letter by letter across each word
        frac = []
        for tk, (a, e) in zip(toks, times):
            dur = max(0.06, (e - a) * 0.85)
            frac.append((_clamp((T - a + 0.02) / dur), 1.0))
        _draw_typed(ctx, pc, frac, alpha=alpha * a_in * a_out, av=av, pop=pop)
        return pc.bbox
    phr = None
    if kind == "caption":
        ids = _phrase_ids(list(toks), list(times))
        first = {}
        for i, p in enumerate(ids):
            first.setdefault(p, i)
        phr = [first[p] for p in ids]
    st = _state_for(T, kind, toks, times, phr)
    if T > t_last:
        st = [(1.0, s[1]) if s is not None else (1.0, 1.0) for s in st]
    if flagify > 0:
        st = _flagify(st, toks, flagify, flag_word, line_id, T)
    _draw_piece(ctx, pc, st, alpha=alpha * a_in * a_out, pop=pop, pop_origin=origin, t=T, av=av,
                jitter=em_j(style, size))
    return pc.bbox


def say_footnote(ctx, fc, line_id, x, y, *, t0=None, anchor="br", size=24, hold=None, T=None, avoid=None,
                 bounds=None, font=None, text=None):
    """the line's meta['footnote'] ('*1ST VEXILLIANS 11s') in a footnote box that slides in at t0"""
    T = fc.T if T is None else T
    l = fc.tl.line(line_id)
    text = text or l.get("meta", {}).get("footnote")
    if not text:
        return None
    if t0 is None:
        ws = fc.tl.words(line_id)
        t0 = ws[-1]["s"] if ws else l["start"]
    t_end = l["end"] + (hold if hold is not None else 1.2)
    if T < t0 or T > t_end + 0.18:
        return None
    toks = text.split()
    pc, av = _build(toks, [font or FONT_LETTER_ALT] * len(toks), x=x, y=y, w=None, tail=None, kind="footnote",
                    style=None, size=size, anchor=anchor, avoid=avoid, bounds=bounds, align="center", t=T,
                    seed=zlib.crc32(text.encode()))
    u = _clamp((T - t0) / 0.22)
    slide = (1 - _out_back(u, 1.6)) * (pc.bbox[2] - pc.bbox[0] + size) * (1 if anchor.endswith("r") else -1)
    a = _clamp((T - t0) / 0.08) * (1 - _smooth((T - t_end) / 0.18))
    _draw_typed(ctx, pc, [(1.0, 1.0)] * len(toks), alpha=a, av=av, dx=slide)
    return pc.bbox


# ================================================================== onomatopoeia
def sfx(ctx, text, x, y, size, *, rot=0.0, t=None, alpha=1.0, depth=0.0, depth_ang=0.9, font=None, avoid=None,
        seed=0):
    """onomatopoeia centred on (x, y)"""
    f = _Face(font or FONT_SFX)
    if t is not None and t < 0:
        return None
    chars = list(text)
    rng = np.random.default_rng(zlib.crc32(text.encode()) + seed)
    n = len(chars)
    ws = [f.width(c, size) for c in chars]
    tr = size * 0.02
    total = sum(ws) + tr * (n - 1)
    cap = f.cap * size
    inkc, paper = _inkc(), _paper()
    av = _geom_avoid(avoid, size * 0.15)
    ctx.save()
    if av is not None:
        _clip_out(ctx, av)
    if alpha < 0.999:
        ctx.push_group()
    ctx.translate(x, y)
    ctx.rotate(rot)
    if t is not None and t < 0.35:
        sh = (1 - t / 0.35) * size * 0.06
        ctx.translate(sh * math.sin(t * 91), sh * math.cos(t * 77))
    # glyph placements
    placed = []
    xx = -total / 2
    for i, c in enumerate(chars):
        s = 1.0 + 0.12 * (rng.random() - 0.4) * (1 if n > 1 else 0)
        r = (rng.random() - 0.5) * 0.18
        dy = (rng.random() - 0.5) * size * 0.08
        if t is not None:
            ti = t - i * 0.035
            if ti < 0:
                xx += ws[i] + tr
                continue
            s *= 0.3 + 0.7 * _out_back(ti / 0.16, 2.6)
        placed.append((c, xx, dy, s, r, ws[i]))
        xx += ws[i] + tr

    def glyphs(dx=0.0, dy0=0.0):
        ctx.new_path()
        for c, gx, dy, s, r, wc in placed:
            ctx.save()
            ctx.translate(gx + wc / 2 + dx, dy + dy0)
            ctx.rotate(r)
            ctx.scale(s, s)
            f.path(ctx, c, -wc / 2, cap / 2, size)
            ctx.restore()

    if depth > 0:
        steps = max(2, int(depth / 2))
        for k in range(steps, 0, -1):
            d = depth * k / steps
            glyphs(math.cos(depth_ang) * d, math.sin(depth_ang) * d)
            ctx.set_source_rgba(*inkc, 1.0)
            ctx.set_line_width(size * 0.3)
            ctx.stroke_preserve()
            ctx.fill()
    glyphs()
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_source_rgba(*inkc, 1.0)
    ctx.set_line_width(size * 0.3)
    ctx.stroke_preserve()
    ctx.set_source_rgba(*paper, 1.0)
    ctx.set_line_width(size * 0.17)
    ctx.stroke_preserve()
    ctx.set_source_rgba(*inkc, 1.0)
    ctx.fill()
    if alpha < 0.999:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(alpha)
    ctx.restore()
    hw, hh = total / 2 + size * 0.2, cap / 2 + size * 0.25
    c, s_ = abs(math.cos(rot)), abs(math.sin(rot))
    ex, ey = hw * c + hh * s_, hw * s_ + hh * c
    return (x - ex, y - ey, x + ex, y + ey)


# ================================================================== backgrounds / furniture
def burst_bg(ctx, cx, cy, rect, *, t=0.0, rays=48, inner=None, boil=12, bg=True):
    """Chick radial burst: black spikes converging on (cx, cy) around a white core; boils on twos"""
    x0, y0, w, h = rect
    R = math.hypot(max(cx - x0, x0 + w - cx), max(cy - y0, y0 + h - cy)) * 1.05
    r0 = inner if inner is not None else min(w, h) * 0.2
    fr = int(t * boil) if boil else 0
    rng = np.random.default_rng(1000 + fr)
    ctx.save()
    ctx.rectangle(x0, y0, w, h)
    ctx.clip()
    if bg:
        ctx.set_source_rgba(*_paper(), 1.0)
        ctx.paint()
    ctx.set_source_rgba(*_inkc(), 1.0)
    rot = t * 0.05
    for k in range(rays):
        a = 2 * math.pi * (k + 0.35 * (_h01(k, 1) - 0.5)) / rays + rot
        wid = (0.35 + 0.6 * _h01(k, 2)) * math.pi / rays
        rin = r0 * (0.75 + 0.6 * _h01(k, 3)) * (0.92 + 0.16 * rng.random())
        ctx.move_to(cx + math.cos(a) * rin, cy + math.sin(a) * rin)
        ctx.line_to(cx + math.cos(a - wid) * R, cy + math.sin(a - wid) * R)
        ctx.line_to(cx + math.cos(a + wid) * R, cy + math.sin(a + wid) * R)
        ctx.close_path()
    ctx.fill()
    ctx.restore()


def speed_lines(ctx, rect, *, focus=None, angle=0.0, t=0.0, n=90, inner=None, boil=12, weight=1.0):
    x0, y0, w, h = rect
    ctx.save()
    ctx.rectangle(x0, y0, w, h)
    ctx.clip()
    inkc = _inkc()
    ctx.set_source_rgba(*inkc, 1.0)
    fr = int(t * boil) if boil else 0
    if focus is not None:
        fx, fy = focus
        R = math.hypot(w, h) * 1.2
        r0 = inner if inner is not None else min(w, h) * 0.28
        for k in range(n):
            a = 2 * math.pi * (k + _h01(k, fr, 1)) / n
            rin = r0 * (0.8 + 0.7 * _h01(k, fr, 2))
            wd = (0.004 + 0.012 * _h01(k, 3)) * weight
            ctx.move_to(fx + math.cos(a) * rin, fy + math.sin(a) * rin)
            ctx.line_to(fx + math.cos(a - wd) * R, fy + math.sin(a - wd) * R)
            ctx.line_to(fx + math.cos(a + wd) * R, fy + math.sin(a + wd) * R)
            ctx.close_path()
        ctx.fill()
    else:
        ca, sa = math.cos(angle), math.sin(angle)
        diag = math.hypot(w, h)
        mx, my = x0 + w / 2, y0 + h / 2
        for k in range(n):
            off = (k / n - 0.5) * diag + (_h01(k, 5) - 0.5) * diag / n
            ln = diag * (0.15 + 0.35 * _h01(k, 6))
            sp = diag * (1.5 + _h01(k, 7))
            pos = ((t * sp + _h01(k, 8) * diag * 2) % (diag * 2)) - diag
            px, py = mx - sa * off + ca * pos, my + ca * off + sa * pos
            ctx.move_to(px, py)
            ctx.line_to(px - ca * ln, py - sa * ln)
            ctx.set_line_width((1.0 + 3.0 * _h01(k, 9)) * weight)
            ctx.stroke()
    ctx.restore()


def _drop(ctx, x, y, rr, a, lw, paper, inkc, rot=0.0):
    """a teardrop (point up) centred on its round bottom at (x, y): paper fill, ink outline, highlight"""
    if a <= 0.01:
        return
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    ctx.new_path()
    ctx.move_to(0, -rr * 2.3)
    ctx.curve_to(rr * 0.25, -rr * 1.3, rr, -rr * 0.5, rr, 0)
    ctx.arc(0, 0, rr, 0, math.pi)
    ctx.curve_to(-rr, -rr * 0.5, -rr * 0.25, -rr * 1.3, 0, -rr * 2.3)
    ctx.close_path()
    ctx.set_source_rgba(*paper, a)
    ctx.fill_preserve()
    ctx.set_source_rgba(*inkc, a)
    ctx.set_line_width(lw * 0.8)
    ctx.stroke()
    ctx.new_path()
    ctx.arc(-rr * 0.38, -rr * 0.15, rr * 0.2, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()


def emanata(ctx, x, y, r, kind="shock", *, t=0.0, ang=-math.pi / 2, seed=0):
    inkc, paper = _inkc(), _paper()
    ctx.save()
    ctx.set_source_rgba(*inkc, 1.0)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    lw = max(1.5, r * 0.07)
    if kind == "shock":
        pulse = 1.0 + 0.12 * math.sin(t * 25)
        for k in range(9):
            a = ang + (k - 4) * 0.3
            r0 = r * (1.0 + 0.1 * _h01(k, seed))
            r1 = r0 + r * (0.45 + 0.25 * _h01(k, seed, 2)) * pulse
            ctx.move_to(x + math.cos(a) * r0, y + math.sin(a) * r0)
            ctx.line_to(x + math.cos(a) * r1, y + math.sin(a) * r1)
            ctx.set_line_width(lw * (1.4 - 0.6 * abs(k - 4) / 4))
            ctx.stroke()
    elif kind == "sweat":
        # a big drop sliding down the temple + two little ones flicking off
        per = 1.6
        ph = ((t + _h01(seed, 1) * per) % per) / per
        _drop(ctx, x, y + ph * r * 0.35, r * 0.2, 1.0 if ph < 0.85 else (1 - ph) / 0.15, lw, paper, inkc)
        for k in range(2):
            pk = ((t * 1.3 + k * 0.5 + _h01(seed, k, 2)) % 1.0)
            side = 1 if k == 0 else -1
            dx = side * r * (0.35 + pk * 0.5)
            dy = -r * 0.25 + pk * pk * r * 0.9 - pk * r * 0.4
            _drop(ctx, x + dx, y + dy, r * 0.09, 1 - pk, lw * 0.8, paper, inkc, rot=side * (0.5 + pk))
    elif kind == "tears":
        # a streaming tear down the cheek from the eye at (x, y), drops falling from its end
        L = r * 1.1
        wob = math.sin(t * 5 + seed) * r * 0.03
        ctx.new_path()
        n = 16
        left, right = [], []
        for j in range(n + 1):
            v = j / n
            cxj = x + math.sin(v * 2.4) * r * 0.08 + wob * v
            wj = r * (0.05 + 0.1 * v)
            left.append((cxj - wj, y + v * L))
            right.append((cxj + wj, y + v * L))
        ctx.move_to(*left[0])
        for p_ in left[1:]:
            ctx.line_to(*p_)
        ctx.arc(x + math.sin(2.4) * r * 0.08 + wob, y + L, r * 0.15, math.pi, 0)
        for p_ in reversed(right):
            ctx.line_to(*p_)
        ctx.close_path()
        ctx.set_source_rgba(*paper, 1.0)
        ctx.fill_preserve()
        ctx.set_source_rgba(*inkc, 1.0)
        ctx.set_line_width(lw * 0.7)
        ctx.stroke()
        for k in range(2):
            pk = (t * 1.1 + k * 0.5 + _h01(seed, k)) % 1.0
            _drop(ctx, x + math.sin(2.4) * r * 0.08 + wob, y + L + r * 0.3 + pk * pk * r * 1.2, r * 0.1,
                  1 - pk, lw * 0.7, paper, inkc)
    elif kind == "stink":
        for k in range(3):
            xx = x + (k - 1) * r * 0.42
            ph = t * 2.2 + k * 1.3 + seed
            ctx.new_path()
            ctx.move_to(xx, y)
            for j in range(1, 25):
                v = j / 24
                ctx.line_to(xx + math.sin(v * 9 - ph * 2.5) * r * 0.1 * (0.5 + v), y - v * r * 1.3 - (ph % 1) * 0)
            ctx.set_line_width(lw * (1.1 - 0.2 * k))
            ctx.stroke()
    elif kind == "sparkle":
        for k in range(4):
            a = 2 * math.pi * _h01(k, seed)
            d = r * (0.3 + 0.8 * _h01(k, seed, 2))
            sx, sy = x + math.cos(a) * d, y + math.sin(a) * d
            tw = 0.5 + 0.5 * math.sin(t * 7 + k * 2.1)
            s = r * (0.14 + 0.18 * _h01(k, seed, 3)) * (0.4 + 0.6 * tw)
            ctx.new_path()
            for j in range(4):
                aa = j * math.pi / 2
                ctx.move_to(sx, sy) if j == 0 else None
                ctx.line_to(sx + math.cos(aa) * s, sy + math.sin(aa) * s)
                ctx.line_to(sx + math.cos(aa + math.pi / 4) * s * 0.18, sy + math.sin(aa + math.pi / 4) * s * 0.18)
            ctx.close_path()
            ctx.set_source_rgba(*paper, 1.0)
            ctx.fill_preserve()
            ctx.set_source_rgba(*inkc, 1.0)
            ctx.set_line_width(lw * 0.6)
            ctx.stroke()
    elif kind == "blush":
        for k in range(5):
            xx = x + (k - 2) * r * 0.16
            ctx.move_to(xx - r * 0.05, y + r * 0.12)
            ctx.line_to(xx + r * 0.07, y - r * 0.12)
        ctx.set_line_width(lw * 0.7)
        ctx.stroke()
    elif kind == "anger":
        # the manga vein mark: four brackets bowing in towards the centre, throbbing
        s = r * 0.4 * (1 + 0.12 * max(0.0, math.sin(t * 16)))
        for q in range(4):
            a = q * math.pi / 2 + math.pi / 4
            px, py = x + math.cos(a) * s * 0.85, y + math.sin(a) * s * 0.85
            ctx.new_path()
            ctx.arc(px, py, s * 0.5, a + math.pi * 0.62, a + math.pi * 1.38)
            ctx.set_line_width(lw * 1.5)
            ctx.stroke()
    elif kind in ("question", "exclaim"):
        f = _Face(FONT_LETTER)
        ch = "?" if kind == "question" else "!"
        s = r * (1.0 + 0.15 * math.sin(t * 9))
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(0.12 * math.sin(t * 5))
        ctx.new_path()
        f.path(ctx, ch, -f.width(ch, s) / 2, s * f.cap / 2, s)
        ctx.set_source_rgba(*paper, 1.0)
        ctx.set_line_width(s * 0.12)
        ctx.stroke_preserve()
        ctx.set_source_rgba(*inkc, 1.0)
        ctx.fill()
        ctx.restore()
    elif kind == "dizzy":
        ctx.new_path()
        for j in range(60):
            a = j * 0.35 + t * 4
            d = r * 0.1 + j * r * 0.012
            px, py = x + math.cos(a) * d, y + math.sin(a) * d * 0.5
            ctx.move_to(px, py) if j == 0 else ctx.line_to(px, py)
        ctx.set_line_width(lw * 0.8)
        ctx.stroke()
    ctx.restore()


def panel(ctx, rect, *, border=6.0, fill=True):
    x0, y0, w, h = rect
    ctx.save()
    ctx.rectangle(x0, y0, w, h)
    if fill:
        ctx.set_source_rgba(*_paper(), 1.0)
        ctx.fill_preserve()
    ctx.set_source_rgba(*_inkc(), 1.0)
    ctx.set_line_width(border)
    ctx.set_line_join(cairo.LINE_JOIN_MITER)
    ctx.stroke()
    ctx.restore()


PAGE_BG = (0x2B / 255, 0x2B / 255, 0x2D / 255)


class Page:
    """portrait tract page (the scans: 935 x 1210) with n wide panels on the dark page and a page number.
    (x, y) = page top-left, h = page height in design units."""

    def __init__(self, x, y, h, *, n=3, number=None, aspect=935 / 1210):
        self.x, self.y, self.h, self.w = x, y, h, h * aspect
        self.number = number
        k = self.w / 935.0
        mx0, mx1, my0, my1, g = 72 * k, 35 * k, 48 * k, 80 * k, 18 * k
        ph = (h - my0 - my1 - g * (n - 1)) / n
        self.panels = [(x + mx0, y + my0 + i * (ph + g), self.w - mx0 - mx1, ph) for i in range(n)]
        self.rect = (x, y, self.w, h)
        self.k = k

    def draw(self, ctx, *, panels=True, border=None):
        x, y, w, h = self.rect
        ctx.save()
        ctx.rectangle(x, y, w, h)
        ctx.set_source_rgb(*PAGE_BG)
        ctx.fill()
        if panels:
            for r in self.panels:
                panel(ctx, r, border=border or 3.0 * self.k)
        if self.number is not None:
            f = _Face(FONT_BODY)
            s = 20 * self.k
            txt = str(self.number)
            odd = self.number % 2 == 1
            px = x + w - 35 * self.k - f.width(txt, s) if odd else x + 35 * self.k
            ctx.new_path()
            f.path(ctx, txt, px, y + h - 32 * self.k, s)
            ctx.set_source_rgb(0.92, 0.92, 0.92)
            ctx.fill()
        ctx.restore()


def frame(rect, *, pad=0.0):
    """vx.camera.Camera that fits panel rect (x, y, w, h) to the 1920x1080 screen (cover, centred)"""
    x, y, w, h = rect
    z = min(W / (w * (1 + pad)), H / (h * (1 + pad)))
    return Camera(x + w / 2, y + h / 2, z)


def glide(rect_a, rect_b, u, *, pad=0.0, lift=0.12):
    """eased camera move between two panels (pulls back by `lift` mid-way, like reading)"""
    a, b = frame(rect_a, pad=pad), frame(rect_b, pad=pad)
    e = _smooth(_clamp(u))
    e = e * e * (3 - 2 * e)
    z = math.exp(math.log(a.zoom) + (math.log(b.zoom) - math.log(a.zoom)) * e) * (1 - lift * math.sin(math.pi * e))
    return Camera(a.x + (b.x - a.x) * e, a.y + (b.y - a.y) * e, z)


# ================================================================== FLAGS FLAGS FLAGS
def flags_flood(ctx, rect, amount, *, t=0.0, size=34, avoid=None, pulse=0.0, seed=0, word="FLAGS"):
    """kinetic 'FLAGS FLAGS FLAGS' wall (page 25). amount 0..1:
    0-0.6 rows drop in and stack bottom-up (each row streaming sideways), 0.45-0.9 an overprinted second
    layer slides in between them (white knock-out halo, like the scan), 0.8-1 a third, bigger layer —
    at 1.0 the rect is a solid wall of type. Words reflow around `avoid` (never touch the cloth)."""
    amount = _clamp(amount)
    if amount <= 0:
        return None
    x0, y0, w, h = rect
    f = _Face(FONT_LETTER)
    inkc, paper = _inkc(), _paper()
    av = _geom_avoid(avoid, size * 0.3)
    ctx.save()
    ctx.rectangle(x0, y0, w, h)
    ctx.clip()
    if av is not None:
        _clip_out(ctx, av)

    def blocked(yt, yb):
        if av is None:
            return []
        from shapely.geometry import box
        band = box(x0 - 10, yt, x0 + w + 10, yb)
        g = band.intersection(av)
        if g.is_empty:
            return []
        gs = getattr(g, "geoms", [g])
        return [(q.bounds[0], q.bounds[2]) for q in gs if not q.is_empty]

    def row(yb, sz, shift, alpha, halo, jolt=1.0):
        ww = f.width(word + " ", sz)
        blk = blocked(yb - f.cap * sz - sz * 0.1, yb + sz * 0.12)
        start = x0 - ww + (shift % ww)
        ctx.new_path()
        xx = start
        while xx < x0 + w + ww:
            wd = f.width(word, sz)
            if not any(a < xx + wd and xx < b for a, b in blk):
                if jolt != 1.0:
                    ctx.save()
                    ctx.translate(xx + wd / 2, yb - f.cap * sz / 2)
                    ctx.scale(jolt, jolt)
                    ctx.translate(-(xx + wd / 2), -(yb - f.cap * sz / 2))
                    f.path(ctx, word, xx, yb, sz)
                    ctx.restore()
                else:
                    f.path(ctx, word, xx, yb, sz)
            xx += ww
        if halo:
            ctx.set_source_rgba(*paper, alpha)
            ctx.set_line_width(sz * 0.2)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            ctx.stroke_preserve()
        ctx.set_source_rgba(*inkc, alpha)
        ctx.fill()

    rh = size * 1.02
    nrows = int(math.ceil(h / rh)) + 1
    jolt = 1.0 + 0.05 * pulse
    # layer 1: rows drop in, stacking from the bottom up
    for k in range(nrows):
        a0 = 0.6 * k / nrows
        u = (amount - a0) / 0.07
        if u <= 0:
            continue
        rest = y0 + h - k * rh
        yb = rest - (1 - _out_back(_clamp(u), 1.4)) * (rest - y0 + size)
        dirn = 1 if (k + seed) % 2 == 0 else -1
        spd = size * (1.5 + 1.2 * _h01(k, seed))
        row(yb, size, dirn * t * spd + _h01(k, seed, 1) * 400, 1.0, False, jolt)
    # layer 2: overprinted rows between the rows, sliding in from the sides
    if amount > 0.45:
        for k in range(nrows):
            a0 = 0.45 + 0.4 * _h01(k, seed, 5)
            u = (amount - a0) / 0.08
            if u <= 0:
                continue
            yb = y0 + h - (k + 0.5) * rh
            dirn = -1 if (k + seed) % 2 == 0 else 1
            slide = (1 - _smooth(_clamp(u))) * w * 1.2 * dirn
            spd = size * (2.0 + 1.5 * _h01(k, seed, 6))
            row(yb, size * 1.0, dirn * t * spd + slide + 137 * k, 1.0, True, jolt)
    # layer 3: big FLAGS rows slam in over everything
    if amount > 0.8:
        big = size * 2.3
        n3 = int(math.ceil(h / (big * 1.1)))
        for k in range(n3):
            a0 = 0.8 + 0.18 * k / max(1, n3)
            u = (amount - a0) / 0.05
            if u <= 0:
                continue
            yb = y0 + (k + 0.85) * big * 1.1
            s = 1.0 + 0.5 * (1 - _smooth(_clamp(u)))
            spd = big * (1.0 + _h01(k, seed, 9))
            ctx.save()
            cxm, cym = x0 + w / 2, yb - big * 0.35
            ctx.translate(cxm, cym)
            ctx.scale(s, s)
            ctx.translate(-cxm, -cym)
            row(yb, big, (1 if k % 2 else -1) * t * spd + 71 * k, _clamp(u * 2), True, jolt)
            ctx.restore()
    ctx.restore()
    return None


# ================================================================== the absolute guard
def cloth_avoid(flag_rgba, s=1.0, *, pole=16.0, grow=6.0, min_area=400.0, poles=True):
    """flag layer (premultiplied rgba pixels at scale s) -> avoid polygons in design units: convex hulls of
    each cloth (thin poles are opened away first, pole = design px) plus, with poles=True, a thin rotated
    rectangle hugging each pole (so the hull of an L-shaped flag doesn't block a whole triangle)."""
    import cv2
    a0 = (flag_rgba[..., 3] > 0.02).astype(np.uint8)
    k = max(1, int(round(pole * s)))
    a = cv2.morphologyEx(a0, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) if k > 1 else a0

    def grown(h, g):
        ctr = h.mean(0)
        d = h - ctr
        n = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
        return h + d / n * g

    out = []
    for c in cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) < min_area * s * s:
            continue
        h = cv2.convexHull(c)[:, 0, :].astype(np.float64) / s
        out.append([tuple(p) for p in (grown(h, grow) if grow else h)])
    if poles:
        rest = cv2.subtract(a0, cv2.dilate(a, np.ones((3, 3), np.uint8)))
        for c in cv2.findContours(rest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) < 20 * s * s:
                continue
            box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64) / s
            out.append([tuple(p) for p in grown(box, grow * 0.5)])
    return out


def track_avoid(track, frames, *, pole=True, grow=6.0):
    """vx.flag.FlagTrack -> avoid shapes (design units) straight from the simulation: the convex hull of
    the cloth over `frames` (an index, float, or iterable of them) + the pole. Cheap (no rendering).
    Stable balloons next to a moving flag:  avoid=lambda T: comic.track_avoid(track, (T - t0) * 24)"""
    from shapely.geometry import MultiPoint, LineString
    fr = np.atleast_1d(np.asarray(frames, np.float64))
    pts, out = [], []
    L = track.meta["pole"]
    pw = track.meta.get("pole_w", 7.0)
    for f in fr:
        V, (x, y, a) = track._frame(float(f))
        pts.append(np.asarray(V)[..., :2].reshape(-1, 2))
        if pole:
            out.append(LineString([(x, y), (x + math.sin(a) * L, y - math.cos(a) * L)]).buffer(pw / 2 + grow * 0.5, 4))
    out.insert(0, MultiPoint(np.concatenate(pts)).convex_hull.buffer(grow, 4))
    return out


def protect(letter_rgba, flag_rgba, grow=1):
    """knock the lettering layer out wherever flag cloth/pole is (premultiplied float arrays)"""
    a = flag_rgba[..., 3]
    m = (a > 0.004).astype(np.float32)
    if grow:
        import cv2
        m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return letter_rgba * (1.0 - m)[..., None]

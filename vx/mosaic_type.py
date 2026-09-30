"""vx.mosaic_type - TITULI: Latin inscriptions set in tesserae (owner: Tessellator). Re-exported by vx.mosaic.

Fonts (OFL, Google Fonts; `python -m vx.mosaic_type` builds + installs them into ~/.local/share/fonts/vx_tessellator
and keeps copies + licences in assets/fonts/; run once per machine):
  'roman'        VX Cinzel        Cinzel wght 700  - Roman square capitals (Trajan column), the default
  'roman_black'  VX Cinzel Black  Cinzel wght 900  - heavier, reads best at small cap heights (40-70 px)
  'roman_light'  VX Cinzel Light  Cinzel wght 500
  'byzantine'    Marcellus        flared capitals with a late-antique / Byzantine feel
  'initial'      Cinzel Decorative Bold - ornamented initials

Markup inside the text:
  [VXL]  overline over an abbreviation (e.g. [VXL] for VEXILLVM, [MAG] for MAGISTER)
  ·      interpunct (a diamond tessera); with sep='·' every space becomes one
  +      rosette (six petals)     *  square tessera     ~  hedera (ivy leaf)      \n  new line
  (no crosses or Christian nomina sacra anywhere: the theology of the film is Vexillomantic)
latin=True: upper-case, U->V, J->I, W->VV (so 'Explicit opus schismaticum' -> EXPLICIT OPVS SCHISMATICVM).
"""
import math
import shutil
import subprocess
from pathlib import Path

import cairo
import numpy as np

from .config import ROOT

ASSETS = ROOT / "assets" / "fonts"
USER = Path.home() / ".local/share/fonts/vx_tessellator"
FACES = {"roman": "VX Cinzel", "roman_black": "VX Cinzel Black", "roman_light": "VX Cinzel Light",
         "byzantine": "Marcellus", "initial": "Cinzel Decorative"}
_FALLBACK = "P052"
_INSTANCES = [("Cinzel[wght].ttf", "VX Cinzel", {"wght": 700}), ("Cinzel[wght].ttf", "VX Cinzel Black", {"wght": 900}),
              ("Cinzel[wght].ttf", "VX Cinzel Light", {"wght": 500})]
_STATIC = ["Marcellus-Regular.ttf", "CinzelDecorative-Bold.ttf"]
_GF = "https://github.com/google/fonts/raw/main/ofl/"
_SRC = {"Cinzel[wght].ttf": "cinzel/Cinzel%5Bwght%5D.ttf", "OFL-Cinzel.txt": "cinzel/OFL.txt",
        "CinzelDecorative-Bold.ttf": "cinzeldecorative/CinzelDecorative-Bold.ttf",
        "OFL-CinzelDecorative.txt": "cinzeldecorative/OFL.txt",
        "Marcellus-Regular.ttf": "marcellus/Marcellus-Regular.ttf", "OFL-Marcellus.txt": "marcellus/OFL.txt"}


def build_fonts():
    """download (if needed) + instantiate + install the titulus faces. Idempotent."""
    import urllib.request
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    from .comic_fonts import _rename
    src = ASSETS / "src"
    src.mkdir(parents=True, exist_ok=True)
    USER.mkdir(parents=True, exist_ok=True)
    for fn, rel in _SRC.items():
        dst = (ASSETS / fn) if fn.startswith("OFL") else (src / fn)
        if not dst.exists():
            urllib.request.urlretrieve(_GF + rel, dst)
    out = []
    for fn, family, axes in _INSTANCES:
        f = instantiateVariableFont(TTFont(src / fn), axes)
        _rename(f, family)
        p = ASSETS / (family.replace(" ", "") + ".ttf")
        f.save(p)
        out.append(p)
    for fn in _STATIC:
        shutil.copy2(src / fn, ASSETS / fn)
        out.append(ASSETS / fn)
    for p in out:
        shutil.copy2(p, USER / p.name)
    subprocess.run(["fc-cache", "-f", str(USER)], check=False)
    return out


_AVAIL = {}


def face(name):
    """cairo family name for a titulus face key (falls back to URW P052 if the fonts are not installed)."""
    fam = FACES.get(name, name)
    if fam not in _AVAIL:
        try:
            out = subprocess.run(["fc-list", f":family={fam}", "family"], capture_output=True, text=True, timeout=10)
            _AVAIL[fam] = bool(out.stdout.strip())
        except Exception:
            _AVAIL[fam] = False
    return fam if _AVAIL[fam] else _FALLBACK


def latinize(s):
    s = s.upper().replace("U", "V").replace("J", "I").replace("W", "VV")
    return s


def _tokens(text, sep):
    """-> list of (kind, char, overline_group_id). kinds: 'ch', 'dot', 'rose', 'sq', 'leaf', 'nl', 'sp'."""
    toks = []
    grp = -1
    ng = 0
    for ch in text:
        if ch == "[":
            grp = ng
            ng += 1
            continue
        if ch == "]":
            grp = -1
            continue
        if ch == "\n":
            toks.append(("nl", ch, -1))
        elif ch in "·•":
            toks.append(("dot", ch, -1))
        elif ch == "+":
            toks.append(("rose", ch, grp))
        elif ch == "*":
            toks.append(("sq", ch, grp))
        elif ch == "~":
            toks.append(("leaf", ch, grp))
        elif ch == " ":
            toks.append(("dot", "·", -1) if sep else ("sp", ch, -1))
        else:
            toks.append(("ch", ch, grp))
    return toks


class Titulus:
    """A laid-out inscription. size = CAP HEIGHT in design units (60-120 reads well in tesserae; use fine tiles
    of ~0.085*size for the letters: Layout.flow(fine=<draw_fine mask>, fine_tile=0.085*size)).
    align: 'center' | 'left' | 'right' around x; y = BASELINE of the first line."""

    def __init__(self, text, x, y, size, color="#101A45", font="roman", align="center", tracking=0.14,
                 latin=True, sep=None, leading=1.45, dot_color=None, over_color=None):
        from .canvas import col
        self.text = latinize(text) if latin else text
        self.x, self.y, self.size = float(x), float(y), float(size)
        self.color = col(color)
        self.dot_color = col(dot_color) if dot_color is not None else self.color
        self.over_color = col(over_color) if over_color is not None else self.color
        self.font = face(font)
        self.align = align
        self.tracking = tracking
        self.leading = leading
        self.toks = _tokens(self.text, sep)
        self._layout()

    def _ctx(self):
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 4, 4)
        ctx = cairo.Context(surf)
        ctx.select_font_face(self.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(100.0)
        cap = -ctx.text_extents("H").y_bearing / 100.0
        self.fsize = self.size / max(cap, 0.3)
        ctx.set_font_size(self.fsize)
        return ctx

    def _layout(self):
        ctx = self._ctx()
        S = self.size
        tr = self.tracking * S
        lines = [[]]
        for t in self.toks:
            if t[0] == "nl":
                lines.append([])
            else:
                lines[-1].append(t)
        self.glyphs = []       # dicts: kind, ch, x, y (baseline-left), w, grp, line, idx
        idx = 0
        for li, toks in enumerate(lines):
            xs = []
            cx = 0.0
            for kind, ch, grp in toks:
                if kind == "ch":
                    ext = ctx.text_extents(ch)
                    w = ext.x_advance
                elif kind == "sp":
                    w = 0.45 * S
                elif kind == "dot":
                    w = 0.62 * S
                elif kind == "rose":
                    w = 0.95 * S
                elif kind == "sq":
                    w = 0.5 * S
                else:
                    w = 1.0 * S
                xs.append((kind, ch, grp, cx, w))
                cx += w + tr
            total = cx - tr if xs else 0.0
            if self.align == "center":
                ox = self.x - total / 2
            elif self.align == "right":
                ox = self.x - total
            else:
                ox = self.x
            by = self.y + li * self.leading * S * 1.0 / 0.72 * 0.72
            by = self.y + li * self.leading * S * 1.3
            for kind, ch, grp, gx, w in xs:
                g = dict(kind=kind, ch=ch, grp=grp, x=ox + gx, y=by, w=w, line=li, idx=-1)
                if kind not in ("sp",):
                    g["idx"] = idx
                    idx += 1
                self.glyphs.append(g)
        self.n = idx
        self.width = max((g["x"] + g["w"] for g in self.glyphs), default=self.x) - min(
            (g["x"] for g in self.glyphs), default=self.x)
        self.boxes = []
        for g in self.glyphs:
            if g["idx"] < 0:
                continue
            self.boxes.append((g["x"], g["y"] - 1.3 * S, g["x"] + g["w"], g["y"] + 0.15 * S))

    # ---------------------------------------------------------------- drawing
    def _glyph(self, ctx, g, fill=True):
        S = self.size
        k = g["kind"]
        if k == "ch":
            ctx.move_to(g["x"], g["y"])
            ctx.text_path(g["ch"])
        elif k == "dot":
            cx, cy, r = g["x"] + g["w"] / 2, g["y"] - 0.5 * S, 0.13 * S
            ctx.move_to(cx, cy - r)
            ctx.line_to(cx + r, cy)
            ctx.line_to(cx, cy + r)
            ctx.line_to(cx - r, cy)
            ctx.close_path()
        elif k == "rose":
            cx, cy = g["x"] + g["w"] / 2, g["y"] - 0.5 * S
            for j in range(6):                 # six petals round a boss: a rosette (hexafoil)
                a = j * math.pi / 3 + math.pi / 6
                ctx.save()
                ctx.translate(cx + 0.23 * S * math.cos(a), cy + 0.23 * S * math.sin(a))
                ctx.rotate(a)
                ctx.scale(0.2 * S, 0.1 * S)
                ctx.new_sub_path()
                ctx.arc(0, 0, 1, 0, 2 * math.pi)
                ctx.restore()
            ctx.new_sub_path()
            ctx.arc(cx, cy, 0.09 * S, 0, 2 * math.pi)
        elif k == "sq":
            cx, cy, r = g["x"] + g["w"] / 2, g["y"] - 0.5 * S, 0.12 * S
            ctx.rectangle(cx - r, cy - r, 2 * r, 2 * r)
        elif k == "leaf":
            cx, cy = g["x"] + g["w"] / 2, g["y"] - 0.45 * S
            r = 0.36 * S
            ctx.move_to(cx, cy + r * 1.1)
            ctx.curve_to(cx - r * 1.6, cy - r * 0.2, cx - r * 0.5, cy - r * 1.4, cx, cy - r * 0.5)
            ctx.curve_to(cx + r * 0.5, cy - r * 1.4, cx + r * 1.6, cy - r * 0.2, cx, cy + r * 1.1)
            ctx.close_path()
            ctx.move_to(cx - 0.05 * S, cy - r * 0.6)
            ctx.curve_to(cx - 0.3 * S, cy - r * 1.4, cx - 0.1 * S, cy - r * 1.9, cx + 0.25 * S, cy - r * 2.0)
            ctx.line_to(cx + 0.25 * S, cy - r * 1.8)
            ctx.curve_to(cx + 0.02 * S, cy - r * 1.7, cx - 0.12 * S, cy - r * 1.3, cx + 0.05 * S, cy - r * 0.6)
            ctx.close_path()

    def _overlines(self, ctx):
        S = self.size
        groups = {}
        for g in self.glyphs:
            if g["grp"] >= 0:
                groups.setdefault((g["grp"], g["line"]), []).append(g)
        for gs in groups.values():
            x0 = min(g["x"] for g in gs) - 0.08 * S
            x1 = max(g["x"] + g["w"] for g in gs) + 0.08 * S
            y = gs[0]["y"] - 1.24 * S
            t = 0.1 * S
            ctx.rectangle(x0, y - t / 2, x1 - x0, t)
            # little end ticks (the scribe's flourish)
            ctx.rectangle(x0, y - t / 2, 0.08 * S, t * 1.9)
            ctx.rectangle(x1 - 0.08 * S, y - t / 2, 0.08 * S, t * 1.9)

    def draw(self, ctx, alpha=1.0, upto=None):
        """draw the inscription in its colours. upto: only glyphs with idx < upto (letter-by-letter reveals)."""
        ctx.save()
        ctx.select_font_face(self.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(self.fsize)
        for g in self.glyphs:
            if g["idx"] < 0 or (upto is not None and g["idx"] >= upto):
                continue
            c = self.dot_color if g["kind"] in ("dot", "leaf", "rose", "sq") else self.color
            ctx.set_source_rgba(c[0], c[1], c[2], alpha)
            ctx.new_path()
            self._glyph(ctx, g)
            ctx.fill()
        ctx.set_source_rgba(*self.over_color, alpha)
        ctx.new_path()
        self._overlines(ctx)
        ctx.fill()
        ctx.restore()

    def draw_mask(self, ctx, margin=0.0, color=(1, 1, 1)):
        """white letters (+ overlines, dots) grown by margin*size: the FINE-tile mask (margin ~0.35 so the
        halo course around each letter is fine too) or an exact letter mask (margin 0)."""
        ctx.save()
        ctx.select_font_face(self.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(self.fsize)
        ctx.set_source_rgb(*color)
        ctx.new_path()
        for g in self.glyphs:
            if g["idx"] >= 0:
                self._glyph(ctx, g)
        self._overlines(ctx)
        if margin > 0:
            ctx.set_line_width(2 * margin * self.size)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            ctx.stroke_preserve()
        ctx.fill()
        ctx.restore()

    def draw_fine(self, ctx, margin=0.38):
        self.draw_mask(ctx, margin)

    def draw_labels(self, ctx, margin=0.3):
        """glyph index labels for Layout.flow(groups=...): glyph i is drawn with red channel (i+1)/255
        (antialiasing off). Read back with labels_from(canvas_rgb) -> int image (-1 = none)."""
        ctx.save()
        ctx.set_antialias(cairo.ANTIALIAS_NONE)
        ctx.select_font_face(self.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(self.fsize)
        for g in self.glyphs:
            if g["idx"] < 0:
                continue
            v = min(g["idx"] + 1, 255) / 255.0
            ctx.set_source_rgb(v, 0, 0)
            ctx.new_path()
            self._glyph(ctx, g)
            if margin > 0:
                ctx.set_line_width(2 * margin * self.size)
                ctx.stroke_preserve()
            ctx.fill()
        ctx.restore()


def labels_from(rgb):
    """int labels from a draw_labels canvas (rgb float image): -1 where no glyph."""
    return (np.asarray(rgb)[..., 0] * 255 + 0.5).astype(np.int32) - 1


def titulus(ctx, text, x, y, size, color="#101A45", font="roman", align="center", tracking=0.14, latin=True,
            sep=None, **kw):
    """draw an inscription into a cartoon context and return the Titulus (for masks, labels, boxes)."""
    t = Titulus(text, x, y, size, color=color, font=font, align=align, tracking=tracking, latin=latin, sep=sep, **kw)
    t.draw(ctx)
    return t


if __name__ == "__main__":
    for p in build_fonts():
        print(p)

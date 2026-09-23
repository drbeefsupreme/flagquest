"""s01 helper: the IVG Swiss-grid title page (after the Flagazine #69 IVG cover, itself after
Mueller-Brockmann's 'Grid systems'). Real glyph outlines from Nimbus Sans Bold/Regular, shaped and kerned
by HarfBuzz, drawn as cairo paths so every letter can be animated individually.

draw_page(ctx, t, T_slam) draws the page in page-local design coordinates (0..1920 x 0..1080); the caller
translates it (drop-in / slide-out). `t` is global seconds. `letter_landings(T_slam)` returns the per-letter
impact times for the sound team.
"""
import math
from functools import lru_cache

import cairo
import numpy as np
import uharfbuzz as hb

from vx.config import W, H
from vx.ease import clamp, ease_out, ease_in_out

BOLD = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"
REG = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"

PAGE = (0.945, 0.735, 0.105)        # printed IVG yellow: a hair deeper/duller than the Flag, it is paper not cloth
PAGE_SH = (0.88, 0.665, 0.08)
INK = (0.075, 0.066, 0.062)
GRID = (1.0, 0.985, 0.93)

# ------------------------------------------------------------------ grid (landscape Swiss grid)
ML, MR, MT, MB = 92, 92, 58, 58
COLS, ROWS, GX, GY = 20, 10, 14, 14
CW = (W - ML - MR - GX * (COLS - 1)) / COLS
RH = (H - MT - MB - GY * (ROWS - 1)) / ROWS


def col_x(c):
    return ML + c * (CW + GX)


def row_y(r):
    return MT + r * (RH + GY)


# ------------------------------------------------------------------ fonts
class _Pen:
    def __init__(self, ctx, x, y, k):
        self.c, self.x, self.y, self.k = ctx, x, y, k

    def _p(self, p):
        return self.x + p[0] * self.k, self.y - p[1] * self.k

    def moveTo(self, p):
        self.c.move_to(*self._p(p))

    def lineTo(self, p):
        self.c.line_to(*self._p(p))

    def curveTo(self, a, b, p):
        self.c.curve_to(*self._p(a), *self._p(b), *self._p(p))

    def qCurveTo(self, *pts):
        # quadratic spline (not used by CFF, kept for completeness)
        cur = self.c.get_current_point()
        pts = [self._p(p) for p in pts]
        for i in range(len(pts) - 1):
            q = pts[i]
            e = pts[i + 1] if i == len(pts) - 2 else ((pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2)
            c1 = (cur[0] + 2 / 3 * (q[0] - cur[0]), cur[1] + 2 / 3 * (q[1] - cur[1]))
            c2 = (e[0] + 2 / 3 * (q[0] - e[0]), e[1] + 2 / 3 * (q[1] - e[1]))
            self.c.curve_to(*c1, *c2, *e)
            cur = e

    def closePath(self):
        self.c.close_path()

    endPath = closePath


@lru_cache(maxsize=4)
def _font(path):
    face = hb.Face(hb.Blob.from_file_path(path))
    f = hb.Font(face)
    return f, face.upem


@lru_cache(maxsize=64)
def shape(s, path=BOLD):
    """-> list of (gid, x_em, char) and total advance in em"""
    f, upem = _font(path)
    buf = hb.Buffer()
    buf.add_str(s)
    buf.guess_segment_properties()
    hb.shape(f, buf, {"kern": True, "liga": False})
    out, x = [], 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        out.append((info.codepoint, (x + pos.x_offset) / upem, s[info.cluster]))
        x += pos.x_advance
    return out, x / upem


def glyph_path(ctx, gid, x, y, size, path=BOLD):
    f, upem = _font(path)
    f.draw_glyph_with_pen(gid, _Pen(ctx, x, y, size / upem))


def text_run(ctx, s, x, y, size, path=BOLD, tracking=0.0, color=INK, alpha=1.0):
    gl, adv = shape(s, path)
    ctx.new_path()
    for k, (gid, gx, ch) in enumerate(gl):
        glyph_path(ctx, gid, x + (gx + tracking * k) * size, y, size, path)
    ctx.set_source_rgba(*color, alpha)
    ctx.fill()
    return (adv + tracking * (len(gl) - 1)) * size


def run_width(s, size, path=BOLD, tracking=0.0):
    gl, adv = shape(s, path)
    return (adv + tracking * (len(gl) - 1)) * size


# ------------------------------------------------------------------ layout
TITLE = "THE UNBABELING"
T_TRACK = -0.012
T_SIZE = (W - ML - MR + 4) / (shape(TITLE)[1] + T_TRACK * (len(TITLE) - 1))   # spans the grid exactly
T_BASE = row_y(3) - GY * 0.5 - 2                # baseline sits on the gutter under row 2
DE = "Die Entbabelung"
DE_SIZE = T_SIZE * 0.56
DE_BASE = row_y(6) - GY * 0.5 - 2
SMALL = 30
TINY = 25

RIGHT_COL = 11                                  # right text column group (like the cover)
EN_BODY = ["Being a true account of the Survey Flags,",
           "the recursive balkanization of every",
           "full-stack memeplex, and the fall of Babel"]
DE_BODY = ["Ein wahrer Bericht von den Vermessungsfahnen,",
           "der rekursiven Balkanisierung aller",
           "Memplexe und dem Fall von Babel"]
LEAD = RH / 2 + GY / 2                          # two text lines per grid row (baseline grid)
DE_T = 1.28                                     # the German title wipes in after the last letter lands


def _title_letters():
    gl, _ = shape(TITLE)
    out = []
    for k, (gid, gx, ch) in enumerate(gl):
        if ch == " ":
            continue
        out.append((gid, ML + (gx + T_TRACK * k) * T_SIZE, ch, k))
    return out


LETTERS = _title_letters()


def letter_times(T_slam):
    """(start, land) per title letter"""
    out = []
    for n, (gid, x, ch, k) in enumerate(LETTERS):
        t0 = T_slam + 0.16 + 0.062 * n + 0.012 * (n % 3)
        out.append((t0, t0 + 0.30))
    return out


def letter_landings(T_slam):
    return [(land, x + 0.5 * T_SIZE * 0.66) for (t0, land), (gid, x, ch, k) in zip(letter_times(T_slam), LETTERS)]


def grid_cells():
    cells = []
    for r in range(ROWS):
        for c in range(COLS):
            cells.append((c, r, col_x(c), row_y(r)))
    return cells


def _cell_outline(ctx, x, y, w, h, p):
    """draw the fraction p of a rectangle's perimeter: two strokes from the top-left corner, one going right
    then down, the other going down then right, meeting at the bottom-right corner"""
    if p <= 0:
        return
    half = (w + h) * p
    ctx.move_to(x, y)
    if half <= w:
        ctx.line_to(x + half, y)
    else:
        ctx.line_to(x + w, y)
        ctx.line_to(x + w, y + min(h, half - w))
    ctx.move_to(x, y)
    if half <= h:
        ctx.line_to(x, y + half)
    else:
        ctx.line_to(x, y + h)
        ctx.line_to(x + min(w, half - h), y + h)


def draw_page(ctx, t, T_slam, paper=None):
    """page-local drawing; returns nothing. `paper` optional pre-made cairo pattern for texture."""
    tt = t - T_slam
    # ---- paper
    ctx.save()
    B = 80                                   # bleed above/left/right (the page is bigger than the frame)
    ctx.rectangle(-B, -B, W + 2 * B, H + B)
    ctx.set_source_rgb(*PAGE)
    ctx.fill()
    g = cairo.RadialGradient(W * 0.42, H * 0.36, 80, W * 0.5, H * 0.5, W * 0.75)
    g.add_color_stop_rgba(0, 1, 0.93, 0.6, 0.10)
    g.add_color_stop_rgba(1, *PAGE_SH, 0.55)
    ctx.rectangle(-B, -B, W + 2 * B, H + B)
    ctx.set_source(g)
    ctx.fill()
    if paper is not None:
        ctx.set_source(paper)
        ctx.paint()
    ctx.restore()

    # ---- grid: every cell outlines itself, cascading diagonally from the top-left
    ctx.save()
    ctx.set_line_width(1.7)
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    for c, r, x, y in grid_cells():
        d = 0.012 * c + 0.022 * r + 0.01 * ((c * 7 + r * 3) % 5)
        p = ease_out(clamp((tt - d) / 0.32), 2.2)
        if p <= 0:
            continue
        _cell_outline(ctx, x, y, CW, RH, p)
    ctx.set_source_rgba(*GRID, 0.92)
    ctx.stroke()
    ctx.restore()

    # ---- the title, letter by letter, dropping in with overshoot and a motion smear
    for (t0, land), (gid, x, ch, k) in zip(letter_times(T_slam), LETTERS):
        if t < t0:
            continue
        u = clamp((t - t0) / (land - t0))
        if u < 1:
            # falling (accelerating) onto the baseline, with a true 180-degree-shutter motion blur
            n_s = 16
            for q in range(n_s):
                tq = t - (q / (n_s - 1)) * (1 / 48)
                uq = clamp((tq - t0) / (land - t0))
                if uq <= 0:
                    continue
                yq = T_BASE - (1 - uq * uq) * 520
                ctx.new_path()
                glyph_path(ctx, gid, x, yq, T_SIZE)
                ctx.set_source_rgba(*INK, min(1.0, uq * 3.0) / n_s * 1.8)
                ctx.fill()
            continue
        s_t = t - land                              # squash & settle after the impact
        y = T_BASE + 14 * math.exp(-9 * s_t) * math.sin(s_t * 30) * (1 if s_t < 0.6 else 0)
        alpha = 1.0
        sq = 1.0 + 0.10 * math.exp(-14 * s_t) * math.cos(s_t * 26)
        ctx.save()
        ctx.translate(x, T_BASE)
        ctx.scale(1.0 / math.sqrt(sq), sq)          # volume-preserving squash about the baseline
        ctx.translate(-x, -T_BASE)
        ctx.new_path()
        glyph_path(ctx, gid, x, y, T_SIZE)
        ctx.set_source_rgba(*INK, alpha)
        ctx.fill()
        ctx.restore()

    # ---- the German title: a clean left-to-right wipe along the grid
    wipe = ease_in_out(clamp((tt - DE_T) / 0.55), 3)
    if wipe > 0:
        wd = run_width(DE, DE_SIZE, tracking=-0.01)
        ctx.save()
        ctx.rectangle(ML - 10, DE_BASE - DE_SIZE, (wd + 20) * wipe, DE_SIZE * 1.3)
        ctx.clip()
        text_run(ctx, DE, ML + (1 - wipe) * -30, DE_BASE, DE_SIZE, tracking=-0.01)
        ctx.restore()

    # ---- small type on the baseline grid, fading up line by line
    def small(s, x, y, t_in, size=SMALL, path=REG, color=INK):
        a = ease_out(clamp((tt - t_in) / 0.4), 3)
        if a <= 0:
            return
        text_run(ctx, s, x, y + (1 - a) * 10, size, path=path, color=color, alpha=a)

    y1 = row_y(3) + SMALL * 0.95
    small("a Vexillomantic parable", ML, y1, 1.10)
    for i, s in enumerate(EN_BODY):
        small(s, col_x(RIGHT_COL), y1 + i * LEAD, 1.18 + 0.08 * i)
    y2 = row_y(6) + SMALL * 0.95
    small("ein vexillomantisches Gleichnis", ML, y2, DE_T + 0.45)
    for i, s in enumerate(DE_BODY):
        small(s, col_x(RIGHT_COL), y2 + i * LEAD, DE_T + 0.55 + 0.08 * i)
    small("IVG", col_x(COLS - 2), MT + TINY * 0.95, 2.05, TINY, BOLD)
    small("revision 7", ML, row_y(ROWS - 1) + RH - 12, 2.2, TINY)
    small("Expected release date: May 2577", col_x(RIGHT_COL), row_y(ROWS - 1) + RH - 12, 2.3, TINY)


def paper_texture(s, seed=7):
    """subtle paper grain + fibres as a premultiplied float RGBA image at scale s (cache it)."""
    import cv2
    w, h = int(round(W * s)), int(round(H * s))
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), max(0.5, 0.7 * s)) * 0.5
    lo = cv2.resize(rng.standard_normal((h // 40 + 2, w // 40 + 2)).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC) * 0.25
    # fibres: short random strokes
    fib = np.zeros((h, w), np.float32)
    for _ in range(int(900 * s * s) + 60):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        a = rng.uniform(0, math.pi)
        L = rng.uniform(6, 22) * s
        cv2.line(fib, (int(x), int(y)), (int(x + math.cos(a) * L), int(y + math.sin(a) * L)),
                 float(rng.uniform(0.3, 1.0)), 1, cv2.LINE_AA)
    v = n + lo - fib * 0.8
    v = np.clip(v, -2.5, 2.5)
    # light = white with alpha where v>0, dark = brown with alpha where v<0
    rgba = np.zeros((h, w, 4), np.float32)
    pos = np.clip(v, 0, None) * 0.022
    neg = np.clip(-v, 0, None) * 0.028
    a = pos + neg
    dark = np.array([0.35, 0.22, 0.02], np.float32)
    rgba[..., :3] = pos[..., None] * 1.0 + neg[..., None] * dark
    rgba[..., 3] = a
    return rgba

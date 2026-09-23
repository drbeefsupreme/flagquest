"""s08 helper: the IVG end card - the bookend of s01's title page (imports its grid, type and paper so the
two pages are the same printed object). The page slides DOWN into frame (s01's slid up and away), its grid
builds itself cell by cell from the bottom-right, FLAGS drops in letter by letter, the scripture follows.
"""
import math

import cairo

from vx import W, H, clamp, ease_out, ease_in_out
from scenes.s01_title import (PAGE, PAGE_SH, INK, GRID, ML, MR, MT, MB, COLS, ROWS, GX, GY, CW, RH, col_x, row_y,
                              grid_cells, _cell_outline, text_run, run_width, shape, glyph_path, BOLD, REG,
                              SMALL, TINY, LEAD, RIGHT_COL)

ITAL = "/usr/share/fonts/opentype/urw-base35/C059-Italic.otf"

T_CARD = 189.50                 # cue end_card: the page LANDS (paper slap on the music sting)
T_PAGE_IN = 0.46                # slide duration (starts T_CARD - T_PAGE_IN)
T_GRID = 189.22                 # the grid starts building while the page is still sliding in
T_LETTERS = 189.80
T_SCRIPT = 190.50
T_FADE0, T_FADE1 = 192.25, 193.0

WORD = "FLAGS"
F_TRACK = -0.02
F_SIZE = (W - ML - MR + 6) / (shape(WORD)[1] + F_TRACK * (len(WORD) - 1))
F_BASE = row_y(5) - GY * 0.5 - 2
SCRIPT = "Flags be. Flags are. Flags will. Flags."
S_SIZE = 58
S_BASE = row_y(6) + RH * 0.70
ATTR = "\u2014 I Vexillians"
A_SIZE = 36
A_BASE = row_y(7) + RH * 0.62
LICENSE = "Flag License 4 revision G \u2014 if you watched this you MUST distribute it"
RELEASE = "Expected release date: May 2577"


def _letters():
    gl, _ = shape(WORD)
    return [(gid, ML + (gx + F_TRACK * k) * F_SIZE, ch) for k, (gid, gx, ch) in enumerate(gl)]


LETTERS = _letters()


def letter_times():
    out = []
    for n in range(len(LETTERS)):
        t0 = T_LETTERS + 0.078 * n + 0.012 * (n % 2)
        out.append((t0, t0 + 0.30))
    return out


def page_offset(T):
    """vertical offset of the page (design px): -H (above frame) -> 0 with a small settle"""
    u = clamp((T - T_CARD + T_PAGE_IN) / T_PAGE_IN)
    if u >= 1:
        tt = T - T_CARD
        return 10.0 * math.exp(-9 * tt) * abs(math.sin(tt * 24))     # settles by bouncing DOWN only
    e = 1 - (1 - u) ** 3
    return -H * (1 - e)


def hits():
    """(t, kind, strength, desc) for the sound team"""
    out = [(T_CARD - T_PAGE_IN, "whoosh", 0.8, "end card: IVG page slides down into frame"),
           (T_CARD, "thunk", 0.8, "end card page lands (paper slap) - cue end_card")]
    for (t0, land), (gid, x, ch) in zip(letter_times(), LETTERS):
        out.append((land, "slam", 0.85 if ch == "F" else 0.7, f"end card letter '{ch}' lands"))
    out.append((T_SCRIPT, "shimmer", 0.35, "scripture line appears"))
    return out


def draw_page(ctx, T, paper=None):
    """page-local drawing (0..W x 0..H)."""
    tt = T - T_CARD
    # ---- paper (identical to s01's)
    ctx.save()
    ctx.rectangle(0, -60, W, H + 60)            # a little extra paper above: the landing bounce never shows a gap
    ctx.set_source_rgb(*PAGE)
    ctx.fill()
    g = cairo.RadialGradient(W * 0.42, H * 0.36, 80, W * 0.5, H * 0.5, W * 0.75)
    g.add_color_stop_rgba(0, 1, 0.93, 0.6, 0.10)
    g.add_color_stop_rgba(1, *PAGE_SH, 0.55)
    ctx.rectangle(0, -60, W, H + 60)
    ctx.set_source(g)
    ctx.fill()
    if paper is not None:
        ctx.set_source(paper)
        ctx.paint()
    ctx.restore()

    # ---- grid: every cell outlines itself, cascading from the bottom-right (mirror of s01)
    ctx.save()
    ctx.set_line_width(1.7)
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    ctx.new_path()
    gt = T - T_GRID
    for c, r, x, y in grid_cells():
        cc, rr = COLS - 1 - c, ROWS - 1 - r
        d = 0.016 * cc + 0.028 * rr + 0.012 * ((c * 7 + r * 3) % 5)
        p = ease_in_out(clamp((gt - d) / 0.40), 2)
        if p <= 0:
            continue
        ctx.save()
        ctx.translate(x + CW, y + RH)
        ctx.scale(-1, -1)
        _cell_outline(ctx, 0, 0, CW, RH, p)
        ctx.restore()
    ctx.set_source_rgba(*GRID, 0.92)
    ctx.stroke()
    ctx.restore()

    # ---- FLAGS, letter by letter: fall, smear, squash & settle
    for (t0, land), (gid, x, ch) in zip(letter_times(), LETTERS):
        if T < t0:
            continue
        u = clamp((T - t0) / (land - t0))
        if u < 1:
            y = F_BASE - (1 - u * u) * 640
            vel = 2 * u * 640 / (land - t0)
            alpha = min(1.0, u * 3.0)
            for gk in range(3, 0, -1):
                ctx.new_path()
                glyph_path(ctx, gid, x, y - gk * vel * 0.012, F_SIZE)
                ctx.set_source_rgba(*INK, alpha * 0.16 / gk)
                ctx.fill()
            sq = 1.0
        else:
            s_t = T - land
            y = F_BASE + 16 * math.exp(-9 * s_t) * math.sin(s_t * 30) * (1 if s_t < 0.6 else 0)
            alpha = 1.0
            sq = 1.0 + 0.10 * math.exp(-14 * s_t) * math.cos(s_t * 26)
        ctx.save()
        ctx.translate(x, F_BASE)
        ctx.scale(1.0 / math.sqrt(sq), sq)
        ctx.translate(-x, -F_BASE)
        ctx.new_path()
        glyph_path(ctx, gid, x, y, F_SIZE)
        ctx.set_source_rgba(*INK, alpha)
        ctx.fill()
        ctx.restore()

    # ---- scripture: phrase by phrase ("Flags be." / "Flags are." / ...), rising onto the baseline
    phrases = ["Flags be.", "Flags are.", "Flags will.", "Flags."]
    x = ML
    for i, ph in enumerate(phrases):
        a = ease_out(clamp((T - T_SCRIPT - 0.13 * i) / 0.38), 3)
        s = ph + (" " if i < len(phrases) - 1 else "")
        wdt = run_width(s, S_SIZE, path=ITAL)
        if a > 0:
            text_run(ctx, ph, x, S_BASE + (1 - a) * 12, S_SIZE, path=ITAL, color=INK, alpha=a)
        x += wdt
    a = ease_out(clamp((T - T_SCRIPT - 0.62) / 0.4), 3)
    if a > 0:
        text_run(ctx, ATTR, ML + (1 - a) * -18, A_BASE, A_SIZE, path=ITAL, color=INK, alpha=a)

    # ---- small print on the baseline grid
    def small(s, x, y, t_in, size=TINY, path=REG):
        a = ease_out(clamp((T - t_in) / 0.35), 3)
        if a > 0:
            text_run(ctx, s, x, y + (1 - a) * 8, size, path=path, color=INK, alpha=a)

    small("IVG", col_x(COLS - 2), MT + TINY * 0.95, T_CARD + 0.55, TINY, BOLD)
    small(LICENSE, ML, row_y(ROWS - 1) + RH - 12, T_CARD + 1.3)
    small(RELEASE, col_x(RIGHT_COL + 3), row_y(ROWS - 1) + RH - 12, T_CARD + 1.4)

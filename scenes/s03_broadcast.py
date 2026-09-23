"""s03 broadcast: what the TV station transmits (signal space: 800 x 600 units, 4:3).

President Jaguar at a plain navy podium, a row of PLAIN YELLOW Survey Flags behind him (the gag),
news graphics (BREAKING lower third, ticker, LIVE bug), fist slam, the Act unrolling off the podium,
a giant-pen signature flourish, the conspiratorial lean-in + wink, the pricecoin into the breast pocket.
"""
import math

import cairo
import numpy as np

from vx import (Canvas, set_color, rrect, circle, ellipse, poly, smooth, lin_grad, rad_grad, text, text_width,
                mix, shade, col, clamp, lerp, seg, ease_in_out, ease_out, ease_in, ease_out_back, smoothstep, pulse,
                fbm1, noise1, hash01)
from vx.chars import Character
from vx.flag import Flag, FlagTrack

SW, SH = 800, 600
NARROW = "Nimbus Sans Narrow"
SANS = "Nimbus Sans"

TICKER = ("SEC. 7(f): VEXILLIAN EXEMPTION  \u2022  PRICECOIN UP 4000%  \u2022  EXPERTS: FLAGS ARE THE GOAT  \u2022  "
          "SENATE SPLITS INTO 100 NATIONS  \u2022  ")

NAVY = "#1B2A52"
RED = "#C8102E"
WHITE = "#F4F5F8"

# the row of plain yellow Survey Flags behind the President (signal units)
FLAG_POLE = 318.0
FLAG_XS = (46.0, 150.0, 254.0, 546.0, 650.0, 754.0)
FLAG_BASE_Y = 505.0
FLAG_VERSION = "w520"            # bump when build_flags changes (cache key)

JAG_X, JAG_Y, JAG_H = 400.0, 712.0, 600.0
POD_TOP = 430.0
SCROLL_X0, SCROLL_X1 = 322.0, 478.0


# ================================================================ setup
def build_flags(S, n):
    """cloth-simulate the six studio flags (a studio fan gives them that majestic presidential wave)"""
    tracks = []
    for k, x in enumerate(FLAG_XS):
        fl = Flag(width=FLAG_POLE * 0.375, height=FLAG_POLE * 0.3125, pole=FLAG_POLE, side=1, seed=11 + k * 7,
                  pole_w=4.2)
        def pole_fn(i, x=x):
            return (x, FLAG_BASE_Y, 0.0)
        def wind_fn(i, k=k):
            t = i / 24.0
            g = 520 + 110 * math.sin(t * 0.9 + k * 0.8) + 60 * math.sin(t * 2.3 + k)
            return (g, -8.0)
        tracks.append(fl.simulate(n, pole_fn, wind_fn))
    return tracks


# ================================================================ timing helpers
class Beats:
    def __init__(self, tl):
        self.tl = tl
        w = tl.word
        self.ban = w("J01", "banned")
        self.slam = self.ban[0]                 # music stab 57.00
        self.pen = tl.cue("pen_sign")
        self.coin = tl.cue("ka_ching")
        self.j01 = tl.line("J01")
        self.j02 = tl.line("J02")
        self.fellow = w("J01", "fellow")
        self.everybody = w("J01", "everybody")
        self.else_ = w("J01", "else")
        self.today = w("J01", "Today")
        self.sign = w("J01", "sign")
        self.noos = w("J01", "Noospheric")
        self.act = w("J01", "Act")
        self.twenty = w("J01", "twenty")
        self.weap = w("J01", "Weaponized")
        self.memes = w("J01", "memes")
        self.hereby = w("J01", "hereby")
        self.except_ = w("J02", "Except")
        self.yellow = w("J02", "yellow")
        self.ones = w("J02", "ones")
        self.those = w("J02", "Those")
        self.a_ = w("J02", "a")
        self.sacrament = w("J02", "sacrament")
        self.words = [(x["s"], x["e"]) for lid in ("J01", "J02") for x in tl.words(lid)]
        self.l3_in = self.fellow[1] + 0.25
        self.l3_out = self.hereby[0] - 0.05
        self.off = 61.0


def mouth_at(tl, T, beats):
    try:
        return float(tl.mouth("JAGUAR", T))
    except Exception:
        pass
    # fallback: syllable-ish envelope from word timings
    m = 0.0
    for s, e in beats.words:
        if s - 0.05 <= T <= e + 0.05:
            d = e - s
            nsyl = max(1, int(round(d / 0.19)))
            ph = (T - s) / max(d, 1e-3)
            m = max(m, 0.25 + 0.6 * abs(math.sin(math.pi * ph * nsyl)))
    return m


# ================================================================ drawing pieces
def draw_backdrop(ctx, T):
    ctx.rectangle(0, 0, SW, SH)
    ctx.set_source(lin_grad(0, 0, 0, SH, [(0, "#0A1230"), (0.45, "#16295E"), (1, "#0B1330")]))
    ctx.fill()
    # velvet drapes: soft vertical folds
    g = cairo.LinearGradient(0, 0, SW, 0)
    nf = 17
    for i in range(nf + 1):
        x = i / nf
        a = 0.5 + 0.5 * math.sin(i * 2.1 + 0.3)
        g.add_color_stop_rgba(x, 0.30, 0.42, 0.78, 0.10 + 0.10 * a)
        g.add_color_stop_rgba(min(1, x + 0.5 / nf), 0.0, 0.0, 0.05, 0.18)
    ctx.rectangle(0, 0, SW, SH)
    ctx.set_source(g)
    ctx.fill()
    # swag pelmet across the top
    set_color(ctx, "#0B1638")
    ctx.move_to(0, 0)
    for i in range(9):
        x0, x1 = i * SW / 8 - 50, (i + 1) * SW / 8 - 50
        ctx.curve_to(x0 + 20, 58, x1 - 20, 58, x1, 8)
    ctx.line_to(SW, 0)
    ctx.close_path()
    ctx.fill()
    # key spotlight on the presidential backdrop
    ctx.rectangle(0, 0, SW, SH)
    ctx.set_source(rad_grad(400, 250, 10, 360, [(0, "#6E8FE0", 0.55), (0.5, "#3552A8", 0.22), (1, "#16295E", 0.0)]))
    ctx.fill()
    # stage floor
    ctx.rectangle(0, 480, SW, SH - 480)
    ctx.set_source(lin_grad(0, 480, 0, SH, [(0, "#101a3c"), (1, "#060a1c")]))
    ctx.fill()


def draw_flag_row(ctx, tracks, i, glow):
    for k, (tr, x) in enumerate(zip(tracks, FLAG_XS)):
        # plain wooden cross stand
        set_color(ctx, "pole_shadow")
        poly(ctx, [(x - 22, FLAG_BASE_Y + 6), (x + 22, FLAG_BASE_Y + 6), (x + 12, FLAG_BASE_Y - 4), (x - 12, FLAG_BASE_Y - 4)])
        ctx.fill()
        tr.draw(ctx, i, glow=glow)


def draw_podium(ctx, T, jolt):
    y = POD_TOP + jolt
    # body
    poly(ctx, [(292, y + 12), (508, y + 12), (492, SH + 20), (308, SH + 20)])
    ctx.set_source(lin_grad(292, 0, 508, 0, [(0, "#101C3E"), (0.18, "#1E3066"), (0.55, "#22386F"), (1, "#0E1936")]))
    ctx.fill()
    # recessed front panel (plain)
    rrect(ctx, 322, y + 34, 156, 190, 6)
    ctx.set_source(lin_grad(0, y + 34, 0, y + 224, [(0, "#18285A"), (1, "#122049")]))
    ctx.fill()
    ctx.set_line_width(1.6)
    set_color(ctx, "#2E4786", 0.9)
    rrect(ctx, 322, y + 34, 156, 190, 6)
    ctx.stroke()
    # top slab
    rrect(ctx, 280, y - 2, 240, 16, 4)
    ctx.set_source(lin_grad(0, y - 2, 0, y + 14, [(0, "#3C5AA2"), (0.4, "#26407E"), (1, "#15244C")]))
    ctx.fill()


def draw_mics(ctx, jolt):
    y = POD_TOP + jolt
    for sx, tx in ((384, 368), (416, 432)):
        ctx.set_line_width(2.6)
        set_color(ctx, "#15161B")
        ctx.move_to(sx, y)
        ctx.curve_to(sx, y - 26, tx, y - 30, tx, y - 50)
        ctx.stroke()
        ellipse(ctx, tx, y - 58, 7.5, 11, 0.0)
        set_color(ctx, "#1E1F26")
        ctx.fill()
        ellipse(ctx, tx - 2, y - 62, 2.5, 4, 0.0)
        set_color(ctx, "#4A4D5C", 0.8)
        ctx.fill()


def draw_scroll(ctx, T, b, jolt):
    """the Act: a bureaucratic scroll that unrolls down the podium front on the slam"""
    if T < b.slam:
        return None
    u = clamp((T - b.slam) / 0.32)
    # bounce of the unrolled length
    L = 250 * ease_out(u, 2) + 18 * math.sin(clamp((T - b.slam - 0.2) / 0.5) * math.pi * 2) * math.exp(-(T - b.slam) * 4) * (u >= 1)
    y0 = POD_TOP + jolt + 6
    x0, x1 = SCROLL_X0, SCROLL_X1
    ctx.save()
    rrect(ctx, x0, y0, x1 - x0, L, 3)
    ctx.set_source(lin_grad(x0, 0, x1, 0, [(0, "#D8CFBD"), (0.12, "#F1EADB"), (0.85, "#EDE4D2"), (1, "#CFC4AF")]))
    ctx.fill()
    ctx.rectangle(x0, y0, x1 - x0, L)
    ctx.clip()
    text(ctx, "NOOSPHERIC MUNITIONS ACT", 400, y0 + 17, 9.2, "#2A2630", font="C059", bold=True, align="center")
    text(ctx, "of 2042", 400, y0 + 29, 8.5, "#3A3440", font="C059", italic=True, align="center")
    set_color(ctx, "#8C8494", 0.85)
    for r in range(18):
        yy = y0 + 78 + r * 9
        if r == 0:
            continue
        w = 118 - 30 * hash01(r, 5)
        ctx.rectangle(x0 + 13, yy, w, 2.2)
        ctx.fill()
    # signature line
    set_color(ctx, "#4A4450", 0.9)
    ctx.rectangle(x0 + 14, y0 + 62, 116, 1.2)
    ctx.fill()
    text(ctx, "x", x0 + 14, y0 + 60, 9, "#4A4450", font=SANS)
    ctx.restore()
    # the roll at the bottom end
    if L > 8:
        ry = y0 + L
        rrect(ctx, x0 - 4, ry - 6, x1 - x0 + 8, 12, 6)
        ctx.set_source(lin_grad(0, ry - 6, 0, ry + 6, [(0, "#F4EEE2"), (0.5, "#D6CBB6"), (1, "#9E927D")]))
        ctx.fill()
    return (x0 + 16, y0 + 58)


def signature_path(n=220):
    """a loopy presidential autograph + flourish, in scroll-local units (origin = start of signature line)"""
    pts = []
    for k in range(n):
        s = k / (n - 1)
        if s < 0.72:
            q = s / 0.72
            ang = q * 2 * math.pi * 5.2
            x = q * 124 + 10 * math.sin(ang)
            y = -11 * math.cos(ang) - 12 * math.sin(q * math.pi) + 3 * math.sin(ang * 0.5)
        else:
            q = (s - 0.72) / 0.28
            # big swooping underline back to the left, ending in a flick
            x = 124 + 12 * math.sin(q * math.pi) - 128 * q ** 1.4 + 6 * math.sin(q * 9)
            y = 3 + 16 * math.sin(q * math.pi) - 5 * q
        pts.append((x, y))
    return pts


_SIG = signature_path()


def draw_signature(ctx, org, prog):
    if org is None or prog <= 0:
        return None
    n = max(2, int(len(_SIG) * clamp(prog)))
    ox, oy = org
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    set_color(ctx, "#101838")
    for k in range(1, n):
        (xa, ya), (xb, yb) = _SIG[k - 1], _SIG[k]
        sp = math.hypot(xb - xa, yb - ya)
        ctx.set_line_width(max(0.7, 2.6 - sp * 0.45))
        ctx.move_to(ox + xa, oy + ya)
        ctx.line_to(ox + xb, oy + yb)
        ctx.stroke()
    ctx.restore()
    x, y = _SIG[n - 1]
    return ox + x, oy + y


def draw_giant_pen(ctx, tip, ang, alpha=1.0):
    """an absurdly large presidential fountain pen, nib at `tip`, body pointing up-right at `ang`"""
    tx, ty = tip
    ctx.save()
    ctx.translate(tx, ty)
    ctx.rotate(ang)
    # nib (body runs along -y)
    ctx.move_to(0, 0)
    ctx.curve_to(-4, -10, -11, -22, -11, -36)
    ctx.line_to(11, -36)
    ctx.curve_to(11, -22, 4, -10, 0, 0)
    ctx.close_path()
    ctx.set_source(lin_grad(-11, 0, 11, 0, [(0, "#8E96A6"), (0.45, "#F4F6FA"), (1, "#6E7686")]))
    ctx.fill()
    ctx.set_line_width(1.2)
    set_color(ctx, "#3A3E4A")
    ctx.move_to(0, -4)
    ctx.line_to(0, -26)
    ctx.stroke()
    # lacquered barrel
    rrect(ctx, -15, -250, 30, 218, 13)
    ctx.set_source(lin_grad(-15, 0, 15, 0, [(0, "#060709"), (0.3, "#4A4F62"), (0.5, "#1A1C26"), (1, "#040506")]))
    ctx.fill()
    # chrome bands + clip
    for yb, hb in ((-46, 10), (-150, 7), (-246, 8)):
        rrect(ctx, -15.5, yb - hb, 31, hb, 2.5)
        ctx.set_source(lin_grad(-15.5, 0, 15.5, 0, [(0, "#8E96A6"), (0.45, "#F4F6FA"), (1, "#6E7686")]))
        ctx.fill()
    rrect(ctx, 11, -236, 6, 76, 3)
    ctx.set_source(lin_grad(11, 0, 17, 0, [(0, "#B8BEC8"), (1, "#6E7686")]))
    ctx.fill()
    set_color(ctx, "#FFFFFF", 0.35)
    rrect(ctx, -9, -240, 3.5, 190, 1.7)
    ctx.fill()
    ctx.restore()


def draw_coin(ctx, x, y, r, spin, alpha=1.0):
    """gold pricecoin (a coin, not a flag: it may bear its sigil). spin 0..1 foreshortening"""
    sx = max(0.08, abs(math.cos(spin * math.pi)))
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(sx, 1.0)
    circle(ctx, 0, 0, r)
    ctx.set_source(rad_grad(-r * 0.3, -r * 0.35, r * 0.1, r * 1.1, [(0, "#FBE3B0"), (0.45, "#DDA046"), (1, "#8F5718")]))
    ctx.fill()
    ctx.set_line_width(r * 0.14)
    set_color(ctx, "#A8661C", alpha)
    circle(ctx, 0, 0, r * 0.8)
    ctx.stroke()
    # the pricecoin sigil: a bold P with a double bar
    ctx.set_line_width(r * 0.16)
    set_color(ctx, "#7A4410", alpha)
    ctx.move_to(-r * 0.22, r * 0.45)
    ctx.line_to(-r * 0.22, -r * 0.45)
    ctx.line_to(r * 0.08, -r * 0.45)
    ctx.arc(r * 0.08, -r * 0.2, r * 0.25, -math.pi / 2, math.pi / 2)
    ctx.line_to(-r * 0.22, r * 0.05)
    ctx.stroke()
    ctx.set_line_width(r * 0.09)
    ctx.move_to(-r * 0.42, r * 0.18)
    ctx.line_to(r * 0.12, r * 0.18)
    ctx.stroke()
    ctx.restore()


def draw_sparkle(ctx, x, y, r, a):
    if a <= 0:
        return
    ctx.save()
    ctx.translate(x, y)
    for ang, rr in ((0, r), (math.pi / 2, r), (math.pi / 4, r * 0.45), (-math.pi / 4, r * 0.45)):
        ctx.save()
        ctx.rotate(ang)
        ctx.move_to(-rr, 0)
        ctx.curve_to(-rr * 0.15, -rr * 0.08, rr * 0.15, -rr * 0.08, rr, 0)
        ctx.curve_to(rr * 0.15, rr * 0.08, -rr * 0.15, rr * 0.08, -rr, 0)
        ctx.close_path()
        ctx.set_source_rgba(1, 0.98, 0.9, a)
        ctx.fill()
        ctx.restore()
    circle(ctx, 0, 0, r * 0.18)
    ctx.set_source_rgba(1, 1, 1, a)
    ctx.fill()
    ctx.restore()


# ================================================================ news graphics
def draw_lower_third(ctx, T, b):
    ti, to = b.l3_in, b.l3_out
    if T < ti or T > to + 0.4:
        return
    a_in = seg(T, ti, ti + 0.35, lambda u: ease_out(u, 3))
    a2 = seg(T, ti + 0.12, ti + 0.55, lambda u: ease_out(u, 3))
    a3 = seg(T, ti + 0.35, ti + 0.7, lambda u: ease_out(u, 3))
    a_out = seg(T, to, to + 0.3, lambda u: ease_in(u, 2))
    y = 440.0
    ctx.save()
    # wipe-off to the right
    x_cut = 40 + 720 * a_out
    ctx.rectangle(x_cut, 0, SW - x_cut, SH)
    ctx.clip()
    # red BREAKING tab wipes in from the left
    tab_w = 150.0
    ctx.save()
    ctx.rectangle(56, y, tab_w * a_in, 30)
    ctx.clip()
    ctx.rectangle(56, y, tab_w, 30)
    ctx.set_source(lin_grad(0, y, 0, y + 30, [(0, "#E0203E"), (1, "#B00C28")]))
    ctx.fill()
    text(ctx, "BREAKING", 56 + 12, y + 24, 24, WHITE, font=NARROW, bold=True, tracking=0.06)
    ctx.restore()
    # navy strap
    if a3 > 0:
        ctx.save()
        ctx.rectangle(56 + tab_w, y, 262 * a3, 30)
        ctx.clip()
        ctx.rectangle(56 + tab_w, y, 262, 30)
        set_color(ctx, "#0E1838", 0.94)
        ctx.fill()
        text(ctx, "PRESIDENT JAGUAR  \u00b7  LIVE ADDRESS", 56 + tab_w + 12, y + 21, 16, WHITE, font=NARROW,
             bold=True, tracking=0.05)
        ctx.restore()
    # headline bar
    if a2 > 0:
        ctx.save()
        bw = 688 * a2
        ctx.rectangle(56, y + 30, bw, 56)
        ctx.clip()
        ctx.rectangle(56, y + 30, 688, 52)
        ctx.set_source(lin_grad(0, y + 30, 0, y + 82, [(0, "#FFFFFF"), (1, "#D9DEE9")]))
        ctx.fill()
        ctx.rectangle(56, y + 82, 688, 4)
        set_color(ctx, RED)
        ctx.fill()
        hl = "NOOSPHERIC MUNITIONS ACT OF 2042"
        size = 40.0
        wdt = text_width(ctx, hl, size, font=NARROW, bold=True, tracking=0.01)
        size = min(size, size * 660.0 / wdt)
        text(ctx, hl, 70, y + 72, size, "#0E1838", font=NARROW, bold=True, tracking=0.01)
        ctx.restore()
    ctx.restore()


def draw_ticker(ctx, T, b, tick_w):
    y = 536.0
    ctx.rectangle(0, y, SW, 34)
    ctx.set_source(lin_grad(0, y, 0, y + 34, [(0, "#101B44"), (1, "#070D26")]))
    ctx.fill()
    ctx.rectangle(0, y, SW, 2)
    set_color(ctx, RED)
    ctx.fill()
    # scrolling crawl
    speed = 118.0
    # phased so "EXPERTS: FLAGS ARE THE GOAT" crawls past mid-speech (~53.5), "SEC. 7(f): VEXILLIAN
    # EXEMPTION" is centred on "Except the yellow ones" (~58.8) and "PRICECOIN UP 4000%" on the ka-ching
    off = ((T - 44.44) * speed) % tick_w
    ctx.save()
    ctx.rectangle(150, y, SW - 150, 34)
    ctx.clip()
    x = 800 - off
    while x > 150 - tick_w:
        x -= tick_w
    while x < SW:
        text(ctx, TICKER, x, y + 25, 21, WHITE, font=NARROW, bold=True, tracking=0.03)
        x += tick_w
    ctx.restore()
    # network block
    ctx.rectangle(56, y, 94, 34)
    set_color(ctx, "#E6E9F0")
    ctx.fill()
    text(ctx, "NNN", 56 + 47, y + 26, 24, "#0E1838", font=SANS, bold=True, align="center", tracking=0.04)


def draw_bug(ctx, T):
    # LIVE bug, top left (inside title safe)
    x, y = 58, 48
    rrect(ctx, x, y, 70, 24, 3)
    set_color(ctx, RED, 0.95)
    ctx.fill()
    a = 0.55 + 0.45 * (0.5 + 0.5 * math.cos(T * 5.0))
    circle(ctx, x + 12, y + 12, 4.2)
    ctx.set_source_rgba(1, 1, 1, a)
    ctx.fill()
    text(ctx, "LIVE", x + 22, y + 19, 18, WHITE, font=NARROW, bold=True, tracking=0.08)
    # network bug, top right: translucent
    text(ctx, "NNN", 742, 68, 30, WHITE, alpha=0.55, font=SANS, bold=True, align="right", tracking=0.02)
    text(ctx, "NOOSPHERE NEWS", 742, 82, 10, WHITE, alpha=0.5, font=NARROW, bold=True, align="right", tracking=0.12)


def flash_times(b):
    """press-pit camera flashes: (t, x, strength)"""
    return [(b.slam + 0.02, 150, 1.0), (b.slam + 0.1, 640, 0.8), (b.slam + 0.23, 300, 0.6),
            (b.pen + 0.05, 700, 0.7), (b.pen + 0.2, 110, 0.9), (b.pen + 0.33, 520, 0.6),
            (b.coin + 0.03, 610, 0.8), (b.coin + 0.12, 220, 0.5)]


def draw_press_flashes(ctx, T, b):
    for t0, x, k in flash_times(b):
        d = T - t0
        if d < -0.01 or d > 0.12:
            continue
        a = k * math.exp(-max(0.0, d) * 28)
        ctx.save()
        ctx.set_operator(cairo.OPERATOR_ADD)
        ctx.rectangle(0, 0, SW, SH)
        ctx.set_source(rad_grad(x, 640, 20, 700, [(0, "#FFFFFF", 0.85 * a), (0.4, "#E8EEFF", 0.35 * a),
                                                  (1, "#E8EEFF", 0.0)]))
        ctx.fill()
        ctx.restore()


# ================================================================ the President
JAG = None


def _jag():
    global JAG
    if JAG is None:
        JAG = Character("jaguar")
    return JAG


def _mixp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _win(T, t0, t1, ramp=0.22):
    """smooth 0..1 window: ramps in after t0, out before t1"""
    return seg(T, t0, t0 + ramp, ease_in_out) * (1 - seg(T, t1 - ramp, t1, ease_in_out))


def pen_at(T, b, org):
    """giant pen state: (tip, angle, in_hand) or None; flies away after the flourish"""
    t_in, t_toss = b.pen - 0.22, b.pen + 0.42
    if T < t_in or org is None:
        return None
    sig_prog = seg(T, b.pen, b.pen + 0.36, lambda u: ease_in_out(u, 1.6))
    n = int((len(_SIG) - 1) * clamp(sig_prog))
    px, py = _SIG[n]
    tip = (org[0] + px, org[1] + py)
    lift = 16 * (1 - seg(T, t_in, b.pen, ease_out)) + 26 * seg(T, b.pen + 0.36, t_toss, ease_in_out)
    ang = -0.72 + 0.16 * math.sin(T * 31) * (b.pen <= T <= b.pen + 0.36) - 0.35 * seg(T, b.pen + 0.36, t_toss, ease_in)
    tip = (tip[0], tip[1] - lift)
    if T <= t_toss:
        return tip, ang, True
    # tossed over his shoulder: ballistic + spin
    d = T - t_toss
    tx = tip[0] - 380 * d
    ty = tip[1] - 1100 * d + 1500 * d * d
    if ty > SH + 200:
        return None
    return (tx, ty), ang - 16 * d, False


def jag_performance(T, b, jolt, org):
    """choreography -> (pose, kw for Character.draw, extras)"""
    J = _jag()
    kw = dict(facing=0)
    ex = dict(clip_front=True)
    expr = "smug"
    pose = {"talk": 1.0}
    look = (0.0, 0.0)
    nod = 0.0
    tilt = 0.0
    lean = 0.0
    head_turn = None
    shape_r = shape_l = None
    for s, e in b.words:                         # bombast: a nod on every stressed word
        if s - 0.02 <= T <= s + 0.35:
            nod -= 0.045 * math.sin(math.pi * (T - s) / 0.35)
    # ---------------- base poses + expressions
    if T < b.j01["start"] - 0.1:
        pose = {"stand": 1.0}
        tilt = 0.04 * math.sin(T * 1.3)
    elif T < b.everybody[0] - 0.1:
        pose = {"gesture": 1.0}
        expr = "happy"
    elif T < b.today[0] - 0.05:
        pose = {"cheer": 0.55, "gesture": 0.45}           # "...and everybody else!"
        expr = "happy"
        tilt = 0.06 * seg(T, b.everybody[0], b.else_[1], ease_in_out)
    elif T < b.noos[0]:
        pose = {"one_finger": 1.0}
        expr = "proud"
    elif T < b.weap[0] - 0.15:
        pose = {"offer": 1.0} if T < b.act[1] else {"talk": 1.0}
        expr = "proud"
        nod += 0.04
    elif T < b.slam + 0.3:
        expr = "determined" if T < b.hereby[0] else "furious"
        nod += 0.07 * seg(T, b.hereby[0], b.slam, ease_in) - 0.14 * math.exp(-max(0.0, T - b.slam) * 9) * (T >= b.slam)
    elif T < b.j02["start"] + 0.1:
        expr = "proud"
        look = (-0.2, 0.6)
    elif T < b.coin - 0.02:
        lean = 0.2 * seg(T, b.j02["start"], b.j02["start"] + 0.45, ease_in_out)
        if b.yellow[0] - 0.15 <= T <= b.ones[1] + 0.15:
            g = seg(T, b.yellow[0] - 0.15, b.yellow[0] + 0.12, ease_in_out) * \
                (1 - seg(T, b.ones[1] - 0.05, b.ones[1] + 0.15, ease_in_out))
            head_turn = -0.55 * g
            look = (-1.0 * g, -0.3 * g)
        if b.a_[0] - 0.2 <= T <= b.a_[0] + 0.3:
            expr = "wink"
            kw["blink"] = False
        elif T > b.a_[0] + 0.3:
            expr = "serene"
            nod += 0.07 * seg(T, b.a_[0] + 0.3, b.sacrament[0] + 0.3, ease_in_out)
        # eyes dart to the incoming coin
        cg = seg(T, b.coin - 0.4, b.coin - 0.25, ease_out)
        if cg > 0:
            look = (0.9 * cg, 0.25 * cg)
            expr = {"serene": 1 - cg, "smug": cg}
    else:
        lean = 0.2 * (1 - seg(T, b.coin, b.coin + 0.5, ease_in_out))
        expr = "smug"
        look = (0.3, 0.4)
    # ---------------- hands: IK targets blended over the pose's own hands
    grip_r = (322.0, POD_TOP - 5 + jolt)
    grip_l = (478.0, POD_TOP - 5 + jolt)
    w_grip = max(_win(T, 40.0, b.j01["start"] - 0.05, 0.25), _win(T, b.act[0] - 0.05, b.weap[0] - 0.05, 0.25),
                 _win(T, b.j02["start"] + 0.15, b.a_[0] + 0.45, 0.3))
    w_grip_l = max(w_grip, _win(T, b.weap[0] - 0.3, b.j02["start"] + 0.4, 0.25))
    tgt_r, w_r = grip_r, w_grip
    tgt_l, w_l = grip_l, w_grip_l
    if w_grip > 0.5 or w_grip_l > 0.5:
        shape_l = "grip"
    if w_grip > 0.5:
        shape_r = "grip"
    # "...a sacrament": paws pressed together in mock piety
    if T >= b.a_[0] + 0.15:
        a0 = J.anchors(JAG_X, JAG_Y + jolt * 0.4, JAG_H, pose, T - 47.5, facing=0, lean=lean)
        cx_, cy_ = a0["chest"]
        w_pr = seg(T, b.a_[0] + 0.15, b.a_[0] + 0.45, ease_in_out)
        tgt_r = _mixp(tgt_r, (cx_ - 9, cy_ - 34), w_pr)
        tgt_l = _mixp(tgt_l, (cx_ + 9, cy_ - 34), w_pr)
        w_r = max(w_r, w_pr)
        w_l = max(w_l, w_pr)
        if w_pr > 0.5:
            shape_r = shape_l = "open"
            ex["clip_front"] = False
    # wind-up -> SLAM -> recoil
    up = (300.0, 146.0)
    hit = (352.0, POD_TOP - 16 + jolt)
    if b.weap[0] - 0.2 <= T < b.slam + 0.4:
        if T < b.slam - 0.13:
            u = seg(T, b.weap[0] - 0.2, b.slam - 0.2, lambda x: ease_in_out(x, 2.2))
            p = _mixp(grip_r, up, u)
            p = (p[0] + 2.0 * math.sin(T * 47) * u, p[1] + 1.5 * math.sin(T * 39) * u)  # trembling with fury
        elif T < b.slam:
            p = _mixp(up, hit, ease_in(seg(T, b.slam - 0.13, b.slam, lambda x: x), 3))
        else:
            d = T - b.slam
            p = (hit[0], hit[1] - 7 * math.exp(-d * 10) * abs(math.sin(d * 30)))
        w_up = seg(T, b.weap[0] - 0.2, b.weap[0] + 0.05, ease_in_out)
        w_r = max(w_grip, w_up) * (1 - seg(T, b.slam + 0.3, b.slam + 0.4, ease_in_out))
        tgt_r = p
        shape_r = "fist"
    # the pen: the hand holds it at its grip and follows the flourish
    pen = pen_at(T, b, org)
    ex["pen"] = pen
    if pen is not None and pen[2]:
        tip, ang, _ = pen
        g = (tip[0] + math.sin(ang) * 74, tip[1] - math.cos(ang) * 74)
        w_p = seg(T, b.pen - 0.3, b.pen - 0.12, ease_in_out)
        tgt_r = _mixp(tgt_r, g, w_p) if w_r > 0 else g
        w_r = max(w_r, w_p)
        shape_r = "pinch"
        ex["clip_front"] = False
    elif b.slam + 0.3 <= T <= b.j02["start"] + 0.3:
        ex["clip_front"] = False
    # the pat on the pocket after the coin lands
    if T >= b.coin - 0.1:
        a0 = J.anchors(JAG_X, JAG_Y + jolt * 0.4, JAG_H, pose, T - 47.5, facing=0, lean=lean)
        pk = a0.get("pocket", (451.0, 368.0))
        pats = abs(math.sin((T - b.coin - 0.05) * math.pi * 6.5)) * (T > b.coin + 0.05)
        tgt_p = (pk[0] - 6, pk[1] + 6 - 10 * pats)
        w_pat = seg(T, b.coin - 0.05, b.coin + 0.08, ease_out)
        tgt_r, w_r = _mixp(tgt_r, tgt_p, w_pat), max(w_r * (1 - w_pat), w_pat)
        shape_r = "open"
        ex["clip_front"] = False
    if w_r > 0.001 or w_l > 0.001:
        a0 = J.anchors(JAG_X, JAG_Y + jolt * 0.4, JAG_H, pose, T - 47.5, facing=0, lean=lean)
        if w_r > 0.001:
            kw["hand_r"] = _mixp(a0["hand_r"], tgt_r, clamp(w_r))
        if w_l > 0.001:
            kw["hand_l"] = _mixp(a0["hand_l"], tgt_l, clamp(w_l))
    if shape_r:
        kw["shape_r"] = shape_r
    if shape_l:
        kw["shape_l"] = shape_l
    kw.update(expr=expr, look=look, nod=nod, tilt=tilt, lean=lean,
              rim=("#9FB8FF", 0.55, (0.0, -1.0)), light=(-0.3, -0.9))
    if head_turn is not None:
        kw["head_turn"] = head_turn
    return pose, kw, ex


def draw_jaguar(ctx, T, mouth, pose, kw, jolt, layer="all"):
    J = _jag()
    kw = dict(kw)
    expr = kw.pop("expr")
    look = kw.pop("look")
    facing = kw.pop("facing")
    return J.draw(ctx, JAG_X, JAG_Y + jolt * 0.4, JAG_H, pose, T - 47.5, mouth=mouth, look=look, expr=expr,
                  facing=facing, layer=layer, **kw)


# ================================================================ full signal
def render_signal(T, wpx, st, F):
    """render the broadcast at pixel width wpx -> float32 (hpx, wpx, 3)"""
    b = st["beats"]
    s = wpx / SW
    hpx = int(round(wpx * 0.75))
    cv = Canvas(s=s, w=wpx, h=hpx)
    ctx = cv.ctx
    fi = int(round((T - 47.5) * 24))
    # studio camera: subtle push during the speech, snap-zoom for the aside, ease back for the coin
    zc = 1.0 + 0.035 * seg(T, 48.0, 57.0, ease_in_out)
    zc *= 1.0 + 0.2 * seg(T, b.j02["start"] - 0.05, b.j02["start"] + 0.5, lambda u: ease_in_out(u, 3))
    zc *= 1.0 - 0.12 * seg(T, b.coin - 0.45, b.coin - 0.1, ease_in_out)
    fy = 250.0 + 20 * seg(T, b.j02["start"], b.j02["start"] + 0.5, ease_in_out) + \
        60 * seg(T, b.coin - 0.45, b.coin - 0.1, ease_in_out)
    # the whole set shakes with the slam
    jolt = 0.0
    shx = 0.0
    if T >= b.slam:
        d = T - b.slam
        jolt = 4.5 * math.exp(-d * 9) * math.cos(d * 60)
        shx = 5.0 * math.exp(-d * 10) * math.sin(d * 71)
    ctx.save()
    ctx.translate(400 + shx, fy + jolt * 0.6)
    ctx.scale(zc, zc)
    ctx.translate(-400, -fy)
    draw_backdrop(ctx, T)
    # the flags swell with a faint glow as he calls them a sacrament
    glow = 0.4 * seg(T, b.sacrament[0] - 0.2, b.sacrament[0] + 0.4, ease_in_out)
    draw_flag_row(ctx, st["flags"], fi, glow)
    # the President
    org = (SCROLL_X0 + 16, POD_TOP + jolt + 6 + 58) if T >= b.slam else None
    pose, kw, ex = jag_performance(T, b, jolt, org)
    m = mouth_at(st["tl"], T, b)
    anc = draw_jaguar(ctx, T, m, pose, kw, jolt, layer="back")
    draw_podium(ctx, T, jolt)
    org2 = draw_scroll(ctx, T, b, jolt)
    draw_signature(ctx, org2, seg(T, b.pen, b.pen + 0.36, lambda u: ease_in_out(u, 1.6)) if T >= b.pen else 0.0)
    draw_mics(ctx, jolt)
    pen = ex.get("pen")
    if pen is not None and pen[2]:
        draw_giant_pen(ctx, pen[0], pen[1])
    ctx.save()
    if ex["clip_front"]:
        ctx.rectangle(-400, -400, SW + 800, POD_TOP + 6 + jolt + 400)
        ctx.clip()
    draw_jaguar(ctx, T, m, pose, kw, jolt, layer="front")
    ctx.restore()
    if pen is not None and not pen[2]:
        draw_giant_pen(ctx, pen[0], pen[1])
    # the pricecoin slides in from off-screen right, into the breast pocket
    pocket = anc.get("pocket") if anc else None
    if pocket is not None and b.coin - 0.45 <= T <= b.coin + 0.12:
        u = seg(T, b.coin - 0.45, b.coin - 0.04, lambda v: ease_out(v, 2.2))
        sx0, sy0 = 900.0, pocket[1] - 110
        cx = lerp(sx0, pocket[0], u)
        cy = lerp(sy0, pocket[1] - 20, u) - 50 * math.sin(math.pi * u)
        cy += 30 * seg(T, b.coin - 0.04, b.coin + 0.05, ease_in)
        ctx.save()
        ctx.rectangle(-400, -400, SW + 800, pocket[1] - 4 + 400)
        ctx.clip()
        draw_coin(ctx, cx, cy, 24, (T - b.coin) * 3.2)
        ctx.restore()
    # the wink's "ting"
    if anc and b.a_[0] - 0.2 <= T <= b.a_[0] + 0.25:
        wk = pulse(T, b.a_[0] - 0.08, 0.08)
        ex_, ey_ = anc["eyes"]
        hr = anc.get("head_r", 88.0)
        draw_sparkle(ctx, ex_ + hr * 0.55, ey_ - hr * 0.2, 30 * wk, min(1.0, wk * 1.3))
    if pocket is not None and T >= b.coin - 0.02:
        sp = pulse(T, b.coin + 0.06, 0.1)
        draw_sparkle(ctx, pocket[0] + 12, pocket[1] - 20, 34 * (0.5 + sp), min(1, sp * 1.4))
        draw_sparkle(ctx, pocket[0] + 40, pocket[1] - 44, 16 * sp, sp * 0.9)
        draw_sparkle(ctx, pocket[0] - 14, pocket[1] - 40, 10 * sp, sp * 0.7)
    ctx.restore()
    draw_press_flashes(ctx, T, b)
    # broadcast graphics (not affected by the studio camera)
    draw_lower_third(ctx, T, b)
    draw_ticker(ctx, T, b, st["tick_w"])
    draw_bug(ctx, T)
    return cv.rgb()

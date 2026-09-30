"""m03 cartoon: the Justinian panel of San Vitale, with a Jaguar. Everything is painted in WALL units
(1920x1080 = the whole panel at zoom 1) and later set in tesserae.

Every drawing function takes a LAYER like vx.icon: 'color' paints the cartoon; 'gold' / 'pearl' paint opaque
white where that material is and ERASE under every other shape (so occlusion is exact). Gold leaf is painted with
GOLD / GOLD_D / GOLD_L (normal / shadowed / bright leaf), pearls with PEARL.
"""
import math

import cairo
import numpy as np

from vx.canvas import set_color, circle, ellipse, poly, smooth
from vx import mosaic as mz
from vx import icon as I

SM = mz.SMALTI
GOLD, GOLD_D, GOLD_L = SM["gold"], SM["gold_d"], SM["gold_l"]
PEARL = "#F3EFE4"
GOLDS = {GOLD, GOLD_D, GOLD_L}
INK = "#1E1612"
WHITE = SM["marble"]
PORPH = SM["porphyry"]
GREEN, GREEN_D, GREEN_L = "#2F6B45", "#1B4430", "#4C8A5A"

# ---------------------------------------------------------------- the panel geometry (wall units)
TIT = (30, 112)            # titulus band y0, y1
MAIN = (126, 966)          # gold register
FEET = 930
GROUND = (868, 966)
EMP = dict(x=830.0, y=FEET, h=720.0)
CODEX = dict(cx=1075.0, cy=600.0, pw=140.0, ph=196.0, m=9.0)
TITULUS_TEXT = "HIC IMPERATOR IAGVAR LEGEM SANCIT"
TITULUS = mz.Titulus(TITULUS_TEXT, 960, (TIT[0] + TIT[1]) / 2 + 20, 40, color=GOLD, font="roman_black", sep="·",
                     tracking=0.12)

# the court of San Vitale (vx.icon kinds), back to front
COURT = [  # (kind, x, h, seed, style)
    ("courtier", 1812, 676, 7, {}),
    ("courtier", 1652, 684, 6, {}),
    ("courtier", 1492, 680, 5, {}),
    ("courtier", 1322, 690, 8, dict(hair="bald", beard="full", hair_color="#A8A298", over="sand_c")),
    ("deacon", 1150, 682, 9, dict(prop="none")),
    ("guard", 72, 668, 0, dict(shield=None)),
    ("guard", 212, 662, 2, dict(shield=None)),
    ("guard", 142, 674, 1, {}),
    ("guard", 290, 670, 3, {}),
    ("courtier", 505, 684, 11, {}),
    ("courtier", 655, 692, 12, {}),
]
_FIG = {}


def figure(kind, seed, style=None, tile=None):
    key = (kind, seed, tuple(sorted((style or {}).items())))
    f = _FIG.get(key)
    if f is None:
        f = _FIG[key] = I.Figure(kind, seed=seed, tile=tile, **(style or {}))
    return f


def jaguar():
    return figure("jaguar", 0, dict(prop="none"), tile=8)


# ================================================================== layer-routed painting
def _src(ctx, c, L, a=1.0):
    if L == "color":
        ctx.set_operator(cairo.OPERATOR_OVER)
        set_color(ctx, c, a)
        return
    on = (c in GOLDS) if L == "gold" else (c == PEARL) if L == "pearl" else False
    if on:
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_source_rgba(1, 1, 1, a)
    else:
        ctx.set_operator(cairo.OPERATOR_DEST_OUT)
        ctx.set_source_rgba(0, 0, 0, a)


def fill(ctx, c, L="color", a=1.0):
    _src(ctx, c, L, a)
    ctx.fill()
    ctx.set_operator(cairo.OPERATOR_OVER)


def fillp(ctx, c, L="color", a=1.0):
    _src(ctx, c, L, a)
    ctx.fill_preserve()
    ctx.set_operator(cairo.OPERATOR_OVER)


def stroke(ctx, c, w, L="color", a=1.0):
    _src(ctx, c, L, a)
    ctx.set_line_width(w)
    ctx.stroke()
    ctx.set_operator(cairo.OPERATOR_OVER)


def line(ctx, pts, c, w, L="color", sm=False):
    ctx.new_path()
    (smooth if sm else poly)(ctx, pts, close=False)
    stroke(ctx, c, w, L)


def draw_text(ctx, tit, L, upto=None):
    """a vx.mosaic Titulus on one layer (gold letters -> white in the gold mask)."""
    if L == "color":
        tit.draw(ctx, upto=upto)
        return
    gold = tuple(round(v, 3) for v in tit.color) == tuple(round(v, 3) for v in mz.SC["gold"])
    ctx.save()
    if gold:
        c0, c1, c2 = tit.color, tit.dot_color, tit.over_color
        tit.color = tit.dot_color = tit.over_color = (1.0, 1.0, 1.0)
        tit.draw(ctx, upto=upto)
        tit.color, tit.dot_color, tit.over_color = c0, c1, c2
    else:
        ctx.set_operator(cairo.OPERATOR_DEST_OUT)
        tit.draw(ctx, upto=upto)
    ctx.restore()


# ================================================================== borders, ground, titulus
def draw_ground(ctx, L="color"):
    ctx.new_path()
    ctx.rectangle(0, 0, 1920, 1080)
    fill(ctx, SM["lapis_d"], L)
    ctx.rectangle(0, MAIN[0], 1920, MAIN[1] - MAIN[0])
    fill(ctx, GOLD, L)
    y0, y1 = GROUND
    ctx.rectangle(0, y0, 1920, y1 - y0)
    fill(ctx, GREEN, L)
    ctx.rectangle(0, y0, 1920, 16)
    fill(ctx, GREEN_L, L)
    ctx.rectangle(0, y1 - 26, 1920, 26)
    fill(ctx, GREEN_D, L)
    rng = np.random.default_rng(3)
    for _ in range(70):
        x, y = rng.uniform(10, 1910), rng.uniform(y0 + 24, y1 - 30)
        circle(ctx, x, y, 3.4)
        fill(ctx, ["#E9E1D0", "#B8433A", "#E9E1D0"][rng.integers(0, 3)], L)
    # top jewelled band: gold cabochons and pearls on porphyry
    ctx.rectangle(0, 0, 1920, TIT[0])
    fill(ctx, PORPH, L)
    for i in range(48):
        x = 20 + i * 40
        if i % 2 == 0:
            ctx.rectangle(x - 9, 7, 18, 16)
            fill(ctx, GOLD, L)
        else:
            circle(ctx, x, 15, 6)
            fill(ctx, PEARL, L)
    ctx.rectangle(0, TIT[0], 1920, TIT[1] - TIT[0])
    fill(ctx, SM["lapis"], L)
    for y, c in ((TIT[1], WHITE), (TIT[1] + 5, SM["porphyry_l"]), (TIT[1] + 10, PORPH)):
        ctx.rectangle(0, y, 1920, 5)
        fill(ctx, c, L)
    for y, c in ((MAIN[1], PORPH), (MAIN[1] + 5, WHITE), (MAIN[1] + 10, SM["lapis_l"])):
        ctx.rectangle(0, y, 1920, 5)
        fill(ctx, c, L)
    # bottom guilloche band
    by0 = MAIN[1] + 15
    ctx.rectangle(0, by0, 1920, 1080 - by0)
    fill(ctx, SM["lapis_d"], L)
    cy = (by0 + 1080) / 2
    amp = (1080 - by0) * 0.32
    for ph, c1 in ((0.0, WHITE), (math.pi, SM["porphyry_l"])):
        pts = [(x, cy + amp * math.sin(x / 1920 * 2 * math.pi * 22 + ph)) for x in np.linspace(-20, 1940, 400)]
        line(ctx, pts, c1, 11, L)
    for i in range(22):
        circle(ctx, (i + 0.25) * 1920 / 22 + 1920 / 88, cy, 7)
        fill(ctx, GOLD, L)
    # side rainbow bands
    for x0 in (0, 1920 - 22):
        for k, c in enumerate((PORPH, WHITE, SM["lapis_l"])):
            ctx.rectangle(x0 + (k * 7 if x0 == 0 else 15 - k * 7), MAIN[0], 7, MAIN[1] - MAIN[0])
            fill(ctx, c, L)


def draw_titulus(ctx, L="color", upto=None):
    draw_text(ctx, TITULUS, L, upto)


# ================================================================== the codex, signature, seal
def codex_rect():
    c = CODEX
    return c["cx"] - c["pw"] - c["m"], c["cy"] - c["ph"] / 2 - c["m"], 2 * (c["pw"] + c["m"]), c["ph"] + 2 * c["m"]


_top = CODEX["cy"] - CODEX["ph"] / 2
_lx = CODEX["cx"] - CODEX["pw"] / 2
_rx = CODEX["cx"] + CODEX["pw"] / 2
CODEX_TEXT = [
    mz.Titulus("LEX", _lx, _top + 42, 25, color=GOLD, font="roman_black", tracking=0.1),
    mz.Titulus("NOOS\nPHAE\nRICA", _lx, _top + 74, 15.5, color=GOLD, font="roman_black", tracking=0.1, leading=1.24),
]
CODEX_DATE = mz.Titulus("MMXLII", _rx, _top + 42, 17, color=GOLD, font="roman_black", tracking=0.06)
SIG_BOX = (CODEX["cx"] - CODEX["pw"] + 12, CODEX["cy"] + 40, CODEX["pw"] - 22, 44)
SIG_PTS = [(0.03, 0.30), (0.10, 0.95), (0.05, 0.98), (0.01, 0.72), (0.14, 0.40), (0.21, 0.60), (0.18, 0.86),
           (0.27, 0.52), (0.31, 0.78), (0.37, 0.46), (0.40, 0.86), (0.35, 1.18), (0.44, 0.62), (0.50, 0.80),
           (0.56, 0.50), (0.60, 0.76), (0.66, 0.52), (0.71, 0.74), (0.79, 0.40), (0.83, 0.66), (0.97, 0.30),
           (0.92, 1.02), (0.55, 1.10), (0.12, 1.06)]
SEAL = dict(cx=_rx, cy=CODEX["cy"] + 22, r=38.0)


def sig_path():
    """dense polyline of the signature in wall units + normalised arc length (0..1)."""
    x0, y0, w, h = SIG_BOX
    P = np.array([(x0 + u * w, y0 + v * h) for u, v in SIG_PTS], np.float64)
    out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(0, i - 1)], P[i], P[i + 1], P[min(len(P) - 1, i + 2)]
        for t in np.linspace(0, 1, 18, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    out = np.array(out)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(out, axis=0).T))]
    return out, seg / seg[-1]


def draw_codex(ctx, L="color", sig=0.0, sig_pts=None, mm=6, seal=False):
    """the open imperial codex: purple vellum pages, gold letters (chrysography), gold binding.
    sig: 0..1 signature progress; mm: numerals of MMXLII set (0..6); seal: the gold seal impressed."""
    c = CODEX
    x0, y0, w, h = codex_rect()
    ctx.new_path()
    ctx.rectangle(x0, y0, w, h)
    fillp(ctx, GOLD, L)
    stroke(ctx, "#5A3A12", 3, L)
    for (jx, jy) in ((x0 + 5, y0 + 5), (x0 + w - 5, y0 + 5), (x0 + 5, y0 + h - 5), (x0 + w - 5, y0 + h - 5)):
        circle(ctx, jx, jy, 5)
        fill(ctx, SM["porphyry_l"], L)
    for p in (0, 1):
        px = c["cx"] - c["pw"] if p == 0 else c["cx"]
        ctx.new_path()
        ctx.rectangle(px, c["cy"] - c["ph"] / 2, c["pw"], c["ph"])
        fill(ctx, SM["tyrian"], L)
    ctx.rectangle(c["cx"] - 3, c["cy"] - c["ph"] / 2, 6, c["ph"])
    fill(ctx, "#2E0E28", L)
    for t in CODEX_TEXT:
        draw_text(ctx, t, L)
    if mm > 0:
        draw_text(ctx, CODEX_DATE, L, upto=int(mm))
    line(ctx, [(SIG_BOX[0] - 2, SIG_BOX[1] + SIG_BOX[3] + 5), (SIG_BOX[0] + SIG_BOX[2], SIG_BOX[1] + SIG_BOX[3] + 5)],
         "#7E4A72", 1.6, L)
    if sig > 0 and sig_pts is not None:
        pts, u = sig_pts
        k = int(np.searchsorted(u, sig))
        if k >= 2:
            ctx.new_path()
            ctx.move_to(*pts[0])
            for p in pts[1:k]:
                ctx.line_to(*p)
            stroke(ctx, GOLD, 4.6, L)
    if seal:
        draw_seal_mark(ctx, SEAL["cx"], SEAL["cy"], SEAL["r"], L)


def draw_seal_mark(ctx, cx, cy, r, L="color"):
    """the gold seal: raised rim, a ring of pellets and the imperial paw print (shadowed leaf)."""
    circle(ctx, cx, cy, r)
    fill(ctx, GOLD, L)
    ctx.new_path()
    ctx.arc(cx, cy, r - 3.5, 0, 2 * math.pi)
    stroke(ctx, GOLD_D, 3.0, L)
    for k in range(18):
        a = k / 18 * 2 * math.pi
        circle(ctx, cx + (r - 9) * math.cos(a), cy + (r - 9) * math.sin(a), 1.9)
        fill(ctx, GOLD_D, L)
    k = r / 31.0
    ellipse(ctx, cx, cy + 6 * k, 9.5 * k, 8 * k)
    fill(ctx, GOLD_D, L)
    for dx, dy, rr in ((-11, -5, 4.2), (-4, -12, 4.4), (4, -12, 4.4), (11, -5, 4.2)):
        ellipse(ctx, cx + dx * k, cy + dy * k, rr * 0.85 * k, rr * 1.1 * k)
        fill(ctx, GOLD_D, L)


def deacon_hands(ctx, L="color"):
    """the deacon's hands holding the codex's lower corners (over the binding)."""
    x0, y0, w, h = codex_rect()
    for hx in (x0 + 18, x0 + w - 18):
        ellipse(ctx, hx, y0 + h - 2, 14, 11)
        fillp(ctx, "#D8A887", L)
        stroke(ctx, "#5A3526", 2.2, L)


# ================================================================== the halo
def halo_geom():
    return EMP["x"], EMP["y"] - 0.94 * EMP["h"], 0.148 * EMP["h"]


def draw_halo(ctx, cx, cy, r, L="color", crack=0.0):
    circle(ctx, cx, cy, r)
    fill(ctx, GOLD, L)
    ctx.new_path()
    ctx.arc(cx, cy, r - 3.5, 0, 2 * math.pi)
    stroke(ctx, PORPH, 7.0, L)
    for k in range(40):
        a = k / 40 * 2 * math.pi
        circle(ctx, cx + (r - 12) * math.cos(a), cy + (r - 12) * math.sin(a), 3.0)
        fill(ctx, PEARL, L)
    if crack > 0:
        # one hairline fracture through the halo, grows with crack 0..1
        # it runs down the right side of the nimbus, the side his leaning head leaves bare
        pts = [(cx + r * 0.22, cy - r * 0.97), (cx + r * 0.38, cy - r * 0.72), (cx + r * 0.3, cy - r * 0.5),
               (cx + r * 0.52, cy - r * 0.3), (cx + r * 0.48, cy - r * 0.06), (cx + r * 0.7, cy + r * 0.12),
               (cx + r * 0.64, cy + r * 0.34), (cx + r * 0.86, cy + r * 0.44), (cx + r * 0.95, cy + r * 0.3)]
        k = max(2, int(round(crack * len(pts))))
        line(ctx, pts[:k], INK, 3.6, L)


# ================================================================== the court
_SPRITES = {}


def _surf_rgba(rgba):
    """premultiplied float rgba (h,w,4) -> cairo ARGB32 surface (keeps its buffer alive)."""
    h, w = rgba.shape[:2]
    a8 = np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)
    buf = np.empty((h, w, 4), np.uint8)
    buf[..., 0], buf[..., 1], buf[..., 2], buf[..., 3] = a8[..., 2], a8[..., 1], a8[..., 0], a8[..., 3]
    surf = cairo.ImageSurface.create_for_data(buf, cairo.FORMAT_ARGB32, w, h, w * 4)
    return surf, buf


def _surf_a8(m):
    h, w = m.shape
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, w)
    buf = np.zeros((h, stride), np.uint8)
    buf[:, :w] = np.clip(m * 255.0 + 0.5, 0, 255).astype(np.uint8)
    surf = cairo.ImageSurface.create_for_data(buf, cairo.FORMAT_A8, w, h, stride)
    return surf, buf


def court_sprite(i, look=(0.0, 0.0), blink=0.0):
    """court figure i pre-rendered on every layer at 1 px per wall unit (cached per process):
    (X0, Y0, colour, alpha, gold, pearl) cairo surfaces."""
    key = (i, round(look[0] * 4) / 4, round(look[1] * 4) / 4, round(blink * 2) / 2)
    sp = _SPRITES.get(key)
    if sp is not None:
        return sp
    kind, x, hh, seed, style = COURT[i]
    f = figure(kind, seed, style, tile=8)
    bx0, by0, bx1, by1 = f.box(hh)
    pad = 30
    X0, Y0 = int(math.floor(x + bx0 - pad)), int(math.floor(FEET + by0 - pad))
    w, h = int(math.ceil(bx1 - bx0 + 2 * pad)), int(math.ceil(by1 - by0 + 2 * pad))
    kw = dict(t=0.0, look=(key[1], key[2]), blink=key[3])
    if kind == "deacon":
        cx0, cy0, cw, ch = codex_rect()
        kw.update(hand_r=(cx0 + 30, cy0 + ch - 18), hand_l=(cx0 + cw - 30, cy0 + ch - 18), shape_r="hold",
                  shape_l="hold")

    def draw(ctx, L):
        ctx.translate(-X0, -Y0)
        f.draw(ctx, x, FEET, hh, layer=L, **kw)

    c = I.cartoon(draw, s=1.0, w=w, h=h, layers=("color", "gold", "pearl"))
    sp = (X0, Y0, _surf_rgba(c["rgba"]), _surf_a8(c["alpha"]), _surf_a8(c["gold"]), _surf_a8(c["pearl"]))
    if len(_SPRITES) > 600:
        _SPRITES.clear()
    _SPRITES[key] = sp
    return sp


def draw_court(ctx, L="color", looks=None, blinks=None):
    """every figure except the Emperor, back to front (vx.icon sprites). looks/blinks: {index: value}."""
    looks = looks or {}
    blinks = blinks or {}
    for i in range(len(COURT)):
        X0, Y0, col, alpha, gold, pearl = court_sprite(i, looks.get(i, (0.0, 0.0)), blinks.get(i, 0.0))
        if L == "color":
            ctx.set_source_surface(col[0], X0, Y0)
            ctx.paint()
            continue
        ctx.set_operator(cairo.OPERATOR_DEST_OUT)
        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.mask_surface(alpha[0], X0, Y0)
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_source_rgba(1, 1, 1, 1)
        ctx.mask_surface((gold if L == "gold" else pearl)[0], X0, Y0)


# ================================================================== props (local units)
SCEPTRE_EXT = (84, 272)
SCEPTRE_DISC = (42.0, 40.0)           # the seal matrix at the top end
SCEPTRE_GRIP = (42.0, 176.0)          # where the paw holds it
SCEPTRE_NIB = (42.0, 264.0)           # the writing point


def draw_sceptre(ctx, L="color"):
    """the imperial stylus-sceptre: a golden reed pen whose other end is the seal matrix (paw print)."""
    x = SCEPTRE_DISC[0]
    ctx.new_path()
    ctx.move_to(x - 4.4, 64)
    ctx.line_to(x + 4.4, 64)
    ctx.line_to(x + 3.8, 242)
    ctx.line_to(x, 266)
    ctx.line_to(x - 3.8, 242)
    ctx.close_path()
    fillp(ctx, GOLD, L)
    stroke(ctx, "#4E3210", 1.4, L)
    ctx.new_path()
    ctx.move_to(x - 2.8, 246)
    ctx.line_to(x + 2.8, 246)
    ctx.line_to(x, 268)
    ctx.close_path()
    fill(ctx, "#2A180A", L)
    for yy, c in ((72, GOLD_L), (110, SM["porphyry_l"]), (224, GOLD_L)):
        ctx.rectangle(x - 6.5, yy, 13, 5)
        fill(ctx, c, L)
    draw_seal_mark(ctx, SCEPTRE_DISC[0], SCEPTRE_DISC[1], 33.0, L)


COIN_EXT = (56, 56)
COIN_R = 22.0


def draw_coin(ctx, L="color"):
    """the gold solidus: beaded rim and a tiny jaguar bust (shadowed leaf)."""
    cx = cy = COIN_EXT[0] / 2
    r = COIN_R
    k = r / 19.0
    circle(ctx, cx, cy, r)
    fill(ctx, GOLD, L)
    ctx.new_path()
    ctx.arc(cx, cy, r - 2.2 * k, 0, 2 * math.pi)
    stroke(ctx, GOLD_D, 1.6 * k, L)
    for i in range(28):
        a = i / 28 * 2 * math.pi
        circle(ctx, cx + (r - 4.4 * k) * math.cos(a), cy + (r - 4.4 * k) * math.sin(a), 0.9 * k)
        fill(ctx, GOLD_D, L)
    # the bust: shoulders, ears, head, crown, eyes
    ctx.new_path()
    ctx.arc(cx, cy + 12.5 * k, 9.5 * k, math.pi, 2 * math.pi)
    fill(ctx, GOLD_D, L)
    for sx in (-1, 1):
        circle(ctx, cx + sx * 5.6 * k, cy - 5.6 * k, 2.4 * k)
        fill(ctx, GOLD_D, L)
    circle(ctx, cx, cy + 0.5 * k, 6.6 * k)
    fill(ctx, GOLD_D, L)
    circle(ctx, cx, cy + 1.0 * k, 4.9 * k)
    fill(ctx, GOLD_L, L)
    ctx.rectangle(cx - 5.2 * k, cy - 6.4 * k, 10.4 * k, 2.4 * k)
    fill(ctx, GOLD_D, L)
    for sx in (-1, 1):
        circle(ctx, cx + sx * 1.9 * k, cy - 0.2 * k, 0.95 * k)
        fill(ctx, GOLD_D, L)
    circle(ctx, cx, cy + 2.4 * k, 0.9 * k)
    fill(ctx, GOLD_D, L)


HAND_EXT = (300, 230)
HAND_TIP = (262.0, 58.0)               # where the coin sits between finger and thumb


def draw_vex_hand(ctx, L="color", pinch=1.0):
    """an ochre Vexillian sleeve with a dark cuff and a hand, reaching up from the lower left; pinch 1 = holding
    the coin, 0 = released. (A garment: deep folds, a cuff band - it must never read as cloth of a Flag.)"""
    robe, robe_d, robe_k, robe_l = "#B98418", "#7E5610", "#4E3408", "#D2A040"
    skin, skin_d, line_c = SM["flesh"], SM["flesh_d"], "#5A3526"
    # the sleeve: a wide, heavily folded cylinder coming up from the lower left
    ctx.new_path()
    ctx.move_to(0, 150)
    ctx.curve_to(60, 118, 130, 88, 186, 70)
    ctx.line_to(214, 118)
    ctx.curve_to(160, 150, 90, 196, 40, 230)
    ctx.line_to(0, 230)
    ctx.close_path()
    fillp(ctx, robe, L)
    stroke(ctx, robe_k, 3.2, L)
    for k, (a, b) in enumerate(((0.18, 0.2), (0.42, 0.45), (0.66, 0.7))):
        p0 = (0 + a * 40, 150 + a * 80)
        p1 = (186 + b * 28, 70 + b * 48)
        line(ctx, [p0, ((p0[0] + p1[0]) / 2 + 6, (p0[1] + p1[1]) / 2 + 8), p1], robe_d, 5.0, L, sm=True)
    line(ctx, [(4, 140), (100, 104), (180, 76)], robe_l, 3.4, L, sm=True)
    # the cuff: a dark band with a porphyry stripe (a garment edge)
    ctx.new_path()
    ctx.move_to(178, 64)
    ctx.line_to(200, 54)
    ctx.line_to(228, 110)
    ctx.line_to(206, 122)
    ctx.close_path()
    fillp(ctx, robe_k, L)
    stroke(ctx, robe_k, 1.5, L)
    line(ctx, [(190, 61), (216, 116)], SM["porphyry_l"], 3.2, L)
    # hand: back of the hand, thumb and index pinching the coin at HAND_TIP
    op = 1.0 - pinch
    ctx.new_path()
    ctx.move_to(204, 60)
    ctx.curve_to(214, 44, 232, 36, 246, 40)
    ctx.curve_to(256, 44, 262 - 6 * op, 50, 266, 54 - 4 * op)
    ctx.curve_to(270, 60, 264, 66, 256, 64)
    ctx.curve_to(248, 64, 240, 70, 236, 82)
    ctx.curve_to(232, 96, 222, 108, 214, 110)
    ctx.close_path()
    fillp(ctx, skin, L)
    stroke(ctx, line_c, 2.4, L)
    # thumb
    ctx.new_path()
    ctx.move_to(236, 84)
    ctx.curve_to(246, 76 + 6 * op, 256, 70 + 8 * op, 264, 64 + 10 * op)
    stroke(ctx, line_c, 2.4, L)
    line(ctx, [(222, 56), (246, 48)], skin_d, 1.8, L)
    line(ctx, [(224, 68), (250, 58)], skin_d, 1.8, L)


# ================================================================== tile sizes for the wall layout
def tile_size_map(k=1.0, paw_spots=()):
    """per-pixel tessera size (wall units): big gold-ground tiles, figure tiles, fine tiles for faces, letters,
    the codex, and around paw_spots (where the Emperor's paws act)."""
    from vx.canvas import Canvas

    class _F:
        s = k
    cv = Canvas(_F)
    ctx = cv.ctx

    def zone(val, draw):
        ctx.set_source_rgba(val / 16.0, 0, 0, 1)
        draw()
        ctx.fill()

    zone(10.5, lambda: ctx.rectangle(0, MAIN[0], 1920, MAIN[1] - MAIN[0]))
    zone(6.0, lambda: ctx.rectangle(0, 0, 1920, MAIN[0]))
    zone(6.5, lambda: ctx.rectangle(0, MAIN[1], 1920, 1080 - MAIN[1]))
    zone(3.0, lambda: ctx.rectangle(0, TIT[0] + 6, 1920, TIT[1] - TIT[0] - 12))
    # figures (bodies) 7, faces and hands 4
    for kind, x, hh, seed, style in COURT:
        f = figure(kind, seed, style, tile=8)
        bx0, by0, bx1, by1 = f.box(hh)
        zone(7.0, lambda: ctx.rectangle(x + bx0, FEET + by0, bx1 - bx0, by1 - by0))
    e = EMP
    jb = jaguar().box(e["h"])
    zone(7.0, lambda: ctx.rectangle(e["x"] + jb[0], e["y"] + jb[1], jb[2] - jb[0], jb[3] - jb[1]))
    for kind, x, hh, seed, style in COURT:
        zone(4.0, lambda x=x, hh=hh: ctx.arc(x, FEET - 0.905 * hh, 0.1 * hh, 0, 2 * math.pi))
    hx, hy, hr = halo_geom()
    zone(4.6, lambda: ctx.arc(hx, hy, hr + 6, 0, 2 * math.pi))
    zone(2.5, lambda: ctx.arc(e["x"], e["y"] - 0.9 * e["h"], 0.14 * e["h"], 0, 2 * math.pi))
    for px, py in paw_spots:
        zone(3.4, lambda px=px, py=py: ctx.arc(px, py, 72, 0, 2 * math.pi))
    x0, y0, cw, chh = codex_rect()
    zone(2.2, lambda: ctx.rectangle(x0 - 3, y0 - 3, cw + 6, chh + 6))
    a = cv.rgba()
    return np.clip(a[..., 0] * 16.0, 2.0, 12.0).astype(np.float32)

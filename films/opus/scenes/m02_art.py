"""m02 art (owner M02): the cartoon of the nave wall - HVMANITAS, the colossal crowned icon of Humanity.

World units: the wide view of the whole picture frames [0,1920]x[0,1080] (camera zoom 1). The wall continues up
into the starry lapis of the upper border (seen only at the start, where m01's tilt hands over).

    lapis border  y < 18 (stars) | jewelled border 18..60 | gold ground below
    the CROWN of machines (the kamelaukion of Humanity), x 660..1260, y ~80..334:
        NATIONES      a crowned walled city with a gilded gear train in its gatehouse
        FIDES         a domed temple, saints processing into its door, a turning dome and a cog in its tympanum
        PHILOSOPHIAE  seven philosophers under a tree hung with gilded cogs, a sundial, a celestial globe
        band          lapis band with pearls and the three tituli
    the BUST (face, hair, neck, jewelled collar, purple robe) is a card stunt: thousands of tiny people, each
    one tessera-cluster, each wearing the colour of the picture where it stands.
    halo, pendilia, the name HVMA | NITAS flanking the head, processions of peoples in the lower corners.

Everything here is geometry + cairo painting at K guide pixels per world unit; m02_slop builds the tile layout
from it (cached) and animates the tiles by re-colouring them.
"""
import math

import cairo
import cv2
import numpy as np

from vx.canvas import col, set_color, circle, ellipse, poly, smooth
from vx.config import C
from vx.mosaic import SMALTI

K = 2.0                                   # guide px per world unit
X0, Y0, WW, WH = -200.0, -560.0, 2320.0, 1760.0
GW, GH = int(WW * K), int(WH * K)

S_ = {k: col(v) for k, v in SMALTI.items()}
P_ = {k: tuple(v) for k, v in C.items()}


def rgbx(h):
    return col(h)


# ------------------------------------------------------------------ palette extras (non-flag)
FLESH = rgbx("#D8A585")
FLESH_L = rgbx("#EFD2B6")
FLESH_H = rgbx("#F7E6D2")
FLESH_S = rgbx("#9C7358")
OLIVE = rgbx("#77714F")
ROSE = rgbx("#C4725F")
HAIR = rgbx("#3A2217")
HAIR_L = rgbx("#6A3B22")
IRIS = rgbx("#4A2A17")
PUPIL = rgbx("#120B09")
LID = rgbx("#24150E")
LIP_U = rgbx("#8C3424")
LIP_L = rgbx("#B85A48")
WHITE = S_["marble"]
PURPLE = rgbx("#4E1A45")
PURPLE_L = rgbx("#7A2C68")
PURPLE_D = rgbx("#2E0F2A")
GOLDC = S_["gold"]
GREEN_L = rgbx("#5E9A58")
LEAF = rgbx("#2F6B45")
LEAF_D = rgbx("#1B4630")
WOOD = rgbx("#7A4B2A")
WOOD_D = rgbx("#4A2C18")

# ------------------------------------------------------------------ geometry (single source of truth)
HALO = (960.0, 470.0, 400.0)
CROWN_X = (660.0, 1260.0)
BAND_Y = (282.0, 334.0)
TITULI = [("NATIONES", 765.0), ("FIDES", 960.0), ("PHILOSOPHI\u00c6", 1155.0)]   # AE ligature, as carved
TIT_Y, TIT_H = 318.0, 20.0                # baseline, cap height
MACHINE_X = [(664.0, 856.0), (856.0, 1064.0), (1064.0, 1256.0)]   # city, temple, philosophers
MACHINE_C = [(760.0, 205.0), (960.0, 185.0), (1160.0, 200.0)]

FACE_C = (960.0, 530.0)
FACE_R = (166.0, 220.0)
EYES = [(879.0, 505.0), (1041.0, 505.0)]
EYE_W, EYE_H = 112.0, 52.0
MOUTH_Y = 672.0

# rotating discs: (cx, cy, r, teeth, spokes, speed rad/s, machine, style)
#   machine 0 city, 1 temple, 2 philosophers; style: 'cog' gilded wheel, 'crown' a cog whose teeth are crowns
DISCS = []


def _mesh(c0, r0, r1, ang_deg, gap=0.8):
    a = math.radians(ang_deg)
    d = r0 + r1 + gap
    return (c0[0] + d * math.cos(a), c0[1] + d * math.sin(a))


def _build_discs():
    D = []
    # city gatehouse: a great crowned wheel with two meshing cogs below it
    big = (760.0, 176.0)
    D.append(dict(c=big, r=20.0, teeth=14, spokes=6, w=0.9, m=0, style="crown"))
    for side in (-1, 1):
        c = _mesh(big, 20.0, 11.5, 90 - side * 52)
        D.append(dict(c=c, r=11.5, teeth=8, spokes=4, w=-0.9 * 20.0 / 11.5, m=0, style="cog"))
    # temple: cog in the tympanum above the door + two small ones in the drum's flanks
    D.append(dict(c=(960.0, 214.0), r=13.0, teeth=10, spokes=5, w=0.7, m=1, style="cog"))
    for side in (-1, 1):
        c = _mesh((960.0, 214.0), 13.0, 8.0, 90 - side * 90)
        D.append(dict(c=c, r=8.0, teeth=6, spokes=3, w=-0.7 * 13.0 / 8.0, m=1, style="cog"))
    # philosophers: gilded cogs hanging in the tree like fruit (a meshing pair + one alone)
    a = (1134.0, 124.0)
    D.append(dict(c=a, r=10.0, teeth=8, spokes=4, w=1.1, m=2, style="cog"))
    D.append(dict(c=_mesh(a, 10.0, 7.0, 20), r=7.0, teeth=6, spokes=3, w=-1.1 * 10 / 7.0, m=2, style="cog"))
    D.append(dict(c=(1190.0, 150.0), r=9.0, teeth=8, spokes=4, w=-0.8, m=2, style="cog"))
    return D


DISCS = _build_discs()

DOME = (960.0, 160.0, 64.0)               # centre of the springing line, radius
GLOBE = (1160.0, 252.0, 12.5)
SUNDIAL = (1075.0, 186.0, 11.0)
TREE = (1160.0, 138.0, 60.0)

# processions (strips of vx.icon figures gliding by re-colouring): x-range, ground y, figure height, speed wu/s
# (signed), period L, machine (3 = the corner registers), kinds, pose, gap = door opening they vanish into
PROCS = [
    dict(x0=668.0, x1=852.0, y=282.0, h=21.0, v=6.0, L=48.0, m=0, kinds=("citizen", "people"), pose="offer",
         gap=(742.0, 778.0), gap_k=0.75),
    dict(x0=862.0, x1=940.0, y=282.0, h=20.0, v=5.0, L=39.0, m=1, kinds=("heretic",), pose="clasp", gap=None,
         gap_k=0.65),
    dict(x0=980.0, x1=1058.0, y=282.0, h=20.0, v=-5.0, L=39.0, m=1, kinds=("heretic",), pose="clasp", gap=None,
         gap_k=0.65),
    dict(x0=-60.0, x1=384.0, y=1018.0, h=84.0, v=14.0, L=300.0, m=3,
         kinds=("people", "citizen", "deacon", "philosopher", "people", "heretic"), pose="candle", gap=None,
         gap_k=0.56),
    dict(x0=1536.0, x1=1980.0, y=1018.0, h=84.0, v=-14.0, L=300.0, m=3,
         kinds=("people", "citizen", "deacon", "pope", "people", "heretic"), pose="candle", gap=None, gap_k=0.56),
]
PROC_TOP = 1.28      # band height = PROC_TOP * figure height


# ------------------------------------------------------------------ canvas helpers
class Surf:
    """a cairo surface covering the world at K px/wu; draw in world units."""

    def __init__(self, bg=None):
        self.surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, GW, GH)
        self.ctx = cairo.Context(self.surf)
        self.ctx.scale(K, K)
        self.ctx.translate(-X0, -Y0)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        if bg is not None:
            self.ctx.save()
            self.ctx.identity_matrix()
            set_color(self.ctx, bg)
            self.ctx.paint()
            self.ctx.restore()

    def u8(self):
        """(GH, GW, 4) uint8 view BGRA premultiplied"""
        self.surf.flush()
        st = self.surf.get_stride()
        buf = np.ndarray((GH, st // 4, 4), np.uint8, self.surf.get_data())
        return buf[:, :GW]

    def alpha(self):
        return self.u8()[..., 3].copy()

    def rgb(self):
        b = self.u8()
        return np.ascontiguousarray(b[..., 2::-1])   # RGB uint8 (premultiplied over black)


def to_px(x, y):
    return (x - X0) * K, (y - Y0) * K


def sample(img, xy):
    """nearest guide-pixel sample of img at world points xy (N,2)."""
    ix = np.clip(((xy[:, 0] - X0) * K).astype(np.int32), 0, img.shape[1] - 1)
    iy = np.clip(((xy[:, 1] - Y0) * K).astype(np.int32), 0, img.shape[0] - 1)
    return img[iy, ix]


def fill(ctx, c, a=1.0):
    set_color(ctx, c, a)
    ctx.fill()


def stroke(ctx, c, w, a=1.0):
    set_color(ctx, c, a)
    ctx.set_line_width(w)
    ctx.stroke()


def star8(ctx, x, y, r, rot=0.0):
    pts = []
    for i in range(16):
        rr = r if i % 2 == 0 else r * 0.46
        a = rot + i * math.pi / 8
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    poly(ctx, pts)


def text_path(ctx, s, x, y, size, font="P052", bold=True, align="center", tracking=0.08):
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    ws = [ctx.text_extents(ch).x_advance for ch in s]
    tw = sum(ws) + tracking * size * (len(s) - 1)
    x0 = x - tw / 2 if align == "center" else (x - tw if align == "right" else x)
    xs = []
    for ch, w_ in zip(s, ws):
        xs.append(x0)
        ctx.move_to(x0, y)
        ctx.text_path(ch)
        x0 += w_ + tracking * size
    return xs, ws


def crown_icon(ctx, x, y, w, h, gems=True, pts=3):
    """a small Byzantine crown, base centre (x, y), width w, height h (drawn upward)."""
    b = h * 0.42
    ctx.rectangle(x - w / 2, y - b, w, b)
    fill(ctx, GOLDC)
    # points
    for i in range(pts):
        px = x - w / 2 + (i + 0.5) * w / pts
        poly(ctx, [(px - w / pts * 0.42, y - b), (px, y - h), (px + w / pts * 0.42, y - b)])
        fill(ctx, GOLDC)
        circle(ctx, px, y - h, w * 0.06 + 0.4)
        fill(ctx, WHITE)
    ctx.rectangle(x - w / 2, y - b, w, b)
    stroke(ctx, S_["gold_d"], max(0.5, w * 0.05))
    if gems:
        for i in range(pts):
            gx = x - w / 2 + (i + 0.5) * w / pts
            circle(ctx, gx, y - b * 0.5, max(0.8, w * 0.07))
            fill(ctx, (S_["porphyry_l"], S_["malachite"], S_["lapis_l"])[i % 3])


# ------------------------------------------------------------------ the colossal face (target image of the card stunt)
def paint_face(ctx):
    """HVMANITAS: a crowned Byzantine face, drawn bold enough to read when rebuilt from thousands of tiny people."""
    cx, cy = FACE_C
    rx, ry = FACE_R
    # hair mass behind the face, falling to the shoulders
    ctx.new_path()
    smooth(ctx, [(cx - 186, 334), (cx - 222, 392), (cx - 232, 500), (cx - 222, 606), (cx - 204, 694),
                 (cx - 178, 772), (cx - 120, 804), (cx + 120, 804), (cx + 178, 772), (cx + 204, 694),
                 (cx + 222, 606), (cx + 232, 500), (cx + 222, 392), (cx + 186, 334)], close=True)
    fill(ctx, HAIR)
    for side in (-1, 1):             # a few lighter locks
        for k in range(3):
            ctx.new_path()
            x0 = cx + side * (178 + k * 12)
            smooth(ctx, [(x0, 350), (x0 + side * 12, 470), (x0 + side * 6, 600), (x0 - side * 10, 720)])
            stroke(ctx, HAIR_L, 7)
    # neck
    ctx.new_path()
    poly(ctx, [(cx - 62, 690), (cx + 62, 690), (cx + 70, 830), (cx - 70, 830)])
    fill(ctx, FLESH)
    ctx.new_path()
    poly(ctx, [(cx + 20, 700), (cx + 62, 690), (cx + 70, 830), (cx + 30, 830)])
    fill(ctx, FLESH_S, 0.55)
    ctx.new_path()
    smooth(ctx, [(cx - 50, 772), (cx, 784), (cx + 50, 772)])
    stroke(ctx, FLESH_S, 8)
    # face oval
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(rx, ry)
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()
    fill(ctx, FLESH)
    # olive shadow round the contour (proplasmos), highlight in the centre
    g = cairo.RadialGradient(cx, cy - 20, 40, cx, cy + 10, rx * 1.18)
    g.add_color_stop_rgba(0, *FLESH_L, 1)
    g.add_color_stop_rgba(0.55, *FLESH, 1)
    g.add_color_stop_rgba(0.86, *FLESH_S, 1)
    g.add_color_stop_rgba(1.0, *OLIVE, 1)
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(rx, ry)
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()
    ctx.set_source(g)
    ctx.fill()
    # cheeks
    for side in (-1, 1):
        g = cairo.RadialGradient(cx + side * 88, 606, 4, cx + side * 88, 606, 46)
        g.add_color_stop_rgba(0, *ROSE, 0.85)
        g.add_color_stop_rgba(1, *ROSE, 0.0)
        circle(ctx, cx + side * 88, 606, 46)
        ctx.set_source(g)
        ctx.fill()
    # hair over the temples (the crown hides the parting)
    for side in (-1, 1):
        ctx.new_path()
        smooth(ctx, [(cx + side * 40, 334), (cx + side * 120, 352), (cx + side * 170, 420), (cx + side * 182, 520),
                     (cx + side * 206, 520), (cx + side * 204, 334)], close=True)
        fill(ctx, HAIR)
    # jaw line shadow
    ctx.new_path()
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(rx, ry)
    ctx.arc(0, 0, 1, math.radians(25), math.radians(155))
    ctx.restore()
    stroke(ctx, FLESH_S, 10)
    # brows: long arches joined to the nose
    for i, (ex, ey) in enumerate(EYES):
        side = -1 if i == 0 else 1
        ctx.new_path()
        smooth(ctx, [(cx + side * 14, 474), (ex - side * 14, 444), (ex + side * 26, 440), (ex + side * 66, 462)])
        stroke(ctx, HAIR, 14)
    # eyes
    for i, (ex, ey) in enumerate(EYES):
        side = -1 if i == 0 else 1
        w2, h2 = EYE_W / 2, EYE_H / 2
        # shadow under the eye
        ctx.new_path()
        smooth(ctx, [(ex - w2, ey + 4), (ex, ey + h2 + 18), (ex + w2, ey + 4)])
        stroke(ctx, OLIVE, 9, 0.7)
        # almond
        ctx.new_path()
        ctx.move_to(ex - w2, ey)
        ctx.curve_to(ex - w2 * 0.5, ey - h2 * 1.25, ex + w2 * 0.5, ey - h2 * 1.25, ex + w2, ey + 2)
        ctx.curve_to(ex + w2 * 0.5, ey + h2 * 1.05, ex - w2 * 0.5, ey + h2 * 1.05, ex - w2, ey)
        ctx.close_path()
        ctx.save()
        fill(ctx, WHITE)
        ctx.restore()
        # iris + pupil, looking slightly toward the viewer's centre
        ix = ex - side * 4
        ctx.save()
        ctx.new_path()
        ctx.move_to(ex - w2, ey)
        ctx.curve_to(ex - w2 * 0.5, ey - h2 * 1.25, ex + w2 * 0.5, ey - h2 * 1.25, ex + w2, ey + 2)
        ctx.curve_to(ex + w2 * 0.5, ey + h2 * 1.05, ex - w2 * 0.5, ey + h2 * 1.05, ex - w2, ey)
        ctx.close_path()
        ctx.clip()
        circle(ctx, ix, ey + 1, 25)
        fill(ctx, IRIS)
        circle(ctx, ix, ey + 1, 12.5)
        fill(ctx, PUPIL)
        ctx.restore()
        circle(ctx, ix + 7, ey - 7, 3.0)
        fill(ctx, FLESH_H, 0.85)
        # heavy upper lid line + lower lid
        ctx.new_path()
        ctx.move_to(ex - w2 - 4, ey + 1)
        ctx.curve_to(ex - w2 * 0.5, ey - h2 * 1.3, ex + w2 * 0.5, ey - h2 * 1.3, ex + w2 + 6, ey + 4)
        stroke(ctx, LID, 13)
        ctx.new_path()
        ctx.move_to(ex - w2 + 6, ey + 5)
        ctx.curve_to(ex - w2 * 0.4, ey + h2 * 1.05, ex + w2 * 0.4, ey + h2 * 1.05, ex + w2 - 2, ey + 6)
        stroke(ctx, FLESH_S, 5)
        # lid crease
        ctx.new_path()
        ctx.move_to(ex - w2 * 0.7, ey - h2 - 6)
        ctx.curve_to(ex - w2 * 0.2, ey - h2 * 1.7, ex + w2 * 0.4, ey - h2 * 1.7, ex + w2 * 0.8, ey - h2 - 2)
        stroke(ctx, FLESH_S, 4)
    # nose: long, straight, shadowed on one side, a highlight ridge
    ctx.new_path()
    smooth(ctx, [(cx - 14, 470), (cx - 16, 540), (cx - 22, 598), (cx - 30, 616)])
    stroke(ctx, FLESH_S, 10)
    ctx.new_path()
    smooth(ctx, [(cx + 4, 486), (cx + 6, 560), (cx + 4, 598)])
    stroke(ctx, FLESH_H, 8)
    ctx.new_path()
    smooth(ctx, [(cx - 30, 618), (cx - 16, 628), (cx, 624), (cx + 16, 628), (cx + 30, 618)])
    stroke(ctx, LID, 8)
    for side in (-1, 1):
        circle(ctx, cx + side * 14, 622, 5)
        fill(ctx, LID)
    # mouth: small, full, dark upper lip
    ctx.new_path()
    ctx.move_to(cx - 34, MOUTH_Y)
    ctx.curve_to(cx - 18, MOUTH_Y - 16, cx - 6, MOUTH_Y - 10, cx, MOUTH_Y - 6)
    ctx.curve_to(cx + 6, MOUTH_Y - 10, cx + 18, MOUTH_Y - 16, cx + 34, MOUTH_Y)
    ctx.curve_to(cx + 14, MOUTH_Y + 4, cx - 14, MOUTH_Y + 4, cx - 34, MOUTH_Y)
    fill(ctx, LIP_U)
    ctx.new_path()
    ctx.move_to(cx - 30, MOUTH_Y + 1)
    ctx.curve_to(cx - 16, MOUTH_Y + 22, cx + 16, MOUTH_Y + 22, cx + 30, MOUTH_Y + 1)
    ctx.curve_to(cx + 12, MOUTH_Y + 5, cx - 12, MOUTH_Y + 5, cx - 30, MOUTH_Y + 1)
    fill(ctx, LIP_L)
    ctx.new_path()
    ctx.move_to(cx - 36, MOUTH_Y)
    ctx.curve_to(cx - 12, MOUTH_Y + 5, cx + 12, MOUTH_Y + 5, cx + 36, MOUTH_Y)
    stroke(ctx, LID, 5)
    # chin highlight and a shadow under the lower lip
    ctx.new_path()
    smooth(ctx, [(cx - 20, MOUTH_Y + 36), (cx, MOUTH_Y + 30), (cx + 20, MOUTH_Y + 36)])
    stroke(ctx, FLESH_S, 7)
    ellipse(ctx, cx, MOUTH_Y + 62, 26, 12)
    fill(ctx, FLESH_H, 0.6)
    # forehead highlight
    ellipse(ctx, cx, 392, 70, 22)
    fill(ctx, FLESH_H, 0.45)


def paint_collar_robe(ctx):
    cx = FACE_C[0]
    # robe: purple, shoulders sloping to the bottom border
    ctx.new_path()
    smooth(ctx, [(cx - 150, 800), (cx - 300, 830), (cx - 440, 880), (cx - 560, 980), (cx - 620, 1090),
                 (cx + 620, 1090), (cx + 560, 980), (cx + 440, 880), (cx + 300, 830), (cx + 150, 800)], close=True)
    fill(ctx, PURPLE)
    # folds
    for k in range(-5, 6):
        if k == 0:
            continue
        ctx.new_path()
        x0 = cx + k * 92
        smooth(ctx, [(x0 * 0.9 + cx * 0.1, 900), (x0 + k * 10, 990), (x0 + k * 18, 1090)])
        stroke(ctx, PURPLE_D if k % 2 else PURPLE_L, 12)
    # gold clavi / tablion stripes
    for side in (-1, 1):
        ctx.new_path()
        poly(ctx, [(cx + side * 150, 900), (cx + side * 186, 900), (cx + side * 214, 1090), (cx + side * 170, 1090)])
        fill(ctx, GOLDC)
    # the jewelled collar (maniakion): a broad crescent of gold with pearls and gems
    ctx.new_path()
    smooth(ctx, [(cx - 250, 812), (cx - 150, 870), (cx, 902), (cx + 150, 870), (cx + 250, 812),
                 (cx + 150, 818), (cx, 838), (cx - 150, 818)], close=True)
    fill(ctx, GOLDC)
    for i in range(-9, 10):
        a = i / 9.0
        x = cx + a * 215
        y = 842 + 42 * (1 - a * a)
        circle(ctx, x, y, 8.5)
        fill(ctx, (S_["porphyry_l"], S_["malachite"], S_["lapis_l"])[(i + 30) % 3])
    for i in range(-14, 15):
        a = i / 14.0
        x = cx + a * 238
        y = 816 + 24 * (1 - a * a) + 2
        circle(ctx, x, y, 5.5)
        fill(ctx, WHITE)
        x = cx + a * 205
        y = 866 + 38 * (1 - a * a)
        circle(ctx, x, y, 5.5)
        fill(ctx, WHITE)


def face_image():
    """the target picture of the card stunt (world, K px/wu) as float32 rgb + coverage alpha."""
    sf = Surf()
    ctx = sf.ctx
    paint_face(ctx)
    paint_collar_robe(ctx)
    b = sf.u8()
    a = b[..., 3].astype(np.float32) / 255.0
    rgb = b[..., 2::-1].astype(np.float32) / 255.0
    rgb = rgb / np.maximum(a[..., None], 1e-3)
    return np.clip(rgb, 0, 1), a


# ------------------------------------------------------------------ the card stunt: tiny people
def bust_cells():
    """figure cells over the bust: list of (cx, cy, w, h). Smaller people in the face, larger in the robe."""
    cells = []
    rng = np.random.default_rng(42)
    fa = None
    for (y0, y1, cw, ch) in ((334.0, 812.0, 10.0, 15.0), (812.0, 1096.0, 13.0, 19.0)):
        row = 0
        y = y0 + ch / 2
        while y < y1:
            off = (cw / 2) if row % 2 else 0.0
            x = 290.0 + off
            while x < 1630.0:
                cells.append((x + rng.uniform(-0.6, 0.6), y + rng.uniform(-0.6, 0.6), cw, ch))
                x += cw
            y += ch
            row += 1
    return np.array(cells, np.float32)


CARD = (0.19, -0.07, 0.26)      # card half-width (x w), top and bottom (x h) relative to the cell centre


def card_rect(x, y, w, h):
    """the card each tiny person holds - one tessera, later a phone. (cx, cy, width, height)"""
    return x, y + h * (CARD[1] + CARD[2]) / 2, 2 * CARD[0] * w, (CARD[2] - CARD[1]) * h


def figure_path(ctx, x, y, w, h, part):
    """one tiny person in a cell centred (x, y): 'body' | 'head' | 'hair' | 'card'."""
    if part == "head":
        circle(ctx, x, y - h * 0.285, w * 0.27)
    elif part == "hair":
        ctx.new_sub_path()
        ctx.arc(x, y - h * 0.29, w * 0.28, math.pi * 0.92, math.pi * 2.08)
        ctx.close_path()
    elif part == "body":
        ctx.move_to(x - w * 0.41, y + h * 0.48)
        ctx.curve_to(x - w * 0.42, y + h * 0.10, x - w * 0.42, y - h * 0.10, x - w * 0.18, y - h * 0.14)
        ctx.line_to(x + w * 0.18, y - h * 0.14)
        ctx.curve_to(x + w * 0.42, y - h * 0.10, x + w * 0.42, y + h * 0.10, x + w * 0.41, y + h * 0.48)
        ctx.close_path()
    elif part == "card":
        cx_, cy_, cw_, ch_ = card_rect(x, y, w, h)
        ctx.rectangle(cx_ - cw_ / 2, cy_ - ch_ / 2, cw_, ch_)


def paint_card_stunt(ctx, cells, face_rgb, face_a, skin_seed=3):
    """paint every tiny person in the colour of the picture where it stands. Returns per-cell colour (n,3)."""
    n = len(cells)
    # the picture colour per cell = mean over the cell
    cols = np.zeros((n, 3), np.float32)
    covs = np.zeros(n, np.float32)
    for i, (x, y, w, h) in enumerate(cells):
        x0, y0 = int((x - w * 0.45 - X0) * K), int((y - h * 0.45 - Y0) * K)
        x1, y1 = int((x + w * 0.45 - X0) * K), int((y + h * 0.45 - Y0) * K)
        a = face_a[y0:y1, x0:x1]
        covs[i] = a.mean() if a.size else 0
        if covs[i] > 0.02:
            cols[i] = (face_rgb[y0:y1, x0:x1] * a[..., None]).sum((0, 1)) / max(1e-3, a.sum())
    keep = covs > 0.12
    skins = [rgbx("#E9C3A6"), rgbx("#C98E6B"), rgbx("#8D5A3F"), rgbx("#5C3A2A")]
    hairs = [rgbx("#2A1A12"), rgbx("#4A2C18"), rgbx("#7A5230"), rgbx("#9A9690")]
    rng = np.random.default_rng(skin_seed)
    for i in np.nonzero(keep)[0]:
        x, y, w, h = cells[i]
        L = cols[i]
        lum = float(L @ np.array([0.3, 0.59, 0.11]))
        ctx.rectangle(x - w / 2, y - h / 2, w, h)
        fill(ctx, tuple(L * 0.36))
        figure_path(ctx, x, y, w, h, "body")
        fill(ctx, tuple(np.clip(L * 1.08, 0, 1)))
        figure_path(ctx, x, y, w, h, "body")
        stroke(ctx, tuple(L * 0.3), w * 0.07)
        sk = np.array(skins[rng.integers(0, 4)], np.float32)
        head = L * 0.6 + sk * 0.4 if lum > 0.22 else L * 0.78 + sk * 0.22
        figure_path(ctx, x, y, w, h, "head")
        fill(ctx, tuple(np.clip(head, 0, 1)))
        hr = np.array(hairs[rng.integers(0, 4)], np.float32)
        figure_path(ctx, x, y, w, h, "hair")
        fill(ctx, tuple(np.clip(L * 0.45 + hr * 0.55, 0, 1)))
        figure_path(ctx, x, y, w, h, "card")
        fill(ctx, tuple(np.clip(L * 1.16 + 0.03, 0, 1)))
    return cols, keep


def figure_part(xy, cells, cell_idx):
    """part id of tiles (0 ground, 1 body, 2 head, 3 card) given their cell."""
    c = cells[cell_idx]
    dx = (xy[:, 0] - c[:, 0]) / c[:, 2]
    dy = (xy[:, 1] - c[:, 1]) / c[:, 3]
    part = np.zeros(len(xy), np.int8)
    body = (np.abs(dx) < 0.41) & (dy > -0.14) & (dy < 0.48)
    part[body] = 1
    card = (np.abs(dx) < CARD[0]) & (dy > CARD[1]) & (dy < CARD[2])
    part[card] = 3
    hd = (dx * c[:, 2]) ** 2 + ((dy + 0.285) * c[:, 3]) ** 2 < (0.27 * c[:, 2]) ** 2
    part[hd] = 2
    return part


# each tiny person is a designed micro-mosaic (a glyph of ~14 tesserae) in a 10 x 15 cell:
#   (dx, dy, size u, aspect v/u, angle, part)   parts: 1 body, 2 head, 3 card (the phone-to-be), 4 hair, 5 hands
GLYPH = [
    (0.0, -6.35, 4.3, 0.36, 0.0, 4),
    (0.0, -4.15, 3.7, 0.9, 0.0, 2),
    (-2.95, -1.35, 2.5, 0.72, 0.3, 1),
    (2.95, -1.35, 2.5, 0.72, -0.3, 1),
    (-3.15, 0.95, 2.2, 1.0, 0.0, 1),
    (-3.2, 3.3, 2.2, 1.0, 0.0, 1),
    (-3.25, 5.65, 2.2, 1.0, 0.0, 1),
    (3.15, 0.95, 2.2, 1.0, 0.0, 1),
    (3.2, 3.3, 2.2, 1.0, 0.0, 1),
    (3.25, 5.65, 2.2, 1.0, 0.0, 1),
    (0.0, 1.45, 3.5, 1.3, 0.0, 3),
    (0.0, 5.55, 3.4, 0.62, 0.0, 1),
    (-1.95, 3.35, 1.15, 1.0, 0.0, 5),
    (1.95, 3.35, 1.15, 1.0, 0.0, 5),
]
SKINS = [rgbx("#E9C3A6"), rgbx("#C98E6B"), rgbx("#8D5A3F"), rgbx("#5C3A2A")]
HAIRS = [rgbx("#2A1A12"), rgbx("#4A2C18"), rgbx("#7A5230"), rgbx("#9A9690")]


def glyph_tiles(cells, keep, cols, seed=5):
    """the tesserae of every tiny person: dict(xy, ang, size, aspect, part, ci, rgb)."""
    rng = np.random.default_rng(seed)
    G = np.array(GLYPH, np.float32)
    idx = np.nonzero(keep)[0]
    m, g = len(idx), len(G)
    c = cells[idx]
    sx = (c[:, 2] / 10.0)[:, None]
    sy = (c[:, 3] / 15.0)[:, None]
    xy = np.stack([c[:, 0:1] + G[None, :, 0] * sx, c[:, 1:2] + G[None, :, 1] * sy], -1).reshape(-1, 2)
    size = (G[None, :, 2] * sx).reshape(-1)
    aspect = np.broadcast_to(G[None, :, 3] * (sy / sx), (m, g)).reshape(-1)
    ang = np.broadcast_to(G[None, :, 4] + rng.normal(0, 0.035, (m, g)), (m, g)).reshape(-1)
    part = np.broadcast_to(G[None, :, 5], (m, g)).reshape(-1).astype(np.int8)
    ci = np.repeat(idx, g).astype(np.int32)
    L = cols[idx]
    lum = L @ np.array([0.3, 0.59, 0.11], np.float32)
    sk = np.array(SKINS, np.float32)[rng.integers(0, 4, m)]
    hr = np.array(HAIRS, np.float32)[rng.integers(0, 4, m)]
    head = np.where((lum > 0.22)[:, None], L * 0.55 + sk * 0.45, L * 0.78 + sk * 0.22)
    hair = L * 0.4 + hr * 0.6
    body = np.clip(L * 1.06, 0, 1)
    card = np.clip(L * 1.18 + 0.03, 0, 1)
    hand = np.where((lum > 0.22)[:, None], sk * 0.9, sk * 0.5 + L * 0.3)
    per = {1: body, 2: head, 3: card, 4: hair, 5: hand}
    rgb = np.zeros((m, g, 3), np.float32)
    for j in range(g):
        rgb[:, j] = per[int(G[j, 5])]
    rgb *= (1.0 + rng.normal(0, 0.035, (m, g, 1))).astype(np.float32)
    return dict(xy=xy.astype(np.float32), ang=ang.astype(np.float32), size=size.astype(np.float32),
                aspect=aspect.astype(np.float32), part=part, ci=ci, rgb=np.clip(rgb.reshape(-1, 3), 0, 1))


# ------------------------------------------------------------------ the crown of machines
def paint_band(ctx, tit_mask_ctx=None):
    x0, x1 = CROWN_X
    y0, y1 = BAND_Y
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    fill(ctx, S_["lapis"])
    ctx.rectangle(x0, y0, x1 - x0, 4)
    fill(ctx, GOLDC)
    ctx.rectangle(x0, y1 - 4, x1 - x0, 4)
    fill(ctx, GOLDC)
    for yy in (y0 + 7.5, y1 - 7.5):
        x = x0 + 5
        while x < x1 - 3:
            circle(ctx, x, yy, 3.1)
            fill(ctx, WHITE)
            x += 8.4
    # gems between the tituli and at the ends
    for gx in (x0 + 10, 860.0, 1060.0, x1 - 10):
        ellipse(ctx, gx, (y0 + y1) / 2, 7.5, 10.5)
        fill(ctx, S_["porphyry_l"])
        ellipse(ctx, gx - 1.8, (y0 + y1) / 2 - 3, 2.2, 3.0)
        fill(ctx, WHITE, 0.8)
        for dx, dy in ((-11, -9), (11, -9), (-11, 9), (11, 9)):
            circle(ctx, gx + dx, (y0 + y1) / 2 + dy, 2.6)
            fill(ctx, WHITE)
    # the tituli themselves live in a separate mask (they set on their words); here the cartouches only
    for word, x in TITULI:
        pass


TIT_MAXW = 148.0            # between the jewels of the band (each machine section is 200 wide)


def _tit_fit(ctx, word):
    """font size + tracking so that the word fits its section of the band."""
    tr = 0.06 if len(word) > 8 else 0.14
    size = TIT_H * 1.36
    ctx.select_font_face("P052", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    w_ = sum(ctx.text_extents(ch).x_advance for ch in word) + tr * size * (len(word) - 1)
    if w_ > TIT_MAXW:
        size *= TIT_MAXW / w_
    return size, tr


def paint_tituli(ctx, colour=WHITE):
    for word, x in TITULI:
        ctx.new_path()
        size, tr = _tit_fit(ctx, word)
        text_path(ctx, word, x, TIT_Y - (TIT_H * 1.36 - size) * 0.36, size, tracking=tr)
        fill(ctx, colour)


def titulus_letters():
    """per-letter boxes of the three tituli: list of (word_index, letter_index, x0, x1)."""
    sf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 8, 8)
    ctx = cairo.Context(sf)
    out = []
    for wi, (word, x) in enumerate(TITULI):
        size, tr = _tit_fit(ctx, word)
        xs, ws = text_path(ctx, word, x, TIT_Y, size, tracking=tr)
        ctx.new_path()
        for li, (a, w_) in enumerate(zip(xs, ws)):
            out.append((wi, li, a, a + w_))
    return out


def paint_city(ctx):
    xa, xb = MACHINE_X[0]
    # houses and roofs behind the wall
    rng = np.random.default_rng(11)
    for i in range(9):
        hx = xa + 12 + i * 20 + rng.uniform(-3, 3)
        top = 176 + rng.uniform(-14, 10)
        ctx.rectangle(hx - 9, top, 18, 240 - top)
        fill(ctx, (S_["bone"], S_["marble_d"], S_["terracotta"])[i % 3])
        poly(ctx, [(hx - 11, top), (hx, top - 11), (hx + 11, top)])
        fill(ctx, (S_["ochre_red"], S_["lapis_l"], S_["porphyry"])[i % 3])
        for wy in (top + 6, top + 16):
            ctx.rectangle(hx - 2.5, wy, 5, 6)
            fill(ctx, S_["ink"])
    for dxh in (-50.0, 52.0):
        circle(ctx, 760 + dxh, 170, 11)
        fill(ctx, GOLDC)
        ctx.rectangle(760 + dxh - 11, 170, 22, 40)
        fill(ctx, S_["marble_d"])
    # curtain wall
    ctx.rectangle(xa + 4, 222, xb - xa - 8, 282 - 222)
    fill(ctx, S_["bone"])
    for yy in range(230, 282, 8):
        ctx.move_to(xa + 4, yy)
        ctx.line_to(xb - 4, yy)
        stroke(ctx, S_["marble_d"], 0.9)
    x = xa + 6
    while x < xb - 8:
        ctx.rectangle(x, 214, 8, 9)
        fill(ctx, S_["bone"])
        ctx.rectangle(x, 214, 8, 9)
        stroke(ctx, S_["marble_d"], 0.8)
        x += 14
    # towers with crowns
    for tx, top in ((684.0, 150.0), (718.0, 136.0), (802.0, 136.0), (836.0, 150.0)):
        ctx.rectangle(tx - 13, top, 26, 282 - top)
        fill(ctx, S_["marble"])
        ctx.rectangle(tx + 4, top, 9, 282 - top)
        fill(ctx, S_["marble_d"])
        ctx.rectangle(tx - 13, top, 26, 282 - top)
        stroke(ctx, S_["slate"], 1.3)
        for k in range(3):
            ctx.rectangle(tx - 13 + k * 10, top - 7, 6, 7)
            fill(ctx, S_["marble"])
        for wy in range(int(top) + 14, 262, 22):
            ctx.new_path()
            ctx.move_to(tx - 3, wy + 10)
            ctx.line_to(tx - 3, wy + 3)
            ctx.arc(tx, wy + 3, 3, math.pi, 0)
            ctx.line_to(tx + 3, wy + 10)
            ctx.close_path()
            fill(ctx, S_["ink"])
        crown_icon(ctx, tx, top - 8, 22, 16)
    # the gatehouse: a machine wall of three cogs above the gate
    ctx.rectangle(728, 122, 64, 160)
    fill(ctx, S_["marble_d"])
    ctx.rectangle(728, 122, 64, 160)
    stroke(ctx, S_["slate"], 1.5)
    ctx.rectangle(732, 146, 56, 98)
    fill(ctx, S_["lapis_d"])              # the dark machine chamber behind the cogs
    for k in range(4):
        ctx.rectangle(728 + k * 18, 114, 10, 8)
        fill(ctx, S_["marble_d"])
    crown_icon(ctx, 760, 112, 46, 30, pts=5)
    # pendilia of the big crown
    for side in (-1, 1):
        for j in range(3):
            circle(ctx, 760 + side * 21, 116 + j * 6, 1.8)
            fill(ctx, WHITE)
    # gate
    ctx.new_path()
    ctx.move_to(744, 282)
    ctx.line_to(744, 256)
    ctx.arc(760, 256, 16, math.pi, 0)
    ctx.line_to(776, 282)
    ctx.close_path()
    fill(ctx, S_["ink"])
    ctx.new_path()
    ctx.arc(760, 256, 19, math.pi, 0)
    stroke(ctx, S_["terracotta"], 3)
    circle(ctx, 760, 262, 2.5)
    fill(ctx, S_["gold_l"])


def paint_temple(ctx):
    cx = 960.0
    dcx, dcy, dr = DOME
    # side semi-domes
    for side in (-1, 1):
        sx = cx + side * 68
        ctx.new_path()
        ctx.arc(sx, 196, 36, math.pi, 0)
        ctx.close_path()
        fill(ctx, GOLDC)
        ctx.new_path()
        ctx.arc(sx, 196, 36, math.pi, 0)
        stroke(ctx, S_["gold_d"], 2)
    # drum with windows
    ctx.rectangle(dcx - 58, dcy, 116, 22)
    fill(ctx, S_["marble"])
    for k in range(7):
        wx = dcx - 48 + k * 16
        ctx.new_path()
        ctx.move_to(wx - 3.5, dcy + 18)
        ctx.line_to(wx - 3.5, dcy + 8)
        ctx.arc(wx, dcy + 8, 3.5, math.pi, 0)
        ctx.line_to(wx + 3.5, dcy + 18)
        ctx.close_path()
        fill(ctx, S_["ink"])
    # the dome (its ribs are animated; here the gold shell)
    ctx.new_path()
    ctx.arc(dcx, dcy, dr, math.pi, 0)
    ctx.close_path()
    fill(ctx, GOLDC)
    ctx.new_path()
    ctx.arc(dcx, dcy, dr, math.pi, 0)
    stroke(ctx, S_["gold_d"], 2.5)
    ctx.rectangle(dcx - dr - 3, dcy - 3, 2 * dr + 6, 5)
    fill(ctx, S_["porphyry"])
    circle(ctx, dcx, dcy - dr - 6, 5.5)
    fill(ctx, GOLDC)
    circle(ctx, dcx, dcy - dr - 6, 5.5)
    stroke(ctx, S_["gold_d"], 1)
    # main body
    ctx.rectangle(cx - 98, 182, 196, 100)
    fill(ctx, S_["marble"])
    ctx.rectangle(cx - 98, 182, 196, 100)
    stroke(ctx, S_["slate"], 1.5)
    # cosmati band
    for k in range(24):
        x = cx - 96 + k * 8
        poly(ctx, [(x, 192), (x + 4, 188), (x + 8, 192), (x + 4, 196)])
        fill(ctx, (S_["porphyry"], S_["malachite"], GOLDC)[k % 3])
    ctx.rectangle(cx - 98, 186, 196, 2)
    fill(ctx, S_["slate"])
    ctx.rectangle(cx - 98, 196, 196, 2)
    fill(ctx, S_["slate"])
    # gable over the door holding the cog
    poly(ctx, [(cx - 30, 234), (cx, 196), (cx + 30, 234)])
    fill(ctx, S_["lapis_d"])
    poly(ctx, [(cx - 30, 234), (cx, 196), (cx + 30, 234)])
    stroke(ctx, GOLDC, 2)
    # columns and arches
    for k, ax in enumerate((cx - 74, cx - 44, cx + 44, cx + 74)):
        ctx.rectangle(ax - 3, 206, 6, 76)
        fill(ctx, S_["porphyry"])
        ctx.rectangle(ax - 4.5, 204, 9, 3)
        fill(ctx, GOLDC)
    for ax in (cx - 59, cx + 59):
        ctx.new_path()
        ctx.move_to(ax - 11, 282)
        ctx.line_to(ax - 11, 226)
        ctx.arc(ax, 226, 11, math.pi, 0)
        ctx.line_to(ax + 11, 282)
        ctx.close_path()
        fill(ctx, S_["lapis_d"])
        ctx.rectangle(ax - 5, 234, 10, 18)
        fill(ctx, S_["gold_l"], 0.8)
    for ax in (cx - 88, cx + 88):
        ctx.rectangle(ax - 5, 214, 10, 22)
        fill(ctx, S_["ink"])
    # the door
    ctx.new_path()
    ctx.move_to(cx - 18, 282)
    ctx.line_to(cx - 18, 252)
    ctx.arc(cx, 252, 18, math.pi, 0)
    ctx.line_to(cx + 18, 282)
    ctx.close_path()
    fill(ctx, S_["ink"])
    ctx.new_path()
    ctx.arc(cx, 252, 21, math.pi, 0)
    stroke(ctx, GOLDC, 3)
    # a lamp in the door
    ctx.move_to(cx, 236)
    ctx.line_to(cx, 250)
    stroke(ctx, GOLDC, 0.8)
    circle(ctx, cx, 253, 3.2)
    fill(ctx, S_["gold_l"])
    # steps
    for k in range(3):
        ctx.rectangle(cx - 100 - k * 4, 276 + k * 2, 200 + k * 8, 2)
        fill(ctx, S_["marble_d"])


def paint_philosophers(ctx):
    xa, xb = MACHINE_X[2]
    # distant acropolis
    ctx.rectangle(xa + 4, 206, xb - xa - 8, 30)
    fill(ctx, S_["marble_d"])
    for k in range(10):
        ctx.rectangle(xa + 8 + k * 18, 196, 12, 12)
        fill(ctx, S_["bone"])
        poly(ctx, [(xa + 6 + k * 18, 196), (xa + 14 + k * 18, 189), (xa + 22 + k * 18, 196)])
        fill(ctx, S_["terracotta"])
    # tree: twisting trunk + canopy of leaves (the cogs hang in it)
    tx, ty, tr = TREE
    ctx.new_path()
    smooth(ctx, [(tx - 10, 282), (tx - 6, 240), (tx - 12, 205), (tx - 4, 170), (tx + 4, 170), (tx + 6, 205),
                 (tx + 12, 240), (tx + 12, 282)], close=True)
    fill(ctx, WOOD)
    ctx.new_path()
    smooth(ctx, [(tx + 2, 282), (tx + 4, 240), (tx - 2, 205), (tx + 2, 175)])
    stroke(ctx, WOOD_D, 2.5)
    for a, L in ((-2.3, 42), (-0.8, 44), (-1.6, 30)):
        ctx.move_to(tx, 182)
        ctx.line_to(tx + L * math.cos(a), 182 + L * math.sin(a))
        stroke(ctx, WOOD, 4)
    rng = np.random.default_rng(5)
    for i in range(120):
        a = rng.uniform(0, 2 * math.pi)
        r = tr * math.sqrt(rng.uniform(0, 1))
        x = tx + r * math.cos(a) * 1.15
        y = ty + r * math.sin(a) * 0.85
        c = (LEAF, LEAF_D, GREEN_L, S_["verdigris"])[i % 4]
        ellipse(ctx, x, y, 9 + rng.uniform(0, 5), 6 + rng.uniform(0, 3), rng.uniform(0, 3))
        fill(ctx, c)
    # sundial on a column
    sx, sy, sr = SUNDIAL
    ctx.rectangle(sx - 4.5, sy, 9, 282 - sy)
    fill(ctx, S_["marble"])
    ctx.rectangle(sx - 7, 276, 14, 6)
    fill(ctx, S_["marble_d"])
    ctx.new_path()
    ctx.arc(sx, sy, sr, 0, math.pi)
    ctx.close_path()
    fill(ctx, S_["bone"])
    ctx.new_path()
    ctx.arc(sx, sy, sr, 0, math.pi)
    stroke(ctx, S_["slate"], 1.2)
    # the globe's box
    gx, gy, gr = GLOBE
    ctx.rectangle(gx - 15, 266, 30, 16)
    fill(ctx, WOOD)
    ctx.rectangle(gx - 15, 266, 30, 16)
    stroke(ctx, WOOD_D, 1.2)
    # the bench
    ctx.new_path()
    ctx.move_to(xa + 12, 270)
    ctx.curve_to(xa + 60, 262, xb - 60, 262, xb - 12, 270)
    stroke(ctx, S_["marble_d"], 7)


# the seven philosophers of the Academy (vx.icon), feet on the bench line, gesturing at globe and sky
PHILOSOPHERS = [
    # x, pose, seed
    (1093.0, "speak", 11),
    (1113.0, "stand", 12),
    (1134.0, "point", 13),
    (1186.0, "object", 14),
    (1207.0, "speak", 15),
    (1227.0, "stand", 16),
    (1246.0, "bless", 17),
]
PHIL_H = 46.0


def paint_philosopher_figures(ctx, layer="color"):
    icon = _icon()
    for x, pose, sd in PHILOSOPHERS:
        icon.Figure("philosopher", seed=sd).draw(ctx, x, 280.5, PHIL_H, pose=pose, layer=layer)


def paint_pendilia(ctx):
    for side in (-1, 1):
        for j in range(3):
            x = (CROWN_X[0] + 14 + j * 11) if side < 0 else (CROWN_X[1] - 14 - j * 11)
            L = 190 + 22 * j
            y = BAND_Y[1] + 5
            while y < BAND_Y[1] + L:
                circle(ctx, x, y, 3.5)
                fill(ctx, WHITE)
                y += 8.6
            ctx.new_path()
            ctx.move_to(x, y + 4)
            ctx.curve_to(x - 7, y + 14, x - 5, y + 24, x, y + 26)
            ctx.curve_to(x + 5, y + 24, x + 7, y + 14, x, y + 4)
            fill(ctx, S_["porphyry_l"])


def pendilia_mask(ctx):
    for side in (-1, 1):
        for j in range(3):
            x = (CROWN_X[0] + 14 + j * 11) if side < 0 else (CROWN_X[1] - 14 - j * 11)
            L = 190 + 22 * j
            ctx.rectangle(x - 5, BAND_Y[1], 10, L + 30)
    set_color(ctx, (1, 1, 1))
    ctx.fill()


# ------------------------------------------------------------------ border, lapis, halo, labels
def rosette(ctx, x, y, R, rot=0.0):
    """a hollow gold 8-fold star-rosette (m01's vault stars): a ring of 8 petals round a dark centre, 8 outer
    points between them."""
    for k in range(8):
        a = rot + k * math.pi / 4
        ellipse(ctx, x + 0.53 * R * math.cos(a), y + 0.53 * R * math.sin(a), 0.34 * R, 0.27 * R, a + math.pi / 2)
        fill(ctx, S_["star"])
        a2 = a + math.pi / 8
        c2, s2 = math.cos(a2), math.sin(a2)
        poly(ctx, [(x + 0.6 * R * c2 - 0.2 * R * s2, y + 0.6 * R * s2 + 0.2 * R * c2),
                   (x + 1.08 * R * c2, y + 1.08 * R * s2),
                   (x + 0.6 * R * c2 + 0.2 * R * s2, y + 0.6 * R * s2 - 0.2 * R * c2)])
        fill(ctx, S_["gold_l"])
    circle(ctx, x, y, 0.13 * R)
    fill(ctx, S_["star"])


def lapis_stars(rng):
    """rosette centres + radii, ~1 per 77x77 wu (230 px at the cut), alternating big / small."""
    stars = []
    tries = 0
    while tries < 40000:
        tries += 1
        x = rng.uniform(430, 1530)
        y = rng.uniform(-560, 2)
        if all((x - a) ** 2 + (y - b) ** 2 > 50 ** 2 for a, b, _ in stars):
            stars.append((x, y, 0.0))
    stars = [(x, y, 20.0 if i % 2 else 15.0) for i, (x, y, _) in enumerate(stars)]
    for x in np.arange(-190, 2120, 72):
        y = rng.uniform(-30, -6)
        if all((x - a) ** 2 + (y - b) ** 2 > 50 ** 2 for a, b, _ in stars):
            stars.append((float(x), float(y), 11.0))
    return stars


def paint_lapis(ctx, rng):
    ctx.rectangle(X0, Y0, WW, 18 - Y0)
    fill(ctx, S_["lapis"])
    stars = lapis_stars(rng)
    for x, y, R in stars:
        rosette(ctx, x, y, R, rng.uniform(0, 0.4))
    return stars


def paint_border(ctx):
    x0, x1 = X0, X0 + WW
    ctx.rectangle(x0, 18, x1 - x0, 42)
    fill(ctx, GOLDC)
    ctx.rectangle(x0, 18, x1 - x0, 4)
    fill(ctx, WHITE)
    ctx.rectangle(x0, 22, x1 - x0, 12)
    fill(ctx, S_["porphyry"])
    x = x0 + 4
    while x < x1:
        circle(ctx, x, 28, 4.0)
        fill(ctx, WHITE)
        x += 10.5
    k = 0
    x = x0 + 12
    while x < x1:
        if k % 2 == 0:
            poly(ctx, [(x, 36), (x + 8, 43), (x, 50), (x - 8, 43)])
            fill(ctx, S_["lapis_l"])
            poly(ctx, [(x, 38.5), (x + 5, 43), (x, 47.5), (x - 5, 43)])
            fill(ctx, S_["lapis"])
        else:
            ellipse(ctx, x, 43, 8.5, 5.5)
            fill(ctx, S_["porphyry_l"])
            ellipse(ctx, x - 2, 41.5, 2.5, 1.5)
            fill(ctx, WHITE, 0.8)
            for dx in (-13, 13):
                circle(ctx, x + dx, 43, 1.8)
                fill(ctx, WHITE)
        x += 26
        k += 1
    ctx.rectangle(x0, 52, x1 - x0, 4)
    fill(ctx, S_["porphyry"])
    ctx.rectangle(x0, 56, x1 - x0, 4)
    fill(ctx, WHITE)
    # the lower border (frames the wide picture)
    ctx.rectangle(x0, 1040, x1 - x0, 60)
    fill(ctx, S_["porphyry"])
    ctx.rectangle(x0, 1040, x1 - x0, 4)
    fill(ctx, WHITE)
    x = x0 + 4
    while x < x1:
        circle(ctx, x, 1050, 3.5)
        fill(ctx, WHITE)
        x += 9.5
    ctx.rectangle(x0, 1058, x1 - x0, 44)
    fill(ctx, S_["lapis_d"])


def paint_halo(ctx):
    """a radiant nimbus: red and pearl rims, gold laid in 36 rays of two values (the leaf takes their value)."""
    hx, hy, hr = HALO
    circle(ctx, hx, hy, hr + 11)
    fill(ctx, S_["porphyry"])
    circle(ctx, hx, hy, hr + 5)
    fill(ctx, WHITE)
    circle(ctx, hx, hy, hr)
    fill(ctx, rgbx("#C8993A"))
    n = 36
    for i in range(0, n, 2):
        a0 = i * 2 * math.pi / n
        a1 = (i + 1) * 2 * math.pi / n
        ctx.move_to(hx, hy)
        ctx.arc(hx, hy, hr, a0, a1)
        ctx.close_path()
        fill(ctx, rgbx("#AE7F2A"))
    circle(ctx, hx, hy, hr - 14)
    set_color(ctx, rgbx("#AE7F2A"), 0.0)


def paint_labels(ctx):
    for s, x in (("HVMA", 372.0), ("NITAS", 1548.0)):
        ctx.new_path()
        xs, ws = text_path(ctx, s, x, 505.0, 64.0, tracking=0.1)
        fill(ctx, S_["porphyry"])
        ctx.new_path()
        tw = xs[-1] + ws[-1] - xs[0]
        ctx.move_to(xs[0] + 4, 438)
        ctx.line_to(xs[0] + tw - 4, 438)
        stroke(ctx, S_["porphyry"], 5)


def _icon():
    from vx import icon
    return icon


def procession_members(P, seed):
    """the people of one procession strip: [(kind, seed, pose, x_in_period)]"""
    L, h = P["L"], P["h"]
    n = max(1, int(round(L / (h * P.get("gap_k", 0.62)))))
    kinds = P["kinds"]
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = kinds[int(rng.integers(0, len(kinds)))]
        out.append((k, int(seed * 1000 + i), P["pose"], (i + 0.5) * L / n))
    return out


def procession_strip(P, seed=0):
    """periodic strip of Byzantine figures (vx.icon) for procession P: float32 rgba (h_px, L_px, 4) + masks
    (h_px, L_px, 2) = [candle flame, face]. Gliding figures (Sant'Apollinare), moving by the tiles re-colouring."""
    icon = _icon()
    L, h = P["L"], P["h"]
    Hb = h * PROC_TOP
    wpx, hpx = int(math.ceil(L * K)), int(math.ceil(Hb * K))
    mem = procession_members(P, seed)

    def paint(layer):
        sf = cairo.ImageSurface(cairo.FORMAT_ARGB32, wpx, hpx)
        ctx = cairo.Context(sf)
        ctx.scale(K, K)
        if P["v"] < 0:
            ctx.translate(L, 0)
            ctx.scale(-1, 1)
        anchors = []
        for kind, sd, pose, x in mem:
            for wrap in (-L, 0.0, L):
                a = icon.Figure(kind, seed=sd).draw(ctx, x + wrap, Hb - 0.4, h, pose=pose, layer=layer)
                if wrap == 0.0:
                    anchors.append(a)
        sf.flush()
        b = np.ndarray((hpx, sf.get_stride() // 4, 4), np.uint8, sf.get_data())[:, :wpx].astype(np.float32) / 255
        return b, anchors

    b, anchors = paint("color")
    rgba = np.empty((hpx, wpx, 4), np.float32)
    rgba[..., 0] = b[..., 2]
    rgba[..., 1] = b[..., 1]
    rgba[..., 2] = b[..., 0]
    rgba[..., 3] = b[..., 3]
    a = rgba[..., 3:4]
    rgba[..., :3] = rgba[..., :3] / np.maximum(a, 1e-3)
    masks = [np.zeros((hpx, wpx), np.float32), np.zeros((hpx, wpx), np.float32)]
    for an in anchors:
        for key, ch, rr in (("flame", 0, 0.05), ("face", 1, 0.075)):
            if key in an:
                fx, fy = an[key][0], an[key][1]
                if P["v"] < 0:
                    fx = L - fx
                for wrap in (-L, 0.0, L):
                    cv2.circle(masks[ch], (int(round((fx + wrap) * K)), int(round(fy * K))),
                               max(1, int(round(rr * h * K))), 1.0, -1)
    return rgba, np.dstack(masks)


def paint_procession_ground(ctx):
    """the lower corner registers: a green meadow ground line under each procession."""
    for P in PROCS[3:]:
        ctx.rectangle(P["x0"], P["y"] - 2, P["x1"] - P["x0"], 22)
        fill(ctx, LEAF_D)
        x = P["x0"]
        rng = np.random.default_rng(abs(int(P["x0"])) + 9)
        while x < P["x1"]:
            circle(ctx, x, P["y"] + rng.uniform(4, 16), rng.uniform(1.5, 2.8))
            fill(ctx, (S_["porphyry_l"], WHITE, GREEN_L, S_["gold_l"])[rng.integers(0, 4)])
            x += rng.uniform(6, 14)


# ------------------------------------------------------------------ assemble
def build():
    """paint the whole wall. Returns dict of guide maps (uint8/float) + geometry."""
    rng = np.random.default_rng(2024)
    face_rgb, face_a = face_image()
    cells = bust_cells()

    col_s = Surf(bg=GOLDC)
    ctx = col_s.ctx
    gold_s = Surf()
    gctx = gold_s.ctx
    ts_s = Surf()                  # tile size encoded in the red channel (value = size*40)
    tctx = ts_s.ctx
    edge_s = Surf()                # explicit course lines
    ectx = edge_s.ctx

    def ts_fill(c_ctx_path_fn, size):
        v = min(1.0, size * 40 / 255.0)
        tctx.save()
        c_ctx_path_fn(tctx)
        tctx.set_source_rgba(v, v, v, 1)
        tctx.fill()
        tctx.restore()

    # gold ground everywhere below the border
    gctx.rectangle(X0, 60, WW, WH)
    set_color(gctx, (1, 1, 1))
    gctx.fill()
    ts_fill(lambda c: c.rectangle(X0, 60, WW, WH), 4.0)

    # lapis + stars (only where the camera looks at the start, and the thin top strip)
    stars = paint_lapis(ctx, rng)
    ts_fill(lambda c: (c.rectangle(430, Y0, 1110, 18 - Y0 + 2), c.rectangle(X0, -44, WW, 64)), 4.6)
    for x, y, R in stars:
        ts_fill(lambda c, x=x, y=y, R=R: circle(c, x, y, R * 1.08), 2.5)
        rosette(gctx, x, y, R)
    # course lines of the lapis: huge gentle arcs (ring courses round a zenith far above), a darker
    # separation course every ~50 wu (150 px at the cut), as in m01's vault
    for k in range(16):
        ectx.new_path()
        ectx.arc(960, -2300, 1800 + k * 50.0, 0, 2 * math.pi)
        stroke(ectx, (1, 1, 1), 1.0)

    # jewelled border (+ lower border)
    paint_border(ctx)
    gctx.save()
    gctx.set_operator(cairo.OPERATOR_CLEAR)
    gctx.rectangle(X0, 18, WW, 4)
    gctx.rectangle(X0, 22, WW, 12)
    gctx.rectangle(X0, 52, WW, 8)
    gctx.rectangle(X0, 1040, WW, 70)
    gctx.fill()
    gctx.restore()
    ts_fill(lambda c: c.rectangle(X0, 18, WW, 44), 2.4)
    ts_fill(lambda c: c.rectangle(X0, 1038, WW, 70), 2.8)
    for yy in (18, 60, 1040):
        ectx.move_to(X0, yy)
        ectx.line_to(X0 + WW, yy)
        stroke(ectx, (1, 1, 1), 1.0)

    # halo (behind everything of the icon)
    paint_halo(ctx)
    hx, hy, hr = HALO
    gctx.save()
    gctx.set_operator(cairo.OPERATOR_CLEAR)
    circle(gctx, hx, hy, hr + 11)
    gctx.fill()
    gctx.restore()
    circle(gctx, hx, hy, hr)
    set_color(gctx, (1, 1, 1))
    gctx.fill()
    ts_fill(lambda c: circle(c, hx, hy, hr + 12), 3.8)
    ts_fill(lambda c: (circle(c, hx, hy, hr + 12), c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD),
                       circle(c, hx, hy, hr - 1)), 2.2)
    tctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    for r in (hr, hr + 5, hr + 11):
        ectx.new_path()
        circle(ectx, hx, hy, r)
        stroke(ectx, (1, 1, 1), 1.0)

    # labels HVMA | NITAS (porphyry on gold)
    paint_labels(ctx)
    lab_s = Surf()
    paint_labels(lab_s.ctx)
    lab_a = lab_s.alpha()

    # processions in the corners: ground strips
    paint_procession_ground(ctx)
    for P in PROCS[3:]:
        ts_fill(lambda c, P=P: c.rectangle(P["x0"], P["y"] - P["h"] * PROC_TOP, P["x1"] - P["x0"],
                                           P["h"] * PROC_TOP + 22), 2.6)

    # the bust: card stunt of tiny people, clipped to the icon's smooth silhouette (people cut to fit)
    cs = Surf()
    fc_cols, keep = paint_card_stunt(cs.ctx, cells, face_rgb, face_a)
    mask = np.clip(face_a * 255 + 0.5, 0, 255).astype(np.uint8)
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, GW)
    mbuf = np.zeros((GH, stride), np.uint8)
    mbuf[:, :GW] = mask
    msurf = cairo.ImageSurface.create_for_data(mbuf, cairo.FORMAT_A8, GW, GH, stride)
    cs.surf.flush()
    ctx.save()
    ctx.identity_matrix()
    ctx.set_source_surface(cs.surf, 0, 0)
    ctx.mask_surface(msurf, 0, 0)
    ctx.restore()
    bust_cov = (face_a > 0.5)

    # crown: machines (painted over the bust top / halo)
    crown_path = lambda c: c.rectangle(CROWN_X[0], 70, CROWN_X[1] - CROWN_X[0], BAND_Y[1] - 70 + 1)
    paint_city(ctx)
    paint_temple(ctx)
    paint_philosophers(ctx)
    paint_philosopher_figures(ctx)
    paint_band(ctx)
    paint_pendilia(ctx)

    # gold mask inside the crown: gold only where the machine art is gold-coloured (decided per tile later)
    crown_s = Surf()
    cc = crown_s.ctx
    for fn in (paint_city, paint_temple, paint_philosophers, paint_philosopher_figures, paint_band):
        fn(cc)
    crown_a = crown_s.alpha()
    pend_s = Surf()
    pendilia_mask(pend_s.ctx)
    pend_a = pend_s.alpha()

    # tituli (separate mask: they set on their words)
    tit_s = Surf()
    paint_tituli(tit_s.ctx)
    tit_a = tit_s.alpha()

    rgb = col_s.rgb()
    gold = gold_s.alpha()
    ts = ts_s.u8()[..., 2].astype(np.float32) / 40.0
    # region-specific tile sizes on top of the painted map
    ts[bust_cov] = 0.0                 # the bust is set person by person (glyph_tiles), not by the flow
    ts[crown_a > 8] = 2.3
    ts[tit_a > 8] = 1.15
    ts[pend_a > 8] = 1.6
    ts[lab_a > 8] = 2.2
    # fine tiles for the small figures/faces in the crown
    fine_s = Surf()
    fcx = fine_s.ctx
    paint_philosopher_figures(fcx)
    for P in PROCS[:3]:
        fcx.rectangle(P["x0"], P["y"] - P["h"] * PROC_TOP, P["x1"] - P["x0"], P["h"] * PROC_TOP)
        set_color(fcx, (1, 1, 1))
        fcx.fill()
    fine_a = fine_s.alpha()
    ts[fine_a > 8] = 1.5
    # holes for the rotating discs (their own ring layouts)
    for d in DISCS:
        cxp, cyp = to_px(*d["c"])
        cv2.circle(ts, (int(round(cxp)), int(round(cyp))), int(math.ceil((d["r"] + 0.6) * K)), 0.0, -1)
    # nothing outside the used lapis window
    yy = (np.arange(GH) / K + Y0)[:, None]
    xx = (np.arange(GW) / K + X0)[None, :]
    ts[(yy < -44) & ((xx < 440) | (xx > 1540))] = 0

    # edges for the andamento: colour boundaries + course lines + every tiny person's outline
    q = cv2.GaussianBlur(rgb, (0, 0), 0.8)
    e = cv2.Canny(cv2.cvtColor(q, cv2.COLOR_RGB2GRAY), 18, 42) > 0
    lab = cv2.cvtColor(q, cv2.COLOR_RGB2LAB)
    for ch in (1, 2):
        e |= cv2.Canny(np.ascontiguousarray(lab[..., ch]), 10, 26) > 0
    e |= edge_s.alpha() > 60
    return dict(rgb=rgb, gold=gold, ts=ts, edges=e, crown_a=crown_a, tit_a=tit_a, pend_a=pend_a, lab_a=lab_a,
                face_rgb=None, cells=cells, cell_keep=keep, cell_cols=fc_cols, stars=np.array(stars, np.float32),
                bust_cov=bust_cov)

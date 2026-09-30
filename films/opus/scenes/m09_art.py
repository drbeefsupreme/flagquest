"""m09 helper: the cartoons of the dome of Flagistan and their baking into mosaic "sheets".

Two kinds of sheet, both baked once (cached) into the format the per-pixel kernel (m09_hyper) samples:
  * the SECTOR sheet (one per people, 8 variants): the fundamental domain of the {8,3} tiling in Poincare-disk
    coordinates zeta: oculus + medallion rings at the tile centre, a triple ray, the arched niche with its people
    standing on the tile edge (feet on a band of earth), gold ground, the vertex window. Every octagon of the
    infinite dome is 8 of these.
  * the ARCHITECTURE sheet: the plane z = 1 of the dome's sphere (w coordinates) outside the dome: springing
    cornice, the drum with 8 windows (Hagia Sophia's ring of light), the titulus ring, pendentives with seraphim,
    the four great arches, the starry barrel vaults beyond.

Sheet format (per texel): lab int32 (tessera id, -1 = mortar), rim uint8 (bevel), u/v int8 (tile-local coords);
per tessera: col, gold, tilt, ray, emit, grp, rnd, pos, size, ang; plus a mip atlas (levels >= 1) of flat colour
(non-gold part), gold fraction and emission for level-of-detail.
"""
import math

import cairo
import cv2
import numpy as np

from vx import mosaic as mz
from vx.canvas import Canvas, col

from . import m09_hyper as HY

SM = mz.SMALTI
SC = mz.SC

# ------------------------------------------------------------------ sector sheet geometry (zeta units)
S_RES = 2400.0                        # texels per zeta unit
S_TILE, S_FINE = 18.0, 7.0            # tesserae (texels): ground / faces, hands, letters, thin ornaments
N_TILE, N_FINE = 10.0, 4.6            # tesserae inside the niches (the peoples)
S_X0, S_X1 = -0.012, 0.442
S_Y0, S_Y1 = -0.192, 0.192
S_W = int(round((S_X1 - S_X0) * S_RES))
S_H = int(round((S_Y1 - S_Y0) * S_RES))
R_OC = 0.040                          # oculus (light)
R_MED = 0.058                         # medallion rings end, gold ground begins
RAY0, RAY1 = 0.061, 0.163             # ray from the medallion to the niche
NICHE_X, NICHE_R = 0.2150, 0.0470     # arch centre (on the axis) and radius; niche runs to the ground
FEET_X = 0.3505                       # the people stand here (just inside the tile edge at XM = 0.3646)
FIG_H = 0.163                         # figure height (head top at 0.1875: the cloth flies clear above it)
POLE_L = 0.171                        # pole length (zeta), held beside the head
LIFT_D, LIFT_L = 0.042, 1.32           # 'Flags be.': the pole slides up through the fist by LIFT_D and telescopes x LIFT_L
GROUND_W = 0.021                      # band of earth along the tile edge
WIN_RHO = 0.058                       # vertex lamp (a small round window): hyperbolic radius
VTX = HY.RV * complex(math.cos(HY.AL), math.sin(HY.AL))

# ------------------------------------------------------------------ architecture sheet geometry (w units)
A_RES = 470.0
A_EXT = 2.62
A_N = int(round(2 * A_EXT * A_RES))
R_CORN0, R_CORN1 = 0.992, 1.032       # springing cornice
R_DRUM1 = 1.19                        # drum (windows) outer edge
R_TIT0, R_TIT1 = 1.192, 1.292         # titulus ring
R_BAY = 1.318                         # the square bay / arch crowns
ARCH_W = 0.22                         # arch intrados width
SERAPH_R = 1.60                       # the pendentive seraphim sit on the diagonals at this radius
TITULUS = ("HIC \u00b7 OMNES \u00b7 GENTES \u00b7 SVB \u00b7 VNO \u00b7 CAELO \u00b7 CAELVM \u00b7 ET \u00b7 TERRA \u00b7 "
           "TRANSIBVNT \u00b7 DECEM \u00b7 MILIA \u00b7 VEXILLORVM \u00b7 MANEBVNT \u00b7 ")
TIT_SIZE = 0.080                      # font size (w units); cap height ~0.056 w

# emission groups: 0 none, 1 oculus, 2 vertex windows of the tiling, 3..10 drum windows 0..7
G_OCULUS, G_VWIN, G_DRUM = 1, 2, 3


def hcircle(p, rho):
    """Euclidean centre and radius of the hyperbolic circle of radius rho about p (Poincare disk)"""
    t = math.tanh(rho / 2)
    p2 = abs(p) ** 2
    c = p * (1 - t * t) / (1 - t * t * p2)
    r = t * (1 - p2) / (1 - t * t * p2)
    return c, r


def _rgb(c):
    return col(SM[c]) if isinstance(c, str) and c in SM else col(c)


# ================================================================== cairo helpers (layer-aware, like vx.icon's Painter)
def _poly(ctx, pts):
    ctx.new_path()
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()


def _ell(ctx, x, y, rx, ry):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(rx, ry)
    ctx.new_sub_path()
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()


def _paint(ctx, layer, c, on_gold, on_fine, op):
    """run op (ctx.fill / ctx.stroke) on the right layer: colour paints c; a mask layer paints white where the
    shape carries that material and ERASES it elsewhere (a later non-gold shape removes gold beneath it)."""
    if layer == "color":
        r, g, b = _rgb(c)
        ctx.set_source_rgb(r, g, b)
        op()
        return
    on = (layer == "gold" and on_gold) or (layer == "fine" and on_fine)
    if on:
        ctx.set_source_rgb(1, 1, 1)
        op()
    else:
        ctx.save()
        ctx.set_operator(cairo.OPERATOR_CLEAR)
        op()
        ctx.restore()


def _fill(ctx, c, layer, gold=False, fine=False, stroke=None, lw=1.2):
    """fill the current path (+ optional outline stroke that shares the fine flag but is never gold)."""
    if stroke is not None:
        _paint(ctx, layer, c, gold, fine, ctx.fill_preserve)
        ctx.set_line_width(lw)
        _paint(ctx, layer, stroke, False, fine, ctx.stroke)
    else:
        _paint(ctx, layer, c, gold, fine, ctx.fill)


def _line(ctx, pts, c, lw, layer, fine=False, gold=False):
    ctx.new_path()
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.set_line_width(lw)
    _paint(ctx, layer, c, gold, fine, ctx.stroke)


# ================================================================== the peoples (vx.icon figures)
# eight peoples of the earth, each holding a Flag (the pole is vx.flag's; icon draws only the fist) and raising the
# other hand toward the oculus.
PEOPLES = [
    ("citizen", 3, dict(robe="white", over="porphyry", skin="light")),
    ("people", 11, dict(sex="f", hair="veil", robe="white", over="lapis", skin="brown")),
    ("people", 12, dict(sex="m", hair="cap", beard="short", robe="sand_c", over="teal_c", skin="olive")),
    ("philosopher", 2, dict(robe="white", over="plum_c", skin="warm")),
    ("people", 14, dict(sex="f", hair="long_f", robe="white", over="porphyry", skin="dark")),
    ("people", 15, dict(sex="m", hair="bald", beard="patriarch", robe="lapis", over="white", skin="warm")),
    ("people", 16, dict(sex="m", hair="short", beard=None, robe="white", over="rust_c", skin="brown")),
    ("people", 17, dict(sex="f", hair="bun", robe="teal_c", over="sand_c", skin="light")),
]
POSE = dict(pose="hold_pole1", arm_l="orans", pole_ang=-0.2)   # the pole leans out: the cloth flies in the gold beside the head
_FIGS = {}


def figure(v):
    from vx import icon
    f = _FIGS.get(v)
    if f is None:
        kind, seed, style = PEOPLES[v % len(PEOPLES)]
        f = _FIGS[v] = icon.Figure(kind, seed=seed, **style)
    return f


def _person_frame(ctx):
    """figure local frame (texel-sized units, as vx.icon expects px-like units): feet at the tile edge, head
    toward the oculus (local up = -zeta x)"""
    ctx.translate(FEET_X, 0.0)
    ctx.rotate(-math.pi / 2)
    ctx.scale(1.0 / S_RES, 1.0 / S_RES)


def person_anchors(v):
    """pole base, unit direction (base -> top) and grip in zeta coordinates for people v"""
    a = figure(v).anchors(0.0, 0.0, FIG_H * S_RES, t=0.0, **POSE)
    bx, by, ang = a["pole"]
    gx, gy = a["grip"]

    def z(x, y):                                   # local -> zeta: (x, y) -> (FEET_X + y, -x)
        return complex(FEET_X + y / S_RES, -x / S_RES)
    base = z(bx, by)
    tip = z(bx + math.sin(ang), by - math.cos(ang))  # one local unit up the pole
    d = (tip - base) / abs(tip - base)
    return base, d, z(gx, gy)


# ================================================================== the sector cartoon (zeta coords)
def _sector_frame(ctx):
    """cairo transform: zeta units on a canvas of S_W x S_H texels"""
    ctx.scale(S_RES, S_RES)
    ctx.translate(-S_X0, -S_Y0)


def _wedge(ctx, r0, r1, half, rot=0.0):
    """annular wedge around angle rot of half-angle `half` between radii r0..r1"""
    ctx.new_path()
    ctx.arc(0, 0, r1, rot - half, rot + half)
    if r0 > 0:
        ctx.arc_negative(0, 0, r0, rot + half, rot - half)
    else:
        ctx.line_to(0, 0)
    ctx.close_path()


def _sector_content(ctx, layer, v, rot):
    """everything of ONE sector (oculus wedge, rings, ray, niche, ground, window) rotated by rot (radians);
    the people itself only for rot == 0 (the rotated neighbours are margin for filtering)."""
    half = HY.AL + 0.02
    ctx.save()
    ctx.rotate(rot)
    # gold ground over the whole wedge
    _wedge(ctx, 0.0, 0.47, half)
    _fill(ctx, "gold", layer, gold=True)
    # medallion rings: lapis_d / pearl / porphyry / lapis_l, then the oculus of light
    for r1, c in ((R_MED, "lapis_l"), (0.0535, "porphyry"), (0.0495, "marble"), (0.0455, "lapis_d")):
        _wedge(ctx, 0.0, r1, half)
        _fill(ctx, c, layer, fine=True)
    _wedge(ctx, 0.0, R_OC, half)
    _fill(ctx, "#FBEBC8", layer)
    # pearls on the white ring
    for k in (-1, 0, 1):
        a = k * HY.AL * 0.66
        _ell(ctx, 0.0475 * math.cos(a), 0.0475 * math.sin(a), 0.0019, 0.0019)
        _fill(ctx, "#F3EFE4", layer, fine=True)
    # the ray: a white core between two thin porphyry lines, widening toward the people
    def rayq(w0, w1, c):
        _poly(ctx, [(RAY0, -w0), (RAY1, -w1), (RAY1, w1), (RAY0, w0)])
        _fill(ctx, c, layer, fine=True)
    rayq(0.0058, 0.0140, "porphyry_l")
    rayq(0.0042, 0.0104, "#ECE7DC")
    rayq(0.0014, 0.0036, "#FFFFFF")
    # ground: a band of earth along the tile edge (inside the tile)
    ctx.new_path()
    ctx.arc(HY.CC, 0, HY.RR + GROUND_W, math.pi - 0.45, math.pi + 0.45)
    ctx.arc_negative(HY.CC, 0, HY.RR, math.pi + 0.45, math.pi - 0.45)
    ctx.close_path()
    _fill(ctx, "malachite", layer)
    ctx.new_path()
    ctx.arc(HY.CC, 0, HY.RR + GROUND_W, math.pi - 0.45, math.pi + 0.45)
    _line_path_stroke(ctx, "#163F30", 0.0016, layer)
    for k in range(-9, 10):
        a = math.pi + k * 0.022
        rr = HY.RR + GROUND_W * (0.35 + 0.3 * ((k * 7) % 3) / 2)
        _ell(ctx, HY.CC + rr * math.cos(a), rr * math.sin(a), 0.0022, 0.0022)
        _fill(ctx, "#E8E1D3" if k % 3 else "#B01E2E", layer)
    # the niche: arched, deep lapis inside, pearl + porphyry frame
    def niche_path(grow):
        ctx.new_path()
        ctx.move_to(FEET_X + 0.004, -(NICHE_R + grow))
        ctx.line_to(NICHE_X, -(NICHE_R + grow))
        ctx.arc_negative(NICHE_X, 0, NICHE_R + grow, -math.pi / 2, math.pi / 2)
        ctx.line_to(FEET_X + 0.004, NICHE_R + grow)
        ctx.close_path()
    niche_path(0.0085)
    _fill(ctx, "porphyry", layer, fine=True)
    niche_path(0.0050)
    _fill(ctx, "#E9E3D5", layer, fine=True)
    niche_path(0.0)
    _fill(ctx, "gold_l", layer, gold=True)
    # colonnettes with gold capitals
    for s in (-1, 1):
        y = s * (NICHE_R + 0.0028)
        _poly(ctx, [(NICHE_X, y - 0.0036), (FEET_X, y - 0.0036), (FEET_X, y + 0.0036), (NICHE_X, y + 0.0036)])
        _fill(ctx, "marble_d", layer, fine=True)
        _poly(ctx, [(NICHE_X - 0.002, y - 0.0058), (NICHE_X + 0.009, y - 0.0058), (NICHE_X + 0.009, y + 0.0058),
                    (NICHE_X - 0.002, y + 0.0058)])
        _fill(ctx, "gold", layer, gold=True, fine=True)
    # the ground inside the niche (the people stand on it)
    ctx.new_path()
    ctx.arc(HY.CC, 0, HY.RR + GROUND_W, math.pi - 0.06, math.pi + 0.06)
    ctx.arc_negative(HY.CC, 0, HY.RR, math.pi + 0.06, math.pi - 0.06)
    ctx.close_path()
    _fill(ctx, "verdigris", layer)
    # the vertex lamps (small round windows of warm light), both corners of the sector
    for sgn in (1, -1):
        vtx = complex(VTX.real, sgn * VTX.imag)
        c, r = hcircle(vtx, WIN_RHO)
        for rr, cc in ((r * 1.35, "ink"), (r * 1.2, "gold"), (r * 1.0, "#F0C27E")):
            ctx.new_path()
            ctx.arc(c.real, c.imag, rr, 0, 2 * math.pi)
            _fill(ctx, cc, layer, gold=(cc == "gold"), fine=True)
    if rot == 0.0:
        # the people, standing on the tile edge, head toward the oculus
        ctx.save()
        _person_frame(ctx)
        figure(v).draw(ctx, 0.0, 0.0, FIG_H * S_RES, t=0.0, layer=layer, tile=N_TILE, **POSE)
        ctx.restore()
    ctx.restore()


def _line_path_stroke(ctx, c, lw, layer, fine=False):
    ctx.set_line_width(lw)
    _paint(ctx, layer, c, False, fine, ctx.stroke)


def paint_sector(v):
    """-> dict(rgb, gold, fine) at S_W x S_H texels"""
    out = {}
    for layer in ("color", "gold", "fine", "silver", "pearl", "edges"):
        cv = Canvas(s=1.0, w=S_W, h=S_H)
        if layer == "color":
            cv.fill(SM["gold"])
        ctx = cv.ctx
        _sector_frame(ctx)
        # margin: the two rotational neighbours first, then the sector itself
        for rot in (2 * HY.AL, -2 * HY.AL, 0.0):
            _sector_content(ctx, layer, v, rot)
        a = cv.rgba()
        out["rgb" if layer == "color" else layer] = a[..., :3].copy() if layer == "color" else a[..., 3].copy()
    return out


def sector_fields():
    """analytic per-texel fields of the sector: zeta, inside-sector mask, ray coordinate, emission + group"""
    xs = S_X0 + (np.arange(S_W) + 0.5) / S_RES
    ys = S_Y0 + (np.arange(S_H) + 0.5) / S_RES
    X, Y = np.meshgrid(xs, ys)
    Zc = X + 1j * Y
    R = np.abs(Zc)
    A = np.angle(Zc)
    inside = (np.abs(A) <= HY.AL + 1e-9) & ((X - HY.CC) ** 2 + Y ** 2 >= HY.RR ** 2)
    ray = np.full(X.shape, -1.0, np.float32)
    # the ray wedge (widening), same as painted
    w_at = 0.0058 + (0.0140 - 0.0058) * np.clip((X - RAY0) / (RAY1 - RAY0), 0, 1)
    rm = (X >= RAY0) & (X <= RAY1) & (np.abs(Y) <= w_at)
    ray[rm] = ((X[rm] - RAY0) / (RAY1 - RAY0)).astype(np.float32)
    emit = np.zeros(X.shape, np.float32)
    grp = np.zeros(X.shape, np.int32)
    oc = R < R_OC
    emit[oc] = 1.0
    grp[oc] = G_OCULUS
    for sgn in (1, -1):
        vtx = complex(VTX.real, sgn * VTX.imag)
        c, r = hcircle(vtx, WIN_RHO)
        m = np.abs(Zc - c) < r
        emit[m] = 1.0
        grp[m] = G_VWIN
    return dict(zeta=Zc, inside=inside, ray=ray, emit=emit, grp=grp)


def sector_parts(cart, fields):
    """layout parts: the niches (finer tesserae for the peoples) and the rest"""
    Zc = fields["zeta"]
    X, Y = Zc.real, Zc.imag
    niche = ((X >= NICHE_X) & (X <= FEET_X + 0.012) & (np.abs(Y) <= NICHE_R + 0.009)) | \
        (np.abs(Zc - NICHE_X) <= NICHE_R + 0.009)
    fine = cart["fine"] > 0.5
    return [dict(alpha=~niche, tile=S_TILE, fine=fine & ~niche, fine_tile=S_FINE),
            dict(alpha=niche, tile=N_TILE, fine=fine & niche, fine_tile=N_FINE)]


def sector_edges():
    """explicit course lines for the tesserae: the sector boundaries, the tile edge, the medallion circles"""
    E = np.zeros((S_H, S_W), np.uint8)
    k = S_RES

    def P(z):
        return (int(round((z.real - S_X0) * k)), int(round((z.imag - S_Y0) * k)))
    for s in (1, -1):
        e = complex(math.cos(s * HY.AL), math.sin(s * HY.AL))
        cv2.line(E, P(0j), P(0.5 * e), 1, 1)
    cxy = (int(round((HY.CC - S_X0) * k)), int(round((0 - S_Y0) * k)))
    cv2.circle(E, cxy, int(round(HY.RR * k)), 1, 1)
    oc = P(0j)
    for r in (R_OC, R_MED):
        cv2.circle(E, oc, int(round(r * k)), 1, 1)
    return E.astype(bool)


# ================================================================== the architecture cartoon (w coords)
def _arch_frame(ctx):
    ctx.translate(A_N / 2, A_N / 2)
    ctx.scale(A_RES, A_RES)


def _star(ctx, x, y, r, layer):
    """a vault star as Galla Placidia sets it at this scale: a round gold boss with a pale centre (no points)"""
    ctx.new_path()
    ctx.arc(x, y, r * 0.62, 0, 2 * math.pi)
    _fill(ctx, "star", layer, gold=True)


def _bez(P0, P1, P2, t):
    u = 1.0 - t
    return (u * u * P0[0] + 2 * u * t * P1[0] + t * t * P2[0], u * u * P0[1] + 2 * u * t * P1[1] + t * t * P2[1])


def _blade(ctx, layer, P0, P1, P2, W, n_feathers=7, ink="#241512"):
    """a curved wing blade along the quadratic curve P0 (shoulder) -> P1 -> P2 (tip), width W at its widest.
    Feather bands run along it: pearl-white coverts on the outer (leading) edge, porphyry secondaries, lapis
    primaries whose tips scallop the inner (trailing) edge; dark feather quills across the porphyry band."""
    n = 44
    cx0 = (P0[0] + P1[0] + P2[0]) / 3.0          # the blade's own centre, to tell outer from inner
    cy0 = (P0[1] + P1[1] + P2[1]) / 3.0
    C, N, Wt = [], [], []
    for i in range(n + 1):
        t = i / n
        c = _bez(P0, P1, P2, t)
        a = _bez(P0, P1, P2, max(0.0, t - 0.01))
        b = _bez(P0, P1, P2, min(1.0, t + 0.01))
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) + 1e-9
        nx, ny = -dy / L, dx / L
        C.append(c)
        N.append((nx, ny))
        Wt.append(W * min(1.0, 0.5 + 2.5 * t) * (1.0 - t * t) ** 0.5 * (0.35 + 0.65 * (1.0 - t) ** 0.3))
    # outer side = away from the curve's own centre of mass (the convex side)
    mid = n // 2
    if (C[mid][0] - cx0) * N[mid][0] + (C[mid][1] - cy0) * N[mid][1] < 0:
        N = [(-x, -y) for x, y in N]

    def edge(frac, scallop=0.0):
        pts = []
        for i in range(n + 1):
            t = i / n
            o = (0.5 - frac) * Wt[i]
            if scallop > 0.0:
                o -= scallop * Wt[i] * abs(math.sin(math.pi * n_feathers * t))
            pts.append((C[i][0] + N[i][0] * o, C[i][1] + N[i][1] * o))
        return pts
    bands = ((0.0, 0.34, "marble"), (0.34, 0.70, "porphyry"), (0.70, 1.0, "lapis"))
    for f0, f1, col_ in bands:
        a = edge(f0)
        b = edge(f1, scallop=0.26 if f1 >= 1.0 else 0.0)
        _poly(ctx, a + b[::-1])
        _fill(ctx, col_, layer, gold=(col_ == "gold"))
    # feather quills across the porphyry band
    for k in range(1, n_feathers):
        i = int(round(n * k / n_feathers))
        p = (C[i][0] + N[i][0] * (0.5 - 0.40) * Wt[i], C[i][1] + N[i][1] * (0.5 - 0.40) * Wt[i])
        q = (C[i][0] + N[i][0] * (0.5 - 0.92) * Wt[i], C[i][1] + N[i][1] * (0.5 - 0.92) * Wt[i])
        _line(ctx, [p, q], "porphyry_l", W * 0.07, layer)
    # outline
    outline = edge(0.0) + edge(1.0, scallop=0.26)[::-1]
    _poly(ctx, outline)
    ctx.set_line_width(W * 0.08)
    _paint(ctx, layer, ink, False, False, ctx.stroke)


def _seraph(ctx, layer, x, y, ang, s):
    """six-winged seraph of the pendentives (Isaiah 6: two wings cover the face, two the feet, two fly), after
    Hagia Sophia: a small face in a gold nimbus, the upper pair arching over the head, the lower pair curving
    under it (together a mandorla of feathers), the flying pair swept out and up. size s (w units) at (x, y),
    'up' along ang (toward the dome)."""
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    ctx.scale(s, s)
    ink = "#241512"
    for sx in (-1, 1):
        # the flying pair: from the shoulders, out and then up
        _blade(ctx, layer, (sx * 0.07, 0.03), (sx * 0.60, 0.08), (sx * 0.78, -0.30), 0.27, 8)
    for sx in (-1, 1):
        # the pair that covers the feet: down, out and back under
        _blade(ctx, layer, (sx * 0.05, 0.09), (sx * 0.40, 0.36), (sx * 0.10, 0.56), 0.23, 6)
    for sx in (-1, 1):
        # the pair that covers the face: up, out and back over the head
        _blade(ctx, layer, (sx * 0.05, -0.09), (sx * 0.42, -0.40), (sx * 0.10, -0.60), 0.23, 6)
    # the face, in a gold nimbus: pale flesh, two large dark almond eyes (legible in tesserae at ~50 px)
    _ell(ctx, 0, -0.006, 0.150, 0.150)
    _fill(ctx, "gold", layer, gold=True, stroke=ink, lw=0.016)
    _ell(ctx, 0, 0.0, 0.104, 0.126)
    _fill(ctx, "#EBC2A2", layer, fine=True, stroke="#8A5A40", lw=0.010)
    _ell(ctx, 0, -0.106, 0.100, 0.030)
    _fill(ctx, "#6E4E37", layer, fine=True)
    for sx in (-1, 1):
        _ell(ctx, sx * 0.045, -0.020, 0.036, 0.019)
        _fill(ctx, "#1A120E", layer, fine=True)
    _line(ctx, [(0.0, -0.004), (0.004, 0.046)], "#9A6448", 0.014, layer, fine=True)
    _line(ctx, [(-0.024, 0.080), (0.024, 0.080)], "#A8403A", 0.020, layer, fine=True)
    ctx.restore()


def _guilloche(ctx, layer, x0, y0, x1, y1, horiz):
    """a band of interlaced circles (arch intrados) inside the rectangle"""
    L = (x1 - x0) if horiz else (y1 - y0)
    Wd = (y1 - y0) if horiz else (x1 - x0)
    n = max(4, int(L / (Wd * 0.9)))
    for i in range(n):
        f = (i + 0.5) / n
        cx = x0 + f * (x1 - x0) if horiz else (x0 + x1) / 2
        cy = (y0 + y1) / 2 if horiz else y0 + f * (y1 - y0)
        for rr, c in ((Wd * 0.40, "porphyry"), (Wd * 0.33, "marble"), (Wd * 0.25, "lapis"), (Wd * 0.12, "gold")):
            ctx.new_path()
            ctx.arc(cx, cy, rr, 0, 2 * math.pi)
            _fill(ctx, c, layer, gold=(c == "gold"))


def paint_architecture():
    out = {}
    for layer in ("color", "gold", "fine"):
        cv = Canvas(s=1.0, w=A_N, h=A_N)
        if layer == "color":
            cv.fill(SM["night_vault"])
        ctx = cv.ctx
        _arch_frame(ctx)
        E = A_EXT + 0.1
        # starry barrel vaults beyond the arches (Galla Placidia)
        rng = np.random.default_rng(9)
        for k in range(900):
            x, y = rng.uniform(-E, E, 2)
            if max(abs(x), abs(y)) < R_BAY + ARCH_W + 0.04:
                continue
            _star(ctx, x, y, rng.uniform(0.012, 0.026), layer)
        # corner piers (beyond both arches): dark porphyry revetment with a pearl frame
        for sx in (-1, 1):
            for sy in (-1, 1):
                x0 = sx * (R_BAY + ARCH_W)
                y0 = sy * (R_BAY + ARCH_W)
                _poly(ctx, [(x0, y0), (sx * E, y0), (sx * E, sy * E), (x0, sy * E)])
                _fill(ctx, "porphyry", layer)
                _poly(ctx, [(x0 + sx * 0.04, y0 + sy * 0.04), (sx * E, y0 + sy * 0.04), (sx * E, sy * E),
                            (x0 + sx * 0.04, sy * E)])
                _fill(ctx, "#4A111B", layer)
        # the four great arches: gold intrados, borders, a band of roundels
        B0, B1 = R_BAY, R_BAY + ARCH_W
        for s in (-1, 1):
            for horiz in (True, False):
                if horiz:
                    x0, y0, x1, y1 = -B1, min(s * B0, s * B1), B1, max(s * B0, s * B1)
                else:
                    x0, y0, x1, y1 = min(s * B0, s * B1), -B1, max(s * B0, s * B1), B1
                _poly(ctx, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                _fill(ctx, "gold", layer, gold=True)
                # borders
                if horiz:
                    for yy in (y0, y1 - 0.02):
                        _poly(ctx, [(x0, yy), (x1, yy), (x1, yy + 0.02), (x0, yy + 0.02)])
                        _fill(ctx, "lapis", layer)
                    _guilloche(ctx, layer, x0 + 0.03, y0 + 0.03, x1 - 0.03, y1 - 0.03, True)
                else:
                    for xx in (x0, x1 - 0.02):
                        _poly(ctx, [(xx, y0), (xx + 0.02, y0), (xx + 0.02, y1), (xx, y1)])
                        _fill(ctx, "lapis", layer)
                    _guilloche(ctx, layer, x0 + 0.03, y0 + 0.03, x1 - 0.03, y1 - 0.03, False)
        # pendentives: gold, one seraph each, facing the dome
        ctx.new_path()
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.rectangle(-B0, -B0, 2 * B0, 2 * B0)
        ctx.new_sub_path()
        ctx.arc(0, 0, R_BAY, 0, 2 * math.pi)
        _fill(ctx, "gold", layer, gold=True)
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
        for k in range(4):
            a = math.pi / 4 + k * math.pi / 2
            _seraph(ctx, layer, SERAPH_R * math.cos(a), SERAPH_R * math.sin(a), a - math.pi / 2, 0.46)
        # titulus ring: lapis band with gold letters (tops toward the dome), pearl borders
        ctx.new_path()
        ctx.arc(0, 0, R_BAY, 0, 2 * math.pi)
        _fill(ctx, "porphyry", layer)
        ctx.new_path()
        ctx.arc(0, 0, R_BAY - 0.012, 0, 2 * math.pi)
        _fill(ctx, "marble", layer)
        ctx.new_path()
        ctx.arc(0, 0, R_TIT1, 0, 2 * math.pi)
        _fill(ctx, "lapis_d", layer)
        _ring_text(ctx, layer, TITULUS, 0.5 * (R_TIT0 + R_TIT1), TIT_SIZE)
        # the drum: gold piers, 8 windows at the vertex directions
        ctx.new_path()
        ctx.arc(0, 0, R_TIT0, 0, 2 * math.pi)
        _fill(ctx, "gold", layer, gold=True)
        for k in range(8):
            a = HY.AL + k * 2 * HY.AL
            _drum_window(ctx, layer, a)
            # pier ornament: a jewelled column on the figure's axis
            b = k * 2 * HY.AL
            ca, sa = math.cos(b), math.sin(b)
            for rr in np.arange(1.05, 1.18, 0.022):
                _ell(ctx, rr * ca, rr * sa, 0.0075, 0.0075)
                _fill(ctx, "#E9E3D5" if int(rr * 100) % 2 else "#B01E2E", layer)
        # springing cornice
        ctx.new_path()
        ctx.arc(0, 0, R_CORN1, 0, 2 * math.pi)
        _fill(ctx, "marble", layer)
        n = 180
        for k in range(n):
            a = 2 * math.pi * k / n
            _poly(ctx, [(R_CORN1 * math.cos(a - 0.008), R_CORN1 * math.sin(a - 0.008)),
                        (R_CORN1 * math.cos(a + 0.008), R_CORN1 * math.sin(a + 0.008)),
                        (1.012 * math.cos(a + 0.008), 1.012 * math.sin(a + 0.008)),
                        (1.012 * math.cos(a - 0.008), 1.012 * math.sin(a - 0.008))])
            _fill(ctx, "porphyry", layer)
        ctx.new_path()
        ctx.arc(0, 0, R_CORN0 + 0.006, 0, 2 * math.pi)
        _fill(ctx, "ink", layer)
        # the dome itself is not sampled here (except the 1-px rim blend): gold
        ctx.new_path()
        ctx.arc(0, 0, R_CORN0, 0, 2 * math.pi)
        _fill(ctx, "gold", layer, gold=True)
        a = cv.rgba()
        out["rgb" if layer == "color" else layer] = a[..., :3].copy() if layer == "color" else a[..., 3].copy()
    z = np.zeros((A_N, A_N), np.float32)
    out.update(silver=z, pearl=z, edges=z)
    return out


def _drum_window(ctx, layer, a):
    """arched window of light in the drum (w plane): radial trapezoid, arch toward the dome"""
    hw = math.radians(8.5)
    r0, r1 = 1.056, 1.172
    ca, sa = math.cos(a), math.sin(a)

    def P(r, da):
        return (r * math.cos(a + da), r * math.sin(a + da))
    for grow, c in ((0.018, "ink"), (0.010, "marble"), (0.0, "#F6D9A0")):
        g = grow / 1.1
        ctx.new_path()
        ctx.move_to(*P(r1 + grow, -hw - g))
        ctx.line_to(*P(r1 + grow, hw + g))
        ctx.line_to(*P(r0 + 0.035, hw + g))
        # arch cap (toward the centre)
        mx, my = (r0 + 0.035) * ca, (r0 + 0.035) * sa
        rad = (r0 + 0.035) * math.sin(hw + g)
        ctx.arc(mx, my, rad, a + math.pi / 2, a + 3 * math.pi / 2)
        ctx.line_to(*P(r0 + 0.035, -hw - g))
        ctx.close_path()
        _fill(ctx, c, layer)
    # round panes (transenna)
    for rr in (1.085, 1.12, 1.152):
        for da in (-0.5, 0.0, 0.5):
            x, y = P(rr, da * hw)
            ctx.new_path()
            ctx.arc(x, y, 0.0105, 0, 2 * math.pi)
            _line_path_stroke(ctx, "#8A7B62", 0.0028, layer)


def _ring_text(ctx, layer, txt, r, size):
    """Roman capitals around a ring, tops toward the centre, reading counter-clockwise as seen from inside"""
    if layer not in ("color", "gold", "fine"):
        return
    ctx.save()
    from vx.mosaic_type import face
    ctx.select_font_face(face("roman_black"), cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    adv = [ctx.text_extents(ch).x_advance * 1.06 for ch in txt]
    total = sum(adv)
    circ = 2 * math.pi * r
    reps = max(1, int(circ / total))
    sc = circ / (total * reps)
    a = math.pi / 2          # start at the bottom
    for _ in range(reps):
        for ch, ad in zip(txt, adv):
            da = ad * sc / r
            am = a - da / 2
            ctx.save()
            x, y = r * math.cos(am), r * math.sin(am)
            ctx.translate(x, y)
            # letter "up" points to the centre; reading direction is -angle (counter-clockwise on screen)
            ctx.rotate(am - math.pi / 2)
            ext = ctx.text_extents(ch)
            ctx.move_to(-ext.x_advance / 2, size * 0.345)
            ctx.text_path(ch)
            if layer == "color":
                ctx.set_source_rgb(*_rgb("gold_l"))
                ctx.fill()
            else:
                ctx.set_source_rgb(1, 1, 1)
                ctx.fill()
            ctx.restore()
            a -= da
    ctx.restore()


def arch_fields():
    xs = (np.arange(A_N) + 0.5 - A_N / 2) / A_RES
    X, Y = np.meshgrid(xs, xs)
    R = np.hypot(X, Y)
    A = np.arctan2(Y, X)
    emit = np.zeros(X.shape, np.float32)
    grp = np.zeros(X.shape, np.int32)
    ring = (R > 1.056) & (R < 1.172)
    for k in range(8):
        a = HY.AL + k * 2 * HY.AL
        d = np.angle(np.exp(1j * (A - a)))
        m = ring & (np.abs(d) < math.radians(8.5))
        # arch cap: approximately the same wedge
        emit[m] = 1.0
        grp[m] = G_DRUM + k
    ray = np.full(X.shape, -1.0, np.float32)
    return dict(emit=emit, grp=grp, ray=ray, R=R)


def arch_titulus_parts(cart):
    """layout parts for the architecture: everything but the titulus ring (normal tesserae), and the ring
    itself with opus vermiculatum letters (tiny tiles, ~0.085 x cap height) so the Latin reads, and the four
    seraphs' faces in small tesserae."""
    xs = (np.arange(A_N) + 0.5 - A_N / 2) / A_RES
    X, Y = np.meshgrid(xs, xs)
    R = np.hypot(X, Y)
    ring = (R > R_TIT0 - 0.004) & (R < R_TIT1 + 0.004)
    faces = np.zeros_like(ring)
    for k in range(4):                          # the seraphs' faces: opus vermiculatum too
        a = math.pi / 4 + k * math.pi / 2
        faces |= np.hypot(X - SERAPH_R * math.cos(a), Y - SERAPH_R * math.sin(a)) < 0.074
    fine = cart["fine"] > 0.5
    rest = ~(ring | faces)
    return [dict(alpha=rest, tile=11.0, fine=fine & rest, fine_tile=5.0),
            dict(alpha=ring, tile=7.0, fine=fine & ring, fine_tile=2.4),
            dict(alpha=faces, tile=6.0, fine=fine & faces, fine_tile=3.8, joint=0.55)]


def arch_edges():
    E = np.zeros((A_N, A_N), np.uint8)
    c = (A_N // 2, A_N // 2)
    for r in (R_CORN0, R_CORN1, R_DRUM1, R_TIT0, R_TIT1, R_BAY):
        cv2.circle(E, c, int(round(r * A_RES)), 1, 1)
    for s in (-1, 1):
        for b in (R_BAY, R_BAY + ARCH_W):
            p = int(round(A_N / 2 + s * b * A_RES))
            cv2.line(E, (p, 0), (p, A_N - 1), 1, 1)
            cv2.line(E, (0, p), (A_N - 1, p), 1, 1)
    return E.astype(bool)


# ================================================================== baking
def _colour_edges(rgb, tile):
    q = np.clip(rgb[..., :3] * 255, 0, 255).astype(np.uint8)
    q = cv2.GaussianBlur(q, (0, 0), max(0.6, 0.12 * tile))
    e = cv2.Canny(cv2.cvtColor(q, cv2.COLOR_RGB2GRAY), 18, 40) > 0
    lab = cv2.cvtColor(q, cv2.COLOR_RGB2LAB)
    for ch in range(1, 3):
        e |= cv2.Canny(np.ascontiguousarray(lab[..., ch]), 10, 25) > 0
    return e


def bake(cart, emit, grp, ray, tile, fine_tile, seed, extra_edges=None, grout=1.3, fan_center=None, parts=None):
    """cartoon dict (rgb, gold, fine, silver, pearl, edges; texel units) -> sheet dict: per-texel tessera id, bevel
    and local coords, per-tessera colour (smalti melts), gold + gold-leaf tint, tilt, ray, emission, and a flat
    level-0 colour for the mip pyramid."""
    rgb = cart["rgb"]
    H, W = rgb.shape[:2]
    e = _colour_edges(rgb, tile) | (cart["edges"] > 0.5)
    if extra_edges is not None:
        e |= extra_edges
    gold = cart["gold"] > 0.5
    if parts is None:
        parts = [dict(alpha=None, tile=tile, fine=cart["fine"] > 0.5, fine_tile=fine_tile)]
    lay = None
    for k, pt in enumerate(parts):
        lk = mz.Layout.flow(rgb, tile=pt["tile"], seed=seed + 31 * k, extent=(W, H), edges=e, fine=pt["fine"],
                            fine_tile=pt["fine_tile"], gold=gold, silver=cart["silver"] > 0.5,
                            pearl=cart["pearl"] > 0.5, ground=gold,
                            ground_style="fan" if fan_center is not None else "rows", fan_center=fan_center,
                            alpha=None if pt["alpha"] is None else pt["alpha"].astype(np.float32),
                            joint=pt.get("joint", 1.0))
        lay = lk if lay is None else lay + lk
    lab, edge, uu, vv = mz.raster(lay, 1.0, grout)
    n = len(lay)
    L = lab.ravel()
    m = L >= 0
    Lm = L[m]
    cnt = np.bincount(Lm, minlength=n).astype(np.float64) + 1e-9

    def tmean(a):
        return (np.bincount(Lm, a.ravel()[m].astype(np.float64), minlength=n) / cnt).astype(np.float32)
    srgb = mz.sample(rgb, lay)
    tcol = np.clip(mz.lin_to_srgb(mz.smalti_colours(srgb, lay.jit, lay.rnd)), 0, 1).astype(np.float32)
    mat = np.asarray(lay.mat)
    tg = (mat == 1).astype(np.float32)
    # gold leaf: warmer / paler leaves, a few reversed (dull) tesserae
    gt = np.ones((n, 3), np.float32)
    gt *= np.clip(1.0 + 0.10 * lay.jit[:, :1], 0.7, 1.25)
    gt[:, 2] *= np.clip(1.0 + 0.25 * lay.jit[:, 1], 0.5, 1.6)
    lum = mz.srgb_to_lin(srgb) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lref = float(mz.srgb_to_lin(SC["gold"]) @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    gt *= (np.clip(lum / lref, 0.35, 1.6) ** 0.7)[:, None]      # the cartoon's VALUE: gold_d .. gold .. gold_l
    dull = lay.rnd < 0.035
    gt[dull] = np.array([0.30, 0.30, 0.34], np.float32)
    # silver and pearl: brighter glass that glints a little
    sil = (mat == 2) | (mat == 4)
    tg[sil] = 0.0
    tem = tmean(emit)
    tem = np.where(tem > 0.5, 1.0, 0.0).astype(np.float32)
    tgrp = np.zeros(n, np.int32)
    gv = grp.ravel()[m]
    for gi in np.unique(gv):
        if gi == 0:
            continue
        c = np.bincount(Lm, (gv == gi).astype(np.float64), minlength=n) / cnt
        tgrp[c > 0.5] = gi
    tem[tgrp == 0] = 0.0
    rv = ray.ravel()[m]
    valid = (rv >= 0).astype(np.float64)
    vcnt = np.bincount(Lm, valid, minlength=n)
    rsum = np.bincount(Lm, np.where(rv >= 0, rv, 0.0), minlength=n)
    tray = np.where(vcnt / cnt > 0.5, rsum / np.maximum(1e-9, vcnt), -1.0).astype(np.float32)
    tg[tem > 0] = 0.0                 # emitters are light, not leaf
    lab_c = np.where(lab >= 0, lab, 0)
    rim = np.clip(edge / np.clip(0.28 * np.asarray(lay.size, np.float32)[lab_c], 0.7, 2.2), 0, 1)   # bevel ~ tile size
    shade = (0.72 + 0.28 * rim).astype(np.float32)
    lab = lab.astype(np.int32)
    Lc = np.where(lab >= 0, lab, 0)
    flat = np.zeros((H, W, 5), np.float32)
    flat[..., :3] = (1.0 - tg)[Lc][..., None] * tcol[Lc] * shade[..., None]
    flat[..., 3] = tg[Lc] * shade * gt.mean(1)[Lc]
    flat[..., 4] = tem[Lc]
    mort = lab < 0
    flat[mort, :3] = SC["mortar"] * 0.85
    flat[mort, 3] = 0.0
    flat[mort, 4] = 0.0
    return dict(lab=lab, rim=np.clip(rim * 255 + 0.5, 0, 255).astype(np.uint8),
                u=np.clip(uu * 127, -127, 127).astype(np.int8), v=np.clip(vv * 127, -127, 127).astype(np.int8),
                col=tcol, gold=tg, gt=gt, tilt=np.asarray(lay.tilt, np.float32), ray=tray, emit=tem, grp=tgrp,
                rnd=np.asarray(lay.rnd, np.float32), pos=np.asarray(lay.xy, np.float32),
                size=np.asarray(lay.size, np.float32), ang=np.asarray(lay.ang, np.float32), flat=flat)


def mip_atlas(flat):
    """levels >= 1 of a (H, W, 5) flat image packed in one atlas -> (atlas uint8, lvl (L+1, 4) int32)"""
    levels = []
    cur = flat
    while min(cur.shape[:2]) > 3:
        h, w = cur.shape[:2]
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        pad = np.pad(cur, ((0, h2 * 2 - h), (0, w2 * 2 - w), (0, 0)), mode="edge")
        cur = pad.reshape(h2, 2, w2, 2, 5).mean(axis=(1, 3))
        levels.append(cur)
    W1 = levels[0].shape[1]
    H1 = levels[0].shape[0]
    W2 = levels[1].shape[1] if len(levels) > 1 else 0
    Hs = sum(Lv.shape[0] + 1 for Lv in levels[1:])
    atl = np.zeros((max(H1, Hs) + 1, W1 + W2 + 2, 5), np.uint8)
    lvl = np.zeros((len(levels) + 1, 4), np.int32)
    y = 0
    for i, Lv in enumerate(levels):
        h, w = Lv.shape[:2]
        if i == 0:
            x0, y0 = 0, 0
        else:
            x0, y0 = W1 + 1, y
            y += h + 1
        atl[y0:y0 + h, x0:x0 + w] = np.clip(Lv * 255 + 0.5, 0, 255).astype(np.uint8)
        lvl[i + 1] = (x0, y0, w, h)
    return atl, lvl

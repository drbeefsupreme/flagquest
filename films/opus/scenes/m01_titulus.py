"""m01 - tituli: Roman square capitals laid as courses of tesserae (a stroke font), on a gold (or lapis) band.

Real mosaic inscriptions are not rasterised fonts: every stroke is a course of tesserae laid along the pen path,
and the ground tiles hug the letters.  This module designs the letters as pen strokes, lays one course of tiles
along each (thick strokes in larger tiles, thin strokes in smaller: the Roman stress), and lays the band ground
around them (vx.mosaic.Layout.flow, whose courses echo the letter contours), with a porphyry rule and a row of
pearls along each edge.  Everything is in band texture units (x along the band, y across, y down).

Usable on a flat wall too (M10's EXPLICIT band):
    band = Band("EXPLICIT·OPVS·SCHISMATICVM", height=90, letter_h=50, ground="gold", ink="lapis_d")
    band.tiles -> dict of arrays (x, y, ang, hu, hv, rgb, gold, glyph, order, letter(bool))
    band.set_times(glyph_times) -> per-tile t_set (a letter's tiles set in stroke order across its window)
    pop(t, t_set) -> (scale, flash) per tile for the setting animation
Glyphs: A B C D E F G H I K L M N O P Q R S T U V W X Y Z and '·' (interpunct, a lozenge tile), ' ' (space).
"""
import math

import cv2
import numpy as np

from vx.mosaic import Layout, SC

# ============================================================ the pen strokes (cap height 1, y down)
THICK, THIN = 1.18, 0.82


def _line(x0, y0, x1, y1, w=THICK, n=None):
    return (np.array([[x0, y0], [x1, y1]], float), w)


def _arc(cx, cy, rx, ry, a0, a1, w=THICK, n=48):
    a = np.radians(np.linspace(a0, a1, n))
    return (np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1), w)


def _spline(pts, w=THICK, n=64):
    P = np.asarray(pts, float)
    # Catmull-Rom through the points
    out = []
    Q = np.vstack([P[0], P, P[-1]])
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        for t in np.linspace(0, 1, max(4, n // (len(P) - 1)), endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    return (np.array(out), w)


# angles in degrees, 0 = +x, 90 = +y (down)
GLYPHS = {
    "A": (0.80, [_line(0.40, 0.0, 0.04, 1.0, THIN), _line(0.40, 0.0, 0.76, 1.0), _line(0.20, 0.64, 0.60, 0.64, THIN)]),
    "B": (0.62, [_line(0.07, 0, 0.07, 1), _line(0.07, 0, 0.36, 0, THIN), _arc(0.36, 0.235, 0.21, 0.235, -90, 90, THIN, 24),
                 _line(0.36, 0.47, 0.07, 0.47, THIN), _line(0.07, 1, 0.40, 1, THIN),
                 _arc(0.40, 0.735, 0.235, 0.265, 90, -90, THICK, 24), _line(0.40, 0.47, 0.20, 0.47, THIN)]),
    "C": (0.74, [_arc(0.46, 0.5, 0.40, 0.5, -48, -300, THICK, 56)]),
    "D": (0.80, [_line(0.07, 0, 0.07, 1), _line(0.07, 0, 0.34, 0, THIN), _line(0.07, 1, 0.34, 1, THIN),
                 _arc(0.34, 0.5, 0.40, 0.5, -90, 90, THICK, 40)]),
    "E": (0.58, [_line(0.07, 0, 0.07, 1), _line(0.07, 0, 0.54, 0, THIN), _line(0.07, 0.49, 0.46, 0.49, THIN),
                 _line(0.07, 1, 0.56, 1, THIN)]),
    "F": (0.56, [_line(0.07, 0, 0.07, 1), _line(0.07, 0, 0.54, 0, THIN), _line(0.07, 0.49, 0.46, 0.49, THIN)]),
    "G": (0.80, [_arc(0.46, 0.5, 0.40, 0.5, -48, -300, THICK, 56), _line(0.80, 0.62, 0.80, 0.93, THIN),
                 _line(0.58, 0.60, 0.80, 0.60, THIN)]),
    "H": (0.78, [_line(0.07, 0, 0.07, 1), _line(0.71, 0, 0.71, 1), _line(0.07, 0.5, 0.71, 0.5, THIN)]),
    "I": (0.20, [_line(0.10, 0, 0.10, 1)]),
    "L": (0.56, [_line(0.07, 0, 0.07, 1), _line(0.07, 1, 0.54, 1, THIN)]),
    "M": (0.96, [_line(0.06, 1.0, 0.10, 0.0, THIN), _line(0.10, 0.0, 0.48, 1.0), _line(0.48, 1.0, 0.86, 0.0, THIN),
                 _line(0.86, 0.0, 0.90, 1.0)]),
    "N": (0.80, [_line(0.07, 1, 0.07, 0, THIN), _line(0.07, 0, 0.73, 1), _line(0.73, 1, 0.73, 0, THIN)]),
    "O": (0.88, [_arc(0.44, 0.5, 0.40, 0.5, -90, 270, THICK, 64)]),
    "P": (0.62, [_line(0.07, 1, 0.07, 0), _line(0.07, 0, 0.34, 0, THIN), _arc(0.34, 0.26, 0.24, 0.26, -90, 90, THIN, 28),
                 _line(0.34, 0.52, 0.07, 0.52, THIN)]),
    "Q": (0.88, [_arc(0.44, 0.5, 0.40, 0.5, -90, 270, THICK, 64), _line(0.52, 0.86, 0.86, 1.10, THIN)]),
    "R": (0.68, [_line(0.07, 1, 0.07, 0), _line(0.07, 0, 0.34, 0, THIN), _arc(0.34, 0.25, 0.24, 0.25, -90, 90, THIN, 28),
                 _line(0.34, 0.50, 0.07, 0.50, THIN), _line(0.30, 0.50, 0.66, 1.0)]),
    "S": (0.58, [_spline([(0.52, 0.10), (0.34, 0.0), (0.12, 0.06), (0.08, 0.24), (0.22, 0.42), (0.38, 0.54),
                          (0.52, 0.72), (0.48, 0.92), (0.26, 1.0), (0.05, 0.90)], THICK, 72)]),
    "T": (0.72, [_line(0.0, 0, 0.72, 0, THIN), _line(0.36, 0, 0.36, 1)]),
    "V": (0.74, [_line(0.03, 0, 0.37, 1), _line(0.37, 1, 0.71, 0, THIN)]),
    "X": (0.72, [_line(0.03, 0, 0.69, 1), _line(0.69, 0, 0.03, 1, THIN)]),
    "Y": (0.72, [_line(0.03, 0, 0.36, 0.52), _line(0.69, 0, 0.36, 0.52, THIN), _line(0.36, 0.52, 0.36, 1)]),
    "U": (0.76, [_line(0.07, 0, 0.07, 0.64), _arc(0.38, 0.64, 0.31, 0.36, 180, 0, THICK, 32),
                 _line(0.69, 0.64, 0.69, 0.0, THIN)]),
    "K": (0.70, [_line(0.07, 0, 0.07, 1), _line(0.64, 0.0, 0.10, 0.56, THIN), _line(0.28, 0.40, 0.68, 1.0)]),
    "W": (1.16, [_line(0.03, 0, 0.31, 1), _line(0.31, 1, 0.58, 0.05, THIN), _line(0.58, 0.05, 0.85, 1),
                 _line(0.85, 1, 1.13, 0, THIN)]),
    "Z": (0.66, [_line(0.04, 0, 0.62, 0, THIN), _line(0.62, 0, 0.04, 1), _line(0.04, 1, 0.64, 1, THIN)]),
    "·": (0.36, []),
    " ": (0.36, []),
}


def _resample(P, step):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    L = s[-1]
    if L < 1e-9:
        return P[:1], np.zeros(1)
    n = max(1, int(round(L / step)))
    q = np.linspace(0, L, n + 1)
    x = np.interp(q, s, P[:, 0])
    y = np.interp(q, s, P[:, 1])
    Q = np.stack([x, y], 1)
    tg = np.gradient(Q, axis=0)
    return Q, np.arctan2(tg[:, 1], tg[:, 0])


def lay_text(text, letter_h, tile, tracking=0.16, seed=0):
    """letter tiles for a line of text starting at x=0, cap top at y=0.
    returns dict(x, y, ang, hu, hv, glyph, order) and the total advance width."""
    rng = np.random.default_rng(seed)
    xs, ys, angs, hus, hvs, gl, order = [], [], [], [], [], [], []
    pen = 0.0
    centers = []
    for gi, ch in enumerate(text):
        w, strokes = GLYPHS.get(ch.upper(), GLYPHS[" "])
        k = 0
        if ch == "·":
            xs.append(pen + w * letter_h * 0.5)
            ys.append(0.55 * letter_h)
            angs.append(math.pi / 4)
            hus.append(tile * 0.62)
            hvs.append(tile * 0.62)
            gl.append(gi)
            order.append(0)
        glyph_pts = []
        for P, wt in strokes:
            Pd = P * letter_h
            Pd[:, 0] += pen
            courses = [0.0] if wt < 1.0 else [-0.53, 0.53]
            ts = tile * (0.92 if wt < 1.0 else 1.0)
            for off in courses:
                Q, a = _resample(Pd, ts * 1.03)
                if off:
                    Q = Q + off * ts * np.stack([-np.sin(a), np.cos(a)], 1)
                    Q, a = _resample(Q, ts * 1.04)
                for q, aa in zip(Q, a):
                    # skip tiles overlapping this glyph's earlier tiles (stroke junctions)
                    if any((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < (0.46 * (ts + p[2])) ** 2 for p in glyph_pts):
                        continue
                    glyph_pts.append((q[0], q[1], ts))
                    xs.append(q[0] + rng.normal(0, 0.03 * ts))
                    ys.append(q[1] + rng.normal(0, 0.03 * ts))
                    angs.append(aa + rng.normal(0, 0.05))
                    hus.append(0.5 * ts * (0.95 + rng.uniform(-0.05, 0.05)))
                    hvs.append(0.5 * ts * (0.95 + rng.uniform(-0.05, 0.05)))
                    gl.append(gi)
                    order.append(k)
                    k += 1
        centers.append(pen + w * letter_h * 0.5)
        pen += (w + tracking) * letter_h
    adv = pen - tracking * letter_h
    return dict(x=np.array(xs), y=np.array(ys), ang=np.array(angs), hu=np.array(hus), hv=np.array(hvs),
                glyph=np.array(gl, int), order=np.array(order, int)), adv, np.array(centers)


class Band:
    """a titulus band in texture units: width x height, text centred at x=cx (or several (text, cx) runs).
    ground: palette key of the field ('gold' = gold leaf), ink: letter colour key."""

    def __init__(self, runs, width, height=90.0, letter_h=50.0, tile=8.0, letter_tile=5.4, ground="gold",
                 ink="lapis_d", rule="porphyry", pearl="marble", seed=0, border=True, guide_scale=1.0,
                 tracking=0.16):
        self.W, self.H = float(width), float(height)
        rng = np.random.default_rng(seed)
        # ---------------------------------------------------------- letters
        LX, LY, LA, LU, LV, LG, LO = [], [], [], [], [], [], []
        self.glyph_x = []
        self.text = ""
        gbase = 0
        for text, cx in runs:
            t, adv, cen = lay_text(text, letter_h, letter_tile, tracking, seed + len(self.text))
            x0 = cx - adv / 2
            y0 = (height - letter_h) / 2
            LX.append(t["x"] + x0)
            LY.append(t["y"] + y0)
            LA.append(t["ang"])
            LU.append(t["hu"])
            LV.append(t["hv"])
            LG.append(t["glyph"] + gbase)
            LO.append(t["order"])
            self.glyph_x.extend(list(cen + x0))
            gbase += len(text)
            self.text += text
        lx = np.concatenate(LX) if LX else np.zeros(0)
        ly = np.concatenate(LY) if LY else np.zeros(0)
        la = np.concatenate(LA) if LA else np.zeros(0)
        lu = np.concatenate(LU) if LU else np.zeros(0)
        lv = np.concatenate(LV) if LV else np.zeros(0)
        lg = np.concatenate(LG) if LG else np.zeros(0, int)
        lo = np.concatenate(LO) if LO else np.zeros(0, int)
        self.glyph_x = np.array(self.glyph_x)
        # ---------------------------------------------------------- border rows (rule + pearls) top and bottom
        bx, by, ba, bu, bv, bc, bgd = [], [], [], [], [], [], []
        edge_pad = 0.0
        if border:
            rt = 5.0      # rule tile
            pt = 6.4      # pearl tile
            for side in (0, 1):
                yr = rt * 0.55 if side == 0 else height - rt * 0.55
                yp = rt * 1.1 + pt * 0.62 if side == 0 else height - rt * 1.1 - pt * 0.62
                n = int(width / (rt * 1.1))
                x = (np.arange(n) + 0.5) * width / n
                bx.append(x + rng.normal(0, 0.15, n))
                by.append(np.full(n, yr) + rng.normal(0, 0.15, n))
                ba.append(rng.normal(0, 0.04, n))
                bu.append(np.full(n, rt * 0.47))
                bv.append(np.full(n, rt * 0.47))
                bc.append(np.tile(SC[rule], (n, 1)))
                bgd.append(np.full(n, 1.0 if rule == "gold" else 0.0))
                n2 = int(width / (pt * 1.55))
                x2 = (np.arange(n2) + 0.5) * width / n2
                bx.append(x2)
                by.append(np.full(n2, yp))
                ba.append(np.full(n2, math.pi / 4) + rng.normal(0, 0.05, n2))
                bu.append(np.full(n2, pt * 0.40))
                bv.append(np.full(n2, pt * 0.40))
                bc.append(np.tile(SC[pearl], (n2, 1)))
                bgd.append(np.zeros(n2))
            edge_pad = rt * 1.1 + pt * 1.3
        # ---------------------------------------------------------- ground: one course hugging each letter, then
        # straight brick-bond rows (opus tessellatum) cut around them
        from scipy.spatial import cKDTree
        gap = 1.1
        s = 4.0 / max(1.0, letter_tile)          # mask px per unit (~4 px per letter tile)
        gw, gh = int(math.ceil(width * s)) + 2, int(math.ceil(height * s)) + 2
        mask = np.zeros((gh, gw), np.uint8)
        for x, y, a, u, v in zip(lx, ly, la, lu, lv):
            c, sn = math.cos(a), math.sin(a)
            pts = np.array([[x + c * du - sn * dv, y + sn * du + c * dv]
                            for du, dv in ((-u, -v), (u, -v), (u, v), (-u, v))]) * s
            cv2.fillConvexPoly(mask, np.round(pts).astype(np.int32), 255)
        ox, oy, oa, osz = [], [], [], []
        ct = tile * 0.82                          # the contour course is laid in slightly smaller tiles
        if len(lx):
            r = (gap + ct * 0.5) * s
            k = int(2 * round(r) + 1)
            dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            cnts, _ = cv2.findContours(dil, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for cn in cnts:
                P = cn[:, 0, :].astype(float) / s
                if len(P) < 4:
                    continue
                P = np.vstack([P, P[:1]])
                Q, A = _resample(P, ct * 1.04)
                for q, a in zip(Q[:-1], A[:-1]):
                    if not (edge_pad + ct * 0.45 < q[1] < height - edge_pad - ct * 0.45):
                        continue
                    ox.append(q[0])
                    oy.append(q[1])
                    oa.append(a + rng.normal(0, 0.04))
                    osz.append(ct * (0.92 + rng.uniform(-0.05, 0.04)))
        ox, oy, oa, osz = np.array(ox), np.array(oy), np.array(oa), np.array(osz)
        if len(ox):
            # thin out the contour course where it crowds itself (inner corners)
            keepo = np.ones(len(ox), bool)
            tr_o = cKDTree(np.stack([ox, oy], 1))
            for i, js in enumerate(tr_o.query_ball_point(np.stack([ox, oy], 1), ct * 0.8)):
                if keepo[i]:
                    for j in js:
                        if j > i:
                            keepo[j] = False
            ox, oy, oa, osz = ox[keepo], oy[keepo], oa[keepo], osz[keepo]
        # straight rows
        rows_y = np.arange(edge_pad + tile * 0.55, height - edge_pad - tile * 0.45, tile * 1.02)
        rx, ry = [], []
        for ri_, yy in enumerate(rows_y):
            off = (0.5 * tile if ri_ % 2 else 0.0) + rng.uniform(0, 0.3) * tile
            xs = np.arange(off, width + tile, tile * 1.03)
            rx.append(xs)
            ry.append(np.full(len(xs), yy))
        rx = np.concatenate(rx) + rng.normal(0, 0.05 * tile, sum(len(a) for a in ry))
        ry = np.concatenate(ry) + rng.normal(0, 0.04 * tile, len(rx))
        keep = (rx > tile * 0.3) & (rx < width - tile * 0.3)
        occ = np.vstack([np.stack([lx, ly], 1), np.stack([ox, oy], 1)]) if len(lx) else np.zeros((0, 2))
        if len(occ):
            d, _ = cKDTree(occ).query(np.stack([rx, ry], 1))
            keep &= d > tile * 1.02
        rx, ry = rx[keep], ry[keep]
        gx = np.concatenate([ox, rx])
        gy = np.concatenate([oy, ry])
        gang = np.concatenate([oa, rng.normal(0, 0.035, len(rx))])
        gsz = np.concatenate([osz, tile * (0.9 + rng.uniform(-0.05, 0.04, len(rx)))])
        ng = len(gx)
        gcol = np.tile(SC[ground], (ng, 1))
        ggold = np.full(ng, 1.0 if ground == "gold" else 0.0)
        # ---------------------------------------------------------- assemble
        nl = len(lx)
        ink_c = np.tile(SC[ink], (nl, 1)) if ink != "gold" else np.tile(SC["gold"], (nl, 1))
        ink_g = np.full(nl, 1.0 if ink == "gold" else 0.0)
        self.tiles = dict(
            x=np.concatenate([lx, gx] + bx), y=np.concatenate([ly, gy] + by),
            ang=np.concatenate([la, gang] + ba),
            hu=np.concatenate([lu, gsz * 0.5] + bu), hv=np.concatenate([lv, gsz * 0.5] + bv),
            rgb=np.vstack([ink_c, gcol] + bc).astype(np.float32),
            gold=np.concatenate([ink_g, ggold] + bgd),
            letter=np.concatenate([np.ones(nl, bool), np.zeros(ng + sum(len(b) for b in bx), bool)]),
            glyph=np.concatenate([lg, -np.ones(ng + sum(len(b) for b in bx), int)]),
            order=np.concatenate([lo, np.zeros(ng + sum(len(b) for b in bx), int)]),
        )
        self.n = len(self.tiles["x"])

    def set_times(self, glyph_windows, jitter=0.012, seed=0):
        """glyph_windows: {glyph_index: (t0, t1)} -> t_set per tile (letters set in stroke order across the
        window; other tiles -inf)."""
        rng = np.random.default_rng(seed)
        T = np.full(self.n, -1e9)
        g, o = self.tiles["glyph"], self.tiles["order"]
        for gi, (t0, t1) in glyph_windows.items():
            m = np.nonzero(g == gi)[0]
            if len(m) == 0:
                continue
            oo = o[m].astype(float)
            fr = oo / max(1.0, oo.max())
            T[m] = t0 + fr * (t1 - t0) + rng.uniform(0, jitter, len(m))
        return T


def pop(t, t_set, dur=0.16):
    """setting animation of a tile: (visible(bool), scale, flash 0..1) - a tile arrives a little large and
    bright (pressed into the bed), settles to size and colour over `dur` seconds."""
    a = (np.asarray(t, float) - t_set) / dur
    vis = a >= 0
    a = np.clip(a, 0, 1)
    scale = 1.0 + 0.55 * (1 - a) ** 2.2
    flash = (1 - a) ** 1.6
    return vis & (a < 1), scale, flash

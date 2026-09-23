"""s06 helper: LORD EGREGORE — a colossal faceless hooded mass woven from stacked screens of static.

Billboard in local units (1 unit = 1 world unit): origin = the tower's top centre, +y down.
Construction: concentric rings of screens frame the empty face (the hood), pointed peak above;
below, rows of screens drape over the shoulders in folds down to a ragged hem that hangs over the
top tiers, with cables dangling into the tower. Every screen flickers static; a few show slop,
dead glass, no-signal blue or glitch bars. Faceless: the opening is a void.
"""
import math

import cairo
import numpy as np

from vx import clamp, hash01

FX, FY = 30.0, -1660.0          # face centre
FA, FB = 330.0, 400.0           # face half width / half height
NR = 8                          # hood rings
HEM_Y = 760.0
A0, A1 = -1.13 * math.pi, 0.13 * math.pi   # hood arc (through the top, y down => negative = up)


def face_pt(alpha, k):
    """point on hood ring k (k=0: the face opening) at angle alpha"""
    s = 1.0 + 0.13 * k
    up = 1.0 + 0.075 * k
    c, sn = np.cos(alpha), np.sin(alpha)
    top = np.clip(-sn, 0, 1)
    x = FA * s * c * (1 - 0.33 * top ** 2)
    y = np.where(sn > 0, FB * s * sn, -FB * s * up * top ** 0.9)
    lean = 22.0 * k * top            # the cowl's tip bends toward screen-right
    return FX + x + lean, FY + y


from scipy.interpolate import PchipInterpolator as _Pchip

_SH = _Pchip([-1460, -1250, -1050, -750, -200, 400, 900], [690, 860, 960, 1040, 1200, 1400, 1560])


def shoulder_x(y):
    return _SH(np.clip(np.asarray(y, float), -1460, 900))


def fold(u, v):
    ph = u * 4.6
    d = (110 * np.sin(ph + 0.7) + 50 * np.sin(ph * 2.3 + 2.1) + 20 * np.sin(ph * 4.1)) * v
    ds = (110 * np.cos(ph + 0.7) + 50 * 2.3 * np.cos(ph * 2.3 + 2.1) + 20 * 4.1 * np.cos(ph * 4.1)) * v / 150
    return d, ds


def cloak_xy(u, y):
    v = np.clip((y + 1420.0) / (HEM_Y + 1420.0), 0, 1)
    w = shoulder_x(y)
    d, _ = fold(u, v)
    x = u * w
    droop = 400.0 * np.clip((np.abs(x) - 560.0) / 700.0, 0, 1) ** 1.2 * (1 - v) ** 2
    sag = 40.0 * np.sin(u * 9.0 + 1.0) * v - 170.0 * u * u * v * v
    return x + d * np.clip(v * 3, 0, 1), y + droop + sag


class Egregore:
    def __init__(self, seed=13):
        rng = np.random.default_rng(seed)
        Q = []   # (x0..x3, y0..y3, type, rnd, row_key, shade, region)

        def typ():
            r = rng.random()
            return 0 if r < 0.76 else (1 if r < 0.89 else (4 if r < 0.92 else (3 if r < 0.975 else 2)))

        # ---- cloak rows (drawn first; the hood rings overlap their top)
        ys = np.linspace(-1440.0, HEM_Y, 19)
        ys[1:-1] += rng.uniform(-18, 18, len(ys) - 2)
        for j in range(len(ys) - 1):
            y0, y1 = ys[j], ys[j + 1]
            if j == len(ys) - 2:
                pass
            ym = (y0 + y1) / 2
            w = float(shoulder_x(ym))
            n = max(2, int(round(2 * w / rng.uniform(150, 210))))
            us = np.linspace(-1, 1, n + 1)
            us[1:-1] += rng.uniform(-0.3, 0.3, n - 1) / n
            i = 0
            while i < n:
                span = 2 if (rng.random() < 0.1 and i < n - 2) else 1
                u0, u1 = us[i], us[i + span]
                last = len(ys) - 2
                hem = 0.0
                if j == last:
                    hem = rng.uniform(-0.2, 2.6) if rng.random() < 0.8 else -0.6
                elif j == last - 1 and rng.random() < 0.3:
                    hem = rng.uniform(0.3, 1.4)
                y1e = y1 + hem * (y1 - y0)
                xa, ya = cloak_xy(u0, y0)
                xb, yb = cloak_xy(u1, y0)
                xc, yc = cloak_xy(u1, y1e)
                xd, yd = cloak_xy(u0, y1e)
                cx = (xa + xb) / 2
                # skip cells hidden inside the hood / face
                inside = False
                if ym < -1250:
                    hw = FA * (1 + 0.13 * NR) * 0.98
                    inside = abs(cx - FX) < hw
                if not inside:
                    v = (ym + 1420) / (HEM_Y + 1420)
                    _, sl = fold((u0 + u1) / 2, v)
                    um = (u0 + u1) / 2
                    chest = math.exp(-((ym + 1250) / 260) ** 2) * math.exp(-(um / 0.45) ** 2)
                    shade = (1.0 + 0.4 * math.tanh(-sl * 1.3) * min(1, v * 3) - 0.3 * v - 0.5 * um * um
                             - 0.45 * chest)
                    Q.append((xa, xb, xc, xd, ya, yb, yc, yd, typ(), rng.random(), j + 100, shade, 1))
                i += span
        # ---- hood rings
        for k in range(NR):
            rk = (k + 0.5) / NR
            length = (A1 - A0) * FA * (1 + 0.13 * (k + 0.5)) * 1.25
            n = max(6, int(round(length / rng.uniform(150, 185))))
            al = np.linspace(A0, A1, n + 1)
            al[1:-1] += rng.uniform(-0.35, 0.35, n - 1) * (A1 - A0) / n
            for i in range(n):
                a0, a1 = al[i], al[i + 1]
                xa, ya = face_pt(a0, k)
                xb, yb = face_pt(a1, k)
                xc, yc = face_pt(a1, k + 1)
                xd, yd = face_pt(a0, k + 1)
                am = (a0 + a1) / 2
                topness = max(0.0, -math.sin(am))
                shade = (0.38 + 0.62 * rk ** 0.7) * (0.72 + 0.35 * topness)
                Q.append((xa, xb, xc, xd, ya, yb, yc, yd, typ() if k > 0 else 1 if rng.random() < 0.5 else 0,
                          rng.random(), k, shade, 0))
        A = np.array(Q, float)
        self.X = A[:, 0:4]
        self.Y = A[:, 4:8]
        self.typ = A[:, 8].astype(int)
        self.rnd = A[:, 9]
        self.row = A[:, 10].astype(int)
        self.shade = np.clip(A[:, 11], 0.18, 1.2)
        self.region = A[:, 12].astype(int)
        self.nc = len(A)
        self.tiles = [(rng.random((30, 40)) ** 0.85).astype(np.float32) for _ in range(12)]
        # outline of the whole mass (for the dark body)
        al = np.linspace(A0, A1, 60)
        hx, hy = face_pt(al, NR)
        ysh = np.linspace(float(hy[-1]), HEM_Y, 30)
        xr = [float(cloak_xy(1.0, y)[0]) + 20 for y in ysh]
        xl = [float(cloak_xy(-1.0, y)[0]) - 20 for y in ysh]
        pts = list(zip(xl[::-1], ysh[::-1])) + list(zip(hx, hy)) + list(zip(xr, ysh))
        self.outline = pts
        px, py = face_pt(-0.5 * math.pi, NR)
        self.peak = (float(px), float(py))
        ants = [(self.peak[0], self.peak[1] + 40, 820, 0.22, 0), (self.peak[0] - 150, self.peak[1] + 330, 640, -0.3, 1),
                (self.peak[0] + 190, self.peak[1] + 380, 560, 0.6, 2)]
        for side in (-1, 1):
            for q in range(3):
                x = side * (720 + 200 * q)
                y = float(-1300 + 150 * q + 60 * q * q)
                ants.append((x, y, rng.uniform(380, 700), side * rng.uniform(0.3, 0.75), int(rng.integers(0, 3))))
        self.ants = ants
        self.cables = [(rng.uniform(-0.95, 0.95), rng.uniform(300, 1000), rng.uniform(-220, 220)) for _ in range(18)]

    def antenna_tips(self):
        return [(x + math.sin(a) * L, y - math.cos(a) * L) for (x, y, L, a, _) in self.ants]

    # --------------------------------------------------------------------------------------------
    def draw(self, ctx, gctx, x, y, s, t, on=1.0, glitch=0.0, flash=0.0, fog=0.0, fog_col=(0.1, 0.1, 0.2),
             rot=0.0, lod=1.0):
        """draw at screen (x, y) = tower top centre, scale s (px per unit). on = power-on cascade 0..1"""
        breathe = 1 + 0.012 * math.sin(t * 0.9)
        for c in (ctx, gctx):
            if c is None:
                continue
            c.save()
            c.translate(x, y)
            c.rotate(rot)
            c.scale(s * breathe, s * breathe)
        fc = np.asarray(fog_col, float)

        def fogc(c, k=1.0):
            c = np.asarray(c, float) * k
            return tuple(c * (1 - fog) + fc * fog)

        # dark body
        ctx.move_to(*self.outline[0])
        for p in self.outline[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.set_source_rgb(*fogc((0.03, 0.03, 0.05)))
        ctx.fill()
        self._antennas(ctx, gctx, t, fogc, flash)
        # cables into the tower
        ctx.set_line_width(13)
        for (u, L, sw) in self.cables:
            x0, y0 = cloak_xy(u, HEM_Y - 60)
            ctx.move_to(float(x0), float(y0))
            ctx.curve_to(x0 + sw * 0.3, y0 + L * 0.6, x0 + sw, y0 + L * 0.8, x0 + sw * 1.2, y0 + L)
            ctx.set_source_rgb(*fogc((0.05, 0.05, 0.08)))
            ctx.stroke()
        fr = int(t * 24)
        # power-on cascade (hem -> peak) + per-screen jitter
        yn = (self.Y.mean(1) - HEM_Y) / (-2960.0 - HEM_Y)
        t_on = np.clip(on * 1.45 - yn * 0.45 - self.rnd * 0.25, 0, 1)
        dxg = np.zeros(self.nc)
        if glitch > 0:
            band = np.array([hash01(int(r) * 13 + fr // 2, 21) for r in self.row])
            dxg = np.where(band > 0.5, (band - 0.5) * 1000 * glitch * np.sin(self.row * 1.7 + t * 37), 0)
        bez = fogc((0.07, 0.07, 0.10))
        for i in range(self.nc):
            xs, ys = self.X[i] + dxg[i], self.Y[i]
            ctx.move_to(xs[0], ys[0])
            ctx.line_to(xs[1], ys[1])
            ctx.line_to(xs[2], ys[2])
            ctx.line_to(xs[3], ys[3])
            ctx.close_path()
            ctx.set_source_rgb(*bez)
            ctx.fill()
            ton = t_on[i]
            cx, cy = xs.mean(), ys.mean()
            g = 0.84
            ix, iy = cx + (xs - cx) * g, cy + (ys - cy) * g
            if ton <= 0:
                # not yet awake: dead glass reflecting the storm, so the structure reads before power-on
                self._screen(ctx, None, ix, iy, 1, self.rnd[i], i, fr, 1.0, 0.0, fog, fc, flash, lod, s)
                continue
            ty = int(self.typ[i])
            flick = 0.72 + 0.28 * hash01(i * 7 + fr, 3)
            if glitch > 0 and hash01(i * 3 + fr // 2, 4) < glitch * 0.5:
                ty = 5 if hash01(i + fr, 6) < 0.45 else 2
            if hash01(i * 11 + fr // 4, 5) < 0.006:
                ty = 2
            k = self.shade[i] * flick * min(1.0, ton * 1.6) * (1 + 0.6 * flash)
            crt = 1.0 - clamp(ton * 3.0)
            self._screen(ctx, gctx, ix, iy, ty, self.rnd[i], i, fr, k, crt, fog, fc, flash, lod, s)
        self._face(ctx, gctx, t, fogc, flash, on)
        for c in (ctx, gctx):
            if c is not None:
                c.restore()

    def _screen(self, ctx, gctx, xs, ys, typ, rnd, i, fr, k, crt, fog, fc, flash, lod, s):
        xx = ((xs[1] - xs[0]) + (xs[2] - xs[3])) / 2
        yx = ((ys[1] - ys[0]) + (ys[2] - ys[3])) / 2
        xy = ((xs[3] - xs[0]) + (xs[2] - xs[1])) / 2
        yy = ((ys[3] - ys[0]) + (ys[2] - ys[1])) / 2
        if abs(xx * yy - xy * yx) < 1e-3:
            return
        M = cairo.Matrix(xx, yx, xy, yy, xs[0], ys[0])
        ctx.save()
        ctx.transform(M)
        ctx.rectangle(0, 0, 1, 1)
        ctx.clip()
        fk = 1 - fog * 0.8
        glow_c = None
        if typ == 0:       # static
            tint = [(0.80, 0.83, 0.90), (0.70, 0.78, 0.92), (0.82, 0.80, 0.88), (0.72, 0.88, 0.95)][int(rnd * 4)]
            j = (i * 5 + fr) % len(self.tiles)
            sz = math.hypot(xx, yx) * s
            if sz * lod > 5:
                tile = self.tiles[j]
                surf = _tile_surface(tile, j, tint, k * fk)
                ctx.scale(1 / tile.shape[1], 1 / tile.shape[0])
                pat = cairo.SurfacePattern(surf)
                pat.set_filter(cairo.FILTER_NEAREST if sz > 28 else cairo.FILTER_GOOD)
                ctx.set_source(pat)
                ctx.paint()
            else:
                ctx.set_source_rgb(*(np.array(tint) * 0.45 * k * fk + fc * fog))
                ctx.paint()
            glow_c = np.array(tint) * 0.42 * k
        elif typ == 1:     # dead glass reflecting the storm
            ctx.set_source_rgb(*(np.array((0.045, 0.05, 0.08)) * (1 - fog) + fc * fog))
            ctx.paint()
            ctx.move_to(0.12, 0.0)
            ctx.line_to(0.38, 0.0)
            ctx.line_to(0.12, 0.55)
            ctx.close_path()
            ctx.set_source_rgba(0.5, 0.55, 0.75, 0.1 + 0.35 * flash)
            ctx.fill()
        elif typ == 2:     # glitch bars
            cols = [(1.0, 0.18, 0.53), (0.0, 0.9, 1.0), (0.92, 0.92, 1.0), (0.3, 0.3, 0.9)]
            for b in range(5):
                c = cols[int(hash01(i * 17 + b + fr, 8) * 4)]
                ctx.rectangle(0, b / 5, 1, 0.21)
                ctx.set_source_rgb(*(np.array(c) * min(1, k) * fk + fc * fog))
                ctx.fill()
            glow_c = np.array((0.6, 0.4, 0.8)) * k
        elif typ == 3:     # slop feed card
            c = [(0.91, 0.78, 0.82), (0.72, 0.88, 0.91), (0.54, 0.5, 0.56)][int(rnd * 3)]
            ctx.set_source_rgb(*(np.array(c) * min(1, k) * fk + fc * fog))
            ctx.paint()
            ctx.set_source_rgba(1, 1, 1, 0.35 * min(1, k))
            ctx.arc(0.5, 0.4, 0.22, 0, 2 * math.pi)
            ctx.fill()
            ctx.set_source_rgba(0.1, 0.1, 0.15, 0.35)
            ctx.rectangle(0.15, 0.72, 0.7, 0.07)
            ctx.rectangle(0.15, 0.84, 0.45, 0.07)
            ctx.fill()
            glow_c = np.array(c) * 0.35 * k
        elif typ == 4:     # no-signal blue
            ctx.set_source_rgb(*(np.array((0.12, 0.2, 0.62)) * min(1, k) * fk + fc * fog))
            ctx.paint()
            ctx.set_source_rgba(0.6, 0.7, 1.0, 0.3 * min(1, k))
            ctx.rectangle(0, 0.45, 1, 0.1)
            ctx.fill()
            glow_c = np.array((0.12, 0.2, 0.62)) * k
        else:              # blown-out white
            ctx.set_source_rgb(0.95 * fk + fc[0] * fog, 0.95 * fk + fc[1] * fog, 1.0 * fk + fc[2] * fog)
            ctx.paint()
            glow_c = np.array((0.8, 0.8, 1.0))
        if crt > 0:
            ctx.set_source_rgba(0.9, 0.95, 1.0, crt * 0.8)
            ctx.paint()
            ctx.rectangle(0, 0.46, 1, 0.08)
            ctx.set_source_rgba(1, 1, 1, crt)
            ctx.fill()
        ctx.restore()
        if gctx is not None and (glow_c is not None or crt > 0):
            g = (glow_c if glow_c is not None else np.zeros(3)) * fk + crt * 0.7
            gctx.save()
            gctx.transform(M)
            gctx.rectangle(0, 0, 1, 1)
            gctx.set_source_rgba(min(1, g[0]), min(1, g[1]), min(1, g[2]), 0.85)
            gctx.fill()
            gctx.restore()

    def _face(self, ctx, gctx, t, fogc, flash, on):
        al = np.linspace(0, 2 * math.pi, 90)
        x, y = face_pt(al, 0)
        ctx.move_to(x[0], y[0])
        for a, b in zip(x[1:], y[1:]):
            ctx.line_to(a, b)
        ctx.close_path()
        g = cairo.RadialGradient(FX, FY + 60, 20, FX, FY, FA * 1.3)
        g.add_color_stop_rgb(0, 0.0, 0.0, 0.0)
        g.add_color_stop_rgb(0.75, 0.005, 0.005, 0.012)
        g.add_color_stop_rgb(1, 0.05, 0.05, 0.08)
        ctx.set_source(g)
        ctx.fill()
        # the lip of the hood catches cold light
        a2 = np.linspace(A0 + 0.05, A1 - 0.05, 70)
        x, y = face_pt(a2, 0.05)
        for c, w, a in ((ctx, 10, 0.55), (gctx, 26, 0.35 * on)):
            if c is None:
                continue
            c.set_line_width(w)
            c.move_to(x[0], y[0])
            for p, q in zip(x[1:], y[1:]):
                c.line_to(p, q)
            c.set_source_rgba(0.6, 0.72, 1.0, a * (0.5 + 0.5 * on) * (1 + flash))
            c.stroke()

    def _antennas(self, ctx, gctx, t, fogc, flash):
        dark = fogc((0.055, 0.055, 0.085))
        for j, (x, y, L, a, typ) in enumerate(self.ants):
            aa = a + 0.03 * math.sin(t * 1.3 + j)
            tx, ty = x + math.sin(aa) * L, y - math.cos(aa) * L
            ctx.set_source_rgb(*dark)
            ctx.set_line_width(16)
            ctx.move_to(x, y)
            ctx.line_to(tx, ty)
            ctx.stroke()
            nx, ny = math.cos(aa), math.sin(aa)
            if typ == 0:
                ctx.set_line_width(9)
                for q in range(5):
                    f = 0.45 + q * 0.12
                    px, py = x + (tx - x) * f, y + (ty - y) * f
                    w = 120 - q * 16
                    ctx.move_to(px - nx * w, py - ny * w)
                    ctx.line_to(px + nx * w, py + ny * w)
                ctx.stroke()
            elif typ == 1:
                px, py = x + (tx - x) * 0.7, y + (ty - y) * 0.7
                ctx.save()
                ctx.translate(px, py)
                ctx.rotate(aa - 0.9)
                ctx.scale(1, 0.45)
                ctx.arc(0, 0, 150, 0, math.pi)
                ctx.restore()
                ctx.fill()
            else:
                ctx.set_line_width(6)
                for q in range(8):
                    f0, f1 = q / 8, (q + 1) / 8
                    ctx.move_to(x + (tx - x) * f0 + nx * 30 * (1 - f0), y + (ty - y) * f0 + ny * 30 * (1 - f0))
                    ctx.line_to(x + (tx - x) * f1 - nx * 30 * (1 - f1), y + (ty - y) * f1 - ny * 30 * (1 - f1))
                ctx.stroke()
            blink = 0.5 + 0.5 * math.sin(t * 5.5 + j * 1.3)
            if blink > 0.35:
                ctx.set_source_rgb(0.9, 0.12 + 0.1 * blink, 0.2)
                ctx.arc(tx, ty, 20, 0, 2 * math.pi)
                ctx.fill()
                if gctx is not None:
                    gctx.set_source_rgba(1.0, 0.15, 0.25, 0.8 * blink)
                    gctx.arc(tx, ty, 50, 0, 2 * math.pi)
                    gctx.fill()


_TILES = {}


def _tile_surface(tile, key, tint, k):
    """cached ARGB surface for a static tile, tint baked in (quantised to limit cache)"""
    kq = int(max(0.0, min(1.6, k)) * 12)
    ck = (key, tuple(int(c * 20) for c in tint), kq)
    e = _TILES.get(ck)
    if e is None:
        h, w = tile.shape
        a = np.clip(tile[..., None] * np.array(tint)[None, None, :] * (kq / 12.0) * 1.3, 0, 1)
        stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
        b2 = np.zeros((h, stride // 4, 4), np.uint8)
        b2[:, :w, 0] = (a[..., 2] * 255).astype(np.uint8)
        b2[:, :w, 1] = (a[..., 1] * 255).astype(np.uint8)
        b2[:, :w, 2] = (a[..., 0] * 255).astype(np.uint8)
        b2[:, :w, 3] = 255
        e = (cairo.ImageSurface.create_for_data(memoryview(b2), cairo.FORMAT_ARGB32, w, h, stride), b2)
        if len(_TILES) > 4000:
            _TILES.clear()
        _TILES[ck] = e
    return e[0]

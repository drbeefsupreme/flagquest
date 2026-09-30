"""The two non-human icons of vx.icon: the HYPERMIND (an ophanim-polycandelon) and LORD EGREGORE (a faceless
anti-Pantocrator of static and billboards). Owner: Iconographer. API: see the vx.icon manual.

Hypermind(seed=0).draw(ctx, x, y, h, t=0.0, state='idle', wake=None, canons=None, spin=None, look=(0, 0), flag=0.0,
                       layer='color', tile=None, part='all', alpha=1.0, tint=None) -> anchors
    (x, y) = centre of the wheels, h = height of the machine (top plate .. bottom of the mixing chamber is ~0.9 h;
    chains run up out of frame, wings span ~1.9 h).
    state   'idle' (eyes shut, wings folded + dark, wheels still) | 'training' (wheels turning, eyes opening and
            darting, the six wings igniting one per canon: canons 0..6) | 'compiling' (fast wheels, every eye wide,
            all wings aflame) | 'done' (wheels locked, every eye looking down at the Flag, wings spread)
    wake    0..1 eyelids (default from state), canons 0..6 lit wings (fractional = igniting),
    spin    wheel rotation offset override (radians), look (dx, dy) gaze direction or None (state default),
    flag    0..1 descent of the Flag out of the aperture (anchors['pole'] follows), part 'back' (behind the Flag:
            everything) / 'front' (only the lower lip of the aperture, to composite over the emerging Flag).
    anchors: center, top (chain anchor), aperture (x, y, rx, ry), pole (base_x, base_y, 0.0) for the descending Flag
             (base = aperture + flag * drop, drop = 0.95 h), eyes [(x, y, w, open)], wings [(root_x, root_y, tip_x,
             tip_y, lit)], wheels [(cx, cy, rx, ry, rot)], lamps [(x, y)], bbox.
Egregore(seed=0).draw(ctx, x, y, h, t=0.0, layer='color', tile=None, glitch=0.0, alpha=1.0, tint=None) -> anchors
    a colossal bust (Pantocrator composition, for a dome): (x, y) = bottom centre of the bust, h = bust height (to the
    top of the anti-halo). Face = static (animated by t), halo = a ring of screens, robe = stacked billboards, right
    hand raises a phone (anti-blessing), left arm holds a glowing tablet (anti-Gospel). glitch 0..1 slices the image.
    anchors: face (x, y, r), halo (x, y, r), phone (x, y), tablet (x0, y0, x1, y1), screens [(x0, y0, x1, y1)], bbox.
"""
import math

import numpy as np

from .ease import hash01, clamp
from .icon import Painter, rgb, mixc, TAU, lerp2, auto_blink


def _hs(seed, k):
    return hash01(k, 3000 + seed)


# ================================================================ HYPERMIND
_STATES = {
    "idle": dict(wake=0.0, canons=0.0, rate=0.0, lamps=0.25, dart=0.0),
    "training": dict(wake=0.75, canons=3.0, rate=0.35, lamps=0.7, dart=1.0),
    "compiling": dict(wake=1.0, canons=6.0, rate=1.3, lamps=1.0, dart=0.4),
    "done": dict(wake=1.0, canons=6.0, rate=0.0, lamps=1.0, dart=0.0),
}
# the six seraphic wings, one per holy canon, in lighting order: (screen angle of the wing axis in degrees)
_WING_ANG = (120.0, 60.0, 180.0, 0.0, 240.0, 300.0)
R0, R1, R2 = 38.0, 31.0, 25.0


class Hypermind:
    kind = "hypermind"

    def __init__(self, seed=0, **style):
        self.seed = int(seed)
        self.style = dict(style)

    def box(self, h):
        return (-1.0 * h, -2.2 * h, 1.0 * h, 1.1 * h)

    def _params(self, t, state, wake, canons, spin, look):
        st = _STATES.get(state, _STATES["idle"])
        wk = st["wake"] if wake is None else float(wake)
        cn = st["canons"] if canons is None else float(canons)
        sp = (t * st["rate"]) if spin is None else float(spin)
        if look is None or (state == "done" and tuple(look) == (0.0, 0.0)):
            look = (0.0, 0.9) if state == "done" else (0.0, 0.0)
        return wk, cn, sp, look, st

    def _wheels(self, spin):
        """(radius, ry, rot, index, phase) of the three wheels within wheels"""
        out = [(R0, 1.0, 0.0, 0, -spin * 0.5)]
        for i, (r, ax, rate, ph) in enumerate(((R1, 0.5, 1.0, 0.6), (R2, -1.0, -1.35, 1.9)), 1):
            ang = spin * rate + ph
            ry = 0.18 + 0.82 * abs(math.cos(ang))
            rot = ax + 0.2 * math.sin(spin * 0.6 * rate + ph)
            out.append((r, ry, rot, i, spin * (0.7 + 0.3 * i) * (1 if i % 2 else -1)))
        return out

    def _eyes_on(self, r, ry, rot, i, ph):
        n = (12, 10, 8)[i]
        c, s = math.cos(rot), math.sin(rot)
        out = []
        for k in range(n):
            a = ph + k * TAU / n
            px, py = r * math.cos(a), r * ry * math.sin(a)
            front = True if i == 0 else math.sin(a) >= 0.0
            sc = 1.0 if i == 0 else 0.45 + 0.55 * abs(math.sin(a)) * (1.0 if ry > 0.5 else 0.8)
            out.append((c * px - s * py, s * px + c * py, front, sc, k))
        return out

    def anchors(self, x, y, h, t=0.0, state="idle", wake=None, canons=None, spin=None, look=(0.0, 0.0), flag=0.0,
                **kw):
        wk, cn, sp, look, st = self._params(t, state, wake, canons, spin, look)
        u = h / 100.0
        W = lambda p: (x + p[0] * u, y + p[1] * u)
        a = {"center": (x, y), "scale": u, "h": h, "kind": "hypermind"}
        a["top"] = W((0.0, -R0 - 3.0))
        ap = W((0.0, 30.0))
        a["aperture"] = (ap[0], ap[1], 6.0 * u, 2.0 * u)
        a["pole"] = (ap[0], ap[1] + clamp(flag, 0.0, 1.0) * 0.95 * h, 0.0)
        wh = self._wheels(sp)
        a["wheels"] = [(x, y, r * u, r * ry * u, rot) for r, ry, rot, i, _ in wh]
        eyes = []
        for r, ry, rot, i, ph in wh:
            for ex, ey, front, sc, k in self._eyes_on(r, ry, rot, i, ph):
                eyes.append((*W((ex, ey)), (8.0, 6.5, 5.5)[i] * sc * u, wk if front else 0.0))
        eyes.append((*W((0.0, 3.0)), 13.0 * u, wk))
        a["eyes"] = eyes
        wings = []
        for k, ang in enumerate(_WING_ANG):
            lit = clamp(cn - k, 0.0, 1.0)
            c_, s_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            root = W((c_ * (R0 + 2.0), s_ * (R0 + 2.0)))
            tip = W((c_ * (R0 + 52.0), s_ * (R0 + 52.0)))
            wings.append((root[0], root[1], tip[0], tip[1], lit))
        a["wings"] = wings
        a["lamps"] = [W(p) for p in self._lamps()]
        a["bbox"] = (x - 0.95 * h, y - 0.95 * h, x + 0.95 * h, y + 0.95 * h)
        return a

    def _lamps(self):
        pts = []
        for yy, r, n in ((-22.0, 19.0, 7), (-6.0, 14.0, 5)):
            for k in range(n):
                a = math.pi * (0.12 + 0.76 * k / (n - 1))
                pts.append((r * math.cos(math.pi - a), yy + 0.25 * r * math.sin(a) + 4.5))
        return pts

    def draw(self, ctx, x, y, h, t=0.0, state="idle", wake=None, canons=None, spin=None, look=(0.0, 0.0), flag=0.0,
             layer="color", tile=None, part="all", alpha=1.0, tint=None, mouth=0.0, **kw):
        wk, cn, sp, look, st = self._params(t, state, wake, canons, spin, look)
        u = h / 100.0
        P = Painter(ctx, layer, tile, alpha, tint)
        P.push(x, y, 0.0, u)
        P.lod = 0 if ((tile and h / tile > 60) or (not tile and h > 420)) else 1
        wh = self._wheels(sp)
        if part in ("all", "back"):
            self._chains(P)
            for k, ang in enumerate(_WING_ANG):
                lit = clamp(cn - k, 0.0, 1.0)
                spread = 0.55 + 0.45 * max(lit, 0.35 * wk)
                self._wing(P, ang, spread, lit, t, k, wk)
            for w_ in wh[1:]:
                self._wheel(P, w_, wk, look, t, "back", st["dart"])
            self._stack(P, t, st["lamps"] * (0.35 + 0.65 * wk))
            self._core(P, t, wk, look)
            for w_ in wh[1:]:
                self._wheel(P, w_, wk, look, t, "front", st["dart"])
            self._wheel(P, wh[0], wk, look, t, "front", st["dart"])
        if part in ("all", "front"):
            self._aperture_lip(P)
        P.pop()
        return self.anchors(x, y, h, t=t, state=state, wake=wake, canons=canons, spin=spin, look=look, flag=flag)

    # ------------------------------------------------------------ parts
    def _chains(self, P):
        for xx in (-0.55, 0.0, 0.55):
            bot = (xx * R0 * 0.9, -R0 * (0.84 if xx else 1.0) - 1.0)
            top = (xx * 90.0, -260.0)
            n = 28
            for k in range(n):
                p, q = lerp2(top, bot, k / n), lerp2(top, bot, (k + 1) / n)
                m = lerp2(p, q, 0.5)
                ang = math.atan2(q[1] - p[1], q[0] - p[0])
                L = math.hypot(q[0] - p[0], q[1] - p[1])
                if k % 2:
                    P.ellipse(m[0], m[1], L * 0.6, 1.3, ang, fill="gold", line="gold_d", lw=0.45, cls="fold", mat="gold")
                else:
                    P.ellipse(m[0], m[1], L * 0.6, 0.55, ang, fill="gold_d")

    def _wing(self, P, ang_deg, spread, lit, t, k, wk):
        a0 = math.radians(ang_deg)
        c, s = math.cos(a0), math.sin(a0)
        root = (c * (R0 - 2.0), s * (R0 - 2.0))
        nf = 6
        cool = ("lapis_d", "lapis", "lapis_l")
        hot = ("red_d", "red", "red_l")
        fl = 0.5 + 0.5 * math.sin(t * 8.0 + k * 1.3)
        fan = math.radians(26.0) * spread
        for j in range(nf):
            f = j / (nf - 1) - 0.5
            a = a0 + f * 2.0 * fan
            L = (40.0 + 16.0 * (1.0 - abs(f) * 1.6)) * (0.8 + 0.2 * spread)
            dx, dy = math.cos(a), math.sin(a)
            nx, ny = -dy, dx
            w = 7.0
            base = (root[0] + nx * f * 6.0, root[1] + ny * f * 6.0)
            tip = (base[0] + dx * L, base[1] + dy * L)
            mid = lerp2(base, tip, 0.62)
            pts = [(base[0] + nx * w * 0.35, base[1] + ny * w * 0.35), (mid[0] + nx * w * 0.5, mid[1] + ny * w * 0.5),
                   tip, (mid[0] - nx * w * 0.5, mid[1] - ny * w * 0.5), (base[0] - nx * w * 0.35, base[1] - ny * w * 0.35)]
            col_ = mixc(cool[j % 3], hot[j % 3], lit)
            P.poly(pts, fill=col_, line="ink", lw=0.7, cls="contour", mat="glow" if lit > 0.5 else None, smooth=True)
            # the flame / highlight edge of each feather, and a gold vane
            if P.lod == 0 or lit > 0.5:
                edge = [lerp2(base, tip, 0.2), (mid[0] + nx * w * 0.28, mid[1] + ny * w * 0.28), lerp2(base, tip, 0.93)]
                P.line(edge, mixc("white_l", "gold_l", lit), 0.9, "light", "gold glow" if lit > 0.5 else None)
            P.line([lerp2(base, tip, 0.1), lerp2(base, tip, 0.85)], "gold", 0.6, "gold", "gold", smooth=False)
            if j in (1, 4):
                e = lerp2(base, tip, 0.58)
                self._eye(P, e[0], e[1], 5.5, (0.25 + 0.75 * wk) * (1.0 - auto_blink(t + j, k * 7 + j)), (0.0, 0.0),
                          a - math.pi / 2)
            if lit > 0.5 and P.lod == 0:
                tipf = (tip[0] + dx * 3.0 * fl, tip[1] + dy * 3.0 * fl)
                P.poly([lerp2(base, tip, 0.9), (tip[0] + nx * 1.5, tip[1] + ny * 1.5), tipf, (tip[0] - nx * 1.5, tip[1] - ny * 1.5)],
                       fill="gold_l", mat="gold glow", smooth=True)

    def _stack(self, P, t, lamps):
        plates = ((-26.0, 19.0), (-10.0, 14.0), (4.0, 10.5), (16.0, 8.0), (25.0, 6.0))
        for (y0, r0), (y1, r1) in zip(plates[:-1], plates[1:]):
            for xx in (-0.6, 0.6):
                P.line([(xx * r0 * 0.8, y0), (xx * r1 * 0.8, y1)], "gold_d", 1.3, "fold", smooth=False)
            if P.lod == 0:
                pts = []
                for k in range(21):
                    f = k / 20.0
                    yy = y0 + (y1 - y0) * f
                    rr = (r0 + (r1 - r0) * f) * 0.4
                    pts.append((rr * math.sin(f * 12.0 + t * 0.6), yy))
                P.line(pts, "silver", 0.7, "fold", "silver")
        for yy, r in plates:
            ry = 0.24 * r
            P.ellipse(0.0, yy + 1.3, r, ry, fill="gold_d")
            P.ellipse(0.0, yy, r, ry, fill="gold", line="gold_d", lw=0.6, cls="contour", mat="gold")
        for (px, py) in self._lamps():
            flick = 0.75 + 0.25 * math.sin(t * 13.0 + px * 0.7)
            P.line([(px, py - 4.5), (px, py - 1.4)], "gold_d", 0.4, "thin", smooth=False)
            P.poly([(px - 1.7, py - 1.4), (px + 1.7, py - 1.4), (px + 0.9, py + 1.9), (px - 0.9, py + 1.9)],
                   fill=mixc("#2F5A45", "#E8F4D0", clamp(lamps * flick, 0, 1)), line="ink", lw=0.3, cls="fold",
                   mat="glow" if lamps > 0.3 else None, smooth=False)

    def _wheel(self, P, w_, wk, look, t, which, dart):
        r, ry, rot, i, ph = w_
        bw = (5.6, 4.6, 4.0)[i]
        c, s = math.cos(rot), math.sin(rot)
        n = 72
        outer, inner = [], []
        for k in range(n + 1):
            a = k * TAU / n
            so = math.sin(a)
            for rr, lst in ((r + bw / 2, outer), (r - bw / 2, inner)):
                px, py = rr * math.cos(a), rr * ry * so
                lst.append((c * px - s * py, s * px + c * py, so))
        seg = [k for k in range(n) if i == 0 or ((outer[k][2] >= 0.0) == (which == "front"))]
        if not seg:
            return
        ctx = P.ctx
        ctx.new_path()
        for k in seg:
            q = [outer[k][:2], outer[k + 1][:2], inner[k + 1][:2], inner[k][:2]]
            ctx.move_to(*q[0])
            for p in q[1:]:
                ctx.line_to(*p)
            ctx.close_path()
        P._src("gold", "gold", False)
        ctx.fill()
        # rims: draw contiguous runs
        runs, cur = [], []
        for k in seg:
            if cur and k != cur[-1] + 1:
                runs.append(cur)
                cur = []
            cur.append(k)
        if cur:
            runs.append(cur)
        for run in runs:
            ks = run + [run[-1] + 1]
            P.line([outer[k][:2] for k in ks], "gold_d", 0.6, "contour", smooth=False)
            P.line([inner[k][:2] for k in ks], "gold_d", 0.6, "contour", smooth=False)
        if which != "front":
            return
        for ex, ey, front, sc, k in self._eyes_on(r, ry, rot, i, ph):
            if not front:
                continue
            seed = i * 31 + k
            op = wk * (1.0 - auto_blink(t + 0.37 * k, seed))
            lx, ly = look
            if dart:
                lx += dart * 0.7 * math.sin(t * 1.3 + seed * 2.1)
                ly += dart * 0.5 * math.sin(t * 1.1 + seed * 1.3)
            self._eye(P, ex, ey, (8.0, 6.5, 5.5)[i] * sc, op, (clamp(lx, -1, 1), clamp(ly, -1, 1)), 0.0)

    def _eye(self, P, x, y, w, op, look, rot):
        from .icon_body import single_eye
        P.push(x, y, rot, w)
        single_eye(P, op, look, "gold", None, False)
        P.pop()

    def _core(self, P, t, wk, look):
        # the mixing chamber with the great central eye
        P.circle(0.0, 3.0, 9.0, fill="lapis_d", line="gold_d", lw=0.8, cls="contour")
        P.circle(0.0, 3.0, 9.0 - 1.2, line="gold", lw=1.2, cls="gold", mat="gold")
        self._eye(P, 0.0, 3.0, 13.0, wk * (1.0 - auto_blink(t, 99)), look, 0.0)
        P.poly([(-6.0, 22.0), (6.0, 22.0), (5.2, 30.0), (-5.2, 30.0)], fill="gold", line="gold_d", lw=0.6, cls="contour",
               mat="gold", smooth=False)
        P.ellipse(0.0, 30.0, 6.0, 2.0, fill="ink", line="gold_d", lw=0.5, cls="contour")

    def _aperture_lip(self, P):
        P.poly([(-6.4, 30.0), (-3.5, 31.8), (3.5, 31.8), (6.4, 30.0), (6.6, 31.2), (3.5, 33.0), (-3.5, 33.0),
                (-6.6, 31.2)], fill="gold", line="gold_d", lw=0.45, cls="contour", mat="gold", smooth=True)


# ================================================================ EGREGORE
_SLOP = ("slop_flesh", "slop_cyan", "slop_beige", "slop_mauve", "glitch_m", "glitch_c", "screen_glow", "concrete_l")


class Egregore:
    kind = "egregore"

    def __init__(self, seed=0, **style):
        self.seed = int(seed)
        self.style = dict(style)

    def box(self, h):
        return (-0.6 * h, -1.02 * h, 0.6 * h, 0.02 * h)

    def _static(self, t, n, m, k=0, tint=None, amt=0.35):
        rng = np.random.default_rng((int(t * 24) * 7919 + self.seed * 131 + k) & 0x7FFFFFFF)
        v = rng.random((n, m)).astype(np.float32) ** 1.3
        img = np.repeat(v[..., None], 3, 2)
        if tint is not None:
            img = img * (1 - amt) + np.array(rgb(tint), np.float32) * amt
        return np.clip(img, 0, 1)

    _BUST = [(-50.0, 0.0), (-48.0, -18.0), (-43.0, -29.0), (-30.0, -37.0), (-10.0, -43.0), (10.0, -43.0), (30.0, -37.0),
             (43.0, -29.0), (48.0, -18.0), (50.0, 0.0)]
    _FACE = [(0.0, -45.0), (8.0, -48.0), (12.0, -56.0), (12.5, -64.0), (10.5, -72.0), (6.0, -77.0), (0.0, -78.5),
             (-6.0, -77.0), (-10.5, -72.0), (-12.5, -64.0), (-12.0, -56.0), (-8.0, -48.0)]

    def anchors(self, x, y, h, t=0.0, **kw):
        u = h / 100.0
        W = lambda p: (x + p[0] * u, y + p[1] * u)
        a = {"scale": u, "h": h, "kind": "egregore"}
        f = W((0.0, -62.0))
        a["face"] = (f[0], f[1], 14.0 * u)
        hc = W((0.0, -64.0))
        a["halo"] = (hc[0], hc[1], 35.0 * u)
        a["phone"] = W((-27.0, -44.0))
        t0, t1 = W((12.0, -34.0)), W((38.0, -4.0))
        a["tablet"] = (t0[0], t0[1], t1[0], t1[1])
        a["screens"] = [(*W((c[0], c[1])), *W((c[2], c[3]))) for c in self._cells()]
        a["bbox"] = (x - 0.6 * h, y - 1.02 * h, x + 0.6 * h, y + 0.02 * h)
        return a

    def _cells(self):
        out = []
        for r in range(6):
            y0 = -42.0 + r * 7.0
            for c in range(-7, 7):
                x0 = c * 7.2 + (3.6 if r % 2 else 0.0)
                out.append((x0, y0, x0 + 6.6, y0 + 6.4))
        return out

    def draw(self, ctx, x, y, h, t=0.0, layer="color", tile=None, glitch=0.0, alpha=1.0, tint=None, **kw):
        u = h / 100.0
        P = Painter(ctx, layer, tile, alpha, tint)
        P.push(x, y, 0.0, u)
        P.lod = 0
        k = int(t * 24)
        # the anti-halo: a ring of billboards (every screen its own feed) inside a dark ring
        P.circle(0.0, -64.0, 35.0, fill="ink", line="concrete_d", lw=0.8, cls="contour")
        n = 18
        for i in range(n):
            a = i * TAU / n + 0.05 * math.sin(t * 0.4)
            c, s = math.cos(a), math.sin(a)
            P.push(c * 29.5, -64.0 + s * 29.5, a + math.pi / 2, 1.0)
            col_ = _SLOP[(i * 5 + k // 8) % len(_SLOP)]
            P.poly([(-4.6, -3.0), (4.6, -3.0), (4.6, 3.0), (-4.6, 3.0)], fill=col_, line="ink", lw=0.4, cls="fold",
                   mat="glow", smooth=False)
            if (i + k // 4) % 3 == 0:
                P.paint_image(self._static(t, 5, 8, i, col_), -4.2, -2.6, 8.4, 5.2, mat="glow")
            P.pop()
        P.circle(0.0, -64.0, 23.5, fill="night2", line="ink", lw=0.8, cls="contour")
        # the bust: a video wall arranged like the Pantocrator's himation
        P.clip(self._BUST, smooth=True)
        P.poly([(-60, 5), (60, 5), (60, -50), (-60, -50)], fill="ink", smooth=False)
        for i, (x0, y0, x1, y1) in enumerate(self._cells()):
            col_ = _SLOP[(i * 7 + (k // 10) * (1 + i % 3)) % len(_SLOP)]
            if (i + k // 6) % 4 == 0:
                P.paint_image(self._static(t, 4, 6, 50 + i, col_, 0.5), x0, y0, x1 - x0, y1 - y0, mat="glow")
            else:
                P.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill=col_, mat="glow", smooth=False)
        # drapery lines across the screens (the himation's folds, drawn in black cable)
        for pts in ([(-30.0, -37.0), (-10.0, -20.0), (6.0, 0.0)], [(30.0, -37.0), (18.0, -22.0), (14.0, 0.0)],
                    [(-43.0, -29.0), (-34.0, -12.0), (-30.0, 0.0)], [(-10.0, -43.0), (0.0, -30.0), (-4.0, 0.0)]):
            P.line(pts, "ink", 1.6, "contour")
        P.unclip()
        P.poly(self._BUST, line="ink", lw=1.0, cls="contour", smooth=True)
        # cable hair falling to the shoulders, the neck, the faceless head of static
        P.poly([(-7.0, -50.0), (7.0, -50.0), (8.0, -40.0), (-8.0, -40.0)], fill="concrete_d", line="ink", lw=0.6,
               cls="contour", smooth=False)
        for sg in (-1, 1):
            for j in range(5):
                x0 = sg * (8.0 + 1.7 * j)
                pts = [(x0 * 0.55, -79.0), (x0, -70.0), (x0 * 1.1 + sg * 1.2 * math.sin(t * 0.8 + j), -54.0),
                       (x0 * 1.25, -40.0)]
                P.line(pts, "ink", 1.7, "contour")
                P.line(pts, "slate", 0.6, "fold")
        P.poly(self._FACE, fill="concrete", mat="fine glow", smooth=True)
        P.clip(self._FACE)
        P.paint_image(self._static(t, 24, 18, 7), -13.0, -79.0, 26.0, 34.0, mat="fine glow")
        P.unclip()
        P.poly(self._FACE, line="ink", lw=0.9, cls="contour", smooth=True)
        # right hand raised (anti-blessing): holding up a phone
        # forearm (a sleeve of dark screens) up to the hand
        P.poly([(-44.0, -14.0), (-36.0, -30.0), (-27.0, -38.0), (-21.0, -33.0), (-30.0, -20.0), (-36.0, -8.0)],
               fill="concrete_d", line="ink", lw=0.8, cls="contour", smooth=True)
        P.push(-25.0, -50.0, -0.1, 1.0)
        P.poly([(-6.0, -11.0), (6.0, -11.0), (6.0, 11.0), (-6.0, 11.0)], fill="ink", line="concrete_l", lw=0.7,
               cls="contour", smooth=False)
        P.paint_image(self._static(t, 14, 8, 9, "screen_glow", 0.55), -5.0, -10.0, 10.0, 20.0, mat="glow")
        # the grey hand gripping the phone (thumb over the edge, fingers behind)
        P.poly([(-8.0, 6.0), (-5.5, 1.0), (-2.0, 4.0), (-3.0, 10.0), (-6.5, 13.0), (-9.0, 11.0)], fill="concrete_l",
               line="ink", lw=0.6, cls="fine2", mat="fine", smooth=True)
        P.pop()
        # left arm: the tablet (anti-Gospel)
        P.poly([(12.0, -34.0), (38.0, -34.0), (38.0, -4.0), (12.0, -4.0)], fill="ink", line="concrete_l", lw=0.9,
               cls="contour", smooth=False)
        P.paint_image(self._static(t, 20, 16, 11, "glitch_c", 0.4), 13.6, -32.4, 22.8, 26.8, mat="glow")
        P.pop()
        if glitch > 0 and layer == "color":
            _glitch_slices(ctx, x, y, h, t, glitch, self.seed)
        return self.anchors(x, y, h, t=t)


def _glitch_slices(ctx, x, y, h, t, amount, seed):
    """shift horizontal bands of what was just drawn (datamosh), deterministic in t"""
    surf = ctx.get_target()
    try:
        surf.flush()
    except Exception:
        return
    rng = np.random.default_rng(int(t * 24) * 13 + seed)
    m = ctx.get_matrix()
    for _ in range(int(3 + 6 * amount)):
        yy = y - h * rng.random()
        hh = h * (0.01 + 0.04 * rng.random())
        dx = (rng.random() - 0.5) * 0.2 * h * amount
        ctx.save()
        ctx.rectangle(x - h, yy, 2 * h, hh)
        ctx.clip()
        ctx.set_source_surface(surf, dx * m.xx, 0.0)
        ctx.restore()

"""s05b NATION (100.5-107.8): a planted Flag cracks the map of a Nation; borders subdivide recursively while the
camera falls through a seamless, scale-free zoom until every cell is an island-country of one crowned person.
"""
import math

import cv2
import numpy as np

from vx import *
from vx.flag import Flag
from vx.chars import Character
from scenes import s05b_mapk as mk
from scenes.s05b_common import T0, lg, crown, mini_person, COOL_CLOTH, pose_ok

NG = 7                  # generations of subdivision (level 0 = first split of the nation)
S1 = 0.5                # world size of generation-0 cells (nation radius = 1)
Q = 2.35                # linear scale ratio between generations
R0 = 1.0
Z0 = 470.0              # design px per world unit at the start
BIRTH = 210.0           # a generation is born when its cells reach this size on screen (design px)
T_PLANT = lg(100.6)
T_GEN1 = lg(100.72)
ZA, ZB = 0.30, 2.95     # zoom fall (local s)
T_END = lg(107.8)
FLAG_POLE = 0.52        # world units

PAL_T = ["#6F9CC6", "#5FAA9E", "#8E8CD0", "#7AAF8E", "#9AAFD6", "#7D95B5", "#A08EC0", "#5E8CA3", "#86B9BD",
         "#7F83B8", "#6E86A8", "#93A6C9"]
BASE_TINT = "#8AA597"
WATER_D, WATER_S, FOAM, CRACK = "#0E1D33", "#2B5770", "#CFE6F0", "#EAF6FF"


def _a(c):
    return np.array(col(c), np.float64)


def smoother(a, b, x):
    return smootherstep(a, b, x)


class Nation:
    def __init__(self, S):
        self.S = S
        rng = np.random.default_rng(20420)
        self.LS = np.array([S1 * Q ** -g for g in range(NG)])
        th = rng.uniform(0, math.pi, NG)
        self.LC, self.LSN = np.cos(th), np.sin(th)
        self.LSEED = (np.arange(NG) * 101 + 13).astype(np.int64)
        self.LGAP = self.LS * 0.075
        self.LGAP[-1] = self.LS[-1] * 0.06
        self.LCR = np.full(NG, 0.26)
        self.PAL = np.array([col(c) for c in PAL_T], np.float64)
        self.BASE = _a(BASE_TINT)
        self.LT = np.full(NG, 1e3)          # temporary: everything open (for layout)
        self._pick_target()
        self._build_zoom()
        self._build_people()
        self._build_flag()

    # ------------------------------------------------------------------ layout
    def _args(self):
        return (NG, self.LS, self.LC, self.LSN, self.LSEED, self.LT, self.LGAP, self.LCR, self.PAL)

    def _score(self, x, y):
        sc = mk.coast_sd(x, y, R0) * 0.6
        for g in range(NG - 1):
            _, _, e, _, _ = mk.level_cell(x, y, self.LS[g], self.LC[g], self.LSN[g], self.LSEED[g])
            sc = min(sc, e / self.LS[g])
        return sc

    def _pick_target(self):
        """a point deep inside a cell at every generation (so the fall never ends in a sea), away from the flag;
        then a pair of adjacent final cells (the proud citizen A and the neighbour B) sharing one tiny border."""
        rng = np.random.default_rng(7)
        best = (-1, None)
        for _ in range(900):
            r, a = rng.uniform(0.3, 0.52), rng.uniform(0, 2 * math.pi)
            x, y = r * math.cos(a), r * math.sin(a)
            s = self._score(x, y)
            if s > best[0]:
                best = (s, (x, y))
        px, py = best[1]
        g = NG - 1
        s6 = self.LS[g]
        n = 240
        ext = 3.2 * s6
        xs = np.linspace(px - ext, px + ext, n)
        ys = np.linspace(py - ext, py + ext, n)
        lab, pcs = self._pieces(xs, ys)
        # the proud citizen gets a small ("doormat") island; the neighbour a normal one, sharing one tiny border
        cand = []
        for la, A in pcs.items():
            if math.hypot(A["c"][0] - px, A["c"][1] - py) > 1.6 * s6 or not (0.28 <= A["rel"] <= 0.62):
                continue
            par = mk.level_cell(A["c"][0], A["c"][1], self.LS[g - 1], self.LC[g - 1], self.LSN[g - 1],
                                self.LSEED[g - 1])[:2]
            dil = cv2.dilate((lab == la).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            for lb in np.unique(lab[dil]):
                if lb in (0, la) or lb not in pcs:
                    continue
                B = pcs[lb]
                pp = mk.level_cell(B["c"][0], B["c"][1], self.LS[g - 1], self.LC[g - 1], self.LSN[g - 1],
                                   self.LSEED[g - 1])[:2]
                if pp != par or B["rel"] < 0.7:
                    continue
                dx, dy = B["c"][0] - A["c"][0], B["c"][1] - A["c"][1]
                rot = -math.atan2(dy, dx)
                d = math.hypot(dx, dy) / s6
                sc = -abs(rot) * 0.4 - abs(d - 0.75) * 2 - abs(A["rel"] - 0.45) * 2 - abs(min(B["rel"], 1.2) - 1)
                cand.append((sc, la, lb, rot))
        cand.sort(reverse=True)
        _, la, lb, rot = cand[0]
        A, B = pcs[la], pcs[lb]
        self.A, self.B = A["c"], B["c"]
        self.rot_f = rot
        self.pair_d = math.hypot(B["c"][0] - A["c"][0], B["c"][1] - A["c"][1])
        self.P = ((A["c"][0] + B["c"][0]) / 2, (A["c"][1] + B["c"][1]) / 2)

        def contour(m):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=len)[:, 0, :].astype(np.float64)[::3]
            return np.c_[np.interp(c[:, 0], np.arange(n), xs), np.interp(c[:, 1], np.arange(n), ys)]
        self.polyA, self.polyB = contour(lab == la), contour(lab == lb)

    def _pieces(self, xs, ys, min_frac=0.0):
        """land pieces at the final time on a world grid: every connected bit of one final cell.
        -> (label image, {label: dict(c=pole of inaccessibility (world), area, r=inscribed radius (world))})"""
        from scipy import ndimage
        g = NG - 1
        X, Y = np.meshgrid(xs, ys)
        sd, _, ids = mk.land_points(X.ravel(), Y.ravel(), 50.0, *self._args(), R0, g, 1e-9, self.BASE)
        ny, nx = X.shape
        land = (sd > 0).reshape(ny, nx)
        key = (ids[:, 0] * 100003 + ids[:, 1]).reshape(ny, nx)
        edge = np.zeros_like(land)
        edge[:, 1:] |= key[:, 1:] != key[:, :-1]
        edge[1:, :] |= key[1:, :] != key[:-1, :]
        m = land & ~edge
        lab, nl = ndimage.label(m)
        if nl == 0:
            return lab, {}
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        idx = np.arange(1, nl + 1)
        area = ndimage.sum(m, lab, idx)
        pos = ndimage.maximum_position(dt, lab, idx)
        big = np.median(area[area > np.percentile(area, 60)]) if nl > 3 else area.max()
        dx = xs[1] - xs[0]
        out = {}
        for i, (a, p) in enumerate(zip(area, pos)):
            if a < max(12, min_frac * big):
                continue
            out[i + 1] = dict(c=(xs[p[1]], ys[p[0]]), area=float(a), rel=float(a / big),
                              r=float(dt[p]) * dx)
        return lab, out

    # ------------------------------------------------------------------ camera
    def _build_zoom(self):
        self.zf = 580.0 / self.pair_d                  # A and B end ~580 design px apart
        dL = math.log(self.zf / Z0)
        ts = np.linspace(0, T_END + 1, 20001)
        v = np.array([smoothstep(ZA, ZA + 0.5, t) * smoothstep(ZB, ZB - 0.9, t) for t in ts])
        Lc = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) * 0.5 * np.diff(ts))])
        Lc *= dL / Lc[-1]
        self._ts, self._L = ts, Lc + math.log(Z0)
        # generations are born at a constant on-screen size -> the balkanization is scale-free
        LT = np.zeros(NG)
        for g in range(NG):
            if g == 0:
                LT[g] = T_GEN1
                continue
            need = math.log(BIRTH / self.LS[g])
            i = np.searchsorted(self._L, need)
            LT[g] = max(ts[min(i, len(ts) - 1)], LT[g - 1] + 0.22)
        self.LT = LT

    def zpx(self, t):
        z = math.exp(float(np.interp(t, self._ts, self._L)))
        return z * (1 + 0.012 * max(0.0, t - ZB) + 0.004 * max(0.0, t - ZB) ** 2)

    def camera(self, t):
        z = self.zpx(t) * (1 + 0.2 * smoother(lg(103.15), lg(104.9), t))
        u = smoother(1.85, 2.95, t)
        kt = 1.0 - 0.42 * u
        depth = 7.0 + 21.0 * u
        rot = self.rot_f * smoother(ZA, ZB + 0.2, t) + 0.012 * max(0.0, t - ZB)
        # screen offset of the target point P (midpoint of the two speakers' feet)
        s_init = np.array(self.P) * Z0
        s_fin = np.array([20.0, 340.0])
        s_fin = s_fin + np.array([168.0, 0.0]) * smoother(lg(103.0), lg(103.7), t) \
            - np.array([188.0, 0.0]) * smoother(lg(105.3), lg(105.85), t)
        m = smoother(ZA, 2.7, t)
        S = s_init * (1 - m) + s_fin * m
        # shake: plant thump + a small jolt per generation
        sh = np.zeros(2)
        k = math.exp(-max(0.0, t - T_PLANT) * 7.0) * (t >= T_PLANT)
        sh += np.array(shake(t, 14 * k, 18, 3))
        for g in range(NG):
            dt = t - self.LT[g] - 0.05
            if 0 <= dt < 0.4:
                sh += np.array(shake(t, 4.0 * math.exp(-dt * 12), 24, 10 + g))
        S = S + sh
        c, s = math.cos(rot), math.sin(rot)
        qx, qy = S[0] / z, S[1] / (z * kt)
        camx = self.P[0] - (c * qx + s * qy)
        camy = self.P[1] - (-s * qx + c * qy)
        return camx, camy, z, kt, rot, depth

    def proj(self, cam, x, y):
        camx, camy, z, kt, rot, _ = cam
        dx, dy = np.asarray(x) - camx, np.asarray(y) - camy
        c, s = math.cos(rot), math.sin(rot)
        return 960 + (c * dx - s * dy) * z, 540 + (s * dx + c * dy) * z * kt

    def view_aabb(self, cam, margin=80):
        xs = np.array([-margin, W + margin, -margin, W + margin], float)
        ys = np.array([-margin, -margin, H + margin, H + margin], float)
        camx, camy, z, kt, rot, _ = cam
        c, s = math.cos(rot), math.sin(rot)
        qx, qy = (xs - 960) / z, (ys - 540) / (z * kt)
        wx, wy = camx + c * qx + s * qy, camy - s * qx + c * qy
        return wx.min(), wy.min(), wx.max(), wy.max()

    # ------------------------------------------------------------------ cast
    def _build_people(self):
        """one citizen per island piece at the final generation, over the whole region the camera sees once the
        people are tall enough to draw (precomputed once)."""
        self.hero_h = 0.9 * self.pair_d     # world height of a citizen
        self.cA = Character("citizen", seed=11, crown=True, robe="cloth_blue", skin="skin2", hair="#3b3440")
        self.cB = Character("citizen", seed=23, crown=True, robe="cloth_plum", skin="skin1", hair="#6b5a52")
        self.extras = {}
        ts = np.linspace(ZA, ZB, 400)
        t14 = next(t for t in ts if self.hero_h * self.zpx(t) >= 12.0)
        self.t_people = t14
        x0, y0, x1, y1 = self.view_aabb(self.camera(t14), 120)
        res = self.LS[NG - 1] / 8.0
        xs = np.arange(x0, x1, res)
        ys = np.arange(y0, y1, res)
        _, pcs = self._pieces(xs, ys, min_frac=0.3)
        ppl = []
        for k, p in pcs.items():
            x, y = p["c"]
            inside = any(cv2.pointPolygonTest(pg.astype(np.float32), (float(x), float(y)), False) >= 0
                         for pg in (self.polyA, self.polyB))
            if inside or min(math.hypot(x - self.A[0], y - self.A[1]),
                             math.hypot(x - self.B[0], y - self.B[1])) < self.LS[NG - 1] * 0.45:
                continue
            ppl.append((x, y, min(1.0, p["rel"]), int(hash01(int(x * 1e6) ^ int(y * 1e6) * 7, 1) * 1e6)))
        self.people = np.array(ppl, np.float64)

    def _build_flag(self):
        n = int(round((T_END + 0.6) * 24)) + 2
        self.flag_n = n
        f_plant = T_PLANT * 24

        def pole(i):
            k = clamp((i - 0.0) / max(f_plant, 1e-3))
            return (0.0, -90.0 * (1 - k * k), 0.0)

        self.flag = Flag(pole=300, seed=31).simulate(n, pole, lambda i: (520 + 90 * math.sin(i / 11.0), 0))

    # ------------------------------------------------------------------ render
    def render(self, fc, t=None):
        t = fc.T - T0 if t is None else t
        cam = self.camera(t)
        camx, camy, z, kt, rot, depth = cam
        h, w = fc.h, fc.w
        out = np.empty((h, w, 3), np.float32)
        cover = np.empty((h, w), np.float32)
        buf = np.empty((h, w, 10), np.float32)
        rview = math.hypot(960, 540 / kt) / z + depth / (z * kt)
        g0, stsd, sttint, stpar, stpid = mk.resolve(camx, camy, rview, t, *self._args(), self.BASE, R0)
        mk.render_map(out, cover, buf, fc.s, camx, camy, z, kt, rot, t, depth, g0, *self._args(), stsd, sttint,
                      stpar, stpid, R0, _a(WATER_D), _a(WATER_S), _a(FOAM), _a(CRACK), 45.0,
                      1.0 - smoothstep(1.7, 2.4, t))
        # ---- things printed on the land (masked by the island tops): roads, towns
        cvm = Canvas(fc)
        self._draw_towns(cvm.ctx, cam, t)
        lay = cvm.rgba()
        lay *= cover[..., None]
        out = over(out, lay)
        # ---- things standing on it
        cvo = Canvas(fc)
        ctx = cvo.ctx
        self._draw_crowns(ctx, cam, t)
        self._draw_people(ctx, cam, t, fc)
        self._draw_flag(ctx, cam, t, fc, shadow_only=True)
        out = over(out, cvo.rgba())
        # aerial perspective once the camera tilts: the far archipelago melts into a pale haze
        if kt < 0.995:
            ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
            a = (1 - kt) / 0.42 * 0.42 * (1 - ys) ** 2.2
            out = out * (1 - a) + np.array(col("#A9B8CE"), np.float32) * a
        # ---- opening: the breach light condenses into the Flag (drawn over the veil)
        if t < 0.95:
            out = self._glow(out, cam, t, fc)
        cvf = Canvas(fc)
        self._draw_shock(cvf.ctx, cam, t)
        self._draw_flag(cvf.ctx, cam, t, fc)
        return over(out, cvf.rgba())

    # towns & roads: capitals of every generation, fading in as they grow on screen (map level-of-detail)
    def _level_cells(self, cam, g, t):
        x0, y0, x1, y1 = self.view_aabb(cam, 60)
        return mk.seeds_in_rect(x0, y0, x1, y1, self.LS[g], self.LC[g], self.LSN[g], self.LSEED[g])

    def _draw_towns(self, ctx, cam, t):
        z = cam[2]
        kt = cam[3]
        for g in range(NG):
            size = self.LS[g] * z
            a = smoothstep(60, 140, size) * smoothstep(2600, 1500, size) * (g <= 4)
            if a <= 0.01:
                continue
            cells = self._level_cells(cam, g, t)
            if len(cells) == 0 or len(cells) > 900:
                continue
            X, Y = self.proj(cam, cells[:, 2], cells[:, 3])
            idx = {(int(ci), int(cj)): k for k, (ci, cj) in enumerate(cells[:, :2])}
            # highways between the big cities only (the first two generations)
            ra = a * (1.0 if g == 0 else 0.6 if g == 1 else 0.0)
            ctx.set_line_width(clamp(size * 0.006, 0.8, 2.6))
            set_color(ctx, "#E3E7EE", 0.42 * ra)
            for k, (ci, cj) in enumerate(cells[:, :2] if ra > 0 else []):
                for di, dj in ((1, 0), (0, 1), (1, 1)):
                    j = idx.get((int(ci) + di, int(cj) + dj))
                    if j is None:
                        continue
                    xa, ya, xb, yb = X[k], Y[k], X[j], Y[j]
                    mx, my = (xa + xb) / 2, (ya + yb) / 2
                    bend = (hash01(int(ci) * 31 + int(cj) * 17 + di * 5 + g, 3) - 0.5) * 0.35
                    nx, ny = -(yb - ya) * bend, (xb - xa) * bend
                    ctx.move_to(xa, ya)
                    ctx.curve_to(mx + nx, my + ny, mx + nx, my + ny, xb, yb)
            ctx.stroke()
            # towns: clusters of pale blocks
            r = clamp(size * (0.05 if g < 2 else 0.035), 2.0, 22.0)
            for k, (ci, cj) in enumerate(cells[:, :2]):
                sd0 = int(ci) * 7919 + int(cj) * 104729 + g * 31
                for b in range(5):
                    ox = (hash01(sd0, b * 3 + 1) - 0.5) * r * 1.8
                    oy = (hash01(sd0, b * 3 + 2) - 0.5) * r * 1.8 * kt
                    bw = r * (0.35 + 0.4 * hash01(sd0, b * 3 + 3))
                    set_color(ctx, "#2A3547", 0.35 * a)
                    rect(ctx, X[k] + ox - bw / 2 + 1, Y[k] + oy - bw * kt / 2 + 1, bw, bw * kt)
                    ctx.fill()
                    set_color(ctx, "#EEF1F6", 0.9 * a)
                    rect(ctx, X[k] + ox - bw / 2, Y[k] + oy - bw * kt / 2, bw, bw * kt)
                    ctx.fill()

    def _draw_crowns(self, ctx, cam, t):
        z = cam[2]
        for g in range(NG - 1):
            size = self.LS[g] * z
            a = smoothstep(120, 190, size) * smoothstep(1400, 900, size)
            if a <= 0.01 or t < self.LT[g]:
                continue
            cells = self._level_cells(cam, g, t)
            if len(cells) == 0 or len(cells) > 600:
                continue
            X, Y = self.proj(cam, cells[:, 2], cells[:, 3])
            sd, _, _ = mk.land_points(cells[:, 2].copy(), cells[:, 3].copy(), 50.0, *self._args(), R0, g, 1e-9,
                                      self.BASE)
            cw = clamp(size * 0.085, 8.0, 30.0)
            for k in range(len(cells)):
                if sd[k] < 0:
                    continue
                d = hash01(int(cells[k, 0]) * 131 + int(cells[k, 1]) * 71 + g, 5)
                tp = t - (self.LT[g] + 0.12 + 0.18 * d)
                if tp <= 0:
                    continue
                pop = ease_out_back(clamp(tp / 0.28), 2.4)
                set_color(ctx, "#101a2a", 0.25 * a)
                ellipse(ctx, X[k], Y[k], cw * 0.5 * pop, cw * 0.16 * pop)
                ctx.fill()
                crown(ctx, X[k], Y[k] - cw * 0.35 - (1 - pop) * 10, cw * pop, a * min(1, tp * 6))

    def _draw_shock(self, ctx, cam, t):
        tp = t - T_PLANT
        if tp < 0 or tp > 0.9:
            return
        bx, by = self.proj(cam, 0.0, 0.0)
        z, kt = cam[2], cam[3]
        for k, (sp, wd, al) in enumerate(((1.5, 5.0, 0.8), (1.0, 2.5, 0.5))):
            r = ease_out(clamp(tp / 0.9), 2.2) * sp * z
            a = al * (1 - clamp(tp / 0.9)) ** 1.6
            ellipse(ctx, bx, by, r, r * kt)
            set_color(ctx, "#F4F8FF", a)
            ctx.set_line_width(wd * (1 - 0.5 * tp))
            ctx.stroke()
        # impact dust
        rng = np.random.default_rng(5)
        for i in range(26):
            a0 = rng.uniform(0, 2 * math.pi)
            sp = rng.uniform(0.05, 0.16) * z
            r = sp * ease_out(clamp(tp / 0.6), 2)
            al = 0.5 * (1 - clamp(tp / 0.7))
            set_color(ctx, "#DCE4EE", al)
            circle(ctx, bx + math.cos(a0) * r, by + math.sin(a0) * r * kt, rng.uniform(4, 10) * (1 + tp))
            ctx.fill()

    def _draw_flag(self, ctx, cam, t, fc, shadow_only=False):
        bx, by = self.proj(cam, 0.0, 0.0)
        z = cam[2]
        fall = 1.0 + 1.4 * (1 - clamp(t / T_PLANT)) ** 2 if t < T_PLANT else 1.0 + 0.05 * math.exp(
            -(t - T_PLANT) * 9) * math.sin((t - T_PLANT) * 40)
        k = FLAG_POLE * z / 300.0 * fall
        if 300 * k > 6000 or bx < -900 - 300 * k or bx > W + 900 + 300 * k or by < -200 or by > H + 300 * k * 1.3:
            return
        alpha = smoothstep(0.015, 0.07, t) * (1 - smoothstep(620, 1000, 300 * k))
        if alpha <= 0:
            return
        if shadow_only:
            if t >= T_PLANT - 0.02:
                ctx.save()
                ctx.translate(bx, by)
                ctx.scale(k, k * cam[3])
                set_color(ctx, "#08101c", 0.3 * alpha)
                poly(ctx, [(-4, 0), (150, 70), (250, 60), (240, 110), (140, 118), (6, 6)])
                ctx.fill()
                ctx.restore()
            return
        ctx.save()
        ctx.translate(bx, by)
        ctx.scale(k, k)
        self.flag.draw(ctx, min(fc.f, self.flag_n - 1), alpha=alpha, light=(-0.4, -0.7, 0.55),
                       glow=0.35 + 0.6 * (1 - smoothstep(0.1, 0.9, t)))
        ctx.restore()

    def _glow(self, out, cam, t, fc):
        """frame 0 = flat flag_glow (hand-off from s05a); the veil lifts while a halo stays on the Flag"""
        bx, by = self.proj(cam, 0.0, 0.0)
        fx, fy = bx + 30 * FLAG_POLE / 0.3, by - 150 * FLAG_POLE / 0.3
        veil = 1 - ease_out(clamp((t - 0.02) / 0.42), 2.2)
        hal = (1 - smoothstep(0.12, 0.7, t)) * 0.85
        a = radial_mask(fc, fx, fy, 0, 120 + 520 * clamp(1 - 1.4 * t))[..., 0] * hal
        a = np.maximum(a, veil)[..., None]
        g = np.array(col("flag_glow"), np.float32)
        return out * (1 - a) + g * a

    def _draw_people(self, ctx, cam, t, fc):
        z, kt = cam[2], cam[3]
        hh = self.hero_h * z
        if hh < 12:
            return
        P = self.people
        X, Y = self.proj(cam, P[:, 0], P[:, 1])
        vis = (X > -150) & (X < W + 150) & (Y > -60) & (Y < H + hh * 1.3)
        ax, ay = self.proj(cam, *self.A)
        bx_, by_ = self.proj(cam, *self.B)
        # nobody stands in front of the two speakers
        for hx, hy in ((ax, ay), (bx_, by_)):
            vis &= ~((Y > hy - hh * 0.05) & (np.abs(X - hx) < hh * 0.75))
            vis &= ~((Y > hy - hh * 0.8) & (np.abs(X - hx) < hh * 0.5))
        items = [(Y[k], "x", X[k], k) for k in np.nonzero(vis)[0]]
        items.append((ay, "A", ax, None))
        items.append((by_, "B", bx_, None))
        items.sort(key=lambda it: it[0])
        pop_all = smoothstep(12, 22, hh)
        for y, kind, x, k in items:
            # contact shadow on the island
            set_color(ctx, "#0B1426", 0.28 * pop_all)
            ellipse(ctx, x, y, hh * 0.17, hh * 0.05)
            ctx.fill()
            if kind == "A":
                self._fence(ctx, cam, t, back=True)
                self._hero_A(ctx, x, y, hh, t, fc, pop_all)
                self._fence(ctx, cam, t, back=False)
                continue
            if kind == "B":
                self._hero_B(ctx, x, y, hh, t, fc, pop_all)
                continue
            sdn = int(P[k, 3])
            hk = hh * (0.8 + 0.22 * math.sqrt(P[k, 2])) * (0.92 + 0.14 * hash01(sdn, 1))
            cloth = COOL_CLOTH[int(hash01(sdn, 2) * len(COOL_CLOTH)) % len(COOL_CLOTH)]
            skin = ("skin1", "skin2", "skin3", "skin4")[int(hash01(sdn, 3) * 4) % 4]
            if hk < 70:
                arm = max(0.0, math.sin(t * 2.3 + sdn % 11)) * (hash01(sdn, 4) > 0.5)
                mini_person(ctx, x, y, hk, cloth, skin, t, sdn % 97, True, pop_all, arm)
                continue
            ch = self.extras.get(sdn)
            if ch is None:
                ch = Character("citizen", seed=sdn % 1000, crown=True, robe=cloth, skin=skin)
                self.extras[sdn] = ch
            poses = ("declare", "stand", "cheer", "one_finger", "point_self", "talk")
            pose = poses[int(hash01(sdn, 5) * len(poses)) % len(poses)]
            exprs = ("proud", "smug", "happy", "angry", "determined")
            ex = exprs[int(hash01(sdn, 6) * len(exprs)) % len(exprs)]
            babble = max(0.0, math.sin(t * 9 + sdn % 13)) * 0.6 * (math.sin(t * 1.3 + sdn % 7) > 0)
            ch.draw(ctx, x, y, hk, pose_ok(pose), t + sdn % 5, mouth=babble, expr=ex,
                    facing=1 if hash01(sdn, 7) > 0.5 else -1, alpha=pop_all)

    def _hero_A(self, ctx, x, y, hh, t, fc, alpha):
        T = fc.T
        m = fc.tl.mouth("CITIZEN1", T)
        speaking = 103.15 <= T <= 105.6
        if hh < 70:
            mini_person(ctx, x, y, hh, "cloth_blue", "skin2", t, 3, True, alpha, 0.0)
            return
        k = smoother(103.0, 103.35, T) * (1 - smoother(105.6, 106.0, T))
        pose = {"declare": k, "stand": 1 - k} if k < 0.999 else "declare"
        ex = "proud" if T < 105.7 else ("surprised" if T < 106.3 else "skeptical")
        look = (0.6, -0.4) if speaking else (1.0, 0.0)
        self.cA.draw(ctx, x, y, hh, pose_ok(pose), t, mouth=m, expr=ex, facing=1, look=look, alpha=alpha,
                     wind=0.3, turn=0.35)

    def _hero_B(self, ctx, x, y, hh, t, fc, alpha):
        T = fc.T
        m = fc.tl.mouth("CITIZEN2", T)
        if hh < 70:
            mini_person(ctx, x, y, hh, "cloth_plum", "skin1", t, 5, True, alpha, 0.0)
            return
        w_one = fc.tl.word("X02", "one")
        k = smoother(w_one[0] - 0.35, w_one[0] - 0.05, T) * (1 - smoother(107.3, 107.7, T))
        k2 = smoother(105.5, 105.8, T) * (1 - k)
        pose = {"one_finger": k + 1e-3, "talk": k2 + 1e-3, "stand": max(0.0, 1 - k - k2) + 1e-3}
        ex = "skeptical" if T < 105.6 else "smug"
        self.cB.draw(ctx, x, y, hh, pose_ok(pose), t, mouth=m, expr=ex, facing=-1, look=(-1.0, 0.0), alpha=alpha,
                     wind=0.3, turn=0.35)

    def _fence(self, ctx, cam, t, back):
        z, kt = cam[2], cam[3]
        hh = self.hero_h * z
        if hh < 60:
            return
        a = smoothstep(60, 110, hh)
        P = self.polyA
        c = P.mean(0)
        Pi = c + (P - c) * 0.86
        X, Y = self.proj(cam, Pi[:, 0], Pi[:, 1])
        cx, cy = self.proj(cam, c[0], c[1])
        # resample every ~26 px
        seg = np.hypot(np.diff(X, append=X[:1]), np.diff(Y, append=Y[:1]))
        L = np.concatenate([[0], np.cumsum(seg)])
        n = max(8, int(L[-1] / 26))
        s = np.linspace(0, L[-1], n, endpoint=False)
        XX = np.interp(s, L, np.append(X, X[0]))
        YY = np.interp(s, L, np.append(Y, Y[0]))
        ph = hh * 0.13
        sel = (YY < cy) if back else (YY >= cy)
        wood, wood_d = "#9AA2AE", "#5D6573"
        for i in range(n):
            j = (i + 1) % n
            if sel[i] and sel[j]:
                for fr in (0.35, 0.75):
                    ctx.move_to(XX[i], YY[i] - ph * fr)
                    ctx.line_to(XX[j], YY[j] - ph * fr)
                set_color(ctx, wood_d, a)
                ctx.set_line_width(max(1.2, hh * 0.012))
                ctx.stroke()
        for i in range(n):
            if not sel[i]:
                continue
            set_color(ctx, wood_d, a)
            rect(ctx, XX[i] - hh * 0.009 - 0.8, YY[i] - ph - 0.8, hh * 0.018 + 1.6, ph + 1.6)
            ctx.fill()
            set_color(ctx, wood, a)
            rect(ctx, XX[i] - hh * 0.009, YY[i] - ph, hh * 0.018, ph)
            ctx.fill()
            poly(ctx, [(XX[i] - hh * 0.009, YY[i] - ph), (XX[i], YY[i] - ph - hh * 0.012),
                       (XX[i] + hh * 0.009, YY[i] - ph)])
            ctx.fill()

    # ------------------------------------------------------------------ sound
    def events(self):
        ev = [dict(t=100.6, kind="thunk", strength=1.0, pan=self._pan0(), desc="Flag planted at the centre of the Nation map; shock ring"),
              dict(t=100.62, kind="whoosh", strength=0.6, pan=0.0, desc="shock ring sweeps across the map")]
        for g in range(NG):
            ev.append(dict(t=T0 + self.LT[g] + 0.05, kind="crack", strength=0.95 - 0.05 * g, pan=0.0,
                           desc=f"generation {g + 1} of borders cracks across every cell (crack fronts race out from each capital)"))
            ev.append(dict(t=T0 + self.LT[g] + 0.35, kind="rumble", strength=0.5, pan=0.0,
                           desc=f"generation {g + 1} channels open: land slabs part, sea floods in"))
            if g < NG - 1:
                ev.append(dict(t=T0 + self.LT[g] + 0.2, kind="pop", strength=0.3, pan=0.0,
                               desc="tiny silver crowns pop up on every new capital"))
        ev.append(dict(t=T0 + self.LT[NG - 1] + 0.1, kind="pop", strength=0.6, pan=0.0,
                       desc="every island now holds exactly one crowned citizen"))
        return ev

    def _pan0(self):
        cam = self.camera(T_PLANT)
        x, _ = self.proj(cam, 0.0, 0.0)
        return float(np.clip((x - W / 2) / (W / 2), -1, 1))

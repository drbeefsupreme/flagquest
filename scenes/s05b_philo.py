"""s05b PHILOSOPHY (113.9-125.5): a colossal marble philosopher head in an agora; a Flag is planted on its crown;
Voronoi cracks race from the pole, the head bursts into hundreds of fragments that drift into a starry void; a tiny
toga'd crackpot screams on every fragment. X06 / X07 heroes. K07 pull-back: the Commander in silhouette with his Flag
before a galaxy of every shard of every memeplex -> storm swirl on dusk1 (hand-off to s06).
"""
import math

import cairo
import cv2
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point

from vx import *
from vx.flag import Flag
from vx.chars import Character
from scenes import s05b_head as HD
from scenes.s05b_common import (T0, lg, pose_ok, crown, shout_rings, bubble, scribble, pseudo_eq, shard_cells,
                                draw_shards)

T_PLANT = 114.30
T_BURST = 114.60
T_POP = 115.68
T_SCREAM = 116.27
T_VOID = 116.95
T_GAL = 123.3
T_END = 125.5
HC = np.array([960.0, 560.0])          # head centre on the stage (design px)
HS = 0.45                               # design px per head unit
GC = np.array([960.0, 620.0])          # galaxy centre (agreed with Scene06)


def _wt(T):
    return T


class Philo:
    def __init__(self, S):
        self.S = S
        rgba, mask = HD.build()
        rgba = rgba * np.array([0.9, 0.9, 0.92, 1.0], np.float32)     # sit the marble below bloom clipping
        self._dcache = {}
        self.tex = rgba
        self.tex_surf = surface_from_array(rgba)
        self._fracture(mask)
        self._physics()
        self._pick_heroes()
        self._build_flag()
        self._build_void()
        self._build_sky()
        self._build_agora()
        self._build_galaxy()
        self._build_commander()
        self.pots = {}

    # ================================================================== geometry
    def _fracture(self, mask):
        out = HD.outline(mask, 5)
        U = np.c_[out[:, 0] / HD.TEX_K - HD.UW / 2, out[:, 1] / HD.TEX_K - HD.UH / 2]
        head = Polygon(U).buffer(0)
        rng = np.random.default_rng(1618)
        pts = []
        minx, miny, maxx, maxy = head.bounds
        cr = np.array(HD.CROWN)
        while len(pts) < 230:
            p = np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)])
            d = np.linalg.norm(p - cr)
            # denser near the Flag, sparser (bigger fragments) across the face
            keep = 0.35 + 0.65 * math.exp(-d / 500.0)
            if abs(p[0]) < 260 and -120 < p[1] < 320:
                keep *= 0.35
            if rng.uniform() < keep and head.contains(Point(p)):
                pts.append(p)
        pts = np.array(pts)
        # two big chunks for the hero crackpots: the nose and the left brow/cheek
        for hx, hy, rr in ((20.0, 90.0, 190.0), (-215.0, -60.0, 175.0)):
            keep = np.hypot(pts[:, 0] - hx, pts[:, 1] - hy) > rr
            pts = np.r_[pts[keep], [[hx, hy]]]
        far = np.array([[-5000, -5000], [5000, -5000], [-5000, 5000], [5000, 5000]])
        vor = Voronoi(np.r_[pts, far])
        frags = []
        for i in range(len(pts)):
            reg = vor.regions[vor.point_region[i]]
            if -1 in reg or not reg:
                continue
            pg = Polygon(vor.vertices[reg]).intersection(head)
            if pg.is_empty:
                continue
            if pg.geom_type != "Polygon":
                pg = max(pg.geoms, key=lambda g: g.area)
            if pg.area < 300:
                continue
            pg = pg.simplify(1.5)
            xy = np.array(pg.exterior.coords)[:-1]
            c = np.array(pg.centroid.coords[0])
            frags.append(dict(xy=xy, c=c, area=pg.area, loc=xy - c))
        self.frags = frags
        # crack edges (for the crack-race before the burst)
        E = []
        for f in frags:
            xy = f["xy"]
            for a, b in zip(xy, np.roll(xy, -1, 0)):
                E.append((a, b))
        E = np.array(E)
        d = np.linalg.norm((E[:, 0] + E[:, 1]) / 2 - cr, axis=1)
        self.edges = E
        self.edge_t = T_PLANT + 0.02 + d / d.max() * 0.2

    def _physics(self):
        rng = np.random.default_rng(314)
        cr = np.array(HD.CROWN)
        O = np.array([0.0, -250.0])
        for k, f in enumerate(self.frags):
            c = f["c"]
            d = c - O
            dn = d / (np.linalg.norm(d) + 1e-6)
            f["v"] = dn * rng.uniform(260, 620) + rng.normal(0, 120, 2)
            f["drift"] = dn * rng.uniform(8, 26) + rng.normal(0, 8, 2)
            f["vz"] = rng.uniform(-0.5, 0.9)
            f["zd"] = rng.uniform(-0.02, 0.05)
            f["w"] = rng.uniform(-2.2, 2.2)
            f["wd"] = rng.uniform(-0.06, 0.06)
            f["k"] = rng.uniform(1.3, 1.9)
            f["seed"] = k
            f["pot"] = f["area"] > 3200 and rng.uniform() < 0.85
            if f["pot"]:
                f["w"] = rng.uniform(-0.5, 0.5)
            # local 'top' point (the crackpot stands here): highest vertex region of the fragment
            loc = f["loc"]
            j = np.argmin(loc[:, 1])
            f["top"] = loc[j] * 0.8 + np.array([0.0, 0.0])

    def frag_state(self, f, T):
        """-> (world design x, y), depth z (1 = head plane, <1 nearer), angle"""
        tau = T - T_BURST
        c = f["c"]
        if tau <= 0:
            shake_ = 0.0
            if T > T_PLANT:
                shake_ = (T - T_PLANT) / (T_BURST - T_PLANT) * 2.5 * math.sin(T * 90 + f["seed"])
            return HC + HS * c + shake_, 1.0, 0.0
        k = f["k"]
        e = (1 - math.exp(-k * tau)) / k
        p = c + f["v"] * e + f["drift"] * tau
        z = max(0.25, 1.0 - f["vz"] * e * 0.9 - f["zd"] * tau)
        a = f["w"] * e + f["wd"] * tau
        return HC + HS * p, z, a

    def _pick_heroes(self):
        # hero 1 stands on the nose region, hero 2 on a cheek/eye region: big fragments that drift toward camera
        fr = self.frags
        cand1 = sorted(range(len(fr)), key=lambda i: np.linalg.norm(fr[i]["c"] - np.array([20, 90])))
        cand2 = sorted(range(len(fr)), key=lambda i: np.linalg.norm(fr[i]["c"] - np.array([-215, -60])))
        self.hA = cand1[0]
        self.hB = cand2[0] if cand2[0] != self.hA else cand2[1]
        for i, (vx_, vy_), vz in ((self.hA, (-420, 140), 0.62), (self.hB, (760, -170), 0.55)):
            f = fr[i]
            f["v"] = np.array([vx_, vy_], float)
            f["vz"] = vz
            f["zd"] = 0.012
            f["w"] = 0.0
            f["wd"] = 0.0
            f["pot"] = True

    # ================================================================== flag on the crown
    def _build_flag(self):
        cr = np.array(HD.CROWN)
        ci = min(range(len(self.frags)), key=lambda i: np.linalg.norm(self.frags[i]["c"] - cr))
        self.crown_i = ci
        f = self.frags[ci]
        self.crown_off = cr - f["c"] + np.array([0.0, 20.0])
        self.flag_T0 = 113.7
        n = int((T_END - self.flag_T0) * 24) + 4
        self.flag_n = n
        self._flag_pole = []

        def pole(i):
            T = self.flag_T0 + i / 24
            p, z, a = self.frag_state(f, T)
            off = self.crown_off
            ca, sa = math.cos(a), math.sin(a)
            base = p + HS * np.array([ca * off[0] - sa * off[1], sa * off[0] + ca * off[1]])
            drop = 0.0
            if T < T_PLANT:
                u = clamp((T - 113.95) / (T_PLANT - 113.95))
                drop = -900 * (1 - u * u)
            return (float(base[0]), float(base[1] + drop), float(a))

        self.flag_pole = 260.0 * HS          # stage px (a 96" pole on a ~32 m head)
        self.flag = Flag(pole=self.flag_pole, seed=91).simulate(n, pole, lambda i: (240, -40))
        self._pole_fn = pole

    # ================================================================== backgrounds
    def _build_void(self):
        rng = np.random.default_rng(2718)
        n = 1400
        self.stars = np.c_[rng.uniform(-400, W + 400, n), rng.uniform(-300, H + 300, n), rng.uniform(0.3, 1.0, n),
                           rng.uniform(0, 6.28, n), rng.uniform(0.02, 0.2, n)]
        # nebula (quarter-res), cool dusk tones
        h, w = 270, 480
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = fbm2(xx / 70.0, yy / 70.0, 5, 5) * 0.5 + 0.5
        b = fbm2(xx / 40.0 + 9, yy / 40.0, 6, 5) * 0.5 + 0.5
        neb = np.zeros((h, w, 3), np.float32)
        neb += np.array(col("dusk2"), np.float32) * np.clip(a - 0.45, 0, 1)[..., None] * 1.3
        neb += np.array(col("#2E3F74"), np.float32) * np.clip(b - 0.5, 0, 1)[..., None] * 1.4
        neb += np.array(col("dusk3"), np.float32) * np.clip(a * b - 0.42, 0, 1)[..., None] * 0.9
        self.nebula = cv2.GaussianBlur(neb, (0, 0), 3)

    def _build_sky(self):
        h, w = 270, 480
        ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        stops = [(0, "#1E2A55"), (0.45, "#56639A"), (0.64, "#AEB4D2"), (0.7, "#C9CCE0"), (1, "#8990AE")]
        sky = np.zeros((h, w, 3), np.float32)
        ts = [a for a, _ in stops]
        for c in range(3):
            sky[..., c] = np.interp(ys, ts, [col(b)[c] for _, b in stops])
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        n = fbm2(xx / 60.0, yy / 16.0, 21, 5) * 0.5 + 0.5
        band = np.clip(1 - np.abs(yy / h - 0.38) / 0.22, 0, 1)
        cl = np.clip((n - 0.52) * 3.0, 0, 1) * band
        lit = np.clip((fbm2(xx / 60.0, (yy + 3) / 16.0, 21, 5) * 0.5 + 0.5) - n + 0.5, 0, 1)
        cc = np.array(col("#D9DAF0"), np.float32) * (0.7 + 0.3 * lit[..., None])
        sky = sky * (1 - cl[..., None] * 0.75) + cc * cl[..., None] * 0.75
        self.sky = cv2.GaussianBlur(sky, (0, 0), 0.8)

    def _build_agora(self):
        """static agora layer (plaza, stoa, tiny citizens) pre-rendered at 1:1 design scale"""
        class _FC:
            s = 1.0
            w, h = W, H
        cv = Canvas(_FC(), s=1.0)
        ctx = cv.ctx
        hy = 720.0
        # plaza (paved) with perspective joints
        ctx.set_source(lin_grad(0, hy, 0, H, [(0, "#7F86A3"), (1, "#4B5170")]))
        rect(ctx, 0, hy, W, H - hy)
        ctx.fill()
        set_color(ctx, "#3C4260", 0.35)
        ctx.set_line_width(1.2)
        for k in range(-30, 31):
            ctx.move_to(960 + k * 30, hy)
            ctx.line_to(960 + k * 240, H + 200)
        ctx.stroke()
        for k in range(12):
            y = hy + (H - hy) * (k / 12) ** 1.8
            ctx.move_to(0, y)
            ctx.line_to(W, y)
        ctx.stroke()
        # stoa: long colonnade on the horizon both sides of the head
        for side in (-1, 1):
            x0 = 960 + side * 430
            x1 = 960 + side * 1150
            xa, xb = min(x0, x1), max(x0, x1)
            set_color(ctx, "#8A90AC")
            rect(ctx, xa, hy - 150, xb - xa, 22)
            ctx.fill()
            poly(ctx, [(xa - 10, hy - 150), (xb + 10, hy - 150), ((xa + xb) / 2, hy - 196)])
            set_color(ctx, "#9BA1BC")
            ctx.fill()
            for k in range(14):
                cx = xa + 20 + k * (xb - xa - 40) / 13
                ctx.set_source(lin_grad(cx - 9, 0, cx + 9, 0, [(0, "#6E7492"), (0.4, "#B3B8CF"), (1, "#6E7492")]))
                rect(ctx, cx - 9, hy - 128, 18, 128)
                ctx.fill()
            set_color(ctx, "#666C8A")
            rect(ctx, xa - 12, hy - 4, xb - xa + 24, 8)
            ctx.fill()
        # steps in front
        for k in range(4):
            set_color(ctx, mix("#9AA0BA", "#6A7090", k / 4))
            rect(ctx, 0, hy + 10 + k * 12, W, 12)
            ctx.fill()
        # tiny citizens (for scale) milling about the plaza
        rng = np.random.default_rng(99)
        from vx.chars import draw_crowd
        ppl = []
        for i in range(46):
            x = rng.uniform(40, W - 40)
            y = rng.uniform(800, 1070)
            if abs(x - 960) < 360 and y < 1000:
                continue
            hh = 26 + (y - 800) * 0.09
            ppl.append(dict(x=x, y=y, h=hh, kind="philosopher" if rng.uniform() < 0.4 else "citizen",
                            seed=int(rng.integers(1000)), pose="look_up" if rng.uniform() < 0.6 else "stand",
                            facing=1 if x < 960 else -1, look=(0.3 if x < 960 else -0.3, -1)))
        ppl.sort(key=lambda p: p["y"])
        draw_crowd(ctx, ppl, 0.0)
        self.agora = cv.rgba()
        self.agora_cells = shard_cells(404, 18)

    def _build_galaxy(self):
        """extra shards of every memeplex that join the swirl: map shards, temple stones, slop cards, phones, crowns"""
        rng = np.random.default_rng(1234)
        items = []
        kinds = ["map"] * 70 + ["stone"] * 45 + ["slop"] * 45 + ["phone"] * 22 + ["crown"] * 16
        for k, kind in enumerate(kinds):
            arm = k % 3
            r = 80 + 820 * rng.uniform() ** 0.8
            th0 = arm * 2 * math.pi / 3 + math.log(r / 80) / math.tan(0.42) * -1 + rng.normal(0, 0.28)
            size = rng.uniform(12, 40) * (1.25 if kind == "map" else 1.0)
            n = int(rng.integers(4, 7))
            ang = np.sort(rng.uniform(0, 2 * math.pi, n))
            shp = np.c_[np.cos(ang), np.sin(ang)] * rng.uniform(0.6, 1.0, (n, 1))
            colr = {"map": ["#6F9CC6", "#5FAA9E", "#8E8CD0", "#7AAF8E", "#9AAFD6"],
                    "stone": ["#5A4A66", "#6E6480", "cloth_grey", "#4E4A5E"],
                    "slop": ["slop_flesh", "slop_cyan", "slop_beige", "slop_mauve"],
                    "phone": ["#1C2030"], "crown": [SILVER_COL]}[kind]
            items.append(dict(kind=kind, r=r, th0=th0, size=size, shp=shp, rot0=rng.uniform(0, 6.28),
                              spin=rng.uniform(-1.5, 1.5), col=colr[int(rng.integers(len(colr)))],
                              r_in=rng.uniform(1300, 2300), th_in=rng.uniform(0, 6.28), delay=rng.uniform(0, 0.5)))
        self.gal = items
        # head fragments get galaxy slots too
        for f in self.frags:
            arm = f["seed"] % 3
            r = 90 + 780 * rng.uniform() ** 0.9
            f["g_r"] = r
            f["g_th"] = arm * 2 * math.pi / 3 - math.log(r / 80) / math.tan(0.42) + rng.normal(0, 0.3)
            f["g_size"] = rng.uniform(14, 48)
        # storm wisps: log-spiral noise (quarter res), rotated per frame
        h, w = 360, 360
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
        r = np.hypot(dx, dy) + 1e-3
        th = np.arctan2(dy, dx)
        u = th + np.log(r) / math.tan(0.42) * 1.0
        n1 = fbm2(np.cos(u * 3) * 2 + r * 3, np.sin(u * 3) * 2 + r * 3, 7, 4)
        wis = np.clip(n1 * 0.9 + 0.2, 0, 1) * np.clip(1.3 - r, 0, 1) * np.clip(r * 3, 0, 1)
        self.wisps = cv2.GaussianBlur(wis.astype(np.float32), (0, 0), 1.5)

    def _build_commander(self):
        self.cmd = Character("commander", seed=7, headset=True)
        n = int((T_END - (T_GAL - 0.4)) * 24) + 4
        self.cmd_T0 = T_GAL - 0.4
        self.cmd_n = n
        self.cmd_h = 540.0

        def pole(i):
            T = self.cmd_T0 + i / 24
            x, y, s = self.cmd_xy(T)
            a = self.cmd.anchors(0.0, 0.0, self.cmd_h, "hold_flag", T - T0, facing=1, back=True,
                                 pole_len=self.cmd_h * 1.65)
            bx, by, ang = a["pole"]
            return (bx, by, ang)

        self.cmd_flag = Flag(pole=self.cmd_h * 1.65, seed=12).simulate(n, pole, lambda i: (650, -60))

    def cmd_xy(self, T):
        """commander foreground placement (feet point) and scale: rises into frame as the camera pulls back,
        then the camera sails past him toward the swirl"""
        u = ease_out(clamp((T - (T_GAL - 0.15)) / 0.75), 3)
        v = ease_in(clamp((T - 124.45) / 0.62), 2.2)
        x = 400 - 1350 * v
        y = 1110 + (1 - u) * 800 + 650 * v
        s = 1.0 + 0.6 * v
        return x, y, s

    # ================================================================== camera
    def camera(self, T):
        """-> (cx, cy, zoom) stage camera; the heroes' frames follow their fragments"""
        wide = (960.0, 540.0, 1.0)
        if T < T_BURST:
            z = 1.0 + 0.04 * smoothstep(113.9, T_BURST, T)
            return (960.0, 560.0 - 10 * smoothstep(113.9, T_BURST, T), z)
        pb = smoothstep(T_BURST, 116.6, T)
        base = (960.0, 560.0, 1.04 - 0.22 * ease_out(pb, 2))
        # X06 hero A
        kA = smootherstep(117.35, 117.95, T) * (1 - smootherstep(120.8, 121.05, T))
        kB = smootherstep(120.8, 121.05, T) * (1 - smootherstep(122.55, 123.4, T))
        cam = np.array(base)
        for k, hi, sx, zm in ((kA, self.hA, 760.0, 3.5), (kB, self.hB, 1170.0, 3.5)):
            if k <= 0:
                continue
            f = self.frags[hi]
            p, z, a = self.frag_state(f, T)
            tgt_zoom = zm
            # screen x of the hero = sx  ->  camera centre
            top = p + HS * f["top"]
            eff = tgt_zoom / z
            cx = top[0] - (sx - 960) / eff
            cy = top[1] - 170 / eff
            tgt = np.array([cx, cy, tgt_zoom])
            cam = cam + (tgt - cam) * k
        # K07 pull-back
        g = smootherstep(122.6, 124.9, T)
        cam = cam + (np.array([960.0, 600.0, 0.18]) - cam) * g
        return tuple(cam)

    def to_screen(self, cam, p, z=1.0):
        cx, cy, zm = cam
        eff = zm / z
        return 960 + (p[0] - cx) * eff, 540 + (p[1] - cy) * eff, eff

    # ================================================================== render
    def render(self, fc, T=None):
        T = fc.T if T is None else T
        cam = self.camera(T)
        img = self._background(fc, T, cam)
        cv = Canvas(fc)
        ctx = cv.ctx
        glw = Canvas(fc)
        gctx = glw.ctx
        g = smootherstep(T_GAL - 0.5, 124.7, T)
        if g > 0:
            self._galaxy_items(ctx, T, g)
        if T < T_BURST:
            self._head_whole(ctx, gctx, T, cam)
        else:
            self._fragments(ctx, gctx, T, cam, g)
        if T >= T_GAL - 0.45:
            self._commander(ctx, gctx, T, fc)
        img = over(img, cv.rgba())
        gl = glw.rgba()[..., :3]
        img = img + blur(gl, 8 * fc.s) * 1.0 + gl * 0.5
        # burst flash
        if T_BURST - 0.02 <= T < T_BURST + 0.6:
            k = math.exp(-(T - T_BURST) * 9.0)
            img = img + np.array(col("#E9E6FF"), np.float32) * 0.22 * min(k, 1.0)
        # whip blur between the heroes
        wv = pulse(T, 120.93, 0.07)
        if wv > 0.05:
            L = max(3, int(90 * wv * fc.s))
            ker = np.zeros((1, L), np.float32)
            ker[0, :] = 1.0 / L
            img = cv2.filter2D(img, -1, ker)
        return img

    def _background(self, fc, T, cam):
        # void: nebula + stars (little parallax), cross-fading from the agora sky at the burst,
        # and into the storm (dusk1 radial + spiral wisps) for the galaxy
        h, w = fc.h, fc.w
        sky = cv2.resize(self.sky, (w, h), interpolation=cv2.INTER_CUBIC)
        vo = np.empty((h, w, 3), np.float32)
        vo[:] = col("night")
        neb = cv2.resize(self.nebula, (w, h), interpolation=cv2.INTER_LINEAR)
        vo += neb * 0.9
        cv = Canvas(fc)
        ctx = cv.ctx
        cx, cy, zm = cam
        par = 0.08
        sx = self.stars
        tw = 0.65 + 0.35 * np.sin(T * 3 * sx[:, 4] * 10 + sx[:, 3])
        X = 960 + (sx[:, 0] - 960) * (1 + (zm - 1) * par) - (cx - 960) * par
        Y = 540 + (sx[:, 1] - 540) * (1 + (zm - 1) * par) - (cy - 540) * par
        for x, y, b, t_ in zip(X, Y, sx[:, 2], tw):
            if -4 < x < W + 4 and -4 < y < H + 4:
                set_color(ctx, "#E8ECFF", b * t_)
                circle(ctx, x, y, 0.6 + b * 1.2)
                ctx.fill()
        st = cv.rgba()
        vo = over(vo, st)
        vo += blur(st[..., :3], 3 * fc.s) * 0.8
        k = smoothstep(T_BURST, T_BURST + 0.55, T)
        img = sky * (1 - k) + vo * k
        # agora ground: whole, then shattered slabs falling away into the void
        if T < T_BURST + 1.2:
            ag = self.agora
            if fc.s != 1.0:
                ag = cv2.resize(ag, (w, h), interpolation=cv2.INTER_AREA)
            if T < T_BURST:
                img = over(img, ag)
            else:
                cva = Canvas(fc)
                self._fall_shards(cva.ctx, ag, T - T_BURST, fc.s)
                img = over(img, cva.rgba())
        # storm for the hand-off
        g = smootherstep(T_GAL, 125.0, T)
        if g > 0:
            storm = np.empty((h, w, 3), np.float32)
            key = (h, w)
            if key not in self._dcache:
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                self._dcache[key] = np.hypot(xx / fc.s - GC[0], yy / fc.s - GC[1]) / 1100.0
            d = self._dcache[key]
            t_ = np.clip(d, 0, 1)[..., None]
            storm[:] = np.array(col("dusk1"), np.float32) * (1 - t_) + np.array(col("night"), np.float32) * t_
            ang = -(T - T_GAL) * 0.22 * 180 / math.pi
            M = cv2.getRotationMatrix2D((180, 180), ang, 1.0)
            wz = cv2.warpAffine(self.wisps, M, (360, 360), borderMode=cv2.BORDER_CONSTANT)
            sz = int(2 * 1150 * fc.s)
            wz = cv2.resize(wz, (sz, sz), interpolation=cv2.INTER_LINEAR)
            x0, y0 = int(GC[0] * fc.s - sz / 2), int(GC[1] * fc.s - sz / 2)
            lay = np.zeros((h, w), np.float32)
            xa, ya = max(0, x0), max(0, y0)
            xb, yb = min(w, x0 + sz), min(h, y0 + sz)
            lay[ya:yb, xa:xb] = wz[ya - y0:yb - y0, xa - x0:xb - x0]
            storm += lay[..., None] * np.array(col("dusk2"), np.float32) * 0.8
            core = np.exp(-(d * 1100 / 170) ** 2)[..., None]
            storm += core * np.array(col("#C9C2FF"), np.float32) * 0.55
            img = img * (1 - g) + storm * g
        return img

    def _fall_shards(self, ctx, ag, tau, s):
        surf = surface_from_array(ag)
        rng = np.random.default_rng(12)
        for k, (xy, c) in enumerate(self.agora_cells):
            if c[1] < 600:
                continue
            vx = (c[0] - 960) * rng.uniform(0.2, 0.6)
            vy = rng.uniform(-80, 60)
            spin = rng.uniform(-0.9, 0.9)
            dx = vx * tau
            dy = vy * tau + 0.5 * 1500 * tau * tau
            a = spin * tau
            alpha = 1 - smoothstep(0.3, 1.1, tau)
            ctx.save()
            ctx.translate(c[0] + dx, c[1] + dy)
            ctx.rotate(a)
            ctx.translate(-c[0], -c[1])
            poly(ctx, [tuple(v) for v in xy])
            ctx.clip()
            ctx.scale(1 / s, 1 / s)
            ctx.set_source_surface(surf, 0, 0)
            ctx.paint_with_alpha(alpha)
            ctx.restore()

    # ------------------------------------------------------------------ the head
    def _head_whole(self, ctx, gctx, T, cam):
        cx, cy, zm = cam
        sx, sy, eff = self.to_screen(cam, HC)
        k = HS * eff
        shake_ = 0.0
        if T > T_PLANT:
            shake_ = (T - T_PLANT) / (T_BURST - T_PLANT) * 3.0 * math.sin(T * 95)
        ctx.save()
        ctx.translate(sx + shake_, sy)
        ctx.scale(k, k)
        # contact shadow on the plaza
        set_color(ctx, "#1d2138", 0.45)
        ellipse(ctx, 0, 945, 520, 60)
        ctx.fill()
        ctx.save()
        ctx.scale(1 / HD.TEX_K, 1 / HD.TEX_K)
        ctx.set_source_surface(self.tex_surf, -HD.TEX_W / 2, -HD.TEX_H / 2)
        ctx.paint()
        ctx.restore()
        # crack race from the Flag
        if T >= T_PLANT:
            for (a, b), te in zip(self.edges, self.edge_t):
                if T < te:
                    continue
                u = clamp((T - te) / 0.05)
                m = a + (b - a) * u
                ctx.move_to(*a)
                ctx.line_to(*m)
            set_color(ctx, "#171428", 0.95)
            ctx.set_line_width(4.0)
            ctx.stroke_preserve()
            gctx.save()
            gctx.set_matrix(ctx.get_matrix())
            gctx.append_path(ctx.copy_path())
            set_color(gctx, "#B9B2FF", 0.7)
            gctx.set_line_width(7.0)
            gctx.stroke()
            gctx.restore()
            ctx.new_path()
        ctx.restore()
        self._crown_flag(ctx, gctx, T, cam)

    def _crown_flag(self, ctx, gctx, T, cam, alpha=1.0):
        i = int(round((T - self.flag_T0) * 24))
        if i < 0 or T < 113.97 or alpha <= 0:
            return
        i = min(i, self.flag_n - 1)
        f = self.frags[self.crown_i]
        _, z, _ = self.frag_state(f, T)
        cx, cy, zm = cam
        eff = zm / z
        if self.flag_pole * eff < 3:
            return
        ctx.save()
        ctx.translate(960, 540)
        ctx.scale(eff, eff)
        ctx.translate(-cx, -cy)
        self.flag.draw(ctx, i, alpha=alpha, light=(-0.4, -0.6, 0.6), glow=0.4)
        ctx.restore()

    def _fragments(self, ctx, gctx, T, cam, g):
        tau = T - T_BURST
        items = []
        for fi, f in enumerate(self.frags):
            p, z, a = self.frag_state(f, T)
            items.append((z, fi, p, a))
        items.sort(key=lambda it: -it[0])
        surf = self.tex_surf
        pop_on = T >= T_POP - 0.1
        # keep the hero crackpots unobstructed while they speak (nearer chunks are not drawn in front of them)
        clear = None
        for hi, a_, b_ in ((self.hA, 117.3, 121.0), (self.hB, 120.8, 122.9)):
            if a_ <= T <= b_:
                p_, z_, _ = self.frag_state(self.frags[hi], T)
                xh, yh, _ = self.to_screen(cam, p_, z_)
                clear = (xh, yh - 150, z_, 520)
        for z, fi, p, a in items:
            f = self.frags[fi]
            x, y, eff = self.to_screen(cam, p, z)
            k = HS * eff
            # galaxy blend: move toward a spiral slot, shrink
            if g > 0:
                th = f["g_th"] - self._omega(f["g_r"]) * (T - T_GAL)
                gx, gy = GC[0] + f["g_r"] * math.cos(th), GC[1] + f["g_r"] * math.sin(th)
                gk = f["g_size"] / (math.sqrt(f["area"]) + 1e-6)
                gg = ease_in_out(g, 2)
                x, y = x + (gx - x) * gg, y + (gy - y) * gg
                k = k + (gk - k) * gg
                a = a + (T - T_GAL) * 0.8 * (1 if fi % 2 else -1) * gg
            R = math.sqrt(f["area"]) * k
            if x < -R * 2 or x > W + R * 2 or y < -R * 2 or y > H + R * 2 or R < 0.6:
                continue
            if clear is not None and fi not in (self.hA, self.hB) and z < clear[2] + 0.05:
                if math.hypot(x - clear[0], y - clear[1]) < clear[3] + R * 0.7:
                    continue
            loc = f["loc"]
            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(a)
            ctx.scale(k, k)
            # chunk thickness (a darker extruded side toward the lower right)
            th_ = 14.0 * min(1.0, tau * 4)
            poly(ctx, [tuple(v + np.array([th_ * 0.4, th_])) for v in loc])
            set_color(ctx, "#3E4460")
            ctx.fill()
            poly(ctx, [tuple(v) for v in loc])
            ctx.save()
            ctx.clip_preserve()
            ctx.translate(-f["c"][0], -f["c"][1])
            ctx.scale(1 / HD.TEX_K, 1 / HD.TEX_K)
            ctx.set_source_surface(surf, -HD.TEX_W / 2, -HD.TEX_H / 2)
            ctx.paint()
            ctx.restore()
            # settle the marble into the void's light (keeps it from blooming out)
            poly(ctx, [tuple(v) for v in loc])
            ctx.set_source(lin_grad(0, -40, 0, 60, [(0, "#1B1F3B", 0.02), (1, "#1B1F3B", 0.3)]))
            ctx.fill_preserve()
            # fresh fracture faces glow a little
            set_color(ctx, "#D9D4FF", 0.5 * math.exp(-tau * 1.2))
            ctx.set_line_width(2.5 / max(k, 1e-3) * 1.0)
            ctx.stroke()
            ctx.restore()
            if fi == self.crown_i:
                self._crown_flag(ctx, gctx, T, cam, alpha=1 - smoothstep(124.2, 124.9, T))
            if pop_on and f["pot"]:
                self._crackpot(ctx, gctx, T, f, fi, x, y, a, k, g)

    @staticmethod
    def _omega(r):
        return 0.5 * math.sqrt(300.0 / max(r, 40.0))

    # ------------------------------------------------------------------ crackpots
    def _crackpot(self, ctx, gctx, T, f, fi, x, y, a, k, g):
        hero = fi in (self.hA, self.hB)
        d = hash01(fi, 3) * 0.55
        tp = T - (T_POP + (0 if hero else d))
        if tp <= 0:
            return
        pop = ease_out_back(clamp(tp / 0.25), 2.0)
        fade = 1 - smoothstep(123.8, 124.7, T)
        if fade <= 0:
            return
        top = f["top"]
        ca, sa = math.cos(a), math.sin(a)
        fx = x + (ca * top[0] - sa * top[1]) * k
        fy = y + (sa * top[0] + ca * top[1]) * k
        hscale = (0.78 if hero else 0.42) * math.sqrt(f["area"]) * k
        h = hscale * pop
        if h < 3:
            return
        ch = self.pots.get(fi)
        if ch is None:
            robes = ["bone", "#DCD8E6", "#C8CFDC", "#E6E2DA"]
            ch = Character("crackpot", seed=fi, robe=robes[fi % 4])
            self.pots[fi] = ch
        facing = 1 if hash01(fi, 5) > 0.5 else -1
        mouth = 0.0
        pose = "stand"
        expr = "furious"
        if T >= T_SCREAM + (0 if hero else hash01(fi, 6) * 0.4):
            pose = ("cup_shout", "shout", "cup_shout", "point", "gesture")[fi % 5]
            mouth = 0.55 + 0.4 * abs(math.sin(T * (6 + fi % 5) + fi))
            expr = ("furious", "angry", "exasperated", "surprised")[fi % 4]
        if fi == self.hA:
            facing, pose, expr, mouth = self._heroA(T)
        elif fi == self.hB:
            facing, pose, expr, mouth = self._heroB(T)
        ctx.save()
        ctx.translate(fx, fy)
        ctx.rotate(a * 0.35)
        an = ch.draw(ctx, 0.0, 0.0, h, pose_ok(pose), T - T0 + fi * 0.21, mouth=mouth, expr=expr, facing=facing,
                     turn=0.35 if not hero else 0.3, alpha=fade, wind=0.4)
        ctx.restore()
        # shout rings / bubbles
        if T >= T_SCREAM and fade > 0.05:
            mx, my = an.get("mouth", (0, -h * 0.85))
            ca2, sa2 = math.cos(a * 0.35), math.sin(a * 0.35)
            mx, my = fx + ca2 * mx - sa2 * my, fy + sa2 * mx + ca2 * my
            if hero:
                self._hero_fx(ctx, gctx, T, fi, mx, my, h, facing)
            elif h > 10:
                shout_rings(gctx, mx, my, T + fi * 0.13, rate=2.2, r0=h * 0.1, r1=h * 0.9, n=3, alpha=0.55 * fade,
                            ang=0.0 if facing > 0 else math.pi, spread=0.8, lw=max(0.8, h * 0.025))
                if fi % 4 == 0 and h > 16:
                    ph = int((T - T_SCREAM) / 1.4 + fi) % 3
                    bw, bh = h * 1.3, h * 0.75
                    bxx = mx + facing * h * 0.9
                    byy = my - h * 0.85
                    pb = ease_out_back(clamp((T - T_SCREAM - hash01(fi, 8) * 0.6) / 0.2), 2)
                    if pb > 0:
                        bubble(ctx, bxx, byy, bw * pb, bh * pb, tail=(mx + facing * h * 0.15, my - h * 0.1),
                               alpha=0.92 * fade, lw=max(0.8, h * 0.03))
                        if pb > 0.8:
                            if ph == 0:
                                scribble(ctx, bxx - bw * 0.36, byy - bh * 0.3, bw * 0.72, bh * 0.6, fi * 7, lw=max(0.6, h * 0.022),
                                         alpha=fade)
                            else:
                                pseudo_eq(ctx, bxx, byy + bh * 0.18, bh * 0.5, fi * 13 + ph, alpha=fade, n=4)

    def _heroA(self, T):
        tl = self.S.tl
        m = tl.mouth("CRACKPOT1", T)
        if T < 117.8:
            return 1, "stand", "furious", 0.0
        if T < 118.85:
            return 1, "cup_shout", "furious", m
        if T < 119.75:
            return 1, "gesture", "exasperated", m
        if T < 121.2:
            return 1, "point", "furious", m
        return 1, "cup_shout", "angry", 0.5 + 0.3 * abs(math.sin(T * 7))

    def _heroB(self, T):
        tl = self.S.tl
        m = tl.mouth("CRACKPOT2", T)
        if T < 120.9:
            return -1, "cup_shout", "angry", 0.5 + 0.3 * abs(math.sin(T * 6.5))
        wt = tl.word("X07", "therefore")
        if T < wt[0] - 0.05:
            return -1, "point_self", "proud", m
        if T < wt[1]:
            return -1, "declare", "proud", m
        if T < 122.7:
            return -1, "point_self", "smug", m
        return -1, "point_self", "smug", 0.0

    def _hero_fx(self, ctx, gctx, T, fi, mx, my, h, facing):
        if fi == self.hA and 117.85 <= T <= 121.0:
            # big bubble of pseudo-maths and scribbles, re-scrawled on each phrase
            words = self.S.tl.words("X06")
            ph = sum(1 for w_ in words if w_["s"] <= T)
            pb = ease_out_back(clamp((T - 117.95) / 0.25), 2)
            bw, bh = min(h * 1.55, 560), min(h * 0.95, 340)
            bxx = min(mx + h * 0.35 + bw * 0.55, W - bw * 0.55 - 30)
            byy = max(my - h * 0.55, bh * 0.55 + 30)
            bubble(ctx, bxx, byy, bw * pb, bh * pb, tail=(mx + h * 0.2, my - h * 0.05), wobble=1.0, t=T, lw=3)
            if pb > 0.85:
                pseudo_eq(ctx, bxx, byy - bh * 0.08, bh * 0.3, 1000 + ph // 2, n=5)
                scribble(ctx, bxx - bw * 0.35, byy + bh * 0.08, bw * 0.7, bh * 0.25, 2000 + ph, lw=2.2, lines=2)
            shout_rings(gctx, mx, my, T, rate=2.6, r0=h * 0.1, r1=h * 1.4, n=3, alpha=0.7, ang=0.0,
                        spread=0.7, lw=3.5)
        if fi == self.hB and 120.95 <= T <= 123.2:
            # echo rings: every spoken word spawns a ring that repeats itself, fading, into the void
            for w_ in self.S.tl.words("X07"):
                for e in range(4):
                    t0 = w_["s"] + e * 0.22
                    u = (T - t0) / 1.1
                    if 0 <= u <= 1:
                        r = h * (0.25 + 2.6 * ease_out(u, 2))
                        circle(gctx, mx, my + h * 0.25, r)
                        set_color(gctx, "#DAD6FF", 0.75 * (1 - u) * (0.6 ** e))
                        gctx.set_line_width(max(1.0, h * 0.02 * (1 - u) + 1))
                        gctx.stroke()

    # ------------------------------------------------------------------ galaxy + commander
    def _galaxy_items(self, ctx, T, g):
        for it in self.gal:
            u = ease_in_out(clamp((T - T_GAL + 0.3 - it["delay"]) / 1.6), 2)
            if u <= 0:
                continue
            th = it["th0"] - self._omega(it["r"]) * (T - T_GAL)
            r = it["r"] + (it["r_in"] - it["r"]) * (1 - u)
            th = th + (1 - u) * 1.8
            x, y = GC[0] + r * math.cos(th), GC[1] + r * math.sin(th)
            s = it["size"]
            if x < -80 or x > W + 80 or y < -80 or y > H + 80:
                continue
            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(it["rot0"] + it["spin"] * (T - T_GAL))
            kind = it["kind"]
            if kind in ("map", "stone"):
                poly(ctx, [tuple(v) for v in it["shp"] * s * (1.0 if kind == "map" else 0.8)])
                set_color(ctx, it["col"])
                ctx.fill_preserve()
                set_color(ctx, "#E8F0FF" if kind == "map" else "#2A2438", 0.6)
                ctx.set_line_width(1.2)
                ctx.stroke()
            elif kind == "slop":
                rrect(ctx, -s * 0.7, -s * 0.45, s * 1.4, s * 0.9, s * 0.12)
                set_color(ctx, it["col"])
                ctx.fill()
                set_color(ctx, "#5A5068", 0.5)
                for L in range(3):
                    rect(ctx, -s * 0.55, -s * 0.28 + L * s * 0.2, s * (0.9 - 0.25 * (L % 2)), s * 0.07)
                ctx.fill()
            elif kind == "phone":
                rrect(ctx, -s * 0.3, -s * 0.55, s * 0.6, s * 1.1, s * 0.1)
                set_color(ctx, it["col"])
                ctx.fill()
                rrect(ctx, -s * 0.24, -s * 0.46, s * 0.48, s * 0.9, s * 0.06)
                set_color(ctx, "screen_glow", 0.85)
                ctx.fill()
            elif kind == "crown":
                crown(ctx, 0, s * 0.3, s * 1.1)
            ctx.restore()

    def _commander(self, ctx, gctx, T, fc):
        x, y, s = self.cmd_xy(T)
        if y > H + self.cmd_h * s * 1.4 or x < -self.cmd_h:
            return
        i = int(round((T - self.cmd_T0) * 24))
        i = min(max(i, 0), self.cmd_n - 1)
        h = self.cmd_h
        m = self.S.tl.mouth("COMMANDER", T)
        kw = dict(facing=1, back=True, silhouette="#0A0C1A", rim=("#BDB6FF", 0.9, (1.0, -0.4)), wind=0.6,
                  pole_len=h * 1.65)
        # the rock he stands on (a dark shard); a soft backlight so the silhouette reads against the void
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(s, s)
        ctx.set_source(rad_grad(40, -h * 0.6, 0, h * 0.95, [(0, "#5E58B0", 0.42), (1, "#5E58B0", 0.0)]))
        ctx.paint()
        poly(ctx, [(-700, 30), (-160, -6), (180, 8), (330, 60), (420, 400), (-700, 420)])
        set_color(ctx, "#080A16")
        ctx.fill()
        self.cmd.draw(ctx, 0.0, 0.0, h, "hold_flag", T - T0, mouth=m, layer="back", **kw)
        self.cmd_flag.draw(ctx, i, light=(0.6, -0.3, -0.7), glow=0.7)
        self.cmd.draw(ctx, 0.0, 0.0, h, "hold_flag", T - T0, mouth=m, layer="front", **kw)
        ctx.restore()

    # ================================================================== sound
    def events(self):
        ev = [dict(t=T_PLANT, kind="thunk", strength=1.0, pan=0.0, desc="Flag planted on the crown of the colossal marble head"),
              dict(t=T_PLANT + 0.03, kind="crack", strength=0.9, pan=0.0, desc="Voronoi cracks race down the marble from the pole"),
              dict(t=T_BURST, kind="shatter", strength=1.0, pan=0.0, desc="the head bursts into ~200 fragments; the agora falls away into a starry void"),
              dict(t=T_BURST + 0.1, kind="rumble", strength=0.8, pan=0.0, desc="plaza slabs fall away"),
              dict(t=T_POP, kind="pop", strength=0.5, pan=0.0, desc="tiny crackpots pop up on every fragment"),
              dict(t=T_SCREAM, kind="walla", strength=0.7, pan=0.0, desc="hundreds of crackpots start screaming (shout rings everywhere)")]
        for w_ in self.S.tl.words("X07"):
            ev.append(dict(t=w_["s"], kind="echo", strength=0.6, pan=0.25, desc=f"echo ring on '{w_['w']}'"))
        ev.append(dict(t=120.93, kind="whoosh", strength=0.7, pan=0.2, desc="whip pan crackpot 1 -> crackpot 2"))
        ev.append(dict(t=T_GAL, kind="whoosh", strength=0.8, pan=-0.3, desc="pull back: everything swirls into a galaxy; Commander silhouette"))
        ev.append(dict(t=124.6, kind="whoosh", strength=0.7, pan=-0.6, desc="camera sails past the Commander into the storm swirl"))
        return ev


SILVER_COL = "#D9DEE8"

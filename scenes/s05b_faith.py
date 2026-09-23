"""s05b FAITH (107.8-114.6): a vast temple; a Flag is planted in the aisle; the floor cracks and the congregation
splits into cabals, cabals split again (4 -> 16 -> 64) until every hooded figure stands alone in a tiny circle with
a tiny altar, a candle and a halo of their own. Heretic! / No, you're the heretic! / I am my own church!
"""
import math

import cairo
import numpy as np

from vx import *
from vx.flag import Flag
from vx.chars import Character
from scenes.s05b_common import T0, lg, pose_ok

F_PLANT = 107.8
SPLITS = (108.05, 108.40, 108.83)          # "faith" / "cabal" / "one"
ROBES = ["cloth_plum", "#5A4A66", "cloth_grey", "#6E6480", "#4E4A5E", "#7A6A86", "#5F5A70", "#6B4A6B"]
STONE_D, STONE, STONE_L = "#1B1830", "#35304F", "#5C5678"
HALO = "#EDEBFF"
FLAME = "#DCE8FF"

# ------------------------------------------------------------------ formation (floor coords: x right, z away, m)
QC = [(-5.6, 13.0), (5.6, 13.0), (-5.6, 24.0), (5.6, 24.0)]     # stage-1 cabal centres (quadrants)
OFF2 = [(-2.6, -2.6), (2.6, -2.6), (-2.6, 2.6), (2.6, 2.6)]
OFF3 = [(-1.25, -1.25), (1.25, -1.25), (-1.25, 1.25), (1.25, 1.25)]
FLAG_XZ = (0.0, 18.5)


def _digits(i):
    return i // 16, (i // 4) % 4, i % 4


def formation():
    """positions (64, 4 stages, 2) and face targets"""
    P = np.zeros((64, 4, 2))
    for r in range(8):
        for c in range(8):
            q = (r // 4) * 2 + (c // 4)
            k = ((r % 4) // 2) * 2 + (c % 4) // 2
            m = (r % 2) * 2 + c % 2
            i = q * 16 + k * 4 + m
            gx = (c - 3.5) * 1.75 + (1.3 if c >= 4 else -1.3)
            gz = 11.0 + r * 2.1 + (0.9 if r >= 4 else 0.0)
            P[i, 0] = (gx, gz)
    for i in range(64):
        q, k, m = _digits(i)
        qc = np.array(QC[q])
        # stage 1: ring of 16 around the cabal centre, ordered by original angle
        P[i, 3] = qc + np.array(OFF2[k]) + np.array(OFF3[m])
        P[i, 2] = qc + np.array(OFF2[k]) + np.array(OFF3[m]) * 0.62
    for q in range(4):
        idx = [i for i in range(64) if i // 16 == q]
        qc = np.array(QC[q])
        ang = [math.atan2(P[i, 0, 1] - qc[1], P[i, 0, 0] - qc[0]) for i in idx]
        order = np.argsort(ang)
        for n, j in enumerate(order):
            a = n / 16 * 2 * math.pi + math.pi / 16
            P[idx[j], 1] = qc + 3.3 * np.array([math.cos(a), math.sin(a)])
    return P


class Cam3:
    """pinhole camera: position (x, y, z) m, yaw/pitch rad (pitch + = looking down), focal f design px"""

    def __init__(self, x, y, z, pitch, yaw=0.0, f=1150.0, cx=960.0, cy=540.0):
        self.p = np.array([x, y, z], float)
        self.pitch, self.yaw, self.f, self.cx, self.cy = pitch, yaw, f, cx, cy
        cp, sp = math.cos(pitch), math.sin(pitch)
        cyw, syw = math.cos(yaw), math.sin(yaw)
        # world -> camera: yaw about y, then pitch about x
        self.R = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]]) @ np.array([[cyw, 0, -syw], [0, 1, 0], [syw, 0, cyw]])

    def cam(self, P):
        P = np.asarray(P, float)
        return (P - self.p) @ self.R.T

    def proj(self, P):
        c = self.cam(P)
        z = np.maximum(c[..., 2], 0.05)
        return np.stack([self.cx + self.f * c[..., 0] / z, self.cy - self.f * c[..., 1] / z], -1), z

    def scale(self, z):
        return self.f / z


def lerp3(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


# camera keys: (T, x, y, z, pitch, yaw, f)
CAM_KEYS = [
    (107.30, 0.0, 3.4, -2.5, 0.07, 0.0, 1000),
    (107.95, 0.0, 3.6, -1.0, 0.08, 0.0, 1000),
    (108.50, 0.0, 11.0, 1.0, 0.58, 0.0, 1080),
    (109.20, 0.0, 12.0, 2.2, 0.62, 0.0, 1080),
    (109.82, 0.0, 1.95, 4.9, 0.05, 0.0, 1200),
    (110.75, -0.3, 1.95, 5.1, 0.05, -0.02, 1200),
    (111.80, 0.3, 1.95, 5.1, 0.05, 0.02, 1200),
    (112.08, 6.8, 2.2, 3.9, 0.0, 0.0, 1100),
    (113.80, 6.95, 2.2, 4.3, 0.0, 0.0, 1100),
    (114.60, 6.95, 3.0, 3.2, 0.15, 0.0, 1100),
]


def cam_at(T):
    ks = CAM_KEYS
    if T <= ks[0][0]:
        k = ks[0]
    elif T >= ks[-1][0]:
        k = ks[-1]
    else:
        for a, b in zip(ks, ks[1:]):
            if a[0] <= T <= b[0]:
                u = (T - a[0]) / (b[0] - a[0])
                # whip into the chapel is fast, the rest glides
                u = ease_in_out(u, 3) if b[0] - a[0] < 0.4 else ease_in_out_sine(u)
                k = (T,) + lerp3(a[1:], b[1:], u)
                break
    _, x, y, z, pi, yw, f = k
    dx, dy = shake(T, 0.06 * math.exp(-max(0.0, T - F_PLANT) * 6) * (T >= F_PLANT), 14, 41)
    return Cam3(x + dx, y + dy, z, pi, yw, f)


class Faith:
    def __init__(self, S):
        self.S = S
        self.P = formation()
        self.people = [Character("cultist", seed=100 + i, hood=True, robe=ROBES[(i * 7) % len(ROBES)],
                                 skin=("skin1", "skin2", "skin3", "skin4")[(i * 5) % 4]) for i in range(64)]
        # heretics: two neighbours across the first crack (the aisle) + the chapel man further right
        pick = lambda x, z: min(range(64), key=lambda i: math.hypot(self.P[i, 3, 0] - x, self.P[i, 3, 1] - z))
        self.h1, self.h2, self.h3 = pick(-1.75, 9.15), pick(1.75, 9.15), pick(6.95, 9.15)
        self.chapel_xz = tuple(self.P[self.h3, 3])
        n = int(round((lg(114.8) - lg(F_PLANT - 0.5)) * 24)) + 2
        self.flag_f0 = int(round(lg(F_PLANT - 0.5) * 24))
        fp = 0.5 * 24

        def pole(i):
            k = clamp(i / fp)
            return (0.0, -700.0 * (1 - k * k), 0.0)

        self.flag = Flag(pole=300, seed=77).simulate(n, pole, lambda i: (260 + 80 * math.sin(i / 9.0), 0))
        rng = np.random.default_rng(3)
        self.dust = rng.uniform(-1, 1, (160, 3)) * np.array([14, 9, 22]) + np.array([0, 9, 24])

    # ---------------------------------------------------------------- choreography
    def stage_pos(self, i, T):
        """floor position of person i at time T (staggered, eased moves between formations)"""
        P = self.P[i]
        pos = P[0].copy()
        moving = 0.0
        for s, Ts in enumerate(SPLITS):
            d = hash01(i, 40 + s) * 0.08
            u = clamp((T - Ts - d) / 0.42)
            if u <= 0:
                break
            e = ease_in_out(u, 2) if u < 1 else 1.0
            pos = P[s] + (P[s + 1] - P[s]) * e
            if u < 1:
                moving = max(moving, math.sin(math.pi * u))
        return pos, moving

    def facing_target(self, i, T):
        q, k, m = _digits(i)
        if T < SPLITS[0]:
            return np.array([self.P[i, 0, 0] * 0.3, -5.0])            # the altar (toward camera)
        if T < SPLITS[1]:
            return np.array(QC[q])
        if T < SPLITS[2]:
            return np.array(QC[q]) + np.array(OFF2[k])
        return self.P[i, 3] + np.array([0.0, -0.8])                    # their own tiny altar

    # ---------------------------------------------------------------- render
    def render(self, fc, T=None):
        T = fc.T if T is None else T
        t = T - T0
        cam = cam_at(T)
        cv = Canvas(fc)
        ctx = cv.ctx
        self._architecture(ctx, cam, T)
        self._floor(ctx, cam, T)
        base = cv.rgb()
        # light shafts (additive)
        cvl = Canvas(fc)
        self._shafts(cvl.ctx, cam, T)
        lay = cvl.rgba()
        base = base + blur(lay[..., :3], 6 * fc.s) * 0.9
        heroes = {self.h1, self.h2, self.h3}
        dof = 3.2 * smoothstep(109.35, 109.8, T) * (1 - smoothstep(113.9, 114.3, T))
        bgc, bgg = Canvas(fc), Canvas(fc)
        self._actors(bgc.ctx, bgg.ctx, cam, T, fc, only=None if dof <= 0 else ("bg", heroes))
        img = over(base, bgc.rgba())
        gl = bgg.rgba()[..., :3]
        img = img + blur(gl, 10 * fc.s) * 0.9 + gl * 0.6
        if dof > 0:
            img = blur(img, dof * fc.s) * (1 - 0.3 * dof / 3.2)
            fgc, fgg = Canvas(fc), Canvas(fc)
            self._actors(fgc.ctx, fgg.ctx, cam, T, fc, only=("fg", heroes), halos=False)
            img = over(img, fgc.rgba())
            gl = fgg.rgba()[..., :3]
            img = img + blur(gl, 10 * fc.s) * 0.9 + gl * 0.6
        return img

    def _architecture(self, ctx, cam, T):
        # background: deep indigo gradient
        ctx.set_source(lin_grad(0, 0, 0, H, [(0, "#0E0C1E"), (0.5, STONE_D), (1, "#231F3A")]))
        ctx.paint()
        # far wall (entrance facade) with a great rose window at z = 40
        zf = 40.0
        wall = [(-13, 0, zf), (13, 0, zf), (13, 30, zf), (-13, 30, zf)]
        pts, _ = cam.proj(np.array(wall))
        poly(ctx, [tuple(p) for p in pts])
        ctx.set_source(lin_grad(0, pts[2][1], 0, pts[0][1], [(0, "#141128"), (1, "#2C2744")]))
        ctx.fill()
        rc, rz = cam.proj(np.array([0, 17.0, zf]))
        rr = 6.0 * cam.scale(rz)
        for k, (r, c, a) in enumerate(((1.25, "#6E6AA8", 0.18), (1.0, "#B9B6F0", 0.9), (0.78, "#7F7BC8", 1.0),
                                       (0.5, "#CFCBFF", 1.0), (0.18, "#F2F0FF", 1.0))):
            circle(ctx, rc[0], rc[1], rr * r)
            set_color(ctx, c, a)
            ctx.fill()
        # tracery
        set_color(ctx, "#1E1A36", 1.0)
        ctx.set_line_width(max(1.0, rr * 0.05))
        for k in range(12):
            a = k / 12 * 2 * math.pi
            ctx.move_to(rc[0] + math.cos(a) * rr * 0.18, rc[1] + math.sin(a) * rr * 0.18)
            ctx.line_to(rc[0] + math.cos(a) * rr, rc[1] + math.sin(a) * rr)
        ctx.stroke()
        for r in (0.5, 0.78, 1.0):
            circle(ctx, rc[0], rc[1], rr * r)
            ctx.stroke()
        # side walls + columns (nave arcade), vault ribs
        for side in (-1, 1):
            xw = 13.0 * side
            wall = [(xw, 0, -20), (xw, 0, zf), (xw, 30, zf), (xw, 30, -20)]
            pts, _ = cam.proj(np.array(wall))
            poly(ctx, [tuple(p) for p in pts])
            set_color(ctx, "#1A1730")
            ctx.fill()
        self._columns = []
        for z in np.arange(-2.0, 40.0, 5.0):
            for side in (-1, 1):
                self._columns.append((z, side))

    def _floor(self, ctx, cam, T):
        # tiled floor between the walls, z from -20 to 40 (clip against the camera near plane)
        zs = np.arange(-4, 41, 2.0)
        xs = np.arange(-13, 13.01, 2.0)
        for a, b in zip(zs[:-1], zs[1:]):
            for i, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
                q = [(x0, 0, a), (x1, 0, a), (x1, 0, b), (x0, 0, b)]
                c = cam.cam(np.array(q))
                if (c[:, 2] < 0.3).any():
                    continue
                pts, zz = cam.proj(np.array(q))
                poly(ctx, [tuple(p) for p in pts])
                chk = (int(a / 2) + i) % 2
                d = clamp((a - 5) / 40)
                base = mix("#3B3556", "#2A2642", chk * 0.6)
                set_color(ctx, mix(base, "#1B1830", d * 0.6))
                ctx.fill_preserve()
                set_color(ctx, "#15122A", 0.6)
                ctx.set_line_width(0.8)
                ctx.stroke()
        # central carpet runner (the aisle) toward the altar
        q = [(-1.1, 0, -6), (1.1, 0, -6), (1.1, 0, 40), (-1.1, 0, 40)]
        c = cam.cam(np.array(q))
        q2 = np.array(q, float)
        q2[:, 2] = np.maximum(q2[:, 2], cam.p[2] + 0.6)
        pts, _ = cam.proj(q2)
        poly(ctx, [tuple(p) for p in pts])
        set_color(ctx, "#4A2E4E", 0.85 * (1 - smoothstep(F_PLANT, SPLITS[0] + 0.3, T) * 0.7))
        ctx.fill()
        # cracks: quadtree schism lines racing out from the Flag
        self._cracks(ctx, cam, T)
        # ritual circles + tiny altars' shadows
        self._circles(ctx, cam, T)

    def _crack_lines(self):
        """list of (T_birth, origin(x,z), polyline (n,2)) on the floor"""
        L = []
        fx, fz = FLAG_XZ
        rng = np.random.default_rng(9)

        def jag(a, b, n=14, amp=0.35):
            a, b = np.array(a, float), np.array(b, float)
            u = np.linspace(0, 1, n)[:, None]
            p = a + (b - a) * u
            d = b - a
            nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
            p += nrm * (rng.normal(0, amp, (n, 1)) * np.sin(np.pi * u))
            return p

        T1, T2, T3 = SPLITS
        # stage 1: the cross through the Flag
        L += [(T1 - 0.2, (fx, fz), jag((fx, fz), (fx, 6.0))), (T1 - 0.2, (fx, fz), jag((fx, fz), (fx, 31.0))),
              (T1 - 0.15, (fx, fz), jag((fx, fz), (-12.0, fz))), (T1 - 0.15, (fx, fz), jag((fx, fz), (12.0, fz)))]
        # stage 2: each quadrant quartered
        for q, (cx, cz) in enumerate(QC):
            o = (cx, cz)
            L += [(T2 - 0.12, o, jag(o, (cx, cz - 5.2), 10, 0.2)), (T2 - 0.12, o, jag(o, (cx, cz + 5.2), 10, 0.2)),
                  (T2 - 0.12, o, jag(o, (cx - 5.4, cz), 10, 0.2)), (T2 - 0.12, o, jag(o, (cx + 5.4, cz), 10, 0.2))]
        # stage 3: each cabal of four quartered
        for q, (cx, cz) in enumerate(QC):
            for k, (ox, oz) in enumerate(OFF2):
                o = (cx + ox, cz + oz)
                for d in ((0, -2.5), (0, 2.5), (-2.5, 0), (2.5, 0)):
                    L.append((T3 - 0.1, o, jag(o, (o[0] + d[0], o[1] + d[1]), 7, 0.12)))
        return L

    def _cracks(self, ctx, cam, T):
        if not hasattr(self, "_CL"):
            self._CL = self._crack_lines()
        for Tb, o, pl in self._CL:
            u = (T - Tb) / 0.28
            if u <= 0:
                continue
            n = len(pl)
            m = int(clamp(u) * (n - 1)) + 1
            fr = clamp(u) * (n - 1) - (m - 1)
            pts3 = np.c_[pl[:, 0], np.zeros(n), pl[:, 1]]
            c = cam.cam(pts3)
            if (c[:, 2] < 0.3).any():
                keep = c[:, 2] >= 0.3
                if keep.sum() < 2:
                    continue
            pr, _ = cam.proj(pts3)
            path = [tuple(pr[i]) for i in range(m)]
            if m < n:
                path.append(tuple(pr[m - 1] + (pr[m] - pr[m - 1]) * fr))
            if len(path) < 2:
                continue
            fresh = math.exp(-max(0.0, T - Tb - 0.28) * 2.2)
            for wdt, colr, al in ((7.0, "#8E86FF", 0.18 * fresh), (3.0, "#0A0816", 0.9), (1.4, "#E8E4FF", 0.35 + 0.6 * fresh)):
                poly(ctx, path, close=False)
                set_color(ctx, colr, al)
                ctx.set_line_width(wdt)
                ctx.stroke()

    def _circles(self, ctx, cam, T):
        """chalk-pale ritual circles on the floor for each cabal level"""
        T1, T2, T3 = SPLITS
        rings = []
        a1 = smoothstep(T1 + 0.2, T1 + 0.5, T) * (1 - smoothstep(T2, T2 + 0.3, T))
        a2 = smoothstep(T2 + 0.2, T2 + 0.5, T) * (1 - smoothstep(T3, T3 + 0.3, T))
        a3 = smoothstep(T3 + 0.25, T3 + 0.55, T)
        for q, (cx, cz) in enumerate(QC):
            if a1 > 0:
                rings.append(((cx, cz), 4.3, a1))
            for k, (ox, oz) in enumerate(OFF2):
                if a2 > 0:
                    rings.append(((cx + ox, cz + oz), 1.9, a2))
                for m, (px, pz) in enumerate(OFF3):
                    if a3 > 0:
                        rings.append(((cx + ox + px * 1.0, cz + oz + pz * 1.0), 0.72, a3))
        ang = np.linspace(0, 2 * math.pi, 48)
        for (cx, cz), r, a in rings:
            pts3 = np.c_[cx + r * np.cos(ang), np.zeros_like(ang), cz + r * np.sin(ang)]
            if (cam.cam(pts3)[:, 2] < 0.3).any():
                continue
            pr, _ = cam.proj(pts3)
            poly(ctx, [tuple(p) for p in pr])
            set_color(ctx, "#C9C4F0", 0.55 * a)
            ctx.set_line_width(1.6)
            ctx.stroke()

    def _shafts(self, ctx, cam, T):
        # slanted light from clerestory windows on the left
        for k, z in enumerate(np.arange(4.0, 36.0, 7.0)):
            top = [(-13, 24, z), (-13, 24, z + 2.2)]
            bot = [(6, 0, z + 5.0), (6, 0, z + 7.2)]
            pts3 = np.array([top[0], top[1], bot[1], bot[0]])
            if (cam.cam(pts3)[:, 2] < 0.3).any():
                continue
            pr, _ = cam.proj(pts3)
            poly(ctx, [tuple(p) for p in pr])
            ctx.set_source(lin_grad(pr[0][0], pr[0][1], pr[3][0], pr[3][1],
                                    [(0, "#9C9AE0", 0.16), (1, "#9C9AE0", 0.0)]))
            ctx.fill()

    # ---------------------------------------------------------------- actors
    def _actors(self, ctx, gctx, cam, T, fc, only=None, halos=True):
        t = T - T0
        items = []
        for i in range(64):
            if only is not None and ((i in only[1]) != (only[0] == "fg")):
                continue
            (x, z), mv = self.stage_pos(i, T)
            items.append((z, "p", i, x, mv))
        if only is None or only[0] == "bg":
            for z, side in getattr(self, "_columns", []):
                items.append((z, "col", side, 0, 0))
            items.append((FLAG_XZ[1], "flag", 0, FLAG_XZ[0], 0))
        if T > 111.7:
            items.append((self.chapel_xz[1] + 0.001, "chapel_back", 0, 0, 0))
        items.sort(key=lambda it: -it[0])
        for z, kind, i, x, mv in items:
            if kind == "col":
                self._column(ctx, cam, i, z)
            elif kind == "flag":
                self._flag(ctx, gctx, cam, T, fc)
            elif kind == "chapel_back":
                continue
            else:
                self._person(ctx, gctx, cam, T, i, x, z, mv, fc)
        if halos:
            self._halos(gctx, ctx, cam, T)

    def _column(self, ctx, cam, side, z):
        x = 10.0 * side
        base = np.array([x, 0, z])
        c = cam.cam(base)
        if c[2] < 0.5:
            return
        (bx, by), zz = cam.proj(base)
        (tx, ty), _ = cam.proj(np.array([x, 22, z]))
        w = 1.3 * cam.scale(zz)
        g = lin_grad(bx - w / 2, 0, bx + w / 2, 0, [(0, "#16132A"), (0.35 if side > 0 else 0.65, "#4A4468"),
                                                     (1, "#16132A")])
        ctx.move_to(bx - w / 2, by)
        ctx.line_to(tx - w / 2, ty)
        ctx.line_to(tx + w / 2, ty)
        ctx.line_to(bx + w / 2, by)
        ctx.close_path()
        ctx.set_source(g)
        ctx.fill()
        # base + capital
        for (yy, hh) in ((0.0, 0.8), (21.2, 0.8)):
            (x0, y0), _ = cam.proj(np.array([x, yy, z]))
            (x1, y1), _ = cam.proj(np.array([x, yy + hh, z]))
            set_color(ctx, "#2A2542")
            rect(ctx, x0 - w * 0.7, y1, w * 1.4, y0 - y1)
            ctx.fill()

    def _flag(self, ctx, gctx, cam, T, fc):
        base = np.array([FLAG_XZ[0], 0, FLAG_XZ[1]])
        if cam.cam(base)[2] < 0.5:
            return
        (bx, by), zz = cam.proj(base)
        k = 2.44 * cam.scale(zz) / 300.0
        i = fc.f - self.flag_f0
        if i < 0:
            return
        i = min(i, self.flag.n - 1)
        ctx.save()
        ctx.translate(bx, by)
        ctx.scale(k, k)
        self.flag.draw(ctx, i, light=(-0.5, -0.6, 0.6), glow=0.4)
        ctx.restore()
        # impact ring on the floor
        tp = T - F_PLANT
        if 0 <= tp < 0.8:
            r = 1.0 + 9.0 * ease_out(tp / 0.8, 2)
            ang = np.linspace(0, 2 * math.pi, 64)
            pts3 = np.c_[FLAG_XZ[0] + r * np.cos(ang), np.zeros(64), FLAG_XZ[1] + r * np.sin(ang)]
            if (cam.cam(pts3)[:, 2] > 0.3).all():
                pr, _ = cam.proj(pts3)
                poly(gctx, [tuple(p) for p in pr])
                set_color(gctx, "#E8E4FF", 0.9 * (1 - tp / 0.8))
                gctx.set_line_width(4 * (1 - tp / 0.8) + 1)
                gctx.stroke()

    def _person(self, ctx, gctx, cam, T, i, x, z, mv, fc):
        base = np.array([x, 0, z])
        if cam.cam(base)[2] < 0.6:
            return
        (px, py), zz = cam.proj(base)
        s = cam.scale(zz)
        h = 1.7 * s
        if px < -h or px > W + h or py < -10 or py > H + h * 1.5:
            return
        ch = self.people[i]
        tgt = self.facing_target(i, T)
        dx, dz = tgt[0] - x, tgt[1] - z
        # screen facing: side component decides left/right, depth component -> front vs profile
        dist = math.hypot(dx, dz) + 1e-6
        side = dx / dist
        facing = 1 if side >= 0 else -1
        turn = clamp(abs(side) * 1.1)
        if dz > 0.2 * dist:
            turn = max(turn, 0.75)       # facing away: show profile-ish
        pose = "walk" if mv > 0.25 else "stand"
        expr = "serene"
        mouth = 0.0
        look = (0.0, 0.0)
        wkw = {}
        if T > SPLITS[2] + 0.4:
            pose = "meditate" if hash01(i, 3) > 0.6 else "stand"
            expr = ("serene", "smug", "skeptical", "determined")[int(hash01(i, 4) * 4) % 4]
        T3a = 109.75
        if i in (self.h1, self.h2, self.h3):
            wkw["rim"] = ("#D4CFFF", 0.75, (0.2, -1.0))
        if i == self.h1:
            pose, expr, facing, turn = self._h1(T, fc)
            mouth = fc.tl.mouth("HERETIC1", T)
        elif i == self.h2:
            pose, expr, facing, turn = self._h2(T, fc)
            mouth = fc.tl.mouth("HERETIC2", T)
        elif i == self.h3:
            pose, expr, facing, turn = self._h3(T, fc)
            mouth = fc.tl.mouth("HERETIC3", T)
        elif T > T3a:
            # background cabals-of-one mutter at their altars
            mouth = max(0.0, math.sin(T * 8 + i)) * 0.4 * (math.sin(T * 1.7 + i * 0.7) > 0.2)
        chapel = i == self.h3 and T > 111.75
        if chapel:
            self._chapel(ctx, gctx, cam, T, x, z, back=True)
        # tiny altar + candle in front (toward camera) once alone
        a3 = smoothstep(SPLITS[2] + 0.3, SPLITS[2] + 0.7, T)
        if a3 > 0 and not chapel:
            self._altar(ctx, gctx, cam, T, x, z - 0.75, a3, i, front=False)
        ch.draw(ctx, px, py, h, pose_ok(pose), T - T0 + i * 0.37, mouth=mouth, expr=expr, facing=facing, turn=turn,
                look=look, alpha=1.0, **wkw)
        if chapel:
            self._chapel(ctx, gctx, cam, T, x, z, back=False)

    def _h1(self, T, fc):
        w0, w1 = 109.9, 110.7
        if w0 - 0.15 <= T <= w1 + 0.6:
            return "point", "angry", 1, 0.55
        if 110.8 <= T <= 112.0:
            return "recoil", "surprised", 1, 0.55
        return ("stand" if T < SPLITS[2] + 0.4 else "stand"), "skeptical", 1, 0.45

    def _h2(self, T, fc):
        if 110.75 <= T <= 112.0:
            return "point", "furious", -1, 0.55
        if 109.95 <= T < 110.75:
            return "recoil", "surprised", -1, 0.55
        return "stand", "skeptical", -1, 0.45

    def _h3(self, T, fc):
        if T >= 112.05:
            k = fc.tl.word("X05", "church")
            if T >= k[0] - 0.1:
                return "declare", "proud", 0, 0.1
            return "point_self", "proud", 0, 0.15
        return "stand", "serene", 0, 0.2

    def _altar(self, ctx, gctx, cam, T, x, z, a, i, front=False):
        base = np.array([x, 0, z])
        if cam.cam(base)[2] < 0.6:
            return
        (bx, by), zz = cam.proj(base)
        s = cam.scale(zz) * (0.4 + 0.6 * ease_out_back(clamp(a), 2))
        w, hgt = 0.42 * s, 0.55 * s
        set_color(ctx, "#6E6A88")
        rect(ctx, bx - w / 2, by - hgt, w, hgt)
        ctx.fill()
        set_color(ctx, "#8F8AAE")
        rect(ctx, bx - w * 0.62, by - hgt - 0.06 * s, w * 1.24, 0.08 * s)
        ctx.fill()
        set_color(ctx, "#48445E")
        rect(ctx, bx + w * 0.1, by - hgt, w * 0.4, hgt)
        ctx.fill()
        # candle
        cw, chh = 0.05 * s, 0.2 * s
        set_color(ctx, "#E9E6F2")
        rect(ctx, bx - cw / 2, by - hgt - 0.06 * s - chh, cw, chh)
        ctx.fill()
        fy = by - hgt - 0.06 * s - chh
        fl = 1 + 0.15 * math.sin(T * 23 + i * 1.3) + 0.1 * math.sin(T * 37 + i)
        ellipse(ctx, bx, fy - 0.06 * s * fl, 0.028 * s, 0.07 * s * fl)
        set_color(ctx, FLAME, a)
        ctx.fill()
        circle(gctx, bx, fy - 0.06 * s, 0.12 * s)
        set_color(gctx, "#9FB2FF", 0.6 * a)
        gctx.fill()

    def _chapel(self, ctx, gctx, cam, T, x, z, back):
        """phone-booth-sized chapel with a steeple that assembles around HERETIC3"""
        u = ease_out_back(clamp((T - 111.75) / 0.35), 1.6)
        v = ease_out_back(clamp((T - 111.95) / 0.3), 2.2)
        base = np.array([x, 0, z])
        (bx, by), zz = cam.proj(base)
        s = cam.scale(zz)
        w, hh = 1.0 * s, 2.25 * s * u
        if back:
            # back wall + interior (a pale glow behind him)
            set_color(ctx, "#2D2945")
            rect(ctx, bx - w / 2, by - hh, w, hh)
            ctx.fill()
            ctx.set_source(rad_grad(bx, by - hh * 0.6, 0, w * 0.8, [(0, "#8C88C8", 0.5 * u), (1, "#2D2945", 0.0)]))
            rect(ctx, bx - w / 2, by - hh, w, hh)
            ctx.fill()
            return
        # front frame: two jambs + arch lintel (the open door shows him inside)
        wall = "#B9B6CE"
        wall_d = "#7E7A98"
        jw = 0.14 * s
        for sx in (-1, 1):
            set_color(ctx, wall)
            rect(ctx, bx + sx * (w / 2 - jw / 2) - jw / 2, by - hh, jw, hh)
            ctx.fill()
            set_color(ctx, wall_d)
            rect(ctx, bx + sx * (w / 2 - jw / 2) + (jw / 2 - jw * 0.35 if sx > 0 else -jw / 2), by - hh, jw * 0.35, hh)
            ctx.fill()
        # arch
        ctx.new_path()
        ctx.move_to(bx - w / 2, by - hh + 0.35 * s * u)
        ctx.line_to(bx - w / 2, by - hh)
        ctx.line_to(bx + w / 2, by - hh)
        ctx.line_to(bx + w / 2, by - hh + 0.35 * s * u)
        ctx.arc_negative(bx, by - hh + 0.35 * s * u + w * 0.1, w / 2 - jw, -0.2, math.pi + 0.2)
        ctx.close_path()
        set_color(ctx, wall)
        ctx.fill()
        # steeple: drops in
        if v > 0:
            top = by - hh - (1 - v) * 1.5 * s
            set_color(ctx, "#3E3A5A")
            poly(ctx, [(bx - w * 0.62, top), (bx + w * 0.62, top), (bx, top - 1.3 * s)])
            ctx.fill()
            set_color(ctx, "#2A2742")
            poly(ctx, [(bx, top), (bx + w * 0.62, top), (bx, top - 1.3 * s)])
            ctx.fill()
            # spire + a tiny halo of his own
            set_color(ctx, "#C9C6DE")
            ctx.set_line_width(0.05 * s)
            ctx.move_to(bx, top - 1.3 * s)
            ctx.line_to(bx, top - 1.75 * s)
            ctx.stroke()
            ellipse(gctx, bx, top - 1.95 * s, 0.22 * s, 0.07 * s)
            set_color(gctx, HALO, 0.95 * v)
            gctx.set_line_width(0.045 * s)
            gctx.stroke()
            # tiny bell swinging in a little window
            (bw) = 0.12 * s
            ang = 0.5 * math.sin((T - 112.0) * 12) * math.exp(-max(0.0, T - 112.0) * 1.5)
            set_color(ctx, "#1B1830")
            circle(ctx, bx, top - 0.5 * s, 0.2 * s)
            ctx.fill()
            ctx.save()
            ctx.translate(bx, top - 0.62 * s)
            ctx.rotate(ang)
            set_color(ctx, "#9A96B8")
            poly(ctx, [(-bw * 0.6, bw * 1.3), (bw * 0.6, bw * 1.3), (bw * 0.3, 0), (-bw * 0.3, 0)])
            ctx.fill()
            ctx.restore()

    def _halos(self, gctx, ctx, cam, T):
        """one great halo splits 1 -> 4 -> 16 -> 64 and each flies to its cabal"""
        T1, T2, T3 = SPLITS
        items = []
        cen = np.array([0.0, 18.5])
        if T < T1 + 0.45:
            u = clamp((T - T1) / 0.45)
            if u <= 0:
                items.append((cen, 9.0, 9.5, 1.0))
            else:
                e = ease_in_out(u, 2)
                for q in range(4):
                    p = cen + (np.array(QC[q]) - cen) * e
                    items.append((p, 9.0 - (9.0 - 3.4) * e, 9.5 - 3.5 * e, 1.0))
        if T1 + 0.45 <= T < T2 + 0.45:
            u = clamp((T - T2) / 0.45)
            e = ease_in_out(u, 2)
            for q in range(4):
                for k in range(4):
                    p = np.array(QC[q]) + np.array(OFF2[k]) * e
                    items.append((p, 3.4 - 1.9 * e, 6.0 - 2.6 * e, 1.0))
        if T >= T2 + 0.45:
            u = clamp((T - T3) / 0.45)
            e = ease_in_out(u, 2)
            for q in range(4):
                for k in range(4):
                    for m in range(4):
                        i = q * 16 + k * 4 + m
                        if i == self.h3 and T > 111.9:
                            continue
                        c = np.array(QC[q]) + np.array(OFF2[k])
                        p = c + np.array(OFF3[m]) * e
                        items.append((p, 1.5 - 1.15 * e, 3.4 - 1.25 * e, 1.0))
        ang = np.linspace(0, 2 * math.pi, 40)
        for p, r, y, a in items:
            pts3 = np.c_[p[0] + r * np.cos(ang), np.full(40, y), p[1] + r * np.sin(ang)]
            if (cam.cam(pts3)[:, 2] < 0.4).any():
                continue
            pr, zz = cam.proj(pts3)
            s = cam.scale(zz.mean())
            lw = max(1.2, 0.09 * r * s * 0.35)
            for wmul, al in ((3.2, 0.18), (1.6, 0.4), (0.7, 1.0)):
                poly(gctx, [tuple(q) for q in pr])
                set_color(gctx, HALO if wmul < 1 else "#A9A2FF", a * al)
                gctx.set_line_width(lw * wmul)
                gctx.stroke()

    def events(self):
        ev = [dict(t=F_PLANT, kind="thunk", strength=1.0, pan=0.0, desc="Flag planted in the temple aisle; floor impact ring"),
              dict(t=F_PLANT + 0.02, kind="crack", strength=0.9, pan=0.0, desc="stone floor cracks begin under the Flag")]
        for k, Ts in enumerate(SPLITS):
            ev.append(dict(t=Ts - 0.15 if k < 2 else Ts - 0.1, kind="crack", strength=0.9 - 0.15 * k, pan=0.0,
                           desc=f"floor schism cracks, generation {k + 1} (1->4->16->64)"))
            ev.append(dict(t=Ts, kind="shuffle", strength=0.6, pan=0.0,
                           desc=f"congregation splits into {4 ** (k + 1)} cabals: robes rustle, feet shuffle"))
            ev.append(dict(t=Ts + 0.05, kind="chime", strength=0.5, pan=0.0,
                           desc=f"halo splits into {4 ** (k + 1)} halos (glassy shimmer)"))
        ev.append(dict(t=SPLITS[2] + 0.4, kind="candles", strength=0.4, pan=0.0, desc="64 tiny candles ignite"))
        ev.append(dict(t=111.75, kind="thunk", strength=0.6, pan=0.3, desc="phone-booth chapel walls pop up"))
        ev.append(dict(t=111.95, kind="thunk", strength=0.5, pan=0.3, desc="steeple drops onto the chapel"))
        ev.append(dict(t=112.0, kind="bell", strength=0.5, pan=0.3, desc="tiny chapel bell swings and tings"))
        return ev

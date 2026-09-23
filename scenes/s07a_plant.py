"""s07a — THE UNBABELING (139.0–154.5).

From black a thin dawn line; dawn over a vast plain of dried slop cracked into ten thousand little
"nations of one", each survivor alone in the glow of their private feed. Lutie climbs the rubble hill and
plants the hypermeme Flag (142.0 flag_planted: THUNK, dust ring, first sunlight, the cloth unfurls in the
first wind). Crocus rises behind her: "A Flag needs no translation. It means only... Flag." The survivors
turn and look up (148.5). N08 "Each of them alone. All of them together." 150.0 convergence: flags appear
in every hand in a front that nucleates and spreads like a phase transition (first-passage percolation on
the plate graph) while the camera cranes up into the high aerial that s07b continues.

The world (camera, plain, survivors, hill) lives in scenes/s07a_world.py (a true 3D perspective renderer).
This module adds the heroes, the plant, dust, god rays, the grade, and the foley exports.
"""
import math

import cairo
import cv2
import numpy as np

from vx import *
from vx.flag import Flag, FlagTrack
from vx.chars import Character
from vx import foley
from scenes import s07a_world as WD
from scenes.s07a_world import (camera_at, cam_basis, Proj, render_world, get_world, mound_z, light_curves, sun_dir,
                               wind_az, HILL_TOP, T_PLANT, T_LOOK, T_CONV)

POST = dict(bloom=0.42, bloom_thresh=0.8, bloom_radius=24.0, grain=0.03, vignette=0.3)

HERO_VER = "h12"
EXTRA_FRAMES = 96          # hero sim runs past 154.5 so s07b can keep drawing it
U = 100.0                  # hero plane units: centimetres

# plane of the plant: the crest, x = the plant camera's right (horizontal), y = down
_C0 = camera_at(141.0)
_, _R0, _U0, _F0, _ = cam_basis(_C0)
EX0 = np.array([_R0[0], _R0[1], 0.0]) / math.hypot(_R0[0], _R0[1])
FH0 = np.array([_F0[0], _F0[1], 0.0]) / math.hypot(_F0[0], _F0[1])
PLANE_AZ0 = math.degrees(math.atan2(EX0[0], EX0[1]))
CREST = np.array([0.0, 0.0, HILL_TOP])


def plane_az(T):
    """the hero cloth's plane turns with the veering wind"""
    return PLANE_AZ0 + (wind_az(T) - wind_az(139.0))


def ex_at(T):
    a = math.radians(plane_az(T))
    return np.array([math.sin(a), math.cos(a), 0.0])


def ground_at(P):
    return float(mound_z(np.array([P[0]]), np.array([P[1]]))[0])


# ============================================================== choreography (global T)
T_WALK0, T_TOP, T_LIFT, T_SLAM = 139.95, 141.58, 141.62, 141.90


def lutie_x(T):
    """plane-x (m) of Lutie's feet: a scramble up the slope from the left"""
    u = clamp((T - T_WALK0) / (T_TOP - T_WALK0))
    u = 1 - (1 - u) ** 1.7                                   # arrives running, brakes at the top
    x = lerp(-9.3, -0.30, u)
    x += 0.05 * math.sin(max(0.0, T - T_WALK0) * 9.0) * (1 - seg(T, T_TOP - 0.2, T_TOP))
    x -= 0.04 * seg(T, 142.7, 143.3, ease_in_out)          # settles back a touch to look up at it
    return x


def lutie_foot(T):
    P = CREST + EX0 * lutie_x(T)
    P[2] = ground_at(P)
    return P


def crocus_foot(T):
    """Crocus rises over the crest behind her (from the far side), then stands"""
    k = seg(T, 142.75, 143.75, lambda u: ease_out(u, 2))
    depth = lerp(5.2, 2.3, k)
    x = lerp(-2.6, -1.75, k)
    P = CREST + EX0 * x + FH0 * depth
    P[2] = ground_at(P)
    return P


def pole_plane(T):
    """hero pole BASE in plane coords (cm, y down, origin = crest) and angle (rad, + = leaning right)"""
    xl = lutie_x(T) * U
    Pl = lutie_foot(T)
    yl = (HILL_TOP - Pl[2]) * U
    bob = 3.0 * math.sin(max(0.0, T - T_WALK0) * 9.0 * 2)
    # carried on the shoulder, pointing ahead and up
    a_c = 0.98
    gx, gy = xl + 20.0, yl - 116.0 + bob
    d = (math.sin(a_c), -math.cos(a_c))
    carry = (gx - d[0] * 92.0, gy - d[1] * 92.0, a_c)
    # lifted upright above the plant spot
    lift = (0.0, -46.0, 0.03)
    # slammed down into the rubble
    planted = (0.0, 20.0, 0.0)
    if T < T_LIFT:
        return carry
    if T < T_SLAM:
        k = seg(T, T_LIFT, T_SLAM, lambda u: ease_in_out(u, 2))
        return tuple(lerp(a, b, k) for a, b in zip(carry, lift))
    if T < T_PLANT:
        k = seg(T, T_SLAM, T_PLANT, lambda u: ease_in(u, 2.4))
        return tuple(lerp(a, b, k) for a, b in zip(lift, planted))
    dt = T - T_PLANT
    wob = 0.06 * math.exp(-dt / 0.28) * math.sin(2 * math.pi * 3.4 * dt)
    return (0.0, 20.0, wob + 0.012 * seg(T, 142.3, 143.5))


def wind_plane(T):
    """dawn wind in the hero plane (design px/s on the canonical pole: breeze 300, strong 700, gale 1200)"""
    base = 70.0
    gust1 = 980.0 * seg(T, 142.02, 142.22, ease_out) * (1 - 0.45 * seg(T, 142.5, 143.6, ease_in_out))
    steady = 560.0 * seg(T, 142.4, 143.2, ease_in_out)
    g2 = 330.0 * pulse(T, 145.2, 0.45)
    g3 = 520.0 * pulse(T, 147.55, 0.35)                       # the snap on "...Flag."
    g4 = 260.0 * seg(T, 149.8, 151.0, ease_in_out)
    w = base + max(gust1, steady) + g2 + g3 + g4
    return (w, -25.0, 60.0)


def furl(T):
    return 1.0 - seg(T, 142.0, 142.42, lambda u: ease_in_out(u, 2))


# ============================================================== hero drawing
_CH = {}


def _ch(kind):
    if kind not in _CH:
        _CH[kind] = Character(kind, seed=3 if kind == "lutie" else 5)
    return _CH[kind]


def billboard(pr, P, cheat=0.62):
    """camera-facing billboard matrix (local cm, y down, origin at the feet) at world point P"""
    sx, sy, z = pr.pt(*P)
    tx, ty, _ = pr.pt(P[0], P[1], P[2] + 1.0)
    kx = pr.F / max(z, 1e-3)
    upx, upy = tx - sx, ty - sy
    ul = math.hypot(upx, upy) + 1e-9
    if ul < cheat * kx:
        upx, upy = upx / ul * cheat * kx, upy / ul * cheat * kx
    return cairo.Matrix(kx / U, 0.0, -upx / U, -upy / U, sx, sy), z, kx


def plane_matrix(pr, T):
    """hero-flag plane (cm, y down, origin crest) -> screen"""
    ex = ex_at(T)
    sx, sy, z = pr.pt(*CREST)
    ax, ay, _ = pr.pt(*(CREST + ex))
    ux, uy, _ = pr.pt(*(CREST + np.array([0, 0, 1.0])))
    kx = pr.F / max(z, 1e-3)
    upx, upy = ux - sx, uy - sy
    ul = math.hypot(upx, upy) + 1e-9
    if ul < 0.62 * kx:
        upx, upy = upx / ul * 0.62 * kx, upy / ul * 0.62 * kx
    wx, wy = ax - sx, ay - sy
    wl = math.hypot(wx, wy) + 1e-9
    if wl < 0.5 * kx:
        wx, wy = wx / wl * 0.5 * kx, wy / wl * 0.5 * kx
    return cairo.Matrix(wx / U, wy / U, -upx / U, -upy / U, sx, sy), z


def to_local(M, x, y):
    Mi = cairo.Matrix(*M)
    Mi.invert()
    return Mi.transform_point(x, y)


class Heroes:
    """Lutie, Crocus and the hypermeme Flag on the crest — plugs into s07a_world.render_world"""

    def __init__(self, st, T, f=None):
        self.st, self.T = st, T
        self.i = (T - 139.0) * 24.0 if f is None else f

    # --- hero flag
    def _flag_i(self):
        return min(self.i, self.st["hero"].n - 1.0)

    def items(self, pr, T):
        out = []
        Pl = lutie_foot(T)
        _, _, zl = pr.pt(*Pl)
        Pc = crocus_foot(T)
        _, _, zc = pr.pt(*Pc)
        if T > 142.7:
            out.append((zc, lambda ctx: self.draw_crocus(ctx, pr, T)))
        out.append((zl - 1.2, lambda ctx: self.draw_lutie_and_flag(ctx, pr, T)))
        return out

    def light_params(self, pr, T):
        L = light_curves(T)
        sd = sun_dir(T)
        _, r, u, f, _ = cam_basis(pr.cam)
        # sun direction in screen terms for rims
        sxv = float(sd @ r)
        syv = -float(sd @ u)
        n = math.hypot(sxv, syv) + 1e-9
        return L, (sxv / n, syv / n), float(sd @ f)

    def draw_lutie_and_flag(self, ctx, pr, T):
        st = self.st
        L, sdir, sfwd = self.light_params(pr, T)
        Pl = lutie_foot(T)
        M, z, kx = billboard(pr, Pl)
        Mp, _ = plane_matrix(pr, T)
        tr = st["hero"]
        fi = self._flag_i()
        # pole grip points (plane cm) -> Lutie's local coords
        bx, by, ang = tr.poles[int(min(fi, tr.n - 1))]
        d = (math.sin(ang), -math.cos(ang))

        def grip(s_):
            X, Y = Mp.transform_point(bx + d[0] * s_, by + d[1] * s_)
            return to_local(M, X, Y)
        ch = _ch("lutie")
        h = 155.0
        # ---- pose timeline
        if T < T_LIFT:
            kr = 1.0 - seg(T, 141.05, 141.5, ease_in_out)
            pose = {"run": max(1e-3, kr), "walk": max(1e-3, 1.0 - kr)}
            kw = dict(speed=1.0, lean=0.32 * kr + 0.12)
            hr, hl = grip(92.0), grip(55.0)
        elif T < 142.35:
            p = 0.6 * seg(T, T_LIFT, T_PLANT, lambda u: u) + 0.4 * seg(T, T_PLANT, 142.35, ease_out)
            pose = "plant_flag"
            kw = dict(progress=p)
            hr, hl = grip(96.0 + 30 * seg(T, T_SLAM, T_PLANT)), grip(138.0)
        else:
            pose = {"hold_flag": 1.0 - seg(T, 142.35, 142.9), "look_up": max(1e-3, seg(T, 142.35, 142.9))}
            kw = dict(nod=0.55 * seg(T, 142.3, 142.9, ease_in_out))
            hr = grip(104.0)
            hl = grip(140.0) if T < 142.55 else None
        expr = "determined" if T < 142.2 else ("awe" if T < 147.5 else "happy")
        wind = 0.25 + 0.9 * seg(T, 142.05, 142.4)
        # warm bounce light from the glowing cloth after it unfurls
        flagk = seg(T, 142.05, 142.6)
        rim_k = 0.7 + 0.3 * L["sun"]
        tint_k = 0.86 - 0.2 * L["sun"] - 0.1 * flagk
        base = dict(pose=pose, t=T, facing=1, expr=expr, wind=wind, mouth=0.0, light=(sdir[0], sdir[1]),
                    look=(0.3, -0.9 * seg(T, 142.3, 142.9)),
                    rim=((1.0, 0.70, 0.58), rim_k, (sdir[0], sdir[1])),
                    tint=((0.07, 0.06, 0.14), max(0.0, tint_k)), shape_r="grip", **kw)
        ik = dict(hand_r=hr)
        if hl is not None:
            ik["hand_l"] = hl
            base["shape_l"] = "grip"
        light = (-0.62, -0.25, -0.74)
        glow = 0.12 + 0.5 * flagk
        # ---- draw: Lutie (back layer) -> Flag -> Lutie (front layer: hands on the pole)
        ctx.save()
        ctx.transform(M)
        ch.draw(ctx, 0.0, 0.0, h, layer="back", **base, **ik)
        ctx.restore()
        ctx.save()
        ctx.transform(Mp)
        tr.draw(ctx, fi, light=light, glow=glow)
        ctx.restore()
        ctx.save()
        ctx.transform(M)
        ch.draw(ctx, 0.0, 0.0, h, layer="front", **base, **ik)
        ctx.restore()

    def draw_crocus(self, ctx, pr, T):
        L, sdir, _ = self.light_params(pr, T)
        Pc = crocus_foot(T)
        M, z, kx = billboard(pr, Pc)
        ch = _ch("crocus")
        walking = T < 143.75
        m = get_timeline().mouth("CROCUS", T)
        if walking:
            pose = "walk"
            kw = dict(speed=0.6, phase=(T - 142.75) * 1.1, lean=0.15)
        elif 147.35 < T < 148.4:
            pose = {"talk": 1.0 - seg(T, 147.35, 147.6), "gesture": max(1e-3, seg(T, 147.35, 147.6))}
            kw = {}
        else:
            pose = "talk" if 143.8 <= T <= 148.1 else "stand"
            kw = {}
        ctx.save()
        ctx.transform(M)
        ch.draw(ctx, 0.0, 0.0, 186.0, pose=pose, t=T, facing=1, expr="serene", mouth=m,
                look=(0.5, -0.5 * seg(T, 143.9, 144.6)), wind=0.9,
                rim=((1.0, 0.70, 0.58), 0.4 + 0.6 * L["sun"], (sdir[0], sdir[1])),
                tint=((0.07, 0.06, 0.14), 0.86 - 0.2 * L["sun"]), light=(sdir[0], sdir[1]), **kw)
        ctx.restore()

    def shadows(self, sctx, pr, T, sd):
        """the Flag's (and Lutie's) long shadow falls from the hill across the plain"""
        st = self.st
        tr = st["hero"]
        fi = int(min(self.i, tr.n - 1))
        ex = ex_at(T)
        V = tr.verts[fi]
        # cloth outline in world coords (plane cm -> world m)
        edge = np.concatenate([V[0, :, :2], V[:, -1, :2], V[-1, ::-1, :2], V[::-1, 0, :2]])
        W3 = CREST[None, :] + ex[None, :] * (edge[:, 0:1] / U) + np.array([0, 0, -1.0])[None, :] * (edge[:, 1:2] / U)
        W3 += np.array([0, 0, 0])
        G = WD.ground_shadow_pts(W3, sd)
        _poly(sctx, pr, G)
        bx, by, ang = tr.poles[fi]
        top = CREST + ex * ((bx + math.sin(ang) * 244) / U) + np.array([0, 0, -(by - math.cos(ang) * 244) / U])
        foot = CREST + np.array([0, 0, 0.0])
        G2 = WD.ground_shadow_pts(np.array([foot, top]), sd)
        pp = np.array([-(G2[1, 1] - G2[0, 1]), G2[1, 0] - G2[0, 0]])
        pp = pp / (np.linalg.norm(pp) + 1e-9) * 0.08
        _poly(sctx, pr, np.array([G2[0] + pp, G2[1] + pp, G2[1] - pp, G2[0] - pp]))
        # Lutie
        Pl = lutie_foot(T)
        Gl = WD.ground_shadow_pts(np.array([Pl, Pl + np.array([0, 0, 1.5])]), sd)
        pp = np.array([-(Gl[1, 1] - Gl[0, 1]), Gl[1, 0] - Gl[0, 0]])
        pp = pp / (np.linalg.norm(pp) + 1e-9) * 0.25
        _poly(sctx, pr, np.array([Gl[0] + pp, Gl[1] + pp * 0.7, Gl[1] - pp * 0.7, Gl[0] - pp]))

    def glows(self, pr, T):
        out = []
        if T > 142.0:
            tr = self.st["hero"]
            fi = self._flag_i()
            Mp, z = plane_matrix(pr, T)
            cx, cy = tr.cloth_center(fi)
            X, Y = Mp.transform_point(cx, cy)
            k = pr.F / max(z, 1e-3)
            fl = 1.2 * math.exp(-(T - 142.0) / 0.35) + 0.35 * seg(T, 142.0, 142.6)
            out.append((X, Y, 0.6 * k, (1.0 * fl, 0.84 * fl, 0.45 * fl)))
            # the first Flag stays a beacon at the heart of the ripple as the camera climbs away
            bk = seg(T, 149.5, 152.5, ease_in_out)
            if bk > 0:
                out.append((X, Y, 26.0, (0.9 * bk, 0.72 * bk, 0.34 * bk)))
        return out


def _poly(ctx, pr, G):
    P3 = np.concatenate([G, np.zeros((len(G), 1))], 1)
    xs, ys, zs = pr(P3)
    if np.any(zs < 0.5):
        return
    ctx.move_to(xs[0], ys[0])
    for a, b in zip(xs[1:], ys[1:]):
        ctx.line_to(a, b)
    ctx.close_path()
    ctx.fill()


# ============================================================== dust ring + debris at the plant
def dust_items(st, pr, T):
    dt = T - T_PLANT
    if dt < 0 or dt > 2.4:
        return []
    L = light_curves(T)
    sdir = sun_dir(T)
    P0 = CREST + np.array([0, 0, 0.05])

    def draw(ctx):
        rng = np.random.default_rng(42)
        # shockwave ring on the crest (projected circle)
        if dt < 0.45:
            k = ease_out(dt / 0.45, 2)
            R = 0.25 + 2.6 * k
            angs = np.linspace(0, 2 * math.pi, 48)
            ring = P0[None, :] + np.stack([np.cos(angs) * R, np.sin(angs) * R, np.zeros_like(angs)], 1)
            ring[:, 2] = np.array([mound_z(np.array([p[0]]), np.array([p[1]]))[0] for p in ring]) + 0.08
            xs, ys, zs = pr(ring)
            ctx.move_to(xs[0], ys[0])
            for a, b in zip(xs[1:], ys[1:]):
                ctx.line_to(a, b)
            ctx.close_path()
            ctx.set_line_width(3.0 * (1 - k) + 0.5)
            ctx.set_source_rgba(0.95, 0.72, 0.66, 0.55 * (1 - k))
            ctx.stroke()
        # dust puffs
        n = 90
        ang = rng.uniform(0, 2 * math.pi, n)
        spd = rng.uniform(1.2, 3.6, n)
        up = rng.uniform(0.15, 1.1, n)
        sz = rng.uniform(0.22, 0.6, n)
        life = rng.uniform(1.1, 2.3, n)
        for q in np.argsort(-rng.random(n)):
            if dt > life[q]:
                continue
            u = dt / life[q]
            rr = spd[q] * (1 - math.exp(-dt * 3.0)) / 3.0 * 2.2
            P = P0 + np.array([math.cos(ang[q]) * rr, math.sin(ang[q]) * rr, up[q] * (1 - math.exp(-dt * 2.5))])
            P[2] += ground_at(P) - HILL_TOP + 0.1 if rr > 0.6 else 0.0
            x, y, z = pr.pt(*P)
            if z < 0.5:
                continue
            rad = pr.F / z * sz[q] * (0.6 + 1.6 * u)
            a = 0.34 * (1 - u) ** 1.6 * min(1.0, dt / 0.06)
            lit = 0.5 + 0.5 * max(0.0, -math.cos(ang[q]) * 0.5)
            # backlit dust glows rose-gold against the dawn
            c = np.array((0.30, 0.27, 0.38)) * (0.7 + 0.6 * L["amb"]) + np.array((0.70, 0.42, 0.36)) * L["sun"] * lit
            g = cairo.RadialGradient(x, y, 0, x, y, rad)
            g.add_color_stop_rgba(0, *np.clip(c, 0, 1), a)
            g.add_color_stop_rgba(1, *np.clip(c, 0, 1), 0)
            ctx.set_source(g)
            ctx.arc(x, y, rad, 0, 2 * math.pi)
            ctx.fill()
        # pebbles thrown up
        m = 20
        ang = rng.uniform(0, 2 * math.pi, m)
        v = rng.uniform(1.5, 3.5, m)
        vz = rng.uniform(1.5, 3.8, m)
        for q in range(m):
            tt = min(dt, 2 * vz[q] / 9.8 + 0.05)
            P = P0 + np.array([math.cos(ang[q]) * v[q] * tt, math.sin(ang[q]) * v[q] * tt, vz[q] * tt - 4.9 * tt * tt])
            gz = ground_at(P)
            P[2] = max(P[2], gz - HILL_TOP + HILL_TOP) if P[2] < gz else P[2]
            x, y, z = pr.pt(*P)
            if z < 0.5:
                continue
            r = pr.F / z * 0.022 * (0.6 + 0.8 * ((q * 37) % 10) / 10.0)
            ctx.arc(x, y, max(0.5, r), 0, 2 * math.pi)
            ctx.set_source_rgba(0.12, 0.11, 0.17, 0.9 * (1 - seg(dt, 1.4, 2.2)))
            ctx.fill()

    _, _, z = pr.pt(*P0)
    return [(z - 1.6, draw)]


# ============================================================== god rays + flare (screen space)
def god_rays(img, fc, pr, T, alpha):
    sd = sun_dir(T)
    P = pr.C + sd * 5000.0
    sx, sy, z = pr.pt(*P)
    if z <= 0:
        return img
    s = fc.s
    k = 4
    h, w = img.shape[0] // k, img.shape[1] // k
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    occ = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
    lum = small.mean(axis=2)
    bright = np.clip(lum - 0.55, 0, None) * (1 - occ)
    cx, cy = sx * s / k, sy * s / k
    acc = np.zeros_like(bright)
    N = 22
    for q in range(N):
        sc = 1.0 - q * 0.028
        M = np.array([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]], np.float32)
        acc += cv2.warpAffine(bright, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * (1 - q / N)
    acc /= N
    rays = cv2.resize(acc, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    L = light_curves(T)
    kk = 0.9 * seg(T, 141.9, 142.3) * (1 - 0.5 * seg(T, 148, 151)) + 0.25 * seg(T, 140.2, 141.9)
    col_ = np.array([1.0, 0.72, 0.62], np.float32)
    return img + rays[..., None] * col_ * kk * 1.4


# ============================================================== setup
def _sim_hero(S):
    n = S.n + EXTRA_FRAMES
    Ts = [S.start + i / S.fps for i in range(n)]
    fl = Flag(pole=244.0, seed=9, side=1)
    tr = fl.simulate(n, lambda i: pole_plane(Ts[i]), lambda i: wind_plane(Ts[i]), fps=S.fps, substeps=12,
                     warmup=24, furl_fn=lambda i: furl(Ts[i]))
    return tr


def _state(S):
    world = get_world(S)
    p = S.cache / f"hero_{HERO_VER}.npz"
    if p.exists():
        tr = FlagTrack.load(p)
    else:
        tr = _sim_hero(S)
        tr.save(p)
    return dict(world=world, hero=tr)


def setup(S):
    st = _state(S)
    try:
        _export_foley(S, st)
    except Exception as ex:                       # never block rendering on the sound exports
        print("s07a foley export failed:", ex)
    return st


def cam_for(T):
    cam = camera_at(T)
    dt = T - T_PLANT
    if 0 <= dt < 0.6:
        amp = 9.0 * math.exp(-dt / 0.12)
        cam["shake"] = shake(T, amp, 26.0, seed=7)
    return cam


def render(fc, st):
    return render_at(fc, fc.T, st)


_HANDOFF = {}


def handoff_state():
    """world + hero flag track (from cache) for rendering this world outside the s07a runner (s07b hand-off)"""
    if "st" not in _HANDOFF:
        from vx.render import load_scene
        _, S = load_scene("s07a", 1.0)
        _HANDOFF["st"] = _state(S)
    return _HANDOFF["st"]


def render_at(fc, T, st=None):
    """Pure: the s07a frame at GLOBAL time T (any T, also beyond 154.5 where the crane keeps rising).
    fc needs .w .h .s (design->pixel scale). Returns float32 (fc.h, fc.w, 3) sRGB, before the film look (POST)."""
    st = st or handoff_state()
    cam = cam_for(T)
    pr = Proj(cam)
    heroes = Heroes(st, T)
    extras = dust_items(st, pr, T)
    img, lay = render_world(fc, cam, T, st["world"], heroes, extras, return_layers=True)
    img = god_rays(img, fc, pr, T, lay["alpha"])
    return np.clip(img, 0, 8).astype(np.float32)


def post(fc, st):
    T = fc.T
    d = {}
    # 139.00-139.05: pure black (s06 hard-cut); the dawn line then draws itself across
    d["fade"] = 1.0 - seg(T, 139.02, 139.12)
    d["exposure"] = 1.0 + 0.12 * pulse(T, 142.05, 0.18)
    d["bloom"] = 0.42 + 0.3 * pulse(T, 142.1, 0.35) + 0.25 * seg(T, 150.0, 153.5)
    return d


# ============================================================== foley exports
def _export_foley(S, st):
    wd = st["world"]
    ev = []
    # Lutie's climb: scrambling steps on rubble
    t = T_WALK0 + 0.12
    k = 0
    while t < T_TOP:
        x, y, _ = Proj(camera_at(t)).pt(*lutie_foot(t))
        ev.append(dict(t=t, kind="step_rubble", strength=0.35 + 0.15 * (k % 2), pan=foley.screen_pan(x),
                       desc="Lutie scrambles up the rubble hill (loose stones)"))
        t += 0.31 - 0.04 * (k % 3 == 0)
        k += 1
    xp, _, _ = Proj(camera_at(T_PLANT)).pt(*CREST)
    pan0 = foley.screen_pan(xp)
    ev += [
        dict(t=T_LIFT, kind="whoosh", strength=0.3, pan=pan0, desc="she heaves the furled Flag upright over her head"),
        dict(t=T_PLANT, kind="thunk", strength=1.0, pan=pan0, desc="FLAG PLANTED: pole slammed into rubble, deep wooden THUNK"),
        dict(t=T_PLANT + 0.02, kind="debris", strength=0.6, pan=pan0, desc="dust ring bursts out, pebbles scatter and patter down"),
        dict(t=142.02, kind="gust", strength=0.8, pan=0.2, desc="first dawn wind arrives; the cloth unrolls and cracks open"),
        dict(t=142.25, kind="snap", strength=0.9, pan=pan0 + 0.1, desc="cloth snaps fully open, catching the first sunlight"),
        dict(t=142.0, kind="sunrise", strength=0.6, pan=-0.5, desc="the sun breaks the horizon (shimmer / swell)"),
        dict(t=142.75, kind="step_rubble", strength=0.25, pan=pan0 - 0.25, desc="Crocus climbs up behind her"),
        dict(t=143.2, kind="step_rubble", strength=0.25, pan=pan0 - 0.25, desc="Crocus step"),
        dict(t=143.6, kind="step_rubble", strength=0.2, pan=pan0 - 0.25, desc="Crocus stops"),
        dict(t=147.55, kind="snap", strength=0.8, pan=pan0 + 0.1, desc="gust: the Flag cracks on the word 'Flag.'"),
        dict(t=147.4, kind="whoosh", strength=0.35, pan=0.0, desc="camera begins to crane up (air)"),
    ]
    # survivors turn and look up: a rustle spreading outward
    for q, tt in enumerate(np.arange(148.5, 149.4, 0.15)):
        ev.append(dict(t=float(tt), kind="rustle_crowd", strength=0.5 - 0.05 * q, pan=float(np.sin(q * 1.7) * 0.6),
                       desc="thousands of survivors lower their glowing screens and look up"))
    # convergence: the first individual raises (with pans from the actual render camera)
    order = wd.order
    act = wd.act
    first = []
    for i in order[:400]:                      # the first raises that are actually on screen
        pr = Proj(camera_at(float(act[i])))
        x, y, z = pr.pt(float(wd.x[i]), float(wd.y[i]), 2.0)
        if z > 1 and 0 < x < W and 0 < y < H:
            first.append(i)
        if len(first) >= 14:
            break
    for i in first:
        T = float(act[i])
        pr = Proj(camera_at(T))
        x, y, z = pr.pt(float(wd.x[i]), float(wd.y[i]), 2.0)
        near = float(np.clip(40.0 / max(z, 1.0), 0.15, 1.0))
        ev.append(dict(t=T, kind="flag_pop", strength=near, pan=foley.screen_pan(x),
                       desc="a survivor raises a plain yellow Flag (pop + cloth snap)"))
    foley.export_events(S, "hits", ev)

    # raise wave as a per-frame crowd track: flags raised per frame (in view), mean pan
    n = S.n
    energy = np.zeros(n, np.float32)
    pan = np.zeros(n, np.float32)
    cnt = np.zeros(n, np.float32)
    fin = np.isfinite(act)
    fa = np.floor((act[fin] - S.start) * S.fps).astype(int)
    idx = np.nonzero(fin)[0]
    for f in range(max(0, int((T_CONV - S.start) * S.fps) - 1), n):
        sel = idx[fa == f]
        if len(sel) == 0:
            continue
        T = S.start + f / S.fps
        pr = Proj(camera_at(T))
        P = np.stack([wd.x[sel], wd.y[sel], np.full(len(sel), 2.0)], 1).astype(float)
        sx, sy, z = pr(P)
        vis = (z > 1) & (sx > -100) & (sx < W + 100) & (sy > -100) & (sy < H + 100)
        wgt = np.clip(60.0 / np.maximum(z, 1.0), 0.02, 1.0) * vis
        cnt[f] = vis.sum()
        energy[f] = wgt.sum()
        if wgt.sum() > 0:
            pan[f] = float(np.clip(((sx - W / 2) / (W / 2) * wgt).sum() / wgt.sum(), -1, 1))
    if energy.max() > 0:
        energy = energy / energy.max()
    pev = [[float(act[i]), 1.0, 0.0] for i in first]
    foley.export_track(S, "raise", energy, pan, gain=1.0, kind="crowd", events=pev, count=cnt)
    # hero flag flutter
    tr = st["hero"]
    xs = []
    for f in range(n):
        T = S.start + f / S.fps
        pr = Proj(camera_at(T))
        x, _, z = pr.pt(*(CREST + np.array([0, 0, 1.8])))
        xs.append(foley.screen_pan(x))
    gain = 1.0
    tr.export_foley(S, "hero", pan=np.array(xs, np.float32), gain=gain)


# ============================================================== s07b hand-off
def export_handoff():
    """Write out/handoff/s07a_last.png (+ _post.png) of the last s07a frame and cache/s07a/handoff.npz
    (camera at 154.458 / 154.5 / 155.5, sun, wind, survivor & flag positions, plate lattice, palette)."""
    import cv2
    from types import SimpleNamespace
    from vx.config import OUT, CACHE
    from vx import post as vpost
    st = handoff_state()
    d = OUT / "handoff"
    d.mkdir(parents=True, exist_ok=True)
    T_last = 139.0 + 371 / 24.0
    fc = SimpleNamespace(s=1.0, w=W, h=H, F=3336 + 371, T=T_last, f=371)
    img = render_at(fc, T_last, st)
    cv2.imwrite(str(d / "s07a_last.png"), cv2.cvtColor(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    p = dict(POST)
    p.update(post(fc, st))
    fin = vpost.finish(img, fc, **p)
    cv2.imwrite(str(d / "s07a_last_post.png"), cv2.cvtColor(fin, cv2.COLOR_RGB2BGR))
    wd = st["world"]
    cams = {f"cam_{k}": camera_at(t) for k, t in (("last", T_last), ("1545", 154.5), ("1555", 155.5))}
    arr = {}
    for k, c in cams.items():
        arr[k] = np.array([*c["pos"], c["yaw"], c["pitch"], c["hfov"]], np.float64)
    np.savez_compressed(CACHE / "s07a" / "handoff.npz",
                        cam_fields=np.array(["x", "y", "z", "yaw_deg", "pitch_down_deg", "hfov_deg"]), **arr,
                        sun_dir_1545=sun_dir(154.5), sun_az_deg=WD.SUN_AZ, sun_elev_1545=WD.sun_elev(154.5),
                        wind_az_1545=wind_az(154.5),
                        survivor_x=wd.x, survivor_y=wd.y, survivor_height=wd.height, survivor_raise_t=wd.act,
                        survivor_kind=wd.kind, plate_cell=WD.CELL, plate_jitter=WD.JIT,
                        hill=np.array([0.0, 0.0, WD.HILL_R, WD.HILL_H]),
                        palette_plate_albedo=WD.PLATE_ALB, post=np.array([repr(p)]))
    return d

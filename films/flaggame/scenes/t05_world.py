"""t05 — the effigy world (113.10 -> 122.29): the Seraphlag carries the ALL YELLOW FLAG up the effigy, the
rain freezes, the clouds roll back and the FIRMAMENT is revealed resting on the pole tip.

Everything lives in scenes.effigy's 3D world (metres, y up). The hero flag is simulated in the vertical
"billboard plane" (plane px = PPM per metre, u = x * PPM, v = -y * PPM) and drawn through the camera's local
affine at the pole base, so it stays glued to the 3D world while the camera cranes and dollies.
"""
import math

import cairo
import numpy as np

from vx import ink, comic
from vx.canvas import Canvas
from vx.config import W, H
from vx.ease import clamp, lerp, smoothstep, smootherstep, ease_in_out, ease_out, ease_in, fbm1
from vx.flag import Flag, FlagTrack, draw_flag

from scenes import effigy as E

PPM = 100.0                     # plane px per metre for the hero flag + seraph
T_D0 = 113.10                   # shot start (cut from the map)
T_LIFT = 113.32                 # the Seraphlag's first downstroke
T_TOP = 116.35                  # arrives above the head
T_SEAT = 116.901                # flag_on_effigy
T_FREEZE = 118.917              # rain_freeze
T_REVEAL = 119.555              # firmament_reveal
T_END = 122.29167
SERAPH_H = 4.1                  # metres, a big angel (wingspan ~12 m)
FLAP_HZ = 1.25


def _ss(a, b, x):
    return smoothstep(a, b, x)


# ============================================================ camera path
_KEYS = [  # (T, pos, yaw, pitch, f)
    (T_D0, (3.3, 0.9, -11.5), -0.14, 0.36, 1050.0),
    (114.2, (3.1, 3.4, -11.2), -0.13, 0.28, 1050.0),
    (115.4, (2.5, 8.2, -10.4), -0.11, 0.22, 1080.0),
    (116.35, (1.7, 12.6, -9.0), -0.09, 0.2, 1120.0),
    (116.9, (1.4, 13.5, -8.2), -0.08, 0.18, 1150.0),
    (117.25, (1.4, 13.6, -8.4), -0.08, 0.2, 1150.0),
    (118.45, (0.55, 2.3, -31.2), -0.02, 0.262, 1300.0),
    (T_FREEZE, (0.6, 2.3, -30.9), -0.02, 0.26, 1300.0),
    (T_REVEAL, (-0.3, 2.45, -29.3), 0.016, 0.262, 1300.0),
    (121.35, (0.0, 1.7, -31.0), 0.0, 0.26, 1300.0),
    (T_END, (0.0, 1.7, -31.0), 0.0, 0.26, 1300.0),
]


def _catmull(p0, p1, p2, p3, u):
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


def cam_at(T):
    ks = _KEYS
    if T <= ks[0][0]:
        k = ks[0]
        return E.Cam3(k[1], k[2], k[3], 0.0, k[4])
    if T >= ks[-1][0]:
        return E.HANDOFF
    i = max(j for j in range(len(ks) - 1) if ks[j][0] <= T)
    a, b = ks[i], ks[i + 1]
    u = (T - a[0]) / (b[0] - a[0])
    # the big pull-back and the reveal move ease; the crane keeps momentum (catmull-rom)
    if i in (5, 8):
        u = ease_in_out(u, 2.2)
    vec = lambda k: np.array(list(k[1]) + [k[2], k[3], k[4]], np.float64)
    p0 = vec(ks[max(0, i - 1)]) if i not in (6, 9) else vec(a)
    p3 = vec(ks[min(len(ks) - 1, i + 2)]) if i not in (4, 7) else vec(b)
    if i in (5, 8):
        v = vec(a) + (vec(b) - vec(a)) * u
    else:
        v = _catmull(p0, vec(a), vec(b), p3, u)
    cam = E.Cam3(v[:3], v[3], v[4], 0.0, v[5])
    # impact shake on the seat
    if T_SEAT <= T < T_SEAT + 0.5:
        k = (1 - (T - T_SEAT) / 0.5) ** 2
        cam.__init__(cam.pos + np.array([0.035 * k * math.sin(T * 71), 0.05 * k * math.sin(T * 53 + 1), 0]),
                     cam.yaw, cam.pitch + 0.004 * k * math.sin(T * 61), 0.003 * k * math.sin(T * 47), cam.f)
    return cam


# ============================================================ the Seraphlag's flight + the pole
def pole_world(T):
    """pole base (x, y, z) m and angle (rad, + leans right) of the ALL YELLOW FLAG"""
    if T < T_LIFT:
        return np.array([3.55, 1.25, -2.6]), -0.5
    if T < T_TOP:
        u = (T - T_LIFT) / (T_TOP - T_LIFT)
        e = u * u * (3 - 2 * u) * 0.75 + u * 0.25
        # climbs in surges: each downstroke lifts
        surge = 0.22 * math.sin(2 * math.pi * FLAP_HZ * (T - T_LIFT)) * (1 - u)
        y = lerp(1.25, 13.15, e) + surge
        x = lerp(3.55, 2.05, e) + 0.12 * math.sin(T * 1.7)
        z = lerp(-2.6, -1.6, e)
        ang = -0.5 + 0.06 * math.sin(2 * math.pi * FLAP_HZ * (T - T_LIFT) + 1.0)
        return np.array([x, y, z]), ang
    if T < T_SEAT:
        u = (T - T_TOP) / (T_SEAT - T_TOP)
        # swing over the head and up-end the pole, then drive it down into the socket
        e = ease_in_out(min(1.0, u / 0.72), 2.5)
        x = lerp(2.05, 0.0, e)
        z = lerp(-1.6, 0.0, e)
        y = lerp(13.15, E.HEAD_TOP + 0.75, ease_out(min(1.0, u / 0.72), 2))
        if u > 0.72:
            v = (u - 0.72) / 0.28
            y = lerp(E.HEAD_TOP + 0.75, E.HEAD_TOP, ease_in(v, 2.6))
        ang = lerp(-0.5, 0.0, ease_in_out(min(1.0, u / 0.8), 2.2))
        return np.array([x, y, z]), ang
    # seated: the socket holds it; a tiny spring wobble after the impact
    dt = T - T_SEAT
    wob = 0.045 * math.exp(-dt * 5.0) * math.sin(dt * 26.0)
    return np.array([0.0, E.HEAD_TOP, 0.0]), wob


def seraph_state(T):
    """(x, y, z) of the Seraphlag's feet (m), flap phase, spread 0..1, grip 0..1 (hands on pole), alpha"""
    base, ang = pole_world(T)
    carry = np.array([0.55, -1.55, -0.45])
    place = np.array([1.75, -1.9, 0.9])
    k = _ss(T_TOP - 0.1, T_SEAT - 0.25, T)
    off = carry * (1 - k) + place * k
    if T < T_SEAT + 0.12:
        # it holds the pole in both hands, its body to the right of the pole
        feet = base + off
        grip = 1.0
    else:
        dt = T - (T_SEAT + 0.12)
        u = min(1.0, dt / 1.6)
        # lets go, rises away and backwards into the light
        feet = np.array([0.0, E.HEAD_TOP, 0.0]) + place + np.array([1.4 * ease_in(u, 2), 8.5 * ease_in(u, 2.2), 7.0 * ease_in(u, 2)])
        grip = max(0.0, 1.0 - dt / 0.25)
    if T < T_LIFT:
        ph = 0.62 + 0.38 * (T - T_D0) / (T_LIFT - T_D0)       # raising the wings for the first downstroke
    else:
        ph = (FLAP_HZ * (T - T_LIFT)) % 1.0
    spread = 1.0
    if T_TOP - 0.2 < T < T_SEAT + 0.15:
        # glides with wings held wide while it places the pole
        k = _ss(T_TOP - 0.2, T_TOP + 0.1, T) * (1 - _ss(T_SEAT, T_SEAT + 0.15, T))
        ph = ph * (1 - k) + 0.04 * k
    alpha = 1.0 - _ss(T_SEAT + 1.3, T_SEAT + 2.0, T)
    return feet, ph, spread, grip, alpha


def wing_beats(t0, t1):
    """times of the downstroke onsets (for the 'wing' foley)"""
    out = []
    k = 0
    while True:
        t = T_LIFT + k / FLAP_HZ
        if t > t1:
            break
        if t >= t0 and not (T_TOP - 0.1 < t < T_SEAT + 0.1):
            out.append(t)
        k += 1
    return out


# ============================================================ hero flag simulation
def plane_uv(p):
    return float(p[0] * PPM), float(-p[1] * PPM)


def path_key(fps=24):
    T0 = T_D0 - 0.25
    n = int(math.ceil((T_END - T0) * fps)) + 2
    out = []
    for i in range(0, n, 3):
        p, a = pole_world(T0 + i / fps)
        out += [p[0], p[1], a, _wind(T0 + i / fps)[0], _wind(T0 + i / fps)[1]]
    return out + [PPM, E.POLE_M]


def _wind(T):
    storm = 620.0 * (1 - _ss(T_FREEZE - 0.3, 120.4, T)) + 420.0 * _ss(T_FREEZE - 0.3, 120.4, T)
    # the seraph's wing downwash while it carries
    wash = 160.0 * (1 - _ss(T_SEAT, T_SEAT + 0.8, T)) * max(0.0, math.sin(2 * math.pi * FLAP_HZ * (T - T_LIFT)))
    return (storm, wash * 0.6, 40.0 * math.sin(T * 0.9))


def simulate_flag(fps=24):
    T0 = T_D0 - 0.25
    n = int(math.ceil((T_END - T0) * fps)) + 2

    def pole_fn(i):
        T = T0 + i / fps
        p, ang = pole_world(T)
        u, v = plane_uv(p)
        return (u, v, ang)

    def wind_fn(i):
        return _wind(T0 + i / fps)

    fl = Flag(pole=E.POLE_M * PPM, seed=505, side=1, turbulence=1.0)
    tr = fl.simulate(n, pole_fn, wind_fn, fps=fps, substeps=10, warmup=48)
    tr.meta["T0"] = T0
    tr.meta["plane"] = "billboard plane: u = x*PPM, v = -y*PPM, PPM=%g; draw with cam.plane_matrix(world_base, PPM)" % PPM
    return tr


def flag_index(tr, T):
    return (T - tr.meta["T0"]) * 24.0


def flag_matrix(cam, tr, T):
    """cairo matrix plane px -> design px, local to the current pole base"""
    base, _ = pole_world(T)
    M = cam.plane_matrix(base, PPM)
    u, v = plane_uv(base)
    M2 = cairo.Matrix(1, 0, 0, 1, -u, -v)
    return M2.multiply(M)


def cloth_screen_poly(cam, tr, T):
    """screen-space bbox of the cloth (+ margin) for lettering avoidance"""
    i = int(round(min(max(flag_index(tr, T), 0), tr.n - 1)))
    V = tr.verts[i][..., :2].reshape(-1, 2)
    M = flag_matrix(cam, tr, T)
    xs, ys = [], []
    for u, v in V[::3]:
        x, y = M.transform_point(float(u), float(v))
        xs.append(x)
        ys.append(y)
    return (min(xs) - 24, min(ys) - 24, max(xs) + 24, max(ys) + 24)


# ============================================================ rain
class Rain:
    """a 3D rain field: falls (streaks), freezes at T_FREEZE, falls away (accelerating) after T_REVEAL"""

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        n = 7500
        self.p0 = np.stack([rng.uniform(-32, 32, n), rng.uniform(0, 32, n), rng.uniform(-35, 22, n)], 1)
        # a denser swarm around the camera path of the bullet-time drift (the drops we fly through)
        m = 2600
        near = np.stack([rng.uniform(-5.5, 5.5, m), rng.uniform(0.0, 14.0, m), rng.uniform(-30.3, -19.0, m)], 1)
        # hero drops: placed in the view of the bullet-time camera, 0.9-6 m from the lens
        cam = cam_at(T_FREEZE)
        k = 110
        zc = 0.9 + 5.1 * rng.uniform(0, 1, k) ** 1.6
        sx = rng.uniform(-80, 2000, k)
        sy = rng.uniform(-60, 1140, k)
        D = cam.ray(sx, sy)
        dep = zc / (D @ cam.F)
        hero = cam.pos[None, :] + D * dep[:, None]
        self.n_far = n
        self.p0 = np.concatenate([self.p0, near, hero], 0)       # positions AT the freeze
        self.size = np.concatenate([rng.uniform(0.6, 1.4, n), rng.uniform(0.7, 1.6, m), rng.uniform(0.6, 1.1, k)])
        self.v = np.array([2.2, -9.5, 0.0])
        self.lo = np.concatenate([np.tile([[-32.0, 0.0]], (n, 1)), np.tile([[-5.5, 0.0]], (m, 1)),
                                  np.tile([[-1e6, -1e6]], (k, 1))])
        self.span = np.concatenate([np.tile([[64.0, 32.0]], (n, 1)), np.tile([[11.0, 14.0]], (m, 1)),
                                    np.tile([[2e6, 2e6]], (k, 1))])
        self.fade = rng.uniform(0, 1, len(self.p0))

    def positions(self, T):
        """world positions, streak vector (m), state alpha"""
        Tq = min(T, T_FREEZE)
        P = self.p0 + self.v[None, :] * (Tq - T_FREEZE)
        P[:, 0] = self.lo[:, 0] + np.mod(P[:, 0] - self.lo[:, 0], self.span[:, 0])
        P[:, 1] = self.lo[:, 1] + np.mod(P[:, 1] - self.lo[:, 1], self.span[:, 1])
        streak = np.broadcast_to(self.v * 0.055, P.shape).copy()
        alpha = np.ones(len(P))
        if T >= T_FREEZE:
            # the streak snaps shut over two frames
            k = clamp((T - T_FREEZE) / 0.09)
            streak *= (1 - k)
        if T >= T_REVEAL:
            dt = T - T_REVEAL
            fall = 0.5 * 38.0 * dt * dt * (0.7 + 0.6 * self.fade)
            P[:, 1] -= fall
            vy = 38.0 * dt * (0.7 + 0.6 * self.fade)
            streak = np.stack([np.zeros(len(P)), -vy * 0.04, np.zeros(len(P))], 1)
            alpha = np.clip(1.0 - (dt - 0.15 - 0.5 * self.fade) / 0.45, 0, 1)
        return P, streak, alpha

    def draw(self, ctx, cam, T, bg, fc, *, amount=1.0):
        """bg: the grey world under the rain (for contrast: white streaks on dark, dark ones on light).
        returns a list of frozen drops big enough to hold a refracted image [(x, y, r)]"""
        if amount <= 0:
            return []
        P, st, al = self.positions(T)
        sx, sy, z = cam.project(P)
        ex, ey, ez = cam.project(P - st)
        vis = (z > 0.35) & (ez > 0.35) & (sx > -60) & (sx < W + 60) & (sy > -60) & (sy < H + 60) & (al > 0.01)
        idx = np.nonzero(vis)[0]
        if len(idx) == 0:
            return []
        idx = idx[np.argsort(-z[idx])]
        h, w = bg.shape[:2]
        bx = np.clip((sx[idx] * fc.s).astype(int), 0, w - 1)
        by = np.clip((sy[idx] * fc.s).astype(int), 0, h - 1)
        tone = bg[by, bx]
        frozen = T >= T_FREEZE and (T < T_REVEAL + 0.05)
        k_fr = clamp((T - T_FREEZE) / 0.09) if T >= T_FREEZE else 0.0
        big = []
        ctx.save()
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for n, i in enumerate(idx):
            zz = z[i]
            ppm = cam.f / zz
            dark = tone[n] < 0.62
            c = 0.98 if dark else 0.14
            a = al[i] * amount * min(1.0, 0.25 + 3.0 / (zz ** 0.8))
            if not frozen or T >= T_REVEAL:
                wpx = min(3.2, max(0.55, 0.011 * self.size[i] * ppm))
                ctx.set_source_rgba(c, c, c, a)
                ctx.set_line_width(wpx)
                ctx.move_to(sx[i], sy[i])
                ctx.line_to(ex[i], ey[i])
                ctx.stroke()
            else:
                r = 0.024 * self.size[i] * ppm
                if r < 0.9:
                    ctx.set_source_rgba(c, c, c, a)
                    ctx.arc(sx[i], sy[i], max(0.45, r), 0, 2 * math.pi)
                    ctx.fill()
                    continue
                r = min(r, 70.0)
                x, y = sx[i], sy[i]
                # a drop hanging in the air: a clear lens (inverted world), dark rim, a hot specular
                if r > 24:
                    # too close to be in focus: a soft bokeh bubble with an engraved ring
                    g = cairo.RadialGradient(x, y, r * 0.2, x, y, r)
                    cc = 0.97 if dark else 0.62
                    g.add_color_stop_rgba(0, cc, cc, cc, 0.12 * a)
                    g.add_color_stop_rgba(0.85, cc, cc, cc, 0.4 * a)
                    g.add_color_stop_rgba(1.0, cc, cc, cc, 0.0)
                    ctx.set_source(g)
                    ctx.arc(x, y, r, 0, 2 * math.pi)
                    ctx.fill()
                    ctx.set_source_rgba(0.0, 0.0, 0.0, 0.35 * a)
                    ctx.set_line_width(1.2)
                    ctx.arc(x, y, r * 0.93, 0, 2 * math.pi)
                    ctx.stroke()
                    continue
                g = cairo.LinearGradient(x, y - r, x, y + r)
                g.add_color_stop_rgba(0, 0.25, 0.25, 0.25, a)      # the ground, upside down
                g.add_color_stop_rgba(0.5, 0.55, 0.55, 0.55, a)
                g.add_color_stop_rgba(1, 0.97, 0.97, 0.97, a)      # the sky, upside down
                ctx.set_source(g)
                ctx.arc(x, y, r, 0, 2 * math.pi)
                ctx.fill_preserve()
                ctx.set_source_rgba(0.0, 0.0, 0.0, a)
                ctx.set_line_width(max(0.6, r * 0.14))
                ctx.stroke()
                ctx.set_source_rgba(1, 1, 1, a)
                ctx.arc(x - r * 0.35, y - r * 0.38, max(0.6, r * 0.2), 0, 2 * math.pi)
                ctx.fill()
                if r >= 5.5:
                    big.append((x, y, r, a))
        ctx.restore()
        return big


# ============================================================ shot timing helpers
def reveal_r(T):
    if T < T_REVEAL:
        return 0.0
    u = T - T_REVEAL
    return 6.0 * (math.exp(3.3 * u) - 1.0)


def cloud_cover(T):
    return 1.0 - smoothstep(T_REVEAL + 1.9, T_REVEAL + 2.4, T)


def awe(T):
    """the crowd: arms go up when the flag seats, then again (and stay) at the reveal"""
    a = 0.55 * smoothstep(T_SEAT + 0.1, T_SEAT + 0.6, T) * (1 - smoothstep(117.9, 118.6, T))
    a += smoothstep(T_REVEAL + 0.25, T_REVEAL + 1.2, T)
    return min(1.0, a)


def storm_dark(T):
    """the storm is dark while the Seraphlag climbs (it is the light); it lifts once the Flag is set"""
    return 0.75 * (1 - smoothstep(T_SEAT - 0.2, 118.6, T))


def sky_glow(T):
    """light breaking behind the clouds towards the apex: it grows once the Flag is set"""
    return 0.12 + 0.45 * smoothstep(T_SEAT, 118.2, T) + 0.25 * smoothstep(118.6, T_FREEZE, T)


def effigy_tone(T):
    # lit timber in the close crane; a black lattice silhouette against the storm / the opened heaven
    return lerp(0.95, 0.32, smoothstep(117.2, 118.3, T))


_SERAPH = {}


def _seraph():
    if "c" not in _SERAPH:
        from vx.chars import Character
        _SERAPH["c"] = Character("seraph", seed=7, render="ink", wings=False, halo=False, face_mask=1.0)
    return _SERAPH["c"]


def draw_seraph(ctx, cam, T, layer):
    """the masked Seraphlag at its depth; vast wings + great halo behind; hands on the pole while it carries"""
    from scenes import t05_seraph as SR
    feet, ph, spread, grip, alpha = seraph_state(T)
    if alpha <= 0.01:
        return None
    base, ang = pole_world(T)
    sx, sy, z = cam.project(feet[None, :])
    if z[0] < 0.5:
        return None
    hpx = SERAPH_H * cam.f / z[0]
    bx, by, _ = cam.project(base[None, :])
    tip = base + np.array([math.sin(ang) * E.POLE_M, math.cos(ang) * E.POLE_M, 0.0])
    tx, ty, _ = cam.project(tip[None, :])

    def on(f):
        return (float(bx[0] + (tx[0] - bx[0]) * f), float(by[0] + (ty[0] - by[0]) * f))
    kw = dict(alpha=alpha, look=(-0.35, -0.5), mouth=0.0, expr="serene", wind=(0.5, 0.9))
    if grip > 0.5:
        kw.update(hand_r=on(0.47), hand_l=on(0.16), shape_r="grip", shape_l="grip", pole_ang=-ang, lean=0.22)
    ch = _seraph()
    x, y = float(sx[0]), float(sy[0])
    if layer == "back":
        a = ch.anchors(x, y, hpx, "fly", T, facing=-1, **kw)
        hx, hy = a["head"]
        SR.draw_halo(ctx, hx + 0.02 * hpx, hy - 0.01 * hpx, 0.16 * hpx, t=T, alpha=alpha)
        SR.draw_wings(ctx, a, hpx, ph, spread, alpha=alpha, facing=-1)
    return ch.draw(ctx, x, y, hpx, "fly", T, facing=-1, layer=layer, **kw)


def _impact(ctx, cam, dt):
    """flag_on_effigy: an engraved burst of light lines around the socket, and a ring of shock"""
    sx, sy = E.seat_screen(cam)
    s = cam.px_per_m(cam.cam_coords(np.array([E.SEAT]))[2][0])
    k = 1.0 - dt / 0.45
    rng = np.random.default_rng(3)
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for i in range(40):
        a = rng.uniform(0, 2 * math.pi)
        r0 = s * (0.35 + 1.4 * dt)
        r1 = r0 + s * rng.uniform(0.6, 1.8) * k
        ctx.move_to(sx + math.cos(a) * r0, sy + math.sin(a) * r0)
        ctx.line_to(sx + math.cos(a) * r1, sy + math.sin(a) * r1)
        c = 1.0 if i % 3 else 0.0
        ctx.set_source_rgba(c, c, c, k)
        ctx.set_line_width(max(1.0, s * 0.05 * k))
        ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, 0.8 * k)
    ctx.set_line_width(max(1.5, s * 0.08 * k))
    ctx.arc(sx, sy, s * (0.3 + 2.4 * dt), 0, 2 * math.pi)
    ctx.stroke()
    ctx.restore()


def render(fc, T, st):
    cam = cam_at(T)
    tr = st["hero"]
    rr = reveal_r(T)
    fir = T >= T_REVEAL - 0.02
    d = E.sky(fc, cam, T, cover=cloud_cover(T), reveal_r=rr, firmament=fir, res=0.5, glow=sky_glow(T),
              storm=storm_dark(T))
    base = d["water"]
    if fir:
        base = base * (1 - d["pane_a"]) + d["pane"] * d["pane_a"]
    base = base * d["hit"] + d["ground"] * (1 - d["hit"])
    world = Canvas(fc)
    world.paint_array(np.repeat(base[..., None], 3, 2))
    ctx = world.ctx
    if fir:
        E.draw_ribs(ctx, cam, reveal_r=rr, t=T)
    ca = d["cloud_a"][..., None]
    world.paint_array(np.concatenate([np.repeat(d["cloud"][..., None], 3, 2) * ca, ca], 2))
    behind = seraph_state(T)[0][2] > 0.3
    if behind:
        draw_seraph(ctx, cam, T, "back")
    E.draw_effigy(ctx, cam, tone=effigy_tone(T))
    aw = awe(T)
    E.draw_people(ctx, cam, st["people"], T, awe=aw, only="ring")
    if not behind:
        draw_seraph(ctx, cam, T, "back")
    if T_SEAT <= T < T_SEAT + 0.45:
        _impact(ctx, cam, T - T_SEAT)
    grey = world.rgb()
    big = st["rain"].draw(ctx, cam, T, grey[..., 0], fc)
    grey = world.rgb()
    img = ink.engrave(fc, grey)
    if T_FREEZE <= T < T_FREEZE + 0.25:
        # time stops: the print blanches for a blink (the flag, laid on after, is untouched)
        k = 0.4 * (1.0 - (T - T_FREEZE) / 0.25) ** 2
        img = img * (1 - k) + np.array(ink.PAPER, np.float32) * k
    # ---- lettering layout first (drop images must not sit under a caption)
    top = Canvas(fc)
    avoid = lambda Tq: [cloth_screen_poly(cam_at(Tq), tr, Tq)]
    b15 = comic.measure(top.ctx, fc, "P15", 56, 48, w=720, avoid=avoid, T=117.5)
    y16 = (b15[3] + 16) if b15 else 190
    boxes = [b15, (56, y16, 800, y16 + 140)]
    # ---- the spot plate: the ALL YELLOW FLAG (+ its tiny inverted images in the frozen drops)
    flags = Canvas(fc)
    fctx = flags.ctx
    fi = min(max(flag_index(tr, T), 0.0), tr.n - 1.0)
    fctx.save()
    fctx.transform(flag_matrix(cam, tr, T))
    glow = 0.35 * smoothstep(T_REVEAL, T_REVEAL + 0.8, T) + 0.25 * math.exp(-max(0.0, T - T_SEAT) * 3) * (T >= T_SEAT)
    tr.draw(fctx, fi, glow=glow)
    fctx.restore()
    if big:
        ftx, fty = tr.cloth_center(int(round(fi)))
        M = flag_matrix(cam, tr, T)
        fx, fy = M.transform_point(ftx, fty)
        for (x, y, r, a) in big:
            if any(bx and bx[0] - r < x < bx[2] + r and bx[1] - r < y < bx[3] + r for bx in boxes):
                continue
            dx, dy = fx - x, fy - y
            dn = math.hypot(dx, dy) + 1e-6
            ox, oy = -dx / dn * r * 0.42, -dy / dn * r * 0.42
            draw_flag(fctx, x + ox, y + oy + r * 0.28, pole=r * 0.9, t=T, wind=0.7, side=1 if dx < 0 else -1,
                      ang=math.pi, seed=int(x * 7) % 97, alpha=a)
    img = ink.spot(img, flags.rgba())
    # ---- front: the Seraphlag's hands on the pole; the watchers in the foreground
    front = Canvas(fc)
    draw_seraph(front.ctx, cam, T, "front")
    E.draw_people(front.ctx, cam, st["people"], T, awe=aw, only="fore", style=FORE_STYLE)
    fr = ink.print_layer(fc, front.rgba(), "engrave")
    img = fr[..., :3] + img * (1 - fr[..., 3:4])
    # ---- lettering
    comic.say(top.ctx, fc, "P15", 56, 48, w=720, avoid=avoid, hold=0.1)
    comic.say(top.ctx, fc, "P16", 56, y16, w=720, avoid=avoid)
    if T_SEAT <= T < T_SEAT + 0.7:
        sx, sy = E.seat_screen(cam)
        comic.sfx(top.ctx, "THUNK!", sx + 300, sy + 150, 130, rot=0.1, t=T - T_SEAT + 0.2, depth=10, avoid=avoid(T))
    return top.over(img)


FORE_STYLE = dict(render="ink")


def post(T):
    return dict(vignette=0.14 + 0.08 * smoothstep(T_REVEAL, T_REVEAL + 1.0, T))

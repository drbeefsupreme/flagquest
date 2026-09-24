"""THE EFFIGY and THE FIRMAMENT — shared composition for t05 (owner T05Worked) and t06 (T06Firmament).

World: metres, x right, y UP, z away from the camera. The effigy's axis is x = z = 0, the playa is y = 0.
Everything here draws GREY (tone) for the print (ink.engrave); flags are never drawn here.

Constants
---------
PLINTH_H   2.4 m lattice plinth; the 40-ft (12.19 m) figure stands on it.
HEAD_TOP   14.6 m: top of the pole socket on the head.   SEAT = (0, HEAD_TOP, 0): where the pole base seats.
POLE_M     4.0 m: the ALL YELLOW FLAG (a scaled canonical flag: cloth 0.375 x 0.3125 of the pole).
APEX_Y     HEAD_TOP + POLE_M = the pole tip = the firmament apex.   APEX = (0, APEX_Y, 0).
CANOPY_L   canopy fall-off: the crystal canopy height is h(r) = APEX_Y * exp(-r / CANOPY_L) (a tent whose king
           pole is the Flag), r = horizontal distance from the axis. N_RIBS meridian ribs, ring seams at RINGS (m).
HANDOFF    Cam3 of t05's last frame (122.29) == t06's first frame; HANDOFF_SKY = the sky() kwargs of that frame.
crowd(seed=0) / draw_people(ctx, cam, people, t, awe=, dusk=, style=, only='fore'|'ring'|None): the burners
           ringed round the effigy (t05 draws the 'fore' watchers on its front layer, awe=1 at the hand-off).

Camera
------
Cam3(pos=(x, y, z), yaw=0, pitch=0, roll=0, f=1300)   pinhole; yaw + = turn right, pitch + = look up,
    roll + = image turns clockwise. f in design px. Robust at any pitch/roll.
    .project(P (..., 3)) -> sx, sy, depth (design px; depth <= 0 is behind the camera)
    .px_per_m(depth) -> design px per metre at that depth
    .ray(sx, sy) -> unit world direction(s) through design pixel(s)
    .plane_matrix(origin, ppm=100.0) -> cairo.Matrix mapping a "billboard plane" (u right, v down, ppm px per
          metre, (0, 0) at world point `origin`, plane parallel to x-y) to the screen (local affine of the
          projection: use it for characters/flags that live near `origin`).
    .lerp(other, a) -> Cam3
Cam3.fit(x_base, y_base, height_px)  near-orthographic camera putting the effigy base at (x_base, y_base) with
    ground-to-apex height height_px (2D use).

Effigy
------
beams() -> dict(p0 (n,3), p1 (n,3), w (n,) m, part (n,) int, u (n,) 0..1 height fraction)
    parts: 0 plinth, 1 legs, 2 torso, 3 arms, 4 head, 5 socket, 6 hands
draw_effigy(ctx, cam, *, light=(-0.55, 0.7, -0.45), dusk=0.0, char=None, glow=None, gone=None, tone=1.0,
            mirror=False)
    grey lumber, depth sorted, per-beam shading. char (n,) 0..1 -> charcoal black, glow (n,) 0..1 -> white-hot,
    gone (n,) bool -> not drawn. tone scales the wood brightness (backlit storm < 1). mirror=True draws the
    puddle reflection (y -> -y).
seat_screen(cam) / apex_screen(cam) -> (sx, sy)

Firmament + sky (per-pixel, numba)
-------------------------------------
canopy_h(r) ; canopy_point(r, th) -> (x, y, z) ; RINGS, N_RIBS
sky(fc, cam, t, *, cover=1.0, reveal_r=0.0, dusk=0.0, res=0.5, firmament=True, glow=0.0, storm=0.0, angles=False)
    -> dict of float32 (fc.h, fc.w). glow: light breaking behind the clouds towards the apex; storm 0..1 darkens them:
    hit    1 where the view ray meets the sky (above the horizon), 0 on the ground
    water  tone of the waters above seen through the crystal (the "sky" behind the panes)
    pane   tone of the crystal panes, pane_a their coverage (0 where firmament=False)
    cloud  tone of the storm clouds, cloud_a their coverage (cover=0 -> clear; reveal_r = radius (m, on the
           canopy) of the hole the clouds have rolled back from around the apex)
    ground tone of the wet playa (mud, puddles reflecting the sky)
    r, th  canopy coordinates of each pixel; cx, cz the canopy hit (m) and mpp metres per design px there
    angle  (angles=True) suggested engraving LINE direction map (radians)
compose_sky(d) -> (h, w) tone: ground | water < panes < clouds
draw_ribs(ctx, cam, *, alpha=1.0, dusk=0.0, reveal_r=1e9, t=0.0, boss=1.0) vector ribs + ring seams + the apex boss
    (boss 0..1 scales/fades the starburst where the crystal rests on the pole tip)
facets(cam, rmax=900) -> list of (poly (k,2) screen, (r0, r1, th0, th1)) visible canopy panes (clipped)
"""
import math

import cairo
import cv2
import numpy as np
from numba import njit

from vx.config import W, H

PLINTH_H = 2.4
FIG_H = 12.19
HEAD_TOP = 14.6
SEAT = (0.0, HEAD_TOP, 0.0)
POLE_M = 4.0
APEX_Y = HEAD_TOP + POLE_M
APEX = (0.0, APEX_Y, 0.0)
CANOPY_L = 150.0
N_RIBS = 24
RIB_OFF = 0.5 * 2 * math.pi / N_RIBS         # no rib points straight at the camera
RINGS = (1.2, 3.0, 6.0, 10.0, 16.0, 25.0, 38.0, 56.0, 80.0, 115.0, 165.0, 240.0, 350.0, 520.0, 800.0, 1300.0)
CLOUD_H = 420.0


# ============================================================ camera
class Cam3:
    def __init__(self, pos=(0.0, 1.7, -34.0), yaw=0.0, pitch=0.0, roll=0.0, f=1300.0):
        self.pos = np.asarray(pos, np.float64).reshape(3)
        self.yaw, self.pitch, self.roll, self.f = float(yaw), float(pitch), float(roll), float(f)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        fwd0 = np.array([sy, 0.0, cy])
        right = np.array([cy, 0.0, -sy])
        fwd = fwd0 * cp + np.array([0.0, 1.0, 0.0]) * sp
        up = np.array([0.0, 1.0, 0.0]) * cp - fwd0 * sp
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        self.R = right * cr - up * sr
        self.U = up * cr + right * sr
        self.F = fwd

    def __repr__(self):
        p = self.pos
        return (f"Cam3(pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), yaw={self.yaw:.4f}, pitch={self.pitch:.4f}, "
                f"roll={self.roll:.4f}, f={self.f:.1f})")

    def cam_coords(self, P):
        d = np.asarray(P, np.float64) - self.pos
        return d @ self.R, d @ self.U, d @ self.F

    def project(self, P):
        xc, yc, zc = self.cam_coords(P)
        zz = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
        return W / 2 + self.f * xc / zz, H / 2 - self.f * yc / zz, zc

    def px_per_m(self, depth):
        return self.f / np.maximum(depth, 1e-3)

    def ray(self, sx, sy):
        sx = np.asarray(sx, np.float64)[..., None]
        sy = np.asarray(sy, np.float64)[..., None]
        d = self.R * (sx - W / 2) / self.f + self.U * (H / 2 - sy) / self.f + self.F
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def plane_matrix(self, origin, ppm=100.0):
        o = np.asarray(origin, np.float64)
        e = 0.25
        pts = np.array([o, o + [e, 0, 0], o + [0, e, 0]])
        sx, sy, _ = self.project(pts)
        ox, oy = sx[0], sy[0]
        ux, uy = (sx[1] - ox) / (e * ppm), (sy[1] - oy) / (e * ppm)
        vx_, vy_ = -(sx[2] - ox) / (e * ppm), -(sy[2] - oy) / (e * ppm)
        return cairo.Matrix(ux, uy, vx_, vy_, ox, oy)

    def lerp(self, other, a):
        a = float(a)
        return Cam3(self.pos + (other.pos - self.pos) * a, self.yaw + (other.yaw - self.yaw) * a,
                    self.pitch + (other.pitch - self.pitch) * a, self.roll + (other.roll - self.roll) * a,
                    self.f + (other.f - self.f) * a)

    @staticmethod
    def fit(x_base, y_base, height_px, dist=4000.0):
        """near-orthographic camera: effigy base at (x_base, y_base), ground..apex = height_px"""
        f = height_px * dist / APEX_Y
        cx = -(x_base - W / 2) * dist / f
        cy = (y_base - H / 2) * dist / f
        return Cam3((cx, cy, -dist), 0.0, 0.0, 0.0, f)


# the t05 -> t06 hand-off camera (t05's last frame at 122.29, t06's first frame) and the sky() args of that frame
HANDOFF = Cam3(pos=(0.0, 1.7, -31.0), yaw=0.0, pitch=0.26, roll=0.0, f=1300.0)
HANDOFF_SKY = dict(t=122.29, cover=0.0, reveal_r=1e6, dusk=0.0, res=0.5, firmament=True, glow=0.0, storm=0.0)


# ============================================================ effigy model
_BEAMS = {}


def _perp(t):
    """two unit vectors perpendicular to axis t: u in the x-y plane (width), v ~ z (depth)"""
    z = np.array([0.0, 0.0, 1.0])
    u = np.cross(t, z)
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(u, t)
    v /= np.linalg.norm(v)
    return u, v


def _truss(out, a, b, wa, wb, da, db, n, part, lw=(0.14, 0.09, 0.07), xbrace=True):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = b - a
    t /= np.linalg.norm(t)
    u, v = _perp(t)

    def corners(k):
        s = k / n
        c = a + (b - a) * s
        w = (wa + (wb - wa) * s) / 2
        d = (da + (db - da) * s) / 2
        return [c + u * w + v * d, c - u * w + v * d, c - u * w - v * d, c + u * w - v * d]

    rings = [corners(k) for k in range(n + 1)]
    for c in range(4):
        for k in range(n):
            out.append((rings[k][c], rings[k + 1][c], lw[0], part))
    for k in range(n + 1):
        for c in range(4):
            out.append((rings[k][c], rings[k][(c + 1) % 4], lw[1], part))
    for k in range(n):
        for c in range(4):
            c2 = (c + 1) % 4
            out.append((rings[k][c], rings[k + 1][c2], lw[2], part))
            if xbrace:
                out.append((rings[k][c2], rings[k + 1][c], lw[2], part))
    return rings


def beams():
    """the effigy as lumber: dict(p0, p1, w, part, u)"""
    if "b" in _BEAMS:
        return _BEAMS["b"]
    B = []
    P = PLINTH_H
    # plinth: truncated square pyramid, two storeys, X-braced, plank deck, a ladder up the front
    lv = [(0.0, 4.4), (1.3, 3.25), (P, 2.3)]
    sq = [[np.array([sx * hw, y, sz * hw]) for sx, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))] for y, hw in lv]
    for k in range(len(lv)):
        for c in range(4):
            B.append((sq[k][c], sq[k][(c + 1) % 4], 0.22, 0))
            if k + 1 < len(lv):
                B.append((sq[k][c], sq[k + 1][c], 0.26, 0))
                B.append((sq[k][c], sq[k + 1][(c + 1) % 4], 0.13, 0))
                B.append((sq[k][(c + 1) % 4], sq[k + 1][c], 0.13, 0))
    for i in range(11):
        x = -2.2 + 4.4 * i / 10
        B.append((np.array([x, P + 0.05, -2.25]), np.array([x, P + 0.05, 2.25]), 0.17, 0))
    for sgn in (-1, 1):
        B.append((np.array([sgn * 0.45, 0.0, -4.6]), np.array([sgn * 0.45, P, -2.35]), 0.1, 0))
    for i in range(1, 7):
        s = i / 7
        B.append((np.array([-0.45, P * s, -4.6 + 2.25 * s]), np.array([0.45, P * s, -4.6 + 2.25 * s]), 0.07, 0))
    # legs
    for sgn in (-1, 1):
        _truss(B, (sgn * 1.3, P + 0.1, 0.0), (sgn * 0.78, P + 5.0, 0.0), 1.15, 0.9, 0.95, 0.85, 6, 1)
        B.append((np.array([sgn * 1.3 - 0.8, P + 0.12, 0.0]), np.array([sgn * 1.3 + 0.8, P + 0.12, 0.0]), 0.22, 1))
    # pelvis beam + torso (inverted trapezoid: a broad chest)
    B.append((np.array([-1.3, P + 5.0, 0.0]), np.array([1.3, P + 5.0, 0.0]), 0.22, 2))
    _truss(B, (0.0, P + 4.95, 0.0), (0.0, P + 8.9, 0.0), 2.1, 3.9, 1.1, 1.3, 5, 2, lw=(0.16, 0.1, 0.08))
    # arms raised in a V (upper arm, forearm), splayed hands
    for sgn in (-1, 1):
        sh = np.array([sgn * 1.85, P + 8.55, 0.0])
        el = np.array([sgn * 3.35, P + 9.95, 0.0])
        ha = np.array([sgn * 4.05, P + 12.05, 0.0])
        _truss(B, sh, el, 0.75, 0.66, 0.7, 0.62, 3, 3)
        _truss(B, el, ha, 0.66, 0.55, 0.62, 0.52, 3, 3)
        for k, ang in enumerate((-0.6, -0.2, 0.2, 0.6)):
            dvec = np.array([sgn * math.sin(ang + sgn * 0.3), math.cos(ang + sgn * 0.3), 0.0])
            B.append((ha, ha + dvec * (0.95 if k in (1, 2) else 0.7), 0.11, 6))
    # neck
    _truss(B, (0.0, P + 8.9, 0.0), (0.0, P + 9.3, 0.0), 0.7, 0.65, 0.7, 0.65, 1, 4)
    # head: lattice lantern (elongated octahedron with two rings)
    bot = np.array([0.0, P + 9.25, 0.0])
    top = np.array([0.0, HEAD_TOP - 0.25, 0.0])
    rA = [np.array([1.25 * math.cos(a), P + 10.35, 1.25 * math.sin(a)]) for a in np.arange(8) * math.pi / 4 + math.pi / 8]
    rB = [np.array([0.95 * math.cos(a), P + 11.25, 0.95 * math.sin(a)]) for a in np.arange(8) * math.pi / 4 + math.pi / 8]
    for i in range(8):
        B.append((bot, rA[i], 0.12, 4))
        B.append((rA[i], rB[i], 0.12, 4))
        B.append((rB[i], top, 0.12, 4))
        B.append((rA[i], rA[(i + 1) % 8], 0.1, 4))
        B.append((rB[i], rB[(i + 1) % 8], 0.1, 4))
        B.append((rA[i], rB[(i + 1) % 8], 0.06, 4))
        B.append((rB[i], rA[(i + 1) % 8], 0.06, 4))
    # the socket the Flag's pole seats in
    B.append((top - [0, 0.1, 0], np.array(SEAT), 0.2, 5))
    p0 = np.array([b[0] for b in B], np.float64)
    p1 = np.array([b[1] for b in B], np.float64)
    w = np.array([b[2] for b in B], np.float64)
    part = np.array([b[3] for b in B], np.int32)
    u = np.clip(0.5 * (p0[:, 1] + p1[:, 1]) / HEAD_TOP, 0, 1)
    _BEAMS["b"] = dict(p0=p0, p1=p1, w=w, part=part, u=u)
    return _BEAMS["b"]


def _clip_seg(cam, p0, p1, znear=0.3):
    """clip world segments to the camera near plane; returns (q0, q1, keep)"""
    _, _, z0 = cam.cam_coords(p0)
    _, _, z1 = cam.cam_coords(p1)
    keep = (z0 > znear) | (z1 > znear)
    t = np.clip((znear - z0) / np.where(np.abs(z1 - z0) < 1e-9, 1e-9, z1 - z0), 0, 1)
    q0 = np.where((z0 < znear)[:, None], p0 + (p1 - p0) * t[:, None], p0)
    t2 = np.clip((znear - z1) / np.where(np.abs(z0 - z1) < 1e-9, 1e-9, z0 - z1), 0, 1)
    q1 = np.where((z1 < znear)[:, None], p1 + (p0 - p1) * t2[:, None], p1)
    return q0, q1, keep


def draw_effigy(ctx, cam, *, light=(-0.55, 0.7, -0.45), dusk=0.0, char=None, glow=None, gone=None, tone=1.0,
                mirror=False, alpha=1.0):
    b = beams()
    p0, p1, w = b["p0"].copy(), b["p1"].copy(), b["w"]
    if mirror:
        p0[:, 1] *= -1
        p1[:, 1] *= -1
    n = len(w)
    q0, q1, keep = _clip_seg(cam, p0, p1)
    if gone is not None:
        keep &= ~np.asarray(gone, bool)
    x0, y0, z0 = cam.project(q0)
    x1, y1, z1 = cam.project(q1)
    zm = 0.5 * (z0 + z1)
    d = p1 - p0
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    # diffuse of a square beam: the visible faces' light; beams parallel to the light get little
    lam = np.sqrt(np.clip(1 - (d @ L) ** 2, 0, 1))
    k = np.clip(float(tone), 0, 1.5) * (1 - 0.55 * dusk)
    lit = np.clip((0.42 + 0.46 * lam) * k, 0, 1)
    sh = np.clip((0.16 + 0.14 * lam) * k, 0, 1)
    if char is not None:
        c = np.clip(np.asarray(char, float), 0, 1)
        lit = lit * (1 - c) + 0.06 * c
        sh = sh * (1 - c) + 0.02 * c
    gl = np.zeros(n) if glow is None else np.clip(np.asarray(glow, float), 0, 1)
    ww = w * cam.f / np.maximum(z0, 0.05), w * cam.f / np.maximum(z1, 0.05)
    order = np.argsort(-zm)
    # screen direction of the light for the lit side of each beam
    lx, ly = L[0], -L[1]
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    ctx.set_line_join(cairo.LINE_JOIN_MITER)
    for i in order:
        if not keep[i]:
            continue
        ax, ay, bx, by = x0[i], y0[i], x1[i], y1[i]
        dx, dy = bx - ax, by - ay
        ln = math.hypot(dx, dy)
        if ln < 0.05:
            continue
        nx, ny = -dy / ln, dx / ln
        if nx * lx + ny * ly < 0:
            nx, ny = -nx, -ny
        wa, wb = ww[0][i] * 0.5, ww[1][i] * 0.5
        if wa > 1e4 or wb > 1e4:
            continue
        ex, ey = dx / ln * 0.35 * wa, dy / ln * 0.35 * wa
        g = gl[i]
        cl = lit[i] * (1 - g) + g
        cs = sh[i] * (1 - g) + 0.8 * g
        if max(wa, wb) < 1.2:
            ctx.set_source_rgba(0.5 * (cl + cs), 0.5 * (cl + cs), 0.5 * (cl + cs), alpha)
            ctx.set_line_width(max(0.5, wa + wb))
            ctx.move_to(ax - ex, ay - ey)
            ctx.line_to(bx + ex, by + ey)
            ctx.stroke()
            continue
        # whole beam in shadow tone, lit half on the light side, dark contour
        ctx.move_to(ax + nx * wa - ex, ay + ny * wa - ey)
        ctx.line_to(bx + nx * wb + ex, by + ny * wb + ey)
        ctx.line_to(bx - nx * wb + ex, by - ny * wb + ey)
        ctx.line_to(ax - nx * wa - ex, ay - ny * wa - ey)
        ctx.close_path()
        ctx.set_source_rgba(cs, cs, cs, alpha)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.04, 0.04, 0.04, alpha * min(1.0, 0.4 + 0.2 * (wa + wb)) * (1 - g))
        ctx.set_line_width(min(2.2, 0.35 + 0.12 * (wa + wb)))
        ctx.stroke()
        ctx.move_to(ax + nx * wa * 0.8 - ex, ay + ny * wa * 0.8 - ey)
        ctx.line_to(bx + nx * wb * 0.8 + ex, by + ny * wb * 0.8 + ey)
        ctx.line_to(bx + nx * wb * 0.05 + ex, by + ny * wb * 0.05 + ey)
        ctx.line_to(ax + nx * wa * 0.05 - ex, ay + ny * wa * 0.05 - ey)
        ctx.close_path()
        ctx.set_source_rgba(cl, cl, cl, alpha)
        ctx.fill()
    ctx.restore()


def seat_screen(cam):
    x, y, _ = cam.project(np.array([SEAT]))
    return float(x[0]), float(y[0])


def apex_screen(cam):
    x, y, _ = cam.project(np.array([APEX]))
    return float(x[0]), float(y[0])


# ============================================================ the crowd around the effigy
def crowd(seed=0, n_ring=120, n_fore=9):
    """the burners ringed around the effigy + a row of watchers seen from behind near the HANDOFF camera.
    list of dict(x, z (m), h (m), seed, fore, t_off, kind)"""
    rng = np.random.default_rng(seed + 4242)
    people = []
    for i in range(n_ring):
        for _ in range(20):
            ang = rng.uniform(0, 2 * math.pi)
            r = 8.0 + min(24.0, rng.gamma(2.2, 3.6))
            x, z = r * math.cos(ang), r * math.sin(ang)
            if abs(x) < 5.2 and -6.0 < z < 5.2:        # the plinth footprint and its ladder
                continue
            break
        people.append(dict(x=x, z=z, h=float(rng.uniform(1.62, 1.92)), seed=int(rng.integers(1, 10 ** 6)),
                           fore=False, t_off=float(rng.uniform(0, 7)), kind="hippie"))
    xs = np.linspace(-6.2, 6.2, n_fore) + rng.uniform(-0.45, 0.45, n_fore)
    for i, x in enumerate(xs):
        if abs(x) < 0.9:                                # keep the sight line to the effigy base clear
            x = 0.9 * (1 if x >= 0 else -1) + x * 0.3
        people.append(dict(x=float(x), z=float(-26.4 + rng.uniform(-0.7, 0.9)), h=float(rng.uniform(1.66, 1.9)),
                           seed=int(rng.integers(1, 10 ** 6)), fore=True, t_off=float(rng.uniform(0, 7)),
                           kind="hippie"))
    return people


def draw_people(ctx, cam, people, t, *, awe=0.0, dusk=0.0, tone=1.0, style=None, only=None):
    """draw `people` (from crowd()) as billboards; they face the effigy. awe 0..1: heads up, arms raised.
    only = "fore" | "ring" | None. Returns [(bbox, depth)] of what was drawn."""
    from vx.chars import draw_crowd
    items = []
    for p in people:
        if only == "fore" and not p["fore"] or only == "ring" and p["fore"]:
            continue
        sx, sy, z = cam.project(np.array([[p["x"], 0.0, p["z"]]]))
        z = float(z[0])
        if z < 0.8:
            continue
        hpx = p["h"] * cam.f / z
        sx, sy = float(sx[0]), float(sy[0])
        if sx < -hpx or sx > W + hpx or sy - hpx * 1.4 > H or sy < -hpx * 0.2:
            continue
        # facing the effigy: towards the axis on screen; back view when between camera and effigy
        ex, ey, _ = cam.project(np.array([[0.0, 0.0, 0.0]]))
        dxs = float(ex[0]) - sx
        facing = 1 if dxs > 0 else -1
        toward_cam = p["z"] > 1.0
        a = min(1.0, max(0.0, awe * (1.25 - 0.25 * math.sin(p["seed"] % 17))))
        ph = (p["seed"] % 1000) / 1000.0
        if a > 0.5:
            pose = {"stand": 1 - a, "cheer": a}
        else:
            pose = {"stand": 1 - a * 0.6, "cheer": a * 0.6} if a > 0.05 else "stand"
        d = dict(x=sx, y=sy, h=hpx, kind=p.get("kind", "hippie"), seed=p["seed"], pose=pose, facing=facing,
                 look=(0.0, -0.4 - 0.6 * a), t_off=p["t_off"] + ph, turn=0.2 if toward_cam else 2.0,
                 mouth=0.35 * a * (0.5 + 0.5 * math.sin(7 * t + p["seed"])))
        if style:
            d.update(style)
        items.append((z, d))
    items.sort(key=lambda it: -it[0])
    if not items:
        return []
    draw_crowd(ctx, [d for _, d in items], t)
    return [((d["x"] - d["h"] * 0.4, d["y"] - d["h"] * 1.3, d["x"] + d["h"] * 0.4, d["y"]), z) for z, d in items]


# ============================================================ canopy geometry
def canopy_h(r):
    return APEX_Y * np.exp(-np.asarray(r, np.float64) / CANOPY_L)


def canopy_point(r, th):
    r = np.asarray(r, np.float64)
    th = np.asarray(th, np.float64)
    return np.stack([r * np.cos(th), canopy_h(r), r * np.sin(th)], axis=-1)


def _clip_poly_near(Pc, znear=0.4):
    """Sutherland-Hodgman against z > znear in camera coords; Pc (k,3)"""
    out = []
    k = len(Pc)
    for i in range(k):
        a, b = Pc[i], Pc[(i + 1) % k]
        ia, ib = a[2] > znear, b[2] > znear
        if ia:
            out.append(a)
        if ia != ib:
            t = (znear - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * t)
    return np.array(out) if out else np.zeros((0, 3))


def facets(cam, rmax=900.0, nsub=4):
    """visible canopy panes as screen polygons (clipped at the near plane)"""
    res = []
    rings = (0.0,) + tuple(r for r in RINGS if r <= rmax)
    dth = 2 * math.pi / N_RIBS
    for k in range(len(rings) - 1):
        r0, r1 = rings[k], rings[k + 1]
        for j in range(N_RIBS):
            th0 = RIB_OFF + j * dth
            th1 = th0 + dth
            ths = np.linspace(th0, th1, nsub + 1)
            pts = [canopy_point(r0, th) for th in (ths if r0 > 0 else ths[:1])]
            pts += [canopy_point(r1, th) for th in ths[::-1]]
            P = np.array(pts)
            xc, yc, zc = cam.cam_coords(P)
            Pc = _clip_poly_near(np.stack([xc, yc, zc], 1))
            if len(Pc) < 3:
                continue
            sx = W / 2 + cam.f * Pc[:, 0] / Pc[:, 2]
            sy = H / 2 - cam.f * Pc[:, 1] / Pc[:, 2]
            if sx.max() < -200 or sx.min() > W + 200 or sy.max() < -200 or sy.min() > H + 200:
                continue
            res.append((np.stack([sx, sy], 1), (r0, r1, th0, th1)))
    return res


def _polyline_world(ctx, cam, P, wm, col, alpha, lw_min=0.4, lw_max=40.0):
    """stroke a world polyline with perspective width wm (m); per-segment widths"""
    xc, yc, zc = cam.cam_coords(P)
    for i in range(len(P) - 1):
        za, zb = zc[i], zc[i + 1]
        if za < 0.4 and zb < 0.4:
            continue
        a = np.array([xc[i], yc[i], za])
        b = np.array([xc[i + 1], yc[i + 1], zb])
        if za < 0.4:
            a = a + (b - a) * (0.4 - za) / (zb - za)
        if zb < 0.4:
            b = b + (a - b) * (0.4 - zb) / (za - zb)
        ax, ay = W / 2 + cam.f * a[0] / a[2], H / 2 - cam.f * a[1] / a[2]
        bx, by = W / 2 + cam.f * b[0] / b[2], H / 2 - cam.f * b[1] / b[2]
        if max(ax, bx) < -50 or min(ax, bx) > W + 50 or max(ay, by) < -50 or min(ay, by) > H + 50:
            continue
        lw = min(lw_max, max(lw_min, wm * cam.f / (0.5 * (a[2] + b[2]))))
        ctx.set_source_rgba(col, col, col, alpha * min(1.0, lw / lw_min) if lw_min > 0 else alpha)
        ctx.set_line_width(lw)
        ctx.move_to(ax, ay)
        ctx.line_to(bx, by)
        ctx.stroke()


def draw_ribs(ctx, cam, *, alpha=1.0, dusk=0.0, reveal_r=1e9, t=0.0, rmax=1300.0, boss=1.0):
    """the crystal's leading: meridian ribs, ring seams (dark with a bright specular core), the apex boss"""
    if alpha <= 0:
        return
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    rr = np.concatenate([np.linspace(0.0, 4.0, 9), np.geomspace(4.5, min(rmax, reveal_r * 1.15 + 5), 60)])
    rr = rr[rr <= reveal_r * 1.15 + 1e-6] if reveal_r < 1e8 else rr
    if len(rr) >= 2:
        for j in range(N_RIBS):
            th = RIB_OFF + j * 2 * math.pi / N_RIBS
            P = canopy_point(rr, np.full_like(rr, th))
            _polyline_world(ctx, cam, P, 0.6, 0.0, alpha, lw_min=1.0, lw_max=7.0)
            _polyline_world(ctx, cam, P, 0.2, 1.0 - 0.2 * dusk, alpha, lw_min=0.0, lw_max=3.0)
    ths = np.linspace(0, 2 * math.pi, 145)
    for r in RINGS:
        if r > min(rmax, 520.0) or r > reveal_r:
            continue
        P = canopy_point(np.full_like(ths, r), ths)
        _polyline_world(ctx, cam, P, 0.4, 0.0, alpha * 0.9, lw_min=0.8, lw_max=5.0)
        _polyline_world(ctx, cam, P, 0.12, 1.0 - 0.2 * dusk, alpha * 0.9, lw_min=0.0, lw_max=2.0)
    # apex boss: where the firmament rests on the pole tip
    ax, ay, az = cam.project(np.array([APEX]))
    if az[0] > 0.5 and boss > 0:
        s = cam.px_per_m(az[0]) * (0.4 + 0.6 * boss)
        x, y = float(ax[0]), float(ay[0])
        alpha = alpha * min(1.0, boss)
        rad = cairo.RadialGradient(x, y, 0, x, y, 1.4 * s)
        rad.add_color_stop_rgba(0, 1, 1, 1, alpha)
        rad.add_color_stop_rgba(0.35, 0.95, 0.95, 0.95, alpha * 0.8)
        rad.add_color_stop_rgba(1, 0.9, 0.9, 0.9, 0)
        ctx.set_source(rad)
        ctx.arc(x, y, 1.4 * s, 0, 2 * math.pi)
        ctx.fill()
        # a star of engraved light where the crystal rests on the pole tip
        for i in range(32):
            a = i * math.pi / 16 + 0.15 * math.sin(t * 0.7)
            ln = s * (5.5 if i % 4 == 0 else 3.4 if i % 2 == 0 else 2.2) * (1 + 0.08 * math.sin(t * 3 + i))
            w0 = max(0.8, (0.14 if i % 4 == 0 else 0.08) * s)
            ctx.move_to(x + math.cos(a + 0.5 * math.pi) * w0, y + math.sin(a + 0.5 * math.pi) * w0)
            ctx.line_to(x + math.cos(a) * ln, y + math.sin(a) * ln)
            ctx.line_to(x - math.cos(a + 0.5 * math.pi) * w0, y - math.sin(a + 0.5 * math.pi) * w0)
            ctx.close_path()
            ctx.set_source_rgba(1, 1, 1, alpha * 0.95)
            ctx.fill()
    ctx.restore()


# ============================================================ per-pixel sky (numba)
@njit(cache=True, inline="always")
def _h2(ix, iy, seed):
    n = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 32767.5 - 1.0


@njit(cache=True)
def _vn(x, y, seed):
    xi = math.floor(x)
    yi = math.floor(y)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3 - 2 * xf)
    v = yf * yf * (3 - 2 * yf)
    ix = np.int64(xi)
    iy = np.int64(yi)
    a = _h2(ix, iy, seed)
    b = _h2(ix + 1, iy, seed)
    c = _h2(ix, iy + 1, seed)
    d = _h2(ix + 1, iy + 1, seed)
    return (a + (b - a) * u) * (1 - v) + (c + (d - c) * u) * v


@njit(cache=True)
def _fbm(x, y, seed, oct):
    s = 0.0
    a = 1.0
    nrm = 0.0
    for o in range(oct):
        s += a * _vn(x, y, seed + o * 31)
        nrm += a
        a *= 0.5
        x = x * 2.03 + 17.1
        y = y * 2.03 - 5.3
    return s / nrm


@njit(cache=True)
def _hash1(k, j):
    n = (k * 73856093 + j * 19349663 + 7) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


@njit(cache=True)
def _cloud(cx, cz, t, seed):
    """storm cloud density + light at cloud-plane coords (m)"""
    u = cx / 520.0 + t * 0.012
    v = cz / 520.0
    wx = _fbm(u * 0.7, v * 0.7, seed + 3, 3)
    wy = _fbm(u * 0.7 + 3.1, v * 0.7 - 1.7, seed + 5, 3)
    u2 = u + 0.55 * wx
    v2 = v + 0.55 * wy
    n = _fbm(u2, v2, seed, 5)
    # light from up-left on the billows: difference toward the light
    nl = _fbm(u2 - 0.035, v2 + 0.05, seed, 5)
    return n, n - nl


@njit(cache=True)
def _sky_kernel(ww, hh, sc, pos, R, U, F, f, A, L, t, reveal_r, cover, dusk, fir, gdx, gdy, gdz, gstr, gax, gaz,
                roll, storm, hit, rr, cxo, czo, mppo, water, pane, pane_a, cloud, cloud_a, ground, gang):
    TWO_PI = 2.0 * math.pi
    dth = TWO_PI / 24.0
    lq = math.log(1.55)
    for i in range(hh):
        sy = (i + 0.5) / sc
        for j in range(ww):
            sx = (j + 0.5) / sc
            px = (sx - 960.0) / f
            py = (540.0 - sy) / f
            dx = R[0] * px + U[0] * py + F[0]
            dy = R[1] * px + U[1] * py + F[1]
            dz = R[2] * px + U[2] * py + F[2]
            nn = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= nn
            dy /= nn
            dz /= nn
            ox, oy, oz = pos[0], pos[1], pos[2]
            if dy <= 0.0:
                # the playa: wet mud, puddles mirror the sky tone
                hit[i, j] = 0.0
                sg = -oy / min(dy, -1e-6)
                gx = ox + dx * sg
                gz = oz + dz * sg
                mpp = sg / f / max(1e-3, -dy) * 0.5
                m = _fbm(gx / 3.0, gz / 1.3, 91, 3)
                pud = 0.0
                if mpp < 0.8:
                    pud = min(1.0, max(0.0, (_fbm(gx / 6.0 + 4.0, gz / 2.2, 77, 3) - 0.18) * 6.0)) * (1.0 - mpp / 0.8)
                el = -dy
                haze = math.exp(-el * 14.0)
                g = 0.52 + 0.14 * m - 0.1 * (1.0 - haze)
                refl = 0.8 + 0.15 * haze - 0.3 * dusk
                g = g * (1.0 - pud) + refl * pud
                g = g * (1.0 - haze * 0.6) + (0.72 - 0.35 * dusk) * haze * 0.6
                ground[i, j] = g * (1.0 - 0.45 * dusk)
                gang[i, j] = 0.0
                continue
            hit[i, j] = 1.0
            # ---- canopy hit (Illinois false position on [0, s_hi])
            r0 = math.sqrt(ox * ox + oz * oz)
            s_lo = 0.0
            f_lo = oy - A * math.exp(-r0 / L)
            if oy >= A:
                s_hi = 1e-3
            else:
                s_hi = (A - oy) / max(dy, 1e-5)
            s_hi = min(s_hi, 4.0e4)
            hx = ox + dx * s_hi
            hz = oz + dz * s_hi
            f_hi = oy + dy * s_hi - A * math.exp(-math.sqrt(hx * hx + hz * hz) / L)
            side = 0
            s = s_hi
            for it in range(26):
                s = (s_lo * f_hi - s_hi * f_lo) / (f_hi - f_lo) if f_hi != f_lo else 0.5 * (s_lo + s_hi)
                hx = ox + dx * s
                hz = oz + dz * s
                fm = oy + dy * s - A * math.exp(-math.sqrt(hx * hx + hz * hz) / L)
                if fm * f_hi > 0:
                    s_hi = s
                    f_hi = fm
                    if side == -1:
                        f_lo *= 0.5
                    side = -1
                elif fm * f_lo > 0:
                    s_lo = s
                    f_lo = fm
                    if side == 1:
                        f_hi *= 0.5
                    side = 1
                else:
                    break
                if abs(s_hi - s_lo) < 1e-4 * s:
                    break
            hx = ox + dx * s
            hz = oz + dz * s
            r = math.sqrt(hx * hx + hz * hz)
            a = math.atan2(hz, hx)
            rr[i, j] = r
            cxo[i, j] = hx
            czo[i, j] = hz
            mpp = s / f                      # metres per design px at the hit
            mppo[i, j] = mpp
            # ---- facet of the crystal this ray passes through (ring k, meridian jj)
            k = int(math.floor(math.log(max(r, 1e-3) / 1.2) / lq)) + 1
            if r < 1.2:
                k = 0
            aa = a - 0.1308996938995747
            if aa < 0.0:
                aa += TWO_PI
            jj = int(aa / dth)
            hsh = _hash1(k, jj)
            hs2 = _hash1(k + 101, jj + 7)
            fr = aa / dth - jj
            # ---- the waters above, seen from beneath through the crystal: lit from above (bright overhead,
            # rippling with caustics), deepening towards the horizon, a Fresnel sheen right at the rim
            graze = 1.0 - min(1.0, dy * 4.5)
            wx = hx + fir * (hsh - 0.5) * 5.0
            wz = hz + fir * (hs2 - 0.5) * 5.0
            wn = _fbm(wx / 34.0 + t * 0.03, wz / 34.0 - t * 0.02, 11, 3)
            wv = _fbm(wx / 9.0 - t * 0.1, wz / 9.0, 13, 2)
            glory = math.exp(-r / 14.0)
            up = min(1.0, max(0.0, (dy - 0.08) / 0.5))
            wt = 0.2 + 0.1 * wn + 0.04 * wv + 0.42 * up + 0.8 * glory + 0.5 * graze ** 6
            # caustic network (bright wavering lines), faded when sub-pixel
            cf = max(0.0, 1.0 - mpp / 0.08)
            if cf > 0.0:
                q = _fbm(wx / 4.5 + t * 0.25, wz / 4.5 - t * 0.15, 17, 2) * 2.6
                q2 = _fbm(wx / 3.4 - t * 0.2, wz / 3.4 + 0.22 * t, 19, 2) * 2.6
                lw = 0.07 + 2.5 * mpp / 3.4
                c = max(0.0, 1.0 - abs(math.sin(math.pi * q)) / lw) + max(0.0, 1.0 - abs(math.sin(math.pi * q2)) / lw)
                wt += 0.45 * min(1.0, c) * cf * (0.35 + 0.65 * up)
            # the Leviathan, swimming in the waters above (a dark shape gliding over the crystal)
            lcx = -26.0 - 2.4 * (t - 119.0)
            lvx = (hx - lcx) / 17.0
            lvz = (hz - 20.0) / 4.2
            lv = lvx * lvx + lvz * lvz * (1.0 + 0.6 * max(0.0, lvx))
            fl1 = ((lvx - 1.25) / 0.28) ** 2 + ((lvz - 0.9) / 0.9) ** 2
            fl2 = ((lvx - 1.25) / 0.28) ** 2 + ((lvz + 0.9) / 0.9) ** 2
            fin = ((lvx + 0.2) / 0.3) ** 2 + ((lvz - 1.35) / 0.55) ** 2
            if lv < 1.0 or fl1 < 1.0 or fl2 < 1.0 or fin < 1.0:
                wt = wt * 0.3 + 0.03
            wt *= 1.0 - 0.5 * dusk
            water[i, j] = min(1.0, max(0.0, wt))
            # ---- crystal panes: a cut gem seen from below. Each facet's (slightly irregular) normal reflects
            # the bright horizon into a curved band of blazing facets; a few smoked facets; bevelled edges.
            if fir > 0.0:
                fsize = max(r, 1.0) * 0.26 / max(mpp, 1e-4)
                ff = min(1.0, max(0.0, (fsize - 3.0) / 9.0))
                bev = (1.0 - 4.0 * fr * (1.0 - fr)) ** 4
                rc = 0.6 if k == 0 else 1.2 * 1.55 ** (k - 0.5)
                thc = 0.1308996938995747 + (jj + 0.5) * dth
                hp = -A / L * math.exp(-rc / L)
                nx_ = -hp * math.cos(thc) + (hsh - 0.5) * 0.35
                nz_ = -hp * math.sin(thc) + (hs2 - 0.5) * 0.35
                nl = math.sqrt(nx_ * nx_ + 1.0 + nz_ * nz_)
                nx_ /= nl
                ny_ = 1.0 / nl
                nz_ /= nl
                dn = dx * nx_ + dy * ny_ + dz * nz_
                ry = dy - 2.0 * dn * ny_
                spec = math.exp(-(ry + 0.02) * (ry + 0.02) / 0.0035)
                # a sweeping band of blazing facets (the crystal catching the heavenly light), plus glitter
                db = math.atan2(math.sin(thc - 2.35), math.cos(thc - 2.35))
                band = math.exp(-db * db / 0.09) * (0.55 + 0.45 * math.sin(k * 1.7 + 0.8))
                if hsh < 0.13:
                    tn = 0.1
                    pa = 0.3
                else:
                    tn = 0.97
                    pa = 0.05 + 0.75 * spec + 0.7 * band * (1.0 if hs2 > 0.35 else 0.3) + 0.6 * (hsh > 0.93)
                pa = pa * ff + 0.22 * graze * (1.0 - ff)
                tn = tn * (1.0 - bev) + 0.98 * bev
                pa = max(pa, 0.4 * bev * ff)
                pane[i, j] = min(1.0, tn * (1.0 - 0.3 * dusk))
                pane_a[i, j] = min(1.0, pa) * fir
            # ---- storm clouds: an overcast plane (parallax-free at this height), rolling back from the apex
            if cover > 0.0:
                sc_ = (CLOUD_H - oy) / max(dy, 1e-4)
                cx = ox + dx * sc_
                cz = oz + dz * sc_
                # rolling back: the cloud texture streams outward from the apex as the hole grows
                cx = gax + (cx - gax) * roll
                cz = gaz + (cz - gaz) * roll
                n, lt = _cloud(cx, cz, t, 5)
                bil = 1.0 - abs(_fbm(cx / 300.0 + 7.0, cz / 300.0 - t * 0.006, 41, 3))
                el = dy
                haze = math.exp(-el * 7.0)
                lit = min(1.0, max(0.0, 0.5 + 7.0 * lt))
                dens = min(1.0, max(0.0, (n + 0.3) * 1.4))
                ct = 0.36 + 0.52 * lit + 0.16 * bil * bil - 0.16 * dens * (1.0 - lit)
                ct *= 1.0 - 0.5 * storm
                # the light behind the clouds where the heavens will open (towards the apex)
                cg = dx * gdx + dy * gdy + dz * gdz
                gl = gstr * math.exp((cg - 1.0) / 0.012)
                ct += gl * (0.5 + 0.5 * bil)
                ct = ct * (1.0 - haze) + 0.78 * haze
                ca = cover
                if reveal_r > 0.0:
                    ca_, sa_ = math.cos(a), math.sin(a)
                    edge = reveal_r * (0.8 + 0.4 * (0.5 + 0.5 * _fbm(ca_ * 2.2 + 5.0, sa_ * 2.2 + 0.002 * reveal_r, 23, 3)))
                    m = (r - edge) / max(4.0, 0.3 * edge)
                    ca *= min(1.0, max(0.0, m))
                    # the rolling rim is lit from the opened heaven
                    if m > -0.3 and m < 1.5:
                        ct += 0.45 * max(0.0, 1.0 - abs(m - 0.5) / 0.9)
                cloud[i, j] = min(1.0, max(0.0, ct)) * (1.0 - 0.55 * dusk)
                cloud_a[i, j] = min(1.0, max(0.0, ca))
                gang[i, j] = lt


def _grid_res(fc, res):
    return max(8, int(round(fc.w * res))), max(8, int(round(fc.h * res)))


def sky(fc, cam, t, *, cover=1.0, reveal_r=0.0, dusk=0.0, res=0.5, firmament=True, angles=False, glow=0.0,
        storm=0.0):
    ww, hh = _grid_res(fc, res)
    sc = fc.s * (ww / fc.w)
    keys = ("hit", "r", "cx", "cz", "mpp", "water", "pane", "pane_a", "cloud", "cloud_a", "ground", "gang")
    arrs = {k: np.zeros((hh, ww), np.float32) for k in keys}
    # the direction towards the apex (light behind the clouds) and the apex's point on the cloud plane
    g = np.array(APEX) - cam.pos
    g /= np.linalg.norm(g)
    sc_ = (CLOUD_H - cam.pos[1]) / max(g[1], 1e-3)
    gax, gaz = cam.pos[0] + g[0] * sc_, cam.pos[2] + g[2] * sc_
    roll = 1.0 / (1.0 + max(0.0, float(reveal_r)) / 260.0)
    _sky_kernel(ww, hh, sc, cam.pos, cam.R, cam.U, cam.F, cam.f, APEX_Y, CANOPY_L, float(t), float(reveal_r),
                float(cover), float(dusk), 1.0 if firmament else 0.0, float(g[0]), float(g[1]), float(g[2]),
                float(glow), float(gax), float(gaz), float(roll), float(storm), *[arrs[k] for k in keys])
    out = {}
    for k, v in arrs.items():
        out[k] = cv2.resize(v, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR) if (ww, hh) != (fc.w, fc.h) else v
    out["th"] = np.arctan2(out["cz"], out["cx"]).astype(np.float32)
    if angles:
        # engraving LINE direction: canopy -> along the rings; clouds -> along the billows
        gy, gx = np.gradient(np.log1p(out["r"]))
        ang_c = np.arctan2(gy, gx) + math.pi / 2
        cl = cv2.GaussianBlur(out["cloud"], (0, 0), max(1.0, 6 * fc.s))
        cy_, cx_ = np.gradient(cl)
        ang_k = np.arctan2(cy_, cx_ + 1e-6) + math.pi / 2
        ang = np.where(out["cloud_a"] > 0.5, ang_k, ang_c)
        out["angle"] = np.where(out["hit"] < 0.5, 0.0, ang).astype(np.float32)
    return out


def compose_sky(d, firmament=True):
    s = d["water"].copy()
    if firmament:
        s = s * (1 - d["pane_a"]) + d["pane"] * d["pane_a"]
    s = s * (1 - d["cloud_a"]) + d["cloud"] * d["cloud_a"]
    hit = d["hit"]
    return s * hit + d["ground"] * (1 - hit)

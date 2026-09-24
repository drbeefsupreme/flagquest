"""INTERIM stand-in for scenes/effigy.py (owner T05Worked) with the same API, used by t06 only until the
real module lands. World: metres, x right, y UP, z away from the camera; effigy axis x=z=0, ground y=0."""
import math

import numpy as np

HEAD_TOP = 14.6
POLE_M = 3.6
SEAT = (0.0, HEAD_TOP, 0.0)
APEX = (0.0, HEAD_TOP + POLE_M, 0.0)
APEX_Y = APEX[1]
CANOPY_L = 70.0
N_RIBS = 24


def canopy_y(r):
    return APEX_Y * np.exp(-np.asarray(r, np.float64) / CANOPY_L)


def canopy_point(r, th):
    r = np.asarray(r, np.float64)
    th = np.asarray(th, np.float64)
    return np.stack([r * np.sin(th), canopy_y(r), r * np.cos(th)], -1)


class Cam3:
    def __init__(self, pos=(0.0, 1.7, -34.0), yaw=0.0, pitch=0.21, roll=0.0, f=1300.0, cx=960.0, cy=540.0):
        self.pos = np.asarray(pos, np.float64)
        self.yaw, self.pitch, self.roll, self.f = float(yaw), float(pitch), float(roll), float(f)
        self.cx, self.cy = cx, cy
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        fw = np.array([math.sin(self.yaw) * cp, sp, math.cos(self.yaw) * cp])
        rt = np.array([math.cos(self.yaw), 0.0, -math.sin(self.yaw)])
        up = np.cross(fw, rt)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        self.R = np.stack([rt * cr + up * sr, up * cr - rt * sr, fw])   # rows: right, up, forward

    def cam_coords(self, P):
        d = np.asarray(P, np.float64).reshape(-1, 3) - self.pos
        return d @ self.R.T

    def project(self, P):
        c = self.cam_coords(P)
        z = np.maximum(c[:, 2], 1e-3)
        return self.cx + self.f * c[:, 0] / z, self.cy - self.f * c[:, 1] / z, c[:, 2]

    def px_per_m(self, depth):
        return self.f / np.maximum(depth, 1e-3)

    def lerp(self, o, a):
        return Cam3(self.pos + (o.pos - self.pos) * a, self.yaw + (o.yaw - self.yaw) * a,
                    self.pitch + (o.pitch - self.pitch) * a, self.roll + (o.roll - self.roll) * a,
                    self.f + (o.f - self.f) * a)


HANDOFF = Cam3()


def seat_screen(cam):
    x, y, _ = cam.project(np.array([SEAT]))
    return float(x[0]), float(y[0])


def apex_screen(cam):
    x, y, _ = cam.project(np.array([APEX]))
    return float(x[0]), float(y[0])


_B = None


def beams():
    global _B
    if _B is not None:
        return _B
    P0, P1, Wd, Pt = [], [], [], []

    def add(a, b, w, part):
        P0.append(a)
        P1.append(b)
        Wd.append(w)
        Pt.append(part)

    def lattice(cx, y0, y1, hw0, hw1, part, n, w=0.22, depth=0.8):
        for zz in (-depth, depth):
            for s in (-1, 1):
                add((cx + s * hw0, y0, zz), (cx + s * hw1, y1, zz), w * 1.3, part)
            for k in range(n + 1):
                t = k / n
                y = y0 + (y1 - y0) * t
                hw = hw0 + (hw1 - hw0) * t
                add((cx - hw, y, zz), (cx + hw, y, zz), w, part)
                if k < n:
                    t2 = (k + 1) / n
                    y2 = y0 + (y1 - y0) * t2
                    hw2 = hw0 + (hw1 - hw0) * t2
                    add((cx - hw, y, zz), (cx + hw2, y2, zz), w * 0.8, part)
                    add((cx + hw, y, zz), (cx - hw2, y2, zz), w * 0.8, part)

    lattice(0.0, 0.0, 2.4, 3.2, 3.0, 0, 3, 0.3, 1.6)
    for s in (-1, 1):                                     # legs
        lattice(s * 1.2, 2.4, 7.2, 0.55, 0.6, 1, 6)
    lattice(0.0, 7.2, 11.6, 1.8, 2.3, 2, 5)              # torso
    for s in (-1, 1):                                     # raised arms
        a0 = np.array([s * 2.1, 11.2, 0.0])
        a1 = np.array([s * 5.6, 15.8, 0.0])
        for o in np.linspace(-0.4, 0.4, 3):
            add(tuple(a0 + (0, o, 0)), tuple(a1 + (0, o, 0)), 0.22, 3)
        for k in range(7):
            t = k / 6
            p = a0 + (a1 - a0) * t
            add(tuple(p + (-0.3, -0.45, 0)), tuple(p + (0.3, 0.45, 0)), 0.18, 3)
    lattice(0.0, 12.0, 14.4, 0.9, 1.0, 4, 3, 0.2, 0.7)   # head
    add((0.0, 14.4, 0.0), (0.0, HEAD_TOP, 0.0), 0.35, 5)
    _B = dict(p0=np.array(P0, np.float64), p1=np.array(P1, np.float64), w=np.array(Wd), part=np.array(Pt))
    return _B


def draw_effigy(ctx, cam, *, light=(-0.5, 0.75, -0.4), dusk=0.0, char=None, glow=None, gone=None):
    b = beams()
    n = len(b["w"])
    x0, y0, z0 = cam.project(b["p0"])
    x1, y1, z1 = cam.project(b["p1"])
    order = np.argsort(-(z0 + z1))
    base = 0.62 - 0.35 * dusk
    for k in order:
        if gone is not None and gone[k]:
            continue
        if min(z0[k], z1[k]) < 0.2:
            continue
        g = base + 0.12 * (b["p0"][k, 2] < 0)
        if char is not None:
            g = g * (1 - char[k]) + 0.02 * char[k]
        if glow is not None:
            g = g + (1.0 - g) * glow[k]
        ctx.set_source_rgb(g, g, g)
        ctx.set_line_width(max(0.6, b["w"][k] * cam.f / (0.5 * (z0[k] + z1[k]))))
        ctx.move_to(x0[k], y0[k])
        ctx.line_to(x1[k], y1[k])
        ctx.stroke()


def draw_firmament(ctx, cam, t, *, reveal=1.0, dusk=0.0, alpha=1.0, panes=True, waters=True):
    if waters:
        ctx.save()
        ctx.set_source_rgba(0.22, 0.22, 0.23, alpha)
        ctx.paint()
        ctx.restore()
    if not panes:
        return
    ctx.save()
    ctx.set_line_width(1.2)
    ctx.set_source_rgba(0.75, 0.75, 0.76, 0.6 * alpha)
    for rr in (4, 9, 16, 26, 40, 60, 90, 140, 220):
        th = np.linspace(0, 2 * math.pi, 180)
        P = canopy_point(np.full_like(th, rr), th)
        _poly3(ctx, cam, P, True)
    for k in range(N_RIBS):
        th = 2 * math.pi * k / N_RIBS
        r = np.linspace(0.0, 260.0, 120)
        P = canopy_point(r, np.full_like(r, th))
        _poly3(ctx, cam, P, False)
    ctx.restore()


def _poly3(ctx, cam, P, close):
    x, y, z = cam.project(P)
    ok = z > 0.5
    started = False
    for i in range(len(x)):
        if not ok[i]:
            if started:
                ctx.stroke()
            started = False
            continue
        if not started:
            ctx.move_to(x[i], y[i])
            started = True
        else:
            ctx.line_to(x[i], y[i])
    if started:
        ctx.stroke()


def facets(cam):
    return []

"""THE FLAG. Plain yellow cloth on an untreated wooden pole. NO markings, NO outlines, NO borders,
NO text, NO emblems, NO other colours on the cloth: only smooth yellow shading from light.

(STUB v0 by Main: procedural waves. FlagSmith replaces the internals with a real verlet cloth sim,
keeping every public signature below.)

Public API
----------
Flag(width=150, height=100, pole=420, nx=22, ny=15, side=1, seed=0)
    .simulate(n, pole_fn, wind_fn, fps=24, substeps=10, warmup=48) -> FlagTrack
        pole_fn(i) -> (x, y, ang)  pole BASE (bottom) in world/design coords, ang radians (0 = straight up,
                                    positive = leaning clockwise / to the right)
        wind_fn(i) -> (wx, wy)      mean wind in design px/s at frame i (breeze 300, strong 700, gale 1200)
FlagTrack
    .n, .verts (n, ny, nx, 3), .energy (n,), .snap (n,)
    .draw(ctx, i, alpha=1.0, glow=0.0, pole=True, cloth=True, light=(-0.45, -0.8, 0.4))
    .pole_top(i) -> (x, y)
    .save(path) / FlagTrack.load(path)
draw_flag(ctx, x, y, pole=420, t=0.0, wind=1.0, side=1, ang=0.0, seed=0, alpha=1.0,
          cloth_w=None, cloth_h=None, glow=0.0)
    cheap animated flag for crowds/backgrounds (x, y = pole base)
"""
import math

import cairo
import numpy as np

from .config import C

FLAG_Y = np.array(C["flag"])
FLAG_S = np.array(C["flag_shadow"])
FLAG_D = np.array(C["flag_deep"])
FLAG_L = np.array(C["flag_light"])


def cloth_color(k):
    """k in [-1, 1]: -1 deep shadow, 0 base yellow, +1 highlight. Only yellows."""
    if k >= 0:
        return tuple(FLAG_Y + (FLAG_L - FLAG_Y) * min(k, 1))
    k = min(-k, 1)
    if k < 0.6:
        return tuple(FLAG_Y + (FLAG_S - FLAG_Y) * (k / 0.6))
    return tuple(FLAG_S + (FLAG_D - FLAG_S) * ((k - 0.6) / 0.4))


def draw_pole(ctx, x0, y0, x1, y1, width=7.0, alpha=1.0):
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) + 1e-9
    nx, ny = -dy / L * width / 2, dx / L * width / 2
    g = cairo.LinearGradient(x0 + nx, y0 + ny, x0 - nx, y0 - ny)
    ps, pl = C["pole_shadow"], C["pole"]
    g.add_color_stop_rgba(0, *ps, alpha)
    g.add_color_stop_rgba(0.45, *pl, alpha)
    g.add_color_stop_rgba(1, *ps, alpha)
    ctx.move_to(x0 + nx, y0 + ny)
    ctx.line_to(x1 + nx, y1 + ny)
    ctx.line_to(x1 - nx, y1 - ny)
    ctx.line_to(x0 - nx, y0 - ny)
    ctx.close_path()
    ctx.set_source(g)
    ctx.fill()


def _mesh_draw(ctx, P, shade, alpha):
    """P (ny,nx,2) grid points, shade (ny,nx) in [-1,1] -> smooth yellow mesh"""
    ny, nx = shade.shape
    m = cairo.MeshPattern()
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = P[j, i], P[j, i + 1], P[j + 1, i + 1], P[j + 1, i]
            m.begin_patch()
            m.move_to(*a)
            m.line_to(*b)
            m.line_to(*c)
            m.line_to(*d)
            for k, (jj, ii) in enumerate(((j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i))):
                m.set_corner_color_rgba(k, *cloth_color(shade[jj, ii]), alpha)
            m.end_patch()
    ctx.set_source(m)
    # fill the outline so patch seams never show background
    ctx.new_path()
    ctx.move_to(*P[0, 0])
    for i in range(1, nx):
        ctx.line_to(*P[0, i])
    for j in range(1, ny):
        ctx.line_to(*P[j, nx - 1])
    for i in range(nx - 2, -1, -1):
        ctx.line_to(*P[ny - 1, i])
    for j in range(ny - 2, 0, -1):
        ctx.line_to(*P[j, 0])
    ctx.close_path()
    ctx.fill()


class FlagTrack:
    def __init__(self, verts, poles, energy, snap, meta):
        self.verts, self.poles, self.energy, self.snap, self.meta = verts, poles, energy, snap, meta
        self.n = len(verts)

    def pole_top(self, i):
        x, y, a = self.poles[min(i, self.n - 1)]
        L = self.meta["pole"]
        return x + math.sin(a) * L, y - math.cos(a) * L

    def draw(self, ctx, i, alpha=1.0, glow=0.0, pole=True, cloth=True, light=(-0.45, -0.8, 0.4)):
        i = int(min(max(i, 0), self.n - 1))
        x, y, a = self.poles[i]
        tx, ty = self.pole_top(i)
        if pole:
            draw_pole(ctx, x, y, tx, ty, self.meta.get("pole_w", 7.0), alpha)
        if cloth:
            V = self.verts[i]
            P = V[..., :2]
            # shading from finite-difference normals
            dz_dx = np.gradient(V[..., 2], axis=1)
            dz_dy = np.gradient(V[..., 2], axis=0)
            nrm = np.stack([-dz_dx, -dz_dy, np.ones_like(dz_dx) * 6.0], -1)
            nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
            L = np.array(light) / np.linalg.norm(light)
            sh = np.clip((nrm @ L) * 2.2 - 0.55, -1, 1)
            _mesh_draw(ctx, P, sh, alpha)

    def save(self, path):
        np.savez_compressed(path, verts=self.verts, poles=self.poles, energy=self.energy, snap=self.snap,
                            meta=np.array([repr(self.meta)]))

    @staticmethod
    def load(path):
        z = np.load(path)
        return FlagTrack(z["verts"], z["poles"], z["energy"], z["snap"], eval(str(z["meta"][0])))


class Flag:
    def __init__(self, width=150.0, height=100.0, pole=420.0, nx=22, ny=15, side=1, seed=0, pole_w=7.0):
        self.width, self.height, self.pole, self.nx, self.ny = width, height, pole, nx, ny
        self.side, self.seed, self.pole_w = side, seed, pole_w

    def simulate(self, n, pole_fn, wind_fn, fps=24, substeps=10, warmup=48):
        nx, ny = self.nx, self.ny
        verts = np.zeros((n, ny, nx, 3), np.float32)
        poles = np.zeros((n, 3), np.float32)
        energy = np.zeros(n, np.float32)
        u = np.linspace(0, 1, nx)[None, :]
        v = np.linspace(0, 1, ny)[:, None]
        for i in range(n):
            x, y, a = pole_fn(i)
            poles[i] = (x, y, a)
            wx, wy = wind_fn(i)
            wmag = math.hypot(wx, wy)
            k = min(1.0, wmag / 600.0)
            t = i / fps
            ph = t * (5 + 5 * k) - u * 7.0 + self.seed
            wave = np.sin(ph) * (0.18 + 0.1 * k) + 0.06 * np.sin(2.3 * ph + v * 4)
            droop = (1 - k) * 0.85
            ca, sa = math.cos(a), math.sin(a)
            topx, topy = x + sa * self.pole, y - ca * self.pole
            lx = u * self.width * (1 - 0.35 * droop) * np.cos(wave * 0.6)
            ly = v * self.height + wave * self.height * 0.25 * u + droop * u * self.height * 0.9
            dirx = self.side * (1 if wx >= 0 else -1) if wmag > 1 else self.side
            X = topx + dirx * (lx * ca) - ly * sa
            Y = topy + dirx * (lx * sa) + ly * ca
            Z = np.cos(ph) * u * self.width * 0.25
            verts[i] = np.stack([X + 0 * v, Y + 0 * u, Z + 0 * v], -1)
            energy[i] = k * (0.6 + 0.4 * abs(math.sin(t * 7)))
        return FlagTrack(verts, poles, energy, np.zeros(n, np.float32),
                         {"pole": self.pole, "pole_w": self.pole_w, "side": self.side})


def draw_flag(ctx, x, y, pole=420, t=0.0, wind=1.0, side=1, ang=0.0, seed=0, alpha=1.0,
              cloth_w=None, cloth_h=None, glow=0.0):
    cw = cloth_w or pole * 0.36
    ch = cloth_h or pole * 0.24
    ca, sa = math.cos(ang), math.sin(ang)
    tx, ty = x + sa * pole, y - ca * pole
    draw_pole(ctx, x, y, tx, ty, max(1.2, pole * 0.017), alpha)
    nx, ny = 9, 6
    u = np.linspace(0, 1, nx)[None, :]
    v = np.linspace(0, 1, ny)[:, None]
    k = min(1.0, wind)
    ph = t * (5 + 4 * k) - u * 6.5 + seed * 1.7
    wave = np.sin(ph) * 0.2
    droop = (1 - k) * 0.8
    lx = u * cw * (1 - 0.3 * droop)
    ly = v * ch + wave * ch * 0.3 * u + droop * u * ch * 0.9
    X = tx + side * lx * ca - ly * sa
    Y = ty + side * lx * sa + ly * ca
    P = np.stack([X + 0 * v, Y + 0 * u], -1)
    sh = np.clip(np.cos(ph) * 0.8 * (0.3 + u), -1, 1) + 0 * v
    _mesh_draw(ctx, P, sh, alpha)

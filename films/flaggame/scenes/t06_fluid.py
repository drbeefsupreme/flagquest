"""t06 fire: a real 2D incompressible gas simulation (Stam stable fluids) of the effigy burn and its steam
explosion, run once in setup and cached.

State on a cell-centred grid (world px = design px of the establishing shot):
  u, v      velocity (cells/s), buoyancy from temperature, vorticity confinement for curl
  temp      temperature (drives buoyancy; cools slowly)
  flame     flame intensity (emitted at burning lumber, decays in ~0.4 s -> tongues)
  smoke     soot (emitted at the flame tips; dark in print)
  steam     water vapour (born where the deluge water meets heat; white in print)
Semi-Lagrangian MacCormack advection (clamped), red-black SOR projection with a DIVERGENCE SOURCE (steam
generation expands the gas -> a genuine explosion), solid mask (ground, crystal firmament ceiling).
Sparks: embers advected by the flow (+ own lift, drag, jitter), recorded per frame.
"""
import math

import numba as nb
import numpy as np


# ============================================================ kernels
@nb.njit(cache=True, fastmath=True, inline="always")
def _bil(f, x, y):
    ny, nx = f.shape
    if x < 0.0:
        x = 0.0
    if y < 0.0:
        y = 0.0
    if x > nx - 1.001:
        x = nx - 1.001
    if y > ny - 1.001:
        y = ny - 1.001
    i = int(x)
    j = int(y)
    a = x - i
    b = y - j
    return (f[j, i] * (1 - a) * (1 - b) + f[j, i + 1] * a * (1 - b) + f[j + 1, i] * (1 - a) * b
            + f[j + 1, i + 1] * a * b)


@nb.njit(cache=True, fastmath=True, parallel=True)
def _advect(q, u, v, dt, out):
    ny, nx = q.shape
    for j in nb.prange(ny):
        for i in range(nx):
            out[j, i] = _bil(q, i - dt * u[j, i], j - dt * v[j, i])


@nb.njit(cache=True, fastmath=True, parallel=True)
def _maccormack(q, u, v, dt, fwd, bwd, out):
    """clamped MacCormack: sharper than plain semi-Lagrangian, stays bounded"""
    ny, nx = q.shape
    _advect(q, u, v, dt, fwd)
    _advect(fwd, u, v, -dt, bwd)
    for j in nb.prange(ny):
        for i in range(nx):
            x = i - dt * u[j, i]
            y = j - dt * v[j, i]
            if x < 0.0:
                x = 0.0
            if y < 0.0:
                y = 0.0
            if x > nx - 1.001:
                x = nx - 1.001
            if y > ny - 1.001:
                y = ny - 1.001
            ii = int(x)
            jj = int(y)
            lo = min(min(q[jj, ii], q[jj, ii + 1]), min(q[jj + 1, ii], q[jj + 1, ii + 1]))
            hi = max(max(q[jj, ii], q[jj, ii + 1]), max(q[jj + 1, ii], q[jj + 1, ii + 1]))
            r = fwd[j, i] + 0.5 * (q[j, i] - bwd[j, i])
            if r < lo:
                r = lo
            if r > hi:
                r = hi
            out[j, i] = r


@nb.njit(cache=True, fastmath=True, parallel=True)
def _divergence(u, v, solid, src, div):
    ny, nx = u.shape
    for j in nb.prange(1, ny - 1):
        for i in range(1, nx - 1):
            if solid[j, i]:
                div[j, i] = 0.0
                continue
            ur = 0.0 if solid[j, i + 1] else u[j, i + 1]
            ul = 0.0 if solid[j, i - 1] else u[j, i - 1]
            vd = 0.0 if solid[j + 1, i] else v[j + 1, i]
            vu = 0.0 if solid[j - 1, i] else v[j - 1, i]
            div[j, i] = 0.5 * (ur - ul + vd - vu) - src[j, i]


@nb.njit(cache=True, fastmath=True, parallel=True)
def _sor(p, div, solid, iters, omega):
    ny, nx = p.shape
    for it in range(iters):
        for col in range(2):
            for j in nb.prange(1, ny - 1):
                i0 = 1 + (j + col) % 2
                for i in range(i0, nx - 1, 2):
                    if solid[j, i]:
                        continue
                    s = 0.0
                    n = 0.0
                    if not solid[j, i + 1]:
                        s += p[j, i + 1]
                        n += 1.0
                    if not solid[j, i - 1]:
                        s += p[j, i - 1]
                        n += 1.0
                    if not solid[j + 1, i]:
                        s += p[j + 1, i]
                        n += 1.0
                    if not solid[j - 1, i]:
                        s += p[j - 1, i]
                        n += 1.0
                    if n > 0.0:
                        pn = (s - div[j, i]) / n
                        p[j, i] += omega * (pn - p[j, i])


@nb.njit(cache=True, fastmath=True, parallel=True)
def _subgrad(u, v, p, solid):
    ny, nx = u.shape
    for j in nb.prange(1, ny - 1):
        for i in range(1, nx - 1):
            if solid[j, i]:
                u[j, i] = 0.0
                v[j, i] = 0.0
                continue
            pc = p[j, i]
            pr = pc if solid[j, i + 1] else p[j, i + 1]
            pl = pc if solid[j, i - 1] else p[j, i - 1]
            pd = pc if solid[j + 1, i] else p[j + 1, i]
            pu = pc if solid[j - 1, i] else p[j - 1, i]
            u[j, i] -= 0.5 * (pr - pl)
            v[j, i] -= 0.5 * (pd - pu)


@nb.njit(cache=True, fastmath=True, parallel=True)
def _vorticity(u, v, eps, dt, w, out_u, out_v):
    ny, nx = u.shape
    for j in nb.prange(1, ny - 1):
        for i in range(1, nx - 1):
            w[j, i] = 0.5 * (v[j, i + 1] - v[j, i - 1] - u[j + 1, i] + u[j - 1, i])
    for j in nb.prange(2, ny - 2):
        for i in range(2, nx - 2):
            gx = 0.5 * (abs(w[j, i + 1]) - abs(w[j, i - 1]))
            gy = 0.5 * (abs(w[j + 1, i]) - abs(w[j - 1, i]))
            l = math.sqrt(gx * gx + gy * gy) + 1e-5
            gx /= l
            gy /= l
            out_u[j, i] = u[j, i] + eps * dt * gy * w[j, i]
            out_v[j, i] = v[j, i] - eps * dt * gx * w[j, i]


@nb.njit(cache=True, fastmath=True)
def _hash(i, j, k):
    h = (i * 374761393 + j * 668265263 + k * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0x1000000)


@nb.njit(cache=True, fastmath=True)
def _vnoise3(x, y, z):
    i = int(math.floor(x))
    j = int(math.floor(y))
    k = int(math.floor(z))
    fx = x - i
    fy = y - j
    fz = z - k
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    uz = fz * fz * (3 - 2 * fz)
    r = 0.0
    for dk in range(2):
        wz = uz if dk else 1 - uz
        for dj in range(2):
            wy = uy if dj else 1 - uy
            for di in range(2):
                wx = ux if di else 1 - ux
                r += wx * wy * wz * _hash(i + di, j + dj, k + dk)
    return r * 2.0 - 1.0


@nb.njit(cache=True, fastmath=True, parallel=True)
def _forces(u, v, temp, flame, smoke, steam, water, fuel, burn, solid, t, dt, prm, gen):
    """sources + buoyancy + deluge coupling. burn (ny,) = how lit each row is (0..1).
    prm: [emit_flame, emit_temp, emit_smoke, lift_T, weight_S, lift_M, cool_T, decay_F, jet, noise_f,
          water_quench, water_steam, water_push, wind_x, soot_gen, fuel_gain]"""
    ny, nx = u.shape
    eF, eT, eS, bT, wS, bM, cT, dF, jet, nf, wq, ws, wp, wx, sg, fg = (prm[0], prm[1], prm[2], prm[3], prm[4],
                                                                      prm[5], prm[6], prm[7], prm[8], prm[9],
                                                                      prm[10], prm[11], prm[12], prm[13],
                                                                      prm[14], prm[15])
    for j in nb.prange(ny):
        for i in range(nx):
            gen[j, i] = 0.0
            if solid[j, i]:
                continue
            fu = fuel[j, i] * burn[j] * fg
            if fu > 0.0:
                n = _vnoise3(i * nf, j * nf + t * 3.1, t * 1.7)
                n2 = _vnoise3(i * nf * 2.7 + 11.0, j * nf * 2.7 + t * 6.0, t * 3.3)
                k = max(0.0, 0.55 + 0.6 * n + 0.35 * n2) * fu
                flame[j, i] += dt * eF * k
                temp[j, i] += dt * eT * k
                smoke[j, i] += dt * eS * k
                v[j, i] -= dt * jet * k
                u[j, i] += dt * jet * 0.35 * n2 * k
            # soot where flames die
            f = flame[j, i]
            if f > 0.0:
                smoke[j, i] += dt * sg * f * dF
            # buoyancy (up is -y)
            v[j, i] -= dt * (bT * temp[j, i] + bM * steam[j, i] - wS * smoke[j, i])
            u[j, i] += dt * wx * temp[j, i] * 0.05
            # deluge water: drags air down, quenches heat into steam, the steam expands (divergence source)
            w = water[j, i]
            if w > 0.0:
                q = 1.0 - math.exp(-wq * w * dt)
                h = temp[j, i] * q
                temp[j, i] -= h
                flame[j, i] *= 1.0 - q
                steam[j, i] += ws * h
                gen[j, i] = ws * h / dt
                v[j, i] += dt * wp * w
            temp[j, i] *= 1.0 - cT * dt
            flame[j, i] *= 1.0 - dF * dt


@nb.njit(cache=True, fastmath=True)
def _sparks_step(px, py, pvx, pvy, age, life, u, v, temp, solid, dt, drag, lift, jit, seed, k):
    n = px.shape[0]
    ny, nx = u.shape
    for s in range(n):
        if age[s] < 0.0 or age[s] > life[s]:
            continue
        x = px[s]
        y = py[s]
        fu = _bil(u, x, y)
        fv = _bil(v, x, y)
        tt = _bil(temp, x, y)
        pvx[s] += (fu - pvx[s]) * min(1.0, drag * dt)
        pvy[s] += (fv - pvy[s]) * min(1.0, drag * dt) - lift * min(tt, 2.0) * dt + 4.0 * dt
        pvx[s] += jit * dt * (_hash(s, k, seed) - 0.5)
        pvy[s] += jit * dt * (_hash(s, k, seed + 7) - 0.5)
        px[s] = x + pvx[s] * dt
        py[s] = y + pvy[s] * dt
        age[s] += dt
        ii = int(px[s])
        jj = int(py[s])
        if ii < 0 or jj < 0 or ii >= nx or jj >= ny or solid[jj, ii]:
            age[s] = life[s] + 1.0


# ============================================================ driver
class FireSim:
    """grid covering world x in [x0, x0 + nx*h], y (metres, UP) in [ytop - ny*h, ytop]; row j grows DOWN."""

    def __init__(self, nx, ny, h, x0, ytop):
        self.nx, self.ny, self.h, self.x0, self.ytop = nx, ny, float(h), float(x0), float(ytop)
        z = lambda: np.zeros((ny, nx), np.float32)
        self.u, self.v, self.temp, self.flame, self.smoke, self.steam = z(), z(), z(), z(), z(), z()
        self.water, self.fuel, self.p = z(), z(), z()
        self._t = [z() for _ in range(6)]
        self.solid = np.zeros((ny, nx), np.bool_)

    def world(self, i, j):
        return self.x0 + (np.asarray(i) + 0.5) * self.h, self.ytop - (np.asarray(j) + 0.5) * self.h

    def cell(self, x, y):
        return (np.asarray(x) - self.x0) / self.h - 0.5, (self.ytop - np.asarray(y)) / self.h - 0.5

    def step(self, t, dt, burn, prm, vort=2.0, iters=40, spark=None):
        u, v = self.u, self.v
        a, b, c, d, e, f = self._t
        _forces(u, v, self.temp, self.flame, self.smoke, self.steam, self.water, self.fuel, burn, self.solid,
                t, dt, prm, a)
        gen = a.copy()
        if getattr(self, "boom", None) is not None:
            gen += self.boom
        _vorticity(u, v, vort, dt, b, c, d)
        u[2:-2, 2:-2] = c[2:-2, 2:-2]
        v[2:-2, 2:-2] = d[2:-2, 2:-2]
        _advect(u, u, v, dt, b)
        _advect(v, u, v, dt, c)
        u[:] = b
        v[:] = c
        # expansion rate (cells^2/s -> per-step divergence source)
        _divergence(u, v, self.solid, gen * dt * 0.5, e)
        self.p *= 0.6
        _sor(self.p, e, self.solid, iters, 1.8)
        _subgrad(u, v, self.p, self.solid)
        # divergent (potential) flow of an explosion, re-imposed after the projection while it lasts
        bu = getattr(self, "boom_u", None)
        if bu is not None:
            u += bu
            v += self.boom_v
        # open borders: zero-gradient
        for q in (u, v):
            q[:, 0] = q[:, 1]
            q[:, -1] = q[:, -2]
            q[0] = q[1]
            q[-1] = 0.0
        for q in (self.temp, self.flame, self.smoke, self.steam):
            _maccormack(q, u, v, dt, b, c, d)
            q[:] = d
            q[self.solid] = 0.0
        if bu is not None:
            u -= 0.85 * bu
            v -= 0.85 * self.boom_v
            u[self.solid] = 0.0
            v[self.solid] = 0.0
        u *= 1.0 - 0.02 * dt
        v *= 1.0 - 0.02 * dt
        self.smoke *= 1.0 - 0.015 * dt
        self.steam *= 1.0 - 0.05 * dt

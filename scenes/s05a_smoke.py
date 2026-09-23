"""s05a breach smoke: a real 2D incompressible fluid simulation (Stam 'stable fluids': semi-Lagrangian
advection, Jacobi pressure projection, vorticity confinement, buoyancy) of the blast cloud pouring through
the blown door. Screen-space grid; precomputed once in setup and cached to S.cache."""
import math

import numba as nb
import numpy as np

NX, NY = 256, 144          # cell = 7.5 design px


@nb.njit(cache=True)
def _bilerp(f, x, y):
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


@nb.njit(cache=True)
def _advect(q, u, v, dt):
    ny, nx = q.shape
    out = np.empty_like(q)
    for j in range(ny):
        for i in range(nx):
            x = i - dt * u[j, i]
            y = j - dt * v[j, i]
            out[j, i] = _bilerp(q, x, y)
    return out


@nb.njit(cache=True)
def _project(u, v, iters):
    ny, nx = u.shape
    div = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            div[j, i] = -0.5 * (u[j, i + 1] - u[j, i - 1] + v[j + 1, i] - v[j - 1, i])
    for it in range(iters):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                p[j, i] = (div[j, i] + p[j, i - 1] + p[j, i + 1] + p[j - 1, i] + p[j + 1, i]) * 0.25
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u[j, i] -= 0.5 * (p[j, i + 1] - p[j, i - 1])
            v[j, i] -= 0.5 * (p[j + 1, i] - p[j - 1, i])


@nb.njit(cache=True)
def _vorticity(u, v, eps, dt):
    ny, nx = u.shape
    w = np.zeros((ny, nx))
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            w[j, i] = 0.5 * ((v[j, i + 1] - v[j, i - 1]) - (u[j + 1, i] - u[j - 1, i]))
    for j in range(2, ny - 2):
        for i in range(2, nx - 2):
            gx = 0.5 * (abs(w[j, i + 1]) - abs(w[j, i - 1]))
            gy = 0.5 * (abs(w[j + 1, i]) - abs(w[j - 1, i]))
            L = math.sqrt(gx * gx + gy * gy) + 1e-6
            gx /= L
            gy /= L
            u[j, i] += eps * dt * gy * w[j, i]
            v[j, i] -= eps * dt * gx * w[j, i]


def simulate(n_frames, door_rect, center, t_of_frame, seed=3, steps_per_frame=3):
    """door_rect = (x0, y0, x1, y1) design px of the blown opening; center = blast centre (design px);
    t_of_frame(k) -> sim seconds at output frame k (speed ramp). Returns density (n, NY, NX) float16 and
    the injected 'heat' (brightness) field (n, NY, NX) float16."""
    rng = np.random.default_rng(seed)
    sc = NX / 1920.0
    cx, cy = center[0] * sc, center[1] * sc
    x0, y0, x1, y1 = [c * sc for c in door_rect]
    yy, xx = np.mgrid[0:NY, 0:NX].astype(np.float64)
    u = np.zeros((NY, NX))
    v = np.zeros((NY, NX))
    d = np.zeros((NY, NX))
    heat = np.zeros((NY, NX))
    inside = ((xx > x0) & (xx < x1) & (yy > y0) & (yy < y1)).astype(np.float64)
    ring = np.exp(-((np.maximum(np.abs(xx - (x0 + x1) / 2) - (x1 - x0) / 2, 0) ** 2 +
                     np.maximum(np.abs(yy - (y0 + y1) / 2) - (y1 - y0) / 2, 0) ** 2) / 6.0))
    noise = rng.normal(size=(NY, NX))
    from scipy.ndimage import gaussian_filter
    noise = gaussian_filter(noise, 3.0)
    noise /= noise.std() + 1e-9
    dens_out = np.zeros((n_frames, NY, NX), np.float16)
    heat_out = np.zeros((n_frames, NY, NX), np.float16)
    t_prev = 0.0
    for k in range(n_frames):
        t_target = t_of_frame(k)
        dt_total = max(0.0, t_target - t_prev)
        t_prev = t_target
        n_sub = steps_per_frame
        for s in range(n_sub):
            ts = t_target - dt_total * (1 - (s + 1) / n_sub)
            dt = dt_total / n_sub * 60.0          # grid-steps of 1/60 s
            if dt <= 0:
                continue
            # blast impulse: strongest right at detonation, decaying
            burst = math.exp(-ts / 0.05)
            src = inside * 0.8 + ring * 1.4
            dx, dy = xx - cx, yy - cy
            r = np.sqrt(dx * dx + dy * dy) + 1.0
            push = (7.0 * burst + 0.5) * src
            u += dt * push * dx / r * (1.0 + 0.35 * noise)
            v += dt * push * dy / r * (1.0 + 0.35 * noise)
            src_d = inside * 0.7 * burst + ring * (0.5 * burst + 0.06)
            d += dt * src_d * 0.45 * (1.0 + 0.3 * noise)
            heat += dt * src * (0.5 * burst + 0.02)
            v -= dt * 0.04 * d                   # buoyancy (up = -y)
            _vorticity(u, v, 0.8, dt)
            u = _advect(u, u, v, dt)
            v = _advect(v, u, v, dt)
            _project(u, v, 24)
            d = _advect(d, u, v, dt) * (1 - 0.002 * dt)
            heat = _advect(heat, u, v, dt) * (1 - 0.06 * dt)
            u *= (1 - 0.01 * dt)
            v *= (1 - 0.01 * dt)
        dens_out[k] = np.clip(d, 0, 60).astype(np.float16)
        heat_out[k] = np.clip(heat, 0, 60).astype(np.float16)
    return dens_out, heat_out

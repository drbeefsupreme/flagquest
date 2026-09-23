"""s07a world — the dawn plain of the Unbabeling, rendered as a true 3D perspective world.

Used by scenes/s07a_plant.py and (read-only) by scenes/s07b_flagistan.py for a pixel-exact hand-off.

WORLD
  units: metres.  Z is up.  +Y = north, +X = east.  Ground plane z = 0.
  Lutie's rubble hill: centre (0, 0), radius HILL_R, height HILL_H (the crest / plant point is at (0, 0, HILL_TOP)).
  Plates: the dried-slop crust is cracked into Voronoi plates of a jittered grid (cell CELL m, jitter JIT);
          every plate cell (i, j) may hold one survivor at its seed = the plate's centre of mass-ish.
          Plates are procedural and infinite; survivors live on a finite disc (POP_R0 .. POP_R1 m).
  Sun: azimuth SUN_AZ (degrees clockwise from north; -70 = WNW), elevation sun_elev(T) degrees.
       Light direction (towards the sun) = sun_dir(T).  Wind blows towards azimuth wind_az(T).

CAMERA  (see camera_at)
  dict(pos=(x, y, z) metres, yaw=deg (0 = looking north, + = turning right/east), pitch=deg (+ = looking DOWN),
       hfov=deg (horizontal field of view), roll=deg, shake=(dx, dy) design px)
  Pinhole: screen x = W/2 + F*xc/zc, y = H/2 - F*yc/zc with F = (W/2)/tan(hfov/2) in design px.
  camera_at(T) is defined for any T (extrapolates the final crane beyond 154.5 by continuing its exponential rise).

ENTRY POINTS
  camera_at(T) -> cam dict
  get_world(S_or_None) -> World (population + activation times; cached in cache/s07a/world.npz)
  render_world(fc, cam, T, world=None, heroes=None) -> float32 (fc.h, fc.w, 3) sRGB, pre-post
      (fc needs .w .h .s; any camera works: altitude up to km, pitch up to 90).
"""
import math
from pathlib import Path

import cairo
import cv2
import numpy as np
import numba as nb
from numba import njit, prange

from vx.config import W, H, C, CACHE
from vx.canvas import Canvas, set_color, circle, ellipse, poly, blur, col, over
from vx.ease import clamp, lerp, seg, smoothstep, ease_in_out, ease_out, ease_in, ease_in_out_sine, hash01, fbm1, noise1

# ============================================================== constants
CELL = 4.0
JIT = 0.86
HILL_R = 21.0
HILL_H = 7.2
HILL_TOP = HILL_H
POP_R0 = 27.0
POP_R1 = 1250.0
SUN_AZ = -73.5
T_LOOK = 148.5
T_CONV = 150.0
T_PLANT = 142.0

GLOW_COLS = np.array([
    (0.35, 0.86, 1.00),   # cyan feed
    (1.00, 0.32, 0.72),   # magenta feed
    (0.40, 1.00, 0.58),   # green feed
    (0.66, 0.46, 1.00),   # violet feed
    (0.62, 0.78, 1.00),   # cold white-blue feed
    (1.00, 0.45, 0.45),   # red notification
], np.float32)

# plate albedos (dried slop crust, cool and desaturated)
PLATE_ALB = np.array([
    (0.46, 0.45, 0.56),
    (0.41, 0.42, 0.55),
    (0.50, 0.47, 0.56),
    (0.43, 0.46, 0.58),
    (0.53, 0.49, 0.57),
    (0.39, 0.38, 0.50),
    (0.48, 0.44, 0.52),
], np.float32)


# ============================================================== time curves
def sun_elev(T):
    """degrees; the first limb breaks the horizon right on the plant (142.0)"""
    a = -1.6 + 1.25 * seg(T, 139.0, 142.0, lambda x: x)            # slow approach
    b = 1.05 * seg(T, 141.7, 143.2, ease_out)                       # quick emergence around the plant
    c = 2.4 * seg(T, 143.0, 158.0, lambda x: x)                      # slow climb
    return a + b + c


def sun_dir(T):
    e = math.radians(sun_elev(T))
    az = math.radians(SUN_AZ)
    return np.array([math.sin(az) * math.cos(e), math.cos(az) * math.cos(e), math.sin(e)])


def wind_az(T):
    """azimuth the wind blows TOWARDS (deg). veers from NE to E during the crane."""
    return 28.0 + 55.0 * seg(T, 147.0, 154.5, ease_in_out_sine) + 8.0 * seg(T, 154.5, 160.0, ease_out)


def light_curves(T):
    """global dawn light levels"""
    return dict(
        line=seg(T, 139.06, 139.42, ease_out),                    # the thin dawn line draws across
        dawn=seg(T, 139.30, 141.60, lambda x: ease_in_out(x, 2)),  # gradient blooms upward
        sky=0.10 + 0.90 * seg(T, 139.25, 141.4, lambda x: ease_in_out(x, 2)),
        sun=seg(T, 141.85, 144.2, ease_out) * (0.8 + 0.2 * seg(T, 144, 154, ease_in_out)),
        amb=0.35 + 0.5 * seg(T, 139.6, 142.5, ease_in_out) + 0.1 * seg(T, 150, 156, ease_in_out),
        glow=seg(T, 139.25, 140.6, lambda x: ease_in_out(x, 2)),
        stars=seg(T, 139.1, 139.8) * (1 - seg(T, 140.3, 141.6)),
    )


# ============================================================== camera
def _orbit_pos(ang_deg, rad, z):
    a = math.radians(ang_deg)
    return (rad * math.cos(a), rad * math.sin(a), z)


# Camera keys framed on the hill top: (T, orbit angle deg, orbit radius m, altitude m, hfov deg, hill_x, hill_y)
# hill_x/hill_y = where the crest (0, 0, HILL_TOP) sits on screen (design px); yaw/pitch are derived from it.
CAM_KEYS = [
    (139.00, -26.0, 19.5, HILL_TOP + 1.9, 31.0, 0.575 * W, 0.79 * H),
    (140.10, -26.0, 19.5, HILL_TOP + 1.7, 31.0, 0.575 * W, 0.775 * H),
    (141.45, -26.0, 19.5, HILL_TOP - 0.55, 31.0, 0.575 * W, 0.665 * H),
    (142.30, -26.0, 19.5, HILL_TOP - 0.55, 31.0, 0.575 * W, 0.665 * H),
    (146.60, -25.5, 18.8, HILL_TOP - 0.45, 20.5, 0.545 * W, 0.75 * H),
    (148.00, -31.0, 21.5, HILL_TOP + 2.8, 25.0, 0.51 * W, 0.80 * H),
    (149.50, -40.0, 26.0, HILL_TOP + 7.0, 30.0, 0.50 * W, 0.90 * H),
    (151.00, -54.0, 36.0, 18.7, 40.0, 0.50 * W, 0.96 * H),
    (152.00, -66.0, 44.0, 38.0, 44.0, 0.50 * W, 0.95 * H),
    (153.40, -80.0, 54.0, 80.0, 46.0, 0.50 * W, 0.92 * H),
    (154.50, -90.0, 60.0, 140.0, 48.0, 0.50 * W, 0.93 * H),
]
T_CRANE1 = 154.5
_PCHIP = None


def _hermite_params(T):
    """monotone (PCHIP) interpolation of the camera keys -> (values, d/dT); log-altitude"""
    global _PCHIP
    if _PCHIP is None:
        from scipy.interpolate import PchipInterpolator
        ts = np.array([k[0] for k in CAM_KEYS])
        vals = np.array([[k[1], k[2], math.log(k[3]), k[4], k[5], k[6]] for k in CAM_KEYS], float)
        f = PchipInterpolator(ts, vals, axis=0, extrapolate=True)
        _PCHIP = (f, f.derivative(), ts)
    f, df, ts = _PCHIP
    Tc = min(max(T, ts[0]), ts[-1])
    return f(Tc), df(Tc)


def _frame_on_hill(ang, rad, z, hfov, hx, hy):
    px, py, pz = _orbit_pos(ang, rad, z)
    dx, dy, dz = -px, -py, HILL_TOP - pz
    az = math.degrees(math.atan2(dx, dy))
    dep = math.degrees(math.atan2(-dz, math.hypot(dx, dy)))
    F = (W / 2) / math.tan(math.radians(hfov) / 2)
    pitch = dep - math.degrees(math.atan((hy - H / 2) / F))
    yaw = az - math.degrees(math.atan((hx - W / 2) / F))
    return dict(pos=(px, py, pz), yaw=yaw, pitch=pitch, hfov=hfov, roll=0.0)


def camera_at(T):
    """camera for global time T (valid for any T; beyond 154.5 the crane keeps rising and tilting down)"""
    if T <= T_CRANE1:
        v, _ = _hermite_params(T)
        ang, rad, lz, hfov, hx, hy = v
        # a breath of handheld drift in the locked-off section
        dr = 0.18 * math.sin((T - 139) * 0.37) * (1 - seg(T, 146.5, 148, ease_in_out))
        return _frame_on_hill(ang + dr, rad, math.exp(lz), hfov, hx, hy)
    # ---- beyond the hand-off: keep the exponential rise, ease the tilt toward straight down
    v, tg = _hermite_params(T_CRANE1)
    ang, rad, lz, hfov, hx, hy = v
    dt = T - T_CRANE1
    end = _frame_on_hill(ang, rad, math.exp(lz), hfov, hx, hy)
    prev = _frame_on_hill(*_hermite_params(T_CRANE1 - 0.02)[0][:2], math.exp(_hermite_params(T_CRANE1 - 0.02)[0][2]),
                          *_hermite_params(T_CRANE1 - 0.02)[0][3:])
    rate = max(1e-3, (end["pitch"] - prev["pitch"]) / 0.02)
    tau = max(0.5, (90.0 - end["pitch"]) / rate)
    pitch = end["pitch"] + (90.0 - end["pitch"]) * (1 - math.exp(-dt / tau))
    z = math.exp(lz + tg[2] * dt)
    return dict(pos=(end["pos"][0], end["pos"][1], z), yaw=end["yaw"], pitch=min(90.0, pitch), hfov=hfov, roll=0.0)


def cam_basis(cam):
    yaw, pitch, roll = math.radians(cam["yaw"]), math.radians(cam["pitch"]), math.radians(cam.get("roll", 0.0))
    f = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), -math.sin(pitch)])
    r = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    u = np.cross(r, f)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        r, u = r * c + u * s, u * c - r * s
    F = (W / 2) / math.tan(math.radians(cam["hfov"]) / 2)
    return np.array(cam["pos"], float), r, u, f, F


class Proj:
    """vectorised world -> design-px projection for one camera"""

    def __init__(self, cam):
        self.cam = cam
        self.C, self.r, self.u, self.f, self.F = cam_basis(cam)
        sh = cam.get("shake", (0.0, 0.0))
        self.cx, self.cy = W / 2 + sh[0], H / 2 + sh[1]

    def cam_xyz(self, P):
        P = np.asarray(P, float)
        d = P - self.C
        return d @ self.r, d @ self.u, d @ self.f

    def __call__(self, P):
        """P (..., 3) -> sx, sy, depth"""
        x, y, z = self.cam_xyz(P)
        zz = np.maximum(z, 1e-3)
        return self.cx + self.F * x / zz, self.cy - self.F * y / zz, z

    def pt(self, x, y, z):
        sx, sy, d = self(np.array([x, y, z], float))
        return float(sx), float(sy), float(d)


# ============================================================== hashing (shared by numba + python)
@njit(cache=True, inline="always")
def ih(i, j, k):
    h = (i * 374761393 + j * 668265263 + k * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0


@njit(cache=True, inline="always")
def seed_xy(i, j, cell, jit):
    return ((i + 0.5 + jit * (ih(i, j, 1) - 0.5)) * cell, (j + 0.5 + jit * (ih(i, j, 2) - 0.5)) * cell)


@njit(cache=True)
def _grid_seeds(i0, j0, n, cell, jit):
    X = np.empty((n, n), np.float64)
    Y = np.empty((n, n), np.float64)
    R = np.empty((n, n), np.float64)
    for a in range(n):
        for b in range(n):
            x, y = seed_xy(i0 + a, j0 + b, cell, jit)
            X[a, b] = x
            Y[a, b] = y
            R[a, b] = ih(i0 + a, j0 + b, 7)
    return X, Y, R


# ============================================================== population + activation (phase transition)
class World:
    """survivors on plates; activation times from first-passage percolation on the plate graph"""

    def __init__(self, d):
        self.__dict__.update(d)
        self.n = len(self.x)

    @staticmethod
    def build():
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        from scipy.spatial import cKDTree
        n = int(math.ceil(2 * POP_R1 / CELL)) + 4
        i0 = j0 = -n // 2
        X, Y, Rh = _grid_seeds(i0, j0, n, CELL, JIT)
        r = np.hypot(X, Y)
        occ = (r > POP_R0) & (r < POP_R1) & (Rh < 0.66)
        for kx_, ky_, kr_ in RUIN_KEEPOUT:
            occ &= np.hypot(X - kx_, Y - ky_) > kr_
        ii, jj = np.nonzero(occ)
        x, y = X[occ], Y[occ]
        rr = r[occ]
        N = len(x)
        rng = np.random.default_rng(7007)
        # kinds: 0 nation-of-one (crown), 1 cabal-of-one (hood + candle), 2 crackpot (toga, rock), 3 feed-zombie (phone)
        kind = rng.choice(4, N, p=[0.34, 0.22, 0.18, 0.26]).astype(np.int8)
        height = rng.normal(1.68, 0.09, N).clip(1.45, 1.95).astype(np.float32)
        glow = rng.integers(0, len(GLOW_COLS), N).astype(np.int8)
        glow[(kind == 1) | (kind == 2)] = -1                     # hooded mystics and crackpots have no screens
        facing = np.where(rng.random(N) < 0.5, -1, 1).astype(np.int8)
        seed = rng.integers(0, 10_000, N).astype(np.int32)
        # --- attention: they look up at the Flag, a ripple spreading outward
        look = (T_LOOK + 0.05 + 0.55 * np.clip((rr - POP_R0) / 400.0, 0, 1) ** 0.7
                + 0.35 * rng.random(N) ** 2).astype(np.float32)
        # --- convergence: first-passage percolation on the plate graph. Front speed v(r) = K max(r, R_ONE):
        # a few individuals first ("one by one"), then an exponentially accelerating front with a rough,
        # KPZ-like edge, plus nucleation just ahead of it (domains that merge) - a phase transition.
        K, R_ONE = 1.8, 40.0
        tree = cKDTree(np.stack([x, y], 1))
        pairs = tree.query_pairs(CELL * 1.75, output_type="ndarray")
        a, b = pairs[:, 0], pairs[:, 1]
        dl = np.hypot(x[a] - x[b], y[a] - y[b])
        rm = 0.5 * (rr[a] + rr[b])
        sm = np.clip((rm - 32.0) / 30.0, 0, 1)
        sm = sm * sm * (3 - 2 * sm)
        v = 11.0 + (K * np.maximum(rm, R_ONE) - 11.0) * sm       # walking pace by the hill, then the rush
        w = dl / v * rng.lognormal(0.0, 0.45, len(a))
        # "one by one": the first raiser on the cue, then five more near it, then the inner ring floods in
        first = _first_raiser(x, y)
        inner = np.nonzero(rr < POP_R0 + 8.0)[0]
        inner = inner[inner != first]
        dfirst = np.hypot(x[inner] - x[first], y[inner] - y[first])
        vis = _in_view(x[inner], y[inner])
        lead = inner[vis][np.argsort(dfirst[vis])][:5]
        rest = np.setdiff1d(inner, lead)
        rest = rest[rng.random(len(rest)) < 0.5]
        inner = np.concatenate([[first], lead, rest])
        t_in = np.concatenate([[0.0], np.array([0.14, 0.25, 0.33, 0.39, 0.44])[:len(lead)],
                               0.46 + 0.16 * rng.random(len(rest))])
        # the front then takes over from radius R_ONE at ~150.5
        cand = np.nonzero((rr > 70) & (rr < 700))[0]
        nuc = rng.choice(cand, 26, replace=False)
        t_front = 0.5 + np.log(rr[nuc] / R_ONE) / K
        t_nuc = t_front - rng.uniform(0.1, 0.35, len(nuc))
        src = np.concatenate([inner, nuc])
        tsrc = np.concatenate([t_in, t_nuc])
        rows = np.concatenate([a, b, np.full(len(src), N)])
        cols = np.concatenate([b, a, src])
        vals = np.concatenate([w, w, tsrc])
        G = coo_matrix((vals, (rows, cols)), shape=(N + 1, N + 1)).tocsr()
        dist = dijkstra(G, directed=True, indices=N)[:N]
        act = (T_CONV + dist).astype(np.float32)
        act[first] = T_CONV            # THE first one raises exactly on the cue
        act[~np.isfinite(act)] = np.inf
        order = np.argsort(act)
        d = dict(x=x.astype(np.float32), y=y.astype(np.float32), r=rr.astype(np.float32), gi=(ii + i0).astype(np.int32),
                 gj=(jj + j0).astype(np.int32), kind=kind, height=height, glow=glow, facing=facing, seed=seed,
                 look=look, act=act, order=order.astype(np.int32))
        return World(d)

    def grid_index(self):
        """(GRID, g0): GRID[i - g0, j - g0] = survivor index on plate cell (i, j), or -1"""
        if getattr(self, "_grid", None) is None:
            g0 = int(min(self.gi.min(), self.gj.min())) - 2
            n = int(max(self.gi.max(), self.gj.max())) - g0 + 3
            G = np.full((n, n), -1, np.int32)
            G[self.gi - g0, self.gj - g0] = np.arange(self.n, dtype=np.int32)
            self._grid = (G, g0)
        return self._grid

    def save(self, path):
        np.savez_compressed(path, **{k: v for k, v in self.__dict__.items()
                                     if isinstance(v, np.ndarray) and not k.startswith("_")})

    @staticmethod
    def load(path):
        z = np.load(path)
        return World({k: z[k] for k in z.files})


def _in_view(x, y, T=T_CONV):
    pr = Proj(camera_at(T))
    P = np.stack([x, y, np.full_like(x, 1.0)], 1).astype(float)
    sx, sy, z = pr(P)
    return (z > 5) & (sx > W * 0.12) & (sx < W * 0.88) & (sy > H * 0.25) & (sy < H * 0.95)


def _first_raiser(x, y):
    """the survivor nearest to the hill inside the 150.0 camera view (so the first raise is seen)"""
    cam = camera_at(T_CONV)
    pr = Proj(cam)
    P = np.stack([x, y, np.full_like(x, 1.0)], 1).astype(float)
    sx, sy, z = pr(P)
    ok = (z > 5) & (sx > W * 0.3) & (sx < W * 0.7) & (sy > H * 0.35) & (sy < H * 0.9)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return int(np.argmin(np.hypot(x, y)))
    return int(idx[np.argmin(np.hypot(x[idx], y[idx]))])


_WORLD = {}
WORLD_VERSION = 20


def get_world(S=None):
    if "w" in _WORLD:
        return _WORLD["w"]
    d = CACHE / "s07a"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"world_v{WORLD_VERSION}.npz"
    if p.exists():
        w = World.load(p)
    else:
        w = World.build()
        w.save(p)
    _WORLD["w"] = w
    return w


# ============================================================== the per-pixel shader (sky + ground plates)
@njit(cache=True, inline="always")
def _mix3(a0, a1, a2, b0, b1, b2, t):
    return a0 + (b0 - a0) * t, a1 + (b1 - a1) * t, a2 + (b2 - a2) * t


@njit(cache=True, inline="always")
def _sstep(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, inline="always")
def _vnoise1(x, seed):
    i = math.floor(x)
    f = x - i
    u = f * f * (3 - 2 * f)
    a = ih(int(i), seed, 3) * 2 - 1
    b = ih(int(i) + 1, seed, 3) * 2 - 1
    return a + (b - a) * u


@njit(cache=True, inline="always")
def _vnoise2(x, y, seed):
    i = math.floor(x)
    j = math.floor(y)
    fx = x - i
    fy = y - j
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    ii = int(i)
    jj = int(j)
    a = ih(ii, jj, seed)
    b = ih(ii + 1, jj, seed)
    c = ih(ii, jj + 1, seed)
    d = ih(ii + 1, jj + 1, seed)
    return (a + (b - a) * ux) * (1 - uy) + (c + (d - c) * ux) * uy


# sky stops: elevation (deg) -> colour away from sun / towards sun
_SKY_E = np.array([0.0, 1.2, 4.0, 10.0, 22.0, 45.0, 90.0])
_SKY_FAR = np.array([(0.70, 0.50, 0.60), (0.50, 0.37, 0.52), (0.32, 0.26, 0.45), (0.19, 0.18, 0.37),
                     (0.10, 0.11, 0.26), (0.055, 0.065, 0.165), (0.03, 0.04, 0.105)])
_SKY_NEAR = np.array([(1.00, 0.84, 0.78), (0.94, 0.60, 0.58), (0.62, 0.37, 0.50), (0.32, 0.23, 0.43),
                      (0.14, 0.13, 0.30), (0.06, 0.07, 0.17), (0.03, 0.04, 0.105)])


@njit(cache=True)
def _sky(e, near, dawn, SE, SF, SN):
    """e = elevation deg (>=0), near = 0..1 towards-sun weight, dawn = 0..1 spread of the gradient"""
    # early dawn: the gradient is compressed into a thin band on the horizon
    comp = 0.02 + 0.98 * dawn * dawn
    ee = e / comp
    n = SE.shape[0]
    k = n - 2
    for q in range(n - 1):
        if ee < SE[q + 1]:
            k = q
            break
    t = (ee - SE[k]) / (SE[k + 1] - SE[k])
    if t > 1.0:
        t = 1.0
    if t < 0.0:
        t = 0.0
    r0 = SF[k, 0] + (SF[k + 1, 0] - SF[k, 0]) * t
    g0 = SF[k, 1] + (SF[k + 1, 1] - SF[k, 1]) * t
    b0 = SF[k, 2] + (SF[k + 1, 2] - SF[k, 2]) * t
    r1 = SN[k, 0] + (SN[k + 1, 0] - SN[k, 0]) * t
    g1 = SN[k, 1] + (SN[k + 1, 1] - SN[k, 1]) * t
    b1 = SN[k, 2] + (SN[k + 1, 2] - SN[k, 2]) * t
    r, g, b = _mix3(r0, g0, b0, r1, g1, b1, near)
    # before full dawn the upper sky is still night
    lift = 0.25 + 0.75 * dawn
    fall = math.exp(-ee / (6.0 + 30.0 * dawn))
    k2 = lift * (0.35 + 0.65 * fall) if dawn < 1.0 else 1.0
    k2 = k2 + (1.0 - k2) * dawn * dawn
    return r * k2, g * k2, b * k2


@njit(parallel=True, fastmath=True, cache=True)
def shade(out_a, out_s, out_d, P, SE, SF, SN, ALB, GRID, g0, GLI, LOOK, ACT, GC):
    h, w = out_a.shape[0], out_a.shape[1]
    Cx, Cy, Cz = P[0], P[1], P[2]
    rx, ry, rz = P[3], P[4], P[5]
    ux, uy, uz = P[6], P[7], P[8]
    fx, fy, fz = P[9], P[10], P[11]
    Fp, cx, cy = P[12], P[13], P[14]
    sx_, sy_, sz_ = P[15], P[16], P[17]
    sky_k, line_k, dawn, sun_k, amb_k = P[18], P[19], P[20], P[21], P[22]
    cell, jit, T, hazeL, sun_rad = P[23], P[24], P[25], P[26], P[27]
    sunh = math.sqrt(sx_ * sx_ + sy_ * sy_) + 1e-9
    shx, shy = sx_ / sunh, sy_ / sunh
    sun_e = math.degrees(math.asin(sz_))
    gn = GRID.shape[0]
    pool_k = P[28]
    for py in prange(h):
        for px in range(w):
            ox = (px + 0.5 - cx) / Fp
            oy = (py + 0.5 - cy) / Fp
            dx = fx + rx * ox - ux * oy
            dy = fy + ry * ox - uy * oy
            dz = fz + rz * ox - uz * oy
            dl = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= dl
            dy /= dl
            dz /= dl
            hl = math.sqrt(dx * dx + dy * dy) + 1e-9
            near = (dx * shx + dy * shy) / hl
            near = max(0.0, near)
            near = near * near * near * near
            if dz >= -1e-5:
                # ------------------------------------------------ sky
                e = math.degrees(math.asin(min(1.0, dz)))
                r, g, b = _sky(e, near, dawn, SE, SF, SN)
                # the thin dawn line (horizon slit) at first light
                az = math.atan2(dx, dy)
                ln = math.exp(-e / (0.04 + 0.4 * dawn)) * line_k * (0.35 + 0.65 * near) * (1 - dawn) ** 1.5
                r += ln * 1.0
                g += ln * 0.80
                b += ln * 0.74
                # stratus bands lit from below
                for q in range(3):
                    ec = 2.6 + q * 3.1 + 0.9 * _vnoise1(az * 3.0 + q * 7.1, 11 + q)
                    th = 0.28 + 0.22 * q + 0.35 * _vnoise1(az * 9.0 + q * 3.3, 21 + q)
                    th = max(0.0, th) * (0.6 + 0.4 * dawn)
                    dd = (e - ec) / (th + 1e-3)
                    if dd > -1.0 and dd < 1.0:
                        den = (1 - dd * dd) * _sstep(0.25, 0.75, _vnoise1(az * 14.0 + q * 5.0, 31 + q) * 0.5 + 0.5 + 0.25)
                        den *= dawn
                        lit = 0.5 + 0.5 * (-dd)        # underside brighter
                        cr = 0.20 + (1.00 * near + 0.55 * (1 - near)) * lit * 0.95
                        cg = 0.17 + (0.60 * near + 0.33 * (1 - near)) * lit * 0.95
                        cb = 0.30 + (0.60 * near + 0.45 * (1 - near)) * lit * 0.75
                        r, g, b = _mix3(r, g, b, cr * 0.8, cg * 0.8, cb * 0.8, den * 0.75)
                # last stars, fading as the dawn spreads
                stk = P[29]
                if stk > 0.01 and e > 2.0:
                    ga = math.degrees(az) * 3.0
                    ge = e * 3.0
                    ci = int(math.floor(ga))
                    cj = int(math.floor(ge))
                    if ih(ci, cj, 77) > 0.93:
                        cxs = (ci + 0.2 + 0.6 * ih(ci, cj, 78)) / 3.0
                        cys = (cj + 0.2 + 0.6 * ih(ci, cj, 79)) / 3.0
                        ddx = (math.degrees(az) - cxs) * math.cos(math.radians(e))
                        dde = e - cys
                        dpx = math.sqrt(ddx * ddx + dde * dde) * Fp / 57.29578
                        sb = max(0.0, 1.2 - dpx) * (0.25 + 0.75 * ih(ci, cj, 80)) ** 3
                        sb *= stk * _sstep(2.0, 9.0, e) * (0.8 + 0.2 * math.sin(T * 7.0 + ci))
                        sb /= max(sky_k, 0.05)          # stars do not follow the dawn exposure
                        r += sb * 0.9
                        g += sb * 0.9
                        b += sb * 1.0
                # sun: disc + glow
                cosang = dx * sx_ + dy * sy_ + dz * sz_
                ang = math.degrees(math.acos(min(1.0, max(-1.0, cosang))))
                gl = (math.exp(-ang / 0.7) * 0.30 + math.exp(-ang / 3.0) * 0.13 + math.exp(-ang / 12.0) * 0.06)
                glk = sky_k * (0.25 + 0.75 * _sstep(-3.0, 0.5, sun_e))
                r += gl * glk * 1.0
                g += gl * glk * 0.74
                b += gl * glk * 0.66
                aa = 1.2 / Fp * 57.29578
                disc = _sstep(sun_rad + aa, sun_rad - aa, ang)
                if disc > 0:
                    r += disc * 1.9
                    g += disc * 1.55
                    b += disc * 1.45
                out_a[py, px, 0] = r * sky_k + ln * 1.0 * (1 - sky_k)
                out_a[py, px, 1] = g * sky_k + ln * 0.80 * (1 - sky_k)
                out_a[py, px, 2] = b * sky_k + ln * 0.74 * (1 - sky_k)
                out_s[py, px, 0] = 0.0
                out_s[py, px, 1] = 0.0
                out_s[py, px, 2] = 0.0
                out_d[py, px] = 1e9
                continue
            # ---------------------------------------------------- ground
            t = -Cz / dz
            X = Cx + t * dx
            Y = Cy + t * dy
            out_d[py, px] = t
            fp = t / Fp / math.sqrt(max(-dz, 0.02))
            gx = X / cell
            gy = Y / cell
            ix = int(math.floor(gx))
            iy = int(math.floor(gy))
            # nearest seed
            best = 1e18
            bi = 0
            bj = 0
            bx = 0.0
            by = 0.0
            plr = 0.0
            plg = 0.0
            plb = 0.0
            for a in range(-1, 2):
                for c in range(-1, 2):
                    qx, qy = seed_xy(ix + a, iy + c, cell, jit)
                    d2 = (qx - X) * (qx - X) + (qy - Y) * (qy - Y)
                    # light pools: each survivor's private feed lights their plate; later the Flag's gold
                    gi_ = ix + a - g0
                    gj_ = iy + c - g0
                    if pool_k > 0 and gi_ >= 0 and gj_ >= 0 and gi_ < gn and gj_ < gn:
                        sid = GRID[gi_, gj_]
                        if sid >= 0:
                            gk = 1.0 - min(1.0, max(0.0, (T - LOOK[sid]) / 0.4))
                            if gk > 0 and GLI[sid] >= 0:
                                ff = 1.6 / (d2 + 1.6)
                                ff = ff * math.sqrt(ff) * gk * 0.55
                                q_ = GLI[sid]
                                plr += GC[q_, 0] * ff
                                plg += GC[q_, 1] * ff
                                plb += GC[q_, 2] * ff
                            ta = T - ACT[sid]
                            if ta > 0:
                                pk = min(1.0, ta / 0.25)
                                fl = 0.36 + 1.1 * math.exp(-ta / 0.22)
                                ff = 4.2 / (d2 + 4.2)
                                ff = ff * ff * pk * fl
                                plr += 1.00 * ff
                                plg += 0.76 * ff
                                plb += 0.30 * ff
                    if d2 < best:
                        best = d2
                        bi = ix + a
                        bj = iy + c
                        bx = qx
                        by = qy
            # distance to nearest edge
            ed = 1e9
            nx_ = 0.0
            ny_ = 0.0
            for a in range(-2, 3):
                for c in range(-2, 3):
                    if a == 0 and c == 0:
                        continue
                    qx, qy = seed_xy(bi + a, bj + c, cell, jit)
                    vx = qx - bx
                    vy = qy - by
                    vl = math.sqrt(vx * vx + vy * vy) + 1e-9
                    vx /= vl
                    vy /= vl
                    mx = 0.5 * (qx + bx)
                    my = 0.5 * (qy + by)
                    de = (mx - X) * vx + (my - Y) * vy
                    if de < ed:
                        ed = de
                        nx_ = vx
                        ny_ = vy
            hsh = ih(bi, bj, 5)
            k = int(hsh * ALB.shape[0]) % ALB.shape[0]
            ar = ALB[k, 0]
            ag = ALB[k, 1]
            ab = ALB[k, 2]
            vv = 0.92 + 0.16 * ih(bi, bj, 6)
            # large-scale stains of dried slop (desaturated pastel residue) + mottling
            n1 = _vnoise2(X / 60.0, Y / 60.0, 41)
            n2 = _vnoise2(X / 17.0, Y / 17.0, 43)
            n3 = _vnoise2(X / 2.3, Y / 2.3, 47)
            mott = 0.82 + 0.3 * n1 + 0.12 * n2 + 0.08 * n3 * _sstep(0.4, 0.08, fp)
            st = _sstep(0.62, 0.8, n2 * 0.6 + n1 * 0.4)
            ar = (ar + (0.66 - ar) * st * 0.35) * vv * mott
            ag = (ag + (0.55 - ag) * st * 0.35) * vv * mott
            ab = (ab + (0.62 - ab) * st * 0.35) * vv * mott
            # cracks (anti-aliased, energy preserving at distance)
            wc = 0.07 + 0.08 * ih(bi, bj, 8)
            fw = max(wc, fp * 1.2)
            cov = (fw * 0.5 - ed) / (fp * 1.2 + 1e-6) + 0.5
            cov = min(1.0, max(0.0, cov)) * (wc / fw)
            # plate rim: curled edge catching the low sun on the sun-facing side
            face = nx_ * shx + ny_ * shy
            rw = max(0.16, fp)
            rim = math.exp(-ed / rw) * (0.16 / rw) * max(0.0, face)
            dark = math.exp(-ed / rw) * (0.16 / rw) * max(0.0, -face)
            # lighting
            amb_r, amb_g, amb_b = 0.30 * amb_k, 0.31 * amb_k, 0.52 * amb_k
            cr_ = 1.0 - cov * 0.55
            la = (1.0 - 0.18 * dark) * cr_
            ls = sun_k * (0.2 + 1.5 * rim) * (1.0 - cov)
            # haze toward the horizon
            fog = 1.0 - math.exp(-t / hazeL)
            e0 = 0.35
            hr, hg, hb = _sky(e0, near, dawn, SE, SF, SN)
            hz = 0.62 + 0.38 * near
            # the plant's pulse: a ring of warm light racing out across the plates from the hill (142.0)
            pdt = T - P[30]
            if pdt > 0.0 and pdt < 3.0:
                rr_ = math.sqrt(X * X + Y * Y)
                wq = (rr_ - 22.0 - 75.0 * pdt) / (3.0 + 7.0 * pdt)
                ring = math.exp(-wq * wq) * math.exp(-pdt / 0.9) * 0.55 * (1.0 - cov)
                plr += 1.00 * ring
                plg += 0.70 * ring
                plb += 0.40 * ring
            pf = (1 - fog) * max(pool_k, 1.0 if pdt > 0.0 and pdt < 3.0 else 0.0) * (1.0 - 0.5 * cov)
            out_a[py, px, 0] = (ar * amb_r * la * (1 - fog) + fog * hr * hz) * sky_k + ar * plr * pf * 1.6
            out_a[py, px, 1] = (ag * amb_g * la * (1 - fog) + fog * hg * hz) * sky_k + ag * plg * pf * 1.6
            out_a[py, px, 2] = (ab * amb_b * la * (1 - fog) + fog * hb * hz) * sky_k + ab * plb * pf * 1.6
            out_s[py, px, 0] = ar * 1.00 * ls * (1 - fog) * sky_k
            out_s[py, px, 1] = ag * 0.60 * ls * (1 - fog) * sky_k
            out_s[py, px, 2] = ab * 0.52 * ls * (1 - fog) * sky_k


def shade_world(fc, cam, T):
    """run the shader -> (amb (h,w,3), sun (h,w,3), depth (h,w))"""
    Cp, r, u, f, F = cam_basis(cam)
    sh = cam.get("shake", (0.0, 0.0))
    s = fc.s
    L = light_curves(T)
    sd = sun_dir(T)
    P = np.array([*Cp, *r, *u, *f, F * s, (W / 2 + sh[0]) * s, (H / 2 + sh[1]) * s, *sd,
                  L["sky"], L["line"], L["dawn"], L["sun"], L["amb"],
                  CELL, JIT, T, 1700.0, 0.62, L["glow"], L["stars"], T_PLANT], np.float64)
    a = np.empty((fc.h, fc.w, 3), np.float32)
    b = np.empty((fc.h, fc.w, 3), np.float32)
    d = np.empty((fc.h, fc.w), np.float32)
    wd = get_world()
    grid, g0 = wd.grid_index()
    shade(a, b, d, P, _SKY_E, _SKY_FAR, _SKY_NEAR, PLATE_ALB, grid, g0, wd.glow, wd.look, wd.act, GLOW_COLS)
    return a, b, d


# ============================================================== the rubble hill + ruins (low-poly, flat shaded, true 3D)
def mound_z(x, y):
    """ground height of Lutie's rubble hill (metres)"""
    r = np.hypot(x, y)
    re = np.sqrt(r * r + 0.9 ** 2) - 0.9
    ang = np.arctan2(y, x)
    wob = 1.0 + 0.16 * np.sin(3 * ang + 0.7) + 0.09 * np.sin(5 * ang + 2.1) + 0.05 * np.sin(9 * ang + 0.3)
    q = np.clip(1.0 - re / (HILL_R * wob), 0.0, 1.0)
    return HILL_H * q ** 1.3


class Mesh:
    """flat-shaded convex polygons in world space"""

    def __init__(self):
        self.polys, self.alb, self.two, self.emis, self.grp, self.obj = [], [], [], [], [], []
        self.cur = 0

    def new_obj(self):
        self.cur += 1
        return self.cur

    def add(self, pts, alb, two_sided=False, emis=(0.0, 0.0, 0.0), grp=0):
        self.obj.append(self.cur)
        self.polys.append(np.asarray(pts, float))
        self.alb.append(col(alb) if isinstance(alb, str) else tuple(alb))
        self.two.append(two_sided)
        self.emis.append(tuple(emis))
        self.grp.append(grp)

    def box(self, c, dims, R, alb, grp=0, emis=(0, 0, 0)):
        """oriented box: centre c, dims (a, b, h), rotation matrix R (columns = local axes)"""
        self.new_obj()
        a, b, h = np.asarray(dims) / 2
        L = np.array([[-a, -b, -h], [a, -b, -h], [a, b, -h], [-a, b, -h], [-a, -b, h], [a, -b, h], [a, b, h], [-a, b, h]])
        V = np.asarray(c) + L @ R.T
        for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            self.add(V[list(f)], alb, grp=grp, emis=emis)

    def prism(self, p0, p1, rad, k, alb, grp=0, phase=0.0):
        """k-sided closed prism (broken column) from p0 to p1"""
        self.new_obj()
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        ax = p1 - p0
        ax /= np.linalg.norm(ax)
        t = np.cross(ax, [0, 0, 1.0])
        if np.linalg.norm(t) < 1e-6:
            t = np.array([1.0, 0, 0])
        t /= np.linalg.norm(t)
        b = np.cross(ax, t)
        ring = [t * math.cos(phase + 2 * math.pi * q / k) * rad + b * math.sin(phase + 2 * math.pi * q / k) * rad
                for q in range(k)]
        A = [p0 + v for v in ring]
        B = [p1 + v for v in ring]
        for q in range(k):
            self.add([A[q], A[(q + 1) % k], B[(q + 1) % k], B[q]][::-1], alb, grp=grp)
        self.add(A, alb, grp=grp)
        self.add(B[::-1], alb, grp=grp)

    def finalize(self):
        n = len(self.polys)
        self.cnt = np.array([len(p) for p in self.polys])
        self.off = np.concatenate([[0], np.cumsum(self.cnt)])
        self.V = np.concatenate(self.polys, 0)
        cen, nrm = [], []
        for p in self.polys:
            c0 = p.mean(0)
            nn = np.zeros(3)
            for q in range(len(p)):   # Newell normal
                a, b = p[q], p[(q + 1) % len(p)]
                nn += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
            nn /= np.linalg.norm(nn) + 1e-12
            cen.append(c0)
            nrm.append(nn)
        self.cen = np.array(cen)
        self.nrm = np.array(nrm)
        self.alb = np.array(self.alb, float)
        self.two = np.array(self.two)
        self.emis = np.array(self.emis, float)
        self.grp = np.array(self.grp)
        self.obj = np.array(self.obj)
        # orient normals outward: mound faces up, closed solids away from their object centre
        for o in np.unique(self.obj):
            sel = np.nonzero(self.obj == o)[0]
            oc = self.cen[sel].mean(0)
            for k in sel:
                if self.grp[k] == 1:
                    if self.nrm[k, 2] < 0:
                        self.nrm[k] *= -1
                elif not self.two[k] and np.dot(self.nrm[k], self.cen[k] - oc) < 0:
                    self.nrm[k] *= -1
        return self


def _rot(yaw, tilt_x, tilt_y):
    cz, sz = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(tilt_x), math.sin(tilt_x)
    cy, sy = math.cos(tilt_y), math.sin(tilt_y)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rz @ Rx @ Ry


RUBBLE = [(0.36, 0.36, 0.44), (0.42, 0.41, 0.48), (0.31, 0.31, 0.39), (0.47, 0.45, 0.50), (0.38, 0.35, 0.43)]
MARBLE = (0.80, 0.78, 0.80)
IRON = (0.33, 0.33, 0.40)


def build_hill(seed=11):
    rng = np.random.default_rng(seed)
    m = Mesh()
    m.new_obj()
    # --- mound: irregular Delaunay triangulation (no radial starburst)
    from scipy.spatial import Delaunay
    pts = [(0.0, 0.0)]
    nring = 56
    for q in range(nring):
        a_ = 2 * math.pi * (q + rng.uniform(-0.3, 0.3)) / nring
        wob = 1.0 + 0.16 * math.sin(3 * a_ + 0.7) + 0.09 * math.sin(5 * a_ + 2.1) + 0.05 * math.sin(9 * a_ + 0.3)
        pts.append((HILL_R * 1.04 * wob * math.cos(a_), HILL_R * 1.04 * wob * math.sin(a_)))
    while len(pts) < 300:
        r = HILL_R * 0.84 * math.sqrt(rng.random()) ** 1.25
        a_ = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a_), r * math.sin(a_)
        if min(math.hypot(x - px_, y - py_) for px_, py_ in pts) > 0.55 + 0.09 * r:
            pts.append((x, y))
    pts = np.array(pts)
    zz = mound_z(pts[:, 0], pts[:, 1]) + rng.uniform(-0.25, 0.25, len(pts)) * np.clip(1 - np.hypot(*pts.T) / HILL_R, 0, 1)
    zz[1:nring + 1] = 0.0
    zz[0] = HILL_H
    P3 = np.concatenate([pts, np.maximum(zz, 0)[:, None]], 1)
    tri = Delaunay(pts)
    for t_ in tri.simplices:
        alb = tuple(c * rng.uniform(0.93, 1.05) for c in (0.38, 0.37, 0.46))
        m.add(P3[t_], alb, grp=1)
    # --- rubble strewn over the mound (kept off Lutie's run-up strip, Crocus's climb and the plant spot)
    c0 = camera_at(141.0)
    yw = math.radians(c0["yaw"])
    ex = np.array([math.cos(yw), -math.sin(yw)])
    fw = np.array([math.sin(yw), math.cos(yw)])

    def clear(x, y, rad):
        px_, py_ = x * ex[0] + y * ex[1], x * fw[0] + y * fw[1]
        if -10.5 < px_ < 1.0 and -1.1 - rad < py_ < 0.9 + rad:
            return False
        if abs(px_ + 1.75) < 1.0 + rad and 1.2 < py_ < 6.0:
            return False
        return math.hypot(x, y) > 0.9 + rad
    placed = 0
    while placed < 230:
        r = HILL_R * (rng.random() ** 0.75) * 0.97
        ang = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(ang), r * math.sin(ang)
        big = rng.random() < 0.12 and r > 8.0
        s_ = rng.uniform(0.9, 2.2) if big else rng.uniform(0.2, 0.8)
        if r < 6.0:
            s_ = min(s_, 0.3 + 0.08 * r)
        if not clear(x, y, s_ * 0.6):
            continue
        z = float(mound_z(x, y))
        if rng.random() < 0.55:
            dims = (s_ * rng.uniform(1.2, 2.4), s_ * rng.uniform(0.8, 1.5), s_ * rng.uniform(0.18, 0.4))
        else:
            dims = (s_ * rng.uniform(0.8, 1.3), s_ * rng.uniform(0.8, 1.3), s_ * rng.uniform(0.6, 1.1))
        R = _rot(rng.uniform(0, 6.3), rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7))
        roll = rng.random()
        alb = MARBLE if roll < 0.1 else (SLOPC if roll < 0.22 else RUBBLE[rng.integers(len(RUBBLE))])
        alb = tuple(c * rng.uniform(0.85, 1.12) for c in alb)
        m.box((x, y, z + dims[2] * 0.1), dims, R, alb, grp=2)
        placed += 1

    # --- crest silhouette pieces seen against the dawn (plane coords: along the plant camera's right / depth)
    def at(px_, py_):
        x, y = ex * px_ + fw * py_
        return x, y, float(mound_z(x, y))
    x, y, z = at(2.0, 1.2)
    m.prism((x, y, z - 0.3), (x + 0.1, y + 0.1, z + 1.25), 0.3, 8, MARBLE, grp=2)           # standing column stump
    x, y, z = at(-3.4, 2.4)
    m.box((x, y, z + 0.5), (0.3, 2.6, 1.9), _rot(yw + 0.3, 0.2, -0.55), RUBBLE[1], grp=2)     # tilted slab
    x, y, z = at(3.6, -0.3)
    m.box((x, y, z + 0.2), (1.8, 1.2, 0.5), _rot(0.9, 0.25, 0.15), RUBBLE[3], grp=2)
    x, y, z = at(-5.2, 1.6)
    m.prism((x, y, z + 0.4), (x - 1.5, y + 2.0, z - 0.2), 0.45, 8, MARBLE, grp=2, phase=0.4)  # fallen drum
    x, y, z = at(5.0, 1.6)
    m.box((x, y, z + 0.6), (2.4, 0.3, 1.6), _rot(yw - 0.2, 0.15, 0.6), (0.16, 0.17, 0.22), grp=2)   # dead screen
    return m.finalize()


SLOPC = (0.60, 0.52, 0.60)


def build_ruins(seed=12):
    """big story fragments strewn on the plain: a colossal crown shard (the Nation), a dome shard (the Faith),
    the marble philosopher's fallen column, a dead Egregore screen"""
    rng = np.random.default_rng(seed)
    m = Mesh()
    # fallen colossal column (the Philosophy)
    for s0, s1 in ((0.0, 7.0), (7.6, 12.5), (13.4, 21.0)):
        az = math.radians(64)
        ux, uy = math.sin(az), math.cos(az)
        bx, by = -40.0, 55.0
        m.prism((bx + ux * s0, by + uy * s0, 1.2), (bx + ux * s1, by + uy * s1, 1.25 + 0.1 * rng.random()), 1.45, 10,
                tuple(c * (0.95 + 0.08 * rng.random()) for c in MARBLE), grp=3, phase=rng.random())
    # a debris apron around the hill, then scattered rubble blocks on the plain
    for q in range(170):
        r = HILL_R * rng.uniform(0.95, 1.9)
        a = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        s_ = rng.uniform(0.25, 1.0)
        dims = (s_ * rng.uniform(0.8, 2.0), s_ * rng.uniform(0.7, 1.4), s_ * rng.uniform(0.25, 0.8))
        alb = RUBBLE[rng.integers(len(RUBBLE))] if rng.random() > 0.12 else MARBLE
        m.box((x, y, dims[2] * 0.3), dims, _rot(rng.uniform(0, 6.3), rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)),
              tuple(c * rng.uniform(0.85, 1.1) for c in alb), grp=5)
    for q in range(260):
        r = 30 + 170 * rng.random() ** 1.6
        a = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        s_ = rng.uniform(0.3, 1.3) * (1.6 if rng.random() < 0.1 else 1.0)
        dims = (s_ * rng.uniform(0.8, 2.0), s_ * rng.uniform(0.7, 1.4), s_ * rng.uniform(0.3, 0.9))
        alb = RUBBLE[rng.integers(len(RUBBLE))] if rng.random() > 0.1 else MARBLE
        m.box((x, y, dims[2] * 0.3), dims, _rot(rng.uniform(0, 6.3), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)),
              tuple(c * rng.uniform(0.85, 1.1) for c in alb), grp=5)
    return m.finalize()


RUIN_KEEPOUT = [(-33.0, 60.0, 5.0), (-37.0, 65.0, 5.0), (-31.0, 69.0, 5.0)]


# far horizon silhouettes (km away): drawn in 2D at the horizon, parallax-free
FAR_RUINS = [  # azimuth deg, distance m, kind, scale — the fallen memeplexes on the horizon
    (-88.0, 5200.0, "babel", 0.55),    # the broken stump of Babel, still smoking
    (-44.0, 4300.0, "dome", 0.45),     # the Faith's cracked dome
    (-54.0, 6400.0, "head", 0.5),      # the Philosophy's colossal marble head, toppled on its back
    (-101.0, 3900.0, "crown", 0.4),    # the Nation's crown, tipped on its side
    (-30.0, 7200.0, "babel2", 0.4),
]


def _far_shape(ctx, kind, x, y, s, t):
    """fill the silhouette of a fallen memeplex at horizon point (x, y) with metres->px scale s"""
    def P(pts):
        ctx.move_to(x + pts[0][0] * s, y + pts[0][1] * s)
        for px_, py_ in pts[1:]:
            ctx.line_to(x + px_ * s, y + py_ * s)
        ctx.close_path()
        ctx.fill()
    if kind in ("babel", "babel2"):
        # a leaning, spiralling stump with broken ledges
        pts = [(-95, 0)]
        n = 9
        for q in range(n):
            yy = -q * 58
            w_ = 95 - q * 7
            pts.append((-w_ - 10 * (q % 2), yy - 20))
            pts.append((-w_ + 6, yy - 52))
        pts += [(-30, -560), (-18, -600), (-4, -540), (8, -640), (22, -560), (30, -590), (38, -520)]
        for q in range(n - 1, -1, -1):
            yy = -q * 58
            w_ = 95 - q * 7
            pts.append((w_ + 4 + 8 * ((q + 1) % 2), yy - 50))
            pts.append((w_, yy - 18))
        pts.append((100, 0))
        P([(px_ + 0.08 * -py_, py_) for px_, py_ in pts])
    elif kind == "dome":
        # drum + great dome, broken open on the right, the lantern still standing
        pts = [(-190, 0), (-180, -60), (-160, -60), (-160, -80)]
        for a in np.linspace(math.pi, math.pi * 0.52, 12):
            pts.append((160 * math.cos(a), -80 - 150 * math.sin(a)))
        pts += [(-14, -228), (-14, -262), (0, -280), (14, -262), (14, -226)]
        for a in np.linspace(math.pi * 0.44, math.pi * 0.36, 3):
            pts.append((160 * math.cos(a), -80 - 150 * math.sin(a)))
        pts += [(66, -186), (58, -150), (84, -158), (78, -120), (110, -128), (104, -96), (140, -92), (150, -60),
                (186, -58), (196, 0)]
        P(pts)
    elif kind == "head":
        # a colossal marble head lying on its back: the face in profile along the top
        P([(-270, 0), (-266, -52), (-244, -96), (-196, -124), (-124, -134), (-62, -140), (-36, -156), (-18, -140),
           (8, -146), (44, -222), (58, -152), (72, -148), (84, -164), (96, -150), (108, -162), (124, -142),
           (146, -138), (164, -152), (180, -124), (206, -84), (222, -44), (246, -16), (272, 0)])
    else:
        # the Nation's crown, tipped on its side: band + five points tipped with balls
        ang = -0.28
        c_, s_ = math.cos(ang), math.sin(ang)
        R = lambda px_, py_: (px_ * c_ - py_ * s_, px_ * s_ + py_ * c_)
        band = [(-150, 0), (-150, -60), (150, -60), (150, 0)]
        tips = [(-150, -190), (-75, -160), (0, -205), (75, -160), (150, -190)]
        pts = [R(*band[0]), R(*band[1])]
        for k_, (tx, ty) in enumerate(tips):
            pts.append(R(tx, ty))
            if k_ < 4:
                pts.append(R(tx + 37, -95))
        pts += [R(*band[2]), R(*band[3])]
        low = max(py_ for _, py_ in pts) - 12.0            # sunk a little into the plain
        P([(px_, py_ - low) for px_, py_ in pts])
        for tx, ty in tips:
            bx, by = R(tx, ty - 16)
            ctx.arc(x + bx * s, y + (by - low) * s, 18 * s, 0, 2 * math.pi)
            ctx.fill()


# ============================================================== survivors: far splats (numba)
@njit(cache=True)
def splat_survivors(img, glow, P, X, Y, HH, GI, ACT, LOOK, GC, T, lod_lo, lod_hi, flagcol, body, haze):
    """draw distant survivors directly into img (over) and glow (additive). P = camera params"""
    h, w = img.shape[0], img.shape[1]
    Cx, Cy, Cz = P[0], P[1], P[2]
    rx, ry, rz = P[3], P[4], P[5]
    ux, uy, uz = P[6], P[7], P[8]
    fx, fy, fz = P[9], P[10], P[11]
    Fp, cx, cy = P[12], P[13], P[14]
    wdx, wdy = P[15], P[16]
    light = P[17]
    n = X.shape[0]
    for i in range(n):
        dx = X[i] - Cx
        dy = Y[i] - Cy
        dz = -Cz
        zc = dx * fx + dy * fy + dz * fz
        if zc < 1.0:
            continue
        xc = dx * rx + dy * ry + dz * rz
        yc = dx * ux + dy * uy + dz * uz
        sx = cx + Fp * xc / zc
        sy = cy - Fp * yc / zc
        size = HH[i] * Fp / zc
        if size > lod_hi or sx < -20 or sx > w + 20 or sy < -40 or sy > h + 60:
            continue
        wgt = 1.0
        if size > lod_lo:
            wgt = 1.0 - (size - lod_lo) / (lod_hi - lod_lo)
        # screen up vector per metre
        zc1 = zc + fz * 1.0
        xc1 = xc + rz * 1.0
        yc1 = yc + uz * 1.0
        upx = cx + Fp * xc1 / zc1 - sx
        upy = cy - Fp * yc1 / zc1 - sy
        ul = math.sqrt(upx * upx + upy * upy) + 1e-9
        kk = Fp / zc
        if ul < 0.62 * kk:
            s_ = 0.62 * kk / ul
            upx *= s_
            upy *= s_
        hgt = HH[i]
        # --- body: a short tapered stroke
        bw = max(0.55, 0.22 * size)
        hx = sx + upx * hgt * 0.92
        hy = sy + upy * hgt * 0.92
        x0 = int(math.floor(min(sx, hx) - bw - 1))
        x1 = int(math.ceil(max(sx, hx) + bw + 1))
        y0 = int(math.floor(min(sy, hy) - bw - 1))
        y1 = int(math.ceil(max(sy, hy) + bw + 1))
        vx = hx - sx
        vy = hy - sy
        vl2 = vx * vx + vy * vy + 1e-9
        a_body = 0.92 * wgt * min(1.0, size * 0.9)
        fog = 1.0 - math.exp(-zc / 1700.0)
        b0 = body[0] * (1 - fog) + haze[0] * fog
        b1 = body[1] * (1 - fog) + haze[1] * fog
        b2 = body[2] * (1 - fog) + haze[2] * fog
        for yy in range(max(0, y0), min(h, y1 + 1)):
            for xx in range(max(0, x0), min(w, x1 + 1)):
                qx = xx + 0.5 - sx
                qy = yy + 0.5 - sy
                tt = (qx * vx + qy * vy) / vl2
                if tt < 0.0:
                    tt = 0.0
                elif tt > 1.0:
                    tt = 1.0
                ex = qx - tt * vx
                ey = qy - tt * vy
                d = math.sqrt(ex * ex + ey * ey)
                rad = bw * (1.0 - 0.3 * tt) * 0.5 + 0.35
                cov = rad + 0.5 - d
                if cov <= 0:
                    continue
                if cov > 1:
                    cov = 1.0
                a = cov * a_body
                img[yy, xx, 0] = img[yy, xx, 0] * (1 - a) + b0 * a
                img[yy, xx, 1] = img[yy, xx, 1] * (1 - a) + b1 * a
                img[yy, xx, 2] = img[yy, xx, 2] * (1 - a) + b2 * a
        # --- phone / feed glow (until they look up)
        gk = 1.0 - min(1.0, max(0.0, (T - LOOK[i]) / 0.35))
        if gk > 0 and GI[i] >= 0:
            gx = sx + upx * hgt * 0.62 + (vx * 0.0)
            gy = sy + upy * hgt * 0.62
            gidx = GI[i]
            inten = 1.2 * gk * wgt * min(1.0, (size / 6.0) ** 2) * math.exp(-zc / 420.0)
            _add_dot(glow, gx, gy, max(0.6, 0.06 * size), GC[gidx, 0] * inten, GC[gidx, 1] * inten, GC[gidx, 2] * inten)
        # --- the Flag
        ta = T - ACT[i]
        if ta > 0:
            pop = min(1.0, ta / 0.22)
            pop = 1.0 - (1.0 - pop) * (1.0 - pop)
            pop = pop * (1.0 + 0.25 * math.sin(min(ta, 0.4) / 0.4 * math.pi))
            ph = ta * 6.0 + i * 0.37
            cxw = 0.42 + 0.05 * math.sin(ph)
            fx0 = sx + upx * (hgt + 1.25 * pop) + (wdx * cxw * kk) * pop
            fy0 = sy + upy * (hgt + 1.25 * pop) + (wdy * cxw * kk) * pop
            fr = max(0.5, 0.36 * kk) * pop
            # pole
            px0 = sx + upx * hgt * 0.75
            py0 = sy + upy * hgt * 0.75
            ptx = sx + upx * (hgt + 1.62 * pop)
            pty = sy + upy * (hgt + 1.62 * pop)
            _line_over(img, px0, py0, ptx, pty, max(0.35, 0.05 * kk), 0.85, 0.73, 0.55, 0.7 * wgt)
            _ell_over(img, fx0, fy0, fr * 1.15, fr * 0.8, flagcol[0] * light, flagcol[1] * light, flagcol[2] * light, wgt)
            fl = (2.2 * math.exp(-ta / 0.16) + 0.45) * min(1.0, (size / 6.0) ** 1.5)
            _add_dot(glow, fx0, fy0, max(0.8, fr * 1.2), 1.0 * fl * wgt, 0.85 * fl * wgt, 0.45 * fl * wgt)


@njit(cache=True, inline="always")
def _add_dot(buf, x, y, r, cr, cg, cb):
    h, w = buf.shape[0], buf.shape[1]
    R = int(math.ceil(r * 2.5)) + 1
    ix = int(x)
    iy = int(y)
    inv = 1.0 / (r * r)
    norm = 1.0 / max(1.0, r * r * 1.2)
    for yy in range(max(0, iy - R), min(h, iy + R + 1)):
        for xx in range(max(0, ix - R), min(w, ix + R + 1)):
            dx = xx + 0.5 - x
            dy = yy + 0.5 - y
            g = math.exp(-(dx * dx + dy * dy) * inv) * norm * 1.2
            if g > 0.003:
                buf[yy, xx, 0] += cr * g
                buf[yy, xx, 1] += cg * g
                buf[yy, xx, 2] += cb * g


@njit(cache=True, inline="always")
def _ell_over(img, x, y, rx, ry, cr, cg, cb, alpha):
    h, w = img.shape[0], img.shape[1]
    for yy in range(max(0, int(y - ry - 2)), min(h, int(y + ry + 2) + 1)):
        for xx in range(max(0, int(x - rx - 2)), min(w, int(x + rx + 2) + 1)):
            dx = (xx + 0.5 - x) / (rx + 0.5)
            dy = (yy + 0.5 - y) / (ry + 0.5)
            d = math.sqrt(dx * dx + dy * dy)
            cov = (1.0 - d) * (min(rx, ry) + 0.5) + 0.5
            if cov <= 0:
                continue
            a = min(1.0, cov) * alpha
            img[yy, xx, 0] = img[yy, xx, 0] * (1 - a) + cr * a
            img[yy, xx, 1] = img[yy, xx, 1] * (1 - a) + cg * a
            img[yy, xx, 2] = img[yy, xx, 2] * (1 - a) + cb * a


@njit(cache=True, inline="always")
def _line_over(img, x0, y0, x1, y1, wd, cr, cg, cb, alpha):
    h, w = img.shape[0], img.shape[1]
    vx = x1 - x0
    vy = y1 - y0
    vl2 = vx * vx + vy * vy + 1e-9
    for yy in range(max(0, int(min(y0, y1) - 2)), min(h, int(max(y0, y1) + 2) + 1)):
        for xx in range(max(0, int(min(x0, x1) - 2)), min(w, int(max(x0, x1) + 2) + 1)):
            qx = xx + 0.5 - x0
            qy = yy + 0.5 - y0
            tt = (qx * vx + qy * vy) / vl2
            tt = min(1.0, max(0.0, tt))
            ex = qx - tt * vx
            ey = qy - tt * vy
            d = math.sqrt(ex * ex + ey * ey)
            cov = wd * 0.5 + 0.5 - d
            if cov <= 0:
                continue
            a = min(1.0, cov) * alpha * min(1.0, wd + 0.3)
            img[yy, xx, 0] = img[yy, xx, 0] * (1 - a) + cr * a
            img[yy, xx, 1] = img[yy, xx, 1] * (1 - a) + cg * a
            img[yy, xx, 2] = img[yy, xx, 2] * (1 - a) + cb * a


@njit(cache=True)
def add_dots(buf, xs, ys, rs, cs):
    for i in range(xs.shape[0]):
        _add_dot(buf, xs[i], ys[i], rs[i], cs[i, 0], cs[i, 1], cs[i, 2])


# ============================================================== world lighting helpers (python side)
AMB_COL = np.array([0.30, 0.31, 0.52])
SUN_COL = np.array([1.00, 0.60, 0.52])
HOR_COL = np.array([0.62, 0.40, 0.44])
SHADOW_K = 0.9


def haze_color(T, near=0.3):
    L = light_curves(T)
    r, g, b = _sky(0.35, near, L["dawn"], _SKY_E, _SKY_FAR, _SKY_NEAR)
    return np.array([r, g, b]) * (0.62 + 0.38 * near) * L["sky"]


def lit_colors(alb, nrm, dist, T, L=None, sd=None):
    """flat-shaded colours for faces/objects: albedo (n,3), normals (n,3), distance (n,) -> (n,3)"""
    L = L or light_curves(T)
    sd = sun_dir(T) if sd is None else sd
    sh = sd[:2] / (np.linalg.norm(sd[:2]) + 1e-9)
    nz = nrm[:, 2:3]
    amb = AMB_COL * L["amb"] * (0.7 + 0.3 * nz)
    hor = HOR_COL * L["dawn"] * 0.75 * np.clip(nrm[:, :2] @ sh, 0, 1)[:, None]
    sun = SUN_COL * L["sun"] * 1.7 * np.clip(nrm @ sd, 0, 1)[:, None]
    c = alb * (amb + hor + sun)
    fog = (1 - np.exp(-dist / 1700.0))[:, None]
    hz = haze_color(T)
    return (c * (1 - fog) + hz * fog) * L["sky"]


# ============================================================== survivors (vector tiers)
CLOTH = [col("cloth_grey"), col("cloth_blue"), col("cloth_teal"), col("cloth_rust"), (0.36, 0.38, 0.46),
         (0.30, 0.34, 0.44)]
SKINS = [col("skin1"), col("skin2"), col("skin3"), col("skin4")]
KIND_COL = {1: col("cloth_plum"), 2: (0.66, 0.63, 0.70)}
PEWTER = (0.66, 0.68, 0.76)
RIG_KIND = {0: "citizen", 1: "cultist", 2: "crackpot", 3: "citizen"}

_CHAR = {}


def _char(kind, seed):
    from vx.chars import Character
    key = (kind, seed % 23)
    ch = _CHAR.get(key)
    if ch is None:
        st = dict(crown=True, crown_color=PEWTER) if kind == 0 else {}
        ch = Character(RIG_KIND[kind], seed=seed % 23, **st)
        _CHAR[key] = ch
    return ch


def survivor_state(i, wd, T):
    look = float(wd.look[i])
    act = float(wd.act[i])
    lk = seg(T, look, look + 0.55, ease_in_out)
    ta = T - act
    pop = clamp(ta / 0.3) if ta > 0 else 0.0
    return lk, ta, pop


def _pop_curve(ta):
    """flag appear: 0 -> overshoot -> 1"""
    if ta <= 0:
        return 0.0
    x = clamp(ta / 0.32)
    return 1 - (1 - x) ** 3 + 0.22 * math.sin(math.pi * x) * (1 - x * 0.5)


def survivor_look(wd, i):
    """(clothing colour, skin colour, hair colour) for survivor i — shared by glyph + rig tiers"""
    kind = int(wd.kind[i])
    sd = int(wd.seed[i])
    cloth = KIND_COL.get(kind, CLOTH[sd % len(CLOTH)])
    skin = SKINS[(sd // 7) % 4]
    hair = (0.8, 0.8, 0.82) if kind == 2 else ((0.13, 0.12, 0.16), (0.24, 0.18, 0.15), (0.42, 0.40, 0.44))[(sd // 3) % 3]
    return np.array(cloth), np.array(skin), np.array(hair)


_BODY_TROUSER = [(-0.17, 0.0), (-0.05, 0.0), (0.0, 0.44), (0.05, 0.0), (0.17, 0.0), (0.17, 0.5), (0.15, 0.62),
                 (0.21, 0.79), (0.16, 0.83), (0.06, 0.845), (-0.06, 0.845), (-0.16, 0.83), (-0.21, 0.79),
                 (-0.15, 0.62), (-0.17, 0.5)]
_BODY_ROBE = [(-0.26, 0.0), (0.26, 0.0), (0.19, 0.45), (0.16, 0.62), (0.21, 0.79), (0.16, 0.83), (0.06, 0.845),
              (-0.06, 0.845), (-0.16, 0.83), (-0.21, 0.79), (-0.16, 0.62), (-0.19, 0.45)]


def draw_survivor_glyph(ctx, wd, i, sx, sy, kx, upx, upy, T, lk, ta, face_hill, sun_side, lightk, rimk):
    """compact vector survivor for ~10..46 px figures (screen space). returns the raised hand position"""
    kind = int(wd.kind[i])
    hgt = float(wd.height[i])
    fac = int(wd.facing[i]) if lk < 0.5 else face_hill
    cloth, skin, hair = survivor_look(wd, i)
    fog, haze = FOGSTATE
    tone = lambda c: np.asarray(c) * lightk * (1 - fog) + haze * fog
    bc = tone(cloth)
    sk = tone(skin) * 0.9
    lift = 0.0
    if kind == 2:   # crackpot on his own rock
        lift = 0.34
        rp = [(-0.5, -0.02), (0.55, -0.02), (0.42, 0.3), (0.12, 0.37), (-0.36, 0.3)]
        ctx.move_to(sx + rp[0][0] * kx + upx * rp[0][1], sy + upy * rp[0][1])
        for a_, b_ in rp[1:]:
            ctx.line_to(sx + a_ * kx + upx * b_, sy + upy * b_)
        ctx.close_path()
        ctx.set_source_rgb(*tone((0.30, 0.30, 0.37)))
        ctx.fill()

    def P(a_, b_):
        return sx + a_ * kx * fac + upx * (b_ + lift), sy + upy * (b_ + lift)
    body = _BODY_ROBE if kind in (1, 2) else _BODY_TROUSER
    pts = [P(x_, y_ * hgt) for x_, y_ in body]
    ctx.move_to(*pts[0])
    for q in pts[1:]:
        ctx.line_to(*q)
    ctx.close_path()
    ctx.set_source_rgb(*bc)
    ctx.fill()
    if kind == 0:   # royal cape of a nation-of-one
        cp = [P(-0.19, 0.8 * hgt), P(-0.1, 0.82 * hgt), P(-0.22, 0.05 * hgt), P(-0.34, 0.08 * hgt)]
        ctx.move_to(*cp[0])
        for q in cp[1:]:
            ctx.line_to(*q)
        ctx.close_path()
        ctx.set_source_rgb(*tone((0.45, 0.20, 0.28)))
        ctx.fill()
    # rim on the sun side (after sunrise)
    if rimk > 0.02:
        side = sun_side * fac
        ctx.move_to(*P(0.17 * side, 0.03 * hgt))
        ctx.line_to(*P(0.16 * side, 0.6 * hgt))
        ctx.line_to(*P(0.21 * side, 0.79 * hgt))
        ctx.set_line_width(max(0.5, 0.05 * kx))
        ctx.set_source_rgba(1.0, 0.64, 0.58, min(1.0, rimk) * (1 - fog))
        ctx.stroke()
    # head: tips back to look up at the Flag
    hx, hy = P(0.02 + 0.03 * lk, hgt * 0.915 - 0.03 * (1 - lk))
    hr = 0.105 * kx
    if kind == 1:  # hood
        ctx.arc(hx, hy, hr * 1.28, 0, 2 * math.pi)
        ctx.set_source_rgb(*(bc * 0.8))
        ctx.fill()
        fx_, fy_ = hx + fac * hr * 0.42, hy + hr * 0.12 - upy * 0.0
        ctx.arc(fx_, fy_ - lk * hr * 0.25, hr * 0.6, 0, 2 * math.pi)
        ctx.set_source_rgb(*sk)
        ctx.fill()
    else:
        ctx.arc(hx, hy, hr, 0, 2 * math.pi)
        ctx.set_source_rgb(*sk)
        ctx.fill()
        ctx.arc(hx - fac * hr * 0.22, hy - hr * 0.18, hr * 0.86, math.pi * (0.95 if fac > 0 else 1.05), math.pi * 2.05)
        ctx.set_source_rgb(*tone(hair))
        ctx.fill()
    if kind == 0:  # crown
        cw, ch = hr * 0.95, hr * 0.8
        cy0 = hy - hr * 0.62
        ctx.move_to(hx - cw, cy0)
        for dx_, dy_ in ((-1, -1), (-0.5, -0.45), (0, -1.1), (0.5, -0.45), (1, -1), (1, 0)):
            ctx.line_to(hx + cw * dx_, cy0 + ch * dy_)
        ctx.close_path()
        ctx.set_source_rgb(*tone(np.array(PEWTER) * 1.5))
        ctx.fill()
    # arms
    ctx.set_line_width(max(0.7, 0.075 * kx))
    ctx.set_source_rgb(*(bc * 0.82))
    sh_ = P(0.14, hgt * 0.78)
    if ta > 0:
        r = clamp(ta / 0.28)
        hand = P(0.17 + 0.06 * (1 - r), hgt * (0.64 + 0.52 * ease_out(r)))
    elif kind == 2 and lk < 0.5:
        hand = P(0.15, hgt * 0.9)                      # cupped hands at the mouth, screaming into the void
    else:
        hand = P(0.19 * (1 - lk) + 0.1 * lk, hgt * (0.64 - 0.16 * lk))
    ctx.move_to(*sh_)
    ctx.line_to(*hand)
    ctx.stroke()
    sh2 = P(-0.14, hgt * 0.78)
    h2 = P(-0.12 - 0.05 * lk, hgt * (0.5 + (0.12 if (kind == 2 and lk < 0.5) else 0.0)))
    ctx.move_to(*sh2)
    ctx.line_to(*h2)
    ctx.stroke()
    return hand


def render_world(fc, cam, T, world=None, heroes=None, extras=None, return_layers=False):
    """Render the dawn plain for any camera. heroes: object with .items(pr, T) -> [(depth, fn(ctx))],
    .shadows(ctx, pr, T), .glows(pr, T) -> list of (x, y, r, (r,g,b)) in design px."""
    wd = world or get_world()
    s = fc.s
    pr = Proj(cam)
    L = light_curves(T)
    sd = sun_dir(T)
    sh_dir = -sd[:2] / (np.linalg.norm(sd[:2]) + 1e-9)          # shadows point away from the sun
    e_s = max(math.radians(1.6), math.asin(max(0.0, sd[2])) if sd[2] > 0 else math.radians(1.6))
    shadow_len = 1.0 / math.tan(e_s)
    shadow_alpha = L["sun"]
    waz = math.radians(wind_az(T))
    wvec = np.array([math.sin(waz), math.cos(waz), 0.0])
    lightk = float((0.10 + 0.16 * L["amb"] + 0.16 * L["sun"]) * L["sky"])

    amb, sun, depth = shade_world(fc, cam, T)
    HAZE_NOW[0] = haze_color(T, 0.3)

    # ------------------------------------------------ visible survivors (vectorised projection)
    X, Y, Hh = wd.x.astype(float), wd.y.astype(float), wd.height.astype(float)
    xc, yc, zc = pr.cam_xyz(np.stack([X, Y, np.zeros_like(X)], 1))
    ok = zc > 1.0
    zz = np.maximum(zc, 1e-3)
    sx = pr.cx + pr.F * xc / zz
    sy = pr.cy - pr.F * yc / zz
    size = Hh * pr.F * s / zz          # pixel height
    ok &= (sx > -80) & (sx < W + 80) & (sy > -60) & (sy < H + 260)
    LOD_LO, LOD_HI = 7.0, 10.0
    near = np.nonzero(ok & (size > LOD_LO))[0]
    near = near[~_behind_hill(pr, sx[near], sy[near], zc[near], size[near] / s)]

    # ------------------------------------------------ shadow mask (ground only)
    ssurf = cairo.ImageSurface(cairo.FORMAT_A8, fc.w, fc.h)
    sctx = cairo.Context(ssurf)
    sctx.scale(s, s)
    sctx.set_source_rgba(0, 0, 0, 1)
    if shadow_alpha > 0.01:
        _draw_mesh_shadows(sctx, pr, HILL_MESH, sd, whole=True)
        _draw_mesh_shadows(sctx, pr, RUIN_MESH, sd, whole=False)
        if len(near):
            _draw_survivor_shadows(sctx, pr, wd, near, T, sh_dir, shadow_len, wvec)
        if heroes is not None:
            heroes.shadows(sctx, pr, T, sd)
    ssurf.flush()
    a8 = np.ndarray((fc.h, ssurf.get_stride()), np.uint8, ssurf.get_data())[:, :fc.w]
    mask = a8[..., None].astype(np.float32) * (SHADOW_K * shadow_alpha / 255.0)
    base = amb + sun * (1.0 - mask)


    # ------------------------------------------------ far survivors: numba splats
    glowbuf = np.zeros_like(base)
    Cp, r_, u_, f_, F = cam_basis(cam)
    shk = cam.get("shake", (0.0, 0.0))
    wr = float(wvec @ r_)
    wu = float(wvec @ u_)
    P = np.array([*Cp, *r_, *u_, *f_, F * s, (W / 2 + shk[0]) * s, (H / 2 + shk[1]) * s, wr, -wu,
                  0.75 + 0.35 * L["sun"]], np.float64)
    flagcol = np.array(C["flag"], np.float64)
    body = np.array((0.08, 0.08, 0.13)) * (0.5 + 0.5 * L["amb"]) * L["sky"]
    splat_survivors(base, glowbuf, P, wd.x, wd.y, wd.height, wd.glow, wd.act, wd.look, GLOW_COLS, float(T),
                    LOD_LO, LOD_HI, flagcol, body, haze_color(T, 0.3))

    # ------------------------------------------------ painter's list
    items = []
    _mesh_items(items, pr, HILL_MESH, T, L, sd)
    _mesh_items(items, pr, RUIN_MESH, T, L, sd)
    for i in near:
        items.append((float(zc[i]), 1, int(i)))
    if heroes is not None:
        for d, fn in heroes.items(pr, T):
            items.append((float(d), 2, fn))
    if extras:
        for d, fn in extras:
            items.append((float(d), 2, fn))
    items.sort(key=lambda it: -it[0])

    # two passes: A = everything beyond the hill's far side, B = the hill, heroes and anything nearer.
    # glows of pass-A survivors are laid down before B, so the hill and the heroes occlude them.
    _, _, zhill = pr.pt(0.0, 0.0, HILL_TOP)
    split = zhill + HILL_R
    sun_side = 1 if float(sd @ r_) > 0 else -1
    hill_sx, _, _ = pr.pt(0.0, 0.0, HILL_TOP)
    gl_k = L["glow"]

    def glow_img(buf, glows):
        if glows:
            xs = np.array([q[0] * s for q in glows], np.float64)
            ys = np.array([q[1] * s for q in glows], np.float64)
            rs = np.array([max(0.8, q[2] * s) for q in glows], np.float64)
            cs = np.array([q[3] for q in glows], np.float64)
            add_dots(buf, xs, ys, rs, cs)
        h2, w2 = max(1, buf.shape[0] // 2), max(1, buf.shape[1] // 2)
        small = cv2.resize(buf, (w2, h2), interpolation=cv2.INTER_AREA)
        b = blur(small, 1.5 * s) * 1.3 + blur(small, 7 * s) * 1.1
        return buf * 0.8 + cv2.resize(b, (buf.shape[1], buf.shape[0]), interpolation=cv2.INTER_LINEAR)

    img = base + glow_img(glowbuf, []) * gl_k          # distant feeds (splats), occluded by everything drawn later
    alpha_all = np.zeros(img.shape[:2], np.float32)
    for part in (0, 1):
        cv = Canvas(fc)
        ctx = cv.ctx
        glows = []
        if part == 0:
            _draw_far_ruins(ctx, pr, T, L)
        for d, typ, obj in items:
            if (d > split) != (part == 0):
                continue
            if typ == 0:
                pts, c = obj
                ctx.move_to(*pts[0])
                for p in pts[1:]:
                    ctx.line_to(*p)
                ctx.close_path()
                ctx.set_source_rgb(*c)
                ctx.fill_preserve()
                ctx.set_line_width(0.6)
                ctx.stroke()
            elif typ == 1:
                _draw_survivor(ctx, pr, wd, obj, T, float(sx[obj]), float(sy[obj]), float(size[obj]) / s, wvec,
                               hill_sx, sun_side, lightk, L, glows, LOD_HI)
            else:
                obj(ctx)
        if part == 1 and heroes is not None:
            glows.extend(heroes.glows(pr, T))
        lay = cv.rgba()
        img = over(img, lay)
        alpha_all = alpha_all + lay[..., 3] * (1 - alpha_all)
        if glows:
            img = img + glow_img(np.zeros_like(base), glows)
            
    if return_layers:
        return img, dict(depth=depth, alpha=alpha_all, pr=pr)
    return img


def _behind_hill(pr, sx, sy, zc, size_d):
    """survivors completely hidden behind the mound (screen-space test against its projected core)"""
    if len(sx) == 0:
        return np.zeros(0, bool)
    M = HILL_MESH
    core = M.V[(M.grp[np.repeat(np.arange(len(M.cnt)), M.cnt)] == 1)]
    xs, ys, zs = pr(core)
    if zs.min() < 0.5:
        return np.zeros(len(sx), bool)
    hull = _hull([tuple(p) for p in np.round(np.stack([xs, ys], 1), 1).tolist()])
    if len(hull) < 3:
        return np.zeros(len(sx), bool)
    poly = np.array(hull, np.float32).reshape(-1, 1, 2)
    _, _, zh = pr.pt(0.0, 0.0, HILL_TOP)
    out = np.zeros(len(sx), bool)
    cand = np.nonzero(zc > zh + HILL_R * 0.5)[0]
    for k in cand:
        # the whole figure (feet .. head + raised flag) must be inside the silhouette, with a margin
        top = sy[k] - size_d[k] * 1.9
        if (cv2.pointPolygonTest(poly, (float(sx[k]), float(sy[k])), True) > 4 and
                cv2.pointPolygonTest(poly, (float(sx[k]), float(top)), True) > 4):
            out[k] = True
    return out


def _mesh_items(items, pr, M, T, L, sd):
    xs, ys, zs = pr(M.V)
    xc, yc, zc = pr.cam_xyz(M.cen)
    view = M.cen - pr.C
    nrm = M.nrm.copy()
    facing = np.einsum("ij,ij->i", nrm, view)
    flip = M.two & (facing > 0)
    nrm[flip] *= -1
    vis = (facing < 0) | M.two
    dist = np.linalg.norm(view, axis=1)
    cols = lit_colors(M.alb, nrm, dist, T, L, sd)
    vis &= zc > 0.5
    # every vertex of the face in front of the camera
    for k in np.nonzero(vis)[0]:
        a, b = M.off[k], M.off[k + 1]
        if np.any(zs[a:b] < 0.3):
            continue
        px = xs[a:b]
        py = ys[a:b]
        if px.max() < -50 or px.min() > W + 50 or py.max() < -50 or py.min() > H + 50:
            continue
        items.append((float(zc[k]), 0, (list(zip(px.tolist(), py.tolist())), tuple(np.clip(cols[k], 0, 4)))))


def _hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def ground_shadow_pts(P, sd, max_len=260.0):
    """project 3D points along the sun onto z=0"""
    P = np.asarray(P, float)
    k = np.minimum(P[:, 2] / max(sd[2], math.sin(math.radians(1.6))), max_len)
    return P[:, :2] - sd[None, :2] * k[:, None] / np.linalg.norm(sd[:2]) * np.linalg.norm(sd[:2])


def _poly_ground(ctx, pr, G):
    """G (k,2) ground points -> fill (clipped to be in front of the camera)"""
    P3 = np.concatenate([G, np.zeros((len(G), 1))], 1)
    xs, ys, zs = pr(P3)
    if np.all(zs < 0.5):
        return
    if np.any(zs < 0.5):
        return _poly_ground_clip(ctx, pr, P3)
    ctx.move_to(xs[0], ys[0])
    for a, b in zip(xs[1:], ys[1:]):
        ctx.line_to(a, b)
    ctx.close_path()
    ctx.fill()


def _poly_ground_clip(ctx, pr, P3):
    """clip polygon against the near plane (camera z = 0.5) then fill"""
    xc, yc, zc = pr.cam_xyz(P3)
    out = []
    n = len(P3)
    for q in range(n):
        a, b = q, (q + 1) % n
        za, zb = zc[a] - 0.5, zc[b] - 0.5
        if za >= 0:
            out.append(P3[a])
        if (za >= 0) != (zb >= 0):
            t = za / (za - zb)
            out.append(P3[a] + (P3[b] - P3[a]) * t)
    if len(out) < 3:
        return
    xs, ys, _ = pr(np.array(out))
    ctx.move_to(xs[0], ys[0])
    for a, b in zip(xs[1:], ys[1:]):
        ctx.line_to(a, b)
    ctx.close_path()
    ctx.fill()


def _draw_mesh_shadows(ctx, pr, M, sd, whole):
    if whole:
        G = ground_shadow_pts(M.V, sd)
        base_pts = M.V[:, :2]
        hull = _hull([tuple(p) for p in np.round(np.concatenate([G, base_pts]), 3).tolist()])
        if len(hull) >= 3:
            _poly_ground(ctx, pr, np.array(hull))
        return
    big, small = _obj_groups(M)
    for V in big:
        G = ground_shadow_pts(V, sd)
        hull = _hull([tuple(p) for p in np.round(np.concatenate([G, V[:, :2]]), 3).tolist()])
        if len(hull) >= 3:
            _poly_ground(ctx, pr, np.array(hull))
    if len(small):
        c, rad, top = small[:, :2], small[:, 2], small[:, 3]
        shd = -sd[:2] / (np.linalg.norm(sd[:2]) + 1e-9)
        pp = np.array([-shd[1], shd[0]])
        Ls = np.minimum(top / max(sd[2], math.sin(math.radians(1.6))), 60.0)
        Q = np.zeros((len(c), 4, 3))
        Q[:, 0, :2] = c + pp * rad[:, None]
        Q[:, 1, :2] = c - pp * rad[:, None]
        Q[:, 2, :2] = c - pp * rad[:, None] * 0.6 + shd * Ls[:, None]
        Q[:, 3, :2] = c + pp * rad[:, None] * 0.6 + shd * Ls[:, None]
        xs, ys, zs = pr(Q.reshape(-1, 3))
        xs, ys, zs = xs.reshape(-1, 4), ys.reshape(-1, 4), zs.reshape(-1, 4)
        for k in range(len(c)):
            if zs[k].min() < 0.5:
                continue
            if max(xs[k].max(), -1) < -50 or xs[k].min() > W + 50 or ys[k].max() < -50 or ys[k].min() > H + 50:
                continue
            ctx.move_to(xs[k, 0], ys[k, 0])
            ctx.line_to(xs[k, 1], ys[k, 1])
            ctx.line_to(xs[k, 2], ys[k, 2])
            ctx.line_to(xs[k, 3], ys[k, 3])
            ctx.close_path()
        ctx.fill()


_OBJG = {}


def _obj_groups(M):
    """per-object vertex sets: big objects (hull shadows) and small ones as (cx, cy, radius, top)"""
    key = id(M)
    if key not in _OBJG:
        big, small = [], []
        for o in np.unique(M.obj):
            sel = np.nonzero(M.obj == o)[0]
            V = np.concatenate([M.V[M.off[k]:M.off[k + 1]] for k in sel])
            top = V[:, 2].max()
            if top < 0.05:
                continue
            ext = np.ptp(V[:, :2], axis=0).max()
            if ext > 3.0 or top > 2.0:
                big.append(V)
            else:
                c = V[:, :2].mean(0)
                small.append((c[0], c[1], 0.5 * ext, top))
        _OBJG[key] = (big, np.array(small).reshape(-1, 4))
    return _OBJG[key]


def _draw_survivor_shadows(ctx, pr, wd, idx, T, sh_dir, slen, wvec):
    """long dawn shadows of people + their raised flags (vectorised projection, cairo fill)"""
    n = len(idx)
    X, Y = wd.x[idx].astype(float), wd.y[idx].astype(float)
    Hh = wd.height[idx].astype(float)
    pp = np.array([-sh_dir[1], sh_dir[0]])
    L = np.minimum(Hh * slen, 60.0)
    wdt = 0.24
    # person: tapered quad
    q = np.zeros((n, 4, 3))
    q[:, 0, :2] = np.stack([X, Y], 1) + pp * wdt
    q[:, 1, :2] = np.stack([X, Y], 1) - pp * wdt
    tip = np.stack([X, Y], 1) + sh_dir[None] * L[:, None]
    q[:, 2, :2] = tip - pp * wdt * 0.7
    q[:, 3, :2] = tip + pp * wdt * 0.7
    xs, ys, zs = pr(q.reshape(-1, 3))
    xs, ys, zs = xs.reshape(n, 4), ys.reshape(n, 4), zs.reshape(n, 4)
    for k in range(n):
        if zs[k].min() < 0.5:
            continue
        ctx.move_to(xs[k, 0], ys[k, 0])
        ctx.line_to(xs[k, 1], ys[k, 1])
        ctx.line_to(xs[k, 2], ys[k, 2])
        ctx.line_to(xs[k, 3], ys[k, 3])
        ctx.close_path()
    ctx.fill()
    # raised flags: pole line + cloth parallelogram
    act = wd.act[idx]
    ta = T - act
    sel = np.nonzero(ta > 0)[0]
    if len(sel) == 0:
        return
    pops = np.array([_pop_curve(float(t)) for t in ta[sel]])
    Xs, Ys, Hs = X[sel], Y[sel], Hh[sel]
    zb = Hs * 0.62                              # hand
    zt = Hs + 1.55 * pops                       # pole top
    cw = 0.91 * pops
    chh = 0.76 * pops
    base = np.stack([Xs, Ys], 1) + pp * 0.15
    k_ = np.minimum(slen, 60.0)
    p0 = base + sh_dir[None] * (zb * k_)[:, None]
    p1 = base + sh_dir[None] * (zt * k_)[:, None]
    wv = wvec[:2]
    c0 = p1
    c1 = p1 + wv[None] * cw[:, None]
    c2 = base + sh_dir[None] * ((zt - chh) * k_)[:, None] + wv[None] * cw[:, None]
    c3 = base + sh_dir[None] * ((zt - chh) * k_)[:, None]
    m = len(sel)
    Q = np.zeros((m, 6, 3))
    Q[:, 0, :2], Q[:, 1, :2], Q[:, 2, :2], Q[:, 3, :2], Q[:, 4, :2], Q[:, 5, :2] = p0, p1, c0, c1, c2, c3
    xs, ys, zs = pr(Q.reshape(-1, 3))
    xs, ys, zs = xs.reshape(m, 6), ys.reshape(m, 6), zs.reshape(m, 6)
    ctx.save()
    ctx.set_line_width(1.2)
    for k in range(m):
        if zs[k].min() < 0.5:
            continue
        ctx.move_to(xs[k, 0], ys[k, 0])
        ctx.line_to(xs[k, 1], ys[k, 1])
        ctx.stroke()
        ctx.move_to(xs[k, 2], ys[k, 2])
        ctx.line_to(xs[k, 3], ys[k, 3])
        ctx.line_to(xs[k, 4], ys[k, 4])
        ctx.line_to(xs[k, 5], ys[k, 5])
        ctx.close_path()
        ctx.fill()
    ctx.restore()


def _draw_far_ruins(ctx, pr, T, L):
    hz = haze_color(T, 0.5)
    for az, dist, kind, sc in FAR_RUINS:
        a = math.radians(az)
        P = pr.C + np.array([math.sin(a) * dist, math.cos(a) * dist, -pr.C[2]])
        x, y, z = pr.pt(*P)
        if z < 1 or x < -600 or x > W + 600:
            continue
        s = pr.F / z * sc
        if s * 500 < 2:
            continue
        near = max(0.0, math.cos(math.radians(az - SUN_AZ))) ** 4
        c = hz * (0.55 - 0.15 * near) + np.array([0.06, 0.05, 0.10]) * L["sky"]
        ctx.set_source_rgb(*c)
        _far_shape(ctx, kind, x, y + 1.5, s, T)


_RIG_BROKEN = []
FOGSTATE = (0.0, np.zeros(3))
HAZE_NOW = [np.zeros(3)]
RIG_MIN = 46.0    # design px: above this, survivors are drawn with the full character rig


def _flag_frame(pr, X, Y, hgt, wvec, kx, upx, upy):
    """affine (x -> wind direction on screen, y -> world down) per metre for a crowd flag at (X, Y)"""
    wx1, wy1, _ = pr.pt(X + wvec[0], Y + wvec[1], hgt)
    wx0, wy0, _ = pr.pt(X, Y, hgt)
    ax, ay = wx1 - wx0, wy1 - wy0
    al = math.hypot(ax, ay) + 1e-9
    if al < 0.5 * kx:        # keep the cloth readable when the wind blows at the camera
        ax, ay = ax / al * 0.5 * kx, ay / al * 0.5 * kx
    if ax < 0:               # the loop bank flies to +x; mirror the frame instead of the cloth
        pass
    return ax, ay


def _draw_crowd_flag(ctx, pr, wd, i, T, bx, by, X, Y, hgt, wvec, kx, upx, upy, sun_side, ta, glows, ang=0.0):
    from vx.flag import draw_flag
    pop = _pop_curve(ta)
    ax, ay = _flag_frame(pr, X, Y, hgt, wvec, kx, upx, upy)
    u = 100.0
    ctx.save()
    ctx.transform(cairo.Matrix(ax / u, ay / u, -upx / u, -upy / u, bx, by))
    light = (-0.55 * sun_side, -0.3, -0.7)
    draw_flag(ctx, 0.0, 0.0, pole=2.44 * u, t=T + (int(wd.seed[i]) % 97) * 0.13, wind=0.8, side=1,
              seed=int(wd.seed[i]), pop=min(1.0, pop), lean=0.05, light=light, ang=ang)
    ctx.restore()
    cx_ = bx + upx * 2.1 + ax * 0.45
    cy_ = by + upy * 2.1 + ay * 0.45
    fl = 1.7 * math.exp(-ta / 0.16) + 0.55
    glows.append((cx_, cy_, max(2.6, 0.75 * kx), (1.0 * fl, 0.80 * fl, 0.36 * fl)))


def _draw_survivor(ctx, pr, wd, i, T, sx, sy, size_d, wvec, hill_sx, sun_side, lightk, L, glows, LOD_HI):
    """mid/near LOD survivor (+ the Flag in their hands). size_d = figure height in design px"""
    X, Y, hgt = float(wd.x[i]), float(wd.y[i]), float(wd.height[i])
    kx = size_d / hgt                                  # design px per metre (horizontal)
    tx, ty, _ = pr.pt(X, Y, 1.0)
    upx, upy = tx - sx, ty - sy
    ul = math.hypot(upx, upy) + 1e-9
    if ul < 0.62 * kx:
        upx, upy = upx / ul * 0.62 * kx, upy / ul * 0.62 * kx
    lk = seg(T, float(wd.look[i]), float(wd.look[i]) + 0.55, ease_in_out)
    ta = T - float(wd.act[i])
    face_hill = 1 if hill_sx > sx else -1
    rimk = L["sun"] * 0.85
    dist = math.hypot(X - pr.C[0], Y - pr.C[1])
    gk = (1.0 - clamp((T - float(wd.look[i])) / 0.35)) * L["glow"] * math.exp(-dist / 420.0)
    if wd.glow[i] < 0:
        gk = 0.0
    global FOGSTATE
    FOGSTATE = (1.0 - math.exp(-dist / 1700.0), HAZE_NOW[0])
    if size_d >= RIG_MIN and not _RIG_BROKEN:
        ctx.save()
        try:
            _draw_survivor_rig(ctx, pr, wd, i, T, sx, sy, kx, upx, upy, lk, ta, face_hill, sun_side, lightk, rimk,
                               gk, wvec, glows, L)
            return
        except Exception as ex:          # rig mid-update by its owner: fall back to the glyph tier
            _RIG_BROKEN.append(repr(ex))
            import sys
            print("s07a: rig tier disabled:", ex, file=sys.stderr)
        finally:
            ctx.restore()
    hand = draw_survivor_glyph(ctx, wd, i, sx, sy, kx, upx, upy, T, lk, ta, face_hill, sun_side, lightk, rimk)
    if gk > 0 and wd.glow[i] >= 0:   # the private feed, until they look up
        gc = GLOW_COLS[int(wd.glow[i])]
        gx, gy = hand[0], hand[1] - upy * 0.05
        glows.append((gx, gy, max(0.8, 0.08 * kx), tuple(gc * 0.9 * gk)))
        if size_d > 20:
            ctx.rectangle(gx - 0.045 * kx, gy - 0.07 * kx, 0.09 * kx, 0.14 * kx)
            ctx.set_source_rgba(*(0.45 + 0.55 * gc), gk)
            ctx.fill()
    if ta > 0:
        bx, by = hand[0] - upx * 0.5, hand[1] - upy * 0.5     # pole base below the raised hand
        _draw_crowd_flag(ctx, pr, wd, i, T, bx, by, X, Y, hgt, wvec, kx, upx, upy, sun_side, ta, glows)


def _pose_ok(pose):
    """map pose names the installed rig does not know to close relatives"""
    from vx import chars
    known = set(getattr(chars, "POSES", ()))
    alt = {"cup_shout": "shout", "look_up": "look_up", "meditate": "stand"}
    fix = lambda k: k if k in known else (alt.get(k) if alt.get(k) in known else "stand")
    if isinstance(pose, dict):
        out = {}
        for k, v in pose.items():
            out[fix(k)] = out.get(fix(k), 0.0) + v
        return out
    return fix(pose)


def _draw_survivor_rig(ctx, pr, wd, i, T, sx, sy, kx, upx, upy, lk, ta, face_hill, sun_side, lightk, rimk, gk,
                       wvec, glows, L):
    kind = int(wd.kind[i])
    hgt = float(wd.height[i])
    u = 100.0
    ch = _char(kind, int(wd.seed[i]))
    cloth, skin, hair = survivor_look(wd, i)
    fac = int(wd.facing[i]) if lk < 0.5 else face_hill
    t_loc = T + (int(wd.seed[i]) % 50) * 0.21
    if ta > 0:
        r = clamp(ta / 0.3)
        pose = {"look_up": 1.0 - r, "raise_flag": max(1e-3, r)}
        expr = "awe" if r < 1 else "happy"
    elif lk > 0.02:
        idle = {0: "stand", 1: "stand", 2: "cup_shout", 3: "stand"}[kind]
        pose = {idle: 1.0 - lk, "look_up": max(1e-3, lk)}
        expr = "awe"
    else:
        pose = {0: "stand", 1: "meditate", 2: "cup_shout", 3: "stand"}[kind]
        expr = {0: "smug", 1: "serene", 2: "furious", 3: "neutral"}[kind]
    nod = -0.35 * (1 - lk) * (kind != 2) + 0.45 * lk
    pose = _pose_ok(pose)
    lift = 0.34 if kind == 2 else 0.0
    kw = dict(pose=pose, t=t_loc, facing=fac, expr=expr, nod=nod, look=(0.0, -lk), wind=0.35,
              rim=(tuple(GLOW_COLS[int(wd.glow[i])]), 0.9 * gk, (0.0, 1.0)) if gk > 0.3 else
                  ((1.0, 0.64, 0.58), rimk, (sun_side, -0.2)),
              tint=(tuple((0.25 * np.array((0.22, 0.24, 0.42)) * (1 - FOGSTATE[0]) + FOGSTATE[1] * FOGSTATE[0])
                          * min(1.0, 1.2 * L["sky"])),
                    min(0.97, 1.0 - lightk + FOGSTATE[0])), robe=tuple(cloth), skin=tuple(skin), hair=tuple(hair),
              mouth=(0.55 + 0.35 * math.sin(T * 17 + i)) * (1 - lk) if kind == 2 else 0.0,
              pole_len=2.44 * u)
    if kw["tint"][1] > 0.85:            # deep dusk: flat silhouettes (no glinting eye whites)
        kw["silhouette"] = tuple(np.asarray(kw["tint"][0]) * 0.9 + np.asarray(cloth) * 0.1 * lightk)
    if gk > 0.02 and kind != 2:
        kw["prop"] = "phone"
    if kind == 2:
        rp = [(-0.5, -0.02), (0.55, -0.02), (0.42, 0.3), (0.12, 0.37), (-0.36, 0.3)]
        ctx.move_to(sx + rp[0][0] * kx + upx * rp[0][1], sy + upy * rp[0][1])
        for a_, b_ in rp[1:]:
            ctx.line_to(sx + a_ * kx + upx * b_, sy + upy * b_)
        ctx.close_path()
        ctx.set_source_rgb(*(np.array((0.30, 0.30, 0.37)) * lightk))
        ctx.fill()
    ox, oy = sx + upx * lift, sy + upy * lift
    M = cairo.Matrix(kx / u, 0.0, -upx / u, -upy / u, ox, oy)
    ctx.save()
    ctx.transform(M)
    try:
        a = ch.anchors(0.0, 0.0, hgt * u, **{k: v for k, v in kw.items() if k in ("pose", "t", "facing", "pole_len")})
    except TypeError:
        a = ch.anchors(0.0, 0.0, hgt * u, kw["pose"], kw["t"], kw["facing"], kw["pole_len"])
    if ta > 0:
        ch.draw(ctx, 0.0, 0.0, hgt * u, layer="back", **kw)
    else:
        ch.draw(ctx, 0.0, 0.0, hgt * u, **kw)
    ctx.restore()
    if ta > 0:
        pb = a.get("pole")
        if pb is not None:
            bx, by = M.transform_point(pb[0], pb[1])
            ang = float(pb[2]) * 0.6
        else:
            hx, hy = M.transform_point(*a["hand_r"])
            bx, by, ang = hx - upx * 0.5, hy - upy * 0.5, 0.0
        _draw_crowd_flag(ctx, pr, wd, i, T, bx, by, float(wd.x[i]), float(wd.y[i]), hgt, wvec, kx, upx, upy,
                         sun_side, ta, glows, ang=ang)
        ctx.save()
        ctx.transform(M)
        ch.draw(ctx, 0.0, 0.0, hgt * u, layer="front", **kw)
        ctx.restore()
    if gk > 0.02:
        gc = GLOW_COLS[int(wd.glow[i])]
        hx, hy = M.transform_point(*a.get("hand_r", (0.0, -hgt * u * 0.6)))
        glows.append((hx, hy, max(1.0, 0.12 * kx), tuple(gc * 0.9 * gk)))


HILL_MESH = None
RUIN_MESH = None


def _init_meshes():
    global HILL_MESH, RUIN_MESH
    if HILL_MESH is None:
        HILL_MESH = build_hill()
        RUIN_MESH = build_ruins()


_init_meshes()

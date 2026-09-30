"""m08 helper - staging: times, the camera path, where Crocus and Lutie walk, the planted Flag, the light model and
the dust in the air. Everything is a pure function of global time T.

WORLD = m07's world shifted so that m07's locked end camera shows the wall 1:1 on screen (M07_SHIFT: x' = x - 914,
y' = y - 620, z' = z). Design units, X right, Y DOWN, Z toward the viewer; the wall is z = 0, the nave floor is
y = FLOOR = 880. The camera is m07's: a frontal pinhole with horizontal fov 50 (focal F_CAM) and a lens shift (principal
point AT); at 146.0 it IS m07's CAM_END. The fallen tesserae, their heightfield, the candle and the pair's positions
come from m07's hand-off (cache/m07/handoff.npz).
"""
import math

import numpy as np

from vx.config import CACHE
from vx.ease import clamp, lerp, smoothstep, smootherstep, ease_in_out, fbm1

HANDOFF = CACHE / "m07" / "handoff.npz"
_H = np.load(HANDOFF)
F_CAM = 960.0 / math.tan(math.radians(float(_H["cam_fov"])) / 2.0)
AT = (float(_H["cam_at"][0]), float(_H["cam_at"][1]))
_E07 = np.asarray(_H["cam_eye"], np.float64)
if abs(F_CAM / _E07[2] - 1.0) > 1e-6:
    raise RuntimeError("m07's end camera must show the wall 1:1 (zoom 1) - update m08_stage for its zoom")
M07_SHIFT = np.array([AT[0] - _E07[0], AT[1] - _E07[1], 0.0])     # m07 world -> m08 world
FLOOR = float(_H["crocus_world"][1]) + M07_SHIFT[1]

# ------------------------------------------------------------------ times (global seconds)
T0, T1 = 146.0, 163.0
T_WIND = (146.0, 146.5)
T_WALK = (146.5, 148.1)
T_PLANT = 149.0                 # flag_planted: the foot of the pole strikes the heap
PLANT_DUR = 1.5                 # vx.icon pose='plant' progress 0..1 with impact at icon.PLANT_IMPACT = 0.6
T_PLANT0 = T_PLANT - 0.6 * PLANT_DUR
T_UNFURL = (149.3, 149.85)
T_C05 = (150.6, 154.87)
T_RISE = 155.0
T_N08 = (156.5, 158.88)
T_CONV = 159.0
T_TILT = (162.0, 163.0)

# ------------------------------------------------------------------ the heaps (m07's heightfield)
_HF = np.asarray(_H["heightfield"], np.float64)
_HX0, _HDX, _HZ0, _HDZ = (float(v) for v in _H["heightfield_grid"])


def heap_h(x, z):
    """height of the fallen tesserae above the floor at world (x, z) (bilinear in m07's heightfield)"""
    fx = (np.asarray(x, np.float64) - M07_SHIFT[0] - _HX0) / _HDX
    fz = (np.asarray(z, np.float64) - _HZ0) / _HDZ
    i0 = np.clip(np.floor(fx).astype(int), 0, _HF.shape[0] - 2)
    k0 = np.clip(np.floor(fz).astype(int), 0, _HF.shape[1] - 2)
    u = np.clip(fx - i0, 0.0, 1.0)
    v = np.clip(fz - k0, 0.0, 1.0)
    return (_HF[i0, k0] * (1 - u) * (1 - v) + _HF[i0 + 1, k0] * u * (1 - v) + _HF[i0, k0 + 1] * (1 - u) * v
            + _HF[i0 + 1, k0 + 1] * u * v)


# ------------------------------------------------------------------ places (world)
CROCUS_Z = LUTIE_Z = 150.0      # both climb onto the rubble slope in front of the tower's heap
CROCUS_H = float(_H["crocus_h"])
LUTIE_H = float(_H["lutie_h"])
CROCUS_X0 = float(_H["crocus_world"][0] + M07_SHIFT[0])   # m07 left them in the dark here
LUTIE_X0 = float(_H["lutie_world"][0] + M07_SHIFT[0])
LUTIE_X1 = 905.0


def ground_y(x, z=CROCUS_Z):
    """where a foot rests on the rubble at (x, z): the heap surface, smoothed over a stride"""
    xs = np.asarray(x, np.float64)[..., None] + np.linspace(-30.0, 30.0, 7)
    y = FLOOR - heap_h(xs, z).mean(-1)
    return float(y) if np.ndim(x) == 0 else y


PLANT_X = 1131.0                # the near right slope of the tower's heap (~(1145,790) on screen at 146.0)
PLANT = (PLANT_X, ground_y(PLANT_X) + 4.0, CROCUS_Z)
CROCUS_PLANT_X = PLANT_X - 5.3 * CROCUS_H / 250.0   # vx.icon 'plant' (facing -0.25): pole 0.021 h right of the feet
CROCUS_REST_X = PLANT_X - 34.9 * CROCUS_H / 250.0   # 'rest_pole' (facing -0.25): the pole at his right hand
POLE_LEN = 350.0


class Cam:
    """pinhole camera: eye E, pitch up by `pitch` radians (looking toward -Y), focal f (design px), principal point
    `at` (lens shift). project((N,3)) -> (N,2) screen design units, depth (N,)."""

    def __init__(self, ex, ey, ez, pitch=0.0, f=F_CAM, at=AT):
        self.E = np.array([ex, ey, ez], np.float64)
        self.pitch, self.f = float(pitch), float(f)
        self.at = (float(at[0]), float(at[1]))
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        self.F = np.array([0.0, -s, -c])       # forward
        self.D = np.array([0.0, c, -s])        # screen down
        self.R = np.array([1.0, 0.0, 0.0])     # screen right

    def project(self, P):
        d = np.asarray(P, np.float64) - self.E
        zc = np.maximum(d @ self.F, 1e-3)
        return np.stack([self.at[0] + self.f * (d @ self.R) / zc, self.at[1] + self.f * (d @ self.D) / zc], -1), zc

    def scale_at(self, P):
        """design px per world unit for an object facing the camera at P"""
        _, zc = self.project(np.atleast_2d(P))
        return self.f / zc

    def wall_homography(self):
        """3x3 H mapping wall (x, y, 1) at z=0 -> screen (homogeneous)"""
        E, R, D, F, f = self.E, self.R, self.D, self.F, self.f
        H = np.zeros((3, 3))
        for row, vec, c0 in ((0, R, self.at[0]), (1, D, self.at[1])):
            H[row, 0] = f * vec[0] + c0 * F[0]
            H[row, 1] = f * vec[1] + c0 * F[1]
            H[row, 2] = -(f * (vec @ E) + c0 * (F @ E))
        H[2, 0], H[2, 1], H[2, 2] = F[0], F[1], -(F @ E)
        return H

    def affine_at(self, P):
        """local 2x3 affine (world x,y at the depth of P -> screen) - for cairo drawing of flat things at P"""
        P = np.asarray(P, np.float64)
        h = 2.0
        s, _ = self.project(np.array([P, P + (h, 0, 0), P + (0, h, 0)]))
        ax = (s[1] - s[0]) / h
        ay = (s[2] - s[0]) / h
        b = s[0] - ax * P[0] - ay * P[1]
        return np.array([[ax[0], ay[0], b[0]], [ax[1], ay[1], b[1]]])

    def mz(self, offset=(0.0, 0.0, 0.0)):
        """the same camera as a vx.mosaic Camera; offset shifts the world (-M07_SHIFT: m07's world)"""
        from vx import mosaic as mz
        E = self.E + np.asarray(offset, np.float64)
        fov = math.degrees(2.0 * math.atan(960.0 / self.f))
        return mz.Camera(tuple(E), tuple(E + self.F * 1000.0), up=(0.0, -1.0, 0.0), fov=fov, at=self.at)


# ------------------------------------------------------------------ the camera path
def _k(t, t0, t1, fn=smootherstep):
    return fn(t0, t1, t)


def camera(T):
    """Framing = (wall point at the screen centre, zoom at the wall); the distance (1/zoom) is eased, a dolly.
    146.0 = m07's locked end camera. Push in on Crocus and the Flag during C05; pull back to take in the whole new
    wall with the rise; drift up toward the titulus; tilt up into the dome (whip: content moving down ~3000 px/s at
    162.958 for the m09 hand-off)."""
    cx, cy, iz = 960.0, 540.0, 1.0
    k = _k(T, 149.7, 154.9, smoothstep)
    cx, cy, iz = lerp(cx, 1150.0, k), lerp(cy, 520.0, k), lerp(iz, 1.0 / 2.0, k)
    k = _k(T, 155.05, 158.35)
    cx, cy, iz = lerp(cx, 1150.0, k), lerp(cy, 170.0, k), lerp(iz, 1.0 / 0.7, k)
    k = _k(T, 159.2, 162.4, smoothstep)
    cy, iz = lerp(cy, -240.0, k), lerp(iz, 1.0 / 0.74, k)
    ex, ey, ez = cx, cy - (540.0 - AT[1]) * iz, F_CAM * iz
    if T_PLANT <= T < T_PLANT + 0.5:                    # thud
        ey += math.exp(-(T - T_PLANT) / 0.09) * 5.0 * math.sin((T - T_PLANT) * 90.0)
    pitch = 0.0
    if T > T_TILT[0]:                                   # phi = a (t - t0)^3, f * phi'(162.958) = 3000 px/s
        a = (3000.0 / F_CAM) / (3.0 * (162.958 - T_TILT[0]) ** 2)
        pitch = a * (T - T_TILT[0]) ** 3
    return Cam(ex, ey, ez, pitch=pitch)


def tilt_blur(T):
    """vertical motion blur length (design px) for the hand-off whip"""
    return 150.0 * smoothstep(162.72, 162.958, T) ** 1.5


# ------------------------------------------------------------------ the pair
def crocus_x(T):
    return lerp(CROCUS_X0, CROCUS_PLANT_X, _k(T, T_WALK[0], T_WALK[1], smoothstep))


def lutie_x(T):
    return lerp(LUTIE_X0, LUTIE_X1, _k(T, T_WALK[0] + 0.2, T_WALK[1] + 0.35, smoothstep))


def steps(t0, t1, x0, x1, rate=1.8):
    """footfall times on the heap (for foley)"""
    out = []
    t = t0 + 0.12
    while t < t1:
        out.append(t)
        t += 1.0 / rate
    return out


def plant_progress(T):
    return clamp((T - T_PLANT0) / PLANT_DUR)


def planted_quiver(T):
    """angle wobble of the planted pole after the strike"""
    dt = T - T_PLANT
    if dt < 0:
        return 0.0
    return 0.05 * math.exp(-dt / 0.3) * math.sin(dt * 2 * math.pi * 5.5)


def wind(T):
    """mean wind (design px/s on the canonical pole) at the Flag: dead calm until the planting, then the breath
    arrives; the rushing wind of the rise; a steady strong wind of the completed world"""
    w = 0.0
    w += 380.0 * smoothstep(149.15, 149.9, T)
    w += 180.0 * smoothstep(150.5, 152.0, T)
    w += 520.0 * smoothstep(154.6, 155.4, T) * (1 - 0.45 * smoothstep(158.6, 160.0, T))
    return w * (1.0 + 0.18 * fbm1(T * 0.8, 3))


def gust(T):
    """the candle flame's wind (-1..1): the faint breath at 146.0-146.5, the rush of the rising tiles"""
    g = 0.35 * math.sin(math.pi * clamp((T - T_WIND[0]) / 0.8))
    return g + 0.25 * smoothstep(155.0, 155.3, T) * (1.0 - smoothstep(156.5, 158.0, T))


def furl(T):
    return 1.0 - smoothstep(T_UNFURL[0], T_UNFURL[1], T)


def flag_light(T):
    """power of the planted Flag's glow (0..1)"""
    return smoothstep(149.35, 150.6, T) * (1.0 + 0.25 * smoothstep(154.8, 155.6, T))


def flag_depth(T):
    """the Flag lives in Crocus's plane (carried, then planted beside him)"""
    return CROCUS_Z


def glory(T):
    """the completed wall's own light: gold wakes up with the wave"""
    return smoothstep(157.8, 159.05, T)


# ------------------------------------------------------------------ light for my own layers (same lights as the mosaic)
AMBIENT = np.array([0.014, 0.013, 0.013])
GLORY_RGB = np.array([1.0, 0.86, 0.64])


def wave_light(T, rho, rho_max):
    """the new world's own light: it wakes up behind the gold wave (0..1), per distance from the Flag's foot"""
    tw = 157.85 + 1.15 * (np.clip(rho / rho_max, 0, 1) ** 0.8)
    return np.clip((T - tw) / 0.45, 0.0, 1.0) ** 1.5


def light_at(P, T, lights, centre=None, rho_max=1.0):
    """rgb light (N,3) at world points P (N,3) facing the camera (no shadows); lights = [(pos, rgb, power, radius)]
    with the mosaic's falloff power / (1 + (d / radius)^2)"""
    P = np.asarray(P, np.float64)
    out = np.tile(AMBIENT * (1.0 + 0.6 * smoothstep(146.2, 148.5, T)), (len(P), 1))
    for pos, rgb, p, r in lights:
        d2 = np.sum((P - pos) ** 2, axis=1)
        out += (p / (1.0 + d2 / (r * r)))[:, None] * rgb[None, :]
    if centre is not None and T > 157.8:
        rho = np.hypot(P[:, 0] - centre[0], P[:, 1] - centre[1])
        g = wave_light(T, rho, rho_max)
        boost = 0.72 + 0.28 / (1.0 + (rho / 700.0) ** 2)
        out += (g * boost)[:, None] * GLORY_RGB[None, :] * 0.78
    return out


# ------------------------------------------------------------------ dust in the light
class Dust:
    """motes drifting in the church air (seen only where light falls on them), the breath of wind at 146.0, the
    puff of plaster dust when the pole strikes, the haze the rising tiles stir. Pure function of T."""

    def __init__(self, n=220, seed=5):
        rng = np.random.default_rng(seed)
        self.p0 = np.stack([rng.uniform(500, 1800, n), rng.uniform(250, FLOOR - 10, n), rng.uniform(40, 420, n)], 1)
        self.ph = rng.uniform(0, 2 * math.pi, (n, 3))
        self.fr = rng.uniform(0.05, 0.16, (n, 3))
        self.sz = rng.uniform(0.8, 2.2, n)
        m = 160
        self.puff_dir = rng.normal(0, 1, (m, 3)) * np.array([1.0, 0.45, 0.6])
        self.puff_v = rng.uniform(40, 150, m)
        self.puff_sz = rng.uniform(1.5, 4.0, m)

    def at(self, T):
        """-> pos (K,3), size (K,), alpha (K,)"""
        t = T - T0
        amp = np.array([22.0, 14.0, 10.0])
        drift = np.sin(self.ph + self.fr * t * 2 * np.pi) * amp
        g = smoothstep(T_WIND[0], T_WIND[1] + 0.6, T)       # the breath of wind carries them right and up
        drift[:, 0] += 38.0 * g
        drift[:, 1] -= 9.0 * g
        rise = smoothstep(154.9, 158.5, T)
        drift[:, 1] -= 120.0 * rise * (0.4 + 0.6 * np.sin(self.ph[:, 0]) ** 2)
        pos = self.p0 + drift
        alpha = np.full(len(pos), 0.55) * (1.0 - 0.6 * smoothstep(158.5, 160.5, T)) * smoothstep(146.1, 146.9, T)
        size = self.sz.copy()
        if T > T_PLANT - 0.02:
            tp = T - T_PLANT
            k = 1 - math.exp(-tp / 0.35)
            pp = np.array(PLANT)[None, :] + self.puff_dir * (self.puff_v * k)[:, None] * 0.9
            pp[:, 1] -= 40.0 * tp
            pos = np.vstack([pos, pp])
            alpha = np.concatenate([alpha, np.full(len(pp), 0.5 * math.exp(-tp / 0.9))])
            size = np.concatenate([size, self.puff_sz * (1 + 1.5 * k)])
        return pos, size, alpha

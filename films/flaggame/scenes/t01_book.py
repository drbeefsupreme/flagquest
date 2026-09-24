"""t01 — direction of the physical tract: the spine's rigid path (the director's hand), the weather and the
paper simulation run (cached).  Owner: T01Tract.

The stapled spine is driven along a scripted rigid path; every leaf is a real bending-paper simulation hanging
from it (t01_paper): wet droop, inertia, flutter, the pages clinging together and peeling apart in the wind.
  T < -1.558       (t09's tail) Raven's toss: the tract tumbles in from above the lens and SPLATS face-down
  -1.558 .. 2.52   LOOP POSE: face-down, back page up, spine on the right (rain, a little edge flutter)
  2.52 - 3.08      tract_flip: the gust lifts the spine edge; the booklet rolls over its free edge and slaps down
                   face-up (cover up, spine on the left), one page width to the left
  4.75 - 6.20      pages_riffle: the wind peels the pages; the cover goes over and stays; pages flap and fall back
  6.2 -            open on page 1 (cover to the left of the spine)
"""
import math

import numpy as np

from . import t01_mud as M
from . import t01_paper as PS

W_CM, H_CM = PS.PAGE_W_CM, PS.PAGE_H_CM
L = PS.LEAVES
YAW_A = math.radians(-9.0)          # the loop pose's yaw (spine direction vs +y)
YAW_B = math.radians(-3.0)
T_SIM0, T_SIM1 = -2.5, 7.6          # simulated span (T < T_SIM0 holds the first frame)
LAND_T = 197.4 - (199.0 - 1.0 / 24.0)   # t09's tract_lands cue in t01-local time (-1.558): the SPLAT
FLIP = [(2.50, 0.0), (2.60, 0.22), (2.70, 0.80), (2.82, 1.52), (2.93, 2.22), (3.02, 2.80), (3.07, math.pi)]
RIFFLE0, RIFFLE1 = 4.75, 6.2


def _pchip(keys, t):
    xs = np.array([k[0] for k in keys])
    ys = np.array([k[1] for k in keys])
    if t <= xs[0]:
        return float(ys[0])
    if t >= xs[-1]:
        return float(ys[-1])
    from scipy.interpolate import PchipInterpolator
    return float(PchipInterpolator(xs, ys)(t))


class Book:
    def __init__(self, ground):
        self.g = ground
        cx, cy = M.TRACT_XY
        self.c = np.array([cx, cy])
        # loop pose frame: v = along the spine (page up), u = across (spine -> free edge), in the ground plane
        self.vA = np.array([math.sin(YAW_A), math.cos(YAW_A)])
        self.uA = -np.array([math.cos(YAW_A), -math.sin(YAW_A)])   # face-down: free edge to the LEFT
        self.spineA = self.c - self.uA * (W_CM / 2)                  # spine line centre
        self.pivot = self.spineA + self.uA * W_CM                    # free edge line centre (flip axis)
        self._pch = None
        self._off = np.zeros(3)

    def z_on_ground(self, xy, extra=0.0):
        return float(self.g.height(xy[0], xy[1])) + 0.03 + (L - 1) * PS.THICK * 0.5 + extra

    def drop_pose(self, T):
        """rigid pose of the tossed booklet before the splat: (spine centre, U across, V along) in world"""
        u = min(1.0, max(0.0, (T - T_SIM0) / (LAND_T - T_SIM0)))
        th = 3.55 * (1 - u) ** 1.5                     # rolls about its spine, cover showing -> face-down
        yaw = 0.55 * (1 - u) ** 2
        U0 = np.array([self.uA[0], self.uA[1], 0.0])
        V0 = np.array([self.vA[0], self.vA[1], 0.0])
        N0 = np.cross(U0, V0)
        U = U0 * math.cos(th) + N0 * math.sin(th)
        cz, sz = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
        U, V = Rz @ U, Rz @ V0
        ca = np.array([self.spineA[0], self.spineA[1], self.z_on_ground(self.spineA)])
        c = ca + np.array([5.0 * (1 - u), -26.0 * (1 - u), 24.0 * (1 - u * u) + 0.2])
        # pivot the roll about the booklet's middle, not its spine
        mid = U0 * (W_CM / 2)
        c = c + (mid - U * (W_CM / 2)) * 1.0
        return c, U, V

    def spine(self, T, relax=True):
        """spine endpoints (p0 bottom, p1 top) at time T (scripted path; the sim adds a settling offset)"""
        if T < LAND_T:
            c, U, V = self.drop_pose(T)
            return c - V * H_CM / 2, c + V * H_CM / 2
        phi = _pchip(FLIP, T)
        k = phi / math.pi
        yaw = YAW_A + (YAW_B - YAW_A) * (k * k * (3 - 2 * k))
        v = np.array([math.sin(yaw), math.cos(yaw)])
        uA = -np.array([math.cos(yaw), -math.sin(yaw)])
        # centre of the spine rotates about the pivot line (which drifts a little during the roll)
        piv = self.pivot + np.array([-0.6, 0.9]) * k
        r = W_CM
        cxy = piv - uA * r * math.cos(phi)
        z_lift = r * math.sin(phi)
        zc = self.z_on_ground(cxy) * (1 - math.sin(phi)) + (self.z_on_ground(piv) + z_lift) * math.sin(phi)
        if phi >= math.pi - 1e-6:
            zc = self.z_on_ground(cxy)
        p0 = np.array([cxy[0] - v[0] * H_CM / 2, cxy[1] - v[1] * H_CM / 2, 0.0])
        p1 = np.array([cxy[0] + v[0] * H_CM / 2, cxy[1] + v[1] * H_CM / 2, 0.0])
        if phi <= 1e-6 or phi >= math.pi - 1e-6:
            p0[2] = self.z_on_ground(p0[:2])
            p1[2] = self.z_on_ground(p1[:2])
        else:
            p0[2] = p1[2] = zc
        return p0, p1

    def _relax(self, T, X):
        """after the flip lands, the stapled spine settles to where the leaves actually lie (no kink)"""
        if X is None or not (FLIP[-1][0] <= T < RIFFLE0 - 0.1):
            return np.zeros(3)
        c1 = np.mean([X[PS.NV + k * (PS.NU - 1) * PS.NV: PS.NV + k * (PS.NU - 1) * PS.NV + PS.NV] for k in range(L)], 0)
        c2 = np.mean([X[PS.NV + k * (PS.NU - 1) * PS.NV + PS.NV: PS.NV + k * (PS.NU - 1) * PS.NV + 2 * PS.NV]
                      for k in range(L)], 0)
        d = c1 - c2
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        nat = (c1 + d * PS.DX_CM).mean(0)
        p0, p1 = self.spine(T, relax=False)
        cur = 0.5 * (p0 + p1)
        err = nat - cur - self._off
        err[2] = 0.0
        self._off = self._off + err * 0.004
        return self._off

    def controls(self, T, X=None):
        # weather: a steady leftward breeze with gusts (cm/s), the flip gust, the riffle wind
        base = -110.0 - 60.0 * max(0.0, math.sin(T * 1.7)) ** 3
        if T > RIFFLE0:
            base *= 0.5
        flip_g = -650.0 * math.exp(-((T - 2.66) / 0.14) ** 2)
        R0 = RIFFLE0
        # the riffle wind: gusty from the right, then a back-gust from the left
        a = math.exp(-((T - (R0 + 0.30)) / 0.30) ** 2)
        c = math.exp(-((T - (R0 + 1.02)) / 0.26) ** 2)
        turb = 1.0 + 0.30 * math.sin(T * 13.0) + 0.2 * math.sin(T * 29.0 + 1.0)
        wind = (base + flip_g - 520.0 * a * turb + 380.0 * c * turb, 25.0 * math.sin(T * 0.9), 0.0)
        lift = [1.0] * L
        adh = [1.0] * (L - 1)
        crest = [PS.CREST_CLOSED] * (L // 2)
        turn = [0.0] * L

        def bump(t0, t1):
            if T <= t0 or T >= t1:
                return 0.0
            u = (T - t0) / (t1 - t0)
            return math.sin(math.pi * u) ** 2

        if T > R0 - 0.05:
            crest[0] = PS.crease_rest(math.pi * min(1.0, (T - R0 + 0.05) / 0.45))
            for k in range(4):
                if T >= R0 + 0.02 + 0.08 * k:
                    adh[k] = 0.0 if (k == 0 or T < R0 + 1.65) else min(1.0, (T - R0 - 1.65) / 0.6)
            # over: the cover, then pages 1 and 2; page 3 rises and falls back
            turn[0] = 5200.0 * bump(R0 + 0.00, R0 + 0.30)
            turn[1] = 4600.0 * bump(R0 + 0.14, R0 + 0.44)
            turn[2] = 4300.0 * bump(R0 + 0.26, R0 + 0.56)
            turn[3] = 2600.0 * bump(R0 + 0.36, R0 + 0.62)
            # back: page 2 then page 1 return to the right (the cover stays, glued to the mud)
            turn[2] -= 5200.0 * bump(R0 + 0.82, R0 + 1.12)
            turn[1] -= 5200.0 * bump(R0 + 0.96, R0 + 1.26)
            lift[0] = 1.0 if T < R0 + 0.6 else 0.0
            for k in range(4, L):
                lift[k] = 0.3
        off = self._relax(T, X)
        p0, p1 = self.spine(T, relax=False)
        if T >= FLIP[-1][0]:
            p0, p1 = p0 + self._off, p1 + self._off
        d = dict(spine=(p0, p1), wind=wind, lift=lift, adh=adh, crest=crest, wet=1.0,
                 fric=0.25, crease_comp=0.004, turn=turn)
        if T < LAND_T + 0.02:
            # in the air: the director's hand keeps the tumbling booklet closed (the free edge rides the pose)
            c, U, V = self.drop_pose(min(T, LAND_T))
            fe = c + U * W_CM
            d["pin"] = (fe - V * H_CM / 2, fe + V * H_CM / 2)
            d["pin_w"] = 0.08
            d["fric"] = 0.25
            return d
        # the flip hinge: the downwind (free) edge stays planted in the mud until the roll passes vertical
        phi = _pchip(FLIP, T)
        if FLIP[0][0] - 0.05 < T < FLIP[-1][0] and phi < 1.75:
            k = phi / math.pi
            yaw = YAW_A + (YAW_B - YAW_A) * (k * k * (3 - 2 * k))
            v = np.array([math.sin(yaw), math.cos(yaw)])
            piv = self.pivot + np.array([-0.6, 0.9]) * k
            q0 = np.array([piv[0] - v[0] * H_CM / 2, piv[1] - v[1] * H_CM / 2, 0.0])
            q1 = np.array([piv[0] + v[0] * H_CM / 2, piv[1] + v[1] * H_CM / 2, 0.0])
            q0[2] = float(self.g.height(q0[0], q0[1])) + 0.03
            q1[2] = float(self.g.height(q1[0], q1[1])) + 0.03
            d["pin"] = (q0, q1)
            d["pin_w"] = 0.35 * min(1.0, (1.75 - phi) / 0.3)
        return d

    def simulate(self, cache_dir=None):
        path = cache_dir / "book_v3.npz" if cache_dir is not None else None
        if path is not None and path.exists():
            z = np.load(path)
            return PS.BookTrack(z["verts"], float(z["t0"]), int(z["fps"]), z["energy"], [])
        bt = PS.simulate(self.g, T_SIM0, T_SIM1, controls=self.controls, center=tuple(self.c), warmup=0.6,
                         init=self.initial)
        if path is not None:
            np.savez(path, verts=bt.verts, t0=bt.t0, fps=bt.fps, energy=bt.energy)
        return bt

    def initial(self):
        """closed booklet in the air at the start of the toss (rigid drop pose at T_SIM0)"""
        X = np.zeros((PS.N_PART, 3))
        dx = W_CM / (PS.NU - 1)
        dy = H_CM / (PS.NV - 1)
        c, U, V = self.drop_pose(T_SIM0)
        N = np.cross(U, V)
        for k in range(L):
            for i in range(PS.NU):
                for j in range(PS.NV):
                    p = c + U * (i * dx) + V * (j * dy - H_CM / 2) + N * (k - (L - 1) / 2) * PS.THICK * (i > 0)
                    X[PS.pid(k, i, j)] = p
        return X

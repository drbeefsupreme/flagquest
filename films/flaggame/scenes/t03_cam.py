"""t03 perspective camera (owner T03Genesis).

World: X right, Y up, Z toward the default viewer; the ground is the plane Y = 0.
Cam(yaw, elev, dist, f, target) orbits `target`: yaw turns around Y, elev = angle above the ground plane.
Screen = design px (1920x1080). `project` maps world points to (sx, sy, depth).
"""
import math

import numpy as np


class Cam:
    def __init__(self, yaw=0.0, elev=0.5, dist=8.0, f=1400.0, target=(0.0, 0.0, 0.0), cx=960.0, cy=540.0, roll=0.0):
        self.yaw, self.elev, self.dist, self.f = float(yaw), float(elev), float(dist), float(f)
        self.target = np.asarray(target, np.float64)
        self.cx, self.cy = float(cx), float(cy)
        ce, se = math.cos(elev), math.sin(elev)
        self.C = self.target + dist * np.array([ce * math.sin(yaw), se, ce * math.cos(yaw)])
        fwd = self.target - self.C
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        if roll:
            cr, sr = math.cos(roll), math.sin(roll)
            right, down = right * cr + down * sr, -right * sr + down * cr
        self.fwd, self.right, self.down = fwd, right, down

    @staticmethod
    def look(C, target, f=1400.0, cx=960.0, cy=540.0):
        """camera at world point C looking at target"""
        C = np.asarray(C, np.float64)
        T = np.asarray(target, np.float64)
        d = C - T
        dist = float(np.linalg.norm(d))
        elev = math.asin(max(-1.0, min(1.0, d[1] / dist)))
        yaw = math.atan2(d[0], d[2])
        return Cam(yaw, elev, dist, f, T, cx, cy)

    def params(self):
        """flat float64 array for the numba kernels: C(3) right(3) down(3) fwd(3) f cx cy"""
        return np.concatenate([self.C, self.right, self.down, self.fwd, [self.f, self.cx, self.cy]]).astype(np.float64)

    def project(self, P):
        """P (..., 3) world -> (..., 3) [sx, sy, depth]"""
        P = np.asarray(P, np.float64)
        d = P - self.C
        z = d @ self.fwd
        zs = np.maximum(z, 1e-6)
        x = self.cx + self.f * (d @ self.right) / zs
        y = self.cy + self.f * (d @ self.down) / zs
        return np.stack([x, y, z], axis=-1)

    def scale_at(self, P):
        """design px per world unit at world point P"""
        d = np.asarray(P, np.float64) - self.C
        return self.f / max(1e-6, float(d @ self.fwd))

    def light_view(self, Lw):
        """world direction-to-light -> vx.flag light (x right, y down, z toward camera)"""
        L = np.asarray(Lw, np.float64)
        L = L / np.linalg.norm(L)
        return (float(L @ self.right), float(L @ self.down), float(-(L @ self.fwd)))

    def ground_ray(self, sx, sy):
        """world point on Y=0 under screen point (or None above the horizon)"""
        d = self.fwd + self.right * (sx - self.cx) / self.f + self.down * (sy - self.cy) / self.f
        if d[1] >= -1e-9:
            return None
        t = -self.C[1] / d[1]
        return self.C + t * d

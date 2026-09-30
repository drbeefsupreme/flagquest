"""vx.mosaic_view - cameras and architectural surfaces for vx.mosaic (owner: Tessellator). Re-exported by vx.mosaic.

World coordinates (design units): X right, Y DOWN, Z toward the viewer. The default flat wall is the plane z = 0
whose chart is its own (x, y). A layout always lives in its surface's 2D CHART; views map chart/world -> screen
design units (x right, y down, 1920x1080); the renderer multiplies by fc.s for pixels.

Views (all have .project3(P) -> (screen xy (n,2), depth (n,)), .eye (3,), .plane_to_screen(uv) for flat walls):
  View(cx, cy, zoom=1, rot=0, at=(960, 540), eye=None)   2D pan/zoom/rotate over a flat wall (orthographic look
        with a virtual eye at distance ~1.3*W/zoom for view-dependent gold and lifted-tile parallax)
  View.affine(M)         any 2x3 affine wall -> screen design units
  View.homography(Hm)    any 3x3 homography wall -> screen design units (tilted wall / keystone)
  Camera(eye, target, up=(0,-1,0), fov=50, at=(960, 540))  3D pinhole; fov = horizontal degrees.
        Camera.frontal(cx, cy, zoom=1, dist=1500): looks straight at the wall z=0, wall (cx,cy) at screen centre,
        1:1 scale at zoom 1 (a perspective twin of View).
Surfaces (chart -> world): Plane, Conch, Hemicycle, Dome, Vault. Each has .embed(uv)->P (n,3), .frame(uv) ->
  (P, T1, T2, N) with N the unit normal toward the visible (interior/front) side, and .k(u, v) = physical length
  per chart unit (1 for isometric charts) so layouts can keep tesserae a constant physical size
  (Layout.flow(..., density=surface.k)).
"""
import math

import cv2
import numpy as np

from .config import W, H


def _n(v):
    v = np.asarray(v, np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


# ============================================================ views
class View:
    """2D view of a flat wall (chart = wall coords). Also wraps arbitrary affines / homographies."""
    kind = "affine"

    def __init__(self, cx=W / 2, cy=H / 2, zoom=1.0, rot=0.0, at=(W / 2, H / 2), eye=None):
        c, s = math.cos(rot), math.sin(rot)
        A = zoom * np.array([[c, -s], [s, c]])
        b = np.asarray(at, np.float64) - A @ np.array([cx, cy], np.float64)
        self._set_affine(np.hstack([A, b[:, None]]), eye)

    def _set_affine(self, M, eye=None):
        self.kind = "affine"
        self.M = np.asarray(M, np.float64).reshape(2, 3)
        A = self.M[:, :2]
        self.Ainv = np.linalg.inv(A)
        self.zoom = math.sqrt(abs(np.linalg.det(A)))
        c = self.Ainv @ (np.array([W / 2, H / 2]) - self.M[:, 2])
        E = float(eye) if eye is not None else 1.3 * W / max(self.zoom, 1e-6)
        self.eye = np.array([c[0], c[1], E])
        self.Hm = np.vstack([self.M, [0, 0, 1]])
        return self

    @classmethod
    def affine(cls, M, eye=None):
        v = cls.__new__(cls)
        return v._set_affine(M, eye)

    @classmethod
    def homography(cls, Hm, eye=None):
        v = cls.__new__(cls)
        v.kind = "homog"
        v.Hm = np.asarray(Hm, np.float64).reshape(3, 3)
        v.Hinv = np.linalg.inv(v.Hm)
        c = v.Hinv @ np.array([W / 2, H / 2, 1.0])
        c = c[:2] / c[2]
        # local scale at the centre
        J = _homog_jac(v.Hm, c)
        v.zoom = math.sqrt(abs(np.linalg.det(J)))
        E = float(eye) if eye is not None else 1.3 * W / max(v.zoom, 1e-6)
        v.eye = np.array([c[0], c[1], E])
        v.M = None
        return v

    # ---------------------------------------------------------------- mapping
    def plane_to_screen(self, uv):
        uv = np.asarray(uv, np.float64)
        if self.kind == "affine":
            return uv @ self.M[:, :2].T + self.M[:, 2]
        p = uv @ self.Hm[:, :2].T + self.Hm[:, 2]
        return p[..., :2] / p[..., 2:3]

    def screen_to_plane(self, xy):
        xy = np.asarray(xy, np.float64)
        if self.kind == "affine":
            return (xy - self.M[:, 2]) @ self.Ainv.T
        p = xy @ self.Hinv[:, :2].T + self.Hinv[:, 2]
        return p[..., :2] / p[..., 2:3]

    def project3(self, P):
        """world points (wall x, y, z toward viewer) -> screen design units + depth (distance-like)."""
        P = np.asarray(P, np.float64)
        e = self.eye
        E = e[2]
        dz = np.maximum(E - P[..., 2], 1e-3)
        f = (E / dz)[..., None]
        q = e[:2] + (P[..., :2] - e[:2]) * f
        return self.plane_to_screen(q), dz

    def warp(self, fc, img, s_img=None, border=cv2.BORDER_REFLECT, interp=cv2.INTER_LINEAR):
        """wall-space image (any scale s_img px per wall unit; default fc.s) -> screen image (fc.h, fc.w)."""
        s_img = fc.s if s_img is None else s_img
        S = np.diag([fc.s, fc.s, 1.0]) @ self.Hm @ np.diag([1 / s_img, 1 / s_img, 1.0])
        if self.kind == "affine":
            return cv2.warpAffine(img, S[:2], (fc.w, fc.h), flags=interp, borderMode=border)
        return cv2.warpPerspective(img, S, (fc.w, fc.h), flags=interp, borderMode=border)

    def world_grid(self, fc, step=24):
        """coarse screen grid -> (gx, gy) pixel coords, world points (gh, gw, 3), normals (for lighting maps)."""
        xs = (np.arange(0, fc.w + step, step, dtype=np.float64))
        ys = (np.arange(0, fc.h + step, step, dtype=np.float64))
        gx, gy = np.meshgrid(xs, ys)
        uv = self.screen_to_plane(np.stack([gx / fc.s, gy / fc.s], -1))
        P = np.concatenate([uv, np.zeros(uv.shape[:2] + (1,))], -1)
        N = np.zeros_like(P)
        N[..., 2] = 1
        return gx, gy, P, N

    def __repr__(self):
        return f"View({self.kind}, zoom={self.zoom:.3f})"


def _homog_jac(Hm, p):
    x, y = p
    d = Hm[2, 0] * x + Hm[2, 1] * y + Hm[2, 2]
    u = Hm[0, 0] * x + Hm[0, 1] * y + Hm[0, 2]
    v = Hm[1, 0] * x + Hm[1, 1] * y + Hm[1, 2]
    return np.array([[(Hm[0, 0] * d - u * Hm[2, 0]) / d ** 2, (Hm[0, 1] * d - u * Hm[2, 1]) / d ** 2],
                     [(Hm[1, 0] * d - v * Hm[2, 0]) / d ** 2, (Hm[1, 1] * d - v * Hm[2, 1]) / d ** 2]])


class Camera:
    """3D pinhole camera in world design units. fov = horizontal field of view (degrees)."""
    kind = "camera"

    def __init__(self, eye, target, up=(0.0, -1.0, 0.0), fov=50.0, at=(W / 2, H / 2), near=1.0):
        self.eye = np.asarray(eye, np.float64)
        self.target = np.asarray(target, np.float64)
        f = _n(self.target - self.eye)
        r = np.cross(np.asarray(up, np.float64), f)
        if np.linalg.norm(r) < 1e-9:
            r = np.cross(np.array([0.0, 0.0, 1.0]), f)
        r = _n(r)
        d = np.cross(r, f)
        self.f, self.r, self.d = f, r, d
        self.fov = fov
        self.F = (W / 2) / math.tan(math.radians(fov) / 2)
        self.at = np.asarray(at, np.float64)
        self.near = near
        self.zoom = 1.0

    @classmethod
    def frontal(cls, cx=W / 2, cy=H / 2, zoom=1.0, dist=1500.0, at=(W / 2, H / 2)):
        fov = math.degrees(2 * math.atan((W / 2) / (dist * zoom)))
        return cls((cx, cy, dist), (cx, cy, 0.0), fov=fov, at=at)

    def project3(self, P):
        rel = np.asarray(P, np.float64) - self.eye
        x = rel @ self.r
        y = rel @ self.d
        z = rel @ self.f
        zs = np.maximum(z, 1e-6)
        out = np.stack([self.at[0] + self.F * x / zs, self.at[1] + self.F * y / zs], -1)
        return out, z

    def ray(self, xy):
        """screen design units -> unit ray directions (n,3)."""
        xy = np.asarray(xy, np.float64)
        x = (xy[..., 0] - self.at[0]) / self.F
        y = (xy[..., 1] - self.at[1]) / self.F
        d = self.f + x[..., None] * self.r + y[..., None] * self.d
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def world_grid(self, fc, surface, step=24):
        xs = (np.arange(0, fc.w + step, step, dtype=np.float64))
        ys = (np.arange(0, fc.h + step, step, dtype=np.float64))
        gx, gy = np.meshgrid(xs, ys)
        D = self.ray(np.stack([gx / fc.s, gy / fc.s], -1))
        P, N, ok = surface.intersect(self.eye, D)
        return gx, gy, P, N, ok

    def __repr__(self):
        return f"Camera(eye={self.eye.round(1).tolist()}, fov={self.fov:.1f})"


# ============================================================ surfaces
class Surface:
    def frame(self, uv, h=0.5):
        uv = np.asarray(uv, np.float64)
        P = self.embed(uv)
        du = self.embed(uv + [h, 0]) - self.embed(uv - [h, 0])
        dv = self.embed(uv + [0, h]) - self.embed(uv - [0, h])
        T1 = du / np.maximum(np.linalg.norm(du, axis=-1, keepdims=True), 1e-12)
        T2 = dv / np.maximum(np.linalg.norm(dv, axis=-1, keepdims=True), 1e-12)
        N = self.normal(uv, P)
        return P, T1, T2, N

    def k(self, u, v):
        return np.ones_like(np.asarray(u, np.float64))


class Plane(Surface):
    """flat wall: P = origin + u*ex + v*ey (u, v design units); normal = ex x ey (toward the viewer for defaults)."""

    def __init__(self, origin=(0.0, 0.0, 0.0), ex=(1.0, 0.0, 0.0), ey=(0.0, 1.0, 0.0)):
        self.o = np.asarray(origin, np.float64)
        self.ex = np.asarray(ex, np.float64)
        self.ey = np.asarray(ey, np.float64)
        self.nrm = _n(np.cross(self.ex, self.ey))

    def embed(self, uv):
        uv = np.asarray(uv, np.float64)
        return self.o + uv[..., :1] * self.ex + uv[..., 1:2] * self.ey

    def normal(self, uv, P=None):
        return np.broadcast_to(self.nrm, np.asarray(uv).shape[:-1] + (3,)).copy()

    def intersect(self, o, D):
        den = D @ self.nrm
        t = ((self.o - o) @ self.nrm) / np.where(np.abs(den) < 1e-9, 1e-9, den)
        P = o + D * t[..., None]
        ok = t > 0
        N = np.broadcast_to(self.nrm, P.shape).copy()
        return P, N, ok

    def to_chart(self, P):
        rel = np.asarray(P) - self.o
        G = np.array([[self.ex @ self.ex, self.ex @ self.ey], [self.ex @ self.ey, self.ey @ self.ey]])
        b = np.stack([rel @ self.ex, rel @ self.ey], -1)
        return b @ np.linalg.inv(G).T


class Conch(Surface):
    """Apse semi-dome (quarter sphere) seen from the nave. Opening plane z = 0 (facing +Z), springing line
    through world (cx, cy, 0), vault above it (y < cy), radius R. CHART = the front view: stereographic
    projection from the point (cx, cy, +R) onto the opening plane, i.e. chart (u, v) with the springing midpoint
    at chart (cx, cy); the conch fills the upper half-disc |(u,v)-(cx,cy)| <= R, v <= cy. The chart is exactly
    what Camera(eye=(cx, cy, R), target=(cx, cy, 0), fov=90) sees; conformal; k = 2/(1+(rho/R)^2) physical
    units per chart unit (2 at the crown's deepest point, 1 at the rim)."""

    def __init__(self, R, cx=W / 2, cy=H * 0.75, cz=0.0):
        self.R, self.c = float(R), np.array([cx, cy, cz], np.float64)

    def embed(self, uv):
        uv = np.asarray(uv, np.float64)
        du = uv[..., 0] - self.c[0]
        dv = uv[..., 1] - self.c[1]
        rho = np.sqrt(du * du + dv * dv)
        th = 2 * np.arctan(rho / self.R)
        s = np.where(rho > 1e-9, np.sin(th) / np.maximum(rho, 1e-9), 2 / self.R)
        return self.c + self.R * np.stack([s * du, s * dv, -np.cos(th)], -1)

    def normal(self, uv, P=None):
        P = self.embed(uv) if P is None else P
        return (self.c - P) / self.R

    def k(self, u, v):
        rho2 = (np.asarray(u) - self.c[0]) ** 2 + (np.asarray(v) - self.c[1]) ** 2
        return 2.0 / (1.0 + rho2 / self.R ** 2)

    def intersect(self, o, D):
        return _sphere_hit(o, D, self.c, self.R, lambda P: (P[..., 2] <= self.c[2] + 1e-6) & (P[..., 1] <= self.c[1] + 1e-6))

    def to_chart(self, P):
        rel = (np.asarray(P) - self.c) / self.R
        th = np.arccos(np.clip(-rel[..., 2], -1, 1))
        rho = self.R * np.tan(th / 2)
        l = np.maximum(np.hypot(rel[..., 0], rel[..., 1]), 1e-12)
        return np.stack([self.c[0] + rho * rel[..., 0] / l, self.c[1] + rho * rel[..., 1] / l], -1)


class Hemicycle(Surface):
    """Standing half-cylinder wall of an apse below the conch: axis vertical through world (cx, *, cz), radius R,
    receding from the viewer (the wall bends away to z = cz - R at its centre). CHART: u = u0 + arc length from the
    centre line (|u - u0| <= pi*R/2; u0 defaults to pi*R/2 so the chart starts at 0), v = world y (design units,
    down). Isometric (k = 1)."""

    def __init__(self, R, cx=W / 2, cz=0.0, u0=None):
        self.R, self.cx, self.cz = float(R), float(cx), float(cz)
        self.u0 = math.pi * self.R / 2 if u0 is None else float(u0)

    def embed(self, uv):
        uv = np.asarray(uv, np.float64)
        psi = (uv[..., 0] - self.u0) / self.R
        return np.stack([self.cx + self.R * np.sin(psi), uv[..., 1], self.cz - self.R * np.cos(psi)], -1)

    def normal(self, uv, P=None):
        psi = (np.asarray(uv, np.float64)[..., 0] - self.u0) / self.R
        return np.stack([-np.sin(psi), np.zeros_like(psi), np.cos(psi)], -1)

    def intersect(self, o, D):
        ox, oz = o[0] - self.cx, o[2] - self.cz
        a = D[..., 0] ** 2 + D[..., 2] ** 2
        b = 2 * (ox * D[..., 0] + oz * D[..., 2])
        c = ox * ox + oz * oz - self.R ** 2
        disc = b * b - 4 * a * c
        sq = np.sqrt(np.maximum(disc, 0))
        t = (-b + sq) / np.maximum(2 * a, 1e-12)
        P = o + D * t[..., None]
        ok = (disc > 0) & (t > 0) & (P[..., 2] <= self.cz + 1e-6)
        psi = np.arctan2(P[..., 0] - self.cx, -(P[..., 2] - self.cz))
        N = np.stack([-np.sin(psi), np.zeros_like(psi), np.cos(psi)], -1)
        return P, N, ok

    def to_chart(self, P):
        psi = np.arctan2(P[..., 0] - self.cx, -(P[..., 2] - self.cz))
        return np.stack([self.u0 + self.R * psi, P[..., 1]], -1)


class Dome(Surface):
    """Hemispherical dome seen from below. Springing circle centre at world `center`, radius R, crown at
    center + (0, -R, 0) (up = -Y). CHART: stereographic from the nadir (center + (0, R, 0)) onto the tangent plane
    at the crown, scaled so chart units = physical units at the crown: rho = 2R tan(theta/2) (theta = angle from
    the crown), the dome fills the disc of radius 2R around chart point (ccx, ccy); chart +u = world +X, chart +v =
    world +Z (toward the default viewer). Conformal; k = 1/(1+(rho/2R)^2) (1 at the crown, 0.5 at the springing).
    A camera at the nadir looking up (dome_up) sees the chart undistorted."""

    def __init__(self, R, center=(0.0, 0.0, 0.0), chart_center=(0.0, 0.0)):
        self.R = float(R)
        self.c = np.asarray(center, np.float64)
        self.cc = np.asarray(chart_center, np.float64)

    def embed(self, uv):
        uv = np.asarray(uv, np.float64)
        du = uv[..., 0] - self.cc[0]
        dv = uv[..., 1] - self.cc[1]
        rho = np.sqrt(du * du + dv * dv)
        th = 2 * np.arctan(rho / (2 * self.R))
        s = np.where(rho > 1e-9, np.sin(th) / np.maximum(rho, 1e-9), 1 / self.R)
        return self.c + self.R * np.stack([s * du, -np.cos(th), s * dv], -1)

    def normal(self, uv, P=None):
        P = self.embed(uv) if P is None else P
        return (self.c - P) / self.R

    def k(self, u, v):
        rho2 = (np.asarray(u) - self.cc[0]) ** 2 + (np.asarray(v) - self.cc[1]) ** 2
        return 1.0 / (1.0 + rho2 / (4 * self.R ** 2))

    def intersect(self, o, D):
        return _sphere_hit(o, D, self.c, self.R, lambda P: P[..., 1] <= self.c[1] + 1e-6)

    def to_chart(self, P):
        rel = (np.asarray(P) - self.c) / self.R
        th = np.arccos(np.clip(-rel[..., 1], -1, 1))
        rho = 2 * self.R * np.tan(th / 2)
        l = np.maximum(np.hypot(rel[..., 0], rel[..., 2]), 1e-12)
        return np.stack([self.cc[0] + rho * rel[..., 0] / l, self.cc[1] + rho * rel[..., 2] / l], -1)


class Vault(Surface):
    """Barrel vault over a nave: semicircular section of radius R centred at world (cx, cy) (springing level),
    crown at y = cy - R, running from z = z0 away from the viewer (-Z). CHART: u = u0 + arc length across the
    vault from the crown line (|u - u0| <= pi*R/2; u0 defaults to pi*R/2 so the chart starts at 0), v = distance
    along the axis from z0. Isometric."""

    def __init__(self, R, cx=W / 2, cy=H, z0=0.0, u0=None):
        self.R, self.cx, self.cy, self.z0 = float(R), float(cx), float(cy), float(z0)
        self.u0 = math.pi * self.R / 2 if u0 is None else float(u0)

    def embed(self, uv):
        uv = np.asarray(uv, np.float64)
        a = (uv[..., 0] - self.u0) / self.R
        return np.stack([self.cx + self.R * np.sin(a), self.cy - self.R * np.cos(a), self.z0 - uv[..., 1]], -1)

    def normal(self, uv, P=None):
        a = (np.asarray(uv, np.float64)[..., 0] - self.u0) / self.R
        return np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], -1)

    def intersect(self, o, D):
        ox, oy = o[0] - self.cx, o[1] - self.cy
        a = D[..., 0] ** 2 + D[..., 1] ** 2
        b = 2 * (ox * D[..., 0] + oy * D[..., 1])
        c = ox * ox + oy * oy - self.R ** 2
        disc = b * b - 4 * a * c
        t = (-b + np.sqrt(np.maximum(disc, 0))) / np.maximum(2 * a, 1e-12)
        P = o + D * t[..., None]
        ok = (disc > 0) & (t > 0) & (P[..., 1] <= self.cy + 1e-6)
        ang = np.arctan2(P[..., 0] - self.cx, -(P[..., 1] - self.cy))
        N = np.stack([-np.sin(ang), np.cos(ang), np.zeros_like(ang)], -1)
        return P, N, ok

    def to_chart(self, P):
        ang = np.arctan2(P[..., 0] - self.cx, -(P[..., 1] - self.cy))
        return np.stack([self.u0 + self.R * ang, self.z0 - P[..., 2]], -1)


def _sphere_hit(o, D, c, R, keep):
    oc = o - c
    b = D @ oc
    cc = oc @ oc - R * R
    disc = b * b - cc
    sq = np.sqrt(np.maximum(disc, 0))
    t = -b + sq          # far hit: we look at the INSIDE of the sphere
    P = o + D * t[..., None]
    ok = (disc > 0) & (t > 0) & keep(P)
    N = (c - P) / R
    return P, N, ok


def dome_up(dome, height=None, fov=None, rot=0.0, at=(W / 2, H / 2)):
    """Camera under a Dome looking straight up at the crown. height = distance below the springing plane
    (default R = the nadir: the chart appears undistorted). rot rotates the view about the vertical axis."""
    h = dome.R if height is None else float(height)
    eye = dome.c + np.array([0.0, h, 0.0])
    up = np.array([math.sin(rot), 0.0, -math.cos(rot)])
    if fov is None:
        # show the whole dome: springing circle at angle atan(R/h) off-axis
        fov = min(2 * math.degrees(math.atan(dome.R / h)) + 8, 170)
    return Camera(eye, dome.c + np.array([0.0, -dome.R, 0.0]), up=up, fov=fov, at=at)

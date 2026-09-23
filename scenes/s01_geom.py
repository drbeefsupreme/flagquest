"""s01 helper: exact 4D geometry of THE CRYSTAL (the 24-cell) and its hyperplane sections.

The 24-cell is the one regular convex polytope with no analogue in any other dimension: 24 vertices
(all permutations of (+-1, +-1, 0, 0)), 96 edges of length sqrt(2), 96 triangles, 24 octahedral cells.
As a half-space intersection it is  |x_i| <= 1  and  |x1 +- x2 +- x3 +- x4| <= 2.

Everything here is real 4D math:
  * rotations are products of Givens rotations in the six coordinate planes (XW, YW, ZW, XY, YZ, XZ);
    rotations by multiples of pi/2 in coordinate planes are symmetries of the 24-cell, so landing every
    angle on such a multiple returns the Crystal to its canonical pose exactly;
  * projection is a true central projection 4D -> 3D (eye on the +W axis) followed by 3D -> 2D;
  * the section by the 2-plane {z = z0, w = w0} is computed exactly by clipping with the 24 half-spaces
    of the (rotated) polytope. In canonical pose the central section is the square [-1, 1]^2.
"""
import itertools
import math

import numpy as np

# ------------------------------------------------------------------ the polytope
def cell24():
    V = []
    for i, j in itertools.combinations(range(4), 2):
        for si in (-1, 1):
            for sj in (-1, 1):
                v = [0, 0, 0, 0]
                v[i], v[j] = si, sj
                V.append(v)
    V = np.array(V, np.float64)
    D = np.linalg.norm(V[:, None] - V[None], axis=-1)
    E = np.array([(a, b) for a in range(24) for b in range(a + 1, 24) if abs(D[a, b] - math.sqrt(2)) < 1e-9])
    adj = np.abs(D - math.sqrt(2)) < 1e-9
    T = np.array([(a, b, c) for a, b, c in itertools.combinations(range(24), 3) if adj[a, b] and adj[b, c] and adj[a, c]])
    N, d = [], []
    for i in range(4):
        for s in (-1, 1):
            n = [0, 0, 0, 0]
            n[i] = s
            N.append(n)
            d.append(1.0)
    for signs in itertools.product((-1, 1), repeat=4):
        N.append(list(signs))
        d.append(2.0)
    return V, E, T, np.array(N, np.float64), np.array(d, np.float64)


V24, E24, T24, N24, D24 = cell24()
assert len(E24) == 96 and len(T24) == 96 and len(N24) == 24

PLANES = {"xy": (0, 1), "xz": (0, 2), "xw": (0, 3), "yz": (1, 2), "yw": (1, 3), "zw": (2, 3)}
ORDER = ("xw", "yw", "zw", "xy", "yz", "xz")


def givens(i, j, a):
    R = np.eye(4)
    c, s = math.cos(a), math.sin(a)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


# the 24 octahedral cells: the 6 vertices lying on each facet hyperplane, and the facet centre
CELLS24 = np.array([np.nonzero(np.abs(V24 @ n - d) < 1e-9)[0] for n, d in zip(N24, D24)])
CELL_C24 = N24 * (D24 / np.sum(N24 * N24, 1))[:, None]
assert CELLS24.shape == (24, 6)


def hull2(P):
    """2D convex hull (monotone chain), CCW"""
    P = sorted(map(tuple, np.asarray(P, np.float64)))
    if len(P) < 3:
        return np.array(P)

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def rot4(angles):
    """angles: dict plane -> radians. Returns 4x4 matrix (applied as p = R @ v)."""
    R = np.eye(4)
    for k in ORDER:
        a = angles.get(k, 0.0)
        if a:
            i, j = PLANES[k]
            R = givens(i, j, a) @ R
    return R


def rot3(yaw, pitch, roll=0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


# ------------------------------------------------------------------ projection
class Proj:
    """4D -> 3D central projection (eye at w = d4) then 3D view rotation and perspective (eye at z = d3).
    Screen: centre (cx, cy), `unit` design px per unit at depth z = 0, w = 0."""

    def __init__(self, cx, cy, unit, d4=3.0, d3=7.0, view=None):
        self.cx, self.cy, self.unit, self.d4, self.d3 = cx, cy, unit, d4, d3
        self.view = np.eye(3) if view is None else view

    def to3(self, P4):
        P4 = np.atleast_2d(P4)
        k = self.d4 / (self.d4 - P4[:, 3])
        return P4[:, :3] * k[:, None], k

    def view3(self, P3):
        return P3 @ self.view.T

    def to2(self, Q):
        """Q: view-space 3D points -> (screen xy (n,2), perspective factor (n,))"""
        k = self.d3 / (self.d3 - Q[:, 2])
        x = self.cx + self.unit * Q[:, 0] * k
        y = self.cy - self.unit * Q[:, 1] * k
        return np.stack([x, y], 1), k

    def full(self, P4):
        P3, k4 = self.to3(P4)
        Q = self.view3(P3)
        S, k3 = self.to2(Q)
        return S, Q, k4, k3

    def inv2(self, x, y, z=0.0):
        """screen point at view depth z -> view-space 3D point"""
        k = self.d3 / (self.d3 - z)
        return np.array([(x - self.cx) / (self.unit * k), -(y - self.cy) / (self.unit * k), z])


# ------------------------------------------------------------------ exact sections
def clip_halfplane(poly, a, b, c):
    """keep a*x + b*y <= c  (Sutherland-Hodgman on a convex polygon list [(x,y)...])"""
    out = []
    n = len(poly)
    for k in range(n):
        p, q = poly[k], poly[(k + 1) % n]
        fp = a * p[0] + b * p[1] - c
        fq = a * q[0] + b * q[1] - c
        if fp <= 0:
            out.append(p)
        if (fp < 0 < fq) or (fq < 0 < fp):
            t = fp / (fp - fq)
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    return out


def section(R, z0=0.0, w0=0.0, big=10.0):
    """exact section of the rotated 24-cell (points p = R v) by the 2-plane {z = z0, w = w0}.
    Returns list of (a, b) with 4D points (a, b, z0, w0), CCW, or [] if empty."""
    Nw = N24 @ R.T          # rotated facet normals: (R n) . p <= d
    poly = [(-big, -big), (big, -big), (big, big), (-big, big)]
    for n, d in zip(Nw, D24):
        poly = clip_halfplane(poly, n[0], n[1], d - n[2] * z0 - n[3] * w0)
        if len(poly) < 3:
            return []
    # merge near-duplicate vertices
    out = []
    for p in poly:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-7:
            out.append(p)
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < 1e-7:
        out.pop()
    return out if len(out) >= 3 else []


def resample(poly, n=96, start_angle=math.pi * 0.75):
    """resample closed polygon to n points by arc length, starting at the boundary point in the direction
    start_angle from its centroid (so different polygons correspond point-to-point for morphing)."""
    P = np.array(poly, np.float64)
    c = P.mean(0)
    seg = np.roll(P, -1, 0) - P
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(L)])
    tot = cum[-1]
    # find start: ray from centroid
    d = np.array([math.cos(start_angle), math.sin(start_angle)])
    best, s0 = 1e18, 0.0
    for k in range(len(P)):
        a, e = P[k] - c, seg[k]
        M = np.array([[d[0], -e[0]], [d[1], -e[1]]])
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        t, u = np.linalg.solve(M, a)
        if t > 0 and 0 <= u <= 1 and t < best:
            best, s0 = t, cum[k] + u * L[k]
    ss = (s0 + np.linspace(0, tot, n, endpoint=False)) % tot
    idx = np.searchsorted(cum, ss, side="right") - 1
    idx = np.clip(idx, 0, len(P) - 1)
    u = (ss - cum[idx]) / np.maximum(L[idx], 1e-12)
    return P[idx] + seg[idx] * u[:, None]

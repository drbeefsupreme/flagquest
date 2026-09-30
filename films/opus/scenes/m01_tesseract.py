"""m01 - the Tesseract (tessera -> tesseract): 4D hypercube geometry, its rotation, projections, and its tiles.

A tesseract is 16 vertices {-1,1}^4, 32 edges, 24 square faces, 8 cubic cells.  Rotating it in a plane that
contains the 4th axis W (XW, YW) and projecting W with perspective turns the inner cube inside-out through the
outer one.  Seen straight along all four axes its shadow is a plain square: a tessera.
"""
import itertools
import math

import numpy as np
from scipy.linalg import expm, logm

V = np.array(list(itertools.product([-1.0, 1.0], repeat=4)))          # (16, 4)
EDGES = np.array([(i, j) for i in range(16) for j in range(i + 1, 16) if np.sum(V[i] != V[j]) == 1])  # (32, 2)
EAXIS = np.array([int(np.nonzero(V[i] != V[j])[0][0]) for i, j in EDGES])   # axis each edge runs along


def rot(i, j, a):
    R = np.eye(4)
    c, s = math.cos(a), math.sin(a)
    R[i, i], R[j, j], R[i, j], R[j, i] = c, c, -s, s
    return R


def rot3(ax, a):
    c, s = math.cos(a), math.sin(a)
    if ax == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _petrie():
    """rotation whose first two rows span the B4 Coxeter (Petrie) plane: the 8-fold 'star' shadow."""
    u1 = np.array([math.cos(k * math.pi / 4) for k in range(4)]) / math.sqrt(2)
    u2 = np.array([math.sin(k * math.pi / 4) for k in range(4)]) / math.sqrt(2)
    A = np.stack([u1, u2, np.eye(4)[2], np.eye(4)[3]], 0)
    Q, _ = np.linalg.qr(A.T)
    Q = Q.T
    Q[0] *= np.sign(Q[0] @ u1)
    Q[1] *= np.sign(Q[1] @ u2)
    if np.linalg.det(Q) < 0:
        Q[3] *= -1
    return Q


PETRIE = _petrie()


def slerp4(Ra, Rb, w):
    """geodesic between two 4x4 rotations"""
    if w <= 0:
        return Ra
    if w >= 1:
        return Rb
    Lg = np.real(logm(Ra.T @ Rb))
    return Ra @ np.real(expm(w * Lg))


def project(X, d4=3.2, V3=np.eye(3), d3=6.0, persp3=True):
    """X (..., 4) -> (xy (..., 2), depth (...), scale (...)). depth: larger = nearer the viewer."""
    k4 = d4 / (d4 - X[..., 3])
    P3 = X[..., :3] * k4[..., None]
    Z = P3 @ V3.T
    if persp3:
        k3 = d3 / (d3 - Z[..., 2])
    else:
        k3 = np.ones_like(Z[..., 2])
    return Z[..., :2] * k3[..., None], Z[..., 2] + 0.35 * X[..., 3], k4 * k3


class Tiling:
    """tile slots along the tesseract's edges (object space) + its vertex jewels."""

    def __init__(self, per_edge=12, seed=5):
        rng = np.random.default_rng(seed)
        f = (np.arange(per_edge) + 0.5) / per_edge
        A = V[EDGES[:, 0]]
        B = V[EDGES[:, 1]]
        self.per_edge = per_edge
        self.edge = np.repeat(np.arange(len(EDGES)), per_edge)
        self.f = np.tile(f, len(EDGES))
        self.X = A[self.edge] + (B - A)[self.edge] * self.f[:, None]           # (N, 4) object space
        wA, wB = A[:, 3], B[:, 3]
        # gold = the w=+1 cell's edges, silver = the w=-1 cell's; the W edges alternate (the 4th direction)
        kind = np.where(EAXIS == 3, 2, np.where(wA > 0, 0, 1))
        self.kind = kind[self.edge]
        idx = np.tile(np.arange(per_edge), len(EDGES))
        self.kind = np.where(self.kind == 2, idx % 2, self.kind)
        self.wedge = (EAXIS == 3)[self.edge]
        self.jit = rng.normal(0, 1, (len(self.X), 3))
        self.tilt = rng.normal(0, 1, (len(self.X), 2))

    def slots(self, R, zs=1.0, ws=1.0, d4=3.2, V3=np.eye(3), d3=6.0, scale=100.0, persp3=True):
        """screen-space tiles of one tesseract (centred on 0,0) in px: dict(x, y, ang, hu, hv, depth, kind, k)"""
        S = np.array([1.0, 1.0, zs, ws])
        X = (self.X * S) @ R.T
        Aw = (V[EDGES[:, 0]] * S) @ R.T
        Bw = (V[EDGES[:, 1]] * S) @ R.T
        xy, dep, k = project(X, d4, V3, d3, persp3)
        pa, _, ka = project(Aw, d4, V3, d3, persp3)
        pb, _, kb = project(Bw, d4, V3, d3, persp3)
        dv = (pb - pa)[self.edge]
        ln = np.linalg.norm(dv, axis=1) + 1e-9
        ang = np.arctan2(dv[:, 1], dv[:, 0])
        along = ln / self.per_edge * 0.5 * scale
        return dict(x=xy[:, 0] * scale, y=xy[:, 1] * scale, ang=ang, along=along, depth=dep, k=k,
                    kind=self.kind, edge=self.edge)

    @staticmethod
    def vertices(R, zs=1.0, ws=1.0, d4=3.2, V3=np.eye(3), d3=6.0, scale=100.0, persp3=True):
        S = np.array([1.0, 1.0, zs, ws])
        xy, dep, k = project((V * S) @ R.T, d4, V3, d3, persp3)
        return xy * scale, dep, k


def star_shape(R, zs=1.0, ws=1.0, d4=3.2, V3=np.eye(3), d3=6.0, persp3=True, ortho=False):
    """unit-scale 2D shadow of the tesseract: vertices (16,2) + edge midpoints (32,2) -> (48, 2), depth (48,)"""
    S = np.array([1.0, 1.0, zs, ws])
    Vx = (V * S) @ R.T
    M = 0.5 * (Vx[EDGES[:, 0]] + Vx[EDGES[:, 1]])
    X = np.vstack([Vx, M])
    if ortho:
        return X[:, :2].copy(), X[:, 2]
    xy, dep, _ = project(X, d4, V3, d3, persp3)
    return xy, dep


def petrie_shape():
    """the 8-fold star (orthographic Petrie shadow), normalised to outer radius 1"""
    X = np.vstack([V, 0.5 * (V[EDGES[:, 0]] + V[EDGES[:, 1]])]) @ PETRIE.T
    xy = X[:, :2]
    return xy / np.max(np.linalg.norm(xy[:16], axis=1))


def wire_shape(R, zs=1.0, ws=1.0, d4=3.2, V3=np.eye(3), d3=6.0, per_edge=3, ortho=False):
    """a small tesseract drawn in tiles: 16 vertex tiles + per_edge tiles along each of the 32 edges.
    -> xy (16 + 32*per_edge, 2), edge direction angle per tile (vertex tiles: 0), kind (0 vertex, 1 edge),
    depth (larger = nearer)."""
    S = np.array([1.0, 1.0, zs, ws])
    Vx = (V * S) @ R.T
    f = (np.arange(per_edge) + 0.5) / per_edge
    A = Vx[EDGES[:, 0]]
    B = Vx[EDGES[:, 1]]
    X = (A[:, None, :] + (B - A)[:, None, :] * f[None, :, None]).reshape(-1, 4)
    allX = np.vstack([Vx, X])
    if ortho:
        xy = allX[:, :2].copy()
        pa, pb = A[:, :2], B[:, :2]
        dep = allX[:, 2]
    else:
        xy, dep, _ = project(allX, d4, V3, d3)
        pa, _, _ = project(A, d4, V3, d3)
        pb, _, _ = project(B, d4, V3, d3)
    d = pb - pa
    ea = np.arctan2(d[:, 1], d[:, 0])
    ang = np.concatenate([np.zeros(16), np.repeat(ea, per_edge)])
    kind = np.concatenate([np.zeros(16, int), np.ones(32 * per_edge, int)])
    return xy, ang, kind, dep


def petrie_wire(per_edge=3):
    """the 8-fold star (the tesseract's Petrie shadow) as tiles, outer radius 1"""
    xy, ang, kind, dep = wire_shape(PETRIE, per_edge=per_edge, ortho=True)
    r = np.max(np.linalg.norm(xy[:16], axis=1))
    return xy / r, ang, kind

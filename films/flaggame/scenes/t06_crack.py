"""t06 firmament fracture: a Voronoi tessellation of the crystal canopy (in the canopy's horizontal (X, Z)
coordinates, X = r cos th, Z = r sin th) whose edges crack open as a front racing along the edges from the apex
(Dijkstra arrival times over the Voronoi graph, speed growing with distance), then the cells fall as 3D shards.

Per-pixel evaluation (numba): for every canopy pixel, the nearest two sites give the Voronoi edge, its two end
vertices give the crack arrival time along it -> anti-aliased crack core (white, the crystal catching the
firelight) + dark flanks, the cell id (for per-shard facet tones/angles and the holes).
"""
import heapq
import math

import numba as nb
import numpy as np


def build(seed=69, rmax=700.0):
    """spiderweb fracture: sites on jittered rings (geometric radii) x sectors -> radial + concentric cracks"""
    rng = np.random.default_rng(seed)
    pts = [(0.25, -0.15)]
    r = 0.75
    k = 0
    while r < rmax:
        n = 24 if r > 3.0 else 9 + int(r * 4)
        a0 = rng.uniform(0, 2 * math.pi) if k < 3 else 0.0
        for j in range(n):
            if rng.uniform() < 0.12 and r > 4.0:
                continue                      # a missing site merges two cells: irregular shards
            a = a0 + 2 * math.pi * (j + 0.5 * (k % 2) + rng.uniform(-0.32, 0.32)) / n
            rr = r * math.exp(rng.uniform(-0.09, 0.09))
            pts.append((rr * math.cos(a), rr * math.sin(a)))
        r *= rng.uniform(1.2, 1.36)
        k += 1
    P = np.array(pts)
    # far guard ring so every real cell is bounded
    ga = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    G = np.stack([np.cos(ga), np.sin(ga)], 1) * rmax * 1.6
    from scipy.spatial import Voronoi
    vor = Voronoi(np.concatenate([P, G]))
    n = len(P)
    V = vor.vertices
    # edge lookup: pair (i, j) of sites -> (va, vb)
    pair = {}
    adj = [[] for _ in range(len(V))]
    for (i, j), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in rv:
            continue
        a, b = rv
        pair[(min(i, j), max(i, j))] = (a, b)
        L = float(np.linalg.norm(V[a] - V[b]))
        mid = 0.5 * (V[a] + V[b])
        rm = float(np.linalg.norm(mid))
        radial = abs(float(np.dot((V[b] - V[a]) / (L + 1e-9), mid / (rm + 1e-9))))
        sp = (45.0 + 2.2 * rm) * (0.55 + 1.3 * radial ** 2) * rng.uniform(0.7, 1.2)
        adj[a].append((b, L / sp))
        adj[b].append((a, L / sp))
    # Dijkstra from the apex: start at the vertices of the apex cell
    tv = np.full(len(V), 1e9)
    reg = vor.regions[vor.point_region[0]]
    h = []
    for a in reg:
        if a >= 0:
            tv[a] = float(np.linalg.norm(V[a])) / 40.0
            heapq.heappush(h, (tv[a], a))
    while h:
        t, a = heapq.heappop(h)
        if t > tv[a]:
            continue
        for b, w in adj[a]:
            if t + w < tv[b]:
                tv[b] = t + w
                heapq.heappush(h, (tv[b], b))
    # dense pair table (sites x sites -> ridge index)
    keys = np.array(list(pair.keys()), np.int64)
    vals = np.array(list(pair.values()), np.int64)
    # bucket grid for nearest-site queries
    cell = 3.0
    lo = P.min(0) - 1.0
    gx = int((P[:, 0].max() - lo[0]) / cell) + 2
    gy = int((P[:, 1].max() - lo[1]) / cell) + 2
    # variable density -> use a coarse bucket + search radius from local spacing
    bid = ((P[:, 1] - lo[1]) // cell).astype(np.int64) * gx + ((P[:, 0] - lo[0]) // cell).astype(np.int64)
    order = np.argsort(bid, kind="stable")
    counts = np.bincount(bid, minlength=gx * gy)
    start = np.concatenate([[0], np.cumsum(counts)])
    # cell polygons (for shards)
    polys = []
    for i in range(n):
        rg = vor.regions[vor.point_region[i]]
        if -1 in rg or len(rg) < 3:
            polys.append(None)
            continue
        polys.append(V[rg])
    rc = np.hypot(P[:, 0], P[:, 1])
    # per-cell randomness: facet tone, hatch angle, fall delay, spin
    fac = rng.uniform(-1, 1, n)
    fang = rng.uniform(-0.9, 0.9, n)
    delay = rc / 70.0 + rng.uniform(0.0, 0.25, n) + 0.08 * np.minimum(rc, 12.0) * 0
    axis = rng.normal(size=(n, 3))
    axis[:, 1] *= 0.3
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    spin = rng.uniform(0.6, 2.4, n) * rng.choice([-1, 1], n)
    # sub-shards: each cell breaks into 2-5 pieces (fan from an off-centre point, adjacent triangles merged)
    subs = []
    for i, poly in enumerate(polys):
        if poly is None or rc[i] > 200.0:
            continue
        k = len(poly)
        c = poly.mean(0)
        c = c + (poly[rng.integers(k)] - c) * rng.uniform(0.0, 0.35)
        j = 0
        while j < k:
            m = int(rng.integers(1, 3))
            pts = [c] + [poly[(j + q) % k] for q in range(m + 1)]
            subs.append((i, np.array(pts), rng.normal(size=3), rng.uniform(0.8, 3.0) * rng.choice([-1, 1]),
                         rng.uniform(0.0, 0.12)))
            j += m
    return dict(P=P, V=V, tv=tv, keys=keys, vals=vals, lo=lo, cell=cell, gx=gx, gy=gy, order=order, start=start,
                polys=polys, rc=rc, fac=fac, fang=fang, delay=delay, axis=axis, spin=spin, n=n, subs=subs)


def pair_table(fr):
    n = fr["n"] + 64
    tab = -np.ones((n, n), np.int32) if n <= 4000 else None
    for (i, j), (a, b) in zip(fr["keys"], fr["vals"]):
        tab[i, j] = a
        tab[j, i] = b
    return tab


@nb.njit(cache=True, fastmath=True, parallel=True)
def _eval(X, Z, mpp, valid, P, n, lo0, lo1, cell, gx, gy, order, start, tab, V, tv, t, width,
          core, flank, cid, front):
    """X, Z: canopy coords per pixel (m); mpp: metres per pixel; t: seconds since the crack began"""
    h, w = X.shape
    R = 3
    for y in nb.prange(h):
        for x in range(w):
            cid[y, x] = -1
            core[y, x] = 0.0
            flank[y, x] = 0.0
            front[y, x] = 0.0
            if not valid[y, x]:
                continue
            px = X[y, x]
            pz = Z[y, x]
            bx = int((px - lo0) / cell)
            bz = int((pz - lo1) / cell)
            d1 = 1e18
            d2 = 1e18
            i1 = -1
            i2 = -1
            rr = R
            # grow the search until two sites are found and the ring is beyond d2
            while True:
                for jz in range(bz - rr, bz + rr + 1):
                    if jz < 0 or jz >= gy:
                        continue
                    for jx in range(bx - rr, bx + rr + 1):
                        if jx < 0 or jx >= gx:
                            continue
                        b = jz * gx + jx
                        for q in range(start[b], start[b + 1]):
                            s = order[q]
                            dx = P[s, 0] - px
                            dz = P[s, 1] - pz
                            d = dx * dx + dz * dz
                            if d < d1:
                                d2 = d1
                                i2 = i1
                                d1 = d
                                i1 = s
                            elif d < d2:
                                d2 = d
                                i2 = s
                if i2 >= 0 and math.sqrt(d2) < (rr - 0.5) * cell:
                    break
                if rr > 60:
                    break
                rr = rr * 2
            cid[y, x] = i1
            if i2 < 0:
                continue
            # distance to the bisector of (i1, i2)
            ex = P[i2, 0] - P[i1, 0]
            ez = P[i2, 1] - P[i1, 1]
            el = math.sqrt(ex * ex + ez * ez) + 1e-9
            dist = (d2 - d1) / (2.0 * el)                 # metres to the edge
            dpx = dist / max(mpp[y, x], 1e-6)             # pixels to the edge
            if dpx > width * 4.0 + 3.0:
                continue
            va = tab[i1, i2]
            vb = tab[i2, i1]
            if va < 0 or vb < 0:
                continue
            ta = tv[va]
            tb = tv[vb]
            if ta > 1e8 and tb > 1e8:
                continue
            # arrival time at the foot point along the edge (front runs both ways from the vertices)
            ax = V[va, 0]
            az = V[va, 1]
            bxx = V[vb, 0]
            bz_ = V[vb, 1]
            la = math.sqrt((px - ax) ** 2 + (pz - az) ** 2)
            lb = math.sqrt((px - bxx) ** 2 + (pz - bz_) ** 2)
            lab = math.sqrt((ax - bxx) ** 2 + (az - bz_) ** 2) + 1e-6
            sp = lab / max(abs(ta - tb), 1e-3) if abs(ta - tb) > 1e-3 else 200.0
            sp = max(sp, 40.0)
            tarr = min(ta + la / sp, tb + lb / sp)
            age = t - tarr
            if age <= 0.0:
                continue
            grow = min(1.0, age / 0.12)
            mx = 0.5 * (ax + bxx)
            mz = 0.5 * (az + bz_)
            mr = math.sqrt(mx * mx + mz * mz) + 1e-6
            rad = abs(((bxx - ax) * mx + (bz_ - az) * mz) / (lab * mr))
            wc = width * (0.45 + 0.9 * rad * rad) * (0.55 + 0.45 * grow) * (1.0 + min(1.0, age / 1.5))
            c = 1.0 - (dpx - wc * 0.5)
            if c > 0.0:
                core[y, x] = min(1.0, c) * grow
            fl = 1.0 - (dpx - wc * 1.6) / 1.5
            if fl > 0.0:
                flank[y, x] = min(1.0, fl) * grow
            if age < 0.08:
                front[y, x] = 1.0 - age / 0.08


def evaluate(fr, tab, X, Z, mpp, valid, t, width=1.6):
    h, w = X.shape
    core = np.zeros((h, w), np.float32)
    flank = np.zeros((h, w), np.float32)
    front = np.zeros((h, w), np.float32)
    cid = np.zeros((h, w), np.int32)
    _eval(X.astype(np.float32), Z.astype(np.float32), mpp.astype(np.float32), valid, fr["P"], fr["n"],
          float(fr["lo"][0]), float(fr["lo"][1]), float(fr["cell"]), fr["gx"], fr["gy"], fr["order"],
          fr["start"], tab, fr["V"], fr["tv"], float(t), float(width), core, flank, cid, front)
    return core, flank, cid, front


def sub_pose(sub, t_since, canopy_h):
    """3D polygon of a sub-shard t_since seconds after it let go"""
    i, pts, ax, spin, dl = sub
    t_since = t_since - dl
    r = np.hypot(pts[:, 0], pts[:, 1])
    Pw = np.stack([pts[:, 0], canopy_h(r), pts[:, 1]], 1)
    c = Pw.mean(0)
    if t_since <= 0:
        return Pw, c
    ts = t_since
    drop = 0.5 * 9.81 * ts * ts + 4.5 * ts
    ang = spin * ts * (0.7 + 0.5 * ts)
    a = ax / (np.linalg.norm(ax) + 1e-9)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    Rm = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
    out = (Pw - c) @ Rm.T + c
    dn = math.hypot(c[0], c[2]) + 1e-6
    out = out + np.array([c[0] / dn, 0.0, c[2] / dn]) * (0.9 * ts) - np.array([0.0, drop, 0.0])
    return out, out.mean(0)


def shard_pose(fr, i, t_since, canopy_h):
    """3D polygon of shard i t_since seconds after it let go (subdivided edges on the canopy)"""
    poly = fr["polys"][i]
    k = len(poly)
    pts = []
    for a in range(k):
        p, q = poly[a], poly[(a + 1) % k]
        for s in (0.0, 0.5):
            pts.append(p + (q - p) * s)
    pts = np.array(pts)
    r = np.hypot(pts[:, 0], pts[:, 1])
    Pw = np.stack([pts[:, 0], canopy_h(r), pts[:, 1]], 1)
    c = Pw.mean(0)
    if t_since <= 0:
        return Pw, c
    g = 9.81
    ts = t_since
    drop = 0.5 * g * ts * ts + 1.2 * ts
    ang = fr["spin"][i] * ts * (0.6 + 0.4 * ts)
    ax = fr["axis"][i]
    # Rodrigues rotation about the centroid
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    Rm = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
    out = (Pw - c) @ Rm.T + c
    drift = np.array([c[0], 0.0, c[2]])
    dn = np.linalg.norm(drift) + 1e-6
    out = out + drift / dn * (0.8 * ts) - np.array([0.0, drop, 0.0])
    return out, c - np.array([0.0, drop, 0.0])

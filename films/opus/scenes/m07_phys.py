"""m07 helper: the fall of the whole mosaic, as deterministic tile physics (pure functions of t).

World (vx.mosaic v1 convention): X right, Y DOWN, Z toward the viewer; the wall is z=0, the floor is y = FLOOR.

Every tessera i lives through four phases (all times GLOBAL seconds):
    set        t <  td[i]                 in its socket on the wall (home xy, in-plane angle ang)
    sheet      td[i] <= t < tb[i]         carried rigidly by its chunk (a sheet of setting bed peeling off / the
                                          tower's top hinging over): p = piv + R_c(th(t)) (home - piv) + drop(t)
    flight     tb[i] <= t < tl[i]         ballistic from the chunk's state at tb, tumbling about its own axis
    rattle     tl[i] <= t < tr[i]         lands on the heap (or the floor), bounces 0-3 times and skids to rest
    rest       t >= tr[i]
The heap is built once in setup (build()): tiles are dropped in landing order onto a heightfield over the floor
(x, z), slide down slopes steeper than the angle of repose and are deposited where they stop.
State at any t: Fall.at(t) -> (loose mask (N,), pos (N,3), quat (N,4)).
"""
import math

import numpy as np
from numba import njit

G = 1500.0                    # gravity, world units / s^2 (+y is down)
THICK = 6.5                   # tessera thickness
REPOSE = math.tan(math.radians(33.0))
HX0, HX1, HZ1, HD = -300.0, 3300.0, 1400.0, 6.0     # heightfield extent (x) and depth (z), cell size


# ============================================================ quaternions (w, x, y, z), vectorised
def q_axis(axis, ang):
    axis = np.asarray(axis, np.float64)
    n = np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-12
    a = axis / n
    h = 0.5 * np.asarray(ang, np.float64)[..., None]
    return np.concatenate([np.cos(h), a * np.sin(h)], -1)


def q_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def q_rot(q, v):
    """rotate vectors v (...,3) by unit quaternions q (...,4)"""
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def q_slerp(a, b, s):
    s = np.asarray(s, np.float64)[..., None]
    d = np.sum(a * b, -1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1))
    sn = np.sin(th)
    small = sn < 1e-4
    wa = np.where(small, 1 - s, np.sin((1 - s) * th) / np.where(small, 1, sn))
    wb = np.where(small, s, np.sin(s * th) / np.where(small, 1, sn))
    q = wa * a + wb * b
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def q_matrix(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def q_axis_angle(q):
    """unit quaternions -> axis-angle vectors (N,3)"""
    q = np.where(q[..., :1] < 0, -q, q)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    ang = 2 * np.arctan2(s, q[..., :1])
    return q[..., 1:] / np.maximum(s, 1e-12) * ang


# ============================================================ the heap
@njit(cache=True)
def _hget(Hf, x, z):
    fx = (x - HX0) / HD
    fz = z / HD
    nx, nz = Hf.shape
    if fx < 0:
        fx = 0.0
    if fz < 0:
        fz = 0.0
    if fx > nx - 1.001:
        fx = nx - 1.001
    if fz > nz - 1.001:
        fz = nz - 1.001
    i = int(fx)
    k = int(fz)
    a = fx - i
    b = fz - k
    return (Hf[i, k] * (1 - a) + Hf[i + 1, k] * a) * (1 - b) + (Hf[i, k + 1] * (1 - a) + Hf[i + 1, k + 1] * a) * b


@njit(cache=True)
def _build(order, pb, vb, tb, size, floor, g, skid_k, seeds, Hf, wall_z, obst):
    """drop tiles in `order` onto the heightfield. Returns landing time/pos/vel, rest pos, surface normal at rest."""
    n = pb.shape[0]
    tl = np.zeros(n)
    pl = np.zeros((n, 3))
    vl = np.zeros((n, 3))
    pr = np.zeros((n, 3))
    nr = np.zeros((n, 3))
    nxg, nzg = Hf.shape
    dt = 1.0 / 400.0
    for oi in range(n):
        i = order[oi]
        np.random.seed(seeds[i])
        px, py, pz = pb[i, 0], pb[i, 1], pb[i, 2]
        vx, vy, vz = vb[i, 0], vb[i, 1], vb[i, 2]
        tau = 0.0
        # ---- flight until the tile touches the heap surface (or the floor)
        while True:
            h = floor - py
            top = _hget(Hf, px, max(pz, 0.0)) + THICK * 0.5
            if h <= top and vy > 0:
                break
            if tau > 6.0:
                break
            px += vx * dt
            pz += vz * dt
            py += vy * dt + 0.5 * g * dt * dt
            vy += g * dt
            if pz < wall_z:            # the wall stops it
                pz = wall_z
                vz = abs(vz) * 0.3
            tau += dt
        tl[i] = tb[i] + tau
        pl[i, 0], pl[i, 1], pl[i, 2] = px, py, pz
        vl[i, 0], vl[i, 1], vl[i, 2] = vx, vy, vz
        # ---- skid along the horizontal velocity, then slide downhill while too steep
        vh = math.sqrt(vx * vx + vz * vz) + 1e-6
        hl = _hget(Hf, px, max(pz, 0.0))
        k = skid_k * (1.6 if hl < 2 * THICK else 1.0) * (0.5 + np.random.random())
        d = vh * k + 0.08 * vy * np.random.random()
        ux, uz = vx / vh, vz / vh
        # the impact scatters the direction a little
        a = (np.random.random() - 0.5) * 1.1
        ca, sa = math.cos(a), math.sin(a)
        ux, uz = ux * ca - uz * sa, ux * sa + uz * ca
        if uz < -0.2:                  # never into the wall: rebound off it
            uz = -uz
        x, z = px + ux * d, max(pz + uz * d, wall_z + size[i] * 0.3)
        for it in range(90):
            h0 = _hget(Hf, x, z)
            gx = (_hget(Hf, x + HD, z) - _hget(Hf, x - HD, z)) / (2 * HD)
            gz = (_hget(Hf, x, z + HD) - _hget(Hf, x, max(z - HD, 0.0))) / (2 * HD)
            sl = math.sqrt(gx * gx + gz * gz)
            if sl < REPOSE * (0.75 + 0.5 * np.random.random()):
                break
            step = 3.0
            x -= gx / sl * step + (np.random.random() - 0.5) * 1.5
            z -= gz / sl * step + (np.random.random() - 0.5) * 1.5
            if z < wall_z + 1.0:
                z = wall_z + 1.0
        # obstacles (discs on the floor: the candle stand) push the tile out
        for q in range(obst.shape[0]):
            ox, oz, orad = obst[q, 0], obst[q, 1], obst[q, 2]
            dx, dz = x - ox, z - oz
            dd = math.sqrt(dx * dx + dz * dz)
            if dd < orad:
                if dd < 1e-3:
                    dx, dz, dd = 1.0, 0.0, 1.0
                x = ox + dx / dd * orad
                z = oz + dz / dd * orad
        h0 = _hget(Hf, x, z)
        gx = (_hget(Hf, x + HD, z) - _hget(Hf, x - HD, z)) / (2 * HD)
        gz = (_hget(Hf, x, z + HD) - _hget(Hf, x, max(z - HD, 0.0))) / (2 * HD)
        pr[i, 0] = x
        pr[i, 1] = floor - (h0 + THICK * 0.5)
        pr[i, 2] = z
        nn = math.sqrt(gx * gx + gz * gz + 1.0)
        nr[i, 0], nr[i, 1], nr[i, 2] = -gx / nn, -1.0 / nn, -gz / nn      # up is -y
        # ---- deposit the tile's volume (loose packing) on a small kernel
        vol = size[i] * size[i] * THICK / 0.58
        fx = (x - HX0) / HD
        fz = z / HD
        ci, ck = int(fx + 0.5), int(fz + 0.5)
        rad = max(1, int(size[i] * 0.6 / HD + 0.5))
        wsum = 0.0
        for di in range(-rad, rad + 1):
            for dk in range(-rad, rad + 1):
                wsum += math.exp(-(di * di + dk * dk) / (0.6 * rad * rad + 0.3))
        for di in range(-rad, rad + 1):
            for dk in range(-rad, rad + 1):
                ii, kk = ci + di, ck + dk
                if 0 <= ii < nxg and 0 <= kk < nzg:
                    w = math.exp(-(di * di + dk * dk) / (0.6 * rad * rad + 0.3)) / wsum
                    Hf[ii, kk] += vol * w / (HD * HD)
    return tl, pl, vl, pr, nr


class Fall:
    def __init__(self, home, ang, size, td, chunk, piv, axis, alpha, t_sheet, v_kick, spin, floor, seed=0,
                 skid=0.16, obst=None, g=G, sag=None, arrays=None, damp=None):
        """home (N,2) wall chart xy; ang (N,) in-plane angle; td (N,) detach; chunk (N,) int; per chunk c:
        piv (C,3) pivot, axis (C,3) rotation axis, alpha (C,) angular accel (rad/s^2), t_sheet (C,) how long the
        sheet holds together, sag (C,) fraction of g the sheet drops with while it hangs (0 = hinged); per tile:
        v_kick (N,3) extra velocity at breakup, spin (N,3) tumble axis*rate. obst (K,3) floor discs (x, z, r).
        damp (N,): factor on the sheet's rigid velocity at breakup (pieces colliding lose momentum).
        arrays: {'f_<name>': array} from a previous build (skips the heap simulation)."""
        n = len(home)
        self.n = n
        self.g = g
        self.floor = floor
        self.home = np.c_[home, np.zeros(n)].astype(np.float64)
        self.q_home = q_axis(np.tile([0.0, 0.0, 1.0], (n, 1)), ang)
        self.size = np.asarray(size, np.float64)
        self.td = np.asarray(td, np.float64)
        self.chunk = np.asarray(chunk, np.int64)
        self.piv = np.asarray(piv, np.float64)
        self.axis = np.asarray(axis, np.float64) / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12)
        self.alpha = np.asarray(alpha, np.float64)
        self.tsh = np.asarray(t_sheet, np.float64)
        self.sag = np.full(len(self.tsh), 0.55) if sag is None else np.asarray(sag, np.float64)
        if arrays is not None:
            for k, v in arrays.items():
                setattr(self, k[2:], v)
            return
        rng = np.random.default_rng(seed)
        # per tile breakup time: the sheet's duration +- a little (edges go first)
        self.tb = self.td + self.tsh[self.chunk] * rng.uniform(0.7, 1.15, n)
        # state at breakup (rigid sheet motion)
        pb, qb, vb = self._sheet(self.tb)
        self.pb = pb
        self.qb = qb
        dmp = np.ones(n) if damp is None else np.asarray(damp, np.float64)
        self.vb = vb * dmp[:, None] + np.asarray(v_kick, np.float64)
        spin = np.asarray(spin, np.float64)
        self.spin_rate = np.linalg.norm(spin, axis=1)
        self.spin_axis = spin / (self.spin_rate[:, None] + 1e-12)
        # heap
        order = np.argsort(self.tb + np.sqrt(np.maximum(0, 2 * (floor - self.pb[:, 1])) / g), kind="stable")
        Hf = np.zeros((int((HX1 - HX0) / HD) + 1, int(HZ1 / HD) + 1))
        seeds = rng.integers(0, 2 ** 31 - 1, n)
        ob = np.zeros((0, 3)) if obst is None else np.asarray(obst, np.float64).reshape(-1, 3)
        self.tl, self.pl, self.vl, self.pr, nr = _build(order, self.pb, self.vb, self.tb, self.size, float(floor),
                                                        float(g), float(skid), seeds, Hf, 0.0, ob)
        self.heightfield = Hf
        # landing orientation (tumbling), rest orientation (lying on the heap surface, face up or down)
        self.ql = self._tumble(self.tl)
        face_up = rng.uniform(size=n) < 0.62
        target_n = np.where(face_up[:, None], nr, -nr)
        # minimal rotation taking the tile's current normal to the target normal, then a random yaw
        n_l = q_rot(self.ql, np.tile([0.0, 0.0, 1.0], (n, 1)))
        ax = np.cross(n_l, target_n)
        s = np.linalg.norm(ax, axis=1)
        c = np.sum(n_l * target_n, axis=1)
        ang_ = np.arctan2(s, c)
        ax = np.where(s[:, None] > 1e-6, ax / np.maximum(s, 1e-9)[:, None], np.tile([1.0, 0, 0], (n, 1)))
        q_align = q_axis(ax, ang_)
        yaw = q_axis(target_n, rng.normal(0, 0.6, n))
        self.qr = q_mul(yaw, q_mul(q_align, self.ql))
        # bounces: impact speed -> first hop height, number of hops, settle time
        vy = np.maximum(self.vl[:, 1], 0)
        e = rng.uniform(0.18, 0.34, n)
        h0 = np.minimum((e * vy) ** 2 / (2 * g), 26.0)
        self.hop = h0
        self.nhop = np.clip((h0 / 4.0).astype(int), 0, 3)
        hop_t = 2 * np.sqrt(2 * h0 / g)
        self.tr = self.tl + np.minimum(np.maximum(0.06, hop_t * (1 + e + e * e)) + rng.uniform(0.02, 0.1, n), 0.5)
        self.e = e

    # ---------------------------------------------------------------- phases
    def _sheet(self, t, i=None):
        """rigid chunk motion at times t (for tile indices i; all if None) (t >= td): pos, quat, velocity"""
        i = np.arange(self.n) if i is None else i
        t = np.broadcast_to(np.asarray(t, np.float64), (len(i),))
        c = self.chunk[i]
        tau = np.maximum(0.0, t - self.td[i])
        al = self.alpha[c]
        th = 0.5 * al * tau * tau
        om = al * tau
        qc = q_axis(self.axis[c], th)
        r = self.home[i] - self.piv[c]
        p = self.piv[c] + q_rot(qc, r)
        # the sheet also drops (it is no longer bedded): a fraction of g, growing (0 for a hinged body)
        sg = self.sag[c]
        drop = 0.5 * self.g * sg * tau * tau
        p[:, 1] += drop
        w = self.axis[c] * om[:, None]
        v = np.cross(w, p - self.piv[c] - np.c_[np.zeros(len(t)), drop, np.zeros(len(t))])
        v[:, 1] += self.g * sg * tau
        return p, q_mul(qc, self.q_home[i]), v

    def _tumble(self, t, i=None):
        i = np.arange(self.n) if i is None else i
        tau = np.maximum(0.0, np.asarray(t, np.float64) - self.tb[i])
        return q_mul(q_axis(self.spin_axis[i], self.spin_rate[i] * tau), self.qb[i])

    def at(self, t):
        """-> (idx, pos (M,3), quat (M,4)) of the M tiles that have left their sockets at global time t
        (quat = full world orientation incl. the in-plane angle; relative to home: q * conj(q_home))."""
        idx = np.nonzero(t >= self.td)[0]
        pos = np.empty((len(idx), 3))
        quat = np.empty((len(idx), 4))
        if not len(idx):
            return idx, pos, quat
        tb, tl, tr = self.tb[idx], self.tl[idx], self.tr[idx]
        # sheet
        m = t < tb
        if m.any():
            ii = idx[m]
            p, q, _ = self._sheet(t, ii)
            pos[m], quat[m] = p, q
        # flight
        m = (t >= tb) & (t < tl)
        if m.any():
            ii = idx[m]
            tau = (t - self.tb[ii])[:, None]
            pos[m] = self.pb[ii] + self.vb[ii] * tau + 0.5 * np.array([0.0, self.g, 0.0]) * tau * tau
            quat[m] = self._tumble(t, ii)
        # rattle: hops of decaying height while skidding to the rest point
        m = (t >= tl) & (t < tr)
        if m.any():
            ii = idx[m]
            s = np.clip((t - self.tl[ii]) / (self.tr[ii] - self.tl[ii]), 0, 1)
            k = 1 - (1 - s) ** 2.2                                  # skid eases out
            a, b = self.pl[ii], self.pr[ii]
            p = a + (b - a) * k[:, None]
            nh = self.nhop[ii]
            ph = s * (nh + 1)
            j = np.floor(ph)
            fr = ph - j
            amp = self.hop[ii] * (self.e[ii] ** (2 * j)) * (j <= nh)
            p[:, 1] -= amp * np.sin(np.pi * fr)
            pos[m] = p
            quat[m] = q_slerp(self.ql[ii], self.qr[ii], 1 - (1 - s) ** 3)
        m = t >= tr
        if m.any():
            ii = idx[m]
            pos[m] = self.pr[ii]
            quat[m] = self.qr[ii]
        return idx, pos, quat

    def moving(self, t):
        return (t >= self.td) & (t < self.tr)

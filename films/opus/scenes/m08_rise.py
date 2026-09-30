"""m08 helper - the Unbabeling itself: every tessera of the new wall has a source in the heaps (one of m07's fallen
tesserae), a launch time, a flight and a landing. Pure functions of global time T (frames render out of order).

World (vx.mosaic v1): design units, X right, Y DOWN, Z toward the viewer; the wall is the plane z = 0.

Choreography
  149.0        the pole strikes: the tiles around its foot hop and slide down the slope
  153.4-155.0  the heaps tremble ("It means only... Flag.")
  155.0        tiles_rise: the whole floor erupts; the NICHE tiles fly as ~100 separate flocks (one per person),
               each a comet that sets its own niche's person, inner courses first ("Each of them alone."); every
               niche that completes gets its Flag (pop) and its light.
  157.85-159.0 the GOLD: the rest of the heaps is swept up by one reversed avalanche, a front that rolls out of the
               planted Flag's foot and reaches the edges of the frame at exactly 159.0 ("All of them together." /
               convergence); then a ring of light runs back and converges on the Flag.
"""
import math

import numpy as np

T_TREMBLE = 153.4
T_RISE = 155.0
T_CELLS_END = 157.75
T_WAVE0 = 157.85          # first gold lands at the Flag's foot
T_CONV = 159.0            # the last tile lands


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _hash(i, k=0):
    x = (np.asarray(i, np.uint64) * np.uint64(2654435761) + np.uint64(k * 97 + 13)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(15)
    x = (x * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(13)
    return (x & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def rodrigues(v):
    """axis-angle (N,3) -> rotation matrices (N,3,3)"""
    v = np.asarray(v, np.float64)
    th = np.linalg.norm(v, axis=1)
    k = v / np.maximum(th, 1e-9)[:, None]
    K = np.zeros((len(v), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def log_rot(R):
    """rotation matrices (N,3,3) -> axis-angle (N,3) (used only as the start of a tumbling flight)"""
    R = np.asarray(R, np.float64)
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(tr)
    w = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.sin(th)
    k = np.where(s[:, None] > 1e-6, w / (2 * np.maximum(s, 1e-9))[:, None], 0.0)
    near = th > math.pi - 1e-3
    if np.any(near):
        k[near] = np.sqrt(np.clip((np.diagonal(R[near], axis1=1, axis2=2) + 1) / 2, 0, 1))
    return k * th[:, None]


class Rise:
    """wall: dict(xy (N,2) wall coords of the sockets, part (N,) int (niche id >= 0, negative = gold regions))
    cells: (C,) niche info: dict(k (row), th (angle), centre (x,y wall))
    heap: dict(pos (M,3), rot (M,3,3), size (M,), rgb (M,3), gold (M,), hidden (M,) bool) - the fallen tesserae;
          hidden = deep inside a heap (not drawn until it rises)
    centre: wall point of the planted Flag's foot (the compass point of the new picture)
    rho_vis: distance from centre to the farthest corner of the frame at 159.0 (the gold front arrives there)"""

    def __init__(self, wall, cells, heap, centre, plant=None, seed=11, rho_vis=None):
        rng = np.random.default_rng(seed)
        xy = np.asarray(wall["xy"], np.float64)
        part = np.asarray(wall["part"], np.int64)
        N = len(xy)
        self.N = N
        cx, cy = centre
        self.centre = (float(cx), float(cy))
        hp = np.asarray(heap["pos"], np.float64)
        M = len(hp)
        self.M = max(M, 1)
        # ------------------------------------------------ sources: the heap (+ tiles beneath it if sockets > heap)
        src = np.arange(N) % self.M
        buried = np.arange(N) >= M
        depth = np.where(buried, 1 + np.arange(N) // self.M, 0).astype(np.float64)
        jit = rng.normal(0, 1, (N, 3))
        off = np.stack([jit[:, 0] * 5, depth * 7 + np.abs(jit[:, 1]) * 3, -depth * 9 - np.abs(jit[:, 2]) * 4], 1)
        pos0 = hp[src] + off * buried[:, None]
        pos0[buried, 2] = np.maximum(pos0[buried, 2], 2.0)
        order = np.empty(N, np.int64)
        cells_n = len(cells)
        self.cells = cells
        is_cell = part >= 0
        free = np.ones(N, bool)
        slot_x = pos0[:, 0].copy()
        slot_depth = depth + rng.uniform(0, 0.4, N)
        # ---- niche flocks: each niche draws its tiles from a patch of heap below it (surface first); inner
        # courses first, and within a course from the top outward
        corder = sorted(range(cells_n), key=lambda c: (cells[c]["k"], abs(cells[c]["th"])))
        self.cell_rank = np.empty(cells_n, np.int64)
        self.cell_rank[corder] = np.arange(cells_n)
        by_x = np.argsort(slot_x)
        sorted_x = slot_x[by_x]
        for c in corder:
            idx = np.nonzero(part == c)[0]
            if len(idx) == 0:
                continue
            tx = cx + (cells[c]["centre"][0] - cx) * 0.85
            lo = np.searchsorted(sorted_x, tx)
            want = len(idx)
            a, b = max(0, lo - want * 3), min(N, lo + want * 3)
            cand = by_x[a:b]
            cand = cand[free[cand]]
            if len(cand) < want:
                cand = np.nonzero(free)[0]
            cost = np.abs(slot_x[cand] - tx) * 0.02 + slot_depth[cand]
            pick = cand[np.argsort(cost, kind="stable")[:want]]
            free[pick] = False
            order[idx[np.argsort(xy[idx, 0])]] = pick[np.argsort(slot_x[pick])]
        # ---- the gold: matched by distance from the compass point (near tiles take near slots)
        gidx = np.nonzero(~is_cell)[0]
        rho_t = np.hypot(xy[gidx, 0] - cx, xy[gidx, 1] - cy)
        fs = np.nonzero(free)[0]
        rho_s = np.hypot(pos0[fs, 0] - cx, (pos0[fs, 1] - cy) * 0.5) + slot_depth[fs] * 40
        order[gidx[np.argsort(rho_t + rng.uniform(0, 120, len(gidx)))]] = fs[np.argsort(rho_s)][:len(gidx)]
        self.slot = order
        self.hidx = src[order]                      # heap tessera each socket is made from
        P0 = pos0[order]
        self.P0 = P0
        self.R0 = np.asarray(heap["rot"], np.float64)[self.hidx]
        self.v0 = log_rot(self.R0)
        self.old_rgb = np.asarray(heap["rgb"], np.float32)[self.hidx]
        self.old_gold = np.asarray(heap["gold"], np.float32)[self.hidx]
        self.src_size = np.asarray(heap["size"], np.float32)[self.hidx]
        hid = np.asarray(heap.get("hidden", np.zeros(M, bool)), bool)
        self.hidden = hid[self.hidx] | buried[order]
        self.P3 = np.concatenate([xy, np.zeros((N, 1))], 1)
        self.part = part
        self.is_cell = is_cell
        # ------------------------------------------------ timing
        cr = self.cell_rank[np.clip(part, 0, None)]
        frac = cr / max(1, cells_n - 1)
        # niches: the whole floor erupts at once (155.0 .. 156.0), the flocks set their people from the inner
        # course outward, 155.9 .. 157.75 ("Each of them alone")
        u = _hash(np.arange(N), 1)
        tl_c = T_RISE + 0.85 * frac ** 0.7 + 0.15 * u
        land_c = np.minimum(155.9 + 1.75 * frac + 0.1 * _hash(np.arange(N), 2), T_CELLS_END)
        # gold: lands with a front that rolls out of the Flag's foot and reaches the frame's edges at 159.0
        rho = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
        self.rho_max = float(rho_vis) if rho_vis else (float(rho[~is_cell].max()) if np.any(~is_cell) else 1.0)
        land_g = T_WAVE0 + (T_CONV - T_WAVE0) * np.clip(rho / self.rho_max, 0, 1) ** 0.8
        land_g = np.minimum(land_g + 0.05 * (_hash(np.arange(N), 3) - 0.5), T_CONV)
        du_g = 0.5 + 0.25 * _hash(np.arange(N), 4)
        self.t_launch = np.where(is_cell, tl_c, land_g - du_g)
        self.dur = np.where(is_cell, land_c - tl_c, du_g)
        self.t_land = self.t_launch + self.dur
        self.rho = rho
        # a niche is complete when its last tile has landed
        self.cell_done = np.full(cells_n, T_CELLS_END)
        for c in range(cells_n):
            m = part == c
            if np.any(m):
                self.cell_done[c] = self.t_land[m].max()
        # ------------------------------------------------ flock geometry (niches)
        self.flock_from = np.zeros((cells_n, 3))
        self.flock_to = np.zeros((cells_n, 3))
        for c in range(cells_n):
            m = part == c
            if np.any(m):
                self.flock_from[c] = P0[m].mean(0)
                self.flock_to[c] = (cells[c]["centre"][0], cells[c]["centre"][1], 0.0)
        self.spin = rng.normal(0, 1, (N, 3)) * np.where(is_cell, 5.0, 7.0)[:, None]
        self.swirl = rng.uniform(-1, 1, N)
        # ------------------------------------------------ the strike: tiles round the pole's foot hop down the slope
        self.t_plant = None
        if plant is not None:
            px, py, pz, t_pl = plant
            d = np.hypot(P0[:, 0] - px, (P0[:, 1] - py) * 1.2)
            near = (d < 90) & (P0[:, 2] > pz - 60)
            ang = np.arctan2(P0[:, 1] - py, P0[:, 0] - px)
            amp = np.where(near, (1 - d / 90) * 38 * (0.5 + _hash(np.arange(N), 5)), 0.0)
            self.kick = np.stack([np.cos(ang) * amp, np.abs(np.sin(ang)) * amp * 0.5 + amp * 0.35, amp * 0.5], 1)
            self.kick_h = np.where(near, (1 - d / 90) * 60 * (0.4 + _hash(np.arange(N), 6)), 0.0)
            self.t_plant = t_pl
            self.P0_pre = P0
            self.P0 = P0 + self.kick

    # ---------------------------------------------------------------- queries
    def landed(self, T):
        return self.t_land <= T

    def gain(self, T):
        """wall-tile light multiplier: a flash when a tile sets; at 159.0 (convergence) a ring of light runs back
        from the frame's edges and converges on the Flag's foot (159.0 .. 159.5)"""
        dt = T - self.t_land
        g = np.where(dt >= 0, 1.0 + 1.6 * np.exp(-np.maximum(dt, 0) / 0.16), 1.0)
        if T_CONV - 0.05 <= T <= T_CONV + 0.7:
            r_in = self.rho_max * (1.0 - np.clip((T - T_CONV) / 0.5, 0.0, 1.0))
            g = g + 1.1 * np.exp(-((self.rho - r_in) / 170.0) ** 2) * (1.0 - np.clip((T - T_CONV - 0.5) / 0.2, 0, 1))
        return g

    def pop(self, T):
        """per-niche Flag appearance 0..1 (pop animation of vx.flag)"""
        return np.clip((T - self.cell_done - 0.04) / 0.55, 0.0, 1.0)

    def loose(self, T):
        """every tessera not (yet) set into the wall that can be seen: flying, hopping, or resting on a heap (those
        buried deep inside a heap are skipped until they rise).
        -> dict(idx, pos (K,3), rot (K,3,3), vel (K,3), flying (K,) bool, tau (K,))"""
        idx = np.nonzero(self.t_land > T)[0]
        fly = T >= self.t_launch[idx]
        keep = fly | ~self.hidden[idx]
        idx, fly = idx[keep], fly[keep]
        pos = np.empty((len(idx), 3))
        rot = np.empty((len(idx), 3, 3))
        vel = np.zeros((len(idx), 3))
        tau = np.zeros(len(idx))
        # ---- resting (with the strike's splash + the tremble)
        r = idx[~fly]
        p = self._splash(r, T) if self.t_plant is not None and T < self.t_plant + 0.8 else self.P0[r].copy()
        R = self.R0[r]
        if T > T_TREMBLE:
            a = min(1.0, (T - T_TREMBLE) / (T_RISE - T_TREMBLE)) ** 2
            ph = T * 31.0
            p[:, 0] += np.sin(ph + r * 1.7) * np.cos(ph * 1.3 + r * 0.37) * 2.2 * a
            p[:, 1] -= np.abs(np.sin(ph * 1.9 + r * 2.3)) * 3.0 * a * (0.3 + _hash(r, 7))
            wob = np.stack([np.sin(ph + r), np.cos(ph * 0.7 + r), np.sin(ph * 1.1 + 2 * r)], 1) * 0.12 * a
            R = R @ rodrigues(wob)
        pos[~fly] = p
        rot[~fly] = R
        # ---- flying
        f = idx[fly]
        if len(f):
            tt = (T - self.t_launch[f]) / self.dur[f]
            P, V, RV = self._flight(f, np.clip(tt, 0, 1))
            pos[fly] = P
            vel[fly] = V
            rot[fly] = rodrigues(RV)
            tau[fly] = tt
        return dict(idx=idx, pos=pos, rot=rot, vel=vel, flying=fly, tau=tau)

    def _splash(self, r, T):
        """resting tiles around the planted foot: a hop from the old rest to the new one"""
        pre = self.P0_pre[r]
        t = T - self.t_plant
        if t < 0:
            return pre.copy()
        s = min(1.0, t / 0.45)
        p = pre + self.kick[r] * (s * (2 - s))
        hop = self.kick_h[r] * 4 * s * (1 - s)
        p[:, 1] -= hop
        p[:, 2] += hop * 0.3
        return p

    def _flight(self, f, tau):
        """positions of flying tiles at progress tau (0 launch -> 1 set). A fall run backwards: they leave the heap
        fast and slow down into their sockets, arriving flat (identity rotation) in the wall plane."""
        P0, P3 = self.P0[f], self.P3[f]
        u = 1 - (1 - tau) ** 2.2
        du = 2.2 * (1 - tau) ** 1.2 / np.maximum(self.dur[f], 1e-3)
        cell = self.is_cell[f]
        rise = np.abs(P3[:, 1] - P0[:, 1])
        z0 = np.zeros(len(f))
        P1 = P0 + np.stack([z0, -(0.35 * rise + 60), 60 + 0.1 * rise], 1)
        P2 = P3 + np.stack([z0, 0.25 * rise + 30, 120 + 0.05 * rise], 1)
        pos, dpos = _bez(P0, P1, P2, P3, u)
        if np.any(cell):
            # niche flocks follow their flock's shared comet path; offsets shrink mid-flight (a tight comet)
            c = self.part[f[cell]]
            A, D = self.flock_from[c], self.flock_to[c]
            r = np.abs(D[:, 1] - A[:, 1])
            zc = np.zeros(len(c))
            B1 = A + np.stack([zc, -(0.4 * r + 80), 140 + 0.15 * r], 1)
            B2 = D + np.stack([self.swirl[f[cell]] * 60, 0.3 * r + 60, 180 + 0.1 * r], 1)
            uc = u[cell]
            F, dF = _bez(A, B1, B2, D, uc)
            squeeze = 1 - 0.75 * np.sin(np.pi * uc) ** 0.7
            w = _smooth((uc - 0.55) / 0.45)
            off = ((P0[cell] - A) * (1 - w)[:, None] + (P3[cell] - D) * w[:, None]) * squeeze[:, None]
            a = self.swirl[f[cell]] * 2.2 * np.sin(np.pi * uc)           # the comet spirals round its own axis
            ca, sa = np.cos(a), np.sin(a)
            off = np.stack([off[:, 0] * ca - off[:, 2] * sa, off[:, 1], off[:, 0] * sa + off[:, 2] * ca], 1)
            pos[cell] = F + off
            dpos[cell] = dF
        vel = dpos * du[:, None]
        k = 1 - u                                   # tumble: from the heap orientation to flat, spinning between
        rv = self.v0[f] * k[:, None] + self.spin[f] * (np.sin(np.pi * u) * k)[:, None]
        return pos, vel, rv

    # ---------------------------------------------------------------- sound
    def foley(self, S):
        """per-frame tracks for the sound team (scene-local frames)"""
        n = S.n
        T = S.start + np.arange(n) / S.fps
        lift = np.zeros(n, np.float32)
        land = np.zeros(n, np.float32)
        infl = np.zeros(n, np.float32)
        panv = np.zeros(n, np.float32)
        np.add.at(lift, np.clip(((self.t_launch - S.start) * S.fps).astype(np.int64), 0, n - 1), 1.0)
        np.add.at(land, np.clip(((self.t_land - S.start) * S.fps).astype(np.int64), 0, n - 1), 1.0)
        for i, t in enumerate(T):
            m = (self.t_launch <= t) & (self.t_land > t)
            infl[i] = m.mean()
            if np.any(m):
                panv[i] = float(np.clip((self.P3[m, 0].mean() - 960) / 960, -1, 1))
        set_frac = np.array([(self.t_land <= t).mean() for t in T], np.float32)
        return dict(energy=infl, pan=panv, lift=lift, land=land, set=set_frac)


def _bez(P0, P1, P2, P3, u):
    u = u[:, None]
    v = 1 - u
    p = v * v * v * P0 + 3 * v * v * u * P1 + 3 * v * u * u * P2 + u * u * u * P3
    d = 3 * v * v * (P1 - P0) + 6 * v * u * (P2 - P1) + 3 * u * u * (P3 - P2)
    return p, d

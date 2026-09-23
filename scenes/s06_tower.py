"""s06 helper: a real 3D Tower of Babel assembled from every fragment of the shattered memeplexes.

World units: X right, Y up, Z away from the default camera. The tower is a leaning stack of NT
cylindrical tiers; every tier wall is masonry made of *fragments* (temple stones, marble chunks,
map shards, slop cards, phones, arcade openings). Each block has a life: it orbits in the flat
galaxy swirl inherited from s05b, lifts along a helix, tumbles, and snaps into its slot. Later the
same blocks crack loose and fly. Everything is a pure function of time (no stepping).
"""
import math

import cairo
import numpy as np

from vx import W, H, C, col, mix, ease_in_out, ease_out_back, clamp, smoothstep, hash01

# ------------------------------------------------------------------------------------------ camera


class Cam3:
    """pinhole camera; yaw about +Y (turn right), pitch about camera X (look up), roll on screen"""

    def __init__(self, pos=(0, 0, -3000), yaw=0.0, pitch=0.0, roll=0.0, f=1250.0, cx=W / 2, cy=H / 2):
        self.pos = np.asarray(pos, np.float64)
        self.yaw, self.pitch, self.roll, self.f, self.cx, self.cy = yaw, pitch, roll, f, cx, cy
        cp, sp = math.cos(pitch), math.sin(pitch)
        cyw, syw = math.cos(yaw), math.sin(yaw)
        self.fw = np.array([syw * cp, sp, cyw * cp])
        self.rt = np.array([cyw, 0.0, -syw])
        self.up = np.cross(self.fw, self.rt)
        self.cr, self.sr = math.cos(roll), math.sin(roll)

    @classmethod
    def orbit(cls, target, dist, yaw, pitch, **kw):
        c = cls((0, 0, 0), yaw, pitch, **kw)
        c.pos = np.asarray(target, np.float64) - c.fw * dist
        return c

    pre = None   # optional world-space deformation applied before projection (tower collapse)

    def proj(self, P):
        """P (...,3) -> (sx, sy, depth) arrays in design px"""
        P = np.asarray(P, np.float64)
        if self.pre is not None:
            P = self.pre(P)
        d = P - self.pos
        x = d @ self.rt
        y = d @ self.up
        z = d @ self.fw
        zc = np.maximum(z, 1.0)
        sx = x / zc * self.f
        sy = -y / zc * self.f
        if self.roll:
            sx, sy = sx * self.cr - sy * self.sr, sx * self.sr + sy * self.cr
        return sx + self.cx, sy + self.cy, z

    def scale_at(self, z):
        return self.f / np.maximum(z, 1.0)

    def ray_dirs(self, xs, ys):
        """world ray directions for screen coords (arrays)"""
        dx, dy = xs - self.cx, ys - self.cy
        if self.roll:
            dx, dy = dx * self.cr + dy * self.sr, -dx * self.sr + dy * self.cr
        d = (self.fw[None, None, :] * self.f + self.rt[None, None, :] * dx[..., None]
             - self.up[None, None, :] * dy[..., None])
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


# ------------------------------------------------------------------------------------------ materials
STONE, MARBLE, MAP, SLOP, PHONE, ARCH = range(6)
_MAT_COLS = {
    STONE: ["#5E5668", "#6E7280", "#4A4E5C", "#6B5A6E", "#565A6A", "#7A7486"],
    MARBLE: ["#DCD8D2", "#E6DFD0", "#CFCBD0", "#D8D4DA"],
    MAP: ["#3F6E6E", "#46597A", "#5A6E7A", "#4E6A60", "#6A7F8C", "#587066"],
    SLOP: ["#E8C8D0", "#B8E0E8", "#C9C0A8", "#8A7F8F", "#D8B8C8", "#A8C8D8"],
    PHONE: ["#1A1C24", "#22252E", "#15131A"],
    ARCH: ["#4A4E5C", "#5E5668", "#565A6A"],
}
SCREEN_COLS = ["#9FD8FF", "#B8E0E8", "#9FD8FF", "#DDE8FF", "#C8D0FF", "#9FD8FF", "#B8E0E8", "#FF2E88",
               "#00E5FF", "#E8C8D0", "#AFC8FF", "#DDE8FF"]


def _hex(c):
    return np.array(col(c), np.float64)


# ------------------------------------------------------------------------------------------ tower


LEAN_X, LEAN_Z = 760.0, -120.0


class Tower:
    NT = 11
    YSW = -300.0          # swirl plane height
    PERSON_H = 58.0

    def __init__(self, seed=6):
        rng = np.random.default_rng(seed)
        NT = self.NT
        self.h = np.array([430.0 - 12 * k for k in range(NT)])
        self.y0 = -520.0 + np.concatenate([[0], np.cumsum(self.h)[:-1]])
        self.y1 = self.y0 + self.h
        self.r = np.array([1450.0 - 108 * k for k in range(NT)])
        self.top = float(self.y1[-1])
        self.total = self.top - self.y0[0]
        # leaning stack: each drum is vertical, shifted along +X (and a touch toward camera)
        ym = (self.y0 + self.y1) / 2
        u = (ym - self.y0[0]) / self.total
        self.ax = LEAN_X * u ** 1.7
        self.az = LEAN_Z * u ** 2
        self.ym = ym
        self._build_blocks(rng)
        self._build_people(rng)
        self._build_crowns(rng)

    def axis_at(self, k):
        return self.ax[k], self.az[k]

    def axis_y(self, y):
        u = np.clip((np.asarray(y) - self.y0[0]) / self.total, 0, 1.2)
        return LEAN_X * u ** 1.7, LEAN_Z * u ** 2

    # ---------------------------------------------------------------- blocks
    def _build_blocks(self, rng):
        B = []
        for k in range(self.NT):
            r = self.r[k]
            nc = max(12, int(round(2 * math.pi * r / 165)))
            dth = 2 * math.pi / nc
            fr = rng.uniform(0.28, 0.38, 3)
            fr = fr / fr.sum()
            ys = self.y0[k] + np.concatenate([[0], np.cumsum(fr)]) * self.h[k]
            hk = k / (self.NT - 1)
            # strata of meaning: temples & marble below, nations mid, slop & phones on top
            pm = np.array([
                0.46 - 0.40 * hk,        # stone
                0.34 - 0.24 * hk,        # marble
                0.14 + 0.24 * math.sin(math.pi * hk),  # map shards
                0.04 + 0.38 * hk,        # slop cards
                0.02 + 0.24 * hk ** 1.5,  # phones
            ])
            pm = np.maximum(pm, 0.01)
            pm /= pm.sum()
            for row in range(3):
                off = (0.5 * (row % 2) + rng.uniform(-0.15, 0.15)) * dth
                j = 0
                while j < nc:
                    span = 2 if (rng.random() < 0.22 and j < nc - 1) else 1
                    th0 = off + j * dth
                    th1 = off + (j + span) * dth
                    if row == 1 and span == 1 and (j % 3 == 1) and rng.random() < 0.85:
                        mat = ARCH
                    else:
                        mat = int(rng.choice(5, p=pm))
                    cols = _MAT_COLS[mat]
                    c = _hex(cols[int(rng.integers(len(cols)))]) * rng.uniform(0.9, 1.08)
                    prot = rng.uniform(-4, 8) if rng.random() < 0.7 else rng.uniform(12, 30)
                    if mat == ARCH:
                        prot = rng.uniform(-8, -2)
                    B.append((k, row, th0, th1, ys[row], ys[row + 1], r + prot, mat,
                              c[0], c[1], c[2], rng.random()))
                    j += span
        A = np.array(B, np.float64)
        self.nb = len(A)
        self.b_tier = A[:, 0].astype(int)
        self.b_row = A[:, 1].astype(int)
        self.b_th0, self.b_th1 = A[:, 2], A[:, 3]
        self.b_thm = (A[:, 2] + A[:, 3]) / 2
        self.b_y0, self.b_y1 = A[:, 4], A[:, 5]
        self.b_ym = (A[:, 4] + A[:, 5]) / 2
        self.b_r = A[:, 6]
        self.b_mat = A[:, 7].astype(int)
        self.b_col = A[:, 8:11]
        self.b_seed = A[:, 11]
        self.b_scr = np.array([_hex(SCREEN_COLS[int(s * 997) % len(SCREEN_COLS)]) for s in self.b_seed])
        n = self.nb
        # the shard each fragment is before it is squared into a brick (unit-square coords, 7 pts)
        sh = np.zeros((n, 7, 2))
        for i in range(n):
            m = self.b_mat[i]
            if m in (SLOP, PHONE):          # cards & phones stay rectangles, just smaller
                w, h = rng.uniform(0.55, 0.75), rng.uniform(0.6, 0.85)
                pts = [(0.5 - w / 2, 0.5 - h / 2), (0.5 + w / 2, 0.5 - h / 2), (0.5 + w / 2, 0.5 + h / 2),
                       (0.5 - w / 2, 0.5 + h / 2)]
            else:
                k = 3 if (m == MARBLE and rng.random() < 0.6) else int(rng.integers(4, 7))
                ang = np.sort(rng.uniform(0, 2 * math.pi, k))
                rad = rng.uniform(0.28, 0.5, k)
                pts = [(0.5 + r * math.cos(a), 0.5 + r * math.sin(a)) for a, r in zip(ang, rad)]
            pts = pts + [pts[-1]] * (7 - len(pts))
            sh[i] = np.clip(np.array(pts), 0.0, 1.0)
        self.b_shard = sh
        self.b_spinax = rng.normal(size=(n, 3))
        self.b_spinax /= np.linalg.norm(self.b_spinax, axis=1, keepdims=True)
        self.b_spin = rng.uniform(2.5, 7.0, n) * rng.choice([-1, 1], n)
        self.b_psi0 = rng.uniform(0, 2 * math.pi, n)
        self.b_psid = rng.uniform(-1.5, 1.5, n)
        # detach (collapse) parameters
        self.b_vout = rng.uniform(250, 900, n)
        self.b_vup = rng.uniform(150, 700, n)
        self.b_vtan = rng.uniform(-300, 300, n)
        self.b_detr = rng.random(n)

    def schedule(self, tier_windows, rng_seed=66):
        """tier_windows[k] = (t_start, t_end) local seconds during which tier k's blocks snap in.
        Assign arrival (ta), lift (tl) times and the initial galaxy swirl position of every block."""
        rng = np.random.default_rng(rng_seed)
        n = self.nb
        ta = np.zeros(n)
        for k in range(self.NT):
            m = self.b_tier == k
            a, b = tier_windows[k]
            # helical build order: around (CCW) and up
            th = np.mod(self.b_thm[m] - 1.2, 2 * math.pi) / (2 * math.pi)
            order = (self.b_row[m] + th) / 3.0
            ta[m] = a + (b - a) * np.clip(order + rng.uniform(-0.06, 0.06, m.sum()), 0, 1)
        self.ta = ta
        # galaxy points (2 log-spiral arms + halo) — matches s05b's last frame by construction
        R0, R1 = 300.0, 3900.0
        m = n
        s = rng.random(m) ** 0.85
        rho = R0 + (R1 - R0) * s
        arm = rng.integers(0, 2, m)
        phi = arm * math.pi + 2.2 * np.log(rho / R0) + rng.normal(0, 0.32, m)
        # greedy match: earliest blocks take the swirl points nearest their slot radius
        order = np.argsort(ta)
        free = np.ones(m, bool)
        rho0 = np.zeros(n)
        phi0 = np.zeros(n)
        for i in order:
            want = self.b_r[i] + 160
            cost = np.abs(rho - want) + (~free) * 1e9
            j = int(np.argmin(cost))
            free[j] = False
            rho0[i], phi0[i] = rho[j], phi[j]
        self.rho0, self.phi0 = rho0, phi0
        dist = np.abs(self.b_ym - self.YSW)
        flight = np.clip(0.55 + dist / 2600.0, 0.55, 2.3) + rng.uniform(0, 0.35, n)
        self.tl = np.maximum(ta - flight, 0.1 + rng.uniform(0, 0.35, n))
        self.tl = np.minimum(self.tl, ta - 0.3)
        self.tier_done = np.array([ta[self.b_tier == k].max() for k in range(self.NT)])

    @staticmethod
    def omega(rho):
        """swirl angular speed (rad/s): 0.5 rad/s at 300 screen px (rho≈960), ~1/sqrt(r)"""
        return 0.5 * np.sqrt(960.0 / np.maximum(rho, 150.0))

    def slot_corners(self, idx=None, lift=None):
        """exact landed corners (n,4,3): TL, TR, BR, BL (TL = th0 top)"""
        i = slice(None) if idx is None else idx
        k = self.b_tier[i]
        ax, az = self.ax[k], self.az[k]
        r = self.b_r[i]
        th0, th1 = self.b_th0[i], self.b_th1[i]
        y0, y1 = self.b_y0[i], self.b_y1[i]
        c0x, c0z = ax + r * np.cos(th0), az + r * np.sin(th0)
        c1x, c1z = ax + r * np.cos(th1), az + r * np.sin(th1)
        P = np.stack([np.stack([c0x, y1, c0z], -1), np.stack([c1x, y1, c1z], -1),
                      np.stack([c1x, y0, c1z], -1), np.stack([c0x, y0, c0z], -1)], 1)
        return P

    # ---------------------------------------------------------------- per-frame block state
    def blocks_at(self, t, collapse=None):
        """returns dict: P (n,4,3) corners, N (n,3) normals, landed (n,) bool, alive (n,) bool,
        two_sided (n,) bool. collapse = dict(t0=..., front=fn(t)->y, ...) for the crack phase."""
        n = self.nb
        P = self.slot_corners()
        N = np.stack([np.cos(self.b_thm), np.zeros(n), np.sin(self.b_thm)], -1)
        landed = t >= self.ta
        two = ~landed
        fl = ~landed
        morph = np.ones(n)
        if fl.any():
            idx = np.nonzero(fl)[0]
            Pf, Nf = self._flight(idx, t)
            # blend final approach into exact slot corners
            u = np.clip((t - self.tl[idx]) / (self.ta[idx] - self.tl[idx]), 0, 1)
            a = smooth_arr(0.93, 1.0, u)[:, None, None]
            P[idx] = Pf * (1 - a) + P[idx] * a
            N[idx] = Nf
            morph[idx] = smooth_arr(0.72, 0.96, u)
        if collapse is not None:
            det, Pd, Nd = collapse_state(self, t, collapse, P)
            P[det] = Pd
            N[det] = Nd
            two = two | det
        return dict(P=P, N=N, landed=landed, two=two, morph=morph)

    def _flight(self, idx, t):
        k = self.b_tier[idx]
        tl, ta = self.tl[idx], self.ta[idx]
        rho0, phi0 = self.rho0[idx], self.phi0[idx]
        w0 = self.omega(rho0)
        ts = np.minimum(t, tl)
        # in the swirl: slow inward drift, orbit
        rho_s = rho0 * (1 - 0.04 * ts)
        phi_s = phi0 + w0 * ts
        T = np.maximum(ta - tl, 1e-3)
        u = np.clip((t - tl) / T, 0, 1)
        flying = t > tl
        # helix to the slot
        th_e = self.b_thm[idx]
        wavg = 1.25
        nturn = np.round((phi_s + wavg * T - th_e) / (2 * math.pi))
        phi_e = th_e + 2 * math.pi * nturn
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        phi = h00 * phi_s + h10 * T * w0 + h01 * phi_e + h11 * T * 0.25
        r_s = self.b_r[idx]
        u1 = np.clip(u / 0.84, 0, 1)
        e1 = u1 * u1 * (3 - 2 * u1)
        rho = rho_s + (r_s + 230 - rho_s) * e1
        u2 = np.clip((u - 0.84) / 0.16, 0, 1)
        s = 1.25
        u2m = u2 - 1
        eb = u2m * u2m * ((s + 1) * u2m + s) + 1
        rho = np.where(u > 0.84, r_s + 230 * (1 - eb), rho)
        Y = self.YSW + (self.b_ym[idx] - self.YSW) * smooth_arr(0.0, 0.9, u)
        # swirl bob before lift
        Y = Y + np.where(flying, 0, 18 * np.sin(phi0 * 3 + ts * 2.0))
        axx, azz = self.axis_y(Y)
        cx = axx + rho * np.cos(phi)
        cz = azz + rho * np.sin(phi)
        Cc = np.stack([cx, Y, cz], -1)
        # orientation: flat (facing up) in the swirl -> vertical tangent plate at the slot
        psi = self.b_psi0[idx] + self.b_psid[idx] * t
        Uf = np.stack([np.cos(psi), np.zeros_like(psi), np.sin(psi)], -1)
        Vf = np.stack([-np.sin(psi), np.zeros_like(psi), np.cos(psi)], -1)
        Us = np.stack([-np.sin(phi), np.zeros_like(phi), np.cos(phi)], -1)
        Vs = np.tile(np.array([0.0, -1.0, 0.0]), (len(idx), 1))
        a = smooth_arr(0.15, 0.95, u)[:, None]
        U = Uf * (1 - a) + Us * a
        V = Vf * (1 - a) + Vs * a
        # tumble mid-flight about a random axis
        ang = self.b_spin[idx] * np.sin(math.pi * u) * 0.55 * (1 - a[:, 0] ** 4)
        U = rodrigues(U, self.b_spinax[idx], ang)
        V = rodrigues(V, self.b_spinax[idx], ang)
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-9
        V = V - (V * U).sum(1, keepdims=True) * U
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        hw = self.b_r[idx] * np.sin((self.b_th1[idx] - self.b_th0[idx]) / 2)
        hh = (self.b_y1[idx] - self.b_y0[idx]) / 2
        Uw, Vh = U * hw[:, None], V * hh[:, None]
        P = np.stack([Cc - Uw + Vh * -1, Cc + Uw - Vh, Cc + Uw + Vh, Cc - Uw + Vh], 1)
        Nn = np.cross(U, V)
        Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-9
        return P, Nn

    # ---------------------------------------------------------------- people on terraces
    def _build_people(self, rng):
        L = []
        kinds = ["citizen"] * 6 + ["cultist"] * 2 + ["crackpot", "philosopher"]
        for k in range(self.NT - 1):
            r = self.r[k]
            rin = self.r[k + 1]
            n = int(2 * math.pi * r / 40)
            th = np.sort(rng.uniform(0, 2 * math.pi, n))
            for i in range(n):
                rr = rng.uniform(rin + 18, r - 18)
                L.append((k, th[i], rr, rng.integers(len(kinds)), rng.integers(0, 10 ** 6),
                          rng.choice([-1, 1]), rng.random(), rng.random(), rng.uniform(0.85, 1.15)))
        A = np.array(L, np.float64)
        self.np = len(A)
        self.p_tier = A[:, 0].astype(int)
        self.p_th = A[:, 1]
        self.p_r = A[:, 2]
        self.p_kind = [kinds[int(i)] for i in A[:, 3]]
        self.p_seed = A[:, 4].astype(int)
        self.p_face = A[:, 5].astype(int)
        self.p_prop = A[:, 6]
        self.p_rnd = A[:, 7]
        self.p_hs = A[:, 8]

    def people_pos(self):
        k = self.p_tier
        x = self.ax[k] + self.p_r * np.cos(self.p_th)
        z = self.az[k] + self.p_r * np.sin(self.p_th)
        y = self.y1[k]
        return np.stack([x, y, z], -1)

    def _build_crowns(self, rng):
        L = []
        for k in range(self.NT - 1):
            r = self.r[k]
            n = int(2 * math.pi * r / 95)
            for i in range(n):
                th = (i + rng.uniform(-0.2, 0.2)) / n * 2 * math.pi
                L.append((k, th, rng.integers(0, 3), rng.uniform(0.85, 1.2), rng.random()))
        A = np.array(L, np.float64)
        self.c_tier = A[:, 0].astype(int)
        self.c_th = A[:, 1]
        self.c_type = A[:, 2].astype(int)
        self.c_s = A[:, 3]
        self.c_rnd = A[:, 4]


# ------------------------------------------------------------------------------------------ math helpers


def smooth_arr(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def rodrigues(v, k, ang):
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return v * c + np.cross(k, v) * s + k * (k * v).sum(1, keepdims=True) * (1 - c)


def collapse_state(tw, t, cl, P_landed):
    """blocks shaken loose by the crack: ballistic debris. returns (mask, P, N)"""
    td = cl["detach_t"]
    det = t > td
    if not det.any():
        return det, P_landed[det], np.zeros((0, 3))
    idx = np.nonzero(det)[0]
    dt = t - td[idx]
    th = tw.b_thm[idx]
    nrm = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], -1)
    tan = np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], -1)
    v = nrm * tw.b_vout[idx, None] + tan * tw.b_vtan[idx, None]
    v[:, 1] += tw.b_vup[idx]
    g = 1500.0
    disp = v * dt[:, None]
    disp[:, 1] -= 0.5 * g * dt ** 2
    P0 = P_landed[idx]
    Cc = P0.mean(1)
    ang = tw.b_spin[idx] * dt * 0.8
    rel = P0 - Cc[:, None, :]
    rel2 = np.stack([rodrigues(rel[:, j], tw.b_spinax[idx], ang) for j in range(4)], 1)
    P = Cc[:, None, :] + disp[:, None, :] + rel2
    U = P[:, 1] - P[:, 0]
    V = P[:, 3] - P[:, 0]
    N = np.cross(U, V)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    return det, P, N


class Orbiters:
    """loose fragments riding the vortex around the tower (never land): depth & speed during the crane"""

    def __init__(self, n=170, seed=17):
        rng = np.random.default_rng(seed)
        self.n = n
        self.rho = rng.uniform(1750, 2750, n)
        self.phi0 = rng.uniform(0, 2 * math.pi, n)
        self.w = rng.uniform(0.35, 0.7, n)
        self.y0 = rng.uniform(-400, 5200, n)
        self.vy = rng.uniform(150, 520, n)
        self.hw = rng.uniform(40, 95, n)
        self.hh = self.hw * rng.uniform(0.6, 1.0, n)
        self.mat = rng.choice(5, n, p=[0.25, 0.2, 0.2, 0.2, 0.15])
        cols = []
        for m in self.mat:
            cs = _MAT_COLS[int(m)]
            cols.append(_hex(cs[int(rng.integers(len(cs)))]))
        self.col = np.array(cols)
        self.seed = rng.random(n)
        self.scr = np.array([_hex(SCREEN_COLS[int(s * 997) % len(SCREEN_COLS)]) for s in self.seed])
        self.ax = rng.normal(size=(n, 3))
        self.ax /= np.linalg.norm(self.ax, axis=1, keepdims=True)
        self.spin = rng.uniform(1.0, 3.5, n) * rng.choice([-1, 1], n)
        self.appear = rng.uniform(0.6, 1.8, n)

    def at(self, t, tw):
        span = 5600.0
        Y = (self.y0 + self.vy * t + 400) % span - 400
        phi = self.phi0 + self.w * t
        axx, azz = tw.axis_y(np.clip(Y, tw.y0[0], tw.top))
        C = np.stack([axx + self.rho * np.cos(phi), Y, azz + self.rho * np.sin(phi)], -1)
        U0 = np.stack([-np.sin(phi), 0 * phi, np.cos(phi)], -1)
        V0 = np.tile([0.0, -1.0, 0.0], (self.n, 1))
        ang = self.spin * t
        U = rodrigues(U0, self.ax, ang)
        V = rodrigues(V0, self.ax, ang)
        Uw, Vh = U * self.hw[:, None], V * self.hh[:, None]
        P = np.stack([C - Uw - Vh, C + Uw - Vh, C + Uw + Vh, C - Uw + Vh], 1)
        N = np.cross(U, V)
        return P, N

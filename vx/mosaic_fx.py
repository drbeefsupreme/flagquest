"""vx.mosaic_fx - tile events (owner: Tessellator). Available as vx.mosaic.fx.

Every event is DETERMINISTIC and a PURE FUNCTION OF t (frames render in parallel, out of order): build the event
once in setup() from the layout, then per frame ask it for per-tile poses and pass them straight to render():

    fall = fx.Fall(lay, t_detach=fx.wave_times(lay, (960, 0), speed=300, t0=2.0), floor=1150)   # setup
    img = mz.render(fc, cart, lay, **fall.pose(fc.t), view=cam)                                    # per frame

Poses are dicts with any of: offset (N,2) chart units, lift (N,) toward the viewer, rot (N,3,3), gain (N,),
present (N,). Tiles with a non-trivial lift/rot are drawn loose (depth-sorted, shadowed) and leave their
socket (the setting bed with the tile's imprint) on the wall. Combine poses with fx.combine(a, b).

Time: every event uses whatever clock you give it (scene-local fc.t or global fc.T) - just be consistent.
Foley: .events(T0=S.start) -> [dict(t=GLOBAL_s, kind, strength, pan, desc)] (impacts aggregated per frame) ->
    vx.foley.export_events(S, name, events)   (or fx.export(S, name, events)).
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .config import W, H, FPS


# ============================================================ helpers
def _ease(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def wave_times(lay_or_xy, origin, speed, t0=0.0, jitter=0.0, seed=0, power=1.0):
    """per-tile start times of a travelling front: t0 + (distance from origin / speed)^power (+ jitter seconds).
    origin: (x, y) point, or ('x', x0) / ('y', y0) for a straight front."""
    xy = lay_or_xy.xy if hasattr(lay_or_xy, "xy") else np.asarray(lay_or_xy, np.float32)
    if isinstance(origin, tuple) and origin and origin[0] in ("x", "y"):
        d = np.abs(xy[:, 0 if origin[0] == "x" else 1] - origin[1])
    else:
        d = np.hypot(xy[:, 0] - origin[0], xy[:, 1] - origin[1])
    t = (d / max(speed, 1e-6)) ** power if power != 1.0 else d / max(speed, 1e-6)
    if jitter:
        t = t + np.random.default_rng(seed).normal(0, jitter, len(xy))
    return (t0 + t).astype(np.float32)


def combine(*poses):
    """merge pose dicts: offsets add, lifts add, rotations compose (later first), gains multiply, present ANDs."""
    out = {}
    for p in poses:
        if not p:
            continue
        for k, v in p.items():
            if v is None:
                continue
            if k not in out:
                out[k] = np.array(v, copy=True)
            elif k in ("offset", "lift", "emit"):
                out[k] = out[k] + v
            elif k == "rot":
                out[k] = np.einsum("nij,njk->nik", v, out[k]).astype(np.float32)
            elif k == "gain":
                out[k] = out[k] * v
            elif k == "present":
                out[k] = np.minimum(out[k], v)
            else:
                out[k] = v
    return out


def aggregate(times, strength, x, kind="tiles", desc="", T0=0.0, fps=FPS, min_strength=0.02, per_frame=True):
    """many small impacts -> foley events: one per frame bin with strength = sqrt(sum of energies) (0..1),
    pan from the energy-weighted x (design units)."""
    times = np.asarray(times, np.float64)
    ok = np.isfinite(times)
    times, strength, x = times[ok], np.asarray(strength, np.float64)[ok], np.asarray(x, np.float64)[ok]
    if len(times) == 0:
        return []
    b = np.floor(times * fps).astype(np.int64) if per_frame else np.arange(len(times))
    ub, inv = np.unique(b, return_inverse=True)
    e = np.bincount(inv, strength ** 2)
    xe = np.bincount(inv, strength ** 2 * x) / np.maximum(e, 1e-12)
    cnt = np.bincount(inv)
    tt = np.bincount(inv, times) / cnt
    out = []
    for i in range(len(ub)):
        s = float(min(1.0, math.sqrt(e[i])))
        if s < min_strength:
            continue
        out.append(dict(t=float(T0 + tt[i]), kind=kind, strength=s, pan=float(np.clip((xe[i] - W / 2) / (W / 2), -1, 1)),
                        desc=f"{desc} ({int(cnt[i])} tesserae)" if desc else f"{int(cnt[i])} tesserae"))
    return out


def export(S, name, events):
    from .foley import export_events
    export_events(S, name, events)


# ============================================================ fall
class Fall:
    """Tesserae falling off the wall (Babel): each tile detaches at t_detach[i], pops off with a small random kick,
    tumbles in 3D under gravity, hits the floor, bounces (restitution `bounce`), rattles and comes to rest on a
    growing heap. Pure function of t.
      lay: Layout (or an object with .xy/.size); t_detach: (N,) times (np.inf = never falls)
      floor: world y of the floor (design units; the wall's chart y); g: gravity (design units/s^2)
      v0: optional (N,3) initial velocities (x, y, z toward viewer) - rigid chunks toppling etc.
      spin: optional (N,3) angular velocity vectors (rad/s) for the flight; turns: extra tumbling turns
      heap: tiles pile up where they land (bins along x); depth: range of z (toward the viewer) they scatter to.
    .pose(t) -> dict(offset, lift, rot) for render(); .state(t) -> (pos (N,3), rot (N,3,3), phase (N,))
    .rest() -> (pos (N,3), rot (N,3,3)) final poses (feed them to Fly for the Unbabeling)
    .events(T0) -> foley impacts;  .energy(n_frames, t_start) -> per-frame rattle energy."""

    def __init__(self, lay, t_detach, floor=None, g=2400.0, bounce=0.34, friction=0.55, v0=None, spin=None,
                 turns=2.0, heap=True, depth=(30.0, 320.0), kick=70.0, spread=60.0, bounces=4, seed=0,
                 rest_tilt=(0.25, 1.25), heap_rate=1.0):
        rng = np.random.default_rng(seed)
        self.xy = np.asarray(lay.xy if hasattr(lay, "xy") else lay, np.float64).reshape(-1, 2)
        n = len(self.xy)
        self.n = n
        self.size = np.broadcast_to(np.asarray(getattr(lay, "size", 12.0), np.float64), (n,)).copy()
        self.t0 = np.broadcast_to(np.asarray(t_detach, np.float64), (n,)).copy()
        self.g = float(g)
        self.floor = float(floor if floor is not None else getattr(lay, "extent", (W, H))[1] + 40)
        if v0 is None:
            v = np.zeros((n, 3))
            v[:, 0] = rng.normal(0, spread, n)
            v[:, 1] = -rng.uniform(0, kick, n)
            v[:, 2] = rng.uniform(depth[0], depth[1], n)            # outward; scaled to the fall time below
        else:
            v = np.asarray(v0, np.float64).reshape(n, 3).copy()
        # fall time without heap (to scale the outward speed so tiles land within `depth`)
        h = np.maximum(self.floor - self.xy[:, 1], 1.0)
        tf = (-v[:, 1] + np.sqrt(v[:, 1] ** 2 + 2 * self.g * h)) / self.g
        if v0 is None:
            v[:, 2] = v[:, 2] / np.maximum(tf, 0.15)
        self.v = v
        # landing position (x) without heap, then the heap height at landing order
        xl = self.xy[:, 0] + v[:, 0] * tf
        drift = (v[:, 0] * friction) * (2 * bounce * np.abs(v[:, 1] + self.g * tf) / self.g) / max(1 - bounce, 0.2)
        xf = xl + drift
        land_t = self.t0 + tf
        yrest = np.full(n, self.floor)
        if heap:
            order = np.argsort(land_t)
            bw = 10.0
            bins = np.floor(xf / bw).astype(np.int64)
            b0 = bins.min() if n else 0
            hh = np.zeros(int(bins.max() - b0 + 3) if n else 1)
            area = self.size ** 2 / (bw * max(depth[1] - depth[0], 40.0)) * 3.2 * heap_rate
            fin = np.isfinite(land_t)
            for i in order:
                if not fin[i]:
                    continue
                k = bins[i] - b0 + 1
                y = max(hh[k - 1] * 0.5, hh[k], hh[k + 1] * 0.5)
                yrest[i] = self.floor - y - 0.25 * self.size[i]
                hh[k] = y + area[i] * self.size[i] * 0.35
                hh[k - 1] = max(hh[k - 1], hh[k] * 0.6)
                hh[k + 1] = max(hh[k + 1], hh[k] * 0.6)
        self.yrest = yrest
        h = np.maximum(yrest - self.xy[:, 1], 0.5)
        self.tf = (-v[:, 1] + np.sqrt(v[:, 1] ** 2 + 2 * self.g * h)) / self.g
        vimp = v[:, 1] + self.g * self.tf
        self.vimp = vimp
        # bounce schedule (N, B)
        B = int(bounces)
        k = np.arange(1, B + 1)
        vup = vimp[:, None] * bounce ** k[None, :]
        dur = 2 * vup / self.g
        self.bstart = self.tf[:, None] + np.concatenate([np.zeros((n, 1)), np.cumsum(dur, 1)[:, :-1]], 1)
        self.bdur = dur
        self.vup = vup
        self.hfac = np.broadcast_to(friction ** k, (n, B)).copy()   # horizontal speed factor per bounce
        self.trest = self.tf + dur.sum(1)
        # positions at the start of each bounce
        x1 = self.xy[:, 0] + v[:, 0] * self.tf
        z1 = v[:, 2] * self.tf
        bx = [x1]
        bz = [z1]
        for j in range(B):
            bx.append(bx[-1] + v[:, 0] * self.hfac[:, j] * dur[:, j])
            bz.append(bz[-1] + v[:, 2] * self.hfac[:, j] * dur[:, j] * 0.5)
        self.bx = np.stack(bx, 1)
        self.bz = np.stack(bz, 1)
        # rest orientation: lying on the heap, tilted toward the viewer, random spin in its plane
        tilt = rng.uniform(rest_tilt[0], rest_tilt[1], n)            # angle between tile normal and the viewer
        yaw = rng.uniform(0, 2 * np.pi, n)
        face_down = rng.random(n) < 0.3
        # tilt about the tile's x axis: top edge falls back so the tile faces up-and-out
        R_rest = Rotation.from_rotvec(np.stack([np.pi / 2 - tilt, np.zeros(n), np.zeros(n)], 1)) * \
            Rotation.from_rotvec(np.stack([np.zeros(n), np.zeros(n), yaw], 1))
        fd = Rotation.from_rotvec(np.stack([np.where(face_down, np.pi, 0.0), np.zeros(n), np.zeros(n)], 1))
        R_rest = R_rest * fd
        rv = R_rest.as_rotvec()
        ang = np.linalg.norm(rv, axis=1)
        axis = rv / np.maximum(ang, 1e-9)[:, None]
        if spin is not None:
            sp = np.asarray(spin, np.float64).reshape(n, 3)
            axis = sp / np.maximum(np.linalg.norm(sp, axis=1, keepdims=True), 1e-9)
            extra = np.linalg.norm(sp, axis=1) * self.tf
            # keep the rest orientation reachable: rotate about the spin axis then finish at R_rest
            self.spin_axis, self.spin_total = axis, extra
            self.R_rest = R_rest
            self.mode = "spin"
        else:
            k_turns = np.round(rng.uniform(0, turns, n))
            self.axis, self.total = axis, ang + 2 * np.pi * k_turns
            self.mode = "axis"
        self.R_rest = R_rest

    # ---------------------------------------------------------------- per frame
    def state(self, t):
        n = self.n
        tau = t - self.t0
        pos = np.zeros((n, 3))
        pos[:, 0], pos[:, 1] = self.xy[:, 0], self.xy[:, 1]
        phase = np.zeros(n, np.int8)                 # 0 wall, 1 flight, 2 bouncing, 3 rest
        on = tau > 0
        fl = on & (tau < self.tf)
        v = self.v
        if fl.any():
            tt = tau[fl]
            pos[fl, 0] = self.xy[fl, 0] + v[fl, 0] * tt
            pos[fl, 1] = self.xy[fl, 1] + v[fl, 1] * tt + 0.5 * self.g * tt * tt
            pos[fl, 2] = v[fl, 2] * tt
            phase[fl] = 1
        bo = on & (tau >= self.tf) & (tau < self.trest)
        if bo.any():
            ii = np.nonzero(bo)[0]
            tb = tau[ii][:, None]
            j = np.clip((tb >= self.bstart[ii]).sum(1) - 1, 0, self.bstart.shape[1] - 1)
            s0 = self.bstart[ii, j]
            u = tau[ii] - s0
            vu = self.vup[ii, j]
            pos[ii, 1] = self.yrest[ii] - vu * u + 0.5 * self.g * u * u
            pos[ii, 1] = np.minimum(pos[ii, 1], self.yrest[ii])
            hf = self.hfac[ii, j]
            pos[ii, 0] = self.bx[ii, j] + v[ii, 0] * hf * u
            pos[ii, 2] = self.bz[ii, j] + v[ii, 2] * hf * u * 0.5
            phase[ii] = 2
        re = on & (tau >= self.trest)
        if re.any():
            pos[re, 0] = self.bx[re, -1]
            pos[re, 1] = self.yrest[re]
            pos[re, 2] = self.bz[re, -1]
            phase[re] = 3
        # rotation
        rot = np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy()
        mv = on
        if mv.any():
            ii = np.nonzero(mv)[0]
            u = np.clip(tau[ii] / np.maximum(self.trest[ii], 1e-3), 0, 1)
            e = 1 - (1 - u) ** 2.2
            if self.mode == "axis":
                rv = self.axis[ii] * (self.total[ii] * e)[:, None]
                rot[ii] = Rotation.from_rotvec(rv).as_matrix().astype(np.float32)
            else:
                spin = Rotation.from_rotvec(self.spin_axis[ii] * (self.spin_total[ii] * np.minimum(
                    tau[ii] / np.maximum(self.tf[ii], 1e-3), 1.0))[:, None])
                w = _ease(np.clip((tau[ii] - self.tf[ii]) / np.maximum(self.trest[ii] - self.tf[ii], 1e-3), 0, 1))
                # blend from the spun orientation to the rest orientation over the bounces
                a = spin.as_rotvec()
                b = self.R_rest[ii].as_rotvec()
                rot[ii] = Rotation.from_rotvec(a * (1 - w)[:, None] + b * w[:, None]).as_matrix().astype(np.float32)
        return pos, rot, phase

    def pose(self, t):
        """dict(offset, lift, rot) for render(fc, cart, lay, **fall.pose(t))."""
        pos, rot, phase = self.state(t)
        return dict(offset=(pos[:, :2] - self.xy).astype(np.float32), lift=pos[:, 2].astype(np.float32), rot=rot)

    def rest(self):
        big = float(np.max(np.where(np.isfinite(self.t0 + self.trest), self.t0 + self.trest, 0))) + 1.0
        pos, rot, _ = self.state(big)
        return pos, rot

    def fallen(self, t):
        return (t - self.t0) > 0

    def events(self, T0=0.0, desc="tesserae hit the floor", kind="tiles"):
        """floor impacts (first hit + bounces), aggregated per frame. T0 = global time of this event's t=0."""
        tt = [self.t0 + self.tf]
        ss = [np.abs(self.vimp) / 2500.0 * np.sqrt(self.size / 12.0)]
        xx = [self.bx[:, 0]]
        for j in range(self.bdur.shape[1]):
            tt.append(self.t0 + self.bstart[:, j] + self.bdur[:, j])
            ss.append(self.vup[:, j] / 2500.0 * np.sqrt(self.size / 12.0))
            xx.append(self.bx[:, j + 1])
        times = np.concatenate(tt)
        st = np.concatenate(ss) * 0.12
        x = np.concatenate(xx)
        ok = np.isfinite(times)
        return aggregate(times[ok], st[ok], x[ok], kind=kind, desc=desc, T0=T0)

    def detach_events(self, T0=0.0, desc="tesserae break loose"):
        ok = np.isfinite(self.t0)
        return aggregate(self.t0[ok], np.full(ok.sum(), 0.05), self.xy[ok, 0], kind="crack", desc=desc, T0=T0)

    def energy(self, n_frames, t_start=0.0, fps=FPS):
        """per-frame rattle energy (number of floor impacts per frame, normalised) for vx.foley.export_track."""
        ev = self.events(0.0)
        e = np.zeros(n_frames, np.float32)
        for d in ev:
            f = int(round((d["t"] - t_start) * fps))
            if 0 <= f < n_frames:
                e[f] += d["strength"]
        return np.clip(e, 0, 1)


# ============================================================ fly (re-setting)
class Fly:
    """Tesserae flying from source poses (e.g. Fall.rest(): the heap on the floor) back to their slots on the wall
    and re-setting themselves with a click and a glint (the Unbabeling). Pure function of t.
      slots: (N,2) chart positions (lay.xy); src_pos (N,3) world (x, y, z); src_rot (N,3,3) or None
      t0 (N,) start times, dur (N,) or float flight durations; arc: lift of the path toward the viewer (design
      units); rise: how high above the straight line the path swings (fraction of distance); turns: extra spins.
    .pose(t) -> dict(offset, lift, rot, gain);  .events(T0) -> arrival clicks."""

    def __init__(self, slots, src_pos, t0, dur=1.2, src_rot=None, arc=160.0, rise=0.25, turns=1.0, seed=0,
                 settle=0.18, glint=1.8):
        rng = np.random.default_rng(seed)
        self.dst = np.asarray(slots, np.float64).reshape(-1, 2)
        n = len(self.dst)
        self.n = n
        self.src = np.asarray(src_pos, np.float64).reshape(n, 3)
        self.t0 = np.broadcast_to(np.asarray(t0, np.float64), (n,)).copy()
        self.dur = np.broadcast_to(np.asarray(dur, np.float64), (n,)).copy()
        self.settle, self.glint = settle, glint
        d = np.hypot(self.dst[:, 0] - self.src[:, 0], self.dst[:, 1] - self.src[:, 1])
        self.c1 = self.src + np.stack([rng.normal(0, 0.15, n) * d, -rise * d, arc * (0.6 + 0.8 * rng.random(n))], 1)
        self.c2 = np.concatenate([self.dst, np.zeros((n, 1))], 1) + np.stack(
            [rng.normal(0, 0.05, n) * d, np.zeros(n), arc * (0.5 + 0.5 * rng.random(n))], 1)
        R0 = Rotation.from_matrix(src_rot) if src_rot is not None else Rotation.identity(n)
        rv = R0.as_rotvec()
        ang = np.linalg.norm(rv, axis=1)
        self.axis = rv / np.maximum(ang, 1e-9)[:, None]
        no = ang < 1e-6
        if no.any():
            a = rng.normal(0, 1, (int(no.sum()), 3))
            self.axis[no] = a / np.linalg.norm(a, axis=1, keepdims=True)
        self.ang0 = ang + 2 * np.pi * np.round(rng.uniform(0, turns, n))

    def pose(self, t):
        n = self.n
        u = np.clip((t - self.t0) / np.maximum(self.dur, 1e-3), 0, 1)
        e = _ease(u)
        a = 1 - e
        b = e
        P = (a ** 3)[:, None] * self.src + (3 * a * a * b)[:, None] * self.c1 + (3 * a * b * b)[:, None] * self.c2 + \
            (b ** 3)[:, None] * np.concatenate([self.dst, np.zeros((n, 1))], 1)
        # click: tiny overshoot into the wall then flush
        ta = t - (self.t0 + self.dur)
        arrived = ta >= 0
        z = P[:, 2].copy()
        if arrived.any():
            s = np.clip(ta[arrived] / self.settle, 0, 1)
            z[arrived] = -0.9 * np.sin(np.pi * s) * (s < 1)
        rv = self.axis * (self.ang0 * (1 - _ease(np.clip(u * 1.15, 0, 1))))[:, None]
        rot = Rotation.from_rotvec(rv).as_matrix().astype(np.float32)
        gain = np.ones(n, np.float32)
        if arrived.any():
            gain[arrived] = 1.0 + self.glint * np.exp(-ta[arrived] / 0.12)
        done = ta > self.settle
        rot[done] = np.eye(3, dtype=np.float32)
        z[done] = 0.0
        return dict(offset=(P[:, :2] - self.dst).astype(np.float32), lift=z.astype(np.float32), rot=rot, gain=gain)

    def events(self, T0=0.0, desc="tessera re-sets itself", kind="click"):
        t = self.t0 + self.dur
        return aggregate(t, np.full(self.n, 0.06), self.dst[:, 0], kind=kind, desc=desc, T0=T0)


# ============================================================ flip / pop / lift
def flip(t, t0, dur=0.3, axis="x", hop=0.18, size=12.0):
    """split-flap re-colour: each tile turns over about its local x (or y) axis between t0 and t0+dur; halfway
    it shows its other face. Returns dict(rot, lift, new) - new (N,) bool: show the NEW colour.
        f = fx.flip(t, t0); render(..., rgb=np.where(f['new'][:, None], B, A), rot=f['rot'], lift=f['lift'])"""
    t0 = np.asarray(t0, np.float64)
    u = np.clip((t - t0) / max(dur, 1e-3), 0, 1)
    th = np.pi * _ease(u)
    new = th >= np.pi / 2
    shown = np.where(new, th - np.pi, th)
    from .mosaic import rot_x, rot_y
    rot = rot_x(shown) if axis == "x" else rot_y(shown)
    lift = (hop * np.asarray(size, np.float64) * np.sin(th)).astype(np.float32)
    return dict(rot=rot, lift=lift, new=new)


def pop(t, t0, dur=0.5, height=14.0, wobble=0.25, seed=0, n=None):
    """tiles lift out of the wall and settle back (a shiver, a seal slam). Returns dict(lift, rot)."""
    t0 = np.asarray(t0, np.float64)
    n = len(t0) if t0.ndim else (n or 1)
    u = np.clip((t - t0) / max(dur, 1e-3), 0, 1)
    z = height * np.sin(np.pi * u) * (1 - u) * 1.6
    rng = np.random.default_rng(seed)
    ax = rng.normal(0, 1, (n, 3))
    ax[:, 2] *= 0.2
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    ang = wobble * np.sin(np.pi * u * 2) * (1 - u)
    from .mosaic import axis_angle
    return dict(lift=z.astype(np.float32), rot=axis_angle(ax * ang[:, None]))


def tremble(t, n, amp=1.2, freq=17.0, seed=0, t0=-1e9, t1=1e9):
    """tiles loosening: small in-plane jitter (still bedded). Returns dict(offset)."""
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, (n, 2))
    fq = freq * rng.uniform(0.7, 1.3, (n, 1))
    on = float(t0 <= t <= t1)
    return dict(offset=(on * amp * np.sin(t * fq * 2 * np.pi + ph)).astype(np.float32))


# ============================================================ schism (split)
def schism(child, t, t0, dur=0.6, spread=0.35, twist=0.12, seed=0, parent_xy=None):
    """animate a split layout (Layout.split): children drift apart from their parent's centre (the joints open)
    and turn a little. t0: per-PARENT (len = max(parent)+1) or per-child start times. Returns dict(offset, rot)."""
    par = child.parent
    n = len(child)
    t0 = np.asarray(t0, np.float64)
    tt = t0[par] if t0.ndim and len(t0) != n else np.broadcast_to(t0, (n,))
    u = _ease(np.clip((t - tt) / max(dur, 1e-3), 0, 1))
    if parent_xy is None:
        # centre of each parent = mean of its children
        m = par.max() + 1
        cnt = np.bincount(par, minlength=m).astype(np.float64)
        cx = np.bincount(par, child.xy[:, 0], minlength=m) / np.maximum(cnt, 1)
        cy = np.bincount(par, child.xy[:, 1], minlength=m) / np.maximum(cnt, 1)
        pc = np.stack([cx, cy], 1)[par]
    else:
        pc = np.asarray(parent_xy)[par]
    off = (child.xy - pc) * (spread * u)[:, None]
    rng = np.random.default_rng(seed)
    from .mosaic import rot_z
    return dict(offset=off.astype(np.float32), rot=rot_z(rng.normal(0, twist, n) * u))


# ============================================================ crack
class Crack:
    """A crack running through the setting bed: a jagged path (random walk with persistence + branches) revealed
    from its start at `speed` units/s from t0; tiles on either side are pushed apart (`open` units at the crack,
    fading over `band` units) and tilted, so the crack opens along the joints.
      .pose(t) -> dict(offset, rot);  .near(t, width) -> (N,) 0..1 weight of tiles along the revealed crack (glow
      the joints with render(joint_emit=w[:, None] * colour));  .paths(t) -> revealed polylines (chart units);
      .events(T0) -> crack sound events (the front advancing)."""

    def __init__(self, lay, start, direction=0.0, length=900.0, t0=0.0, speed=900.0, open=3.5, band=40.0,
                 branch=0.18, wander=0.35, seed=0, tilt=0.08, max_branches=6):
        rng = np.random.default_rng(seed)
        self.xy = np.asarray(lay.xy if hasattr(lay, "xy") else lay, np.float64).reshape(-1, 2)
        self.t0, self.speed, self.open, self.band, self.tilt = t0, speed, open, band, tilt
        self.paths_ = []                  # list of (pts (M,2), s (M,) arc length from the crack origin)
        todo = [(np.asarray(start, np.float64), direction, length, 0.0)]
        nb = 0
        while todo:
            p, d, L, s0 = todo.pop(0)
            pts = [p.copy()]
            ss = [s0]
            step = 10.0
            k = 0
            while ss[-1] - s0 < L:
                d += rng.normal(0, wander)
                d = 0.85 * d + 0.15 * direction if len(self.paths_) == 0 else d
                p = p + step * np.array([math.cos(d), math.sin(d)])
                pts.append(p.copy())
                ss.append(ss[-1] + step)
                k += 1
                if nb < max_branches and rng.random() < branch * step / 100.0:
                    nb += 1
                    todo.append((p.copy(), d + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1),
                                 (L - (ss[-1] - s0)) * rng.uniform(0.3, 0.7), ss[-1]))
            self.paths_.append((np.array(pts), np.array(ss)))
        # per tile: nearest crack point, side, arc position
        best = np.full(len(self.xy), np.inf)
        self.side = np.zeros((len(self.xy), 2))
        self.s_at = np.full(len(self.xy), np.inf)
        for pts, ss in self.paths_:
            for j in range(len(pts) - 1):
                a, b = pts[j], pts[j + 1]
                ab = b - a
                L2 = ab @ ab
                u = np.clip(((self.xy - a) @ ab) / L2, 0, 1)
                q = a + u[:, None] * ab
                dv = self.xy - q
                d = np.hypot(dv[:, 0], dv[:, 1])
                m = d < best
                if m.any():
                    best[m] = d[m]
                    nrm = np.array([-ab[1], ab[0]]) / math.sqrt(L2)
                    sgn = np.sign(dv[m] @ nrm)
                    sgn[sgn == 0] = 1
                    self.side[m] = sgn[:, None] * nrm[None, :]
                    self.s_at[m] = ss[j] + u[m] * math.sqrt(L2)
        self.dist = best
        self.rng_ax = rng.normal(0, 1, (len(self.xy), 3))
        self.rng_ax[:, 2] = 0
        self.rng_ax /= np.maximum(np.linalg.norm(self.rng_ax, axis=1, keepdims=True), 1e-9)

    def reveal(self, t):
        return max(0.0, (t - self.t0) * self.speed)

    def pose(self, t):
        r = self.reveal(t)
        n = len(self.xy)
        tip = np.clip((r - self.s_at) / 60.0, 0, 1)             # opens behind the advancing tip
        fall = np.exp(-(self.dist / max(self.band, 1e-3)) ** 2)
        k = tip * fall
        off = self.side * (self.open * 0.5 * k)[:, None]
        from .mosaic import axis_angle
        rot = axis_angle(self.rng_ax * (self.tilt * k)[:, None])
        return dict(offset=off.astype(np.float32), rot=rot)

    def near(self, t, width=18.0):
        r = self.reveal(t)
        return (np.clip((r - self.s_at) / 40.0, 0, 1) * np.exp(-(self.dist / width) ** 2)).astype(np.float32)

    def paths(self, t):
        r = self.reveal(t)
        out = []
        for pts, ss in self.paths_:
            m = ss <= r
            if m.sum() >= 2:
                out.append(pts[m])
        return out

    def events(self, T0=0.0, desc="the setting bed cracks"):
        tt, st, xx = [], [], []
        for pts, ss in self.paths_:
            for j in range(0, len(pts), 3):
                tt.append(self.t0 + ss[j] / self.speed)
                st.append(0.08)
                xx.append(pts[j, 0])
        return aggregate(np.array(tt), np.array(st), np.array(xx), kind="crack", desc=desc, T0=T0)


def draw_cracks(ctx, paths, width=1.6, color=(0.05, 0.04, 0.035), alpha=0.9):
    """draw revealed crack polylines with cairo (chart/design units)."""
    ctx.save()
    ctx.set_source_rgba(color[0], color[1], color[2], alpha)
    ctx.set_line_width(width)
    for p in paths:
        ctx.move_to(float(p[0, 0]), float(p[0, 1]))
        for q in p[1:]:
            ctx.line_to(float(q[0]), float(q[1]))
    ctx.stroke()
    ctx.restore()

"""s07b — FLAGISTAN (154.5-173.0). Owner: Scene07b.

Synced to the sung hymn (84 BPM; phrases 154.50 "Flags be" / 157.36 "Flags are" / 160.21 "Flags will" /
163.07 "Flags." sustained to 165.93):

  154.5  hand-off from s07a's high aerial; keep rising (true ray-traced planet, s07b_planet): the flag-carpeted
         plain becomes a map of recursive Voronoi borders (s05b's balkanization) with a flag in every country,
         the planet's curve and the sunrise on the limb, then the whole globe, flags glittering on the night side.
  159.6  the globe becomes the Poincare disk: FLAGISTAN, an exact {5,4} hyperbolic tiling (s07b_hyper, per-pixel
         reflection-group folding) crystallises outward from Lutie's hill, a plain yellow flag at every vertex.
  160.2  Mobius glide through the hyperbolic plane; each phrase a ring of light through the tiles;
  163.07 "Flags." = the biggest bloom; the disk spins, heaven turns gold.
  165.9  the camera tilts, every flag lifts off like an ember into the golden sky.
  166.3  the Gateless Gate: a torii of pure light standing on the sea of clouds; the 10,000-foot Flag (real cloth
         sim) rises through it, god rays, ember-flags orbiting. 170.9 N10: bloom to flag_glow, flat by 172.6.
"""
import math

import cairo
import numpy as np

from vx import *
from vx import foley
import cv2
from vx.flag import Flag, FlagTrack, draw_flag_field

from scenes import s07b_hyper as HY
from scenes import s07b_planet as PL
from scenes import s07b_sky as SK

try:                                    # s07a's pure world renderer for the seamless hand-off
    from scenes.s07a_plant import render_at as _S07A
except Exception:                       # pragma: no cover - fall back to the planet renderer alone
    _S07A = None
H07A = (155.3, 156.1)                   # s07a world -> planet dissolve window
# ------------------------------------------------------------------ timing (global seconds)
BEAT = 60.0 / 84.0
PH = [154.5 + 4 * BEAT * k for k in range(4)]      # 154.50, 157.357, 160.214, 163.071
HYMN_END = 154.5 + 16 * BEAT                        # 165.929
LIFT = 165.9
GATE = 166.3
N10 = 170.9
END = 173.0
T_DISK0, T_DISK1 = 159.62, 160.30                   # globe -> Poincare disk
T_WHITE = 172.6                                     # full-frame flag_glow from here

F_PX = 960.0 / math.tan(math.radians(25.0))        # focal length (design px), HFOV 50 like s07a
TH = math.tan(math.radians(25.0))
HC = F_PX / 534.0                                   # heaven camera height: disk radius 534 px top-down

POST = dict(bloom=0.5, bloom_thresh=0.7, grain=0.03, vignette=0.22)

PLATES = ['#2A2F5A', '#3A3470', '#2B4A72', '#2F5E6C', '#473A6E', '#353E66', '#24406A', '#4B3F68']
PCOLS = ['#14132A', '#B8707A', '#9A5A70', '#171A36', '#FFC41A', '#0B0E1F', '#E89A86', '#5A6FB0', '#F4BCB2', '#86A2F2']
TILES = ['#1E2C5C', '#253E6E', '#1F5566', '#2D3A7E', '#3A2F6E', '#1D4A66', '#2C5A86', '#44306A', '#28467A']
SEAM = '#DCEBFF'
# heaven: zenith, mid, horizon, sun, cloud lit, cloud shadow, cloud rim, haze
HCOLS = ['#3A2758', '#B86C66', '#F0A45E', '#FFE2B0', '#DE9F7E', '#4A2E5A', '#FFE3C0', '#B87866']


def _rgb(hexes):
    return np.array([hex2rgb(h) for h in hexes], np.float64)


# ------------------------------------------------------------------ small maths
def _pchip(T, keys, log=False):
    ts = np.array([k[0] for k in keys], float)
    vs = np.array([k[1] for k in keys], float)
    if log:
        vs = np.log(vs)
    from scipy.interpolate import PchipInterpolator
    v = float(PchipInterpolator(ts, vs, extrapolate=False)(min(max(T, ts[0]), ts[-1])))
    return math.exp(v) if log else v


def cam_vec(pos, yaw, pitch_down, hfov=50.0):
    """camera array [pos, right, up, fwd, tan(hfov/2)]; yaw 0 = north (+y), pitch_down > 0 looks down"""
    cp, sp = math.cos(pitch_down), math.sin(pitch_down)
    f = np.array([math.sin(yaw) * cp, math.cos(yaw) * cp, -sp])
    r = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    u = np.cross(r, f)
    return np.array([*pos, *r, *u, *f, math.tan(math.radians(hfov) / 2)], np.float64)


def project(cam, pts):
    """3D (N,3) -> design px (N,2), depth (N,)"""
    P = cam[0:3]
    r, u, f, th = cam[3:6], cam[6:9], cam[9:12], cam[12]
    d = np.atleast_2d(pts) - P
    z = d @ f
    zz = np.maximum(z, 1e-9)
    x = (d @ r) / zz / th
    y = (d @ u) / zz / (th * H / W)
    return np.stack([(x + 1) * 0.5 * W, (1 - y) * 0.5 * H], -1), z


def plane_homography(cam, z0, w, h):
    """render-pixel (X+.5, Y+.5, 1) -> homogeneous point on the plane z = z0 (x, y, 1)"""
    P = cam[0:3].copy()
    P[2] -= z0
    r, u, f, th = cam[3:6], cam[6:9], cam[9:12], cam[12]
    a = h / w
    cX = (2 * th / w) * r
    cY = -(2 * th * a / h) * u
    c0 = f - th * r + th * a * u
    A = np.stack([cX, cY, c0], 1)          # d = A @ (X, Y, 1)
    Hm = np.stack([P[2] * A[0] - P[0] * A[2], P[2] * A[1] - P[1] * A[2], -A[2]], 0)
    return Hm


# ------------------------------------------------------------------ cameras
def planet_cam(T):
    """rise from s07a's aerial (s07a_world.camera_at: 140 m, 53.5 deg down, looking north; matched exactly until
    156.3) to the whole globe (19000 km, straight down)"""
    alt = _pchip(T, [(154.5, 0.140), (155.3, 0.2086), (155.9, 0.2814), (156.3, 0.3435), (156.8, 2.6),
                     (157.36, 38.0), (158.0, 380.0), (158.6, 2300.0), (159.0, 6800.0), (159.35, 14000.0),
                     (159.62, 19000.0), (161.0, 19000.0)], log=True)
    pitch = _pchip(T, [(154.5, 53.53), (155.3, 59.72), (155.9, 63.67), (156.3, 66.0), (156.8, 64.0),
                       (157.36, 58.0), (158.0, 52.0), (158.6, 50.0), (158.9, 57.0), (159.2, 74.0), (159.5, 88.5),
                       (159.62, 90.0), (161.0, 90.0)])
    y = -0.06 * (1 - seg(T, 156.3, 158.5, ease_in_out))
    hfov = 48.0 + 2.0 * seg(T, 156.3, 157.2)
    return cam_vec((0.0, y, alt), 0.0, math.radians(min(pitch, 89.999)), hfov), alt


def heaven_cam(T):
    """camera in heaven units (Flagistan disk radius 1 at z = 0)"""
    keys_y = [(159.0, 0.0), (164.0, 0.0), (165.4, -1.9), (165.9, -2.6), (166.6, -2.8), (167.5, -1.8),
              (169.5, 7.5), (170.2, 17.0), (171.2, 26.5), (172.6, 40.0), (173.0, 44.0)]
    keys_z = [(159.0, HC), (164.0, HC), (165.4, 2.1), (165.9, 1.45), (166.6, 1.6), (167.5, 1.9), (169.5, 3.6),
              (170.2, 4.6), (171.2, 7.0), (172.6, 11.0), (173.0, 12.0)]
    keys_p = [(159.0, 90.0), (164.0, 90.0), (165.4, 42.0), (165.9, 21.0), (166.2, 5.0), (166.6, -2.0),
              (167.5, -4.0), (169.5, -7.0), (170.2, -9.0), (171.2, -11.0), (172.6, -12.0), (173.0, -12.0)]
    y = _pchip(T, keys_y)
    z = _pchip(T, keys_z)
    p = _pchip(T, keys_p)
    return cam_vec((0.0, y, z), 0.0, math.radians(min(p, 89.999)))


# ------------------------------------------------------------------ Flagistan motion (world -> screen Mobius)
def flag_mobius(T):
    """S(T): world disk -> screen disk. Glide (hyperbolic translation along a slowly turning geodesic) from
    'Flags will'; from 'Flags.' an elliptic spin about the centre with a decelerating ease."""
    t = max(0.0, T - PH[2])
    # glide distance with an accent hop on each beat-pair of the phrase
    d = 0.22 * t + 0.35 * sum(ease_out_back(clamp((T - PH[2] - k * 2 * BEAT) / 0.9)) for k in range(2))
    d += 0.3 * ease_out(clamp((T - PH[3]) / 1.8), 3)
    phi = 0.35 + 0.08 * t
    S = HY.mob_translate(-d, phi)
    spin = 0.05 * t + (2 * math.pi / 5) * ease_in_out(clamp((T - PH[3] + 0.05) / 2.6), 3)
    S = HY.mob_compose(HY.mob_rotate(spin), S)
    return S


def _mob_apply_arr(m, z):
    a, b = m
    return (a * z + b) / (np.conj(b) * z + np.conj(a))


# ------------------------------------------------------------------ setup
def _stars(S, seed=7):
    rng = np.random.default_rng(seed)
    n = 900
    return dict(x=rng.uniform(0, W, n), y=rng.uniform(0, H, n), m=rng.uniform(0.15, 1.0, n) ** 2.2,
                tw=rng.uniform(0, 6.28, n))


def _clouds(seed=11):
    """heaven's sea of clouds as puffs (x, y, z, r, tone): cauliflower cumulus heaps (clusters of small puffs
    on domes) round the Flagistan disk + rows of banks receding to the horizon. Tops just below z = -0.4."""
    rng = np.random.default_rng(seed)
    P = []
    # near field: heaps on a jittered grid, a moat round the disk
    g = 0.9
    for yy in np.arange(-5.0, 16.0, g):
        for xx in np.arange(-10.0, 10.0, g):
            hx = xx + rng.uniform(-0.4, 0.4) * g
            hy = yy + rng.uniform(-0.4, 0.4) * g
            if math.hypot(hx, hy) < 1.35:
                continue
            R = g * rng.uniform(0.55, 0.95) * (1 + 0.025 * max(0.0, hy))
            base = -0.72 + 0.15 * math.sin(hx * 0.5 + 1.3) * math.cos(hy * 0.33) + rng.uniform(-0.06, 0.06)
            tall = rng.uniform(0.35, 0.8) * (1.8 if rng.uniform() < 0.08 else 1.0)
            n = int(10 + 14 * R / g)
            for _ in range(n):
                a = rng.uniform(0, 2 * math.pi)
                q = math.sqrt(rng.uniform(0, 1))
                px, py = hx + math.cos(a) * q * R, hy + math.sin(a) * q * R
                if math.hypot(px, py) < 1.08:
                    continue
                dome = math.sqrt(max(0.0, 1 - q * q))
                r = R * rng.uniform(0.22, 0.42) * (0.6 + 0.4 * dome)
                P.append((px, py, base + tall * R * dome * 0.6, r, rng.uniform(-1, 1)))
    # far banks
    for y in np.geomspace(16.0, 320.0, 30):
        span = 12 + y * 1.25
        n = int(30 + min(80, y * 0.4))
        r0 = 0.5 + 0.03 * y
        base = -0.75 - 0.004 * y
        for x in rng.uniform(-span, span, n):
            P.append((x, y + rng.uniform(-0.3, 0.3) * r0, base + rng.uniform(-0.25, 0.4) * r0,
                      r0 * rng.uniform(0.6, 1.4), rng.uniform(-1, 1)))
    return np.array(P, np.float64)


def setup(S):
    st = {}
    cen, verts = HY.enumerate_tiling(7.6)
    st["verts"] = verts
    vh = np.array([hash01(i, 5) for i in range(len(verts))])
    st["vseed"] = (vh * 1e6).astype(np.int64)
    st["stars"] = _stars(S)
    st["clouds"] = _clouds()
    # ember lift-off parameters per vertex
    rng = np.random.default_rng(3)
    nv = len(verts)
    st["ember"] = dict(delay=rng.uniform(0, 0.3, nv), speed=rng.uniform(0.7, 1.4, nv),
                       swirl=rng.uniform(-1, 1, nv), ph=rng.uniform(0, 6.28, nv))
    # ember-flags orbiting the colossal flag on three tilted rings
    n_orb = 330
    ring = rng.integers(0, 3, n_orb)
    st["orbit"] = dict(r=np.array([12.0, 16.5, 21.0])[ring] + rng.uniform(-1.2, 1.2, n_orb),
                       alpha=np.array([0.32, -0.42, 0.2])[ring], beta=np.array([0.25, -0.18, 0.05])[ring],
                       w=np.array([0.42, -0.3, 0.22])[ring] * rng.uniform(0.9, 1.1, n_orb),
                       ph=rng.uniform(0, 6.28, n_orb), jz=rng.uniform(-0.6, 0.6, n_orb),
                       seed=rng.integers(0, 1 << 30, n_orb))
    st["colossal"] = _colossal_flag(S)
    _export_foley(S, st)
    return st


# ------------------------------------------------------------------ the colossal 10,000-foot Flag
LSIM = 600.0      # simulated pole length (design px); drawn at heaven scale
LP = 45.0         # pole length in heaven units (the disk has radius 1: this Flag is 10,000 ft)
FLAG_Y = 60.0     # depth of the pole
FLAG_X = -4.5


def _flag_rise(T):
    """z of the pole top (heaven units)"""
    return _pchip(T, [(165.8, -14.0), (166.3, -9.5), (167.5, 5.0), (169.5, 17.5), (171.5, 23.5), (173.0, 25.5)])


def _colossal_flag(S):
    path = S.cache / "colossal_v4.npz"
    # frame range: from 165.8 to the end; slow motion via fractional draw index
    n = int(round((END - 165.8) * S.fps)) + 2
    if path.exists():
        try:
            return FlagTrack.load(path)
        except Exception:
            pass

    def pole_fn(i):
        return (0.0, 0.0, 0.0)

    def wind_fn(i):
        T = 165.8 + i / S.fps
        rise = max(0.0, (_flag_rise(T + 0.05) - _flag_rise(T - 0.05)) / 0.1)   # units/s upward
        return (850.0 + 150 * math.sin(i * 0.07), 12.0 * rise, 0.0)

    tr = Flag(pole=LSIM, seed=7, nx=30, ny=24).simulate(n, pole_fn, wind_fn, fps=S.fps, warmup=72)
    tr.save(path)
    return tr


def _flag_index(T):
    """fractional sim frame (majestic 0.6x slow motion: a 3 km flag flaps slowly)"""
    return max(0.0, (T - 165.8) * 24.0 * 0.62)


# ------------------------------------------------------------------ foley
def _export_foley(S, st):
    ev = [
        dict(t=PH[0], kind="swell", strength=0.6, pan=0.0, desc="'Flags be' — ripple of light across the plain of flags"),
        dict(t=PH[1], kind="swell", strength=0.7, pan=0.0, desc="'Flags are' — ring of light races across the map's borders"),
        dict(t=PH[2], kind="bloom", strength=0.85, pan=0.0, desc="'Flags will' — Flagistan crystallises: every vertex flag pops open"),
        dict(t=PH[3], kind="bloom", strength=1.0, pan=0.0, desc="'Flags.' — biggest bloom: triple ring flash, the tiling spins"),
        dict(t=T_DISK0, kind="shimmer", strength=0.5, pan=0.0, desc="globe crystallises into the hyperbolic tiling (glassy)"),
        dict(t=LIFT, kind="whoosh", strength=0.8, pan=0.0, desc="thousands of flags lift off the tiling like embers (rising)"),
        dict(t=GATE, kind="impact", strength=0.9, pan=0.0, desc="the Gateless Gate ignites: torii of pure light"),
        dict(t=170.2, kind="whoosh", strength=1.0, pan=0.0, desc="camera passes through the Gateless Gate toward the colossal Flag"),
        dict(t=N10, kind="swell", strength=0.8, pan=0.0, desc="'And the Survey was completed' — bloom builds to white-gold"),
        dict(t=T_WHITE, kind="flash", strength=1.0, pan=0.0, desc="full-frame white-gold (flag_glow) — hand-off to s08"),
    ]
    for k in range(16):
        tb = 154.5 + k * BEAT
        if all(abs(tb - p) > 0.05 for p in PH):
            ev.append(dict(t=tb, kind="shimmer", strength=0.25, pan=0.0, desc="hymn beat: flags breathe light"))
    foley.export_events(S, "hits", ev)
    tr = st["colossal"]
    n = S.n
    energy = np.zeros(n, np.float32)
    f0 = int(round((165.8 - S.start) * S.fps))
    for f in range(f0, n):
        T = S.start + f / S.fps
        i = int(_flag_index(T))
        vis = seg(T, 166.3, 167.2)
        energy[f] = tr.energy[min(i, tr.n - 1)] * vis
    foley.export_track(S, "colossal_flag", energy, pan=0.0, gain=1.0, kind="flag")
    # vertex flags (a shimmering field) — energy follows their visibility
    fe = np.zeros(n, np.float32)
    for f in range(n):
        T = S.start + f / S.fps
        fe[f] = 0.35 * seg(T, T_DISK0, PH[2]) * (1 - seg(T, LIFT, LIFT + 1.5))
    foley.export_track(S, "flag_field", fe, pan=0.0, gain=0.5, kind="flag_field")


# ================================================================== rendering: planet phase
def _front_km(T):
    return _pchip(T, [(154.5, 1.2), (155.5, 2.5), (156.5, 30.0), (157.5, 600.0), (158.5, 25000.0), (160.0, 40000.0)],
                  log=True)


def _planet_ring(T):
    """ring of light racing out across the map on 'Flags be' / 'Flags are' -> (radius km, strength)"""
    for p, r0, r1 in ((PH[0], 0.02, 6.0), (PH[1], 3.0, 1500.0)):
        u = (T - p) / 2.4
        if 0.0 <= u <= 1.0:
            rk = r0 * (r1 / r0) ** ease_out(u, 2)
            rs = 1.3 * math.sin(math.pi * min(1.0, u * 1.6)) ** 0.7 * (1 - u) ** 0.5
            return rk, rs
    return 0.0, 0.0


def _render_planet(fc, st, T, flat=0.0):
    cam, alt = planet_cam(T)
    ring_km, ring_s = _planet_ring(T)
    el, az = math.radians(6.0), math.radians(-72.0)
    sun = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
    bgl = 0.6 * seg(T, 155.9, 157.0, ease_in_out)
    fg = 1.0 + 0.8 * (1 - seg(T, 155.8, 156.8))          # s07a's dense gold speckle carries through the dissolve
    prm = np.array([*sun, T, _front_km(T), ring_km, ring_s, fg, 1.0, 1.6, flat, 0.45, bgl], np.float64)
    out = np.zeros((fc.h, fc.w, 3), np.float32)
    PL.render_planet(out, cam, prm, _rgb(PLATES), _rgb(PCOLS))
    return out, cam, alt


def _ray_hits(cam, n=13, m=8):
    """map coords (km) of a grid of screen rays hitting the planet (for culling)"""
    R = PL.R_PLANET
    th = cam[12]
    sxs, sys_ = np.meshgrid(np.linspace(-1, 1, n) * th, np.linspace(-1, 1, m) * th * H / W)
    d = cam[9:12][None, :] + sxs.ravel()[:, None] * cam[3:6][None, :] + sys_.ravel()[:, None] * cam[6:9][None, :]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    P = cam[0:3] + np.array([0, 0, R])
    b = d @ P
    c = P @ P - R * R
    disc = b * b - c
    ok = (disc > 0) & (b < 0)
    t = c / (-b[ok] + np.sqrt(disc[ok]))
    hit = P + d[ok] * t[:, None] - np.array([0, 0, R])
    return PL.world_to_map(hit)


def _map_flags(ctx, cam, alt, T, fc, alpha=1.0, ring=(0.0, 0.0)):
    """real drawn flags at every country capital large enough on screen (one flag per country, every level)"""
    th = cam[12]
    pa = 2 * th / W                       # radians per design px
    R = PL.R_PLANET
    front = _front_km(T)
    mxs, mys = _ray_hits(cam)
    if len(mxs) == 0:
        return
    x0, x1, y0, y1 = mxs.min(), mxs.max(), mys.min(), mys.max()
    CP0, CP1 = 26.0, 150.0
    xs_all, ys_all, poles, seeds, pops, brs = [], [], [], [], [], []
    for k in range(PL.NLEV):
        sk = PL.S0 * PL.RATIO ** k
        if sk / (alt * pa) < CP0:        # even at the nadir too small
            continue
        dmax = sk / (CP0 * pa)           # beyond this distance cells are too small
        cx, cy = PL.world_to_map(cam[0:3] * np.array([1, 1, 0]))
        bx0, bx1 = max(x0, cx - dmax), min(x1, cx + dmax)
        by0, by1 = max(y0, cy - dmax), min(y1, cy + dmax)
        if bx1 <= bx0 or by1 <= by0:
            continue
        seeds_xy, hs = PL.seeds_in(k, bx0, bx1, by0, by1, max_n=150000)
        if len(seeds_xy) == 0:
            continue
        dm = np.hypot(seeds_xy[:, 0], seeds_xy[:, 1])
        keep = dm < front * 1.02
        seeds_xy, hs, dm = seeds_xy[keep], hs[keep], dm[keep]
        if len(seeds_xy) == 0:
            continue
        P3 = PL.map_to_world(seeds_xy[:, 0], seeds_xy[:, 1])
        scr, z = project(cam, P3)
        nrm = (P3 - np.array([0, 0, -R])) / R
        facing_k = np.einsum("ij,ij->i", nrm, cam[0:3] - P3) / np.maximum(np.linalg.norm(cam[0:3] - P3, axis=1), 1e-9)
        facing = facing_k > 0.05
        cp = sk / (np.maximum(z, 1e-9) * pa) * np.sqrt(np.clip(facing_k, 0, 1))
        ok = facing & (z > 0) & (scr[:, 0] > -60) & (scr[:, 0] < W + 30) & (scr[:, 1] > -10) & (scr[:, 1] < H + 90) \
            & (cp > CP0) & (cp < CP1)
        if not ok.any():
            continue
        scr, cp, hs, dmk = scr[ok], cp[ok], hs[ok], dm[ok]
        a = np.clip((cp - CP0) / 14, 0, 1) * np.clip((CP1 - cp) / 40, 0, 1)
        fr = np.clip((front - dmk) / (0.08 * front + 1e-3), 0, 1)
        a = a * np.clip(fr * 4, 0, 1)
        rk, rs = ring
        rb = rs * np.exp(-((dmk - rk) / (0.12 * rk + 1e-6)) ** 2) if rs > 0 else np.zeros_like(dmk)
        xs_all.append(scr[:, 0])
        ys_all.append(scr[:, 1])
        poles.append(np.clip(0.42 * cp, 8, 64) * (1 + 0.25 * rb))
        seeds.append((hs % 100003).astype(np.int64))
        pops.append(np.stack([a, fr], -1))
        brs.append(np.clip(0.15 + rb, 0, 1))
    if not xs_all:
        return
    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)
    pl = np.concatenate(poles)
    sd = np.concatenate(seeds)
    pp = np.concatenate(pops)
    bb = np.concatenate(brs)
    m = pp[:, 0] > 0.02
    if m.sum() == 0:
        return
    draw_flag_field(ctx, xs[m], ys[m], pl[m], T, seeds=sd[m], wind=0.8, side=1, pop=np.clip(pp[m, 1] * 1.5, 0, 1),
                    alpha=alpha * pp[m, 0], light=(-0.7, -0.3, -0.35), glow=0.3, bright=bb[m])


# ================================================================== rendering: Flagistan disk
def _disk_rings(T):
    """(rho, strength) hyperbolic rings flashing through the tiles"""
    rings = []
    for p, amp, n in ((PH[2], 1.0, 1), (PH[3], 0.85, 3)):
        for j in range(n):
            u = T - (p + j * 0.28)
            if 0 <= u < 3.2:
                rho = 7.5 * ease_out(u / 3.2, 2)
                rings.append((rho, amp * (1 - u / 3.2) ** 1.2 * (1.0 if j == 0 else 0.25)))
    # beat breathing
    for k in range(16):
        tb = 154.5 + k * BEAT
        u = T - tb
        if 0 <= u < 1.2 and tb >= PH[2] - 0.01:
            rings.append((5.0 * ease_out(u / 1.2), 0.18 * (1 - u / 1.2)))
    if not rings:
        rings.append((0.0, 0.0))
    return np.array(rings, np.float64)


def _disk_params(T):
    reveal = 0.0 if T < T_DISK0 else 9.0 * ease_in_out(clamp((T - T_DISK0) / (T_DISK1 - T_DISK0 + 0.4)), 2)
    bright = 1.0 + 0.15 * pulse(T, PH[3] + 0.2, 0.5)
    seam_glow = 0.32 + 0.2 * pulse(T, PH[2], 0.4) + 0.3 * pulse(T, PH[3] + 0.1, 0.6)
    warm = seg(T, PH[3] - 0.15, PH[3])
    fcol = np.array([0.88, 0.94, 1.0]) * (1 - warm) + np.array([1.0, 0.9, 0.72]) * warm
    return np.array([0.017, seam_glow, 0.09, 0.33, reveal, 0.6, 0.9, 0.45, T, 0.9, 0.75, bright, *fcol], np.float64)


def _render_disk(fc, T, cam, Sm):
    out = np.zeros((fc.h, fc.w, 4), np.float32)
    Hm = plane_homography(cam, 0.0, fc.w, fc.h)
    ma, mb = HY.mob_inverse(Sm)
    pal = _rgb(TILES)
    avg = pal.mean(0) * 0.95 + np.array(hex2rgb(SEAM)) * 0.12
    Lr = 0.6 * T
    L = np.array([math.cos(2.3 + 0.05 * (T - 160)), math.sin(2.3 + 0.05 * (T - 160)), 0.9])
    L /= np.linalg.norm(L)
    HY.render_disk(out, Hm, complex(ma), complex(mb), pal, avg, np.array(hex2rgb(SEAM)), _disk_params(T),
                   _disk_rings(T), L)
    return out


def _vertex_flags(ctx, st, T, cam, Sm, alpha=1.0):
    """a plain yellow flag at every vertex, conformally shrinking toward the boundary; embers after LIFT"""
    zs = _mob_apply_arr(Sm, st["verts"])
    r2 = (zs.real ** 2 + zs.imag ** 2)
    keep = r2 < 0.9985 ** 2
    idx = np.nonzero(keep)[0]
    zs = zs[idx]
    r2 = r2[idx]
    lam = (1 - r2) / 2                            # disk units per hyperbolic unit
    Lh = 0.6                                     # pole length (hyperbolic units)
    eb = st["ember"]
    t_l = T - LIFT
    P3 = np.stack([zs.real, zs.imag, np.zeros_like(zs.real)], -1)
    pole3 = Lh * lam
    if t_l > 0:
        d = np.clip(t_l - eb["delay"][idx] * (0.4 + 0.6 * np.sqrt(r2)), 0, None)
        rise = eb["speed"][idx] * (2.6 * d * d + 2.0 * d)
        sw = eb["swirl"][idx]
        ang = eb["ph"][idx] + sw * d * 1.5
        P3[:, 0] += np.cos(ang) * 0.3 * d * sw
        P3[:, 1] += np.sin(ang) * 0.3 * d * sw - 0.3 * d * d
        P3[:, 2] += rise
        # embers grow a little as they rise toward the camera
        pole3 = pole3 * (1 + 0.5 * np.clip(d, 0, 1.2)) + 0.03 * np.clip(d * 3, 0, 1)
    scr, z = project(cam, P3)
    ok = z > 0.05
    px_per_unit = F_PX / np.maximum(z, 1e-6)
    pole_px = pole3 * px_per_unit
    ok &= (pole_px > 1.6) & (scr[:, 0] > -60) & (scr[:, 0] < W + 60) & (scr[:, 1] > -120) & (scr[:, 1] < H + 150)
    if not ok.any():
        return
    # pop-in with the reveal (tile-wise wave from the centre), and ring pulses
    rho = 2 * np.arctanh(np.sqrt(np.clip(r2, 0, 0.999999)))
    prm = _disk_params(T)
    pop = np.clip((prm[4] - rho - 0.4) / 0.8, 0, 1)
    rings = _disk_rings(T)
    fl = np.zeros_like(rho)
    for r_, s_ in rings:
        fl += s_ * np.exp(-((rho - r_) / 0.7) ** 2)
    bright = 0.6 * np.clip(fl, 0, 1.2)
    scale = 1 + 0.18 * np.clip(fl, 0, 1.5)
    a = alpha * np.ones_like(rho)
    g_scalar = 0.2 + 0.5 * float(np.clip(rings[:, 1].max(), 0, 1)) + 0.3 * pulse(T, PH[3] + 0.2, 0.6)
    if t_l > 0:
        d = np.clip(t_l - eb["delay"][idx] * (0.4 + 0.6 * np.sqrt(r2)), 0, None)
        bright = bright + 0.7 * np.clip(d * 1.5, 0, 1)
        g_scalar += 0.35 * clamp(t_l * 1.5)
        a = a * np.clip(1.2 - 0.5 * d, 0, 1)
    sel = ok & (pop > 0.01) & (a > 0.01)
    if not sel.any():
        return
    order = np.argsort(-z[sel])
    xs = scr[sel, 0][order]
    ys = scr[sel, 1][order]
    pl = (pole_px * scale)[sel][order]
    sd = st["vseed"][idx][sel][order]
    draw_flag_field(ctx, xs, ys, pl, T, seeds=sd, wind=0.75, side=1, pop=pop[sel][order],
                    alpha=a[sel][order], light=(-0.5, -0.6, 0.35), glow=g_scalar,
                    bright=np.clip(bright[sel][order], 0, 1), sort=False)


def _space_bg(fc, st, T, warm):
    """deep space with stars; `warm` 0..1 blends toward a luminous gold-rose heaven haze"""
    cv = Canvas(fc)
    ctx = cv.ctx
    top = mix("night", "#3A2A5E", warm)
    bot = mix("#10132A", "#E0906E", warm)
    ctx.set_source(lin_grad(0, 0, 0, H, [(0, top), (1, bot)]))
    ctx.paint()
    s = st["stars"]
    for x, y, m, tw in zip(s["x"], s["y"], s["m"], s["tw"]):
        a = m * (0.75 + 0.25 * math.sin(T * 2.1 + tw)) * (1 - warm)
        if a < 0.03:
            continue
        set_color(ctx, "#E8ECFF", a)
        circle(ctx, x, y, 0.6 + 1.1 * m)
        ctx.fill()
    return cv.rgb()


# ================================================================== heaven: clouds, gate, flag, embers
def _sun_dir():
    el = math.radians(10.0)
    return np.array([math.sin(0.11) * math.cos(el), math.cos(0.11) * math.cos(el), math.sin(el)])


def _sun_screen(cam):
    sd = _sun_dir()
    scr, z = project(cam, cam[0:3][None, :] + sd[None, :] * 1000.0)
    return sd, scr[0], z[0]


def _puffs_view(st, cam):
    """project every cloud puff: returns dict of arrays (culled), sorted far -> near"""
    P = st["clouds"]
    scr, d = project(cam, P[:, :3])
    rp = P[:, 3] * F_PX / np.maximum(d, 1e-6)
    ok = (d > 0.12) & (scr[:, 0] + rp > -20) & (scr[:, 0] - rp < W + 20) & (scr[:, 1] + rp > -20) \
        & (scr[:, 1] - rp < H + 20) & (rp > 0.6)
    idx = np.nonzero(ok)[0]
    idx = idx[np.argsort(-d[idx])]
    # screen direction toward the sun for each puff (its lit side)
    sd = _sun_dir()
    lit, _ = project(cam, P[idx, :3] + sd[None, :] * P[idx, 3:4])
    v = lit - scr[idx]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.where(n > 1e-3, v / np.maximum(n, 1e-6), np.array([[0.0, -1.0]]))
    return dict(scr=scr[idx], d=d[idx], rp=rp[idx], tone=P[idx, 4], v=v, down=float(clamp((-cam[11] - 0.5) / 0.5)))


def _draw_puffs(img, pv, s, dmin=0.0, dmax=1e9, alpha=1.0):
    """composite soft, sun-lit cumulus puffs within camera depth [dmin, dmax) into img (in place, far -> near)"""
    sel = (pv["d"] >= dmin) & (pv["d"] < dmax)
    if not sel.any():
        return img
    d = pv["d"][sel]
    haze = (1 - np.exp(-d / 90.0))[:, None]
    k = (1 + 0.07 * pv["tone"][sel])[:, None]
    hz = np.array(hex2rgb(HCOLS[7]))[None, :]
    cl = np.array(hex2rgb(HCOLS[4]))[None, :] * k * (1 - haze) + hz * haze
    cs = np.array(hex2rgb(HCOLS[5]))[None, :] * k * (1 - haze * 0.8) + hz * haze * 0.8
    cs = cs + (cl - cs) * 0.45 * pv["down"]          # seen from above the heaps are softer, sunlit
    cr = np.array(hex2rgb(HCOLS[6]))[None, :] * (1 - haze * 0.5) + hz * haze * 0.5
    xy = pv["scr"][sel] * s
    v = pv["v"][sel]
    SK.splat_puffs(img, np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1]),
                   np.ascontiguousarray(pv["rp"][sel] * s), np.ascontiguousarray(v[:, 0]),
                   np.ascontiguousarray(v[:, 1]), cl, cs, cr, np.full(len(d), alpha), 0.14)
    return img


GATE_Y = 17.0


def _gate_parts():
    """the Gateless Gate: a torii of pure light (no doors, no plaque), heaven units"""
    y = GATE_Y
    parts = []

    def box(x0, x1, z0, z1):
        return [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]

    for s in (-1, 1):
        xb, xt = s * 4.4, s * 4.12
        parts.append([(xb - 0.17, y, -2.5), (xb + 0.17, y, -2.5), (xt + 0.13, y, 6.2), (xt - 0.13, y, 6.2)])
    parts.append(box(-5.1, 5.1, 4.9, 5.08))            # nuki
    parts.append(box(-5.55, 5.55, 6.26, 6.42))          # shimaki
    xs = np.linspace(-6.9, 6.9, 49)
    lift = 0.62 * (np.abs(xs) / 6.9) ** 2.4
    parts.append([(x, y, 6.84 + l) for x, l in zip(xs, lift)] +
                 [(x, y, 6.52 + 0.9 * l) for x, l in zip(xs[::-1], lift[::-1])])   # kasagi, upswept
    return parts


def _gate_layer(fc, cam):
    cv = Canvas(fc)
    ctx = cv.ctx
    for p in _gate_parts():
        scr, z = project(cam, np.array(p))
        if (z <= 0.05).any():
            continue
        poly(ctx, [tuple(q) for q in scr])
        ctx.set_source_rgba(1.0, 0.96, 0.88, 1.0)
        ctx.fill()
    return cv.rgba()


def _flag_cloth_centre(T):
    top = _flag_rise(T)
    return np.array([FLAG_X + 0.2 * LP, FLAG_Y, top - 0.16 * LP])


def _draw_colossal(ctx, st, cam, T, alpha=1.0, glow=0.35):
    tr = st["colossal"]
    top_z = _flag_rise(T)
    sb, zb = project(cam, np.array([[FLAG_X, FLAG_Y, top_z - LP]]))
    stp, zt = project(cam, np.array([[FLAG_X, FLAG_Y, top_z]]))
    if zt[0] <= 0.5:
        return
    k = (stp[0, 1] - sb[0, 1]) / (-LSIM)       # screen px per sim px
    ctx.save()
    ctx.translate(sb[0, 0], sb[0, 1])
    ctx.scale(k, k)
    tr.draw(ctx, _flag_index(T), alpha=alpha, glow=glow, light=(-0.15, -0.4, -0.8))
    ctx.restore()


def _orbit_flags(st, T, cam):
    """tiny ember-flags orbiting the colossal Flag: returns (xs, ys, pole, seeds, depth, behind)"""
    o = st["orbit"]
    c = _flag_cloth_centre(T)
    ang = o["ph"] + o["w"] * (T - 165.0)
    rr = o["r"] * (1.3 - 0.3 * seg(T, 166.3, 169.5))
    lx, ly = rr * np.cos(ang), rr * np.sin(ang)
    al, be = o["alpha"], o["beta"]
    # ring plane tilted about x (alpha) then about the view axis (beta)
    x0, y0, z0 = lx, ly * np.cos(al), ly * np.sin(al) + o["jz"]
    x = c[0] + x0 * np.cos(be) - z0 * np.sin(be)
    y = c[1] + y0
    z = c[2] + x0 * np.sin(be) + z0 * np.cos(be)
    P3 = np.stack([x, y, z], -1)
    scr, d = project(cam, P3)
    pole = 0.8 * F_PX / np.maximum(d, 1e-6)
    ok = (d > 0.5) & (pole > 1.5)
    return scr[ok, 0], scr[ok, 1], pole[ok], o["seed"][ok], d[ok], (y > c[1])[ok]


def _render_heaven(fc, st, T, cam, include_disk=None, aux=None):
    """golden sky + sea of clouds + gate + colossal flag + orbiting ember-flags + god rays.
    include_disk: callable(img)->img inserting the Flagistan disk and its flags at their depth."""
    sd, sun_xy, sun_z = _sun_screen(cam)
    lw, lh = max(64, fc.w // 4), max(36, fc.h // 4)
    sky = np.zeros((lh, lw, 3), np.float32)
    SK.render_heaven(sky, cam, np.array([*sd, T, 0.9, 0.0, -99.0, 0, 0, 0, 1.0, 0.0], np.float64), _rgb(HCOLS))
    img = cv2.resize(sky, (fc.w, fc.h), interpolation=cv2.INTER_CUBIC) * 0.78
    pv = _puffs_view(st, cam)
    show_gate = T > GATE - 0.35
    d_flag = FLAG_Y - cam[1]
    d_gate = GATE_Y - cam[1]
    occl = None
    img = np.ascontiguousarray(img, dtype=np.float64)
    s = fc.s
    if show_gate:
        # THE ONE RULE: nothing is ever composited over the colossal Flag's cloth. Draw order: far clouds, the gate
        # of light (masked out of the Flag's silhouette so no bar or its glow can cross the cloth), the nearer
        # clouds, the orbiting ember-flags (all behind), and the Flag itself last: it passes in front of / through
        # the gate's opening, never cut by a bar.
        _draw_puffs(img, pv, s, dmin=d_flag)
        cvc = Canvas(fc)
        _draw_colossal(cvc.ctx, st, cam, T, glow=0.35 + 0.5 * seg(T, N10, T_WHITE))
        fl = cvc.rgba()
        # the pole's foot disappears into the sea of clouds
        cut, _ = project(cam, np.array([[FLAG_X, FLAG_Y, -1.1]]))
        ycut = cut[0, 1] * s
        rows = np.arange(fc.h, dtype=np.float32)[:, None, None]
        fl = fl * np.clip((ycut - rows) / (30 * s) + 0.3, 0, 1)
        occl = fl[..., 3:4]
        # a generous silhouette (dilated + feathered) that keeps every light source off the cloth
        rad = max(3, int(round(24 * s)))
        shield = cv2.dilate(np.ascontiguousarray(occl[..., 0]), np.ones((rad, rad), np.uint8))
        shield = np.clip(cv2.GaussianBlur(shield, (0, 0), 10 * s) * 1.5, 0, 1)[..., None]
        gk = seg(T, GATE - 0.3, GATE + 0.45, ease_out) * (1 + 0.4 * pulse(T, GATE + 0.08, 0.2))
        if gk > 0 and d_gate > 0.3:
            g = _gate_layer(fc, cam)[..., :3] * gk
            gl = blur(g, 1.6 * s) * 0.35 + blur(g, 7 * s) * 0.2 + blur(g, 28 * s) * 0.12
            light = (g * 0.95 + gl * np.array([1.0, 0.84, 0.62], np.float32)) * (1 - shield)
            img = img * (1 - np.clip(g, 0, 1) * 0.6 * (1 - shield)) + light
        img = np.ascontiguousarray(img, dtype=np.float64)
        _draw_puffs(img, pv, s, dmax=d_flag)
        xs, ys, pl, sds, dd, behind = _orbit_flags(st, T, cam)
        if len(xs):
            cvo = Canvas(fc)
            draw_flag_field(cvo.ctx, xs, ys, pl, T, seeds=sds, wind=0.9, glow=0.8, light=(-0.1, -0.3, -0.9),
                            bright=0.5)
            img = over(img, cvo.rgba() * (1 - shield))
        img = over(img, fl)
    else:
        d_near = 0.0
        if include_disk is not None:
            rim = np.stack([np.cos(np.linspace(0, 6.283, 24)), np.sin(np.linspace(0, 6.283, 24)), np.zeros(24)], -1)
            _, dr = project(cam, rim)
            d_near = float(dr.min())
        _draw_puffs(img, pv, s, dmin=d_near)
        if include_disk is not None:
            img = np.ascontiguousarray(include_disk(img), dtype=np.float64)
            _draw_puffs(img, pv, s, dmax=d_near)
    shield_ = shield if show_gate else None
    if include_disk is not None and show_gate:
        lay = include_disk(img)
        img = lay * (1 - shield_) + img * shield_
    img = img.astype(np.float32)
    # god rays: the sun behind the Flag scattered through the air, occluded by the Flag (crepuscular fans)
    if sun_z > 0 and T > 165.0:
        q = 4
        hq, wq = fc.h // q, fc.w // q
        sx, sy = sun_xy[0] * fc.s / q, sun_xy[1] * fc.s / q
        yy, xx = np.mgrid[0:hq, 0:wq].astype(np.float32)
        dd = np.hypot(xx - sx, yy - sy) * (q / fc.s)            # design px from the sun
        src = (2.4 * np.exp(-(dd / 60.0) ** 2) + 0.55 * np.exp(-(dd / 240.0) ** 2))[..., None] \
            * np.array([1.0, 0.86, 0.62], np.float32)
        if occl is not None:
            src = src * (1 - cv2.resize(occl, (wq, hq), interpolation=cv2.INTER_AREA)[..., None])
        rays = SK.god_rays(src.astype(np.float32), sx, sy, passes=7, step=0.03, decay=0.985)
        rays = cv2.resize(rays, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)
        if shield_ is not None:
            rays = rays * (1 - shield_)
        img = img + rays * 0.3 * seg(T, 165.5, 166.8)
    if aux is not None:
        aux["shield"] = shield_
    return img


# ================================================================== main render
def render(fc, st):
    T = fc.T
    if T >= T_WHITE:
        return np.ones((fc.h, fc.w, 3), np.float32) * np.array(C["flag_glow"], np.float32)
    # ---------------------------------------------------------------- planet phase
    if T < T_DISK1:
        # hand-off: s07a's own world renderer (pixel-exact continuation of its crane), dissolving into the
        # ray-traced planet while the plates shrink below a pixel
        k07 = 1.0 - seg(T, H07A[0], H07A[1], ease_in_out) if _S07A is not None else 0.0
        if k07 >= 1.0:
            return _S07A(fc, T).astype(np.float32)
        flat = seg(T, T_DISK0 - 0.2, T_DISK1, ease_in_out)
        img, pcam, alt = _render_planet(fc, st, T, flat=flat * 0.85)
        cv = Canvas(fc)
        fa = 1.0 - seg(T, T_DISK0 - 0.3, T_DISK0 + 0.3)
        if fa > 0:
            _map_flags(cv.ctx, pcam, alt, T, fc, alpha=fa, ring=_planet_ring(T))
        img = cv.over(img)
        if k07 > 0.0:
            img = img * (1 - k07) + _S07A(fc, T).astype(np.float32) * k07
        if T >= T_DISK0:
            hcam = heaven_cam(T)
            Sm = flag_mobius(T)
            disk = _render_disk(fc, T, hcam, Sm)
            img = over(img, disk * seg(T, T_DISK0, T_DISK0 + 0.25))
            cv = Canvas(fc)
            _vertex_flags(cv.ctx, st, T, hcam, Sm)
            img = cv.over(img)
        return img
    # ---------------------------------------------------------------- Flagistan + heaven
    hcam = heaven_cam(T)
    Sm = flag_mobius(T)
    # 'Flags.': a shock-front of light leaves the disk rim and paints heaven into the void behind it
    wave_r = 520.0 + 900.0 * ease_out(clamp((T - PH[3] - 0.02) / 0.95), 2)
    wave_on = T >= PH[3] - 0.02

    def disk_and_flags(img):
        disk = _render_disk(fc, T, hcam, Sm)
        img = over(img, disk * (1 - seg(T, LIFT + 0.6, LIFT + 1.6)))
        cv = Canvas(fc)
        _vertex_flags(cv.ctx, st, T, hcam, Sm)
        return cv.over(img)

    if T < PH[3] + 1.0:
        bg = _space_bg(fc, st, T, 0.0)
        if wave_on:
            hv = _render_heaven(fc, st, T, hcam)
            yy, xx = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
            dist = np.hypot(xx / fc.s - W / 2, yy / fc.s - H / 2)[..., None]
            m = np.clip((wave_r - dist) / 60.0 + 0.5, 0, 1)
            bg = bg * (1 - m) + hv * m
            ringk = 0.9 * (1 - clamp((T - PH[3]) / 1.0)) ** 1.5
            bg = bg + np.array([1.0, 0.9, 0.74], np.float32) * ringk * np.exp(-((dist - wave_r) / 34.0) ** 2)
        img = disk_and_flags(bg)
    else:
        aux = {}
        img = _render_heaven(fc, st, T, hcam, include_disk=disk_and_flags if T < LIFT + 1.7 else None, aux=aux)
        if T >= LIFT + 1.7 and T < LIFT + 4.0:
            cv = Canvas(fc)
            _vertex_flags(cv.ctx, st, T, hcam, Sm)
            lay = cv.rgba()
            if aux.get("shield") is not None:
                lay = lay * (1 - aux["shield"])       # never over the colossal cloth
            img = over(img, lay)
    # "Flags." biggest bloom: a white-gold flash washing from the disk
    fl = 0.32 * pulse(T, PH[3] + 0.12, 0.3) + 0.12 * pulse(T, PH[2] + 0.05, 0.18)
    if fl > 0.01:
        img = img + np.array(C["flag_glow"], np.float32) * fl * radial_mask(fc, W / 2, H / 2, 0, 900)
    return img


def post(fc, st):
    T = fc.T
    p = {}
    hv = seg(T, 165.4, 166.4)                   # in heaven: bright scene, keep bloom for highlights only
    b = 0.5 + 0.3 * pulse(T, PH[0] + 0.1, 0.4) + 0.3 * pulse(T, PH[1] + 0.1, 0.4) \
        + 0.4 * pulse(T, PH[2] + 0.1, 0.5) + 0.75 * pulse(T, PH[3] + 0.2, 0.7)
    b = b * (1 - 0.4 * hv) + 1.5 * seg(T, N10, T_WHITE, ease_in)
    p["bloom"] = b
    p["bloom_thresh"] = 0.7 + 0.18 * hv - 0.35 * seg(T, N10, T_WHITE, ease_in)
    p["exposure"] = 1.0 + 0.9 * seg(T, N10 + 0.3, T_WHITE, ease_in)
    # at the cut, s07a's film look (bloom 0.67 / thresh 0.8 / radius 24 / vignette 0.3), easing into ours
    k = 1.0 - seg(T, H07A[0], H07A[1])
    if k > 0:
        p["bloom"] = p["bloom"] * (1 - k) + (0.67 + 0.3 * pulse(T, PH[0] + 0.1, 0.4)) * k
        p["bloom_thresh"] = p["bloom_thresh"] * (1 - k) + 0.8 * k
        p["bloom_radius"] = 22.0 * (1 - k) + 24.0 * k
        p["vignette"] = 0.22 * (1 - k) + 0.3 * k
    fade = seg(T, N10 + 0.7, T_WHITE, ease_in)
    if fade > 0:
        p["fade"] = fade
        p["fade_color"] = C["flag_glow"]
    if T >= T_WHITE - 0.25:
        k = seg(T, T_WHITE - 0.25, T_WHITE)
        p["vignette"] = 0.22 * (1 - k)
    if T >= T_WHITE:
        p.update(fade=1.0, fade_color=C["flag_glow"], vignette=0.0, bloom=0.0, ca=0.0, exposure=1.0)
    return p

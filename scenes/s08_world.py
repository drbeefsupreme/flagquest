"""s08 helper: the morning meadow of flags - a small perspective world.

World units are metres: X right, Y up, Z forward (away from camera). The heroes stand near the origin.
PCam projects world -> design px (1920x1080). Everything is a pure function of (camera, T).
"""
import math
from types import SimpleNamespace

import cairo
import cv2
import numpy as np

from vx import W, H, C, col, mix, set_color, fbm1, fbm2, hash01, smoothstep, clamp, lin_grad, rad_grad
from vx.canvas import Canvas, paint_array
from scenes import s08_people as ppl

try:  # FlagSmith's batch API (preferred); fallback keeps previews working until it lands
    from vx.flag import draw_flag_field as _draw_flag_field
except ImportError:  # pragma: no cover
    _draw_flag_field = None
from vx.flag import draw_flag

# ------------------------------------------------------------------ palette (morning, green-grey, NOT yellow)
M = {k: col(v) for k, v in dict(
    sky_top="#86A0B8", sky_mid="#A9BCCB", sky_low="#CCD6DC", horizon="#E9E6E4", sun="#FFFAF4",
    sun_warm="#F8E6DF", cloud_sh="#A9A9BC", cloud_lit="#F4ECEA",
    hill_far="#B9C3C9", hill_mid="#9EABAF", hill_near="#879790",
    grass_far="#98A89C", grass_mid="#7B8F78", grass_near="#627862", grass_dark="#4A5E4B",
    grass_tip="#C9D5BE", path="#A9AE98", mist="#EEF1F0", bird="#4B5361", rim="#FFF3EA",
    flower_w="#F1F2EE", flower_l="#B9A8CC", flower_b="#A8C0D8",
).items()}

SUN_AZ, SUN_EL = 0.16, 0.16           # sun direction (radians) - morning sun, a little right, ahead
WIND_X = 1.0                          # gentle breeze blowing to the right


# ------------------------------------------------------------------ camera
class PCam:
    def __init__(self, x=0.0, y=1.4, z=-6.0, pitch=0.0, F=1500.0, roll=0.0):
        self.x, self.y, self.z, self.pitch, self.F, self.roll = x, y, z, pitch, F, roll

    def project(self, X, Y, Z):
        """world -> (sx, sy, depth) design px; numpy friendly"""
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        dx, dy, dz = np.asarray(X, float) - self.x, np.asarray(Y, float) - self.y, np.asarray(Z, float) - self.z
        depth = dz * cp - dy * sp
        up = dy * cp + dz * sp
        dd = np.maximum(depth, 1e-3)
        return W / 2 + self.F * dx / dd, H / 2 - self.F * up / dd, depth

    def k(self, depth):
        return self.F / np.maximum(depth, 1e-3)

    def horizon(self):
        return H / 2 - self.F * math.tan(self.pitch)

    def ray(self, sx, sy):
        """design px -> unit world direction arrays"""
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        right = sx - W / 2
        up = H / 2 - sy
        X = right
        Y = up * cp - self.F * sp
        Z = up * sp + self.F * cp
        n = np.sqrt(X * X + Y * Y + Z * Z)
        return X / n, Y / n, Z / n

    def stage(self, X, Z, units=200.0):
        """affine for drawing a 2D 'stage' (units per metre, y down, ground y=0) standing at world (X, 0, Z):
        returns (ox, oy, sc): stage point (u, v) -> screen (ox + u*sc, oy + v*sc)"""
        sx, sy, d = self.project(X, 0.0, Z)
        return float(sx), float(sy), float(self.k(d)) / units


def lerp_cam(a, b, t):
    return PCam(*(pa + (pb - pa) * t for pa, pb in zip((a.x, a.y, a.z, a.pitch, a.F, a.roll),
                                                       (b.x, b.y, b.z, b.pitch, b.F, b.roll))))


def lowres_fc(fc, f):
    """proxy frame context rendering at fc.s * f"""
    s = fc.s * f
    return SimpleNamespace(s=s, w=int(round(W * s)), h=int(round(H * s)), T=fc.T, t=fc.t, f=fc.f, F=fc.F)


def _stops(x, stops):
    ts = [a for a, _ in stops]
    cs = np.array([c for _, c in stops], np.float32)
    return np.stack([np.interp(x, ts, cs[:, k]) for k in range(3)], -1)


def sun_dir():
    return (math.sin(SUN_AZ) * math.cos(SUN_EL), math.sin(SUN_EL), math.cos(SUN_AZ) * math.cos(SUN_EL))


def sun_screen(cam):
    sx, sy, sz = sun_dir()
    x, y, d = cam.project(cam.x + sx * 1e5, cam.y + sy * 1e5, cam.z + sz * 1e5)
    return float(x), float(y)


# ------------------------------------------------------------------ sky + ground (per pixel, low res)
def environment(fc, cam, T, lowres=4):
    """(fc.h, fc.w, 3) sky (gradient, sun, perspective stratus) + meadow ground (texture, mist, haze)"""
    lw, lh = max(16, fc.w // lowres), max(9, fc.h // lowres)
    ys, xs = np.mgrid[0:lh, 0:lw].astype(np.float64)
    sx = (xs + 0.5) / lw * W
    sy = (ys + 0.5) / lh * H
    X, Y, Z = cam.ray(sx, sy)
    sd = sun_dir()
    cosang = np.clip(X * sd[0] + Y * sd[1] + Z * sd[2], -1, 1)
    ang = np.arccos(cosang)
    # ---- sky
    el = np.arcsin(np.clip(Y, -1, 1))
    sky = _stops(np.clip(el, 0, 2), [(0.0, M["horizon"]), (0.035, mix(M["horizon"], M["sky_low"], 0.5)),
                                     (0.14, M["sky_low"]), (0.42, M["sky_mid"]), (1.0, M["sky_top"])])
    glow = (0.35 * np.exp(-ang / 0.075) + 0.20 * np.exp(-ang / 0.32) + 0.5 * np.exp(-ang / 0.025))[..., None]
    sky = sky * (1 - np.clip(glow * 0.6, 0, 0.9)) + np.array(M["sun_warm"]) * np.clip(glow * 0.6, 0, 0.9)
    # perspective stratus clouds on a plane 1400 m up
    up = np.maximum(Y, 0.004)
    tpl = 1400.0 / up
    cx = X * tpl / 2600.0 + T * 0.006
    cz = Z * tpl / 900.0
    dn = fbm2(cx, cz, seed=11, octaves=5)
    dens = np.clip((dn - 0.02) / 0.42, 0, 1) ** 1.4
    dens *= np.clip((Y - 0.004) / 0.06, 0, 1) * 0.85
    lit = np.clip(0.35 + 0.65 * np.exp(-ang / 0.35) + 0.25 * (dn - 0.2), 0, 1)[..., None]
    ccol = np.array(M["cloud_sh"]) * (1 - lit) + np.array(M["cloud_lit"]) * lit
    sky = sky * (1 - dens[..., None]) + ccol * dens[..., None]
    # ---- ground
    down = np.minimum(Y, -1e-4)
    tg = cam.y / (-down)
    gx = cam.x + X * tg
    gz = cam.z + Z * tg
    dist = tg
    n1 = fbm2(gx * 0.045, gz * 0.045, seed=3, octaves=4)
    n2 = fbm2(gx * 0.33, gz * 0.33, seed=5, octaves=3)
    # wind-combed swathes (long, lighter streaks where the grass bends and shows its silver side)
    comb = fbm2(gx * 0.012 + gz * 0.004 - T * 0.05, gz * 0.05, seed=9, octaves=3)
    near = np.exp(-dist / 60.0)
    base = _stops(np.log1p(dist) / np.log(3000.0), [(0.0, M["grass_near"]), (0.45, M["grass_mid"]),
                                                     (0.8, M["grass_far"]), (1.0, M["grass_far"])])
    n3 = fbm2(gx * 2.2, gz * 2.2, seed=7, octaves=2)          # fine grass mottling (only reads up close)
    g = base * (1 + (0.13 * n1 + 0.08 * n2 * near + 0.11 * np.clip(comb, 0, 1) + 0.07 * n3 * near ** 2))[..., None]
    # slow cloud shadows drifting across the meadow with the breeze
    csh = fbm2(gx / 520.0 - T * 0.018, gz / 300.0, seed=13, octaves=3)
    shadow = np.clip((csh - 0.08) / 0.3, 0, 1) * 0.22
    g = g * (1 - shadow[..., None])
    # the trodden path through the field, straight to the horizon
    pw = 0.95 + 0.004 * np.maximum(gz, 0)
    path = np.clip((pw + 0.25 - np.abs(gx)) / 0.5, 0, 1) * (gz > -4) * (0.55 + 0.45 * np.exp(-dist / 400.0))
    g = g * (1 - 0.42 * path[..., None]) + np.array(M["path"]) * 0.42 * path[..., None]
    # backlit sheen toward the sun
    az = np.arctan2(X, Z)
    sheen = np.exp(-np.abs(az - SUN_AZ) / 0.30) * (1 - near * 0.7) * 0.28 * (1 - shadow / 0.22 * 0.6)
    g = g + np.array(M["sun_warm"]) * sheen[..., None] * 0.6
    # low morning mist lying on the meadow
    mn = fbm2(gx * 0.0045 + T * 0.012, gz * 0.011, seed=21, octaves=4)
    mist = np.clip((mn + 0.05) / 0.5, 0, 1) * np.clip((dist - 25) / 160, 0, 1) * 0.5
    g = g * (1 - mist[..., None]) + np.array(M["mist"]) * mist[..., None]
    haze = (1 - np.exp(-dist / 520.0))[..., None]
    g = g * (1 - haze) + np.array(M["horizon"]) * haze
    # merge at the horizon with a soft seam
    kgrd = np.clip(-Y / 0.0025, 0, 1)[..., None]
    img = sky * (1 - kgrd) + g * kgrd
    img = cv2.resize(img.astype(np.float32), (fc.w, fc.h), interpolation=cv2.INTER_CUBIC)
    return img


# ------------------------------------------------------------------ hills
_RIDGES = [  # distance m, base height, amplitude, x-scale, colour key, seed, haze
    (7000.0, 320.0, 320.0, 3000.0, "hill_far", 1, 0.40),
    (4600.0, 110.0, 130.0, 1300.0, "hill_mid", 2, 0.30),
    (3000.0, 30.0, 45.0, 600.0, "hill_near", 3, 0.20),
]


def draw_hills(ctx, cam, T):
    for D, hb, amp, sc, ck, seed, hz in _RIDGES:
        half = (D - cam.z) * (W / 2) / cam.F * 1.25 + 50
        xs = np.linspace(cam.x - half, cam.x + half, 160)
        hs = hb + amp * np.array([fbm1(x / sc, seed, 4) for x in xs])
        if seed == 1:
            hs += 260 * np.exp(-((xs + 2400) / 1500) ** 2)
        px, py, _ = cam.project(xs, np.maximum(hs, 0), np.full_like(xs, D))
        c = mix(M[ck], M["horizon"], hz)
        _, by, _ = cam.project(cam.x, 0.0, D)
        by = float(by) + 1.5
        ctx.new_path()
        ctx.move_to(px[0], by)
        for x, y in zip(px, py):
            ctx.line_to(x, y)
        ctx.line_to(px[-1], by)
        ctx.close_path()
        top = float(py.min())
        g = lin_grad(0, top, 0, by, [(0, mix(c, M["horizon"], 0.05)), (0.7, c), (1, mix(c, M["mist"], 0.7))])
        ctx.set_source(g)
        ctx.fill()


def draw_mist_bands(ctx, cam, T, strength=1.0):
    """soft horizontal mist banks lying on the meadow at several depths (drawn over the far crowd)"""
    for D, thick, a, seed in ((2200.0, 60.0, 0.30, 4), (1000.0, 26.0, 0.26, 5), (420.0, 10.0, 0.2, 6),
                              (170.0, 4.0, 0.14, 7)):
        _, y0, _ = cam.project(cam.x, 0.0, D)
        _, y1, _ = cam.project(cam.x, thick, D)
        y0, y1 = float(y0), float(y1)
        hgt = max(4.0, (y0 - y1))
        for j in range(5):
            xc = ((hash01(j, seed) * 1.4 - 0.2) * W + T * 6 * (1 + j * 0.3)) % (W * 1.6) - W * 0.3
            wd = W * (0.35 + 0.4 * hash01(j + 9, seed))
            g = rad_grad(0, 0, 0, 1, [(0, M["mist"], a * strength), (0.55, M["mist"], a * strength * 0.5),
                                      (1, M["mist"], 0)])
            ctx.new_path()
            ctx.save()
            ctx.translate(xc, y0 - hgt * 0.35)
            ctx.scale(wd * 0.5, hgt * 1.3)
            ctx.arc(0, 0, 1, 0, 2 * math.pi)
            ctx.set_source(g)
            ctx.fill()
            ctx.restore()


# ------------------------------------------------------------------ the crowd (people with flags + planted flags)
_ROBES = [col(c) for c in ("cloth_grey", "cloth_blue", "cloth_teal", "cloth_plum", "cloth_rust", "robe_shadow",
                           "#5E6B7C", "#7A6A78", "#556B66", "#6F7A8C", "#8A6F5E", "robe")]
_SKINS = [col(c) for c in ("skin1", "skin2", "skin3", "skin4")]
_HAIR = [col(c) for c in ("#2A2426", "#3A2A22", "#5A4636", "#D8D6D0", "#1E1C22")]


def make_crowd(seed=8):
    rng = np.random.default_rng(seed)
    items = []
    bands = [(3.0, 90.0, 3.3), (90.0, 350.0, 7.0), (350.0, 1100.0, 15.0), (1100.0, 2600.0, 32.0)]
    for z0, z1, cell in bands:
        zs = np.arange(z0, z1, cell)
        for z in zs:
            half = z * 1.05 + 40
            xs = np.arange(-half, half, cell)
            for x in xs:
                X = x + rng.uniform(-0.45, 0.45) * cell
                Z = z + rng.uniform(-0.45, 0.45) * cell
                if Z < 3.0:
                    continue
                if abs(X) < 2.8 + 0.012 * max(Z, 0):          # the path the heroes walk into, to the horizon
                    continue
                if Z < 15 and abs(X) < 5.0:                  # a clearing around the heroes
                    continue
                items.append((X, Z))
    # a few near the camera sides for the high wide shot
    for _ in range(90):
        X = rng.uniform(5.5, 40) * rng.choice([-1, 1])
        Z = rng.uniform(-14, 3)
        items.append((X, Z))
    P = np.array(items)
    n = len(P)
    person = rng.random(n) < 0.55
    hgt = np.where(person, rng.uniform(1.45, 1.88, n), 0.0)
    pole = np.where(person, hgt * 1.4, rng.uniform(2.2, 2.45, n))
    cr = SimpleNamespace(
        X=P[:, 0], Z=P[:, 1], n=n, person=person, hgt=hgt, pole=pole,
        seed=rng.integers(0, 10 ** 6, n), lean=rng.normal(0, 0.035, n),
        side=np.where(rng.random(n) < 0.5, -1, 1), phase=rng.uniform(0, 6.3, n),
        body=np.array([_ROBES[i] for i in rng.integers(0, len(_ROBES), n)], np.float64),
        skin=np.array([_SKINS[i] for i in rng.integers(0, len(_SKINS), n)], np.float64),
        hair=np.array([_HAIR[i] for i in rng.integers(0, len(_HAIR), n)], np.float64),
        hood=rng.random(n) < 0.4,
        jit=rng.uniform(-0.06, 0.06, n),
    )
    return cr


def salute_delay(d):
    """seconds after the cue at which someone d metres from the heroes raises: fast nearby, the front reaches
    the horizon (~2.6 km) in 1.2 s - 'at once', yet visibly rippling"""
    return 1.2 * (np.maximum(d, 0) / 2600.0) ** 0.4


def salute_raise(cr, T, T0, origin=(0.0, 4.0)):
    """0 -> 1 (+overshoot) raise of every person's flag; a front rippling out from `origin` (X, Z)"""
    d = np.hypot(cr.X - origin[0], cr.Z - origin[1])
    delay = salute_delay(d) + cr.jit * 0.6
    tt = np.maximum(T - T0 - np.maximum(delay, 0), 0.0)
    r = 1 - np.exp(-7.5 * tt) * np.cos(2 * math.pi * 1.35 * tt)   # spring with overshoot
    return np.where(tt > 0, r, 0.0), delay


def crowd_geometry(cr, cam, idx, raise_r):
    """screen geometry (design px) for items idx. returns dict of arrays."""
    sx, sy, d = cam.project(cr.X[idx], 0.0, cr.Z[idx])
    k = cam.k(d)
    hp = cr.hgt[idx] * k
    side = cr.side[idx]
    r = raise_r[idx]
    person = cr.person[idx]
    # hand + pole base for people
    hx = sx + side * hp * (0.20 + (0.13 - 0.20) * r)
    hy = sy - hp * (0.50 + (1.22 - 0.50) * r)
    L = cr.pole[idx] * k
    ang = cr.lean[idx] + side * (0.035 + 0.07 * r)
    grip = 0.30 - 0.20 * np.clip(r, 0, 1)
    bx = np.where(person, hx - np.sin(ang) * L * grip, sx)
    by = np.where(person, hy + np.cos(ang) * L * grip, sy)
    # rest pose: pole butt on the ground beside the person
    by = np.where(person & (r < 0.02), np.minimum(by, sy - 0.0), by)
    ang = np.where(person, ang, cr.lean[idx])
    shx = sx + side * hp * 0.085
    shy = sy - hp * 0.77
    return dict(sx=sx, sy=sy, d=d, k=k, L=L, ang=ang, bx=bx, by=by, hx=hx, hy=hy, shx=shx, shy=shy)


def draw_field_flags(ctx, bx, by, L, ang, t, seeds, wind, alpha=1.0, glow=0.0, light=None, bright=None):
    if len(bx) == 0:
        return
    if _draw_flag_field is not None:
        _draw_flag_field(ctx, bx, by, L, t, seeds=seeds, wind=wind, side=1, ang=ang, alpha=alpha, light=light,
                         glow=glow, bright=bright)
        return
    # fallback (slow): cheap flags for the biggest ones only
    order = np.argsort(L)
    for i in order[-400:]:
        if L[i] < 2.0:
            continue
        draw_flag(ctx, float(bx[i]), float(by[i]), pole=float(L[i]), t=t, wind=float(np.mean(wind) if np.ndim(wind) else wind),
                  side=1, ang=float(ang[i]), seed=int(seeds[i]) % 97, alpha=alpha)


def draw_crowd_bands(fc, cam, cr, T, raise_r, wind_k, zbands, base_img, s=None, rimk=0.6, zmin=1.0, light=None,
                     near_people=None, near_max=0.0, bright=None):
    """composite the crowd over base_img band by band (far -> near) with atmospheric haze per band.
    near_people(ctx, sel, geometry) draws people of bands with z1 <= near_max itself (e.g. with the full rig)."""
    s = fc.s if s is None else s
    img = base_img
    sx_all, sy_all, d_all = cam.project(cr.X, 0.0, cr.Z)
    vis = (d_all > zmin) & (sx_all > -300) & (sx_all < W + 300)
    for z0, z1 in zbands:
        sel = np.nonzero(vis & (d_all >= z0) & (d_all < z1))[0]
        if len(sel) == 0:
            continue
        sel = sel[np.argsort(-d_all[sel])]
        g = crowd_geometry(cr, cam, sel, raise_r)
        ok = (g["sy"] > -50) & (g["sy"] - g["L"] * 1.6 < H + 50)
        sel = sel[ok]
        g = {kk: v[ok] for kk, v in g.items()}
        if len(sel) == 0:
            continue
        cv = Canvas(w=fc.w, h=fc.h, s=s)
        pm = cr.person[sel]
        wk = wind_k[sel] if np.ndim(wind_k) else np.full(len(sel), wind_k)
        br = None if bright is None else bright[sel]
        if near_people is not None and z1 <= near_max:
            near_people(cv.ctx, sel, g, wk, br)
        else:
            if pm.any():
                buf = np.zeros((fc.h, fc.w, 4), np.float32)
                pi = np.nonzero(pm)[0]
                hand = np.stack([g["shx"][pi], g["shy"][pi], g["hx"][pi], g["hy"][pi]], 1)
                ppl.draw_people(buf, g["sx"][pi], g["sy"][pi], g["k"][pi], cr.hgt[sel][pi], cr.lean[sel][pi] * 0.5,
                                cr.body[sel][pi], cr.skin[sel][pi], cr.hair[sel][pi], cr.hood[sel][pi], hand,
                                np.array(M["rim"]), np.full(len(pi), rimk), np.arange(len(pi)), s)
                cv.paint_array(buf, 0, 0)
            draw_field_flags(cv.ctx, g["bx"], g["by"], g["L"], g["ang"], T, cr.seed[sel], wk, light=light,
                             glow=0.1 if br is None else 0.1 + 0.5 * br, bright=br)
        lay = cv.rgba()
        zc = 0.5 * (z0 + min(z1, 6000))
        hz = 1 - math.exp(-zc / 500.0)
        if hz > 0.01:
            a = lay[..., 3:4]
            lay[..., :3] = lay[..., :3] * (1 - hz) + np.array(M["horizon"], np.float32) * a * hz
        img = lay[..., :3] + img * (1 - lay[..., 3:4])
    return img


# ------------------------------------------------------------------ birds
def draw_birds(ctx, cam, T, flock, alpha=1.0):
    """flock: list of dict(X,Y,Z,vx,vy,vz,phase,span) world; wings flap"""
    for b in flock:
        t = T - b["t0"]
        X = b["X"] + b["vx"] * t
        Y = b["Y"] + b["vy"] * t + 0.6 * math.sin(t * 0.9 + b["phase"])
        Z = b["Z"] + b["vz"] * t
        sx, sy, d = cam.project(X, Y, Z)
        if d < 1:
            continue
        k = float(cam.k(d))
        span = b["span"] * k
        if span < 1.2 or not (-50 < sx < W + 50 and -50 < sy < H + 50):
            continue
        sx, sy = float(sx), float(sy)
        fl = math.sin(T * b["rate"] * 2 * math.pi + b["phase"])
        wy = -fl * span * 0.28
        hz = 1 - math.exp(-float(d) / 900.0)
        c = mix(M["bird"], M["horizon"], hz * 0.8)
        set_color(ctx, c, alpha)
        ctx.new_path()
        ctx.move_to(sx - span * 0.5, sy + wy)
        ctx.curve_to(sx - span * 0.3, sy + wy * 0.3 - span * 0.06, sx - span * 0.1, sy - span * 0.02, sx, sy + span * 0.05)
        ctx.curve_to(sx + span * 0.1, sy - span * 0.02, sx + span * 0.3, sy + wy * 0.3 - span * 0.06, sx + span * 0.5, sy + wy)
        ctx.curve_to(sx + span * 0.28, sy + wy * 0.2 + span * 0.02, sx + span * 0.08, sy + span * 0.1, sx, sy + span * 0.12)
        ctx.curve_to(sx - span * 0.08, sy + span * 0.1, sx - span * 0.28, sy + wy * 0.2 + span * 0.02, sx - span * 0.5, sy + wy)
        ctx.close_path()
        ctx.fill()


def make_flock(seed, n, center, vel, spread, t0, span=0.34, rate=3.2):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(dict(X=center[0] + rng.normal(0, spread[0]), Y=center[1] + rng.normal(0, spread[1]),
                        Z=center[2] + rng.normal(0, spread[2]), vx=vel[0] * rng.uniform(0.9, 1.1),
                        vy=vel[1] + rng.normal(0, 0.3), vz=vel[2], phase=rng.uniform(0, 6.3),
                        span=span * rng.uniform(0.8, 1.2), rate=rate * rng.uniform(0.85, 1.15), t0=t0))
    return out


# ------------------------------------------------------------------ grass
def make_tufts(seed, n, xr, zr):
    rng = np.random.default_rng(seed)
    tufts = []
    for i in range(n):
        X = rng.uniform(*xr)
        Z = rng.uniform(*zr)
        blades = []
        for b in range(rng.integers(5, 11)):
            blades.append((rng.normal(0, 0.05), rng.uniform(0.12, 0.38), rng.normal(0, 0.25), rng.uniform(0, 6.3),
                           rng.uniform(0.012, 0.022)))
        flower = rng.random() < 0.35
        tufts.append(dict(X=X, Z=Z, blades=blades, flower=flower, fcol=int(rng.integers(0, 3)),
                          fh=rng.uniform(0.25, 0.45)))
    return tufts


def draw_tufts(ctx, cam, T, tufts, wind=1.0, alpha=1.0, zmin=0.4):
    sxs, sys_, ds = cam.project(np.array([t["X"] for t in tufts]), 0.0, np.array([t["Z"] for t in tufts]))
    order = np.argsort(-ds)
    fcs = [M["flower_w"], M["flower_l"], M["flower_b"]]
    for i in order:
        tf = tufts[i]
        d = float(ds[i])
        if d < zmin:
            continue
        k = float(cam.k(d))
        x0, y0 = float(sxs[i]), float(sys_[i])
        if x0 < -200 or x0 > W + 200 or y0 < -50 or y0 > H + 400:
            continue
        hz = 1 - math.exp(-d / 420.0)
        dark = mix(M["grass_dark"], M["horizon"], hz)
        tip = mix(M["grass_tip"], M["horizon"], hz)
        for (ox, hb, bend, ph, wb) in tf["blades"]:
            sway = wind * (0.10 + 0.06 * math.sin(T * 1.7 + ph + tf["X"] * 0.8)) + bend * 0.4
            bx = x0 + ox * k
            L = hb * k
            tx = bx + sway * L
            ty = y0 - L * (1 - 0.12 * abs(sway))
            w2 = wb * k * 0.5
            g = lin_grad(bx, y0, tx, ty, [(0, dark), (0.65, mix(dark, tip, 0.35)), (1, tip)])
            ctx.new_path()
            ctx.move_to(bx - w2, y0)
            ctx.curve_to(bx - w2, y0 - L * 0.5, tx - sway * L * 0.3, ty + L * 0.3, tx, ty)
            ctx.curve_to(tx - sway * L * 0.3 + w2 * 0.6, ty + L * 0.3, bx + w2, y0 - L * 0.5, bx + w2, y0)
            ctx.close_path()
            ctx.set_source(g)
            ctx.fill()
        if tf["flower"]:
            fx = x0 + (wind * 0.12 + 0.03 * math.sin(T * 1.5 + tf["X"])) * tf["fh"] * k
            fy = y0 - tf["fh"] * k
            set_color(ctx, dark, alpha)
            ctx.set_line_width(max(0.6, 0.006 * k))
            ctx.move_to(x0, y0)
            ctx.curve_to(x0, y0 - tf["fh"] * k * 0.5, fx, fy + tf["fh"] * k * 0.3, fx, fy)
            ctx.stroke()
            set_color(ctx, mix(fcs[tf["fcol"]], M["horizon"], hz), alpha)
            r = max(0.8, 0.018 * k)
            for m in range(5):
                a = m * 2 * math.pi / 5 + tf["X"]
                ctx.arc(fx + math.cos(a) * r * 0.9, fy + math.sin(a) * r * 0.9, r * 0.7, 0, 2 * math.pi)
                ctx.fill()

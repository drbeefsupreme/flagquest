"""t06 — THE FIRMAMENT OPENED (122.29–142.04).  Owner: T06Firmament.

One continuous engraved shot (Gustave Doré): dusk falls on the effigy with the ALL YELLOW FLAG on its head
and the crystal firmament propped on the pole's tip; the effigy burns (a real gas simulation of fire, soot
and steam, printed as engraving); the flames reach the Flag and it lifts FREE and ascends like an ember,
through the crystal; the unpropped firmament cracks (Voronoi fracture racing from the apex) and falls in
shards; the waters above pour down in torrents; the fire dies in a colossal steam explosion; the flood
rises and spills over the panel's borders, soaking the tract page. The Flag stays pristine throughout.

Compositing per frame: grey world (cairo + per-pixel sky + simulated fields) -> vx.ink.engrave -> page/soak
-> vx.ink.spot(flag layer) -> lettering -> POST.
"""
import math

import cairo
import cv2
import numpy as np

from vx import *
from vx import ink, comic, foley
from vx.flag import Flag, FlagTrack
from vx.config import CACHE as FILM_CACHE

from scenes import effigy as E
from scenes import t06_fire as FI
from scenes import t06_crack as CR
from scenes import t06_water as WA

POST = dict(bloom=0.0, grain=0.0, vignette=0.14)


def post(fc, state):
    """per-frame film look (defining post() also shadows the vx.post module that `import *` pulls in)"""
    return {}


TL = get_timeline()
T_EVE = TL.scene("t06")["start"]
T_END = TL.scene("t06")["end"]
T_IGN = TL.cue("ignite")
T_REACH = TL.cue("flames_reach_flag")
T_CRACK = TL.cue("firmament_crack")
T_DELUGE = TL.cue("deluge")
T_LIFT = T_REACH + 0.06
T_TINK = TL.word("P19", "back")[0] - 0.05     # the first tiny star-crack where the pole left the apex
T_BREAK = T_DELUGE + 0.05                     # shards let go
T_HIT = 138.30                                # first torrents reach the fire
T_PAGE = 139.85                               # the panel recedes into the page

KF = 100.0                                    # flag-space px per metre (== t05's hero-flag plane, PPM 100)


# ============================================================ camera
# position, look-at target and focal length keyframes; between T_REACH and the crack the target follows the
# Flag itself (the camera cranes up with the ember).
_KEYS = [  # T, pos(x, y, z), target(x, y, z), f
    (T_EVE, tuple(E.HANDOFF.pos), tuple(E.HANDOFF.pos + E.HANDOFF.F * 40.0), E.HANDOFF.f),   # == t05's last frame
    (T_EVE + 1.5, (0.0, 4.2, -29.0), (0.0, 10.4, 0.0), 1250.0),           # dusk falls; push in
    (T_IGN - 0.25, (0.0, 8.4, -21.5), (0.0, 10.2, 0.0), 1060.0),          # plinth to Flag: ignition at the legs
    (T_IGN + 2.2, (0.4, 10.0, -16.5), (0.0, 11.6, 0.0), 1090.0),          # crane up with the climbing fire
    (T_REACH - 0.3, (1.0, 13.2, -9.8), (0.0, 16.3, 0.0), 1420.0),         # P18: the Flag, big, as the flames reach it
    (T_LIFT + 1.6, (1.1, 14.6, -9.3), (0.0, 19.5, 0.0), 1400.0),          # follow the ember...
    (T_LIFT + 3.4, (1.0, 15.3, -9.0), (0.0, 23.5, 0.0), 1350.0),          # ... up through the crystal
    (T_CRACK - 0.5, (0.4, 3.5, -28.0), (0.0, 19.5, 0.0), 930.0),          # whoosh back: the whole dome
    (T_CRACK + 0.4, (0.2, 3.1, -29.0), (0.0, 19.0, 0.0), 930.0),
    (T_DELUGE + 0.3, (0.1, 2.8, -30.0), (0.0, 19.5, 0.0), 930.0),
    (T_HIT - 0.1, (0.0, 3.0, -38.0), (0.0, 10.4, 0.0), 1000.0),           # tilt down with the torrents
    (T_PAGE, (0.0, 4.4, -40.5), (0.0, 8.0, 0.0), 985.0),
    (T_END + 0.2, (0.0, 4.8, -42.0), (0.0, 7.4, 0.0), 975.0),
]


def _cam_interp():
    from scipy.interpolate import PchipInterpolator
    ts = np.array([k[0] for k in _KEYS])
    vals = np.array([[*k[1], *k[2], k[3]] for k in _KEYS])
    return PchipInterpolator(ts, vals, axis=0, extrapolate=True)


_CAMI = _cam_interp()


def flag_center(T):
    bx, by, a, _ = flag_path(T)
    L = E.POLE_M * 0.82
    return bx + L * math.sin(a) + 0.35, by + L * math.cos(a)


def cam_at(T):
    v = _CAMI(min(max(T, _KEYS[0][0]), _KEYS[-1][0]))
    pos = np.array(v[:3])
    tgt = np.array(v[3:6])
    f = float(v[6])
    wf = smoothstep(T_REACH - 0.5, T_LIFT + 0.9, T) * (1.0 - smoothstep(T_TINK - 0.3, T_CRACK - 0.6, T))
    if wf > 0:
        fx, fy = flag_center(T)
        tgt = tgt * (1 - wf) + np.array([fx * 0.8, fy - 0.35, 0.0]) * wf
    d = tgt - pos
    yaw = math.atan2(d[0], d[2])
    pitch = math.atan2(d[1], math.hypot(d[0], d[2]))
    if T <= T_EVE + 1e-6:
        pitch, yaw = E.HANDOFF.pitch, E.HANDOFF.yaw
    roll = 0.0
    amp = (0.004 * pulse(T, T_IGN + 0.1, 0.25) + 0.006 * pulse(T, T_CRACK + 0.1, 0.25)
           + 0.010 * pulse(T, T_BREAK + 0.3, 0.5) + 0.016 * pulse(T, T_HIT + 0.35, 0.6))
    if amp > 1e-5:
        pitch += amp * fbm1(T * 11.0, 3, 3)
        yaw += amp * fbm1(T * 11.0, 9, 3)
        roll += amp * 0.8 * fbm1(T * 9.0, 21, 3)
    return E.Cam3(pos=tuple(pos), yaw=yaw, pitch=pitch, roll=roll, f=f)


def _proj(cam, P):
    x, y, z = cam.project(np.asarray(P, np.float64).reshape(-1, 3))
    return np.asarray(x), np.asarray(y), np.asarray(z)


def plane_affine(cam, x, y):
    """cairo matrix mapping flag-space px (X = x*KF, Y = -y*KF, plane z=0) to screen near world (x, y)"""
    m = cam.plane_matrix((x, y, 0.0), KF)
    # plane_matrix has (0, 0) at the origin; shift so flag-space coords are absolute
    return cairo.Matrix(m.xx, m.yx, m.xy, m.yy, m.x0 - m.xx * x * KF + m.xy * y * KF,
                        m.y0 - m.yx * x * KF + m.yy * y * KF)


# ============================================================ the Flag
def flag_path(T):
    """pole base (x, y) m, pole angle (rad, + clockwise) and glow at global time T"""
    sx, sy = E.SEAT[0], E.SEAT[1]
    if T < T_LIFT:
        buffet = smoothstep(T_IGN + 1.5, T_REACH, T)
        ang = 0.02 * buffet * math.sin(T * 7.3) + 0.012 * buffet * math.sin(T * 12.1)
        return sx, sy, ang, 0.0
    t = T - T_LIFT
    # like an ember: a slow, floating start, then a steady climb on the updraft
    a, b = 0.55, 0.28
    if t < 6.0:
        rise = a * t * t / (1.0 + b * t)
    else:
        t6 = 6.0
        r6 = a * t6 * t6 / (1.0 + b * t6)
        v6 = (2 * a * t6 * (1 + b * t6) - a * t6 * t6 * b) / (1 + b * t6) ** 2
        rise = r6 + v6 * (t - t6)
    drift = 0.9 * math.sin(t * 0.55) * min(1.0, t / 2.0) + 0.25 * t
    spin = 0.05 * t * t / (1.0 + 0.12 * t)
    L = E.POLE_M
    cx, cy = sx + drift, sy + 0.78 * L + rise
    bx = cx - 0.78 * L * math.sin(spin)
    by = cy - 0.78 * L * math.cos(spin)
    glow = smoothstep(0.0, 1.2, t)
    return bx, by, spin, glow


def build_flag(S):
    """hero flag: the same cloth as t05's (seed, scale, wind) continued without a seam: our first 12 frames
    blend from t05's saved track into our simulation, which then carries it through the fire and the ascent"""
    path = S.cache / f"flag_v6_{E.POLE_M:.2f}.npz"
    if path.exists():
        return FlagTrack.load(path)

    def pole_fn(i):
        bx, by, a, _ = flag_path(S.start + i / S.fps)
        return bx * KF, -by * KF, a

    def wind_fn(i):
        T = S.start + i / S.fps
        up = 520.0 * smoothstep(T_IGN + 0.8, T_REACH - 0.3, T)
        if T > T_LIFT:
            up *= math.exp(-(T - T_LIFT) / 2.5)
        calm = smoothstep(T_EVE, T_EVE + 2.5, T)
        return (420.0 * (1 - calm) + (240.0 + 90.0 * math.sin(T * 0.7)) * calm, -up)

    tr = Flag(pole=E.POLE_M * KF, seed=505, turbulence=1.1).simulate(S.n, pole_fn, wind_fn, fps=S.fps,
                                                                   substeps=10, warmup=48)
    hero = FILM_CACHE / "t05" / "flag_hero.npz"
    if hero.exists():
        h5 = FlagTrack.load(hero)
        if h5.verts.shape[1:] == tr.verts.shape[1:]:
            T0 = float(h5.meta.get("T0", 112.85))
            V = tr.verts.copy()
            for i in range(12):
                j = min((S.start + i / S.fps - T0) * S.fps, h5.n - 1.0)
                vj, _ = h5._frame(j)
                w = smoothstep(0.0, 11.0, i)
                V[i] = vj * (1 - w) + tr.verts[i] * w
            tr = FlagTrack(V, tr.poles, tr.energy, tr.snap, tr.meta, tr.flap_hz)
    tr.save(path)
    return tr


def flag_layer(fc, st, cam, T, f, extra=None):
    """the spot-yellow plate: ONLY the flag. extra = cairo.Matrix applied after the camera (page pull-back)"""
    c = Canvas(fc)
    ctx = c.ctx
    bx, by, a, glow = flag_path(T)
    ctx.save()
    if extra is not None:
        ctx.transform(extra)
    ctx.transform(plane_affine(cam, bx, by + E.POLE_M * 0.7))
    lightdir = (0.1, 0.85, 0.35) if T < T_LIFT else (-0.2, 0.6, 0.5)
    st["flag"].draw(ctx, f, glow=0.35 * glow, light=lightdir)
    ctx.restore()
    return c.rgba()


# ============================================================ crowd: T05's burners, relit by the fire
def crowd_pos(people, T, panic):
    """the burners scatter outward from the pyre once the sky breaks"""
    run = max(0.0, T - T_BREAK - 0.3) * 2.4 * panic
    out = []
    for p in people:
        q = dict(p)
        r = math.hypot(p["x"], p["z"]) + 1e-6
        k = 1.0 + run * (0.6 + 0.4 * ((p["seed"] % 7) / 7.0)) / r
        q["x"] = p["x"] * k
        q["z"] = p["z"] * k
        out.append(q)
    return out


def draw_crowd(ctx, cam, people, T, fire_screen, fire_amt, panic, front, level=0.0, dusk=1.0, only=None,
               rmin=-1.0):
    """T05's burners (same people as the hand-off), relit: at night solid-ink silhouettes rim-lit by the fire"""
    sel = [p for p in people if ((p["z"] < 0.0) == front) and math.hypot(p["x"], p["z"]) > rmin]
    if only is not None:
        sel = [p for p in sel if (only == "fore") == bool(p["fore"])]
    if not sel:
        return
    awe = 1.0 - 0.45 * smoothstep(T_EVE, T_IGN, T) + 0.45 * smoothstep(T_IGN, T_IGN + 1.0, T)
    awe = max(awe, panic)
    night = smoothstep(0.35, 0.9, dusk)
    if night <= 0.01:
        E.draw_people(ctx, cam, sel, T, awe=awe, dusk=dusk, style=dict(render="ink"))
        return
    g = 0.03 + 0.1 * (1.0 - fire_amt)
    fx, fy = fire_screen
    for p in sel:
        sx, sy, z = cam.project(np.array([[p["x"], 0.0, p["z"]]]))
        if z[0] < 0.8:
            continue
        hpx = p["h"] * cam.f / z[0]
        ctx.save()
        if level > 0.01:
            _, wy, _ = cam.project(np.array([[p["x"], level, p["z"]]]))
            ctx.rectangle(-4000, -4000, 10000, float(wy[0]) + 4000)
            ctx.clip()
        if fire_amt > 0.03 or panic > 0.1:
            dx, dy = fx - float(sx[0]), fy - (float(sy[0]) - hpx * 0.6)
            dn = math.hypot(dx, dy) + 1e-6
            rim = min(4.5, 0.02 * hpx + 0.7)
            ctx.save()
            ctx.translate(dx / dn * rim, dy / dn * rim)
            wv = 0.62 + 0.36 * max(fire_amt, panic)
            E.draw_people(ctx, cam, [p], T, awe=awe, style=dict(silhouette=(wv, wv, wv)))
            ctx.restore()
        E.draw_people(ctx, cam, [p], T, awe=awe, style=dict(silhouette=(g, g, g)))
        ctx.restore()


# ============================================================ setup
def _beam_fire(fire, S):
    """per-frame per-beam flame exposure -> glow (white-hot) and char (charcoal) for draw_effigy"""
    b = E.beams()
    mid = 0.5 * (b["p0"] + b["p1"])
    ci, cj = (mid[:, 0] - FI.X0) / FI.HC - 0.5, (FI.YTOP - mid[:, 1]) / FI.HC - 0.5
    ii = np.clip(np.round(ci).astype(int), 0, FI.NX - 1)
    jj = np.clip(np.round(cj).astype(int), 0, FI.NY - 1)
    fl = fire["flame"][:, jj, ii].astype(np.float32) / 90.0
    glow = np.clip((fl - 0.35) / 0.9, 0, 1)
    char = np.clip(np.cumsum(fl, 0) / S.fps / 1.8, 0, 1)
    return glow.astype(np.float32), char.astype(np.float32)


def setup(S):
    st = {}
    st["people"] = E.crowd(seed=0)
    st["flag"] = build_flag(S)
    st["frac"] = CR.build()
    st["ptab"] = CR.pair_table(st["frac"])
    st["fall"] = T_BREAK + st["frac"]["delay"]
    st["holes"] = WA.build(st["frac"], E.canopy_h, st["fall"])
    st["t_ground"] = float(WA.ground_contact(st["holes"]).min())
    holes = st["holes"]
    ev = dict(ignite=T_IGN, reach=T_REACH, crack=T_CRACK, open=T_BREAK + 0.25, quench0=T_HIT - 0.1,
              quench1=T_HIT + 1.2, boom=T_HIT)
    st["fire"] = FI.simulate(S, E, lambda f: S.start + f / S.fps, ev,
                             water_fn=lambda T, sim: WA.water_plane(holes, T, sim))
    st["bglow"], st["bchar"] = _beam_fire(st["fire"], S)
    export_sound(S, st)
    return st


def export_sound(S, st):
    fire = st["fire"]
    e = fire["energy"].astype(np.float64)
    e = e / max(1e-6, np.percentile(e, 99.5))
    foley.export_track(S, "fire", np.clip(e, 0, 1.5), fire["pan"], gain=1.0, kind="fire")
    st["flag"].export_foley(S, "flag")
    ev = [
        dict(t=T_IGN, kind="whoosh", strength=1.0, pan=0.0, desc="FWOOSH: accelerant ignites, fire races up the effigy"),
        dict(t=T_REACH, kind="whoosh", strength=0.7, pan=0.0, desc="flames reach the Flag"),
        dict(t=T_LIFT + 0.05, kind="miracle", strength=0.8, pan=0.0, desc="the Flag lifts free (pole and all), ember-like"),
        dict(t=T_TINK, kind="crack", strength=0.25, pan=0.0, desc="tink: a tiny star-crack where the pole left the apex"),
        dict(t=T_CRACK, kind="crack", strength=1.0, pan=0.0, desc="firmament cracks race from the apex across the whole sky-dome"),
        dict(t=T_BREAK, kind="shatter", strength=1.0, pan=0.0, desc="the firmament gives way: crystal shards fall"),
        dict(t=T_BREAK + 0.2, kind="torrent", strength=1.0, pan=0.0, desc="the waters above pour down (deluge roar)"),
        dict(t=T_HIT, kind="steam", strength=1.0, pan=0.0, desc="torrents hit the fire: colossal steam explosion"),
        dict(t=T_HIT + 0.25, kind="scream", strength=1.0, pan=0.0, desc="the crowd screams"),
    ]
    foley.export_events(S, "hits", ev)


# ============================================================ render helpers
def _sky(fc, cam, T, dusk):
    kw = dict(getattr(E, "HANDOFF_SKY", {}) or {})
    kw.pop("t", None)
    kw["dusk"] = dusk
    # the storm closes back in over the deluge
    st_ = smoothstep(T_BREAK + 0.2, T_BREAK + 3.0, T)
    if st_ > 0:
        kw["cover"] = max(kw.get("cover", 0.0), st_)
        kw["reveal_r"] = 90.0 + 900.0 * (1.0 - st_)
        kw["storm"] = st_
    d = E.sky(fc, cam, T, **kw)
    if "cx" in d:
        X, Z = d["cx"], d["cz"]
    else:
        X, Z = d["r"] * np.cos(d["th"]), d["r"] * np.sin(d["th"])
    return d, X.astype(np.float32), Z.astype(np.float32)


def _mpp(X, Z, s):
    gxX = np.abs(np.gradient(X, axis=1)) + np.abs(np.gradient(X, axis=0))
    gxZ = np.abs(np.gradient(Z, axis=1)) + np.abs(np.gradient(Z, axis=0))
    return (np.hypot(gxX, gxZ) * 0.7 * s).astype(np.float32)   # metres per DESIGN px


def _ground_r(fc, cam, res=0.25):
    """distance (m) from the effigy's axis of the playa point seen by each pixel (inf above the horizon)"""
    hh, ww = max(8, int(fc.h * res)), max(8, int(fc.w * res))
    sy, sx = np.mgrid[0:hh, 0:ww].astype(np.float64)
    sx = (sx + 0.5) * (1920.0 / ww)
    sy = (sy + 0.5) * (1080.0 / hh)
    d = cam.ray(sx, sy)
    dy = d[..., 1]
    ok = dy < -1e-4
    sg = np.where(ok, -cam.pos[1] / np.where(ok, dy, -1.0), 1e6)
    R = np.hypot(cam.pos[0] + d[..., 0] * sg, cam.pos[2] + d[..., 2] * sg)
    R = np.where(ok, R, 1e4).astype(np.float32)
    return cv2.resize(R, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)


def _crack_time(T):
    """seconds since the crack began (Dijkstra time), with the little prelude at the apex"""
    if T < T_TINK:
        return None
    if T < T_CRACK:
        return min(T - T_TINK, 0.035)
    return 0.035 + (T - T_CRACK) * (0.85 + 0.15 * smoothstep(T_CRACK, T_CRACK + 1.0, T))


def _fall_start(st):
    fr = st["frac"]
    return T_BREAK + fr["delay"]


def _fire_fields(fc, cam, st, f):
    fire = st["fire"]
    # homography grid -> screen from the frame's corners cast onto the plane z = 0 (robust when the camera
    # is close to the plane and the grid's far corners are behind it)
    scr = np.array([[0.0, 0.0], [1920.0, 0.0], [1920.0, 1080.0], [0.0, 1080.0]])
    d = cam.ray(scr[:, 0], scr[:, 1])
    if (d[:, 2] <= 1e-4).any():
        return None
    tt = -cam.pos[2] / d[:, 2]
    wx = cam.pos[0] + d[:, 0] * tt
    wy = cam.pos[1] + d[:, 1] * tt
    gi = ((wx - FI.X0) / FI.HC).astype(np.float32)
    gj = ((FI.YTOP - wy) / FI.HC).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(np.stack([gi, gj], 1).astype(np.float32), (scr * fc.s).astype(np.float32))
    fl = fire["flame"][f].astype(np.float32) * (1.0 / 90.0)
    sm = fire["smoke"][f].astype(np.float32) * (1.0 / 160.0)
    stm = fire["steam"][f].astype(np.float32) * (1.0 / 60.0)
    # soft edges where the simulated domain ends (no hard cut-offs in the sky)
    if "edge" not in _TEX:
        ey = np.clip(np.arange(FI.NY, dtype=np.float32) / 40.0, 0, 1)[:, None]
        xx = np.arange(FI.NX, dtype=np.float32)
        ex = np.clip(np.minimum(xx, FI.NX - 1 - xx) / 60.0, 0, 1)[None, :]
        e = ey * ex
        _TEX["edge"] = (e * e * (3 - 2 * e)).astype(np.float32)
    sm = sm * _TEX["edge"]
    stm = stm * _TEX["edge"]
    # lighting in grid space: smoke lit from below by the flames, steam self-shadowed from the sky
    lg = cv2.GaussianBlur(fl, (0, 0), 10.0)
    smoke_tone = 0.16 + 0.7 * np.clip(lg * 1.6, 0, 1)
    shadow = np.cumsum(stm, axis=0) * (FI.HC * 0.9)
    steam_tone = 0.34 + 0.7 * np.exp(-shadow * 0.16) + 0.25 * np.clip(lg * 1.5, 0, 1)
    out = {}
    wp = lambda a: cv2.warpPerspective(a, Hm, (fc.w, fc.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if fl.any():
        H2 = Hm @ np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], np.float64)
        det = _flame_detail(fl, fc.T)
        out["flame"] = cv2.warpPerspective(det, H2, (fc.w, fc.h), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT)
        out["flame_soft"] = wp(fl)
    else:
        out["flame"] = None
    if sm.any():
        out["smoke_a"] = wp(1.0 - np.exp(-1.4 * sm))
        out["smoke_t"] = wp(smoke_tone.astype(np.float32))
    else:
        out["smoke_a"] = None
    if stm.any():
        out["steam_a"] = wp(1.0 - np.exp(-2.2 * stm))
        out["steam_t"] = wp(np.clip(steam_tone, 0, 1).astype(np.float32))
    else:
        out["steam_a"] = None
    # embers (plane z=0)
    ea = fire["ember_a"][f]
    k = np.nonzero(ea > 10)[0]
    if len(k):
        ex = fire["embers"][f, k].astype(np.float64)
        P0 = np.stack([ex[:, 0], ex[:, 1], np.zeros(len(k))], 1)
        P1 = P0 - np.stack([ex[:, 2], ex[:, 3], np.zeros(len(k))], 1) / 36.0
        x0, y0, _ = _proj(cam, P0)
        x1, y1, _ = _proj(cam, P1)
        out["embers"] = (x0, y0, x1, y1, ea[k] / 255.0)
    else:
        out["embers"] = None
    return out


def _clip_near(Pc, zn=0.5):
    out = []
    k = len(Pc)
    for i in range(k):
        a, b = Pc[i], Pc[(i + 1) % k]
        ia, ib = a[2] > zn, b[2] > zn
        if ia:
            out.append(a)
        if ia != ib:
            out.append(a + (b - a) * (zn - a[2]) / (b[2] - a[2]))
    return np.array(out) if out else np.zeros((0, 3))


def _to_screen(cam, P):
    xc, yc, zc = cam.cam_coords(P)
    Pc = _clip_near(np.stack([xc, yc, zc], 1))
    if len(Pc) < 3:
        return None
    return np.stack([960.0 + cam.f * Pc[:, 0] / Pc[:, 2], 540.0 - cam.f * Pc[:, 1] / Pc[:, 2]], 1), Pc[:, 2].mean()


def draw_shards(ctx, cam, st, T, fire_amt, near=None, zsplit=0.0):
    """the falling crystal: every fallen Voronoi cell breaks into tumbling 3D shards, glassy (the dark waters
    show through them), their facets and edges catching the last of the firelight"""
    fr = st["frac"]
    fall = st["fall"]
    items = []
    fire = np.array([0.0, 8.0, 0.0])
    for sub in fr["subs"]:
        i = sub[0]
        ts = T - fall[i] - sub[4]
        if ts <= 0 or fr["rc"][i] > 120.0:
            continue
        if 0.5 * 9.81 * ts * ts + 4.5 * ts > float(E.canopy_h(fr["rc"][i])) + 2.0:
            continue
        P, c = CR.sub_pose(sub, ts, E.canopy_h)
        if c[1] < -1.0:
            continue
        sc = _to_screen(cam, P)
        if sc is None:
            continue
        pts, depth = sc
        if depth < 9.0 or (near is not None and ((depth < zsplit) != near)):
            continue
        ext = pts.max(0) - pts.min(0)
        if ext[0] * ext[1] > 90000.0:
            continue
        if pts[:, 0].max() < -50 or pts[:, 0].min() > 1970 or pts[:, 1].max() < -50 or pts[:, 1].min() > 1130:
            continue
        nrm = np.cross(P[1] - P[0], P[2] - P[0])
        nrm /= np.linalg.norm(nrm) + 1e-9
        L = fire - c
        L /= np.linalg.norm(L) + 1e-9
        lam = abs(float(np.dot(nrm, L)))
        items.append((depth, pts, lam, fr["fac"][i]))
    items.sort(key=lambda it: -it[0])
    for depth, pts, lam, fac in items:
        ctx.move_to(*pts[0])
        for q in pts[1:]:
            ctx.line_to(*q)
        ctx.close_path()
        glint = lam ** 6
        g = 0.55 + 0.25 * fac + 0.4 * glint
        ctx.set_source_rgba(g, g, g, 0.35 + 0.5 * glint)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.0, 0.0, 0.0, 0.9)
        ctx.set_line_width(max(0.7, min(2.4, 45.0 / depth)))
        ctx.stroke_preserve()
        ctx.set_source_rgba(1, 1, 1, 0.95)
        ctx.set_line_width(max(0.5, min(1.4, 22.0 / depth)))
        ctx.stroke()


def _column_hull(cam, poly, h, y_bot):
    """screen silhouette of a vertical prism (hole polygon from y_bot up to h), clipped at the near plane"""
    k = len(poly)
    step = max(1, k // 7)
    P = poly[::step]
    top = np.stack([P[:, 0], np.full(len(P), h), P[:, 1]], 1)
    bot = np.stack([P[:, 0], np.full(len(P), y_bot), P[:, 1]], 1)
    _, _, zt = cam.cam_coords(top)
    _, _, zb = cam.cam_coords(bot)
    pts = []
    for q in range(len(P)):
        a, b, za, zb_ = top[q], bot[q], zt[q], zb[q]
        if za < 0.6 and zb_ < 0.6:
            continue
        if za < 0.6:
            a = a + (b - a) * (0.6 - za) / (zb_ - za)
        if zb_ < 0.6:
            b = b + (a - b) * (0.6 - zb_) / (za - zb_)
        pts.append(a)
        pts.append(b)
    if len(pts) < 3:
        return None
    x, y, z = cam.project(np.array(pts))
    hull = cv2.convexHull(np.stack([x, y], 1).astype(np.float32)).reshape(-1, 2)
    return hull, float(np.mean(z))


def draw_torrents(ctx, cam, st, T, front, ground):
    """the waters above pour through every hole: each column's silhouette as BLACK coverage (-> textured
    falling water in numpy), its ragged spray front and landing mist as WHITE"""
    items = []
    for H in st["holes"]:
        tau = T - H["t0"]
        if tau <= 0:
            continue
        zc = H["c"][1]
        if (zc < -2.0) != front or H["r"] > 34.0:
            continue
        yb = max(ground, H["h"] - float(WA.drop(tau)))
        # each hole pours a finite slug: once the sea above has drained through it, the tail falls too
        dur = 0.9 + 0.8 * ((H["seed"] % 997) / 997.0)
        yt = H["h"] - float(WA.drop(tau - dur)) if tau > dur else H["h"]
        if yt <= ground + 0.2:
            continue
        c = H["c"]
        _, _, dz = cam.cam_coords(np.array([[c[0], 0.5 * (yt + yb), c[1]]]))
        if dz[0] < 12.0:
            continue
        # pour through the inner part of the hole (the rim of the hole sheds only a drizzle)
        poly = c + (H["poly"] - c) * 0.62
        hull = _column_hull(cam, poly, yt, yb)
        if hull is None:
            continue
        pts, depth = hull
        if pts[:, 0].max() < -60 or pts[:, 0].min() > 1980 or pts[:, 1].max() < -60 or pts[:, 1].min() > 1140:
            continue
        items.append((depth, pts, H, tau, yb))
    items.sort(key=lambda it: -it[0])
    for depth, pts, H, tau, yb in items:
        ppm = cam.f / max(depth, 0.5)
        ctx.move_to(*pts[0])
        for q in pts[1:]:
            ctx.line_to(*q)
        ctx.close_path()
        ctx.set_source_rgba(0, 0, 0, 0.72)
        ctx.fill()
        c = H["c"]
        (fx,), (fy,), (fz,) = cam.project(np.array([[c[0], yb, c[1]]]))
        if fz < 0.6:
            continue
        r2 = np.random.default_rng(H["seed"])
        wcol = 0.62 * math.sqrt(H["area"]) * ppm
        landed = yb <= ground + 1e-3
        rr = wcol * (0.35 if not landed else 0.6 + 0.3 * min(2.5, tau - 1.2))
        for k in range(1 if not landed else 3):
            ox = r2.uniform(-0.6, 0.6) * wcol
            oy = r2.uniform(-0.1, 0.25) * rr - (rr * 0.5 if landed else 0)
            rk = rr * r2.uniform(0.5, 0.95)
            g = cairo.RadialGradient(fx + ox, fy + oy, 0, fx + ox, fy + oy, rk)
            g.add_color_stop_rgba(0, 1, 1, 1, 0.6)
            g.add_color_stop_rgba(0.6, 1, 1, 1, 0.22)
            g.add_color_stop_rgba(1, 1, 1, 1, 0.0)
            ctx.set_source(g)
            ctx.arc(fx + ox, fy + oy, rk, 0, 2 * math.pi)
            ctx.fill()


def _curtain_tex(fc, cam, T, res=0.5):
    """falling-water texture in perspective: streaks radiate from the vertical vanishing point and stream
    away from the zenith (downwards) at the water's speed"""
    hh, ww = int(fc.h * res), int(fc.w * res)
    ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float32)
    xs = xs / (ww / 1920.0)
    ys = ys / (hh / 1080.0)
    Fy = float(cam.F[1])
    tex = _tex("streak")
    if abs(Fy) > 0.05:
        vx = 960.0 + cam.f * float(cam.R[1]) / Fy
        vy = 540.0 - cam.f * float(cam.U[1]) / Fy
        dx, dy = xs - vx, ys - vy
        ang = np.arctan2(dx, dy if Fy > 0 else -dy)
        dist = np.log(np.hypot(dx, dy) + 1.0)
        u = ang * 900.0
        v = dist * 160.0 - T * 260.0 * (1 if Fy > 0 else -1)
    else:
        u = xs * 1.0
        v = ys * 0.25 - T * 260.0
    a = cv2.remap(tex, (u % 256).astype(np.float32), (v % 256).astype(np.float32), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_WRAP)
    ridge = np.clip((a - 0.25) / 0.5, 0, 1)
    t = 0.26 + 0.74 * ridge ** 1.3
    return cv2.resize(t, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)


def _apply_torrents(tone, rgba, tex):
    if rgba is None:
        return tone
    al = rgba[..., 3]
    wht = rgba[..., 1]
    blk = np.clip(al - wht, 0, 1)
    return tone * (1 - al) + tex * blk + wht


def draw_rain(ctx, T, amt, seed=3):
    """engraved white rain streaks, two depths"""
    if amt <= 0:
        return
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for layer, (n, ln, lw, v, al) in enumerate(((900, 26.0, 1.1, 1500.0, 0.55), (240, 70.0, 2.2, 2600.0, 0.8))):
        rng = np.random.default_rng(seed + layer)
        x0 = rng.uniform(-200, 2100, n)
        y0 = rng.uniform(0, 1400, n)
        m = rng.uniform(0, 1, n) < amt
        dx, dy = 0.22, 1.0
        y = (y0 + v * T) % 1400 - 150
        x = (x0 + dx * v * T) % 2300 - 200
        ctx.set_source_rgba(1, 1, 1, al)
        ctx.set_line_width(lw)
        for i in np.nonzero(m)[0]:
            ctx.move_to(x[i], y[i])
            ctx.line_to(x[i] - dx * ln, y[i] - dy * ln)
        ctx.stroke()
    ctx.restore()


def flood_tone(fc, cam, T, level, r_front, res=0.5):
    """per-pixel flood surface (y = level) seen by the camera: dark water, foam crests racing toward us.
    returns (tone, coverage) at full res"""
    hh, ww = int(fc.h * res), int(fc.w * res)
    sy, sx = np.mgrid[0:hh, 0:ww].astype(np.float64)
    sx = (sx + 0.5) / (ww / 1920.0)
    sy = (sy + 0.5) / (hh / 1080.0)
    d = cam.ray(sx, sy)
    dy = d[..., 1]
    ok = dy < -1e-4
    sg = np.where(ok, (level - cam.pos[1]) / np.where(ok, dy, -1.0), 0.0)
    X = cam.pos[0] + d[..., 0] * sg
    Z = cam.pos[2] + d[..., 2] * sg
    R = np.hypot(X, Z)
    n1 = value_noise2(X / 7.0, (Z + T * 9.0) / 2.2, 5)
    n2 = value_noise2(X / 2.5 + 3.0, (Z + T * 14.0) / 0.9, 9)
    ridge = 1.0 - np.abs(n1 + 0.4 * n2)
    crest = np.clip((ridge - 0.72) / 0.28, 0, 1) ** 2
    dist = np.clip(sg / 120.0, 0, 1)
    tone = 0.13 + 0.08 * n2 + 0.62 * crest * (1 - dist) + 0.28 * dist
    edge = np.clip((r_front - R) / 3.0, 0, 1)
    cov = np.clip(edge * ok, 0, 1)
    tone = cv2.resize(tone.astype(np.float32), (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)
    cov = cv2.resize(cov.astype(np.float32), (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)
    return tone, cov


# ============================================================ the surge (Doré wave) and the page
T_SURGE = 138.95               # the torrents' water gathers into a surge wave rolling out from the pyre
PAGE_Z = 0.68                  # panel scale once pulled back
PAGE_CY = 592.0                # panel centre (design px) once pulled back: room above for the Flag, below for the spill


def surge(T):
    """(radius m, height m) of the surge front"""
    if T < T_SURGE:
        return None
    t = T - T_SURGE
    R = 7.0 + 31.5 * (1.0 - math.exp(-t / 1.05))
    H = 1.0 + 4.2 * smoothstep(0.0, 2.2, t)
    return R, H


def page_matrix(T):
    """scene (screen design px) -> page (screen design px): the frame recedes into a panel of the tract"""
    k = ease_in_out(clamp((T - T_PAGE) / 1.15))
    z = 1.0 + (PAGE_Z - 1.0) * k
    cy = 540.0 + (PAGE_CY - 540.0) * k
    return cairo.Matrix(z, 0, 0, z, 960.0 - 960.0 * z, cy - 540.0 * z), z, k


def draw_surge(ctx, cam, T, level, seed=8):
    """the wall of water rolling toward us: engraved face, white foam crest, Doré/Hokusai curling claws"""
    sg = surge(T)
    if sg is None:
        return None
    R, H = sg
    t = T - T_SURGE
    cp = cam.pos
    ths = np.linspace(-math.pi, math.pi, 181)
    X, Z = R * np.cos(ths), R * np.sin(ths)
    facing = (X * cp[0] + Z * cp[2]) > 0.25 * R * math.hypot(cp[0], cp[2])
    idx = np.nonzero(facing)[0]
    if len(idx) < 3:
        return None
    # keep the visible arc contiguous (the near side)
    X, Z, th = X[idx], Z[idx], ths[idx]
    order = np.argsort(np.arctan2(Z - cp[2], X - cp[0]))
    X, Z, th = X[order], Z[order], th[order]
    ox, oz = np.cos(th), np.sin(th)
    hvar = H * (0.78 + 0.22 * np.sin(th * 7.0 + t * 1.3) + 0.12 * np.sin(th * 17.0 - t * 2.1))
    rcam = math.hypot(cp[0], cp[2])
    reach = np.minimum(hvar * 1.1, max(0.4, rcam - R - 3.0))
    toe = np.stack([X + ox * reach, np.full_like(X, level), Z + oz * reach], 1)
    crest = np.stack([X, level + hvar, Z], 1)
    lo_ = np.minimum(hvar * 0.55, max(0.3, rcam - R - 3.5))
    lip = np.stack([X + ox * lo_, level + hvar * 0.92, Z + oz * lo_], 1)
    # keep the contiguous run of the arc that is well in front of the camera and near the frame
    _, _, z1 = cam.cam_coords(toe)
    _, _, z2 = cam.cam_coords(crest)
    ok = (z1 > 2.0) & (z2 > 2.0)
    ok &= np.abs(np.asarray(cam.project(crest)[0]) - 960.0) < 2600.0
    if ok.sum() < 3:
        return None
    runs, cur = [], []
    for i in range(len(ok)):
        if ok[i]:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    keep = np.array(max(runs, key=len))
    toe, crest, lip, th = toe[keep], crest[keep], lip[keep], th[keep]

    def pr(P):
        x, y, _ = cam.project(P)
        return np.asarray(x), np.asarray(y)
    tx, ty = pr(toe)
    cx, cy = pr(crest)
    lx, ly = pr(lip)
    # body
    ctx.move_to(tx[0], ty[0])
    for i in range(1, len(tx)):
        ctx.line_to(tx[i], ty[i])
    for i in range(len(cx) - 1, -1, -1):
        ctx.line_to(cx[i], cy[i])
    ctx.close_path()
    gy0, gy1 = float(np.min(cy)), float(np.max(ty))
    g = cairo.LinearGradient(0, gy0, 0, max(gy1, gy0 + 1))
    g.add_color_stop_rgb(0.0, 0.62, 0.62, 0.63)
    g.add_color_stop_rgb(0.35, 0.3, 0.3, 0.31)
    g.add_color_stop_rgb(1.0, 0.1, 0.1, 0.11)
    ctx.set_source(g)
    ctx.fill()
    # the overhanging lip: a dark tube under the curl, lit edge on the lip
    ctx.move_to(cx[0], cy[0])
    for i in range(1, len(cx)):
        ctx.line_to(cx[i], cy[i])
    for i in range(len(lx) - 1, -1, -1):
        ctx.line_to(lx[i], ly[i] + 0.25 * abs(ty[i] - cy[i]))
    ctx.close_path()
    ctx.set_source_rgb(0.05, 0.05, 0.06)
    ctx.fill()
    ctx.set_source_rgba(1, 1, 1, 0.9)
    ctx.set_line_width(max(1.5, float(np.median(np.abs(ty - cy))) * 0.025))
    ctx.move_to(lx[0], ly[0] + 0.25 * abs(ty[0] - cy[0]))
    for i in range(1, len(lx)):
        ctx.line_to(lx[i], ly[i] + 0.25 * abs(ty[i] - cy[i]))
    ctx.stroke()
    # flowing lines down the face (the water sheets over the curl)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    rng = np.random.default_rng(seed)
    for i in range(0, len(tx), 2):
        u = rng.uniform(0.25, 0.9)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        ctx.set_line_width(max(0.8, 0.012 * abs(ty[i] - cy[i]) + 0.6))
        ctx.move_to(cx[i] + (lx[i] - cx[i]) * 0.2, cy[i] + (ly[i] - cy[i]) * 0.2)
        ctx.line_to(cx[i] + (tx[i] - cx[i]) * u, cy[i] + (ty[i] - cy[i]) * u)
        ctx.stroke()
    # foam crest
    for w_, a_ in ((14.0, 1.0), (5.0, 1.0)):
        ctx.set_source_rgba(1, 1, 1, a_)
        ctx.move_to(cx[0], cy[0])
        for i in range(1, len(cx)):
            ctx.line_to(cx[i], cy[i])
        wscale = float(np.median(np.abs(ty - cy))) / 60.0
        ctx.set_line_width(max(2.0, min(w_ * wscale * 0.35, 28.0)))
        ctx.stroke()
    # curling claws along the crest
    n_curl = max(4, len(cx) // 6)
    for k in range(n_curl):
        i = int((k + 0.5 + 0.3 * math.sin(k * 2.3)) / n_curl * (len(cx) - 1))
        i = min(max(i, 0), len(cx) - 1)
        rad = min(110.0, 0.3 * abs(ty[i] - cy[i]) * (0.7 + 0.4 * abs(math.sin(k * 1.7))))
        if rad < 3:
            continue
        ccx = cx[i] + (lx[i] - cx[i]) * 0.9
        ccy = cy[i] + rad * 0.9
        a0 = -math.pi * 0.5 - (t * 1.5 + k) % 0.6
        pts = []
        for q in range(46):
            u = q / 45.0
            ang = a0 + u * 2.6 * math.pi
            r = rad * (1.0 - 0.8 * u)
            pts.append((ccx + math.cos(ang) * r, ccy + math.sin(ang) * r))
        ctx.set_source_rgb(1, 1, 1)
        ctx.set_line_width(max(2.0, rad * 0.32))
        ctx.move_to(*pts[0])
        for q in pts[1:]:
            ctx.line_to(*q)
        ctx.stroke_preserve()
        ctx.set_source_rgb(0.02, 0.02, 0.02)
        ctx.set_line_width(max(0.8, rad * 0.06))
        ctx.stroke()
        # claws: spray fingers flung off the lip
        for j in range(5):
            ang = -math.pi * 0.85 + j * 0.28
            L = rad * (1.1 + 0.5 * math.sin(j * 3.1 + k))
            ctx.set_source_rgba(1, 1, 1, 0.9)
            ctx.set_line_width(max(1.2, rad * 0.1))
            ctx.move_to(ccx + math.cos(ang) * rad, ccy + math.sin(ang) * rad)
            ctx.curve_to(ccx + math.cos(ang) * (rad + L * 0.5), ccy + math.sin(ang) * (rad + L * 0.5) - L * 0.2,
                         ccx + math.cos(ang + 0.3) * (rad + L), ccy + math.sin(ang + 0.3) * (rad + L),
                         ccx + math.cos(ang + 0.5) * (rad + L * 1.2), ccy + math.sin(ang + 0.5) * (rad + L * 1.2))
            ctx.stroke()
    # spray dots over the crest
    for i in range(0, len(cx), 3):
        for j in range(3):
            h = abs(ty[i] - cy[i])
            ctx.set_source_rgba(1, 1, 1, 0.8)
            ctx.arc(cx[i] + rng.normal(0, 0.1) * h, cy[i] - rng.uniform(0.05, 0.5) * h, max(1.0, 0.02 * h), 0,
                    2 * math.pi)
            ctx.fill()
    return R


def draw_spill(ctx, T, z, k):
    """water pouring over the panel's bottom border and down the page (page screen coords)"""
    if k <= 0.05:
        return None
    t = T - (T_PAGE + 0.55)
    if t <= 0:
        return None
    pb = 540.0 + (PAGE_CY - 540.0) * k + 540.0 * z    # panel bottom border
    x0, x1 = 960.0 - 960.0 * z, 960.0 + 960.0 * z
    rng = np.random.default_rng(31)
    fall = min(1.0, t / 0.9)
    ybot = pb + (1080.0 + 60.0 - pb) * fall
    # sheets over the lip
    xs = np.linspace(x0 - 30, x1 + 30, 34)
    ctx.move_to(xs[0], pb - 18)
    for i, x in enumerate(xs):
        ctx.line_to(x, pb - 18 - 10 * math.sin(i * 1.3 + T * 5))
    for i, x in enumerate(xs[::-1]):
        wob = 18 * math.sin(i * 0.7 + T * 3.0)
        ctx.line_to(x + wob, ybot - 30 * abs(math.sin(i * 2.1)))
    ctx.close_path()
    ctx.set_source_rgba(0.28, 0.28, 0.29, 0.97)
    ctx.fill()
    # streaming white lines
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for j in range(70):
        x = rng.uniform(x0 - 20, x1 + 20)
        sp = (rng.uniform(0, 1) + t * rng.uniform(0.7, 1.3)) % 1.0
        ya = pb + (ybot - pb) * sp
        yb = min(ybot, ya + rng.uniform(40, 140))
        ctx.set_source_rgba(1, 1, 1, 0.85)
        ctx.set_line_width(rng.uniform(1.5, 4.0))
        ctx.move_to(x, ya)
        ctx.line_to(x + rng.uniform(-6, 6), yb)
        ctx.stroke()
    # foam where it goes over the border
    ctx.set_source_rgb(1, 1, 1)
    ctx.set_line_width(9.0)
    ctx.move_to(x0 - 30, pb - 6)
    for i, x in enumerate(np.linspace(x0 - 30, x1 + 30, 60)):
        ctx.line_to(x, pb - 6 - 8 * abs(math.sin(i * 0.9 + T * 6.0)))
    ctx.stroke()
    return pb, ybot


def _tile_noise(n=256, seed=5, sx=6.0, sy=2.0):
    """tileable smooth noise, anisotropic (stretched vertically: flame tongues), range ~[-1, 1]"""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, n))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    k = np.exp(-((fx * sx * 6.0) ** 2 + (fy * sy * 6.0) ** 2))
    a = np.real(np.fft.ifft2(np.fft.fft2(w) * k))
    a /= np.abs(a).max() + 1e-9
    return a.astype(np.float32)


_TEX = {}


def _tex(name):
    if name not in _TEX:
        _TEX[name] = {"tongue": lambda: _tile_noise(256, 5, 1.4, 5.5),
                      "streak": lambda: _tile_noise(256, 21, 0.7, 9.0),
                      "tongue2": lambda: _tile_noise(256, 9, 2.6, 8.0)}[name]()
    return _TEX[name]


def _flame_detail(fl, T):
    """sim flame (grid) -> 2x grid with licking tongues: the envelope is sampled through an upward
    displacement that is long vertically and narrow horizontally (tongues), plus dark rifts between licks"""
    up = 2
    f2 = cv2.resize(fl, (fl.shape[1] * up, fl.shape[0] * up), interpolation=cv2.INTER_CUBIC)
    h, w = f2.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    n1 = _tex("tongue")
    n2 = _tex("tongue2")
    a = cv2.remap(n1, (xs * 0.9) % 256, (ys * 0.35 + T * 70.0) % 256, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    b = cv2.remap(n2, (xs * 1.6 + 17) % 256, (ys * 0.6 + T * 150.0) % 256, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_WRAP)
    D = np.clip(0.5 + 0.5 * a, 0, 1) ** 1.6 * 34.0
    det = cv2.remap(f2, xs + 3.0 * b, ys + D, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return det * (0.78 + 0.34 * b)


def _canvas_tone(c):
    a = c.rgba()
    return a[..., 1], a[..., 3]


# ============================================================ render
def render(fc, st):
    T = fc.T
    f = fc.f
    cam = cam_at(T)
    dusk = smoothstep(T_EVE, T_EVE + 1.7, T)
    fire = st["fire"]
    fr = st["frac"]
    fl_amt = float(np.clip(fire["energy"][f] / max(1e-6, fire["energy"].max()) * 1.8, 0, 1))
    panic = smoothstep(T_BREAK, T_HIT + 0.3, T)
    s = fc.s

    # ---------------------------------------------------------------- sky, firmament, cracks
    d, X, Z = _sky(fc, cam, T, dusk)
    hit = d["hit"]
    pane_a = d["pane_a"].copy()
    water = d["water"].copy()
    # the fire lights the canopy's underside around the apex
    rr = d["r"]
    fl_glow = fl_amt * (0.5 * np.exp(-rr / 9.0) + 0.14 * np.exp(-rr / 40.0)) * hit
    ct = _crack_time(T)
    core = flank = None
    if ct is not None and T < T_BREAK + 3.2:
        valid = (hit > 0.5) & (rr < 700.0)
        mpp_r = d["mpp"] / s if "mpp" in d else _mpp(X, Z, 1.0)      # metres per render px
        core, flank, cid, front = CR.evaluate(fr, st["ptab"], X, Z, mpp_r, valid, ct, width=1.5 * s)
        # facets: each shard catches the firelight differently once its crack has closed around it
        facet = np.where(cid >= 0, fr["fac"][np.maximum(cid, 0)], 0.0).astype(np.float32)
        fk = smoothstep(T_CRACK, T_CRACK + 1.2, T) * valid
        pane_a = np.clip(pane_a + 0.25 * fk * np.abs(facet), 0, 1)
        water = water + 0.07 * fk * facet
        if T >= T_BREAK:
            fs = _fall_start(st)
            gone_cells = (fs <= T).astype(np.float32)
            gone = np.where(cid >= 0, gone_cells[np.maximum(cid, 0)], 0.0) * valid
            pane_a *= 1.0 - gone
            core *= 1.0 - gone
            flank *= 1.0 - gone
            water = water * (1.0 - 0.45 * gone)          # the waters above come down through the hole
    ptone = d["pane"]
    sky_t = water * (1 - pane_a) + ptone * pane_a
    sky_t = sky_t * (1 - d["cloud_a"]) + d["cloud"] * d["cloud_a"]
    # night: the sky goes to near-solid ink; the fire lights the crystal's underside around the apex
    # (a band of dying twilight stays along the horizon so the effigy and the crowd read as black silhouettes)
    hy = 540.0 + cam.f * math.tan(cam.pitch)
    yy = (np.arange(fc.h, dtype=np.float32) / fc.s)[:, None]
    twi = np.exp(-np.maximum(hy - yy, 0.0) / 260.0) * (yy < hy)
    sky_t = sky_t * (1.0 - 0.74 * dusk * (1.0 - 0.55 * twi)) + 0.22 * dusk * twi
    sky_t = sky_t + (0.92 - sky_t) * np.clip(fl_glow, 0, 0.9)
    gR = _ground_r(fc, cam)
    pool = fl_amt * (0.8 * np.exp(-gR / 7.0) + 0.3 * np.exp(-gR / 22.0))
    ground = d["ground"] * (1.0 - 0.72 * dusk)
    ground = ground + (0.85 - ground) * np.clip(pool, 0, 0.85)
    tone = sky_t * hit + ground * (1 - hit)
    if core is not None:
        tone = tone * (1 - flank)
        tone = np.maximum(tone, core)

    # ---------------------------------------------------------------- the flood
    level = WA.flood_level(T, st["t_ground"])
    if level > 0.0:
        r_front = 34.0 * (T - st["t_ground"])
        ft_, fcov = flood_tone(fc, cam, T, level, r_front, res=0.3)
        g = (1 - hit) * fcov
        tone = tone * (1 - g) + ft_ * g

    # ---------------------------------------------------------------- back world (cairo)
    _, _, zeff = _proj(cam, np.array([[0.0, 8.0, 0.0]]))
    zeff = float(zeff[0])
    wc = Canvas(fc)
    ctx = wc.ctx
    rib_a = 1.0 - smoothstep(T_BREAK, T_BREAK + 0.35, T)
    if rib_a > 0:
        boss = (1.0 - 0.4 * smoothstep(T_EVE, T_EVE + 2.0, T)) * (1.0 - smoothstep(T_LIFT, T_LIFT + 1.2, T))
        E.draw_ribs(ctx, cam, alpha=rib_a, dusk=dusk, t=T, boss=boss)
    fsx, fsy, _ = _proj(cam, np.array([[0.0, 7.0, 0.0]]))
    people = crowd_pos(st["people"], T, panic)
    draw_crowd(ctx, cam, people, T, (fsx[0], fsy[0]), fl_amt, panic, front=False, level=level, dusk=dusk)
    tor_tex = None
    if T >= T_BREAK:
        tb = Canvas(fc)
        draw_torrents(tb.ctx, cam, st, T, False, level)
        tor_tex = _curtain_tex(fc, cam, T)
        tone = _apply_torrents(tone, tb.rgba(), tor_tex)
        draw_shards(ctx, cam, st, T, fl_amt, near=False, zsplit=zeff)
    bt, ba = _canvas_tone(wc)
    tone = bt + tone * (1 - ba)

    # ---------------------------------------------------------------- fire, effigy
    F = _fire_fields(fc, cam, st, f)
    light = None
    if F is not None and F["flame"] is not None:
        fl = F["flame"]
        light = blur(F["flame_soft"], 55 * s)
        tone = tone + (0.8 - tone) * np.clip(blur(F["flame_soft"], 140 * s) * 0.9, 0, 0.5)
        if F["smoke_a"] is not None:
            a = F["smoke_a"]
            tone = tone * (1 - a) + F["smoke_t"] * a
        w = np.clip((fl - 0.5) / 0.14, 0, 1)
        mant = np.clip((fl - 0.22) / 0.3, 0, 1) * 0.5
        fw = np.maximum(w * w * (3 - 2 * w), mant)
        tone = np.maximum(tone, fw)
    elif F is not None and F["smoke_a"] is not None:
        a = F["smoke_a"]
        tone = tone * (1 - a) + F["smoke_t"] * a
    ec = Canvas(fc)
    E.draw_effigy(ec.ctx, cam, dusk=dusk, char=st["bchar"][f], glow=st["bglow"][f], tone=0.32)
    et, ea = _canvas_tone(ec)
    eu = et / np.maximum(ea, 1e-4)
    if light is not None:
        ea = ea * (1 - 0.7 * fw)
    tone = eu * ea + tone * (1 - ea)
    if light is not None:
        tone = tone + np.clip(fw - tone, 0, 1) * 0.3
    if F is not None and F["steam_a"] is not None:
        a = F["steam_a"]
        tone = tone * (1 - a) + F["steam_t"] * a

    # ---------------------------------------------------------------- front world: embers, torrents, crowd, rain
    fcv = Canvas(fc)
    ctx = fcv.ctx
    if F is not None and F["embers"] is not None:
        x0, y0, x1, y1, br = F["embers"]
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for k in range(len(x0)):
            ctx.set_source_rgba(1, 1, 1, float(min(1.0, br[k] * 1.4)))
            ctx.set_line_width(max(1.2, 2.4 * br[k] * cam.f / 1300.0))
            ctx.move_to(x0[k], y0[k])
            ctx.line_to(x1[k], y1[k])
            ctx.stroke()
    if T >= T_BREAK:
        tfc = Canvas(fc)
        draw_torrents(tfc.ctx, cam, st, T, True, level)
        tone = _apply_torrents(tone, tfc.rgba(), tor_tex)
        draw_shards(ctx, cam, st, T, fl_amt, near=True, zsplit=zeff)
    if surge(T) is None:
        draw_crowd(ctx, cam, people, T, (fsx[0], fsy[0]), fl_amt, panic, front=True, level=level, dusk=dusk)
    draw_rain(ctx, T, smoothstep(T_BREAK + 0.3, T_BREAK + 1.6, T))
    ft, fa = _canvas_tone(fcv)
    tone = ft + tone * (1 - fa)

    # ---------------------------------------------------------------- print
    img = ink.engrave(fc, np.clip(tone, 0, 1))
    if light is not None:
        paper = np.array(ink.PAPER, np.float32)
        k = np.clip(light * 0.45 - 0.05, 0, 0.3)[..., None]
        img = img + (paper - img) * k

    # ---------------------------------------------------------------- the page: the frame recedes into a panel
    M, z, pk = page_matrix(T)
    if pk > 0:
        A = np.array([[z, 0.0, M.x0 * s], [0.0, z, M.y0 * s]], np.float32)
        pagecol = np.array(ink.INK, np.float32) * 0.55 + np.array(ink.PAPER, np.float32) * 0.13
        img = cv2.warpAffine(img, A, (fc.w, fc.h), flags=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(c) for c in pagecol))
        bc = Canvas(fc)
        x0, y0 = M.x0, M.y0
        bc.ctx.rectangle(x0, y0, 1920 * z, 1080 * z)
        bc.ctx.set_source_rgb(0.95, 0.95, 0.94)
        bc.ctx.set_line_width(4.0)
        bc.ctx.stroke()
        img = bc.over(img)

    # ---------------------------------------------------------------- the surge + the spill (printed, unclipped)
    wv = Canvas(fc)
    wctx = wv.ctx
    wctx.save()
    if pk > 0:      # the wave may only break out of the panel downwards (over the bottom border)
        wctx.rectangle(M.x0 + 2, M.y0 + 2, 1920 * z - 4, 3000)
        wctx.clip()
    wctx.transform(M)
    Rw = draw_surge(wctx, cam, T, level) if T >= T_SURGE else None
    if Rw is not None:
        # the burners still ahead of the wave, running from it, black against its foam
        draw_crowd(wctx, cam, people, T, (960.0, -400.0), 0.0, 1.0, front=True, level=level, dusk=1.0,
                   rmin=Rw + 1.0)
    wctx.restore()
    sp = draw_spill(wctx, T, z, pk)
    wet_mask = None
    if Rw is not None or sp is not None:
        lay = wv.rgba()
        pl = ink.print_layer(fc, lay, "engrave")
        img = pl[..., :3] + img * (1 - pl[..., 3:4])
        if pk > 0:
            # the page soaks: wherever water went, plus capillary creep that climbs with time
            wa = cv2.dilate((lay[..., 3] > 0.3).astype(np.float32), np.ones((9, 9), np.uint8))
            creep = blur(wa, 45 * s)
            wet_mask = np.clip(np.maximum(wa, creep * 1.8) * smoothstep(T_PAGE + 0.5, T_PAGE + 1.4, T), 0, 1)
    if wet_mask is not None and hasattr(ink, "wet"):
        img = ink.wet(fc, img, wet_mask, 1.0)

    # ---------------------------------------------------------------- the Flag (spot yellow plate), rising off the panel
    FM = M
    if pk > 0:
        # the Flag is not printed: it rises off the panel, over its top border, and comes toward us
        fx_, fy_ = flag_center(T)
        (px_,), (py_,), _ = cam.project(np.array([[fx_ * 0.9, fy_, 0.0]]))
        qx, qy = M.transform_point(px_, py_)
        e = smoothstep(T_PAGE + 0.2, T_END, T)
        tx_, ty_ = qx + (960.0 - qx) * 0.5 * e, qy + (150.0 - qy) * e
        g = 1.0 + 1.6 * e
        FM = M.multiply(cairo.Matrix(g, 0, 0, g, tx_ - g * qx, ty_ - g * qy))
    flag_rgba = flag_layer(fc, st, cam, T, f, extra=FM)
    img = ink.spot(img, flag_rgba)

    # ---------------------------------------------------------------- lettering (never on the cloth)
    top = Canvas(fc)
    av = comic.cloth_avoid(flag_rgba, fc.s) if hasattr(comic, "cloth_avoid") else None
    for lid, x, y in (("P17", 60, 50), ("P18", 60, 50), ("P19", 60, 50), ("P20", 1100, 60)):
        comic.say(top.ctx, fc, lid, x, y, w=760, avoid=av)
    # onomatopoeia, printed on the hits (never near the cloth: protect() below is the absolute guard)
    for t0, t1, txt, x, y, size, rot in ((T_IGN, T_IGN + 1.3, "FWOOSH!", 560, 820, 150, -0.12),
                                         (T_CRACK, T_CRACK + 1.1, "KRAKK!", 1500, 330, 170, 0.14),
                                         (T_HIT, T_HIT + 1.3, "KSSSHHHH!", 1280, 690, 150, -0.08)):
        if t0 <= T <= t1:
            a = min(1.0, (t1 - T) / 0.25)
            comic.sfx(top.ctx, txt, x, y, size, rot=rot, t=T - t0, alpha=a, depth=12.0, avoid=av)
    if hasattr(comic, "protect"):
        return over(img, comic.protect(top.rgba(), flag_rgba))
    return top.over(img)

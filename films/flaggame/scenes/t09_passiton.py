"""t09 — PASS IT ON (181.17–199.00).  Owner: T09PassItOn.

Page 8 of the tract (t01_page design).  p1 = T08's prayer two-shot (continuous): Crow, "Welcome to the Survey,
sister. Now go..." — glide to p2 (over Summer's shoulder): on PASS (184.43) Crow's hand presses the tract into
her hands, she opens it... the page she holds IS page 8.  185.4 droste_start: we dive into it — an exact
de Smit–Lenstra Print-Gallery Droste of the self-similar page (scenes/t09_droste.py): straight dive x10, the
glide toward p3 bends into Escher's twist, the infinite spiral, and it resolves (191.6 droste_end) flat on
p3: THE NEXT DAY.  Raven: "Ugh, there goes another one of those YELLOW FLAGS." — callback 194.76: it is
Summer carrying the Flag (she waves).  196.3 Raven tosses the tract out of the panel, into the real rain:
T01's physical world; it lands face-down (197.4) and the last frame IS the film's first frame.
"""
import math

import cairo
import cv2
import numpy as np

from vx import W, H, Canvas
from vx import ink, comic
from vx.ease import clamp, smoothstep, smootherstep
from vx.flag import Flag
from vx.foley import export_events, screen_pan
from vx.post import DEFAULT as POST_DEFAULT

from scenes import t01_page as P
from scenes import t09_droste as D
from scenes import t09_art as A

POST = dict(bloom=0.10, bloom_thresh=0.97, grain=0.0, vignette=0.12)

FPS = 24
TWIST_M = 2                      # strands: one turn of the output = S**2 (Escher-strength twist)
BETA_E = D.beta_for(TWIST_M, 1.0)
A_SCR = complex(W / 2, H / 2)

# beats (global seconds)
T_GLIDE12 = (183.55, 184.25)
T_DIVE = (185.40, 186.70)          # droste_start: straight dive x10 about the fixed point (one page deeper)
T_GLIDE23 = (186.70, 187.30)       # straight glide p2 -> p3 on the page one level down
T_BEND = (187.30, 187.90)          # Escher bend (singularity off screen)
T_SPIN_IN = (187.90, 188.80)       # the singularity swings to the centre
T_SPIN_OUT = (189.95, 190.85)      # ... and away again
T_UNBEND = (190.85, 191.45)
T_LIVE3 = 191.60                   # droste_end: p3 comes alive
SPIRAL_LEVELS = 1                  # levels fallen through during the spiral (one full Escher turn of the lattice)
T_TOSS = 196.30
T_CUT = 196.92                     # the tossed tract fills the lens -> T01's physical world (its tract bursts in)
T_END_FRAME0 = 199.0 - 1.0 / 24.0  # our last frame == T01 frame 0


# ============================================================ cameras
def cam_p1():
    return P.frame(P.PANELS["p1"])


def cam_p2():
    return P.frame(P.PANELS["p2"])


def cam_p3():
    return P.frame(P.PANELS["p3"])


def _hermite(t, t0, t1, g0, g1, v0, v1):
    u = (t - t0) / (t1 - t0)
    d = t1 - t0
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * g0 + h10 * d * v0 + h01 * g1 + h11 * d * v1


def dive_gain(T):
    """log zoom about the fixed point from the p2 framing: gentle push, then the x10 dive"""
    g1, v1 = 0.10, 0.22
    if T <= T_GLIDE12[1]:
        return 0.0
    if T <= T_DIVE[0]:
        return _hermite(T, T_GLIDE12[1], T_DIVE[0], 0.0, g1, 0.0, v1)
    if T <= T_DIVE[1]:
        return _hermite(T, T_DIVE[0], T_DIVE[1], g1, D.LN_S, v1, 0.0)
    return D.LN_S


def dive_cam(T):
    """PageCam zooming about the fixed point (its screen position frozen at the p2 framing)"""
    c2 = cam_p2()
    qx, qy = c2.to_screen(*D.FIX)
    k = c2.zoom * math.exp(dive_gain(T))
    return P.PageCam(D.FIX[0] + (W / 2 - qx) / k, D.FIX[1] + (H / 2 - qy) / k, k)


def _cam_p1_open():
    try:
        from scenes import t08_camp as T8
        return T8.camera8(T8.T1)
    except Exception:
        return cam_p1()


CAM_P1_T09 = P.PageCam(P.PANELS["p1"][0] + 540, P.PANELS["p1"][1] + 186, 2.45)

# p3, the next day: land on the reading framing (droste_end), push into the Raven/walkway two-shot, punch in on
# the callback, then close on Raven for the toss.  (keys: global T, panel-local cx, cy, zoom; None = reading)
P3_KEYS = [(191.60, None), (192.25, None), (193.60, (400.0, 196.0, 2.62)), (194.62, (400.0, 196.0, 2.62)),
           (194.95, (420.0, 190.0, 2.80)), (195.80, (410.0, 192.0, 2.80)), (196.22, (300.0, 214.0, 3.25))]


def _p3_cam(T):
    x0, y0 = P.PANELS["p3"][:2]
    keys = [(t, cam_p3() if k is None else P.PageCam(x0 + k[0], y0 + k[1], k[2])) for t, k in P3_KEYS]
    if T <= keys[0][0]:
        return keys[0][1]
    for (ta, a), (tb, b) in zip(keys[:-1], keys[1:]):
        if T <= tb:
            return P.lerp_cam(a, b, smootherstep(ta, tb, T))
    return keys[-1][1]


def live_cam(T):
    if T < T_GLIDE12[0]:
        u = smootherstep(181.20, 182.10, T)                    # settle on the two-shot + balloon, then breathe in
        c = P.lerp_cam(_cam_p1_open(), CAM_P1_T09, u)
        k = 1.0 + 0.035 * smoothstep(182.10, T_GLIDE12[0], T)
        return P.PageCam(c.cx, c.cy, c.zoom * k)
    if T < T_GLIDE12[1]:
        u = smootherstep(T_GLIDE12[0], T_GLIDE12[1], T)
        c = P.lerp_cam(CAM_P1_T09, cam_p2(), u)
        return P.PageCam(c.cx, c.cy, c.zoom * (1.0 - 0.10 * math.sin(math.pi * u)))
    if T < T_DIVE[1] + 1e-9:
        return dive_cam(T)
    c3 = cam_p3()
    if T < T_LIVE3:
        return c3
    return _p3_cam(T)


# ============================================================ the Droste view path (beta, q, C, psi)
def _straight(cam):
    q = complex(*cam.to_screen(*D.FIX))
    Aq = A_SCR - q
    return dict(beta=complex(1, 0), q=q, C=complex(-math.log(cam.zoom), 0.0), psi=math.atan2(Aq.imag, Aq.real))


def _anchored(zA, k, beta, psi_ref=None):
    """view where the screen centre shows page point zA at k px/PU, upright; singularity where it must be"""
    d = complex(*zA) - complex(*D.FIX)
    Aq = k * beta * d
    q = A_SCR - Aq
    psi = math.atan2(Aq.imag, Aq.real)
    if psi_ref is not None:
        psi += 2 * math.pi * round((psi_ref - psi) / (2 * math.pi))
    ld = complex(math.log(abs(d)), math.atan2(d.imag, d.real))
    C = ld - beta * complex(math.log(abs(Aq)), psi)
    return dict(beta=beta, q=q, C=C, psi=psi)


def droste_view(T):
    """the Print-Gallery camera at global time T (valid for T_DIVE[0] <= T < T_LIVE3)"""
    c3 = cam_p3()
    z3 = (c3.cx, c3.cy)
    if T < T_DIVE[1]:
        return _straight(dive_cam(T))
    if T < T_BEND[0]:
        u = smootherstep(T_GLIDE23[0], T_GLIDE23[1], T)
        return _straight(P.lerp_cam(cam_p2(), c3, u))
    if T < T_SPIN_IN[0]:
        kap = smootherstep(T_BEND[0], T_BEND[1], T)
        return _anchored(z3, c3.zoom, D.beta_for(TWIST_M, kap), psi_ref=-math.pi / 2)
    if T < T_UNBEND[0]:
        base = _anchored(z3, c3.zoom, BETA_E, psi_ref=-math.pi / 2)
        ws = smootherstep(T_SPIN_IN[0], T_SPIN_IN[1], T) * (1 - smootherstep(T_SPIN_OUT[0], T_SPIN_OUT[1], T))
        e = smootherstep(T_SPIN_IN[0], T_SPIN_OUT[1], T)
        C = base["C"] - SPIRAL_LEVELS * D.LN_S * e
        q = base["q"] + (A_SCR - base["q"]) * ws
        Aq = A_SCR - q
        psi = math.atan2(Aq.imag, Aq.real) if abs(Aq) > 1.0 else base["psi"]
        return dict(beta=BETA_E, q=q, C=C, psi=psi)
    kap = 1.0 - smootherstep(T_UNBEND[0], T_UNBEND[1], T)
    return _anchored(z3, c3.zoom, D.beta_for(TWIST_M, kap), psi_ref=-math.pi / 2)


# ============================================================ p3: Summer walks by with the Flag
WAVE = (194.80, 195.75)   # 'YELLOW FLAGS' callback: she stops, turns and waves at Raven
WALK_SPEED = 118.0        # PU/s (matches the rig's ground speed at this height, no foot sliding)
X_STOP = P.PANELS["p3"][0] + 540.0
WALK_IN = WAVE[0] - (P.PANELS["p3"][0] + P.PANELS["p3"][2] + 40.0 - X_STOP) / WALK_SPEED   # enters from the right


def summer_walk(T):
    """state of Summer in p3 (absolute PU; None when not in the panel yet)"""
    if T < WALK_IN:
        return None
    t_stop0, t_stop1 = WAVE
    if T < t_stop0:
        x = X_STOP + WALK_SPEED * (t_stop0 - T)
    elif T < t_stop1:
        x = X_STOP - 6.0 * smoothstep(t_stop0 - 0.1, t_stop0 + 0.3, T)
    else:
        x = X_STOP - 6.0 - WALK_SPEED * (T - t_stop1) * smoothstep(t_stop1, t_stop1 + 0.35, T)
    stop = smoothstep(t_stop0 - 0.18, t_stop0 + 0.12, T) * (1 - smoothstep(t_stop1 - 0.1, t_stop1 + 0.25, T))
    pose = {"walk": 1.0 - stop, "stand": stop, "shoulder_flag": 1.0}
    kw = dict(pole_len=A.SUMMER_POLE, speed=1.0)
    if stop > 0.01:
        wv = math.sin((T - t_stop0) * 11.0)
        kw["hand_l"] = (x - 34 - 10 * wv * stop, A.SUMMER_WALK_Y - A.SUMMER_H * (0.62 + 0.42 * stop))
        kw["shape_l"] = "open"
    return dict(x=x, y=A.SUMMER_WALK_Y, h=A.SUMMER_H, pose=pose, facing=-1,
                expr={"cheerful": 1.0}, look=(-1.0 + 0.6 * stop, -0.1 * stop), kw=kw, stop=stop)


def summer_pole_fn(ch, i, f0):
    T = (f0 + i) / FPS
    wk = summer_walk(max(T, WALK_IN))
    a = ch.anchors(wk["x"], wk["y"], wk["h"], wk["pose"], T, facing=wk["facing"], **wk["kw"])
    return a["pole"]


# ============================================================ setup
def _p1_drawer():
    try:
        from scenes import t08_camp as T8
        return P1Wrap(A.make_p1(T8), T8), T8
    except Exception as e:           # T08 not landed yet
        print(f"[t09] t08_camp unavailable ({type(e).__name__}: {e}); using fallback p1")
    return A.P1Fallback(), None


class P1Wrap:
    """T08's p1 drawer (continued) + t09's first P30 balloon"""

    def __init__(self, d, T8):
        self.d, self.T8 = d, T8
        tr = T8.tracks()["p8"]
        self.av = [T8.cloth_union(tr, 181.3, 186.0, pad=10.0)]

    def world(self, ctx, T, rect):
        self.d.world(ctx, T, rect)

    def flags(self, ctx, T, rect):
        self.d.flags(ctx, T, rect)

    def front(self, ctx, T, rect):
        self.d.front(ctx, T, rect)

    def flag_bbox(self, T):
        return self.av[0]

    def letters(self, ctx, fc, T, rect, hold=30.0):
        self.d.letters(ctx, fc, T, rect)
        x0, y0, w, h = rect
        T8 = self.T8
        c = T8._anchors(T8.CROW(), self.d.crow(T))
        mx, my = c["mouth"][0] + x0, c["mouth"][1] + y0
        comic.say(ctx, fc, "P30", x0 + 700, y0 + 66, tokens=(0, 7), tail=(mx - 4, my - 2), w=200,
                  size=P.LETTER["balloon"], avoid=self.av, hold=hold, T=T,
                  bounds=(x0 + 6, y0 + 6, x0 + w - 6, y0 + h - 6))


class _FCStub:
    def __init__(self, tl):
        self.tl, self.T, self.s, self.w, self.h = tl, 0.0, 1.0, W, H
        self.f = self.F = 0
        self.t = 0.0
        self.fps = FPS

def setup(S):
    tl = S.tl
    f0 = S.f0
    n = S.n
    # --- p2: Crow's planted flag behind Summer (lifting in the post-rain breeze)
    bx, by = A.P2_FLAG_BASE
    p2flag = Flag(pole=A.P2_FLAG_POLE, seed=91, side=1).simulate(
        n, lambda i: (bx, by, 0.035), lambda i: (230 + 90 * math.sin(i / 31.0) + 60 * math.sin(i / 7.3), -10),
        warmup=72)
    reading = A.Reading(p2flag, f0)
    # --- p3: Summer's flag (shoulder carry) — pole from the rig's anchors
    from vx.chars import Character
    sch = Character("summer", render="ink")
    fw0 = int(math.floor(WALK_IN * FPS)) - f0
    nw = n - fw0
    sflag = Flag(pole=A.SUMMER_POLE, seed=5, side=1).simulate(
        nw, lambda i: summer_pole_fn(sch, i, f0 + fw0), lambda i: (380, -20), warmup=48)
    nextday = A.NextDay(summer_walk, sflag, f0 + fw0)
    nextday.tl = tl
    p1, T8 = _p1_drawer()
    drawers = {"p1": p1, "p2": reading, "p3": nextday}
    # --- the self-similar page -> log-polar texture (cached)
    level = A.StaticLevel(p1, reading, nextday, _FCStub(tl))
    nu = 1280.0 * max(0.5, S.s)
    tex, _ = D.build_texture(level, nu, cache_dir=S.cache, nb=8, tag=f"t09-level-v1-m{TWIST_M}",
                             log=lambda m: print(m, flush=True))
    D.make_pyramid(tex)
    # --- the alive spot plate inside the spiral: every nested Flag flutters (same Flag, same instant)
    fl = []
    if T8 is not None:
        tr8 = T8.tracks()["p8"]
        fl.append((lambda T, tr=tr8: _track_bbox(tr, T8.fidx(T)),
                   lambda c, T: p1.flags(c, T, P.PANELS["p1"])))
    fl.append((lambda T: _track_bbox(p2flag, reading._fi(T)), lambda c, T: reading.flags(c, T, P.PANELS["p2"])))
    baker = D.FlagBaker(tex, fl)
    # --- T01's physical world for the tail (the loop)
    t01 = None
    try:
        from scenes import t01_tract as T1
        if hasattr(T1, "loop_setup"):
            t01 = (T1, T1.loop_setup(S))
    except Exception as e:
        print(f"[t09] t01 loop unavailable ({type(e).__name__}: {e})")
    tex01 = None
    if t01 is not None:
        try:
            from scenes import t01_pages as PG
            tx = t01[1]["tex"]
            tex01 = dict(cover_static=tx["cover_static"], back=tx["back"], ppu_cover=T1.PPU["cover"],
                         ppu_back=T1.PPU["back"], PG=PG)
            reading.fallback_covers = False
        except Exception as e:
            print(f"[t09] t01 textures unavailable ({type(e).__name__}: {e}); inking stand-in covers")
    st = dict(drawers=drawers, tex=tex, baker=baker, t01=t01, tex01=tex01, T8=T8, p2flag=p2flag, sflag=sflag)
    _exports(S, st)
    return st


def _track_bbox(tr, fi, pad=6.0):
    """cloth + pole bbox (PU) of a FlagTrack at (fractional) frame fi"""
    i = int(round(min(max(fi, 0), tr.n - 1)))
    V = tr.verts[i]
    x, y, a = tr.poles[i]
    L = tr.meta["pole"]
    xs = [float(V[..., 0].min()), float(V[..., 0].max()), x, x + math.sin(a) * L]
    ys = [float(V[..., 1].min()), float(V[..., 1].max()), y, y - math.cos(a) * L]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _exports(S, st):
    ev = [
        dict(t=181.53, kind="balloon", strength=0.6, pan=0.1, desc="P30 balloon 1 pops (Welcome to the Survey)"),
        dict(t=T_GLIDE12[0], kind="glide", strength=0.5, pan=0.0, desc="camera glides p1 -> p2 (page 8)"),
        dict(t=184.02, kind="whoosh", strength=0.35, pan=0.7, desc="Crow's arm reaches in with the tract"),
        dict(t=184.18, kind="balloon", strength=0.55, pan=0.3, desc="P30 balloon 2 pops (...and PASS IT ON.)"),
        dict(t=A.T_PASS, kind="paper", strength=0.9, pan=0.15, desc="PASS: tract pressed into Summer's hands"),
        dict(t=A.T_OPEN0, kind="paper", strength=0.7, pan=0.1, desc="she opens the tract (cover flap swings)"),
        dict(t=A.T_OPEN1, kind="paper", strength=0.5, pan=-0.05, desc="the tract lies open: page 8 | page 9"),
        dict(t=T_DIVE[0], kind="whoosh", strength=0.8, pan=0.0, desc="droste_start: dive into the page she holds"),
        dict(t=186.30, kind="whoosh", strength=0.5, pan=0.0, desc="through the page she holds: the same page again, one level down"),
        dict(t=T_GLIDE23[0], kind="glide", strength=0.4, pan=0.0, desc="glide toward p3 inside the tract"),
        dict(t=T_BEND[0], kind="warp", strength=0.8, pan=0.0, desc="the page bends into Escher's twist"),
        dict(t=T_SPIN_IN[0], kind="whoosh", strength=0.9, pan=0.0, desc="the Print-Gallery spiral: falling in"),
        dict(t=T_SPIN_OUT[0], kind="whoosh", strength=0.6, pan=0.0, desc="the spiral releases us"),
        dict(t=T_UNBEND[0], kind="warp", strength=0.6, pan=0.0, desc="the page unbends, flat on p3"),
        dict(t=T_LIVE3, kind="glide", strength=0.5, pan=0.0, desc="droste_end: THE NEXT DAY (sun, drying mud)"),
        dict(t=192.0, kind="balloon", strength=0.5, pan=-0.6, desc="R02 balloon pops (Raven)"),
        dict(t=194.763, kind="sting", strength=0.8, pan=0.2, desc="callback: it is SUMMER carrying the Flag"),
        dict(t=WAVE[0] + 0.1, kind="wave", strength=0.4, pan=0.2, desc="Summer waves at Raven"),
        dict(t=T_TOSS, kind="whoosh", strength=0.9, pan=-0.5, desc="Raven tosses the tract (flick)"),
        dict(t=196.45, kind="thunder", strength=0.5, pan=0.0, desc="clouds return over the sun; first drops"),
        dict(t=T_CUT, kind="whoosh", strength=0.8, pan=0.0, desc="the tract flies out of the panel into the real rain"),
        dict(t=197.0, kind="whoosh", strength=0.6, pan=0.0, desc="the tract bursts in from the top of the frame, tumbling"),
        dict(t=197.4, kind="splash", strength=1.0, pan=0.0, desc="tract_lands: face-down SPLAT in the mud (T01 world)"),
        dict(t=197.52, kind="paper", strength=0.5, pan=0.0, desc="wet pages slap flat, back page up"),
        dict(t=197.8, kind="paper", strength=0.2, pan=0.0, desc="the soaked tract settles into the loop pose; rain on paper"),
    ]
    # Summer's footsteps on the dry, cracked mud
    ch = st["drawers"]["p3"].summer
    try:
        steps = ch.footsteps(WALK_IN, T_CUT, pose="walk", speed=1.0)
        for t, side in steps:
            wk = summer_walk(t)
            if wk is None or wk["stop"] > 0.3:
                continue
            ev.append(dict(t=float(t), kind="step", strength=0.25, pan=0.3, desc=f"Summer step ({side}) on dry cracked mud"))
    except Exception:
        pass
    export_events(S, "hits", ev)
    try:
        st["p2flag"].export_foley(S, "p2flag", gain=0.5)
        fw0 = int(math.floor(WALK_IN * FPS)) - S.f0
        st["sflag"].export_foley(S, "summerflag", f0=fw0, gain=0.7)
    except Exception as e:
        print(f"[t09] flag foley export failed: {e}")


# ============================================================ printed textures (T01's tract pages) on screen
def _cam_matrix(cam):
    M = cairo.Matrix()
    M.translate(W / 2, H / 2)
    if cam.rot:
        M.rotate(cam.rot)
    M.scale(cam.zoom, cam.zoom)
    M.translate(-cam.cx, -cam.cy)
    return M


def _screen_rect(cam, r):
    x0, y0 = cam.to_screen(r[0], r[1])
    x1, y1 = cam.to_screen(r[0] + r[2], r[1] + r[3])
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def warp_page(fc, tex, ppu, M, clip=None):
    """lay an already-PRINTED page texture (h, w, 4 straight rgba, `ppu` px per page unit) on screen through
    cairo.Matrix M (page PU -> screen design px).  Prefiltered (INTER_AREA) when minified, so the halftone never
    aliases; the print is resampled, never re-screened (its flag stays a plain spot yellow)."""
    k = math.sqrt(abs(M.xx * M.yy - M.xy * M.yx)) * fc.s          # screen px per page unit
    t = tex
    p = float(ppu)
    if p > 1.4 * k and k > 0:
        f = max(1.4 * k / p, 4.0 / min(t.shape[:2]))
        nw, nh = max(2, int(round(t.shape[1] * f))), max(2, int(round(t.shape[0] * f)))
        t = cv2.resize(t, (nw, nh), interpolation=cv2.INTER_AREA)
        p = p * nw / tex.shape[1]
    pm = np.concatenate([t[..., :3] * t[..., 3:4], t[..., 3:4]], axis=-1).astype(np.float32)
    s = fc.s
    A = np.array([[M.xx * s / p, M.xy * s / p, M.x0 * s], [M.yx * s / p, M.yy * s / p, M.y0 * s]], np.float64)
    A[:, 2] += A[:, 0] * 0.5 + A[:, 1] * 0.5 - 0.5                   # pixel-centre conventions
    out = cv2.warpAffine(pm, A, (fc.w, fc.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if clip is not None:
        x0, y0, x1, y1 = [int(round(v * s)) for v in clip]
        m = np.zeros((fc.h, fc.w, 1), np.float32)
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1.0
        out *= m
    return out


_COV = {}


def _covers_fn(st):
    """key -> value accessor for the tract's printed cover (with its flag alive at T) / back page"""
    t = st.get("tex01")
    if t is None:
        return None

    def get(key, T):
        if key == "cover":
            k = round(T * FPS)
            if k not in _COV:
                _COV.clear()
                pg = t["PG"]
                _COV[k] = pg.with_flag(t["cover_static"], pg.cover_flag(t["ppu_cover"], T))
            return _COV[k]
        return t[key]
    return get


# ============================================================ live page compositor (render_page + window)
def render_live(fc, cam, T, drawers, window=None, overlay=None, covers=None):
    """t01_page.render_page pipeline with t09's window: world -> print -> spot -> front -> WINDOW (the page
    she holds, sampled from the Print-Gallery texture) -> holders (thumbs/cover/Crow's hand) -> stock/keylines
    -> letters."""
    items = [(P.PANELS[k], d) for k, d in drawers.items()]
    vis = [(r, d) for r, d in items if P._visible(cam, r)]
    kw = cam.ink_kw()
    world = Canvas(fc, bg=(1, 1, 1))
    for r, d in vis:
        c = world.ctx
        c.save()
        cam.apply(c)
        c.rectangle(*r)
        c.clip()
        d.world(c, T, r)
        c.restore()
    img = ink.tract(fc, world.rgb(), **kw)
    fl = Canvas(fc)
    for r, d in vis:
        if hasattr(d, "flags"):
            c = fl.ctx
            c.save()
            cam.apply(c)
            c.rectangle(*r)
            c.clip()
            d.flags(c, T, r)
            c.restore()
    flag_rgba = fl.rgba()
    img = ink.spot(img, flag_rgba)
    fr = Canvas(fc)
    for r, d in vis:
        if hasattr(d, "front"):
            c = fr.ctx
            c.save()
            cam.apply(c)
            c.rectangle(*r)
            c.clip()
            d.front(c, T, r)
            c.restore()
    lay = fr.rgba()
    if lay[..., 3].max() > 0:
        front = ink.print_layer(fc, lay, "tract", **kw)
        img = front[..., :3] + img * (1.0 - front[..., 3:4])
    if window is not None:
        img = window(img)
    if covers is not None:
        for r, d in vis:
            if hasattr(d, "cover_quads"):
                for M in d.cover_quads(T):
                    Ms = cairo.Matrix(M.xx, M.yx, M.xy, M.yy, M.x0, M.y0).multiply(_cam_matrix(cam))
                    lay = warp_page(fc, covers("cover", T), covers("ppu_cover", T), Ms, clip=_screen_rect(cam, r))
                    img = lay[..., :3] + img * (1.0 - lay[..., 3:4])
    hd = Canvas(fc)
    for r, d in vis:
        if hasattr(d, "holders"):
            c = hd.ctx
            c.save()
            cam.apply(c)
            c.rectangle(*r)
            c.clip()
            d.holders(c, T, r)
            c.restore()
    hl = hd.rgba()
    if hl[..., 3].max() > 0:
        hp = ink.print_layer(fc, hl, "tract", **kw)
        img = hp[..., :3] + img * (1.0 - hp[..., 3:4])
    top = Canvas(fc)
    top.ctx.save()
    cam.apply(top.ctx)
    P.draw_page_frame(top.ctx, A.PAGE_NO, [r for r, _ in items])
    top.ctx.restore()
    for r, d in vis:
        if hasattr(d, "letters"):
            top.ctx.save()
            cam.apply(top.ctx)
            d.letters(top.ctx, fc, T, r)
            top.ctx.restore()
    if overlay is not None:
        overlay(top.ctx)
    lt = top.rgba()
    lt = comic.protect(lt, flag_rgba) if hasattr(comic, "protect") else lt
    return lt[..., :3] + img * (1.0 - lt[..., 3:4])


def _screen_box(cam, x0, y0, x1, y1, fc):
    sx0, sy0 = cam.to_screen(x0, y0)
    sx1, sy1 = cam.to_screen(x1, y1)
    return sx0 * fc.s, sy0 * fc.s, sx1 * fc.s, sy1 * fc.s


def window_fn(fc, cam, T, st):
    """composite the page she holds (straight Droste sample) into the visible part of page 8"""
    ext = A.window_extent(T)
    if ext is None:
        return None
    wx0, wx1 = ext
    y0, y1 = A.BOOK_Y0, A.BOOK_Y1

    def fn(img):
        v = _straight(cam)
        px0, py0, px1, py1 = _screen_box(cam, wx0, y0, wx1, y1, fc)
        ix0, iy0 = max(0, int(math.floor(px0))), max(0, int(math.floor(py0)))
        ix1, iy1 = min(fc.w, int(math.ceil(px1))), min(fc.h, int(math.ceil(py1)))
        if ix1 <= ix0 or iy1 <= iy0:
            return img
        # page 8 may be foreshortened by the opening flap: stretch x back to the full page
        full0 = D.WIN_X
        sc = (wx1 - wx0) / D.WIN_W
        box = (ix0, iy0, ix1, iy1)
        smp = D.droste_box(fc, _tex(st, T), v["q"], v["beta"], v["C"], v["psi"], box,
                           xmap=None if sc > 0.999 else (cam, wx0, wx1, full0))
        # antialiased coverage of the page rect
        xs = (np.arange(ix0, ix1) + 0.5)
        ys = (np.arange(iy0, iy1) + 0.5)
        cx = np.clip(np.minimum(xs - px0, px1 - xs) + 0.5, 0, 1)
        cy = np.clip(np.minimum(ys - py0, py1 - ys) + 0.5, 0, 1)
        a = (cy[:, None] * cx[None, :])[..., None].astype(np.float32)
        out = img.copy()
        out[iy0:iy1, ix0:ix1] = smp * a + img[iy0:iy1, ix0:ix1] * (1 - a)
        return out

    return fn


# ============================================================ the toss: the tract flies out of the panel
def toss_state(T, st, cam):
    """(screen x, y, scale px per PU, rotation, flip) of the flying tract, or None"""
    if T < T_TOSS + 0.02:
        return None
    nd = st["drawers"]["p3"]
    hx, hy = nd.raven_hand_xy(T_TOSS + 0.02)
    sx, sy = cam.to_screen(hx, hy)
    u = clamp((T - T_TOSS) / (T_CUT - T_TOSS), 0, 1)
    e = u ** 2.4
    k = cam.zoom * (0.16 + 30.0 * e)          # grows toward the lens
    x = sx + (W * 0.56 - sx) * smoothstep(0, 1, u) + 30 * math.sin(u * 5)
    y = sy - 190 * math.sin(math.pi * min(u * 1.15, 1)) + (H * 0.46 - sy) * u ** 1.6
    rot = -0.5 + 4.6 * u
    flip = math.cos(u * math.pi * 3.0 + 0.3)
    return x, y, k, rot, flip


def flying_tract_layer(fc, ts, st, T):
    """the tossed tract: THE printed tract (T01's cover — its flag alive — and back page, the one that lands
    face-down in the mud), tumbling toward the lens; the print rides with it, never re-screened"""
    x, y, k, rot, flip = ts
    w, h = D.WIN_W, D.WIN_H
    M = cairo.Matrix()
    M.translate(x, y)
    M.rotate(rot)
    M.scale(k * max(abs(flip), 0.03), k)
    M.translate(-w / 2, -h / 2)
    cov = _covers_fn(st)
    if cov is not None:
        Mt = cairo.Matrix(w / P.PAGE_W, 0, 0, w / P.PAGE_W, 0, 0).multiply(M)
        if flip >= 0:
            return warp_page(fc, cov("cover", T), cov("ppu_cover", T), Mt)
        return warp_page(fc, cov("back", T), cov("ppu_back", T), Mt)
    cv = Canvas(fc)
    c = cv.ctx
    c.save()
    c.transform(M)
    if flip >= 0:
        A.draw_cover(c, 0, 0, w, h)
    else:
        A.draw_page9(c, 0, 0, w, h)
    c.restore()
    lay = cv.rgba()
    Mp = cairo.Matrix(0.1, 0, 0, 0.1, 0, 0).multiply(M)      # the tract's own page units (1000 PU wide)
    try:
        return ink.print_layer(fc, lay, "tract", space="page", matrix=Mp, lpi=P.TRACT_LPI, angle=P.TRACT_ANGLE)
    except TypeError:
        return lay


def toss_overlay(T, ts):
    """speed lines converging on the flying tract + the FWIP! (drawn on the lettering layer, protected)"""
    def fn(ctx):
        if ts is None:
            return
        x, y, k, rot, flip = ts
        a = smoothstep(T_TOSS, T_TOSS + 0.08, T) * (1 - smoothstep(T_CUT - 0.12, T_CUT, T))
        if a > 0:
            ctx.save()
            ctx.push_group()
            comic.speed_lines(ctx, (0, 0, W, H), focus=(x, y), t=T, n=70, inner=max(60.0, k * 60.0))
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(0.85 * a)
            ctx.restore()
        comic.sfx(ctx, "FWIP!", x - 140, y - 130, 92, rot=-0.2, t=T - T_TOSS, depth=6)
    return fn


# ============================================================ render
_TEXC = {}


def _tex(st, T):
    """the Print-Gallery texture at time T (alive flags re-laid), cached for the current frame"""
    k = round(T * FPS, 3)
    if k not in _TEXC:
        _TEXC.clear()
        _TEXC[k] = st["baker"].frame(T)
    return _TEXC[k]


def _droste_frame(fc, st, T):
    v = droste_view(T)
    return D.droste(fc, _tex(st, T), v["q"], v["beta"], v["C"], v["psi"])


def _window_covers_screen(cam):
    x0, y0 = cam.to_screen(D.WIN_X, D.WIN_Y)
    x1, y1 = cam.to_screen(D.WIN_X + D.WIN_W, D.WIN_Y + D.WIN_H)
    return x0 <= -2 and y0 <= -2 and x1 >= W + 2 and y1 >= H + 2


def render(fc, st):
    T = fc.T
    dr = st["drawers"]
    if T >= T_CUT and st["t01"] is not None:
        T1, s1 = st["t01"]
        # exact frame arithmetic: our last frame (global F 4775) is T01's t = 0.0 bit-for-bit
        return T1.render_at(fc, (fc.F - (fc.S.f0 + fc.S.n - 1)) / fc.fps, s1, landing=True)
    if T < T_DIVE[1]:
        cam = live_cam(T)
        if T >= T_DIVE[0] and _window_covers_screen(cam):
            return _droste_frame(fc, st, T)
        return render_live(fc, cam, T, dr, window=window_fn(fc, cam, T, st), covers=_covers_fn(st))
    if T < T_LIVE3:
        img = _droste_frame(fc, st, T)
        if T > T_LIVE3 - 3.5 / FPS:             # hide the texture -> vector hand-off under a 3-frame blend
            a = clamp((T - (T_LIVE3 - 3.5 / FPS)) / (3.5 / FPS), 0, 1)
            live = render_live(fc, cam_p3(), T_LIVE3, dr)
            img = img * (1 - a) + live * a
        return img
    cam = live_cam(T)
    ts = toss_state(T, st, cam)
    img = render_live(fc, cam, T, dr, overlay=toss_overlay(T, ts) if ts is not None else None)
    if ts is not None:
        lay = flying_tract_layer(fc, ts, st, T)
        img = lay[..., :3] + img * (1.0 - lay[..., 3:4])
    return img


def post(fc, st):
    if fc.T >= T_CUT and st.get("t01") is not None:
        T1, s1 = st["t01"]
        p = {k: POST_DEFAULT[k] for k in POST}
        p.update(getattr(T1, "POST", {}))
        if hasattr(T1, "post"):
            p.update(T1.post(fc, s1) or {})
        return p
    return {}

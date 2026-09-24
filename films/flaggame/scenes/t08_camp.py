"""t08 CAMP — layout, camera, panel drawers and the Crow flag sims for t08 (Blaspheme, 159.83-181.17);
shared with t07 (hand-in) and t09 (continues page 8 p1).  Owner: T08Blaspheme.

The tract pages (scenes.t01_page geometry, page units PU):
  PAGE 7  p1   the camp in the rain, Crow preaching; BLASPHEME lightning -> hard black & white
          p2a  Summer's horror close-up (lightning-lit, hard black & white)
          p2b  Summer drops to her knees in the mud: "What must I do?"
          p3   "Repeat after me." — Crow drives his flag into the mud (the hymn begins)
  -- page turn 7 -> 8 (TURN0..TURN1) --
  PAGE 8  p1   the lorem-ipsum prayer: both kneel, the flag planted between them; Amen: the rain stops,
               the clouds part, a halftone sunbeam falls on Summer. (t09 continues this panel and owns
               page 8 p2/p3.)

Public (for t07 / t09)
----------------------
camera(T) -> (page_no, PageCam)            my camera (page units -> design px); at rest at T >= 181.17
DRAWERS[(page, pid)] -> drawer             render_page contract: world(ctx, T, rect), flags(ctx, T, rect),
                                           front(ctx, T, rect), letters(ctx, fc, T, rect); panel-alive clock
P8P1 = DRAWERS[(8, "p1")]                  the prayer panel; P8P1.hold = T after which the characters hold
                                           their pose (t09 may set attributes / subclass).
draw_camp(ctx, T) / draw_rain(ctx, T)      t07 hand-in: page 7 p1 grey world (no Crow, no flags) in DESIGN
                                           units at my 159.83 camera (transform-safe).
CROW_T07 -> dict(x, y, h, facing, pose, kw) Crow at 159.83 in design units;  FLAG_T07 -> pole/flag spec.
tracks() -> {name: FlagTrack}              the hero flag sims (global frame index - F0), built once, cached.
crow_flag_track(extra_pole_fn=None, extra_wind_fn=None) -> FlagTrack   page 8 flag, frames 3836..4775;
                                           identical to mine up to 181.17 when the extras only change later frames.
print_kw(cam) -> ink.tract kwargs;   POST
"""
import math
import os

import cairo
import numpy as np

from vx import W, H, FPS, CACHE
from vx.ease import clamp, lerp, seg, ease_in_out, ease_out, ease_in, smoothstep, noise1, hash01, spring
from vx.flag import Flag, FlagTrack
from vx.chars import Character
from vx import comic

from scenes import t01_page as P
from scenes import t08_art as A

# ============================================================ time
T0 = 3836 / 24.0              # 159.8333  scene start (global frame F0)
T1 = 4348 / 24.0              # 181.1667  scene end
F0 = 3836
N_SCENE = 512
N_EXT = 4776 - F0             # sims run to the end of the film (t09 continues the page-8 flag)
STRIKE = 166.223              # 'blaspheme' cue: lightning
KNEEL = 167.81                # knees hit the mud
HYMN = 169.09
PLANT = 169.34                # Crow drives the pole into the mud ('after')
AMEN = 180.13
TURN0, TURN1 = 170.26, 170.64  # page turn 7 -> 8 (curl u 0..1)
POST = dict(bloom=0.10, bloom_thresh=0.97, bloom_radius=26.0, grain=0.0, vignette=0.12)

# ============================================================ page geometry (PU)
R1 = P.PANELS["p1"]
R2A, R2B = P.split(P.PANELS["p2"], 0.42)
R3 = P.PANELS["p3"]
R8 = P.PANELS["p1"]
PAGE7_RECTS = [R1, R2A, R2B, R3]
LET = P.LETTER


def print_kw(cam):
    return cam.ink_kw()


# ============================================================ camera
def _cam(cx, cy, z):
    return P.PageCam(cx, cy, z)


def _chain(T, keys):
    """keys = [(t, PageCam), ...] with eased segments between successive keys (holds when equal)."""
    if T <= keys[0][0]:
        return keys[0][1]
    for (ta, a), (tb, b) in zip(keys[:-1], keys[1:]):
        if T <= tb:
            u = ease_in_out(clamp((T - ta) / max(1e-6, tb - ta)), 3)
            return P.lerp_cam(a, b, u)
    return keys[-1][1]


def _glide(a, b, u, lift=0.10):
    """panel-to-panel glide with a small zoom-out bump mid-way (feels like the eye lifting off the page)."""
    c = P.lerp_cam(a, b, u)
    k = 1.0 - lift * math.sin(math.pi * u)
    return P.PageCam(c.cx, c.cy, c.zoom * k)


# framings
_Z_READ = P.frame(R1).zoom                       # 2.054
CAM_OPEN = _cam(R1[0] + 548, R1[1] + 176, 1080.0 / R1[3] * 1.02)      # full-bleed on Crow (t07 lands here)
CAM_P1 = _cam(R1[0] + 433, R1[1] + 176, 2.25)
CAM_P1_PUSH = _cam(R1[0] + 482, R1[1] + 170, 2.80)
CAM_P1_HIT = _cam(R1[0] + 470, R1[1] + 166, 3.00)
CAM_P2A = P.frame(R2A, margin=0.05)
CAM_P2A_PUSH = _cam(CAM_P2A.cx + 6, CAM_P2A.cy - 8, CAM_P2A.zoom * 1.10)
CAM_P2B = P.frame(R2B, margin=0.05)
CAM_P2B_PUSH = _cam(CAM_P2B.cx - 20, CAM_P2B.cy + 6, CAM_P2B.zoom * 1.10)
CAM_P3 = _cam(R3[0] + R3[2] / 2, R3[1] + R3[3] / 2, _Z_READ * 1.02)
CAM_P3_PUSH = _cam(R3[0] + 420, R3[1] + 176, 2.45)
CAM_P8 = _cam(R8[0] + R8[2] / 2, R8[1] + R8[3] / 2, _Z_READ * 1.0)
CAM_P8_PUSH = _cam(R8[0] + 505, R8[1] + 176, 3.2)
CAM_P8_AMEN = _cam(R8[0] + 470, R8[1] + 172, 2.72)


def _shake(T, t0, amp, dur, seed):
    if T < t0 or T > t0 + dur:
        return 0.0, 0.0
    k = (1 - (T - t0) / dur) ** 2
    return (amp * k * noise1((T - t0) * 38, seed), amp * k * noise1((T - t0) * 38, seed + 9))


def _rot(c, r):
    return P.PageCam(c.cx, c.cy, c.zoom, r)


def camera(T):
    """(page_no, PageCam) at global time T.  Page 7 until the turn is complete (TURN1)."""
    if T < 166.60:
        c = _chain(T, [(T0, CAM_OPEN), (160.05, CAM_OPEN), (163.95, CAM_P1), (164.45, CAM_P1),
                       (166.10, CAM_P1_PUSH)])
        rot = -0.022 * seg(T, 164.45, 166.10)            # the frame slowly lists as he warns her
        if T >= STRIKE - 0.02:
            # impact: a hard reframe punch + the dutch angle snaps the other way + shake
            u = clamp((T - (STRIKE - 0.02)) / 0.30)
            c = P.lerp_cam(CAM_P1_PUSH, CAM_P1_HIT, 1 - (1 - u) ** 4)
            rot = lerp(-0.022, 0.045, 1 - (1 - clamp(u * 3)) ** 3) - 0.012 * seg(T, STRIKE + 0.1, 166.6)
        dx, dy = _shake(T, STRIKE - 0.02, 9.0, 0.9, 3)
        return 7, P.PageCam(c.cx + dx / c.zoom, c.cy + dy / c.zoom, c.zoom, rot)
    if T < 166.86:
        u = ease_in_out(clamp((T - 166.60) / 0.26), 3)
        c = _glide(_rot(CAM_P1_HIT, 0.033), CAM_P2A, u, 0.12)
        c = P.PageCam(c.cx, c.cy, c.zoom, lerp(0.033, 0.0, u))
    elif T < 167.30:
        c = P.lerp_cam(CAM_P2A, CAM_P2A_PUSH, ease_out(clamp((T - 166.86) / 0.44), 2))
    elif T < 167.56:
        c = _glide(CAM_P2A_PUSH, CAM_P2B, ease_in_out(clamp((T - 167.30) / 0.26), 3), 0.10)
    elif T < 168.72:
        c = P.lerp_cam(CAM_P2B, CAM_P2B_PUSH, ease_in_out(clamp((T - 167.56) / 1.16), 2))
    elif T < 169.09:
        c = _glide(CAM_P2B_PUSH, CAM_P3, ease_in_out(clamp((T - 168.72) / 0.37), 3), 0.12)
    else:
        c = P.lerp_cam(CAM_P3, CAM_P3_PUSH, ease_in_out(clamp((T - 169.09) / 1.4), 2))
    dx, dy = _shake(T, STRIKE - 0.02, 9.0, 0.9, 3)
    dx2, dy2 = _shake(T, KNEEL, 4.0, 0.35, 5)
    dx3, dy3 = _shake(T, PLANT, 3.0, 0.3, 7)
    sx, sy = dx + dx2 + dx3, dy + dy2 + dy3
    if T < TURN1:
        return 7, P.PageCam(c.cx + sx / c.zoom, c.cy + sy / c.zoom, c.zoom, c.rot)
    return 8, camera8(T)


_PRAYER = (("P27", 1), ("S07", -1), ("P28", 1), ("S08", -1), ("P29", 1), ("S09", -1))


def _speaker_drift(T):
    """-1 (Summer) .. +1 (Crow): the camera leans gently toward whoever is praying aloud."""
    from vx.timeline import get_timeline
    tl = get_timeline()
    v = 0.0
    for lid, sgn in _PRAYER:
        l = tl.line(lid)
        v += sgn * seg(T, l["start"] - 0.45, l["start"] + 0.35) * (1 - seg(T, l["end"] + 0.05, l["end"] + 0.7))
    return clamp(v, -1.0, 1.0)


def camera8(T):
    """page-8 camera (valid for any T; under the turning leaf it is already at the page-8 framing)."""
    c = _chain(T, [(TURN0, CAM_P8), (170.9, CAM_P8), (179.95, CAM_P8_PUSH), (T1, CAM_P8_AMEN)])
    k = seg(T, 170.4, 172.0) * (1 - 0.6 * seg(T, AMEN - 0.2, AMEN + 0.6))
    return P.PageCam(c.cx + 42.0 * k * _speaker_drift(T), c.cy, c.zoom)


# ============================================================ characters
_CH = {}


def ch(kind, seed=0, **style):
    k = (kind, seed, tuple(sorted(style.items())))
    if k not in _CH:
        _CH[k] = Character(kind, seed=seed, **style)
    return _CH[k]


def SUMMER():
    return ch("summer", 1, render="ink", wet=True, mud=0.15)


def CROW():
    return ch("crow", 2, render="ink", wet=True, mud=0.3, face_mask=0.4)   # mask pulled under the chin (t07 unmask)


def RAVEN():
    return ch("raven", 3, render="ink", wet=False)


def mouth(spk, T):
    try:
        from vx.timeline import get_timeline
        return get_timeline().mouth(spk, T)
    except Exception:
        return 0.0


def _draw(c, ctx, a):
    a = dict(a)
    x, y, h, pose = a.pop("x"), a.pop("y"), a.pop("h"), a.pop("pose")
    return c.draw(ctx, x, y, h, pose, **a)


def _anchors(c, a):
    a = dict(a)
    x, y, h, pose = a.pop("x"), a.pop("y"), a.pop("h"), a.pop("pose")
    for k in ("alpha", "layer"):
        a.pop(k, None)
    return c.anchors(x, y, h, pose, **a)


# ============================================================ flags
_TRACKS = {}
FLAG_VER = 7


def _wind_p1(i):
    T = T0 + i / FPS
    w = 250.0 + 60.0 * noise1(T * 0.6, 11)
    w += 950.0 * math.exp(-max(0.0, T - STRIKE) * 2.2) * (T >= STRIKE - 0.05)
    return (w, -20.0)


P1_POLE = (R1[0] + 660.0, R1[1] + 520.0)            # page units: planted butt (below the panel edge)
P1_ANG = 0.05
P1_L = 330.0 * 96.0 / 68.0


def _pole_p1(i):
    T = T0 + i / FPS
    a = P1_ANG + 0.012 * math.sin(T * 1.3) + 0.03 * spring(T - (STRIKE - 0.05), 3.0, 5.0) * (T > STRIKE)
    return (P1_POLE[0], P1_POLE[1], a)


P3_POLE = (R3[0] + 640.0, R3[1] + 690.0)
P3_L = 600.0 * 96.0 / 68.0
P8_POLE = (R8[0] + 512.0, R8[1] + 384.0)
P8_L = 525.0 * 96.0 / 68.0 * 0.5
P2B_POLE = (R2B[0] + 398.0, R2B[1] + 480.0)
P2B_L = 440.0 * 96.0 / 68.0


def _pole_p3(i):
    T = T0 + i / FPS
    # held above the mud, then driven in on 'after' (PLANT): down 26 PU with a jolt
    u = seg(T, PLANT - 0.16, PLANT, ease_in)
    dy = -44.0 * (1 - u)
    a = 0.12 - 0.04 * u + 0.012 * math.sin(T * 1.1) * (1 - u)
    return (P3_POLE[0] - 12 * (1 - u), P3_POLE[1] + dy, a)


def _wind_p3(i):
    T = T0 + i / FPS
    return (230.0 + 50.0 * noise1(T * 0.5, 21), -10.0)


def _pole_p8(i):
    return (P8_POLE[0], P8_POLE[1], 0.035)


def _wind_p8(i):
    T = T0 + i / FPS
    a = seg(T, AMEN, AMEN + 1.6, ease_in_out)
    return (110.0 + 30.0 * noise1(T * 0.5, 31) + 380.0 * a, -10.0 - 25.0 * a)


def _pole_p2b(i):
    T = T0 + i / FPS
    return (P2B_POLE[0], P2B_POLE[1], 0.22 + 0.01 * math.sin(T * 1.2))


def _wind_p2b(i):
    T = T0 + i / FPS
    return (240.0 + 50.0 * noise1(T * 0.5, 41), -10.0)


def _sim(name, pole_fn, wind_fn, L, seed, n=N_SCENE):
    path = CACHE / "t08" / f"flag_{name}_v{FLAG_VER}.npz"
    if path.exists():
        return FlagTrack.load(path)
    tr = Flag(pole=L, seed=seed, side=1, turbulence=0.8, flutter=0.8).simulate(n, pole_fn, wind_fn, warmup=72)
    path.parent.mkdir(parents=True, exist_ok=True)
    tr.save(path)
    return tr


def _blend_t07(tr, n=24):
    """zero-pop hand-in: ease the first n frames of my p1 cloth from t07's walker flag (same planted pole
    from 158.25; cache/t07/crow_flag.npz, index 0 = global frame 3516)."""
    path = CACHE / "t07" / "crow_flag.npz"
    if not path.exists():
        return tr
    try:
        t7 = FlagTrack.load(path)
    except Exception:
        return tr
    k0 = F0 - 3516
    if k0 >= t7.n or t7.verts.shape[1:] != tr.verts.shape[1:]:
        return tr
    V = tr.verts.copy()
    for i in range(n):
        j = k0 + i
        if j >= t7.n:
            break
        w = ease_in_out(i / float(n), 2)
        V[i] = _t07_to_page(t7.verts[j]) * (1 - w) + V[i] * w
    return FlagTrack(V, tr.poles, tr.energy, tr.snap, tr.meta, tr.flap_hz)


def _t07_to_page(Vd):
    """t07 verts are in design px at my opening camera -> page units"""
    c = CAM_OPEN
    P_ = Vd.copy()
    P_[..., 0] = (Vd[..., 0] - W / 2) / c.zoom + c.cx
    P_[..., 1] = (Vd[..., 1] - H / 2) / c.zoom + c.cy
    P_[..., 2] = Vd[..., 2] / c.zoom
    return P_


def tracks():
    if not _TRACKS:
        _TRACKS["p1"] = _blend_t07(_sim("p1", _pole_p1, _wind_p1, P1_L, 81))
        _TRACKS["p2b"] = _sim("p2b", _pole_p2b, _wind_p2b, P2B_L, 82)
        _TRACKS["p3"] = _sim("p3", _pole_p3, _wind_p3, P3_L, 83)
        _TRACKS["p8"] = crow_flag_track()
    return _TRACKS


def crow_flag_track(extra_pole_fn=None, extra_wind_fn=None):
    """page 8 flag (planted between them), sim frames 0..N_EXT-1 = global frames F0..4775.
    extra_*_fn(i) override frames with global T > T1 (t09's action after my scene)."""
    if extra_pole_fn is None and extra_wind_fn is None:
        return _sim("p8", _pole_p8, _wind_p8, P8_L, 84, n=N_EXT)

    def pf(i):
        return extra_pole_fn(i) if (extra_pole_fn and T0 + i / FPS > T1) else _pole_p8(i)

    def wf(i):
        return extra_wind_fn(i) if (extra_wind_fn and T0 + i / FPS > T1) else _wind_p8(i)
    return Flag(pole=P8_L, seed=84, side=1, turbulence=0.8, flutter=0.8).simulate(N_EXT, pf, wf, warmup=72)


def fidx(T):
    return clamp((T - T0) * FPS, 0, N_EXT - 1)


def cloth_bbox(tr, T, pad=8.0):
    i = int(round(min(fidx(T), tr.n - 1)))
    V = tr.verts[i]
    return (float(V[..., 0].min()) - pad, float(V[..., 1].min()) - pad,
            float(V[..., 0].max()) + pad, float(V[..., 1].max()) + pad)


def cloth_union(tr, Ta, Tb, pad=8.0):
    ia, ib = int(fidx(Ta)), int(fidx(Tb)) + 1
    ib = min(ib, tr.n)
    V = tr.verts[ia:ib]
    return (float(V[..., 0].min()) - pad, float(V[..., 1].min()) - pad,
            float(V[..., 0].max()) + pad, float(V[..., 1].max()) + pad)


def grip(tr, T, frac):
    """point on the pole at `frac` of its length from the butt (page units)."""
    i = fidx(T)
    _, (x, y, a) = tr._frame(min(i, tr.n - 1))
    L = tr.meta["pole"]
    return (float(x + math.sin(a) * L * frac), float(y - math.cos(a) * L * frac))


# ============================================================ lightning state
def strike_state(T):
    """(bw, flash, neg, bolt_alpha, bolt_grow): hard B&W after the strike, flash flicker, negative frame."""
    k = int(math.floor((T - STRIKE) * FPS + 0.5))      # frames since the strike frame (0 = strike)
    if k < 0:
        return 0.0, 0.0, False, 0.0, 0.0
    flicker = {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.6, 4: 0.0, 5: 0.0, 6: 0.35, 7: 0.0, 11: 0.55, 12: 0.0}
    flash = flicker.get(k, 0.0)
    neg = k == 0
    bolt = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.9, 5: 0.8, 6: 1.0, 7: 0.6, 8: 0.35, 9: 0.2, 11: 0.9, 12: 0.5,
            13: 0.2}.get(k, 0.0)
    return 1.0, flash, neg, bolt, 1.0


# ============================================================ drawers
class Drawer:
    rect = None
    t_on, t_off = -1e9, 1e9

    def tp(self, T):
        return clamp(T, self.t_on, self.t_off)

    def local(self, ctx, rect):
        ctx.translate(rect[0], rect[1])

    # subclasses: _world(ctx, T), _flags(ctx, T), _front(ctx, T), _letters(ctx, fc, T)
    def world(self, ctx, T, rect):
        ctx.save()
        self.local(ctx, rect)
        self._world(ctx, self.tp(T))
        ctx.restore()

    def flags(self, ctx, T, rect):
        if not hasattr(self, "_flags"):
            return
        ctx.save()
        x, y, w, h = P.inset(rect, P.BORDER / 2 + 0.5)
        ctx.rectangle(x, y, w, h)
        ctx.clip()
        self._flags(ctx, self.tp(T))
        ctx.restore()

    def front(self, ctx, T, rect):
        if not hasattr(self, "_front"):
            return
        ctx.save()
        self.local(ctx, rect)
        self._front(ctx, self.tp(T))
        ctx.restore()

    def letters(self, ctx, fc, T, rect):
        if not hasattr(self, "_letters"):
            return
        ctx.save()
        self._letters(ctx, fc, self.tp(T) if T > self.t_off else T, rect)
        ctx.restore()


def _view_bounds(lid, page, pad=14.0, vt=None):
    """page rect visible at the end of line `lid` (+hold) — the tightest framing while it is on screen."""
    from vx.timeline import get_timeline
    l = get_timeline().line(lid)
    te = l["end"] + 0.3 if vt is None else vt
    cam = camera8(te) if page == 8 else camera(te)[1]
    x0, y0, x1, y1 = cam.view_rect()
    return (x0 + pad, y0 + pad, x1 - pad, y1 - pad)


def _say(ctx, fc, lid, x, y, T, *, tail=None, w=None, kind=None, avoid=None, rect=None, hold=0.5, page=7,
         vt=None, **kw):
    kind = kind or fc.tl.line(lid).get("meta", {}).get("letter", "balloon")
    size = kw.pop("size", LET.get(kind, 15.0))
    bounds = None
    if rect is not None:
        bounds = (rect[0] - 30, rect[1] - 30, rect[0] + rect[2] + 30, rect[1] + rect[3] + 30)
        vb = _view_bounds(lid, page, vt=vt)
        bounds = (max(bounds[0], vb[0]), max(bounds[1], vb[1]), min(bounds[2], vb[2]), min(bounds[3], vb[3]))
    return comic.say(ctx, fc, lid, x, y, w=w, tail=tail, kind=kind, size=size, avoid=avoid, bounds=bounds,
                     hold=hold, T=T, **kw)


def _trav(name, extra=()):
    """avoid callable for comic.say: the flag's cloth hull + pole over the balloon's lifetime (+ static boxes)"""
    tr = tracks()[name]
    ex = list(extra)
    return lambda TT: comic.track_avoid(tr, fidx(TT)) + ex


def _tip_above(a, gap=8.0):
    """balloon tail tip for a balloon ABOVE the speaker: just over the front of the head (never across the face)"""
    return (a["mouth"][0], a["top"][1] - gap)


def _add(p, q):
    return (p[0] + q[0], p[1] + q[1])


_OFF = {}


def head_ground(c, pose, h, head, facing, key, **kw):
    """ground point (x, y) that puts the character's head anchor at `head` (framing by the face)."""
    if key not in _OFF:
        a = c.anchors(0.0, 0.0, 1000.0, pose, 0.0, facing=facing, **kw)
        _OFF[key] = (a["head"][0] / 1000.0, a["head"][1] / 1000.0)
    ox, oy = _OFF[key]
    return head[0] - ox * h, head[1] - oy * h


def _pang(a, facing):
    """world pole angle (+ = leaning right) -> rig pole_ang (+ = leaning toward facing)"""
    return a * (1 if facing >= 0 else -1)


# ------------------------------------------------------------ page 7, p1: the camp, Crow preaching
class P7P1(Drawer):
    rect = R1
    t_on, t_off = -1e9, 166.86
    W_, H_ = R1[2], R1[3]
    HORIZON = 150.0
    SUMMER_HEAD = (180.0, 196.0)
    SUMMER_FACE = (110.0, 130.0, 260.0, 262.0)      # local box kept clear of balloons
    CROW_XYH = (575.0, 505.0, 330.0)

    def __init__(self):
        self.ceiling = A.Ceiling(self.W_, 66, 701, rows=3, dy=28, r=(30, 58), tone=(0.72, 0.86))
        self.rain = A.Rain(self.W_, self.H_, 300, 702, slant=0.2, length=(12, 30), width=(0.6, 1.3))
        self.rain_far = A.Rain(self.W_, self.HORIZON + 20, 240, 703, slant=0.2, speed=(380, 520),
                               length=(6, 12), width=(0.5, 0.8), white=0.6)
        self.splash = A.Splashes(self.W_, self.HORIZON + 20, self.H_, 60, 704)
        self.pud = [(470, 318, 64, 8, 1), (800, 300, 44, 6, 2)]
        self.bolt = A.bolt_paths(452, -60, 402, self.HORIZON + 1, 705, levels=7, rough=0.22, branches=4)

    # characters -------------------------------------------------------------------------------------
    def crow(self, T):
        tr = tracks()["p1"]
        gx, gy = grip(tr, T, 0.47)
        gx -= self.rect[0]
        gy -= self.rect[1]
        m = mouth("PREACHER", T)
        x, y, h = self.CROW_XYH
        k = h / 232.0
        # back-hand acting on the words of P24 / P25
        sh = (x - 8 * k, y - h * 0.80)
        k_finger = seg(T, 160.35, 160.7) * (1 - seg(T, 161.6, 161.95))
        k_heart = seg(T, 161.95, 162.25) * (1 - seg(T, 163.15, 163.45))
        k_open = seg(T, 163.3, 163.6) * (1 - seg(T, 164.35, 164.6))
        k_point = seg(T, 164.55, 164.95) * (1 - seg(T, 166.0, 166.12))
        k_sky = seg(T, 166.0, STRIKE - 0.02, ease_out)
        hl = (sh[0] - 30 * k, sh[1] + 70 * k)          # rest
        tgt = [((sh[0] - 62 * k, sh[1] - 36 * k), k_finger), ((sh[0] + 6 * k, sh[1] + 38 * k), k_heart),
               ((sh[0] - 74 * k, sh[1] + 18 * k), k_open), ((sh[0] - 92 * k, sh[1] + 6 * k), k_point),
               ((sh[0] - 40 * k, sh[1] - 118 * k), k_sky)]
        for p, kk in tgt:
            hl = (lerp(hl[0], p[0], kk), lerp(hl[1], p[1], kk))
        shape_l = "finger" if (k_finger > 0.5 or k_sky > 0.3) else ("point" if k_point > 0.5 else
                                                                      ("open" if k_open > 0.5 else "relax"))
        expr = {"grave": 1.0}
        if T > 164.43:
            expr = {"grave": 1 - seg(T, 164.43, 165.5), "stern": seg(T, 164.43, 165.5)}
        if T > STRIKE - 0.05:
            expr = {"fervent": 1.0}
        lean = 0.05 + 0.08 * seg(T, 164.5, 166.0) - 0.06 * seg(T, STRIKE - 0.05, STRIKE + 0.15)
        _, (_, _, pa) = tr._frame(min(fidx(T), tr.n - 1))
        return dict(x=x, y=y, h=h, pose="stand", t=T, mouth=m, facing=-1, expr=expr,
                    look=(-1, 0.05 - 0.6 * k_sky), hand_r=(gx, gy), shape_r="grip", pole_ang=_pang(pa, -1),
                    hand_l=hl, shape_l=shape_l, lean=lean, nod=0.3 * k_sky - 0.05, prop=None)

    def summer(self, T):
        s = seg(T, STRIKE - 0.05, STRIKE + 0.10, ease_out)
        expr = {"rapt": 1.0 - seg(T, 164.4, 165.6), "worried": seg(T, 164.4, 165.6) * (1 - s)}
        if s > 0:
            expr = {"horror": s, "worried": 1 - s}
        h = 520.0
        x, y = head_ground(SUMMER(), "sit", h, self.SUMMER_HEAD, 1, "p1s")
        return dict(x=x, y=y, h=h, pose={"sit": 1.0, "cheeks": s}, t=T, chair="camp", facing=1, expr=expr,
                    look=(1, -0.15), mouth=0.15 * s, lean=0.05 - 0.14 * s, nod=0.05 - 0.08 * s)

    def raven(self, T):
        s = seg(T, STRIKE, STRIKE + 0.1, ease_out) * (1 - seg(T, STRIKE + 0.3, STRIKE + 0.8))
        h = 640.0
        x, y = head_ground(RAVEN(), {"sit": 1.0, "film": 1.0}, h, (64.0, 176.0 - 10 * s), 1, "p1r", turn=1.35)
        return dict(x=x, y=y, h=h, pose={"sit": 1.0, "film": 1.0}, t=T, chair="camp", facing=1, turn=1.35,
                    prop="phone", expr={"disdain": 1.0 - s, "shocked": s}, look=(1, 0.0))

    # layers ------------------------------------------------------------------------------------------
    def _world(self, ctx, T, crow=True, rain=True):
        bw, flash, neg, bolt_a, _ = strike_state(T)
        Wd, Hd = self.W_, self.H_
        A.sky(ctx, Wd, Hd, T, top=0.86, bottom=0.95, horizon=self.HORIZON, bw=bw, flash=flash)
        self.ceiling.draw(ctx, T, bw=bw, flash=flash, bolt_x=430.0 if bolt_a > 0 else None)
        if bolt_a > 0:
            A.draw_bolt(ctx, *self.bolt, edge=13.0, neg=flash > 0.5, alpha=1.0)
        if rain:
            self.rain_far.draw(ctx, T, bw=bw)
        A.tents(ctx, Wd, self.HORIZON, 706, tone=0.62, bw=bw, n=8, scale=0.7)
        A.ground(ctx, Wd, Hd, self.HORIZON, 707, T, near=0.90, far=0.96, bw=bw, ruts=40)
        A.puddles(ctx, self.pud, T, bw=bw, sky_tone=0.97)
        low = A.shade_structure(ctx, -80, 330, -14, 700, T, depth=40, sag=30, bw=bw, posts="tarp", tarp=0.92)
        A.lantern(ctx, 112, 4, 12, T, bw=bw, glow=0.8)
        if crow:
            _draw(CROW(), ctx, dict(self.crow(T), layer="back"))
        if rain:
            self.rain.draw(ctx, T, bw=bw)
            self.splash.draw(ctx, T, bw=bw)
        _draw(SUMMER(), ctx, self.summer(T))
        _draw(RAVEN(), ctx, self.raven(T))
        if rain:
            A.drips(ctx, [(low[0] + dx, low[1] - abs(dx) * 0.1) for dx in (-90, -40, 5, 50, 120)], Hd, T, 708,
                    bw=bw, size=3.2)
        self.draw_fx(ctx, T)

    def draw_fx(self, ctx, T):
        """shock lines around Summer's head (printed): ink by day, white in the lightning night"""
        if T < STRIKE - 0.02:
            return
        a = _anchors(SUMMER(), self.summer(T))
        hx, hy = a["head"]
        u = T - STRIKE
        k = 1.0 + 0.15 * math.sin(T * 40)
        col = (1, 1, 1) if strike_state(T)[1] <= 0.5 else (0, 0, 0)
        for j in range(11):
            ang = -math.pi / 2 + (j - 5) * 0.26
            r0, r1 = 72 * k, 104 + 6 * (j % 2)
            ctx.move_to(hx + math.cos(ang) * r0, hy - 10 + math.sin(ang) * r0)
            ctx.line_to(hx + math.cos(ang) * r1, hy - 10 + math.sin(ang) * r1)
        ctx.set_line_width(4.5)
        ctx.set_source_rgb(*col)
        ctx.stroke()

    def _flags(self, ctx, T):
        bw, flash, neg, bolt_a, _ = strike_state(T)
        light = (0.35, -0.6, -0.55) if flash > 0.5 else (-0.45, -0.8, 0.4)
        tracks()["p1"].draw(ctx, fidx(T), glow=0.35 * flash, light=light)

    def _front(self, ctx, T):
        _draw(CROW(), ctx, dict(self.crow(T), layer="front"))

    def _letters(self, ctx, fc, T, rect):
        x0, y0 = rect[0], rect[1]
        f = self.SUMMER_FACE
        av_face = _trav("p1", [(f[0] + x0, f[1] + y0, f[2] + x0, f[3] + y0)])
        av = comic.track_avoid(tracks()["p1"], fidx(T))
        a = _anchors(CROW(), self.crow(T))
        mx, my = _add(a["mouth"], (x0, y0))
        _say(ctx, fc, "P24", x0 + 425, y0 + 52, T, tail=(mx - 8, my), w=250, avoid=av_face, rect=rect,
             hold=0.22, vt=164.3)
        _say(ctx, fc, "P25", x0 + 405, y0 + 42, T, tail=(mx - 6, my + 2), w=340, avoid=av_face, rect=rect,
             hold=0.6, vt=166.4)
        if T >= STRIKE - 0.03:
            comic.sfx(ctx, "KRAKOOM!!", x0 + 420, y0 + 292, 70, rot=-0.08, t=T - STRIKE, depth=8, avoid=av)


# ------------------------------------------------------------ page 7, p2a: horror close-up
class P7P2A(Drawer):
    rect = R2A
    t_on, t_off = 166.40, 167.56
    HEAD = (170.0, 168.0)

    def __init__(self):
        w, h = R2A[2], R2A[3]
        self.W_, self.H_ = w, h
        self.rain = A.Rain(w, h, 170, 721, slant=0.2, length=(16, 34), width=(0.9, 1.8), white=1.0)

    def summer(self, T):
        h = 700.0
        tx = 2.0 * noise1(T * 22, 5)
        hx, hy = self.HEAD
        x, y = head_ground(SUMMER(), {"sit": 1.0, "cheeks": 1.0}, h, (hx + tx, hy), 1, "p2a")
        return dict(x=x, y=y, h=h, pose={"sit": 1.0, "cheeks": 1.0}, t=T, facing=1, expr={"horror": 1.0},
                    look=(0.9, -0.1), mouth=0.2 + 0.08 * math.sin(T * 18), chair=False,
                    tilt=-0.05 + 0.02 * math.sin(T * 30))

    def _world(self, ctx, T):
        w, h = self.W_, self.H_
        # black night; a lightning-white Chick "shock burst" radiating behind her head
        ctx.rectangle(-10, -10, w + 20, h + 20)
        A.fill(ctx, A.g(0.02))
        cx, cy = self.HEAD[0] - 10, self.HEAD[1] - 20
        rng = np.random.default_rng(int(T * 12))          # boils on twos
        R = 700.0
        n = 44
        for k in range(n):
            a0 = 2 * math.pi * (k + 0.2 * rng.random()) / n
            a1 = a0 + 2 * math.pi / n * (0.35 + 0.2 * rng.random())
            r0 = 150 + 30 * rng.random()
            ctx.move_to(cx + math.cos(a0) * r0, cy + math.sin(a0) * r0)
            ctx.line_to(cx + math.cos(a0) * R, cy + math.sin(a0) * R)
            ctx.line_to(cx + math.cos(a1) * R, cy + math.sin(a1) * R)
            ctx.close_path()
        A.fill(ctx, A.PAPER)
        self.rain.draw(ctx, T, bw=1.0)
        _draw(SUMMER(), ctx, self.summer(T))


# ------------------------------------------------------------ page 7, p2b: she drops to her knees
class P7P2B(Drawer):
    rect = R2B
    t_on, t_off = 167.30, 169.09
    HORIZON = 132.0

    def __init__(self):
        w, h = R2B[2], R2B[3]
        self.W_, self.H_ = w, h
        self.ceiling = A.Ceiling(w, 58, 731, rows=3, dy=26, r=(26, 50), tone=(0.70, 0.84))
        self.rain = A.Rain(w, h, 320, 732, slant=0.2, length=(12, 30))
        self.splash = A.Splashes(w, self.HORIZON + 30, h, 50, 733)
        self.pud = [(250, 340, 90, 11, 4)]

    def fall(self, T):
        return seg(T, 167.52, KNEEL, ease_in)

    def summer(self, T):
        u = self.fall(T)
        h = 440.0
        bounce = 0.0
        if T > KNEEL:
            bounce = 7.0 * math.exp(-(T - KNEEL) * 9) * math.sin((T - KNEEL) * 40)
        x = lerp(150.0, 222.0, u)
        y = 474.0 + 24.0 * math.sin(math.pi * u) + bounce
        pray = seg(T, 167.98, 168.32)
        expr = {"horror": 1 - seg(T, 167.8, 168.1), "tearful": seg(T, 167.8, 168.1)}
        pose = {"sit": 1 - u, "kneel": u, "pray": pray}
        m = mouth("SUMMER", T)
        return dict(x=x, y=y, h=h, pose=pose, t=T, facing=1, expr=expr, look=(1, -0.8), chair=False,
                    mouth=m, lean=0.18 * u - 0.12 * pray, nod=0.3 * u + 0.1 * pray)

    def crow(self, T):
        tr = tracks()["p2b"]
        _, (_, _, pa) = tr._frame(min(fidx(T), tr.n - 1))
        return dict(x=452.0, y=474.0, h=440.0, pose="stand", t=T, facing=-1, expr={"grave": 1.0},
                    look=(-1, 0.8), mouth=0.0, nod=-0.35, prop=None,
                    hand_r=_add(grip(tr, T, 0.40), (-self.rect[0], -self.rect[1])), shape_r="grip",
                    pole_ang=_pang(pa, -1))

    def _world(self, ctx, T):
        w, h = self.W_, self.H_
        A.sky(ctx, w, h, T, top=0.86, bottom=0.95, horizon=self.HORIZON)
        self.ceiling.draw(ctx, T)
        A.tents(ctx, w, self.HORIZON, 734, tone=0.62, n=4, scale=0.8)
        A.ground(ctx, w, h, self.HORIZON, 735, T, near=0.90, far=0.96, ruts=26)
        A.puddles(ctx, self.pud, T, sky_tone=0.97)
        A.camp_chair(ctx, 120.0, 474.0, 118.0, facing=1, tone=0.86)
        _draw(CROW(), ctx, dict(self.crow(T), layer="back"))
        _draw(SUMMER(), ctx, self.summer(T))
        A.drips(ctx, [(20, 8), (70, 14), (120, 10)], h, T, 736)
        self.rain.draw(ctx, T)
        self.splash.draw(ctx, T)
        # the knees hit the mud: SPLUT crown
        if T >= KNEEL - 0.02:
            self.mud_splash(ctx, T - KNEEL, 236.0, 346.0)

    def mud_splash(self, ctx, u, x, y):
        if u > 0.6:
            return
        rng = np.random.default_rng(77)
        for k in range(18):
            ang = -math.pi / 2 + (rng.random() - 0.5) * 2.6
            v = 160 + 200 * rng.random()
            px = x + (rng.random() - 0.5) * 80 + math.cos(ang) * v * u
            py = y - 4 + math.sin(ang) * v * u + 0.5 * 900 * u * u
            r = (3 + 5 * rng.random()) * (1 - u * 0.7)
            ctx.new_path()
            ctx.arc(px, py, r, 0, 2 * math.pi)
            A.fill(ctx, A.g(0.4))
            ctx.new_path()
            ctx.arc(px, py, r, 0, 2 * math.pi)
            ctx.set_line_width(1.0)
            ctx.set_source_rgb(0, 0, 0)
            ctx.stroke()

    def _flags(self, ctx, T):
        tracks()["p2b"].draw(ctx, fidx(T))

    def _front(self, ctx, T):
        _draw(CROW(), ctx, dict(self.crow(T), layer="front"))

    def _letters(self, ctx, fc, T, rect):
        x0, y0 = rect[0], rect[1]
        a = _anchors(SUMMER(), self.summer(T))
        mx, my = _add(_tip_above(a), (x0, y0))
        _say(ctx, fc, "S06", x0 + 214, y0 + 46, T, tail=(mx + 6, my - 4), w=190, rect=rect, hold=0.8, vt=168.6,
             avoid=_trav("p2b"))
        if KNEEL - 0.02 <= T < KNEEL + 0.9:
            comic.sfx(ctx, "SPLUT!", x0 + 120, y0 + 306, 40, rot=-0.12, t=T - KNEEL, depth=3)


# ------------------------------------------------------------ page 7, p3: "Repeat after me."
class P7P3(Drawer):
    rect = R3
    t_on, t_off = 168.72, 170.64
    HORIZON = 262.0
    CROW_HEAD = (548.0, 138.0)

    def __init__(self):
        w, h = R3[2], R3[3]
        self.W_, self.H_ = w, h
        self.ceiling = A.Ceiling(w, 118, 741, rows=4, dy=30, r=(30, 58), tone=(0.70, 0.84))
        self.rain = A.Rain(w, h, 420, 742, slant=0.2, length=(12, 30))
        self.splash = A.Splashes(w, self.HORIZON + 20, h, 40, 743)
        self.pud = [(760, 330, 70, 8, 6)]

    def summer(self, T):
        m = mouth("SUMMER", T)
        h = 640.0
        pose = {"kneel": 1.0, "pray": 1.0}
        x, y = head_ground(SUMMER(), pose, h, (196.0, 262.0), 1, "p3s", turn=1.3)
        return dict(x=x, y=y, h=h, pose=pose, t=T, facing=1, turn=1.3, expr={"tearful": 1.0},
                    look=(1, -0.9), mouth=m, nod=0.45)

    def crow(self, T):
        tr = tracks()["p3"]
        g_ = _add(grip(tr, T, 0.46), (-self.rect[0], -self.rect[1]))
        _, (_, _, pa) = tr._frame(min(fidx(T), tr.n - 1))
        m = mouth("PREACHER", T)
        k = seg(T, 169.5, 169.85) * (1 - seg(T, 170.35, 170.6))
        h = 600.0
        x, y = head_ground(CROW(), "stand", h, self.CROW_HEAD, -1, "p3c")
        sh = (x - 12, y - h * 0.80)
        hl = (lerp(sh[0] - 30, sh[0] - 190, k), lerp(sh[1] + 170, sh[1] + 90, k))
        return dict(x=x, y=y, h=h, pose="stand", t=T, facing=-1, prop=None,
                    expr={"serene": 0.6, "grave": 0.4}, look=(-1, 0.8), mouth=m, lean=0.12, nod=-0.3,
                    hand_r=g_, shape_r="grip", pole_ang=_pang(pa, -1), hand_l=hl, shape_l="open")

    def _world(self, ctx, T):
        w, h = self.W_, self.H_
        A.sky(ctx, w, h, T, top=0.90, bottom=0.96, horizon=self.HORIZON)
        # the hymn: heavenly rays behind Crow's head fade in with the organ
        gl = seg(T, HYMN, HYMN + 1.2)
        self.ceiling.draw(ctx, T)
        if gl > 0:
            a = _anchors(CROW(), self.crow(T))
            hx, hy = a["head"]
            A.glory_lines(ctx, hx, hy, gl * 0.85, n=90, r0=90, r1=520, a0=-math.pi, a1=math.pi, lw=1.3, seed=9)
        A.tents(ctx, w, self.HORIZON, 744, tone=0.62, n=7, scale=0.9)
        A.ground(ctx, w, h, self.HORIZON, 745, T, near=0.90, far=0.96, ruts=24)
        A.puddles(ctx, self.pud, T, sky_tone=0.98)
        self.rain.draw(ctx, T)
        _draw(CROW(), ctx, dict(self.crow(T), layer="back"))
        _draw(SUMMER(), ctx, self.summer(T))
        self.splash.draw(ctx, T)

    def _flags(self, ctx, T):
        tracks()["p3"].draw(ctx, fidx(T))

    def _front(self, ctx, T):
        _draw(CROW(), ctx, dict(self.crow(T), layer="front"))

    def _letters(self, ctx, fc, T, rect):
        x0, y0 = rect[0], rect[1]
        av = _trav("p3")
        a = _anchors(CROW(), self.crow(T))
        mx, my = _add(a["mouth"], (x0, y0))
        _say(ctx, fc, "P26", x0 + 330, y0 + 56, T, tail=(mx - 6, my), w=240, avoid=av, rect=rect, hold=0.6, vt=170.2)
        if PLANT - 0.02 <= T < PLANT + 0.8:
            comic.sfx(ctx, "THUNK", x0 + 720, y0 + 318, 30, rot=0.1, t=T - PLANT, depth=2,
                      avoid=comic.track_avoid(tracks()["p3"], fidx(T)))


# ------------------------------------------------------------ page 8, p1: the prayer, Amen, the sunbeam
class P8P1(Drawer):
    rect = R8
    t_on, t_off = 170.20, 1e9
    HORIZON = 200.0
    SUN = (380.0, -320.0)
    SUMMER_HEAD = (380.0, 196.0)
    CROW_HEAD = (642.0, 190.0)
    hold = T1                        # t09 may set a later value

    def __init__(self):
        w, h = R8[2], R8[3]
        self.W_, self.H_ = w, h
        self.ceiling = A.Ceiling(w, 104, 801, rows=4, dy=28, r=(30, 58), tone=(0.70, 0.84))
        self.rain = A.Rain(w, h, 300, 802, slant=0.2, length=(12, 30), width=(0.6, 1.2))
        self.splash = A.Splashes(w, self.HORIZON + 30, h, 40, 803)
        self.pud = [(90, 330, 60, 8, 7), (800, 318, 60, 8, 8)]

    def amen(self, T):
        return seg(T, AMEN - 0.02, AMEN + 0.75, ease_out)

    def beam(self, T):
        return seg(T, AMEN + 0.04, AMEN + 0.34, ease_out)

    def summer(self, T):
        T = min(T, self.hold)
        am = self.amen(T)
        m = mouth("SUMMER", T)
        look_up = seg(T, AMEN + 0.45, AMEN + 0.9)
        expr = {"tearful": 1 - am, "beatific": am * (1 - look_up), "rapt": look_up}
        h = 520.0
        pose = {"kneel": 1.0, "pray": 1.0}
        x, y = head_ground(SUMMER(), pose, h, self.SUMMER_HEAD, 1, "p8s", turn=0.35)
        sway = 1.5 * math.sin(T * 1.3)
        return dict(x=x + sway, y=y, h=h, pose=pose, t=T, facing=1, turn=0.35, expr=expr,
                    look=(0.4 - 0.9 * look_up, 0.2 - 1.0 * look_up), mouth=m, eyes_closed=look_up < 0.5,
                    nod=-0.12 + 0.5 * look_up)

    def crow(self, T):
        T = min(T, self.hold)
        m = mouth("PREACHER", T)
        h = 530.0
        pose = {"kneel": 1.0, "pray": 1.0}
        x, y = head_ground(CROW(), pose, h, self.CROW_HEAD, -1, "p8c", turn=0.3)
        return dict(x=x, y=y, h=h, pose=pose, t=T, facing=-1, turn=0.3, prop=None,
                    expr={"serene": 1.0}, look=(-0.5, 0.3), mouth=m, eyes_closed=True, nod=-0.14)

    def raven(self, T):
        T = min(T, self.hold)
        u = seg(T, 171.0, 180.6, ease_in_out)
        h = 190.0
        x, y = 222.0, 300.0
        return dict(x=x, y=y, h=h, pose={"sit": 1.0, "film": 1.0 - u}, t=T, chair="camp", facing=1, turn=0.6,
                    prop="phone", expr={"disdain": 1.0 - 0.6 * u, "bewildered": 0.6 * u}, look=(1, 0.1))

    def _world(self, ctx, T):
        w, h = self.W_, self.H_
        am = self.amen(T)
        bm = self.beam(T)
        A.sky(ctx, w, h, T, top=0.93 + 0.06 * am, bottom=0.97 + 0.03 * am, horizon=self.HORIZON)
        A.glory_lines(ctx, self.SUN[0], self.SUN[1], am, n=64, r0=60, r1=900, a0=math.pi * 0.10,
                      a1=math.pi * 0.90, lw=1.0, clip_r=720 * am * 1.05)
        self.ceiling.draw(ctx, T, part=am, sun=self.SUN, reach=720, lit=am)
        A.tents(ctx, w, self.HORIZON, 804, tone=0.62, n=8, scale=0.7)
        A.ground(ctx, w, h, self.HORIZON, 805, T, near=0.90, far=0.96, ruts=36)
        A.puddles(ctx, self.pud, T, stop=AMEN + 0.3, sky_tone=0.98)
        low = A.shade_structure(ctx, 96, 340, 118, 296, T, depth=24, sag=14)
        _draw(RAVEN(), ctx, self.raven(T))
        A.drips(ctx, [(low[0] + dx, low[1] - abs(dx) * 0.1) for dx in (-60, -15, 40, 90)], 296, T, 806, size=1.8)
        self.rain.draw(ctx, T, stop=AMEN)
        self.splash.draw(ctx, T, stop=AMEN + 0.25)
        a = _anchors(SUMMER(), self.summer(T))
        tx, ty = a["head"]
        # the beam lights the air and the mud (behind the figures: their ink stays solid)
        A.spotlight(ctx, w, h, self.SUN[0], self.SUN[1], tx + 4, ty + 60, 0.12, clamp(bm * 1.6), dim=0.96,
                    reach=0.2 + 0.8 * bm)
        # a halo of Chick-tract radiance around the saved head
        rad = seg(T, AMEN + 0.25, AMEN + 0.7)
        if rad > 0:
            A.glory_lines(ctx, tx, ty, rad, n=44, r0=78, r1=150, a0=-math.pi, a1=math.pi, lw=1.3, seed=12)
        _draw(SUMMER(), ctx, self.summer(T))
        _draw(CROW(), ctx, self.crow(T))
        self.draw_fx(ctx, T)

    def draw_fx(self, ctx, T):
        T = min(T, self.hold)
        # rain dripping off Crow's cap brim
        c = _anchors(CROW(), self.crow(T))
        hx, hy = c["head"]
        if T < AMEN + 1.5:
            A.drips(ctx, [(hx - 52, hy - 34), (hx - 30, hy - 30)], hy + 200, T, 809, size=2.4)

    def _flags(self, ctx, T):
        am = self.amen(T)
        tr = tracks()["p8"]
        tr.draw(ctx, fidx(T), glow=0.5 * am, light=(-0.55 - 0.3 * am, -0.8, 0.4 - 0.5 * am))

    def _letters(self, ctx, fc, T, rect):
        x0, y0 = rect[0], rect[1]
        c = _anchors(CROW(), self.crow(T))
        cm = _add(_tip_above(c), (x0, y0))
        s = _anchors(SUMMER(), self.summer(T))
        sm = _add(_tip_above(s), (x0, y0))
        # call (Crow, upper right) and response (Summer, upper left); each pair replaces the last
        cx, cy = x0 + 680, y0 + 56
        sx, sy = x0 + 318, y0 + 56
        av = _trav("p8")
        for lid in ("P27", "P28", "P29"):
            _say(ctx, fc, lid, cx, cy, T, tail=(cm[0] - 4, cm[1] - 2), w=180, avoid=av, rect=rect, hold=0.35, page=8)
        for lid in ("S07", "S08", "S09"):
            _say(ctx, fc, lid, sx, sy, T, tail=(sm[0] + 4, sm[1] - 2), w=180, avoid=av, rect=rect, hold=0.35, page=8)


DRAWERS = {}


def drawers():
    if not DRAWERS:
        DRAWERS[(7, "p1")] = P7P1()
        DRAWERS[(7, "p2a")] = P7P2A()
        DRAWERS[(7, "p2b")] = P7P2B()
        DRAWERS[(7, "p3")] = P7P3()
        DRAWERS[(8, "p1")] = P8P1()
    return DRAWERS


RECTS = {"p1": R1, "p2a": R2A, "p2b": R2B, "p3": R3}


def page_panels(page):
    d = drawers()
    if page == 7:
        return {RECTS[k]: d[(7, k)] for k in ("p1", "p2a", "p2b", "p3")}
    return {R8: d[(8, "p1")]}


# ============================================================ t07 hand-in helpers (design units)
def _open_cam():
    return CAM_OPEN


def draw_camp(ctx, T, crow=False, rain=False):
    """page 7 p1 grey world at my opening camera, in DESIGN units (default: no Crow, no flags, no rain).
    Clip it to the panel yourself if your camera shows more than the panel."""
    d = drawers()[(7, "p1")]
    ctx.save()
    _open_cam().apply(ctx)
    ctx.translate(R1[0], R1[1])
    d._world(ctx, d.tp(T), crow=crow, rain=rain)
    ctx.restore()


def draw_rain(ctx, T):
    d = drawers()[(7, "p1")]
    ctx.save()
    _open_cam().apply(ctx)
    ctx.translate(R1[0], R1[1])
    d.rain_far.draw(ctx, T)
    d.rain.draw(ctx, T)
    d.splash.draw(ctx, T)
    ctx.restore()


def _to_design(px, py):
    return _open_cam().to_screen(px, py)


def crow_t07():
    """Crow at 159.83 in design units (for t07's last frame)."""
    d = drawers()[(7, "p1")]
    a = d.crow(T0)
    x, y = _to_design(a["x"] + R1[0], a["y"] + R1[1])
    z = _open_cam().zoom
    kw = {k: v for k, v in a.items() if k not in ("x", "y", "h", "pose")}
    for k in ("hand_r", "hand_l"):
        if kw.get(k) is not None:
            kw[k] = _to_design(kw[k][0] + R1[0], kw[k][1] + R1[1])
    return dict(x=x, y=y, h=a["h"] * z, facing=-1, pose=a["pose"], kw=kw)


def flag_t07():
    z = _open_cam().zoom
    bx, by = _to_design(*P1_POLE)
    return dict(base=(bx, by), ang=P1_ANG, pole=P1_L * z, wind=_wind_p1(0), seed=81, turbulence=0.8, flutter=0.8,
                side=1, note="pole = butt planted in the mud; track index 0 = global frame 3836")

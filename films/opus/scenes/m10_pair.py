"""m10 - the donor pair: Lutie offers Master Crocus a small Flag; it is sticky (owner M10).

Everything here is in CONCH CHART units (the pair is painted on the apse conch). Pure functions of the global
time T: poses are keyframed (eased) and drawn through vx.icon; anchors come from Figure.anchors (no drawing).

    pose_lutie(T) / pose_crocus(T) -> kwargs for Figure.draw / Figure.anchors
    draw(ctx, T, layer='color', part='all')     both figures (Lutie behind Crocus's arm)
    anchors(T) -> dict(grip=(x, y), pole_ang, elbow=(x, y), wrist=(x, y), cuff=(x, y, ang), head_c, head_l, ...)
    FLAG: the small Flag's size; flag_pole(T) -> (base_x, base_y, ang) with the pole stuck in Lutie's grip
"""
import math

import numpy as np

from vx import icon
from vx.ease import ease_in_out, clamp
from vx.flag import Flag
from vx.timeline import get_timeline

from . import m10_art as A
from . import m10_cloth as Cl

TL = get_timeline()
W_ = TL.word

# ------------------------------------------------------------------ placement (conch chart)
_L = next(f for f in A.conch_figures() if f["role"] == "lutie")
_C = next(f for f in A.conch_figures() if f["role"] == "crocus")
LX, LY, LH = _L["u"], _L["v"], _L["h"]
CX, CY, CH = _C["u"], _C["v"], _C["h"]
LUTIE = icon.Figure("lutie", seed=1, halo="square")
CROCUS = icon.Figure("crocus", seed=2, halo="square")
TILE = 5.4 / 1.92                  # chart units: the pair's tile (opus vermiculatum; m10_explicit.PAIR_TILE)
FLAG = dict(pole=0.64 * LH, width=0.30 * LH, height=0.235 * LH)     # the small Flag (chart units)
GRIP_FRAC = 0.34                   # where along the pole (from the base) her fist holds it


# ------------------------------------------------------------------ keyframes
def _kf(T, keys):
    """eased interpolation through [(T, value), ...] (values float or tuples)"""
    if T <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if T <= t1:
            u = ease_in_out((T - t0) / max(t1 - t0, 1e-6), 2)
            if isinstance(v0, tuple):
                return tuple(a + (b - a) * u for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def _w(lid, k, n=0):
    return W_(lid, k, n)


# cue times (global s), from the locked timeline
T_MASTER = _w("L04", "Master")[0]
T_FORYOU = _w("L04", "you")[0]
T_WANT = _w("C06", "want")[0]
T_THIS = _w("C06", "this")[0]
T_FLAGW = _w("C06", "Flag")
T_ITIS = _w("C06", "It")[0]
T_STICKY1 = _w("C06", "sticky")
T_BUT = _w("L05", "But")[0]
T_ALL = _w("L05", "all")[0]
T_ONE = _w("L05", "one")[0]
T_FLAG2 = _w("L05", "Flag")
T_THIS2 = _w("C07", "This")[0]
T_STICKY2 = _w("C07", "sticky")
T_GLORIOUS = TL.cue("glorious")
T_STICK_ON = T_WANT + 0.05          # from here on, cloth touching the sleeve sticks


def pose_lutie(T):
    """Figure.draw kwargs for Lutie at global time T"""
    kw = dict(facing=0.8, halo="square", tile=TILE, mouth=TL.mouth("LUTIE", T), t=T)
    # pole angle: offered toward the Master, pulled a little when the cloth sticks
    pa = _kf(T, [(187.5, 0.12), (T_MASTER, 0.14), (T_MASTER + 0.5, 0.36), (T_FORYOU + 0.3, 0.44),
                 (T_THIS + 0.4, 0.46), (T_FLAGW[0] + 0.4, 0.55), (T_ITIS + 0.4, 0.62), (T_BUT, 0.62),
                 (T_BUT + 0.5, 0.84), (T_GLORIOUS - 0.25, 0.82), (T_GLORIOUS + 0.3, 0.08)])
    lean = _kf(T, [(187.5, 0.0), (T_MASTER + 0.3, 0.06), (T_FORYOU + 0.4, 0.09), (T_ITIS + 0.3, 0.12),
                   (T_BUT, 0.05), (T_ALL, 0.1), (T_FLAG2[1], 0.06), (T_GLORIOUS, 0.0)])
    kw["lean"] = lean
    kw["pole_ang"] = pa
    # poses: present -> (hands off: let go, the pole stays at her palm) -> explain (all Flags) -> object (one Flag)
    t_letgo = T_STICKY1[1] - 0.05
    if T < t_letgo:
        kw["pose"] = "present"
        kw["expr"] = "serene" if T < T_MASTER else "joyful" if T < T_ITIS else "worried"
    elif T < T_BUT + 0.05:
        # hands off! she lets go and shakes her hand: the pole stays glued to her palm
        w = clamp((T - t_letgo) / 0.3)
        kw["pose"] = {"present": 1 - w, "let_go": max(w, 1e-3)}
        kw["expr"] = "worried"
        base = LUTIE.anchors(LX, LY, LH, "let_go", T, 0.8, lean=lean, pole_ang=pa)["hand_r"]
        u = clamp((T - t_letgo - 0.1) / 0.5)
        sh = 0.03 * LH * math.sin(2 * math.pi * 4.6 * (T - t_letgo)) * math.sin(math.pi * u)
        kw["hand_r"] = (float(base[0]) - 0.05 * LH * ease_in_out(u, 2) + sh, float(base[1]) - 0.02 * LH * u)
    elif T < T_ONE - 0.1:
        w = clamp((T - T_BUT) / 0.35)
        kw["pose"] = {"let_go": 1 - w, "explain": max(w, 1e-3)}
        kw["sweep"] = _kf(T, [(T_BUT, 0.0), (T_ALL, 0.2), (T_ALL + 0.45, 1.0)])
        kw["expr"] = "joyful"
    elif T < T_GLORIOUS - 0.1:
        kw["pose"] = "hold_pole1"
        kw["arm_l"] = "object"
        kw["expr"] = "joyful" if T < T_FLAG2[1] + 0.5 else "serene"
    else:
        kw["pose"] = "raise"
        kw["expr"] = "joyful"
    look_x = 1.0 if T < T_GLORIOUS else 0.2
    kw["look"] = (look_x, -0.1)
    return kw


def pose_crocus(T):
    kw = dict(facing=0.0, halo="square", tile=TILE, mouth=TL.mouth("CROCUS", T), t=T + 3.1, pose="fend")
    fend = _kf(T, [(187.5, 0.0), (T_WANT - 0.05, 0.0), (T_THIS + 0.15, 0.52), (T_FLAGW[1], 0.6),
                   (T_ITIS, 0.6), (T_ITIS + 0.35, 0.93), (T_BUT, 0.9), (T_THIS2, 0.88), (T_THIS2 + 0.3, 0.94),
                   (T_GLORIOUS - 0.2, 0.92), (T_GLORIOUS + 0.25, 1.0)])
    kw["fend"] = fend
    # the shake: a few flicks of the wrist on "sticky" (both times), deadpan
    sx = 0.0
    for (a, b), amp, hz in ((T_STICKY1, 0.028, 3.4), (T_STICKY2, 0.016, 3.0)):
        if a <= T <= b:
            env = math.sin(math.pi * clamp((T - a) / (b - a)))
            sx += amp * CH * env * math.sin(2 * math.pi * hz * (T - a))
    kw["shake"] = (sx, -0.3 * abs(sx))
    # gaze: to the Flag when it is offered, to the camera to refuse it, to Lutie when she explains
    look = _kf(T, [(187.5, (0.0, 0.0)), (T_MASTER + 0.2, (-0.7, 0.05)), (T_FORYOU + 0.35, (-0.55, 0.45)),
                   (T_WANT - 0.35, (-0.5, 0.4)), (T_WANT, (0.0, 0.0)), (T_BUT, (0.0, 0.0)), (T_BUT + 0.4, (-0.75, 0.0)),
                   (T_FLAG2[1] + 0.2, (-0.75, 0.0)), (T_THIS2 - 0.15, (0.0, 0.0))])
    kw["look"] = look
    kw["expr"] = "wry" if T_STICKY2[1] < T < T_GLORIOUS else "neutral"
    if T >= T_GLORIOUS - 0.15:
        # Glorious: the Master lifts his arm (and the cloth glued to it) with her
        w = ease_in_out(clamp((T - T_GLORIOUS + 0.15) / 0.45), 2)
        kw["pose"] = {"fend": 1 - w + 1e-3, "declare": max(w, 1e-3)}
        kw["expr"] = "serene"
    return kw


def _anch(fig, x, y, h, kw):
    kk = {k: v for k, v in kw.items() if k not in ("mouth", "halo", "tile", "expr", "look")}
    pose = kk.pop("pose")
    t = kk.pop("t", 0.0)
    facing = kk.pop("facing", 0.0)
    return fig.anchors(x, y, h, pose, t, facing, **kk)


def anchors(T):
    kl = pose_lutie(T)
    kc = pose_crocus(T)
    al = _anch(LUTIE, LX, LY, LH, kl)
    ac = _anch(CROCUS, CX, CY, CH, kc)
    grip = al["grip"] if "grip" in al else al["hand_r"]
    return dict(grip=(float(grip[0]), float(grip[1])), pole_ang=float(kl["pole_ang"]), lutie=al, crocus=ac,
                elbow=tuple(map(float, ac["elbow_r"][:2])), wrist=tuple(map(float, ac["wrist_r"][:2])),
                cuff=tuple(map(float, ac["cuff_r"])))


def flag_pole(T, a=None):
    """pole base (x, y, ang) of the small Flag: glued in Lutie's fist/palm at GRIP_FRAC of its length"""
    a = a or anchors(T)
    ang = a["pole_ang"]
    if T >= T_GLORIOUS - 0.1:
        ang = float(a["lutie"]["pole"][2])
    gx, gy = a["grip"]
    L = FLAG["pole"] * GRIP_FRAC
    return gx - L * math.sin(ang), gy + L * math.cos(ang), ang


def draw(ctx, T, layer="color", part="all"):
    """both figures in the chart (Crocus in front: his fending arm crosses toward Lutie)"""
    kl = pose_lutie(T)
    kc = pose_crocus(T)
    LUTIE.draw(ctx, LX, LY, LH, layer=layer, part=part, **kl)
    if part != "front":
        CROCUS.draw(ctx, CX, CY, CH, layer=layer, **kc)


def bbox(pad=0.25):
    """chart rectangle that contains the pair through the whole choreography (+ pad of the height)"""
    xs, ys = [], []
    for T in np.arange(187.5, 212.0, 0.25):
        a = anchors(T)
        for an in (a["lutie"], a["crocus"]):
            x0, y0, x1, y1 = an["bbox"]
            xs += [x0, x1]
            ys += [y0, y1]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    p = pad * CH
    return x0 - p, y0 - p, x1 + p, y1 + p * 0.3


# ------------------------------------------------------------------ the sticky Flag
ARM_R = 0.05          # sleeve radius, fraction of Crocus's height


def simulate(n, t0=187.5, fps=24, seed=7):
    """the small Flag through the whole scene: the pole glued in Lutie's hand, the cloth glued to Crocus's sleeve
    wherever it touches it after T_STICK_ON. Two passes: the sleeve comes up just behind wherever the free cloth
    is, so the first contact is a real touch (then the glue holds). -> (FlagTrack (chart units), info)"""
    fl = Flag(width=FLAG["width"], height=FLAG["height"], pole=FLAG["pole"], seed=seed)
    As = [anchors(t0 + i / fps) for i in range(n)]
    poles = [flag_pole(t0 + i / fps, As[i]) for i in range(n)]
    def wind(i):
        T = t0 + i / fps
        w = 240.0 + 60.0 * math.sin(T * 0.7)
        return (w, -20.0)
    r = ARM_R * CH
    tr0, _ = Cl.simulate_sticky(fl, n, lambda i: poles[i], wind, lambda i: None, lambda i: 0.0, arm_r=r, fps=fps)
    zs = np.full(n, np.nan)
    for i in range(n):
        a = As[i]
        (ex, ey), (wx, wy) = a["elbow"], a["wrist"]
        V = tr0.verts[i].reshape(-1, 3)
        d = np.array([wx - ex, wy - ey])
        L2 = max(d @ d, 1e-9)
        s = np.clip(((V[:, 0] - ex) * d[0] + (V[:, 1] - ey) * d[1]) / L2, 0, 1)
        dist = np.hypot(V[:, 0] - (ex + s * d[0]), V[:, 1] - (ey + s * d[1]))
        k = dist < 2.5 * r
        if k.sum() >= 3:
            zs[i] = np.median(V[k, 2])
    ok = ~np.isnan(zs)
    idx = np.arange(n)
    zs = np.interp(idx, idx[ok], zs[ok]) if ok.any() else np.zeros(n)
    zs = np.convolve(np.pad(zs, 6, mode="edge"), np.ones(13) / 13, mode="same")[6:-6] - 0.6 * r
    i_on = int(round((T_STICK_ON - t0) * fps))
    z_hold = zs[min(max(i_on + int(0.9 * fps), 0), n - 1)]
    def arm_fn(i):
        a = As[i]
        z = zs[i] if i <= i_on + int(0.9 * fps) else z_hold
        (ex, ey), (wx, wy) = a["elbow"], a["wrist"]
        # the sleeve runs a little past the wrist (the cuff is wide)
        return (ex, ey, z, wx + 0.25 * (wx - ex), wy + 0.25 * (wy - ey), z)
    def stick_fn(i):
        return 1.0 if t0 + i / fps >= T_STICK_ON else 0.0
    return Cl.simulate_sticky(fl, n, lambda i: poles[i], wind, arm_fn, stick_fn, arm_r=r, fps=fps)

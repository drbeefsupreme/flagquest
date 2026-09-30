"""m03 - THE EDICT OF THE JAGUAR (48.5-62.5). Owner: M03.

The Justinian panel of San Vitale, with a Jaguar. The slop's pastel static (m02) crystallises into screen-tesserae
that flip, one by one from his eyes outward, into glass and gold. "My fellow Americans" (paw on heart); the camera
pulls back on "and everybody else!" to reveal the whole court while the titulus HIC IMPERATOR IAGVAR LEGEM SANCIT
sets itself. He signs LEX NOOSPHAERICA - MMXLII with the nib of his stylus-sceptre, twirls it, and slams its seal
end on "banned": every tessera on the wall jumps. On "Except the yellow ones" the icon LEANS OUT OF ITS OWN WALL
towards us - leaving its halo (and a man-shaped socket of bare setting bed) behind - and an ochre Vexillian sleeve
slips a gold solidus into his paw while the whole court's eyes slide to it; he elevates the coin like the Host
("a sacrament"), winks, and its glint flares into the gold point m04 opens from.

Hand-offs: frame 0 (48.5) = m02's pastel_static (scenes/m02_static.py); last frame (62.458) = black with one gold
point at (960, 540) (agreed with M04).
"""
import math

import cv2
import numpy as np

from vx import *
from vx import mosaic as mz
from vx import icon as I
from vx.foley import export_events, export_track, screen_pan

from scenes import m03_panel as P
from scenes.m02_static import pastel_static

POST = dict(bloom=0.3, bloom_thresh=0.76, bloom_radius=20.0, grain=0.012, vignette=0.36)

EMP = P.EMP
HINGE = (EMP["x"], EMP["y"])               # the Emperor's slab tips out of the wall about his feet
LAYERS = ("color", "gold", "pearl")


# ============================================================ small helpers
def track(T, keys):
    """eased keyframes [(t, v), ...] (v float or tuple); held before/after."""
    if T <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if T <= t1:
            u = smootherstep(t0, t1, T)
            if isinstance(v0, tuple):
                return tuple(a + (b - a) * u for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def lerp2(p, q, u):
    return (p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u)


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


# ============================================================ beats (global seconds, from the locked timeline)
class Beats:
    def __init__(self, tl):
        J = lambda w, n=0: tl.word("J01", w, n)
        K = lambda w, n=0: tl.word("J02", w, n)
        self.sign = tl.cue("edict_sign")          # 52.5 (word "sign")
        self.seal = tl.cue("edict_seal")          # 58.0 (word "banned")
        self.kaching = tl.cue("ka_ching")         # 62.1
        self.my = J("My")[0]
        self.and_ = J("and")[0]
        self.else_ = J("else")[1]
        self.twenty = J("twenty")[0]
        self.fortytwo = J("forty-two")
        self.weapon = J("Weaponized")[0]
        self.hereby = J("hereby")
        self.except_ = K("Except")[0]
        self.yellow = K("yellow")[0]
        self.those = K("Those")[0]
        self.are = K("are")
        self.a = K("a")[0]
        self.sacr = K("sacrament")
        self.sig0 = self.sign
        self.sig1 = self.sign + 1.85                 # the flourish ends
        self.tit = (self.and_ + 0.25, self.else_ + 0.05)
        self.lean0 = self.except_ - 0.05
        self.lean1 = self.lean0 + 0.7
        self.hand_in = (self.those - 0.12, self.those + 0.42)
        self.drop = self.are[1] - 0.12                # coin released
        self.land = self.drop + 0.16                  # coin lands in the paw
        self.hand_out = (self.land + 0.05, self.land + 0.5)
        self.elev = (self.a + 0.02, self.sacr[0] + 0.62)
        self.wink = (self.sacr[0] + 0.24, self.sacr[0] + 0.62)
        self.push = (self.elev[0], self.kaching - 0.06)


def numeral_times(B):
    t0, t1 = B.twenty + 0.05, B.fortytwo[1] - 0.1
    return [t0 + (t1 - t0) * i / 5 for i in range(6)]


def titulus_times(B, n):
    t0, t1 = B.tit
    return [t0 + (t1 - t0) * i / max(1, n - 1) for i in range(n)]


def sig_schedule(B):
    """signature progress 0..1 over time: quick strokes with little hesitations (a flourish)."""
    ts = np.linspace(B.sig0, B.sig1, 96)
    u = (ts - B.sig0) / (B.sig1 - B.sig0)
    speed = 1.0 + 0.55 * np.sin(u * 19.0) ** 2 - 0.45 * np.exp(-((u - 0.62) / 0.05) ** 2)
    prog = np.cumsum(speed)
    prog = (prog - prog[0]) / (prog[-1] - prog[0])
    return ts, prog


# ============================================================ the Emperor's choreography
REST_L = (EMP["x"] + 0.108 * EMP["h"], EMP["y"] - 0.53 * EMP["h"])    # left paw holding the sceptre upright
HEART = (EMP["x"] + 0.04 * EMP["h"], EMP["y"] - 0.645 * EMP["h"])     # right paw on his heart
SIDE_R = (EMP["x"] - 0.07 * EMP["h"], EMP["y"] - 0.5 * EMP["h"])      # right paw resting
PALM = (EMP["x"] - 0.26 * EMP["h"], EMP["y"] - 0.42 * EMP["h"])       # palm out, palm up (the bribe)
ELEV = (EMP["x"] - 0.158 * EMP["h"], EMP["y"] - 0.855 * EMP["h"])     # the coin held up beside his face


def sceptre_pose(T, B, st):
    """(paw x, paw y, angle, local grip y) of the stylus-sceptre in slab-rest wall coordinates.
    angle 0 = seal up (rotation clockwise); the paw holds the rod at local (DISC x, grip y) - it chokes up towards
    the nib to write."""
    g0 = P.SCEPTRE_GRIP[1]
    g_write = P.SCEPTRE_NIB[1] - 38.0
    disc_off = g0 - P.SCEPTRE_DISC[1]
    pts, u = st["sig_pts"]
    ts, prog = st["sig_sched"]
    a_sign = -0.55                      # the seal end leans back toward his shoulder, like a pen

    def paw_for_nib(nib, a, gl):
        d = P.SCEPTRE_NIB[1] - gl
        return nib[0] + d * math.sin(a), nib[1] - d * math.cos(a)

    rest = REST_L
    if T < B.sig0 - 0.55:
        return rest[0], rest[1], 0.0, g0
    if T < B.sig0:
        k = smootherstep(B.sig0 - 0.55, B.sig0 - 0.02, T)
        gl = g0 + (g_write - g0) * k
        g = lerp2(rest, paw_for_nib(pts[0], a_sign, g_write), k)
        return g[0], g[1], a_sign * k, gl
    if T <= B.sig1:
        p = float(np.interp(T, ts, prog))
        i = min(len(pts) - 1, int(np.searchsorted(u, p)))
        a = a_sign + 0.06 * math.sin(T * 23.0)
        g = paw_for_nib(pts[i], a, g_write)
        return g[0], g[1], a, g_write
    g_end = paw_for_nib(pts[-1], a_sign, g_write)
    t_back = B.sig1 + 0.55
    if T < t_back:
        k = smootherstep(B.sig1, t_back, T)
        g = lerp2(g_end, rest, k)
        return g[0], g[1], a_sign * (1 - k), g_write + (g0 - g_write) * k
    # the twirl (Weaponized memes), the raise seal-down (hereby), the slam (banned), the return
    seal_c = (P.SEAL["cx"], P.SEAL["cy"])
    tw0, tw1 = B.weapon + 0.05, B.weapon + 0.95
    up1 = B.seal - 0.11
    hi = (seal_c[0] - 40, seal_c[1] - disc_off - 170)
    down = (seal_c[0], seal_c[1] - disc_off)                    # paw when the seal (pointing down) hits the page
    if T < tw0:
        return rest[0], rest[1], 0.0, g0
    if T < tw1:
        k = smootherstep(tw0, tw1, T)
        g = lerp2(rest, (rest[0] + 16, rest[1] - 40), math.sin(k * math.pi))
        return g[0], g[1], 2 * math.pi * ease_in_out(k, 3), g0
    if T < up1:
        k = smootherstep(tw1, up1, T)
        g = lerp2(rest, hi, k)
        return g[0], g[1], math.pi * ease_in_out(k, 2), g0
    if T < B.seal:
        k = ((T - up1) / (B.seal - up1)) ** 2
        g = lerp2(hi, down, k)
        return g[0], g[1], math.pi, g0
    if T < B.seal + 0.36:
        return down[0], down[1] + 5.0 * math.exp(-(T - B.seal) / 0.05), math.pi, g0
    t_r = B.seal + 0.36
    if T < t_r + 0.62:
        k = smootherstep(t_r, t_r + 0.62, T)
        g = lerp2(down, rest, k)
        return g[0], g[1], math.pi * (1 - k), g0
    return rest[0], rest[1], 0.0, g0


def paw_r_goal(T, B):
    """(desired point, what it is ('palm'|'pinch'), shape) for the right paw (screen left)."""
    if T < B.my - 0.35:
        return SIDE_R, "palm", "relax"
    if T < B.my + 0.1:
        return lerp2(SIDE_R, HEART, smootherstep(B.my - 0.35, B.my + 0.1, T)), "palm", "palm"
    if T < B.else_ - 0.1:
        return HEART, "palm", "palm"
    if T < B.yellow + 0.05:
        return lerp2(HEART, SIDE_R, smootherstep(B.else_ - 0.1, B.else_ + 0.5, T)), "palm", "relax"
    if T < B.land:
        k = smootherstep(B.yellow + 0.05, B.yellow + 0.7, T)
        return lerp2(SIDE_R, PALM, k), "palm", "open" if k > 0.35 else "relax"
    if T < B.elev[0]:
        dip = 7.0 * math.exp(-(T - B.land) / 0.07) * max(0.0, math.sin((T - B.land) * 38))
        return (PALM[0], PALM[1] + dip), "palm", "open"
    k = smootherstep(B.elev[0], B.elev[1], T)
    if k < 0.3:
        return lerp2(PALM, ELEV, k), "palm", "open"
    return lerp2(PALM, ELEV, k), "pinch", "pinch"


def lean_angles(T, B):
    """slab tilt toward camera (phi) and in-plane lean (psi, negative = toward the viewer's left)."""
    if T < B.lean0:
        return 0.0, 0.0
    k = smootherstep(B.lean0, B.lean1, T)
    d = max(0.0, T - B.lean1)
    over = 0.05 * math.exp(-d / 0.25) * math.sin(d * 12.0)
    return math.radians(21.0) * k + over * k, math.radians(-6.0) * k


def slab_matrix(T, B):
    """rotation of the Emperor's slab + fn(wall points (N,2)) -> 3D (N,3)."""
    phi, psi = lean_angles(T, B)
    R = Rx(-phi) @ Rz(psi)
    H3 = np.array([HINGE[0], HINGE[1], 0.0])

    def fn(xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        d = np.concatenate([xy - HINGE, np.zeros((len(xy), 1))], 1)
        return d @ R.T + H3

    return R, fn


def _solve(fig, base, key, goal, kw, it=3):
    """wrist target so that anchors[key] lands on goal (the IK places the wrist; palms sit beyond it)."""
    tgt = goal
    side = "hand_r" if key in ("hand_r", "pinch") else "hand_l"
    for _ in range(it):
        a = fig.anchors(*base, **dict(kw, **{side: tgt}))
        got = a.get(key) or a[side]
        tgt = (tgt[0] + goal[0] - got[0], tgt[1] + goal[1] - got[1])
    return tgt


def emperor_kw(T, B, st):
    """keyword arguments for Figure('jaguar').draw at global time T (+ the sceptre pose). Pure in T; memoised
    per process (it is asked for by the cartoon, the camera, the lights and the props of the same frame)."""
    memo = st.setdefault("_ekw", {})
    hit = memo.get(T)
    if hit is None:
        if len(memo) > 8:
            memo.clear()
        hit = memo[T] = _emperor_kw(T, B, st)
    return hit


def _emperor_kw(T, B, st):
    fig = P.jaguar()
    base = (EMP["x"], EMP["y"], EMP["h"])
    tl = st["tl"]
    look = (0.0, -0.1)
    if B.sig0 - 0.3 < T < B.sig1 + 0.2:
        look = (0.9, 0.8)                        # watching his own signature
    elif B.hereby[0] - 0.3 < T < B.seal + 0.35:
        look = (0.8, 0.5)
    elif B.yellow < T < B.elev[0] + 0.15:
        look = (-1.0, 0.45)                      # the hand, the coin
    elif T > B.elev[0] + 0.15:
        look = (0.0, 0.0)                        # straight at us
    lean = 0.0
    if B.sig0 - 0.5 < T < B.seal + 0.9:
        lean = 0.028 * smootherstep(B.sig0 - 0.5, B.sig0, T) * (1 - smootherstep(B.seal + 0.3, B.seal + 0.9, T))
    wl, wr = 0.0, 0.0
    for tb in (48.95, 51.95, 55.3, 57.05, 60.1):
        v = float(np.clip(1 - abs(T - tb) / 0.075, 0, 1))
        wl, wr = max(wl, v), max(wr, v)
    if B.wink[0] < T < B.wink[1]:
        k = math.sin(math.pi * (T - B.wink[0]) / (B.wink[1] - B.wink[0]))
        wr = max(wr, min(1.0, 1.6 * k))
    if T < 48.62:
        wl = wr = 0.0
    blink = (wl, wr)
    expr = "proud"
    if T >= B.except_ - 0.12:
        expr = "sly"
        if B.wink[0] - 0.08 < T < B.wink[1] + 0.12:
            expr = "joyful"                      # one eye wide open, the other winks
    elif B.sig0 - 0.2 < T < B.sig1:
        expr = "neutral"
    kw = dict(pose="stand", t=T - 48.5, mouth=tl.mouth("JAGUAR", T), look=look, expr=expr, blink=blink, halo=False,
              lean=lean, pend=st["pend_fn"](T))
    gx, gy, ga, gl = sceptre_pose(T, B, st)
    kw["shape_l"] = "grip"
    kw["hand_l"] = (gx, gy)
    goal_r, what, shape_r = paw_r_goal(T, B)
    kw["shape_r"] = shape_r
    kw["hand_r"] = goal_r
    kw["hand_l"] = _solve(fig, base, "hand_l", (gx, gy), kw)
    kw["hand_r"] = _solve(fig, base, "pinch" if what == "pinch" else "hand_r", goal_r, kw)
    return kw, (gx, gy, ga, gl)


def pend_fn_build(B, st):
    """pendilia: damped pendulums driven by the head's motion (precomputed, then looked up by T)."""
    dt = 1.0 / 240.0
    Ts = np.arange(48.4, 62.6, dt)
    hx = np.zeros(len(Ts))
    for i, T in enumerate(Ts):
        lean = 0.0
        if B.sig0 - 0.5 < T < B.seal + 0.9:
            lean = 0.028 * smootherstep(B.sig0 - 0.5, B.sig0, T) * (1 - smootherstep(B.seal + 0.3, B.seal + 0.9, T))
        R, fn = slab_matrix(T, B)
        p = fn(np.array([[EMP["x"] + lean * 0.9 * EMP["h"], EMP["y"] - 0.9 * EMP["h"]]]))[0]
        jolt = 6.0 * math.exp(-(T - B.seal) / 0.05) if T > B.seal else 0.0
        hx[i] = p[0] + jolt
    acc = np.gradient(np.gradient(hx, dt), dt)
    L = 60.0
    w0 = math.sqrt(2400.0 / L)
    th = np.zeros(len(Ts))
    v = x = 0.0
    for i in range(len(Ts)):
        a = -w0 * w0 * math.sin(x) - 2 * 0.1 * w0 * v - acc[i] / L * 0.7
        v += a * dt
        x += v * dt
        th[i] = x

    def fn(T):
        a = float(np.interp(T, Ts, th))
        return (a, a * 0.92)
    return fn


# ============================================================ the coin and the Vexillian hand
def palm_world(T, B, st):
    """3D position of the right paw's palm (or pinch) point."""
    kw, _ = emperor_kw(T, B, st)
    a = P.jaguar().anchors(EMP["x"], EMP["y"], EMP["h"], **{k: v for k, v in kw.items() if k not in ("mouth",)})
    goal, what, shape = paw_r_goal(T, B)
    p = a.get("pinch") if what == "pinch" and a.get("pinch") else a["hand_r"]
    R, fn = slab_matrix(T, B)
    return fn(np.array([p]))[0]


def hand_matrix(T, B, st):
    """2x3 affine: Vexillian-hand local units -> wall units (its fingertip over the palm when the coin drops)."""
    palm = st["palm_land"]
    tgt = (palm[0] + 2.0, palm[1] - 34.0)
    start = (tgt[0] - 330.0, tgt[1] + 330.0)
    if T < B.hand_in[1]:
        k = smootherstep(B.hand_in[0], B.hand_in[1], T)
        p = lerp2(start, tgt, ease_out(k, 2))
        ang = 0.25 * (1 - k)
    elif T < B.hand_out[0]:
        p, ang = tgt, 0.0
        if T > B.drop:
            p = (tgt[0] + 3.0 * math.sin((T - B.drop) * 30) * math.exp(-(T - B.drop) / 0.1), tgt[1])
    else:
        k = smootherstep(B.hand_out[0], B.hand_out[1], T)
        p = lerp2(tgt, start, ease_in(k, 2))
        ang = 0.3 * k
    c, s = math.cos(ang), math.sin(ang)
    A = np.array([[c, -s], [s, c]])
    b = np.array(p) - A @ np.array(P.HAND_TIP)
    return np.hstack([A, b[:, None]])


def coin_state(T, B, st):
    """(x, y, z, tilt, spin) of the solidus centre in world units."""
    zh = st["palm_land"][2] + 30.0
    if T < B.drop:
        M = hand_matrix(T, B, st)
        tip = M @ np.array([P.HAND_TIP[0], P.HAND_TIP[1], 1.0])
        return float(tip[0]), float(tip[1]), zh, 0.3, 0.0
    pw = palm_world(T, B, st)
    rest_y = pw[1] - 10.0
    if T < B.land:
        k = (T - B.drop) / (B.land - B.drop)
        M = hand_matrix(B.drop, B, st)
        tip = M @ np.array([P.HAND_TIP[0], P.HAND_TIP[1], 1.0])
        x = tip[0] + (pw[0] - tip[0]) * k
        y = tip[1] + (rest_y - tip[1]) * k * k
        z = zh + (pw[2] + 6.0 - zh) * k
        return float(x), float(y), float(z), 0.3 + 0.65 * k, (T - B.drop) * 30.0
    if T < B.elev[0]:
        bounce = 5.0 * math.exp(-(T - B.land) / 0.05)
        return float(pw[0]), float(rest_y - bounce), float(pw[2]) + 6.0, 0.95, 0.0
    k = smootherstep(B.elev[0], B.elev[1] - 0.1, T)
    # held up between two claws by its lower rim, turned to face us
    return (float(pw[0]), float(pw[1] - 10.0 - (P.COIN_R - 14.0) * k), float(pw[2]) + 6.0 + 14.0 * k,
            0.95 * (1 - k), 0.0)


# ============================================================ camera and light
def camera(T, B, st):
    cx, cy, z = track(T, [
        (48.5, (830.0, 380.0, 2.0)),
        (B.and_ - 0.04, (830.0, 372.0, 2.08)),
        (B.else_ + 0.05, (960.0, 540.0, 1.0)),
        (B.else_ + 0.14, (960.0, 540.0, 1.0)),
        (B.sign + 0.32, (995.0, 440.0, 1.55)),
        (B.fortytwo[1], (1008.0, 452.0, 1.6)),
        (B.weapon + 0.3, (1000.0, 445.0, 1.6)),
        (B.seal - 0.02, (985.0, 440.0, 1.5)),
        (B.lean0 + 0.1, (975.0, 445.0, 1.5)),
        (B.lean1 + 0.25, (790.0, 440.0, 1.72)),
        (B.hand_in[1], (745.0, 470.0, 1.78)),
        (B.elev[0], (740.0, 462.0, 1.82)),
    ])
    rot = 0.0
    if T >= B.seal:
        d = T - B.seal
        amp = 9.0 * math.exp(-d / 0.18)
        cx += amp * math.sin(d * 71.0)
        cy += amp * 0.8 * math.sin(d * 57.0 + 1.3)
        z *= 1.0 + 0.035 * math.exp(-d / 0.12)
        rot = 0.004 * math.exp(-d / 0.2) * math.sin(d * 43.0)
    if T >= B.push[0]:
        # two-shot of the raised coin and the winking face, then into the coin
        x, y, zc, _, _ = coin_state(T, B, st)
        hx, hy = st["head_world"](T)
        k1 = smootherstep(B.push[0], B.push[1] - 0.2, T)
        k2 = smootherstep(B.push[1] - 0.2, B.push[1], T)
        mx, my = (x + hx) / 2, (y + hy) / 2 + 12.0
        cx = cx + (mx - cx) * k1
        cy = cy + (my - cy) * k1
        cx = cx + (x - cx) * k2
        cy = cy + (y - cy) * k2
        z = z * (1 - k1) + 3.1 * k1
        z = z + (5.0 - z) * k2 + 0.6 * smootherstep(B.push[1], B.kaching + 0.4, T)
        if T > B.push[1]:
            cx, cy = x, y
    else:
        hw, hh = W / 2 / z, H / 2 / z
        cx = min(max(cx, hw), W - hw)
        cy = min(max(cy, hh), H - hh)
    return cx, cy, z, rot


def lights_at(T, B, V, st):
    """two candle stands below the panel + the pilgrim's candle near the eye (it wakes the gold where we look);
    at the end the world dims and one light finds the solidus."""
    L = mz.candles(T, [(430.0, 1150.0, 600.0), (1500.0, 1150.0, 600.0)], power=1.2, radius=900.0, flicker_amt=0.22,
                   sway=8.0, seed=3)
    e = V.eye
    wob = 60.0 * math.sin(T * 0.4)
    L.append(mz.Light(pos=(e[0] - 300 + wob, e[1] - 200 + 0.5 * wob, e[2] * 0.8), color=(1.0, 0.82, 0.58),
                      power=1.3 * mz.flicker(T, 11, 0.12), radius=e[2] * 1.3))
    L.append(mz.Light(dir=(-0.35, -0.7, 0.62), color=(0.95, 0.9, 0.85), power=0.2))
    k = focus_k(T, B)
    if k > 0:
        for l in L:
            l.power *= 1.0 - 0.62 * k
        x, y, z, _, _ = coin_state(T, B, st)
        L.append(mz.Light(pos=(x - 60.0, y - 110.0, z + 360.0), color=(1.0, 0.88, 0.66), power=2.6 * k, radius=420.0))
    return L


def focus_k(T, B):
    return smootherstep(B.elev[0] + 0.1, B.kaching - 0.05, T)


# ============================================================ cartoons
def court_looks(T, B, st):
    """the court's eyes: a side-eye at the Emperor on "and everybody else!", then everyone glances at the solidus
    as it drops into his paw."""
    looks = {}
    k0 = smootherstep(B.and_ + 0.05, B.and_ + 0.45, T) * (1 - smootherstep(B.else_ + 0.1, B.else_ + 0.5, T))
    if k0 > 0:
        for i, (kind, x, hh, seed, style) in enumerate(P.COURT):
            d = -1.0 if EMP["x"] < x else 1.0
            looks[i] = (d * k0, -0.15 * k0)
    k = smootherstep(B.drop - 0.3, B.drop + 0.05, T) * (1 - smootherstep(B.kaching - 0.55, B.kaching - 0.25, T))
    if k > 0:
        cx = coin_state(T, B, st)[0]
        for i, (kind, x, hh, seed, style) in enumerate(P.COURT):
            d = -1.0 if cx < x else 1.0
            looks[i] = (d * k, 0.45 * k)
    return looks


def court_blinks(T, B):
    """slow, unsynchronised blinks of the court (explicit, so the figures can be cached) - and one reflexive
    blink of the whole court when the seal slams."""
    out = {}
    flinch = float(np.clip(1 - abs(T - (B.seal + 0.06)) / 0.07, 0, 1))
    for i in range(len(P.COURT)):
        period = 3.1 + 0.53 * (i % 5)
        tb = 48.9 + (i * 0.618 % 1.0) * period
        v = flinch
        while tb < 62.6:
            v = max(v, float(np.clip(1 - abs(T - tb) / 0.075, 0, 1)))
            tb += period
        if v > 0:
            out[i] = round(v * 2) / 2
    return out


def wall_cartoon(T, B, st, emperor=True, halo_crack=0.0, s=1.0, layers=LAYERS):
    """the whole wall (wall units at pixel scale s) on every layer -> icon.cartoon dict."""
    ts, prog = st["sig_sched"]
    sig = 0.0 if T < B.sig0 else float(np.interp(T, ts, prog))
    mm = sum(1 for tn in numeral_times(B) if T >= tn)
    upto = sum(1 for tt in st["tit_times"] if T >= tt)
    looks = court_looks(T, B, st)
    blinks = court_blinks(T, B)
    ekw = emperor_kw(T, B, st)[0] if emperor else None
    hx, hy, hr = P.halo_geom()

    def draw(ctx, L):
        P.draw_ground(ctx, L)
        P.draw_titulus(ctx, L, upto)
        P.draw_halo(ctx, hx, hy, hr, L, crack=halo_crack)
        P.draw_court(ctx, L, looks=looks, blinks=blinks)
        P.draw_codex(ctx, L, sig=sig, sig_pts=st["sig_pts"], mm=mm, seal=T >= B.seal)
        P.deacon_hands(ctx, L)
        if ekw is not None:
            P.jaguar().draw(ctx, EMP["x"], EMP["y"], EMP["h"], layer=L, **ekw)

    return I.cartoon(draw, s=s, layers=layers)


def emperor_cartoon(T, B, st, s=1.0):
    kw = emperor_kw(T, B, st)[0]

    def draw(ctx, L):
        P.jaguar().draw(ctx, EMP["x"], EMP["y"], EMP["h"], layer=L, **kw)

    return I.cartoon(draw, s=s, layers=LAYERS)


def prop_cartoon(draw_fn, ext, s):
    def draw(ctx, L):
        draw_fn(ctx, L)
    return I.cartoon(draw, s=s, extent=ext, layers=LAYERS)


def materials(c, lay, alpha=False):
    """per-tile rgb + material codes (+ alpha) from an icon.cartoon dict."""
    rgba = c["rgba"]
    rgb = mz.sample(rgba if alpha else c["rgb"], lay)
    mat = np.full(len(lay), mz.GLASS, np.uint8)
    mat[mz.sample_mask(c["gold"], lay) >= 0.5] = mz.GOLD
    mat[mz.sample_mask(c["pearl"], lay) >= 0.5] = mz.PEARL
    a = mz.sample_mask(c["alpha"], lay, mode="centre") if alpha else None
    return rgb, mat, a


# ============================================================ setup
def _prop_layout(draw_fn, ext, tile, seed, s, rings=None):
    c = prop_cartoon(draw_fn, ext, s)
    if rings is not None:
        cx, cy, r = rings
        lay = mz.Layout.rings(cx, cy, 0.0, r, tile=tile, seed=seed, extent=ext)
    else:
        lay = mz.Layout.flow(c["rgba"], tile=tile, seed=seed, extent=ext, gold=c["gold"], pearl=c["pearl"])
    keep = mz.sample_mask(c["alpha"], lay, mode="centre") >= 0.35
    return lay.subset(keep)


def setup(S):
    tl = S.tl
    B = Beats(tl)
    st = dict(tl=tl, B=B)
    st["sig_pts"] = P.sig_path()
    st["sig_sched"] = sig_schedule(B)
    st["pend_fn"] = lambda T: (0.0, 0.0)
    n_tit = P.TITULUS.n
    st["tit_times"] = titulus_times(B, n_tit)

    # ---- the wall layout (cached on disk by vx.mosaic)
    rest_T = 49.9
    c = wall_cartoon(rest_T, B, st)
    guide = c["rgb"]
    ts = P.tile_size_map(1.0, paw_spots=(HEART, ELEV, PALM, REST_L, SIDE_R))
    gold = c["gold"]
    ground = cv2.erode((gold > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ground[:P.MAIN[0] + 4] = False
    ground[P.MAIN[1] - 4:] = False
    lab = Canvas(s=1.0)
    P.TITULUS.draw_labels(lab.ctx)
    grp = mz.labels_from(lab.rgb())
    lab2 = Canvas(s=1.0)
    P.CODEX_DATE.draw_labels(lab2.ctx)
    g2 = mz.labels_from(lab2.rgb())
    grp[g2 >= 0] = 100 + g2[g2 >= 0]
    tile = 10.5

    def dens(x, y):
        ix = np.clip(np.asarray(x).astype(int), 0, W - 1)
        iy = np.clip(np.asarray(y).astype(int), 0, H - 1)
        return tile / ts[iy, ix]

    lay = mz.Layout.flow(guide, tile=tile, seed=31, density=dens, gold=gold, pearl=c["pearl"], ground=ground, echo=2,
                         groups=grp)
    st["lay"] = lay
    n = len(lay)

    # ---- the Emperor's slab: his silhouette at the lean (the socket) and everything his paws reach later
    st["pend_fn"] = pend_fn_build(B, st)
    st.pop("_ekw", None)                    # poses memoised before the pendilia existed
    hole = emperor_cartoon(B.lean0, B, st)["alpha"]
    union = hole.copy()
    for T in np.arange(B.lean0, 62.5, 0.1):
        union = np.maximum(union, emperor_cartoon(T, B, st)["alpha"])
    union = cv2.dilate((union > 0.2).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    st["hole"] = mz.sample_mask(hole, lay, mode="centre") >= 0.5
    st["slab_idx"] = np.nonzero(mz.sample_mask(union, lay, mode="centre") >= 0.5)[0]

    # ---- per-tile schedules
    xy = lay.xy.astype(np.float64)
    eyes = P.jaguar().anchors(EMP["x"], EMP["y"], EMP["h"])["eyes"]
    ex, ey = (eyes[0][0] + eyes[1][0]) / 2, (eyes[0][1] + eyes[1][1]) / 2
    st["eye_c"] = (ex, ey)
    d_eye = np.minimum(np.hypot(xy[:, 0] - eyes[0][0], (xy[:, 1] - eyes[0][1]) * 1.1),
                       np.hypot(xy[:, 0] - eyes[1][0], (xy[:, 1] - eyes[1][1]) * 1.1))
    rnd = lay.rnd.astype(np.float64)
    st["t_flip"] = 48.64 + 1.55 * (d_eye / 1450.0) ** 0.8 + 0.1 * (rnd - 0.5)
    st["t_cryst"] = 48.5 + 0.24 * (d_eye / 1450.0) ** 0.6 + 0.03 * rnd
    t_set = np.full(n, -1.0)
    for i, tt in enumerate(st["tit_times"]):
        t_set[lay.grp == i] = tt
    for j, tn in enumerate(numeral_times(B)):
        t_set[lay.grp == 100 + j] = tn
    st["t_set"] = t_set
    pts, u = st["sig_pts"]
    ts_, prog = st["sig_sched"]
    t_of_u = np.interp(u, prog, ts_)
    x0, y0, bw, bh = P.SIG_BOX
    near = np.nonzero((xy[:, 0] > x0 - 10) & (xy[:, 0] < x0 + bw + 10) & (xy[:, 1] > y0 - 12)
                      & (xy[:, 1] < y0 + bh + 14))[0]
    t_sig = np.full(n, -1.0)
    for i in near:
        d = np.hypot(pts[:, 0] - xy[i, 0], pts[:, 1] - xy[i, 1])
        j = int(np.argmin(d))
        if d[j] < 4.0:
            t_sig[i] = t_of_u[j]
    st["t_sig"] = t_sig
    d_seal = np.hypot(xy[:, 0] - P.SEAL["cx"], xy[:, 1] - P.SEAL["cy"])
    rng = np.random.default_rng(7)
    st["t_jump"] = B.seal + d_seal / 2300.0 + rng.uniform(0, 0.03, n)
    st["jumper"] = (rng.random(n) < 0.42) | (d_seal < 75)
    st["a_jump"] = (6.0 + 18.0 * np.exp(-d_seal / 220.0)) * (0.5 + rng.random(n))
    st["r_jump"] = (0.18 + 0.45 * np.exp(-d_seal / 320.0)) * rng.choice([-1.0, 1.0], n)
    st["d_seal"] = d_seal
    st["in_seal"] = d_seal < P.SEAL["r"] + 2
    x0c, y0c, wc, hc = P.codex_rect()
    cand = np.nonzero((d_seal > 150) & (d_seal < 280) & (lay.mat == mz.GOLD) & (xy[:, 1] < y0c))[0]
    pop = rng.choice(cand, size=min(14, len(cand)), replace=False) if len(cand) else np.zeros(0, int)
    st["pop"] = pop
    st["pop_v"] = np.stack([(xy[pop, 0] - P.SEAL["cx"]) / np.maximum(d_seal[pop], 1) * rng.uniform(180, 420, len(pop)),
                            -rng.uniform(250, 520, len(pop)), rng.uniform(200, 420, len(pop))], 1)
    st["pop_spin"] = rng.normal(0, 14, (len(pop), 3))

    # ---- props: the stylus-sceptre, the solidus, the Vexillian hand
    st["sceptre"] = _prop_layout(P.draw_sceptre, P.SCEPTRE_EXT, 3.0, 5, s=6.0)
    cc = P.COIN_EXT[0] / 2
    st["coin"] = _prop_layout(P.draw_coin, P.COIN_EXT, 2.1, 6, s=14.0, rings=(cc, cc, P.COIN_R + 0.5))
    st["hand"] = _prop_layout(P.draw_vex_hand, P.HAND_EXT, 3.6, 8, s=3.0)
    st["palm_land"] = palm_world(B.land, B, st)

    def head_world(T):
        kw, _ = emperor_kw(T, B, st)
        hd = P.jaguar().anchors(EMP["x"], EMP["y"], EMP["h"], **{k: v for k, v in kw.items() if k != "mouth"})["head"]
        p = slab_matrix(T, B)[1](np.array([hd]))[0]
        return float(p[0]), float(p[1])
    st["head_world"] = head_world

    _export_foley(S, B, st)
    return st


# ============================================================ foley / music hooks
def signature_strokes(st):
    """the signature's pen strokes (global seconds): the flourish is written as five gestures - the initial loop,
    'ag', the 'u' loop, 'ar', and the underline swoosh (boundaries at control points 3, 10, 13, 20)."""
    pts, u = st["sig_pts"]
    ts, prog = st["sig_sched"]
    t_of_u = np.interp(u, prog, ts)
    per = (len(pts) - 1) // (len(P.SIG_PTS) - 1)
    cuts = [0] + [k * per for k in (3, 10, 13, 20)] + [len(pts) - 1]
    return [(float(t_of_u[a]), float(t_of_u[b])) for a, b in zip(cuts, cuts[1:])]


def _export_foley(S, B, st):
    ev = [
        dict(t=48.5, kind="flip_wave", strength=0.7, pan=-0.1,
             desc="the pastel static crystallises into tesserae; they flip, tile by tile from the Jaguar's eyes "
                  "outward, into glass and gold (to 50.25)"),
        dict(t=48.64, kind="resolve", strength=0.6, pan=-0.1,
             desc="first tiles flip: the Jaguar's two eyes stare out of the static; the visible frame is gold by "
                  "~49.4, the whole wall by 50.25 (to 50.25)"),
        dict(t=50.25, kind="resolve", strength=0.4, pan=0.0, desc="the last tesserae of the wall set (resolve end)"),
        dict(t=B.my - 0.35, kind="rustle", strength=0.25, pan=-0.1, desc="paw on heart"),
        dict(t=B.tit[0], kind="click", strength=0.35, pan=-0.6,
             desc=f"titulus HIC IMPERATOR IAGVAR LEGEM SANCIT sets letter by letter, left to right (to {B.tit[1]:.2f})"),
        dict(t=B.sig0, kind="stylus", strength=0.9, pan=0.2,
             desc=f"the golden nib touches the purple codex: a flourishing signature in gold (to {B.sig1:.2f})"),
        dict(t=B.sig0, kind="stroke", strength=0.8, pan=0.2, desc=f"signature flourish (to {B.sig1:.2f})"),
        dict(t=B.weapon + 0.05, kind="whoosh", strength=0.5, pan=0.2,
             desc=f"he twirls the stylus-sceptre a full turn (to {B.weapon + 0.95:.2f})"),
        dict(t=B.weapon + 0.95, kind="whoosh", strength=0.4, pan=0.25,
             desc=f"he raises it seal-end down above the page (to {B.seal - 0.11:.2f})"),
        dict(t=B.seal, kind="seal", strength=1.0, pan=0.25,
             desc="the gold seal SLAMS onto the edict (chrysobull, paw-print impression)"),
        dict(t=B.seal + 0.01, kind="tiles_jump", strength=0.9, pan=0.1,
             desc=f"shock ring: every tessera on the wall jumps and resettles; 14 pop out and fall (to {B.seal + 0.9:.2f})"),
        dict(t=B.lean0, kind="rustle", strength=0.6, pan=-0.2,
             desc=f"the Emperor leans OUT of the wall toward us (stone slab grinding, cloth), leaving his halo "
                  f"(to {B.lean1:.2f})"),
        dict(t=B.hand_in[0], kind="whoosh", strength=0.35, pan=-0.8,
             desc=f"an ochre Vexillian sleeve reaches in from the lower left with a gold solidus (to {B.hand_in[1]:.2f})"),
        dict(t=B.land, kind="coin_drop", strength=0.8, pan=-0.35, desc="the solidus drops into his open paw"),
        dict(t=B.land, kind="coin", strength=0.8, pan=-0.35, desc="coin lands in the paw"),
        dict(t=B.land + 0.02, kind="crack", strength=0.3, pan=-0.1, desc="a hairline crack runs through the halo"),
        dict(t=B.hand_out[0], kind="whoosh", strength=0.25, pan=-0.8, desc="the sleeve withdraws"),
        dict(t=B.elev[0], kind="whoosh", strength=0.3, pan=0.0,
             desc=f"he elevates the coin like the Host; the camera pushes into it (to {B.elev[1]:.2f})"),
        dict(t=B.wink[0], kind="click", strength=0.2, pan=0.0, desc="he winks"),
        dict(t=B.kaching, kind="ka_ching", strength=1.0, pan=0.0,
             desc="the solidus glints: a four-point star flares; the world drains to one gold point (to 62.45)"),
        dict(t=B.kaching, kind="flare", strength=1.0, pan=0.0, desc="glint flare -> gold point (hand-off to m04)"),
    ]
    for j, tn in enumerate(numeral_times(B)):
        ev.append(dict(t=tn, kind="click", strength=0.3, pan=0.3, desc=f"numeral {'MMXLII'[j]} of MMXLII sets in gold"))
    for t0, t1 in signature_strokes(st):
        ev.append(dict(t=t0, kind="stylus", strength=0.55, pan=0.2, desc=f"pen stroke (to {t1:.2f})"))
    export_events(S, "sfx", ev)
    t_flip = st["t_flip"]
    Tn = S.start + np.arange(S.n) / S.fps
    xs = st["lay"].xy[:, 0]
    cnt = np.zeros(S.n, np.float32)
    pan = np.zeros(S.n, np.float32)
    for f, T in enumerate(Tn):
        m = (t_flip >= T - 0.5 / S.fps) & (t_flip < T + 0.5 / S.fps)
        cnt[f] = m.sum()
        if cnt[f] > 0:
            pan[f] = screen_pan(float(np.mean(xs[m])))
    export_track(S, "flips", cnt / max(1.0, cnt.max()), pan=pan, kind="tiles", flips=cnt)


# ============================================================ render
def render(fc, st):
    B = st["B"]
    T = fc.T
    lay = st["lay"]
    n = len(lay)
    cx, cy, z, rot = camera(T, B, st)
    V = mz.View(cx, cy, z, rot, eye=1.3 * W / min(z, 1.9))
    L = lights_at(T, B, V, st)
    amb = 0.2 * (1.0 - 0.6 * focus_k(T, B))
    detached = T >= B.lean0

    # ---- the wall cartoon -> per-tile colour + material
    crack = smootherstep(B.land, B.land + 0.12, T)
    c = wall_cartoon(T, B, st, emperor=not detached, halo_crack=crack)
    rgb, mat, _ = materials(c, lay)
    gold = mat == mz.GOLD
    gain = np.ones(n, np.float32)
    present = np.ones(n, bool)
    poses = []

    # ---- the static resolve: screen faces split-flap over to glass and gold
    t_flip = st["t_flip"]
    dur = 0.15
    static_img = None
    if T < 50.4:
        f = mz.fx.flip(T, t_flip, dur=dur, size=lay.size, hop=0.12)
        old = ~f["new"]
        if old.any():
            static_img = np.asarray(pastel_static(fc.w, fc.h, fc.s, T), np.float32)[..., :3]
            scr = V.plane_to_screen(lay.xy.astype(np.float64)) * fc.s
            ix = np.clip(scr[:, 0].astype(int), 0, fc.w - 1)
            iy = np.clip(scr[:, 1].astype(int), 0, fc.h - 1)
            rgb[old] = static_img[iy, ix][old]
            mat[old] = mz.SCREEN
            gain[old & (lay.size < 3.6)] *= 1.6
        poses.append(dict(rot=f["rot"], lift=f["lift"]))
        fresh = np.clip(1 - (T - (t_flip + dur)) / 0.35, 0, 1) * (T >= t_flip + dur)
        gain += (0.9 * gold + 0.35) * fresh
    # ---- letters / numerals flip in (the cartoon shows each glyph from its set time = the flip's midpoint)
    t_set = st["t_set"]
    sel = (t_set > 0) & (np.abs(T - t_set) < 0.07)
    if sel.any():
        f = mz.fx.flip(T, np.where(sel, t_set - 0.07, np.inf), dur=0.14, size=lay.size, hop=0.12)
        poses.append(dict(rot=f["rot"], lift=f["lift"]))
    glow = (t_set > 0) & (T > t_set) & (T < t_set + 0.6)
    gain[glow] += 1.2 * np.exp(-(T - t_set[glow]) / 0.12)
    # ---- the signature ignites tile by tile
    t_sig = st["t_sig"]
    lit = (t_sig > 0) & (T > t_sig) & (T < t_sig + 0.6)
    gain[lit] += 1.6 * np.exp(-(T - t_sig[lit]) / 0.1)
    # ---- the seal: impression flash; the shock ring makes tiles jump out and settle, the rest tremble
    if T >= B.seal:
        gain[st["in_seal"]] += 1.8 * math.exp(-(T - B.seal) / 0.15)
        tj = st["t_jump"]
        active = (T > tj) & (T < tj + 0.45)
        if active.any():
            jm = active & st["jumper"]
            poses.append(mz.fx.pop(T, np.where(jm, tj, np.inf), dur=0.42, height=st["a_jump"],
                                   wobble=np.abs(st["r_jump"]), seed=7))
            tr = active & ~st["jumper"]
            amp = np.where(tr, 1.8 * np.exp(-st["d_seal"] / 700.0) * np.clip(1 - (T - tj) / 0.45, 0, 1), 0.0)
            poses.append(dict(offset=(mz.fx.tremble(T, n, amp=1.0, freq=19.0, seed=5)["offset"]
                                      * amp[:, None]).astype(np.float32)))
        present[st["pop"]] = False
    if detached:
        present[st["hole"]] = False

    pose = mz.fx.combine(*poses)
    img = mz.render(fc, None, lay, rgb=rgb, mat=mat, view=V, light=L, present=present, gain=gain,
                    lift=pose.get("lift"), rot=pose.get("rot"), offset=pose.get("offset"), ambient=amb, t=T)

    # ---- loose things over the wall: the leaning slab, the sceptre, the hand, the coin, popped tiles
    loose = []
    ekw, sp = emperor_kw(T, B, st)
    slab = slab_matrix(T, B) if detached else None
    if detached:
        loose += _slab_tiles(T, B, st)
    loose.append(_sceptre_tiles(T, B, st, sp, slab))
    if B.hand_in[0] <= T <= B.hand_out[1]:
        loose.append(_hand_tiles(T, B, st))
    if T >= B.hand_in[0]:
        loose += _coin_tiles(T, B, st)
    if T >= B.seal:
        loose.append(_pop_tiles(T, B, st, rgb))
    loose = [d for d in loose if d is not None]
    if loose:
        img = mz.render_tiles(fc, _concat_tiles(loose), view=V, light=L, loose=True, bg=img, ambient=amb, t=T)

    # ---- frame 0 is exactly m02's static; the tile structure crystallises out of it
    if static_img is not None and T < 48.8:
        scr = V.plane_to_screen(lay.xy.astype(np.float64))
        cmap = _splat(fc, scr, np.clip((T - st["t_cryst"]) / 0.12, 0, 1))
        img = static_img * (1 - cmap[..., None]) + img * cmap[..., None]

    # ---- the world recedes around the solidus, then its glint flares to a gold point
    k = focus_k(T, B)
    if k > 0:
        q = 8
        yy, xx = np.ogrid[:fc.h // q + 1, :fc.w // q + 1]
        r2 = ((xx * q - fc.w / 2) ** 2 + (yy * q - fc.h / 2) ** 2) / (fc.s * fc.s)
        spot = np.exp(-r2 / (2 * (620.0 - 330.0 * k) ** 2)).astype(np.float32)
        m = cv2.resize(1.0 - 0.7 * k * (1.0 - spot), (spot.shape[1] * q, spot.shape[0] * q),
                       interpolation=cv2.INTER_LINEAR)[:fc.h, :fc.w]
        img = img * m[..., None]
    if T >= B.kaching - 0.04:
        img = _flare(fc, img, T, B)
    return img


def _splat(fc, scr, c):
    """per-tile crystallisation value -> smooth screen map (low-res splat)."""
    k = 8
    h, w = fc.h // k + 1, fc.w // k + 1
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    ix = np.clip((scr[:, 0] * fc.s / k).astype(int), 0, w - 1)
    iy = np.clip((scr[:, 1] * fc.s / k).astype(int), 0, h - 1)
    np.add.at(acc, (iy, ix), c.astype(np.float32))
    np.add.at(cnt, (iy, ix), 1.0)
    m = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)
    m = np.where(cnt > 0, m, cv2.dilate(m, np.ones((3, 3), np.uint8)))
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    return cv2.resize(m, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)[:fc.h, :fc.w]


def _tiles_dict(xy, z, rot, rgb, mat, ang, size, tilt, jit, rnd, gain=None, emit=None):
    n = len(xy)
    return dict(xy=np.asarray(xy, np.float32), z=np.asarray(z, np.float32), rot=np.asarray(rot, np.float32),
                rgb=np.asarray(rgb, np.float32), mat=np.asarray(mat, np.uint8), ang=np.asarray(ang, np.float32),
                size=np.asarray(size, np.float32), tilt=tilt, jit=jit, rnd=rnd,
                gain=np.ones(n, np.float32) if gain is None else np.asarray(gain, np.float32),
                emit=np.zeros((n, 3), np.float32) if emit is None else np.asarray(emit, np.float32))


def _slab_tiles(T, B, st):
    """the Emperor's tiles, tipped out of the wall about his feet, on a mortar backing."""
    lay = st["lay"]
    sub = lay.subset(st["slab_idx"])
    c = emperor_cartoon(T, B, st)
    rgb, mat, alpha = materials(c, sub, alpha=True)
    keep = alpha >= 0.5
    if not keep.any():
        return []
    R, fn = slab_matrix(T, B)
    P3 = fn(sub.xy[keep])
    nk = int(keep.sum())
    Rm = np.broadcast_to(R.astype(np.float32), (nk, 3, 3))
    z = np.maximum(P3[:, 2], 0.0) + 0.8
    tiles = _tiles_dict(P3[:, :2], z, Rm, rgb[keep], mat[keep], sub.ang[keep], sub.size[keep] * 1.03,
                        sub.tilt[keep], sub.jit[keep], sub.rnd[keep])
    # the slab's own setting bed: slightly larger matte tiles just behind every tessera
    back = _tiles_dict(P3[:, :2] - R[:2, 2] * 0.6, z - 0.6, Rm, np.broadcast_to(mz.SC["mortar"], (nk, 3)),
                       np.full(nk, mz.STONE, np.uint8), sub.ang[keep], sub.size[keep] * 1.32,
                       sub.tilt[keep] * 0, sub.jit[keep] * 0, sub.rnd[keep])
    return [back, tiles]


def _prop_rgb(st, key, draw_fn, ext, s, lay):
    cache = st.setdefault("_prop_rgb", {})
    if key not in cache:
        c = prop_cartoon(draw_fn, ext, s)
        rgb, mat, _ = materials(c, lay)
        cache[key] = (rgb, mat)
    return cache[key]


def _place(lay, rgb, mat, M, z, slab=None, gain=None):
    """local prop tiles -> world tile dict. M: 2x3 local -> wall; slab: (R, fn) applied after (props he holds)."""
    xy = lay.xy.astype(np.float64) @ M[:, :2].T + M[:, 2]
    ang = lay.ang + math.atan2(M[1, 0], M[0, 0])
    n = len(xy)
    if slab is not None:
        Rs, fn = slab
        P3 = fn(xy)
        P3[:, 2] += z
        Rm = np.broadcast_to(Rs.astype(np.float32), (n, 3, 3))
    else:
        P3 = np.concatenate([xy, np.full((n, 1), float(z))], 1)
        Rm = np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3))
    return _tiles_dict(P3[:, :2], P3[:, 2], Rm, rgb, mat, ang, lay.size * math.hypot(M[0, 0], M[1, 0]), lay.tilt,
                       lay.jit, lay.rnd, gain)


def _sceptre_tiles(T, B, st, sp, slab):
    lay = st["sceptre"]
    rgb, mat = _prop_rgb(st, "sceptre", P.draw_sceptre, P.SCEPTRE_EXT, 6.0, lay)
    gx, gy, ga, gl = sp
    c, s = math.cos(ga), math.sin(ga)
    A = np.array([[c, -s], [s, c]])
    M = np.hstack([A, (np.array([gx, gy]) - A @ np.array([P.SCEPTRE_GRIP[0], gl]))[:, None]])
    d = _place(lay, rgb, mat, M, 3.0, slab=slab)
    if T < 50.4:
        tf = 48.64 + 1.55 * (math.hypot(gx - st["eye_c"][0], gy - st["eye_c"][1]) / 1450) ** 0.8
        if T < tf + 0.075:
            d["rgb"] = np.tile(np.array(C["slop_cyan"], np.float32) * 0.92, (len(rgb), 1))
            d["mat"] = np.full(len(rgb), mz.SCREEN, np.uint8)
    return d


def _hand_tiles(T, B, st):
    lay = st["hand"]
    pinch = 1.0 if T < B.drop else 1.0 - smootherstep(B.drop, B.drop + 0.08, T)
    key = round(pinch, 1)
    rgb, mat = _prop_rgb(st, ("hand", key), lambda ctx, L: P.draw_vex_hand(ctx, L, pinch=key), P.HAND_EXT, 3.0,
                         lay)
    return _place(lay, rgb, mat, hand_matrix(T, B, st), st["palm_land"][2] + 36.0)


def _coin_tiles(T, B, st):
    lay = st["coin"]
    rgb, mat = _prop_rgb(st, "coin", P.draw_coin, P.COIN_EXT, 16.0, lay)
    x, y, z, tilt, spin = coin_state(T, B, st)
    R = (Rx(tilt) @ Rz(spin)).astype(np.float32)
    cc = P.COIN_EXT[0] / 2
    n = len(lay.xy)
    loc = np.concatenate([lay.xy.astype(np.float64) - cc, np.zeros((n, 1))], 1) @ R.T.astype(np.float64)
    k = focus_k(T, B)
    gain = np.full(n, 1.5 + 0.6 * k, np.float32)
    emit = np.outer(np.full(n, 0.07 + 0.08 * k, np.float32), mz.srgb_to_lin(mz.SC["gold_l"]))
    Rn = np.broadcast_to(R, (n, 3, 3))
    xy = loc[:, :2] + (x, y)
    # a struck coin is one polished disc: flat tesserae (little tilt) on a solid gold backing
    face = _tiles_dict(xy, loc[:, 2] + z, Rn, rgb, mat, lay.ang + spin, lay.size * 1.06, lay.tilt * 0.25, lay.jit,
                       lay.rnd, gain, emit)
    nrm = R[:, 2].astype(np.float64)
    back = _tiles_dict(xy - nrm[:2] * 0.5, loc[:, 2] + z - nrm[2] * 0.5, Rn, np.broadcast_to(mz.SC["gold_d"], (n, 3)),
                       np.full(n, mz.GOLD, np.uint8), lay.ang + spin, lay.size * 1.5, lay.tilt * 0.0, lay.jit,
                       lay.rnd, gain * 0.8)
    return [back, face]


def _pop_tiles(T, B, st, rgb_wall):
    pop = st["pop"]
    if len(pop) == 0:
        return None
    lay = st["lay"]
    d = T - B.seal
    v = st["pop_v"]
    xy0 = lay.xy[pop].astype(np.float64)
    x = xy0[:, 0] + v[:, 0] * d
    y = xy0[:, 1] + v[:, 1] * d + 0.5 * 2600.0 * d * d
    zz = np.maximum(0.0, v[:, 2] * d - 0.5 * 900.0 * d * d) + 1.0
    keep = y < 1150
    if not keep.any():
        return None
    R = mz.axis_angle(st["pop_spin"] * d)
    k = np.nonzero(keep)[0]
    return _tiles_dict(np.stack([x, y], 1)[k], zz[k], R[k], rgb_wall[pop][k], np.full(len(k), mz.GOLD, np.uint8),
                       lay.ang[pop][k], lay.size[pop][k], lay.tilt[pop][k], lay.jit[pop][k], lay.rnd[pop][k])


def _concat_tiles(ds):
    keys = ("xy", "z", "rgb", "mat", "ang", "size", "tilt", "jit", "rnd", "gain", "emit")
    o = {k: np.concatenate([np.asarray(d[k]) for d in ds]) for k in keys}
    rot = np.concatenate([np.asarray(d["rot"], np.float32) for d in ds])
    return mz.Tiles(o["xy"], ang=o["ang"], size=o["size"], rgb=o["rgb"], mat=o["mat"], z=o["z"], rot=rot,
                    gain=o["gain"], emit=o["emit"] if o["emit"].any() else None, tilt=o["tilt"], jit=o["jit"],
                    rnd=o["rnd"])


def _flare(fc, img, T, B):
    """the solidus glint -> a four-point star -> one gold point on black at the screen centre."""
    d = T - B.kaching
    s = fc.s
    cx, cy = fc.w / 2, fc.h / 2
    rise = smootherstep(-0.04, 0.05, d)
    dark = smootherstep(0.03, 0.3, d)
    spike_len = 900.0 * smootherstep(-0.02, 0.08, d) * (1 - smootherstep(0.12, 0.33, d)) + 1e-3
    glow_r = 60.0 + 200.0 * math.exp(-max(0.0, d - 0.06) / 0.09) * rise
    white = np.array([1.0, 0.965, 0.855], np.float32)
    goldc = np.array(col("#F3D27A"), np.float32)
    out = img * (1 - dark)
    # soft glows at quarter resolution
    k = 4
    hs, ws = fc.h // k + 1, fc.w // k + 1
    yy, xx = np.ogrid[:hs, :ws]
    r2 = (((xx * k - cx) / s) ** 2 + ((yy * k - cy) / s) ** 2).astype(np.float32)
    g = (0.75 * np.exp(-r2 / (2 * (glow_r * 0.42) ** 2)) + 0.14 * np.exp(-r2 / (2 * (glow_r * 1.4) ** 2)))
    diag = (np.exp(-(((xx * k - cx) - (yy * k - cy)) / s) ** 2 / (2 * 1.6 ** 2))
            + np.exp(-(((xx * k - cx) + (yy * k - cy)) / s) ** 2 / (2 * 1.6 ** 2))) * np.exp(-np.sqrt(r2) / (spike_len * 0.12))
    lo = cv2.resize((g[..., None] * goldc + 0.3 * diag[..., None] * (white + goldc) * 0.45).astype(np.float32),
                    (ws * k, hs * k), interpolation=cv2.INTER_LINEAR)[:fc.h, :fc.w]
    out += lo * rise
    # the four spikes are separable (full resolution)
    xs = (np.arange(fc.w, dtype=np.float32) - cx) / s
    ys = (np.arange(fc.h, dtype=np.float32) - cy) / s
    hx = np.exp(-np.abs(xs) / (spike_len * 0.35))
    hy = np.exp(-(ys * ys) / (2 * 2.2 ** 2))
    vx_ = np.exp(-(xs * xs) / (2 * 2.2 ** 2))
    vy = np.exp(-np.abs(ys) / (spike_len * 0.35))
    sp = (np.outer(hy, hx) + np.outer(vy, vx_)) * (0.9 * rise)
    out += sp[..., None] * (white * 0.5 + goldc * 0.5)
    # the hot core
    r = int(30 * s) + 2
    y0, y1, x0, x1 = int(cy) - r, int(cy) + r, int(cx) - r, int(cx) + r
    wy, wx = np.ogrid[y0:y1, x0:x1]
    core = np.exp(-(((wx - cx) / s) ** 2 + ((wy - cy) / s) ** 2) / (2 * 4.5 ** 2)).astype(np.float32)
    out[y0:y1, x0:x1] += core[..., None] * white * (1.25 * rise)
    return out.astype(np.float32)


def post(fc, st):
    """m02 hands over neutral static (no bloom/vignette/grain/ca): the look ramps in as the tiles crystallise;
    at the end the vignette and grain leave with the world (only the gold point remains)."""
    B = st["B"]
    T = fc.T
    k = smootherstep(48.5, 48.95, T) * (1 - smootherstep(B.kaching, B.kaching + 0.3, T))
    return dict(bloom=POST["bloom"] * (k if T < 49.0 else 1.0), vignette=POST["vignette"] * k,
                grain=POST["grain"] * k, ca=0.0)

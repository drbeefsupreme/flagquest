"""s08 helper: the performance. Keyframed acting for Lutie and Crocus + the hero Flag's pole path.

All positions are STAGE units: 200 per metre, x right, y DOWN, ground at y = 0 (feet line of the heroes).
All times are GLOBAL seconds (words from timeline.json, see comments).
"""
import math

from vx import ease_in_out, ease_out, ease_out_back, clamp, lerp, smoothstep, seg, pulse

U = 200.0                   # stage units per metre
L_POLE = 2.3 * U            # hero pole length
LUT_H = 1.52 * U
CRO_H = 1.86 * U
X_C = 0.52 * U              # Crocus stands here, facing left
X_L = -0.28 * U             # Lutie's mark, facing right

EASES = {"io": ease_in_out, "o": lambda t: ease_out(t, 2), "i": lambda t: t * t, "back": ease_out_back,
         "lin": lambda t: t, "step": lambda t: 1.0 if t >= 1 else 0.0, "sine": lambda t: 0.5 - 0.5 * math.cos(math.pi * t)}


def _mix(a, b, t):
    if a is None or b is None:
        return b if t >= 1.0 else a
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(x + (y - x) * t for x, y in zip(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        a = a if isinstance(a, dict) else {a: 1.0}
        b = b if isinstance(b, dict) else {b: 1.0}
        out = {}
        for k in set(a) | set(b):
            v = a.get(k, 0.0) * (1 - t) + b.get(k, 0.0) * t
            if v > 1e-3:
                out[k] = v
        return out
    return b if t >= 1.0 else a          # discrete values (hand shapes) switch exactly on their key


class Track:
    """keys: [(T, value[, ease])]; the ease of key k shapes the transition INTO key k.
    value may be a callable f(T) (evaluated at query time) - lets hands follow the pole.
    Strings step on their key, unless blend=True (expressions: str -> {str: 1} dict cross-fades)."""

    def __init__(self, *keys, blend=False):
        cv = (lambda v: {v: 1.0} if isinstance(v, str) else v) if blend else (lambda v: v)
        self.keys = sorted([(k[0], cv(k[1]), k[2] if len(k) > 2 else "io") for k in keys], key=lambda k: k[0])

    def __call__(self, T):
        ks = self.keys
        ev = lambda v: v(T) if callable(v) else v
        if T <= ks[0][0]:
            return ev(ks[0][1])
        for (t0, v0, _), (t1, v1, e) in zip(ks, ks[1:]):
            if T < t1:
                u = EASES[e](clamp((T - t0) / (t1 - t0)))
                return _mix(ev(v0), ev(v1), u)
        return ev(ks[-1][1])


# =========================================================== Lutie's run (173.0 -> 174.85)
RUN_T0, RUN_T1 = 173.0, 174.85
RUN_X0 = -4.9 * U


def lutie_x_run(T):
    u = clamp((T - RUN_T0) / (RUN_T1 - RUN_T0))
    # constant speed, then a skidding deceleration over the last ~35%
    k = 0.62
    v = 1.0 / (k + (1 - k) / 2.0)            # normalised so pos(1) = 1
    if u < k:
        p = v * u
    else:
        w = (u - k) / (1 - k)
        p = v * (k + (1 - k) * (w - w * w / 2))
    return RUN_X0 + (X_L - RUN_X0) * p


RUN_SPEED = 2.4             # rig gait speed matching ~3 m/s top speed for Lutie
RUN_STRIDE = 4 * 14.0 * RUN_SPEED ** 0.6 * LUT_H / 100.0      # stage units per run cycle (vx.chars gait)


def run_phase(T):
    """run cycle phase (cycles) from distance travelled -> no foot sliding"""
    return (lutie_x_run(min(T, RUN_T1)) - RUN_X0) / RUN_STRIDE


def run_bob(T):
    ph = run_phase(T)
    amt = 1 - smoothstep(174.55, 174.9, T)
    return abs(math.sin(ph * 2 * math.pi)) * 0.035 * U * amt


# =========================================================== the Flag's pole (stage units)
def _off(ang, d):
    return math.sin(ang) * d, -math.cos(ang) * d


def pole_point(p, frac):
    """point on pole p=(bx,by,ang) at fraction frac of its length from the base"""
    bx, by, a = p
    dx, dy = _off(a, frac * L_POLE)
    return bx + dx, by + dy


def base_for(point, frac, ang):
    """pole base such that the pole passes through `point` at `frac`"""
    dx, dy = _off(ang, frac * L_POLE)
    return point[0] - dx, point[1] - dy


OFFER_BASE = (-0.06 * U, -0.715 * U)
OFFER_ANG = 0.28


def _offer_pole(T):
    sway = 0.010 * math.sin(T * 1.9) + 0.006 * math.sin(T * 3.1 + 1)
    # finger nudge at the touch, sticky tug while pulling away, spring back at the snap
    tug = 0.012 * pulse(T, 176.42, 0.12) + 0.028 * smoothstep(176.95, 178.4, T) * (1 - smoothstep(178.88, 178.95, T))
    if T > 178.9:
        tt = T - 178.9
        tug += 0.028 * math.exp(-5 * tt) * math.cos(2 * math.pi * 1.6 * tt)
    return (OFFER_BASE[0], OFFER_BASE[1] + 1.5 * math.sin(T * 1.3)), OFFER_ANG + sway + tug


def _run_pole(T):
    x = lutie_x_run(T)
    bob = run_bob(T)
    ang = -0.13 + 0.02 * math.sin(run_phase(T) * 4 * math.pi)
    return (x + 0.16 * U, -0.44 * U - bob), ang


# Crocus pinch point while he holds it (181.4 -> 183.2): arm's length, toward Lutie, top tipped away from his face
def _pinch_pt(T):
    a = (0.12 * U, -1.12 * U)
    b = (0.08 * U, -1.07 * U)
    u = seg(T, 182.45, 183.05, ease_in_out)
    return (lerp(a[0], b[0], u), lerp(a[1], b[1], u))


PINCH_FRAC = 0.30


def _crocus_pole(T):
    tt = max(0.0, T - 181.4)
    swing = 0.09 * math.exp(-2.0 * tt) * math.sin(2 * math.pi * 0.7 * tt + 0.3)     # dangles from two fingers
    ang = -0.07 + swing + 0.03 * seg(T, 182.45, 183.05)
    return base_for(_pinch_pt(T), PINCH_FRAC, ang), ang


def _lutie_back_pole(T):
    """after taking it back: held upright in her back hand"""
    ang = 0.05 + 0.012 * math.sin(T * 1.7)
    hand = (X_L + 0.36 * U, -0.95 * U)
    return base_for(hand, 0.20, ang), ang


def _exasperated_pole(T):
    """L05: 'But Master,' - she plants it; 'all Flags are one Flag!' - she lifts and thrusts it at him"""
    planted, a0 = (X_L - 0.30 * U, -0.36 * U), -0.03 + 0.012 * math.sin(T * 2.3)   # held up like a staff at her side
    thrust, a1 = (-0.18 * U, -0.30 * U), 0.26
    u = seg(T, 180.85, 181.08, lambda t: ease_out(t, 3))
    bx, by = lerp(planted[0], thrust[0], u), lerp(planted[1], thrust[1], u)
    ang = lerp(a0, a1, u)
    jab = 1.2 * math.exp(-((T - 181.17) / 0.08) ** 2)          # the stressed "Flag!"
    return (bx + jab * 0.05 * U, by - jab * 0.015 * U), ang - jab * 0.05


# pole ownership segments: (T0, T1, fn); blended over the gaps between segments
_POLE_SEGS = [
    (173.0, 174.62, _run_pole),
    (175.12, 179.18, _offer_pole),
    (179.5, 181.3, _exasperated_pole),
    (181.45, 182.95, _crocus_pole),
    (183.2, 185.1, _lutie_back_pole),
]


def pole_state(T):
    """(bx, by, ang) in stage units at global time T"""
    segs = _POLE_SEGS
    for i, (t0, t1, fn) in enumerate(segs):
        if T <= t1 and (T >= t0 or i == 0):
            (bx, by), a = fn(T)
            return bx, by, a
        if i + 1 < len(segs) and t1 < T < segs[i + 1][0]:
            ta, tb = t1, segs[i + 1][0]
            u = (T - ta) / (tb - ta)
            u = ease_out_back(u, 1.2) if fn is _run_pole else ease_in_out(u)
            (b0x, b0y), a0 = fn(T if fn is not _run_pole else min(T, ta))
            (b1x, b1y), a1 = segs[i + 1][2](T)
            return lerp(b0x, b1x, u), lerp(b0y, b1y, u), lerp(a0, a1, u)
    (bx, by), a = segs[-1][2](T)
    return bx, by, a


def pole_grip(T, frac):
    return pole_point(pole_state(T), frac)


# =========================================================== Lutie
def _lx(T):
    if T < RUN_T1:
        return lutie_x_run(T)
    return X_L + 0.04 * U * pulse(T, 181.12, 0.3)


HIP_R = (X_L + 0.11 * U, -0.83 * U)
HIP_L = (X_L - 0.07 * U, -0.84 * U)

LUTIE = dict(
    x=_lx,
    turn=Track((173.0, 0.8), (174.6, 0.8), (175.0, 0.58), (184.9, 0.58)),
    pose=Track((173.0, {"run": 1.0}), (174.55, {"run": 1.0}), (174.95, {"stand": 1.0, "offer": 1.0}),
               (179.25, {"stand": 1.0, "offer": 1.0}), (179.5, {"stand": 1.0, "gesture": 1.0}),
               (180.8, {"stand": 1.0, "gesture": 1.0}), (181.05, {"stand": 1.0, "offer": 1.0}),
               (181.5, {"stand": 1.0, "offer": 1.0}), (181.8, {"stand": 1.0}), (183.0, {"stand": 1.0}),
               (183.3, {"stand": 1.0, "hold_flag": 1.0}), (183.8, {"stand": 1.0, "hold_flag": 1.0}),
               (184.05, {"stand": 1.0, "hold_flag": 0.5, "sniff": 1.0}),
               (184.75, {"stand": 1.0, "hold_flag": 0.5, "sniff": 1.0}), (185.0, {"stand": 1.0, "hold_flag": 1.0})),
    expr=Track((173.0, "happy"), (175.0, "proud"), (176.0, "proud"), (176.25, "surprised"),
               (177.3, "surprised"), (178.1, "worried"), (179.1, "worried"), (179.3, "exasperated"),
               (180.25, "exasperated"), (180.5, "awe"), (180.8, "awe"), (180.95, "determined"),
               (181.45, "determined"), (181.65, "proud"), (182.45, "proud"), (182.7, "surprised"),
               (183.35, "sad"), (183.55, "skeptical"), (184.45, "skeptical"), (184.55, "surprised"),
               (184.75, "happy"), blend=True),
    look=Track((173.0, (1.0, 0.0)), (174.9, (1.0, -0.25)), (175.9, (1.0, -0.3)), (176.3, (0.6, 0.35)),
               (177.3, (0.7, 0.2)), (178.0, (1.0, -0.3)), (179.3, (1.0, -0.3)), (180.15, (0.8, -0.3)),
               (180.45, (-0.3, -0.3)), (180.75, (-0.2, -0.25)), (180.92, (1.0, -0.3)), (181.3, (0.8, 0.05)),
               (181.8, (1.0, -0.3)), (183.4, (0.8, 0.1)), (183.55, (0.5, 0.6)), (183.95, (0.4, 0.5)),
               (184.3, (0.0, 0.1)), (184.6, (-0.7, -0.4)), (184.9, (-0.5, -0.2))),
    lean=Track((173.0, 0.16), (174.6, 0.16), (174.85, -0.04, "o"), (175.15, 0.10), (179.2, 0.08),
               (179.5, -0.08, "back"), (180.15, -0.04), (180.7, 0.02), (181.1, 0.13, "o"), (181.5, 0.05),
               (181.85, -0.05), (182.6, -0.05), (183.0, 0.04), (183.4, 0.0)),
    nod=Track((173.0, 0.05), (174.9, 0.0), (175.2, -0.10), (175.8, -0.06), (179.3, 0.0), (179.55, 0.12, "back"),
              (180.15, 0.02), (180.6, 0.08), (181.1, -0.06), (181.85, 0.10), (182.6, 0.08), (183.4, -0.12),
              (183.9, -0.02), (184.2, 0.07, "o"), (184.32, 0.0), (184.45, 0.07, "o"), (184.58, 0.0),
              (184.8, 0.04)),
    tilt=Track((173.0, 0.0), (175.2, 0.08), (175.9, 0.05), (176.4, -0.04), (179.3, 0.0), (179.6, -0.14),
               (180.2, 0.0), (180.6, 0.07), (181.1, -0.04), (181.9, 0.12), (182.4, 0.06), (183.5, -0.06),
               (184.6, 0.1), (185.0, 0.12)),
    blink=Track((173.0, 0.0), (184.15, 0.0), (184.22, 0.65), (184.52, 0.65), (184.56, 0.0)),
    # front hand (near camera)
    hand_r_w=Track((173.0, 1.0), (185.0, 1.0)),
    hand_r=Track((173.0, lambda T: pole_grip(T, 0.30)), (174.62, lambda T: pole_grip(T, 0.30)),
                 (175.1, lambda T: pole_grip(T, 0.20)), (179.3, lambda T: pole_grip(T, 0.20)),
                 (179.52, lambda T: pole_grip(T, 0.33)), (180.85, lambda T: pole_grip(T, 0.33)),
                 (181.05, lambda T: pole_grip(T, 0.40)), (181.45, lambda T: pole_grip(T, 0.40)),
                 (181.8, HIP_R), (183.45, HIP_R), (183.6, (X_L + 0.24 * U, -0.98 * U)),
                 (183.8, (X_L + 0.27 * U, -1.02 * U)), (184.05, (X_L + 0.185 * U, -1.265 * U)),
                 (184.75, (X_L + 0.185 * U, -1.265 * U)), (185.0, (X_L + 0.18 * U, -0.9 * U))),
    shape_r=Track((173.0, "grip"), (181.55, "fist"), (183.45, "open"), (184.0, "cup"), (184.8, "relax")),
    # back hand (far side): the appeal and the sweep over "all Flags" reach toward Crocus
    hand_l_w=Track((173.0, 1.0), (185.0, 1.0)),
    hand_l=Track((173.0, lambda T: pole_grip(T, 0.18)), (174.62, lambda T: pole_grip(T, 0.18)),
                 (175.1, lambda T: pole_grip(T, 0.11)), (179.3, lambda T: pole_grip(T, 0.11)),
                 (179.52, (X_L + 0.33 * U, -1.08 * U), "back"), (180.05, (X_L + 0.31 * U, -1.13 * U)),
                 (180.22, (X_L + 0.25 * U, -1.02 * U)), (180.72, (X_L + 0.34 * U, -1.45 * U), "sine"),
                 (180.95, lambda T: pole_grip(T, 0.33)), (181.45, lambda T: pole_grip(T, 0.33)),
                 (181.8, HIP_L), (182.78, HIP_L), (183.0, lambda T: pole_grip(T, 0.20))),
    shape_l=Track((173.0, "grip"), (179.38, "open"), (180.85, "grip"), (181.55, "fist"), (182.8, "open"),
                  (183.0, "grip")),
)


# =========================================================== Crocus
def _clasp(T):
    return (X_C - 0.07 * U, -0.98 * U)


POLE_HALF = 0.0167 * L_POLE / 2 + 0.5     # pole radius (vx.flag POLE_W) + a film of whatever it is


def _touch_pt(T):
    """fingertip contact on the pole surface (the side facing Crocus) at his chest height (1.30 m)"""
    p = pole_state(min(T, 176.95))
    bx, by, a = p
    fr = ((1.30 * U) + by) / (math.cos(a) * L_POLE)
    x, y = pole_point(p, fr)
    return x + math.cos(a) * POLE_HALF, y + math.sin(a) * POLE_HALF


def fingertip_target(T):
    """where Crocus's fingertip is (stage units) during the sticky gag"""
    c = _touch_pt(T)
    if T < 176.95:
        return c
    u1 = seg(T, 176.95, 177.95, lambda t: t * t * (3 - 2 * t) * 0.85 + t * 0.15)   # slow, suspicious pull
    u2 = seg(T, 177.95, 178.5, ease_in_out)
    p1 = (c[0] + 0.15 * U, c[1] - 0.03 * U)
    p2 = (c[0] + 0.24 * U, c[1] - 0.17 * U)
    x = lerp(c[0], p1[0], u1)
    y = lerp(c[1], p1[1], u1)
    return lerp(x, p2[0], u2), lerp(y, p2[1], u2)


def _finger_hand(T):
    # hand_r target = fingertip minus finger length along the pointing direction (pointing left)
    fx, fy = fingertip_target(T)
    return fx + 0.085 * U, fy + 0.012 * U


def _pinch_hand(T):
    x, y = _pinch_pt(T)
    return x + 0.02 * U, y


CROCUS = dict(
    x=lambda T: X_C,
    pose=Track((173.0, {"stand": 1.0}), (175.95, {"stand": 1.0}), (176.25, {"stand": 1.0, "touch": 1.0}),
               (179.5, {"stand": 1.0, "touch": 1.0}), (180.0, {"stand": 1.0}), (181.1, {"stand": 1.0}),
               (181.4, {"stand": 1.0, "hold_flag": 1.0}), (183.2, {"stand": 1.0, "hold_flag": 1.0}),
               (183.6, {"stand": 1.0})),
    expr=Track((173.0, "serene"), (175.85, "serene"), (176.1, "deadpan"), (176.45, "skeptical"),
               (179.3, "skeptical"), (179.6, "deadpan"), (181.3, "deadpan"), (181.5, "disgust"),
               (181.9, "deadpan"), (182.25, "serene"), (184.0, "serene"), blend=True),
    look=Track((173.0, (-0.4, -0.15)), (174.0, (-0.4, -0.15)), (174.25, (1.0, 0.0)), (175.3, (1.0, 0.1)),
               (175.9, (0.9, 0.45)), (176.95, (0.9, 0.5)), (177.95, (0.4, 0.25)), (178.4, (0.6, 0.0)),
               (179.3, (0.8, 0.1)), (180.2, (1.0, 0.0)), (181.2, (0.8, 0.4)), (182.0, (0.9, 0.2))),
    turn=Track((173.0, 0.3), (174.1, 0.3), (174.45, 0.72), (184.9, 0.72)),
    lean=Track((173.0, 0.0), (175.9, 0.0), (176.35, 0.06), (177.0, 0.05), (177.9, -0.02), (179.3, 0.0),
               (181.2, 0.05), (181.6, -0.05), (182.3, -0.02), (183.0, 0.05), (183.5, 0.0)),
    nod=Track((173.0, 0.10), (174.1, 0.10), (174.4, 0.0), (175.9, 0.0), (176.2, -0.10), (177.9, -0.05),
              (178.4, 0.02), (182.2, 0.02), (182.5, 0.08), (184.2, 0.10)),
    tilt=Track((173.0, 0.0), (176.5, 0.0), (176.8, -0.06), (178.3, -0.08), (179.4, 0.0), (182.3, 0.0),
               (182.6, 0.05)),
    blink=Track((173.0, 0.35), (174.1, 0.35), (174.3, 0.0), (178.93, 0.0), (178.97, 0.8), (179.08, 0.0),
                (180.28, 0.0), (180.45, 0.85), (180.62, 0.85), (180.78, 0.0), (182.1, 0.0), (182.35, 1.0),
                (185.0, 1.0)),
    hand_r_w=Track((173.0, 1.0), (185.0, 1.0)),
    hand_r=Track((173.0, _clasp), (175.95, _clasp), (176.3, lambda T: _finger_hand(T), "io"),
                 (179.35, lambda T: _finger_hand(T)), (179.9, _clasp), (181.05, _clasp),
                 (181.4, _pinch_hand, "o"), (183.2, _pinch_hand), (183.45, (X_C - 0.2 * U, -1.2 * U)),
                 (183.9, _clasp)),
    shape_r=Track((173.0, "relax"), (176.05, "point"), (179.7, "relax"), (181.2, "pinch"), (183.25, "open"),
                  (183.6, "relax")),
    hand_l_w=Track((173.0, 1.0), (185.0, 1.0)),
    hand_l=Track((173.0, (X_C - 0.02 * U, -0.97 * U))),
    shape_l=Track((173.0, "relax")),
)


def sample(perf, T):
    return {k: (v(T) if callable(v) else v) for k, v in perf.items()}

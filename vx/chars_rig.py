"""Rig internals for vx.chars (owned by RigMaster).

Body space is 3D-lite, in units of h/100 (h = full standing height):
    a = forward (the way the character faces), u = up (0 = ground), b = lateral toward the NEAR side
    (the side closest to camera; the character's right when facing +1).
Projection with body yaw phi (turn * pi/2; 0 = front view, pi/2 = profile):
    x_local = a*sin(phi) - b*cos(phi),  y_local = -u,  z (towards camera) = a*cos(phi) + b*sin(phi)
Local screen space is then mirrored by `facing` and scaled by h/100 into world/design coordinates.

Poses are functions (t, kw, spec) -> dict of channel values; blending is per channel
(legs / spine / head / arms) with per-pose masks so e.g. {"walk": 1, "hold_flag": 1} walks with the legs
and holds the flag with the arms. Hand targets are specs resolved against the blended trunk:
    ("sh", fwd, down, out)    relative to that arm's shoulder, in arm-length units
    ("head", fwd, down, out)  relative to the head centre, in head radii
    ("chest", fwd, down, out) relative to the neck base, in h/100 units
    ("hip", fwd, down, out)   relative to the hip centre, in h/100 units
("out" = lateral away from the body midline on that arm's side.)
Flag poses carry a `flag` dict: ang (sagittal lean of the pole, + forward), tilt (lateral lean toward near
side), one of gf (front-hand distance from the pole base) / bu (base height) / bus (base height as a
fraction of shoulder height), and gb (back-hand grip offset along the pole from the front hand, None = free).
The pole always passes exactly through the solved front hand.
"""
import math

from .ease import noise1, hash01, clamp, smoothstep

TAU = math.tau
PHI_DEFAULT = 0.45

# ============================================================ small vector helpers (tuples: fast in CPython)


def v_add(p, q):
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2])


def v_sub(p, q):
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def v_mul(p, k):
    return (p[0] * k, p[1] * k, p[2] * k)


def v_dot(p, q):
    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]


def v_len(p):
    return math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])


def v_lerp(p, q, t):
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t)


def rot_au(p, ang):
    """rotate (a,u,b) in the sagittal plane; + = top goes forward"""
    c, s = math.cos(ang), math.sin(ang)
    return (p[0] * c + p[1] * s, -p[0] * s + p[1] * c, p[2])


def ik3(S, T, l1, l2, hint):
    """two-bone IK in 3D. returns (elbow, end, reached)"""
    d = v_sub(T, S)
    dist = v_len(d)
    if dist < 1e-6:
        d, dist = (0.0, -1.0, 0.0), 1e-6
    u = v_mul(d, 1.0 / dist)
    mx = (l1 + l2) * 0.9985
    mn = abs(l1 - l2) + 1e-3
    reached = mn <= dist <= mx
    dc = min(max(dist, mn), mx)
    E = v_add(S, v_mul(u, dc))
    x = (l1 * l1 - l2 * l2 + dc * dc) / (2 * dc)
    hh = math.sqrt(max(l1 * l1 - x * x, 0.0))
    p = v_sub(hint, v_mul(u, v_dot(hint, u)))
    pl = v_len(p)
    if pl < 1e-6:
        p = v_sub((0.0, 0.0, 1.0), v_mul(u, u[2]))
        pl = v_len(p) or 1.0
    p = v_mul(p, 1.0 / pl)
    return v_add(v_add(S, v_mul(u, x)), v_mul(p, hh)), E, reached


def proj(p, cphi, sphi):
    """body 3D -> local screen (x, y), z towards camera"""
    return (p[0] * sphi - p[2] * cphi, -p[1])


def depth(p, cphi, sphi):
    return p[0] * cphi + p[2] * sphi


# ============================================================ body specs per kind
BASE = dict(
    head_r=10.0, head_y=-89.0, sh_y=-73.0, sh_w=11.0, dep=7.5, chest_w=12.5, waist_w=10.5,
    hip_y=-44.0, hip_w=5.6, arm_u=17.5, arm_f=16.0, hand=4.4, arm_r=4.4,
    leg_u=22.0, leg_l=20.0, foot=8.0, ank=2.4, leg_r=5.0,
    hem_y=-1.5, hem_w=18.0, lean0=0.0,
    eye_w=0.19, eye_h=0.25, eye_y=0.02, eye_sep=0.42, jaw=(0.80, 0.70), nose=0.9, lid0=0.10,
)

SPECS = {
    "lutie": dict(head_r=12.2, head_y=-87.8, sh_y=-71.0, sh_w=10.0, dep=7.0, chest_w=11.0, waist_w=10.0,
                  hip_y=-41.0, hip_w=5.0, arm_u=16.0, arm_f=15.0, hand=4.5, arm_r=4.2, leg_u=20.5, leg_l=18.5,
                  hem_w=16.5, eye_w=0.215, eye_h=0.30, eye_sep=0.43, jaw=(0.80, 0.66), nose=0.7, lid0=0.06),
    "crocus": dict(head_r=8.4, head_y=-91.6, sh_y=-80.0, sh_w=12.2, dep=8.0, chest_w=12.8, waist_w=12.4,
                   hip_y=-49.0, hip_w=6.0, arm_u=21.0, arm_f=19.5, hand=4.3, arm_r=4.6, leg_u=25.0, leg_l=22.0,
                   hem_w=21.0, lean0=0.045, eye_w=0.17, eye_h=0.19, eye_sep=0.40, jaw=(0.74, 0.86), nose=1.35,
                   lid0=0.34),
    "schismmancer": dict(head_r=9.4, head_y=-90.2, sh_y=-75.0, sh_w=13.5, dep=9.5, chest_w=15.0, waist_w=13.0,
                         hip_y=-46.0, hip_w=6.2, arm_u=19.0, arm_f=17.0, hand=4.9, arm_r=5.2, leg_u=23.0,
                         leg_l=21.0, leg_r=5.6, hem_y=-27.0, hem_w=17.5, eye_w=0.16, eye_h=0.19, jaw=(0.82, 0.74),
                         nose=1.0, lid0=0.22),
    "citizen": dict(),
    "cultist": dict(head_r=10.2, head_y=-89.0, sh_y=-73.0, sh_w=11.5, hem_w=19.0, eye_w=0.17, eye_h=0.21,
                    lid0=0.18),
    "acolyte": dict(head_r=10.4, head_y=-89.0, sh_y=-73.0, sh_w=11.0, hem_w=18.0, eye_w=0.18, eye_h=0.23),
    "philosopher": dict(head_r=10.0, head_y=-89.5, sh_y=-74.0, sh_w=12.0, dep=8.0, chest_w=13.0, waist_w=12.5,
                        hip_y=-45.0, arm_r=3.9, hem_y=-17.0, hem_w=15.5, eye_w=0.17, eye_h=0.2, nose=1.3,
                        jaw=(0.78, 0.8), lid0=0.2),
    "jaguar": dict(head_r=14.8, head_y=-84.5, sh_y=-66.0, sh_w=14.5, dep=10.0, chest_w=16.5, waist_w=15.5,
                   hip_y=-40.0, hip_w=6.6, arm_u=16.5, arm_f=15.0, hand=5.3, arm_r=5.0, leg_u=19.5, leg_l=17.8,
                   leg_r=5.8, hem_y=-38.0, hem_w=15.0, eye_w=0.17, eye_h=0.2, eye_sep=0.40, eye_y=0.1,
                   jaw=(0.9, 0.72), lid0=0.12),
}
# film 2: realistic Chick-tract proportions (~7.5 heads tall)
_TRACT = dict(head_r=7.9, head_y=-92.1, sh_y=-77.2, sh_w=11.6, dep=7.4, chest_w=12.0, waist_w=10.6, hip_y=-50.0,
              hip_w=5.6, arm_u=20.5, arm_f=18.5, hand=3.2, arm_r=3.7, leg_u=25.5, leg_l=23.0, leg_r=4.6, foot=7.4,
              ank=2.1, hem_w=16.0, eye_w=0.21, eye_h=0.14, eye_sep=0.4, eye_y=0.04, jaw=(0.78, 0.86), nose=1.1,
              lid0=0.12)
SPECS["summer"] = dict(_TRACT, head_r=8.3, head_y=-91.7, sh_w=10.2, chest_w=10.6, waist_w=9.0, hip_w=5.6,
                       dep=6.8, arm_r=3.2, leg_r=4.2, hand=2.95, jaw=(0.76, 0.8), nose=0.8, lid0=0.0, eye_w=0.23,
                       eye_h=0.155)
SPECS["raven"] = dict(SPECS["summer"], head_r=8.1, head_y=-91.9, jaw=(0.74, 0.84), nose=0.9, lid0=0.42)
SPECS["crow"] = dict(_TRACT, head_r=8.0, sh_w=12.6, chest_w=13.4, waist_w=12.8, dep=8.4, arm_r=4.3, hem_y=-24.0,
                     hem_w=15.5, jaw=(0.82, 0.9), nose=1.35, lid0=0.2)
SPECS["flagmaker"] = dict(_TRACT, head_r=7.2, head_y=-92.8, sh_y=-78.4, sh_w=12.2, hip_y=-50.5, arm_u=21.5,
                          arm_f=19.5, hem_w=19.5, jaw=(0.74, 0.9), nose=1.25, lid0=0.3, eye_w=0.18)
SPECS["hippie"] = dict(_TRACT, head_r=8.0)
SPECS["pharisee"] = dict(_TRACT, head_r=7.8, sh_w=12.5, chest_w=13.0, waist_w=12.8, dep=8.2, hem_w=20.0,
                         nose=1.5, jaw=(0.78, 0.92), lid0=0.3)
SPECS["devil"] = dict(_TRACT, head_r=8.6, head_y=-91.4, sh_w=12.8, chest_w=13.6, waist_w=12.0, hem_w=19.0,
                      jaw=(0.7, 1.0), nose=1.3, eye_w=0.2, eye_h=0.13, lid0=0.15)
SPECS["seraph"] = dict(_TRACT, head_r=7.4, head_y=-92.6, sh_y=-78.0, hem_w=19.0, jaw=(0.74, 0.84), lid0=0.3)
for _k, _jk in (("crow", 1.5), ("flagmaker", 1.1), ("pharisee", 1.4), ("devil", 1.6), ("summer", 0.35), ("raven", 0.55),
                ("seraph", 0.6), ("hippie", 1.0)):
    SPECS[_k]["jawk"] = _jk
SPECS["commander"] = dict(SPECS["schismmancer"])
SPECS["crackpot"] = dict(SPECS["philosopher"])


def make_spec(kind, seed, style):
    sp = dict(BASE)
    sp.update(SPECS.get(kind, {}))
    rnd = lambda k: hash01(seed * 7919 + k, 1234)  # noqa: E731
    build = style.get("build")
    if kind in ("citizen", "cultist", "acolyte", "schismmancer", "commander", "philosopher", "crackpot", "hippie",
                "pharisee"):
        if build is None:
            build = 0.82 + 0.5 * rnd(1) ** 1.4 if kind != "schismmancer" else 0.95 + 0.25 * rnd(1)
        hr = 1.0 + (rnd(2) - 0.5) * 0.16
        sp["head_r"] *= hr
        sp["head_y"] = -100.0 + sp["head_r"]
        sp["sh_y"] += (rnd(3) - 0.5) * 3.0
        sp["eye_w"] *= 0.9 + 0.2 * rnd(4)
    build = 1.0 if build is None else build
    face_ = style.get("face", {"summer": "cartoon", "raven": "cartoon"}.get(kind))
    if (face_ == "cartoon" or (face_ == "anime" and kind in ("summer", "raven"))) and kind in SPECS \
            and "jawk" in SPECS[kind]:
        # film-1 cartoon construction: bigger round head, big expressive eyes, simple features
        sp["head_r"] = sp["head_r"] * 1.32
        sp["head_y"] = -100.0 + sp["head_r"]
        sp["sh_y"] = sp["head_y"] + sp["head_r"] * 1.72
        for k_, v_ in (("eye_w", 0.2), ("eye_h", 0.26), ("eye_sep", 0.42), ("eye_y", 0.02), ("jaw", (0.8, 0.7)),
                       ("nose", 0.9), ("lid0", {"raven": 0.32, "crow": 0.2, "devil": 0.12, "flagmaker": 0.28,
                                                "pharisee": 0.25}.get(kind, 0.06))):
            sp[k_] = v_
        if kind in ("summer", "raven", "seraph"):
            sp["eye_w"], sp["eye_h"], sp["jaw"], sp["nose"] = 0.215, 0.29, (0.8, 0.66), 0.7
    for k in ("sh_w", "chest_w", "waist_w", "dep", "hip_w", "hem_w", "arm_r", "leg_r"):
        f = build if k not in ("sh_w",) else 1 + (build - 1) * 0.55
        if k in ("waist_w", "dep"):
            f = 1 + (build - 1) * 1.3
        sp[k] *= f
    sp["build"] = build
    sp["sh_y"] -= 2.2  # short, stylised necks
    sp["neck"] = max(1.0, (-sp["sh_y"]) - (-sp["head_y"]) + sp["head_r"] * 0 - sp["head_r"])  # gap
    sp["neck"] = (-sp["head_y"]) - sp["head_r"] - (-sp["sh_y"])
    sp["L"] = sp["arm_u"] + sp["arm_f"]
    return sp


# ============================================================ expressions
EXPR_DEFAULT = dict(by=0.0, bi=0.0, ba=0.0, lt=0.0, lb=0.0, es=1.0, ps=1.0, sm=0.0, mw=1.0, mo=0.0, sk=0.0,
                    wn=0.0, bl=0.0, th=0.0, sq=0.0, tears=0.0, sweat=0.0, poss=0.0, fold=0.0, gape=0.0,
                    wob=0.0, glow=0.0, shock=0.0)
EXPR_TABLE = {
    "neutral": dict(),
    "happy": dict(by=0.18, sm=0.85, lb=0.38, bl=0.55, mo=0.10, th=0.5),
    "smile": dict(by=0.08, sm=0.55, lb=0.18, bl=0.25),
    "surprised": dict(by=0.95, es=1.18, lt=-0.1, ps=0.72, mo=0.38, mw=0.62),
    "worried": dict(by=0.35, bi=0.95, sm=-0.38, mw=0.8, lt=0.04),
    "determined": dict(by=-0.28, bi=-0.55, lt=0.28, sm=-0.12, mw=0.92),
    "angry": dict(by=-0.45, bi=-1.0, lt=0.28, sm=-0.62, ps=0.82, sq=0.3),
    "furious": dict(by=-0.5, bi=-1.1, lt=0.2, sm=-0.4, ps=0.75, mo=0.55, th=1.0, mw=1.15, sq=0.35),
    "serene": dict(lt=0.46, sm=0.32, by=0.06, bi=0.15),
    "awe": dict(by=0.72, bi=0.25, es=1.12, ps=1.28, mo=0.2, mw=0.66, lt=-0.06),
    "sad": dict(bi=1.0, by=0.1, sm=-0.62, lt=0.32, mw=0.85),
    "smug": dict(lt=0.42, sm=0.55, sk=0.65, by=-0.05, ba=0.45),
    "wink": dict(sm=0.7, sk=0.45, lb=0.25, wn=1.0, ba=0.3, th=0.3),
    "disgust": dict(bi=-0.55, by=-0.18, sm=-0.55, lb=0.5, sk=-0.55, lt=0.3, mw=0.9, sq=0.5),
    "skeptical": dict(ba=0.95, lt=0.34, sm=-0.18, sk=0.45, by=-0.05),
    "exasperated": dict(by=0.55, bi=0.65, lt=0.46, sm=-0.42, mo=0.08),
    "deadpan": dict(lt=0.5, sm=0.0, by=-0.06, mw=0.9),
    "proud": dict(lt=0.3, sm=0.5, by=0.22, lb=0.2),
    # film 2 (Chick-tract acting)
    "horror": dict(by=0.95, bi=0.9, es=1.28, ps=0.36, mo=0.55, mw=0.9, th=0.7, lt=-0.12, sweat=1.0, fold=0.7, sm=-0.45,
                   gape=1.0, shock=1.0),
    "terrified": dict(by=1.0, bi=1.0, es=1.3, ps=0.34, mo=0.45, mw=0.8, th=0.8, lt=-0.12, sweat=1.0, tears=0.5, fold=0.6,
                      sm=-0.5, gape=0.8, shock=1.0, wob=0.6),
    "tearful": dict(bi=1.05, by=0.3, sm=-0.5, tears=1.0, lt=0.2, mw=0.8, lb=0.3, fold=0.3, wob=0.7),
    "crying": dict(bi=1.15, by=0.35, sm=-0.7, tears=1.0, lt=0.55, mo=0.25, mw=0.9, th=0.5, lb=0.4, fold=0.5, wob=1.0,
                   gape=0.3),
    "beatific": dict(by=0.35, bi=0.35, sm=0.62, lt=0.97, bl=0.3, fold=0.2, glow=1.0),
    "sardonic": dict(ba=1.1, lt=0.55, sm=0.4, sk=1.0, by=-0.1, fold=0.3),
    "possessed": dict(poss=1.0, es=1.3, ps=0.22, by=-0.35, bi=-1.25, sm=0.95, th=1.0, mo=0.3, mw=1.45, fold=1.0,
                      lt=-0.1),
    "fervent": dict(by=-0.15, bi=-0.85, es=1.12, ps=0.8, mo=0.3, th=0.85, mw=1.12, fold=0.7),
    "grave": dict(bi=0.25, by=-0.15, lt=0.32, sm=-0.32, fold=0.45),
    "stern": dict(by=-0.4, bi=-0.75, lt=0.25, sm=-0.4, mw=0.9, fold=0.5, sq=0.2),
    "bewildered": dict(by=0.85, ba=0.45, es=1.12, ps=0.8, mo=0.14, sm=-0.1, sweat=0.7),
    "rapt": dict(by=0.42, es=1.06, ps=1.12, mo=0.08, lt=0.0, bi=0.2),
    "cheerful": dict(sm=0.95, by=0.25, lb=0.35, th=0.7, mo=0.08, bl=0.3, fold=0.3),
    "disdain": dict(lt=0.52, by=-0.1, ba=0.35, sm=-0.2, sk=0.35, mw=0.9),
    "shocked": dict(by=1.0, bi=0.5, es=1.2, ps=0.55, mo=0.5, mw=0.7, th=0.3, lt=-0.1, sweat=0.5),
}


def blend_expr(expr):
    e = dict(EXPR_DEFAULT)
    if isinstance(expr, dict):
        tot = sum(expr.values()) or 1.0
        for k in e:
            e[k] = sum(w * EXPR_TABLE.get(n, {}).get(k, EXPR_DEFAULT[k]) for n, w in expr.items()) / tot
    else:
        e.update(EXPR_TABLE.get(expr, {}))
    return e


# ============================================================ pose library
CH_LEGS = ("hip", "ff", "fb", "spread", "sit", "hem_lift", "seat", "cross")
CH_SPINE = ("lean", "chest", "sway", "shrug", "air", "flap")
CH_HEAD = ("head", "yell")
CH_ARMS = ("hf", "hb", "ef", "eb")

DEFAULT = dict(
    hip=(0.0, 0.0), ff=(3.0, 0.0, 0.0), fb=(-2.2, 0.0, 0.0), spread=1.0, sit=0.0, hem_lift=0.0, seat=0.0, cross=0.0,
    lean=0.0, chest=0.0, sway=0.0, shrug=0.0, air=0.0, flap=0.0,
    head=(0.0, 0.0), yell=0.0,
    hf=("sh", 0.10, 0.97, 0.10), hb=("sh", -0.03, 0.98, 0.10), ef=0.0, eb=0.0,
    sf="relax", sb="relax", flag=None,
)
PLANT_IMPACT = 0.6


def _cyc(t, kw, freq):
    ph = kw.get("phase")
    return (ph if ph is not None else t * freq) % 1.0


def _walk_foot(c, s, lift_k):
    """c in [0,1): 0 = heel strike forward. returns (a, lift, toe)"""
    if c < 0.5:
        k = c * 2
        return (s * (1 - 2 * k), 0.0, -0.18 * (1 - k) ** 3 + 0.35 * k ** 4)
    k = (c - 0.5) * 2
    e = 0.5 - 0.5 * math.cos(math.pi * k)
    return (-s + 2 * s * e, math.sin(math.pi * k) ** 1.3 * s * lift_k, 0.35 * (1 - k) - 0.25 * k)


def p_stand(t, kw, sp):
    return dict()


def p_walk(t, kw, sp):
    spd = kw.get("speed", 1.0)
    c = _cyc(t, kw, 0.95 * spd ** 0.5)
    s = 8.0 * spd ** 0.6
    ff = _walk_foot(c, s, 0.42)
    fb = _walk_foot((c + 0.5) % 1, s, 0.42)
    sw = math.cos(TAU * c)
    bob = 0.9 * spd * (0.5 - 0.5 * math.cos(4 * math.pi * c))
    return dict(hip=(0.0, bob - 0.6 * spd), ff=ff, fb=fb, lean=0.04 * spd, sway=1.4 * math.sin(TAU * c * 2 - 1.2),
                head=(-0.02 + 0.015 * math.sin(4 * math.pi * c), 0.0),
                hf=("sh", 0.08 - 0.26 * sw * spd ** 0.5, 0.95, 0.12), hb=("sh", 0.05 + 0.26 * sw * spd ** 0.5, 0.95, 0.12),
                air=-0.3 * spd, mask=dict(arms=0.3, head=0.4))


def p_run(t, kw, sp):
    spd = kw.get("speed", 1.0)
    c = _cyc(t, kw, 1.35 * spd ** 0.5)
    s = 14.0 * spd ** 0.6
    ff = _walk_foot(c, s, 0.7)
    fb = _walk_foot((c + 0.5) % 1, s, 0.7)
    sw = math.cos(TAU * c)
    bob = 2.4 * (0.5 - 0.5 * math.cos(4 * math.pi * c - 0.8))
    return dict(hip=(1.0, bob - 2.5), ff=ff, fb=fb, lean=0.2 * spd ** 0.5, chest=0.04, sway=-3.0 - 1.5 * math.sin(4 * math.pi * c),
                head=(0.1, 0.0), hem_lift=2.0,
                hf=("sh", 0.28 - 0.42 * sw, 0.52 + 0.1 * sw, 0.12), hb=("sh", 0.28 + 0.42 * sw, 0.52 - 0.1 * sw, 0.12),
                ef=-0.2, eb=-0.2, sf="fist", sb="fist", air=-1.0 * spd, mask=dict(arms=0.35, head=0.5))


def p_hold_flag(t, kw, sp):
    return dict(hf=("sh", 0.5, 0.6, -0.55), flag=dict(ang=0.07, bu=1.0, gb=14.0), sf="grip", sb="grip",
                lean=-0.01, mask=dict(legs=0.1, spine=0.4, head=0.25))


def p_raise_flag(t, kw, sp):
    return dict(hf=("sh", 0.22, -0.9, -0.22), flag=dict(ang=0.04, gf=15.0, gb=-9.0), sf="grip", sb="grip",
                lean=-0.07, chest=-0.04, head=(0.3, 0.0), hip=(0.0, 0.8), mask=dict(legs=0.15, spine=0.6, head=0.5))


def p_salute(t, kw, sp):
    return dict(hf=("sh", 0.1, -1.0, 0.04), flag=dict(ang=0.015, gf=5.0, gb=None), sf="grip", sb="fist",
                hb=("sh", 0.02, 0.92, 0.18), lean=-0.06, chest=-0.04, head=(0.26, 0.0), hip=(0.0, 0.6),
                mask=dict(legs=0.15, spine=0.6, head=0.5))


def _kf(keys, p):
    """piecewise ease between keyframes [(p, dict)] of numeric/tuple/dict values"""
    p = clamp(p)
    for i in range(len(keys) - 1):
        p0, d0 = keys[i]
        p1, d1 = keys[i + 1]
        if p <= p1:
            k = (p - p0) / (p1 - p0) if p1 > p0 else 1.0
            k = k * k * (3 - 2 * k)
            return _mixd(d0, d1, k)
    return keys[-1][1]


def _mixv(x, y, k):
    if isinstance(x, tuple):
        if isinstance(x[0], str):
            return (x[0],) + tuple(a + (b - a) * k for a, b in zip(x[1:], y[1:]))
        return tuple(a + (b - a) * k for a, b in zip(x, y))
    if isinstance(x, dict):
        return _mixd(x, y, k)
    if isinstance(x, str) or x is None or y is None:
        return x if k < 0.5 else y
    return x + (y - x) * k


def _mixd(d0, d1, k):
    out = {}
    for key in set(d0) | set(d1):
        if key in d0 and key in d1:
            out[key] = _mixv(d0[key], d1[key], k)
        else:
            out[key] = d0.get(key, d1.get(key))
    return out


_PLANT = [
    (0.0, dict(hf=("sh", 0.45, 0.42, -0.5), flag=dict(ang=0.08, bus=0.34, gb=12.0), lean=0.0, chest=0.0,
               hip=(0.0, 0.0), head=(0.05, 0.0))),
    (0.42, dict(hf=("sh", 0.3, -0.62, -0.24), flag=dict(ang=-0.04, bus=0.72, gb=-9.0), lean=-0.1, chest=-0.05,
                hip=(0.0, 1.2), head=(0.22, 0.0))),
    (0.6, dict(hf=("sh", 0.55, 0.7, -0.5), flag=dict(ang=0.03, bus=-0.035, gb=13.0), lean=0.24, chest=0.1,
               hip=(0.0, -4.5), head=(-0.1, 0.0))),
    (0.72, dict(hf=("sh", 0.52, 0.6, -0.5), flag=dict(ang=0.03, bus=-0.035, gb=13.0), lean=0.18, chest=0.07,
                hip=(0.0, -3.4), head=(-0.04, 0.0))),
    (1.0, dict(hf=("sh", 0.48, 0.48, -0.5), flag=dict(ang=0.03, bus=-0.035, gb=13.0), lean=0.02, chest=0.0,
               hip=(0.0, 0.0), head=(0.08, 0.0))),
]


def p_plant_flag(t, kw, sp):
    d = _kf(_PLANT, kw.get("progress", 1.0))
    d.update(sf="grip", sb="grip", mask=dict(legs=0.5, spine=0.8, head=0.5))
    return d


def p_shoulder_flag(t, kw, sp):
    return dict(hf=("sh", 0.33, 0.17, -0.08), flag=dict(ang=-0.92, tilt=0.18, gf=12.0, gb=None), sf="grip",
                mask=dict(legs=0.1, spine=0.3, head=0.2))


def p_offer(t, kw, sp):
    return dict(hf=("sh", 0.70, 0.40, -0.32), flag=dict(ang=0.32, gf=17.0, gb=10.0), sf="grip", sb="grip",
                lean=0.06, head=(-0.02, 0.0), mask=dict(legs=0.1, spine=0.5, head=0.3))


def p_ram(t, kw, sp):
    return dict(hf=("sh", 0.55, 0.62, -0.25), flag=dict(ang=1.32, gf=9.0, gb=-11.0), sf="grip", sb="grip",
                lean=0.26, chest=0.05, hip=(1.0, -3.5), ff=(9.0, 0.0, 0.0), fb=(-8.0, 0.0, 0.25), head=(0.12, 0.0),
                mask=dict(legs=0.6, spine=0.7, head=0.5))


def p_stack(t, kw, sp):
    return dict(hf=("sh", 0.5, 0.6, -0.2), flag=dict(ang=1.05, gf=20.0, gb=None), sf="grip",
                hb=("sh", 0.95, 0.12, -0.05), sb="open", lean=0.36, chest=0.06, hip=(1.5, -9.0),
                ff=(10.0, 0.0, 0.0), fb=(-9.0, 0.0, 0.3), spread=1.2, head=(0.2, 0.0))


def p_leap(t, kw, sp):
    pr = kw.get("progress", 0.5)
    tuck = math.sin(math.pi * clamp(pr)) if "progress" in kw else 1.0
    return dict(hf=("sh", 0.88, 0.05, -0.2), flag=dict(ang=0.95, gf=12.0, gb=-10.0), sf="grip", sb="grip",
                ff=(9.0 + 3 * tuck, 6.0 + 12 * tuck, 0.3), fb=(-7.0 - 2 * tuck, 3.0 + 9 * tuck, -0.2),
                hip=(0.0, 1.0), lean=0.14, chest=0.04, sway=-3.5, hem_lift=6.0 * tuck, head=(0.1, 0.0), air=-1.2)


def p_point(t, kw, sp):
    return dict(hf=("sh", 0.96, -0.12, 0.06), sf="point", lean=0.04, head=(0.04, 0.0),
                mask=dict(legs=0.1, spine=0.5, head=0.4))


def p_point_self(t, kw, sp):
    return dict(hf=("chest", 6.5, 7.0, -3.0), sf="point", ef=-0.4, lean=-0.03, chest=-0.03, head=(0.12, 0.05),
                mask=dict(legs=0.1, spine=0.5, head=0.4))


def p_cheer(t, kw, sp):
    b = abs(math.sin(t * 5.2 + 0.4))
    return dict(hf=("sh", 0.2, -0.88 - 0.06 * b, 0.34), hb=("sh", 0.16, -0.9 + 0.05 * b, 0.36), sf="open", sb="open",
                hip=(0.0, 1.8 * b), lean=-0.05, chest=-0.04, head=(0.28, 0.0), sway=0.8 * math.sin(t * 5.2))


def p_shout(t, kw, sp):
    return dict(hf=("sh", -0.06, 0.86, 0.34), hb=("sh", -0.1, 0.86, 0.34), sf="fist", sb="fist", ef=-0.2, eb=-0.2,
                lean=0.12, chest=-0.1, head=(0.16, 0.0), shrug=1.2, yell=1.0)


def p_cup_shout(t, kw, sp):
    return dict(hf=("head", 1.45, 0.45, 0.62), hb=("head", 1.4, 0.45, 0.62), sf="cup", sb="cup", ef=-0.3, eb=-0.3,
                lean=0.12, chest=-0.06, head=(0.16, 0.0), shrug=1.5, yell=1.0, mask=dict(legs=0.1))


def p_meditate(t, kw, sp):
    hu = -sp["hip_y"]
    return dict(hip=(0.0, 12.0 - hu), sit=1.0, hf=("hip", 11.0, 7.0, 10.0), hb=("hip", 11.0, 7.0, 10.0),
                sf="open", sb="open", lean=0.0, head=(-0.02, 0.0), sway=0.0)


def p_crouch(t, kw, sp):
    return dict(hip=(1.0, -17.0), lean=0.4, chest=0.08, ff=(10.0, 0.0, 0.0), fb=(-7.0, 0.0, 0.55), spread=1.25,
                hf=("sh", 0.62, 0.86, 0.12), hb=("sh", 0.35, 0.9, 0.12), head=(0.32, 0.0), hem_lift=0.0)


def p_look_up(t, kw, sp):
    return dict(head=(0.5, 0.0), lean=-0.07, chest=-0.05, hf=("sh", 0.0, 0.97, 0.16), hb=("sh", -0.06, 0.97, 0.16),
                mask=dict(legs=0.2, arms=0.4))


def p_talk(t, kw, sp):
    g1, g2 = noise1(t * 1.3 + 5.1), noise1(t * 1.7 + 9.3)
    return dict(hf=("sh", 0.5 + 0.1 * g1, 0.46 + 0.12 * g2, 0.22), sf="open", head=(0.03 * g2, 0.03 * g1),
                lean=0.02, mask=dict(legs=0.1, spine=0.4, head=0.35))


def p_radio(t, kw, sp):
    return dict(hf=("head", -0.05, 0.12, 1.12), sf="open", ef=-0.1, head=(-0.05, 0.12), lean=0.03,
                mask=dict(legs=0.1, spine=0.3, head=0.5))


def p_declare(t, kw, sp):
    return dict(hf=("sh", 0.42, -0.72, 0.5), sf="fist", hb=("hip", 1.0, -2.0, 12.0), sb="fist", eb=0.0,
                lean=-0.06, chest=-0.07, head=(0.16, 0.0))


def p_one_finger(t, kw, sp):
    return dict(hf=("sh", 0.86, 0.02, 0.28), sf="finger", lean=-0.02, head=(0.08, 0.0),
                mask=dict(legs=0.1, spine=0.4, head=0.35))


_SLAM = [
    (0.0, dict(hf=("sh", 0.5, 0.5, 0.1), lean=0.02, head=(0.0, 0.0), hip=(0.0, 0.0))),
    (0.45, dict(hf=("sh", 0.32, -0.42, 0.12), lean=-0.06, head=(0.12, 0.0), hip=(0.0, 0.5))),
    (0.6, dict(hf=("sh", 0.76, 0.6, -0.08), lean=0.14, head=(-0.08, 0.0), hip=(0.0, -1.0))),
    (1.0, dict(hf=("sh", 0.74, 0.62, -0.08), lean=0.1, head=(-0.02, 0.0), hip=(0.0, -0.6))),
]


def p_slam(t, kw, sp):
    d = _kf(_SLAM, kw.get("progress", 1.0))
    d.update(sf="fist", mask=dict(legs=0.1, spine=0.6, head=0.5))
    return d


def p_sign(t, kw, sp):
    w = kw.get("progress", t * 1.7)
    return dict(hf=("sh", 0.72 + 0.05 * math.sin(w * 9.0), 0.64 + 0.02 * math.sin(w * 17.0), -0.34), sf="pinch",
                lean=0.12, head=(-0.28, 0.0), mask=dict(legs=0.1, spine=0.6, head=0.6))


def p_pat(t, kw, sp):
    d = 0.8 * abs(math.sin(t * 7.0))
    return dict(hf=("chest", 1.0, 9.0 - d, -9.0), sf="open", ef=0.0, lean=-0.02, head=(0.02, 0.0),
                mask=dict(legs=0.1, spine=0.3, head=0.2))


def p_touch(t, kw, sp):
    return dict(hf=("sh", 0.86, 0.32, -0.22), sf="point", lean=0.05, head=(-0.08, 0.0),
                mask=dict(legs=0.1, spine=0.5, head=0.4))


def p_recoil(t, kw, sp):
    return dict(hf=("sh", 0.32, 0.4, -0.05), sf="open", lean=-0.07, chest=-0.03, head=(0.04, -0.05),
                mask=dict(legs=0.1, spine=0.6, head=0.5))


def p_sniff(t, kw, sp):
    return dict(hf=("head", 1.35, 0.25, 0.25), sf="open", ef=-0.2, lean=0.03, head=(-0.1, 0.06),
                mask=dict(legs=0.1, spine=0.5, head=0.6))


def p_hold_hood(t, kw, sp):
    return dict(hf=("head", 0.15, -0.9, 0.55), sf="grip", ef=-0.1, head=(-0.08, 0.06),
                mask=dict(legs=0.1, spine=0.2, head=0.3))


def p_gesture(t, kw, sp):
    return dict(hf=("sh", 0.76, 0.28, 0.34), sf="open", lean=0.02, head=(0.0, 0.03),
                mask=dict(legs=0.1, spine=0.4, head=0.3))


def p_shrug(t, kw, sp):
    return dict(hf=("sh", 0.34, 0.6, 0.55), hb=("sh", 0.3, 0.6, 0.55), sf="open", sb="open", shrug=3.5,
                head=(0.03, 0.12), lean=-0.02, mask=dict(legs=0.1))


def p_wave(t, kw, sp):
    return dict(hf=("sh", 0.32 + 0.14 * math.sin(t * 9.0), -0.84, 0.36), sf="open", head=(0.08, 0.04),
                mask=dict(legs=0.1, spine=0.3, head=0.3))


def p_phone(t, kw, sp):
    return dict(hf=("head", 1.7, 1.7, 0.2), sf="pinch", ef=-0.2, head=(-0.32, 0.0), lean=0.04,
                mask=dict(legs=0.1, spine=0.5, head=0.8))


def p_megaphone(t, kw, sp):
    return dict(hf=("head", 1.55, 0.55, 0.12), sf="pinch", ef=-0.3, head=(0.12, 0.0), lean=0.1, chest=-0.05, yell=1.0,
                mask=dict(legs=0.1, spine=0.6, head=0.6))


# ------------------------------------------------------------ film 2 (THE FLAG GAME?) poses
def p_sit(t, kw, sp):
    hu = -sp["hip_y"]
    seat_u = kw.get("seat_h", 27.0)
    d = dict(hip=(-2.5, seat_u - hu), ff=(16.0, 0.0, 0.0), fb=(14.5, 0.0, 0.0), spread=1.15, seat=1.0,
             cross=1.0 if kw.get("legs_crossed") else 0.0, lean=-0.09, chest=-0.02, head=(0.02, 0.0),
             hf=("hip", 12.0, -4.0, 4.5), hb=("hip", 11.0, -4.0, 5.0), sf="relax", sb="relax", sway=0.0,
             mask=dict(arms=0.2, head=0.5))
    return d


def p_kneel(t, kw, sp):
    hu = -sp["hip_y"]
    ku = sp["leg_u"] + 2.0
    return dict(hip=(-2.0, ku - hu), ff=(-sp["leg_l"] * 0.82, 0.5, 1.35), fb=(-sp["leg_l"] * 0.86, 0.5, 1.35),
                spread=1.05, seat=1.0, lean=0.06, head=(-0.05, 0.0), hem_lift=0.0,
                hf=("sh", 0.12, 0.95, 0.12), hb=("sh", 0.08, 0.95, 0.12), mask=dict(arms=0.2, head=0.5))


def p_pray(t, kw, sp):
    return dict(hf=("head", 1.05, 2.05, -0.02), hb=("head", 1.05, 2.05, -0.02), sf="clasp", sb="clasp", ef=-1.3,
                eb=-1.3, head=(-0.16, 0.0), lean=0.03, mask=dict(legs=0.0, spine=0.5, head=0.8))


def p_cheeks(t, kw, sp):
    """Home-Alone horror: hands fly up to the cheeks"""
    return dict(hf=("head", 0.55, 0.62, 0.92), hb=("head", 0.55, 0.62, 0.92), sf="open", sb="open", ef=-1.0, eb=-1.0,
                head=(0.06, 0.0), lean=-0.04, shrug=2.0, mask=dict(legs=0.0, spine=0.5, head=0.4))


def p_palms_out(t, kw, sp):
    return dict(hf=("sh", 0.62, -0.05, 0.45), hb=("sh", 0.58, -0.02, 0.5), sf="open", sb="open", ef=-0.4, eb=-0.4,
                lean=-0.08, head=(0.04, -0.05), shrug=2.0, mask=dict(legs=0.0, spine=0.6, head=0.4))


def p_film(t, kw, sp):
    """holding a phone up in front of the face, filming"""
    return dict(hf=("head", 2.7, 0.15, 0.05), sf="pinch", ef=-0.6, head=(0.0, 0.0), lean=0.02,
                mask=dict(legs=0.0, spine=0.4, head=0.3))


def p_read(t, kw, sp):
    return dict(hf=("chest", 9.5, 12.5, 3.2), hb=("chest", 9.5, 12.5, 3.2), sf="pinch", sb="pinch", ef=-0.2, eb=-0.2,
                head=(-0.3, 0.0), lean=0.04, mask=dict(legs=0.1, spine=0.5, head=0.8))


def p_offer_tract(t, kw, sp):
    return dict(hf=("sh", 0.86, 0.36, -0.12), sf="pinch", lean=0.07, head=(-0.02, 0.0),
                mask=dict(legs=0.1, spine=0.6, head=0.4))


_TOSS = [
    (0.0, dict(hf=("sh", 0.3, 0.55, 0.1), lean=0.0)),
    (0.35, dict(hf=("sh", -0.45, 0.45, 0.25), lean=-0.06)),
    (0.7, dict(hf=("sh", 0.8, -0.1, 0.1), lean=0.1)),
    (1.0, dict(hf=("sh", 0.6, 0.35, 0.1), lean=0.05)),
]


def p_toss(t, kw, sp):
    d = _kf(_TOSS, kw.get("progress", 0.7))
    d.update(sf="pinch" if kw.get("progress", 0.7) < 0.68 else "open", mask=dict(legs=0.1, spine=0.6, head=0.3))
    return d


def p_preach(t, kw, sp):
    return dict(hf=("sh", 0.32, -0.86, 0.22), sf="point", hb=("chest", 6.0, 13.0, 5.0), sb="grip", eb=-0.3,
                lean=-0.03, chest=-0.03, head=(0.12, 0.0), mask=dict(legs=0.1, spine=0.6, head=0.5))


def p_lean_in(t, kw, sp):
    return dict(hf=("chest", 9.0, 15.0, 5.0), sf="open", lean=0.2, chest=0.08, head=(0.08, 0.06),
                mask=dict(legs=0.2, arms=0.6))


def p_fist_raise(t, kw, sp):
    return dict(hf=("sh", 0.2, -0.96, 0.18), sf="fist", lean=-0.05, chest=-0.04, head=(0.22, 0.0), yell=0.6,
                mask=dict(legs=0.1, spine=0.6, head=0.5))


def p_fly(t, kw, sp):
    sw = math.sin(t * 2.0)
    return dict(hip=(0.0, 0.0), ff=(-9.0, 9.0 + sw, 0.9), fb=(-13.0, 6.0 - sw, 1.0), spread=0.9, lean=0.28,
                chest=0.04, hem_lift=4.0, air=-1.4, flap=1.0, head=(0.18, 0.0),
                hf=("sh", 0.55, 0.3, 0.45), hb=("sh", 0.5, 0.3, 0.45), sf="open", sb="open")


def p_throne(t, kw, sp):
    hu = -sp["hip_y"]
    seat_u = kw.get("seat_h", 30.0)
    return dict(hip=(-3.0, seat_u - hu), ff=(14.0, 0.0, 0.0), fb=(13.0, 0.0, 0.0), spread=1.7, seat=1.0,
                lean=-0.05, head=(0.04, 0.0), hf=("hip", 14.0, -9.0, 2.0), sf="grip",
                hb=("hip", 9.0, -8.5, 13.0), sb="relax", mask=dict(arms=0.4, head=0.6))


POSE_FNS = {
    "stand": p_stand, "walk": p_walk, "run": p_run, "hold_flag": p_hold_flag, "raise_flag": p_raise_flag,
    "plant_flag": p_plant_flag, "point": p_point, "cheer": p_cheer, "shout": p_shout, "meditate": p_meditate,
    "salute": p_salute, "crouch": p_crouch, "look_up": p_look_up, "talk": p_talk,
    "offer": p_offer, "shoulder_flag": p_shoulder_flag, "ram": p_ram, "stack": p_stack, "leap": p_leap,
    "radio": p_radio, "cup_shout": p_cup_shout, "point_self": p_point_self, "declare": p_declare,
    "one_finger": p_one_finger, "slam": p_slam, "sign": p_sign, "pat": p_pat, "touch": p_touch,
    "recoil": p_recoil, "sniff": p_sniff, "hold_hood": p_hold_hood, "gesture": p_gesture, "shrug": p_shrug,
    "wave": p_wave, "phone": p_phone, "megaphone": p_megaphone,
    "sit": p_sit, "kneel": p_kneel, "pray": p_pray, "read": p_read, "offer_tract": p_offer_tract, "toss": p_toss,
    "preach": p_preach, "lean_in": p_lean_in, "walk_away": p_walk, "fist_raise": p_fist_raise, "fly": p_fly,
    "throne": p_throne, "cheeks": p_cheeks, "palms_out": p_palms_out, "film": p_film,
}
ALIASES = {"shout_cupped": "cup_shout", "shout_hood": {"shout": 1.0, "hold_hood": 1.0}, "carry": "hold_flag",
           "idle": "stand", "hold": "hold_flag", "raise": "raise_flag", "plant": "plant_flag", "seated": "sit",
           "pray_kneel": {"kneel": 1.0, "pray": 1.0}}
FLAG_POSES = tuple(k for k in POSE_FNS if k in ("hold_flag", "raise_flag", "plant_flag", "salute", "offer",
                                                "shoulder_flag", "ram", "stack", "leap"))


def norm_pose(pose):
    if isinstance(pose, str):
        a = ALIASES.get(pose, pose)
        if isinstance(a, dict):
            return dict(a)
        return {a if a in POSE_FNS else "stand": 1.0}
    out = {}
    for k, w in pose.items():
        if w <= 0:
            continue
        a = ALIASES.get(k, k)
        if isinstance(a, dict):
            for k2, w2 in a.items():
                out[k2] = out.get(k2, 0.0) + w * w2
        else:
            a = a if a in POSE_FNS else "stand"
            out[a] = out.get(a, 0.0) + w
    return out or {"stand": 1.0}


# ============================================================ solver
class Rig:
    """solved skeleton in local screen space (x forward-ish, y down, units h/100) + 3D body points"""


def _resolve_target(spec, side, T, sp):
    """side +1 near arm, -1 far arm. returns 3D point"""
    kind = spec[0]
    if kind == "sh":
        S = T["sh_n"] if side > 0 else T["sh_f"]
        L = sp["L"]
        fa, fd, fo = spec[1] * L, spec[2] * L, spec[3] * L
        return (S[0] + fa, S[1] - fd, S[2] + side * fo)
    if kind == "head":
        C = T["head"]
        r = sp["head_r"]
        return (C[0] + spec[1] * r, C[1] - spec[2] * r, C[2] + side * spec[3] * r)
    if kind == "chest":
        N = T["neck"]
        return (N[0] + spec[1], N[1] - spec[2], N[2] + side * spec[3])
    if kind == "hip":
        Hh = T["hip"]
        return (Hh[0] + spec[1], Hh[1] - spec[2], Hh[2] + side * spec[3])
    raise ValueError(spec)


def solve(sp, seed, pose, t, kw, cphi, sphi, mouth=0.0, wind=0.0):
    poses = norm_pose(pose)
    evald = []
    for name, w in poses.items():
        d = POSE_FNS[name](t, kw, sp)
        m = d.pop("mask", None) or {}
        evald.append((name, w, d, m))
    if any(d.get("seat", 0.0) > 0 for _, _, d, _ in evald):
        evald = [(n, w, d, (dict(m, legs=0.0) if m.get("legs", 1.0) < 0.5 else m)) for n, w, d, m in evald]

    def blend(keys, ch):
        tot = 0.0
        acc = {}
        wsum = sum(w * m.get(ch, 1.0) for _, w, _, m in evald)
        for name, w, d, m in evald:
            ww = w * m.get(ch, 1.0)
            if wsum <= 1e-9:
                ww = w
            if ww <= 0:
                continue
            tot += ww
            for k in keys:
                v = d.get(k, DEFAULT[k])
                if k in acc:
                    a = acc[k]
                    acc[k] = tuple(x + y * ww for x, y in zip(a, v)) if isinstance(v, tuple) else a + v * ww
                else:
                    acc[k] = tuple(y * ww for y in v) if isinstance(v, tuple) else v * ww
        inv = 1.0 / (tot or 1.0)
        return {k: (tuple(x * inv for x in v) if isinstance(v, tuple) else v * inv) for k, v in acc.items()}

    Lg = blend(CH_LEGS, "legs")
    Sp = blend(CH_SPINE, "spine")
    Hd = blend(CH_HEAD, "head")

    # ---- secondary: breathing, idle drift, speech energy
    ph = hash01(seed, 77) * TAU
    breath = math.sin(TAU * t / 3.9 + ph)
    idle = noise1(t * 0.33 + seed * 1.7, 5)
    lean = sp["lean0"] + Sp["lean"] + kw.get("lean", 0.0) + 0.006 * idle
    chest = Sp["chest"]
    hip_u = -sp["hip_y"] + Lg["hip"][1] + 0.12 * breath * (1 - Lg["sit"])
    hip = (Lg["hip"][0] + 0.35 * idle, hip_u, 0.0)
    spine = (-sp["sh_y"]) - (-sp["hip_y"])
    neck = rot_au((0.0, spine + 0.28 * breath, 0.0), lean)
    N = v_add(hip, neck)
    shrug = Sp["shrug"] + 0.25 * breath
    up_dir = rot_au((0.0, 1.0, 0.0), lean)
    sh_c = v_add(N, v_mul(up_dir, -1.2 + shrug * 0.5))
    sw = sp["sh_w"]
    sh_n = (sh_c[0], sh_c[1], sw)
    sh_f = (sh_c[0], sh_c[1], -sw)
    pitch = Hd["head"][0] + kw.get("nod", 0.0) + 0.05 * mouth + 0.012 * noise1(t * 0.7 + seed, 9)
    roll = Hd["head"][1] + kw.get("tilt", 0.0) + 0.015 * noise1(t * 0.5 + seed, 11)
    hdir = rot_au((0.0, 1.0, 0.0), lean + chest * 1.4 - pitch * 0.25)
    head = v_add(N, v_mul(hdir, sp["neck"] + sp["head_r"] + 0.35 * mouth))
    T = {"sh_n": sh_n, "sh_f": sh_f, "head": head, "neck": N, "hip": hip}

    # ---- arms: resolve per-pose targets against the blended trunk
    wsum_arm = sum(w * m.get("arms", 1.0) for _, w, _, m in evald) or 1.0
    tf = [0.0, 0.0, 0.0]
    tb = [0.0, 0.0, 0.0]
    wb_free = 0.0
    ef = eb = 0.0
    flag_w = 0.0
    fl_ang = fl_tilt = 0.0
    fl_items = []
    best_f = best_b = (-1.0, "relax")
    for name, w, d, m in evald:
        ww = w * m.get("arms", 1.0) / wsum_arm
        if ww <= 0:
            continue
        pf = _resolve_target(d.get("hf", DEFAULT["hf"]), 1, T, sp)
        for i in range(3):
            tf[i] += pf[i] * ww
        ef += d.get("ef", 0.0) * ww
        eb += d.get("eb", 0.0) * ww
        fl = d.get("flag")
        grips_back = fl is not None and fl.get("gb") is not None
        if not grips_back:
            pb = _resolve_target(d.get("hb", DEFAULT["hb"]), -1, T, sp)
            for i in range(3):
                tb[i] += pb[i] * ww
            wb_free += ww
        if fl is not None:
            flag_w += ww
            fl_ang += fl.get("ang", 0.0) * ww
            fl_tilt += fl.get("tilt", 0.0) * ww
            fl_items.append((ww, fl))
        if ww > best_f[0]:
            best_f = (ww, d.get("sf", "relax"))
        sbv = d.get("sb", "relax")
        if ww > best_b[0]:
            best_b = (ww, sbv)
    tf = tuple(tf)

    Lsh = sp["arm_u"]
    Lfo = sp["arm_f"]
    rig = Rig()
    rig.sp, rig.cphi, rig.sphi = sp, cphi, sphi
    # world-space hand overrides (already converted to local screen by caller): kw["_hf_xy"], kw["_hb_xy"]

    def override(tgt, key):
        xy = kw.get(key)
        if xy is None:
            return tgt
        x0 = tgt[0] * sphi - tgt[2] * cphi
        dx = xy[0] - x0
        return (tgt[0] + dx * sphi, -xy[1], tgt[2] - dx * cphi)

    tf = override(tf, "_hf_xy")
    hint_f = (-0.45 + 1.2 * ef, -1.0, 0.55)
    el_f, hand_f, _ = ik3(sh_n, tf, Lsh, Lfo, hint_f)

    pole = None
    grip_b = None
    if flag_w > 1e-6:
        fl_ang /= flag_w
        fl_tilt /= flag_w
        ca, sa = math.cos(fl_ang), math.sin(fl_ang)
        ct, st = math.cos(fl_tilt), math.sin(fl_tilt)
        pdir = (sa * ct, ca * ct, st)
        gf = 0.0
        gb = 0.0
        gbw = 0.0
        for ww, fl in fl_items:
            if "gf" in fl:
                g = fl["gf"]
            elif "bus" in fl:
                g = (hand_f[1] - fl["bus"] * (-sp["sh_y"])) / max(pdir[1], 0.2)
            else:
                g = (hand_f[1] - fl.get("bu", 0.0)) / max(pdir[1], 0.2)
            gf += g * ww
            if fl.get("gb") is not None:
                gb += fl["gb"] * ww
                gbw += ww
        gf /= flag_w
        base = v_sub(hand_f, v_mul(pdir, gf))
        pole = (base, pdir, gf)
        if gbw > 0:
            gb /= gbw
            G = v_add(hand_f, v_mul(pdir, gb))
            # keep the back grip reachable: clamp its position along the pole into the arm's reach interval
            reach = (Lsh + Lfo) * 0.985
            D = v_sub(hand_f, sh_f)
            g0 = -v_dot(D, pdir)
            perp2 = v_dot(D, D) - g0 * g0
            if perp2 < reach * reach:
                wr = math.sqrt(reach * reach - perp2)
                gb = min(max(gb, g0 - wr), g0 + wr)
            else:
                gb = g0
            G = v_add(hand_f, v_mul(pdir, gb))
            grip_b = G
            # blend with free back-hand target when the flag poses don't own the arms entirely
            s = smoothstep(0.35, 0.75, gbw)
            if wb_free > 1e-6:
                fb = (tb[0] / wb_free, tb[1] / wb_free, tb[2] / wb_free)
                tb = v_lerp(fb, G, s)
            else:
                tb = G
        else:
            tb = (tb[0] / wb_free, tb[1] / wb_free, tb[2] / wb_free) if wb_free > 0 else \
                _resolve_target(DEFAULT["hb"], -1, T, sp)
    else:
        tb = (tb[0] / wb_free, tb[1] / wb_free, tb[2] / wb_free) if wb_free > 0 else \
            _resolve_target(DEFAULT["hb"], -1, T, sp)
    tb = override(tb, "_hb_xy")
    hint_b = (-0.45 + 1.2 * eb, -1.0, -0.55)
    el_b, hand_b, _ = ik3(sh_f, tb, Lsh, Lfo, hint_b)
    if pole is not None:
        # re-anchor pole so it passes exactly through the SOLVED front hand
        base = v_sub(hand_f, v_mul(pole[1], pole[2]))
        pole = (base, pole[1], pole[2])

    # ---- legs
    spread = Lg["spread"]
    hw = sp["hip_w"]
    fl_n = hw * 0.95 * spread
    hip_n = (hip[0], hip[1] - 1.0, hw)
    hip_f = (hip[0], hip[1] - 1.0, -hw)
    ffv, fbv = Lg["ff"], Lg["fb"]
    ank = sp["ank"]
    ank_n = (ffv[0], ffv[1] + ank, fl_n)
    ank_f = (fbv[0], fbv[1] + ank, -fl_n)
    knee_f, ank_f, _ = ik3(hip_f, ank_f, sp["leg_u"], sp["leg_l"], (1.0, 0.15, -0.25))
    cr = Lg.get("cross", 0.0)
    if cr > 1e-3:
        tgt = (knee_f[0] + 7.0, knee_f[1] - sp["leg_l"] * 0.72, knee_f[2] * 0.2 + 1.5)
        ank_n = v_lerp(ank_n, tgt, min(1.0, cr))
        hip_n = (hip_n[0], hip_n[1] + 1.5 * cr, hip_n[2] * (1 - 0.35 * cr))
        knee_n, ank_n, _ = ik3(hip_n, ank_n, sp["leg_u"], sp["leg_l"], (0.8, 0.6, 0.35))
    else:
        knee_n, ank_n, _ = ik3(hip_n, ank_n, sp["leg_u"], sp["leg_l"], (1.0, 0.15, 0.25))

    B = dict(hip=hip, neck=N, sh_n=sh_n, sh_f=sh_f, el_n=el_f, el_f=el_b, hand_n=hand_f, hand_f=hand_b,
             head=head, hip_n=hip_n, hip_f=hip_f, knee_n=knee_n, knee_f=knee_f, ank_n=ank_n, ank_f=ank_f,
             sh_c=sh_c)
    P = {}
    Z = {}
    for k, p in B.items():
        x, y = p[0] * sphi - p[2] * cphi, -p[1]
        if k in ("knee_n", "knee_f", "ank_n", "ank_f"):
            y += p[0] * cphi * 0.16
        P[k] = (x, y)
        Z[k] = p[0] * cphi + p[2] * sphi
    rig.P, rig.Z, rig.B = P, Z, B
    rig.toe = None
    rig.pole = None
    if pole is not None:
        base, pdir, gf = pole
        bx, by = base[0] * sphi - base[2] * cphi, -base[1]
        dx, dy = pdir[0] * sphi - pdir[2] * cphi, -pdir[1]
        rig.pole = (bx, by, math.atan2(dx, -dy), (dx, dy))
    rig.grip_f = pole is not None and best_f[1] == "grip"
    rig.grip_b = grip_b is not None and v_len(v_sub(hand_b, grip_b)) < 1.5
    rig.shape_f = kw.get("shape_r") or ("grip" if rig.grip_f else (best_f[1] if best_f[1] != "grip" else "fist"))
    rig.shape_b = kw.get("shape_l") or ("grip" if rig.grip_b else (best_b[1] if best_b[1] != "grip" else "fist"))
    rig.feet = (ffv, fbv)
    rig.lean = lean + chest
    rig.head_pitch = pitch
    rig.head_rot = roll + (lean + chest * 1.4) * sphi * 0.45 - pitch * 0.12 * sphi
    rig.sway = Sp["sway"]
    rig.hem_lift = Lg["hem_lift"]
    rig.sit = Lg["sit"]
    rig.seat = Lg.get("seat", 0.0)
    rig.flap_on = Sp.get("flap", 0.0)
    rig.breath = breath
    rig.mouth = mouth
    rig.wind = wind
    rig.yell = Hd.get("yell", 0.0)
    rig.air = Sp.get("air", 0.0) * sphi
    # does the far hand come in front of the torso? (then draw the far forearm over the body)
    zc = depth(N, cphi, sphi) + sp["dep"] * cphi * 0.9
    rig.back_front = Z["hand_f"] > zc - 1.0 and cphi > 0.25
    return rig


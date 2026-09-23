"""s04 — FLAGARTHA: THE HYPERMIND (61.5–90.0)

Signature: a colossal gold quantum chandelier inhaling rivers of holy scripture — and compiling… a plain Flag.
Everything is a real 3D scene (scenes/s04_geo.py pinhole camera, cm units) rendered as flat vector art:
columnar-basalt pentagonal chamber + shaft (s04_world), the Hypermind dilution-refrigerator chandelier
(s04_hyper), a projective-textured polar map of the real Earth (s04_map), scripture/scrolls/beam (s04_fx).
Characters (vx.chars) and the hypermeme Flag (vx.flag real cloth sim) are cylindrical billboards in that world.

Shots (global T):
  61.50 descent   from black: fall down the pentagonal shaft (motion-blurred), through the suspension truss, past
                  the top plate, tilt up into the chamber; Hypermind reveal H01 63.0; six canon scrolls absorbed
                  65.50..67.25; scripture streams spiral up like incense; slow push-in
  67.90 console   low 3/4 on the cold stages: readout ring "LOSS 0.0000 · FLAGS: TRUE" ('TRUE' faces camera on 70.11)
  71.20 compile   wide; tilt down the stack following the H02 energy rings -> tip -> beam 72.35 -> table
  72.60 manifest  pedestal medium: 72.70 the hypermeme (a plain yellow Flag) materializes, waves in an impossible breeze
  74.60 lutie     L01 "It's... a Flag."            76.20 crocus  C01 "It was always going to be a Flag."
  79.10 map       L02 over the slop-crazed map table (Lutie points at the fractures on "shattered")
  83.30 summon    C02: Crocus lifts the Flag out of its socket and over his head
  85.55 raise_    closer: holds it aloft; thrusts it up on 'rise' (86.81)
  87.30 summon    strikes the floor with the pole butt on 'Summon' (87.85): shockwave, the pentagonal seal opens
                  (88.3-88.8), light column, gold klaxons sweep, push in; 89.40 braam accent; flash; cut 90.0
"""
import math

import cairo
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from vx import *
from vx import foley
from vx.flag import Flag
from vx.chars import Character

from scenes.s04_geo import Cam3, poly_screen, path_pts, ngon
from scenes import s04_world as Wd
from scenes import s04_hyper as HM
from scenes import s04_map as MP
from scenes import s04_fx as FX

POST = dict(bloom=0.45, bloom_thresh=0.7, bloom_radius=26.0, grain=0.03, vignette=0.36, contrast=1.06, saturation=1.04)

SHOTS = [("descent", 61.5), ("console", 67.9), ("compile", 71.2), ("manifest", 72.6), ("lutie", 74.6),
         ("crocus", 76.2), ("map", 79.1), ("summon", 83.3), ("raise_", 85.55), ("summon", 87.3)]

# cast positions (world cm). Table at the origin.
LUTIE = dict(x=-150.0, z=-30.0, h=158.0)
CROCUS = dict(x=106.0, z=40.0, h=192.0)
POLE = 190.0                  # the hypermeme is a small Survey Flag: pole length (cm), cloth 0.375 x 0.3125 pole
SIM_K = 420.0 / POLE          # sim space (design px calibrated on a 420 px pole) per world cm
T_MAT = 72.70                 # hypermeme cue
T_BEAM = 72.35
T_STRIKE = 87.85              # pole butt hits the floor on "Summon"
T_SEAL = 88.15                # seal unlocks
T_BRAAM = 89.40


def shot_at(T):
    cur = SHOTS[0]
    for s in SHOTS:
        if T >= s[1] - 1e-6:
            cur = s
    return cur


# ------------------------------------------------------------------ camera paths
class Path:
    def __init__(self, keys):
        ts = np.array([k[0] for k in keys])
        self.names = [n for n in keys[0][1]]
        self.f = {n: PchipInterpolator(ts, [k[1][n] for k in keys]) for n in self.names}
        self.t0, self.t1 = ts[0], ts[-1]

    def __call__(self, T):
        T = min(max(T, self.t0), self.t1)
        return {n: float(self.f[n](T)) for n in self.names}


DESCENT = Path([
    (61.50, dict(r=980, y=16200, phi=math.pi - 1.1, pitch=-89.9, f=760, spin=2.4)),
    (62.10, dict(r=1000, y=10600, phi=math.pi - 0.9, pitch=-89.9, f=780, spin=1.1)),
    (62.80, dict(r=1380, y=6000, phi=math.pi - 0.62, pitch=-80.0, f=800, spin=0.2)),
    (63.40, dict(r=1700, y=5650, phi=math.pi - 0.45, pitch=-60.0, f=820, spin=0.0)),
    (64.10, dict(r=3100, y=3900, phi=math.pi - 0.30, pitch=-4.0, f=840, spin=0.0)),
    (64.90, dict(r=3950, y=2500, phi=math.pi - 0.20, pitch=9.0, f=860, spin=0.0)),
    (67.90, dict(r=3350, y=2050, phi=math.pi + 0.28, pitch=17.0, f=900, spin=0.0)),
])


def cam_descent(T):
    k = DESCENT(T)
    phi = k["phi"]
    pos = np.array([k["r"] * math.sin(phi), k["y"], k["r"] * math.cos(phi)])
    yaw = phi + math.pi + k["spin"]
    return Cam3(pos, yaw, math.radians(k["pitch"]), 0.0, k["f"])


def cam_console(T):
    # low 3/4 on the cold stages: the readout ring around the mixing-chamber can, scripture streaming past
    p = seg(T, 67.9, 71.2, ease_in_out_sine)
    th = math.pi + 0.55 - 0.10 * p
    r = 1950 - 260 * p
    pos = np.array([r * math.sin(th), 1120 + 40 * p, r * math.cos(th)])
    tgt = np.array([0.0, 1560.0 + 30 * p, 0.0])
    return Cam3.look_at(pos, tgt, f=1060 + 90 * p)


def cam_compile(T):
    # wide 3/4 of the whole stack; tilt down with the energy to the tip, then down the beam to the table
    p = seg(T, 71.2, 72.55, lambda x: ease_in_out(x, 2.0))
    th = math.pi - 0.42
    pos = np.array([4300 * math.sin(th), 520.0, 4300 * math.cos(th)])
    ty = lerp(3350, 700, p)
    return Cam3.look_at(pos, np.array([0.0, ty, 0.0]), f=lerp(860, 1250, p))


def cam_manifest(T):
    p = seg(T, 72.6, 74.6, ease_out)
    pos = np.array([-10.0, 150.0, -400.0 + 50 * p])
    tgt = np.array([0.0, 176.0 - 8 * p, 0.0])
    return Cam3.look_at(pos, tgt, f=1320 + 90 * p)


def cam_lutie(T):
    p = seg(T, 74.6, 76.2, ease_in_out_sine)
    pos = np.array([-20.0, 128.0, -205.0 + 8 * p])
    tgt = np.array([-150.0, 140.0, -30.0])
    return Cam3.look_at(pos, tgt, f=2300 + 160 * p, shift=(-250, -10))


def cam_crocus(T):
    p = seg(T, 76.2, 79.1, ease_in_out_sine)
    pos = np.array([-70.0, 170.0, -300.0 + 14 * p])
    tgt = np.array([70.0, 172.0 - 3 * p, 30.0])
    return Cam3.look_at(pos, tgt, f=1650 + 140 * p)


def cam_map(T):
    p = seg(T, 79.1, 83.3, ease_in_out_sine)
    pos = np.array([-80.0 + 10 * p, 372.0 - 22 * p, -250.0 + 22 * p])
    tgt = np.array([-10.0, 128.0, 18.0])
    return Cam3.look_at(pos, tgt, f=1060 + 130 * p)


def cam_summon(T):
    # low heroic angle from Crocus's front-right: the Flag raised against the dark vault, the seal behind him
    push = seg(T, 88.35, 90.0, lambda x: ease_in(x, 2.0)) * 0.7 + seg(T, T_BRAAM - 0.05, 90.0, ease_out) * 0.3
    lift = seg(T, 84.0, 86.9, ease_in_out_sine)
    pos = np.array([330.0 - 90 * push, 72.0 + 40 * push, -390.0 + 150 * push])
    tgt = np.array([40.0 + 20 * push, 205.0 + 25 * lift - 25 * push, 300.0 - 230 * push])
    f = 1300 + 60 * seg(T, 83.3, 88.3) + 1250 * push
    amp = 7 * math.exp(-max(0.0, T - T_STRIKE) * 5) * (T > T_STRIKE) + 3.0 * seg(T, 88.35, 88.6)
    sh = shake(T, amp, 14, 7)
    return Cam3.look_at(pos, tgt, f=f, shift=(sh[0], sh[1]))


def cam_raise(T):
    # low-angle close on Crocus holding the Flag aloft ("before the true one can RISE")
    p = seg(T, 85.55, 87.3, ease_in_out_sine)
    up = seg(T, 86.4, 86.9, ease_out)
    pos = np.array([300.0 - 12 * p, 150.0, -262.0 + 18 * p])
    tgt = np.array([96.0, 250.0 + 25 * up, 40.0])
    return Cam3.look_at(pos, tgt, f=1150 + 60 * p, shift=(80, 0))


CAMS = dict(descent=cam_descent, console=cam_console, compile=cam_compile, manifest=cam_manifest,
            lutie=cam_lutie, crocus=cam_crocus, map=cam_map, summon=cam_summon, raise_=cam_raise)


# ------------------------------------------------------------------ acting
def crocus_pose(T):
    """-> (pose, kw, grip) for Crocus. grip 0..1 = how much the Flag is in his hand"""
    kw = dict(expr="serene", look=(-0.6, -0.2))
    if T < 72.7:
        return "stand", kw, 0.0
    if T < 83.55:
        kw["lean"] = 0.08 * seg(T, 72.7, 73.4) * (1 - seg(T, 74.0, 75.0))
        if T > 76.2:
            kw["look"] = (-0.5, -0.3)
            kw["nod"] = 0.05
        pose = {"stand": 1.0, "talk": 0.6 * seg(T, 76.2, 76.5) * (1 - seg(T, 78.8, 79.2))}
        if 79.2 < T:
            kw["look"] = (-0.8, 0.1)          # listening to Lutie
        return pose, kw, 0.0
    # C02: reach, lift the Flag out of its socket and up over his head, raise it on "rise",
    # strike the floor with the pole butt on "Summon"
    reach = seg(T, 83.5, 84.05)
    lift = seg(T, 84.05, 85.0, ease_in_out)
    raise_ = seg(T, 86.4, 86.85, ease_out_back) * (1 - seg(T, 87.3, 87.62))
    plant = seg(T, T_STRIKE - 0.36, T_STRIKE + 0.24)
    kw["expr"] = "determined" if T > 87.2 else "serene"
    kw["look"] = (-0.3, -0.4 - 0.4 * raise_)
    kw["nod"] = 0.12 * lift + 0.1 * raise_
    if T > T_STRIKE - 0.36:
        kw["progress"] = plant
        kw["nod"] = 0.0
        kw["look"] = (0.5, -0.1) if T > 88.3 else (-0.2, 0.2)   # turns to the opening seal
        kw["_hand_dx"] = -34.0 * seg(T, T_STRIKE - 0.36, T_STRIKE - 0.05, ease_in_out)   # plants it at arm's length
        return {"plant_flag": 1.0}, kw, 1.0
    up = lift
    pose = {"stand": max(0.0, 1 - reach), "hold_flag": reach * (1 - up) * (1 - raise_),
            "raise_flag": reach * up * (1 - raise_) + raise_}
    kw["lean"] = 0.14 * reach * (1 - lift)
    if lift < 0.35:
        kw["hand_r"] = (-CROCUS["x"] * 0.8, -(Wd.PED_Y + POLE * 0.45))
    return pose, kw, reach * (0.3 + 0.7 * lift)


def lutie_pose(T):
    kw = dict(expr="neutral", look=(0.5, -0.1))
    if T < 72.7:
        return {"stand": 1.0}, kw
    if T < 79.1:
        kw["lean"] = 0.1 * seg(T, 72.8, 73.6)
        kw["expr"] = "awe" if T < 76.2 else {"awe": 0.4, "deadpan": 0.6}
        kw["look"] = (0.6, -0.5)
        pose = {"stand": 1.0, "talk": 0.3 * (74.8 < T < 76.0)}
        if T > 76.2:
            kw["look"] = (0.9, -0.2)
        return pose, kw
    kw["expr"] = "worried"
    point = seg(T, 80.05, 80.4) * (1 - seg(T, 82.0, 82.5))
    kw["look"] = (0.6, 0.5 * point)
    if T > 81.8:
        kw["look"] = (0.9, -0.2)            # "Why break it further?" -> to Crocus
    return {"talk": 1 - point, "point": point}, kw


def crocus_full(T, st):
    """crocus_pose with IK offsets resolved against the rig (pure)"""
    pose, kw, grip = crocus_pose(T)
    dx = kw.pop("_hand_dx", 0.0)
    if dx:
        a = st["crocus"].anchors(0, 0, CROCUS["h"], pose, T, facing=-1, pole_len=POLE, **kw)
        hx, hy = a["hand_r"]
        kw["hand_r"] = (hx + dx, hy)
    return pose, kw, grip


def flag_pole_world(T, st):
    """pole base (X, Y, Z) in world cm + ang for the hypermeme Flag at time T"""
    ped = np.array([0.0, Wd.PED_Y, 0.0])
    pose, kw, grip = crocus_full(T, st)
    if grip <= 0:
        return ped, 0.0
    a = st["crocus"].anchors(0, 0, CROCUS["h"], pose, T, facing=-1, pole_len=POLE, **kw)
    if "pole" not in a:
        return ped, 0.0
    bx, by, ang = a["pole"]
    hand = np.array([CROCUS["x"] + bx, -by, CROCUS["z"]])
    g = smootherstep(0.35, 1.0, grip) if grip < 1 else 1.0
    P = ped * (1 - g) + hand * g
    # "before the true one can RISE": thrust the Flag up through his grip
    P[1] += 26.0 * seg(T, 86.55, 86.9, ease_out_back) * (1 - seg(T, 87.3, 87.62))
    return P, ang * g


# ------------------------------------------------------------------ setup
def setup(S):
    st = {}
    st["world"] = Wd.build()
    st["hm"] = HM.build()
    st["floor"] = Wd.floor_texture(S.cache)
    st["map"] = MP.build(S.cache)
    st["streams"] = FX.build_streams(st["world"]["consoles"])
    st["crocus"] = Character("crocus", 1)
    st["lutie"] = Character("lutie", 2)
    st["acolytes"] = [Character("acolyte", 10 + k) for k in range(5)]
    # map-table node flags (tiny plain Survey Flags on the geomantic nodes)
    nodes = []
    for nm, (la, lo) in MP.NODES.items():
        u, v = MP.ll_to_uv(la, lo)
        X, Z = MP.uv_to_world(u, v, Wd.RT)
        nodes.append((X, Z, hash01(len(nodes), 9)))
    st["nodes"] = nodes
    # hypermeme Flag: real cloth sim from just before materialization to the cut
    f_sim0 = max(0, int(round((72.45 - S.start) * S.fps)))
    n = S.n - f_sim0
    st["f_sim0"] = f_sim0

    def pole_fn(i):
        T = S.start + (f_sim0 + i) / S.fps
        P, ang = flag_pole_world(T, st)
        return (P[0] * SIM_K, -(P[1] - Wd.PED_Y) * SIM_K, ang)

    def wind_fn(i):
        T = S.start + (f_sim0 + i) / S.fps
        w = 380 + 110 * math.sin(T * 0.7) + 420 * seg(T, 72.6, 72.75) * (1 - seg(T, 72.9, 73.6))
        w += 520 * seg(T, T_SEAL + 0.1, T_SEAL + 0.6)      # the seal's updraft
        return (-w, -30.0, 40.0 * math.sin(T * 1.3))

    fl = Flag(pole=420, seed=7, side=-1)
    tr = fl.simulate(n, pole_fn, wind_fn, fps=S.fps, warmup=24)
    st["flag"] = tr
    _export_foley(S, st)
    return st


def _export_foley(S, st):
    hits = [
        dict(t=61.50, kind="whoosh", strength=0.7, pan=0.0, desc="from black: plunge down the pentagonal basalt shaft (fast fall, lamps whip past)"),
        dict(t=62.95, kind="whoosh", strength=0.8, pan=0.2, desc="camera exits the shaft mouth past the Hypermind's top plate, tilts up (air opens up, huge reverb)"),
        dict(t=63.00, kind="hum", strength=0.9, pan=0.0, desc="Hypermind reveal: vast cryogenic hum / pulse tone (it speaks H01)"),
    ]
    for k, t in enumerate(FX.SCROLL_T):
        hits.append(dict(t=t, kind="absorb", strength=0.65 + 0.05 * k, pan=float(np.clip(math.sin(k / 6 * 2 * math.pi + t * 0.32) * 0.6, -1, 1)),
                         desc=f"canon scroll {FX.CANON_NAMES[k]} sucked into the Hypermind (paper swirl -> bright chime/flash)"))
    for i, t in enumerate(np.arange(63.1, 64.7, 0.2)):
        hits.append(dict(t=float(t), kind="tick", strength=0.25, pan=0.0, desc="readout LOSS digits tick down"))
    hits.append(dict(t=64.67, kind="click", strength=0.5, pan=0.0, desc="LOSS reaches 0.0000 (readout lock)"))
    hits.append(dict(t=70.11, kind="chime", strength=0.7, pan=0.1, desc="readout prints FLAGS: TRUE (on 'true')"))
    for k in range(6):
        hits.append(dict(t=71.30 + 0.17 * k, kind="energy", strength=0.5 + 0.08 * k, pan=0.0,
                         desc=f"H02: energy pulse descends plate {k} of the chandelier (descending thrum)"))
    hits.append(dict(t=T_BEAM, kind="beam", strength=0.9, pan=0.0, desc="beam fires from the Hypermind tip down to the pedestal"))
    hits.append(dict(t=T_MAT, kind="materialize", strength=1.0, pan=0.0, desc="HYPERMEME: a plain yellow Flag materializes on the pedestal (shimmer -> soft cloth unfurl), then reverent hush"))
    hits.append(dict(t=84.05, kind="grab", strength=0.4, pan=0.25, desc="Crocus grips the pole, lifts the Flag out of the pedestal socket (wood scrape)"))
    hits.append(dict(t=86.81, kind="whoosh", strength=0.5, pan=0.25, desc="Crocus raises the Flag high on 'rise'"))
    hits.append(dict(t=T_STRIKE, kind="thunk", strength=1.0, pan=0.25, desc="pole butt STRIKES the basalt floor on 'Summon' (deep stone thunk + gold shockwave)"))
    hits.append(dict(t=T_SEAL, kind="unlock", strength=0.8, pan=0.5, desc="pentagonal floor seal unlocks (heavy mechanical clacks)"))
    hits.append(dict(t=88.35, kind="grind", strength=0.9, pan=0.5, desc="seal petals grind open 88.35-89.0, column of gold light roars up"))
    for i, t in enumerate(np.arange(88.40, 90.0, 0.45)):
        hits.append(dict(t=float(t), kind="klaxon", strength=0.8, pan=float(math.sin(i * 2.1) * 0.7), desc="gold klaxon sweep"))
    hits.append(dict(t=T_BRAAM, kind="swell", strength=0.9, pan=0.0, desc="push-in accent (braam) + yellow flash swell to the cut"))
    hits.append(dict(t=89.95, kind="cut", strength=1.0, pan=0.0, desc="hard cut to s05a"))
    foley.export_events(S, "hits", hits)
    # the hypermeme Flag: physics-derived flutter (silent until it materializes at 72.7)
    from vx.flag import FlagTrack
    tr = st["flag"]
    f0 = st["f_sim0"]
    k0 = max(0, int(round((T_MAT - 0.04 - S.start) * S.fps)) - f0)
    en = tr.energy.copy()
    sn = tr.snap.copy()
    en[:k0] = 0
    sn[:k0] = 0
    quiet = FlagTrack(tr.verts, tr.poles, en, sn, tr.meta, tr.flap_hz)
    pan = np.zeros(S.n, np.float32)
    gain = np.zeros(S.n, np.float32)
    for f in range(S.n):
        sid = shot_at(S.start + f / S.fps)[0]
        pan[f] = {"lutie": 0.55, "crocus": -0.35, "summon": -0.1, "raise_": 0.1, "map": 0.0}.get(sid, 0.0)
    quiet.export_foley(S, "hypermeme", pan=pan, f0=f0, gain=0.7)


# ------------------------------------------------------------------ render helpers
def lights_for(T, shot, st):
    m = get_timeline().mouth("HYPERMIND", T)
    L = [Wd.Light((0, 3200, 0), Wd.WARM, 1.05 + 0.6 * m, 5200),
         Wd.Light((0, 5300, 0), Wd.WARM, 2.2, 1500),
         Wd.Light((0, 300, 0), col("#6FB7C8"), 0.35, 1200),
         Wd.Light((0, 160, -900), col("#5A6E96"), 0.55, 1400)]
    if T > T_BEAM:
        b = seg(T, T_BEAM, T_BEAM + 0.15) * (0.5 + 0.5 * (1 - seg(T, 73.0, 74.0)))
        L.append(Wd.Light((0, 500, 0), Wd.WARM, 1.6 * b, 900))
    if T > T_MAT and st is not None:
        a = seg(T, T_MAT - 0.05, T_MAT + 0.2)
        L.append(Wd.Light(flag_cloth_world(T, st), col("#FFD35A"), 0.9 * a, 260))
    if T > T_SEAL:
        o = seg(T, 88.3, 88.8)
        L.append(Wd.Light(Wd.SEAL_C + np.array([0, 900, 0]), col("#FFC060"), 2.6 * o, 1800))
        k = klaxon_level(T)
        for i in range(5):
            th = i * 2 * math.pi / 5 + math.pi / 5 + math.pi
            p = np.array([4000 * math.sin(th), 2350, 4000 * math.cos(th)])
            sw = 0.5 + 0.5 * math.cos(T * 7.0 + i * 1.3)
            L.append(Wd.Light(p, col("#FFA030"), 2.2 * k * sw ** 4, 2600))
    return L


def klaxon_level(T):
    return seg(T, 88.4, 88.7) if T > 88.4 else 0.0


def readout_text(T):
    if T < 64.67:
        v = 0.0137 * (1 - seg(T, 62.6, 64.67, lambda x: ease_out(x, 2.5)))
        loss = f"{v:.4f}"
    else:
        loss = "0.0000"
    flags = "TRUE" if T >= 70.11 else "\u00b7\u00b7\u00b7\u00b7"
    one = f"LOSS {loss} \u00b7 FLAGS: {flags} \u00b7 CANONS I\u2013VI \u00b7 "
    return one * 3


def ring_base(T, text):
    """rotate the readout ring so that 'FLAGS: TRUE' faces the console camera exactly when it prints (70.11)"""
    n = len(text)
    step = 2 * math.pi / n
    j = text.index("FLAGS") + 5
    c = cam_console(70.11).pos
    phi = math.atan2(c[0], c[2])
    return phi + j * step + 0.10 * (T - 70.11)


def energy_fn(T):
    """H02: an energy pulse drops down the cold stages; plates stay a little brighter after it passes"""
    def e(k):
        if T < 71.2:
            return 0.0
        tk = 71.30 + 0.17 * k
        return 1.8 * math.exp(-((T - tk) / 0.11) ** 2) + 0.35 * seg(T, tk, tk + 0.1) * (1 - seg(T, 73.2, 74.5))
    return e


def fogf_for(cam):
    def f(P):
        return Wd.fog_amount(P, cam)
    return f


def draw_char_bb(ctx, cam, ch, x, z, h, pose, T, facing, alpha=1.0, **kw):
    """character as a cylindrical billboard linearised at mid-height (accurate for low/high angles)"""
    ym = h * 0.55
    M, zz = cam.billboard(np.array([x, ym, z]))
    if M is None:
        return None
    ctx.save()
    ctx.transform(M)
    ctx.translate(0, ym)
    try:
        a = ch.draw(ctx, 0, 0, h, pose, T, facing=facing, alpha=alpha, **kw)
    finally:
        ctx.restore()
    return a


# ------------------------------------------------------------------ render
def render(fc, st):
    """motion blur (3 sub-frames over a 180-degree shutter) while the camera plunges down the shaft"""
    if 61.5 <= fc.T < 63.2:
        import dataclasses
        acc = None
        subs = (0.0, -1 / 72, -2 / 72)
        for dt in subs:
            img = _render(dataclasses.replace(fc, T=fc.T + dt, t=fc.t + dt), st)
            acc = img if acc is None else acc + img
        return acc / len(subs)
    return _render(fc, st)


def _render(fc, st):
    T = fc.T
    sid, t_shot = shot_at(T)
    cam = CAMS[sid](T)
    tl = fc.tl
    lights = lights_for(T, sid, st)
    fogf = fogf_for(cam)
    hm_m = tl.mouth("HYPERMIND", T)

    # ---- base: floor (projective texture) under everything
    base = np.zeros((fc.h, fc.w, 3), np.float32)
    if cam.pos[1] > 0:
        fl = Wd.warp_plane(st["floor"], cam, 0.0, Wd.RC, fc)
        glowk = 1.0 + 0.35 * hm_m
        base = base * (1 - fl[..., 3:]) + fl[..., :3] * glowk
    L = FX.Layers(fc, base)
    ops = []
    # ---- shell (farthest)
    shaft_first = cam.pos[1] < Wd.YV
    ops.append((1e9, "c", lambda c, g: Wd.draw_shell(c, g, cam, st["world"], lights, T, shaft_first)))
    ops.append((9e8, "c", lambda c, g: _haze(c, cam, T, hm_m)))
    zh = float(cam.to_cam(np.array([0.0, HM.HUB_Y, 0.0]))[2])
    if zh > cam.near or cam.pos[1] > HM.HUB_Y:
        ops.append((zh, "c", lambda c, g: Wd.draw_truss(c, g, cam, lights, HM.HUB_Y)))
    # ---- Hypermind (one op at its axis depth)
    phi = math.atan2(cam.pos[0], cam.pos[2]) - 0.7
    light_dir = np.array([math.sin(phi), 0, math.cos(phi)])
    z_axis = float(cam.to_cam(np.array([0.0, np.clip(cam.pos[1], 1300, 5300), 0.0]))[2])
    en = energy_fn(T)
    tipg = 0.0
    if T > 71.2:
        tipg = 2.0 * math.exp(-((T - T_BEAM) / 0.25) ** 2) + 0.6 * seg(T, T_BEAM - 0.05, T_BEAM + 0.1) * (1 - seg(T, 73.2, 74.2))
    ops.append((z_axis, "c", lambda c, g: HM.draw_hypermind(c, g, cam, st, T, tl, light_dir,
                                                               cam.pos * np.array([1, 0, 1]), fogf,
                                                               energy=en, tip_glow=tipg)))
    # readout ring around the mixing-chamber can
    ring_a = seg(T, 62.9, 63.6)
    rtxt = readout_text(T)
    ops.append((z_axis - 5, "c", lambda c, g: HM.draw_readout_ring(
        c, g, cam, T, rtxt, y=1360.0, r=700.0, size=104.0, alpha=ring_a, hot=_true_flash(T),
        base=ring_base(T, rtxt))))
    # ---- consoles, acolytes, table, pedestal, cast
    _world_ops(ops, cam, st, T, lights, fogf, sid, fc)
    # ---- scripture, scrolls
    amt = seg(T, 62.7, 63.6) * (1 - seg(T, 70.6, 71.6)) * (1 - 0.6 * seg(T, 67.7, 68.2))
    FX.stream_ops(st["streams"], cam, T, amt, fogf, ops)
    for k in range(6):
        s_ = FX.scroll_state(k, T)
        if s_ is None:
            continue
        P, sc, al, flash = s_
        z = float(cam.to_cam(P)[2])
        if z > cam.near:
            ops.append((z, "c", (lambda P_, k_, sc_, al_, fl_: lambda c, g: _scroll_op(c, g, cam, P_, k_, sc_, al_, fl_, fogf, T))(P, k, sc, al, flash)))
    # ---- beam
    if T_BEAM - 0.05 < T < 74.6:
        ba = seg(T, T_BEAM - 0.05, T_BEAM + 0.06) * (1 - 0.92 * seg(T, T_MAT - 0.02, T_MAT + 0.14)) * (1 - seg(T, 73.4, 74.2))
        zb = float(cam.to_cam(np.array([0.0, 700.0, 0.0]))[2])
        ops.append((zb - 8, "c", lambda c, g: FX.draw_beam(c, g, cam, np.array([0.0, HM.TIP_Y, 0.0]),
                                                              np.array([0.0, Wd.PED_Y, 0.0]), 4.5, ba, T)))
    # ---- klaxon beacons
    kl = klaxon_level(T)
    if kl > 0:
        for i in range(5):
            th = i * 2 * math.pi / 5 + math.pi / 5 + math.pi
            p = np.array([Wd.RC * 0.78 * math.sin(th), 2330.0, Wd.RC * 0.78 * math.cos(th)])
            zk = float(cam.to_cam(p)[2])
            if zk > cam.near:
                ops.append((zk, "c", (lambda p_, i_: lambda c, g: _draw_klaxon(c, g, cam, p_, i_, T, kl))(p, i)))
    # ---- depth of field for the close shots (everything behind the focus plane is softened)
    dof = {"manifest": (620, 2.2), "lutie": (420, 7.0), "crocus": (560, 5.0), "summon": (700, 3.2),
           "raise_": (520, 4.0)}.get(sid)
    if dof is not None:
        ops.append((dof[0], "n", lambda img: blur(img, dof[1] * fc.s) * 0.86))
    L.run(ops)
    # ---- motes & atmosphere
    ctx = L.ctx()
    gctx = L.gctx
    FX.motes(ctx, gctx, cam, T, 1, 70, np.array([0.0, 2400.0, 0.0]), np.array([3200.0, 2400.0, 3200.0]), rise=60, size=7, alpha=0.55)
    if sid in ("manifest", "lutie", "crocus", "map", "summon", "raise_"):
        conv = None
        if T < T_MAT + 0.1 and T > T_BEAM:
            conv = (np.array([0.0, 260.0, 0.0]), seg(T, T_BEAM, T_MAT, ease_in))
        FX.motes(ctx, gctx, cam, T, 2, 90, np.array([0.0, 260.0, 0.0]), np.array([420.0, 260.0, 420.0]), rise=18, size=1.6, alpha=0.7, converge=conv)
    img = L.finish(glow_k=1.0)
    # god rays from the Hypermind core / tip / seal
    gx, gy, gz = cam.p2(np.array([0.0, 2600.0, 0.0]))
    if gz > 0:
        img = FX.god_rays(img, gx, gy, 0.3 + 0.25 * hm_m, fc, thresh=0.72)
    if T > 88.3 and sid == "summon":
        sx, sy, sz = cam.p2(Wd.SEAL_C + np.array([0, 1200, 0]))
        img = FX.god_rays(img, sx, sy, 0.55 * seg(T, 88.35, 89.0), fc, thresh=0.6, tint="#FFD27A")
    return np.clip(img, 0, 4)


def _draw_klaxon(ctx, gctx, cam, p, i, T, kl):
    """a rotating amber beacon on the wall: housing, hot lamp, sweeping light blade (+ flare toward camera)"""
    ang = T * 7.0 + i * 1.3
    x, y, z = cam.p2(p)
    r = max(1.5, 45 * cam.f / z)
    ctx.set_source_rgb(*col("#1A1410"))
    ctx.rectangle(x - r * 1.3, y, r * 2.6, r * 0.9)
    ctx.fill()
    d = np.array([math.cos(ang), -0.22, math.sin(ang)])
    far = p + d * 3200
    perp = np.array([-math.sin(ang), 0.0, math.cos(ang)]) * 520
    tri = np.array([p, far + perp, far - perp])
    pts = poly_screen(cam, tri)
    if pts is not None:
        gx, gy = pts[0]
        mx, my = (pts[1] + pts[2]) / 2
        for c_, a_ in ((ctx, 0.22), (gctx, 0.35)):
            g = cairo.LinearGradient(gx, gy, mx, my)
            g.add_color_stop_rgba(0, *col("#FFC060"), a_ * kl)
            g.add_color_stop_rgba(1, *col("#FF9A20"), 0)
            path_pts(c_, pts)
            c_.set_source(g)
            c_.fill()
    # lamp + flare when the blade faces the camera
    to_cam = cam.pos - p
    to_cam /= np.linalg.norm(to_cam)
    face = max(0.0, float(np.dot(d / np.linalg.norm(d), to_cam))) ** 6
    for c_, k, a_ in ((ctx, 1.0, 1.0), (gctx, 4.0 + 10 * face, 0.6 + 0.4 * face)):
        g = cairo.RadialGradient(x, y, 0, x, y, r * k)
        g.add_color_stop_rgba(0, *col("#FFE6B0"), a_ * kl)
        g.add_color_stop_rgba(0.4, *col("#FFA530"), a_ * kl * 0.6)
        g.add_color_stop_rgba(1, *col("#FF8A10"), 0)
        c_.set_source(g)
        c_.arc(x, y, r * k, 0, 2 * math.pi)
        c_.fill()


def _haze(ctx, cam, T, m):
    """painterly atmosphere between the shell and the objects: warm volumetric blooms around the Hypermind
    (cairo additive radial gradients, no numpy flush)"""
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    for yy, rr, k in ((3600.0, 2600.0, 0.11), (1900.0, 1300.0, 0.07)):
        hx, hy, hz = cam.p2(np.array([0.0, yy, 0.0]))
        if hz <= cam.near:
            continue
        r0 = rr * cam.f / hz
        if r0 > 6000 or not (-2 * r0 - 2000 < hx < 1920 + 2 * r0 + 2000 and -3 * r0 - 2000 < hy < 1080 + 3 * r0 + 2000):
            continue
        ctx.save()
        ctx.translate(hx, hy)
        ctx.scale(r0, r0 * 1.5)
        g = cairo.RadialGradient(0, 0, 0, 0, 0, 2.2)
        a = min(1.0, k * (1 + 0.6 * m))
        for t_ in np.linspace(0, 1, 7):
            g.add_color_stop_rgba(t_, *Wd.WARM, a * math.exp(-(t_ * 2.2) ** 2))
        ctx.set_source(g)
        ctx.arc(0, 0, 2.2, 0, 2 * math.pi)
        ctx.fill()
        ctx.restore()
    ctx.restore()


def _true_flash(T):
    return math.exp(-((T - 70.15) / 0.2) ** 2) if T > 69.9 else 0.0


def _scroll_op(ctx, gctx, cam, P, k, sc, al, flash, fogf, T):
    x, y, z = cam.p2(P)
    if -600 < x < 2520 and -600 < y < 1680:
        occluding(ctx, gctx, lambda c: FX.draw_scroll(c, gctx, cam, P, k, sc, al, fogf(P), T))
        M, _ = cam.billboard(P)
        if M is not None and al > 0.02:
            # warm rim glow around the sheet (drawn after the knock-out so it survives)
            gctx.save()
            gctx.transform(M)
            gctx.rotate(math.sin(T * 1.7 + k) * 0.06)
            gctx.rectangle(-190 * sc, -250 * sc, 380 * sc, 500 * sc)
            gctx.set_source_rgba(*HM.EMIT, 0.55 * al)
            gctx.set_line_width(26 * sc)
            gctx.stroke()
            gctx.restore()
    if flash > 0.02:
        x, y, z = cam.p2(P)
        r = 260 * cam.f / max(z, 1) * (0.4 + flash)
        g = cairo.RadialGradient(x, y, 0, x, y, r)
        g.add_color_stop_rgba(0, *HM.EMIT_HOT, flash)
        g.add_color_stop_rgba(1, *HM.EMIT, 0)
        for c_ in (ctx, gctx):
            c_.set_source(g)
            c_.arc(x, y, r, 0, 2 * math.pi)
            c_.fill()


def _world_ops(ops, cam, st, T, lights, fogf, sid, fc):
    geo = st["world"]
    # consoles + acolytes
    for k, cs in enumerate(geo["consoles"]):
        th = cs["th"]
        P = cs["pos"]
        z = float(cam.to_cam(P)[2])
        if z < cam.near:
            continue
        dim = 0.45 if sid in ("lutie", "crocus", "raise_") else 1.0
        ops.append((z, "c", (lambda th_, P_: lambda c, g: _draw_console(c, g, cam, th_, P_, lights, T, dim))(th, P)))
        pa = np.array([930 * math.sin(th), 0.0, 930 * math.cos(th)])
        za = float(cam.to_cam(pa)[2])
        if za > cam.near * 5:
            ops.append((za, "c", (lambda k_, pa_, th_: lambda c, g: _draw_acolyte(c, g, cam, st, k_, pa_, th_, T))(k, pa, th)))
    # table + pedestal (+ map texture as a numpy op)
    zt = float(cam.to_cam(np.array([0.0, Wd.YT, 0.0]))[2])
    if zt > cam.near:
        ops.append((zt + 1.0, "c", lambda c, g: _draw_table(c, g, cam, lights)))
        if cam.pos[1] > Wd.YT:
            ops.append((zt + 0.5, "c", lambda c, g: _map_op(c, cam, st, T, fc)))
        ops.append((zt - 0.5, "c", lambda c, g: _draw_pedestal_and_nodes(c, g, cam, st, T, lights)))
    # Lutie, Crocus and the hypermeme Flag
    tl = get_timeline()
    zl = float(cam.to_cam(np.array([LUTIE["x"], 80, LUTIE["z"]]))[2])
    if zl > cam.near * 5:
        pose, kw = lutie_pose(T)
        ops.append((zl, "c", lambda c, g: _draw_hero(c, g, cam, st["lutie"], LUTIE, pose, T, 1, tl.mouth("LUTIE", T), kw, lights, st=st)))
    cpose, ckw, grip = crocus_full(T, st)
    zc = float(cam.to_cam(np.array([CROCUS["x"], 90, CROCUS["z"]]))[2])
    if grip > 0:
        # in his hand: back layer, Flag, front layer (one op)
        if zc > cam.near * 5:
            def _held(c, g):
                _draw_hero(c, g, cam, st["crocus"], CROCUS, cpose, T, -1, tl.mouth("CROCUS", T), ckw, lights, layer="back", st=st)
                _draw_hypermeme(c, g, cam, st, T)
                _draw_hero(c, g, cam, st["crocus"], CROCUS, cpose, T, -1, tl.mouth("CROCUS", T), ckw, lights, layer="front", st=st)
            ops.append((zc, "c", _held))
    else:
        if zc > cam.near * 5:
            ops.append((zc, "c", lambda c, g: _draw_hero(c, g, cam, st["crocus"], CROCUS, cpose, T, -1, tl.mouth("CROCUS", T), ckw, lights, st=st)))
        if T >= T_BEAM:
            zf = float(cam.to_cam(np.array([0.0, Wd.PED_Y + 60, 0.0]))[2])
            if zf > cam.near:
                ops.append((zf - 1.0, "c", lambda c, g: _draw_hypermeme(c, g, cam, st, T)))
    # the seal
    zs = float(cam.to_cam(Wd.SEAL_C)[2])
    if zs > cam.near:
        ops.append((zs, "c", lambda c, g: _draw_seal(c, g, cam, T, lights)))


def _light_rim(cam, P, lights):
    """strongest light direction on screen at world point P -> (dx, dy) + colour for a rim light"""
    best, bc, bd = 0.0, None, (0.0, -1.0)
    for Lt in lights:
        d = Lt.pos - P
        dist = float(np.linalg.norm(d)) + 1e-3
        att = Lt.power / (1 + (dist / Lt.rad) ** 2)
        if att > best:
            best = att
            bc = Lt.col
            sx0, sy0, _ = cam.p2(P)
            sx1, sy1, z1 = cam.p2(P + d / dist * 50)
            v = np.array([sx1 - sx0, sy1 - sy0])
            n = np.linalg.norm(v) + 1e-6
            bd = (float(v[0] / n), float(v[1] / n))
    return best, bc, bd


def flag_cloth_world(T, st):
    """approximate world position of the hypermeme cloth centre (for key light / glow)"""
    P, ang = flag_pole_world(T, st)
    return P + np.array([-0.2 * POLE, 0.85 * POLE, 0.0])


def _screen_dir(cam, P0, P1):
    x0, y0, _ = cam.p2(P0)
    x1, y1, z1 = cam.p2(P1)
    v = np.array([x1 - x0, y1 - y0])
    n = np.linalg.norm(v) + 1e-6
    return float(v[0] / n), float(v[1] / n)


def _draw_hero(ctx, gctx, cam, ch, pos, pose, T, facing, mouth, kw, lights, layer="all", st=None):
    head = np.array([pos["x"], pos["h"] * 0.85, pos["z"]])
    kw = dict(kw)
    if st is not None and T > T_MAT:
        kw.setdefault("light", _screen_dir(cam, head, flag_cloth_world(T, st)))
    else:
        k, c, d = _light_rim(cam, head, lights)
        kw.setdefault("light", d)
    seal = seg(T, 88.35, 88.9)
    if seal > 0:
        kw.setdefault("rim", (tuple(col("#FFD890")), 0.35 + 0.3 * seal, _screen_dir(cam, head, Wd.SEAL_C + [0, 400, 0])))
    else:
        kw.setdefault("rim", (tuple(np.clip(Wd.WARM * 0.9 + 0.1, 0, 1)), 0.38, _screen_dir(cam, head, head + [0, 300, 0])))
    kw.setdefault("shade", 0.55)
    kw.setdefault("tint", ("#18203A", 0.30 * (1 - 0.5 * seal)))
    return occluding(ctx, gctx, lambda c: draw_char_bb(c, cam, ch, pos["x"], pos["z"], pos["h"], pose, T, facing,
                                                       mouth=mouth, layer=layer, **kw))


def occluding(ctx, gctx, draw):
    """draw an opaque thing via a temp surface so it also knocks out the additive glow canvas behind it"""
    tgt = ctx.get_target()
    w, h = tgt.get_width(), tgt.get_height()
    tmp = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    tc = cairo.Context(tmp)
    tc.set_matrix(ctx.get_matrix())
    tc.set_line_join(cairo.LINE_JOIN_ROUND)
    tc.set_line_cap(cairo.LINE_CAP_ROUND)
    res = draw(tc)
    tmp.flush()
    for c_, op in ((ctx, cairo.OPERATOR_OVER), (gctx, cairo.OPERATOR_DEST_OUT)):
        if c_ is None:
            continue
        c_.save()
        c_.identity_matrix()
        c_.set_operator(op)
        c_.set_source_surface(tmp, 0, 0)
        c_.paint()
        c_.restore()
    return res


def _draw_acolyte(ctx, gctx, cam, st, k, pa, th, T):
    # facing: toward the axis, as seen from this camera
    sx0, _, z0 = cam.p2(pa)
    sx1, _, z1 = cam.p2(np.array([0.0, 0.0, 0.0]))
    far = z0 > z1 + 150
    dx = sx1 - sx0
    lean = 0.14 * seg(T, T_MAT - 0.1, T_MAT + 0.6) * (1 - seg(T, 75.0, 76.5))
    look_up = 0.6 * seg(T, 62.9, 63.6) * (1 - seg(T, 72.3, 72.8))
    kw = dict(expr="awe", look=(0.0, -look_up), lean=lean)
    if far:
        facing = 0 if abs(dx) < 150 else (1 if dx > 0 else -1)
        kw["turn"] = 0.35
    else:
        facing = 1 if dx > 0 else -1
    near_back = not far and abs(dx) < 500
    if near_back:
        kw = dict(back=True, look=(0.0, -look_up), rim=(tuple(Wd.WARM), 0.5, (0.0, -1.0)), shade=0.7)
        facing = 1 if dx > 0 else -1
    kw.setdefault("tint", ("#141B32", 0.42))
    ch = st["acolytes"][k]
    draw_char_bb(ctx, cam, ch, pa[0], pa[2], 165.0 + 6 * math.sin(k * 3.1), "stand", T + k, facing, **kw)


def _draw_console(ctx, gctx, cam, th, P, lights, T, dim=1.0):
    faces = Wd.box_faces(P, 120, 38, 88, th)
    Wd.draw_faces(ctx, cam, faces, Wd.BASALT * 0.8, lights)
    # slanted screen facing outward (toward the acolyte), glowing amber text lines
    ct, st_ = math.cos(th), math.sin(th)
    ex = np.array([ct, 0, -st_])
    ez = np.array([st_, 0, ct])
    c0 = np.array([P[0], 88.0, P[2]]) + ez * 10
    tilt = math.radians(58)
    up = np.array([0, math.sin(tilt), 0]) - ez * math.cos(tilt) * -1
    up = np.array([ez[0] * math.cos(tilt), math.sin(tilt), ez[2] * math.cos(tilt)])
    w, h = 110.0, 62.0
    S = np.array([c0 - ex * w, c0 + ex * w, c0 + ex * w + up * h, c0 - ex * w + up * h])
    nrm = np.cross(ex, up)
    pts = poly_screen(cam, S)
    if pts is None:
        return
    ctx.set_source_rgb(*col("#0A0C12"))
    path_pts(ctx, pts)
    ctx.fill()
    if np.dot(cam.pos - S[0], ez) <= 0 and np.dot(cam.pos - S[0], nrm) <= 0:
        pass
    # text lines (little glowing bars), scroll with time
    rows = 7
    for r_ in range(rows):
        v = (r_ + 0.5) / rows
        ln = 0.3 + 0.6 * hash01(int(T * 3) + r_ * 17 + int(th * 10), 3)
        a = c0 - ex * w * 0.85 + up * h * (1 - v) * 0.9
        b = a + ex * w * 1.7 * ln
        sp = poly_screen(cam, np.array([a, b, b + up * 1, a + up * 1]))
        if sp is None:
            continue
        z = float(min(cam.to_cam(a)[2], cam.to_cam(b)[2]))
        if z < 60:
            continue
        lw = min(18.0, max(0.5, 4.5 * cam.f / z))
        for c_, aa, ww in ((ctx, 0.9 * dim, lw), (gctx, 0.5 * dim * dim, lw * 3)):
            c_.move_to(*sp[0])
            c_.line_to(*sp[1])
            c_.set_source_rgba(*col("#FFC98A"), aa)
            c_.set_line_width(ww)
            c_.stroke()


def _draw_table(ctx, gctx, cam, lights):
    fs, top = Wd.prism_faces(Wd.RT * 0.9, 0.0, Wd.YT - 8)
    Wd.draw_faces(ctx, cam, fs, np.array(col("#3A4258")), lights)
    fs2, top2 = Wd.prism_faces(Wd.RT, Wd.YT - 8, Wd.YT)
    Wd.draw_faces(ctx, cam, fs2, np.array(col("#8A6A3A")), lights)
    # gold trims: base ring, a band under the slab, and the slab edge
    for ring, a_ in ((ngon(5, Wd.RT * 0.9, 6.0, math.pi), 0.55), (ngon(5, Wd.RT * 0.9, Wd.YT - 20, math.pi), 0.7),
                     (top2, 0.95)):
        pts = poly_screen(cam, ring)
        if pts is None:
            continue
        zz = float(np.max(cam.to_cam(ring)[:, 2]))
        lw = min(4.0, max(0.6, 1.4 * cam.f / max(zz, 1)))
        path_pts(ctx, pts)
        ctx.set_source_rgba(*col("#E0A860"), a_)
        ctx.set_line_width(lw)
        ctx.stroke()
        path_pts(gctx, pts)
        gctx.set_source_rgba(*col("#FFB456"), 0.25 * a_)
        gctx.set_line_width(lw * 3)
        gctx.stroke()


def _map_op(ctx, cam, st, T, fc):
    """the map-table top: projective texture warps restricted to the table's screen bbox, composited in
    numpy and painted into the cairo layer (no full-frame flush)"""
    tex = st["map"]
    r = Wd.warp_plane(tex["base"], cam, Wd.YT, Wd.RT, fc, roi=True)
    if r is None:
        return
    base, x0, y0 = r
    if base.shape[0] < 3 or base.shape[1] < 3:
        return
    pulse = 0.55 + 0.25 * math.sin(T * 2.2) + 0.3 * get_timeline().mouth("HYPERMIND", T)
    ck = 0.35 + 0.65 * seg(T, 79.9, 80.6) * (1 - 0.5 * seg(T, 82.5, 83.3))
    if T > 83.3:
        ck = 0.4
    img = base.copy()
    cr = Wd.warp_plane(tex["cracks"], cam, Wd.YT, Wd.RT, fc, roi=True)[0]
    img = cr * (ck * 0.85) + img * (1 - cr[..., 3:] * ck * 0.85)
    ley = Wd.warp_plane(tex["ley"], cam, Wd.YT, Wd.RT, fc, roi=True)[0]
    cg = Wd.warp_plane(tex["crack_glow"], cam, Wd.YT, Wd.RT, fc, roi=True)[0]
    img[..., :3] += (ley * (pulse * 0.7) + cg * (ck * 0.5)) * base[..., 3:]
    paint_array(ctx, np.clip(img, 0, 1), x0 / fc.s, y0 / fc.s, fc.s)


def _draw_pedestal_and_nodes(ctx, gctx, cam, st, T, lights):
    # tiny plain Survey Flags on the geomantic nodes (draw_flag, billboards)
    from vx.flag import draw_flag
    z0 = float(cam.to_cam(np.array([0.0, Wd.YT, 0.0]))[2])
    if cam.f / z0 * 14 > 3.0:
        for i, (X, Z, h) in enumerate(st["nodes"]):
            M, z = cam.billboard(np.array([X, Wd.YT, Z]))
            if M is None:
                continue
            ctx.save()
            ctx.transform(M)
            ctx.scale(0.03, 0.03)
            draw_flag(ctx, 0, 0, pole=420, t=T + h * 5, wind=0.35, seed=i)
            ctx.restore()
    # pedestal: small pentagonal plinth with a gold socket ring
    fs, top = Wd.prism_faces(20.0, Wd.YT, Wd.PED_Y)
    Wd.draw_faces(ctx, cam, fs, np.array(col("#C98A3A")), lights)
    pts = poly_screen(cam, top)
    if pts is not None:
        path_pts(ctx, pts)
        ctx.set_source_rgb(*col("#2A2018"))
        ctx.fill_preserve()
        ctx.set_source_rgba(*col("#E8B06A"), 0.9)
        ctx.set_line_width(max(0.6, 1.2 * cam.f / max(float(cam.to_cam(top[0])[2]), 1)))
        ctx.stroke()


def _draw_hypermeme(ctx, gctx, cam, st, T):
    tr = st["flag"]
    i = (T - 61.5) * 24.0 - st["f_sim0"]
    if i < 0:
        return
    Pb, ang = flag_pole_world(T, st)
    # anchor: billboard linearised at the Flag's mid-height, at the pole's current depth; sim space carries X/Y
    ym = Pb[1] + POLE * 0.6
    M, z = cam.billboard(np.array([0.0, ym, Pb[2]]))
    if M is None:
        return
    M = cairo.Matrix(1, 0, 0, 1, 0, ym - Wd.PED_Y).multiply(M)
    # materialization: the pole draws itself as a line of light, then the cloth appears
    grow = seg(T, T_BEAM + 0.02, T_MAT - 0.02, ease_out)
    appear = seg(T, T_MAT - 0.06, T_MAT + 0.14)
    light = (-0.3, -0.55, 0.75)
    ctx.save()
    ctx.transform(M)
    ctx.scale(1 / SIM_K, 1 / SIM_K)
    if appear < 1:
        # light line
        xb, yb, a_ = tr.poles[min(int(i), tr.n - 1)]
        L_ = 420 * grow
        for c_, w_, aa in ((ctx, 5.0, 0.95), (gctx, 22.0, 0.8)):
            if c_ is gctx:
                c_.save()
                c_.transform(M)
                c_.scale(1 / SIM_K, 1 / SIM_K)
            c_.move_to(xb, yb)
            c_.line_to(xb, yb - L_)
            c_.set_source_rgba(*col("#FFF1B8"), aa * (1 - appear))
            c_.set_line_width(w_)
            c_.stroke()
            if c_ is gctx:
                c_.restore()
    if appear > 0:
        try:
            tr.draw(ctx, min(i, tr.n - 1), alpha=appear, glow=0.05 + 1.2 * (1 - appear), light=light)
        except TypeError:
            tr.draw(ctx, int(min(i, tr.n - 1)), alpha=appear)
    ctx.restore()
    # glow halo on the glow canvas
    if appear > 0:
        cxw, cyw = tr.cloth_center(int(min(i, tr.n - 1))) if hasattr(tr, "cloth_center") else (80, -330)
        gctx.save()
        gctx.transform(M)
        gctx.scale(1 / SIM_K, 1 / SIM_K)
        fl = math.exp(-((T - T_MAT - 0.05) / 0.18) ** 2)
        r = 260 + 400 * fl
        g = cairo.RadialGradient(cxw, cyw, 0, cxw, cyw, r)
        g.add_color_stop_rgba(0, *col("#FFF1B8"), 0.12 * appear + 0.9 * fl)
        g.add_color_stop_rgba(1, *col("#FFC41A"), 0)
        gctx.set_source(g)
        gctx.arc(cxw, cyw, r, 0, 2 * math.pi)
        gctx.fill()
        gctx.restore()


def _draw_seal(ctx, gctx, cam, T, lights):
    C = Wd.SEAL_C
    o = seg(T, 88.3, 88.8, ease_in_out) if T > 88.3 else 0.0
    unlock = seg(T, T_SEAL, T_SEAL + 0.12)
    # pit (glowing) visible as petals slide away
    if o > 0:
        pit = ngon(5, Wd.SEAL_R, 0.2, math.pi, C[0], C[2])
        pts = poly_screen(cam, pit)
        if pts is not None:
            path_pts(ctx, pts)
            ctx.set_source_rgba(*col("#FFE2A0"), min(1, o * 1.5))
            ctx.fill()
    # 5 triangular petals (bronze) retracting radially + sinking
    V = ngon(5, Wd.SEAL_R, 0.0, math.pi, C[0], C[2])
    for k in range(5):
        a, b = V[k], V[(k + 1) % 5]
        m = (a + b) / 2
        d = m - C
        d[1] = 0
        d = d / (np.linalg.norm(d) + 1e-6)
        off = d * (Wd.SEAL_R * 0.95 * o) + np.array([0, -30 * o, 0])
        P = np.array([C + off, a + off, b + off]) + np.array([0, 1.0, 0])
        nrm = np.array([0, 1.0, 0])
        cs = Wd.shade(P, nrm, np.array(col("#6A4A26")) * (1.0 + 0.6 * unlock), lights, cam)
        pts = poly_screen(cam, P)
        if pts is None:
            continue
        path_pts(ctx, pts)
        ctx.set_source_rgb(*cs.mean(0))
        ctx.fill_preserve()
        ctx.set_source_rgba(*col("#E8B06A"), 0.7 + 0.3 * unlock)
        ctx.set_line_width(max(0.5, 3 * cam.f / max(float(cam.to_cam(C)[2]), 1)))
        ctx.stroke()
    # column of light
    if o > 0:
        FX.draw_beam(ctx, gctx, cam, C + np.array([0, 5200, 0]), C + np.array([0, 5, 0]), Wd.SEAL_R * 0.55,
                     min(0.75, o * 1.0), T)
    # strike shockwave ring travelling from the pole butt to the seal
    if T_STRIKE < T < T_STRIKE + 0.6:
        p = (T - T_STRIKE) / 0.6
        Pb = np.array([CROCUS["x"] - 40, 1.0, CROCUS["z"] - 10])
        rr = 60 + p * 2400
        ring = ngon(64, rr, 1.0, 0, Pb[0], Pb[2])
        pts = poly_screen(cam, ring)
        if pts is not None:
            for c_, w_, aa in ((ctx, 3.0, 0.8), (gctx, 12.0, 0.9)):
                path_pts(c_, pts)
                c_.set_source_rgba(*col("#FFC878"), aa * (1 - p) ** 1.5)
                c_.set_line_width(w_)
                c_.stroke()


def post(fc, st):
    T = fc.T
    d = {}
    d["fade"] = 1.0 - seg(T, 61.5, 61.85, ease_out)
    fl = seg(T, T_BRAAM, 90.0, lambda x: ease_in(x, 2.5))
    if fl > 0:
        d["exposure"] = 1.0 + 0.3 * fl
    if T > 89.78:
        d["fade"] = 0.7 * seg(T, 89.78, 90.0, ease_in)
        d["fade_color"] = C["flag_glow"]
    if T > 88.4 and T < 90:
        d["ca"] = 2.5 * seg(T, 88.4, 89.0)
    return d

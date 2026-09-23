"""s08 - The Sticky Flag (173.0-193.0). Comic acting with real timing, then the whole world salutes.

Shots (global s):
  A 173.00-176.95  from the white-gold glow, crane down over the morning meadow of flags into a two-shot;
                   Lutie runs up (L04 "Master! For you.") and offers her Flag with both hands.
  B 176.95-177.95  INSERT: Crocus's fingertip leaves the POLE - a glistening pale strand stretches (C05).
  C 177.95-183.90  two-shot: "It is sticky." (snap) / L05 exasperated / Crocus pinches it, C06 eyes closed,
                   serene, hands it back.
  D 183.90-185.00  closer on Lutie: the beat - she sniffs her own hand.
  E 185.00-190.00  wide (N11): they walk off into the field, camera rises; 188.30 `glorious`: everyone across
                   the meadow raises their flag at once (a rippling front); birds burst up.
  F 189.50-193.00  `end_card`: the IVG page slides down, grid builds, FLAGS drops in, scripture; fade to black.

World: scenes/s08_world.py (perspective meadow), acting: s08_acting.py, strand: s08_strand.py, card: s08_card.py.
"""
import hashlib
import math

import cairo
import cv2
import numpy as np

from vx import *
from vx import foley
from vx.flag import Flag, FlagTrack
from vx.chars import Character, draw_crowd
import vx.chars as _chars
from scenes import s08_world as wd
from scenes import s08_acting as act
from scenes import s08_strand as strand
from scenes import s08_card as card
from scenes.s01_title import paper_texture

_FALLBACK = {"touch": "point", "offer": "hold_flag", "gesture": "shout", "sniff": "talk", "shrug": "stand",
             "recoil": "stand", "shoulder_flag": "hold_flag", "wave": "cheer"}


def _P(pose):
    """map poses the installed rig does not know yet to their nearest cousin (rig v0 stub compatibility)"""
    known = getattr(_chars, "POSES", ())
    if isinstance(pose, dict):
        out = {}
        for k, w in pose.items():
            k2 = k if k in known else _FALLBACK.get(k, "stand")
            out[k2] = out.get(k2, 0.0) + w
        return out
    return pose if pose in known else _FALLBACK.get(pose, "stand")


POST = dict(bloom=0.42, bloom_thresh=0.74, grain=0.03, vignette=0.24)

T0 = 173.0
T_GLORIOUS = 188.30
T_CUT_B, T_CUT_C, T_CUT_D, T_CUT_E = 176.95, 177.95, 183.90, 185.00
U = act.U
LIGHT = (0.25, -0.55, -0.8)                   # to the light: low morning sun ahead of camera -> backlit cloth
RIM = ("#FFF3EA", 0.75, (0.25, -1.0))
WIDE_ZB = [(1100, 1e9), (350, 1100), (120, 350), (36, 120), (0.3, 36)]
TWO_ZB = [(350, 1e9), (60, 350), (0.3, 60)]
NEAR_RIG = 36.0                                # crowd closer than this (m) is drawn with the full rig


# =========================================================== cameras
def _cam(x, y, z, p, F=1500.0):
    return wd.PCam(x, y, z, p, F)


CAM_HIGH = _cam(0.45, 9.5, -19.0, 0.37)
CAM_MED = _cam(0.28, 1.35, -4.1, -0.061)
CAM_TWO = _cam(0.30, 1.30, -2.55, -0.098)
CAM_TWO2 = _cam(0.28, 1.30, -2.40, -0.100)
CAM_W0 = _cam(0.02, 1.45, -2.9, -0.02, 1300.0)
CAM_W1 = _cam(0.02, 5.8, -9.8, 0.17, 1300.0)
CAM_W2 = _cam(0.02, 8.8, -12.0, 0.22, 1300.0)


def shot(T):
    """-> (id, PCam, dof sigma in design px)"""
    if T < T_CUT_B:
        if T < 175.25:
            u = seg(T, T0, 175.25, lambda t: 1 - (1 - t) ** 2.6)
            cam = wd.lerp_cam(CAM_HIGH, CAM_MED, u)
        else:
            cam = wd.lerp_cam(CAM_MED, CAM_TWO, seg(T, 175.25, T_CUT_B, ease_in_out_sine))
        dof = 6.5 * smoothstep(174.5, 176.3, T)
        return "A", cam, dof
    if T < T_CUT_C:
        tx, ty = act._touch_pt(176.9)
        u = seg(T, T_CUT_B, T_CUT_C, ease_in_out_sine)
        cam = _cam(tx / U + 0.19 + 0.05 * u, -ty / U + 0.02 + 0.01 * u, -0.86 + 0.03 * u, 0.0)
        return "B", cam, 16.0
    if T < T_CUT_D:
        u = seg(T, T_CUT_C, T_CUT_D, ease_in_out_sine)
        return "C", wd.lerp_cam(CAM_TWO, CAM_TWO2, u), 7.0
    if T < T_CUT_E:
        u = seg(T, T_CUT_D, T_CUT_E, ease_in_out_sine)
        return "D", _cam(-0.08 + 0.02 * u, 1.16, -1.85 + 0.12 * u, -0.08), 8.0
    # one continuous rise (quadratic Bezier W0 -> W1 -> W2) - eases in, never stops under the page
    u = seg(T, T_CUT_E, 190.1, lambda t: t * t * (3 - 2 * t) * 0.55 + t * 0.45)
    a, b = wd.lerp_cam(CAM_W0, CAM_W1, u), wd.lerp_cam(CAM_W1, CAM_W2, u)
    return "E", wd.lerp_cam(a, b, u), 0.0


# =========================================================== walk-off (shot E), world metres
def walk_z(T, lane):
    """distance walked into the field (m) - they stop for the salute"""
    t = max(0.0, T - T_CUT_E)
    z = 0.4 + 1.05 * t - 0.12 * t * t / 3.3
    zs = 0.4 + 1.05 * 3.3 - 0.12 * 3.3
    if T > 188.2:
        z = zs + 0.25 * (1 - math.exp(-(T - 188.2) * 4))
    return z + (0.15 if lane == "C" else 0.0)


WALK_SPEED = {"L": 2.0, "C": 1.68}            # rig gait speeds giving ~1 m/s for both (no foot sliding)


def walk_phase(T, lane="L"):
    h = act.LUT_H if lane == "L" else act.CRO_H
    stride = 4 * 8.0 * WALK_SPEED[lane] ** 0.6 * h / 100.0          # stage units per gait cycle (vx.chars)
    return walk_z(T, lane) * U / stride


# =========================================================== setup
def _hash(*arrs):
    h = hashlib.sha1()
    for a in arrs:
        h.update(np.ascontiguousarray(a, np.float64).tobytes())
    return h.hexdigest()[:12]


def _sim(S, name, n, pole_fn, wind_fn, seed):
    poles = np.array([pole_fn(i) for i in range(n)])
    winds = np.array([wind_fn(i) for i in range(n)])
    ver = (ROOT / "vx/flag_sim.py").stat().st_mtime + (ROOT / "vx/flag.py").stat().st_mtime
    key = _hash(poles, winds, [seed, act.L_POLE, ver])
    path = S.cache / f"{name}_{key}.npz"
    if path.exists():
        return FlagTrack.load(path)
    for old in S.cache.glob(f"{name}_*.npz"):
        old.unlink()
    tr = Flag(pole=act.L_POLE, seed=seed).simulate(n, pole_fn, wind_fn)
    tr.save(path)
    return tr


def _breeze(T):
    b = 330 + 90 * math.sin(T * 0.7) + 60 * math.sin(T * 1.9 + 1)
    # gust as Lutie arrives (the cloth snaps out nicely during "For you"), and the salute gust
    b += 220 * pulse(T, 175.1, 0.5) + 160 * pulse(T, 180.9, 0.6) + 700 * pulse(T, 188.6, 0.55)
    # the wind drops while Crocus dangles it and hands it back: the Flag hangs limp (and off his face)
    b += 120 * smoothstep(179.3, 179.8, T) * (1 - smoothstep(180.8, 181.2, T))    # keeps it flying over her head
    lull = 0.8 * smoothstep(181.0, 181.9, T) * (1 - smoothstep(184.7, 185.0, T))
    return b * (1 - lull)


def lutie_pole_wide(ch, T):
    """pole in STAGE units relative to Lutie's feet at (0,0) for shot E (rig anchors: carry -> raise)"""
    pose = _P(_wide_pose_L(T))
    a = ch.anchors(0.0, 0.0, act.LUT_H, pose, T, facing=_WIDE_FACING, pole_len=act.L_POLE, **_wide_kw(T, "L"))
    if "pole" in a:
        return a["pole"]
    return (0.12 * U, -0.5 * U, 0.05)


_WIDE_FACING = 1
_WIDE_KW = dict(back=True)


def _wide_pose_L(T):
    r = seg(T, T_GLORIOUS - 0.12, T_GLORIOUS + 0.28, ease_out)
    w = 1 - seg(T, 188.0, 188.35)
    return {"walk": w, "hold_flag": 1.0 - r, "raise_flag": r, "stand": 1 - w}


def setup(S):
    tl = S.tl
    fps = S.fps
    lut, cro = Character("lutie", seed=1), Character("crocus", seed=2)
    cr = wd.make_crowd(seed=8)
    # --- hero flag, sim 1: the two-shot sequence (173.0 -> 185.0)
    nA = int(round((T_CUT_E - T0) * fps)) + 2

    def pole_A(i):
        return act.pole_state(T0 + i / fps)

    def wind_A(i):
        return (_breeze(T0 + i / fps), 0.0, 0.0)

    trA = _sim(S, "heroA", nA, pole_A, wind_A, seed=3)
    # --- hero flag, sim 2: the wide walk-off (185.0 -> end), relative to Lutie's feet
    fB = int(round((T_CUT_E - T0) * fps))
    nB = S.n - fB

    def pole_B(i):
        return lutie_pole_wide(lut, T_CUT_E + i / fps)

    def wind_B(i):
        return (_breeze(T_CUT_E + i / fps) * 1.1, 0.0, 0.0)

    trB = _sim(S, "heroB", nB, pole_B, wind_B, seed=3)

    rng = np.random.default_rng(4)
    flocks = (wd.make_flock(11, 9, (-70.0, 42.0, 230.0), (7.0, 0.6, 0.0), (9, 4, 12), T0, span=0.42) +
              wd.make_flock(12, 6, (-18.0, 14.0, 95.0), (3.2, 0.4, -0.5), (5, 2, 6), T0, span=0.36))
    burst = []
    for i in range(46):
        X = rng.uniform(-45, 45)
        Z = rng.uniform(25, 140)
        burst.append(dict(X=X, Y=rng.uniform(1.5, 2.5), Z=Z, vx=rng.normal(0, 2.5) + 1.5, vy=rng.uniform(5, 9),
                          vz=rng.uniform(-6, -1), phase=rng.uniform(0, 6.3), span=rng.uniform(0.3, 0.45),
                          rate=rng.uniform(4.0, 5.5), t0=T_GLORIOUS + 0.15 + (abs(X) + Z) * 0.004))
    tufts_near = wd.make_tufts(5, 90, (-3.4, 3.6), (-1.2, 1.4))
    tufts_path = wd.make_tufts(6, 260, (-9.0, 9.0), (-2.5, 16.0))
    st = dict(lut=lut, cro=cro, crowd=cr, trA=trA, trB=trB, fB=fB, flocks=flocks, burst=burst,
              tufts_near=tufts_near, tufts_path=tufts_path)
    _export_foley(S, st)
    return st


# =========================================================== foley
def _export_foley(S, st):
    ev = []
    # running footsteps (from the run cycle phase: one step per half cycle)
    last = None
    for f in range(0, int((act.RUN_T1 - T0) * S.fps) + 1):
        T = T0 + f / S.fps
        k = int(act.run_phase(T) * 2)
        if last is not None and k != last:
            ev.append(dict(t=T, kind="footstep", strength=0.35 + 0.35 * smoothstep(173.3, 174.6, T),
                           pan=-0.8 + 0.6 * smoothstep(173.0, 174.8, T), desc="Lutie running on meadow grass"))
        last = k
    ev += [
        dict(t=174.80, kind="skid", strength=0.5, pan=-0.25, desc="Lutie skids to a stop in the grass"),
        dict(t=175.05, kind="whoosh", strength=0.45, pan=-0.1, desc="she swings the Flag forward to offer it"),
        dict(t=strand.T_TOUCH, kind="touch", strength=0.3, pan=0.05, desc="Crocus's fingertip touches the pole (tiny tap)"),
        dict(t=strand.T_PULL, kind="squelch", strength=0.55, pan=0.05, desc="sticky: finger peels off the pole (INSERT)"),
        dict(t=177.45, kind="stretch", strength=0.4, pan=0.08, desc="the sticky strand stretches, thins, beads"),
        dict(t=strand.T_SNAP, kind="snap", strength=0.5, pan=0.12, desc="sticky strand snaps and recoils (wet tick)"),
        dict(t=179.40, kind="whoosh", strength=0.35, pan=-0.35, desc="Lutie swings the Flag up to her side like a staff ('But Master,')"),
        dict(t=180.50, kind="cloth", strength=0.25, pan=-0.15, desc="sleeve swish: her arm sweeps over 'all Flags'"),
        dict(t=180.95, kind="whoosh", strength=0.5, pan=-0.2, desc="she swings the Flag across and thrusts it at him ('one')"),
        dict(t=181.17, kind="thrust", strength=0.55, pan=-0.05, desc="the thrust lands in front of his face ('Flag!')"),
        dict(t=181.40, kind="grab", strength=0.35, pan=0.15, desc="Crocus pinches the pole between two fingers"),
        dict(t=183.05, kind="grab", strength=0.4, pan=-0.2, desc="Lutie takes the Flag back"),
        dict(t=184.22, kind="sniff", strength=0.45, pan=-0.3, desc="Lutie sniffs her own hand (sniff 1)"),
        dict(t=184.47, kind="sniff", strength=0.5, pan=-0.3, desc="sniff 2"),
    ]
    # walk-off footsteps (both, alternating, receding)
    for lane, pan in (("L", -0.12), ("C", 0.12)):
        last = None
        for f in range(int((T_CUT_E - T0) * S.fps), int((188.3 - T0) * S.fps)):
            T = T0 + f / S.fps
            k = int(walk_z(T, lane) / 0.62)
            if last is not None and k != last:
                ev.append(dict(t=T, kind="footstep", strength=0.25 * (1 - 0.5 * seg(T, 185, 188.3)), pan=pan,
                               desc=f"{'Lutie' if lane == 'L' else 'Crocus'} walking away through meadow grass"))
            last = k
    # the salute wave: grouped by distance rings, panned by azimuth
    cr = st["crowd"]
    rr, delay = wd.salute_raise(cr, T_GLORIOUS, T_GLORIOUS, origin=(0.0, 4.0))
    person = cr.person
    d = np.hypot(cr.X, cr.Z - 4.0)
    for lo, hi in ((0, 30), (30, 90), (90, 250), (250, 700), (700, 3000)):
        for az0 in (-1.0, -0.33, 0.33):
            m = person & (d >= lo) & (d < hi) & (np.clip(cr.X / np.maximum(cr.Z + 20, 1), -1, 1) >= az0) & \
                (np.clip(cr.X / np.maximum(cr.Z + 20, 1), -1, 1) < az0 + 0.67)
            if m.sum() < 3:
                continue
            ev.append(dict(t=float(T_GLORIOUS + np.median(delay[m])), kind="salute_wave",
                           strength=float(np.clip(1.2 - np.log10(lo + 10) * 0.35, 0.15, 1.0)),
                           pan=float(az0 + 0.335), desc=f"{int(m.sum())} people raise flags ({lo}-{hi} m away)"))
    ev.append(dict(t=T_GLORIOUS, kind="salute", strength=1.0, pan=0.0,
                   desc="GLORIOUS: Lutie + everyone across the meadow raise their flags at once (front ripples out ~0.9 s)"))
    ev.append(dict(t=T_GLORIOUS + 0.3, kind="gust", strength=0.8, pan=0.0, desc="a gust sweeps the field, ten thousand flags snap"))
    ev.append(dict(t=T_GLORIOUS + 0.2, kind="birds", strength=0.6, pan=0.0, desc="flock of birds bursts up from the meadow"))
    for t, kind, s, desc in card.hits():
        ev.append(dict(t=t, kind=kind, strength=s, pan=0.0, desc=desc))
    foley.export_events(S, "hits", ev)
    # hero flag tracks (physics-derived flutter), panned by where the cloth is on screen in each shot
    trA, trB = st["trA"], st["trB"]
    mA = int(round((T_CUT_E - T0) * S.fps))
    panA = np.zeros(mA, np.float32)
    for i in range(mA):
        T = T0 + i / S.fps
        _, cam, _ = shot(T)
        ox, oy, sc = cam.stage(0.0, 0.0, U)
        cx, cy = trA.cloth_center(i)
        panA[i] = foley.screen_pan(ox + cx * sc)
    mB = trB.n
    fadeB = np.array([1.0 - smoothstep(card.T_CARD - 0.4, card.T_CARD + 0.1, T_CUT_E + i / S.fps) for i in range(mB)],
                     np.float32)
    panB = np.zeros(mB, np.float32)
    for i in range(mB):
        T = T_CUT_E + i / S.fps
        _, cam, _ = shot(T)
        ox, oy, sc = cam.stage(0.0, walk_z(T, "L"), U)
        cx, cy = trB.cloth_center(i)
        panB[i] = foley.screen_pan(ox + (cx - 0.32 * U) * sc)

    def cut(tr, m, k=None):
        k = np.ones(m, np.float32) if k is None else k
        fl = getattr(tr, "flap_hz", None)
        return FlagTrack(tr.verts[:m], tr.poles[:m], tr.energy[:m] * k, tr.snap[:m] * k, tr.meta,
                         None if fl is None else fl[:m])

    fullA = np.full(S.n, panA[-1], np.float32)
    fullA[:mA] = panA
    fullB = np.full(S.n, panB[0], np.float32)
    fullB[st["fB"]:st["fB"] + mB] = panB[:S.n - st["fB"]]
    cut(trA, mA).export_foley(S, "hero_flag", pan=fullA, f0=0, gain=0.9)
    cut(trB, mB, fadeB).export_foley(S, "hero_flag_wide", pan=fullB, f0=st["fB"], gain=0.4)
    # crowd flags: a whole-field flutter bed that swells with the salute gust
    n = S.n
    en = np.array([0.35 + 0.1 * math.sin((T0 + f / S.fps) * 0.7) + 0.6 * pulse(T0 + f / S.fps, 188.7, 0.6)
                   for f in range(n)], np.float32)
    en[: int((T_CUT_E - T0) * S.fps)] *= 0.55
    en *= np.array([1 - smoothstep(card.T_CARD - 0.4, card.T_CARD + 0.1, T0 + f / S.fps) for f in range(n)], np.float32)
    foley.export_track(S, "crowd_flags", en, 0.0, gain=0.6, kind="flag_field")


# =========================================================== drawing helpers
def _mouth(tl, who, T):
    try:
        return tl.mouth(who, T)
    except Exception:
        return 0.0


def _ik(ch, x, h, pose, T, facing, perf, kw):
    """blend rig's natural hand positions with the performance IK targets"""
    out = {}
    nat = None
    for hand in ("hand_r", "hand_l"):
        w = perf.get(hand + "_w", 0.0) or 0.0
        tgt = perf.get(hand)
        if w <= 0.001 or tgt is None:
            continue
        if w < 0.999:
            if nat is None:
                nat = ch.anchors(x, 0.0, h, pose, T, facing=facing, pole_len=None, **kw)
            nx, ny = nat[hand]
            out[hand] = (nx + (tgt[0] - nx) * w, ny + (tgt[1] - ny) * w)
        else:
            out[hand] = tgt
    return out


def _char_kw(perf, speaker_mouth):
    kw = dict(lean=perf.get("lean", 0.0), nod=perf.get("nod", 0.0), tilt=perf.get("tilt", 0.0),
              shape_r=perf.get("shape_r", "relax"), shape_l=perf.get("shape_l", "relax"), rim=RIM, wind=0.35)
    if "turn" in perf:
        kw["turn"] = perf["turn"]
    return kw


def draw_heroes_stage(ctx, st, fc, T):
    """two-shot sequence (A-D): stage units at Z=0; returns fingertip for the strand"""
    tl = fc.tl
    lut, cro = st["lut"], st["cro"]
    pl = act.sample(act.LUTIE, T)
    pc = act.sample(act.CROCUS, T)
    pl["pose"], pc["pose"] = _P(pl["pose"]), _P(pc["pose"])
    xl, xc = pl["x"], pc["x"]
    kwl = _char_kw(pl, None)
    kwc = _char_kw(pc, None)
    if T < act.RUN_T1 + 0.2:
        kwl["phase"] = act.run_phase(T)
        kwl["speed"] = act.RUN_SPEED
    ikl = _ik(lut, xl, act.LUT_H, pl["pose"], T, 1, pl, kwl)
    ikc = _ik(cro, xc, act.CRO_H, pc["pose"], T, -1, pc, kwc)
    # Crocus's fingertip: correct the palm target so the rig's actual index tip lands on the pole
    wf = seg(T, 176.0, 176.3) * (1 - seg(T, 179.35, 179.8))
    if wf > 0 and "hand_r" in ikc:
        probe = cro.anchors(xc, 0.0, act.CRO_H, pc["pose"], T, facing=-1, **kwc, **ikc)
        if "tip_r" in probe:
            want = act.fingertip_target(T)
            tx, ty = probe["tip_r"]
            hx, hy = ikc["hand_r"]
            ikc["hand_r"] = (hx + (want[0] - tx) * wf, hy + (want[1] - ty) * wf)
    ml, mc = _mouth(tl, "LUTIE", T), _mouth(tl, "CROCUS", T)
    common_l = dict(mouth=ml, look=pl["look"], expr=pl["expr"], facing=1, blink=pl["blink"], **kwl, **ikl)
    common_c = dict(mouth=mc, look=pc["look"], expr=pc["expr"], facing=-1, blink=pc["blink"], **kwc, **ikc)
    cro.draw(ctx, xc, 0.0, act.CRO_H, pc["pose"], T, layer="back", **common_c)
    lut.draw(ctx, xl, 0.0, act.LUT_H, pl["pose"], T, layer="back", **common_l)
    i = (T - T0) * fc.fps
    st["trA"].draw(ctx, i, light=LIGHT, glow=0.22)
    ac = cro.draw(ctx, xc, 0.0, act.CRO_H, pc["pose"], T, layer="front", **common_c)
    lut.draw(ctx, xl, 0.0, act.LUT_H, pl["pose"], T, layer="front", **common_l)
    return ac


def _stage_transform(ctx, cam, X, Z):
    ox, oy, sc = cam.stage(X, Z, U)
    ctx.translate(ox, oy)
    ctx.scale(sc, sc)
    return sc


def draw_strand(ctx, st, T, sc, ac):
    if T < strand.T_TOUCH - 0.05 or T > T_CUT_E:
        return
    B = act._touch_pt(176.9)
    # contact point rides ON the pole: keep its distance along the pole axis, sit on the surface
    b0x, b0y, a0 = act.pole_state(176.9)
    s = (B[0] - b0x) * math.sin(a0) - (B[1] - b0y) * math.cos(a0)
    bx, by, a = act.pole_state(T)
    Bp = (bx + math.sin(a) * s + math.cos(a) * act.POLE_HALF, by - math.cos(a) * s + math.sin(a) * act.POLE_HALF)
    if T > 179.6:
        # the glistening dab left on the pole (the evidence)
        strand._blob(ctx, Bp[0], Bp[1] + 0.6, 1.8, 1.0, sc)
        return
    tip = ac.get("tip_r") if ac is not None else None
    if tip is None:
        tip = act.fingertip_target(T)
    strand.draw(ctx, T, tip, Bp, sc)


def _wide_kw(T, lane):
    return dict(phase=walk_phase(T, lane), speed=WALK_SPEED[lane], rim=RIM, wind=0.6, **_WIDE_KW)


def draw_wide_heroes(ctx, st, fc, cam, T):
    lut, cro = st["lut"], st["cro"]
    zl, zc = walk_z(T, "L"), walk_z(T, "C")
    xl, xc = -0.32, 0.36
    items = sorted([(zl, "L"), (zc, "C")], reverse=True)
    for z, who in items:
        ctx.save()
        _stage_transform(ctx, cam, 0.0, z)
        if who == "L":
            pose = _P(_wide_pose_L(T))
            x = xl * U
            kw = _wide_kw(T, "L")
            lut.draw(ctx, x, 0.0, act.LUT_H, pose, T, facing=_WIDE_FACING, layer="back", pole_len=act.L_POLE,
                     expr="happy", **kw)
            ctx.save()
            ctx.translate(x, 0.0)
            st["trB"].draw(ctx, (T - T_CUT_E) * fc.fps, light=LIGHT, glow=0.25)
            ctx.restore()
            lut.draw(ctx, x, 0.0, act.LUT_H, pose, T, facing=_WIDE_FACING, layer="front", pole_len=act.L_POLE,
                     expr="happy", **kw)
        else:
            # Crocus has no flag: he salutes with the sticky finger, raised high
            r = seg(T, T_GLORIOUS - 0.02, T_GLORIOUS + 0.3, ease_out)
            w = 1 - seg(T, 188.0, 188.35)
            pose = _P({"walk": w, "stand": 1 - w + 1e-3})
            kw = _wide_kw(T, "C")
            if r > 0.001:
                nat = cro.anchors(xc * U, 0.0, act.CRO_H, pose, T, facing=_WIDE_FACING, **kw)["hand_r"]
                up = (xc * U + 0.06 * U, -2.12 * U)
                kw["hand_r"] = (nat[0] + (up[0] - nat[0]) * r, nat[1] + (up[1] - nat[1]) * r)
            cro.draw(ctx, xc * U, 0.0, act.CRO_H, pose, T, facing=_WIDE_FACING, expr="serene", blink=0.3,
                     shape_r="point" if r > 0.3 else "relax", **kw)
        ctx.restore()


# =========================================================== render
def _background(fc, st, cam, T, dof):
    """sky, hills, crowd, mist, birds -> (h, w, 3) float at full res (blurred when dof > 0)"""
    lo = 0.5 if dof > 2.5 else 1.0
    fl = wd.lowres_fc(fc, lo) if lo != 1.0 else fc
    img = wd.environment(fl, cam, T, lowres=4 if dof > 0 else 3)
    cv = Canvas(w=fl.w, h=fl.h, s=fl.s)
    sx, sy = wd.sun_screen(cam)
    if -60 < sy < H + 60:
        # the sun disc, veiled by morning haze (behind the hills and everyone)
        g = rad_grad(sx, sy, 0, 34, [(0, "#FFFFFF", 0.97), (0.5, "#FFFDF8", 0.92), (0.78, "#FFF3EB", 0.35),
                                     (1, "#FFF3EB", 0.0)])
        cv.ctx.arc(sx, sy, 34, 0, 2 * math.pi)
        cv.ctx.set_source(g)
        cv.ctx.fill()
    wd.draw_hills(cv.ctx, cam, T)
    img = cv.over(img)
    cr = st["crowd"]
    rr, delay = wd.salute_raise(cr, T, T_GLORIOUS, origin=(0.0, 4.0))
    # a glint rides the salute front: each cloth flashes as it is thrown up into the sun
    bright = 0.6 * np.exp(-((T - T_GLORIOUS - delay - 0.14) / 0.16) ** 2) if T > T_GLORIOUS - 0.2 else None
    gust = 0.55 + 0.1 * math.sin(T * 0.8)
    if T > 188.0:
        tg = T - T_GLORIOUS - 0.12 - wd.salute_delay(np.hypot(cr.X, cr.Z - 4.0))
        gust = gust + 0.75 * np.clip(tg * 3.0, 0, 1) * np.exp(-np.maximum(tg, 0) * 0.9)

    def near_people(ctx, sel, g, wk, br):
        """the closest meadow folk with the full character rig (flags from their hands' pole anchors)"""
        bx, by, L, ang = g["bx"].copy(), g["by"].copy(), g["L"].copy(), g["ang"].copy()
        people, idx = [], []
        for j, i in enumerate(sel):
            if not cr.person[i]:
                continue
            r = float(np.clip(rr[i], 0.0, 1.0))
            fa = 0 if cr.seed[i] % 10 < 3 else (1 if cr.X[i] < 0 else -1)
            people.append(dict(x=float(g["sx"][j]), y=float(g["sy"][j]), h=float(cr.hgt[i] * g["k"][j]),
                               kind="acolyte" if (cr.hood[i] and cr.seed[i] % 3 == 0) else "citizen", seed=int(cr.seed[i] % 997),
                               pose=_P({"hold_flag": 1.0 - r + 1e-3, "raise_flag": r + 1e-3}), facing=fa,
                               expr="happy" if r > 0.3 else "serene", t_off=float(cr.phase[i])))
            idx.append(j)
        an = draw_crowd(ctx, people, T) or []
        for j, p, a in zip(idx, people, an):
            if isinstance(a, dict) and "pole" in a:
                bx[j], by[j], ang[j] = a["pole"]
                L[j] = p["h"] * 1.35
            lift = max(0.0, float(rr[sel[j]]) - 1.0) * p["h"] * 0.35     # spring overshoot of the raise
            by[j] -= lift
        wd.draw_field_flags(ctx, bx, by, L, ang, T, cr.seed[sel], wk, light=LIGHT,
                            glow=0.1 if br is None else 0.1 + 0.5 * br, bright=br)

    img = wd.draw_crowd_bands(fl, cam, cr, T, rr, gust, WIDE_ZB if dof == 0 else TWO_ZB, img, zmin=0.3,
                              light=LIGHT, near_people=near_people if dof == 0 else None, near_max=NEAR_RIG,
                              bright=bright)
    cv = Canvas(w=fl.w, h=fl.h, s=fl.s)
    wd.draw_mist_bands(cv.ctx, cam, T)
    wd.draw_birds(cv.ctx, cam, T, st["flocks"])
    if T > T_GLORIOUS:
        wd.draw_birds(cv.ctx, cam, T, [b for b in st["burst"] if T > b["t0"]])
    img = cv.over(img)
    if lo != 1.0:
        img = cv2.resize(img, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)
    if dof > 0:
        img = blur(img, dof * fc.s)
    return img


def _sun_glow(fc, cam, img, k=1.0):
    sx, sy = wd.sun_screen(cam)
    cv = Canvas(w=fc.w, h=fc.h, s=fc.s)
    ctx = cv.ctx
    g = rad_grad(sx, sy, 0, 900, [(0, "#FFF6EE", 0.28 * k), (0.25, "#FBEAE2", 0.10 * k), (1, "#FBEAE2", 0.0)])
    ctx.rectangle(0, 0, W, H)
    ctx.set_source(g)
    ctx.fill()
    img = screen(img, cv.rgba()[..., :3])
    return img


def _god_rays(fc, cam, img, k):
    """crepuscular rays: the bright sky/cloth radially smeared from the sun (occluders cut shafts)"""
    if k <= 0.01:
        return img
    sx, sy = wd.sun_screen(cam)
    q = 0.5
    h, w = int(fc.h * q), int(fc.w * q)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    L = small.mean(axis=2, keepdims=True)
    src = np.clip((L - 0.9) / 0.1, 0, 1) ** 2 * small
    cx, cy = sx * fc.s * q, sy * fc.s * q
    acc = np.zeros_like(src)
    n = 14
    for i in range(n):
        sc = 1.0 - 0.035 * i
        M = np.array([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]], np.float32)
        acc += cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * (1 - i / n)
    acc *= 1.0 / (n * 0.5)
    rays = cv2.resize(acc, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR)
    return img + rays * np.array([1.0, 0.96, 0.9], np.float32) * k


def _ray_strength(sid, T):
    if sid == "A":
        return 0.22
    if sid == "E":
        return 0.2 + 0.3 * smoothstep(T_GLORIOUS - 0.1, T_GLORIOUS + 0.5, T) * (1 - 0.5 * smoothstep(189.0, 189.5, T))
    return 0.08


def render(fc, st):
    T = fc.T
    # full page -> just the card
    if T >= card.T_CARD:
        return _card_frame(fc, st, T, None)
    sid, cam, dof = shot(T)
    img = _background(fc, st, cam, T, dof)
    cv = Canvas(w=fc.w, h=fc.h, s=fc.s)
    ctx = cv.ctx
    if sid in "ABCD":
        # grass right at the heroes' feet line (behind them) - only seen in the wider framings
        if sid == "A":
            wd.draw_tufts(ctx, cam, T, [t for t in st["tufts_near"] if t["Z"] > 0.25], wind=0.8)
        ctx.save()
        sc = _stage_transform(ctx, cam, 0.0, 0.0)
        ac = draw_heroes_stage(ctx, st, fc, T)
        draw_strand(ctx, st, T, sc * fc.s, ac)
        ctx.restore()
        if sid == "A":
            wd.draw_tufts(ctx, cam, T, [t for t in st["tufts_near"] if t["Z"] <= 0.25], wind=0.8)
    else:
        wd.draw_tufts(ctx, cam, T, st["tufts_path"], wind=0.9)
        draw_wide_heroes(ctx, st, fc, cam, T)
    img = cv.over(img)
    img = _sun_glow(fc, cam, img, 1.0 if sid != "B" else 0.6)
    img = _god_rays(fc, cam, img, _ray_strength(sid, T))
    if T >= card.T_CARD - card.T_PAGE_IN:
        img = _card_frame(fc, st, T, img)
    return img


_PAPER = {}


def _paper(s):
    if s not in _PAPER:
        arr = paper_texture(s)
        surf = surface_from_array(arr)
        _PAPER[s] = (cairo.SurfacePattern(surf), surf)
    return _PAPER[s][0]


def _card_frame(fc, st, T, under):
    off = card.page_offset(T)
    cv = Canvas(w=fc.w, h=fc.h, s=fc.s)
    ctx = cv.ctx
    if under is not None and off < -0.5:
        # soft shadow under the page's lower edge falling on the meadow
        y = H + off
        g = lin_grad(0, y, 0, y + 70, [(0, "#1a1a22", 0.35), (1, "#1a1a22", 0.0)])
        ctx.rectangle(0, y, W, 70)
        ctx.set_source(g)
        ctx.fill()
    ctx.save()
    ctx.translate(0, off)
    pat = _paper(fc.s)
    m = cairo.Matrix()
    m.scale(fc.s, fc.s)
    pat.set_matrix(m)
    card.draw_page(ctx, T, paper=pat)
    ctx.restore()
    if under is None:
        return cv.rgb()
    return cv.over(under)


def post(fc, st):
    T = fc.T
    p = {}
    fin = 1 - ease_out(seg(T, T0, 174.05, lambda t: t), 1.6)
    if fin > 0:
        p.update(fade=fin, fade_color="flag_glow")
        p["vignette"] = POST["vignette"] * (1 - fin)
        p["bloom"] = POST["bloom"] * (1 - fin) ** 2
    if T >= card.T_CARD:
        p["vignette"] = 0.12
        p["bloom"] = 0.12
    out = seg(T, card.T_FADE0, card.T_FADE1 - 0.03, lambda t: ease_in_out(t, 2))
    if out > 0:
        p.update(fade=out, fade_color=(0, 0, 0))
    return p

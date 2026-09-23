"""s05a - The Schismmancers (90.0-100.5). A tactical-breach sequence whose breaching tools are plain yellow flags.

Shots (global seconds; cut frames derived from the locked timeline in setup):
  90.00 gear_up montage: velcro -> tacks -> telescoping poles -> NVG flip -> boot stomp   (s05a_gear)
  91.21 K01 wide: slow-motion hero walk down the pentagonal basalt tunnel, backlit gold     (s05a_tunnel)
  93.29 K01 MCU: the Commander on the radio (lip-sync), furled flag on his shoulder        (s05a_tunnel)
  96.08 K02 holographic HUD globe, a region balkanizing recursively (spherical Voronoi)     (s05a_hud)
  98.42 K03 'Breach!' x3: colossal door wide / the stack squeeze / flag butt on the seam   (s05a_door)
  99.71 BREACH: the door blows in, flag-yellow light, smoke (fluid sim), slow-mo leap with
        flags unfurling (cloth sims) -> full-frame flag_glow by 100.46                     (s05a_door)
"""
import math

import numpy as np

from vx import *
from vx import foley
from scenes import s05a_gear as gear
from scenes import s05a_tunnel as tunnel
from scenes import s05a_hud as hudm
from scenes import s05a_door as door

POST = dict(bloom=0.5, bloom_thresh=0.72, bloom_radius=24.0, grain=0.035, vignette=0.3, letterbox=0.16)

FPS = 24


def _lf(S, T):
    return int(round((T - S.start) * S.fps))


def setup(S):
    tl = S.tl
    b = [_lf(S, tl.word("K03", "Breach", k)[0]) for k in range(3)]
    boom = _lf(S, tl.cue("breach"))
    cuts = [("velcro", 0), ("tacks", 6), ("poles", 13), ("nvg", 19), ("boots", 25),
            ("walk", _lf(S, tl.line("K01")["start"])), ("mcu", _lf(S, tl.word("K01", "Target")[0])),
            ("hud", _lf(S, tl.line("K02")["start"])), ("beat1", b[0]), ("beat2", b[1]), ("beat3", b[2]),
            ("breach", boom)]
    shots = []
    for k, (name, f0) in enumerate(cuts):
        f1 = cuts[k + 1][1] if k + 1 < len(cuts) else S.n
        shots.append(dict(name=name, f0=f0, f1=f1, T0=S.start + f0 / S.fps, T1=S.start + f1 / S.fps))
    st = dict(shots=shots, boom_f=boom, shot={s["name"]: s for s in shots})
    tunnel.setup(S, st)
    hudm.setup(S, st)
    door.setup(S, st)
    _export_hits(S, st)
    return st


def shot_at(st, f):
    for sh in st["shots"]:
        if sh["f0"] <= f < sh["f1"]:
            return sh
    return st["shots"][-1]


def _mblur(fn, fc, st, u, n=5, shutter=0.5 / FPS):
    acc = None
    for i in range(n):
        img = fn(fc, st, u + (i / (n - 1) - 0.5) * shutter)
        acc = img if acc is None else acc + img
    return acc * (1.0 / n)


def render(fc, st):
    sh = shot_at(st, fc.f)
    u = (fc.f - sh["f0"]) / fc.fps
    name = sh["name"]
    if name in gear.SHOTS:
        img = _mblur(gear.SHOTS[name], fc, st, u)
    elif name in tunnel.SHOTS:
        img = tunnel.SHOTS[name](fc, st, u)
    elif name in hudm.SHOTS:
        img = hudm.SHOTS[name](fc, st, u)
    elif name in door.SHOTS:
        img = door.SHOTS[name](fc, st, u)
    else:
        img = np.zeros((fc.h, fc.w, 3), np.float32)
    return np.clip(img, 0, 4).astype(np.float32)


def post(fc, st):
    """cinemascope bars for the whole action sequence; the breach light blows them open; the last frame is a
    clean, flat flag_glow for the hand-off (no bloom/vignette/letterbox)."""
    boom = st["shot"]["breach"]["T0"]
    tb = fc.T - boom
    p = {}
    if tb >= 0:
        p["letterbox"] = 0.16 * (1 - ease_out(clamp(tb / 0.22), 2.5))
        p["ca"] = 2.2 * math.exp(-tb * 5)
        end = smoothstep(0.56, 0.75, tb)
        p["vignette"] = 0.3 * (1 - end)
        p["bloom"] = 0.36 * (1 - end)
        p["grain"] = 0.035 * (1 - end) + 0.008 * end
        if fc.f >= fc.n - 1:
            p.update(vignette=0.0, bloom=0.0, grain=0.006, ca=0.0, letterbox=0.0)
    elif st["shot"]["velcro"]["f0"] <= fc.f < st["shot"]["walk"]["f0"]:
        p["ca"] = 0.8
    return p


# ------------------------------------------------------------------------------------------ sound
def _export_hits(S, st):
    """every visual hit of the sequence (global seconds) + the five unfurling hero flags' physics tracks"""
    fr = 1.0 / FPS
    sh = st["shot"]
    T = lambda name, df=0: sh[name]["T0"] + df * fr
    ev = [dict(t=T("velcro"), kind="velcro_rip", strength=0.95, pan=0.1, desc="macro: hook strap peels off the loop panel (rip ~0.25s)"),
          dict(t=T("tacks", 1), kind="hammer", strength=0.9, pan=-0.1, desc="hammer drives carpet tack through the hoist into the pole (tap 1), cloth ripples"),
          dict(t=T("tacks", 4), kind="hammer", strength=1.0, pan=-0.1, desc="hammer tap 2, tack flush"),
          dict(t=T("poles", 0), kind="click", strength=0.9, pan=-0.3, desc="telescoping pole section shoots out & locks (click 1)"),
          dict(t=T("poles", 2), kind="click", strength=0.9, pan=0.0, desc="telescoping pole click 2"),
          dict(t=T("poles", 4), kind="click", strength=1.0, pan=0.3, desc="telescoping pole click 3"),
          dict(t=T("nvg", 0), kind="whoosh", strength=0.4, pan=0.0, desc="quad-tube NVG swings down on its hinge"),
          dict(t=T("nvg", 2), kind="click", strength=0.9, pan=0.0, desc="NVG mount locks down (mechanical snap)"),
          dict(t=T("nvg", 2) + 0.01, kind="power_on", strength=0.6, pan=0.0, desc="NVG tubes power on: high whine + green glow (flicker)"),
          dict(t=T("boots", 1), kind="stomp", strength=1.0, pan=-0.45, desc="boot stomps onto wet stone, backlit dust + splash")]
    # slow-motion footsteps of the five walkers (heavy, reverberant tunnel)
    for t, kind, i, dx in tunnel.footfalls(st["walk_T0"], sh["hud"]["T0"], 4.9):
        ev.append(dict(t=t, kind="footstep_slowmo", strength=0.8 if kind == "commander" else 0.5,
                       pan=float(np.clip(dx / 3.0, -1, 1)), desc=f"slow-motion boot fall ({kind}) in the basalt tunnel"))
    ev.append(dict(t=T("hud"), kind="holo_on", strength=0.7, pan=0.3, desc="wrist projector throws the hologram globe (scan build)"))
    ev.append(dict(t=st["hud_t"]["method"], kind="lock_on", strength=0.6, pan=0.3, desc="HUD brackets lock onto the first memeplex region"))
    for reg in st["hud_regions"]:
        for d in range(1, hudm.LEVELS + 1):
            ev.append(dict(t=reg["t0"] + (d - 1) * reg["dt"], kind="blip", strength=0.35 + 0.08 * d, pan=0.35,
                           desc=f"globe region balkanizes: recursion depth {d}"))
    for k, name in enumerate(("beat1", "beat2", "beat3")):
        ev.append(dict(t=st["beats"][k], kind="breach_beat", strength=0.8 + 0.1 * k, pan=-0.2,
                       desc=f"'Breach!' {k + 1}: flag butt slams the steel door, flag-yellow shock ring + cracks"))
    ev.append(dict(t=st["beats"][2] + 0.05, kind="crack_spread", strength=0.8, pan=0.0,
                   desc="cracks race across the door and blaze (rising, until the blow at 99.7)"))
    ev.append(dict(t=st["boom_T"], kind="explosion", strength=1.0, pan=0.0,
                   desc="BREACH: door blows in, blinding flag light, smoke; speed ramp into slow motion"))
    for c in st["cracks"]["cells"]:
        if c["r"] < 3.0:
            ev.append(dict(t=st["boom_T"] + c["t_rel"] / 0.3 + 0.12, kind="debris", strength=0.3, pan=float(np.clip(-c["c"][0] / 5, -1, 1)),
                           desc="door shard tumbling past camera (slow motion)"))
    st["hits"] = ev
    foley.export_events(S, "hits", ev)
    for i, tr in enumerate(st["leap_flags"]):
        tr.export_foley(S, f"leap_flag{i}", f0=st["boom_f"], gain=0.8 if i != 3 else 1.0)

"""t05 — AND IT WORKED (105.67 -> 122.29).  Owner: T05Worked.

A  105.667  the white card: BUT THERE WERE SOME THAT LISTENED... (P13), the engraving prints back in
B  108.44   'Found one!' (H01): a hippie pulls a clean plain Flag out of the mud (muddy pole, pristine cloth)
C  110.05   'Move it! MOVE IT!' (H02): the engraved plan of the Burn — Flags carried to the five corners of
            the fence, the ley-line pentagram strikes, the circle of cloud opens; 'AND IT WORKED.' (P14)
D  113.10   the masked Seraphlag carries the ALL YELLOW FLAG up the effigy (P15); 116.90 flag_on_effigy
E  118.917  rain_freeze: every raindrop stops in mid-air; bullet-time drift through them (P16)
F  119.555  firmament_reveal: the drops fall away, the clouds roll back — the FIRMAMENT rests on the pole
            tip. Ends on effigy.HANDOFF (t06 starts on the same composition, dusk).
"""
from vx import *
from vx import ink, comic, foley
from vx.flag import FlagTrack

from scenes import effigy as E
from scenes import t05_card as A
from scenes import t05_found as B
from scenes import t05_map as MP
from scenes import t05_world as WD

POST = dict(bloom=0.12, bloom_thresh=0.82, grain=0.0, vignette=0.14)

T_B = A.T_PRINT0
T_C = MP.T0
T_D = WD.T_D0


def setup(S):
    st = {}
    st["found_flag"] = _cached_track(S, "flag_found", B.simulate_flag, B.path_key())
    st["hero"] = _cached_track(S, "flag_hero", WD.simulate_flag, WD.path_key())
    st["map"] = MP.MapPlan()
    st["rain"] = WD.Rain()
    st["people"] = E.crowd()
    _export_sound(S, st)
    return st


def _cached_track(S, name, sim, key):
    """flag sims are cached under a hash of their pole path + wind (re-simulated whenever the motion changes)"""
    import hashlib
    h = hashlib.sha1(np.asarray(key, np.float64).tobytes()).hexdigest()[:12]
    p = S.cache / f"{name}.npz"
    try:
        tr = FlagTrack.load(p)
        if tr.meta.get("key") == h:
            return tr
    except Exception:
        pass
    tr = sim()
    tr.meta["key"] = h
    tr.save(p)
    return tr


def _export_sound(S, st):
    ev = [dict(t=A.T_CUT, kind="cut", strength=1.0, pan=0.0,
               desc="hard cut from the FLAGS wall to the silent white card: everything stops"),
          dict(t=106.43, kind="paper", strength=0.35, pan=0.0, desc="title card: heavy caps inked word by word"),
          dict(t=A.T_PRINT0, kind="whoosh", strength=0.3, pan=0.0, desc="the engraving prints back in; rain returns"),
          dict(t=B.T_PULL, kind="pull", strength=1.0, pan=B.PAN, desc="a Flag yanked out of the mud: SHLUPP, mud clods"),
          dict(t=B.T_PULL + 0.32, kind="flag", strength=0.7, pan=B.PAN, desc="the furled cloth snaps open, clean"),
          dict(t=MP.T0, kind="glide", strength=0.4, pan=0.0, desc="cut to the engraved map of the Burn, rain on paper")]
    for c in st["map"].carriers:
        x, _ = MP.map_to_screen(c["t_arr"], *c["p1"])
        ev.append(dict(t=c["t_arr"], kind="thunk", strength=0.5, pan=foley.screen_pan(x),
                       desc="a Flag planted at a corner of the trash fence (map)"))
    ev += [dict(t=MP.T_LEY0, kind="shimmer", strength=0.6, pan=0.0, desc="ley lines strike between the Flags"),
           dict(t=MP.T_LEY1, kind="chime", strength=0.8, pan=0.0, desc="the pentagram closes into a circle"),
           dict(t=MP.T_OPEN, kind="whoosh", strength=0.8, pan=0.0, desc="the circle of cloud opens over the effigy"),
           dict(t=WD.T_D0, kind="cut", strength=0.5, pan=0.3, desc="cut to the effigy in the rain, the Seraphlag"),
           dict(t=WD.T_SEAT, kind="thunk", strength=1.0, pan=0.0,
                desc="flag_on_effigy: the pole seats in the effigy's head socket (big wooden THUNK)"),
           dict(t=WD.T_SEAT + 0.35, kind="whoosh", strength=0.7, pan=0.2, desc="the Seraphlag lets go and soars away"),
           dict(t=WD.T_FREEZE, kind="freeze", strength=1.0, pan=0.0,
                desc="rain_freeze: every raindrop stops dead in mid-air: the rain SOUND cuts to silence"),
           dict(t=WD.T_REVEAL, kind="release", strength=0.9, pan=0.0,
                desc="the frozen drops fall away all at once (patter), then silence"),
           dict(t=WD.T_REVEAL + 0.1, kind="reveal", strength=1.0, pan=0.0,
                desc="firmament_reveal: the clouds roll back, the crystal sky-dome rings out (glassy swell)"),
           dict(t=WD.T_REVEAL + 0.6, kind="crowd", strength=0.8, pan=0.0, desc="crowd awe: gasps, then cheers")]
    for t in WD.wing_beats(WD.T_D0, 118.4):
        p, _ = WD.pole_world(t)
        ev.append(dict(t=t, kind="wing", strength=0.8, pan=0.3, desc="the Seraphlag's great downstroke"))
    foley.export_events(S, "hits", ev)
    f0 = int(round((WD.T_D0 - 0.25 - S.start) * S.fps))
    st["hero"].export_foley(S, "hero_flag", f0=f0, gain=0.8)
    f0b = int(round((B.T0_SIM - S.start) * S.fps))
    st["found_flag"].export_foley(S, "found_flag", f0=f0b, gain=0.8)


def render(fc, st):
    T = fc.T
    if T < T_B:
        img = A.render(fc, T)
        top = Canvas(fc)
        A.title(top.ctx, fc, T)
        return top.over(img)
    if T < T_C:
        img, top = B.render(fc, T, st, print_k=A.print_k(T))
        A.title(top.ctx, fc, T)
        return top.over(img)
    if T < T_D:
        return MP.render(fc, T, st)
    return WD.render(fc, T, st)


def post(fc, st):
    return WD.post(fc.T) if fc.T >= T_D else {}

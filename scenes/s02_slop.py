"""s02 - The Age of Slop (21.0-47.5).  Owner: Scene02.

Shots (global seconds):
  A 21.00-34.80  the plain of memeplexes: fog lifts from flat concrete_l (s01 hand-off), crane up, NATION /
                 FAITH / PHILOSOPHY light up on their N04 words, broadcast rings + lockstep on N05, and at
                 33.6 (N06) a thin beige line of slop rises on the horizon.            -> scenes/s02_plain.py
  B 34.80-40.60  THE SLOP WAVE: a Hokusai curl whose body is a flowing field of content cards spiralling into
                 an infinite scroll; 39.2 slop_crash; memeplexes crack/topple/sink.    -> scenes/s02_wave.py
  C 40.60-47.50  waist-deep isolated people lit by their feeds; the frame splits 1->4->16->64->256;
                 47.0 push into one screen -> full-frame static (s03 hand-off).        -> scenes/s02_feeds.py
No flags appear in this scene; nothing here is yellow.
"""
from vx import *
from scenes import s02_plain as PLAIN

T_WAVE = 34.8
T_FEED = 40.6

POST = dict(bloom=0.4, grain=0.03, vignette=0.25)


def setup(S):
    st = {}
    PLAIN.setup(S, st)
    try:
        from scenes import s02_wave as WAVE
        WAVE.setup(S, st)
    except ImportError:
        pass
    try:
        from scenes import s02_feeds as FEEDS
        FEEDS.setup(S, st)
    except ImportError:
        pass
    export_hits(S, st)
    return st


def render(fc, st):
    T = fc.T
    if T < T_WAVE:
        return PLAIN.render(fc, st)
    if T < T_FEED:
        from scenes import s02_wave as WAVE
        return WAVE.render(fc, st)
    from scenes import s02_feeds as FEEDS
    return FEEDS.render(fc, st)


def post(fc, st):
    T = fc.T
    if T < T_WAVE:
        p = dict(bloom=0.4 + 0.25 * max(0.0, PLAIN.lit_curve(T, PLAIN.T_N) - 1.0)
                 + 0.25 * max(0.0, PLAIN.lit_curve(T, PLAIN.T_F) - 1.0)
                 + 0.25 * max(0.0, PLAIN.lit_curve(T, PLAIN.T_P) - 1.0))
        if T < 21.1:
            p["bloom"] = 0.0
        return p
    if T < T_FEED:
        from scenes import s02_wave as WAVE
        return WAVE.post(fc, st)
    from scenes import s02_feeds as FEEDS
    return FEEDS.post(fc, st)


def export_hits(S, st):
    from vx.foley import export_events
    ev = []
    ev.append(dict(t=21.0, kind="wind", strength=0.4, pan=0.0, desc="fog lifts over the grey plain; crane up begins"))
    ev.append(dict(t=PLAIN.T_N, kind="power_on", strength=0.9, pan=-0.6,
                   desc="NATION lights up: windows cascade, searchlights ignite (word 'Nations')"))
    ev.append(dict(t=PLAIN.T_F, kind="power_on", strength=0.85, pan=0.05,
                   desc="FAITH lights up: halo ring ignites, dome windows glow (word 'Faiths')"))
    ev.append(dict(t=PLAIN.T_P, kind="power_on", strength=0.85, pan=0.65,
                   desc="PHILOSOPHY lights up: floodlit marble bust (word 'Philosophies')"))
    for te in st.get("ring_t", []):
        ev.append(dict(t=te, kind="pulse", strength=0.5, pan=0.0,
                       desc="broadcast ring pulses from all three antennas"))
    for k in range(8):
        ev.append(dict(t=28.56 + k * 0.63, kind="march", strength=0.35, pan=0.0,
                       desc="lockstep: every marcher on the plain plants a foot in unison (beat grid)"))
    ev.append(dict(t=33.62, kind="rumble", strength=0.5, pan=0.0,
                   desc="thin beige line of slop rises on the horizon; lights stutter; marchers halt"))
    ev += st.get("wave_events", [])
    ev += st.get("feed_events", [])
    export_events(S, "hits", ev)

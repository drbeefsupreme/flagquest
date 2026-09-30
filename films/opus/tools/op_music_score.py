"""OPVS SCHISMATICVM — the score (film 3, owned by Composer).

  source .venv/bin/activate && export VX_FILM=opus
  python films/opus/tools/op_music_score.py                  # every section + master -> audio/stems/music.wav
  python films/opus/tools/op_music_score.py --only m04,m06   # re-render those sections (others from cache)
  python films/opus/tools/op_music_score.py --master         # master only
  python tools/music_verify.py [--spec all]                  # format, sacred silence, loudness, sync points

A sacred score for a candlelit basilica whose mosaics come alive. Tonal centre D for the whole film.
The score keeps the film's one rule in its own terms: everything is made of pieces (struck and plucked
tesserae of sound: bells, harp, psaltery, celesta, organ chords set block by block) EXCEPT the one seamless
thing, the ison — a held D sung by a group of male voices breathing in turn so that it never breaks. The
ison is the Flag of the score: it is there when the candle is lit, it is the plain D the Hypermind's eleven-
note lock chord collapses into, it splits in the schism, it dies with Babel, and it is the one tone that
comes back out of the silence.

Materials
- the leitmotif is the choir's hymn line (Vocalist):  A G | A C | B C# | D   ("Flags be. Flags are. Flags will.
  Flags."), Dorian (C) turning major (C#) on the last step.
- the Tesseract's harmony: sixteen chords at the vertices of a 4-cube (each voice toggles between two pitches),
  walked in Gray-code order so that exactly one voice moves at every turn; the bass (the w-axis, "a direction
  you cannot point to") moves once — A to D, dominant to tonic — at the moment the Flag rotates into existence.
- the schism canon (m06): the leitmotif in canon whose voices fork at every crack of the picture — each voice
  splits in two (new key, new tempo), recursively, until the music is glittering dust.
- Babel: an organ cluster that grows one pipe per tier of the tower, rises, and dies when the wind fails.

Sections write into named buses (global time) and are cached in films/opus/audio/music/cache/sec_<id>.npz.
Sync points are logged to films/opus/audio/music/cues.json (checked by tools/music_verify.py).
"""
import os as _os

_os.environ.setdefault("VX_FILM", "opus")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ.setdefault(_k, "4")
import argparse
import functools
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from op_music_lib import (CACHE, CUE, DUR, LINES, MUS, N, SF_GU, SF_MS, SR, SF2, TLJ, W, Buf, S,  # noqa: F401
                          _resample, bell_to, butter, cel_part, crush, db, envelope, eq, fade, glass, hz, I, ison,
                          m, make_ir, mosaic, noise, organ, PAN, peal, PERC, perc, phone, place, psaltery_to,
                          rev_clip, scene_events, stereo_place, supersaw, svf, tt, V1P, vari, wow, hymn_grid, ding,
                          bell, psaltery, sing_vowel, up48, kick, clap, hat, bass808, saw_note, ep_note, glass_note_)
from music_orch import (brass, crash, cut_after, harp_gliss, harp_roll, limiter, lufs, root, strings,  # noqa: F401
                        swell_to, taiko, timp, timp_roll, tones, true_peak)

assert DUR == 212.0, "run with VX_FILM=opus"
OUT = os.path.join(os.path.dirname(MUS), "stems", "music.wav")
HITS = []


def hit(t, name, desc="", kind="onset"):
    """log a sync point: kind 'onset' (attack lands on t) or 'cut' (music drops away at t)."""
    HITS.append(dict(t=round(float(t), 4), id=name, desc=desc, kind=kind))


def ev(scene, kind, fallback, contains=None, lo=None, hi=None):
    """first exported event of `kind` (optionally inside [lo, hi]) or the fallback time."""
    ts = [x for x in scene_events(scene, kind, contains) if (lo is None or x >= lo) and (hi is None or x <= hi)]
    return ts[0] if ts else fallback


def evs(scene, kind, lo=None, hi=None, contains=None):
    return [x for x in scene_events(scene, kind, contains) if (lo is None or x >= lo) and (hi is None or x <= hi)]


def _events(scene, kind=None):
    """raw exported event dicts of a scene (every cache/foley/<scene>_*events.json), sorted by time."""
    import glob
    import re as _re  # noqa: F401
    out = []
    for f in sorted(glob.glob(os.path.join(os.path.dirname(MUS), "..", "cache", "foley", f"{scene}_*events.json"))):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        for e in (j["events"] if isinstance(j, dict) else j):
            if kind is None or e.get("kind") == kind:
                out.append(e)
    return sorted(out, key=lambda e: e["t"])


def clusters(ts, gap=0.05):
    """first time of each group of events closer than `gap` s (e.g. eleven poles clacking as one)."""
    out = []
    for t in sorted(ts):
        if not out or t - last > gap:
            out.append(t)
        last = t
    return out


def ev_to(scene, kind, fallback, lo=None, hi=None):
    """end time written as '(to X)' in the description of the first `kind` event, else fallback."""
    import re
    for e in _events(scene, kind):
        if (lo is None or e["t"] >= lo) and (hi is None or e["t"] <= hi):
            mm = re.search(r"to (\d+(?:\.\d+)?)", str(e.get("desc", "")))
            if mm:
                return float(mm.group(1))
    return fallback


def ramp(t, bp):
    """piecewise-linear value of breakpoints [(t, v), ...] at global time(s) t."""
    ts, vs = zip(*bp)
    return np.interp(t, ts, vs)


def lines_gain(t, depth=0.5, pad=(0.15, 0.3), ids=None):
    """1 outside dialogue, `depth` inside lines (smooth): lets the music breathe between the words."""
    t = np.asarray(t, float)
    g = np.ones_like(t)
    for ln in TLJ["lines"]:
        if ids is not None and ln["id"] not in ids:
            continue
        a, z = ln["start"] - pad[0], ln["end"] + pad[1]
        w = np.clip(np.minimum((t - (a - 0.35)) / 0.35, ((z + 0.5) - t) / 0.5), 0, 1)
        g = np.minimum(g, 1 - (1 - depth) * w)
    return g


# ================================================================================================ materials
MOTIF = ["A4", "G4", "A4", "C5", "B4", "C#5", "D5"]          # the choir's hymn line (beats 2 2 2 2 2 2 4)
MOTIF_B = [2, 2, 2, 2, 2, 2, 4]

# the Tesseract: voice -> (state 0, state 1); Gray-code walk visits all 16 vertices, one voice per turn
TESS = [("D5", "E5"), ("A4", "B4"), ("E3", "F#3"), ("A1", "D2")]   # bit0 soprano .. bit3 bass (the w-axis)
GRAY = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]


def tess_chord(g):
    return [m(TESS[k][(g >> k) & 1]) for k in range(4)]


def voice_segments(times, chords, t_end):
    """[(voice, t0, t1, midi)] — a voice is re-attacked only when it moves."""
    out = []
    for v in range(len(chords[0])):
        t0, cur = times[0], chords[0][v]
        for t, ch in zip(times[1:], chords[1:]):
            if ch[v] != cur:
                out.append((v, t0, t, cur))
                t0, cur = t, ch[v]
        out.append((v, t0, t_end, cur))
    return out


# ================================================================================================ m01
def m01(b):
    """Incipit: the Tesseract (0–22). One church bell as the candle is lit and the ison enters under the
    cantor's INCIPIT; the vault's rings of stars turn (glass and celesta); the Tesseract unfolds out of one
    tessera and turns through its sixteen chords as a mosaic of struck notes over an A pedal; on 'rotated'
    the bass moves A→D and the first seamless sound in the film — strings and voices, legato — swells under
    the Flag; the organ and the bell set the title."""
    rng = np.random.default_rng(101)
    t_c = ev("m01", "strike", CUE["candle_lit"], lo=0.3, hi=1.5)
    bell_to(b, "bell", t_c, "D3", gain=0.55, pan=0.15, strike=0.85, seed=3, size=1.15)
    hit(t_c, "candle_lit", "the basilica's bell (strike D3, hum D2) as the candle is lit")
    # the ison: under the cantor, receding under the lector, swelling between lines
    inc = hymn_grid().get("cantor_incipit") or [{"t": 4.48}]
    t_fin = float(inc[-1]["t"])  # the cantor's cadence on D3: the ison blooms with it
    lv = lambda t: (lines_gain(t, 0.62, ids=["N01", "N02", "N03"]) * ramp(t, [(0, 1.0), (14.5, 1.0), (16.0, 0.75),
                                                                               (19.0, 0.8), (22.0, 0.7)])
                    * (1 + 0.35 * np.exp(-((t - t_fin - 0.5) / 0.6) ** 2)))
    ison(b, "ison", t_c + 0.35, 21.9, notes=("D3", "D3", "D3", "D2", "D3", "D2"), gain=0.5, fin=3.2, fout=2.4,
         seed=11, level=lv)
    t_tess = CUE["tesseract"]
    t_flag = ev("m01", "emerge", W("N03", "rotated"), lo=14.0, hi=15.6)
    t_title = ev("m01", "title", CUE["title"], lo=19.0, hi=20.2)
    # --- 5.6 the vault turns: rings of stars (glass + celesta), pitch = the Tesseract's first chord
    c0 = tess_chord(GRAY[0])
    rings = [(["A5", "E6", "B5", "D6", "A6"], 3.3, 0.0), (["E6", "D7", "A6", "B6"], 2.2, 0.9),
             (["A6", "E7", "D6"], 1.6, 0.4)]
    cel = cel_part(1.4)
    for k, (notes, per, ph) in enumerate(rings):
        t = CUE["tesseract"] - 2.8 + ph
        j = 0
        while t < t_flag - 0.1:
            x = (t - (t_tess - 2.8)) / (t_flag - t_tess + 2.8)
            g = np.clip(x / 0.25, 0, 1) * (1 - 0.6 * np.clip((t - t_tess) / 4.0, 0, 1))
            ang = 2 * np.pi * (t / (per * len(notes)) + k / 3)
            n = m(notes[j % len(notes)])
            if k == 0:
                y = glass(hz(n), 2.2, att=0.003, decay=1.3)
                b.add("glass", stereo_place(np.stack([y, y], 1), 0.75 * np.sin(ang), 1.0), t=t, gain=0.09 * g)
            else:
                cel.note(t, n, 0.3, 0.18 + 0.2 * g)
            t += per / len(notes) * (1 + 0.04 * rng.normal())
            j += 1
    # --- 8.4 the Tesseract: one gold tessera, unfolding into the first chord, then its 4D turning
    I("glock").note(b, "keys", t_tess, "A6", None, 0.55, 0.0, gain=0.07)
    cel.note(t_tess, m("A6"), 0.6, 0.5)
    hit(t_tess, "tesseract", "one tessera (glock+celesta A6); the Tesseract unfolds")
    # every quarter turn and every inside-out passage of the 4D rotation moves one voice (M01's events)
    turns = sorted(set(evs("m01", "turn", 9.3, t_flag - 0.3) + evs("m01", "inside_out", 9.3, t_flag - 0.3)))
    if len(turns) >= 7:
        turns = [turns[int(round(i))] for i in np.linspace(0, len(turns) - 1, 7)]
    else:
        turns = list(np.linspace(9.6, t_flag - 0.72, 7))
    ctimes = [t_tess] + turns + [t_flag]
    chords = [tess_chord(GRAY[i]) for i in range(9)]
    # the organ holds the chord, re-attacking only the voice that moves (quiet flue stops)
    # the unfolding: square (the star-shadows), cube, tesseract — bass, tenor, alto, soprano enter in turn
    unf = sorted(evs("m01", "star_shadow", t_tess, t_tess + 1.0) + evs("m01", "whoosh", t_tess + 0.2, t_tess + 1.2))
    unfold = (unf[:3] + [unf[2] + 0.35]) if len(unf) >= 3 else [t_tess + 0.05, t_tess + 0.35, t_tess + 0.7,
                                                                 t_tess + 1.05]
    for (v, a, z, n) in voice_segments(ctimes, chords, t_flag + 0.6):
        if a == t_tess:
            a = unfold[3 - v]
        if v == 3:
            I("organ_pedal").note(b, "org", a, n, z - a, 0.8, 0.0, gain=0.26, att=0.6, rel=0.9)
        else:
            I("organ_soft").note(b, "org", a, n, z - a + 0.05, 0.8, [-0.3, 0.3, 0.0][v], gain=0.13, att=0.25,
                                 rel=0.6)
    # the turning, set in tesserae: struck chord tones; the moving voice glints at each turn
    def cur(t):
        i = int(np.searchsorted(ctimes, t, "right") - 1)
        return chords[max(0, min(i, 8))]

    moved = {}
    for i in range(1, 8):
        d = [v for v in range(4) if chords[i][v] != chords[i - 1][v]]
        moved[ctimes[i]] = chords[i][d[0]]
    mosaic(b, t_tess + 1.0, t_flag - 0.05, cur, lambda t: 4.0 + 7.0 * ((t - t_tess) / (t_flag - t_tess)) ** 1.5,
           lo="A4", hi="E7", seed=12, gain=0.9, cel=cel, vel=(0.18, 0.45),
           inst=(("cel", 1.0), ("glass", 0.8), ("harp", 0.7), ("glock", 0.25)))
    for tm, n in moved.items():
        top = n
        while top < m("A5"):
            top += 12
        cel.note(tm, top, 0.5, 0.55)
        I("harp").note(b, "keys", tm, min(top, 100), None, 0.55, rng.uniform(-0.5, 0.5), gain=0.2)
        y = glass(hz(top + 12), 2.0, att=0.002, decay=1.2)
        b.add("glass", stereo_place(np.stack([y, y], 1), rng.uniform(-0.6, 0.6), 1.0), t=tm, gain=0.08)
    for k, tio in enumerate(evs("m01", "inside_out", t_tess, t_flag)):
        I("glock").note(b, "keys", tio, "E7", None, 0.35, 0.3 * (-1) ** k, gain=0.05)
    # after the Flag: the Tesseract's tiles scatter and circle the seamless cloth (sparse, orbiting)
    t_sh1 = ev_to("m01", "shatter", 19.3, lo=t_flag - 0.2, hi=t_flag + 0.5)
    mosaic(b, t_flag + 0.35, t_sh1, lambda t: [2, 6, 9, 4], lambda t: 2.2, lo="A5", hi="E7", seed=13, gain=0.55,
           cel=cel, vel=(0.12, 0.3), pan=lambda t, r: 0.85 * np.sin(2 * np.pi * (t - t_flag) / 2.6),
           inst=(("cel", 1.0), ("glass", 1.0)))
    # --- 14.8 the Flag rotates into existence: the bass moves A→D; the first seamless sound
    hit(t_flag, "flag_rotates_in", "bass A→D (V→I) + legato strings/voices swell: the first seamless sound")
    I("organ_pedal").note(b, "org", t_flag, "D2", t_title - t_flag + 0.3, 0.8, 0.0, gain=0.3, att=0.05, rel=1.0)
    I("harp").note(b, "keys", t_flag, "D2", None, 0.6, -0.3, gain=0.3)
    timp(b, t_flag, "D2", 0.35, gain=0.5)
    # woven without seam: one voice at a time, slurred, never re-struck together (vertices 9 -> 12 -> 9)
    wv = [(t_flag, GRAY[8]), (16.15, GRAY[9]), (17.2, GRAY[10]), (W("N03", "without"), GRAY[11]),
          (W("N03", "seam"), GRAY[8])]
    tw_end = t_title - 0.05
    wt = [x[0] for x in wv]
    wch = [tess_chord(g) for _, g in wv]
    sdyn = lambda a, z: [(0, float(ramp(a, SW))), (z - a, float(ramp(z, SW)))]
    SW = [(t_flag, 0.08), (t_flag + 1.6, 0.42), (18.0, 0.46), (18.9, 0.52), (t_title - 0.2, 0.3), (t_title, 0.2)]
    for (v, a, z, n) in voice_segments(wt, wch, tw_end):
        if v == 3:
            strings(b, a, z - a, dyn=sdyn(a, z), att=0.35, rel=0.8, cb=["D2"], vc=["D3"], gain=0.55)
        elif v == 2:
            strings(b, a, z - a, dyn=sdyn(a, z), att=0.35, rel=0.8, va=[n], gain=0.5)
        elif v == 1:
            strings(b, a, z - a + 0.12, dyn=sdyn(a, z), att=0.3 if a == t_flag else 0.18, rel=0.8, v2=[n], gain=0.5)
        else:
            strings(b, a, z - a + 0.12, dyn=sdyn(a, z), att=0.3 if a == t_flag else 0.18, rel=0.8, v1=[n], gain=0.55)
    I("vln_solo").note(b, "str", t_flag + 0.4, "D6", tw_end - t_flag - 0.4, 0.5, -0.1, gain=0.22,
                       dyn=[(0, 0.05), (2.0, 0.3), (tw_end - t_flag - 0.9, 0.35), (tw_end - t_flag - 0.4, 0.15)],
                       att=1.2, rel=0.8)
    oo = SF2(SF_MS, 0, 53, gain=0.9)
    for n in ["D4", "F#4", "A4", "D5"]:
        oo.note(t_flag + 0.2, m(n), tw_end - t_flag, 0.55)
    y = oo.render()[0]
    if y is not None:
        e = envelope([(0, 0), (1.8, 0.8), (tw_end - t_flag - 1.0, 1.0), (len(y) / SR, 0.0)], len(y))
        place(b, "choir", y * e[:, None], t_flag + 0.2, gain=0.45)
    # --- 19.6 title: the bell again, the organ sets OPVS SCHISMATICVM in gold
    timp_roll(b, t_title - 0.9, 0.9, "D2", [(0, 0.05), (0.9, 0.55)], gain=0.55)
    bell_to(b, "bell", t_title, "D3", gain=0.5, pan=-0.1, strike=1.0, seed=4, size=1.15)
    organ(b, t_title, 2.4, ["D3", "A3", "D4", "F#4", "A4", "D5"], "plenum", pedal="D2", gain=0.9, att=0.03, rel=1.6,
          dyn=[(0, 0.95), (1.2, 0.8), (2.4, 0.6)])
    strings(b, t_title, 2.0, dyn=[(0, 0.6), (2.0, 0.35)], att=0.02, rel=1.2, v1=["D5", "A5"], v2=["F#5"],
            va=["A4"], vc=["D3", "A3"], cb=["D2"], gain=0.7)
    harp_roll(b, t_title, ["D2", "A2", "D3", "F#3", "A3", "D4"], 0.7, 0.012, gain=1.0)
    swell_to(b, t_title - 0.01, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.22, after=0.0)
    hit(t_title, "title", "bell + organ plenum D major + timpani: the title is set")
    for k, tl_ in enumerate(clusters(evs("m01", "set", t_title + 0.03, t_title + 1.0), 0.015)[:16]):  # letters
        cel.note(tl_, m(["D6", "E6", "F#6", "A6", "B6", "D7", "E7"][k % 7]), 0.3, 0.3)
    cel.to(b, "keys", pan=0.2, gain=0.9)


# ================================================================================================ m02
PB = 22.43  # the procession's first downbeat; one beat per second (the reveals land on beats 4, 5, 6)
PROC = [  # beat: (chord tones low->high for the organ, pedal, machine accent)
    ("Dm", ["D3", "F3", "A3", "D4"], "D2", None), ("Dm/C", ["C3", "F3", "A3", "D4"], "C2", None),
    ("Bb", ["Bb2", "F3", "A3", "D4"], "Bb1", None), ("A", ["A2", "E3", "A3", "C#4"], "A1", None),  # Bbmaj7: the litany recites on A3
    ("D", ["D3", "F#3", "A3", "D4"], "D2", "nations"), ("G", ["D3", "G3", "B3", "D4"], "G1", "faiths"),
    ("Bm", ["D3", "F#3", "B3", "D4"], "B1", "philosophies"), ("A", ["C#3", "E3", "A3", "E4"], "A1", None),
    ("D/F#", ["D3", "F#3", "A3", "D4"], "F#1", None), ("G", ["D3", "G3", "B3", "D4"], "G1", None),
    ("D/A", ["D3", "F#3", "A3", "D4"], "A1", "picture"), ("A", ["C#3", "E3", "A3", "E4"], "A1", None)]


def procession(b, sick=False):
    """the great machines of meaning: a stately ground (organ pedal) at 60 BPM; the machines' gilded cogs as
    harp/celesta sixteenths, their processions as walking pizzicati; NATIONES / FIDES / PHILOSOPHIAE enter as
    brass / organ+choir / winds on their words; the whole picture (tutti) promises a cadence that never comes."""
    t_rev = [ev("m02", "reveal", PB + 4 + i, lo=PB + 3.4 + i, hi=PB + 4.6 + i) for i in range(3)]
    beats = [PB + k for k in range(12)]
    beats[4:7] = t_rev  # the three machines light up exactly on their words
    cut = CUE["slop"] - 0.04
    dyn = [0.28, 0.3, 0.32, 0.36, 0.5, 0.52, 0.55, 0.62, 0.62, 0.68, 0.82, 0.92]
    lv = lambda t: lines_gain(t, 0.7, ids=["N04", "N05"])
    cel = cel_part(1.2)
    for k, (sym, notes, ped, acc) in enumerate(PROC):
        a = beats[k]
        z = beats[k + 1] if k < 11 else cut
        d = z - a
        v = dyn[k] * float(lv(a + 0.3))
        reg = "flute" if k < 4 else ("principal" if k < 10 else "plenum")
        organ(b, a, d + 0.04, notes, reg, pedal=ped, gain=0.55 + 0.6 * v, att=0.05, rel=0.9 if k < 11 else 0.25)
        top = [m(x) + 12 for x in notes[1:]]
        if k >= 4:  # the strings join when the machines are revealed
            strings(b, a, d + 0.05, dyn=[(0, v * 0.8), (d, v * 0.85)], att=0.08 if k > 4 else 0.02,
                    rel=0.6 if k < 11 else 0.2, v1=[top[-1]], v2=[top[-2]], va=[m(notes[1])], vc=[m(ped) + 12],
                    gain=0.5)
        # cogs: a rotating 16th figure of chord tones (harp + celesta), 4 per beat
        cyc = [m(x) + 24 for x in notes[1:]] + [m(notes[-1]) + 12]
        for q in range(4):
            tq = a + q * d / 4
            n = cyc[(k * 4 + q) % len(cyc)]
            if q % 2 == 0:
                I("harp").note(b, "keys", tq, min(n, 96), None, 0.35 + 0.3 * v, -0.45, gain=0.16, seed=k * 4 + q)
            else:
                cel.note(tq, min(n + 12, 100), 0.2, 0.2 + 0.3 * v)
        # the processions: walking pizzicato eighths
        for q in range(2):
            n = m(ped) + 12 + (0 if q == 0 else 7)
            I("vc_pizz").note(b, "str", a + q * d / 2, n, None, 0.45 + 0.3 * v, PAN["vc"], gain=0.3 + 0.25 * v,
                              seed=k * 2 + q)
        if acc == "nations":
            brass(b, a, 3.0, dyn=[(0, 0.75), (0.5, 0.45), (3.0, 0.4)], att=0.02, rel=0.6, tpt=["A4", "D5", "F#5"],
                  hn=["D4", "F#4", "A4"], tbn=["D3", "A3"], gain=0.3)
            timp(b, a, "D2", 0.7, gain=0.8)
            hit(a, "nationes", "NATIONES: brass + timpani")
        elif acc == "faiths":
            organ(b, a, 2.2, ["G3", "B3", "D4", "G4"], "principal", gain=0.6, att=0.04, rel=0.8)
            ch = SF2(SF_MS, 0, 52, gain=0.8)
            for n in ["G3", "D4", "G4", "B4"]:
                ch.note(a, m(n), 2.2, 0.62)
            ch.to(b, "choir", pan=0.0)
            hit(a, "fides", "FIDES: organ + choir")
        elif acc == "philosophies":
            for (key, n, p) in (("fl_sus", "F#5", PAN["fl"]), ("ob_sus", "D5", PAN["ob"]), ("cl_sus", "B4", PAN["cl"])):
                I(key).note(b, "ww", a, n, 2.0, 0.6, p, gain=0.45, dyn=[(0, 0.7), (2.0, 0.45)], att=0.03, rel=0.5)
            harp_roll(b, a, ["B2", "F#3", "B3", "D4", "F#4", "B4"], 0.65, 0.03, gain=1.0)
            hit(a, "philosophiae", "PHILOSOPHIAE: flute/oboe/clarinet + harp")
        elif acc == "picture":
            brass(b, a, 2.0, dyn=[(0, 0.8), (2.0, 0.85)], att=0.03, rel=0.4, tpt=["A4", "D5", "F#5"],
                  hn=["D4", "F#4", "A4"], tbn=["A2", "D3"], tuba=["A1"], gain=0.3)
            timp(b, a, "A2", 0.75, gain=0.8)
            I("fl_sus").note(b, "ww", a, "F#5", 2.0, 0.6, PAN["fl"], gain=0.4, dyn=[(0, 0.6), (2.0, 0.7)], att=0.05)
            hit(a, "whole_picture", "tutti on 'picture': the three machines as one image")
    # the dominant swells towards a resolution that never comes
    a = beats[11]
    brass(b, a, cut - a, dyn=[(0, 0.75), (cut - a, 1.0)], att=0.04, rel=0.15, tpt=["A4", "C#5", "E5"],
          hn=["E4", "A4", "C#5"], tbn=["A2", "E3"], tuba=["A1"], gain=0.28)
    timp_roll(b, a, cut - a, "A2", [(0, 0.4), (cut - a, 0.95)], gain=0.7)
    cel.to(b, "keys", pan=0.25, gain=0.8)
    cut_after(b, cut + 0.03, 0.06)
    hit(cut + 0.03, "slop_cut", "the procession's cadence is cut off before 'Then came the slop.'", kind="cut")


def bus_mix(b, t0, t1, keys=None):
    """stereo sum of (some) buses of b over [t0, t1)."""
    y = np.zeros((S(t1) - S(t0), 2), np.float32)
    for k, x in b.b.items():
        if keys is None or k in keys:
            y += x[S(t0):S(t1)]
    return y


def m02(b):
    """The Great Machines and the Slop (22–48.5): the procession (above), silence, one phone ding; the wave:
    the machines' music keeps running but sickens (warped, crushed, sagging tape) while cheap pop feeds in
    every key and tempo pour in wave upon wave through tiny speakers into the basilica; the push into one
    tessera: one feed fills the frame and dissolves into static."""
    rng = np.random.default_rng(202)
    procession(b)
    # the tape of the machines (dry, all buses) for the sick reprise
    tape = bus_mix(b, PB, CUE["slop"])
    t_flip = ev("m02", "flip", W("N06", "slop") + 0.1, lo=34.4, hi=36.2)
    y = phone(np.stack([ding()] * 2, 1), 700, 9000, 1.2)
    place(b, "slop", y, t_flip, gain=0.3, pan=0.35)
    hit(t_flip, "slop_flip", "one tessera flips into a phone: notification ding (phone speaker in the basilica)")
    f0 = phone(L_feed("kids", "E", 1.6, 1), 800, 7000)
    place(b, "slop", f0 * np.linspace(0, 1, len(f0))[:, None] ** 0.5, t_flip + 0.5, gain=0.35, pan=0.35)
    # --- 36.3 the wave: the machines' music, sick
    t_w = ev("m02", "wave", CUE["slop_crash"], lo=35.8, hi=37.0)
    t_push = ev("m02", "push", 47.0, lo=46.0, hi=47.8)
    t_static = ev("m02", "push", 48.3, lo=47.8, hi=48.49)
    dur = t_push + 0.6 - t_w
    n = S(dur)
    tn = np.arange(n) / SR
    rate = wow(n, 45, 0.45, 9, seed=5) * (1 - 0.14 * (tn / dur) ** 1.6)  # warped and sagging
    src = tape[S(1.0):]  # start from beat 2 so the ground is recognisable
    sick = vari(np.concatenate([src, src]), rate)
    sick = crush(sick, 10, 3)
    sick = eq(sick, ("hp", 90), ("lp", 5200), ("pk", 1400, 3.0, 0.8))
    env = envelope([(0, 0.0), (0.05, 1.0), (dur - 3.5, 0.7), (dur, 0.0)], n)
    place(b, "slop", sick * env[:, None], t_w, gain=0.8)
    # the crash itself: the procession's unresolved dominant, smashed — crushed organ/brass stab diving in pitch,
    # an 808 boom, a crash cymbal (all in the slop's cheap colours)
    stab = np.zeros((S(1.2), 2), np.float32)
    for n in ["A2", "E3", "A3", "C#4", "E4", "A4"]:
        stab += stereo_place(I("organ_loud").render(m(n), 1.0, 0.8)[:S(1.2)], 0.0, 1.0)
    ns = len(stab)
    stab = vari(stab, np.linspace(1.0, 0.45, ns) ** 1.5) * np.linspace(1, 0, ns)[:, None] ** 1.2
    stab = phone(crush(stab, 6, 3), 200, 7000, 2.5)
    place(b, "slop", stab, t_w, gain=0.55)
    boom = butter(bass808("D2", 0.9, glide_from="D3"), "highpass", 45, 2)
    place(b, "slop", boom, t_w, gain=0.5)
    crash(b, t_w, "mp", 0.6)
    hit(t_w, "slop_crash", "the wave: a smashed, diving organ stab + 808 + crash; the procession returns warped")
    # feeds: (t, style, key, pan) — wave upon wave, a different feed for every face
    fronts = [(t_w, "edm", "E", -0.8), (t_w + 0.35, "kids", "C", -0.4), (t_w + 0.7, "ad", "G", 0.0),
              (t_w + 1.2, "lofi", "Bb", 0.45), (t_w + 2.2, "trap", "C#", 0.85),
              (W("N07", "Wave"), "hyper", "F#", -0.6), (W("N07", "wave"), "phonk", "G", 0.6)]
    tf = W("N07", "different")
    for j, (st, k) in enumerate([("asmr", "A"), ("edm", "Ab"), ("kids", "F#"), ("lofi", "D"), ("ad", "Eb"),
                                 ("trap", "F"), ("hyper", "B"), ("kids", "Db")]):
        fronts.append((tf + 0.16 * j, st, k, float(np.clip(rng.normal(0, 0.6), -0.95, 0.95))))
    fronts += evs_fallback_waves(t_w)
    t_end = t_push + 0.25
    for j, (t, st, k, p) in enumerate(fronts):
        if t >= t_end - 0.5:
            continue
        y = L_feed(st, k, t_end - t, 20 + j)
        nn = len(y)
        yr = vari(y, wow(nn, 20 + 30 * rng.random(), 0.3 + rng.random(), 5, seed=j))
        yr = phone(yr, float(rng.uniform(350, 900)), float(rng.uniform(4500, 9000)), float(rng.uniform(1.2, 2.5)))
        tt_ = t + np.arange(nn) / SR
        g = np.clip((tt_ - t) / 0.08, 0, 1) * np.clip((t_end - tt_) / 0.4, 0, 1)
        g = g * lines_gain(tt_, 0.6, ids=["N07"]) * ramp(tt_, [(36, 1.25), (40.9, 1.25), (41.3, 0.85), (44.8, 1.0),
                                                               (46.9, 1.15), (47.4, 0.5)])
        place(b, "slop", yr * g[:, None], t, gain=0.75, pan=p, width=0.6)
    # 'until no two tesserae showed the same world': every feed stutters apart
    a0 = W("N07", "until")
    stutter_bus(b, "slop", a0, t_push, rng)
    # --- push into one tessera: one feed fills the frame, then static
    d1 = t_static - t_push + 0.2
    one = L_feed("edm", "E", d1 + 0.3, 99)
    n1 = S(d1)
    one = one[:n1]
    x = np.arange(n1) / n1
    op = phone(one, 600, 7000, 2.0) * (1 - x[:, None] ** 2) + one * (x[:, None] ** 2)  # the speaker opens up
    nz = np.random.default_rng(7).standard_normal((n1, 2)).astype(np.float32) * 0.12
    mixw = np.clip((x * d1 - (d1 - 0.55)) / 0.45, 0, 1)[:, None]  # dissolves into static in the last 0.55 s
    op = crush(op, 6, 2) * (1 - mixw) + nz * mixw
    op *= (0.3 + 0.9 * x ** 1.5)[:, None]
    k = S(0.03)
    op[-k:] *= np.linspace(1, 0, k)[:, None]
    place(b, "slop", op, t_push, gain=0.9)
    hit(t_push, "push_in", "one feed fills the frame, dissolves into static")


@functools.lru_cache(maxsize=None)
def _feed(style, key, dur, seed):
    from op_music_lib import pop_feed
    return pop_feed(style, key, dur, seed=seed)


def L_feed(style, key, dur, seed):
    return _feed(style, key, round(float(dur), 3), seed).copy()


def evs_fallback_waves(t_w):
    """later wave fronts exported by M02 (each adds a feed)."""
    out = []
    styles = [("kids", "A"), ("edm", "C"), ("ad", "D")]
    for j, t in enumerate(evs("m02", "wave", t_w + 1.0, 46.0)[:3]):
        out.append((t, styles[j][0], styles[j][1], [-0.7, 0.7, 0.0][j]))
    return out


def stutter_bus(b, bus, t0, t1, rng):
    """buffer-repeat glitches on one bus between t0 and t1 (segments shrink: the image breaks into tiles)."""
    x = b.get(bus)
    pos = S(t0)
    L = 0.12
    while pos < S(t1) - S(0.02):
        n = S(L)
        reps = int(rng.integers(2, 4))
        chunk = x[pos:pos + n].copy()
        r = min(48, n // 4)
        chunk[:r] *= np.linspace(0, 1, r)[:, None]
        chunk[-r:] *= np.linspace(1, 0, r)[:, None]
        for q in range(1, reps):
            a = pos + q * n
            e = min(a + n, S(t1))
            if e > a:
                x[a:e] = chunk[:e - a]
        pos += n * reps + S(float(rng.uniform(0.05, 0.25)))
        L = max(0.025, L * 0.9)


# ================================================================================================ m03
def m03(b):
    """The Edict of the Jaguar (48.5–62.5): as the static resolves into gold, an imperial acclamation — the
    Byzantine court had an organ, so: organ + brass + timpani in D, pompous; a soft ceremonial bed under the
    broadcast; the golden stylus signs in a flourish of violins/harp/celesta; 'banned!' — the seal: tutti
    hit; then sly pizzicati under 'Except the yellow ones', a pious organ A-men on 'sacrament' that lands with
    the ka-ching sting (bell, glock, celesta, triangle, muted trumpets), and the glint thins to one high tone."""
    rng = np.random.default_rng(303)
    t0 = ev("m03", "resolve", 48.7, lo=48.45, hi=49.2)
    # --- fanfare: timpani roll into a triplet pickup and the acclamation chord
    timp_roll(b, 48.5, t0 + 0.3 - 48.5, "A2", [(0, 0.2), (t0 + 0.3 - 48.5, 0.8)], gain=0.7)
    for j in range(3):
        brass(b, t0 + 0.1 * j, None, "stac", 0.85, tpt=["D5", "F#5"], gain=0.2)
    ta = t0 + 0.3
    brass(b, ta, 1.1, dyn=[(0, 0.95), (0.35, 0.55), (1.1, 0.3)], att=0.0, rel=0.5, tpt=["A4", "D5", "F#5", "A5"],
          hn=["D4", "F#4", "A4"], tbn=["D3", "A3"], tuba=["D2"], gain=0.22)
    organ(b, ta, 1.2, ["D3", "A3", "D4", "F#4", "A4"], "plenum", pedal="D2", gain=0.8, att=0.02, rel=0.9,
          dyn=[(0, 1.0), (0.5, 0.6), (1.2, 0.45)])
    timp(b, ta, "D2", 0.9, gain=1.0)
    crash(b, ta, "mp", 0.5)
    hit(ta, "acclamation", "imperial acclamation chord as the gold resolves (brass + organ + timpani)")
    # --- the ceremonial bed under the broadcast (horns + organ flute, timpani heartbeat, snare taps)
    t_sign = ev("m03", "stroke", W("J01", "sign"), lo=52.2, hi=52.9)
    t_seal = ev("m03", "seal", W("J01", "banned"), lo=57.7, hi=58.3)
    bed = [(ta + 1.0, ["D3", "A3", "D4", "F#4"], "D2"), (W("J01", "and"), ["D3", "G3", "B3", "D4"], "G1"),
           (W("J01", "Today"), ["D3", "F#3", "A3", "D4"], "D2"), (t_sign, ["E3", "G3", "B3", "E4"], "E2"),
           (W("J01", "Munitions"), ["D3", "G3", "B3", "D4"], "G1"), (W("J01", "twenty"), ["E3", "A3", "C#4", "E4"], "A1"),
           (W("J01", "Weaponized"), ["C#3", "E3", "A3", "E4"], "A1")]
    for i, (a, notes, ped) in enumerate(bed):
        z = bed[i + 1][0] if i + 1 < len(bed) else t_seal - 0.02
        organ(b, a, z - a + 0.05, notes, "flute", pedal=ped, gain=0.6, att=0.15, rel=0.5)
        brass(b, a, z - a + 0.05, dyn=[(0, 0.22), (z - a, 0.26 + (0.4 if i == len(bed) - 1 else 0.0))], att=0.2,
              rel=0.3, hn=[notes[1], notes[2]], gain=0.32)
    for k in range(int(t_seal - ta - 1.0)):
        tk = ta + 1.0 + k
        if tk < t_seal - 2.0:
            timp(b, tk, "D2" if k % 2 == 0 else "A1", 0.3, gain=0.35)
    # 'and everybody else!': the whole court (tituli letters set with glints)
    t_all = ev("m03", "reveal", W("J01", "and"), lo=50.3, hi=51.2)
    harp_gliss(b, t_all, t_all + 1.2, "D4", "A6", scale=(2, 4, 6, 7, 9, 11, 1), vel=0.45, gain=0.8)
    brass(b, W("J01", "else"), 0.7, dyn=[(0, 0.3), (0.55, 0.55), (0.7, 0.3)], att=0.1, rel=0.3,
          tpt=["F#5", "A5"], gain=0.28)
    # --- the golden stylus: a calligraphic flourish (violins + harp + celesta), light and high
    t_se = ev_to("m03", "stroke", t_sign + 1.8, lo=t_sign - 0.3, hi=t_sign + 0.3)
    fl = ["D5", "E5", "F#5", "A5", "B5", "D6", "C#6", "B5", "A5", "B5", "C#6", "D6"]
    for j, n in enumerate(fl):
        tj = t_sign + (t_se - t_sign - 0.5) * (j / (len(fl) - 1)) ** 1.25
        I("vln_sus").note(b, "str", tj, n, 0.16 if j < len(fl) - 1 else t_se - tj + 0.3, 0.55, PAN["vln"], gain=0.28,
                          att=0.01, rel=0.2, seed=j)
    I("vln_trem").note(b, "str", t_se - 0.5, "D6", 0.55, 0.5, PAN["vln"], gain=0.2, att=0.02, rel=0.25)
    harp_gliss(b, t_sign, t_sign + 0.55, "A4", "D7", vel=0.5, gain=0.9)
    cel = cel_part(1.4)
    for j, n in enumerate(["D6", "F#6", "A6", "D7"]):
        cel.note(t_se + 0.04 * j, m(n), 0.4, 0.45)
    hit(t_sign, "edict_sign", "the stylus flourish (harp gliss + violins)")
    # --- 'Weaponized memes are hereby banned!' -> the seal
    a = W("J01", "Weaponized")
    timp_roll(b, a, t_seal - a, "A2", [(0, 0.15), (t_seal - a, 0.9)], gain=0.8)
    swell_to(b, t_seal - 0.01, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.25, after=0.0)
    brass(b, t_seal, None, "stac", 1.0, tpt=["D5", "F#5", "A5"], hn=["D4", "A4"], tbn=["D3", "A3"], tuba=["D2"],
          gain=0.2)
    brass(b, t_seal, 0.9, dyn=[(0, 0.9), (0.3, 0.5), (0.9, 0.35)], att=0.0, rel=0.6, tpt=["A4", "D5"],
          hn=["F#4", "A4"], tbn=["D3"], tuba=["D2"], gain=0.22)
    organ(b, t_seal, 0.95, ["D3", "A3", "D4", "F#4", "A4", "D5"], "full", pedal="D2", gain=0.75, att=0.01, rel=0.9)
    strings(b, t_seal, None, "spic", 0.9, v1=["D5", "A5"], v2=["F#5"], va=["A4"], vc=["D3"], cb=["D2"], gain=0.4)
    timp(b, t_seal, "D2", 1.0, gain=1.2)
    taiko(b, t_seal, "f", gain=0.45)
    crash(b, t_seal, "ff", 0.8)
    hit(t_seal, "edict_seal", "the seal: brass + organ + timpani + crash tutti on 'banned'")
    cut_after(b, W("J02", "Except") - 0.02, 0.12)
    # --- 'Except the yellow ones...': sly pizzicati + bassoon, a pious organ A-men lands with ka-ching
    te = W("J02", "Except")
    sly = [("D3", 0.0), ("F3", 0.25), ("G#3", 0.5), ("A3", 0.75), ("A2", 1.1), ("C#3", 1.35), ("E3", 1.6),
           ("A3", 1.85)]
    for j, (n, dt) in enumerate(sly):
        I("vc_pizz").note(b, "str", te + dt, n, None, 0.55, PAN["vc"], gain=0.42, seed=j)
        if j % 2 == 0:
            I("bsn_stac").note(b, "ww", te + dt, m(n) - 12, None, 0.5, PAN["bsn"], gain=0.4, seed=j)
    t_coin = ev("m03", "coin", 60.95, lo=60.3, hi=61.6)
    cel.note(t_coin, m("A6"), 0.3, 0.45)
    I("harp").note(b, "keys", t_coin, "A5", None, 0.5, 0.3, gain=0.2)
    t_k = ev("m03", "flare", CUE["ka_ching"], lo=61.8, hi=62.3)
    t_am = W("J02", "sacrament")
    organ(b, t_am, t_k - t_am + 0.02, ["G3", "B3", "D4", "G4"], "flute", pedal="G2", gain=0.75, att=0.12, rel=0.1)
    organ(b, t_k, 0.9, ["F#3", "A3", "D4", "F#4"], "flute", pedal="D2", gain=0.7, att=0.02, rel=0.6)
    # ka-ching
    for j, n in enumerate(["D6", "F#6", "A6", "D7"]):
        cel.note(t_k + 0.035 * j, m(n), 0.5, 0.7)
    I("glock").note(b, "keys", t_k, "D7", None, 0.8, 0.25, gain=0.09)
    I("glock").note(b, "keys", t_k + 0.07, "A7", None, 0.7, 0.3, gain=0.07)
    bell_to(b, "bell", t_k, "A5", gain=0.22, pan=0.3, strike=0.9, seed=7, size=0.35)
    perc(b, t_k, PERC + "Triangle3-Hit_v2_rr1_Sum.wav", 0.3, 0.3)
    brass(b, t_k + 0.02, None, "stac", 0.6, tpt=["F#5", "A5"], gain=0.12)
    brass(b, t_k + 0.16, 0.35, "mute", dyn=[(0, 0.6), (0.35, 0.3)], att=0.01, rel=0.2, tpt=["A5", "D6"], gain=0.3)
    hit(t_k, "ka_ching", "comic sting: bell + glock + celesta + triangle + muted trumpets, organ A-men lands")
    # the glint thins to one high tone (-> m04's darkness)
    glass_note_(b, t_k + 0.1, "A6", 1.8, gain=0.12, pan=0.0, decay=0.7)
    cel.to(b, "keys", pan=0.2, gain=0.9)


# ================================================================================================ m04
FIFTHS = ["D2", "A2", "E3", "B3", "F#4", "C#5"]  # the six canons: a stack of fifths (D lydian)


def m04(b):
    """Flagartha (62.5–95.75): an eye opens on a 32' pedal; the wheels within wheels are interlocking organ
    ostinati (3 against 4) on the stack of fifths; six wings ignite as six rising entries (D A E B F# C#)
    that pile into one Lydian chord under the Hypermind's words; the wheels lock (74.2) on the whole stack —
    and as the plain Flag descends, every colour drains out of it until one plain D is left (the ison).
    Then the disputation, set as a scholastic two-voice counterpoint: VIDETVR QVOD NON — a psaltery alone
    (the objection, the hymn line); SED CONTRA — the harp answers in contrary motion; a too-perfect synth
    'world' cracks and falls; 'rise' glints; RESPONDEO — the voices close on the octave, and the organ pedal
    swells towards the hard cut on 'schismmancers'."""
    rng = np.random.default_rng(404)
    t_eye = ev("m04", "eye", CUE["hypermind_wake"], lo=62.5, hi=63.4)
    t_lock = ev("m04", "lock", CUE["hypermeme"], lo=73.9, hi=74.5)
    wings = evs("m04", "wing", 65.5, 71.0)
    if len(wings) != 6:
        wings = list(np.linspace(W("H01", "Six"), 68.35, 6))
    lv = lambda t: lines_gain(t, 0.6, ids=["H01", "H02"])
    # --- the eye: a 32' pedal and a low hum
    I("organ_pedal").note(b, "org", t_eye, "D1", t_lock - t_eye, 0.8, 0.0, gain=0.3, att=1.5, rel=0.3,
                          dyn=[(0, 0.4), (t_lock - t_eye - 1.0, 0.9), (t_lock - t_eye, 1.0)])
    I("organ_softped").note(b, "org", t_eye, "D2", t_lock - t_eye, 0.8, 0.0, gain=0.3, att=1.0, rel=0.3)
    glass_note_(b, t_eye, "D6", 3.0, gain=0.1, decay=2.0)
    hit(t_eye, "hypermind_wake", "32' pedal D + glass: the first eye opens")
    oo = SF2(SF_MS, 0, 53, gain=0.9)
    for n in ["D3", "A3", "D4"]:
        oo.note(t_eye + 0.4, m(n), t_lock - t_eye - 0.3, 0.5)
    y = oo.render()[0]
    if y is not None:
        tt_ = t_eye + 0.4 + np.arange(len(y)) / SR
        e = np.clip((tt_ - t_eye - 0.4) / 2.5, 0, 1) * (0.55 + 0.45 * np.clip((tt_ - 71.6) / 2.4, 0, 1))
        place(b, "choir", y * e[:, None], t_eye + 0.4, gain=0.42)
    # --- wheels within wheels: organ flute ostinati, 3 against 4, accelerating towards the lock
    for (notes, per0, pan) in ((["D4", "A4", "E5"], 1.05, -0.35), (["A3", "E4", "B4", "F#5"], 1.4, 0.35)):
        t, j = t_eye + 0.8, 0
        while t < t_lock - 0.08:
            x = (t - t_eye) / (t_lock - t_eye)
            per = per0 * (1 - 0.55 * x ** 2)
            d = per / len(notes)
            g = (0.35 + 0.65 * x) * float(lv(t))
            I("organ_soft").note(b, "org", t, notes[j % len(notes)], d * 0.92, 0.8, pan, gain=0.1 * g, att=0.02,
                                 rel=0.12, seed=j)
            t += d
            j += 1
    # --- six wings: six rising entries (a rip into each fifth), held and swelling into the lock
    parts = [(("cb", "tuba"), "D2"), (("vc", "tbn"), "A2"), (("va", "hn"), "E3"), (("va", "hn"), "B3"),
             (("v2", "cl"), "F#4"), (("v1", "ob"), "C#6")]  # sixth canon an octave up: no semitone against the Hypermind's D5
    for i, (tw, (who, n)) in enumerate(zip(wings, parts)):
        d = t_lock - tw
        dyn = [(0, 0.3), (0.3, 0.22), (max(0.4, 71.6 - tw), 0.3), (d - 0.5, 0.75), (d, 0.85)]
        for j, dn in enumerate((-5, -2)):
            I("harp").note(b, "keys", tw - 0.12 + 0.06 * j, m(n) + dn + 12, None, 0.5, -0.4 + 0.16 * i, gain=0.16)
        s, br = who
        strings(b, tw, d, dyn=dyn, att=0.08, rel=0.1, gain=0.45, **{s: [n] if s != "cb" else [n]})
        if br in ("tuba", "tbn", "hn"):
            brass(b, tw, d, dyn=[(p, v * 0.8) for p, v in dyn], att=0.12, rel=0.1, gain=0.22, **{br: [n]})
        else:
            key = {"cl": "cl_sus", "ob": "ob_sus"}[br]
            I(key).note(b, "ww", tw, n, d, 0.6, PAN[br], gain=0.35, dyn=dyn, att=0.1, rel=0.1)
        organ(b, tw, d, [n], "flute", gain=0.7, att=0.1, rel=0.1)
        glass_note_(b, tw, m(n) + 24 if m(n) + 24 <= 100 else m(n) + 12, 1.5, gain=0.07, pan=-0.6 + 0.24 * i)
        hit(tw, f"wing_{i + 1}", f"canon {i + 1}: entry on {n}")
    # rising tension after the words: tremolo, timpani roll, choir aahs, cymbal into the lock
    t_h2 = LINES["H02"]["end"]
    strings(b, 71.8, t_lock - 71.8, "trem", dyn=[(0, 0.1), (t_lock - 71.8, 0.7)], att=0.5, rel=0.05,
            v1=["C#6", "E6"], v2=["F#5", "B5"], gain=0.4)
    timp_roll(b, 72.2, t_lock - 72.2, "D2", [(0, 0.1), (t_h2 - 72.2, 0.35), (t_lock - 72.2, 0.95)], gain=0.8)
    swell_to(b, t_lock - 0.01, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.3, after=0.0)
    # --- 74.2 the lock: the whole stack on a full organ + everything; then the colours drain to one plain D
    stack = ["D2", "A2", "D3", "E3", "A3", "B3", "D4", "F#4", "A4", "C#5", "E5"]
    drain = {"E5": 0.9, "C#5": 1.2, "B3": 1.45, "F#4": 1.7, "E3": 1.95, "A4": 2.2, "A3": 2.45, "A2": 2.7}
    for n in stack:
        dd = drain.get(n, 4.6)
        reg = "full" if n in ("D2", "D3", "D4") else "plenum"
        organ(b, t_lock, dd, [n], reg, gain=0.62, att=0.005, rel=0.9 if dd < 4 else 1.4,
              dyn=[(0, 1.0), (0.4, 0.7), (dd, 0.45 if dd < 4 else 0.3)])
    I("organ_pedal").note(b, "org", t_lock, "D1", 4.6, 0.8, 0.0, gain=0.4, att=0.005, rel=1.5)
    brass(b, t_lock, None, "stac", 1.0, tpt=["A4", "C#5", "E5"], hn=["D4", "F#4", "B4"], tbn=["D3", "A3"],
          tuba=["D2"], gain=0.2)
    brass(b, t_lock, 1.2, dyn=[(0, 0.95), (0.35, 0.45), (1.2, 0.2)], att=0.0, rel=0.8, hn=["D4", "A4"],
          tbn=["D3"], gain=0.22)
    strings(b, t_lock, 2.4, dyn=[(0, 1.0), (0.4, 0.5), (2.4, 0.2)], att=0.0, rel=1.2, v1=["D6"], v2=["A5"],
            va=["D4"], vc=["D3"], cb=["D2"], gain=0.6)
    timp(b, t_lock, "D2", 1.0, gain=1.3)
    taiko(b, t_lock, "ff", gain=0.55)
    perc(b, t_lock, PERC + "gongHit_fff.wav", 0.5, 0.0)
    bell_to(b, "bell", t_lock, "D3", gain=0.45, strike=1.0, seed=9, size=1.3)
    ch = SF2(SF_MS, 0, 52, gain=0.9)
    for n in ["D3", "A3", "D4", "E4", "F#4", "A4", "C#5", "D5"]:
        ch.note(t_lock, m(n), 1.4 if n not in ("D3", "D4", "D5") else 3.8, 0.8)
    ch.to(b, "choir", pan=0.0, gain=0.8)
    hit(t_lock, "hypermeme", "the wheels lock: the six canons' stack on full organ + tutti + gong + bell")
    # --- the plain Flag: one plain D (ison) under L01 / C01
    t_d = ev("m04", "descend", t_lock + 0.2, lo=t_lock, hi=t_lock + 1.5)
    ison(b, "ison", t_lock + 1.2, LINES["C01"]["end"] + 1.2, notes=("D3", "D3", "D2", "D3"), gain=0.42, fin=1.8,
         fout=1.2, seed=41)
    I("organ_soft").note(b, "org", t_lock + 2.5, "D4", LINES["C01"]["end"] - t_lock - 1.8, 0.8, 0.0, gain=0.1,
                         att=1.0, rel=1.0)
    tf = W("C01", "Flag")
    for j, n in enumerate(["A5", "G5", "A5"]):  # the leitmotif's head, a wink
        I("harp").note(b, "keys", LINES["C01"]["end"] + 0.1 + 0.28 * j, n, None, 0.4, -0.3, gain=0.18)
    bell_to(b, "bell", tf, "D5", gain=0.08, pan=-0.2, strike=0.5, seed=10, size=0.4)
    # --- the disputation (VIDETVR QVOD NON / SED CONTRA / RESPONDEO)
    t_v = ev("m04", "videtur", CUE["videtur"], lo=80.8, hi=81.6)
    t_s = CUE["sed_contra"]
    t_r = CUE["respondeo"]
    cantus = ["A4", "G4", "A4", "C5", "B4", "C5", "D5"]          # the hymn line, Dorian (C natural)
    counter = ["D4", "E4", "F4", "E4", "G4", "E4", "D4"]          # first species, contrary motion
    tv_ = list(np.linspace(t_v + 0.1, LINES["L02"]["end"] - 0.3, 7))
    for j, n in enumerate(cantus):
        psaltery_to(b, "keys", tv_[j], m(n) + 12, 2.2, 0.55, gain=0.95, pan=0.3)
    hit(t_v, "videtur", "VIDETVR QVOD NON: the psaltery alone (the objection)")
    ts_ = list(np.linspace(t_s + 0.05, W("C02", "fall") - 0.1, 7))
    for j, (n, c) in enumerate(zip(cantus, counter)):
        psaltery_to(b, "keys", ts_[j], m(n) + 12, 2.0, 0.5, gain=0.8, pan=0.3)
        I("harp").note(b, "keys", ts_[j], c, None, 0.55, -0.35, gain=0.48, seed=j)
        I("harp").note(b, "keys", ts_[j], m(c) - 12, None, 0.45, -0.35, gain=0.32, seed=j + 9)
    hit(t_s, "sed_contra", "SED CONTRA: the harp answers in contrary motion")
    # the false coherence: a too-perfect 'smiling world' pad (cheap, glossy) that cracks and falls
    t_cr = ev("m04", "crack", W("C02", "fall"), lo=88.8, hi=90.4)
    a = W("C02", "false") - 0.2
    d = t_cr - a
    pad = np.zeros((S(d + 1.6), 2), np.float32)
    for n in ["D4", "F#4", "A4", "D5"]:
        pad += supersaw(hz(m(n)), d + 1.6, voices=5, spread=0.1, cutoff=2400.0, seed=m(n))
    pad /= np.abs(pad).max() + 1e-9
    ta_ = np.arange(len(pad)) / SR
    rate = np.where(ta_ < d, 1.0, 1.0 - 0.75 * np.clip((ta_ - d) / 1.2, 0, 1) ** 1.3)  # falls off the wall
    fall = vari(pad, rate)
    env = np.clip(ta_ / 0.6, 0, 1) * np.where(ta_ < d, 1.0, np.clip(1 - (ta_ - d) / 1.3, 0, 1))
    fall = crush(fall, 10, 1) * env[:, None]
    place(b, "slop", phone(fall, 250, 9000, 1.1), a, gain=0.16, pan=0.0, width=1.0)
    glass_note_(b, t_cr, "F#6", 1.0, gain=0.08, decay=0.4)
    glass_note_(b, t_cr + 0.03, "G6", 1.0, gain=0.07, decay=0.4)
    hit(t_cr, "coherence_falls", "the too-perfect synth world cracks and falls (tape drop)")
    # 'before the true one can rise': the voices rise to the gold glint
    t_g = ev("m04", "glint", W("C02", "rise"), lo=90.9, hi=92.2)
    run = ["D4", "E4", "F4", "G4", "A4", "C5", "D5", "E5", "F5", "A5"]
    for j, n in enumerate(run):
        tj = W("C02", "before") + (t_g - W("C02", "before")) * j / (len(run) - 1)
        I("harp").note(b, "keys", tj, n, None, 0.45 + 0.03 * j, -0.3 + 0.06 * j, gain=0.2, seed=40 + j)
    psaltery_to(b, "keys", t_g, "D6", 2.5, 0.8, gain=0.55, pan=0.2)
    glass_note_(b, t_g, "A6", 2.5, gain=0.1, decay=1.4)
    I("glock").note(b, "keys", t_g, "D7", None, 0.5, 0.2, gain=0.05)
    hit(t_g, "rise_glint", "harp run rises to the gold glint (psaltery + glass + glock D)")
    # RESPONDEO: 'I answer that:' — the voices close on the octave; the blessing chord
    t_b = ev("m04", "bless", t_r, lo=92.4, hi=93.4)
    t_ans = W("C03", "that")
    psaltery_to(b, "keys", t_b, "C#6", 1.2, 0.6, gain=0.5, pan=0.3)
    I("harp").note(b, "keys", t_b, "E4", None, 0.55, -0.35, gain=0.3)
    psaltery_to(b, "keys", t_ans + 0.25, "D6", 2.5, 0.7, gain=0.55, pan=0.3)
    I("harp").note(b, "keys", t_ans + 0.25, "D4", None, 0.6, -0.35, gain=0.32)
    I("harp").note(b, "keys", t_ans + 0.25, "D3", None, 0.5, -0.35, gain=0.22)
    organ(b, t_b, 1.6, ["D3", "A3", "D4", "F#4"], "flute", pedal="D2", gain=0.7, att=0.35, rel=0.8)
    hit(t_b, "respondeo", "RESPONDEO: the counterpoint closes (sixth -> octave), organ blessing chord")
    # 'summon the schismmancers': the pedal swells to the hard cut
    t_cut = CUE["schismmancers"]
    a = W("C03", "summon")
    I("organ_pedal").note(b, "org", a, "D1", t_cut - a, 0.8, 0.0, gain=0.35, att=0.8, rel=0.02,
                          dyn=[(0, 0.3), (t_cut - a, 1.0)])
    I("organ_pedal").note(b, "org", a + 0.5, "Eb2", t_cut - a - 0.5, 0.8, 0.0, gain=0.22, att=0.8, rel=0.02,
                          dyn=[(0, 0.2), (t_cut - a - 0.5, 1.0)])
    timp_roll(b, a + 0.3, t_cut - a - 0.3, "D2", [(0, 0.1), (t_cut - a - 0.3, 0.9)], gain=0.7)
    strings(b, a + 0.4, t_cut - a - 0.4, "trem", dyn=[(0, 0.05), (t_cut - a - 0.4, 0.8)], att=0.3, rel=0.01,
            vc=["D3", "Eb3"], cb=["D2"], gain=0.5)
    y = rev_clip(np.ascontiguousarray(I("organ_loud").render(m("D5"), 1.4, 0.8)))
    b.add("org", stereo_place(y, 0.0, 1.0), i0=S(t_cut) - len(y), gain=0.12)
    cut_after(b, t_cut, 0.012)
    hit(t_cut, "m04_cut", "hard cut on 'schismmancers'", kind="cut")


# ================================================================================================ m05
def m05(b):
    """The Schismmancers (95.75–106.5): hard cut into a tactical aksak — 7/8 as 2+2+3 (Balkan meters for a
    'recursive balkanization'), taiko and frame drum, spiccato low strings on D with the Phrygian Eb, a loud
    organ pedal; goggles ripple (pizzicati), the lances telescope (brass stabs), the flags unfurl (string
    snaps), the charge (snare roll, rising cluster), BREACH (tutti on a D/Eb cluster) blazing into D major as
    flag-yellow light floods the frame."""
    rng = np.random.default_rng(505)
    t0 = CUE["schismmancers"]
    t_br = ev("m05", "breach", CUE["breach"], lo=105.4, hi=106.1)
    bar = (t_br - t0) / 10.0
    e8 = bar / 7.0
    lv = lambda t: lines_gain(t, 0.7, ids=["K01", "K02", "K03"])
    hit(t0, "schismmancers", "hard cut: taiko + organ pedal + spiccato, 7/8 (2+2+3)")
    I("organ_pedal").note(b, "org", t0, "D1", t_br - t0, 0.8, 0.0, gain=0.34, att=0.005, rel=0.05,
                          dyn=[(0, 0.8), (t_br - t0 - 2.0, 0.85), (t_br - t0, 1.0)])
    I("organ_pedal").note(b, "org", t0, "D2", t_br - t0, 0.8, 0.0, gain=0.22, att=0.005, rel=0.05)
    t_tel = clusters(evs("m05", "telescope", 101.4, 103.8)) or [W("K02", "recursive"), W("K02", "balkanization")]
    t_ch = ev("m05", "charge", CUE["breach_call"], lo=103.5, hi=104.3)
    for k in range(10):
        tb = t0 + k * bar
        x = k / 10
        for q in range(7):
            t = tb + q * e8
            g = float(lv(t))
            acc = q in (0, 2, 4)
            if acc:
                taiko(b, t, "ff" if q == 0 else "f", gain=(0.55 if q == 0 else 0.4) * (0.75 + 0.25 * x), rr=k % 2 + 1)
            elif q == 6 and k % 2 == 1:
                taiko(b, t, "mf", gain=0.25, sticks=True)
            sub = 2 if (t >= t_tel[0] and q >= 4) else 1   # the meter itself starts splitting
            for s in range(sub):
                ts = t + s * e8 / sub
                n = "Eb3" if q == 5 else ("C3" if q == 6 and k % 4 == 3 else "D3")
                I("vc_spic").note(b, "str", ts, n, None, 0.55 + 0.3 * acc, PAN["vc"], gain=(0.32 + 0.12 * acc) * g,
                                  seed=k * 7 + q + s)
                if acc:
                    I("cb_spic").note(b, "str", ts, "D2", None, 0.7, PAN["cb"], gain=0.36 * g, seed=k * 7 + q)
            f = PERC + ("Snare2-taps_v2_rr1_Sum.wav" if q % 2 else "Snare2-taps_v3_rr2_Sum.wav")
            perc(b, t, f, gain=(0.12 + 0.1 * acc) * (0.6 + 0.4 * x) * g, pan=0.25, width=0.5)
    # goggles ripple
    gg_ = clusters(evs("m05", "goggles", 96.3, 99.0), 0.03)
    if len(gg_) < 3:
        t_ = ev("m05", "goggles", W("K01", "Schism"), lo=96.3, hi=98.5)
        gg_ = [t_ + 0.05 * j for j in range(6)]
    for j, (t_, n) in enumerate(zip(gg_, ["D6", "A5", "F5", "D5", "A4", "F4", "D4"])):  # centre outwards
        I("vln_pizz" if m(n) >= 55 else "vla_pizz").note(b, "str", t_, n, None, 0.6, [0, -0.3, 0.3, -0.55, 0.55,
                                                          -0.8, 0.8][j], gain=0.3, seed=j)
    hit(gg_[0], "goggles", "a pizzicato per goggle, rippling out from the centre")
    # the lances telescope (brass stabs), the flags unfurl (string snaps)
    for j, t in enumerate(t_tel[:3]):  # three clacks: each section of the eleven lances locks
        brass(b, t, None, "stac", 0.8, hn=[["D4"], ["D4", "Eb4"], ["D4", "Eb4", "A4"]][j], tbn=["D3", "A2"],
              gain=0.16)
        hit(t, f"telescope_{j + 1}", "brass stab: lances telescope")
    unf = clusters(evs("m05", "unfurl", 101.4, 104.0), 0.03)
    if unf:
        du = ev_to("m05", "unfurl", unf[-1] + 0.35, lo=unf[-1] - 0.01, hi=unf[-1] + 0.01) - unf[0]
        strings(b, unf[0], du, "trem", dyn=[(0, 0.2), (du * 0.8, 0.75), (du, 0.35)], att=0.02, rel=0.25,
                v1=["D5", "A5"], v2=["F5"], gain=0.4)
        for j, t in enumerate(unf[:6]):
            I("vln_pizz").note(b, "str", t, ["D5", "F5", "A5", "D6", "F6", "A6"][j], None, 0.65, -0.5 + 0.2 * j,
                               gain=0.3, seed=j)
        hit(unf[0], "unfurl", "the flags snap open: tremolo swell + a rising pizzicato per flag")
    # the charge: snare roll, rising cluster, a taiko on each 'Breach!' ... then dead calm, the inhale, BREACH
    t_calm = ev("m05", "calm", 105.17, lo=104.8, hi=105.6)
    t_in = ev("m05", "inhale", t_br - 0.25, lo=t_calm, hi=t_br - 0.05)
    d = t_calm - t_ch
    perc(b, t_ch, PERC + "Snare2-rollNS_v5_rr1_Sum.wav", gain=0.25, pan=0.2, dur=d)
    strings(b, t_ch, d, "trem", dyn=[(0, 0.2), (d, 1.0)], att=0.1, rel=0.02, v1=["D5", "Eb5"], v2=["A4", "Bb4"],
            va=["D4", "Eb4"], vc=["D3", "Eb3"], bend=lambda x: 2.0 * (x / max(d, 0.1)) ** 1.5, gain=0.55)
    brass(b, t_ch, d, dyn=[(0, 0.25), (d, 1.0)], att=0.2, rel=0.02, hn=["D4", "Eb4", "A4"], tbn=["D3", "Eb3"],
          tuba=["D2"], bend=lambda x: 2.0 * (x / max(d, 0.1)) ** 1.5, gain=0.3)
    for k in range(3):
        taiko(b, W("K03", "Breach", k), "ff", gain=0.55)
    cut_after(b, t_calm, 0.04)  # dead calm: the flags fall limp, light leaks through the mortar joints
    hit(t_calm, "dead_calm", "everything stops dead; a thread of high harmonic light", kind="cut")
    I("vln_solo").note(b, "str", t_calm + 0.02, "A6", t_in - t_calm + 0.1, 0.4, 0.2,
                       gain=0.12, dyn=[(0, 0.2), (t_in - t_calm, 0.5)], att=0.08, rel=0.05)
    glass_note_(b, t_calm + 0.02, "A6", t_br - t_calm, gain=0.05, decay=0.6, att=0.05)
    # the inhale: everything is sucked back (reversed organ cluster + reversed cymbal) into the blast
    y = rev_clip(np.ascontiguousarray(I("organ_loud").render(m("Eb4"), 1.6, 0.8)))
    y = y[-S(t_br - t_in + 0.12):]
    b.add("org", stereo_place(y, 0.0, 1.0), i0=S(t_br) - len(y), gain=0.3)
    from music_lib import load_wav
    cy, csr = load_wav(os.path.join("assets", "sf2", "vsco2", PERC + "susCymb1-hit_f_rr1.wav"))
    cy = _resample(np.ascontiguousarray(cy, np.float32), 0.0, np.array([csr / SR]), int(len(cy) * SR / csr) - 4)
    rc = rev_clip(cy)[-S(t_br - t_in + 0.05):]
    rc *= np.linspace(0, 1, len(rc))[:, None] ** 2
    b.add("perc", rc, i0=S(t_br) - len(rc), gain=0.35)
    cut_after(b, t_br - 0.003, 0.02)
    # BREACH
    organ(b, t_br, 0.7, ["D3", "Eb3", "A3", "D4", "Eb4", "A4"], "full", pedal="D2", gain=0.8, att=0.003, rel=0.4)
    brass(b, t_br, None, "stac", 1.0, tpt=["D5", "Eb5", "A5"], hn=["D4", "Eb4", "A4"], tbn=["D3", "Eb3"],
          tuba=["D2"], gain=0.2)
    strings(b, t_br, None, "spic", 1.0, v1=["D5", "Eb5"], v2=["A4"], va=["D4", "Eb4"], vc=["D3"], cb=["D2"], gain=0.45)
    timp(b, t_br, "D2", 1.0, gain=1.3)
    taiko(b, t_br, "ff", gain=0.7)
    perc(b, t_br, PERC + "gongHit_fff.wav", 0.45, 0.0)
    crash(b, t_br, "fff", 0.9)
    hit(t_br, "breach", "BREACH: tutti on a D/Eb cluster (organ, brass, strings, timpani, taiko, gong, crash)")
    # the blaze: flag-yellow light floods the frame (D major, rising to the glow and fading into m06)
    t_gl = ev("m05", "glow", CUE["breach"] + 0.7, lo=t_br + 0.2, hi=106.5)
    a = t_br + 0.12
    dd = 108.0 - a
    blaze = [(0, 0.3), (t_gl - a, 0.9), (t_gl - a + 0.4, 0.8), (dd, 0.0)]
    brass(b, a, dd, dyn=blaze, att=0.15, rel=0.3, tpt=["A4", "D5", "F#5"], hn=["D4", "A4"], gain=0.3)
    strings(b, a, dd, "trem", dyn=blaze, att=0.1, rel=0.4, v1=["D6", "F#6"], v2=["A5", "D6"], va=["F#5"], gain=0.4)
    organ(b, a, dd, ["D3", "A3", "D4", "F#4", "A4"], "plenum", pedal="D2", gain=0.5, att=0.2, rel=0.6,
          dyn=[(p, v) for p, v in blaze])
    ch = SF2(SF_MS, 0, 52, gain=0.8)
    for n in ["D4", "F#4", "A4", "D5", "F#5"]:
        ch.note(a, m(n), dd, 0.75)
    y = ch.render()[0]
    if y is not None:
        e = envelope([(p, v) for p, v in blaze] + [(len(y) / SR, 0.0)], len(y))
        place(b, "choir", y * e[:, None], a, gain=0.6)
    glass_note_(b, t_gl, "D7", 2.0, gain=0.1, decay=1.2)
    hit(t_gl, "glow", "the blaze peaks as flag-yellow light fills the frame")


# ================================================================================================ m06
SUBJ = [(0, 2), (-2, 2), (0, 2), (3, 2), (2, 2), (3, 2), (5, 4)]   # the hymn line (Dorian) from A: 16 beats
SUBJ_CB = np.cumsum([0] + [d for _, d in SUBJ])
INST = {  # sampler key, bus, lo, hi, gain, sustained?
    "hn": ("hn_sus", "brs", 41, 72, 0.3, True), "tpt": ("tpt_sus", "brs", 58, 80, 0.24, True),
    "tbn": ("tbn_sus", "brs", 40, 62, 0.3, True), "tuba": ("tuba_sus", "brs", 33, 55, 0.32, True),
    "cl": ("cl_sus", "ww", 52, 84, 0.34, True), "ob": ("ob_sus", "ww", 60, 86, 0.3, True),
    "fl": ("fl_sus", "ww", 62, 93, 0.34, True), "bsn": ("bsn_sus", "ww", 36, 64, 0.36, True),
    "vln": ("vln_solo", "str", 60, 93, 0.24, True), "vc": ("vc_sus", "str", 40, 72, 0.3, True),
    "va": ("vla_sus", "str", 50, 79, 0.3, True), "org": ("organ_soft", "org", 48, 84, 0.13, True),
    "orgL": ("organ_loud", "org", 40, 79, 0.1, True), "glock": ("glock", "keys", 79, 105, 0.05, False),
    "harp": ("harp", "keys", 48, 98, 0.17, False), "pizz": ("vln_pizz", "str", 55, 88, 0.22, False),
    "mar": ("marimba", "keys", 50, 96, 0.08, False), "xylo": ("xylo", "keys", 67, 105, 0.045, False),
    "vpizz": ("vla_pizz", "str", 48, 84, 0.22, False), "cel": ("cel", "keys", 72, 106, 1.0, False)}


def fold(n, lo, hi):
    n = float(n)
    while n < lo:
        n += 12
    while n > hi:
        n -= 12
    return n


class Voice:
    def __init__(self, tb, pos, f, tr, inst, pan, gen=0, parent_pitch=None, glide=0.25, beat=0.25, gain=1.0):
        self.tb, self.pos, self.f, self.tr, self.inst, self.pan, self.gen = tb, pos, f, tr, inst, pan, gen
        self.parent_pitch, self.glide, self.beat, self.gain = parent_pitch, glide, beat, gain

    def p(self, t):  # position in beats
        return self.pos + (t - self.tb) * self.f / self.beat

    def t_of(self, p):
        return self.tb + (p - self.pos) * self.beat / self.f

    def pitch(self, p, base=69):
        k = int(np.searchsorted(SUBJ_CB, p % 16, "right") - 1)
        lo, hi = INST[self.inst][2], INST[self.inst][3]
        return fold(base + SUBJ[k][0] + self.tr, lo, hi)


def play_voice(b, v, t_end, lv, nvoices, cel, dust=1.0, rng=None):
    """render voice v from its birth until t_end (a fork or the section end)."""
    key, bus, lo, hi, g0, sus = INST[v.inst]
    p = v.p(v.tb)
    first = True
    while True:
        loop, r = divmod(p, 16)
        k = int(np.searchsorted(SUBJ_CB, r, "right") - 1)
        s_abs = loop * 16 + SUBJ_CB[k]
        e_abs = loop * 16 + SUBJ_CB[k + 1]
        ta = max(v.t_of(s_abs), v.tb)
        tz = min(v.t_of(e_abs), t_end)
        if ta >= t_end - 0.01:
            break
        n = v.pitch(s_abs + 0.001)
        g = g0 * v.gain * float(lv(ta)) / max(1.0, nvoices ** 0.5)
        cont = first and v.t_of(s_abs) < v.tb - 0.01   # the note sounding at the fork continues, gliding
        if v.inst == "cel":
            cel.note(ta, int(round(min(n, 106))), 0.3 * dust, 0.35)
        elif not sus:
            I(key).note(b, bus, ta, int(round(n)), None, 0.45 + 0.3 * rng.random(), v.pan, gain=g * dust,
                        seed=int(ta * 997) % 10007)
        else:
            dur = tz - ta
            bend = None
            if cont and v.parent_pitch is not None and abs(v.parent_pitch - n) > 0.01:
                dp, gl = v.parent_pitch - n, v.glide
                bend = lambda x, dp=dp, gl=gl: dp * np.clip(1 - x / gl, 0, 1) ** 1.5
            elif abs(n - round(n)) > 0.01:
                dp = n - round(n)
                bend = lambda x, dp=dp: dp + 0 * x
            I(key).note(b, bus, ta, int(round(n)), max(0.06, dur), 0.6, v.pan, gain=g, att=0.03 if cont else 0.04,
                        rel=0.12 if tz >= t_end - 1e-3 else 0.18, bend=bend, offset=0.25 if cont else 0.0,
                        align=not cont, seed=int(ta * 991) % 10007)
        first = False
        p = e_abs + 1e-6
        if tz >= t_end - 1e-3:
            break


def fork_canon(b, t0, t1, voices, forks, lv, cel, rng, dust=1.0, converge=None):
    """run the forking canon: forks = [(t, rule)] where rule(voice, rng) -> list of child Voices born at t."""
    edges = [t for t, _ in forks if t0 < t < t1]
    rules = [r for t, r in forks if t0 < t < t1]
    cur = voices
    seg = [t0] + edges + [t1]
    for i in range(len(seg) - 1):
        a, z = seg[i], seg[i + 1]
        for v in cur:
            play_voice(b, v, z, lv, len(cur), cel, dust, rng)
        if i < len(rules):
            nxt = []
            for v in cur:
                nxt += rules[i](v, z, rng)
            cur = nxt
    return cur


def split_rule(interval, tempo, insts, glide=0.3, spread=0.35, cap=48):
    def rule(v, t, rng):
        pp = v.pitch(v.p(t))
        kids = []
        for s, (dtr, ff) in enumerate(((0.0, 1.0), (interval, tempo))):
            inst = v.inst if s == 0 else insts[(insts.index(v.inst) + 1 + v.gen) % len(insts)] if v.inst in insts \
                else insts[0]
            kids.append(Voice(t, v.p(t), v.f * ff, v.tr + dtr, inst, float(np.clip(v.pan + (s - 0.5) * spread *
                              (0.8 ** v.gen) * 2, -0.95, 0.95)), v.gen + 1, pp, glide, v.beat, v.gain))
        return kids
    return rule


def quad_rule(insts, cap=200):
    def rule(v, t, rng):
        pp = v.pitch(v.p(t))
        kids = []
        for s in range(4):
            dtr = float(rng.choice([0, 2, 5, 7, 12, -5, -7, 10])) + float(rng.normal(0, 0.3))
            kids.append(Voice(t, v.p(t) + float(rng.uniform(0, 4)), v.f * float(rng.uniform(0.85, 1.45)),
                              v.tr + dtr, insts[int(rng.integers(len(insts)))],
                              float(np.clip(v.pan + rng.normal(0, 0.4), -0.95, 0.95)), v.gen + 1, pp, 0.2,
                              v.beat * 0.8, v.gain))
        return kids
    return rule


def m06(b):
    """Recursive Balkanization (106.5–131.5): the hymn line as a canon that schisms. Every crack in the picture
    forks every voice in two: one keeps its road, the other slides away into a new key and a new tempo, so
    one anthem becomes 2, 4, 8, 16 nations (brass & winds: fifth, tritone, semitone, minor third); the dive
    into the congregation pulls them back into one organ voice that splits into churches of one (anathema
    stabs, a papal fanfare in the wrong key); the philosophers' voices drift apart in quarter-tones on
    floating rocks; 'May it schism endlessly': every voice splits into four, and four, and four — dust."""
    rng = np.random.default_rng(606)
    cel = cel_part(1.3)
    # -------- nations: 1 -> 16 anthems (brass/winds)
    t_n = CUE["nations"]
    dive1 = ev("m06", "dive", 113.1, lo=112.5, hi=113.7)
    dive2 = ev("m06", "dive", 119.3, lo=118.6, hi=120.0)
    sp = evs("m06", "split", t_n + 0.2, dive1 - 0.3, contains="nations") or [107.75, 108.8, 110.2, 111.7]
    lvn = lambda t: lines_gain(t, 0.55, ids=["K04", "X01", "X02"]) * ramp(t, [(t_n, 0.9), (dive1, 1.1)])
    nat = ["hn", "tpt", "cl", "tbn", "ob", "fl", "bsn", "tuba", "vln", "hn", "tpt", "cl", "vc", "ob", "fl", "bsn"]
    intervals = [(7, 1.12), (6, 0.9), (1, 1.07), (3, 0.94), (10, 1.05)]
    forks = [(t, split_rule(intervals[min(i, 4)][0], intervals[min(i, 4)][1], nat)) for i, t in enumerate(sp[:5])]
    v0 = [Voice(t_n, 0.0, 1.0, -12, "hn", 0.0, beat=0.26, gain=2.2)]
    I("hn_sus")  # warm
    last = fork_canon(b, t_n, dive1, v0, forks, lvn, cel, rng)
    hit(t_n, "nations", "the canon's first voice (horn): one anthem")
    for i, t in enumerate(sp[:5]):
        hit(t, f"split_nations_{i + 1}", f"every voice forks (generation {i + 1})")
    dive(b, last, dive1, CUE["faiths"], "D4", cel)
    # -------- faiths: churches of one (organ + strings), anathemas, a pope of one
    t_f = CUE["faiths"]
    spf = evs("m06", "split", t_f + 0.2, dive2 - 0.3, contains="faiths") or [114.5, 115.3, 116.3, 117.4]
    lvf = lambda t: lines_gain(t, 0.55, ids=["K05", "X03", "X04", "X05"])
    fai = ["orgL", "va", "org", "vc", "hn", "orgL", "cl", "va", "org", "bsn", "vc", "orgL"]
    ints = [(5, 0.92), (-6, 1.1), (1, 0.95), (4, 1.08), (-1, 1.0)]
    forks = [(t, split_rule(ints[min(i, 4)][0], ints[min(i, 4)][1], fai, glide=0.35)) for i, t in enumerate(spf[:5])]
    v0 = [Voice(t_f, 0.0, 1.0, -12, "orgL", 0.0, beat=0.3, gain=3.2)]
    last = fork_canon(b, t_f, dive2, v0, forks, lvf, cel, rng)
    hit(t_f, "faiths", "one organ voice: the chorale that will split into churches of one")
    an = evs("m06", "anathema", 115.3, 117.4) or [W("X03", "Anathema"), W("X04", "anathema")]
    for j, t in enumerate(an[:2]):
        cl = [["C#3", "D3", "G#3", "D4"], ["G2", "G#3", "D4", "C#5"]][j]
        organ(b, t, 0.45, cl, "full", pedal=["D2", "G#1"][j], gain=0.55, att=0.01, rel=0.5)
        brass(b, t, None, "stac", 0.9, tbn=[cl[0], cl[1]], gain=0.15)
        hit(t, f"anathema_{j + 1}", "ANATHEMA: dissonant organ stab")
    tp = LINES["X05"]["end"] + 0.05
    for j, n in enumerate(["Bb5", "Eb6", "G6", "Bb6"]):
        brass(b, tp + 0.07 * j, None, "stac", 0.7, tpt=[n] if m(n) <= 84 else [m(n) - 12], gain=0.14)
    I("glock").note(b, "keys", tp + 0.21, "Eb7", None, 0.6, 0.4, gain=0.05)
    hit(tp, "own_pope", "a papal fanfare of one, in the wrong key (Eb)")
    dive(b, last, dive2, CUE["philosophies"], "A4", cel)
    # -------- philosophies: voices drift apart in quarter-tones (each alone on a floating rock)
    t_p = CUE["philosophies"]
    t_e = CUE["endless"]
    spp = evs("m06", "split", t_p + 0.2, t_e - 0.2, contains="philo") or [121.0, 122.6, 124.4, 126.2, 127.6]
    lvp = lambda t: lines_gain(t, 0.5, ids=["K06", "X06", "X07"])
    phi = ["fl", "vln", "cl", "ob", "vc", "bsn", "hn", "va"]
    ints = [(0.5, 0.86), (-1.5, 1.13), (3.5, 0.9), (-0.5, 1.2), (6.5, 0.8)]
    forks = [(t, split_rule(ints[min(i, 4)][0], ints[min(i, 4)][1], phi, glide=1.1, spread=0.45))
             for i, t in enumerate(spp[:5])]
    v0 = [Voice(t_p, 0.0, 1.0, 12, "fl", 0.0, beat=0.36, gain=3.0)]
    last = fork_canon(b, t_p, t_e, v0, forks, lvp, cel, rng)
    hit(t_p, "philosophies", "a flute alone; voices drift apart in quarter tones")
    # -------- endless: quadtree, every voice splits into four -> dust (into m07's spiral)
    spe = evs("m06", "split", t_e + 0.1, 131.4, contains="endless") or [128.95, 129.55, 130.1, 130.6]
    dust = ["cel", "glock", "harp", "pizz", "mar", "xylo", "vpizz", "cel", "harp"]
    keep = [v for v in last]
    rng.shuffle(keep)
    for v in keep[:6]:
        v.inst = dust[int(rng.integers(len(dust)))]
    voices = keep[:6]
    for v in voices:
        v.tb, v.pos = t_e, v.p(t_e)
    lve = lambda t: lines_gain(t, 0.6, ids=["K07"]) * ramp(t, [(t_e, 0.8), (130.5, 1.2), (131.5, 1.0), (132.6, 0.0)])
    forks = [(t, quad_rule(dust)) for t in spe[:4]]
    t_end = 132.6
    cur = voices
    seg = [t_e] + [t for t, _ in forks] + [t_end]
    for i in range(len(seg) - 1):
        a, z = seg[i], seg[i + 1]
        if len(cur) > 180:
            idx = rng.choice(len(cur), 180, replace=False)
            cur = [cur[j] for j in idx]
        for v in cur:
            play_voice(b, v, z, lve, max(1, len(cur) / 3), cel, dust=0.9, rng=rng)
        if i < len(forks):
            nxt = []
            for v in cur:
                nxt += forks[i][1](v, z, rng)
            cur = nxt
    for i, t in enumerate(spe[:4]):
        hit(t, f"quadtree_{i + 1}", "every voice splits into four")
    hit(t_e, "endless", "'May it schism endlessly': quadtree forks to dust")
    cel.to(b, "keys", pan=0.0, gain=0.85)


def dive(b, voices, t0, t1, target, cel):
    """the dive into one tessera: every voice slides to one pitch and swells, cut at t1."""
    tgt = m(target)
    d = t1 - t0
    seen = set()
    for v in voices[:24]:
        key, bus, lo, hi, g0, sus = INST[v.inst]
        if not sus or v.inst in seen and len(seen) > 6:
            continue
        seen.add(v.inst)
        n0 = v.pitch(v.p(t0))
        nt = fold(tgt, lo, hi)
        I(key).note(b, bus, t0, int(round(nt)), d, 0.6, v.pan * (1 - 0.5), gain=g0 * 0.5,
                    dyn=[(0, 0.35), (d * 0.8, 0.7), (d, 0.8)], att=0.05, rel=0.03,
                    bend=lambda x, dp=n0 - nt, d=d: dp * np.clip(1 - x / (0.8 * d), 0, 1) ** 2)
    swell_to(b, t1 - 0.01, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.18, after=0.0)
    y = rev_clip(np.ascontiguousarray(I("harp").render(tgt, None, 0.6)))
    b.add("keys", stereo_place(y, 0.0, 1.0), i0=S(t1) - len(y), gain=0.2)
    hit(t1, f"dive_{target}", "the dive into one tessera: all voices converge, cut")


# ================================================================================================ m07
BABEL = ["D2", "G#2", "C#3", "G3", "C4", "F#4", "B4"]          # one pipe per tier: tritone, fourth, tritone ...
BABEL_MORE = ["D#4", "E5", "A5", "A#3", "F5", "D#6", "E3"]     # every other language joins


def m07(b):
    """Babel (131.5–146.0): the dust spirals up; the tower rises one organ pipe per tier (tritones and fourths:
    every tier another language) until all twelve tones sound; the cluster rises like an endless Shepard tone
    while the workers' aksak runs in 9/8; Lord Egregore enters as a reversed hymn — an A-men sucked backwards —
    and a tritone pedal; under 'Good. Now they are ready.' a single voice holds a plain D. 141.8: the organ's
    wind fails — every pipe sags and dies with the blower — and the whole mosaic falls as an avalanche of
    struck notes; 145.0: digital silence."""
    rng = np.random.default_rng(707)
    t0 = CUE["babel_rise"]
    t_c = ev("m07", "topple", CUE["collapse"], lo=141.4, hi=142.2)
    tiers = evs("m07", "tier", t0, 134.4)
    if len(tiers) < 7:
        tiers = list(np.linspace(131.9, 134.0, 7))
    tiers = tiers[:7]
    t_eg = ev("m07", "egregore", CUE["egregore"], lo=133.6, hi=134.6)
    lv = lambda t: lines_gain(t, 0.72, ids=["L03"]) * lines_gain(t, 0.42, ids=["C04"])
    rise = lambda x, T: 2.5 * (np.clip(x / T, 0, 1)) ** 1.6       # the whole cluster climbs 2.5 semitones
    # the dust spirals up into the tower (celesta/glock/harp arpeggios, accelerating, rising)
    cel = cel_part(1.3)
    t, k = t0, 0
    while t < tiers[-1]:
        x = (t - t0) / (tiers[-1] - t0)
        n = 72 + int((k * 5) % 24) + int(12 * x)
        if k % 3 == 0:
            cel.note(t, min(n, 104), 0.25, 0.3)
        elif k % 3 == 1:
            I("glock").note(b, "keys", t, max(79, min(n + 7, 105)), None, 0.35, np.sin(k), gain=0.035)
        else:
            I("harp").note(b, "keys", t, min(n - 12, 96), None, 0.4, np.cos(k), gain=0.13)
        t += 0.11 * (1 - 0.6 * x)
        k += 1
    # the tower: one organ pipe per tier (+ strings & brass), rising towards the collapse
    for i, (tt_, n) in enumerate(zip(tiers + list(np.linspace(t_eg + 0.3, 136.2, len(BABEL_MORE))),
                                     BABEL + BABEL_MORE)):
        d = t_c - tt_
        T = t_c - tt_
        bend = lambda x, T=T, off=tt_ - t0: rise(x + off, t_c - t0) - rise(off, t_c - t0) + rise(off, t_c - t0)
        dyn = [(0, 0.55), (max(0.1, d - 2.0), 0.8), (d, 1.0)]
        I("organ_loud").note(b, "org", tt_, n, d, 0.8, -0.8 + 1.6 * ((i * 5) % 14) / 13, gain=0.12, att=0.01,
                             rel=0.05, bend=bend, dyn=dyn, seed=i)
        if i < 7:
            hit(tt_, f"tier_{i + 1}", f"tier {i + 1}: organ pipe {n}")
            if m(n) < 50:
                I("organ_pedal").note(b, "org", tt_, n, d, 0.8, 0.0, gain=0.25, att=0.01, rel=0.05, bend=bend, dyn=dyn)
        sec = "vc" if m(n) < 55 else ("va" if m(n) < 64 else ("v2" if m(n) < 72 else "v1"))
        strings(b, tt_, d, "trem", dyn=[(0, 0.35), (d, 0.9)], att=0.15, rel=0.05, bend=bend, gain=0.32,
                **{sec: [n]})
    brass(b, t_eg, t_c - t_eg, dyn=[(0, 0.3), (t_c - t_eg, 0.95)], att=0.3, rel=0.05, hn=["D4", "G#4", "C#5"],
          tbn=["D3", "G#3"], tuba=["D2"], bend=lambda x: rise(x + t_eg - t0, t_c - t0), gain=0.26)
    # the workers' aksak (9/8 = 2+2+2+3), accelerating
    t, q = t0 + 0.2, 0
    while t < t_c - 0.05:
        x = (t - t0) / (t_c - t0)
        e8 = 0.15 * (1 - 0.3 * x)
        pos = q % 9
        acc = pos in (0, 2, 4, 6)
        if acc:
            taiko(b, t, "f" if pos else "ff", gain=(0.3 + 0.25 * x), rr=q % 2 + 1)
        I("cb_spic" if acc else "vc_spic").note(b, "str", t, "D2" if acc else "D3", None, 0.6, PAN["vc"],
                                                  gain=(0.25 + 0.15 * x), seed=q)
        t += e8
        q += 1
    # Lord Egregore: a hymn sucked backwards (reversed A-men), a tritone pedal, the choir cluster
    ch = np.zeros((S(2.2), 2), np.float32)
    for n in ["G3", "B3", "D4", "G4"]:
        ch[:S(1.0)] += stereo_place(I("organ_loud").render(m(n), 0.9, 0.8)[:S(1.0)], 0, 1)
    for n in ["F#3", "A3", "D4", "F#4"]:
        yy = I("organ_loud").render(m(n), 1.0, 0.8)
        ch[S(1.0):S(1.0) + len(yy)] += stereo_place(yy, 0, 1)[:S(2.2) - S(1.0)]
    rch = rev_clip(ch)
    b.add("org", rch, i0=S(t_eg) - len(rch), gain=0.2)
    I("organ_pedal").note(b, "org", t_eg, "G#1", t_c - t_eg, 0.8, 0.0, gain=0.3, att=0.01, rel=0.05)
    timp(b, t_eg, "G#2" if m("G#2") <= 55 else "D2", 0.9, gain=1.0)
    perc(b, t_eg, PERC + "gongHit_f.wav", 0.4, 0.0)
    cho = SF2(SF_MS, 0, 52, gain=0.8)
    for n in ["D3", "G#3", "C#4", "G4", "C5", "F#5"]:
        cho.note(t_eg, m(n), t_c - t_eg, 0.7)
    y = cho.render()[0]
    if y is not None:
        tt_ = t_eg + np.arange(len(y)) / SR
        e = np.clip((tt_ - t_eg) / 1.5, 0, 1) * np.clip((t_c + 0.05 - tt_) / 0.1, 0, 1)
        place(b, "choir", y * e[:, None], t_eg, gain=0.5)
    hit(t_eg, "egregore", "Egregore: a reversed A-men sucked in, tritone pedal, gong, choir cluster")
    # one crescendo for the whole tower (org/str/brs/choir/perc), dipping under Lutie and further under Crocus
    envf = lambda t: (ramp(t, [(t0, 0.2), (t_eg, 0.38), (137.0, 0.58), (140.0, 0.82), (t_c, 1.0)])
                      * lines_gain(t, 0.88, ids=["L03"]) * lines_gain(t, 0.62, ids=["C04"]))
    a_, z_ = S(t0), S(t_c + 0.01)
    ee = envf(np.arange(a_, z_) / SR).astype(np.float32)[:, None]
    for kb in ("org", "str", "brs", "choir", "perc"):
        if kb in b.b:
            b.b[kb][a_:z_] *= ee
    # under 'Good. Now they are ready.': one voice holds a plain D (the ison, alone)
    a = LINES["C04"]["start"] - 0.4
    one = up48(sing_vowel("nˈɔː", "am_adam", t_c - a + 0.3, 50, 50, seed=7070))
    on = int(np.argmax(np.abs(one) > np.abs(one).max() * 0.05))
    one = one[on:]
    na = S(0.5)
    one[:na] *= np.linspace(0, 1, na) ** 2
    one = one[:S(t_c - a)]
    one[-S(0.05):] *= np.linspace(1, 0, S(0.05))
    one *= 0.1 / (np.sqrt(np.mean(one[S(0.5):] ** 2)) + 1e-6)
    place(b, "ison", np.stack([one, one], 1), a, gain=0.55, pan=-0.1)
    # 141.8 the collapse: the wind fails (pitch sag, blower winds down), brass & strings fall, the avalanche
    cut_after(b, t_c + 0.002, 0.01)
    hit(t_c, "collapse", "the organ's wind fails: every pipe sags; the avalanche of struck notes")
    for i, n in enumerate(BABEL + BABEL_MORE):
        n2 = m(n) + 2.5
        sag = lambda x: -6.0 * np.clip(x / 2.4, 0, 1) ** 1.4 - 0.5 * np.sin(2 * np.pi * 5.5 * x) * np.clip(x / 2.4, 0, 1)
        I("organ_loud").note(b, "org", t_c, int(np.floor(n2)), 2.6, 0.8, -0.8 + 1.6 * ((i * 5) % 14) / 13, gain=0.12,
                             att=0.003, rel=0.4, bend=lambda x, fr=n2 - np.floor(n2): fr + sag(x),
                             dyn=[(0, 1.0), (0.6, 0.65), (2.6, 0.0)], offset=0.3, align=False, seed=i)
    from music_lib import load_wav
    bl, sr = load_wav(os.path.join("assets", "sf2", "vsco2", "Keys", "Organ", "Rode_Blower.wav"))
    seg = bl[: int(3.5 * sr)]
    n3 = S(2.8)
    rate = (sr / SR) * np.clip(1 - np.linspace(0, 1, n3) ** 0.8, 0.05, 1)
    wind = _resample(np.ascontiguousarray(seg, np.float32), 0.0, rate.astype(np.float64), n3)
    wind *= np.linspace(1, 0, n3)[:, None] ** 1.5
    b.add("org", wind, t=t_c, gain=0.6)
    brass(b, t_c, 2.2, dyn=[(0, 1.0), (2.2, 0.0)], att=0.0, rel=0.2, tbn=["D3", "G#3"], hn=["C#4", "G4"],
          tuba=["D2"], bend=lambda x: -9.0 * np.clip(x / 2.0, 0, 1) ** 1.3, gain=0.35)
    strings(b, t_c, 2.6, dyn=[(0, 0.9), (2.6, 0.0)], att=0.0, rel=0.2, v1=["F#6", "B5"], v2=["C5"], va=["G4"],
            vc=["C#3"], bend=lambda x: -14.0 * np.clip(x / 2.6, 0, 1) ** 1.2, gain=0.45)
    timp(b, t_c, "D2", 1.0, gain=1.2)
    taiko(b, t_c, "ff", gain=0.6)
    crash(b, t_c, "fff", 0.8)
    # the avalanche: tens of thousands of tesserae — falling cascades of struck notes, thinning to nothing
    t_av0 = ev("m07", "avalanche", t_c + 0.3, lo=t_c, hi=143.0)
    t_last = ev("m07", "last_tile", 144.55, lo=143.8, hi=144.95)
    t_av1 = t_last - 0.35  # the avalanche thins out; then the very last tessera lands, alone
    t_imp = ev("m07", "tower_impact", 143.4, lo=142.5, hi=144.3)
    timp(b, t_imp, "D2", 0.9, gain=0.9)
    perc(b, t_imp, PERC + "gongHit_f.wav", 0.35, 0.1)
    I("organ_pedal").note(b, "org", t_imp, "D1", 1.2, 0.8, 0.0, gain=0.25, att=0.005, rel=0.8,
                          bend=lambda x: -1.5 * np.clip(x / 1.2, 0, 1))
    hit(t_imp, "tower_impact", "the tower's top hits the floor: timpani + gong + a sagging 32' D")
    I("glock").note(b, "keys", t_last, "D7", None, 0.35, -0.2, gain=0.05)
    cel.note(t_last, m("D7"), 0.4, 0.3)
    hit(t_last, "last_tile", "the very last tessera lands, alone (one high D)")
    t, k = t_av0, 0
    kinds = ["glock", "xylo", "cel", "harp", "mar", "pizz", "piano"]
    while t < t_av1:
        x = (t - t_av0) / (t_av1 - t_av0)
        dens = 45 * np.exp(-2.2 * x) + 6
        casc = k % 11
        n = int(100 - 4.2 * casc - 30 * x + rng.integers(-3, 4))
        kd = kinds[int(rng.integers(len(kinds)))]
        g = (1 - 0.7 * x) * (0.6 + 0.4 * rng.random())
        p = float(rng.uniform(-0.9, 0.9))
        if kd == "cel":
            cel.note(t, int(np.clip(n, 60, 104)), 0.15, 0.3 * g + 0.1)
        elif kd == "piano":
            I("piano").note(b, "keys", t, int(np.clip(n - 12, 36, 100)), None, 0.3 + 0.3 * g, p, gain=0.08 * g,
                            seed=k)
        else:
            key, bus, lo, hi, g0, _ = INST[{"glock": "glock", "xylo": "xylo", "harp": "harp", "mar": "mar",
                                             "pizz": "pizz"}[kd]]
            I(key).note(b, bus, t, int(np.clip(n, lo, hi)), None, 0.4 + 0.4 * g, p, gain=g0 * g, seed=k)
        t += float(rng.exponential(1.0 / dens))
        k += 1
    cel.to(b, "keys", pan=0.0, gain=0.8)
    cut_after(b, SILENCE[0], 0.12)
    hit(SILENCE[0], "silence", "HARD STOP: digital silence 145.0–146.5", kind="cut")


# ================================================================================================ m08
CONV = ["D2", "A2", "D3", "A3", "D4", "F#4", "A4", "D5"]   # the Vocalist's convergence chord (159.0)


def m08(b):
    """The Unbabeling (146.0–163.0): out of the silence one voice holds one tone (D) — no words, no language;
    when the Flag is planted a second voice and the 16' pedal join; on 'Flag' the candle's bell, far away;
    the fallen tesserae rise as reversed notes whose attacks land where the tiles set, each one alone, then
    all of them together, and everything converges on one chord at 159.0 (the choir's D major)."""
    rng = np.random.default_rng(808)
    t_one = SILENCE[1] + 0.35
    t_pl = ev("m08", "plant", CUE["flag_planted"], lo=148.4, hi=149.6)
    t_rise = ev("m08", "rise", CUE["tiles_rise"], lo=154.5, hi=155.6)
    t_cv = ev("m08", "complete", CUE["convergence"], lo=158.6, hi=159.4)
    # one voice, alone (breathes once); then two; then all of them
    def solo(v, note, a, z, seed, g=0.5, pan=0.0, fin=1.4):
        mi = m(note)
        y = up48(sing_vowel("nˈɔː", v, z - a + 0.4, mi, mi if mi >= 47 else mi + 12, seed=seed))
        on = int(np.argmax(np.abs(y) > np.abs(y).max() * 0.05))
        y = y[on:on + S(z - a)]
        na, nr = S(fin), S(0.8)
        y[:na] *= np.linspace(0, 1, na) ** 2
        y[-nr:] *= np.cos(np.linspace(0, np.pi / 2, nr)) ** 2
        y *= 0.1 / (np.sqrt(np.mean(y[na:len(y) - nr] ** 2)) + 1e-6)
        place(b, "ison", np.stack([y, y], 1), a, gain=g, pan=pan)

    solo("am_adam", "D3", t_one, 153.6, 8001, 0.27, -0.05, fin=2.2)
    hit(t_one, "one_tone", "from nothing: one voice holds one tone (D3)")
    solo("bm_daniel", "D2", t_pl, 156.2, 8002, 0.22, 0.15, fin=0.9)
    solo("am_adam", "D3", 153.9, t_cv + 0.2, 8003, 0.3, -0.05, fin=0.8)
    I("organ_softped").note(b, "org", t_pl, "D2", t_cv - t_pl, 0.8, 0.0, gain=0.15, att=0.4, rel=0.3)
    # the heaps tremble (153.4 -> 155.0): a glassy shimmer gathering
    t_tr = ev("m08", "tremble", 153.4, lo=152.5, hi=154.8)
    for j, n in enumerate(["D6", "A6", "E6", "F#6", "D7", "A5"]):
        glass_note_(b, t_tr + 0.25 * j, n, t_rise - t_tr + 1.0, gain=0.035 + 0.01 * j, pan=-0.6 + 0.24 * j,
                    decay=(t_rise - t_tr) * 1.2, att=(t_rise - t_tr) * 0.8)
    I("harp").note(b, "keys", t_pl, "D2", None, 0.55, -0.2, gain=0.26)
    I("harp").note(b, "keys", t_pl + 0.02, "D3", None, 0.45, -0.2, gain=0.2)
    hit(t_pl, "flag_planted", "a second voice (D2), the 16' pedal and a low harp D as the Flag is planted")
    tf = W("C05", "Flag", 1)
    bell_to(b, "bell", tf, "D3", gain=0.2, pan=0.2, strike=0.6, seed=12, size=1.15)
    hit(tf, "flag_bell", "the candle's bell, far away, on 'Flag'")
    # the other voices join as the tiles rise: each alone, then together
    t_al, t_tg = W("N08", "alone"), W("N08", "together")
    joins = [("am_onyx", "D3", t_rise + 0.3, 0.8), ("am_liam", "A2", t_al, 0.7), ("bm_lewis", "D3", t_al + 0.5, 0.7),
             ("am_echo", "A3", W("N08", "All"), 0.6), ("am_eric", "D2", t_tg, 0.6), ("am_puck", "D3", t_tg + 0.15, 0.6)]
    for j, (v, n, a, fin) in enumerate(joins):
        solo(v, n, a, 162.2 + 0.2 * j, 8010 + j, 0.24, -0.7 + 0.28 * j, fin=fin)
    # the rising tesserae: reversed struck notes whose attacks land where tiles set. The density follows
    # the picture's own set-fraction (M08 'land' events: fraction of the new picture set).
    lev = [e for e in _events("m08", "land") if t_rise <= e["t"] <= t_cv + 0.01]
    if len(lev) >= 3:
        ft = [t_rise] + [e["t"] for e in lev]
        fv = [0.0] + [float(e["strength"]) for e in lev]
    else:
        ft = [t_rise, W("N08", "alone"), W("N08", "All"), t_cv]
        fv = [0.0, 0.1, 0.3, 1.0]
    total = 150
    u = np.sort(rng.uniform(0.02, 0.995, total))
    lands = list(np.interp(u, np.maximum.accumulate(np.array(fv) + 1e-6 * np.arange(len(fv))), ft))
    lands = [t for t in lands if t_rise + 0.2 < t < t_cv - 0.02]
    lands += list(np.linspace(t_rise + 0.3, ft[1] if len(ft) > 1 else t_rise + 1.5, 6))  # the first few, alone
    lands.sort()
    pcs = [2, 6, 9, 4]
    for j, tl_ in enumerate(lands):
        x = (tl_ - t_rise) / (t_cv - t_rise)
        oc = int(4 + 3 * x + rng.integers(-1, 2))
        n = 12 * (oc + 1) + pcs[j % 4 if x < 0.6 else int(rng.integers(4))]
        kd = ["harp", "glock", "cel", "pizz", "bell"][j % 5 if x > 0.25 else j % 2]
        g = 0.5 + 0.5 * x
        p = float(np.clip(rng.normal(0, 0.25 + 0.6 * x), -0.95, 0.95))
        if kd == "bell":
            y = bell(int(np.clip(n, 62, 96)), 2.2, 0.8, seed=j, size=0.3)
        elif kd == "cel":
            y = np.ascontiguousarray(SF2_one(SF_MS, 8, int(np.clip(n, 60, 104)), 1.2))
        else:
            key, bus, lo, hi, g0, _ = INST[{"harp": "harp", "glock": "glock", "pizz": "pizz"}[kd]]
            y = I(key).render(int(np.clip(n, lo, hi)), None, 0.6, seed=j)
        y = y[:S(2.0)]
        k0 = int(np.argmax(np.abs(y).max(1)))
        pre = rev_clip(y[k0:])[-S(1.6):]                     # the tile's ring, reversed: it flies up...
        post = y[k0:k0 + S(0.25)]                            # ...and clicks into place
        post = post * np.linspace(1, 0, len(post))[:, None]
        clip = np.concatenate([pre, post])
        clip *= 0.3 / (np.abs(clip).max() + 1e-9)
        b.add("keys", stereo_place(clip, p, 1.0), i0=S(tl_) - len(pre), gain=0.22 * g)
    hit(t_rise, "tiles_rise", "the fallen tesserae rise as reversed notes")
    # convergence: one chord
    swell_to(b, t_cv - 0.01, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.22, after=0.0)
    timp_roll(b, t_cv - 2.2, 2.2, "D2", [(0, 0.05), (2.2, 0.55)], gain=0.6)
    d = 163.0 - t_cv
    organ(b, t_cv, d, CONV[1:], "principal", pedal="D2", gain=0.75, att=0.08, rel=0.9,
          dyn=[(0, 0.9), (1.5, 1.0), (d - 1.2, 0.55), (d, 0.4)])
    strings(b, t_cv, d, dyn=[(0, 0.7), (1.5, 0.75), (d - 1.2, 0.35), (d, 0.25)], att=0.05, rel=0.9,
            v1=["D5", "A5"], v2=["F#5", "D5"], va=["A4", "F#4"], vc=["D3", "A3"], cb=["D2"], gain=0.5)
    harp_roll(b, t_cv, ["D2", "A2", "D3", "F#3", "A3", "D4", "F#4", "A4", "D5"], 0.7, 0.018, gain=1.1)
    for n in ["D6", "A6"]:
        glass_note_(b, t_cv, n, 3.0, gain=0.08, decay=2.0)
    hit(t_cv, "convergence", "everything lands on one chord: D major (organ, strings, harp, voices)")


@functools.lru_cache(maxsize=None)
def _sf2_one(path, preset, n, dur):
    p = SF2(path, 0, preset, gain=1.0)
    p.note(0.0, n, dur, 0.7)
    y, _ = p.render(0.0, dur + 1.0)
    return y


def SF2_one(path, preset, n, dur):
    return _sf2_one(path, preset, int(n), float(dur)).copy()


# ================================================================================================ m09
BT = 60.0 / 84.0
HYMN_FALLBACK = [(0, 2, [69, 62, 57, 50]), (2, 2, [67, 60, 55, 48]), (4, 2, [69, 65, 60, 50]),
                 (6, 2, [72, 67, 62, 50]), (8, 2, [71, 67, 62, 43]), (10, 2, [73, 64, 57, 45]),
                 (12, 4, [74, 66, 57, 50])]


def hymn_notes():
    """[(t, dur, [s, a, t, b])] of the choir (Vocalist's hymn_grid.json, else the agreed fallback)."""
    g = hymn_grid().get("choir")
    t0 = CUE["hymn"]
    if g:
        out = []
        for e in g:
            t = float(e.get("t", t0 + BT * float(e.get("beat", 0))))
            d = float(e.get("dur", BT * float(e.get("beats", 2))))
            out.append((t, d, [int(x) for x in e["satb"]]))
        return out
    return [(t0 + bb * BT, nb * BT, satb) for (bb, nb, satb) in HYMN_FALLBACK]


def m09(b):
    """The Dome of Flagistan (163.0–187.5): under the cantor, the ison on the organ (16' + 8', D) and a breath of
    low strings; under the choir, organum — the organ doubles every choir voice exactly, over a D pedal, with
    the strings joining in the third phrase and brass + timpani crowning the last 'Flags.'; then heaven and
    earth pass away: the chord breaks into drifting tesserae while one sung D remains; bVII–I on 'completed';
    a white-gold swell into the flash."""
    rng = np.random.default_rng(909)
    t0 = CUE["hymn_cantor"]
    th = CUE["hymn"]
    te = CUE["hymn_end"]
    # ---- cantor: organ ison
    I("organ_softped").note(b, "org", t0 - 0.2, "D2", th - t0 + 0.4, 0.8, 0.0, gain=0.3, att=0.6, rel=0.2)
    I("organ_soft").note(b, "org", t0 - 0.2, "D3", th - t0 + 0.4, 0.8, -0.1, gain=0.12, att=0.6, rel=0.2)
    I("organ_soft").note(b, "org", t0 + 1.5, "A3", th - t0 - 1.1, 0.8, 0.1, gain=0.07, att=1.5, rel=0.2)
    strings(b, t0, th - t0, dyn=[(0, 0.1), (2.5, 0.18), (th - t0, 0.22)], att=1.0, rel=0.2, cb=["D2"], vc=["D3"],
            gain=0.4)
    hit(t0, "hymn_cantor", "organ ison D (16'+8') under the cantor")
    # the drum windows catch the light one by one (glass chimes), the oculus swells into the choir's entry
    cel0 = cel_part(1.2)
    for j, tcm in enumerate(evs("m09", "chime", t0, th)[:8]):
        n = ["A5", "D6", "E6", "A6", "F#6", "D7", "B6", "A6"][j % 8]
        glass_note_(b, tcm, n, 2.4, gain=0.06, pan=-0.7 + 0.2 * j, decay=1.8)
        cel0.note(tcm, m(n), 0.3, 0.25)
    t_sw = ev("m09", "swell", th - 1.35, lo=t0 + 2.0, hi=th - 0.3)
    strings(b, t_sw, th - t_sw, "trem", dyn=[(0, 0.05), (th - t_sw, 0.4)], att=0.3, rel=0.4, v1=["A5", "D6"],
            v2=["F#5"], gain=0.3)
    cel0.to(b, "keys", pan=0.1, gain=0.8)
    # ---- choir: organum doubled by the organ, over the D pedal
    notes = hymn_notes()
    phrase_lv = [0.55, 0.62, 0.75, 0.92]
    I("organ_pedal").note(b, "org", th, "D1", te - th + 0.3, 0.8, 0.0, gain=0.22, att=0.1, rel=0.9)
    cel = cel_part(1.2)
    for (t, d, satb) in notes:
        ph = int(np.clip((t - th) / (4 * BT) + 1e-3, 0, 3))
        lv = phrase_lv[ph]
        last = t + d >= te - 0.2
        dd = d - (0.08 if not last and abs(((t - th) / BT + d / BT) % 4) < 0.05 else 0.0)
        reg = "flute" if ph == 0 else ("principal" if ph < 3 else "plenum")
        organ(b, t, dd + (0.4 if last else 0.02), satb, reg, pedal=satb[3] - 12 if satb[3] - 12 >= 24 else satb[3],
              gain=0.55 + 0.4 * lv, att=0.03, rel=0.25 if not last else 1.2, spread=0.6)
        if ph >= 2:
            strings(b, t, dd + 0.02, dyn=[(0, lv * 0.7), (dd, lv * 0.75)], att=0.04, rel=0.3 if not last else 1.0,
                    v1=[satb[0] + 12] if satb[0] + 12 <= 86 else [satb[0]], v2=[satb[1] + 12], va=[satb[2]],
                    vc=[satb[3]], cb=[satb[3] - 12] if satb[3] - 12 >= 28 else [satb[3]], gain=0.45)
        # a shimmer ring on each beat (the tiling revealed): celesta chord tones
        rings = [e for e in _events("m09", "ring") if th - 0.05 <= e["t"] < te]
        for q in range(int(round(d / BT))):
            tq = t + q * BT
            rs = [float(e["strength"]) for e in rings if abs(e["t"] - tq) < 0.06]  # M09's accents 1.0/.3/.7/.3
            cel.note(tq, satb[0] + 12, 0.3, (0.12 + 0.3 * rs[0]) if rs else (0.2 + 0.15 * lv))
    for j, tp in enumerate([th + 4 * BT * k for k in range(4)]):
        hit(tp, f"hymn_phrase_{j + 1}", "organum phrase downbeat (organ doubles the choir)")
    t4 = th + 12 * BT
    timp_roll(b, t4 - 1.4, 1.4, "D2", [(0, 0.1), (1.4, 0.7)], gain=0.7)
    swell_to(b, t4 - 0.01, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.22, after=0.0)
    timp(b, t4, "D2", 0.8, gain=0.9)
    brass(b, t4, te - t4 + 0.2, dyn=[(0, 0.55), (te - t4, 0.85), (te - t4 + 0.2, 0.6)], att=0.05, rel=0.8,
          tpt=["A4", "D5", "F#5"], hn=["D4", "F#4", "A4"], tbn=["D3", "A3"], tuba=["D2"], gain=0.3)
    harp_roll(b, t4, ["D2", "A2", "D3", "F#3", "A3", "D4", "F#4", "A4", "D5"], 0.7, 0.02, gain=1.0)
    # ---- the gateless gate: the chord breaks into drifting tesserae; one D remains
    tg = CUE["gateless_gate"]
    td = ev("m09", "dissolve", tg, lo=180.0, hi=181.0)
    t_n10 = LINES["N10"]["start"]
    ison(b, "ison", te - 0.3, 187.3, notes=("D3", "D2", "D3", "D3"), gain=0.42, fin=1.2, fout=1.6, seed=91,
         level=lambda t: lines_gain(t, 0.7, ids=["N09", "N10"]))
    hit(td, "gateless_gate", "the organ chord breaks into drifting tesserae; one sung D remains")
    drift = ["D5", "F#5", "A5", "D6", "A4", "F#4", "E5", "B5", "D4"]
    t, k = td, 0
    while t < t_n10 - 0.2:
        x = (t - td) / (t_n10 - td)
        n = m(drift[k % len(drift)]) + (12 if rng.random() < 0.25 else 0) - int(round(5 * x * rng.random()))
        kd = k % 4
        g = (1 - 0.8 * x) * float(lines_gain(t, 0.7, ids=["N09"]))
        p = float(np.clip(rng.normal(0, 0.5 + 0.4 * x), -0.95, 0.95))
        if kd == 0:
            cel.note(t, min(n + 12, 104), 0.3, 0.2 + 0.25 * g)
        elif kd == 1:
            I("harp").note(b, "keys", t, min(n, 96), None, 0.3 + 0.3 * g, p, gain=0.15 * g, seed=k)
        elif kd == 2:
            glass_note_(b, t, min(n + 12, 100), 1.6, gain=0.06 * g, pan=p, decay=0.9)
        else:
            psaltery_to(b, "keys", t, min(n, 93), 1.6, 0.3 + 0.3 * g, gain=0.25 * g, pan=p)
        t += float(rng.exponential(1.0 / (7.0 * (1 - 0.75 * x))))
        k += 1
    organ(b, te - 0.02, 1.8, [57, 62, 66, 69], "principal", pedal="D2", gain=0.6, att=0.02, rel=1.4,
          dyn=[(0, 0.8), (1.8, 0.0)])
    t_fl_ = ev("m09", "glow", W("N09", "Flags"), lo=182.8, hi=184.0)  # '...the ten thousand FLAGS remain'
    harp_roll(b, t_fl_, ["D4", "A4", "D5", "F#5", "A5", "D6"], 0.55, 0.035, gain=0.8)
    glass_note_(b, t_fl_, "D7", 2.5, gain=0.08, decay=1.8)
    glass_note_(b, t_fl_ + 0.04, "A6", 2.5, gain=0.07, decay=1.8)
    hit(t_fl_, "flags_remain", "the floating Flags flare: harp + glass D major")
    # ---- 'And the Survey was completed.': bVII - I
    t_sv = LINES["N10"]["start"]  # survey_done cue: the bVII chord opens under 'And the Survey...'
    t_cp = ev("m09", "settle", W("N10", "completed"), lo=185.4, hi=186.3)
    organ(b, t_sv, t_cp - t_sv + 0.03, ["C3", "E3", "G3", "C4", "E4"], "flute", pedal="C2", gain=0.75, att=0.08, rel=0.3)
    strings(b, t_sv, t_cp - t_sv + 0.03, dyn=[(0, 0.2), (t_cp - t_sv, 0.35)], att=0.1, rel=0.3, v2=["G4", "C5"],
            va=["E4"], vc=["C3"], gain=0.45)
    I("harp").note(b, "keys", t_sv, "C3", None, 0.5, -0.3, gain=0.22)
    hit(t_sv, "survey_done", "bVII (C major) opens under 'And the Survey...'")
    d = 187.46 - t_cp
    organ(b, t_cp, d, ["D3", "F#3", "A3", "D4", "F#4", "A4"], "principal", pedal="D2", gain=0.7, att=0.05, rel=0.25,
          dyn=[(0, 0.75), (d, 1.0)])
    strings(b, t_cp, d, dyn=[(0, 0.45), (d, 0.9)], att=0.05, rel=0.15, v1=["F#5", "A5", "D6"], v2=["D5", "A4"],
            va=["F#4"], vc=["D3", "A3"], cb=["D2"], gain=0.5)
    brass(b, t_cp, d, dyn=[(0, 0.3), (d, 0.7)], att=0.2, rel=0.15, hn=["D4", "F#4", "A4"], gain=0.28)
    harp_roll(b, t_cp, ["D3", "A3", "D4", "F#4", "A4", "D5"], 0.6, 0.03, gain=0.9)
    timp(b, t_cp, "D2", 0.6, gain=0.7)
    hit(t_cp, "completed", "bVII-I lands on 'completed'")
    # ---- the white-gold flash
    t_fl = ev("m09", "flash", 187.0, lo=186.6, hi=187.3)
    t_pk = ev_to("m09", "flash", 187.46, lo=t_fl - 0.01, hi=t_fl + 0.01)
    strings(b, t_fl - 0.6, t_pk - t_fl + 0.6, "trem", dyn=[(0, 0.2), (t_pk - t_fl + 0.6, 1.0)], att=0.2, rel=0.3,
            v1=["A6", "D7"] if m("D7") <= 86 else ["A5", "D6"], v2=["F#6"] if m("F#6") <= 86 else ["F#5"], gain=0.4)
    swell_to(b, t_pk, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.3, after=0.12)
    for n in ["D6", "A6", "D7", "F#7"]:
        glass_note_(b, t_fl - 0.3, n, 1.4, gain=0.07, decay=1.5, att=0.6)
    cel.to(b, "keys", pan=0.2, gain=0.8)
    hit(t_pk, "flash", "white-gold swell peaks into the flash")


# ================================================================================================ m10
def m10(b):
    """The Sticky Flag: Explicit (187.5–212.0): dawn in the apse — a recorder and a psaltery over a soft D; the
    sticky koan as an organ CIPHER: the organist lets go on 'I do not want this Flag' but one pipe (A5) will
    not stop sounding — through 'all Flags are one Flag!' (the stuck A turns out to be the first note of the
    hymn line, which the psaltery completes) and 'This one is sticky.' It never lets go: it is the fifth of
    GLORIOUS! (full organ, a peal of bells, brass, timpani — the organ opens stop by stop while the strings
    climb the hymn line under 'Go forth, O signifier'), it hangs over the cantor's EXPLICIT (the ison
    returns) and is a sticky ninth in the gratias' plagal A-men; the basilica's bell rings once; the candle
    is snuffed, everything fades with the light — and alone in the black the stuck pipe finally sighs out
    with the sticky Flag."""
    rng = np.random.default_rng(1010)
    t0 = 187.5
    # ---- dawn: recorder (the hymn line, major) + psaltery over a soft organ D
    I("organ_soft").note(b, "org", t0 + 0.05, "D4", 3.1, 0.8, 0.0, gain=0.08, att=0.8, rel=0.6)
    I("organ_softped").note(b, "org", t0 + 0.05, "D2", 3.1, 0.8, 0.0, gain=0.22, att=0.8, rel=0.6)
    rec = SF2(SF_MS, 0, 74, gain=1.0)
    for (dt, n, d) in [(0.15, "A5", 0.3), (0.45, "G5", 0.3), (0.75, "A5", 0.6)]:
        rec.note(t0 + dt, m(n), d, 0.55)
    rec.to(b, "ww", pan=-0.25, gain=0.5)
    for j, n in enumerate(["D5", "F#5", "A5", "D6"]):
        psaltery_to(b, "keys", t0 + 0.1 + 0.12 * j, n, 2.2, 0.5, gain=0.35, pan=0.3)
    t_l4 = LINES["L04"]["end"]
    rec2 = SF2(SF_MS, 0, 74, gain=1.0)
    for j, n in enumerate(["C#6", "B5", "C#6", "D6"]):
        rec2.note(t_l4 + 0.05 + 0.18 * j, m(n), 0.18 if j < 3 else 0.5, 0.5)
    rec2.to(b, "ww", pan=-0.25, gain=0.45)
    # ---- the cipher: a chord is played and released; one pipe (A5, a 2' flute) keeps sounding — and never lets
    # go: it is the fifth of the Glorious chord, a sticky ninth in the gratias' IV, and alone in the black it
    # finally sighs out with the sticky Flag (M10: the Flag lingers until 211.72, gone by 211.96)
    t_ch = LINES["L04"]["end"] + 0.1        # a little presentation chord as Lutie offers the Flag...
    t_rel = CUE["sticky"]                    # ...released on 'I do not want this Flag' — but one pipe sticks
    organ(b, t_ch, t_rel - t_ch, ["D4", "F#4", "A4", "D5"], "flute", pedal="D2", gain=0.6, att=0.2, rel=0.35)
    t_lin, t_off = 211.72, 211.96  # the sticky Flag lingers alone in the black, then fades (M10)
    dc = t_off - t_ch
    wob = lambda x: 0.06 * np.sin(2 * np.pi * 0.3 * x) + 0.03 * np.sin(2 * np.pi * 1.1 * x + 1)
    sag = lambda x: wob(x) - 0.8 * np.clip((x - (t_lin - t_ch)) / (t_off - t_lin), 0, 1) ** 1.4
    cy = I("organ_soft").render(m("A5"), dc + 0.1, 0.8, bend=sag, att=0.2, rel=0.05, seed=5)[:S(dc)]
    tc_ = t_ch + np.arange(len(cy)) / SR
    cenv = ramp(tc_, [(t_ch, 1.0), (199.4, 0.95), (202.9, 0.65), (205.0, 0.8), (t_lin, 0.85), (t_off, 0.0)])
    b.add("cipher", stereo_place(cy * cenv[:, None].astype(np.float32), 0.35, 1.0), t=t_ch, gain=0.3)
    hit(t_rel, "sticky", "the organist lets go, one pipe (A5) keeps sounding: the sticky cipher", kind="cut")
    hit(t_lin, "cipher_off", "alone in the black, the stuck pipe finally sighs out with the sticky Flag",
        kind="cut")
    # the gags, as M10 staged them: touch (pizz), stretch (a violin glissando that cannot let go), shakes
    # (flexatone wobbles), the pole that will not leave Lutie's hand (flexatone roll), deadpan twinkles
    FLX = V1P + "varMetal/various/"
    gags = [e for e in _events("m10", "sticky") + _events("m10", "cling") if 189.0 <= e["t"] <= 199.0]
    for j, e in enumerate(sorted(gags, key=lambda e: e["t"])):
        t, desc = float(e["t"]), str(e.get("desc", ""))
        if desc.startswith("touch"):
            I("vln_pizz").note(b, "str", t, "A5", None, 0.5, 0.3, gain=0.24)
        elif desc.startswith("stretch"):
            dd = 0.34
            I("vln_solo").note(b, "str", t, "E5", dd, 0.35, 0.3, gain=0.2, bend=lambda x, dd=dd: 5.0 * (x / dd) ** 1.6,
                               dyn=[(0, 0.15), (dd, 0.35)], att=0.05, rel=0.03)
        elif desc.startswith("shake"):
            perc(b, t, FLX + ("flexatone_slap1.wav" if j % 2 else "flexatone_slap2.wav"), gain=0.12, pan=0.25)
        elif e.get("kind") == "cling" and "pole" in desc:
            perc(b, t, FLX + "flexatone_fast.wav", gain=0.1, pan=-0.3,
                 dur=ev_to("m10", "cling", t + 0.4, lo=t - 0.01, hi=t + 0.01) - t)
        else:
            continue
        hit(t, f"sticky_{j + 1}", f"sticky gag: {desc[:40]}")
    for j, t in enumerate(evs("m10", "twinkle", 198.5, 199.7)[:5]):
        cel0 = [("D7", 0.5), ("A6", 0.44), ("F#6", 0.38), ("E6", 0.32), ("D6", 0.26)][j]
        I("glock").note(b, "keys", t, cel0[0], None, cel0[1], 0.4 - 0.2 * j, gain=0.05)
    # 'all Flags are one Flag!' — the stuck A is the hymn line's first note; the psaltery completes it
    tq = LINES["L05"]["end"] + 0.1
    for j, n in enumerate(["G5", "A5", "C#6", "B5", "C#6", "D6"]):
        psaltery_to(b, "keys", tq + 0.2 * j, n, 1.5 if j < 5 else 2.5, 0.55, gain=0.4, pan=0.25)
    I("harp").note(b, "keys", tq + 1.0, "D3", None, 0.45, -0.3, gain=0.2)
    hit(tq, "one_flag", "the psaltery completes the hymn line from the stuck A")
    # ---- 'Go forth, O signifier, and become glorious.': the organ opens, the strings climb the hymn line
    t_n = LINES["N11"]["start"]
    t_g = ev("m10", "raise", CUE["glorious"], lo=202.6, hi=203.2)
    ws = [t_n, W("N11", "forth"), W("N11", "O"), W("N11", "signifier"), W("N11", "and"), W("N11", "glorious"), t_g]
    line = ["A4", "G4", "A4", "C#5", "B4", "C#5"]
    harm = [["D3", "F#3", "A3"], ["E3", "G3", "B3"], ["F#3", "A3", "D4"], ["E3", "A3", "C#4"], ["D3", "G3", "B3"],
            ["E3", "A3", "C#4"]]
    ped = ["D2", "E2", "F#2", "A1", "G1", "A1"]
    for j in range(6):
        a, z = ws[j], ws[j + 1]
        d = z - a
        lv = 0.3 + 0.1 * j
        reg = "flute" if j < 2 else ("principal" if j < 4 else "plenum")
        organ(b, a, d + 0.03, harm[j], reg, pedal=ped[j], gain=0.5 + 0.4 * lv, att=0.06, rel=0.3)
        strings(b, a, d + 0.04, dyn=[(0, lv), (d, lv + 0.08)], att=0.08, rel=0.3, v1=[m(line[j]) + 12], v2=[line[j]],
                vc=[m(ped[j]) + 12], gain=0.5)
    timp_roll(b, ws[5], t_g - ws[5], "A2", [(0, 0.2), (t_g - ws[5], 0.9)], gain=0.8)
    swell_to(b, t_g - 0.01, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.3, after=0.0)
    # ---- GLORIOUS!: full organ, a peal of bells, brass, timpani
    organ(b, t_g, 1.8, ["D3", "A3", "D4", "F#4", "A4", "D5"], "full", pedal="D2", gain=0.8, att=0.01, rel=1.4,
          dyn=[(0, 1.0), (0.5, 0.75), (1.8, 0.5)])
    brass(b, t_g, None, "stac", 1.0, tpt=["D5", "F#5", "A5"], hn=["D4", "A4"], tbn=["D3", "A3"], tuba=["D2"], gain=0.2)
    brass(b, t_g, 1.4, dyn=[(0, 0.9), (0.4, 0.55), (1.4, 0.3)], att=0.0, rel=0.8, tpt=["A4", "D5"], hn=["F#4", "A4"],
          tbn=["D3"], gain=0.22)
    strings(b, t_g, 1.6, dyn=[(0, 0.95), (0.4, 0.55), (1.6, 0.3)], att=0.0, rel=1.0, v1=["D6", "A5"], v2=["F#5"],
            va=["A4"], vc=["D3"], cb=["D2"], gain=0.6)
    timp(b, t_g, "D2", 1.0, gain=1.2)
    crash(b, t_g, "ff", 0.7)
    bell_to(b, "bell", t_g, "D3", gain=0.4, strike=1.0, seed=20, size=1.15)
    peal(b, "bell", t_g + 0.25, 205.9, ["D5", "C#5", "B4", "A4", "G4", "F#4", "E4", "D4"], interval=0.19, gain=0.16,
         seed=21, method="plain", size=0.5)
    hit(t_g, "glorious", "GLORIOUS!: full organ + bells (peal) + brass + timpani")
    # ---- EXPLICIT: the ison returns under the cantor (mirror of the incipit)
    t_x = CUE["explicit"]
    ison(b, "ison", t_x - 0.6, CUE["gratias"] + 0.2, notes=("D3", "D3", "D2", "D3", "D2"), gain=0.45, fin=1.5,
         fout=0.9, seed=101)
    I("organ_softped").note(b, "org", t_x - 0.4, "D2", CUE["gratias"] - t_x + 0.3, 0.8, 0.0, gain=0.24, att=1.0, rel=0.4)
    hit(t_x, "explicit", "the ison returns under the cantor's EXPLICIT")
    # ---- Vexillo gratias: plagal I - IV - I doubled on the organ (from the vocal grid if exported)
    gg = hymn_grid().get("gratias")
    if gg:
        syl = [(float(e["t"]), float(e.get("dur", 0.6)), [int(x) for x in e["satb"]]) for e in gg]
        ch = []
        for (t, d, satb) in syl:  # the organ holds each chord across its syllables (I - IV - I)
            if ch and ch[-1][2] == satb:
                ch[-1] = (ch[-1][0], t + d - ch[-1][0], satb)
            else:
                if ch:
                    ch[-1] = (ch[-1][0], t - ch[-1][0], ch[-1][2])
                ch.append((t, d, satb))
    else:
        ch = [(207.6, 0.94, [69, 66, 62, 50]), (208.54, 0.62, [71, 67, 62, 43]), (209.16, 0.62, [69, 66, 62, 50])]
    for i, (t, d, satb) in enumerate(ch):
        last = i == len(ch) - 1
        dd = d + (0.35 if last else 0.02)
        organ(b, t, dd, satb, "flute", pedal=satb[3] - 12 if satb[3] - 12 >= 36 else satb[3],
              gain=0.7, att=0.06, rel=1.6 if last else 0.2)
        strings(b, t, dd, dyn=[(0, 0.25), (d, 0.3)], att=0.08, rel=1.4 if last else 0.2,
                v2=[satb[0]], va=[satb[1]], vc=[satb[3]], gain=0.4)
    hit(ch[0][0], "gratias", "plagal A-men under Vexillo gratias (organ doubles the choir)")
    # ---- the basilica's bell, once; black by 212.0
    t_b = CUE["end_card"]  # over the choir's held '-as': the bell of the candle, once more; the snuff is silent
    bell_to(b, "bell", t_b, "D3", gain=0.45, pan=0.15, strike=0.9, seed=3, size=1.15)
    hit(t_b, "end_card", "the basilica's bell, once, over the held A-men (end card)")


SECTIONS = dict(m01=m01, m02=m02, m03=m03, m04=m04, m05=m05, m06=m06, m07=m07, m08=m08, m09=m09, m10=m10)

# ================================================================================================ master
BUSES = {
    #          gain(dB)  reverb      wet    eq
    "org":    (0.0, "basilica", 0.70, [("hp", 24), ("pk", 550, -2.0, 0.7), ("hs", 6000, -1.5)]),
    "ison":   (0.0, "basilica", 0.55, [("hp", 45), ("pk", 2800, -2.5, 1.0), ("hs", 7500, -4.0)]),
    "bell":   (0.0, "basilica", 0.65, [("hp", 40)]),
    "str":    (0.0, "basilica", 0.46, [("hp", 30), ("hs", 7000, 0.5)]),
    "brs":    (-1.0, "basilica", 0.50, [("hp", 40), ("pk", 2500, -1.5, 0.8)]),
    "ww":     (-2.0, "basilica", 0.45, [("hp", 80)]),
    "perc":   (-1.0, "basilica", 0.42, [("hp", 55)]),  # Foley owns the sub (< 60 Hz) of impacts
    "keys":   (-2.0, "basilica", 0.50, [("hp", 60)]),
    "choir":  (-1.0, "basilica", 0.60, [("hp", 80)]),
    "glass":  (-2.0, "void", 0.70, [("hp", 150)]),
    "slop":   (-2.0, "basilica", 0.30, [("hp", 60)]),
    "fx":     (-3.0, "basilica", 0.30, [("hp", 40)]),
    "cipher": (0.0, "basilica", 0.45, [("hp", 300)]),   # the stuck pipe (m10), outlives the end fade
}
# reverb tails do not cross these cuts: (t, fade-out seconds after t, buses or None = all)
CUTS = [(95.75, 0.012, None), (145.0, 0.0, None)]
SILENCE = (145.0, 146.5)
# macro-dynamics: per-section level trims (dB) applied when the cached sections are summed
TRIM = {"m06": 7.0, "m07": 4.0}
END_FADE = (210.45, 212.0)  # the candle is snuffed at 210.45 (M10); lights out -> black at 212.0


@functools.lru_cache(maxsize=None)
def room(name):
    if name == "basilica":  # nave of a large stone basilica, candlelit: long, dark, diffuse
        return make_ir(rt60=5.4, pre=0.038, bright=0.72, seed=31, lo_mult=1.3, hi_mult=0.36, er=0.32, build=0.035)
    if name == "void":
        return make_ir(rt60=8.0, pre=0.05, bright=1.0, seed=32, lo_mult=1.0, hi_mult=0.7, build=0.2, er=0.0)
    raise KeyError(name)


def reverb_cut(x, name, wet, cuts):
    """reverb per segment between cuts (tails faded out after each cut)."""
    from music_lib import conv_stereo
    ir = room(name)
    y = np.zeros_like(x)
    edges = [0] + [S(c[0]) for c in cuts] + [N]
    fades = [c[1] for c in cuts] + [None]
    xx = x + 0.25 * x[:, ::-1]
    for i, (a, z) in enumerate(zip(edges[:-1], edges[1:])):
        seg = xx[a:z]
        nz = np.flatnonzero(np.abs(seg).max(1) > 1e-7)
        if len(nz) == 0:
            continue
        s0, s1 = a + nz[0], a + nz[-1] + 1
        w = conv_stereo(xx[s0:s1], ir)
        if fades[i] is not None:
            fd = S(fades[i])
            lim = z - s0 + fd
            w = w[:lim].copy()
            if fd > 0:
                w[-fd:] *= np.linspace(1, 0, fd)[:, None] ** 2
        e = min(N, s0 + len(w))
        y[s0:e] += w[:e - s0]
    return y * np.float32(wet)


def dialogue_env():
    """0..1 envelope of dialogue presence (all timeline lines incl. the sung ones), attack 60 ms, release 250 ms."""
    fr = 1000
    n = int(DUR * fr)
    g = np.zeros(n)
    for ln in TLJ["lines"]:
        a, z = int((ln["start"] - 0.06) * fr), int((ln["end"] + 0.12) * fr)
        g[max(0, a):min(n, z)] = 1.0
    out = np.zeros(n)
    s = 0.0
    ca, cr = 1 - np.exp(-1 / (0.06 * fr)), 1 - np.exp(-1 / (0.25 * fr))
    for i in range(n):
        s += (g[i] - s) * (ca if g[i] > s else cr)
        out[i] = s
    return np.interp(np.arange(N) / SR, np.arange(n) / fr, out).astype(np.float32)


def glue(x, lufs_in, above=6.0, ratio=2.5):
    import pedalboard as pb
    c = pb.Compressor(threshold_db=float(lufs_in + above), ratio=ratio, attack_ms=15.0, release_ms=300.0)
    return c(x.T.astype(np.float32), SR).T.copy()


def master(sec_files):
    t0 = time.time()
    buses = {}
    for f in sec_files:
        z = np.load(f)
        tr = db(TRIM.get(os.path.basename(f)[4:-4], 0.0))
        for key in z.files:
            if key.endswith("__i0"):
                continue
            i0 = int(z[key + "__i0"])
            x = z[key] * np.float32(tr)
            buses.setdefault(key, np.zeros((N, 2), np.float32))
            e = min(N, i0 + len(x))
            buses[key][i0:e] += x[:e - i0]
    mix = np.zeros((N, 2), np.float32)
    a_end = S(END_FADE[0])
    fend = (np.cos(np.linspace(0, np.pi / 2, N - a_end)) ** 2)[:, None].astype(np.float32)
    for key, x in buses.items():
        g, rv, wet, fx = BUSES[key]
        if fx:
            x = eq(x, *fx)
        y = x * db(g)
        if rv:
            cuts = sorted(c for c in CUTS if c[2] is None or key in c[2])
            w = reverb_cut(x, rv, wet, cuts) * db(g)
            if key == "cipher":  # its own ring dies in the last 0.3 s of black
                a2 = S(211.7)
                w[a2:] *= np.linspace(1, 0, N - a2)[:, None] ** 2
            y = y + w
        if key != "cipher":  # black by 212.0: the last bell and every tail fade with the picture
            y[a_end:] *= fend
        mix += y
        print(f"  bus {key:6s} peak {np.abs(y).max():.3f}", flush=True)
    d = dialogue_env()
    lo = butter(mix, "lowpass", 250, 2, zero_phase=True)
    hi = butter(mix, "highpass", 4000, 2, zero_phase=True)
    mid = mix - lo - hi
    mix = lo + hi + mid * (1 - (1 - db(-4.5)) * d)[:, None]
    mix[S(SILENCE[0]):S(SILENCE[1])] = 0.0
    L0 = lufs(mix)
    mix = glue(mix, L0)
    L0 = lufs(mix)
    mix *= db(-19.0 - L0)
    mix, gmin = limiter(mix, -1.2)
    mix[S(SILENCE[0]):S(SILENCE[1])] = 0.0  # sacred silence: digital zero
    mix[-S(0.004):] *= np.linspace(1, 0, S(0.004))[:, None]
    mix[-1] = 0.0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sf.write(OUT, mix.astype(np.float32), SR, subtype="FLOAT")
    print(f"master: {time.time() - t0:.1f}s  LUFS pre {L0:.1f} -> {lufs(mix):.2f}  limiter min gain "
          f"{20 * np.log10(gmin):.1f} dB  true peak {20 * np.log10(true_peak(mix)):.2f} dBTP", flush=True)


def render_section(name):
    global HITS
    HITS = []
    t0 = time.time()
    b = Buf()
    SECTIONS[name](b)
    out = {}
    for k, x in b.b.items():
        nz = np.flatnonzero(np.abs(x).max(1) > 1e-8)
        if len(nz) == 0:
            continue
        a, z = nz[0], nz[-1] + 1
        out[k] = x[a:z]
        out[k + "__i0"] = np.array(a)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, f"sec_{name}.npz"), **out)
    with open(os.path.join(CACHE, f"hits_{name}.json"), "w") as f:
        json.dump(HITS, f, indent=1)
    print(f"{name}: {time.time() - t0:.1f}s buses={sorted(k for k in out if not k.endswith('__i0'))} hits={len(HITS)}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--master", action="store_true")
    a = ap.parse_args()
    if not a.master:
        for n in (a.only.split(",") if a.only else list(SECTIONS)):
            render_section(n)
    files = [os.path.join(CACHE, f"sec_{n}.npz") for n in SECTIONS if os.path.exists(os.path.join(CACHE, f"sec_{n}.npz"))]
    hits = []
    for n in SECTIONS:
        p = os.path.join(CACHE, f"hits_{n}.json")
        if os.path.exists(p):
            hits += json.load(open(p))
    with open(os.path.join(MUS, "cues.json"), "w") as f:
        json.dump(sorted(hits, key=lambda h: h["t"]), f, indent=1)
    with open(os.path.join(MUS, "silence.json"), "w") as f:
        json.dump([list(SILENCE)], f)
    master(files)


if __name__ == "__main__":
    main()

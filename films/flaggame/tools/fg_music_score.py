"""THE FLAG GAME? — the score (film 2, owned by Composer).

  source films/flaggame/env.sh
  python films/flaggame/tools/fg_music_score.py                 # all sections + master -> audio/stems/music.wav
  python films/flaggame/tools/fg_music_score.py --only t04,t05  # re-render those sections (rest from cache)
  python films/flaggame/tools/fg_music_score.py --master        # master only
  python tools/music_verify.py [--spec all]                     # checks (film-aware via VX_FILM)

An evangelical-revival palette played absolutely straight — harmonium, pipe organ, upright piano, choir —
framing a Cecil-B-DeMille biblical epic for the flashback. Tonal centre D for the whole film (D minor for the
storm, the Devil, the burn and BLASPHEME; the altar-call hymn is the Vocalist's E-flat, a semitone lift).

Leitmotif ("the Altar Call"): the incipit of WOODWORTH (Bradbury 1849, PD), the hymn tune of "Just as I Am":
    D-E | F#  F# | A  G F# | E  F# G | F#      (pickup, then 3/4)
It is the harmonium alone in the rain (t01, cut off by the tract's radio sting), the Flag Maker's organ voluntary
(t04), a pizzicato march (t05), the firmament's glory in trumpets (t05), the fragile rising line of the ascending
Flag (t06), a fast-forward Sousa-style march over 100 raisings (t07), a harmonica on the empty road (t07), the
hymn itself (t08, sung), and a gospel Amen that lands the tract in the mud and loops the film (t09 -> t01).
Also quoted: J. S. Bach, Toccata in D minor BWV 565 (PD) for the Devil.

Sections write into named buses (global time) and are cached in films/flaggame/audio/music/cache/.
The buffer WRAPS: anything that rings past 199.0 s (notes and reverb) continues at 0.0, because the film loops.
Sync points are logged to films/flaggame/audio/music/cues.json (checked by tools/music_verify.py).
"""
import os as _os

_os.environ.setdefault("VX_FILM", "flaggame")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ.setdefault(_k, "4")
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO, "tools"))
from music_lib import (DUR, FILM_ROOT, N, SF_GU, SF_MS, SR, SF2, Buf, IR, S, _resample, butter, conv_stereo,  # noqa
                       db, envelope, eq, fade, fm_bell, glass, hz, m, noise, riser, shepard, stereo_place, sub_drone,
                       supersaw, svf, tt, _saw_bank)
from music_orch import (CH, I, PAN, PERC, V1P, bell_note, crash, cut_after, glass_note, harp_gliss, harp_roll,  # noqa
                        limiter, lufs, perc, root, sf2part, strings, brass, swell_to, taiko, timp, timp_roll, tones,
                        true_peak, tutti)

assert DUR == 199.0, "run with VX_FILM=flaggame (source films/flaggame/env.sh)"
MUS = os.path.join(FILM_ROOT, "audio", "music")
CACHE = os.path.join(MUS, "cache")
OUT = os.path.join(FILM_ROOT, "audio", "stems", "music.wav")
TLJ = json.load(open(os.path.join(FILM_ROOT, "timeline.json")))
CUE = {c["id"]: c["t"] for c in TLJ["cues"]}
LINES = {l["id"]: l for l in TLJ["lines"]}
GRID = json.load(open(os.path.join(MUS, "grid.json")))
HITS = []


# ------------------------------------------------------------------------------------------------ plumbing
class LoopBuf(Buf):
    """Buf whose timeline is a circle: whatever rings past the end of the film continues at 0.0."""

    def add(self, bus, x, t=None, i0=None, gain=1.0):
        if x is None or len(x) == 0:
            return
        if x.ndim == 1:
            x = np.stack([x, x], 1)
        i0 = S(t) if i0 is None else i0
        n = len(x)
        if i0 + n > N and i0 < N:
            k = N - i0
            super().add(bus, x[:k], i0=i0, gain=gain)
            super().add(bus, x[k:k + N], i0=0, gain=gain)
        else:
            super().add(bus, x, i0=i0, gain=gain)


def scene_events(scene, kind=None, contains=None, fallback=()):
    """visual hits exported by the scene owners (films/flaggame/cache/foley/<scene>_*events.json)."""
    d = os.path.join(FILM_ROOT, "cache", "foley")
    ts = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if not (f.startswith(scene + "_") and f.endswith("events.json")):
                continue
            try:
                j = json.load(open(os.path.join(d, f)))
            except Exception:
                continue
            ev = j["events"] if isinstance(j, dict) else j
            ts += [e["t"] for e in ev if (kind is None or e.get("kind") == kind)
                   and (contains is None or contains.lower() in str(e.get("desc", "")).lower())]
    return sorted(ts) if ts else list(fallback)


def W(lid, word, k=0):
    ws = [w for w in LINES[lid]["words"] if w["w"].lower().strip(",.!?:") == word.lower()]
    return ws[k]["s"]


def hit(t, name, desc="", kind="onset"):
    HITS.append(dict(t=round(float(t), 4), id=name, desc=desc, kind=kind))


# motif: WOODWORTH incipit in D (beats relative to the pickup; durations in beats)
MOTIF = [(0.0, 0.5, "D4"), (0.5, 0.5, "E4"), (1.0, 2.0, "F#4"), (3.0, 1.0, "F#4"), (4.0, 1.5, "A4"), (5.5, 0.5, "G4"),
         (6.0, 1.0, "F#4"), (7.0, 1.5, "E4"), (8.5, 0.5, "F#4"), (9.0, 1.0, "G4"), (10.0, 2.0, "F#4")]
MOTIF_CH = [(0, "D"), (1, "D"), (4, "D"), (7, "A"), (9, "A"), (10, "D")]


def motif(t0, beat, octv=0, tr=0, upto=None):
    """[(t, dur, midi)] of the leitmotif starting at t0 with the given beat length."""
    out = []
    for (b, d, n) in MOTIF:
        if upto is not None and b >= upto:
            break
        out.append((t0 + b * beat, d * beat, m(n) + 12 * octv + tr))
    return out


def organ(b, t, dur, notes, loud=False, pedal=None, gain=1.0, att=0.04, rel=0.25, pan=0.0, bus="org"):
    key = "organ_loud" if loud else "organ_soft"
    for j, n in enumerate(notes):
        I(key).note(b, bus, t, n, dur, 0.8, pan - 0.25 + 0.5 * j / max(1, len(notes) - 1), gain=gain * 0.2,
                    att=att, rel=rel, seed=j)
    if pedal is not None:
        I("organ_pedal" if loud else "organ_softped").note(b, bus, t, pedal, dur, 0.8, 0.0, gain=gain * 0.4, att=att,
                                                            rel=rel)


def piano(b, t, notes, dur=None, vel=0.6, gain=1.0, pan=-0.1, bus="keys"):
    for j, n in enumerate(notes if isinstance(notes, (list, tuple)) else [notes]):
        I("piano", rel=0.25).note(b, bus, t, n, dur, vel, pan + 0.04 * j, gain=gain * 0.22, seed=int(t * 100) + j)


def sf2(b, preset_path, bank, preset, events, bus="keys", pan=0.0, width=1.0, gain=1.0, cc=None):
    """events: [(t, midi, dur, vel)] rendered through one SF2 preset."""
    p = SF2(preset_path, bank, preset, gain=gain)
    for (t, n, d, v) in events:
        p.note(t, m(n), d, v)
    return p.to(b, bus, pan=pan, width=width, cc=cc)


def stutter(b, t0, t1, seg=0.06, reps=2, shrink=0.85, buses=None):
    """buffer-repeat glitch on everything already in b between t0 and t1."""
    pattern, pos, L = [], S(t0), seg
    while pos < S(t1):
        n = S(L)
        pattern.append((pos, n))
        pos += n * reps
        L = max(0.012, L * shrink)
    for k in (buses or list(b.b)):
        x = b.b[k]
        src = x[S(t0):S(t1)].copy()
        for (p, n) in pattern:
            chunk = src[p - S(t0): p - S(t0) + n].copy()
            if len(chunk) < n:
                break
            r = min(48, n // 4)
            chunk[:r] *= np.linspace(0, 1, r)[:, None]
            chunk[-r:] *= np.linspace(1, 0, r)[:, None]
            for q in range(reps):
                a = p + q * n
                e = min(a + n, S(t1))
                if e > a:
                    x[a:e] = chunk[:e - a]


def radio(x, lo=320, hi=4200, drive=2.5):
    """AM-radio / small speaker: band-limit, saturate, mono-ish."""
    y = eq(x, ("hp", lo), ("hp", lo * 0.8), ("lp", hi), ("lp", hi * 1.2), ("pk", 1800, 4.0, 0.9))
    y = np.tanh(y * drive) / drive
    mid = y.mean(1, keepdims=True)
    return (0.85 * mid + 0.15 * y).astype(np.float32)


def sub(b, a, z):
    """render-local copy of every bus in [a, z) (for processing a group)."""
    return {k: x[S(a):S(z)].copy() for k, x in b.b.items()}


# ================================================================================================ t01
def t01(b):
    """The Tract (0–18.46): the harmonium alone in the rain (the leitmotif, cut off), the gospel-radio sting on the
    cover (2.6), a page riffle, the dive into the comic (6.4), and a gentle Sunday-school piano under the camp."""
    bt = 0.62
    # the lonely harmonium: a D chord continuing the previous loop's Amen, then the motif's first phrase …
    hm = [(0.0, "D3", 2.55, 0.35), (0.0, "A3", 2.55, 0.33), (0.0, "F#4", 1.0, 0.3)]
    for (t, d, n) in motif(0.62, bt, 0, upto=5.5):
        hm.append((t, n, min(d, 2.55 - t) - 0.02, 0.42))
    sf2(b, SF_MS, 0, 20, hm, bus="keys", pan=-0.15, width=0.5, gain=1.3)
    hit(0.62, "harmonium_motif", "the leitmotif on a lone harmonium in the rain")
    # … interrupted: 2.6 GOSPEL-RADIO STING (Hammond gliss, choir, piano, tambourine) through a little speaker
    rb = Buf()
    t0 = CUE["tract_flip"]
    gl = [(t0 - 0.3 + 0.3 * j / 13, n, 0.05, 0.6) for j, n in enumerate(range(m("D4"), m("D6") + 1, 2))]
    sting = [(t0, n, 0.26, 0.95) for n in ["D4", "F#4", "A4", "B4", "E5"]] + \
            [(t0 + 0.3, n, 0.26, 0.9) for n in ["D4", "G4", "B4", "D5"]] + \
            [(t0 + 0.6, n, 1.35, 0.95) for n in ["D4", "F#4", "A4", "B4", "E5", "A5"]]
    sf2(rb, SF_GU, 0, 16, gl + sting, bus="x", gain=1.6, cc=[(t0 + 0.6, 1, 90)])
    sf2(rb, SF_MS, 0, 52, [(t0 + 0.6, n, 1.3, 0.9) for n in ["D4", "A4", "D5", "F#5"]] +
        [(t0, n, 0.28, 0.85) for n in ["D4", "A4", "D5", "F#5"]], bus="x", gain=1.5)
    piano(rb, t0, ["D2", "D3", "A3", "D4", "F#4"], 0.25, 0.9, gain=1.2, bus="x")
    piano(rb, t0 + 0.6, ["D2", "D3", "A3", "D4", "F#4", "A4"], 1.2, 0.85, gain=1.1, bus="x")
    perc(rb, t0, PERC + "Tamb1-Hit_v2_rr1_Sum.wav", 0.5, 0.0, bus="x")
    perc(rb, t0 + 0.3, PERC + "Tamb1-Hit_v1_rr1_Sum.wav", 0.35, 0.0, bus="x")
    perc(rb, t0 + 0.6, PERC + "Tamb1-Shake_v1_rr1_Sum.wav", 0.45, 0.0, bus="x")
    x = rb.b["x"][S(t0 - 0.4):S(t0 + 2.4)]
    x = radio(x) * np.float32(1.0)
    x = fade(x, 0.0, 0.5)
    b.add("radio", x, t=t0 - 0.4)
    hit(t0, "tract_flip", "gospel-radio sting on the cover")
    # 4.9 the pages riffle: celesta/harp flutter
    for j in range(9):
        n = [74, 76, 78, 81, 83, 86, 88, 90, 93][j]
        I("harp").note(b, "keys", CUE["pages_riffle"] + 0.07 * j, n, None, 0.35, 0.3 - 0.07 * j, gain=0.09, seed=j)
    # 6.4 dive into the page: a warm swell into the comic
    strings(b, 5.4, 1.05, dyn=[(0, 0.05), (1.0, 0.4)], att=0.3, rel=0.4, v1=["A4", "D5"], va=["F#4"], vc=["D3"],
            gain=0.5)
    harp_gliss(b, 6.0, CUE["dive_in"], "D4", "A6", vel=0.4, gain=0.8)
    hit(CUE["dive_in"], "dive_in", "harp gliss lands as the comic comes alive")
    # Sunday-school piano (D major, 3/4, the hymn's pulse) — very soft, thins out under every line
    bt3 = 0.6
    prog = ["D", "G", "D", "A", "D", "G", "A", "D", "D", "G", "D", "A", "Bm", "G", "A", "D"]
    t, k = CUE["dive_in"] + 0.1, 0
    while t < 18.3 and k < len(prog) * 3:
        ch = prog[(k // 3) % len(prog)]
        if k % 3 == 0:
            piano(b, t, [root(ch, "C2") + 12], 0.9, 0.4, gain=3.2)
        else:
            piano(b, t, tones(ch, "F#3", 3), 0.4, 0.3, gain=2.2)
        t += bt3
        k += 1
    # 10.09 'YELLOW FLAGS': the only colour passes — celesta shimmer
    tf = CUE["flag_passes"]
    sf2(b, SF_MS, 0, 8, [(tf + 0.06 * j, n, 0.5, 0.45) for j, n in enumerate(["A5", "D6", "F#6", "A6"])], pan=0.3)
    glass_note(b, tf, "A6", 1.5, gain=db(-28), pan=0.35)
    hit(tf, "flag_passes", "celesta shimmer: the flag passes")
    # "LAZY and ENTITLED HIPPY": a cheeky bassoon hiccup after the confession
    tb = LINES["S01"]["end"] + 0.05
    I("bsn_stac").note(b, "ww", tb, "A2", None, 0.5, PAN["bsn"], gain=0.4)
    I("bsn_stac").note(b, "ww", tb + 0.16, "D2", None, 0.55, PAN["bsn"], gain=0.45)


# ================================================================================================ t02
def t02(b):
    """Every Word Was Wrong (18.46–43.29): church-social stride piano, the organ stab on WRONG!!, silent-film
    tremolo, a Byzantine ison for the icon, heavenly choir on ESSENCE, a pharma-ad bed for the footnote, the J-pop
    kira-kira for the anime snap, a tiptoe for the conspiracy, and the harp-gliss flashback ripple."""
    ta = CUE["preacher_appears"]
    # he is suddenly there: organ 'dun!' + pizz
    organ(b, ta, 0.45, [m("D3"), m("F3"), m("A3"), m("D4")], loud=True, pedal="D2", gain=0.7, att=0.01, rel=0.3)
    I("cb_pizz").note(b, "str", ta, "D2", None, 0.7, PAN["cb"], gain=0.5)
    hit(ta, "preacher_appears", "organ 'dun!'")
    # church-social stride piano (G major, 2/4 at 120) under P01, stopping for the punchline
    bt = 0.5
    prog = [("G", "G1"), ("D", "A1"), ("G", "G1"), ("C", "C2"), ("G", "D2"), ("D", "A1")]
    t, k = 19.0, 0
    while t < 21.6:
        ch, bs = prog[(k // 2) % len(prog)]
        piano(b, t, [m(bs) + 12] if k % 2 == 0 else tones(ch, "B3", 3), 0.3, 0.4, gain=2.0)
        if k % 2 == 1:
            I("vln_pizz").note(b, "str", t, tones(ch, "B4", 1)[0], None, 0.35, PAN["vln"], gain=0.2, seed=k)
        t += bt
        k += 1
    # 22.12 WRONG!! — dramatic organ stab (full organ, 32' pedal) + timpani + low brass
    tw = CUE["wrong_burst"]
    organ(b, tw, 1.3, [m(n) for n in ["D3", "F3", "A3", "D4", "F4", "A4", "D5"]], loud=True, pedal="D2", gain=1.0,
          att=0.005, rel=0.9)
    y = sub_drone(hz(m("D1")), 1.6, amp_bp=[(0, 1), (1.2, 0.6), (1.6, 0)], harm=0.4)
    b.add("org", y, t=tw, gain=db(-22))
    brass(b, tw, None, "stac", 0.95, tbn=["D2", "A2"], tuba=["D2"], hn=["D4", "F4", "A4"], gain=0.8)
    timp(b, tw, "D2", 1.0)
    crash(b, tw, "ff", 0.8)
    hit(tw, "wrong_burst", "dramatic organ stab on WRONG!!")
    # 23.22 WHAT? HOW CAN THIS BE??? — silent-film tremolo (diminished) crescendo
    strings(b, 23.1, 2.05, "trem", dyn=[(0, 0.2), (1.9, 0.45), (2.05, 0.1)], att=0.05, rel=0.2,
            v1=["G#4", "B4"], va=["D4", "F4"], vc=["G#2"], gain=0.45)
    # 25.2 ICON: Byzantine ison — low male 'ooh' drone on D + soft organ + a bell toll
    ti = CUE["icon"]
    sf2(b, SF_MS, 0, 53, [(ti, "D3", 6.9, 0.55), (ti, "A3", 6.9, 0.45), (ti + 2.4, "D2", 4.5, 0.4)], bus="org",
        gain=1.4, width=0.8)
    sf2(b, SF_MS, 0, 14, [(ti, "D4", 3.0, 0.5), (ti + 2.62, "A3", 3.0, 0.35)], bus="keys", pan=0.2, gain=1.3)
    organ(b, ti, 6.9, [m("D3"), m("A3")], loud=False, pedal="D2", gain=0.5, att=0.6, rel=0.8)
    hit(ti, "icon", "Byzantine ison + bell")
    # 29.29 ESSENCE — heavenly choir radiance
    te = CUE["essence"]
    sf2(b, SF_MS, 0, 52, [(te, n, 2.2, 0.8) for n in ["A4", "D5", "F#5", "A5", "D6"]], bus="org", gain=1.6)
    strings(b, te, 2.0, dyn=[(0, 0.5), (1.0, 0.35), (2.0, 0.0)], att=0.02, rel=0.3, v1=["A5", "D6"],
            v2=["F#5"], gain=0.5)
    harp_gliss(b, te, te + 0.5, "D5", "D7", vel=0.55, gain=1.0)
    for n in ["D6", "A6", "D7"]:
        glass_note(b, te, n, 2.4, gain=db(-24), pan=0.2)
    hit(te, "essence", "heavenly choir radiance")
    # 31.06–33.3 the ** footnote at pharma-ad speed: a bland pharma-commercial bed (nylon guitar, pad, glock)
    td = CUE["disclaimer"]
    ev, t, k = [], td, 0
    arp = ["G3", "D4", "G4", "B4", "D5", "B4", "G4", "D4", "C4", "E4", "G4", "C5", "E5", "C5", "G4", "E4"]
    while t < 33.28:
        ev.append((t, arp[k % 16], 0.3, 0.5))
        t += 0.14
        k += 1
    sf2(b, SF_GU, 0, 24, ev, bus="keys", pan=-0.2, gain=1.2)
    sf2(b, SF_GU, 0, 89, [(td, n, 1.1, 0.5) for n in ["G3", "B3", "D4"]] +
        [(td + 1.12, n, 1.1, 0.5) for n in ["C4", "E4", "G4"]], bus="syn", gain=1.0)
    I("glock").note(b, "keys", td, "D6", None, 0.4, 0.4, gain=0.05)
    cut_after(b, 33.32, 0.05)
    # 33.75 ANIME: kira-kira
    tk = CUE["anime"]
    perc(b, tk - 0.02, "Miscellania Raw/Misc 2/glock_glisses/glock_fx_up_pentatonic_med_01.wav", 0.35, 0.3, dur=0.6)
    sf2(b, SF_GU, 0, 98, [(tk, n, 0.9, 0.8) for n in ["D6", "F#6", "A6", "C#7"]], pan=0.35, gain=1.3)
    sf2(b, SF_MS, 0, 8, [(tk + 0.05 * j, n, 0.4, 0.7) for j, n in enumerate(["A6", "C#7", "E7", "F#7", "A7"])], pan=-0.3)
    sf2(b, SF_GU, 0, 112, [(tk, "D7", 0.8, 0.8)], pan=0.1, gain=1.2)
    brass(b, tk, None, "stac", 0.6, tpt=["F#5", "A5", "C#6"], gain=0.35)
    hit(tk, "anime", "J-pop kira-kira sparkle sting")
    # sweat drop '...' — a falling pizz plink
    tsw = W("S04", "is") + 0.42
    for j, n in enumerate(["A5", "E5", "A4"]):
        I("vln_pizz").note(b, "str", tsw + 0.09 * j, n, None, 0.4, 0.3, gain=0.3, seed=j)
    # 36.75 P05 conspiratorial tiptoe: pizz + low clarinet, G minor
    t, k = 36.75, 0
    lines_ = ["G2", "D3", "Bb2", "D3", "C3", "D3", "A2", "D3"]
    while t < 40.0:
        I("vc_pizz").note(b, "str", t, lines_[k % 8], None, 0.5, PAN["vc"], gain=0.65, seed=k)
        t += 0.4
        k += 1
    I("cl_sus").note(b, "ww", 37.95, "Bb3", 1.1, 0.4, PAN["cl"], gain=0.4, dyn=[(0, 0.2), (1.1, 0.35)], att=0.2, rel=0.3)
    I("cl_sus").note(b, "ww", 39.1, "A3", 0.9, 0.4, PAN["cl"], gain=0.4, dyn=[(0, 0.3), (0.9, 0.15)], att=0.1, rel=0.3)
    # 40.03 P06 warm swell into the flashback
    strings(b, 40.0, 1.2, dyn=[(0, 0.08), (0.9, 0.35), (1.2, 0.2)], att=0.3, rel=0.5, v1=["A4", "D5"], va=["F#4"],
            vc=["D3"], gain=0.5)
    # 41.34 FLASHBACK: the harp-gliss ripple (up, down, up), with a wavy detuned pad
    tfb = CUE["flashback"]
    harp_gliss(b, tfb, tfb + 0.45, "D4", "D7", vel=0.6, gain=1.3)
    sf2(b, SF_MS, 0, 8, [(tfb, "D6", 0.6, 0.7), (tfb, "A5", 0.6, 0.6)], pan=0.2, gain=1.4)
    harp_gliss(b, tfb + 0.5, tfb + 0.95, "D5", "D7", vel=0.45, gain=0.9, down=True)
    harp_gliss(b, tfb + 1.0, tfb + 1.5, "D4", "A6", vel=0.4, gain=0.8)
    for j, n in enumerate(["D4", "F#4", "A4", "E5"]):
        y = supersaw(hz(m(n)), 2.1, voices=4, spread=0.12, cutoff=1500, q=0.7, drift=0.012, seed=60 + j)
        y *= envelope([(0, 0), (0.4, 0.8), (1.6, 0.6), (2.1, 0)], len(y))[:, None]
        b.add("syn", stereo_place(y, -0.4 + 0.27 * j, 0.9), t=tfb, gain=db(-30))
    hit(tfb, "flashback", "harp-gliss flashback ripple")


# ================================================================================================ t03
E8 = 0.2  # Glass machine eighth (150 BPM eighths); bars of 8 eighths = 1.6 s from 43.29


def t03(b):
    """In the Beginning (43.29–67.46): a Philip-Glass arpeggio machine (organ, marimba, flute; Dm–Bb–Gm–A additive
    patterns) for the op-art genesis, which darkens into storm music; the stench motif (tuba/bassoon), the
    hurricane swell (59.34), the downpour hit on RAIN (62.95), and a thinning storm under the Ranger's radio."""
    tg = CUE["genesis"]
    bars = ["Dm", "Bb", "Gm", "A", "Dm", "Bb", "Gm", "A", "Dm", "Bb", "Gm", "A", "Dm", "Bb", "Gm", "A"]
    pats = [[0, 1, 2, 1, 0, 1, 2, 1], [0, 1, 2, 3, 2, 1, 0, 1], [0, 1, 2, 0, 1, 2, 0, 1], [0, 2, 1, 3, 0, 2, 1, 3]]
    hit(tg, "genesis", "the arpeggio machine starts")
    org_ev, fl_ev = [], []
    t_end = 56.2
    k = 0
    while True:
        t = tg + k * E8
        if t >= t_end:
            break
        bar = k // 8
        ch = bars[bar % 16]
        pat = pats[(bar // 2) % 4]
        tn = tones(ch, "D4", 4)
        n = tn[pat[k % 8]]
        vol = 0.5 + 0.15 * ((k % 8) in (0, 3, 6))
        org_ev.append((t, n, E8 * 0.9, vol))
        if t > 47.0:  # marimba doubles an octave down, additive 3+3+2 accents
            if (k % 8) in (0, 3, 6, 7):
                I("marimba").note(b, "keys", t, n - 12, None, 0.55, 0.35, gain=0.14, seed=k)
        if t > 51.0 and k % 2 == 1:
            fl_ev.append((t, tones(ch, "A5", 3)[(k // 2) % 3], E8 * 1.6, 0.45))
        if k % 8 == 0:  # bass: whole-bar cello/organ pedal
            strings(b, t, 8 * E8 + 0.02, dyn=[(0, 0.3), (1.6, 0.28)], att=0.05, rel=0.2, vc=[root(ch, "C2") + 12],
                    cb=[root(ch, "C2")] if t > 53.0 else [], gain=0.45)
        k += 1
    sf2(b, SF_GU, 0, 16, org_ev, bus="keys", pan=-0.2, width=0.7, gain=2.6)
    for (t, n, d, v) in fl_ev:
        I("fl_stac").note(b, "ww", t, n, None, v, PAN["fl"] + 0.3, gain=0.3)
    # 55.5 → the engraving: the machine darkens into storm (spiccato D minor ostinato under P09)
    t, k = CUE["stench"] - 10 * E8, 0  # grid lands on the stench (57.339) and the hurricane (59.339)
    while t < 59.3:
        if abs(t - CUE["stench"]) < 0.25:  # the ostinato rests for the stench stab
            t += E8
            k += 1
            continue
        I("vc_spic").note(b, "str", t, "D3" if k % 4 != 3 else "C#3", None, 0.45 + 0.1 * (t - 55.4), PAN["vc"], gain=0.4,
                          seed=k)
        if k % 2 == 0:
            I("cb_spic").note(b, "str", t, "D2", None, 0.5, PAN["cb"], gain=0.4, seed=k)
        t += E8
        k += 1
    # 57.34 STENCH: the stink motif — tuba + bassoon lurching chromatic sneer, trombone 'pew' slide
    ts = CUE["stench"]
    for j, (dt, n, d) in enumerate([(0.38, "Eb2", 0.3), (0.72, "D2", 0.22), (0.98, "C#2", 0.5)]):  # after the stab
        I("tuba_sus").note(b, "brs", ts + dt, n, d, 0.6, PAN["tuba"], gain=0.6, att=0.02, rel=0.1,
                           bend=lambda x: -0.3 * np.clip(x / 0.3, 0, 1))
        I("bsn_sus").note(b, "ww", ts + dt, m(n) + 12, d, 0.6, PAN["bsn"], gain=0.5, att=0.02, rel=0.1,
                          bend=lambda x: -0.4 * np.clip(x / 0.25, 0, 1))
    I("tbn_sus").note(b, "brs", ts + 1.55, "A3", 0.55, 0.6, PAN["tbn"], gain=0.5, att=0.02, rel=0.1,
                      bend=lambda x: -7.0 * np.clip(x / 0.5, 0, 1) ** 1.6)
    I("tuba_stac").note(b, "brs", ts, "D2", None, 0.9, PAN["tuba"], gain=1.3)
    I("bsn_stac").note(b, "ww", ts, "D3", None, 0.85, PAN["bsn"], gain=1.1)
    I("cl_stac").note(b, "ww", ts, "Eb4", None, 0.9, PAN["cl"], gain=1.0)
    I("tpt_stac").note(b, "brs", ts, "D4", None, 0.85, PAN["tpt"], gain=0.8)
    I("tpt_mute").note(b, "brs", ts + 0.38, "Eb4", 0.5, 0.6, PAN["tpt"], gain=0.35, att=0.01, rel=0.1,
                       bend=lambda x: -1.5 * np.clip(x / 0.45, 0, 1))
    hit(ts, "stench", "stench motif (tuba + bassoon + a muted-trumpet sneer)")
    # 57.9 → 59.34 HURRICANE SWELL: the stink twists into the sky
    th_ = CUE["hurricane"]
    sw = th_ - 57.9
    strings(b, 57.9, sw - 0.03, "trem", dyn=[(0, 0.1), (sw, 0.95)], att=0.2, rel=0.02, v1=["D5", "F5", "A5"],
            va=["A3", "D4"], vc=["D3", "A3"], bend=lambda x: 2.0 * (x / sw) ** 2, gain=0.6)
    brass(b, 58.3, th_ - 58.3 - 0.03, dyn=[(0, 0.1), (th_ - 58.3, 0.9)], att=0.2, rel=0.02, hn=["D4", "F4", "A4"],
          tbn=["D3"], gain=0.6)
    timp_roll(b, 58.0, th_ - 58.03, "D2", [(0, 0.15), (th_ - 58.03, 1.0)], gain=0.9)
    amp = envelope([(0, 0), (1.0, 0.4), (sw, 1.0)], S(sw))
    y = shepard(sw, fbase=hz(m("D1")), n_oct=9, center_oct=4.8, width_oct=1.3, amp=amp,
                rate_fn=lambda x: 0.2 + 0.6 * (x / sw) ** 2)
    b.add("fx", y, t=57.9, gain=db(-17))
    swell_to(b, th_ - 0.02, PERC + "susCymb1-cresc-Short_v1.wav", gain=0.3, after=0.0)
    tutti(b, th_, "Dm", 0.85, gong=True)
    hit(th_, "hurricane", "hurricane swell arrives")
    # the storm: whirling chromatic runs + tremolo, lower under the radio lines
    t, k = th_ + 0.2, 0
    while t < 62.9:
        run = [62, 63, 65, 67, 69, 70, 72, 74, 72, 70, 69, 67, 65, 63]
        I("vln_spic").note(b, "str", t, run[k % 14], None, 0.45, PAN["vln"] + 0.3 * np.sin(k), gain=0.25, seed=k)
        if k % 2 == 0:
            I("vc_spic").note(b, "str", t, "D3", None, 0.5, PAN["vc"], gain=0.35, seed=k)
        t += 0.1
        k += 1
    strings(b, th_ + 0.1, 62.95 - th_ - 0.15, "trem", dyn=[(0, 0.5), (1.5, 0.3), (62.95 - th_ - 0.15, 0.55)],
            att=0.1, rel=0.05, va=["D4", "F4"], vc=["D3"], cb=["D2"], gain=0.5)
    # 62.95 RAIN: downpour hit
    tr = CUE["rain_begins"]
    tutti(b, tr, "Dm", 0.9, gong=False)
    organ(b, tr, 1.4, [m(n) for n in ["D3", "A3", "D4", "F4", "A4"]], loud=True, pedal="D2", gain=0.7, att=0.005,
          rel=0.6)
    harp_gliss(b, tr, tr + 0.6, "D6", "D3", scale=(2, 4, 5, 7, 9, 10, 0), vel=0.55, gain=0.9, down=True)
    hit(tr, "rain_begins", "downpour hit on RAIN")
    # storm bed under G01 (the Ranger), receding into t04
    strings(b, tr + 0.4, 67.46 - tr - 0.4, "trem", dyn=[(0, 0.35), (2.0, 0.18), (67.46 - tr - 0.4, 0.1)], att=0.3,
            rel=0.6, va=["D4", "F4"], vc=["D3"], cb=["D2"], gain=0.45)
    I("bsn_stac").note(b, "ww", LINES["G01"]["end"] + 0.1, "D2", None, 0.45, PAN["bsn"], gain=0.4)  # 'where's my dirt'


# ================================================================================================ t04
def t04(b):
    """The Flag Maker (67.46–105.67): the leitmotif as an organ voluntary + choir, B minor rejection, the 'Shut up!'
    stab, the FLAGS chant machine on the crowd's accelerating grid (87.15 → 95.72), the Devil's Toccata (BWV 565)
    that chokes on 'a… a… a…', the possession glitch (102.42), FLAGS! stabs, hard stop 105.67."""
    tf = CUE["flagmaker"]
    # sacred D major: soft organ + choir 'ooh' + the leitmotif on a flute stop
    for (a, z, ch, bs) in [(tf, 70.1, "D", "D2"), (70.1, 72.0, "G", "G1"), (72.0, 73.66, "D", "D2"),
                           (73.66, 74.73, "A", "A1")]:
        organ(b, a, z - a + 0.03, tones(ch, "D3", 4), pedal=bs, gain=0.6, att=0.3 if a == tf else 0.08, rel=0.3)
    sf2(b, SF_MS, 0, 53, [(tf, "D4", 7.2, 0.4), (tf, "A4", 7.2, 0.35)], bus="org", gain=1.1)
    for (t, d, n) in motif(tf + 0.2, 0.62, 1):
        I("fl_sus").note(b, "ww", t, n, d + 0.03, 0.45, PAN["fl"], gain=0.35, att=0.04, rel=0.2,
                         dyn=[(0, 0.3), (d, 0.35)])
    hit(tf, "flagmaker", "organ voluntary: the leitmotif")
    # 74.73 rejection: B minor, low strings
    strings(b, 74.73, 2.2, dyn=[(0, 0.15), (2.0, 0.3)], att=0.4, rel=0.2, va=["F#3", "B3"], vc=["B2"], cb=["B1"],
            gain=0.5)
    # 76.98 SHUT UP! — stab
    tx = CUE["shut_up"]
    organ(b, tx, 0.5, [m(n) for n in ["C3", "Eb3", "G3", "C4", "Eb4"]], loud=True, pedal="C2", gain=0.8, att=0.004,
          rel=0.35)
    brass(b, tx, None, "stac", 0.9, tbn=["C3", "G3"], hn=["C4", "Eb4", "G4"], gain=0.7)
    timp(b, tx, "D2", 0.8, gain=0.8)
    taiko(b, tx, "fff", gain=0.5, sticks=True)
    perc(b, tx, V1P + "drums/snare/drum3_marching/snare3_rimshot_ff_1.wav", 0.3, 0.1)
    hit(tx, "shut_up", "stab on 'Shut up!'")
    # 78.02 P12 solemn, then 82.74 M02 sacred again
    for (a, z, ch, bs) in [(78.0, 80.6, "Em", "E2"), (80.6, 82.7, "A", "A1"), (82.7, 85.0, "D", "D2"),
                           (85.0, 87.12, "G", "G1")]:
        organ(b, a, z - a + 0.03, tones(ch, "D3", 4), pedal=bs, gain=0.5, att=0.1, rel=0.25)
    strings(b, 82.7, 4.4, dyn=[(0, 0.12), (2, 0.2), (4.4, 0.28)], att=0.5, rel=0.2, v1=["F#5"], va=["A4"], gain=0.4)
    # 87.153 FLAGS FLOOD: the chant machine locked to the crowd's grid (126 → 154 BPM, 20 beats to the peak)
    beats = GRID["chant"]["beats"]
    harm = ["Dm", "Dm", "Bb", "A", "Dm"]
    for i, t in enumerate(beats[:20]):
        bar = i // 4
        ch = harm[bar]
        prog = i / 20
        nxt = beats[i + 1]
        d = nxt - t
        # organ pedal pulse on every beat, manual chord on beats 1 & 3
        I("organ_pedal").note(b, "org", t, root(ch, "C2"), d * 0.8, 0.8, 0.0, gain=0.25 + 0.15 * prog, att=0.005,
                              rel=0.05)
        if i % 2 == 0:
            organ(b, t, d * 1.8, tones(ch, "D4", 3), loud=prog > 0.5, gain=0.35 + 0.35 * prog, att=0.005, rel=0.08)
        # spiccato eighths, taiko on downbeats, snare from bar 3
        for h in (0, 0.5):
            I("vc_spic").note(b, "str", t + h * d, root(ch, "C3"), None, 0.5 + 0.3 * prog, PAN["vc"], gain=0.4, seed=i)
            I("vla_spic").note(b, "str", t + h * d, tones(ch, "A3", 2)[int(h * 2)], None, 0.4 + 0.3 * prog, PAN["vla"],
                               gain=0.3, seed=i + 7)
        if i % 4 == 0:
            taiko(b, t, "f" if prog < 0.5 else "ff", gain=0.45 + 0.35 * prog)
            timp(b, t, root(ch, "D2") if root(ch, "D2") <= 50 else root(ch, "D2") - 12, 0.6 + 0.3 * prog, gain=0.7)
        if bar >= 2:
            perc(b, t, V1P + "drums/snare/drum3_marching/snare3_f_%d.wav" % (1 + i % 2), 0.12 + 0.2 * prog, 0.1)
            perc(b, t + 0.5 * d, V1P + "drums/snare/drum3_marching/snare3_p_%d.wav" % (1 + i % 4), 0.08 + 0.12 * prog,
                 0.1)
        if bar >= 3:  # horns: the chant's two-note 'FLA-GS' call, rising
            brass(b, t, d * 0.9, dyn=[(0, 0.5 + 0.4 * prog), (d * 0.9, 0.4)], att=0.01, rel=0.05,
                  hn=[tones(ch, "A4", 1)[0]], gain=0.5)
    hit(beats[0], "flags_flood", "chant machine starts on the crowd's grid")
    tp = CUE["flood_peak"]
    swell_to(b, tp - 0.01, PERC + "susCymb1-cresc-Short_v1.wav", gain=0.35, after=0.0)
    tutti(b, tp, "Dm", 1.0, gong=True)
    organ(b, tp, 1.55, [m(n) for n in ["D3", "F3", "A3", "D4", "F4", "A4", "D5"]], loud=True, pedal="D2", gain=0.9,
          att=0.005, rel=0.2)
    hit(tp, "flood_peak", "the wall of FLAGS: tutti")
    for t in [x for x in beats[21:] if x < CUE["devil"] - 0.05]:  # frenzy continues at 154 BPM to the Devil
        taiko(b, t, "ff", gain=0.6, sticks=True)
        I("vc_spic").note(b, "str", t, "D3", None, 0.8, PAN["vc"], gain=0.45)
        I("organ_pedal").note(b, "org", t, "D2", 0.3, 0.8, 0.0, gain=0.4, att=0.005, rel=0.05)
    cut_after(b, CUE["devil"] - 0.005, 0.03)
    # 97.30 THE DEVIL: Toccata in D minor (BWV 565), full organ, doubled at the octave below
    td = CUE["devil"]
    tri = 0.07
    toc = [(0.0, tri, "A5"), (tri, tri, "G5"), (2 * tri, 0.75, "A5"),
           (1.05, 0.12, "G5"), (1.17, 0.12, "F5"), (1.29, 0.12, "E5"), (1.41, 0.12, "D5"), (1.53, 0.2, "C#5"),
           (1.78, 0.9, "D5")]
    for (dt, d, n) in toc:
        for o in (0, -12):
            I("organ_loud").note(b, "org", td + dt, m(n) + o, d + 0.02, 0.8, 0.1 if o else -0.1, gain=0.85, att=0.002,
                                 rel=0.25)
    I("organ_pedal").note(b, "org", td + 1.78, "D2", 1.0, 0.8, 0.0, gain=0.9, att=0.01, rel=0.4)
    timp(b, td, "D2", 0.8, gain=0.9)
    perc(b, td, PERC + "susCymb1-hitstick_mp_rr1.wav", 0.3, 0.2)
    hit(td, "devil", "Toccata in D minor (BWV 565)")
    # second statement (8vb) under 'The Flags are nothing but a…'
    t2 = 98.95
    for (dt, d, n) in [(0.0, tri, "A4"), (tri, tri, "G4"), (2 * tri, 0.4, "A4"), (0.6, 0.12, "E4"), (0.72, 0.12, "F4"),
                       (0.84, 0.2, "C#4"), (1.06, 0.55, "D4")]:
        I("organ_loud").note(b, "org", t2 + dt, n, d + 0.02, 0.8, 0.0, gain=0.45, att=0.004, rel=0.2)
    # 'a… a… a…': the organ tries the third statement and chokes — each time lower, weaker, detuned
    for j, ta in enumerate([W("V01", "a", 0), W("V01", "a", 1), W("V01", "a", 2)]):
        det = -0.3 * j
        for (dt, d, n) in [(0.0, tri, "A3"), (tri, tri, "G3"), (2 * tri, 0.32 - 0.05 * j, "A3")]:
            I("organ_loud").note(b, "org", ta + dt, m(n) - 2 * j, d + 0.02, 0.8, 0.0, gain=0.5 - 0.08 * j, att=0.004,
                                 rel=0.08, bend=lambda x, det=det: det * np.clip(x / 0.3, 0, 1))
        hit(ta, f"devil_a_{j + 1}", "organ chokes on 'a…'")
    # 102.42 POSSESSION: the diminished chord, a shriek, glitch; then FLAGS! stabs
    tgl = CUE["devil_glitch"]
    rs = riser(0.5, 800, 12000, q=2.0, curve=3.0, seed=21)
    b.add("fx", rs, i0=S(tgl) - len(rs), gain=db(-20))
    for n in ["D2", "C#3", "E3", "G3", "Bb3", "C#4", "E4", "G4", "Bb4"]:
        I("organ_loud").note(b, "org", tgl, n, 0.38, 0.8, 0.0, gain=0.22, att=0.003, rel=0.05)
    sf2(b, SF_MS, 0, 52, [(tgl, n, 0.4, 1.0) for n in ["C#5", "G5", "Bb5", "E6"]], bus="org", gain=1.6)
    brass(b, tgl, None, "stac", 1.0, tbn=["C#3", "G3"], hn=["E4", "Bb4"], tpt=["G5", "C#6"], gain=0.8)
    timp(b, tgl, "D2", 1.0)
    crash(b, tgl, "ff", 0.9)
    hit(tgl, "devil_glitch", "possession: diminished full organ + shriek + glitch")
    for j, tflag in enumerate([W("V02", "flags", 1), W("V02", "flags", 2), W("V02", "flags", 3)]):
        organ(b, tflag, 0.3, [m(n) for n in ["D3", "F3", "A3", "D4", "F4", "A4"]], loud=True, pedal="D2", gain=0.75,
              att=0.003, rel=0.08)
        taiko(b, tflag, "ff", gain=0.6, sticks=True)
        hit(tflag, f"devil_flags_{j + 1}", "organ stab on FLAGS!")
    # the held, glitching D minor to the hard stop
    organ(b, 104.1, 105.67 - 104.1, [m(n) for n in ["D3", "F3", "A3", "D4", "F4", "A4", "D5"]], loud=True,
          pedal="D2", gain=0.7, att=0.02, rel=0.02)
    stutter(b, tgl + 0.02, tgl + 0.42, seg=0.05, reps=2, shrink=0.8)
    stutter(b, 104.9, 105.52, seg=0.09, reps=2, shrink=0.8)
    cut_after(b, 105.67, 0.004)
    hit(105.67, "hard_stop", "HARD STOP", kind="cut")


# ================================================================================================ t05
def t05(b):
    """And It Worked (105.67–122.29): one celesta note on the white card; a hopeful pizzicato march (the leitmotif)
    for finding and moving the flags; triumph on AND IT WORKED; the Seraphlag's ascent; the rain freeze (everything
    stops but one suspended chord); the firmament: full choir + organ glory."""
    tw = CUE["some_listened"]
    sf2(b, SF_MS, 0, 8, [(tw, "A5", 2.0, 0.5)], pan=0.1, gain=1.4)
    glass_note(b, tw, "A6", 3.0, gain=db(-30), pan=0.1)
    hit(tw, "some_listened", "white card: a single celesta note")
    # 108.79 Found one! → hopeful pizzicato march, D major, 2/4 at 112 BPM
    t0 = CUE["found_one"]
    bt = 60 / 112
    tend = CUE["it_worked"] - 0.03
    t, k = t0, 0
    ch_seq = ["D", "D", "A", "D", "G", "D", "A", "D"]
    while t < tend - 0.01:
        ch = ch_seq[(k // 4) % 8]
        if k % 2 == 0:
            I("cb_pizz").note(b, "str", t, root(ch, "C2") + (7 if k % 4 == 2 else 0), None, 0.6, PAN["cb"], gain=0.45,
                              seed=k)
        else:
            for j, n in enumerate(tones(ch, "F#4", 2)):
                I("vla_pizz").note(b, "str", t, n, None, 0.45, PAN["vla"], gain=0.3, seed=k + j)
        if k % 4 == 1:
            perc(b, t, PERC + "Triangle3-HitM_v1_rr1_Sum.wav", 0.12, 0.4)
        t += bt / 2
        k += 1
    for (t, d, n) in motif(t0 + 2 * bt, bt / 2, 1):
        if t < tend:
            I("cl_stac" if t < CUE["found_one"] + 1.3 else "fl_stac", ).note(b, "ww", t, n, None, 0.5, 0.0, gain=0.4)
    brass(b, W("H02", "move", 0), 0.5, dyn=[(0, 0.4), (0.5, 0.5)], att=0.02, rel=0.1, hn=["D4", "F#4"], gain=0.45)
    brass(b, W("H02", "move", 1), 0.6, dyn=[(0, 0.5), (0.6, 0.6)], att=0.02, rel=0.05, hn=["E4", "A4"], gain=0.5)
    hit(t0, "found_one", "pizzicato march begins")
    # 111.86 AND IT WORKED: brass fanfare + timpani
    ti = CUE["it_worked"]
    tutti(b, ti, "D", 0.75, crash_on=True)
    brass(b, ti, 0.9, dyn=[(0, 0.9), (0.9, 0.6)], att=0.0, rel=0.4, tpt=["D5", "F#5", "A5"], hn=["D4", "A4"],
          tbn=["D3", "A3"], gain=0.7)
    hit(ti, "it_worked", "triumphant fanfare")
    # 113.15 → 116.90 the Seraphlag carries the ALL YELLOW FLAG up the effigy
    tu = CUE["flag_on_effigy"]
    a = 113.15
    for j in range(18):
        tj = a + (tu - a - 0.2) * j / 17
        I("harp").note(b, "keys", tj, [62, 66, 69, 74, 78, 81][j % 6] + 12 * (j // 6), None, 0.4, -0.5 + j / 17, gain=0.2,
                       seed=j)
    strings(b, a, tu - a, dyn=[(0, 0.15), (tu - a, 0.55)], att=0.4, rel=0.1, v1=["A4", "D5"], va=["F#4"], vc=["D3"],
            gain=0.5, bend=None)
    sf2(b, SF_MS, 0, 52, [(a + 0.5, n, tu - a - 0.4, 0.55) for n in ["D4", "A4", "D5"]], bus="org", gain=1.1)
    for (t, d, n) in motif(a + 0.3, 0.33, 0):
        I("hn_sus").note(b, "brs", t, n, d + 0.03, 0.5, PAN["hn"], gain=0.45, att=0.03, rel=0.2, dyn=[(0, 0.4), (d, 0.45)])
    brass(b, tu, 1.2, dyn=[(0, 0.7), (1.2, 0.4)], att=0.0, rel=0.4, hn=["D4", "F#4", "A4"], gain=0.6)
    strings(b, tu, 1.4, dyn=[(0, 0.6), (1.4, 0.3)], att=0.0, rel=0.4, v1=["F#5", "A5"], va=["D4"], vc=["D3"], cb=["D2"],
            gain=0.55)
    timp(b, tu, "D2", 0.6, gain=0.6)
    I("glock").note(b, "keys", tu, "D6", None, 0.6, 0.3, gain=0.06)
    hit(tu, "flag_on_effigy", "arrival chord")
    # 118.35 'and the rain stopped' — soft rising strings, then 118.92 FREEZE: everything stops but one chord
    strings(b, 118.3, 118.92 - 118.3, dyn=[(0, 0.2), (0.6, 0.4)], att=0.2, rel=0.02, v1=["A5"], va=["D5"], gain=0.4)
    cut_after(b, CUE["rain_freeze"], 0.015)
    tz = CUE["rain_freeze"]
    tr = CUE["firmament_reveal"]
    for j, n in enumerate(["A5", "D6", "E6", "F#6", "A6"]):
        y = glass(hz(m(n)), 3.8, att=0.004, decay=6.0, beat=0.3)
        y = y * envelope([(0, 1), (tr - tz, 0.85), (3.8, 0.0)], len(y))
        b.add("space", stereo_place(np.stack([y, y], 1), -0.6 + 0.3 * j, 1.0), t=tz, gain=db(-24))
    strings(b, tz, 3.0, dyn=[(0, 0.35), (3.0, 0.2)], att=0.0, rel=0.8, v1=["A5", "E6"], gain=0.35)
    hit(tz, "rain_freeze", "everything stops; a suspended high chord")
    # 119.56 FIRMAMENT: full choir + organ glory (the leitmotif in trumpets)
    organ(b, tr, 122.6 - tr, [m(n) for n in ["D3", "A3", "D4", "F#4", "A4", "D5"]], loud=True, pedal="D2", gain=0.75,
          att=0.02, rel=1.0)
    sf2(b, SF_MS, 0, 52, [(tr, n, 122.4 - tr, 0.85) for n in ["D4", "A4", "D5", "F#5", "A5"]], bus="org", gain=1.5)
    strings(b, tr, 122.5 - tr, dyn=[(0, 0.8), (1.0, 0.6), (122.5 - tr, 0.3)], att=0.0, rel=0.8, v1=["A5", "D6"],
            v2=["F#5"], va=["A4", "D4"], vc=["D3", "A3"], cb=["D2"], gain=0.6)
    tutti(b, tr, "D", 0.8, gong=False)
    for (t, d, n) in motif(tr + 0.35, 0.42, 0):
        brass(b, t, d + 0.03, dyn=[(0, 0.75), (d, 0.7)], att=0.02, rel=0.3, tpt=[n + 12], hn=[n], gain=0.55)
    swell_to(b, tr - 0.01, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.25, after=0.0)
    harp_gliss(b, tr, tr + 0.6, "D4", "D7", vel=0.6, gain=1.1)
    hit(tr, "firmament_reveal", "full choir + organ glory")


# ================================================================================================ t06
def t06(b):
    """The Firmament Opened (122.29–142.04): the burn-night drums (123.49 BPM, locked with the crowd), the fire swell
    on ignite, the drums dropping dead when the flames reach the Flag — its ascent a fragile rising line — the
    apocalyptic crack (135.58) and the catastrophic deluge (137.12) to the smash cut at 142.04."""
    burn = GRID["burn"]["beats"]
    ti, tfl = CUE["ignite"], CUE["flames_reach_flag"]
    strings(b, 122.29, tfl - 122.29, dyn=[(0, 0.15), (3.0, 0.25), (tfl - 122.29, 0.45)], att=1.0, rel=0.02,
            vc=["D2", "A2"], cb=["D2"], gain=0.5)
    for i, t in enumerate(burn):
        pre = t < ti - 0.01
        lv = 0.35 + 0.3 * i / len(burn) if pre else 0.75 + 0.25 * (t - ti) / (tfl - ti)
        taiko(b, t, "f" if pre else "ff", gain=0.5 * lv)
        perc(b, t + 0.2429, V1P + "drums/other/ethnic/giant/hand/EthnicLargeHand_hit_mp_%d.wav" % (1 + i % 3),
             0.35 * lv, -0.3)
        if i % 2 == 1 or not pre:
            perc(b, t, V1P + "drums/tenor/tenor_lower/tenor_%s_%d.wav" % ("mf" if pre else "ff", 1 + i % 3), 0.28 * lv,
                 0.3)
        if not pre:  # low brass fire ostinato: D-D-F-E
            n = ["D2", "D2", "F2", "E2"][i % 4]
            I("tbn_stac").note(b, "brs", t, m(n) + 12, None, 0.7, PAN["tbn"], gain=0.45 * lv)
            I("tuba_stac").note(b, "brs", t, n, None, 0.7, PAN["tuba"], gain=0.4 * lv)
    hit(burn[0], "burn_drums", "burn drums begin")
    # fire swell into ignite
    sw = ti - 124.0
    strings(b, 124.0, sw - 0.02, "trem", dyn=[(0, 0.1), (sw, 0.9)], att=0.2, rel=0.02, v1=["D5", "A5"], va=["F4", "A4"],
            bend=lambda x: 1.0 * (x / sw) ** 2, gain=0.55)
    brass(b, 124.3, ti - 124.32, dyn=[(0, 0.1), (ti - 124.3, 0.85)], att=0.2, rel=0.02, hn=["D4", "F4", "A4"],
          gain=0.55)
    r = riser(sw, 200, 6000, q=1.4, curve=2.0, seed=31)
    b.add("fx", r, i0=S(ti) - len(r), gain=db(-20))
    swell_to(b, ti - 0.01, PERC + "susCymb1-cresc-Median_v1.wav", gain=0.3, after=0.0)
    tutti(b, ti, "Dm", 0.8)
    hit(ti, "ignite", "fire swell lands on 'burned'")
    strings(b, ti, tfl - ti - 0.02, "trem", dyn=[(0, 0.5), (tfl - ti, 0.7)], att=0.0, rel=0.02, v1=["D5", "F5"],
            va=["A3", "D4"], gain=0.45)
    cut_after(b, tfl - 0.004, 0.02)  # the drums drop dead: the miracle
    # 129.73 the Flag lifts free: a fragile ascending line (solo violin + celesta + glass), the leitmotif rising
    hit(tfl, "flames_reach_flag", "drums drop out; the fragile ascending line begins")
    line = motif(tfl, 0.36, 1)
    for (t, d, n) in line:
        I("vln_solo").note(b, "str", t, n, d + 0.05, 0.35, 0.15, gain=0.4, dyn=[(0, 0.25), (d, 0.3)], att=0.03, rel=0.3)
        sf2(b, SF_MS, 0, 8, [(t, n + 12, 0.5, 0.35)], pan=0.3, gain=1.1)
    t_end_line = line[-1][0] + line[-1][1]
    for j, n in enumerate(["A4", "D5", "F#5", "A5", "D6"]):  # … and keeps climbing out of sight
        tj = t_end_line + 0.3 * j
        I("vln_solo").note(b, "str", tj, n, 0.34 if j < 4 else 1.2, 0.3, 0.15, gain=0.35 - 0.03 * j,
                           dyn=[(0, 0.25), (0.3, 0.2)], att=0.03, rel=0.3)
        sf2(b, SF_MS, 0, 8, [(tj, m(n) + 12, 0.4, 0.3)], pan=0.3, gain=1.0)
    for j, n in enumerate(["A5", "D6", "F#6", "A6", "D7", "F#7"]):
        glass_note(b, tfl + 0.2 + 0.8 * j, n, 2.5, gain=db(-28), pan=0.2 * np.sin(j), decay=1.2)
    y = sub_drone(hz(m("D5")), 3.0, amp_bp=[(0, 0), (1.0, 1), (3.0, 0)], harm=0.05)
    b.add("space", y, t=tfl, gain=db(-40))
    # 132.37 P19: dread gathers — low tremolo cluster rising
    tc = CUE["firmament_crack"]
    strings(b, 132.3, tc - 132.33, "trem", dyn=[(0, 0.08), (tc - 132.3, 0.75)], att=0.5, rel=0.02,
            va=["D3", "Eb3"], vc=["C#3", "D3"], cb=["D2"], bend=lambda x: 1.5 * (x / 3.3) ** 2, gain=0.5)
    brass(b, 134.0, tc - 134.03, dyn=[(0, 0.1), (tc - 134.0, 0.8)], att=0.3, rel=0.02, tbn=["D2", "Eb2"], hn=["A3", "Bb3"],
          gain=0.5)
    # 135.58 CRACK: brass cluster + organ + glass shatter
    tutti(b, tc, "Dm", 0.8, gong=False)
    brass(b, tc, None, "stac", 1.0, hn=["C#4", "D4", "Eb4", "G#4"], tbn=["C3", "C#3"], gain=0.7)
    organ(b, tc, 0.9, [m(n) for n in ["C#3", "D3", "G#3", "A3", "Eb4"]], loud=True, pedal="D2", gain=0.6, att=0.004,
          rel=0.4)
    rng = np.random.default_rng(66)
    t = tc
    for q in range(90):
        age = (t - tc) / 1.2
        glass_note(b, t, int(rng.integers(79, 103)), 1.2, gain=db(-24 - 10 * age) * rng.uniform(0.5, 1),
                   pan=float(np.clip(rng.normal(0, 0.3 + 0.7 * age), -1, 1)), decay=0.5, bus="space")
        t += rng.exponential(0.004 + 0.02 * age)
        if t > tc + 1.3:
            break
    hit(tc, "firmament_crack", "apocalyptic crack")
    # 137.12 DELUGE: catastrophic tutti — full organ, choir, brass, timpani, a falling Shepard (the waters above)
    tdl = CUE["deluge"]
    end = CUE["deluge"] + (142.04 - tdl)
    tutti(b, tdl, "Dm", 1.0, gong=True)
    organ(b, tdl, end - tdl, [m(n) for n in ["D3", "F3", "A3", "D4", "F4", "A4", "D5"]], loud=True, pedal="D2",
          gain=0.8, att=0.005, rel=0.02)
    sf2(b, SF_MS, 0, 52, [(tdl, n, end - tdl, 1.0) for n in ["D4", "F4", "A4", "D5", "F5"]], bus="org", gain=1.5)
    strings(b, tdl, end - tdl, "trem", dyn=[(0, 1.0), (1.0, 0.7), (end - tdl, 0.9)], att=0.0, rel=0.02,
            v1=["D5", "F5", "A5"], v2=["D5"], va=["A3", "D4", "F4"], vc=["D3", "A2"], cb=["D2"],
            bend=lambda x: -2.0 * np.clip(x / 4.5, 0, 1), gain=0.6)
    brass(b, tdl, end - tdl, dyn=[(0, 1.0), (0.5, 0.7), (end - tdl, 0.85)], att=0.0, rel=0.02,
          tbn=["D2", "A2", "D3"], tuba=["D2"], hn=["D4", "F4", "A4"], tpt=["D5", "F5"], gain=0.6)
    timp_roll(b, tdl + 0.3, end - tdl - 0.3, "D2", [(0, 0.6), (end - tdl - 0.3, 1.0)], gain=0.9)
    amp = envelope([(0, 0), (0.6, 0.8), (end - tdl, 1.0)], S(end - tdl))
    y = shepard(end - tdl, rate_oct_s=-0.35, fbase=hz(m("D1")), n_oct=9, center_oct=4.6, width_oct=1.3, amp=amp)
    b.add("fx", y, t=tdl, gain=db(-15))
    hit(tdl, "deluge", "catastrophic deluge tutti")
    cut_after(b, 142.04, 0.006)
    hit(142.04, "smash_cut", "smash cut", kind="cut")


# ================================================================================================ t07
def t07(b):
    """Raised Again (142.04–159.83): 'HOWEVER!' stinger, a snare roll, then a fast-forward Sousa-style march on the
    leitmotif locked to T07Raised's 100 raisings — a snare on EVERY raising, a cymbal on every full-frame hero cut
    (strength 1.0), a '50!' landing and a held breath over the 50-panel page, then the second run to the 100th
    (146.77); then a lonely Americana harmonica + fiddle walk through the empty years, and a warm resolve on the
    unmask (158.8)."""
    th_ = W("P21", "however")
    brass(b, th_, None, "stac", 0.85, tpt=["D5", "F#5", "A5"], hn=["D4", "A4"], tbn=["D3"], gain=0.6)
    timp(b, th_, "D2", 0.7, gain=0.6)
    hit(th_, "however", "HOWEVER! stinger")
    r1, r100 = CUE["raise_1"], CUE["raise_100"]
    perc(b, r1 - 1.0, PERC + "Snare2-rollSN_v3_rr1_Sum.wav", 0.3, 0.1, dur=0.99)
    d = os.path.join(FILM_ROOT, "cache", "foley", "t07_hits_events.json")
    try:
        j = json.load(open(d))
        ev = j["events"] if isinstance(j, dict) else j
        raises = sorted((e["t"], float(e.get("strength", 0.55))) for e in ev if e.get("kind") == "raise")
    except Exception:
        raises = []
    if len(raises) < 10:  # fallback: 100 raisings accelerating geometrically from raise_1 to raise_100
        g = np.geomspace(1.0, 0.035, 99)
        g = g / g.sum() * (r100 - r1)
        raises = [(t, 1.0 if i in (0, 49, 99) else 0.55) for i, t in enumerate(r1 + np.concatenate([[0], np.cumsum(g)]))]
    # the march: every raising gets a snare (quieter when they crowd into a roll); the band (oom-pah, melody =
    # leitmotif) runs on its own clock that follows the raisings but never plays faster than a real band could
    mel = [n for (_, _, n) in motif(0, 1, 1)] * 10
    prog = ["D", "D", "G", "D", "A", "A", "D", "D"]
    t_band, k_band, t_mel, k_mel = -1.0, 0, -1.0, 0
    for i, (t, st) in enumerate(raises):
        lvl = 0.55 + 0.45 * i / len(raises)
        gap = min(t - raises[i - 1][0] if i else 1.0, raises[i + 1][0] - t if i + 1 < len(raises) else 1.0)
        roll = float(np.clip(gap / 0.08, 0.25, 1.0))  # a roll, not a machine gun, when raisings are 5 ms apart
        perc(b, t, PERC + "Snare2-HitSN_v%d_rr%d_Sum.wav" % ([1, 3, 5, 7][min(3, i // 25)], 1 + i % 2),
             0.22 * lvl * roll, 0.1)
        if st >= 0.99:  # full-frame hero cut
            crash(b, t, "mp" if i < 90 else "ff", 0.5 + 0.5 * lvl, dur=0.9)
        if t - t_band >= 0.11:
            ch = prog[(k_band // 4) % 8]
            if k_band % 2 == 0:
                I("tuba_stac").note(b, "brs", t, root(ch, "C2"), None, 0.7, PAN["tuba"], gain=0.45 * lvl, seed=k_band)
                if k_band % 4 == 0:
                    perc(b, t, V1P + "drums/bass/bdrum_f_%d.wav" % (1 + (k_band // 4) % 2), 0.35 * lvl, 0.0)
            else:
                brass(b, t, None, "stac", 0.6, hn=tones(ch, "D4", 2), gain=0.35 * lvl)
            t_band, k_band = t, k_band + 1
        if t - t_mel >= 0.085:
            n = mel[k_mel % len(mel)]
            I("tpt_stac").note(b, "brs", t, n, None, 0.7, PAN["tpt"], gain=0.4 * lvl, seed=k_mel)
            I("picc_stac").note(b, "ww", t, n + 12, None, 0.7, PAN["fl"], gain=0.25 * lvl)
            t_mel, k_mel = t, k_mel + 1
        if i in (0, 49):
            hit(t, f"raise_{i + 1}", "raising (snare + cymbal)")
    # '50!' — the first run lands, and the band holds its breath over the 50-panel page until counting resumes
    t50 = raises[49][0] if len(raises) >= 50 else None
    if t50 is not None and len(raises) > 50 and raises[50][0] - t50 > 0.3:
        brass(b, t50, None, "stac", 0.9, tpt=["D5", "F#5", "A5"], hn=["D4", "A4"], tbn=["D3"], gain=0.6)
        timp(b, t50, "D2", 0.8, gain=0.8)
        hold = raises[50][0] - t50
        strings(b, t50 + 0.05, hold - 0.1, "trem", dyn=[(0, 0.2), (hold - 0.1, 0.45)], att=0.05, rel=0.03,
                v1=["A5"], va=["E4", "A4"], gain=0.4)
        perc(b, raises[50][0] - 0.45, PERC + "Snare2-rollSN_v3_rr1_Sum.wav", 0.25, 0.1, dur=0.44)
    tutti(b, r100, "D", 0.95, gong=False)
    brass(b, r100, 0.9, dyn=[(0, 1.0), (0.9, 0.6)], att=0.0, rel=0.5, tpt=["D5", "F#5", "A5", "D6"], hn=["D4", "A4"],
          tbn=["D3", "A3"], tuba=["D2"], gain=0.7)
    hit(r100, "raise_100", "the 100th raising: tutti")
    # 149.58 the Spirit walks on: harmonica + fiddle, D major → B minor (no fire) → G minor (no Burns)
    ts = CUE["spirit"]
    fid = [(ts, "D4", "A4", 3.0), (152.6, "D4", "A4", 2.9), (155.45, "D4", "F#4", 2.05), (157.5, "D4", "Bb4", 1.3)]
    for (t, lo, hi, d) in fid:
        for n in (lo, hi):
            I("vln_solo").note(b, "str", t, n, d, 0.3, -0.3, gain=0.45, dyn=[(0, 0.2), (d * 0.5, 0.3), (d, 0.2)],
                               att=0.3, rel=0.5)
    har = [(t, n, d * 0.95, 0.55) for (t, d, n) in motif(ts + 0.5, 0.55, 0)]
    har += [(155.45, "F#4", 0.9, 0.5), (156.4, "D4", 0.9, 0.45), (157.5, "D4", 0.6, 0.45), (158.1, "Bb3", 0.65, 0.4)]
    sf2(b, SF_GU, 0, 22, har, bus="ww", pan=0.2, gain=2.4)
    for (t, ch) in [(ts, "D"), (152.6, "G"), (155.45, "Bm"), (157.5, "Gm")]:
        sf2(b, SF_GU, 0, 25, [(t + 0.02 * j, n, 1.5, 0.35) for j, n in enumerate(tones(ch, "D3", 4))], bus="keys",
            pan=-0.3, gain=1.8)
    hit(ts, "spirit", "harmonica + fiddle walk")
    # 158.8 unmask: warm resolve (G minor → D major: the minor plagal)
    tu = CUE["unmask"]
    strings(b, tu, 1.6, dyn=[(0, 0.45), (1.0, 0.35), (1.6, 0.1)], att=0.05, rel=0.6, v1=["F#5", "A5"], va=["D4", "A4"],
            vc=["D3"], cb=["D2"], gain=0.55)
    harp_roll(b, tu, ["D3", "A3", "D4", "F#4", "A4", "D5"], 0.5, 0.0, gain=0.8)
    sf2(b, SF_MS, 0, 8, [(tu, "F#6", 0.8, 0.5)], pan=0.2, gain=1.2)
    hit(tu, "unmask", "warm resolve")


# ================================================================================================ t08
HYMN_BT = 0.525714


def hymn_grid():
    p = os.path.join(FILM_ROOT, "audio", "vocal", "hymn_grid.json")
    try:
        j = json.load(open(p))
        rows = j.get("notes", j)
        return [(r["t"] if "t" in r else 169.09 + r["beat"] * HYMN_BT, r.get("beats", r.get("dur_beats")) * HYMN_BT,
                 r.get("satb") or [r[k] for k in ("S", "A", "T", "B")]) for r in rows]
    except Exception:
        g = [("Just", 0, 1, [63, 58, 55, 51]), ("as", 1, 2, [67, 63, 58, 51]), ("I", 3, 1, [67, 63, 60, 48]),
             ("am", 4, 2, [70, 63, 58, 51]), ("with", 6, 1, [67, 63, 58, 55]), ("out", 7, 1, [65, 62, 58, 46]),
             ("one", 8, 1, [68, 63, 60, 44]), ("Flag", 9, 3, [67, 63, 58, 51]), ("I", 12, 1, [67, 63, 58, 51]),
             ("come", 13, 2, [65, 62, 58, 46]), ("I", 15, 1, [70, 67, 63, 51]), ("come", 16, 3, [67, 63, 58, 51]),
             ("A-", 19, 2, [68, 60, 56, 44]), ("men", 21, 2, [67, 58, 55, 51])]
        return [(169.09 + bb * HYMN_BT, nb * HYMN_BT, satb) for (_, bb, nb, satb) in g]


def t08(b):
    """Blaspheme (159.83–181.17): rain and a grave organ pedal, the thunder-organ on BLASPHEME (166.22), Bb7 lifting
    'What must I do?' into the Vocalist's E-flat altar-call hymn, which the organ doubles exactly; plagal A-men on
    180.13 as the rain stops (glass + harp sunbeam)."""
    organ(b, 159.9, 166.2 - 159.9, [m("D3"), m("A3")], pedal="D2", gain=0.8, att=1.5, rel=0.05)
    strings(b, 163.0, 3.2, "trem", dyn=[(0, 0.05), (3.2, 0.45)], att=0.5, rel=0.02, va=["D3", "F3"], vc=["D2"],
            gain=0.45)
    tb = CUE["blaspheme"]
    organ(b, tb, 1.7, [m(n) for n in ["D3", "F3", "A3", "D4", "F4", "A4", "D5"]], loud=True, pedal="D2", gain=1.0,
          att=0.004, rel=1.0)
    b.add("org", sub_drone(hz(m("D1")), 2.2, amp_bp=[(0, 1), (1.6, 0.5), (2.2, 0)], harm=0.4), t=tb, gain=db(-20))
    tutti(b, tb, "Dm", 0.9, gong=True)
    hit(tb, "blaspheme", "thunder-organ")
    # kneel: B-flat 7 (V of E-flat) under 'What must I do?'
    strings(b, 167.7, 169.09 - 167.7 - 0.12, dyn=[(0, 0.1), (1.0, 0.3), (1.27, 0.15)], att=0.4, rel=0.08, va=["Ab3", "D4"],
            vc=["Bb2"], v1=["F4"], gain=0.45)
    # the hymn (Vocalist's E-flat grid): organ doubles S/A/T/B, pedal 8vb; soft strings on S+B
    for (t, d, satb) in hymn_grid():
        last = t > 180.0
        dd = d + (1.3 if last else 0.0)
        for j, n in enumerate(satb):
            I("organ_soft").note(b, "org", t, n, dd - 0.03, 0.8, -0.2 + 0.13 * j, gain=0.2, att=0.03,
                                 rel=0.9 if last else 0.12)
        I("organ_softped").note(b, "org", t, satb[3] - 12 if satb[3] - 12 >= 36 else satb[3], dd - 0.03, 0.8, 0.0,
                                gain=0.35, att=0.03, rel=0.9 if last else 0.12)
        strings(b, t, dd - 0.03, dyn=[(0, 0.2), (dd, 0.22)], att=0.05, rel=0.9 if last else 0.15, v1=[satb[0]],
                vc=[satb[3]], gain=0.35)
    hit(169.09, "hymn", "organ doubles the altar-call hymn")
    for n in ["Eb3", "Bb3", "Eb4"]:
        I("harp").note(b, "keys", 169.09, n, None, 0.55, PAN["harp"], gain=0.25)
    ta = CUE["amen"]
    harp_roll(b, ta, ["Eb3", "Bb3", "Eb4", "G4", "Bb4", "Eb5"], 0.45, 0.06, gain=0.8)
    for n in ["Eb6", "Bb6", "G6"]:
        glass_note(b, ta, n, 3.0, gain=db(-28), pan=0.3)
    hit(ta, "amen", "plagal A-men lands; sunbeam")


# ================================================================================================ t09
def t09(b):
    """Pass It On (181.17–199.0): E-flat Amen fades; A7 on 'Now go…' → tender D on PASS IT ON (184.43); an endless
    Shepard–Risset rise through the Droste spiral (185.4→191.6) that resolves into the morning; a sitcom sting on the
    callback (194.76); a harp toss; the gospel A-men (G → D) as the tract lands (197.4), ringing on into the loop."""
    tn = W("P30", "now")
    strings(b, tn, 184.43 - tn - 0.02, dyn=[(0, 0.1), (0.8, 0.28)], att=0.4, rel=0.05, va=["G4", "C#4"], vc=["A2"],
            gain=0.45)
    tp = CUE["pass_it_on"]
    strings(b, tp, 1.2, dyn=[(0, 0.4), (1.2, 0.3)], att=0.05, rel=0.6, v1=["F#5", "A5"], va=["D4", "A4"], vc=["D3"],
            cb=["D2"], gain=0.5)
    piano(b, tp, ["D3", "A3", "D4", "F#4", "A4"], 1.5, 0.5, gain=0.9)
    harp_roll(b, tp, ["D4", "F#4", "A4", "D5"], 0.4, 0.06, gain=0.6)
    hit(tp, "pass_it_on", "tender resolve")
    # 185.4 → 191.6 DROSTE: endless Shepard–Risset rise (synth + chromatic tremolo Shepard scale + celesta spiral)
    a, z = CUE["droste_start"], CUE["droste_end"]
    dur = z - a
    amp = envelope([(0, 0), (1.0, 0.6), (dur - 0.4, 1.0), (dur, 0.0)], S(dur))
    y = shepard(dur, fbase=hz(m("D1")), n_oct=9, center_oct=4.8, width_oct=1.3, amp=amp,
                rate_fn=lambda x: 0.12 + 0.2 * (x / dur))
    b.add("fx", y, t=a, gain=db(-14))
    t, k = a, 0
    while t < z - 0.1:
        step = 0.5 - 0.2 * (t - a) / dur
        pc = (2 + k) % 12
        for o in range(2, 7):
            n = 12 * (o + 1) + pc
            g = np.exp(-0.5 * ((n - 64) / 11.0) ** 2)
            key = "vc_trem" if n < 55 else ("vla_trem" if n < 67 else "vln_trem")
            if g < 0.05 or n > 86:
                continue
            I(key).note(b, "str", t, n, step * 2.0, 0.7, 0.0, gain=0.35 * g, dyn=[(0, 0.1), (step, 0.5), (step * 2, 0.1)],
                        att=0.05, rel=0.1, seed=k * 7 + o)
        t += step
        k += 1
    ev = []
    for j in range(int(dur / 0.125)):
        tj = a + 0.125 * j
        ev.append((tj, [62, 66, 69, 74, 78, 81, 86][j % 7] + (j // 7) % 2, 0.2, 0.3 + 0.2 * (j / (dur / 0.125))))
    sf2(b, SF_MS, 0, 8, ev, pan=0.0, width=1.0, gain=0.9)
    hit(a, "droste_start", "Shepard rise begins")
    I("glock").note(b, "keys", a, "A6", None, 0.6, 0.0, gain=0.07)
    I("harp").note(b, "keys", a, "D4", None, 0.6, -0.3, gain=0.3)
    # 191.6 resolve into the morning: bright D major
    strings(b, z, 2.0, dyn=[(0, 0.5), (2.0, 0.15)], att=0.03, rel=0.6, v1=["A5", "D6"], va=["F#4", "A4"], vc=["D3"],
            cb=["D2"], gain=0.5)
    piano(b, z, ["D2", "D3", "A3", "D4", "F#4", "A4", "D5"], 1.8, 0.55, gain=0.9)
    I("glock").note(b, "keys", z, "A6", None, 0.5, 0.3, gain=0.06)
    hit(z, "droste_end", "morning chord")
    # 194.76 CALLBACK: sitcom sting — slap bass riff + rimshot + horns
    tc = CUE["callback"]
    sl = [(tc, "D2", 0.12, 1.0), (tc + 0.13, "D3", 0.1, 0.9), (tc + 0.26, "A2", 0.12, 0.9), (tc + 0.39, "C3", 0.12, 0.9),
          (tc + 0.52, "D3", 0.35, 1.0)]
    sf2(b, SF_GU, 0, 36, sl, bus="keys", pan=0.0, gain=1.6)
    perc(b, tc, V1P + "drums/snare/drum3_marching/snare3_rimshot_f_1.wav", 0.3, 0.1)
    perc(b, tc + 0.52, V1P + "drums/snare/drum3_marching/snare3_rimshot_ff_1.wav", 0.3, 0.1)
    brass(b, tc + 0.52, None, "stac", 0.7, hn=["D4", "F#4", "A4"], tpt=["C5"], gain=0.4)
    hit(tc, "callback", "sitcom sting")
    # 196.3 toss: harp gliss down
    harp_gliss(b, CUE["tract_toss"], CUE["tract_toss"] + 0.35, "D7", "D5", vel=0.45, gain=0.8, down=True)
    # 197.4 THE GOSPEL AMEN: 'A-' (G) → 'men' (D, lands with the tract) — Hammond + pipe organ + piano + choir,
    # ringing on through 199.0 into the film's first frame (the buffer wraps)
    tl_ = CUE["tract_lands"]
    ta = tl_ - 0.78
    g_, d_ = [m(n) for n in ["G3", "B3", "D4", "G4"]], [m(n) for n in ["D3", "A3", "D4", "F#4", "A4"]]
    sf2(b, SF_GU, 0, 16, [(ta, n, 0.76, 0.75) for n in g_] + [(tl_, n, 3.2, 0.8) for n in d_], bus="keys", gain=1.2,
        cc=[(ta, 1, 60)])
    organ(b, ta, 0.76, g_, pedal="G1", gain=0.6, att=0.04, rel=0.1)
    organ(b, tl_, 3.4, d_, pedal="D2", gain=0.65, att=0.02, rel=1.2)
    piano(b, ta, ["G2", "G3", "B3", "D4", "G4"], 0.7, 0.5, gain=0.8)
    piano(b, tl_, ["D2", "D3", "A3", "D4", "F#4", "A4"], 2.5, 0.55, gain=0.85)
    sf2(b, SF_MS, 0, 52, [(ta, n, 0.76, 0.55) for n in ["G4", "B4", "D5"]] +
        [(tl_, n, 3.0, 0.6) for n in ["F#4", "A4", "D5"]], bus="org", gain=1.1)
    hit(tl_, "tract_lands", "gospel A-men lands with the tract")


SECTIONS = dict(t01=t01, t02=t02, t03=t03, t04=t04, t05=t05, t06=t06, t07=t07, t08=t08, t09=t09)

# ================================================================================================ master
BUSES = {
    #        gain(dB)  reverb     wet    eq
    "str":   (0.0, "hall", 0.30, [("hp", 32), ("hs", 7000, 1.0)]),
    "brs":   (-1.0, "hall", 0.38, [("hp", 40), ("pk", 2500, -1.5, 0.8)]),
    "ww":    (-2.0, "hall", 0.33, [("hp", 80)]),
    "perc":  (-1.0, "hall", 0.30, [("hp", 55)]),  # Foley owns the sub (< 60 Hz)
    "keys":  (-2.0, "hall", 0.38, [("hp", 60)]),
    "org":   (-2.0, "cathedral", 0.45, [("hp", 28), ("pk", 550, -2.5, 0.7)]),
    "syn":   (-2.0, "plate", 0.28, [("hp", 25)]),
    "fx":    (-3.0, "hall", 0.22, [("hp", 40)]),
    "space": (-2.0, "void", 0.75, [("hp", 60)]),
    "radio": (0.0, "room", 0.12, []),
}
# reverb tails do not cross these cuts: (t, fade-out seconds after t, buses or None = all)
CUTS = [(33.32, 0.05, ["keys", "syn"]), (97.30, 0.03, None), (105.67, 0.0, None),
        (118.917, 0.02, ["str", "brs", "ww", "perc", "keys", "org", "syn", "fx"]), (129.729, 0.03, ["perc", "brs"]),
        (142.04, 0.0, None)]


def dialogue_env():
    fr = 1000
    n = int(DUR * fr)
    g = np.zeros(n)
    for l in TLJ["lines"]:
        a, z = int((l["start"] - 0.06) * fr), int((l["end"] + 0.12) * fr)
        g[max(0, a):min(n, z)] = 1.0
    out = np.zeros(n)
    s = 0.0
    ca, cr = 1 - np.exp(-1 / (0.06 * fr)), 1 - np.exp(-1 / (0.25 * fr))
    for i in range(n):
        s += (g[i] - s) * (ca if g[i] > s else cr)
        out[i] = s
    return np.interp(np.arange(N) / SR, np.arange(n) / fr, out).astype(np.float32)


def reverb_loop(x, name, wet, cuts):
    """reverb per segment between cuts; the tail of the last segment wraps to the film start (the loop)."""
    from music_lib import IR as _IR
    ir = _IR(name)
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
        lim = z - s0 + (S(fades[i]) if fades[i] else 0)
        if fades[i] is not None:
            fd = S(fades[i])
            w = w[:lim].copy()
            if fd > 0:
                w[-fd:] *= np.linspace(1, 0, fd)[:, None] ** 2
        e = min(N, s0 + len(w))
        y[s0:e] += w[:e - s0]
        if s0 + len(w) > N:  # the loop: tail continues at 0.0
            ov = w[N - s0:]
            y[:len(ov)] += ov[:N]
    return y * np.float32(wet)


def glue(x, lufs_in, above=6.0, ratio=3.0):
    """gentle 'glue' compression of the score's peaks (hits) relative to its own loudness; 15 ms attack keeps the
    attacks (and therefore the sync) intact."""
    import pedalboard as pb
    c = pb.Compressor(threshold_db=float(lufs_in + above), ratio=ratio, attack_ms=15.0, release_ms=280.0)
    return c(x.T.astype(np.float32), SR).T.copy()


def master(sec_files):
    t0 = time.time()
    buses = {}
    for f in sec_files:
        z = np.load(f)
        for key in z.files:
            if key.endswith("__i0"):
                continue
            i0 = int(z[key + "__i0"])
            x = z[key]
            buses.setdefault(key, np.zeros((N, 2), np.float32))
            e = min(N, i0 + len(x))
            buses[key][i0:e] += x[:e - i0]
    mix = np.zeros((N, 2), np.float32)
    for key, x in buses.items():
        g, rv, wet, fx = BUSES[key]
        if fx:
            x = eq(x, *fx)
        y = x * db(g)
        if rv:
            cuts = sorted(c for c in CUTS if c[2] is None or key in c[2])
            y = y + reverb_loop(x, rv, wet, cuts) * db(g)
        mix += y
        print(f"  bus {key:6s} peak {np.abs(y).max():.3f}", flush=True)
    d = dialogue_env()
    lo = butter(mix, "lowpass", 250, 2, zero_phase=True)
    hi = butter(mix, "highpass", 4000, 2, zero_phase=True)
    mid = mix - lo - hi
    mix = lo + hi + mid * (1 - (1 - db(-5)) * d)[:, None]
    L0 = lufs(mix)
    mix = glue(mix, L0)
    L0 = lufs(mix)
    mix *= db(-19.0 - L0)
    mix, gmin = limiter(mix, -1.2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sf.write(OUT, mix.astype(np.float32), SR, subtype="FLOAT")
    print(f"master: {time.time() - t0:.1f}s  LUFS pre {L0:.1f} -> {lufs(mix):.2f}  limiter min gain "
          f"{20 * np.log10(gmin):.1f} dB  true peak {20 * np.log10(true_peak(mix)):.2f} dBTP", flush=True)


def render_section(name):
    global HITS
    HITS = []
    t0 = time.time()
    b = LoopBuf()
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
    master(files)


if __name__ == "__main__":
    main()

"""THE UNBABELING — the score (owned by Composer).

  python tools/music_score.py                 # render every section + master -> audio/stems/music.wav
  python tools/music_score.py --only s03,s04  # re-render those sections (others from cache) + master
  python tools/music_score.py --master        # master only (all sections from cache)

Leitmotif ("the Flag"): the hymn's soprano line  A F# B G C# A | D  (scale degrees 5-3-6-4-7-5-1:
falling thirds that climb by step and land on the tonic) over D | D | G | Em | A | A | D.
It is transformed per scene: innocent glass (s01), minor + detuned (s02), TV presidential pastiche (s03),
mystical arpeggio / reverent glass (s04), Dorian brass (s05a), splintered canon in twelve keys (s06),
solo violin -> tutti converging onto the hymn downbeat (s07a), the hymn itself + augmented brass (s07b),
pizzicato quote and warm horn statement (s08).

Every section writes into named buses of a Buf (global time). Sections are cached in
audio/music/cache/sec_<id>.npz so iteration on one section does not re-render the rest.
Hits placed on picture are logged to audio/music/cues.json (used by tools/music_verify.py).
"""
import os as _os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ.setdefault(_k, "4")  # shared machine: stay polite
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from music_lib import (DUR, N, ROOT, SF_GU, SF_MS, SR, SF2, Buf, IR, S, Sampler, _resample, adsr, butter,  # noqa
                       conv_stereo, db, envelope, eq, fade, fm_bell, glass, hz, m, noise, oneshot, pan_gains,
                       reverb_full, reverse_swell, riser, shepard, stereo_place, sub_drone, supersaw, svf,
                       tape_stop, tt, _saw_bank)

CACHE = os.path.join(ROOT, "audio", "music", "cache")
OUT = os.path.join(ROOT, "audio", "stems", "music.wav")
TLJ = json.load(open(os.path.join(ROOT, "timeline.json")))
CUE = {c["id"]: c["t"] for c in TLJ["cues"]}
LINES = {l["id"]: l for l in TLJ["lines"]}
HITS = []  # (t, id, desc) — logged per section


def scene_events(scene, kind=None, contains=None, fallback=()):
    """times of visual hits exported by a scene owner (cache/foley/<scene>_hits_events.json), so the score
    follows the picture; `fallback` is used when the export is missing or empty."""
    p = os.path.join(ROOT, "cache", "foley", f"{scene}_hits_events.json")
    try:
        d = json.load(open(p))
        ev = d["events"] if isinstance(d, dict) else d
        ts = sorted(e["t"] for e in ev if (kind is None or e.get("kind") == kind)
                    and (contains is None or contains in str(e.get("desc", ""))))
    except Exception:
        ts = []
    return ts if ts else list(fallback)


def W(lid, word, k=0):
    """start time of the k-th occurrence of `word` in line lid."""
    ws = [w for w in LINES[lid]["words"] if w["w"].lower().strip(",.!?:") == word.lower()]
    return ws[k]["s"]


def hit(t, name, desc="", kind="onset"):
    """log a sync point: kind 'onset' (attack lands on t) or 'cut' (music drops away at t)."""
    HITS.append(dict(t=round(float(t), 4), id=name, desc=desc, kind=kind))


# ------------------------------------------------------------------------------------------------ theme
TH_MAJ = ["A4", "F#4", "B4", "G4", "C#5", "A4", "D5"]
TH_MIN = ["A4", "F4", "Bb4", "G4", "C5", "A4", "D5"]
TH_DOR = ["A4", "F4", "B4", "G4", "C5", "A4", "D5"]
TH_CHORDS = ["D", "D", "G", "Em", "A", "A", "D"]


def th(kind="maj", octv=0, tr=0):
    src = dict(maj=TH_MAJ, min=TH_MIN, dor=TH_DOR)[kind]
    return [m(n) + 12 * octv + tr for n in src]


CH = {"D": (2, [0, 4, 7]), "Dm": (2, [0, 3, 7]), "G": (7, [0, 4, 7]), "Em": (4, [0, 3, 7]), "A": (9, [0, 4, 7]),
      "Bm": (11, [0, 3, 7]), "Bb": (10, [0, 4, 7]), "C": (0, [0, 4, 7]), "F": (5, [0, 4, 7]), "Gm": (7, [0, 3, 7]),
      "F#m": (6, [0, 3, 7]), "Bbm": (10, [0, 3, 7]), "A7": (9, [0, 4, 7, 10]), "Cm": (0, [0, 3, 7]),
      "B": (11, [0, 4, 7]), "F#": (6, [0, 4, 7]), "Asus4": (9, [0, 5, 7]), "Dadd9": (2, [0, 4, 7, 2])}


def tones(ch, lo, n):
    """first n chord tones >= lo (midi)"""
    root, iv = CH[ch]
    pcs = {(root + i) % 12 for i in iv}
    out, x = [], m(lo)
    while len(out) < n:
        if x % 12 in pcs:
            out.append(x)
        x += 1
    return out


def root(ch, lo):
    r = CH[ch][0]
    x = m(lo)
    while x % 12 != r:
        x += 1
    return x


# ------------------------------------------------------------------------------------------------ players
_INS = {}


def I(key, **kw):
    k = (key, tuple(sorted(kw.items())))
    if k not in _INS:
        _INS[k] = Sampler(key, **kw)
    return _INS[k]


PAN = dict(vln=-0.5, vln2=-0.22, vla=0.12, vc=0.36, cb=0.52, hn=-0.32, tpt=0.24, tbn=0.4, tuba=0.52,
           timp=0.0, harp=-0.58, cel=0.42, glock=0.3, fl=-0.14, ob=0.1, cl=-0.06, bsn=0.18, org=0.0)
STR_KEYS = dict(v1=("vln", "vln"), v2=("vln", "vln2"), va=("vla", "vla"), vc=("vc", "vc"), cb=("cb", "cb"))


def strings(b, t, dur, art="sus", vel=0.6, dyn=None, att=0.08, rel=0.7, bus="str", gain=1.0, bend=None, **parts):
    """parts: v1=[notes], v2=[...], va=[...], vc=[...], cb=[...]  (art: sus|trem|spic|pizz)"""
    for part, notes in parts.items():
        key, pk = STR_KEYS[part]
        from music_lib import index
        name = f"{key}_{art}" if f"{key}_{art}" in index() else f"{key}_sus"
        inst = I(name)
        for j, n in enumerate(notes):
            one = art in ("spic", "pizz")
            inst.note(b, bus, t, n, None if one else dur, vel, PAN[pk] + 0.06 * j, gain=gain * (0.8 if key == "cb" else 1.0),
                      dyn=dyn if not one else None, att=att if not one else 0.0, rel=rel, bend=bend,
                      seed=int(t * 997) + j + hash(part) % 7)
            if key == "cb" and not one:  # solo bass doubled -> small section
                inst.note(b, bus, t + 0.012, n, dur, vel * 0.9, PAN[pk] - 0.1, gain=gain * 0.6, dyn=dyn, att=att,
                          rel=rel, bend=bend, seed=int(t * 131) + 5)


def brass(b, t, dur, art="sus", vel=0.6, dyn=None, att=0.03, rel=0.5, bus="brs", gain=1.0, bend=None, **parts):
    """parts: tpt=[..], hn=[..], tbn=[..], tuba=[..]; art sus|stac (tpt also 'mute')"""
    for part, notes in parts.items():
        key = f"{part}_{art}" if not (part == "tpt" and art == "mute") else "tpt_mute"
        inst = I(key)
        one = art == "stac"
        for j, n in enumerate(notes):
            inst.note(b, bus, t, n, None if one else dur, vel, PAN[part] + 0.07 * j - 0.035 * len(notes), gain=gain,
                      dyn=None if one else dyn, att=0.0 if one else att, rel=rel, bend=bend, seed=int(t * 577) + j)


def timp(b, t, note, vel=0.8, gain=1.0, bus="perc"):
    I("timp").note(b, bus, t, note, None, vel, 0.0, gain=gain * 0.55, width=0.7)


def timp_roll(b, t, dur, note, dyn, gain=1.0, bus="perc"):
    I("timp_roll").note(b, bus, t, note, dur, 0.8, 0.0, gain=gain * 0.5, dyn=dyn, att=0.05, rel=0.25, width=0.7)


def harp_roll(b, t, notes, vel=0.5, spread=0.03, gain=1.0, bus="keys", pan=None):
    for j, n in enumerate(sorted(m(x) for x in notes)):
        I("harp").note(b, bus, t + j * spread, n, None, vel, PAN["harp"] if pan is None else pan, gain=gain * 0.28,
                       seed=j)


def harp_gliss(b, t0, t1, lo, hi, scale=(2, 4, 6, 9, 11), vel=0.5, gain=1.0, down=False, bus="keys"):
    ns = [x for x in range(m(lo), m(hi) + 1) if x % 12 in scale]
    if down:
        ns = ns[::-1]
    for j, n in enumerate(ns):
        tj = t0 + (t1 - t0) * (j / max(1, len(ns) - 1)) ** 1.15
        I("harp").note(b, bus, tj, n, None, vel * (0.8 + 0.2 * j / len(ns)), PAN["harp"] + 0.4 * j / len(ns),
                       gain=gain * 0.22, seed=j)


PERC = "Percussion/"
V1P = "VSCO 1 Percussion/"


def perc(b, t, f, gain=1.0, pan=0.0, width=0.8, bus="perc", **kw):
    return oneshot(b, bus, t, f, gain=gain, pan=pan, width=width, **kw)


def crash(b, t, level="ff", gain=1.0, pan=0.15, dur=None):
    f = {"ff": V1P + "varMetal/Cymbals/clash/crash_hit_ff_loose.wav",
         "fff": V1P + "varMetal/Cymbals/clash/crash_hit_fff_loose.wav",
         "mp": V1P + "varMetal/Cymbals/clash/crash_hit_mp_loose.wav",
         "pp": V1P + "varMetal/Cymbals/clash/crash_hit_pp_loose.wav"}[level]
    perc(b, t, f, gain * 0.28, pan, dur=dur)


DYN_ORDER = ["ppp", "pp", "p", "mp", "mf", "f", "ff", "fff"]


def taiko(b, t, level="ff", gain=1.0, pan=0.0, sticks=False, rr=1):
    """giant ethnic drum (mallet = taiko-like boom, sticks = drier crack). Nearest available dynamic."""
    d = V1P + ("drums/other/ethnic/giant/sticks/" if sticks else "drums/other/ethnic/giant/mallet/")
    pre = "EthnicLargeSticks_hit_" if sticks else "EthnicLargeMallet_hit_"
    have = sorted({f[len(pre):].split("_")[0] for f in os.listdir(os.path.join(ROOT, "assets", "sf2", "vsco2", d))
                   if f.startswith(pre)}, key=DYN_ORDER.index)
    lv = min(have, key=lambda h: abs(DYN_ORDER.index(h) - DYN_ORDER.index(level)))
    fs = sorted(f for f in os.listdir(os.path.join(ROOT, "assets", "sf2", "vsco2", d)) if f.startswith(pre + lv + "_"))
    perc(b, t, d + fs[(rr - 1) % len(fs)], gain * 0.9, pan, 0.6)


def swell_to(b, t_peak, f, gain=1.0, after=0.15, pan=0.0, bus="perc", cut=True):
    """align the loudest point of a crescendo sample (e.g. cymbal roll) to t_peak; cut `after` s later."""
    from music_lib import load_wav
    x, sr = load_wav(os.path.join("assets", "sf2", "vsco2", f))
    env = np.convolve(np.abs(x).max(1), np.ones(441) / 441, "same")
    pk = int(np.argmax(env))
    y = _resample(x, 0.0, np.array([sr / SR]), int(len(x) * SR / sr) - 4)
    pk48 = int(pk * SR / sr)
    if cut:
        k = S(after)
        y = y[:pk48 + k].copy()
        y[pk48:] *= np.linspace(1, 0, len(y) - pk48)[:, None] ** 2
    y = stereo_place(y, pan, 0.9)
    b.add(bus, y, i0=S(t_peak) - pk48, gain=gain)


def cut_after(b, t, fade=0.03):
    """cut-off: everything already written to b fades out over `fade` s ending at t, silence after t."""
    a, z = S(t - fade), S(t)
    for k in b.b:
        b.b[k][a:z] *= (np.cos(np.linspace(0, np.pi / 2, z - a)) ** 2)[:, None]
        b.b[k][z:] = 0


def sf2part(path, bank, preset, gain=1.0):
    return SF2(path, bank, preset, gain=gain)


def glass_note(b, t, n, dur=3.0, gain=1.0, pan=0.0, bus="space", decay=None, att=0.004):
    y = glass(hz(m(n)), dur, att=att, decay=decay or dur * 0.6)
    b.add(bus, stereo_place(np.stack([y, y], 1), pan, 1.0), t=t, gain=gain)


def bell_note(b, t, n, dur=3.5, gain=1.0, pan=0.0, bus="keys", ratio=3.5, index=1.6, decay=1.4):
    y = fm_bell(hz(m(n)), dur, ratio=ratio, index=index, decay=decay)
    b.add(bus, stereo_place(np.stack([y, y], 1), pan, 1.0), t=t, gain=gain)


def tutti(b, t, ch, level=1.0, perc_on=True, sus=0.0, crash_on=True, tpt=True, gong=False, low=True, crash_dur=None):
    """orchestral hit. sus>0 adds sustained strings+brass for sus seconds."""
    v = 0.55 + 0.45 * level
    g = level
    brass(b, t, None, "stac", v, tpt=tones(ch, "D5", 3) if tpt else [], hn=tones(ch, "D4", 4),
          tbn=[root(ch, "C3"), root(ch, "C3") + 7] if low else [], tuba=[root(ch, "C2")] if low else [], gain=g)
    strings(b, t, None, "spic", v, v1=tones(ch, "D5", 2), v2=tones(ch, "A4", 2), va=tones(ch, "D4", 2),
            vc=[root(ch, "C3"), root(ch, "C3") + 7], cb=[root(ch, "C2")] if low else [], gain=g)
    if sus > 0:
        dyn = [(0, 0.9 * level), (0.4, 0.55 * level), (sus, 0.45 * level)]
        strings(b, t, sus, "sus", dyn=dyn, att=0.0, v1=tones(ch, "A5", 2), v2=tones(ch, "D5", 2),
                va=tones(ch, "F4", 2), vc=[root(ch, "C3")], cb=[root(ch, "C2")] if low else [], gain=g * 0.9)
        brass(b, t, sus, "sus", dyn=[(0, 0.8 * level), (0.5, 0.45 * level), (sus, 0.4 * level)], att=0.0,
              hn=tones(ch, "D4", 3), tbn=[root(ch, "C3")], gain=g * 0.8)
    if perc_on:
        r = root(ch, "C2")
        timp(b, t, r if r >= 38 else r + 12, 0.95 * level + 0.05, gain=g)
        taiko(b, t, "ff" if level > 0.7 else "f", gain=g * 0.55)
        if crash_on:
            crash(b, t, "ff" if level > 0.6 else "mp", gain=g, dur=crash_dur)
        if gong:
            perc(b, t, PERC + "gongHit_fff.wav", 0.5 * g, 0.0)


# ================================================================================================ s01
def s01(b):
    """Rotating Into Existence (0–21): cosmic drone, glass stars, rotating crystal arpeggio, Shepard turn,
    the Flag theme (innocent glass/celesta, 12.0→14.2), echoes, reverse suck, title hit 16.0."""
    rng = np.random.default_rng(101)
    # sub hum D1 + dark supersaw pad (open fifths)
    y = sub_drone(hz(m("D1")), 16.2, amp_bp=[(0, 0), (2.5, 0.7), (10, 0.8), (12, 1.0), (15.2, 0.7), (15.95, 0), (16.2, 0)])
    b.add("syn", y, t=0.0, gain=db(-17))
    y2 = sub_drone(hz(m("D2")), 16.2, amp_bp=[(0, 0), (3, 0.4), (12, 0.6), (14.2, 1.0), (15.5, 0.6), (15.95, 0), (16.2, 0)],
                   harm=0.1)
    b.add("syn", y2, t=0.0, gain=db(-26))
    for j, (note, pan) in enumerate([("D3", -0.35), ("A3", 0.35), ("E4", 0.0), ("A4", -0.6), ("D5", 0.6)]):
        n = S(15.95)
        cut = envelope([(0, 250), (5.9, 500), (12.0, 1100), (14.2, 2600), (15.95, 1400)], n)
        y = supersaw(hz(m(note)), 15.95, voices=5, spread=0.1, cutoff=cut, q=0.9, drift=0.0015, seed=j)
        y *= envelope([(0, 0), (3.0, 0.45), (5.9, 0.55), (12.0, 0.85), (14.2, 1.0), (15.3, 0.7), (15.95, 0)], len(y))[:, None]
        b.add("syn", stereo_place(y, pan, 0.8), t=0.0, gain=db(-33))
    # stars: sparse glass pings, D major pentatonic, high
    t = 0.5
    while t < 11.6:
        note = int(rng.choice([86, 88, 90, 93, 95, 98, 100, 102, 105]))
        glass_note(b, t, note, 2.6, gain=db(-33) * rng.uniform(0.35, 1.0), pan=rng.uniform(-0.95, 0.95), decay=1.0)
        t += rng.exponential(0.75) + 0.12
    # 2.6 the Crystal condenses: reverse shimmer into 2.6
    shim = sum(glass(hz(m(n)), 2.5, att=0.002, decay=1.5) for n in ["D6", "A6", "E7"])
    rs = reverse_swell(np.stack([shim, shim], 1), IR("void"))
    b.add("space", fade(rs[-S(2.2):], fin=0.3), i0=S(2.6) - S(2.2), gain=db(-22))
    # rotating crystal arpeggio: 4 notes (tesseract), accelerating from 5.9 (W-plane turn), pans in a circle
    t, k = 3.0, 0
    fig = ["D5", "A5", "E6", "F#6"]
    while t < 11.75:
        rate = 2.2 if t < 5.9 else 2.2 + 7.5 * ((t - 5.9) / 6.0) ** 1.6
        lvl = -31 + 6 * np.clip((t - 3.0) / 9.0, 0, 1)
        n = fig[k % 4]
        bell_note(b, t, n, 2.2, gain=db(lvl), pan=0.85 * np.sin(k * np.pi / 2 + t * 0.4), bus="space",
                  ratio=2.0, index=0.9, decay=0.8)
        t += 1.0 / rate
        k += 1
    # Shepard turn 5.9 -> 12.1 ("a direction you cannot point to"), resolving onto D partials at 12.0
    dur = 6.2
    rate_fn = lambda tt_: 0.05 + 0.33 * (tt_ / dur) ** 1.8
    tl = np.arange(S(dur)) / SR
    pos = np.cumsum(rate_fn(tl)) / SR
    start = np.ceil(pos[S(6.1)]) - pos[S(6.1)]
    amp = envelope([(0, 0), (2.0, 0.35), (5.4, 1.0), (6.0, 0.12), (6.1, 0.0), (6.2, 0)], len(tl))
    y = shepard(dur, fbase=hz(m("D1")), n_oct=9, center_oct=5.6, width_oct=1.0, amp=amp, rate_fn=rate_fn,
                start_pos=float(start))
    b.add("space", y, t=5.9, gain=db(-21))
    # ---- the Flag theme, innocent: 6/8 lilt, final D ON the snap (14.2)
    e = (CUE["flag_snap_open"] - CUE["flag_rotates_in"]) / 9.0
    ts = [CUE["flag_rotates_in"] + e * k for k in (0, 2, 3, 5, 6, 8, 9)]
    cel = sf2part(SF_MS, 0, 8, gain=2.2)
    for j, (tj, n) in enumerate(zip(ts, th("maj", 1))):
        last = j == 6
        cel.note(tj, n, 1.2 if last else e * 2, 0.62 if not last else 0.95)
        glass_note(b, tj, n + 12, 3.0 if last else 1.6, gain=db(-24 if not last else -19), pan=0.3 * np.sin(j), decay=1.2)
        I("harp").note(b, "keys", tj, n - 12, None, 0.45, PAN["harp"], gain=0.12, seed=j)
    hit(ts[0], "flag_rotates_in", "theme begins (glass A)")
    hit(ts[-1], "flag_snap_open", "theme lands on D, bloom")
    chords = [("D", ["D3", "A3", "D4", "F#4"]), None, ("G", ["G2", "D3", "G3", "B3"]), ("Em", ["E3", "B3", "E4", "G4"]),
              ("A", ["A2", "E3", "A3", "C#4"]), None, ("D", ["D2", "A2", "D3", "F#3", "A3", "D4"])]
    for j, c in enumerate(chords):
        if c:
            harp_roll(b, ts[j], c[1], 0.45 if j < 6 else 0.7, 0.02, gain=0.9)
    # soft string bed following the harmony (very soft, slow attacks)
    seg = [(ts[0], ts[2], ["D4", "F#4"], "D3"), (ts[2], ts[3], ["D4", "G4"], "G2"), (ts[3], ts[4], ["E4", "G4"], "E3"),
           (ts[4], ts[6], ["E4", "C#4"], "A2")]
    for (a, z, va, vc) in seg:
        strings(b, a, z - a + 0.1, vel=0.3, dyn=[(0, 0.18), (z - a, 0.28)], att=0.25, rel=0.4, va=va, vc=[vc], gain=0.7)
    strings(b, ts[0], ts[6] - ts[0], vel=0.3, dyn=[(0, 0.1), (2.2, 0.3)], att=0.8, rel=0.3, v1=["D6"], gain=0.5)
    # 14.2 bloom
    tb = ts[6]
    strings(b, tb, 1.5, vel=0.5, dyn=[(0, 0.55), (1.0, 0.35), (1.5, 0.2)], att=0.02, rel=0.5, v1=["A5", "D6"],
            v2=["F#5"], va=["A4", "D4"], vc=["D3"], cb=["D2"], gain=0.8)
    harp_gliss(b, tb, tb + 0.42, "D4", "D7", vel=0.55, gain=1.1)
    I("glock").note(b, "keys", tb, "D6", None, 0.8, PAN["glock"], gain=0.16)
    I("glock").note(b, "keys", tb + 0.004, "A6", None, 0.7, PAN["glock"] + 0.2, gain=0.1)
    # "A sign of itself": the last figure reflected endlessly, smaller and higher each time
    fig3 = [m("C#6"), m("A5"), m("D6")]
    for r in range(1, 5):
        for q, n in enumerate(fig3):
            tq = tb + 0.1 + (r - 1) * 0.3 + q * 0.085
            glass_note(b, tq, n + 12 * min(r, 2), 1.4, gain=db(-24 - 5 * r), pan=(-1) ** (r + q) * 0.25 * r, decay=0.7)
    cel.note(tb + 0.6, m("A6"), 0.4, 0.35)
    cel.note(tb + 0.8, m("D7"), 0.6, 0.3)
    cel.to(b, "keys", pan=PAN["cel"], width=0.8)
    # ---- 15.3 -> 16.0 reverse suck into the title slam
    ts_ = CUE["title_slam"]
    chord = sum(fm_bell(hz(m(n)), 2.0, ratio=1.0, index=0.6, decay=1.2) for n in ["D3", "A3", "D4", "F#4", "A4", "D5"])
    rs = reverse_swell(np.stack([chord, chord], 1), IR("bighall"))
    L = S(1.1)
    b.add("fx", rs[-L:] * np.linspace(0, 1, L)[:, None] ** 1.5, i0=S(ts_) - L, gain=db(-18))
    swell_to(b, ts_, PERC + "susCymb1-cresc-Short_v1.wav", gain=0.3, after=0.02)
    # ---- 16.0 TITLE SLAM (tonal tutti; Foley owns the sub/noise)
    tutti(b, ts_, "D", 0.8, sus=0.0)
    hit(ts_, "title_slam", "D major tutti hit")
    I("glock").note(b, "keys", ts_, "D6", None, 0.9, 0.3, gain=0.12)
    strings(b, ts_, 4.3, vel=0.7, dyn=[(0, 0.8), (0.6, 0.45), (3.0, 0.4), (4.3, 0.55)], att=0.0, rel=0.4,
            v1=["A5", "D6"], v2=["F#5", "D5"], va=["A4", "F#4"], vc=["D3", "A3"], cb=["D2"], gain=0.75)
    brass(b, ts_, 4.3, dyn=[(0, 0.7), (0.6, 0.3), (4.3, 0.35)], att=0.0, hn=["D4", "F#4", "A4"], gain=0.6)
    # THE UNBABELING drops letter by letter (Scene01's letter landings): one pluck per letter —
    # T-H-E = the D major triad, U-N-B-A-B-E-L = the Flag theme, I-N-G = the triad an octave up
    cel2 = sf2part(SF_MS, 0, 8, gain=2.0)
    letters = scene_events("s01", "thunk", "title letter",
                           [16.46, 16.534, 16.608, 16.646, 16.72, 16.794, 16.832, 16.906, 16.98, 17.018, 17.092, 17.166, 17.204])
    lp = [m("D5"), m("F#5"), m("A5")] + th("maj", 1) + [m("F#6"), m("A6"), m("D7")]
    for j, (tj, n) in enumerate(zip(letters, lp)):
        cel2.note(tj, n, 0.25 if j < len(lp) - 1 else 1.0, 0.55)
        I("marimba").note(b, "keys", tj, n - 12, None, 0.65, 0.35 - 0.05 * (j % 3), gain=0.11, seed=j)
        I("vln_pizz").note(b, "str", tj, min(n - 12, 86), None, 0.5, PAN["vln"] + 0.04 * (j % 4), gain=0.35, seed=j)
        hit(tj, f"title_letter_{j + 1}", "pluck on title letter landing")
    sw = scene_events("s01", "swipe", None, [17.02])[0]
    harp_gliss(b, sw, sw + 0.5, "A5", "A6", vel=0.35, gain=0.6)
    for j, n in enumerate(th("maj", 2)):
        tj = 18.0 + j * 0.19 + (0.12 if j == 6 else 0)
        cel2.note(tj, n, 0.3 if j < 6 else 0.9, 0.5)
        I("glock").note(b, "keys", tj, n if n >= 79 else n + 12, None, 0.45, 0.3, gain=0.05, seed=j)
        I("vln_pizz").note(b, "str", tj, n - 24, None, 0.5, PAN["vln"], gain=0.4, seed=j)
    cel2.to(b, "keys", pan=PAN["cel"])
    # 20.3 page slides up: harmony darkens D -> Dm, low swell into s02 downbeat 21.0
    strings(b, 20.3, 0.9, vel=0.5, dyn=[(0, 0.35), (0.7, 0.6)], att=0.2, rel=0.3, va=["F4", "A4"], vc=["D3"], gain=0.7)
    brass(b, 20.35, 0.75, dyn=[(0, 0.1), (0.65, 0.65)], att=0.3, rel=0.25, tbn=["D2", "A2"], hn=["D3", "A3"], gain=0.7)
    timp_roll(b, 20.4, 0.62, "D2", [(0, 0.1), (0.6, 0.7)], gain=0.8)


# ================================================================================================ s02
B2 = 0.63  # beat (95.24 BPM); bars at 21.00, 23.52, 26.04, 28.56, 31.08, 33.60


def s02(b):
    """The Age of Slop (21–47.5): the lockstep machines of meaning (D minor ostinato, stabs on
    Nations/Faiths/Philosophies), drop-out at 33.6, the corrupted theme drowned by the slop surge, massive
    impact 39.2, sinking, then feeds multiplying 1→64 under N07, static at 47.5."""
    rng = np.random.default_rng(202)
    t_nat, t_fai, t_phi = W("N04", "Nations"), W("N04", "Faiths"), W("N04", "Philosophies")
    harm = [(21.0, "Dm"), (23.52, "Bb"), (t_nat, "Dm"), (t_fai, "Bb"), (t_phi, "Gm"), (28.56, "A"), (31.08, "Dm"),
            (33.6, None)]

    def chord_at(t):
        c = None
        for (a, ch) in harm:
            if t >= a - 1e-6:
                c = ch
        return c

    # ostinato: celli/basses spiccato 8ths, 3+3+2 accents, following the harmony root
    t = 21.0
    k = 0
    while t < 33.6 - 1e-6:
        ch = chord_at(t)
        r = root(ch, "C2")
        acc = (k % 8) in (0, 3, 6)
        prog = (t - 21.0) / 12.6
        v = (0.55 if acc else 0.32) + 0.25 * prog
        oct_up = (k % 8) in (2, 6)
        I("vc_spic").note(b, "str", t, r + 12 + (12 if oct_up else 0), None, v, PAN["vc"], gain=0.55, seed=k)
        if k % 2 == 0:
            I("cb_spic").note(b, "str", t, r, None, v, PAN["cb"], gain=0.5, seed=k)
        if not acc:
            fifth = tones(ch, "A3", 2)
            I("vla_spic").note(b, "str", t, fifth[k % 2], None, v * 0.8, PAN["vla"], gain=0.35, seed=k)
        t += B2 / 2
        k += 1
    # low brass chord pads per harmony segment (pp -> mp), timpani on bar downbeats
    for i, (a, ch) in enumerate(harm[:-1]):
        z = harm[i + 1][0]
        lv = 0.25 + 0.4 * (a - 21.0) / 12.6
        brass(b, a, z - a + 0.05, dyn=[(0, lv * 0.6), (min(0.4, z - a), lv), (z - a, lv * 0.9)], att=0.15, rel=0.3,
              hn=tones(ch, "F3", 3), tbn=[root(ch, "C2") + 12], tuba=[root(ch, "C2")], gain=0.55)
    for a in (21.0, 23.52, 26.04, 28.56, 31.08):
        timp(b, a, root(chord_at(a), "D2") if root(chord_at(a), "D2") <= 50 else root(chord_at(a), "D2") - 12, 0.45, gain=0.7)
    # the three monoliths light up on their words
    for (tw, ch, name) in [(t_nat, "Dm", "N04_nations"), (t_fai, "Bb", "N04_faiths"), (t_phi, "Gm", "N04_philosophies")]:
        brass(b, tw, None, "stac", 0.75, hn=tones(ch, "D4", 3), tbn=[root(ch, "C2") + 12, root(ch, "C2") + 19],
              tuba=[root(ch, "C2")], gain=0.45)
        rr = root(ch, "D2")
        timp(b, tw, rr if rr <= 50 else rr - 12, 0.65, gain=0.55)
        taiko(b, tw, "mp", gain=0.45, sticks=True)
        strings(b, tw, 1.0, dyn=[(0, 0.6), (0.3, 0.35), (1.0, 0.3)], att=0.0, rel=0.4, v1=tones(ch, "A5", 2),
                va=tones(ch, "D4", 2), gain=0.35)
        hit(tw, name, f"{ch} stab: monolith lights up")
    # 28.56: broadcast rings + lockstep: march snare backbeats, radio pings every beat
    t = 28.56
    k = 0
    while t < 33.6 - 1e-6:
        prog = (t - 28.56) / 5.04
        if k % 2 == 1:
            perc(b, t, V1P + "drums/snare/drum3_marching/snare3_p_%d.wav" % (1 + k % 4), 0.12 + 0.12 * prog, 0.1)
        t += B2
        k += 1
    for k, tp_ in enumerate(scene_events("s02", "pulse", None, [28.56 + B2 * i for i in range(8)])):
        prog = (tp_ - 28.4) / 5.2
        bell_note(b, tp_, "A6" if k % 3 else "D7", 1.2, gain=db(-34 + 6 * prog), pan=(-0.6, 0.6)[k % 2], bus="syn",
                  ratio=1.0, index=0.3, decay=0.25)
    # 31.08 -> 33.6: crescendo (tremolo strings + horn swell) that is CUT at 33.6 ("Then came the slop.")
    strings(b, 31.08, 2.52, "trem", dyn=[(0, 0.2), (2.5, 0.85)], att=0.1, rel=0.02, v1=["D5", "F5"], va=["A3", "D4"],
            vc=["D3"], gain=0.6)
    brass(b, 31.08, 2.52, dyn=[(0, 0.2), (2.5, 0.8)], att=0.2, rel=0.02, hn=["D4", "F4", "A4"], gain=0.6)
    # after the cut: eerie thin tone + low cello, wobbly beige pad fading in
    strings(b, 33.6, 1.4, vel=0.3, dyn=[(0, 0.25), (1.4, 0.3)], att=0.05, rel=0.6, vc=["D2"], gain=0.6)
    t_ = tt(1.6)
    eerie = (np.sin(2 * np.pi * hz(m("Bb6")) * t_) + 0.7 * np.sin(2 * np.pi * (hz(m("Bb6")) + 3.3) * t_)) * \
        envelope([(0, 0), (0.3, 1), (1.6, 0.6)], len(t_))
    b.add("space", np.stack([eerie, eerie * 0.8], 1), t=33.6, gain=db(-36))
    # ---- 34.8 -> 39.2 THE SLOP WAVE: corrupted theme (minor, detuned, seasick), canon, drowned by surge
    crash_t = CUE["slop_crash"]
    for v, (t0, tr, lvl, wob) in enumerate([(34.8, 0, -18, 0.25), (35.6, -1, -21, 0.45), (36.3, 6, -22, 0.7),
                                           (36.9, 5, -22, 1.0), (37.4, -3, -21, 1.4), (37.8, 1, -20, 1.9)]):
        for j, n in enumerate(th("min", 0, tr)):
            tj = t0 + j * (0.4 - 0.04 * v)
            if tj > crash_t - 0.05:
                break
            dur = 0.9
            tn = tt(dur)
            drift = -0.35 * wob * tn / dur + 0.18 * wob * np.sin(2 * np.pi * (4.5 + v) * tn)
            f = hz(n) * 2 ** (drift / 12)
            ph = 2 * np.pi * np.cumsum(f) / SR
            y = np.sin(ph + 1.2 * np.exp(-tn / 0.15) * np.sin(3.5 * ph)) * np.exp(-tn / 0.5)
            y2 = np.sin(ph * 1.012 + 0.8 * np.sin(2 * ph))
            y = (y + 0.35 * y2 * np.exp(-tn / 0.4)) * np.clip(tn / 0.004, 0, 1)
            b.add("fx", stereo_place(np.stack([y, y], 1), rng.uniform(-0.7, 0.7)), t=tj, gain=db(lvl))
            # sick solo violin doubling the first voice
            if v == 0:
                I("vln_solo").note(b, "str", tj, n + 12, 0.38, 0.5, -0.2, gain=0.35,
                                   bend=lambda x, w=wob: -0.4 * x + 0.3 * np.sin(2 * np.pi * 5.0 * x), att=0.02, rel=0.2)
    # surge: tremolo cluster crescendo rising, Shepard rise, noise riser, timpani roll, reverse cymbal
    dur = crash_t - 35.0
    strings(b, 35.0, dur, "trem", dyn=[(0, 0.05), (dur * 0.6, 0.45), (dur, 1.0)], att=0.3, rel=0.02,
            v1=["D5", "Eb5", "E5", "F5"], v2=["A4", "Bb4"], va=["D4", "Eb4", "F4"], vc=["D3", "A3", "Eb3"],
            bend=lambda x: 2.5 * (x / 4.2) ** 2, gain=0.85)
    amp = envelope([(0, 0), (2.0, 0.3), (4.4, 1.0)], S(4.4))
    y = shepard(4.4, fbase=hz(m("D1")), n_oct=9, center_oct=4.5, width_oct=1.4, amp=amp,
                rate_fn=lambda x: 0.15 + 0.5 * (x / 4.4) ** 2)
    b.add("fx", y, t=crash_t - 4.4, gain=db(-12))
    r = riser(3.0, 250, 9000, q=1.5, curve=2.2, seed=3, tonal=(m("D3"), m("D5")))
    b.add("fx", r, i0=S(crash_t) - len(r), gain=db(-15))
    timp_roll(b, 37.0, crash_t - 37.0, "D2", [(0, 0.1), (crash_t - 37.0, 1.0)], gain=0.9)
    swell_to(b, crash_t, PERC + "susCymb1-cresc-Median_v1.wav", gain=0.35, after=0.01)
    # glitch stutters of the corrupted theme 37.6 -> 39.1
    for q in range(10):
        tq = 37.6 + q * 0.15
        n = th("min", 1)[q % 7] + int(rng.integers(-2, 3))
        y = fm_bell(hz(n), 0.06, ratio=1.5, index=3.0, decay=0.05)
        for rep in range(3):
            b.add("fx", np.stack([y, y], 1) * 0.8 ** rep, t=tq + rep * 0.03, gain=db(-24))
    # ---- 39.2 MASSIVE IMPACT (tonal braam, sinks)
    sink = lambda x: -3.0 * np.clip((x - 0.35) / 1.6, 0, 1) ** 1.3
    brass(b, crash_t, 1.7, dyn=[(0, 1.0), (0.3, 0.75), (1.7, 0.4)], att=0.0, rel=0.8, tbn=["D2", "A2", "D3"],
          tuba=["D2"], hn=["A3", "D4", "Eb4"], bend=sink, gain=1.0)
    strings(b, crash_t, 1.7, dyn=[(0, 1.0), (0.3, 0.7), (1.7, 0.35)], att=0.0, rel=0.8, v1=["D5", "Eb5"],
            va=["A3", "D4"], vc=["D3", "A2"], cb=["D2"], bend=sink, gain=0.9)
    tutti(b, crash_t, "Dm", 1.0, tpt=True, gong=True)
    hit(crash_t, "slop_crash", "D-minor braam + cluster, sinks")
    # ---- 40.6 N07: slow waves under the narration + feeds multiplying
    for (a, lv) in [(40.3, 0.3), (42.8, 0.27), (45.3, 0.25)]:
        strings(b, a, 2.6 if a < 45 else 1.6, dyn=[(0, 0.08), (1.2, lv), (2.6, 0.05)], att=0.4, rel=0.5, vc=["D3", "A3"], cb=["D2"],
                gain=0.75)
    sp = scene_events("s02", "split", None, [43.493, 43.843, 44.58, 45.168])
    first = scene_events("s02", "hit", "feed", [42.45])[0]
    marks = [(first, 1)] + list(zip(sp, [4, 16, 48, 96]))
    feeds = []
    for (tm, cnt) in marks:
        while len(feeds) < cnt:
            feeds.append(tm + rng.uniform(0, 0.12))
    fb = Buf()
    t_end = 47.45
    for i, t0 in enumerate(feeds):
        key = int(rng.integers(0, 12))
        bpm = rng.uniform(96, 168)
        st = 60 / bpm / 2
        scale = [0, 2, 4, 7, 9] if rng.random() < 0.6 else [0, 3, 5, 7, 10]
        pat = [int(rng.choice(scale)) + 12 * int(rng.integers(0, 2)) for _ in range(int(rng.integers(3, 7)))]
        base = 60 + key + 12 * int(rng.integers(0, 2))
        timbre = int(rng.integers(0, 4))
        pan = float(np.clip(rng.normal(0, 0.55), -1, 1))
        g = db(-24) * (1.0 / np.sqrt(1 + i * 0.35))
        t, k = t0, 0
        while t < t_end:
            f = hz(base + pat[k % len(pat)])
            d = st * 0.9
            tn = tt(d)
            if timbre == 0:   # chip square
                y = np.sign(np.sin(2 * np.pi * f * tn)) * 0.4 * np.exp(-tn / 0.12)
            elif timbre == 1:  # pluck
                y = fm_bell(f, d, ratio=1.0, index=2.0, decay=0.12, idx_decay=0.05)
            elif timbre == 2:  # music box
                y = fm_bell(f * 2, d, ratio=5.0, index=0.8, decay=0.3)
            else:              # saw blip
                y = (2 * ((f * tn) % 1) - 1) * 0.35 * np.exp(-tn / 0.08)
            fb.add("x", stereo_place(np.stack([y, y], 1).astype(np.float32), pan), t=t, gain=g)
            t += st
            k += 1
    x = fb.b["x"]
    a, z = S(42.4), S(47.5)
    seg = x[a:z]
    seg = butter(seg, "highpass", 450, 2)
    seg = butter(seg, "lowpass", 5200, 2)
    # 46.6 -> 47.45: the feeds smear into static (push into one screen)
    nseg = len(seg)
    wash = noise(nseg, 7) * envelope([(0, 0), (4.2, 0), (4.95, 0.5), (5.05, 0)], nseg)[:, None]
    seg = seg * envelope([(0, 1), (4.2, 1), (5.02, 0), (5.1, 0)], nseg)[:, None] + butter(wash, "highpass", 1500, 2) * 0.05
    b.add("feed", seg, i0=a)


# ================================================================================================ s03
def s03(b):
    """The Noospheric Munitions Act (47.5–61.5): presidential fanfare pastiche of the Flag theme, all of
    it heard THROUGH the CRT (band-limited, saturated, small room); stab on 'banned', pen gliss, organ
    'amen' on 'sacrament', ka-ching ta-da button 60.9, tape-stop when the CRT dies (61.0→61.45)."""
    tv = Buf()
    t0 = CUE["tv_on"]
    lock = scene_events("s03", "click", "vertical hold", [48.02])[0]
    # fanfare tuning in through the static: triplet pickups warble, the D major chord lands as the picture locks
    for j in range(3):
        dt = j * (lock - t0) / 3
        brass(tv, t0 + dt, None, "stac", 0.9, tpt=["A4", "D5"], hn=["F#4"], gain=0.9)
        perc(tv, t0 + dt, V1P + "drums/snare/drum3_marching/snare3_f_%d.wav" % (1 + j % 2), 0.35, 0.0)
    ta = lock
    hit(lock, "tv_lock", "fanfare chord as the picture locks")
    brass(tv, ta, 0.55, dyn=[(0, 0.95), (0.55, 0.7)], att=0.0, rel=0.35, tpt=["D5", "F#5", "A5"], hn=["D4", "A4"],
          tbn=["D3", "A3"], tuba=["D2"], gain=0.9)
    timp(tv, ta, "D2", 0.9)
    crash(tv, ta, "ff", 0.8)
    hit(t0, "tv_on", "fanfare (through the TV)")
    # patriotic bed under J01: snare roll pp + brass chords, muted trumpet theme in dotted pomp
    t_ban = W("J01", "banned")
    prog = [(48.3, "D"), (50.2, "G"), (51.9, "D"), (53.7, "A"), (55.2, "Bm"), (56.1, "G"), (56.55, "A")]
    for i, (a, ch) in enumerate(prog):
        z = prog[i + 1][0] if i + 1 < len(prog) else t_ban
        lv = 0.3 + 0.35 * (a - 48.3) / 8.7
        brass(tv, a, z - a + 0.05, dyn=[(0, lv), (z - a, lv)], att=0.12, rel=0.2, hn=tones(ch, "D4", 3),
              tbn=[root(ch, "C3")], gain=0.6)
        strings(tv, a, z - a + 0.05, dyn=[(0, lv * 0.8), (z - a, lv * 0.8)], att=0.1, rel=0.2, v1=tones(ch, "D5", 2),
                vc=[root(ch, "C3")], gain=0.4)
    t = 48.3
    while t < t_ban - 0.2:
        perc(tv, t, PERC + "Snare2-rollNS_v1_rr1_Sum.wav", 0.12 + 0.1 * (t - 48.3) / 8.7, 0.0, dur=1.6)
        t += 1.45
    rhy = [(0.0, 0.75), (0.75, 0.25), (1.0, 0.75), (1.75, 0.25), (2.0, 0.75), (2.75, 0.25), (3.0, 1.0)]
    for rep, tb_ in enumerate([51.0, 53.6]):
        for (dt, d), n in zip(rhy, th("maj", 0)):
            brass(tv, tb_ + dt * 0.6, d * 0.6, "mute", dyn=[(0, 0.5), (d * 0.6, 0.45)], att=0.01, rel=0.1,
                  tpt=[n], gain=0.55 + 0.15 * rep)
    snare_roll = t_ban - 0.9
    perc(tv, snare_roll, PERC + "Snare2-rollSN_v5_rr1_Sum.wav", 0.3, 0.0, dur=0.9)
    # 'banned!' fist slam: tutti stab
    brass(tv, t_ban, None, "stac", 1.0, tpt=["D5", "F#5", "A5"], hn=["D4", "F#4", "A4"], tbn=["D3", "A3"],
          tuba=["D2"], gain=1.0)
    timp(tv, t_ban, "D2", 1.0)
    crash(tv, t_ban, "ff", 0.9)
    hit(t_ban, "J01_banned", "stab on 'banned!' (fist slam)")
    # pen flourish
    tp = CUE["pen_sign"]
    harp_gliss(tv, tp, tp + 0.3, "D5", "D7", vel=0.7, gain=1.5)
    I("glock").note(tv, "keys", tp + 0.33, "A6", None, 0.8, 0.0, gain=0.12)
    # 'sacrament': churchy plagal amen (reed organ), sly
    ts = W("J02", "sacrament")
    ro = sf2part(SF_MS, 0, 20, gain=1.0)
    for n in ["G3", "B3", "D4", "G4"]:
        ro.note(ts, m(n), 0.3, 0.5)
    for n in ["F#3", "A3", "D4", "F#4"]:
        ro.note(ts + 0.3, m(n), 0.28, 0.45)
    ro.to(tv, "keys", pan=0.0)
    # 60.9 ka-ching: brass 'ta-DA' + glock sparkle
    tk = CUE["ka_ching"]
    brass(tv, tk, 0.4, dyn=[(0, 1.0), (0.4, 0.8)], att=0.0, rel=0.3, tpt=["D5", "F#5", "A5"], hn=["D4", "A4"],
          tbn=["D3"], gain=1.0)
    timp(tv, tk, "D2", 0.8)
    for j, n in enumerate(["D6", "F#6", "A6", "D7"]):
        I("glock").note(tv, "keys", tk + 0.035 * j, n, None, 0.8, 0.0, gain=0.11)
    perc(tv, tk, PERC + "Triangle3-Hit_v2_rr1_Sum.wav", 0.25, 0.0)
    hit(tk, "ka_ching", "brass ta-DA button + glock")
    # render the TV: mono-ish, band-limited, saturated, small room; CRT dies -> tape stop
    a, z = S(47.3), S(61.9)
    x = sum(tv.b[k][a:z] for k in tv.b)
    x = eq(x, ("hp", 220), ("hp", 180), ("lp", 5200), ("lp", 6500), ("pk", 1900, 4.0, 0.9), ("pk", 400, -3, 0.8))
    x = np.tanh(x * 2.2) / 2.2
    # tuning in: pitch warble + narrower band until the vertical hold locks
    i0, i1 = S(t0 - 47.3), S(lock + 0.08 - 47.3)
    n = i1 - i0
    tl_ = np.arange(n) / SR
    depth = 0.018 * np.clip(1 - tl_ / (lock - t0), 0, 1) ** 1.5
    steps = 1 + depth * np.sin(2 * np.pi * 6.5 * tl_)
    wob = _resample(np.ascontiguousarray(x), float(i0), steps.astype(np.float64), n)
    narrow = eq(wob, ("hp", 700), ("lp", 2600))
    w = np.clip(1 - tl_ / (lock - t0), 0, 1)[:, None]
    x[i0:i1] = narrow * w + wob * (1 - w)
    mid = x.mean(1, keepdims=True)
    x = 0.8 * mid + 0.2 * x
    x = x + 0.22 * conv_stereo(x, IR("room"))[: len(x)]
    x = tape_stop(x, 61.0 - 47.3, 0.45, 0.0)
    x[S(61.45 - 47.3):] = 0
    x[S(61.43 - 47.3):S(61.45 - 47.3)] *= np.linspace(1, 0, S(61.45 - 47.3) - S(61.43 - 47.3))[:, None]
    b.add("tv", x.astype(np.float32), i0=a)


# ================================================================================================ s04
BAR4 = 2.8
B4 = 0.7  # 85.7 BPM; bars 61.5, 64.3, 67.1, 69.9 → hush 72.7


def pluck(f, dur, cutoff, decay=0.35):
    n = S(dur)
    tn = np.arange(n) / SR
    s = _saw_bank(np.stack([np.full(n, f), np.full(n, f * 1.004)]).astype(np.float64), np.array([0.0, 0.5]), n, SR) / 2
    fc = cutoff * (0.25 + 0.75 * np.exp(-tn / (decay * 0.5)))
    y = svf(s, fc, 1.2, 0) * np.exp(-tn / decay) * np.clip(tn / 0.002, 0, 1)
    return y.astype(np.float32)


def s04(b):
    """Flagartha (61.5–90.0): D pedal (organ 32'+16'), low choir, the Hypermind's arpeggiator built from the
    theme; six canon-scroll bells; the six plates stepping down; HUSH at 72.7 with the theme on glass; worried
    B minor under L02; solemn organ under C02; klaxon build and braam 89.4; cut 90.0."""
    rng = np.random.default_rng(404)
    t0, th_ = 61.5, CUE["hypermeme"]
    # organ D pedal (32' synth + 16' sampled) and soft manual chords
    y = sub_drone(hz(m("D1")), th_ - t0 + 0.3, amp_bp=[(0, 0), (1.5, 0.8), (th_ - t0 - 1.5, 1.0), (th_ - t0, 0.9),
                                                        (th_ - t0 + 0.05, 0), (th_ - t0 + 0.3, 0)], harm=0.35)
    b.add("org", y, t=t0, gain=db(-20))
    I("organ_softped").note(b, "org", t0, "D2", th_ - t0, 0.8, 0.0, gain=0.5, att=1.2, rel=0.05)
    bars = [(61.5, "D", ["A3", "D4", "F#4", "E5"]), (64.3, "Bm", ["B3", "D4", "F#4", "C#5"]),
            (67.1, "G", ["B3", "D4", "G4", "F#5"]), (69.9, "A", ["A3", "C#4", "E4", "E5"])]
    for i, (a, ch, vo) in enumerate(bars):
        z = bars[i + 1][0] if i + 1 < len(bars) else th_
        for j, n in enumerate(vo):
            I("organ_soft").note(b, "org", a, n, z - a + (0.03 if i < 3 else -0.04), 0.8, -0.3 + 0.2 * j, gain=0.22,
                                 att=0.6 if i == 0 else 0.08, rel=0.02 if i == 3 else 0.2)
    ch_ = sf2part(SF_MS, 0, 53, gain=1.3)  # voice oohs, low
    for (a, ch, vo) in bars:
        for n in [root(ch, "D3"), root(ch, "D3") + 7]:
            ch_.note(a, n, BAR4 if a < 69.9 else th_ - a, 0.45)
    # arpeggiator (16ths), filter opening as the Hypermind wakes
    patt = {"D": ["D4", "A4", "E5", "F#5", "A5", "F#5", "E5", "A4"], "Bm": ["B3", "F#4", "C#5", "D5", "F#5", "D5", "C#5", "F#4"],
            "G": ["G3", "D4", "A4", "B4", "D5", "F#5", "D5", "B4"], "A": ["A3", "E4", "B4", "C#5", "E5", "C#5", "B4", "E4"]}
    t, k = 62.2, 0
    while t < 71.2 - 1e-6:
        ch = [c for (a, c, _) in bars if t >= a - 1e-6][-1]
        n = m(patt[ch][k % 8])
        prog = (t - 62.2) / 9.0
        y = pluck(hz(n), 0.5, 700 + 3500 * prog ** 1.5, 0.22)
        pan = 0.5 * np.sin(k * 0.7)
        b.add("syn", stereo_place(np.stack([y, y], 1), pan, 1.0), t=t, gain=db(-24 + 4 * prog))
        # dotted-eighth echo
        b.add("syn", stereo_place(np.stack([y, y], 1), -pan, 1.0), t=t + 0.525, gain=db(-33 + 4 * prog))
        t += B4 / 4
        k += 1
    # reveal ~63.0: harp gliss + glock
    harp_gliss(b, 62.75, 63.0, "D4", "D6", vel=0.45, gain=0.9)
    I("glock").note(b, "keys", 63.0, "A6", None, 0.5, 0.3, gain=0.06)
    # six canon scrolls absorbed (Scene04 times): rising tubular/glass bells
    tub = sf2part(SF_MS, 0, 14, gain=1.9)
    for j, (tj, n) in enumerate(zip([65.50, 65.85, 66.20, 66.55, 66.90, 67.25], ["D5", "E5", "F#5", "A5", "B5", "D6"])):
        tub.note(tj, m(n), 1.5, 0.45)
        glass_note(b, tj, m(n) + 12, 2.0, gain=db(-26), pan=-0.6 + 0.24 * j, decay=0.9)
        hit(tj, f"canon_{j + 1}", "scroll absorbed bell")
    tub.to(b, "keys", pan=0.2)
    # 'TRUE' 70.11: small bright chord
    tt_ = W("H01", "true")
    for n in ["D6", "F#6", "A6"]:
        glass_note(b, tt_, n, 1.8, gain=db(-28), pan=0.1, decay=0.8)
    # H02: energy steps down the six plates 71.25 -> 72.30; beam 72.35; swell into the hush
    plates = scene_events("s04", "energy", None, [71.3 + 0.17 * i for i in range(6)])
    for j, n in enumerate(["D7", "A6", "F#6", "D6", "A5", "D5"][:len(plates)]):
        tj = plates[j]
        bell_note(b, tj, n, 1.2, gain=db(-24 + j), pan=0.4 - 0.16 * j, bus="keys", ratio=1.41, index=2.2, decay=0.35)
    I("organ_soft").note(b, "org", 71.2, "D5", th_ - 71.2 - 0.04, 0.8, 0.0, gain=0.18, att=0.4, rel=0.02)
    r = riser(th_ - 72.3, 400, 7000, q=2.0, curve=2.0, seed=4)
    b.add("fx", r * 0.6, i0=S(th_) - len(r), gain=db(-22))
    # ---- 72.7 HUSH: the room holds its breath (cut-off), then the theme on glass, pp, over high harmonics
    ch_.to(b, "org", pan=0.0, width=1.0, gain=0.8)
    cut_after(b, th_, 0.012)
    e = 2.0 / 9
    ts = [th_ + e * k for k in (0, 2, 3, 5, 6, 8, 9)]
    cel = sf2part(SF_MS, 0, 8, gain=1.6)
    for j, (tj, n) in enumerate(zip(ts, th("maj", 1))):
        glass_note(b, tj, n, 2.5 if j == 6 else 1.4, gain=db(-20 if j == 0 else -24), pan=0.2 * np.sin(j * 1.3), decay=1.2)
        cel.note(tj, n, 0.4, 0.55 if j == 0 else 0.4)
    cel.to(b, "keys", pan=0.3)
    hit(th_, "hypermeme", "reverent hush: everything cuts away", kind="cut")
    hit(th_, "hypermeme_glass", "first glass note of the theme")
    strings(b, th_, 79.4 - th_, dyn=[(0, 0.2), (2, 0.28), (6.4, 0.25)], att=0.9, rel=0.8, v1=["A5", "D6"],
            v2=["F#5"], va=["D5"], gain=0.45)
    strings(b, 76.3, 3.0, dyn=[(0, 0.1), (1.5, 0.22), (3.0, 0.15)], att=1.0, rel=0.8, vc=["D3"], cb=["D2"], gain=0.5)
    harp_roll(b, 78.75, ["D4", "A4", "D5"], 0.35, 0.09, gain=0.8)
    # ---- 79.2 L02 worried: B minor -> G -> Em
    for (a, z, v1, va, vc) in [(79.2, 81.0, ["F#5"], ["D4", "B4"], ["B2"]), (81.0, 82.4, ["G5"], ["D4", "B4"], ["G2"]),
                               (82.4, 83.45, ["G5"], ["E4", "B4"], ["E3"])]:
        strings(b, a, z - a + 0.1, dyn=[(0, 0.22), (z - a, 0.28)], att=0.35, rel=0.5, v1=v1, va=va, vc=vc, gain=0.55)
    # ---- 83.4 C02 solemn: organ + choir, bass falls on 'fall' and rises on 'rise'
    t_fall, t_rise, t_sum = W("C02", "fall"), W("C02", "rise"), W("C02", "Summon")
    t_sch = W("C02", "schismmancers")
    prog = [(83.4, ["E3", "G3", "B3", "E4"], "E2"), (t_fall, ["E3", "G3", "C4", "E4"], "C2"),
            (85.6, ["D3", "F#3", "A3", "D4"], "D2"), (86.3, ["E3", "G3", "B3", "E4"], "E2"),
            (t_rise, ["D3", "G3", "B3", "D4", "G4"], "G2"), (87.3, ["E3", "A3", "C#4", "E4", "A4"], "A2")]
    for i, (a, vo, bs) in enumerate(prog):
        z = prog[i + 1][0] if i + 1 < len(prog) else t_sum
        lv = 0.22 + 0.25 * (a - 83.4) / 4.4
        for j, n in enumerate(vo):
            I("organ_soft").note(b, "org", a, n, z - a + 0.03, 0.8, -0.3 + 0.15 * j, gain=0.2 + 0.1 * lv, att=0.12, rel=0.15)
        I("organ_softped").note(b, "org", a, bs, z - a + 0.03, 0.8, 0.0, gain=0.35, att=0.1, rel=0.15)
        strings(b, a, z - a + 0.05, dyn=[(0, lv), (z - a, lv * 1.1)], att=0.15, rel=0.3, vc=[bs], cb=[bs],
                va=[vo[-1]], gain=0.5)
    # 87.85 STRIKE 'Summon': D minor, klaxon build, taiko accelerating, braam 89.4
    strike = t_sum
    timp(b, strike, "D2", 0.9, gain=0.9)
    taiko(b, strike, "f", gain=0.8)
    strings(b, strike, 89.4 - strike, "trem", dyn=[(0, 0.3), (89.4 - strike, 1.0)], att=0.05, rel=0.05,
            v1=["D5", "F5", "A5"], va=["A3", "D4", "F4"], vc=["D3", "A3"], cb=["D2"], gain=0.6)
    for j, tj in enumerate([x for x in scene_events("s04", "klaxon", None, [88.4, 88.85, 89.3]) if x < 89.35]):
        # klaxon: each gold sweep = a horn 'wee-oo' (A -> Bb -> A)
        brass(b, tj, 0.42, dyn=[(0, 0.45 + 0.1 * j), (0.2, 0.7 + 0.1 * j), (0.42, 0.4)], att=0.02, rel=0.06,
              hn=["A4", "D4"], bend=lambda x: np.sin(np.pi * np.clip(x / 0.42, 0, 1)) * 1.0, gain=0.55)
    timp_roll(b, 88.15, 89.4 - 88.15, "D2", [(0, 0.2), (89.4 - 88.15, 1.0)], gain=0.9)
    tk = [88.35, 88.75, 89.0, 89.18, 89.3]
    for j, tj in enumerate(tk):
        taiko(b, tj, "f" if j < 3 else "ff", gain=0.55 + 0.1 * j, sticks=j % 2 == 1)
    r = riser(89.4 - 88.0, 300, 8000, q=1.2, curve=2.5, seed=5, tonal=(m("D3"), m("A4")))
    b.add("fx", r, i0=S(89.4) - len(r), gain=db(-20))
    # 89.4 BRAAM (D minor), then reverse suck 89.7→90.0, hard cut at 90.0
    tbm = 89.4
    brass(b, tbm, 0.6, dyn=[(0, 1.0), (0.2, 0.8), (0.6, 0.7)], att=0.0, rel=0.02, tbn=["D2", "A2", "D3", "F3"],
          tuba=["D2"], hn=["D3", "A3", "F4", "A4"], gain=1.0)
    strings(b, tbm, 0.6, dyn=[(0, 1.0), (0.6, 0.7)], att=0.0, rel=0.02, v1=["D5", "A5"], va=["F4", "D4"],
            vc=["D3", "A2"], cb=["D2"], gain=0.9)
    tutti(b, tbm, "Dm", 0.9, crash_on=False)
    hit(tbm, "schismmancers", "D-minor braam (push-in flash)")
    swell_to(b, 90.0, PERC + "susCymb1-cresc-Short_v1.wav", gain=0.3, after=0.0)
    # everything s04 is hard-cut at 90.0 (the cut); applied to this section's buses
    for k in b.b:
        b.b[k][S(89.985):S(90.0)] *= np.linspace(1, 0, S(90.0) - S(89.985))[:, None]
        b.b[k][S(90.0):] = 0


# ================================================================================================ s05a
B5 = 0.525  # 114.29 BPM; bars 90.0, 92.1, 94.2, 96.3 → 98.4 stack-up


def s05a(b):
    """The Schismmancers (90.0–100.5): tactical 3+3+2 spiccato ostinato + taiko, the Flag theme in Dorian
    low brass, snare build, three 'Breach!' stabs (G–A–Bb rising), reverse suck, BREACH = D major blaze."""
    t0 = CUE["gear_up"]
    harm = [(90.0, "Dm", "D3", "D2"), (92.1, "G", "G2", "G1"), (94.2, "F", "F2", "F1"), (96.3, "Dm", "D3", "D2")]
    stop = W("K03", "Breach", 0)
    s16 = B5 / 4
    k = 0
    t = t0
    while t < stop - 1e-6:
        bar = int((t - 90.0 + 1e-6) // (4 * B5))
        ch, rv, rb = harm[min(bar, 3)][1:]
        pos = k % 8
        acc = pos in (0, 3, 6)
        dlg = 91.2 <= t <= 95.8 or 96.1 <= t <= 97.9
        v = (0.75 if acc else 0.4) * (0.85 if dlg else 1.0) + (0.15 if bar == 3 else 0.0)
        up = 12 if (bar == 3 and k % 16 >= 8) else 0
        I("vc_spic").note(b, "str", t, m(rv) + up, None, v, PAN["vc"], gain=0.6, seed=k)
        if not acc:
            I("vla_spic").note(b, "str", t, m(rv) + 19 + up, None, v * 0.8, PAN["vla"], gain=0.35, seed=k)
        if pos % 2 == 0:
            I("cb_spic").note(b, "str", t, m(rb) + 12 if m(rb) < 30 else m(rb), None, v, PAN["cb"], gain=0.5, seed=k)
        if bar >= 2 and pos % 2 == 0:
            I("vln_spic").note(b, "str", t, "D5" if bar == 2 else "A5", None, 0.5 + 0.2 * (bar == 3), PAN["vln"],
                               gain=0.3, seed=k)
        # percussion: taiko on accents, tenor ghosts, rimshots on 2 & 4
        if acc:
            taiko(b, t, "ff" if pos == 0 and k % 16 == 0 else "f", gain=(0.55 if dlg else 0.75) * (1.0 if pos == 0 else 0.7),
                  sticks=pos == 6)
        elif pos % 2 == 0:
            perc(b, t, V1P + "drums/tenor/tenor_lower/tenor_mf_%d.wav" % (1 + k % 4), 0.2, -0.2)
        if k % 8 == 4:
            perc(b, t, V1P + "drums/snare/drum3_marching/snare3_rimshot_f_1.wav", 0.18, 0.15)
        t += s16
        k += 1
    # gear-up downbeat
    timp(b, t0, "D2", 0.9)
    taiko(b, t0, "ff", gain=1.0)
    perc(b, t0, V1P + "drums/snare/drum3_marching/snare3_rimshot_ff_1.wav", 0.35, 0.1)
    brass(b, t0, None, "stac", 0.8, tbn=["D2", "A2"], tuba=["D2"], hn=["D3", "A3"], gain=0.8)
    hit(t0, "gear_up", "taiko + rimshot + low brass downbeat")
    # the Flag theme in D Dorian, low brass (2 beats per note)
    ts = [t0 + 2 * B5 * j for j in range(7)]
    for j, (tj, n) in enumerate(zip(ts, th("dor", -1))):
        d = (2 * B5 if j < 6 else stop - tj) + 0.04
        dlg = 91.2 <= tj <= 97.9
        lv = 0.55 if dlg else 0.7
        brass(b, tj, d, dyn=[(0, lv), (d * 0.5, lv * 0.85), (d, lv * (1.2 if j == 6 else 0.9))], att=0.03, rel=0.2,
              hn=[n], tbn=[n - 12], gain=0.75)
    for (a, ch, rv, rb), z in zip(harm, [92.1, 94.2, 96.3, stop]):
        strings(b, a, z - a + 0.05, dyn=[(0, 0.3), (z - a, 0.35 if a < 96 else 0.8)], att=0.1, rel=0.1,
                v1=tones(ch, "D5", 2), gain=0.35)
    # build: snare roll into the stack-up
    perc(b, stop - 1.05, PERC + "Snare2-rollSN_v5_rr1_Sum.wav", 0.35, 0.1, dur=1.05)
    # three Breach! stabs, bass G–A–Bb rising, then C riser → D major BREACH
    for j, ch in enumerate(["Gm", "A", "Bb"]):
        tj = W("K03", "Breach", j)
        tutti(b, tj, ch, 0.6 + 0.07 * j, crash_on=j == 2)
        taiko(b, tj, "f", gain=0.5, sticks=True)
        perc(b, tj, V1P + "drums/snare/drum3_marching/snare3_rimshot_f_1.wav", 0.25, 0.1)
        hit(tj, f"K03_breach_{j + 1}", f"{ch} stab")
    tb_ = CUE["breach"]
    t3 = W("K03", "Breach", 2)
    strings(b, t3 + 0.08, tb_ - t3 - 0.08, "trem", dyn=[(0, 0.3), (tb_ - t3 - 0.08, 1.0)], att=0.05, rel=0.01,
            v1=["C5", "G5"], va=["E4", "C4"], vc=["C3"], bend=lambda x: 2.0 * (x / 0.6) ** 2, gain=0.6)
    brass(b, t3 + 0.1, tb_ - t3 - 0.1, dyn=[(0, 0.3), (tb_ - t3 - 0.1, 0.95)], att=0.1, rel=0.01, hn=["C4", "E4", "G4"],
          tbn=["C3"], gain=0.6)
    swell_to(b, tb_, PERC + "susCymb1-cresc-Short_v1.wav", gain=0.4, after=0.0)
    r = riser(tb_ - t3, 500, 10000, q=1.5, curve=2.0, seed=9)
    b.add("fx", r, i0=S(tb_) - len(r), gain=db(-20))
    # BREACH: D major blaze (the flag light)
    sus = CUE["nation_fracture"] - tb_ - 0.06
    tutti(b, tb_, "D", 1.0, gong=False, crash_dur=0.8)
    brass(b, tb_, sus, dyn=[(0, 1.0), (0.3, 0.75), (sus, 0.3)], att=0.0, rel=0.1, tpt=["D5", "F#5", "A5"],
          hn=["D4", "F#4", "A4", "D5"], tbn=["D3", "A3"], tuba=["D2"], gain=0.9)
    strings(b, tb_, sus, dyn=[(0, 1.0), (0.4, 0.7), (sus, 0.3)], att=0.0, rel=0.1, v1=["A5", "D6"], v2=["F#5", "D5"],
            va=["A4", "F#4"], vc=["D3", "A3"], cb=["D2"], gain=0.9)
    cho = sf2part(SF_MS, 0, 52, gain=1.6)
    for n in ["D4", "A4", "D5", "F#5", "A5"]:
        cho.note(tb_, m(n), sus, 0.9)
    cho.to(b, "org", pan=0.0)
    harp_gliss(b, tb_, tb_ + 0.5, "D4", "D7", vel=0.7, gain=1.4)
    for j, n in enumerate(["D6", "A6", "D7"]):
        I("glock").note(b, "keys", tb_ + 0.05 * j, n, None, 0.8, 0.2, gain=0.1)
    hit(tb_, "breach", "D major blaze, gong")
    cut_after(b, CUE["nation_fracture"] - 0.03, 0.12)


# ================================================================================================ s05b
def s05b(b):
    """Recursive Balkanization (100.5–125.5): three stingers on a symmetric division of the octave
    (D minor 100.6 → F# minor 107.8 → Bb minor 114.6), fractal pizzicato subdivision, comic micro-stingers
    (tiny anthem for the Republic of Me, lonely triangle, heretics' organ duel a tritone apart, harmonium amen,
    flexatone, 'I am' echo), the shattered-marble glass cascade and the void."""
    rng = np.random.default_rng(505)
    tn, tf, tp = CUE["nation_fracture"], CUE["faith_fracture"], CUE["philo_shatter"]
    # ---- NATION (Dm)
    tutti(b, tn, "Dm", 0.6, crash_on=False)
    taiko(b, tn, "ff", gain=0.45, sticks=True)
    hit(tn, "nation_fracture", "D minor stinger")
    pent = [2, 5, 7, 9, 0]  # D minor pentatonic
    insts = ["vln_pizz", "vla_pizz", "marimba", "xylo", "harp", "vln_pizz", "glock"]
    gens = scene_events("s05b", "crack", "generation", [tn + 0.17 + 0.2625 * i for i in range(7)])
    gens = [g for g in gens if g < tf][:7]
    for L in range(len(gens)):
        tL = gens[L]
        span = (gens[L + 1] - tL) if L + 1 < len(gens) else 0.35
        cnt = 2 ** L
        for q in range(cnt):
            tq = tL + span * (q + rng.uniform(0, 0.6)) / cnt
            lo_oct = max(3, 5 - L // 2)
            hi_oct = min(7, 5 + (L + 1) // 2)
            n = 12 * (int(rng.integers(lo_oct, hi_oct + 1)) + 1) + int(rng.choice(pent))
            key = insts[(q + L) % len(insts)]
            if key == "glock" and n < 79:
                n += 24
            if key == "xylo" and n < 67:
                n += 12
            if key in ("vln_pizz",) and n < 55:
                n += 12
            if key == "vla_pizz" and n > 80:
                n -= 12
            g = {"vln_pizz": 0.4, "vla_pizz": 0.4, "marimba": 0.08, "xylo": 0.06, "harp": 0.2, "glock": 0.04}[key]
            I(key).note(b, "keys" if key in ("marimba", "xylo", "harp", "glock") else "str", tq, n, None,
                        0.6, float(rng.uniform(-0.8, 0.8)) * min(1, 0.2 + L / 5), gain=g * 0.93 ** L, seed=q)
    # tiny national anthem (oompah, D major) under X01; snare roll into "Me!"
    t_me = W("X01", "Me")
    t = LINES["X01"]["start"]
    k = 0
    while t < t_me - 0.1:
        if k % 2 == 0:
            I("tuba_stac").note(b, "brs", t, "D2" if k % 4 == 0 else "A1", None, 0.45, PAN["tuba"], gain=0.5, seed=k)
        else:
            for j, n in enumerate(["F#4", "A4", "D5"]):
                I("vln_pizz").note(b, "str", t, n, None, 0.4, -0.3 + 0.2 * j, gain=0.22, seed=k + j)
        t += 0.35
        k += 1
    perc(b, W("X01", "Sovereign"), PERC + "Snare2-rollSN_v3_rr1_Sum.wav", 0.18, 0.1, dur=t_me - W("X01", "Sovereign") - 0.03)
    brass(b, t_me, None, "stac", 0.75, tpt=["D5", "F#5", "A5"], gain=0.55)
    I("picc_stac").note(b, "ww", t_me, "D6", None, 0.7, PAN["fl"], gain=0.4)
    crash(b, t_me, "mp", 0.6)
    timp(b, t_me, "D2", 0.6, gain=0.6)
    hit(t_me, "X01_me", "tiny anthem ta-DA on 'Me!'")
    t_one = LINES["X02"]["end"] + 0.03
    perc(b, t_one, PERC + "Triangle3-Hit_v1_rr1_Sum.wav", 0.25, 0.3)
    I("vln_pizz").note(b, "str", t_one, "D5", None, 0.4, PAN["vln"], gain=0.3)
    hit(t_one, "X02_one", "lonely triangle + pizz")
    # ---- FAITH (F#m): organ chord that splits like the congregation
    tutti(b, tf, "F#m", 0.55, crash_on=False, tpt=False)
    hit(tf, "faith_fracture", "F# minor stinger + organ")
    # the congregation splits 1 -> 4 -> 16 -> 64: every organ voice splits into a fractal cluster of detuned voices
    fg = scene_events("s05b", "crack", "floor schism", [107.9, 108.25, 108.73])[:3]
    sm = lambda x, a: np.clip((x - a) / 0.18, 0, 1) ** 2 * (3 - 2 * np.clip((x - a) / 0.18, 0, 1))
    for j, n in enumerate(["F#2", "C#3", "F#3", "A3", "C#4"]):
        for q in range(8):
            sg = [1 if (q >> k) & 1 else -1 for k in range(3)]
            bend = (lambda x, sg=sg: 0.6 * sg[0] * sm(x + tf, fg[0]) + 0.3 * sg[1] * sm(x + tf, fg[1])
                    + 0.15 * sg[2] * sm(x + tf, fg[2]))
            I("organ_loud").note(b, "org", tf, n, 2.0, 0.8, 0.7 * (q / 7 - 0.5), gain=0.35 / 8 * 1.6, bend=bend,
                                 dyn=[(0, 0.9), (1.3, 0.6), (2.0, 0.0)], att=0.0, rel=0.2, seed=q)
    for (tq, ch, vo, bs, name) in [(LINES["X03"]["end"] + 0.02, "Cm", ["C3", "Eb3", "G3", "C4"], "C3", "X03_heretic"),
                                   (LINES["X04"]["end"] + 0.02, "F#m", ["F#3", "A3", "C#4", "F#4"], "F#2", "X04_heretic")]:
        for n in vo:
            I("organ_loud").note(b, "org", tq, n, 0.3, 0.8, 0.0, gain=0.3, att=0.0, rel=0.12)
        I("tbn_stac").note(b, "brs", tq, bs, None, 0.8, PAN["tbn"], gain=0.6)
        timp(b, tq, bs if m(bs) >= 41 else m(bs) + 12, 0.7, gain=0.6)
        hit(tq, name, f"organ duel stab {ch}")
    ta = LINES["X05"]["end"] + 0.02
    ro = sf2part(SF_MS, 0, 20, gain=1.3)
    for n in ["B2", "D#3", "F#3", "B3"]:
        ro.note(ta, m(n), 0.38, 0.55)
    for n in ["F#2", "A#2", "C#3", "F#3"]:
        ro.note(ta + 0.38, m(n), 0.4, 0.55)
    ro.to(b, "keys", pan=0.15)
    hit(ta, "X05_amen", "harmonium plagal amen")
    # ---- PHILOSOPHY (Bbm): marble shatters into glass fragments drifting into the void
    tplant = scene_events("s05b", "thunk", "marble head", [tp - 0.3])[0]
    if tplant < tp - 0.05:
        timp(b, tplant, "Bb2", 0.5, gain=0.6)
        I("vc_pizz").note(b, "str", tplant, "Bb2", None, 0.6, PAN["vc"], gain=0.4)
    tutti(b, tp, "Bbm", 0.6, crash_on=False)
    hit(tp, "philo_shatter", "Bb minor stinger + glass shatter")
    bpent = [10, 1, 3, 5, 8]
    burst = scene_events("s05b", "shatter", "fragments", [tp + 0.25])[0]
    t = burst
    for q in range(170):
        n = 12 * (int(rng.integers(5, 8)) + 1) + int(rng.choice(bpent))
        age = (t - burst) / 1.6
        glass_note(b, t, n, 1.6, gain=db(-24 - 12 * age) * rng.uniform(0.5, 1), pan=float(np.clip(rng.normal(0, 0.2 + 0.8 * age), -1, 1)),
                   decay=0.5 + 0.8 * age, bus="space")
        t += rng.exponential(0.004 + 0.02 * age)
        if t > burst + 2.2:
            break
    hit(burst, "philo_burst", "head bursts: glass cascade")
    for q in range(5):
        I("marimba").note(b, "keys", burst + 0.01 * q, 12 * 4 + int(rng.choice(bpent)), None, 0.8, rng.uniform(-0.5, 0.5),
                          gain=0.08, seed=q)
    # the void: dark Bb minor pad + slowly FALLING Shepard + glints
    vd = 123.3 - tp
    for j, (note, pan) in enumerate([("Bb2", -0.3), ("F3", 0.3), ("Db4", 0.0), ("Ab4", -0.5)]):
        y = supersaw(hz(m(note)), vd + 1.0, voices=5, spread=0.14, cutoff=650, q=0.8, drift=0.002, seed=40 + j)
        y *= envelope([(0, 0), (1.5, 0.7), (vd, 0.8), (vd + 1.0, 0)], len(y))[:, None]
        b.add("syn", stereo_place(y, pan, 0.9), t=tp, gain=db(-25))
    amp = envelope([(0, 0), (2.0, 0.6), (vd - 1, 0.6), (vd, 0)], S(vd))
    y = shepard(vd, rate_oct_s=-0.06, fbase=hz(m("Bb0")), n_oct=9, center_oct=5.0, width_oct=1.2, amp=amp)
    b.add("space", y, t=tp, gain=db(-24))
    t = tp + 2.0
    while t < 123.0:
        glass_note(b, t, 12 * (int(rng.integers(6, 8)) + 1) + int(rng.choice(bpent)), 2.0, gain=db(-29), pan=rng.uniform(-1, 1),
                   decay=1.0)
        t += rng.exponential(0.8) + 0.2
    # crackpot punchlines
    tfx = LINES["X06"]["end"] + 0.03
    perc(b, tfx, V1P + "varMetal/various/flexatone_slap1.wav", 0.8, 0.25)
    hit(tfx, "X06_flexatone", "flexatone after 'poets!'")
    tam = LINES["X07"]["end"] + 0.05
    cel = sf2part(SF_MS, 0, 8, gain=1.8)
    for r in range(5):
        for q, n in enumerate(["F5", "Bb4"]):
            cel.note(tam + r * 0.2 + q * 0.09, m(n), 0.2, 0.6 * 0.62 ** r)
    cel.to(b, "keys", pan=0.2, width=1.0)
    hit(tam, "X07_am", "'I am' celesta echo")


# ================================================================================================ s06
def s06(b):
    """Babel (123.3–138.5): endless Shepard–Risset rise (synth + chromatic tremolo Shepard scale), the Flag
    theme in stretto canon entering in all twelve keys, two pulses (114.3 & 95.2 BPM) grinding, a steady
    pure D under Crocus, babel_crack 136.0, maximal stutter, HARD STOP 138.5."""
    rng = np.random.default_rng(606)
    t_end = CUE["silence"]
    tc = CUE["babel_crack"]
    ts0 = 123.3
    # dialogue windows → chaos dips
    dips = [(LINES["L03"]["start"] - 0.1, LINES["L03"]["end"] + 0.1), (LINES["C03"]["start"] - 0.1, LINES["C03"]["end"] + 0.1)]

    def dip(t):
        return 0.45 if any(a <= t <= z for (a, z) in dips) else 1.0

    # synth Shepard rise, accelerating
    dur = t_end - ts0
    amp = envelope([(0, 0), (2.2, 0.25), (tc - ts0, 0.8), (dur, 1.0)], S(dur))
    rate = lambda x: 0.08 + 0.55 * (x / dur) ** 2.2
    y = shepard(dur, fbase=hz(m("D1")), n_oct=9, center_oct=4.6, width_oct=1.3, amp=amp, rate_fn=rate, detune=0.006)
    y = np.tanh(y * 3.0) / 3.0
    b.add("fx", y, t=ts0, gain=db(-13))
    # sampled Shepard scale: chromatic tremolo steps, accelerating
    t, k = 125.5, 0
    while t < t_end - 0.05:
        step = 0.75 - 0.5 * ((t - 125.5) / (t_end - 125.5)) ** 1.2
        pc = (2 + k) % 12
        for o in range(2, 7):
            n = 12 * (o + 1) + pc
            g = np.exp(-0.5 * ((n - 64) / 11.0) ** 2)
            if g < 0.05:
                continue
            key = "vc_trem" if n < 55 else ("vla_trem" if n < 67 else "vln_trem")
            if key == "vln_trem" and n > 86:
                continue
            lv = 0.35 + 0.6 * (t - 125.5) / (t_end - 125.5)
            I(key).note(b, "str", t, n, step * 2.0, 0.7, {"vc_trem": PAN["vc"], "vla_trem": PAN["vla"], "vln_trem": PAN["vln"]}[key],
                        gain=0.45 * g * dip(t), dyn=[(0, 0.2 * lv), (step, lv), (step * 2, 0.2 * lv)], att=0.05, rel=0.1,
                        seed=k * 7 + o)
        t += step
        k += 1
    # stretto canon: the theme enters in all twelve keys, each voice looping at its own tempo
    voices = [("hn_sus", 0), ("tpt_mute", 7), ("ob_sus", 2), ("cl_sus", 9), ("vln_solo", 4), ("organ_loud", 11),
              ("xylo", 6), ("tbn_sus", 1), ("fl_sus", 8), ("marimba", 3), ("bsn_sus", 10), ("glock", 5)]
    rngs = {"hn_sus": (41, 77), "tpt_mute": (58, 81), "ob_sus": (58, 89), "cl_sus": (50, 88), "vln_solo": (55, 96),
            "organ_loud": (36, 84), "xylo": (67, 108), "tbn_sus": (40, 65), "fl_sus": (60, 96), "marimba": (48, 96),
            "bsn_sus": (34, 72), "glock": (79, 108)}
    ptb = {"hn_sus": "brs", "tpt_mute": "brs", "tbn_sus": "brs", "ob_sus": "ww", "cl_sus": "ww", "fl_sus": "ww",
           "bsn_sus": "ww", "vln_solo": "str", "organ_loud": "org", "xylo": "keys", "marimba": "keys", "glock": "keys"}
    gains = {"hn_sus": 0.5, "tpt_mute": 0.45, "ob_sus": 0.4, "cl_sus": 0.4, "vln_solo": 0.35, "organ_loud": 0.18,
             "xylo": 0.07, "tbn_sus": 0.45, "fl_sus": 0.4, "marimba": 0.09, "bsn_sus": 0.45, "glock": 0.05}
    tiers = scene_events("s06", "thunk", "tier", [])
    first = scene_events("s06", "stack", None, [125.95])[0]
    ent = ([first] + tiers)[:12] if len(tiers) >= 6 else [125.5 + 0.85 * i for i in range(12)]
    ent = ent + [ent[-1] + 0.4 * (k + 1) for k in range(12 - len(ent))]
    for i, (key, tr) in enumerate(voices):
        tv = ent[i]
        nl = 0.3 + 0.13 * rng.random()
        lo, hi = rngs[key]
        notes = [x + tr - (12 if tr > 5 else 0) for x in th("maj")]
        while min(notes) < lo:
            notes = [x + 12 for x in notes]
        while max(notes) > hi:
            notes = [x - 12 for x in notes]
        pan = float(np.clip(-0.9 + 1.8 * ((i * 5) % 12) / 11, -0.9, 0.9))
        one = key in ("xylo", "marimba", "glock")
        t, j = tv, 0
        while t < t_end - 0.02:
            d = nl * (2 if j % 7 == 6 else 1)
            lv = (0.5 + 0.5 * (t - 125.5) / (t_end - 125.5)) * dip(t)
            if t >= tc:
                lv = 1.0
            I(key).note(b, ptb[key], t, notes[j % 7], None if one else min(d, t_end - t), 0.5 + 0.4 * lv, pan,
                        gain=gains[key] * lv, att=0.01, rel=0.08, seed=j)
            t += d
            j += 1
    # two grinding pulses: 114.3 BPM taiko/spiccato (from the breach) and 95.2 BPM machine bass
    t, k = ts0, 0
    while t < t_end - 0.01:
        lv = (0.4 + 0.6 * (t - ts0) / (t_end - ts0)) * dip(t)
        if k % 8 in (0, 3, 6):
            taiko(b, t, "f" if t < tc else "ff", gain=0.5 * lv, sticks=k % 8 == 6)
        I("vc_spic").note(b, "str", t, "D3" if k % 8 != 6 else "Eb3", None, 0.5 + 0.3 * lv, PAN["vc"], gain=0.4 * lv, seed=k)
        t += B5 / 4
        k += 1
    t, k = 125.5, 0
    while t < t_end - 0.01:
        lv = (0.4 + 0.6 * (t - 125.5) / (t_end - 125.5)) * dip(t)
        I("cb_spic").note(b, "str", t, "D2" if k % 4 else "Ab1", None, 0.7, PAN["cb"], gain=0.45 * lv, seed=k)
        t += B2 / 2
        k += 1
    # lightning wakes Lord Egregore (screens power on): a cold low cluster swell
    tw_ = scene_events("s06", "power_on", "Egregore", [129.0])[0]
    brass(b, tw_, 1.05, dyn=[(0, 0.95), (0.25, 0.6), (1.05, 0.25)], att=0.0, rel=0.3, tbn=["D2", "Eb2", "A2"],
          tuba=["D2"], hn=["D3", "Eb3", "Ab3", "A3"], gain=0.8)
    timp(b, tw_, "D2", 0.9, gain=0.9)
    taiko(b, tw_, "ff", gain=0.7)
    hit(tw_, "egregore_wakes", "low brass cluster as Egregore's screens power on")
    # C03: Crocus' flag = the one steady, glowing thing: a pure D over the chaos
    a, z = LINES["C03"]["start"] - 0.2, tc + 0.1
    tn = tt(z - a)
    y = np.sin(2 * np.pi * hz(m("D6")) * tn) + 0.3 * np.sin(2 * np.pi * hz(m("D5")) * tn)
    y *= envelope([(0, 0), (0.4, 1), (z - a - 0.1, 1), (z - a, 0)], len(tn))
    b.add("space", np.stack([y, y], 1), t=a, gain=db(-24))
    I("organ_softped").note(b, "org", a, "D2", z - a, 0.8, 0.0, gain=0.4, att=0.4, rel=0.1)
    # 136.0 babel_crack: cluster hit, then everything bends upward
    tutti(b, tc, "Dm", 1.0, gong=True)
    brass(b, tc, None, "stac", 1.0, hn=["D4", "Eb4", "E4", "F4"], tbn=["C#3", "D3", "Eb3"], gain=0.9)
    hit(tc, "babel_crack", "cluster hit; max chaos")
    strings(b, tc, t_end - tc, "trem", dyn=[(0, 0.6), (t_end - tc, 1.0)], att=0.0, rel=0.01,
            v1=["D5", "Eb5", "E5", "F5", "F#5"], va=["A3", "Bb3", "B3"], vc=["D3", "Eb3"],
            bend=lambda x: 5.0 * (x / 2.5) ** 1.6, gain=0.6)
    brass(b, tc, t_end - tc, dyn=[(0, 0.6), (t_end - tc, 1.0)], att=0.0, rel=0.01, hn=["A3", "Bb3", "D4"],
          tbn=["D2", "Eb2"], bend=lambda x: 4.0 * (x / 2.5) ** 1.6, gain=0.7)
    r = riser(t_end - tc, 300, 12000, q=1.2, curve=1.6, seed=13, tonal=(m("D3"), m("D6")))
    b.add("fx", r, t=tc, gain=db(-17))
    # stutter the last 1.3 s of every bus (same pattern everywhere) and HARD STOP at 138.5
    ss = S(t_end - 1.3)
    pos = ss
    seglen = 0.11
    pattern = []
    while pos < S(t_end):
        L = S(seglen)
        reps = 2 if seglen > 0.05 else 3
        pattern.append((pos, L, reps))
        pos += L * reps
        seglen = max(0.025, seglen * 0.82)
    for kb in b.b:
        x = b.b[kb]
        src = x[ss - S(0.4):S(t_end)].copy()
        off = S(0.4) - ss
        for (p, L, reps) in pattern:
            chunk = src[p + off - S(0.2): p + off - S(0.2) + L].copy()
            ramp = min(64, L // 4)
            chunk[:ramp] *= np.linspace(0, 1, ramp)[:, None]
            chunk[-ramp:] *= np.linspace(1, 0, ramp)[:, None]
            for q in range(reps):
                e = min(p + (q + 1) * L, S(t_end))
                if e > p + q * L:
                    x[p + q * L:e] = chunk[:e - p - q * L] * (1.0 + 0.1 * q)
        x[S(t_end) - 96:S(t_end)] *= np.linspace(1, 0, 96)[:, None]
        x[S(t_end):] = 0
    hit(t_end, "silence", "HARD STOP to digital zero", kind="cut")


# ================================================================================================ s07a
def string_harmonic(f, dur, amp_bp, seed=0):
    rng = np.random.default_rng(seed)
    tn = tt(dur)
    flut = 1 + 0.03 * np.sin(2 * np.pi * 0.23 * tn) + 0.02 * np.sin(2 * np.pi * 0.61 * tn + 1)
    y = np.sin(2 * np.pi * f * tn) + 0.04 * np.sin(2 * np.pi * 2 * f * tn + 0.4)
    bow = svf(rng.standard_normal(len(tn)).astype(np.float32), np.full(len(tn), f), 25.0, 1)
    y = y + 0.05 * bow / (np.abs(bow).max() + 1e-9)
    y *= flut * envelope(amp_bp, len(tn))
    return np.stack([y, y * 0.97], 1).astype(np.float32)


def s07a(b):
    """The Unbabeling (140.0–154.5): from digital silence a single high string harmonic (A6 = the theme's first
    note) swells; the planted Flag (142.0) opens a dawn chord; celesta on 'Flag'; solo violin begins the theme on
    'Each of them alone', the section joins on 'together'; convergence 150.0: plucks multiply like a phase
    transition; the theme lands on D exactly at the hymn downbeat 154.5."""
    rng = np.random.default_rng(707)
    th_ = CUE["hymn"]
    y = string_harmonic(hz(m("A6")), th_ + 0.6 - 140.0,
                        [(0, 0), (1.0, 0.25), (2.0, 0.3), (8.0, 0.45), (10.0, 0.6), (14.5, 0.9), (15.1, 0.0)], 1)
    b.add("space", y, t=140.0, gain=db(-27))
    b.add("str", y, t=140.0, gain=db(-33))
    hit(140.0, "harmonic_in", "single high string harmonic A6")
    strings(b, 141.0, 13.5, dyn=[(0, 0.08), (2, 0.18), (13.5, 0.4)], att=2.0, rel=0.5, cb=["D2"], gain=0.5)
    tp = CUE["flag_planted"]
    I("harp").note(b, "keys", tp, "D2", None, 0.6, -0.3, gain=0.3)
    I("harp").note(b, "keys", tp, "D3", None, 0.5, -0.3, gain=0.25)
    timp(b, tp, "D2", 0.6, gain=0.8)
    I("cb_pizz").note(b, "str", tp, "D2", None, 0.7, PAN["cb"], gain=0.5)
    I("vc_pizz").note(b, "str", tp, "D3", None, 0.7, PAN["vc"], gain=0.5)
    hit(tp, "flag_planted", "soft low D (harp/timp/pizz) under the thunk")
    snap = scene_events("s07a", "snap", "first sunlight", [tp + 0.25])[0]
    for j, n in enumerate(["D3", "A3", "D4", "F#4", "A4", "D5", "E5", "F#5", "A5"]):
        I("harp").note(b, "keys", tp + 0.02 + j * (snap - tp - 0.02) / 8, n, None, 0.45 + 0.03 * j, -0.5 + 0.1 * j,
                       gain=0.22, seed=j)
    cel = sf2part(SF_MS, 0, 8, gain=1.6)
    cel.note(snap, m("A5"), 0.8, 0.5)
    cel.note(snap, m("D6"), 0.8, 0.4)
    I("glock").note(b, "keys", snap, "A6", None, 0.4, 0.2, gain=0.05)
    strings(b, tp + 0.3, 7.3, dyn=[(0, 0.08), (2.5, 0.22), (7.3, 0.25)], att=1.5, rel=0.6, v1=["D5", "A5"], vc=["D3"],
            gain=0.5)
    strings(b, tp + 0.3, 3.2, dyn=[(0, 0.08), (2.5, 0.2), (3.2, 0.2)], att=1.5, rel=0.5, va=["F#4"], gain=0.5)
    strings(b, tp + 3.5, 1.1, dyn=[(0, 0.2), (1.1, 0.2)], att=0.4, rel=0.5, va=["G4"], gain=0.5)
    strings(b, tp + 4.6, 3.0, dyn=[(0, 0.2), (3.0, 0.22)], att=0.4, rel=0.6, va=["F#4"], gain=0.5)
    tf = W("C04", "Flag", 1)
    cel.note(tf, m("D6"), 1.2, 0.5)
    I("harp").note(b, "keys", tf, "D5", None, 0.45, -0.4, gain=0.22)
    # 148.5 survivors look up: distant horns
    brass(b, 148.5, th_ - 148.5 - 0.05, dyn=[(0, 0.05), (2.0, 0.25), (th_ - 148.5 - 0.4, 0.65), (th_ - 148.5 - 0.05, 0.35)], att=0.8, rel=0.05,
          hn=["D4", "A4"], gain=0.5)
    # the theme: solo violin (alone) → section (together) → D on the hymn downbeat
    tw = [LINES["N08"]["start"], W("N08", "alone"), W("N08", "All"), W("N08", "together"), 152.2, 153.2, th_]
    for j, (tj, n) in enumerate(zip(tw[:6], th("maj", 1)[:6])):
        d = tw[j + 1] - tj
        if j < 2:
            I("vln_solo").note(b, "str", tj, n, d + 0.05, 0.5, -0.15, gain=0.5, dyn=[(0, 0.35), (d, 0.45)], att=0.12, rel=0.3)
        else:
            lv = 0.35 + 0.1 * j
            strings(b, tj, d + 0.03, dyn=[(0, lv), (d, lv + 0.08)] if j < 6 else [(0, 0.6), (d, 0.5)], att=0.08, rel=0.4,
                    v1=[n], gain=0.7)
            I("vln_solo").note(b, "str", tj, n, d + 0.03 if j < 5 else d - 0.05, 0.6, -0.15, gain=0.35,
                               dyn=[(0, lv), (d, lv)] if j < 5 else [(0, lv), (d - 0.4, lv), (d, lv * 0.5)], att=0.08,
                               rel=0.3 if j < 5 else 0.05)
    # D major swell (pedal) under the convergence
    strings(b, 150.0, th_ - 150.0 - 0.04, dyn=[(0, 0.15), (th_ - 150.0 - 0.4, 0.72), (th_ - 150.0 - 0.04, 0.3)], att=0.8, rel=0.05, v2=["F#5", "D5"],
            va=["A4", "F#4"], vc=["D3", "A3"], cb=["D2"], gain=0.6)
    for n in ["D3", "A3", "D4", "F#4"]:
        I("organ_soft").note(b, "org", 152.0, n, th_ - 152.0 - 0.05, 0.8, 0.0, gain=0.22, att=1.2, rel=0.04)
    I("organ_softped").note(b, "org", 152.5, "D2", th_ - 152.5 - 0.05, 0.8, 0.0, gain=0.4, att=0.8, rel=0.04)
    # convergence: plucks, one by one then a flood (rate grows exponentially), pans spreading outward
    tcv = CUE["convergence"]
    hit(tcv, "convergence", "first pluck; flood begins")
    cel.note(tcv, m("A5"), 0.6, 0.9)
    glass_note(b, tcv, "A6", 1.5, gain=db(-18), pan=0.0, decay=0.8)
    I("harp").note(b, "keys", tcv, "A4", None, 0.7, -0.3, gain=0.45)
    t, q = tcv, 0
    dmaj = [2, 4, 6, 9, 11]
    while t < th_ - 0.12:
        x = (t - tcv) / (th_ - tcv)
        n = 12 * (int(rng.integers(5, 8)) + 1) + int(rng.choice(dmaj))
        pan = float(np.clip(rng.normal(0, 0.1 + 0.8 * x), -1, 1))
        g = 0.6 + 0.4 * x
        choice = q % 4
        if q == 0:
            n = m("A5")
        if choice == 0:
            I("harp").note(b, "keys", t, min(n, 100), None, 0.5, pan, gain=0.14 * g, seed=q)
        elif choice == 1:
            cel.note(t, min(n, 96), 0.3, 0.3 + 0.2 * g)
        elif choice == 2:
            I("vln_pizz").note(b, "str", t, min(n - 12, 86), None, 0.5, pan, gain=0.22 * g, seed=q)
        else:
            glass_note(b, t, n, 1.2, gain=db(-34) * g, pan=pan, decay=0.6)
        rate = 1.3 * np.exp(3.9 * x)
        t += 1.0 / rate
        q += 1
    cel.to(b, "keys", pan=0.3)
    cut_after(b, th_ - 0.03, 0.25)
    # the theme's D lands with the choir on the hymn downbeat
    strings(b, th_, 1.2, dyn=[(0, 0.6), (1.2, 0.5)], att=0.01, rel=0.4, v1=["D6"], gain=0.7)
    timp_roll(b, 152.8, th_ - 152.8 - 0.04, "D2", [(0, 0.08), (th_ - 152.8 - 0.3, 0.75), (th_ - 152.8 - 0.04, 0.5)], gain=0.7)
    swell_to(b, th_ - 0.02, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.14, after=0.0)


# ================================================================================================ s07b
BH = 60.0 / 84.0


def s07b(b):
    """Flagistan (154.5–173.0): the hymn accompanied (organ + strings doubling the SATB, strict 84 BPM,
    mp→mf→f→ff, timpani on phrase downbeats, brass only on the final chord), ember glissando, then the
    gateless-gate climax: the theme augmented in trumpets/horns, bVI–bVII–I lift landing on 'completed', radiant
    fade into the white-gold flash at 173.0."""
    h = TLJ["hymn"]
    t0 = h["start"]
    bt = 60.0 / h["bpm"]
    dyn_of = {0: 0.5, 2: 0.55, 4: 0.65, 6: 0.7, 8: 0.8, 10: 0.85, 12: 1.0}
    for (w, b0, nb, satb) in h["notes"]:
        a = t0 + b0 * bt
        d = nb * bt
        lv = dyn_of[b0]
        s_, al, te, bs = satb
        last = b0 == 12
        dd = d - (0.1 if (b0 + nb) in (4, 8, 12) else 0.0)  # breath before each phrase
        for j, n in enumerate([s_, al, te, bs]):
            I("organ_loud").note(b, "org", a, n, dd, 0.8, -0.25 + 0.17 * j, gain=0.18 * (0.6 + 0.5 * lv), att=0.03, rel=0.08 if dd < d else 0.35)
        I("organ_pedal").note(b, "org", a, bs - 12, dd, 0.8, 0.0, gain=0.4 * (0.6 + 0.5 * lv), att=0.03, rel=0.08 if dd < d else 0.4)
        dyn = [(0, lv * 0.75), (d * 0.5, lv * 0.9), (d, lv * 0.8)] if not last else [(0, 0.8), (d, 1.0)]
        strings(b, a, dd, dyn=dyn, att=0.015, rel=0.35 if dd < d else 0.45, v1=[s_ + 12], v2=[al + 12 if al + 12 <= 86 else al], vc=[bs], cb=[bs - 12],
                gain=0.9)
        if b0 >= 8:
            strings(b, a, dd, dyn=dyn, att=0.06, rel=0.45, va=[te + 12], gain=0.45)
        if b0 > 0:
            harp_roll(b, a, [bs, te + 12, al + 12, s_ + 12], 0.45 + 0.2 * lv, 0.0, gain=0.9)
    hit(t0, "hymn", "hymn downbeat (organ+strings+timpani)")
    for n in ["D3", "A3", "D4", "F#4", "A4"]:
        I("harp").note(b, "keys", t0, n, None, 0.7, PAN["harp"], gain=0.22)
    I("glock").note(b, "keys", t0, "D6", None, 0.5, 0.3, gain=0.06)
    perc(b, t0, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_mp.wav", 0.35, -0.1)
    for (b0, n, v) in [(0, "D2", 0.85), (4, "G2", 0.75), (8, "A2", 0.85), (12, "D2", 0.95)]:
        timp(b, t0 + b0 * bt, n, v, gain=1.3 if b0 == 0 else 1.1)
        hit(t0 + b0 * bt, f"hymn_phrase_{b0 // 4 + 1}", "timpani on phrase downbeat")
    t12 = t0 + 12 * bt
    tend = t0 + 16 * bt
    brass(b, t0 + 8 * bt, 4 * bt, dyn=[(0, 0.4), (4 * bt, 0.55)], att=0.1, rel=0.3, hn=["A3", "E4"], gain=0.5)
    brass(b, t12, tend - t12, dyn=[(0, 0.55), (tend - t12, 0.95)], att=0.05, rel=0.35, tpt=["A4", "D5", "F#5"],
          hn=["D4", "F#4", "A4"], tbn=["D3", "A3"], tuba=["D2"], gain=0.7)
    swell_to(b, t12 - 0.01, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.3, after=0.0)
    crash(b, t12, "mp", 0.7)
    I("glock").note(b, "keys", t12, "D6", None, 0.7, 0.3, gain=0.08)
    I("glock").note(b, "keys", t12, "A6", None, 0.6, 0.1, gain=0.06)
    timp_roll(b, 164.6, CUE["gateless_gate"] - 164.6, "D2", [(0, 0.2), (CUE["gateless_gate"] - 164.6, 1.0)], gain=0.8)
    # embers: flags lift off (165.93 → 166.3)
    harp_gliss(b, tend, CUE["gateless_gate"] - 0.02, "D4", "D7", vel=0.6, gain=1.2)
    cel = sf2part(SF_MS, 0, 8, gain=1.6)
    for j, n in enumerate(["A5", "D6", "F#6", "A6", "D7"]):
        cel.note(tend + 0.07 * j, m(n), 0.4, 0.45)
    # ---- 166.3 GATELESS GATE climax
    tg = CUE["gateless_gate"]
    tcomp = W("N10", "completed")
    tutti(b, tg, "D", 1.0, gong=True)
    for n in ["D6", "A6", "D7"]:
        I("glock").note(b, "keys", tg, n, None, 0.8, 0.25, gain=0.08)
    for n in ["D3", "A3", "D4", "F#4", "A4", "D5"]:
        I("harp").note(b, "keys", tg, n, None, 0.8, PAN["harp"], gain=0.2)
    hit(tg, "gateless_gate", "tutti bloom; augmented theme")
    mel_t = [tg, 167.1, 167.55, 168.35, 168.8, 169.6, 170.05, LINES["N10"]["start"], 171.35, tcomp]
    mel = th("maj", 0) + [m("D5"), m("E5"), m("F#5")]
    harm = ["D", "D", "G", "Em", "A", "A7", "D", "Bb", "C", "D"]
    bass = ["D2", "F#2", "G2", "E2", "A2", "A2", "D2", "Bb1", "C2", "D2"]
    for j in range(10):
        a = mel_t[j]
        z = mel_t[j + 1] if j < 9 else 173.0
        d = z - a
        lv = 0.9 if j < 7 else (0.75 if j < 9 else 0.8)
        fin = [(0, lv), (d, lv)] if j < 9 else [(0, 0.85), (0.6, 0.6), (d, 0.25)]
        brass(b, a, d + 0.03, dyn=fin, att=0.03, rel=0.35, tpt=[mel[j]], hn=[mel[j] - 12], gain=1.0)
        brass(b, a, d + 0.03, dyn=[(0, lv * 0.7), (d, lv * 0.7)] if j < 9 else [(0, 0.6), (d, 0.15)], att=0.05, rel=0.35,
              tbn=[root(harm[j], "C3")], tuba=[m(bass[j])] if m(bass[j]) >= 34 else [m(bass[j]) + 12], gain=0.6)
        top = mel[j] + 12
        if top <= 86:
            strings(b, a, d + 0.03, dyn=fin, att=0.03, rel=0.4, v1=[top], gain=0.6)
        else:
            I("fl_sus").note(b, "ww", a, top, d + 0.03, 0.8, PAN["fl"], gain=0.5, dyn=fin, att=0.03, rel=0.4)
            I("vln_solo").note(b, "str", a, top, d + 0.03, 0.8, PAN["vln"], gain=0.45, dyn=fin, att=0.03, rel=0.4)
        strings(b, a, d + 0.03, dyn=fin, att=0.03, rel=0.4, v2=tones(harm[j], "A4", 2), va=tones(harm[j], "D4", 2),
                vc=[m(bass[j]) + 12], cb=[m(bass[j])], gain=0.55)
        for n in tones(harm[j], "D4", 3):
            I("organ_loud").note(b, "org", a, n, d + 0.03, 0.8, 0.0, gain=0.1 if j < 9 else 0.06, att=0.03, rel=0.5)
        I("organ_pedal").note(b, "org", a, m(bass[j]), d + 0.03, 0.8, 0.0, gain=0.25 if j < 9 else 0.12, att=0.03, rel=0.5)
    for (a, n, v) in [(170.05, "D2", 0.8), (LINES["N10"]["start"], "Bb2", 0.8), (171.35, "C3", 0.85), (tcomp, "D2", 1.0)]:
        timp(b, a, n, v, gain=0.7 if a < tcomp else 1.1)
    hit(tcomp, "survey_completed", "bVI-bVII-I lands on D ('completed')")
    cho = sf2part(SF_MS, 0, 52, gain=1.3)
    for n in ["D5", "F#5", "A5"]:
        cho.note(tg, m(n), 170.9 - tg, 0.7)
    for n in ["D5", "F5", "Bb5"]:
        cho.note(170.9, m(n), 0.45, 0.7)
    for n in ["E5", "G5", "C6"]:
        cho.note(171.35, m(n), 0.46, 0.7)
    for n in ["D5", "F#5", "A5", "E6"]:
        cho.note(tcomp, m(n), 173.0 - tcomp, 0.6)
    cho.to(b, "org", pan=0.0, gain=0.8)
    swell_to(b, tcomp - 0.01, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.3, after=0.0)
    perc(b, tcomp, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_f.wav", 0.35, 0.1)
    # radiant shimmer into the white-gold flash
    for j in range(14):
        tj = tcomp + 0.08 * j
        cel.note(tj, m(["D7", "A6", "F#6", "E6", "D6", "A5"][j % 6]) + (12 if j % 5 == 0 else 0), 0.3, 0.35 * 0.9 ** j)
    strings(b, tcomp, 173.2 - tcomp, "trem", dyn=[(0, 0.3), (0.6, 0.25), (173.2 - tcomp, 0.08)], att=0.2, rel=0.3,
            v1=["A5", "D6"], gain=0.4)
    cel.to(b, "keys", pan=0.3)


# ================================================================================================ s08
def s08(b):
    """The Sticky Flag (173.0–193.0): morning (flute + harp), pizzicato comedy that punctuates the koan between
    lines (run-up, ta-da, sticky strand, the theme quoted on celesta after 'all Flags are one Flag', the sniff),
    a warm horn statement of the theme under N11 to the 'Glorious!' tutti 188.3, the IVG end-card sting 189.5,
    the innocent theme once more, silence by 193.0."""
    rng = np.random.default_rng(808)
    t0 = 173.0
    strings(b, t0, 1.4, dyn=[(0, 0.15), (1.4, 0.12)], att=0.3, rel=0.6, v1=["D5", "A5"], va=["F#4"], vc=["D3"], gain=0.45)
    I("fl_sus").note(b, "ww", t0 + 0.1, "A5", 1.0, 0.5, PAN["fl"], gain=0.4, dyn=[(0, 0.2), (1.0, 0.3)], att=0.2, rel=0.5)
    harp_roll(b, t0 + 0.05, ["D3", "A3", "D4", "F#4", "A4", "D5"], 0.4, 0.07, gain=0.8)
    # Lutie runs up: pizz run
    run = ["D5", "E5", "F#5", "G5", "A5", "B5", "C#6", "D6"]
    tl4 = LINES["L04"]["start"]
    for j, n in enumerate(run):
        I("vln_pizz").note(b, "str", tl4 - 0.56 + j * 0.07, n, None, 0.5 + 0.04 * j, PAN["vln"], gain=0.4, seed=j)
    I("vc_pizz").note(b, "str", tl4 - 0.56, "D3", None, 0.6, PAN["vc"], gain=0.4)
    cel = sf2part(SF_MS, 0, 8, gain=1.8)
    tt_ = LINES["L04"]["end"] + 0.05
    cel.note(tt_, m("A5"), 0.15, 0.55)
    cel.note(tt_ + 0.13, m("D6"), 0.5, 0.6)
    I("vla_pizz").note(b, "str", tt_ + 0.13, "D4", None, 0.5, PAN["vla"], gain=0.3)
    hit(tt_ + 0.13, "L04_tada", "celesta ta-da")
    # light tiptoe bed (marimba + pizz, 100 BPM, pp) under the first exchange and Lutie's protest;
    # it stops dead for the punchlines (the strand snap, "This one is sticky.")
    bt8 = 0.3
    mar = [("D5", "A4"), ("F#5", "D5"), ("A5", "F#4"), ("F#5", "D5"), ("E5", "G4"), ("G5", "B4"), ("B5", "G4"), ("A5", "C#5")]
    for (a, z, lv) in [(175.45, 178.75, 0.8), (179.35, 181.6, 1.0)]:
        t, k = a, 0
        while t < z - 0.05:
            if k % 2 == 0:
                hi_, _ = mar[(k // 2) % 8]
                I("marimba").note(b, "keys", t, hi_, None, 0.45, 0.25, gain=0.055 * lv, seed=k)
            else:
                _, lo_ = mar[(k // 2) % 8]
                I("vln_pizz").note(b, "str", t, lo_, None, 0.35, PAN["vln"], gain=0.2 * lv, seed=k)
            if k % 4 == 0:
                I("vc_pizz").note(b, "str", t, "D3" if (k // 8) % 2 == 0 else "A2", None, 0.4, PAN["vc"], gain=0.25 * lv, seed=k)
            t += bt8
            k += 1
    # sticky strand: a slow stretching glissando (solo violin, sul pont.) then a droopy pizz + bassoon blup
    st0 = scene_events("s08", "stretch", None, [177.45])[0]
    st1 = scene_events("s08", "snap", "strand", [178.9])[0]
    dd = st1 - st0
    I("vln_solo").note(b, "str", st0, "G5", dd, 0.25, 0.2, gain=0.3, bend=lambda x: 5.0 * (x / dd) ** 1.6,
                       dyn=[(0, 0.08), (dd * 0.8, 0.2), (dd, 0.25)], att=0.2, rel=0.02)
    I("vln_pizz").note(b, "str", st1, "C6", None, 0.7, PAN["vln"] + 0.4, gain=0.5)
    I("vc_pizz").note(b, "str", st1 + 0.06, "F3", None, 0.6, PAN["vc"], gain=0.45, bend=lambda x: -2.0 * np.clip(x / 0.35, 0, 1))
    I("bsn_stac").note(b, "ww", st1 + 0.06, "F2", None, 0.6, PAN["bsn"], gain=0.5)
    hit(st1, "C05_sticky", "sticky strand snaps: pizz plink, droopy pizz + bassoon blup")
    # 'all Flags are one Flag!' → the theme quoted, quick, on celesta + pizz
    tq = LINES["L05"]["end"] + 0.05
    for j, n in enumerate(th("maj", 1)):
        cel.note(tq + j * 0.075, n, 0.12 if j < 6 else 0.4, 0.5)
        if j == 6:
            I("vln_pizz").note(b, "str", tq + j * 0.075, n - 12, None, 0.5, PAN["vln"], gain=0.35)
    hit(tq, "L05_theme_quote", "celesta theme quote")
    # C06 → beat of silence → Lutie sniffs her hand (two questioning clarinet notes)
    sn = scene_events("s08", "sniff", None, [LINES["C06"]["end"] + 0.35, LINES["C06"]["end"] + 0.6])
    tsn = sn[0]
    I("cl_stac").note(b, "ww", sn[0], "E5", None, 0.4, PAN["cl"], gain=0.4)
    I("cl_stac").note(b, "ww", sn[1] if len(sn) > 1 else sn[0] + 0.25, "F5", None, 0.45, PAN["cl"], gain=0.4)
    hit(tsn, "sniff", "clarinet sniff-sniff")
    # N11 warm swell: horns state the theme → 'Glorious!'
    tg = CUE["glorious"]
    n11 = LINES["N11"]["start"]
    mel_t = [n11, 185.55, W("N11", "signifier"), 186.6, W("N11", "glorious"), 187.65, tg]
    harm = ["D", "D", "G", "Em", "A", "A7", "D"]
    bass = ["D3", "D3", "G2", "E3", "A2", "A2", "D2"]
    for j in range(6):
        a, z = mel_t[j], mel_t[j + 1]
        d = z - a
        lv = 0.3 + 0.08 * j
        brass(b, a, d + 0.04, dyn=[(0, lv), (d, lv + 0.05)], att=0.06, rel=0.25, hn=[th("maj", 0)[j]], gain=0.85)
        strings(b, a, d + 0.04, dyn=[(0, lv * 0.8), (d, lv * 0.9)], att=0.1, rel=0.3, v2=tones(harm[j], "D5", 2),
                va=tones(harm[j], "F#4", 1), vc=[bass[j]], cb=[m(bass[j]) - 12] if m(bass[j]) - 12 >= 30 else [],
                gain=0.7)
        harp_roll(b, a, tones(harm[j], "D4", 3), 0.35, 0.05, gain=0.6)
    timp_roll(b, 187.3, tg - 187.3, "A2", [(0, 0.1), (tg - 187.3, 0.85)], gain=0.7)
    swell_to(b, tg, V1P + "varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav", gain=0.3, after=0.1)
    # 188.3 GLORIOUS! tutti + rippling salute (fanfare triplets, harp gliss)
    tutti(b, tg, "D", 1.0, sus=1.15)
    hit(tg, "glorious", "final tutti")
    for j, n in enumerate(["D5", "F#5", "A5"]):
        brass(b, tg + 0.3 + j * 0.1, None, "stac", 0.85, tpt=[n], gain=0.7)
    brass(b, tg + 0.6, 0.55, dyn=[(0, 0.9), (0.55, 0.8)], att=0.0, rel=0.25, tpt=["A5", "F#5"], hn=["D5", "A4"], gain=0.8)
    harp_gliss(b, tg, tg + 0.6, "D4", "D7", vel=0.7, gain=1.3)
    # 189.5 IVG end-card sting: crisp staccato D + glock/triangle/celesta
    te = CUE["end_card"]
    brass(b, te, None, "stac", 0.8, tpt=["D5", "F#5", "A5"], hn=["D4", "A4"], tbn=["D3"], gain=0.8)
    strings(b, te, None, "pizz", 0.8, v1=["D5", "A5"], va=["F#4", "D4"], vc=["D3"], cb=["D2"], gain=0.9)
    timp(b, te, "D2", 0.7, gain=0.7)
    perc(b, te, PERC + "Triangle3-Hit_v2_rr1_Sum.wav", 0.3, 0.3)
    for j, n in enumerate(["D6", "A6", "D7"]):
        I("glock").note(b, "keys", te + 0.02 * j, n, None, 0.75, 0.3, gain=0.1)
    hit(te, "end_card", "IVG end-card sting")
    # under the card: soft D major + the innocent theme once more; silent by 193.0
    strings(b, te + 0.25, 193.0 - te - 0.3, dyn=[(0, 0.25), (1.0, 0.2), (193.0 - te - 0.3, 0.0)], att=0.4, rel=0.05,
            v1=["A5", "D6"], va=["F#4"], vc=["D3"], cb=["D2"], gain=0.45)
    fl = scene_events("s08", "slam", "end card letter", [190.1, 190.19, 190.256, 190.346, 190.412])
    for j, (tj, n) in enumerate(zip(fl, ["D5", "F#5", "A5", "D6", "A6"])):
        cel.note(tj, m(n), 0.3 if j < 4 else 0.8, 0.55)
        I("marimba").note(b, "keys", tj, m(n) - 12, None, 0.6, 0.3, gain=0.1, seed=j)
        I("vln_pizz").note(b, "str", tj, min(m(n) - 12, 86), None, 0.5, PAN["vln"], gain=0.35, seed=j)
        hit(tj, f"endcard_letter_{j + 1}", "pluck on end-card letter")
    e = 0.2
    for j, (k, n) in enumerate(zip((0, 2, 3, 5, 6, 8, 9), th("maj", 1))):
        tj = 190.55 + e * k
        cel.note(tj, n, 0.4 if j < 6 else 1.2, 0.45 if j < 6 else 0.5)
        glass_note(b, tj, n + 12, 1.4, gain=db(-30), pan=0.2 * np.sin(j), decay=0.9)
    harp_roll(b, 190.55 + e * 9, ["D3", "A3", "D4", "F#4", "A4", "D5"], 0.35, 0.08, gain=0.6)
    cel.to(b, "keys", pan=0.3)


SECTIONS = dict(s01=s01, s02=s02, s03=s03, s04=s04, s05a=s05a, s05b=s05b, s06=s06, s07a=s07a, s07b=s07b, s08=s08)


# ================================================================================================ master
BUSES = {
    #        gain(dB)  reverb     wet    eq
    "str":   (0.0, "hall", 0.30, [("hp", 32), ("hs", 7000, 1.0)]),
    "brs":   (-1.0, "hall", 0.38, [("hp", 40), ("pk", 2500, -1.5, 0.8)]),
    "ww":    (-2.0, "hall", 0.33, [("hp", 80)]),
    "perc":  (-1.0, "hall", 0.30, [("hp", 55)]),  # Foley owns the sub (< 60 Hz)
    "keys":  (-2.0, "hall", 0.40, [("hp", 60)]),
    "org":   (-2.0, "cathedral", 0.50, [("hp", 28), ("pk", 550, -2.5, 0.7)]),
    "syn":   (-2.0, "plate", 0.28, [("hp", 25)]),
    "fx":    (-3.0, "hall", 0.22, [("hp", 40)]),
    "space": (-2.0, "void", 0.80, [("hp", 60)]),
    "tv":    (0.0, None, 0.0, []),
    "feed":  (-3.0, "room", 0.15, []),
}
# reverb tails do not cross these cuts: (t, fade-out seconds after t, buses or None=all)
CUTS = [(72.7, 0.015, None), (90.0, 0.0, None), (100.55, 0.05, None), (138.5, 0.0, None),
        (154.5, 0.15, ["space", "org", "str", "keys", "brs", "perc"])]
FOLEY_CUES = ["title_slam", "slop_crash", "tv_on", "hypermeme", "breach", "nation_fracture", "faith_fracture",
              "philo_shatter", "babel_crack", "flag_planted", "gateless_gate", "glorious"]


def dialogue_env():
    """0..1 envelope of dialogue presence (lines of timeline), smoothed (attack 60 ms, release 250 ms)."""
    fr = 1000
    n = int(DUR * fr)
    g = np.zeros(n)
    for l in TLJ["lines"]:
        a, z = int((l["start"] - 0.06) * fr), int((l["end"] + 0.12) * fr)
        g[max(0, a):min(n, z)] = 1.0
    out = np.zeros(n)
    s = 0.0
    for i in range(n):
        c = 1 - np.exp(-1 / (0.06 * fr)) if g[i] > s else 1 - np.exp(-1 / (0.25 * fr))
        s += (g[i] - s) * c
        out[i] = s
    return np.interp(np.arange(N) / SR, np.arange(n) / fr, out).astype(np.float32)


def lufs(x):
    import pyloudnorm as pyln
    return pyln.Meter(SR).integrated_loudness(x.astype(np.float64))


def true_peak(x):
    from scipy.signal import resample_poly
    return float(np.abs(resample_poly(x, 4, 1, axis=0)).max())


def limiter(x, ceiling_db=-1.2, look=0.004, rel=0.12):
    """lookahead peak limiter on 4x-oversampled peak estimate."""
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import resample_poly
    c = db(ceiling_db)
    up = np.abs(resample_poly(x, 4, 1, axis=0)).max(1)
    pk = up.reshape(-1, 4).max(1)[:len(x)]
    need = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    la = int(look * SR)
    need = -maximum_filter1d(-need, size=2 * la + 1)
    g = np.empty_like(need)
    s = 1.0
    ar = np.exp(-1 / (rel * SR))
    for i in range(len(need)):
        s = need[i] if need[i] < s else ar * s + (1 - ar) * need[i]
        g[i] = s
    g = np.convolve(g, np.ones(la) / la, "same")
    g = np.minimum(g, need)
    return (x * g[:, None]).astype(np.float32), float(g.min())


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
            if key not in buses:
                buses[key] = np.zeros((N, 2), np.float32)
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
            edges = [0] + [S(c[0]) for c in cuts] + [N]
            fades = [c[1] for c in cuts] + [0.0]
            for i, (a, z) in enumerate(zip(edges[:-1], edges[1:])):
                seg = np.zeros_like(x)
                seg[a:z] = x[a:z]
                if not np.any(seg[a:z]):
                    continue
                w = reverb_full(seg, rv, wet)
                if z < N:
                    fd = S(fades[i])
                    if fd > 0:
                        w[z:z + fd] *= np.linspace(1, 0, fd)[:, None] ** 2
                    w[z + fd:] = 0
                y = y + w * db(g)
        mix += y
        print(f"  bus {key:6s} peak {np.abs(y).max():.3f}", flush=True)
    # dialogue space: dip 250 Hz–4 kHz by up to 5 dB while lines play
    d = dialogue_env()
    lo = butter(mix, "lowpass", 250, 2, zero_phase=True)
    hi = butter(mix, "highpass", 4000, 2, zero_phase=True)
    mid = mix - lo - hi
    mix = lo + hi + mid * (1 - (1 - db(-5)) * d)[:, None]
    # loudness normalise, limit, sacred silence, final fade
    L0 = lufs(mix)
    mix *= db(-19.0 - L0)
    mix, gmin = limiter(mix, -1.2)
    mix[S(138.5):S(140.0)] = 0.0
    nf = N - S(191.8)  # the picture fades to black by 193.0: so does the music (dry + reverb)
    mix[S(191.8):] *= (np.cos(np.linspace(0, np.pi / 2, nf)) ** 2)[:, None]
    mix[-1] = 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sf.write(OUT, mix.astype(np.float32), SR, subtype="FLOAT")
    print(f"master: {time.time() - t0:.1f}s  LUFS pre {L0:.1f} -> {lufs(mix):.2f}  limiter min gain {20 * np.log10(gmin):.1f} dB"
          f"  true peak {20 * np.log10(true_peak(mix)):.2f} dBTP", flush=True)
    return mix


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
    ap.add_argument("--master", action="store_true", help="master only")
    a = ap.parse_args()
    names = list(SECTIONS) if not a.only else a.only.split(",")
    if not a.master:
        for n in names:
            render_section(n)
    files = [os.path.join(CACHE, f"sec_{n}.npz") for n in SECTIONS if os.path.exists(os.path.join(CACHE, f"sec_{n}.npz"))]
    hits = []
    for n in SECTIONS:
        p = os.path.join(CACHE, f"hits_{n}.json")
        if os.path.exists(p):
            hits += json.load(open(p))
    with open(os.path.join(ROOT, "audio", "music", "cues.json"), "w") as f:
        json.dump(sorted(hits, key=lambda h: h["t"]), f, indent=1)
    master(files)


if __name__ == "__main__":
    main()

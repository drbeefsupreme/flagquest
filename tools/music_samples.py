"""Index the VSCO2-CE multisamples (assets/sf2/vsco2) into playable instruments.

  python tools/music_samples.py        # (re)build audio/music/cache/sample_index.json

Each pitched sample is analysed (YIN f0 on the steady part) so the map uses the *measured* pitch:
the octave convention of every folder is inferred by majority vote and every sample gets a cents
correction so the whole orchestra plays in exact A=440 equal temperament. The attack onset of each
sample is detected too, so notes can be placed sample-accurately (hits land on picture).
"""
import os as _os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ.setdefault(_k, "4")  # shared machine: stay polite
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VS = os.path.join(ROOT, "assets", "sf2", "vsco2")
INDEX = os.path.join(ROOT, "audio", "music", "cache", "sample_index.json")

# key: (folder, file-regex filter or None, kind)  kind: sus (sustain, loopable) | one (one-shot pitched)
INSTR = {
    "vln_sus": ("Strings/Violin Section/susVib", None, "sus"),
    "vln_trem": ("Strings/Violin Section/Trem", None, "sus"),
    "vln_pizz": ("Strings/Violin Section/Pizz", None, "one"),
    "vln_spic": ("Strings/Violin Section/Spic", None, "one"),
    "vla_sus": ("Strings/Viola Section/susvib", None, "sus"),
    "vla_trem": ("Strings/Viola Section/trem", None, "sus"),
    "vla_pizz": ("Strings/Viola Section/pizz", None, "one"),
    "vla_spic": ("Strings/Viola Section/spic", None, "one"),
    "vc_sus": ("Strings/Cello Section/susvib", None, "sus"),
    "vc_trem": ("Strings/Cello Section/trem", None, "sus"),
    "vc_pizz": ("Strings/Cello Section/pizzT", None, "one"),
    "vc_spic": ("Strings/Cello Section/spic", None, "one"),
    "cb_sus": ("Strings/Solo Contrabass/SusVib", None, "sus"),
    "cb_pizz": ("Strings/Solo Contrabass/Pizz", None, "one"),
    "cb_spic": ("Strings/Solo Contrabass/Spic", None, "one"),
    "vln_solo": ("Strings/Solo Violin/Arco Vib", None, "sus"),
    "harp": ("Strings/Harp", None, "one"),
    "hn_sus": ("Brass/F Horn/sus", None, "sus"),
    "hn_stac": ("Brass/F Horn/stac", None, "one"),
    "tpt_sus": ("Brass/Trumpet/sus", None, "sus"),
    "tpt_stac": ("Brass/Trumpet/stac", None, "one"),
    "tpt_mute": ("Brass/Trumpet/straightM-sus", None, "sus"),
    "tbn_sus": ("Brass/Tenor Trombone/sus", None, "sus"),
    "tbn_stac": ("Brass/Tenor Trombone/stac", None, "one"),
    "tuba_sus": ("Brass/Tuba/sus", None, "sus"),
    "tuba_stac": ("Brass/Tuba/stac", None, "one"),
    "fl_sus": ("Woodwinds/Flute/susNV", None, "sus"),
    "fl_stac": ("Woodwinds/Flute/stac", None, "one"),
    "picc_stac": ("Woodwinds/Piccolo/Stac", None, "one"),
    "picc_sus": ("Woodwinds/Piccolo/Sus", None, "sus"),
    "ob_sus": ("Woodwinds/Oboe/Sus", None, "sus"),
    "cl_sus": ("Woodwinds/Clarinet/susLong", None, "sus"),
    "cl_stac": ("Woodwinds/Clarinet/stac", None, "one"),
    "bsn_stac": ("Woodwinds/Bassoon/stac", None, "one"),
    "bsn_sus": ("Woodwinds/Bassoon/sus", None, "sus"),
    "timp": ("Percussion/Timpani", None, "one"),
    "timp_roll": ("Percussion/Timpani/Rolls", None, "sus"),
    "glock": ("Percussion/Glock", None, "one"),
    "marimba": ("Percussion/Marimba", None, "one"),
    "xylo": ("Percussion/Xylo", None, "one"),
    "piano": ("Keys/Upright Nr1", None, "one"),
    "organ_loud": ("Keys/Organ/Loud", r"Man3Open", "sus"),
    "organ_pedal": ("Keys/Organ/Loud", r"Pedal", "sus"),
    "organ_soft": ("Keys/Organ/Quiet", r"Man3Quiet", "sus"),
    "organ_softped": ("Keys/Organ/Quiet", r"PedalQuiet", "sus"),
}

NOTE_RE = re.compile(r"(?<![A-Za-z])([A-G])(#|b)?(-?\d)(?=[_.])")
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DYN = {"ppp": 1, "pp": 2, "p": 3, "mp": 4, "mf": 5, "f": 6, "ff": 7, "fff": 8, "loud": 7}


def parse_note(fn):
    m = NOTE_RE.search(fn)
    if not m:
        return None
    n = PC[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)
    return 12 * (int(m.group(3)) + 1) + n


def parse_vel(fn):
    m = re.search(r"_v(\d+)", fn)
    if m:
        return int(m.group(1))
    for tok in re.split(r"[_.\-]", fn):
        if tok in DYN:
            return DYN[tok]
    return 1


def parse_rr(fn):
    m = re.search(r"rr(\d+)", fn, re.I)
    return int(m.group(1)) if m else 1


def load_mono(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(1), sr


def onset_index(x, sr):
    a = np.abs(x)
    pk = a.max() + 1e-12
    thr = pk * 10 ** (-40 / 20)
    i = int(np.argmax(a > thr))
    return max(0, i - int(0.001 * sr))


# measured by hand (spectral principal (1,1)-mode of each drum; YIN locks onto sub/overtones) and the
# organ files are numbered chromatically (numbers verified against YIN where it did not octave-slip)
TIMP_PRINCIPAL = {"Hit": {1: 41.46, 2: 46.77, 3: 49.48, 4: 52.43, 5: 54.58},
                  "Roll": {1: 42.2, 2: 47.6, 3: 49.3, 4: 52.4, 5: 55.6}}
ORGAN_OFFSET = {"Man3Open": 23, "Pedal_": 23, "Man3Quiet": -86, "PedalQuiet": -28}


def override_pitch(fn):
    mt = re.match(r"Timpani(\d)_(Hit|Roll)", fn)
    if mt:
        return TIMP_PRINCIPAL[mt.group(2)][int(mt.group(1))]
    for k, off in ORGAN_OFFSET.items():
        mo = re.search(k + r"_?(\d+)", fn)
        if mo and ("Rode" in fn or "NT5" in fn):
            return int(mo.group(1)) + off
    return None


def f0_of(x, sr, kind):
    import librosa
    i0 = onset_index(x, sr)
    n = len(x) - i0
    if kind == "sus":
        a, b = i0 + int(0.25 * n), i0 + int(0.6 * n)
    else:
        a, b = i0 + int(0.04 * sr), i0 + int(0.04 * sr) + int(0.5 * sr)
    seg = x[a:min(b, len(x))]
    if len(seg) < 4096:
        seg = x[i0:i0 + 8192]
    if len(seg) < 4096:
        seg = np.pad(seg, (0, 4096 - len(seg)))
    f = librosa.yin(seg.astype(np.float64), fmin=25, fmax=4200, sr=sr, frame_length=4096, trough_threshold=0.1)
    f = f[np.isfinite(f)]
    if len(f) == 0:
        return None
    return float(np.median(f))


def build():
    idx = {}
    for key, (folder, filt, kind) in INSTR.items():
        d = os.path.join(VS, folder)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav") and (not filt or re.search(filt, f)))
        rows = []
        for fn in files:
            p = os.path.join(d, fn)
            x, sr = load_mono(p)
            f0 = f0_of(x, sr, kind)
            det = 69 + 12 * np.log2(f0 / 440.0) if f0 else None
            rows.append(dict(path=os.path.relpath(p, ROOT), parsed=parse_note(fn), det=det, vel=parse_vel(fn),
                             rr=parse_rr(fn), onset=onset_index(x, sr), sr=sr, len=len(x),
                             peak=float(np.abs(x).max())))
        # octave convention: majority vote of (det - parsed) rounded to 12
        offs = Counter(int(round((r["det"] - r["parsed"]) / 12.0)) * 12 for r in rows
                       if r["det"] is not None and r["parsed"] is not None)
        off = offs.most_common(1)[0][0] if offs else 0
        out = []
        for r in rows:
            ov = override_pitch(os.path.basename(r["path"]))
            if ov is not None:
                midi = int(round(ov))
                cents = 100.0 * (ov - midi)
                if r["det"] is not None and abs(r["det"] - ov) < 0.45:
                    cents = 100.0 * (r["det"] - midi)
            elif r["parsed"] is not None:
                midi = r["parsed"] + off
                cents = 0.0
                if r["det"] is not None and abs(r["det"] - midi) < 0.45:
                    cents = 100.0 * (r["det"] - midi)
            elif r["det"] is not None:
                midi = int(round(r["det"]))
                cents = 100.0 * (r["det"] - midi)
            else:
                continue
            r.update(midi=int(midi), cents=float(cents))
            out.append(r)
        # normalise velocity layers to 1..L
        layers = sorted({r["vel"] for r in out})
        for r in out:
            r["layer"] = layers.index(r["vel"]) + 1
            r["nlayers"] = len(layers)
        idx[key] = dict(kind=kind, rows=out)
        mids = sorted({r["midi"] for r in out})
        bad = sum(1 for r in out if r["det"] is not None and abs(r["det"] - r["midi"]) >= 0.45)
        print(f"{key:14s} n={len(out):3d} oct_off={off:+d} layers={len(layers)} range={mids[0]}-{mids[-1]} "
              f"unreliable_f0={bad}", flush=True)
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    with open(INDEX, "w") as f:
        json.dump(idx, f)


if __name__ == "__main__":
    sys.exit(build())

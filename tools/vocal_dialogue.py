"""Builds audio/stems/dialogue.wav: every timeline line, placed sample-exactly at its locked start
(same int(start*SR) placement as the dry reference), processed per cast fx + scene acoustics, gently panned.

Per-line processed renders (stereo 48 kHz, sample 0 = line start, plus `pre` seconds of pre-roll) are
written to audio/vocal/dialogue_lines/<ID>.wav + index.json for verification.

usage: python tools/vocal_dialogue.py [ID ...]      (IDs = only rebuild these lines into the cache)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import (Compressor, Distortion, HighpassFilter, HighShelfFilter, LowpassFilter,
                        LowShelfFilter, Pedalboard, PeakFilter)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocal_lib import (NS, ROOT, SR, TL, VOCAL, add_at, bandpass, convolve, db, echo, ir, lufs, midi_hz,
                       norm_lufs, pan, psola, reverb, rng_for, up48, write_stem)

OUT = VOCAL / "dialogue_lines"

PAN = {"J01": 0.08, "J02": 0.0, "L01": -0.3, "C01": 0.3, "L02": -0.35, "C02": 0.1,
       "K02": -0.35, "K03": 0.25, "K07": -0.55,
       "X01": -0.17, "X02": 0.36, "X03": -0.44, "X04": 0.48, "X05": 0.0, "X06": -0.21, "X07": 0.22,
       "L03": -0.35, "C03": 0.2, "C04": -0.3, "L04": -0.3, "C05": 0.3, "L05": -0.3, "C06": 0.3}
PAN_AMT = 0.6

# target loudness of the processed *dry* voice (LUFS, per line) before rooms are added
TARGET = {"none": -20.0, "radio": -20.5, "tv": -21.0, "robot": -19.5, "shout": -18.5, "void": -18.5}

# scene rooms (procedural IRs) — name: (make_ir kwargs, wet dB)
ROOMS = {
    "hall":    (dict(rt60=4.2, predelay=0.05, lo_mult=1.1, hi_mult=0.55, er_n=6, er_gain=0.25, width=1.0,
                     build=0.06, seed=11), -12.5),        # s01 / s07b ethereal hall (narrator)
    "plate":   (dict(rt60=1.0, predelay=0.012, lo_mult=0.8, hi_mult=0.7, er_n=0, width=0.9, build=0.004,
                     seed=12), -24.0),                   # drier narration
    "dawn":    (dict(rt60=1.6, predelay=0.03, lo_mult=0.9, hi_mult=0.5, er_n=4, er_gain=0.2, width=1.0,
                     seed=13), -21.0),                   # s07a open plain at dawn
    "stone":   (dict(rt60=3.2, predelay=0.045, lo_mult=1.2, hi_mult=0.42, er_n=16, er_span=0.14,
                     er_gain=0.5, width=1.0, seed=14), -15.0),  # s04 Flagartha: 110 m basalt pentagon, long & dark
    "cavern":  (dict(rt60=6.0, predelay=0.075, lo_mult=0.9, hi_mult=0.6, er_n=10, er_span=0.14,
                     er_gain=0.35, width=1.0, build=0.09, seed=15), -10.0),   # Hypermind (huge overhead)
    "storm":   (dict(rt60=0.9, predelay=0.02, lo_mult=0.7, hi_mult=0.5, er_n=3, er_gain=0.25, width=1.0,
                     seed=16), -24.0),                   # s06 rubble ledge outdoors
    "meadow":  (dict(rt60=0.7, predelay=0.006, lo_mult=0.7, hi_mult=0.5, er_n=3, er_span=0.02,
                     er_gain=0.35, width=0.9, seed=17), -25.0),   # s08 open air
    "islands": (dict(rt60=0.6, predelay=0.01, lo_mult=0.6, hi_mult=0.5, er_n=2, width=0.9, seed=18), -24.0),
    "temple":  (dict(rt60=3.4, predelay=0.04, lo_mult=1.2, hi_mult=0.5, er_n=14, er_span=0.12,
                     er_gain=0.45, width=1.0, seed=19), -13.0),   # s05b vast temple
    "void":    (dict(rt60=9.0, predelay=0.09, lo_mult=0.7, hi_mult=0.7, er_n=0, width=1.0, build=0.15,
                     seed=20), -11.0),                   # s05b starry void
    "press":   (dict(rt60=0.55, predelay=0.01, er_n=6, er_span=0.03, er_gain=0.3, width=0.0, seed=21), -20.0),
    "flooded": (dict(rt60=0.85, predelay=0.006, lo_mult=0.8, hi_mult=0.55, er_n=14, er_span=0.035,
                     er_gain=0.7, width=0.9, seed=22), -9.0),   # flooded living room around the CRT
}

NARR_ROOM = {"s01": "hall", "s02": "plate", "s07a": "dawn", "s07b": "hall", "s08": "plate"}
DUO_ROOM = {"s04": "stone", "s06": "storm", "s07a": "dawn", "s08": "meadow"}


ROOM_HP = {"cavern": 140.0, "void": 160.0}   # reverbs never carry rumble


def room(name):
    kw, wet = ROOMS[name]
    k = ("hp", name)
    if k not in _HP:
        _HP[k] = bandpass(ir(name, **kw), ROOM_HP.get(name, 90.0), None, order=2)
    return _HP[k], wet


_HP = {}


# ------------------------------------------------------------------------------------------------ chains
def pb(x, *fx):
    return Pedalboard(list(fx))(np.asarray(x, np.float32)[None, :], SR)[0]


def excite(x, semis=2.0, spread=1.3, sr=SR):
    """Shouted/screamed prosody by PSOLA: raise F0 and expand the contour around its median (timing kept)."""
    def fn(ts, hz):
        med = np.median(hz)
        return med * 2 ** (semis / 12) * (hz / med) ** spread
    return psola(x, sr, fn)


def narrator(x, ln):
    y = pb(x, HighpassFilter(65), LowShelfFilter(190, 2.5, 0.7), PeakFilter(3300, -1.5, 1.0),
           HighShelfFilter(7500, 2.0, 0.7), Compressor(-24, 2.5, 8, 140))
    y = norm_lufs(y, TARGET["none"])
    name = NARR_ROOM[ln["scene"]]
    h, wet = room(name)
    st = pan(y, 0.0)
    out = reverb(st, h, wet)
    if name == "hall":
        # shimmer: an octave-up (formant-preserving) ghost of the voice, only inside the reverb
        sh = psola(y, SR, lambda ts, hz: hz * 2.0)
        sh = bandpass(sh, 900, 9000, order=2)
        w = convolve(sh, h) * db(wet - 12)
        out[:len(w)] += w[:len(out)]
    return out, 0.0


def duo(x, ln):
    sc, sp, lid = ln["scene"], ln["speaker"], ln["id"]
    fx = [HighpassFilter(75 if sp == "LUTIE" else 60)]
    if sp == "CROCUS":
        fx += [LowShelfFilter(160, 1.5, 0.7), PeakFilter(3000, 1.0, 1.0)]
    else:
        fx += [PeakFilter(2800, 1.5, 1.0)]
    fx += [HighShelfFilter(8000, 1.5, 0.7), Compressor(-24, 2.5, 6, 120)]
    y = pb(x, *fx)
    if lid == "L03":   # shouting over the gale
        y = excite(y, 1.5, 1.2)
        y = pb(y, PeakFilter(3000, 3.0, 0.9), Compressor(-22, 4.0, 2, 80))
        y = 0.8 * y + 0.2 * np.tanh(3 * y / (np.abs(y).max() + 1e-9)) * np.abs(y).max()
    y = norm_lufs(y, TARGET["shout"] if lid == "L03" else TARGET["none"])
    if sc == "s06":    # wind buffeting: slow random gain + a far slap off the tower
        r = rng_for("buffet", lid)
        n = len(y)
        k = np.cumsum(r.standard_normal(n // 480 + 2))
        k = bandpass(k - k.mean(), None, 6.0, sr=100, order=2)
        g = db(1.4 * k / (np.abs(k).max() + 1e-9))
        y = y * np.interp(np.arange(n), np.arange(len(g)) * 480, g).astype(np.float32)
    x_pan = PAN.get(lid, 0.0) * PAN_AMT
    if lid == "L04":  # running in from the left
        x_pan = np.linspace(-0.35, -0.25, len(y)) * PAN_AMT
    st = pan(y, x_pan)
    if sc == "s06":
        e = echo(st, 0.17, 0.22, n=2, lp=2200)
        e[:len(st)] -= st
        st2 = np.zeros_like(e)
        st2[:len(st)] = st
        st = st2 + e[:, ::-1] * 0.8
    h, wet = room(DUO_ROOM[sc])
    return reverb(st, h, wet), 0.0


def radio(x, ln):
    lid = ln["id"]
    r = rng_for("radio", lid)
    hot = lid == "K03"
    y = pb(x, HighpassFilter(90), Compressor(-26, 5.0, 1, 60))
    y = norm_lufs(y, -18)
    drive = 16 if hot else 11
    y = pb(y, HighpassFilter(380), HighpassFilter(380), LowpassFilter(3100), LowpassFilter(3100),
           PeakFilter(1700, 5.0, 1.1), Distortion(drive), Compressor(-20, 6.0, 1, 50),
           PeakFilter(900, 2.0, 1.5), LowpassFilter(3400))
    # sample-rate / bit degradation (8 kHz-ish codec character, blended)
    q = np.round(y * 96) / 96
    y = (0.75 * y + 0.25 * q).astype(np.float32)
    y = bandpass(y, 330, 3100, order=6)      # a real handset band: steep skirts after the grit
    y = norm_lufs(y, TARGET["radio"])
    pre, post = 0.16, 0.26
    n = int((pre + post) * SR) + len(y)
    out = np.zeros(n, np.float32)
    i0 = int(pre * SR)
    out[i0:i0 + len(y)] += y
    lvl = np.sqrt(np.mean(y ** 2)) + 1e-9
    # carrier hiss while keyed
    hiss = bandpass(r.standard_normal(n).astype(np.float32), 450, 3200, order=2) * lvl * db(-24)
    env = np.zeros(n, np.float32)
    k0, k1 = int(0.005 * SR), i0 + len(y) + int(0.07 * SR)
    env[k0:k1] = 1.0
    env = np.convolve(env, np.ones(96) / 96, mode="same")
    out += hiss * env
    # key-up click (PTT) + short squelch opening
    click = np.zeros(int(0.05 * SR), np.float32)
    click[:24] = np.hanning(48)[:24] * 3 * lvl
    click += bandpass(r.standard_normal(len(click)).astype(np.float32), 600, 3000, order=2) * \
        np.exp(-np.arange(len(click)) / (0.006 * SR)) * lvl * 1.6
    out[k0:k0 + len(click)] += click
    # breath (inhale) right after key-up
    bn = int(0.11 * SR)
    b = bandpass(r.standard_normal(bn).astype(np.float32), 700, 2600, order=2)
    b *= np.sin(np.linspace(0, np.pi, bn)) ** 2 * lvl * db(-11)
    bi = i0 - bn - int(0.012 * SR)
    out[bi:bi + bn] += b
    # key-down: squelch tail burst + release click
    sq_n = int(0.19 * SR)
    sq = bandpass(r.standard_normal(sq_n).astype(np.float32), 350, 3300, order=2)
    sq *= np.exp(-np.arange(sq_n) / (0.07 * SR)) * lvl * db(-3)
    sq[:48] *= np.linspace(0, 1, 48)
    out[k1:k1 + sq_n] += sq[:max(0, min(sq_n, n - k1))]
    kc = np.zeros(int(0.02 * SR), np.float32)
    kc[:16] = -np.hanning(32)[:16] * 2.2 * lvl
    out[k1:k1 + len(kc)] += kc[:max(0, min(len(kc), n - k1))]
    st = pan(out, PAN.get(lid, 0.0) * PAN_AMT)
    return st, pre


def tv(x, ln):
    lid = ln["id"]
    y = pb(x, HighpassFilter(80), Compressor(-26, 4.0, 3, 90))
    y = norm_lufs(y, -18)
    hp, wetp = room("press")
    y = reverb(pan(y, 0), hp, wetp).mean(1)             # the podium room, inside the broadcast
    # the CRT's little boxy speaker
    y = pb(y, HighpassFilter(230), HighpassFilter(230), LowpassFilter(5200), LowpassFilter(6500),
           PeakFilter(430, 4.0, 3.0), PeakFilter(1150, 5.0, 1.2), PeakFilter(2700, 3.0, 2.0),
           Distortion(5), Compressor(-22, 3.0, 2, 80))
    y = bandpass(y, 200, 6000, order=4)
    y = norm_lufs(y, TARGET["tv"])
    # water under the set: a slowly wobbling short reflection (watery comb)
    n = len(y)
    t = np.arange(n) / SR
    dly = (0.0032 + 0.0012 * np.sin(2 * np.pi * 0.23 * t) + 0.0006 * np.sin(2 * np.pi * 0.61 * t + 1)) * SR
    src = np.clip(np.arange(n) - dly, 0, n - 1)
    i = np.floor(src).astype(int)
    f = (src - i).astype(np.float32)
    wob = y[i] * (1 - f) + y[np.minimum(i + 1, n - 1)] * f
    y = (y + 0.35 * bandpass(wob, 300, 4000, order=1)).astype(np.float32)
    h, wet = room("flooded")
    return reverb(pan(y, PAN.get(lid, 0.0) * PAN_AMT), h, wet), 0.0


def robot(x, ln):
    lid = ln["id"]
    D3, D2 = float(midi_hz(50)), float(midi_hz(38))
    y = pb(x, HighpassFilter(70), Compressor(-26, 3.0, 4, 100))
    mono = psola(y, SR, lambda ts, hz: np.full_like(hz, D3), floor=60, ceil=500)     # dead-flat monotone
    sub = psola(y, SR, lambda ts, hz: np.full_like(hz, D2), floor=60, ceil=500)      # octave-down double
    n = len(mono)
    t = np.arange(n) / SR
    ring = mono * np.sin(2 * np.pi * float(midi_hz(74)) * t).astype(np.float32)      # metallic sidebands
    v = mono + db(-6.5) * sub + db(-19) * ring
    # feedback comb tuned to D3: the voice resonates inside the machine
    d = int(round(SR / D3))
    c = v.copy()
    for k in range(d, n):
        c[k] += 0.42 * c[k - d]
    v = (0.7 * v + 0.3 * c / 1.7).astype(np.float32)
    v = pb(v, PeakFilter(2400, 2.5, 1.5), HighShelfFilter(6000, 3.0, 0.7), Compressor(-22, 3.0, 3, 90))
    v = norm_lufs(v, TARGET["robot"])
    if lid == "H01":   # digital buffer-stutter on the last word: "…is true-tru-tru-t"
        w = next(w for w in ln["words"] if w["w"].lower().startswith("true"))
        ws = int((w["s"] - ln["start"]) * SR)
        env = np.abs(v[:len(x)])
        we = int(np.flatnonzero(env > env.max() * 0.03)[-1])  # Kokoro's word end overshoots the audio
        g0 = ws + int(0.45 * (we - ws))          # inside the vowel of "true"
        grain = v[g0:g0 + int(0.085 * SR)].copy()
        grain *= np.minimum(1, np.minimum(np.arange(len(grain)), len(grain) - 1 - np.arange(len(grain))) / 64.0)
        tail = np.zeros(int(0.5 * SR), np.float32)
        step = int(0.085 * SR)
        for k in range(5):
            gk = grain * (0.8 ** k)
            if k >= 2:
                gk = np.round(gk * (48 >> (k - 1))) / (48 >> (k - 1))   # crumbling bit depth
            if k == 4:
                gk = gk[: len(gk) // 2]
            tail[k * step:k * step + len(gk)] += gk
        end = we - int(0.02 * SR)
        v2 = np.zeros(max(len(v), end + len(tail)), np.float32)
        v2[:end] = v[:end]
        v2[end - 96:end] *= np.linspace(1, 0, 96)
        v2[end:end + len(tail)] += tail
        v = v2
    st = pan(v, 0.0)
    h1, w1 = room("cavern")
    h2, w2 = room("stone")
    out = reverb(st, h1, w1)
    out[:len(out)] += np.pad(convolve(st, h2) * db(w2 - 2),
                             ((0, max(0, len(out) - len(st) - len(h2) + 1)), (0, 0)))[:len(out)]
    return out, 0.0


def shout(x, ln):
    lid, sc = ln["id"], ln["scene"]
    y = pb(x, HighpassFilter(110), Compressor(-24, 3.0, 3, 80))
    y = excite(y, 2.5, 1.35)
    pk = np.abs(y).max() + 1e-9
    y = 0.65 * y + 0.35 * np.tanh(4 * y / pk) * pk
    y = pb(y, PeakFilter(3100, 5.0, 0.9), PeakFilter(260, -2.5, 1.0), HighShelfFilter(7000, 2.0, 0.7),
           Compressor(-20, 4.0, 1.5, 70))
    y = norm_lufs(y, TARGET["shout"])
    st = pan(y, PAN.get(lid, 0.0) * PAN_AMT)
    e = echo(st, 0.105, 0.22, n=2, lp=3000, spread=0.4)       # short slap
    e[:len(st)] -= st
    out = np.zeros_like(e)
    out[:len(st)] = st
    out += e
    name = "islands" if lid in ("X01", "X02") else "temple"
    h, wet = room(name)
    return reverb(out, h, wet), 0.0


def void(x, ln):
    lid = ln["id"]
    y = pb(x, HighpassFilter(110), Compressor(-24, 3.0, 3, 80))
    y = excite(y, 2.0, 1.5)
    pk = np.abs(y).max() + 1e-9
    y = 0.6 * y + 0.4 * np.tanh(5 * y / pk) * pk
    y = pb(y, PeakFilter(2900, 5.0, 0.9), HighShelfFilter(7000, 2.0, 0.7), Compressor(-20, 4.0, 1.5, 70))
    y = norm_lufs(y, TARGET["void"])
    st = pan(y, PAN.get(lid, 0.0) * PAN_AMT)
    # hurled into the infinite: ping-pong repeats that darken, thin and sag in pitch as they recede
    e = echo(st, 0.36 if lid == "X06" else 0.29, 0.52, n=12, lp=3400, hp=260, spread=0.75, detune=0.006)
    e[:len(st)] -= st
    wet = np.zeros_like(e)
    wet[:len(st)] = st
    wet += e * 0.55
    h, w = room("void")
    return reverb(wet, h, w), 0.0


FX = {"none": None, "radio": radio, "tv": tv, "robot": robot, "shout": shout, "void": void}


def process(ln):
    sp = ln["speaker"]
    fx = TL["cast"][sp]["fx"]
    a24, _ = sf.read(ROOT / ln["file"], dtype="float32")
    x = up48(a24)
    if fx == "none":
        return narrator(x, ln) if sp == "NARRATOR" else duo(x, ln)
    return FX[fx](x, ln)


def main(only=()):
    OUT.mkdir(parents=True, exist_ok=True)
    idx_path = OUT / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    for ln in TL["lines"]:
        lid = ln["id"]
        if only and lid not in only and (OUT / f"{lid}.wav").exists():
            continue
        st, pre = process(ln)
        # tails: fade to silence cleanly
        st = st.astype(np.float32)
        nz = np.flatnonzero(np.abs(st).max(1) > 1e-5)
        st = st[: (nz[-1] + 1) if len(nz) else len(st)]
        sf.write(OUT / f"{lid}.wav", st, SR, subtype="FLOAT")
        idx[lid] = {"pre": pre, "start": ln["start"], "len": round(len(st) / SR, 3),
                    "fx": TL["cast"][ln["speaker"]]["fx"], "pan": PAN.get(lid, 0.0) * PAN_AMT}
        print(f"{lid} {ln['speaker']:10s} {idx[lid]['fx']:6s} len {len(st)/SR:5.2f}s pre {pre:.2f} "
              f"{lufs(st):6.1f} LUFS", flush=True)
    idx_path.write_text(json.dumps(idx, indent=1))
    mix = np.zeros((NS, 2), np.float32)
    for ln in TL["lines"]:
        lid = ln["id"]
        st, _ = sf.read(OUT / f"{lid}.wav", dtype="float32")
        i0 = int(ln["start"] * SR) - int(round(idx[lid]["pre"] * SR))   # same placement as dialogue_dry
        add_at(mix, st, i0 / SR)
    # sacred silence 138.5-140.0 (nothing in dialogue should be there anyway)
    a, b = int(138.5 * SR), int(140.0 * SR)
    mix[a:b] = 0
    write_stem("dialogue", mix)


if __name__ == "__main__":
    main(set(sys.argv[1:]))

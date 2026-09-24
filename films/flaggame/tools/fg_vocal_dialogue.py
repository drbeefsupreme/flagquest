"""THE FLAG GAME? — builds films/flaggame/audio/stems/dialogue.wav.   Owner: Vocalist.

Every line is placed on the same sample as dialogue_dry.wav (int(start*48000)).  All processing is
timing-preserving (EQ, dynamics, PSOLA pitch shaping, reverbs/doubles *after* the dry onset, pre-roll
elements like the Devil's reverse-reverb and radio key-ups are *before* it), so lip-sync and the on-screen
lettering stay locked to the word timestamps.

 PREACHER present (t01/t02/t08/t09) : fervent open-air read under a dripping camp shade (short canopy room)
 PREACHER flashback (t03-t07)       : warm vintage "church filmstrip / record" V.O. — close, tape-warm,
                                      soft top, faint crackle; CAPS words of the lettering lifted by PSOLA
                                      (+1.3 st) and energy (+2.5 dB)
 SUMMER / RAVEN                      : camp shade in the rain (S05 + P08 = off-panel, from the present, dry/close:
                                      same camp acoustics, a touch distant, over the flashback)
 FLAGMAKER                           : divine — warm, octave-down PSOLA double, radiant cathedral
 DEVIL                               : -5 st PSOLA, grit, sub double, reverse-reverb swell into each line;
                                      V02 possessed: stacked octave doubles, ring-mod, heavier drive
 FORECAST                            : AM radio on a boombox (intelligible band 250-4800 Hz)
 RANGER                              : handheld radio with key-up click / squelch tail
 DISCLAIMER                          : pharma-ad legal read, dry, tight, bright
 PHARISEES                           : "Shut up!" thickened into a mob (PSOLA-shifted doubles, +-12 ms)
 HIPPIE1/2                           : shouted in the rain

usage: source films/flaggame/env.sh && python films/flaggame/tools/fg_vocal_dialogue.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import (Compressor, Distortion, HighpassFilter, HighShelfFilter, LowpassFilter,
                        LowShelfFilter, Pedalboard, PeakFilter)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from vocal_lib import (FILM, FILM_ROOT, NS, SR, TL, VOCAL, add_at, bandpass, convolve, db, echo, ir, lufs,
                       norm_lufs, pan, psola, reverb, rng_for, up48, write_stem)

assert FILM == "flaggame", "source films/flaggame/env.sh first (VX_FILM=flaggame)"
OUT = VOCAL / "dialogue_lines"
PRESENT = {"t01", "t02", "t08", "t09"}

# on-screen x of the speaker (-1..1) from the scene owners; applied at PAN_AMT
PAN: dict[str, float] = {}
PAN_AMT = 0.6

ROOMS = {  # name: (make_ir kwargs, wet dB)
    "shade":     (dict(rt60=0.45, predelay=0.004, lo_mult=0.8, hi_mult=0.5, er_n=8, er_span=0.018,
                       er_gain=0.55, width=0.8, seed=101), -17.0),   # camp shade canopy, open sides, rain
    "filmstrip": (dict(rt60=0.7, predelay=0.01, lo_mult=0.9, hi_mult=0.45, er_n=0, width=0.7, build=0.005,
                       seed=102), -26.0),                             # a hint of plate on the old record
    "cathedral": (dict(rt60=4.8, predelay=0.06, lo_mult=1.1, hi_mult=0.55, er_n=10, er_span=0.12,
                       er_gain=0.3, width=1.0, build=0.08, seed=103), -11.0),
    "hell":      (dict(rt60=3.0, predelay=0.03, lo_mult=1.3, hi_mult=0.35, er_n=12, er_span=0.08,
                       er_gain=0.4, width=1.0, seed=104), -13.0),
    "field":     (dict(rt60=0.8, predelay=0.012, lo_mult=0.7, hi_mult=0.5, er_n=3, width=0.9, seed=105), -22.0),
    "mudroom":   (dict(rt60=0.6, predelay=0.005, er_n=10, er_span=0.03, er_gain=0.5, width=0.8, seed=106), -18.0),
}
_R = {}


def room(name):
    if name not in _R:
        kw, wet = ROOMS[name]
        _R[name] = (bandpass(ir(name, **kw), 90.0, None, order=2), wet)
    return _R[name]


def pb(x, *fx):
    return Pedalboard(list(fx))(np.asarray(x, np.float32)[None, :], SR)[0]


def caps_spans(ln):
    """(start, end) seconds re: line start of every emphasised (ALL-CAPS, >= 2 letters) display word."""
    out = []
    for w in ln["words"]:
        d = w.get("d", "")
        letters = [c for c in d if c.isalpha()]
        if len(letters) >= 2 and all(c.isupper() for c in letters):
            out.append((w["s"] - ln["start"], w["e"] - ln["start"]))
    return out


def emphasis(x, ln, semis=1.3, gain_db=2.5):
    """Lift the lettering's CAPS words: PSOLA pitch raise + energy, smooth 40 ms ramps; timing untouched."""
    spans = caps_spans(ln)
    if not spans:
        return x
    n = len(x)
    t = np.arange(n) / SR
    m = np.zeros(n, np.float32)
    for s, e in spans:
        m = np.maximum(m, np.clip(np.minimum((t - s + 0.03) / 0.04, (e + 0.03 - t) / 0.06), 0, 1))

    def fn(ts, hz):
        return hz * 2 ** (semis * np.interp(ts, t, m) / 12)
    y = psola(x, SR, fn, floor=60, ceil=450)
    return (y * db(gain_db * m)).astype(np.float32)


def excite(x, semis, spread):
    return psola(x, SR, lambda ts, hz: np.median(hz) * 2 ** (semis / 12) * (hz / np.median(hz)) ** spread)


ALTAR = {"P27", "S07", "P28", "S08", "P29", "S09"}      # the altar call: intimate, close-mic (T08)
AFTER_RAIN = {"P30"}                                     # the rain has stopped dead: dry, sunny open air


def finish(y, name, p, wet_extra=0.0, lid=None):
    if name == "shade" and lid in ALTAR:
        y = pb(y, LowShelfFilter(200, 1.5, 0.7))           # proximity warmth, almost no room
        wet_extra -= 7.0
    elif name == "shade" and lid in AFTER_RAIN:
        name, wet_extra = "field", wet_extra - 3.0
    h, wet = room(name)
    return reverb(pan(y, p * PAN_AMT), h, wet + wet_extra)


# ------------------------------------------------------------------------------------------------ chains
def preacher(x, ln):
    y = pb(x, HighpassFilter(70), LowShelfFilter(180, 2.0, 0.7), PeakFilter(3200, 2.0, 1.0),
           Compressor(-24, 3.0, 5, 110))
    y = emphasis(y, ln)
    if ln["meta"].get("letter") == "shout":
        y = excite(y, 1.5, 1.2)
        y = pb(y, PeakFilter(2800, 2.5, 1.0), Compressor(-20, 4.0, 2, 80))
    offpanel = ln["meta"].get("letter") == "offpanel"      # P08: Crow answering from the present (T03 freeze)
    if ln["scene"] in PRESENT or offpanel:
        y = norm_lufs(y, -19.5)
        out = finish(y, "shade", PAN.get(ln["id"], 0.0), wet_extra=-8.0 if offpanel else 0.0, lid=ln["id"])
        if ln["id"] == "P06":   # "Alchemy IX..." dissolves into the flashback ripple: wobbling echo tail
            e = echo(out, 0.23, 0.45, n=7, lp=3500, hp=200, spread=0.6, detune=0.012)
            e[:len(out)] -= out
            e = e * np.clip((np.arange(len(e)) / SR - 1.2) / 0.6, 0, 1)[:, None]   # only after 'Alchemy'
            e[:len(out)] += out
            out = e
        return out, 0.0
    # flashback: the warm old filmstrip record
    y = pb(y, HighpassFilter(85), LowShelfFilter(260, 1.5, 0.7), PeakFilter(1400, 1.2, 0.8),
           HighShelfFilter(6500, -3.0, 0.7), LowpassFilter(10500), Distortion(3), Compressor(-22, 2.5, 8, 150))
    y = norm_lufs(y, -20.0)
    r = rng_for("crackle", ln["id"])
    n = len(y)
    crackle = np.zeros(n, np.float32)
    k = r.random(n) < 14.0 / SR
    crackle[k] = r.standard_normal(k.sum()).astype(np.float32)
    crackle = bandpass(crackle, 1500, 7000, order=1) * (np.sqrt(np.mean(y ** 2)) * db(-18))
    hiss = bandpass(r.standard_normal(n).astype(np.float32), 2000, 9000, order=1) * np.sqrt(np.mean(y ** 2)) * db(-44)
    return finish(y + crackle + hiss, "filmstrip", 0.0), 0.0


def camp_voice(x, ln):
    sp = ln["speaker"]
    y = pb(x, HighpassFilter(90), PeakFilter(2900, 1.5, 1.0), HighShelfFilter(8000, 1.5, 0.7),
           Compressor(-24, 2.5, 5, 110))
    if sp == "SUMMER":
        y = emphasis(y, ln, semis=1.0, gain_db=2.0)
    y = norm_lufs(y, -20.0)
    if ln["meta"].get("letter") == "offpanel":   # S05: from the present over the frozen flashback: dry, close
        return finish(y, "shade", PAN.get(ln["id"], 0.0), wet_extra=-8.0), 0.0
    return finish(y, "shade", PAN.get(ln["id"], 0.0), lid=ln["id"]), 0.0


def divine(x, ln):
    y = pb(x, HighpassFilter(60), LowShelfFilter(200, 2.5, 0.7), PeakFilter(3000, 1.5, 1.0),
           HighShelfFilter(9000, 2.5, 0.7), Compressor(-24, 2.5, 8, 150))
    sub = psola(y, SR, lambda ts, hz: hz * 0.5, floor=55, ceil=400)
    sub = bandpass(sub, 60, 2500, order=2)
    y = norm_lufs(y + sub * db(-11), -19.5)
    h, wet = room("cathedral")
    st = pan(y, PAN.get(ln["id"], 0.0) * PAN_AMT)
    out = reverb(st, h, wet)
    shimmer = bandpass(psola(y, SR, lambda ts, hz: hz * 2.0, floor=55, ceil=400), 1200, 10000, order=2)
    w = convolve(shimmer, h) * db(wet - 13)                 # radiant octave-up halo only in the reverb
    out[:len(w)] += w[:len(out)]
    return out, 0.0


def demonic(x, ln):
    possessed = ln["id"] == "V02"
    y = pb(x, HighpassFilter(60), Compressor(-26, 3.0, 3, 90))
    lo = psola(y, SR, lambda ts, hz: hz * 2 ** (-5 / 12), floor=55, ceil=500)
    sub = psola(y, SR, lambda ts, hz: hz * 2 ** (-17 / 12), floor=55, ceil=500)
    v = lo + sub * db(-8 if possessed else -12)
    if possessed:
        hi = psola(y, SR, lambda ts, hz: hz * 2 ** (7 / 12), floor=55, ceil=500)
        t = np.arange(len(y)) / SR
        ring = lo * np.sin(2 * np.pi * 73.4 * t).astype(np.float32)
        v = v + hi * db(-12) + ring * db(-9)
    pk = np.abs(v).max() + 1e-9
    g = 0.45 if possessed else 0.3
    v = (1 - g) * v + g * np.tanh((5 if possessed else 3) * v / pk) * pk
    v = pb(v, PeakFilter(180, 3.0, 0.8), PeakFilter(2400, 2.5, 1.2), HighShelfFilter(7000, -2.0, 0.7),
           Compressor(-20, 4.0, 2, 80))
    v = norm_lufs(v, -18.5 if possessed else -19.5)
    h, wet = room("hell")
    st = pan(v, PAN.get(ln["id"], 0.0) * PAN_AMT)
    out = reverb(st, h, wet)
    # reverse-reverb swell sucked into the line (entirely before the dry onset)
    pre = 0.7
    seg = pan(v[: int(1.2 * SR)], 0.0)[::-1]
    rr = convolve(seg, h)[::-1]                           # x's first sample sits at index len(h)-1
    z = len(h) - 1
    rv = rr[z - int(pre * SR):z] * (np.linspace(0, 1, int(pre * SR)) ** 1.5)[:, None] * db(-2)
    full = np.zeros((int(pre * SR) + len(out), 2), np.float32)
    full[:int(pre * SR)] += rv
    full[int(pre * SR):] += out
    return full, pre


def am_radio(x, ln):
    y = pb(x, HighpassFilter(90), Compressor(-26, 5.0, 1, 60))
    y = norm_lufs(y, -18)
    y = pb(y, HighpassFilter(230), LowpassFilter(5200), PeakFilter(1900, 3.0, 1.0), PeakFilter(700, 1.5, 1.2),
           Distortion(3), Compressor(-20, 4.0, 2, 60))
    y = bandpass(y, 250, 4800, order=4)
    y = norm_lufs(y, -20.5)
    h, wet = room("mudroom")                              # a boombox in the mud under a shade
    return reverb(pan(y, PAN.get(ln["id"], 0.0) * PAN_AMT), h, wet), 0.0


def handheld(x, ln):
    r = rng_for("hh", ln["id"])
    y = pb(x, HighpassFilter(90), Compressor(-26, 5.0, 1, 60))
    y = norm_lufs(y, -18)
    y = pb(y, HighpassFilter(260), LowpassFilter(5000), PeakFilter(2200, 3.0, 1.0), Distortion(5),
           Compressor(-20, 4.0, 2, 60))
    y = bandpass(y, 280, 4400, order=4)
    y = norm_lufs(y, -20.5)
    pre = 0.14
    n = int((pre + 0.28) * SR) + len(y)
    out = np.zeros(n, np.float32)
    i0 = int(pre * SR)
    out[i0:i0 + len(y)] += y
    lvl = np.sqrt(np.mean(y ** 2)) + 1e-9
    k1 = i0 + len(y) + int(0.06 * SR)
    env = np.zeros(n, np.float32)
    env[int(0.005 * SR):k1] = 1
    out += bandpass(r.standard_normal(n).astype(np.float32), 450, 4000, order=2) * lvl * db(-30) * env
    click = np.zeros(int(0.04 * SR), np.float32)
    click[:24] = np.hanning(48)[:24] * 3 * lvl
    out[int(0.005 * SR):int(0.005 * SR) + len(click)] += click
    sq = bandpass(r.standard_normal(int(0.17 * SR)).astype(np.float32), 350, 3500, order=2)
    sq *= np.exp(-np.arange(len(sq)) / (0.06 * SR)) * lvl * db(-4)
    out[k1:k1 + len(sq)] += sq[:max(0, min(len(sq), n - k1))]
    h, wet = room("field")
    return reverb(pan(out, PAN.get(ln["id"], 0.0) * PAN_AMT), h, wet), pre


def legal(x, ln):
    y = pb(x, HighpassFilter(120), PeakFilter(3500, 2.0, 1.0), HighShelfFilter(8000, 2.0, 0.7),
           Compressor(-26, 5.0, 1, 50), Compressor(-16, 8.0, 0.5, 30))
    return pan(norm_lufs(y, -21.0), 0.0), 0.0


def mob(x, ln):
    y = pb(x, HighpassFilter(100), Compressor(-24, 3.0, 3, 80))
    y = excite(y, 1.5, 1.3)
    r = rng_for("mob", ln["id"])
    acc = pan(y, 0.0)
    for k in range(7):                                   # hooded elders in a knot around the lead
        sh = float(r.uniform(-4, 3))
        d = psola(y, SR, lambda ts, hz, sh=sh: hz * 2 ** (sh / 12), floor=60, ceil=500)
        d = d * db(float(r.uniform(-7, -3)))
        lag = int(abs(r.normal(0, 0.008)) * SR)          # only late, never early (lettering onset)
        dd = np.zeros_like(d)
        dd[lag:] = d[:len(d) - lag]
        acc = acc + pan(dd, float(r.uniform(-0.6, 0.6)))
    acc = acc.astype(np.float32)
    pk = np.abs(acc).max() + 1e-9
    acc = 0.7 * acc + 0.3 * np.tanh(3 * acc / pk) * pk
    acc = np.stack([pb(acc[:, c], PeakFilter(2800, 3.0, 1.0)) for c in range(2)], 1)
    acc = acc * db(-19.0 - lufs(acc))
    h, wet = room("field")
    return reverb(acc, h, wet), 0.0


def shout(x, ln):
    y = pb(x, HighpassFilter(110), Compressor(-24, 3.0, 3, 80))
    y = excite(y, 2.5, 1.35)
    pk = np.abs(y).max() + 1e-9
    y = 0.65 * y + 0.35 * np.tanh(4 * y / pk) * pk
    y = pb(y, PeakFilter(3100, 4.0, 0.9), Compressor(-20, 4.0, 1.5, 70))
    y = norm_lufs(y, -18.5)
    return finish(y, "field", PAN.get(ln["id"], 0.0)), 0.0


def process(ln):
    fx = TL["cast"][ln["speaker"]]["fx"]
    a24, _ = sf.read(FILM_ROOT / ln["file"], dtype="float32")
    x = up48(a24)
    return {"preacher": preacher, "none": camp_voice, "divine": divine, "demonic": demonic,
            "radio": am_radio if ln["speaker"] == "FORECAST" else handheld, "legal": legal, "mob": mob,
            "shout": shout}[fx](x, ln)


def main(only=()):
    global PAN
    pp = VOCAL / "pans.json"                              # scene owners' positions (optional)
    if pp.exists():
        PAN = {k: float(v) for k, v in json.loads(pp.read_text()).items()}
    OUT.mkdir(parents=True, exist_ok=True)
    idx_path = OUT / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    for ln in TL["lines"]:
        lid = ln["id"]
        if only and lid not in only and (OUT / f"{lid}.wav").exists():
            continue
        st, pre = process(ln)
        st = st.astype(np.float32)
        nz = np.flatnonzero(np.abs(st).max(1) > 1e-5)
        st = st[: (nz[-1] + 1) if len(nz) else len(st)]
        sf.write(OUT / f"{lid}.wav", st, SR, subtype="FLOAT")
        idx[lid] = {"pre": pre, "start": ln["start"], "len": round(len(st) / SR, 3),
                    "fx": TL["cast"][ln["speaker"]]["fx"], "pan": PAN.get(lid, 0.0) * PAN_AMT,
                    "caps": len(caps_spans(ln))}
        print(f"{lid} {ln['speaker']:10s} {idx[lid]['fx']:8s} len {len(st)/SR:5.2f}s pre {pre:.2f} "
              f"caps {idx[lid]['caps']}  {lufs(st):6.1f} LUFS", flush=True)
    idx_path.write_text(json.dumps(idx, indent=1))
    mix = np.zeros((NS, 2), np.float32)
    for ln in TL["lines"]:
        lid = ln["id"]
        st, _ = sf.read(OUT / f"{lid}.wav", dtype="float32")
        i0 = int(ln["start"] * SR) - int(round(idx[lid]["pre"] * SR))   # same sample as dialogue_dry
        tg = i0 / SR + np.arange(len(st)) / SR
        g = np.ones(len(st), np.float32)                  # tails duck 12 dB under later lines
        for o in TL["lines"]:
            if ln["end"] < o["start"] < tg[-1]:
                d = np.clip(np.minimum((tg - o["start"] + 0.25) / 0.12, (o["end"] + 0.15 - tg) / 0.3), 0, 1)
                g = np.minimum(g, (1 - d * (1 - db(-12))).astype(np.float32))
        add_at(mix, st * g[:, None], i0 / SR)
    mix[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))[:, None].astype(np.float32)
    pk = np.abs(mix).max()
    if pk > db(-1.0):                                     # headroom for the mix (uniform gain: timing-safe)
        mix *= db(-1.0) / pk
    write_stem("dialogue", mix)


if __name__ == "__main__":
    main(set(sys.argv[1:]))

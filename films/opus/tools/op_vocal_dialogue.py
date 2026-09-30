"""OPVS SCHISMATICVM — builds films/opus/audio/stems/dialogue.wav.   Owner: Vocalist-2.

Every spoken line sits on the same sample as dialogue_dry.wav (int(start*48000)); all processing is timing-preserving
(EQ, dynamics, PSOLA, reverbs after the onset; radio key-ups before it).  The two CANTOR lines are re-sung as
plainchant (op_vocal_score: recto tono on A3, cadence to D3) and placed by their vowel onsets.

 LECTOR   (N01-N11)  the abbess at the ambo: warm close voice + the basilica (RT 5 s, far-wall answers at 145/212 ms),
                     the tail side-chain ducked by her voice so it blooms only in the pauses.  Location follows the
                     programme: nave (m01/m02), the empty plastered basilica after the fall (m08), the dome (m09),
                     the apse (m10).
 CANTOR   (A01/A02)  im_nicola made to chant (Kokoro prosody injection), under the dome, big and barely ducked.
 JAGUAR   (J01/J02)  imperial broadcast: podium mic -> PA amplifier -> horn loudspeakers + the PA delay towers down the
                     gold throne hall (95 / 190 ms), reverberant octagon.
 HYPERMIND(H01/H02)  the ophanim: one utterance spoken by eleven Kokoro voices in lock-step (every voice forced onto the
                     reference voice's phoneme durations), each on a tone of an open-fifth D chord D2..D5, plus two
                     whispered voices (the wings); every voice circles the stereo field at its own speed (wheels
                     within wheels); dark sanctuary.
 COMMANDER(K01-K07)  tactical radio (film 1's chain): 220-5200 Hz, drive, PTT key-up click + inhale before, squelch
                     tail after.
 LUTIE / CROCUS      the room of their scene: m04 sanctuary/disputation, m07 Babel (L03 shouted over the roar),
                     m08 the bare basilica, m10 the apse.
 CITIZEN1/2          shouted, tiny islands of the broken mappa mundi (open air + slap).
 HERETIC1-3          shouted, each in his own little domed chapel, the basilica beyond.
 CRACKPOT1/2         screamed into the void: receding ping-pong repeats + RT 9 s void.

usage: python films/opus/tools/op_vocal_dialogue.py [ID ...]      (IDs = only re-process these lines)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import (Compressor, Distortion, HighpassFilter, HighShelfFilter, LowpassFilter, LowShelfFilter,
                        PeakFilter)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import op_vocal_score as SC  # noqa: E402
from op_vocal_lib import (FILM_ROOT, NS, SR, TL, VOCAL, VSR, add_at, align_lag, bandpass, chant, convolve, db,  # noqa: E402
                          ducked_reverb, echo, kokoro, lufs, norm_lufs, pan, pb, pb_st, psola, ref_durations,
                          reverb_add, rng_for, room, silence, stereo, sync_voice, up48, whisperize, write_stem, dump,
                          export_cantor, world_shift_to,
                          midi_hz, vocab_ps)

OUT = VOCAL / "dialogue_lines"
PAN_AMT = 0.6
PAN = {"L01": -0.3, "C01": 0.3, "L02": -0.3, "C02": 0.3, "C03": 0.25,
       "X01": -0.3, "X02": 0.35, "X03": -0.45, "X04": 0.45, "X05": 0.0, "X06": -0.25, "X07": 0.25,
       "L03": -0.25, "C04": 0.2, "C05": 0.08, "L04": -0.3, "C06": 0.3, "L05": -0.3, "C07": 0.3}
LECTOR_ROOM = {"m01": "basilica", "m02": "basilica", "m08": "bare", "m09": "dome", "m10": "apse_lector"}
DUO_ROOM = {"m04": "sanctuary", "m07": "babel", "m08": "bare", "m10": "apse"}
CANTOR_T0 = {"A01": (SC.INCIPIT_PS, SC.INCIPIT, SC.INCIPIT_T0), "A02": (SC.EXPLICIT_PS, SC.EXPLICIT, SC.EXPLICIT_T0)}


def eq_voice(x, sp):
    if sp == "LECTOR":
        fx = [HighpassFilter(70), LowShelfFilter(180, 2.0, 0.7), PeakFilter(3200, 1.5, 1.0),
              HighShelfFilter(8000, 1.5, 0.7), Compressor(-24, 2.5, 8, 140)]
    elif sp == "CROCUS":
        fx = [HighpassFilter(60), LowShelfFilter(160, 1.5, 0.7), PeakFilter(3000, 1.5, 1.0),
              HighShelfFilter(8000, 1.5, 0.7), Compressor(-24, 2.5, 6, 120)]
    else:
        fx = [HighpassFilter(80), PeakFilter(2800, 1.5, 1.0), HighShelfFilter(8000, 1.5, 0.7),
              Compressor(-24, 2.5, 6, 120)]
    return pb(x, *fx)


def excite(x, semis=2.0, spread=1.3):
    """shouted prosody by PSOLA: raise F0 and expand the contour around its median (timing kept)"""
    return psola(x, SR, lambda ts, hz: np.median(hz) * 2 ** (semis / 12) * (hz / np.median(hz)) ** spread)


def pan_of(ln):
    return PAN.get(ln["id"], 0.0) * PAN_AMT


# ------------------------------------------------------------------------------------------------ LECTOR
def lector(x, ln):
    y = norm_lufs(eq_voice(x, "LECTOR"), -20.0)
    name = LECTOR_ROOM[ln["scene"]]
    st = pan(y, 0.0)
    if name == "apse_lector":                 # N11 over the finished apse: the apse close, the nave beyond
        h1, w1 = room("apse")
        h2, w2 = room("basilica")
        out = ducked_reverb(st, h1, w1 - 2.0, duck_db=9.0)
        far = ducked_reverb(st, h2, w2 - 5.0, duck_db=11.0)
        far[:len(st)] -= st
        out = _sum(out, far)
        return out, 0.0
    h, wet = room(name)
    return ducked_reverb(st, h, wet, duck_db=11.0), 0.0


def _sum(a, b):
    n = max(len(a), len(b))
    out = np.zeros((n, 2), np.float32)
    out[:len(a)] += a
    out[:len(b)] += b
    return out


# ------------------------------------------------------------------------------------------------ CANTOR
def cantor(ln):
    ps, syl, t0 = CANTOR_T0[ln["id"]]
    a, info = chant(ps, "im_nicola", SC.sched(syl), cons_mult=0.8, cons_max=2, seed=7 if ln["id"] == "A01" else 8)
    x = up48(a)
    y = pb(x, HighpassFilter(75), LowShelfFilter(220, 1.5, 0.7), PeakFilter(2600, 1.0, 1.0),
           HighShelfFilter(9000, 1.0, 0.7), Compressor(-24, 2.0, 15, 200))
    y = norm_lufs(y, -21.0)
    st = pan(y, 0.0)
    h, wet = room("dome")
    out = ducked_reverb(st, h, wet + 1.0, duck_db=5.0, hp=150.0, lp=8000.0)
    start = t0 - info["v0"]                     # audio time 0 of the render, global
    pre = ln["start"] - start
    syl_on = [dict(syl=s[0], t=round(t0 + (info["onsets"][k] - info["v0"]), 3),
                   end=round(t0 + (info["ends"][k] - info["v0"]), 3)) for k, s in enumerate(syl)]
    return out, pre, dict(syllables=syl_on, cents=info.get("cents"), sung_end=round(start + info["sung_end"], 3))


# ------------------------------------------------------------------------------------------------ JAGUAR
def imperial(x, ln):
    y = pb(x, HighpassFilter(80), Compressor(-26, 4.0, 3, 90))
    y = norm_lufs(y, -18.0)
    horn = pb(y, HighpassFilter(260), LowpassFilter(6000), PeakFilter(700, -2.0, 1.0), PeakFilter(1400, 4.0, 1.4),
              PeakFilter(2900, 3.0, 1.6), Distortion(5), Compressor(-20, 3.0, 2, 80))
    horn = bandpass(horn, 240, 6200, order=4)
    direct = pb(y, HighpassFilter(120), PeakFilter(3000, 2.0, 1.0))
    v = norm_lufs(horn + direct * db(-9), -21.0)
    st = pan(v, 0.0)
    n = len(v)
    towers = np.zeros((n + int(0.3 * SR), 2), np.float32)       # the PA delay towers down the hall
    for dly, g, p, lo, hi in ((0.095, -10.0, -0.65, 300, 4500), (0.19, -15.0, 0.7, 350, 3500)):
        e = pan(bandpass(v, lo, hi, order=2) * db(g), p)
        i = int(dly * SR)
        towers[i:i + n] += e
    towers[:n] += st
    h, wet = room("goldhall")
    return ducked_reverb(towers, h, wet, duck_db=7.0, key_sig=st), 0.0


# ------------------------------------------------------------------------------------------------ HYPERMIND
OPHANIM = [  # (voice, midi, gain dB, wheel rate Hz, whisper)
    ("af_nicole", 57, 0.0, 0.00, False),     # the lead on A3, dead centre
    ("am_onyx", 50, -8.0, 0.11, False), ("bm_george", 50, -10.0, -0.07, False),
    ("am_adam", 45, -9.0, 0.05, False), ("em_alex", 57, -11.0, -0.13, False),
    ("af_heart", 62, -10.0, 0.17, False), ("bf_emma", 62, -11.0, -0.19, False),
    ("af_sky", 69, -13.0, 0.23, False), ("jf_alpha", 69, -14.0, -0.29, False),
    ("zf_xiaoyi", 74, -16.0, 0.31, False), ("hm_omega", 38, -11.0, -0.04, False),
    ("af_bella", 0, -15.0, 0.37, True), ("am_michael", 0, -16.0, -0.41, True),
]


def hypermind(x, ln):
    lid = ln["id"]
    K = kokoro()
    ps = vocab_ps(K.phonemes(ln["text_say"], "a"))
    durs, F0r, _ = ref_durations(ps, "af_nicole", 1.0)
    ref = up48(sync_voice(ps, "af_nicole", durs, f0_mode="ref", ref_f0=F0r))
    raw = x
    lag = align_lag(raw, ref[:len(raw) + int(0.5 * SR)], SR, maxlag=0.5)   # render vs raw file (same durations)
    n = len(raw) + int(0.6 * SR)
    t = np.arange(n) / SR
    bus = np.zeros((n, 2), np.float32)
    for i, (voice, midi, g, rate, wh) in enumerate(OPHANIM):
        r = rng_for("ophanim", lid, voice)
        if wh:
            a = sync_voice(ps, voice, durs, f0_mode="ref", ref_f0=F0r, semis=0.0)
            a = whisperize(a, VSR, formant=1.0)
        elif midi < 45:                         # the lowest wheel: a D3 render taken down an octave by PSOLA
            a = sync_voice(ps, voice, durs, f0_mode="mono", midi=50.0 + 0.05 * r.uniform(-1, 1), seed=i)
            a = psola(a, VSR, lambda ts, hz: np.full_like(hz, float(midi_hz(38))), floor=60, ceil=400)
        elif midi > 62:                         # high wheels: rendered on D4, carried up by WORLD (no sub-harmonics)
            a = sync_voice(ps, voice, durs, f0_mode="mono", midi=62.0 + 0.06 * r.uniform(-1, 1), seed=i)
            a = world_shift_to(a, [0.0, 1e3], [float(midi_hz(62))] * 2, midi - 62, 1.06)
        else:
            a = sync_voice(ps, voice, durs, f0_mode="mono", midi=midi + 0.06 * r.uniform(-1, 1), seed=i)
        y = up48(a)
        y = np.roll(np.pad(y, (0, max(0, n - len(y))))[:n], -int(round(lag * SR)))
        y = pb(y, HighpassFilter(60 if midi and midi < 45 else 90))
        ph = r.uniform(0, 2 * np.pi)
        p = np.sin(2 * np.pi * rate * t + ph) * (0.85 if rate else 0.0)
        bus += pan(y * db(g), p)
    bus = pb_st(bus, PeakFilter(2600, 2.5, 1.0), HighShelfFilter(7000, 2.0, 0.7), Compressor(-22, 2.5, 5, 120))
    bus = bus * db(-20.0 - lufs(bus))
    h1, w1 = room("sanctuary")
    h2, w2 = room("dome")
    out = ducked_reverb(bus, h1, w1, duck_db=9.0)
    far = ducked_reverb(bus, h2, w2 - 4.0, duck_db=11.0)
    far[:len(bus)] -= bus
    return _sum(out, far), 0.0


# ------------------------------------------------------------------------------------------------ COMMANDER
def radio(x, ln):
    lid = ln["id"]
    r = rng_for("radio", lid)
    hot = lid == "K03"
    y = pb(x, HighpassFilter(90), Compressor(-26, 5.0, 1, 60))
    y = norm_lufs(y, -18)
    drive = 7 if hot else 4
    y = pb(y, HighpassFilter(200), LowpassFilter(5600), PeakFilter(1800, 2.5, 1.0), PeakFilter(2600, 3.0, 1.2),
           Distortion(drive), Compressor(-20, 4.0, 2, 60))
    q = np.round(y * 256) / 256
    y = (0.92 * y + 0.08 * q).astype(np.float32)
    y = bandpass(y, 220, 5200, order=4)
    y = norm_lufs(y, -20.5)
    pre, post = 0.16, 0.26
    n = int((pre + post) * SR) + len(y)
    out = np.zeros(n, np.float32)
    i0 = int(pre * SR)
    out[i0:i0 + len(y)] += y
    lvl = np.sqrt(np.mean(y ** 2)) + 1e-9
    hiss = bandpass(r.standard_normal(n).astype(np.float32), 450, 4500, order=2) * lvl * db(-30)
    env = np.zeros(n, np.float32)
    k0, k1 = int(0.005 * SR), i0 + len(y) + int(0.07 * SR)
    env[k0:k1] = 1.0
    env = np.convolve(env, np.ones(96) / 96, mode="same")
    out += hiss * env
    click = np.zeros(int(0.05 * SR), np.float32)
    click[:24] = np.hanning(48)[:24] * 3 * lvl
    click += bandpass(r.standard_normal(len(click)).astype(np.float32), 600, 3000, order=2) * \
        np.exp(-np.arange(len(click)) / (0.006 * SR)) * lvl * 1.6
    out[k0:k0 + len(click)] += click
    bn = int(0.11 * SR)
    b = bandpass(r.standard_normal(bn).astype(np.float32), 700, 2600, order=2)
    b *= np.sin(np.linspace(0, np.pi, bn)) ** 2 * lvl * db(-11)
    bi = i0 - bn - int(0.012 * SR)
    out[bi:bi + bn] += b
    sq_n = int(0.19 * SR)
    sq = bandpass(r.standard_normal(sq_n).astype(np.float32), 350, 3300, order=2)
    sq *= np.exp(-np.arange(sq_n) / (0.07 * SR)) * lvl * db(-3)
    sq[:48] *= np.linspace(0, 1, 48)
    out[k1:k1 + sq_n] += sq[:max(0, min(sq_n, n - k1))]
    kc = np.zeros(int(0.02 * SR), np.float32)
    kc[:16] = -np.hanning(32)[:16] * 2.2 * lvl
    out[k1:k1 + len(kc)] += kc[:max(0, min(len(kc), n - k1))]
    return pan(out, 0.0), pre


# ------------------------------------------------------------------------------------------------ LUTIE / CROCUS
def duo(x, ln):
    sc, sp, lid = ln["scene"], ln["speaker"], ln["id"]
    y = eq_voice(x, sp)
    if lid == "L03":                         # shouting over Babel's roar
        y = excite(y, 1.5, 1.2)
        y = pb(y, PeakFilter(3000, 3.0, 0.9), Compressor(-22, 4.0, 2, 80))
        pk = np.abs(y).max() + 1e-9
        y = 0.8 * y + 0.2 * np.tanh(3 * y / pk) * pk
        y = norm_lufs(y, -18.5)
    else:
        y = norm_lufs(y, -20.0)
    st = pan(y, pan_of(ln))
    name = DUO_ROOM[sc]
    h, wet = room(name)
    if name == "bare":                       # C05: the only voice left in the world, close, the empty church answers
        return ducked_reverb(st, h, wet, duck_db=11.0), 0.0
    return ducked_reverb(st, h, wet, duck_db=9.0), 0.0


# ------------------------------------------------------------------------------------------------ shouts / void
def shout(x, ln):
    lid = ln["id"]
    y = pb(x, HighpassFilter(110), Compressor(-24, 3.0, 3, 80))
    y = excite(y, 2.5, 1.35)
    pk = np.abs(y).max() + 1e-9
    y = 0.65 * y + 0.35 * np.tanh(4 * y / pk) * pk
    y = pb(y, PeakFilter(3100, 5.0, 0.9), PeakFilter(260, -2.5, 1.0), HighShelfFilter(7000, 2.0, 0.7),
           Compressor(-20, 4.0, 1.5, 70))
    y = norm_lufs(y, -18.5)
    st = pan(y, pan_of(ln))
    if lid in ("X01", "X02"):                # tiny islands: open air, a slap off the next island
        e = echo(st, 0.105, 0.22, n=2, lp=3000, spread=0.4)
        e[:len(st)] -= st
        out = _sum(st, e)
        h, wet = room("islands")
        return ducked_reverb(out, h, wet, duck_db=3.0, key_sig=st), 0.0
    h1, w1 = room("chapel")                   # a church of one
    h2, w2 = room("basilica")                 # ... inside the great church
    out = ducked_reverb(st, h1, w1, duck_db=3.0)
    far = ducked_reverb(st, h2, w2 - 8.0, duck_db=8.0)
    far[:len(st)] -= st
    return _sum(out, far), 0.0


def void(x, ln):
    lid = ln["id"]
    y = pb(x, HighpassFilter(110), Compressor(-24, 3.0, 3, 80))
    y = excite(y, 2.0, 1.5)
    pk = np.abs(y).max() + 1e-9
    y = 0.6 * y + 0.4 * np.tanh(5 * y / pk) * pk
    y = pb(y, PeakFilter(2900, 5.0, 0.9), HighShelfFilter(7000, 2.0, 0.7), Compressor(-20, 4.0, 1.5, 70))
    y = norm_lufs(y, -18.5)
    st = pan(y, pan_of(ln))
    e = echo(st, 0.36 if lid == "X06" else 0.29, 0.52, n=12, lp=3400, hp=260, spread=0.75, detune=0.006)
    e[:len(st)] -= st
    wet = np.zeros_like(e)
    wet[:len(st)] = st
    wet += e * 0.55
    h, w = room("void")
    return ducked_reverb(wet, h, w, duck_db=4.0, key_sig=st), 0.0


FX = {"basilica": lector, "tv": imperial, "robot": hypermind, "radio": radio, "none": duo, "shout": shout,
      "void": void}


# takes whose pronunciation hides the word (Whisper hears the dry X03 as "Oi Natsuma!"): the stress mark sits
# before the n.  Re-spoken by the same voice in lock-step (the dry take's own phoneme durations, re-ordered with
# the symbols), so every syllable stays where the lip-sync and the tituli expect it.
RESPEAK = {"X03": ("əˈnæθəmə", "ənˈæθəmə"), "X04": ("əˈnæθəmə", "ənˈæθəmə")}


def respeak(ln, raw48):
    old, new = RESPEAK[ln["id"]]
    cast = TL["cast"][ln["speaker"]]
    K = kokoro()
    ps = vocab_ps(K.phonemes(ln["text_say"], cast["lang"]))
    assert old in ps, (ps, old)
    _, dn, _, _ = K.forward(ps, cast["voice"], cast["speed"])
    j0 = ps.index(old)
    used, sub = set(), []
    for c in new:                                   # durations follow their symbols through the re-ordering
        k = next(i for i, o in enumerate(old) if o == c and i not in used)
        used.add(k)
        sub.append(int(dn[j0 + k + 1]))
    ps2 = ps[:j0] + new + ps[j0 + len(old):]
    d2 = dn.copy()
    d2[j0 + 1:j0 + 1 + len(new)] = sub
    a, _, _, _ = K.forward(ps2, cast["voice"], cast["speed"], dur_fn=lambda _d: d2.copy())
    y = up48(a)
    lag = align_lag(raw48, y[:len(raw48) + int(0.5 * SR)], SR, maxlag=0.5)
    y = np.roll(y, -int(round(lag * SR)))[:len(raw48) + int(0.05 * SR)]
    return y


def load_raw(ln):
    a24, sr = sf.read(FILM_ROOT / ln["file"], dtype="float32")
    assert sr == VSR
    x = up48(a24)
    return respeak(ln, x) if ln["id"] in RESPEAK else x


def process(ln):
    fx = TL["cast"][ln["speaker"]]["fx"]
    if fx == "cantor":
        return cantor(ln)
    st, pre = FX[fx](load_raw(ln), ln)
    return st, pre, {}


def main(only=()):
    OUT.mkdir(parents=True, exist_ok=True)
    idx_path = OUT / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    spec = json.loads((FILM_ROOT / "script" / "lines.json").read_text())
    say = {l["id"]: l.get("say", l["text"]) for l in spec["lines"]}
    for ln in TL["lines"]:
        lid = ln["id"]
        ln = dict(ln, text_say=say.get(lid, ln["text"]))
        if only and lid not in only and (OUT / f"{lid}.wav").exists() and lid in idx:
            continue
        st, pre, extra = process(ln)
        st = np.asarray(st, np.float32)
        nz = np.flatnonzero(np.abs(st).max(1) > 1e-5)
        st = st[: (nz[-1] + 1) if len(nz) else len(st)]
        st[-int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))[:, None].astype(np.float32)
        sf.write(OUT / f"{lid}.wav", st, SR, subtype="FLOAT")
        idx[lid] = dict(pre=round(float(pre), 6), start=ln["start"], len=round(len(st) / SR, 3),
                        fx=TL["cast"][ln["speaker"]]["fx"], pan=round(pan_of(ln), 3), **extra)
        print(f"{lid} {ln['speaker']:10s} {idx[lid]['fx']:8s} len {len(st)/SR:5.2f}s pre {pre:5.3f} "
              f"{lufs(st):6.1f} LUFS", flush=True)
    idx_path.write_text(json.dumps(idx, indent=1))
    mix = np.zeros((NS, 2), np.float32)
    lines = TL["lines"]
    for ln in lines:
        lid = ln["id"]
        st, _ = sf.read(OUT / f"{lid}.wav", dtype="float32")
        i0 = int(ln["start"] * SR) - int(round(idx[lid]["pre"] * SR))       # same sample as dialogue_dry
        tg = i0 / SR + np.arange(len(st)) / SR
        g = np.ones(len(st), np.float32)
        for o in lines:                         # a line's tail ducks under every later line it overlaps
            if ln["end"] < o["start"] < tg[-1]:
                same = o["speaker"] == ln["speaker"] == "LECTOR"
                depth = db(-7.0) if same else db(-12.0)
                d = np.clip(np.minimum((tg - o["start"] + 0.25) / 0.12, (o["end"] + 0.2 - tg) / 0.35), 0, 1)
                g = np.minimum(g, (1 - d * (1 - depth)).astype(np.float32))
        add_at(mix, st * g[:, None], i0 / SR)
    silence(mix)
    mix[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))[:, None].astype(np.float32)
    pk = np.abs(mix).max()
    if pk > db(-1.0):
        mix *= np.float32(db(-1.0) / pk)
    write_stem("dialogue", mix)
    export_cantor("incipit", idx["A01"].get("syllables"))
    export_cantor("explicit", idx["A02"].get("syllables"))


if __name__ == "__main__":
    main(set(sys.argv[1:]))

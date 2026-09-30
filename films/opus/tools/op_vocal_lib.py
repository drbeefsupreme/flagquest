"""OPVS SCHISMATICVM — shared toolkit for the vocal stems (dialogue / crowd / choir).   Owner: Vocalist-2.

Built on film 1's tools/vocal_lib.py (imported, not copied: Kokoro with prosody injection, WORLD, PSOLA, procedural
IRs, stem I/O; VX_FILM=opus points it at films/opus/) plus what this film needs:

* hall_ir()        stone-church impulse responses: octave-band decay (lows long, highs short), image-source-like early
                   reflections, a few discrete late reflections off far walls, slow diffuse build-up.
* ducked_reverb()  the ambo trick: a huge tail side-chain ducked by the dry voice, blooming only in the pauses.
* chant()          Kokoro made to CHANT: a score of vowel onsets (phrase-relative seconds) and per-vowel note lists
                   (melismas); straight tone, small scoops, portamento between notes, voiced consonants carry the
                   pitch glide; notes above the voice's comfortable register are reached by a WORLD shift of the
                   whole phrase (formants kept).  Returns audio + measured vowel onsets + per-note pitch check.
* sync_voices()    one utterance spoken by many Kokoro voices in lock-step: every voice gets the reference voice's
                   phoneme durations, and its own F0 contour (monotone chord tone, or the reference contour moved
                   to the voice's register).

usage: imported by films/opus/tools/op_vocal_{dialogue,crowd,choir,check}.py  (VX_FILM=opus)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Pedalboard

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import vocal_lib as VL  # noqa: E402
from vocal_lib import (ALL_VOICES, CACHE, FILM, FILM_ROOT, NS, SR, TL, VOCAL, VOICES, VSR, add_at, bandpass,  # noqa: E402,F401
                       convolve, db, echo, fade, is_male, key, kokoro, lufs, midi_hz, norm_lufs, pan, peak_db,
                       pitch_track, psola, resample, rng_for, smoothstep, stereo, trim, up48, warp_sp, whisperize,
                       world_analyze, world_stretch, world_synth, write_stem)

assert FILM == "opus", "export VX_FILM=opus first"
FR = 600 / VSR                    # Kokoro alignment frame: 25 ms (F0 / energy frames are half of it)
SILENCE = (145.0, 146.5)          # sacred silence: every vocal stem digital zero here


def line(lid):
    return VL.line(lid)


def cue(cid):
    return VL.cue(cid)


def pb(x, *fx, sr=SR):
    """mono pedalboard chain"""
    return Pedalboard(list(fx))(np.asarray(x, np.float32)[None, :], sr)[0]


def pb_st(x, *fx, sr=SR):
    """stereo pedalboard chain (n, 2)"""
    return np.asarray(Pedalboard(list(fx))(np.ascontiguousarray(np.asarray(x, np.float32).T), sr).T, np.float32)


def cached_wav(kind, parts, fn, sr=VSR):
    """cache a mono/stereo render under audio/vocal/cache/<kind>/<hash>.wav"""
    path = CACHE / kind / f"{key(*parts)}.wav"
    if path.exists():
        return sf.read(path, dtype="float32")[0]
    a = np.asarray(fn(), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, a, sr, subtype="FLOAT")
    return a


def cached_json(kind, parts, fn):
    """cache (audio24k, info-dict) pairs"""
    path = CACHE / kind / f"{key(*parts)}.wav"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        return sf.read(path, dtype="float32")[0], json.loads(meta.read_text())
    a, info = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(a, np.float32), VSR, subtype="FLOAT")
    meta.write_text(json.dumps(info))
    return np.asarray(a, np.float32), info


# ------------------------------------------------------------------------------------------------ rooms
OCT = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
# stone + plaster + glass: RT relative to the 1 kHz value per octave band (air absorption rolls the top off)
STONE_RT = [1.30, 1.25, 1.15, 1.05, 1.00, 0.88, 0.68, 0.45, 0.28]


def hall_ir(rt60=5.0, predelay=0.03, er=((0.011, 0.55), (0.019, 0.45), (0.027, 0.38), (0.041, 0.32),
            (0.058, 0.28), (0.077, 0.22)), late=((0.145, 0.10), (0.212, 0.08)), build=0.09, width=1.0,
            rt_shape=STONE_RT, tail_gain=0.33, length=None, seed=0, sr=SR):
    """Stereo stone-church impulse response.  er / late: (delay s, gain) reflections (jittered per channel,
    lightly low-passed so they read as walls, not clicks); tail: octave-band noise, each band decaying with its own
    RT, decorrelated L/R, slow diffuse build-up (`build` s)."""
    rng = np.random.default_rng(seed)
    n = int((length or rt60 * 1.35) * sr)
    t = np.arange(n) / sr
    tail = np.zeros((n, 2))
    edges = [(None, 88)] + [(OCT[i] * 2 ** -0.5, OCT[i] * 2 ** 0.5) for i in range(1, len(OCT) - 1)] + [(11300, 20000)]
    for ch in range(2):
        nz = rng.standard_normal(n)
        for (lo, hi), m in zip(edges, rt_shape):
            b = bandpass(nz, lo, min(hi, sr / 2 * 0.95) if hi else None, sr, order=2)
            tail[:, ch] += b * np.exp(-6.91 * t / max(rt60 * m, 1e-3))
    tail *= (1 - np.exp(-t / max(build, 1e-4)))[:, None] ** 2
    mid = tail.mean(1, keepdims=True)
    tail = mid + (tail - mid) * width
    tail /= np.sqrt((tail ** 2).sum() / 2) + 1e-9
    erb = np.zeros((n, 2))
    for d, g in list(er) + list(late):
        for ch in range(2):
            dd = d * rng.uniform(0.93, 1.07) + (0.0015 * ch * rng.uniform(-1, 1))
            i = int(dd * sr)
            if i < n:
                erb[i, ch] += g * rng.choice([-1, 1]) * rng.uniform(0.8, 1.0)
    erb = bandpass(erb, 120, 7000, sr, order=1)          # walls are not perfect mirrors
    ir = tail * tail_gain + erb
    ir = np.concatenate([np.zeros((int(predelay * sr), 2)), ir])
    return ir.astype(np.float32)


_ROOMS = {}
# name: (hall_ir kwargs, default wet dB)
ROOMS = {
    # the Lector at the ambo: the basilica nave, ~5 s, far walls answering at 145/212 ms
    "basilica": (dict(rt60=5.0, predelay=0.034, build=0.10, seed=301), 0.0),
    # the dome of Flagistan: longer, more diffuse, a gallery flutter
    "dome":     (dict(rt60=6.2, predelay=0.045, build=0.14, late=((0.118, 0.10), (0.236, 0.08), (0.354, 0.05)),
                      seed=302), 0.0),
    # the apse at dawn: half-dome, warm, shorter
    "apse":     (dict(rt60=3.0, predelay=0.022, build=0.06, seed=303, late=((0.095, 0.08),)), -6.0),
    # Flagartha sanctuary / disputation (m04): dark stone, dense early reflections
    "sanctuary": (dict(rt60=2.6, predelay=0.024, build=0.05, seed=304,
                       er=((0.009, 0.5), (0.016, 0.45), (0.023, 0.4), (0.031, 0.35), (0.044, 0.3), (0.052, 0.25),
                           (0.066, 0.2)), late=((0.12, 0.08),)), -6.0),
    # the empty basilica after the fall (m08): bare plaster, biggest and emptiest
    "bare":     (dict(rt60=5.6, predelay=0.04, build=0.12, seed=305, rt_shape=[1.3, 1.25, 1.15, 1.05, 1.0, 0.92,
                                                                         0.75, 0.52, 0.32]), -2.0),
    # a church of one: tiny domed chapel
    "chapel":   (dict(rt60=1.3, predelay=0.006, build=0.012, seed=306,
                      er=((0.004, 0.6), (0.007, 0.5), (0.011, 0.45), (0.015, 0.4), (0.019, 0.3), (0.024, 0.25)),
                      late=()), -13.0),
    # the gold throne hall (m03): octagon, reverberant, bright
    "goldhall": (dict(rt60=3.4, predelay=0.03, build=0.07, seed=307,
                      rt_shape=[1.2, 1.15, 1.1, 1.03, 1.0, 0.92, 0.78, 0.55, 0.35],
                      late=((0.13, 0.12), (0.19, 0.08))), -7.0),
    # Babel under the collapsing tower (m07)
    "babel":    (dict(rt60=3.2, predelay=0.03, build=0.08, seed=308), -10.0),
    # the tiny islands of the broken mappa mundi (open air, short)
    "islands":  (dict(rt60=0.7, predelay=0.01, build=0.01, seed=309, er=((0.006, 0.4), (0.013, 0.3)), late=(),
                      rt_shape=[0.8, 0.8, 0.8, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3]), -21.0),
    # the black void between floating rocks
    "void":     (dict(rt60=9.0, predelay=0.09, build=0.2, seed=310, er=(), late=(),
                      rt_shape=[1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.55]), -11.0),
}


def room(name, hp=110.0):
    """(IR high-passed so reverbs never carry rumble, default wet dB)"""
    if name not in _ROOMS:
        kw, wet = ROOMS[name]
        _ROOMS[name] = (bandpass(hall_ir(**kw), hp, None, order=2), wet)
    return _ROOMS[name]


def follower(x, sr=SR, hop=0.005, att=0.015, rel=0.35):
    """level envelope in dB (per sample) with attack/release smoothing (in dB)."""
    m = x if x.ndim == 1 else np.sqrt(np.mean(x ** 2, 1))
    h = max(1, int(hop * sr))
    k = len(m) // h + 1
    mm = np.pad(m, (0, k * h - len(m)))
    e = 10 * np.log10(np.mean(mm.reshape(k, h) ** 2, 1) + 1e-14)
    a_att = np.exp(-hop / max(att, 1e-4))
    a_rel = np.exp(-hop / max(rel, 1e-4))
    out = np.empty_like(e)
    s = e[0]
    for i, v in enumerate(e):
        a = a_att if v > s else a_rel
        s = a * s + (1 - a) * v
        out[i] = s
    return np.interp(np.arange(len(m)), np.arange(k) * h + h / 2, out).astype(np.float32)


def ducked_reverb(dry_st, h, wet_db, duck_db=9.0, hp=180.0, lp=7500.0, att=0.02, rel=0.45, key_sig=None):
    """dry stereo + convolved wet, the wet side-chain ducked by the dry voice (duck_db at full speech level,
    fading back over `rel`).  The send is band-limited (no boom, no fizz in the tail).  Output = len(dry)+len(h)."""
    dry_st = stereo(np.asarray(dry_st, np.float32))
    send = bandpass(dry_st, hp, lp, order=2)
    wet = convolve(send, h) * db(wet_db)
    k = stereo(key_sig) if key_sig is not None else dry_st
    e = follower(k.mean(1), att=att, rel=rel)
    ref = np.percentile(e[e > e.max() - 40], 90) if (e > e.max() - 40).any() else e.max()
    act = np.clip((e - (ref - 24)) / 18, 0, 1)
    g = db(-duck_db * act).astype(np.float32)
    wet[:len(g)] *= g[:, None]
    out = wet
    out[:len(dry_st)] += dry_st
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------------ phonemes
VOWELS = set("aeiouæɑɐəɛɜɔʊɪʌɒɚɯɨyøœAIOWYᵊ")
VOICED = set("lmnrvzɾʎɲŋjwʒðβɣʣʤbdɡɹʋɫ")     # voiced consonants carry the sung pitch
MARKS = set("ˈˌː")
PUNCT = set(".,!?;:—…\"“” ")


def vocab_ps(ps):
    m = kokoro().model
    return "".join(c for c in ps if c in m.vocab)


def vowel_positions(ps):
    return [j for j, c in enumerate(ps) if c in VOWELS]


def frame_classes(ps, d):
    """per F0/energy frame (12.5 ms): 2 = vowel, 1 = voiced consonant, 0 = anything else (incl. bos/eos)."""
    cls = np.array([0] + [2 if c in VOWELS else 1 if c in VOICED else 0 for c in ps] + [0])
    return np.repeat(np.repeat(cls, np.asarray(d, int)), 2)


def pitch_blend(F0, tgt, cls, thr=30.0):
    """Binary voicing (Kokoro's sine source is voiced whenever F0 > 10 Hz): vowels, voiced consonants and any
    frame the model itself voices get the target pitch; unvoiced frames keep the model's ~0 (never an
    intermediate pitch from mixing a low natural F0 into the target)."""
    n = len(F0)
    tgt = np.pad(np.asarray(tgt, float)[:n], (0, max(0, n - len(tgt))), mode="edge")
    cls = np.pad(np.asarray(cls)[:n], (0, max(0, n - len(cls))))
    voiced = (cls > 0) | (F0 > thr)
    return np.where(voiced, tgt, np.where(F0 < 10, F0, 0.0))


# ------------------------------------------------------------------------------------------------ chant
SIBILANT = set("szʃʒʧʤʦʣfvθðçx")


def _cons_shape(ps, d, cons_mult, cons_max):
    """sung consonants: crisp.  Stress marks / spaces 1 frame; a doubled consonant's second half 1 frame;
    sibilants/fricatives may keep one frame more than `cons_max` so they stay audible."""
    d = np.array(d).copy()
    for j in range(1, len(d) - 1):
        c = ps[j - 1]
        if c in MARKS or c == " ":
            d[j] = 1
        elif c not in VOWELS and c not in PUNCT:
            if j >= 2 and ps[j - 2] == c:
                d[j] = 1
                continue
            cap = cons_max + (1 if c in SIBILANT else 0)
            d[j] = int(np.clip(round(d[j] * cons_mult), 1, cap))
    return d


def chant(ps, voice, sched, *, cap=None, speed=1.0, cons_mult=1.0, cons_max=4, tail=8, scoop=-0.35,
          glide=0.05, drift=3.0, energy=0.0, formant=1.0, seed=0, lead=None, check=True, takes=3):
    """Sing a phrase.  ps: phoneme string; sched: one entry per vowel of ps (in order):
        {"t": vowel onset (s, phrase-relative, first = 0), "notes": [(midi, dur_s), ...], optional "gain" (energy
         offset, model units), "level" (dB re: the phrase's vowels), "coda" (s: hum the final voiced consonant),
         "scoop" (bool: scoop into this vowel; phrase-initial vowels always scoop)}
    Pauses: if a punctuation symbol sits between two vowels it absorbs the time the notes leave free; otherwise the
    vowel is held (legato).  cap: highest MIDI the voice renders natively; above it the whole phrase is rendered
    transposed down and WORLD-shifted back up (formant factor `formant`).
    Takes are seeded (reproducible); up to `takes` are rendered and the most in-tune one is kept.
    Returns (audio24k, info) with info["v0"] = time in the audio of the first vowel's onset (alignment),
    info["onsets"] = alignment onset of every vowel (audio time), info["cents"] = per-note pitch check."""
    ps = vocab_ps(ps)
    vp = vowel_positions(ps)
    assert len(vp) == len(sched), (len(vp), len(sched), ps)
    mids = [m for s in sched for m, _ in s["notes"]]
    shift = 0.0
    if cap is not None and max(mids) > cap:
        shift = float(max(mids) - cap)
    parts = ("chant12", ps, voice, json.dumps(sched), cap, speed, cons_mult, cons_max, tail, scoop, glide, drift,
             energy, formant, seed, lead)

    def make(take=0):
        K = kokoro()
        K.torch.manual_seed(int(key(*parts, take)[:8], 16))       # the decoder's noise: reproducible takes
        _, dn, F0n, Nn = K.forward(ps, voice, speed)
        d = _cons_shape(ps, dn, cons_mult, cons_max)
        if lead is not None:
            d[0] = lead
        # vowel durations from the score
        for k, j in enumerate(vp):
            nd = sum(x for _, x in sched[k]["notes"])
            if k + 1 < len(vp):
                j2 = vp[k + 1]
                ioi = round((sched[k + 1]["t"] - sched[k]["t"]) / FR)
                between = list(range(j + 2, j2 + 1))                   # d indices of symbols between the vowels
                pun = [i for i in between if ps[i - 1] in ".,!?;:—…"]
                fixed = sum(d[i] for i in between if i not in pun)
                vf = max(2, int(round(nd / FR)))
                if pun:
                    rest = ioi - fixed - vf
                    if rest < len(pun):
                        vf = max(2, ioi - fixed - len(pun))
                        rest = len(pun)
                    for q, i in enumerate(pun):
                        d[i] = rest // len(pun) + (1 if q < rest % len(pun) else 0)
                else:
                    vf = max(2, ioi - fixed)
                d[j + 1] = vf
            else:
                d[j + 1] = max(2, int(round(nd / FR)))
                codas = [i for i in range(j + 2, len(d) - 1) if ps[i - 1] in VOICED]
                for i in range(j + 2, len(d) - 1):
                    if ps[i - 1] in ".,!?;:—…":
                        d[i] = 1
                if sched[k].get("coda") and codas:       # a hummed final consonant (…-cum: m on the final)
                    d[codas[-1]] = max(1, int(round(sched[k]["coda"] / FR)))
                d[-1] = tail
        c = np.concatenate([[0], np.cumsum(d)])                      # alignment frame boundaries
        v_on = [float(c[j + 1] * FR) for j in vp]                     # vowel alignment onsets (s)
        v_end = [float(c[j + 2] * FR) for j in vp]
        n_f0 = int(c[-1] * 2)
        tf = np.arange(n_f0) * FR / 2
        r = rng_for("chant", voice, seed, ps)
        # target pitch (semitones) as a function of time
        pts_t, pts_m = [], []
        for k in range(len(vp)):
            t = v_on[k]
            for m, du in sched[k]["notes"]:
                g = min(glide, 0.4 * du)
                pts_t += [t + g * 0.5, t + max(du - g * 0.5, g * 0.5 + 1e-3)]
                pts_m += [m, m]
                t += du
        order = np.argsort(pts_t, kind="stable")
        pts_t, pts_m = np.asarray(pts_t)[order], np.asarray(pts_m, float)[order]
        mt = np.interp(tf, pts_t, pts_m)
        # round the corners
        kk = max(1, int(0.5 * glide / (FR / 2)))
        mt = np.convolve(np.pad(mt, kk, mode="edge"), np.hanning(2 * kk + 1) / np.hanning(2 * kk + 1).sum(), "valid")
        # scoop into phrase-initial vowels (and any marked "scoop"), from below, fading in ~60 ms
        for k in range(len(vp)):
            if not (k == 0 or sched[k].get("scoop", False)):
                continue
            u = tf - v_on[k]
            m = (u >= -0.03) & (u < 0.2)
            mt[m] += scoop * np.exp(-np.maximum(u[m], 0) / 0.06)
        # slow drift (a living voice), seeded
        wk = np.cumsum(r.standard_normal(n_f0)) * 0.02
        wk = np.convolve(wk - wk.mean(), np.ones(40) / 40, mode="same")
        mt += (wk / (np.abs(wk).max() + 1e-9)) * drift / 100.0
        tgt = midi_hz(mt - shift)

        cls = frame_classes(ps, d)

        def dur_fn(_d):
            return d.copy()

        def f0_fn(F0, _d):
            return pitch_blend(F0, tgt, cls)

        # energy: every vowel on the natural plateau of the utterance (+gain); voiced consonants between vowels
        # held just under it (legato: low energy + falling pitch is where the model creaks); smooth inside holds
        base = float(np.percentile(Nn[Nn > np.percentile(Nn, 30)], 85)) if len(Nn) else 0.0
        tgs = [base + sched[k].get("gain", 0.0) + energy for k in range(len(vp))]

        jl = vp[-1]
        coda_end = rel_end = c[jl + 2]
        for i in range(jl + 2, len(d) - 1):                # coda consonants of the last syllable
            ch = ps[i - 1]
            if ch in ".,!?;:—…":
                break
            rel_end = c[i + 1]
            if ch in VOICED and coda_end == c[i]:
                coda_end = c[i + 1]                        # voiced run straight after the vowel: hummed
        sung_end = float(coda_end * FR)

        def n_fn(N, _d):
            out = N.copy()
            n = len(N)
            i = np.arange(n)
            cl = np.pad(cls[:n], (0, max(0, n - len(cls))))
            for k in range(len(vp)):
                a, b = int(v_on[k] / (FR / 2)), int(v_end[k] / (FR / 2))
                m = np.clip((i - a + 1) / 3, 0, 1) * np.clip((b - i) / 4, 0, 1)
                out = (1 - m) * out + m * np.maximum(out, tgs[k])
                if k + 1 < len(vp):
                    a2 = int(v_on[k + 1] / (FR / 2))
                    seg = slice(max(0, b - 2), a2 + 2)
                    lvl = min(tgs[k], tgs[k + 1]) - 0.4
                    out[seg] = np.where(cl[seg] == 1, np.maximum(out[seg], lvl), out[seg])
            b, e = int(v_end[-1] / (FR / 2)) - 2, int(sung_end / (FR / 2))
            if e > b + 2:                                   # hummed coda: sustained, dying away at its end
                seg = slice(b, e)
                lvl = tgs[-1] - 0.5 - 0.8 * np.clip((i[seg] - b) / max(1, e - b), 0, 1) ** 2
                out[seg] = np.where(cl[seg] == 1, np.maximum(out[seg], lvl), out[seg])
            sm = np.convolve(np.pad(out, 6, mode="edge"), np.ones(13) / 13, mode="valid")
            return np.where(cl == 2, np.maximum(sm, out), out)
        K.torch.manual_seed(int(key(*parts, take, "sing")[:8], 16))
        a, dd, _, _ = K.forward(ps, voice, speed, dur_fn=dur_fn, f0_fn=f0_fn, n_fn=n_fn)
        assert np.array_equal(dd, d)
        a, fixed = decreak(a, tf, tgt, cls)
        # release: the phrase ends with its last sung sound (no model fry in the <eos> tail)
        i1 = int((float(rel_end * FR) + 0.04) * VSR)
        nf = int(0.07 * VSR)
        if i1 < len(a):
            a = a.copy()
            a[max(0, i1 - nf):i1] *= np.cos(np.linspace(0, np.pi / 2, i1 - max(0, i1 - nf))).astype(np.float32) ** 2
            a[i1:] = 0
        if shift or formant != 1.0:
            a = world_shift_to(a, tf, tgt, shift, formant)
        a, even_db = even_vowels(a, v_on, v_end, [s.get("level", 0.0) for s in sched])
        info = dict(ps=ps, v0=v_on[0], onsets=v_on, ends=v_end, shift=shift, durs=[int(x) for x in d],
                    decreak_s=round(fixed, 3), sung_end=sung_end, even_db=even_db, take=take)
        return a, info

    def best_take():
        """up to `takes` renders; keep the one whose notes are most in tune (octave slips / unvoiced notes are
        what a retake is for)"""
        best = None
        for take in range(max(1, takes)):
            a, info = make(take)
            info["cents"] = chant_cents(a, info, sched)
            errs = [abs(c) if np.isfinite(c) else 600.0 for _, c, _ in info["cents"]]
            score = float(np.sum(np.minimum(errs, 600.0)) + 200.0 * sum(e > 50 for e in errs))
            if best is None or score < best[0]:
                best = (score, a, info)
            if max(errs) <= 35.0:
                break
        return best[1], best[2]
    a, info = cached_json("chant", parts, best_take)
    if check and "cents" not in info:
        info["cents"] = chant_cents(a, info, sched)
    return a, info


def world_shift_to(a, tt, thz, shift, formant=1.0, sr=VSR):
    """WORLD pitch shift that trusts the score, not the tracker: voiced frames are re-synthesized on the intended
    render pitch (thz at times tt) x 2^(shift/12) — a tracker octave error can no longer become a sung wrong note;
    the spectral envelope / aperiodicity are analysed on that same corrected F0 (formants kept, optionally warped)."""
    import pyworld as pw
    x = np.asarray(a, np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=55, f0_ceil=900, frame_period=5.0)
    tg = np.interp(t, tt, thz)
    f0c = np.where(f0 > 0, tg, 0.0)
    sp = pw.cheaptrick(x, f0c, t, sr)
    ap = pw.d4c(x, f0c, t, sr)
    if formant != 1.0:
        sp = warp_sp(sp, formant)
    y = world_synth(f0c * 2 ** (shift / 12), sp, ap, sr)[:len(x)]
    return np.pad(y, (0, max(0, len(x) - len(y)))).astype(np.float32)


def even_vowels(a, on, end, level, sr=VSR, lo=-6.0, hi=6.0):
    """Sung vowels at an even level (the model swallows some vowels, e.g. the a of "gratias"): each vowel's RMS is
    brought to the phrase median (+ its `level` dB) by a smooth gain curve through the vowel centres (clamped
    lo..hi dB); consonants follow the curve.  Returns (audio, applied dB per vowel)."""
    x = np.asarray(a, np.float32)
    meas = []
    for s, e in zip(on, end):
        seg = x[int((s + 0.02) * sr):max(int((s + 0.02) * sr) + 1, int((e - 0.02) * sr))]
        meas.append(10 * np.log10(np.mean(seg ** 2) + 1e-12) if len(seg) else -120.0)
    meas = np.array(meas)
    ref = np.median(meas)
    g = np.clip(ref + np.asarray(level, float) - meas, lo, hi)
    cen = (np.asarray(on) + np.asarray(end)) / 2
    t = np.arange(len(x)) / sr
    curve = np.interp(t, cen, g)
    k = int(0.03 * sr)
    curve = np.convolve(np.pad(curve, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    return (x * db(curve)).astype(np.float32), [round(float(v), 1) for v in g]


def decreak(a, tt, thz, cls, sr=VSR, tol=250.0, fade=0.015):
    """Repair creak / period-doubling (the model's learned phrase-final fry): inside vowels, where WORLD's F0
    strays more than `tol` cents from the intended pitch (thz at times tt) or drops out, resynthesize with WORLD on
    the intended pitch (clean frames' aperiodicity) and cross-fade that in.  Returns (audio, repaired seconds)."""
    import pyworld as pw
    x = np.asarray(a, np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=45, f0_ceil=900, frame_period=5.0)
    tg = np.interp(t, tt, thz)
    ct = np.arange(len(cls)) * FR / 2
    cl = np.round(np.interp(t, ct, np.asarray(cls, float))).astype(int)
    vow = cl >= 1
    off = np.abs(1200 * np.log2(np.maximum(f0, 1.0) / np.maximum(tg, 1.0))) > tol
    bad = ((cl == 2) & ((f0 <= 0) | off)) | ((cl == 1) & (f0 > 0) & off)
    if not bad.any():
        return np.asarray(a, np.float32), 0.0
    bad = np.convolve(bad.astype(float), np.ones(5), mode="same") > 0          # +-10 ms
    f0f = np.where(bad, tg, f0)
    f0f = np.where(vow & (f0f <= 0), tg, f0f)
    sp = pw.cheaptrick(x, f0f, t, sr)
    ap = pw.d4c(x, f0f, t, sr)
    good = vow & ~bad & (f0 > 0)
    if good.any():
        ap[bad] = np.minimum(ap[bad], np.median(ap[good], 0))
    y = pw.synthesize(f0f, sp, ap, sr, frame_period=5.0)[:len(x)]
    y = np.pad(y, (0, max(0, len(x) - len(y))))
    m = np.interp(np.arange(len(x)) / sr, t, bad.astype(float))
    k = max(1, int(fade * sr))
    m = np.clip(np.convolve(m, np.ones(k) / k, mode="same") * 1.5, 0, 1)
    out = x * (1 - m) + y * m
    return out.astype(np.float32), float(bad.sum() * 0.005)


def chant_cents(a, info, sched, sr=VSR):
    """median cents error of every note (its middle ~55%, clipped to the vowel actually sung — a note longer than
    its vowel spills into consonants), + voiced fraction.  Pitch floor just below the lowest note keeps the
    analysis window short (short melisma notes)."""
    lo = min(float(midi_hz(m)) for s in sched for m, _ in s["notes"])
    ts, hz = pitch_track(a, sr, max(40.0, 0.75 * lo), 900, 0.004)
    out = []
    for k, s in enumerate(sched):
        v_s = info["onsets"][k] + 0.015
        v_e = info["ends"][k] - 0.01
        if k == len(sched) - 1:
            v_e = max(v_e, info.get("sung_end", v_e) - 0.03)
        t = v_s
        for m, du in s["notes"]:
            a0, a1 = max(t + 0.25 * du, v_s), min(t + 0.8 * du, v_e)
            if a1 - a0 < 0.02:
                a0, a1 = max(t, v_s), min(t + du, v_e)
            t += du
            win = (ts >= a0) & (ts <= a1)
            sel = win & (hz > 0)
            if not sel.any():
                out.append((m, float("nan"), 0.0))
                continue
            c = 1200 * np.log2(hz[sel] / float(midi_hz(m)))
            out.append((m, round(float(np.median(c)), 1), round(float(sel.sum() / max(1, win.sum())), 2)))
    return out


# ------------------------------------------------------------------------------------------------ lock-step voices
def ref_durations(ps, voice, speed=1.0):
    _, dn, F0n, Nn = kokoro().forward(vocab_ps(ps), voice, speed)
    return dn, F0n, Nn


def sync_voice(ps, voice, durs, f0_mode="mono", midi=50.0, ref_f0=None, semis=0.0, spread=1.0, energy=0.0,
               seed=0):
    """`ps` spoken by `voice` with the given alignment durations (lock-step with the reference).
    f0_mode: "mono" = monotone at `midi` (unvoiced frames stay unvoiced), "ref" = the reference contour moved by
    `semis` and expanded by `spread` around its median."""
    ps = vocab_ps(ps)
    parts = ("sync4", ps, voice, [int(x) for x in durs], f0_mode, round(float(midi), 3), semis, spread, energy, seed,
             None if ref_f0 is None else key(np.round(np.asarray(ref_f0), 1).tolist()))

    def make():
        K = kokoro()
        K.torch.manual_seed(int(key(*parts)[:8], 16))
        cls = frame_classes(ps, durs)

        def f0_fn(F0, _d):
            if f0_mode == "mono":
                r = rng_for("syncdrift", voice, seed)
                wk = np.convolve(np.cumsum(r.standard_normal(len(F0))) * 0.02, np.ones(30) / 30, mode="same")
                wk = (wk - wk.mean()) / (np.abs(wk - wk.mean()).max() + 1e-9)
                tgt = midi_hz(midi + 0.04 * wk)
                return pitch_blend(F0, tgt, cls)
            rf = np.asarray(ref_f0, float)[:len(F0)]
            rf = np.pad(rf, (0, max(0, len(F0) - len(rf))), mode="edge")
            v = rf > 40
            med = np.median(rf[v]) if v.any() else 150.0
            rfi = np.interp(np.arange(len(rf)), np.flatnonzero(v), rf[v]) if v.any() else np.full(len(rf), med)
            tgt = med * 2 ** (semis / 12) * (rfi / med) ** spread
            return pitch_blend(np.maximum(F0, np.where(v, rf, 0.0)), tgt, cls)

        def n_fn(N, _d):
            return N + energy
        a, dd, _, _ = K.forward(ps, voice, 1.0, dur_fn=lambda _d: np.asarray(durs).copy(), f0_fn=f0_fn, n_fn=n_fn)
        return a
    return cached_wav("sync", parts, make)


def align_lag(a, b, sr, maxlag=0.08):
    """lag (s) of b relative to a from 2 ms band envelopes (positive = b later)."""
    def env(x):
        y = bandpass(x, 300, 3400, sr, order=2)
        h = int(0.002 * sr)
        n = len(y) // h
        e = 20 * np.log10(np.sqrt(np.mean(y[:n * h].reshape(n, h) ** 2, 1)) + 1e-7)
        return np.maximum(e, -60) + 60, h
    ea, h = env(a)
    eb, _ = env(b)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    L = int(maxlag * sr / h)
    cs = [np.dot(ea[max(0, -k):n - max(0, k)], eb[max(0, k):n - max(0, -k)]) for k in range(-L, L + 1)]
    return (int(np.argmax(cs)) - L) * h / sr


# ------------------------------------------------------------------------------------------------ misc
def gate_after(buf, t_end, fade_s=0.0015):
    """razor cut: raised-cosine over `fade_s` ending exactly at t_end, digital zero afterwards."""
    i1 = int(round(t_end * SR))
    nf = int(round(fade_s * SR))
    buf[i1 - nf:i1] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, nf)))[:, None].astype(np.float32)
    buf[i1:] = 0


def silence(buf):
    """the sacred silence 145.0-146.5: digital zero (anything reaching it is cut with a 20 ms fade)"""
    i0, i1 = int(SILENCE[0] * SR), int(SILENCE[1] * SR)
    nf = int(0.02 * SR)
    if np.abs(buf[i0:i1]).max() > 0:
        buf[i0 - nf:i0] *= np.linspace(1, 0, nf)[:, None].astype(np.float32)
    buf[i0:i1] = 0
    return buf


def section_norm(buf, t0, t1, target):
    L = lufs(buf[int(t0 * SR):int(t1 * SR)])
    if np.isfinite(L):
        buf *= np.float32(db(target - L))
    return buf


def place(buf, mono48, t, p, g_db=0.0, fin=0.01, fout=0.03):
    add_at(buf, pan(fade(mono48, fin, fout) * db(g_db), p), t)


def reverb_add(sec, h, wet_db, t_pad=0.0):
    """add the convolved wet of a sparse full-length section buffer back into it (only its active span)."""
    nz = np.flatnonzero(np.abs(sec).max(1) > 1e-7)
    if not len(nz):
        return sec
    a, b = nz[0], min(NS, nz[-1] + 1)
    out = sec.copy()
    add_at(out, convolve(sec[a:b], h) * db(wet_db), a / SR)
    return out


def export_cantor(section, data):
    """per-syllable cantor onsets for the scene owners (M01 incipit, M09 antiphon, M10 explicit):
    films/opus/cache/foley/cantor_syllables.json {section: [{syl, t, end}, ...]} (global seconds)"""
    d = FILM_ROOT / "cache" / "foley"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "cantor_syllables.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[section] = data
    p.write_text(json.dumps(cur, indent=1))


def limit(x, ceiling_db=-2.0, look_ms=5.0, release_ms=80.0):
    """look-ahead brickwall limiter on a stereo buffer (sample peak), transparent below the ceiling"""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    ceil = db(ceiling_db)
    pk = np.abs(x).max(axis=1)
    if pk.max() <= ceil:
        return x
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    la = max(1, int(look_ms * SR / 1000))
    g = minimum_filter1d(g, size=2 * la + 1)
    g = uniform_filter1d(g, size=la)
    rel = int(release_ms * SR / 1000)
    g = np.minimum(g, uniform_filter1d(minimum_filter1d(g, size=rel), size=rel))
    return (x * g[:, None]).astype(np.float32)


def dump(name, obj):
    VOCAL.mkdir(parents=True, exist_ok=True)
    (VOCAL / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False))

"""Builds audio/stems/choir.wav — Kokoro voices made to SING.

(a) 150.0-154.5  THE UNBABELING CONVERGENCE: 32 babble voices in 8 languages speak Vexillomantic lore; their
    phoneme durations are progressively stretched (speech slows into chant) and their F0 is glided, in the
    log domain, from the model's own natural speech contour onto D-major chord tones spread D2..A5
    (the THX 'Deep Note' idea: chaos converging into consonance).  Each voice ends on a held vowel that
    lands on the chord at 154.5 exactly as the hymn begins.  Targets above a voice's comfortable register
    are reached by a time-varying WORLD vocoder shift (formants preserved).
(b) 154.5-165.93 THE HYMN "Flags be. Flags are. Flags will. Flags." 84 BPM, SATB from timeline.json.
    Each note of each of 24 singers (6 per part) is a separate Kokoro forward pass with prosody injection:
    the vowel's alignment duration is set to the note length, F0 is forced to the MIDI pitch (A=440 ET) with
    a small scoop and delayed natural vibrato (~5.5 Hz), energy is held (and swelled on the last 'Flags.').
    The rendered vowel onset is measured and placed exactly on the beat (+-15 ms jitter, +-8 cent detune).
    High notes are rendered in the singer's comfortable register and WORLD-shifted up.  Cathedral reverb.

usage: python tools/vocal_choir.py            (renders are cached in audio/vocal/cache/sing, conv)
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocal_corpus import BABEL_TX, LORE
from vocal_lib import (CACHE, NS, SR, TL, VOCAL, VSR, add_at, bandpass, convolve, db, is_male, key, kokoro,
                       make_ir, midi_hz, pan, pitch_track, rng_for, smoothstep, up48, warp_sp, world_analyze,
                       world_synth, write_stem)

HY = TL["hymn"]
T0 = HY["start"]                 # 154.5
SPB = 60.0 / HY["bpm"]           # 0.714 s
NOTES = HY["notes"]              # [word, beat, beats, [S, A, T, B]]
END = T0 + HY["beats"] * SPB     # 165.93
FR = 600 / VSR                   # Kokoro alignment frame = 25 ms
ONSET_MAX, CODA_MAX = 4, 3       # sung consonants: crisp (<=100 ms onset, <=75 ms per coda consonant)
FRICATIVES = set("zsʃʒfvθð")     # ...except coda fricatives, which get one frame more so 'Flags' keeps its z
PARTS_T0 = 149.0                 # choir_parts/*.wav start time

# choral diction (tall British vowels for every singer): Flahgs / bee / ah / will
PS = {"Flags": "flˈaɡz", "be": "bˈiː", "are": "ˈɑː", "will": "wˈɪl"}
VOWELS = set("aæiɑɪAIOWYəɛɜɔuʊʌɐɒeoɚɯɨ")

PARTS = {
    "S": ["af_heart", "af_bella", "bf_emma", "bf_isabella", "af_aoede", "af_kore"],
    "A": ["af_sarah", "bf_alice", "bf_lily", "af_jessica", "af_nova", "af_river"],
    "T": ["am_michael", "am_eric", "am_puck", "am_liam", "bm_daniel", "am_echo"],
    "B": ["am_fenrir", "bm_george", "am_onyx", "am_adam", "em_santa", "bm_fable"],
}
PART_PAN = {"S": -0.55, "A": -0.18, "T": 0.18, "B": 0.55}
PART_GAIN = {"S": 0.0, "A": -0.5, "T": -0.5, "B": 0.5}
# dynamics per note (dB), phrase by phrase: mp -> mf -> f -> ff swell
NOTE_DB = [-7.0, -7.5, -5.0, -5.5, -3.0, -3.5, -2.0]
PHRASE_ENDS = {4, 8, 12, 16}


def vowel_index(ps):
    return next(j for j, c in enumerate(ps) if c in VOWELS)


def acoustic_onset(a, sr=VSR, win=0.01, rel=-7.0):
    """time (s) where the level first reaches within `rel` dB of the sustained (median of top half) level."""
    h = int(win * sr)
    n = len(a) // h
    e = 20 * np.log10(np.sqrt(np.mean(a[:n * h].reshape(n, h) ** 2, 1)) + 1e-9)
    ref = np.median(np.sort(e)[n // 2:])
    k = np.flatnonzero(e > ref + rel)
    return (k[0] + 0.5) * h / sr if len(k) else 0.0


# ------------------------------------------------------------------------------------------------ singing
def sing(ps, voice, vowel_s, midi, render_midi=None, vib=(5.5, 0.3, 0.0), scoop=-0.45, detune=0.0,
         n_off=-0.5, swell=None, cons_mult=1.35, lead=None, tail=5, formant=1.0, coda_max=CODA_MAX):
    """One sung syllable.  Returns (audio24k, vowel_onset_s, vowel_end_s).
    vowel_s: desired acoustic length of the sustained vowel.  render_midi: pitch Kokoro actually sings
    (comfortable register); the result is WORLD-shifted to `midi`."""
    rm = midi if render_midi is None else render_midi
    k_ = key("sing6", ps, voice, round(vowel_s, 3), midi, rm, vib, scoop, round(detune, 2), n_off,
             swell is not None and swell.__name__, cons_mult, lead, tail, round(formant, 3), coda_max)
    path = CACHE / "sing" / f"{voice}_{k_}.wav"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        m = json.loads(meta.read_text())
        return sf.read(path, dtype="float32")[0], m["on"], m["end"]
    K = kokoro()
    vi = vowel_index(ps)
    k = vi + 1                         # duration index of the vowel (0 = <bos>)
    _, dn, _, Nnat = K.forward(ps, voice)
    cn = np.concatenate([[0], np.cumsum(dn)]) * 2
    plateau = float(Nnat[int(cn[max(1, k - 2)]):int(cn[k + 1]) + 6].max()) + n_off
    f_tgt = float(midi_hz(rm + detune / 100.0))
    rate, depth, phase = vib

    def build(vframes):
        def dur_fn(d):
            d = shape_consonants(ps, d, k, cons_mult, coda_max)
            if lead is not None:      # keep the model's natural <bos> silence by default: onsets need it
                d[0] = lead
            d[k] = vframes
            d[-1] = tail
            return d

        def region(d):
            c = np.concatenate([[0], np.cumsum(d)]) * 2
            return int(c[k]), int(c[k + 1])

        def f0_fn(F0, d):
            v0, v1 = region(d)
            n = len(F0)
            tv = (np.arange(n) - v0) / 80.0
            vb = depth * np.sin(2 * np.pi * rate * np.maximum(tv, 0) + phase) * np.clip((tv - 0.3) / 0.4, 0, 1)
            sc = scoop * np.exp(-np.maximum(tv + 0.04, 0) / 0.08)
            tgt = f_tgt * 2 ** ((vb + sc) / 12)
            w = np.clip(F0 / 60, 0, 1)
            w[v0 + 4:v1 - 2] = 1.0
            return w * tgt + (1 - w) * F0

        def n_fn(N, d):
            v0, v1 = region(d)
            i = np.arange(len(N))
            m = np.clip((i - (v0 + 2)) / 8, 0, 1) * np.clip((v1 - i) / 6, 0, 1)
            tgt = np.full(len(N), plateau)
            if swell is not None:
                tgt[v0:v1] += swell(np.arange(v1 - v0) / 80.0, (v1 - v0) / 80.0)
            out = (1 - m) * N + m * np.maximum(N, tgt)
            # inside the held vowel: a smooth energy contour (the model's own N collapses mid-vowel)
            a0, a1 = v0 + 2, max(v0 + 3, v1 - 3)
            k9 = np.ones(13) / 13
            sm = np.convolve(np.pad(out, 6, mode="edge"), k9, mode="valid")
            out[a0:a1] = np.maximum(sm[a0:a1], tgt[a0:a1] * m[a0:a1] + out[a0:a1] * (1 - m[a0:a1]))
            return out
        return K.forward(ps, voice, dur_fn=dur_fn, f0_fn=f0_fn, n_fn=n_fn)

    vf = max(4, int(round(vowel_s / FR)))
    a, d, _, _ = build(vf)
    on = acoustic_onset(a)
    align_end = float(np.sum(d[:k + 1]) * FR)
    # the acoustic vowel starts a little after its alignment boundary: lengthen to keep the vowel's end
    extra = on - float(np.sum(d[:k]) * FR)
    if abs(extra) > FR:
        vf2 = max(4, int(round((vowel_s + extra) / FR)))
        if vf2 != vf:
            a, d, _, _ = build(vf2)
            on = acoustic_onset(a)
            align_end = float(np.sum(d[:k + 1]) * FR)
    if rm != midi or formant != 1.0:
        f0, t, sp, ap = world_analyze(a, VSR, f0_floor=55, f0_ceil=900)
        if formant != 1.0:
            sp = warp_sp(sp, formant)
        a = world_synth(f0 * 2 ** ((midi - rm) / 12), sp, ap, VSR)[:len(a)]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, a, VSR, subtype="FLOAT")
    meta.write_text(json.dumps({"on": on, "end": align_end}))
    return a, on, align_end


def final_swell(t, T):
    """energy offset over the last note: crescendo into the cutoff (N units ~ 5 dB each)."""
    return 0.9 * smoothstep(t / max(T * 0.8, 1e-3)) - 0.2


def singer_plan(part, voice, idx):
    r = rng_for("singer", part, voice)
    male = is_male(voice)
    cap = (57 + r.integers(0, 3)) if male else (61 + r.integers(0, 3))
    return dict(detune=float(r.uniform(-8, 8)), bias=float(r.uniform(-0.006, 0.006)),
                rate=float(r.uniform(5.1, 5.9)), depth=float(r.uniform(0.22, 0.36)),
                cap=int(cap), pan=float(np.clip(PART_PAN[part] + r.uniform(-0.14, 0.14), -0.9, 0.9)),
                gain=float(r.uniform(-1.5, 1.0)), seed=idx)


def shape_consonants(ps, d, k, cons_mult=1.35, coda_max=CODA_MAX):
    """consonant durations for singing: onsets slightly lengthened (diction), codas short; marks = 1 frame."""
    d = np.array(d).copy()
    for j in range(1, len(d) - 1):
        if j == k:
            continue
        if ps[j - 1] in "ˈˌː":
            d[j] = 1
        elif j < k:
            d[j] = int(np.clip(round(d[j] * cons_mult), 1, ONSET_MAX))
        else:
            cap = coda_max + (1 if ps[j - 1] in FRICATIVES else 0)
            d[j] = int(np.clip(round(d[j] * cons_mult), 1, cap))
    return d


@functools.lru_cache(maxsize=None)
def cons_dur(ps, voice, cons_mult=1.35, coda_max=CODA_MAX):
    """(onset, coda) consonant durations (s) of a sung word."""
    _, dn, _, _ = kokoro().forward(ps, voice)
    k = vowel_index(ps) + 1
    d = shape_consonants(ps, dn, k, cons_mult, coda_max)
    return float(np.sum(d[1:k]) * FR), float(np.sum(d[k + 1:-1]) * FR)


def render_hymn():
    """returns dict part -> stereo 48k buffer (full length) of the dry choir, plus per-note check data."""
    bus = {p: np.zeros((NS, 2), np.float32) for p in PARTS}
    checks = []
    for part, voices in PARTS.items():
        pi = "SATB".index(part)
        for vi_, voice in enumerate(voices):
            pl = singer_plan(part, voice, vi_)
            r = rng_for("hymn", voice)
            for ni, (word, beat, beats, pitches) in enumerate(NOTES):
                ps = PS[word]
                midi = pitches[pi]
                last = ni == len(NOTES) - 1
                cmax = 4 if last else CODA_MAX
                t_on = T0 + beat * SPB + float(r.uniform(-0.015, 0.015)) + pl["bias"]
                end_beat = beat + beats
                _, coda = cons_dur(ps, voice, 1.35, cmax)
                if last:
                    v_end = END - coda                                             # final cutoff on 165.93
                else:
                    nxt_on, _ = cons_dur(PS[NOTES[ni + 1][0]], voice)
                    gap = 0.16 if end_beat in PHRASE_ENDS else 0.01               # breath / legato
                    v_end = T0 + end_beat * SPB - nxt_on - coda - gap
                formant = 1.0 + 0.12 * float(np.clip((midi - 64) / 10, 0, 1))
                vib = (pl["rate"], pl["depth"], float(r.uniform(0, 6.28)))
                scoop = float(r.uniform(-0.6, -0.25))
                vdur = max(0.2, v_end - t_on)
                # render in the singer's comfortable register; verify the pitch; if the model misbehaves
                # (sub-harmonic / fry), drop the render register and let WORLD carry the rest
                best = None
                for attempt, cap in enumerate((pl["cap"], pl["cap"] - 3, pl["cap"] - 6, pl["cap"] - 9)):
                    rm = min(midi, cap)
                    a, on, _ = sing(ps, voice, vdur, midi, rm, vib=vib, scoop=scoop, detune=pl["detune"],
                                    n_off=-0.5 + (0.3 if last else 0.0), swell=final_swell if last else None,
                                    formant=formant, coda_max=cmax)
                    med, p10, p90, vf = note_cents(a, on, vdur, midi)
                    err = abs(med - pl["detune"]) if np.isfinite(med) else 1e9
                    score = err + (0.9 - vf) * 200 * (vf < 0.9)
                    if best is None or score < best[0]:
                        best = (score, a, on, rm, med, p10, p90, vf, attempt)
                    if err <= 12 and vf >= 0.9:
                        break
                _, a, on, rm, med, p10, p90, vf, attempt = best
                x = up48(a)
                # note dynamics: messa di voce + phrase level + final crescendo
                tt = np.arange(len(x)) / SR - on
                if last:
                    env = db(NOTE_DB[ni] + 5.0 * smoothstep(tt / (vdur * 0.85)))
                else:
                    env = db(NOTE_DB[ni] + 1.2 * np.sin(np.pi * np.clip(tt / vdur, 0, 1)))
                x = x * env.astype(np.float32) * db(pl["gain"] + PART_GAIN[part])
                add_at(bus[part], pan(x, pl["pan"]), t_on - on)
                checks.append(dict(part=part, voice=voice, note=ni, word=word, midi=midi, render_midi=rm,
                                   attempt=attempt, vowel_onset=round(t_on, 4), vowel_dur=round(vdur, 3),
                                   detune=round(pl["detune"], 1), median_cents=round(med, 1),
                                   p10=round(p10, 1), p90=round(p90, 1), voiced_frac=round(vf, 2)))
            print(f"hymn {part} {voice} done", flush=True)
    return bus, checks


def note_cents(a, on, vdur, midi):
    """pitch of the sustained vowel (from 0.25 s after the vowel onset to 0.1 s before its end), in cents
    re: the equal-tempered note.  Returns (median, p10, p90, voiced fraction)."""
    ts, hz = pitch_track(a, VSR, 50, 1100, 0.01)
    win = (ts > on + 0.25) & (ts < on + vdur - 0.1)
    sel = win & (hz > 0)
    if not sel.any():
        return float("nan"), float("nan"), float("nan"), 0.0
    c = 1200 * np.log2(hz[sel] / float(midi_hz(midi)))
    return (float(np.median(c)), float(np.percentile(c, 10)), float(np.percentile(c, 90)),
            float(sel.sum() / max(1, win.sum())))


# ------------------------------------------------------------------------------------------------ breaths
def breaths(buf):
    """catch-breaths of the whole choir before each phrase (tiny, filtered noise inhales)."""
    for ph_beat in (0, 4, 8, 12):
        for part, voices in PARTS.items():
            for voice in voices:
                r = rng_for("breath", voice, ph_beat)
                dur = float(r.uniform(0.18, 0.3)) if ph_beat else float(r.uniform(0.35, 0.5))
                n = int(dur * SR)
                x = bandpass(r.standard_normal(n).astype(np.float32), 600, 3800, order=2)
                x = bandpass(x, 900 if is_male(voice) else 1300, None, order=1)
                env = np.sin(np.linspace(0, np.pi, n)) ** 1.5 * np.linspace(0.6, 1.0, n)
                x = x * env.astype(np.float32) * db(-47 if ph_beat else -44)
                t = T0 + ph_beat * SPB - 0.13 - dur + float(r.uniform(-0.04, 0.02))
                add_at(buf, pan(x, PART_PAN[part] + float(r.uniform(-0.15, 0.15))), t)


# ------------------------------------------------------------------------------------------------ convergence
CONV_START, CONV_LAND = 150.0, T0


def conv_voices():
    """32 voices across 8 languages, each assigned a chord tone (males low, females high)."""
    male_tones = [38, 38, 45, 45, 45, 50, 50, 50, 54, 54, 57, 57, 57, 62, 62, 50]
    fem_tones = [62, 62, 66, 66, 66, 69, 69, 69, 74, 74, 74, 78, 78, 81, 81, 69]
    males = ["am_fenrir", "bm_george", "am_onyx", "am_adam", "em_alex", "hm_omega", "im_nicola", "pm_alex",
             "zm_yunxi", "jm_kumo", "am_michael", "bm_daniel", "hm_psi", "zm_yunjian", "am_liam", "pm_santa"]
    fems = ["af_heart", "bf_emma", "ef_dora", "ff_siwis", "if_sara", "pf_dora", "hf_alpha", "jf_alpha",
            "zf_xiaoxiao", "af_bella", "bf_isabella", "zf_xiaoyi", "jf_nezumi", "af_aoede", "hf_beta", "bf_lily"]
    out = []
    for v, m in zip(males, male_tones):
        out.append((v, m))
    for v, m in zip(fems, fem_tones):
        out.append((v, m))
    return out


def conv_text(voice, r):
    lang = voice[0]
    if lang in "ab":
        return LORE[int(r.integers(0, len(LORE)))]
    return BABEL_TX[lang][int(r.integers(0, len(BABEL_TX[lang])))]


def converge_voice(voice, tone, idx):
    """One voice of the convergence.  Returns (stereo48k, start_time)."""
    r = rng_for("conv", voice, idx)
    k_ = key("conv5", voice, tone, idx)
    path = CACHE / "conv" / f"{voice}_{k_}.wav"
    meta = path.with_suffix(".json")
    t_speech = CONV_START + float(r.uniform(0.0, 0.45))          # first phoneme
    if path.exists() and meta.exists():
        return sf.read(path, dtype="float32")[0], json.loads(meta.read_text())["t_start"]
    K = kokoro()
    lang = voice[0]
    text = conv_text(voice, r)
    ps = K.phonemes(text, lang)
    ps = "".join(c for c in ps if c in K.model.vocab)[:300]
    _, dn, F0n, Nn = K.forward(ps, voice)
    t_start = t_speech - float(dn[0]) * FR    # keep the model's natural leading silence (onsets need it)
    male = is_male(voice)
    cap = 57 if male else 62
    shift = max(0.0, tone - cap)          # reached by WORLD (time-varying), formants kept
    hold_at = 152.7 + float(r.uniform(-0.35, 0.35))
    land = CONV_LAND + 0.25 + float(r.uniform(0.0, 0.2))    # vowel keeps sounding just past 154.5

    def stretch(t):
        return 1.0 + 2.6 * smoothstep((t - 150.25) / 2.6)
    # walk the symbols: stretched durations until hold_at, then hold the next open vowel
    d_new, t = [int(dn[0])], t_speech
    cut = None
    for j in range(1, len(dn) - 1):
        c = ps[j - 1]
        if t >= hold_at and c in VOWELS and c not in "əɚɐ":
            cut = j
            break
        dj = max(1, int(round(dn[j] * stretch(t))))
        d_new.append(dj)
        t += dj * FR
    if cut is None:             # sentence too short: hold its last vowel
        cut = max(j for j in range(1, len(dn) - 1) if ps[j - 1] in VOWELS)
        d_new = d_new[:cut]
        t = t_start + sum(d_new) * FR
    ps2 = ps[:cut]              # symbols up to and including the held vowel
    hold = max(8, int(round((land - t) / FR)))
    d_new = d_new[:cut] + [hold, 3]
    k = cut                     # duration index of the held vowel
    f_tgt_r = float(midi_hz(tone - shift + float(r.uniform(-6, 6)) / 100))
    cd = np.concatenate([[0], np.cumsum(d_new)]) * 2
    v0, v1 = int(cd[k]), int(cd[k + 1])
    n_f0 = int(cd[-1])
    tf = t_start + np.arange(n_f0) / 80.0                 # global time of each F0 frame
    w = smoothstep((tf - 150.2) / (152.9 - 150.2)) ** 1.3
    w[v0:] = np.maximum(w[v0:], smoothstep((tf[v0:] - tf[v0]) / 0.5))

    def dur_fn(d):
        return np.array(d_new)

    def f0_fn(F0, d):
        F0 = F0.copy()
        voiced = F0 > 40
        med = np.median(F0[voiced]) if voiced.any() else 150.0
        nat = np.where(voiced, F0, med)
        vib = 0.12 * np.sin(2 * np.pi * float(r.uniform(4.8, 5.8)) * (tf - tf[v0])) * np.clip((tf - tf[v0] - 0.4) / 0.6, 0, 1)
        lf = (1 - w) * np.log(nat) + w * np.log(f_tgt_r * 2 ** (vib / 12))
        out = np.where(voiced, np.exp(lf), F0)
        out[v0 + 3:v1 - 2] = np.exp(lf[v0 + 3:v1 - 2])       # the held vowel is always voiced
        return out

    def n_fn(N, d):
        pl = float(np.percentile(N[:v0], 92)) if v0 > 8 else float(N.max())
        i = np.arange(len(N))
        m = np.clip((i - (v0 + 2)) / 8, 0, 1) * np.clip((v1 - i) / 4, 0, 1)
        return (1 - m) * N + m * np.maximum(N, pl - 0.3)
    a, d, _, _ = K.forward(ps2, voice, dur_fn=dur_fn, f0_fn=f0_fn, n_fn=n_fn)
    # time-varying WORLD shift for tones above the comfortable register
    if shift > 0:
        f0, tt, sp, ap = world_analyze(a, VSR, f0_floor=55, f0_ceil=1000)
        gt = t_start + tt
        ww = np.interp(gt, tf, w)
        sp = warp_sp(sp, 1.0 + 0.1 * float(np.clip((tone - 64) / 12, 0, 1)))
        a = world_synth(f0 * 2 ** (shift * ww / 12), sp, ap, VSR)[:len(a)]
    x = up48(a)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, SR, subtype="FLOAT")
    meta.write_text(json.dumps({"t_start": t_start, "text": text, "tone": tone}))
    return x, t_start


def render_convergence():
    buf = np.zeros((NS, 2), np.float32)
    info = []
    for i, (voice, tone) in enumerate(conv_voices()):
        x, t0 = converge_voice(voice, tone, i)
        r = rng_for("convmix", voice, i)
        n = len(x)
        tg = t0 + np.arange(n) / SR
        # crescendo from a murmur under N08 to full chord on 154.5; release after the hymn's first onset
        env = db(-26 + 23 * smoothstep((tg - 150.0) / 4.4) ** 1.6)
        env *= np.clip(1 - (tg - (CONV_LAND + 0.05)) / 0.55, 0, 1) ** 1.5
        x = x * env.astype(np.float32)
        p = float(np.clip((tone - 60) / 30 * -0.9 + r.uniform(-0.35, 0.35), -0.95, 0.95))
        # voices start scattered wide and gather toward their section as they converge
        p_start = float(r.uniform(-0.95, 0.95))
        pp = p_start + (p - p_start) * smoothstep((tg - 150.5) / 3.5)
        add_at(buf, pan(x, pp), t0)
        info.append(dict(voice=voice, tone=tone, t0=round(t0, 3)))
        print(f"conv {voice} -> {tone}", flush=True)
    return buf, info


# ------------------------------------------------------------------------------------------------ main
def cathedral():
    return make_ir(rt60=3.3, predelay=0.03, lo_mult=1.15, hi_mult=0.5, er_n=18, er_span=0.1, er_gain=0.35,
                   width=1.0, build=0.05, seed=31)


def main():
    conv, conv_info = render_convergence()
    bus, checks = render_hymn()
    dry = conv.copy()
    for p in PARTS:
        dry += bus[p]
    # dry section buses (149-170 s, sample 0 = 149.0) for verification / optional rebalancing by Main
    parts_dir = VOCAL / "choir_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    a, b = int(PARTS_T0 * SR), int(170.0 * SR)
    for name, buf in [("convergence", conv)] + [(p, bus[p]) for p in PARTS]:
        sf.write(parts_dir / f"{name}.wav", buf[a:b], SR, subtype="FLOAT")
    breaths(dry)
    h = cathedral()
    wet = convolve(dry, h)[:NS]
    mix = dry * db(-3.5) + wet * db(-6.0)
    # let the tail settle by ~168.5 (Composer's lift starts 167), keep 138.5-140 silent (it is anyway)
    t = np.arange(NS) / SR
    g = np.clip(1 - (t - 166.6) / 2.2, 0, 1) ** 2
    g[t < 166.6] = 1.0
    mix *= g[:, None].astype(np.float32)
    mix[: int(149.0 * SR)] = 0
    pk = np.abs(mix).max()
    mix *= db(-3.0) / pk if pk > db(-3.0) else 1.0
    write_stem("choir", mix)
    bad = [c for c in checks if not (abs(c["median_cents"]) <= 25 and c["voiced_frac"] >= 0.85)]
    print(f"hymn notes: {len(checks)} sung, {len(bad)} outside +-25 cents / voiced<85%", flush=True)
    for c in bad:
        print("  ", c, flush=True)
    (VOCAL / "choir_notes.json").write_text(json.dumps({"convergence": conv_info, "hymn": checks}, indent=1))


if __name__ == "__main__":
    main()

"""Builds audio/stems/crowd.wav — walla & babble, all synthesized from Kokoro voices.

 s02  34.8-47.0   the slop whisper-chorus: slop phrases turned into TRUE whispers (WORLD resynthesis with
                  all-aperiodic excitation), time-smeared/formant-smeared, creeping in with the wave, a
                  reverse-swell into the crash at 39.2, then swirling around the room under N07.
 s05b 100.5-125.5 sparse background: nations-of-one declaring (open air), cabals-of-one praying & whispering
                  (temple), crackpots hurling lore into the void (echo) — low, under the main lines.
 s06  125.5-138.5 THE BABEL CACOPHONY: ~380 overlapping voice streams, all 54 Kokoro voices in 9 languages
                  shouting lore, density rising to a roar at 136.0 (+ a rising scream layer), then a razor
                  HARD STOP to digital zero at 138.5 exactly.
 s08  188.3       "GLORIOUS!" — every voice x2 takes, shouted by prosody injection, rippling outward from
                  the centre over ~0.7 s (Scene08's salute wave), open air.

usage: python tools/vocal_crowd.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Compressor, HighpassFilter, HighShelfFilter, Pedalboard, PeakFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocal_corpus import BABEL_TX, CRACKPOTS, FAITHS, LORE, NATIONS, SLOP
from vocal_lib import (ALL_VOICES, CACHE, NS, SR, TL, VOICES, VSR, add_at, bandpass, convolve, db, echo, fade, key,
                       kokoro, lufs, make_ir, pan, psola, resample, rng_for, smoothstep, up48, warp_sp,
                       trim, whisperize, world_analyze, world_stretch, world_synth, write_stem)

EN = VOICES["a"] + VOICES["b"]
HARD_STOP = 138.5


# ------------------------------------------------------------------------------------------------ material
def cached(kind, parts, fn, sr=VSR):
    path = CACHE / kind / f"{key(*parts)}.wav"
    if path.exists():
        return sf.read(path, dtype="float32")[0]
    a = np.asarray(fn(), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, a, sr, subtype="FLOAT")
    return a


def say(text, voice, speed=1.0):
    return kokoro().say(text, voice, speed)


def shouted(text, voice, speed=1.0, semis=3.0, spread=1.35):
    """PSOLA-raised, contour-expanded, driven version of an utterance (24 kHz)."""
    def fn():
        a = say(text, voice, speed)
        try:
            y = psola(a, VSR, lambda ts, hz: np.median(hz) * 2 ** (semis / 12) * (hz / np.median(hz)) ** spread,
                      floor=60, ceil=600)
        except Exception:
            y = a
        pk = np.abs(y).max() + 1e-9
        y = 0.6 * y + 0.4 * np.tanh(4 * y / pk) * pk
        return Pedalboard([HighpassFilter(140), PeakFilter(3000, 5.0, 0.9)])(y[None, :].astype(np.float32), VSR)[0]
    return cached("shout", ("shout1", text, voice, speed, semis, spread), fn)


def whisper(text, voice, speed=1.0, formant=1.0, stretch=1.0):
    return cached("whisper", ("wh1", text, voice, speed, round(formant, 3), stretch),
                  lambda: whisperize(say(text, voice, speed), VSR, formant, stretch))


def slowed(text, voice, speed=1.0, factor=0.7, stretch=1.6):
    """demonic 'slowed' voice: WORLD pitch-down + time-stretch, formants lowered a little."""
    def fn():
        f0, t, sp, ap = world_analyze(say(text, voice, speed), VSR, fast=True)
        f0, sp, ap = world_stretch(f0, sp, ap, stretch)
        return world_synth(f0 * factor, warp_sp(sp, 0.88), ap, VSR)
    return cached("slowed", ("sl1", text, voice, speed, factor, stretch), fn)


def lore_for(voice, r):
    lang = voice[0]
    if lang in "ab":
        return LORE[int(r.integers(0, len(LORE)))]
    return BABEL_TX[lang][int(r.integers(0, len(BABEL_TX[lang])))]


def place(buf, mono48, t, p, g_db=0.0, fin=0.01, fout=0.03):
    add_at(buf, pan(fade(mono48, fin, fout) * db(g_db), p), t)


def gate_after(buf, t_end, fade_s=0.0015):
    """razor cut: raised-cosine over `fade_s` ending exactly at t_end, digital zero afterwards."""
    i1 = int(round(t_end * SR))
    nf = int(round(fade_s * SR))
    buf[i1 - nf:i1] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, nf)))[:, None].astype(np.float32)
    buf[i1:] = 0


def section_norm(buf, t0, t1, target):
    L = lufs(buf[int(t0 * SR):int(t1 * SR)])
    if np.isfinite(L):
        buf *= db(target - L)
    return buf


# ------------------------------------------------------------------------------------------------ s02
def slop_whispers():
    buf = np.zeros((NS, 2), np.float32)
    r = rng_for("slop")
    events = []
    t = 34.8
    while t < 39.1:                       # creeping build with the wave
        u = (t - 34.8) / 4.4
        t += float(r.exponential(1.0 / (1.6 + 13 * u ** 2)))
        pool = SLOP[:8] if (t < 37.0 or r.random() < 0.6) else SLOP
        events.append(dict(t=t, ph=pool[int(r.integers(0, len(pool)))], g=-24 + 21 * u ** 1.4,
                           p=float(r.uniform(-0.9, 0.9)), mv=0.0))
    for k in range(40):                   # the crash: everyone at once
        events.append(dict(t=39.2 + float(r.uniform(-0.02, 0.07)), ph=SLOP[int(r.integers(0, len(SLOP)))],
                           g=float(r.uniform(-4, 1)), p=float(r.uniform(-1, 1)), mv=float(r.uniform(-1, 1))))
    t = 39.5
    while t < 46.6:                       # swirling under N07
        u = (t - 39.5) / 7.1
        t += float(r.exponential(1.0 / (6.0 - 3.0 * u)))
        events.append(dict(t=t, ph=SLOP[int(r.integers(0, len(SLOP)))], g=-12 - 8 * u,
                           p=float(r.uniform(-1, 1)), mv=float(r.choice([-1, 1]) * r.uniform(0.4, 1.0))))
    crash = np.zeros((NS, 2), np.float32)
    for i, e in enumerate(events):
        v = EN[int(r.integers(0, len(EN)))]
        sp = float(r.choice([0.9, 1.0, 1.1]))
        fm = float(r.choice([0.86, 0.93, 1.0, 1.0, 1.08, 1.16]))
        st = float(r.choice([1.0, 1.0, 1.0, 1.35, 1.8, 2.6]))
        x = up48(whisper(e["ph"], v, sp, fm, st))
        if r.random() < 0.1:
            x = x[::-1].copy()            # a few backwards whispers
        n = len(x)
        tt = e["t"] + np.arange(n) / SR
        if e["mv"]:                        # circling around the room
            p = np.sin(2 * np.pi * 0.35 * e["mv"] * (tt - 39.2) + np.arcsin(np.clip(e["p"], -1, 1)))
        else:
            p = e["p"]
        target = crash if abs(e["t"] - 39.23) < 0.08 else buf
        place(target, x, e["t"], p, e["g"])
    # a few slowed, pitched-down 'AI voice' phrases smeared under the crash and after
    for k, (ph, t) in enumerate([("as an AI language model", 39.25), ("it's important to note", 40.2),
                                 ("delve", 41.9), ("in today's fast-paced world", 43.1), ("tapestry", 45.0)]):
        v = EN[int(r.integers(0, len(EN)))]
        x = up48(slowed(ph, v, 1.0, float(r.uniform(0.62, 0.75)), float(r.uniform(1.4, 1.9))))
        place(buf, x, t, float(r.uniform(-0.6, 0.6)), -10 if k == 0 else -17)
    # reverse swell sucked into the crash
    room = make_ir(rt60=2.4, predelay=0.02, lo_mult=1.0, hi_mult=0.6, er_n=6, width=1.0, seed=41)
    i0, i1 = int(39.1 * SR), int(41.5 * SR)
    wet_crash = convolve(crash[i0:i1], room)
    rev = wet_crash[::-1][: int(1.6 * SR)]
    rev = rev * np.linspace(0, 1, len(rev))[:, None] ** 2
    add_at(buf, rev * db(-2), 39.2 - len(rev) / SR)
    buf += crash
    wet = convolve(buf[int(34.0 * SR):int(47.5 * SR)], room) * db(-7)
    add_at(buf, wet, 34.0)
    t = np.arange(NS) / SR
    g = np.clip((47.15 - t) / 0.6, 0, 1)            # gone as the TV static takes over (s03 at 47.5)
    buf *= g[:, None].astype(np.float32)
    buf[: int(34.6 * SR)] = 0
    return buf


# ------------------------------------------------------------------------------------------------ s05b
def murmurs():
    buf = np.zeros((NS, 2), np.float32)
    r = rng_for("murmur")
    isl = make_ir(rt60=0.7, predelay=0.012, er_n=3, width=0.9, seed=42)
    temple = make_ir(rt60=3.4, predelay=0.04, lo_mult=1.2, hi_mult=0.5, er_n=14, er_span=0.12, er_gain=0.45, seed=43)
    void = make_ir(rt60=9.0, predelay=0.09, lo_mult=1.2, hi_mult=0.7, er_n=0, build=0.15, seed=44)
    # nations of one: distant declarations across the little seas
    sec = np.zeros_like(buf)
    t = 100.9
    while t < 107.4:
        t += float(r.exponential(1 / 1.5))
        v = ALL_VOICES[int(r.integers(0, len(ALL_VOICES)))]
        txt = NATIONS[int(r.integers(0, len(NATIONS)))] if v[0] in "ab" else lore_for(v, r)
        x = up48(shouted(txt, v, float(r.uniform(0.95, 1.15)), 2.0, 1.2))
        d = float(r.uniform(2, 6))
        x = bandpass(x, 150, 7000 / d ** 0.5, order=2)
        place(sec, x, t, float(r.uniform(-1, 1)), -20 * np.log10(d))
    e = echo(sec[int(100 * SR):int(108.5 * SR)], 0.13, 0.25, n=2, lp=2500, spread=0.6)
    sec[:] = 0
    add_at(sec, e, 100.0)
    buf += reverb_add(sec, isl, -10)
    # cabals of one: murmured prayers and whispered heresies in the temple
    sec = np.zeros_like(buf)
    t = 108.0
    while t < 114.3:
        t += float(r.exponential(1 / 1.8))
        v = EN[int(r.integers(0, len(EN)))]
        txt = FAITHS[int(r.integers(0, len(FAITHS)))]
        if r.random() < 0.5:
            x = up48(whisper(txt, v, float(r.uniform(0.85, 1.0))))
            g = -2.0
        else:
            x = bandpass(up48(say(txt, v, float(r.uniform(0.8, 0.95)))), 120, 2600, order=2)
            g = -8.0
        place(sec, x, t, float(r.uniform(-1, 1)), g - float(r.uniform(0, 6)))
    buf += reverb_add(sec, temple, -3)
    # crackpots screaming lore into the void (far away), gathering into the swirl toward Babel
    sec = np.zeros_like(buf)
    t = 114.8
    while t < 125.3:
        u = smoothstep((t - 122.5) / 3.0)
        t += float(r.exponential(1 / (0.9 + 3.5 * u)))
        v = ALL_VOICES[int(r.integers(0, len(ALL_VOICES)))]
        txt = CRACKPOTS[int(r.integers(0, len(CRACKPOTS)))] if v[0] in "ab" else lore_for(v, r)
        x = up48(shouted(txt, v, float(r.uniform(1.0, 1.2)), 3.0, 1.45))
        d = float(r.uniform(2.5, 7))
        x = bandpass(x, 200, 8000 / d ** 0.5, order=2)
        place(sec, x, t, float(r.uniform(-1, 1)), -20 * np.log10(d) + 4 * u)
    e = echo(sec[int(114 * SR):int(126 * SR)], 0.33, 0.5, n=9, lp=3000, hp=250, spread=0.8, detune=0.006)
    sec[:] = 0
    add_at(sec, e, 114.0)
    buf += reverb_add(sec, void, -4)
    t = np.arange(NS) / SR
    buf *= (np.clip((t - 100.5) / 0.4, 0, 1) * np.clip((126.2 - t) / 0.7, 0, 1))[:, None].astype(np.float32)
    # keep the leads clear: -7 dB under every s05b line (K07 especially sits on the densifying swirl)
    g = np.ones(NS, np.float32)
    for ln in TL["lines"]:
        if ln["scene"] == "s05b":
            d = np.clip(np.minimum((t - ln["start"] + 0.2) / 0.15, (ln["end"] + 0.2 - t) / 0.3), 0, 1)
            g = np.minimum(g, (1 - d * (1 - db(-7))).astype(np.float32))
    buf *= g[:, None]
    return buf


def reverb_add(sec, h, wet_db):
    nz = np.flatnonzero(np.abs(sec).max(1) > 1e-7)
    if not len(nz):
        return sec
    a, b = nz[0], min(NS, nz[-1] + 1)
    out = sec.copy()
    w = convolve(sec[a:b], h) * db(wet_db)
    add_at(out, w, a / SR)
    return out


# ------------------------------------------------------------------------------------------------ s06
def babel():
    buf = np.zeros((NS, 2), np.float32)
    r = rng_for("babel")
    T_S, T_R = 125.5, 136.0
    order = list(ALL_VOICES)
    r.shuffle(order)
    # utterance library (5 per voice), plain + shouted
    lib = {}
    for v in order:
        rv = rng_for("babel_lib", v)
        texts = []
        while len(texts) < 5:
            tx = lore_for(v, rv)
            if tx not in texts or len(texts) >= len(BABEL_TX.get(v[0], LORE)):
                texts.append(tx)
        lib[v] = [(tx, float(rv.uniform(0.98, 1.2))) for tx in texts]
        print("babel lib", v, flush=True)
    N_STREAMS = 380

    def start_time(i):
        if i < 10:
            return T_S + 0.3 * i / 10
        if i < 320:
            return T_S + 10.5 * ((i - 10) / 310) ** 0.55
        return T_R + 2.3 * (i - 320) / 60
    lvl_t = np.arange(NS) / SR
    for i in range(N_STREAMS):
        v = order[i % len(order)]
        ts = start_time(i) + float(r.uniform(-0.15, 0.15))
        d = float(r.uniform(1.0, 6.0)) if i >= 10 else float(r.uniform(1.0, 2.0))
        p = float(r.uniform(-1, 1)) * (0.5 + 0.5 * min(1, d / 3))
        track = []
        t = max(T_S, ts)
        while t < HARD_STOP + 0.1:
            tx, sp = lib[v][int(r.integers(0, len(lib[v])))]
            shout_p = smoothstep((t - 129.0) / 6.5)
            a = shouted(tx, v, sp, 3.0, 1.4) if r.random() < shout_p else say(tx, v, sp)
            ratio = 1.0 + 0.11 * smoothstep((t - 134.5) / 4.0) + float(r.uniform(-0.03, 0.03))
            x = up48(a)
            if abs(ratio - 1) > 0.004:
                x = resample(x, 1000, int(round(1000 * ratio)))       # faster & higher: rising panic
            if r.random() < 0.35:                                       # cut people off mid-sentence
                x = x[: max(int(0.4 * SR), int(len(x) * r.uniform(0.3, 0.8)))]
            track.append((t, fade(x, 0.008, 0.04), float(r.uniform(-4, 0))))
            t += len(x) / SR + float(r.uniform(0.04, 0.45) * (1.2 - 0.8 * smoothstep((t - 128) / 7)))
        lp = 9500 / d ** 0.6
        for t, x, g in track:
            y = bandpass(x, 90, lp, order=1)
            add_at(buf, pan(y * db(g - 18 * np.log10(d)), p), t)
    # the scream layer: people shouting sustained vowels, rising in pitch into the crack
    scream = np.zeros_like(buf)
    words = [("nˈO", "No"), ("wˈI", "Why"), ("mˈi", "Me"), ("hˈA", "Hey"), ("stˈɑp", "Stop"), ("mˈIn", "Mine")]
    K = kokoro()
    for j in range(56):
        v = ALL_VOICES[(j * 7) % len(ALL_VOICES)]
        ps, _ = words[j % len(words)]
        rj = rng_for("scream", j)
        t0 = 135.45 + float(rj.uniform(0, 0.7))           # render starts with the model's own lead-in silence
        hold = HARD_STOP + 0.1 - t0
        rise = float(rj.uniform(3.0, 5.5))
        base = float(rj.uniform(2.0, 5.0))

        def mk(ps=ps, v=v, hold=hold, rise=rise, base=base):
            vi = next(i for i, c in enumerate(ps) if c in "OIiAɑ")
            k = vi + 1

            def dur_fn(d):
                d = d.copy()
                d[k] = max(8, int(hold / 0.025))
                return d

            def f0_fn(F0, d):
                n = len(F0)
                c = np.concatenate([[0], np.cumsum(d)]) * 2
                v0, v1 = int(c[k]), int(c[k + 1])
                med = np.median(F0[F0 > 40]) if (F0 > 40).any() else 150.0
                tt = np.arange(n) / 80.0
                tgt = med * 2 ** ((base + rise * smoothstep(tt / hold)) / 12)
                tgt *= 2 ** (0.25 * np.sin(2 * np.pi * 6.5 * tt) / 12)
                w = np.clip(F0 / 60, 0, 1)
                w[v0 + 2:v1] = 1
                return w * tgt + (1 - w) * F0

            def n_fn(N, d):
                c = np.concatenate([[0], np.cumsum(d)]) * 2
                v0, v1 = int(c[k]), int(c[k + 1])
                out = N.copy()
                out[v0 + 2:v1] = np.maximum(N[v0 + 2:v1], N.max() + 0.5)
                return out
            return K.forward(ps, v, dur_fn=dur_fn, f0_fn=f0_fn, n_fn=n_fn)[0]
        a = cached("scream", ("scr2", ps, v, round(hold, 3), round(rise, 3), round(base, 3)), mk)
        x = up48(a)
        pk = np.abs(x).max() + 1e-9
        x = 0.5 * x + 0.5 * np.tanh(3 * x / pk) * pk
        tt = np.arange(len(x)) / SR
        x = x * db(-12 + 10 * smoothstep(tt / 1.6)).astype(np.float32)
        place(scream, x, t0, float(rj.uniform(-1, 1)), float(rj.uniform(-9, -3)) - 6, fin=0.02, fout=0.001)
    buf += scream
    # overall shape: murmur -> roar; small dips under L03 / C03 so the leads stay on top
    t = lvl_t
    lvl = -27 + 21 * smoothstep((t - T_S) / (T_R - T_S)) ** 1.25 + 3 * smoothstep((t - 135.6) / 0.6)
    dip = -3.5 * (smoothstep((t - 130.0) / 0.3) * smoothstep((133.1 - t) / 0.3)
                  + smoothstep((t - 133.6) / 0.3) * smoothstep((135.8 - t) / 0.3))
    buf *= db(lvl + dip)[:, None].astype(np.float32)
    storm = make_ir(rt60=1.5, predelay=0.02, lo_mult=1.0, hi_mult=0.5, er_n=4, width=1.0, seed=45)
    seg = buf[int(125.0 * SR):int(HARD_STOP * SR) + 1]
    w = convolve(seg, storm) * db(-9)
    add_at(buf, w, 125.0)
    buf = np.asarray(Pedalboard([Compressor(-18, 2.5, 10, 150)])(buf.T.copy(), SR).T, np.float32)
    buf *= np.clip((t - 124.9) / 0.6, 0, 1)[:, None].astype(np.float32)
    gate_after(buf, HARD_STOP)
    return buf


# ------------------------------------------------------------------------------------------------ s08
def glorious():
    """'GLORIOUS!': a tight core (every voice once, +-25 ms) on 188.3 + a second take of every voice rippling
    outward to the edges of the meadow over ~0.7 s (quieter, darker, wetter)."""
    core = np.zeros((NS, 2), np.float32)
    ripple = np.zeros((NS, 2), np.float32)
    K = kokoro()
    ps = "ɡlˈɔːɹiəs!"
    k_vow = ps.index("ɔ") + 1
    T = 188.3
    for take in range(2):
        for i, v in enumerate(ALL_VOICES):
            r = rng_for("glorious2", v, take)
            speed = float(r.uniform(0.96, 1.06))
            semis = float(r.uniform(4.0, 7.0))
            vow = float(r.uniform(1.45, 1.75))
            nb = float(r.uniform(0.4, 0.8))

            def mk(v=v, speed=speed, semis=semis, vow=vow, nb=nb):
                def dur_fn(d):          # draw out the stressed vowel: GLOOO-rious
                    d = d.copy()
                    d[k_vow] = max(2, int(round(d[k_vow] * vow)))
                    return d

                def f0_fn(F0, d):       # shouted: higher, wider contour
                    vz = F0 > 40
                    med = np.median(F0[vz]) if vz.any() else 150.0
                    return np.where(vz, med * 2 ** (semis / 12) * (np.maximum(F0, 1) / med) ** 1.25, F0)

                def n_fn(N, d):         # more vocal effort
                    return N + nb * (N > -3)
                return K.forward(ps, v, speed, dur_fn=dur_fn, f0_fn=f0_fn, n_fn=n_fn)[0]
            a = trim(cached("glorious", ("gl2", v, take, speed, semis, vow, nb), mk), 2e-3, 0.01)
            x = up48(a)
            pk = np.abs(x).max() + 1e-9
            x = 0.7 * x + 0.3 * np.tanh(3.0 * x / pk) * pk
            x = Pedalboard([HighpassFilter(120), PeakFilter(2900, 3.5, 0.9), HighShelfFilter(7000, 2.0, 0.7)])(
                x[None, :].astype(np.float32), SR)[0]
            e = np.abs(x)
            on = int(np.flatnonzero(e > e.max() * 0.1)[0]) / SR
            if take == 0:
                p = float(r.uniform(-0.6, 0.6))
                place(core, x, T + float(r.uniform(-0.02, 0.025)) - on, p, float(r.uniform(-2.5, 0)),
                      fin=0.003, fout=0.05)
            else:
                s = float(r.choice([-1, 1]))
                dist = float(r.uniform(0.35, 1.0))
                delay = 0.12 + 0.6 * (dist - 0.35) / 0.65 + float(r.uniform(-0.03, 0.04))
                x = bandpass(x, None, 10000 - 5000 * dist, order=1)
                place(ripple, x, T + delay - on, s * dist, -4.0 - 6.0 * dist - float(r.uniform(0, 2)),
                      fin=0.003, fout=0.05)
    meadow = make_ir(rt60=1.7, predelay=0.015, lo_mult=0.9, hi_mult=0.55, er_n=6, er_span=0.05, width=1.0, seed=46)
    buf = core + ripple
    a0 = int(187.8 * SR)
    add_at(buf, convolve(core[a0:int(191.0 * SR)], meadow) * db(-11), 187.8)
    add_at(buf, convolve(ripple[a0:int(191.5 * SR)], meadow) * db(-5), 187.8)
    seg = core[a0:int(191.0 * SR)] + ripple[a0:int(191.0 * SR)]
    hills = echo(seg, 0.42, 0.28, n=3, lp=2200, spread=0.7)     # the salute comes back off the hills
    hills[:len(seg)] -= seg
    add_at(buf, hills * db(-6), 187.8)
    t = np.arange(NS) / SR
    buf *= np.clip((193.0 - t) / 1.0, 0, 1)[:, None].astype(np.float32)
    return buf


# ------------------------------------------------------------------------------------------------ main
def main():
    parts = {}
    parts["slop"] = section_norm(slop_whispers(), 34.8, 47.0, -25.0)
    print("s02 slop whispers done", flush=True)
    parts["murmur"] = section_norm(murmurs(), 100.5, 125.5, -35.0)
    print("s05b murmurs done", flush=True)
    parts["babel"] = section_norm(babel(), 125.5, HARD_STOP, -21.0)
    print("s06 babel done", flush=True)
    parts["glorious"] = section_norm(glorious(), 188.2, 190.5, -18.0)
    print("s08 glorious done", flush=True)
    mix = sum(parts.values())
    i0, i1 = int(HARD_STOP * SR), int(140.0 * SR)
    mix[i0:i1] = 0                          # the sacred silence (babel is already razor-cut at 138.5)
    assert np.abs(mix[i0:i1]).max() == 0.0
    pk = np.abs(mix).max()
    if pk > db(-1.5):
        mix *= db(-1.5) / pk
    write_stem("crowd", mix)
    for name, (a, b) in {"slop": (34.8, 47.0), "murmur": (100.5, 125.5), "babel": (125.5, HARD_STOP),
                         "glorious": (188.2, 190.5)}.items():
        print(f"  {name:9s} {a:6.1f}-{b:6.1f}  {lufs(mix[int(a*SR):int(b*SR)]):6.1f} LUFS", flush=True)


if __name__ == "__main__":
    main()

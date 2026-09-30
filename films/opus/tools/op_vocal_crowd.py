"""OPVS SCHISMATICVM — builds films/opus/audio/stems/crowd.wav: walla & babble, all synthesized from Kokoro voices.
Owner: Vocalist-2.

 m02   22.0-34.2   THE PROCESSION: far down the nave a leader intones a litany (recto tono on A3) and the procession
                   answers "ora pro nobis" in murmured unison on D (sixteen voices in lock-step, +-50 ms), under a
                   bed of the machines' own processions murmuring their creeds (nations, faiths, philosophies, in
                   English and Latin); all very far, deep in the basilica; gone before "Then came the slop."
 m02   35.6-48.3   THE SLOP: one whisper out of the first flipped tessera (35.65), a quick creep and a reverse swell
                   into slop_crash 36.3 (forty whispers at once + a slowed "as an AI language model"), then whispers
                   circling the room under N07 (true whispers: WORLD resynthesis with all-aperiodic excitation,
                   time/formant smeared, some backwards); gone by 48.3 (m03 opens on static).
 m06  106.9-113.4  nations of one: distant declarations across the little seas of the broken mappa mundi.
      113.7-120.0  THE HERETICS' COUNCIL under X03-X05: anathemas in Latin and English hurled between the churches
                   of one, murmured private creeds, whispered heresies (little chapels inside the great church).
      120.0-131.5  crackpots screaming lore into the void (echoes), gathering into the swirl that becomes Babel.
 m07  131.5-145.0  BABEL: ~400 voice streams, all 54 Kokoro voices in 9 languages (+ Latin and Church Greek), lore
                   and heresies, density and shouting rising; dips under L03 and C04; at the collapse (141.8) the
                   whole city screams (sustained vowels rising in pitch) as the mosaic falls off the wall; razor
                   HARD STOP at 145.000 (digital zero 145.0-146.5).
 m10  202.9        "GLORIOUS!": the congregation of the basilica — every voice once, vowel onsets on 202.90 (+-25 ms),
                   a second rank rippling out to ~203.5, the basilica answering; gone by 204.45 (A02 at 204.6).

usage: python films/opus/tools/op_vocal_crowd.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pedalboard import Compressor, HighpassFilter, HighShelfFilter, Pedalboard, PeakFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import op_vocal_corpus as CO  # noqa: E402
from op_vocal_lib import (ALL_VOICES, NS, SR, TL, VOICES, VSR, add_at, bandpass, cached_wav, convolve, db,  # noqa: E402
                          echo, fade, gate_after, hall_ir, kokoro, lufs, pan, place, psola, ref_durations, resample,
                          reverb_add, rng_for, room, section_norm, silence, smoothstep, sync_voice, trim, up48,
                          vocab_ps, write_stem)
from vocal_crowd import lore_for, say, shouted, slowed, whisper  # noqa: E402  (film 1's cached helpers)

EN = VOICES["a"] + VOICES["b"]
HARD_STOP = 145.0
T_COLLAPSE = 145.0 - 3.2          # 141.8


def say_lang(text, voice, lang, speed=1.0):
    """Kokoro read of `text` through the `lang` G2P by any voice pack (Latin through Italian), 24 kHz, trimmed."""
    def fn():
        K = kokoro()
        ps = vocab_ps(K.phonemes(text, lang))
        a, _, _, _ = K.forward(ps, voice, speed)
        return trim(a)
    return cached_wav("saylang", ("sl1", text, voice, lang, speed), fn)


def line_mask(t, ids, before=0.2, after=0.3, depth=-5.0, ramp=(0.15, 0.3)):
    """gain curve dipping `depth` dB under the given dialogue lines"""
    g = np.ones(len(t), np.float32)
    for ln in TL["lines"]:
        if ln["id"] in ids:
            d = np.clip(np.minimum((t - ln["start"] + before) / ramp[0], (ln["end"] + after - t) / ramp[1]), 0, 1)
            g = np.minimum(g, (1 - d * (1 - db(depth))).astype(np.float32))
    return g


# ------------------------------------------------------------------------------------------------ m02 procession
def procession():
    buf = np.zeros((NS, 2), np.float32)
    r = rng_for("procession")
    # the litany: leader far down the nave, recto tono on A3; the procession answers on D in lock-step unison
    K = kokoro()
    resp_ps = vocab_ps(K.phonemes(CO.LITANY_RESP, "i"))
    rdur, _, _ = ref_durations(resp_ps, "im_nicola", 0.9)
    crowd_v = ["am_adam", "am_michael", "bm_george", "em_alex", "pm_alex", "im_nicola", "am_echo", "bm_daniel",
               "af_heart", "bf_emma", "if_sara", "ef_dora", "pf_dora", "af_sarah", "bf_alice", "ff_siwis"]
    lead_t = [22.35, 24.95, 27.55, 30.15, 32.6]
    for k, t0 in enumerate(lead_t):
        inv = CO.LITANY_LEAD[k % len(CO.LITANY_LEAD)]
        ps = vocab_ps(K.phonemes(inv, "i"))
        d, _, _ = ref_durations(ps, "im_nicola", 0.9)
        a = up48(sync_voice(ps, "im_nicola", d, f0_mode="mono", midi=57.0, seed=k))
        a = bandpass(trim(a, 1e-3, 0.02, SR), 120, 3200, order=2)
        place(buf, a, t0, -0.15, -3.0)
        t_resp = t0 + len(a) / SR + 0.12
        for j, v in enumerate(crowd_v):
            rj = rng_for("resp", k, v)
            midi = 50.0 if v[1] == "m" else 62.0
            if j % 5 == 4:
                midi -= 12.0 if v[1] == "m" else 0.0
            b = sync_voice(resp_ps, v, rdur, f0_mode="mono", midi=midi + float(rj.uniform(-0.08, 0.08)), seed=j,
                           energy=-0.4)
            b = bandpass(trim(up48(b), 1e-3, 0.02, SR), 110, 2600, order=2)
            place(buf, b, t_resp + float(rj.normal(0, 0.045)), float(rj.uniform(-0.8, 0.8)),
                  -5.0 + float(rj.uniform(-3, 1)))
    # the machines' processions murmuring their creeds, underneath
    t = 22.2
    while t < 33.6:
        t += float(r.exponential(1 / 2.6))
        if r.random() < 0.55:
            v = EN[int(r.integers(0, len(EN)))]
            a = say(CO.MACHINE_MURMUR["a"][int(r.integers(0, len(CO.MACHINE_MURMUR["a"])))], v,
                    float(r.uniform(0.8, 0.95)))
        else:
            v = ALL_VOICES[int(r.integers(0, len(ALL_VOICES)))]
            a = say_lang(CO.MACHINE_MURMUR["i"][int(r.integers(0, len(CO.MACHINE_MURMUR["i"])))], v, "i",
                         float(r.uniform(0.8, 0.95)))
        x = bandpass(up48(a), 120, 1700, order=2)
        place(buf, x, t, float(r.uniform(-1, 1)), -12.0 - float(r.uniform(0, 6)))
    h, _ = room("basilica")
    buf = reverb_add(buf, h, 4.0)                       # far down the nave: mostly church
    tt = np.arange(NS) / SR
    buf *= (np.clip((tt - 22.0) / 1.6, 0, 1) * np.clip((34.15 - tt) / 1.1, 0, 1))[:, None].astype(np.float32)
    return buf


# ------------------------------------------------------------------------------------------------ m02 slop
T_CRASH = 36.3


def slop_whispers():
    buf = np.zeros((NS, 2), np.float32)
    crash = np.zeros((NS, 2), np.float32)
    r = rng_for("opus_slop")
    events = [dict(t=35.65, ph="you won't believe", g=-6.0, p=0.0, mv=0.0)]      # the first flipped tessera
    t = 35.85
    while t < T_CRASH - 0.04:                                                     # the quick creep
        u = (t - 35.85) / (T_CRASH - 35.85)
        t += float(r.exponential(1.0 / (5 + 30 * u ** 2)))
        events.append(dict(t=t, ph=CO.SLOP[int(r.integers(0, 8))], g=-16 + 13 * u ** 1.3,
                           p=float(r.uniform(-0.8, 0.8)), mv=0.0))
    for k in range(40):                                                           # the crash
        events.append(dict(t=T_CRASH + float(r.uniform(-0.02, 0.07)), ph=CO.SLOP[int(r.integers(0, len(CO.SLOP)))],
                           g=float(r.uniform(-4, 1)), p=float(r.uniform(-1, 1)), mv=float(r.uniform(-1, 1))))
    t = T_CRASH + 0.3
    while t < 47.4:                                                               # circling under N07
        u = (t - T_CRASH) / (47.4 - T_CRASH)
        t += float(r.exponential(1.0 / (6.5 - 3.5 * u)))
        events.append(dict(t=t, ph=CO.SLOP[int(r.integers(0, len(CO.SLOP)))], g=-11 - 8 * u,
                           p=float(r.uniform(-1, 1)), mv=float(r.choice([-1, 1]) * r.uniform(0.4, 1.0))))
    for e in events:
        v = EN[int(r.integers(0, len(EN)))]
        sp = float(r.choice([0.9, 1.0, 1.1]))
        fm = float(r.choice([0.86, 0.93, 1.0, 1.0, 1.08, 1.16]))
        st = float(r.choice([1.0, 1.0, 1.0, 1.35, 1.8, 2.6])) if e["t"] > T_CRASH else 1.0
        x = up48(whisper(e["ph"], v, sp, fm, st))
        if r.random() < 0.1 and e["t"] > T_CRASH:
            x = x[::-1].copy()
        n = len(x)
        tt = e["t"] + np.arange(n) / SR
        p = np.sin(2 * np.pi * 0.35 * e["mv"] * (tt - T_CRASH) + np.arcsin(np.clip(e["p"], -1, 1))) if e["mv"] \
            else e["p"]
        place(crash if abs(e["t"] - T_CRASH - 0.025) < 0.08 else buf, x, e["t"], p, e["g"])
    for k, (ph, t) in enumerate([("as an AI language model", T_CRASH + 0.05), ("it's important to note", 37.6),
                                 ("delve", 39.4), ("in today's fast-paced world", 42.2), ("tapestry", 45.3)]):
        v = EN[int(r.integers(0, len(EN)))]
        x = up48(slowed(ph, v, 1.0, float(r.uniform(0.62, 0.75)), float(r.uniform(1.4, 1.9))))
        place(buf, x, t, float(r.uniform(-0.6, 0.6)), -10 if k == 0 else -17)
    rm = hall_ir(rt60=2.6, predelay=0.02, build=0.03, seed=411)
    i0, i1 = int((T_CRASH - 0.1) * SR), int((T_CRASH + 2.2) * SR)
    rev = convolve(crash[i0:i1], rm)[::-1][:int(0.55 * SR)]                      # reverse swell into the crash
    rev = rev * np.linspace(0, 1, len(rev))[:, None] ** 2
    add_at(buf, rev * db(-3), T_CRASH - len(rev) / SR)
    buf += crash
    add_at(buf, convolve(buf[int(35.0 * SR):int(48.6 * SR)], rm) * db(-7), 35.0)
    tt = np.arange(NS) / SR
    buf *= (np.clip((48.3 - tt) / 0.9, 0, 1) * line_mask(tt, {"N07"}, depth=-3.0))[:, None].astype(np.float32)
    buf[:int(35.5 * SR)] = 0
    return buf


# ------------------------------------------------------------------------------------------------ m06
def m06_walla():
    tt = np.arange(NS) / SR
    r = rng_for("opus_m06")
    out = np.zeros((NS, 2), np.float32)
    # nations of one: distant declarations across the little seas
    sec = np.zeros((NS, 2), np.float32)
    t = 107.2
    while t < 113.1:
        t += float(r.exponential(1 / 1.6))
        v = ALL_VOICES[int(r.integers(0, len(ALL_VOICES)))]
        txt = CO.NATIONS[int(r.integers(0, len(CO.NATIONS)))] if v[0] in "ab" else lore_for(v, r)
        x = up48(shouted(txt, v, float(r.uniform(0.95, 1.15)), 2.0, 1.2))
        d = float(r.uniform(2, 6))
        place(sec, bandpass(x, 150, 7000 / d ** 0.5, order=2), t, float(r.uniform(-1, 1)), -20 * np.log10(d))
    e = echo(sec[int(106.5 * SR):int(114.5 * SR)], 0.13, 0.25, n=2, lp=2500, spread=0.6)
    sec[:] = 0
    add_at(sec, e, 106.5)
    sec = reverb_add(sec, room("islands")[0], -8.0)
    sec *= (np.clip((tt - 106.9) / 0.5, 0, 1) * np.clip((113.9 - tt) / 0.6, 0, 1))[:, None].astype(np.float32)
    out += section_norm(sec, 107.0, 113.5, -33.0)
    # the heretics' council: anathemas hurled between the churches of one, private creeds murmured
    sec = np.zeros((NS, 2), np.float32)
    t = 113.8
    while t < 119.9:
        t += float(r.exponential(1 / 3.2))
        kind = r.random()
        if kind < 0.35:                      # a Latin anathema, shouted from some chapel
            v = ALL_VOICES[int(r.integers(0, len(ALL_VOICES)))]
            x = up48(say_lang(CO.COUNCIL_LATIN[int(r.integers(0, len(CO.COUNCIL_LATIN)))], v, "i",
                              float(r.uniform(1.0, 1.15))))
            x = psola(x, SR, lambda ts, hz: np.median(hz) * 2 ** (2.5 / 12) * (hz / np.median(hz)) ** 1.3)
            g = -4.0
        elif kind < 0.6:                     # English heresies
            v = EN[int(r.integers(0, len(EN)))]
            x = up48(shouted(CO.COUNCIL_EN[int(r.integers(0, len(CO.COUNCIL_EN)))], v, float(r.uniform(1.0, 1.15)),
                             2.5, 1.3))
            g = -5.0
        elif kind < 0.8:                     # whispered heresies
            v = EN[int(r.integers(0, len(EN)))]
            x = up48(whisper(CO.FAITHS[int(r.integers(0, len(CO.FAITHS)))], v, float(r.uniform(0.85, 1.0))))
            g = -3.0
        else:                                # murmured private creeds
            v = EN[int(r.integers(0, len(EN)))]
            x = bandpass(up48(say(CO.FAITHS[int(r.integers(0, len(CO.FAITHS)))], v, float(r.uniform(0.8, 0.95)))),
                         120, 2600, order=2)
            g = -9.0
        d = float(r.uniform(1.5, 4.5))
        place(sec, bandpass(x, 140, 8000 / d ** 0.4, order=2), t, float(r.uniform(-1, 1)), g - 14 * np.log10(d))
    h1, _ = room("chapel")
    h2, _ = room("basilica")
    sec = reverb_add(sec, h1, -10.0)
    sec = reverb_add(sec, h2, -9.0)
    sec *= (np.clip((tt - 113.7) / 0.4, 0, 1) * np.clip((120.6 - tt) / 0.8, 0, 1))[:, None].astype(np.float32)
    out += section_norm(sec, 113.8, 120.0, -28.0)
    # crackpots in the void, far, echoing; gathering into the swirl toward Babel
    sec = np.zeros((NS, 2), np.float32)
    t = 120.2
    while t < 131.3:
        u = smoothstep((t - 128.0) / 3.5)
        t += float(r.exponential(1 / (0.9 + 5.0 * u)))
        v = ALL_VOICES[int(r.integers(0, len(ALL_VOICES)))]
        txt = CO.CRACKPOTS[int(r.integers(0, len(CO.CRACKPOTS)))] if v[0] in "ab" else lore_for(v, r)
        x = up48(shouted(txt, v, float(r.uniform(1.0, 1.2)), 3.0, 1.45))
        d = float(r.uniform(2.5, 7))
        place(sec, bandpass(x, 200, 8000 / d ** 0.5, order=2), t, float(r.uniform(-1, 1)),
              -20 * np.log10(d) + 5 * u)
    e = echo(sec[int(120 * SR):int(131.6 * SR)], 0.33, 0.5, n=9, lp=3000, hp=250, spread=0.8, detune=0.006)
    sec[:] = 0
    add_at(sec, e, 120.0)
    sec = reverb_add(sec, room("void")[0], -4.0)
    sec *= (np.clip((tt - 120.0) / 0.5, 0, 1) * np.clip((131.9 - tt) / 0.5, 0, 1))[:, None].astype(np.float32)
    out += section_norm(sec, 120.2, 131.5, -31.0)
    ids = {ln["id"] for ln in TL["lines"] if ln["scene"] == "m06"}
    out *= line_mask(tt, ids, depth=-5.0)[:, None]
    return out


# ------------------------------------------------------------------------------------------------ m07 Babel
def babel():
    buf = np.zeros((NS, 2), np.float32)
    r = rng_for("opus_babel")
    T_S = 131.5
    order = list(ALL_VOICES)
    r.shuffle(order)
    lib = {}
    for v in order:
        rv = rng_for("opus_babel_lib", v)
        items = []
        while len(items) < 5:
            if v[0] == "i" and rv.random() < 0.5:
                items.append(("i", CO.BABEL_LATIN[int(rv.integers(0, len(CO.BABEL_LATIN)))]))
            elif v[0] == "e" and rv.random() < 0.35:
                items.append(("e", CO.BABEL_GREEK[int(rv.integers(0, len(CO.BABEL_GREEK)))]))
            elif v[0] in "ab" and rv.random() < 0.25:
                items.append(("a", CO.COUNCIL_EN[int(rv.integers(0, len(CO.COUNCIL_EN)))]))
            else:
                items.append((v[0], lore_for(v, rv)))
        lib[v] = [(lang, tx, float(rv.uniform(0.98, 1.2))) for lang, tx in items]
    print("babel library ready", flush=True)

    def utter(v, lang, tx, sp, shout_):
        if lang == v[0] or (lang == "a" and v[0] in "ab"):
            return shouted(tx, v, sp, 3.0, 1.4) if shout_ else say(tx, v, sp)
        a = say_lang(tx, v, lang, sp)
        if shout_:
            a = cached_wav("shoutlang", ("shl1", tx, v, lang, sp), lambda: psola(
                a, VSR, lambda ts, hz: np.median(hz) * 2 ** (3 / 12) * (hz / np.median(hz)) ** 1.4, floor=60,
                ceil=600))
        return a
    N_STREAMS = 400

    def start_time(i):
        if i < 12:
            return T_S + 0.4 * i / 12
        if i < 330:
            return T_S + 9.6 * ((i - 12) / 318) ** 0.6
        return T_COLLAPSE - 0.4 + 1.6 * (i - 330) / 70
    for i in range(N_STREAMS):
        v = order[i % len(order)]
        ts = start_time(i) + float(r.uniform(-0.15, 0.15))
        d = float(r.uniform(1.0, 6.0)) if i >= 12 else float(r.uniform(1.0, 2.0))
        p = float(r.uniform(-1, 1)) * (0.5 + 0.5 * min(1, d / 3))
        t = max(T_S, ts)
        while t < HARD_STOP + 0.1:
            lang, tx, sp = lib[v][int(r.integers(0, len(lib[v])))]
            shout_p = smoothstep((t - 134.0) / 6.5)
            x = up48(utter(v, lang, tx, sp, r.random() < shout_p))
            ratio = 1.0 + 0.1 * smoothstep((t - 139.5) / 4.0) + float(r.uniform(-0.03, 0.03))
            if abs(ratio - 1) > 0.004:
                x = resample(x, 1000, int(round(1000 * ratio)))
            if r.random() < 0.35:
                x = x[: max(int(0.4 * SR), int(len(x) * r.uniform(0.3, 0.8)))]
            y = bandpass(fade(x, 0.008, 0.04), 90, 9500 / d ** 0.6, order=1)
            add_at(buf, pan(y * db(float(r.uniform(-4, 0)) - 18 * np.log10(d)), p), t)
            t += len(x) / SR + float(r.uniform(0.04, 0.45) * (1.2 - 0.8 * smoothstep((t - 133) / 7)))
        if i % 50 == 0:
            print("babel stream", i, flush=True)
    # the collapse: the whole city screams, sustained vowels rising in pitch as the mosaic falls
    scream = np.zeros_like(buf)
    words = [("nˈO", "No"), ("wˈI", "Why"), ("mˈi", "Me"), ("hˈA", "Hey"), ("stˈɑp", "Stop"), ("mˈIn", "Mine"),
             ("ˈɑ", "Ah")]
    K = kokoro()
    for j in range(64):
        v = ALL_VOICES[(j * 7) % len(ALL_VOICES)]
        ps, _ = words[j % len(words)]
        rj = rng_for("opus_scream", j)
        t0 = T_COLLAPSE - 0.35 + float(rj.uniform(0, 0.8))
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
                tt_ = np.arange(n) / 80.0
                tgt = med * 2 ** ((base + rise * smoothstep(tt_ / hold)) / 12)
                tgt *= 2 ** (0.25 * np.sin(2 * np.pi * 6.5 * tt_) / 12)
                w = np.clip(F0 / 60, 0, 1)
                w[v0 + 2:v1] = 1
                return w * tgt + (1 - w) * F0

            def n_fn(N, d):
                c = np.concatenate([[0], np.cumsum(d)]) * 2
                v0, v1 = int(c[k]), int(c[k + 1])
                o = N.copy()
                o[v0 + 2:v1] = np.maximum(N[v0 + 2:v1], N.max() + 0.5)
                return o
            return K.forward(ps, v, dur_fn=dur_fn, f0_fn=f0_fn, n_fn=n_fn)[0]
        a = cached_wav("scream", ("oscr1", ps, v, round(hold, 3), round(rise, 3), round(base, 3)), mk)
        x = up48(a)
        pk = np.abs(x).max() + 1e-9
        x = 0.5 * x + 0.5 * np.tanh(3 * x / pk) * pk
        tt_ = np.arange(len(x)) / SR
        x = x * db(-12 + 10 * smoothstep(tt_ / 1.4)).astype(np.float32)
        place(scream, x, t0, float(rj.uniform(-1, 1)), float(rj.uniform(-9, -3)) - 5, fin=0.02, fout=0.001)
    buf += scream
    t = np.arange(NS) / SR
    lvl = -27 + 18 * smoothstep((t - T_S) / (T_COLLAPSE - T_S)) ** 1.25 + 4 * smoothstep((t - (T_COLLAPSE - 0.3)) / 0.6)
    dip = -4.0 * (smoothstep((t - 135.75) / 0.3) * smoothstep((139.0 - t) / 0.3)
                  + smoothstep((t - 139.2) / 0.3) * smoothstep((141.5 - t) / 0.3))
    buf *= db(lvl + dip)[:, None].astype(np.float32)
    h, _ = room("babel")
    seg = buf[int(131.0 * SR):int(HARD_STOP * SR) + 1]
    add_at(buf, convolve(seg, h) * db(-8), 131.0)
    buf = np.asarray(Pedalboard([Compressor(-18, 2.5, 10, 150)])(buf.T.copy(), SR).T, np.float32)
    buf *= np.clip((t - 131.3) / 0.5, 0, 1)[:, None].astype(np.float32)
    gate_after(buf, HARD_STOP)
    return buf


# ------------------------------------------------------------------------------------------------ m10 Glorious
T_GLOR = 202.9


def glorious():
    """every voice once (vowel onsets on 202.90 +-25 ms), then a second rank rippling outward, in the basilica."""
    core = np.zeros((NS, 2), np.float32)
    ripple = np.zeros((NS, 2), np.float32)
    K = kokoro()
    ps = "ɡlˈɔːɹiəs!"
    k_vow = ps.index("ɔ") + 1
    for take in range(2):
        for v in ALL_VOICES:
            r = rng_for("glorious2", v, take)
            speed = float(r.uniform(0.96, 1.06))
            semis = float(r.uniform(4.0, 7.0))
            vow = float(r.uniform(1.45, 1.75))
            nb = float(r.uniform(0.4, 0.8))

            def mk(v=v, speed=speed, semis=semis, vow=vow, nb=nb):
                def dur_fn(d):
                    d = d.copy()
                    d[k_vow] = max(2, int(round(d[k_vow] * vow)))
                    return d

                def f0_fn(F0, d):
                    vz = F0 > 40
                    med = np.median(F0[vz]) if vz.any() else 150.0
                    return np.where(vz, med * 2 ** (semis / 12) * (np.maximum(F0, 1) / med) ** 1.25, F0)

                def n_fn(N, d):
                    return N + nb * (N > -3)
                return K.forward(ps, v, speed, dur_fn=dur_fn, f0_fn=f0_fn, n_fn=n_fn)[0]
            a = trim(cached_wav("glorious", ("gl2", v, take, speed, semis, vow, nb), mk), 2e-3, 0.01)
            x = up48(a)
            pk = np.abs(x).max() + 1e-9
            x = 0.7 * x + 0.3 * np.tanh(3.0 * x / pk) * pk
            x = Pedalboard([HighpassFilter(120), PeakFilter(2900, 3.5, 0.9), HighShelfFilter(7000, 2.0, 0.7)])(
                x[None, :].astype(np.float32), SR)[0]
            e = np.abs(x)
            on = int(np.flatnonzero(e > e.max() * 0.1)[0]) / SR
            vo = on + 0.05                                   # "Gl" leads the vowel by ~50 ms
            if take == 0:
                place(core, x, T_GLOR + float(r.uniform(-0.025, 0.025)) - vo, float(r.uniform(-0.65, 0.65)),
                      float(r.uniform(-2.5, 0)), fin=0.003, fout=0.05)
            else:
                s = float(r.choice([-1, 1]))
                dist = float(r.uniform(0.35, 1.0))
                delay = 0.1 + 0.45 * (dist - 0.35) / 0.65 + float(r.uniform(-0.03, 0.04))
                x = bandpass(x, None, 10000 - 5500 * dist, order=1)
                place(ripple, x, T_GLOR + delay - vo, s * dist, -4.0 - 6.0 * dist - float(r.uniform(0, 2)),
                      fin=0.003, fout=0.05)
    buf = core + ripple
    h, _ = room("basilica")
    a0 = int(202.5 * SR)
    add_at(buf, convolve(core[a0:int(204.3 * SR)], h) * db(-7), 202.5)
    add_at(buf, convolve(ripple[a0:int(204.3 * SR)], h) * db(-3), 202.5)
    t = np.arange(NS) / SR
    g = np.maximum(np.clip((204.45 - t) / 0.55, 0, 1) ** 1.5, db(-14) * np.clip((206.6 - t) / 1.2, 0, 1))
    buf *= g[:, None].astype(np.float32)
    return buf


# ------------------------------------------------------------------------------------------------ main
def main():
    parts = {}
    parts["procession"] = section_norm(procession(), 22.0, 34.2, -38.0)
    print("m02 procession done", flush=True)
    parts["slop"] = section_norm(slop_whispers(), 35.5, 48.3, -25.0)
    print("m02 slop done", flush=True)
    parts["m06"] = m06_walla()
    print("m06 walla done", flush=True)
    parts["babel"] = section_norm(babel(), 131.5, HARD_STOP, -21.0)
    print("m07 babel done", flush=True)
    parts["glorious"] = section_norm(glorious(), 202.8, 204.45, -18.0)
    print("m10 glorious done", flush=True)
    mix = sum(parts.values())
    gate_after(mix[:int(146.5 * SR)], HARD_STOP)            # razor cut ...
    silence(mix)                                            # ... and the sacred silence
    assert np.abs(mix[int(145.0 * SR):int(146.5 * SR)]).max() == 0.0
    pk = np.abs(mix).max()
    if pk > db(-1.5):
        mix *= np.float32(db(-1.5) / pk)
    write_stem("crowd", mix)
    for name, (a, b) in {"procession": (22.0, 34.2), "slop": (35.5, 48.3), "nations": (107.0, 113.5),
                         "council": (113.8, 120.0), "crackpots": (120.2, 131.5), "babel": (131.5, HARD_STOP),
                         "roar": (141.8, HARD_STOP), "glorious": (202.8, 204.45)}.items():
        print(f"  {name:10s} {a:6.1f}-{b:6.1f}  {lufs(mix[int(a*SR):int(b*SR)]):6.1f} LUFS", flush=True)


if __name__ == "__main__":
    main()

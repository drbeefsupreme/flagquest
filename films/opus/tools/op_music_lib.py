"""OPVS SCHISMATICVM — instruments and helpers for the score (film 3, owned by Composer).

Built on the shared engine (tools/music_lib.py, tools/music_orch.py: VSCO2 sampler, SF2 parts, reverbs).
Film-3 voices defined here:
- bell()        additive church bell with the real campanology partial set (hum, prime, tierce, quint, nominal,
                superquint, octave nominal ...), doublet warble, clapper strike; peal() rings changes.
- ison          Byzantine ison: male voices made to hold one vowel (Kokoro with prosody injection, the vocal
                team's engine), staggered breathing so the drone never breaks.
- organ()       VSCO2 Rode pipe organ with registrations (flute / principal / full = 16'+8'+4'+2 2/3'+2').
- psaltery()    plucked double-course metal strings (Karplus-Strong with stiffness), the disputation's voice.
- mosaic()      a chord set as tesserae: hundreds of tiny struck notes (celesta, glock, harp, glass, psaltery).
- slop          cheap pop machinery (kick/clap/hats/supersaw/808, phone-speaker EQ, bitcrush, wow & flutter).
All times are GLOBAL seconds.
"""
import functools
import json
import os
import sys

import numpy as np
from numba import njit

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO, "tools"))
from music_lib import (DUR, FILM_ROOT, N, SF_GU, SF_MS, SR, SF2, Buf, S, _resample, butter, db, envelope,  # noqa
                       eq, fade, glass, hz, m, make_ir, noise, stereo_place, supersaw, svf, tt)
from music_orch import I, PAN, PERC, V1P, perc  # noqa: F401

MUS = os.path.join(FILM_ROOT, "audio", "music")
CACHE = os.path.join(MUS, "cache")
TLJ = json.load(open(os.path.join(FILM_ROOT, "timeline.json")))
CUE = {c["id"]: c["t"] for c in TLJ["cues"]}
LINES = {ln["id"]: ln for ln in TLJ["lines"]}


def W(lid, word, k=0):
    """start of the k-th occurrence of `word` in line lid (timeline word timings)."""
    ws = [w for w in LINES[lid]["words"] if w["w"].lower().strip(",.!?:") == word.lower()]
    return ws[k]["s"]


def scene_events(scene, kind=None, contains=None, fallback=()):
    """visual hits exported by scene owners (films/opus/cache/foley/<scene>_*events.json), sorted times."""
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
            ts += [float(e["t"]) for e in ev if (kind is None or e.get("kind") == kind)
                   and (contains is None or contains.lower() in str(e.get("desc", "")).lower())]
    return sorted(set(ts)) if ts else list(fallback)


def hymn_grid():
    p = os.path.join(FILM_ROOT, "audio", "vocal", "hymn_grid.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def place(b, bus, y, t, gain=1.0, pan=None, width=1.0):
    """add a mono or stereo clip at global time t (optionally re-panned)."""
    if y.ndim == 1:
        y = np.stack([y, y], 1)
    if pan is not None:
        y = stereo_place(y, pan, width)
    b.add(bus, y.astype(np.float32), t=t, gain=gain)


# ------------------------------------------------------------------------------------------------ bells
# (ratio to the strike note, relative amplitude, T60 in seconds for a ~1 t bell whose strike note is D3)
BELL_PARTIALS = [(0.5, 0.36, 30.0), (1.0, 0.42, 22.0), (1.2, 0.55, 16.0), (1.5, 0.22, 11.0), (2.0, 1.00, 11.0),
                 (2.5, 0.30, 6.5), (2.67, 0.24, 5.5), (3.0, 0.42, 5.0), (3.36, 0.12, 3.6), (4.0, 0.30, 3.2),
                 (4.24, 0.10, 2.4), (4.53, 0.14, 2.2), (5.0, 0.16, 1.9), (5.34, 0.09, 1.5), (5.95, 0.10, 1.2),
                 (6.47, 0.07, 1.0), (7.02, 0.07, 0.8), (7.94, 0.05, 0.6), (8.81, 0.04, 0.5), (9.97, 0.03, 0.4)]


def bell(note, dur=None, strike=1.0, seed=0, size=1.0, bright=1.0, hum=1.0):
    """additive church bell (stereo float32). note = strike note (midi or name). size scales decay times
    (bigger bell = longer), bright scales the upper partials; strike 0..1 = clapper force."""
    rng = np.random.default_rng(seed)
    f = float(hz(m(note)))
    scale = size * (hz(m("D3")) / f) ** 0.55
    if dur is None:
        dur = min(45.0, 1.25 * 42.0 * scale)
    n = int(dur * SR)
    t = np.arange(n) / SR
    L = np.zeros(n)
    R = np.zeros(n)
    for j, (r, a, t60) in enumerate(BELL_PARTIALS):
        fr = f * r * (1 + rng.normal(0, 0.0015))
        if fr > 17000:
            continue
        amp = a * (hum if r == 0.5 else 1.0) * (bright ** max(0.0, np.log2(r) - 1.0) if r > 2.0 else 1.0)
        amp *= 0.55 + 0.45 * strike ** (0.3 + 0.35 * max(0, r - 2))  # harder strike = brighter
        k = 6.91 / (t60 * scale)
        env = np.exp(-k * t)
        # doublet: two nearly-degenerate modes beat slowly (the bell's warble)
        split = rng.uniform(0.15, 1.2) * (1 + 0.25 * r)
        w2 = rng.uniform(0.25, 0.8)
        ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
        x1 = np.sin(2 * np.pi * fr * t + ph1)
        x2 = np.sin(2 * np.pi * (fr + split) * t + ph2)
        pan = rng.uniform(-0.35, 0.35)
        s = amp * env * (x1 + w2 * x2) / (1 + w2)
        L += s * (1 - pan)
        R += s * (1 + pan)
    # clapper: a short metallic clank (dense inharmonic modes + filtered noise, ~60 ms)
    na = min(n, S(0.25))
    ta = t[:na]
    clank = np.zeros(na)
    for q in range(14):
        fr = f * rng.uniform(6.0, 28.0)
        if fr > 16000:
            continue
        clank += rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * fr * ta + rng.uniform(0, 6.28)) * np.exp(-ta / rng.uniform(0.01, 0.05))
    nz = butter(rng.standard_normal(na).astype(np.float32), "bandpass", [900, 7000], 2) * np.exp(-ta / 0.012)
    clank = (0.05 * clank / 14 + 0.035 * nz) * strike
    L[:na] += clank
    R[:na] += clank
    y = np.stack([L, R], 1)
    att = min(n, S(0.0015))
    y[:att] *= np.linspace(0, 1, att)[:, None]
    nf = min(n // 3, S(1.5))  # truncated ring: fade the last 1.5 s
    y[n - nf:] *= (np.cos(np.linspace(0, np.pi / 2, nf)) ** 2)[:, None]
    y /= (np.abs(y).max() + 1e-9)
    return (y * (0.35 + 0.65 * strike)).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _bell_cached(note, dur, strike, seed, size, bright, hum):
    return bell(note, dur, strike, seed, size, bright, hum)


def bell_to(b, bus, t, note, gain=1.0, pan=0.0, width=0.8, **kw):
    y = _bell_cached(note, kw.get("dur"), kw.get("strike", 1.0), kw.get("seed", 0), kw.get("size", 1.0),
                     kw.get("bright", 1.0), kw.get("hum", 1.0))
    b.add(bus, stereo_place(y, pan, width), t=t, gain=gain)
    return y


def peal(b, bus, t0, t1, notes, interval=0.23, gain=1.0, seed=0, method="rounds", size=0.55, fade_last=True):
    """change ringing: the bells (highest = treble first) ring in rows; each row permutes the order
    (plain hunt on the row) — rounds dissolve into changes. Returns the list of strike times."""
    order = list(range(len(notes)))
    t, row, strikes = t0, 0, []
    rng = np.random.default_rng(seed)
    while t < t1:
        for k, i in enumerate(order):
            tj = t + k * interval + rng.normal(0, 0.012)
            if tj >= t1:
                break
            pan = -0.7 + 1.4 * i / max(1, len(notes) - 1)
            bell_to(b, bus, tj, notes[i], gain=gain * (0.85 + 0.3 * rng.random()), pan=pan, seed=seed + 31 * i,
                    size=size, strike=0.75 + 0.25 * rng.random())
            strikes.append(tj)
        t += len(order) * interval + interval  # handstroke gap
        row += 1
        if method != "rounds" and row >= 1:  # plain hunt: swap pairs alternately
            st = (row + 1) % 2
            for k in range(st, len(order) - 1, 2):
                order[k], order[k + 1] = order[k + 1], order[k]
    return strikes


# ------------------------------------------------------------------------------------------------ psaltery
@njit(cache=True, fastmath=True)
def _ks(exc, delay, g, c, ap, n):
    """Karplus-Strong string: fractional delay (linear interp), loop lowpass c, gain g, one allpass for
    stiffness (inharmonic stretch)."""
    L = int(delay) + 3
    buf = np.zeros(L)
    out = np.zeros(n)
    w = 0
    lp = 0.0
    apx = 0.0
    apy = 0.0
    fr = delay - int(delay)
    for i in range(n):
        r0 = (w - int(delay)) % L
        r1 = (r0 - 1) % L
        y = buf[r0] * (1 - fr) + buf[r1] * fr
        lp = (1 - c) * y + c * lp
        v = ap * lp + apx - ap * apy  # first-order allpass (dispersion)
        apx = lp
        apy = v
        x = exc[i] if i < len(exc) else 0.0
        buf[w] = x + g * v
        out[i] = y
        w = (w + 1) % L
    return out


def psaltery(note, dur=3.0, vel=0.7, seed=0, bright=0.6, courses=2):
    """plucked metal-strung psaltery (quill): double courses slightly detuned, soundboard colour."""
    rng = np.random.default_rng(seed)
    f = float(hz(m(note)))
    n = int(dur * SR)
    out = np.zeros(n)
    for c in range(courses):
        fc = f * 2 ** (rng.normal(0, 2.5) / 1200)
        d = SR / fc
        ne = max(8, int(d))
        exc = rng.standard_normal(ne)
        pos = 0.12 + 0.05 * rng.random()  # plucking point: comb
        k = max(1, int(pos * ne))
        exc = exc - np.concatenate([np.zeros(k), exc[:-k]])
        exc = np.convolve(exc, np.hanning(7) / np.hanning(7).sum(), "same") * vel  # a quill, not a click
        ra = min(len(exc), 24)
        exc[:ra] *= np.linspace(0, 1, ra)
        t60 = 3.2 * (hz(m("A4")) / f) ** 0.4
        g = 10 ** (-3 / (t60 * fc))
        cc = 0.025 + 0.12 * (1 - bright)
        out += _ks(exc, d - 0.5, g, cc, -0.08, n)
    out /= courses
    y = eq(np.stack([out, out], 1).astype(np.float32), ("hp", 120), ("pk", 280, 3.0, 1.2), ("pk", 1100, 2.0, 1.5),
           ("pk", 3200, 2.5, 2.0), ("hs", 7000, -4))
    na = S(0.0012)
    y[:na] *= np.linspace(0, 1, na)[:, None]
    nf = min(len(y), S(0.08))
    y[-nf:] *= np.linspace(1, 0, nf)[:, None]
    return (y / (np.abs(y).max() + 1e-9) * (0.4 + 0.6 * vel)).astype(np.float32)


def psaltery_to(b, bus, t, note, dur=3.0, vel=0.7, gain=1.0, pan=0.0, seed=None):
    y = psaltery(note, dur, vel, seed=int(t * 1000) + m(note) if seed is None else seed)
    b.add(bus, stereo_place(y, pan, 0.6), t=t - 0.0015, gain=gain)


# ------------------------------------------------------------------------------------------------ ison
ISON_VOICES = ["am_adam", "bm_daniel", "am_onyx", "am_liam", "am_echo", "bm_lewis", "am_eric", "am_puck"]
_VOWELS = set("aæiɑɪAIOWYəɛɜɔuʊʌɐɒeoɚɯɨ")


def sing_vowel(ps, voice, vowel_s, midi, render_midi=None, seed=0, drift=5.0, scoop=-0.35, tail=5, life=0.35):
    """one held vowel sung by a Kokoro voice (prosody injection, as the vocal team's choir): exact pitch,
    straight tone with a slow natural drift (cents), flattened energy. Returns 24 kHz mono float32.
    render_midi: pitch Kokoro sings (comfortable register); WORLD shifts it (formants kept) to `midi`."""
    import hashlib
    import soundfile as sf
    rm = midi if render_midi is None else render_midi
    key = hashlib.sha1(json.dumps([ps, voice, round(vowel_s, 3), midi, rm, seed, drift, scoop, tail, life]).encode()).hexdigest()[:16]
    path = os.path.join(CACHE, "ison", f"{voice}_{key}.wav")
    if os.path.exists(path):
        return sf.read(path, dtype="float32")[0]
    from vocal_lib import kokoro, midi_hz, world_analyze, world_synth
    K = kokoro()
    vi = next(j for j, c in enumerate(ps) if c in _VOWELS)
    k = vi + 1
    _, dn, _, Nnat = K.forward(ps, voice)
    cn = np.concatenate([[0], np.cumsum(dn)]) * 2
    plateau = float(Nnat[int(cn[max(1, k - 2)]):int(cn[k + 1]) + 6].max()) - 0.4
    f_tgt = float(midi_hz(rm))
    rng = np.random.default_rng(seed)
    vframes = max(4, int(round(vowel_s / 0.025)))

    def dur_fn(d):
        d = np.array(d).copy()
        for j in range(1, len(d) - 1):
            if j != k:
                d[j] = 1 if ps[j - 1] in "ˈˌː" else int(np.clip(round(d[j] * 1.3), 1, 4))
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
        walk = np.cumsum(rng.normal(0, 1, n))
        walk = np.convolve(walk - np.linspace(walk[0], walk[-1], n), np.ones(61) / 61, "same")
        walk = drift * walk / (np.abs(walk).max() + 1e-9) / 100.0
        sc = scoop * np.exp(-np.maximum(tv + 0.04, 0) / 0.09)
        # a breathing singer, not an oscillator: a gentle vibrato that blooms after the first second
        rate = 4.9 + 0.5 * rng.random()
        vb = 0.2 * life * np.sin(2 * np.pi * rate * np.maximum(tv, 0) + 6.28 * rng.random()) \
            * np.clip((tv - 1.0) / 1.5, 0, 1)
        tgt = f_tgt * 2 ** ((walk + sc + vb) / 12)
        w = np.clip(F0 / 60, 0, 1)
        w[v0 + 4:v1 - 2] = 1.0
        return w * tgt + (1 - w) * F0

    def n_fn(Nn, d):
        v0, v1 = region(d)
        i = np.arange(len(Nn))
        msk = np.clip((i - (v0 + 2)) / 8, 0, 1) * np.clip((v1 - i) / 6, 0, 1)
        tgt = np.full(len(Nn), plateau)
        if life > 0 and v1 - v0 > 16:  # messa di voce + a slow wander of the breath (energy units ~5 dB each)
            x = np.arange(v1 - v0) / (v1 - v0)
            wn = np.convolve(rng.normal(0, 1, v1 - v0 + 80), np.ones(81) / 81, "valid")[:v1 - v0]
            wn = wn / (np.abs(wn).max() + 1e-9)
            tgt[v0:v1] += life * (np.sin(np.pi * x) + 0.5 * wn - 0.35)
        out = (1 - msk) * Nn + msk * np.maximum(Nn, tgt)
        a0, a1 = v0 + 2, max(v0 + 3, v1 - 3)
        sm = np.convolve(np.pad(out, 6, mode="edge"), np.ones(13) / 13, mode="valid")
        out[a0:a1] = np.maximum(sm[a0:a1], tgt[a0:a1] * msk[a0:a1] + out[a0:a1] * (1 - msk[a0:a1]))
        return out

    a, d, _, _ = K.forward(ps, voice, dur_fn=dur_fn, f0_fn=f0_fn, n_fn=n_fn)
    if rm != midi:
        f0, _, sp, ap = world_analyze(a, 24000, f0_floor=45, f0_ceil=700)
        a = world_synth(f0 * 2 ** ((midi - rm) / 12), sp, ap, 24000)[:len(a)].astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, a, 24000, subtype="FLOAT")
    return a


def up48(a):
    from scipy.signal import resample_poly
    return resample_poly(a, 2, 1).astype(np.float32)


def ison(b, bus, t0, t1, notes=("D3", "D3", "D3", "D2", "D3", "D2"), gain=1.0, fin=2.0, fout=2.0, seed=0,
         ps="nˈɔː", stagger=1.2, level=None, voices=None, pan_w=0.7):
    """a group of isokrates holding one tone t0..t1 with staggered breathing (the drone never breaks).
    notes: one pitch per singer (low singers are rendered in the tenor register and WORLD-shifted down).
    level: optional callable(global_t_array) -> gain multiplier (swells)."""
    rng = np.random.default_rng(seed)
    voices = voices or ISON_VOICES
    out = np.zeros((S(t1 - t0 + 3.0), 2), np.float32)
    for i, nt in enumerate(notes):
        mi = m(nt)
        rm = mi if mi >= 47 else mi + 12
        v = voices[i % len(voices)]
        pan = pan_w * (-1 + 2 * ((i * 3) % len(notes)) / max(1, len(notes) - 1)) if len(notes) > 1 else 0.0
        t = t0 + (rng.uniform(0, stagger) if i else 0.0)
        q = 0
        while t < t1 - 1.0:
            L = float(rng.uniform(5.5, 9.0))
            if t + L > t1 - 2.5:
                L = t1 - t + 0.4
            a = up48(sing_vowel(ps, v, L + 0.3, mi, rm, seed=seed * 97 + i * 13 + q))
            # vowel onset: soft attack; soft release (breath)
            on = int(np.argmax(np.abs(a) > np.abs(a).max() * 0.05))
            a = a[max(0, on - S(0.05)):]
            na = min(len(a), S(0.35 if q else 0.6))
            a[:na] *= np.sin(np.linspace(0, np.pi / 2, na)) ** 2
            nr = min(len(a), S(0.6))
            a[-nr:] *= np.cos(np.linspace(0, np.pi / 2, nr)) ** 2
            core = a[S(0.4):max(S(0.5), len(a) - S(0.6))]
            g = db(float(rng.uniform(-2.0, 1.0))) * 0.1 / (float(np.sqrt(np.mean(core ** 2))) + 1e-6)  # -20 dBFS
            st = stereo_place(np.stack([a, a], 1), pan, 1.0)
            i0 = S(t - t0)
            e = min(len(out), i0 + len(st))
            out[i0:e] += st[:e - i0] * g
            t += L + float(rng.uniform(0.25, 0.7))  # this singer breathes; the others hold the tone
            q += 1
    tt_ = t0 + np.arange(len(out)) / SR
    env = np.clip((tt_ - t0) / max(fin, 1e-3), 0, 1) ** 1.5 * np.clip((t1 - tt_) / max(fout, 1e-3), 0, 1) ** 1.2
    if level is not None:
        env = env * level(tt_)
    out *= env[:, None].astype(np.float32)
    out /= max(1.0, len(notes) ** 0.5)
    b.add(bus, out, t=t0, gain=gain)
    return out


# ------------------------------------------------------------------------------------------------ organ
ORG_GAIN = {"organ_loud": 0.2, "organ_soft": 0.2, "organ_pedal": 0.4, "organ_softped": 0.4}


def organ(b, t, dur, notes, reg="flute", pedal=None, gain=1.0, att=0.04, rel=0.35, pan=0.0, spread=0.5,
          bus="org", bend=None, dyn=None):
    """pipe organ chord. reg: 'flute' (quiet manual), 'principal' (open 8'), 'plenum' (8'+4'+2'),
    'full' (16'+8'+4'+2 2/3'+2'); pedal = midi of the pedal line (16' + 8' at 'full'/'plenum')."""
    notes = [m(x) for x in notes]
    ranks = {"flute": [("organ_soft", 0, 1.0)],
             "principal": [("organ_loud", 0, 1.0)],
             "plenum": [("organ_loud", 0, 1.0), ("organ_loud", 12, 0.55), ("organ_soft", 24, 0.35)],
             "full": [("organ_loud", 0, 1.0), ("organ_loud", 12, 0.6), ("organ_loud", 19, 0.28),
                      ("organ_soft", 24, 0.4), ("organ_loud", -12, 0.45)]}[reg]
    k = 0
    for j, n in enumerate(sorted(notes)):
        p = pan + spread * (j / max(1, len(notes) - 1) - 0.5)
        for (key, tr, g) in ranks:
            x = n + tr
            lo, hi = (24, 84) if key == "organ_loud" else (36, 96)
            while x > hi:
                x -= 12
            while x < lo:
                x += 12
            I(key).note(b, bus, t, x, dur, 0.8, p, gain=gain * g * ORG_GAIN[key], att=att, rel=rel, seed=k + j,
                        bend=bend, dyn=dyn)
            k += 1
    if pedal is not None:
        pk = "organ_pedal" if reg in ("principal", "plenum", "full") else "organ_softped"
        pm = m(pedal)
        lo = 24 if pk == "organ_pedal" else 36
        while pm < lo:
            pm += 12
        I(pk).note(b, bus, t, pm, dur, 0.8, pan, gain=gain * ORG_GAIN[pk], att=att, rel=rel, bend=bend, dyn=dyn)
        if reg in ("plenum", "full") and pm + 12 <= 54:
            I(pk).note(b, bus, t, pm + 12, dur, 0.8, pan, gain=gain * 0.5 * ORG_GAIN[pk], att=att, rel=rel,
                       bend=bend, dyn=dyn, seed=7)


# ------------------------------------------------------------------------------------------------ DSP helpers
def vari(y, rate):
    """vari-speed: play y with a per-sample rate curve (1 = normal); returns len(rate) samples."""
    rate = np.asarray(rate, np.float64)
    src = y if y.ndim == 2 else np.stack([y, y], 1)
    need = int(np.sum(rate)) + 8
    if need > len(src):
        src = np.concatenate([src, np.zeros((need - len(src), 2), np.float32)])
    return _resample(np.ascontiguousarray(src, np.float32), 0.0, rate, len(rate))


def wow(n, depth_cents=25.0, rate=0.6, flutter=6.0, seed=0):
    """tape wow & flutter rate curve (n samples)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    c = depth_cents * (np.sin(2 * np.pi * rate * t + rng.uniform(0, 6)) + 0.4 * np.sin(2 * np.pi * rate * 2.7 * t))
    c += flutter * np.sin(2 * np.pi * rng.uniform(5, 9) * t)
    return 2 ** (c / 1200)


def crush(y, bits=8, down=4):
    """bitcrush + sample-and-hold downsampling."""
    q = 2 ** (bits - 1)
    z = np.round(y * q) / q
    if down > 1:
        idx = (np.arange(len(z)) // down) * down
        z = z[idx]
    return z.astype(np.float32)


def phone(y, lo=500, hi=6500, drive=1.8):
    """tiny phone speaker: band-limited, a little nasal and saturated."""
    z = eq(y, ("hp", lo), ("hp", lo * 0.9), ("lp", hi), ("pk", 2400, 5.0, 1.0), ("pk", 900, -3.0, 0.8))
    return (np.tanh(z * drive) / drive).astype(np.float32)


def rev_clip(y):
    return y[::-1].copy()


# ------------------------------------------------------------------------------------------------ mosaic
def cel_part(gain=1.6):
    """MuseScore General celesta (SF2), collect notes then .to(b, bus)."""
    return SF2(SF_MS, 0, 8, gain=gain)


def glass_note_(b, t, n, dur=2.0, gain=1.0, pan=0.0, decay=None, bus="glass", att=0.002):
    """a struck/rubbed glass tone (slightly inharmonic, gently beating)."""
    y = glass(hz(m(n)), dur, att=att, decay=decay or dur * 0.6)
    b.add(bus, stereo_place(np.stack([y, y], 1), pan, 1.0), t=t, gain=gain)


def mosaic(b, t0, t1, tones, density, lo="A4", hi="E7", inst=(("cel", 1.0), ("glass", 1.0), ("harp", 0.6),
           ("glock", 0.3)), seed=0, gain=1.0, vel=(0.25, 0.55), pan=0.8, bus="keys", cel=None, times=None,
           accent=None):
    """a chord set as tesserae: many tiny struck notes, each a chord tone in [lo, hi].
    tones(t) -> list of allowed pitch classes (or midi numbers, matched by pitch class);
    density(t) -> notes per second; pan: spread (scalar) or callable(t, rng) -> pan.
    accent(t) -> extra velocity 0..1 (glints). Returns the list of (t, midi, instrument)."""
    rng = np.random.default_rng(seed)
    lo, hi = m(lo), m(hi)
    own_cel = cel is None
    cel = cel_part() if own_cel else cel
    names = [k for k, _ in inst]
    wts = np.array([w for _, w in inst], float)
    wts /= wts.sum()
    evs = []
    t = t0
    if times is None:
        times = []
        while t < t1:
            d = max(1e-3, float(density(t)))
            t += float(rng.exponential(1.0 / d))
            if t < t1:
                times.append(t)
    for t in times:
        pcs = {x % 12 for x in tones(t)}
        cands = [x for x in range(lo, hi + 1) if x % 12 in pcs]
        if not cands:
            continue
        n = int(rng.choice(cands))
        k = names[int(rng.choice(len(names), p=wts))]
        v = float(rng.uniform(*vel)) + (float(accent(t)) if accent else 0.0)
        v = float(np.clip(v, 0.05, 1.0))
        p = float(pan(t, rng)) if callable(pan) else float(rng.uniform(-pan, pan))
        if k == "cel":
            cel.note(t, min(n, 108), 0.25, v)
        elif k == "glass":
            y = glass(hz(n), 1.6, att=0.002, decay=0.9 + 0.6 * rng.random())
            b.add("glass", stereo_place(np.stack([y, y], 1), p, 1.0), t=t, gain=gain * 0.12 * v)
        elif k == "harp":
            I("harp").note(b, bus, t, min(n, 100), None, 0.25 + 0.5 * v, p, gain=gain * 0.16, seed=int(t * 1000))
        elif k == "glock":
            if n >= 79:
                I("glock").note(b, bus, t, n, None, v, p, gain=gain * 0.035 * (0.5 + v), seed=int(t * 1000))
        elif k == "psal":
            psaltery_to(b, bus, t, n, 1.8, 0.3 + 0.6 * v, gain=gain * 0.25, pan=p)
        evs.append((t, n, k))
    if own_cel:
        cel.to(b, bus, pan=0.0, gain=gain * 0.9)
    return evs


# ------------------------------------------------------------------------------------------------ slop (pop)
def _env_exp(n, tau):
    return np.exp(-np.arange(n) / SR / tau)


def kick(vel=1.0):
    n = S(0.45)
    t = np.arange(n) / SR
    f = 46 + 110 * np.exp(-t / 0.035)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env_exp(n, 0.22)
    y[:S(0.004)] += np.random.default_rng(1).standard_normal(S(0.004)) * 0.3
    return (np.tanh(1.6 * y) * vel).astype(np.float32)


def clap(vel=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = S(0.3)
    x = rng.standard_normal(n)
    env = np.zeros(n)
    for k, d in enumerate((0.0, 0.011, 0.022)):
        i = S(d)
        env[i:] += np.exp(-np.arange(n - i) / SR / (0.007 if k < 2 else 0.09))
    y = butter(x.astype(np.float32), "bandpass", [900, 5200], 2) * env
    return (y * 0.7 * vel).astype(np.float32)


def hat(open_=False, vel=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = S(0.35 if open_ else 0.06)
    y = butter(rng.standard_normal(n).astype(np.float32), "highpass", 7500, 2) * _env_exp(n, 0.12 if open_ else 0.018)
    return (y * 0.35 * vel).astype(np.float32)


def bass808(note, dur, glide_from=None, vel=1.0):
    n = S(dur)
    t = np.arange(n) / SR
    f1 = float(hz(m(note)))
    f = np.full(n, f1)
    if glide_from is not None:
        f0 = float(hz(m(glide_from)))
        f = f1 + (f0 - f1) * np.exp(-t / 0.06)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, _env_exp(n, 0.9) * 1.2)
    y = np.tanh(2.2 * y) * vel
    k = min(n, S(0.01))
    y[-k:] *= np.linspace(1, 0, k)
    return y.astype(np.float32)


def saw_note(note, dur, cutoff=2500.0, env_amt=3000.0, dec=0.15, voices=3, spread=0.12, seed=0, sq=False):
    """plucky analog-ish synth note (filter envelope)."""
    n = S(dur)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    f = float(hz(m(note)))
    y = np.zeros(n)
    for v in range(voices):
        fv = f * 2 ** ((spread * (v - (voices - 1) / 2)) / 12)
        ph = (fv * t + rng.random()) % 1.0
        y += (np.sign(ph - 0.5) if sq else 2 * ph - 1)
    y /= voices
    fc = np.clip(cutoff + env_amt * np.exp(-t / dec), 80, 18000)
    y = svf(y.astype(np.float32), fc, 0.9, 0)
    a = np.minimum(1, t / 0.004) * np.minimum(1, (dur - t) / 0.02)
    return (y * a).astype(np.float32)


def ep_note(note, dur, bright=1.0):
    """cheap FM electric piano."""
    n = S(dur)
    t = np.arange(n) / SR
    f = float(hz(m(note)))
    idx = 1.8 * bright * np.exp(-t / 0.25)
    y = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t)) * _env_exp(n, 0.9)
    y += 0.15 * np.sin(2 * np.pi * 14 * f * t) * _env_exp(n, 0.02)  # tine
    k = min(n, S(0.02))
    y[-k:] *= np.linspace(1, 0, k)
    return (y * 0.6).astype(np.float32)


def pluck_note(note, dur, seed=0):
    """ukulele / cheap plucked string (Karplus-Strong, nylon-ish)."""
    rng = np.random.default_rng(seed)
    f = float(hz(m(note)))
    d = SR / f
    exc = rng.standard_normal(max(8, int(d)))
    y = _ks(exc, d - 0.5, 10 ** (-3 / (0.9 * f)), 0.45, 0.0, S(dur))
    k = min(len(y), S(0.02))
    y[-k:] *= np.linspace(1, 0, k)
    return (y * 0.5).astype(np.float32)


def bowl(note, dur=6.0, seed=0):
    """'healing' singing bowl (for the meditation-app feed)."""
    rng = np.random.default_rng(seed)
    n = S(dur)
    t = np.arange(n) / SR
    f = float(hz(m(note)))
    y = np.zeros(n)
    for r, a, d in ((1.0, 1.0, 5.0), (2.71, 0.5, 3.0), (5.13, 0.25, 1.5), (8.3, 0.1, 0.8)):
        y += a * np.sin(2 * np.pi * f * r * t + rng.random() * 6) * (1 + 0.3 * np.sin(2 * np.pi * 3.1 * t)) * _env_exp(n, d)
    return (y * 0.3).astype(np.float32)


SCALES = {"maj": [0, 2, 4, 5, 7, 9, 11], "min": [0, 2, 3, 5, 7, 8, 10]}


def deg(key, mode, d, octave=4):
    """scale degree d (0-based, may exceed 6) in key ('E', 'F#' ...) -> midi."""
    root = m(key + str(octave))
    sc = SCALES[mode]
    return root + 12 * (d // 7) + sc[d % 7]


def pop_feed(style, key, dur, seed=0, bpm=None):
    """one slop feed: a few bars of cheap pop in its own key and tempo (dry, full band, stereo)."""
    rng = np.random.default_rng(seed)
    mode = "min" if style in ("edm", "trap", "phonk", "hyper") else "maj"
    bpm = bpm or {"edm": 128, "trap": 142, "lofi": 82, "kids": 120, "hyper": 164, "asmr": 70, "ad": 116,
                  "phonk": 132}[style]
    bt = 60.0 / bpm
    n = S(dur)
    out = np.zeros((n + S(1.0), 2), np.float32)

    def put(y, t, g=1.0, p=0.0):
        if y.ndim == 1:
            y = np.stack([y, y], 1)
        y = stereo_place(y, p, 1.0)
        i = S(t)
        if i >= n:
            return
        e = min(len(out), i + len(y))
        out[i:e] += y[:e - i] * g

    prog = [[0, 2, 4], [5, 7, 9], [2, 4, 6], [6, 8, 10]] if mode == "min" else [[0, 2, 4], [4, 6, 8], [5, 7, 9], [3, 5, 7]]
    nbeats = int(dur / bt) + 1
    hook = [int(x) for x in rng.choice([0, 2, 4, 5, 4, 2, 7, 6], 8)]
    for k in range(nbeats):
        t = k * bt
        bar, beat = divmod(k, 4)
        ch = prog[bar % 4]
        if style in ("edm", "kids", "ad", "phonk", "hyper"):
            put(kick(0.9), t, 0.9)
        if style in ("trap", "lofi") and beat in (0, 2) and not (style == "trap" and beat == 2):
            put(kick(0.9), t + (0.03 if style == "lofi" else 0), 0.9)
        if beat in (1, 3) and style in ("edm", "kids", "ad", "lofi", "phonk", "hyper"):
            put(clap(0.8, seed=k), t + (0.04 if style == "lofi" else 0), 0.6)
        if style == "trap" and beat == 2:
            put(clap(1.0, seed=k), t, 0.7)
        # hats
        sub = 4 if style in ("trap", "hyper") else 2
        for q in range(sub):
            if style == "asmr":
                break
            roll = style == "trap" and beat == 3 and bar % 2 == 1
            for r in range(3 if roll else 1):
                put(hat(open_=(style == "edm" and q == 1), vel=0.5 + 0.4 * rng.random(), seed=k * 8 + q),
                    t + q * bt / sub + r * bt / sub / 3, 0.5, 0.3)
        # bass
        rt = deg(key, mode, ch[0], 2)
        if style in ("trap", "phonk", "hyper") and beat in (0, 2):
            put(bass808(rt, 2 * bt, glide_from=rt + 12 if rng.random() < 0.3 else None), t, 0.55)
        elif style in ("edm", "ad", "kids") and beat % 2 == 0:
            put(saw_note(rt, bt * 0.9, 300, 800, 0.08, voices=2), t + bt / 2, 0.35)
        # chords
        if beat == 0:
            notes = [deg(key, mode, d, 4) for d in ch]
            if style == "edm":
                for j, nn in enumerate(notes):
                    y = supersaw(hz(nn), 4 * bt, voices=5, spread=0.2, cutoff=2600.0, seed=k + j)
                    # sidechain pump on every beat
                    pump = 1 - 0.8 * np.exp(-((np.arange(len(y)) / SR) % bt) / 0.09)
                    put(y * pump[:, None], t, 0.12)
            elif style == "lofi":
                for j, nn in enumerate(notes + [deg(key, mode, ch[0] + 6, 4)]):
                    put(ep_note(nn, 4 * bt * 0.95, 0.6), t + 0.012 * j, 0.22, -0.2 + 0.13 * j)
            elif style == "kids":
                for s in range(4):
                    for j, nn in enumerate(notes):
                        put(pluck_note(nn, bt, seed=k * 3 + j), t + s * bt + 0.008 * j, 0.3, 0.2)
            elif style == "ad":
                for j, nn in enumerate(notes):
                    put(saw_note(nn + 12, 0.18, 900, 5000, 0.06, voices=3), t, 0.18)
                    put(saw_note(nn + 12, 0.35, 900, 5000, 0.06, voices=3), t + 1.5 * bt, 0.18)
            elif style == "asmr":
                put(bowl(deg(key, mode, ch[0], 4), 4 * bt * 1.5, seed=k), t, 0.5)
        # hook melody (8th notes, cheap lead)
        if style in ("edm", "kids", "hyper", "trap", "phonk", "ad") and bar % 2 == 1 or style == "hyper":
            for q in range(2):
                d = hook[(beat * 2 + q) % 8]
                nn = deg(key, mode, d, 5)
                tq = t + q * bt / 2
                if style == "kids":
                    y = glass(hz(nn + 12), 0.5, att=0.001, decay=0.3)
                    put(y, tq, 0.2, 0.3)
                elif style == "phonk":
                    y = ep_note(nn, bt / 2, 2.5)
                    put(np.tanh(3 * y), tq, 0.2, -0.2)
                elif style == "trap":
                    put(ep_note(nn, bt, 1.4), tq, 0.2, -0.3)
                else:
                    put(saw_note(nn + (12 if style == "hyper" else 0), bt / 2 * 0.9, 1200, 4000, 0.05, voices=2,
                                 sq=style == "hyper"), tq, 0.14, -0.1)
    if style == "lofi":  # vinyl
        cr = (rng.random(len(out)) < 0.0006) * rng.standard_normal(len(out)) * 0.3
        out += np.stack([cr, cr], 1).astype(np.float32)
        out = butter(out, "lowpass", 4200, 2)
    if style == "hyper":
        out = np.tanh(out * 2.0) / 2.0
    out = out[:n]
    k = min(n, S(0.01))
    out[:k] *= np.linspace(0, 1, k)[:, None]
    out[-k:] *= np.linspace(1, 0, k)[:, None]
    return (out / (np.sqrt(np.mean(out ** 2)) + 1e-9) * 0.1).astype(np.float32)  # -20 dBFS RMS


def ding(seed=0):
    """a phone notification: two bright FM tones."""
    a = ep_note("E6", 0.5, 1.4)
    bb = ep_note("B6", 0.9, 1.4)
    y = np.zeros(S(1.1), np.float32)
    y[:len(a)] += a
    y[S(0.11):S(0.11) + len(bb)] += bb[:len(y) - S(0.11)]
    return y

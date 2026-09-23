"""Score engine for THE UNBABELING (owned by Composer).

Building blocks used by tools/music_score.py:
- Sampler      real multisampled orchestra (VSCO2-CE, indexed by tools/music_samples.py): velocity-layer
               crossfades, dynamic swells that morph timbre, round robins, cents-corrected ET tuning,
               sample-accurate attack alignment, crossfade-looped sustains, pitch bends.
- SF2          GM SoundFont parts (GeneralUser GS / MuseScore General) rendered through tinysoundfont with
               sample-accurate event scheduling.
- synths       numpy/numba DSP: FM bells & glass, sub drones, supersaw pads, Shepard-Risset glissandi,
               noise risers, reverse swells, tape stops, glitches.
- space        synthetic multiband convolution reverbs (hall / cathedral / room / void), EQ, panning.
All times are GLOBAL seconds (T=0 = film start). SR = 48 kHz. Buffers are float32 (n, 2).
"""
import functools
import json
import os

import numpy as np
import soundfile as sf
from numba import njit
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 48000
DUR = 193.0
N = int(round(DUR * SR))  # 9,264,000
INDEX_PATH = os.path.join(ROOT, "audio", "music", "cache", "sample_index.json")
SF_GU = os.path.join(ROOT, "assets", "sf2", "GeneralUser-GS.sf2")
SF_MS = os.path.join(ROOT, "assets", "sf2", "MuseScore_General.sf2")


def hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def S(t):
    return int(round(t * SR))


def db(x):
    return 10.0 ** (x / 20.0)


NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def m(name):
    """'F#4' -> 66, 'Bb2' -> 46 (C4 = 60)."""
    if isinstance(name, (int, float, np.integer)):
        return int(name)
    n = NOTE_PC[name[0]]
    i = 1
    while i < len(name) and name[i] in "#b":
        n += 1 if name[i] == "#" else -1
        i += 1
    return 12 * (int(name[i:]) + 1) + n


def pan_gains(p):
    """constant-power pan, p in [-1, 1]"""
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.cos(a) * np.sqrt(2), np.sin(a) * np.sqrt(2)


def envelope(bp, n, t0=0.0):
    """piecewise-linear breakpoints [(t, v), ...] (seconds relative to t0) sampled at n samples."""
    bp = sorted(bp)
    ts = np.array([b[0] for b in bp], dtype=np.float64)
    vs = np.array([b[1] for b in bp], dtype=np.float64)
    t = t0 + np.arange(n) / SR
    return np.interp(t, ts, vs).astype(np.float32)


# ------------------------------------------------------------------------------------------------ buffers
class Buf:
    """Sparse global stereo buffer store: bus -> full-length float32 (N, 2) allocated on demand."""

    def __init__(self):
        self.b = {}

    def get(self, bus):
        if bus not in self.b:
            self.b[bus] = np.zeros((N, 2), np.float32)
        return self.b[bus]

    def add(self, bus, x, t=None, i0=None, gain=1.0):
        if x is None or len(x) == 0:
            return
        if x.ndim == 1:
            x = np.stack([x, x], 1)
        i0 = S(t) if i0 is None else i0
        b = self.get(bus)
        a, e = i0, i0 + len(x)
        xa = 0
        if a < 0:
            xa, a = -a, 0
        e = min(e, N)
        if e <= a:
            return
        b[a:e] += x[xa:xa + (e - a)] * np.float32(gain)


# ------------------------------------------------------------------------------------------------ samples
@functools.lru_cache(maxsize=None)
def load_wav(rel):
    p = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    return np.ascontiguousarray(x[:, :2]), sr


@functools.lru_cache(maxsize=None)
def index():
    with open(INDEX_PATH) as f:
        return json.load(f)


@njit(cache=True, fastmath=True)
def _resample(src, pos0, steps, n_out):
    """cubic-hermite resampler with per-sample step (pitch) — steps may be length 1 (constant)."""
    out = np.zeros((n_out, 2), np.float32)
    p = pos0
    ns = src.shape[0]
    const = steps.shape[0] == 1
    for i in range(n_out):
        j = int(p)
        if j + 2 >= ns:
            break
        f = p - j
        jm = j - 1 if j > 0 else 0
        for c in range(2):
            y0 = src[jm, c]
            y1 = src[j, c]
            y2 = src[j + 1, c]
            y3 = src[j + 2, c]
            c1 = 0.5 * (y2 - y0)
            c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
            c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
            out[i, c] = ((c3 * f + c2) * f + c1) * f + y1
        p += steps[0] if const else steps[i]
    return out


def _extend(src, onset, need, seed):
    """Crossfade-loop the sustain of src (n,2) so at least `need` samples exist after onset."""
    n = len(src)
    if n - onset >= need + 16:
        return src
    rng = np.random.default_rng(seed)
    a = onset + int(0.30 * (n - onset))
    b = n - int(0.08 * (n - onset))
    seg_len = max(int(0.5 * (b - a)), 4096)
    xf = min(int(0.35 * 44100), seg_len // 3)
    out = [src[:b]]
    total = b
    w = np.sin(np.linspace(0, np.pi / 2, xf))[:, None].astype(np.float32)
    while total - onset < need + 16:
        s0 = int(rng.integers(a, max(a + 1, b - seg_len)))
        seg = src[s0:s0 + seg_len].copy()
        prev = out[-1]
        tail = prev[-xf:].copy()
        out[-1] = prev[:-xf]
        seg[:xf] = tail * w[::-1] + seg[:xf] * w
        out.append(seg)
        total += seg_len - xf
    return np.concatenate(out, 0)


class Sampler:
    """Multisampled VSCO2 instrument.

    note(t, midi, dur, vel, pan) — t/dur seconds, vel 0..1 (crossfades velocity layers), dur=None for
    one-shots (ring out). dyn=[(t_rel, level0..1), ...] morphs layers + level over the note (swells).
    bend=callable(t_rel_array)->semitones for glides.
    """

    def __init__(self, key, gain=1.0, rel=0.35, att=0.0, width=1.0, transpose=0, maxshift=None):
        idx = index()[key]
        self.key, self.kind = key, idx["kind"]
        self.rows = idx["rows"]
        self.nl = max(r["layer"] for r in self.rows)
        self.gain, self.rel, self.att, self.width, self.transpose = gain, rel, att, width, transpose
        self.cal = self._calibrate()

    def _calibrate(self):
        # normalise so the loudest layer near the middle of the range sits at ~ -16 dBFS RMS (first 0.8 s)
        rows = [r for r in self.rows if r["layer"] == self.nl]
        mids = sorted(r["midi"] for r in rows)
        mid = mids[len(mids) // 2]
        r = min(rows, key=lambda r: abs(r["midi"] - mid))
        x, sr = load_wav(r["path"])
        seg = x[r["onset"]:r["onset"] + int(0.8 * sr)]
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        return db(-16) / rms

    def _pick(self, midi, layer, seed):
        rows = [r for r in self.rows if r["layer"] == layer] or self.rows
        best = min(abs(r["midi"] + r["cents"] / 100 - midi) for r in rows)
        c = [r for r in rows if abs(r["midi"] + r["cents"] / 100 - midi) - best < 0.6]
        return c[seed % len(c)]

    def _voice(self, r, midi, n_out, seed, bend=None, offset=0.0):
        x, sr = load_wav(r["path"])
        base = sr / SR * 2.0 ** ((midi - (r["midi"] + r["cents"] / 100.0)) / 12.0)
        if bend is not None:
            tt = np.arange(n_out) / SR
            steps = (base * 2.0 ** (np.asarray(bend(tt), dtype=np.float64) / 12.0)).astype(np.float64)
            need = int(np.sum(steps)) + 8
        else:
            steps = np.array([base], np.float64)
            need = int(n_out * base) + 8
        on = r["onset"] + int(offset * sr)
        if self.kind == "sus":
            x = _extend(x, on, need, seed)
        return _resample(x, float(on), steps, n_out)

    def render(self, midi, dur=None, vel=0.8, dyn=None, bend=None, rel=None, att=None, seed=0, offset=0.0):
        midi = midi + self.transpose
        rel = self.rel if rel is None else rel
        att = self.att if att is None else att
        if dur is None:  # one-shot: natural length
            r = self._pick(midi, max(1, int(round(1 + vel * (self.nl - 1)))), seed)
            x, sr = load_wav(r["path"])
            n_out = int((r["len"] - r["onset"]) * SR / sr / 2.0 ** ((midi - r["midi"]) / 12.0)) - 4
            tot = max(n_out, 64)
        else:
            tot = int((dur + rel) * SR)
        if dyn is not None:
            lv = envelope(dyn, tot)
            lo = self._voice(self._pick(midi, 1, seed), midi, tot, seed, bend, offset)
            hi = self._voice(self._pick(midi, self.nl, seed), midi, tot, seed + 1, bend, offset) if self.nl > 1 else lo
            w = np.clip(lv, 0, 1)[:, None]
            y = lo * (1 - w) * 1.6 + hi * w
            y *= (0.25 + 0.75 * lv)[:, None]
        else:
            L = 1 + vel * (self.nl - 1)
            l0, l1 = int(np.floor(L)), int(np.ceil(L))
            fr = L - l0
            y = self._voice(self._pick(midi, l0, seed), midi, tot, seed, bend, offset)
            if l1 != l0 and fr > 0.04:
                y2 = self._voice(self._pick(midi, l1, seed), midi, tot, seed + 1, bend, offset)
                y = y * (1 - fr) + y2 * fr
            y *= 0.55 + 0.45 * vel if self.nl > 1 else vel ** 1.1
        if att > 0:
            na = min(int(att * SR), tot)
            y[:na] *= (np.sin(np.linspace(0, np.pi / 2, na)) ** 2)[:, None]
        if dur is not None:
            nd = int(dur * SR)
            nr = tot - nd
            if nr > 0:
                y[nd:] *= np.exp(-np.linspace(0, 6.5, nr))[:, None] * np.linspace(1, 0, nr)[:, None] ** 0.3
        return y * np.float32(self.cal * self.gain)

    def note(self, buf, bus, t, midi, dur=None, vel=0.8, pan=0.0, gain=1.0, width=None, align=True, **kw):
        """align=True: the perceptual attack (first -12 dB point) lands on t, not the first sample."""
        y = self.render(m(midi), dur, vel, seed=kw.pop("seed", int(t * 1000) + m(midi)), **kw)
        y = stereo_place(y, pan, self.width if width is None else width)
        sh = min(flux_shift(y, 0.05), attack_shift(y, 0.03, -20.0)) if align else 0  # attacks coincide on t
        buf.add(bus, y, i0=S(t) - sh, gain=gain)
        return y


def attack_shift(y, cap=0.05, rel_db=None):
    """samples from the first sample to the note's own onset-strength peak (steepest rise of a 4 ms RMS
    envelope within the first 150 ms), capped: the audible attack — not the silent pre-onset — lands on t.
    rel_db (legacy) instead uses the first point within rel_db of the early peak."""
    a = (y[: S(0.15)] ** 2).sum(1) if y.ndim == 2 else y[: S(0.15)] ** 2
    if len(a) < S(0.01):
        return 0
    if rel_db is not None:
        r = np.sqrt(a)
        return min(int(np.argmax(r > r.max() * db(rel_db))), S(cap))
    w = S(0.004)
    env = np.sqrt(np.convolve(a, np.ones(w) / w, "full")[: len(a)])
    k = S(0.003)
    d = env[k:] - env[:-k]
    i = int(np.argmax(d)) + k // 2
    return min(i, S(cap))


def flux_shift(y, cap=0.05):
    """samples from the first sample to the note's own spectral-flux onset peak (the same measure the
    verifier uses on the mix: librosa onset strength, n_fft 1024, hop 64, centred). Aligning every note's
    flux peak to its beat makes ensemble hits land as one attack exactly on the picture."""
    import librosa
    mono = (y[: S(0.25)].mean(1) if y.ndim == 2 else y[: S(0.25)]).astype(np.float32)
    if len(mono) < 2048 or not np.any(mono):
        return 0
    env = librosa.onset.onset_strength(y=np.concatenate([np.zeros(1024, np.float32), mono]), sr=SR, n_fft=1024,
                                       hop_length=64, lag=1, max_size=1, center=True)
    k = int(np.argmax(env)) * 64 - 1024
    return int(np.clip(k, 0, S(cap)))


def stereo_place(y, pan=0.0, width=1.0):
    mid = (y[:, 0] + y[:, 1]) * 0.5
    side = (y[:, 0] - y[:, 1]) * 0.5 * width
    gl, gr = pan_gains(pan)
    return np.stack([(mid + side) * gl, (mid - side) * gr], 1).astype(np.float32)


def oneshot(buf, bus, t, rel_path, gain=1.0, pan=0.0, width=1.0, rate=1.0, trim=True, rev=False, dur=None):
    """Unpitched one-shot sample from vsco2 (path relative to assets/sf2/vsco2). Onset-aligned to t
    (rev=True: reversed so its *end* (the original attack) lands on t)."""
    x, sr = load_wav(os.path.join("assets", "sf2", "vsco2", rel_path))
    a = np.abs(x).max(1)
    on = int(np.argmax(a > a.max() * db(-40))) if trim else 0
    on = max(0, on - int(0.001 * sr))
    steps = np.array([sr / SR * rate])
    n_out = int((len(x) - on) / steps[0]) - 4
    y = _resample(x, float(on), steps, n_out)
    if dur is not None:
        n = min(len(y), int(dur * SR))
        y = y[:n] * np.linspace(1, 0, n)[:, None] ** 2
    y = stereo_place(y, pan, width)
    if rev:
        y = y[::-1].copy()
        buf.add(bus, y, i0=S(t) - len(y), gain=gain)
    else:
        buf.add(bus, y, i0=S(t) - min(flux_shift(y, 0.05), attack_shift(y, 0.03, -20.0)), gain=gain)
    return y


# ------------------------------------------------------------------------------------------------ SF2
_SF2_CACHE = {}


class SF2:
    """GM SoundFont part rendered through tinysoundfont; one part = one preset, events sample-accurate."""

    def __init__(self, path, bank, preset, gain=1.0, drums=False):
        self.path, self.bank, self.preset, self.gain, self.drums = path, bank, preset, gain, drums
        self.notes = []

    def note(self, t, midi, dur, vel=0.8):
        self.notes.append((t, m(midi), dur, vel))

    def render(self, t0=None, t1=None, cc=None):
        import tinysoundfont as tsf
        if not self.notes:
            return None, 0
        t0 = min(n[0] for n in self.notes) if t0 is None else t0
        t1 = max(n[0] + n[2] for n in self.notes) + 4.0 if t1 is None else t1
        syn = tsf.Synth(samplerate=SR, gain=0.0)
        sfid = syn.sfload(self.path)
        syn.program_select(0, sfid, self.bank, self.preset, is_drums=self.drums)
        ev = []
        for (t, k, d, v) in self.notes:
            ev.append((S(t - t0), 1, k, int(np.clip(v * 127, 1, 127))))
            ev.append((S(t + d - t0), 0, k, 0))
        for (t, ctl, val) in (cc or []):
            ev.append((S(t - t0), 2, ctl, int(val)))
        ev.sort(key=lambda e: (e[0], e[1]))
        total = S(t1 - t0)
        out = np.zeros((total, 2), np.float32)
        cur = 0
        for (pos, typ, k, v) in ev + [(total, -1, 0, 0)]:
            pos = min(pos, total)
            if pos > cur:
                blk = np.frombuffer(syn.generate(pos - cur), dtype=np.float32).reshape(-1, 2)
                out[cur:pos] = blk
                cur = pos
            if typ == 1:
                syn.noteon(0, k, v)
            elif typ == 0:
                syn.noteoff(0, k)
            elif typ == 2:
                syn.control_change(0, k, v)
        nf = min(len(out), S(0.6))
        out[-nf:] *= np.linspace(1, 0, nf)[:, None] ** 2  # no truncated tails
        return out * np.float32(self.gain), S(t0)

    def to(self, buf, bus, pan=0.0, width=1.0, gain=1.0, **kw):
        y, i0 = self.render(**kw)
        if y is not None:
            buf.add(bus, stereo_place(y, pan, width), i0=i0, gain=gain)
        return y


# ------------------------------------------------------------------------------------------------ synths
def tt(dur):
    return np.arange(int(dur * SR)) / SR


def adsr(n, a=0.01, d=0.1, s=0.7, r=0.2, dur=None):
    """n samples total; dur = gate length (s) (default n/SR - r)."""
    t = np.arange(n) / SR
    g = (n / SR - r) if dur is None else dur
    e = np.where(t < a, t / max(a, 1e-6), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-6)))
    rel = np.clip((t - g) / max(r, 1e-6), 0, 1)
    e = e * np.where(t > g, np.exp(-5 * rel) * (1 - rel), 1.0)
    return e.astype(np.float32)


def fm_bell(f, dur=4.0, ratio=3.5, index=2.0, decay=1.6, idx_decay=0.5, att=0.002, detune=0.0, bright=1.0):
    t = tt(dur)
    I = index * bright * np.exp(-t / idx_decay)
    mod = np.sin(2 * np.pi * f * ratio * t)
    y = np.sin(2 * np.pi * f * (1 + detune) * t + I * mod)
    y += 0.25 * np.sin(2 * np.pi * f * 2.0 * t + 0.5 * I * mod) * np.exp(-t / (decay * 0.4))
    env = np.exp(-t / decay) * np.clip(t / max(att, 1e-4), 0, 1)
    return (y * env).astype(np.float32)


def glass(f, dur=3.0, att=0.02, decay=2.5, beat=0.6, partials=((1, 1.0), (2.756, 0.18), (5.404, 0.06))):
    """struck/rubbed glass: slightly inharmonic partials, gentle beating."""
    t = tt(dur)
    y = np.zeros_like(t)
    for (k, a) in partials:
        y += a * (np.sin(2 * np.pi * f * k * t) + 0.6 * np.sin(2 * np.pi * (f * k + beat) * t + 1.3))
    env = np.exp(-t / decay) * (1 - np.exp(-t / max(att, 1e-4)))
    return (y * env / 1.6).astype(np.float32)


@njit(cache=True, fastmath=True)
def _saw_bank(freqs, phases0, n, sr):
    """sum of PolyBLEP saws; freqs (k, n) per-sample frequency."""
    k = freqs.shape[0]
    out = np.zeros(n, np.float32)
    for j in range(k):
        ph = phases0[j]
        for i in range(n):
            dt = freqs[j, i] / sr
            ph += dt
            if ph >= 1.0:
                ph -= 1.0
            v = 2.0 * ph - 1.0
            if ph < dt:
                x = ph / dt
                v -= x + x - x * x - 1.0
            elif ph > 1.0 - dt:
                x = (ph - 1.0) / dt
                v -= x * x + x + x + 1.0
            out[i] += v
    return out


@njit(cache=True, fastmath=True)
def _svf(x, fc, q, sr, mode):
    """time-varying state-variable filter (Chamberlin/TPT). mode 0 LP, 1 BP, 2 HP. fc per sample."""
    n = x.shape[0]
    y = np.zeros(n, np.float32)
    ic1 = 0.0
    ic2 = 0.0
    k = 1.0 / q
    for i in range(n):
        g = np.tan(np.pi * min(fc[i], 0.45 * sr) / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            y[i] = v2
        elif mode == 1:
            y[i] = v1
        else:
            y[i] = x[i] - k * v1 - v2
    return y


def svf(x, fc, q=0.707, mode=0):
    fc = np.broadcast_to(np.asarray(fc, dtype=np.float64), x.shape[:1]).copy()
    if x.ndim == 2:
        return np.stack([_svf(x[:, c].astype(np.float32), fc, q, SR, mode) for c in range(x.shape[1])], 1)
    return _svf(x.astype(np.float32), fc, q, SR, mode)


def supersaw(f, dur, voices=7, spread=0.18, cutoff=1800.0, q=0.8, drift=0.0, seed=0, stereo=True):
    """detuned saw stack (semitone spread), filtered. cutoff scalar or array (per sample). Returns (n,2)."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    outs = []
    for ch in range(2 if stereo else 1):
        det = np.linspace(-spread, spread, voices) + rng.normal(0, spread * 0.15, voices)
        fr = np.array([f * 2 ** (d / 12) * (1 + drift * np.sin(2 * np.pi * (0.1 + 0.07 * j) * t + rng.uniform(0, 6)))
                       for j, d in enumerate(det)])
        y = _saw_bank(fr.astype(np.float64), rng.uniform(0, 1, voices), n, SR) / voices
        outs.append(svf(y, cutoff, q, 0))
    y = np.stack(outs, 1) if stereo else np.stack([outs[0], outs[0]], 1)
    return y.astype(np.float32)


def sub_drone(f, dur, amp_bp=None, wobble=0.15, harm=0.25):
    t = tt(dur)
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.002 * np.sin(2 * np.pi * wobble * t))) / SR
    y = np.sin(ph) + harm * np.sin(2 * ph + 0.3) + 0.08 * np.sin(3 * ph)
    if amp_bp:
        y *= envelope(amp_bp, len(t))
    return np.stack([y, y], 1).astype(np.float32)


@njit(cache=True, fastmath=True)
def _shepard(n, sr, t0, oct_rate, fbase, n_oct, center_oct, width_oct, amp, partial_detune):
    out = np.zeros((n, 2), np.float32)
    for k in range(n_oct):
        ph = 0.0
        ph2 = 0.0
        for i in range(n):
            t = t0 + i / sr
            pos = (k + oct_rate[i]) % n_oct  # octave position 0..n_oct
            f = fbase * 2.0 ** pos
            w = np.exp(-0.5 * ((pos - center_oct) / width_oct) ** 2)
            ph += 2 * np.pi * f / sr
            ph2 += 2 * np.pi * f * (1.0 + partial_detune) / sr
            a = amp[i] * w
            out[i, 0] += a * np.sin(ph)
            out[i, 1] += a * np.sin(ph2 + 0.7)
    return out


def shepard(dur, rate_oct_s=0.08, fbase=36.7, n_oct=8, center_oct=4.0, width_oct=1.3, amp=None, detune=0.003,
            start_pos=0.0, rate_fn=None):
    """Shepard–Risset glissando: endlessly rising (rate>0) or falling. amp: per-sample array or None.
    rate_fn(t) -> octaves/sec (per-sample) allows acceleration."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    if rate_fn is not None:
        pos = start_pos + np.cumsum(rate_fn(t)) / SR
    else:
        pos = start_pos + rate_oct_s * t
    a = np.ones(n) if amp is None else np.asarray(amp, dtype=np.float64)
    return _shepard(n, SR, 0.0, pos.astype(np.float64), fbase, n_oct, center_oct, width_oct, a, detune) / 3.0


def noise(n, seed=0, color="white"):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    if color == "pink":
        b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
        x = signal.lfilter(b, a, x, axis=0).astype(np.float32) * 4
    return x


def riser(dur, f0=200.0, f1=6000.0, q=2.0, curve=2.5, seed=0, tonal=None):
    """noise riser: band-pass sweep with exponential level ramp; tonal=(m0, m1) adds a rising saw sweep."""
    n = int(dur * SR)
    p = np.linspace(0, 1, n)
    fc = f0 * (f1 / f0) ** (p ** 1.5)
    x = noise(n, seed, "pink")
    y = svf(x, fc, q, 1)
    y *= (p ** curve)[:, None]
    if tonal is not None:
        mm = tonal[0] + (tonal[1] - tonal[0]) * p ** 1.3
        fr = hz(mm)
        s = _saw_bank(np.stack([fr, fr * 1.006, fr * 0.994]).astype(np.float64), np.array([0.0, 0.3, 0.6]), n, SR) / 3
        s = svf(s, fc * 0.8, 0.9, 0) * (p ** curve)
        y += 0.5 * np.stack([s, s], 1)
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def tape_stop(y, t_stop_rel, dur_stop, end_rate=0.0):
    """slow a buffer to a halt starting at t_stop_rel (seconds within y) over dur_stop."""
    i0 = S(t_stop_rel)
    n = S(dur_stop)
    rate = np.linspace(1.0, end_rate, n) ** 1.6
    pos = i0 + np.cumsum(rate)
    tail = _resample(y.astype(np.float32), float(i0), rate.astype(np.float64), n)
    out = y.copy()
    out[i0:i0 + n] = tail[: len(out[i0:i0 + n])]
    out[i0 + n:] = 0
    out[i0:i0 + n] *= np.linspace(1, 0, min(n, len(out) - i0))[:, None] ** 0.5
    return out


def reverse_swell(y, ir):
    """classic reverse reverb: convolve, reverse; the result ENDS where y's attack was."""
    w = conv_stereo(y, ir, wet_only=True)
    return w[::-1].copy()


# ------------------------------------------------------------------------------------------------ space
def make_ir(rt60=2.6, pre=0.02, bright=1.0, er=0.35, seed=1, length=None, lo_mult=1.25, hi_mult=0.45,
            build=0.04):
    """synthetic stereo multiband exponential-decay IR (normalised to unit energy)."""
    rng = np.random.default_rng(seed)
    L = int((length or rt60 * 1.25) * SR)
    t = np.arange(L) / SR
    bands = [(20, 250, lo_mult), (250, 1000, 1.0), (1000, 3000, 0.85), (3000, 7000, 0.65 * bright + 0.1),
             (7000, 20000, hi_mult * bright)]
    ir = np.zeros((L, 2), np.float64)
    for (lo, hi, mult) in bands:
        nz = rng.standard_normal((L, 2))
        sos = signal.butter(4, [lo, min(hi, 0.45 * SR)], btype="band", fs=SR, output="sos")
        nb = signal.sosfilt(sos, nz, axis=0)
        rt = rt60 * mult
        ir += nb * (10 ** (-3 * t / rt))[:, None]
    ir *= (1 - np.exp(-t / max(build, 1e-3)))[:, None]
    # early reflections
    for k in range(14):
        d = rng.uniform(0.004, 0.09)
        g = er * rng.uniform(0.3, 1.0) * np.exp(-d / 0.06)
        ch = k % 2
        ir[int(d * SR), ch] += g * rng.choice([-1, 1])
    npre = int(pre * SR)
    ir = np.concatenate([np.zeros((npre, 2)), ir], 0)
    ir /= np.sqrt(np.sum(ir ** 2) / 2)
    return ir.astype(np.float32)


@functools.lru_cache(maxsize=None)
def IR(name):
    presets = {
        "hall": dict(rt60=2.8, pre=0.025, bright=0.9, seed=11),
        "bighall": dict(rt60=4.2, pre=0.035, bright=0.85, seed=12),
        "cathedral": dict(rt60=6.5, pre=0.045, bright=0.7, seed=13, lo_mult=1.3, hi_mult=0.35),
        "void": dict(rt60=9.0, pre=0.06, bright=1.0, seed=14, lo_mult=1.0, hi_mult=0.7, build=0.25, er=0.0),
        "room": dict(rt60=0.45, pre=0.004, bright=0.8, seed=15, er=0.6),
        "plate": dict(rt60=1.9, pre=0.01, bright=1.3, seed=16, hi_mult=0.8, build=0.005),
    }
    return make_ir(**presets[name])


def conv_stereo(x, ir, wet_only=True):
    y = np.stack([signal.oaconvolve(x[:, c], ir[:, c])[: len(x) + len(ir)] for c in range(2)], 1)
    return y.astype(np.float32)


def reverb_full(x, name, wet=0.3, cross=0.25):
    """reverb over a full-length (N,2) bus; returns wet signal cropped to N (tails past 193 s discarded)."""
    ir = IR(name)
    nz = np.flatnonzero(np.abs(x).max(1) > 1e-7)
    if len(nz) == 0:
        return np.zeros_like(x)
    a, b = nz[0], nz[-1] + 1
    seg = x[a:b]
    segx = seg + cross * seg[:, ::-1]
    y = conv_stereo(segx, ir)
    out = np.zeros_like(x)
    e = min(N, a + len(y))
    out[a:e] = y[: e - a]
    return out * np.float32(wet)


def eq(x, *filters):
    """pedalboard chain helper: eq(x, ('hp', 80), ('lp', 9000), ('ls', 200, -3), ('hs', 6000, 2),
    ('pk', 1000, -4, 1.0))"""
    import pedalboard as pb
    fx = []
    for f in filters:
        k = f[0]
        if k == "hp":
            fx.append(pb.HighpassFilter(cutoff_frequency_hz=f[1]))
        elif k == "lp":
            fx.append(pb.LowpassFilter(cutoff_frequency_hz=f[1]))
        elif k == "ls":
            fx.append(pb.LowShelfFilter(cutoff_frequency_hz=f[1], gain_db=f[2], q=0.7))
        elif k == "hs":
            fx.append(pb.HighShelfFilter(cutoff_frequency_hz=f[1], gain_db=f[2], q=0.7))
        elif k == "pk":
            fx.append(pb.PeakFilter(cutoff_frequency_hz=f[1], gain_db=f[2], q=f[3] if len(f) > 3 else 1.0))
    board = pb.Pedalboard(fx)
    return board(x.T.astype(np.float32), SR).T.copy()


def butter(x, kind, fc, order=4, zero_phase=False):
    sos = signal.butter(order, fc, btype=kind, fs=SR, output="sos")
    if zero_phase:
        return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def fade(y, fin=0.0, fout=0.0):
    y = y.copy()
    n = len(y)
    if fin > 0:
        k = min(n, S(fin))
        y[:k] *= np.sin(np.linspace(0, np.pi / 2, k))[:, None] ** 2
    if fout > 0:
        k = min(n, S(fout))
        y[n - k:] *= np.cos(np.linspace(0, np.pi / 2, k))[:, None] ** 2
    return y

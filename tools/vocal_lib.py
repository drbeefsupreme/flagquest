"""Shared toolkit for the vocal stems (dialogue / crowd / choir).  Owner: Vocalist.

* Kokoro engine with *prosody injection*: the StyleTTS2-style forward pass is re-implemented so that the
  predicted phoneme durations, the F0 curve (80 Hz frames, Hz) and the energy curve can be overridden
  before the iSTFTNet decoder runs.  That is what lets a speech model hold a vowel for three seconds and
  sing it on an exact MIDI pitch with vibrato (tools/vocal_choir.py), chant, or scream (tools/vocal_crowd.py).
* WORLD vocoder helpers (formant-preserving pitch shift, true whisper by all-aperiodic resynthesis).
* PSOLA helpers via Praat/parselmouth.
* Procedural stereo reverb impulse responses, echoes, panning, loudness, stem I/O.

Everything is deterministic (seeded) and cached under audio/vocal/cache/.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, oaconvolve, resample_poly, sosfilt

ROOT = Path(__file__).resolve().parents[1]
SR = 48000            # stem rate
VSR = 24000           # Kokoro rate
DUR = 193.0
NS = int(round(DUR * SR))   # 9,264,000
TL = json.loads((ROOT / "timeline.json").read_text())
VOCAL = ROOT / "audio/vocal"
CACHE = VOCAL / "cache"
STEMS = ROOT / "audio/stems"

# every Kokoro-82M voice, by language code (first letter of the voice id)
VOICES = {
    "a": ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
          "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
          "am_michael", "am_onyx", "am_puck", "am_santa"],
    "b": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
    "e": ["ef_dora", "em_alex", "em_santa"],
    "f": ["ff_siwis"],
    "h": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "i": ["if_sara", "im_nicola"],
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "p": ["pf_dora", "pm_alex", "pm_santa"],
    "z": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"],
}
ALL_VOICES = [v for vs in VOICES.values() for v in vs]


def is_male(voice: str) -> bool:
    return voice[1] == "m"


def line(lid: str) -> dict:
    return next(l for l in TL["lines"] if l["id"] == lid)


def cue(cid: str) -> float:
    return next(c["t"] for c in TL["cues"] if c["id"] == cid)


def key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------------------
# Kokoro with prosody injection
# ----------------------------------------------------------------------------------------------------------
class Kokoro:
    """Lazy Kokoro-82M wrapper.  `say` = ordinary TTS (cached).  `forward` = TTS with overridable
    durations (alignment frames of 25 ms), F0 (12.5 ms frames, Hz) and energy."""

    def __init__(self, threads: int = 4):
        import torch
        torch.set_num_threads(threads)
        self.torch = torch
        self._model = None
        self.pipes = {}

    @property
    def model(self):
        if self._model is None:
            from kokoro import KModel
            self._model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
        return self._model

    def pipe(self, lang: str):
        if lang not in self.pipes:
            if lang == "j":   # misaki's Japanese frontend: use the bundled unidic-lite dictionary
                import fugashi
                import unidic_lite
                d = unidic_lite.DICDIR
                if not getattr(fugashi.Tagger, "_lite", False):
                    base = fugashi.Tagger

                    class LiteTagger(base):
                        _lite = True

                        def __init__(self, arg=""):
                            super().__init__(f'-d "{d}" -r "{d}/mecabrc" ' + arg)
                    fugashi.Tagger = LiteTagger
            from kokoro import KPipeline
            self.pipes[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=self.model)
        return self.pipes[lang]

    def pack(self, voice: str):
        return self.pipe(voice[0]).load_voice(voice)

    def phonemes(self, text: str, lang: str) -> str:
        p = self.pipe(lang)
        if lang in "ab":
            _, toks = p.g2p(text)
            return p.tokens_to_ps(toks)
        ps, _ = p.g2p(text)
        return ps

    def say(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """Plain Kokoro TTS at 24 kHz (cached, leading/trailing silence trimmed)."""
        path = CACHE / "say" / f"{voice}_{key(text, voice, speed)}.wav"
        if path.exists():
            return sf.read(path, dtype="float32")[0]
        chunks = [r.audio.detach().cpu().numpy() for r in self.pipe(voice[0])(text, voice=voice, speed=speed)
                  if r.audio is not None]
        a = trim(np.concatenate(chunks).astype(np.float32)) if chunks else np.zeros(1, np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, a, VSR, subtype="FLOAT")
        return a

    def forward(self, ps: str, voice: str, speed: float = 1.0, dur_fn=None, f0_fn=None, n_fn=None):
        """Kokoro forward pass with prosody injection.
        dur_fn(d)->d'      : int frames per input symbol incl. <bos>/<eos> (1 frame = 600 samples = 25 ms)
        f0_fn(F0, d')->F0' : Hz per 12.5 ms frame (2 per alignment frame); 0 = unvoiced
        n_fn(N, d')->N'    : energy per 12.5 ms frame
        returns (audio24k, durations, F0_natural, N_natural)"""
        torch = self.torch
        m = self.model
        ids = [m.vocab[c] for c in ps if c in m.vocab]
        pack = self.pack(voice)
        ref_s = pack[min(len(ids), len(pack)) - 1]
        with torch.no_grad():
            input_ids = torch.LongTensor([[0, *ids, 0]])
            L = input_ids.shape[1]
            lens = torch.LongTensor([L])
            mask = torch.zeros(1, L, dtype=torch.bool)
            bert = m.bert(input_ids, attention_mask=(~mask).int())
            d_en = m.bert_encoder(bert).transpose(-1, -2)
            s = ref_s[:, 128:]
            d = m.predictor.text_encoder(d_en, s, lens, mask)
            x, _ = m.predictor.lstm(d)
            dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1) / speed
            dur = torch.round(dur).clamp(min=1).long().squeeze(0)
            if dur_fn is not None:
                dur = torch.as_tensor(np.maximum(1, np.asarray(dur_fn(dur.numpy().copy()))), dtype=torch.long)
            idx = torch.repeat_interleave(torch.arange(L), dur)
            aln = torch.zeros((L, idx.shape[0]))
            aln[idx, torch.arange(idx.shape[0])] = 1
            aln = aln.unsqueeze(0)
            en = d.transpose(-1, -2) @ aln
            F0, N = m.predictor.F0Ntrain(en, s)
            F0n = F0.squeeze(0).numpy().copy()
            Nn = N.squeeze(0).numpy().copy()
            dn = dur.numpy()
            if f0_fn is not None:
                F0 = torch.as_tensor(np.asarray(f0_fn(F0n.copy(), dn), np.float32)).unsqueeze(0)
            if n_fn is not None:
                N = torch.as_tensor(np.asarray(n_fn(Nn.copy(), dn), np.float32)).unsqueeze(0)
            t_en = m.text_encoder(input_ids, lens, mask)
            asr = t_en @ aln
            audio = m.decoder(asr, F0, N, ref_s[:, :128]).squeeze().numpy().astype(np.float32)
        return audio, dn, F0n, Nn


_KOKORO = None


def kokoro() -> Kokoro:
    global _KOKORO
    if _KOKORO is None:
        _KOKORO = Kokoro()
    return _KOKORO


# ----------------------------------------------------------------------------------------------------------
# basic DSP
# ----------------------------------------------------------------------------------------------------------
def trim(a, thr=1e-3, pad=0.02, sr=VSR):
    nz = np.flatnonzero(np.abs(a) > thr)
    if not len(nz):
        return a[:1]
    s = max(0, nz[0] - int(pad * sr))
    e = min(len(a), nz[-1] + int(pad * sr))
    return a[s:e]


def up48(a):
    return resample_poly(a, 2, 1, axis=0).astype(np.float32)


def resample(a, ratio_num, ratio_den):
    return resample_poly(a, ratio_num, ratio_den, axis=0).astype(np.float32)


def db(x):
    return 10 ** (x / 20.0)


def midi_hz(m):
    return 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fade(a, fin=0.005, fout=0.02, sr=SR):
    a = a.copy()
    ni, no = int(fin * sr), int(fout * sr)
    if ni:
        a[:ni] *= np.linspace(0, 1, ni)[:, None] if a.ndim == 2 else np.linspace(0, 1, ni)
    if no:
        a[-no:] *= np.linspace(1, 0, no)[:, None] if a.ndim == 2 else np.linspace(1, 0, no)
    return a


def bandpass(a, lo, hi, sr=SR, order=4):
    if lo and hi:
        sos = butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    elif lo:
        sos = butter(order, lo, btype="high", fs=sr, output="sos")
    else:
        sos = butter(order, hi, btype="low", fs=sr, output="sos")
    return sosfilt(sos, a, axis=0).astype(np.float32)


def pan(mono, x):
    """Constant-power pan, x in -1..1 (scalar or per-sample array) -> (n, 2)."""
    th = (np.clip(x, -1, 1) + 1) * np.pi / 4
    return np.stack([mono * np.cos(th), mono * np.sin(th)], 1).astype(np.float32) * np.float32(np.sqrt(2))


def stereo(a):
    return a if a.ndim == 2 else np.stack([a, a], 1)


def add_at(buf, clip, t, sr=SR):
    """Mix clip into buf starting at global time t (sample-exact, clipped to buffer)."""
    i0 = int(round(t * sr))
    j0 = max(0, -i0)
    i0 = max(0, i0)
    n = min(len(buf) - i0, len(clip) - j0)
    if n > 0:
        buf[i0:i0 + n] += clip[j0:j0 + n]


def peak_db(x):
    return 20 * np.log10(np.abs(x).max() + 1e-12)


def lufs(x, sr=SR):
    import pyloudnorm as pyln
    x = stereo(np.asarray(x, np.float64))
    if len(x) < int(0.5 * sr):
        x = np.concatenate([x, np.zeros((int(0.5 * sr) - len(x), 2))])
    return pyln.Meter(sr).integrated_loudness(x)


def norm_lufs(x, target, sr=SR):
    L = lufs(x, sr)
    if not np.isfinite(L):
        return x
    return (x * db(target - L)).astype(np.float32)


def write_stem(name: str, st: np.ndarray):
    st = np.asarray(st, np.float32)
    assert st.shape == (NS, 2), st.shape
    STEMS.mkdir(parents=True, exist_ok=True)
    sf.write(STEMS / f"{name}.wav", st, SR, subtype="FLOAT")
    print(f"wrote audio/stems/{name}.wav  peak {peak_db(st):.1f} dBFS  {lufs(st):.1f} LUFS", flush=True)


# ----------------------------------------------------------------------------------------------------------
# reverbs & echoes
# ----------------------------------------------------------------------------------------------------------
def make_ir(rt60=2.0, predelay=0.02, lo_mult=1.3, hi_mult=0.45, er_n=12, er_span=0.07, er_gain=0.5,
            width=1.0, build=0.02, length=None, seed=0, sr=SR, bright=1.0):
    """Procedural stereo room impulse response: sparse early reflections + a diffuse, decorrelated tail
    whose decay time depends on frequency (3 bands: <300 Hz, mid, >3 kHz)."""
    rng = np.random.default_rng(seed)
    n = int((length or rt60 * 1.25) * sr)
    t = np.arange(n) / sr
    tail = np.zeros((n, 2))
    bands = [(None, 300, rt60 * lo_mult), (300, 3000, rt60), (3000, 11000 * bright, rt60 * hi_mult)]
    for ch in range(2):
        nz = rng.standard_normal(n)
        for lo, hi, rt in bands:
            b = bandpass(nz, lo, hi, sr, order=2)
            tail[:, ch] += b * np.exp(-6.91 * t / max(rt, 1e-3))
    tail *= (1 - np.exp(-t / max(build, 1e-4)))[:, None]
    mid = tail.mean(1, keepdims=True)
    tail = mid + (tail - mid) * width
    tail /= np.sqrt((tail ** 2).sum() / 2) + 1e-9
    er = np.zeros((n, 2))
    for k in range(er_n):
        d = rng.uniform(0.004, er_span)
        i = int(d * sr)
        if i < n:
            g = er_gain * (1 - d / (er_span * 1.3)) * rng.uniform(0.4, 1.0)
            er[i, 0] += g * rng.choice([-1, 1])
            er[min(n - 1, i + int(rng.uniform(0, 0.004) * sr)), 1] += g * rng.choice([-1, 1])
    ir = tail * 0.35 + er
    pd = int(predelay * sr)
    ir = np.concatenate([np.zeros((pd, 2)), ir])
    return ir.astype(np.float32)


_IR_CACHE = {}


def ir(name, **kw):
    k = key(name, kw)
    if k not in _IR_CACHE:
        _IR_CACHE[k] = make_ir(**kw)
    return _IR_CACHE[k]


def convolve(x, h):
    """mono/stereo x (n,) or (n,2) * stereo IR (m,2) -> stereo (n+m-1, 2)."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        return np.stack([oaconvolve(x, h[:, 0]), oaconvolve(x, h[:, 1])], 1).astype(np.float32)
    return np.stack([oaconvolve(x[:, 0], h[:, 0]), oaconvolve(x[:, 1], h[:, 1])], 1).astype(np.float32)


def reverb(dry_st, h, wet_db, dry_db=0.0, tail=None):
    """dry stereo + convolved wet; output extended by the IR length."""
    wet = convolve(dry_st, h) * db(wet_db)
    out = np.zeros_like(wet)
    out[:len(dry_st)] += stereo(dry_st) * db(dry_db)
    out += wet
    if tail is not None:
        out = out[:len(dry_st) + int(tail * SR)]
    return out.astype(np.float32)


def echo(x, delay, fb, n=8, lp=None, hp=None, spread=0.0, sr=SR, detune=0.0):
    """Multi-tap feedback echo (stereo ping-pong if spread>0).  Each repeat is re-filtered (darker/thinner)."""
    x = stereo(x)
    d = int(delay * sr)
    out = np.zeros((len(x) + d * n + 1, 2), np.float32)
    out[:len(x)] += x
    rep = x.copy()
    for k in range(1, n + 1):
        if lp:
            rep = bandpass(rep, None, lp, sr, order=1)
        if hp:
            rep = bandpass(rep, hp, None, sr, order=1)
        if detune:
            L = len(rep)
            rep = resample_poly(rep, 1000, int(1000 * (1 + detune)), axis=0).astype(np.float32)[:L]
        g = fb ** k
        p = spread * (1 if k % 2 else -1)
        th = (p + 1) * np.pi / 4
        i0 = d * k
        out[i0:i0 + len(rep), 0] += rep[:, 0] * g * np.cos(th) * np.sqrt(2)
        out[i0:i0 + len(rep), 1] += rep[:, 1] * g * np.sin(th) * np.sqrt(2)
    return out


# ----------------------------------------------------------------------------------------------------------
# WORLD vocoder & PSOLA
# ----------------------------------------------------------------------------------------------------------
def world_analyze(a, sr=VSR, f0_floor=55.0, f0_ceil=1000.0, fp=5.0, fast=False):
    import pyworld as pw
    x = np.asarray(a, np.float64)
    if fast:
        f0, t = pw.dio(x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=fp)
        f0 = pw.stonemask(x, f0, t, sr)
    else:
        f0, t = pw.harvest(x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=fp)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    return f0, t, sp, ap


def world_synth(f0, sp, ap, sr=VSR, fp=5.0):
    import pyworld as pw
    return pw.synthesize(np.ascontiguousarray(f0, np.float64), np.ascontiguousarray(sp),
                         np.ascontiguousarray(ap), sr, frame_period=fp).astype(np.float32)


def warp_sp(sp, factor):
    """Formant shift: scale the spectral envelope along frequency by `factor` (>1 = brighter/smaller)."""
    n = sp.shape[1]
    src = np.clip(np.arange(n) / factor, 0, n - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = src - i0
    return sp[:, i0] * (1 - w) + sp[:, i1] * w


def world_shift(a, semis, sr=VSR, formant=1.0):
    f0, t, sp, ap = world_analyze(a, sr)
    if formant != 1.0:
        sp = warp_sp(sp, formant)
    return world_synth(f0 * 2 ** (semis / 12), sp, ap, sr)


def world_stretch(f0, sp, ap, factor):
    """time-stretch WORLD parameters by `factor` (>1 = slower), linear frame interpolation."""
    n = len(f0)
    m = max(2, int(round(n * factor)))
    src = np.linspace(0, n - 1, m)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (src - i0)[:, None]
    return f0[np.round(src).astype(int)], sp[i0] * (1 - w) + sp[i1] * w, ap[i0] * (1 - w) + ap[i1] * w


def whisperize(a, sr=VSR, formant=1.0, stretch=1.0):
    """True whisper: resynthesize the WORLD spectral envelope with fully aperiodic (noise) excitation."""
    f0, t, sp, ap = world_analyze(a, sr, fast=True)
    if formant != 1.0:
        sp = warp_sp(sp, formant)
    if stretch != 1.0:
        f0, sp, ap = world_stretch(f0, sp, ap, stretch)
    return world_synth(f0, sp, np.ones_like(ap), sr) * 0.7


def psola(a, sr, pitch_fn, floor=70.0, ceil=700.0):
    """Praat overlap-add resynthesis with a new pitch contour.  pitch_fn(times, hz) -> new hz
    (evaluated on the voiced pitch points).  Timing is untouched (lip-sync safe)."""
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(np.asarray(a, np.float64), sr)
    man = call(snd, "To Manipulation", 0.005, floor, ceil)
    tier = call(man, "Extract pitch tier")
    n = int(call(tier, "Get number of points"))
    ts = np.array([call(tier, "Get time from index", i + 1) for i in range(n)])
    hz = np.array([call(tier, "Get value at index", i + 1) for i in range(n)])
    new = np.asarray(pitch_fn(ts, hz), float)
    call(tier, "Remove points between", 0, snd.duration)
    for t, h in zip(ts, new):
        call(tier, "Add point", float(t), float(max(h, 20.0)))
    call([man, tier], "Replace pitch tier")
    out = call(man, "Get resynthesis (overlap-add)")
    return out.values[0].astype(np.float32)[:len(a)]


def pitch_track(a, sr, floor=60, ceil=1100, step=0.01):
    import parselmouth
    snd = parselmouth.Sound(np.asarray(a, np.float64), sr)
    p = snd.to_pitch_ac(time_step=step, pitch_floor=floor, pitch_ceiling=ceil)
    return p.xs(), p.selected_array["frequency"]


def rng_for(*parts):
    return np.random.default_rng(int(key(*parts)[:8], 16))

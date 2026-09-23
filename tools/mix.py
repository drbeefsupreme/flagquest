"""Final mix: sums the department stems with sidechain ducking, bus compression, limiting and
loudness normalisation.  usage: python tools/mix.py [--target -14] [--out audio/final_mix.wav]"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import Pedalboard, Compressor, HighpassFilter
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import resample_poly


def limit(x, ceiling_db=-1.0, look_ms=5.0, release_ms=80.0):
    """look-ahead brickwall limiter (sample-peak, with 4x oversampled true-peak safety margin)"""
    ceil = 10 ** (ceiling_db / 20)
    pk = np.abs(x).max(axis=1)
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    la = max(1, int(look_ms * SR / 1000))
    g = minimum_filter1d(g, size=2 * la + 1)
    g = uniform_filter1d(g, size=la)
    rel = int(release_ms * SR / 1000)
    g = np.minimum(g, uniform_filter1d(minimum_filter1d(g, size=rel), size=rel))
    return x * g[:, None]


def true_peak_db(x):
    tp = max(np.abs(resample_poly(x[:, c], 4, 1)).max() for c in range(x.shape[1]))
    return 20 * np.log10(tp + 1e-12)

ROOT = Path(__file__).resolve().parents[1]
SR = 48000
DUR = json.loads((ROOT / "timeline.json").read_text())["duration"]
N = int(round(DUR * SR))

# stem -> gain dB (tuned by ear-proxy: loudness report below)
STEMS = {
    "dialogue": 0.0,
    "crowd": -2.0,
    "choir": -1.0,
    "music": -5.0,
    "sfx": -3.0,
    "ambience": -7.0,
}
DUCK = {"music": 7.0, "ambience": 4.0, "sfx": 2.0, "crowd": 4.0}   # max dB reduction under dialogue


def load(name):
    p = ROOT / f"audio/stems/{name}.wav"
    if not p.exists():
        if name == "dialogue" and (ROOT / "audio/stems/dialogue_dry.wav").exists():
            p = ROOT / "audio/stems/dialogue_dry.wav"
        else:
            return None
    a, sr = sf.read(p, dtype="float32", always_2d=True)
    assert sr == SR, (p, sr)
    if a.shape[1] == 1:
        a = np.repeat(a, 2, 1)
    out = np.zeros((N, 2), np.float32)
    out[: min(N, len(a))] = a[:N]
    return out


def envelope(x, attack_ms=15, release_ms=350):
    """smoothed dialogue presence 0..1"""
    m = np.abs(x).max(axis=1)
    hop = 240
    k = len(m) // hop
    r = np.sqrt(np.mean(m[: k * hop].reshape(k, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(r + 1e-9)
    act = np.clip((db + 50) / 20, 0, 1)            # -50 dBFS -> 0, -30 dBFS -> 1
    act = maximum_filter1d(act, size=int(release_ms / 5))
    act = uniform_filter1d(act, size=int(attack_ms / 5) + 1)
    return np.repeat(act, hop)[:N] if len(act) * hop >= N else np.pad(np.repeat(act, hop), (0, N - len(act) * hop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=-14.0)
    ap.add_argument("--out", default=str(ROOT / "audio/final_mix.wav"))
    a = ap.parse_args()
    meter = pyln.Meter(SR)
    stems = {k: load(k) for k in STEMS}
    dia = stems["dialogue"]
    env = envelope(dia) if dia is not None else np.zeros(N, np.float32)
    mix = np.zeros((N, 2), np.float32)
    for k, g in STEMS.items():
        s = stems[k]
        if s is None:
            print(f"  {k:9s} MISSING")
            continue
        lufs = meter.integrated_loudness(s) if np.abs(s).max() > 1e-6 else -99
        gain = 10 ** (g / 20)
        if k in DUCK:
            gain = gain * 10 ** (-DUCK[k] * env / 20)[:, None]
        mix += s * gain
        print(f"  {k:9s} {lufs:6.1f} LUFS  peak {20*np.log10(np.abs(s).max()+1e-9):6.1f} dBFS  gain {g:+.1f} dB")
    # sacred silence 138.5 -> 139.5 (fade back in by 140.0)
    t = np.arange(N) / SR
    sil = np.ones(N, np.float32)
    sil[(t >= 138.5) & (t < 139.5)] = 0.0
    ramp = (t >= 139.5) & (t < 140.0)
    sil[ramp] = ((t[ramp] - 139.5) / 0.5) ** 2
    mix *= sil[:, None]
    board = Pedalboard([HighpassFilter(22), Compressor(threshold_db=-16, ratio=2.0, attack_ms=12, release_ms=180)])
    mix = board(mix.T, SR).T.astype(np.float32)
    gain_db = a.target - meter.integrated_loudness(mix)
    for _ in range(6):   # limit -> re-measure -> correct (limiting eats loudness)
        out_mix = limit(mix * 10 ** (gain_db / 20), ceiling_db=-1.5)
        lufs = meter.integrated_loudness(out_mix)
        if abs(lufs - a.target) < 0.1:
            break
        gain_db += a.target - lufs
    mix = (out_mix * sil[:, None]).astype(np.float32)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, mix, SR, subtype="FLOAT")
    print(f"master {meter.integrated_loudness(mix):.2f} LUFS  sample peak {20*np.log10(np.abs(mix).max()):.2f} dBFS"
          f"  true peak {true_peak_db(mix):.2f} dBTP -> {out}")


if __name__ == "__main__":
    main()

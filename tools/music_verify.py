"""Verify audio/stems/music.wav against the picture.

  python tools/music_verify.py              # report (format, silence, peaks, loudness, hit onsets)
  python tools/music_verify.py --spec s04   # + spectrogram PNG of a scene -> audio/music/spec_<scene>.png

Onset method: 1 ms hop, energy of the 80 Hz–16 kHz band in 3 ms windows; the onset of a hit is the frame of
maximum positive log-energy slope (dB/ms) inside ±40 ms of the cue — i.e. the attack of the transient.
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 48000
N = 9264000


def band(x, lo=80, hi=16000):
    sos = signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def env_db(xm, a_t, z_t, w_ms=3, hop_ms=1):
    hop, w = int(SR * hop_ms / 1000), int(SR * w_ms / 1000)
    a = int(a_t * SR)
    seg = xm[a:int(z_t * SR) + w]
    e = np.array([np.mean(seg[i:i + w] ** 2) for i in range(0, len(seg) - w, hop)]) + 1e-14
    return (a + np.arange(len(e)) * hop) / SR, 10 * np.log10(e)


def onset_near(xm, t, win=0.04):
    """attack onset with backtracking: energy envelope in 4 ms windows (0.5 ms hop); find the steepest 4 ms
    rise within ±win of t, then backtrack to the knee: the last point (≤ 20 ms back) within 2 dB of the floor before it.
    A window that starts at ts covers [ts, ts+4 ms], so the transient starts at ts_knee + 4 ms.
    Returns (t_onset, rise_dB = level 10 ms after the steepest point − level at the minimum)."""
    w_ms = 4
    ts, le = env_db(xm, t - win - 0.03, t + win + 0.03, w_ms=w_ms, hop_ms=0.5)
    k = 8  # 4 ms
    d = le[k:] - le[:-k]
    tc = ts[:-k] + w_ms / 1000
    msk = (tc >= t - win) & (tc <= t + win)
    i = int(np.argmax(np.where(msk, d, -1e9)))
    j = i + k // 2
    lo = max(0, j - 40)
    floor = le[lo:j + 1].min()
    q = lo + int(np.flatnonzero(le[lo:j + 1] <= floor + 2.0)[-1])  # knee: last point near the floor
    t_on = ts[q] + w_ms / 1000
    rise = float(le[min(len(le) - 1, j + 20)] - floor)
    return float(t_on), rise


def flux_onset(x_full, t, win=0.04, _cache={}):
    """standard spectral-flux onset (librosa onset_strength, n_fft 1024, hop 64 = 1.33 ms, centred frames):
    the time of the onset-strength peak nearest the cue within ±win — the perceptual attack."""
    import librosa
    a = max(0, int((t - 0.5) * SR))
    y = x_full[a:int((t + 0.5) * SR)].astype(np.float32)
    env = librosa.onset.onset_strength(y=y, sr=SR, n_fft=1024, hop_length=64, lag=1, max_size=1, center=True)
    tt = a / SR + np.arange(len(env)) * 64 / SR
    msk = (tt >= t - win) & (tt <= t + win)
    i = int(np.argmax(np.where(msk, env, -1)))
    return float(tt[i]), float(env[i] / (np.median(env) + 1e-9))


def cut_near(xm, t, win=0.04):
    """steepest drop (largest 4 ms fall) within ±win. Returns (t_cut, drop_dB)."""
    ts, le = env_db(xm, t - win - 0.01, t + win + 0.01)
    d = le[:-4] - le[4:]
    tt = ts[:-4] + 0.0015 + 0.003
    msk = (tt >= t - win) & (tt <= t + win)
    i = np.argmax(np.where(msk, d, -1e9))
    return float(tt[i]), float(d[i])


def lufs(x):
    import pyloudnorm as pyln
    return pyln.Meter(SR).integrated_loudness(x.astype(np.float64))


def spec_png(x, t0, t1, path, title=""):
    from PIL import Image, ImageDraw
    xm = x[int(t0 * SR):int(t1 * SR)].mean(1)
    f, t, Z = signal.stft(xm, SR, nperseg=4096, noverlap=4096 - 480)
    mag = 20 * np.log10(np.abs(Z) + 1e-7)
    # log-frequency axis 30 Hz–16 kHz, 360 rows
    fl = np.geomspace(30, 16000, 360)
    rows = np.array([np.interp(fl, f, mag[:, j]) for j in range(mag.shape[1])]).T[::-1]
    v = np.clip((rows + 100) / 80, 0, 1)
    img = (np.stack([v ** 0.7, v ** 1.5, 0.4 * v + 0.6 * v ** 3], -1) * 255).astype(np.uint8)
    im = Image.fromarray(img).resize((min(2400, img.shape[1]), 360))
    d = ImageDraw.Draw(im)
    W = im.size[0]
    for s in range(int(np.ceil(t0)), int(t1) + 1):
        xx = int((s - t0) / (t1 - t0) * W)
        d.line([(xx, 0), (xx, 6 if s % 5 else 14)], fill=(255, 255, 255))
        if s % 5 == 0:
            d.text((xx + 2, 14), str(s), fill=(255, 255, 255))
    for hz_ in (100, 1000, 10000):
        y = int((1 - np.log(hz_ / 30) / np.log(16000 / 30)) * 360)
        d.line([(0, y), (8, y)], fill=(255, 255, 0))
        d.text((10, y - 6), f"{hz_}Hz", fill=(255, 255, 0))
    # loudness curve (short-term RMS, dB) overlay
    hop = int(0.05 * SR)
    r = np.array([np.sqrt(np.mean(xm[i:i + hop] ** 2) + 1e-12) for i in range(0, len(xm) - hop, hop)])
    rd = 20 * np.log10(r)
    pts = [(int(i / len(rd) * W), int(360 - (np.clip(v_, -60, 0) + 60) / 60 * 360)) for i, v_ in enumerate(rd)]
    d.line(pts, fill=(0, 255, 120), width=1)
    d.text((W - 300, 2), title, fill=(255, 255, 255))
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="")
    ap.add_argument("--path", default=os.path.join(ROOT, "audio", "stems", "music.wav"))
    a = ap.parse_args()
    info = sf.info(a.path)
    x, sr = sf.read(a.path, dtype="float32", always_2d=True)
    tl = json.load(open(os.path.join(ROOT, "timeline.json")))
    print(f"format: {info.samplerate} Hz, {info.channels} ch, {info.subtype}, {info.frames} frames "
          f"({'OK' if (info.frames == N and info.samplerate == SR and info.channels == 2 and info.subtype == 'FLOAT') else 'BAD'})")
    sil = x[int(138.5 * SR):int(140.0 * SR)]
    print(f"silence 138.5–140.0: max |x| = {np.abs(sil).max():.3g} ({'digital zero' if not np.any(sil) else 'NOT ZERO'})")
    pk = np.abs(x).max()
    tp = np.abs(signal.resample_poly(x, 4, 1, axis=0)).max()
    print(f"sample peak {20 * np.log10(pk):.2f} dBFS, true peak {20 * np.log10(tp):.2f} dBTP, clipping: {'NO' if tp < 1 else 'YES'}")
    print(f"integrated loudness: {lufs(x):.2f} LUFS")
    for s in tl["scenes"]:
        seg = x[int(s['start'] * SR):int(s['end'] * SR)]
        try:
            L = lufs(seg)
        except Exception:
            L = float('nan')
        print(f"  {s['id']:5s} {s['start']:6.1f}-{s['end']:6.1f}  {L:6.1f} LUFS  peak {20 * np.log10(np.abs(seg).max() + 1e-12):6.1f} dBFS")
    xm = band(x.mean(1))
    hits = json.load(open(os.path.join(ROOT, "audio", "music", "cues.json")))
    cues = {c["id"]: c["t"] for c in tl["cues"]}
    print("sync points (logged t → measured attack/cut, error ms, rise above pre-level / 4 ms drop):")
    worst = worst_all = 0
    xmono = x.mean(1)
    print("  (flux = spectral-flux onset peak = perceptual attack; knee = start of the energy rise)")
    for h in hits:
        kind = h.get("kind", "onset")
        if kind == "cut":
            t_on, sl = cut_near(xm, h["t"])
            err = (t_on - h["t"]) * 1000
            knee = ""
        else:
            t_on, sl = flux_onset(xmono, h["t"])
            err = (t_on - h["t"]) * 1000
            tk, _ = onset_near(xm, h["t"])
            knee = f"knee {1000 * (tk - h['t']):+6.1f} ms"
        tag = "cue" if h["id"] in cues else "   "
        worst = max(worst, abs(err)) if tag == "cue" else worst
        worst_all = max(worst_all, abs(err))
        what = f"drop {sl:5.1f} dB" if kind == "cut" else f"flux x{sl:5.1f}"
        print(f"  {tag} {h['id']:22s} {h['t']:8.3f} → {t_on:8.3f}  {err:+6.1f} ms  {knee:15s} {what}  {h['desc']}")
    print(f"worst error: named cues {worst:.1f} ms, all sync points {worst_all:.1f} ms")
    if a.spec:
        sc = {s["id"]: s for s in tl["scenes"]}
        ids = a.spec.split(",") if a.spec != "all" else list(sc)
        for i in ids:
            s = sc[i]
            p = os.path.join(ROOT, "audio", "music", f"spec_{i}.png")
            spec_png(x, max(0, s["start"] - 0.5), min(193.0, s["end"] + 0.5), p, f"music {i}")
            print("wrote", p)


if __name__ == "__main__":
    sys.exit(main())

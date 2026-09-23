"""Verification for the vocal stems.  usage: python tools/vocal_check.py [dialogue] [crowd] [choir] [levels]
 dialogue : per-line timing (processed vs dry: envelope cross-correlation lag + threshold onset vs the
            locked start) and intelligibility (faster-whisper small.en on each processed line vs script)
 crowd    : format, the sacred silence 138.5-140.0, the hard stop at 138.5, 'Glorious' timing
 choir    : pitch of every sung note (median cents error of the sustained vowel, per voice part render)
 levels   : format + integrated LUFS / true-peak-ish per stem
Writes audio/vocal/check_<what>.json."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocal_lib import NS, ROOT, SR, STEMS, TL, VOCAL, bandpass, lufs, midi_hz, peak_db, pitch_track


def env_db(x, sr=SR, win=0.005):
    x = x if x.ndim == 1 else x.mean(1)
    h = int(win * sr)
    n = len(x) // h
    e = np.sqrt(np.mean(x[:n * h].reshape(n, h) ** 2, 1) + 1e-14)
    return 20 * np.log10(e), h


def onset(x, sr=SR, rel=-28.0, band=(250, 4000)):
    """first 5 ms frame whose band-limited level is within `rel` dB of the clip's 99th percentile level."""
    y = bandpass(x if x.ndim == 1 else x.mean(1), band[0], band[1], sr, order=4)
    e, h = env_db(y, sr)
    ref = np.percentile(e, 99)
    k = np.flatnonzero(e > ref + rel)
    return (k[0] * h / sr) if len(k) else None


def xcorr_lag(a, b, sr=SR, maxlag=0.1):
    """lag (s) of b relative to a using 2 ms band envelopes (positive = b later)."""
    ea, h = env_db(bandpass(a, 300, 3400, sr), sr, 0.002)
    eb, _ = env_db(bandpass(b, 300, 3400, sr), sr, 0.002)
    ea, eb = np.maximum(ea, -60) + 60, np.maximum(eb, -60) + 60
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    L = int(maxlag * sr / h)
    lags = range(-L, L + 1)
    c = [np.dot(ea[max(0, -k):n - max(0, k)], eb[max(0, k):n - max(0, -k)]) for k in lags]
    return lags[int(np.argmax(c))] * h / sr


def norm_words(s):
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s.lower())
    s = s.replace("twenty forty-two", "2042").replace("forty-two", "42")
    return re.findall(r"[a-z0-9']+", s)


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(1, len(r))


def check_dialogue():
    from faster_whisper import WhisperModel
    wm = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
    idx = json.loads((VOCAL / "dialogue_lines/index.json").read_text())
    stem, _ = sf.read(STEMS / "dialogue.wav", dtype="float32")
    dry, _ = sf.read(STEMS / "dialogue_dry.wav", dtype="float32")
    rows, bad = [], 0
    for ln in TL["lines"]:
        lid = ln["id"]
        pre = idx[lid]["pre"]
        proc, _ = sf.read(VOCAL / f"dialogue_lines/{lid}.wav", dtype="float32")
        i0 = int(ln["start"] * SR)
        n = int((ln["dur"] + 0.3) * SR)
        d = dry[i0 - int(0.2 * SR): i0 + n, 0]
        s = stem[i0 - int(0.2 * SR): i0 + n]
        lag = xcorr_lag(d, s)                                   # processed speech vs dry, in the stem
        p_speech = proc[int(pre * SR):]
        on_p = onset(p_speech)                                   # energy onset re: locked start
        on_d = onset(dry[i0:i0 + n, 0])
        v_p = voiced_onset(p_speech[:n].mean(1))                 # first voiced frame re: locked start
        v_d = voiced_onset(dry[i0:i0 + n, 0])
        hyp = transcribe(wm, p_speech[: int((ln["dur"] + 0.25) * SR)].mean(1))
        hyp_d = transcribe(wm, dry[i0:i0 + int((ln["dur"] + 0.25) * SR), 0])
        w, w_d = wer(ln["text"], hyp), wer(ln["text"], hyp_d)
        ok = abs(lag) <= 0.010 and abs(v_p - v_d) <= 0.0105 and w <= max(0.25, w_d + 0.05)
        bad += not ok
        rows.append(dict(id=lid, fx=idx[lid]["fx"], lag_ms=round(lag * 1000, 1),
                         voiced_onset_ms=round(v_p * 1000, 1), dry_voiced_onset_ms=round(v_d * 1000, 1),
                         energy_onset_ms=round(on_p * 1000, 1), dry_energy_onset_ms=round(on_d * 1000, 1),
                         wer=round(w, 2), wer_dry=round(w_d, 2), hyp=hyp, ok=bool(ok)))
        print(f"{lid} {idx[lid]['fx']:6s} lag {lag*1000:5.1f}ms  voiced {v_p*1000:5.1f}ms (dry {v_d*1000:5.1f})"
              f"  energy {on_p*1000:5.1f} (dry {on_d*1000:5.1f})  WER {w:4.2f} (dry {w_d:4.2f})"
              f"  {'OK ' if ok else 'BAD'} | {hyp}", flush=True)
    print("dialogue lines failing:", bad)
    (VOCAL / "check_dialogue.json").write_text(json.dumps(rows, indent=1))


def voiced_onset(x, sr=SR):
    ts, hz = pitch_track(x, sr, 60, 700, 0.005)
    v = hz > 0
    run = np.convolve(v.astype(int), np.ones(4, int), mode="valid") == 4    # >= 20 ms voiced
    k = np.flatnonzero(run)
    return float(ts[k[0]] - 0.0025) if len(k) else float("nan")


def transcribe(wm, x, sr=SR):
    seg16 = resample_poly(x, 1, sr // 16000).astype(np.float32)
    segs, _ = wm.transcribe(seg16, language="en", beam_size=5, vad_filter=False)
    return " ".join(s.text.strip() for s in segs)


def check_choir():
    hy = TL["hymn"]
    spb = 60.0 / hy["bpm"]
    notes = json.loads((VOCAL / "choir_notes.json").read_text())
    singers = notes["hymn"]
    bad = [c for c in singers if not (abs(c["median_cents"]) <= 25 and c["voiced_frac"] >= 0.85)]
    errs = [abs(c["median_cents"]) for c in singers]
    print(f"per-singer sustained vowels: {len(singers)} notes, |median| mean {np.mean(errs):.1f} max "
          f"{np.max(errs):.1f} cents, outside +-25: {len(bad)}")
    # each section bus (6 voices in unison) pitch-tracked on every note
    t0p = 149.0
    rows = []
    for part_i, part in enumerate("SATB"):
        x, _ = sf.read(VOCAL / f"choir_parts/{part}.wav", dtype="float32")
        x = x.mean(1)
        for ni, (word, beat, beats, pitches) in enumerate(hy["notes"]):
            a = hy["start"] + beat * spb + 0.3
            b = hy["start"] + (beat + beats) * spb - 0.35
            seg = x[int((a - t0p) * SR):int((b - t0p) * SR)]
            ts, hz = pitch_track(seg, SR, 50, 1100, 0.01)
            v = hz[hz > 0]
            c = 1200 * np.log2(v / float(midi_hz(pitches[part_i]))) if len(v) else np.array([np.nan])
            rows.append(dict(part=part, note=ni, word=word, midi=pitches[part_i], median_cents=round(float(np.median(c)), 1),
                             voiced=round(len(v) / max(1, len(hz)), 2)))
    print("section buses (median cents per note):")
    for part in "SATB":
        print(" ", part, [(r["word"], r["midi"], r["median_cents"]) for r in rows if r["part"] == part])
    nbad = sum(1 for r in rows if not abs(r["median_cents"]) <= 25)
    print("section notes outside +-25 cents:", nbad)
    # convergence chord just before the hymn: spectral peaks at the chord tones
    cv, _ = sf.read(VOCAL / "choir_parts/convergence.wav", dtype="float32")
    seg = cv[int((153.9 - t0p) * SR):int((154.45 - t0p) * SR)].mean(1)
    seg = seg * np.hanning(len(seg))
    nfft = 1 << 21
    S = np.abs(np.fft.rfft(seg, nfft))
    fr = np.fft.rfftfreq(nfft, 1 / SR)
    chord = []
    for tone in sorted({v for _, v in __import__("vocal_choir").conv_voices()}):
        f = float(midi_hz(tone))
        m = (fr > f * 2 ** (-60 / 1200)) & (fr < f * 2 ** (60 / 1200))
        k = np.flatnonzero(m)[np.argmax(S[m])]
        chord.append((tone, round(1200 * np.log2(fr[k] / f), 1)))
    print("convergence chord peaks (midi, cents):", chord)
    # can a speech recognizer hear the words in the sung hymn?
    from faster_whisper import WhisperModel
    wm = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
    ch, _ = sf.read(STEMS / "choir.wav", dtype="float32")
    hyp = transcribe(wm, ch[int(154.3 * SR):int(166.5 * SR)].mean(1))
    print("whisper hears the hymn as:", hyp)
    (VOCAL / "check_choir.json").write_text(json.dumps(dict(sections=rows, convergence_chord=chord, whisper=hyp,
                                                          singers_outside_25c=len(bad)), indent=1))


def check_crowd():
    x, _ = sf.read(STEMS / "crowd.wav", dtype="float32")
    i = int(138.5 * SR)
    pre = x[i - int(0.5 * SR):i]
    out = dict(peak_138_5_to_140=float(np.abs(x[i:int(140.0 * SR)]).max()),
               rms_dbfs_last_500ms_before_stop=round(float(20 * np.log10(np.sqrt(np.mean(pre ** 2)) + 1e-12)), 1),
               last_nonzero_sample_before_140=round(float(np.flatnonzero(np.abs(x[:int(140 * SR)]).max(1) > 0)[-1] / SR), 5))
    g = x[int(188.0 * SR):int(190.0 * SR)].mean(1)
    e, h = env_db(g, SR, 0.005)
    k = np.flatnonzero(e > e.max() - 20)
    out["glorious_onset_s"] = round(188.0 + k[0] * h / SR, 3)
    for name, (a, b) in {"slop_34.8-47": (34.8, 47.0), "murmur_100.5-125.5": (100.5, 125.5),
                         "babel_125.5-138.5": (125.5, 138.5), "babel_roar_136-138.5": (136.0, 138.5),
                         "glorious_188.2-190.5": (188.2, 190.5)}.items():
        out[f"lufs_{name}"] = round(float(lufs(x[int(a * SR):int(b * SR)])), 1)
    for k_, v in out.items():
        print(f"  {k_}: {v}")
    (VOCAL / "check_crowd.json").write_text(json.dumps(out, indent=1))



def check_levels():
    out = {}
    for name in ("dialogue", "crowd", "choir"):
        p = STEMS / f"{name}.wav"
        if not p.exists():
            continue
        info = sf.info(p)
        x, sr = sf.read(p, dtype="float32")
        ok = info.samplerate == SR and info.channels == 2 and info.frames == NS and info.subtype == "FLOAT"
        sil = np.abs(x[int(138.5 * SR):int(140.0 * SR)]).max()
        out[name] = dict(ok_format=bool(ok), frames=int(info.frames), lufs=round(float(lufs(x)), 1),
                         peak_dbfs=round(float(peak_db(x)), 1),
                         silence_138_5_140_peak_dbfs=round(float(peak_db(np.array([sil]))), 1))
        print(name, out[name])
    (VOCAL / "check_levels.json").write_text(json.dumps(out, indent=1))


def spec_img(x, sr=SR, fmax=8000, h=180, px_per_s=60, title=""):
    import cv2
    from scipy.signal import stft
    hop = max(64, int(sr / px_per_s))
    f, t, Z = stft(x, sr, nperseg=2048, noverlap=2048 - hop if hop < 2048 else 0)
    S = 20 * np.log10(np.abs(Z[f <= fmax]) + 1e-7)
    S = np.clip((S - (S.max() - 80)) / 80, 0, 1)
    img = cv2.applyColorMap((S[::-1] * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    img = cv2.resize(img, (img.shape[1], h), interpolation=cv2.INTER_AREA)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def check_view():
    """spectrogram contact sheet of every vocal event -> audio/vocal/spectra.png (look at it)."""
    import cv2
    rows = []
    for stem, a, b, title in [("crowd", 34.0, 47.5, "crowd s02 slop whisper-chorus (crash 39.2)"),
                              ("crowd", 100.0, 126.5, "crowd s05b murmurs"),
                              ("crowd", 125.0, 140.5, "crowd s06 Babel -> roar 136 -> HARD STOP 138.5"),
                              ("crowd", 187.5, 192.0, "crowd s08 GLORIOUS 188.3"),
                              ("choir", 149.5, 169.0, "choir: convergence 150-154.5, hymn 154.5-165.93"),
                              ("dialogue", 60.0, 75.0, "dialogue: Hypermind H01/H02, L01"),
                              ("dialogue", 90.0, 126.0, "dialogue: radio K01-K07, shouts X01-X05, void X06/X07")]:
        x, _ = sf.read(STEMS / f"{stem}.wav", dtype="float32", start=int(a * SR), stop=int(b * SR))
        rows.append(spec_img(x.mean(1), px_per_s=int(1400 / (b - a)), title=f"{title}  [{a}-{b}s]"))
    W = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0))) for r in rows], 0)
    cv2.imwrite(str(VOCAL / "spectra.png"), sheet)
    print("wrote audio/vocal/spectra.png", sheet.shape)



if __name__ == "__main__":
    what = sys.argv[1:] or ["dialogue", "levels"]
    for w in what:
        globals()[f"check_{w}"]()

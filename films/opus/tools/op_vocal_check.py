"""OPVS SCHISMATICVM — verification of the vocal stems.   Owner: Vocalist-2.
usage: python films/opus/tools/op_vocal_check.py [dialogue] [crowd] [choir] [levels] [view]   (default: all but view)

 dialogue : per line, from the stem itself: timing vs dialogue_dry.wav (2 ms envelope cross-correlation, whole line
            and first 450 ms), Whisper small.en transcript + WER vs the script (window start-0.15 .. end+0.35, as
            tools/qa_dialogue.py), the dry take's WER for reference, reverb bloom (level 0.35-0.8 s after the line
            re: its speech level).  CANTOR (Latin) lines are reported but excluded from the mean.
 crowd    : the sacred silence 145.0-146.5 (digital zero), Babel's hard stop at 145.0, section loudness, "Glorious!" onset
 choir    : pitch of every sung note (cantor, hymn section buses, gratias) from choir_notes.json + a bus re-measure,
            the convergence chord at 159.0, Whisper on the hymn
 levels   : format (48 kHz / 2 ch / float32 / 10,176,000 frames), integrated LUFS, peak, silence window peak
 view     : spectrogram contact sheet -> audio/vocal/spectra.png
Writes films/opus/audio/vocal/check_<what>.json."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from op_vocal_lib import (NS, SILENCE, SR, TL, VOCAL, align_lag, bandpass, lufs, midi_hz, peak_db,  # noqa: E402
                          pitch_track)
from op_vocal_lib import VL  # noqa: E402

STEMS = VL.STEMS
_WM = None


def whisper():
    global _WM
    if _WM is None:
        from faster_whisper import WhisperModel
        _WM = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
    return _WM


def transcribe(x, sr=SR):
    seg16 = resample_poly(np.asarray(x, np.float32), 1, sr // 16000).astype(np.float32)
    segs, _ = whisper().transcribe(seg16, language="en", beam_size=5, vad_filter=False)
    return " ".join(s.text.strip() for s in segs)


NUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
       "eight": "8", "nine": "9", "ten": "10"}


def norm_words(s):
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s.lower())
    s = s.replace("twenty forty-two", "2042").replace("forty-two", "42").replace("-", " ")
    s = s.replace("ten thousand", "10000")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    return [NUM.get(w, w) for w in re.findall(r"[a-z0-9']+", s)]


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(1, len(r))


def env_db(x, win=0.05):
    x = x if x.ndim == 1 else x.mean(1)
    h = int(win * SR)
    n = len(x) // h
    return 20 * np.log10(np.sqrt(np.mean(x[:n * h].reshape(n, h) ** 2, 1)) + 1e-12)


def check_dialogue():
    stem, _ = sf.read(STEMS / "dialogue.wav", dtype="float32")
    dry, _ = sf.read(STEMS / "dialogue_dry.wav", dtype="float32")
    dry = dry if dry.ndim == 1 else dry[:, 0]
    idx = json.loads((VOCAL / "dialogue_lines/index.json").read_text())
    rows = []
    for ln in TL["lines"]:
        lid = ln["id"]
        latin = TL["cast"][ln["speaker"]]["fx"] == "cantor"
        a, b = int((ln["start"] - 0.15) * SR), int((ln["end"] + 0.35) * SR)
        seg = stem[a:b].mean(1)
        hyp = transcribe(seg)
        hyp_d = transcribe(dry[a:b])
        w, w_d = wer(ln["text"], hyp), wer(ln["text"], hyp_d)
        i0 = int(ln["start"] * SR)
        n = int((ln["dur"] + 0.3) * SR)
        lag = lag_on = float("nan")
        if not latin:
            d = dry[i0 - int(0.2 * SR):i0 + n]
            s = stem[i0 - int(0.2 * SR):i0 + n].mean(1)
            lag = align_lag(d, s, SR, 0.1)
            lag_on = align_lag(d[:int(0.65 * SR)], s[:int(0.65 * SR)], SR, 0.05)
        # reverb bloom of the processed line itself: level just after its speech vs its speech level
        proc, _ = sf.read(VOCAL / f"dialogue_lines/{lid}.wav", dtype="float32")
        pre = idx[lid]["pre"]
        sp_end = pre + (idx[lid].get("sung_end", ln["end"]) - ln["start"]) if latin else pre + ln["dur"]
        sp = proc[int(pre * SR):int(sp_end * SR)]
        e = env_db(sp)
        speech = float(np.percentile(e, 80)) if len(e) else float("nan")
        tail = proc[int((sp_end + 0.35) * SR):int((sp_end + 0.8) * SR)]
        bloom = float(20 * np.log10(np.sqrt(np.mean(tail ** 2)) + 1e-12) - speech) if len(tail) else float("nan")
        rows.append(dict(id=lid, speaker=ln["speaker"], fx=idx[lid]["fx"], latin=latin, wer=round(w, 2),
                         wer_dry=round(w_d, 2), lag_ms=round(lag * 1000, 1), onset_lag_ms=round(lag_on * 1000, 1),
                         bloom_db=round(bloom, 1), hyp=hyp, hyp_dry=hyp_d))
        print(f"{lid} {idx[lid]['fx']:8s} WER {w:4.2f} (dry {w_d:4.2f})  lag {lag*1000:5.1f}/{lag_on*1000:5.1f} ms  "
              f"bloom {bloom:6.1f} dB | {hyp}", flush=True)
    en = [r for r in rows if not r["latin"]]
    summ = dict(mean_wer=round(float(np.mean([r["wer"] for r in en])), 3),
                mean_wer_dry=round(float(np.mean([r["wer_dry"] for r in en])), 3),
                max_abs_lag_ms=float(np.nanmax([abs(r["lag_ms"]) for r in en])),
                lines_over_0_25=[r["id"] for r in en if r["wer"] > 0.25])
    print("summary (English lines):", summ)
    (VOCAL / "check_dialogue.json").write_text(json.dumps(dict(summary=summ, lines=rows), indent=1))


def cer(ref, hyp):
    """character error rate on letters only (Latin: spaces and punctuation ignored)"""
    r = re.sub(r"[^a-z]", "", ref.lower())
    h = re.sub(r"[^a-z]", "", hyp.lower())
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(1, len(r))


def check_latin():
    """the sung Latin, heard by multilingual Whisper small (language forced to Latin and to Italian): letter CER"""
    from faster_whisper import WhisperModel
    wm = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
    dia, _ = sf.read(STEMS / "dialogue.wav", dtype="float32")
    ch, _ = sf.read(STEMS / "choir.wav", dtype="float32")
    items = [("A01 incipit", dia, 1.9, 5.2, "Incipit opus schismaticum"),
             ("antiphon", ch, 162.9, 168.9, "Vexilla sint vexilla sunt vexilla erunt vexilla"),
             ("A02 explicit", dia, 204.5, 207.8, "Explicit opus schismaticum"),
             ("gratias", ch, 207.4, 210.2, "Vexillo gratias")]
    out = []
    for name, x, a, b, ref in items:
        seg16 = resample_poly(x[int(a * SR):int(b * SR)].mean(1), 1, 3).astype(np.float32)
        row = dict(name=name, ref=ref)
        for lang in ("la", "it"):
            segs, _ = wm.transcribe(seg16, language=lang, beam_size=5, vad_filter=False)
            hyp = " ".join(s.text.strip() for s in segs)
            row[lang] = hyp
            row[f"cer_{lang}"] = round(cer(ref, hyp), 2)
        out.append(row)
        print(f"{name:12s} la: {row['la']!r} (CER {row['cer_la']})  it: {row['it']!r} (CER {row['cer_it']})", flush=True)
    (VOCAL / "check_latin.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


def check_levels():
    out = {}
    for name in ("dialogue", "crowd", "choir"):
        p = STEMS / f"{name}.wav"
        if not p.exists():
            continue
        info = sf.info(p)
        x, sr = sf.read(p, dtype="float32")
        ok = info.samplerate == SR and info.channels == 2 and info.frames == NS and info.subtype == "FLOAT"
        sil = float(np.abs(x[int(SILENCE[0] * SR):int(SILENCE[1] * SR)]).max())
        out[name] = dict(ok_format=bool(ok), frames=int(info.frames), lufs=round(float(lufs(x)), 1),
                         peak_dbfs=round(float(peak_db(x)), 1), silence_145_146_5_peak=sil,
                         nan=bool(np.isnan(x).any()))
        print(name, out[name], flush=True)
    (VOCAL / "check_levels.json").write_text(json.dumps(out, indent=1))


def check_crowd():
    x, _ = sf.read(STEMS / "crowd.wav", dtype="float32")
    i = int(145.0 * SR)
    nzb = np.flatnonzero(np.abs(x[:int(146.5 * SR)]).max(1) > 0)
    out = dict(peak_145_to_146_5=float(np.abs(x[i:int(146.5 * SR)]).max()),
               last_nonzero_before_146_5=round(float(nzb[-1] / SR), 5) if len(nzb) else None,
               rms_dbfs_last_500ms_before_stop=round(float(20 * np.log10(
                   np.sqrt(np.mean(x[i - int(0.5 * SR):i] ** 2)) + 1e-12)), 1))
    g = x[int(202.0 * SR):int(204.0 * SR)].mean(1)
    e = env_db(g, 0.005)
    k = np.flatnonzero(e > e.max() - 20)
    out["glorious_onset_s"] = round(202.0 + k[0] * 0.005, 3) if len(k) else None
    for name, (a, b) in SECTIONS.items():
        out[f"lufs_{name}"] = round(float(lufs(x[int(a * SR):int(b * SR)])), 1)
    try:
        out["whisper_glorious"] = transcribe(x[int(202.6 * SR):int(204.2 * SR)].mean(1))
    except Exception as ex:  # pragma: no cover
        out["whisper_glorious"] = repr(ex)
    for k_, v in out.items():
        print(f"  {k_}: {v}", flush=True)
    (VOCAL / "check_crowd.json").write_text(json.dumps(out, indent=1))


SECTIONS = {"procession_22-34.4": (22.0, 34.4), "slop_34.4-48.5": (34.4, 48.5), "nations_106.5-113.6": (106.5, 113.6),
            "council_113.6-120": (113.6, 120.0), "crackpots_120-131.5": (120.0, 131.5),
            "babel_131.5-145": (131.5, 145.0), "babel_roar_141.8-145": (141.8, 145.0),
            "glorious_202.8-204.5": (202.8, 204.5)}


def check_choir():
    notes = json.loads((VOCAL / "choir_notes.json").read_text())
    for part in ("antiphon", "hymn", "gratias"):
        rows = notes.get(part, [])
        c = [abs(r["cents"]) for r in rows if r.get("cents") is not None and np.isfinite(r["cents"])]
        if c:
            bad = [r for r in rows if r.get("cents") is None or not np.isfinite(r["cents"]) or abs(r["cents"]) > 25]
            print(f"{part}: {len(rows)} notes, |cents| mean {np.mean(c):.1f} max {np.max(c):.1f}, >25c or unmeasured: "
                  f"{len(bad)}", flush=True)
    ch, _ = sf.read(STEMS / "choir.wav", dtype="float32")
    # the convergence chord: spectral peaks at the chord tones just after it lands
    from op_vocal_score import CONV_CHORD, CONV_T
    seg = ch[int((CONV_T + 0.3) * SR):int((CONV_T + 1.6) * SR)].mean(1)
    seg = seg * np.hanning(len(seg))
    nfft = 1 << 21
    S = np.abs(np.fft.rfft(seg, nfft))
    fr = np.fft.rfftfreq(nfft, 1 / SR)
    chord = []
    for tone in CONV_CHORD:
        f = float(midi_hz(tone))
        m = (fr > f * 2 ** (-60 / 1200)) & (fr < f * 2 ** (60 / 1200))
        k = np.flatnonzero(m)[np.argmax(S[m])]
        chord.append((tone, round(1200 * np.log2(fr[k] / f), 1)))
    print("convergence chord peaks (midi, cents):", chord, flush=True)
    hyp = transcribe(ch[int(168.5 * SR):int(180.6 * SR)].mean(1))
    print("whisper hears the hymn as:", hyp, flush=True)
    (VOCAL / "check_choir.json").write_text(json.dumps(dict(convergence_chord=chord, whisper_hymn=hyp), indent=1))


def spec_img(x, fmax=8000, h=180, px_per_s=60, title=""):
    import cv2
    cv2.setNumThreads(1)
    from scipy.signal import stft
    hop = max(64, int(SR / px_per_s))
    f, t, Z = stft(x, SR, nperseg=2048, noverlap=2048 - hop if hop < 2048 else 0)
    S = 20 * np.log10(np.abs(Z[f <= fmax]) + 1e-7)
    S = np.clip((S - (S.max() - 80)) / 80, 0, 1)
    img = cv2.applyColorMap((S[::-1] * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    img = cv2.resize(img, (img.shape[1], h), interpolation=cv2.INTER_AREA)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def check_view():
    import cv2
    rows = []
    for stem, a, b, title in [("dialogue", 0.0, 22.0, "dialogue m01: A01 cantor, N01-N03 basilica"),
                              ("dialogue", 62.0, 96.0, "dialogue m04: Hypermind ophanim, disputation"),
                              ("crowd", 22.0, 48.5, "crowd m02: procession, slop whispers"),
                              ("crowd", 106.0, 146.5, "crowd m06/m07: nations, council, void, BABEL -> 145.0"),
                              ("crowd", 202.0, 205.0, "crowd: GLORIOUS 202.9"),
                              ("choir", 154.0, 181.0, "choir: convergence 159, ison, cantor 163, hymn 168.7"),
                              ("choir", 206.0, 212.0, "choir: Vexillo gratias 207.6")]:
        p = STEMS / f"{stem}.wav"
        if not p.exists():
            continue
        x, _ = sf.read(p, dtype="float32", start=int(a * SR), stop=int(b * SR))
        rows.append(spec_img(x.mean(1), px_per_s=int(1400 / (b - a)), title=f"{title}  [{a}-{b}s]"))
    W = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0))) for r in rows], 0)
    cv2.imwrite(str(VOCAL / "spectra.png"), sheet)
    print("wrote", VOCAL / "spectra.png", sheet.shape)


if __name__ == "__main__":
    what = sys.argv[1:] or ["levels", "dialogue", "crowd", "choir"]
    for w in what:
        p = STEMS / f"{w}.wav"
        if w in ("dialogue", "crowd", "choir") and not p.exists():
            print(f"{w}: no stem yet")
            continue
        globals()[f"check_{w}"]()

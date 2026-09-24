"""THE FLAG GAME? — builds films/flaggame/audio/stems/choir.wav: Kokoro voices that SING.   Owner: Vocalist.

(a) 169.09-181.0  altar-call hymn on WOODWORTH (Bradbury 1849, public domain), E-flat major, 3/4, abridged to
    fit the prayer: "Just as I am, without one Flag ... I come, I come ... A-men", 21 beats at 114.13 BPM
    (beat = 0.52571 s) from the pickup 'Just' at 169.09, plagal A-men (A-flat -> E-flat) with 'men' on 180.13
    (Summer's Amen).  Grid written to films/flaggame/audio/vocal/hymn_grid.json for Composer (organ doubles
    it).  SATB, 3 singers per part, vowel onsets exactly on the beats, crisp onset consonants just before.
    Soft; dips 4 dB under the lorem-ipsum prayer lines.
(b) 143.94-146.77 one "HUP!" per flag raising in the t07 montage (T07Raised's 'raise' events from
    films/flaggame/cache/foley/*.json if present, else an accelerating 100-hit schedule), rising in pitch.
(c) 197.4-199.0  a radiant sung "FLAGS" on D major (Composer's G->D lands on 197.40; D loops into the film's
    opening chord) as the tract lands; released to silence by 199.0.

Built on film 1's singing engine (tools/vocal_choir.py: prosody-injected sing(), consonant shaping, pitch
check); its module reads timeline["hymn"] bpm/notes, which are supplied here before import.
usage: source films/flaggame/env.sh && python films/flaggame/tools/fg_vocal_choir.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vocal_lib as VL
from vocal_lib import (FILM, FILM_ROOT, NS, SR, TL, VOCAL, add_at, convolve, db, kokoro, lufs, make_ir, pan,
                       resample, smoothstep, up48, write_stem)

assert FILM == "flaggame", "source films/flaggame/env.sh first (VX_FILM=flaggame)"
T0, AMEN = TL["hymn"]["start"], TL["hymn"]["amen"]          # 169.09, 180.13
SPB = (AMEN - T0) / 21.0                                      # 0.52571 s  (114.13 BPM) — Composer follows this grid
# (word, beat, beats, [S, A, T, B])  E-flat major
NOTES = [("Just", 0, 1, [63, 58, 55, 51]), ("as", 1, 2, [67, 63, 58, 51]), ("I", 3, 1, [67, 63, 60, 48]),
         ("am", 4, 2, [70, 63, 58, 51]), ("with", 6, 1, [67, 63, 58, 55]), ("out", 7, 1, [65, 62, 58, 46]),
         ("one", 8, 1, [68, 63, 60, 44]), ("Flag", 9, 3, [67, 63, 58, 51]),
         ("I", 12, 1, [67, 63, 58, 51]), ("come", 13, 2, [65, 62, 58, 46]), ("I", 15, 1, [70, 67, 63, 51]),
         ("come", 16, 3, [67, 63, 58, 51]), ("A", 19, 2, [68, 60, 56, 44]), ("men", 21, 2, [67, 58, 55, 51])]
TL["hymn"].update(bpm=60.0 / SPB, beats=23, notes=NOTES)       # what tools/vocal_choir.py reads at import
FINAL = {"S": 69, "A": 66, "T": 62, "B": 50}                  # the closing 'FLAGS': D major (film tonic, loops to 0.0)
FINAL_T = 197.40                                              # vowel onset on cue tract_lands
import vocal_choir as VC                                       # noqa: E402
from fg_vocal_crowd import onset, yell                         # noqa: E402

PARTS = {"S": ["af_heart", "af_sky", "bf_emma"], "A": ["bf_alice", "af_jessica", "af_nova"],
         "T": ["am_puck", "am_liam", "bm_daniel"], "B": ["am_fenrir", "bm_george", "am_onyx"]}
PHRASE_END = {12, 19}


def ps_of(word):
    if word == "A":
        return "ˈA"
    if word == "men":
        return "mˈɛn"
    ps = kokoro().phonemes(word, "a").replace(".", "").strip()
    if "ˈ" not in ps:
        k = VC.vowel_index(ps)
        ps = ps[:k] + "ˈ" + ps[k:]
    return ps


def hymn():
    bus = np.zeros((NS, 2), np.float32)
    rows = []
    for part, voices in PARTS.items():
        pi = "SATB".index(part)
        for vi, voice in enumerate(voices):
            pl = VC.singer_plan(part, voice, vi)
            r = VL.rng_for("fg_hymn", voice)
            for ni, (word, beat, beats, pitches) in enumerate(NOTES):
                ps, midi = ps_of(word), pitches[pi]
                last = ni == len(NOTES) - 1
                t_on = T0 + beat * SPB + float(r.uniform(-0.012, 0.012))
                _, coda = VC.cons_dur(ps, voice)
                if last:
                    v_end = T0 + (beat + beats) * SPB - 0.25 - coda
                else:
                    nxt_on, _ = VC.cons_dur(ps_of(NOTES[ni + 1][0]), voice)
                    gap = 0.12 if (beat + beats) in PHRASE_END else 0.02
                    v_end = T0 + (beat + beats) * SPB - nxt_on - coda - gap
                vdur = max(0.18, v_end - t_on)
                best = None
                for attempt, cap in enumerate((pl["cap"], pl["cap"] - 3, pl["cap"] - 6)):
                    a, on, _ = VC.sing(ps, voice, vdur, midi, min(midi, cap), vib=(pl["rate"], pl["depth"] * 0.8, 1.0),
                                       scoop=-0.3, detune=pl["detune"] * 0.6, cons_mult=1.5,
                                       formant=1.0 + 0.1 * float(np.clip((midi - 64) / 10, 0, 1)))
                    med, p10, p90, vf = VC.note_cents(a, on, vdur, midi) if vdur > 0.4 else \
                        _short_cents(a, on, vdur, midi)
                    err = abs(med - pl["detune"] * 0.6) if np.isfinite(med) else 1e9
                    sc = err + (0.85 - vf) * 200 * (vf < 0.85)
                    if best is None or sc < best[0]:
                        best = (sc, a, on, med, vf, attempt)
                    if err <= 15 and vf >= 0.85:
                        break
                _, a, on, med, vf, attempt = best
                x = up48(a)
                tt = np.arange(len(x)) / SR - on
                lvl = -3.0 if last else -5.0 + 1.5 * (beats > 1)
                env = db(lvl + 1.0 * np.sin(np.pi * np.clip(tt / vdur, 0, 1)))
                add_at(bus, pan(x * env.astype(np.float32) * db(pl["gain"]), pl["pan"]), t_on - on)
                rows.append(dict(part=part, voice=voice, word=word, midi=midi, t_on=round(t_on, 3),
                                 vowel_dur=round(vdur, 3), median_cents=round(med, 1), voiced=round(vf, 2),
                                 attempt=attempt))
            print("hymn", part, voice, flush=True)
    return bus, rows


def _short_cents(a, on, vdur, midi):
    ts, hz = VL.pitch_track(a, VL.VSR, 50, 1100, 0.01)
    win = (ts > on + 0.06) & (ts < on + vdur - 0.03)
    sel = win & (hz > 0)
    if not sel.any():
        return float("nan"), 0, 0, 0.0
    c = 1200 * np.log2(hz[sel] / float(VL.midi_hz(midi)))
    return float(np.median(c)), 0, 0, float(sel.sum() / max(1, win.sum()))


def hup_times():
    ts = []
    for f in glob.glob(str(FILM_ROOT / "cache/foley/*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        ev = d.get("events", []) if isinstance(d, dict) else d
        ts += [(float(e["t"]), float(e.get("strength", 1.0))) for e in ev
               if isinstance(e, dict) and e.get("kind") == "raise"]
    if len(ts) >= 5:
        ts.sort()
        HUP_STRENGTH[:] = [s for _, s in ts]
        return [t for t, _ in ts], "T07Raised raise events (cache/foley/t07_hits_events.json)"
    a, b = TL_cue("raise_1"), TL_cue("raise_100")
    HUP_STRENGTH[:] = [1.0] * 100
    return [a + (b - a) * (k / 99) ** 0.6 for k in range(100)], "synthetic accelerating schedule (no raise events yet)"


HUP_STRENGTH: list[float] = []


def TL_cue(cid):
    return next(c["t"] for c in TL["cues"] if c["id"] == cid)


def hups():
    buf = np.zeros((NS, 2), np.float32)
    times, src = hup_times()
    voices = [v for v in VL.VOICES["a"] + VL.VOICES["b"] if v not in {"af_nicole", "bm_lewis"}]
    clips = []
    for i in range(16):
        x = yell("Hup!", voices[(i * 5) % len(voices)], 3.0 + (i % 4), 1.3, 0.9)
        x = x[int(max(0, onset(x) - 0.004) * SR):int((onset(x) + 0.32) * SR)]
        clips.append(x * np.minimum(1, (len(x) - np.arange(len(x))) / (0.06 * SR)).astype(np.float32))
    errs = []
    n = len(times)
    for k, t in enumerate(times):
        u = k / max(1, n - 1)
        ratio = 1.0 + 0.3 * u                              # rising with the acceleration
        x = resample(clips[k % len(clips)], 1000, int(1000 * ratio))
        o = onset(x)
        dens = 1.0 / max(0.03, (times[min(k + 1, n - 1)] - t) if k < n - 1 else 0.03)
        g = -2.0 - 8 * np.log10(max(1.0, dens / 3)) - 4.0 * (1.0 - HUP_STRENGTH[k])   # hero cuts louder
        add_at(buf, pan(x * db(g), float(np.sin(k * 2.39)) * 0.85), t - o)
        errs.append(abs((round((t - o) * SR) / SR + o) - t))
    return buf, dict(count=n, source=src, max_err_ms=round(1000 * max(errs), 3))


def final_flags():
    buf = np.zeros((NS, 2), np.float32)
    t_on, vdur = FINAL_T, 1.2
    chord = FINAL
    for part, voices in PARTS.items():
        for vi, voice in enumerate(voices[:2]):
            pl = VC.singer_plan(part, voice, vi)
            midi = chord[part]
            a, on, _ = VC.sing("flˈaɡz", voice, vdur, midi, min(midi, pl["cap"]), vib=(pl["rate"], pl["depth"], 2.0),
                               scoop=-0.3, detune=pl["detune"] * 0.6, cons_mult=1.5, swell=VC.final_swell)
            x = up48(a)
            tt = np.arange(len(x)) / SR - on
            x = x * db(-6 + 6 * smoothstep(tt / 0.8)).astype(np.float32)
            add_at(buf, pan(x, pl["pan"]), t_on - on)
    h = make_ir(rt60=2.2, predelay=0.03, lo_mult=1.0, hi_mult=0.6, er_n=8, width=1.0, seed=302)
    a0 = int(196.8 * SR)
    add_at(buf, convolve(buf[a0:], h)[: NS - a0] * db(-6), 196.8)
    t = np.arange(NS) / SR
    buf *= np.clip((198.95 - t) / 0.35, 0, 1)[:, None].astype(np.float32)
    return buf


def main():
    grid = {"key": "E-flat major", "meter": "3/4", "bpm": round(60 / SPB, 3), "beat_s": round(SPB, 5),
            "pickup_at": T0, "amen_men_at": AMEN,
            "notes": [dict(word=w, t=round(T0 + b * SPB, 4), beats=n, satb=p) for w, b, n, p in NOTES]}
    (VOCAL / "hymn_grid.json").write_text(json.dumps(grid, indent=1))
    hb, rows = hymn()
    h = make_ir(rt60=1.5, predelay=0.02, lo_mult=1.0, hi_mult=0.5, er_n=10, er_span=0.06, er_gain=0.35, width=1.0,
                seed=301)
    a0 = int(168.5 * SR)
    wet = convolve(hb[a0:int(182.0 * SR)], h) * db(-7)
    hb = hb * db(-2)
    add_at(hb, wet, 168.5)
    solo = hb.copy()
    t = np.arange(NS) / SR
    g = np.ones(NS, np.float32)                               # leave room for the prayer
    for ln in TL["lines"]:
        if 169.0 <= ln["start"] < 186.0 and ln["id"] != "S09":
            d = np.clip(np.minimum((t - ln["start"] + 0.1) / 0.1, (ln["end"] + 0.1 - t) / 0.25), 0, 1)
            g = np.minimum(g, (1 - d * (1 - db(-4))).astype(np.float32))
    hb *= g[:, None]
    hb *= np.clip((183.5 - t) / 1.5, 0, 1)[:, None].astype(np.float32)
    L = lufs(hb[int(169.0 * SR):int(181.0 * SR)])
    hb *= db(-25.0 - L)
    solo *= db(-25.0 - L)
    hp, hinfo = hups()
    L = lufs(hp[int(143.8 * SR):int(147.0 * SR)])
    hp *= db(-24.0 - L)
    ff = final_flags()
    L = lufs(ff[int(197.3 * SR):int(199.0 * SR)])
    ff *= db(-20.0 - L)
    mix = hb + hp + ff
    pk = np.abs(mix).max()
    if pk > db(-1.5):
        mix *= db(-1.5) / pk
    mix[-int(0.01 * SR):] = 0
    write_stem("choir", mix)
    # ---- verification
    bad = [r for r in rows if not (abs(r["median_cents"]) <= 25 and r["voiced"] >= 0.8)]
    errs = [abs(r["median_cents"]) for r in rows if np.isfinite(r["median_cents"])]
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly
    wm = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
    seg = solo[int(168.8 * SR):int(181.6 * SR)].mean(1)
    hyp = " ".join(s.text.strip() for s in wm.transcribe(resample_poly(seg, 1, 3).astype(np.float32),
                                                         language="en", beam_size=5)[0])
    lev = {k: round(float(lufs(mix[int(a * SR):int(b * SR)])), 1) for k, (a, b) in
           {"hymn_169-181": (169.0, 181.0), "hups_143.9-146.8": (143.9, 146.8), "flags_197.4-199": (197.4, 199.0)}.items()}
    report = dict(grid=grid, notes=rows, notes_outside_25c=len(bad), mean_abs_cents=round(float(np.mean(errs)), 1),
                  max_abs_cents=round(float(np.max(errs)), 1), whisper_hymn=hyp, hups=hinfo, levels=lev,
                  stem_lufs=round(float(lufs(mix)), 1))
    (VOCAL / "choir_notes.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("notes", "grid")}, indent=1))
    for r in bad:
        print("  off:", r)


if __name__ == "__main__":
    main()

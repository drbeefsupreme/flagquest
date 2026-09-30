"""OPVS SCHISMATICVM — builds films/opus/audio/stems/choir.wav: Kokoro voices made to SING.   Owner: Vocalist-2.
The notes are op_vocal_score.py (also written to audio/vocal/hymn_grid.json for the Composer's organ).

 155.0-163.0  THE CONVERGENCE: 30 voices in 9 languages rise out of the fallen tesserae speaking lore; their phoneme
              durations stretch (speech slows into chant) and their F0 glides, in the log domain, from each voice's own
              speech contour onto a tone of D major (D2 A2 D3 A3 D4 F#4 A4 D5), each voice ending on a held vowel:
              a murmur under N08, the chord blooming on 159.0 (convergence) as the Lector finishes "All of them
              together."  The upper voices then fall away one by one (160.4-162.6), leaving the D.
 161.8-168.6  THE ISON: four low voices hold D2+D3 on "o" with staggered breaths (Byzantine isokratema) under
 163.0-168.7  THE ANTIPHON: the Cantor alone, mode I, free rhythm: "Vexilla sint. Vexilla sunt. Vexilla erunt.
              Vexilla." (im_nicola made to chant: vowel onsets on the score, melismas, straight tone), under the dome.
 168.7-180.1  THE HYMN "Flags be. Flags are. Flags will. Flags." 84 BPM, 16 beats, 24 singers (6 per part) + divisi:
              organum that blossoms into triads — strict parallel fifths, then fifths over a D drone, then thirds, then
              the D major chord; every note a separate Kokoro forward pass (film 1's sing(): vowel held for the note,
              F0 on the MIDI pitch with scoop and a light late vibrato), vowel onsets on the beat; breaths between
              phrases; the dome answers.  The tail ducks under N09.
 207.6-209.9  "VEXILLO GRATIAS." the choir's response to the Explicit: I I I | IV IV | I (plagal), sixteen singers
              + divisi chanting the whole phrase in Latin; the basilica rings out to black.

usage: python films/opus/tools/op_vocal_choir.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Compressor, HighpassFilter, HighShelfFilter, LowShelfFilter, PeakFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import op_vocal_score as SC  # noqa: E402
from op_vocal_lib import (CACHE, FILM_ROOT, FR, NS, SR, TL, VOCAL, VSR, add_at, bandpass, chant, convolve, db,  # noqa: E402
                          dump, export_cantor, is_male, key, kokoro, limit, lufs, midi_hz, pan, pb, pb_st, rng_for, room, silence,
                          smoothstep, up48, vocab_ps, warp_sp, world_analyze, world_synth, write_stem, VOWELS)
from vocal_corpus import BABEL_TX, LORE  # noqa: E402

# film 1's singing engine reads timeline["hymn"] at import: give it this film's notes first
TL["hymn"].update(start=SC.HYMN_T0, bpm=SC.BPM, beats=SC.BEATS,
                  notes=[[w, b, n, satb] for w, b, n, satb, _ in SC.HYMN])
import vocal_choir as VC  # noqa: E402

PARTS = {
    "S": ["af_heart", "af_sky", "bf_emma", "bf_isabella", "af_aoede", "af_kore"],
    "A": ["af_sarah", "bf_alice", "bf_lily", "af_jessica", "af_nova", "af_river"],
    "T": ["am_michael", "am_eric", "am_puck", "am_liam", "bm_daniel", "am_echo"],
    "B": ["am_fenrir", "bm_george", "am_onyx", "am_adam", "em_santa", "bm_fable"],
}
PART_PAN = {"S": -0.5, "A": -0.16, "T": 0.16, "B": 0.5}
PART_GAIN = {"S": 0.0, "A": -0.5, "T": -0.5, "B": 0.5}
NOTE_DB = [-8.0, -8.5, -6.0, -6.0, -3.5, -3.0, -2.0]      # mp .. f .. the final ff swell
PHRASE_ENDS = {4, 8, 12, 16}


# ------------------------------------------------------------------------------------------------ convergence
CONV_START = 155.0
CONV_VOICES = [  # (voice, tone)  males low, females high; nine languages
    ("am_fenrir", 38), ("bm_george", 38), ("hm_omega", 45), ("am_onyx", 45), ("pm_alex", 45),
    ("am_adam", 50), ("im_nicola", 50), ("em_alex", 50), ("zm_yunxi", 50), ("jm_kumo", 57),
    ("am_michael", 57), ("bm_daniel", 57), ("hm_psi", 57), ("zm_yunjian", 50), ("am_liam", 57),
    ("af_heart", 62), ("bf_emma", 62), ("ef_dora", 62), ("ff_siwis", 66), ("if_sara", 66),
    ("pf_dora", 66), ("hf_alpha", 69), ("jf_alpha", 69), ("zf_xiaoxiao", 69), ("af_bella", 62),
    ("bf_isabella", 74), ("zf_xiaoyi", 74), ("jf_nezumi", 66), ("af_aoede", 69), ("hf_beta", 74),
]


def conv_text(voice, r):
    lang = voice[0]
    if lang in "ab":
        return LORE[int(r.integers(0, len(LORE)))]
    return BABEL_TX[lang][int(r.integers(0, len(BABEL_TX[lang])))]


def release_time(tone, r):
    if tone in (38, 50):
        return float(r.uniform(162.4, 163.3))        # the D stays longest (into the ison)
    if tone in (45, 57):
        return float(r.uniform(161.4, 162.6))
    return float(r.uniform(160.4, 162.1))


def converge_voice(voice, tone, idx):
    """one voice: lore -> slowed -> glided onto `tone`, held vowel until its release.  (stereo48k? no: mono48k, t0)"""
    r = rng_for("opus_conv", voice, idx)
    t_speech = CONV_START + float(r.uniform(0.0, 0.5))
    hold_at = 157.75 + float(r.uniform(-0.35, 0.3))
    rel = release_time(tone, r)
    k_ = key("oconv2", voice, tone, idx)
    path = CACHE / "conv" / f"{voice}_{k_}.wav"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        m = json.loads(meta.read_text())
        return sf.read(path, dtype="float32")[0], m["t_start"], m
    K = kokoro()
    text = conv_text(voice, r)
    ps = vocab_ps(K.phonemes(text, voice[0]))[:300]
    _, dn, _, _ = K.forward(ps, voice)
    t_start = t_speech - float(dn[0]) * FR
    cap = 57 if is_male(voice) else 62
    shift = max(0.0, tone - cap)

    def stretch(t):
        return 1.0 + 2.6 * smoothstep((t - 155.3) / 2.4)
    d_new, t = [int(dn[0])], t_speech
    cut = None
    for j in range(1, len(dn) - 1):
        c = ps[j - 1]
        if t >= hold_at and c in VOWELS and c not in "əɚɐᵊ":
            cut = j
            break
        dj = max(1, int(round(dn[j] * stretch(t))))
        d_new.append(dj)
        t += dj * FR
    if cut is None:
        cut = max(j for j in range(1, len(dn) - 1) if ps[j - 1] in VOWELS)
        d_new = d_new[:cut]
        t = t_start + sum(d_new) * FR
    ps2 = ps[:cut]
    hold = max(8, int(round((rel - t) / FR)))
    d_new = d_new[:cut] + [hold, 3]
    k = cut
    f_tgt = float(midi_hz(tone - shift + float(r.uniform(-5, 5)) / 100))
    cd = np.concatenate([[0], np.cumsum(d_new)]) * 2
    v0, v1 = int(cd[k]), int(cd[k + 1])
    tf = t_start + np.arange(int(cd[-1])) / 80.0
    w = smoothstep((tf - 155.2) / (157.9 - 155.2)) ** 1.3
    w[v0:] = np.maximum(w[v0:], smoothstep((tf[v0:] - tf[v0]) / 0.5))
    vr = float(r.uniform(4.8, 5.8))

    def f0_fn(F0, _d):
        voiced = F0 > 40
        med = np.median(F0[voiced]) if voiced.any() else 150.0
        nat = np.where(voiced, F0, med)
        vib = 0.1 * np.sin(2 * np.pi * vr * (tf - tf[v0])) * np.clip((tf - tf[v0] - 0.5) / 0.8, 0, 1)
        lf = (1 - w) * np.log(nat) + w * np.log(f_tgt * 2 ** (vib / 12))
        out = np.where(voiced, np.exp(lf), F0)
        out[v0 + 3:v1 - 2] = np.exp(lf[v0 + 3:v1 - 2])
        return out

    def n_fn(N, _d):
        pl = float(np.percentile(N[:v0], 92)) if v0 > 8 else float(N.max())
        i = np.arange(len(N))
        m = np.clip((i - (v0 + 2)) / 8, 0, 1) * np.clip((v1 - i) / 10, 0, 1)
        sm = np.convolve(np.pad(N, 6, mode="edge"), np.ones(13) / 13, mode="valid")
        return (1 - m) * N + m * np.maximum(sm, pl - 0.3)
    a, d, _, _ = K.forward(ps2, voice, dur_fn=lambda _d: np.array(d_new), f0_fn=f0_fn, n_fn=n_fn)
    if shift > 0:
        f0, tt, sp, ap = world_analyze(a, VSR, f0_floor=55, f0_ceil=1000)
        ww = np.interp(t_start + tt, tf, w)
        sp = warp_sp(sp, 1.0 + 0.1 * float(np.clip((tone - 64) / 12, 0, 1)))
        a = world_synth(f0 * 2 ** (shift * ww / 12), sp, ap, VSR)[:len(a)]
    x = up48(a)
    info = dict(t_start=t_start, text=text, tone=tone, hold_at=round(tf[v0], 3), release=round(rel, 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, SR, subtype="FLOAT")
    meta.write_text(json.dumps(info))
    return x, t_start, info


def render_convergence():
    buf = np.zeros((NS, 2), np.float32)
    info = []
    for i, (voice, tone) in enumerate(CONV_VOICES):
        x, t0, inf = converge_voice(voice, tone, i)
        r = rng_for("opus_convmix", voice, i)
        tg = t0 + np.arange(len(x)) / SR
        # a murmur under N08, blooming into the chord on 159.0; each voice fades on its own release
        env = db(-27 + 24 * smoothstep((tg - 155.0) / 3.9) ** 1.8 - 5.0 * smoothstep((tg - 156.4) / 0.3)
                 * smoothstep((158.95 - tg) / 0.25))
        rel = inf["release"]
        env *= np.clip((rel + 0.05 - tg) / 0.7, 0, 1) ** 1.3
        x = x * env.astype(np.float32)
        p = float(np.clip((tone - 56) / 20 * -0.8 + r.uniform(-0.3, 0.3), -0.9, 0.9))
        p0 = float(r.uniform(-0.95, 0.95))
        pp = p0 + (p - p0) * smoothstep((tg - 155.8) / 3.0)
        add_at(buf, pan(x, pp), t0)
        info.append(dict(voice=voice, tone=tone, t0=round(t0, 3), hold_at=inf["hold_at"], release=rel,
                         text=inf["text"]))
        print(f"conv {voice} -> {tone}", flush=True)
    return buf, info


# ------------------------------------------------------------------------------------------------ ison
ISON_SINGERS = [("am_onyx", 38), ("am_adam", 50), ("bm_george", 50), ("am_fenrir", 38)]


def render_ison():
    buf = np.zeros((NS, 2), np.float32)
    rows = []
    a0, a1 = SC.ISON_SPAN
    for i, (voice, midi) in enumerate(ISON_SINGERS):
        r = rng_for("ison", voice, i)
        t = a0 - 0.3 + 0.35 * i + float(r.uniform(0, 0.2))
        while t < a1 - 0.8:
            L = min(float(r.uniform(2.6, 3.8)), a1 + 0.1 - t)
            a, info = chant("ˈo", voice, [dict(t=0.0, notes=[(midi, L)])], cons_mult=1.0, scoop=-0.2, drift=4.0,
                            seed=int(t * 10), energy=-0.3)
            x = up48(a)
            tt = np.arange(len(x)) / SR - info["v0"]
            env = np.clip(tt / 0.45, 0, 1) * np.clip((L - tt) / 0.5, 0, 1)
            add_at(buf, pan(x * env.astype(np.float32) * db(-3.0 if midi == 38 else -1.0),
                            float(r.uniform(-0.3, 0.3))), t - info["v0"])
            rows.append(dict(voice=voice, midi=midi, t=round(t, 3), dur=round(L, 2),
                             cents=info["cents"][0][1] if info.get("cents") else None))
            t += L - float(r.uniform(0.6, 0.9))            # the next breath overlaps the others
    tg = np.arange(NS) / SR
    buf *= (np.clip((tg - a0) / 1.2, 0, 1) * np.clip((a1 - tg) / 0.4, 0, 1))[:, None].astype(np.float32)
    return buf, rows


# ------------------------------------------------------------------------------------------------ antiphon
def render_antiphon():
    a, info = chant(SC.ANTIPHON_PS, "im_nicola", SC.sched(SC.ANTIPHON), cons_mult=0.8, cons_max=2, seed=11)
    x = up48(a)
    y = pb(x, HighpassFilter(75), LowShelfFilter(220, 1.5, 0.7), PeakFilter(2600, 1.0, 1.0),
           HighShelfFilter(9000, 1.0, 0.7), Compressor(-24, 2.0, 15, 200))
    start = SC.ANTIPHON_T0 - info["v0"]
    buf = np.zeros((NS, 2), np.float32)
    add_at(buf, pan(y, 0.0), start)
    syl = [dict(syl=s[0], t=round(SC.ANTIPHON_T0 + info["onsets"][k] - info["v0"], 3),
                end=round(SC.ANTIPHON_T0 + info["ends"][k] - info["v0"], 3)) for k, s in enumerate(SC.ANTIPHON)]
    phrases = [dict(text=" ".join(s["syl"] for s in syl[a_:b_]), t=syl[a_]["t"], end=syl[b_ - 1]["end"])
               for a_, b_ in SC.ANTIPHON_PHRASES]
    notes = []
    k = 0
    for s in SC.ANTIPHON:
        for m, _ in s[2]:
            c = info["cents"][k]
            notes.append(dict(syl=s[0], midi=m, cents=c[1], voiced=c[2]))
            k += 1
    return buf, syl, phrases, notes


# ------------------------------------------------------------------------------------------------ hymn
def note_render(ps, voice, pl, midi, vdur, last, vib_depth=0.55):
    """film 1's sing() with its register fall-back (sub-harmonic guard) and pitch check"""
    formant = 1.0 + 0.12 * float(np.clip((midi - 64) / 10, 0, 1))
    r = rng_for("opus_note", voice, midi, round(vdur, 3))
    vib = (pl["rate"], pl["depth"] * vib_depth, float(r.uniform(0, 6.28)))
    scoop = float(r.uniform(-0.45, -0.2))
    cmax = 4 if last else VC.CODA_MAX
    best = None
    for attempt, cap in enumerate((pl["cap"], pl["cap"] - 3, pl["cap"] - 6, pl["cap"] - 9)):
        rm = min(midi, cap)
        a, on, _ = VC.sing(ps, voice, vdur, midi, rm, vib=vib, scoop=scoop, detune=pl["detune"],
                           n_off=-0.5 + (0.3 if last else 0.0), swell=VC.final_swell if last else None,
                           formant=formant, coda_max=cmax)
        med, p10, p90, vf = VC.note_cents(a, on, vdur, midi) if vdur > 0.45 else short_cents(a, on, vdur, midi)
        err = abs(med - pl["detune"]) if np.isfinite(med) else 1e9
        score = err + (0.9 - vf) * 200 * (vf < 0.9)
        if best is None or score < best[0]:
            best = (score, a, on, rm, med, vf, attempt)
        if err <= 12 and vf >= 0.9:
            break
    return best


def short_cents(a, on, vdur, midi):
    ts, hz = VC.pitch_track(a, VSR, 50, 1100, 0.01)
    win = (ts > on + 0.06) & (ts < on + vdur - 0.03)
    sel = win & (hz > 0)
    if not sel.any():
        return float("nan"), 0.0, 0.0, 0.0
    c = 1200 * np.log2(hz[sel] / float(midi_hz(midi)))
    return float(np.median(c)), 0.0, 0.0, float(sel.sum() / max(1, win.sum()))


def render_hymn():
    bus = {p: np.zeros((NS, 2), np.float32) for p in PARTS}
    rows = []
    T0, SPB = SC.HYMN_T0, SC.SPB
    END = SC.HYMN_END
    for part, voices in PARTS.items():
        pi = "SATB".index(part)
        for vi, voice in enumerate(voices):
            pl = VC.singer_plan(part, voice, vi)
            r = rng_for("opus_hymn", voice)
            for ni, (word, beat, beats, satb, div) in enumerate(SC.HYMN):
                ps = SC.HYMN_PS[word]
                midi = satb[pi]
                last = ni == len(SC.HYMN) - 1
                if last and div:                          # divisi on the final chord: half the section splits
                    if part == "S" and vi % 2 == 1:
                        midi = div[0]
                    if part == "B" and vi % 2 == 1:
                        midi = div[1]
                t_on = T0 + beat * SPB + float(r.uniform(-0.012, 0.012)) + pl["bias"]
                end_beat = beat + beats
                _, coda = VC.cons_dur(ps, voice, 1.35, 4 if last else VC.CODA_MAX)
                if last:
                    v_end = END - coda
                else:
                    nxt_on, _ = VC.cons_dur(SC.HYMN_PS[SC.HYMN[ni + 1][0]], voice)
                    gap = 0.17 if end_beat in PHRASE_ENDS else 0.07     # a breath / clear word separation
                    v_end = T0 + end_beat * SPB - nxt_on - coda - gap
                vdur = max(0.2, v_end - t_on)
                _, a, on, rm, med, vf, attempt = note_render(ps, voice, pl, midi, vdur, last)
                x = up48(a)
                tt = np.arange(len(x)) / SR - on
                if last:
                    env = db(NOTE_DB[ni] + 5.0 * smoothstep(tt / (vdur * 0.85)))
                else:
                    env = db(NOTE_DB[ni] + 1.2 * np.sin(np.pi * np.clip(tt / vdur, 0, 1)))
                x = x * env.astype(np.float32) * db(pl["gain"] + PART_GAIN[part])
                add_at(bus[part], pan(x, pl["pan"]), t_on - on)
                rows.append(dict(part=part, voice=voice, note=ni, word=word, midi=midi, render_midi=rm,
                                 attempt=attempt, vowel_onset=round(t_on, 4), vowel_dur=round(vdur, 3),
                                 detune=round(pl["detune"], 1), cents=round(med, 1), voiced=round(vf, 2)))
            print(f"hymn {part} {voice} done", flush=True)
    return bus, rows


def breaths(buf):
    for ph_beat in (0, 4, 8, 12):
        for part, voices in PARTS.items():
            for voice in voices:
                r = rng_for("opus_breath", voice, ph_beat)
                dur = float(r.uniform(0.18, 0.3)) if ph_beat else float(r.uniform(0.3, 0.42))
                n = int(dur * SR)
                x = bandpass(r.standard_normal(n).astype(np.float32), 600, 3800, order=2)
                x = bandpass(x, 900 if is_male(voice) else 1300, None, order=1)
                env = np.sin(np.linspace(0, np.pi, n)) ** 1.5 * np.linspace(0.6, 1.0, n)
                x = x * env.astype(np.float32) * db(-47 if ph_beat else -45)
                t = SC.HYMN_T0 + ph_beat * SC.SPB - 0.13 - dur + float(r.uniform(-0.04, 0.02))
                add_at(buf, pan(x, PART_PAN[part] + float(r.uniform(-0.15, 0.15))), t)


# ------------------------------------------------------------------------------------------------ gratias
GRATIAS_SINGERS = {"S": PARTS["S"][:4], "A": PARTS["A"][:4], "T": PARTS["T"][:4] + ["am_puck"],
                   "B": PARTS["B"][:4] + ["am_onyx"]}


def render_gratias():
    buf = np.zeros((NS, 2), np.float32)
    rows = []
    for part, voices in GRATIAS_SINGERS.items():
        pi = "SATB".index(part)
        for vi, voice in enumerate(voices):
            pl = VC.singer_plan(part, voice, vi)
            r = rng_for("opus_gratias", voice, vi)
            sched = []
            for k, (syl, t, du, satb) in enumerate(SC.GRATIAS):
                m = satb[pi]
                if k == len(SC.GRATIAS) - 1 and vi == len(voices) - 1 and part in "TB":     # divisi final
                    m = SC.GRATIAS_DIV[0] if part == "B" else SC.GRATIAS_DIV[1]
                e = dict(t=t, notes=[(m, du)])
                if k == len(SC.GRATIAS) - 1:
                    e["gain"] = 0.3
                sched.append(e)
            cap = int(pl["cap"])
            a, info = chant(SC.GRATIAS_PS, voice, sched, cap=cap, cons_mult=0.9, cons_max=2, scoop=-0.25,
                            drift=4.0, seed=vi, formant=1.0 + 0.1 * float(np.clip((max(
                                n[0] for s in sched for n in s["notes"]) - 64) / 10, 0, 1)))
            x = up48(a)
            tt = np.arange(len(x)) / SR - info["v0"]
            env = db(-3.0 + 2.0 * smoothstep((tt - 0.9) / 0.6) - 1.5 * smoothstep((tt - 1.6) / 0.7))
            x = x * env.astype(np.float32) * db(pl["gain"] + PART_GAIN[part])
            t0 = SC.GRATIAS_T0 + float(r.uniform(-0.015, 0.015)) - info["v0"]
            add_at(buf, pan(x, pl["pan"]), t0)
            for k, c in enumerate(info["cents"]):
                rows.append(dict(part=part, voice=voice, syl=SC.GRATIAS[k][0], midi=c[0], cents=c[1], voiced=c[2]))
            print(f"gratias {part} {voice} done", flush=True)
    return buf, rows


# ------------------------------------------------------------------------------------------------ main
def main():
    dump("hymn_grid.json", SC.grid())
    conv, conv_info = render_convergence()
    ison, ison_rows = render_ison()
    anti, syl, phrases, anti_notes = render_antiphon()
    export_cantor("antiphon", syl)
    (FILM_ROOT / "cache" / "foley" / "cantor_phrases.json").write_text(json.dumps(phrases, indent=1))
    bus, hymn_rows = render_hymn()
    grat, grat_rows = render_gratias()
    t = np.arange(NS) / SR
    # --- 155-168.7: the convergence and the ison, under the dome; the cantor on top
    h_dome, _ = room("dome")
    conv = pb_st(conv, HighpassFilter(55), PeakFilter(3000, 1.0, 1.0), HighShelfFilter(8000, 1.0, 0.7))
    conv *= np.float32(db(-21.0 - lufs(conv[int(159.0 * SR):int(161.0 * SR)])))
    ison = pb_st(ison, HighpassFilter(45))
    ison *= np.float32(db(-31.0 - lufs(ison[int(164.0 * SR):int(168.0 * SR)])))
    low = conv + ison
    wet = convolve(low[int(154.5 * SR):int(169.5 * SR)], h_dome) * db(-5.0)
    add_at(low, wet, 154.5)
    anti *= np.float32(db(-20.0 - lufs(anti[int(163.0 * SR):int(168.7 * SR)])))
    aw = convolve(anti[int(162.5 * SR):int(169.2 * SR)], h_dome) * db(-3.0)
    add_at(anti, aw, 162.5)
    # --- the hymn
    dry = np.zeros((NS, 2), np.float32)
    for p in PARTS:
        dry += bus[p]
    breaths(dry)
    parts_dir = VOCAL / "choir_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    a0, a1 = int(154.0 * SR), int(182.0 * SR)
    for name, b in [("convergence", conv), ("ison", ison), ("antiphon", anti)] + [(p, bus[p]) for p in PARTS]:
        sf.write(parts_dir / f"{name}.wav", b[a0:a1], SR, subtype="FLOAT")
    dry = pb_st(dry, PeakFilter(3500, 3.0, 0.8), HighShelfFilter(7500, 1.5, 0.7))     # diction
    dry *= np.float32(db(-19.0 - lufs(dry[int(168.7 * SR):int(180.1 * SR)])))
    hw = convolve(dry[int(168.0 * SR):int(181.0 * SR)], h_dome) * db(-6.0)
    hymn = dry.copy()
    add_at(hymn, hw, 168.0)
    g = np.ones(NS, np.float32)                     # the dome's tail steps back for the Lector (N09, N10)
    for ln in TL["lines"]:
        if ln["id"] in ("N09", "N10"):
            d = np.clip(np.minimum((t - ln["start"] + 0.3) / 0.25, (ln["end"] + 0.3 - t) / 0.4), 0, 1)
            g = np.minimum(g, (1 - d * (1 - db(-10))).astype(np.float32))
    hymn *= g[:, None]
    hymn *= np.clip((187.0 - t) / 2.0, 0, 1)[:, None].astype(np.float32)
    # --- gratias
    grat = pb_st(grat, HighpassFilter(60))
    grat *= np.float32(db(-19.0 - lufs(grat[int(207.5 * SR):int(210.0 * SR)])))
    h_b, _ = room("basilica")
    gw = convolve(grat[int(207.2 * SR):int(210.6 * SR)], h_b) * db(-4.0)
    add_at(grat, gw, 207.2)
    grat *= np.clip((211.95 - t) / 1.6, 0, 1)[:, None].astype(np.float32) ** 1.5
    mix = low + anti + hymn + grat
    mix[:int(154.5 * SR)] = 0
    silence(mix)
    mix = limit(mix, -2.0)
    mix[-int(0.02 * SR):] = 0
    write_stem("choir", mix)
    for name, (x0, x1) in {"convergence_155-159": (155.0, 159.0), "chord_159-161": (159.0, 161.0),
                           "ison+cantor_163-168.7": (163.0, 168.7), "hymn_168.7-180.1": (168.7, 180.1),
                           "gratias_207.6-209.9": (207.6, 209.9)}.items():
        print(f"  {name:22s} {lufs(mix[int(x0 * SR):int(x1 * SR)]):6.1f} LUFS", flush=True)
    bad = [r for r in hymn_rows if not (np.isfinite(r["cents"]) and abs(r["cents"]) <= 25 and r["voiced"] >= 0.8)]
    print(f"hymn: {len(hymn_rows)} notes, {len(bad)} outside +-25 cents / voiced < 80%", flush=True)
    for r in bad:
        print("   ", r, flush=True)
    ac = [abs(n["cents"]) for n in anti_notes if n["cents"] is not None and np.isfinite(n["cents"])]
    print(f"antiphon: {len(anti_notes)} notes, mean |cents| {np.mean(ac):.1f}, max {np.max(ac):.1f}", flush=True)
    gc = [abs(r["cents"]) for r in grat_rows if r["cents"] is not None and np.isfinite(r["cents"])]
    print(f"gratias: {len(grat_rows)} notes, mean |cents| {np.mean(gc):.1f}, max {np.max(gc):.1f}", flush=True)
    dump("choir_notes.json", dict(antiphon=anti_notes, antiphon_syllables=syl, antiphon_phrases=phrases,
                                  hymn=hymn_rows, gratias=grat_rows, ison=ison_rows, convergence=conv_info))


if __name__ == "__main__":
    main()

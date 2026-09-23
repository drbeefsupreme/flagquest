"""Locks the master timeline: scene boundaries, dialogue placement, sync cues.
Writes timeline.json, audio/voice/mouth.npz (lip-sync envelopes, 100 Hz per speaker)
and audio/stems/dialogue_dry.wav (48 kHz stereo, dry voices placed on the timeline).
usage: python tools/build_timeline.py"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
FPS, SR = 24, 48000

SCENES = [
    # id, module, start, end, title
    ("s01", "scenes.s01_crystal",   0.0,  21.0, "Rotating Into Existence"),
    ("s02", "scenes.s02_slop",      21.0, 47.5, "The Age of Slop"),
    ("s03", "scenes.s03_jaguar",    47.5, 61.5, "The Noospheric Munitions Act"),
    ("s04", "scenes.s04_flagartha", 61.5, 90.0, "Flagartha: the Hypermind"),
    ("s05a", "scenes.s05a_breach",  90.0, 100.5, "The Schismmancers"),
    ("s05b", "scenes.s05b_balkan",  100.5, 125.5, "Recursive Balkanization"),
    ("s06", "scenes.s06_babel",     125.5, 139.0, "Babel"),
    ("s07a", "scenes.s07a_plant",   139.0, 154.5, "The Unbabeling"),
    ("s07b", "scenes.s07b_flagistan", 154.5, 173.0, "Flagistan"),
    ("s08", "scenes.s08_sticky",    173.0, 193.0, "The Sticky Flag"),
]

PLACE = {  # line id -> global start seconds
    "N01": 2.6, "N02": 5.9, "N03": 10.3,
    "N04": 22.0, "N05": 28.4, "N06": 33.6, "N07": 40.6,
    "J01": 48.3, "J02": 58.0,
    "H01": 63.0, "H02": 71.2, "L01": 74.8, "C01": 76.3, "L02": 79.2, "C02": 83.4,
    "K01": 91.2, "K02": 96.1, "K03": 98.4, "K04": 100.6, "X01": 103.2, "X02": 105.7,
    "K05": 107.8, "X03": 109.9, "X04": 110.8, "X05": 112.1,
    "K06": 114.6, "X06": 117.9, "X07": 121.0, "K07": 123.3,
    "L03": 130.2, "C03": 133.8,
    "C04": 143.8, "N08": 149.6, "N09": 166.3, "N10": 170.9,
    "L04": 174.0, "C05": 175.9, "L05": 179.3, "C06": 182.3, "N11": 185.0,
}

CUES = [
    (0.0,   "film_start",       "Black. Sub-bass hum fades in; points of light."),
    (12.0,  "flag_rotates_in",  "The Flag appears as a 4D cross-section of the Crystal (word 'rotated')."),
    (14.2,  "flag_snap_open",   "First wind gust: the Flag snaps open. First big flap."),
    (16.0,  "title_slam",       "Title card THE UNBABELING slams in (IVG grid)."),
    (21.0,  "s02_start",        "Grey world of monolithic memeplexes."),
    (33.6,  "slop_appears",     "'Then came the slop.' A tide rises on the horizon."),
    (39.2,  "slop_crash",       "The slop wave crashes over the memeplexes."),
    (47.5,  "tv_on",            "CRT TV clicks on: President Jaguar."),
    (57.6,  "pen_sign",         "Jaguar signs the Act (pen scratch)."),
    (60.9,  "ka_ching",         "A pricecoin slides into his pocket. Ka-ching."),
    (61.5,  "descent",          "Descent into Flagartha, the Geomantic Command Center."),
    (72.7,  "hypermeme",        "The hypermeme materializes: it is a plain yellow Flag."),
    (90.0,  "gear_up",          "Schismmancers gear up (velcro, clicks, telescoping poles)."),
    (99.7,  "breach",           "BREACH: door blown, yellow light floods in."),
    (100.6, "nation_fracture",  "Map of a Nation fractures recursively into nations of one."),
    (107.8, "faith_fracture",   "A Temple fractures into cabals, then individuals."),
    (114.6, "philo_shatter",    "A colossal marble philosopher head shatters into the void."),
    (125.5, "babel_rise",       "The shards and slop spiral into a new Tower of Babel. Cacophony."),
    (136.0, "babel_crack",      "The tower cracks under its own babble."),
    (138.5, "silence",          "HARD CUT TO SILENCE (black)."),
    (142.0, "flag_planted",     "Lutie plants the Flag in the rubble (thunk), it unfurls."),
    (150.0, "convergence",      "Everyone looks up and raises a Flag; babble pitch-glides into one chord."),
    (154.5, "hymn",             "Hymn: 'Flags be. Flags are. Flags will. Flags.' 84 BPM, 16 beats, D major."),
    (165.93,"hymn_end",         "Hymn final chord lands (sustain)."),
    (166.3, "gateless_gate",    "The 10,000-foot Flag passes through the gateless gate."),
    (173.0, "s08_start",        "Morning. Field of flags. The sticky koan."),
    (188.3, "glorious",         "Crowd: 'GLORIOUS!' (the Signifier's salute)."),
    (189.5, "end_card",         "IVG end card."),
    (193.0, "film_end",         "End."),
]

HYMN = {
    "start": 154.5, "bpm": 84, "key": "D major", "beats": 16,
    "words": ["Flags", "be", "Flags", "are", "Flags", "will", "Flags"],
    # (word, beat_offset, beats, SATB midi notes)
    "notes": [
        ["Flags", 0, 2, [69, 66, 62, 50]], ["be", 2, 2, [66, 62, 57, 50]],
        ["Flags", 4, 2, [71, 67, 62, 43]], ["are", 6, 2, [67, 64, 59, 52]],
        ["Flags", 8, 2, [73, 64, 57, 45]], ["will", 10, 2, [69, 64, 61, 45]],
        ["Flags", 12, 4, [74, 69, 66, 50]],
    ],
}


def snap(t):
    return round(t * FPS) / FPS


def main():
    spec = json.loads((ROOT / "script/lines.json").read_text())
    idx = json.loads((ROOT / "audio/voice/raw/index.json").read_text())
    total = SCENES[-1][3]
    scenes = []
    for sid, mod, s, e, title in SCENES:
        scenes.append({"id": sid, "module": mod, "start": snap(s), "end": snap(e),
                       "f0": int(round(s * FPS)), "f1": int(round(e * FPS)), "title": title})
    lines = []
    n = int(total * SR) + SR
    dry = np.zeros(n, np.float32)
    env_sr = 100
    speakers = sorted({ln["speaker"] for ln in spec["lines"]})
    mouth = {sp: np.zeros(int(total * env_sr) + env_sr, np.float32) for sp in speakers}
    sos = butter(4, [250, 3500], btype="band", fs=24000, output="sos")
    for ln in spec["lines"]:
        lid = ln["id"]
        info = idx[lid]
        start = PLACE[lid]
        end = start + info["dur"]
        sc = next(x for x in scenes if x["start"] <= start < x["end"])
        assert sc["id"] == ln["scene"], (lid, sc["id"], ln["scene"])
        if end > sc["end"] + 0.35:
            print(f"WARNING {lid} overruns scene {sc['id']} by {end - sc['end']:.2f}s")
        a24, _ = sf.read(ROOT / f"audio/voice/raw/{lid}.wav", dtype="float32")
        a48 = resample_poly(a24, 2, 1).astype(np.float32)
        i0 = int(start * SR)
        dry[i0:i0 + len(a48)] += a48
        # lip-sync envelope: band-limited RMS, 100 Hz, fast attack, slower release, normalised per line
        b = sosfiltfilt(sos, a24)
        hop = 240
        frames = len(b) // hop
        rms = np.sqrt(np.mean(b[:frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
        ref = np.percentile(rms, 95) + 1e-9
        o = np.clip((rms / ref), 0, 1.2) ** 0.7
        o = np.clip((o - 0.08) / 0.92, 0, 1)
        sm = np.zeros_like(o)
        v = 0.0
        for k, x in enumerate(o):
            v = v + (x - v) * (0.65 if x > v else 0.30)
            sm[k] = v
        j0 = int(start * env_sr)
        m = mouth[ln["speaker"]]
        m[j0:j0 + len(sm)] = np.maximum(m[j0:j0 + len(sm)], sm[: len(m) - j0])
        words = [{"w": w["w"], "s": round(start + w["s"], 3), "e": round(start + w["e"], 3)} for w in info["words"]]
        text = ln["text"]
        import re
        clean = re.sub(r"\[([^\]]+)\]\(/[^)]*/\)", r"\1", text)
        lines.append({"id": lid, "scene": sc["id"], "speaker": ln["speaker"], "text": clean,
                      "start": round(start, 3), "end": round(end, 3), "dur": info["dur"],
                      "file": f"audio/voice/raw/{lid}.wav", "words": words})
    # overlap check per speaker
    for sp in speakers:
        ls = sorted([l for l in lines if l["speaker"] == sp], key=lambda l: l["start"])
        for a, b in zip(ls, ls[1:]):
            if b["start"] < a["end"]:
                print("OVERLAP", sp, a["id"], b["id"])
    tl = {
        "title": "THE UNBABELING", "fps": FPS, "width": 1920, "height": 1080, "sr": SR,
        "duration": total, "frames": int(round(total * FPS)),
        "scenes": scenes,
        "lines": lines,
        "cast": spec["cast"],
        "cues": [{"t": t, "id": i, "desc": d} for t, i, d in CUES],
        "hymn": HYMN,
    }
    (ROOT / "timeline.json").write_text(json.dumps(tl, indent=1, ensure_ascii=False))
    np.savez_compressed(ROOT / "audio/voice/mouth.npz", sr=env_sr, **mouth)
    (ROOT / "audio/stems").mkdir(parents=True, exist_ok=True)
    st = np.stack([dry, dry], 1)[: int(total * SR)]
    sf.write(ROOT / "audio/stems/dialogue_dry.wav", st * 0.9, SR, subtype="FLOAT")
    for sc in scenes:
        ls = [l for l in lines if l["scene"] == sc["id"]]
        print(f"{sc['id']} {sc['start']:7.2f}-{sc['end']:7.2f} ({sc['end']-sc['start']:5.1f}s) lines={len(ls)} {sc['title']}")
    print("total", total, "s,", tl["frames"], "frames")


if __name__ == "__main__":
    main()

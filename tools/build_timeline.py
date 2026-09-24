"""Locks a film's master timeline from its plan: scene boundaries, dialogue placement, sync cues.

Reads  <film>/script/plan.json   (scenes, place, cues, extra, mix, subs)
       <film>/script/lines.json  (cast + lines; `text` = on-screen/display text, optional `say` = TTS text)
       <film>/audio/voice/raw/index.json (+ <ID>.wav) from tools/tts.py
Writes <film>/timeline.json, <film>/audio/voice/mouth.npz (lip-sync envelopes, 100 Hz per speaker)
       and <film>/audio/stems/dialogue_dry.wav (48 kHz stereo, dry voices placed on the timeline).
The film is selected with VX_FILM (see vx/config.py).  usage: python tools/build_timeline.py"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vx.config import FPS, SR, FILM, FILM_ROOT, SCRIPT, AUDIO, TIMELINE  # noqa: E402

MARKUP = re.compile(r"\[([^\]]+)\]\(/[^)]*/\)")      # misaki phoneme override: [word](/phonemes/)
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")   # a spoken token as it appears in display text


def snap(t):
    return round(t * FPS) / FPS


def spoken(tokens):
    return [w for w in tokens if any(ch.isalnum() for ch in w["w"])]


def main():
    plan = json.loads((SCRIPT / "plan.json").read_text())
    spec = json.loads((SCRIPT / "lines.json").read_text())
    idx = json.loads((AUDIO / "voice/raw/index.json").read_text())
    total = plan["scenes"][-1][3]
    scenes = []
    for sid, mod, s, e, title in plan["scenes"]:
        scenes.append({"id": sid, "module": mod, "start": snap(s), "end": snap(e),
                       "f0": int(round(s * FPS)), "f1": int(round(e * FPS)), "title": title})
    lines = []
    dry = np.zeros(int(total * SR) + SR, np.float32)
    env_sr = 100
    speakers = sorted({ln["speaker"] for ln in spec["lines"]})
    mouth = {sp: np.zeros(int(total * env_sr) + env_sr, np.float32) for sp in speakers}
    sos = butter(4, [250, 3500], btype="band", fs=24000, output="sos")
    for ln in spec["lines"]:
        lid = ln["id"]
        info = idx[lid]
        start = plan["place"][lid]
        end = start + info["dur"]
        sc = next(x for x in scenes if x["start"] <= start < x["end"])
        assert sc["id"] == ln["scene"], (lid, sc["id"], ln["scene"])
        if end > sc["end"] + 0.35:
            print(f"WARNING {lid} overruns scene {sc['id']} by {end - sc['end']:.2f}s")
        a24, _ = sf.read(AUDIO / f"voice/raw/{lid}.wav", dtype="float32")
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
        display = MARKUP.sub(r"\1", ln["text"])
        # map every spoken TTS token to its display word (same order) so lettering can type on in sync
        dwords = WORD.findall(display)
        sp = spoken(words)
        if len(dwords) == len(sp):
            for w, d in zip(sp, dwords):
                w["d"] = d
        elif "say" in ln:
            print(f"WARNING {lid}: display has {len(dwords)} words, speech has {len(sp)}; no word mapping")
        lines.append({"id": lid, "scene": sc["id"], "speaker": ln["speaker"], "text": display,
                      "start": round(start, 3), "end": round(end, 3), "dur": info["dur"],
                      "file": str((AUDIO / f"voice/raw/{lid}.wav").relative_to(FILM_ROOT)), "words": words,
                      **({"meta": ln["meta"]} if "meta" in ln else {})})
    for sp in speakers:
        ls = sorted([l for l in lines if l["speaker"] == sp], key=lambda l: l["start"])
        for a, b in zip(ls, ls[1:]):
            if b["start"] < a["end"]:
                print("OVERLAP", sp, a["id"], b["id"])
    tl = {
        "title": plan["title"], "fps": FPS, "width": 1920, "height": 1080, "sr": SR,
        "duration": total, "frames": int(round(total * FPS)),
        "scenes": scenes,
        "lines": lines,
        "cast": spec["cast"],
        "cues": [{"t": t, "id": i, "desc": d} for t, i, d in plan["cues"]],
        **plan.get("extra", {}),
    }
    TIMELINE.write_text(json.dumps(tl, indent=1, ensure_ascii=False))
    np.savez_compressed(AUDIO / "voice/mouth.npz", sr=env_sr, **mouth)
    (AUDIO / "stems").mkdir(parents=True, exist_ok=True)
    st = np.stack([dry, dry], 1)[: int(total * SR)]
    sf.write(AUDIO / "stems/dialogue_dry.wav", st * 0.9, SR, subtype="FLOAT")
    for sc in scenes:
        ls = [l for l in lines if l["scene"] == sc["id"]]
        print(f"{sc['id']:5s} {sc['start']:7.2f}-{sc['end']:7.2f} ({sc['end']-sc['start']:5.1f}s) lines={len(ls):2d} {sc['title']}")
    print(f"[{FILM}] total {total} s, {tl['frames']} frames -> {TIMELINE}")


if __name__ == "__main__":
    main()

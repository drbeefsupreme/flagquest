"""Intelligibility QA of the final mix: transcribe every dialogue line out of the mixed audio and
compare with the script (word error rate).  usage: python tools/qa_dialogue.py [audio/final_mix.wav]"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
TL = json.loads((ROOT / "timeline.json").read_text())


def norm(s):
    s = s.lower().replace("twenty forty-two", "2042").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]", "", s).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(1, len(r))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "audio/final_mix.wav"
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = a.mean(axis=1)
    m = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
    tot, bad = [], []
    for l in TL["lines"]:
        seg = mono[int((l["start"] - 0.15) * sr): int((l["end"] + 0.35) * sr)]
        seg16 = resample_poly(seg, 1, 3).astype(np.float32)
        segs, _ = m.transcribe(seg16, beam_size=3, language="en", vad_filter=False)
        hyp = " ".join(s.text.strip() for s in segs)
        e = wer(l["text"], hyp)
        tot.append(e)
        flag = "  <--" if e > 0.25 else ""
        if flag:
            bad.append(l["id"])
        print(f"{l['id']:4s} WER {e:4.2f} | {hyp}{flag}")
    print(f"mean WER {np.mean(tot):.3f}; lines needing attention: {bad}")


if __name__ == "__main__":
    main()

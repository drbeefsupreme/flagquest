"""Render every dialogue line of the selected film (VX_FILM) with Kokoro.
Reads <film>/script/lines.json (a line's optional `say` is spoken instead of its display `text`),
writes <film>/audio/voice/raw/<ID>.wav (24 kHz mono) and index.json (durations + word timestamps).
usage: python tools/tts.py [ID ...]"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vx.config import SCRIPT, AUDIO  # noqa: E402

spec = json.loads((SCRIPT / "lines.json").read_text())
out = AUDIO / "voice/raw"
out.mkdir(parents=True, exist_ok=True)
idx_path = out / "index.json"
index = json.loads(idx_path.read_text()) if idx_path.exists() else {}

only = set(sys.argv[1:])
pipes = {}


def pipe(lang):
    if lang not in pipes:
        pipes[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
    return pipes[lang]


def trim(a, thr=1e-3, pad=0.03, sr=24000):
    nz = np.flatnonzero(np.abs(a) > thr)
    if not len(nz):
        return a, 0
    s = max(0, nz[0] - int(pad * sr))
    e = min(len(a), nz[-1] + int(pad * sr))
    return a[s:e], s


for ln in spec["lines"]:
    if only and ln["id"] not in only:
        continue
    cast = spec["cast"][ln["speaker"]]
    p = pipe(cast["lang"])
    speed = ln.get("speed", cast["speed"])
    voice = ln.get("voice", cast["voice"])
    say = ln.get("say", ln["text"])
    chunks, words, t0 = [], [], 0.0
    for r in p(say, voice=voice, speed=speed, split_pattern=None):
        a = r.audio.detach().cpu().numpy().astype(np.float32)
        for tk in (r.tokens or []):
            if tk.start_ts is not None and tk.end_ts is not None:
                words.append({"w": tk.text, "s": t0 + float(tk.start_ts), "e": t0 + float(tk.end_ts)})
        chunks.append(a)
        t0 += len(a) / 24000
    a = np.concatenate(chunks)
    a, off = trim(a)
    shift = off / 24000
    for w in words:
        w["s"] = round(max(0.0, w["s"] - shift), 3)
        w["e"] = round(max(0.0, w["e"] - shift), 3)
    sf.write(out / f"{ln['id']}.wav", a, 24000, subtype="FLOAT")
    index[ln["id"]] = {"speaker": ln["speaker"], "dur": round(len(a) / 24000, 3), "words": words,
                       "text": ln["text"], "say": say}
    print(f"{ln['id']:4s} {ln['speaker']:10s} {len(a)/24000:6.2f}s  {say[:70]}", flush=True)

idx_path.write_text(json.dumps(index, indent=1, ensure_ascii=False))
print("total speech", round(sum(v["dur"] for v in index.values()), 2), "s")

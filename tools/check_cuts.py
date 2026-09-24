"""Hand-off QA: last frame of each scene next to the first frame of the next one.
usage: python tools/check_cuts.py  -> out/cuts.jpg"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vx.config import OUT, TIMELINE  # noqa: E402

TL = json.loads(TIMELINE.read_text())


def frame(video, n, w=640):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{n}),scale={w}:-2",
                        "-vsync", "0", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR) if r.stdout else None


rows = []
sc = TL["scenes"]
for a, b in zip(sc, sc[1:]):
    va, vb = OUT / f"scenes/{a['id']}.mp4", OUT / f"scenes/{b['id']}.mp4"
    if not (va.exists() and vb.exists()):
        continue
    na = a["f1"] - a["f0"]
    fa, fb = frame(va, na - 1), frame(vb, 0)
    fb2 = frame(vb, 6)
    if fa is None or fb is None:
        continue
    row = np.concatenate([fa, fb, fb2 if fb2 is not None else fb], axis=1)
    cv2.putText(row, f"{a['id']} last | {b['id']} first | {b['id']} +6f   (T={b['start']:.1f})", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    rows.append(row)
out = OUT / "cuts.jpg"
cv2.imwrite(str(out), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out, len(rows))

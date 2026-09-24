"""QA contact sheets from rendered scene videos (or the final film).
usage: python tools/review.py [--every 2.0] [--video out/film_video.mp4] [--scene s04] [--cols 6]"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vx.config import OUT, TIMELINE  # noqa: E402

TL = json.loads(TIMELINE.read_text())


def grab(video, t, w=480):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                        "-vf", f"scale={w}:-2", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    if not r.stdout:
        return None
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    items = []
    if a.video:
        v = Path(a.video)
        n = int(TL["duration"] / a.every)
        items = [(v, k * a.every, f"T={k * a.every:6.1f}") for k in range(n)]
    else:
        for sc in TL["scenes"]:
            if a.scene and sc["id"] != a.scene:
                continue
            v = OUT / "scenes" / f"{sc['id']}.mp4"
            if not v.exists():
                v = OUT / "preview" / f"{sc['id']}.mp4"
            if not v.exists():
                continue
            d = sc["end"] - sc["start"]
            k = 0
            while k * a.every < d - 0.05:
                items.append((v, k * a.every + 0.02, f"{sc['id']} T={sc['start'] + k * a.every:6.1f}"))
                k += 1
    tiles = []
    for v, t, lab in items:
        im = grab(v, t)
        if im is None:
            continue
        cv2.putText(im, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    if not tiles:
        raise SystemExit("nothing rendered yet")
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + a.cols - 1) // a.cols
    sheet = np.full((rows * th, a.cols * tw, 3), 20, np.uint8)
    for k, im in enumerate(tiles):
        r, c = divmod(k, a.cols)
        im = cv2.resize(im, (tw, th))
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im
    out = Path(a.out) if a.out else OUT / f"review_{a.scene or 'all'}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out, len(tiles), "tiles")


if __name__ == "__main__":
    main()

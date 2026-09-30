"""One Rule QA: scan rendered scenes for anything enclosed INSIDE flag cloth (text, lines, dots, stains).
Flags are the only strongly yellow thing in film 2, so: yellow mask -> fill holes -> holes = non-yellow pixels
fully enclosed by cloth. Clean cloth has no holes; a hand in front touches the cloth edge (not enclosed).
--smooth (film 3, where gold tesserae are yellow too): cloth = yellow AND smooth (mosaic has mortar joints,
cloth has none), so gold grounds are ignored and only real cloth is checked.
usage: python tools/qa_onerule.py [--fps 2] [--min 40] [--smooth] [scene ...]  -> out/onerule/<scene>_<T>.png crops"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vx.config import OUT, TIMELINE  # noqa: E402

TL = json.loads(TIMELINE.read_text())


def frames(video, fps):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = 1920 * 1080 * 3
    k = 0
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield k / fps, np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3)
        k += 1


def yellow(img):
    f = img.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return (np.minimum(r, g) - b > 0.28) & (r > 0.55) & (g > 0.35)


def pole(img):
    """the flag's own untreated-wood pole (vx.flag palette) may cross its cloth: not a marking"""
    f = img.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return (r - b > 0.12) & (r - b < 0.42) & (r > 0.5) & (g > 0.4) & (np.minimum(r, g) - b <= 0.28)


def smooth_cloth(img):
    """yellow pixels in low-texture neighbourhoods (flag cloth), without the textured gold of a mosaic."""
    f = img.astype(np.float32) / 255.0
    lum = f @ np.float32([0.299, 0.587, 0.114])
    mu = cv2.blur(lum, (7, 7))
    sd = np.sqrt(np.maximum(cv2.blur(lum * lum, (7, 7)) - mu * mu, 0))
    m = yellow(img) & (sd < 0.04)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.flatnonzero(st[:, cv2.CC_STAT_AREA] >= 300)
    keep = keep[keep > 0]
    return np.isin(lab, keep)


def highlight(img):
    """the cloth's own lit highlights (flag_light .. flag_glow): warm, bright, not a marking"""
    f = img.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return (r > 0.85) & (g > 0.72) & (r - b > 0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--min", type=int, default=40, help="min enclosed pixels to report")
    ap.add_argument("--smooth", action="store_true", help="cloth must be smooth (mosaic films: ignore gold tesserae)")
    a = ap.parse_args()
    od = OUT / "onerule"
    od.mkdir(parents=True, exist_ok=True)
    total = 0
    for sc in TL["scenes"]:
        if a.scenes and sc["id"] not in a.scenes:
            continue
        v = OUT / "scenes" / f"{sc['id']}.mp4"
        if not v.exists():
            print(f"{sc['id']}: no render")
            continue
        hits = 0
        for t, img in frames(v, a.fps):
            m = smooth_cloth(img) if a.smooth else yellow(img)
            if m.sum() < 200:
                continue
            if not a.smooth:
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
            holes = binary_fill_holes(m) & ~m & ~pole(img)
            if a.smooth:
                holes &= ~highlight(img)
            lab, n = label(holes)
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel())[1:]
            big = np.flatnonzero(sizes >= a.min) + 1
            if len(big) == 0:
                continue
            ys, xs = np.nonzero(np.isin(lab, big))
            y0, y1, x0, x1 = max(0, ys.min() - 60), ys.max() + 60, max(0, xs.min() - 60), xs.max() + 60
            T = sc["start"] + t
            crop = img[y0:y1, x0:x1]
            cv2.imwrite(str(od / f"{sc['id']}_{T:07.2f}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            print(f"{sc['id']} T={T:7.2f}  enclosed px={int(sizes[big - 1].sum()):6d}  bbox=({x0},{y0})-({x1},{y1})")
            hits += 1
        total += hits
        print(f"{sc['id']}: {hits} flagged frames")
    print("TOTAL flagged frames:", total, "->", od)


if __name__ == "__main__":
    main()

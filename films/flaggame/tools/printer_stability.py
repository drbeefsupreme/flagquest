"""Printer: temporal-stability measurements for vx.ink (pan glued to the page, moving object).

python films/flaggame/tools/printer_stability.py  -> prints metrics, writes out/stills/printer_stability.png
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import printer_test as pt  # noqa: E402
from vx import ink  # noqa: E402


def main():
    fc = pt.mkfc(1.0)
    rows = []
    for style in ("tract", "engrave"):
        fn = getattr(ink, style)
        # --- pan glued to the page: shift by exactly 5 px/frame; the page must move rigidly with it
        errs = []
        prev = None
        for k in range(4):
            pan = (40.0 + 5 * k, 0.0)
            img = fn(fc, pt.world(fc, sphere_x=1220, pan=pan), space="page", origin=pan)
            if prev is not None:
                a = img[60:-60, 60:-65]
                b = prev[60:-60, 65:-60]
                errs.append(np.abs(a - b).mean(axis=2))
            prev = img
        e = np.mean(errs, axis=0)
        # interior (away from frame edges where new content enters) vs all
        inner = e[:, 200:-200]
        print(f"{style:8s} pan: mean |diff| {e.mean():.4f}, interior {inner.mean():.4f}, "
              f"interior px >0.25: {(inner > 0.25).mean() * 100:.2f}%")
        # --- moving object, static camera: where do lines change?
        a = fn(fc, pt.world(fc, sphere_x=700))
        b = fn(fc, pt.world(fc, sphere_x=706))
        d = np.abs(a - b).mean(axis=2)
        far = d[:, 1300:]                     # far from the sphere (flag region excluded below)
        print(f"{style:8s} object: changed px (>0.25) near {(d[:, :1300] > 0.25).mean() * 100:.2f}%, "
              f"far {(far[:, :200] > 0.25).mean() * 100:.3f}%")
        # --- animated rain (fresh white streaks every frame): lines between the streaks must not boil
        g1, g2 = pt.world(fc, rain=1), pt.world(fc, rain=2)
        a, b = fn(fc, g1), fn(fc, g2)
        streak = (g1.min(axis=2) > 0.97) | (g2.min(axis=2) > 0.97)
        streak = cv2.dilate(streak.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        dr = np.abs(a - b).mean(axis=2)
        keep = ~streak
        keep[:200] = False                    # the white page top has no tone
        print(f"{style:8s} rain: changed px (>0.25) outside the streaks {(dr[keep] > 0.25).mean() * 100:.3f}%, "
              f"mean |diff| {dr[keep].mean():.4f}")
        ep = np.zeros_like(d)
        ep[60:-60, 60:-65] = e
        vis = np.concatenate([np.clip(ep * 3, 0, 1), np.clip(d * 3, 0, 1)], axis=1)
        rows.append(vis[:, ::2][::2])
    out = np.concatenate(rows, axis=0)
    p = pt.STILLS / "printer_stability.png"
    cv2.imwrite(str(p), (255 - out * 255).astype(np.uint8))
    print("wrote", p)


if __name__ == "__main__":
    main()

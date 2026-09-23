"""Scene runner.

  python -m vx.render s01                      # full quality -> out/scenes/s01.mp4
  python -m vx.render s01 --scale 0.5 --step 2 # fast preview  -> out/preview/s01.mp4
  python -m vx.render s01 --at 1.0,4.5,9       # stills at local seconds -> out/stills/s01/*.png + sheet
  python -m vx.render s01 --stills 0,24,48     # stills at local frame numbers
  python -m vx.render s01 --frames 100:160     # render a local frame range only (video)

Scene module contract (scenes/<module>.py):
  def setup(S) -> state            # once, in parent; deterministic; heavy sims here (cache in S.cache)
  def render(fc, state) -> ndarray # float32 (fc.h, fc.w, 3) sRGB in [0,1]
  POST = {...}                     # optional film-look defaults (see vx.post.DEFAULT)
  def post(fc, state) -> dict      # optional per-frame overrides
"""
import argparse
import importlib
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)   # parallelism comes from worker processes; avoid 32-thread pools per worker

from .config import W, H, FPS, OUT, CACHE, ROOT
from .timeline import get_timeline
from . import post as _post


@dataclass
class SceneInfo:
    id: str
    start: float          # global seconds
    end: float
    dur: float
    n: int                # frame count
    f0: int               # global frame index of local frame 0
    fps: int
    s: float              # render scale (1.0 = 1920x1080)
    cache: Path
    tl: object = field(repr=False, default=None)
    meta: dict = field(default_factory=dict)


@dataclass
class FrameCtx:
    f: int                # local frame
    t: float              # local seconds
    F: int                # global frame
    T: float              # global seconds
    n: int
    dur: float
    p: float              # local progress 0..1
    s: float              # scale; draw in design units, pixel arrays are (h, w)
    w: int
    h: int
    W: int = W
    H: int = H
    fps: int = FPS
    S: SceneInfo = field(repr=False, default=None)
    tl: object = field(repr=False, default=None)

    def rng(self, k=0):
        return np.random.default_rng(self.F * 1009 + k * 7 + 3)

    def px(self, v):
        """design units -> pixels"""
        return v * self.s


_G = {}


def make_fc(S, f):
    t = f / S.fps
    return FrameCtx(f=f, t=t, F=S.f0 + f, T=S.start + t, n=S.n, dur=S.dur, p=f / max(1, S.n - 1),
                    s=S.s, w=int(round(W * S.s)), h=int(round(H * S.s)), S=S, tl=S.tl)


def render_one(f):
    mod, S, state, use_post = _G["mod"], _G["S"], _G["state"], _G["post"]
    fc = make_fc(S, f)
    img = mod.render(fc, state)
    if img.dtype != np.float32:
        img = img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)
    if img.shape[:2] != (fc.h, fc.w):
        raise ValueError(f"{S.id} frame {f}: render returned {img.shape}, expected {(fc.h, fc.w, 3)}")
    if use_post:
        p = dict(getattr(mod, "POST", {}))
        if hasattr(mod, "post"):
            p.update(mod.post(fc, state) or {})
        return _post.finish(img[..., :3], fc, **p)
    return np.clip(img[..., :3] * 255 + 0.5, 0, 255).astype(np.uint8)


def _encode_chunk(args):
    frames, path, w, h, fps = args
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "11",
           "-pix_fmt", "yuv444p", "-profile:v", "high444", str(path)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        pr.stdin.write(render_one(f).tobytes())
    pr.stdin.close()
    pr.wait()
    return path, len(frames)


def _still(args):
    f, path = args
    img = render_one(f)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def contact_sheet(paths, out, cols=4, width=1600):
    ims = [cv2.imread(str(p)) for p in paths]
    ims = [i for i in ims if i is not None]
    if not ims:
        return
    tw = width // cols
    th = int(tw * ims[0].shape[0] / ims[0].shape[1])
    rows = math.ceil(len(ims) / cols)
    sheet = np.full((rows * (th + 24), cols * tw, 3), 30, np.uint8)
    for k, (im, p) in enumerate(zip(ims, paths)):
        r, c = divmod(k, cols)
        sheet[r * (th + 24):r * (th + 24) + th, c * tw:(c + 1) * tw] = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, Path(p).stem, (c * tw + 6, r * (th + 24) + th + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def load_scene(sid, scale=1.0, meta=None):
    tl = get_timeline()
    sc = tl.scene(sid)
    sys.path.insert(0, str(ROOT))
    mod = importlib.import_module(sc["module"])
    S = SceneInfo(id=sid, start=sc["start"], end=sc["end"], dur=sc["end"] - sc["start"],
                  n=sc["f1"] - sc["f0"], f0=sc["f0"], fps=tl.fps, s=scale,
                  cache=CACHE / sid, tl=tl, meta=meta or {})
    S.cache.mkdir(parents=True, exist_ok=True)
    return mod, S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--frames", default=None, help="a:b local frame range")
    ap.add_argument("--stills", default=None, help="comma list of local frames")
    ap.add_argument("--at", default=None, help="comma list of local seconds")
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() // 4)))
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    t0 = time.time()
    mod, S = load_scene(a.scene, a.scale)
    state = mod.setup(S) if hasattr(mod, "setup") else None
    print(f"[{S.id}] setup {time.time() - t0:.1f}s  ({S.n} frames, {S.dur:.2f}s)", flush=True)
    _G.update(mod=mod, S=S, state=state, post=not a.no_post)
    ctx = mp.get_context("fork")
    w, h = int(round(W * a.scale)), int(round(H * a.scale))

    if a.stills or a.at:
        if a.at:
            frames = [min(S.n - 1, max(0, int(round(float(x) * S.fps)))) for x in a.at.split(",")]
        else:
            frames = [min(S.n - 1, max(0, int(x))) for x in a.stills.split(",")]
        d = OUT / "stills" / S.id
        d.mkdir(parents=True, exist_ok=True)
        jobs = [(f, d / f"{S.id}_f{f:04d}.png") for f in frames]
        with ctx.Pool(min(a.workers, len(jobs))) as pool:
            paths = pool.map(_still, jobs)
        sheet = OUT / "stills" / f"{S.id}_sheet.jpg"
        contact_sheet(paths, sheet, cols=min(4, len(paths)))
        print(f"[{S.id}] {len(paths)} stills -> {d}  sheet -> {sheet}  ({time.time() - t0:.1f}s)")
        return

    lo, hi = 0, S.n
    if a.frames:
        x, y = a.frames.split(":")
        lo, hi = int(x or 0), int(y or S.n)
    frames = list(range(lo, min(hi, S.n), a.step))
    preview = a.scale != 1.0 or a.step != 1 or a.frames
    outp = Path(a.out) if a.out else (OUT / ("preview" if preview else "scenes") / f"{S.id}.mp4")
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.parent / f".{S.id}_chunks"
    tmp.mkdir(exist_ok=True)
    for old in tmp.glob("*.mp4"):
        old.unlink()
    nw = max(1, min(a.workers, len(frames)))
    per = math.ceil(len(frames) / nw)
    fps_out = S.fps / a.step
    jobs = [(frames[i * per:(i + 1) * per], tmp / f"c{i:03d}.mp4", w, h, fps_out) for i in range(nw)
            if frames[i * per:(i + 1) * per]]
    with ctx.Pool(len(jobs)) as pool:
        res = pool.map(_encode_chunk, jobs)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p, _ in res))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(outp)], check=True)
    el = time.time() - t0
    print(f"[{S.id}] {len(frames)} frames -> {outp}  {el:.1f}s ({el / max(1, len(frames)):.3f}s/frame)")


if __name__ == "__main__":
    main()

"""Letterer's verification sheets: `python -m vx.comic_sheet [name ...]` -> films/flaggame/out/stills/letterer_*.png
(every balloon kind on real timeline lines at several reveal times, flags_flood at 4 amounts with a live flag,
emanata, sfx, furniture, avoid/flagify) and `python -m vx.comic_sheet clip` ->
films/flaggame/out/preview/letterer_test.mp4 (R01/S01 typing on, muxed with the dry dialogue stem).
Every frame with a flag reports how many lettering pixels touch the cloth (must be 0)."""
import math
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from .canvas import Canvas, over
from .config import OUT, AUDIO, W, H
from .timeline import get_timeline
from . import comic, ink

STILLS = OUT / "stills"
PREVIEW = OUT / "preview"
S = 0.5


def fcx(T, s=S):
    return SimpleNamespace(T=T, tl=get_timeline(), s=s, w=int(round(W * s)), h=int(round(H * s)), W=W, H=H)


def save(img, name):
    STILLS.mkdir(parents=True, exist_ok=True)
    p = STILLS / f"letterer_{name}.png"
    cv2.imwrite(str(p), cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(p)
    return p


def label(img, text):
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)
    return out.astype(np.float32) / 255


def grey_world(fc, seed=0):
    """a stand-in grey tract drawing: soft shapes printed as halftone"""
    c = Canvas(fc, bg=(1, 1, 1))
    ctx = c.ctx
    rng = np.random.default_rng(seed)
    for k in range(7):
        g = 0.6 + 0.35 * rng.random()
        ctx.set_source_rgb(g, g, g)
        ctx.arc(rng.random() * W, H * (0.45 + 0.6 * rng.random()), 120 + 240 * rng.random(), 0, 2 * math.pi)
        ctx.fill()
    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(0, H * 0.88, W, H * 0.12)
    ctx.fill()
    return ink.tract(fc, c.rgb())


def head(ctx, x, y, r=60):
    ctx.set_source_rgb(0.1, 0.1, 0.1)
    ctx.set_line_width(4)
    ctx.new_path()
    ctx.arc(x, y, r, 0, 2 * math.pi)
    ctx.stroke()
    ctx.arc(x + r * 0.35, y + r * 0.35, r * 0.12, 0, 2 * math.pi)
    ctx.fill()


def shot(T, draw, *, seed=0, world=True, flags=(), paper_only=False):
    """world -> print -> flag spot plate -> lettering (avoiding the cloth). draw(ctx, fc, avoid)"""
    from .flag import draw_flag
    fc = fcx(T)
    img = ink.paper(fc) if paper_only else (grey_world(fc, seed) if world else ink.paper(fc))
    av = []
    frgba = None
    if flags:
        fl = Canvas(fc)
        for (x, y, pole) in flags:
            draw_flag(fl.ctx, x, y, pole=pole, t=T, wind=1.0)
        frgba = fl.rgba()
        img = ink.spot(img, frgba)
        av = comic.cloth_avoid(frgba, fc.s)
    top = Canvas(fc)
    draw(top.ctx, fc, av)
    lay = top.rgba()
    if frgba is not None:
        a = (frgba[..., 3] > 0.02).astype(np.uint8)
        k = max(1, int(round(16 * fc.s)))
        cloth = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 0
        lt = lay[..., 3] > 0.02
        print(f"  T={T:.2f}: lettering pixels on cloth = {int((lt & cloth).sum())}, "
              f"on pole = {int((lt & (a > 0) & ~cloth).sum())}")
    return over(img, lay)


def tile(frames, cols):
    h, w = frames[0].shape[:2]
    rows = int(math.ceil(len(frames) / cols))
    out = np.full((rows * h + (rows + 1) * 8, cols * w + (cols + 1) * 8, 3), 0.17, np.float32)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        out[8 + r * (h + 8): 8 + r * (h + 8) + h, 8 + c * (w + 8): 8 + c * (w + 8) + w] = f
    return out


def at(lid, frac):
    l = get_timeline().line(lid)
    return l["start"] + (l["end"] - l["start"]) * frac


# ---------------------------------------------------------------------------- sheets
def sheet_balloons():
    """voice-synced reveal: R01 (with the flag-bearer passing: avoid), S01, P02 shout, S03"""
    shots = []
    for frac in (0.02, 0.4, 0.75, 1.1):
        T = at("R01", frac)

        def d(ctx, fc, av):
            head(ctx, 1450, 760)
            comic.say(ctx, fc, "R01", 900, 330, tail=(1420, 730), avoid=av)
        shots.append(label(shot(T, d, seed=1, flags=[(560 + 60 * T, 900, 560)]), f"R01 T={T:.2f}"))
    for frac in (0.05, 0.45, 0.8, 1.08):
        T = at("S01", frac)

        def d(ctx, fc, av):
            head(ctx, 520, 790)
            comic.say(ctx, fc, "S01", 1050, 360, tail=(560, 750))
        shots.append(label(shot(T, d, seed=2), f"S01 T={T:.2f}"))
    for frac in (0.1, 0.4, 0.95):
        T = at("P02", frac)

        def d(ctx, fc, av):
            comic.burst_bg(ctx, 1400, 700, (0, 0, W, H), t=fc.T)
            head(ctx, 1400, 760, 90)
            comic.say(ctx, fc, "P02", 800, 360, tail=(1340, 720))
            comic.say_footnote(ctx, fc, "P02", 1890, 1060)
        shots.append(label(shot(T, d, seed=3), f"P02 T={T:.2f}"))
    T = at("S03", 0.8)

    def d(ctx, fc, av):
        head(ctx, 450, 800)
        comic.emanata(ctx, 450, 740, 110, "shock", t=fc.T)
        comic.say(ctx, fc, "S03", 1000, 380, tail=(500, 760))
    shots.append(label(shot(T, d, seed=4), f"S03 T={T:.2f}"))
    save(tile(shots, 4), "balloons")


def sheet_kinds():
    specs = [
        ("F01", (900, 420), dict(tail=(560, 860)), 0.97, "radio"),
        ("G01", (1000, 420), dict(tail=(1350, 860)), 0.97, "radio"),
        ("M01", (900, 360), dict(tail=(1350, 820)), 0.97, "divine"),
        ("M03", (1000, 360), dict(tail=(560, 820)), 0.97, "divine"),
        ("V01", (900, 360), dict(tail=(1400, 850)), 0.97, "devil"),
        ("V02", (960, 380), dict(tail=(620, 870)), 0.97, "devil shout"),
        ("X01", (960, 380), dict(tail=(1000, 800)), 0.97, "shout"),
        ("S04", (1150, 380), dict(tail=(560, 820)), 0.97, "anime"),
        ("S05", (600, 330), dict(tail=(W, 150)), 0.97, "offpanel"),
        ("P08", (1150, 360), dict(tail=(0, 300)), 0.97, "offpanel"),
        ("H01", (900, 380), dict(tail=(1300, 820)), 0.97, "shout"),
        ("H02", (1000, 380), dict(tail=(600, 820)), 0.97, "shout"),
        ("P25", (900, 360), dict(tail=(1400, 850)), 0.97, "shout"),
        ("S06", (1000, 380), dict(tail=(560, 820)), 0.97, "balloon"),
        ("P30", (900, 360), dict(tail=(1400, 850)), 0.97, "balloon"),
        ("R02", (1000, 330), dict(tail=(560, 820)), 0.97, "balloon (loop)"),
    ]
    shots = []
    for lid, (x, y), kw, frac, name in specs:
        T = at(lid, frac)

        def d(ctx, fc, av, lid=lid, kw=kw, x=x, y=y):
            tx, ty = kw["tail"]
            if 0 < tx < W:
                head(ctx, tx, ty + 40)
            comic.say(ctx, fc, lid, x, y, **kw)
        shots.append(label(shot(T, d, seed=len(shots)), f"{lid} {name}"))
    save(tile(shots, 4), "kinds")


def sheet_captions():
    shots = []

    def cap(lid, x, y, w, frac, name, paper=False, **kw):
        T = at(lid, frac)

        def d(ctx, fc, av):
            comic.say(ctx, fc, lid, x, y, w=w, **kw)
        shots.append(label(shot(T, d, seed=len(shots), paper_only=paper), f"{lid} {name} T={T:.2f}"))

    cap("P09", 60, 50, 1000, 0.35, "caption, phrase reveal")
    cap("P09", 60, 50, 1000, 1.0, "caption")
    cap("P21", 80, 60, 700, 0.55, "condensed caption")
    cap("P21", 80, 60, 700, 1.0, "condensed caption")
    cap("P04", 1000, 60, 860, 0.7, "italic emphasis")
    cap("P07", 60, 50, 1100, 1.0, "caption")
    cap("P13", 960, 540, 1500, 0.5, "title card", paper=True)
    cap("P13", 960, 540, 1500, 1.0, "title card", paper=True)
    cap("D01", 100, 1000, None, 0.5, "footnote typing", anchor="bl")
    cap("D01", 100, 1000, None, 1.0, "footnote", anchor="bl")
    T = at("P02", 1.0) + 0.2

    def fn(ctx, fc, av):
        comic.say(ctx, fc, "P02", 900, 380, tail=(1350, 800))
        comic.say_footnote(ctx, fc, "P02", 1880, 1040)
    shots.append(label(shot(T, fn, seed=11), "P02 + say_footnote"))
    T = at("P12", 1.0)

    def p12(ctx, fc, av):
        comic.say(ctx, fc, "P12", 60, 60, w=1100, tokens=(0, 9))
        comic.say(ctx, fc, "P12", 1860, 1030, kind="footnote", tokens=(9, None), anchor="br", size=40)
    shots.append(label(shot(T, p12, seed=12), "P12 tokens split + citation"))
    save(tile(shots, 4), "captions")


def sheet_flood():
    shots = []
    for a in (0.25, 0.5, 0.75, 1.0):
        T = 88.0 + a * 7.7

        def d(ctx, fc, av, a=a):
            comic.flags_flood(ctx, (0, 0, W, H), a, t=fc.T, size=34, avoid=av, pulse=0.5 + 0.5 * math.cos(fc.T * 13))
        shots.append(label(shot(T, d, seed=5, flags=[(900, 980, 700)]), f"flags_flood amount={a}"))
    save(tile(shots, 2), "flood")


def sheet_furniture():
    shots = []
    kinds = ["shock", "sweat", "stink", "sparkle", "tears", "blush", "anger", "question", "exclaim", "dizzy"]
    for T in (0.3, 0.75):
        def em(ctx, fc, av):
            for i, k in enumerate(kinds):
                x, y = 200 + (i % 5) * 380, 330 + (i // 5) * 450
                head(ctx, x, y + 60, 70)
                comic.emanata(ctx, x, y - 30 if k not in ("sweat", "tears", "blush") else y + 40, 80, k, t=fc.T)
        shots.append(label(shot(T, em, paper_only=True), f"emanata t={T}"))

    def sf(ctx, fc, av):
        comic.sfx(ctx, "KRAKOOM!", 560, 300, 150, rot=-0.12, depth=14)
        comic.sfx(ctx, "FWOOSH", 1380, 280, 120, rot=0.1)
        comic.sfx(ctx, "SPLOOSH", 600, 760, 120, rot=0.08)
        comic.sfx(ctx, "THUNK", 1400, 760, 130, rot=-0.2)
    shots.append(label(shot(0, sf, seed=6), "sfx"))
    for t in (0.04, 0.12, 0.3):
        def sp(ctx, fc, av, t=t):
            comic.sfx(ctx, "KRAKOOM!", 960, 540, 190, rot=-0.1, t=t, depth=16)
        shots.append(label(shot(0, sp, seed=7), f"sfx pop t={t}"))

    def bb(ctx, fc, av):
        comic.burst_bg(ctx, 960, 540, (0, 0, W, H), t=fc.T)
        head(ctx, 960, 540, 120)
    shots.append(label(shot(0.2, bb, paper_only=True), "burst_bg"))

    def sp(ctx, fc, av):
        comic.speed_lines(ctx, (0, 0, W, H), focus=(960, 540), t=fc.T)
        head(ctx, 960, 540, 120)
    shots.append(label(shot(0.5, sp, paper_only=True), "speed_lines focus"))

    def sl(ctx, fc, av):
        comic.speed_lines(ctx, (0, 0, W, H), angle=0.2, t=fc.T)
    shots.append(label(shot(0.5, sl, paper_only=True), "speed_lines parallel"))

    def pg(ctx, fc, av):
        ctx.set_source_rgb(0.1, 0.1, 0.1)
        ctx.paint()
        p = comic.Page(560, 20, 1040, number=23)
        p.draw(ctx)
    shots.append(label(shot(0.5, pg, paper_only=True), "Page"))
    save(tile(shots, 4), "furniture")


def sheet_avoid():
    """a balloon asked to sit on top of the cloth gets nudged clear; flagify; the flag-bearer passes"""
    shots = []
    for T in (8.4, 9.4, 10.4):
        def d(ctx, fc, av):
            head(ctx, 1500, 800)
            comic.say(ctx, fc, "R01", 1000, 420, tail=(1470, 770), avoid=av)
        shots.append(label(shot(T, d, seed=8, flags=[(1000, 980, 700)]), f"avoid: asked at the cloth T={T}"))
    for u in (0.3, 0.8):
        def d(ctx, fc, av, u=u):
            head(ctx, 1500, 850)
            comic.say(ctx, fc, "M02", 800, 330, tail=(1470, 820), T=91.8, flagify=u)
        shots.append(label(shot(91.8, d, seed=9), f"M02 flagify={u}"))

    def d(ctx, fc, av):
        head(ctx, 1500, 850)
        comic.say(ctx, fc, "V01", 800, 330, tail=(1470, 820), flagify=0.6)
    shots.append(label(shot(101.9, d, seed=10), "V01 flagify=0.6"))
    save(tile(shots, 3), "avoid")


def clip():
    """6-s clip R01 -> S01 typing on (a flag-bearer walks through; balloons placed against the flag's whole
    path via a callable avoid, so they never jump), muxed with the dry dialogue stem"""
    from .flag import Flag
    PREVIEW.mkdir(parents=True, exist_ok=True)
    t0, dur, fps = 7.2, 6.0, 24
    s = 0.5
    n = int(dur * fps)
    trk = Flag(pole=560, seed=3).simulate(n + 24, lambda i: (260 + 140 * i / fps, 960, 0.0),
                                          lambda i: (-160.0, 0.0, 0.0))
    raw = PREVIEW / "letterer_test_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(W * s), int(H * s)))
    base = None
    worst = 0
    for i in range(n):
        T = t0 + i / fps
        fc = fcx(T, s)
        if base is None:
            base = grey_world(fc, 5)
        fl = Canvas(fc)
        trk.draw(fl.ctx, i)
        frgba = fl.rgba()
        img = ink.spot(base, frgba)
        top = Canvas(fc)

        def av(TT):
            return comic.track_avoid(trk, min(n + 23, max(0.0, (TT - t0) * fps)))
        comic.say(top.ctx, fc, "R01", 1320, 300, tail=(1480, 680), hold=0.3, avoid=av)
        comic.say(top.ctx, fc, "S01", 640, 300, tail=(560, 720), avoid=av)
        lay = top.rgba()
        worst = max(worst, int((lay[..., 3][frgba[..., 3] > 0.02] > 0.02).sum()))
        hd = Canvas(fc)
        head(hd.ctx, 1500, 760, 90)
        head(hd.ctx, 560, 800, 90)
        img = over(over(img, hd.rgba()), lay)
        vw.write(cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"  worst lettering pixels on the flag in any frame: {worst}")
    vw.release()
    out = PREVIEW / "letterer_test.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-ss", str(t0), "-t", str(dur),
                    "-i", str(AUDIO / "stems/dialogue_dry.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)],
                   check=True)
    raw.unlink()
    print(out)


SHEETS = dict(balloons=sheet_balloons, kinds=sheet_kinds, captions=sheet_captions, flood=sheet_flood,
              furniture=sheet_furniture, avoid=sheet_avoid, clip=clip)

if __name__ == "__main__":
    for n in (sys.argv[1:] or [k for k in SHEETS if k != "clip"]):
        SHEETS[n]()

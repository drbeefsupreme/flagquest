"""Printer's verification rig for vx.ink.

python films/flaggame/tools/printer_test.py sheet    -> out/stills/printer_{tract,engrave,manga,reprint,crops}.png
python films/flaggame/tools/printer_test.py clip     -> out/preview/printer_test.mp4 (48 frames: object + pan)
python films/flaggame/tools/printer_test.py bench    -> ms per 1080p frame per style
"""
import math
import subprocess
import sys
import time

import cv2
import numpy as np

from vx import ink
from vx.canvas import Canvas, circle, ellipse, lin_grad, rad_grad, text, smooth
from vx.config import OUT
from vx.flag import draw_flag
from vx.render import FrameCtx

STILLS = OUT / "stills"
PREV = OUT / "preview"


def mkfc(s=1.0, f=0):
    return FrameCtx(f=f, t=f / 24, F=f, T=f / 24, n=48, dur=2.0, p=0.0, s=s,
                    w=int(round(1920 * s)), h=int(round(1080 * s)))


def _figure(ctx, x, y, h, t=0.0):
    """a character: try the rig's ink mode, else a hand-drawn grey silhouette with black brush contours"""
    try:
        from vx.chars import Character
        Character("summer", seed=3, render="ink").draw(ctx, x, y, h, "stand", t=t, expr="bewildered",
                                                        render="ink")
        return
    except Exception as e:  # noqa: BLE001
        print("chars fallback:", e)
    sc = h / 700.0
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(sc, sc)
    body = [(-120, 0), (-140, -260), (-110, -420), (-60, -470), (60, -470), (110, -420), (140, -260), (120, 0)]
    smooth(ctx, body, close=True)
    ctx.set_source(lin_grad(-140, 0, 140, 0, [(0, (0.72, 0.72, 0.72)), (0.6, (0.45, 0.45, 0.45)),
                                                (1, (0.2, 0.2, 0.2))]))
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(6)
    ctx.stroke()
    circle(ctx, 0, -560, 95)
    ctx.set_source(rad_grad(-30, -590, 10, 110, [(0, (0.97, 0.97, 0.97)), (1, (0.55, 0.55, 0.55))]))
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.stroke()
    ctx.restore()


def world(fc, sphere_x=430.0, pan=(0.0, 0.0), t=0.0, flags=None, rain=None):
    """grey test illustration (design px); pan = page offset of the camera; flags: Canvas for the spot plate"""
    cv = Canvas(fc, bg=(1, 1, 1))
    ctx = cv.ctx
    ctx.save()
    ctx.translate(-pan[0], -pan[1])
    # sky + ground
    ctx.rectangle(-400, 200, 3000, 560)
    ctx.set_source(lin_grad(0, 200, 0, 760, [(0, (0.93, 0.93, 0.93)), (1, (0.62, 0.62, 0.62))]))
    ctx.fill()
    ctx.rectangle(-400, 760, 3000, 700)
    ctx.set_source(lin_grad(0, 760, 0, 1100, [(0, (0.5, 0.5, 0.5)), (1, (0.25, 0.25, 0.25))]))
    ctx.fill()
    # distant hills with black contour
    smooth(ctx, [(-400, 700), (0, 610), (400, 660), (800, 590), (1200, 650), (1600, 600), (2000, 640),
                 (2600, 620), (2600, 780), (-400, 780)], close=True)
    ctx.set_source_rgb(0.72, 0.72, 0.72)
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(3)
    ctx.stroke()
    # gradient strip + step wedge
    ctx.rectangle(60, 30, 1800, 70)
    ctx.set_source(lin_grad(60, 0, 1860, 0, [(0, (1, 1, 1)), (1, (0, 0, 0))]))
    ctx.fill()
    for k in range(11):
        v = 1 - k / 10
        ctx.rectangle(60 + k * 1800 / 11, 110, 1800 / 11, 70)
        ctx.set_source_rgb(v, v, v)
        ctx.fill()
    # cast shadow + sphere
    ellipse(ctx, sphere_x + 90, 905, 230, 48)
    ctx.set_source_rgba(0, 0, 0, 0.55)
    ctx.fill()
    circle(ctx, sphere_x, 690, 220)
    ctx.set_source(rad_grad(sphere_x - 80, 610, 8, 300, [(0, (1, 1, 1)), (0.25, (0.85, 0.85, 0.85)),
                                                         (0.7, (0.35, 0.35, 0.35)), (1, (0.08, 0.08, 0.08))]))
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(5)
    ctx.stroke()
    # a character
    _figure(ctx, 1010, 1040, 760, t)
    # black lettering + hairlines of increasing weight
    text(ctx, "THE FLAGS!", 1330, 330, 118, color=(0, 0, 0), bold=True)
    for k in range(8):
        ctx.set_line_width(0.5 + k * 0.5)
        ctx.move_to(1340 + k * 26, 380)
        ctx.curve_to(1380 + k * 26, 440, 1300 + k * 26, 500, 1350 + k * 26, 560)
        ctx.set_source_rgb(0, 0, 0)
        ctx.stroke()
    if rain is not None:   # Dore rain: pure-white streaks, a fresh set every frame
        rng = np.random.default_rng(1000 + int(rain))
        ctx.set_source_rgb(1, 1, 1)
        for _ in range(420):
            x, y = rng.uniform(-200, 2200), rng.uniform(150, 1100)
            L = rng.uniform(40, 90)
            ctx.set_line_width(rng.uniform(1.2, 2.6))
            ctx.move_to(x, y)
            ctx.line_to(x - 0.18 * L, y + L)
            ctx.stroke()
    ctx.restore()
    if flags is not None:
        fctx = flags.ctx
        fctx.save()
        fctx.translate(-pan[0], -pan[1])
        draw_flag(fctx, 1650, 960, pole=560, t=t, wind=1.2, seed=4)
        fctx.restore()
    return cv.rgb()


def printed(fc, style, **kw):
    flags = Canvas(fc)
    g = world(fc, flags=flags, **{k: kw.pop(k) for k in ("sphere_x", "pan", "t") if k in kw})
    if style == "grey":
        img = g
    elif style == "reprint":
        prog = np.clip((np.linspace(-0.2, 1.2, fc.w)[None, :]).repeat(fc.h, 0), 0, 1).astype(np.float32)
        img = ink.reprint(fc, g, "tract", "engrave", prog, **kw)
    else:
        img = getattr(ink, style)(fc, g, **kw)
    return ink.spot(img, flags.rgba())


def save(path, img):
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("wrote", path)


def sheet():
    STILLS.mkdir(parents=True, exist_ok=True)
    fc = mkfc(1.0)
    crops = []
    for style in ("grey", "tract", "engrave", "manga", "reprint"):
        img = printed(fc, style)
        save(STILLS / f"printer_{style}.png", img)
        for (y0, x0) in ((560, 250), (240, 830), (200, 1330)):
            c = img[y0:y0 + 270, x0:x0 + 400]
            crops.append(cv2.resize(c, (c.shape[1] * 2, c.shape[0] * 2), interpolation=cv2.INTER_NEAREST))
    rows = [np.concatenate(crops[i:i + 3], axis=1) for i in range(0, len(crops), 3)]
    save(STILLS / "printer_crops.png", np.concatenate(rows, axis=0))
    # the deluge: a soaked engraved page (wet() before spot(): the flag stays pristine)
    flags = Canvas(fc)
    g = world(fc, flags=flags)
    ys, xs = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
    rng = np.random.default_rng(3)
    wob = cv2.GaussianBlur(rng.standard_normal((fc.h, fc.w)).astype(np.float32), (0, 0), 25) * 900
    front = (xs * 0.8 + (fc.h - ys) * 1.0 + wob) / fc.s
    mask = np.clip((1500 - front) / 160, 0, 1)
    for style in ("engrave", "tract"):
        img = ink.wet(fc, getattr(ink, style)(fc, g), mask)
        save(STILLS / f"printer_wet_{style}.png", ink.spot(img, flags.rgba()))
    # the anime gag: focus lines (vx.comic.speed_lines) + anime Summer, printed as fine screentone
    from vx import comic
    from vx.chars import Character
    cv = Canvas(fc, bg=(1, 1, 1))
    ctx = cv.ctx
    ctx.rectangle(0, 0, 1920, 1080)
    ctx.set_source(rad_grad(960, 470, 80, 1100, [(0, (0.97, 0.97, 0.97)), (1, (0.55, 0.55, 0.55))]))
    ctx.fill()
    comic.speed_lines(ctx, (0, 0, 1920, 1080), focus=(960, 470), t=0.0)
    try:
        Character("summer", seed=3, render="ink").draw(ctx, 960, 1480, 1500, "stand", t=0.0,
                                                        expr="bewildered", face="anime")
    except Exception as e:  # noqa: BLE001
        print("anime summer fallback:", e)
    g = cv.rgb()
    save(STILLS / "printer_manga_gag.png", ink.manga(fc, g))
    save(STILLS / "printer_manga_gag_tract.png", ink.tract(fc, g))

def panels():
    """two real-cast panels: a tract panel (reads like page 23) and a Dore flashback panel"""
    from vx import comic
    from vx.chars import Character
    fc = mkfc(1.0)
    # ---- TRACT: Raven (solid black hair) sneers, Summer beams, a flag-bearer passes in the distance
    cv = Canvas(fc, bg=(1, 1, 1))
    ctx = cv.ctx
    ctx.rectangle(0, 0, 1920, 1080)
    ctx.set_source(lin_grad(0, 0, 0, 1080, [(0, (0.80, 0.80, 0.80)), (0.55, (0.93, 0.93, 0.93)),
                                            (0.56, (0.72, 0.72, 0.72)), (1, (0.55, 0.55, 0.55))]))
    ctx.fill()
    comic.burst_bg(ctx, 1480, 420, (1080, 0, 840, 1080), t=0.0, bg=False)
    Character("crow", seed=1, render="ink").draw(ctx, 960, 700, 330, "stand", t=0.0, expr="fervent")
    Character("raven", seed=2, render="ink").draw(ctx, 420, 1500, 1350, "stand", t=0.0, expr="disdain",
                                                  facing=1)
    Character("summer", seed=3, render="ink").draw(ctx, 1500, 1480, 1300, "stand", t=0.0, expr="cheerful",
                                                   facing=-1)
    flags = Canvas(fc)
    draw_flag(flags.ctx, 1010, 640, pole=300, t=0.4, wind=1.3, seed=2)
    img = ink.spot(ink.tract(fc, cv.rgb()), flags.rgba())
    top = Canvas(fc)
    comic.balloon(top.ctx, "Ugh, there goes another one of those YELLOW FLAGS.", 760, 150, 620,
                  tail=(560, 520))
    comic.panel(top.ctx, (8, 8, 1904, 1064), border=10, fill=False)
    save(STILLS / "printer_panel_tract.png", top.over(img))
    # ---- ENGRAVING: the Flag Maker walks amongst the muddy hippies in Dore rain
    cv = Canvas(fc, bg=(1, 1, 1))
    ctx = cv.ctx
    ctx.rectangle(0, 0, 1920, 1080)
    ctx.set_source(lin_grad(0, 0, 0, 1080, [(0, (0.18, 0.18, 0.18)), (0.45, (0.62, 0.62, 0.62)),
                                            (0.62, (0.80, 0.80, 0.80)), (1, (0.35, 0.35, 0.35))]))
    ctx.fill()
    ctx.set_source(rad_grad(960, 300, 20, 700, [(0, (1, 1, 1), 0.9), (1, (1, 1, 1), 0.0)]))
    ctx.paint()
    for k, (x, hgt, sd) in enumerate([(250, 620, 11), (520, 700, 12), (1400, 680, 13), (1680, 640, 14),
                                      (380, 820, 15), (1560, 840, 16)]):
        Character("hippie", seed=sd, render="ink", mud=0.7).draw(ctx, x, 700 + hgt * 0.55, hgt, "stand",
                                                                 t=0.0, facing=1 if x < 960 else -1)
    Character("flagmaker", seed=5, render="ink").draw(ctx, 960, 1060, 900, "preach", t=0.0, expr="serene")
    rng = np.random.default_rng(7)
    ctx.set_source_rgb(1, 1, 1)
    for _ in range(520):
        x, y = rng.uniform(-100, 2000), rng.uniform(-50, 1080)
        L = rng.uniform(40, 110)
        ctx.set_line_width(rng.uniform(1.0, 2.4))
        ctx.move_to(x, y)
        ctx.line_to(x - 0.2 * L, y + L)
        ctx.stroke()
    flags = Canvas(fc)
    draw_flag(flags.ctx, 1250, 1000, pole=520, t=0.2, wind=0.6, seed=5)
    img = ink.spot(ink.engrave(fc, cv.rgb()), flags.rgba())
    top = Canvas(fc)
    comic.caption(top.ctx, "THE FLAG MAKER WALKED AMONGST THEM...", 60, 50, 760)
    save(STILLS / "printer_panel_engrave.png", top.over(img))


def contact():
    """out/stills/printer_sheet.png: every verification still on one page (960 px tiles)"""
    names = ["grey", "tract", "engrave", "manga_gag", "reprint", "wet_engrave", "panel_tract", "panel_engrave"]
    tiles = []
    for n in names:
        im = cv2.imread(str(STILLS / f"printer_{n}.png"))
        im = cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.putText(im, n, (14, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2, cv2.LINE_AA)
        tiles.append(im)
    rows = [np.concatenate(tiles[i:i + 2], axis=1) for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(STILLS / "printer_sheet.png"), np.concatenate(rows, axis=0))
    print("wrote", STILLS / "printer_sheet.png")



def bench():
    fc = mkfc(1.0)
    g = world(fc)
    for style in ("tract", "engrave", "manga"):
        fn = getattr(ink, style)
        fn(fc, g)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            fn(fc, g)
            ts.append(time.perf_counter() - t0)
        print(f"{style:8s} {1000 * min(ts):6.1f} ms (min of 5), {1000 * np.median(ts):6.1f} ms median")
    t0 = time.perf_counter()
    ink.reprint(fc, g, "tract", "engrave", 0.5)
    t0 = time.perf_counter()
    ink.reprint(fc, g, "tract", "engrave", 0.5)
    print(f"reprint  {1000 * (time.perf_counter() - t0):6.1f} ms")


def clip(n=48, s=1.0):
    """frames 0..23: the sphere rolls (static camera); 24..47: camera pans over the page (space='page').
    Left half of every frame is tract, right half engrave (the split runs through the action)."""
    PREV.mkdir(parents=True, exist_ok=True)
    out = PREV / "printer_test.mp4"
    fc0 = mkfc(s)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{fc0.w}x{fc0.h}", "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "12",
                             "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    half = n // 2
    for f in range(n):
        fc = mkfc(s, f)
        if f < half:
            p = f / (half - 1)
            kw = dict(sphere_x=700 + 520 * (0.5 - 0.5 * math.cos(math.pi * p)), t=f / 24)
            pk = {}
        else:
            p = (f - half) / (half - 1)
            pan = (-180 + 360 * (0.5 - 0.5 * math.cos(math.pi * p)), 30 * math.sin(math.pi * p))
            kw = dict(sphere_x=1220, pan=pan, t=f / 24)
            pk = dict(space="page", origin=pan, zoom=1.0)
        flags = Canvas(fc)
        g = world(fc, flags=flags, **kw)
        a = ink.tract(fc, g, **pk)
        b = ink.engrave(fc, g, **pk)
        img = a.copy()
        img[:, fc.w // 2:] = b[:, fc.w // 2:]
        img[:, fc.w // 2 - 1:fc.w // 2 + 1] = 0.0
        img = ink.spot(img, flags.rgba())
        proc.stdin.write(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8).tobytes())
        if f in (0, half - 1, half, n - 1):
            save(PREV / f"printer_test_{f:02d}.png", img)
    proc.stdin.close()
    proc.wait()
    print("wrote", out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sheet"
    {"sheet": sheet, "bench": bench, "clip": clip, "panels": panels, "contact": contact}[cmd]()

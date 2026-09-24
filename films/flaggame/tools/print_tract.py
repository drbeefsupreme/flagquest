"""THE FLAG GAME? as a real, printable tract (the film's own frames and page art, Flag License: distribute it).

Builds, from the assembled film (out/film_video.mp4) and T01's printed page art:
  out/print/THE_FLAG_GAME_tract.pdf          16 reader pages (US-letter page aspect, like the Flagazine scans)
  out/print/THE_FLAG_GAME_tract_imposed.pdf   saddle-stitch imposition on tabloid (11x17) sheets, duplex:
                                              print both sides (flip on short edge), fold, staple twice
  out/print/plates/pNN_K.png, pNN_Y.png       2-colour separations per page: black plate + spot yellow plate
                                              (the Flags are the only thing on the yellow plate)
usage: source films/flaggame/env.sh && python films/flaggame/tools/print_tract.py [--dpi 300]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cairo
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from vx.config import OUT, FILM_ROOT  # noqa: E402
from vx.canvas import surface_from_array  # noqa: E402

sys.path.insert(0, str(FILM_ROOT))
from scenes import t01_page as P  # noqa: E402
from scenes import t01_pages as TP  # noqa: E402

# story pages: three panels each, a film timestamp per panel (moments where the camera rests on a panel)
STORY = [
    [9.6, 13.2, 16.6],        # 1  Raven / Summer confesses / doesn't fit in
    [20.6, 22.4, 23.9],       # 2  Amazing... / WAS WRONG!! / HOW CAN THIS BE???
    [28.9, 34.9, 41.0],       # 3  the icon / anime gag / Alchemy IX
    [46.2, 51.6, 58.2],       # 4  six Flags / They look yellow / the stench
    [60.4, 63.4, 65.8],       # 5  hurricane / RAIN / the Ranger
    [69.4, 77.2, 80.8],       # 6  the Flag Maker / Shut up! / FLAGS 4:20
    [88.8, 95.8, 102.8],      # 7  the explanation / the FLAGS flood / the Devil
    [106.9, 109.5, 112.4],    # 8  SOME THAT LISTENED / Found one! / AND IT WORKED
    [116.6, 119.2, 120.8],    # 9  the Seraphlag / the freeze / the firmament
    [126.4, 130.4, 136.0],    # 10 the burn / the Flag ascends / the crack
    [138.2, 145.0, 147.2],    # 11 the deluge / raised again / 100 time!
    [152.0, 157.9, 166.4],    # 12 the Spirit / the years without / BLASPHEME
    [171.2, 180.4, 186.8],    # 13 Lorem ipsum / Amen / the Print Gallery
]
PAGE_RGB = np.array(P.PAGE_RGB, np.float32)
YELLOW = np.array([1.0, 0.768, 0.102], np.float32)     # PAL flag: the spot ink


def grab(video, t):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(1080, 1920, 3).astype(np.float32) / 255.0


def panel_crop(img):
    """the panel the camera is resting on, or a centre crop of a full-frame shot, at the panel aspect"""
    aspect = P.PANEL_W / P.PANEL_H
    stock = np.all(np.abs(img - PAGE_RGB) < 0.035, axis=2)
    h, w = stock.shape
    if stock.mean() > 0.04:
        rows = np.where((~stock[:, w // 2 - 200:w // 2 + 200]).mean(axis=1) > 0.9)[0]
        cols = np.where((~stock[h // 2 - 100:h // 2 + 100, :]).mean(axis=0) > 0.9)[0]
        cy, cx = h // 2, w // 2
        if len(rows) and len(cols):
            # the contiguous run of non-stock rows/cols through the centre = the panel
            def run(idx, c):
                idx = np.asarray(idx)
                k = np.searchsorted(idx, c)
                k = min(max(k, 0), len(idx) - 1)
                lo = hi = k
                while lo > 0 and idx[lo - 1] == idx[lo] - 1:
                    lo -= 1
                while hi < len(idx) - 1 and idx[hi + 1] == idx[hi] + 1:
                    hi += 1
                return idx[lo], idx[hi]
            y0, y1 = run(rows, cy)
            x0, x1 = run(cols, cx)
            if (y1 - y0) > 200 and (x1 - x0) > 500:
                pad = 10
                img = img[y0 + pad:y1 - pad, x0 + pad:x1 - pad]
                h, w = img.shape[:2]
    # fit to the panel aspect (centre crop)
    if w / h > aspect:
        nw = int(h * aspect)
        img = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
    else:
        nh = int(w / aspect)
        img = img[(h - nh) // 2:(h - nh) // 2 + nh]
    return img


def story_page(video, times, number, ppu):
    Wp, Hp = int(round(P.PAGE_W * ppu)), int(round(P.PAGE_H * ppu))
    page = np.empty((Hp, Wp, 3), np.float32)
    page[:] = PAGE_RGB
    for (x, y, w, h), t in zip(P.PANELS.values(), times):
        crop = panel_crop(grab(video, t))
        px, py, pw, ph = int(round(x * ppu)), int(round(y * ppu)), int(round(w * ppu)), int(round(h * ppu))
        page[py:py + ph, px:px + pw] = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_CUBIC)
    surf = surface_from_array(page)
    ctx = cairo.Context(surf)
    ctx.scale(ppu, ppu)
    P.draw_page_frame(ctx, number, list(P.PANELS.values()), stock=False)
    surf.flush()
    buf = np.ndarray((Hp, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :Wp]
    return np.stack([buf[..., 2], buf[..., 1], buf[..., 0]], -1).astype(np.float32) / 255.0


def flatten(tex):
    """premultiplied page texture -> rgb on white"""
    a = tex[..., 3:4]
    return np.clip(tex[..., :3] + (1 - a), 0, 1)


def fit_page(img, ppu):
    Wp, Hp = int(round(P.PAGE_W * ppu)), int(round(P.PAGE_H * ppu))
    return cv2.resize(img, (Wp, Hp), interpolation=cv2.INTER_AREA if img.shape[1] > Wp else cv2.INTER_CUBIC)


def separate(rgb):
    """-> (K, Y) plate coverages in [0,1]: yellow = the Flags' spot ink, black = everything else"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    chroma = np.clip((np.minimum(r, g) - b - 0.12) / 0.45, 0, 1)            # neutral ink/paper -> 0
    Y = chroma
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    K_neutral = np.clip(1 - lum / 0.945, 0, 1)
    K_flag = np.clip(1 - r / YELLOW[0], 0, 1) * 0.9                          # the flag's own shading
    K = K_neutral * (1 - chroma) + K_flag * chroma
    return K, Y


def write_pdf(path, pages, dpi):
    wpt = pages[0].shape[1] / dpi * 72
    hpt = pages[0].shape[0] / dpi * 72
    pdf = cairo.PDFSurface(str(path), wpt, hpt)
    ctx = cairo.Context(pdf)
    for img in pages:
        s = surface_from_array(img)
        ctx.save()
        ctx.scale(72 / dpi, 72 / dpi)
        ctx.set_source_surface(s, 0, 0)
        ctx.paint()
        ctx.restore()
        ctx.show_page()
    pdf.finish()


def impose(path, pages, dpi):
    """saddle stitch: sheet sides pair (n-1-i, i) / (i+1, n-2-i); 2-up side by side on a landscape sheet"""
    n = len(pages)
    assert n % 4 == 0
    pw, ph = pages[0].shape[1] / dpi * 72, pages[0].shape[0] / dpi * 72
    pdf = cairo.PDFSurface(str(path), pw * 2, ph)
    ctx = cairo.Context(pdf)
    for s in range(n // 4):
        for left, right in ((n - 1 - 2 * s, 2 * s), (2 * s + 1, n - 2 - 2 * s)):
            for k, idx in enumerate((left, right)):
                src = surface_from_array(pages[idx])
                ctx.save()
                ctx.translate(k * pw, 0)
                ctx.scale(72 / dpi, 72 / dpi)
                ctx.set_source_surface(src, 0, 0)
                ctx.paint()
                ctx.restore()
            ctx.show_page()
    pdf.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=float, default=300.0)
    ap.add_argument("--video", default=str(OUT / "film_video.mp4"))
    a = ap.parse_args()
    ppu = a.dpi * 8.5 / P.PAGE_W                     # page width = 8.5 in
    video = Path(a.video)
    out = OUT / "print"
    (out / "plates").mkdir(parents=True, exist_ok=True)
    pages = [fit_page(flatten(TP.with_flag(TP.cover_static(ppu), TP.cover_flag(ppu, 3.6))), ppu),
             fit_page(flatten(TP.inside_cover(ppu)), ppu)]
    for k, times in enumerate(STORY, 1):
        pages.append(story_page(video, times, k, ppu))
        print(f"page {k:2d}: {times}", flush=True)
    pages.append(fit_page(flatten(TP.inside_back(ppu)), ppu))
    back = TP.back_page(ppu)
    Wp, Hp = pages[0].shape[1], pages[0].shape[0]
    bk = np.ones((Hp, Wp, 3), np.float32)
    bk[:] = PAGE_RGB
    b = flatten(back)
    bh, bw = b.shape[:2]
    sc = min(Wp * 0.92 / bw, Hp * 0.92 / bh)
    b = cv2.resize(b, (int(bw * sc), int(bh * sc)), interpolation=cv2.INTER_AREA)
    y0, x0 = (Hp - b.shape[0]) // 2, (Wp - b.shape[1]) // 2
    bk[y0:y0 + b.shape[0], x0:x0 + b.shape[1]] = b
    pages.append(bk)
    while len(pages) % 4:
        pages.insert(-1, np.ones_like(pages[0]))
    write_pdf(out / "THE_FLAG_GAME_tract.pdf", pages, a.dpi)
    impose(out / "THE_FLAG_GAME_tract_imposed.pdf", pages, a.dpi)
    for i, pg in enumerate(pages):
        K, Y = separate(pg)
        cv2.imwrite(str(out / "plates" / f"p{i:02d}_K.png"), np.clip((1 - K) * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(str(out / "plates" / f"p{i:02d}_Y.png"), np.clip((1 - Y) * 255, 0, 255).astype(np.uint8))
    print(f"{len(pages)} pages -> {out}")


if __name__ == "__main__":
    main()

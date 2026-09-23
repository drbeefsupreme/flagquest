"""s06 helper: the Babel glyph atlas — shouted gibberish in ~40 real writing systems.

Strips are shaped with PIL + libraqm (so Arabic joins, Devanagari stacks, etc.), rendered once as
8-bit alpha masks and cached in cache/s06/glyphs.npz. Each strip is random letters of one script:
never real words, never anything flag-like (no symbol fonts, no Egyptian hieroglyphs).
"""
import json
import subprocess

import cairo
import numpy as np

# (name, fontconfig pattern, [(lo, hi) code point ranges], mode)
#   mode: "word" = letters grouped in short words; "block" = no spaces (CJK-like); "abugida" = consonant+sign
SCRIPTS = [
    ("latin", "Noto Sans:bold", [(0x41, 0x5A), (0xC0, 0xDD), (0x106, 0x17E)], "word"),
    ("greek", "Noto Sans:bold", [(0x391, 0x3A9), (0x3B1, 0x3C9)], "word"),
    ("cyrillic", "Noto Sans:bold", [(0x410, 0x44F), (0x404, 0x40F)], "word"),
    ("armenian", "Noto Sans Armenian:bold", [(0x531, 0x556), (0x561, 0x586)], "word"),
    ("georgian", "Noto Sans Georgian:bold", [(0x10D0, 0x10F0)], "word"),
    ("hebrew", "Noto Sans Hebrew:bold", [(0x5D0, 0x5EA)], "word"),
    ("arabic", "Noto Sans Arabic:bold", [(0x628, 0x63A), (0x641, 0x64A)], "word"),
    ("syriac", "Noto Sans Syriac", [(0x710, 0x72C)], "word"),
    ("thaana", "Noto Sans Thaana:bold", [(0x780, 0x7A5)], "word"),
    ("nko", "Noto Sans NKo", [(0x7CA, 0x7EA)], "word"),
    ("devanagari", "Noto Sans Devanagari:bold", [(0x915, 0x939)], "abugida:0x93E-0x94C"),
    ("bengali", "Noto Sans Bengali:bold", [(0x995, 0x9B9)], "abugida:0x9BE-0x9CC"),
    ("gujarati", "Noto Sans Gujarati:bold", [(0xA95, 0xAB9)], "abugida:0xABE-0xACC"),
    ("tamil", "Noto Sans Tamil:bold", [(0xB95, 0xBB9)], "abugida:0xBBE-0xBCC"),
    ("kannada", "Noto Sans Kannada:bold", [(0xC95, 0xCB9)], "abugida:0xCBE-0xCCC"),
    ("sinhala", "Noto Sans Sinhala:bold", [(0xD9A, 0xDC6)], "abugida:0xDCF-0xDDE"),
    ("thai", "Noto Sans Thai:bold", [(0xE01, 0xE2E)], "block"),
    ("khmer", "Noto Sans Khmer:bold", [(0x1780, 0x17A2)], "abugida:0x17B6-0x17C5"),
    ("myanmar", "Noto Sans Myanmar:bold", [(0x1000, 0x1020)], "abugida:0x102B-0x1032"),
    ("javanese", "Noto Sans Javanese:bold", [(0xA98F, 0xA9B2)], "block"),
    ("ethiopic", "Noto Sans Ethiopic:bold", [(0x1200, 0x1357)], "word"),
    ("cherokee", "Noto Sans Cherokee:bold", [(0x13A0, 0x13F4)], "word"),
    ("syllabics", "Noto Sans Canadian Aboriginal:bold", [(0x1401, 0x166C)], "word"),
    ("vai", "Noto Sans Vai", [(0xA500, 0xA60B)], "word"),
    ("yi", "Noto Sans Yi", [(0xA000, 0xA48C)], "block"),
    ("tifinagh", "Noto Sans Tifinagh", [(0x2D30, 0x2D67)], "word"),
    ("runic", "Noto Sans Runic", [(0x16A0, 0x16EA)], "word"),
    ("ogham", "Noto Sans Ogham", [(0x1681, 0x169A)], "word"),
    ("glagolitic", "Noto Sans Glagolitic", [(0x2C00, 0x2C2E)], "word"),
    ("old_turkic", "Noto Sans Old Turkic", [(0x10C00, 0x10C48)], "word"),
    ("old_persian", "Noto Sans Old Persian", [(0x103A0, 0x103C3)], "word"),
    ("cuneiform", "Noto Sans Cuneiform", [(0x12000, 0x1236E)], "block"),
    ("linear_b", "Noto Sans Linear B", [(0x10000, 0x1004D)], "block"),
    ("phoenician", "Noto Sans Phoenician", [(0x10900, 0x10915)], "word"),
    ("adlam", "Noto Sans Adlam:bold", [(0x1E922, 0x1E943)], "word"),
    ("hangul", "Noto Sans CJK KR:bold", [(0xAC00, 0xD7A3)], "block"),
    ("han", "Noto Sans CJK SC:bold", [(0x4E00, 0x9FA5)], "block"),
    ("kana", "Noto Sans CJK JP:bold", [(0x30A1, 0x30FA), (0x3041, 0x3096)], "block"),
]


def _fc(pattern):
    out = subprocess.run(["fc-match", "-f", "%{file}:%{index}", pattern], capture_output=True, text=True).stdout
    f, _, i = out.rpartition(":")
    return f, int(i or 0)


def _cmap(path, index):
    from fontTools.ttLib import TTFont
    ft = TTFont(path, fontNumber=index, lazy=True)
    return set(ft.getBestCmap().keys())


def _pool(ranges, cmap):
    return [c for lo, hi in ranges for c in range(lo, hi + 1) if c in cmap]


def build(path_npz, n_per=7, size=64, seed=606):
    """render n_per strips for every script -> npz of alpha masks + json meta"""
    from PIL import Image, ImageDraw, ImageFont
    rng = np.random.default_rng(seed)
    masks, meta = [], []
    lp, li = _fc("Noto Sans:bold")
    latin = ImageFont.truetype(lp, size, index=li, layout_engine=ImageFont.Layout.RAQM)
    for name, pat, ranges, mode in SCRIPTS:
        fpath, idx = _fc(pat)
        cm = _cmap(fpath, idx)
        pool = _pool(ranges, cm)
        signs = []
        if mode.startswith("abugida"):
            lo, hi = (int(x, 16) for x in mode.split(":")[1].split("-"))
            signs = _pool([(lo, hi)], cm)
        if len(pool) < 4:
            continue
        font = ImageFont.truetype(fpath, size, index=idx, layout_engine=ImageFont.Layout.RAQM)
        for k in range(n_per):
            pieces = []
            nw = int(rng.integers(1, 4)) if mode != "block" else 1
            for _ in range(nw):
                L = int(rng.integers(2, 6)) if mode != "block" else int(rng.integers(2, 5))
                w = ""
                for _ in range(L):
                    w += chr(int(rng.choice(pool)))
                    if signs and rng.random() < 0.55:
                        w += chr(int(rng.choice(signs)))
                pieces.append((w, font, size * 0.32))
            r = rng.random()
            p = "!" if r < 0.45 else ("?!" if r < 0.6 else ("!!" if r < 0.72 else ""))
            if p:
                f2 = font if all(ord(c) in cm for c in p) else latin
                pieces[-1] = (pieces[-1][0], pieces[-1][1], size * 0.04)
                pieces.append((p, f2, 0))
            B = int(size * 1.4)
            im = Image.new("L", (int(size * 1.2 * (sum(len(s) for s, _, _ in pieces) + 4)), int(size * 2.4)), 0)
            dr = ImageDraw.Draw(im)
            x = size * 0.3
            try:
                for s, f, gap in pieces:
                    dr.text((x, B), s, fill=255, font=f, anchor="ls")
                    x += f.getlength(s) + gap
            except Exception:
                continue
            a = np.asarray(im, np.uint8)
            ys, xs = np.nonzero(a > 8)
            if len(xs) == 0:
                continue
            a = np.ascontiguousarray(a[max(0, ys.min() - 3):ys.max() + 4, max(0, xs.min() - 3):xs.max() + 4])
            masks.append(a)
            meta.append({"script": name, "text": "".join(s for s, _, _ in pieces), "w": int(a.shape[1]),
                         "h": int(a.shape[0])})
    np.savez_compressed(path_npz, meta=np.array([json.dumps(meta)]), **{f"g{i:04d}": m for i, m in enumerate(masks)})
    return load(path_npz)


def load(path_npz):
    z = np.load(path_npz)
    meta = json.loads(str(z["meta"][0]))
    return {"meta": meta, "masks": [z[f"g{i:04d}"] for i in range(len(meta))]}


def get(S):
    p = S.cache / "glyphs.npz"
    if p.exists():
        try:
            return load(p)
        except Exception:
            pass
    return build(p)


# ---------------------------------------------------------------- cairo surfaces (lazy, per process)
_SURF = {}


def surface(atlas, i):
    """A8 cairo surface for glyph strip i (built lazily in each worker)"""
    key = (id(atlas), i)
    e = _SURF.get(key)
    if e is None:
        m = atlas["masks"][i]
        h, w = m.shape
        stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, w)
        buf = np.zeros((h, stride), np.uint8)
        buf[:, :w] = m
        e = (cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_A8, w, h, stride), buf)
        _SURF[key] = e
    return e[0]


def draw_strip(ctx, atlas, i, x, y, height, color, alpha=1.0, center=True):
    """paint strip i scaled to `height` design px; (x, y) = centre (or top-left)"""
    m = atlas["masks"][i]
    h, w = m.shape
    k = height / h
    ctx.save()
    if center:
        ctx.translate(x - w * k / 2, y - h * k / 2)
    else:
        ctx.translate(x, y)
    ctx.scale(k, k)
    r, g, b = color
    ctx.set_source_rgba(r, g, b, alpha)
    ctx.mask_surface(surface(atlas, i), 0, 0)
    ctx.restore()
    return w * k

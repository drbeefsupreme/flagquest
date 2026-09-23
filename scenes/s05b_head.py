"""s05b PHILOSOPHY: a colossal marble philosopher head (a Socratic herm), sculpted as a relief height field from
~200 hand-placed vector forms (dome, brow, snub nose, deep-set blank eyes, moustache and beard locks with drilled
curls, hair wreath, herm plinth), each rasterised, blurred into a rounded bump and combined (max / add / carve);
lit per pixel (key + cool fill + sky rim + cavity occlusion + specular) with marble veins.
Built once, cached to cache/s05b/head_*.npz.

texture: TEX_W x TEX_H px, TEX_K px per head unit; head units are centred at (0, 0), crown at y ~ -820,
herm base at y ~ +945.
"""
import math

import cairo
import cv2
import numpy as np

from vx import CACHE, col

TEX_K = 1.4
UW, UH = 1300, 1900
TEX_W, TEX_H = int(UW * TEX_K), int(UH * TEX_K)
CROWN = (0.0, -805.0)          # head units: where the Flag is planted
VERSION = "v11"


class Sculpt:
    def __init__(self):
        self.H = np.zeros((TEX_H, TEX_W), np.float32)

    def _raster(self, draw, pad):
        """rasterise a path (head units) into a padded crop; returns (mask, y0, x0)"""
        rec = cairo.RecordingSurface(cairo.CONTENT_ALPHA, None)
        c = cairo.Context(rec)
        c.scale(TEX_K, TEX_K)
        c.translate(UW / 2, UH / 2)
        draw(c)
        x0, y0, x1, y1 = c.fill_extents()
        x0, y0 = c.user_to_device(x0, y0)
        x1, y1 = c.user_to_device(x1, y1)
        x0, y0 = int(max(0, math.floor(x0 - pad))), int(max(0, math.floor(y0 - pad)))
        x1, y1 = int(min(TEX_W, math.ceil(x1 + pad))), int(min(TEX_H, math.ceil(y1 + pad)))
        if x1 <= x0 or y1 <= y0:
            return None, 0, 0
        s = cairo.ImageSurface(cairo.FORMAT_A8, x1 - x0, y1 - y0)
        c2 = cairo.Context(s)
        c2.translate(-x0, -y0)
        c2.scale(TEX_K, TEX_K)
        c2.translate(UW / 2, UH / 2)
        draw(c2)
        c2.fill()
        s.flush()
        a = np.ndarray((y1 - y0, s.get_stride()), np.uint8, s.get_data())[:, :x1 - x0].astype(np.float32) / 255
        return a, y0, x0

    def form(self, draw, h, rnd, op="max", k=1.0):
        """a sculpted form shaped like path `draw` (head units).
        max/add: rounded 'pillow' of height h whose edges round over radius `rnd` (head units)
                 (circular fillet from the silhouette inward: a full dome when rnd >= half the width);
        carve:   a soft groove/hollow of depth h, softness rnd."""
        if op == "carve":
            sig = max(0.6, rnd * TEX_K)
            m, y0, x0 = self._raster(draw, sig * 3)
            if m is None:
                return
            b = cv2.GaussianBlur(m, (0, 0), sig) if sig > 0.7 else m
            self.H[y0:y0 + b.shape[0], x0:x0 + b.shape[1]] -= b * h
            return
        m, y0, x0 = self._raster(draw, 3)
        if m is None:
            return
        d = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5) / TEX_K
        d = cv2.GaussianBlur(d, (0, 0), max(1.0, 0.35 * rnd * TEX_K)) * (m > 0.5)   # melt the medial-axis ridge
        u = np.clip(d / max(rnd, 1e-3), 0, 1)
        if op == "max":
            p = np.sin(u * (math.pi / 2))                  # silhouette masses: steep at the edge, flat on top
        else:
            p = u * u * (3 - 2 * u)                        # surface forms melt into what they sit on
            p = 0.5 * p + 0.5 * np.sin(u * (math.pi / 2)) * p
        p = cv2.GaussianBlur(p.astype(np.float32), (0, 0), 1.5)
        b = p * h
        R = self.H[y0:y0 + b.shape[0], x0:x0 + b.shape[1]]
        if op == "max":
            R[:] = np.maximum(R, b)
        elif op == "add":
            R += b * k
        elif op == "over":                                  # sits on the snapshot surface; higher wins
            B0 = self.B0[y0:y0 + b.shape[0], x0:x0 + b.shape[1]]
            R[:] = np.maximum(R, B0 + b)

    def snapshot(self):
        self.B0 = self.H.copy()


def _ell(x, y, a, b, rot=0.0):
    def d(c):
        c.save()
        c.translate(x, y)
        c.rotate(rot)
        c.scale(a, b)
        c.arc(0, 0, 1, 0, 2 * math.pi)
        c.restore()
    return d


def _poly(pts, smooth=True):
    def d(c):
        P = list(pts)
        if not smooth or len(P) < 3:
            c.move_to(*P[0])
            for p in P[1:]:
                c.line_to(*p)
            c.close_path()
            return
        n = len(P)
        c.move_to(*P[0])
        for i in range(n):
            p0, p1, p2, p3 = P[i - 1], P[i], P[(i + 1) % n], P[(i + 2) % n]
            c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
            c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
            c.curve_to(*c1, *c2, *p2)
        c.close_path()
    return d


def _stroke(pts, w):
    """a thick polyline (as a filled path)"""
    def d(c):
        c.save()
        c.move_to(*pts[0])
        for p in pts[1:]:
            c.line_to(*p)
        c.set_line_width(w)
        c.set_line_cap(cairo.LINE_CAP_ROUND)
        c.set_line_join(cairo.LINE_JOIN_ROUND)
        path = c.copy_path()
        c.new_path()
        c.append_path(path)
        c.restore()
        c.set_line_width(w)
        c.set_line_cap(cairo.LINE_CAP_ROUND)
        # convert stroke to fill region via stroke-to-path emulation: approximate with circles along the line
        c.new_path()
        P = np.array(pts, float)
        L = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        n = max(2, int(L[-1] / (w * 0.3)) + 1)
        for s in np.linspace(0, L[-1], n):
            x = np.interp(s, L, P[:, 0])
            y = np.interp(s, L, P[:, 1])
            c.new_sub_path()
            c.arc(x, y, w / 2, 0, 2 * math.pi)
    return d


def _lock(x, y, length, width, ang, curl=1, bend=0.35):
    """a teardrop lock of hair/beard hanging from (x, y) toward angle `ang` (0 = down), ending in a curl"""
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    n = 14
    for i in range(n + 1):
        u = i / n
        wdt = width * (0.55 + 0.45 * math.sin(math.pi * min(1.0, u * 1.15))) * (1 - 0.35 * u)
        off = curl * bend * width * math.sin(u * math.pi * 1.2)
        pts.append((off, u * length, wdt))
    left = [(-p[2] / 2 + p[0], p[1]) for p in pts]
    right = [(p[2] / 2 + p[0], p[1]) for p in pts][::-1]
    P = left + right
    Q = [(x + px * ca - py * sa, y + px * sa + py * ca) for px, py in P]
    tip = (x + pts[-1][0] * ca - pts[-1][1] * sa, y + pts[-1][0] * sa + pts[-1][1] * ca)
    spine = [(x + p[0] * ca - p[1] * sa, y + p[0] * sa + p[1] * ca) for p in pts]
    return Q, tip, spine


def build():
    path = CACHE / "s05b" / f"head_{TEX_W}x{TEX_H}_{VERSION}.npz"
    if path.exists():
        z = np.load(path)
        return z["rgba"], z["mask"]
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(399)
    S = Sculpt()
    F = S.form
    # ------------------------------------------------ great masses (rounded volumes)
    F(_ell(0, -380, 432, 438), 330, 420)                     # bald dome
    F(_poly([(-330, -250), (0, -290), (330, -250), (405, -60), (380, 160), (300, 330), (0, 400), (-300, 330),
             (-380, 160), (-405, -60)]), 320, 230)            # face / skull sides
    F(_poly([(-395, -20), (395, -20), (370, 300), (250, 600), (0, 720), (-250, 600), (-370, 300)]), 300, 190)  # beard
    F(_poly([(-262, 540), (262, 540), (256, 900), (-256, 900)], False), 250, 150)   # neck
    F(_poly([(-335, 885), (335, 885), (335, 945), (-335, 945)], False), 262, 8)     # herm cap
    for s in (-1, 1):
        F(_ell(s * 232, 55, 150, 118), 40, 110, "add")        # cheekbones
        F(_ell(s * 440, -70, 46, 104, s * 0.12), 250, 60)    # ears
        F(_ell(s * 452, -70, 30, 74, s * 0.12), 36, 9, "carve")
            # brow: gull wing + glabella; forehead wrinkles
    for s in (-1, 1):
        F(_poly([(s * 20, -200), (s * 160, -232), (s * 320, -204), (s * 368, -160), (s * 170, -160), (s * 30, -146)]),
          48, 34, "add")
    F(_ell(0, -175, 64, 48), 30, 40, "add")
    for k, yy in enumerate((-350, -305, -262)):
        F(_stroke([(-250 + k * 25, yy + 14), (-120, yy - 4), (0, yy + 6), (120, yy - 4), (250 - k * 25, yy + 14)],
                  10), 9, 5, "carve")
    # eyes: deep almond sockets, blank eyeballs, heavy lids, bags
    for s in (-1, 1):
        ex = s * 152
        F(_poly([(ex - 122, -84), (ex - 40, -142), (ex + 50, -140), (ex + 124, -92), (ex + 60, -36), (ex - 50, -34)]),
          62, 22, "carve")
        F(_poly([(ex - 80, -84), (ex - 20, -114), (ex + 40, -112), (ex + 82, -86), (ex + 34, -58), (ex - 34, -58)]),
          52, 40, "add")
        F(_poly([(ex - 98, -94), (ex - 30, -138), (ex + 40, -136), (ex + 100, -98), (ex + 40, -114), (ex - 30, -116)]),
          26, 14, "add")
        F(_poly([(ex - 72, -52), (ex, -40), (ex + 76, -54), (ex + 20, -24), (ex - 30, -26)]), 18, 16, "add")
        for k in range(3):
            a = -0.5 + k * 0.45
            x0 = ex + s * 118
            F(_stroke([(x0, -86), (x0 + s * 60 * math.cos(a), -86 + 60 * math.sin(a))], 6), 7, 3, "carve")
    # nose: broad snub
    F(_poly([(-36, -160), (36, -160), (72, 40), (98, 118), (-86, 118), (-62, 40)]), 100, 48, "add")
    F(_ell(8, 108, 88, 68), 55, 60, "add")
    for s in (-1, 1):
        F(_ell(8 + s * 84, 136, 56, 44), 42, 40, "add")
        F(_ell(8 + s * 48, 168, 22, 12, s * 0.3), 40, 5, "carve")
        F(_stroke([(8 + s * 118, 124), (8 + s * 172, 210), (8 + s * 196, 268)], 14), 16, 7, "carve")
    # moustache: two sweeping locks with strands
    for s in (-1, 1):
        Q, tip, spine = _lock(8 + s * 16, 186, 260, 128, -s * 1.12, curl=s, bend=0.25)
        F(_poly(Q), 62, 40, "add")
        for k in range(4):
            off = (k - 1.5) * 22
            P = [(x + off * 0.4, y + off * 0.3) for x, y in spine[1:-3]]
            F(_stroke(P, 5), 8, 2.5, "carve")
    F(_ell(8, 318, 70, 26), 30, 22, "add")                # lower lip
    F(_ell(8, 297, 82, 8), 26, 4, "carve")                # mouth line
    # beard locks: rows hanging from the moustache to the tip; upper rows drawn last (overlap lower ones)
    rows = [(280, 350, 124), (352, 350, 122), (425, 322, 118), (500, 272, 112), (565, 200, 104), (615, 115, 96)]
    S.snapshot()
    for ri, (yy, half, wdt) in reversed(list(enumerate(rows))):
        n = max(2, int(2 * half / (wdt * 0.72)))
        for j in range(n):
            x = -half + (j + 0.5) * 2 * half / n + rng.uniform(-10, 10)
            if ri == 0 and abs(x) < 130:
                continue
            ang = -x / 900.0 + rng.uniform(-0.12, 0.12)
            curl = 1 if (j + ri) % 2 else -1
            Q, tip, spine = _lock(x, yy - 70, (160 if ri < 4 else 125) + rng.uniform(-10, 20), wdt, ang, curl=curl)
            F(_poly(Q), 48 + (5 - ri) * 7, 34, "over")
            for k in range(3):
                off = (k - 1) * wdt * 0.22
                P = [(px + off, py) for px, py in spine[2:-3]]
                F(_stroke(P, 4.5), 7, 2.2, "carve")
            F(_ell(tip[0], tip[1], 36, 32), 50 + (5 - ri) * 7, 30, "over")
            F(_ell(tip[0] + curl * 7, tip[1] - 2, 9, 9), 20, 3, "carve")
    # hair wreath around the sides of the bald crown
    S.snapshot()
    for s in (-1, 1):
        for k in range(10):
            a = -0.9 + k * 0.26
            x = s * (392 + 28 * math.cos(a * 1.4))
            y = -320 + k * 40
            Q, tip, spine = _lock(x, y, 125 + rng.uniform(0, 30), 74, s * (0.35 + 0.11 * k), curl=s, bend=0.4)
            F(_poly(Q), 46 + k * 2, 30, "over")
            for kk in range(2):
                off = (kk - 0.5) * 16
                F(_stroke([(px + off, py) for px, py in spine[2:-3]], 4), 6, 2, "carve")
            F(_ell(tip[0], tip[1], 28, 25), 46 + k * 2, 22, "over")
            F(_ell(tip[0] + s * 5, tip[1], 7, 7), 16, 2.5, "carve")
    Hh = S.H
    mask = (Hh > 2.0).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.9)
    return _shade(Hh, mask, path)


def _noise(shape, sigma, seed):
    r = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    r = cv2.GaussianBlur(r, (0, 0), sigma)
    return r / (r.std() + 1e-6)


def _shade(Hh, mask, path):
    ys = (np.arange(TEX_H) + 0.5) / TEX_K - UH / 2
    xs = (np.arange(TEX_W) + 0.5) / TEX_K - UW / 2
    X, Y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    Hc = np.clip(Hh, 0, None)
    Hs = cv2.GaussianBlur(Hc, (0, 0), 0.8 * TEX_K)
    gy, gx = np.gradient(Hs, 1.0 / TEX_K)
    n = np.stack([-gx, -gy, np.ones_like(gx) * 1.0], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L1 = np.array([-0.5, -0.62, 0.6])
    L1 /= np.linalg.norm(L1)
    L2 = np.array([0.85, 0.1, 0.4])
    L2 /= np.linalg.norm(L2)
    dif = np.clip(n @ L1, 0, 1)
    wrap = np.clip((n @ L1 + 0.4) / 1.4, 0, 1)
    fill = np.clip(n @ L2, 0, 1)
    rim = np.clip(1 - n[..., 2], 0, 1) ** 2 * np.clip(n[..., 0] * 0.9 + 0.2, 0, 1)
    cav = np.clip((cv2.GaussianBlur(Hc, (0, 0), 16 * TEX_K) - Hc) / 50.0, 0, 1)
    cav2 = np.clip((cv2.GaussianBlur(Hc, (0, 0), 4 * TEX_K) - Hc) / 10.0, 0, 1)
    ao = (1 - 0.5 * cav) * (1 - 0.45 * cav2)
    lit = np.array(col("#F7F5F1"), np.float32)
    mid = np.array(col("#C3C6D2"), np.float32)
    shd = np.array(col("#5C6482"), np.float32)
    k = (0.65 * dif + 0.35 * wrap)[..., None]
    rgb = shd + (mid - shd) * np.clip(k * 1.7, 0, 1) + (lit - mid) * np.clip(k * 1.7 - 0.7, 0, 1)
    hv = L1 + np.array([0, 0, 1.0])
    hv /= np.linalg.norm(hv)
    spec = np.clip(n @ hv, 0, 1) ** 36
    rgb += spec[..., None] * 0.1
    rgb += fill[..., None] * np.array(col("#7488BC"), np.float32) * 0.14
    rgb += rim[..., None] * np.array(col("#CBD3FF"), np.float32) * 0.38
    rgb *= ao[..., None]
    w1 = _noise(X.shape, 50 * TEX_K, 5) * 2.4 + _noise(X.shape, 10 * TEX_K, 6) * 0.45
    v = np.abs(np.sin((X * 0.0035 + Y * 0.0022) * 7 + w1 * 2.0))
    vein = np.clip(1 - v / 0.05, 0, 1) * 0.2 + np.clip(1 - v / 0.22, 0, 1) * 0.04
    rgb = rgb * (1 - vein[..., None]) + np.array(col("#737C96"), np.float32) * vein[..., None]
    grain = _noise(X.shape, 1.0, 7) * 0.01
    rgb = np.clip(rgb + grain[..., None], 0, 1)
    rgba = np.concatenate([rgb * mask[..., None], mask[..., None]], -1).astype(np.float32)
    np.savez_compressed(path, rgba=rgba, mask=mask)
    return rgba, mask


def outline(mask, step=6):
    """outer contour of the head (texture px) as (n,2)"""
    m = (mask > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0, :].astype(np.float64)
    return c[::step]


def to_tex(x, y):
    """head units -> texture px"""
    return (x + UW / 2) * TEX_K, (y + UH / 2) * TEX_K

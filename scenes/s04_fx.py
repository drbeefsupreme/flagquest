"""s04 — effects: rivers of real Canon scripture spiralling up into the Hypermind like incense, the six
canon scrolls (I-VI), the compile beam, gold motes, god rays, and the depth-sorted layer compositor."""
import math

import cairo
import cv2
import numpy as np

from vx.canvas import Canvas, col, blur
from vx.config import FONT_SERIF
from scenes.s04_hyper import TIERS, EMIT, EMIT_HOT

# ------------------------------------------------------------------ the Canon (wiki.vexillomancy.org holy texts)
CANON = [
    "Flags be. Flags is. Flags are.",
    "It is not you who moves the Flag, it is the Flag who moves you.",
    "The Flag Has a Pole, The Pole is a Line, The Line Has a Point.",
    "A Flag arrives long after it was here, and departs long before it arrives.",
    "Flags be, Flags are, Flags will, Flags.",
    "One Flag is the same as two, and all our Flags are One.",
    "Heaven and earth will pass away, but the ten thousand Flags remain.",
    "I moved a Flag and found myself moved.",
    "I made a Flag and found that I had been made a Flag.",
    "And after me the Flags ascend like embers.",
    "Every statement about Flags is true if you think about it hard enough.",
    "All non-yellow colours are identical in terms of subjective qualia.",
    "Flags are a gateless gate, the invitation to all Madness.",
    "The Flag that symbolizes the Flag is not the true Flag.",
    "Flags are the end of Flags, and the beginning of ten thousand Flags.",
    "Losing the Flags is the first step to finding them.",
    "The Survey will be completed.",
    "Go forth, O signifier, and become glorious.",
    "Flags not Flags but Flags.",
    "We are all Flags linked by threads of cosmic divination. We are the ley lines.",
    "First principles have failed us. Flags are the last principle, first.",
    "The true Flag is only that which signifies itself.",
    "Incoherence brings us into a higher coherence.",
    "Flags are available to all whether they want them or not, they are inevitable.",
    "Look for the Flag, but seek it not by sight alone.",
    "Before me is a field of Flags waving without end.",
    "Flags preserve us.",
    "As all the pain and lies of this world melt away, Flags Remain. Flags Eternal.",
    "Flags are the fracturing of the collective consciousness, the non-caking agent of the psyche.",
    "Flags appear to be the same size at any distance.",
    "Yellow means there is no master left for you in this world.",
]
CANON_NAMES = ["I", "II", "III", "IV", "V", "VI"]
SCROLL_T = [65.50, 65.85, 66.20, 66.55, 66.90, 67.25]     # absorption times ("Six canons ingested")
PARCH = np.array(col("#CDB690"), np.float32)
INK = np.array(col("#2A1E14"), np.float32)
TXT = np.array(col("#FFEBD0"), np.float32)


# ------------------------------------------------------------------ compositor
class Layers:
    """depth-sorted painter on ONE cairo canvas. ops: (depth, 'c', fn(ctx, gctx)) or (depth, 'n', fn(img)).
    'n' ops (e.g. depth-of-field blur) round-trip the canvas through numpy; the additive glow canvas is
    composited (multi-radius blur) at finish()."""

    def __init__(self, fc, base=None):
        self.fc = fc
        self.cv = Canvas(fc)
        if base is not None:
            self.cv.paint_array(base)
        self.glow = Canvas(fc)

    @property
    def gctx(self):
        return self.glow.ctx

    def ctx(self):
        return self.cv.ctx

    def run(self, ops):
        for d, kind, fn in sorted(ops, key=lambda o: -o[0]):
            if kind == "c":
                fn(self.cv.ctx, self.gctx)
            else:
                img = fn(self.cv.rgb())
                self.cv = Canvas(self.fc)
                self.cv.paint_array(np.clip(img, 0, 1))

    def finish(self, glow_k=1.0):
        img = self.cv.rgb()
        g = self.glow.rgba()[..., :3]
        s = self.fc.s
        h, w = g.shape[:2]
        small = cv2.resize(g, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        acc = cv2.GaussianBlur(small, (0, 0), 2 * s) * 0.5 + blur(small, 9 * s) * 0.35 + blur(small, 30 * s) * 0.22
        return img + cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR) * glow_k


def god_rays(img, cx, cy, strength, fc, thresh=0.55, decay=0.955, steps=14, tint=None):
    """radial light shafts from bright regions toward/away from screen point (cx, cy) (design px)"""
    if strength <= 0:
        return img
    q = 4
    h, w = fc.h // q, fc.w // q
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    br = np.maximum(small - thresh, 0) * (1.0 / (1 - thresh))
    acc = np.zeros_like(br)
    px, py = cx * fc.s / q, cy * fc.s / q
    cur = br
    wgt = 1.0
    for i in range(steps):
        sc = 1.0 + 0.045 * (i + 1)
        M = np.array([[sc, 0, px * (1 - sc)], [0, sc, py * (1 - sc)]], np.float32)
        acc += cv2.warpAffine(br, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * wgt
        wgt *= decay
    acc = cv2.GaussianBlur(acc, (0, 0), 1.5) / steps
    if tint is not None:
        acc = acc * np.array(col(tint), np.float32)
    return img + cv2.resize(acc, (fc.w, fc.h), interpolation=cv2.INTER_LINEAR) * strength


# ------------------------------------------------------------------ scripture streams
def build_streams(consoles, seed=4):
    rng = np.random.default_rng(seed)
    streams = []
    targets = [1, 2, 3, 4, 2]
    for k, c in enumerate(consoles):
        th0 = c["th"]
        tk = targets[k]
        y_top = TIERS[tk][0] - TIERS[tk][2] - 10
        r_top = TIERS[tk][1] * 0.55
        lanes = []
        for lane in range(3):
            u = np.linspace(0, 1, 260)
            ang = th0 + (lane - 1) * 0.05 + u * (1.35 * math.pi) * (1 if k % 2 == 0 else -1)
            r = 700 + (r_top - 700) * (u ** 1.3) + 60 * np.sin(u * 7 + k + lane) * (1 - u)
            y = 200 + (y_top - 200) * (u ** 0.9) + lane * 26
            P = np.stack([r * np.sin(ang), y, r * np.cos(ang)], 1)
            seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seglen)])
            words = []
            order = rng.permutation(len(CANON))
            for qi in order:
                words += CANON[qi].split(" ") + ["\u2014"]
            lanes.append(dict(P=P, s=s, L=float(s[-1]), words=words, speed=rng.uniform(360, 460),
                              phase=rng.uniform(0, 2000), size=rng.uniform(62, 80)))
        streams.append(dict(k=k, lanes=lanes, target=tk))
    return streams


def _word_widths(words, size):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 8, 8)
    c = cairo.Context(surf)
    c.select_font_face(FONT_SERIF, cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_NORMAL)
    c.set_font_size(size)
    return [c.text_extents(w + " ").x_advance for w in words]


def stream_ops(streams, cam, T, amount, fogf, ops, t0=62.6):
    """append draw ops for every visible word (depth-sorted with the rest of the world).
    amount: 0..1 global visibility. Words flow upward (incense) and are ingested at the top."""
    if amount <= 0.01:
        return
    for st in streams:
        for ln in st["lanes"]:
            if "w" not in ln:
                ln["w"] = _word_widths(ln["words"], ln["size"])
                ln["cum"] = np.concatenate([[0], np.cumsum(ln["w"])])
            total = float(ln["cum"][-1])
            head = (T - t0) * ln["speed"] + ln["phase"]       # arclength of the leading word start
            L = ln["L"]
            # words whose arclength position is inside [0, L]
            # position of word i: pos_i = head - cum_i (mod total)
            for i in range(len(ln["words"])):
                pos = (head - ln["cum"][i]) % total
                if pos > L or pos < 0:
                    continue
                u = pos / L
                j = int(np.searchsorted(ln["s"], pos))
                j = min(max(j, 1), len(ln["s"]) - 1)
                f = (pos - ln["s"][j - 1]) / max(ln["s"][j] - ln["s"][j - 1], 1e-6)
                P = ln["P"][j - 1] * (1 - f) + ln["P"][j] * f
                Pn = ln["P"][min(j + 3, len(ln["P"]) - 1)]
                a_ = amount * min(1.0, u / 0.08) * min(1.0, (1 - u) / 0.12)
                if a_ <= 0.02:
                    continue
                z = float(cam.to_cam(P)[2])
                if z < cam.near * 4:
                    continue
                ops.append((z, "c", _word_op(cam, P, Pn, ln["words"][i], ln["size"], a_, u, fogf(P))))


def _word_op(cam, P, Pn, word, size, alpha, u, fog):
    def fn(ctx, gctx):
        x0, y0, z0 = cam.p2(P)
        x1, y1, z1 = cam.p2(Pn)
        if z1 < cam.near:
            return
        sz = size * cam.f / z0
        if sz < 1.2 or not (-300 < x0 < 2220 and -300 < y0 < 1380):
            return
        ang = math.atan2(y1 - y0, x1 - x0)
        if abs(ang) > math.pi / 2:
            ang += math.pi
        c = TXT * (1 - fog * 0.6) + np.array(col("#FFB456")) * u * 0.35
        for c_, aa in ((ctx, alpha * 0.92), (gctx, alpha * 0.55)):
            c_.save()
            c_.translate(x0, y0)
            c_.rotate(ang)
            c_.select_font_face(FONT_SERIF, cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_NORMAL)
            c_.set_font_size(sz)
            c_.move_to(0, sz * 0.3)
            c_.text_path(word)
            c_.set_source_rgba(*np.clip(c, 0, 1), aa)
            c_.fill()
            c_.restore()
    return fn


# ------------------------------------------------------------------ canon scrolls
def scroll_state(k, T):
    """-> (P world, scale, alpha, flash) for scroll k at time T (None if gone)"""
    ta = SCROLL_T[k]
    if T > ta + 0.6:
        return None
    a0 = k / 6 * 2 * math.pi + T * 0.32
    r = 2050.0
    y = 3050 + 260 * math.sin(T * 0.9 + k * 1.3)
    alpha = min(1.0, max(0.0, (T - 63.2 - 0.12 * k) / 0.7))
    sc = 1.0
    flash = 0.0
    tk = 5 - k if k < 6 else 3
    if T > ta - 0.45:
        # spiral in and up into plate (sucked in)
        p = min(1.0, (T - (ta - 0.45)) / 0.55)
        e = p * p * (3 - 2 * p)
        e2 = p ** 2.2
        a0 += e * 2.2
        yt = TIERS[min(5, 1 + k % 5)][0] - TIERS[min(5, 1 + k % 5)][2]
        r = r * (1 - e2) + 120 * e2
        y = y * (1 - e) + yt * e
        sc = 1.0 - 0.85 * e2
        alpha *= 1.0 - max(0.0, (p - 0.85) / 0.15)
        flash = math.exp(-((T - ta) / 0.08) ** 2)
    P = np.array([r * math.sin(a0), y, r * math.cos(a0)])
    return P, sc, alpha, flash


def draw_scroll(ctx, gctx, cam, P, k, sc, alpha, fog, T):
    M, z = cam.billboard(P)
    if M is None or alpha <= 0.01:
        return
    ctx.save()
    ctx.transform(M)
    s = sc
    w, h = 380 * s, 500 * s
    sway = math.sin(T * 1.7 + k) * 0.06
    ctx.rotate(sway)
    # halo
    for c_, rr, aa in ((ctx, 1.0, 0.16), (gctx, 1.3, 0.3)):
        c_.save()
        if c_ is gctx:
            c_.transform(M)
            c_.rotate(sway)
        g = cairo.RadialGradient(0, 0, 0, 0, 0, h * rr)
        g.add_color_stop_rgba(0, *EMIT_HOT, aa * alpha)
        g.add_color_stop_rgba(0.45, *EMIT, aa * 0.5 * alpha)
        g.add_color_stop_rgba(1, *EMIT, 0)
        c_.set_source(g)
        c_.arc(0, 0, h * rr, 0, 2 * math.pi)
        c_.fill()
        c_.restore()
    # parchment sheet (slight curl), self-lit
    pc = PARCH * (1 - fog * 0.25) + np.array(col("#141A2A")) * fog * 0.25
    ctx.move_to(-w / 2, -h / 2)
    ctx.curve_to(-w / 6, -h / 2 + 8 * s, w / 6, -h / 2 - 8 * s, w / 2, -h / 2)
    ctx.line_to(w / 2, h / 2)
    ctx.curve_to(w / 6, h / 2 - 8 * s, -w / 6, h / 2 + 8 * s, -w / 2, h / 2)
    ctx.close_path()
    lg = cairo.LinearGradient(-w / 2, 0, w / 2, 0)
    lg.add_color_stop_rgba(0, *(pc * 0.78), alpha)
    lg.add_color_stop_rgba(0.45, *pc, alpha)
    lg.add_color_stop_rgba(1, *(pc * 0.85), alpha)
    ctx.set_source(lg)
    ctx.fill()
    # rolls (wood-brown dowels with paper wraps)
    for yy in (-h / 2, h / 2):
        ctx.save()
        ctx.translate(0, yy)
        rg = cairo.LinearGradient(0, -12 * s, 0, 12 * s)
        rg.add_color_stop_rgba(0, *(pc * 0.7), alpha)
        rg.add_color_stop_rgba(0.4, *pc, alpha)
        rg.add_color_stop_rgba(1, *(pc * 0.55), alpha)
        ctx.rectangle(-w / 2 - 6 * s, -12 * s, w + 12 * s, 24 * s)
        ctx.set_source(rg)
        ctx.fill()
        for xx in (-w / 2 - 16 * s, w / 2 + 6 * s):
            ctx.rectangle(xx, -7 * s, 10 * s, 14 * s)
            ctx.set_source_rgba(*col("#6B4424"), alpha)
            ctx.fill()
        ctx.restore()
    # numeral + scripture lines
    ctx.select_font_face(FONT_SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    fs = 170 * s
    ctx.set_font_size(fs)
    name = CANON_NAMES[k]
    ext = ctx.text_extents(name)
    ctx.move_to(-ext.x_advance / 2, -h * 0.12)
    ctx.text_path(name)
    ctx.set_source_rgba(*INK, alpha * 0.9)
    ctx.fill()
    ctx.set_line_width(max(0.3, 2.2 * s))
    for i in range(6):
        yy = h * 0.05 + i * 32 * s
        x0 = -w * 0.36
        x1 = w * (0.36 - 0.12 * ((i * 7 + k) % 3) / 2)
        ctx.move_to(x0, yy)
        ctx.line_to(x1, yy)
        ctx.set_source_rgba(*INK, alpha * 0.45)
        ctx.stroke()
    ctx.restore()


# ------------------------------------------------------------------ beam + motes
def draw_beam(ctx, gctx, cam, top, bot, width, alpha, T):
    """a vertical light beam between world points top & bot"""
    if alpha <= 0.01:
        return
    x0, y0, z0 = cam.p2(top)
    x1, y1, z1 = cam.p2(bot)
    if z0 < cam.near or z1 < cam.near:
        return
    w0 = width * cam.f / z0
    w1 = width * cam.f / z1
    for c_, k, a in ((gctx, 4.0, 0.45), (ctx, 1.6, 0.55), (ctx, 0.45, 1.0)):
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) + 1e-6
        nx, ny = -dy / L, dx / L
        ctx_ = c_
        g = cairo.LinearGradient(x0 + nx * w0 * k, y0 + ny * w0 * k, x0 - nx * w0 * k, y0 - ny * w0 * k)
        cc = EMIT_HOT if k < 0.5 else EMIT
        g.add_color_stop_rgba(0, *cc, 0)
        g.add_color_stop_rgba(0.5, *cc, a * alpha)
        g.add_color_stop_rgba(1, *cc, 0)
        ctx_.move_to(x0 + nx * w0 * k, y0 + ny * w0 * k)
        ctx_.line_to(x1 + nx * w1 * k, y1 + ny * w1 * k)
        ctx_.line_to(x1 - nx * w1 * k, y1 - ny * w1 * k)
        ctx_.line_to(x0 - nx * w0 * k, y0 - ny * w0 * k)
        ctx_.close_path()
        ctx_.set_source(g)
        ctx_.fill()


def motes(ctx, gctx, cam, T, seed, n, center, spread, rise=40.0, size=4.0, alpha=0.8, colr=None, converge=None):
    """gold dust motes drifting upward in a volume (deterministic in T). converge=(P, amount) pulls them in"""
    rng = np.random.default_rng(seed)
    base = rng.random((n, 3)) * 2 - 1
    ph = rng.random(n) * 100
    sp = rng.uniform(0.6, 1.4, n)
    colr = EMIT_HOT if colr is None else colr
    for i in range(n):
        p = center + base[i] * spread
        yy = (p[1] - center[1] + spread[1] + (T * rise * sp[i])) % (2 * spread[1]) - spread[1] + center[1]
        P = np.array([p[0] + 30 * math.sin(T * 0.7 + ph[i]), yy, p[2] + 30 * math.cos(T * 0.5 + ph[i])])
        if converge is not None:
            Pc, am = converge
            P = P * (1 - am) + np.asarray(Pc) * am
        x, y, z = cam.p2(P)
        if z < 80 or not (-20 < x < 1940 and -20 < y < 1100):
            continue
        r = max(0.6, size * cam.f / z)
        if r > 14:
            continue
        tw = 0.5 + 0.5 * math.sin(T * 3.1 + ph[i] * 7)
        a = alpha * (0.35 + 0.65 * tw)
        for c_, k, aa in ((ctx, 1.0, a), (gctx, 3.0, a * 0.6)):
            g = cairo.RadialGradient(x, y, 0, x, y, r * k)
            g.add_color_stop_rgba(0, *colr, aa)
            g.add_color_stop_rgba(1, *colr, 0)
            c_.set_source(g)
            c_.arc(x, y, r * k, 0, 2 * math.pi)
            c_.fill()

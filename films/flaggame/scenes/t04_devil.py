"""t04 — the Devil on his throne with a lyre (page 25 panel 3), 97.02–105.67.

"Listen, my children. The Flags are nothing but a... a... a..." — every word he letters is stamped over
with FLAGS; the FLAGS wall squeezes his panel from above and below; at devil_glitch (102.42) he is possessed
by the word: the print jams, his eyes turn to FLAGS, he screams FLAGS! FLAGS FLAGS FLAGS! and the wall closes.
"""
import math

import cairo
import numpy as np

from vx import *
from vx import ink, comic
from vx.chars import Character

from scenes import t04_world as tw
from scenes.t04_satan import draw_satan
from scenes import t04_type as ty

T_V1, T_GL = 97.30, 102.415
T_SHUT = 104.94            # last FLAGS ends: the wall has closed
HEAD = (1200.0, 420.0)             # the Devil's head centre in the panel (design px)


def devil_params(T, tl):
    """acting: sly Cheshire grin and a strum ('Listen, my children'), then each overwrite knocks him further
    off balance (worried brows, grimace, sweat), then possession (102.42): eyes roll white, jaw unhinged, jerks"""
    st_ = stamps(tl)
    n = 0.0
    last = None
    for ts, _ in st_:
        if T >= ts:
            n += 1
            last = ts
    flinch = math.exp(-(T - last) * 7) if last is not None else 0.0
    lean_in = smoothstep(97.7, 98.3, T) * (1 - smoothstep(99.4, 99.8, T))
    q = clamp(n / 4)
    p = dict(brow=lerp(1.0, -0.7, q ** 0.8) - 0.6 * flinch, lid=lerp(0.5, 1.0, max(q, flinch)),
             grin=lerp(1.0, 0.05, q ** 0.7), jaw=0.85 * tl.mouth("DEVIL", T), gaze=(-0.55 + 0.4 * q, -0.1),
             sweat=smoothstep(0.3, 0.9, q), tilt=-0.04 * lean_in + 0.06 * flinch * math.sin(T * 40),
             pluck=1.0 if T < 98.45 else 0.0, s=1.05 + 0.06 * lean_in, dy=18 * lean_in)
    if T >= T_GL - 0.04:
        u = T - T_GL
        jerk = 0.0
        for w in tl.words("V02"):
            if w["s"] <= T:
                jerk = math.exp(-(T - w["s"]) * 6)
        p.update(possessed=smoothstep(-0.04, 0.08, u), brow=1.0, grin=0.6, lid=1.0, sweat=0.0,
                 tilt=0.22 * jerk * math.sin(T * 23) - 0.1 * jerk, jitter=0.4 + 0.6 * jerk, snap=u,
                 jaw=0.9 * tl.mouth("DEVIL", T) + 0.2, s=1.12 + 0.05 * jerk, dy=-20 * jerk)
    return p


def G(v):
    return (v, v, v)


def stamps(tl):
    """(t_slam, which word) — the Devil's words keep getting overwritten by FLAGS"""
    ws = tl.words("V01")
    by = {}
    out = []
    for w in ws:
        by.setdefault(w["w"].lower(), []).append(w)
    nothing = by["nothing"][0]
    out.append((nothing["e"] - 0.08, "nothing"))
    a = by["a"]
    for k, w in enumerate(a[:3]):
        out.append((w["e"] + 0.22, f"a{k}"))
    return out


def band_heights(T, tl):
    """top / bottom FLAGS band heights (design px)"""
    st = stamps(tl)
    n = sum(1 for t, _ in st if t <= T)
    base = 104 + 30 * n
    # settle spring after each stamp
    last = max([t for t, _ in st if t <= T], default=None)
    if last is not None:
        base += 22 * math.exp(-(T - last) * 9) * math.sin((T - last) * 40)
    if T >= T_GL:
        ws = [w for w in tl.words("V02")]
        # each FLAGS! jerks the bands in; the last one closes the frame
        k = sum(1 for w in ws if w["s"] <= T)
        base += 22 * k
        if k >= 4:
            base += (H / 2 - base) * ease_in_out(clamp((T - ws[3]["s"]) / (T_SHUT - ws[3]["s"] - 0.15)), 3)
    return base, base


def _throne(ctx, cx, fy, s, t):
    """a tall carved gothic throne with bat wings and horned finials (grey, lit from the hellfire at left)"""
    ctx.save()
    # bat wings spread behind
    for side in (-1, 1):
        ctx.move_to(cx + side * 120 * s, fy - 620 * s)
        pts = [(cx + side * 520 * s, fy - 880 * s), (cx + side * 700 * s, fy - 700 * s),
               (cx + side * 610 * s, fy - 610 * s), (cx + side * 560 * s, fy - 520 * s),
               (cx + side * 470 * s, fy - 540 * s), (cx + side * 400 * s, fy - 430 * s),
               (cx + side * 300 * s, fy - 470 * s), (cx + side * 170 * s, fy - 360 * s)]
        ctx.curve_to(cx + side * 250 * s, fy - 800 * s, cx + side * 420 * s, fy - 900 * s, *pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.set_source(lin_grad(cx, fy - 900 * s, cx + side * 700 * s, fy - 400 * s,
                                [(0, G(0.30)), (1, G(0.10))]))
        ctx.fill_preserve()
        set_color(ctx, G(0.0))
        ctx.set_line_width(3 * s)
        ctx.stroke()
        # wing ribs
        ctx.set_line_width(4 * s)
        for q in (0.25, 0.5, 0.75):
            ctx.move_to(cx + side * 150 * s, fy - 600 * s)
            ctx.line_to(cx + side * (300 + 380 * q) * s, fy - (820 - 300 * q) * s)
            ctx.stroke()
    # back slab with a pointed arch
    bw, bh = 260 * s, 820 * s
    ctx.move_to(cx - bw, fy - 250 * s)
    ctx.line_to(cx - bw, fy - bh + 120 * s)
    ctx.curve_to(cx - bw, fy - bh - 60 * s, cx - 40 * s, fy - bh - 40 * s, cx, fy - bh - 140 * s)
    ctx.curve_to(cx + 40 * s, fy - bh - 40 * s, cx + bw, fy - bh - 60 * s, cx + bw, fy - bh + 120 * s)
    ctx.line_to(cx + bw, fy - 250 * s)
    ctx.close_path()
    ctx.set_source(lin_grad(cx - bw, 0, cx + bw, 0, [(0, G(0.62)), (0.35, G(0.40)), (1, G(0.14))]))
    ctx.fill_preserve()
    set_color(ctx, G(0.0))
    ctx.set_line_width(5 * s)
    ctx.stroke()
    # carved panel lines
    ctx.set_line_width(3 * s)
    for k in range(5):
        y = fy - (330 + 90 * k) * s
        ctx.move_to(cx - bw * 0.8, y)
        ctx.curve_to(cx - bw * 0.3, y - 30 * s, cx + bw * 0.3, y - 30 * s, cx + bw * 0.8, y)
        ctx.stroke()
    # horned finials
    for side in (-1, 1):
        x0 = cx + side * bw
        ctx.move_to(x0 - side * 26 * s, fy - bh + 110 * s)
        ctx.curve_to(x0 + side * 10 * s, fy - bh - 40 * s, x0 + side * 110 * s, fy - bh - 80 * s,
                     x0 + side * 90 * s, fy - bh - 200 * s)
        ctx.curve_to(x0 + side * 60 * s, fy - bh - 110 * s, x0 + side * 20 * s, fy - bh - 40 * s,
                     x0 + side * 26 * s, fy - bh + 110 * s)
        ctx.close_path()
        ctx.set_source(lin_grad(x0, fy - bh - 200 * s, x0, fy - bh + 110 * s, [(0, G(0.9)), (1, G(0.3))]))
        ctx.fill_preserve()
        set_color(ctx, G(0.0))
        ctx.set_line_width(4 * s)
        ctx.stroke()
    # seat block + arms
    ctx.rectangle(cx - bw - 60 * s, fy - 330 * s, (bw + 60 * s) * 2, 330 * s)
    ctx.set_source(lin_grad(cx - bw, 0, cx + bw, 0, [(0, G(0.55)), (0.5, G(0.30)), (1, G(0.12))]))
    ctx.fill_preserve()
    set_color(ctx, G(0.0))
    ctx.set_line_width(5 * s)
    ctx.stroke()
    for side in (-1, 1):
        ax = cx + side * (bw + 30 * s)
        ctx.rectangle(ax - 45 * s, fy - 470 * s, 90 * s, 470 * s)
        ctx.set_source(lin_grad(ax - 45 * s, 0, ax + 45 * s, 0, [(0, G(0.7)), (1, G(0.2))]))
        ctx.fill_preserve()
        set_color(ctx, G(0.0))
        ctx.stroke()
        # skull knob
        ctx.arc(ax, fy - 500 * s, 48 * s, 0, 2 * math.pi)
        ctx.set_source(rad_grad(ax - 18 * s, fy - 520 * s, 5 * s, 50 * s, [(0, G(0.97)), (1, G(0.35))]))
        ctx.fill_preserve()
        set_color(ctx, G(0.0))
        ctx.stroke()
        for ex in (-1, 1):
            ellipse(ctx, ax + ex * 16 * s, fy - 505 * s, 11 * s, 13 * s)
            set_color(ctx, G(0.0))
            ctx.fill()
    ctx.restore()


def _hell(ctx, t, T):
    """the throne room: hellfire pillar (white-hot) at left, smoke, steps"""
    ctx.save()
    ctx.rectangle(-100, -100, W + 200, H + 200)
    ctx.set_source(rad_grad(380, 420, 0, 1700, [(0, G(0.95)), (0.18, G(0.62)), (0.5, G(0.26)), (1, G(0.10))]))
    ctx.fill()
    # hellfire: tall white tongues licking up at left, flickering
    for k in range(14):
        ph = hash01(k, 3)
        x = 180 + 420 * hash01(k, 4)
        hgt = 380 + 420 * hash01(k, 5) * (0.8 + 0.2 * math.sin(T * (5 + 3 * ph) + k))
        wdt = 50 + 80 * hash01(k, 6)
        sway = 40 * math.sin(T * (2.3 + ph) + k * 1.7)
        y0 = 980
        ctx.move_to(x - wdt, y0)
        ctx.curve_to(x - wdt * 0.8, y0 - hgt * 0.5, x + sway - wdt * 0.2, y0 - hgt * 0.7, x + sway * 1.6, y0 - hgt)
        ctx.curve_to(x + sway + wdt * 0.2, y0 - hgt * 0.6, x + wdt * 0.9, y0 - hgt * 0.4, x + wdt, y0)
        ctx.close_path()
        ctx.set_source(lin_grad(0, y0 - hgt, 0, y0, [(0, (1, 1, 1), 0.0), (0.4, (1, 1, 1), 0.8), (1, (1, 1, 1), 1.0)]))
        ctx.fill()
    # steps / dais
    for k in range(3):
        y = 960 + k * 40
        ctx.rectangle(560 - k * 80, y, 1360 + k * 160, 40)
        ctx.set_source(lin_grad(0, y, 0, y + 40, [(0, G(0.55 - 0.1 * k)), (1, G(0.22 - 0.05 * k))]))
        ctx.fill_preserve()
        set_color(ctx, G(0.0))
        ctx.set_line_width(3)
        ctx.stroke()
    ctx.restore()


def _glitch(img, fc, T, tl):
    """the print jams on every FLAGS!: horizontal slabs slip sideways, some print in negative"""
    if T < T_GL:
        return img
    ws = tl.words("V02")
    e = 0.0
    kk = 0
    for k, w in enumerate(ws):
        if w["s"] <= T:
            e = max(e, math.exp(-(T - w["s"]) / 0.16))
            kk = k
    if e < 0.02:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    rng = np.random.default_rng(int(T * 24) * 7 + 3)
    n = 7 + kk * 3
    for _ in range(n):
        y0 = int(rng.uniform(0, h))
        hh = int(rng.uniform(0.01, 0.07) * h)
        dx = int(rng.normal(0, 60) * e * fc.s)
        y1 = min(h, y0 + hh)
        out[y0:y1] = np.roll(img[y0:y1], dx, axis=1)
        if rng.random() < 0.25 * e:
            out[y0:y1] = 1.0 - out[y0:y1] * 0.95
    return out


def render(fc, st, T):
    from scenes import t04_flagmaker as M
    tl = fc.tl
    if T < M.T_SCROLL1:
        # the page scroll: the wall (which scrolls itself, flag and all) slides up, the Devil's panel follows
        wall = M.shot_flood(fc, st, T)
        off = int(round(M.scroll_offset(T) * fc.s))
        if off <= 0:
            return wall
        dev = render_panel(fc, st, T)
        out = wall.copy()
        top = fc.h - off
        out[top:] = dev[:off]
        return out
    return render_panel(fc, st, T)


def render_panel(fc, st, T):
    tl = fc.tl
    t = T - T_V1
    cam = Camera(x=960 + 10 * math.sin(t * 0.3), y=540, zoom=1.0 + 0.10 * ease_in_out_sine(clamp(t / 5.0)))
    world = Canvas(fc, bg=G(0.2))
    ctx = world.ctx
    ctx.save()
    cam.apply(ctx)
    _hell(ctx, t, T)
    _throne(ctx, HEAD[0], 1160, 1.15, t)
    p = devil_params(T, tl)
    hx, hy = HEAD
    jit = p.pop("jitter", 0.0)
    if jit > 0:
        hx += 14 * jit * math.sin(T * 97.0)
        hy += 8 * jit * math.sin(T * 71.0 + 1.3)
    anc = draw_satan(ctx, hx, hy + p.pop("dy", 0.0), p.pop("s", 1.05), p, T)
    ctx.restore()
    img = M_engrave(fc, world.rgb(), cam)
    img = _glitch(img, fc, T, tl)
    bt, bb = band_heights(T, tl)
    top = Canvas(fc)
    c = top.ctx
    # panel borders + FLAGS bands (comic.flags_flood), on paper
    for (y0, hh, seed) in ((0, bt, 1), (H - bb, bb, 2)):
        c.save()
        c.rectangle(0, y0, W, hh)
        set_color(c, ink.PAPER)
        c.fill()
        c.restore()
        comic.flags_flood(c, (0, y0, W, hh), 1.0 if T >= T_GL else 0.62, t=T - 97.0, size=46, seed=seed,
                          pulse=1.0 if T >= T_GL else 0.0)
    c.save()
    set_color(c, ink.INK)
    c.set_line_width(8)
    c.move_to(-10, bt)
    c.line_to(W + 10, bt)
    c.move_to(-10, H - bb)
    c.line_to(W + 10, H - bb)
    c.stroke()
    c.restore()
    ms = cam.to_screen(*anc["mouth"])
    if T < T_GL + 0.1:
        _v01(c, fc, T, ms, bt)
    else:
        _possessed_type(c, fc, T, anc, cam)
        comic.say(c, fc, "V02", 760, max(bt + 150, 300), tail=ms, w=1100)
    return top.over(img)


def M_engrave(fc, grey, cam):
    from scenes import t04_flagmaker as M
    return M.engrave(fc, grey, cam)


def _v01(c, fc, T, ms, bt):
    """V01: his balloon, lettered as spoken — and overwritten: each failed word flips to FLAGS and a rubber
    stamp FLAGS slams on the balloon"""
    tl = fc.tl
    st_ = stamps(tl)
    n = 0.0
    for (ts, _) in st_:
        n += clamp((T - ts) / 0.12)
    u = clamp(n / len(st_)) * 0.72
    bb = comic.say(c, fc, "V01", 600, bt + 190, tail=ms, w=820, hold=0.0, flagify=u)
    if bb is None:
        return
    x0, y0, x1, y1 = bb
    for j, (ts, key) in enumerate(st_):
        if T >= ts:
            fx = [0.30, 0.62, 0.80, 0.48][j]
            fy = [0.35, 0.62, 0.36, 0.58][j]
            ty.stamp(c, x0 + (x1 - x0) * fx - 150, y0 + (y1 - y0) * fy - 40, 300, 80, T - ts, seed=j + 3,
                     size=62 + 8 * j, rot=[-0.18, 0.12, -0.08, 0.2][j])


def _possessed_type(c, fc, T, anc, cam):
    """possessed by the word: FLAGS printed in both rolled-back eyes, and FLAGS pouring out of his mouth,
    flying at the camera"""
    mx, my = cam.to_screen(*anc["mouth"])
    t = T - T_GL
    words = []
    for key in ("eye_l", "eye_r"):
        ex, ey = cam.to_screen(*anc[key])
        words.append((ex, ey - 2, 0.0, 22 * cam.zoom * (1 + 0.08 * math.sin(T * 31)), clamp(t / 0.06), ty.WORD))
    rate = 26.0
    n = int(min(70, t * rate))
    for k in range(n):
        age = t - k / rate
        if age > 1.6:
            continue
        ang = -2.2 + 1.7 * hash01(k, 3)                 # a fan of words erupting up and out
        v = 700 + 500 * hash01(k, 4)
        x = mx + math.cos(ang) * v * age
        y = my + math.sin(ang) * v * age + 260 * age * age
        s = 16 + 120 * age ** 1.3
        words.append((x, y, (hash01(k, 5) - 0.5) * 1.2 * age, s, clamp(1.6 - age), ty.WORD))
    ty.draw_words(c, words, keyline=0.16)


def events(S, st):
    tl = S.tl
    ev = []
    ev.append(dict(t=97.33, kind="lyre", strength=0.6, pan=0.2, desc="the Devil strums his lyre (Listen...)"))
    for j, (ts, key) in enumerate(stamps(tl)):
        ev.append(dict(t=ts, kind="thunk", strength=0.7 + 0.08 * j, pan=-0.3,
                       desc=f"rubber stamp FLAGS slams over the Devil's word ({key}); the FLAGS bands close in"))
    for k, w in enumerate(tl.words("V02")):
        ev.append(dict(t=w["s"], kind="glitch", strength=0.8 + 0.05 * k, pan=0.0,
                       desc="print jam: slabs of the engraving slip, the Devil jerks (FLAGS!)"))
    ev.append(dict(t=T_GL, kind="snap", strength=0.9, pan=0.2, desc="the lyre's strings snap"))
    ev.append(dict(t=T_SHUT - 0.05, kind="slam", strength=1.0, pan=0.0, desc="the FLAGS wall slams shut over him"))
    return ev

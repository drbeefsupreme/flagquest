"""s02 shot C (40.6-47.5): a different feed for every face.

C1 40.60-42.45  dusk after the flood: isolated people stand waist-deep in slop, far apart, each hunched over a
                phone that lights their face from below in its own colour (never yellow). Swells of slop roll in.
C2 42.45-47.00  a whip-push into the nearest face becomes a full-frame portrait; the frame then tiles into
                split-screens of faces subdividing recursively 1->4->16->64->256 (cells pop in with a cascade).
C3 47.00-47.50  push into one tile's screen; it degrades (scanlines, RGB tear) into full-frame grey TV static,
                bit-identical to s03's first frame (scenes.s03_static.static_frame).
"""
import math

import cairo
import cv2
import numpy as np

from vx import (W, H, col, mix, shade, set_color, circle, ellipse, poly, rrect, lin_grad, rad_grad, blur, clamp,
                lerp, seg, smoothstep, smootherstep, ease_in_out, ease_out, ease_in, ease_out_back, hash01, noise1,
                fbm1, Canvas, gradient_v, surface_from_array)
from vx.chars import Character
from scenes import s02_monuments as MN
from scenes.s02_cards import _heart

T_C = 40.6
T_GRID = 42.45
T_PUSH = 47.0
T_STATIC = 47.35
SPLITS = [43.49, 43.84, 44.58, 45.17]          # overwritten from N07 word timings in setup
HOR = 470.0
FOC = 1500.0
CAM_H = 1.35
GLOWS = ["#00E5FF", "#FF2E88", "#39FF88", "#A66BFF", "#3D7BF2", "#FF5CE1", "#2FC6D6", "#7CFFB2", "#8C6BFF",
         "#FF4D6D", "#4DFFF3", "#C77DFF", "#00B3FF", "#FF7AA8", "#5CFF5C", "#6E8BFF"]
# the waist-deep people of the wide shot: (X metres, depth metres, glow index, seed)
PEOPLE = [(-4.0, 62.0, 3, 11), (9.0, 70.0, 6, 12), (2.2, 40.0, 9, 13), (-11.5, 33.0, 12, 14), (8.5, 24.0, 2, 15),
          (-6.2, 17.0, 5, 16), (4.4, 12.5, 7, 17), (-2.9, 7.8, 1, 18), (1.45, 4.6, 0, 19)]
HERO = 8          # index of the person we push into (nearest)
TARGET = (8, 7)   # the 256-grid tile we push into at the end


# =============================================================================================== setup
def setup(S, st):
    global SPLITS
    tl = S.tl
    SPLITS = [tl.word("N07", "every")[0], tl.word("N07", "face")[0], tl.word("N07", "until")[0],
              tl.word("N07", "two")[0]]
    st["feed_chars"] = {}
    st["tile_cache"] = {}
    rng = np.random.default_rng(9)
    st["rain"] = dict(x=rng.uniform(-100, W + 100, 70), y0=rng.uniform(0, H, 70), sp=rng.uniform(40, 110, 70),
                      sz=rng.uniform(6, 20, 70), idx=rng.integers(0, 10 ** 6, 70), ph=rng.uniform(0, 6.3, 70))
    ev = [dict(t=T_C, kind="cut", strength=0.5, pan=0.0, desc="dusk: isolated people waist-deep in the slop, phones glowing"),
          dict(t=41.3, kind="swell", strength=0.4, pan=-0.3, desc="a swell of slop rolls past the waist-deep people"),
          dict(t=42.2, kind="whoosh", strength=0.7, pan=0.2, desc="whip push into the nearest face"),
          dict(t=T_GRID, kind="hit", strength=0.5, pan=0.0, desc="full-frame face lit by a cyan feed")]
    names = ["1 -> 4", "4 -> 16", "16 -> 64", "64 -> 256"]
    for k, ts in enumerate(SPLITS):
        ev.append(dict(t=ts, kind="split", strength=0.55 + 0.1 * k, pan=0.0,
                       desc=f"split-screen subdivides {names[k]} (cells pop in over ~0.35 s cascade)"))
    ev.append(dict(t=T_PUSH, kind="whoosh", strength=0.8, pan=0.0, desc="push into one tile's screen"))
    ev.append(dict(t=47.2, kind="glitch", strength=0.6, pan=0.0, desc="the screen tears into noise"))
    ev.append(dict(t=T_STATIC, kind="static", strength=0.9, pan=0.0, desc="full-frame grey TV static (hand-off to s03)"))
    st["feed_events"] = ev


def _char(st, seed):
    d = st["feed_chars"]
    if seed not in d:
        d[seed] = Character("citizen", seed=seed)
    return d[seed]


# =============================================================================================== dispatch
def render(fc, st):
    T = fc.T
    if T < T_GRID:
        return render_wide(fc, st, T)
    if T < T_STATIC:
        return render_grid(fc, st, T)
    return static(fc, T)


def post(fc, st):
    T = fc.T
    if T >= T_STATIC - 0.08:
        return dict(bloom=0.42, bloom_thresh=0.74, bloom_radius=20, vignette=0.3, grain=0.03, ca=0.0)
    if T < T_GRID:
        return dict(bloom=0.55, vignette=0.34, ca=0.0)
    return dict(bloom=0.5, vignette=0.3, ca=0.0)


def static(fc, T):
    try:
        from scenes.s03_static import static_frame
        return static_frame(T, fc.w, fc.h, fc.s)
    except Exception:
        rng = np.random.default_rng(int(round(T * 24)))
        n = rng.random((fc.h, fc.w)).astype(np.float32)
        return np.repeat(n[..., None], 3, axis=2) * 0.8 + 0.1


# =============================================================================================== C1 wide
def _swells(T):
    """depths (m) of swell crests rolling toward camera"""
    period, v = 7.0, 5.5
    base = (T - T_C) * v
    return [70.0 - ((base + k * period) % 70.0) for k in range(10)]


def _bob(d, T):
    b = 0.0
    for zs in _swells(T):
        b += 0.18 * math.exp(-((d - zs) / 1.6) ** 2)
    return b + 0.03 * math.sin(T * 1.7 + d)


def render_wide(fc, st, T):
    s = fc.s
    camx = 0.55 * (T - T_C)
    # push-in toward the hero's face at the end
    hero = PEOPLE[HERO]
    pz = ease_in(clamp((T - 41.9) / (T_GRID - 41.9)), 2.2)
    img = _sky(fc, T, st)
    cv = Canvas(fc)
    gl = Canvas(fc)
    ctx, g = cv.ctx, gl.ctx
    # camera zoom about the hero's face
    hx = W / 2 + FOC * (hero[0] - camx) / hero[1]
    hp = 1.7 * FOC / hero[1]
    wl = HOR + FOC * CAM_H / hero[1]
    hp *= (0.94 + 0.12 * hash01(hero[3], 3))
    feet = wl + hp * 0.47 - _bob(hero[1], T) * FOC / hero[1] * 0.4
    hy = feet - hp * 0.84
    z = math.exp(math.log(5.5) * pz)
    for c in (ctx, g):
        c.translate(W / 2, H / 2)
        c.scale(z, z)
        c.translate(-lerp(W / 2, hx, min(1, pz * 1.6)), -lerp(H / 2, hy, min(1, pz * 1.6)))
    _ruins(ctx, g, T)
    sea = _sea_surface(fc, st, T, camx)
    sea_surf = surface_from_array(sea)
    # the sea behind everyone
    ctx.save()
    ctx.rectangle(-4000, HOR, 10000, 4000)
    ctx.clip()
    ctx.scale(1 / s, 1 / s)
    ctx.set_source_surface(sea_surf, 0, 0)
    ctx.paint()
    ctx.restore()
    order = sorted(range(len(PEOPLE)), key=lambda i: -PEOPLE[i][1])
    dprev = 1e9
    for i in order:
        X, d, gi, seed = PEOPLE[i]
        _floaters(ctx, T, camx, d, dprev)
        _person_wide(ctx, g, st, T, X, d, gi, seed, camx, sea_surf, s)
        dprev = d
    _floaters(ctx, T, camx, 0.0, dprev)
    _rain(ctx, st, T)
    img = cv.over(img)
    gg = gl.rgba()[..., :3]
    img = img + blur(gg, 4 * s) * 0.7 + blur(gg, 22 * s) * 0.6
    if pz > 0.85:
        fl = (pz - 0.85) / 0.15
        img = img * (1 - 0.5 * fl) + np.array(col(GLOWS[hero[2]]), np.float32) * 0.5 * fl
    return img.astype(np.float32)


def _sky(fc, T, st=None):
    img = gradient_v(fc, [(0, "#141624"), (0.22, "#22243A"), (HOR / H - 0.06, "#4A4560"), (HOR / H, "#6E6478"),
                          (1, "#6E6478")])
    if st is not None and "sky_tex" in st:
        tex = st["sky_tex"]
        h, w = fc.h, fc.w
        oy = int((tex.shape[0] - h) * 0.6)
        ox = int(((T - T_C) * 20 * fc.s + (tex.shape[1] - w) * 0.5) % max(1, tex.shape[1] - w))
        n = tex[oy:oy + h, ox:ox + w]
        ys = (np.arange(h, dtype=np.float32) + 0.5) / fc.s
        amp = (np.clip(1 - ys / HOR, 0, 1) * 0.35)[:, None, None]
        img = img * (1 + n[..., None] * amp)
    return img


def _ruins(ctx, g, T):
    """the drowned memeplexes on the horizon: a tilted tower, a halo half under, a marble crown"""
    M = MN.Look("#4A4560", 0.5, exposure=0.42)
    for (kind, x, sc, rot, sink) in (("nation", 330.0, 0.5, 0.3, 190.0), ("faith", 1230.0, 0.42, -0.1, 250.0),
                                     ("philo", 1700.0, 0.44, -0.42, 300.0)):
        for c in (ctx, g):
            c.save()
            c.translate(x, HOR + sink)
            c.rotate(rot)
            c.scale(sc, sc)
        if kind == "nation":
            MN.draw_nation(ctx, g, M, T, 0.0, crack=1.0, top=(260, 300, 1.4))
        elif kind == "faith":
            MN.draw_faith(ctx, g, M, T, 0.0, crack=1.0, top=(0, 0, 0), halo=(260, 700, 1.1))
        else:
            MN.draw_philosophy(ctx, g, M, T, 0.0, crack=1.0)
        for c in (ctx, g):
            c.restore()
    # a dead red aviation light still blinking on the tilted tower
    on = 0.5 + 0.5 * math.sin(T * 5)
    g.set_source_rgba(1, 0.2, 0.25, 0.4 * on)
    circle(g, 480, 40, 12)
    g.fill()


def _sea_surface(fc, st, T, camx):
    """numpy slop sea below the horizon: beige-mauve, swells rolling toward camera, texture"""
    h, w, s = fc.h, fc.w, fc.s
    img = np.zeros((h, w, 3), np.float32)
    j0 = int(HOR * s)
    ys = (np.arange(j0, h, dtype=np.float32) + 0.5) / s
    d = FOC * CAM_H / np.maximum(ys - HOR, 0.5)
    far = np.array(col("#5A5266"), np.float32)
    near = np.array(col("#4A3F4A"), np.float32)
    tn = np.clip(1.0 - d / 40.0, 0, 1)[:, None]
    base = far * (1 - tn) + near * tn
    # swells: lit crest + shadowed back
    sh = np.zeros_like(d)
    for zs in _swells(T):
        u = (d - zs) / (1.2 + zs * 0.04)
        sh += 0.16 * np.exp(-u * u) - 0.1 * np.exp(-((u - 1.6) ** 2))
    tex = st["sea_tex"]
    th, tw = tex.shape
    rows = np.clip(((ys - HOR) * 0.9 * s).astype(int), 0, th - 1)
    ox = int((camx * 40 + T * 12) * s) % max(1, tw - w)
    n = tex[rows][:, ox:ox + w]
    val = base[:, None, :] * (1 + sh[:, None, None] + 0.07 * n[..., None] * (0.3 + tn[:, None]))
    # sky reflection gleam near the horizon
    gleam = np.exp(-((ys - HOR) / 18.0) ** 2)[:, None, None] * 0.25
    val = val * (1 - gleam) + np.array(col("#857A8E"), np.float32) * gleam
    img[j0:] = val
    return img


def _floaters(ctx, T, camx, dlo, dhi):
    """content cards floating on the flood, flattened by perspective"""
    atlas = _atlas()
    for i in range(140):
        d = 4.0 + 90 * hash01(i, 41) ** 1.5
        if not (dlo <= d < dhi):
            continue
        X = (hash01(i, 42) - 0.5) * d * 2.8 + 0.3 * math.sin(T * 0.5 + i)
        x = W / 2 + FOC * (X - camx) / d
        y = HOR + FOC * (CAM_H - _bob(d, T) * 0.6) / d
        size = 0.55 * FOC / d
        if x < -100 or x > W + 100 or y > H + 80:
            continue
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(1, 0.35)
        atlas.draw(ctx, i * 17 + 5, 0, 0, size, math.sin(i) * 0.6, shade_k=0.55, alpha=0.9)
        ctx.restore()


_ATLAS = []


def _atlas():
    if not _ATLAS:
        from scenes.s02_cards import Atlas
        _ATLAS.append(Atlas(n_variants=1, seed=21))
    return _ATLAS[0]


def _rain(ctx, st, T):
    """a slow fall of cards still coming down after the crash"""
    atlas = _atlas()
    R = st["rain"]
    for i in range(len(R["x"])):
        y = (R["y0"][i] + (T - T_C) * R["sp"][i]) % (H + 80) - 40
        x = R["x"][i] + 20 * math.sin(T * 1.3 + R["ph"][i])
        atlas.draw(ctx, int(R["idx"][i]), x, y, R["sz"][i], T * 1.5 + R["ph"][i], sx=math.cos(T * 2 + R["ph"][i]),
                   shade_k=0.75, alpha=0.55)


def _uplit(ctx, draw_fn, glow, phone_xy, top_y, dark=0.6, T=0.0, seed=0, k=1.0):
    """draw a figure (draw_fn), sink it into darkness from the top, then ADD the phone's coloured light from below,
    masked to the figure so the light only falls on it"""
    px, py = phone_xy
    R = max(4.0, (py - top_y) * 1.15)
    ctx.push_group()
    draw_fn(ctx)
    ctx.set_operator(cairo.OPERATOR_ATOP)
    gr = cairo.LinearGradient(0, top_y, 0, py)
    gr.add_color_stop_rgba(0, 0.02, 0.02, 0.06, dark)
    gr.add_color_stop_rgba(1, 0.02, 0.02, 0.06, dark * 0.25)
    ctx.set_source(gr)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)
    fig = ctx.pop_group()
    ctx.set_source(fig)
    ctx.paint()
    r, gg, b = col(glow)
    flick = 0.85 + 0.15 * math.sin(T * 11 + seed) * math.sin(T * 3.7 + seed * 2)
    rg = cairo.RadialGradient(px, py + R * 0.05, 0, px, py - R * 0.1, R)
    rg.add_color_stop_rgba(0, r, gg, b, 0.95 * flick * k)
    rg.add_color_stop_rgba(0.3, r, gg, b, 0.5 * flick * k)
    rg.add_color_stop_rgba(0.7, r, gg, b, 0.14 * flick * k)
    rg.add_color_stop_rgba(1, r, gg, b, 0.0)
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.set_source(rg)
    ctx.mask(fig)
    # scrolling feed light sweeping up the face
    for j in range(2):
        yy = py - ((T * 0.55 * R + seed * 97 + j * R * 0.7) % (R * 1.1))
        lg = cairo.LinearGradient(0, yy - R * 0.06, 0, yy + R * 0.06)
        lg.add_color_stop_rgba(0, r, gg, b, 0.0)
        lg.add_color_stop_rgba(0.5, r, gg, b, 0.16 * k)
        lg.add_color_stop_rgba(1, r, gg, b, 0.0)
        ctx.set_source(lg)
        ctx.mask(fig)
    ctx.set_operator(cairo.OPERATOR_OVER)


def _person_wide(ctx, g, st, T, X, d, gi, seed, camx, sea_surf, s):
    hp = 1.7 * FOC / d * (0.94 + 0.12 * hash01(seed, 3))
    bob = _bob(d, T) * FOC / d
    x = W / 2 + FOC * (X - camx) / d
    wl = HOR + FOC * CAM_H / d                     # waterline on screen
    feet = wl + hp * 0.47 - bob * 0.4
    if x < -hp or x > W + hp:
        return
    ch = _char(st, seed)
    glow = GLOWS[gi]
    an = ch.anchors(x, feet, hp, "stand", T, facing=0)
    hx, hy = an["head"]
    hand = (x + 0.03 * hp, hy + 0.3 * hp)
    turn = (hash01(seed, 5) - 0.5) * 0.5

    def draw(c):
        ch.draw(c, x, feet, hp, "stand", T + seed, facing=0, turn=turn, prop="phone", phone_color=col(glow),
                hand_r=hand, shape_r="grip", look=(0, 0.95), nod=-0.3, lean=0.12, expr="deadpan")
    _uplit(ctx, draw, glow, (hand[0], hand[1] - 0.02 * hp), an["top"][1], dark=0.62, T=T, seed=seed, k=0.8)
    # the slop in front of this person (painter's algorithm): clip below a rippling waterline
    ctx.save()
    pts = []
    for k in range(-60, 61, 4):
        xx = x + k * hp * 0.02
        rip = math.sin(k * 0.35 - T * 4) * hp * 0.008 * math.exp(-(k / 30) ** 2)
        pts.append((xx, wl - bob * 0.4 + rip))
    pts = [(-4000, pts[0][1])] + pts + [(8000, pts[-1][1]), (8000, 5000), (-4000, 5000)]
    poly(ctx, pts)
    ctx.clip()
    ctx.scale(1 / s, 1 / s)
    ctx.set_source_surface(sea_surf, 0, 0)
    ctx.paint()
    ctx.restore()
    # ring of disturbed slop + the pool of screen light on the surface around them
    r, gg, b = col(glow)
    ctx.set_source_rgba(0.95, 0.9, 0.9, 0.35)
    ctx.set_line_width(max(1.0, hp * 0.006))
    ellipse(ctx, x, wl - bob * 0.4, hp * 0.2, hp * 0.03)
    ctx.stroke()
    rg = cairo.RadialGradient(x, wl, 0, x, wl, hp * 0.5)
    rg.add_color_stop_rgba(0, r, gg, b, 0.7)
    rg.add_color_stop_rgba(0.4, r, gg, b, 0.25)
    rg.add_color_stop_rgba(1, r, gg, b, 0.0)
    g.save()
    g.translate(0, wl)
    g.scale(1, 0.22)
    g.translate(0, -wl)
    g.set_source(rg)
    circle(g, x, wl, hp * 0.5)
    g.fill()
    g.restore()
    g.set_source_rgba(r, gg, b, 0.8)
    circle(g, hand[0], hand[1] - 0.02 * hp, hp * 0.03)
    g.fill()


# =============================================================================================== C2 grid
def _level_state(T):
    """(level, progress-of-that-level's pop-in 0..1 base time)"""
    lvl = 0
    for k, ts in enumerate(SPLITS):
        if T >= ts:
            lvl = k + 1
    return lvl


def _tile_person(L, i, j):
    if L == 0:
        X, d, gi, seed = PEOPLE[HERO]
        return seed, GLOWS[gi]
    hsh = int(hash01(L * 100003 + i * 1009 + j * 37, 77) * 10 ** 6)
    seed = 200 + hsh % 5000
    # neighbouring tiles never share a glow
    gi = (i * 7 + j * 11 + L * 3) % len(GLOWS)
    return seed, GLOWS[gi]


def _child_delay(L, i, j):
    n = 2 ** L
    cx, cy = (i + 0.5) / n - 0.5, (j + 0.5) / n - 0.5
    return 0.26 * min(1.0, math.hypot(cx * 1.2, cy) / 0.65) + 0.04 * hash01(L * 999 + i * 31 + j, 3)


def _draw_tile(ctx, st, x, y, w, h, seed, glow, T, s, live=False):
    """one feed window: a dark room in the feed's colour, a face uplit by its phone"""
    r, gg, b = col(glow)
    ctx.save()
    ctx.rectangle(x, y, w, h)
    ctx.clip()
    bg = cairo.LinearGradient(0, y, 0, y + h)
    bg.add_color_stop_rgb(0, 0.02 + r * 0.03, 0.02 + gg * 0.03, 0.05 + b * 0.04)
    bg.add_color_stop_rgb(1, 0.03 + r * 0.16, 0.03 + gg * 0.16, 0.06 + b * 0.18)
    ctx.set_source(bg)
    ctx.paint()
    # their own world: bokeh of their feed's light and shapes drifting behind them
    for k in range(7 if w > 150 else 3):
        u = (hash01(seed * 11 + k, 61) + (T - T_GRID) * 0.05 * (1 + k % 3)) % 1.0
        bx = x + w * hash01(seed * 11 + k, 62)
        by = y + h * (1.15 - 1.3 * u)
        br = h * (0.06 + 0.12 * hash01(seed * 11 + k, 63))
        rgb_ = cairo.RadialGradient(bx, by, 0, bx, by, br)
        rgb_.add_color_stop_rgba(0, r, gg, b, 0.22)
        rgb_.add_color_stop_rgba(0.7, r, gg, b, 0.12)
        rgb_.add_color_stop_rgba(1, r, gg, b, 0.0)
        ctx.set_source(rgb_)
        circle(ctx, bx, by, br)
        ctx.fill()
    if w > 200:
        atlas = _atlas()
        for k in range(4):
            u = (hash01(seed * 7 + k, 51) + (T - T_GRID) * 0.08) % 1.0
            px = x + w * (0.12 + 0.76 * hash01(seed * 7 + k, 52))
            py = y + h * (1.1 - 1.2 * u)
            atlas.draw(ctx, seed * 13 + k * 7, px, py, h * 0.22, 0.3 * math.sin(k + T), shade_k=0.9, alpha=0.35)
    hch = h * 2.55
    cx = x + w * (0.5 + 0.04 * (hash01(seed, 9) - 0.5))
    feet = y + h * 0.04 + hch
    ch = _char(st, seed)
    an = ch.anchors(cx, feet, hch, "stand", 0.0, facing=0, look=(0, 0.95), nod=-0.25)
    hx, hy = an["head"]
    hand = (cx + 0.03 * hch, hy + 0.3 * hch)
    blink = ((T * 0.9 + hash01(seed, 4) * 7) % 4.1) < 0.12 if live else False

    def draw(c):
        ch.draw(c, cx, feet, hch, "stand", 0.0, facing=0, prop="phone", phone_color=col(glow), hand_r=hand,
                shape_r="grip", look=(0, 0.95), nod=-0.25, expr="deadpan", blink=blink)
    dw, dh = ctx.user_to_device_distance(w, h)
    dw, dh = abs(dw), abs(dh)
    q = 1.25 ** math.ceil(math.log(max(dw, 4.0)) / math.log(1.25))       # quantised device width
    key = (seed, glow, int(q), int(q * h / w))
    if live or dw > 700:
        _uplit(ctx, draw, glow, hand, an["top"][1], dark=0.62, T=T, seed=seed, k=0.72)
        # the feed reflected in both eyes
        ex, ey = an["eyes"]
        hr = an.get("head_r", hch * 0.1)
        hr = hr if isinstance(hr, (int, float)) else hch * 0.1
        for sx in (-1, 1):
            ctx.set_source_rgba(min(1, r + 0.5), min(1, gg + 0.5), min(1, b + 0.5), 0.85)
            rrect(ctx, ex + sx * hr * 0.36 - hr * 0.035, ey + hr * 0.05, hr * 0.07, hr * 0.1, hr * 0.015)
            ctx.fill()
    else:
        cache = st["tile_cache"]
        if key not in cache:
            pw, ph = key[2], key[3]
            surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, pw, ph)
            c2 = cairo.Context(surf)
            c2.scale(pw / w, ph / h)
            c2.translate(-x, -y)
            draw(c2)
            cache[key] = surf
        surf = cache[key]
        _uplit(ctx, lambda c: _paint_surf(c, surf, x, y, w, h), glow, hand, an["top"][1], dark=0.62, T=T, seed=seed)
    # vignette inside the window
    vg = cairo.RadialGradient(x + w / 2, y + h / 2, h * 0.3, x + w / 2, y + h / 2, w * 0.62)
    vg.add_color_stop_rgba(0, 0, 0, 0, 0)
    vg.add_color_stop_rgba(1, 0, 0, 0, 0.45)
    ctx.set_source(vg)
    ctx.paint()
    ctx.restore()


def _paint_surf(c, surf, x, y, w, h):
    c.save()
    c.translate(x, y)
    c.scale(w / surf.get_width(), h / surf.get_height())
    c.set_source_surface(surf, 0, 0)
    c.paint()
    c.restore()


def _gutter(L):
    return (0.0, 7.0, 4.5, 3.0, 2.0)[L]


def render_grid(fc, st, T):
    s = fc.s
    cv = Canvas(fc, bg="#050508")
    ctx = cv.ctx
    L = _level_state(T)
    # final push into one tile
    pz = ease_in(clamp((T - T_PUSH) / (T_STATIC - T_PUSH)), 2.2)
    n = 2 ** L
    tw, th = W / n, H / n
    if pz > 0:
        ti, tj = TARGET
        tx, ty = (ti + 0.5) * tw, (tj + 0.5) * th
        z = math.exp(math.log(n * 1.02) * min(1.0, pz * 1.25))
        ctx.translate(W / 2, H / 2)
        ctx.scale(z, z)
        ctx.translate(-lerp(W / 2, tx, min(1.0, pz * 2.0)), -lerp(H / 2, ty, min(1.0, pz * 2.0)))
    # visible world rect (cull)
    ix0, iy0 = ctx.device_to_user(0, 0)
    ix1, iy1 = ctx.device_to_user(fc.w, fc.h)
    # gentle drift apart at "same world"
    spread = smoothstep(SPLITS[-1] + 1.0, SPLITS[-1] + 2.2, T) * (1 - pz)
    if L == 0:
        seed, glow = _tile_person(0, 0, 0)
        _draw_tile(ctx, st, 0, 0, W, H, seed, glow, T, s, live=True)
    else:
        ts = SPLITS[L - 1]
        gpar, gch = _gutter(L - 1), _gutter(L)
        np_ = 2 ** (L - 1)
        pw, ph = W / np_, H / np_
        all_done = T > ts + 0.26 + 0.06 + 0.24
        if not all_done:
            for i in range(np_):
                for j in range(np_):
                    x, y = i * pw, j * ph
                    if x > ix1 or x + pw < ix0 or y > iy1 or y + ph < iy0:
                        continue
                    seed, glow = _tile_person(L - 1, i, j)
                    _draw_tile(ctx, st, x + gpar / 2, y + gpar / 2, pw - gpar, ph - gpar, seed, glow, T, s,
                               live=(L - 1 == 0))
        for i in range(n):
            for j in range(n):
                p = clamp((T - ts - _child_delay(L, i, j)) / 0.24)
                if p <= 0:
                    continue
                sc = ease_out_back(p, 2.2) if p < 1 else 1.0
                cxt, cyt = (i + 0.5) * tw, (j + 0.5) * th
                cxt += (cxt - W / 2) * 0.012 * spread
                cyt += (cyt - H / 2) * 0.012 * spread
                nw, nh = tw - gch, th - gch
                ww, hh = nw * sc, nh * sc
                if sc < 0.02:
                    continue
                if cxt + ww / 2 < ix0 or cxt - ww / 2 > ix1 or cyt + hh / 2 < iy0 or cyt - hh / 2 > iy1:
                    continue
                seed, glow = _tile_person(L, i, j)
                live = (pz > 0 and (i, j) == TARGET and L == 4)
                ctx.save()
                ctx.translate(cxt, cyt)
                ctx.scale(sc, sc)
                if p < 1:
                    # a thin black gutter frame pops first
                    ctx.set_source_rgb(0.02, 0.02, 0.03)
                    ctx.rectangle(-nw / 2 - gch, -nh / 2 - gch, nw + 2 * gch, nh + 2 * gch)
                    ctx.fill()
                _draw_tile(ctx, st, -nw / 2, -nh / 2, nw, nh, seed, glow, T, s, live=live)
                if p < 1:
                    r, gg, b = col(glow)
                    ctx.set_source_rgba(r, gg, b, 0.5 * (1 - p))
                    ctx.rectangle(-nw / 2, -nh / 2, nw, nh)
                    ctx.fill()
                ctx.restore()
    img = cv.rgb()
    # the target screen degrades into static
    if pz > 0:
        tear = smoothstep(47.16, T_STATIC, T)
        if tear > 0:
            img = _tear(img, fc, T, tear)
    return img


def _tear(img, fc, T, a):
    """scanlines + RGB tearing + noise creeping in, then static takes over"""
    h, w = img.shape[:2]
    rng = np.random.default_rng(int(T * 1000))
    out = img.copy()
    sh = int(round(14 * a * fc.s))
    if sh:
        out[..., 0] = np.roll(img[..., 0], sh, axis=1)
    lines = (np.arange(h) % max(2, int(3 * fc.s)) == 0)[:, None, None]
    out = out * (1 - 0.35 * a * lines)
    for _ in range(int(3 + 12 * a)):
        y0 = int(rng.uniform(0, h))
        y1 = min(h, y0 + int(rng.uniform(3, 30) * fc.s))
        out[y0:y1] = np.roll(out[y0:y1], int(rng.normal(0, 120 * a) * fc.s), axis=1)
    st_img = static(fc, T)
    k = a ** 1.5
    return out * (1 - k) + st_img * k

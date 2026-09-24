"""t04 — kinetic typography: the crowd's incomprehension becomes a wall of type that drowns even the Devil.

All in SCREEN design space (1920x1080), lettered with vx.comic's faces. Nothing here may touch flag cloth:
callers pass the flag-layer alpha, and every layer drawn here is masked out of it (plus a margin).

- scramble(src, dst, p, seed)        split-flap style letter scramble from one word into another
- WordPile                           pymunk rigid-body sim of FLAGS word-blocks raining/spilling, piling up and
                                     flowing AROUND the (kinematic) flag cloth; cached per scene
- wall_slots(...)                    typeset wall of FLAGS rows, reflowed live around the flag's silhouette
- draw_words(ctx, words)             batch-draw words (x, y, ang, size, alpha, text) black caps, white keyline
- stamp(ctx, ...)                    rubber-stamp FLAGS over a word (the Devil's balloon)
"""
import hashlib
import json
import math

import cairo
import cv2
import numpy as np

from vx import W, H, hash01, clamp, ease_out_back, set_color
from vx import comic, ink

WORD = "FLAGS"


def F(family=None):
    return comic.face(family or comic.FONT_LETTER)


def text_w(s, size, family=None):
    return F(family).width(s, size)


def metrics(size=100.0):
    f = F()
    return dict(adv=f.width(WORD, 1.0), cap=f.cap)


_LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def scramble(src, dst, p, seed=0):
    """p 0 -> src, 1 -> dst; between: a split-flap cascade (letters settle left to right)"""
    if p <= 0:
        return src
    if p >= 1:
        return dst
    n = max(len(src), len(dst))
    L = int(round(len(src) + (len(dst) - len(src)) * min(1.0, p * 1.6)))
    out = []
    for i in range(L):
        settle = (i + 1) / (L + 1) * 0.8 + 0.2
        if p >= settle:
            out.append(dst[i] if i < len(dst) else "")
        else:
            k = int(p * 40 + i * 7 + seed * 13)
            out.append(_LET[int(hash01(k, seed + i) * 26)])
    return "".join(out)


# ============================================================ drawing
def draw_words(ctx, words, *, keyline=0.13, inkc=None, paper=None, family=None):
    """words: iterable of (x, y, ang, size, alpha, text) — x, y = CENTRE of the word box. Black heavy caps,
    white keyline (so they read over the engraving)."""
    f = F(family)
    inkc = inkc or ink.INK
    paper = paper or ink.PAPER
    ctx.save()
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    for (x, y, a, size, al, s) in words:
        if al <= 0.003 or size < 2:
            continue
        ww = f.width(s, size)
        ctx.save()
        ctx.translate(x, y)
        if a:
            ctx.rotate(a)
        ctx.new_path()
        f.path(ctx, s, -ww / 2, f.cap * size / 2, size)
        if keyline > 0:
            set_color(ctx, paper, al)
            ctx.set_line_width(size * keyline)
            ctx.stroke_preserve()
        set_color(ctx, inkc, al)
        ctx.fill()
        ctx.restore()
    ctx.restore()


def stamp(ctx, x, y, w, h, t, *, seed=0, size=None, text=WORD, rot=None, alpha=1.0):
    """slam a rubber-stamp FLAGS over the box (x, y, w, h); t = seconds since the slam"""
    if t < 0:
        return None
    sc = 1.0 + 0.9 * max(0.0, 1.0 - t / 0.09) ** 2          # comes down from above, lands at 0.09 s
    wob = math.exp(-t * 18) * math.sin(t * 90) * 0.04
    a = (rot if rot is not None else (hash01(seed, 5) - 0.5) * 0.35) + wob
    size = size or h * 1.05
    tw = text_w(text, size)
    bw, bh = tw + size * 0.5, size * 1.15
    cx, cy = x + w / 2, y + h / 2
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(a)
    ctx.scale(sc, sc)
    al = alpha * min(1.0, t / 0.05 + 0.2)
    # stamp frame (double rule), slightly ragged ink
    paper, inkc = ink.PAPER, ink.INK
    ctx.rectangle(-bw / 2, -bh / 2, bw, bh)
    set_color(ctx, paper, al * 0.94)
    ctx.fill_preserve()
    set_color(ctx, inkc, al)
    ctx.set_line_width(size * 0.09)
    ctx.stroke()
    ctx.rectangle(-bw / 2 + size * 0.1, -bh / 2 + size * 0.1, bw - size * 0.2, bh - size * 0.2)
    ctx.set_line_width(size * 0.03)
    ctx.stroke()
    ctx.new_path()
    F().path(ctx, text, -tw / 2, F().cap * size / 2, size)
    ctx.fill()
    # dry-brush speckle holes (paper showing through the ink)
    rng = np.random.default_rng(seed * 31 + 7)
    set_color(ctx, paper, al)
    for _ in range(26):
        ctx.arc(rng.uniform(-bw / 2, bw / 2), rng.uniform(-bh / 2, bh / 2), rng.uniform(0.6, 2.2) * size / 40, 0,
                6.3)
        ctx.fill()
    ctx.restore()
    return (cx - bw / 2 * sc, cy - bh / 2 * sc, cx + bw / 2 * sc, cy + bh / 2 * sc)


# ============================================================ masks
def flag_block_mask(flag_alpha, s, margin=18.0):
    """(h, w) flag-layer alpha (device px) -> dilated block mask (device px) with `margin` design px"""
    r = max(1, int(round(margin * s)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = (flag_alpha > 0.02).astype(np.uint8)
    return cv2.dilate(m, k).astype(np.float32)


def mask_out(rgba, block):
    """premultiplied rgba * (1 - block): nothing lettered may lie on the flag"""
    if block is None:
        return rgba
    return rgba * (1.0 - block[..., None])


# ============================================================ wall of FLAGS
def wall_rows(rect, size, lead=1.08):
    x0, y0, w, h = rect
    lh = size * lead
    n = int(math.ceil(h / lh)) + 1
    return [y0 + lh * (i + 0.5) for i in range(n)], lh


def free_intervals(block_rows, x0, x1):
    """block_rows: bool array over columns (design px) -> list of free (a, b) intervals within [x0, x1]"""
    out = []
    n = len(block_rows)
    j = max(0, int(x0))
    j1 = min(n, int(x1))
    while j < j1:
        while j < j1 and block_rows[j]:
            j += 1
        a = j
        while j < j1 and not block_rows[j]:
            j += 1
        if j > a:
            out.append((a, j))
    return out


def wall_slots(rect, size, block=None, s=1.0, *, lead=1.06, gap=0.30, pad=4.0, brick=True):
    """typeset a wall of FLAGS rows in rect, REFLOWED around `block` (device-px mask, e.g. the flag), like a
    magazine text-wrap: the words run flush up to the obstacle on both sides, justified between obstacles.
    Returns ([(x, y, row, col)] word centres, word width)."""
    x0, y0, w, h = rect
    ys, lh = wall_rows(rect, size, lead)
    ww = text_w(WORD, size)
    adv = ww + size * gap
    cols = None
    if block is not None:
        bh, bw = block.shape
        cols = cv2.resize(block, (int(W), int(round(bh / s))), interpolation=cv2.INTER_NEAREST) if s != 1.0 else block
    L, R = x0 - adv, x0 + w + adv
    out = []
    for r, yc in enumerate(ys):
        if yc - lh / 2 > y0 + h:
            break
        base = (adv * 0.5 if (brick and r % 2) else 0.0)
        if cols is not None:
            ya = int(max(0, yc - lh * 0.5))
            yb = int(min(cols.shape[0], yc + lh * 0.5))
            band = cols[ya:yb].max(axis=0) > 0 if yb > ya else np.zeros(cols.shape[1], bool)
            band = np.concatenate([np.zeros(int(adv) + 2, bool), band, np.zeros(int(adv) + 2, bool)])
            ivs = [(a - int(adv) - 2, b - int(adv) - 2) for a, b in free_intervals(band, 0, len(band))]
            ivs = [(max(a, L), min(b, R)) for a, b in ivs if min(b, R) > max(a, L)]
        else:
            ivs = [(L, R)]
        for (a, b) in ivs:
            open_l, open_r = a <= L + 1, b >= R - 1
            if open_l and open_r:                        # a full row: the plain grid
                xx = L + base + ww / 2
                c = 0
                while xx - ww / 2 < R:
                    out.append((xx, yc, r, c))
                    xx += adv
                    c += 1
                continue
            span = b - a - 2 * pad
            k = int((span + size * gap) // adv)
            if k <= 0:
                continue
            if open_l:                                   # flush right against the obstacle
                xs = [b - pad - ww / 2 - c * adv for c in range(k)]
            elif open_r:                                 # flush left after the obstacle
                xs = [a + pad + ww / 2 + c * adv for c in range(k)]
            elif k > 1:                                  # between obstacles: justified
                g = (span - k * ww) / (k - 1)
                xs = [a + pad + ww / 2 + c * (ww + g) for c in range(k)]
            else:
                xs = [(a + b) / 2]
            for c, xx in enumerate(xs):
                out.append((xx, yc, r, c))
    return out, ww


# ============================================================ the pile (pymunk)
def _hash(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def simulate_pile(spawns, n, fps, obstacles_fn, *, cache=None, floor=H + 4, walls=(-4, W + 4), sub=8,
                  key=None, gravity=2600.0):
    """Rigid-body FLAGS word blocks.
    spawns: list of dict(t=local frame float, x, y, vx, vy, a, av, size) (screen design px)
    obstacles_fn(i) -> list of (x, y, r) kinematic circles at frame i (e.g. the flag cloth + margin)
    Returns dict(pos=(n, m, 3) x,y,angle, born=(m,) frame, size=(m,))"""
    import pymunk
    if cache is not None and key is not None:
        p = cache / f"pile_{_hash(key)}.npz"
        if p.exists():
            z = np.load(p)
            return {k: z[k] for k in z.files}
    space = pymunk.Space()
    space.gravity = (0, gravity)
    space.iterations = 20
    space.damping = 0.985
    st = space.static_body
    segs = [pymunk.Segment(st, (walls[0] - 400, floor), (walls[1] + 400, floor), 6),
            pymunk.Segment(st, (walls[0], floor), (walls[0], -3000), 6),
            pymunk.Segment(st, (walls[1], floor), (walls[1], -3000), 6)]
    for sg in segs:
        sg.friction = 0.9
        sg.elasticity = 0.05
    space.add(*segs)
    m = len(spawns)
    order = sorted(range(m), key=lambda k: spawns[k]["t"])
    ratio = metrics()["adv"]
    bodies = [None] * m
    pos = np.full((n, m, 3), np.nan, np.float32)
    born = np.array([spawns[k]["t"] for k in range(m)], np.float32)
    sizes = np.array([spawns[k]["size"] for k in range(m)], np.float32)
    kin = []
    nxt = 0
    dt = 1.0 / fps / sub
    prev_obs = None
    for i in range(n):
        obs = obstacles_fn(i)
        while len(kin) < len(obs):
            b = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            c = pymunk.Circle(b, 10)
            c.friction = 0.6
            c.elasticity = 0.1
            space.add(b, c)
            kin.append((b, c))
        for j, (b, c) in enumerate(kin):
            if j < len(obs):
                x, y, r = obs[j]
                if prev_obs is not None and j < len(prev_obs):
                    px, py, _ = prev_obs[j]
                    b.velocity = ((x - px) * fps, (y - py) * fps)
                else:
                    b.velocity = (0, 0)
                b.position = (x, y)
                c.unsafe_set_radius(r)
            else:
                b.position = (-9999, -9999)
        space.reindex_shapes_for_body(kin[0][0]) if kin else None
        for b, c in kin:
            space.reindex_shapes_for_body(b)
        prev_obs = obs
        while nxt < m and spawns[order[nxt]]["t"] <= i:
            k = order[nxt]
            sp = spawns[k]
            sz = sp["size"]
            bw, bh = sz * ratio * 0.98, sz * 0.86
            mass = bw * bh * 0.001
            body = pymunk.Body(mass, pymunk.moment_for_box(mass, (bw, bh)))
            body.position = (sp["x"], sp["y"])
            body.velocity = (sp.get("vx", 0.0), sp.get("vy", 0.0))
            body.angle = sp.get("a", 0.0)
            body.angular_velocity = sp.get("av", 0.0)
            shp = pymunk.Poly.create_box(body, (bw, bh), radius=1.0)
            shp.friction = 0.75
            shp.elasticity = 0.08
            space.add(body, shp)
            bodies[k] = body
            nxt += 1
        for _ in range(sub):
            space.step(dt)
        for k, b in enumerate(bodies):
            if b is not None:
                pos[i, k] = (b.position.x, b.position.y, b.angle)
    out = dict(pos=pos, born=born, size=sizes)
    if cache is not None and key is not None:
        np.savez_compressed(cache / f"pile_{_hash(key)}.npz", **out)
    return out

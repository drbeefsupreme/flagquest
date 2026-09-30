"""vx.mosaic_sheet - the reference sheet for the film's mosaic look (owner: Tessellator).

  VX_FILM=opus python -m vx.mosaic_sheet        -> films/opus/out/mosaic_sheet.png (+ panels in out/mosaic_sheet/)

Panels: gold ground at two candle positions, a saint's face in opus vermiculatum, tituli, a dome from below,
a sinopia patch where the mosaic has fallen, tiles falling to the floor. The cartoons here double as examples of
how to paint for vx.mosaic (flat smalti colours, dark contour strokes ~1.1 tile wide, masks for gold/fine).
"""
import math
import time

import cairo
import cv2
import numpy as np

from .config import W, H, OUT
from .canvas import Canvas, set_color, col
from . import mosaic as mz

S = mz.SMALTI


class _FC:
    def __init__(self, s=1.0, T=0.0, w=None, h=None):
        self.s = s
        self.w = w or int(round(W * s))
        self.h = h or int(round(H * s))
        self.T = self.t = T
        self.f = self.F = 0


def _c(ctx, name, a=1.0):
    set_color(ctx, S[name] if name in S else name, a)


def _ell(ctx, cx, cy, rx, ry, rot=0.0):
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(rot)
    ctx.scale(rx, ry)
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()


def saint(ctx, cx, cy, k=1.0, layer="color", tile=12.0):
    """A haloed sage, bust length, frontal (Ravenna / Hosios Loukas manner). Head centre (cx, cy), k = scale
    (k=1: face ~330 units tall). layer: 'color' | 'gold' | 'fine' (white masks)."""
    lw = 1.15 * tile                      # contour strokes = one course of dark tesserae
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(k, k)
    lw /= k
    if layer == "gold":
        ctx.set_source_rgb(1, 1, 1)
        ctx.arc(0, -20, 300, 0, 2 * math.pi)
        ctx.fill()
        # gold clavus + chrysography lines on the pallium
        ctx.rectangle(-236, 330, 30, 300)
        ctx.fill()
        ctx.restore()
        return
    if layer == "fine":
        ctx.set_source_rgb(1, 1, 1)
        _ell(ctx, 0, 20, 170, 230)             # face + beard + hair edge
        ctx.fill()
        ctx.rectangle(-60, 200, 120, 150)      # neck
        ctx.fill()
        ctx.restore()
        return
    # ---- halo: gold disc, dark red outline, inner pearl ring
    _c(ctx, "gold")
    ctx.arc(0, -20, 300, 0, 2 * math.pi)
    ctx.fill()
    ctx.set_line_width(lw)
    _c(ctx, "porphyry")
    ctx.arc(0, -20, 300, 0, 2 * math.pi)
    ctx.stroke()
    # ---- shoulders: purple tunic, white pallium with folds
    _c(ctx, "tyrian")
    ctx.move_to(-420, 640)
    ctx.curve_to(-400, 420, -300, 330, -120, 290)
    ctx.line_to(120, 290)
    ctx.curve_to(300, 330, 400, 420, 420, 640)
    ctx.close_path()
    ctx.fill()
    _c(ctx, "marble")
    ctx.move_to(-420, 640)
    ctx.curve_to(-410, 470, -330, 380, -190, 330)
    ctx.curve_to(-120, 420, -60, 520, 10, 640)
    ctx.close_path()
    ctx.fill()
    _c(ctx, "marble_d")
    ctx.set_line_width(lw * 0.85)
    for a, b, c_, d in ((-330, 430, -250, 640), (-260, 380, -150, 640), (-380, 520, -330, 640)):
        ctx.move_to(a, b)
        ctx.curve_to(a + 20, b + 80, c_ - 10, d - 80, c_, d)
        ctx.stroke()
    _c(ctx, "gold")
    ctx.rectangle(-236, 330, 30, 300)
    ctx.fill()
    # tunic folds
    _c(ctx, "wine")
    for a in (90, 190, 290):
        ctx.move_to(a, 330)
        ctx.curve_to(a + 20, 420, a + 40, 540, a + 50, 640)
        ctx.stroke()
    _c(ctx, "ink")
    ctx.set_line_width(lw)
    ctx.move_to(-420, 640)
    ctx.curve_to(-400, 420, -300, 330, -120, 290)
    ctx.move_to(120, 290)
    ctx.curve_to(300, 330, 400, 420, 420, 640)
    ctx.stroke()
    # ---- neck
    _c(ctx, "flesh_d")
    ctx.rectangle(-52, 190, 104, 130)
    ctx.fill()
    _c(ctx, "flesh")
    ctx.rectangle(-30, 190, 60, 120)
    ctx.fill()
    # ---- hair (dark brown cap) and beard
    _c(ctx, "hair")
    _ell(ctx, 0, -20, 170, 205)
    ctx.fill()
    _c(ctx, "umber")
    ctx.move_to(-150, 60)
    ctx.curve_to(-160, 200, -80, 265, 0, 270)
    ctx.curve_to(80, 265, 160, 200, 150, 60)
    ctx.close_path()
    ctx.fill()
    # hair strands (lighter course lines)
    _c(ctx, "#5E3E28")
    ctx.set_line_width(lw * 0.7)
    for i in range(-3, 4):
        ctx.move_to(i * 36, -215)
        ctx.curve_to(i * 50, -170, i * 62, -120, i * 70, -70)
        ctx.stroke()
    # ---- face oval (flesh in three tones: shadow, mid, light)
    _c(ctx, "flesh_d")
    _ell(ctx, 0, 10, 122, 168)
    ctx.fill()
    _c(ctx, "flesh")
    _ell(ctx, 8, 0, 100, 150)
    ctx.fill()
    _c(ctx, "flesh_l")
    _ell(ctx, 18, -60, 58, 60)
    ctx.fill()
    _ell(ctx, 36, 40, 26, 50)
    ctx.fill()
    # green-olive proplasmos shadow along the left jaw
    _c(ctx, "#8C7B58", 0.85)
    ctx.move_to(-118, -10)
    ctx.curve_to(-122, 70, -90, 140, -40, 170)
    ctx.curve_to(-70, 120, -96, 60, -100, -10)
    ctx.close_path()
    ctx.fill()
    # cheeks
    _c(ctx, "rose")
    _ell(ctx, -62, 58, 22, 16)
    ctx.fill()
    _ell(ctx, 70, 58, 22, 16)
    ctx.fill()
    # ---- brows (one arched course each), nose, eyes, mouth
    _c(ctx, "hair")
    ctx.set_line_width(lw * 0.95)
    ctx.move_to(-92, -44)
    ctx.curve_to(-70, -72, -30, -72, -10, -48)
    ctx.move_to(10, -48)
    ctx.curve_to(30, -72, 70, -72, 92, -44)
    ctx.stroke()
    # nose: long line from the brow + nostril curl
    _c(ctx, "flesh_s")
    ctx.set_line_width(lw * 0.7)
    ctx.move_to(-8, -44)
    ctx.curve_to(-12, 0, -16, 40, -20, 66)
    ctx.curve_to(-8, 76, 12, 76, 22, 64)
    ctx.stroke()
    # eyes: white almond, dark iris, dark upper lid
    for ex in (-54, 54):
        _c(ctx, "white")
        _ell(ctx, ex, -16, 30, 15)
        ctx.fill()
        _c(ctx, "hair")
        ctx.arc(ex + (4 if ex < 0 else -4), -15, 11, 0, 2 * math.pi)
        ctx.fill()
        _c(ctx, "ink")
        ctx.set_line_width(lw * 0.6)
        ctx.move_to(ex - 32, -12)
        ctx.curve_to(ex - 18, -34, ex + 18, -34, ex + 32, -12)
        ctx.stroke()
        _c(ctx, "flesh_s")
        ctx.set_line_width(lw * 0.5)
        ctx.move_to(ex - 26, 6)
        ctx.curve_to(ex - 10, 14, ex + 10, 14, ex + 26, 6)
        ctx.stroke()
    # mouth
    _c(ctx, "porphyry_l")
    _ell(ctx, 0, 112, 30, 10)
    ctx.fill()
    _c(ctx, "ink")
    ctx.set_line_width(lw * 0.5)
    ctx.move_to(-30, 110)
    ctx.curve_to(-10, 116, 10, 116, 30, 110)
    ctx.stroke()
    # face outline
    _c(ctx, "flesh_s")
    ctx.set_line_width(lw * 0.8)
    _ell(ctx, 0, 10, 122, 168)
    ctx.stroke()
    ctx.restore()


def saint_panel(s=1.0, tile=12.0, fine_tile=5.5, cy=540):
    """cartoon + fine mask + titulus for a saint on a gold ground (the sheet's test wall)."""
    fc = _FC(s)
    cart = Canvas(fc, bg=S["gold"])
    saint(cart.ctx, 960, cy, 1.0, "color", tile)
    t = mz.Titulus("[MAG] · CROCVS", 960, cy - 350, 60, color=S["lapis_d"], font="roman_black", tracking=0.16)
    t.draw(cart.ctx)
    fine = Canvas(fc)
    saint(fine.ctx, 960, cy, 1.0, "fine", tile)
    t.draw_fine(fine.ctx)
    return fc, cart, fine, t


def saint_layout(cart, fine, tile=12.0, fine_tile=5.5, seed=7):
    rgb = cart.rgb()
    gold = np.abs(rgb - mz.SC["gold"]).sum(-1) < 0.08
    lay = mz.Layout.flow(rgb, tile=tile, fine=fine.rgba()[..., 3] > 0.5, fine_tile=fine_tile, gold=gold,
                         ground=gold, seed=seed)
    return rgb, gold, lay


def dome_cartoon(R, s=0.25, seed=5):
    """a dome programme in the Dome chart (disc of radius 2R): gold medallion with a twelve-petalled rosette, twelve gold
    ribs, a starry lapis sky, a jewelled band and an inscription round the springing."""
    ext = (int(4 * R), int(4 * R))
    c = 2 * R
    cv = Canvas(s=s, w=int(ext[0] * s), h=int(ext[1] * s), bg=S["night_vault"])
    ctx = cv.ctx
    rng = np.random.default_rng(seed)
    # sky gradient rings (deeper blue toward the crown)
    for r, name in ((2 * R, "lapis"), (1.5 * R, "lapis_d"), (0.9 * R, "night_vault")):
        _c(ctx, name)
        ctx.arc(c, c, r, 0, 2 * math.pi)
        ctx.fill()
    # stars
    for i in range(420):
        r = rng.uniform(0.3 * R, 1.62 * R)
        a = rng.uniform(0, 2 * math.pi)
        x, y = c + r * math.cos(a), c + r * math.sin(a)
        k = 10 + 18 * (r / (2 * R)) + rng.uniform(0, 8)
        _c(ctx, "star" if rng.random() < 0.7 else "gold_l")
        ctx.move_to(x, y - k)
        ctx.line_to(x + k * 0.28, y)
        ctx.line_to(x, y + k)
        ctx.line_to(x - k * 0.28, y)
        ctx.close_path()
        ctx.fill()
        ctx.move_to(x - k, y)
        ctx.line_to(x, y + k * 0.28)
        ctx.line_to(x + k, y)
        ctx.line_to(x, y - k * 0.28)
        ctx.close_path()
        ctx.fill()
    # ribs (offset so no pair lines up into a cross)
    for k in range(12):
        a = k * math.pi / 6 + math.pi / 12
        w0, w1 = 26.0, 70.0
        r0, r1 = 0.36 * R, 1.66 * R
        ca, sa = math.cos(a), math.sin(a)
        pts = [(c + r0 * ca - w0 * sa, c + r0 * sa + w0 * ca), (c + r1 * ca - w1 * sa, c + r1 * sa + w1 * ca),
               (c + r1 * ca + w1 * sa, c + r1 * sa - w1 * ca), (c + r0 * ca + w0 * sa, c + r0 * sa - w0 * ca)]
        _c(ctx, "gold")
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.fill()
        _c(ctx, "porphyry")
        ctx.set_line_width(12)
        ctx.move_to(c + (r0 + 20) * ca, c + (r0 + 20) * sa)
        ctx.line_to(c + (r1 - 20) * ca, c + (r1 - 20) * sa)
        ctx.stroke()
    # medallion + star
    _c(ctx, "gold")
    ctx.arc(c, c, 0.4 * R, 0, 2 * math.pi)
    ctx.fill()
    _c(ctx, "ink")
    ctx.set_line_width(18)
    ctx.arc(c, c, 0.4 * R, 0, 2 * math.pi)
    ctx.stroke()
    _c(ctx, "lapis")
    ctx.arc(c, c, 0.3 * R, 0, 2 * math.pi)
    ctx.fill()
    _c(ctx, "gold_l")                     # a twelve-petalled rosette
    for k in range(12):
        a = k * math.pi / 6 + math.pi / 12
        ctx.save()
        ctx.translate(c + 0.16 * R * math.cos(a), c + 0.16 * R * math.sin(a))
        ctx.rotate(a)
        ctx.scale(0.12 * R, 0.035 * R)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.fill()
    ctx.arc(c, c, 0.05 * R, 0, 2 * math.pi)
    ctx.fill()
    # jewelled band
    _c(ctx, "gold")
    ctx.set_line_width(0.12 * R)
    ctx.arc(c, c, 1.76 * R, 0, 2 * math.pi)
    ctx.stroke()
    for i in range(64):
        a = i * 2 * math.pi / 64
        x, y = c + 1.76 * R * math.cos(a), c + 1.76 * R * math.sin(a)
        _c(ctx, ("porphyry", "emerald", "white", "lapis_l")[i % 4])
        ctx.arc(x, y, 0.035 * R, 0, 2 * math.pi)
        ctx.fill()
    # border lines
    _c(ctx, "ink")
    ctx.set_line_width(14)
    for r in (1.69 * R, 1.83 * R):
        ctx.arc(c, c, r, 0, 2 * math.pi)
        ctx.stroke()
    _c(ctx, "porphyry")
    ctx.arc(c, c, 2 * R, 0, 2 * math.pi)
    ctx.arc_negative(c, c, 1.86 * R, 2 * math.pi, 0)
    ctx.fill()
    rgb = cv.rgb()
    gold = (np.abs(rgb - mz.SC["gold"]).sum(-1) < 0.1) | (np.abs(rgb - mz.SC["gold_l"]).sum(-1) < 0.1)
    return ext, rgb, gold


def _u8(img):
    return cv2.cvtColor((np.clip(img[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, img.shape[0] - 34), (img.shape[1], img.shape[0]), (18, 16, 20), -1)
    cv2.putText(img, text, (12, img.shape[0] - 11), cv2.FONT_HERSHEY_DUPLEX, 0.62, (225, 215, 200), 1, cv2.LINE_AA)
    return img


def main():
    from . import mosaic_fx as fx
    out = OUT / "mosaic_sheet"
    out.mkdir(parents=True, exist_ok=True)
    times = {}
    t0 = time.time()
    fc, cart, fine, tit = saint_panel()
    rgb, gold, lay = saint_layout(cart, fine)
    cols = mz.sample(rgb, lay)
    times["layout saint (setup)"] = time.time() - t0
    panels = {}
    amb = 0.24
    # ---- gold at two candle positions (same wall, same tiles)
    for name, pos in (("goldA", (430, 180, 620)), ("goldB", (1620, 900, 620))):
        L = mz.candles(1.3, [pos], power=1.9, radius=900) + [mz.Light(dir=(-0.3, -0.55, 0.78), power=0.35)]
        t1 = time.time()
        img = mz.render(fc, None, lay, rgb=cols, light=L, ambient=amb)
        times[f"render {name} (1080p, {len(lay)} tiles)"] = time.time() - t1
        cv2.imwrite(str(out / f"{name}_full.png"), _u8(img))
        panels[name] = img[60:600, 0:960]
    # ---- the whole wall (andamento overview)
    L = mz.candles(2.0, [(700, 250, 900)], power=1.5, radius=1300) + [mz.Light(dir=(-0.3, -0.55, 0.78), power=0.5)]
    img = mz.render(fc, None, lay, rgb=cols, light=L, ambient=0.3)
    cv2.imwrite(str(out / "saint_full.png"), _u8(img))
    panels["wall"] = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
    # ---- vermiculatum face: zoom 1.7 on the face
    t1 = time.time()
    img = mz.render(fc, None, lay, rgb=cols, light=L, ambient=0.3, view=mz.View(960, 545, zoom=1.7))
    times["render zoom 1.7"] = time.time() - t1
    panels["face"] = img[270:810, 480:1440]
    # ---- sinopia patch + sockets: the lower right of the wall has fallen
    t1 = time.time()
    sin = mz.Sinopia((W, H), cartoon=rgb, seed=3, s=0.75)
    times["sinopia build (setup)"] = time.time() - t1
    rng = np.random.default_rng(4)
    x, y = lay.xy[:, 0], lay.xy[:, 1]
    d = np.hypot(x - 1330, y - 560) + 60 * np.sin(x / 37.0) + 40 * np.cos(y / 23.0) + rng.normal(0, 18, len(lay))
    pres = np.where(d < 300, -1, np.where(d < 400, 0, 1))
    L2 = mz.candles(0.5, [(1250, 420, 700)], power=1.2, radius=1100) + [mz.Light(dir=(-0.3, -0.55, 0.78), power=0.45)]
    bg = sin.view(fc, None, light=L2, ambient=0.28)
    img = mz.render(fc, None, lay, rgb=cols, present=pres, bg=bg, light=L2, ambient=0.28)
    panels["sinopia"] = img[290:830, 880:1840]
    # ---- falling tiles (Babel) and the heap
    td = fx.wave_times(lay, (1920, 60), speed=520, t0=0.1, jitter=0.1, seed=2)
    td[x < 1060] = np.inf
    fall = fx.Fall(lay, td, floor=1075, seed=1)
    L3 = mz.candles(1.0, [(1350, 350, 800)], power=1.4, radius=1400) + [mz.Light(dir=(-0.3, -0.55, 0.78), power=0.4)]
    t1 = time.time()
    tt = 1.45
    pres = np.where(fall.fallen(tt), 0, 1)
    img = mz.render(fc, None, lay, rgb=cols, present=pres, light=L3, ambient=0.26, **fall.pose(tt))
    times[f"render falling ({int(fall.fallen(tt).sum())} loose tiles)"] = time.time() - t1
    panels["fall"] = img[60:600, 960:1920]
    tt = 5.0
    pres = np.where(fall.fallen(tt), np.where(tt - td > 2.2, -1, 0), 1)
    cam = mz.Camera(eye=(1480, 560, 1350), target=(1420, 930, 0), fov=58)
    bgw = sin.view(fc, cam, surface=mz.Plane(), light=L3, ambient=0.26)
    floor_y = 1075
    # a floor of dark marble slabs (painted, not tessellated) under the heap
    fl = np.zeros_like(bgw)
    gx, gy = np.meshgrid(np.arange(fc.w), np.arange(fc.h))
    D = cam.ray(np.stack([gx / fc.s, gy / fc.s], -1).astype(np.float64))
    tfl = (floor_y - cam.eye[1]) / np.where(np.abs(D[..., 1]) < 1e-6, 1e-6, D[..., 1])
    hit = (tfl > 0) & ((cam.eye[2] + tfl * D[..., 2]) > 0)
    fl[hit] = np.array([0.20, 0.17, 0.15], np.float32) * (1.0 - 0.25 * ((((cam.eye[0] + tfl * D[..., 0]) // 160) +
                                                                       ((cam.eye[2] + tfl * D[..., 2]) // 160)) % 2))[hit, None]
    bgc = np.where(hit[..., None], fl, bgw)
    img = mz.render(fc, None, lay, rgb=cols, present=pres, bg=bgc, light=L3, ambient=0.26, view=cam,
                    **fall.pose(tt))
    panels["heap"] = img[380:920, 480:1440]
    # ---- dome from below
    R = 900.0
    t1 = time.time()
    ext, drgb, dgold = dome_cartoon(R)
    dome = mz.Dome(R, center=(0, 0, 0), chart_center=(2 * R, 2 * R))
    dlay = mz.Layout.flow(drgb, tile=15, extent=ext, gold=dgold, ground=dgold, density=dome.k, seed=3)
    dlay = dlay.keep(lambda u, v: np.hypot(u - 2 * R, v - 2 * R) < 2 * R - 4)
    times[f"layout dome (setup, {len(dlay)} tiles)"] = time.time() - t1
    cam = mz.dome_up(dome, height=1.15 * R, rot=0.3)
    Ld = mz.candles(0.7, [(500, 950, 300), (-600, 950, -200), (100, 950, -700)], power=2.2, radius=1500)
    t1 = time.time()
    img = mz.render(fc, drgb, dlay, view=cam, surface=dome, light=Ld, ambient=0.22, bg=(0.02, 0.018, 0.02))
    times["render dome"] = time.time() - t1
    cv2.imwrite(str(out / "dome_full.png"), _u8(img))
    panels["dome"] = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
    # ---- tituli band (2880 x 540 at 1:1)
    tw, th = 2880, 540
    fct = _FC(1.0, w=tw, h=th)
    cvt = Canvas(s=1.0, w=tw, h=th, bg=S["gold"])
    c = cvt.ctx
    _c(c, "lapis")
    c.rectangle(0, 200, tw, 150)
    c.fill()
    _c(c, "ink")
    c.set_line_width(13)
    for yy in (200, 350):
        c.move_to(0, yy)
        c.line_to(tw, yy)
    c.stroke()
    _c(c, "marble")
    c.rectangle(0, 400, tw, 120)
    c.fill()
    t_a = mz.Titulus("Explicit opus schismaticum", tw / 2, 160, 96, color=S["lapis_d"], font="roman_black",
                     sep="·", tracking=0.13)
    t_b = mz.Titulus("omnia vexilla + [vnvm] vexillvm", tw / 2, 312, 76, color=S["white"], font="roman",
                     tracking=0.16, dot_color=S["gold"])
    t_c = mz.Titulus("~ hoc est [vxl] signvm sine svtvra ~", tw / 2, 490, 62, color=S["porphyry"], font="byzantine",
                     tracking=0.12)
    for tt_ in (t_a, t_b, t_c):
        tt_.draw(c)
    fine_t = Canvas(s=1.0, w=tw, h=th)
    for tt_ in (t_a, t_b, t_c):
        tt_.draw_fine(fine_t.ctx)
    trgb = cvt.rgb()
    tgold = np.abs(trgb - mz.SC["gold"]).sum(-1) < 0.08
    t1 = time.time()
    tlay = mz.Layout.flow(trgb, tile=13, extent=(tw, th), fine=fine_t.rgba()[..., 3] > 0.5, fine_tile=6.5,
                          gold=tgold, ground=tgold, seed=11)
    times[f"layout tituli (setup, {len(tlay)} tiles)"] = time.time() - t1
    Lt = mz.candles(0.4, [(700, 120, 700), (2200, 300, 700)], power=1.25, radius=1000) + \
        [mz.Light(dir=(-0.3, -0.55, 0.78), power=0.45)]
    img = mz.render(fct, trgb, tlay, light=Lt, ambient=0.3, view=mz.View(tw / 2, th / 2, at=(tw / 2, th / 2)))
    band = img
    # ---- compose
    rows = [
        [_label(_u8(panels["goldA"]), "gold ground, candle upper left"),
         _label(_u8(panels["goldB"]), "same tiles, candle lower right (view/light-dependent glitter)"),
         _label(_u8(panels["wall"]), "andamento: outline + halo courses, straight ground rows")],
        [_label(_u8(panels["face"]), "opus vermiculatum face (fine tiles, zoom 1.7, crisp)"),
         _label(_u8(panels["dome"]), "dome from below (stereographic chart, ribs, stars)"),
         _label(_u8(panels["sinopia"]), "fallen: setting bed with imprints, plaster + red sinopia")],
        [_label(_u8(panels["fall"]), "Babel: tesserae tumbling off the wall (gravity, 3D spin)"),
         _label(_u8(panels["heap"]), "rest: bounced, rattled, heaped; sinopia revealed"),
         _label(_u8(cv2.resize(band[:, 960:1920], (960, 540))), "tituli (detail)")],
    ]
    grid = np.vstack([np.hstack(r) for r in rows])
    band8 = _label(_u8(band), "tituli: Roman capitals set in fine tesserae; overlines, interpuncts, rosette, hedera")
    sheet = np.vstack([grid, band8])
    title = np.full((70, sheet.shape[1], 3), 18, np.uint8)
    cv2.putText(title, "OPVS SCHISMATICVM - vx.mosaic reference sheet (every pixel below is code: tesserae, gold, "
                       "mortar, sinopia)", (20, 46), cv2.FONT_HERSHEY_DUPLEX, 1.0, (225, 215, 200), 1, cv2.LINE_AA)
    sheet = np.vstack([title, sheet])
    cv2.imwrite(str(OUT / "mosaic_sheet.png"), sheet)
    for k, v in times.items():
        print(f"{k:48s} {v:6.2f} s")
    print("->", OUT / "mosaic_sheet.png", sheet.shape)


if __name__ == "__main__":
    main()

"""s05b shared helpers: tiny crowds, crowns, speech bubbles, shout rings, screen-shard transitions."""
import math

import cairo
import numpy as np
from scipy.spatial import Voronoi

from vx import *

T0 = 100.5            # scene start (global s)


_FALLBACK = {"declare": "cheer", "one_finger": "point", "point_self": "talk", "cup_shout": "shout", "gesture": "talk",
             "recoil": "stand", "shrug": "talk", "radio": "salute", "wave": "cheer"}


def pose_ok(pose):
    """map poses to what the installed vx.chars supports (RigMaster's rig adds poses over time)"""
    from vx import chars
    known = set(getattr(chars, "POSES", ()))
    fix = lambda p: p if p in known else _FALLBACK.get(p, "stand")
    if isinstance(pose, dict):
        out = {}
        for k, v in pose.items():
            k2 = fix(k)
            out[k2] = out.get(k2, 0.0) + v
        return out
    return fix(pose)


COOL_CLOTH = ["cloth_grey", "cloth_blue", "cloth_teal", "cloth_plum", "#5C6B80", "#6E7C96", "#4F6470", "#7A6E8A"]
SILVER = "#D9DEE8"
SILVER_D = "#8E97A8"


def lg(T):
    """global -> scene-local seconds"""
    return T - T0


def crown(ctx, x, y, w, alpha=1.0, tilt=0.0, col=SILVER, dark=SILVER_D):
    """tiny silver crown, base centre at (x, y), width w (never yellow)"""
    if w < 0.8 or alpha <= 0:
        return
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(tilt)
    h = w * 0.62
    pts = [(-w / 2, 0), (-w / 2, -h * 0.75), (-w / 4, -h * 0.35), (0, -h), (w / 4, -h * 0.35), (w / 2, -h * 0.75),
           (w / 2, 0)]
    poly(ctx, pts)
    ctx.set_source(lin_grad(-w / 2, -h, w / 2, 0, [(0, "#F4F6FA", alpha), (0.55, col, alpha), (1, dark, alpha)]))
    ctx.fill()
    if w > 6:
        set_color(ctx, dark, alpha * 0.9)
        rect(ctx, -w / 2, -h * 0.16, w, h * 0.16)
        ctx.fill()
        # gem tips: pale cool glints
        for gx, gy in ((-w / 2, -h * 0.75), (0, -h), (w / 2, -h * 0.75)):
            set_color(ctx, "#EEF3FF", alpha)
            circle(ctx, gx, gy, max(0.6, w * 0.06))
            ctx.fill()
    ctx.restore()


def mini_person(ctx, x, y, h, cloth, skin, t=0.0, seed=0, crowned=True, alpha=1.0, arm=0.0):
    """cheap readable tiny figure (for far/zooming crowds): robe, head, optional crown. arm: 0 down .. 1 raised"""
    if h < 1.5 or alpha <= 0:
        return
    bob = math.sin(t * 3.1 + seed) * h * 0.01
    r = h * 0.13
    set_color(ctx, "#000000", 0.18 * alpha)
    ellipse(ctx, x, y, h * 0.2, h * 0.06)
    ctx.fill()
    set_color(ctx, cloth, alpha)
    poly(ctx, [(x - h * 0.17, y), (x + h * 0.17, y), (x + h * 0.09, y - h * 0.72 + bob), (x - h * 0.09, y - h * 0.72 + bob)])
    ctx.fill()
    set_color(ctx, shade(cloth, 0.72), alpha)
    poly(ctx, [(x + h * 0.02, y), (x + h * 0.17, y), (x + h * 0.09, y - h * 0.72 + bob), (x + h * 0.02, y - h * 0.72 + bob)])
    ctx.fill()
    if arm > 0.01 and h > 6:
        set_color(ctx, cloth, alpha)
        ctx.set_line_width(h * 0.06)
        a = -0.3 - arm * 2.3
        sx, sy = x + h * 0.08, y - h * 0.64 + bob
        ctx.move_to(sx, sy)
        ctx.line_to(sx + math.cos(a) * h * 0.3, sy + math.sin(a) * h * 0.3)
        ctx.stroke()
    set_color(ctx, skin, alpha)
    circle(ctx, x, y - h * 0.72 - r * 0.9 + bob, r)
    ctx.fill()
    if crowned:
        crown(ctx, x, y - h * 0.72 - r * 1.55 + bob, r * 1.7, alpha)


def shout_rings(ctx, x, y, t, rate=3.0, r0=10, r1=120, n=3, col="#E8ECF8", alpha=0.8, ang=0.0, spread=0.9, lw=2.5):
    """expanding arcs from a mouth (x, y) toward direction ang (rad); spread = arc half-angle"""
    for k in range(n):
        ph = (t * rate + k / n) % 1.0
        r = r0 + (r1 - r0) * ph
        a = alpha * (1 - ph) ** 1.5
        if a <= 0.01:
            continue
        ctx.new_path()
        ctx.arc(x, y, r, ang - spread, ang + spread)
        set_color(ctx, col, a)
        ctx.set_line_width(lw * (1.2 - 0.6 * ph))
        ctx.stroke()


def bubble(ctx, x, y, w, h, tail=(0, 0), fill="#F3F5FA", edge="#2A2F45", alpha=1.0, wobble=0.0, t=0.0, lw=2.4):
    """rounded speech bubble; tail points at `tail` (absolute coords)"""
    ctx.save()
    pts = []
    n = 28
    for i in range(n):
        a = i / n * 2 * math.pi
        rr = 1 + wobble * 0.04 * math.sin(a * 5 + t * 9)
        pts.append((x + math.cos(a) * w / 2 * rr, y + math.sin(a) * h / 2 * rr))
    smooth(ctx, pts, close=True)
    # tail
    tx, ty = tail
    ang = math.atan2(ty - y, tx - x)
    bx, by = x + math.cos(ang) * w * 0.38, y + math.sin(ang) * h * 0.38
    px, py = -math.sin(ang), math.cos(ang)
    bw = min(w, h) * 0.14
    ctx.move_to(bx + px * bw, by + py * bw)
    ctx.line_to(tx, ty)
    ctx.line_to(bx - px * bw, by - py * bw)
    ctx.close_path()
    set_color(ctx, fill, alpha)
    ctx.fill_preserve()
    set_color(ctx, edge, alpha)
    ctx.set_line_width(lw)
    ctx.stroke()
    ctx.restore()


def scribble(ctx, x, y, w, h, seed, t=0.0, col="#2A2F45", alpha=1.0, lw=1.6, lines=3):
    """illegible handwriting scrawl inside a box (no real text)"""
    rng = np.random.default_rng(seed)
    set_color(ctx, col, alpha)
    ctx.set_line_width(lw)
    for L in range(lines):
        yy = y + (L + 0.5) * h / lines
        xx = x
        ctx.move_to(xx, yy)
        end = x + w * rng.uniform(0.6, 1.0)
        while xx < end:
            step = rng.uniform(3, 9) * lw
            amp = rng.uniform(0.2, 0.5) * h / lines
            ctx.curve_to(xx + step * 0.3, yy - amp, xx + step * 0.7, yy + amp, xx + step, yy + rng.uniform(-1, 1) * amp * 0.3)
            xx += step
        ctx.stroke()


GLYPHS = ["∑", "∫", "π", "∞", "√", "≠", "∂", "=", "≈", "?", "!", "∴", "∀", "∃", "λ", "Ω", "φ", "±", "÷", "×", "△"]


def pseudo_eq(ctx, x, y, size, seed, col="#2A2F45", alpha=1.0, n=5):
    """a nonsense 'equation' of maths glyphs (unreadable as language)"""
    rng = np.random.default_rng(seed)
    s = "".join(GLYPHS[i] for i in rng.integers(0, len(GLYPHS), n))
    text(ctx, s, x, y, size, col, font=FONT_SERIF, align="center")


# ------------------------------------------------------------------ screen-shard transition
def shard_cells(seed, n=26, w=W, h=H):
    """Voronoi cells over the frame (design units), each a polygon list + centroid"""
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.uniform(-0.05, 1.05, n) * w, rng.uniform(-0.05, 1.05, n) * h]
    far = np.array([[-w * 3, -h * 3], [w * 4, -h * 3], [-w * 3, h * 4], [w * 4, h * 4]])
    vor = Voronoi(np.r_[pts, far])
    from shapely.geometry import Polygon, box
    frame = box(-2, -2, w + 2, h + 2)
    cells = []
    for i in range(n):
        reg = vor.regions[vor.point_region[i]]
        if -1 in reg or not reg:
            continue
        pg = Polygon(vor.vertices[reg]).intersection(frame)
        if pg.is_empty or pg.area < 10:
            continue
        xy = np.array(pg.exterior.coords)[:-1]
        c = np.array(pg.centroid.coords[0])
        cells.append((xy, c))
    return cells


def draw_shards(ctx, img_rgb, cells, p, s, seed=0, origin=(W / 2, H / 2), fly=1.0, edge_col="#EAF6FF"):
    """draw the image cut into `cells` flying toward the camera (p 0..1). img_rgb: pixel array of the old frame."""
    surf = surface_from_array(img_rgb)
    rng = np.random.default_rng(seed)
    order = []
    for k, (xy, c) in enumerate(cells):
        d = c - np.array(origin)
        dist = np.hypot(*d) / 1100.0
        delay = dist * 0.35 + rng.uniform(0, 0.12)
        q = clamp((p - delay) / max(1e-3, 1 - delay))
        order.append((q, k, xy, c, d, rng.uniform(-1, 1), rng.uniform(0.6, 1.4)))
    order.sort(key=lambda o: o[0])
    for q, k, xy, c, d, spin, sp in order:
        e = q * q
        if e >= 0.999:
            continue
        sc = 1 + e * 2.6 * sp * fly
        ang = spin * e * 1.4
        off = d * e * 1.3 * sp * fly
        ctx.save()
        ctx.translate(c[0] + off[0], c[1] + off[1] + e * e * 200)
        ctx.rotate(ang)
        ctx.scale(sc, sc)
        ctx.translate(-c[0], -c[1])
        poly(ctx, [tuple(v) for v in xy])
        ctx.save()
        ctx.clip_preserve()
        ctx.scale(1.0 / s, 1.0 / s)
        ctx.set_source_surface(surf, 0, 0)
        ctx.paint_with_alpha(1 - e ** 3)
        ctx.restore()
        set_color(ctx, edge_col, (1 - e) * min(1, q * 8) * 0.9)
        ctx.set_line_width(2.0 / sc)
        ctx.stroke()
        ctx.restore()

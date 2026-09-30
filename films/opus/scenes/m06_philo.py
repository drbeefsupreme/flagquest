"""m06 PHILOSOPHIES (120.0-128.6): the Academy (the Pompeii philosophers' mosaic, set in Byzantine gold): seven
philosophers under a tree, a sundial column, a celestial globe, the city on its hill, HIC SAPIENTES DISPVTANT.
At K06 ("crackpot") the floor breaks into rocks; the gold, the garland frame, the city and the titulus dissolve
into floating tesserae drifting away into the black void; each philosopher is left alone on his floating rock.
X06: CRACKPOT1 rants on his rock (it splits under him at "poets"). X07: CRACKPOT2 points at himself while
SVM ERGO SVM sets itself in the void beside him.

World = panel design units; a square wall (y -420..1500) so the portal tessera shows all of it.
"""
import math

import cv2
import numpy as np

from vx import W, H, Canvas, set_color, circle, ellipse, poly, smooth, value_noise2, hash01, smootherstep, ease_out
from vx import icon
from vx import mosaic as mz
from vx.canvas import col, text

from scenes import m06_tess as tx

S = mz.SMALTI
FONT = "P052"
WORLD_Y0, WORLD_H = -420.0, 1920
T_BREAK = 121.08                      # "crackpot": the floor breaks
T_DRIFT = (121.1, 123.6)              # the rocks float apart
T_CRUMBLE = (121.1, 122.9)            # the gold dissolves into the void
T_POETS = 125.66                      # CRACKPOT1's rock splits again ("poets")
SVM = [("SVM", 126.69), ("ERGO", 127.04), ("SVM", 127.67)]

# the seven: feet (x, y), height, pose, kind, seed
PHILS = [(360, 905, 330, "stand", "philosopher", 3), (560, 888, 300, "sit", "philosopher", 5),
         (760, 878, 336, "stand", "crackpot", 2), (960, 872, 330, "stand", "philosopher", 8),
         (1160, 878, 300, "sit", "philosopher", 11), (1360, 888, 334, "stand", "crackpot", 6),
         (1560, 905, 328, "stand", "philosopher", 13)]
HEROES = {2: 1, 5: 2}                 # index -> CRACKPOT1 / CRACKPOT2
TREE_X, DIAL_X = 175.0, 1745.0
FLOOR = (905.0, 1010.0)               # the floor band (y)
# rocks: 0 = the tree, 1..7 the philosophers, 8 = the sundial + globe
ROCK_X = [TREE_X] + [p[0] for p in PHILS] + [DIAL_X]
KH = 400.0
SPR_W, SPR_H = 420.0, 520.0
GROUND = (210.0, 470.0)
VOID = np.array([0.018, 0.017, 0.028], np.float32)


# ============================================================ the Academy (static cartoon)
def _garland(ctx, x0, y0, x1, y1, w, colr):
    """a garland frame of leaves, fruits and ribbons."""
    set_color(ctx, S["malachite"] if colr else (1, 1, 1))
    ctx.set_line_width(w)
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.stroke()
    if not colr:
        return
    per = 2 * ((x1 - x0) + (y1 - y0))
    n = int(per / 34)
    for k in range(n):
        d = k * per / n
        if d < x1 - x0:
            x, y = x0 + d, y0
        elif d < (x1 - x0) + (y1 - y0):
            x, y = x1, y0 + d - (x1 - x0)
        elif d < 2 * (x1 - x0) + (y1 - y0):
            x, y = x1 - (d - (x1 - x0) - (y1 - y0)), y1
        else:
            x, y = x0, y1 - (d - 2 * (x1 - x0) - (y1 - y0))
        h = hash01(k, 3)
        set_color(ctx, S["verdigris"] if h < 0.5 else S["leaf"])
        ellipse(ctx, x, y, w * 0.42, w * 0.22, rot=k * 0.9)
        ctx.fill()
        if k % 5 == 0:
            set_color(ctx, (S["porphyry_l"], S["amethyst"], S["coral"])[k // 5 % 3])
            circle(ctx, x + w * 0.1, y - w * 0.05, w * 0.2)
            ctx.fill()


def draw_academy(ctx, layer):
    colr = layer == "color"
    wht = (1, 1, 1)
    # the wall around the panel: night-dark stone
    if colr:
        set_color(ctx, S["night_vault"])
        ctx.rectangle(-400, WORLD_Y0 - 100, W + 800, WORLD_H + 200)
        ctx.fill()
    # the gold ground
    if layer in ("color", "gold"):
        set_color(ctx, S["gold"] if colr else wht)
        ctx.rectangle(110, 70, W - 220, FLOOR[1] - 70)
        ctx.fill()
    if layer in ("color", "gold"):
        # everything painted on the gold is NOT gold leaf: in the gold layer it is painted black (value mask)
        def pc(c):
            set_color(ctx, c if colr else (0, 0, 0))

        # the city on its hill (upper right)
        pc(S["olive"])
        ctx.move_to(1180, 560)
        ctx.curve_to(1330, 330, 1600, 300, 1810, 470)
        ctx.line_to(1810, 560)
        ctx.close_path()
        ctx.fill()
        for k, (bx, bw, bh) in enumerate(((1330, 70, 90), (1415, 55, 130), (1480, 90, 70), (1580, 60, 110),
                                          (1650, 80, 80))):
            by = 470 - 0.3 * (bx - 1330) * (1 if bx < 1500 else -1) - 40
            pc(S["marble"])
            ctx.rectangle(bx, by - bh, bw, bh)
            ctx.fill()
            pc(S["terracotta"])
            poly(ctx, [(bx - 6, by - bh), (bx + bw / 2, by - bh - 28), (bx + bw + 6, by - bh)])
            ctx.fill()
            pc(S["ink"])
            ctx.set_line_width(3)
            ctx.rectangle(bx, by - bh, bw, bh)
            ctx.stroke()
        # the tree (left): trunk and a crown of leaves over the philosophers
        pc(S["umber"])
        poly(ctx, [(TREE_X - 22, FLOOR[0]), (TREE_X + 22, FLOOR[0]), (TREE_X + 14, 420), (TREE_X - 12, 420)])
        ctx.fill()
        ctx.set_line_width(16)
        for bx, by in ((TREE_X + 170, 300), (TREE_X - 60, 280), (TREE_X + 90, 200)):
            ctx.move_to(TREE_X, 450)
            ctx.line_to(bx, by)
            ctx.stroke()
        for k in range(26):
            a = hash01(k, 7) * 2 * math.pi
            r = 60 + 150 * hash01(k, 8)
            x = TREE_X + 70 + r * math.cos(a) * 1.4
            y = 290 + r * math.sin(a) * 0.7
            pc((S["malachite"], S["leaf"], S["verdigris"], S["emerald"])[k % 4])
            circle(ctx, x, y, 44 + 18 * hash01(k, 9))
            ctx.fill()
        for k in range(12):
            pc(S["gold_l"])
            circle(ctx, TREE_X + 70 + 190 * math.cos(k * 2.1), 290 + 90 * math.sin(k * 1.7), 6)
            ctx.fill()
        # the sundial on its column (right) and the celestial globe at its foot
        pc(S["marble"])
        ctx.rectangle(DIAL_X - 20, 470, 40, FLOOR[0] - 470)
        ctx.fill()
        pc(S["marble_d"])
        ctx.rectangle(DIAL_X + 4, 470, 12, FLOOR[0] - 470)
        ctx.fill()
        pc(S["bone"])
        ctx.arc(DIAL_X, 462, 46, 0, math.pi)
        ctx.close_path()
        ctx.fill()
        pc(S["ink"])
        ctx.set_line_width(3)
        ctx.arc(DIAL_X, 462, 46, 0, math.pi)
        ctx.close_path()
        ctx.stroke()
        for k in range(7):
            a = math.pi * (k + 0.5) / 7
            ctx.move_to(DIAL_X, 462)
            ctx.line_to(DIAL_X + 44 * math.cos(a), 462 + 44 * math.sin(a))
        ctx.stroke()
        gx, gy = DIAL_X - 5, FLOOR[0] - 70
        pc(S["sienna"])
        ctx.rectangle(gx - 70, gy + 20, 140, 50)
        ctx.fill()
        pc(S["lapis"])
        circle(ctx, gx, gy, 48)
        ctx.fill()
        pc(S["gold_l"])
        ctx.set_line_width(5)
        ellipse(ctx, gx, gy, 48, 14, rot=0.35)
        ctx.stroke()
        ellipse(ctx, gx, gy, 14, 48, rot=0.35)
        ctx.stroke()
        # the floor: an earthen terrace with stones
        pc(S["sienna"])
        ctx.rectangle(110, FLOOR[0], W - 220, FLOOR[1] - FLOOR[0])
        ctx.fill()
        for k in range(46):
            x = 130 + k * 37 + 10 * hash01(k, 11)
            y = FLOOR[0] + 20 + (FLOOR[1] - FLOOR[0] - 40) * hash01(k, 12)
            pc((S["ochre"], S["umber"], S["bone"], S["terracotta"])[k % 4])
            ellipse(ctx, x, y, 14 + 6 * hash01(k, 13), 9, rot=hash01(k, 14))
            ctx.fill()
        pc(S["ink"])
        ctx.set_line_width(4)
        ctx.move_to(110, FLOOR[0])
        ctx.line_to(W - 110, FLOOR[0])
        ctx.stroke()
    # the titulus band at the top
    if layer in ("color", "gold"):
        set_color(ctx, S["lapis_d"] if colr else (0, 0, 0))
        ctx.rectangle(110, 70, W - 220, 70)
        ctx.fill()
    if layer in ("color", "gold", "fine"):
        text(ctx, "HIC · SAPIENTES · DISPVTANT", W / 2, 105, 44, S["gold_l"] if colr else wht, font=FONT,
             bold=True, align="center", valign="middle", tracking=0.16)
    # the garland frame (never gold leaf)
    if layer == "gold":
        set_color(ctx, (0, 0, 0))
        ctx.set_line_width(40)
        ctx.rectangle(95, 55, W - 190, FLOOR[1] + 15 - 55)
        ctx.stroke()
    if layer in ("color", "fine"):
        _garland(ctx, 95, 55, W - 95, FLOOR[1] + 15, 36, colr)


def draw_labels(ctx):
    """label image: which rock a static tile leaves with (grey value = (rock + 1) * 20), 0 = dissolves."""
    for r, x in enumerate(ROCK_X):
        set_color(ctx, ((r + 1) * 20 / 255.0,) * 3)
        if r == 0:        # the tree with its crown and the floor under it
            ctx.rectangle(110, FLOOR[0] - 2, 170, FLOOR[1] - FLOOR[0] + 4)
            ctx.fill()
            poly(ctx, [(TREE_X - 26, FLOOR[0]), (TREE_X + 26, FLOOR[0]), (TREE_X + 18, 420), (TREE_X - 16, 420)])
            ctx.fill()
            ellipse(ctx, TREE_X + 70, 290, 270, 170)
            ctx.fill()
        elif r == 8:      # the sundial column and the globe
            ctx.rectangle(1660, 400, 160, FLOOR[1] - 400 + 2)
            ctx.fill()
        else:
            ctx.rectangle(x - 92, FLOOR[0] - 2, 184, FLOOR[1] - FLOOR[0] + 4)
            ctx.fill()


# ============================================================ figures
def phil_sprite(idx, ppu=0.9, tile=7.5, fine_tile=3.8):
    x, y, h, pose, kind, seed = PHILS[idx]
    fig = icon.Figure(kind, seed)
    hero = HEROES.get(idx, 0)

    def draw(ctx, layer, pose=pose, t=0.0, mouth=0.0, expr="wry", look=(0.0, 0.0), blink=False):
        fig.draw(ctx, GROUND[0], GROUND[1], KH, pose=pose, t=t, mouth=mouth, look=look, expr=expr, layer=layer,
                 tile=tile, blink=blink)
    poses = [dict(pose=pose)]
    if hero == 1:
        poses += [dict(pose="rant", t=0.0), dict(pose="rant", t=0.3), dict(pose="rant", t=0.6)]
    if hero == 2:
        poses += [dict(pose="point_self")]
    return tx.Sprite(SPR_W, SPR_H, draw, tile=tile, fine_tile=fine_tile, seed=800 + idx, poses=poses, ppu=ppu,
                     anchor=GROUND)


def word_sprite(word, size=70.0):
    w = size * (0.8 * len(word) + 0.4)

    def draw(ctx, layer, **st):
        if layer in ("color", "fine", "gold"):
            text(ctx, word, w / 2, size * 0.95, size, S["gold_l"] if layer == "color" else (1, 1, 1), font=FONT,
                 bold=True, align="center", tracking=0.1, stroke=S["porphyry"] if layer == "color" else None,
                 stroke_w=size * 0.08 if layer == "color" else 0.0)
    return tx.Sprite(w, size * 1.25, draw, tile=6.0, fine_tile=6.0, seed=990 + len(word), ppu=1.6,
                     anchor=(w / 2, size * 0.6))


def rock_underside(width, depth, seed):
    """the rocky underside of a floating chunk of floor (stone strata tapering to a point)."""
    w, h = width + 40, depth + 20

    def draw(ctx, layer, **st):
        if layer != "color":
            return
        rng = np.random.default_rng(seed)
        pts = [(20, 0), (w - 20, 0)]
        for k in range(6):
            u = (k + 1) / 7
            pts.append((w - 20 - u * (w / 2 - 20) + rng.normal(0, 10), u * depth))
        pts.append((w / 2 + rng.normal(0, 8), depth))
        for k in range(6):
            u = 1 - (k + 1) / 7
            pts.append((20 + (1 - u) * (w / 2 - 20) + rng.normal(0, 10), u * depth))
        set_color(ctx, S["sienna"])
        poly(ctx, pts)
        ctx.fill_preserve()
        set_color(ctx, S["ink"])
        ctx.set_line_width(4)
        ctx.stroke()
        for k in range(5):
            y = depth * (k + 0.6) / 6
            set_color(ctx, (S["ochre"], S["grey"], S["terracotta"])[k % 3])
            ctx.set_line_width(6)
            ctx.move_to(40 + y * 0.3, y)
            ctx.line_to(w - 40 - y * 0.3, y + rng.normal(0, 4))
            ctx.stroke()
    return tx.Sprite(w, h, draw, tile=11.0, fine_tile=11.0, seed=seed, ppu=0.6, anchor=(w / 2, 0.0))


# ============================================================ the panel
class PhiloPanel:
    def __init__(self, S_):
        self.tl = S_.tl

        def canvas(s, fn, layer=None):
            cv = Canvas(s=s, w=int(W * s), h=int(WORLD_H * s))
            cv.ctx.translate(0, -WORLD_Y0)
            fn(cv.ctx, layer) if layer is not None else fn(cv.ctx)
            return cv
        guide = canvas(0.5, draw_academy, "color").rgb()
        fine = canvas(0.5, draw_academy, "fine").rgba()[..., 3] > 0.5
        lay = mz.Layout.flow(guide, tile=12.0, seed=91, fine=fine, fine_tile=6.0, extent=(W, WORLD_H))
        self.lay = lay
        gold = canvas(1.0, draw_academy, "gold").rgb()[..., 0]
        self.pyr = tx.Pyramid(np.dstack([canvas(1.0, draw_academy, "color").rgb(), gold]), 0.0, WORLD_Y0, 1.0)
        self.xy = lay.xy.astype(np.float64) + np.array([0.0, WORLD_Y0])
        self.ang = lay.ang.astype(np.float64)
        self.size = lay.size.astype(np.float64)
        self.uid = np.arange(len(lay), dtype=np.int64) + 6 * 10 ** 7
        rgb, g = self.pyr.sample(self.xy[:, 0], self.xy[:, 1], self.size)
        self.rgb, self.gold = rgb, np.clip((g - 0.35) / 0.35, 0, 1)
        labs = canvas(1.0, draw_labels).rgb()[..., 0]
        ix = np.clip(self.xy[:, 0].astype(int), 0, W - 1)
        iy = np.clip((self.xy[:, 1] - WORLD_Y0).astype(int), 0, WORLD_H - 1)
        self.rock = np.round(labs[iy, ix] * 255 / 20).astype(int) - 1          # -1 = dissolves
        inside = (self.xy[:, 0] > 90) & (self.xy[:, 0] < W - 90) & (self.xy[:, 1] > 50) & (self.xy[:, 1] < FLOOR[1] + 25)
        self.floor = inside & (self.xy[:, 1] > FLOOR[0]) & (self.rock < 0)
        self.outside = ~inside
        # the dissolving picture: every loose tessera floats off on its own (pure f(T))
        rng = np.random.default_rng(93)
        n = len(self.xy)
        c = np.array([960.0, 620.0])
        d = self.xy - c
        dn = d / np.maximum(np.linalg.norm(d, axis=1), 1)[:, None]
        dist_floor = np.abs(self.xy[:, 1] - FLOOR[0])
        self.t_go = T_CRUMBLE[0] + 0.05 + (T_CRUMBLE[1] - T_CRUMBLE[0]) * np.clip(dist_floor / 900.0, 0, 1) \
            + rng.uniform(0, 0.35, n)
        self.t_go[self.floor] = T_BREAK + rng.uniform(0.0, 0.25, self.floor.sum())
        self.v = dn * rng.uniform(25, 90, (n, 1)) + rng.normal(0, 22, (n, 2))
        self.v[self.floor] += np.array([0.0, 40.0])
        self.w = rng.normal(0, 1.2, n)
        self.fade = np.where(self.floor, 99.0, rng.uniform(1.4, 2.6, n))
        # rocks: where each floats to (relative offset at full drift) and how it bobs
        R = len(ROCK_X)
        self.rock_d = np.zeros((R, 2))
        for r, x in enumerate(ROCK_X):
            self.rock_d[r] = ((x - 960) * 0.95 + rng.normal(0, 40), rng.uniform(-330, 260))
        self.rock_d[3] = (-150.0, -120.0)          # CRACKPOT1 floats up-left of centre
        self.rock_d[6] = (360.0, 170.0)            # CRACKPOT2 down-right
        self.rock_spin = rng.normal(0, 0.06, R)
        self.rock_phase = rng.uniform(0, 2 * math.pi, R)
        self.under = [rock_underside(190 if 0 < r < 8 else 175, 150 + 60 * rng.random(), 950 + r) for r in range(R)]
        # the philosophers
        self.spr = [phil_sprite(i) if i not in HEROES else phil_sprite(i, ppu=1.8, tile=6.0, fine_tile=3.0)
                    for i in range(len(PHILS))]
        self.words = [word_sprite(w_) for w_, _ in SVM]
        # CRACKPOT1's rock splits again at "poets": its right third drifts off
        r1 = 3
        k = np.nonzero(self.rock == r1)[0]
        self.poet_piece = k[self.xy[k, 0] > PHILS[2][0] + 38]

    # ------------------------------------------------------------------ motion (pure f(T))
    def rock_M(self, r, T, extra=None):
        """rock-local (= home coords) -> world affine."""
        u = float(np.clip((T - T_DRIFT[0]) / (T_DRIFT[1] - T_DRIFT[0]), 0, 1))
        q = u * u * (3 - 2 * u)
        lin = max(0.0, T - T_DRIFT[1]) * 0.05
        d = self.rock_d[r] * (q + lin)
        lift = smootherstep(T_BREAK, T_BREAK + 0.4, T)
        bob = np.array([0.0, 7.0 * math.sin(T * 1.3 + self.rock_phase[r])]) * lift
        th = self.rock_spin[r] * (q + lin) + 0.015 * math.sin(T * 0.9 + self.rock_phase[r]) * lift
        cx, cy = ROCK_X[r], FLOOR[0]
        c, s = math.cos(th), math.sin(th)
        p = np.array([cx, cy]) + d + bob
        return np.array([[c, -s, p[0] - (c * cx - s * cy)], [s, c, p[1] - (s * cx + c * cy)]])

    def phil_M(self, i, T):
        x, y, h, *_ = PHILS[i]
        k = h / KH
        L = np.array([[k, 0, x - k * GROUND[0]], [0, k, y - k * GROUND[1]]])
        return tx.compose(self.rock_M(i + 1, T)[None], L[None])[0]

    def phil_pos(self, i, T):
        M = self.phil_M(i, T)
        return M[:, :2] @ np.array(GROUND) + M[:, 2], PHILS[i][2]

    def hero_state(self, i, T):
        hid = HEROES[i]
        spk = "CRACKPOT1" if hid == 1 else "CRACKPOT2"
        tl = self.tl
        m = round(tl.mouth(spk, T) / 0.05) * 0.05
        talking = tl.speaking(spk, T, 0.2) is not None
        bl = tx.blink(T, 20 + hid) and not talking
        if hid == 1:
            if talking:
                return dict(pose="rant", t=round(T, 3), mouth=m, expr="shouting", blink=False)
            return dict(pose="stand", t=math.floor(T * 2) / 2, mouth=m, expr="wry", blink=bl)
        if talking:
            return dict(pose="point_self", t=math.floor(T * 2) / 2, mouth=m, expr="proud", blink=False)
        return dict(pose="stand", t=math.floor(T * 2) / 2, mouth=m, expr="wry", blink=bl)

    # ------------------------------------------------------------------ state of every static tessera
    def static_state(self, T):
        """per-tile home->world affines (N,2,3), alpha (N,), crack glow (N,3) and the rock affines at T."""
        n = len(self.xy)
        M = np.zeros((n, 2, 3))
        M[:, 0, 0] = M[:, 1, 1] = 1.0
        rock_mats = [self.rock_M(r, T) for r in range(len(ROCK_X))]
        for r, Rm in enumerate(rock_mats):
            M[self.rock == r] = Rm
        # the poet's piece: splits off CRACKPOT1's rock and drifts away
        u = T - T_POETS
        if u > 0:
            k = self.poet_piece
            q = ease_out(min(u / 0.35, 1.0), 3) * 0.4 + 0.6 * (1 - math.exp(-u / 1.2))
            th = 0.25 * q
            cx, cy = PHILS[2][0] + 60, FLOOR[0] + 40
            c, s = math.cos(th), math.sin(th)
            A = np.array([[c, -s, cx - (c * cx - s * cy) + 40 * q], [s, c, cy - (s * cx + c * cy) + 70 * q]])
            M[k] = tx.compose(np.repeat(rock_mats[3][None], len(k), 0), np.repeat(A[None], len(k), 0))
        # loose tesserae: float away on their own, the gold fades into the dark
        loose = self.rock < 0
        tau = np.where(loose, T - self.t_go, -1.0)
        mv = tau > 0
        alpha = np.ones(n, np.float32)
        if mv.any():
            k = np.nonzero(mv)[0]
            tk = tau[k]
            th = self.w[k] * tk
            c, s = np.cos(th), np.sin(th)
            A = np.zeros((len(k), 2, 3))
            A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] = c, -s, s, c
            cx, cy = self.xy[k, 0], self.xy[k, 1]
            off = self.v[k] * tk[:, None]
            A[:, 0, 2] = cx - (c * cx - s * cy) + off[:, 0]
            A[:, 1, 2] = cy - (s * cx + c * cy) + off[:, 1]
            M[k] = A
            alpha[k] = np.clip(1 - (tk - 0.3) / self.fade[k], 0, 1)
        # the outer wall (beyond the garland) is gone the moment the floor breaks
        alpha[self.outside] *= 1.0 - smootherstep(T_BREAK - 0.1, T_BREAK + 0.5, T)
        crack = np.zeros(n, np.float32)
        uu = T - T_BREAK
        if -0.08 < uu < 1.0:
            e = (math.exp(-max(uu, 0) / 0.25) if uu > 0 else 1 + uu / 0.08)
            crack[(self.rock >= 0) & (np.abs(self.xy[:, 1] - FLOOR[0]) < 60)] = e
            crack[self.floor] = e
        joint = (crack[:, None] * np.array([1.0, 0.86, 0.55], np.float32)[None] * 1.4).astype(np.float32)
        return M, alpha, joint, rock_mats

    def under_M(self, r, Rm, T):
        us = self.under[r]
        lift = smootherstep(T_BREAK + 0.05, T_BREAK + 0.7, T)
        L = np.array([[1.0, 0, ROCK_X[r] - us.anchor[0]], [0, lift, FLOOR[1] - 4]])
        return tx.compose(Rm[None], L[None])[0], lift

    def snapshot(self, T):
        """every visible tessera of the panel at T in WORLD units: (xy, ang, size, rgb, alpha) - the raw
        material of the endless schism (the dust)."""
        M, alpha, _, rock_mats = self.static_state(T)
        out = [(tx.apply(M, self.xy), self.ang + tx.mat_angle(M), self.size * tx.mat_scale(M), self.rgb, alpha)]
        for r, Rm in enumerate(rock_mats):
            Mu, lift = self.under_M(r, Rm, T)
            us = self.under[r]
            if lift > 0:
                Mr = np.repeat(Mu[None], us.n, 0)
                out.append((tx.apply(Mr, us.xy), us.ang + tx.mat_angle(Mr), us.size * tx.mat_scale(Mr),
                            us.static["rgb"], us.static["alpha"]))
        for i in range(len(PHILS)):
            sp = self.spr[i]
            colr = sp.colours(**self.hero_state(i, T)) if i in HEROES else sp.static
            Mi = np.repeat(self.phil_M(i, T)[None], sp.n, 0)
            out.append((tx.apply(Mi, sp.xy), sp.ang + tx.mat_angle(Mi), sp.size * tx.mat_scale(Mi), colr["rgb"],
                        colr["alpha"]))
        cat = [np.concatenate([o[k] for o in out]) for k in range(5)]
        keep = cat[4] > 0.05
        return tuple(c[keep] for c in cat)

    # ------------------------------------------------------------------ render
    def render(self, fc, img, V, T, light, portal=False, lam_extra=0.0):
        zoom = math.hypot(V[0, 0], V[1, 0])
        img[...] = VOID
        M, alpha, joint, rock_mats = self.static_state(T)
        rgb, gold = self.rgb, self.gold
        # rock undersides appear below the floating chunks
        lift = smootherstep(T_BREAK + 0.05, T_BREAK + 0.7, T)
        if lift > 0:
            b = []
            for r, Rm in enumerate(rock_mats):
                us = self.under[r]
                Mu, _ = self.under_M(r, Rm, T)
                tx.render_set(fc, img, V, us.xy, us.ang, us.size, us.tilt, us.jit, us.uid + r * 7919,
                              us.colour_fn(us.static), M=np.repeat(Mu[None], us.n, 0), light=light,
                              alpha=us.static["alpha"], lmax=2, bed=1.3, batch=b, lam_extra=lam_extra)
            tx.flush(fc, img, b, light, bed=1.3)
        tx.render_set(fc, img, V, self.xy, self.ang, self.size, self.lay.tilt, self.lay.jit, self.uid,
                      lambda lxy, lsize, pidx: (rgb[pidx], gold[pidx]), M=M, light=light, alpha=alpha, joint=joint,
                      lmax=3, lam_extra=lam_extra)
        # the philosophers
        b = []
        for i in range(len(PHILS)):
            sp = self.spr[i]
            colr = sp.colours(**self.hero_state(i, T)) if i in HEROES else sp.static
            Mi = self.phil_M(i, T)
            tx.render_set(fc, img, V, sp.xy, sp.ang, sp.size, sp.tilt, sp.jit, sp.uid, sp.colour_fn(colr),
                          M=np.repeat(Mi[None], sp.n, 0), light=light, alpha=colr["alpha"], bed=1.35, margin=10.0,
                          lmax=2, batch=b, lam_extra=lam_extra)
        # SVM ERGO SVM sets itself in the void beside CRACKPOT2
        p2, h2 = self.phil_pos(5, T)
        for j, ((w_, tw), sp) in enumerate(zip(SVM, self.words)):
            u = T - (tw - 0.05)
            if u <= 0 or T > 128.55:
                continue
            k = min(1.0, u / 0.18)
            sc = (1 + 0.25 * math.sin(k * math.pi)) * k
            pos = p2 + np.array([210.0 + (j - 1) * 0.0, -h2 * 0.95 + j * 95.0])
            ax, ay = sp.anchor
            Mw = np.array([[sc, 0, pos[0] - sc * ax], [0, sc, pos[1] - sc * ay]])
            tx.render_set(fc, img, V, sp.xy, sp.ang, sp.size, sp.tilt, sp.jit, sp.uid + j * 50000,
                          sp.colour_fn(sp.static), M=np.repeat(Mw[None], sp.n, 0), light=light,
                          alpha=sp.static["alpha"], lmax=1, bed=1.3, batch=b,
                          gain=np.full(sp.n, 1.0 + 0.8 * math.exp(-u / 0.2), np.float32))
        tx.flush(fc, img, b, light, bed=1.35)

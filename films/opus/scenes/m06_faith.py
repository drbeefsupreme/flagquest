"""m06 FAITHS (113.7-120.0): a congregation of saints under one great dome, ECCLESIA VNA. At K05 the dome cracks
down the middle (East / West), the halves split into cabals, the cabals into ones; the ragged pieces of the one
mosaic crumble away and every saint stands in an icon of his own: gold ground, frame, a dome on two columns, a
tiny altar. X03/X04: two of them fire ANATHEMA SIT at each other (letters of tesserae) and each other's halos
crack in two. X05: the third crowns himself; at "pope" every saint in the field pops a tiara.

World = panel design units (the frame at zoom 1 = the view at the end of the dive into it).
"""
import math

import cv2
import numpy as np

from vx import W, H, Canvas, set_color, circle, ellipse, poly, value_noise2, hash01, smootherstep, ease_out
from vx import icon
from vx import mosaic as mz
from vx.canvas import col, text

from scenes import m06_tess as tx

S = mz.SMALTI
FONT = "P052"
T_SPLIT = {1: 113.97, 2: 114.32, 3: 114.74}     # "faith", "cabal", "one"
T_ICON = 114.95                                 # the ragged pieces crumble; icons of one set themselves
T_ANA1 = (115.62, 116.36)                       # X03 launch -> hit
T_ANA2 = (116.72, 117.86)                       # X04 launch -> hit
T_CROWN = (117.8, 118.55)                       # X05 crown_self progress 0 -> 1 ("pope" 118.53)
T_TIARAS = 118.58                               # the wave of tiaras

# the congregation: front row 7, back row 6 (feet positions, heights)
FRONT = [(960 + (i - 3) * 250, 985.0, 380.0) for i in range(7)]
BACK = [(960 + (j - 2.5) * 250, 905.0, 365.0) for j in range(6)]
SAINTS = BACK + FRONT                                # painter's order: back row first
SEEDS = [21, 22, 23, 24, 25, 26, 11, 12, 3, 4, 7, 14, 16]
HEROES = {8: 1, 9: 2, 10: 3}                         # index in SAINTS -> hero id (H1, H2, H3)
FACING = {8: -1, 9: 1, 10: 0}
# the field of icons: back row above, front row below (feet)
GRID = [(960 + (j - 2.5) * 420, 430.0) for j in range(6)] + [(960 + (i - 3) * 420, 1190.0) for i in range(7)]
WORLD_Y0, WORLD_H = -420.0, 1920                     # the one church is laid on a square wall (the portal is square)

KH = 400.0                                          # figure height in sprite units
SPR_W, SPR_H = 560.0, 760.0
GROUND = (280.0, 700.0)


# ============================================================ the one church (static cartoon)
def draw_church(ctx, layer):
    colr = layer == "color"
    wht = (1, 1, 1)
    cx = 960.0
    if colr:
        set_color(ctx, S["gold"])
        ctx.rectangle(-400, -600, W + 800, H + 1200)
        ctx.fill()
    if layer == "gold":
        set_color(ctx, wht)
        ctx.rectangle(-400, -600, W + 800, H + 1200)
        ctx.fill()
    def pc(c):          # non-gold smalti: their own colour, black (= no leaf) in the gold layer
        set_color(ctx, c if colr else (0, 0, 0))

    # the great dome: lapis vault with rings of gold stars, a jewelled rim
    R, cy = 430.0, 472.0
    if colr:
        set_color(ctx, S["lapis_d"])
        ctx.arc(cx, cy, R, math.pi, 2 * math.pi)
        ctx.close_path()
        ctx.fill()
        for k, (rr, n) in enumerate(((360, 22), (290, 17), (215, 12), (140, 8))):
            for j in range(n):
                a = math.pi + math.pi * (j + 0.5) / n
                x, y = cx + rr * math.cos(a), cy + rr * math.sin(a)
                set_color(ctx, S["star"])
                circle(ctx, x, y, 7.5 - k)
                ctx.fill()
    if layer in ("gold", "color"):
        if layer == "gold":
            set_color(ctx, (0, 0, 0))            # the gold mask is read from the value channel
            ctx.arc(cx, cy, R, math.pi, 2 * math.pi)
            ctx.close_path()
            ctx.fill()
            for k, (rr, n) in enumerate(((360, 22), (290, 17), (215, 12), (140, 8))):
                for j in range(n):
                    a = math.pi + math.pi * (j + 0.5) / n
                    set_color(ctx, wht)
                    circle(ctx, cx + rr * math.cos(a), cy + rr * math.sin(a), 7.5 - k)
                    ctx.fill()
        set_color(ctx, S["gold_l"] if colr else wht)
        ctx.set_line_width(22)
        ctx.arc(cx, cy, R + 11, math.pi, 2 * math.pi)
        ctx.stroke()
    if layer in ("color", "gold"):
        pc(S["ink"])
        ctx.set_line_width(4)
        ctx.arc(cx, cy, R, math.pi, 2 * math.pi)
        ctx.stroke()
        ctx.arc(cx, cy, R + 22, math.pi, 2 * math.pi)
        ctx.stroke()
        # windows of the drum: arched lights
        for j in range(9):
            x = cx - 360 + j * 90
            pc(S["bone"])
            ctx.rectangle(x - 12, cy - 70, 24, 60)
            ctx.arc(x, cy - 70, 12, math.pi, 2 * math.pi)
            ctx.fill()
    # the cornice under the dome with the titulus ECCLESIA VNA
    if layer in ("color", "gold"):
        pc(S["porphyry"])
        ctx.rectangle(150, cy + 2, W - 300, 44)
        ctx.fill()
        pc(S["ink"])
        ctx.set_line_width(3)
        ctx.rectangle(150, cy + 2, W - 300, 44)
        ctx.stroke()
    if layer in ("color", "gold", "fine"):
        text(ctx, "ECCLESIA · VNA", cx, cy + 25, 34, S["gold_l"] if colr else wht, font=FONT, bold=True,
             align="center", valign="middle", tracking=0.18)
    # side columns and half domes
    for x in (120.0, W - 120.0):
        if layer in ("color", "gold"):
            pc(S["marble"])
            ctx.rectangle(x - 26, 360, 52, 640)
            ctx.fill()
            pc(S["marble_d"])
            ctx.rectangle(x + 6, 360, 16, 640)
            ctx.fill()
            pc(S["gold"])
            ctx.rectangle(x - 38, 330, 76, 34)
            ctx.fill()
            pc(S["ink"])
            ctx.set_line_width(3)
            ctx.rectangle(x - 26, 360, 52, 640)
            ctx.stroke()
            pc(S["lapis"])
            ctx.arc(x, 330, 110, math.pi, 2 * math.pi)
            ctx.close_path()
            ctx.fill()
            pc(S["ink"])
            ctx.arc(x, 330, 110, math.pi, 2 * math.pi)
            ctx.stroke()
    # the pavement (opus sectile: porphyry discs in marble)
    if layer in ("color", "gold"):
        pc(S["marble_d"])
        ctx.rectangle(-400, 985, W + 800, 700)
        ctx.fill()
        for row, yy in enumerate((1035, 1165, 1295, 1425)):
            for j in range(-2, 14):
                x = 60 + j * 160 + (80 if row % 2 else 0)
                pc(S["porphyry"])
                circle(ctx, x, yy, 30)
                ctx.fill()
                pc(S["verdigris"])
                circle(ctx, x + 80, yy, 16)
                ctx.fill()
        pc(S["ink"])
        ctx.set_line_width(4)
        ctx.move_to(-400, 985)
        ctx.line_to(W + 400, 985)
        ctx.stroke()


# ============================================================ a saint (and later his own church)
def _aedicula(ctx, layer, cx, ground, fig_top, open_=1.0):
    """two columns + a dome on a gold icon panel with a frame: the church of one."""
    colr = layer == "color"
    wht = (1, 1, 1)
    x0, x1 = cx - 205, cx + 205
    y1 = ground + 36
    ctop = fig_top - 20
    dome_r = 205.0
    y0 = ctop - dome_r
    # the icon panel (gold ground, arched top) with a red frame
    if layer in ("color", "gold"):
        set_color(ctx, S["gold"] if colr else wht)
        ctx.move_to(x0, y1)
        ctx.line_to(x0, ctop)
        ctx.arc(cx, ctop, dome_r, math.pi, 2 * math.pi)
        ctx.line_to(x1, y1)
        ctx.close_path()
        ctx.fill()
    if colr:
        set_color(ctx, S["porphyry"])
        ctx.set_line_width(22)
        ctx.move_to(x0, y1)
        ctx.line_to(x0, ctop)
        ctx.arc(cx, ctop, dome_r, math.pi, 2 * math.pi)
        ctx.line_to(x1, y1)
        ctx.close_path()
        ctx.stroke()
        # the dome of his own church: lapis cap with gold ribs over the head
        set_color(ctx, S["lapis"])
        ctx.arc(cx, ctop + 18, 150, math.pi, 2 * math.pi)
        ctx.close_path()
        ctx.fill()
        set_color(ctx, S["gold_l"])
        ctx.set_line_width(7)
        for k in range(5):
            a = math.pi + math.pi * (k + 1) / 6
            ctx.move_to(cx, ctop + 18)
            ctx.line_to(cx + 150 * math.cos(a), ctop + 18 + 150 * math.sin(a))
        ctx.stroke()
        set_color(ctx, S["ink"])
        ctx.set_line_width(5)
        ctx.arc(cx, ctop + 18, 150, math.pi, 2 * math.pi)
        ctx.stroke()
        # columns
        for x in (cx - 165, cx + 165):
            set_color(ctx, S["marble"])
            ctx.rectangle(x - 17, ctop + 18, 34, ground - ctop - 18)
            ctx.fill()
            set_color(ctx, S["marble_d"])
            ctx.rectangle(x + 4, ctop + 18, 10, ground - ctop - 18)
            ctx.fill()
            set_color(ctx, S["ink"])
            ctx.set_line_width(4)
            ctx.rectangle(x - 17, ctop + 18, 34, ground - ctop - 18)
            ctx.stroke()
        set_color(ctx, S["marble_d"])
        ctx.rectangle(x0 + 11, ground, x1 - x0 - 22, 25)
        ctx.fill()
    if layer in ("color", "gold", "fine"):
        icon.draw_altar(ctx, cx + 120, ground + 4, 105, layer=layer)


def saint_sprite(idx, ppu=0.9, tile=7.5, fine_tile=3.8):
    seed = SEEDS[idx]
    fig = icon.Figure("heretic", seed)
    hero = HEROES.get(idx, 0)
    facing = FACING.get(idx, 0)

    def draw(ctx, layer, church=0.0, tiara=False, pose="orans", t=0.0, mouth=0.0, expr="serene", look=(0.0, 0.0),
             progress=1.0, halo=True, blink=False):
        if church > 0.5:
            _aedicula(ctx, layer, GROUND[0], GROUND[1], GROUND[1] - KH - 60)
        kw = dict(pose=pose, t=t, mouth=mouth, look=look, expr=expr, facing=facing if pose == "point" else 0,
                  layer=layer, tile=tile, halo=halo and not hero, tiara=tiara, blink=blink)
        if pose == "crown_self":
            kw["progress"] = progress
        fig.draw(ctx, GROUND[0], GROUND[1], KH, **kw)
    poses = [dict(church=0.0), dict(church=1.0), dict(church=1.0, tiara=True)]
    if hero:
        poses += [dict(church=1.0, pose="point", expr="shouting"), dict(church=1.0, pose="crown_self", progress=0.3),
                  dict(church=1.0, pose="crown_self", progress=1.0)]
    return tx.Sprite(SPR_W, SPR_H, draw, tile=tile, fine_tile=fine_tile, seed=500 + idx, poses=poses, ppu=ppu,
                     anchor=GROUND)


# ============================================================ tituli of tesserae that fly: ANATHEMA SIT
def glyph_sprite(ch, size=78.0):
    w = size * 1.1

    def draw(ctx, layer, **st):
        if layer in ("color", "fine", "gold"):
            text(ctx, ch, w / 2, size * 0.92, size, S["porphyry"] if layer == "color" else (1, 1, 1), font=FONT,
                 bold=True, align="center", stroke=S["gold_l"] if layer == "color" else None,
                 stroke_w=size * 0.1 if layer == "color" else 0.0)
    return tx.Sprite(w, size * 1.2, draw, tile=5.5, fine_tile=5.5, seed=700 + ord(ch), ppu=2.0,
                     anchor=(w / 2, size * 0.6))


# ============================================================ halos that split in two
def halo_tiles(r=41.0, tile=5.0, seed=0):
    lay = mz.Layout.rings(0.0, 0.0, 0.0, r, tile=tile, seed=seed, extent=(1, 1))
    xy = lay.xy.astype(np.float64)
    rr = np.hypot(xy[:, 0], xy[:, 1])
    rgb = np.where((rr > r - tile * 1.2)[:, None], np.array(col(S["porphyry"]), np.float32),
                   np.array(col(S["gold"]), np.float32))
    gold = (rr <= r - tile * 1.2).astype(np.float32)
    return dict(xy=xy, ang=lay.ang.astype(np.float64), size=lay.size.astype(np.float64), tilt=lay.tilt, jit=lay.jit,
                rgb=rgb, gold=gold, side=np.sign(xy[:, 0] + 1e-6))


# ============================================================ the panel
class FaithPanel:
    def __init__(self, S_):
        self.tl = S_.tl
        # --- the one church
        class _F:
            s = 0.5
        def canvas(s, layer):
            cv = Canvas(s=s, w=int(W * s), h=int(WORLD_H * s))
            cv.ctx.translate(0, -WORLD_Y0)
            draw_church(cv.ctx, layer)
            return cv
        guide = canvas(0.5, "color").rgb()
        fine = canvas(0.5, "fine").rgba()[..., 3] > 0.5
        lay = mz.Layout.flow(guide, tile=12.0, seed=81, fine=fine, fine_tile=6.0, extent=(W, WORLD_H))
        self.lay = lay
        gold = canvas(1.0, "gold").rgb()[..., 0]
        self.pyr = tx.Pyramid(np.dstack([canvas(1.0, "color").rgb(), gold]), 0.0, WORLD_Y0, 1.0)
        self.xy = lay.xy.astype(np.float64) + np.array([0.0, WORLD_Y0])
        self.ang = lay.ang.astype(np.float64)
        self.size = lay.size.astype(np.float64)
        self.uid = np.arange(len(lay), dtype=np.int64) + 3 * 10 ** 7
        rgb, g = self.pyr.sample(self.xy[:, 0], self.xy[:, 1], self.size)
        self.rgb, self.gold = rgb, np.clip((g - 0.35) / 0.35, 0, 1)
        # every tile belongs to the saint nearest to it (noisy): that is the piece it will leave with
        sp = np.array([(x, y - 0.55 * h) for x, y, h in SAINTS])
        f = tx.noise_field(17, 90.0)
        d = np.linalg.norm(self.xy[:, None, :] - sp[None], axis=2) + 55.0 * f(self.xy[:, 0], self.xy[:, 1])[:, None]
        self.owner = np.argmin(d, axis=1)
        # tiles crumble away once the icons of one set themselves (wave from the outside in)
        rng = np.random.default_rng(83)
        self.fall_t = T_ICON + 0.05 + 0.35 * rng.random(len(self.xy))
        self.fall_v = np.stack([rng.normal(0, 60, len(self.xy)), rng.uniform(-160, 20, len(self.xy))], 1)
        self.fall_w = rng.normal(0, 5, len(self.xy))
        # --- the saints
        self.spr = [saint_sprite(i) if i not in HEROES else saint_sprite(i, ppu=1.8, tile=6.0, fine_tile=3.0)
                    for i in range(len(SAINTS))]
        self.static = {}
        for i, sp_ in enumerate(self.spr):
            if i not in HEROES:
                self.static[i] = [sp_.colours(church=0.0), sp_.colours(church=1.0), sp_.colours(church=1.0, tiara=True)]
        self.tiara_t = np.array([T_TIARAS + 0.06 * abs(GRID[i][0] - GRID[10][0]) / 330.0 + 0.1 * (GRID[i][1] < 800)
                                 for i in range(len(SAINTS))])
        self.icon_t = np.array([T_ICON + 0.05 + 0.04 * k for k in np.argsort(np.argsort(
            [abs(x - 960) for x, _, _ in SAINTS]))])
        # --- the flying letters and the split halos
        self.glyphs = {ch: glyph_sprite(ch) for ch in set("ANATHEMSI")}
        self.halo = {h: halo_tiles(seed=40 + h) for h in (1, 2, 3)}
        self._anchors = {i: icon.Figure("heretic", SEEDS[i]).anchors(GROUND[0], GROUND[1], KH, pose="orans")
                         for i in HEROES}
        # the portal: the gold knob on top of H3's self-made tiara (it holds the philosophers)
        sp3 = self.spr[10]
        a3 = icon.Figure("heretic", SEEDS[10]).anchors(GROUND[0], GROUND[1], KH, pose="crown_self", progress=1.0)
        tx_, ty_ = a3["top"]
        self.portal_tile = int(np.argmin(np.hypot(sp3.xy[:, 0] - tx_, sp3.xy[:, 1] - ty_ - 8.0)))

    def portal(self, T):
        sp = self.spr[10]
        j = self.portal_tile
        M = self.saint_M(10, T)
        c = M[:, :2] @ sp.xy[j] + M[:, 2]
        return c, float(sp.ang[j] + math.atan2(M[1, 0], M[0, 0])), float(sp.size[j] * math.sqrt(abs(np.linalg.det(M[:, :2]))))

    # ------------------------------------------------------------------ motion of the saints (pure f(T))
    def saint_pos(self, i, T):
        """feet position of saint i at time T: congregation -> halves -> cabals -> ones -> the field of icons."""
        x, y, h = SAINTS[i]
        gx, gy = GRID[i]
        half = -1 if x < 960 else 1
        cab = int((x + 125) // 500)
        d1 = np.array([half * 70.0, 0.0])
        d2 = np.array([(cab - 1.9) * 45.0, -25.0 + 50.0 * (i % 2)])
        d3 = np.array([gx, gy]) - np.array([x, y]) - d1 - d2
        p = np.array([x, y])
        for dd, tg in ((d1, T_SPLIT[1]), (d2, T_SPLIT[2]), (d3, T_SPLIT[3])):
            p = p + dd * float(tx.ease_jolt(np.array([T - tg]), 0.25, 0.35, 0.45)[0])
        spin = 0.0
        for g, amp in ((1, 0.06), (2, 0.09), (3, 0.12)):
            u = T - T_SPLIT[g]
            if u > 0:
                spin += amp * half * math.sin(min(u / 0.9, 1.0) * math.pi) * math.exp(-u)
        return p, h, spin

    def saint_M(self, i, T):
        """sprite-local -> panel-world affine of saint i."""
        p, h, spin = self.saint_pos(i, T)
        k = h / KH
        c, s = math.cos(spin) * k, math.sin(spin) * k
        return np.array([[c, -s, p[0] - (c * GROUND[0] - s * GROUND[1])], [s, c, p[1] - (s * GROUND[0] + c * GROUND[1])]])

    def hero_state(self, i, T):
        hid = HEROES[i]
        tl = self.tl
        spk = {1: "HERETIC1", 2: "HERETIC2", 3: "HERETIC3"}[hid]
        m = round(tl.mouth(spk, T) / 0.05) * 0.05
        talking = tl.speaking(spk, T, 0.15) is not None
        church = 1.0 if T >= self.icon_t[i] else 0.0
        tq = math.floor(T * 2) / 2
        bl = tx.blink(T, 10 + hid) and not talking
        if hid == 3:
            p = float(np.clip((T - T_CROWN[0]) / (T_CROWN[1] - T_CROWN[0]), 0, 1))
            if T >= T_CROWN[0]:
                return dict(church=church, pose="crown_self", progress=round(ease_out(p, 2), 2), mouth=m,
                            expr="proud" if not talking else "shouting", t=tq, blink=bl)
            return dict(church=church, pose="orans", mouth=m, expr="serene", t=tq, blink=bl)
        fire = T_ANA1 if hid == 1 else T_ANA2
        hit_by = T_ANA2[1] if hid == 1 else T_ANA1[1]
        if T >= fire[0] - 0.25 or talking:
            return dict(church=church, pose="point", mouth=m, expr="shouting" if talking else "angry", t=tq,
                        look=(1.0 if hid == 1 else -1.0, 0.0), blink=bl)
        if T >= hit_by:
            return dict(church=church, pose="orans", mouth=m, expr="angry", t=tq, blink=bl)
        return dict(church=church, pose="orans", mouth=m, expr="serene", t=tq, blink=bl)

    # ------------------------------------------------------------------ render
    def render(self, fc, img, V, T, light, portal=False, lam_extra=0.0):
        zoom = math.hypot(V[0, 0], V[1, 0])
        img[...] = mz.SC["mortar"] * 0.35
        # the one mosaic: pieces follow their saints until they crumble away
        n = len(self.xy)
        M = np.zeros((n, 2, 3))
        for i in range(len(SAINTS)):
            p, h, spin = self.saint_pos(i, T)
            x, y, _ = SAINTS[i]
            c, s = math.cos(spin), math.sin(spin)
            k = self.owner == i
            M[k] = np.array([[c, -s, p[0] - (c * x - s * y)], [s, c, p[1] - (s * x + c * y)]])
        tau = T - self.fall_t
        falling = tau > 0
        alpha = np.where(falling, np.clip(1 - tau / 0.9, 0, 1), 1.0).astype(np.float32)
        if falling.any():
            k = np.nonzero(falling)[0]
            tk = tau[k]
            off = self.fall_v[k] * tk[:, None] + np.array([0.0, 900.0]) * (tk ** 2)[:, None] * 0.5
            M[k, :, 2] += off
            th = self.fall_w[k] * tk
            c, s = np.cos(th), np.sin(th)
            A = np.zeros((len(k), 2, 3))
            A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] = c, -s, s, c
            cx, cy = self.xy[k, 0], self.xy[k, 1]
            A[:, 0, 2] = cx - (c * cx - s * cy)
            A[:, 1, 2] = cy - (s * cx + c * cy)
            M[k] = tx.compose(M[k], A)
        joint = None
        crack = np.zeros(n, np.float32)
        for g, tg in T_SPLIT.items():
            u = T - tg
            if -0.06 < u < 0.8:
                crack = np.maximum(crack, (np.exp(-max(u, 0) / 0.2) if u > 0 else 1 + u / 0.06))
        if crack.max() > 0:
            joint = (crack[:, None] * np.array([1.0, 0.86, 0.55], np.float32)[None] * 1.4).astype(np.float32)
        rgb, gold = self.rgb, self.gold
        tx.render_set(fc, img, V, self.xy, self.ang, self.size, self.lay.tilt, self.lay.jit, self.uid,
                      lambda lxy, lsize, pidx: (rgb[pidx], gold[pidx]), M=M, light=light, alpha=alpha, joint=joint,
                      lmax=3)
        # the saints (back row first): one render call per row
        rows = ([], [])
        over, after = [], []
        for i in range(len(SAINTS)):
            sp = self.spr[i]
            Mi = self.saint_M(i, T)
            if i in HEROES:
                colr = sp.colours(**self.hero_state(i, T))
            else:
                st = 2 if T >= self.tiara_t[i] else (1 if T >= self.icon_t[i] else 0)
                colr = self.static[i][st]
            pop = self._pop(i, T)
            Mi = Mi if pop is None else tx.compose(Mi[None], pop[None])[0]
            alpha = colr["alpha"]
            if portal and i == 10:
                alpha = alpha.copy()
                alpha[self.portal_tile] = 0.0
            Mrep = np.repeat(Mi[None], sp.n, axis=0)
            tx.render_set(fc, img, V, sp.xy, sp.ang, sp.size, sp.tilt, sp.jit, sp.uid, sp.colour_fn(colr),
                          M=Mrep, light=light, alpha=alpha, bed=1.35, margin=10.0, lmax=2,
                          batch=rows[0 if i < 6 else 1])
            if i in HEROES:
                self._render_halo(fc, img, V, T, light, i, Mi, over)
            if portal and i == 10:
                j = [self.portal_tile]
                tx.render_set(fc, img, V, sp.xy[j], sp.ang[j], sp.size[j], sp.tilt[j], sp.jit[j], sp.uid[j] + 1,
                              lambda lxy, lsize, pidx: (colr["rgb"][j][pidx], colr["gold"][j][pidx]), M=Mrep[j],
                              light=light, lmax=0, bed=1.2, margin=10.0, batch=after)
        tx.flush(fc, img, rows[0], light, bed=1.35)
        tx.flush(fc, img, rows[1], light, bed=1.35)
        self._render_anathema(fc, img, V, T, light, over)
        tx.flush(fc, img, over, light, bed=1.3)
        tx.flush(fc, img, after, light, bed=1.2)

    def _pop(self, i, T):
        """a little jump when the icon of one sets itself / the tiara lands (local scale about the feet)."""
        u = max(T - self.icon_t[i], -1)
        v = T - self.tiara_t[i] if i not in HEROES else T - T_CROWN[1]
        k = 1.0
        if 0 <= u < 0.3:
            k += 0.06 * math.sin(u / 0.3 * math.pi)
        if 0 <= v < 0.25:
            k += 0.05 * math.sin(v / 0.25 * math.pi)
        if k == 1.0:
            return None
        gx, gy = GROUND
        return np.array([[k, 0, gx - k * gx], [0, k, gy - k * gy]])

    def _halo_split(self, hid, T):
        hit = {1: T_ANA2[1], 2: T_ANA1[1], 3: np.inf}[hid]
        return float(np.clip((T - hit) / 0.35, 0, 1)), T - hit

    def _render_halo(self, fc, img, V, T, light, i, Mi, batch):
        hid = HEROES[i]
        h = self.halo[hid]
        a = self._anchors[i]
        hx, hy, hr = a["halo"]
        k = hr / 41.0
        sp_, u = self._halo_split(hid, T)
        if hid == 3 and T >= T_CROWN[0] + 0.3:
            return
        side = h["side"]
        # the two halves swing apart about the bottom point and drop a little
        ang = side * 0.2 * ease_out(sp_, 3)
        dx = side * 11.0 * ease_out(sp_, 2)
        dy = 7.0 * sp_ + 4.0 * sp_ * sp_
        px, py = 0.0, 41.0
        c, s = np.cos(ang), np.sin(ang)
        lx = h["xy"][:, 0] - px
        ly = h["xy"][:, 1] - py
        xy = np.stack([px + c * lx - s * ly + dx, py + s * lx + c * ly + dy], 1) * k + np.array([hx, hy])
        # the head (and shoulders) stand in front of the nimbus: no halo tesserae there
        hcx, hcy = a["head"]
        hr_ = a.get("head_r", hr * 0.62)
        hr_ = hr_[0] if isinstance(hr_, (tuple, list)) else hr_
        vis = ((xy[:, 0] - hcx) ** 2 + (xy[:, 1] - hcy) ** 2 > (hr_ * 1.08) ** 2) & (xy[:, 1] < hcy + hr_ * 0.9)
        Mw = np.repeat(Mi[None], len(xy), axis=0)
        joint = None
        if -0.05 < u < 0.6:
            e = np.exp(-max(u, 0) / 0.15) * (np.abs(h["xy"][:, 0]) < 8)
            joint = (e[:, None] * np.array([1.0, 0.9, 0.6])[None] * 2.0).astype(np.float32)
        tx.render_set(fc, img, V, xy, h["ang"] + ang, h["size"] * k, h["tilt"], h["jit"],
                      np.arange(len(xy), dtype=np.int64) + 4 * 10 ** 7 + hid * 10 ** 5,
                      lambda lxy, lsize, pidx: (h["rgb"][pidx], h["gold"][pidx]), M=Mw, light=light, joint=joint,
                      lmax=1, bed=1.3, alpha=vis.astype(np.float32), batch=batch)

    def _hand_world(self, i, T, which):
        """world position of saint i's pointing hand / halo centre."""
        st = self.hero_state(i, T)
        fig = icon.Figure("heretic", SEEDS[i])
        a = fig.anchors(GROUND[0], GROUND[1], KH, pose=st["pose"], t=T, facing=FACING.get(i, 0))
        p = a[which] if which != "halo" else a["halo"][:2]
        M = self.saint_M(i, T)
        return M[:, :2] @ np.array(p[:2]) + M[:, 2]

    def _render_anathema(self, fc, img, V, T, light, batch):
        for (t0, t1), src, dst in ((T_ANA1, 8, 9), (T_ANA2, 9, 8)):
            if not (t0 <= T <= t1 + 0.35):
                continue
            a = self._hand_world(src, T, "hand_r")
            b = self._hand_world(dst, T, "halo") + np.array([0.0, -30.0])
            word = "ANATHEMA SIT"
            n = len(word)
            rightward = b[0] > a[0]
            for j, ch in enumerate(word):
                if ch == " ":
                    continue
                # the titulus keeps its reading order in flight: the letter nearest the target leads
                rank = (n - 1 - j) if rightward else j
                lt0 = t0 + 0.05 * rank
                u = (T - lt0) / (t1 - t0 - 0.05 * n * 0.5)
                if u <= 0:
                    continue
                if u >= 1.0:
                    # shattered against the halo: the letter bursts into loose tesserae
                    v = T - (lt0 + (t1 - t0 - 0.05 * n * 0.5))
                    if v > 0.35:
                        continue
                    self._burst(fc, img, V, T, light, ch, b, v, j, batch)
                    continue
                q = u * u * (3 - 2 * u)
                mid = (a + b) / 2 + np.array([0.0, -120.0 - 20 * math.sin(j)])
                p = (1 - q) ** 2 * a + 2 * (1 - q) * q * mid + q * q * b
                p = p + np.array([0.0, (j - n / 2) * 2.0])
                sc = 0.55 + 0.45 * math.sin(min(1.0, u * 1.4) * math.pi / 2)
                gs = self.glyphs[ch]
                th = 0.25 * math.sin(T * 9 + j)
                c, s = math.cos(th) * sc, math.sin(th) * sc
                ax, ay = gs.anchor
                Mg = np.array([[c, -s, p[0] - (c * ax - s * ay)], [s, c, p[1] - (s * ax + c * ay)]])
                colr = gs.static
                tx.render_set(fc, img, V, gs.xy, gs.ang, gs.size, gs.tilt, gs.jit, gs.uid + j * 1000,
                              gs.colour_fn(colr), M=np.repeat(Mg[None], gs.n, axis=0), light=light,
                              alpha=colr["alpha"], lmax=1, bed=1.3,
                              gain=np.full(gs.n, 1.25, np.float32), batch=batch)

    def _burst(self, fc, img, V, T, light, ch, b, v, j, batch):
        gs = self.glyphs[ch]
        rng = np.random.default_rng(900 + j)
        n = gs.n
        d = rng.normal(0, 1, (n, 2)) * 180.0 + np.array([0, -60.0])
        p = b + (gs.xy - gs.anchor) * 0.6 + d * v + np.array([0, 700.0]) * v * v * 0.5
        sxy = np.stack([V[0, 0] * p[:, 0] + V[0, 1] * p[:, 1] + V[0, 2], V[1, 0] * p[:, 0] + V[1, 1] * p[:, 1] + V[1, 2]], 1)
        zoom = math.hypot(V[0, 0], V[1, 0])
        colr = gs.static
        batch.append(dict(xy=sxy, ang=gs.ang + rng.normal(0, 3, n) * v, size=gs.size * 0.6 * zoom,
                          rgb=colr["rgb"], gold=colr["gold"], tilt=gs.tilt, jit=gs.jit,
                          alpha=(np.clip(1 - v / 0.35, 0, 1) * colr["alpha"]).astype(np.float32),
                          joint=np.zeros((n, 3), np.float32), ids=gs.uid + 777 * (j + 1),
                          gain=np.full(n, 1.3, np.float32)))

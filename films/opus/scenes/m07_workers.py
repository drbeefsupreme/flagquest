"""m07 helper: the labourers of Babel - tiny mosaic figures (after the Monreale / San Marco Babel mosaics) that climb
the ladders with glowing screen-bricks on their shoulders, tread the crane's wheel, lay bricks on the terraces, carry
bricks out of the cities, and - at the far left and right - dangle as marionettes from Egregore's cables.

Each labourer is a SPRITE: a fixed local tile layout (the union of all its poses) whose tiles are re-coloured per pose
(pre-sampled per pose in setup, so a frame only indexes tables) and placed on the wall slightly lifted (loose tiles).
When the wall falls, each rides the tessera it stands on (its 'anchor' in the Fall) and vanishes into the heap.
All pure functions of T.
"""
import math

import cairo
import numpy as np

from vx import mosaic as mz
from vx.canvas import Canvas

from . import m07_art as A

EXT = (50.0, 70.0)
FEET = (25.0, 66.0)
TILE = 3.6
TUNICS = ["porphyry_l", "lapis_l", "verdigris", "terracotta", "slate", "ochre_red", "marble_d", "tyrian"]
BRICKS = ["slop_flesh", "slop_cyan", "screen_glow", "slop_beige"]


def _sprite_rgba(pose, tunic, brick, s=4.0):
    cv = Canvas(s=s, w=int(EXT[0] * s), h=int(EXT[1] * s))
    c = cv.ctx
    c.translate(*FEET)
    A.draw_worker(c, pose, tunic, "flesh", brick)
    return cv.rgba()


class Workers:
    def __init__(self, ruin, tier_t, t_crown, seed=77):
        rng = np.random.default_rng(seed)
        self.r = ruin
        # ---- the sprite layout: union of every pose's silhouette (with a brick on the shoulder)
        alphas = [_sprite_rgba(p, "slate", "slop_cyan")[..., 3] for p in A.WORKER_POSES]
        a = np.max(alphas, axis=0)
        guide = _sprite_rgba("stand", "slate", "slop_cyan")
        rgb = guide[..., :3] / np.maximum(guide[..., 3:], 1e-3)
        self.lay = mz.Layout.flow(np.clip(rgb, 0, 1), tile=TILE, seed=5, extent=EXT, alpha=a, fine=a > 0.5,
                                  fine_tile=2.6)
        lay = self.lay
        # ---- per (pose, tunic, brick) tile colours + coverage + emissive brick mask
        self.tab = {}
        for pose in A.WORKER_POSES:
            for ti, tu in enumerate(TUNICS):
                for bi, br in enumerate([None] + BRICKS):
                    rgba = _sprite_rgba(pose, tu, br)
                    al = rgba[..., 3]
                    col = mz.sample(np.clip(rgba[..., :3] / np.maximum(al[..., None], 1e-3), 0, 1), lay)
                    cov = mz.sample_mask(al, lay)
                    self.tab[(pose, ti, bi)] = (col.astype(np.float32), (cov > 0.5).astype(np.float32))
        # brick tiles glow: tiles inside the brick rectangle (local), same for every pose that carries
        bx0, by0 = FEET[0] - 16, FEET[1] - 54
        self.brick_m = ((lay.xy[:, 0] > bx0 - 1) & (lay.xy[:, 0] < bx0 + 14) & (lay.xy[:, 1] > by0 - 1) &
                        (lay.xy[:, 1] < by0 + 10)).astype(np.float32)
        self.loc = (lay.xy - np.array(FEET, np.float32)).astype(np.float32)
        self.ploc = (lay.poly - np.array(FEET, np.float32)).astype(np.float32)
        # ---- the cast
        cast = []
        lad = A.ladders()
        for (foot, top, k) in lad:
            for j in range(2 if k < 4 else 1):
                cast.append(dict(kind="ladder", foot=foot, top=top, t_on=tier_t[k + 1] + 0.05, per=4.2 + 0.5 * j,
                                 ph=rng.uniform(0, 4), tu=rng.integers(len(TUNICS)), br=1 + rng.integers(len(BRICKS))))
        wx, wy, wr = A.CRANE["wheel"]
        cast.append(dict(kind="wheel", x=wx, y=wy + wr - 5, t_on=t_crown + 0.1, tu=0, br=0))
        tw = A.tiers()
        for k in range(6):
            xc, hw, yb, yt = tw[k]
            xn, hn, _, _ = tw[k + 1]
            for side in (-1, 1):
                if rng.uniform() < 0.35:
                    continue
                x = xn + side * (hn + (hw - hn) * 0.55)
                cast.append(dict(kind="terrace", x=x, y=yt + 2, t_on=tier_t[k + 1] + 0.1, ph=rng.uniform(0, 3),
                                 tu=rng.integers(len(TUNICS)), br=1 + rng.integers(len(BRICKS)), face=-side))
        for side in (-1, 1):
            for j in range(3):
                cast.append(dict(kind="carry", side=side, lane=j, y=A.H(110.0) + 4 + 3 * j, t_on=132.4 + 0.3 * j,
                                 speed=42 + 9 * j, ph=rng.uniform(0, 40), tu=rng.integers(len(TUNICS)),
                                 br=1 + rng.integers(len(BRICKS))))
        for (px, ph) in A.PUPPETS:
            cast.append(dict(kind="puppet", x=px, y=A.H(110.0) + 2, t_on=135.2, ph=rng.uniform(0, 1),
                             tu=rng.integers(len(TUNICS)), br=0))
        self.cast = cast
        # ---- anchors: the wall tessera under each labourer's feet at the collapse
        f = ruin.fall
        from scipy.spatial import cKDTree
        tree = cKDTree(ruin.lay.xy)
        for w in cast:
            x, y = self._home(w, 141.7)
            d, i = tree.query([x, y - 4])
            w["anchor"] = int(i)
        self.n_spr = len(lay)

    # ---------------------------------------------------------------- where + how each labourer is at T (no fall)
    def _home(self, w, T):
        k = w["kind"]
        if k == "ladder":
            (x0, y0), (x1, y1) = w["foot"], w["top"]
            u = ((T + w["ph"]) % w["per"]) / w["per"]
            if u < 0.5:                   # up, brick on the shoulder
                s = u / 0.5
            elif u < 0.58:                # at the top: hands the brick over
                s = 1.0
            else:                         # down again
                s = 1 - (u - 0.58) / 0.42
            s = min(max(s, 0), 1) * 0.86
            return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s
        if k in ("wheel", "terrace", "puppet"):
            return w["x"], w["y"]
        if k == "carry":
            side = w["side"]
            x_out, x_in = A.AX + side * 1300, A.AX + side * 470
            L = abs(x_out - x_in)
            d = ((T - 131.5) * w["speed"] + w["ph"]) % L
            return x_out + (x_in - x_out) * d / L, w["y"]
        return 0.0, 0.0

    def _pose(self, w, T):
        k = w["kind"]
        if k == "ladder":
            u = ((T + w["ph"]) % w["per"]) / w["per"]
            step = int(T * 6 + w["ph"] * 10) % 2
            pose = "climb_a" if step else "climb_b"
            brick = w["br"] if u < 0.52 else 0
            return pose, brick, 1.0
        if k == "wheel":
            return ("wheel_a" if int(T * 5) % 2 else "wheel_b"), 0, 1.0
        if k == "terrace":
            step = int((T + w["ph"]) * 2.2) % 3
            return ("carry_a", "stand", "carry_b")[step], (w["br"] if step != 1 else 0), float(w["face"])
        if k == "carry":
            step = int(T * 5 + w["ph"]) % 2
            return ("carry_a" if step else "carry_b"), w["br"], (1.0 if w["side"] < 0 else -1.0)
        if k == "puppet":
            # jerked by the strings: poses switch on an uneven beat
            b = int((T + w["ph"]) * 5.5)
            return ("puppet_a" if (b * 7 + 3) % 5 < 2 else "puppet_b"), 0, 1.0
        return "stand", 0, 1.0

    # ---------------------------------------------------------------- the frame
    def at(self, T, static_level):
        f = self.r.fall
        out = []
        for wi, w in enumerate(self.cast):
            if T < w["t_on"]:
                continue
            a_on = min(1.0, (T - w["t_on"]) / 0.2)
            x, y = self._home(w, T)
            pose, br, flip = self._pose(w, T)
            z = 1.5
            tilt = 0.0
            alpha = a_on
            # marionettes: when the static drains their strings go slack - they crumple where they hang
            if w["kind"] == "puppet" and T > 141.85:
                k = min(1.0, (T - 141.85 - 0.1 * w["ph"]) / 0.35)
                if k > 0:
                    pose = "puppet_b"
                    tilt = k * (1.35 if w["ph"] > 0.5 else -1.35)
                    y += 6 * k
            ia = w["anchor"]
            if T >= f.td[ia]:
                # carried by the tessera beneath the feet, then gone into the heap
                p = _anchor_pos(f, ia, T)
                x, y, z = p[0], p[1], max(p[2], 1.5)
                tilt += (T - f.td[ia]) * (2.0 if ia % 2 else -2.0)
                alpha *= max(0.0, 1.0 - max(0.0, T - f.tl[ia]) / 0.25)
            if alpha <= 0.01:
                continue
            col, cov = self.tab[(pose, int(w["tu"]), int(br))]
            c, s_ = math.cos(tilt), math.sin(tilt)
            R2 = np.array([[c * flip, -s_], [s_ * flip, c]], np.float32)
            loc = self.loc @ R2.T
            ploc = self.ploc @ R2.T
            xy = loc + np.array([x, y], np.float32)
            poly = ploc + np.array([x, y], np.float32)
            emit = np.zeros((len(col), 3), np.float32)
            if br:
                emit += (self.brick_m[:, None] * mz.srgb_to_lin(col) * 1.4)
            out.append(dict(xy=xy, poly=poly, nv=self.lay.nv, size=self.lay.size, ang=self.lay.ang, rgb=col,
                            mat=np.zeros(len(col), np.uint8), z=np.full(len(col), z, np.float32),
                            alpha=(cov * alpha).astype(np.float32), emit=emit, tilt=self.lay.tilt,
                            jit=self.lay.jit, rnd=self.lay.rnd, ids=400000 + wi * 1000 + np.arange(len(col))))
        return out


def _anchor_pos(f, i, T):
    """position of Fall tile i at time T (scalar version of Fall.at for one tile)"""
    if T < f.tb[i]:
        p, _, _ = f._sheet(T, np.array([i]))
        return p[0]
    if T < f.tl[i]:
        tau = T - f.tb[i]
        return f.pb[i] + f.vb[i] * tau + 0.5 * np.array([0.0, f.g, 0.0]) * tau * tau
    if T < f.tr[i]:
        s = (T - f.tl[i]) / max(1e-6, f.tr[i] - f.tl[i])
        k = 1 - (1 - s) ** 2.2
        return f.pl[i] + (f.pr[i] - f.pl[i]) * k
    return f.pr[i]

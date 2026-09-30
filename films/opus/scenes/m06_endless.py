"""m06 ENDLESS (128.6-131.5): K07 "May it schism endlessly." Every tessera in frame splits into four, and those
into four, and those into four... (a recursive quadtree, generation by generation on the words) until the
picture is glittering dust, which the dark begins to swirl (counter-clockwise, rising) for m07's Tower of Babel.

Generation 1 is drawn as real tiles (m06_tess LOD with lam_extra); from generation 2 the tesserae are specks:
every base tile's 64 descendants live at their quadtree offsets, separate, scatter and swirl - a pure f(T).
The state at 131.5 is exported for m07 (films/opus/cache/m06/handoff_dust.npz).
"""
import math

import numpy as np

from vx import col, smootherstep
from scenes import m06_tess as tx

T_GEN = (128.9, 129.22, 129.45, 129.65, 129.82)   # "schism" .. "endlessly": generations 1..5
T_DUST = T_GEN[1]                                 # from here: specks
T_SWIRL = (129.55, 130.6)                         # the vortex spins up
T_END = 131.5
CEN = np.array([960.0, 560.0])
BG_END = np.array(col("#0B0E1F"), np.float32)     # the ground m07 opens on
MAX_SPECKS = 700_000
N_HANDOFF = 26_000                                # specks that continue into m07 (the rest settle out first)
T_THIN = (130.35, 131.45)                         # everything not handed on fades out: the cut is seamless


def lam_extra(T):
    """extra quadtree generations forced on the tiles (generation 1 as tiles, before the dust takes over)."""
    return float(np.clip((T - T_GEN[0]) / 0.1, 0, 1))


def _offsets():
    """unit offsets (in parent-tile sizes) of the 64 great-grandchildren, per depth (3, 64, 2)."""
    k = np.arange(64)
    o = np.zeros((3, 64, 2))
    for d in range(3):
        q = (k >> (2 * (2 - d))) & 3
        o[d, :, 0] = ((q & 1) * 2 - 1) / 2.0 ** (d + 2)
        o[d, :, 1] = ((q >> 1) * 2 - 1) / 2.0 ** (d + 2)
    return o


class Dust:
    def __init__(self, panel, V, seed=606):
        """panel: the panel in view at T_DUST (its snapshot is the raw material); V: its static world->screen view."""
        xy, ang, size, rgb, alpha = panel.snapshot(T_DUST)
        zoom = math.hypot(V[0, 0], V[1, 0])
        vrot = math.atan2(V[1, 0], V[0, 0])
        sxy = xy @ V[:, :2].T + V[:, 2]
        ssize = size * zoom
        vis = (sxy[:, 0] > -40) & (sxy[:, 0] < 1960) & (sxy[:, 1] > -40) & (sxy[:, 1] < 1120) & (ssize > 0.8)
        sxy, ang, ssize, rgb, alpha = sxy[vis], ang[vis] + vrot, ssize[vis], rgb[vis], alpha[vis]
        rng = np.random.default_rng(seed)
        nb = len(sxy)
        per = 64 if nb * 64 <= MAX_SPECKS else 16
        o = _offsets()
        if per == 16:
            o = o[:, ::4]
            o[2] = 0.0
        ca, sa = np.cos(ang), np.sin(ang)
        # every speck: its base tile, its quadrant offsets per depth (rotated into the tile's frame, in px)
        self.base = np.repeat(np.arange(nb), per)
        self.n = len(self.base)
        bo = np.tile(np.arange(per), nb)
        rot = lambda v: np.stack([ca[self.base] * v[:, 0] - sa[self.base] * v[:, 1],
                                  sa[self.base] * v[:, 0] + ca[self.base] * v[:, 1]], 1) * ssize[self.base][:, None]
        self.off = [rot(o[d][bo]).astype(np.float32) for d in range(3)]
        self.p0 = sxy[self.base].astype(np.float32)
        self.size0 = ssize[self.base].astype(np.float32)
        # colours drift apart generation by generation: no two tesserae show the same world
        jit = rng.normal(0, 1, (self.n, 3)).astype(np.float32)
        base_rgb = rgb[self.base].astype(np.float32)
        self.rgb = np.clip(base_rgb * (1 + 0.1 * jit[:, :1]) + 0.05 * jit, 0, 1)
        self.alpha = alpha[self.base].astype(np.float32)
        self.vel = (rng.normal(0, 1, (self.n, 2)) * rng.uniform(35, 170, (self.n, 1))).astype(np.float32)
        self.tw_w = rng.uniform(5, 19, self.n).astype(np.float32)
        self.tw_p = rng.uniform(0, 2 * np.pi, self.n).astype(np.float32)
        self.sep = rng.uniform(0.4, 1.5, self.n).astype(np.float32)
        r = np.linalg.norm(self.p0 - CEN, axis=1)
        self.omega = (1.6 * 320.0 / (r + 320.0) + 0.3).astype(np.float32)       # rad/s at full spin
        self.rise = rng.uniform(15, 60, self.n).astype(np.float32)
        # a few specks are gold leaf: they flash gold as they catch the light
        self.gleam = (rng.random(self.n) < 0.12).astype(np.float32)
        # the specks handed on to m07: chosen once (brighter / bigger preferred); the others fade out before the cut
        self.keep = None
        p, size, rgb, alpha = self.state(T_END)
        on = (p[:, 0] > -20) & (p[:, 0] < 1940) & (p[:, 1] > -20) & (p[:, 1] < 1100) & (alpha > 0.3)
        idx = np.nonzero(on)[0]
        if len(idx) > N_HANDOFF:
            w = (rgb[idx].mean(1) + 0.2) * np.maximum(size[idx], 1.0)
            idx = np.random.default_rng(seed + 1).choice(idx, N_HANDOFF, replace=False, p=w / w.sum())
        self.keep = np.zeros(self.n, bool)
        self.keep[idx] = True

    # ------------------------------------------------------------------ pure f(T)
    def state(self, T):
        """speck positions (screen design px), sizes (px), colours (as displayed), alpha."""
        g = [float(np.clip((T - tg) / 0.09, 0, 1)) for tg in T_GEN]
        # quadtree: depth-d siblings coincide until generation d+1, then part (and keep drifting apart)
        p = self.p0.copy()
        for d in range(3):
            k = g[d + 1] * (1.0 + (self.sep * 2.6 * (d + 1)) * smootherstep(T_GEN[d + 1], T_GEN[d + 1] + 0.8, T))
            p += self.off[d] * k[:, None]
        gens = 1 + g[1] + g[2] + g[3] + g[4]
        size = self.size0 / 2.0 ** gens * 0.9
        size = np.maximum(size, 1.1 + 0.9 * (self.sep - 0.4))
        # scatter: dust drifts
        u = max(0.0, T - T_GEN[2])
        p += self.vel * u
        # the vortex: counter-clockwise on screen, rising, drawing slightly inward
        a = self._spin(T)
        if a > 0:
            d = p - CEN
            th = -self.omega * a
            c, s = np.cos(th), np.sin(th)
            pull = 1.0 - 0.06 * a
            p = CEN + np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1) * pull
            p[:, 1] -= self.rise * a
        tw = np.maximum(0.0, np.sin(self.tw_w * T + self.tw_p)) ** 10
        glitter = smootherstep(T_GEN[1], T_GEN[3], T)
        base = self.rgb * (1.0 + 0.35 * glitter)
        gold = np.array([1.0, 0.78, 0.42], np.float32)
        flash = (tw * glitter)[:, None]
        rgb = base + flash * (1.6 * base + 1.4 * gold * self.gleam[:, None])
        alpha = self.alpha
        if self.keep is not None and T > T_THIN[0]:
            alpha = np.where(self.keep, alpha, alpha * (1.0 - smootherstep(T_THIN[0], T_THIN[1], T)))
        return p.astype(np.float32), size.astype(np.float32), rgb.astype(np.float32), alpha.astype(np.float32)

    @staticmethod
    def _spin(T):
        """integrated spin ramp (seconds at full speed)."""
        t0, t1 = T_SWIRL
        if T <= t0:
            return 0.0
        tr = t1 - t0
        u = min((T - t0) / tr, 1.0)
        a = tr * (u ** 3 - 0.5 * u ** 4)
        return a + max(0.0, T - t1)

    def render(self, fc, img, T):
        p, size, rgb, alpha = self.state(T)
        m = (p[:, 0] > -8) & (p[:, 0] < 1928) & (p[:, 1] > -8) & (p[:, 1] < 1088)
        tx.specks(fc, img, p[m], size[m], rgb[m], alpha[m])
        return img

    def background(self, T, void):
        k = smootherstep(T_GEN[2], 131.0, T)
        return void * (1 - k) + BG_END * k

    def export_handoff(self, path):
        """the dust as it is at 131.5 (one frame past m06's last): exactly the specks still visible then."""
        T = T_END
        p, size, rgb, alpha = self.state(T)
        p1, _, _, _ = self.state(T - 1.0 / 24)
        vel = (p - p1) * 24.0
        idx = np.nonzero(self.keep)[0]
        np.savez_compressed(path, T=np.float64(T), xy=p[idx].astype(np.float32), vel=vel[idx].astype(np.float32),
                            size=size[idx].astype(np.float32), rgb=np.clip(rgb[idx], 0, 1).astype(np.float32),
                            bg=BG_END, center=CEN.astype(np.float32),
                            note=np.array("m06 dust at T=131.5 (screen design px, px/s); swirl CCW about center, "
                                          "rising"))

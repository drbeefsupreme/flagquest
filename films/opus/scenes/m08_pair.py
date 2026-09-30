"""m08 helper - Crocus and Lutie (vx.icon cartoons set in tesserae on a plane at their own depth).

Each actor carries a FIXED local tile layout (built once from the union of every pose it takes), placed in the
world on an upright plane through its ground point (they stand on the rubble: m08_stage.ground_y); per frame only
the cartoon changes (figures move by their tiles re-colouring). The local chart is K times the world scale, so the
tesserae are sized for the close-up (tile ~9 chart units, joints thin) and the plane maps them back down (Plane ex,
ey = 1/K). Crocus grips the Flag's pole: part='back' is set before the Flag, part='front' (the fists) after it,
masked by the cloth's coverage so nothing ever lies on the cloth.
"""
import math

import numpy as np

from vx import icon
from vx import mosaic as mz
from vx.ease import lerp, smoothstep, smootherstep

from scenes import m08_stage as st

S_LOCAL = 0.8            # cartoon pixels per chart unit (sampling only; tiles are rasterised at screen resolution)


class Actor:
    def __init__(self, kind, h, z, tile, fine_tile, K=2.6, seed=0):
        self.kind = kind
        self.K = float(K)
        self.fig = icon.Figure(kind, seed=seed, tile=tile)
        self.h, self.z = float(h), float(z)
        self.hc = self.h * self.K
        x0, y0, x1, y1 = self.fig.box(self.hc)
        pad = 14.0 * self.K
        self.box = (x0 - pad, y0 - pad - 40.0 * self.K, x1 + pad, y1 + pad)   # chart units about the ground point
        self.ext = (int(math.ceil(self.box[2] - self.box[0])), int(math.ceil(self.box[3] - self.box[1])))
        self.tile, self.fine = tile, fine_tile
        self.lay = None

    def _draw_fn(self, kw, part, yg):
        gx, gy = -self.box[0], -self.box[1]
        kw = dict(kw)
        if "ground_y" in kw:                                  # world y -> chart y about the ground point
            kw["ground_y"] = gy + (kw["ground_y"] - yg) * self.K

        def fn(ctx, layer):
            self.fig.draw(ctx, gx, gy, self.hc, layer=layer, part=part, tile=self.tile, **kw)
        return fn

    def cartoon(self, kw, part="all", layers=("color", "gold"), yg=st.FLOOR):
        return icon.cartoon(self._draw_fn(kw, part, yg), s=S_LOCAL, extent=self.ext, layers=layers)

    def build(self, path, poses, yg=st.FLOOR):
        """fixed local layout from the first pose, laid wherever ANY pose puts the figure (union of alphas)"""
        if path.exists():
            self.lay = mz.Layout.load(path)
            return
        main = self.cartoon(poses[0], layers=("color", "fine", "gold", "edges"), yg=yg)
        alpha = main["alpha"].copy()
        for kw in poses[1:]:
            alpha = np.maximum(alpha, self.cartoon(kw, layers=("color",), yg=yg)["alpha"])
        self.lay = mz.Layout.flow(main["rgb"], tile=self.tile, seed=13, extent=self.ext, fine=main["fine"] > 0.5,
                                  fine_tile=self.fine, gold=main["gold"] > 0.5, alpha=alpha,
                                  edges=main["edges"] > 0.5)
        self.lay.save(path)

    def anchors(self, xg, yg, kw):
        """world anchors (vx.icon at the world height h)"""
        return self.fig.anchors(xg, yg, self.h, **kw)

    def render(self, fc, view, xg, yg, kw, lights, ambient, part="all", env=1.0):
        """premultiplied rgba (fc.h, fc.w, 4) of the figure standing at world ground point (xg, yg, self.z)"""
        c = self.cartoon(kw, part, layers=("color", "gold"), yg=yg)
        k = 1.0 / self.K
        plane = mz.Plane(origin=(xg + self.box[0] * k, yg + self.box[1] * k, self.z), ex=(k, 0.0, 0.0),
                         ey=(0.0, k, 0.0))
        return mz.render(fc, c["rgb"], self.lay, gold=c["gold"], alpha=c["alpha"], view=view, surface=plane,
                         light=lights, ambient=ambient, return_alpha=True, env=env)


# ------------------------------------------------------------------ Crocus: where he is and what he does
def crocus_kw(T, tl=None):
    """(ground x, pose kwargs) - walk in carrying the furled Flag, plant it, step aside, speak, look up"""
    mouth = tl.mouth("CROCUS", T) if tl is not None else 0.0
    t_turn = st.T_WALK[1]
    if T < t_turn:
        return st.crocus_x(T), dict(pose="march", arms="hold_pole", facing=0.8, t=T, pole_ang=0.0)
    p = st.plant_progress(T)
    if p <= 0.0:
        k = smoothstep(t_turn, st.T_PLANT0, T)
        return st.CROCUS_PLANT_X, dict(pose={"hold_pole": 1.0}, facing=lerp(0.8, -0.25, k), t=T, look=(0.3, 0.1))
    if p < 1.0:
        return st.CROCUS_PLANT_X, dict(pose="plant", progress=p, facing=-0.25, t=T, ground_y=st.PLANT[1],
                                       expr="stern", look=(0.2, 0.3))
    # step aside so the planted pole stands at his right hand; speak; then look up in awe
    k = smootherstep(st.T_PLANT0 + st.PLANT_DUR, st.T_PLANT0 + st.PLANT_DUR + 0.5, T)
    x = lerp(st.CROCUS_PLANT_X, st.CROCUS_REST_X, k)
    up = smoothstep(154.95, 155.6, T)
    if up <= 0.0:
        bless = smoothstep(150.2, 150.8, T) * (1 - smoothstep(154.4, 154.95, T))
        look = (-0.55, 0.0) if T < 152.9 else (0.0, 0.0)
        kw = dict(pose={"plant": 1 - k, "rest_pole": max(k, 1e-3)} if k < 1 else "rest_pole", facing=-0.25, t=T,
                  ground_y=st.PLANT[1], mouth=mouth, look=look, expr="serene", progress=1.0)
        if bless > 0.5:
            kw["arm_l"] = "bless"
        return x, kw
    return x, dict(pose={"rest_pole": 1 - up, "look_up": max(up, 1e-3)}, facing=-0.25, t=T, ground_y=st.PLANT[1],
                   expr={"awe": up, "serene": 1 - up}, look=(0.1, -0.9 * up), mouth=mouth)


def lutie_kw(T):
    t1 = st.T_WALK[1] + 0.35
    if T < t1:
        return st.lutie_x(T), dict(pose="march", facing=0.8, t=T)
    up = smoothstep(155.1, 155.8, T)
    if up <= 0:
        turn = smoothstep(t1, t1 + 0.4, T)
        look = (0.8, -0.2) if T < 150.4 else (0.6, -0.1)
        if st.T_PLANT <= T < st.T_PLANT + 0.35:
            look = (0.9, 0.2)
        return st.lutie_x(T), dict(pose="clasp", facing=lerp(0.8, 0.3, turn), t=T, look=look,
                                   expr="worried" if T < 149.6 else "serene")
    return st.lutie_x(T), dict(pose={"clasp": 1 - up, "look_up": max(up, 1e-3)}, facing=0.3, t=T,
                               look=(0.2, -0.9 * up), expr={"awe": up, "serene": 1 - up})


CROCUS_POSES = [dict(pose="rest_pole", facing=-0.25, ground_y=st.PLANT[1]),
                dict(pose="march", arms="hold_pole", facing=0.8, t=0.0),
                dict(pose="march", arms="hold_pole", facing=0.8, t=0.4),
                dict(pose="plant", progress=0.3, facing=-0.25, ground_y=st.PLANT[1]),
                dict(pose="plant", progress=0.45, facing=-0.25, ground_y=st.PLANT[1]),
                dict(pose="look_up", facing=-0.25),
                dict(pose="rest_pole", facing=-0.25, ground_y=st.PLANT[1], arm_l="bless")]
LUTIE_POSES = [dict(pose="clasp", facing=0.3),
               dict(pose="march", facing=0.8, t=0.0),
               dict(pose="march", facing=0.8, t=0.4),
               dict(pose="look_up", facing=0.3)]

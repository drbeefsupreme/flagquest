"""Placeholder scene used until the scene owner replaces scenes/sXX_*.py. Shows title, dialogue, a Flag."""
import math

from vx import *
from vx.flag import Flag
from vx.chars import Character


def make(sid):
    def setup(S):
        fl = Flag(width=180, height=120, pole=460, seed=3)
        tr = fl.simulate(S.n, lambda i: (W * 0.5, H * 0.86, 0.04 * math.sin(i / 30)),
                         lambda i: (420 + 200 * math.sin(i / 40), 0))
        return {"flag": tr, "hero": Character("lutie")}

    def render(fc, st):
        cv = Canvas(fc, bg="night2")
        ctx = cv.ctx
        sc = fc.S
        text(ctx, f"{sc.id.upper()}  {fc.tl.scene(sc.id)['title']}", 60, 90, 44, "paper", bold=True)
        text(ctx, f"T={fc.T:6.2f}s  t={fc.t:5.2f}s", 60, 140, 26, "fog", font=FONT_MONO)
        st["flag"].draw(ctx, fc.f)
        st["hero"].draw(ctx, W * 0.3, H * 0.86, 300, "talk", fc.t, mouth=fc.tl.mouth("LUTIE", fc.T))
        sub = fc.tl.subtitle(fc.T)
        if sub:
            text(ctx, f"{sub['speaker']}: {sub['text']}", W / 2, H - 60, 30, "paper", align="center")
        return cv.rgb()

    return setup, render

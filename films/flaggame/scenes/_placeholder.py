"""Placeholder used until a scene owner replaces films/flaggame/scenes/tXX_*.py.
Shows the film-2 pipeline: gray world -> vx.ink print -> vx.flag spot plate -> vx.comic lettering."""
import math

from vx import *
from vx import ink, comic
from vx.flag import Flag
from vx.chars import Character


def make(sid):
    def setup(S):
        tr = Flag(pole=520, seed=4).simulate(S.n, lambda i: (W * 0.62, H * 0.9, 0.03 * math.sin(i / 30)),
                                             lambda i: (500 + 250 * math.sin(i / 37), 0))
        return {"flag": tr, "who": Character("summer")}

    def render(fc, st):
        world = Canvas(fc, bg=(0.82, 0.82, 0.82))
        ctx = world.ctx
        ctx.set_source(lin_grad(0, 0, 0, H, [(0, (0.55, 0.55, 0.55)), (1, (0.95, 0.95, 0.95))]))
        ctx.paint()
        st["who"].draw(ctx, W * 0.3, H * 0.92, 620, "talk", fc.t, mouth=0.4 * (1 + math.sin(fc.t * 9)))
        img = ink.tract(fc, world.rgb())
        flags = Canvas(fc)
        st["flag"].draw(flags.ctx, fc.f)
        img = ink.spot(img, flags.rgba())
        top = Canvas(fc)
        comic.caption(top.ctx, f"{sid.upper()} - {fc.tl.scene(sid)['title'].upper()}", 60, 50, 900, size=40)
        for l in fc.tl.lines_for(scene=sid):
            if comic.say(top.ctx, fc, l["id"], W * 0.36, H * 0.25, tail=(W * 0.3, H * 0.45)):
                break
        return top.over(img)

    return setup, render

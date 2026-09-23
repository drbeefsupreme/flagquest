"""CHARACTERS. (STUB v0 by Main - RigMaster replaces internals, keeping these public signatures.)

Character(kind, seed=0, **style)
  kinds: "lutie" (novice), "crocus" (master), "schismmancer", "commander", "citizen", "cultist",
         "philosopher", "jaguar" (President Jaguar)
  .anchors(x, y, h, pose="stand", t=0.0, facing=1, pole_len=None, **kw) -> dict      (pure, no drawing)
  .draw(ctx, x, y, h, pose="stand", t=0.0, mouth=0.0, look=(0, 0), expr="neutral", facing=1,
        blink=None, alpha=1.0, pole_len=None, **kw) -> anchors dict
      (x, y) = point between the feet on the ground, h = full standing height in design px.
      mouth  = 0..1 lip-sync openness (use tl.mouth(SPEAKER, fc.T))
      look   = (-1..1, -1..1) gaze direction; expr in EXPRS; facing = +1 right / -1 left
      pose   = one of POSES, or a dict {pose: weight} to blend
  anchors keys: head, eyes, mouth, chest, hand_l, hand_r, feet, top  -> (x, y)
                pole -> (base_x, base_y, ang) for poses that hold a Flag (pass to Flag.simulate pole_fn)
draw_crowd(ctx, people, t)  people = [dict(x, y, h, kind="citizen", seed, pose, facing, look, mouth)]
"""
import math

from .canvas import set_color, circle, ellipse, poly, col, shade

POSES = ("stand", "walk", "run", "hold_flag", "raise_flag", "plant_flag", "point", "cheer", "shout",
         "meditate", "salute", "crouch", "look_up", "talk")
EXPRS = ("neutral", "happy", "surprised", "worried", "determined", "angry", "serene", "awe")

_ARMS = {  # (shoulder->hand angle for right arm, left arm) radians from straight down, + = forward
    "stand": (0.15, -0.15), "walk": (0.3, -0.3), "run": (0.9, -0.7), "hold_flag": (0.9, -0.2),
    "raise_flag": (2.7, -0.3), "plant_flag": (1.2, 1.0), "point": (1.6, -0.2), "cheer": (2.6, -2.6),
    "shout": (0.6, -0.6), "meditate": (0.6, -0.6), "salute": (2.2, -0.15), "crouch": (0.7, -0.4),
    "look_up": (0.2, -0.2), "talk": (0.7, -0.25),
}

_LOOK = {
    "lutie":        dict(robe="robe", skin="skin2", hair="#3A2A22", hood=True),
    "crocus":       dict(robe="robe", skin="skin1", hair="#EDEBE6", hood=True, beard=True, inner="#1A1820"),
    "schismmancer": dict(robe="robe", skin="skin3", hair="#222", hood=False, vest="#2A2E36"),
    "commander":    dict(robe="robe", skin="skin2", hair="#222", hood=False, vest="#2A2E36"),
    "citizen":      dict(robe="cloth_grey", skin="skin1", hair="#3b3b44", hood=False),
    "cultist":      dict(robe="cloth_plum", skin="skin2", hair="#2a2a2a", hood=True),
    "philosopher":  dict(robe="bone", skin="skin1", hair="#DDDDDD", hood=False, beard=True),
    "jaguar":       dict(robe="#1E2A44", skin="#D9A441", hair="#D9A441", hood=False),
}


def _blend_arms(pose):
    if isinstance(pose, dict):
        tot = sum(pose.values()) or 1
        r = sum(_ARMS[k][0] * w for k, w in pose.items()) / tot
        l = sum(_ARMS[k][1] * w for k, w in pose.items()) / tot
        return r, l, max(pose, key=pose.get)
    return (*_ARMS.get(pose, _ARMS["stand"]), pose)


class Character:
    def __init__(self, kind, seed=0, **style):
        self.kind, self.seed = kind, seed
        self.look = dict(_LOOK.get(kind, _LOOK["citizen"]))
        self.look.update(style)

    def anchors(self, x, y, h, pose="stand", t=0.0, facing=1, pole_len=None, **kw):
        ar, al, main = _blend_arms(pose)
        bob = math.sin(t * 2.1 + self.seed) * h * 0.004
        head_r = h * 0.11
        hx, hy = x, y - h + head_r + bob
        sh_y = y - h * 0.72 + bob
        arm = h * 0.34
        sx_r, sx_l = x + facing * h * 0.09, x - facing * h * 0.09
        hand_r = (sx_r + facing * math.sin(ar) * arm, sh_y + math.cos(ar) * arm)
        hand_l = (sx_l + facing * math.sin(al) * arm, sh_y + math.cos(al) * arm)
        a = {"head": (hx, hy), "eyes": (hx + facing * head_r * 0.25, hy - head_r * 0.05),
             "mouth": (hx + facing * head_r * 0.3, hy + head_r * 0.45), "chest": (x, y - h * 0.6),
             "hand_l": hand_l, "hand_r": hand_r, "feet": (x, y), "top": (x, y - h)}
        L = pole_len or h * 1.35
        if main in ("hold_flag", "raise_flag", "plant_flag", "salute"):
            ang = {"hold_flag": 0.12, "raise_flag": 0.05, "plant_flag": 0.0, "salute": 0.1}[main] * facing
            grip = 0.28 if main != "raise_flag" else 0.12
            bx = hand_r[0] - math.sin(ang) * L * grip
            by = hand_r[1] + math.cos(ang) * L * grip
            a["pole"] = (bx, by, ang)
        return a

    def draw(self, ctx, x, y, h, pose="stand", t=0.0, mouth=0.0, look=(0, 0), expr="neutral", facing=1,
             blink=None, alpha=1.0, pole_len=None, **kw):
        a = self.anchors(x, y, h, pose, t, facing, pole_len)
        lk = self.look
        hx, hy = a["head"]
        head_r = h * 0.11
        # robe
        top = y - h * 0.78
        set_color(ctx, lk["robe"], alpha)
        poly(ctx, [(x - h * 0.11, top), (x + h * 0.11, top), (x + h * 0.2, y), (x - h * 0.2, y)])
        ctx.fill()
        set_color(ctx, shade(lk["robe"], 0.75), alpha)
        poly(ctx, [(x - h * 0.03 * facing, top), (x + h * 0.11 * facing, top), (x + h * 0.2 * facing, y),
                   (x + h * 0.02 * facing, y)])
        ctx.fill()
        # arms
        ctx.set_line_width(h * 0.05)
        for hand, sx in ((a["hand_l"], x - facing * h * 0.09), (a["hand_r"], x + facing * h * 0.09)):
            set_color(ctx, shade(lk["robe"], 0.85), alpha)
            ctx.move_to(sx, y - h * 0.72)
            ctx.line_to(*hand)
            ctx.stroke()
            set_color(ctx, lk["skin"], alpha)
            circle(ctx, hand[0], hand[1], h * 0.03)
            ctx.fill()
        # head
        set_color(ctx, lk["skin"], alpha)
        circle(ctx, hx, hy, head_r)
        ctx.fill()
        if lk.get("hood"):
            set_color(ctx, shade(lk["robe"], 0.8), alpha)
            ctx.arc(hx, hy, head_r * 1.18, math.pi * 1.05, math.pi * 1.95)
            ctx.line_to(hx + head_r * 1.18, hy + head_r * 0.6)
            ctx.arc_negative(hx, hy, head_r * 1.0, math.pi * 0.1, math.pi * 1.0 + 0.4)
            ctx.close_path()
            ctx.fill()
        if lk.get("beard"):
            set_color(ctx, lk["hair"], alpha)
            poly(ctx, [(hx - head_r * 0.7, hy + head_r * 0.2), (hx + head_r * 0.7, hy + head_r * 0.2),
                       (hx + facing * head_r * 0.2, hy + head_r * 2.0)])
            ctx.fill()
        # eyes (blink every ~3.7 s)
        ex, ey = hx + facing * head_r * 0.25 + look[0] * head_r * 0.12, hy - head_r * 0.05 + look[1] * head_r * 0.1
        bl = blink if blink is not None else ((t + self.seed * 0.37) % 3.7) < 0.12
        set_color(ctx, "ink", alpha)
        for dx in (-head_r * 0.32, head_r * 0.32):
            if bl:
                ctx.set_line_width(head_r * 0.08)
                ctx.move_to(ex + dx - head_r * 0.1, ey)
                ctx.line_to(ex + dx + head_r * 0.1, ey)
                ctx.stroke()
            else:
                ellipse(ctx, ex + dx, ey, head_r * 0.09, head_r * 0.13)
                ctx.fill()
        # mouth
        mx, my = hx + facing * head_r * 0.2, hy + head_r * 0.45
        ellipse(ctx, mx, my, head_r * (0.18 + 0.08 * mouth), head_r * (0.03 + 0.2 * mouth))
        set_color(ctx, "#3a1f1f", alpha)
        ctx.fill()
        return a


def draw_crowd(ctx, people, t):
    cache = {}
    for p in people:
        k = (p.get("kind", "citizen"), p.get("seed", 0))
        if k not in cache:
            cache[k] = Character(k[0], k[1])
        cache[k].draw(ctx, p["x"], p["y"], p["h"], p.get("pose", "stand"), t, p.get("mouth", 0.0),
                      p.get("look", (0, 0)), p.get("expr", "neutral"), p.get("facing", 1), alpha=p.get("alpha", 1.0))

"""t01 — page 1, panel 1: Summer and Raven at their camp in the rain (tract style).  Owner: T01Tract.

Composition after the tract's page 23 panel 1: RAVEN in the left foreground (back-3/4, the great glossy black
mass of hair, hoop earring, deadpan lips), SUMMER on the right (3/4, sunny), and between them, far off in the
rain, the FLAG-BEARER (Crow as a long-coated silhouette in a cap) carrying the only colour in the world.
Behind them: the underside of a pipe-frame shade tarp dripping, a hanging lantern, the open playa in the rain,
far camps and domes.  Tract rain = diagonal ink strokes falling down-left.

Everything is drawn in PANEL-LOCAL page units (0..860 x 0..352) and exposed through the t01_page drawer
protocol:
  PANEL1.world(ctx, T, rect) / .flags(ctx, T, rect) / .front(ctx, T, rect) / .letters(ctx, fc, T, rect)
  (T = global seconds; valid for ANY T: the bearer walks 8.5-13.1, afterwards the camp just rains on)
draw_camp_bg(ctx, rect, T, pan=0.5, rain=1.0, sun=0.0)   the reusable background (T08: sun>0 = amen break)
Drawing ticks on twos (12 fps) like a drawn comic; the flag cloth simulates at 24 fps.
"""
import math

import numpy as np

from vx import set_color
from vx import comic
from vx.chars import Character
from vx.ease import hash01, smoothstep, clamp
from vx.flag import Flag

from . import t01_page as P

PW_, PH_ = P.PANEL_W, P.PANEL_H          # 860 x 352
HORIZON = 212.0
# the flag-bearer
BEARER_T0, BEARER_T1 = 8.45, 13.15
BEARER_X0, BEARER_X1 = 262.0, 560.0
BEARER_Y = 256.0
BEARER_H = 84.0
POLE_L = 112.0
FLAG_F0 = int(round(8.0 * 24))            # global frame of track frame 0
FLAG_N = int(round(6.0 * 24))
# the cast
RAVEN = dict(x=165.0, y=162.0, s=108.0)       # head centre + head radius (PU)
SUMMER = dict(x=636.0, y=158.0, s=84.0)
PRINT_T = 6.90                           # the frozen 'printed' instant of the panel (before it comes alive)


def twos(T):
    return math.floor(T * 12.0) / 12.0


def _bearer_x(T):
    u = clamp((T - BEARER_T0) / (BEARER_T1 - BEARER_T0))
    return BEARER_X0 + (BEARER_X1 - BEARER_X0) * u


_CH = {}


def bearer():
    if "crow" not in _CH:
        _CH["crow"] = Character("crow", 3, render="ink", hat="cap")
    return _CH["crow"]


def _bearer_pose(T):
    return {"walk": 1.0, "hold_flag": 1.0}


def _bearer_anchors(T, pole_len=POLE_L):
    ch = bearer()
    return ch.anchors(_bearer_x(T), BEARER_Y, BEARER_H, _bearer_pose(T), T, facing=1, pole_len=pole_len, speed=2.0)


def simulate_flag():
    """hero flag of the bearer, panel-local page units; track frame i <-> global frame FLAG_F0 + i"""
    def pole_fn(i):
        T = (FLAG_F0 + i) / 24.0
        return _bearer_anchors(T)["pole"]

    def wind_fn(i):
        T = (FLAG_F0 + i) / 24.0
        return (-330.0 - 90.0 * math.sin(T * 1.9), -20.0)

    return Flag(pole=POLE_L, seed=11, side=-1, turbulence=0.8, flutter=0.8).simulate(FLAG_N, pole_fn, wind_fn, warmup=48)


# ============================================================================== background
def _sky(ctx, sun):
    import cairo
    g = cairo.LinearGradient(0, 0, 0, HORIZON)
    g.add_color_stop_rgb(0.0, 0.88, 0.88, 0.88)
    g.add_color_stop_rgb(1.0, 0.96 + 0.03 * sun, 0.96 + 0.03 * sun, 0.96 + 0.03 * sun)
    ctx.set_source(g)
    ctx.rectangle(0, 0, PW_, HORIZON + 2)
    ctx.fill()
    # low rain clouds: soft darker bands
    for k in range(5):
        y = 40 + k * 26
        set_color(ctx, (0.84 + 0.02 * k,) * 3, 0.55)
        ctx.new_path()
        ctx.move_to(0, y)
        for i in range(18):
            x = i * 52
            ctx.line_to(x, y - 10 - 8 * math.sin(i * 1.3 + k * 2.1))
        ctx.line_to(PW_, y + 18)
        ctx.line_to(0, y + 18)
        ctx.close_path()
        ctx.fill()


def _far_camps(ctx):
    # hazy silhouettes on the horizon: domes, tents, a shade frame, an art car, a far effigy
    g = (0.80, 0.80, 0.80)
    set_color(ctx, g)
    for (x, r) in ((40, 26), (380, 20), (520, 34), (820, 24)):
        ctx.new_path()
        ctx.arc(x, HORIZON, r, math.pi, 2 * math.pi)
        ctx.fill()
    for (x, w, h) in ((120, 70, 22), (230, 50, 16), (600, 90, 26), (700, 60, 18)):
        ctx.new_path()
        ctx.move_to(x, HORIZON)
        ctx.line_to(x + w * 0.5, HORIZON - h)
        ctx.line_to(x + w, HORIZON)
        ctx.close_path()
        ctx.fill()
    set_color(ctx, (0.62, 0.62, 0.62))
    ctx.set_line_width(2.0)
    for x in (300, 350):
        ctx.move_to(x, HORIZON)
        ctx.line_to(x, HORIZON - 30)
    ctx.move_to(292, HORIZON - 30)
    ctx.line_to(358, HORIZON - 26)
    ctx.stroke()
    # the far effigy (a lattice man, arms raised) — foreshadowing
    ex, ey = 455.0, HORIZON
    ctx.set_line_width(1.6)
    set_color(ctx, (0.60, 0.60, 0.60))
    ctx.move_to(ex - 6, ey)
    ctx.line_to(ex, ey - 26)
    ctx.line_to(ex + 6, ey)
    ctx.move_to(ex, ey - 26)
    ctx.line_to(ex, ey - 44)
    ctx.move_to(ex, ey - 38)
    ctx.line_to(ex - 10, ey - 52)
    ctx.move_to(ex, ey - 38)
    ctx.line_to(ex + 10, ey - 52)
    ctx.stroke()
    ctx.new_path()
    ctx.arc(ex, ey - 48, 3.2, 0, 2 * math.pi)
    ctx.fill()


def _mud(ctx, T):
    import cairo
    g = cairo.LinearGradient(0, HORIZON, 0, PH_)
    g.add_color_stop_rgb(0.0, 0.90, 0.90, 0.90)
    g.add_color_stop_rgb(1.0, 0.80, 0.80, 0.80)
    ctx.set_source(g)
    ctx.rectangle(0, HORIZON, PW_, PH_ - HORIZON)
    ctx.fill()
    # ruts and churned clods (perspective ellipses), puddles reflecting the sky (paper white)
    rng = np.random.default_rng(23)
    for k in range(70):
        y = HORIZON + 4 + (PH_ - HORIZON) * (rng.random() ** 1.6)
        x = rng.random() * PW_
        sc = 0.25 + 1.4 * (y - HORIZON) / (PH_ - HORIZON)
        set_color(ctx, (0.68 + 0.1 * rng.random(),) * 3)
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(9 * sc * (0.6 + rng.random()), 2.2 * sc)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.fill()
    for (x, y, w) in ((330, 238, 60), (470, 252, 90), (600, 232, 40), (250, 300, 120), (560, 318, 150)):
        sc = 0.3 + 1.2 * (y - HORIZON) / (PH_ - HORIZON)
        set_color(ctx, (1.0, 1.0, 1.0))
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(w * 0.5, 5 * sc)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.fill()
        # ripple rings (ink) animated on twos
        for r in range(2):
            ph = (twos(T) * 1.3 + hash01(int(x) + r * 7) ) % 1.0
            set_color(ctx, (0, 0, 0), 0.9 * (1 - ph))
            ctx.set_line_width(0.9)
            ctx.save()
            ctx.translate(x + (hash01(int(x), r) - 0.5) * w * 0.6, y)
            ctx.scale(4 + ph * 16, (1 + ph * 3.5) * sc * 0.6)
            ctx.arc(0, 0, 1, 0, 2 * math.pi)
            ctx.restore()
            ctx.stroke()
        # dark lower lip of the puddle
        set_color(ctx, (0, 0, 0))
        ctx.set_line_width(1.4)
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(w * 0.5, 5 * sc)
        ctx.arc(0, 0, 1, 0.15 * math.pi, 0.85 * math.pi)
        ctx.restore()
        ctx.stroke()


def _shade(ctx, T):
    # the underside of the shade tarp (dark, sagging) across the top, and its poles; drips from the edge
    set_color(ctx, (0.0, 0.0, 0.0))
    ctx.new_path()
    ctx.move_to(0, 0)
    ctx.line_to(PW_, 0)
    ctx.line_to(PW_, 30)
    for i in range(13, -1, -1):
        x = i * PW_ / 13
        sag = 10 * math.sin(math.pi * ((x / PW_ * 3) % 1.0))
        ctx.line_to(x, 30 + sag)
    ctx.close_path()
    ctx.fill()
    # tarp seams / folds (ink)
    set_color(ctx, (0, 0, 0))
    ctx.set_line_width(2.4)
    ctx.new_path()
    for i in range(14):
        x = i * PW_ / 13
        sag = 10 * math.sin(math.pi * ((x / PW_ * 3) % 1.0))
        (ctx.move_to if i == 0 else ctx.line_to)(x, 30 + sag)
    ctx.stroke()
    set_color(ctx, (1, 1, 1))
    ctx.set_line_width(1.3)
    for k in range(9):
        x = 40 + k * 97
        ctx.move_to(x, 2)
        ctx.line_to(x + 18, 24 + 6 * math.sin(k))
    ctx.stroke()
    set_color(ctx, (0, 0, 0))
    # poles
    ctx.set_line_width(7.0)
    for x in (38.0, 836.0):
        ctx.move_to(x, 20)
        ctx.line_to(x, PH_)
    ctx.stroke()
    # drips along the edge (on twos): each drip grows, falls
    tt = twos(T)
    for k in range(16):
        x = 30 + k * 54 + 9 * math.sin(k * 3.1)
        ph = (tt * (0.7 + 0.3 * hash01(k, 5)) + hash01(k, 9)) % 1.0
        sag = 10 * math.sin(math.pi * ((x / PW_ * 3) % 1.0))
        y0 = 30 + sag + 2
        if ph < 0.45:
            r = 1.5 + 2.5 * ph / 0.45
            _drop(ctx, x, y0 + r, r)
        else:
            y = y0 + 6 + (ph - 0.45) ** 2 * 900
            if y < PH_:
                _drop(ctx, x, y, 2.6)


def _drop(ctx, x, y, r):
    ctx.new_path()
    ctx.move_to(x, y - r * 2.2)
    ctx.curve_to(x + r * 1.1, y - r * 0.6, x + r, y + r, x, y + r)
    ctx.curve_to(x - r, y + r, x - r * 1.1, y - r * 0.6, x, y - r * 2.2)
    set_color(ctx, (1, 1, 1))
    ctx.fill_preserve()
    set_color(ctx, (0, 0, 0))
    ctx.set_line_width(0.9)
    ctx.stroke()


def _lantern(ctx, T):
    x, y = 600.0, 36.0
    sw = 0.06 * math.sin(twos(T) * 2.2)
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(sw)
    set_color(ctx, (0, 0, 0))
    ctx.set_line_width(1.4)
    ctx.move_to(0, 0)
    ctx.line_to(0, 16)
    ctx.stroke()
    ctx.new_path()
    ctx.arc(0, 18, 3, 0, 2 * math.pi)
    ctx.stroke()
    # body (unlit: daytime rain) — glass grey, cage lines
    ctx.rectangle(-9, 22, 18, 26)
    set_color(ctx, (0.78, 0.78, 0.78))
    ctx.fill_preserve()
    set_color(ctx, (0, 0, 0))
    ctx.set_line_width(1.8)
    ctx.stroke()
    ctx.move_to(-11, 22)
    ctx.line_to(11, 22)
    ctx.move_to(-7, 18)
    ctx.line_to(7, 18)
    ctx.move_to(-11, 48)
    ctx.line_to(11, 48)
    ctx.set_line_width(2.6)
    ctx.stroke()
    ctx.set_line_width(1.0)
    for xx in (-4, 4):
        ctx.move_to(xx, 22)
        ctx.line_to(xx, 48)
    ctx.stroke()
    ctx.restore()


def _rain(ctx, T, amount=1.0, n=170, seed=0, x0=0.0, x1=PW_, y0=0.0, y1=PH_, weight=1.0):
    if amount <= 0:
        return
    tt = twos(T)
    set_color(ctx, (0, 0, 0))
    ctx.set_line_width(1.05 * weight)
    ang = math.radians(16)
    dx, dy = -math.sin(ang), math.cos(ang)
    H_ = y1 - y0 + 60
    for k in range(int(n * amount)):
        sp = 420 + 140 * hash01(k, seed + 1)
        L = 12 + 12 * hash01(k, seed + 2)
        yy = (hash01(k, seed + 3) * H_ + tt * sp) % H_ + y0 - 30
        xx = x0 + hash01(k, seed + 4) * (x1 - x0 + 80) - yy * dx * -1.0 * 0.0 + (yy - y0) * dx
        xx = x0 + ((xx - x0) % (x1 - x0 + 60))
        ctx.move_to(xx, yy)
        ctx.line_to(xx + dx * L, yy + dy * L)
    ctx.stroke()


def _splashes(ctx, T, n=40, seed=5):
    tt = twos(T)
    set_color(ctx, (0, 0, 0))
    ctx.set_line_width(0.9)
    for k in range(n):
        ph = (tt * 2.0 + hash01(k, seed)) % 1.0
        if ph > 0.35:
            continue
        y = HORIZON + 10 + (PH_ - HORIZON - 10) * hash01(k, seed + 1)
        x = hash01(k + int(tt * 2 + hash01(k, seed)), seed + 2) * PW_
        sc = 0.4 + (y - HORIZON) / (PH_ - HORIZON)
        r = 3 + 6 * ph
        ctx.move_to(x - r * sc, y - r * 0.8 * sc)
        ctx.line_to(x - r * 0.3 * sc, y)
        ctx.move_to(x + r * sc, y - r * 0.8 * sc)
        ctx.line_to(x + r * 0.3 * sc, y)
    ctx.stroke()


def draw_camp_bg(ctx, rect, T, pan=0.5, rain=1.0, sun=0.0):
    """the camp in the rain (sky, far camps, playa mud, puddles, the shade structure, lantern, rain)"""
    x, y, w, h = rect
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(w / PW_, h / PH_)
    _sky(ctx, sun)
    _far_camps(ctx)
    _mud(ctx, T)
    _rain(ctx, T, rain, n=90, seed=2, y1=HORIZON + 40, weight=0.7)
    _splashes(ctx, T)
    _shade(ctx, T)
    _lantern(ctx, T)
    ctx.restore()


# ============================================================================== the drawer
class Panel1:
    def __init__(self):
        self._track = None

    @property
    def track(self):
        if self._track is None:
            self._track = simulate_flag()
        return self._track

    def _flag_i(self, T):
        return T * 24.0 - FLAG_F0

    def _anim_T(self, T):
        # before the dive the panel is a frozen print; it comes alive after
        return T if T >= PRINT_T else PRINT_T

    def world(self, ctx, T, rect):
        Ta = self._anim_T(T)
        t2 = twos(Ta)
        x, y, w, h = rect
        draw_camp_bg(ctx, rect, Ta)
        ctx.save()
        ctx.translate(x, y)
        # the flag-bearer (back layer): a long-coated silhouette in the rain haze
        if BEARER_T0 - 0.3 <= Ta <= BEARER_T1 + 0.3:
            ch = bearer()
            ch.draw(ctx, _bearer_x(t2), BEARER_Y, BEARER_H, _bearer_pose(t2), t2, facing=1, pole_len=POLE_L,
                    speed=2.0, layer="back", silhouette=(0.0, 0.0, 0.0))
        # mid rain in front of the playa
        _rain(ctx, Ta, 1.0, n=80, seed=9, weight=0.9)
        ctx.restore()

    def flags(self, ctx, T, rect):
        if not (BEARER_T0 - 0.3 <= T <= BEARER_T1 + 0.3):
            return
        i = self._flag_i(T)
        if i < 0 or i > self.track.n - 1:
            return
        x, y, w, h = rect
        ctx.save()
        ctx.translate(x, y)
        self.track.draw(ctx, i, light=(-0.45, -0.8, 0.4))
        ctx.restore()

    def front(self, ctx, T, rect, fc=None):
        Ta = self._anim_T(T)
        t2 = twos(Ta)
        x, y, w, h = rect
        ctx.save()
        ctx.translate(x, y)
        if BEARER_T0 - 0.3 <= Ta <= BEARER_T1 + 0.3:
            bearer().draw(ctx, _bearer_x(t2), BEARER_Y, BEARER_H, _bearer_pose(t2), t2, facing=1, pole_len=POLE_L,
                          speed=2.0, layer="front", silhouette=(0.0, 0.0, 0.0))
        tl = _tl()
        mr = tl.mouth("RAVEN", t2) if Ta > PRINT_T else 0.0
        ms = tl.mouth("SUMMER", t2) if Ta > PRINT_T else 0.0
        # Summer follows the passing flag with her eyes, then turns to Raven, proud of her confession
        lx = -0.9 + 1.2 * smoothstep(9.2, 12.6, t2)
        lx = lx if t2 < 11.35 else -0.55
        blink = _blink(t2, 1)
        draw_summer(ctx, t2, mouth=ms, look=(lx, 0.15), blink=blink)
        draw_raven(ctx, t2, mouth=mr, blink=_blink(t2, 2))
        ctx.restore()

    def letters(self, ctx, fc, T, rect):
        if T < 7.2:
            return
        x, y, w, h = rect
        tr = self.track
        bounds = (x + 8, y + 8, x + w - 8, y + h - 8)

        def avoid(Tq):
            i = Tq * 24.0 - FLAG_F0
            if i < 0 or i > tr.n - 1:
                return []
            sh = comic.track_avoid(tr, i, grow=6.0)
            from shapely import affinity
            return [affinity.translate(g, x, y) for g in sh]
        sz = P.LETTER["balloon"]
        m1 = raven_mouth(twos(T))
        m2 = summer_mouth(twos(T))
        mr = (x + m1[0], y + m1[1])
        ms = (x + m2[0], y + m2[1])
        # Raven's line rides high, above the flag's whole path (the cloth never rises above y~100)
        comic.say(ctx, fc, "R01", x + 400, y + 40, w=380, tail=(mr[0] + 6, mr[1] - 4), size=sz, avoid=avoid,
                  bounds=(x + 150, y + 4, x + 700, y + 92), T=T)
        # Summer's confession sits beside her face (tail runs straight in to her mouth, never across her eyes)
        comic.say(ctx, fc, "S01", x + 390, y + 52, w=400, tail=(ms[0] - 8, ms[1] + 2), size=sz, avoid=avoid,
                  bounds=(x + 150, y + 4, x + 720, y + 100), T=T)
        comic.say(ctx, fc, "S02", x + 390, y + 312, w=320, tail=(ms[0] - 6, ms[1] + 8), size=sz, avoid=avoid,
                  bounds=bounds, T=T)


def place(ch, head_xy, radius, pose, t, facing, **kw):
    """-> (x, y, h) ground point + height putting the head centre at head_xy with the given head radius"""
    h = radius / 0.1163
    a = ch.anchors(0.0, 0.0, h, pose, t, facing=facing, **{k: v for k, v in kw.items() if k in ("turn", "speed")})
    hx, hy = a["head"]
    return head_xy[0] - hx, head_xy[1] - hy, h


def summer():
    if "summer" not in _CH:
        _CH["summer"] = Character("summer", 1, render="ink")
    return _CH["summer"]


def raven():
    if "raven" not in _CH:
        _CH["raven"] = Character("raven", 1, render="ink")
    return _CH["raven"]


SUMMER_KW = dict(turn=0.5)
RAVEN_KW = dict(turn=1.30, chair=False)


def summer_anchors(t):
    ch = summer()
    x, y, h = place(ch, (SUMMER["x"], SUMMER["y"] + 1.5 * math.sin(t * 1.6)), SUMMER["s"], "sit", t, -1, **SUMMER_KW)
    return ch.anchors(x, y, h, "sit", t, facing=-1, **SUMMER_KW), (x, y, h)


def draw_summer(ctx, t, mouth=0.0, look=(0.0, 0.0), blink=None, expr="cheerful"):
    ch = summer()
    x, y, h = place(ch, (SUMMER["x"], SUMMER["y"] + 1.5 * math.sin(t * 1.6)), SUMMER["s"], "sit", t, -1, **SUMMER_KW)
    tilt = 0.05 * math.sin(t * 0.8) - 0.06 * smoothstep(11.4, 11.9, t) * (1 - smoothstep(17.6, 18.2, t))
    return ch.draw(ctx, x, y, h, "sit", t, mouth=mouth, look=look, expr=expr, facing=-1, blink=blink, tilt=tilt,
                   wind=0.25, **SUMMER_KW)


def draw_raven(ctx, t, mouth=0.0, blink=None):
    ch = raven()
    x, y, h = place(ch, (RAVEN["x"], RAVEN["y"] + 1.0 * math.sin(t * 1.2 + 1)), RAVEN["s"], "sit", t, 1, **RAVEN_KW)
    return ch.draw(ctx, x, y, h, "sit", t, mouth=mouth, look=(0.6, 0.1), expr="disdain", facing=1, blink=blink,
                   wind=0.25, **RAVEN_KW)


def raven_mouth(t):
    ch = raven()
    x, y, h = place(ch, (RAVEN["x"], RAVEN["y"]), RAVEN["s"], "sit", t, 1, **RAVEN_KW)
    return ch.anchors(x, y, h, "sit", t, facing=1, **RAVEN_KW)["mouth"]


def summer_mouth(t):
    return summer_anchors(t)[0]["mouth"]


def _blink(t, seed):
    per = 3.1 + seed * 0.7
    ph = (t + seed * 1.37) % per
    return 1.0 if ph < 0.09 else 0.0


_TL = {}


def _tl():
    if "tl" not in _TL:
        from vx.timeline import get_timeline
        _TL["tl"] = get_timeline()
    return _TL["tl"]


PANEL1 = Panel1()

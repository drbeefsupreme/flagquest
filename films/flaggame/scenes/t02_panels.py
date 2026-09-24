"""t02 panels as page drawers (protocol of scenes.t01_page: world/flags/front/letters in PAGE units).

PANELS = {"p2": PanelA(), "p3": PanelB()}   page 1 (ref page 23): Crow's witness + the icon
PAGE2  = {"p1": Page2Row()}                 page 2 top row (ref page 24): anime Summer | Crow leans in
Every drawer is a pure function of the global time T. For T < 6.4 they show the finished 'printed'
panel (all balloons lettered: t01's cover/riffle/dive views); for 6.4 <= T < 18.46 the live state at 18.46.
Owner: T02Wrong."""
import math
import os
import zlib

import numpy as np

from vx import comic
from vx.config import W, H
from vx.config import CACHE
from vx.ease import clamp, smoothstep, ease_out_back, ease_in_out, ease_in_out_sine, spring, hash01, seg
from vx.flag import Flag, FlagTrack
from vx.timeline import get_timeline

from . import t01_page as P
from .t02_brush import stroke, grey, WHITE, dots, poly_fill
from .t02_cast import summer
from .t02_crow import crow, point_arm

TL = get_timeline()
FPS = 24
T0 = TL.scene("t02")["start"]            # 18.4583
T_END = TL.scene("t02")["end"]           # 43.2917
APPEAR = TL.cue("preacher_appears")      # 18.46
WRONG = TL.cue("wrong_burst")            # 22.117
ICON = TL.cue("icon")                    # 25.20
ESSENCE = TL.cue("essence")              # 29.288
DISCL = TL.cue("disclaimer")             # 31.06
ANIME = TL.cue("anime")                  # 33.75
FLASH = TL.cue("flashback")              # 41.335
PRINTED_BEFORE = 6.4                     # t01's dive: before it the tract is seen as a printed object
L = P.LETTER
_VERSION = "v3"


def live_T(T, canon):
    """map t01's times onto my panel's clock: printed still before the dive, frozen start state before 18.46"""
    if T < PRINTED_BEFORE:
        return canon
    return max(T, APPEAR - 1e-3) if T < T0 else T


def mouth_of(speaker, T):
    return clamp(TL.mouth(speaker, T) * 1.25)


def frame_i(T, T_start):
    return (T - T_start) * FPS


# ============================================================ flags (lazy, disk-cached sims)
_TRACKS = {}


def track(name, n, pole_fn, wind_fn, pole, seed, **kw):
    sig = repr((n, pole, seed, sorted(kw.items()), [tuple(round(v, 3) for v in pole_fn(i)) for i in (0, n // 3, n - 1)],
                [tuple(round(v, 2) for v in wind_fn(i)) for i in (0, n // 2, n - 1)]))
    key = f"{name}_{_VERSION}_{zlib.crc32(sig.encode()):08x}"
    tr = _TRACKS.get(key)
    if tr is not None:
        return tr
    d = CACHE / "t02"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"flag_{key}.npz"
    if path.exists():
        try:
            tr = FlagTrack.load(path)
        except Exception:
            tr = None
    if tr is None:
        tr = Flag(pole=pole, seed=seed, **kw).simulate(n, pole_fn, wind_fn)
        tmp = d / f".flag_{key}_{os.getpid()}.npz"
        tr.save(tmp)
        tmp.replace(path)
    _TRACKS[key] = tr
    return tr


def draw_track(ctx, tr, T, T_start, **kw):
    i = clamp(frame_i(T, T_start), 0, tr.n - 1)
    tr.draw(ctx, i, **kw)


def white_rect(ctx, rect, v=1.0):
    ctx.save()
    ctx.rectangle(*rect)
    ctx.set_source_rgb(v, v, v)
    ctx.fill()
    ctx.restore()


def rain(ctx, rect, T, *, n=70, seed=3, length=34.0, ang=0.22, alpha=1.0, width=1.1, speed=900.0):
    """comic rain: thin inked streaks falling, re-drawn on twos"""
    x0, y0, w, h = rect
    rng = np.random.default_rng(seed)
    tt = math.floor(T * 12) / 12
    ctx.save()
    ctx.set_source_rgba(0, 0, 0, alpha)
    ctx.set_line_width(width)
    dx, dy = math.sin(ang), math.cos(ang)
    for i in range(n):
        px = x0 + rng.random() * (w + 80) - 40
        py0 = rng.random() * (h + 200)
        py = y0 - 100 + (py0 + tt * speed * (0.8 + 0.4 * rng.random())) % (h + 200)
        L = length * (0.6 + 0.8 * rng.random())
        ctx.move_to(px + dx * (py - y0) * 0.0, py)
        ctx.line_to(px - dx * L, py - dy * L)
    ctx.stroke()
    ctx.restore()


# ============================================================ PANEL A (page 1, p2): "...WAS WRONG!!"
class PanelA:
    """Summer (left) is confronted by Crow (right), who is suddenly just there."""
    RECT = P.PANELS["p2"]
    SUM = (262.0, 612.0, 92.0)
    CRW = (716.0, 606.0, 86.0)
    CANON = 24.62
    POLE = 520.0

    def __init__(self):
        self._tr = None

    # ---------------------------------------------------------------- flag
    def flag(self):
        if self._tr is None:
            n = int((25.6 - T0) * FPS) + 2
            wrong_i = (WRONG - T0) * FPS

            def pole_fn(i):
                # held low in his far hand; sways with his gestures, jolts on the WRONG!! thrust
                sway = 0.025 * math.sin(i / 24 * 1.3)
                jolt = 0.0
                if i > wrong_i - 5:
                    u = (i - (wrong_i - 5)) / 24
                    jolt = -0.05 * math.exp(-4 * max(0, u - 0.2)) * math.sin(u * 16) if u > 0.2 else -0.03 * u / 0.2
                return (902.0 + 8 * math.sin(i / 24 * 0.9), 925.0, -0.10 + sway + jolt)

            def wind_fn(i):
                return (60 + 50 * math.sin(i / 24 * 0.7), 0.0, 0.0)

            self._tr = track("pA", n, pole_fn, wind_fn, self.POLE, seed=21, turbulence=0.6, flutter=0.5)
        return self._tr

    # ---------------------------------------------------------------- acting
    def state(self, T):
        s = {}
        wr = T - WRONG
        crow_in = T >= APPEAR
        s["crow"] = crow_in
        s["pop"] = clamp((T - APPEAR) / 0.18) if crow_in else 0.0
        # Summer: pleased -> double take (18.62) -> puzzled -> SHOCK on WRONG -> shouting S03
        take = smoothstep(18.60, 18.78, T)
        shock = smoothstep(WRONG - 0.02, WRONG + 0.06, T)
        s3 = TL.line("S03")
        talking = s3["start"] - 0.05 <= T <= s3["end"] + 0.25
        s["sum"] = dict(
            look=(0.35 + 0.65 * take, -0.1 + 0.1 * take - 0.15 * shock),
            eye_open=1.0 + 0.25 * take + 0.35 * shock + 0.08 * math.sin(T * 9) * shock * (1 - smoothstep(23, 24, T)),
            pupil=1.0 - 0.1 * take - 0.55 * shock,
            brow=-0.1 + 0.6 * take + 0.45 * shock,
            smile=0.45 * (1 - take) + 0.05,
            mouth_open=max(0.12 * take * (1 - smoothstep(18.9, 19.3, T)) + 0.35 * shock * (1 - smoothstep(22.6, 23.1, T)),
                           mouth_of("SUMMER", T) if talking else 0.0) if T >= APPEAR else 0.0,
            mshape="o" if (T < 22.0 or not talking) else "shout",
            sweat=(1 if T > 20.4 else 0) + (2 if shock > 0.5 else 0),
            tilt=-0.03 * take - 0.06 * shock * math.exp(-2.5 * max(0, wr)) + 0.02 * math.sin(T * 1.1),
            blush=0.0,
        )
        s["sum_dx"] = -14 * shock * math.exp(-3 * max(0, wr)) * math.cos(max(0, wr) * 18)
        # Crow: lecturing finger (P01) -> anticipation on 'WAS' -> THRUST on 'WRONG'
        was = TL.word("P02", "was")[0]
        antic = smoothstep(was - 0.05, WRONG - 0.02, T)
        thrust = smoothstep(WRONG - 0.03, WRONG + 0.05, T)
        up = (1.10, 0.50)          # lecturing finger raised in front of him
        back = (0.75, 0.55)
        point = (2.25, 0.95)
        hx = up[0] + (back[0] - up[0]) * antic
        hy = up[1] + (back[1] - up[1]) * antic
        hx += (point[0] - hx) * thrust
        hy += (point[1] - hy) * thrust
        rec = spring(max(0.0, wr), 3.5, 7.0) if wr > 0 else 0.0
        if wr > 0:
            hx = point[0] + 0.25 * (1 - rec)
            hy = point[1]
        s["hand"] = (hx, hy)
        speaking = TL.speaking("PREACHER", T, pad=0.05)
        s["crow_kw"] = dict(
            mouth_open=mouth_of("PREACHER", T) if speaking else 0.0,
            mshape="shout" if (was - 0.05 <= T <= 23.0) else "talk",
            brow=-0.35 - 0.65 * thrust + 0.35 * smoothstep(23.3, 23.8, T),
            grin=0.25 * smoothstep(23.2, 23.8, T),
            look=(0.4, 0.05),
            pupil=0.9 - 0.3 * thrust * (1 - smoothstep(23.0, 23.6, T)),
            glint=1.0,
            tilt=0.03 * math.sin(T * 1.4) - 0.07 * antic * (1 - thrust) + 0.05 * thrust * math.exp(-3 * max(0, wr)),
        )
        s["crow_dx"] = -26 * thrust * (1 - 0.6 * smoothstep(WRONG, WRONG + 0.8, T)) + 10 * antic * (1 - thrust)
        s["burst"] = thrust
        return s

    # ---------------------------------------------------------------- protocol
    def world(self, ctx, T, rect):
        T = live_T(T, self.CANON)
        s = self.state(T)
        x, y, w, h = rect
        white_rect(ctx, rect)
        b = s["burst"]
        if b < 1.0:
            # the camp in the rain: dripping shade-structure beam, grey sky, puddled mud horizon
            ctx.save()
            ctx.rectangle(x, y + 60, w, h)
            ctx.set_source_rgb(0.95, 0.95, 0.95)
            ctx.fill()
            ctx.rectangle(x, y + h * 0.72, w, h)
            ctx.set_source_rgb(0.86, 0.86, 0.86)
            ctx.fill()
            ctx.restore()
            poly_fill(ctx, [(x, y), (x + w, y), (x + w, y + 34), (x, y + 58)], grey(0.12))
            stroke(ctx, [(x, y + 58), (x + w, y + 34)], 3.0, taper=(0, 0))
            for i in range(9):
                px = x + 40 + i * 98 + 20 * hash01(i, 5)
                py = y + 58 - (px - x) / w * 24
                dy = ((T * 1.3 + hash01(i, 6)) % 1.0) * 60
                dots(ctx, [(px, py + 6 + dy)], 3.2)
            rain(ctx, (x, y + 40, w, h), T, n=90, alpha=0.85)
        if b > 0:
            cx, cy, k = self.CRW
            ctx.save()
            if b < 1:
                ctx.push_group()
            white_rect(ctx, rect)
            comic.burst_bg(ctx, 560 + 0.5 * s["crow_dx"], 600, rect, t=T - WRONG, rays=70, inner=130, bg=False)
            if b < 1:
                ctx.pop_group_to_source()
                ctx.paint_with_alpha(b)
            ctx.restore()
        # Summer
        sx, sy, sk = self.SUM
        summer(ctx, sx + s["sum_dx"], sy, sk, t=T, **s["sum"])
        if 18.62 <= T < 19.6:
            # the double take: '!?' pops over her head
            a = T - 18.62
            comic.emanata(ctx, sx + 30, sy - 132, 60 * min(1.0, a / 0.1), "exclaim", t=a)
            if a > 0.12:
                comic.emanata(ctx, sx + 78, sy - 124, 52 * min(1.0, (a - 0.12) / 0.1), "question", t=a)
        if T >= WRONG:
            u = T - WRONG
            comic.emanata(ctx, sx + 40, sy - 70, 150, "shock", t=u)
        # Crow
        if s["crow"]:
            cx, cy, k = self.CRW
            dx = s["crow_dx"]
            pop = s["pop"]
            if pop < 1:
                # he is simply THERE: a two-frame 'pop' with little impact ticks
                ctx.save()
                ctx.set_source_rgb(0, 0, 0)
                ctx.set_line_width(4)
                for i in range(10):
                    a = i / 10 * 2 * math.pi
                    r0 = 170 + 120 * pop
                    ctx.move_to(cx + math.cos(a) * r0, cy + 40 + math.sin(a) * r0)
                    ctx.line_to(cx + math.cos(a) * (r0 + 40), cy + 40 + math.sin(a) * (r0 + 40))
                ctx.stroke()
                ctx.restore()
            crow(ctx, cx + dx, cy, k, t=T, **s["crow_kw"])
            point_arm(ctx, cx + dx, cy, k, sh=(0.25, 1.85), hand=s["hand"], t=T, shake=s["burst"] * math.exp(-3 * max(0, T - WRONG)))

    def flags(self, ctx, T, rect):
        T = live_T(T, self.CANON)
        if T < APPEAR:
            return
        draw_track(ctx, self.flag(), T, T0)

    def letters(self, ctx, fc, T, rect):
        T = live_T(T, self.CANON)
        x, y, w, h = rect
        tr = self.flag()
        av = (lambda TT: comic.track_avoid(tr, (TT - T0) * FPS)) if T >= APPEAR else None
        bnd = (x + 6, y + 6, x + w - 6, y + h - 6)
        cx, cy, k = self.CRW
        hold = max(0.5, 25.35 - TL.line("P01")["end"])
        mouth = (cx - 0.52 * k - 8, cy + 0.66 * k)
        comic.say(ctx, fc, "P01", 470, 500, w=330, tail=mouth, size=L["balloon"], hold=hold, avoid=av, bounds=bnd, T=T)
        comic.say(ctx, fc, "P02", 458, 600, w=200, tail=mouth, size=L["shout"] * 1.25, hold=25.35 - TL.line("P02")["end"],
                  avoid=av, bounds=bnd, T=T)
        sx, sy, sk = self.SUM
        comic.say(ctx, fc, "S03", 225, 755, w=300, tail=(sx + 0.36 * sk, sy + 0.72 * sk), size=L["balloon"],
                  hold=25.35 - TL.line("S03")["end"], avoid=av, bounds=bnd, T=T)
        comic.say_footnote(ctx, fc, "P02", 868, y + h - 8, anchor="br", size=L["footnote"] * 1.1,
                           hold=25.35 - TL.line("P02")["end"], T=T)


# ============================================================ PANEL B (page 1, p3): the icon
class PanelB:
    """Crow as a Byzantine icon raising a plain yellow flag; the caption types on; the ** footnote."""
    RECT = P.PANELS["p3"]
    CANON = 33.40
    ICON_W = 452.0

    def __init__(self):
        from .t02_icon import Icon
        x, y, w, h = self.RECT
        self.icon = Icon((x, y, self.ICON_W, h))
        self._tr = None

    def flag(self):
        if self._tr is None:
            n = int((33.9 - T0) * FPS) + 2
            ic = self.icon
            bx, by = ic.pole_base

            def pole_fn(i):
                return (bx, by, ic.pole_ang + 0.004 * math.sin(i / 24 * 0.8))

            def wind_fn(i):
                T = T0 + i / FPS
                ess = smoothstep(ESSENCE - 0.1, ESSENCE + 0.3, T) * (1 - smoothstep(31.2, 33.0, T))
                # a steady heavenly breeze blowing LEFT (over the saint's head), rising on ESSENCE
                return (-(430 + 300 * ess + 70 * math.sin(i / 24 * 1.1)), -30.0, 0.0)

            self._tr = track("pB", n, pole_fn, wind_fn, ic.pole_len, seed=7, turbulence=0.8, flutter=0.9)
        return self._tr

    def radiance(self, T):
        r = smoothstep(ESSENCE - 0.04, ESSENCE + 0.30, T) * (1 - smoothstep(31.4, 32.6, T))
        return r * (0.8 + 0.2 * math.exp(-((T - ESSENCE - 0.35) / 0.35) ** 2))

    def gleam(self, T):
        for t0, dur in ((ICON - 0.18, 0.9), (ESSENCE - 0.05, 0.7)):
            if t0 <= T <= t0 + dur:
                return ease_in_out((T - t0) / dur)
        return None

    # ---------------------------------------------------------------- protocol
    def alt(self, T):
        return ("tract", dict(lpi=1080.0 / 2.4, angle=45.0), 1.0)

    def alt_mask(self, ctx, T, rect):
        x, y, w, h = self.RECT
        ctx.rectangle(x, y, self.ICON_W, h)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill()

    def world(self, ctx, T, rect):
        T = live_T(T, self.CANON)
        x, y, w, h = rect
        white_rect(ctx, rect)
        rad = self.radiance(T)
        if rad > 0:
            # the radiance spills out of the icon across the caption side of the panel (engraved rays)
            hx, hy, hr = self.icon.halo
            ctx.save()
            ctx.rectangle(x + self.ICON_W, y, w - self.ICON_W, h)
            ctx.clip()
            n = 90
            for i in range(n):
                a = i / n * 2 * math.pi + T * 0.04
                wid = math.pi / n * 0.22
                ctx.new_path()
                ctx.move_to(hx, hy)
                ctx.line_to(hx + math.cos(a - wid) * 1400, hy + math.sin(a - wid) * 1400)
                ctx.line_to(hx + math.cos(a + wid) * 1400, hy + math.sin(a + wid) * 1400)
                ctx.close_path()
                ctx.set_source_rgba(0, 0, 0, 0.9 * rad)
                ctx.fill()
            ctx.restore()
        mouth = mouth_of("PREACHER", T) * 0.6 if TL.speaking("PREACHER", T) else 0.0
        g = self.gleam(T)
        self.icon.draw(ctx, T, gleam=g, radiance=rad, mouth=mouth)
        if g is not None:
            # the silver leaf catches the light: glints travel round the halo's rim
            from .t02_manga import star4
            hx, hy, hr = self.icon.halo
            for j, off in enumerate((0.0, 2.1, 4.0)):
                a = -2.4 + 2.2 * g + off * 0.35
                s_ = math.sin(math.pi * clamp(g * 1.3 - j * 0.12)) * (16 - 3 * j)
                if s_ > 0.5:
                    star4(ctx, hx + math.cos(a) * hr, hy + math.sin(a) * hr, s_, rot=0.3 * g, lw=1.0)

    def flags(self, ctx, T, rect):
        T = live_T(T, self.CANON)
        rad = self.radiance(T)
        draw_track(ctx, self.flag(), T, T0, glow=0.15 + 0.85 * rad)

    def front(self, ctx, T, rect):
        self.icon.flag_hand(ctx, T)

    def front_kw(self, T):
        return dict(lpi=1080.0 / 2.4)

    def letters(self, ctx, fc, T, rect):
        T = live_T(T, self.CANON)
        x, y, w, h = rect
        tr = self.flag()
        av = lambda TT: comic.track_avoid(tr, (TT - T0) * FPS)
        cx0 = x + self.ICON_W
        bnd = (cx0 + 6, y + 6, x + w - 6, y + h - 6)
        hold = max(0.5, 33.45 - TL.line("P03")["end"])
        mid = cx0 + (x + w - cx0) / 2
        comic.say(ctx, fc, "P03", mid, y + 26, w=w - self.ICON_W - 60, anchor="t", size=25, hold=hold,
                  text="For ONE THING, the FLAGS are not a GAME.**", bounds=bnd, avoid=av, T=T)
        comic.say(ctx, fc, "P04", mid, y + 150, w=w - self.ICON_W - 50, anchor="t", size=27,
                  hold=max(0.5, 33.45 - TL.line("P04")["end"]), bounds=bnd, avoid=av, T=T)
        comic.say(ctx, fc, "D01", x + w - 14, y + h - 12, kind="footnote", anchor="br", size=L["footnote"] * 1.3,
                  w=w - self.ICON_W - 40,
                  hold=max(0.5, 33.45 - TL.line("D01")["end"]), bounds=(x + 6, y + 6, x + w - 6, y + h - 6),
                  avoid=av, T=T)


# ============================================================ PAGE 2, top row: anime Summer | Crow leans in
SNAP_BACK = 36.47
LEAN = 36.62
GLASSES = 37.10


class Page2Row:
    """ref page 24 panel 1: one wide panel split by a jagged diagonal: Summer (left) | Crow (right)"""
    RECT = P.PANELS["p1"]
    SPLIT = [(506.0, 50.0), (492.0, 118.0), (512.0, 150.0), (478.0, 236.0), (498.0, 268.0), (462.0, 422.0)]
    SUM = (292.0, 246.0, 100.0)
    MANGA = (300.0, 236.0, 116.0)
    CRW = (748.0, 250.0, 88.0)
    POLE = 520.0

    def __init__(self):
        self._tr = None

    def left_poly(self):
        x, y, w, h = self.RECT
        return [(x - 10, y - 10)] + self.SPLIT + [(x - 10, y + h + 10)]

    def right_poly(self):
        x, y, w, h = self.RECT
        return self.SPLIT + [(x + w + 10, y + h + 10), (x + w + 10, y - 10)]

    @staticmethod
    def anime(T):
        return ANIME <= T < SNAP_BACK

    def flag(self):
        if self._tr is None:
            n = int((T_END + 0.2 - CURL_START) * FPS) + 2

            def pole_fn(i):
                T = CURL_START + i / FPS
                lean = self.lean(T)
                return (900.0 - 24 * lean + 6 * math.sin(T * 1.1), 540.0, -0.09 - 0.05 * lean + 0.01 * math.sin(T * 0.9))

            def wind_fn(i):
                return (260 + 70 * math.sin(i / 24 * 0.6), -20.0, 0.0)

            self._tr = track("p2row", n, pole_fn, wind_fn, self.POLE, seed=33, turbulence=0.7, flutter=0.7)
        return self._tr

    # ---------------------------------------------------------------- acting
    @staticmethod
    def lean(T):
        u = clamp((T - LEAN) / 0.42)
        return ease_out_back(u, 1.4) if u > 0 else 0.0

    def crow_pose(self, T):
        cx, cy, k = self.CRW
        ln = self.lean(T)
        p6 = TL.line("P06")
        dream = smoothstep(p6["start"] - 0.1, p6["start"] + 0.5, T)
        beg0, beg1 = TL.word("P05", "from")[0], TL.word("P05", "beginning")[1]
        beg = smoothstep(beg0 - 0.08, beg0 + 0.15, T) * (1 - smoothstep(beg1, beg1 + 0.4, T))
        x = cx - 64 * ln + 10 * dream
        y = cy + 10 * ln - 6 * dream
        kk = k * (1 + 0.14 * ln)
        speaking = TL.speaking("PREACHER", T, pad=0.05)
        kw = dict(
            mouth_open=mouth_of("PREACHER", T) if speaking else 0.0,
            mshape="talk",
            brow=-0.2 + 0.9 * beg + 0.5 * dream - 0.3 * ln * (1 - dream),
            grin=0.35 * ln * (1 - dream) + 0.2 * (1 - ln),
            look=(0.9 - 0.9 * dream, 0.15 - 0.95 * dream),
            eye_open=1.0 + 0.35 * beg - 0.15 * dream,
            pupil=0.95,
            glint=1.0,
            glasses_down=smoothstep(GLASSES, GLASSES + 0.35, T),
            lens=0.62,
            tilt=-0.17 * ln + 0.10 * dream + 0.02 * math.sin(T * 1.3),
        )
        # hand: raised finger 'Here,' -> 'let me tell you' -> sweep toward the distance on P06
        up = (1.15, 0.15)
        far = (1.60, -0.50)
        rest = (0.70, 0.95)
        raise_ = smoothstep(36.70, 36.95, T)
        sweep = smoothstep(TL.word("P06", "started")[0], TL.word("P06", "back")[1], T)
        hx = rest[0] + (up[0] - rest[0]) * raise_
        hy = rest[1] + (up[1] - rest[1]) * raise_
        hx += (far[0] - hx) * sweep
        hy += (far[1] - hy) * sweep
        return x, y, kk, kw, (hx, hy), raise_

    def summer_kw(self, T):
        blink = 1.0 if (SNAP_BACK + 0.08 <= T <= SNAP_BACK + 0.16) else None
        return dict(look=(0.9, -0.05), eye_open=1.2, pupil=0.75, brow=0.8, smile=0.0, mouth_open=0.12,
                    mshape="o", sweat=1 if T < ANIME else 2, tilt=-0.04 + 0.015 * math.sin(T * 1.7),
                    lid=0.0 if blink is None else 0.0, eye_blink=blink)

    # ---------------------------------------------------------------- protocol
    def alt(self, T):
        if self.anime(T):
            return ("manga", dict(lpi=1080.0 / 1.9), 1.0)
        return None

    def alt_mask(self, ctx, T, rect):
        poly_fill(ctx, self.left_poly(), WHITE)

    def world(self, ctx, T, rect):
        x, y, w, h = rect
        white_rect(ctx, rect)
        # ---- left: Summer
        ctx.save()
        ctx.new_path()
        lp = self.left_poly()
        ctx.move_to(*lp[0])
        for p in lp[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.clip()
        if self.anime(T):
            ta = T - ANIME
            mx, my, mk = self.MANGA
            # screentone sky gradient + radial focus lines
            from vx.canvas import lin_grad
            ctx.set_source(lin_grad(0, y, 0, y + h, [(0, (0.72, 0.72, 0.72)), (0.55, (0.95, 0.95, 0.95)), (1, (1, 1, 1))]))
            ctx.paint()
            comic.speed_lines(ctx, (x - 20, y - 20, 470, h + 40), focus=(mx + 40, my + 10), t=ta, n=110,
                              inner=230, weight=0.7)
            from .t02_manga import manga_summer, sparkles
            mouth = mouth_of("SUMMER", T) if TL.speaking("SUMMER", T, pad=0.05) else 0.0
            blink = 1.0 if 35.19 <= T <= 35.27 else 0.0
            bob = 4 * math.sin(ta * 2.4)
            manga_summer(ctx, mx, my + bob, mk, t=ta, mouth=mouth, blink=blink, tilt=-0.06 + 0.03 * math.sin(ta * 1.6),
                         look=(0.5, -0.1 + 0.08 * math.sin(ta * 3)))
            sparkles(ctx, [(x + 40, y + 60, 26, 0.1), (x + 120, y + 300, 18, 0.5), (mx + 170, my - 120, 22, 0.8),
                           (mx - 190, my + 40, 16, 0.3), (mx + 150, my + 120, 14, 0.65), (x + 380, y + 330, 20, 0.2)],
                     ta, lw=1.3)
            if ta < 0.09:
                # impact frame: white flash with black focus lines
                white_rect(ctx, rect)
                comic.speed_lines(ctx, (x - 20, y - 20, 470, h + 40), focus=(mx + 40, my), t=0.0, n=160, inner=60,
                                  weight=1.6)
        else:
            ctx.rectangle(x, y + 40, 480, h)
            ctx.set_source_rgb(0.95, 0.95, 0.95)
            ctx.fill()
            poly_fill(ctx, [(x, y), (x + 480, y), (x + 480, y + 30), (x, y + 44)], grey(0.12))
            rain(ctx, (x, y + 40, 480, h), T, n=45, seed=5, alpha=0.8)
            sx, sy, sk = self.SUM
            kw = self.summer_kw(T)
            bl = kw.pop("eye_blink")
            if bl is not None:
                kw["eye_open"] = 0.05
            summer(ctx, sx, sy, sk, t=T, **kw)
            if SNAP_BACK <= T < SNAP_BACK + 0.12:
                # puff of 'snap back' ticks
                ctx.save()
                ctx.set_source_rgb(0, 0, 0)
                ctx.set_line_width(3)
                for i in range(12):
                    a = i / 12 * 2 * math.pi
                    ctx.move_to(sx + math.cos(a) * 190, sy + math.sin(a) * 170)
                    ctx.line_to(sx + math.cos(a) * 225, sy + math.sin(a) * 200)
                ctx.stroke()
                ctx.restore()
        ctx.restore()
        # ---- right: Crow leans in
        ctx.save()
        ctx.new_path()
        rp = self.right_poly()
        ctx.move_to(*rp[0])
        for p in rp[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.clip()
        ctx.rectangle(x + 400, y + 40, w, h)
        ctx.set_source_rgb(0.95, 0.95, 0.95)
        ctx.fill()
        poly_fill(ctx, [(x + 400, y), (x + w, y), (x + w, y + 26), (x + 400, y + 40)], grey(0.12))
        rain(ctx, (x + 420, y + 30, 460, h), T, n=45, seed=9, alpha=0.8)
        px, py, pk, kw, hand, raise_ = self.crow_pose(T)
        crow(ctx, px, py, pk, t=T, **kw)
        if raise_ > 0.02:
            point_arm(ctx, px, py, pk, sh=(0.30, 1.85), hand=hand, t=T)
        ctx.restore()
        # the jagged divider (the tract's cut between two photos)
        ctx.save()
        ctx.new_path()
        ctx.move_to(*self.SPLIT[0])
        for p in self.SPLIT[1:]:
            ctx.line_to(*p)
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(5.0)
        ctx.set_line_join(0)
        ctx.stroke()
        ctx.restore()

    def flags(self, ctx, T, rect):
        ctx.save()
        ctx.new_path()
        rp = self.right_poly()
        ctx.move_to(*rp[0])
        for p in rp[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.clip()
        draw_track(ctx, self.flag(), T, CURL_START)
        ctx.restore()

    def letters(self, ctx, fc, T, rect):
        x, y, w, h = rect
        bnd = (x + 6, y + 6, x + w - 6, y + h - 6)
        tr = self.flag()
        av = lambda TT: comic.track_avoid(tr, (TT - CURL_START) * FPS)
        sx, sy, sk = self.MANGA
        comic.say(ctx, fc, "S04", 480, 122, w=250, tail=(sx + 0.40 * sk, sy + 0.66 * sk), bounds=(x + 6, y + 6, 700, y + h),
                  size=L["balloon"] * 1.35, hold=max(0.5, 36.9 - TL.line("S04")["end"]), avoid=av, T=T)
        if ANIME <= T < ANIME + 0.9:
            comic.sfx(ctx, "キラッ", x + 70, y + 70, 44, rot=-0.25, t=T - ANIME, font="Droid Sans Fallback")
        px, py, pk, kw, hand, _ = self.crow_pose(T)
        mouth = (px - 0.52 * pk, py + 0.66 * pk)
        comic.say(ctx, fc, "P05", 640, 112, w=300, tail=mouth, size=L["balloon"] * 1.05, bounds=bnd, avoid=av,
                  hold=max(0.5, 41.5 - TL.line("P05")["end"]), T=T)
        if T < FLASH:
            comic.say(ctx, fc, "P06", *self.P06_AT, w=300, tail=mouth, size=L["balloon"] * 1.05, bounds=bnd,
                      avoid=av, hold=2.0, T=T)

    P06_AT = (575.0, 368.0)

    def p06_screen(self, ctx, fc, T):
        """during the dive into his lens the P06 balloon lets go of the page and stays put on screen"""
        cf = page2_cam(FLASH - 1e-3)
        bx, by = cf.to_screen(*self.P06_AT)
        cam = page2_cam(T)
        px, py, pk, kw, hand, _ = self.crow_pose(T)
        mx, my = cam.to_screen(px - 0.52 * pk, py + 0.66 * pk)
        mx = min(max(mx, 40.0), W - 40.0)
        my = min(max(my, 40.0), H - 40.0)
        return comic.say(ctx, fc, "P06", bx, by, w=300 * cf.zoom, tail=(mx, my), size=L["balloon"] * 1.05 * cf.zoom,
                         hold=0.5, T=T)


CURL_START = 33.30
PAGE2_FRAMES = [P.PANELS["p1"], P.PANELS["p2"], P.PANELS["p3"]]
PAGE2 = {"p1": Page2Row()}


def clamp_cam(cam):
    """keep the view inside the page (never show what lies beyond the paper)"""
    hw = W / 2 / cam.zoom
    cx = min(max(cam.cx, hw), P.PAGE_W - hw) if 2 * hw < P.PAGE_W else P.PAGE_W / 2
    return P.PageCam(cx, cam.cy, cam.zoom)


def page2_cam(T):
    """camera on page 2: Summer's half (anime), pan to Crow at the snap-back, slow push on P06"""
    cs = P.PageCam(352.0, 236.0, 2.75)
    cc = P.PageCam(650.0, 236.0, 2.75)
    u = seg(T, SNAP_BACK + 0.03, SNAP_BACK + 0.45, ease_in_out)
    cam = P.lerp_cam(cs, cc, u)
    # anime punch
    a = T - ANIME
    if a > 0 and T < SNAP_BACK:
        cam = P.PageCam(cam.cx + 10, cam.cy, cam.zoom * (1 + 0.07 * math.exp(-5 * a) + 0.02 * smoothstep(0, 2.7, a)))
    # P06: slow dreamy push toward his aviators ... and on 'Alchemy' we dive into his near lens (the flashback
    # is seen through his glasses; the flag leaves the frame on the way in)
    p6 = TL.line("P06")["start"]
    push = seg(T, p6 - 0.2, FLASH + 0.2, ease_in_out_sine)
    if push > 0:
        px, py, pk, kw, hand, _ = PAGE2["p1"].crow_pose(T)
        tx, ty = px - 0.3 * pk, py - 0.05 * pk
        cam = P.PageCam(cam.cx + (min(tx, 700) - cam.cx) * push, cam.cy + (ty - cam.cy) * push, cam.zoom * (1 + 0.32 * push))
        dive = seg(T, FLASH - 0.05, FLASH + 1.5, lambda u: u * u * (3 - 2 * u))
        if dive > 0:
            gd = kw.get("glasses_down", 0.0)
            c, s_ = math.cos(kw["tilt"]), math.sin(kw["tilt"])
            lx, ly = 0.21, 0.08 + 0.22 * gd                       # near lens centre, head units (facing right)
            lxs, lys = -(lx * c - ly * s_), lx * s_ + ly * c      # flipped (he faces left) + tilt
            gx, gy = px + lxs * pk, py + lys * pk
            cam = P.PageCam(cam.cx + (gx - cam.cx) * dive, cam.cy + (gy - cam.cy) * dive, cam.zoom * (1 + 2.6 * dive))
    return clamp_cam(cam)


PANELS = {"p2": PanelA(), "p3": PanelB()}

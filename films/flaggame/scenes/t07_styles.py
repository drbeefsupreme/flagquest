"""t07: the art-historical flag raisings. Every style is greyscale ink/paper; the Flag (vx.flag, spot plate)
is always the same plain yellow cloth on a plain wooden pole.  Owner: T07Raised.

A style draws in cell content units (see t07_cells): height 1000, x in [-hw, hw], centre x = 0.
Interface (all methods receive g: Geo, v: dict of per-raising params):
    layout(g, v) -> L dict (must contain pole=(bx, by, ang, length), cloth_right)
    world(ctx, g, v, L)        grey world
    prn(rgba, g, v, L) -> rgb  the style's print of the world (content-locked patterns)
    flags(ctx, g, v, L)        ONLY vx.flag drawings
    front(ctx, g, v, L)        optional: solid foreground objects over the pole (hands), printed with prn
v keys: t (s since the raise started; <0 before), T (global s), seed, k (raise number), still (bool: thumbnail,
no animation), wind (flag wind), light.
"""
import math

import cairo
import cv2
import numpy as np

from vx.ease import clamp, lerp, ease_out_back, ease_out, ease_in_out, hash01
from vx.flag import draw_flag
from vx import ink
from scenes.t07_soviet import worker, fist, BLK, LI, MID
from scenes import t07_figure as F
from scenes.t07_cells import (grey, P, curve, limb, ik2, pole_geom, halftone_cov, line_cov, cov_rgb, grey_rgb,
                              tone_of, noise, fbm, posterize, kuwahara, PAPER, INKC)

FONT_IMPACT = "Anton"
FONT_HEAVY = "Archivo Black"
FONT_SERIF = "C059"
FONT_PIXEL = "Pixelon"
FONT_JP = "Noto Serif CJK JP"
FONT_EGY = "Noto Sans Egyptian Hieroglyphs"


def raise_ang(t, a0, a1, dur=0.3, still=False):
    """pole angle during a raise: a0 (lowered) -> a1 with an overshoot"""
    if still:
        return a1
    return a0 + (a1 - a0) * ease_out_back(clamp(t / dur), 1.9)


def pop_of(v, dur=0.28):
    if "pop_t" in v:
        return clamp((v["pop_t"] + 0.04) / dur)
    if v.get("still"):
        return 1.0
    return clamp((v["t"] + 0.04) / dur)


def font(ctx, face, size, bold=False, italic=False):
    ctx.select_font_face(face, cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)


def ctext(ctx, s, x, y, face, size, g=0.0, align=0.5, bold=False, a=1.0):
    font(ctx, face, size, bold)
    e = ctx.text_extents(s)
    ctx.move_to(x - e.x_advance * align, y)
    grey(ctx, g, a)
    ctx.show_text(s)
    ctx.new_path()
    return e.x_advance


def mannequin(hip, H, lean=0.0, hand_f=None, hand_b=None, foot_f=None, foot_b=None, facing=1, tilt=0.0,
              sw=0.22, bend_arm=1, head_r=None):
    """simple figure skeleton (content units). hip = pelvis centre, H = standing height, lean (rad, + = toward
    facing). hands/feet: world targets (None = rest). returns joints dict"""
    f = facing
    ts = 0.30 * H
    ua, fa, th, sh = 0.18 * H, 0.17 * H, 0.245 * H, 0.245 * H
    hr = head_r or 0.068 * H
    neck = (hip[0] + f * math.sin(lean) * ts, hip[1] - math.cos(lean) * ts)
    head = (neck[0] + f * math.sin(lean + tilt) * hr * 1.35, neck[1] - math.cos(lean + tilt) * hr * 1.35)
    swx = sw * H * 0.5
    shf = (neck[0] + f * swx * 0.35, neck[1] + 0.03 * H)
    shb = (neck[0] - f * swx * 0.35, neck[1] + 0.03 * H)
    hpf = (hip[0] + f * 0.03 * H, hip[1])
    hpb = (hip[0] - f * 0.03 * H, hip[1])
    hand_f = hand_f or (shf[0] + f * 0.05 * H, shf[1] + ua + fa - 0.02 * H)
    hand_b = hand_b or (shb[0] - f * 0.03 * H, shb[1] + ua + fa - 0.02 * H)
    foot_f = foot_f or (hpf[0] + f * 0.06 * H, hip[1] + th + sh - 0.01 * H)
    foot_b = foot_b or (hpb[0] - f * 0.06 * H, hip[1] + th + sh - 0.01 * H)
    elf = ik2(shf, hand_f, ua, fa, bend=-f * bend_arm)
    elb = ik2(shb, hand_b, ua, fa, bend=-f * bend_arm)
    knf = ik2(hpf, foot_f, th, sh, bend=f)
    knb = ik2(hpb, foot_b, th, sh, bend=f)
    return dict(hip=hip, neck=neck, head=head, hr=hr, shf=shf, shb=shb, elf=elf, elb=elb, haf=hand_f, hab=hand_b,
                hpf=hpf, hpb=hpb, knf=knf, knb=knb, ftf=foot_f, ftb=foot_b, H=H, f=f)


def draw_body(ctx, J, gray_back, gray_front, w=1.0, head=True, torso_w=None):
    """silhouette figure: back limbs darker, then torso, front limbs"""
    H = J["H"] * w
    grey(ctx, gray_back)
    limb(ctx, J["hpb"], J["knb"], 0.085 * H, 0.065 * H); ctx.fill()
    limb(ctx, J["knb"], J["ftb"], 0.065 * H, 0.05 * H); ctx.fill()
    limb(ctx, J["shb"], J["elb"], 0.06 * H, 0.05 * H); ctx.fill()
    limb(ctx, J["elb"], J["hab"], 0.05 * H, 0.04 * H); ctx.fill()
    grey(ctx, gray_front)
    tw = torso_w or 0.13 * H
    limb(ctx, J["neck"], J["hip"], tw, tw * 0.85); ctx.fill()
    if head:
        ctx.new_sub_path(); ctx.arc(J["head"][0], J["head"][1], J["hr"], 0, 2 * math.pi); ctx.fill()
        limb(ctx, J["neck"], J["head"], 0.045 * H, 0.04 * H); ctx.fill()
    limb(ctx, J["hpf"], J["knf"], 0.09 * H, 0.068 * H); ctx.fill()
    limb(ctx, J["knf"], J["ftf"], 0.068 * H, 0.05 * H); ctx.fill()
    limb(ctx, J["shf"], J["elf"], 0.062 * H, 0.052 * H); ctx.fill()
    limb(ctx, J["elf"], J["haf"], 0.052 * H, 0.042 * H); ctx.fill()


def shade_poly(ctx, pts, g, a=1.0, smooth=False):
    (curve if smooth else P)(ctx, pts, close=True)
    grey(ctx, g, a)
    ctx.fill()


# =========================================================================== base
class Style:
    name = "base"
    bg = 1.0
    wind = 1.0
    light = None
    static_world = True        # world does not depend on t after the raise settles (cacheable in thumbnails)
    seed_off = 0

    def layout(self, g, v):
        raise NotImplementedError

    def world(self, ctx, g, v, L):
        pass

    def prn(self, rgba, g, v, L):
        return grey_rgb(tone_of(rgba, self.bg))

    def flags(self, ctx, g, v, L):
        bx, by, ang, ln = L["pole"]
        draw_flag(ctx, bx, by, pole=ln, t=v["T"] + (v.get("seed", 0) % 13) * 0.37, wind=L.get("wind", self.wind),
                  side=1, ang=ang, seed=v.get("seed", 0) + self.seed_off, pop=pop_of(v), light=L.get("light", self.light),
                  lean=0.0, glow=L.get("glow", 0.0))

    front = None


# =========================================================================== 1. SOVIET POSTER
class Soviet(Style):
    """low-angle heroic worker, sunburst, flat poster planes (ref page 26 panel 3)"""
    name = "soviet"
    wind = 1.15

    def layout(self, g, v):
        still = v.get("still")
        t = v["t"]
        if "ang" in v:
            a = v["ang"]
        elif t < 0:          # anticipation: lowered, a small wind-up dip just before the heave
            a = 0.9 + 0.12 * ease_in_out(clamp((t + 0.6) / 0.52)) - 0.02 * math.sin(t * 2.3)
        else:
            a = raise_ang(t, 1.02, 0.46, 0.34, still)
        fl = (60.0, 872.0)
        d_up = 300.0
        ln, d1 = 1150.0, 260.0
        sx, sy = math.sin(a), -math.cos(a)
        base = (fl[0] - d1 * sx, fl[1] - d1 * sy)
        top = (base[0] + ln * sx, base[1] + ln * sy)
        fu = (fl[0] + d_up * sx, fl[1] + d_up * sy)
        up = clamp((0.95 - a) / 0.49)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln, fl=fl, fu=fu, top=top,
                    tilt=-0.08 - 0.34 * up, a=a)

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        grey(ctx, 0.97); ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        # glory: rays from behind the Flag
        cx, cy = 470.0, 330.0
        n = 44
        rot = 0.0 if v.get("still") else v["T"] * 0.03
        for i in range(n):
            a0 = rot + i * 2 * math.pi / n
            a1 = a0 + math.pi / n * 0.9
            ctx.move_to(cx, cy)
            ctx.line_to(cx + 3000 * math.cos(a0), cy + 3000 * math.sin(a0))
            ctx.line_to(cx + 3000 * math.cos(a1), cy + 3000 * math.sin(a1))
            ctx.close_path()
        grey(ctx, 0.88); ctx.fill()
        # the tower at the left (ref)
        tx = -640.0
        for (w, y0, gg) in ((300, 700, 0.62), (230, 520, 0.62), (170, 390, 0.62), (110, 300, 0.62), (64, 235, 0.62)):
            grey(ctx, gg); ctx.rectangle(tx - w / 2, y0, w, 1100 - y0); ctx.fill()
            grey(ctx, 0.45); ctx.rectangle(tx + w * 0.12, y0, w * 0.38, 1100 - y0); ctx.fill()
            grey(ctx, 0.8); ctx.rectangle(tx - w / 2, y0, w, 10); ctx.fill()
        grey(ctx, 0.45); P(ctx, [(tx - 10, 236), (tx, 90), (tx + 10, 236)]); ctx.fill()
        grey(ctx, 0.3)
        for yy in range(560, 1000, 38):
            for xx in (-95, -60, -25, 10, 45, 80):
                ctx.rectangle(tx + xx, yy, 14, 20)
        ctx.fill()
        # marching crowd at the horizon: raised fists, banners-poles without cloth are NOT drawn (only one Flag)
        grey(ctx, 0.5)
        for i in range(34):
            x = -hw + i * 64 + (i % 3) * 9
            y = 880 + (i % 4) * 7
            ctx.new_sub_path(); ctx.arc(x, y - 70, 17, 0, 2 * math.pi)
            limb(ctx, (x, y - 50), (x + 4, y + 60), 44, 50)
            if i % 3 == 0:
                limb(ctx, (x + 14, y - 40), (x + 30, y - 110), 13, 11)
                ctx.new_sub_path(); ctx.arc(x + 31, y - 116, 11, 0, 2 * math.pi)
        ctx.fill()
        # factories with smoke at the right
        grey(ctx, 0.74)
        for (x, y, r) in ((560, 640, 70), (620, 600, 90), (700, 560, 110), (790, 520, 120)):
            ctx.new_sub_path(); ctx.arc(x, y, r, 0, 2 * math.pi)
        ctx.fill()
        grey(ctx, 0.4)
        P(ctx, [(320, 1100), (320, 820), (420, 820), (420, 760), (470, 760), (470, 820), (520, 780), (520, 700),
                (545, 700), (545, 790), (660, 740), (660, 650), (690, 650), (690, 760), (hw + 60, 760), (hw + 60, 1100)])
        ctx.fill()
        # the worker
        mouth = v.get("mouth", 0.0)
        br = 0.0 if v.get("still") else 3 * math.sin(v["T"] * 2.2)
        worker(ctx, L["fl"], L["fu"], L["tilt"], mouth=mouth, breathe=br)
        # slogan band
        grey(ctx, BLK)
        ctx.rectangle(-hw, 925, 2 * hw, 200); ctx.fill()
        ctext(ctx, "СЛАВА ФЛАГАМ!", -600, 988, "Nimbus Sans Narrow", 64, g=0.985, align=0.0, bold=True)

    def front(self, ctx, g, v, L):
        a = L["pole"][2]
        fist(ctx, L["fu"][0], L["fu"][1], a, 1.0, LI)
        fist(ctx, L["fl"][0], L["fl"][1], a, 1.08, MID)

    def flags(self, ctx, g, v, L):
        tr = v.get("track")
        if tr is not None:
            tr.draw(ctx, v["track_i"], light=(-0.2, -0.9, 0.45))
            return
        Style.flags(self, ctx, g, v, L)

    def prn(self, rgba, g, v, L):
        tone = tone_of(rgba, self.bg)
        X, Y = g.XY()
        kk = g.kk
        # hand-pulled poster: ragged plate edges, mottled ink, fine reproduction screen on the greys
        if kk > 0.3:
            dx = noise(X, Y, 9.0, 51) * 1.3 * kk
            dy = noise(X, Y, 9.0, 52) * 1.3 * kk
            h, w = tone.shape
            mx = (np.arange(w, dtype=np.float32)[None, :] + dx).astype(np.float32)
            my = (np.arange(h, dtype=np.float32)[:, None] + dy).astype(np.float32)
            tone = cv2.remap(tone.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        dark = np.clip(1 - tone, 0, 1)
        solid = dark > 0.86
        cov = halftone_cov(dark, X, Y, 7.0, 45, kk, min_px=2.6)
        mott = np.clip(fbm(X, Y, 50.0, 53, 3) * 0.25 + noise(X, Y, 2.0, 54) * 0.12, -1, 1)
        cov = np.where(solid, 1.0 - np.clip(mott, 0, 1) * 0.12, np.where(dark < 0.03, 0.0, cov))
        out = cov_rgb(cov)
        paper = 1 + fbm(X, Y, 25.0, 55, 2) * 0.012
        return out * paper[..., None]


# =========================================================================== 2. IWO JIMA (halftone photo)
class IwoJima(Style):
    name = "iwo"
    wind = 1.2

    def layout(self, g, v):
        a = raise_ang(v["t"], 1.15, 0.7, 0.36, v.get("still")) if "ang" not in v else v["ang"]
        ln = 860.0
        base = (-330.0, 905.0)
        top, at = pole_geom(base[0], base[1], a, ln)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln, at=at, a=a, top=top)

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        # sky: bright overcast, soft cumulus low on the horizon (photographic)
        ctx.set_source(_lg(0, 0, 0, 900, [(0, 0.86), (0.55, 0.95), (1, 0.975)]))
        ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        rng = np.random.default_rng(3 + v.get("seed", 0) % 5)
        for i in range(12):
            x = rng.uniform(-hw, hw); y = rng.uniform(560, 800); r = rng.uniform(90, 200)
            ctx.set_source(_rg(x, y, 0, r, [(0, 0.99, 0.8), (1, 0.99, 0.0)]))
            ctx.arc(x, y, r, 0, 2 * math.pi); ctx.fill()
        # the summit: rubble mound, broken sticks, the shadowed near slope
        grey(ctx, 0.36)
        curve(ctx, [(-hw - 50, 850), (-620, 830), (-430, 870), (-250, 900), (0, 925), (300, 950), (hw + 60, 975),
                    (hw + 60, 1100), (-hw - 60, 1100)], close=True)
        ctx.fill()
        grey(ctx, 0.2)
        curve(ctx, [(-hw - 50, 930), (-500, 950), (-100, 990), (300, 1010), (hw + 60, 1030), (hw + 60, 1100),
                    (-hw - 60, 1100)], close=True)
        ctx.fill()
        grey(ctx, 0.12); ctx.set_line_width(8)
        for i in range(26):
            x = -hw + (i * 137.3) % (2 * hw); y = 890 + (i * 37.1) % 90
            ctx.move_to(x, y); ctx.line_to(x + 30 + (i * 13) % 60, y - 30 + (i * 7) % 40)
        ctx.stroke()
        # six marines heaving the pole up (backlit silhouettes), after the photograph
        at = L["at"]
        a = L["a"]
        pole_dir = (math.sin(a), -math.cos(a))
        men = [  # hip (rel. to the pole point at d_hip), H, lean, hand distances, foot offsets
            (40, 330, 1.2, (30, 110), (80, -150)),
            (200, 400, 1.0, (190, 260), (60, -190)),
            (345, 410, 0.82, (340, 400), (70, -200)),
            (470, 420, 0.6, (470, 540), (60, -170)),
            (270, 380, 0.9, (275, 320), (30, -210)),
        ]
        self._men = []
        for i, (dh, H, lean, (d1, d2), (ff, fb)) in enumerate(men):
            px, py = at(dh)
            # the body stands below-left of the pole, leaning along it
            hip = (px - 0.36 * H - 30, py + 0.26 * H + 40)
            gy = 905 + (hip[0] + 330) * 0.03
            J = F.pose(hip, H, lean=lean, facing=1, hand_f=at(d2), hand_b=at(d1), foot_f=(hip[0] + ff, gy),
                       foot_b=(hip[0] + fb, gy + 8), tilt=-0.3, arm_bend=-1)
            gb = 0.13 + 0.05 * (i % 2)
            F.draw(ctx, J, gb * 0.8, gb, chest=1.1)
            self._helmet(ctx, J, gb * 0.7)
            # light catching the back and shoulder (separates the men)
            grey(ctx, 0.62, 0.9); ctx.set_line_width(6)
            b0 = F.torso(J)[5:9]
            ctx.move_to(*b0[0])
            for q in b0[1:]:
                ctx.line_to(*q)
            ctx.stroke()
            self._men.append(J)
        # the one on the far side of the pole, crouched at the base
        px, py = at(90)
        J = F.pose((px + 90, py - 20), 320, lean=-0.9, facing=-1, hand_f=at(150), hand_b=at(80),
                   foot_f=(px + 40, 912), foot_b=(px + 220, 918), tilt=0.2)
        F.draw(ctx, J, 0.1, 0.14)
        self._helmet(ctx, J, 0.1)

    def _helmet(self, ctx, J, gv):
        hx, hy = J["head"]
        r = J["hr"]
        grey(ctx, gv)
        ctx.save(); ctx.translate(hx, hy - r * 0.15); ctx.rotate(J["lean"] * J["f"] * 0.6)
        ctx.scale(1.3, 1.05); ctx.arc(0, 0, r * 1.12, math.pi, 2 * math.pi); ctx.close_path(); ctx.fill()
        ctx.restore()
        # rim light along the helmet top and the back (light from the upper right)
        grey(ctx, 0.55, 0.6); ctx.set_line_width(3)
        ctx.arc(hx, hy - r * 0.15, r * 1.3, -1.3, -0.4); ctx.stroke()

    def front(self, ctx, g, v, L):
        # the near marines' fists over the pole
        for J in getattr(self, "_men", [])[1:4]:
            for k in ("haf",):
                x, y = J[k]
                grey(ctx, 0.14); ctx.new_sub_path(); ctx.arc(x, y, J["H"] * 0.036, 0, 2 * math.pi); ctx.fill()

    def prn(self, rgba, g, v, L):
        tone = tone_of(rgba, self.bg)
        X, Y = g.XY()
        # photographic softness and film grain, then a coarse newspaper halftone
        r = max(0.5, 1.1 * g.kk)
        tone = cv2.GaussianBlur(tone, (0, 0), r)
        tone = tone + noise(X, Y, 2.0, 7) * 0.05 + fbm(X, Y, 90.0, 8, 3) * 0.04
        dark = np.clip((1 - tone) * 1.12 - 0.03, 0, 1)
        cov = halftone_cov(dark, X, Y, 10.0, 45, g.kk)
        # newsprint: slightly grey paper, a touch of ink spread
        out = cov_rgb(cov)
        return out * 0.97 + 0.02


# =========================================================================== 3. DELACROIX (grisaille oil)
class Delacroix(Style):
    name = "delacroix"
    wind = 1.0

    def layout(self, g, v):
        a = raise_ang(v["t"], -0.5, 0.06, 0.34, v.get("still"))
        ln = 660.0
        # pivot = her raised hand
        hand = (30.0, 340.0)
        d = 420.0
        base = (hand[0] - d * math.sin(a), hand[1] + d * math.cos(a))
        top, at = pole_geom(base[0], base[1], a, ln)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln, hand=hand, top=top)

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        # a dark, smoky sky torn open by a blaze of light behind Liberty (pyramid composition)
        ctx.set_source(_lg(0, Y0, 0, 1000, [(0, 0.18), (0.5, 0.35), (1, 0.2)]))
        ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        ctx.set_source(_rg(20, 380, 0, 620, [(0, 0.99, 1), (0.35, 0.9, 0.95), (0.7, 0.6, 0.5), (1, 0.4, 0)]))
        ctx.arc(20, 380, 620, 0, 2 * math.pi); ctx.fill()
        rng = np.random.default_rng(17)
        for i in range(22):                          # billowing gunsmoke
            a = rng.uniform(0, 2 * math.pi); r = rng.uniform(250, 620)
            x, y = 20 + math.cos(a) * r * 1.3, 420 + math.sin(a) * r * 0.7
            rr = rng.uniform(90, 200)
            gg = 0.55 + 0.4 * (1 - r / 620)
            ctx.set_source(_rg(x, y, 0, rr, [(0, gg, 0.9), (1, gg, 0.0)]))
            ctx.arc(x, y, rr, 0, 2 * math.pi); ctx.fill()
        grey(ctx, 0.62)                              # Notre-Dame's towers in the haze
        for tx in (560, 628):
            ctx.rectangle(tx, 460, 46, 150)
        ctx.fill()
        # the crowd: dark silhouettes with raised muskets, sabres and fists, left and right of her
        at_crowd = [(-620, 0.08), (-500, 0.12), (-380, 0.1), (280, 0.12), (400, 0.09), (520, 0.13), (660, 0.1)]
        for i, (x, gg) in enumerate(at_crowd):
            H = 430 + 60 * math.sin(i * 1.9)
            hip = (x, 960 - H * 0.52)
            up = (x + 60 * (1 if x < 0 else -1), hip[1] - H * 0.62)
            J = F.pose(hip, H, lean=0.12 if x < 0 else -0.12, facing=1 if x < 0 else -1, hand_f=up,
                       foot_f=(x + 70, 975), foot_b=(x - 70, 980))
            F.draw(ctx, J, gg, gg)
            grey(ctx, gg); ctx.set_line_width(9)
            ctx.move_to(*up); ctx.line_to(up[0] + 20 * (1 if x < 0 else -1), up[1] - 170); ctx.stroke()
        # the man in the top hat with his musket (left, lit)
        J = F.pose((-270, 640), 520, lean=0.1, hand_f=(-150, 600), hand_b=(-230, 660), foot_f=(-200, 975),
                   foot_b=(-360, 970), facing=1)
        F.draw(ctx, J, 0.1, 0.16)
        hx, hy = J["head"]
        grey(ctx, 0.05); ctx.rectangle(hx - 32, hy - 100, 64, 76); ctx.rectangle(hx - 50, hy - 30, 100, 12); ctx.fill()
        grey(ctx, 0.08); ctx.set_line_width(11); ctx.move_to(-340, 760); ctx.line_to(-80, 520); ctx.stroke()
        # the boy with two pistols (right)
        J = F.pose((210, 700), 360, lean=-0.05, hand_f=(330, 520), hand_b=(120, 600), foot_f=(270, 960),
                   foot_b=(160, 965), facing=1)
        F.draw(ctx, J, 0.08, 0.14)
        # LIBERTY: backlit, striding forward, the raised arm to the pole; dress with a luminous rim
        hand = L["hand"]
        J = F.pose((-40, 650), 660, lean=0.08, hand_f=hand, hand_b=(-190, 770), foot_f=(60, 975),
                   foot_b=(-150, 985), facing=1, tilt=-0.35)
        dress = [(J["shb"][0] - 14, J["shb"][1] + 10), (J["shf"][0] + 16, J["shf"][1]), (20, 620), (80, 760),
                 (130, 960), (40, 985), (-110, 990), (-230, 975), (-190, 800), (-120, 640)]
        grey(ctx, 0.2); curve(ctx, dress, close=True, tension=0.5); ctx.fill()
        F.draw(ctx, J, 0.14, 0.22, skip=("legf", "legb"))
        # rim light along her back and raised arm (the blaze is behind her)
        grey(ctx, 0.95, 0.9); ctx.set_line_width(7)
        ctx.move_to(*J["hab"]); ctx.line_to(*J["elb"]); ctx.line_to(*J["shb"])
        ctx.curve_to(J["shb"][0] - 30, J["shb"][1] + 120, -160, 760, -230, 975)
        ctx.stroke()
        ctx.set_line_width(5)
        ctx.move_to(*J["shf"]); ctx.line_to(*J["elf"]); ctx.line_to(*hand); ctx.stroke()
        hx, hy = J["head"]
        grey(ctx, 0.1)                                # Phrygian cap
        curve(ctx, [(hx - 44, hy - 4), (hx - 36, hy - 56), (hx + 10, hy - 72), (hx + 58, hy - 50), (hx + 44, hy - 10)],
              close=True); ctx.fill()
        grey(ctx, 0.9, 0.9); ctx.set_line_width(5)
        ctx.arc(hx, hy, J["hr"] + 2, math.pi * 0.9, math.pi * 1.6); ctx.stroke()
        # musket in her left hand
        grey(ctx, 0.06); ctx.set_line_width(12)
        ctx.move_to(-360, 900); ctx.line_to(-120, 720); ctx.stroke()
        # barricade and the fallen in the dark foreground
        grey(ctx, 0.06)
        curve(ctx, [(-hw - 50, 920), (-500, 950), (-200, 1000), (100, 975), (400, 935), (hw + 50, 900),
                    (hw + 50, 1100), (-hw - 50, 1100)], close=True)
        ctx.fill()
        grey(ctx, 0.3)
        limb(ctx, (-560, 990), (-260, 1010), 56, 46); ctx.fill()
        grey(ctx, 0.55)
        limb(ctx, (180, 995), (470, 965), 50, 40); ctx.fill()
        grey(ctx, 0.2); ctx.set_line_width(22)
        ctx.move_to(-720, 890); ctx.line_to(-480, 970); ctx.move_to(520, 940); ctx.line_to(780, 870); ctx.stroke()

    def front(self, ctx, g, v, L):
        hx, hy = L["hand"]
        grey(ctx, 0.84)
        ctx.new_sub_path(); ctx.arc(hx - 2, hy + 4, 22, 0, 2 * math.pi); ctx.fill()
        grey(ctx, 0.55)
        ctx.new_sub_path(); ctx.arc(hx - 10, hy + 10, 12, 0, 2 * math.pi); ctx.fill()

    def prn(self, rgba, g, v, L):
        tone = tone_of(rgba, self.bg)
        X, Y = g.XY()
        kk = g.kk
        # painterly: Kuwahara + directional brush-stroke texture + canvas weave + craquelure + varnish vignette
        if kk > 0.25:
            tone = kuwahara(tone.astype(np.float32), 7.0 * kk)
        strokes = noise(X * 0.35 + Y * 0.2, Y * 2.0 - X * 0.3, 7.0, 22) * 0.05 + noise(X, Y, 25.0, 24) * 0.03
        weave = (np.sin(X * 1.9) * np.sin(Y * 1.9)) * 0.018 if kk > 0.5 else 0.0
        crack = np.abs(fbm(X, Y, 45.0, 23, 3))
        crack = np.clip(1 - crack / 0.012, 0, 1) * 0.08 if kk > 0.3 else 0.0
        vign = np.clip(((X / (g.hw + 1)) ** 2 * 0.6 + ((Y - 480) / 620.0) ** 2), 0, 1)
        t = tone + strokes + weave - crack
        t = t * (1 - 0.35 * vign)
        t = np.clip(t, 0, 1) ** 1.05
        return grey_rgb(0.04 + t * 0.94)


# =========================================================================== 4. MOON LANDING (TV broadcast)
class Moon(Style):
    name = "moon"
    wind = 0.35          # "it's alive": the Flag flutters even in a vacuum
    bg = 0.04

    def layout(self, g, v):
        a = raise_ang(v["t"], 0.5, -0.04, 0.4, v.get("still"))
        ln = 560.0
        base = (40.0, 900.0)
        top, at = pole_geom(base[0], base[1], a, ln)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln, at=at, top=top,
                    light=(-0.8, -0.4, 0.5))

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        grey(ctx, 0.03); ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        # the Earth, a crescent
        ex, ey = -240, 250
        grey(ctx, 0.75); ctx.arc(ex, ey, 46, 0, 2 * math.pi); ctx.fill()
        grey(ctx, 0.03); ctx.arc(ex + 18, ey - 6, 44, 0, 2 * math.pi); ctx.fill()
        # lunar surface
        ctx.set_source(_lg(0, 600, 0, 1000, [(0, 0.5), (1, 0.8)]))
        curve(ctx, [(-hw - 50, 640), (-400, 610), (0, 600), (400, 612), (hw + 50, 640), (hw + 50, 1100),
                    (-hw - 50, 1100)], close=True)
        ctx.fill()
        rng = np.random.default_rng(11)
        for i in range(26):
            x = rng.uniform(-hw, hw); y = rng.uniform(640, 1000); r = rng.uniform(10, 60) * (y - 560) / 400
            ctx.save(); ctx.translate(x, y); ctx.scale(1, 0.32)
            grey(ctx, 0.35); ctx.arc(0, 0, r, 0, 2 * math.pi); ctx.fill()
            grey(ctx, 0.85); ctx.arc(r * 0.15, r * 0.2, r * 0.85, 0, math.pi); ctx.fill()
            ctx.restore()
        # footprints
        grey(ctx, 0.4)
        for i in range(10):
            x = -520 + i * 55; y = 900 - i * 12
            ctx.save(); ctx.translate(x, y + (i % 2) * 18); ctx.scale(1, 0.4); ctx.arc(0, 0, 14, 0, 2 * math.pi)
            ctx.restore(); ctx.fill()
        # the lunar module (left)
        lx, ly = -560, 640
        grey(ctx, 0.62); P(ctx, [(lx - 110, ly - 60), (lx + 110, ly - 60), (lx + 90, ly + 20), (lx - 90, ly + 20)]); ctx.fill()
        grey(ctx, 0.4); P(ctx, [(lx - 10, ly - 60), (lx + 110, ly - 60), (lx + 90, ly + 20), (lx - 10, ly + 20)]); ctx.fill()
        grey(ctx, 0.8); P(ctx, [(lx - 70, ly - 60), (lx - 60, ly - 150), (lx + 20, ly - 170), (lx + 70, ly - 120),
                              (lx + 60, ly - 60)]); ctx.fill()
        grey(ctx, 0.15); P(ctx, [(lx - 20, ly - 140), (lx + 10, ly - 145), (lx + 8, ly - 120), (lx - 18, ly - 118)]); ctx.fill()
        grey(ctx, 0.7); ctx.set_line_width(8)
        for sx in (-1, 1):
            ctx.move_to(lx + sx * 80, ly); ctx.line_to(lx + sx * 150, ly + 70)
        ctx.stroke()
        # astronaut: bulky suit, backpack, gold... no: GREY visor
        at = L["at"]
        hf, hb = at(330), at(230)
        J = mannequin((-110, 700), 430, lean=0.12, hand_f=(hf[0] - 14, hf[1]), hand_b=(hb[0] - 18, hb[1]),
                      foot_f=(-50, 900), foot_b=(-170, 895), facing=1, head_r=44)
        grey(ctx, 0.6)                                  # backpack
        ctx.rectangle(J["neck"][0] - 115, J["neck"][1] - 20, 70, 150); ctx.fill()
        draw_body(ctx, J, 0.62, 0.86, w=1.35, torso_w=110)
        hx, hy = J["head"]
        grey(ctx, 0.9); ctx.arc(hx, hy, 50, 0, 2 * math.pi); ctx.fill()
        grey(ctx, 0.1)                                  # visor
        ctx.save(); ctx.translate(hx + 10, hy + 4); ctx.scale(1.0, 0.8); ctx.arc(0, 0, 36, 0, 2 * math.pi)
        ctx.restore(); ctx.fill()
        grey(ctx, 0.75); ctx.arc(hx + 20, hy - 6, 9, 0, 2 * math.pi); ctx.fill()
        grey(ctx, 0.55); ctx.set_line_width(5)     # suit seams / hoses
        ctx.move_to(J["neck"][0], J["neck"][1] + 40); ctx.curve_to(J["neck"][0] + 60, J["neck"][1] + 80,
                                                                   J["hip"][0] + 40, J["hip"][1] - 60,
                                                                   J["hip"][0] + 20, J["hip"][1] - 10)
        ctx.stroke()

    def front(self, ctx, g, v, L):
        at = L["at"]
        for d in (330, 230):
            x, y = at(d)
            grey(ctx, 0.8); ctx.new_sub_path(); ctx.arc(x - 2, y, 20, 0, 2 * math.pi); ctx.fill()
            grey(ctx, 0.45); ctx.new_sub_path(); ctx.arc(x - 8, y + 6, 11, 0, 2 * math.pi); ctx.fill()

    def prn(self, rgba, g, v, L):
        tone = tone_of(rgba, self.bg)
        X, Y = g.XY()
        kk = g.kk
        # 1969 slow-scan TV: soft, blooming highlights, crushed blacks, ghost, scanlines, snow
        r = max(0.7, 3.0 * kk)
        t = cv2.GaussianBlur(tone, (0, 0), r)
        gh = cv2.GaussianBlur(tone, (0, 0), r * 3)
        sh = int(round(18 * kk))
        if sh > 0:
            gh = np.roll(gh, sh, axis=1)
        t = t * 0.9 + gh * 0.14
        t = np.clip((t - 0.06) * 1.25, 0, 1) ** 0.85
        if kk > 0.3:
            scan = 0.5 + 0.5 * np.cos(Y * (2 * math.pi / 5.0))
            t = t * (0.8 + 0.2 * scan)
        roll = 0.0 if v.get("still") else math.sin(v["T"] * 3.1) * 0.5
        t = t + noise(X + roll * 300, Y, 1.6, 31 + int(v["T"] * 24) % 7) * 0.06
        t = t * (1 - 0.25 * np.clip(((X / (g.hw + 1)) ** 2 + ((Y - 500) / 600.0) ** 2) - 0.3, 0, 1))
        return grey_rgb(np.clip(t, 0, 1))


# =========================================================================== 12. BLUEPRINT
class Blueprint(Style):
    name = "blueprint"
    bg = 0.2
    wind = 0.8

    def layout(self, g, v):
        a = raise_ang(v["t"], -1.2, 0.0, 0.5, v.get("still"))
        ln = 600.0
        base = (-40.0, 880.0)
        top, at = pole_geom(base[0], base[1], a, ln)
        return dict(pole=(base[0], base[1], a, ln), cloth_right=top[0] + 0.375 * ln + 90, at=at, top=top, a=a)

    def world(self, ctx, g, v, L):
        hw, Y0 = g.ext()
        grey(ctx, 0.2); ctx.rectangle(-hw, Y0, 2 * hw, 1200 - Y0); ctx.fill()
        # grid
        ctx.set_line_width(1.2); grey(ctx, 0.27)
        for x in range(-2000, 2000, 50):
            ctx.move_to(x, -100); ctx.line_to(x, 1100)
        for y in range(0, 1001, 50):
            ctx.move_to(-2000, y); ctx.line_to(2000, y)
        ctx.stroke()
        ctx.set_line_width(2.2); grey(ctx, 0.31)
        for x in range(-2000, 2000, 250):
            ctx.move_to(x, -100); ctx.line_to(x, 1100)
        for y in range(0, 1001, 250):
            ctx.move_to(-2000, y); ctx.line_to(2000, y)
        ctx.stroke()
        W = 0.92
        bx, by, a, ln = L["pole"]
        # ground line + hatching
        grey(ctx, W); ctx.set_line_width(4)
        ctx.move_to(-hw, by); ctx.line_to(hw, by); ctx.stroke()
        ctx.set_line_width(2)
        for x in range(int(-hw) // 30 * 30, int(hw), 30):
            ctx.move_to(x, by); ctx.line_to(x - 20, by + 22)
        ctx.stroke()
        # raising arc (dashed) + ghost poles at 30 / 60 degrees
        ctx.set_dash([14, 10]); ctx.set_line_width(2.5)
        ctx.arc(bx, by, ln, -math.pi, -math.pi / 2); ctx.stroke()
        for gd in (-60, -30):
            ga = math.radians(gd)
            ctx.move_to(bx, by); ctx.line_to(bx + ln * math.sin(ga), by - ln * math.cos(ga))
        ctx.stroke()
        ctx.set_dash([])
        # arrowhead on the arc
        ax, ay = bx - ln * math.sin(math.radians(20)), by - ln * math.cos(math.radians(20))
        P(ctx, [(ax, ay), (ax - 24, ay - 12), (ax - 20, ay + 14)]); ctx.fill()
        ctext(ctx, "θ = 90°", bx - 420, by - 330, FONT_SANS, 30, g=W)
        # the figure (patent-drawing outline)
        top, at = pole_geom(bx, by, a, ln)
        J = mannequin((-150, 640), 440, lean=0.1, hand_f=at(330), hand_b=at(230), foot_f=(-100, by),
                      foot_b=(-220, by), facing=1)
        ctx.set_line_width(3.5); grey(ctx, W)
        for s, e in (("hpb", "knb"), ("knb", "ftb"), ("shb", "elb"), ("elb", "hab"), ("neck", "hip"),
                     ("hpf", "knf"), ("knf", "ftf"), ("shf", "elf"), ("elf", "haf"), ("shf", "shb"), ("hpf", "hpb")):
            ctx.move_to(*J[s]); ctx.line_to(*J[e])
        ctx.stroke()
        ctx.arc(J["head"][0], J["head"][1], J["hr"], 0, 2 * math.pi); ctx.stroke()
        for k in ("shf", "shb", "elf", "elb", "hpf", "hpb", "knf", "knb", "haf", "hab"):
            ctx.new_sub_path(); ctx.arc(J[k][0], J[k][1], 7, 0, 2 * math.pi)
        ctx.stroke()
        # dimensions (kept clear of the cloth)
        ctx.set_line_width(2.2)
        ox = bx - 60
        ctx.move_to(ox, by); ctx.line_to(ox, by - ln); ctx.stroke()
        for yy in (by, by - ln):
            ctx.move_to(ox - 14, yy); ctx.line_to(ox + 14, yy)
        ctx.stroke()
        ctx.save(); ctx.translate(ox - 16, by - ln * 0.5); ctx.rotate(-math.pi / 2)
        ctext(ctx, '96"', 0, 0, FONT_SANS, 30, g=W); ctx.restore()
        cw, chh = 0.375 * ln, 0.3125 * ln
        tyy = by - ln - 70
        ctx.move_to(bx, tyy); ctx.line_to(bx + cw, tyy); ctx.stroke()
        for xx in (bx, bx + cw):
            ctx.move_to(xx, tyy - 12); ctx.line_to(xx, tyy + 12)
        ctx.stroke()
        ctext(ctx, '36"', bx + cw / 2, tyy - 14, FONT_SANS, 30, g=W)
        # labels
        ctext(ctx, "FIG. 1", -hw * 0.55, 560, FONT_SERIF, 46, g=W)
        ctext(ctx, "FLAG, PLAIN (YELLOW)", bx + cw + 170, by - ln + 40, FONT_SANS, 24, g=W, align=0.0)
        ctx.move_to(bx + cw + 165, by - ln + 32); ctx.line_to(bx + cw + 60, by - ln + 60); ctx.stroke()
        ctext(ctx, "POLE, WOOD, UNTREATED", bx + 60, by - 150, FONT_SANS, 24, g=W, align=0.0)
        # title block (bottom-left)
        tx0, ty0 = -hw + 40, 780
        ctx.set_line_width(3)
        ctx.rectangle(tx0, ty0, 520, 170); ctx.stroke()
        ctx.set_line_width(1.6)
        for yy in (ty0 + 50, ty0 + 90, ty0 + 130):
            ctx.move_to(tx0, yy); ctx.line_to(tx0 + 520, yy)
        ctx.move_to(tx0 + 300, ty0 + 90); ctx.line_to(tx0 + 300, ty0 + 170)
        ctx.stroke()
        ctext(ctx, "IVG  —  FLAG, RAISING OF", tx0 + 16, ty0 + 36, FONT_SANS, 28, g=W, align=0.0, bold=True)
        ctext(ctx, "DWG. NO. FG-69   REV. 100", tx0 + 16, ty0 + 78, FONT_SANS, 22, g=W, align=0.0)
        ctext(ctx, f"SHEET {int(v.get('k', 1))} OF 100", tx0 + 16, ty0 + 118, FONT_SANS, 22, g=W, align=0.0)
        ctext(ctx, "DRN: F. MAKER", tx0 + 16, ty0 + 158, FONT_SANS, 22, g=W, align=0.0)
        ctext(ctx, "SCALE 1:1", tx0 + 316, ty0 + 118, FONT_SANS, 22, g=W, align=0.0)
        ctext(ctx, "NO MARKINGS", tx0 + 316, ty0 + 158, FONT_SANS, 22, g=W, align=0.0, bold=True)

    def prn(self, rgba, g, v, L):
        tone = tone_of(rgba, self.bg)
        X, Y = g.XY()
        # diazo print: slightly mottled dark ground, lines bleed a hair
        t = tone + fbm(X, Y, 120.0, 41, 3) * 0.03 + noise(X, Y, 2.5, 42) * 0.015
        if g.kk > 0.4:
            t = np.maximum(t, cv2.GaussianBlur(t, (0, 0), 1.2 * g.kk) * 0.9)
        return grey_rgb(np.clip(t, 0, 1))


FONT_SANS = "Nimbus Sans"


def _lg(x0, y0, x1, y1, stops):
    gr = cairo.LinearGradient(x0, y0, x1, y1)
    for s in stops:
        gr.add_color_stop_rgba(s[0], s[1], s[1], s[1], s[2] if len(s) > 2 else 1.0)
    return gr


def _rg(cx, cy, r0, r1, stops):
    gr = cairo.RadialGradient(cx, cy, r0, cx, cy, r1)
    for s in stops:
        gr.add_color_stop_rgba(s[0], s[1], s[1], s[1], s[2] if len(s) > 2 else 1.0)
    return gr


STYLES = {}


def register(*classes):
    for c in classes:
        STYLES[c.name] = c()


register(Soviet, IwoJima, Delacroix, Moon, Blueprint)

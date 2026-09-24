"""CHARACTERS (v1 by RigMaster). Procedural 2D rigs with a 3D-lite skeleton, cel-shaded flat vector look.

Character(kind, seed=0, **style)
  kinds: "lutie" (novice), "crocus" (master), "schismmancer", "commander", "citizen", "cultist",
         "philosopher", "jaguar" (President Jaguar), "acolyte" (generic hooded ochre Vexillian),
         "crackpot" (wild-haired philosopher)
  style (constructor or per-draw kw): build=0.7..1.4, skin/hair/robe/iris/trouser/shoe/... colour overrides,
         hood="up"|"down"|None, crown=True (+crown_color, default silver), phone=True (+phone_color),
         prop="phone"|"megaphone"|"pen"|"candle"|None (held in the FRONT hand), goggles_down=True,
         headset=True, laurel=True, beard="long"|"full"|"stubble"|None, hair_style=..., garment=...

  .anchors(x, y, h, pose="stand", t=0.0, facing=1, pole_len=None, **kw) -> dict      (pure, no drawing)
  .draw(ctx, x, y, h, pose="stand", t=0.0, mouth=0.0, look=(0, 0), expr="neutral", facing=1,
        blink=None, alpha=1.0, pole_len=None, **kw) -> anchors dict  (identical to .anchors(...))
      (x, y) = ground point between the feet, h = full standing height in design px (any ctx transform ok)
      mouth  = 0..1 lip-sync openness (use tl.mouth(SPEAKER, fc.T)); drives jaw, head bob
      look   = (-1..1, -1..1) gaze direction in WORLD screen space (+x right, +y down); head follows a bit
      expr   = one of EXPRS or a dict {expr: weight}
      facing = +1 faces right, -1 faces left, 0 = FRONT view (addresses camera)
      blink  = None (auto blinks), True/False, or 0..1 lid closure
      pose   = one of POSES, or a dict {pose: weight} (per-channel blend: legs/spine/head/arms)
  kw:
      turn=0..1 (0 front, 0.45 default 3/4, 1 profile, up to 2 = back view), back=True (= turn 2),
      head_turn= (override head yaw in turn units), lean= (rad, + forward), nod= (head pitch, + up),
      tilt= (head roll), speed= (walk/run), phase= (cycle 0..1 override; 0 = FRONT(near) heel strike,
      0.5 = back heel strike), progress= (plant_flag: impact at PLANT_IMPACT=0.6; slam impact 0.6; leap 0..1),
      wind= float or (wx, wy) (+x blows right; 1 strong, 2 gale) whips hem/sleeves/hair/beard/veil,
      light=(dx, dy) toward key light (default (-0.5,-0.8)), shade=0..1, rim=(colour, strength[, (dx,dy)]),
      tint=(colour, amount), silhouette=colour (flat single colour; rim still applies),
      layer="all"|"back"|"front"  ("back" = everything but the front arm/hand/prop; "front" = only those),
      hand_r=(x, y) / hand_l=(x, y) world IK targets (hand_r = FRONT hand = pole hand; hand_l = back hand),
      shape_r / shape_l = "relax"|"fist"|"grip"|"point"|"open"|"pinch"|"cup"|"finger",
      wink="near"|"far", lod=0|1|2 (auto from device size), eyes_closed=True
  anchors keys: head, eyes, mouth, chest, neck, hip, hand_r, hand_l, hand_front, hand_back, elbow_r, elbow_l,
                shoulder_r, shoulder_l, tip_r, tip_l (fingertip / pen tip), feet, foot_r, foot_l, top, head_r,
                pocket (breast pocket), ear, prop, bbox=(x0,y0,x1,y1), scale (= h/100)
                pole -> (base_x, base_y, ang) for flag poses (pass to Flag.simulate pole_fn); pole_top if pole_len.
  Holding a flag:   ch.draw(ctx, x, y, h, "hold_flag", t, layer="back"); track.draw(ctx, i);
                    ch.draw(ctx, x, y, h, "hold_flag", t, layer="front")
  .ground_speed(h, pose="walk", speed=1.0) -> design px/s that keeps feet from sliding
  .footsteps(t0, t1, pose="walk", speed=1.0) -> [(t, "near"|"far")] heel-strike times (phase from t)

draw_crowd(ctx, people, t) -> list of anchors-lite dicts (pole, hand_r, head, top) per person
  people = [dict(x, y, h, kind="citizen", seed, pose, facing, look, mouth, expr, crown, phone, prop, alpha,
                 tint, silhouette, rim, t_off, speed, turn)]
  Figures under ~90 device px use a fast LOD path (a handful of fills, cached rigs); larger use .draw.
"""
import math

import cairo

from .canvas import col
from .ease import hash01, clamp
from . import chars_rig as R
from .chars_rig import POSE_FNS, EXPR_TABLE, PLANT_IMPACT, solve, make_spec, blend_expr  # noqa: F401 (PLANT_IMPACT re-exported)
from .chars_paint import (Painter, HP, egg_of, p_circle, p_ellipse, p_capsule, p_chain, p_poly, p_smooth, shadow_of, light_of,
                          mixc, TAU)
from .chars_kinds import (build_palette, _hp, _hood, _beard, _mustache, _stubble, _crown, _laurel, _helmet,
                          _headset, jaguar_head, jaguar_tail, age_lines)
from . import chars_ink as K2

POSES = tuple(POSE_FNS.keys())
EXPRS = tuple(EXPR_TABLE.keys())
KINDS = ("lutie", "crocus", "schismmancer", "commander", "citizen", "cultist", "philosopher", "jaguar",
         "acolyte", "crackpot",
         # film 2 (THE FLAG GAME?)
         "summer", "raven", "crow", "flagmaker", "hippie", "pharisee", "devil", "seraph")
TRACT_KINDS = ("summer", "raven", "crow", "flagmaker", "hippie", "pharisee", "devil", "seraph")
_STYLE_KW = ("crown", "phone", "prop", "goggles_down", "headset", "hood", "laurel", "beard", "crown_color",
             "phone_color", "hat", "glasses", "earrings", "collar", "goggles_neck", "goggles_head", "apron", "wings",
             "horns", "freckles", "face_paint", "lipstick", "hair_style", "garment", "face")


def lk_face(ch):
    return ch.lk.get("face")


def _blink_amount(t, seed):
    """deterministic auto blink: 0 open .. 1 closed"""
    per = 3.4
    k = math.floor((t + seed * 0.731) / per)
    b = 0.0
    for kk in (k - 1, k, k + 1):
        t0 = kk * per - seed * 0.731 + 0.4 + hash01(kk, 900 + seed) * 2.4
        x = (t - t0) / 0.075
        b = max(b, math.exp(-x * x))
        if hash01(kk, 901 + seed) < 0.18:  # occasional double blink
            x = (t - t0 - 0.24) / 0.075
            b = max(b, math.exp(-x * x))
    return b


class Character:
    def __init__(self, kind, seed=0, **style):
        self.kind = kind if kind in KINDS else "citizen"
        self.seed = int(seed)
        self.style = dict(style)
        self.sp = make_spec(self.kind, self.seed, style)
        self.pal, self.lk = build_palette(self.kind, self.seed, style)
        self.render = style.get("render", "color")
        self._ipal = None
        self._variants = {}

    def ink_pal(self):
        if self._ipal is None:
            self._ipal = K2.ink_palette(self.kind, self.pal)
        return self._ipal

    # ------------------------------------------------------------------ helpers
    def _with_style(self, kw):
        """per-draw style kw (crown=, prop=, goggles_down=...) -> variant Character"""
        keys = tuple(sorted((k, kw[k]) for k in _STYLE_KW if k in kw and self.style.get(k) != kw[k]))
        if not keys:
            return self
        v = self._variants.get(keys)
        if v is None:
            st = dict(self.style)
            st.update(dict(keys))
            v = Character(self.kind, self.seed, **st)
            self._variants[keys] = v
        return v

    def _setup(self, x, y, h, pose, t, facing, kw, mouth, look):
        fs = -1 if facing < 0 else 1
        if "turn" in kw:
            turn = kw["turn"]
        elif kw.get("back") or (pose == "walk_away" or (isinstance(pose, dict) and pose.get("walk_away", 0) > 0.5)):
            turn = 2.0
        else:
            turn = 0.0 if facing == 0 else R.PHI_DEFAULT
        turn = max(-0.4, min(2.0, turn))
        phi = turn * math.pi / 2
        cphi, sphi = math.cos(phi), math.sin(phi)
        k = h / 100.0
        kw2 = dict(kw)
        for key, lk in (("hand_r", "_hf_xy"), ("hand_front", "_hf_xy"), ("hand_l", "_hb_xy"), ("hand_back", "_hb_xy")):
            if kw.get(key) is not None:
                X, Y = kw[key]
                kw2[lk] = ((X - x) / k * fs, (Y - y) / k)
        w = kw.get("wind", 0.0)
        wx = w[0] if isinstance(w, (tuple, list)) else float(w)
        rig = solve(self.sp, self.seed, pose, t, kw2, cphi, sphi, mouth=mouth, wind=wx)
        lx = look[0] * fs
        if "head_turn" in kw:
            hy = kw["head_turn"] * math.pi / 2
        else:
            hy = phi + lx * 0.22 * (1.0 if turn < 1.2 else -1.0)
            if turn <= 0.75:
                hy = min(hy, 0.95 + max(0.0, turn - 0.5) * 1.2)
            if self.kind in TRACT_KINDS and turn < 1.35:
                hy = min(hy, 1.18)     # the cartoonist's cheat: a "profile" still shows the far eye
        hy = max(-0.9, min(math.pi, hy))
        return fs, k, rig, cphi, sphi, phi, hy, wx * fs

    def ground_speed(self, h, pose="walk", speed=1.0):
        if pose == "run":
            return 4 * 14.0 * speed ** 0.6 * 1.35 * speed ** 0.5 * h / 100.0
        return 4 * 8.0 * speed ** 0.6 * 0.95 * speed ** 0.5 * h / 100.0

    def footsteps(self, t0, t1, pose="walk", speed=1.0):
        f = (1.35 if pose == "run" else 0.95) * speed ** 0.5
        out = []
        n0 = math.floor(t0 * f * 2) - 1
        n1 = math.ceil(t1 * f * 2) + 1
        for n in range(n0, n1 + 1):
            tt = n / (2 * f)
            if t0 <= tt < t1:
                out.append((tt, "near" if n % 2 == 0 else "far"))
        return out

    # ------------------------------------------------------------------ anchors
    def _anchors(self, x, y, fs, k, rig, hy, pole_len, look):
        P = rig.P
        sp = self.sp
        Wp = lambda p: (x + fs * k * p[0], y + k * p[1])  # noqa: E731
        hr = sp["head_r"]
        hx, hyy = P["head"]
        hp = HP(hr, hy, rig.head_pitch * 0.7 + look[1] * -0.18, egg=egg_of(self, rig))
        cr, sr = math.cos(rig.head_rot), math.sin(rig.head_rot)

        def HW(q):
            return Wp((hx + q[0] * cr - q[1] * sr, hyy + q[0] * sr + q[1] * cr))
        e1, e2 = hp.pt(-sp["eye_sep"], sp["eye_y"]), hp.pt(sp["eye_sep"], sp["eye_y"])
        vis = [e for e in (e1, e2) if e[2] > 0.1] or [hp.pt(0, sp["eye_y"])]
        ex = sum(e[0] for e in vis) / len(vis)
        ey = sum(e[1] for e in vis) / len(vis)
        mo = hp.pt(0.0, -0.5)
        ear = hp.pt(-math.pi / 2, 0.0, 1.0)
        N, Hh = P["neck"], P["hip"]
        B = rig.B
        chest3 = (N[0] * 0.7 + Hh[0] * 0.3, B["neck"][1] * 0.7 + B["hip"][1] * 0.3, 0.0)
        pocket3 = (B["neck"][0] + math.sin(rig.lean) * -8 + sp["dep"] * math.cos(0.7),
                   B["neck"][1] - 9.0, -sp["chest_w"] * math.sin(0.7) * 0.8)
        pk = (pocket3[0] * rig.sphi - pocket3[2] * rig.cphi, -pocket3[1])

        def tip(near):
            Hn = P["hand_n" if near else "hand_f"]
            E = P["el_n" if near else "el_f"]
            ux, uy = Hn[0] - E[0], Hn[1] - E[1]
            ln = math.hypot(ux, uy) or 1.0
            ux, uy = ux / ln, uy / ln
            shp = rig.shape_f if near else rig.shape_b
            hs = sp["hand"]
            if shp == "point":
                return (Hn[0] + ux * hs * 1.74, Hn[1] + uy * hs * 1.74)
            if shp == "finger":
                return (Hn[0] + hs * 0.12, Hn[1] - hs * 1.94)
            if shp == "pinch" and near and self.lk.get("prop") == "pen":
                return (Hn[0] + ux * hs * 2.6, Hn[1] + uy * hs * 2.6 + hs * 0.8)
            return (Hn[0] + ux * hs * 0.9, Hn[1] + uy * hs * 0.9)
        top_l = (hx, hyy - hr * (1.08 if self.lk.get("hair_style") not in ("curly", "wild") else 1.32))
        if self.lk.get("helmet"):
            top_l = (hx, hyy - hr * 1.16)
        if self.lk.get("crown"):
            top_l = (hx, hyy - hr * 1.55)
        a = {
            "head": Wp(P["head"]), "eyes": HW((ex, ey)), "mouth": HW((mo[0], mo[1])), "ear": HW((ear[0], ear[1])),
            "chest": Wp((chest3[0] * rig.sphi, -chest3[1])), "neck": Wp(N), "hip": Wp(Hh),
            "hand_r": Wp(P["hand_n"]), "hand_l": Wp(P["hand_f"]), "elbow_r": Wp(P["el_n"]), "elbow_l": Wp(P["el_f"]),
            "shoulder_r": Wp(P["sh_n"]), "shoulder_l": Wp(P["sh_f"]),
            "tip_r": Wp(tip(True)), "tip_l": Wp(tip(False)),
            "feet": (x, y), "foot_r": Wp(P["ank_n"]), "foot_l": Wp(P["ank_f"]),
            "top": Wp(top_l), "head_r": hr * k, "pocket": Wp(pk), "scale": k,
        }
        a["hand_front"], a["hand_back"] = a["hand_r"], a["hand_l"]
        a["prop"] = a["tip_r"]
        xs = [a[q][0] for q in ("head", "hand_r", "hand_l", "foot_r", "foot_l", "top", "hip")]
        ys = [a[q][1] for q in ("head", "hand_r", "hand_l", "foot_r", "foot_l", "top", "hip")]
        pad = hr * k * 1.5
        a["bbox"] = (min(xs) - pad, min(ys) - pad * 0.3, max(xs) + pad, max(max(ys), y) + k * 3)
        if rig.pole is not None:
            bx, by, ang, _ = rig.pole
            X, Y = Wp((bx, by))
            aw = ang * fs
            a["pole"] = (X, Y, aw)
            if pole_len:
                a["pole_top"] = (X + math.sin(aw) * pole_len, Y - math.cos(aw) * pole_len)
        return a

    def anchors(self, x, y, h, pose="stand", t=0.0, facing=1, pole_len=None, mouth=0.0, look=(0, 0), **kw):
        ch = self._with_style(kw)
        fs, k, rig, cphi, sphi, phi, hy, wxl = ch._setup(x, y, h, pose, t, facing, kw, mouth, look)
        return ch._anchors(x, y, fs, k, rig, hy, pole_len, look)

    # ------------------------------------------------------------------ draw
    def draw(self, ctx, x, y, h, pose="stand", t=0.0, mouth=0.0, look=(0, 0), expr="neutral", facing=1,
             blink=None, alpha=1.0, pole_len=None, **kw):
        ch = self._with_style(kw)
        return ch._draw(ctx, x, y, h, pose, t, mouth, look, expr, facing, blink, alpha, pole_len, kw)

    def _draw(self, ctx, x, y, h, pose, t, mouth, look, expr, facing, blink, alpha, pole_len, kw):
        mouth = clamp(float(mouth or 0.0))
        fs, k, rig, cphi, sphi, phi, hy, wxl = self._setup(x, y, h, pose, t, facing, kw, mouth, look)
        anchors = self._anchors(x, y, fs, k, rig, hy, pole_len, look)
        if alpha <= 0.002 or h <= 0.5:
            return anchors
        layer = kw.get("layer", "all")
        m = ctx.get_matrix()
        dev = math.sqrt(abs(m.xx * m.yy - m.xy * m.yx))
        hpx = h * dev
        lod = kw.get("lod")
        if lod is None:
            lod = 2 if hpx >= 140 else (1 if hpx >= 42 else 0)
        if kw.get("eyes_closed"):
            blink = True
        if blink is None:
            bl = _blink_amount(t, self.seed)
        elif blink is True:
            bl = 1.0
        elif blink is False:
            bl = 0.0
        else:
            bl = float(blink)
        E = blend_expr(expr)
        rim = kw.get("rim")
        ink = kw.get("render", self.render) == "ink"
        if ink:
            rim = None
        lw = max(0.34 * kw.get("ink_w", 1.0), 0.95 / max(dev * k, 1e-6)) if ink else 0.35
        o = dict(t=t, facing_sign=fs, light=kw.get("light", (-0.5, -0.8)), rim=None, tint=kw.get("tint"),
                 silhouette=kw.get("silhouette"), shade=kw.get("shade", 1.0), wind_x=wxl + rig.air, head_yaw=hy,
                 look=(look[0] * fs, look[1]), wink=kw.get("wink"), ink=ink, lw=lw, kw=kw,
                 upx=1.0 / max(dev * k, 1e-6))
        ctx.save()
        group = alpha < 0.999 or bool(rim) or (ink and lod >= 1 and kw.get("outline", True))
        if group:
            x0, y0, x1, y1 = anchors["bbox"]
            ctx.new_path()
            ctx.rectangle(x0 - 20 * k, y0 - 30 * k, (x1 - x0) + 40 * k, (y1 - y0) + 40 * k)
            ctx.clip()
            ctx.push_group()
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(fs * k, k)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        P = Painter(ctx, self, rig, lod, o)
        P.E = E
        self._paint(P, layer, E, bl, o["look"], mouth)
        ctx.restore()
        if group:
            pat = ctx.pop_group()
            if ink and lod >= 1 and kw.get("outline", True):
                # heavier OUTER silhouette line: the figure's mask dilated in 12 directions, in ink, underneath
                wo = lw * k * 0.62 * kw.get("outline_w", 1.0)
                m0 = pat.get_matrix()
                ctx.set_source_rgba(0, 0, 0, alpha)
                for i in range(12):
                    a_ = TAU * i / 12
                    pat.set_matrix(cairo.Matrix(x0=-math.cos(a_) * wo, y0=-math.sin(a_) * wo).multiply(m0))
                    ctx.mask(pat)
                pat.set_matrix(m0)
            ctx.set_source(pat)
            ctx.paint_with_alpha(alpha)
            if rim:
                # rim light on the OUTER silhouette only: S minus S shifted away from the rim light
                rc = col(rim[0])
                rs = float(rim[1]) if len(rim) > 1 else 0.8
                rd = rim[2] if len(rim) > 2 and rim[2] is not None else (0.8, -0.45)
                rl = math.hypot(rd[0], rd[1]) or 1.0
                wr = max(1.15 * k, 0.9 / max(dev, 1e-6)) * kw.get("rim_width", 1.0)
                vx_, vy_ = -rd[0] / rl * wr, -rd[1] / rl * wr
                ctx.push_group()
                ctx.set_source_rgba(rc[0], rc[1], rc[2], 1.0)
                ctx.mask(pat)
                m0 = pat.get_matrix()
                pat.set_matrix(cairo.Matrix(x0=-vx_, y0=-vy_).multiply(m0))
                ctx.set_operator(cairo.OPERATOR_DEST_OUT)
                ctx.set_source_rgba(0, 0, 0, 1)
                ctx.mask(pat)
                ctx.set_operator(cairo.OPERATOR_OVER)
                rp = ctx.pop_group()
                ctx.set_source(rp)
                ctx.paint_with_alpha(min(1.0, rs * alpha))
        ctx.restore()
        return anchors

    # ------------------------------------------------------------------ painting recipe
    def _paint(self, P, layer, E, blink, look, mouth):
        lk = self.lk
        r = P.rig
        g = lk["garment"]
        sleeve = lk["sleeve"]
        back = layer in ("all", "back")
        front = layer in ("all", "front")
        chair = P.opt("chair")
        if chair is None and r.seat > 0.5 and self.kind == "devil":
            chair = "throne"
        if chair is True:
            chair = "camp"
        if r.seat < 0.5:
            chair = None
        if back:
            if P.opt("wings"):
                sp_ = P.opt("wing_spread")
                if sp_ is None:
                    sp_ = 1.0 if r.flap_on > 0.5 else 0.3
                K2.wings(P, False, sp_, P.opt("flap"))
                K2.wings(P, True, sp_, P.opt("flap"))
            if chair == "camp":
                K2.camp_chair(P, "back")
            elif chair == "throne":
                K2.throne(P, "back")
            if lk.get("tail"):
                jaguar_tail(P)
            P.arm(False, sleeve, part="upper" if r.back_front else "all")
            if not r.back_front:
                self._prop(P, back=True)
            self._legs(P, g)
            self._torso(P, g)
            if r.back_front:
                P.arm(False, sleeve, part="fore")
                self._prop(P, back=True)
            if lk.get("hood") == "down":
                P.collar_hood_down("robe")
            if not lk.get("veil"):
                self._neck(P)
            if lk.get("collar") == "fur":
                K2.fur_collar(P)
            if lk.get("goggles_neck"):
                K2.goggles_neck(P)
            self._head(P, E, blink, look, mouth)
            if chair == "camp":
                K2.camp_chair(P, "front")
            elif chair == "throne":
                K2.throne(P, "front")
        if front:
            self._prop(P)
            P.arm(True, sleeve)

    def _neck(self, P):
        sp = self.sp
        r = P.rig
        N = r.P["neck"]
        Hc = r.P["head"]
        nw = sp["head_r"] * (0.3 if self.kind != "jaguar" else 0.46)
        colour = "fur" if self.kind == "jaguar" else "skin"
        if self.kind in TRACT_KINDS:
            nw = sp["head_r"] * 0.4
        nw2 = nw * (0.92 if self.kind in TRACT_KINDS else 1.0)
        path = P.part(lambda c: p_capsule(c, (N[0], N[1] + 1.0), nw, (Hc[0] * 0.5 + N[0] * 0.5, Hc[1] * 0.5 + N[1] * 0.5),
                                          nw2), colour, d=1.0, rim=0.6)
        if P.ink and self.kind in TRACT_KINDS:
            hp_ = _hp(P)
            if hp_.egg is not None:
                K2.neck_ink(P, path, Hc, r.head_rot, hp_)

    def _legs(self, P, g):
        r = P.rig
        if r.sit > 0.5:
            sp = self.sp
            Hh = r.P["hip"]
            colour = "robe" if g in ("robe", "crocus", "tactical", "toga") else "trouser"
            w = sp["hem_w"] * 1.25
            if g == "crocus":
                colour = "black"
            P.part(lambda c: (c.move_to(Hh[0] - w * 0.95, 0.0), c.curve_to(Hh[0] - w * 1.05, Hh[1] + 4, Hh[0] - w * 0.4,
                                                                            Hh[1] - 3, Hh[0], Hh[1] - 2),
                              c.curve_to(Hh[0] + w * 0.5, Hh[1] - 3, Hh[0] + w * 1.1, Hh[1] + 5, Hh[0] + w, 0.0),
                              c.close_path()), colour, d=2.5)
            return
        order = (False, True)
        robey = g in ("robe", "crocus", "toga", "poncho", "coat", "dress")
        if r.seat > 0.5 and robey:
            cloth = "black" if g == "crocus" else ("trouser" if g == "coat" else "robe")
            for near in order:
                self._seated_leg(P, near, cloth)
            return
        for near in order:
            if g == "tactical":
                P.leg(near, "trouser", "shoe", pad="pad")
            elif g == "toga":
                P.leg(near, "skin", "shoe")
            elif g in ("dress",):
                P.leg(near, "skin", "shoe")
            else:
                P.leg(near, "trouser", "shoe")

    def _seated_leg(self, P, near, cloth):
        r = P.rig
        sp = self.sp
        tag = "n" if near else "f"
        Hh, Kn, A = r.P["hip_" + tag], r.P["knee_" + tag], r.P["ank_" + tag]
        lr = sp["leg_r"]
        P.part(lambda c: p_chain(c, [Hh, Kn, A], [lr * 1.9, lr * 1.55, lr * 1.35]), cloth, d=lr * 0.6)
        P.foot(near, "shoe")

    def _torso(self, P, g):
        sp = self.sp
        r = P.rig
        mud = P.opt("mud", 0.0) or 0.0
        if r.seat > 0.5 and g in ("robe", "crocus", "toga", "poncho", "coat", "tactical"):
            hu = r.B["hip"][1]
            pts = P.robe(hem_y=-(hu - 5.0), hem_w=sp["waist_w"] * 1.3)
            P.fill_pts(pts, "black" if g == "crocus" else "robe")
            if g == "crocus":
                self._crocus_panel(P, hu - 5.0)
            if P.opt("apron"):
                K2.apron(P)
            K2.mud_spots(P, mud)
            return
        if g in ("poncho", "furcoat", "tutu"):
            if g == "poncho":
                pts = P.robe(flare=1.28)
                P.fill_pts(pts, "robe")
                P.ctx.save()
                P.ctx.new_path()
                p_smooth(P.ctx, pts, True, 0.33)
                P.ctx.clip()
                P.folds(colour="robe", n=4)
                K2.mud_spots(P, mud)
                P.ctx.restore()
            elif g == "furcoat":
                pts = P.robe(hem_y=-20.0, hem_w=sp["waist_w"] * 1.55)
                P.fill_pts(pts, "fur")
                P.ctx.save()
                P.ctx.new_path()
                p_smooth(P.ctx, pts, True, 0.33)
                P.ctx.clip()
                self._front_strip(P, -0.05, 0.05, r.B["neck"][1] + 0.5, 20.0, sp["chest_w"], sp["waist_w"],
                                  sp["dep"] * 1.06, sp["dep"] * 1.06, shadow_of(P.pal["fur"], 0.6), n=5)
                if P.lod >= 2:
                    for i in range(40):
                        q = P.pj(P.surf((0, 22 + 40 * hash01(i, 3 + self.seed), 0.0), sp["waist_w"] * 1.1,
                                        sp["dep"] * 1.1, -1.4 + 2.8 * hash01(i, 4 + self.seed)))
                        P.stroke(lambda c: (c.move_to(q[0], q[1]), c.line_to(q[0] + 0.3, q[1] + 1.6)), "lash",
                                 P.lw * 0.35, 0.8)
                K2.mud_spots(P, mud)
                P.ctx.restore()
            else:
                hu = r.B["hip"][1]
                pts = P.robe(hem_y=-(hu - 4.0), hem_w=sp["waist_w"] * 1.05)
                P.fill_pts(pts, "robe")
                P.ctx.save()
                P.ctx.new_path()
                p_smooth(P.ctx, pts, True, 0.33)
                P.ctx.clip()
                K2.mud_spots(P, mud)
                P.ctx.restore()
                K2.tutu(P)
            return
        if r.sit > 0.5 and g in ("robe", "crocus", "tactical", "toga"):
            pts = P.robe(hem_y=-(r.B["hip"][1] - 1.0), hem_w=sp["waist_w"] * 1.2)
            P.fill_pts(pts, "black" if g == "crocus" else "robe")
            if g == "crocus":
                self._crocus_panel(P, r.B["hip"][1] - 1.0)
            return
        if g == "robe":
            pts = P.robe()
            P.fill_pts(pts, "robe")
            P.ctx.save()
            P.ctx.new_path()
            p_smooth(P.ctx, pts, True, 0.33)
            P.ctx.clip()
            P.folds(colour="robe")
            K2.mud_spots(P, mud)
            P.ctx.restore()
            if self.lk.get("belt", True):
                P.belt("belt")
            if self.lk.get("pouch"):
                P.pouch()
            if P.opt("apron"):
                K2.apron(P)
        elif g == "crocus":
            pts = P.robe(flare=1.05)
            P.fill_pts(pts, "robe")
            P.ctx.save()
            P.ctx.new_path()
            p_smooth(P.ctx, pts, True, 0.33)
            P.ctx.clip()
            self._crocus_panel(P, -sp["hem_y"])
            P.folds(colour="robe", n=2)
            P.ctx.restore()
        elif g == "tactical":
            pts = P.robe()
            P.fill_pts(pts, "robe")
            P.ctx.save()
            P.ctx.new_path()
            p_smooth(P.ctx, pts, True, 0.33)
            P.ctx.clip()
            P.folds(colour="robe", hem_u=-sp["hem_y"])
            self._vest(P)
            P.ctx.restore()
            P.belt("belt", knot=False, width=3.2, u_off=0.5)
            self._pouches(P)
        elif g == "toga":
            pts = P.robe(flare=0.95)
            P.fill_pts(pts, "robe")
            P.ctx.save()
            P.ctx.new_path()
            p_smooth(P.ctx, pts, True, 0.33)
            P.ctx.clip()
            self._toga_drape(P)
            P.ctx.restore()
        elif g == "suit":
            pts = P.robe(hem_y=sp["hem_y"], hem_w=sp["waist_w"] * 1.06, flare=1.0)
            P.fill_pts(pts, "suit")
            P.ctx.save()
            P.ctx.new_path()
            p_smooth(P.ctx, pts, True, 0.33)
            P.ctx.clip()
            self._suit_front(P)
            P.ctx.restore()
        else:  # citizen garments
            hip_u = r.B["hip"][1]
            if g == "dress":
                hy, hw = -24.0, sp["hem_w"] * 0.95
            elif g == "coat":
                hy, hw = -22.0, sp["waist_w"] * 1.4
            else:
                hy, hw = -(hip_u - 5.0), sp["waist_w"] * 1.06
            pts = P.robe(hem_y=hy, hem_w=hw)
            P.fill_pts(pts, "robe")
            P.ctx.save()
            P.ctx.new_path()
            p_smooth(P.ctx, pts, True, 0.33)
            P.ctx.clip()
            self._citizen_details(P, g)
            K2.mud_spots(P, mud)
            P.ctx.restore()
            if g == "hoodie":
                P.collar_hood_down("robe")

    # ---------------------------------------------------------------- garment details
    def _front_strip(self, P, th0, th1, u_top, u_bot, w_top, w_bot, d_top, d_bot, colour, n=6, a=1.0):
        r = P.rig
        B = r.B
        N, Hh = B["neck"], B["hip"]
        left, right = [], []
        for i in range(n + 1):
            f = i / n
            u = u_top + (u_bot - u_top) * f
            # spine centre at height u (upper body follows lean, skirt hangs)
            if u >= Hh[1]:
                tt = (u - Hh[1]) / max(N[1] - Hh[1], 1e-3)
                cen = (Hh[0] + (N[0] - Hh[0]) * tt, u, 0.0)
            else:
                cen = (Hh[0] * (u / max(Hh[1], 1e-3)) + r.sway * 0.3 * (1 - u / max(Hh[1], 1)), u, 0.0)
            w = w_top + (w_bot - w_top) * f
            d = d_top + (d_bot - d_top) * f
            pl = P.surf(cen, w, d, th0)
            pr = P.surf(cen, w, d, th1)
            left.append(P.pj(pl))
            right.append(P.pj(pr))
        zc = P.zof(P.surf((0, 0, 0), 1, 1, 0.5 * (th0 + th1)))
        if zc < -0.15:
            return
        pts = left + right[::-1]
        P.part(lambda c: p_smooth(c, pts, True, 0.25), colour, d=1.2, a=a, rim=0.4)

    def _crocus_panel(self, P, hem_u):
        sp = self.sp
        r = P.rig
        N = r.B["neck"]
        self._front_strip(P, -0.42, 0.42, N[1] + 1.0, hem_u - 2, sp["head_r"] * 0.5, sp["hem_w"] * 0.5,
                          sp["dep"] * 1.1, sp["hem_w"] * 0.75, "black", n=8)

    def _vest(self, P):
        sp = self.sp
        r = P.rig
        B = r.B
        N, Hh = B["neck"], B["hip"]
        top = N[1] - 3.0
        bot = Hh[1] + 4.0
        # plate carrier wraps the torso: draw as a band loft slightly bigger than the torso
        perp = P.torso_frame()
        rings = []
        for f in (0.0, 0.5, 1.0):
            u = top + (bot - top) * f
            tt = (u - Hh[1]) / max(N[1] - Hh[1], 1e-3)
            cen = (Hh[0] + (N[0] - Hh[0]) * tt, u, 0.0)
            w = (sp["chest_w"] * (1 - f) + sp["waist_w"] * f) * 1.06
            rings.append((cen, w, sp["dep"] * 1.12, perp))
        pts = P.loft(rings)
        P.part(lambda c: p_smooth(c, pts, True, 0.2), "vest", d=2.0)
        if P.lod >= 1:
            # front pouches
            for i, th in enumerate((-0.55, 0.0, 0.55)):
                cen_u = bot + 3.5
                tt = (cen_u - Hh[1]) / max(N[1] - Hh[1], 1e-3)
                cen = (Hh[0] + (N[0] - Hh[0]) * tt, cen_u, 0.0)
                p = P.surf(cen, sp["waist_w"] * 1.12, sp["dep"] * 1.2, th)
                if P.zof(p) < 0.5:
                    continue
                x, y = P.pj(p)
                wv = 2.6 * max(0.3, math.cos(th + (math.pi / 2 - math.atan2(r.cphi, r.sphi)))) ** 0.5
                P.part(lambda c: (c.new_sub_path(), c.rectangle(x - wv, y - 3.4, 2 * wv, 6.2)),
                       shadow_of(P.pal["vest"], 0.8), d=0.8)
                P.flat(lambda c: c.rectangle(x - wv, y - 3.4, 2 * wv, 1.3), light_of(P.pal["vest"], 0.12))

    def _pouches(self, P):
        sp = self.sp
        r = P.rig
        Hh = r.B["hip"]
        for th in (0.9, 1.5, -0.9):
            p = P.surf((Hh[0], Hh[1] - 1.0, 0.0), sp["waist_w"] * 1.12, sp["dep"] * 1.15, th)
            if P.zof(p) < 0:
                continue
            x, y = P.pj(p)
            P.part(lambda c: (c.new_sub_path(), c.rectangle(x - 2.2, y - 2.0, 4.4, 5.0)), "vest", d=0.8)

    def _toga_drape(self, P):
        sp = self.sp
        r = P.rig
        B = r.B
        N, Hh = B["neck"], B["hip"]
        # diagonal sash from far shoulder across the chest to the near hip
        a = P.pj(P.surf((N[0], N[1] - 1, 0.0), sp["sh_w"], sp["dep"], -1.1))
        b = P.pj(P.surf((Hh[0], Hh[1] + 2, 0.0), sp["waist_w"] * 1.05, sp["dep"] * 1.05, 1.2))
        for k, (off, al) in enumerate(((0.0, 0.5), (4.0, 0.4), (8.5, 0.35))):
            P.stroke(lambda c: (c.move_to(a[0] + off * 0.3, a[1] + off), c.curve_to(a[0] + (b[0] - a[0]) * 0.3,
                                                                                     a[1] + (b[1] - a[1]) * 0.6 + off,
                                                                                     b[0] - 3, b[1] - 6 + off * 0.5,
                                                                                     b[0], b[1] + off * 0.6)),
                     shadow_of(P.pal["robe"], 0.8), 1.4, al)
        P.folds(colour="robe", n=3, hem_u=-sp["hem_y"])

    def _suit_front(self, P):
        sp = self.sp
        r = P.rig
        B = r.B
        N, Hh = B["neck"], B["hip"]
        u_top = N[1] + 1.0
        u_v = N[1] - 17.0
        # shirt V
        self._front_strip(P, -0.5, 0.5, u_top, u_v, sp["head_r"] * 0.55, 0.6, sp["dep"] * 0.9, sp["dep"] * 1.08,
                          "shirt", n=4)
        # tie
        self._front_strip(P, -0.12, 0.12, u_top - 1.5, u_v - 4.5, 2.4 / 0.12, 3.0 / 0.12, sp["dep"] * 1.0,
                          sp["dep"] * 1.1, "tie", n=4)
        # lapels
        for side in (-1, 1):
            th0 = side * 0.5
            th1 = side * 0.85
            self._front_strip(P, min(th0, th1), max(th0, th1), u_top, u_v - 2.0, sp["chest_w"] * 0.7,
                              sp["chest_w"] * 0.1, sp["dep"] * 1.05, sp["dep"] * 1.12, "lapel", n=4)
        # breast pocket square hint (far side) + buttons
        if P.lod >= 2:
            for bu in (u_v - 4.0, u_v - 9.0):
                tt = (bu - Hh[1]) / max(N[1] - Hh[1], 1e-3)
                cen = (Hh[0] + (N[0] - Hh[0]) * tt, bu, 0.0)
                p = P.surf(cen, sp["waist_w"], sp["dep"] * 1.12, 0.05)
                if P.zof(p) > 0:
                    x, y = P.pj(p)
                    P.flat(lambda c: p_circle(c, x, y, 0.7), "lapel")

    def _citizen_details(self, P, g):
        sp = self.sp
        r = P.rig
        B = r.B
        N, Hh = B["neck"], B["hip"]
        if P.lod < 1:
            return
        if g in ("jacket", "coat") and P.ink and P.lod >= 2:
            bts = []
            for j in range(5 if g == "coat" else 4):
                u = N[1] - 7.0 - j * (7.5 if g == "coat" else 6.0)
                if u < (18.0 if g == "coat" else Hh[1] - 3):
                    break
                tt = (u - Hh[1]) / max(N[1] - Hh[1], 1e-3)
                cen = (Hh[0] + (N[0] - Hh[0]) * tt, u, 0.0) if u >= Hh[1] else (Hh[0], u, 0.0)
                q = P.surf(cen, sp["waist_w"], sp["dep"] * 1.08, 0.1)
                if P.zof(q) > 0:
                    bts.append(P.pj(q))
            K2.buttons(P, bts)
        if g in ("jacket", "coat"):
            self._front_strip(P, -0.05, 0.05, N[1] + 0.5, 22.0 if g == "coat" else Hh[1] - 4,
                              sp["chest_w"], sp["waist_w"], sp["dep"] * 1.06, sp["dep"] * 1.06,
                              shadow_of(P.pal["robe"], 0.7), n=5)
            self._front_strip(P, -0.55, 0.55, N[1] + 1.2, N[1] - 1.8, sp["head_r"] * 0.62, sp["head_r"] * 0.75,
                              sp["dep"] * 0.9, sp["dep"] * 1.0, shadow_of(P.pal["robe"], 0.82), n=2)
        elif g == "hoodie":
            for side in (-0.18, 0.18):
                p0 = P.pj(P.surf((N[0], N[1] - 2, 0.0), sp["head_r"] * 0.5, sp["dep"], side))
                P.stroke(lambda c: (c.move_to(*p0), c.line_to(p0[0] + 0.3, p0[1] + 9)), "sclera", 0.7, 0.8)
            self._front_strip(P, -0.6, 0.6, Hh[1] + 8, Hh[1] + 1, sp["waist_w"], sp["waist_w"], sp["dep"] * 1.04,
                              sp["dep"] * 1.04, shadow_of(P.pal["robe"], 0.85), n=2)
        elif g == "shirt":
            self._front_strip(P, -0.35, 0.35, N[1] + 0.8, N[1] - 3.5, sp["head_r"] * 0.6, sp["head_r"] * 0.6,
                              sp["dep"] * 0.9, sp["dep"] * 1.0, light_of(P.pal["robe"], 0.25), n=2)
        elif g == "sweater":
            self._front_strip(P, -0.5, 0.5, N[1] + 0.6, N[1] - 1.2, sp["head_r"] * 0.55, sp["head_r"] * 0.6,
                              sp["dep"] * 0.9, sp["dep"] * 0.95, shadow_of(P.pal["robe"], 0.8), n=2)
        elif g == "dress":
            P.belt(shadow_of(P.pal["robe"], 0.7), knot=False, width=1.6, u_off=5.0)

    # ---------------------------------------------------------------- heads
    def _head(self, P, E, blink, look, mouth):
        lk = self.lk
        hp = _hp(P)
        P.head_frame()
        try:
            if self.kind == "jaguar":
                jaguar_head(P, hp, E, blink, look, mouth)
                return
            hs = lk.get("hair_style")
            face = P.opt("face") or ("tract" if (P.ink or self.kind in TRACT_KINDS) else "std")
            if face == "cartoon" and E.get("glow", 0) > 0.05:
                K2.aura(P, hp, E["glow"])
            halo = P.opt("halo")
            if halo:
                K2.halo(P, hp, halo)
            if lk.get("earrings") == "hoop":
                K2.hoop(P, hp, False)
            if hs == "pigtails":
                K2.pigtail(P, hp, False)
            elif hs == "long_glossy" or (hs == "long" and hp.egg is not None):
                K2.long_glossy_back(P, hp)
            elif hs == "dreads":
                K2.dreads(P, hp, front=False)
            else:
                P.hair_back(hp, hs, self.seed)
            if lk.get("hat") == "cap":
                K2.cap_hair(P, hp, "back")
            hooded = lk.get("hood") == "up" or lk.get("veil")
            if lk.get("horns"):
                K2.devil_ear(P, hp, False)
            elif not hooded:
                P.ear(hp, False)
            P.head_base(hp)
            if lk.get("horns"):
                K2.devil_ear(P, hp, True)
            elif not hooded:
                P.ear(hp, True)
            big = lk.get("eyes") == "big"
            beard = lk.get("beard")
            bearded = beard in ("long", "full")
            if hp.yaw < math.pi * 0.72:
                if face == "anime":
                    K2.face_anime(P, hp, E, blink, look, mouth)
                elif face == "tract":
                    K2.face_tract(P, hp, E, blink, look, 0.0 if bearded else mouth, eyes=lk.get("eyes"))
                else:
                    fem = self.kind in ("summer", "raven", "seraph", "lutie")
                    if P.ink:
                        P.pal = dict(P.pal, iris=(0.3, 0.3, 0.3), lid=(0.99, 0.99, 0.99))
                    P.face(hp, E, blink, look, mouth, eyes_style="big" if (big or fem) else "std",
                           show_mouth=not bearded,
                           brow_w=0.1 if self.kind in ("crocus", "crow", "pharisee", "devil") else (0.085 if big else 0.075),
                           brow_len=1.25 if self.kind == "crocus" else 1.0)
                    if face == "cartoon":
                        if lk.get("freckles"):
                            K2.freckles(P, hp)
                        if E["tears"] > 0.05:
                            K2.tears(P, hp, E)
                        if E["sweat"] > 0.05:
                            K2.sweat(P, hp, E, flying=True)
                        if E.get("shock", 0) > 0.05:
                            K2.shock_lines(P, hp, E["shock"])
            if beard == "stubble":
                _stubble(P, hp)
            if face == "std" and self.kind in ("crocus", "philosopher", "crackpot") and hp.yaw < math.pi * 0.7:
                age_lines(P, hp, 1.0 if self.kind == "crocus" else 0.6)
            fp = lk.get("face_paint")
            if fp and hp.yaw < math.pi * 0.7:
                K2.face_paint(P, hp, fp)
            if (lk.get("phone") or lk.get("prop") == "phone") and not P.ink:
                self._phone_glow(P, hp)
            if lk.get("veil"):
                _hood(P, hp, "veil", pointed=0.05, open_g=1.0)
            if bearded and hp.yaw < math.pi * 0.8:
                if hp.egg is not None:
                    K2.beard_ink(P, hp, length=2.0 if beard == "long" else 0.9)
                else:
                    _beard(P, hp, length=2.5 if beard == "long" else 1.2)
                _mustache(P, hp)
                if face == "tract":
                    K2.mouth_tract(P, hp, E, mouth)
                else:
                    P.mouth(hp, E, mouth, -0.52, lip=shadow_of(P.pal["hair"], 0.55))
            elif beard in ("goatee", "goatee_pointed") and hp.yaw < math.pi * 0.8:
                if beard == "goatee" and P.ink:
                    K2.goatee_hatched(P, hp)
                else:
                    K2.goatee(P, hp, pointed=beard == "goatee_pointed",
                              colour="beardc" if beard == "goatee" else "hair")
            if not hooded and not lk.get("helmet") and lk.get("hat") != "cap":
                P.hair_cap(hp, hs, self.seed)
            elif lk.get("hat") == "cap":
                K2.cap_hair(P, hp)          # hair tufts at the temples and nape below the cap
            if hs == "pigtails":
                K2.pigtail(P, hp, True)
            elif hs == "long_glossy" or (hs == "long" and hp.egg is not None):
                K2.long_glossy_front(P, hp)
            elif hs == "dreads":
                K2.dreads(P, hp, front=True)
            if lk.get("horns"):
                K2.horns(P, hp)
            if lk.get("glasses") == "aviator" and hp.yaw < math.pi * 0.75:
                K2.aviators(P, hp)
            fm = P.opt("face_mask", 0.0)
            if fm:
                K2.face_mask(P, hp, 1.0 if fm is True else float(fm))
            if lk.get("goggles_head"):
                K2.goggles_head(P, hp)
            if lk.get("hat") == "cap":
                K2.trucker_cap(P, hp)
            elif lk.get("hat") == "fez":
                K2.fez(P, hp)
            if lk.get("earrings") == "hoop":
                K2.hoop(P, hp, True)
            if lk.get("helmet"):
                _helmet(P, hp, lk.get("goggles_down"))
                if lk.get("headset"):
                    _headset(P, hp)
            if lk.get("hood") == "up":
                P.hair_cap(hp, "short", self.seed)
                _hood(P, hp, "robe", pointed=0.18 if self.kind == "cultist" else 0.05,
                      shadow_face=0.55 if self.kind == "cultist" else 0.25,
                      open_g=0.88 if self.kind == "cultist" else 0.95)
            if lk.get("laurel"):
                _laurel(P, hp)
            if lk.get("crown"):
                _crown(P, hp)
        finally:
            P.ctx.restore()

    def _phone_glow(self, P, hp):
        if P.sil is not None:
            return
        R = self.sp["head_r"]
        ctx = P.ctx
        c = self.lk["phone_color"]
        g = cairo.RadialGradient(R * 0.6 * math.sin(hp.yaw), R * 1.4, 0, R * 0.4 * math.sin(hp.yaw), R * 0.9, R * 2.0)
        g.add_color_stop_rgba(0, c[0], c[1], c[2], 0.42)
        g.add_color_stop_rgba(0.6, c[0], c[1], c[2], 0.12)
        g.add_color_stop_rgba(1, c[0], c[1], c[2], 0.0)
        ctx.save()
        ctx.new_path()
        p_circle(ctx, 0, 0, R * 1.02)
        p_ellipse(ctx, R * 0.3 * math.sin(hp.yaw), R * 0.32, R * 0.8, R * 0.72)
        ctx.clip()
        ctx.set_source(g)
        ctx.paint()
        ctx.restore()

    # ---------------------------------------------------------------- props (front hand)
    def _prop(self, P, back=False):
        prop = P.opt("prop")
        if not prop:
            return
        if (prop == "tracts") != back:
            return
        r = P.rig
        sp = self.sp
        Hn = r.P["hand_f" if back else "hand_n"]
        E = r.P["el_f" if back else "el_n"]
        ux, uy = Hn[0] - E[0], Hn[1] - E[1]
        ln = math.hypot(ux, uy) or 1.0
        ux, uy = ux / ln, uy / ln
        hs = sp["hand"]
        x, y = Hn
        ctx = P.ctx
        if prop in ("tract", "tracts", "tract_open"):
            ang = math.atan2(uy, ux) - math.pi / 2
            if prop == "tract_open":
                Hb = r.P["hand_f"]
                if math.hypot(Hb[0] - x, Hb[1] - y) < hs * 5.5:
                    x, y = (x + Hb[0]) * 0.5, (y + Hb[1]) * 0.5
                ang = 0.0
            K2.tract_prop(P, x + ux * hs * 0.4, y + uy * hs * 0.4 - hs * 0.4, ang * 0.3, prop, hs)
            return
        if prop == "lyre":
            K2.lyre(P, x, y + hs * 0.8, hs)
            return
        if prop == "phone":
            w, hh = hs * 0.95, hs * 1.7
            px, py = x + hs * 0.2, y - hs * 0.9
            P.part(lambda c: _rr(c, px - w / 2, py - hh / 2, w, hh, w * 0.2), "#23252C", d=0.5)
            if P.ink:
                P.part(lambda c: _rr(c, px - w / 2 + 0.3, py - hh / 2 + 0.3, w - 0.6, hh - 0.6, w * 0.15), (1, 1, 1), d=0,
                       line=0.3)
            elif P.sil is None:
                cc = self.lk["phone_color"]
                ctx.save()
                g = cairo.RadialGradient(px, py, 0, px, py, hs * 4)
                g.add_color_stop_rgba(0, cc[0], cc[1], cc[2], 0.35)
                g.add_color_stop_rgba(1, cc[0], cc[1], cc[2], 0.0)
                ctx.set_source(g)
                p_circle(ctx, px, py, hs * 4)
                ctx.fill()
                ctx.set_source_rgba(cc[0], cc[1], cc[2], 0.95)
                _rr(ctx, px - w / 2 - 0.25, py - hh / 2 + 0.3, 0.6, hh - 0.6, 0.3)
                ctx.fill()
                ctx.restore()
        elif prop == "megaphone":
            ang = 0.0 - 0.15
            L = hs * 5.0
            ca, sa = math.cos(ang), math.sin(ang)
            bx, by = x + hs * 0.6, y - hs * 0.5
            ex, ey = bx + ca * L, by + sa * L
            nx, ny = -sa, ca
            P.part(lambda c: p_poly(c, [(bx + nx * hs * 0.5, by + ny * hs * 0.5), (ex + nx * hs * 1.6, ey + ny * hs * 1.6),
                                        (ex - nx * hs * 1.6, ey - ny * hs * 1.6), (bx - nx * hs * 0.5, by - ny * hs * 0.5)]),
                   "#C9CCD4", d=1.0)
            P.part(lambda c: p_ellipse(c, ex, ey, hs * 0.45, hs * 1.6, ang), "#8E93A0", d=0)
        elif prop == "pen":
            tx, ty = x + ux * hs * 2.6, y + uy * hs * 2.6 + hs * 0.8
            bx, by = x - ux * hs * 1.2, y - uy * hs * 1.2 - hs * 0.9
            P.stroke(lambda c: (c.move_to(bx, by), c.line_to(tx, ty)), "#1C1C24", hs * 0.42)
            P.stroke(lambda c: (c.move_to(bx, by), c.line_to(bx + (tx - bx) * 0.25, by + (ty - by) * 0.25)),
                     "#C9CCD6", hs * 0.46)
        elif prop == "candle":
            cx, cy = x + hs * 0.1, y - hs * 0.6
            P.part(lambda c: _rr(c, cx - hs * 0.35, cy - hs * 2.2, hs * 0.7, hs * 2.2, hs * 0.15), "#EDE8E0", d=0.3)
            if P.ink:
                from .chars_paint import p_leaf
                P.part(lambda c: p_leaf(c, cx, cy - hs * 2.2, hs * 1.1, hs * 0.32, -math.pi / 2 + 0.1 * math.sin(P.t * 7)),
                       (1, 1, 1), d=0, line=0.5)
            elif P.sil is None:
                ctx.save()
                g = cairo.RadialGradient(cx, cy - hs * 2.8, 0, cx, cy - hs * 2.8, hs * 3.5)
                g.add_color_stop_rgba(0, 1.0, 0.86, 0.7, 0.45)
                g.add_color_stop_rgba(1, 1.0, 0.8, 0.6, 0.0)
                ctx.set_source(g)
                p_circle(ctx, cx, cy - hs * 2.8, hs * 3.5)
                ctx.fill()
                ctx.set_source_rgba(1.0, 0.93, 0.85, 1.0)
                from .chars_paint import p_leaf
                p_leaf(ctx, cx, cy - hs * 2.2, hs * 1.1, hs * 0.32, -math.pi / 2 + 0.1 * math.sin(P.t * 7))
                ctx.fill()
                ctx.restore()


def _rr(c, x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    c.new_sub_path()
    c.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    c.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    c.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    c.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    c.close_path()


# ============================================================ crowds
_CROWD_CHARS = {}
_TINY_RIGS = {}
_DYNAMIC = ("walk", "run", "cheer", "talk", "wave", "sign", "pat", "leap")


def _crowd_char(kind, seed, style):
    key = (kind, seed, tuple(sorted(style.items())))
    ch = _CROWD_CHARS.get(key)
    if ch is None:
        if len(_CROWD_CHARS) > 6000:
            _CROWD_CHARS.clear()
        ch = Character(kind, seed, **style)
        _CROWD_CHARS[key] = ch
    return ch


def _pose_key(pose):
    if isinstance(pose, str):
        return pose
    return tuple(sorted((k, round(w * 12) / 12) for k, w in pose.items() if w > 0.02))


def _tiny_rig(ch, pose, t, facing, turn, speed):
    """cached rig for tiny figures (spec variant = seed % 6, quantised pose weights and cycle phase)"""
    pk = _pose_key(pose)
    names = (pk,) if isinstance(pk, str) else tuple(k for k, _ in pk)
    dyn = any(n in _DYNAMIC for n in names)
    tq = round(t * 12) / 12 if dyn else 0.0
    key = (ch.kind, ch.seed % 6, pk, tq if dyn else 0, facing, round(turn, 2), round(speed, 2))
    rig = _TINY_RIGS.get(key)
    if rig is None:
        if len(_TINY_RIGS) > 20000:
            _TINY_RIGS.clear()
        base = _crowd_char(ch.kind, ch.seed % 6, {})
        pose_q = pose if isinstance(pk, str) else dict(pk)
        phi = turn * math.pi / 2
        rig = solve(base.sp, base.seed, pose_q, tq, {"speed": speed}, math.cos(phi), math.sin(phi))
        _TINY_RIGS[key] = (rig, base.sp)
        rig = _TINY_RIGS[key]
    return rig


def _tiny_draw(ctx, ch, rs, x, y, h, fs, alpha, tint, sil, ink=False, mud=0.0):
    rig, sp = rs
    P = rig.P
    k = h / 100.0
    pal = ch.ink_pal() if ink else ch.pal
    lk = ch.lk

    def cc(c):
        c = pal.get(c, c) if isinstance(c, str) else c
        c = col(c)
        if sil is not None:
            return sil
        if ink:
            L = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            L = 0.0 if L < 0.16 else L
            return (L, L, L)
        if tint:
            c = mixc(c, tint[0], tint[1])
        return c
    if ink:
        _tiny_ink(ctx, ch, rs, x, y, h, fs, alpha, cc, mud)
        return
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(fs * k, k)
    body = "suit" if lk["garment"] == "suit" else ("black" if lk["garment"] == "crocus" else "robe")
    bc = cc(body)
    g = lk["garment"]
    legs_visible = g not in ("robe", "crocus")
    # legs
    if legs_visible and rig.sit < 0.5:
        lc = cc("skin" if g in ("toga", "dress") else "trouser")
        ctx.set_source_rgba(lc[0], lc[1], lc[2], alpha)
        ctx.set_line_width(sp["leg_r"] * 1.7)
        for tag in ("f", "n"):
            ctx.move_to(*P["hip_" + tag])
            ctx.line_to(*P["knee_" + tag])
            ctx.line_to(*P["ank_" + tag])
        ctx.stroke()
    # far arm
    sc = cc("sleeve" if g != "toga" else "skin")
    ctx.set_source_rgba(sc[0], sc[1], sc[2], alpha)
    ctx.set_line_width(sp["arm_r"] * 2.0)
    ctx.move_to(*P["sh_f"])
    ctx.line_to(*P["el_f"])
    ctx.line_to(*P["hand_f"])
    ctx.stroke()
    # body
    N, Hh = P["neck"], P["hip"]
    cphi, sphi = rig.cphi, rig.sphi
    wsh = math.sqrt((sp["sh_w"] * cphi) ** 2 + (sp["dep"] * sphi) ** 2) + sp["arm_r"] * 0.6
    ww = math.sqrt((sp["waist_w"] * cphi) ** 2 + (sp["dep"] * sphi) ** 2)
    if g in ("robe", "crocus", "tactical", "toga"):
        hem_y = sp["hem_y"] if rig.sit < 0.5 else Hh[1] + 1
        hw = math.sqrt((sp["hem_w"] * cphi) ** 2 + (sp["hem_w"] * 0.72 * sphi) ** 2)
    elif g == "suit":
        hem_y, hw = sp["hem_y"], ww * 1.05
    elif g in ("dress", "coat"):
        hem_y, hw = -23.0, ww * 1.35
    else:
        hem_y, hw = Hh[1] + 5, ww * 1.05
    sw = rig.sway * 0.4
    ctx.set_source_rgba(bc[0], bc[1], bc[2], alpha)
    ctx.move_to(N[0] - wsh * 0.55, N[1] - 1)
    ctx.line_to(N[0] + wsh * 0.55, N[1] - 1)
    ctx.line_to(N[0] + wsh, N[1] + 3)
    ctx.line_to(Hh[0] + ww, Hh[1] - 2)
    ctx.line_to(sw + hw, hem_y)
    ctx.line_to(sw - hw, hem_y)
    ctx.line_to(Hh[0] - ww, Hh[1] - 2)
    ctx.line_to(N[0] - wsh, N[1] + 3)
    ctx.close_path()
    ctx.fill()
    # shadow side (one cheap fill)
    if h * math.sqrt(abs(ctx.get_matrix().xx * ctx.get_matrix().yy)) / max(k, 1e-9) * k > 18 and sil is None:
        sh = cc(shadow_of(pal.get(body, (0.5, 0.5, 0.5)), 0.74))
        ctx.set_source_rgba(sh[0], sh[1], sh[2], alpha)
        ctx.move_to(N[0] + wsh * 0.55, N[1] - 1)
        ctx.line_to(N[0] + wsh, N[1] + 3)
        ctx.line_to(Hh[0] + ww, Hh[1] - 2)
        ctx.line_to(sw + hw, hem_y)
        ctx.line_to(sw + hw * 0.45, hem_y)
        ctx.line_to(Hh[0] + ww * 0.4, Hh[1] - 2)
        ctx.close_path()
        ctx.fill()
    # head
    hx, hy = P["head"]
    R = sp["head_r"]
    hood = lk.get("hood") == "up" or lk.get("veil") or lk.get("helmet")
    skc = cc("fur" if ch.kind == "jaguar" else "skin")
    ctx.set_source_rgba(skc[0], skc[1], skc[2], alpha)
    ctx.move_to(hx + R, hy)
    ctx.arc(hx, hy, R, 0, TAU)
    ctx.fill()
    # hair / hood / helmet cap
    capc = None
    if lk.get("helmet"):
        capc = cc("helmet")
    elif lk.get("veil"):
        capc = cc("veil")
    elif lk.get("hood") == "up":
        capc = cc("robe")
    elif lk.get("hair_style") not in ("bald", "none", None):
        capc = cc("hair")
    if capc is not None:
        ctx.set_source_rgba(capc[0], capc[1], capc[2], alpha)
        a0 = math.pi * (0.95 if not hood else 0.7)
        yaw = rig.sphi
        ctx.move_to(hx, hy)
        ctx.arc(hx, hy, R * (1.08 if not hood else 1.18), a0 + yaw * 0.25, TAU + 0.05 * (1 - yaw) + yaw * 0.3)
        ctx.close_path()
        ctx.fill()
        if lk.get("veil") or lk.get("hood") == "up":
            ctx.move_to(hx - R * 1.15, hy)
            ctx.line_to(hx + R * 0.3, hy - R * 0.2)
            ctx.line_to(hx + R * 0.1, hy + R * 1.5)
            ctx.line_to(hx - R * 1.3, hy + R * 1.5)
            ctx.close_path()
            ctx.fill()
    hpx = h * math.sqrt(abs(ctx.get_matrix().xx * ctx.get_matrix().yy)) / max(k, 1e-9) * k
    if hpx > 30 and sil is None and rig.sphi > -0.2 and rig.cphi > -0.5:
        # two dot eyes (the face turns with the body)
        ec = cc("lash") if ch.kind != "jaguar" else cc("fur_dark")
        ctx.set_source_rgba(ec[0], ec[1], ec[2], alpha)
        yaw = math.atan2(rig.sphi, rig.cphi)
        er = R * (0.13 if hpx > 55 else 0.17)
        for lon in (-sp["eye_sep"], sp["eye_sep"]):
            z = math.cos(lon + yaw)
            if z > 0.15:
                ex = hx + R * math.sin(lon + yaw) * 0.95
                ctx.move_to(ex + er, hy + R * 0.02)
                ctx.arc(ex, hy + R * 0.02, er, 0, TAU)
        ctx.fill()
    if lk.get("beard") in ("long", "full"):
        bc2 = cc("hair")
        ctx.set_source_rgba(bc2[0], bc2[1], bc2[2], alpha)
        L = R * (3.2 if lk["beard"] == "long" else 1.9)
        ctx.move_to(hx - R * 0.55 + R * 0.5 * sphi, hy + R * 0.45)
        ctx.line_to(hx + R * 0.55 + R * 0.5 * sphi, hy + R * 0.45)
        ctx.line_to(hx + R * 0.4 * sphi, hy + L)
        ctx.close_path()
        ctx.fill()
    if lk.get("crown"):
        crc = cc("crown")
        ctx.set_source_rgba(crc[0], crc[1], crc[2], alpha)
        ctx.move_to(hx - R * 0.6, hy - R * 0.85)
        ctx.line_to(hx - R * 0.65, hy - R * 1.55)
        ctx.line_to(hx - R * 0.25, hy - R * 1.2)
        ctx.line_to(hx, hy - R * 1.65)
        ctx.line_to(hx + R * 0.25, hy - R * 1.2)
        ctx.line_to(hx + R * 0.65, hy - R * 1.55)
        ctx.line_to(hx + R * 0.6, hy - R * 0.85)
        ctx.close_path()
        ctx.fill()
    # near arm + hands
    ctx.set_source_rgba(sc[0], sc[1], sc[2], alpha)
    ctx.set_line_width(sp["arm_r"] * 2.1)
    ctx.move_to(*P["sh_n"])
    ctx.line_to(*P["el_n"])
    ctx.line_to(*P["hand_n"])
    ctx.stroke()
    hc = cc("hand")
    ctx.set_source_rgba(hc[0], hc[1], hc[2], alpha)
    for tag in ("n", "f"):
        x0, y0 = P["hand_" + tag]
        ctx.move_to(x0 + sp["hand"], y0)
        ctx.arc(x0, y0, sp["hand"], 0, TAU)
    ctx.fill()
    ctx.restore()


def _tiny_ink(ctx, ch, rs, x, y, h, fs, alpha, cc, mud):
    """print-ready tiny figure: black silhouette contour + light tone fill + a few interior lines + mud spots"""
    rig, sp = rs
    P = rig.P
    k = h / 100.0
    lk = ch.lk
    m = ctx.get_matrix()
    dev = math.sqrt(abs(m.xx * m.yy - m.xy * m.yx)) * k
    lw = max(0.9 / max(dev, 1e-6), 0.5)
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(fs * k, k)
    g = lk["garment"]
    N, Hh = P["neck"], P["hip"]
    cphi, sphi = rig.cphi, rig.sphi
    wsh = math.sqrt((sp["sh_w"] * cphi) ** 2 + (sp["dep"] * sphi) ** 2) + sp["arm_r"] * 0.6
    ww = math.sqrt((sp["waist_w"] * cphi) ** 2 + (sp["dep"] * sphi) ** 2)
    robey = g in ("robe", "crocus", "toga", "poncho")
    if robey:
        hem_y, hw = sp["hem_y"], math.sqrt((sp["hem_w"] * cphi) ** 2 + (sp["hem_w"] * 0.72 * sphi) ** 2)
    elif g in ("coat", "furcoat", "dress"):
        hem_y, hw = -22.0, ww * 1.35
    else:
        hem_y, hw = Hh[1] + 5, ww * 1.05
    if rig.seat > 0.5 and robey:
        hem_y = Hh[1] + 4
    body = [(N[0] - wsh * 0.55, N[1] - 1), (N[0] + wsh * 0.55, N[1] - 1), (N[0] + wsh, N[1] + 3),
            (Hh[0] + ww, Hh[1] - 2), (rig.sway * 0.4 + hw, hem_y), (rig.sway * 0.4 - hw, hem_y),
            (Hh[0] - ww, Hh[1] - 2), (N[0] - wsh, N[1] + 3)]

    def lines(pts, w, c):
        ctx.move_to(*pts[0])
        for q in pts[1:]:
            ctx.line_to(*q)
        ctx.set_line_width(w)
        ctx.set_source_rgba(*c, alpha)
        ctx.stroke()
    # outlines first (drawn fat), then fills (thinner) -> ink contour ring
    legs_vis = not robey or rig.seat > 0.5
    limbs = []
    if legs_vis and rig.sit < 0.5:
        for tag in ("f", "n"):
            limbs.append(([P["hip_" + tag], P["knee_" + tag], P["ank_" + tag]], sp["leg_r"] * 1.7,
                          cc("robe" if robey else ("skin" if g in ("toga", "dress", "tutu") else "trouser"))))
    arms = [([P["sh_" + t_], P["el_" + t_], P["hand_" + t_]], sp["arm_r"] * 2.0, cc("sleeve" if g != "toga" else "skin"))
            for t_ in ("f", "n")]
    hx, hy = P["head"]
    R = sp["head_r"]
    for pts, w, c in limbs + arms[:1]:
        lines(pts, w + lw * 2, (0, 0, 0))
        lines(pts, w, c)
    ctx.move_to(*body[0])
    for q in body[1:]:
        ctx.line_to(*q)
    ctx.close_path()
    path = ctx.copy_path()
    ctx.set_source_rgba(0, 0, 0, alpha)
    ctx.set_line_width(lw * 2)
    ctx.stroke()
    ctx.append_path(path)
    ctx.set_source_rgba(*cc("robe"), alpha)
    ctx.fill()
    if mud > 0.01 and h * math.sqrt(abs(m.xx * m.yy)) > 14:
        ctx.set_source_rgba(0, 0, 0, alpha)
        for i in range(int(3 + 8 * mud)):
            px = -hw * 0.8 + hw * 1.6 * hash01(i, ch.seed + 51)
            py = N[1] + 4 + (hem_y - N[1] - 6) * hash01(i, ch.seed + 52)
            r = 0.9 + 1.6 * hash01(i, ch.seed + 53)
            ctx.move_to(px + r, py)
            ctx.arc(px, py, r, 0, TAU)
        ctx.fill()
    # head
    ctx.move_to(hx + R + lw, hy)
    ctx.arc(hx, hy, R + lw, 0, TAU)
    ctx.set_source_rgba(0, 0, 0, alpha)
    ctx.fill()
    ctx.move_to(hx + R, hy)
    ctx.arc(hx, hy, R, 0, TAU)
    ctx.set_source_rgba(*cc("skin"), alpha)
    ctx.fill()
    hs = lk.get("hair_style")
    capc = None
    if lk.get("hood") == "up" or lk.get("veil"):
        capc = cc("robe")
    elif lk.get("hat") == "cap":
        capc = cc("cap")
    elif hs not in ("bald", "none", None):
        capc = cc("hair")
    if capc is not None:
        ctx.move_to(hx, hy)
        ctx.arc(hx, hy, R * 1.08, math.pi * 0.95 + sphi * 0.25, TAU + 0.05 + sphi * 0.3)
        ctx.close_path()
        ctx.set_source_rgba(*capc, alpha)
        ctx.fill_preserve()
        ctx.set_source_rgba(0, 0, 0, alpha)
        ctx.set_line_width(lw)
        ctx.stroke()
        if hs == "dreads" or hs == "long_glossy" or hs == "long":
            lines([(hx - R * 0.9, hy), (hx - R * 1.0, hy + R * 2.2)], R * 0.5, capc)
    if lk.get("hat") == "fez":
        ctx.rectangle(hx - R * 0.6, hy - R * 1.9, R * 1.2, R * 1.1)
        ctx.set_source_rgba(0.25, 0.25, 0.25, alpha)
        ctx.fill()
    if lk.get("beard") in ("long", "full", "goatee", "goatee_pointed"):
        L = R * (2.6 if lk["beard"] == "long" else 1.5)
        ctx.move_to(hx - R * 0.55 + R * 0.5 * sphi, hy + R * 0.45)
        ctx.line_to(hx + R * 0.55 + R * 0.5 * sphi, hy + R * 0.45)
        ctx.line_to(hx + R * 0.4 * sphi, hy + L)
        ctx.close_path()
        ctx.set_source_rgba(*cc("hair" if lk["beard"] != "goatee" else "beardc"), alpha)
        ctx.fill()
    # near arm + hands
    pts, w, c = arms[1]
    lines(pts, w + lw * 2, (0, 0, 0))
    lines(pts, w, c)
    for tag in ("n", "f"):
        x0, y0 = P["hand_" + tag]
        ctx.move_to(x0 + sp["hand"] + lw, y0)
        ctx.arc(x0, y0, sp["hand"] + lw, 0, TAU)
        ctx.set_source_rgba(0, 0, 0, alpha)
        ctx.fill()
        ctx.move_to(x0 + sp["hand"], y0)
        ctx.arc(x0, y0, sp["hand"], 0, TAU)
        ctx.set_source_rgba(*cc("hand"), alpha)
        ctx.fill()
    ctx.restore()


def draw_crowd(ctx, people, t):
    """draw many figures (sort them back-to-front yourself). returns list of anchors-lite per person"""
    m = ctx.get_matrix()
    dev = math.sqrt(abs(m.xx * m.yy - m.xy * m.yx))
    out = []
    for p in people:
        kind = p.get("kind", "citizen")
        seed = p.get("seed", 0)
        style = {k: p[k] for k in _STYLE_KW if k in p}
        ch = _crowd_char(kind, seed, style)
        h = p["h"]
        x, y = p["x"], p["y"]
        tt = t + p.get("t_off", 0.0)
        facing = p.get("facing", 1)
        pose = p.get("pose", "stand")
        alpha = p.get("alpha", 1.0)
        ink_p = p.get("render", ch.render) == "ink"
        if h * dev >= (120 if ink_p else 90) or p.get("lod") is not None:
            kw = {k: p[k] for k in ("rim", "tint", "silhouette", "wind", "turn", "speed", "layer", "light", "progress",
                                    "lod", "back", "head_turn", "render", "mud", "wet", "face_mask", "face", "chair",
                                    "legs_crossed", "flap", "wing_spread", "halo", "blink") if k in p}
            if ink_p and "lod" not in kw and h * dev < 260:
                kw["lod"] = 1
            a = ch.draw(ctx, x, y, h, pose, tt, p.get("mouth", 0.0), p.get("look", (0, 0)), p.get("expr", "neutral"),
                        facing, p.get("blink"), alpha, p.get("pole_len"), **kw)
            out.append(a)
            continue
        fs = -1 if facing < 0 else 1
        turn = p.get("turn", 0.0 if facing == 0 else R.PHI_DEFAULT)
        rs = _tiny_rig(ch, pose, tt, facing, turn, p.get("speed", 1.0))
        tint = p.get("tint")
        tint = (col(tint[0]), tint[1]) if tint else None
        sil = p.get("silhouette")
        sil = col(sil) if sil is not None else None
        if alpha > 0.002:
            _tiny_draw(ctx, ch, rs, x, y, h, fs, alpha, tint, sil, ink=p.get("render", ch.render) == "ink",
                       mud=p.get("mud", ch.lk.get("mud", 0.0)))
        rig, sp = rs
        k = h / 100.0
        Wp = lambda q: (x + fs * k * q[0], y + k * q[1])  # noqa: E731
        a = {"head": Wp(rig.P["head"]), "hand_r": Wp(rig.P["hand_n"]), "hand_l": Wp(rig.P["hand_f"]),
             "top": Wp((rig.P["head"][0], rig.P["head"][1] - sp["head_r"] * 1.1)), "feet": (x, y),
             "head_r": sp["head_r"] * k, "scale": k}
        if rig.pole is not None:
            bx, by, ang, _ = rig.pole
            X, Y = Wp((bx, by))
            a["pole"] = (X, Y, ang * fs)
        out.append(a)
    return out

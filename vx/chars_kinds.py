"""Per-kind looks for vx.chars (owned by RigMaster): palettes (seeded variety) and body/head drawing recipes.
Never yellow except Vexillian robes (palette `robe*`). Characters never carry markings or flags themselves."""
import math

import cairo

from .canvas import col
from .config import C
from .ease import hash01, noise1, clamp
from .chars_paint import (HP, p_circle, p_ellipse, p_capsule, p_smooth, p_poly, p_leaf, shadow_of, light_of, mixc,
                          BIG, TAU)

COMMON = dict(
    sclera="#F6F1E8", pupil="#17121A", iris="#4E3426", lip="#6E3438", mouth_in="#35131D", teeth="#F3EEE4",
    tongue="#C45E6A", blush="#E57A78", lash="#221A1F", herb="#7DA35A", leather="#6B4630", belt="#6B4A28",
    inner="#3A2616", shoe="#3E2E26", trouser="#3D3533", brow="#2C211E", cuff=None,
)

SKINS = ["skin1", "skin2", "skin3", "skin4"]
HAIRS = ["#1E1A1C", "#2E231F", "#4A3328", "#5B4A42", "#8A8790", "#C9C4BE", "#6A3B2E", "#2A2A33"]
CLOTH = ["cloth_grey", "cloth_blue", "cloth_teal", "cloth_plum", "cloth_rust", "concrete", "concrete_d",
         "#5B6B7F", "#6E6A7A", "#4F6360", "#7A6E73", "#3E4656"]
STYLES_H = ["short", "bob", "long", "bun", "curly", "pony", "buzz", "side", "bald", "messy"]
GARMENTS = ["jacket", "hoodie", "shirt", "dress", "coat", "sweater"]


def _c(x):
    return col(x)


def build_palette(kind, seed, style):
    r = lambda k: hash01(seed * 9173 + k, 4242)  # noqa: E731
    p = {k: _c(v) for k, v in COMMON.items() if v is not None}
    p["robe"] = C["robe"]
    look = dict(hair_style="short", garment="robe", sleeve="bell", hood=None, beard=None, crown=False, phone=False,
                prop=None, laurel=False, helmet=False, goggles_down=False, headset=False, veil=False, eyes="std",
                tail=False, pouch=False, blush_base=0.0)
    if kind == "lutie":
        p.update(skin=C["skin2"], hair=_c("#33241E"), brow=_c("#33241E"), iris=_c("#5A3824"), belt=_c("#7A5427"),
                 trouser=_c("#4A3A30"), shoe=_c("#5A3A28"))
        look.update(hair_style="messy", hood="down", pouch=True, eyes="big", blush_base=0.18)
    elif kind == "crocus":
        p.update(skin=C["skin1"], hair=_c("#EEEAE2"), brow=_c("#F4F1EA"), iris=_c("#4F5E70"), inner=_c("#16141B"),
                 black=_c("#1B1921"), veil=mixc(C["robe"], C["robe_light"], 0.45), shoe=_c("#241E24"),
                 trouser=_c("#1B1921"), belt=_c("#16141B"))
        look.update(hair_style="none", veil=True, beard="long", garment="crocus")
    elif kind in ("schismmancer", "commander"):
        sk = SKINS[int(r(1) * 4) % 4]
        p.update(skin=C[sk], hair=_c("#1D1917"), brow=_c("#1D1917"), vest=_c("#2A2F36"), trouser=_c("#30343A"),
                 pad=_c("#1C1F24"), shoe=_c("#18191D"), helmet=_c("#3E443D"), nvg=_c("#202327"), lens=_c("#78FF9C"),
                 belt=_c("#23272C"), headset=_c("#1A1B1F"), iris=_c("#3A2A22"))
        look.update(garment="tactical", helmet=True, hair_style="none", headset=(kind == "commander"),
                    beard=("stubble" if r(2) < 0.4 else None))
    elif kind in ("cultist", "acolyte"):
        sk = SKINS[int(r(1) * 4) % 4]
        if kind == "cultist":
            robe = _c(["cloth_plum", "#5E5968", "#4F4358", "#6E5A6E"][int(r(2) * 4) % 4])
            if isinstance(robe, str):
                robe = _c(robe)
        else:
            robe = C["robe"]
        p.update(skin=C[sk], robe=robe, hair=_c(HAIRS[int(r(3) * 5) % 5]), brow=_c(HAIRS[int(r(3) * 5) % 5]),
                 iris=_c(["#3E2A20", "#4E5E6E", "#2F3E2F"][int(r(4) * 3) % 3]))
        look.update(hood="up", hair_style="short")
    elif kind in ("philosopher", "crackpot"):
        sk = SKINS[int(r(1) * 4) % 4]
        hc = _c(["#D8D5D0", "#EDEBE7", "#8F8C94", "#3A302C"][int(r(2) * 4) % 4])
        p.update(skin=C[sk], robe=C["bone"], hair=hc, brow=hc, laurel=_c("#6F8A55"), shoe=_c("#6B4630"),
                 iris=_c("#40302A"))
        look.update(garment="toga", sleeve="bare", laurel=(kind == "philosopher" or r(3) < 0.5),
                    hair_style="wild" if kind == "crackpot" else ["curly", "short", "side"][int(r(4) * 3) % 3],
                    beard="full" if r(5) < 0.75 or kind == "philosopher" else None)
    elif kind == "jaguar":
        p.update(skin=_c("#CF8A45"), fur=_c("#CF8A45"), fur_dark=_c("#3A2517"), muzzle=_c("#F2E6D3"),
                 suit=_c("#1F2B48"), lapel=_c("#17213A"), shirt=_c("#EEF0F3"), tie=_c("#B3262E"), shoe=_c("#141419"),
                 trouser=_c("#1F2B48"), nose=_c("#6E3A3C"), iris=_c("#93BA58"), brow=_c("#3A2517"),
                 lash=_c("#2A1A12"), cuff=_c("#EEF0F3"), lid=_c("#C07E3E"))
        look.update(garment="suit", sleeve="fit", tail=True, hair_style="none", eyes="std")
    else:  # citizen
        sk = SKINS[int(r(1) * 4) % 4]
        top = CLOTH[int(r(2) * len(CLOTH)) % len(CLOTH)]
        bot = CLOTH[int(r(3) * len(CLOTH)) % len(CLOTH)]
        topc = _c(top)
        botc = shadow_of(_c(bot), 0.78)
        hs = STYLES_H[int(r(4) * len(STYLES_H)) % len(STYLES_H)]
        hc = _c(HAIRS[int(r(5) * len(HAIRS)) % len(HAIRS)])
        g = GARMENTS[int(r(6) * len(GARMENTS)) % len(GARMENTS)]
        p.update(skin=C[sk], robe=topc, hair=hc, brow=hc, trouser=botc,
                 shoe=_c(["#2A2A30", "#4A3A34", "#5E6270", "#1E2026"][int(r(7) * 4) % 4]),
                 iris=_c(["#3E2A20", "#4E5E6E", "#2F3E2F", "#1E1614"][int(r(8) * 4) % 4]),
                 inner=shadow_of(topc, 0.6))
        look.update(garment=g, sleeve="fit", hair_style=hs, beard=("stubble" if r(9) < 0.2 else
                                                                   ("full" if r(9) < 0.3 else None)))
    # overrides
    for k in ("skin", "hair", "robe", "iris", "trouser", "shoe", "vest", "helmet", "suit", "tie", "fur", "brow"):
        if k in style:
            p[k] = _c(style[k])
            if k == "hair" and "brow" not in style:
                p["brow"] = p["hair"]
    for k in ("hood", "crown", "phone", "prop", "laurel", "goggles_down", "headset", "beard", "garment", "sleeve",
              "hair_style", "pouch", "tail", "veil"):
        if k in style:
            look[k] = style[k]
    if look.get("hood") is True:
        look["hood"] = "up"
    elif look.get("hood") is False:
        look["hood"] = None
    if style.get("hat") == "crown":
        look["crown"] = True
    look["crown_color"] = _c(style.get("crown_color", "#C8CCD6"))
    look["phone_color"] = _c(style.get("phone_color", "screen_glow"))
    if look["phone"] and not look["prop"]:
        look["prop"] = "phone"
    p["hand"] = p.get("fur", p["skin"]) if kind == "jaguar" else p["skin"]
    p.setdefault("sleeve", p["robe"])
    if look["garment"] == "suit":
        p["sleeve"] = p["suit"]
    if look["garment"] == "crocus":
        p["sleeve"] = C["robe"]
    p["blush_base"] = look["blush_base"]
    p["cuffin"] = mixc(shadow_of(p["sleeve"], 0.5), p.get("inner", p["sleeve"]), 0.35)
    p["lid"] = p.get("lid", shadow_of(p["skin"], 0.93))
    p["crown"] = look["crown_color"]
    return p, look


# ============================================================ helpers
def _hp(P):
    r = P.rig
    return HP(P.sp["head_r"], P.o["head_yaw"], r.head_pitch * 0.7 + P.o["look"][1] * -0.18)


def _hood_outer(P, hp, pointed=0.0, drop=1.55, width=1.22):
    R = P.sp["head_r"]
    s = math.sin(hp.yaw)
    back = -s * R * 0.3
    shw = max(P.sp["sh_w"] / R, 0.75) * 0.92
    pts = [
        (back * 0.4 + s * R * 0.1, -R * (1.24 + pointed)),
        (R * width * 0.8 + back * 0.2, -R * 0.84),
        (R * width + back * 0.1, -R * 0.02),
        (R * width * 0.98 + back * 0.35, R * 0.7),
        (R * shw * 1.12 + back * 0.4, R * drop),
        (-R * shw * 1.12 + back * 1.2, R * drop),
        (-R * width * 1.02 + back * 1.4, R * 0.7),
        (-R * width + back * 1.35, -R * 0.08),
        (-R * width * 0.82 + back, -R * 0.9),
    ]
    return pts


def _hood(P, hp, colour, pointed=0.0, shadow_face=0.0, open_g=0.95, lining="inner"):
    ctx = P.ctx
    R = P.sp["head_r"]
    outer = _hood_outer(P, hp, pointed)
    w = P.o["wind_x"]
    if abs(w) > 1e-3:
        t = P.t
        aw = abs(w)
        out = []
        for i, (x, y) in enumerate(outer):
            f = clamp((y + R * 0.9) / (R * 2.5))
            fl = math.sin(t * (8 + 3 * aw) + i * 1.3) * 0.6 + 0.4 * math.sin(t * 13.0 + i)
            down = 1.0 if x * w > 0 else 0.4
            out.append((x + w * R * (0.06 + 0.16 * f) * down + fl * aw * R * 0.05 * f,
                        y - aw * R * 0.05 * f * down + fl * aw * R * 0.03))
        outer = out
    opening = hp.cap(0.0, -0.12, open_g, 1.0, 1.0, 32)
    if hp.yaw > math.pi * 0.62:
        opening = None
    ctx.save()
    if opening:
        ctx.new_path()
        ctx.rectangle(-BIG, -BIG, 2 * BIG, 2 * BIG)
        p_poly(ctx, opening)
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.clip()
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    P.part(lambda c: p_smooth(c, outer, True, 0.42), colour, d=R * 0.22)
    ctx.restore()
    if opening and P.lod >= 1:
        # lining rim + hood shadow over the brow
        big = hp.cap(0.0, -0.12, open_g + 0.1, 1.0, 1.0, 32)
        ctx.save()
        ctx.new_path()
        p_smooth(ctx, outer, True, 0.42)
        ctx.clip()
        ctx.new_path()
        p_poly(ctx, big)
        p_poly(ctx, opening[::-1])
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        P.src(shadow_of(P.pal[colour] if isinstance(colour, str) else colour, 0.62))
        ctx.fill()
        ctx.restore()
        if shadow_face > 0 and P.sil is None:
            ctx.save()
            ctx.new_path()
            p_poly(ctx, opening)
            ctx.clip()
            ys = [q[1] for q in opening]
            y0, y1 = min(ys), max(ys)
            g = cairo.LinearGradient(0, y0, 0, y0 + (y1 - y0) * 0.75)
            sc = P.C(shadow_of(P.pal[colour] if isinstance(colour, str) else colour, 0.35))
            g.add_color_stop_rgba(0, sc[0], sc[1], sc[2], shadow_face)
            g.add_color_stop_rgba(1, sc[0], sc[1], sc[2], 0)
            ctx.set_source(g)
            ctx.paint()
            ctx.restore()


def _beard(P, hp, length=2.4, colour="hair", full=True):
    R = P.sp["head_r"]
    t = P.t
    wind = P.o["wind_x"]
    sw = wind * 0.9 + 0.12 * noise1(t * 0.7 + 3, 61) + P.rig.sway * 0.03
    cap = hp.cap(0.0, -1.2, 0.98 if full else 0.9, 1.07, 1.0, 40)
    if len(cap) < 3:
        return
    low = [q for q in cap if q[1] > R * 0.25]
    hang = None
    if length > 1.3 and len(low) >= 2:
        a = min(low, key=lambda q: q[0])
        b = max(low, key=lambda q: q[0])
        chin = hp.pt(0.0, -0.95, 1.07)
        tipx = chin[0] + sw * R * 0.9
        tipy = R * (0.9 + length)
        wv = [0.25 * math.sin(t * 1.3 + k) for k in range(4)]
        w = (b[0] - a[0]) * 0.5
        hang = [a, (a[0] - w * 0.05 + sw * R * 0.2, a[1] + R * 0.8 + wv[0]),
                (tipx - w * 0.45 + sw * R * 0.5, tipy - R * 0.6), (tipx, tipy),
                (tipx + w * 0.4 + sw * R * 0.5, tipy - R * 0.7), (b[0] + w * 0.03 + sw * R * 0.2, b[1] + R * 0.7 + wv[1]),
                b, ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5 - R * 0.3)]

    def build(c):
        p_poly(c, cap)
        if hang:
            p_smooth(c, hang, True, 0.38)
    P.part(build, colour, d=R * 0.28)
    if P.lod >= 2 and hang:
        tipx, tipy = hang[3]
        for k in range(4):
            x0 = (hang[0][0] + hang[6][0]) * 0.5 + (k - 1.5) * R * 0.3
            P.stroke(lambda c: (c.move_to(x0, R * 0.95), c.curve_to(x0 + sw * R * 0.2, R * 1.5, x0 * 0.5 + tipx * 0.5,
                                                                    tipy - R * 0.9, tipx * 0.6 + x0 * 0.4 + sw * R * 0.3,
                                                                    tipy - R * 0.35)),
                     shadow_of(P.pal[colour], 0.84), R * 0.045, 0.55)


def _mustache(P, hp, colour="hair"):
    R = P.sp["head_r"]
    base = hp.pt(0.0, -0.33, 1.06)
    if base[2] < 0.05:
        return
    sw = P.o["wind_x"] * 0.2
    for side in (-1, 1):
        tip = hp.pt(side * 0.55, -0.58, 1.02)
        if tip[2] < 0:
            continue
        ang = math.atan2(tip[1] - base[1], tip[0] - base[0]) + sw
        L = math.hypot(tip[0] - base[0], tip[1] - base[1]) * 1.2
        P.part(lambda c: p_leaf(c, base[0], base[1], L, R * 0.13, ang), colour, d=R * 0.06)


def age_lines(P, hp, k=1.0):
    """forehead lines + crow's feet for the ancient ones (lod 2 only)"""
    if P.lod < 2:
        return
    R = P.sp["head_r"]
    c = shadow_of(P.pal["skin"], 0.82)
    for j, lat in enumerate((0.62, 0.72)):
        pts = [hp.pt(lon, lat - 0.02 * abs(lon), 1.0) for lon in (-0.45, -0.2, 0.0, 0.2, 0.45)]
        vis = [q[:2] for q in pts if q[2] > 0.2]
        if len(vis) >= 3:
            P.stroke(lambda cc: (cc.move_to(*vis[0]), [cc.line_to(*q) for q in vis[1:]]), c, R * 0.03, 0.55 * k)
    for side in (-1, 1):
        lon = side * P.sp["eye_sep"] * 1.55
        for d in (-0.08, 0.02):
            a = hp.pt(lon, P.sp["eye_y"] + d, 1.0)
            b = hp.pt(lon + side * 0.14, P.sp["eye_y"] + d * 1.8, 1.0)
            if a[2] > 0.2 and b[2] > 0.1:
                P.stroke(lambda cc: (cc.move_to(a[0], a[1]), cc.line_to(b[0], b[1])), c, R * 0.028, 0.6 * k)


def _stubble(P, hp):
    if P.lod < 2:
        return
    cap = hp.cap(0.0, -0.8, 0.62, 1.0, 1.0, 24)
    P.flat(lambda c: p_poly(c, cap), shadow_of(P.pal["skin"], 0.72), 0.35)


def _crown(P, hp):
    R = P.sp["head_r"]
    seed = P.ch.seed
    tilt = (hash01(seed, 71) - 0.5) * 0.5
    ctx = P.ctx
    top = hp.pt(0.0, math.pi / 2 - 0.25, 1.05)
    cx, cy = top[0] * 0.7, -R * 1.02
    w = R * 0.58
    hgt = R * 0.52
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(tilt)
    e = R * 0.1 * (0.5 + 0.5 * abs(math.cos(hp.yaw * 0)))
    pts = [(-w, 0.0)]
    n = 5
    for i in range(n):
        x1 = -w + 2 * w * (i + 0.5) / n
        pts.append((x1, -hgt * (1.0 if i % 2 == 0 else 0.72)))
        pts.append((-w + 2 * w * (i + 1) / n, -hgt * 0.35))
    pts[-1] = (w, 0.0)
    pts.append((w, 0.0))

    def b(c):
        c.move_to(-w, 0)
        for q in pts[1:]:
            c.line_to(*q)
        c.curve_to(w * 0.5, e, -w * 0.5, e, -w, 0)
        c.close_path()
    P.part(b, "crown", d=R * 0.1)
    if P.lod >= 2:
        for i, gx in enumerate((-0.5, 0.0, 0.5)):
            P.flat(lambda c: p_circle(c, gx * w, -hgt * 0.15, R * 0.07), ["#8C4A7E", "#3F7E8C", "#8C4A7E"][i])
        for i in range(0, n, 2):
            x1 = -w + 2 * w * (i + 0.5) / n
            P.flat(lambda c: p_circle(c, x1, -hgt * 1.02, R * 0.05), light_of(P.pal["crown"], 0.5))
    ctx.restore()


def _laurel(P, hp):
    R = P.sp["head_r"]
    for side in (-1, 1):
        for i in range(8):
            lon = side * (2.6 - i * 0.3)
            x, y, z = hp.pt(lon, 0.42 - 0.04 * i * 0, 1.06)
            if z < -0.05:
                continue
            nx, ny, nz = hp.pt(lon + side * -0.3, 0.42, 1.06)
            ang = math.atan2(ny - y, nx - x)
            for k in (-1, 1):
                P.part(lambda c: p_leaf(c, x, y, R * 0.34, R * 0.09, ang + k * 0.55), "laurel", d=R * 0.05, rim=0.5)


def _helmet(P, hp, goggles_down):
    R = P.sp["head_r"]
    dome = hp.cap(math.pi, 1.22, 1.36, 1.12, 1.0, 40)
    P.part(lambda c: p_poly(c, dome), "helmet", d=R * 0.2)
    if P.lod >= 1:
        # brim band along the front edge
        brim = [q for q in hp.cap(math.pi, 1.22, 1.36, 1.125, 1.0, 40)]
        P.stroke(lambda c: (c.move_to(*brim[0]), [c.line_to(*q) for q in brim[1:]], c.close_path()),
                 shadow_of(P.pal["helmet"], 0.7), R * 0.09)
    # side rail
    if P.lod >= 2:
        rx, ry, rz = hp.pt(-math.pi / 2 + 0.2, 0.35, 1.15)
        if rz > -0.3:
            P.flat(lambda c: p_ellipse(c, rx, ry, R * 0.2, R * 0.08, -0.2), shadow_of(P.pal["helmet"], 0.7))
    # chin strap
    if P.lod >= 1:
        for side in (-1, 1):
            pts = [hp.pt(side * lon, lat, 1.02) for lon, lat in ((1.45, 0.18), (1.2, -0.45), (0.75, -0.85), (0.2, -1.02))]
            vis = [q[:2] for q in pts if q[2] > -0.05]
            if len(vis) >= 2:
                P.stroke(lambda c: (c.move_to(*vis[0]), [c.line_to(*q) for q in vis[1:]]),
                         light_of(P.pal["nvg"], 0.12), R * 0.055)
    # NVG
    mnt = hp.pt(0.0, 0.74, 1.14)
    if mnt[2] > -0.35:
        P.part(lambda c: p_ellipse(c, mnt[0], mnt[1], R * 0.2, R * 0.13), "nvg", d=R * 0.05)
        for side in (-1, 1):
            lon = side * P.sp["eye_sep"] * 0.9
            if goggles_down:
                b0 = hp.pt(lon, 0.05, 1.1)
                b1 = hp.pt(lon, 0.02, 1.55)
            else:
                b0 = hp.pt(lon * 0.55, 0.74, 1.14)
                b1 = hp.pt(lon * 0.55, 0.9, 1.34)
            if b1[2] < -0.3 and b0[2] < -0.3:
                continue
            tr = R * (0.17 if goggles_down else 0.12)
            P.part(lambda c: p_capsule(c, b0[:2], tr, b1[:2], tr * 1.05), "nvg", d=R * 0.07)
            if goggles_down and b1[2] > -0.1:
                # lens (green glow — never yellow)
                P.ctx.save()
                P.raw("lens", 1.0)
                p_ellipse(P.ctx, b1[0], b1[1], tr * 0.95 * max(0.35, abs(b1[2])) ** 0.5, tr * 0.95)
                P.ctx.fill()
                g = cairo.RadialGradient(b1[0], b1[1], 0, b1[0], b1[1], tr * 3.2)
                lc = col(P.pal["lens"])
                g.add_color_stop_rgba(0, lc[0], lc[1], lc[2], 0.5)
                g.add_color_stop_rgba(1, lc[0], lc[1], lc[2], 0.0)
                P.ctx.set_source(g)
                p_circle(P.ctx, b1[0], b1[1], tr * 3.2)
                P.ctx.fill()
                P.ctx.restore()
        if goggles_down:
            P.stroke(lambda c: (c.move_to(mnt[0], mnt[1]), c.line_to(*hp.pt(0.0, 0.1, 1.3)[:2])), "nvg", R * 0.12)


def _headset(P, hp):
    R = P.sp["head_r"]
    ex, ey, ez = hp.pt(-math.pi / 2, 0.0, 1.06)
    if ez > -0.4:
        # headband over the helmet: great circle through the ears and the top
        band = []
        for i in range(17):
            v = (math.cos(math.pi * i / 16) * -1, math.sin(math.pi * i / 16), 0.0)
            band.append(hp.vec(v, 1.22))
        vis = [q[:2] for q in band if q[2] > -0.15]
        if len(vis) > 1:
            P.stroke(lambda c: (c.move_to(*vis[0]), [c.line_to(*q) for q in vis[1:]]), "headset", R * 0.1)
        P.part(lambda c: p_ellipse(c, ex, ey, R * 0.28 * max(0.5, abs(ez)) ** 0.5, R * 0.34), "headset", d=R * 0.08)
        mt = hp.pt(-0.28, -0.55, 1.14)
        if mt[2] > -0.2:
            P.stroke(lambda c: (c.move_to(ex, ey + R * 0.15), c.curve_to(ex + (mt[0] - ex) * 0.2, mt[1] + R * 0.3,
                                                                          mt[0] - R * 0.2, mt[1] + R * 0.2, mt[0], mt[1])),
                     "headset", R * 0.06)
            P.part(lambda c: p_circle(c, mt[0], mt[1], R * 0.1), "headset", d=R * 0.03)


# ============================================================ jaguar head
_ROSETTES = [(-0.2, 0.9, 0.09), (0.26, 0.98, 0.08), (-0.62, 1.1, 0.1), (0.05, 1.3, 0.09),
             (-1.18, 0.42, 0.13), (1.18, 0.42, 0.13), (-1.05, -0.12, 0.1), (1.05, -0.12, 0.1),
             (-1.55, 0.75, 0.14), (1.55, 0.75, 0.14), (-1.75, 0.05, 0.12), (1.75, 0.05, 0.12),
             (-2.3, 0.45, 0.15), (2.3, 0.45, 0.15), (3.0, 0.7, 0.15), (-2.7, -0.05, 0.13), (2.8, 0.0, 0.13)]


def jaguar_head(P, hp, E, blink, look, mouth):
    R = P.sp["head_r"]
    t = P.t
    ctx = P.ctx
    lod = P.lod
    yaw = hp.yaw
    # ears (behind)
    for side in (-1, 1):
        ex, ey, ez = hp.pt(side * 0.78, 0.82, 1.0)
        er = R * 0.3
        P.part(lambda c: p_circle(c, ex, ey - R * 0.06, er), "fur", d=R * 0.08)
        if ez > 0.05 and lod >= 1:
            P.flat(lambda c: p_ellipse(c, ex, ey - R * 0.02, er * 0.55 * ez ** 0.5, er * 0.6), "muzzle", 0.9)
    # cheek ruff + head
    m = mouth

    def headshape(c):
        n = 44
        pts = []
        cxo = R * 0.18 * math.sin(yaw)
        for i in range(n):
            a = TAU * i / n
            ca, sa = math.cos(a), math.sin(a)
            rr = R * 1.0
            if sa > 0.05:  # lower half: wider cheeks with fur tufts
                rr = R * (1.08 + 0.06 * (abs(math.sin(a * 6)) if lod >= 1 else 0)) * (1 + 0.08 * sa)
            y = sa * rr * (0.96 if sa < 0 else 0.92 + 0.06 * m)
            pts.append((ca * rr * 1.06 + cxo * max(sa, 0), y))
        p_smooth(c, pts, True, 0.4)
    P.part(headshape, "fur", d=R * 0.2)
    if lod >= 1:
        # rosettes
        for (lon, lat, s) in _ROSETTES:
            x, y, z = hp.pt(lon, lat, 1.0)
            if z < 0.12:
                continue
            f = z ** 0.7
            rx, ry = R * s * f, R * s
            if lod >= 2:
                P.flat(lambda c: p_ellipse(c, x, y, rx * 0.82, ry * 0.82), shadow_of(P.pal["fur"], 0.86), 0.9)
                if s < 0.1:
                    P.flat(lambda c: p_ellipse(c, x, y, rx * 0.7, ry * 0.7), "fur_dark", 0.9)
                else:
                    for k in range(3):
                        a0 = k * TAU / 3 + lon * 3 + 0.3
                        P.stroke(lambda c: (c.new_sub_path(), c.save(), c.translate(x, y), c.scale(rx, ry),
                                            c.arc(0, 0, 1, a0, a0 + 1.72), c.restore()), "fur_dark", R * s * 0.5)
            else:
                P.flat(lambda c: p_ellipse(c, x, y, rx * 0.6, ry * 0.6), "fur_dark", 0.8)
    # pale fur around the eyes
    if lod >= 1:
        for side in (-1, 1):
            ex, ey, ez = hp.pt(side * P.sp["eye_sep"], P.sp["eye_y"] - 0.02, 1.0)
            if ez > 0.15:
                P.flat(lambda c: p_ellipse(c, ex, ey + R * 0.02, R * 0.3 * ez ** 0.6, R * 0.25), "muzzle", 0.55)
    # muzzle (drops with the jaw)
    jaw = m * 0.18
    muz = hp.cap(0.0, -0.42 - jaw * 0.5, 0.62 + jaw * 0.3, 1.0, 1.0, 30)
    P.part(lambda c: p_smooth(c, muz, True, 0.3), "muzzle", d=R * 0.12)
    # eyes (+ dark liner)
    P.face(hp, E, blink, look, 0.0, show_mouth=False, blush=False, brow="fur_dark", lash="fur_dark", brow_w=0.065,
           brow_len=0.9, brow_in=0.62)
    if lod >= 2:
        for side in (-1, 1):
            a = hp.pt(side * 0.24, -0.06, 1.0)
            b = hp.pt(side * 0.26, -0.2, 1.0)
            if a[2] > 0.2:
                P.stroke(lambda c: (c.move_to(a[0], a[1]), c.line_to(b[0], b[1])), "fur_dark", R * 0.045, 0.85)
    # nose
    nz = hp.pt(0.0, -0.2, 1.1)
    if nz[2] > 0.0:
        nw = R * 0.19 * (0.6 + 0.4 * nz[2])
        P.part(lambda c: (c.move_to(nz[0] - nw, nz[1] - R * 0.08), c.line_to(nz[0] + nw, nz[1] - R * 0.08),
                          c.curve_to(nz[0] + nw * 0.4, nz[1] + R * 0.08, nz[0] + nw * 0.2, nz[1] + R * 0.12,
                                     nz[0], nz[1] + R * 0.13),
                          c.curve_to(nz[0] - nw * 0.2, nz[1] + R * 0.12, nz[0] - nw * 0.4, nz[1] + R * 0.08,
                                     nz[0] - nw, nz[1] - R * 0.08), c.close_path()), "nose", d=R * 0.04)
        if lod >= 2:
            P.flat(lambda c: p_ellipse(c, nz[0] - nw * 0.3, nz[1] - R * 0.04, nw * 0.25, R * 0.025),
                   light_of(P.pal["nose"], 0.35), 0.8)
    # mouth + fangs
    mo = clamp(m + E["mo"])
    mx, my, mz = hp.pt(0.0, -0.5 - jaw * 0.4, 1.02)
    if mz > 0.1:
        # philtrum
        P.stroke(lambda c: (c.move_to(nz[0], nz[1] + R * 0.12), c.line_to(mx, my - R * 0.04)), "fur_dark", R * 0.04)
        W = R * 0.33 * E["mw"] * (1 + 0.15 * E["sm"])
        sm = E["sm"]
        if mo > 0.05:
            Hm = R * 0.42 * mo

            def b(c):
                c.move_to(mx - W, my - sm * R * 0.08)
                c.curve_to(mx - W * 0.4, my - R * 0.06, mx + W * 0.4, my - R * 0.06, mx + W, my - sm * R * 0.08)
                c.curve_to(mx + W * 0.7, my + Hm, mx - W * 0.7, my + Hm, mx - W, my - sm * R * 0.08)
                c.close_path()
            ctx.save()
            ctx.new_path()
            b(ctx)
            P.src("mouth_in")
            ctx.fill_preserve()
            ctx.clip()
            if lod >= 2:
                P.src("tongue")
                p_ellipse(ctx, mx, my + Hm * 0.85, W * 0.6, Hm * 0.4)
                ctx.fill()
                P.src("teeth")
                for s in (-1, 1):
                    fx = mx + s * W * 0.55
                    ctx.move_to(fx - R * 0.06, my - R * 0.08)
                    ctx.line_to(fx + R * 0.06, my - R * 0.08)
                    ctx.line_to(fx, my + R * 0.1 + Hm * 0.12)
                    ctx.close_path()
                    ctx.fill()
            ctx.restore()
        else:
            P.stroke(lambda c: (c.move_to(mx - W, my - sm * R * 0.1),
                                c.curve_to(mx - W * 0.4, my + R * 0.05, mx - W * 0.1, my + R * 0.05, mx, my),
                                c.curve_to(mx + W * 0.1, my + R * 0.05, mx + W * 0.4, my + R * 0.05, mx + W,
                                           my - sm * R * 0.1 + E["sk"] * R * 0.08)), "fur_dark", R * 0.05)
    # whiskers
    if lod >= 2:
        sw = 0.08 * math.sin(t * 3.0) + m * 0.12
        for side in (-1, 1):
            base = hp.pt(side * 0.32, -0.38, 1.04)
            if base[2] < 0.1:
                continue
            for k in range(3):
                ang = (0.0 if side > 0 else math.pi) + side * (-0.25 + 0.22 * k + sw)
                L = R * (0.95 - 0.12 * k)
                P.stroke(lambda c: (c.move_to(base[0], base[1] + k * R * 0.05),
                                    c.curve_to(base[0] + math.cos(ang) * L * 0.5, base[1] + math.sin(ang) * L * 0.5,
                                               base[0] + math.cos(ang) * L * 0.8, base[1] + math.sin(ang) * L * 0.8 + R * 0.05,
                                               base[0] + math.cos(ang) * L, base[1] + math.sin(ang) * L + R * 0.1)),
                         "muzzle", R * 0.022, 0.9)
            for k in range(3):
                d = hp.pt(side * (0.2 + 0.09 * k), -0.4 + 0.05 * (k % 2), 1.05)
                P.flat(lambda c: p_circle(c, d[0], d[1], R * 0.025), "fur_dark", 0.8)


def jaguar_tail(P):
    r = P.rig
    sp = P.sp
    Hh = r.B["hip"]
    t = P.t
    sw = math.sin(t * 1.3) * 2.5 + P.o["wind_x"] * 4.0
    d = sp["dep"]
    pts3 = [(Hh[0] - d * 0.9, Hh[1] - 4, 0.0), (Hh[0] - d - 7, Hh[1] - 12, -3 - sw * 0.3),
            (Hh[0] - d - 12 - sw * 0.5, Hh[1] - 24, -8 - sw), (Hh[0] - d - 15 - sw, Hh[1] - 30, -14 - sw * 1.5),
            (Hh[0] - d - 20 - sw, Hh[1] - 26, -19 - sw * 1.6)]
    pts = [P.pj(q) for q in pts3]
    rs = [3.4, 3.1, 2.8, 2.6, 2.4]
    from .chars_paint import p_chain
    P.part(lambda c: p_chain(c, pts, rs), "fur", d=1.2)
    P.part(lambda c: p_circle(c, pts[-1][0], pts[-1][1], 2.5), "fur_dark", d=0.6)
    if P.lod >= 2:
        for i in (1, 2, 3):
            P.flat(lambda c: p_ellipse(c, pts[i][0], pts[i][1], 1.3, 1.0), "fur_dark", 0.85)

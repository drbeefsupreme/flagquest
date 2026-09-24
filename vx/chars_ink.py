"""Film 2 (THE FLAG GAME?) additions to vx.chars (owned by RigMaster):
the "ink" palette (grey tone values only), Chick-tract and anime faces, and the tract cast's accessories/props.

Everything here draws through the Painter (P.part / P.flat / P.stroke / P.ink_line), so in ink mode it yields
grey tones + black brush contours + white highlights, and in colour mode ordinary flat colour.
Coordinates: head pieces in head-local units (after P.head_frame, projected with HP); body pieces in body-local
units (h/100) like vx.chars_paint.
"""
import math

import cairo

from .canvas import col
from .ease import hash01, noise1, clamp
from .chars_paint import (p_circle, p_ellipse, p_capsule, p_chain, p_smooth, p_poly, p_leaf, shadow_of, mixc, TAU)

WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)


def lum(c):
    c = col(c)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def g(v):
    return (v, v, v)


# ============================================================ ink palette (tone values: 0 ink .. 1 paper)
INK_COMMON = dict(skin=0.99, hand=0.99, sclera=1.0, pupil=0.0, iris=0.62, lash=0.0, brow=0.0, lip=0.5,
                  mouth_in=0.0, teeth=1.0, tongue=0.72, blush=0.86, lid=0.99, herb=0.8, leather=0.8, belt=0.8,
                  shoe=0.05, trouser=0.99, inner=0.05, black=0.05, fur=0.99, goggle=0.05, lens=0.8, mask=0.99,
                  tract=0.99, hoop=0.99, cap=0.05, capfront=0.99, beardc=0.99, apron=0.8, tool=0.05, wood=0.8,
                  horn=0.99, lyre=0.99, wing=0.99, halo=1.0, paint=0.99, tutu=0.99, chair=0.8, chairf=0.99,
                  mud=0.05, cuffin=0.05, hair=0.99, nose=0.3, muzzle=0.99, fur_dark=0.05)
INK_KIND = {
    "summer": dict(hair=0.99, brow=0.35, robe=0.99, sleeve=0.99, trouser=0.99, fur=0.99, iris=0.7, lip=0.5),
    "raven": dict(hair=0.0, brow=0.0, robe=0.06, sleeve=0.06, trouser=0.05, shoe=0.02, lip=0.12, iris=0.4,
                  cuffin=0.0),
    "crow": dict(hair=0.99, brow=0.15, robe=0.04, sleeve=0.04, trouser=0.99, fur=0.99, cap=0.02, lens=0.75,
                 cuffin=0.0,
                 beardc=0.99),
    "flagmaker": dict(robe=0.99, sleeve=0.99, hair=0.99, brow=0.15, apron=0.8, tool=0.05),
    "pharisee": dict(inner=0.2),
    "devil": dict(skin=0.99, hand=0.99, robe=0.04, sleeve=0.04, hair=0.0, brow=0.0, iris=0.99, cuffin=0.0, lid=0.8,
                  horn=0.99),
    "seraph": dict(robe=0.99, sleeve=0.99, hair=0.99, brow=0.5, iris=0.7),
    "jaguar": dict(fur=0.86, skin=0.86, hand=0.86, fur_dark=0.0, muzzle=1.0, suit=0.2, lapel=0.1, shirt=1.0,
                   tie=0.4, lid=0.8, iris=0.6),
}


def ink_palette(kind, pal):
    """grey (v, v, v) palette for the print styles; clothes/hair not listed map from their luminance"""
    out = {}
    for k, v in pal.items():
        if isinstance(v, (int, float)):
            out[k] = v
            continue
        L = lum(v)
        out[k] = g(0.99 if L > 0.19 else 0.05)
    for k, v in INK_COMMON.items():
        out[k] = g(v)
    if kind in ("hippie", "citizen", "pharisee", "cultist", "acolyte", "philosopher", "crackpot"):
        for k in ("robe", "sleeve", "trouser", "hair", "brow", "fur"):
            if k in pal and not isinstance(pal[k], (int, float)):
                L = lum(pal[k])
                out[k] = g(0.99 if L > 0.2 else 0.05)
        if "hair" in pal:
            out["brow"] = g(min(0.35, lum(pal["hair"])))
    for k, v in INK_KIND.get(kind, {}).items():
        out[k] = g(v)
    if kind not in INK_KIND and "robe" in pal and kind in ("lutie", "crocus", "acolyte", "schismmancer", "commander"):
        out["robe"] = out["sleeve"] = g(0.9)     # ochre robes print light so Flags rise above them
    out["blush_base"] = pal.get("blush_base", 0.0)
    return out


# ============================================================ helpers
def hq(hp, name, default=1.0):
    """per-kind head construction parameter (vx.chars_paint.HEADP) or default"""
    if hp.egg is None:
        return default
    from .chars_paint import _HKEYS
    return hp.egg[2][_HKEYS.index(name)]


def nose_dn(hp):
    return -0.18 * hq(hp, "nose") if hp.egg is not None else 0.0

def tapered(P, pts, w0, w1, c="lash", wmid=None, a=1.0):
    """filled brush stroke along a polyline: width w0 at start, w1 at end (wmid in the middle)"""
    n = len(pts)
    if n < 2:
        return
    L, Rr = [], []
    for i in range(n):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[min(i + 1, n - 1)]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        ln = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / ln, dx / ln
        u = i / (n - 1)
        if wmid is not None:
            w = (w0 + (wmid - w0) * (u * 2)) if u < 0.5 else (wmid + (w1 - wmid) * (u * 2 - 1))
        else:
            w = w0 + (w1 - w0) * u
        w *= 0.5
        L.append((pts[i][0] + nx * w, pts[i][1] + ny * w))
        Rr.append((pts[i][0] - nx * w, pts[i][1] - ny * w))
    poly = L + Rr[::-1]
    P.flat(lambda cc: p_smooth(cc, poly, True, 0.3), c, a)


def bez(p0, p1, p2, n=7):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in [i / (n - 1) for i in range(n)]]


def drop(P, x, y, r, up=False, a=1.0):
    """sweat drop / tear drop: white with an ink contour"""
    s = -1 if up else 1

    def b(c):
        c.move_to(x, y - s * r * 1.9)
        c.curve_to(x + r * 0.35, y - s * r * 0.9, x + r, y - s * r * 0.2, x + r, y + s * r * 0.25)
        c.curve_to(x + r, y + s * r * 0.85, x - r, y + s * r * 0.85, x - r, y + s * r * 0.25)
        c.curve_to(x - r, y - s * r * 0.2, x - r * 0.35, y - s * r * 0.9, x, y - s * r * 1.9)
        c.close_path()
    P.part(b, WHITE, d=0, line=0.7, a=a)
    P.stroke(lambda c: (c.move_to(x - r * 0.35, y + s * r * 0.05), c.line_to(x - r * 0.2, y + s * r * 0.45)),
             g(0.72), r * 0.2, a)


def star4(P, x, y, r, a=1.0):
    P.flat(lambda c: (c.move_to(x, y - r), c.curve_to(x + r * 0.12, y - r * 0.12, x + r * 0.12, y - r * 0.12, x + r, y),
                      c.curve_to(x + r * 0.12, y + r * 0.12, x + r * 0.12, y + r * 0.12, x, y + r),
                      c.curve_to(x - r * 0.12, y + r * 0.12, x - r * 0.12, y + r * 0.12, x - r, y),
                      c.curve_to(x - r * 0.12, y - r * 0.12, x - r * 0.12, y - r * 0.12, x, y - r), c.close_path()),
           WHITE, a)
    P.stroke(lambda c: (c.move_to(x, y - r), c.curve_to(x + r * 0.12, y - r * 0.12, x + r * 0.12, y - r * 0.12, x + r, y),
                        c.curve_to(x + r * 0.12, y + r * 0.12, x + r * 0.12, y + r * 0.12, x, y + r),
                        c.curve_to(x - r * 0.12, y + r * 0.12, x - r * 0.12, y + r * 0.12, x - r, y),
                        c.curve_to(x - r * 0.12, y - r * 0.12, x - r * 0.12, y - r * 0.12, x, y - r)),
             BLACK, P.lw * 0.5, a)


# ============================================================ Chick-tract face
def _eye_close(P, E, blink, near):
    sp = P.sp
    close = clamp(sp["lid0"] + E["lt"] + blink)
    wink = P.o.get("wink")
    if (not near and E["wn"] > 0.5) or (near and wink == "near") or (not near and wink == "far"):
        close = 1.0
    return close


def face_tract(P, hp, E, blink, look, mouth, eyes="std"):
    sp = P.sp
    R = sp["head_r"]
    lod = P.lod
    lw = P.lw
    lx, ly = look
    esep, ey = sp["eye_sep"], sp["eye_y"]
    skin = "skin"
    poss = E["poss"]
    t = P.t
    for near in (True, False):
        lon = -esep if near else esep
        x, y, z = hp.pt(lon, ey, 1.0)
        if z < 0.1:
            continue
        fore = z ** 0.6
        ew = sp["eye_w"] * R * fore * (0.85 + 0.15 * E["es"])
        eh = sp["eye_h"] * R * E["es"]
        outer = -1 if near else 1
        close = _eye_close(P, E, blink, near)
        if lod == 0:
            continue
        if lod == 1:
            if close > 0.8:
                P.stroke(lambda c: (c.move_to(x - ew, y), c.line_to(x + ew, y)), "lash", eh * 0.4)
            else:
                P.flat(lambda c: p_ellipse(c, x + lx * ew * 0.3, y, ew * 0.55, eh * 0.9 * (1 - close * 0.7)), "lash")
            continue
        # socket shade (halftone in the print)
        if P.ink:
            cw = eh * (0.35 + 0.25 * hq(hp, "brow"))
            P.flat(lambda c: (c.move_to(x - outer * ew * 0.9, y - eh * 1.3),
                              c.curve_to(x - outer * ew * 0.2, y - eh * 1.3 - cw * 1.6, x + outer * ew * 0.7, y - eh * 1.3 - cw,
                                         x + outer * ew * 1.25, y - eh * 0.7),
                              c.curve_to(x + outer * ew * 0.6, y - eh * 1.25, x - outer * ew * 0.2, y - eh * 1.3,
                                         x - outer * ew * 0.9, y - eh * 1.3), c.close_path()), g(0.76), 1.0)
        else:
            P.flat(lambda c: p_ellipse(c, x + outer * ew * 0.1, y - eh * 0.9, ew * 1.3, eh * 1.55),
                   shadow_of(P.pal[skin], 0.86), 0.55)
        tilt = outer * eh * 0.18      # outer corner slightly higher
        xi, xo = x - outer * ew, x + outer * ew
        yi, yo = y + eh * 0.12, y - tilt
        top = y - eh * 1.05
        bot = y + eh * 0.8
        if close > 0.9:
            up = E["sm"] > 0.45 and blink < 0.5 and E["lt"] < 0.9
            yy = y + eh * 0.25
            dip = -eh * 0.55 if up else eh * 0.55
            tapered(P, bez((xi, yi), (x, yy + dip), (xo, yo + eh * 0.2), 9), lw * 0.5, lw * 1.2, "lash", wmid=lw * 1.6)
            if not up:   # lashes of a closed eye
                for k in (0.55, 0.8, 1.0):
                    px = xi + (xo - xi) * k
                    py = yy + dip * (1 - (2 * k - 1) ** 2) * 0.9 + eh * 0.1
                    P.stroke(lambda c: (c.move_to(px, py), c.line_to(px + outer * ew * 0.12, py + eh * 0.35)),
                             "lash", lw * 0.45)
            # crease above
            tapered(P, bez((x - outer * ew * 0.6, top - eh * 0.2), (x, top - eh * 0.75), (xo, yo - eh * 0.6), 7),
                    lw * 0.2, lw * 0.35, "lash", wmid=lw * 0.55)
            continue
        ctx = P.ctx

        def almond(c):
            c.move_to(xi, yi)
            c.curve_to(xi + (xo - xi) * 0.25, top, xi + (xo - xi) * 0.7, top - eh * 0.1, xo, yo)
            c.curve_to(xi + (xo - xi) * 0.7, bot, xi + (xo - xi) * 0.25, bot, xi, yi)
            c.close_path()
        ctx.save()
        ctx.new_path()
        almond(ctx)
        P.src("sclera")
        ctx.fill_preserve()
        ctx.clip()
        ir = eh * 1.08
        ix = x + lx * ew * 0.42
        iy = y + ly * eh * 0.35 + eh * 0.05
        if poss > 0.5:
            P.src("lash")
            p_circle(ctx, ix, iy, ir * 0.17)
            ctx.fill()
        else:
            P.src("iris")
            p_ellipse(ctx, ix, iy, ir * fore ** 0.3, ir)
            ctx.fill()
            P.src("lash")
            ctx.set_line_width(lw * 0.35)
            p_ellipse(ctx, ix, iy, ir * fore ** 0.3, ir)
            ctx.stroke()
            pr = ir * 0.5 * E["ps"]
            if eyes == "slit":
                p_ellipse(ctx, ix, iy, pr * 0.28, ir * 0.92)
            else:
                p_ellipse(ctx, ix, iy, pr * fore ** 0.3, pr)
            ctx.fill()
            P.src("sclera")
            p_circle(ctx, ix - ir * 0.32, iy - ir * 0.34, ir * 0.24)
            ctx.fill()
            if E["tears"] > 0.2:
                p_circle(ctx, ix + ir * 0.35, iy + ir * 0.3, ir * 0.14)
                ctx.fill()
        # lids
        lidy = top + (bot - top) * close
        P.src(skin)
        ctx.rectangle(x - ew * 2, y - eh * 3, ew * 4, lidy - (y - eh * 3))
        ctx.fill()
        lower = clamp(E["lb"] * 0.5 + E["sq"] * 0.4)
        if lower > 0.02:
            by_ = bot - (bot - top) * lower * 0.6
            ctx.rectangle(x - ew * 2, by_, ew * 4, eh * 3)
            ctx.fill()
        ctx.restore()
        # upper lid line: a brush stroke heaviest at the outer corner, with a flick
        lidy_i = yi + (lidy - top) * 0.3 - (yi - top) * (1 - close) * 0.2
        pts = bez((xi, yi), (x - outer * ew * 0.1, lidy - eh * 0.25 * (1 - close)), (xo, yo + (lidy - top) * 0.25), 9)
        tapered(P, pts, lw * 0.45, lw * 1.9, "lash", wmid=lw * 1.35)
        tapered(P, [(xo, yo + (lidy - top) * 0.25), (xo + outer * ew * 0.22, yo - eh * 0.18)], lw * 1.3, lw * 0.2, "lash")
        if P.ch.kind in ("summer", "raven", "seraph") or P.opt("lashes"):
            lashes(P, xo, yo + (lidy - top) * 0.2, outer, ew, eh, close)
        # lower lid (outer two thirds) and crease
        tapered(P, bez((x - outer * ew * 0.35, bot + eh * 0.05), (x + outer * ew * 0.4, bot + eh * 0.12), (xo, yo), 7),
                lw * 0.2, lw * 0.6, "lash", wmid=lw * 0.55)
        tapered(P, bez((x - outer * ew * 0.55, top - eh * 0.45), (x, top - eh * 0.95), (xo, yo - eh * 0.75), 7),
                lw * 0.15, lw * 0.3, "lash", wmid=lw * 0.5)
        if E["tears"] > 0.2:
            P.stroke(lambda c: (c.move_to(x - outer * ew * 0.6, bot - eh * 0.1),
                                c.curve_to(x, bot + eh * 0.12, x + outer * ew * 0.5, bot + eh * 0.05, xo, yo)),
                     WHITE, lw * 0.6)
    _brows_tract(P, hp, E)
    _nose_tract(P, hp)
    if P.opt("freckles"):
        freckles(P, hp)
    if E["fold"] > 0.15 or E["sm"] > 0.55 or clamp(mouth + E["mo"]) > 0.45:
        k = max(E["fold"], (E["sm"] - 0.4) * 1.5, (clamp(mouth + E["mo"]) - 0.35) * 1.2)
        for side in (-1, 1):
            dn = nose_dn(hp)
            a = hp.pt(side * 0.24, -0.22 + dn, 1.03)
            m = hp.pt(side * 0.4, -0.45 + dn, 1.0)
            b = hp.pt(side * (0.44 + 0.08 * E["mw"]), -0.66 + dn - 0.1 * clamp(mouth), 1.0)
            if a[2] > 0.15 and b[2] > 0.05:
                tapered(P, bez(a[:2], m[:2], b[:2], 7), lw * 0.2, lw * 0.15, "lash", wmid=lw * (0.4 + 0.5 * clamp(k)))
    mouth_tract(P, hp, E, mouth)
    if lod >= 2 and hp.egg is not None and P.ink and P.shade_k > 0:
        sd = 1 if P.lv[0] < 0 else -1          # the cheek facing away from the key light
        for k in range(int(3 + 2 * P.shade_k)):
            lat0 = -0.12 - 0.09 * k
            a = hp.pt(sd * 0.72, lat0, 1.01)
            b = hp.pt(sd * 0.9, lat0 - 0.12, 1.01)
            if a[2] > 0.15 and b[2] > 0.05:
                tapered(P, [a[:2], b[:2]], lw * 0.05, lw * 0.05, "lash", wmid=lw * 0.45)
    if lod >= 2 and hp.egg is not None:
        # constructed form lines: cheekbone and jaw corners toward the silhouette, chin shape
        for side in (-1, 1):
            c0 = hp.pt(side * 0.8, -0.05, 1.0)
            c1 = hp.pt(side * 0.95, -0.35, 1.0)
            if 0.12 < c0[2] < 0.8:
                tapered(P, bez(c0[:2], hp.pt(side * 0.92, -0.18, 1.0)[:2], c1[:2], 6), lw * 0.05, lw * 0.05, "lash",
                        wmid=lw * 0.55)
            j0 = hp.pt(side * 0.95, -0.95, 1.0)
            j1 = hp.pt(side * 0.55, -1.2, 1.0)
            if 0.1 < j0[2] < 0.7:
                tapered(P, bez(j0[:2], hp.pt(side * 0.8, -1.12, 1.0)[:2], j1[:2], 6), lw * 0.05, lw * 0.05, "lash",
                        wmid=lw * 0.45)
        ch0 = hp.pt(-0.16, -1.3, 1.0)
        ch1 = hp.pt(0.16, -1.3, 1.0)
        if min(ch0[2], ch1[2]) > 0.1:
            tapered(P, bez(ch0[:2], hp.pt(0.0, -1.38, 1.0)[:2], ch1[:2], 6), lw * 0.05, lw * 0.05, "lash",
                    wmid=lw * 0.5)
    if lod >= 2:
        cl_ = -0.84 if hp.egg is None else -0.98
        c0 = hp.pt(-0.12, cl_, 1.0)
        c1 = hp.pt(0.12, cl_, 1.0)
        cm = hp.pt(0.0, cl_ + 0.04 - 0.05 * clamp(mouth), 1.0)
        if cm[2] > 0.2:
            tapered(P, bez(c0[:2], cm[:2], c1[:2], 5), lw * 0.1, lw * 0.1, "lash", wmid=lw * 0.45)
    ag = {"crow": 0.9, "pharisee": 1.1, "flagmaker": 0.45, "devil": 0.8}.get(P.ch.kind, 0.0)
    if P.opt("age") is not None:
        ag = P.opt("age")
    if ag:
        age_ink(P, hp, ag)
        dn = nose_dn(hp)
        for side in (-1, 1):
            a = hp.pt(side * 0.24, -0.22 + dn, 1.03)
            m_ = hp.pt(side * 0.42, -0.48 + dn, 1.0)
            b = hp.pt(side * 0.46, -0.7 + dn, 1.0)
            if a[2] > 0.15 and b[2] > 0.05:
                tapered(P, bez(a[:2], m_[:2], b[:2], 7), lw * 0.1, lw * 0.1, "lash", wmid=lw * 0.55 * ag)
    if E["tears"] > 0.05:
        tears(P, hp, E)
    if E["sweat"] > 0.05:
        sweat(P, hp, E)


def _brows_tract(P, hp, E):
    sp = P.sp
    R = sp["head_r"]
    lw = P.lw
    if P.lod < 1:
        return
    for near in (True, False):
        lon = -sp["eye_sep"] if near else sp["eye_sep"]
        raise_ = E["by"] + (E["ba"] if not near else -E["ba"] * 0.35)
        lat = sp["eye_y"] + 0.3 + raise_ * 0.13 - E["sq"] * 0.05
        bi = E["bi"]
        p_in = hp.pt(lon * 0.3, lat + bi * 0.12 - (0.03 if bi < 0 else 0), 1.03)
        p_mid = hp.pt(lon * 1.0, lat + 0.08 + raise_ * 0.03 - max(0.0, bi) * 0.02, 1.03)
        p_out = hp.pt(lon * 1.6, lat - 0.06 - bi * 0.05, 1.0)
        if p_mid[2] < 0.12:
            continue
        pts = bez(p_in[:2], p_mid[:2], p_out[:2], 8)
        w0 = R * 0.12 * P.o.get("brow_k", 1.0) * (0.7 + 0.3 * max(p_in[2], 0.2)) * (0.75 + 0.2 * hq(hp, "brow"))
        tapered(P, pts, w0, lw * 0.4, "brow", wmid=w0 * 0.9)
        if E["bi"] > 0.6 and P.lod >= 2:   # worry creases
            q = hp.pt(lon * 0.15, lat + 0.18, 1.02)
            P.stroke(lambda c: (c.move_to(q[0], q[1]), c.line_to(q[0] - lon * R * 0.02, q[1] - R * 0.12)), "lash",
                     lw * 0.4)
        if E["bi"] < -0.6 and P.lod >= 2:  # scowl lines
            q = hp.pt(lon * 0.12, lat - 0.02, 1.02)
            P.stroke(lambda c: (c.move_to(q[0], q[1]), c.line_to(q[0] + (1 if lon > 0 else -1) * R * 0.02, q[1] - R * 0.14)),
                     "lash", lw * 0.45)


def _nose_tract(P, hp):
    sp = P.sp
    R = sp["head_r"]
    lw = P.lw
    if P.lod < 1:
        return
    nk = sp["nose"]
    yaw = hp.yaw
    dn = nose_dn(hp)

    def ridge(lat):
        k = clamp((0.3 - lat) / (0.45 - dn))
        return 1.0 + 0.16 * nk * k * k * (3 - 2 * k)
    top_lat = 0.02 + 0.28 * clamp(abs(math.sin(yaw)) / 0.45)
    pts = []
    for i in range(8):
        lat = top_lat + (-0.2 + dn - top_lat) * i / 7
        q = hp.pt(0.05 if yaw >= 0 else -0.05, lat, ridge(lat))
        if q[2] > -0.1:
            pts.append(q[:2])
    if P.lod >= 2 and len(pts) >= 3:
        tapered(P, pts, lw * 0.15, lw * 0.85, "lash", wmid=lw * 0.55)
    # tip / under-nose shadow on the side away from the light
    if P.lod >= 2:
        tp = hp.pt(0.0, -0.26 + dn, 1.02)
        if tp[2] > 0.2:
            sx = -P.lv[0]
            P.flat(lambda c: p_ellipse(c, tp[0] + sx * R * 0.1, tp[1] + R * 0.06, R * 0.14, R * 0.05), g(0.8), 0.95)
    # underside + nostril wings
    wl = hp.pt(-0.15, -0.21 + dn, 1.04)
    wr = hp.pt(0.15, -0.21 + dn, 1.04)
    un = hp.pt(0.0, -0.25 + dn, 1.0 + 0.1 * nk)
    seg = [q[:2] for q in (wl, un, wr) if q[2] > 0.0]
    if len(seg) >= 2:
        tapered(P, bez(seg[0], un[:2], seg[-1], 7) if len(seg) == 3 else seg, lw * 0.3, lw * 0.3, "lash",
                wmid=lw * 0.95)
    for side in (-1, 1):
        w = hp.pt(side * 0.16, -0.17 + dn, 1.05)
        if w[2] > 0.15 and P.lod >= 2:
            P.stroke(lambda c: (c.move_to(w[0], w[1] - R * 0.05), c.curve_to(w[0] + side * R * 0.035, w[1] - R * 0.01,
                                                                              w[0] + side * R * 0.01, w[1] + R * 0.04,
                                                                              w[0] - side * R * 0.02, w[1] + R * 0.045)),
                     "lash", lw * 0.5)
        n = hp.pt(side * 0.085, -0.235 + dn, 1.07)
        if n[2] > 0.3:
            P.flat(lambda c: p_ellipse(c, n[0], n[1], R * 0.035 * n[2], R * 0.02), "lash")


def mouth_tract(P, hp, E, m):
    sp = P.sp
    R = sp["head_r"]
    lw = P.lw
    lat = -0.52 if hp.egg is None else -0.7
    c = hp.pt(0.0, lat, 1.0)
    if c[2] < 0.1 or P.lod < 1:
        return
    o = clamp(m + E["mo"])
    yell = getattr(P.rig, "yell", 0.0) * clamp(o * 1.6)
    smile, sk = E["sm"], E["sk"]
    vw = 1.0 + 0.14 * noise1(P.t * 8.0 + 3.3, 51) * clamp(m * 3)
    wl = 0.36 * E["mw"] * (1 + 0.15 * smile) * vw * (1 + 0.1 * o + 0.3 * yell)
    c0 = sk * 0.07

    def vis(p):
        if p[2] >= 0.05:
            return p[0], p[1]
        ln = math.hypot(p[0], p[1]) or 1.0
        return p[0] / ln * R * 0.97, p[1]
    L = vis(hp.pt(c0 - wl, lat + smile * 0.12 + sk * 0.07, 1.0))
    Rr = vis(hp.pt(c0 + wl, lat + smile * 0.12 - sk * 0.07, 1.0))
    x, y = c[0], c[1]
    Hm = R * 0.5 * o / vw * (1 + 0.7 * yell)
    lip_dark = P.opt("lipstick")
    if lip_dark and P.lod >= 2:
        up = y - R * 0.1 - Hm * 0.15
        lo = y + Hm + R * 0.14
        P.flat(lambda cc: (cc.move_to(*L), cc.curve_to(x - wl * R * 0.3, up, x - R * 0.05, up - R * 0.03, x, up + R * 0.03),
                           cc.curve_to(x + R * 0.05, up - R * 0.03, x + wl * R * 0.3, up, *Rr),
                           cc.curve_to(x + wl * R * 0.5, lo, x - wl * R * 0.5, lo, *L), cc.close_path()), "lip")
    P._lipk = hq(hp, "lips")
    if o < 0.06 and P.lod >= 2 and not lip_dark:
        lips(P, L, Rr, x, y + smile * R * 0.02, R, smile, False)
    if o < 0.06 or P.lod == 1:
        mid = (x, y + smile * R * 0.04 - (E["sm"] < -0.2) * E["sm"] * R * 0.06 + Hm * 0.4)
        tapered(P, bez(L, (mid[0], mid[1] - smile * R * 0.12), Rr, 9), lw * 0.35, lw * 0.35, "lash",
                wmid=lw * 1.35 + Hm * 0.8)
        if smile > 0.3:   # smile corners
            for q, s in ((L, -1), (Rr, 1)):
                P.stroke(lambda cc: (cc.move_to(q[0] + s * R * 0.03, q[1] - R * 0.05), cc.line_to(q[0] - s * R * 0.01, q[1] + R * 0.04)),
                         "lash", lw * 0.5)
        if P.lod >= 2:  # lower lip
            ly_ = y + R * 0.15 + smile * R * 0.02
            tapered(P, bez((x - wl * R * 0.35, ly_), (x, ly_ + R * 0.04), (x + wl * R * 0.35, ly_), 5), lw * 0.1,
                    lw * 0.1, "lash", wmid=lw * 0.7)
        return
    top_mid = y + smile * R * 0.02 - Hm * 0.12
    bot_mid = y + Hm + smile * R * 0.05

    def b(cc):
        cc.move_to(*L)
        cc.curve_to(L[0] + (x - L[0]) * 0.5, top_mid, Rr[0] + (x - Rr[0]) * 0.5, top_mid, *Rr)
        cc.curve_to(Rr[0] + (x - Rr[0]) * 0.15, bot_mid, L[0] + (x - L[0]) * 0.15, bot_mid, *L)
        cc.close_path()
    ctx = P.ctx
    ctx.save()
    ctx.new_path()
    b(ctx)
    P.src("mouth_in")
    ctx.fill_preserve()
    ctx.clip()
    if P.lod >= 2:
        if E["th"] > 0.15 or o > 0.3:
            th = Hm * 0.3 * (0.55 + E["th"] * 0.6)
            P.src("teeth")
            ctx.rectangle(x - R * 2, top_mid - R, R * 4, R + th)
            ctx.fill()
            P.src("lash")
            ctx.set_line_width(lw * 0.3)
            ctx.move_to(L[0], top_mid + th)
            ctx.line_to(Rr[0], top_mid + th)
            ctx.stroke()
            if E["th"] > 0.6 or yell > 0.5 or E["poss"] > 0.5:
                P.src("teeth")
                ctx.rectangle(x - R * 2, bot_mid - th * 0.8, R * 4, R)
                ctx.fill()
                if E["poss"] > 0.5 or P.ch.kind == "devil":   # the cat grin: rows of pointed teeth
                    P.src("mouth_in")
                    ctx.rectangle(x - R * 2, top_mid - R, R * 4, R * 3)
                    ctx.fill()
                    n_t = 13
                    x0_, x1_ = L[0], Rr[0]
                    tw = (x1_ - x0_) / n_t
                    P.src("teeth")
                    for k in range(n_t):
                        tx = x0_ + tw * (k + 0.5)
                        ln_ = Hm * (0.55 if 3 <= k <= n_t - 4 else 0.4)
                        ctx.move_to(tx - tw * 0.5, top_mid - R * 0.1)
                        ctx.line_to(tx + tw * 0.5, top_mid - R * 0.1)
                        ctx.line_to(tx, top_mid + ln_ * 0.9)
                        ctx.close_path()
                        ctx.move_to(tx - tw * 0.5 + tw * 0.5, bot_mid + R * 0.1)
                        ctx.line_to(tx + tw, bot_mid + R * 0.1)
                        ctx.line_to(tx + tw * 0.5, bot_mid - ln_ * 0.8)
                        ctx.close_path()
                    ctx.fill_preserve()
                    P.src("lash")
                    ctx.set_line_width(lw * 0.3)
                    ctx.stroke()
        if o > 0.28:
            P.src("tongue")
            p_ellipse(ctx, x + R * 0.02, bot_mid + Hm * 0.02, abs(Rr[0] - L[0]) * 0.3, Hm * 0.35)
            ctx.fill()
    ctx.restore()
    ctx.new_path()
    b(ctx)
    P.src("lash")
    ctx.set_line_width(lw * 0.9)
    ctx.stroke()
    tapered(P, bez((L[0] + (x - L[0]) * 0.35, bot_mid + R * 0.12), (x, bot_mid + R * 0.17),
                   (Rr[0] + (x - Rr[0]) * 0.35, bot_mid + R * 0.12), 5), lw * 0.1, lw * 0.1, "lash", wmid=lw * 0.8)


def tears(P, hp, E):
    sp = P.sp
    R = sp["head_r"]
    t = P.t
    for near in (True, False):
        lon = -sp["eye_sep"] if near else sp["eye_sep"]
        x, y, z = hp.pt(lon * 1.05, sp["eye_y"] - 0.15, 1.0)
        if z < 0.3:
            continue
        x2, y2, z2 = hp.pt(lon * 1.15, -0.75, 1.0)
        # wet streak down the cheek
        P.stroke(lambda c: (c.move_to(x, y), c.curve_to(x + (x2 - x) * 0.2, y + (y2 - y) * 0.4, x2, y2 - R * 0.3, x2, y2)),
                 "lash", P.lw * 1.2, 0.9 * E["tears"])
        P.stroke(lambda c: (c.move_to(x, y), c.curve_to(x + (x2 - x) * 0.2, y + (y2 - y) * 0.4, x2, y2 - R * 0.3, x2, y2)),
                 WHITE, P.lw * 0.55, E["tears"])
        for k in range(2):
            ph = (t * 0.55 + k * 0.5 + (0.3 if near else 0.0)) % 1.0
            px = x + (x2 - x) * ph
            py = y + (y2 - y) * ph + R * 0.25 * ph * ph
            drop(P, px, py, R * 0.06 * (0.8 + 0.4 * ph), a=E["tears"] * clamp(ph * 5) * clamp((1 - ph) * 4))


def sweat(P, hp, E, flying=False):
    R = P.sp["head_r"]
    t = P.t
    for k, (lon, lat, s) in enumerate(((0.95, 0.45, 1.0), (1.2, 0.1, 0.8), (-0.45, 0.72, 0.7))):
        x, y, z = hp.pt(lon, lat - 0.04 * ((t * 0.7 + k * 0.33) % 1.0), 1.06)
        if z < 0.05:
            continue
        drop(P, x, y, R * 0.075 * s, a=E["sweat"])
    if flying:
        # drops flung off the head (Chick-tract panic)
        for k in range(4):
            ph = (t * 1.3 + k * 0.27) % 1.0
            a = -2.3 + k * 0.55
            r = R * (1.15 + 0.5 * ph)
            x, y = math.cos(a) * r, math.sin(a) * r
            drop(P, x, y, R * (0.08 - 0.02 * ph), a=E["sweat"] * clamp((1 - ph) * 3))


def shock_lines(P, hp, k=1.0):
    """shock emanata: short radiating strokes around the head"""
    R = P.sp["head_r"]
    t = P.t
    jit = 0.06 * math.sin(t * 23.0)
    for i in range(9):
        a = -math.pi * 0.95 + math.pi * 0.9 * i / 8 + jit
        r0, r1 = R * (1.35 + 0.05 * (i % 2)), R * (1.75 + 0.1 * (i % 3))
        tapered(P, [(math.cos(a) * r0, math.sin(a) * r0), (math.cos(a) * r1, math.sin(a) * r1)], P.lw * 1.4 * k,
                P.lw * 0.2, "lash")


def aura(P, hp, k=1.0):
    """beatific glow: fine radiating light lines behind the head (drawn first)"""
    R = P.sp["head_r"]
    for i in range(28):
        a = TAU * i / 28
        r0, r1 = R * 1.25, R * (1.9 + 0.25 * (i % 2))
        tapered(P, [(math.cos(a) * r0, math.sin(a) * r0 - R * 0.1), (math.cos(a) * r1, math.sin(a) * r1 - R * 0.1)],
                P.lw * 0.5 * k, P.lw * 0.05, "lash")


def freckles(P, hp):
    if P.lod < 2:
        return
    R = P.sp["head_r"]
    seed = P.ch.seed
    for i in range(14):
        side = -1 if i % 2 else 1
        lon = side * (0.14 + 0.4 * hash01(i, 91 + seed))
        lat = -0.04 - 0.22 * hash01(i, 92 + seed)
        x, y, z = hp.pt(lon, lat, 1.01)
        if z > 0.25:
            P.flat(lambda c: p_circle(c, x, y, R * (0.005 + 0.004 * hash01(i, 93))), "lash", 0.95)


# ============================================================ anime face (Summer's one-beat manga gag)
def face_anime(P, hp, E, blink, look, mouth):
    sp = P.sp
    R = sp["head_r"]
    lw = P.lw
    lx, ly = look
    t = P.t
    for near in (True, False):
        lon = -0.42 if near else 0.42
        x, y, z = hp.pt(lon, -0.06, 1.0)
        if z < 0.12:
            continue
        fore = z ** 0.7
        ew = R * 0.3 * fore
        eh = R * 0.42 * (0.9 + 0.1 * E["es"])
        outer = -1 if near else 1
        close = clamp(E["lt"] * 0.6 + blink)
        if close > 0.85:
            up = E["sm"] > 0.3
            tapered(P, bez((x - ew, y), (x, y - eh * (0.7 if up else -0.3)), (x + ew, y), 9), lw, lw, "lash",
                    wmid=lw * 2.2)
            continue
        ctx = P.ctx
        top = y - eh
        ctx.save()
        ctx.new_path()
        p_ellipse(ctx, x, y, ew, eh)
        P.src("sclera")
        ctx.fill_preserve()
        ctx.clip()
        ix, iy = x + lx * ew * 0.25, y + ly * eh * 0.15 + eh * 0.08
        irx, iry = ew * 0.78, eh * 0.86
        P.src(g(0.22))
        p_ellipse(ctx, ix, iy, irx, iry)
        ctx.fill()
        P.src(g(0.55))
        p_ellipse(ctx, ix, iy + iry * 0.4, irx * 0.8, iry * 0.5)
        ctx.fill()
        P.src("lash")
        p_ellipse(ctx, ix, iy - iry * 0.05, irx * 0.42, iry * 0.46)
        ctx.fill()
        P.src("sclera")
        p_ellipse(ctx, ix - irx * 0.38 * outer * -1, iy - iry * 0.42, irx * 0.34, iry * 0.26)
        ctx.fill()
        p_circle(ctx, ix + irx * 0.35 * outer * -1, iy + iry * 0.45, irx * 0.15)
        ctx.fill()
        # sparkle glints
        p_circle(ctx, ix + irx * 0.1, iy - iry * 0.62, irx * 0.08)
        ctx.fill()
        P.src("skin")
        ctx.rectangle(x - ew * 2, y - eh * 3, ew * 4, (top + 2 * eh * close * 0.8) - (y - eh * 3))
        ctx.fill()
        ctx.restore()
        lt_ = top + 2 * eh * close * 0.8
        tapered(P, bez((x - outer * ew * 1.05, y - eh * 0.25), (x, lt_ - eh * 0.25), (x + outer * ew * 1.12, y - eh * 0.45), 9),
                lw * 0.8, lw * 2.6, "lash", wmid=lw * 2.4)
        for k in range(2):   # lash flicks
            bx = x + outer * ew * (1.0 + 0.05 * k)
            by = y - eh * (0.45 - 0.22 * k)
            tapered(P, [(bx, by), (bx + outer * ew * 0.35, by - eh * (0.2 - 0.15 * k))], lw * 1.4, lw * 0.2, "lash")
        tapered(P, bez((x + outer * ew * 0.2, y + eh * 0.98), (x + outer * ew * 0.7, y + eh * 0.85), (x + outer * ew * 0.98, y + eh * 0.4), 5),
                lw * 0.2, lw * 0.9, "lash")
        # thin high brows
        b0 = hp.pt(lon * 0.45, 0.52 + E["by"] * 0.08, 1.02)
        b1 = hp.pt(lon * 1.3, 0.5 + E["by"] * 0.05 - E["bi"] * 0.05, 1.0)
        bm = hp.pt(lon * 0.9, 0.58 + E["by"] * 0.08, 1.02)
        if bm[2] > 0.1:
            tapered(P, bez(b0[:2], bm[:2], b1[:2], 6), lw * 0.9, lw * 0.2, "brow")
    # blush hatches
    for side in (-1, 1):
        cx, cy, cz = hp.pt(side * 0.62, -0.3, 1.0)
        if cz < 0.2:
            continue
        P.flat(lambda c: p_ellipse(c, cx, cy, R * 0.2 * cz, R * 0.09), g(0.8), 0.8)
        for k in range(4):
            px = cx + (k - 1.5) * R * 0.07 * cz
            P.stroke(lambda c: (c.move_to(px + R * 0.03, cy - R * 0.06), c.line_to(px - R * 0.03, cy + R * 0.06)), "lash",
                     lw * 0.45)
    n = hp.pt(0.0, -0.22, 1.02)
    if n[2] > 0.2:
        P.stroke(lambda c: (c.move_to(n[0], n[1] - R * 0.04), c.line_to(n[0] - R * 0.025, n[1] + R * 0.03)), "lash",
                 lw * 0.5)
    mx, my, mz = hp.pt(0.0, -0.5, 1.0)
    if mz > 0.15:
        o = clamp(mouth + E["mo"])
        if o < 0.08:
            wav = E["sweat"] > 0.3 or E["by"] > 0.5
            if wav:
                pts = [(mx - R * 0.12 + i * R * 0.04, my + (R * 0.02 if i % 2 else -R * 0.02)) for i in range(7)]
                P.stroke(lambda c: (c.move_to(*pts[0]), [c.line_to(*q) for q in pts[1:]]), "lash", lw * 0.6)
            else:
                P.stroke(lambda c: (c.move_to(mx - R * 0.08, my), c.curve_to(mx - R * 0.03, my + R * 0.05 * E["sm"],
                                                                              mx + R * 0.03, my + R * 0.05 * E["sm"],
                                                                              mx + R * 0.08, my)), "lash", lw * 0.7)
        else:
            hh = R * 0.2 * o
            P.part(lambda c: p_ellipse(c, mx, my + hh * 0.3, R * 0.085, hh), "mouth_in", d=0, line=0.5)
            P.flat(lambda c: p_ellipse(c, mx, my + hh * 0.8, R * 0.05, hh * 0.35), "tongue")
    # the big sweat drop
    sx, sy, sz = hp.pt(1.05, 0.42, 1.08)
    if sz > -0.2:
        drop(P, sx + R * 0.12, sy + R * 0.05 * math.sin(t * 3), R * 0.17)
    for k, (dx, dy, s) in enumerate(((-1.25, -0.9, 0.16), (1.3, -1.05, 0.12), (-1.45, 0.05, 0.1))):
        tw = 0.6 + 0.4 * math.sin(t * 6 + k * 2)
        star4(P, dx * R, dy * R, R * s * tw)


# ============================================================ hair (film 2 styles)
def strands(P, path, hp, style, colour="hair"):
    """ink strand lines (light hair) or white gloss streaks (black hair) clipped to the hair shape"""
    if P.lod < 2:
        return
    R = P.sp["head_r"]
    tone = lum(P.C(colour))
    if style == "pigtails":
        return strands_summer(P, hp, path)
    if tone < 0.25:
        return shine_black(P, hp, path)
    ctx = P.ctx
    ctx.save()
    ctx.new_path()
    ctx.append_path(path)
    ctx.clip()
    seed = P.ch.seed
    if tone < 0.25:
        for k in range(5):
            lon0 = -1.1 + k * 0.42 + 0.1 * hash01(k, seed + 7)
            pts = [hp.pt(lon0 + 0.12 * math.sin(i * 0.6), 1.25 - i * 0.2, 1.06)[:2] for i in range(7)]
            tapered(P, pts, P.lw * 0.2, P.lw * 0.2, WHITE, wmid=R * (0.05 + 0.04 * hash01(k, seed)))
    else:
        n = 16 if style in ("pigtails", "long", "long_glossy") else 10
        for k in range(n):
            lon0 = -1.6 + 3.2 * k / (n - 1) + 0.08 * hash01(k, seed + 3)
            pts = [hp.pt(lon0 * (0.2 + 0.8 * i / 6) + 0.05 * math.sin(i + k), 1.45 - i * 0.26, 1.08)[:2] for i in range(7)]
            tapered(P, pts, P.lw * 0.1, P.lw * 0.1, "lash", wmid=P.lw * (0.35 + 0.25 * hash01(k, seed + 5)))
    ctx.restore()


def pigtail(P, hp, near, colour="hair"):
    R = P.sp["head_r"]
    side = -1 if near else 1
    x, y, z = hp.pt(side * 1.9, -0.42, 1.0)
    if near and z < -0.45:
        return
    t = P.t
    sw = 0.12 * math.sin(t * 1.9 + (0 if near else 1.3)) + P.o["wind_x"] * 0.25
    ox = (x / (abs(x) + 1e-6)) if abs(x) > R * 0.2 else side * 0.4
    L = R * 1.9
    ang = math.pi / 2 - ox * 0.18 - sw
    ca, sa = math.cos(ang), math.sin(ang)
    nx, ny = -sa, ca
    tipx, tipy = x + ca * L, y + sa * L
    w = R * 0.26
    pts = [(x + nx * w * 0.35, y + ny * w * 0.35), (x + ca * L * 0.35 + nx * w, y + sa * L * 0.35 + ny * w),
           (x + ca * L * 0.75 + nx * w * 0.7, y + sa * L * 0.75 + ny * w * 0.7), (tipx, tipy),
           (x + ca * L * 0.8 - nx * w * 0.55, y + sa * L * 0.8 - ny * w * 0.55),
           (x + ca * L * 0.35 - nx * w * 0.95, y + sa * L * 0.35 - ny * w * 0.95), (x - nx * w * 0.35, y - ny * w * 0.35)]
    path = P.part(lambda c: p_smooth(c, pts, True, 0.4), colour, d=R * 0.15, hi=0.5)
    for k, (da, dl) in enumerate(((-0.28, 0.78), (0.3, 0.7), (0.05, 1.05))):
        a2 = ang + da * 0.5
        P.part(lambda c: p_leaf(c, x + ca * L * 0.25, y + sa * L * 0.25, L * dl, w * 0.55, a2), colour, d=0, line=0.55)
    if P.ink and P.lod >= 2:
        P.ctx.save()
        P.ctx.new_path()
        P.ctx.append_path(path)
        P.ctx.clip()
        for k in range(6):
            f = (k - 2.5) / 3.0
            tapered(P, bez((x + nx * w * 0.2 * f, y + ny * w * 0.2 * f), (x + ca * L * 0.45 + nx * w * 0.9 * f, y + sa * L * 0.45 + ny * w * 0.9 * f),
                           (tipx - nx * w * 0.05 * f, tipy), 7), P.lw * 0.1, P.lw * 0.1, "lash", wmid=P.lw * 0.4)
        P.ctx.restore()
    bx_, by_ = x + ca * R * 0.08, y + sa * R * 0.08
    P.part(lambda c: (c.new_sub_path(), c.save(), c.translate(bx_, by_), c.rotate(ang), c.rectangle(-R * 0.07, -w * 0.55, R * 0.14,
                                                                                                    w * 1.1), c.restore()),
           g(0.05) if P.ink else "brow", d=0, line=0.6)


def long_glossy_back(P, hp, colour="hair"):
    R = P.sp["head_r"]
    t = P.t
    sw = P.o["wind_x"] * 1.1 + 0.08 * math.sin(t * 1.3 + P.ch.seed)
    s = math.sin(hp.yaw)
    back = -s * R * 0.7
    pts = [(-R * 1.1 + back * 0.4, -R * 0.5), (-R * 1.25 + back, R * 1.2), (-R * 1.35 + back + sw * R * 0.4, R * 2.6),
           (-R * 0.8 + back + sw * R * 0.6, R * 3.4), (R * 0.1 + back * 0.5 + sw * R * 0.6, R * 3.1),
           (R * 0.95 + sw * R * 0.4, R * 3.3), (R * 1.25, R * 2.0), (R * 1.15, R * 0.4), (R * 0.9, -R * 0.8),
           (0.0, -R * 1.12)]
    light = lum(P.C(colour)) > 0.5
    path = P.part(lambda c: p_smooth(c, pts, True, 0.38), colour, d=R * 0.2, line=0.45 if light else 1.0)
    if P.lod >= 2:
        P.ctx.save()
        P.ctx.new_path()
        P.ctx.append_path(path)
        P.ctx.clip()
        for k in range(4):
            x0 = -R * 0.9 + k * R * 0.5 + back * 0.5
            tapered(P, bez((x0, R * 0.2), (x0 - R * 0.35 + sw * R * 0.3, R * 1.4), (x0 + R * 0.1 + sw * R * 0.5, R * 2.7), 8),
                    P.lw * 0.2, P.lw * 0.2, WHITE if lum(P.C(colour)) < 0.3 else "lash", wmid=R * 0.07)
        P.ctx.restore()


def long_glossy_front(P, hp, colour="hair"):
    """the curtain of hair falling in front of the near shoulder"""
    R = P.sp["head_r"]
    x, y, z = hp.pt(-1.2, 0.3, 1.08)
    if z < -0.15:
        return
    sw = P.o["wind_x"] * 0.8
    pts = [(x + R * 0.15, y - R * 0.3), (x - R * 0.28, y + R * 0.6), (x - R * 0.35 + sw * R * 0.3, y + R * 1.8),
           (x - R * 0.1 + sw * R * 0.4, y + R * 2.9), (x + R * 0.25 + sw * R * 0.4, y + R * 2.2),
           (x + R * 0.32, y + R * 1.0), (x + R * 0.4, y + R * 0.2)]
    path = P.part(lambda c: p_smooth(c, pts, True, 0.4), colour, d=R * 0.12)
    if P.lod >= 2:
        P.ctx.save()
        P.ctx.new_path()
        P.ctx.append_path(path)
        P.ctx.clip()
        tapered(P, bez((x + R * 0.05, y), (x - R * 0.15, y + R * 1.2), (x + R * 0.05 + sw * R * 0.3, y + R * 2.4), 8),
                P.lw * 0.2, P.lw * 0.2, WHITE, wmid=R * 0.07)
        P.ctx.restore()


def dreads(P, hp, colour="hair", front=False):
    R = P.sp["head_r"]
    t = P.t
    seed = P.ch.seed
    sw = P.o["wind_x"] * 0.8
    for k in range(11):
        lon = math.pi - 1.9 + 3.8 * k / 10
        x, y, z = hp.pt(lon, 0.05 + 0.15 * math.cos(k), 1.02)
        if front != (z > 0.0):
            continue
        L = R * (1.5 + 0.9 * hash01(k, seed + 11))
        a = math.pi / 2 + (0.35 if x > 0 else -0.35) * abs(math.sin(lon)) + 0.12 * math.sin(t * 1.7 + k) - sw * 0.3
        p1 = (x + math.cos(a) * L * 0.5, y + math.sin(a) * L * 0.5)
        p2 = (x + math.cos(a + 0.12) * L, y + math.sin(a + 0.12) * L)
        rr = R * 0.14
        P.part(lambda c: p_chain(c, [(x, y), p1, p2], [rr, rr * 0.95, rr * 0.75]), colour, d=rr * 0.5)
        if P.ink and P.lod >= 2:
            for j in range(1, 5):
                f = j / 5.0
                q = (x + (p2[0] - x) * f, y + (p2[1] - y) * f)
                P.stroke(lambda c: (c.move_to(q[0] - rr * 0.6, q[1] - rr * 0.2), c.line_to(q[0] + rr * 0.6, q[1] + rr * 0.2)),
                         "lash", P.lw * 0.35)


# ============================================================ head accessories
def trucker_cap(P, hp):
    R = P.sp["head_r"]
    seed = P.ch.seed
    crown = hp.cap(math.pi, 1.2, 1.3, 1.12, 1.1, 48)
    P.part(lambda c: p_poly(c, crown), "cap", d=R * 0.2, hi=0.4)
    # foam front panel
    front = []
    for i in range(9):
        front.append(hp.pt(-0.95 + 1.9 * i / 8, 0.58, 1.12))
    for i in range(9):
        lon = 0.8 - 1.6 * i / 8
        front.append(hp.pt(lon, 1.25 - 0.12 * abs(lon), 1.12))
    if min(q[2] for q in front) > -0.2:
        fp = [q[:2] for q in front]
        P.part(lambda c: p_smooth(c, fp, True, 0.2), "capfront", d=R * 0.1)
    # pins and patches (all original: plain round badges and blank patches)
    for i in range(16):
        lon = -1.2 + 2.4 * hash01(i, seed + 21)
        lat = 0.7 + 0.6 * hash01(i, seed + 22)
        x, y, z = hp.pt(lon, lat, 1.13)
        if z < 0.15:
            continue
        r = R * (0.07 + 0.07 * hash01(i, seed + 23))
        kind = int(hash01(i, seed + 24) * 4)
        tone = [0.95, 0.5, 0.1, 0.75][kind]
        if kind == 3:
            P.part(lambda c: (c.new_sub_path(), c.rectangle(x - r * 1.2 * z, y - r * 0.8, r * 2.4 * z, r * 1.6)), g(tone),
                   d=r * 0.2, line=0.5)
            P.stroke(lambda c: (c.new_sub_path(), c.rectangle(x - r * 0.9 * z, y - r * 0.5, r * 1.8 * z, r)), g(0.3),
                     P.lw * 0.25)
        else:
            P.part(lambda c: p_ellipse(c, x, y, r * z ** 0.6, r), g(tone), d=r * 0.35, line=0.6)
            P.flat(lambda c: p_circle(c, x - r * 0.3, y - r * 0.35, r * 0.25), WHITE, 0.9)
            if kind == 1:
                P.flat(lambda c: p_ellipse(c, x, y, r * 0.45 * z ** 0.6, r * 0.45), g(0.15))
    # the brim
    inner, outer = [], []
    for i in range(15):
        lon = -1.25 + 2.5 * i / 14
        inner.append(hp.pt(lon, 0.58, 1.1))
        cl = math.cos(lon)
        v = (math.sin(lon) * 0.94, math.sin(0.58) * 1.06 - 0.03 * cl, math.cos(lon) * 0.94 + 0.6 * cl * cl)
        outer.append(hp.vec(v, 1.0))
    pts = [q[:2] for q in inner] + [q[:2] for q in outer[::-1]]
    # shadow the bill casts on the brow
    sh = [hp.pt(lon, 0.42, 1.0) for lon in [(-1.0 + 2.0 * i / 10) for i in range(11)]]
    shv = [q[:2] for q in sh if q[2] > 0.1]
    if len(shv) > 2:
        tapered(P, shv, R * 0.02, R * 0.02, g(0.75) if P.ink else (0.4, 0.4, 0.45), wmid=R * 0.16, a=0.8)
    P.part(lambda c: p_smooth(c, pts, True, 0.15), g(0.62) if P.ink else "cap", d=R * 0.06)
    # squatchee
    top = hp.pt(0.0, math.pi / 2 - 0.35, 1.1)
    if top[2] > -0.5:
        P.part(lambda c: p_circle(c, top[0], top[1], R * 0.07), "cap", d=0, line=0.5)


def fez(P, hp):
    R = P.sp["head_r"]
    bot = [hp.vec((math.sin(a) * 0.72, 0.72, math.cos(a) * 0.72), 1.0) for a in [TAU * i / 24 for i in range(24)]]
    top = [hp.vec((math.sin(a) * 0.55, 1.45, math.cos(a) * 0.55), 1.0) for a in [TAU * i / 24 for i in range(24)]]
    allp = [q[:2] for q in bot + top]
    hull = _hull(allp)
    P.part(lambda c: p_poly(c, hull), g(0.25) if P.ink else (0.45, 0.12, 0.12), d=R * 0.2, hi=0.4)
    tp = [q[:2] for q in top]
    P.part(lambda c: p_poly(c, tp), g(0.35) if P.ink else (0.52, 0.16, 0.16), d=0, line=0.6)
    c0 = hp.vec((0.0, 1.45, 0.0), 1.0)
    sw = 0.2 * math.sin(P.t * 2.0)
    P.stroke(lambda c: (c.move_to(c0[0], c0[1]), c.curve_to(c0[0] + R * 0.3, c0[1], c0[0] + R * 0.55, c0[1] + R * 0.3,
                                                             c0[0] + R * (0.6 + sw), c0[1] + R * 0.8)), "lash", P.lw * 0.8)
    P.part(lambda c: p_ellipse(c, c0[0] + R * (0.6 + sw), c0[1] + R * 0.95, R * 0.08, R * 0.2), "lash", d=0)


def _hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def aviators(P, hp):
    sp = P.sp
    R = sp["head_r"]
    lw = P.lw
    cen = []
    for near in (True, False):
        lon = -sp["eye_sep"] * 1.02 if near else sp["eye_sep"] * 1.02
        x, y, z = hp.pt(lon, sp["eye_y"] - 0.04, 1.08)
        cen.append((x, y, z, near))
        if z < 0.05:
            continue
        f = z ** 0.6
        w, hh = R * 0.31 * f, R * 0.25
        outer = -1 if near else 1
        xi, xo = x - outer * w, x + outer * w

        def lens(c):
            c.move_to(xi, y - hh * 0.75)
            c.curve_to(x, y - hh * 0.95, xo, y - hh * 1.0, xo, y - hh * 0.55)
            c.curve_to(xo + outer * w * 0.1, y + hh * 0.5, x + outer * w * 0.3, y + hh * 1.05, x - outer * w * 0.2, y + hh * 0.85)
            c.curve_to(xi - outer * w * 0.15, y + hh * 0.6, xi - outer * w * 0.1, y - hh * 0.4, xi, y - hh * 0.75)
            c.close_path()
        P.flat(lens, "lens", 0.72)
        P.stroke(lens, "lash", lw * 0.8)
        P.stroke(lambda c: (c.move_to(x - outer * w * 0.5, y - hh * 0.5), c.line_to(x - outer * w * 0.05, y - hh * 0.05)),
                 WHITE, lw * 0.9, 0.95)
        P.stroke(lambda c: (c.move_to(x - outer * w * 0.1, y - hh * 0.55), c.line_to(x + outer * w * 0.2, y - hh * 0.25)),
                 WHITE, lw * 0.5, 0.9)
        e = hp.pt(-math.pi / 2 if near else math.pi / 2, 0.02, 1.02)
        if e[2] > -0.6 and abs(e[0] - xo) > R * 0.1:
            P.stroke(lambda c: (c.move_to(xo, y - hh * 0.55), c.line_to(e[0], e[1])), "lash", lw * 0.6)
    vis = [c for c in cen if c[2] > 0.05]
    if len(vis) == 2:
        a, b = vis
        m = hp.pt(0.0, sp["eye_y"] + 0.02, 1.12)
        for dy in (-R * 0.17, -R * 0.1):
            P.stroke(lambda c: (c.move_to(a[0], a[1] + dy), c.curve_to(m[0], m[1] + dy - R * 0.03, m[0], m[1] + dy - R * 0.03,
                                                                       b[0], b[1] + dy)), "lash", lw * 0.5)


def goatee(P, hp, pointed=False, colour="beardc"):
    R = P.sp["head_r"]
    lw = P.lw
    seed = P.ch.seed
    chin = hp.cap(0.0, -1.12, 0.3, 1.04, 1.0, 26)
    pts = list(chin)
    if pointed:
        c = hp.pt(0.0, -1.3 if hp.egg is not None else -1.05, 1.07)
        if c[2] > -0.2:
            sw = P.o["wind_x"] * 0.6
            pts = [(c[0] - R * 0.28, c[1] - R * 0.15), (c[0] - R * 0.1, c[1] + R * 0.6),
                   (c[0] + sw * R * 0.3, c[1] + R * 1.25), (c[0] + R * 0.12, c[1] + R * 0.55), (c[0] + R * 0.3, c[1] - R * 0.15)]
    if len(pts) >= 3:
        path = P.part(lambda c: p_smooth(c, pts, True, 0.3), colour, d=R * 0.12)
        if P.ink and P.lod >= 2 and not pointed:
            P.ctx.save()
            P.ctx.new_path()
            P.ctx.append_path(path)
            P.ctx.clip()
            for i in range(18):
                q = hp.pt(-0.28 + 0.56 * hash01(i, seed + 31), -0.95 - 0.3 * hash01(i, seed + 32), 1.05)
                P.stroke(lambda c: (c.move_to(q[0], q[1]), c.line_to(q[0] + R * 0.02, q[1] + R * 0.09)),
                         WHITE if i % 3 == 0 else "lash", lw * 0.35)
            P.ctx.restore()
    base = hp.pt(0.0, -0.56, 1.05)
    if base[2] > 0.05:
        for side in (-1, 1):
            tip = hp.pt(side * 0.36, -0.95, 1.03)
            if tip[2] < 0:
                continue
            mid = hp.pt(side * 0.27, -0.6, 1.05)
            poly = [base[:2], mid[:2], tip[:2], (mid[0] - side * R * 0.02, mid[1] + R * 0.07),
                    (base[0], base[1] + R * 0.06)]
            P.part(lambda c: p_smooth(c, poly, True, 0.3), colour, d=R * 0.05, line=0.8)


def face_mask(P, hp, amt):
    """surgical mask: amt 1 = over nose+mouth, 0.4 = pulled down under the chin, 0 = none"""
    if amt <= 0.02:
        return
    R = P.sp["head_r"]
    lw = P.lw
    f = clamp((amt - 0.4) / 0.6)
    lat_top = -0.95 + 0.85 * f
    lat_bot = -1.28 + 0.36 * f
    top, bot = [], []
    n = 13
    for i in range(n):
        lon = -1.3 + 2.6 * i / (n - 1)
        a = hp.pt(lon, lat_top - 0.06 * abs(lon), 1.03)
        b = hp.pt(lon * 0.95, lat_bot + 0.12 * abs(lon), 1.04 - 0.06 * (1 - f))
        top.append(a)
        bot.append(b)
    vis = [i for i in range(n) if top[i][2] > -0.05 or bot[i][2] > -0.05]
    if len(vis) < 2:
        return

    def pin(q):
        if q[2] >= 0:
            return q[:2]
        ln = math.hypot(q[0], q[1]) or 1.0
        return (q[0] / ln * R * 1.02, q[1] / ln * R * 1.02)
    i0, i1 = vis[0], vis[-1]
    pts = [pin(top[i]) for i in range(i0, i1 + 1)] + [pin(bot[i]) for i in range(i1, i0 - 1, -1)]
    path = P.part(lambda c: p_smooth(c, pts, True, 0.2), "mask", d=R * 0.12)
    ctx = P.ctx
    ctx.save()
    ctx.new_path()
    ctx.append_path(path)
    ctx.clip()
    m = getattr(P.rig, "mouth", 0.0)
    for k in range(3):
        fr = 0.28 + 0.24 * k
        row = [hp.pt(-1.1 + 2.2 * i / 10, lat_top + (lat_bot - lat_top) * fr - 0.02 * m * (k == 1), 1.045) for i in range(11)]
        row = [q[:2] for q in row if q[2] > 0.0]
        if len(row) >= 2:
            P.stroke(lambda c: (c.move_to(*row[0]), [c.line_to(*q) for q in row[1:]]), g(0.55), lw * 0.4)
    ctx.restore()
    for side in (-1, 1):
        e = hp.pt(side * math.pi / 2, 0.0, 1.0)
        q = top[0 if side < 0 else -1]
        b = bot[0 if side < 0 else -1]
        if e[2] > -0.5 and q[2] > -0.6:
            P.stroke(lambda c: (c.move_to(*pin(q)), c.line_to(e[0], e[1] - R * 0.1)), "lash", lw * 0.4)
            P.stroke(lambda c: (c.move_to(*pin(b)), c.line_to(e[0], e[1] + R * 0.08)), "lash", lw * 0.4)


def hoop(P, hp, near):
    R = P.sp["head_r"]
    x, y, z = hp.pt((-math.pi / 2 + 0.1) if near else (math.pi / 2 - 0.1), -0.42, 1.0)
    if z < -0.35:
        return
    r = R * 0.3
    cx, cy = x, y + r
    sw = 0.06 * math.sin(P.t * 2.3 + (0 if near else 1))
    P.ctx.save()
    P.ctx.translate(x, y)
    P.ctx.rotate(sw)
    P.ctx.translate(-x, -y)
    P.stroke(lambda c: p_ellipse(c, cx, cy, r * max(0.25, abs(z) ** 0.5 if not near else 0.5 + 0.5 * abs(z)), r), "lash",
             P.lw * 1.9)
    P.stroke(lambda c: (c.new_sub_path(), c.save(), c.translate(cx, cy), c.scale(r * 0.8, r), c.arc(0, 0, 1, 3.6, 5.4),
                        c.restore()), WHITE, P.lw * 0.5)
    P.ctx.restore()


def goggles_head(P, hp):
    R = P.sp["head_r"]
    band = [hp.pt(a, 0.62, 1.07) for a in [(-1.6 + 3.2 * i / 16) for i in range(17)]]
    vis = [q[:2] for q in band if q[2] > -0.1]
    if len(vis) > 1:
        tapered(P, vis, R * 0.12, R * 0.12, "goggle")
    for side in (-1, 1):
        x, y, z = hp.pt(side * 0.36, 0.64, 1.12)
        if z < 0.1:
            continue
        P.part(lambda c: p_ellipse(c, x, y, R * 0.22 * z ** 0.5, R * 0.2), "goggle", d=R * 0.05)
        P.part(lambda c: p_ellipse(c, x, y, R * 0.15 * z ** 0.5, R * 0.14), "lens", d=0, line=0.5)
        P.flat(lambda c: p_ellipse(c, x - R * 0.05, y - R * 0.05, R * 0.05, R * 0.035), WHITE)


def face_paint(P, hp, kind):
    R = P.sp["head_r"]
    if kind == "stripes":
        for side in (-1, 1):
            for k in range(2):
                pts = [hp.pt(side * (0.45 + 0.4 * i / 4), -0.22 - 0.1 * k, 1.01) for i in range(5)]
                pv = [q[:2] for q in pts if q[2] > 0.15]
                if len(pv) >= 2:
                    tapered(P, pv, R * 0.05, R * 0.02, "paint")
                    tapered(P, pv, P.lw * 0.3, P.lw * 0.1, "lash")
    elif kind == "dots":
        for side in (-1, 1):
            for k in range(3):
                q = hp.pt(side * (0.3 + 0.12 * k), -0.12 - 0.06 * (k % 2), 1.01)
                if q[2] > 0.2:
                    P.part(lambda c: p_circle(c, q[0], q[1], R * 0.035), "paint", d=0, line=0.5)
    elif kind == "bolt":
        pts = [hp.pt(-0.5 + d_, lat, 1.01) for d_, lat in ((0.0, 0.15), (0.12, -0.08), (0.02, -0.1), (0.15, -0.38))]
        if min(q[2] for q in pts) > 0.1:
            pv = [q[:2] for q in pts]
            tapered(P, pv, R * 0.05, R * 0.03, "lash")


def horns(P, hp):
    R = P.sp["head_r"]
    for side in (-1, 1):
        b = hp.pt(side * 0.48, 0.8, 0.98)
        if b[2] < -0.45:
            continue
        up = (side * 0.35 * (1 if abs(b[0]) > 1e-6 else 1), -1.0)
        tx, ty = b[0] + side * R * 0.55, b[1] - R * 0.95
        mx, my = b[0] + side * R * 0.05, b[1] - R * 0.55
        w = R * 0.16

        def hb(c):
            c.move_to(b[0] - w, b[1] + w * 0.4)
            c.curve_to(mx - w * 0.8, my, tx - side * R * 0.2, ty + R * 0.1, tx, ty)
            c.curve_to(tx - side * R * 0.05, ty + R * 0.25, mx + w * 0.9, my + w * 0.3, b[0] + w, b[1] + w * 0.2)
            c.close_path()
        P.part(hb, "horn", d=R * 0.1, hi=0.5)
        if P.lod >= 2:
            for k in range(1, 4):
                f = k / 4.0
                qx = b[0] + (tx - b[0]) * f * 0.9 + (mx - b[0]) * 0.4 * math.sin(math.pi * f)
                qy = b[1] + (ty - b[1]) * f * 0.9
                P.stroke(lambda c: (c.move_to(qx - w * (1 - f) * 0.8, qy + w * 0.2), c.line_to(qx + w * (1 - f) * 0.8, qy - w * 0.1)),
                         "lash", P.lw * 0.35)


def devil_ear(P, hp, near):
    R = P.sp["head_r"]
    x, y, z = hp.pt(-math.pi / 2 if near else math.pi / 2, 0.05, 0.98)
    if z < -0.25:
        return
    s = -1 if x < 0 else 1
    pts = [(x, y - R * 0.18), (x + s * R * 0.3, y - R * 0.55), (x + s * R * 0.2, y + R * 0.05), (x, y + R * 0.25)]
    P.part(lambda c: p_smooth(c, pts, True, 0.2), "skin", d=R * 0.08)


def halo(P, hp, style=True):
    R = P.sp["head_r"]
    if style == "disc":
        P.part(lambda c: p_circle(c, -math.sin(hp.yaw) * R * 0.3, -R * 0.15, R * 1.75), "halo", d=0, line=0.8)
        P.stroke(lambda c: p_circle(c, -math.sin(hp.yaw) * R * 0.3, -R * 0.15, R * 1.55), "lash", P.lw * 0.4)
        return
    c = hp.vec((0.0, 1.35, -0.1), 1.0)
    rx = R * 0.95
    ry = R * (0.18 + 0.2 * abs(math.sin(hp.cp * 0 + 0.3)))
    P.stroke(lambda cc: p_ellipse(cc, c[0], c[1], rx, ry), "lash", P.lw * 3.4)
    P.stroke(lambda cc: p_ellipse(cc, c[0], c[1], rx, ry), "halo", P.lw * 2.0)


def sparkle_eyes(P, hp):
    pass


# ============================================================ body accessories
def fur_collar(P, colour="fur"):
    r = P.rig
    sp = P.sp
    N = r.B["neck"]
    t = P.t
    top, bot = [], []
    n = 41
    for i in range(n):
        th = -math.pi * 0.62 + math.pi * 1.24 * i / (n - 1)
        a = P.surf((N[0], N[1] + 0.8, 0.0), sp["head_r"] * 0.62, sp["dep"] * 0.75, th)
        jag = 1.0 + ((0.08 + 0.22 * hash01(i, 17)) if i % 2 else -0.05 - 0.04 * hash01(i, 19))
        b = P.surf((N[0] - 0.5, N[1] - 5.0 * jag, 0.0), sp["sh_w"] * 0.95 * jag, sp["dep"] * 1.05 * jag, th)
        top.append(P.pj(a))
        bot.append(P.pj(b))
    back = [P.pj(P.surf((N[0], N[1] + 1.2, 0.0), sp["head_r"] * 0.75, sp["dep"] * 0.85, math.pi - th))
            for th in [math.pi * 0.62 - 1.24 * math.pi * i / 8 for i in range(9)]]
    pts = bot + top[::-1]
    path = P.part(lambda c: p_smooth(c, pts, True, 0.25), colour, d=2.0)
    if P.ink and P.lod >= 2:
        P.ctx.save()
        P.ctx.new_path()
        P.ctx.append_path(path)
        P.ctx.clip()
        for i in range(0, n, 1):
            a, b = top[i], bot[i]
            P.stroke(lambda c: (c.move_to(a[0] + (b[0] - a[0]) * 0.35, a[1] + (b[1] - a[1]) * 0.35),
                                c.line_to(a[0] + (b[0] - a[0]) * 0.8, a[1] + (b[1] - a[1]) * 0.85)), "lash", P.lw * 0.3)
        P.ctx.restore()


def goggles_neck(P):
    r = P.rig
    sp = P.sp
    N = r.B["neck"]
    pts = []
    for side in (-1, 1):
        p = P.surf((N[0] + 0.5, N[1] - 10.0, 0.0), sp["chest_w"] * 0.9, sp["dep"] * 1.12, side * 0.16)
        if P.zof(p) < 0:
            continue
        pts.append(P.pj(p))
    if len(pts) == 2:
        tapered(P, [pts[0], ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2 - 0.2), pts[1]], 0.5, 0.5, "goggle")
        nb = P.pj(P.surf((N[0], N[1] - 1.0, 0.0), sp["head_r"] * 0.45, sp["dep"] * 0.8, 0.0))
        for q in pts:
            P.stroke(lambda c: (c.move_to(*q), c.line_to(nb[0] + (q[0] - nb[0]) * 0.3, nb[1])), "lash", P.lw * 0.35)
    for (x, y) in pts:
        P.part(lambda c: p_ellipse(c, x, y, 1.35, 1.1), "goggle", d=0.3)
        P.part(lambda c: p_ellipse(c, x, y, 0.9, 0.72), "lens", d=0.2, line=0.5)
        P.flat(lambda c: p_ellipse(c, x - 0.3, y - 0.3, 0.3, 0.2), WHITE)


def apron(P):
    r = P.rig
    sp = P.sp
    B = r.B
    N, Hh = B["neck"], B["hip"]

    def at(u, th, wk=1.0):
        tt = (u - Hh[1]) / max(N[1] - Hh[1], 1e-3)
        if u >= Hh[1]:
            cen = (Hh[0] + (N[0] - Hh[0]) * tt, u, 0.0)
            w = sp["chest_w"] * (tt) + sp["waist_w"] * (1 - tt)
        else:
            cen = (Hh[0] * (u / max(Hh[1], 1e-3)) + r.sway * 0.2, u, 0.0)
            w = sp["waist_w"] + (sp["hem_w"] * 0.85 - sp["waist_w"]) * (1 - u / max(Hh[1], 1e-3))
        return P.surf(cen, w * 1.07 * wk, sp["dep"] * 1.15 * wk + (Hh[1] - u) * 0.12 * (u < Hh[1]), th)
    if P.zof(at(Hh[1], 0.0)) < 0:
        return
    top_u = N[1] - 8.0
    bot_u = 16.0
    L = [P.pj(at(top_u + (bot_u - top_u) * f, -0.42 - 0.35 * f)) for f in (0, 0.3, 0.55, 0.8, 1.0)]
    Rr = [P.pj(at(top_u + (bot_u - top_u) * f, 0.42 + 0.35 * f)) for f in (0, 0.3, 0.55, 0.8, 1.0)]
    pts = L + Rr[::-1]
    P.part(lambda c: p_smooth(c, pts, True, 0.15), "apron", d=2.0)
    # neck strap + waist tie
    for q in (L[0], Rr[0]):
        nb = P.pj((N[0], N[1] + 1.0, 0.0))
        P.stroke(lambda c: (c.move_to(*q), c.line_to(nb[0] + (q[0] - nb[0]) * 0.4, nb[1])), "apron", 1.2)
    pk = [P.pj(at(Hh[1] - 4.0, th)) for th in (-0.55, 0.55)] + [P.pj(at(Hh[1] - 12.0, th)) for th in (0.62, -0.62)]
    P.part(lambda c: p_poly(c, pk), shadow_of(P.pal["apron"], 0.85), d=0.8, line=0.7)
    cx = (pk[0][0] + pk[1][0]) / 2
    cy = pk[0][1]
    # hammer
    P.part(lambda c: p_capsule(c, (cx - 3.0, cy + 1.0), 0.55, (cx - 4.2, cy - 7.5), 0.6), "wood", d=0.3, line=0.6)
    P.part(lambda c: (c.new_sub_path(), c.save(), c.translate(cx - 4.3, cy - 7.8), c.rotate(-0.15),
                      c.rectangle(-2.2, -0.9, 4.2, 1.8), c.restore()), "tool", d=0.4, line=0.6)
    # pencil
    P.part(lambda c: p_capsule(c, (cx + 0.5, cy + 0.5), 0.35, (cx + 1.2, cy - 5.5), 0.35), g(0.85), d=0, line=0.5)
    # scissors (two loops + blades)
    sx, sy = cx + 3.4, cy - 1.0
    for dx in (-0.8, 0.8):
        P.stroke(lambda c: p_circle(c, sx + dx, sy - 2.2, 0.8), "lash", 0.4)
    P.stroke(lambda c: (c.move_to(sx - 0.5, sy - 1.5), c.line_to(sx + 0.2, sy + 2.5), c.move_to(sx + 0.5, sy - 1.5),
                        c.line_to(sx - 0.2, sy + 2.5)), "lash", 0.45)
    # sandpaper block + tack box on the belt line
    tb = P.pj(at(Hh[1] + 1.5, -0.25))
    P.part(lambda c: (c.new_sub_path(), c.rectangle(tb[0] - 1.8, tb[1] - 1.2, 3.6, 2.4)), g(0.62), d=0.3, line=0.6)
    for i in range(5):
        P.flat(lambda c: p_circle(c, tb[0] - 1.2 + i * 0.6, tb[1] - 0.3 + 0.4 * (i % 2), 0.18), "lash")
    sb = P.pj(at(Hh[1] - 16.0, -0.35))
    P.part(lambda c: (c.new_sub_path(), c.rectangle(sb[0] - 2.2, sb[1] - 1.4, 4.4, 2.8)), g(0.7), d=0.3, line=0.6)
    for i in range(10):
        P.flat(lambda c: p_circle(c, sb[0] - 1.8 + 3.6 * hash01(i, 5), sb[1] - 1.1 + 2.2 * hash01(i, 6), 0.1), "lash")


def mud_spots(P, amt, zone="torso"):
    if amt <= 0.01 or P.lod < 1:
        return
    r = P.rig
    sp = P.sp
    B = r.B
    N, Hh = B["neck"], B["hip"]
    seed = P.ch.seed
    n = int(6 + 26 * amt)
    for i in range(n):
        th = -1.6 + 3.2 * hash01(i, seed + 41)
        f = hash01(i, seed + 42)
        u = N[1] - 4 - f * (N[1] - 4 - max(2.0, -sp["hem_y"] + 2))
        if u >= Hh[1]:
            tt = (u - Hh[1]) / max(N[1] - Hh[1], 1e-3)
            cen = (Hh[0] + (N[0] - Hh[0]) * tt, u, 0.0)
            w = sp["chest_w"] * tt + sp["waist_w"] * (1 - tt)
        else:
            cen = (Hh[0] * 0.5, u, 0.0)
            w = sp["waist_w"] + (sp["hem_w"] - sp["waist_w"]) * (1 - u / max(Hh[1], 1e-3))
        p = P.surf(cen, w, sp["dep"] * 1.05, th)
        if P.zof(p) < 0.5:
            continue
        x, y = P.pj(p)
        rr = (0.6 + 1.8 * hash01(i, seed + 43) ** 2) * (0.6 + 0.6 * amt)
        pts = [(x + math.cos(a) * rr * (0.7 + 0.5 * hash01(i * 7 + j, seed + 44)),
                y + math.sin(a) * rr * (0.7 + 0.5 * hash01(i * 7 + j, seed + 45))) for j, a in
               enumerate([TAU * k / 7 for k in range(7)])]
        P.flat(lambda c: p_smooth(c, pts, True, 0.45), "mud", 0.95)


def tutu(P):
    r = P.rig
    sp = P.sp
    Hh = r.B["hip"]
    pts = []
    n = 17
    for i in range(n):
        th = -math.pi / 2 - 0.2 + (math.pi + 0.4) * i / (n - 1)
        p = P.surf((Hh[0], Hh[1] - 7.0 - (1.2 if i % 2 else -0.8), 0.0), sp["waist_w"] * 2.0, sp["dep"] * 2.0, th)
        pts.append(P.pj(p))
    top = [P.pj(P.surf((Hh[0], Hh[1] + 1.0, 0.0), sp["waist_w"] * 1.05, sp["dep"] * 1.05, th))
           for th in [math.pi / 2 + 0.2 - (math.pi + 0.4) * i / 8 for i in range(9)]]
    poly = pts + top
    path = P.part(lambda c: p_poly(c, poly), "tutu", d=1.5)
    if P.ink and P.lod >= 2:
        P.ctx.save()
        P.ctx.new_path()
        P.ctx.append_path(path)
        P.ctx.clip()
        for i in range(0, n, 2):
            a = pts[i]
            P.stroke(lambda c: (c.move_to(a[0], a[1]), c.line_to(a[0] * 0.5 + Hh[0] * 0.5, -Hh[1] - 1)), "lash", 0.25)
        P.ctx.restore()


def wings(P, near, spread=1.0, flap=None):
    """feathered seraph wing (drawn behind the body)"""
    r = P.rig
    sp = P.sp
    N = r.B["neck"]
    t = P.t
    side = 1 if near else -1
    if flap is None:
        fa = math.sin(TAU * t * 1.1) if r.flap_on > 0.5 else 0.15 * math.sin(t * 1.3)
    else:
        fa = math.cos(TAU * flap)          # flap 0 = wings up, 0.5 = bottom of the downstroke
    s = clamp(spread)
    root = (N[0] - sp["dep"] * 0.7, N[1] - 5.0, side * sp["sh_w"] * 0.35)
    elbow = (root[0] - 6.0 - 4 * (1 - s), root[1] + 10.0 + 6.0 * fa * s + 4 * (1 - s), root[2] + side * 10.0 * s)
    tip = (elbow[0] - 9.0 - 5 * (1 - s), elbow[1] + 18.0 * s + 10.0 * fa * s - 8 * (1 - s),
           elbow[2] + side * 17.0 * s)
    A, E, T = P.pj(root), P.pj(elbow), P.pj(tip)
    down = (0.0, 1.0)
    col_ = "wing" if near else shadow_of(P.pal["wing"], 0.92)

    def lead(u):
        if u < 0.4:
            k = u / 0.4
            return (A[0] + (E[0] - A[0]) * k, A[1] + (E[1] - A[1]) * k)
        k = (u - 0.4) / 0.6
        return (E[0] + (T[0] - E[0]) * k, E[1] + (T[1] - E[1]) * k)
    out = 1 if (T[0] - A[0]) >= 0 else -1
    # silhouette mass
    rows = []
    for layer, (n, L0, L1, wid) in enumerate(((13, 18.0, 30.0, 3.4), (10, 11.0, 15.0, 3.2), (8, 5.5, 7.5, 3.0))):
        row = []
        for i in range(n):
            u = 0.08 + 0.92 * (i / (n - 1)) if layer == 0 else (0.05 + 0.9 * i / (n - 1)) * (1.0 if layer == 1 else 0.95)
            if layer == 0:
                u = 0.35 + 0.65 * (i / (n - 1))
            b = lead(u)
            L = L0 + (L1 - L0) * (u if layer else (1 - abs(u - 0.75) * 1.4))
            ang = math.pi / 2 - out * (0.15 + 0.95 * u ** 1.6) - out * 0.25 * fa * s * u
            row.append((b, L * (0.55 + 0.45 * s), ang, wid))
        rows.append(row)
    for layer in (0, 1, 2):
        for (b, L, ang, wid) in rows[layer][::-1]:
            P.part(lambda c: p_leaf(c, b[0], b[1], L, wid * 0.5, ang), col_, d=1.2, line=0.8)
    # leading-edge bone
    P.part(lambda c: p_chain(c, [A, E, T], [3.2, 2.6, 1.2]), col_, d=1.0)


def tract_prop(P, x, y, ang, kind="tract", hs=4.0):
    """closed booklet, open booklet or a sheaf of tracts"""
    ctx = P.ctx
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    lw = P.lw
    if kind == "tract_open":
        w, h = hs * 2.1, hs * 2.9
        for side in (-1, 1):
            pts = [(0, -h / 2), (side * w, -h / 2 - side * 0 - 0.4), (side * w, h / 2 - 0.4), (0, h / 2)]
            P.part(lambda c: p_poly(c, pts), "tract", d=0.6, line=0.8)
            for k in range(3):   # three panels per page like the real tract
                y0 = -h / 2 + 0.6 + k * (h - 1.2) / 3
                x0 = min(side * 0.5, side * (w - 0.5))
                P.stroke(lambda c: (c.new_sub_path(), c.rectangle(x0 if side > 0 else side * (w - 0.5), y0,
                                                                   w - 1.0, (h - 1.2) / 3 - 0.35)), "lash", lw * 0.3)
        P.stroke(lambda c: (c.move_to(0, -h / 2), c.line_to(0, h / 2)), "lash", lw * 0.5)
    elif kind == "tracts":
        w, h = hs * 1.7, hs * 2.3
        for k in range(4):
            ctx.save()
            ctx.rotate(-0.12 + 0.09 * k)
            ctx.translate(k * 0.35, -k * 0.3)
            P.part(lambda c: (c.new_sub_path(), c.rectangle(-w / 2, -h / 2, w, h)), "tract", d=0.4, line=0.8)
            if k == 3:
                P.flat(lambda c: c.rectangle(-w / 2 + 0.4, -h / 2 + 0.4, w - 0.8, h * 0.22), "lash")
                P.stroke(lambda c: (c.new_sub_path(), c.rectangle(-w / 2 + 0.6, -h / 2 + h * 0.34, w - 1.2, h * 0.5)),
                         "lash", lw * 0.3)
            ctx.restore()
    else:
        w, h = hs * 1.7, hs * 2.3
        P.part(lambda c: (c.new_sub_path(), c.rectangle(-w / 2, -h / 2, w, h)), "tract", d=0.4, line=0.8)
        P.flat(lambda c: c.rectangle(-w / 2 + 0.4, -h / 2 + 0.4, w - 0.8, h * 0.22), "lash")
        P.stroke(lambda c: (c.new_sub_path(), c.rectangle(-w / 2 + 0.6, -h / 2 + h * 0.34, w - 1.2, h * 0.5)), "lash",
                 lw * 0.3)
    ctx.restore()


def lyre(P, x, y, hs=4.0):
    s = hs * 1.25
    arms = []
    for side in (-1, 1):
        pts = [(x + side * s * 0.6, y + s * 0.2), (x + side * s * 1.6, y - s * 1.2), (x + side * s * 1.1, y - s * 2.8),
               (x + side * s * 1.55, y - s * 3.8)]
        arms.append(pts)
        P.part(lambda c: p_chain(c, pts, [s * 0.28, s * 0.3, s * 0.22, s * 0.16]), "lyre", d=0.5)
    P.part(lambda c: p_capsule(c, (x - s * 1.4, y - s * 3.2), s * 0.18, (x + s * 1.4, y - s * 3.2), s * 0.18), "lyre", d=0.3)
    P.part(lambda c: (c.new_sub_path(), c.save(), c.translate(x, y + s * 0.3), c.scale(s * 1.0, s * 0.45), c.arc(0, 0, 1, 0, TAU),
                      c.restore()), "lyre", d=0.5)
    for k in range(6):
        sx = x - s * 0.55 + k * s * 0.22
        P.stroke(lambda c: (c.move_to(sx, y + s * 0.1), c.line_to(sx, y - s * 3.1)), "lash", P.lw * 0.28)


def tube(P, a, b, w=1.2, c="chairf"):
    P.stroke(lambda cc: (cc.move_to(*a), cc.line_to(*b)), "lash", w + P.lw * 1.4)
    P.stroke(lambda cc: (cc.move_to(*a), cc.line_to(*b)), c, w)


def camp_chair(P, part="back"):
    r = P.rig
    sp = P.sp
    Hh = r.B["hip"]
    su = Hh[1] - 3.0
    bw = sp["hip_w"] + 7.5
    pj = P.pj

    def pt(a, u, b):
        return pj((Hh[0] + a, u, b))
    if part == "back":
        # far side frame, backrest and seat sling
        for b in (-bw,):
            tube(P, pt(11, 0, b), pt(-8, su, b))
            tube(P, pt(-9, 0, b), pt(10, su, b))
            tube(P, pt(-8, su + 9, b), pt(10, su + 9, b))
            tube(P, pt(10, su, b), pt(10, su + 9, b))
        back = [pt(-8, su, -bw), pt(-12, su + 27, -bw), pt(-12, su + 27, bw), pt(-8, su, bw)]
        P.part(lambda c: p_poly(c, back), "chair", d=2.0)
        seat = [pt(-8, su, -bw), pt(10, su, -bw), pt(10, su, bw), pt(-8, su, bw)]
        P.part(lambda c: p_poly(c, seat), shadow_of(P.pal["chair"], 0.85), d=0.8)
        tube(P, pt(-8, su, -bw), pt(-12, su + 27, -bw))
        tube(P, pt(-12, su + 27, -bw), pt(-12, su + 27, bw))
    else:
        b = bw
        tube(P, pt(11, 0, b), pt(-8, su, b))
        tube(P, pt(-9, 0, b), pt(10, su, b))
        tube(P, pt(-8, su + 9, b), pt(10, su + 9, b))
        tube(P, pt(10, su, b), pt(10, su + 9, b))
        tube(P, pt(-8, su, b), pt(-12, su + 27, b))


def throne(P, part="back"):
    r = P.rig
    sp = P.sp
    Hh = r.B["hip"]
    su = Hh[1] - 3.0
    bw = sp["sh_w"] + 8.0
    pj = P.pj

    def pt(a, u, b):
        return pj((Hh[0] + a, u, b))
    if part == "back":
        back = [pt(-9, 0, -bw), pt(-11, su + 52, -bw), pt(-12, su + 64, 0), pt(-11, su + 52, bw), pt(-9, 0, bw)]
        P.part(lambda c: p_poly(c, back), "wood", d=4.0)
        inner = [pt(-9.5, su + 4, -bw * 0.7), pt(-11, su + 46, -bw * 0.7), pt(-11.5, su + 55, 0), pt(-11, su + 46, bw * 0.7),
                 pt(-9.5, su + 4, bw * 0.7)]
        P.part(lambda c: p_poly(c, inner), shadow_of(P.pal["wood"], 0.8), d=1.0, line=0.6)
        arm = [pt(-9, su + 11, -bw), pt(12, su + 11, -bw), pt(12, 0, -bw), pt(-9, 0, -bw)]
        P.part(lambda c: p_poly(c, arm), shadow_of(P.pal["wood"], 0.85), d=2.0)
        seat = [pt(-9, su, -bw), pt(12, su, -bw), pt(12, su, bw), pt(-9, su, bw)]
        P.part(lambda c: p_poly(c, seat), "wood", d=1.0)
    else:
        arm = [pt(-9, su + 11, bw), pt(12, su + 11, bw), pt(12, 0, bw), pt(-9, 0, bw)]
        P.part(lambda c: p_poly(c, arm), "wood", d=2.0)
        k = pt(12, su + 13, bw)
        P.part(lambda c: p_circle(c, k[0], k[1], 3.0), "wood", d=0.8)


# ============================================================ v1.1 close-up construction (hands, folds, ears, neck)
def _n(x, y):
    l = math.hypot(x, y) or 1.0
    return x / l, y / l


def _finger(P, a, b, c, r0, r1, skin="hand"):
    P.part(lambda cc: p_chain(cc, [a, b, c], [r0, r0 * 0.92, r1]), skin, d=r0 * 0.5, line=0.55)


def hand_ink(P, H, u, shape, near, pole_dir=None):
    """a real hand with fingers (ink close-ups). H = palm centre, u = forearm direction (wrist -> fingers)"""
    hs = P.sp["hand"]
    ux, uy = u
    nx, ny = -uy, ux
    if nx < 0:
        nx, ny = -nx, -ny
    x, y = H
    fr = hs * 0.15
    skin = "hand"

    def at(a_, b_):
        return (x + ux * a_ + nx * b_, y + uy * a_ + ny * b_)
    ang = math.atan2(uy, ux)
    if shape == "grip" and pole_dir is not None:
        px, py = _n(*pole_dir)
        qx, qy = -py, px
        if qx * ux + qy * uy < 0:
            qx, qy = -qx, -qy
        # back of the fist
        P.part(lambda c: p_ellipse(c, x - qx * hs * 0.25, y - qy * hs * 0.25, hs * 0.72, hs * 0.62,
                                   math.atan2(qy, qx)), skin, d=hs * 0.3)
        # four fingers wrapped round the pole (seen as stacked bands across it)
        for k in range(4):
            o = (k - 1.5) * hs * 0.3
            a = (x + px * o - qx * hs * 0.45, y + py * o - qy * hs * 0.45)
            b = (x + px * o + qx * hs * 0.35, y + py * o + qy * hs * 0.35)
            P.part(lambda c: p_capsule(c, a, fr * 1.05, b, fr), skin, d=fr * 0.6, line=0.55)
        t0 = (x - px * hs * 0.6 - qx * hs * 0.1, y - py * hs * 0.6 - qy * hs * 0.1)
        t1 = (x - px * hs * 0.55 + qx * hs * 0.45, y - py * hs * 0.55 + qy * hs * 0.45)
        P.part(lambda c: p_capsule(c, t0, fr * 1.2, t1, fr * 1.05), skin, d=fr * 0.6, line=0.6)
        return
    if shape in ("fist", "point", "finger", "pinch", "grip"):
        P.part(lambda c: p_ellipse(c, *at(hs * 0.15, 0.0), hs * 0.66, hs * 0.58, ang), skin, d=hs * 0.3)
        for k in range(4):   # knuckles + finger creases
            b = (k - 1.5) * hs * 0.27
            if shape in ("point", "finger") and k == 0:
                continue
            P.stroke(lambda c: (c.move_to(*at(hs * 0.62, b - hs * 0.12)), c.curve_to(*at(hs * 0.78, b - hs * 0.06),
                                                                                    *at(hs * 0.78, b + hs * 0.06),
                                                                                    *at(hs * 0.62, b + hs * 0.12))),
                     "lash", P.lw * 0.45)
            P.stroke(lambda c: (c.move_to(*at(hs * 0.35, b)), c.line_to(*at(hs * 0.5, b))), "lash", P.lw * 0.3)
        if shape == "point":
            _finger(P, at(hs * 0.55, -hs * 0.38), at(hs * 1.05, -hs * 0.4), at(hs * 1.7, -hs * 0.4), fr * 1.1, fr * 0.9)
        elif shape == "finger":
            b0 = at(hs * 0.3, -hs * 0.3)
            _finger(P, b0, (b0[0] + hs * 0.05, b0[1] - hs * 0.8), (b0[0] + hs * 0.12, b0[1] - hs * 1.6), fr * 1.1,
                    fr * 0.9)
        elif shape == "pinch":
            _finger(P, at(hs * 0.55, -hs * 0.35), at(hs * 0.95, -hs * 0.45), at(hs * 1.1, -hs * 0.2), fr * 1.05, fr * 0.9)
        # thumb over the front
        _finger(P, at(-hs * 0.05, hs * 0.5), at(hs * 0.45, hs * 0.55), at(hs * 0.8 if shape != "pinch" else hs * 1.05,
                                                                          hs * 0.2 if shape != "pinch" else -hs * 0.1),
                fr * 1.35, fr * 1.1)
        return
    spread = {"open": 0.16, "cup": 0.05, "clasp": 0.0}.get(shape, 0.04)
    curl = {"open": 0.05, "cup": 0.7, "clasp": 0.0}.get(shape, 0.45)
    P.part(lambda c: p_ellipse(c, *at(hs * 0.12, 0.0), hs * 0.62, hs * 0.5, ang), skin, d=hs * 0.3)
    lens = (0.95, 1.05, 1.0, 0.8)
    for k in range(4):
        b = (k - 1.5) * hs * 0.24
        a0 = at(hs * 0.62, b * 0.9)
        dth = (k - 1.5) * spread
        dx, dy = math.cos(ang + dth), math.sin(ang + dth)
        L = hs * lens[k] * 0.55
        a1 = (a0[0] + dx * L, a0[1] + dy * L)
        cth = ang + dth - curl * 0.9 * (1 if near else -1) * 0 - curl * 1.1
        a2 = (a1[0] + math.cos(cth) * L * 0.85, a1[1] + math.sin(cth) * L * 0.85)
        _finger(P, a0, a1, a2, fr * 1.05, fr * 0.85)
    th = ang + (0.95 if shape == "open" else 0.55) * (1 if (nx * math.sin(ang) - ny * math.cos(ang)) < 0 else -1) * 0 + 0.0
    t0 = at(-hs * 0.05, hs * 0.42)
    t1 = at(hs * 0.35, hs * (0.72 if shape == "open" else 0.55))
    t2 = at(hs * 0.7, hs * (0.9 if shape == "open" else 0.5))
    _finger(P, t0, t1, t2, fr * 1.35, fr * 1.05)


def elbow_folds(P, S, E, H, r):
    ax, ay = _n(S[0] - E[0], S[1] - E[1])
    bx, by = _n(H[0] - E[0], H[1] - E[1])
    ix, iy = ax + bx, ay + by
    bend = math.hypot(ix, iy)
    if bend < 0.15:
        ix, iy = -ay, ax
    ix, iy = _n(ix, iy)
    for k, (sa, sb) in enumerate(((0.9, 0.2), (0.7, 0.45))):
        p0 = (E[0] + ix * r * sa + ax * r * (0.2 + 0.5 * k), E[1] + iy * r * sa + ay * r * (0.2 + 0.5 * k))
        p1 = (E[0] + ix * r * 0.25, E[1] + iy * r * 0.25)
        p2 = (E[0] + ix * r * sa + bx * r * (0.3 + 0.5 * k), E[1] + iy * r * sa + by * r * (0.3 + 0.5 * k))
        tapered(P, bez(p0, p1, p2, 7), P.lw * 0.1, P.lw * 0.1, "lash", wmid=P.lw * (0.8 - 0.25 * k))


def cuff_line(P, W, u, r):
    ux, uy = u
    nx, ny = -uy, ux
    a = (W[0] - ux * r * 0.9 + nx * r, W[1] - uy * r * 0.9 + ny * r)
    b = (W[0] - ux * r * 0.9 - nx * r, W[1] - uy * r * 0.9 - ny * r)
    m = (W[0] - ux * r * 0.6, W[1] - uy * r * 0.6)
    tapered(P, bez(a, m, b, 7), P.lw * 0.2, P.lw * 0.2, "lash", wmid=P.lw * 0.6)


def knee_folds(P, Hh, K, A, r):
    ax, ay = _n(Hh[0] - K[0], Hh[1] - K[1])
    bx, by = _n(A[0] - K[0], A[1] - K[1])
    ix, iy = ax + bx, ay + by
    if math.hypot(ix, iy) < 0.15:
        ix, iy = -ay, ax
    ix, iy = _n(ix, iy)
    for k in range(2):
        p0 = (K[0] + ix * r * 0.9 + ax * r * (0.3 + 0.4 * k), K[1] + iy * r * 0.9 + ay * r * (0.3 + 0.4 * k))
        p1 = (K[0] + ix * r * 0.3, K[1] + iy * r * 0.3)
        p2 = (K[0] + ix * r * 0.9 + bx * r * (0.4 + 0.4 * k), K[1] + iy * r * 0.9 + by * r * (0.4 + 0.4 * k))
        tapered(P, bez(p0, p1, p2, 7), P.lw * 0.1, P.lw * 0.1, "lash", wmid=P.lw * (0.75 - 0.2 * k))
    # bunching at the ankle
    for k in range(2):
        q = (A[0] - bx * r * (1.2 + 0.9 * k), A[1] - by * r * (1.2 + 0.9 * k))
        nx, ny = -by, bx
        tapered(P, bez((q[0] + nx * r * 0.7, q[1] + ny * r * 0.7), (q[0] + bx * r * 0.3, q[1] + by * r * 0.3),
                       (q[0] - nx * r * 0.6, q[1] - ny * r * 0.6), 5), P.lw * 0.1, P.lw * 0.1, "lash",
                wmid=P.lw * 0.5)


def ear_ink(P, hp, near):
    R = P.sp["head_r"]
    lon = -math.pi / 2 + 0.1 if near else math.pi / 2 - 0.1
    x, y, z = hp.pt(lon, -0.14, 1.0)
    if z < -0.35:
        return
    f = max(0.3, 1.0 - abs(z) * 0.55)
    s = -1 if x < 0 else 1
    w, hh = R * 0.2 * f, R * 0.34
    pts = [(x, y - hh * 0.95), (x + s * w * 1.1, y - hh * 0.7), (x + s * w * 1.25, y - hh * 0.1),
           (x + s * w * 0.9, y + hh * 0.55), (x + s * w * 0.35, y + hh * 0.95), (x - s * w * 0.1, y + hh * 0.6),
           (x - s * w * 0.2, y)]
    P.part(lambda c: p_smooth(c, pts, True, 0.35), "skin", d=R * 0.06)
    if P.lod >= 2 and z > -0.1:
        tapered(P, bez((x + s * w * 0.2, y - hh * 0.65), (x + s * w * 1.0, y - hh * 0.4), (x + s * w * 0.55, y + hh * 0.35), 7),
                P.lw * 0.15, P.lw * 0.2, "lash", wmid=P.lw * 0.7)
        P.stroke(lambda c: (c.move_to(x + s * w * 0.1, y - hh * 0.05), c.curve_to(x + s * w * 0.45, y - hh * 0.15,
                                                                                 x + s * w * 0.5, y + hh * 0.15,
                                                                                 x + s * w * 0.25, y + hh * 0.25)),
                 "lash", P.lw * 0.45)


def neck_ink(P, neck_path, head_xy, head_rot, hp):
    """cast shadow of the jaw on the neck + the sterno tendon line (drawn in body coords)"""
    ctx = P.ctx
    R = P.sp["head_r"]
    hx, hy = head_xy
    ol = hp.outline()
    ctx.save()
    ctx.new_path()
    ctx.append_path(neck_path)
    ctx.clip()
    ctx.new_path()
    ctx.save()
    ctx.translate(hx, hy)
    ctx.rotate(head_rot)
    ctx.translate(-P.lv[0] * R * 0.12, R * 0.32)
    p_smooth(ctx, ol, True, 0.25)
    ctx.restore()
    tone = max(0.0, P.C("skin")[0] - 0.18 * P.shade_k)
    ctx.set_source_rgba(tone, tone, tone, 1.0)
    ctx.fill()
    ctx.restore()
    if P.lod >= 2:
        cr, sr = math.cos(head_rot), math.sin(head_rot)
        e = hp.pt(-math.pi / 2 + 0.35, -0.55, 1.0)
        ex, ey = hx + e[0] * cr - e[1] * sr, hy + e[0] * sr + e[1] * cr
        N = P.rig.P["neck"]
        nx_ = N[0] + P.rig.sphi * P.sp["dep"] * 0.5
        P.ctx.save()
        P.ctx.new_path()
        P.ctx.append_path(neck_path)
        P.ctx.clip()
        tapered(P, bez((ex, ey), ((ex + nx_) * 0.5 - R * 0.05, (ey + N[1]) * 0.5), (nx_, N[1] + 0.4), 7),
                P.lw * 0.1, P.lw * 0.3, "lash", wmid=P.lw * 0.7)
        P.ctx.restore()


def lashes(P, xo, yo, outer, ew, eh, close):
    """feminine lashes at the outer end of the upper lid"""
    for k in range(3):
        f = 0.55 + 0.2 * k
        bx = xo - outer * ew * (1 - f) * 1.6
        by = yo - eh * (0.55 - 0.15 * k) * (1 - close) + eh * 0.1
        tapered(P, [(bx, by), (bx + outer * ew * 0.18, by - eh * (0.38 + 0.1 * k))], P.lw * 0.9, P.lw * 0.1, "lash")


def lips(P, L, Rr, x, y, R, smile, dark):
    """upper lip with cupid's bow + lower-lip shadow for a closed mouth"""
    full = getattr(P, "_lipk", 1.0)
    up = y - R * 0.07 * full
    bow = (x, y - R * 0.02 * full)
    shape = [L, (L[0] + (x - L[0]) * 0.45, up), (x - R * 0.035, up - R * 0.015), bow, (x + R * 0.035, up - R * 0.015),
             (Rr[0] + (x - Rr[0]) * 0.45, up), Rr, (x, y + R * 0.01)]
    tone = "lip" if dark else g(max(0.0, P.C("skin")[0] - 0.14))
    P.flat(lambda c: p_smooth(c, shape, True, 0.2), tone, 1.0)
    lo = [(L[0] + (x - L[0]) * 0.3, y + R * 0.05), (x, y + R * 0.15 * full), (Rr[0] + (x - Rr[0]) * 0.3, y + R * 0.05),
          (x, y + R * 0.08)]
    if dark:
        P.flat(lambda c: p_smooth(c, lo, True, 0.3), "lip", 1.0)
        P.stroke(lambda c: (c.move_to(x - R * 0.08, y + R * 0.09), c.line_to(x + R * 0.02, y + R * 0.08)), WHITE,
                 P.lw * 0.6)
    else:
        P.flat(lambda c: p_ellipse(c, x, y + R * 0.2, R * 0.12, R * 0.035), g(max(0.0, P.C("skin")[0] - 0.2)), 1.0)


def age_ink(P, hp, k):
    """forehead lines, crow's feet and smile creases (black ink)"""
    if P.lod < 2 or k <= 0:
        return
    R = P.sp["head_r"]
    for j, lat in enumerate((0.6, 0.7)):
        pts = [hp.pt(lon, lat - 0.02 * abs(lon), 1.0) for lon in (-0.42, -0.2, 0.0, 0.2, 0.42)]
        vis = [q[:2] for q in pts if q[2] > 0.25]
        if len(vis) >= 3:
            tapered(P, vis, P.lw * 0.05, P.lw * 0.05, "lash", wmid=P.lw * 0.5 * k)
    sp = P.sp
    for side in (-1, 1):
        lon = side * sp["eye_sep"] * 1.6
        for d in (-0.06, 0.02, 0.1):
            a = hp.pt(lon, sp["eye_y"] + d, 1.0)
            b = hp.pt(lon + side * 0.16, sp["eye_y"] + d * 2.0, 1.0)
            if a[2] > 0.2 and b[2] > 0.05:
                tapered(P, [a[:2], b[:2]], P.lw * 0.45 * k, P.lw * 0.05, "lash")
        a = hp.pt(side * sp["eye_sep"] * 1.0, sp["eye_y"] - 0.2, 1.0)
        b = hp.pt(side * sp["eye_sep"] * 1.3, sp["eye_y"] - 0.24, 1.0)
        if a[2] > 0.2:
            tapered(P, [a[:2], ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + R * 0.02), b[:2]], P.lw * 0.05, P.lw * 0.05,
                    "lash", wmid=P.lw * 0.4 * k)


def strands_summer(P, hp, path):
    """romance-comic blonde: flowing strand lines from the crown to the bangs and down to the pigtail ties"""
    R = P.sp["head_r"]
    seed = P.ch.seed
    ctx = P.ctx
    ctx.save()
    ctx.new_path()
    ctx.append_path(path)
    ctx.clip()
    crown = (0.25, 1.32)
    n = 34
    for i in range(n):
        u = i / (n - 1)
        le = -2.25 + 4.5 * u + 0.05 * hash01(i, seed + 61)
        a = abs(le)
        side = 1 if le > 0 else -1
        if a < 0.95:
            mid = (le * 0.55 + 0.1, 0.95)
            end = (le * 1.05, 0.4 + 0.06 * hash01(i, seed + 62))
        else:
            mid = (le * 0.92, 0.65)
            end = (side * 1.95, -0.25)
        pts = []
        for j in range(9):
            t = j / 8
            lon = (1 - t) ** 2 * crown[0] + 2 * (1 - t) * t * mid[0] + t * t * end[0]
            lat = (1 - t) ** 2 * crown[1] + 2 * (1 - t) * t * mid[1] + t * t * end[1]
            q = hp.pt(lon, lat, 1.08)
            if q[2] > -0.05:
                pts.append(q[:2])
        if len(pts) >= 3:
            w = P.lw * (0.3 + 0.55 * hash01(i, seed + 63))
            tapered(P, pts, P.lw * 0.05, P.lw * 0.1, "lash", wmid=w)
    ctx.restore()


def shine_black(P, hp, path):
    """sharp white highlight streaks following the flow of black hair (a comic 'shine band')"""
    R = P.sp["head_r"]
    seed = P.ch.seed
    ctx = P.ctx
    ctx.save()
    ctx.new_path()
    ctx.append_path(path)
    ctx.clip()
    for i in range(11):
        lon = -1.2 + 2.2 * i / 10 + 0.06 * hash01(i, seed + 71)
        L = 0.28 + 0.22 * hash01(i, seed + 72)
        la = 0.9 + 0.12 * math.sin(i * 1.7)
        pts = [hp.pt(lon + 0.04 * j * (1 if lon > 0 else -1), la - L * j / 5, 1.1)[:2] for j in range(6)]
        tapered(P, pts, P.lw * 0.02, P.lw * 0.02, WHITE, wmid=R * (0.035 + 0.03 * hash01(i, seed + 73)))
    ctx.restore()


def goatee_hatched(P, hp, colour="beardc"):
    """grizzled goatee + moustache as hatched ink strokes over paper"""
    R = P.sp["head_r"]
    seed = P.ch.seed
    lw = P.lw
    chin = hp.cap(0.0, -1.12, 0.32, 1.04, 1.0, 26)
    mus = []
    for i in range(9):
        lon = -0.42 + 0.84 * i / 8
        mus.append(hp.pt(lon, -0.52 - 0.35 * (abs(lon) / 0.42) ** 2, 1.05))
    for i in range(9):
        lon = 0.36 - 0.72 * i / 8
        mus.append(hp.pt(lon, -0.6 - 0.35 * (abs(lon) / 0.42) ** 2, 1.05))
    shapes = [chin] + ([[q[:2] for q in mus]] if min(q[2] for q in mus) > -0.1 else [])
    for sh in shapes:
        if len(sh) < 3:
            continue
        path = P.part(lambda c: p_smooth(c, sh, True, 0.3), colour, d=R * 0.06, line=0.6)
        if P.lod < 2:
            continue
        ctx = P.ctx
        ctx.save()
        ctx.new_path()
        ctx.append_path(path)
        ctx.clip()
        for i in range(46):
            lon = -0.42 + 0.84 * hash01(i, seed + 81)
            lat = -0.5 - 0.8 * hash01(i, seed + 82)
            q = hp.pt(lon, lat, 1.05)
            if q[2] < 0.05:
                continue
            dx = -lon * R * 0.06
            L = R * (0.07 + 0.06 * hash01(i, seed + 83))
            tapered(P, [q[:2], (q[0] + dx, q[1] + L)], lw * 0.55, lw * 0.05, "lash" if i % 4 else g(0.55))
        ctx.restore()


def buttons(P, pts):
    for (x, y) in pts:
        P.part(lambda c: p_circle(c, x, y, 0.75), g(0.99) if P.ink else (0.8, 0.8, 0.8), d=0.2, line=0.5)
        P.flat(lambda c: (p_circle(c, x - 0.2, y, 0.13), p_circle(c, x + 0.2, y, 0.13)), "lash")


def cap_hair(P, hp, part="front"):
    """grizzled hair showing under a cap: tufts at the temples (front) and a fringe at the nape (back)"""
    R = P.sp["head_r"]
    seed = P.ch.seed
    for side in ((-1, 1) if part == "front" else ()):
        for k in range(4):
            lon = side * (1.35 + 0.12 * k)
            b = hp.pt(lon, 0.5 - 0.06 * k, 1.04)
            if b[2] < -0.2 or b[2] > 0.45:
                continue
            tip = hp.pt(lon + side * 0.05, 0.32 - 0.05 * k, 1.03)
            ang = math.atan2(tip[1] - b[1], tip[0] - b[0])
            L = math.hypot(tip[0] - b[0], tip[1] - b[1]) * (0.8 + 0.4 * hash01(k, seed + 3))
            P.part(lambda c: p_leaf(c, b[0], b[1], L * 0.8, R * 0.05, ang), "hair", d=0, line=0.5)
    for k in (range(7) if part == "back" else ()):
        lon = math.pi - 0.9 + 0.3 * k
        b = hp.pt(lon, 0.45, 1.04)
        if b[2] > 0.1:
            continue
        tip = hp.pt(lon, -0.15 - 0.1 * hash01(k, seed), 1.02)
        ang = math.atan2(tip[1] - b[1], tip[0] - b[0])
        L = math.hypot(tip[0] - b[0], tip[1] - b[1])
        P.part(lambda c: p_leaf(c, b[0], b[1], L, R * 0.1, ang), "hair", d=0, line=0.55)


def beard_ink(P, hp, length=1.2, colour="hair"):
    """beard as clumps of tapered strokes following the jaw, darker at the neck"""
    R = P.sp["head_r"]
    seed = P.ch.seed
    sw = P.o["wind_x"] * 0.8
    # base mass along the jaw
    cap = hp.cap(0.0, -1.38, 0.9, 1.05, 1.0, 40)
    if len(cap) < 3:
        return
    path = P.part(lambda c: p_smooth(c, cap, True, 0.3), colour, d=0, line=0.7)
    ctx = P.ctx
    # clumps: from sideburns down the jaw to the chin, longer toward the chin
    n = 15
    for i in range(n):
        u = i / (n - 1)
        lon = -1.25 + 2.5 * u
        lat = -0.35 - 0.75 * (1 - abs(2 * u - 1)) ** 0.6
        b = hp.pt(lon, lat, 1.05)
        if b[2] < -0.1:
            continue
        cen = 1 - abs(2 * u - 1)
        L = R * (0.35 + length * 0.6 * cen ** 1.5) * (0.8 + 0.4 * hash01(i, seed + 21))
        ang = math.pi / 2 - lon * 0.35 + sw * 0.3
        w = R * (0.12 + 0.06 * cen)
        P.part(lambda c: p_leaf(c, b[0], b[1], L, w, ang), colour, d=0, line=0.6)
        if P.ink and P.lod >= 2:
            tapered(P, [(b[0], b[1] + L * 0.2), (b[0] + math.cos(ang) * L * 0.85, b[1] + math.sin(ang) * L * 0.85)],
                    P.lw * 0.1, P.lw * 0.05, "lash", wmid=P.lw * 0.5)
    # darker where it tucks under the jaw
    if P.ink:
        ctx.save()
        ctx.new_path()
        ctx.append_path(path)
        ctx.clip()
        ch_ = hp.pt(0.0, -1.3, 1.0)
        P.flat(lambda c: p_ellipse(c, ch_[0] - P.lv[0] * R * 0.2, ch_[1] + R * 0.1, R * 0.5, R * 0.18), g(0.5), 1.0)
        ctx.restore()


def hood_folds(P, hp, outer):
    """drapery lines + shadowed interior for hoods"""
    R = P.sp["head_r"]
    if P.lod < 2:
        return
    top = outer[0]
    for k in range(5):
        f = (k - 2) / 2.0
        a = (top[0] + f * R * 0.35, top[1] + R * 0.25)
        m = (top[0] + f * R * 0.9, top[1] + R * 1.2)
        b = (top[0] + f * R * 1.25, top[1] + R * (2.3 + 0.2 * abs(f)))
        tapered(P, bez(a, m, b, 7), P.lw * 0.1, P.lw * 0.2, "lash", wmid=P.lw * (0.9 - 0.2 * abs(f)))

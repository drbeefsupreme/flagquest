"""t03 the hurricane of stink, seen from above (owner T03Genesis).

A 19th-century synoptic weather chart of ALCHEMY IX engraved by Dore: the Burn from above (camp arcs,
the effigy, the crowd as dots), a colossal hurricane whose spiral bands are MADE OF STINK LINES (thousands
of wavy ink strokes streaming out of the crowd along logarithmic-spiral streamlines) over billowing
engraved cloud masses (hatching follows the spiral), isobars, graticule, wind arrows, a compass rose - in a
framed panel on the tract's dark page, the storm spilling over the panel border. The six Flags whirl in it.
Chart space = design px at zoom 1, the eye at EYE.
"""
import math

import cairo
import cv2
import numpy as np

from vx import *
from vx.flag import draw_flag

T_IN = 59.339                     # hurricane cue
PANEL = (-880.0, -470.0, 880.0, 470.0)
EYE = (60.0, 20.0)
R_EYE = 62.0
B = 1.25                          # spiral pitch (log spiral psi = theta - B ln r)
ARMS = 3
PAGE = np.array([0x2B, 0x2B, 0x2D], np.float32) / 255.0   # the tract page
SERIF = "C059"


def spin(T):
    """storm rotation (rad), counter-clockwise on screen, accelerating as it forms"""
    t = T - T_IN
    return -(0.35 * t + 0.25 * t * t / (1 + 0.4 * t))


def form(T):
    """0..1 coalescence of the storm"""
    return smoothstep(T_IN - 0.05, T_IN + 1.0, T)


def view(T):
    """(zoom, centre_x, centre_y) of the chart camera"""
    t = T - T_IN
    # 59.34: close over the Burn -> pull back to show the whole chart spilling over its panel -> dive into the eye
    z_pull = 1.9 * math.exp(-1.25 * max(0.0, t)) + 0.76 * (1 - math.exp(-1.25 * max(0.0, t)))
    dive = smoothstep(60.95, 61.66, T)
    z = z_pull * math.exp(math.log(9.0) * dive ** 2.2)
    cx = EYE[0] * (0.55 + 0.45 * dive) + 30.0 * (1 - smoothstep(59.3, 60.6, T))
    cy = EYE[1] * (0.55 + 0.45 * dive) + 60.0 * (1 - smoothstep(59.3, 60.6, T))
    return z, cx, cy


def _grid(fc, z, cx, cy):
    ys, xs = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
    X = ((xs + 0.5) / fc.s - 960.0) / z + cx
    Y = ((ys + 0.5) / fc.s - 540.0) / z + cy
    return X, Y


def _to_screen(ctx, z, cx, cy):
    ctx.translate(960.0, 540.0)
    ctx.scale(z, z)
    ctx.translate(-cx, -cy)


# ------------------------------------------------------------------ the Burn from above (grey world)
def _burn_ground(ctx, T, rng_seed=3):
    rng = np.random.default_rng(rng_seed)
    x0, y0, x1, y1 = PANEL
    ex, ey = EYE
    # playa: pale, with mottled mud
    ctx.rectangle(x0 - 900, y0 - 700, x1 - x0 + 1800, y1 - y0 + 1400)
    ctx.set_source_rgb(0.9, 0.9, 0.9)
    ctx.fill()
    ctx.save()
    for i in range(160):
        px, py = rng.uniform(x0 - 200, x1 + 200), rng.uniform(y0 - 200, y1 + 200)
        rr = rng.uniform(20, 90)
        g = cairo.RadialGradient(px, py, 0, px, py, rr)
        v = rng.uniform(0.6, 0.8)
        g.add_color_stop_rgba(0, v, v, v, 0.55)
        g.add_color_stop_rgba(1, v, v, v, 0.0)
        ctx.set_source(g)
        ctx.arc(px, py, rr, 0, 2 * math.pi)
        ctx.fill()
    # the city: tight arcs of camps opening toward the open playa, radial streets (tiny under the storm)
    ctx.set_line_width(0.8)
    for ring in range(7):
        R = 96 + ring * 21
        nb = int(R * 0.26)
        for k in range(nb):
            a = math.radians(-60 + 300 * (k + 0.5) / nb)
            if int(math.degrees(a) + 60) % 50 < 5:            # radial streets
                continue
            w_, h_ = rng.uniform(9, 15), rng.uniform(8, 13)
            ctx.save()
            ctx.translate(ex + R * math.cos(a), ey + R * math.sin(a))
            ctx.rotate(a + 0.08 * rng.normal())
            ctx.rectangle(-h_ / 2, -w_ / 2, h_, w_)
            v = rng.uniform(0.5, 0.78)
            ctx.set_source_rgb(v, v, v)
            ctx.fill_preserve()
            ctx.set_source_rgb(0, 0, 0)
            ctx.stroke()
            ctx.restore()
    # the effigy (top view: a cross on a plinth) + the crowd around it
    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(ex - 9, ey - 9, 18, 18)
    ctx.fill()
    ctx.set_line_width(3)
    ctx.move_to(ex - 22, ey)
    ctx.line_to(ex + 22, ey)
    ctx.stroke()
    for i in range(900):
        a = rng.uniform(0, 2 * math.pi)
        r = 30 + abs(rng.normal(0, 55))
        ctx.arc(ex + r * math.cos(a), ey + r * math.sin(a) * 0.95, rng.uniform(1.4, 2.4), 0, 2 * math.pi)
        ctx.close_path()
    ctx.set_source_rgb(0.05, 0.05, 0.05)
    ctx.fill()
    ctx.restore()


# ------------------------------------------------------------------ cloud tone field (numpy, per pixel)
def cloud_field(fc, T, sky, z, cx, cy):
    """returns (tone, alpha, angle) at screen res"""
    X, Y = _grid(fc, z, cx, cy)
    dx, dy = X - EYE[0], Y - EYE[1]
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    th = np.arctan2(dy, dx)
    rot = spin(T)
    lnr = np.log(r / 100.0)
    psi = th - B * lnr - rot
    arms = 0.5 + 0.5 * np.cos(ARMS * psi)
    c = form(T)
    # the storm grows outward from the eye as it forms
    reach = 260.0 + 1150.0 * c
    env = np.clip((r - R_EYE) / (R_EYE * 0.7), 0, 1) * (1 - np.clip((r - reach * 0.7) / (reach * 0.5), 0, 1))
    wall = np.exp(-((r - R_EYE * 1.5) / 30.0) ** 2)
    # billows: noise in the co-rotating, untwisted frame (puffs ride the bands), plus streaks along the bands
    ua = psi
    px = r * np.cos(ua) / 150.0
    py = r * np.sin(ua) / 150.0
    puff = value_noise2(px * 1.0, py * 1.0, 71) * 0.55 + value_noise2(px * 2.3, py * 2.3, 72) * 0.3 \
        + value_noise2(px * 5.1, py * 5.1, 73) * 0.15
    Hs, Ws = sky.shape
    mu = (((th - B * lnr * 0.9 - rot) / (2 * math.pi)) % 1.0) * Ws
    mv = np.clip((lnr + 1.5) / 3.5, 0, 1) * (Hs - 1)
    tex = cv2.remap(sky, mu.astype(np.float32), mv.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP)
    streak = (0.7 - tex)
    h = (arms * 0.95 + 0.45 * puff + 0.35 * streak) * env + 0.9 * wall
    thr = 0.62 - 0.12 * c
    edge = 0.04
    cloud = np.clip((h - thr) / edge + 0.5, 0, 1) * smoothstep(0.0, 0.35, c)
    cloud = cloud.astype(np.float32)
    # Dore light: raking from the upper left: bright billow crowns, silver linings, heavy undersides
    hh = cv2.GaussianBlur(np.clip(h - thr, 0, 0.6).astype(np.float32), (0, 0), max(1.0, 9.0 * fc.s * z))
    gy, gx = np.gradient(hh)
    Hs = 110.0 * fc.s * z               # slopes in chart space (independent of render scale and zoom)
    nx_, ny_ = -gx * Hs, -gy * Hs
    nn = np.sqrt(nx_ * nx_ + ny_ * ny_ + 1.0)
    L = np.array([-0.52, -0.55, 0.66])
    L = L / np.linalg.norm(L)
    lam = np.clip((nx_ * L[0] + ny_ * L[1] + L[2]) / nn, 0, 1)
    depth = np.clip((h - thr) * 2.5, 0, 1)
    tone = 0.16 + 0.84 * lam ** 1.6 - 0.10 * depth * (1 - lam)
    rim = np.exp(-((h - thr) / 0.035) ** 2)
    tone = np.clip(tone + 0.3 * rim, 0.04, 0.97)
    ang = th + math.atan(1.0 / B) + math.pi / 2
    return tone.astype(np.float32), cloud, ang.astype(np.float32), cloud


# ------------------------------------------------------------------ the stink lines that make the bands
def make_stinks(n=1100, seed=19):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        arm = int(rng.integers(ARMS))
        out.append(dict(arm=arm, jit=float(rng.normal(0, 0.2)), r=float(80 + 850 * rng.random() ** 1.25),
                        tb=float(T_IN + rng.random() ** 1.5 * 0.9), L=float(rng.uniform(70, 150)),
                        ph=float(rng.uniform(0, 1)), w=float(rng.uniform(1.6, 2.8)), v=float(rng.uniform(0.6, 1.4))))
    return out


def draw_stinks(ctx, stinks, T, z):
    rot = spin(T)
    ctx.set_line_cap(1)
    ctx.set_source_rgb(0, 0, 0)
    for s in stinks:
        age = T - s["tb"]
        if age <= 0:
            continue
        # born in the crowd around the effigy, flung outward along the spiral to its band
        e = 1 - math.exp(-age * 2.4)
        r0 = 50.0 + (s["r"] - 50.0) * e
        if r0 > 1500:
            continue
        psi0 = 2 * math.pi * s["arm"] / ARMS + s["jit"]
        # its position along the streamline drifts with the flow
        th0 = psi0 + B * math.log(r0 / 100.0) + rot - 0.12 * s["v"] * age
        L = s["L"] * (0.4 + 0.6 * e)
        n = 22
        pts = []
        ds_th = L / (r0 * math.sqrt(1 + 1 / (B * B)))
        for i in range(n + 1):
            u = i / n
            th = th0 + ds_th * (u - 0.5)
            r = r0 * math.exp((th - th0) / B)
            # wavy: offset across the streamline
            wv = 5.5 * math.sin(2 * math.pi * (u * 2.6 - T * 1.6 + s["ph"])) * math.sin(math.pi * u) ** 0.5
            nx, ny = math.cos(th), math.sin(th)
            pts.append((EYE[0] + (r + wv) * nx, EYE[1] + (r + wv) * ny))
        ctx.set_line_width(s["w"] * min(1.0, age / 0.25) / max(0.6, z ** 0.35))
        smooth(ctx, pts)
        ctx.stroke()


# ------------------------------------------------------------------ the chart furniture
def draw_chart(ctx, T, z):
    x0, y0, x1, y1 = PANEL
    c = form(T)
    ctx.save()
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.clip()
    lw = 1.2 / max(0.5, z ** 0.5)
    # graticule
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(lw * 0.8)
    ctx.set_dash([6, 5])
    for gx in np.arange(-800, 801, 200):
        ctx.move_to(gx, y0)
        ctx.line_to(gx + 40, y1)
    for gy in np.arange(-400, 401, 200):
        ctx.move_to(x0, gy)
        ctx.line_to(x1, gy - 20)
    ctx.stroke()
    ctx.set_dash([])
    # isobars: closed, distorted rings drawn on as the storm forms
    draw = smoothstep(T_IN + 0.3, T_IN + 1.4, T)
    ctx.set_line_width(lw * 1.6)
    rot = spin(T)
    for k, (R, lab) in enumerate(((130, "985"), (230, "990"), (350, "995"), (500, "1000"), (680, "1005"))):
        pts = []
        n = 72
        for i in range(n + 1):
            a = 2 * math.pi * i / n
            rr = R * (1 + 0.08 * math.sin(3 * a + rot * 0.5 + k) + 0.05 * math.sin(5 * a - 1.3 * k))
            pts.append((EYE[0] + rr * math.cos(a) * 1.12, EYE[1] + rr * math.sin(a) * 0.92))
        m = max(2, int(n * draw))
        smooth(ctx, pts[:m + 1])
        ctx.stroke()
        if draw > 0.85:
            a = -0.9 + 0.35 * k
            rr = R * (1 + 0.08 * math.sin(3 * a + rot * 0.5 + k) + 0.05 * math.sin(5 * a - 1.3 * k))
            lx, ly = EYE[0] + rr * math.cos(a) * 1.12, EYE[1] + rr * math.sin(a) * 0.92
            _label(ctx, lab, lx, ly, 17, alpha=smoothstep(0.85, 1.0, draw))
    # wind arrows along the spiral (inflow), fletched like an old chart
    ctx.set_line_width(lw * 1.3)
    for i in range(22):
        r = 180 + 30 * i
        th = i * 2.399 + rot * 0.9
        px, py = EYE[0] + r * math.cos(th), EYE[1] + r * math.sin(th)
        ang = th + math.pi / 2 + math.atan(1.0 / B)
        L = 34
        ctx.save()
        ctx.translate(px, py)
        ctx.rotate(ang)
        ctx.move_to(-L / 2, 0)
        ctx.line_to(L / 2, 0)
        ctx.move_to(L / 2, 0)
        ctx.line_to(L / 2 - 8, -5)
        ctx.move_to(L / 2, 0)
        ctx.line_to(L / 2 - 8, 5)
        for fk in range(int(1 + i % 3)):
            ctx.move_to(-L / 2 + fk * 5, 0)
            ctx.line_to(-L / 2 + fk * 5 - 5, -9)
        ctx.stroke()
        ctx.restore()
    ctx.restore()
    # border: double rule + degree bar (alternating black/white), labels
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(7)
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.stroke()
    ctx.set_line_width(2)
    ctx.rectangle(x0 + 16, y0 + 16, x1 - x0 - 32, y1 - y0 - 32)
    ctx.stroke()
    for i, xx in enumerate(np.arange(x0 + 16, x1 - 16, 40)):
        if i % 2 == 0:
            ctx.rectangle(xx, y0 + 4, 40, 8)
            ctx.rectangle(xx, y1 - 12, 40, 8)
    for i, yy in enumerate(np.arange(y0 + 16, y1 - 16, 40)):
        if i % 2 == 0:
            ctx.rectangle(x0 + 4, yy, 8, 40)
            ctx.rectangle(x1 - 12, yy, 8, 40)
    ctx.fill()
    for gx, lab in ((-600, "83°W"), (-200, "82°30'"), (200, "82°W"), (600, "81°30'")):
        _label(ctx, lab, gx, y1 + 26, 18, box=False)
    for gy, lab in ((-200, "30°N"), (200, "29°30'")):
        _label(ctx, lab, x1 + 44, gy, 18, box=False)
    # compass rose (lower right) and the cartouche (lower left)
    _compass(ctx, x1 - 110, y1 - 110, 62, rot * 0.0)
    _cartouche(ctx, x0 + 40, y1 - 128, c)


def _label(ctx, s, x, y, size, alpha=1.0, box=True):
    ctx.save()
    ctx.select_font_face(SERIF, cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    e = ctx.text_extents(s)
    if box:
        ctx.rectangle(x - e.x_advance / 2 - 4, y - size * 0.8, e.x_advance + 8, size * 1.1)
        ctx.set_source_rgba(0.98, 0.98, 0.98, alpha)
        ctx.fill()
    ctx.set_source_rgba(0, 0, 0, alpha)
    ctx.move_to(x - e.x_advance / 2, y)
    ctx.show_text(s)
    ctx.restore()


def _compass(ctx, x, y, R, rot):
    ctx.save()
    ctx.translate(x, y)
    ctx.set_source_rgb(0.98, 0.98, 0.98)
    ctx.arc(0, 0, R * 1.15, 0, 2 * math.pi)
    ctx.fill()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(1.5)
    ctx.arc(0, 0, R * 1.15, 0, 2 * math.pi)
    ctx.stroke()
    ctx.arc(0, 0, R * 0.95, 0, 2 * math.pi)
    ctx.stroke()
    for k in range(8):
        a = k * math.pi / 4 - math.pi / 2
        L = R if k % 2 == 0 else R * 0.6
        for side in (-1, 1):
            ctx.move_to(0, 0)
            ctx.line_to(L * math.cos(a), L * math.sin(a))
            ctx.line_to(0.18 * R * math.cos(a + side * math.pi / 2), 0.18 * R * math.sin(a + side * math.pi / 2))
            ctx.close_path()
            if side > 0:
                ctx.fill()
            else:
                ctx.stroke()
    _label(ctx, "N", 0, -R * 1.25, 20, box=False)
    ctx.restore()


def _cartouche(ctx, x, y, a):
    w_, h_ = 330, 92
    ctx.save()
    ctx.rectangle(x, y, w_, h_)
    ctx.set_source_rgb(0.98, 0.98, 0.98)
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(2.5)
    ctx.stroke()
    ctx.rectangle(x + 6, y + 6, w_ - 12, h_ - 12)
    ctx.set_line_width(1)
    ctx.stroke()
    ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    for s, yy, sz in (("ALCHEMY IX", 38, 28), ("SYNOPTIC CHART OF THE STENCH", 60, 13), ("Barometer falling. Rain.", 80, 14)):
        ctx.set_font_size(sz)
        e = ctx.text_extents(s)
        ctx.move_to(x + w_ / 2 - e.x_advance / 2, y + yy)
        ctx.show_text(s)
    ctx.restore()


# ------------------------------------------------------------------ the flags in the storm
def _flag_state(k, T):
    """(base_x, base_y, ang, pole, centre_x, centre_y) of storm flag k in chart space; tumbling about its middle"""
    rot = spin(T)
    age = max(0.0, T - T_IN)
    a0 = 2 * math.pi * k / 6
    r = 80 + 175 * (1 - math.exp(-age * (0.8 + 0.1 * k))) * (0.8 + 0.04 * k)
    th = a0 - rot * (1.4 - 0.1 * k) + 0.4 * age
    cxk, cyk = EYE[0] + r * math.cos(th), EYE[1] + r * math.sin(th) * 0.95
    ang = th * 2.0 + age * (4.0 + k) * (1 if k % 2 else -1)
    # flung up at the camera as it pulls back (small while the view is close, full size once wide)
    pole = (205 - 8 * k) * min(1.0, 0.85 / view(T)[0])
    bx = cxk - 0.5 * pole * math.sin(ang)
    by = cyk + 0.5 * pole * math.cos(ang)
    return bx, by, ang, pole, cxk, cyk


def draw_storm_flags(ctx, T, z):
    for k in range(6):
        bx, by, ang, pole, _, _ = _flag_state(k, T)
        draw_flag(ctx, bx, by, pole=pole, t=T, wind=1.5, side=1 if k % 2 else -1, ang=ang, seed=40 + k,
                  light=(-0.4, -0.7, 0.6))


def flag_boxes(T, z, cx, cy):
    """screen bboxes of the storm flags (pole + cloth region, small margin) for lettering avoidance"""
    out = []
    for k in range(6):
        bx, by, ang, pole, _, _ = _flag_state(k, T)
        side = 1 if k % 2 else -1
        dx, dy = math.sin(ang), -math.cos(ang)
        qx, qy = math.cos(ang) * side, math.sin(ang) * side
        tx, ty = bx + dx * pole, by + dy * pole
        pts = [(bx, by), (tx, ty), (tx + qx * 0.42 * pole, ty + qy * 0.42 * pole),
               (tx + qx * 0.42 * pole - dx * 0.36 * pole, ty + qy * 0.42 * pole - dy * 0.36 * pole),
               (tx - dx * 0.36 * pole, ty - dy * 0.36 * pole)]
        xs = [(p[0] - cx) * z + 960 for p in pts]
        ys = [(p[1] - cy) * z + 540 for p in pts]
        out.append((min(xs) - 12, min(ys) - 12, max(xs) + 12, max(ys) + 12))
    return out


# ------------------------------------------------------------------ the frame
def render_storm(fc, T, sky, stinks, ink, comic_top=None):
    """returns rgb (engraved chart on the dark page, flags spotted in)"""
    z, cx, cy = view(T)
    wc = Canvas(fc, bg=(0.9, 0.9, 0.9))
    ctx = wc.ctx
    ctx.save()
    _to_screen(ctx, z, cx, cy)
    _burn_ground(ctx, T)
    ctx.restore()
    ground = wc.rgb()[..., 0]
    tone, alpha, ang, dens = cloud_field(fc, T, sky, z, cx, cy)
    world = ground * (1 - alpha) + tone * alpha
    wc2 = Canvas(fc)
    wc2.paint_array(np.repeat(world[..., None], 3, axis=2).astype(np.float32), 0, 0)
    ctx = wc2.ctx
    ctx.save()
    _to_screen(ctx, z, cx, cy)
    draw_stinks(ctx, stinks, T, z)
    draw_chart(ctx, T, z)
    ctx.restore()
    grey = wc2.rgb()
    angle = np.where(alpha > 0.25, ang, 0.0).astype(np.float32)
    img = ink.engrave(fc, grey, angle=angle)
    # outside the panel: the dark tract page; only the storm spills over it
    X, Y = _grid(fc, z, cx, cy)
    x0, y0, x1, y1 = PANEL
    inside = ((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)).astype(np.float32)
    inside = cv2.GaussianBlur(inside, (0, 0), 0.7)
    spill = dens
    a = np.maximum(inside, spill)[..., None]
    img = img * a + PAGE * (1 - a)
    fl = Canvas(fc)
    fctx = fl.ctx
    fctx.save()
    _to_screen(fctx, z, cx, cy)
    draw_storm_flags(fctx, T, z)
    fctx.restore()
    return img, fl.rgba()

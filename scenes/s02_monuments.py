"""s02: the three colossal full-stack memeplexes (NATION, FAITH, PHILOSOPHY) as vector set pieces.

Each draw_* works in local units: origin = base centre on the ground, -y up, nominal height ~1000
(antenna tips reach ~ -1150). Call with a placed transform (x, y, sc). Lights go to `g` (a second
cairo context composited additively with bloom by the scene), stone goes to `ctx`.

  M = Look(fog_col, fog_k)                       colour helper with atmospheric fog
  draw_nation(ctx, g, M, t, lit, **brk)          returns dict(antenna=(x,y) local)
  draw_faith(ctx, g, M, t, lit, **brk)
  draw_philosophy(ctx, g, M, t, lit, **brk)
  brk: crack 0..1, top=(dx, dy, rot) for the crowning element falling off

No flags, no real-world religious symbols. No yellow (yellow belongs to Flags).
"""
import math

import cairo
import numpy as np

from vx.canvas import col, set_color, rrect, circle, ellipse, poly, smooth, lin_grad, rad_grad, mix, shade

STONE_L = "#9097A8"
STONE_M = "#737A8C"
STONE_S = "#555A6B"
STONE_D = "#373A48"
STONE_HL = "#BAC1CF"
WIN_OFF = "#2C2F3C"
LIGHT_N = "#D6E6FF"     # nation: cold searchlight white-blue
LIGHT_F = "#E4DCFF"     # faith: pearl lilac
LIGHT_P = "#CFF3EE"     # philosophy: pale marble teal-white
MARBLE = "#DDDFE3"
MARBLE_S = "#9EA2B0"
MARBLE_D = "#707586"


class Look:
    """fog-aware colour helper"""

    def __init__(self, fog_col, fog_k, exposure=1.0):
        self.fc = col(fog_col)
        self.k = fog_k
        self.e = exposure

    def c(self, c, extra=0.0):
        r, g, b = col(c)
        k = min(1.0, self.k + extra)
        e = self.e
        return (r * e + (self.fc[0] - r * e) * k, g * e + (self.fc[1] - g * e) * k, b * e + (self.fc[2] - b * e) * k)

    def set(self, ctx, c, a=1.0, extra=0.0):
        ctx.set_source_rgba(*self.c(c, extra), a)

    def grad(self, x0, y0, x1, y1, stops):
        g = cairo.LinearGradient(x0, y0, x1, y1)
        for s in stops:
            cc = self.c(s[1])
            g.add_color_stop_rgba(s[0], *cc, s[2] if len(s) > 2 else 1.0)
        return g


def _glow_set(g, c, a):
    r, gg, b = col(c)
    g.set_source_rgba(r, gg, b, a)


# ======================================================================================= helpers
def _prism(ctx, M, x0, x1, y0, y1, split=0.3, lit=STONE_L, sh=STONE_S, top_hl=True):
    """a block seen corner-on: lit left face, shadow right face. returns split x"""
    xs = x0 + (x1 - x0) * (0.5 + split * 0.5)
    ctx.set_source(M.grad(0, y1, 0, y0, [(0, lit), (1, shade(lit, 0.86))]))
    ctx.rectangle(x0, y1, xs - x0, y0 - y1)
    ctx.fill()
    ctx.set_source(M.grad(0, y1, 0, y0, [(0, sh), (1, shade(sh, 0.8))]))
    ctx.rectangle(xs, y1, x1 - xs, y0 - y1)
    ctx.fill()
    if top_hl:
        M.set(ctx, STONE_HL, 0.8)
        ctx.rectangle(x0, y1, xs - x0, 3)
        ctx.fill()
    return xs


def _windows(ctx, g, M, x0, x1, y0, y1, dx, dy, ww, wh, lit, light, seed, t, face_k=1.0, lockstep=None):
    """grid of windows over [x0,x1]x[y1,y0] (y1 < y0). lit windows go to glow ctx g too"""
    rng = np.random.default_rng(seed)
    nx = max(1, int((x1 - x0) / dx))
    ny = max(1, int((y0 - y1) / dy))
    ox = x0 + ((x1 - x0) - nx * dx) / 2 + (dx - ww) / 2
    oy = y1 + ((y0 - y1) - ny * dy) / 2 + (dy - wh) / 2
    order = rng.random((ny, nx))
    M.set(ctx, WIN_OFF)
    for j in range(ny):
        for i in range(nx):
            ctx.rectangle(ox + i * dx, oy + j * dy, ww * face_k, wh)
    ctx.fill()
    if lit <= 0:
        return
    # cascade from bottom to top as `lit` rises 0..1
    lr, lg, lb = col(light)
    for j in range(ny):
        rowk = (ny - 1 - j) / max(1, ny - 1)
        for i in range(nx):
            on = lit * 1.35 - rowk * 0.9 - order[j, i] * 0.35
            if on <= 0:
                continue
            a = min(1.0, on * 3.0) * (0.75 + 0.25 * order[j, i])
            if order[j, i] < 0.08:
                a *= 0.5 + 0.5 * math.sin(t * 13 + i * 7 + j * 3) ** 2   # a few flicker
            x, y = ox + i * dx, oy + j * dy
            ctx.set_source_rgba(*mix(M.c(WIN_OFF), (lr, lg, lb), 0.72), a)
            ctx.rectangle(x, y, ww * face_k, wh)
            ctx.fill()
            g.set_source_rgba(lr, lg, lb, a * 0.2)
            g.rectangle(x - ww * 0.3, y - wh * 0.2, ww * face_k * 1.6, wh * 1.4)
            g.fill()
            if lockstep is not None and wh > 8:
                # tiny identical silhouettes marching in perfect step inside the lit window
                ph = lockstep
                M.set(ctx, "#20222C", a * 0.9)
                for k in range(2):
                    px = x + ww * face_k * (0.3 + 0.4 * k) + ((ph * 0.9) % 1.0 - 0.5) * ww * 0.25
                    bob = abs(math.sin(ph * math.pi)) * wh * 0.05
                    circle(ctx, px, y + wh * 0.35 - bob, wh * 0.09)
                    ctx.rectangle(px - wh * 0.07, y + wh * 0.45 - bob, wh * 0.14, wh * 0.55 + bob)
                ctx.fill()


def _cracks(ctx, M, rng_seed, box, amount, n=7):
    """jagged cracks across a box (x0, y0, x1, y1), grown by `amount` 0..1"""
    if amount <= 0:
        return
    rng = np.random.default_rng(rng_seed)
    x0, y0, x1, y1 = box
    ctx.save()
    ctx.rectangle(x0, y1, x1 - x0, y0 - y1)
    ctx.clip()
    for i in range(n):
        x = rng.uniform(x0, x1)
        y = rng.uniform(y1, y0)
        a = rng.uniform(0, 2 * math.pi)
        pts = [(x, y)]
        L = rng.uniform(0.15, 0.45) * (x1 - x0) * amount
        seg = max(2, int(10 * amount))
        for k in range(seg):
            a += rng.uniform(-0.9, 0.9)
            x += math.cos(a) * L / seg
            y += math.sin(a) * L / seg
            pts.append((x, y))
        ctx.set_line_width((x1 - x0) * 0.006 * (1 + amount))
        M.set(ctx, "#1C1D26", 0.95)
        poly(ctx, pts, close=False)
        ctx.stroke()
        ctx.set_line_width((x1 - x0) * 0.004)
        M.set(ctx, STONE_HL, 0.6)
        ctx.save()
        ctx.translate(-1.5, -1.5)
        poly(ctx, pts, close=False)
        ctx.stroke()
        ctx.restore()
    ctx.restore()


def _antenna(ctx, g, M, x, y0, y1, t, blink_seed=0.0, lit=1.0, light=LIGHT_N):
    """lattice broadcast mast from y0 (base) up to y1 (tip). returns tip"""
    h = y0 - y1
    w0 = h * 0.07
    M.set(ctx, STONE_D)
    poly(ctx, [(x - w0, y0), (x + w0, y0), (x + w0 * 0.12, y1), (x - w0 * 0.12, y1)])
    ctx.fill()
    # lattice
    ctx.set_line_width(h * 0.008)
    M.set(ctx, STONE_M)
    n = 7
    for i in range(n):
        ya = y0 - h * i / n
        yb = y0 - h * (i + 1) / n
        wa = w0 * (1 - 0.88 * i / n)
        wb = w0 * (1 - 0.88 * (i + 1) / n)
        ctx.move_to(x - wa, ya)
        ctx.line_to(x + wb, yb)
        ctx.move_to(x + wa, ya)
        ctx.line_to(x - wb, yb)
    ctx.stroke()
    # dish rings (discs, never crosses)
    for k, yy in enumerate((y0 - h * 0.45, y0 - h * 0.72)):
        M.set(ctx, STONE_S)
        ellipse(ctx, x, yy, w0 * (1.8 - k * 0.6), w0 * 0.25)
        ctx.fill()
    # red aviation light blinking
    on = 0.5 + 0.5 * math.sin(t * 5.0 + blink_seed)
    ctx.set_source_rgba(1.0, 0.25, 0.3, 0.6 + 0.4 * on)
    circle(ctx, x, y1, h * 0.03)
    ctx.fill()
    g.set_source_rgba(1.0, 0.2, 0.25, 0.5 * on)
    circle(ctx=g, x=x, y=y1, r=h * 0.09)
    g.fill()
    return (x, y1)


# ======================================================================================= NATION
def draw_nation(ctx, g, M, t, lit=0.0, crack=0.0, top=(0, 0, 0), lockstep=None, search=None):
    """brutalist fortress-tower wrapped in border walls, crowned with a colossal iron crown"""
    lr, lgc, lb = col(LIGHT_N)
    # --- searchlight beams (behind everything, in glow + faint body)
    if lit > 0:
        for k, (bx, ph) in enumerate(((-170, 0.0), (170, 1.7), (-90, 3.1), (90, 4.4))):
            sw = search if search is not None else t
            ang = -math.pi / 2 + 0.55 * math.sin(sw * 0.7 + ph) + (0.25 if bx > 0 else -0.25)
            L = 2600
            spread = 0.05
            a = min(1.0, lit * 1.6 - k * 0.15)
            if a <= 0:
                continue
            x0, y0 = bx, -790
            pts = [(x0, y0), (x0 + math.cos(ang - spread) * L, y0 + math.sin(ang - spread) * L),
                   (x0 + math.cos(ang + spread) * L, y0 + math.sin(ang + spread) * L)]
            gr = cairo.LinearGradient(x0, y0, x0 + math.cos(ang) * L, y0 + math.sin(ang) * L)
            gr.add_color_stop_rgba(0, lr, lgc, lb, 0.55 * a)
            gr.add_color_stop_rgba(0.5, lr, lgc, lb, 0.16 * a)
            gr.add_color_stop_rgba(1, lr, lgc, lb, 0.0)
            g.set_source(gr)
            poly(g, pts)
            g.fill()
    # --- inner wall (behind)
    _wall(ctx, g, M, -520, 520, -52, -112, t, lit, curve=18, seed=3, towers=(-520, -300, -130, 130, 300, 520))
    # --- tower stack (tech-stack layers)
    ctx.save()
    _prism(ctx, M, -300, 300, -80, -215, split=0.35)
    M.set(ctx, STONE_S, 0.9)
    for i in range(15):                                 # plinth ribs
        x = -290 + i * 40
        ctx.rectangle(x, -210, 9, 128)
    ctx.fill()
    _prism(ctx, M, -255, 255, -215, -475, split=0.35)
    _windows(ctx, g, M, -240, 70, -222, -468, 22, 21, 11, 13, lit, LIGHT_N, 11, t, lockstep=lockstep)
    _windows(ctx, g, M, 94, 240, -222, -468, 16, 21, 7, 13, lit, LIGHT_N, 12, t, lockstep=lockstep)
    # military stratum with buttresses
    M.set(ctx, STONE_M)
    poly(ctx, [(-218, -475), (-275, -475), (-218, -585)])
    ctx.fill()
    M.set(ctx, STONE_S)
    poly(ctx, [(218, -475), (275, -475), (218, -585)])
    ctx.fill()
    _prism(ctx, M, -218, 218, -475, -612, split=0.35)
    for j in range(4):
        y = -490 - j * 31
        M.set(ctx, WIN_OFF)
        ctx.rectangle(-200, y, 330, 7)
        ctx.rectangle(150, y, 52, 7)
        ctx.fill()
        if lit > 0:
            a = max(0.0, min(1.0, lit * 1.5 - 0.3 - j * 0.12))
            if a > 0:
                ctx.set_source_rgba(lr, lgc, lb, a * 0.9)
                ctx.rectangle(-200, y, 330, 7)
                ctx.rectangle(150, y, 52, 7)
                ctx.fill()
                g.set_source_rgba(lr, lgc, lb, a * 0.18)
                g.rectangle(-205, y - 4, 410, 15)
                g.fill()
    # monument stratum: pilasters
    _prism(ctx, M, -178, 178, -612, -765, split=0.35)
    for i in range(11):
        x = -168 + i * 32
        M.set(ctx, STONE_D if x > 60 else STONE_S, 0.85)
        ctx.rectangle(x + 16, -758, 9, 140)
        ctx.fill()
        M.set(ctx, STONE_HL, 0.35)
        ctx.rectangle(x + 13, -758, 3, 140)
        ctx.fill()
    # cornice
    _prism(ctx, M, -200, 200, -765, -792, split=0.35)
    _cracks(ctx, M, 101, (-300, -80, 300, -790), crack, n=9)
    ctx.restore()
    # --- the crown (+ antenna) : can fall off as a unit
    ctx.save()
    ctx.translate(top[0], -792 + top[1])
    ctx.rotate(top[2])
    tip = _crown(ctx, g, M, t, lit)
    ctx.restore()
    # --- outer wall + gatehouse (front)
    _wall(ctx, g, M, -660, 660, 0, -62, t, lit, curve=30, seed=5, towers=(-660, -450, -230, 230, 450, 660))
    _gate(ctx, g, M, 0, 30, lit)
    _cracks(ctx, M, 107, (-660, 30, 660, -62), crack * 0.8, n=6)
    return dict(antenna=(top[0], -792 + top[1] - 360))


def _wall(ctx, g, M, x0, x1, yb, yt, t, lit, curve, seed, towers):
    """crenellated border wall seen slightly from below; bows toward camera in the middle"""
    n = 60
    R = (x1 - x0) / 2
    cx = (x0 + x1) / 2
    xs = np.linspace(x0, x1, n)
    bow = curve * (1 - ((xs - cx) / R) ** 2)
    h = yb - yt
    ctx.set_source(M.grad(0, yt, 0, yb + curve, [(0, STONE_M), (1, STONE_S)]))
    ctx.move_to(xs[0], yb + bow[0])
    for x, b in zip(xs, bow):
        ctx.line_to(x, yt + b)
    for x, b in zip(xs[::-1], bow[::-1]):
        ctx.line_to(x, yb + b)
    ctx.close_path()
    ctx.fill()
    # merlons
    M.set(ctx, STONE_M)
    step = 26
    for x in np.arange(x0, x1, step):
        b = curve * (1 - ((x - cx) / R) ** 2)
        ctx.rectangle(x, yt + b - 12, step * 0.55, 12)
    ctx.fill()
    # string course
    M.set(ctx, STONE_HL, 0.35)
    for x, b in zip(xs[:-1], bow[:-1]):
        ctx.rectangle(x, yt + b + 6, (x1 - x0) / n + 0.5, 2.5)
    ctx.fill()
    # towers
    for tx in towers:
        b = curve * (1 - ((tx - cx) / R) ** 2)
        tw = 26
        _prism(ctx, M, tx - tw, tx + tw, yb + b, yt + b - 40, split=0.3, lit=STONE_L, sh=STONE_S)
        M.set(ctx, STONE_D)
        poly(ctx, [(tx - tw - 6, yt + b - 40), (tx + tw + 6, yt + b - 40), (tx, yt + b - 72)])
        ctx.fill()
        # watch slit
        M.set(ctx, WIN_OFF)
        ctx.rectangle(tx - 5, yt + b - 25, 10, 16)
        ctx.fill()
        if lit > 0.3:
            a = min(1.0, (lit - 0.3) * 2)
            _glow_set(ctx, LIGHT_N, a)
            ctx.rectangle(tx - 5, yt + b - 25, 10, 16)
            ctx.fill()
            _glow_set(g, LIGHT_N, 0.3 * a)
            circle(g, tx, yt + b - 17, 11)
            g.fill()


def _gate(ctx, g, M, x, yb, lit):
    for sx in (-1, 1):
        _prism(ctx, M, x + sx * 48 - 36, x + sx * 48 + 36, yb, -118, split=0.3)
        M.set(ctx, STONE_D)
        ctx.rectangle(x + sx * 48 - 40, -126, 80, 10)
        ctx.fill()
    # arch opening
    M.set(ctx, "#1D1F29")
    ctx.move_to(x - 30, yb)
    ctx.line_to(x - 30, -48)
    ctx.arc(x, -48, 30, math.pi, 2 * math.pi)
    ctx.line_to(x + 30, yb)
    ctx.close_path()
    ctx.fill()
    if lit > 0:
        a = min(1.0, lit * 1.4)
        gr = cairo.LinearGradient(0, -78, 0, yb)
        r, gg, b = col(LIGHT_N)
        gr.add_color_stop_rgba(0, r, gg, b, 0.35 * a)
        gr.add_color_stop_rgba(1, r, gg, b, 0.95 * a)
        ctx.set_source(gr)
        ctx.move_to(x - 30, yb)
        ctx.line_to(x - 30, -48)
        ctx.arc(x, -48, 30, math.pi, 2 * math.pi)
        ctx.line_to(x + 30, yb)
        ctx.close_path()
        ctx.fill()
        _glow_set(g, LIGHT_N, 0.45 * a)
        ellipse(g, x, yb - 30, 40, 45)
        g.fill()


def _crown(ctx, g, M, t, lit):
    """colossal iron crown; local origin = centre of its base; returns antenna tip"""
    IRON_L, IRON_S = "#8A90A0", "#4E5363"
    # back arches (behind band)
    ctx.set_line_width(22)
    M.set(ctx, IRON_S)
    for sx in (-1, 1):
        ctx.move_to(sx * 120, -70)
        ctx.curve_to(sx * 120, -170, sx * 30, -210, 0, -215)
        ctx.stroke()
    # points (spikes) with orbs
    for i, px in enumerate((-150, -75, 0, 75, 150)):
        hgt = 150 if i in (0, 4) else (175 if i in (1, 3) else 120)
        ctx.set_source(M.grad(px - 30, 0, px + 30, 0, [(0, IRON_L), (0.55, IRON_L), (0.56, IRON_S), (1, IRON_S)]))
        poly(ctx, [(px - 30, -60), (px + 30, -60), (px, -60 - hgt)])
        ctx.fill()
        M.set(ctx, STONE_HL)
        circle(ctx, px, -60 - hgt - 10, 13)
        ctx.fill()
        if lit > 0:
            a = min(1.0, lit * 1.3)
            _glow_set(g, LIGHT_N, 0.55 * a)
            circle(g, px, -60 - hgt - 10, 26)
            g.fill()
    # front arches
    ctx.set_line_width(26)
    for sx in (-1, 1):
        ctx.set_source(M.grad(-150, 0, 150, 0, [(0, IRON_L), (0.6, IRON_L), (1, IRON_S)]))
        ctx.move_to(sx * 150, -70)
        ctx.curve_to(sx * 150, -200, sx * 50, -250, 0, -255)
        ctx.stroke()
    # band
    ctx.set_source(M.grad(-165, 0, 165, 0, [(0, IRON_L), (0.62, IRON_L), (0.63, IRON_S), (1, IRON_S)]))
    rrect(ctx, -165, -72, 330, 72, 8)
    ctx.fill()
    M.set(ctx, STONE_HL, 0.7)
    ctx.rectangle(-165, -72, 330, 5)
    ctx.fill()
    # gems: cool glass (cyan/violet/red), glow when lit
    gems = [(-120, "#6FD3E8"), (-60, "#B08CFF"), (0, "#FF5A6E"), (60, "#B08CFF"), (120, "#6FD3E8")]
    for gx, gc in gems:
        M.set(ctx, shade(gc, 0.55))
        ellipse(ctx, gx, -36, 17, 22)
        ctx.fill()
        if lit > 0:
            a = min(1.0, lit * 1.2)
            _glow_set(ctx, gc, 0.9 * a)
            ellipse(ctx, gx, -36, 13, 18)
            ctx.fill()
            _glow_set(g, gc, 0.6 * a)
            circle(g, gx, -36, 32)
            g.fill()
    # monde orb + antenna
    ctx.set_source(M.grad(-26, 0, 26, 0, [(0, IRON_L), (1, IRON_S)]))
    circle(ctx, 0, -270, 26)
    ctx.fill()
    return _antenna(ctx, g, M, 0, -292, -370, t, 0.3, lit)


# ======================================================================================= FAITH
def draw_faith(ctx, g, M, t, lit=0.0, crack=0.0, top=(0, 0, 0), halo=(0, 0, 0), lockstep=None):
    """gigantic domed temple beneath a floating halo ring (no real-world religious symbols)"""
    lr, lgc, lb = col(LIGHT_F)
    bob = math.sin(t * 0.9) * 8
    hx, hy, hr = halo
    HY = -1135 + bob + hy
    # halo back half
    _halo(ctx, g, M, hx, HY, t, lit, front=False, rot=hr)
    # side pylons
    for sx in (-1, 1):
        px = sx * 590
        ctx.set_source(M.grad(px - 40, 0, px + 40, 0, [(0, STONE_L), (0.5, STONE_L), (0.51, STONE_S), (1, STONE_S)]))
        poly(ctx, [(px - 40, 0), (px + 40, 0), (px + 22, -520), (px - 22, -520)])
        ctx.fill()
        M.set(ctx, STONE_D)
        poly(ctx, [(px - 28, -520), (px + 28, -520), (px, -575)])
        ctx.fill()
        M.set(ctx, STONE_HL)
        circle(ctx, px, -590, 16)
        ctx.fill()
        if lit > 0:
            a = min(1.0, lit * 1.3)
            _glow_set(ctx, LIGHT_F, a)
            circle(ctx, px, -590, 14)
            ctx.fill()
            _glow_set(g, LIGHT_F, 0.35 * a)
            circle(g, px, -590, 30)
            g.fill()
    # stepped platform
    for i, (w, y0) in enumerate(((580, 0), (530, -30), (480, -60), (440, -90))):
        ctx.set_source(M.grad(0, y0 - 30, 0, y0, [(0, STONE_L), (1, STONE_S)]))
        ctx.rectangle(-w, y0 - 30, 2 * w, 30)
        ctx.fill()
        M.set(ctx, STONE_HL, 0.7)
        ctx.rectangle(-w, y0 - 30, 2 * w, 3)
        ctx.fill()
    # colonnade interior (dark or glowing)
    if lit > 0:
        gr = cairo.LinearGradient(0, -330, 0, -120)
        a = min(1.0, lit * 1.3)
        c0, c1 = mix(M.c(STONE_D), (lr, lgc, lb), 0.3 * a), mix(M.c(STONE_D), (lr, lgc, lb), 0.62 * a)
        gr.add_color_stop_rgb(0, *c0)
        gr.add_color_stop_rgb(1, *c1)
        ctx.set_source(gr)
        _glow_set(g, LIGHT_F, 0.05 * a)
        g.rectangle(-420, -330, 840, 210)
        g.fill()
    else:
        M.set(ctx, STONE_D)
    ctx.rectangle(-420, -330, 840, 210)
    ctx.fill()
    # congregation silhouettes inside, in lockstep
    if lockstep is not None:
        M.set(ctx, "#252733", 0.9)
        for i in range(40):
            x = -400 + i * 20.5 + ((lockstep * 6) % 20.5)
            bb = abs(math.sin(lockstep * math.pi)) * 1.6
            circle(ctx, x, -150 - bb, 3.2)
            ctx.rectangle(x - 2.6, -147 - bb, 5.2, 27 + bb)
        ctx.fill()
    # columns
    for i in range(19):
        x = -405 + i * 45
        gr = M.grad(x - 12, 0, x + 12, 0, [(0, STONE_L), (0.35, STONE_HL), (0.7, STONE_M), (1, STONE_S)])
        ctx.set_source(gr)
        ctx.rectangle(x - 12, -318, 24, 196)
        ctx.fill()
        M.set(ctx, STONE_M)
        ctx.rectangle(x - 17, -330, 34, 12)
        ctx.rectangle(x - 16, -126, 32, 7)
        ctx.fill()
    # entablature + dentils
    ctx.set_source(M.grad(0, -385, 0, -330, [(0, STONE_L), (1, STONE_M)]))
    ctx.rectangle(-440, -385, 880, 55)
    ctx.fill()
    M.set(ctx, STONE_S)
    for i in range(44):
        ctx.rectangle(-434 + i * 20, -352, 11, 10)
    ctx.fill()
    M.set(ctx, STONE_HL, 0.8)
    ctx.rectangle(-440, -385, 880, 4)
    ctx.fill()
    _cracks(ctx, M, 202, (-440, -120, 440, -385), crack, n=8)
    # dome group (can fall)
    ctx.save()
    ctx.translate(top[0], top[1])
    ctx.translate(0, -385)
    ctx.rotate(top[2])
    ctx.translate(0, 385)
    _dome(ctx, g, M, t, lit)
    _cracks(ctx, M, 207, (-300, -385, 300, -820), crack, n=6)
    ctx.restore()
    # halo front half
    _halo(ctx, g, M, hx, HY, t, lit, front=True, rot=hr)
    return dict(antenna=(top[0], -1010 + top[1]))


def _dome(ctx, g, M, t, lit):
    lr, lgc, lb = col(LIGHT_F)
    # drum
    _prism(ctx, M, -305, 305, -385, -545, split=0.25)
    for i in range(8):
        x = -262 + i * 75
        M.set(ctx, WIN_OFF)
        ctx.move_to(x - 16, -405)
        ctx.line_to(x - 16, -490)
        ctx.arc(x, -490, 16, math.pi, 2 * math.pi)
        ctx.line_to(x + 16, -405)
        ctx.close_path()
        ctx.fill()
        if lit > 0:
            a = max(0.0, min(1.0, lit * 1.5 - 0.2))
            ctx.set_source_rgba(lr, lgc, lb, a)
            ctx.move_to(x - 16, -405)
            ctx.line_to(x - 16, -490)
            ctx.arc(x, -490, 16, math.pi, 2 * math.pi)
            ctx.line_to(x + 16, -405)
            ctx.close_path()
            ctx.fill()
            _glow_set(g, LIGHT_F, 0.22 * a)
            ellipse(g, x, -460, 30, 60)
            g.fill()
    M.set(ctx, STONE_HL, 0.8)
    ctx.rectangle(-320, -552, 640, 8)
    ctx.fill()
    # dome
    rx, ry, cy = 300, 285, -548
    ctx.save()
    ctx.translate(0, cy)
    ctx.scale(1, ry / rx)
    ctx.arc(0, 0, rx, math.pi, 2 * math.pi)
    ctx.close_path()
    ctx.restore()
    ctx.set_source(M.grad(-rx, 0, rx, 0, [(0, STONE_M), (0.25, STONE_HL), (0.55, STONE_L), (0.85, STONE_S), (1, STONE_D)]))
    ctx.fill()
    # ribs (meridians)
    ctx.set_line_width(7)
    for k in range(-5, 6):
        ph = k / 6 * (math.pi / 2)
        pts = [(rx * math.sin(ph) * math.cos(tau), cy - ry * math.sin(tau)) for tau in np.linspace(0, math.pi / 2, 20)]
        M.set(ctx, STONE_S if k > 0 else STONE_M, 0.8)
        poly(ctx, pts, close=False)
        ctx.stroke()
    if lit > 0:
        a = min(1.0, lit * 1.2)
        # dome caught by the halo's light: soft sheen on the crown of the dome
        gr = cairo.RadialGradient(-40, cy - ry * 0.9, 10, -40, cy - ry * 0.6, rx * 0.9)
        gr.add_color_stop_rgba(0, lr, lgc, lb, 0.55 * a)
        gr.add_color_stop_rgba(1, lr, lgc, lb, 0.0)
        ctx.set_source(gr)
        ctx.save()
        ctx.translate(0, cy)
        ctx.scale(1, ry / rx)
        ctx.arc(0, 0, rx, math.pi, 2 * math.pi)
        ctx.close_path()
        ctx.restore()
        ctx.fill()
    # lantern
    _prism(ctx, M, -46, 46, cy - ry + 12, cy - ry - 70, split=0.3)
    for x in (-24, 0, 24):
        if lit > 0:
            ctx.set_source_rgba(lr, lgc, lb, min(1.0, lit * 1.4))
        else:
            M.set(ctx, WIN_OFF)
        rrect(ctx, x - 6, cy - ry - 55, 12, 40, 6)
        ctx.fill()
    M.set(ctx, STONE_M)
    ctx.save()
    ctx.translate(0, cy - ry - 70)
    ctx.scale(1, 0.7)
    ctx.arc(0, 0, 52, math.pi, 2 * math.pi)
    ctx.restore()
    ctx.fill()
    # spire / antenna: a plain needle with a sphere
    M.set(ctx, STONE_S)
    poly(ctx, [(-10, cy - ry - 100), (10, cy - ry - 100), (0, -1010)])
    ctx.fill()
    M.set(ctx, STONE_HL)
    circle(ctx, 0, cy - ry - 118, 14)
    ctx.fill()
    on = 0.5 + 0.5 * math.sin(t * 5.0 + 1.3)
    ctx.set_source_rgba(1.0, 0.25, 0.3, 0.6 + 0.4 * on)
    circle(ctx, 0, -1010, 6)
    ctx.fill()
    g.set_source_rgba(1.0, 0.2, 0.25, 0.5 * on)
    circle(g, 0, -1010, 18)
    g.fill()


def _halo(ctx, g, M, x, y, t, lit, front, rot=0.0):
    rx, ry, th = 470, 62, 30
    lr, lgc, lb = col(LIGHT_F)
    a0, a1 = (0, math.pi) if front else (math.pi, 2 * math.pi)
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    ctx.scale(1, ry / rx)
    ctx.new_path()
    ctx.arc(0, 0, rx, a0, a1)
    ctx.restore()
    ctx.save()
    ctx.set_line_width(th)
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    base = mix(M.c("#AEB5C6"), (lr, lgc, lb), min(1.0, lit * 1.2))
    if front:
        ctx.set_source_rgb(*base)
    else:
        ctx.set_source_rgb(*shade(base, 0.78))
    ctx.stroke_preserve()
    # a travelling highlight shows the slow spin
    ctx.set_line_width(th * 0.3)
    ctx.set_source_rgba(1, 1, 1, 0.25 + 0.35 * lit)
    ctx.stroke()
    ctx.restore()
    if lit > 0:
        a = min(1.0, lit * 1.2)
        for w, al in ((th * 1.5, 0.16), (th * 0.8, 0.35)):
            g.save()
            g.translate(x, y)
            g.rotate(rot)
            g.scale(1, ry / rx)
            g.new_path()
            g.arc(0, 0, rx, a0, a1)
            g.restore()
            g.set_line_width(w)
            g.set_source_rgba(lr, lgc, lb, al * a)
            g.stroke()
        # spark travelling around the ring
        ang = (t * 0.8) % (2 * math.pi)
        if (front and 0 <= ang <= math.pi) or (not front and ang > math.pi):
            sx, sy = x + math.cos(ang) * rx, y + math.sin(ang) * ry
            _glow_set(g, "#FFFFFF", 0.8 * a)
            circle(g, sx, sy, 22)
            g.fill()


# ======================================================================================= PHILOSOPHY
BOOKS = [  # (y0, height, half width, x offset, colour, fore-edge?)
    (0, 96, 380, 0, "#4F5B6E", False),
    (-96, 74, 352, 22, "#6B4A55", True),
    (-170, 90, 330, -16, "#3F6263", False),
    (-260, 70, 302, 12, "#5E4A66", False),
    (-330, 100, 290, -6, "#5B6658", True),
    (-430, 90, 262, 8, "#4A5570", False),
]


def draw_philosophy(ctx, g, M, t, lit=0.0, crack=0.0, top=(0, 0, 0)):
    """a colossal marble bust with a laurel, eyes closed, on a plinth of stacked books"""
    lr, lgc, lb = col(LIGHT_P)
    for (y0, h, w, ox, c, fore) in BOOKS:
        _book(ctx, g, M, y0, h, w, ox, c, fore, lit)
    _cracks(ctx, M, 303, (-380, 0, 380, -520), crack, n=8)
    # a doorway cut into the bottom volume: the marchers file into the library
    M.set(ctx, "#1D1F29")
    ctx.move_to(-34, 0)
    ctx.line_to(-34, -52)
    ctx.arc(0, -52, 34, math.pi, 2 * math.pi)
    ctx.line_to(34, 0)
    ctx.close_path()
    ctx.fill()
    if lit > 0:
        a = min(1.0, lit * 1.4)
        ctx.set_source_rgba(lr, lgc, lb, 0.85 * a)
        ctx.move_to(-34, 0)
        ctx.line_to(-34, -52)
        ctx.arc(0, -52, 34, math.pi, 2 * math.pi)
        ctx.line_to(34, 0)
        ctx.close_path()
        ctx.fill()
        _glow_set(g, LIGHT_P, 0.4 * a)
        ellipse(g, 0, -40, 45, 50)
        g.fill()
    # floodlight cones (behind bust)
    if lit > 0:
        a = min(1.0, lit * 1.3)
        for sx in (-1, 1):
            x0, y0 = sx * 230, -520
            gr = cairo.LinearGradient(x0, y0, x0 - sx * 60, y0 - 520)
            gr.add_color_stop_rgba(0, lr, lgc, lb, 0.1 * a)
            gr.add_color_stop_rgba(1, lr, lgc, lb, 0.0)
            g.set_source(gr)
            poly(g, [(x0 - 20, y0), (x0 + 20, y0), (x0 - sx * 260 + 180, y0 - 560), (x0 - sx * 260 - 180, y0 - 560)])
            g.fill()
    ctx.save()
    ctx.translate(top[0], top[1])
    ctx.translate(0, -520)
    ctx.rotate(top[2])
    ctx.translate(0, 520)
    # floodlit marble reads brighter than the dim world around it
    Mb = Look(M.fc, M.k * (1 - 0.4 * min(1.0, lit)), exposure=M.e + (1.0 - M.e) * 0.8 * min(1.0, lit))
    _bust(ctx, g, Mb, t, lit)
    _cracks(ctx, Mb, 309, (-200, -540, 200, -1000), crack, n=6)
    ctx.restore()
    return dict(antenna=(top[0] + 20, -1150 + top[1]))


def _book(ctx, g, M, y0, h, w, ox, c, fore, lit):
    x0, x1, yt = ox - w, ox + w, y0 - h
    # cover boards slightly proud of the page block
    M.set(ctx, shade(c, 0.7))
    rrect(ctx, x0 - 6, yt, 2 * w + 12, h, 8)
    ctx.fill()
    if fore:
        # fore-edge: stacked pages
        ctx.set_source(M.grad(0, yt + 8, 0, y0 - 8, [(0, "#CFCCC4"), (1, "#9C9AA2")]))
        ctx.rectangle(x0 + 4, yt + 9, 2 * w - 8, h - 18)
        ctx.fill()
        M.set(ctx, "#8A8894", 0.6)
        ctx.set_line_width(1.2)
        for k in range(int((h - 18) / 5)):
            yy = yt + 11 + k * 5
            ctx.move_to(x0 + 6, yy)
            ctx.line_to(x1 - 6, yy)
        ctx.stroke()
    else:
        # spine: rounded, with raised cords and a blank label panel
        ctx.set_source(M.grad(0, yt, 0, y0, [(0, shade(c, 1.25)), (0.35, c), (1, shade(c, 0.62))]))
        rrect(ctx, x0, yt + 4, 2 * w, h - 8, h * 0.32)
        ctx.fill()
        for sx in (-0.82, -0.66, 0.66, 0.82):
            xx = ox + sx * w
            M.set(ctx, shade(c, 0.6))
            ctx.rectangle(xx - 5, yt + 6, 10, h - 12)
            ctx.fill()
            M.set(ctx, shade(c, 1.35), 0.8)
            ctx.rectangle(xx - 5, yt + 6, 3, h - 12)
            ctx.fill()
        M.set(ctx, shade(c, 0.8))
        rrect(ctx, ox - w * 0.28, yt + h * 0.28, w * 0.56, h * 0.44, 4)
        ctx.fill()
        if lit > 0:
            a = min(1.0, lit * 1.2)
            lr, lgc, lb = col(LIGHT_P)
            ctx.set_source_rgba(lr, lgc, lb, 0.35 * a)
            ctx.set_line_width(2.5)
            rrect(ctx, ox - w * 0.28, yt + h * 0.28, w * 0.56, h * 0.44, 4)
            ctx.stroke()
            for sx in (-0.82, -0.66, 0.66, 0.82):
                ctx.rectangle(ox + sx * w - 5, yt + 6, 3, h - 12)
            ctx.fill()
    # shadow of the book above
    gr = M.grad(0, yt, 0, yt + h * 0.35, [(0, "#1A1B24", 0.55), (1, "#1A1B24", 0.0)])
    ctx.set_source(gr)
    ctx.rectangle(x0 - 6, yt, 2 * w + 12, h * 0.35)
    ctx.fill()


def _curls(ctx, M, pts, r0, r1, seed, light=MARBLE, dark=MARBLE_S, line=MARBLE_D):
    """marble curls: shaded discs with a spiral groove"""
    rng = np.random.default_rng(seed)
    for (x, y) in pts:
        r = rng.uniform(r0, r1)
        ctx.set_source(M.grad(x - r, y - r, x + r * 0.8, y + r * 0.8, [(0, light), (1, dark)]))
        circle(ctx, x, y, r)
        ctx.fill()
        M.set(ctx, line, 0.55)
        ctx.set_line_width(r * 0.16)
        a0 = rng.uniform(0, 6.28)
        ctx.arc(x, y, r * 0.55, a0, a0 + 3.8)
        ctx.stroke()


def _face_path(ctx, cx, cy):
    """frontal face oval (forehead to jaw) - the beard covers the lower half"""
    ctx.move_to(cx, cy - 175)
    ctx.curve_to(cx + 88, cy - 175, cx + 125, cy - 110, cx + 124, cy - 20)
    ctx.curve_to(cx + 122, cy + 60, cx + 95, cy + 130, cx, cy + 150)
    ctx.curve_to(cx - 95, cy + 130, cx - 122, cy + 60, cx - 124, cy - 20)
    ctx.curve_to(cx - 125, cy - 110, cx - 88, cy - 175, cx, cy - 175)
    ctx.close_path()


def _bust(ctx, g, M, t, lit):
    """frontal bearded philosopher, eyes closed, laurel wreath, toga; base at y=-520"""
    lr, lgc, lb = col(LIGHT_P)
    # socle
    _prism(ctx, M, -190, 190, -520, -560, split=0.2, lit=MARBLE, sh=MARBLE_S)
    ctx.set_source(M.grad(-150, 0, 150, 0, [(0, MARBLE), (0.6, MARBLE_S), (1, MARBLE_D)]))
    rrect(ctx, -140, -596, 280, 40, 14)
    ctx.fill()
    cx, cy = 0, -905
    # --- chest & shoulders (rounded bust truncation)
    ctx.move_to(-95, -830)
    ctx.curve_to(-170, -825, -262, -800, -298, -752)
    ctx.curve_to(-325, -712, -318, -660, -280, -636)
    ctx.curve_to(-200, -606, -90, -597, 0, -596)
    ctx.curve_to(90, -597, 200, -606, 280, -636)
    ctx.curve_to(318, -660, 325, -712, 298, -752)
    ctx.curve_to(262, -800, 170, -825, 95, -830)
    ctx.close_path()
    ctx.set_source(M.grad(-320, 0, 320, 0, [(0, MARBLE_S), (0.18, MARBLE), (0.55, MARBLE), (0.85, MARBLE_S), (1, MARBLE_D)]))
    ctx.fill()
    # toga: a diagonal drape over the left shoulder, folds
    ctx.move_to(-298, -752)
    ctx.curve_to(-240, -815, -130, -820, -60, -785)
    ctx.curve_to(40, -725, 170, -660, 255, -625)
    ctx.curve_to(200, -606, 150, -600, 110, -598)
    ctx.curve_to(20, -650, -130, -700, -240, -650)
    ctx.curve_to(-290, -640, -325, -700, -298, -752)
    ctx.close_path()
    ctx.set_source(M.grad(-300, -790, 250, -600, [(0, MARBLE), (0.5, "#D2D0CC"), (1, MARBLE_S)]))
    ctx.fill()
    ctx.set_line_width(5)
    for k in range(4):
        M.set(ctx, MARBLE_D, 0.45)
        ctx.move_to(-285 + k * 25, -700 + k * 10)
        ctx.curve_to(-180 + k * 20, -765 + k * 14, -40 + k * 20, -745 + k * 16, 170 + k * 18, -640 + k * 8)
        ctx.stroke()
    # --- neck (mostly under the beard)
    ctx.set_source(M.grad(-80, 0, 80, 0, [(0, MARBLE), (0.6, MARBLE_S), (1, MARBLE_D)]))
    poly(ctx, [(-78, -870), (78, -870), (92, -760), (-92, -760)])
    ctx.fill()
    # --- back hair mass (behind the face)
    back = [(cx + math.cos(a) * 150, cy - 10 + math.sin(a) * 175) for a in np.linspace(math.pi * 0.92, math.pi * 2.08, 22)]
    _curls(ctx, M, back, 30, 40, 7, light="#D8D5CF", dark=MARBLE_D)
    # ears
    for sx in (-1, 1):
        ctx.set_source(M.grad(cx + sx * 115, 0, cx + sx * 150, 0, [(0, MARBLE_S), (1, MARBLE)] if sx < 0 else [(0, MARBLE_S), (1, MARBLE_D)]))
        ellipse(ctx, cx + sx * 128, cy - 5, 26, 45)
        ctx.fill()
    # --- face
    _face_path(ctx, cx, cy)
    ctx.set_source(M.grad(cx - 125, 0, cx + 125, 0, [(0, "#CFCCC6"), (0.25, MARBLE), (0.55, MARBLE), (0.85, MARBLE_S), (1, MARBLE_D)]))
    ctx.fill()
    # forehead crease + temples shading
    M.set(ctx, MARBLE_S, 0.55)
    ctx.set_line_width(4)
    ctx.move_to(cx - 50, cy - 118)
    ctx.curve_to(cx - 20, cy - 126, cx + 20, cy - 126, cx + 50, cy - 118)
    ctx.stroke()
    # brow ridge + closed eyes (deep sockets)
    for sx in (-1, 1):
        ex = cx + sx * 50
        ctx.set_source(M.grad(ex, cy - 60, ex, cy - 10, [(0, MARBLE_S, 0.0), (1, MARBLE_S, 0.85)]))
        ellipse(ctx, ex, cy - 32, 40, 24)
        ctx.fill()
        M.set(ctx, MARBLE_D)
        ctx.set_line_width(5)
        ctx.move_to(ex - 30, cy - 34)
        ctx.curve_to(ex - 12, cy - 22, ex + 12, cy - 22, ex + 30, cy - 34)
        ctx.stroke()
        M.set(ctx, MARBLE_S)
        ctx.set_line_width(7)
        ctx.move_to(ex - 40 * sx * -1 if False else ex - 42, cy - 60)
        ctx.curve_to(ex - 20, cy - 72, ex + 20, cy - 72, ex + 42, cy - 58)
        ctx.stroke()
    # nose: long straight bridge with a shadow flank
    ctx.move_to(cx - 8, cy - 55)
    ctx.line_to(cx - 26, cy + 40)
    ctx.curve_to(cx - 30, cy + 55, cx - 12, cy + 60, cx, cy + 58)
    ctx.curve_to(cx + 14, cy + 60, cx + 30, cy + 55, cx + 26, cy + 40)
    ctx.line_to(cx + 8, cy - 55)
    ctx.close_path()
    ctx.set_source(M.grad(cx - 26, 0, cx + 26, 0, [(0, MARBLE), (0.5, MARBLE), (0.51, MARBLE_S), (1, MARBLE_D)]))
    ctx.fill()
    M.set(ctx, MARBLE_D, 0.8)
    for sx in (-1, 1):
        ellipse(ctx, cx + sx * 12, cy + 55, 7, 4)
        ctx.fill()
    # --- beard (curl rows forming an inverted dome) + moustache
    rows = [(cy + 55, 118, 7), (cy + 95, 112, 7), (cy + 135, 100, 6), (cy + 172, 84, 5), (cy + 205, 60, 4), (cy + 232, 30, 2)]
    pts = []
    for (yy, hw, n) in rows:
        for i in range(n):
            u = (i + 0.5) / n
            pts.append((cx - hw + 2 * hw * u, yy))
    _curls(ctx, M, pts, 22, 28, 13, light="#DDDAD4", dark=MARBLE_S)
    for sx in (-1, 1):
        ctx.move_to(cx, cy + 66)
        ctx.curve_to(cx + sx * 30, cy + 58, cx + sx * 62, cy + 72, cx + sx * 74, cy + 102)
        ctx.curve_to(cx + sx * 50, cy + 88, cx + sx * 25, cy + 84, cx, cy + 82)
        ctx.close_path()
        ctx.set_source(M.grad(cx - 70, 0, cx + 70, 0, [(0, MARBLE), (0.6, MARBLE_S), (1, MARBLE_D)]))
        ctx.fill()
    M.set(ctx, MARBLE_D, 0.9)
    ctx.set_line_width(4)
    ctx.move_to(cx - 18, cy + 96)
    ctx.curve_to(cx - 6, cy + 100, cx + 6, cy + 100, cx + 18, cy + 96)
    ctx.stroke()
    # --- front hair curls under the wreath
    fr = [(cx + math.cos(a) * 118, cy - 18 + math.sin(a) * 150) for a in np.linspace(math.pi * 1.12, math.pi * 1.88, 11)]
    _curls(ctx, M, fr, 22, 28, 21)
    # --- laurel wreath across the brow (patina green-grey)
    LEAF, LEAF_S = "#93AE9F", "#5F776A"
    band = [(cx + math.cos(a) * 138, cy - 30 + math.sin(a) * 118) for a in np.linspace(math.pi * 1.04, math.pi * 1.96, 19)]
    mid = len(band) // 2
    for i in range(len(band) - 1):
        bx, by = band[i]
        nx, ny = band[i + 1]
        ang = math.atan2(ny - by, nx - bx) + (math.pi if i < mid else 0.0)   # leaves point away from the centre
        for side in (-1, 1):
            M.set(ctx, LEAF if (side < 0) == (i < mid) else LEAF_S)
            ctx.save()
            ctx.translate(bx, by)
            ctx.rotate(ang + side * 0.6)
            ellipse(ctx, 17, 0, 21, 8)
            ctx.fill()
            ctx.restore()
    M.set(ctx, LEAF_S)
    circle(ctx, band[mid][0], band[mid][1], 9)
    ctx.fill()
    # --- floodlight wash from below when lit
    if lit > 0:
        a = min(1.0, lit * 1.2)
        ctx.save()
        _face_path(ctx, cx, cy)
        ctx.clip()
        gr = cairo.LinearGradient(0, cy + 150, 0, cy - 150)
        gr.add_color_stop_rgba(0, lr, lgc, lb, 0.2 * a)
        gr.add_color_stop_rgba(1, lr, lgc, lb, 0.0)
        ctx.set_source(gr)
        ctx.paint()
        ctx.restore()
        g.save()
        _face_path(g, cx, cy)
        g.set_line_width(12)
        g.set_source_rgba(lr, lgc, lb, 0.18 * a)
        g.stroke()
        g.restore()
    # antenna out of the crown of the head
    _antenna(ctx, g, M, cx, cy - 170, -1150, t, 2.1, lit)

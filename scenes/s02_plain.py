"""s02 shot A (21.0-34.8): the plain of the memeplexes.

A true perspective plain (camera height / horizon pitch / dolly), three colossal memeplexes as fogged
billboards, ant-streams of people marching into their gates, a foreground procession of grey silhouettes,
light-ups on N04 words, broadcast rings on N05, and the thin beige line of slop on the horizon (N06).
"""
import math

import cairo
import cv2
import numpy as np

from vx import (W, H, C, col, mix, shade, set_color, circle, ellipse, poly, lin_grad, rad_grad, blur, clamp,
                lerp, seg, smoothstep, smootherstep, ease_in_out, ease_out, ease_in, ease_in_out_sine, hash01,
                noise1, fbm1, fbm2, Canvas)
from vx.chars import Character
from scenes import s02_monuments as MN

F = 1500.0                       # focal length (design px)
FOG0 = "#9DA3B0"                 # concrete_l: the s01 hand-off colour
SKY_TOP = "#868C9B"
SKY_HOR = "#AEB2BE"
GROUND_NEAR = "#4E5261"
GROUND_MID = "#687080"
FOG_D = 9500.0                   # fog e-folding distance (m)
T_N, T_F, T_P = 25.20, 26.20, 26.87    # overwritten from the timeline in setup
RING_T = []

MONS = [
    dict(name="faith", X=250.0, Z=8200.0, m=2.55, fn=MN.draw_faith, light=MN.LIGHT_F),
    dict(name="nation", X=-2350.0, Z=5300.0, m=1.9, fn=MN.draw_nation, light=MN.LIGHT_N),
    dict(name="philo", X=2550.0, Z=5700.0, m=1.85, fn=MN.draw_philosophy, light=MN.LIGHT_P),
]
ROADS = []   # built in setup: (P0 gate xz, P1 far-end xz, mon index)
# a whole world of lesser memeplexes receding into the fog (kind, X, Z, metres per local unit)
FAR = [("nation", -9800.0, 19000.0, 1.5), ("philo", -5600.0, 23000.0, 1.3), ("faith", -3900.0, 15500.0, 1.4),
       ("nation", 5200.0, 17000.0, 1.6), ("faith", 8400.0, 21500.0, 1.7), ("philo", 11800.0, 16000.0, 1.3),
       ("nation", 1300.0, 26000.0, 1.8), ("faith", -12500.0, 26000.0, 1.8), ("nation", 15500.0, 25000.0, 1.6),
       ("philo", -15500.0, 14500.0, 1.2)]
FAR_FN = {"nation": MN.draw_nation, "faith": MN.draw_faith, "philo": MN.draw_philosophy}


# ============================================================================ camera
class Cam:
    def __init__(self, T):
        a = smootherstep(21.25, 27.7, T)
        b = ease_in_out_sine(clamp((T - 26.8) / (34.8 - 26.8)))
        push = ease_in(clamp((T - 33.6) / 1.2), 2)
        self.h = 1.7 + 56.0 * ease_in_out(a, 2.2)
        self.hor = 830.0 - 150.0 * ease_in_out(a, 2.0) + 18 * b
        self.z = 140.0 * a + 380.0 * b + 120 * push
        self.x = 110.0 * a - 60.0 * b
        self.f = F * (1 + 0.06 * push)
        self.T = T
        self.D = dim(T)

    def proj(self, X, Y, Z):
        d = Z - self.z
        return W / 2 + self.f * (X - self.x) / d, self.hor + self.f * (self.h - Y) / d, self.f / d, d


def dim(T):
    """the grey world dims like a theatre before the machines of meaning light up"""
    return 1.0 - 0.34 * smootherstep(22.6, 25.1, T)


def dcol(cc, D):
    r, g, b = col(cc)
    return (r * D, g * D, b * D * (1 + 0.05 * (1 - D)))


def fogk(d):
    return 1.0 - math.exp(-max(d, 0.0) / FOG_D)


# ============================================================================ setup
def setup(S, st):
    global T_N, T_F, T_P, RING_T
    tl = S.tl
    T_N = tl.word("N04", "Nations")[0]
    T_F = tl.word("N04", "Faiths")[0]
    T_P = tl.word("N04", "Philosophies")[0]
    ws = tl.words("N05")
    # broadcast pulses: on N05's stressed words, then a steady heartbeat until the slop arrives
    RING_T = sorted(set([round(w["s"], 3) for w in ws if w["w"].lower() in
                         ("full-stack", "memeplexes", "each", "one", "promising", "whole", "story")] +
                        [32.9, 33.35]))
    st["ring_t"] = RING_T
    s = S.s
    w, h = int(round(W * s)), int(round(H * s))
    # overcast sky texture (low-frequency), taller than the frame so the horizon can move
    sh, sw = int(h * 1.6) // 2 + 2, int(w * 1.25) // 2 + 2
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float64)
    k = 2.0 / s
    n = fbm2(xx * k / 640.0, yy * k / 120.0, seed=11, octaves=4)
    n2 = fbm2(xx * k / 240.0, yy * k / 52.0, seed=5, octaves=3)
    m = np.clip((n + 0.1) * 2.2, -1, 1)                    # soft-edged cloud masses
    clouds = (0.75 * m + 0.25 * n2).astype(np.float32)
    clouds = cv2.GaussianBlur(clouds, (0, 0), 3.0)
    st["sky_tex"] = cv2.resize(clouds, (sw * 2, sh * 2), interpolation=cv2.INTER_CUBIC)
    # mist texture for the opening fog-lift
    n3 = fbm2(xx * k / 150.0, yy * k / 90.0, seed=23, octaves=4).astype(np.float32)
    st["mist_tex"] = cv2.resize(n3, (sw * 2, sh * 2), interpolation=cv2.INTER_CUBIC)
    # ground patch texture in plan (X in [-14000,14000], Z in [-1000, 27000]) ~ 27 m / texel
    g = np.mgrid[0:1024, 0:1024].astype(np.float64)
    gt = fbm2(g[1] / 40.0, g[0] / 40.0, seed=3, octaves=5) * 0.7 + fbm2(g[1] / 9.0, g[0] / 9.0, seed=9, octaves=3) * 0.3
    st["ground_tex"] = gt.astype(np.float32)
    # roads: from each gate through a chosen plan point (so they sweep in from the frame bottom), to far behind
    ROADS.clear()
    through = {"faith": ((70.0, 420.0),),
               "nation": ((-90.0, 460.0), (-1900.0, 1400.0), (-4200.0, 2600.0)),
               "philo": ((230.0, 460.0), (2300.0, 1300.0), (4800.0, 2800.0))}
    for mi, m in enumerate(MONS):
        gz = m["Z"] - 60.0
        for (tx, tz) in through[m["name"]]:
            p0 = (m["X"], gz)
            k = (gz - (-2500.0)) / (gz - tz)
            p1 = (m["X"] + (tx - m["X"]) * k, -2500.0)
            ROADS.append((p0, p1, mi))
    # procession characters (foreground at eye level)
    st["walkers"] = [Character("citizen", seed=100 + i) for i in range(6)]


# ============================================================================ lights / timing
def lit_curve(T, t0):
    if T < t0 - 0.05:
        return 0.0
    x = T - t0 + 0.05
    base = ease_out(clamp(x / 0.45), 3)
    flash = 0.45 * math.exp(-x * 5.0) * min(1.0, x * 20)
    lit = base + flash
    # the lights stutter when the slop appears
    if T > 33.75:
        q = T - 33.75
        flick = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(q * 41 + t0 * 7)) * (0.5 + 0.5 * math.sin(q * 17.3 + t0))
        lit *= lerp(1.0, flick, min(1.0, q * 2))
    return lit


def lockstep_phase(T, seedoff):
    """walking phase in steps; people fall into perfect lockstep for N05 and halt when the slop appears"""
    sync = smoothstep(27.9, 28.6, T)
    halt = smoothstep(33.7, 34.3, T)
    # integrate a speed that drops to 0 at the halt (closed form: freeze time)
    Te = T if T < 33.7 else 33.7 + (T - 33.7) * (1 - halt) * 0.5
    # feet land on the score's beat grid: 28.56 + 0.63 k (95.238 BPM)
    return (Te - 28.56) / 0.63 + seedoff * (1 - sync), Te


# ============================================================================ render
def render(fc, st):
    T = fc.T
    s = fc.s
    c = Cam(T)
    w, h = fc.w, fc.h
    img = sky(fc, st, c)
    img = ground(fc, st, c, img)
    cv = Canvas(fc)
    gl = Canvas(fc)
    ctx, g = cv.ctx, gl.ctx
    cv.paint_array(img)
    ctx.save()
    ctx.rectangle(-10, -10, W + 20, c.hor + 11)
    ctx.clip()
    horizon_features(ctx, c, T, g)
    ctx.restore()
    grid_lines(ctx, c)
    roads(ctx, c)
    far_memeplexes(ctx, g, c, T)
    # depth-sorted monuments interleaved with crowd buckets
    order = sorted(range(len(MONS)), key=lambda i: -(MONS[i]["Z"] - c.z))
    people = crowd_points(c, T)
    lastd = 1e9
    for k, mi in enumerate(order):
        d_m = MONS[mi]["Z"] - c.z
        draw_people(ctx, people, d_m, lastd, T, c)
        draw_monument(ctx, g, c, MONS[mi], T)
        lastd = d_m
    draw_people(ctx, people, 0.0, lastd, T, c)
    near_people(ctx, st, people, T, c)
    rings(ctx, g, c, T)
    # foreground procession (eye-level start of the crane)
    if c.h < 14:
        procession(ctx, st, c, T)
    shafts(g, c, T)
    img = cv.rgb()
    img = ground_mist(img, st, c, T, s)
    if T > 21.3:
        gg = gl.rgba()[..., :3]
        img += blur(gg, 2.5 * s) * 0.55 + blur(gg, 16 * s) * 0.4 + blur(gg, 60 * s) * 0.2
        img = reflections(img, gg, c, s)
    # opening fog lift from flat concrete_l
    cover = 1.0 - ease_out(clamp((T - 21.0) / 2.0), 1.7)
    if cover > 0:
        mt = st["mist_tex"]
        oy = int((mt.shape[0] - h) * 0.3)
        ox = int((T - 21.0) * 40 * s) % max(1, mt.shape[1] - w)
        n = mt[oy:oy + h, ox:ox + w]
        a = np.clip(cover * 1.45 - 0.45 * (0.5 + 0.5 * n) * 0.999, 0, 1)[..., None]
        img = img * (1 - a) + np.array(col(FOG0), np.float32) * a
    return img


def ground_mist(img, st, c, T, s):
    """low banks of mist drifting across the plain at the feet of the memeplexes"""
    h, w = img.shape[:2]
    j0 = max(0, int((c.hor - 170) * s))
    j1 = min(h, int((c.hor + 150) * s))
    if j1 <= j0:
        return img
    mt = st["mist_tex"]
    ox = int((T * 28 + c.x * 0.3) * s) % max(1, mt.shape[1] - w)
    oy = int(mt.shape[0] * 0.15)
    n = mt[oy + j0:oy + j1, ox:ox + w]
    ys = (np.arange(j0, j1, dtype=np.float32) + 0.5) / s
    prof = np.exp(-((ys - (c.hor - 25)) / 75.0) ** 2)[:, None]
    blow = 1.0 - smoothstep(33.3, 34.1, T)       # the wind before the wave blows the mist away
    a = (np.clip((0.5 + 0.5 * n) * 1.5 - 0.45, 0, 1) * prof * 0.42 * blow)[..., None]
    fogc = np.array(dcol(SKY_HOR, c.D), np.float32)
    img = img.copy()
    img[j0:j1] = img[j0:j1] * (1 - a) + fogc * a
    return img


def shafts(g, c, T):
    """cool light shafts breaking through the overcast (fade as the world dims before the light-ups)"""
    a = 1.0 - smoothstep(22.8, 24.6, T)
    if a <= 0:
        return
    ox, oy = 1450.0 - c.x * 0.05, -500.0
    for k in range(6):
        ang = math.pi / 2 + 0.12 + (k - 2.5) * 0.11 + 0.03 * math.sin(T * 0.4 + k * 1.7)
        wdt = 0.025 + 0.02 * hash01(k, 71)
        L = 2200.0
        al = a * (0.05 + 0.05 * hash01(k, 72)) * (0.7 + 0.3 * math.sin(T * 0.6 + k * 2.3))
        pts = [(ox, oy), (ox + math.cos(ang - wdt) * L, oy + math.sin(ang - wdt) * L),
               (ox + math.cos(ang + wdt) * L, oy + math.sin(ang + wdt) * L)]
        gr = cairo.LinearGradient(ox, oy, ox + math.cos(ang) * L, oy + math.sin(ang) * L)
        gr.add_color_stop_rgba(0, 0.9, 0.92, 0.96, 0.0)
        gr.add_color_stop_rgba(0.35, 0.9, 0.92, 0.96, al)
        gr.add_color_stop_rgba(0.75, 0.9, 0.92, 0.96, al * 0.6)
        gr.add_color_stop_rgba(1, 0.9, 0.92, 0.96, 0.0)
        g.set_source(gr)
        poly(g, pts)
        g.fill()


def far_memeplexes(ctx, g, c, T):
    for i, (kind, X, Z, m) in enumerate(FAR):
        d = Z - c.z
        sx, sy, k, _ = c.proj(X, 0, Z)
        sc = k * m
        if sx < -700 * sc or sx > W + 700 * sc:
            continue
        M = MN.Look(dcol(SKY_HOR, c.D), min(0.9, fogk(d) * 0.95), exposure=c.D)
        t0 = {"nation": T_N, "faith": T_F, "philo": T_P}[kind] + 0.12 + hash01(i, 4) * 0.35
        lit = lit_curve(T, t0) * 0.8
        for cc in (ctx, g):
            cc.save()
            cc.translate(sx, sy)
            cc.scale(sc, sc)
        g.push_group()
        FAR_FN[kind](ctx, g, M, T + i, lit)
        g.pop_group_to_source()
        g.paint_with_alpha(0.55)
        for cc in (ctx, g):
            cc.restore()


def reflections(img, gg, c, s):
    """the polished plain mirrors the machines' lights (vertical streaks below their bases)"""
    h = img.shape[0]
    axis = int(round((c.hor + 10) * s))
    if axis >= h - 2 or axis < 2:
        return img
    n = min(axis, h - axis)
    refl = np.zeros_like(gg)
    refl[axis:axis + n] = gg[axis - n:axis][::-1]
    small = cv2.resize(refl, (refl.shape[1] // 4, refl.shape[0] // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(0.5, 1.0 * s), sigmaY=max(1.0, 7.0 * s))
    refl = cv2.resize(small, (gg.shape[1], gg.shape[0]), interpolation=cv2.INTER_LINEAR)
    ys = (np.arange(h, dtype=np.float32) + 0.5) / s
    fall = np.where(ys > c.hor + 10, np.exp(-(ys - c.hor - 10) / 260.0), 0.0).astype(np.float32)
    return img + refl * (fall * 0.55)[:, None, None]


def sky(fc, st, c):
    w, h, s = fc.w, fc.h, fc.s
    ys = (np.arange(h, dtype=np.float32) + 0.5) / s
    t = np.clip((ys - (c.hor - 900)) / 900.0, 0, 1) ** 1.4
    top, hor = np.array(col(SKY_TOP), np.float32), np.array(col(SKY_HOR), np.float32)
    colr = top[None, :] * (1 - t[:, None]) + hor[None, :] * t[:, None]
    tex = st["sky_tex"]
    th, tw = tex.shape
    oy = int(np.clip((1080 - c.hor) * 0.35 * s + (th - h) * 0.25, 0, th - h))
    ox = int(np.clip((c.x * 0.02 + (c.T - 21) * 6) * s + (tw - w) * 0.3, 0, tw - w))
    n = tex[oy:oy + h, ox:ox + w]
    # clouds: darker undersides, stronger high in the sky, fading into the horizon haze
    amp = (0.16 * (1 - t) + 0.03)[:, None]
    img = colr[:, None, :] * (1.0 + n[..., None] * amp[..., None])
    img *= np.array(dcol((1, 1, 1), c.D), np.float32)
    # diffuse sun glow behind the overcast (cool, upper right)
    yy = ys[:, None]
    xx = (np.arange(w, dtype=np.float32)[None, :] + 0.5) / s
    sx, sy = 1480.0 - c.x * 0.02, c.hor - 520.0
    r2 = ((xx - sx) ** 2 + ((yy - sy) * 1.6) ** 2) / (700.0 ** 2)
    img += (np.exp(-r2) * 0.06)[..., None] * np.array([0.9, 0.95, 1.0], np.float32)
    return img.astype(np.float32)


def ground(fc, st, c, img):
    w, h, s = fc.w, fc.h, fc.s
    j0 = int(math.ceil((c.hor + 0.5) * s))
    if j0 >= h:
        return img
    ys = (np.arange(j0, h, dtype=np.float32) + 0.5) / s
    dy = np.maximum(ys - c.hor, 0.35)
    d = c.f * c.h / dy                                 # (rows,)
    xs = (np.arange(w, dtype=np.float32) + 0.5) / s
    X = c.x + (xs[None, :] - W / 2) * d[:, None] / c.f
    Z = c.z + d[:, None] + 0 * X
    tex = st["ground_tex"]
    mx = ((X + 14000.0) / 28000.0 * 1023).astype(np.float32)
    my = ((Z + 1000.0) / 28000.0 * 1023).astype(np.float32)
    n = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fk = (1.0 - np.exp(-d / FOG_D)).astype(np.float32)[:, None]
    near = np.array(col(GROUND_NEAR), np.float32)
    mid = np.array(col(GROUND_MID), np.float32)
    fogc = np.array(col(SKY_HOR), np.float32) * 0.985
    tn = np.clip(d / 3000.0, 0, 1).astype(np.float32)[:, None, None]
    base = near * (1 - tn) + mid * tn
    base = base * (1.0 + 0.10 * n[..., None] * (1 - fk[..., None]))
    out = base * (1 - fk[..., None]) + fogc * fk[..., None]
    # horizon mist band
    mist = np.exp(-((ys - c.hor) / 26.0) ** 2).astype(np.float32)[:, None, None] * 0.5
    out = out * (1 - mist) + fogc * mist
    out *= np.array(dcol((1, 1, 1), c.D), np.float32)
    img = img.copy()
    img[j0:] = out
    # soften the horizon seam
    band = int(6 * s)
    if j0 - band > 0:
        img[j0 - band:j0 + band] = cv2.GaussianBlur(img[j0 - band:j0 + band], (0, 0), max(0.6, 1.2 * s))
    return img


def horizon_features(ctx, c, T, g=None):
    """far mesas + (from 33.6) the thin beige line of slop rising on the horizon"""
    fogc = col(SKY_HOR)
    for layer, (dist, amp, seed, k) in enumerate(((42000, 70, 3, 0.8), (26000, 45, 8, 0.66))):
        pts = []
        for i in range(0, 97):
            x = -40 + i * 21
            X = (x - W / 2) * dist / c.f + c.x
            hgt = max(0.0, fbm1(X / 2500.0, seed, 4)) * amp + 6
            y = c.hor + c.f * c.h / dist - hgt * (F / c.f) * (1 + 0 * layer)
            pts.append((x, y))
        pts += [(W + 40, c.hor + 40), (-40, c.hor + 40)]
        ctx.set_source_rgb(*dcol(mix("#6E7484", fogc, k), c.D))
        poly(ctx, pts)
        ctx.fill()
    # --- the slop line
    p = clamp((T - 33.62) / 1.2)
    if p <= 0:
        return
    hb = 4.0 + 78.0 * ease_in(p, 1.5)
    top = []
    for i in range(0, 121):
        x = -20 + i * 16.4
        bump = 0.55 + 0.45 * fbm1(x / 180.0 + T * 0.6, 77, 3)
        top.append((x, c.hor - hb * bump))
    body = top + [(W + 20, c.hor + 6), (-20, c.hor + 6)]
    ctx.set_source(lin_grad(0, c.hor - hb, 0, c.hor + 6, [(0, "#F0E2D2"), (0.35, "#D4C8B0"), (1, "#8A7F8F")]))
    poly(ctx, body)
    ctx.fill()
    # glinting lip: the crest is made of content cards catching light
    ctx.set_line_width(3.0)
    ctx.set_source_rgba(1.0, 0.95, 0.97, 1.0)
    poly(ctx, top, close=False)
    ctx.stroke()
    if g is not None:
        g.set_line_width(5)
        g.set_source_rgba(1.0, 0.75, 0.85, 0.6 * min(1.0, p * 3))
        poly(g, top, close=False)
        g.stroke()
    for i in range(160):
        x = hash01(i, 5) * W
        j = int((x + 20) / 16.4)
        y = top[min(j, len(top) - 1)][1] + hash01(i, 9) * hb * 0.7
        tw = 0.5 + 0.5 * math.sin(T * 9 + i * 1.7)
        cc = ("#FF5C8A", "#7FE8FF", "#FFFFFF", "#F4B6C8", "#3D7BF2")[i % 5]
        ctx.set_source_rgba(*col(cc), 0.9 * tw * p)
        ctx.rectangle(x, y, 2.4, 2.8)
        ctx.fill()


def grid_lines(ctx, c):
    """the IVG grid laid out across the plain (thin, fading into the fog)"""
    d_near = max(c.f * c.h / (H + 40 - c.hor), 3.0)
    ctx.set_line_width(1.1)
    step = 600.0
    z0 = math.ceil((c.z + d_near) / step) * step
    for Z in np.arange(z0, c.z + 16000, step):
        d = Z - c.z
        y = c.hor + c.f * c.h / d
        a = 0.16 * (1 - fogk(d)) ** 2
        ctx.set_source_rgba(0.85, 0.88, 0.95, a)
        ctx.move_to(0, y)
        ctx.line_to(W, y)
        ctx.stroke()
    for X in np.arange(-15000, 15001, step):
        d1, d2 = d_near, 16000.0
        x1, y1 = W / 2 + c.f * (X - c.x) / d1, c.hor + c.f * c.h / d1
        x2, y2 = W / 2 + c.f * (X - c.x) / d2, c.hor + c.f * c.h / d2
        gr = cairo.LinearGradient(x1, y1, x2, y2)
        gr.add_color_stop_rgba(0, 0.85, 0.88, 0.95, 0.16 * (1 - fogk(d1)) ** 2)
        gr.add_color_stop_rgba(1, 0.85, 0.88, 0.95, 0.0)
        ctx.set_source(gr)
        ctx.move_to(x1, y1)
        ctx.line_to(x2, y2)
        ctx.stroke()


def _clip_seg(c, p0, p1, dmin):
    """clip plan segment p0->p1 to depth >= dmin; returns (q0, q1) or None"""
    d0, d1 = p0[1] - c.z, p1[1] - c.z
    if d0 < dmin and d1 < dmin:
        return None
    if d1 < dmin:
        t = (dmin - d0) / (d1 - d0)
        p1 = (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
    if d0 < dmin:
        t = (dmin - d1) / (d0 - d1)
        p0 = (p1[0] + (p0[0] - p1[0]) * t, p1[1] + (p0[1] - p1[1]) * t)
    return p0, p1


def roads(ctx, c):
    dmin = max(c.f * c.h / (H + 60 - c.hor), 4.0)
    for (p0, p1, mi) in ROADS:
        q = _clip_seg(c, p0, p1, dmin)
        if q is None:
            continue
        a, b = q
        dx, dz = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dz)
        nx, nz = -dz / L, dx / L
        half = 24.0
        pts = []
        for (px, pz) in ((a[0] + nx * half, a[1] + nz * half), (b[0] + nx * half, b[1] + nz * half),
                         (b[0] - nx * half, b[1] - nz * half), (a[0] - nx * half, a[1] - nz * half)):
            sx, sy, _, _ = c.proj(px, 0, pz)
            pts.append((sx, sy))
        da, db = a[1] - c.z, b[1] - c.z
        ga = dcol(mix("#6E7484", SKY_HOR, fogk(da)), c.D)
        gb = dcol(mix("#6E7484", SKY_HOR, fogk(db)), c.D)
        (ax, ay), (bx, by) = c.proj(a[0], 0, a[1])[:2], c.proj(b[0], 0, b[1])[:2]
        ctx.set_source(lin_grad(ax, ay, bx, by, [(0, ga), (1, gb)]))
        poly(ctx, pts)
        ctx.fill()
        # the dense far crowd as a darker stream down the road's middle
        pts2 = []
        for (px, pz) in ((a[0] + nx * 17, a[1] + nz * 17), (b[0] + nx * 17, b[1] + nz * 17),
                         (b[0] - nx * 17, b[1] - nz * 17), (a[0] - nx * 17, a[1] - nz * 17)):
            sx, sy, _, _ = c.proj(px, 0, pz)
            pts2.append((sx, sy))
        ctx.set_source(lin_grad(ax, ay, bx, by, [(0, dcol(mix("#2E303C", SKY_HOR, fogk(da)), c.D), 0.6),
                                                 (1, dcol(mix("#2E303C", SKY_HOR, fogk(db)), c.D), 0.6)]))
        poly(ctx, pts2)
        ctx.fill()


def crowd_points(c, T):
    """all individually visible marchers: arrays of (sx, sy_feet, hpx, d, phase)"""
    out = []
    ph_all = None
    sp, v = 3.4, 1.35
    files = np.arange(-16.0, 16.1, 4.0)
    for ri, (p0, p1, mi) in enumerate(ROADS):
        dx, dz = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(dx, dz)
        ux, uz = dx / L, dz / L
        nx, nz = -uz, ux
        _, Te = lockstep_phase(T, 0)
        k = np.arange(int(L / sp))
        u = (k * sp - v * Te) % L                      # distance from the gate; decreasing = marching in
        for fi, fo in enumerate(files):
            jit = (np.sin(k * 12.9898 + fi * 78.233 + ri) * 43758.5453) % 1.0
            X = p0[0] + ux * u + nx * fo + (jit - 0.5) * 0.8
            Z = p0[1] + uz * u + nz * fo
            d = Z - c.z
            ok = (d > 30) & (d < 1900)
            if not ok.any():
                continue
            X, Z, d, jj = X[ok], Z[ok], d[ok], jit[ok]
            sx = W / 2 + c.f * (X - c.x) / d
            sy = c.hor + c.f * c.h / d
            hp = 1.75 * (0.93 + 0.14 * jj) * c.f / d
            vis = (sx > -20) & (sx < W + 20) & (sy > c.hor - 5) & (sy - hp < H + 5) & (sy < H + 40)
            if vis.any():
                out.append(np.stack([sx[vis], sy[vis], hp[vis], d[vis], jj[vis]], 1))
    if not out:
        return np.zeros((0, 5))
    P = np.concatenate(out)
    return P[np.argsort(-P[:, 3])]


def draw_people(ctx, P, dlo, dhi, T, c):
    """batched tiny marchers with depth in [dlo, dhi)"""
    if len(P) == 0:
        return
    sel = P[(P[:, 3] >= dlo) & (P[:, 3] < dhi)]
    if len(sel) == 0:
        return
    # colour by fog buckets
    buckets = np.digitize(sel[:, 3], [300, 600, 1000, 1500])
    for bi in range(5):
        S_ = sel[buckets == bi]
        if len(S_) == 0:
            continue
        dm = float(np.median(S_[:, 3]))
        ctx.set_source_rgb(*dcol(mix("#2A2C38", SKY_HOR, fogk(dm) * 1.2), c.D))
        for (x, y, hp, d, jj) in S_:
            if hp >= 11.0:
                continue
            ph, _ = lockstep_phase(T, jj * 7.0)
            bob = abs(math.sin(ph * math.pi)) * hp * 0.07
            if hp < 3.0:
                ctx.rectangle(x - hp * 0.2, y - hp - bob, hp * 0.4, hp)
            else:
                r = hp * 0.12
                ctx.new_sub_path()
                ctx.arc(x, y - hp + r - bob, r, 0, 2 * math.pi)
                sw = math.sin(ph * math.pi) * hp * 0.12
                ctx.rectangle(x - hp * 0.16, y - hp + 2 * r - bob, hp * 0.32, hp * 0.52)
                ctx.rectangle(x - hp * 0.13 + sw * 0.5, y - hp * 0.5, hp * 0.11, hp * 0.5)
                ctx.rectangle(x + hp * 0.02 - sw * 0.5, y - hp * 0.5, hp * 0.11, hp * 0.5)
        ctx.fill()


def near_people(ctx, st, P, T, c):
    """road marchers close enough to be real characters (grey silhouettes), far to near"""
    if len(P) == 0:
        return
    sel = P[P[:, 2] >= 11.0]
    if len(sel) > 170:
        sel = sel[-170:]
    for i, (x, y, hp, d, jj) in enumerate(sel):
        ph, _ = lockstep_phase(T, jj * 7.0)
        colr = dcol(mix("#2E303C", SKY_HOR, fogk(d) * 1.2 + 0.08), c.D)
        ch = st["walkers"][int(jj * 97) % len(st["walkers"])]
        ch.draw(ctx, x, y, hp, "walk", T, facing=0, silhouette=colr, phase=ph * 0.5, tint=(colr, 1.0))


def draw_monument(ctx, g, c, m, T):
    d = m["Z"] - c.z
    sx, sy, k, _ = c.proj(m["X"], 0, m["Z"])
    sc = k * m["m"]
    fk = fogk(d)
    M = MN.Look(dcol(SKY_HOR, c.D), fk * 0.72, exposure=c.D)
    t0 = {"nation": T_N, "faith": T_F, "philo": T_P}[m["name"]]
    lit = lit_curve(T, t0)
    ph, _ = lockstep_phase(T, 0)
    for cc in (ctx, g):
        cc.save()
        cc.translate(sx, sy)
        cc.scale(sc, sc)
    kw = {}
    if m["name"] in ("nation", "faith"):
        kw["lockstep"] = ph if T > 27.9 else None
    m["fn"](ctx, g, M, T, lit, **kw)
    for cc in (ctx, g):
        cc.restore()


def antenna_screen(c, m, T):
    d = m["Z"] - c.z
    sx, sy, k, _ = c.proj(m["X"], 0, m["Z"])
    sc = k * m["m"]
    tipy = {"nation": -1162, "faith": -1010, "philo": -1150}[m["name"]]
    return sx, sy + tipy * sc, sc, d


def rings(ctx, g, c, T):
    """expanding broadcast rings from each antenna (horizontal circles seen from below)"""
    for m in MONS:
        sx, sy, sc, d = antenna_screen(c, m, T)
        Ytip = 1100 * m["m"]
        tilt = max(0.12, (Ytip - c.h) / d)
        lr, lg, lb = col(m["light"])
        for i, te in enumerate(RING_T):
            age = T - te - {"nation": 0.0, "faith": 0.05, "philo": 0.1}[m["name"]]
            if age < 0 or age > 2.6:
                continue
            p = age / 2.6
            R = (120 + 3200 * ease_out(p, 2.2)) * c.f / d
            a = (1 - p) ** 2
            for gg, lw, al in ((g, 16, 0.35), (g, 5, 0.9), (ctx, 1.6, 0.8)):
                gg.save()
                gg.translate(sx, sy)
                gg.scale(1, tilt)
                gg.new_path()
                gg.arc(0, 0, R, 0, 2 * math.pi)
                gg.restore()
                gg.set_line_width(lw * (1 - 0.5 * p))
                gg.set_source_rgba(lr, lg, lb, a * al)
                gg.stroke()
            # the pulse flash at the tip
            if age < 0.25:
                g.set_source_rgba(lr, lg, lb, (1 - age / 0.25) * 0.9)
                circle(g, sx, sy, 30 * (1 + age * 4))
                g.fill()


def procession(ctx, st, c, T):
    """two rows of grey marchers crossing close to camera, right -> left (backlit by fog)"""
    for row, (Z, off, spacing) in enumerate(((58.0, 0.3, 1.7), (30.0, 0.0, 1.45), (19.0, 0.7, 1.9))):
        d = Z - c.z
        if d < 3:
            continue
        fk = fogk(d) * 3.0 + 0.12 - row * 0.03
        colr = mix("#2E303C", FOG0, clamp(fk))
        half = (W / 2 + 200) * d / c.f
        X0 = c.x - half
        n = int(2 * half / spacing) + 2
        ph0, Te = lockstep_phase(T, 0)
        for i in range(n):
            X = X0 + ((i * spacing - Te * 1.35 + off * 7) % (2 * half))
            sx, sy, k, _ = c.proj(X, 0, Z)
            hp = 1.75 * k * (0.95 + 0.1 * hash01(i + row * 100, 3))
            if sy - hp > H + 10:
                continue
            ch = st["walkers"][(i + row) % len(st["walkers"])]
            ch.draw(ctx, sx, sy, hp, "walk", T + hash01(i, 11) * 0.3, facing=-1, silhouette=colr,
                    phase=(ph0 + hash01(i, 7) * 0.6) * 0.5, speed=1.0, tint=(colr, 1.0))

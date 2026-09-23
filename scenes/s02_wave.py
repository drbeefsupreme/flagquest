"""s02 shot B (34.8-40.6): THE SLOP WAVE.

A Hokusai curl whose body is a flowing particle field of content cards. The wave is a ribbon around a spine
(a rising Bezier that becomes a logarithmic spiral). Cards live in flow coordinates (sigma, n): sigma advances
with time (ds_sigma = ds / thickness), n runs across the ribbon. Because card size follows ribbon thickness, cards
shrink as they wind into the spiral: an infinite scroll. Foam claws (bubbles + hearts) grasp at the memeplexes,
which sit small on the horizon like Mount Fuji. 39.2 slop_crash: the lip slams down, cards burst, the
memeplexes crack / topple / sink, beige flood.
"""
import math

import cairo
import cv2
import numpy as np

from vx import (W, H, col, mix, shade, set_color, circle, ellipse, poly, smooth, lin_grad, rad_grad, blur, clamp,
                lerp, seg, smoothstep, smootherstep, ease_in_out, ease_out, ease_in, ease_out_back, hash01, noise1,
                fbm1, fbm2, shake, Canvas, gradient_v)
from scenes import s02_monuments as MN
from scenes.s02_cards import Atlas, _heart

T0 = 34.8
T_CRASH = 39.2
T_END = 40.6
SEA_Y = 832.0

SKY_TOP = "#6F6A80"
SKY_MID = "#A69FB2"
SKY_HOR = "#D4CACF"
BODY_D = "#3B3150"
BODY_M = "#6E5E7E"
SEA_FAR = "#CDBFB6"
SEA_NEAR = "#7E7088"

MON_POS = (("nation", 1110.0, 0.31), ("faith", 1335.0, 0.26), ("philo", 1545.0, 0.28))


# =============================================================================================== setup
def setup(S, st):
    st["atlas"] = Atlas(n_variants=2)
    rng = np.random.default_rng(42)
    # population A: the hero cards (5 across) flowing through the whole ribbon + spiral
    nA_rows = 420
    KA = 5
    A = []
    for j in range(nA_rows):
        for i in range(KA):
            n = -0.82 + 1.64 * (i + 0.5 + rng.uniform(-0.3, 0.3)) / KA
            A.append((j * 0.23 + rng.uniform(-0.06, 0.06), n, rng.integers(0, 10 ** 6), rng.uniform(0.85, 1.2),
                      rng.uniform(-0.45, 0.45), rng.uniform(0, 6.28), rng.uniform(0.3, 1.4) * (1 if rng.random() < 0.5 else -1)))
    st["popA"] = np.array(A, np.float64)
    # population B: dense body cards (13 across) living in the rising part only
    KB = 13
    B = []
    for j in range(80):
        for i in range(KB):
            n = -0.95 + 1.9 * (i + 0.5 + rng.uniform(-0.35, 0.35)) / KB
            B.append((j * 0.075 + rng.uniform(-0.03, 0.03), n, rng.integers(0, 10 ** 6), rng.uniform(0.8, 1.25),
                      rng.uniform(-0.5, 0.5), rng.uniform(0, 6.28), rng.uniform(0.3, 1.2)))
    st["popB"] = np.array(B, np.float64)
    # burst particles for the crash
    nb = 900
    ang = rng.uniform(-math.pi * 0.97, -math.pi * 0.03, nb)
    spd = rng.uniform(250, 1700, nb) * (0.35 + 0.65 * rng.random(nb) ** 0.5)
    st["burst"] = dict(ang=ang, spd=spd, size=rng.uniform(8, 46, nb), idx=rng.integers(0, 10 ** 6, nb),
                       spin=rng.uniform(-9, 9, nb), ox=rng.normal(0, 90, nb), oy=rng.normal(0, 30, nb),
                       delay=rng.uniform(0, 0.12, nb))
    # sea texture
    s = S.s
    w = int(round(W * s))
    yy, xx = np.mgrid[0:int(300 * s) + 2, 0:int(w * 1.5)].astype(np.float64)
    st["sea_tex"] = fbm2(xx / (60 * s), yy / (9 * s), seed=4, octaves=4).astype(np.float32)
    events = [
        dict(t=T0, kind="whoosh", strength=1.0, pan=-0.3, desc="CUT: the slop wave towers over everything (Hokusai curl)"),
        dict(t=T0 + 0.05, kind="roar", strength=0.8, pan=-0.3, desc="wave body roar begins, builds with the rise (card rustle + glitch)"),
        dict(t=38.5, kind="swell", strength=0.9, pan=0.2, desc="wave at full height, claws reach over the memeplexes; shake peaks"),
        dict(t=38.8, kind="whoosh", strength=0.9, pan=0.4, desc="the lip begins to fall"),
        dict(t=T_CRASH, kind="impact", strength=1.0, pan=0.45, desc="SLOP_CRASH: the lip slams onto the memeplexes, card explosion, flash"),
        dict(t=39.24, kind="crack", strength=0.9, pan=0.3, desc="NATION tower cracks"),
        dict(t=39.3, kind="crack", strength=0.8, pan=0.5, desc="FAITH temple cracks, halo drops"),
        dict(t=39.36, kind="crack", strength=0.8, pan=0.75, desc="PHILOSOPHY bust cracks"),
        dict(t=39.45, kind="topple", strength=0.8, pan=0.35, desc="crown flies off the NATION, tower topples right"),
        dict(t=39.55, kind="topple", strength=0.7, pan=0.75, desc="marble head topples"),
        dict(t=39.5, kind="flood", strength=0.9, pan=0.2, desc="beige flood rises over the plain"),
        dict(t=39.9, kind="sink", strength=0.6, pan=0.5, desc="memeplexes sink into the slop (gurgle)"),
        dict(t=40.0, kind="rain", strength=0.4, pan=0.0, desc="cards rain back down onto the flood"),
    ]
    for i, tg in enumerate(GLITCH_T[:-1]):
        events.append(dict(t=tg, kind="glitch", strength=0.35 + 0.1 * i, pan=0.0,
                           desc="glitch burst: picture tears into RGB-split slices (~0.15 s)"))
    events.append(dict(t=T_CRASH + 0.35, kind="punch_in", strength=0.5, pan=0.4,
                       desc="camera punches in on the toppling memeplexes"))
    st["wave_events"] = events


# =============================================================================================== geometry
def growth(T):
    return 0.5 + 0.5 * ease_out(clamp((T - T0) / 3.6), 1.8)


def fall(T):
    return ease_in(clamp((T - 38.7) / (T_CRASH - 38.7)), 2.4)


def collapse(T):
    return smootherstep(T_CRASH, T_CRASH + 0.55, T)


PIVOT = np.array([640.0, 1180.0])      # the wave grows from / falls about its root


def geometry(T):
    """spine + thickness of the wave in screen space (design units).
    Full-grown layout (g=1): root below the frame at (250,1300); a steep wall rising to the crest; the lip runs
    right along the top to E=(1000,150) and curls down around Cs=(1000,400) -- first as a slow Hokusai hook,
    then tightening into an infinite-scroll coil. The hollow under the lip stays open."""
    g = growth(T)
    q = fall(T)
    cq = collapse(T)
    k = math.log(1 / 0.5) / (2 * math.pi)
    r0 = 255.0
    E = np.array([1000.0 + 18 * math.sin(T * 1.3), 150.0])
    B = np.array([250.0, 1320.0])
    dirE = np.array([1.0, 0.06]) / math.hypot(1, 0.06)
    C1 = np.array([440.0, 720.0])
    C2 = E - dirE * 470.0
    u = np.linspace(0, 1, 170)[:, None]
    rise = (1 - u) ** 3 * B + 3 * (1 - u) ** 2 * u * C1 + 3 * (1 - u) * u ** 2 * C2 + u ** 3 * E
    Te = 0.62 * r0
    Tb = 2300.0
    thick_r = Te + (Tb - Te) * (1 - u[:, 0]) ** 4.5 + 60 * np.sin(u[:, 0] * math.pi) ** 2
    Cs = E + np.array([0.0, r0])
    turns = 3.3
    ph = np.linspace(-math.pi / 2, -math.pi / 2 + 2 * math.pi * turns, 800)[1:]
    pp = ph + math.pi / 2
    # a slow-closing lip (the Hokusai hook) that tightens into an infinite-scroll coil
    k1, k2, phi1 = 0.075, math.log(1 / 0.33) / (2 * math.pi), 1.05 * math.pi
    sm = np.clip((pp - (phi1 - 0.8)) / 1.6, 0, 1)
    kk = k1 + (k2 - k1) * sm * sm * (3 - 2 * sm)
    dph = np.diff(np.concatenate([[0.0], pp]))
    rr = r0 * np.exp(-np.cumsum(kk * dph))
    spiral = Cs + rr[:, None] * np.stack([np.cos(ph), np.sin(ph)], 1)
    tk = np.clip((pp - phi1) / (2 * math.pi), 0, 1)
    thick_s = rr * (0.62 - 0.08 * np.clip(pp / phi1, 0, 1) + 0.12 * tk * tk * (3 - 2 * tk))
    P = np.concatenate([rise, spiral])
    Tk = np.concatenate([thick_r, thick_s])
    # growth: scale about the root; fall: rotate forward about the root and lurch right
    sc = lerp(0.6, 1.0, g)
    ang = math.radians(24.0) * q
    ca, sa = math.cos(ang), math.sin(ang)
    rel = (P - PIVOT) * sc
    P = PIVOT + np.stack([rel[:, 0] * ca - rel[:, 1] * sa, rel[:, 0] * sa + rel[:, 1] * ca], 1) + np.array([90.0, 30.0]) * q
    Tk = Tk * sc
    relc = (Cs - PIVOT) * sc
    Cs = PIVOT + np.array([relc[0] * ca - relc[1] * sa, relc[0] * sa + relc[1] * ca]) + np.array([90.0, 30.0]) * q
    r0 = r0 * sc
    E = P[len(rise) - 1].copy()
    # collapse after the crash: everything sinks into the flood, thins out
    if cq > 0:
        # the lip has exploded: only the body remains, slumping into the flood
        nr = len(rise)
        keep = nr + int((len(P) - nr) * (1 - min(1.0, cq * 3.0)))
        P, Tk, pp = P[:keep].copy(), Tk[:keep].copy(), pp[:max(0, keep - nr)]
        P[:, 1] = P[:, 1] + (SEA_Y + 200 - P[:, 1]) * cq ** 1.1
        P[:, 0] = P[:, 0] + 160 * cq
    d = np.gradient(P, axis=0)
    tl = np.hypot(d[:, 0], d[:, 1]) + 1e-9
    tang = d / tl[:, None]
    N = np.stack([-tang[:, 1], tang[:, 0]], 1)
    ds = tl
    sig = np.concatenate([[0], np.cumsum((ds[1:] + ds[:-1]) * 0.5 / (0.5 * (Tk[1:] + Tk[:-1])))])
    iE = len(rise) - 1
    turn = np.concatenate([np.zeros(len(rise)), pp / (2 * math.pi)])
    return dict(P=P, N=N, tang=tang, Tk=Tk, sig=sig, sigE=sig[iE], iE=iE, Cs=Cs, r0=r0, E=E, g=g, q=q, cq=cq,
                turn=turn, k=k)


def at_sigma(geo, s):
    sig = geo["sig"]
    idx = np.interp(s, sig, np.arange(len(sig)))
    i0 = np.clip(idx.astype(int), 0, len(sig) - 2)
    f = (idx - i0)[:, None]
    P = geo["P"][i0] * (1 - f) + geo["P"][i0 + 1] * f
    N = geo["N"][i0] * (1 - f) + geo["N"][i0 + 1] * f
    Tk = geo["Tk"][i0] * (1 - f[:, 0]) + geo["Tk"][i0 + 1] * f[:, 0]
    tg = geo["tang"][i0] * (1 - f) + geo["tang"][i0 + 1] * f
    turn = geo["turn"][i0] * (1 - f[:, 0]) + geo["turn"][i0 + 1] * f[:, 0]
    return P, N, Tk, tg, turn


# =============================================================================================== render
def render(fc, st):
    T = fc.T
    s = fc.s
    geo = geometry(T)
    atlas = st["atlas"]
    # camera: slow push + rising shake; the crash kicks it, then a punch-in on the dying memeplexes
    rise = clamp((T - T0) / (T_CRASH - T0))
    amp = 2.0 + 16.0 * rise ** 2.2
    if T > T_CRASH:
        amp = 30.0 * math.exp(-(T - T_CRASH) * 2.8) + 3.0
    sx_, sy_ = shake(T, amp, 11.0, seed=3)
    pz = ease_out(clamp((T - T_CRASH - 0.03) / 0.7), 3)
    zoom = (1.0 + 0.06 * ease_in_out(rise)) * (1 + 0.42 * pz)
    zx, zy = lerp(W / 2, 1320.0, pz), lerp(H / 2, 700.0, pz)
    img = sky(fc, st, T)
    cv = Canvas(fc)
    gl = Canvas(fc)
    ctx, g = cv.ctx, gl.ctx
    for c in (ctx, g):
        c.translate(W / 2 + sx_, H / 2 + sy_)
        c.scale(zoom, zoom)
        c.translate(-zx, -zy)
    flood = flood_level(T)
    if T >= T_CRASH - 0.02:
        draw_burst(ctx, g, atlas, st, T, s, back=True)
    draw_memeplexes(ctx, g, T, flood)
    draw_far_sea(ctx, atlas, T, flood, s)
    if geo["cq"] < 0.999:
        draw_wave(ctx, g, atlas, st, geo, T, s)
        draw_claws(ctx, geo, T, atlas, s)
        draw_spray(ctx, atlas, geo, T, s)
    draw_near_sea(ctx, atlas, T, flood, s, geo)
    if T >= T_CRASH - 0.02:
        draw_burst(ctx, g, atlas, st, T, s, back=False)
    img = cv.over(img)
    gg = gl.rgba()[..., :3]
    img = img + blur(gg, 3 * s) * 0.6 + blur(gg, 18 * s) * 0.45
    # crash flash (short)
    if T >= T_CRASH:
        fl = math.exp(-(T - T_CRASH) * 16.0)
        img = img + np.array([1.0, 0.92, 0.97], np.float32) * (0.5 * fl)
    img = glitch(img, fc, T)
    img = ca_rc(img, fc, ca_amount(T))
    return np.clip(img, 0, 4).astype(np.float32)


def ca_amount(T):
    rise = clamp((T - T0) / (T_CRASH - T0))
    ca = 1.5 + 7.0 * rise ** 2
    if T > T_CRASH:
        ca = 3.0 + 16.0 * math.exp(-(T - T_CRASH) * 3.0)
    return ca + 6.0 * glitch_amount(T)


def ca_rc(img, fc, amt):
    """radial chromatic aberration that splits RED from CYAN (G+B travel together) so the fringes are
    red/cyan -- never yellow (yellow belongs to the Flags)"""
    if amt <= 0.05:
        return img
    h, w = img.shape[:2]
    k = 1 + 2 * amt * fc.s / w
    cx, cy = w / 2, h / 2
    M = np.array([[k, 0, cx * (1 - k)], [0, k, cy * (1 - k)]], np.float32)
    out = img.copy()
    out[..., 0] = cv2.warpAffine(np.ascontiguousarray(img[..., 0]), M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return out


def post(fc, st):
    T = fc.T
    rise = clamp((T - T0) / (T_CRASH - T0))
    return dict(ca=0.0, bloom=0.45, vignette=0.32, saturation=1.05)


# ----------------------------------------------------------------------------------------------- sky
def sky(fc, st, T):
    img = gradient_v(fc, [(0, SKY_TOP), (0.45, SKY_MID), (SEA_Y / H, SKY_HOR), (1, SKY_HOR)])
    tex = st["sky_tex"]
    h, w = fc.h, fc.w
    oy = int((tex.shape[0] - h) * 0.2)
    ox = int(((T - T0) * 30 * fc.s) % max(1, tex.shape[1] - w))
    n = tex[oy:oy + h, ox:ox + w]
    ys = (np.arange(h, dtype=np.float32) + 0.5) / fc.s
    amp = np.clip(1 - ys / SEA_Y, 0, 1)[:, None] * 0.22
    xs = (np.arange(w, dtype=np.float32) + 0.5) / fc.s
    left = np.clip(1.2 - xs / 1100, 0.2, 1)[None, :]           # the storm is darker behind the wave
    img = img * (1 + n[..., None] * (amp * left)[..., None]) * (1 - 0.18 * (left * np.clip(1 - ys / 900, 0, 1)[:, None]))[..., None]
    return img.astype(np.float32)


# ----------------------------------------------------------------------------------------------- memeplexes
def flood_level(T):
    """y of the flood surface near the horizon (rises after the crash)"""
    return SEA_Y - 70.0 * smootherstep(T_CRASH + 0.05, T_CRASH + 1.3, T)


def draw_memeplexes(ctx, g, T, flood):
    M = MN.Look(SKY_HOR, 0.2, exposure=0.8)
    rt = [34.85, 35.5, 36.1, 36.8, 37.4, 38.0]
    for i, (kind, x, sc) in enumerate(MON_POS):
        hit = T_CRASH + 0.03 + i * 0.06
        q = clamp((T - hit) / 1.6)
        crack = smoothstep(hit - 0.02, hit + 0.3, T)
        sink = 150.0 * ease_in(q, 2.2)
        # lights: still on (flickering in the wave's shadow), die at the crash
        lit = 0.85 * (0.6 + 0.4 * math.sin(T * 23 + i * 2) * math.sin(T * 7.1 + i)) if T < hit else \
            max(0.0, 0.8 - (T - hit) * 5)
        topple = {"nation": 0.62, "faith": -0.22, "philo": -0.75}[kind] * ease_in_out(clamp(q * 1.25), 2.2)
        ctx.save()
        g.save()
        for c in (ctx, g):
            c.translate(x, SEA_Y + 4 + sink)
            c.rotate(topple)
            c.scale(sc, sc)
        if kind == "nation":
            tq = clamp((T - hit - 0.05) / 1.1)
            top = (160 * tq, -300 * tq + 1300 * tq * tq, 1.6 * tq)
            MN.draw_nation(ctx, g, M, T, lit, crack=crack, top=top)
        elif kind == "faith":
            tq = clamp((T - hit) / 1.1)
            MN.draw_faith(ctx, g, M, T, lit, crack=crack, top=(90 * tq, 700 * tq * tq, -0.7 * tq),
                          halo=(-200 * tq, 900 * tq * tq, 0.9 * tq))
        else:
            tq = clamp((T - hit) / 1.0)
            MN.draw_philosophy(ctx, g, M, T, lit, crack=crack, top=(-120 * tq, 500 * tq * tq, -1.3 * tq))
        ctx.restore()
        g.restore()
        # broadcast rings keep pulsing, oblivious, until the crash
        if T < hit:
            tipy = SEA_Y - {"nation": 1160, "faith": 1010, "philo": 1150}[kind] * sc
            lr, lgc, lb = col({"nation": MN.LIGHT_N, "faith": MN.LIGHT_F, "philo": MN.LIGHT_P}[kind])
            for te in rt:
                age = T - te - i * 0.07
                if 0 <= age < 1.6:
                    p = age / 1.6
                    R = 20 + 420 * ease_out(p, 2)
                    for c, lw, al in ((g, 7, 0.4), (ctx, 1.4, 0.7)):
                        c.save()
                        c.translate(x, tipy)
                        c.scale(1, 0.3)
                        c.new_path()
                        c.arc(0, 0, R, 0, 2 * math.pi)
                        c.restore()
                        c.set_line_width(lw)
                        c.set_source_rgba(lr, lgc, lb, (1 - p) ** 2 * al)
                        c.stroke()
    # tiny crowds on the shore, frozen, looking up at the wave
    if T < T_CRASH + 0.05:
        ctx.set_source_rgba(*col("#3D3848"), 0.9)
        for i in range(140):
            x = 1080 + hash01(i, 1) * 820
            y = SEA_Y + 2 + hash01(i, 2) * 10
            hp = 5.0 + hash01(i, 3) * 2.5 + (y - SEA_Y) * 0.4
            ctx.rectangle(x - hp * 0.17, y - hp * 0.78, hp * 0.34, hp * 0.78)
            ctx.new_sub_path()
            ctx.arc(x, y - hp * 0.88, hp * 0.13, 0, 2 * math.pi)
        ctx.fill()


# ----------------------------------------------------------------------------------------------- sea
def draw_far_sea(ctx, atlas, T, flood, s):
    """the slop sea from the horizon to the foreground (drawn under the wave)"""
    top = min(flood, SEA_Y)
    ctx.set_source(lin_grad(0, top, 0, H + 60, [(0, SEA_FAR), (0.35, "#B3A5A6"), (1, SEA_NEAR)]))
    pts = [(x, top + 3 * math.sin(x / 60 + T * 3) * (1 if T > T_CRASH else 0.3)) for x in range(-100, W + 101, 20)]
    poly(ctx, pts + [(W + 100, H + 100), (-100, H + 100)])
    ctx.fill()
    # swell lines receding in perspective
    for j in range(16):
        dz = (j + (T * 0.9) % 1.0) / 16.0
        y = top + 4 + (H + 60 - top) * dz ** 1.8
        a = 0.35 * dz
        ctx.set_line_width(1.0 + 3 * dz)
        ctx.set_source_rgba(1, 0.96, 0.97, a)
        ctx.move_to(-50, y)
        for x in range(-50, W + 60, 60):
            ctx.line_to(x, y + math.sin(x / (90 + 60 * dz) + T * 2 + j) * 4 * dz)
        ctx.stroke()
    # Hokusai wavelets: little white-lipped crests scattered across the slop in perspective
    for i in range(150):
        dz = hash01(i, 81)
        y = top + 5 + (H + 40 - top) * dz ** 1.5
        x = (hash01(i, 82) * (W + 300) - 150 + T * (20 + 50 * dz)) % (W + 300) - 150
        L = 8 + 60 * dz ** 1.4
        ph = (T * 1.3 + hash01(i, 83) * 6.28) % 6.28
        hgt = L * 0.35 * (0.6 + 0.4 * math.sin(ph))
        ctx.set_line_width(0.8 + 2.4 * dz)
        ctx.set_source_rgba(*col("#7E6F86"), 0.35 + 0.3 * dz)
        ctx.move_to(x - L, y)
        ctx.curve_to(x - L * 0.3, y - hgt * 0.2, x - L * 0.1, y - hgt, x + L * 0.3, y - hgt * 0.8)
        ctx.stroke()
        ctx.set_source_rgba(1, 0.97, 0.98, 0.55 + 0.35 * dz)
        ctx.move_to(x - L * 0.05, y - hgt * 0.95)
        ctx.curve_to(x + L * 0.2, y - hgt, x + L * 0.35, y - hgt * 0.7, x + L * 0.3, y - hgt * 0.45)
        ctx.stroke()
    # floating content cards, flattened by perspective
    for i in range(260):
        dz = hash01(i, 11)
        y = top + 6 + (H + 40 - top) * dz ** 1.6
        x = (hash01(i, 12) * (W + 400) - 200 + T * (30 + 40 * dz) * (1 if i % 2 else -0.4)) % (W + 400) - 200
        size = 4 + 34 * dz ** 1.6
        bob = math.sin(T * 3 + i) * 3 * dz
        atlas.draw(ctx, i * 7 + 3, x, y + bob, size, math.sin(i) * 0.4, sx=1.0, shade_k=0.25 + 0.3 * (1 - dz),
                   alpha=0.9, px_scale=s)


def draw_near_sea(ctx, atlas, T, flood, s, geo):
    """foreground swells: cover the wave's root; after the crash the flood churns up"""
    cq = geo["cq"]
    base = H - 120 + 60 * cq
    lift = 110 * smootherstep(T_CRASH + 0.05, T_CRASH + 1.0, T)
    for layer in range(3):
        y0 = base - lift * (1 - 0.3 * layer) + layer * 70
        pts = []
        for x in range(-100, W + 121, 24):
            yy = y0 + 26 * math.sin(x / (170 - layer * 30) + T * (2.2 + layer) + layer * 2) + 18 * fbm1(x / 300 + T, layer, 3)
            pts.append((x, yy))
        ctx.set_source(lin_grad(0, y0 - 40, 0, H + 40, [(0, mix(BODY_M, SEA_FAR, 0.45 - 0.12 * layer)), (1, BODY_D)]))
        poly(ctx, pts + [(W + 120, H + 200), (-100, H + 200)])
        ctx.fill()
        # cards riding the swell
        n = 70 + layer * 30
        for i in range(n):
            u = ((hash01(i + layer * 999, 5) + T * (0.05 + 0.03 * layer)) % 1.0)
            x = -80 + u * (W + 200)
            j = int((x + 100) / 24)
            j = max(0, min(len(pts) - 1, j))
            yy = pts[j][1] + 8 + hash01(i, 6) * 90
            size = 22 + 30 * hash01(i, 7) + layer * 10
            atlas.draw(ctx, i * 13 + layer, x, yy, size, math.sin(T * 2 + i) * 0.5, sx=math.cos(T * 1.5 + i),
                       shade_k=0.35 + 0.2 * hash01(i, 9), px_scale=s)
        # foam crest line with bubbles
        ctx.set_line_width(4 + layer * 1.5)
        ctx.set_source_rgba(1, 0.97, 0.98, 0.85)
        poly(ctx, pts, close=False)
        ctx.stroke()
        for i in range(0, len(pts), 2):
            x, yy = pts[i]
            r = 3 + 5 * hash01(i + layer * 100, 8)
            ctx.set_source_rgba(1, 0.97, 0.98, 0.9)
            circle(ctx, x + 6 * math.sin(T * 4 + i), yy - r * 0.5, r)
            ctx.fill()


# ----------------------------------------------------------------------------------------------- wave
def _ribbon(ctx, geo, i0, i1, scale=1.0):
    P, N, Tk = geo["P"][i0:i1], geo["N"][i0:i1], geo["Tk"][i0:i1] * 0.5 * scale
    L = P - N * Tk[:, None]
    R = P + N * Tk[:, None]
    pts = np.concatenate([L, R[::-1]])
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()


def draw_wave(ctx, g, atlas, st, geo, T, s):
    P, N, Tk, sig = geo["P"], geo["N"], geo["Tk"], geo["sig"]
    Cs, r0 = geo["Cs"], geo["r0"]
    fade = 1 - geo["cq"]
    # the hollow of the barrel: a dark vortex
    ctx.set_source(rad_grad(Cs[0], Cs[1], r0 * 0.1, r0 * 1.0, [(0, "#2E2542", 0.75 * fade), (0.55, "#4A3F5E", 0.45 * fade), (1, "#4A3F5E", 0.0)]))
    circle(ctx, Cs[0], Cs[1], r0 * 1.0)
    ctx.fill()
    # solid body under the cards
    _ribbon(ctx, geo, 0, len(P), 1.0)
    ctx.set_source(lin_grad(0, geo["E"][1], 0, H, [(0, BODY_M, fade), (1, BODY_D, fade)]))
    ctx.fill()
    v = 1.9
    sigE = geo["sigE"]
    # --- population B: dense body cards in the rising part (they hand over to A at the lip)
    pb = _cards(geo, st["popB"], T, v, 80 * 0.075, -0.4, sigE - 0.2, 1.45 / 13 * 1.0, 95.0, fade_top=0.8)
    # --- population A: the hero cards, flowing up and winding into the infinite spiral
    pa = _cards(geo, st["popA"], T, v, 420 * 0.23, -0.4, 1e9, 1.3 / 5, 120.0)
    for (Pp, n, idx, size, ang, sx, shade_k, alpha) in pb + pa:
        atlas.draw(ctx, int(idx), Pp[0], Pp[1], size, ang, sx=sx, shade_k=shade_k, alpha=alpha * fade, px_scale=s)
    # depth: the body darkens toward the root and inside the face (Hokusai contrast)
    ctx.save()
    _ribbon(ctx, geo, 0, geo["iE"] + 60, 1.02)
    ctx.clip()
    ctx.set_source(lin_grad(0, geo["E"][1], 0, H, [(0, BODY_D, 0.0), (0.45, BODY_D, 0.18 * fade), (1, "#1E1830", 0.62 * fade)]))
    ctx.paint()
    ctx.restore()
    # crest highlight rim along the outer edge of the lip
    i0 = max(0, geo["iE"] - 40)
    i1 = min(len(P), geo["iE"] + 220)
    edge = P[i0:i1] - N[i0:i1] * (Tk[i0:i1, None] * 0.5)
    ctx.set_line_width(5)
    ctx.set_source_rgba(1, 0.97, 1, 0.55 * fade)
    poly(ctx, edge, close=False)
    ctx.stroke()


def _cards(geo, pop, T, v, period, s_lo, s_hi, kfac, cap, fade_top=0.0):
    sig0, n, idx, szj, angj, ph, tumble = (pop[:, i] for i in range(7))
    ss = (sig0 + v * (T - T0)) % period + s_lo
    ok = (ss >= s_lo) & (ss <= min(s_hi, geo["sig"][-1]))
    if not ok.any():
        return []
    ss, n, idx, szj, angj, ph, tumble = ss[ok], n[ok], idx[ok], szj[ok], angj[ok], ph[ok], tumble[ok]
    P, N, Tk, tg, turn = at_sigma(geo, ss)
    pos = P + N * (n * Tk * 0.5)[:, None]
    size = np.minimum(Tk * kfac * szj, cap)
    ang = np.arctan2(tg[:, 1], tg[:, 0]) + angj + 0.25 * np.sin(T * tumble + ph)
    sx = np.cos(T * tumble * 0.7 + ph)
    sx = np.where(np.abs(tumble) > 1.0, sx, 1.0)
    # light: outer side (n<0) catches the pale sky; face and inner turns fall into shadow
    shade_k = np.clip(0.1 + 0.42 * (n + 1) / 2 + 0.22 * turn + 0.25 * (pos[:, 1] > 780), 0, 1)
    alpha = np.ones_like(ss)
    if fade_top > 0:
        alpha = np.clip((min(s_hi, geo["sig"][-1]) - ss) / fade_top, 0, 1)
        size = size * (0.4 + 0.6 * alpha)
    order = np.argsort(ss)
    return [(pos[i], n[i], idx[i], size[i], ang[i], sx[i], shade_k[i], alpha[i]) for i in order]


def _finger(ctx, base, out, fw, L, w0, curl, th0, T, seed, fade):
    """one tapered foam finger that curls forward; returns its centreline points"""
    n = 14
    pts, ws = [], []
    p = np.array(base, np.float64)
    for j in range(n):
        t = j / (n - 1)
        th = th0 + curl * t * t + 0.12 * math.sin(T * 5 + seed + t * 3)
        d = out * math.cos(th) + fw * math.sin(th)
        pts.append(p.copy())
        ws.append(w0 * (1 - t) ** 0.9 + 0.6)
        p = p + d * (L / (n - 1))
    pts = np.array(pts)
    tg = np.gradient(pts, axis=0)
    tg /= (np.hypot(tg[:, 0], tg[:, 1])[:, None] + 1e-9)
    nn = np.stack([-tg[:, 1], tg[:, 0]], 1)
    wv = np.array(ws)[:, None]
    Lp = pts + nn * wv
    Rp = pts - nn * wv
    poly_pts = np.concatenate([Lp, Rp[::-1]])
    ctx.move_to(*poly_pts[0])
    for q in poly_pts[1:]:
        ctx.line_to(*q)
    ctx.close_path()
    ctx.set_source(lin_grad(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1],
                            [(0, "#FFFFFF", 0.97 * fade), (0.7, "#FBEFF5", 0.95 * fade), (1, "#F4B6C8", 0.9 * fade)]))
    ctx.fill_preserve()
    ctx.set_source_rgba(*col("#9FD0E0"), 0.75 * fade)
    ctx.set_line_width(1.8)
    ctx.stroke()
    return pts


def draw_claws(ctx, geo, T, atlas, s):
    """Hokusai's claws: tapered fingers of foam along the lip, grasping forward at the memeplexes;
    each fingertip ends in a little heart, bubbles cling to the knuckles"""
    P, N, Tk, tg = geo["P"], geo["N"], geo["Tk"], geo["tang"]
    iE = geo["iE"]
    fade = 1 - geo["cq"]
    g_ = geo["g"]
    # foam band along the crest first (claws grow out of it)
    for j in range(300):
        i = iE - 90 + int(j * 1.0)
        if i < 0 or i >= len(P):
            continue
        off = 0.44 + 0.14 * hash01(j, 3)
        p = P[i] - N[i] * Tk[i] * off
        r = 3 + 7 * hash01(j, 4) * (Tk[i] / 160) ** 0.5
        ctx.set_source_rgba(1, 0.98, 1, 0.92 * fade)
        circle(ctx, p[0] + 3 * math.sin(T * 5 + j), p[1] + 3 * math.cos(T * 4 + j), r)
        ctx.fill()
    nclaw = 9
    for c in range(nclaw):
        u = (c + 0.3 + 0.4 * hash01(c, 1)) / nclaw
        i = iE + 6 + int(u * 150)
        if i >= len(P):
            break
        base = P[i] - N[i] * Tk[i] * 0.45
        out = -N[i]
        fw = tg[i]
        L = (70 + 110 * math.sin(min(1.0, u * 1.3) * math.pi * 0.9)) * (0.5 + 0.5 * g_) \
            * (1 + 0.12 * noise1(T * 2.2 + c * 3.1, c))
        w0 = L * 0.16
        main = _finger(ctx, base, out, fw, L, w0, 2.2, 0.35, T, c * 1.7, fade)
        tips = [main[-1]]
        for b, (tb, lb, dth) in enumerate(((0.35, 0.55, -0.55), (0.55, 0.45, 0.5), (0.72, 0.35, -0.2))):
            j = int(tb * (len(main) - 1))
            d = main[min(j + 1, len(main) - 1)] - main[j]
            d = d / (np.hypot(*d) + 1e-9)
            o2 = d * math.cos(dth) + np.array([-d[1], d[0]]) * math.sin(dth)
            f2 = np.array([-o2[1], o2[0]])
            if np.dot(f2, fw) < 0:
                f2 = -f2
            sub = _finger(ctx, main[j], o2, f2, L * lb, w0 * 0.55, 1.8, 0.0, T, c * 3.3 + b, fade)
            tips.append(sub[-1])
        for k, tp in enumerate(tips):
            ctx.set_source_rgba(*col("#FF7FA6" if (c + k) % 2 else "#FF5C8A"), 0.95 * fade)
            _heart(ctx, tp[0], tp[1], 3.5 + 2.5 * (k == 0))
            ctx.fill()
        for k in range(0, len(main), 3):
            r = max(1.5, w0 * 0.55 * (1 - k / len(main)))
            ctx.set_source_rgba(1, 1, 1, 0.9 * fade)
            circle(ctx, main[k][0] - out[0] * w0 * 0.6, main[k][1] - out[1] * w0 * 0.6, r)
            ctx.fill()


def draw_spray(ctx, atlas, geo, T, s):
    """bubbles, hearts and tiny cards flung off the crest"""
    fade = 1 - geo["cq"]
    P, N, Tk, tg = geo["P"], geo["N"], geo["Tk"], geo["tang"]
    iE = geo["iE"]
    for k in range(160):
        life = 1.1
        te = T0 + (k * 0.037) % 60
        # each particle loops: emitted periodically
        age = ((T - T0) + hash01(k, 1) * life) % life
        i = iE - 30 + int(hash01(k, 2) * 150)
        i = min(i, len(P) - 1)
        base = P[i] - N[i] * Tk[i] * 0.55
        vel = (-N[i] * (140 + 260 * hash01(k, 3)) + tg[i] * (220 + 200 * hash01(k, 4)))
        x = base[0] + vel[0] * age
        y = base[1] + vel[1] * age + 380 * age * age
        a = (1 - age / life) * fade
        if k % 3 == 0:
            atlas.draw(ctx, k * 11, x, y, 8 + 14 * hash01(k, 5), age * 6 + k, shade_k=0.1, alpha=a, px_scale=s)
        elif k % 3 == 1:
            ctx.set_source_rgba(*col("#FF8FB1"), a)
            _heart(ctx, x, y, 3 + 4 * hash01(k, 6))
            ctx.fill()
        else:
            ctx.set_source_rgba(1, 0.98, 1, a)
            circle(ctx, x, y, 2 + 4 * hash01(k, 7))
            ctx.fill()


BURST_C = (1170.0, SEA_Y - 70)


def draw_burst(ctx, g, atlas, st, T, s, back=False):
    """the lip explodes on the NATION: a fan of cards + foam behind the memeplexes (back=True) and
    a front spray of cards, bubbles and hearts (back=False)"""
    B = st["burst"]
    age = T - T_CRASH - B["delay"]
    cx, cy = BURST_C
    a0 = T - T_CRASH
    if back:
        # a towering fan of slop plumes behind the buildings: streams of cards capped with foam
        if 0 < a0 < 1.5:
            fa = 1 - smoothstep(0.45, 1.1, a0)
            grow = ease_out(clamp(a0 / 0.5), 2.2)
            for k in range(7):
                ang = -math.pi / 2 + (k - 3) * 0.26 + 0.06 * math.sin(k * 3.1)
                L = (380 + 300 * hash01(k, 31)) * grow
                w0 = 55 + 30 * hash01(k, 32)
                bx = cx + (k - 3) * 95
                tipx, tipy = bx + math.cos(ang) * L, cy + math.sin(ang) * L + 120 * max(0, a0 - 0.5) ** 2
                ctx.move_to(bx - w0, cy + 20)
                ctx.curve_to(bx - w0 * 0.8, cy - L * 0.4, tipx - w0 * 0.5, tipy + w0 * 0.6, tipx, tipy)
                ctx.curve_to(tipx + w0 * 0.5, tipy + w0 * 0.6, bx + w0 * 0.8, cy - L * 0.4, bx + w0, cy + 20)
                ctx.close_path()
                ctx.set_source(lin_grad(0, tipy, 0, cy, [(0, "#E8C8D0", 0.3 * fa), (1, BODY_M, 0.55 * fa)]))
                ctx.fill()
                for j in range(18):
                    t = j / 17
                    px = bx + (tipx - bx) * t + (hash01(k * 20 + j, 34) - 0.5) * w0 * (1.4 - t)
                    py = cy + (tipy - cy) * t
                    atlas.draw(ctx, k * 37 + j * 5, px, py, (34 - 20 * t) * (0.8 + 0.4 * hash01(k * 20 + j, 35)),
                               a0 * (3 + k) + j, sx=math.cos(a0 * 5 + j), shade_k=0.1 + 0.3 * (1 - t),
                               alpha=fa, px_scale=s)
                for j in range(6):
                    r = w0 * (0.28 - 0.035 * j) * (0.8 + 0.4 * hash01(k * 10 + j, 33))
                    px = tipx + (hash01(k * 10 + j, 36) - 0.5) * w0 * 0.8
                    py = tipy + j * 9
                    ctx.set_source_rgba(1, 0.98, 1, 0.9 * fa)
                    circle(ctx, px, py, r)
                    ctx.fill()
    # card shrapnel (ballistic, drag, spin): even particles fly in front, odd ones behind the memeplexes
    parity = 1 if back else 0
    ok = age > 0
    idxs = np.nonzero(ok)[0]
    for i in idxs:
        if i % 2 != parity:
            continue
        t = age[i]
        vx = math.cos(B["ang"][i]) * B["spd"][i]
        vy = math.sin(B["ang"][i]) * B["spd"][i]
        drag = 1.6
        f = (1 - math.exp(-drag * t)) / drag
        x = cx + B["ox"][i] + vx * f
        y = cy + B["oy"][i] + vy * f + 520 * t * t * 0.5 + 160 * t
        if y > H + 60 or ((i // 2) % 3 == 0 and t > 1.0):
            continue
        kind = (i // 2) % 4
        if kind == 0:
            r = 2 + B["size"][i] * 0.18
            ctx.set_source_rgba(1, 0.98, 1, 0.9)
            circle(ctx, x, y, r)
            ctx.fill()
        elif kind == 1:
            ctx.set_source_rgba(*col("#FF7FA6"), 0.95)
            _heart(ctx, x, y, 2 + B["size"][i] * 0.15)
            ctx.fill()
        else:
            atlas.draw(ctx, int(B["idx"][i]), x, y, B["size"][i], B["spin"][i] * t,
                       sx=math.cos(B["spin"][i] * t * 0.7), shade_k=0.15 + 0.3 * parity, alpha=1.0, px_scale=s)


# ----------------------------------------------------------------------------------------------- glitch
GLITCH_T = (35.32, 36.58, 37.22, 37.72, 38.24, 38.93, 39.2)


def glitch_amount(T):
    """timed glitch bursts (~0.15 s) that grow with the wave; the crash tears the picture for 0.35 s"""
    rise = clamp((T - T0) / (T_CRASH - T0))
    a = 0.0
    for i, tg in enumerate(GLITCH_T):
        dur = 0.17 if i == len(GLITCH_T) - 1 else 0.13 + 0.05 * rise
        if tg <= T < tg + dur:
            a = max(a, 0.4 + 0.6 * rise if i < len(GLITCH_T) - 1 else 1.0)
    return a


def glitch(img, fc, T):
    """horizontal slice displacement with RGB split during the timed bursts"""
    rise = clamp((T - T0) / (T_CRASH - T0))
    amt = glitch_amount(T)
    burst = T >= T_CRASH
    rng = np.random.default_rng(fc.F * 131 + 7)
    if amt <= 0:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    sh = int(round(9 * amt * fc.s))
    if sh:
        out[..., 0] = np.roll(img[..., 0], sh, axis=1)
    src = out.copy()
    n = int(2 + 9 * amt)
    for _ in range(n):
        y0 = int(rng.uniform(0, h))
        hh = int(rng.uniform(4, 40) * fc.s * (0.5 + amt))
        dx = int(rng.normal(0, 40 + 90 * amt) * fc.s)
        y1 = min(h, y0 + hh)
        if y1 <= y0 or dx == 0:
            continue
        band = src[y0:y1]
        out[y0:y1, :, 0] = np.roll(band[..., 0], dx + int(14 * fc.s), axis=1)
        out[y0:y1, :, 1] = np.roll(band[..., 1], dx, axis=1)
        out[y0:y1, :, 2] = np.roll(band[..., 2], dx, axis=1)
        if rng.random() < 0.25:
            tint = np.array(col("#FF2E88") if rng.random() < 0.5 else col("#00E5FF"), np.float32)
            out[y0:y1] = out[y0:y1] * 0.75 + tint * 0.25
    return out

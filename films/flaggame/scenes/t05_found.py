"""t05 shot B — 'Found one!' (108.44 -> 110.05). Rain, churned mud, a searcher crouched over a pole stuck in
the mud; the furled cloth is the only colour in the storm. SHLUPP: the pole comes out (its lower third
caked in mud, dripping), is hoisted high and the cloth unrolls in the wind — plain, clean, pristine.
Also the inset for the map (H02 'Move it! MOVE IT!')."""
import math

import cairo
import numpy as np

from vx import ink, comic
from vx.canvas import Canvas
from vx.chars import Character, draw_crowd
from vx.ease import clamp, lerp, smoothstep, ease_in_out, ease_out, ease_in, ease_out_back
from vx.flag import Flag, FlagTrack

from scenes import effigy as E

VER = 4
T0_SIM = 108.30
T_PULL = 108.72
T_END = 110.05
H_MAN = 640.0                         # the searcher's standing height (px)
POLE = H_MAN * 96.0 / 69.0            # canonical 96" pole vs a 5'9" hippie
MAN_X, MAN_Y = 1105.0, 1092.0         # feet (just below the frame)
MUD_Y = 1022.0                        # mud surface where the pole is stuck
PAN = 0.05
CAM = E.Cam3(pos=(0.0, 0.9, 0.0), yaw=0.0, pitch=0.06, f=1100.0)
STUCK = (800.0, 1122.0, 0.36)         # base (buried), angle: leaning towards the searcher
HOLE_X = STUCK[0] + math.tan(STUCK[2]) * (STUCK[1] - 1022.0)
HELD = (1045.0, 905.0, -0.6)          # held up, leaning left over the frame, the cloth streaming right


def pole_pose(T):
    """pole base (x, y) and angle; the base is BELOW the mud surface while stuck"""
    if T < T_PULL - 0.34:
        return STUCK
    if T < T_PULL:
        # straining: the pole creaks, trembles and rises a few px
        u = (T - (T_PULL - 0.34)) / 0.34
        return STUCK[0] + 5 * u, STUCK[1] - 16 * u * u, STUCK[2] + 0.018 * math.sin(T * 70) * u
    u = clamp((T - T_PULL) / 0.36)
    e = ease_out_back(u, 1.35)
    x = lerp(STUCK[0], HELD[0], ease_out(u, 2.2))
    y = lerp(STUCK[1], HELD[1], ease_out(u, 2.6)) - 60 * math.sin(math.pi * u)
    a = lerp(STUCK[2], HELD[2], e)
    # held aloft: a triumphant shake
    dt = T - T_PULL - 0.36
    if dt > 0:
        a += 0.06 * math.sin(dt * 9.0) * math.exp(-dt * 1.2)
        y += 5 * math.sin(dt * 7.0)
    return x, y, a


def cam2d(T):
    """a 2D camera: starts close on the struggle, pulls out and tilts up with the flag as it is raised"""
    from vx.camera import Camera
    u = ease_in_out(clamp((T - (T_PULL - 0.08)) / 0.62), 2.0)
    z = lerp(1.34, 1.0, u) + 0.02 * clamp((T - T_PULL - 0.6) / 1.2)
    return Camera(lerp(930.0, 960.0, u), lerp(760.0, 540.0, u), z)


def furl(T):
    return 1.0 - smoothstep(T_PULL + 0.06, T_PULL + 0.34, T)


def path_key(fps=24):
    n = int(math.ceil((T_END - T0_SIM) * fps)) + 2
    out = []
    for i in range(0, n, 2):
        T = T0_SIM + i / fps
        out += list(pole_pose(T)) + [furl(T)]
    return out + [POLE, VER]


def simulate_flag(fps=24):
    n = int(math.ceil((T_END - T0_SIM) * fps)) + 2

    def pole_fn(i):
        return pole_pose(T0_SIM + i / fps)

    def wind_fn(i):
        T = T0_SIM + i / fps
        return (560.0 + 120 * math.sin(T * 2.3), 60.0)

    tr = Flag(pole=POLE, seed=108, side=1).simulate(n, pole_fn, wind_fn, fps=fps, substeps=10, warmup=48,
                                                   furl_fn=lambda i: furl(T0_SIM + i / fps))
    tr.meta["T0"] = T0_SIM
    tr.meta["ver"] = VER
    return tr


def _fi(tr, T):
    return min(max((T - tr.meta["T0"]) * 24.0, 0.0), tr.n - 1.0)


_MAN = {}


def _man(kind="hippie", seed=31, **kw):
    k = (kind, seed, tuple(sorted(kw.items())))
    if k not in _MAN:
        _MAN[k] = Character(kind, seed=seed, render="ink", **kw)
    return _MAN[k]


def man_pose(T):
    """pose dict + hand targets on the pole"""
    x, y, a = pole_pose(T)
    L = POLE
    def on(f):
        return (x + math.sin(a) * L * f, y - math.cos(a) * L * f)
    k = smoothstep(T_PULL - 0.02, T_PULL + 0.32, T)
    crouch = 1.0 - k
    pose = {"kneel": 0.8 * crouch, "stand": 1.0 - 0.8 * crouch}
    hr = on(lerp(0.44, 0.22, k))
    hl = on(lerp(0.33, 0.05, k))
    return pose, hr, hl, crouch


def draw_mud(ctx, fc, T):
    """foreground mud: churned ruts, boot prints, the hole the pole leaves, clods thrown by the pull"""
    rng = np.random.default_rng(5)
    ctx.save()
    # ruts and ridges (dark grooves with bright wet tops)
    for k in range(34):
        y = MUD_Y - 60 + k * 5.5 + rng.uniform(-3, 3)
        x0 = rng.uniform(-100, 1500)
        L = rng.uniform(180, 700) * (1 + (y - MUD_Y) / 200)
        ctx.set_line_width(rng.uniform(1.5, 5.0) * (0.6 + (y - MUD_Y + 60) / 200))
        c = rng.choice([0.08, 0.2, 0.92])
        ctx.set_source_rgba(c, c, c, 0.9)
        ctx.move_to(x0, y)
        ctx.curve_to(x0 + L * 0.3, y + rng.uniform(-8, 8), x0 + L * 0.6, y + rng.uniform(-8, 8), x0 + L, y + rng.uniform(-6, 6))
        ctx.stroke()
    # boot prints
    for k in range(9):
        x = rng.uniform(250, 1700)
        y = rng.uniform(MUD_Y + 10, 1070)
        s = 0.8 + (y - MUD_Y) / 160
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(s, s * 0.45)
        ctx.rotate(rng.uniform(-0.4, 0.4))
        ctx.arc(0, 0, 16, 0, 2 * math.pi)
        ctx.set_source_rgba(0.1, 0.1, 0.1, 0.85)
        ctx.fill()
        ctx.arc(0, -34, 11, 0, 2 * math.pi)
        ctx.fill()
        ctx.restore()
    # the hole where the pole was / the mound it is stuck in
    px, py = HOLE_X, MUD_Y + 6
    ctx.save()
    ctx.translate(px, py)
    ctx.scale(1, 0.32)
    if T < T_PULL:
        ctx.arc(0, 0, 48, 0, 2 * math.pi)
        ctx.set_source_rgba(0.14, 0.14, 0.14, 1)
        ctx.fill()
    else:
        ctx.arc(0, 0, 40, 0, 2 * math.pi)
        ctx.set_source_rgba(0.02, 0.02, 0.02, 1)
        ctx.fill()
        ctx.arc(0, 0, 40, math.pi, 2 * math.pi)
        ctx.set_source_rgba(0.9, 0.9, 0.9, 1)
        ctx.set_line_width(4)
        ctx.stroke()
    ctx.restore()
    ctx.restore()


def draw_clods(ctx, T):
    """mud clods flung by the pull (front layer: they fly in front, never over the cloth — masked later)"""
    if not (T_PULL <= T < T_PULL + 1.2):
        return
    rng = np.random.default_rng(9)
    dt = T - T_PULL
    ox, oy = HOLE_X, MUD_Y
    for k in range(26):
        vx = rng.uniform(-420, 520)
        vy = rng.uniform(-900, -350)
        r = rng.uniform(3, 11)
        x = ox + vx * dt
        y = oy + vy * dt + 0.5 * 2600 * dt * dt
        if y > MUD_Y + 80:
            continue
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(k + dt * 9)
        ctx.scale(1.0, 0.7)
        ctx.arc(0, 0, r, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgb(0.03, 0.03, 0.03)
        ctx.fill()


def draw_pole_mud(ctx, T):
    """the buried third of the pole comes out caked in mud: blobs + drips along the pole (front layer)"""
    if T < T_PULL - 0.3:
        return
    x, y, a = pole_pose(T)
    L = POLE
    sa, ca = math.sin(a), math.cos(a)
    rng = np.random.default_rng(3)
    ctx.save()
    for k in range(40):
        f = rng.uniform(0.0, 0.2) ** 1.1
        px, py = x + sa * L * f, y - ca * L * f
        w = rng.uniform(3, 9) * (1.3 - f * 2)
        ctx.save()
        ctx.translate(px, py)
        ctx.rotate(a)
        ctx.scale(w / 5, rng.uniform(1.0, 3.0))
        ctx.arc(rng.uniform(-3, 3), 0, 5, 0, 2 * math.pi)
        ctx.restore()
        c = 0.06 if k % 5 else 0.35
        ctx.set_source_rgb(c, c, c)
        ctx.fill()
    # drips falling from the butt of the pole
    dt = T - T_PULL
    if dt > 0:
        for k in range(6):
            t = (dt * 1.6 + k / 6.0) % 1.0
            dx = x + rng.uniform(-3, 3)
            dy = y + 20 + 520 * t * t
            ctx.set_source_rgb(0.05, 0.05, 0.05)
            ctx.save()
            ctx.translate(dx, dy)
            ctx.scale(1, 1.8)
            ctx.arc(0, 0, 3.2 * (1 - 0.5 * t), 0, 2 * math.pi)
            ctx.restore()
            ctx.fill()
    ctx.restore()


def background(fc, T, st):
    """sky + far mud (effigy.sky, firmament hidden), other searchers in the rain"""
    d = E.sky(fc, CAM, T - 100.0, cover=1.0, reveal_r=0.0, firmament=False, res=0.35, storm=0.8, glow=0.25)
    g = d["cloud"] * d["cloud_a"] + d["water"] * (1 - d["cloud_a"])
    g = g * d["hit"] + d["ground"] * (1 - d["hit"])
    return np.clip(g * 0.95 + 0.03, 0, 1)


_SEARCHERS = [(-7.5, 14.0, 0.0), (-2.5, 9.5, 1.3), (3.8, 17.0, 2.1), (8.0, 11.0, 0.7), (-11, 22, 3.0),
              (12, 24, 1.1), (0.8, 28, 2.6)]


def draw_searchers(ctx, T, turn_k=0.0):
    people = []
    for i, (x, z, ph) in enumerate(_SEARCHERS):
        sx, sy, zz = CAM.project(np.array([[x, 0.0, z]]))
        h = 1.75 * CAM.f / zz[0]
        # they search bent over; on 'Found one!' heads come up and turn
        up = smoothstep(108.95 + 0.07 * i, 109.35 + 0.07 * i, T) * turn_k
        people.append(dict(x=float(sx[0]), y=float(sy[0]), h=float(h), kind="hippie", seed=200 + i,
                           pose={"kneel": 0.55 * (1 - up), "stand": 1 - 0.55 * (1 - up)},
                           facing=1 if x < 0 else -1, look=(0.6 if x < 0 else -0.6, 0.5 * (1 - up)),
                           t_off=ph, turn=0.6))
    people.sort(key=lambda p: p["h"])
    draw_crowd(ctx, people, T)


def draw_rain(ctx, T, bg, fc, n=900, seed=2, rect=(0, 0, 1920, 1080), speed=1.0, alpha=0.85):
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = rect
    xs = rng.uniform(x0 - 100, x1 + 100, n)
    ys = rng.uniform(y0, y1 + 200, n)
    sp = rng.uniform(0.6, 1.0, n)
    h, w = bg.shape[:2]
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for i in range(n):
        x = x0 + (xs[i] - x0 + T * 330 * sp[i] * speed) % (x1 - x0 + 200) - 100
        y = y0 + (ys[i] - y0 + T * 1500 * sp[i] * speed) % (y1 - y0 + 200) - 100
        L = 60 * sp[i]
        tb = bg[min(h - 1, max(0, int(y * fc.s))), min(w - 1, max(0, int(x * fc.s)))]
        c = 0.97 if tb < 0.62 else 0.15
        ctx.set_source_rgba(c, c, c, alpha)
        ctx.set_line_width(1.0 + 1.2 * sp[i])
        ctx.move_to(x, y)
        ctx.line_to(x - L * 0.22, y - L)
        ctx.stroke()
    ctx.restore()


def render(fc, T, st, print_k=1.0):
    tr = st["found_flag"]
    fi = _fi(tr, T)
    world = Canvas(fc)
    bg = background(fc, T, st)
    world.paint_array(np.repeat(bg[..., None], 3, 2))
    ctx = world.ctx
    cam = cam2d(T)
    ctx.save()
    cam.apply(ctx, 0.35)
    draw_searchers(ctx, T, 1.0)
    ctx.restore()
    ctx.save()
    cam.apply(ctx)
    draw_mud(ctx, fc, T)
    man = _man("hippie", 31, mud=0.85, wet=True)
    pose, hr, hl, crouch = man_pose(T)
    mouth = fc.tl.mouth("HIPPIE1", T)
    expr = "stern" if T < T_PULL + 0.05 else "cheerful"
    pa = pole_pose(T)[2]
    kw = dict(hand_r=hr, hand_l=hl, shape_r="grip", shape_l="grip", mouth=mouth, pole_ang=-pa, expr=expr,
              look=(-0.6, -0.7 if T > T_PULL + 0.1 else 0.5), lean=0.3 * crouch, turn=0.3)
    man.draw(ctx, MAN_X, MAN_Y, H_MAN, pose, T, facing=-1, layer="back", **kw)
    ctx.restore()
    grey = world.rgb()
    draw_rain(world.ctx, T, grey[..., 0], fc, n=300, alpha=0.6)
    grey = world.rgb()
    if print_k < 1.0:
        grey = 1.0 - (1.0 - grey) * print_k
    img = ink.engrave(fc, grey)
    # the flag plate
    flags = Canvas(fc)
    fctx = flags.ctx
    cam.apply(fctx)
    if T < T_PULL + 0.12:
        # the buried part of the pole is under the mud (the ground occludes the pole, never the cloth)
        fctx.rectangle(-100, -100, 2200, MUD_Y + 4 + 100)
        fctx.clip()
    tr.draw(fctx, fi)
    fl = flags.rgba()
    cloth = Canvas(fc)
    cam.apply(cloth.ctx)
    tr.draw(cloth.ctx, fi, pole=False)
    cloth_a = cloth.rgba()[..., 3:4]
    img = ink.spot(img, fl * (1.0 if print_k >= 1 else print_k))
    # front: the hands on the pole, the mud on the pole, flung clods (never over the cloth)
    front = Canvas(fc)
    cam.apply(front.ctx)
    man.draw(front.ctx, MAN_X, MAN_Y, H_MAN, pose, T, facing=-1, layer="front", **kw)
    draw_pole_mud(front.ctx, T)
    draw_clods(front.ctx, T)
    fr = ink.print_layer(fc, front.rgba(), "engrave")
    fr = fr * (1.0 - cloth_a) * print_k
    img = fr[..., :3] + img * (1.0 - fr[..., 3:4])
    # lettering
    top = Canvas(fc)
    def scr(x, y, Tq=T):
        return cam2d(Tq).to_screen(x, y)

    def cbox(Tq):
        x0, y0, x1, y1 = _cloth_bbox(tr, _fi(tr, Tq))
        a, b = scr(x0, y0, Tq)
        c, d = scr(x1, y1, Tq)
        return (a, b, c, d)
    anc = man.anchors(MAN_X, MAN_Y, H_MAN, pose, T, facing=-1, **kw)
    mx, my = scr(*anc["mouth"])
    comic.say(top.ctx, fc, "H01", 1560, 250, tail=(mx + 10, my - 6), avoid=lambda Tq: [cbox(Tq)])
    if T >= T_PULL and T < T_PULL + 0.5:
        hx, hy = scr(HOLE_X, MUD_Y)
        comic.sfx(top.ctx, "SHLUPP!", hx - 250, hy - 120, 120, rot=-0.12, t=T - T_PULL, depth=8,
                  avoid=[cbox(T)])
    return img, top


def _cloth_bbox(tr, fi):
    i = int(round(fi))
    V = tr.verts[i][..., :2].reshape(-1, 2)
    return (float(V[:, 0].min()) - 20, float(V[:, 1].min()) - 20, float(V[:, 0].max()) + 20, float(V[:, 1].max()) + 20)


# ============================================================ the map's inset: 'Move it! MOVE IT!'
INSET = (40.0, 500.0, 700.0, 1040.0)
T_IN0, T_IN1 = 110.05, 111.62


def inset(top, fc, T, st, avoid=()):
    """drawn onto the top (lettering) canvas as a finished printed panel: its own mini-world"""
    if not (T_IN0 - 0.01 <= T < T_IN1 + 0.2):
        return None
    k_in = ease_out_back(clamp((T - T_IN0) / 0.22), 1.6)
    k_out = 1.0 - smoothstep(T_IN1, T_IN1 + 0.2, T)
    x0, y0, x1, y1 = INSET
    w, h = x1 - x0, y1 - y0
    # render the inset content at full frame (design px), then paint it clipped into the panel
    world = Canvas(fc, bg=(0.93, 0.93, 0.93))
    ctx = world.ctx
    # a smeared, rain-lashed background
    g = cairo.LinearGradient(0, y0, 0, y1)
    g.add_color_stop_rgb(0, 0.55, 0.55, 0.55)
    g.add_color_stop_rgb(0.55, 0.8, 0.8, 0.8)
    g.add_color_stop_rgb(1, 0.4, 0.4, 0.4)
    ctx.set_source(g)
    ctx.rectangle(x0, y0, w, h)
    ctx.fill()
    man = _man("hippie", 44, mud=0.9, wet=True)
    runner_x = x0 + w * 0.3
    fy = y1 - 30
    hh = 360.0
    pole_len = hh * 96 / 69
    ang = 0.95 + 0.05 * math.sin(T * 9)
    # the flag carried over the shoulder at a slant, streaming back as she runs at the viewer
    anc = man.anchors(runner_x, fy, hh, "run", T, facing=-1, turn=0.35, speed=1.3)
    sx, sy = anc["shoulder_r"]
    bx, by = sx - math.sin(ang) * pole_len * 0.33, sy + math.cos(ang) * pole_len * 0.33
    tipx, tipy = bx + math.sin(ang) * pole_len, by - math.cos(ang) * pole_len
    hr = (bx + math.sin(ang) * pole_len * 0.3, by - math.cos(ang) * pole_len * 0.3)
    mouth = fc.tl.mouth("HIPPIE2", T)
    kw = dict(hand_r=hr, shape_r="grip", mouth=max(0.25, mouth), look=(-0.3, 0.0), turn=0.35, speed=1.3,
              expr="furious" if T > 110.6 else "neutral")
    man.draw(ctx, runner_x, fy, hh, "run", T, facing=-1, layer="back", **kw)
    draw_rain(ctx, T, np.full((fc.h, fc.w), 0.5, np.float32), fc, n=260, seed=5, rect=INSET, speed=1.2)
    grey = world.rgb()
    pr = ink.engrave(fc, grey)
    flags = Canvas(fc)
    fctx = flags.ctx
    fctx.rectangle(x0, y0, w, h)
    fctx.clip()
    from vx.flag import draw_flag
    draw_flag(fctx, bx, by, pole=pole_len, t=T, wind=1.3, ang=ang, seed=61, lean=0.03)
    pr = ink.spot(pr, flags.rgba())
    front = Canvas(fc)
    man.draw(front.ctx, runner_x, fy, hh, "run", T, facing=-1, layer="front", **kw)
    fr = ink.print_layer(fc, front.rgba(), "engrave")
    pr = fr[..., :3] + pr * (1 - fr[..., 3:4])
    # paint into the lettering canvas: scaled pop-in about the panel centre, clipped to the panel
    ctx = top.ctx
    ctx.save()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ctx.translate(cx, cy)
    ctx.scale(k_in, k_in)
    ctx.rotate(-0.025)
    ctx.translate(-cx, -cy)
    ctx.rectangle(x0, y0, w, h)
    ctx.save()
    ctx.clip()
    from vx.canvas import paint_array
    rgba = np.concatenate([pr, np.ones(pr.shape[:2] + (1,), np.float32)], 2)
    paint_array(ctx, rgba, 0, 0, fc.s, alpha=k_out)
    ctx.restore()
    ctx.rectangle(x0, y0, w, h)
    ctx.set_source_rgba(0, 0, 0, k_out)
    ctx.set_line_width(7)
    ctx.stroke()
    ctx.restore()
    ax, ay = anc["mouth"]
    cb = []
    comic.say(ctx, fc, "H02", x0 + w * 0.5, y0 - 70, tail=(ax - 6, ay - 4), avoid=list(avoid), alpha=k_out)
    return INSET

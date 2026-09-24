"""t07 — the lone flag-bearer (Doré engraving): panel #100 of the page, the dive into it, the walk through the
changing Burns (P22), the Burn without fire and the years without Burns (P23), the unmask and the reprint into
the present-day camp (hand-off to t08 at 159.83).  Owner: T07Raised.

World units = t08's opening framing (scenes.t08_camp.crow_t07 / flag_t07): Crow's feet at W0, h = 1033, the
pole butt planted at FLAG["base"]. The walker walks IN PLACE (facing left); the Burns scroll past him in
parallax layers; the camera zooms about his feet: screen = (world - W0) * z + S0.
Every layer is engraved separately with its page glued to it (ink.print_layer(..., space="page", matrix=M)),
so the hatching rides with the landscape instead of swimming.
"""
import math

import cairo
import numpy as np

from vx import *
from vx import ink, comic
from vx.flag import Flag
from vx.chars import Character
from scenes import t08_camp as C8

# ------------------------------------------------------------------ time
RAISE = 146.769             # raise_100: he raises the Flag in panel #100 ('time')
WALK0, WALK1 = 148.75, 158.25
DIVE0, DIVE1 = 147.85, 149.45
UNMASK = 158.8
PUSH0, PUSH1 = 158.15, 158.72
PULL0 = 159.22
END = 3836 / 24.0           # 159.8333 (t08 frame 0)
HOLD = END - 3 / 24.0       # from here: t08's own opening frame (pixel-exact cut)
NO_FIRE = 155.451
NO_BURNS = 157.501
F_TRACK0 = 3516             # flag sim global frame 0 (146.5 s)
N_TRACK = 3890 - F_TRACK0   # through 162.1 s (t08 blends its first frames from ours)

# stations and the wipes between them (T of the wiper crossing the walker)
WIPES = [(151.35, "forest", "trunk"), (153.3, "lake", "rock"), (155.02, "nofire", "tent"),
         (156.38, "years", "flurry")]
FIRST = "playa"

_C8 = {}


def spec():
    if not _C8:
        _C8["crow"] = C8.crow_t07()
        _C8["flag"] = C8.flag_t07()
    return _C8["crow"], _C8["flag"]


# ------------------------------------------------------------------ the walk
def walk_speed():
    cr, _ = spec()
    return _ch().ground_speed(cr["h"], "walk") * 0.92


_WT = {}


def _walk_table():
    if "t" not in _WT:
        v = walk_speed()
        ts = np.arange(WALK0 - 1.0, WALK1 + 2.0, 0.002)
        r = np.array([stride(t) for t in ts])
        d = np.concatenate([[0.0], np.cumsum((r[1:] + r[:-1]) * 0.5 * 0.002)]) * v
        _WT["t"], _WT["d"] = ts, d
    return _WT["t"], _WT["d"]


def walked(T):
    """world units walked by T (smooth start/stop)"""
    ts, d = _walk_table()
    return float(np.interp(T, ts, d))


def stride(T):
    """0..1 walk intensity (smooth ramps)"""
    a = clamp((T - WALK0) / 0.5)
    b = clamp((WALK1 - T) / 0.5)
    return (a * a * (3 - 2 * a)) * (b * b * (3 - 2 * b))


_CH = {}


def _ch():
    if "c" not in _CH:
        _CH["c"] = Character("crow", 2, render="ink", wet=True, mud=0.3)
    return _CH["c"]


def pole_at(T):
    """pole butt (x, y) and angle in world units"""
    cr, fl = spec()
    bx, by = fl["base"]
    a1 = fl["ang"]
    if T < RAISE:
        return (bx, by, -1.05)
    if T < RAISE + 0.5:
        u = ease_out_back(clamp((T - RAISE) / 0.34), 1.6)
        return (bx, by, -1.05 + (a1 + 1.05) * u)
    s = stride(T)
    ph = _phase(T)
    lift = 26.0 * s
    # the butt swings with the stride; planted again when he stops (a small thump)
    x = bx + 22 * s * math.sin(2 * math.pi * ph)
    y = by - lift - 6 * s * abs(math.sin(2 * math.pi * ph))
    a = a1 + 0.03 * s * math.sin(2 * math.pi * ph + 0.6)
    return (x, y, a)


def _phase(T):
    """walk cycle phase: steps land on the walked distance (no foot sliding)"""
    cr, _ = spec()
    step = _ch().ground_speed(cr["h"], "walk") * 1.0     # distance per cycle ~ speed * cycle time
    return (walked(T) / max(step, 1e-3)) * 0.93


# ------------------------------------------------------------------ camera
def camera(T):
    """(z, S0x, S0y): screen = (world - W0) * z + S0"""
    cr, _ = spec()
    W0 = (cr["x"], cr["y"])
    head_dy = -0.9 * cr["h"]
    walk = (0.56, 1150.0, 1012.0)
    walk2 = (0.66, 1130.0, 1040.0)
    close = (1.5, 980.0, 540.0 - head_dy * 1.5)
    final = (1.0, W0[0], W0[1])
    if T <= 150.0:
        return walk
    if T <= PUSH0:
        u = ease_in_out(clamp((T - 150.0) / (PUSH0 - 150.0)), 2)
        return tuple(lerp(a, b, u) for a, b in zip(walk, walk2))
    if T <= PUSH1:
        u = ease_in_out(clamp((T - PUSH0) / (PUSH1 - PUSH0)), 3)
        z = math.exp(lerp(math.log(walk2[0]), math.log(close[0]), u))
        return (z, lerp(walk2[1], close[1], u), lerp(walk2[2], close[2], u))
    if T <= PULL0:
        d = (T - PUSH1) * 0.02
        return (close[0] * (1 + d), close[1], close[2] + d * 400)
    u = ease_in_out(clamp((T - PULL0) / (HOLD - PULL0)), 3)
    c0 = (close[0] * (1 + (PULL0 - PUSH1) * 0.02), close[1], close[2] + (PULL0 - PUSH1) * 0.02 * 400)
    z = math.exp(lerp(math.log(c0[0]), math.log(1.0), u))
    return (z, lerp(c0[1], final[1], u), lerp(c0[2], final[2], u))


WALKCAM = (0.56, 1150.0, 1012.0)


def cam_matrix(T, outer=None):
    """cairo matrix: world (t08 opening-frame units) -> screen design px"""
    cr, _ = spec()
    z, sx, sy = camera(T)
    m = cairo.Matrix(z, 0, 0, z, sx - cr["x"] * z, sy - cr["y"] * z)
    if outer is not None:
        m = m.multiply(outer)
    return m


def layer_matrix(T, p, outer=None):
    """cairo matrix for a parallax layer authored in WALK-SCREEN units (the frame as seen during the walk:
    his feet at (1150, 1012), 578 px tall). p = parallax (1 = ground at his feet, 0.03 = sky)."""
    z, sx, sy = camera(T)
    zw, wx, wy = WALKCAM
    k = 1 + (z / zw - 1) * p
    ax = wx + (sx - wx) * p
    ay = wy + (sy - wy) * p
    sc = walked(T) * zw * p
    m = cairo.Matrix(k, 0, 0, k, ax + (sc - wx) * k, ay - wy * k)
    if outer is not None:
        m = m.multiply(outer)
    return m


# ------------------------------------------------------------------ setup: the flag
def setup(S):
    cr, fl = spec()
    st = {}
    L = fl["pole"]

    def pole_fn(i):
        T = (F_TRACK0 + i) / 24.0
        return pole_at(T)

    def wind_fn(i):
        T = (F_TRACK0 + i) / 24.0
        s = stride(T)
        gust = 60 * math.sin(T * 1.7) + 40 * math.sin(T * 3.1 + 1)
        # walking into still air = the cloth streams back (right); in the empty years it hangs heavier
        base = fl["wind"][0]
        return (base + 260 * s + gust * s, -20.0)

    key = f"t07_crow_flag_v3_{F_TRACK0}_{N_TRACK}"
    path = CACHE / "t07" / f"{key}.npz"
    try:
        from vx.flag import FlagTrack
        tr = FlagTrack.load(path)
    except Exception:
        tr = Flag(pole=L, seed=fl["seed"], side=1, turbulence=fl["turbulence"], flutter=fl["flutter"]).simulate(
            N_TRACK, pole_fn, wind_fn, fps=24, warmup=48)
        path.parent.mkdir(parents=True, exist_ok=True)
        tr.save(path)
    st["track"] = tr
    try:
        tr.save(CACHE / "t07" / "crow_flag.npz")
    except Exception:
        pass
    return st


def events(S, st):
    ev = []
    for t, side in _ch().footsteps(WALK0, WALK1 + 0.3, "walk"):
        T = t
        if stride(T) < 0.2:
            continue
        ev.append(dict(t=T, kind="step", strength=0.6 + 0.2 * stride(T), pan=0.2,
                       desc="walker footstep (" + _surface(T) + ")"))
    ev.append(dict(t=WALK1 + 0.05, kind="thunk", strength=0.7, pan=0.2, desc="pole butt planted in the snow"))
    for tw, st_, kind in WIPES:
        ev.append(dict(t=tw, kind="whoosh", strength=0.5, pan=0.0, desc=f"foreground {kind} passes: {st_}"))
    ev.append(dict(t=UNMASK, kind="cloth", strength=0.6, pan=0.0, desc="he pulls the face mask down"))
    ev.append(dict(t=PULL0, kind="reprint", strength=0.8, pan=0.0, desc="the engraving reprints into halftone"))
    return ev


def _surface(T):
    s = station_at(T, 1150)
    return {"playa": "dust", "forest": "forest floor", "lake": "gravel", "nofire": "dirt",
            "years": "snow" if not (157.3 < T < 158.1) else "leaves"}.get(s, "dirt")


def station_at(T, x):
    """which station shows at screen x (the wipers sweep left -> right)"""
    cur = FIRST
    for tw, name, _ in WIPES:
        sx = _wiper_x(T, tw)
        if sx is None:
            if T > tw:
                cur = name
            continue
        if x < sx:
            cur = name
    return cur


def _wiper_x(T, tw, span=0.55):
    u = (T - tw) / span
    if u < -1 or u > 1:
        return None
    return 1130 + u * 1500


# ------------------------------------------------------------------ station drawings (grey, world units)
from scenes.t07_stations import draw_layer, draw_wiper, draw_particles, LAYERS  # noqa: E402


def _layer_rgba(fc, T, lname, parallax, outer=None, clip=None):
    cv = Canvas(fc)
    ctx = cv.ctx
    if clip is not None:
        ctx.rectangle(*clip)
        ctx.clip()
    m = layer_matrix(T, parallax, outer)
    segs = _segments(T)
    for (x0, x1, name) in segs:
        ctx.save()
        if outer is not None:
            ox0, _ = outer.transform_point(x0, 0)
            ox1, _ = outer.transform_point(x1, 0)
            x0, x1 = ox0, ox1
        ctx.rectangle(x0, -4000, x1 - x0, 9000)
        ctx.clip()
        ctx.transform(m)
        draw_layer(ctx, name, lname, T, walked(T) * WALKCAM[0] * parallax)
        ctx.restore()
    return cv.rgba(), m


def _segments(T):
    """screen-x intervals per station (in screen design units; the wiper covers each seam)"""
    xs = [(-10000.0, "L")]
    cur = FIRST
    out = []
    edges = []
    for tw, name, _ in WIPES:
        sx = _wiper_x(T, tw)
        if sx is None:
            if T > tw:
                cur = name
        else:
            edges.append((sx, name))
    if not edges:
        return [(-10000.0, 10000.0, cur)]
    # left of a wiper = the new station
    edges.sort()
    left = -10000.0
    names = [e[1] for e in edges]
    for (sx, name) in edges:
        out.append((left, sx, name))
        left = sx
    out.append((left, 10000.0, cur))
    return out


def world_frame(fc, T, outer=None, clip=None, grey_only=False):
    """printed walker world (engraving) -> (rgb, flags_rgba, front_rgba_printed, grey_world)"""
    cr, fl = spec()
    layers = []
    grey_acc = None
    for lname, par in LAYERS:
        rgba, m = _layer_rgba(fc, T, lname, par, outer, clip)
        layers.append((lname, rgba, m))
    # the walker himself (back layer), glued to his own page
    wv = Canvas(fc)
    if clip is not None:
        wv.ctx.rectangle(*clip)
        wv.ctx.clip()
    mw = cam_matrix(T, outer)
    wv.ctx.save()
    wv.ctx.transform(mw)
    walker_draw(wv.ctx, T, "back")
    wv.ctx.restore()
    wv.ctx.transform(layer_matrix(T, 1.0, outer))
    draw_particles(wv.ctx, T, walked(T) * WALKCAM[0])
    layers.append(("walker", wv.rgba(), mw))
    return layers, mw


def walker_kw(T):
    cr, fl = spec()
    kw = dict(cr["kw"])
    kw.pop("t", None)
    kw.pop("facing", None)
    s = stride(T)
    bx, by, a = pole_at(T)
    d = 685.0
    grip = (bx + d * math.sin(a), by - d * math.cos(a))
    kw["hand_r"] = grip
    kw["shape_r"] = "grip"
    kw["pole_ang"] = a
    kw["prop"] = None
    kw["turn"] = 0.85 * (1 - _turn_front(T)) + 0.35 * _turn_front(T)
    mask = 1.0
    if T > UNMASK:
        mask = 1.0 - 0.6 * ease_in_out(clamp((T - UNMASK) / 0.22))
    kw["face_mask"] = mask
    # the back hand: swings with the walk, rises to the mask, then rests like t08's
    rest = kw.get("hand_l")
    if UNMASK - 0.22 < T < UNMASK + 0.5:
        u = clamp((T - (UNMASK - 0.22)) / 0.2)
        dn = clamp((T - UNMASK) / 0.22)
        mouth = (cr["x"] - 60, cr["y"] - 0.86 * cr["h"] + 70 * dn)
        kw["hand_l"] = mouth
        kw["shape_l"] = "pinch"
    elif T >= UNMASK + 0.5:
        u = ease_in_out(clamp((T - UNMASK - 0.5) / 0.4))
        a0 = (cr["x"] - 60, cr["y"] - 0.86 * cr["h"] + 70)
        kw["hand_l"] = (lerp(a0[0], rest[0], u), lerp(a0[1], rest[1], u)) if rest else None
    else:
        kw.pop("hand_l", None)
        kw["shape_l"] = "relax"
    # expression: grave while walking; the reveal = a knowing, sardonic look to camera; then t08's grave
    if UNMASK < T < PULL0 + 0.25:
        kw["expr"] = {"sardonic": 1.0}
        kw["look"] = (-0.3, 0.0)
    kw["lean"] = 0.08 * s + kw.get("lean", 0.0) * (1 - s)
    return kw


def _turn_front(T):
    """head/body turn toward camera around the unmask"""
    return clamp((T - (UNMASK - 0.35)) / 0.3) * (1 - clamp((T - (PULL0 + 0.1)) / 0.45))


def walker_draw(ctx, T, layer):
    cr, fl = spec()
    s = stride(T)
    ph = _phase(T)
    kw = walker_kw(T)
    pose = {"walk": s, "stand": 1 - s} if 0 < s < 1 else ("walk" if s >= 1 else cr["pose"])
    if s > 0:
        kw["phase"] = ph % 1.0
    _ch().draw(ctx, cr["x"], cr["y"], cr["h"], pose, T, facing=-1, layer=layer, **kw)


# ------------------------------------------------------------------ render
def _print_layers(fc, layers, T, reprint=0.0, pmap=None):
    """engrave every layer on its own page and stack them"""
    out = ink.paper(fc)
    for lname, rgba, m in layers:
        if rgba[..., 3].max() <= 0:
            continue
        lay = ink.print_layer(fc, rgba, "engrave", space="page", matrix=m, **LINE_KW.get(lname, {}))
        out = lay[..., :3] + out * (1 - lay[..., 3:4])
    return out


def _pmap(fc, T, mw):
    """the reprint front: a wave spreading from his face (0 engraving -> 1 tract halftone)"""
    cr, _ = spec()
    hx, hy = mw.transform_point(cr["x"] - 20, cr["y"] - 0.88 * cr["h"])
    u = clamp((T - PULL0 + 0.04) / (HOLD - PULL0 - 0.05))
    R = (u ** 1.3) * 2700.0
    ys, xs = np.mgrid[0:fc.h, 0:fc.w].astype(np.float32)
    d = np.hypot(xs / fc.s - hx, (ys / fc.s - hy) * 1.15)
    # a ragged ink front (not a perfect circle)
    ang = np.arctan2(ys / fc.s - hy, xs / fc.s - hx)
    d = d + 60 * np.sin(ang * 5 + 1.3) + 35 * np.sin(ang * 11 + 0.4)
    return np.clip((R - d) / 260.0 + 0.5, 0, 1).astype(np.float32)


def _camp_matrix(T, outer=None):
    """t08's page -> my screen (their dots glued to their page through my pull-back)"""
    cam = C8._open_cam()
    kw = C8.print_kw(cam)
    ox, oy = kw["origin"]
    z = kw["zoom"]
    m = cairo.Matrix(z, 0, 0, z, -ox * z, -oy * z).multiply(cam_matrix(T, outer))
    return m, kw


def _reprint_layer(fc, rgba, pmap, m, kw):
    a = np.clip(rgba[..., 3:4], 0, 1)
    if a.max() <= 0:
        return None
    rgb = np.where(a > 1e-4, rgba[..., :3] / np.maximum(a, 1e-4), 1.0)
    out = ink.reprint(fc, rgb, "engrave", "tract", pmap, space="page", matrix=m, **kw)
    return np.concatenate([out * a, a], -1).astype(np.float32)


def render(fc, st, outer=None, clip=None):
    T = fc.T
    if T >= HOLD and outer is None:
        return _handoff(fc, st, T)
    layers, mw = world_frame(fc, T, outer, clip)
    if T >= PULL0:
        img, flags = _render_reprint(fc, st, T, layers, mw)
        top = Canvas(fc)
        _letters(top.ctx, fc, T, st)
        return over(img, comic.protect(top.rgba(), flags))
    img = _print_layers(fc, layers, T)
    # the Flag: spot plate
    fl = Canvas(fc)
    if clip is not None:
        fl.ctx.rectangle(*clip)
        fl.ctx.clip()
    fl.ctx.transform(mw)
    tr = st["track"]
    i = clamp((T * 24.0) - F_TRACK0, 0, tr.n - 1)
    tr.draw(fl.ctx, i, light=(-0.35, -0.8, 0.45))
    flags = fl.rgba()
    img = ink.spot(img, flags)
    # front: his gripping hand over the pole + the foreground wipers
    fr = Canvas(fc)
    if clip is not None:
        fr.ctx.rectangle(*clip)
        fr.ctx.clip()
    fr.ctx.save()
    fr.ctx.transform(mw)
    walker_draw(fr.ctx, T, "front")
    fr.ctx.restore()
    for tw, name, kind in WIPES:
        sx = _wiper_x(T, tw)
        if sx is not None:
            draw_wiper(fr.ctx, kind, sx, T, outer)
    lay = ink.print_layer(fc, fr.rgba(), "engrave")
    img = lay[..., :3] + img * (1 - lay[..., 3:4])
    if outer is not None:
        return img, flags
    # lettering
    top = Canvas(fc)
    _letters(top.ctx, fc, T, st)
    img = over(img, comic.protect(top.rgba(), flags))
    return img


def _flag_layer(fc, st, T, mw):
    fl = Canvas(fc)
    fl.ctx.transform(mw)
    tr = st["track"]
    i = clamp((T * 24.0) - F_TRACK0, 0, tr.n - 1)
    tr.draw(fl.ctx, i, light=(-0.35, -0.8, 0.45))
    return fl.rgba()


def _render_reprint(fc, st, T, layers, mw):
    """158.8+: the engraving re-screens itself into tract halftone while the present-day camp (t08's world)
    floods in from his face outwards."""
    pm = _pmap(fc, T, mw)
    mc, kw = _camp_matrix(T)
    tkw = dict(lpi=kw.get("lpi"), dot_angle=kw.get("angle"))
    tkw = {k: v for k, v in tkw.items() if v is not None}
    out = ink.paper(fc)
    walker = None
    for lname, rgba, m in layers:
        if lname == "walker":
            walker = (rgba, m)
            continue
        lay = _reprint_layer(fc, rgba, pm, m, tkw)
        if lay is not None:
            out = lay[..., :3] + out * (1 - lay[..., 3:4])
    # the camp: t08's grey world (+ rain) at their opening camera = my world units
    cv = Canvas(fc)
    cv.ctx.transform(cam_matrix(T))
    C8.draw_camp(cv.ctx, T)
    C8.draw_rain(cv.ctx, T)
    camp = cv.rgba()
    camp = camp * pm[..., None]
    lay = _reprint_layer(fc, camp, np.ones_like(pm), mc, dict(tkw, spacing=6.0 / kw["zoom"]))
    if lay is not None:
        out = lay[..., :3] + out * (1 - lay[..., 3:4])
    if walker is not None:
        lay = _reprint_layer(fc, walker[0], pm, mc, dict(tkw, spacing=6.0 / kw["zoom"]))
        if lay is not None:
            out = lay[..., :3] + out * (1 - lay[..., 3:4])
    flags = _flag_layer(fc, st, T, mw)
    out = ink.spot(out, flags)
    fr = Canvas(fc)
    fr.ctx.transform(mw)
    walker_draw(fr.ctx, T, "front")
    lay = _reprint_layer(fc, fr.rgba(), pm, mc, dict(tkw, spacing=6.0 / kw["zoom"]))
    if lay is not None:
        out = lay[..., :3] + out * (1 - lay[..., 3:4])
    return out, flags


def _handoff(fc, st, T):
    """the last frames ARE t08's opening frame (their print, front and page top) with our Flag"""
    from scenes import t08_blaspheme as B8
    L = B8.layers(fc, END)
    flags = _flag_layer(fc, st, T, cam_matrix(T))
    img = ink.spot(L["print"], flags)
    if L["front"] is not None:
        img = over(img, L["front"])
    return over(img, comic.protect(L["top"], flags))


def _letters(ctx, fc, T, st):
    tr = st["track"]

    def avoid(Tq):
        z, sx, sy = camera(Tq)
        return [(sx - 60, 0, 1920, sy - 0.2 * 1033 * z)]

    for lid in ("P22", "P23"):
        comic.say(ctx, fc, lid, 44, 40, w=860, anchor="tl", avoid=None)


def render_cell100(img, fc, rect, T, t):
    """panel #100 on the page: the walker world squeezed into the panel (virtual 1920-wide screen)"""
    x0, y0, x1, y1 = rect
    k = (x1 - x0) / 1920.0
    oy = (y0 + y1) / 2 - 540.0 * k
    outer = cairo.Matrix(k, 0, 0, k, x0, oy)
    clip = (x0, y0, x1 - x0, y1 - y0)
    st = _STATE.get("st")
    if st is None:
        return
    fc2 = fc
    sub, flags = render(fc2, st, outer=outer, clip=clip)
    X0, Y0 = max(0, int(math.floor(x0 * fc.s))), max(0, int(math.floor(y0 * fc.s)))
    X1, Y1 = min(fc.w, int(math.ceil(x1 * fc.s))), min(fc.h, int(math.ceil(y1 * fc.s)))
    if X1 > X0 and Y1 > Y0:
        img[Y0:Y1, X0:X1] = sub[Y0:Y1, X0:X1]


_STATE = {}
# Doré: horizontal ruled skies, gently bent lines on the land, form-following hatching on the figure
LINE_KW = {"sky": dict(angle=0.0, follow=0.25), "far": dict(angle=-0.05, follow=0.5),
           "ground": dict(angle=0.04, follow=0.6)}


def bind(st):
    _STATE["st"] = st


def post(fc, st):
    return dict(vignette=0.14)

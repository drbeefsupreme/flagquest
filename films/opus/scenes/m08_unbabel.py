"""m08 · THE UNBABELING (146.0-163.0) · owner M08

It opens on m07's last frame: the bare wall after Babel (lime plaster, the red sinopia of the fallen programme, m07's
81k fallen tesserae in heaps along its foot, one votive candle) - m07's own ruin, camera and candle (scenes.m07_ruin,
read-only), crossfaded from m07's last frame over the first 6 frames. A breath of wind bends the flame (146.0).
Crocus and Lutie climb out of the dark onto the rubble, the old man leaning on the furled Flag like a crozier; he
raises it and drives it into the slope of the tower's heap (149.0 flag_planted, thud): the cloth stirs, opens, flies -
whole and yellow, the only unbroken thing in the world, and it gives light. C05 "A Flag needs no translation. It means
only... Flag." - while he speaks the plaster receives a NEW sinopia: compass arcs drawn out of the Flag's foot.
155.0 tiles_rise: the fallen tesserae rise like an avalanche run backwards and set themselves into a new picture, the
Rose of the Peoples: every niche's tiles fly as one flock and set one person, alone in their own small space, beside
their own real Flag ("Each of them alone."); then the gold is swept up by one rolling wave out of the Flag's foot that
binds them all ("All of them together."); the last tile lands at 159.0 (convergence). Glory; tilt up into the dome
(whip hand-off to m09 at 163.0: content moving down ~3000 px/s, ~150 px vertical blur).

Titulus: SINGVLI · ET · VNIVERSI. THE ONE RULE: the Flags (vx.flag) are composited last; nothing touches the cloth.
Helpers: m08_stage (world, times, camera, lights, dust), m08_rose (the new picture), m08_rise (the tiles'
choreography), m08_pair (Crocus and Lutie).
"""
import math

import cairo
import cv2
import numpy as np

from vx import *
from vx import mosaic as mz
from vx import foley
from vx.config import CACHE
from vx.flag import Flag, FlagTrack, draw_flag_field

from scenes import m07_ruin as R
from scenes import m08_stage as st
from scenes import m08_pair as pair
from scenes.m08_rose import Rose, Upper, canvas as rose_canvas, part_map, GOLD as R_GOLD, UPPER as R_UPPER
from scenes.m08_rise import Rise, T_RISE, T_CONV

POST = dict(bloom=0.21, bloom_thresh=0.66, bloom_radius=26.0, vignette=0.32, grain=0.0)   # m07's silence look, but no grain: nothing over the cloth

SC = mz.SC
SH = st.M07_SHIFT                    # m07 world -> m08 world
HERO_LIGHT = (-0.3, -0.45, 0.85)     # the planted Flag faces the light it makes
Z_SPLIT = st.CROCUS_Z - 25.0         # loose tiles nearer than this are set after the figures
XFADE = 6                            # frames of crossfade from m07's last frame


# ====================================================================== setup
def build_wall(S):
    """the new picture: rose chart, tile layout (v1 andamento: gold ground in rings about the Flag's foot, niches
    and letters in smaller tesserae, the people in opus vermiculatum), per-tile colour and region (cached)"""
    rose = Rose()
    lpath, path = S.cache / "rose_v6_lay.npz", S.cache / "rose_v6.npz"
    if lpath.exists() and path.exists():
        z = np.load(path)
        return rose, mz.Layout.load(lpath), z["part"], z["rgb"]
    sg = 0.75
    guide = rose_canvas(rose, sg).rgb()
    fine = rose_canvas(rose, sg, "fine").rgba()[..., 3] > 0.5
    gold = rose_canvas(rose, sg, "gold").rgba()[..., 3] > 0.5
    parts = part_map(rose, sg)
    lay = mz.Layout.flow(guide, tile=13, seed=5, extent=rose.ext, fine=fine, fine_tile=10.0, gold=gold,
                         ground=parts == R_GOLD, ground_style="fan", fan_center=(rose.cx, rose.cy),
                         density=rose.density, groups=parts + 16)
    part = (lay.grp - 16).astype(np.int32)
    rgb = mz.sample(rose_canvas(rose, 1.0).rgb(), lay).astype(np.float32)
    lay.save(lpath)
    np.savez_compressed(path, part=part, rgb=rgb)
    return rose, lay, part, rgb


def build_upper(S, rose):
    up = Upper(rose)
    lpath, path = S.cache / "upper_v3_lay.npz", S.cache / "upper_v3.npz"
    if lpath.exists() and path.exists():
        return up, mz.Layout.load(lpath), np.load(path)["rgb"]
    guide = rose_canvas(up, 0.3).rgb()
    gold = rose_canvas(up, 0.3, "gold").rgba()[..., 3] > 0.5
    lay = mz.Layout.flow(guide, tile=24, seed=9, extent=up.ext, gold=gold)
    rgb = mz.sample(rose_canvas(up, 0.3).rgb(), lay).astype(np.float32)
    lay.save(lpath)
    np.savez_compressed(path, rgb=rgb)
    return up, lay, rgb


def load_heap():
    """m07's fallen tesserae at rest (their hand-off), moved into m08's world, with their own material randomness
    (m07's layout) so they look exactly as in m07's last frame"""
    z = np.load(st.HANDOFF)
    lay07 = R.load_ruin().lay
    pos = z["heap_pos"].astype(np.float64) + SH
    if len(lay07) != len(pos):
        raise RuntimeError("m07 hand-off and m07 ruin layout disagree - rebuild m07's cache")
    size = z["heap_size"].astype(np.float32)
    mat = z["heap_mat"].astype(np.uint8)
    return dict(pos=pos, rot=z["heap_rot"].astype(np.float64), size=size, rgb=z["heap_rgb"].astype(np.float32),
                mat=mat, gold=(mat == mz.GOLD).astype(np.float32),
                poly_off=(z["heap_poly_home"] - z["home_xy"][:, None, :]).astype(np.float32),
                nv=z["heap_nv"].astype(np.int8), tilt=lay07.tilt, jit=lay07.jit, rnd=lay07.rnd)


def candle_tiles():
    """m07's bronze stand + beeswax taper (set in tesserae), moved into m08's world"""
    ct = R.candle_tiles()
    d = dict(ct) if isinstance(ct, dict) else dict(vars(ct))
    sh = SH[:2].astype(np.float32)
    kw = {k: d[k] for k in ("ang", "size", "rgb", "mat", "z", "rot", "alpha", "gain", "emit", "present", "nv",
                            "tilt", "jit", "rnd", "ids") if d.get(k) is not None}
    return mz.Tiles(np.asarray(d["xy"], np.float32) + sh, poly=np.asarray(d["poly"], np.float32) + sh, **kw)


def pole_track(S, crocus):
    """the Flag's pole foot per frame, from Crocus's hands (vx.icon anchors) until it strikes the heap; then planted
    (sunk a little into the tesserae) with a quiver"""
    T = S.start + np.arange(S.n) / S.fps
    out = np.zeros((S.n, 3))
    planted = None
    for i, t in enumerate(T):
        if t < st.T_PLANT:
            x, kw = pair.crocus_kw(t)
            out[i] = crocus.anchors(x, st.ground_y(x), kw)["pole"]
        else:
            if planted is None:
                x, kw = pair.crocus_kw(st.T_PLANT - 1e-4)
                planted = np.array(crocus.anchors(x, st.ground_y(x), kw)["pole"], np.float64)
            sink = 22.0 * smoothstep(st.T_PLANT, st.T_PLANT + 0.06, t)
            out[i] = (planted[0], planted[1] + sink, planted[2] - 0.035 + st.planted_quiver(t))
    return out


def simulate_flag(S, poles):
    path = S.cache / "flag_v2.npz"
    if path.exists():
        tr = FlagTrack.load(path)
        if tr.n == S.n and np.allclose(tr.poles, poles, atol=1e-3):
            return tr
    T = S.start + np.arange(S.n) / S.fps
    fl = Flag(pole=st.POLE_LEN, seed=21, side=1, turbulence=1.1)
    tr = fl.simulate(S.n, lambda i: tuple(poles[i]), lambda i: (st.wind(T[i]), -20.0, 0.0),
                     furl_fn=lambda i: st.furl(T[i]), warmup=24)
    tr.save(path)
    return tr


def setup(S):
    rose, lay, part, rgb = build_wall(S)
    ox, oy = rose.origin
    wall_xy = lay.xy.astype(np.float64) + (ox, oy)
    cells = [dict(k=c.k, th=c.th, centre=rose.to_wall(*rose.local(c, 0.0, -(c.ro - c.ri) * 0.5)))
             for c in rose.cells]
    heap = load_heap()
    cx, cy = rose.to_wall(rose.cx, rose.cy)
    # the windows' zone above the cornice never fell (it is never seen bare): only the rest rises. Every fallen
    # tessera rises: the surplus doubles up on sockets of the gold ground (landing together with their twin)
    static = part == R_UPPER
    mov = np.nonzero(~static)[0]
    extra = len(heap["pos"]) - len(mov)
    if extra > 0:
        mov = np.concatenate([mov, np.random.default_rng(17).choice(np.nonzero(part == R_GOLD)[0], extra,
                                                                     replace=False)])
    # the gold front reaches the edges of the frame exactly at 159.0 (convergence)
    Hi = np.linalg.inv(st.camera(T_CONV).wall_homography())
    wc = (Hi @ np.array([[0, 0, 1], [1920, 0, 1], [0, 1080, 1], [1920, 1080, 1]], np.float64).T).T
    rho_vis = float(np.max(np.hypot(wc[:, 0] / wc[:, 2] - cx, wc[:, 1] / wc[:, 2] - cy)))
    rise = Rise(dict(xy=wall_xy[mov], part=part[mov]), cells, heap, (cx, cy),
                plant=(st.PLANT[0], st.PLANT[1], st.PLANT[2], st.T_PLANT), rho_vis=rho_vis)
    crocus = pair.Actor("crocus", st.CROCUS_H, st.CROCUS_Z, tile=9.0, fine_tile=4.6, K=2.6)
    crocus.build(S.cache / "crocus_lay_v4.npz", pair.CROCUS_POSES, yg=st.ground_y(st.CROCUS_REST_X))
    lutie = pair.Actor("lutie", st.LUTIE_H, st.LUTIE_Z, tile=9.0, fine_tile=4.6, K=2.6)
    lutie.build(S.cache / "lutie_lay_v3.npz", pair.LUTIE_POSES)
    poles = pole_track(S, crocus)
    flag = simulate_flag(S, poles)
    upper, lay_u, rgb_u = build_upper(S, rose)
    last = cv2.imread(str(CACHE / "m07" / "handoff_last.png"), cv2.IMREAD_COLOR)
    state = dict(rose=rose, lay=lay, part=part, rgb=rgb, heap=heap, rise=rise, flag=flag, poles=poles,
                 centre=(cx, cy), upper=upper, lay_u=lay_u, rgb_u=rgb_u, crocus=crocus, lutie=lutie,
                 plane=mz.Plane(origin=(ox, oy, 0.0)), plane_u=mz.Plane(origin=(upper.origin[0], upper.origin[1], 0.0)),
                 static=static, mov=mov, cache={}, dust=st.Dust(), tl=S.tl, backdrop=R.load_ruin().backdrop,
                 candle=candle_tiles(), m07_last=(last[..., ::-1] / 255.0).astype(np.float32))
    export_sound(S, state)
    return state


# ====================================================================== sound
def export_sound(S, state):
    rise = state["rise"]
    ev = [dict(t=146.0, kind="wind", strength=0.25, pan=0.3,
               desc="faint breath of wind; the candle flame bends (to 146.5)")]
    for t in st.steps(st.T_WALK[0], st.T_WALK[1], st.CROCUS_X0, st.CROCUS_PLANT_X):
        ev.append(dict(t=t, kind="step", strength=0.45, pan=foley.screen_pan(st.crocus_x(t)),
                       desc="heap: Crocus climbs the fallen tesserae, the Flag's pole taps like a crozier"))
    for t in st.steps(st.T_WALK[0] + 0.25, st.T_WALK[1] + 0.35, st.LUTIE_X0, st.LUTIE_X1, rate=2.2):
        ev.append(dict(t=t, kind="step", strength=0.3, pan=foley.screen_pan(st.lutie_x(t)),
                       desc="heap: Lutie's lighter steps"))
    ev.append(dict(t=st.T_PLANT0 + 0.15, kind="whoosh", strength=0.3, pan=0.15,
                   desc="Crocus raises the furled Flag overhead"))
    ev.append(dict(t=st.T_PLANT, kind="plant", strength=1.0, pan=0.18,
                   desc="THE PLANTING: the pole's foot drives into the heap - thud, tiles jump"))
    ev.append(dict(t=st.T_PLANT + 0.02, kind="scatter", strength=0.6, pan=0.2,
                   desc="tesserae scatter down the slope (to 149.6)"))
    ev.append(dict(t=st.T_UNFURL[0], kind="unfurl", strength=0.6, pan=0.3,
                   desc="the cloth unrolls from the pole and opens in the wind (to 149.85)"))
    ev.append(dict(t=153.4, kind="tremble", strength=0.5, pan=0.0,
                   desc="the heaps tremble and chatter, growing (to 155.0)"))
    ev.append(dict(t=T_RISE, kind="rise", strength=1.0, pan=0.0,
                   desc="tiles_rise: the fallen tesserae leave the floor (reversed avalanche)"))
    ev.append(dict(t=157.0, kind="alone", strength=0.6, pan=0.0,
                   desc="'alone': the niches complete one by one, each Flag pops open"))
    ev.append(dict(t=157.85, kind="together", strength=0.9, pan=0.0,
                   desc="'All ... together': the gold wave rolls out of the Flag's foot (to 159.0)"))
    ev.append(dict(t=T_CONV, kind="convergence", strength=1.0, pan=0.0,
                   desc="the last tiles land in one wave: the picture is complete"))
    ev.append(dict(t=T_CONV, kind="complete", strength=1.0, pan=0.0, desc="complete"))
    tl = np.sort(rise.t_land)
    for q in np.arange(0.1, 1.0001, 0.1):
        k = min(len(tl) - 1, int(q * len(tl)) - 1)
        ev.append(dict(t=float(tl[k]), kind="land", strength=float(q), pan=0.0,
                       desc=f"{int(round(q * 100))}% of the new picture set"))
    fx = state["rose"].flags["xs"]
    for c, tdone in enumerate(rise.cell_done):
        ev.append(dict(t=float(tdone + 0.04), kind="flag_pop", strength=0.35,
                       pan=foley.screen_pan(fx[c] * 0.7 + 1150 * 0.3), desc="a niche is complete: its Flag springs open"))
    ev.append(dict(t=st.T_TILT[0], kind="camera", strength=0.7, pan=0.0, desc="tilt up (to 163.0): whip into the dome"))
    foley.export_events(S, "sfx", ev)
    tr = rise.foley(S)
    foley.export_track(S, "rise", tr["energy"], tr["pan"], kind="tiles", lift=tr["lift"], land=tr["land"],
                       set=tr["set"])
    state["flag"].export_foley(S, "flag")
    T = S.start + np.arange(S.n) / S.fps
    pops = np.array([rise.pop(t).mean() for t in T], np.float32)
    wind = np.array([st.wind(t) for t in T], np.float32)
    foley.export_track(S, "field", pops * (0.55 + 0.25 * wind / 900.0), 0.0, kind="field",
                       count=pops * len(rise.cell_done))


# ====================================================================== lights
def _shift_light(L, d):
    if L.directional:
        return L
    return mz.Light(pos=tuple(L.p + d), color=tuple(L.color), power=L.power, radius=L.radius)


def mz_lights(state, T):
    """(lights, ambient) in m08's world: m07's votive candle, the planted Flag, every niche Flag that has popped,
    and the new world's own light that wakes with the gold wave. At 146.0 this is exactly m07's silence rig."""
    Ls = [_shift_light(R.candle_light(T, 1.0, st.gust(T)), SH)]
    fp = st.flag_light(T)
    if fp > 0:
        tr = state["flag"]
        x, y = tr.cloth_center(min(tr.n - 1, max(0, int(round((T - st.T0) * 24)))))
        Ls.append(mz.Light(pos=(x, y, st.flag_depth(T) + 30.0), color=C["flag_light"], power=1.1 * fp, radius=300.0))
    rise, rose = state["rise"], state["rose"]
    pop = rise.pop(T)
    for i in np.nonzero(pop > 0)[0]:
        Ls.append(mz.Light(pos=(rose.flags["xs"][i], rose.flags["ys"][i] - 55.0, 30.0), color=C["flag_light"],
                           power=0.35 * pop[i], radius=90.0))
    g = st.glory(T)
    if g > 0:
        Ls.append(mz.Light(dir=(-0.15, -0.35, 0.92), color=(1.0, 0.88, 0.7), power=1.05 * g))
    return Ls, 0.012 + 0.012 * smoothstep(146.2, 148.5, T) + 0.1 * g


def env_level(T):
    """brightness of the basilica reflected in the gold: m07's dark church (0.08), then the lit world"""
    return 0.08 + 0.2 * st.flag_light(T) + 0.85 * st.glory(T)


def sparkle(T):
    return lerp(2.25, 1.3, smoothstep(154.5, 158.0, T))          # m07's candle-lit glitter, then the wall's


def lights(state, T, Lm=None):
    """the same point lights for my own layers (dust, figure wrap, the Flag's exposure): [(pos, rgb, power, radius)]"""
    Lm = mz_lights(state, T)[0] if Lm is None else Lm
    return [(L.p.astype(np.float64), L.color.astype(np.float64), L.power, L.radius) for L in Lm if not L.directional]


def light_at(state, P, T, Ls=None):
    return st.light_at(P, T, lights(state, T) if Ls is None else Ls, state["centre"], state["rise"].rho_max)


# ====================================================================== render
def render(fc, state):
    T = fc.T
    cam = st.camera(T)
    view, view07 = cam.mz(), cam.mz(offset=-SH)
    Lm, amb = mz_lights(state, T)
    env, spk = env_level(T), sparkle(T)
    bg = state["backdrop"].render(fc, view07, [_shift_light(L, -SH) for L in Lm], amb, extra=sinopia_hook(state, T))
    img = wall_layer(fc, state, cam, view, T, Lm, amb, env, spk, bg)
    lt = loose_tiles(state, cam, T)
    img = loose_layer(fc, view, Lm, amb, env, spk, lt, True, img)
    img = pair_layer(fc, state, view, T, Lm, amb, env, img)
    img = loose_layer(fc, view, Lm, amb, env, spk, lt, False, img)
    img = mz.render_tiles(fc, state["candle"], view=view, light=Lm, ambient=amb, loose=True, bg=img, env=env,
                          sparkle=spk)
    top = Canvas(fc)
    R.draw_flame(top.ctx, view07, T, fc.s, gust=st.gust(T))
    img = top.over(img)
    img = dust_layer(fc, state, cam, T, Lm, img)
    img = flags_layer(fc, state, cam, view, T, Lm, amb, env, img)
    b = st.tilt_blur(T)
    if b > 0.5:
        img = cv2.blur(img, (1, max(1, int(b * fc.s))))
    if fc.f < XFADE:                                  # out of m07's last frame (both pre-post)
        a = (fc.f + 1) / (XFADE + 1.0)
        last = state["m07_last"]
        if last.shape[:2] != (fc.h, fc.w):
            last = cv2.resize(last, (fc.w, fc.h), interpolation=cv2.INTER_AREA)
        img = last * (1 - a) + img * a
    return img


def post(fc, state):
    k = smoothstep(146.0, 148.5, fc.T)
    return dict(bloom=lerp(0.21, 0.28, k) + 0.14 * st.glory(fc.T), contrast=1.0 + 0.04 * k)


# ---------------------------------------------------------------------- the wall
def draw_new_sinopia(ctx, rose, T):
    """the new design drawn in red ochre while Crocus speaks: compass arcs out of the Flag's foot, then the radial
    divisions, then quick figure contours (white strokes, rose chart units)"""
    t0 = 150.7
    ctx.set_source_rgb(1, 1, 1)
    cx, cy = rose.cx, rose.cy
    radii = [rose.r0 - rose.gap_r * 0.5]
    for k in range(rose.rows):
        radii += [rose.r0 + rose.pitch * k + rose.gap_r / 2, rose.r0 + rose.pitch * (k + 1) - rose.gap_r / 2]
    radii += [rose.rb0, rose.rb1]
    ctx.set_line_width(3.2)
    for i, r in enumerate(radii):
        a = seg(T, t0 + 0.12 * i, t0 + 0.12 * i + 0.9, ease_in_out)
        if a > 0:
            ctx.new_sub_path()
            ctx.arc(cx, cy, r, math.pi, math.pi + math.pi * a)
            ctx.stroke()
    nrow = {}
    for c in rose.cells:
        nrow[c.k] = nrow.get(c.k, 0) + 1
    for c in rose.cells:
        tt = t0 + 1.4 + 0.35 * c.k + 0.25 * (c.th + math.pi / 2) / math.pi
        a = seg(T, tt, tt + 0.35)
        if a <= 0:
            continue
        dth = math.pi / nrow[c.k]
        ctx.set_line_width(2.4)
        for e in (-1, 1):
            th = c.th + e * dth / 2
            ctx.move_to(*rose.at(c.ri, th))
            ctx.line_to(*rose.at(c.ri + (c.ro - c.ri) * a, th))
            ctx.stroke()
        a2 = seg(T, tt + 0.5, tt + 1.0)
        if a2 > 0:
            ctx.save()
            ctx.translate(*c.origin)
            ctx.rotate(c.th)
            hh, u = c.person["h"], c.person["u"]
            ctx.set_line_width(2.0)
            ctx.new_sub_path()
            ctx.arc(u, -9 - hh + hh * 0.1, hh * 0.09, -math.pi / 2, -math.pi / 2 + 2 * math.pi * a2)
            ctx.stroke()
            ctx.move_to(u - hh * 0.12, -9 - hh * 0.72)
            ctx.line_to(u - hh * 0.17, -9 - hh * 0.72 + hh * 0.7 * a2)
            ctx.move_to(u + hh * 0.12, -9 - hh * 0.72)
            ctx.line_to(u + hh * 0.17, -9 - hh * 0.72 + hh * 0.7 * a2)
            ctx.stroke()
            ctx.restore()


def sinopia_hook(state, T):
    """m07's Backdrop hook: paint the new sinopia into its (linear) plaster chart image, in m07's wall chart units"""
    if T < 150.7:
        return None

    def hook(fc_, cam_, img):
        h, w = img.shape[:2]
        cv = Canvas(s=w / R.WX, w=w, h=h)
        rose = state["rose"]
        cv.ctx.translate(rose.origin[0] - SH[0], rose.origin[1] - SH[1])      # rose chart -> m07 wall chart
        draw_new_sinopia(cv.ctx, rose, T)
        a = 0.75 * cv.rgba()[..., 3:4]
        return img * (1 - a + a * mz.srgb_to_lin(SC["sinopia"]))
    return hook


def wall_layer(fc, state, cam, view, T, Lm, amb, env, spk, bg):
    """the new picture setting itself (absent sockets show m07's bare wall), and the zone under the dome"""
    lay, rise, mov = state["lay"], state["rise"], state["mov"]
    landed = state["static"].copy()
    landed[mov] = rise.landed(T)
    gain = np.ones(len(lay), np.float32)
    gain[mov] = rise.gain(T)
    out = mz.render(fc, None, lay, rgb=state["rgb"], view=view, surface=state["plane"], present=landed, gain=gain,
                    holes="bg", bg=bg, light=Lm, ambient=amb, env=env, sparkle=spk)
    up = state["upper"]
    if cam.project(np.array([[1150.0, state["rose"].origin[1], 0.0]]))[0][0, 1] > 0:
        dome = np.clip((up.ext[1] - state["lay_u"].xy[:, 1]) / up.ext[1], 0, 1)
        ua = mz.render(fc, None, state["lay_u"], rgb=state["rgb_u"], view=view, surface=state["plane_u"],
                       light=Lm, ambient=amb, env=env, gain=1.0 + 0.9 * st.glory(T) * dome, return_alpha=True)
        out = over(out, ua)
    return out


# ---------------------------------------------------------------------- loose tesserae (heaps + flight)
def loose_tiles(state, cam, T):
    """every tessera not set in the wall: pose, shape, colour, material, light gain, glow and motion smear. A risen
    tile turns into its new tessera early in its flight (colour, material, cut)."""
    rise, heap, lay = state["rise"], state["heap"], state["lay"]
    L = rise.loose(T)
    idx = L["idx"]
    if len(idx) == 0:
        return None
    full = state["mov"][idx]
    h = rise.hidx[idx]
    fly = L["flying"]
    tau = np.clip(L["tau"], 0, 1)
    new = fly & (tau > 0.15)
    pos = L["pos"]
    rgb = np.where(new[:, None], state["rgb"][full], heap["rgb"][h])
    mat = np.where(new, lay.mat[full], heap["mat"][h]).astype(np.uint8)
    off = np.where(new[:, None, None], lay.poly[full] - lay.xy[full][:, None, :], heap["poly_off"][h])
    gold = (mat == mz.GOLD).astype(np.float32)
    # risen tiles carry the new light: a swell mid-flight (strongest in the gold wave) and a glow of their own
    swell = np.where(fly, np.sin(np.pi * tau), 0.0)
    gain = (1.0 + (0.8 + 2.2 * gold) * swell ** 0.7).astype(np.float32)
    glow = (swell ** 0.8 * (0.10 + 0.35 * gold)).astype(np.float32)
    tint = np.where(gold[:, None] > 0.5, mz.srgb_to_lin(SC["gold_l"])[None, :], mz.srgb_to_lin(rgb))
    # motion smear: a quarter of the frame's screen displacement, never longer than ~5 tiles
    stretch = np.zeros((len(idx), 2), np.float32)
    if np.any(fly):
        p1, _ = cam.project(pos[fly])
        p0, _ = cam.project(pos[fly] - L["vel"][fly] / 96.0)
        d = p1 - p0
        ln = np.linalg.norm(d, axis=1, keepdims=True)
        stretch[fly] = (d * np.minimum(1.0, 42.0 / np.maximum(ln, 1e-6))).astype(np.float32)
    return dict(xy=pos[:, :2], z=pos[:, 2], rot=L["rot"], poly=off + pos[:, None, :2],
                nv=np.where(new, lay.nv[full], heap["nv"][h]), rgb=rgb, mat=mat,
                size=np.where(new, lay.size[full], heap["size"][h]),
                tilt=np.where(new[:, None], lay.tilt[full], heap["tilt"][h]),
                jit=np.where(new[:, None], lay.jit[full], heap["jit"][h]),
                rnd=np.where(new, lay.rnd[full], heap["rnd"][h]), gain=gain, stretch=stretch,
                emit=(tint * glow[:, None]).astype(np.float32))


def loose_layer(fc, view, Lm, amb, env, spk, lt, back, img):
    """the loose tesserae behind (back=True) or in front of the pair's plane, depth-sorted over img"""
    if lt is None:
        return img
    m = lt["z"] < Z_SPLIT if back else lt["z"] >= Z_SPLIT
    if not np.any(m):
        return img
    t = mz.Tiles(lt["xy"][m], ang=0.0, size=lt["size"][m], rgb=lt["rgb"][m], mat=lt["mat"][m], z=lt["z"][m],
                 rot=lt["rot"][m], gain=lt["gain"][m], emit=lt["emit"][m], stretch=lt["stretch"][m],
                 poly=lt["poly"][m], nv=lt["nv"][m], tilt=lt["tilt"][m], jit=lt["jit"][m], rnd=lt["rnd"][m])
    return mz.render_tiles(fc, t, view=view, light=Lm, ambient=amb, loose=True, bg=img, env=env, sparkle=spk)


# ---------------------------------------------------------------------- Crocus and Lutie
def figure_lights(state, T, Lm, x, y, z, h):
    """a figure is a flat panel: light from the side would graze it. Add the light that reaches it (from every
    direction, the same falloff) as a soft frontal wrap, so the pair are lit like round bodies."""
    lum = float(np.max(light_at(state, np.array([[x, y - 0.6 * h, z]]), T)[0]))
    return Lm + [mz.Light(dir=(0.25, -0.3, 0.92), color=(1.0, 0.86, 0.64), power=2.2 * lum ** 1.5)]


def emerge(T):
    """the pair (and the carried Flag) are unseen in m07's silence; they come out of the dark as they walk"""
    return smoothstep(146.25, 147.3, T)


def darken(rgba, k):
    """premultiplied rgba with its colour scaled (a figure in darkness still hides what is behind it)"""
    if k >= 0.999:
        return rgba
    out = rgba.copy()
    out[..., :3] *= k
    return out


def pair_layer(fc, state, view, T, Lm, amb, env, img):
    xl, kwl = pair.lutie_kw(T)
    yl = st.ground_y(xl, st.LUTIE_Z)
    Ll = figure_lights(state, T, Lm, xl, yl, st.LUTIE_Z, st.LUTIE_H)
    vis = emerge(T)
    img = over(img, darken(state["lutie"].render(fc, view, xl, yl, kwl, Ll, amb, env=env), vis))
    xc, kwc = pair.crocus_kw(T, state["tl"])
    yc = st.ground_y(xc, st.CROCUS_Z)
    Lc = figure_lights(state, T, Lm, xc, yc, st.CROCUS_Z, st.CROCUS_H)
    img = over(img, darken(state["crocus"].render(fc, view, xc, yc, kwc, Lc, amb, part="back", env=env), vis))
    state["cache"]["front"] = (T, xc, yc, kwc, Lc)
    return img


# ---------------------------------------------------------------------- dust in the light
def dust_layer(fc, state, cam, T, Lm, img):
    pos, size, alpha = state["dust"].at(T)
    lum = st.light_at(pos, T, lights(state, T, Lm)[:2]).max(axis=1) - st.AMBIENT.max()
    a = np.clip(alpha * lum * 0.9, 0, 0.6)
    keep = a > 0.01
    if not np.any(keep):
        return img
    scr, zc = cam.project(pos[keep])
    k = cam.f / zc
    cv = Canvas(fc)
    ctx = cv.ctx
    for (x, y), r, aa in zip(scr, size[keep] * k, a[keep]):
        ctx.new_path()
        ctx.arc(x, y, max(0.6, r), 0, 2 * math.pi)
        ctx.set_source_rgba(1.0, 0.9, 0.72, float(aa))
        ctx.fill()
    return cv.over(img)


# ---------------------------------------------------------------------- THE FLAGS (last, whole cloth)
def flags_layer(fc, state, cam, view, T, Lm, amb, env, img):
    """The planted Flag and the niche Flags, whole cloth, composited last. Before the planted Flag starts to give
    light it is cloth in a dark church: the whole layer is dimmed uniformly by the light that reaches it
    (exposure, never a marking). Crocus's fists are set over the POLE afterwards, masked off the cloth."""
    tr = state["flag"]
    i = min(tr.n - 1.0, max(0.0, (T - st.T0) * 24.0))
    fp = st.flag_light(T)
    zf = st.flag_depth(T)
    x0, y0, _ = state["poles"][int(round(i))]
    M = cam.affine_at((x0, y0, zf))
    mat = cairo.Matrix(M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2])
    hero = Canvas(fc)
    ctx = hero.ctx
    ctx.save()
    if T >= st.T_PLANT:
        # the foot of the pole is inside the heap: hide it below the surface where it went in
        ex, ey = cam.project(np.array([[x0, y0 - 22.0, zf]]))[0][0]
        ctx.rectangle(-50, -50, W + 100, H + 100)
        ctx.rectangle(ex + 30, ey, -60, H)
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.clip()
    ctx.transform(mat)
    tr.draw(ctx, i, glow=0.35 * fp, light=HERO_LIGHT)
    ctx.restore()
    a = hero.rgba()
    cx, cy = tr.cloth_center(int(round(i)))
    lit = float(np.max(light_at(state, np.array([[cx, cy, zf]]), T)[0]))
    dim = min(1.0, max(0.04, 1.35 * lit) + fp) * emerge(T)
    if dim < 0.999:
        a[..., :3] *= dim
    img = over(img, a)
    # the niches' Flags
    pop = state["rise"].pop(T)
    on = np.nonzero(pop > 0)[0]
    if len(on):
        cv = Canvas(fc)
        f = state["rose"].flags
        scr, zc = cam.project(np.stack([f["xs"][on], f["ys"][on], np.zeros(len(on))], 1))
        draw_flag_field(cv.ctx, scr[:, 0], scr[:, 1], f["pole"][on] * cam.f / zc, T, seeds=f["seed"][on], wind=0.85,
                        side=f["side"][on], ang=f["ang"][on], pop=pop[on], glow=0.15)
        img = cv.over(img)
    # Crocus's fists on the pole (never on the cloth)
    front = state["cache"].pop("front", None)
    if front is None or front[0] != T:
        xc, kwc = pair.crocus_kw(T, state["tl"])
        yc = st.ground_y(xc, st.CROCUS_Z)
        Lc = figure_lights(state, T, Lm, xc, yc, st.CROCUS_Z, st.CROCUS_H)
    else:
        _, xc, yc, kwc, Lc = front
    fists = state["crocus"].render(fc, view, xc, yc, kwc, Lc, amb, part="front", env=env)
    cloth = Canvas(fc)
    cloth.ctx.transform(mat)
    tr.draw(cloth.ctx, i, pole=False)
    fists = darken(fists, emerge(T)) * (1.0 - np.clip(cloth.rgba()[..., 3:4] * 4.0, 0, 1))
    return over(img, fists)

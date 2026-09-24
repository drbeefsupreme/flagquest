"""t01 — THE TRACT (0.00-18.46).  Owner: T01Tract.

A physically simulated paper booklet in real rain; then the camera dives into its page and the printed panel
comes alive.
  0.0        LOOP FRAME: low and close on churned playa mud in the rain; the tract lies face-down, back page up
             (ref page 27 + 'THIS TRACT PLACED BY:' IVG stamp); far off, out of focus, someone walks away
             carrying the only colour in the world.  Identical to the film's last frame (t09).
  2.6        tract_flip: a gust rolls the booklet over its free edge: the cover THE FLAG GAME? (its flag alive)
  4.9        pages_riffle: the wind throws the leaves over and back; it settles open on page 1
  6.4        dive_in: the camera dives into page 1, panel 1: the halftone dots grow; ~6.95 the print comes alive
  7.4-18.4   R01 / S01 / S02 in the camp (t01_camp); 10.09 the flag-bearer passes behind them
  18.42      last frame = t01_page.frame(PANELS['p1']) (T02 glides to panel 2 from there)

Pipeline: physical world = GL (t01_gl: mud, puddles, rain, depth of field, the booklet from t01_paper/t01_book
with printed page textures from t01_pages); page world = t01_page.render_page (tract print, spot flags, page
stock, lettering).

Loop contract (t09): loop_setup(S) -> state;  render_at(fc, T_local, state, landing=True) -> rgb (pre-POST);
loop_frame(fc, state) == render_at(fc, 0.0);  POST (constant in the physical world; post() returns {} there).
"""
import math
import os

import numpy as np

from vx import *
from vx.flag import Flag
from vx.foley import export_events, export_track, screen_pan
from vx.config import CACHE

from . import t01_page as P
from . import t01_mud as M
from . import t01_book as BK
from . import t01_paper as PS
from . import t01_pages as PG
from . import t01_camp as CP
from . import t01_gl as G

POST = dict(bloom=0.10, bloom_thresh=0.80, grain=0.0, vignette=0.14)

T_SWITCH = 6.96            # 3D booklet -> 2D live page
X_FADE = (6.86, 6.99)      # cross-dissolve window
Z_SWITCH = 2.30            # page zoom (design px per page unit) at the switch
CM_PER_PU = PS.PAGE_W_CM / P.PAGE_W
FOVY = 30.0
PPU = dict(back=1.25, cover=1.25, page1=2.45, other=0.95)


# ============================================================================== state
def _cache_dir(S=None):
    d = CACHE / "t01"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _textures(cd):
    """page textures (cached on disk as uint8 npz)"""
    path = cd / "tex_v3.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k].astype(np.float32) / 255.0 for k in z.files}
    tex = {}
    tex["back"] = PG.back_page(PPU["back"])
    tex["cover_static"] = PG.cover_static(PPU["cover"])
    tex["ii"] = PG.inside_cover(PPU["other"])
    tex["iii"] = PG.inside_back(PPU["other"])
    # page 1: the camp panel frozen at its printed instant + T02's panels in their pre-18.46 state
    panels = {"p1": CP.PANEL1}
    page2 = None
    try:
        from . import t02_panels as T2
        panels.update(T2.PANELS)
        page2 = getattr(T2, "PAGE2", None)
    except Exception as e:                                   # neighbour not ready: glimpses
        print("[t01] t02_panels unavailable:", e)
    tex["1"] = PG.story_page(1, PPU["page1"], panels=panels, T=CP.PRINT_T)
    tex["2"] = PG.story_page(2, PPU["other"], panels=page2, T=34.0) if page2 else PG.story_page(2, PPU["other"])
    for n in range(3, 9):
        tex[str(n)] = PG.story_page(n, PPU["other"])
    np.savez(path, **{k: np.clip(v * 255 + 0.5, 0, 255).astype(np.uint8) for k, v in tex.items()})
    return tex


def _backdrop():
    """the far camp in the rain (drawn sharp; the lens blurs it) -> rgba float (h, w, 4)"""
    import cairo
    w, h = 1600, 560
    c = Canvas(s=1.0, w=w, h=h)
    ctx = c.ctx
    rng = np.random.default_rng(12)
    g = cairo.LinearGradient(0, 0, 0, h)
    g.add_color_stop_rgb(0, 0.80, 0.80, 0.81)
    g.add_color_stop_rgb(0.80, 0.70, 0.70, 0.71)
    g.add_color_stop_rgb(1, 0.60, 0.60, 0.61)
    ctx.set_source(g)
    ctx.paint()
    base = h * 0.80
    # far layer: domes, tents, a shade structure, art-car, a lattice effigy far away
    for k in range(22):
        x = rng.random() * w
        r = 22 + rng.random() * 55
        set_color(ctx, (0.42 + 0.12 * rng.random(),) * 3)
        if rng.random() < 0.45:
            ctx.arc(x, base, r, math.pi, 2 * math.pi)
        else:
            ctx.move_to(x - r, base)
            ctx.line_to(x - r * 0.1, base - r * 1.1)
            ctx.line_to(x + r, base)
        ctx.close_path()
        ctx.fill()
    set_color(ctx, (0.45, 0.45, 0.46))
    ctx.set_line_width(5)
    for x0 in (260, 1180):
        ctx.move_to(x0, base)
        ctx.line_to(x0, base - 120)
        ctx.move_to(x0 + 190, base)
        ctx.line_to(x0 + 190, base - 115)
        ctx.stroke()
        ctx.move_to(x0 - 20, base - 124)
        ctx.line_to(x0 + 210, base - 112)
        ctx.line_to(x0 + 210, base - 96)
        ctx.line_to(x0 - 20, base - 106)
        ctx.close_path()
        set_color(ctx, (0.38, 0.38, 0.39))
        ctx.fill()
        set_color(ctx, (0.45, 0.45, 0.46))
    # people standing in the rain far away (dark verticals)
    for k in range(14):
        x = rng.random() * w
        hh = 40 + rng.random() * 25
        set_color(ctx, (0.30 + 0.15 * rng.random(),) * 3)
        ctx.rectangle(x, base - hh, 12, hh)
        ctx.fill()
        ctx.arc(x + 6, base - hh - 6, 6, 0, 2 * math.pi)
        ctx.fill()
    img = c.rgba()
    img[int(base) + 2:] = 0.0            # below the horizon: the real (fogged) far mud shows
    img = blur(img, 2.0)
    return img


class _Far:
    """the figure walking away in the far background with a flag (screen-space layer at its depth)"""
    H_CM = 168.0
    POLE_CM = 244.0

    def __init__(self):
        from vx.chars import Character
        self.ch = Character("summer", 2, render="ink")
        self.h_local = 100.0
        self.pole_local = self.POLE_CM / self.H_CM * self.h_local
        n = int(round((6.0 + 2.5) * 24))
        self.f0 = -int(round(2.5 * 24))

        def pole_fn(i):
            T = (self.f0 + i) / 24.0
            return self.ch.anchors(0.0, 0.0, self.h_local, {"walk_away": 1.0, "hold_flag": 1.0}, T, facing=1,
                                   pole_len=self.pole_local)["pole"]

        def wind_fn(i):
            T = (self.f0 + i) / 24.0
            g = math.exp(-((T - 2.62) / 0.25) ** 2)
            return (-260.0 - 700.0 * g - 60 * math.sin(T * 1.3), -10.0)

        self.track = Flag(pole=self.pole_local, seed=21, side=-1).simulate(n, pole_fn, wind_fn, warmup=48)

    def world_pos(self, T):
        return np.array([-150.0 + 8.0 * (T + 2.5), 1480.0 + 62.0 * (T + 2.5), 0.0])

    def layers(self, fc, cam, T, ground):
        p = self.world_pos(T)
        p[2] = float(ground.height(p[0], min(p[1], 2650.0)))
        xy, depth = cam.project(p)
        d = float(depth[0])
        if d <= 10:
            return []
        k = cam.px_per_cm(d) * self.H_CM / self.h_local
        sx, sy = xy[0]
        if not (-400 < sx < 2300 and -600 < sy < 1600):
            return []
        i = (T * 24.0) - self.f0
        i = min(max(i, 0.0), self.track.n - 1)
        body = Canvas(fc)
        ctx = body.ctx
        ctx.translate(sx, sy)
        ctx.scale(k, k)
        self.ch.draw(ctx, 0.0, 0.0, self.h_local, {"walk_away": 1.0, "hold_flag": 1.0}, T, facing=1,
                     pole_len=self.pole_local, layer="back", silhouette=(0.16, 0.16, 0.17))
        fl = Canvas(fc)
        c2 = fl.ctx
        c2.translate(sx, sy)
        c2.scale(k, k)
        self.track.draw(c2, i, light=(-0.45, -0.8, 0.4))
        fr = Canvas(fc)
        c3 = fr.ctx
        c3.translate(sx, sy)
        c3.scale(k, k)
        self.ch.draw(c3, 0.0, 0.0, self.h_local, {"walk_away": 1.0, "hold_flag": 1.0}, T, facing=1,
                     pole_len=self.pole_local, layer="front", silhouette=(0.16, 0.16, 0.17))
        return [dict(rgba=body.rgba(), depth=d + 1.0), dict(rgba=fl.rgba(), depth=d, flag=True),
                dict(rgba=fr.rgba(), depth=d - 1.0)]


def _base_state(S, full=True):
    cd = _cache_dir(S)
    mud = M.load(cd)
    ground = M.Ground(mud)
    book = BK.Book(ground)
    bt = book.simulate(cd)
    st = dict(mud=mud, ground=ground, book=book, bt=bt, tex=_textures(cd), backdrop=_backdrop(), far=_Far())
    st["geo"] = _geometry(st)
    return st


def setup(S):
    st = _base_state(S)
    _ = CP.PANEL1.track          # simulate the bearer's flag once, before the workers fork
    _export_foley(S, st)
    return st


def loop_setup(S):
    """everything the physical world needs (for t09's tail and the loop frame)"""
    return _base_state(S, full=False)


# ============================================================================== geometry helpers
def _leaves_at(st, T):
    V = st["bt"].at(min(max(T, BK.T_SIM0), BK.T_SIM1 - 1.0 / 24))
    return V


def _refine(grid, n=4):
    from scipy.ndimage import zoom
    ny, nx = grid.shape[:2]
    out = np.stack([zoom(grid[..., c], ((ny - 1) * n + 1) / ny, order=3, grid_mode=False) for c in range(3)], -1)
    return out


def _page_frame(V, k=1):
    """world frame of leaf k's FRONT page: origin = page top-left corner, ex = +x per PU, ey = +y per PU, n"""
    g = V[k]
    top = g[-1]            # row NV-1 = page top
    bot = g[0]
    left = g[:, 0].mean(0)
    right = g[:, -1].mean(0)
    ex = (right - left) / P.PAGE_W
    ey = (bot.mean(0) - top.mean(0)) / P.PAGE_H
    origin = g[-1, 0]
    n = np.cross(right - left, top.mean(0) - bot.mean(0))
    n /= np.linalg.norm(n)
    return origin, ex, ey, n


def _page_to_world(V, px, py, k=1):
    """page point -> world via bilinear on leaf k's grid (front side)"""
    g = V[k]
    ny, nx = g.shape[:2]
    u = px / P.PAGE_W * (nx - 1)
    v = (1 - py / P.PAGE_H) * (ny - 1)
    i, j = min(int(u), nx - 2), min(int(v), ny - 2)
    fu, fv = u - i, v - j
    return (g[j, i] * (1 - fu) * (1 - fv) + g[j, i + 1] * fu * (1 - fv) + g[j + 1, i] * (1 - fu) * fv
            + g[j + 1, i + 1] * fu * fv)


def _geometry(st):
    VA = _leaves_at(st, 0.0)
    VB = _leaves_at(st, 4.2)
    VC = _leaves_at(st, T_SWITCH)
    cA = VA.reshape(-1, 3).mean(0)
    cB = VB.reshape(-1, 3).mean(0)
    cC = VC.reshape(-1, 3).mean(0)
    o, ex, ey, n = _page_frame(VC, 1)
    return dict(cA=cA, cB=cB, cC=cC, page=(o, ex, ey, n), VC=VC)


def _topdown_cam(st, cx, cy, zoom):
    """3D camera looking straight down on page 1 so that page point (cx, cy) is centred at `zoom` px/PU"""
    V = st["geo"]["VC"]
    o, ex, ey, n = st["geo"]["page"]
    tgt = _page_to_world(V, cx, cy)
    f_px = 540.0 / math.tan(math.radians(FOVY) / 2)
    d = f_px * CM_PER_PU / zoom
    up = -ey / np.linalg.norm(ey)
    return tgt + n * d, tgt, up, d


# ============================================================================== cameras
def _lerp(a, b, t):
    return a + (b - a) * t


def _plunge(T):
    """page point + zoom of the dive (shared by the 3D camera and the 2D page camera so they match)"""
    kd = ease_in(clamp((T - 6.35) / (T_SWITCH - 6.35)), 2.2)
    z = math.exp(_lerp(math.log(0.55), math.log(Z_SWITCH), kd))
    return _lerp(520.0, 500.0, kd), _lerp(420.0, 236.0, kd), z


def _cam3d(st, T):
    g = st["geo"]
    cA, cB, cC = g["cA"], g["cB"], g["cC"]
    up = np.array([0.0, 0.0, 1.0])
    # loop shot: low and close across the mud; the tract in the lower third, the far camp (and a figure
    # walking away with a flag) soft in the upper third.  A slow creep in.
    k0 = smootherstep(-2.5, 2.4, T)
    eye0 = cA + np.array([8.0, -41.0, 8.6]) + np.array([-0.8, 3.5, -0.4]) * k0
    tgt0 = cA + np.array([-4.0, 80.0, -3.5])
    # follow the flip, rising to read the cover
    eye1 = cB + np.array([6.0, -28.0, 25.0])
    tgt1 = cB + np.array([0.5, 2.5, 0.0])
    eye1b = cB + np.array([4.0, -25.0, 24.0])
    # the riffle: pull back a touch to see the spread
    eye2 = cC + np.array([3.0, -34.0, 38.0])
    tgt2 = cC + np.array([0.0, 3.0, 0.0])
    if T < 2.55:
        eye, tgt = eye0, tgt0
    elif T < 3.6:
        k = smootherstep(2.55, 3.6, T)
        eye, tgt = _lerp(eye0, eye1, k), _lerp(tgt0, tgt1, k)
    elif T < 4.75:
        k = smootherstep(3.6, 4.75, T)
        eye, tgt = _lerp(eye1, eye1b, k), tgt1
    elif T < 5.9:
        k = smootherstep(4.75, 5.9, T)
        eye, tgt = _lerp(eye1b, eye2, k), _lerp(tgt1, tgt2, k)
    else:
        # the dive: rise and swing over page 1, then plunge perpendicular into panel 1
        eyeH, tgtH, upH, _ = _topdown_cam(st, 520.0, 420.0, 0.55)
        k = smootherstep(5.9, 6.45, T)
        if T < 6.45:
            eye = _lerp(eye2, eyeH, k)
            tgt = _lerp(tgt2, tgtH, k)
            up = _lerp(up, upH, k * k)
            up = up / np.linalg.norm(up)
        else:
            eye, tgt, up = eyeH, tgtH, upH
        if T >= 6.35:
            # log-zoom plunge (dots grow steadily)
            cx, cy, z = _plunge(T)
            e2, t2, u2, _ = _topdown_cam(st, cx, cy, z)
            b = smootherstep(6.35, 6.5, T)
            eye, tgt, up = _lerp(eye, e2, b), _lerp(tgt, t2, b), _lerp(up, u2, b)
            up = up / np.linalg.norm(up)
    # focus on the tract (not on the look-at point)
    fA = float(np.linalg.norm(cA - eye))
    fT = float(np.linalg.norm(tgt - eye))
    focus = fA if T < 2.55 else _lerp(fA, fT, smootherstep(2.55, 3.3, T))
    blur = 40.0 if T < 2.55 else (_lerp(40.0, 28.0, smootherstep(2.55, 3.4, T)) if T < 5.9
                                  else _lerp(28.0, 8.0, smootherstep(5.9, 6.6, T)))
    # handheld breath
    br = np.array([fbm1(T * 0.35, 3), fbm1(T * 0.31, 5), fbm1(T * 0.4, 7)]) * (0.35 if T < 6.2 else 0.0)
    return G.Cam(eye + br, tgt + br * 0.5, fovy=FOVY if T > 2.55 else _lerp(32.0, FOVY, 0.0), focus=focus, blur=blur, up=up)


def _page_cam(T):
    """2D camera on page 1 after the switch"""
    a = P.PageCam(500.0, 236.0, Z_SWITCH)
    inside = P.PageCam(452.0, 236.0, 3.07)
    r01 = P.PageCam(470.0, 238.0, 3.14)
    summ = P.PageCam(548.0, 236.0, 3.16)
    s02 = P.PageCam(560.0, 238.0, 3.26)
    end = P.frame(P.PANELS["p1"])
    if T < T_SWITCH:
        return P.PageCam(*_plunge(T))
    if T < 7.32:
        k = ease_out(clamp((T - T_SWITCH) / (7.32 - T_SWITCH)), 2.2)
        return P.lerp_cam(a, inside, k)
    if T < 10.9:
        return P.lerp_cam(inside, r01, smootherstep(7.32, 10.9, T))
    if T < 11.75:
        return P.lerp_cam(r01, summ, smootherstep(10.9, 11.75, T))
    if T < 17.55:
        return P.lerp_cam(summ, s02, smootherstep(11.75, 17.55, T))
    return P.lerp_cam(s02, end, smootherstep(17.55, 18.30, T))


# ============================================================================== rendering
_UPLOADED = {}


def _world(st, fc):
    w = G.get_world(st["mud"], fc.w, fc.h, st["backdrop"])
    key = (os.getpid(), fc.w, fc.h)
    if not _UPLOADED.get(key):
        for k, v in st["tex"].items():
            if k != "cover_static":
                w.texture(k, v)
        w.texture("cover", st["tex"]["cover_static"])
        _UPLOADED[key] = True
    return w


_COVER_T = {}


def _update_cover(w, st, T):
    # the printed flag on the cover is alive: redraw it while the cover can be seen
    Tc = min(max(T, 2.75), 5.45)
    Tq = round(Tc * 24) / 24
    if _COVER_T.get(os.getpid()) == Tq:
        return
    fl = PG.cover_flag(PPU["cover"], Tq)
    w.texture("cover", PG.with_flag(st["tex"]["cover_static"], fl))
    _COVER_T[os.getpid()] = Tq


def _face_params(T, k):
    """(beads front, beads back, mud front, mud back) for leaf k"""
    rise = smoothstep(3.1, 6.5, T)
    if k == 0:       # cover: lay face-down in the mud until the flip -> mud smears; rain beads after
        return (0.06 + 0.22 * rise, 0.05 + 0.12 * smoothstep(5.2, 6.5, T), 0.75, 0.0)
    if k == PS.LEAVES - 1:   # back cover: up in the rain for the whole loop
        return (0.0, 0.34, 0.0, 0.25)
    if k == 1:
        return (0.02 + 0.10 * smoothstep(5.6, 7.0, T), 0.02, 0.0, 0.0)
    return (0.02, 0.02, 0.0, 0.0)


def _leaf_list(st, T):
    V = _leaves_at(st, T)
    out = []
    L = PS.LEAVES
    for k in range(L):
        g = _refine(V[k], 4)
        # stacking bias along the front normal keeps the leaves' order crisp in the depth buffer
        du = np.gradient(g, axis=1)
        dv = np.gradient(g, axis=0)
        nrm = np.cross(du, dv)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
        g = g + nrm * ((L - 1 - k) * 0.012)
        ny, nx = g.shape[:2]
        U, Vv = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
        f, b = PG.FACES[k]
        out.append(dict(pos=g, uv=np.stack([U, Vv], -1), front=str(f), back=str(b), face=_face_params(T, k),
                        page_cm=(PS.PAGE_W_CM, PS.PAGE_H_CM), seed=(1.0 + k * 3.1, 2.0 + k * 1.7)))
    return out


def _render_physical(fc, st, T):
    w = _world(st, fc)
    if 2.7 <= T <= 5.6:
        _update_cover(w, st, T)
    cam = _cam3d(st, T)
    leaves = _leaf_list(st, T)
    layers = st["far"].layers(fc, cam, T, st["ground"]) if T < 4.8 else []
    rain = 1.0
    img = w.render(cam, T, leaves, layers=layers, rain=rain, s=fc.s, expo=1.0,
                   rain_vel=(-190.0 - 250.0 * math.exp(-((T - 2.66) / 0.2) ** 2), 50.0, -900.0))
    return img


def _render_page(fc, st, T):
    cam = _page_cam(T)
    panels = {"p1": CP.PANEL1}
    try:
        from . import t02_panels as T2
        panels.update(T2.PANELS)
    except Exception:
        pass
    return P.render_page(fc, cam, panels, page_no=1, T=T)


def render_at(fc, T, st, landing=True):
    """the t01 picture at t01-local time T (any T; T < 0 = t09's tail: the toss and splat) -> rgb pre-POST"""
    if T < X_FADE[0]:
        return _render_physical(fc, st, T)
    if T >= X_FADE[1]:
        return _render_page(fc, st, T)
    a = smoothstep(X_FADE[0], X_FADE[1], T)
    return _render_physical(fc, st, T) * (1 - a) + _render_page(fc, st, T) * a


def loop_frame(fc, st):
    return render_at(fc, 0.0, st)


def render(fc, st):
    return render_at(fc, fc.T, st)


def post(fc, st):
    return {}


# ============================================================================== sound exports
def _export_foley(S, st):
    bt = st["bt"]
    f0 = int(round(-bt.t0 * 24))
    n = S.n
    e = np.zeros(n, np.float32)
    src = bt.energy[f0:f0 + n]
    e[:len(src)] = src
    e = np.clip(e / 120.0, 0, 1.5)
    pan = np.zeros(n, np.float32)
    ev = []
    # leaf crossings (page turns) in the riffle: when a leaf's mean angle passes vertical
    V = bt.verts
    for k in range(PS.LEAVES):
        prev = None
        for i in range(f0, min(len(V), f0 + n)):
            g = V[i, k]
            up = g[:, -1].mean(0) - g[:, 0].mean(0)
            side = np.sign(up[0])
            if prev is not None and side != prev and up[2] > 3.0:
                ev.append(dict(t=(i - f0) / 24.0, kind="paper", strength=0.6, pan=-0.1,
                               desc=f"leaf {k} flips over (riffle)"))
            prev = side
    hits = [dict(t=2.62, kind="whoosh", strength=0.8, pan=0.2, desc="gust lifts the tract (tract_flip)"),
            dict(t=3.05, kind="paper", strength=1.0, pan=-0.2, desc="wet tract slaps down face-up on the mud"),
            dict(t=4.80, kind="whoosh", strength=0.6, pan=0.3, desc="riffle gust"),
            dict(t=5.75, kind="whoosh", strength=0.5, pan=-0.3, desc="back-gust returns the pages"),
            dict(t=6.40, kind="whoosh", strength=0.9, pan=0.0, desc="camera dives into page 1 (dive_in)"),
            dict(t=6.96, kind="paper", strength=0.7, pan=0.0, desc="the print comes alive (halftone -> drawing)"),
            dict(t=7.30, kind="balloon", strength=0.6, pan=-0.3, desc="R01 balloon pops"),
            dict(t=10.09, kind="flag", strength=0.5, pan=0.0, desc="the flag-bearer passes behind them (YELLOW FLAGS)"),
            dict(t=11.50, kind="balloon", strength=0.6, pan=0.35, desc="S01 balloon pops"),
            dict(t=15.48, kind="balloon", strength=0.5, pan=0.35, desc="S02 balloon pops"),
            dict(t=17.60, kind="glide", strength=0.4, pan=0.0, desc="camera pulls back to the panel (reading frame)")]
    for i in range(0, n, 12):
        T = i / 24.0
        if 8.5 < T < 13.2:
            ev.append(dict(t=T, kind="step", strength=0.15, pan=screen_pan(700 + (T - 8.5) * 150),
                           desc="bearer footsteps (mud), far"))
    export_events(S, "hits", hits + ev)
    export_track(S, "paper", e, pan=pan, gain=1.0, kind="paper",
                 events=[[h["t"], h["strength"], h["pan"]] for h in hits if h["kind"] == "paper"])
    tr = CP.PANEL1.track
    tr.export_foley(S, "bearer_flag", f0=CP.FLAG_F0, gain=0.7)

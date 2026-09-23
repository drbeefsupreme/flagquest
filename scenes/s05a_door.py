"""s05a K03 + BREACH (98.4-100.5).
Outside the Nation's colossal gate (a brutalist fortress wall, a steel double door, a crown in relief,
searchlights in the fog) the squad stacks up; on each 'Breach!' the point man drives the butt of his furled flag
into the door and a flag-yellow shock ring + Voronoi cracks spread from the strike point. At the cue `breach`
(99.7) we cut inside: the door's own crack cells blow in as rigid shards, blinding flag light floods through a
real fluid-simulated smoke cloud with volumetric shafts, and the squad leaps through in slow motion while their
flags unfurl (real cloth sims). The light swallows everything: flat flag_glow on the last frame (hand-off)."""
import math

import cairo
import cv2
import numpy as np

from vx import W, H, C, Canvas, col, set_color, mix, circle, ellipse, poly, rrect, clamp, lerp, smoothstep, \
    ease_in, ease_out, ease_in_out, hash01, shake, blur
from vx.chars import Character, POSES
from vx import flag as vflag
from vx.flag import Flag
from scenes.s05a_fx import (GOLD, GOLD_HOT, NVG, FLAGGLOW, lin_grad, rad_grad, god_rays, streak, radial_img, comp,
                            grid_xy, noise_layer)
from scenes import s05a_smoke as smoke

DOOR_W, DOOR_H = 8.0, 14.0
WALL_H = 34.0
STRIKE = (1.2, 1.45)            # door-local strike point (m): x right (seen from outside), y up
FLAGC = C["flag"]
FLAG_L = C["flag_light"]
GLOW = C["flag_glow"]
SIL = (0.035, 0.04, 0.06)       # silhouette colour
COOL = (0.62, 0.74, 1.0)


# ================================================================================================ camera
class Pin:
    """pinhole camera at (x, y, z); yaw about +y (0 looks along +z, pi looks back along -z), pitch (+ looks up)"""

    def __init__(self, x, y, z, f, hor=540.0, cx=960.0, yaw=0.0, pitch=0.0, roll=0.0):
        self.x, self.y, self.z, self.f, self.hor, self.cx = x, y, z, f, hor, cx
        self.cy_, self.sy_ = math.cos(yaw), math.sin(yaw)
        self.cp_, self.sp_ = math.cos(pitch), math.sin(pitch)
        self.roll = roll

    def rel(self, x, y, z):
        dx, dy, dz = x - self.x, y - self.y, z - self.z
        xr = dx * self.cy_ - dz * self.sy_
        zr = dx * self.sy_ + dz * self.cy_
        yr = dy * self.cp_ - zr * self.sp_
        zr2 = dy * self.sp_ + zr * self.cp_
        return xr, yr, zr2

    def depth(self, x, y, z):
        return self.rel(x, y, z)[2]

    def p(self, x, y, z):
        xr, yr, zr = self.rel(x, y, z)
        k = self.f / max(0.05, zr)
        sx, sy = xr * k, -yr * k
        if self.roll:
            c, s = math.cos(self.roll), math.sin(self.roll)
            sx, sy = sx * c - sy * s, sx * s + sy * c
        return (self.cx + sx, self.hor + sy)

    def k(self, x, y, z):
        return self.f / max(0.05, self.depth(x, y, z))

    def _proj(self, r):
        xr, yr, zr = r
        k = self.f / zr
        sx, sy = xr * k, -yr * k
        if self.roll:
            c, s = math.cos(self.roll), math.sin(self.roll)
            sx, sy = sx * c - sy * s, sx * s + sy * c
        return (self.cx + sx, self.hor + sy)

    def poly(self, ctx, pts, near=0.15):
        """add a world-space polygon to the path, clipped against the near plane. returns screen pts"""
        rel = [self.rel(*p) for p in pts]
        out = []
        n = len(rel)
        for i in range(n):
            a, b = rel[i], rel[(i + 1) % n]
            ina, inb = a[2] > near, b[2] > near
            if ina:
                out.append(a)
            if ina != inb:
                t = (near - a[2]) / (b[2] - a[2])
                out.append(tuple(a[q] + (b[q] - a[q]) * t for q in range(3)))
        if len(out) < 3:
            return []
        sp = [self._proj(r) for r in out]
        poly(ctx, sp)
        return sp

    def line(self, ctx, a, b, near=0.15):
        """add a clipped world-space segment to the path"""
        ra, rb = self.rel(*a), self.rel(*b)
        if ra[2] <= near and rb[2] <= near:
            return False
        if ra[2] <= near or rb[2] <= near:
            t = (near - ra[2]) / (rb[2] - ra[2])
            m = tuple(ra[q] + (rb[q] - ra[q]) * t for q in range(3))
            if ra[2] <= near:
                ra = m
            else:
                rb = m
        ctx.move_to(*self._proj(ra))
        ctx.line_to(*self._proj(rb))
        return True


def ang_of(p0, p1):
    """flag-API pole angle (0 = up, + = clockwise) for a screen segment base p0 -> top p1"""
    return math.atan2(p1[0] - p0[0], -(p1[1] - p0[1]))


# ================================================================================================ fracture
def build_cracks(seed=7, n=74):
    """Voronoi fracture of the door, seeded densely around the strike point: cells (door coords) + crack segments"""
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon, box, LineString
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        r = 0.25 + 9.0 * rng.random() ** 1.7
        a = rng.random() * math.tau
        p = (STRIKE[0] + r * math.cos(a), STRIKE[1] + r * math.sin(a) * 1.2)
        if -DOOR_W / 2 < p[0] < DOOR_W / 2 and 0 < p[1] < DOOR_H:
            pts.append(p)
    pts = np.array(pts)
    far = np.array([[-50, -50], [50, -50], [50, 60], [-50, 60]])
    vor = Voronoi(np.vstack([pts, far]))
    door = box(-DOOR_W / 2, 0, DOOR_W / 2, DOOR_H)
    cells = []
    for i in range(len(pts)):
        reg = vor.regions[vor.point_region[i]]
        if -1 in reg or not reg:
            continue
        pg = Polygon(vor.vertices[reg]).intersection(door)
        if pg.is_empty or pg.geom_type != "Polygon":
            continue
        cells.append(dict(pts=np.array(pg.exterior.coords)[:-1], c=np.array(pg.centroid.coords[0]), area=pg.area))
    segs = []
    for rv in vor.ridge_vertices:
        if -1 in rv:
            continue
        ls = LineString([vor.vertices[rv[0]], vor.vertices[rv[1]]]).intersection(door)
        if ls.is_empty or ls.geom_type != "LineString":
            continue
        q = np.array(ls.coords)
        # drop ridges lying on the door border
        if (np.all(np.abs(q[:, 0]) > DOOR_W / 2 - 1e-6) or np.all(q[:, 1] < 1e-6)
                or np.all(q[:, 1] > DOOR_H - 1e-6)):
            continue
        dist = min(np.hypot(*(q[0] - STRIKE)), np.hypot(*(q[1] - STRIKE)))
        segs.append((q[0], q[1], dist, np.hypot(*(q.mean(0) - STRIKE))))
    star = [(i / 22 * math.tau + rng.normal() * 0.12, 0.25 + 0.6 * rng.random()) for i in range(22)]
    for c in cells:
        rel = c["c"] - np.array(STRIKE)
        r = np.hypot(*rel) + 0.3
        c["vxy"] = rel / r * (5 + 9 * rng.random()) / (0.6 + 0.15 * r) + rng.normal(size=2) * 1.2
        c["vz"] = (18 + 26 * rng.random()) / (0.7 + 0.1 * r)
        c["w"] = rng.normal() * 7.0
        c["tumble"] = rng.normal() * 12.0
        c["shade"] = 0.05 + 0.07 * rng.random()
        c["r"] = r - 0.3
        c["t_rel"] = max(0.0, (c["r"] - 1.1) / 28.0) + 0.004 * rng.random()   # fracture front ~28 m/s
    return dict(cells=cells, segs=segs, star=star)


def crack_radius(T, st):
    b0, b1, b2, bend = st["beats"]
    if T < b0:
        return 0.0
    if T < b1:
        return 0.35 + 0.35 * ease_out(clamp((T - b0) / 0.15))
    if T < b2:
        return 0.7 + 1.8 * ease_out(clamp((T - b1) / 0.18))
    return 2.5 + 12.0 * ease_in(clamp((T - b2) / (bend - b2)), 1.5)


def crack_heat(T, st):
    b0, b1, b2, bend = st["beats"]
    if T < b2:
        return 0.5
    return 0.5 + 0.7 * ease_in(clamp((T - b2) / (bend - b2)), 2.0)


def draw_cracks(ctx, cam, cracks, R, heat, width_k=1.0, zdoor=0.0):
    if R <= 0:
        return
    k0 = cam.k(STRIKE[0], STRIKE[1], zdoor)
    p0 = cam.p(STRIKE[0], STRIKE[1], zdoor)
    for a, L in cracks["star"]:
        l = min(L, R * 1.2)
        ctx.set_line_width(max(0.8, 0.04 * k0 * width_k))
        ctx.move_to(*p0)
        ctx.line_to(*cam.p(STRIKE[0] + math.cos(a) * l, STRIKE[1] + math.sin(a) * l, zdoor))
        ctx.stroke()
    for q0, q1, dist, dmid in cracks["segs"]:
        if dist > R:
            continue
        d0 = np.hypot(*(q0 - STRIKE))
        d1 = np.hypot(*(q1 - STRIKE))
        a, b = (q0, q1) if d0 < d1 else (q1, q0)
        da, db = min(d0, d1), max(d0, d1)
        f = clamp((R - da) / max(1e-3, db - da))
        e = a + (b - a) * f
        w = max(0.7, max(0.018, 0.06 - 0.004 * dmid) * cam.k(a[0], a[1], zdoor) * width_k * (0.6 + 0.4 * heat))
        ctx.set_line_width(w)
        ctx.move_to(*cam.p(a[0], a[1], zdoor))
        ctx.line_to(*cam.p(e[0], e[1], zdoor))
        ctx.stroke()


def crack_layer(fc, cam, cracks, R, heat, width_k=1.0):
    """additive glowing crack image (flag light leaking through the fracture)"""
    L = Canvas(fc)
    set_color(L.ctx, FLAG_L)
    draw_cracks(L.ctx, cam, cracks, R, heat, width_k=width_k)
    core = L.rgba()[..., 3]
    img = core[..., None] * np.asarray(mix(FLAG_L, (1, 1, 1), 0.45), np.float32) * (0.7 + 0.3 * heat)
    img += blur(core, 3 * fc.s)[..., None] * np.asarray(FLAGC, np.float32) * 0.9 * heat
    img += blur(core, 18 * fc.s)[..., None] * np.asarray(FLAGC, np.float32) * 0.55 * heat
    return img


# ================================================================================================ set
def _grad_v(ctx, cam, x, y0, y1, z, stops):
    ctx.set_source(lin_grad(*cam.p(x, y1, z), *cam.p(x, y0, z), stops))


def draw_fortress(ctx, cam, T, inside=False):
    """brutalist concrete wall with deep panel joints, parapet, slit windows; steel double door; crown relief"""
    z = 0.0
    X = 40.0
    dim = 0.3 if inside else 1.0
    c0 = tuple(c * dim for c in (0.07, 0.078, 0.11))
    c1 = tuple(c * dim for c in (0.12, 0.13, 0.17))
    if cam.poly(ctx, [(-X, 0, z), (X, 0, z), (X, WALL_H, z), (-X, WALL_H, z)]):
        _grad_v(ctx, cam, 0, 0, WALL_H, z, [(0, c0), (1, c1)])
        ctx.fill()
    ctx.new_path()
    k0 = cam.k(0, 3, z)
    ctx.set_line_width(max(0.6, 0.06 * k0))
    for j, yy in enumerate(np.arange(2.4, WALL_H, 2.4)):
        set_color(ctx, (0.02, 0.022, 0.035), 0.9)
        if cam.line(ctx, (-X, yy, z), (X, yy, z)):
            ctx.stroke()
        set_color(ctx, (0.35, 0.4, 0.5), 0.1 * dim)
        if cam.line(ctx, (-X, yy - 0.08, z), (X, yy - 0.08, z)):
            ctx.stroke()
        set_color(ctx, (0.02, 0.022, 0.035), 0.8)
        for xx in np.arange(-X + (j % 2) * 1.8, X, 3.6):
            if abs(xx) < DOOR_W / 2 + 1.7 and yy < DOOR_H + 1.7:
                continue
            if cam.line(ctx, (xx, yy, z), (xx, yy + 2.4, z)):
                ctx.stroke()
    if not inside:
        # slit windows with a cold light
        for i in range(-5, 6):
            if i == 0:
                continue
            for yy in (20.0, 27.0):
                if cam.poly(ctx, [(i * 6 - 0.12, yy, z), (i * 6 + 0.12, yy, z), (i * 6 + 0.12, yy + 2.6, z),
                                  (i * 6 - 0.12, yy + 2.6, z)]):
                    set_color(ctx, (0.55, 0.68, 0.9), 0.6)
                    ctx.fill()
        if cam.poly(ctx, [(-X, WALL_H, z), (X, WALL_H, z), (X, WALL_H + 1.6, z - 0.8), (-X, WALL_H + 1.6, z - 0.8)]):
            set_color(ctx, (0.05, 0.055, 0.075))
            ctx.fill()
    # door surround: stepped deep frame
    for pad, shade_ in ((1.6, 0.035), (1.05, 0.055), (0.45, 0.025)):
        if cam.poly(ctx, [(-DOOR_W / 2 - pad, 0, z), (DOOR_W / 2 + pad, 0, z), (DOOR_W / 2 + pad, DOOR_H + pad, z),
                          (-DOOR_W / 2 - pad, DOOR_H + pad, z)]):
            set_color(ctx, (shade_, shade_ * 1.08, shade_ * 1.4))
            ctx.fill()
    # door leaves: blued steel plates with rivets, a cold sheen high up
    if cam.poly(ctx, [(-DOOR_W / 2, 0, z), (DOOR_W / 2, 0, z), (DOOR_W / 2, DOOR_H, z), (-DOOR_W / 2, DOOR_H, z)]):
        _grad_v(ctx, cam, 0, 0, DOOR_H, z, [(0, tuple(c * dim for c in (0.1, 0.12, 0.16))),
                                             (0.55, tuple(c * dim for c in (0.2, 0.23, 0.3))),
                                             (1, tuple(c * dim for c in (0.12, 0.14, 0.19)))])
        ctx.fill()
    kd = cam.k(0, 3, z)
    for vv in np.arange(1.4, DOOR_H, 1.4):
        ctx.set_line_width(max(0.7, 0.05 * kd))
        set_color(ctx, (0.025, 0.03, 0.045), 0.95)
        if cam.line(ctx, (-DOOR_W / 2, vv, z), (DOOR_W / 2, vv, z)):
            ctx.stroke()
        set_color(ctx, (0.55, 0.62, 0.75), 0.25 * dim)
        if cam.line(ctx, (-DOOR_W / 2, vv - 0.07, z), (DOOR_W / 2, vv - 0.07, z)):
            ctx.stroke()
        if kd > 22:
            for uu in np.arange(-DOOR_W / 2 + 0.3, DOOR_W / 2, 0.6):
                if cam.depth(uu, vv, z) < 0.3:
                    continue
                p = cam.p(uu, vv + 0.2, z)
                set_color(ctx, (0.34, 0.38, 0.46), 0.8 * dim)
                circle(ctx, p[0], p[1], max(0.7, 0.045 * cam.k(uu, vv, z)))
                ctx.fill()
    set_color(ctx, (0.015, 0.015, 0.025))
    ctx.set_line_width(max(1.0, 0.08 * kd))
    if cam.line(ctx, (0, 0, z), (0, DOOR_H, z)):
        ctx.stroke()
    if not inside:
        _crown(ctx, cam, 0.0, DOOR_H + 3.4, z)


def _crown(ctx, cam, cx_, cy_, z):
    """the Nation: a colossal crown in cold bronze relief (never yellow)"""
    S = 1.35
    pts = [(-3.2, 0.0)]
    for i in range(5):
        u0 = -3.2 + i * 1.6
        pts += [(u0 + 0.8, 2.7 if i % 2 == 0 else 2.0), (u0 + 1.6, 0.5)]
    pts[-1] = (3.2, 0.0)
    band = [(3.2, -1.3), (-3.2, -1.3)]
    W3 = [(cx_ + a * S, cy_ + b * S, z) for a, b in pts + band]
    if cam.depth(cx_, cy_, z) < 0.5:
        return
    P = [cam.p(*w) for w in W3]
    poly(ctx, [(x + 6, y + 8) for x, y in P])
    set_color(ctx, (0.01, 0.01, 0.02), 0.8)
    ctx.fill()
    poly(ctx, P)
    ctx.set_source(lin_grad(*cam.p(cx_, cy_ + 2.7 * S, z), *cam.p(cx_, cy_ - 1.3 * S, z),
                            [(0, (0.46, 0.55, 0.58)), (0.45, (0.2, 0.25, 0.27)), (1, (0.08, 0.1, 0.12))]))
    ctx.fill()
    k = cam.k(cx_, cy_, z)
    for i in range(5):
        u0 = -3.2 + i * 1.6
        p = cam.p(cx_ + (u0 + 0.8) * S, cy_ + ((2.7 if i % 2 == 0 else 2.0) + 0.3) * S, z)
        circle(ctx, p[0], p[1], 0.32 * S * k)
        ctx.set_source(rad_grad(p[0] - 0.1 * S * k, p[1] - 0.1 * S * k, 0, 0.35 * S * k,
                                [(0, (0.65, 0.72, 0.75)), (1, (0.15, 0.18, 0.2))]))
        ctx.fill()
    for uu in (-2.0, 0.0, 2.0):
        p = cam.p(cx_ + uu * S, cy_ - 0.65 * S, z)
        ellipse(ctx, p[0], p[1], 0.36 * S * k, 0.3 * S * k)
        set_color(ctx, (0.05, 0.07, 0.09))
        ctx.fill()
    set_color(ctx, (0.6, 0.68, 0.78), 0.4)
    ctx.set_line_width(max(0.8, 0.06 * k))
    poly(ctx, P[:len(pts)], close=False)
    ctx.stroke()


def draw_sky(ctx, cam):
    ctx.rectangle(-50, -50, 2020, 1180)
    ctx.set_source(lin_grad(0, 0, 0, 1080, [(0, (0.03, 0.035, 0.07)), (0.6, (0.07, 0.08, 0.14)), (1, (0.1, 0.11, 0.17))]))
    ctx.fill()


def draw_plaza(ctx, cam):
    if cam.poly(ctx, [(-60, 0, 0), (60, 0, 0), (60, 0, -60), (-60, 0, -60)]):
        ctx.set_source(lin_grad(0, cam.p(0, 0, 0)[1], 0, 1080, [(0, (0.07, 0.075, 0.1)), (1, (0.02, 0.022, 0.03))]))
        ctx.fill()
    ctx.new_path()
    k0 = cam.k(0, 0, -5)
    ctx.set_line_width(max(0.6, 0.035 * k0))
    set_color(ctx, (0.005, 0.005, 0.01), 0.6)
    for xx in np.arange(-30, 31, 2.5):
        if cam.line(ctx, (xx, 0, 0), (xx, 0, -40)):
            ctx.stroke()
    for zz in np.arange(-2.5, -40, -2.5):
        if cam.line(ctx, (-30, 0, zz), (30, 0, zz)):
            ctx.stroke()


def searchlights(fc, cam, T, k=1.0):
    """cold searchlight beams sweeping the fog from the parapet (additive)"""
    B = Canvas(fc)
    b = B.ctx
    for i, (x0, sp, ph, yaw0) in enumerate(((-18, 0.5, 0.3, -0.5), (16, -0.42, 1.9, 0.6), (-4, 0.33, 3.4, 0.1),
                                            (28, 0.27, 5.0, 0.9))):
        base = (x0, WALL_H + 1.8, -0.5)
        a = yaw0 + 0.6 * math.sin(T * sp + ph)
        d = (math.sin(a) * 0.8, 0.55, -math.cos(a) * 0.5)
        L = 70.0
        if cam.depth(*base) < 0.5:
            continue
        p0 = cam.p(*base)
        tip = (base[0] + d[0] * L, base[1] + d[1] * L, base[2] + d[2] * L)
        if cam.depth(*tip) < 0.5:
            continue
        p1 = cam.p(*tip)
        nx, ny = -(p1[1] - p0[1]), p1[0] - p0[0]
        nl = math.hypot(nx, ny) + 1e-6
        wv = 0.12 * math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        b.move_to(*p0)
        b.line_to(p1[0] + nx / nl * wv, p1[1] + ny / nl * wv)
        b.line_to(p1[0] - nx / nl * wv, p1[1] - ny / nl * wv)
        b.close_path()
        b.set_source(lin_grad(*p0, *p1, [(0, (*COOL, 0.28 * k)), (1, (*COOL, 0.0))]))
        b.fill()
        b.set_source(rad_grad(p0[0], p0[1], 0, 30, [(0, (1, 1, 1, 0.9 * k)), (1, (*COOL, 0))]))
        circle(b, p0[0], p0[1], 30)
        b.fill()
    return blur(B.rgba()[..., :3], 8 * fc.s)


# ================================================================================================ squad
STACK = [("schismmancer", 3), ("commander", 1), ("schismmancer", 5), ("schismmancer", 2), ("schismmancer", 4)]


def draw_member(fc, ch, cam, x, z, T, pose, facing=-1, rim_col=GOLD_HOT, rim_k=1.0, rim_dir=(-1.0, -0.3),
                tint=(SIL, 0.88), extra=None, pole_world=None, furl_light=(-0.6, -0.3, 0.6), glow=0.35, mouth=0.0,
                h_m=1.82):
    """one squad member (+ his furled flag). pole_world = (base xyz, top xyz) world metres, or None to use the
    rig's own pole anchor. Returns premultiplied layers [back, flag, front, nvg]."""
    k = cam.k(x, 0.9, z)
    h = h_m * k
    fx, fy = cam.p(x, 0.0, z)
    kw = dict(goggles_down=True, rim=(rim_col, rim_k, rim_dir), tint=tint)
    if extra:
        kw.update(extra)
    if pole_world is not None:
        b0, t0 = pole_world
        pb, pt = cam.p(*b0), cam.p(*t0)
        L = math.hypot(pt[0] - pb[0], pt[1] - pb[1])
        ang = ang_of(pb, pt)
        kw["hand_r"] = (lerp(pb[0], pt[0], 0.42), lerp(pb[1], pt[1], 0.42))
        kw["hand_l"] = (lerp(pb[0], pt[0], 0.62), lerp(pb[1], pt[1], 0.62))
        kw["shape_r"] = kw["shape_l"] = "grip"
    else:
        L = 2.44 * k
    a = ch.anchors(fx, fy, h, pose, T, facing=facing, pole_len=L, **kw)
    if pole_world is None and a.get("pole") is not None:
        pb = a["pole"][:2]
        ang = a["pole"][2]
    layers = []
    lb = Canvas(fc)
    ch.draw(lb.ctx, fx, fy, h, pose, T, facing=facing, pole_len=L, layer="back", expr="determined", mouth=mouth, **kw)
    layers.append(lb.rgba())
    fl = Canvas(fc)
    vflag.draw_furled(fl.ctx, pb[0], pb[1], pole=L, ang=ang, seed=int(abs(x) * 7) % 4, light=furl_light, glow=glow)
    layers.append(fl.rgba())
    lf = Canvas(fc)
    ch.draw(lf.ctx, fx, fy, h, pose, T, facing=facing, pole_len=L, layer="front", expr="determined", mouth=mouth,
            **kw)
    layers.append(lf.rgba())
    g = Canvas(fc)
    ex, ey = a.get("eyes", a["head"])
    r = max(1.2, h * 0.014)
    g.ctx.set_source(rad_grad(ex, ey, 0, r * 6, [(0, (*NVG, 0.85)), (0.2, (*NVG, 0.4)), (1, (*NVG, 0))]))
    circle(g.ctx, ex, ey, r * 6)
    g.ctx.fill()
    set_color(g.ctx, (0.85, 1.0, 0.8))
    circle(g.ctx, ex + facing * r * 0.5, ey, r)
    g.ctx.fill()
    layers.append(g.rgba())
    return layers, a


def ram_pole(T, st, lean=0.0):
    """world pole of the point man: butt on the strike point, levelled back toward him; he pulls back and slams
    it home on every beat"""
    off = 0.0
    for tb in st["beats"][:3]:
        pull = smoothstep(tb - 0.26, tb - 0.07, T) * (1 - smoothstep(tb - 0.05, tb, T))
        rec = 0.08 * math.exp(-max(0, T - tb) * 18) * (T > tb)
        off = max(off, 0.32 * pull + rec)
    d = np.array((math.cos(0.2 + lean), math.sin(0.2 + lean), -0.18))
    d /= np.linalg.norm(d)
    b = np.array((STRIKE[0], STRIKE[1], -0.04)) + d * off
    return tuple(b), tuple(b + d * 2.44)


def beat_ring(fc, cam, T, t_hit, k=1.0):
    """flag-yellow shock ring racing across the door plane from the strike point"""
    a = T - t_hit
    if a < 0 or a > 0.4:
        return None
    L = Canvas(fc)
    c = L.ctx
    r = 0.2 + 12 * ease_out(clamp(a / 0.38), 2.5)
    e = (1 - clamp(a / 0.38)) ** 2
    c.set_line_width(max(1.2, 0.18 * cam.k(STRIKE[0], STRIKE[1], 0)))
    set_color(c, FLAG_L, e * k)
    poly(c, [cam.p(STRIKE[0] + r * math.cos(t), STRIKE[1] + r * math.sin(t), 0) for t in np.linspace(0, math.tau, 96)])
    c.stroke()
    rgba = L.rgba()
    return blur(rgba[..., :3], 6 * fc.s) * 0.9 + rgba[..., :3] * 0.5


def _exterior(fc, st, cam, T, members, t_hits, beams=1.0, fog_k=1.0):
    cv = Canvas(fc)
    ctx = cv.ctx
    draw_sky(ctx, cam)
    draw_fortress(ctx, cam, T)
    draw_plaza(ctx, cam)
    img = cv.rgb()
    heat = crack_heat(T, st)
    R = crack_radius(T, st)
    sp = cam.p(STRIKE[0], STRIKE[1], 0)
    # warm light spilling from the cracks onto door, wall and ground
    img += radial_img(fc, sp[0], sp[1], 60 + 60 * R * cam.k(0, 1, 0) / 60, FLAGC, 2.0) * (0.06 + 0.08 * heat)
    img += crack_layer(fc, cam, st["cracks"], R, heat)
    for th in t_hits:
        r = beat_ring(fc, cam, T, th)
        if r is not None:
            img += r
    # fog + searchlights
    xx, yy = grid_xy(fc)
    img += np.clip(1.0 - np.abs(yy - cam.p(0, 0, 0)[1]) / 500, 0, 1)[..., None] * np.array((0.07, 0.085, 0.12),
                                                                                              np.float32) * fog_k
    img += searchlights(fc, cam, T, beams)
    # squad, far to near
    for m in sorted(members, key=lambda m: -cam.depth(m["x"], 0.9, m["z"])):
        ch = st["dchars"][m["id"]]
        layers, a = draw_member(fc, ch, cam, m["x"], m["z"], T, m["pose"], facing=m.get("facing", -1),
                                pole_world=m.get("pole"), extra=m.get("extra"), rim_dir=m.get("rim_dir", (-1.0, -0.3)),
                                rim_col=m.get("rim_col", GOLD_HOT), rim_k=m.get("rim_k", 1.0),
                                furl_light=(-0.7, -0.2, 0.5))
        for l in layers:
            img = comp(img, l)
    return img


def _stack_members(xs, zs, T, st, lunge=0.0):
    ms = []
    for i, (x, z) in enumerate(zip(xs, zs)):
        idk = STACK[i]
        if i == 0:
            ms.append(dict(id=idk, x=x, z=z, pose="ram", pole=ram_pole(T, st), facing=-1))
        else:
            ph = hash01(i, 3) * 0.1
            ms.append(dict(id=idk, x=x - lunge * (1 - i * 0.12), z=z, pose="stack", facing=-1,
                           extra=dict(lean=0.1 + 0.08 * math.sin(T * 2 + i)), rim_k=0.8))
    return ms


def beat1(fc, st, u):
    """98.42: low wide: the colossal gate towering into the night, searchlights in fog; the squad tiny at its
    foot, the point man's flag driven into the steel: shock ring + first cracks"""
    T = fc.T
    b0 = st["beats"][0]
    imp = T - b0
    sx, sy = shake(T, 6 * math.exp(-max(0, imp) * 10) if imp > 0 else 0, 30, 3)
    cam = Pin(x=6.5 - 0.25 * u, y=0.45, z=-11.0 + 0.7 * u, f=720, hor=540 + sy, cx=960 + sx, yaw=-0.35, pitch=0.42)
    members = _stack_members([STRIKE[0] + 1.25, 4.9, 5.75, 6.6, 7.45], [-0.62, -0.7, -0.66, -0.7, -0.68], T, st)
    return _exterior(fc, st, cam, T, members, [b0])


def beat2(fc, st, u):
    """98.71: along the wall at stack height: five NVG glows in a row, flags levelled; the squeeze runs up the
    line and the point man slams again: more cracks"""
    T = fc.T
    b0, b1 = st["beats"][:2]
    imp = T - b1
    sx, sy = shake(T, 8 * math.exp(-max(0, imp) * 12) if imp > 0 else 0, 35, 5)
    cam = Pin(x=7.9 - 0.2 * u, y=1.1, z=-3.6, f=900, hor=540 + sy, cx=960 + sx, yaw=-0.78, pitch=0.08)
    lunge = 0.12 * math.exp(-max(0, imp) * 6) * (imp > 0)
    members = _stack_members([STRIKE[0] + 1.25, 4.6, 5.45, 6.3, 7.15], [-0.62, -0.7, -0.66, -0.7, -0.68], T, st,
                             lunge=lunge)
    return _exterior(fc, st, cam, T, members, [b0, b1], beams=0.6)


def beat3(fc, st, u):
    """99.00-99.70: close on the strike point: the flag's butt against the riveted steel; cracks race outward and
    blaze through the long third 'Breeeach!'; dust sifts from the cracks; the frame trembles."""
    T = fc.T
    b0, b1, b2, bend = st["beats"]
    k = clamp((T - b2) / (bend - b2))
    sx, sy = shake(T, 2 + 12 * k ** 2, 40, 9)
    cam = Pin(x=STRIKE[0] + 0.7, y=STRIKE[1] + 0.25, z=-2.7 + 0.5 * ease_in(k, 2), f=1150, hor=540 + sy,
              cx=960 + sx, yaw=-0.25, pitch=-0.03, roll=0.015 * math.sin(T * 9) * k)
    cv = Canvas(fc)
    ctx = cv.ctx
    draw_sky(ctx, cam)
    draw_fortress(ctx, cam, T)
    draw_plaza(ctx, cam)
    img = cv.rgb()
    tex = noise_layer(fc, seed=8, scale=1.4, ox=3, oy=5)
    img *= (0.8 + 0.4 * tex)[..., None]
    heat = crack_heat(T, st)
    R = crack_radius(T, st)
    sp = cam.p(STRIKE[0], STRIKE[1], 0)
    img += radial_img(fc, sp[0], sp[1], 120 + 160 * k, FLAGC, 2.2) * (0.04 + 0.12 * k * k)
    img += crack_layer(fc, cam, st["cracks"], R, heat, width_k=0.35) * 0.55
    r = beat_ring(fc, cam, T, b2, k=0.6)
    if r is not None:
        img += r
    # the point man, very close, pressing the butt of his flag into the steel (backlit by the cracks)
    ch = st["dchars"][STACK[0]]
    pole = ram_pole(T, st, lean=0.05)
    layers, a = draw_member(fc, ch, cam, STRIKE[0] + 1.25, -0.62, T, "ram", facing=-1, pole_world=pole,
                            rim_col=mix(FLAG_L, (1, 1, 1), 0.5), rim_k=0.7 + 0.4 * k, rim_dir=(-1.0, 0.1),
                            tint=(SIL, 0.93), furl_light=(-0.8, 0.0, 0.4),
                            extra=dict(shape_r="grip", shape_l="grip"))
    for l in layers:
        img = comp(img, l)
    # dust sifting out of the cracks
    D = Canvas(fc)
    d = D.ctx
    for i in range(160):
        t0 = b2 + hash01(i, 5) * (bend - b2)
        if T < t0:
            continue
        a_ = T - t0
        ang_ = hash01(i, 6) * math.tau
        rr = 0.2 + (R - 0.2) * hash01(i, 7)
        wx, wy = STRIKE[0] + math.cos(ang_) * rr, STRIKE[1] + math.sin(ang_) * rr * 0.8
        p = cam.p(wx, wy - 4.9 * a_ * a_, -0.05 - a_ * 0.6)
        set_color(d, mix((0.5, 0.5, 0.55), FLAG_L, clamp(1 - rr / 5)), 0.85)
        circle(d, p[0], p[1], 1.2 + 2.6 * hash01(i, 8))
        d.fill()
    img = comp(img, D.rgba())
    img += streak(img, fc, 0.95, 500, 0.1 + 0.15 * k)
    return img


# ================================================================================================ breach
def tb_to_ts(tb):
    """speed ramp: real-time burst for 0.06 s, then slow motion (0.3x)"""
    if tb < 0.06:
        return tb
    return 0.06 + (tb - 0.06) * 0.3


LEAP = [  # (kind, seed, start x, end x, peak y, facing(screen), delay, dz)
    ("schismmancer", 3, 0.9, 4.4, 2.3, -1, 0.00, 0.0),
    ("commander", 1, -0.4, -1.6, 2.9, 1, 0.025, -1.0),
    ("schismmancer", 5, -1.6, -4.9, 2.1, 1, 0.05, 0.6),
    ("schismmancer", 2, 0.3, 2.3, 3.2, -1, 0.08, 1.4),
    ("schismmancer", 4, -2.6, -3.6, 1.8, 1, 0.11, 2.2),
]


def breach_cam(tb):
    push = ease_in_out(clamp((tb - 0.04) / 0.72), 2.0)
    return Pin(x=1.0 - 0.3 * push, y=1.35 + 0.35 * push, z=13.5 - 2.5 * push, f=950 + 350 * push,
               hor=600, cx=960, yaw=math.pi, pitch=0.12)


def leap_state(i, tb):
    kind, seed, x0, x1, yp, facing, dl, dz = LEAP[i]
    ts = tb_to_ts(tb) - dl * 0.3
    k = clamp(ts / 0.16)
    x = lerp(x0, x1 * 0.45 + x0 * 0.2, ease_out(k, 1.5))
    y = 0.25 + 0.5 * yp * math.sin(min(1.0, k * 1.2) * math.pi * 0.6)
    z = 0.4 + 6.3 * ease_out(k, 1.3) + 0.3 * dz * k
    return x, y, z, k, facing


def build_leap_flags(S, st):
    """real cloth sims: furled until each flyer clears the door, then the blast gale unrolls them mid-air.
    Simulated in a canonical 420-px frame relative to the hands; drawn in perspective every frame."""
    b0 = st["boom_f"]
    n = S.n - b0
    tracks = []
    for i, (kind, seed, x0, x1, yp, facing, dl, dz) in enumerate(LEAP):
        def pole_fn(j, facing=facing, i=i):
            x, y, z, k, _ = leap_state(i, j / S.fps)
            ang = facing * lerp(1.4, 0.5, ease_in_out(clamp(k * 1.5)))
            return (facing * 50 * k, -40 * k, ang)

        def wind_fn(j, facing=facing):
            tb = j / S.fps
            g = 1800 * math.exp(-tb * 1.5) + 1100
            return (facing * g, -0.3 * g, 250.0)

        def furl_fn(j, dl=dl):
            tb = j / S.fps
            return 1.0 - smoothstep(0.04 + dl, 0.22 + dl, tb)

        fl = Flag(pole=420, seed=20 + i, side=facing)
        tracks.append(fl.simulate(n, pole_fn, wind_fn, fps=S.fps, warmup=4, furl_fn=furl_fn))
    return tracks


def _hall(ctx, cam, lit=0.0):
    """the memeplex's inner hall: dark marble floor, massive columns framing the door (parallax as we push)"""
    ctx.rectangle(-50, -50, 2020, 1180)
    set_color(ctx, (0.015, 0.017, 0.028))
    ctx.fill()
    if cam.poly(ctx, [(-30, 0, 0), (30, 0, 0), (30, 0, 40), (-30, 0, 40)]):
        ctx.set_source(lin_grad(0, cam.p(0, 0, 0)[1], 0, 1080, [(0, (0.06, 0.065, 0.09)), (1, (0.015, 0.017, 0.025))]))
        ctx.fill()
    ctx.new_path()
    cols = [(side * 6.2, zc) for zc in (3.0, 7.5, 11.0) for side in (-1, 1)]
    cols.sort(key=lambda c: -cam.depth(c[0], 3, c[1]))
    for x, zc in cols:
        zf = zc + 0.7                      # face toward the camera
        if cam.depth(x, 3, zf) < 0.3:
            continue
        q = cam.poly(ctx, [(x - 0.7, -0.1, zf), (x + 0.7, -0.1, zf), (x + 0.7, 30, zf), (x - 0.7, 30, zf)])
        if not q:
            continue
        x0_, x1_ = min(q[0][0], q[1][0]), max(q[0][0], q[1][0])
        ctx.set_source(lin_grad(x0_, 0, x1_, 0, [(0, (0.035, 0.04, 0.06)), (0.5, (0.08, 0.09, 0.12)),
                                                 (1, (0.03, 0.035, 0.05))]))
        ctx.fill()
        if lit > 0:     # inner edge catches the flag light from the opening
            inner = q[1] if x < 0 else q[0]
            inner_t = q[2] if x < 0 else q[3]
            ctx.set_line_width(max(1.5, 0.05 * cam.k(x, 3, zf)))
            ctx.move_to(*inner)
            ctx.line_to(*inner_t)
            ctx.set_source(lin_grad(*inner, *inner_t, [(0, (*FLAG_L, 0.9 * lit)), (0.5, (*FLAG_L, 0.35 * lit)),
                                                       (1, (*FLAG_L, 0.0))]))
            ctx.stroke()


def breach(fc, st, u):
    T = fc.T
    tb = T - st["boom_T"]
    ts = tb_to_ts(tb)
    cam = breach_cam(tb)
    j = fc.f - st["boom_f"]
    sp = cam.p(STRIKE[0], STRIKE[1], 0.0)
    sx, sy = shake(T, 14 * math.exp(-tb * 5), 30, 13)
    cam.cx += sx
    cam.hor += sy
    sp = cam.p(STRIKE[0], STRIKE[1], 0.0)
    cv = Canvas(fc)
    ctx = cv.ctx
    _hall(ctx, cam, lit=smoothstep(0.0, 0.03, ts))
    draw_fortress(ctx, cam, T, inside=True)
    img = cv.rgb()
    # ---------------------------------------------------------------- the hole: cells release outward from the
    # strike point (a fracture front); light blazes only through released cells -> a jagged, growing hole
    cells = st["cracks"]["cells"]
    Lc = Canvas(fc)
    lg = rad_grad(sp[0], sp[1], 0, 700, [(0, (1, 1, 1)), (0.2, mix(GLOW, (1, 1, 1), 0.5)), (0.6, GLOW),
                                          (1, mix(FLAG_L, (0.5, 0.35, 0.1), 0.25))])
    released = []
    for ci, c in enumerate(cells):
        te = ts - c["t_rel"]
        if te > 0:
            released.append((ci, te))
            if cam.poly(Lc.ctx, [(px_, py_, 0.0) for px_, py_ in c["pts"]]):
                Lc.ctx.set_source(lg)
                Lc.ctx.fill()
    light = Lc.rgba()
    light = np.maximum(light, blur(light, 1.2 * fc.s))      # close hairline seams between cells
    open_k = smoothstep(0.0, 0.03, ts)
    img = img * (1 - light[..., 3:4]) + light[..., :3] * 0.92
    # every remaining fracture line of the door leaks light
    img += crack_layer(fc, cam, st["cracks"], 16.0, 1.6 - 0.6 * open_k, width_k=0.6) * (1.0 - 0.3 * open_k)
    # light thrown onto the hall floor and columns
    img += radial_img(fc, sp[0], cam.p(0, 0, 0)[1], 500, GLOW, 2.0) * 0.05 * open_k
    # mirror of the blazing opening on the polished marble floor
    fy_ = cam.p(0, 0, 0)[1]
    Mf = np.float32([[1, 0, 0], [0, -1, 2 * fy_ * fc.s]])
    refl = cv2.warpAffine(light, Mf, (fc.w, fc.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    xx_, yy_ = grid_xy(fc)
    fall = (np.clip((yy_ - fy_) / 8.0, 0, 1) * np.exp(-np.maximum(yy_ - fy_, 0) / 260.0))[..., None]
    img += blur(refl[..., :3], 8 * fc.s) * fall * 0.18 * open_k
    # ---------------------------------------------------------------- smoke (fluid sim), backlit by the opening
    jj = min(max(j, 0), len(st["smoke_d"]) - 1)
    dens = cv2.GaussianBlur(st["smoke_d"][jj].astype(np.float32), (0, 0), 1.2)
    Dn = np.clip(cv2.resize(dens, (fc.w, fc.h), interpolation=cv2.INTER_CUBIC), 0, None)
    # the sim lives on the door plane in the first frame's screen space: follow the camera push
    cam0 = breach_cam(0.0)
    c0 = cam0.p(0.0, 4.0, 0.0)
    c1 = cam.p(0.0, 4.0, 0.0)
    zs = cam.k(0.0, 4.0, 0.0) / cam0.k(0.0, 4.0, 0.0)
    Mz = np.float32([[zs, 0, (c1[0] - zs * c0[0]) * fc.s], [0, zs, (c1[1] - zs * c0[1]) * fc.s]])
    Dn = cv2.warpAffine(Dn, Mz, (fc.w, fc.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    od = god_rays(np.repeat(Dn[..., None], 3, 2), fc, sp[0], sp[1], length=0.7, passes=5, decay=1.0)[..., 0]
    lit = np.exp(-od * 0.45)
    alpha = np.clip(1 - np.exp(-Dn * 0.4), 0, 0.85)
    near = np.clip(blur(light[..., 3], 60 * fc.s) * 1.3, 0, 1) * open_k
    scol = (np.asarray(mix(GLOW, (1, 1, 1), 0.2), np.float32) * (lit * (0.1 + 1.25 * near))[..., None]
            + np.asarray((0.015, 0.016, 0.022), np.float32) * (1 - lit * 0.5)[..., None])
    img = img * (1 - alpha[..., None]) + scol * alpha[..., None]
    # ---------------------------------------------------------------- shards flying at the camera
    shards = Canvas(fc)
    sc_ = shards.ctx
    occ = Canvas(fc)
    oc = occ.ctx
    for ci, te in sorted(released, key=lambda r: cells[r[0]]["vz"] * r[1]):
        c = cells[ci]
        zc = c["vz"] * te
        cx_ = c["c"][0] + c["vxy"][0] * te
        cy_ = c["c"][1] + c["vxy"][1] * te - 4.9 * te * te
        if cam.depth(cx_, cy_, zc) < 0.6:
            continue
        rot = c["w"] * te
        tum = math.cos(c["tumble"] * te)
        cr, sr = math.cos(rot), math.sin(rot)
        pts = [cam.p(cx_ + px_ * tum * cr - py_ * sr, cy_ + px_ * tum * sr + py_ * cr, zc) for px_, py_ in c["pts"] - c["c"]]
        poly(sc_, pts)
        s_ = c["shade"] * 0.6
        sc_.set_source_rgb(s_, s_ * 1.05, s_ * 1.3)
        sc_.fill_preserve()
        sc_.set_line_width(max(1.0, 0.03 * cam.k(cx_, cy_, zc)))
        set_color(sc_, FLAG_L, 0.85)
        sc_.stroke()
        poly(oc, pts)
        oc.set_source_rgb(1, 1, 1)
        oc.fill()
    shard_rgba = shards.rgba()
    img = comp(img, shard_rgba)
    occl = occ.rgba()[..., 3]
    # ---------------------------------------------------------------- leapers (slow motion) + unfurling flags
    figs = []
    for i in range(len(LEAP)):
        x, y, z, k, facing = leap_state(i, tb)
        if k > 0:
            figs.append((cam.depth(x, y, z), i, x, y, z, k, facing))
    figs.sort(key=lambda f: -f[0])
    for depth_, i, x, y, z, k, facing in figs:
        ch = st["dchars"][(LEAP[i][0], LEAP[i][1])]
        kz = cam.k(x, y, z)
        h = 1.82 * kz
        fx, fy = cam.p(x, y, z)
        kw = dict(goggles_down=True, progress=clamp(0.2 + k * 0.8), wind=(facing * 1.2, -0.6),
                  silhouette=SIL, rim=(mix(GLOW, (1, 1, 1), 0.3), 1.6, (0.0, -1.0)))
        lay = Canvas(fc)
        a = ch.draw(lay.ctx, fx, fy, h, "leap", ts, facing=facing, expr="determined", **kw)
        body = lay.rgba()
        tr = st["leap_flags"][i]
        fl = Canvas(fc)
        c = fl.ctx
        hx, hy = a.get("hand_r", (fx, fy - h * 0.55))
        scl = (2.44 * kz) / 420.0
        c.translate(hx, hy)
        c.scale(scl, scl)
        c.translate(0, 420 * 0.3)
        tr.draw(c, float(min(tr.n - 1, max(0, j))), light=(0.0, -0.25, -0.95), glow=0.5)
        img = comp(img, body)
        img = comp(img, fl.rgba())
        occl = np.maximum(occl, body[..., 3])
        # smoke curls over the flyers a little
        img = img * (1 - alpha[..., None] * 0.25) + scol * alpha[..., None] * 0.25
    # ---------------------------------------------------------------- volumetric shafts + flare
    src = light[..., :3] * light[..., 3:4] * (1 - occl)[..., None] * np.exp(-Dn * 0.3)[..., None] * open_k
    rays = god_rays(src, fc, sp[0], sp[1], length=0.95, passes=7, decay=0.985)
    img += rays * 0.7
    img += streak(img, fc, 0.97, 700, 0.18)
    flash = math.exp(-max(0.0, tb) / 0.018) * 0.8
    img = img * (1 - flash) + np.asarray(mix(GLOW, (1, 1, 1), 0.5), np.float32) * flash
    # swallowed by the light: flat flag_glow on the last frame (hand-off to s05b)
    wk = smoothstep(0.54, 0.75, tb) ** 1.6
    if fc.f >= fc.n - 1:
        wk = 1.0
    img = img * (1 - wk) + FLAGGLOW * wk
    return img


# ================================================================================================ setup
def setup(S, st):
    tl = S.tl
    b = [tl.word("K03", "Breach", k)[0] for k in range(3)]
    st["beats"] = (b[0], b[1], b[2], tl.cue("breach") - 0.02)
    st["boom_T"] = S.start + st["boom_f"] / S.fps
    st["cracks"] = build_cracks()
    st["dchars"] = {(k, s): Character(k, seed=s, goggles_down=True) for k, s in STACK}
    n = S.n - st["boom_f"]
    cam = breach_cam(0.0)
    q = [cam.p(-DOOR_W / 2, 0, 0), cam.p(DOOR_W / 2, DOOR_H, 0)]
    rect = (min(q[0][0], q[1][0]), min(q[0][1], q[1][1]), max(q[0][0], q[1][0]), max(q[0][1], q[1][1]))
    sp = cam.p(STRIKE[0], STRIKE[1], 0.0)
    path = S.cache / "breach_smoke_v2.npz"
    if path.exists():
        z = np.load(path)
        st["smoke_d"] = z["d"]
    else:
        d, h = smoke.simulate(n, rect, sp, lambda k: tb_to_ts(k / S.fps) * 1.6)
        np.savez_compressed(path, d=d)
        st["smoke_d"] = d
    st["leap_flags"] = build_leap_flags(S, st)
    st["breach_rect"] = rect


SHOTS = {"beat1": beat1, "beat2": beat2, "beat3": beat3, "breach": breach}

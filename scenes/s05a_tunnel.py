"""s05a hero walk (91.2-96.1): a monumental pentagonal basalt tunnel under Flagartha, blazing gold at the far end.
Five schismmancers walk toward camera in slow motion, backlit, furled flags on their shoulders; volumetric
light shafts stream past their silhouettes through drifting dust.

World: metres, x right, y up, z away from camera. Screen projection with a pinhole camera."""
import math

import cairo
import numpy as np

from vx import W, H, C, Canvas, col, set_color, mix, circle, ellipse, poly, clamp, lerp, smoothstep, ease_in_out, \
    ease_out, hash01, noise1, fbm1, blur
from vx.chars import Character, POSES
from vx import flag as vflag
from scenes.s05a_fx import (GOLD, GOLD_HOT, NVG, lin_grad, rad_grad, rimlit, god_rays, streak, motes, radial_img,
                            comp, grid_xy)

RP = 3.7            # pentagon circumradius (m): floor width 4.35 m, apex 6.7 m
RIB_DZ = 3.4        # rib spacing
RIB_D = 0.8         # rib depth along z
RIB_IN = 0.42       # rib inset
Z_END = 75.0        # tunnel mouth (light) distance from z=0
SLOWMO = 0.42
BASALT0 = (0.055, 0.06, 0.09)


def pentagon(r):
    """flat-bottomed regular pentagon, floor at y=0 for the outer radius RP"""
    off = RP * math.sin(math.radians(54))          # 0.809 RP
    pts = []
    for a in (234, 306, 18, 90, 162):
        t = math.radians(a)
        pts.append((r * math.cos(t), off + r * math.sin(t)))
    return pts


class Cam:
    def __init__(self, x=0.0, y=1.2, z=0.0, f=1150.0, hor=560.0, cx=960.0, roll=0.0):
        self.x, self.y, self.z, self.f, self.hor, self.cx, self.roll = x, y, z, f, hor, cx, roll

    def p(self, x, y, z):
        d = max(0.05, z - self.z)
        k = self.f / d
        sx, sy = (x - self.x) * k, -(y - self.y) * k
        if self.roll:
            c, s = math.cos(self.roll), math.sin(self.roll)
            sx, sy = sx * c - sy * s, sx * s + sy * c
        return (self.cx + sx, self.hor + sy)

    def k(self, z):
        return self.f / max(0.05, z - self.cam_z) if False else self.f / max(0.05, z - self.z)


# ------------------------------------------------------------------------------------------------ set
def draw_tunnel(ctx, cam, T, portal_k=1.0):
    """basalt pentagonal tunnel: walls between ribs, ribs, floor, lamps. returns portal screen centre"""
    outer = pentagon(RP)
    inner = pentagon(RP - RIB_IN)
    zc = cam.z
    ribs = []
    z0 = math.floor((zc + 0.5) / RIB_DZ) * RIB_DZ
    z = z0
    while z < Z_END:
        if z > zc + 0.3:
            ribs.append(z)
        z += RIB_DZ
    # far mouth: blazing light
    mouth = [cam.p(x, y, Z_END) for x, y in outer]
    mcx = sum(p[0] for p in mouth) / 5
    mcy = sum(p[1] for p in mouth) / 5 + 10
    ctx.rectangle(-100, -100, 2200, 1300)
    set_color(ctx, (0.02, 0.022, 0.035))
    ctx.fill()
    # walls, far to near (each face between consecutive depth stations)
    stations = [zc + 0.2] + ribs + [Z_END]
    for zi in range(len(stations) - 1, 0, -1):
        za, zb = stations[zi - 1], stations[zi]
        dist = (Z_END - zb) / Z_END
        lit = math.exp(-dist * 3.2) * portal_k
        for fi in range(5):
            a, b = outer[fi], outer[(fi + 1) % 5]
            q = [cam.p(a[0], a[1], za), cam.p(b[0], b[1], za), cam.p(b[0], b[1], zb), cam.p(a[0], a[1], zb)]
            if fi == 0:     # floor: polished dark stone
                base = mix((0.03, 0.03, 0.045), (0.3, 0.2, 0.12), lit * 0.8)
            else:
                shade_ = (0.8, 1.0, 0.9, 0.75, 0.9)[fi]
                base = mix(BASALT0, (0.5, 0.33, 0.17), lit * 0.75)
                base = tuple(c * shade_ for c in base)
            poly(ctx, q)
            ctx.set_source(lin_grad(*cam.p(0, 1, za), *cam.p(0, 1, zb),
                                    [(0, mix(base, (0, 0, 0), 0.35)), (1, base)]))
            ctx.fill()
            # joints between wall stones
            if fi != 0 and zb - za > 1.0:
                set_color(ctx, (0, 0, 0), 0.35)
                ctx.set_line_width(max(0.6, 1.4 * cam.f / max(1.0, za - zc) * 0.02))
                for s in (0.33, 0.66):
                    m0 = (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)
                    ctx.move_to(*cam.p(m0[0], m0[1], za))
                    ctx.line_to(*cam.p(m0[0], m0[1], zb))
                    ctx.stroke()
        # rib at za (front face ring + inner band), skipped for the camera station
        if zi - 1 >= 1:
            zr = za
            dist = (Z_END - zr) / Z_END
            lit = math.exp(-dist * 3.2) * portal_k
            # inner band (visible sides of the rib's inner surface)
            for fi in range(5):
                a, b = inner[fi], inner[(fi + 1) % 5]
                q = [cam.p(a[0], a[1], zr), cam.p(b[0], b[1], zr), cam.p(b[0], b[1], zr + RIB_D),
                     cam.p(a[0], a[1], zr + RIB_D)]
                poly(ctx, q)
                set_color(ctx, mix((0.06, 0.06, 0.08), (0.75, 0.5, 0.25), lit * 0.9))
                ctx.fill()
            ctx.new_path()
            po = [cam.p(x, y, zr) for x, y in outer]
            pi_ = [cam.p(x, y, zr) for x, y in inner]
            poly(ctx, po)
            poly(ctx, pi_[::-1])
            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            set_color(ctx, mix((0.035, 0.037, 0.055), (0.25, 0.16, 0.09), lit * 0.5))
            ctx.fill()
            ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
            # gold rim where the rib's inner edge is silhouetted against the lit tunnel beyond
            pb = [cam.p(x, y, zr + RIB_D) for x, y in inner]
            ctx.set_line_width(max(1.0, 0.035 * cam.f / max(0.5, zr - zc)))
            set_color(ctx, GOLD_HOT, min(1.0, 0.25 + 1.2 * lit))
            poly(ctx, pb)
            ctx.stroke()
            # cool service lamps at the base of each rib
            k = cam.f / max(0.3, zr - zc)
            for sgn in (-1, 1):
                lx = sgn * (RP * 0.588 - 0.1)
                lp = cam.p(lx, 0.35, zr - 0.05)
                r = min(5.0, max(1.2, 0.03 * k))
                ctx.set_source(rad_grad(lp[0], lp[1], 0, r * 4, [(0, (0.55, 0.75, 1.0, 0.45)), (1, (0.4, 0.6, 1.0, 0))]))
                circle(ctx, lp[0], lp[1], r * 4)
                ctx.fill()
                set_color(ctx, (0.8, 0.92, 1.0))
                circle(ctx, lp[0], lp[1], r)
                ctx.fill()
    # the mouth itself
    poly(ctx, mouth)
    ctx.set_source(rad_grad(mcx, mcy, 0, max(40.0, abs(mouth[2][0] - mouth[4][0])),
                            [(0, (1, 0.97, 0.9)), (0.35, GOLD_HOT), (1, (0.95, 0.62, 0.25))]))
    ctx.fill()
    return mcx, mcy


# ------------------------------------------------------------------------------------------------ figures
SQUAD = [  # (kind, seed, x offset m, z offset m, phase offset)
    ("schismmancer", 3, -2.5, 2.6, 0.31),
    ("schismmancer", 5, 2.45, 2.4, 0.77),
    ("schismmancer", 2, -1.3, 1.25, 0.55),
    ("schismmancer", 4, 1.35, 1.2, 0.12),
    ("commander", 1, 0.0, 0.0, 0.0),
]
STEP = 1.05          # seconds per full walk cycle in slow motion (two footfalls)
BODY_H = 1.82


def squad_pos(T, T0, z0, speed=0.62):
    """world positions of the squad at time T (walking toward -z)"""
    out = []
    for kind, seed, dx, dz, ph in SQUAD:
        z = z0 + dz - speed * (T - T0)
        x = dx + 0.05 * math.sin((T - T0) * 1.7 + seed)
        out.append((kind, seed, x, z, ph))
    return out


def footfalls(T0, T1, z0):
    """global times of every heel strike in [T0, T1] -> [(t, kind, idx)]"""
    ev = []
    for i, (kind, seed, dx, dz, ph) in enumerate(SQUAD):
        k0 = math.floor((T0 - T0) / (STEP / 2)) - 2
        for k in range(k0, int((T1 - T0) / (STEP / 2)) + 3):
            t = T0 + (k * 0.5 - ph) * STEP
            if T0 <= t < T1:
                ev.append((t, kind, i, dx))
    return sorted(ev)


def draw_figure(ch, fc, cam, x, z, T, ph, speaking=None, fill=0.16, rim_k=1.4, facing=0, pose_extra=None,
                furl_light=(0.0, -0.3, -0.9), cam_blur=0.0, mouth=0.0, head_turn=None, radio=False):
    """render one backlit schismmancer (+ furled flag on the shoulder) -> premultiplied layers [back, flag, front].
    The rig does the lighting: a dark cool tint (fill) and a clean outer rim from the gold light behind."""
    k = cam.f / max(0.3, z - cam.z)
    h = BODY_H * k
    fx, fy = cam.p(x, 0.0, z)
    phase = ((T * 1.0) / STEP + ph) % 1.0
    pose = {"walk": 1.0, "shoulder_flag": 1.0}
    if pose_extra:
        pose.update(pose_extra)
    kw = dict(phase=phase, speed=0.45, goggles_down=False, wind=(0.12, 0.0),
              tint=((0.035, 0.04, 0.07), 1.0 - fill), rim=(GOLD_HOT, rim_k, (0.0, -1.0)), light=(0.0, -1.0))
    if head_turn is not None:
        kw["head_turn"] = head_turn
    L = h * 1.35
    a = ch.anchors(fx, fy, h, pose, T, facing=facing, pole_len=L, **kw)
    if radio:       # back hand pressed to the headset while he talks
        ex0, ey = a.get("ear", a["head"])
        ex = 2 * a["head"][0] - ex0          # the ear on the back-hand side
        kw["hand_l"] = (ex + h * 0.01, ey + h * 0.02)
        kw["shape_l"] = "pinch"
        a = ch.anchors(fx, fy, h, pose, T, facing=facing, pole_len=L, **kw)
    lay_b = Canvas(fc)
    ch.draw(lay_b.ctx, fx, fy, h, pose, T, mouth=mouth, facing=facing, pole_len=L, layer="back",
            expr="determined", **kw)
    lay_f = Canvas(fc)
    ch.draw(lay_f.ctx, fx, fy, h, pose, T, mouth=mouth, facing=facing, pole_len=L, layer="front",
            expr="determined", **kw)
    fl = Canvas(fc)
    bx, by, ang = a["pole"]
    vflag.draw_furled(fl.ctx, bx, by, pole=L, ang=ang, seed=int(abs(x) * 10) % 4, light=furl_light, glow=0.35)
    layers = []
    for lay in (lay_b, fl, lay_f):
        rgba = lay.rgba()
        if cam_blur > 0:
            rgba = blur(rgba, cam_blur * fc.s)
        layers.append(rgba)
    return layers, dict(feet=(fx, fy), h=h, anchors=a)


# ------------------------------------------------------------------------------------------------ shots
def _walk_common(fc, st, cam, T, T0, z0, fig_fill, rim_k, mcu=False):
    cv = Canvas(fc)
    mcx, mcy = draw_tunnel(cv.ctx, cam, T)
    img = cv.rgb()
    # haze around the mouth (atmosphere)
    img += radial_img(fc, mcx, mcy, 260, GOLD, 1.6) * 0.55
    img += radial_img(fc, mcx, mcy, 60, GOLD_HOT, 2.0) * 0.6
    # squad, far to near
    squad = squad_pos(T, T0, z0)
    squad = sorted(squad, key=lambda s: -s[3])
    occl = np.zeros((fc.h, fc.w), np.float32)
    refl = np.zeros((fc.h, fc.w, 4), np.float32)
    for kind, seed, x, z, ph in squad:
        if z - cam.z < 0.4:
            continue
        ch = st["chars"][(kind, seed)]
        speaking = kind == "commander"
        mouth = fc.tl.mouth("COMMANDER", fc.T) if speaking else 0.0
        blur_px = 0.0
        if mcu and kind != "commander":
            blur_px = 14.0
        layers, info = draw_figure(ch, fc, cam, x, z, T - T0, ph, fill=fig_fill if speaking else 0.1,
                                   rim_k=rim_k, cam_blur=blur_px, mouth=mouth, radio=speaking and mcu)
        for l in layers:
            img = comp(img, l)
            occl = occl + l[..., 3] * (1 - occl)
        # floor reflection (mirror about the feet line)
        fy = info["feet"][1]
        body = layers[0]
        off = int(round(2 * fy * fc.s))
        flipped = np.flipud(body)
        sh = off - fc.h
        if -fc.h < sh < fc.h:
            m = np.zeros_like(flipped)
            if sh >= 0:
                m[sh:] = flipped[:fc.h - sh]
            else:
                m[:fc.h + sh] = flipped[-sh:]
            refl = m + refl * (1 - m[..., 3:4])
    # reflections: blurred, darkened, only below the horizon
    if refl[..., 3].max() > 0:
        rb = blur(refl, 5 * fc.s)
        xx, yy = grid_xy(fc)
        fade = np.clip((yy - cam.hor) / 200.0, 0, 1)[..., None] * 0.45
        img = img * (1 - rb[..., 3:4] * fade) + rb[..., :3] * fade * 0.6
    # the floor's glossy streak of the mouth light
    xx, yy = grid_xy(fc)
    dy = np.maximum(yy - mcy, 0)
    streak_ = np.exp(-((xx - mcx) / (18 + dy * 0.25)) ** 2) * np.clip(dy / 20, 0, 1) * np.exp(-dy / 380)
    img += (streak_ * (1 - occl))[..., None] * np.asarray(GOLD, np.float32) * 0.55
    # volumetric light shafts: bright mouth + haze, occluded by the squad
    light = np.clip(img - 0.55, 0, None) * (1 - occl)[..., None]
    rays = god_rays(light, fc, mcx, mcy, length=0.9, passes=7, decay=0.97)
    img += rays * 1.25
    # dust motes in the beam
    mc = Canvas(fc)

    def lit(x, y):
        d = math.hypot(x - mcx, (y - mcy) * 1.3)
        return clamp(1.2 - d / 900)
    motes(mc.ctx, T, 170, 7, (0, 80, 1920, 1000), lit, rmin=1.0, rmax=3.2 if not mcu else 9.0, drift=(6.0, -3.0),
          alpha=0.85, speed=SLOWMO)
    img = comp(img, mc.rgba())
    img += streak(img, fc, 0.97, 520, 0.18)
    return img, occl


def walk_wide(fc, st, u):
    """91.21-93.29: wide, the squad approaches out of the light; slow dolly back, slight low angle"""
    T = fc.T
    d = u
    cam = Cam(x=0.15 * math.sin(d * 0.4), y=1.05, z=-0.35 * d, f=1100, hor=585 + 3 * math.sin(d * 0.7), cx=960)
    img, _ = _walk_common(fc, st, cam, T, st["walk_T0"], 4.9, fig_fill=0.14, rim_k=1.1)
    return img


def walk_mcu(fc, st, u):
    """93.29-96.08: MCU of the Commander on the radio, tracking backward with him; squad soft behind"""
    T = fc.T
    T0 = st["walk_T0"]
    squad = squad_pos(T, T0, 4.9)
    cz = [s for s in squad if s[0] == "commander"][0][3]
    cam = Cam(x=-0.12 + 0.05 * math.sin(u * 0.5), y=1.5, z=cz - 3.1 - 0.03 * u, f=1850, hor=520, cx=960)
    img, _ = _walk_common(fc, st, cam, T, T0, 4.9, fig_fill=0.4, rim_k=1.1, mcu=True)
    return img


def setup(S, st):
    st["chars"] = {(k, s): Character(k, seed=s, goggles_down=False) for k, s, *_ in SQUAD}
    st["walk_T0"] = S.start + st["shot"]["walk"]["f0"] / S.fps


SHOTS = {"walk": walk_wide, "mcu": walk_mcu}

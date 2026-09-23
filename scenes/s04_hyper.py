"""s04 — THE HYPERMIND: a colossal gold dilution-refrigerator chandelier (stacked gold plates, hundreds of
coiled coax lines, a cold tip with an emitter) in true 3D perspective, plus its readout ring.

Everything is procedural and deterministic (seeded). Draw with draw_hypermind(ctx, gctx, cam, st, T, ...):
ctx = main canvas, gctx = additive glow canvas.
"""
import math

import cairo
import numpy as np

from vx.canvas import col
from vx.config import FONT_MONO
from scenes.s04_geo import polyline3, poly_screen, path_pts

# plates: (Y top, radius, thickness)
TIERS = [(5300.0, 1150.0, 78.0), (4700.0, 1065.0, 66.0), (4090.0, 965.0, 60.0),
         (3490.0, 850.0, 54.0), (2930.0, 725.0, 48.0), (2410.0, 590.0, 44.0)]
TIP_TOP = TIERS[-1][0] - TIERS[-1][2]      # underside of the mixing-chamber plate
CAN_R, CAN_BOT = 175.0, 1960.0
TIP_Y = 1760.0                              # emitter
CROWN_R = 330.0
HUB_Y = 9200.0                              # suspension hub (truss) in the shaft

# metallic gold ramp (amber/copper: kept apart from the Flag's lemon yellow)
_RAMP = np.array([col("#140801"), col("#3E1E05"), col("#84480E"), col("#C9801F"), col("#F2B44A"),
                  col("#FFE9B8")], np.float32)
EMIT = np.array(col("#FFB456"), np.float32)
EMIT_HOT = np.array(col("#FFE6C0"), np.float32)


def gold(m):
    """m in [0,1] -> gold colour"""
    m = float(np.clip(m, 0, 1)) * (len(_RAMP) - 1)
    i = min(int(m), len(_RAMP) - 2)
    f = m - i
    return tuple(_RAMP[i] * (1 - f) + _RAMP[i + 1] * f)


def _metal(nx, nz, cam, light_dir, base=0.42, amp=0.36):
    """shade value for a vertical surface with horizontal outward normal (nx, nz)"""
    ld = light_dir
    diff = nx * ld[0] + nz * ld[2]
    # view-dependent sheen: brighter where the normal points toward the camera horizontally
    vx, vz = cam.pos[0], cam.pos[2]
    L = math.hypot(vx, vz) + 1e-6
    fac = (nx * vx + nz * vz) / L
    return base + amp * diff + 0.34 * max(0.0, fac) ** 10 + 0.12 * max(0.0, fac) ** 2 - 0.26 * (1 - abs(fac))


# ------------------------------------------------------------------ geometry (seeded)
def build(seed=11):
    rng = np.random.default_rng(seed)
    segs = []
    for k in range(len(TIERS) - 1):
        y0 = TIERS[k][0] - TIERS[k][2]
        y1 = TIERS[k + 1][0]
        r_hi, r_lo = TIERS[k][1], TIERS[k + 1][1]
        cables = []
        n = 62 - k * 5
        for i in range(n):
            th = (i + rng.uniform(-0.3, 0.3)) / n * 2 * math.pi
            rho = r_lo * rng.uniform(0.42, 0.93)
            rho2 = rho * rng.uniform(0.96, 1.04)
            kind = rng.choice(["coil", "coil", "s", "loop", "straight"], p=[0.3, 0.15, 0.25, 0.15, 0.15])
            m = 34
            t = np.linspace(0, 1, m)
            r = rho + (rho2 - rho) * t
            a = th + t * rng.uniform(-0.05, 0.05)
            y = y0 + (y1 - y0) * t
            X = r * np.sin(a)
            Z = r * np.cos(a)
            ux, uz = np.sin(th), np.cos(th)          # radial
            tx, tz = np.cos(th), -np.sin(th)         # tangential
            mid = rng.uniform(0.35, 0.65)
            if kind == "s":
                off = rng.uniform(25, 60) * np.sin(np.clip((t - mid + 0.2) / 0.4, 0, 1) * math.pi) * rng.choice([-1, 1])
                X = X + ux * off
                Z = Z + uz * off
            elif kind == "loop":
                # hairpin loop sticking out radially
                lr = rng.uniform(22, 45)
                tt = np.clip((t - mid + 0.12) / 0.24, 0, 1)
                ang = tt * 2 * math.pi
                X = X + ux * lr * (1 - np.cos(ang)) * 0.9
                Z = Z + uz * lr * (1 - np.cos(ang)) * 0.9
                y = y + lr * np.sin(ang) * 0.9
            if kind == "coil":
                # helical coil (a real pigtail): 4-7 turns
                m = 90
                t = np.linspace(0, 1, m)
                r = rho + (rho2 - rho) * t
                a = th + t * 0.02
                y = y0 + (y1 - y0) * t
                X = r * np.sin(a)
                Z = r * np.cos(a)
                turns = rng.integers(4, 8)
                cr = rng.uniform(12, 26)
                span = rng.uniform(0.22, 0.4)
                tt = np.clip((t - mid + span / 2) / span, 0, 1)
                env = np.sin(tt * math.pi) ** 0.3 * (tt > 0) * (tt < 1)
                ph = tt * turns * 2 * math.pi
                X = X + env * cr * (ux * np.cos(ph) + tx * np.sin(ph))
                Z = Z + env * cr * (uz * np.cos(ph) + tz * np.sin(ph))
            P = np.stack([X, y, Z], 1)
            w = rng.uniform(5, 11) * (1.0 - 0.06 * k)
            cables.append(dict(P=P, th=th, rho=rho, w=w, kind=kind, tone=rng.uniform(-0.08, 0.08)))
        posts = []
        for j in range(6):
            th = j / 6 * 2 * math.pi + k * 0.3
            rr = r_lo * 0.97
            posts.append(dict(P=np.array([[rr * math.sin(th), y0, rr * math.cos(th)],
                                          [rr * math.sin(th), y1, rr * math.cos(th)]]), th=th, w=26.0))
        # spiral heat exchanger around the axis (split into half-turn pieces for depth sorting)
        if 1 <= k <= 3:
            hr = r_lo * 0.3
            turns = 3.5
            m = 200
            t = np.linspace(0, 1, m)
            a = t * turns * 2 * math.pi + k
            y = y0 - 60 + (y1 - y0 + 120) * (0.2 + 0.6 * t)
            P = np.stack([hr * np.sin(a), y, hr * np.cos(a)], 1)
            per = m / (turns * 2)
            for j in range(int(turns * 2)):
                i0, i1 = int(j * per), int(min(m, (j + 1) * per + 2))
                am = float(a[(i0 + i1) // 2])
                cables.append(dict(P=P[i0:i1], th=am, rho=hr, w=30.0, kind="hx", tone=0.05))
        # central still line (thick axial tube)
        cables.append(dict(P=np.array([[0.0, y0, 0.0], [0.0, y1, 0.0]]), th=0.0, rho=0.0, w=90.0, kind="axis",
                           tone=-0.02))
        segs.append(dict(k=k, y0=y0, y1=y1, cables=cables, posts=posts))
    # crown cables (bundle rising into the shaft)
    crown = []
    for i in range(40):
        th = i / 40 * 2 * math.pi + rng.uniform(-0.05, 0.05)
        rr = CROWN_R * rng.uniform(0.75, 1.0)
        y = np.linspace(TIERS[0][0], HUB_Y, 12)
        crown.append(dict(P=np.stack([rr * np.sin(th) + 0 * y, y, rr * np.cos(th) + 0 * y], 1), th=th,
                          w=rng.uniform(14, 26)))
    # bolt holes / details per plate
    details = []
    for k, (y, r, th_) in enumerate(TIERS):
        nb = int(r / 26)
        details.append(dict(bolts=np.linspace(0, 2 * math.pi, nb, endpoint=False) + k * 0.1))
    return dict(segs=segs, crown=crown, details=details)


# ------------------------------------------------------------------ drawing helpers
def _circle3(r, y, n=96):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.sin(a), np.full(n, y), r * np.cos(a)], 1), a


def _draw_band(ctx, cam, r, ytop, ybot, light_dir, fog, n=96, emit=0.0, emit_col=EMIT, tone=0.0):
    """side of a vertical cylinder (only camera-facing quads, near-clipped), metallic shading"""
    a = np.linspace(0, 2 * math.pi, n + 1)
    top = np.stack([r * np.sin(a), np.full(n + 1, ytop), r * np.cos(a)], 1)
    bot = np.stack([r * np.sin(a), np.full(n + 1, ybot), r * np.cos(a)], 1)
    sx_t, sy_t, z_t = cam.proj(top)
    sx_b, sy_b, z_b = cam.proj(bot)
    cp = cam.pos
    for i in range(n):
        am = (a[i] + a[i + 1]) / 2
        nx, nz = math.sin(am), math.cos(am)
        # facing test (vertical cylinder)
        if (cp[0] - r * nx) * nx + (cp[2] - r * nz) * nz <= 0:
            continue
        zmin = min(z_t[i], z_t[i + 1], z_b[i], z_b[i + 1])
        if max(z_t[i], z_t[i + 1], z_b[i], z_b[i + 1]) < cam.near:
            continue
        v = _metal(nx, nz, cam, light_dir) + tone
        c = np.array(gold(v))
        c = c * (1 - fog) + fog_col_default * fog + emit * emit_col
        if zmin < cam.near:
            pts = poly_screen(cam, np.array([top[i], top[i + 1], bot[i + 1], bot[i]]))
            if pts is None:
                continue
            path_pts(ctx, pts)
        else:
            ctx.move_to(sx_t[i], sy_t[i])
            ctx.line_to(sx_t[i + 1], sy_t[i + 1])
            ctx.line_to(sx_b[i + 1], sy_b[i + 1])
            ctx.line_to(sx_b[i], sy_b[i])
            ctx.close_path()
        ctx.set_source_rgb(*np.clip(c, 0, 1))
        ctx.fill_preserve()
        ctx.set_line_width(0.8)
        ctx.stroke()


fog_col_default = np.array(col("#0E1528"), np.float32)


def _draw_disc(ctx, cam, r, y, from_above, light_pos, fog, pulse, k, details, T):
    """top or bottom face of a plate: brushed gold with a soft reflection, engraved rings, bolts"""
    ring, a = _circle3(r, y, 128)
    pts = poly_screen(cam, ring)
    if pts is None:
        return
    # gradient: along the screen projection of the light direction
    c0 = cam.proj(np.array([0.0, y, 0.0]))
    if c0[2] < cam.near:
        return
    # brushed-gold sheen: a linear gradient across the disc, perpendicular to the view
    lx, lz = light_pos[0], light_pos[2]
    L = math.hypot(lx, lz) + 1e-6
    far = cam.proj(np.array([-lx / L * r, y, -lz / L * r]))
    near = cam.proj(np.array([lx / L * r, y, lz / L * r]))
    if far[2] < cam.near or near[2] < cam.near:
        far, near = (pts[:, 0].min(), pts[:, 1].mean(), 1), (pts[:, 0].max(), pts[:, 1].mean(), 1)
    if from_above:
        stops = [(0.0, 0.18), (0.45, 0.42), (0.62, 0.55), (1.0, 0.3)]
    else:
        # underside: lit from below by the chamber + emissive core
        p = 0.1 * pulse
        stops = [(0.0, 0.3 + p), (0.38, 0.52 + p), (0.55, 0.74 + p), (0.7, 0.5 + p), (1.0, 0.34)]
    g = cairo.LinearGradient(float(far[0]), float(far[1]), float(near[0]), float(near[1]))
    for t, v in stops:
        c = np.array(gold(v)) * (1 - fog) + fog_col_default * fog
        g.add_color_stop_rgb(t, *np.clip(c, 0, 1))
    path_pts(ctx, pts)
    ctx.set_source(g)
    ctx.fill()
    # engraved rings
    for rr, a_ in ((0.84, 0.35), (0.62, 0.25), (0.36, 0.3)):
        rp = poly_screen(cam, _circle3(r * rr, y, 96)[0])
        if rp is not None:
            path_pts(ctx, rp)
            ctx.set_source_rgba(*gold(0.12), a_ * (1 - fog))
            ctx.set_line_width(max(0.6, 5.0 * cam.f / max(c0[2], 1)))
            ctx.stroke()
    # bolts: small dark dots on a circle
    bolts = details["bolts"]
    Pb = np.stack([r * 0.93 * np.sin(bolts), np.full(len(bolts), y), r * 0.93 * np.cos(bolts)], 1)
    bx, by, bz = cam.proj(Pb)
    rad = 7.0 * cam.f / np.maximum(bz, 1)
    ctx.set_source_rgba(*gold(0.1), 0.8 * (1 - fog))
    for i in range(len(bolts)):
        if bz[i] > cam.near and rad[i] > 0.4:
            ctx.arc(bx[i], by[i], rad[i], 0, 2 * math.pi)
            ctx.fill()


def _draw_cable(ctx, cam, cb, light_dir, fog, dim=0.0, emit=0.0):
    P = cb["P"]
    th = cb["th"]
    v = _metal(math.sin(th), math.cos(th), cam, light_dir, base=0.36, amp=0.3) + cb["tone"] - dim
    c = np.array(gold(v)) * (1 - fog) + fog_col_default * fog
    hl = np.array(gold(v + 0.32)) * (1 - fog) + fog_col_default * fog + emit * EMIT * 0.5
    w = polyline3(ctx, cam, P, np.clip(c * 0.55, 0, 1), cb["w"] * 1.25, max_w=40)
    if w is None:
        return
    polyline3(ctx, cam, P, np.clip(c, 0, 1), cb["w"], max_w=34)
    if w > 1.6:
        polyline3(ctx, cam, P, np.clip(hl, 0, 1), cb["w"] * 0.32, max_w=12)


# ------------------------------------------------------------------ main draw
def _clear_hull(gctx, cam, P):
    """erase glow behind an opaque convex object (painter order keeps halos physically occluded)"""
    import cv2
    C = cam.to_cam(P)
    if (C[:, 2] < cam.near).any():
        return
    sx = cam.cx + cam.f * C[:, 0] / C[:, 2]
    sy = cam.cy - cam.f * C[:, 1] / C[:, 2]
    hull = cv2.convexHull(np.stack([sx, sy], 1).astype(np.float32))[:, 0]
    gctx.save()
    gctx.set_operator(cairo.OPERATOR_CLEAR)
    path_pts(gctx, hull)
    gctx.fill()
    gctx.restore()


def tier_emit(tl, T, k, energy=None):
    """emissive level of plate k: voice pulse cascading down the stages (+ H02 energy drop)"""
    m = tl.mouth("HYPERMIND", T - 0.035 * k)
    e = 0.12 + 1.35 * m
    if energy is not None:
        e += energy(k)
    return e


def draw_hypermind(ctx, gctx, cam, st, T, tl, light_dir, light_pos, fogf, energy=None, core=1.0,
                   tip_glow=0.0, alpha_all=1.0):
    """draw the chandelier. fogf(P) -> fog amount for a world point. energy(k) -> extra emission of stage k
    (k = 0..5 plates, 6 = can/tip). core = interior glow strength."""
    geo = st["hm"]
    Yc = cam.pos[1]
    items = []
    for k, (y, r, th) in enumerate(TIERS):
        items.append((abs((y - th / 2) - Yc), "plate", k))
    for s in geo["segs"]:
        items.append((abs((s["y0"] + s["y1"]) / 2 - Yc) - 1.0, "seg", s["k"]))
    items.append((abs(7200 - Yc), "crown", 0))
    items.append((abs((TIP_TOP + TIP_Y) / 2 - Yc) - 1.0, "tip", 0))
    items.sort(key=lambda x: -x[0])
    for _, kind, k in items:
        if kind == "plate":
            y, r, th = TIERS[k]
            fog = fogf(np.array([0.0, y, 0.0]))
            e = tier_emit(tl, T, k, energy)
            above = Yc > y
            below = Yc < y - th
            _clear_hull(gctx, cam, np.vstack([_circle3(r, y, 48)[0], _circle3(r, y - th, 48)[0]]))
            if above:
                _draw_band(ctx, cam, r, y, y - th, light_dir, fog)
                _draw_disc(ctx, cam, r, y, True, light_pos, fog, e, k, geo["details"][k], T)
            elif below:
                _draw_disc(ctx, cam, r, y - th, False, light_pos, fog, e, k, geo["details"][k], T)
                _draw_band(ctx, cam, r, y, y - th, light_dir, fog)
            else:
                _draw_band(ctx, cam, r, y, y - th, light_dir, fog)
            # H02 energy: a bright shock-ring flares around the plate as the pulse passes
            ek = energy(k) if energy is not None else 0.0
            if ek > 0.05:
                for rr_, aa_ in ((1.0 + 0.25 * min(ek, 1.8), 0.9), (1.0 + 0.1 * min(ek, 1.8), 1.0)):
                    rp = poly_screen(cam, _circle3(r * rr_, y - th / 2, 128)[0])
                    if rp is None:
                        continue
                    zz = cam.proj(np.array([0.0, y, 0.0]))[2]
                    for c_, w_, a_ in ((ctx, 22.0, 0.55), (gctx, 70.0, 0.9)):
                        path_pts(c_, rp)
                        c_.set_source_rgba(*EMIT_HOT, min(1.0, a_ * aa_ * ek / 1.8) * alpha_all)
                        c_.set_line_width(max(1.5, w_ * cam.f / max(zz, 1)))
                        c_.stroke()
            # emissive rim (bottom edge) -> main + glow
            ring = _circle3(r * 1.002, y - th, 128)[0]
            pts = poly_screen(cam, ring)
            if pts is not None:
                z = cam.proj(np.array([0.0, y, 0.0]))[2]
                lw = max(1.0, 9.0 * cam.f / max(z, 1))
                ce = np.clip(EMIT * (0.5 + 0.6 * e) + EMIT_HOT * 0.25 * e, 0, 1)
                for c_, a_, w_ in ((ctx, min(1.0, 0.25 + 0.45 * e), lw), (gctx, min(1.0, 0.12 + 0.4 * e), lw * 3.0)):
                    path_pts(c_, pts)
                    c_.set_source_rgba(*ce, a_ * alpha_all)
                    c_.set_line_width(w_)
                    c_.stroke()
        elif kind == "seg":
            s = geo["segs"][k]
            ymid = (s["y0"] + s["y1"]) / 2
            fog = fogf(np.array([0.0, ymid, 0.0]))
            e = 0.5 * (tier_emit(tl, T, k, energy) + tier_emit(tl, T, k + 1, energy))
            # sort cables by depth
            cabs = s["cables"]
            mids = np.array([c_["P"][len(c_["P"]) // 2] for c_ in cabs])
            zs = cam.to_cam(mids)[:, 2]
            zaxis = cam.to_cam(np.array([0.0, ymid, 0.0]))[2]
            order = np.argsort(-zs)
            back = [i for i in order if zs[i] >= zaxis]
            front = [i for i in order if zs[i] < zaxis]
            for i in back:
                _draw_cable(ctx, cam, cabs[i], light_dir, fog, dim=0.12)
            # posts behind
            for p in s["posts"]:
                pz = cam.to_cam(p["P"][0])[2]
                if pz >= zaxis:
                    _draw_cable(ctx, cam, dict(P=p["P"], th=p["th"], w=p["w"], tone=0.02), light_dir, fog, dim=0.1)
            # interior core glow (between plates): backlights the front cables
            _core_glow(ctx, gctx, cam, s, e * core, alpha_all)
            for p in s["posts"]:
                pz = cam.to_cam(p["P"][0])[2]
                if pz < zaxis:
                    _draw_cable(ctx, cam, dict(P=p["P"], th=p["th"], w=p["w"], tone=0.04), light_dir, fog)
            for i in front:
                _draw_cable(ctx, cam, cabs[i], light_dir, fog)
        elif kind == "crown":
            # the suspension bundle rising into the shaft: dark above, lit by the chamber below
            ys = [TIERS[0][0], 6000, 6800, 7600, 8400, HUB_Y]
            for j in range(len(ys) - 1):
                ym = (ys[j] + ys[j + 1]) / 2
                tone = -0.1 - 0.32 * (1 - math.exp(-(ym - TIERS[0][0]) / 2500.0))
                _draw_band(ctx, cam, CROWN_R, ys[j + 1], ys[j], light_dir, fogf(np.array([0.0, ym, 0.0])),
                           n=48, tone=tone)
            fog = fogf(np.array([0.0, 7000.0, 0.0]))
            cz = cam.to_cam(np.array([c_["P"][0] for c_ in geo["crown"]]))[:, 2]
            for i in np.argsort(-cz):
                cb = geo["crown"][i]
                _draw_cable(ctx, cam, dict(P=cb["P"], th=cb["th"], w=cb["w"], tone=-0.2), light_dir, fog)
        else:
            _draw_tip(ctx, gctx, cam, light_dir, fogf, tl, T, energy, tip_glow, alpha_all)


def _core_glow(ctx, gctx, cam, s, e, alpha_all):
    y0, y1 = s["y0"], s["y1"]
    c0 = cam.proj(np.array([0.0, y0, 0.0]))
    c1 = cam.proj(np.array([0.0, y1, 0.0]))
    if c0[2] < cam.near or c1[2] < cam.near:
        return
    r_lo = TIERS[s["k"] + 1][1]
    zr = (c0[2] + c1[2]) / 2
    rad = r_lo * 0.75 * cam.f / zr
    mx, my = (c0[0] + c1[0]) / 2, (c0[1] + c1[1]) / 2
    hh = abs(c1[1] - c0[1]) / 2 + rad * 0.3
    if not (-900 < mx < 2820 and -900 < my < 1980) or rad > 3000 or hh > 3000:
        return
    for c_, a_ in ((ctx, 0.3), (gctx, 0.45)):
        c_.save()
        c_.translate(mx, my)
        c_.scale(max(rad, 1), max(hh, 1))
        g = cairo.RadialGradient(0, 0, 0, 0, 0, 1)
        g.add_color_stop_rgba(0, *EMIT_HOT, min(1, a_ * e) * alpha_all)
        g.add_color_stop_rgba(0.35, *EMIT, min(1, a_ * e * 0.6) * alpha_all)
        g.add_color_stop_rgba(1, *EMIT, 0)
        c_.set_source(g)
        c_.arc(0, 0, 1, 0, 2 * math.pi)
        c_.fill()
        c_.restore()


def _draw_tip(ctx, gctx, cam, light_dir, fogf, tl, T, energy, tip_glow, alpha_all):
    fog = fogf(np.array([0.0, 1600.0, 0.0]))
    # mixing chamber can
    _draw_band(ctx, cam, CAN_R, TIP_TOP, CAN_BOT, light_dir, fog, n=64, tone=0.03)
    # cone to the emitter
    n = 48
    a = np.linspace(0, 2 * math.pi, n + 1)
    tip = np.array([0.0, TIP_Y, 0.0])
    ring = np.stack([CAN_R * np.sin(a), np.full(n + 1, CAN_BOT), CAN_R * np.cos(a)], 1)
    sx, sy, sz = cam.proj(ring)
    tx, ty, tz = cam.proj(tip)
    cp = cam.pos
    if tz > cam.near:
        for i in range(n):
            am = (a[i] + a[i + 1]) / 2
            nx, nz = math.sin(am), math.cos(am)
            ny = CAN_R / (CAN_BOT - TIP_Y)
            P0 = ring[i]
            nrm = np.array([nx, -ny, nz])
            if np.dot(cp - P0, nrm) <= 0:
                continue
            v = _metal(nx, nz, cam, light_dir) - 0.05
            c = np.array(gold(v)) * (1 - fog) + fog_col_default * fog
            ctx.move_to(sx[i], sy[i])
            ctx.line_to(sx[i + 1], sy[i + 1])
            ctx.line_to(tx, ty)
            ctx.close_path()
            ctx.set_source_rgb(*np.clip(c, 0, 1))
            ctx.fill_preserve()
            ctx.set_line_width(0.7)
            ctx.stroke()
        # emitter
        e = tier_emit(tl, T, 6, energy) * 0.6 + tip_glow
        rad = max(2.0, 30 * cam.f / tz)
        for c_, k in ((ctx, 1.0), (gctx, 2.8)):
            g = cairo.RadialGradient(tx, ty, 0, tx, ty, rad * k * (1 + e))
            g.add_color_stop_rgba(0, *EMIT_HOT, min(1, 0.6 + e) * alpha_all)
            g.add_color_stop_rgba(0.4, *EMIT, min(1, 0.4 * (0.5 + e)) * alpha_all)
            g.add_color_stop_rgba(1, *EMIT, 0)
            c_.set_source(g)
            c_.arc(tx, ty, rad * k * (1 + e), 0, 2 * math.pi)
            c_.fill()


# ------------------------------------------------------------------ readout ring
def draw_readout_ring(ctx, gctx, cam, T, text, y=2300.0, r=760.0, size=58.0, alpha=1.0, base=0.0, hot=0.0):
    """a holographic ring of mono characters around the axis, reading left->right on the camera side
    (front half bright, back half faint), framed by two thin HUD rings. base = angle of character 0."""
    for dy in (size * 0.75, -size * 0.62):
        rp = poly_screen(cam, _circle3(r, y + dy, 160)[0])
        if rp is not None and alpha > 0.01:
            for c_, w_, a_ in ((ctx, 1.6, 0.55), (gctx, 5.0, 0.35)):
                path_pts(c_, rp)
                c_.set_source_rgba(*EMIT, a_ * min(1.0, alpha))
                c_.set_line_width(w_)
                c_.stroke()
    n = len(text)
    step = 2 * math.pi / n
    ctx.save()
    cp = cam.pos
    L = math.hypot(cp[0], cp[2]) + 1e-6
    for c_ in (ctx, gctx):
        c_.select_font_face(FONT_MONO, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        a = base - i * step
        P = np.array([r * math.sin(a), y, r * math.cos(a)])
        sx, sy, z = cam.p2(P)
        if z < cam.near:
            continue
        fac = (math.sin(a) * cp[0] + math.cos(a) * cp[2]) / L        # +1 = facing camera
        tangent = np.array([-math.cos(a), 0.0, math.sin(a)])         # reading direction
        t1 = cam.p2(P + tangent * 10)
        sq = (t1[0] - sx) / (10 * cam.f / z)
        sz = size * cam.f / z
        if sz < 2 or sz > 900 or not math.isfinite(sz) or not (-400 < sx < 2320 and -400 < sy < 1480):
            continue
        a_ = alpha * (0.22 + 0.78 * max(0.0, fac) ** 0.7) if fac > -0.2 else alpha * 0.1
        c = np.clip(EMIT * 0.55 + EMIT_HOT * 0.45 + hot * 0.4, 0, 1) if fac > 0 else EMIT
        for c_, aa in ((ctx, a_), (gctx, a_ * 0.3)):
            c_.save()
            c_.translate(sx, sy)
            c_.scale(sq if abs(sq) > 0.05 else 0.05, 1)
            c_.set_font_size(sz)
            ext = c_.text_extents(ch)
            c_.move_to(-ext.x_advance / 2, sz * 0.35)
            c_.text_path(ch)
            c_.set_source_rgba(*c, min(1.0, aa))
            c_.fill()
            c_.restore()
    ctx.restore()

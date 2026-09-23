"""s03 — The Noospheric Munitions Act (47.5–61.5).

Signature: a physically convincing CRT broadcast of a talking jaguar president, on a curved-glass TV
floating half-submerged in beige slop in a flooded living room lit only by the broadcast itself.

Pipeline per frame:
  broadcast signal (s03_broadcast) -> analog chain: tuning snow, vertical roll, sync jitter, NTSC chroma
  -> tube (s03_crt: barrel faceplate, beam scanlines, aperture grille, halation, collapse)
  -> the room (s03_room) as albedo layers x a light map driven by the broadcast's own mean colour
  -> glossy slop (perspective ripple field, per-object mirrored reflections, fresnel, marbling)
Helpers: scenes/s03_layout.py (geometry, camera, float physics), s03_static.py (hand-off snow for s02).
"""
import math
import os

import cv2
import numpy as np

from vx import *
from vx.canvas import text_width
from vx.flag import FlagTrack
from vx import foley

from scenes import s03_layout as lay
from scenes import s03_crt as crt
from scenes import s03_room as room
from scenes import s03_broadcast as bc
from scenes.s03_static import signal_px, tuning, glass_params

POST = dict(bloom=0.42, bloom_thresh=0.74, bloom_radius=20.0, vignette=0.3, grain=0.03, ca=0.0)


# ================================================================ setup
def _flag_key():
    import vx.flag as vf
    p = vf.__file__
    st = os.stat(p)
    extra = ""
    sim = os.path.join(os.path.dirname(p), "flag_sim.py")
    if os.path.exists(sim):
        extra = str(int(os.stat(sim).st_mtime))
    return f"{int(st.st_mtime)}_{extra}_{bc.FLAG_VERSION}"


def setup(S):
    tl = S.tl
    beats = bc.Beats(tl)
    key = _flag_key()
    flags = []
    paths = [S.cache / f"studio_flag{k}_{key}.npz" for k in range(len(bc.FLAG_XS))]
    if all(p.exists() for p in paths):
        try:
            flags = [FlagTrack.load(p) for p in paths]
        except Exception:
            flags = []
    if not flags:
        flags = bc.build_flags(S, S.n)
        for tr, p in zip(flags, paths):
            try:
                tr.save(p)
            except Exception:
                pass
    import cairo
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 8, 8)
    tick_w = text_width(cairo.Context(surf), bc.TICKER, 21, font=bc.NARROW, bold=True, tracking=0.03)
    st = dict(tl=tl, beats=beats, flags=flags, tick_w=tick_w, tm=lay.times())
    export_hits(S, st)
    return st


def export_hits(S, st):
    b, tm = st["beats"], st["tm"]
    ev = [
        dict(t=tm["tv_on"], kind="tv_on", strength=0.8, pan=0.05,
             desc="CRT already on: full-frame analog snow (hiss) starts tuning in; tube whine"),
        dict(t=tm["tv_on"] + 0.05, kind="static", strength=0.7, pan=0.05,
             desc="static hiss fades out 47.5->48.05 as the broadcast locks (vertical roll x2)"),
        dict(t=tm["lock"], kind="click", strength=0.5, pan=0.05,
             desc="vertical hold locks, picture stabilises, colour kicks in; muffled fanfare through TV speaker"),
        dict(t=b.slam, kind="impact", strength=1.0, pan=0.05,
             desc="President's fist SLAMS the podium on 'banned!' (through the TV speaker: clipped, boomy)"),
        dict(t=b.slam + 0.03, kind="slosh", strength=0.8, pan=0.05,
             desc="the floating TV itself jolts and dips in the slop: heavy viscous slosh + ripples"),
        dict(t=b.slam + 0.02, kind="paper", strength=0.6, pan=0.0,
             desc="the Act unrolls down the podium front (paper whoosh + flap), 57.0->57.32"),
        dict(t=b.slam + 0.08, kind="rattle", strength=0.35, pan=0.2,
             desc="rabbit-ear antenna rattles/boings from the jolt"),
        dict(t=b.slam + 0.45, kind="slosh", strength=0.45, pan=-0.65,
             desc="slosh wave reaches the floating armchair (cat startles, ears flat)"),
        dict(t=b.pen, kind="pen_scratch", strength=0.8, pan=0.0,
             desc="giant pen signature flourish START (fast loopy scribble, velocity peaks mid-way)"),
        dict(t=b.pen + 0.36, kind="pen_scratch_end", strength=0.5, pan=0.0,
             desc="signature flourish END (final flick)"),
        dict(t=b.j02["start"] - 0.05, kind="whoosh", strength=0.35, pan=0.0,
             desc="studio camera snap-zoom + President leans in conspiratorially"),
        dict(t=b.a_[0] - 0.15, kind="wink", strength=0.5, pan=0.0, desc="the wink (cartoon 'ting')"),
        dict(t=b.coin - 0.45, kind="whoosh", strength=0.5, pan=0.6,
             desc="gold pricecoin slides in from off-screen right (spinning)"),
        dict(t=b.coin, kind="ka_ching", strength=1.0, pan=0.0, desc="coin drops into the breast pocket: KA-CHING"),
        dict(t=b.coin + 0.06, kind="sparkle", strength=0.6, pan=0.0, desc="sparkle glint on the pocket"),
        dict(t=b.coin + 0.05, kind="pat", strength=0.4, pan=0.0, desc="he pats the pocket (1/2)"),
        dict(t=b.coin + 0.2, kind="pat", strength=0.3, pan=0.0, desc="he pats the pocket (2/2)"),
        dict(t=b.pen + 0.42, kind="whoosh", strength=0.3, pan=-0.3,
             desc="he tosses the giant pen over his shoulder (spinning out of frame)"),
        dict(t=tm["off"], kind="tv_off", strength=1.0, pan=0.0,
             desc="CRT switch-off: picture collapses to a bright horizontal line (61.0->61.1), room light dies"),
        dict(t=tm["off"] + 0.12, kind="tv_off_dot", strength=0.6, pan=0.0,
             desc="line collapses to a white dot (61.12->61.24); high whine decays"),
        dict(t=tm["off"] + 0.45, kind="black", strength=0.2, pan=0.0, desc="dot fades out: full black by 61.45"),
    ]
    for t0, x, k in bc.flash_times(b):
        ev.append(dict(t=t0, kind="camera_flash", strength=0.3 * k, pan=(x - 400) / 400 * 0.4,
                       desc="press-pit camera shutter + flash (inside the broadcast, through the TV speaker)"))
    # slop drips from the photo frames (room tone detail)
    for fr in room.FRAMES:
        x, y, w, h, rot, kind, seed = fr
        nd = 3 + int(3 * hash01(seed, 99))
        for k in range(1):
            per = 2.2 + 2.6 * hash01(k, seed * 13 + 1)
            ph = hash01(k, seed * 13 + 2) * per
            t = S.start
            # drip resets when (T+ph) % per wraps
            n0 = math.ceil((S.start + ph) / per)
            while True:
                t = n0 * per - ph
                if t > S.end:
                    break
                ev.append(dict(t=t, kind="drip", strength=0.15, pan=foley.screen_pan(x),
                               desc="slop drip from a wall photo lands in the flood"))
                n0 += 1
    foley.export_events(S, "hits", ev)


# ================================================================ helpers
def cam_apply(ctx, cam, sh):
    cx, cy, z = cam
    ctx.translate(960 + sh[0], 540 + sh[1])
    ctx.scale(z, z)
    ctx.translate(-cx, -cy)


def layer(fc, cam, sh, fn, *a):
    cv = Canvas(fc)
    ctx = cv.ctx
    ctx.save()
    cam_apply(ctx, cam, sh)
    fn(ctx, *a)
    ctx.restore()
    return cv.rgba()


def paste(dst, patch, x0, y0):
    """premultiplied patch over dst (h,w,3|4) at pixel x0,y0"""
    ph, pw = patch.shape[:2]
    reg = dst[y0:y0 + ph, x0:x0 + pw]
    if dst.shape[2] == 4:
        reg[...] = patch + reg * (1.0 - patch[..., 3:4])
    else:
        reg[...] = patch[..., :3] + reg * (1.0 - patch[..., 3:4])


def mirror_rows(img, r):
    """reflect image rows about row r (rows >= r get rows <= r)"""
    h = img.shape[0]
    out = np.zeros_like(img)
    r = int(round(r))
    if r < 0:
        return out
    if r >= h:
        return out
    n = min(h - r, r + 1)
    out[r:r + n] = img[r - n + 1:r + 1][::-1]
    return out


# ================================================================ the analog chain
def tube_state(T, b, tm):
    """per-frame analog/tube parameters"""
    p = dict(csx=1.0, csy=1.0, bright=1.0, on=1.0)
    # tuning in (shared with the s02 hand-off static): snow gives way, picture rolls twice and locks
    p.update(tuning(T))
    # the slam: sync tear + jitter (the transmission itself flinches)
    d = T - b.slam
    if d >= 0:
        k = math.exp(-d * 9)
        p["tear"] += 1.1 * k
        p["tear_y"] = 0.62 - 0.4 * min(1, d * 3)
        p["jitter"] += 5.0 * k
        p["bright"] *= 1.0 + 0.25 * k
    # switch off
    off = tm["off"]
    if T >= off:
        a = seg(T, off, off + 0.1, lambda x: ease_in(x, 2.2))
        p["csy"] = max(0.0035, 1.0 - a)
        bb = seg(T, off + 0.1, off + 0.24, lambda x: ease_in(x, 2.0))
        p["csx"] = max(0.004, 1.0 - bb)
        p["jitter"] = 0.0
    return p


def dot_state(T, tm):
    off = tm["off"]
    if T < off + 0.08:
        return 0.0, 0.0
    # the lingering phosphor dot: flares, then shrinks and fades
    a = seg(T, off + 0.08, off + 0.2, ease_out) * (1 - seg(T, off + 0.2, off + 0.5, lambda x: ease_in_out(x, 1.6)))
    r = lerp(1.0, 0.25, seg(T, off + 0.18, off + 0.5, ease_in_out))
    return a, r


# ================================================================ render
def render(fc, st):
    T = fc.T
    s = fc.s
    w, h = fc.w, fc.h
    b, tm = st["beats"], st["tm"]
    cam = lay.camera(T)
    sh = lay.cam_shake(T)
    z = cam[2]
    tx, ty, trot = lay.tv_pose(T)
    tcx, tcy, thw, thh, _ = lay.tube_geom(T, s)
    ts = tube_state(T, b, tm)

    # ---------------------------------------------------------- the signal
    wpx = signal_px(thw)
    upx = wpx / crt.SIG_W
    sig = bc.render_signal(T, wpx, st, fc.F)
    hpx = sig.shape[0]
    # the transmitter's white clip: a soft knee so white fur doesn't burn into a blob on the phosphor
    knee = 0.62
    sig = np.where(sig > knee, knee + (sig - knee) * 0.42, sig)
    if ts["snow"] > 0:
        sn = crt.snow_signal(fc.F, wpx, hpx)
        k = ts["snow"]
        sig = sig * (1 - k) * (1 - 0.3 * k) + sn * k
    sig = crt.ntsc(sig, upx, ts["chroma"])
    lum_mean = crt.raster_mean(sig)
    on = 1.0 if T < tm["off"] else max(0.0, 1.0 - seg(T, tm["off"], tm["off"] + 0.12, ease_in))
    # emitted light colour/intensity of the tube (drives the room)
    tv_rgb = np.clip(lum_mean * 1.0 + 0.08, 0, 1.5) * (0.55 + 1.9 * float(lum_mean.mean()))
    tv_rgb = tv_rgb * on
    if T >= tm["off"]:
        # the collapsing line flashes the room once
        tv_rgb = tv_rgb + np.array([0.7, 0.8, 1.0], np.float32) * 0.35 * pulse(T, tm["off"] + 0.07, 0.04)

    # ---------------------------------------------------------- light map (quarter res, world space)
    qs = 4
    lw, lh = max(8, w // qs), max(8, h // qs)
    ys = (np.arange(lh, dtype=np.float32) + 0.5) * (h / lh) / s
    xs = (np.arange(lw, dtype=np.float32) + 0.5) * (w / lw) / s
    wx = (xs[None, :] - 960 - sh[0]) / z + cam[0]
    wy = (ys[:, None] - 540 - sh[1]) / z + cam[1]
    lp = room.lamp_power(T)
    amb, room_light, glass_refl = glass_params(T, lp, tm["off"])
    d2 = (wx - tx) ** 2 + ((wy - ty) * 1.15) ** 2
    f_tv = (1.0 / (1.0 + d2 / (330.0 ** 2)))[..., None] * 1.25
    # slight directional emphasis: the room in front/below the screen is brighter than above it
    f_tv = f_tv * (0.8 + 0.35 * np.clip((wy - ty + 200) / 500, 0, 1))[..., None]
    lp = room.lamp_power(T)
    lbx, lby, ltx, lty, lang = room.lamp_geom(T)
    d2l = (wx - ltx) ** 2 + (wy - lty - 30) ** 2
    f_lamp = (1.0 / (1.0 + d2l / (230.0 ** 2)))[..., None] * lp
    lamp_rgb = np.array([1.05, 0.55, 0.42], np.float32)
    x0w, y0w, ww, wh = room.WINDOW
    dm = ((wx - (x0w + ww * 0.7)) / 380.0) ** 2 + ((wy - (y0w + wh * 0.9)) / 300.0) ** 2
    moon = (np.exp(-dm) * 0.22)[..., None] * np.array([0.6, 0.72, 1.0], np.float32)
    L_tv = f_tv * tv_rgb[None, None, :]
    L_rest = amb + f_lamp * lamp_rgb + moon
    L_all = L_rest + L_tv
    L_cab = L_rest + L_tv * 0.28

    def up(L):
        return cv2.resize(L.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    L_all_f = up(L_all)
    L_cab_f = up(L_cab)

    def lit(rgba, L):
        out = rgba.copy()
        out[..., :3] *= L
        return out

    # ---------------------------------------------------------- layers
    wall = layer(fc, cam, sh, _draw_wall_all, T)
    wall_l = lit(wall, L_all_f)
    ext = layer(fc, cam, sh, room.draw_exterior_emit, T)
    wall_l[..., :3] += ext[..., :3]
    mull = layer(fc, cam, sh, _draw_window_front, T)
    wall_l = mull_over(wall_l, lit(mull, L_all_f))
    # light bouncing off the rippling slop dances on the wall above the flood line
    wall_l[..., :3] += wall_caustics(fc, T, cam, sh, tx, tv_rgb, b)

    lamp = lit(layer(fc, cam, sh, room.draw_lamp, T), L_all_f)
    lamp_e = layer(fc, cam, sh, room.draw_lamp_emit, T)
    lamp[..., :3] += lamp_e[..., :3]
    lamp[..., 3] = np.maximum(lamp[..., 3], lamp_e[..., 3] * 0.0)

    led = on
    tv = lit(layer(fc, cam, sh, _draw_tv_all, T, tx, ty, trot), L_cab_f)
    ant = lit(layer(fc, cam, sh, _draw_antenna, T, tx, ty, trot), L_all_f)
    tv = tv + ant * (1.0 - tv[..., 3:4])
    tv_e = layer(fc, cam, sh, _draw_tv_emit, T, tx, ty, trot, led)
    tv[..., :3] += tv_e[..., :3]
    r = crt.crt_render(sig, tcx, tcy, thw, thh, trot, w, h, fc.F, roll=ts["roll"], jitter=ts["jitter"],
                       tear=ts["tear"], tear_y=ts["tear_y"], csx=ts["csx"], csy=ts["csy"], bright=ts["bright"],
                       room_light=room_light, refl=glass_refl,
                       lamp_refl=0.16 * lp * seg(T, tm["lock"], tm["lock"] + 0.5, ease_in_out), s=s)
    if r is not None:
        # after switch-off the dark glass still reflects the room
        paste(tv, r[2], r[0], r[1])
    # the lingering dot + phosphor afterglow
    da, dr = dot_state(T, tm)
    tsx, tsy = tcx / s, tcy / s          # tube centre in design px

    def rim_to_tv(lay_rgba, wxo, wyo, width=3.2, k=1.0):
        """rim light on the silhouette edges that face the tube"""
        ox, oy = lay.world_to_screen(wxo, wyo, cam)
        vx, vy = tsx - ox, tsy - oy
        n = math.hypot(vx, vy) or 1.0
        dd = (wxo - tx) ** 2 + ((wyo - ty) * 1.15) ** 2
        f = 1.0 / (1.0 + dd / (330.0 ** 2)) * 1.25
        col_ = tv_rgb * f * k
        if float(col_.max()) < 0.01:
            return
        d = width * z * s
        a = lay_rgba[..., 3]
        M = np.float32([[1, 0, -vx / n * d], [0, 1, -vy / n * d]])
        ash = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        edge = np.clip(a - ash, 0, 1)
        lay_rgba[..., :3] += edge[..., None] * col_.astype(np.float32)

    chair = lit(layer(fc, cam, sh, room.draw_chair, T), L_all_f)
    ccx, ccy, _ = room.chair_pose(T)
    rim_to_tv(chair, ccx + 150, ccy - 200, 4.0, 1.8)
    chair_e = layer(fc, cam, sh, room.draw_cat_emit, T, None)
    chair[..., :3] += chair_e[..., :3] * (0.35 + 0.9 * float(tv_rgb.mean()))
    rim_to_tv(lamp, ltx, lty, 2.4, 0.9)

    debris = lit(layer(fc, cam, sh, room.draw_debris, T), L_all_f)
    rim_to_tv(debris, tx, 900.0, 2.0, 0.8)
    deb_e = layer(fc, cam, sh, room.draw_debris_emit, T)
    debris[..., :3] += deb_e[..., :3]

    # ---------------------------------------------------------- the slop
    def row(yw):
        return (540 + sh[1] + (yw - cam[1]) * z) * s

    wl_wall = row(lay.WALL_WL)
    wl_tv = row(lay.TV_Y + lay.TV_WL_LOCAL)
    wl_lamp = row(room.LAMP_WL)
    wl_chair = row(room.CHAIR_WL)
    slop = slop_surface(fc, T, cam, sh, st, L_all_f, [
        (wall_l, wl_wall), (lamp, wl_lamp), (tv, wl_tv), (chair, wl_chair)], wl_wall, tm, b)

    # ---------------------------------------------------------- composite
    out = wall_l[..., :3].copy()
    y_w = int(max(0, min(h, math.floor(wl_wall))))
    if y_w < h:
        # soft 1.5 px seam at the wall waterline
        yy = np.arange(y_w, h, dtype=np.float32)[:, None, None]
        m = np.clip((yy - wl_wall) / (1.5 * s) + 0.5, 0, 1)
        out[y_w:] = out[y_w:] * (1 - m) + slop[y_w:] * m
    ov = layer(fc, cam, sh, _draw_slop_overlay, T)
    out = over(out, lit(ov, L_all_f))
    out = over(out, lamp)
    out = over(out, tv)
    if b.slam - 0.01 <= T <= b.slam + 1.3:
        c = math.cos(trot)
        spl = layer(fc, cam, sh, room.draw_splash, T, b.slam + 0.06, tx + lay.CAB_L * c, tx + lay.CAB_R * c,
                    lay.TV_Y + lay.TV_WL_LOCAL)
        out = over(out, lit(spl, L_all_f))
    k_air = on * seg(T, tm["lock"], tm["lock"] + 0.5, ease_in_out)   # frame 0 must equal the s02 hand-off static
    if k_air > 0.01:
        motes = layer(fc, cam, sh, _draw_motes, T, (tx, ty), tuple(float(v) for v in tv_rgb))
        out += motes[..., :3] * k_air
    # moonlight falling through the window, a faint beam in the dusty air
    beam = layer(fc, cam, sh, _draw_moonbeam, T)
    out += beam[..., :3] * (1.0 - 0.8 * seg(T, tm["off"], tm["off"] + 0.25, ease_in))
    out = over(out, chair)
    out = over(out, debris)
    # light from the screen scattering in the room's air (soft volumetric haze)
    if r is not None and k_air > 0.01:
        k8 = 8
        sw_, sh_ = max(4, w // k8), max(4, h // k8)
        small = np.zeros((sh_, sw_, 3), np.float32)
        x0p, y0p, pt = r
        ps = cv2.resize(pt[..., :3], (max(1, pt.shape[1] // k8), max(1, pt.shape[0] // k8)), interpolation=cv2.INTER_AREA)
        xa, ya = x0p // k8, y0p // k8
        hh2, ww2 = min(ps.shape[0], sh_ - ya), min(ps.shape[1], sw_ - xa)
        if hh2 > 0 and ww2 > 0:
            small[ya:ya + hh2, xa:xa + ww2] = ps[:hh2, :ww2]
            haze = cv2.GaussianBlur(small, (0, 0), max(1.0, 60.0 * z * s / k8))
            out += cv2.resize(haze, (w, h), interpolation=cv2.INTER_LINEAR) * (0.16 * k_air)
    if da > 0:
        out = add_dot(out, tcx, tcy, thw, da, dr, s)
    return out


def _draw_moonbeam(ctx, T):
    import cairo
    x, y, w, h = room.WINDOW
    ctx.set_operator(cairo.OPERATOR_ADD)
    dx = 330.0
    yb = lay.WALL_WL + 110
    ctx.move_to(x + 10, y + 20)
    ctx.line_to(x + w, y + 20)
    ctx.line_to(x + w + dx, yb)
    ctx.line_to(x + dx * 0.55, yb)
    ctx.close_path()
    fl = 0.8 + 0.2 * math.sin(T * 0.7)
    ctx.set_source(lin_grad(x, y, x + dx * 0.6, yb, [(0, "#9FB4E8", 0.0), (0.25, "#9FB4E8", 0.07 * fl),
                                                     (1, "#9FB4E8", 0.0)]))
    ctx.fill()


def wall_caustics(fc, T, cam, sh, tx, tv_rgb, b):
    """TV light reflected off the undulating slop, dancing on the wall just above the flood line"""
    w, h, s = fc.w, fc.h, fc.s
    z = cam[2]
    q = 4
    lw, lh = max(8, w // q), max(8, h // q)
    ys = (np.arange(lh, dtype=np.float32) + 0.5) * (h / lh) / s
    xs = (np.arange(lw, dtype=np.float32) + 0.5) * (w / lw) / s
    wx = ((xs[None, :] - 960 - sh[0]) / z + cam[0]).astype(np.float32)
    wy = ((ys[:, None] - 540 - sh[1]) / z + cam[1]).astype(np.float32)
    top = lay.WALL_WL - 340.0
    my = np.clip((wy - top) / (lay.WALL_WL - top), 0, 1) ** 1.6 * (wy < lay.WALL_WL)
    mx = np.exp(-((wx - tx) / 480.0) ** 2)
    m = my * mx
    if float(m.max()) < 1e-3:
        return 0.0
    u = wx * 0.018 + 1.4 * np.sin(wy * 0.035 + T * 1.3) + 0.8 * np.sin(wx * 0.007 - T * 0.6)
    v = wy * 0.045 - T * 0.9 + 1.1 * np.sin(wx * 0.021 + T * 0.7)
    c1 = (1 - np.abs(np.sin(u))) ** 6
    c2 = (1 - np.abs(np.sin(v + 0.5 * np.sin(u * 1.3)))) ** 6
    c = c1 * 0.5 + c2 * 0.35 + c1 * c2 * 1.4
    agit = 1.0 + 1.6 * math.exp(-max(0.0, T - b.slam) * 1.1) * (T >= b.slam)
    img = (c * m * 0.34 * agit)[..., None] * np.asarray(tv_rgb, np.float32)[None, None, :]
    return cv2.resize(img.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def add_dot(out, cx, cy, hw, a, r, s):
    h, w = out.shape[:2]
    rad = max(2.0, hw * 0.02 * r)
    x0, x1 = int(max(0, cx - rad * 12)), int(min(w, cx + rad * 12))
    y0, y1 = int(max(0, cy - rad * 12)), int(min(h, cy + rad * 12))
    if x1 <= x0 or y1 <= y0:
        return out
    xx = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5 - cx
    yy = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5 - cy
    d2 = (xx * xx + yy * yy) / (rad * rad)
    core = np.exp(-d2 * 1.2) * 2.2 + np.exp(-d2 * 0.08) * 0.35
    col = np.array([0.92, 0.96, 1.0], np.float32)
    out[y0:y1, x0:x1] += (core * a)[..., None] * col
    return out


def mull_over(dst, src):
    return src + dst * (1.0 - src[..., 3:4])


# ---------------------------------------------------------------- draw bundles
def _draw_wall_all(ctx, T):
    room.draw_wall(ctx, T)


def _draw_window_front(ctx, T):
    room.draw_window_mullions(ctx)
    room.draw_curtains(ctx, T)


def _draw_tv_all(ctx, T, tx, ty, rot):
    ctx.save()
    room.clip_above(ctx, lay.TV_Y + lay.TV_WL_LOCAL, T, amp=1.6, seed=2)
    ctx.translate(tx, ty)
    ctx.rotate(rot)
    room.draw_tv(ctx, T)
    ctx.restore()
    c = math.cos(rot)
    room.draw_meniscus(ctx, tx + lay.CAB_L * c - 4, tx + lay.CAB_R * c + 4, lay.TV_Y + lay.TV_WL_LOCAL, T, amp=1.6,
                       seed=2, lift=7, drop=9)


def _draw_motes(ctx, T, tv_xy, tv_rgb):
    import cairo
    ctx.set_operator(cairo.OPERATOR_ADD)
    room.draw_motes_emit(ctx, T, tv_xy, tv_rgb)


def _draw_antenna(ctx, T, tx, ty, rot):
    ctx.save()
    ctx.translate(tx, ty)
    ctx.rotate(rot)
    room.draw_antenna(ctx, T)
    ctx.restore()


def _draw_tv_emit(ctx, T, tx, ty, rot, led):
    ctx.save()
    ctx.translate(tx, ty)
    ctx.rotate(rot)
    room.draw_tv_emit(ctx, T, led)
    ctx.restore()


def _tv_waterline_clip(fc, cam, sh, T):
    return None


def _draw_slop_overlay(ctx, T):
    """soft contact shadows around everything floating + the bobbing slop floaters"""
    tx, ty, rot = lay.tv_pose(T)
    items = [(tx + 55, lay.TV_Y + lay.TV_WL_LOCAL, 330, 1.0)]
    lbx, lby, _, _, _ = room.lamp_geom(T)
    cx, cy, cr = room.chair_pose(T)
    items.append((cx + 10, room.CHAIR_WL, 230, 1.0))
    for kind, x, wl, sc, seed, r in room.debris_list():
        items.append((x, wl, room.DEBRIS_HW.get(kind, 40) * sc, 0.7))
    for x, y, half, k in items:
        ctx.save()
        ctx.translate(x, y + 5)
        ctx.scale(1.0, 0.13)
        ctx.arc(0, 0, half * 1.2, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source(rad_grad(x, y + 5, half * 0.3, half * 1.2, [(0, "#16121A", 0.5 * k), (1, "#16121A", 0.0)]))
        ctx.fill()
    room.draw_floaters(ctx, T)


# ================================================================ slop surface
_slop_cache = {}


def slop_surface(fc, T, cam, sh, st, L, layers, wl_wall, tm, b):
    """glossy viscous slop: perspective ripple field + per-object mirrored reflections + fresnel"""
    w, h = fc.w, fc.h
    s = fc.s
    z = cam[2]
    y_w = int(max(0, math.floor(wl_wall)))
    out = np.zeros((h, w, 3), np.float32)
    if y_w >= h:
        return out
    # half-res working grid
    hs = 2
    W2, H2 = max(4, w // hs), max(4, h // hs)
    y0 = y_w // hs
    rows = H2 - y0
    if rows <= 1:
        return out
    ys = ((np.arange(y0, H2, dtype=np.float32) + 0.5) * hs) / s          # design px
    xs = ((np.arange(W2, dtype=np.float32) + 0.5) * hs) / s
    eye = 540 + sh[1] + (lay.EYE_Y - cam[1]) * z
    wallr = wl_wall / s
    q = np.clip((wallr - eye) / np.maximum(ys - eye, 1e-3), 0.02, 1.0)[:, None]  # 1 at wall, ->0 near
    # surface coordinates (world units)
    Xs = ((xs[None, :] - 960 - sh[0]) / z + cam[0] - 960) * q * 1.0 + 960
    Zs = 1400.0 * q
    # viscous swell
    hgt = (3.0 * np.sin(Xs * 0.011 + Zs * 0.018 - T * 0.9)
           + 2.2 * np.sin(-Xs * 0.017 + Zs * 0.026 - T * 1.25 + 1.3)
           + 1.3 * np.sin(Xs * 0.031 + Zs * 0.041 - T * 1.7 + 2.1)
           + 0.8 * np.sin(Xs * 0.052 - Zs * 0.047 - T * 2.3 + 0.4))
    # rings: the TV's buoyant bob and the slam
    tx, ty, _ = lay.tv_pose(T)
    tvq = np.clip((wallr - eye) / max(((lay.TV_Y + lay.TV_WL_LOCAL - cam[1]) * z + 540 + sh[1]) - eye, 1e-3), 0.02, 1)
    tX = (tx + 55 - 960) * tvq + 960
    tZ = 1400.0 * tvq
    rr = np.sqrt((Xs - tX) ** 2 + ((Zs - tZ) * 2.2) ** 2)
    hgt += 1.6 * np.sin(rr * 0.09 - T * 4.0) * np.exp(-rr / 260.0)
    d = T - b.slam
    if d > 0:
        front = 40 + 210 * d
        env = np.exp(-((rr - front) / (60 + 40 * d)) ** 2) * math.exp(-d * 0.9)
        hgt += 9.0 * np.sin((rr - front) * 0.07) * env
    # chair bob rings
    cx, cy, _ = room.chair_pose(T)
    cq = np.clip((wallr - eye) / max(((room.CHAIR_WL - cam[1]) * z + 540 + sh[1]) - eye, 1e-3), 0.02, 1)
    cX = (cx - 960) * cq + 960
    rc = np.sqrt((Xs - cX) ** 2 + ((Zs - 1400 * cq) * 2.2) ** 2)
    hgt += 1.2 * np.sin(rc * 0.08 - T * 3.4) * np.exp(-rc / 200.0)
    hgt += (0.45 * np.sin(Xs * 0.063 + Zs * 0.051 - T * 1.5 + 0.7 * np.sin(Zs * 0.02 + T * 0.3))
            + 0.3 * np.sin(-Xs * 0.057 + Zs * 0.071 - T * 1.8 + 2.0))
    gy, gx = np.gradient(hgt.astype(np.float32))
    # slopes in world terms (px gradients corrected for perspective foreshortening)
    dXdp = q * hs / (s * z)
    dZdp = 1400.0 * q * hs / (s * np.maximum(ys - eye, 1e-3))[:, None]
    slx = gx / dXdp
    slz = gy / dZdp
    # displacement (half-res px) of the reflected image; vertical smear dominates (glossy viscous liquid)
    dx = -gx * 2.4 * s * z
    dy = -gy * 7.5 * s * z
    # ---- reflection image (half res) from mirrored layers
    R = np.zeros((H2, W2, 3), np.float32)
    for img, wl in layers:
        small = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_AREA)
        m = mirror_rows(small, wl / hs)
        R = m[..., :3] + R * (1.0 - m[..., 3:4])
    gxs = np.arange(W2, dtype=np.float32)[None, :] + dx
    gys = np.arange(y0, H2, dtype=np.float32)[:, None] + dy
    Rw = cv2.remap(R, gxs.astype(np.float32), gys.astype(np.float32), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    bz = max(0.5, s * z)
    Rw = cv2.GaussianBlur(Rw, (0, 0), sigmaX=1.1 * bz, sigmaY=2.6 * bz)
    # ---- diffuse slop body: marbled beige / mauve / flesh, cream ribbons
    mq = 4
    MX = Xs[::mq, ::mq] * 0.0045
    MZ = Zs[::mq, ::mq] * 0.0045 + T * 0.05
    ncols = MX.shape[1]
    MX = np.broadcast_to(MX, (MZ.shape[0], ncols))
    MZb = np.broadcast_to(MZ, MX.shape)
    n1 = fbm2(MX + 0.3 * np.sin(MZb * 2 + T * 0.2), MZb, seed=31, octaves=3)
    n2 = fbm2(MX * 2.1 + 5.0, MZb * 2.1, seed=47, octaves=2)
    n1 = cv2.resize(n1.astype(np.float32), (W2, rows), interpolation=cv2.INTER_CUBIC)
    n2 = cv2.resize(n2.astype(np.float32), (W2, rows), interpolation=cv2.INTER_CUBIC)
    beige = np.array(C["slop_beige"], np.float32)
    mauve = np.array(C["slop_mauve"], np.float32)
    flesh = np.array(C["slop_flesh"], np.float32)
    cyan = np.array(C["slop_cyan"], np.float32)
    a1 = np.clip(0.5 + 1.1 * n1, 0, 1)[..., None]
    a2 = np.clip(0.3 + 1.6 * n2, 0, 1)[..., None] * 0.5
    body = beige * (1 - a1 * 0.5) + mauve * a1 * 0.5
    body = body * (1 - a2) + flesh * a2
    ph = (Xs * 0.006 + n1 * 2.4 + Zs * 0.004) * 6.0
    ribbon = np.clip(1 - np.abs(np.sin(ph)) * 3, 0, 1)[..., None] * 0.2
    body = body * (1 - ribbon) + np.array([0.96, 0.92, 0.86], np.float32) * ribbon
    rib2 = np.clip(1 - np.abs(np.sin(ph * 0.5 + 1.3 + n2 * 3)) * 4, 0, 1)[..., None] * 0.12
    body = body * (1 - rib2) + cyan * rib2
    Lh = cv2.resize(L, (W2, H2), interpolation=cv2.INTER_AREA)[y0:]
    # light: diffuse with swell shading (slopes tilted back toward the TV catch more), a warm body glow
    nsh = np.clip(1.0 + 3.0 * slz - 0.6 * slx, 0.7, 1.35)[..., None]
    fill = np.array([0.075, 0.06, 0.055], np.float32) * (1.0 - seg(T, tm["off"], tm["off"] + 0.2, ease_in))
    dif = body * (Lh * 0.78 + fill * 0.7) * nsh
    fres = (0.05 + 0.45 * (np.clip(q, 0, 1) ** 2.2))[..., None]
    img = dif * (1 - fres * 0.6) + Rw * fres
    # soft specular sheen on swells facing the light
    spec = np.clip(slz * 3.0 - 0.12, 0, 1) ** 2 * 0.3
    img += spec[..., None] * Lh * 0.9
    full = cv2.resize(img, (w, (H2 - y0) * hs), interpolation=cv2.INTER_LINEAR)
    yy0 = y0 * hs
    hh_ = min(h - yy0, full.shape[0])
    out[yy0:yy0 + hh_] = full[:hh_]
    return out


# ================================================================ film-look per frame
def post(fc, st):
    tm = st["tm"]
    T = fc.T
    ca = 1.2 * seg(T, tm["lock"], tm["lock"] + 0.6, ease_out)
    ca += 3.0 * math.exp(-max(0.0, T - st["beats"].slam) * 8) * (T >= st["beats"].slam)
    fade = seg(T, tm["off"] + 0.28, tm["off"] + 0.5, ease_in_out)
    return dict(ca=ca, fade=fade)

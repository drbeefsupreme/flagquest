"""t08 — BLASPHEME (159.83-181.17).  Owner: T08Blaspheme.

Chick-tract melodrama played absolutely straight, on tract pages 7 and 8 (layout: scenes/t08_camp.py):
the camp in the rain, Crow's warning, the BLASPHEME lightning (the panel goes hard black-and-white),
Summer's horror, her fall to her knees in the mud, "Repeat after me." as the altar-call hymn begins, a real
page turn, and the lorem-ipsum prayer that ends with the rain stopping and a halftone sunbeam.

Compositing per page: grey world -> vx.ink.tract (page-locked dots; hard B&W threshold in the lightning
panels) -> spot(flags) -> print_layer(front: hands on poles) -> page stock/keylines/number + lettering
(knocked out of the flag cloth with comic.protect) -> POST.
"""
import math

import cairo
import numpy as np

from vx import W, H, Canvas, over
from vx import ink, comic, foley
from vx.ease import clamp

from scenes import t01_page as P
from scenes import t08_camp as C

POST = dict(C.POST)


# ============================================================ print styles
def _tone(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _matrix(cam):
    """page PU -> screen design px (exactly cam.apply)."""
    surf = cairo.ImageSurface(cairo.FORMAT_A8, 1, 1)
    c = cairo.Context(surf)
    cam.apply(c)
    return c.get_matrix()


def print_kw(cam):
    """kwargs for vx.ink print functions: the dot screen glued to the page through any camera (incl. rot)."""
    return dict(space="page", matrix=_matrix(cam), lpi=P.TRACT_LPI, angle=P.TRACT_ANGLE)


def _hard(fc, rgb, cam, neg=False, thr=0.5):
    """hard black & white: tone thresholded (antialiased) on the paper stock."""
    t = _tone(rgb)
    aa = 0.06
    cov = np.clip((thr - t) / aa + 0.5, 0.0, 1.0)
    if neg:
        cov = 1.0 - cov
    paper = ink.paper(fc, space="page", matrix=_matrix(cam))
    inkc = np.array(ink.INK, np.float32)
    return (paper * (1 - cov[..., None]) + inkc * cov[..., None]).astype(np.float32)


def _screen_rect_mask(fc, cam, r):
    c = Canvas(fc)
    c.ctx.save()
    cam.apply(c.ctx)
    c.ctx.rectangle(*r)
    c.ctx.set_source_rgb(1, 1, 1)
    c.ctx.fill()
    c.ctx.restore()
    return c.rgba()[..., 3]


def _bw_regions(fc, T, page, cam):
    """[(mask, neg)] panels printed hard black & white at T."""
    if page != 7:
        return []
    out = []
    d1 = C.drawers()[(7, "p1")]
    bw, flash, neg, _, _ = C.strike_state(d1.tp(T))
    if bw > 0 and P._visible(cam, C.R1):
        out.append((_screen_rect_mask(fc, cam, C.R1), neg))
    if P._visible(cam, C.R2A):
        out.append((_screen_rect_mask(fc, cam, C.R2A), False))
    return out


def _print(fc, rgb, cam, regions):
    img = ink.tract(fc, rgb, **print_kw(cam))
    for m, neg in regions:
        if m.max() <= 0:
            continue
        hard = _hard(fc, rgb, cam, neg)
        img = img * (1 - m[..., None]) + hard * m[..., None]
    return img


def _print_front(fc, rgba, cam, regions):
    a = np.clip(rgba[..., 3:4], 0, 1)
    rgb = np.where(a > 1e-4, rgba[..., :3] / np.maximum(a, 1e-4), 1.0).astype(np.float32)
    out = _print(fc, rgb, cam, regions)
    return np.concatenate([out * a, a], axis=-1).astype(np.float32)


# ============================================================ one page
def compose(fc, T, page, cam, want_world=False):
    """-> (print_rgb, flags_rgba, front_rgba|None, top_rgba[, world_rgb]) for page `page` seen through `cam`."""
    panels = C.page_panels(page)
    vis = [(r, d) for r, d in panels.items() if P._visible(cam, r, pad=4)]
    world = Canvas(fc, bg=(1, 1, 1))
    for r, d in vis:
        c = world.ctx
        c.save()
        cam.apply(c)
        c.rectangle(*r)
        c.clip()
        d.world(c, T, r)
        c.restore()
    rgb = world.rgb()
    regions = _bw_regions(fc, T, page, cam)
    img = _print(fc, rgb, cam, regions)

    fl = Canvas(fc)
    for r, d in vis:
        c = fl.ctx
        c.save()
        cam.apply(c)
        d.flags(c, T, r)
        c.restore()
    flags = fl.rgba()

    fr = Canvas(fc)
    for r, d in vis:
        c = fr.ctx
        c.save()
        cam.apply(c)
        c.rectangle(*r)
        c.clip()
        d.front(c, T, r)
        c.restore()
    lay = fr.rgba()
    front = _print_front(fc, lay, cam, regions) if lay[..., 3].max() > 0 else None

    top = Canvas(fc)
    top.ctx.save()
    cam.apply(top.ctx)
    # beyond the page edge: the darker stock of the rest of the booklet
    top.ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    top.ctx.rectangle(-4000, -4000, 9000, 9000)
    top.ctx.rectangle(0, 0, P.PAGE_W, P.PAGE_H)
    top.ctx.set_source_rgb(*(v * 0.62 for v in P.PAGE_RGB))
    top.ctx.fill()
    top.ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    rects = list(panels.keys()) if page == 7 else [C.R8] + [P.PANELS["p2"], P.PANELS["p3"]]
    P.draw_page_frame(top.ctx, page, rects if page == 7 else [C.R8])
    top.ctx.restore()
    for r, d in vis:
        top.ctx.save()
        cam.apply(top.ctx)
        d.letters(top.ctx, fc, T, r)
        top.ctx.restore()
    if want_world:
        return img, flags, front, top.rgba(), rgb
    return img, flags, front, top.rgba()


def layers(fc, T=None):
    """t07 hand-in: my page-7 frame at global T (default fc.T) as separate layers, so the engraving can
    'reprint' into it: dict(world=grey rgb (pre-print), print=printed rgb, flags=rgba (spot plate),
    front=rgba|None (printed hands over poles), top=rgba (page stock, keylines, lettering), cam=PageCam).
    Final frame = over(over(ink.spot(print, flags), front), comic.protect(top, flags))."""
    T = fc.T if T is None else T
    page, cam = C.camera(T)
    img, flags, front, top, rgb = compose(fc, T, page, cam, want_world=True)
    return dict(world=rgb, print=img, flags=flags, front=front, top=top, cam=cam)


def flatten(img, flags, front, top):
    img = ink.spot(img, flags)
    if front is not None:
        img = over(img, front)
    return over(img, comic.protect(top, flags))


def flatten_noflags(img, front, top, flags):
    """print + front + top WITHOUT the flag plate; flags returned with the front/top cut out of them
    (for the page turn, which lays the flag plates itself)."""
    if front is not None:
        img = over(img, front)
    top = comic.protect(top, flags)
    img = over(img, top)
    cut = (1.0 - (front[..., 3:4] if front is not None else 0.0)) * (1.0 - top[..., 3:4])
    return img, (flags * cut).astype(np.float32)


# ============================================================ page 8 underneath: t09 continues it
_T9_PAGE8_P2P3 = None     # t09 may install a drawer for page 8 p2/p3 when it imports us


def _page_curl(img_from, img_to, u, fc, flags_from, flags_to):
    try:
        from scenes.t02_fx import page_curl
    except ImportError:
        page_curl = None
    if page_curl is not None:
        return page_curl(img_from, img_to, u, fc, flags_from=flags_from, flags_to=flags_to)
    # fallback: the leaf sweeps right -> left (blank back of the leaf), page 8 revealed behind it
    xs = np.arange(fc.w, dtype=np.float32)[None, :] / fc.s
    xf = W * (1.0 - u)
    back_w = 260.0 * math.sin(math.pi * u)
    a = ink.spot(img_from, flags_from)
    b = ink.spot(img_to, flags_to)
    m_from = (xs < xf - back_w).astype(np.float32)[..., None]
    m_back = ((xs >= xf - back_w) & (xs < xf)).astype(np.float32)[..., None]
    paper = ink.paper(fc)
    out = a * m_from + paper * 0.97 * m_back + b * (1 - m_from - m_back)
    return out.astype(np.float32)


# ============================================================ scene contract
def setup(S):
    C.tracks()
    C.drawers()
    _export_sound(S)
    return {}


def render(fc, st):
    T = fc.T
    page, cam = C.camera(T)
    if C.TURN0 <= T < C.TURN1:
        u = (T - C.TURN0) / (C.TURN1 - C.TURN0)
        a = compose(fc, T, 7, cam)
        b = compose(fc, T, 8, C.camera8(T))
        ia, fa = flatten_noflags(a[0], a[2], a[3], a[1])
        ib, fb = flatten_noflags(b[0], b[2], b[3], b[1])
        return _page_curl(ia, ib, u, fc, fa, fb)
    return flatten(*compose(fc, T, page, cam))


# ============================================================ sound exports
def _pan_of(T, px, py):
    _, cam = C.camera(T)
    x, _ = cam.to_screen(px, py)
    return float(np.clip((x - W / 2) / (W / 2), -1, 1))


def _export_sound(S):
    tl = S.tl
    ev = [
        dict(t=C.T0, kind="ambience", strength=0.6, pan=0.0,
             desc="back in the present: the camp in the rain (rain on a tarp, drips, distant thunder)"),
        dict(t=160.05, kind="glide", strength=0.25, pan=0.0,
             desc="camera eases back off Crow: the tract panel and page are revealed"),
        dict(t=C.STRIKE, kind="lightning", strength=1.0, pan=-0.15,
             desc="BLASPHEME: lightning strikes behind Crow, the panel goes hard black-and-white (KRAKOOM!!)"),
        dict(t=C.STRIKE + 0.10, kind="thunder", strength=1.0, pan=0.0,
             desc="thunderclap rolling (starts right on the strike, rolls ~3 s)"),
        dict(t=C.STRIKE + 2 / 24, kind="lightning", strength=0.6, pan=-0.15, desc="re-strike flicker"),
        dict(t=C.STRIKE + 11 / 24, kind="lightning", strength=0.45, pan=-0.15, desc="late flicker"),
        dict(t=166.60, kind="glide", strength=0.7, pan=-0.3, desc="whip glide down to Summer's horror close-up"),
        dict(t=166.70, kind="shock", strength=0.6, pan=-0.3, desc="Summer's horror sting (sweat drops, tears)"),
        dict(t=167.30, kind="glide", strength=0.4, pan=0.2, desc="glide right to the kneel panel"),
        dict(t=167.50, kind="creak", strength=0.4, pan=-0.2, desc="Summer slides off her camp chair"),
        dict(t=C.KNEEL, kind="splash", strength=0.9, pan=-0.2, desc="her knees hit the mud: SPLUT! (mud splash)"),
        dict(t=168.72, kind="glide", strength=0.4, pan=0.0, desc="glide down to 'Repeat after me.' panel"),
        dict(t=C.PLANT, kind="thunk", strength=0.8, pan=0.25,
             desc="Crow drives the flag pole into the mud (THUNK), cloth snaps"),
        dict(t=C.TURN0, kind="paper", strength=0.8, pan=0.3, desc="page turn 7 -> 8 (leaf curls right to left)"),
        dict(t=C.AMEN, kind="rain_stop", strength=1.0, pan=-0.2,
             desc="Summer's Amen: the rain stops dead, drips only; the clouds part"),
        dict(t=C.AMEN + 0.05, kind="shimmer", strength=0.7, pan=-0.5,
             desc="halftone sunbeam falls on Summer (heavenly light)"),
        dict(t=C.AMEN + 0.6, kind="flag", strength=0.5, pan=0.0, desc="the flag lifts in the new breeze, glowing"),
    ]
    for lid in ("P24", "P25", "S06", "P26", "P27", "S07", "P28", "S08", "P29", "S09"):
        w = tl.words(lid)
        ev.append(dict(t=w[0]["s"] - 0.12, kind="balloon", strength=0.35, pan=0.0, desc=f"balloon pops ({lid})"))
    foley.export_events(S, "hits", ev)
    # hero flags: energy only while the panel is on screen and alive
    tr = C.tracks()
    wins = {"p1": (C.T0, 166.86, C.P1_POLE), "p2b": (167.30, 169.09, C.P2B_POLE), "p3": (168.72, 170.64, C.P3_POLE),
            "p8": (170.40, C.T1, C.P8_POLE)}
    from vx.flag import FlagTrack
    for name, (ta, tb, base) in wins.items():
        t = tr[name]
        n = S.n
        e = np.zeros(n, np.float32)
        sn = np.zeros(n, np.float32)
        fh = np.zeros(n, np.float32)
        ia, ib = int((ta - C.T0) * 24), min(n, int((tb - C.T0) * 24) + 1)
        e[ia:ib] = t.energy[ia:ib]
        sn[ia:ib] = t.snap[ia:ib]
        fh[ia:ib] = t.flap_hz[ia:ib]
        pan = np.array([_pan_of(C.T0 + k / 24, base[0], base[1] - 150) for k in range(n)], np.float32)
        sub = FlagTrack(t.verts[:n], t.poles[:n], e, sn, t.meta, fh)
        sub.export_foley(S, f"flag_{name}", pan=pan, f0=0, gain=0.8)

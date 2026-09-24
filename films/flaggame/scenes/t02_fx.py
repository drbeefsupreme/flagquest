"""t02 effects: page compositing with per-panel print styles, the page curl, the flashback ripple.

page_layers(fc, cam, panels, T, page_no, frame_rects=None, printer=None) -> (print_rgb, flags_rgba, over_rgba)
    like t01_page.render_page but split into layers (so a page can be curled/rippled); a drawer may ask for
    another print style inside its panel (drawer.alt(T) -> (style, kw, amount) + drawer.alt_mask(ctx, T, rect))
    and for front-print kwargs (drawer.front_kw(T)); printer(fc, world_rgb, cam) overrides the tract print.
page_curl(img_from, img_to, u, fc, flags_from=None, flags_to=None, over_from=None, over_to=None, masks=False,
          angle=0.18, radius=None, edge=None) -> rgb [, under]
    a page turned right->left (u 0..1): the leaf curls over a cylinder whose axis sweeps across the frame;
    its back is blank paper with a faint show-through; the page underneath gets a soft contact shadow.
    Flag layers are never shaded: flags_to shows where the page underneath is uncovered, flags_from rides the
    leaf (flat part and the front of the curl) geometrically. edge = screen x of the leaf's free edge.
curl_frame(fc, T, page1_layers, cam1, window)   t02's page 1 -> page 2 turn
flashback_frame(fc, T)   41.34 'Alchemy' -> 43.29: dive into Crow's lens, wavy ripple, halftone dots melt into
    engraving (ink.reprint), the engraving floods into solid ink and t03's op-art (t03_opart.handoff) opens through
    the ripple; the flag plate is never rippled or printed.
Owner: T02Wrong."""
import math

import cv2
import numpy as np

from vx import Canvas, W
from vx import ink
from vx.ease import clamp, smoothstep, seg

from . import t01_page as P

PAGE_RGB = np.array(P.PAGE_RGB, np.float32)


# ============================================================ layered page compositing
def _visible(cam, rect, pad=0.0):
    vx0, vy0, vx1, vy1 = cam.view_rect()
    x, y, w, h = rect
    return x + w >= vx0 - pad and x <= vx1 + pad and y + h >= vy0 - pad and y <= vy1 + pad


def page_layers(fc, cam, panels, T, page_no, frame_rects=None, printer=None):
    items = [(P.PANELS[k] if isinstance(k, str) else tuple(k), d) for k, d in panels.items()]
    vis = [(r, d) for r, d in items if _visible(cam, r)]
    world = Canvas(fc, bg=(1, 1, 1))
    for r, d in vis:
        if hasattr(d, "world"):
            c = world.ctx
            c.save()
            cam.apply(c)
            c.rectangle(*r)
            c.clip()
            d.world(c, T, r)
            c.restore()
    wrgb = world.rgb()
    img = printer(fc, wrgb, cam) if printer is not None else ink.tract(fc, wrgb, **cam.ink_kw())
    # alternative print styles inside some panels (icon: fine screen, anime: manga screentone)
    for r, d in vis:
        alt = d.alt(T) if hasattr(d, "alt") else None
        if not alt:
            continue
        style, kw, amount = alt
        if amount <= 0:
            continue
        m = Canvas(fc)
        c = m.ctx
        c.save()
        cam.apply(c)
        c.rectangle(*r)
        c.clip()
        d.alt_mask(c, T, r)
        c.restore()
        mask = m.rgba()[..., 3:4] * amount
        if mask.max() <= 0:
            continue
        kk = cam.ink_kw()
        kk.pop("lpi", None)
        kk.update(kw)
        alt_img = {"tract": ink.tract, "manga": ink.manga, "engrave": ink.engrave}[style](fc, wrgb, **kk)
        img = img * (1 - mask) + alt_img * mask
    fl = Canvas(fc)
    has_flags = False
    for r, d in vis:
        if hasattr(d, "flags"):
            c = fl.ctx
            c.save()
            cam.apply(c)
            c.rectangle(*P.inset(r, P.BORDER / 2))
            c.clip()
            d.flags(c, T, r)
            c.restore()
            has_flags = True
    flags = fl.rgba() if has_flags else np.zeros((fc.h, fc.w, 4), np.float32)
    over = np.zeros((fc.h, fc.w, 4), np.float32)
    if any(hasattr(d, "front") for _, d in vis):
        fr = Canvas(fc)
        for r, d in vis:
            if hasattr(d, "front"):
                c = fr.ctx
                c.save()
                cam.apply(c)
                c.rectangle(*r)
                c.clip()
                d.front(c, T, r)
                c.restore()
        lay = fr.rgba()
        if lay[..., 3].max() > 0:
            kw = cam.ink_kw()
            for r, d in vis:
                if hasattr(d, "front_kw"):
                    kw.update(d.front_kw(T))
            over = ink.print_layer(fc, lay, "tract", **kw)
    top = Canvas(fc)
    top.ctx.save()
    cam.apply(top.ctx)
    P.draw_page_frame(top.ctx, page_no, frame_rects or [r for r, _ in items])
    top.ctx.restore()
    for r, d in vis:
        if hasattr(d, "letters"):
            top.ctx.save()
            cam.apply(top.ctx)
            d.letters(top.ctx, fc, T, r)
            top.ctx.restore()
    let = top.rgba()
    if has_flags:
        from vx import comic
        let = comic.protect(let, flags)
    over = let + over * (1 - let[..., 3:4])
    return img, flags, over


def flatten(base, flags, over):
    img = ink.spot(base, flags)
    return over[..., :3] + img * (1 - over[..., 3:4])


# ============================================================ page 2 (placeholders until wired)
def page2_cam(T):
    from . import t02_panels as TP
    return TP.page2_cam(T)


def page2_layers(fc, T, cam, printer=None):
    from . import t02_panels as TP
    return page_layers(fc, cam, TP.PAGE2, T, 2, frame_rects=TP.PAGE2_FRAMES, printer=printer)


# ============================================================ page curl
def page_curl(img_from, img_to, u, fc, flags_from=None, flags_to=None, over_from=None, over_to=None, masks=False,
              angle=0.18, radius=None, edge=None):
    h, w = img_from.shape[:2]
    s = w / W
    R = (radius if radius is not None else 150.0) * s
    u = float(clamp(u))
    th = angle * (1 - 0.6 * u)                      # the axis straightens as the page falls
    n = np.array([math.cos(th), math.sin(th)], np.float32)      # unit normal toward the free edge (right)
    free = (edge if edge is not None else W * 1.04) * s        # free edge distance from the frame's left, on the normal
    # axis position along n: starts at the free edge, ends past the left side (everything turned)
    d_start = free
    d_end = -1.4 * R                                # the flipped part lies left of the axis: gone
    ax = d_start + (d_end - d_start) * u
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # measure positions along n from a pivot at the frame's vertical middle so the tilt reads as a corner lift
    proj = (xx * n[0] + (yy - h * 0.5) * n[1])
    d = proj - ax                                   # signed distance from the curl axis (+ toward free edge)
    dmax = free - ax                                # distance from axis to the leaf's free edge
    sd = np.clip(d / R, -1, 1)
    asn = np.arcsin(sd)
    # arc-length (source distance) of the back layer (on top) and of the front layer on the cylinder
    s_back_cyl = R * (math.pi - asn)
    s_front_cyl = R * asn
    s_back_flat = math.pi * R - d                   # d < 0: flipped flat part
    on_cyl = (d >= 0) & (d <= R)
    back_s = np.where(on_cyl, s_back_cyl, np.where(d < 0, s_back_flat, 1e9))
    has_back = back_s <= dmax
    front_s = np.where(on_cyl, s_front_cyl, np.where(d < 0, d, 1e9))
    has_front = (d <= R) & (front_s <= dmax)
    # source pixel for a layer at arc-length s: p + (s - d) n
    def remap(img, sarr):
        off = sarr - d
        mx = (xx + off * n[0]).astype(np.float32)
        my = (yy + off * n[1]).astype(np.float32)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    front = remap(img_from, np.where(has_front, front_s, d))
    back_src = remap(img_from, np.where(has_back, back_s, d))
    # the back of the leaf: paper with a faint mirrored show-through of the printed front
    lum = back_src[..., :1] * 0.2126 + back_src[..., 1:2] * 0.7152 + back_src[..., 2:3] * 0.0722
    paper = np.array(ink.PAPER, np.float32)
    back = paper * (0.93 + 0.07 * lum)
    # lighting: the front darkens as it curls away, the back of the curl catches a highlight
    cosf = np.sqrt(np.clip(1 - sd * sd, 0, 1))
    shade_front = np.where(on_cyl, 0.55 + 0.45 * cosf, 1.0)[..., None]
    shade_back = np.where(on_cyl, 0.78 + 0.28 * cosf, 0.97)[..., None]
    # contact shadow of the lifted leaf on the page underneath, and of the flipped part on the flat front
    shadow_under = np.where(d > R, 1 - 0.45 * np.exp(-(d - R) / (0.9 * R)), 1.0)[..., None]
    shadow_front = np.where(d < 0, 1 - 0.35 * np.exp(d / (0.5 * R)) * (back_s <= dmax), 1.0)[..., None]
    under_rgb = img_to * shadow_under
    if flags_to is not None:
        # flags underneath are never shaded: lay them after the shadow
        under_rgb = ink.spot(under_rgb, flags_to)
    if over_to is not None:
        under_rgb = over_to[..., :3] + under_rgb * (1 - over_to[..., 3:4])
    front_rgb = front * shade_front * shadow_front
    if flags_from is not None or over_from is not None:
        s_fr = np.where(has_front, front_s, d)
        if flags_from is not None:
            ff = remap(flags_from, s_fr)
            front_rgb = ink.spot(front_rgb, ff)
        if over_from is not None:
            of = remap(over_from, s_fr)
            front_rgb = of[..., :3] * shade_front * shadow_front + front_rgb * (1 - of[..., 3:4])
    back_rgb = back * shade_back
    hb = has_back[..., None].astype(np.float32)
    hf = (has_front & ~has_back)[..., None].astype(np.float32)
    out = back_rgb * hb + front_rgb * hf + under_rgb * (1 - hb - hf)
    # a thin dark edge where the curl turns over (reads as paper thickness)
    edge_line = (np.abs(d - R * 0.98) < 1.2 * s) & has_back
    out = np.where(edge_line[..., None], out * 0.7, out)
    if masks:
        return out.astype(np.float32), (1 - hb - hf)[..., 0]
    return out.astype(np.float32)


def curl_frame(fc, T, page1_layers, cam1, window):
    t0, t1 = window
    b1, f1, o1 = page1_layers(fc, min(T, 33.40), cam1)
    cam2 = page2_cam(T)
    b2, f2, o2 = page2_layers(fc, T, cam2)
    u = seg(T, t0, t1, lambda x: x * x * (3 - 2 * x))
    return page_curl(b1, b2, u, fc, flags_from=f1, flags_to=f2, over_from=o1, over_to=o2,
                     edge=cam1.to_screen(P.PAGE_W, 0)[0] + 10)




# ============================================================ flashback ripple
FLASH_T = 41.335
END_T = 43.2917


def _ripple_maps(fc, u, T):
    """classic wavy flashback displacement (screen space): horizontal sine bands + a gentle vertical wobble.
    amplitude swells and dies exactly at the hand-off (u = 1)."""
    h, w = fc.h, fc.w
    s = fc.s
    A = 30.0 * s * math.sin(math.pi * min(1.0, u)) ** 0.8
    lam = 140.0 * s
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ph = T * 7.0
    dx = A * np.sin(2 * math.pi * yy / lam - ph) + 0.35 * A * np.sin(2 * math.pi * yy / (lam * 2.7) + ph * 0.6)
    dy = 0.28 * A * np.sin(2 * math.pi * xx / (lam * 3.1) - ph * 0.8)
    return (xx + dx).astype(np.float32), (yy + dy).astype(np.float32), yy


def flashback_frame(fc, T):
    """41.34 'Alchemy' -> 43.29: ripple; halftone dots melt into engraving lines (ink.reprint) in wavy bands;
    the engraving thickens into solid ink; t03's op-art void opens through the ripple"""
    from . import t02_panels as TP
    u = clamp((T - FLASH_T) / (END_T - FLASH_T))
    cam = TP.page2_cam(T)
    mx, my, yy = _ripple_maps(fc, u, T)
    lam = 140.0 * fc.s
    band = np.sin(2 * math.pi * yy / (lam * 1.6) + T * 3.0).astype(np.float32)
    prog = np.clip(1.9 * u - 0.15 + 0.22 * band, 0, 1).astype(np.float32)
    dark = smoothstep(0.50, 0.93, u)

    def printer(fc_, wrgb, cam_):
        wv = wrgb * (1.0 - 0.97 * dark) if dark > 0 else wrgb
        kw = cam_.ink_kw()
        return ink.reprint(fc_, wv, "tract", "engrave", prog, space="page", origin=kw["origin"], zoom=kw["zoom"],
                           lpi=kw["lpi"], dot_angle=kw["angle"])

    base, flags, over = page2_layers(fc, T, cam, printer=printer)
    bl = Canvas(fc)
    TP.PAGE2["p1"].p06_screen(bl.ctx, fc, T)
    bal = bl.rgba()
    over = bal + over * (1 - bal[..., 3:4])
    img = over[..., :3] * (1 - dark) + base * (1 - over[..., 3:4] * (1 - dark))
    img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if flags[..., 3].max() > 0:
        # the spot plate is not printed: the flag does not melt with the print (it leaves the frame as we dive)
        img = ink.spot(img, flags)
    hand = smoothstep(0.74, 0.985, u)
    if hand > 0:
        from . import t03_opart
        op = t03_opart.handoff(fc, T).astype(np.float32)
        if u < 1.0:
            op = cv2.remap(op, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wmask = np.clip(hand * 1.6 - 0.3 + 0.3 * band, 0, 1)[..., None] if hand < 1 else 1.0
        img = img * (1 - wmask) + op * wmask
    return img.astype(np.float32)


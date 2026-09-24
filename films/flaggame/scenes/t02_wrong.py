"""t02 — EVERY WORD WAS WRONG (18.46-43.29). Owner: T02Wrong.

One panel after another turns into a different art tradition — Chick tract, Byzantine icon, anime —
while the lettering types itself on with the voices; then the wavy flashback ripple melts the halftone
dots into engraving lines and knits them into t03's op-art void.

Page 1 (ref page 23): p1 = t01's camp (scenes.t01_camp.PANEL1), p2 = Crow's witness, p3 = the icon.
Page 2 (ref page 24): top row = anime Summer | Crow leans in. Page curl 1->2 at 33.30-33.68.
"""
import math

from vx.ease import clamp, smoothstep, seg, ease_in_out, ease_in_out_sine
from vx import ink
from vx.foley import export_events

from . import t01_page as P
from . import t02_panels as TP
from . import t02_fx as FX

POST = dict(bloom=0.0, grain=0.0, vignette=0.10)

CURL0, CURL1 = 33.22, 33.70


# ============================================================ page 1 panel p1 (t01's camp)
class _P1Fallback:
    def world(self, ctx, T, rect):
        TP.white_rect(ctx, rect, 0.9)


def _p1_drawer():
    try:
        from . import t01_camp
        return t01_camp.PANEL1
    except Exception:
        return _P1Fallback()


# ============================================================ camera
def cam_at(T):
    """-> (page_no, PageCam)"""
    f1 = P.frame(P.PANELS["p1"])
    f2 = P.frame(P.PANELS["p2"])
    f3 = P.frame(P.PANELS["p3"])
    if T < 24.78:
        u = seg(T, TP.T0, 18.88, ease_in_out)
        cam = P.lerp_cam(f1, f2, u)
        # slow push while he talks, PUNCH on WRONG, then drift toward Summer for S03
        push = seg(T, 18.88, 22.05, ease_in_out_sine)
        z = cam.zoom * (1 + 0.05 * push)
        cx, cy = cam.cx + 20 * push, cam.cy
        w = T - TP.WRONG
        if w > -0.03:
            k = clamp((w + 0.03) / 0.07)
            settle = 1 - 0.55 * smoothstep(0.25, 1.3, w)
            z *= 1 + 0.16 * k * settle
            cx += 70 * k * settle
            cy += 8 * k * settle
            if w > 0:
                a = 9.0 * math.exp(-4.5 * w)
                cx += a * math.sin(w * 71)
                cy += a * math.cos(w * 57)
        s3 = seg(T, 23.05, 23.7, ease_in_out)
        cx += (-230 - (cx - f2.cx)) * s3 * 0.9
        z = z + (f2.zoom * 1.12 - z) * s3 * 0.8
        return 1, TP.clamp_cam(P.PageCam(cx, cy, z))
    if T < CURL0:
        base = P.PageCam(f2.cx + 20, f2.cy, f2.zoom * 1.05)
        u = seg(T, 24.78, 25.20, ease_in_out)
        cam = P.lerp_cam(base, f3, u)
        # slow reverent push toward the icon, easing back out for the footnote
        push = seg(T, 25.2, 30.2, ease_in_out_sine) * (1 - seg(T, 30.7, 31.3, ease_in_out))
        return 1, TP.clamp_cam(P.PageCam(cam.cx - 12 * push, cam.cy - 4 * push, cam.zoom * (1 + 0.05 * push)))
    return 2, FX.page2_cam(T)


# ============================================================ compositing
def page1_layers(fc, T, cam):
    """print (world->ink), flags RGBA, over RGBA (front print + page furniture + letters) for page 1"""
    panels = {"p1": _p1_drawer(), "p2": TP.PANELS["p2"], "p3": TP.PANELS["p3"]}
    return FX.page_layers(fc, cam, panels, T, 1)


def setup(S):
    # warm the flag sims (disk cached) so forked workers share them
    TP.PANELS["p2"].flag()
    TP.PANELS["p3"].flag()
    TP.PAGE2["p1"].flag()
    _export_hits(S)
    # hero flags -> Foley (f0 = scene-local frame of each track's frame 0)
    TP.PANELS["p2"].flag().export_foley(S, "flag_witness", f0=0, gain=0.6)
    TP.PANELS["p3"].flag().export_foley(S, "flag_icon", f0=0, gain=0.8)
    return {}


def render(fc, st):
    T = fc.T
    if T >= FX.FLASH_T:
        return FX.flashback_frame(fc, T)
    if CURL0 <= T < CURL1:
        return FX.curl_frame(fc, T, page1_layers, cam_at(CURL0 - 1e-3)[1], (CURL0, CURL1))
    page, cam = cam_at(T)
    if page == 1:
        base, flags, over_ = page1_layers(fc, T, cam)
    else:
        base, flags, over_ = FX.page2_layers(fc, T, cam)
    img = ink.spot(base, flags)
    return over_[..., :3] + img * (1 - over_[..., 3:4])


def _export_hits(S):
    tl = S.tl
    ev = [
        dict(t=TP.APPEAR, kind="pop", strength=0.7, pan=0.4, desc="Crow is suddenly just there in panel 2 (pop)"),
        dict(t=TP.T0 + 0.02, kind="glide", strength=0.4, pan=0.0, desc="camera glides panel 1 -> panel 2"),
        dict(t=TP.WRONG, kind="impact", strength=1.0, pan=0.25, desc="WRONG!!: radial burst, finger thrust, camera punch"),
        dict(t=TP.WRONG + 0.03, kind="shock", strength=0.8, pan=-0.45, desc="shock emanata on Summer"),
        dict(t=tl.word("P02", "wrong")[1] + 0.2, kind="whoosh", strength=0.4, pan=0.4,
             desc="footnote box *1ST VEXILLIANS 11s slides in"),
        dict(t=24.78, kind="glide", strength=0.4, pan=0.0, desc="camera glides down to the icon panel"),
        dict(t=TP.ICON, kind="icon", strength=0.9, pan=-0.3,
             desc="the panel is a Byzantine icon: silver-leaf gleam sweeps the halo (choir 'aah')"),
        dict(t=TP.ESSENCE, kind="radiance", strength=1.0, pan=-0.3,
             desc="ESSENCE: heavenly radiance swells from the flag/halo (swell ~0.5 s, sustains to 31.1)"),
        dict(t=TP.DISCL, kind="type", strength=0.3, pan=0.0, desc="** footnote types on at legal speed until 33.37"),
        dict(t=CURL0, kind="paper", strength=0.8, pan=0.3, desc="page turn (curl right->left) 33.22-33.70"),
        dict(t=TP.ANIME, kind="anime_pop", strength=1.0, pan=-0.2,
             desc="Summer SNAPS into anime style: sparkle 'kira', speed lines"),
        dict(t=TP.SNAP_BACK, kind="pop", strength=0.6, pan=-0.2, desc="snaps back to tract style"),
        dict(t=TP.SNAP_BACK + 0.08, kind="glide", strength=0.35, pan=0.2, desc="camera pans to Crow"),
        dict(t=TP.LEAN, kind="whoosh", strength=0.4, pan=0.3, desc="Crow leans in, conspiratorial"),
        dict(t=TP.GLASSES, kind="slide", strength=0.25, pan=0.3, desc="he slides his aviators down his nose"),
        dict(t=TP.FLASH, kind="ripple", strength=1.0, pan=0.0,
             desc="'Alchemy': wavy flashback ripple begins; halftone dots melt into engraving lines until 43.29"),
    ]
    export_events(S, "hits", ev)

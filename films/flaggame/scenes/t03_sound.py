"""t03 sound exports (owner T03Genesis): visual hits + the six hero flags' physics, in scene frames.
The flags simulate in flashback STORY time (they freeze with the flashback during S05), so their energy
is resampled onto the scene's frames: the flutter literally stops while Summer interrupts."""
import math

import numpy as np

from vx import foley
from vx.flag import FlagTrack
from scenes import t03_opart as OP
from scenes import t03_flags as FL
from scenes import t03_burn as BU
from scenes import t03_storm as SM
from scenes import t03_forecast as FC


def _pan(x):
    return float(np.clip((x - 960.0) / 960.0, -1, 1))


def export(S, tracks, tl):
    n = S.n
    # ---- hero flags: resample story-time physics onto scene frames; pan from the projected cloth
    for k, tr in enumerate(tracks):
        e = np.zeros(n, np.float32)
        sn = np.zeros(n, np.float32)
        fh = np.zeros(n, np.float32)
        pan = np.zeros(n, np.float32)
        for f in range(n):
            T = S.start + f / S.fps
            if T >= FC.T_BOOM or FL.sprout(k, T) <= 0:
                continue
            i = OP.tau_of(T) * 24.0
            if i < 0 or i > tr.n - 1:
                continue
            j = int(round(i))
            run = 1.0 - OP.pause_amount(T)
            e[f] = tr.energy[j] * run
            fh[f] = tr.flap_hz[j] * run
            sn[f] = tr.snap[j] * run
            if T < SM.T_IN:
                g = FL.flag_geom(tracks, k, BU.world_cam(T), T)
                if g is not None:
                    bb = g["cloth_bbox"]
                    pan[f] = _pan(0.5 * (bb[0] + bb[2]))
            else:
                _, _, _, _, px, py = SM._flag_state(k, T)
                z, cx, cy = SM.view(T)
                pan[f] = _pan((px - cx) * z + 960.0)
        ft = FlagTrack(np.zeros((1, 1, 1, 3), np.float32), np.zeros((1, 3), np.float32), e, sn, None, fh)
        ft.n = n
        ft.export_foley(S, f"flag{k}", pan=pan, f0=0)

    # ---- visual hits
    ev = []
    ev.append(dict(t=OP.T0, kind="whoosh", strength=0.6, pan=0.2,
                   desc="the black void breathes open into the op-art hexagon floor (genesis)"))
    cam = BU.world_cam(FL.SPROUT_T + 0.4)
    for k in range(6):
        x, _, _ = cam.project(FL.base_world(k))
        ev.append(dict(t=FL.SPROUT_T + 0.09 * k + 0.2, kind="thunk", strength=0.55, pan=_pan(x),
                       desc=f"flag pole {k + 1}/6 shoots up out of the op-art floor (furled)"))
    cam = BU.world_cam(FL.UNFURL_T + 0.3)
    for k in range(6):
        T = OP.T_of_tau(FL.unfurl_tau(k)) + 0.12
        x, _, _ = cam.project(FL.base_world(k) + np.array([0, 1.8, 0]))
        ev.append(dict(t=T, kind="snap", strength=0.75, pan=_pan(x),
                       desc=f"cloth {k + 1}/6 bursts open ('FLAGS') - plain yellow unfurl"))
    ev.append(dict(t=48.95, kind="glint", strength=0.5, pan=0.3, desc="'Yellow': the six flags glow"))
    ev.append(dict(t=OP.PAUSE_A[0] + 0.12, kind="freeze", strength=0.8, pan=0.0,
                   desc="Summer interrupts from the present: the whole flashback FREEZES (tape-stop feel, flags stop)"))
    ev.append(dict(t=tl.line("S05")["start"] - 0.05, kind="balloon", strength=0.5, pan=-0.9,
                   desc="off-panel balloon from the present, bottom-left (S05)"))
    ev.append(dict(t=tl.line("P08")["start"] - 0.05, kind="balloon", strength=0.5, pan=-0.9,
                   desc="off-panel balloon from the present, left (P08)"))
    w0, _ = tl.word("P08", "Second")
    ev.append(dict(t=w0, kind="paper", strength=0.35, pan=-0.9, desc="citation box 'II VEXILLIANS 5' slides in"))
    ev.append(dict(t=OP.PAUSE_B[0] + 0.2, kind="whoosh", strength=0.6, pan=0.0,
                   desc="'ABSENCE': the flashback runs again, flags resume"))
    ev.append(dict(t=53.4, kind="swirl", strength=0.5, pan=0.0, desc="the op-art floor starts to twist (rising swirl 53.4-56)"))
    ev.append(dict(t=54.8, kind="dissolve", strength=0.6, pan=0.0,
                   desc="ink floods the op-art: dissolve into the engraved Burn (54.8-56.4), crowd walla fades in"))
    ev.append(dict(t=56.0, kind="paper", strength=0.3, pan=-0.6, desc="P09 caption box appears"))
    ev.append(dict(t=57.339, kind="stench", strength=0.8, pan=0.0,
                   desc="STENCH: comic stink lines pop up over every head (wobbly, buzzing)"))
    ev.append(dict(t=58.25, kind="wind", strength=0.7, pan=0.0,
                   desc="the stench is sucked upward: a vortex of wind rises 58.25-59.34, flags whip"))
    for k in range(6):
        T = OP.T_of_tau(FL.rip_tau(k))
        ev.append(dict(t=T, kind="pull", strength=0.8, pan=0.0,
                       desc=f"pole {k + 1}/6 torn out of the mud by the wind, flies up out of frame"))
    ev.append(dict(t=SM.T_IN, kind="whoosh", strength=1.0, pan=0.0,
                   desc="HURRICANE: cut to the top-down storm chart, stink-line bands coalesce and spin"))
    ev.append(dict(t=SM.T_IN + 0.5, kind="paper", strength=0.4, pan=0.0, desc="isobars / chart ink drawing on"))
    ev.append(dict(t=60.95, kind="whoosh", strength=0.8, pan=0.0, desc="dive into the eye of the hurricane"))
    ev.append(dict(t=FC.T_BOOM, kind="radio", strength=0.8, pan=0.1,
                   desc="match cut: the eye becomes the boombox speaker cone; radio crackle; camera pulls back"))
    ev.append(dict(t=FC.T_RAIN, kind="rain", strength=1.0, pan=0.0,
                   desc="'RAIN.' the engraved downpour starts at once (splashes on mud and boombox)"))
    ev.append(dict(t=FC.T_RANGER - 0.05, kind="click", strength=0.6, pan=0.3,
                   desc="cut to the Ranger: radio push-to-talk squelch"))
    w0, _ = tl.word("G01", "dirt")
    ev.append(dict(t=w0, kind="step", strength=0.5, pan=0.3, desc="Ranger shifts, boots squelch in deep mud (surface: mud)"))
    ev.append(dict(t=67.2, kind="click", strength=0.4, pan=0.3, desc="radio squelch off"))
    foley.export_events(S, "hits", ev)

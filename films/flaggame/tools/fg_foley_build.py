"""THE FLAG GAME? (film 2) — Foley build (owner: Foley). Re-runnable; re-reads films/flaggame/cache/foley.

    source films/flaggame/env.sh && python films/flaggame/tools/fg_foley_build.py
    python films/flaggame/tools/fg_foley_check.py --spec

Outputs films/flaggame/audio/stems/{sfx,ambience}.wav (48 kHz stereo float32, exactly 199.0 s = 9,552,000 samples)
and films/flaggame/audio/foley/{CUES.md, placements.json, physics.json}.
THE LOOP: the film's last frame is its first. Everything that crosses 199.0 wraps to 0.0 (sample-exact), and the
close rain that ends the film is the same continuous signal that opens it.
"""
import os

os.environ.setdefault("VX_FILM", "flaggame")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

HERE = Path(__file__).resolve().parent
FROOT = HERE.parent
REPO = FROOT.parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(HERE))
import foley_build as B  # noqa: E402  (film-1 build: only its generic event models/limiter are reused)
import foley_flutter as FF  # noqa: E402
import foley_lib as L  # noqa: E402
import fg_foley_lib as G  # noqa: E402
from foley_lib import SR, R, Snd, midi_hz, ns  # noqa: E402

assert FF.FOLEY_CACHE == FROOT / "cache" / "foley", FF.FOLEY_CACHE
TL = json.loads((FROOT / "timeline.json").read_text())
DUR = float(TL["duration"])
NTOT = int(round(DUR * SR))
FREEZE, REVEAL = 118.92, 119.56
SMASH = 142.04                 # t06 -> t07 smash cut (with the score): everything incl. reverb dies here
HARD_CUTS = (FREEZE, SMASH)
CUES = {c["id"]: c["t"] for c in TL["cues"]}
LINES = {l["id"]: l for l in TL["lines"]}
OUTD = FROOT / "audio" / "foley"
STEMS = FROOT / "audio" / "stems"
D_SAFE = [62, 69, 64]          # D, A, E: safe over every section of the score (Composer)

VERB = {"t01": "meadow", "t02": "meadow", "t03": "plain", "t04": "storm", "t05": "storm", "t06": "dawn",
        "t07": "plain", "t08": "meadow", "t09": "meadow"}
DUCKS = {"wrong_burst": (0.4, 0.8), "flood_peak": (0.3, 0.9), "firmament_crack": (0.3, 0.6), "deluge": (0.25, 0.5),
         "blaspheme": (0.3, 0.6), "tract_lands": (0.6, 1.0), "devil_glitch": (0.5, 1.0),
         "shut_up": (0.4, 1.0), "found_one": (0.35, 1.0), "kneel": (0.35, 1.0)}


def scene_at(t):
    t = t % DUR
    for s in TL["scenes"]:
        if s["start"] <= t < s["end"]:
            return s["id"]
    return "t09"


def word(lid, w, k=0):
    ws = [x for x in LINES[lid]["words"] if x["w"].strip(",.!?").lower() == w.lower()]
    return ws[k]["s"] if len(ws) > k else LINES[lid]["start"]


class Mix:
    def __init__(self):
        self.bus = {"sfx": np.zeros((NTOT, 2), np.float32), "amb": np.zeros((NTOT, 2), np.float32)}
        self.sends = {}
        self.log = []
        self.ducks = [(CUES[c], a, b) for c, (a, b) in DUCKS.items()]

    @staticmethod
    def _acc(buf, a, seg):
        """add seg at absolute sample a, wrapping past the end (the loop)."""
        n = len(buf)
        a %= n
        while len(seg):
            k = min(len(seg), n - a)
            buf[a:a + k] += seg[:k]
            seg = seg[k:]
            a = 0

    def add(self, stem, snd, t, gain=1.0, pan=None, send=0.0, verb=None, name="", scene=None, cue=None,
            src="design", cut=None):
        x, sync = (snd.x, snd.sync) if isinstance(snd, Snd) else (snd, 0.0)
        if gain <= 0:
            return
        scene = scene or scene_at(t)
        x = np.asarray(x, np.float32)
        x = L.pan2(x, 0.0 if pan is None else pan) if x.ndim == 1 else (L.pan2(x, pan) if pan is not None else x)
        start = t - sync
        end_t = cut if cut is not None else 1e9
        for hc in HARD_CUTS:               # rain freeze 118.92 / smash cut 142.04: everything sounding stops dead
            if start < hc - 1e-6:
                end_t = min(end_t, hc)
                break
        i0 = int(round(start * SR))
        i1 = min(i0 + len(x), int(round(end_t * SR)))
        a = max(0, i0)
        if i1 <= a:
            return
        seg = x[a - i0:i1 - i0] * gain
        for td, ds, da in self.ducks:
            depth = ds if stem == "sfx" else da
            if depth >= 1 or start >= td - 0.005 or i1 <= int((td - 0.12) * SR) or a >= int((td + 0.4) * SR):
                continue
            g = np.interp(np.arange(a, i1) / SR - td, [-0.12, -0.02, 0.0, 0.4], [1, depth, depth, 1]).astype(np.float32)
            seg = seg * g[:, None]
        if i1 == int(round(end_t * SR)) and i1 - a > 96:
            seg = seg.copy()
            seg[-96:] *= np.linspace(1, 0, 96)[:, None]
        self._acc(self.bus[stem], a, seg)
        verb = verb or VERB.get(scene, "meadow")
        if send > 0:
            key = (stem, verb, end_t if end_t < 1e8 else None)
            if key not in self.sends:
                self.sends[key] = np.zeros((NTOT, 2), np.float32)
            self._acc(self.sends[key], a, seg * send)
        self.log.append(dict(t=round(float(t % DUR if t >= DUR else t), 4), start=round(float(start), 4), stem=stem,
                             name=name, scene=scene, cue=cue, src=src, gain=round(float(gain), 4),
                             peak=round(float(np.abs(seg).max()), 4)))

    def render_verbs(self):
        irs = {}
        for (stem, verb, end_t), buf in self.sends.items():
            if not np.any(buf):
                continue
            irs.setdefault(verb, L.make_ir(**B.IR_SPECS[verb]))
            wet = L.convolve(buf, irs[verb])
            if end_t is not None:          # tails of cut material die at the cut too
                wet[int(round(end_t * SR)):] = 0
            self._acc(self.bus[stem], 0, wet)
        self.sends = {}


# ================================================================================================
# RAIN (the backbone) — every layer is one continuous signal; the close rain wraps the loop point
# ================================================================================================
def rain(mx):
    r = R(1001)
    # A. close physical rain on the tract, mud and puddles: returns 196.6 -> 199.0 == 0.0 -> dives away at 6.4
    t0 = 196.4
    pts = [(196.4, 0.0), (197.3, 0.4), (198.3, 1.0), (DUR + 6.2, 1.0), (DUR + 6.9, 0.0)]
    x = G.rain_layer(r, t0, (DUR + 7.0) - t0, pts, dict(hiss=1.0, paper=1.0, mud=1.0, puddle=1.0), level=0.075)
    mx.add("amb", x, t0, 1.0, name="CLOSE RAIN on paper, mud & puddles (one signal across the loop point)",
           scene="t01")
    # B. comic camp rain on the shade (present day): 6.2 -> flashback ripple 41.34..43.29; t08 159.83 -> amen
    pts = [(6.1, 0.0), (6.9, 0.7), (18.5, 0.6), (41.3, 0.6), (43.0, 0.0)]
    x = G.rain_layer(r, 6.1, 43.3 - 6.1, pts, dict(hiss=0.8, shade=1.0, mud=0.5, puddle=0.5), level=0.045,
                     bright=0.8)
    mx.add("amb", x, 6.1, 1.0, name="camp rain on the shade (comic present)", scene="t01")
    am = CUES["amen"]
    pts = [(159.7, 0.0), (160.2, 0.8), (166.0, 0.9), (am, 0.85), (am + 0.9, 0.0)]
    x = G.rain_layer(r, 159.7, am + 1.0 - 159.7, pts, dict(hiss=0.9, shade=1.0, mud=0.6, puddle=0.6), level=0.05,
                     bright=0.8)
    mx.add("amb", x, 159.7, 1.0, name="camp rain on the shade (t08) — stops at 'Amen.'", scene="t08")
    mx.add("amb", G.shade_drips(r, 6.0, 7.0, 1.3), am + 0.2, 0.05, send=0.3, name="the shade drips on after the rain",
           scene="t08")
    # C. the engraved storm: 62.95 downpour -> 118.92 frozen dead (hard cut)
    rb = CUES["rain_begins"]
    pts = [(rb - 0.1, 0.0), (rb + 0.5, 1.0), (67.4, 0.9), (97.2, 0.9), (97.4, 0.15), (105.6, 0.15), (105.8, 0.35),
           (108.6, 0.4), (109.0, 0.9), (FREEZE, 1.0)]
    x = G.rain_layer(r, rb - 0.1, FREEZE - rb + 0.2, pts, dict(hiss=1.2, mud=1.0, puddle=0.7, shade=0.35), level=0.06,
                     bright=0.9)
    x = x.copy()
    a, b = ns(97.3 - (rb - 0.1)), ns(105.67 - (rb - 0.1))
    x[a:b] = L.lp(x[a:b], 700, 2) * 1.6          # the throne room: the storm heard through rock
    mx.add("amb", x, rb - 0.1, 1.0, name="THE STORM (engraved rain) — frozen dead at 118.92", scene="t03")
    # D. the deluge 137.12 -> fades into the montage by raise_1
    dl = CUES["deluge"]
    pts = [(dl, 0.0), (dl + 0.25, 1.0), (SMASH, 1.0)]
    x = G.rain_layer(r, dl, SMASH - dl + 0.1, pts, dict(hiss=1.5, mud=1.2, puddle=1.0), level=0.09, bright=1.1)
    mx.add("amb", x, dl, 1.0, name="the deluge (torrential rain)", scene="t06")


# ================================================================================================
# scene designs
# ================================================================================================
def t01_t02(mx, ev):
    r = R(11)
    # 2.6 the gust flips the tract
    T = CUES["tract_flip"]
    ev.claim_near("t01", T, 0.3, ["gust", "flip", "whoosh"])
    mx.add("sfx", L.whoosh(r, 1.4, 0.55, 200, 2400, 0.9, -0.7, 0.4, dark=0.35), T, 0.30, send=0.2,
           name="the gust", scene="t01", cue="tract_flip")
    mx.add("amb", L.wind(r, 3.0, 0.5, 0.3, 0.8, 0.1, 0.4) * L.env_pts(ns(3.0), [(0, 0), (0.5, 1), (3.0, 0)])[:, None],
           T - 0.5, 0.25, name="gust body", scene="t01")
    mx.add("sfx", Snd(G.wet_paper_slap(r).x * 0.5, 0.0), T + 0.02, 0.25, 0.1, name="the wet tract flips (flap)",
           scene="t01", cue="tract_flip")
    # 6.4 dive into page 1
    T = CUES["dive_in"]
    ev.claim_near("t01", T, 0.3, ["dive", "whoosh"])
    mx.add("sfx", L.whoosh(r, 1.3, 0.8, 150, 5000, 0.8, 0.0, 0.0, dark=0.2), T, 0.30, send=0.2,
           name="dive into page 1 (whoosh)", scene="t01", cue="dive_in")
    mx.add("sfx", L.swell(r, 0.6, 1000, 12000, 2.0), T, 0.08, name="the comic comes alive (air)", scene="t01")
    # 10.09 the flag-bearer passes behind them: mud steps panning
    T = CUES["flag_passes"]
    if not ev.find("t01", ["step"], all_=True, claim=False, t0=8.5, t1=13.5):
        for k in range(7):
            mx.add("sfx", G.mud_step(r, 0.7), T - 1.0 + 0.55 * k, 0.07, 0.7 - 0.23 * k, send=0.3,
                   name="the flag-bearer's steps in the mud", scene="t01")
    # t02 comic hits
    T = CUES["wrong_burst"]
    ev.claim_near("t02", T, 0.25, ["burst", "wrong", "shock"])
    mx.add("sfx", L.whoosh(r, 0.5, 0.45, 300, 6000, 1.2, 0.0, 0.0), T, 0.18, name="burst panel rushes in", scene="t02")
    mx.add("sfx", L.page_slam(r, 1.0), T, 0.40, send=0.15, name="WRONG!! radial burst (paper boom)", scene="t02",
           cue="wrong_burst")
    mx.add("sfx", L.sub_boom(r, 48, 30, 1.2, 0.2), T, 0.30, name="burst sub", scene="t02", cue="wrong_burst")
    emn = [(0.02 + 0.03 * k, L.ping(r, float(midi_hz([74, 76, 81, 86, 88, 93][k])), 0.12, 2), 1.0, (-1) ** k * 0.6) for k in range(6)]
    mx.add("sfx", L.norm(L.scatter(ns(0.5), emn), 0.8), T, 0.04, name="shock emanata zings", scene="t02")
    for cid, nm in (("icon", "the Preacher becomes an icon (gold glint)"), ("essence", "ESSENCE: heavenly radiance")):
        T = CUES[cid]
        ev.claim_near("t02", T, 0.3, ["icon", "essence", "radiance", "glow", "chime"])
        mx.add("sfx", L.swell(r, 0.9, 500, 11000, 2.0), T, 0.10, send=0.4, name=nm, scene="t02")
        gl = [(r.uniform(0, 1.2) ** 1.4, L.chime(r, float(midi_hz(r.choice(D_SAFE) + 24)), 2.5, t60=1.8, beat=1.2,
                                                   strike=0.1), r.uniform(0.3, 1.0), r.uniform(-0.8, 0.8)) for _ in range(14)]
        mx.add("sfx", L.norm(L.scatter(ns(4.0), gl), 0.9), T, 0.05, send=0.6, name=nm + " (glints)", scene="t02",
               cue=cid)
    T = CUES["disclaimer"]
    d = LINES["D01"]["end"] - T
    mx.add("sfx", G.typewriter(r, d, 16.0, 18.0), T, 0.12, 0.1, name="the ** footnote types at legal speed", scene="t02",
           cue="disclaimer")
    T = CUES["anime"]
    ev.claim_near("t02", T, 0.3, ["anime", "sparkle", "sweat"])
    mx.add("sfx", L.sparkle(r, 1.0, 16, [62, 69, 64], octave=(2, 3)), T, 0.06, 0.2, send=0.3,
           name="anime face: sparkles", scene="t02", cue="anime")
    mx.add("sfx", Snd(L.bubble(r, 2.0, 0.9, 1.0, 0.4), 0.0), T + 0.45, 0.10, 0.3, name="sweat-drop plip", scene="t02")
    T = CUES["flashback"]
    ev.claim_near("t02", T, 0.4, ["flashback", "ripple", "wavy"])
    mx.add("sfx", G.harp_gliss(r, 1.6), T, 0.10, send=0.5, name="flashback ripple: harp gliss up & down (D major)",
           scene="t02", cue="flashback")
    n = ns(2.2)
    rp = L.svf(L.pink(n, r), 1500 * 2 ** (1.5 * np.sin(L.TAU * 1.8 * L.tt(n))), 4, "bp") * np.sin(np.pi * L.tt(n) / 2.2)
    mx.add("sfx", L.decorrelate(L.norm(rp, 0.8), r), T, 0.08, name="wavy ripple (swept resonance)", scene="t02")


def t03(mx, ev):
    r = R(33)
    T = CUES["genesis"]
    n = ns(16.3)
    air = L.stereo(L.hp(L.pink(n, r), 3000), L.hp(L.pink(n, r), 3000)) * 0.3 + L.wind(r, 16.3, 0.15, 0.3, 0.1, 0.0, 0.5)
    mx.add("amb", B.shaped(air, T, [(T, 0), (T + 1.0, 1), (57.0, 1), (59.3, 0.5)]), T, 0.12,
           name="the op-art field: still air", scene="t03")
    T = CUES["stench"]
    ev.claim_near("t03", T, 0.4, ["stench", "stink"])
    mx.add("sfx", G.flies(r, 4.5), T, 0.05, name="flies find the hippies", scene="t03", cue="stench")
    mx.add("sfx", L.bubbles_cloud(r, 2.0, lambda u: 6 + 12 * u, (6.0, 14.0), 1.0, damp_mul=0.6), T, 0.12,
           name="stink rises (gross gloops)", scene="t03")
    n = ns(2.2)
    wob = L.svf(L.brown(n, r), 180 * (1 + 0.3 * np.sin(L.TAU * 3 * L.tt(n))), 3, "bp") * L.env_pts(n, [(0, 0), (2.2, 1)])
    mx.add("sfx", L.norm(wob, 0.8), T, 0.10, name="stink lines wobble up", scene="t03")
    # 59.34 the hurricane: a swirling gale that becomes the storm
    T = CUES["hurricane"]
    ev.claim_near("t03", T, 0.4, ["hurricane", "wind", "swirl"])
    d = 67.5 - T
    n = ns(d)
    ta = T + L.tt(n)
    sp = np.interp(ta, [T, T + 2.5, CUES["rain_begins"], 67.5], [0.2, 1.3, 1.1, 0.8])
    w = L.wind(r, d, whistle=0.7, bright=0.6, speed=sp * (1 + 0.2 * L.smooth_noise(n, r, 0.4, 2)))
    sw = np.sin(L.TAU * np.cumsum(np.interp(ta, [T, T + 3, 67.5], [0.2, 1.2, 0.3])) / SR)
    w = L.pan2(w, 0.7 * sw * np.interp(ta, [T, T + 1, 63, 67.5], [0, 1, 0.6, 0.2]))
    mx.add("amb", w, T, 0.16, name="the stench coalesces into a HURRICANE (swirling gale)", scene="t03", cue="hurricane")
    mx.add("sfx", L.sub_boom(r, 40, 26, 2.5, 0.6), T + 0.3, 0.2, name="hurricane pressure (sub)", scene="t03")
    # boombox forecast + ranger radio
    f1, g1 = LINES["F01"], LINES["G01"]
    mx.add("sfx", G.boombox(r, f1["end"] - f1["start"] + 0.35), f1["start"] - 0.25, 0.10, -0.3,
           name="boombox: play button, speaker hiss, stop", scene="t03")
    mx.add("sfx", G.radio_squelch(r), g1["start"] - 0.12, 0.13, 0.3, name="Ranger radio key-up (squelch)", scene="t03")
    mx.add("sfx", G.radio_squelch(r, 0.3), g1["end"] + 0.03, 0.12, 0.3, name="Ranger radio unkey (squelch tail)",
           scene="t03")
    T = CUES["rain_begins"]
    mx.add("sfx", L.swell(r, 0.6, 400, 8000, 1.5), T, 0.12, name="the sky opens", scene="t03", cue="rain_begins")


def t04(mx, ev):
    r = R(44)
    T = CUES["flagmaker"]
    mx.add("amb", G.candle_bed(r, 97.3 - T), T, 0.035, name="candlesticks guttering in the rain", scene="t04",
           cut=97.3)
    if not ev.find("t04", ["step"], all_=True, claim=False, t0=67.4, t1=76.0):
        for k in range(10):
            mx.add("sfx", G.mud_step(r, 0.8), T + 0.3 + 0.62 * k, 0.07, -0.5 + 0.1 * k, send=0.3,
                   name="the Flag Maker walks amongst them (mud)", scene="t04")
    # 76.98 'Shut up!' — the hooded mob turns: poncho rustle + stomps
    T = CUES["shut_up"]
    ev.claim_near("t04", T, 0.4, ["shut", "mob", "crowd"])
    for k in range(14):
        mx.add("sfx", L.cloth_rustle(r, 0.35, 1.1, 1.2), T + 0.1 + r.uniform(0, 0.35), 0.04, r.uniform(-0.8, 0.8),
               send=0.3, name="ponchos rustle (the Pharisees turn)", scene="t04")
    lurch = L.addm(L.thud(r, 70, 0.4, 1.2, 900), L.bp(L.white(ns(0.05), r), 500, 5000) * L.env_ad(ns(0.05), 0.0005, 0.02))
    mx.add("sfx", Snd(L.norm(lurch, 0.9), 0.0), T, 0.30, send=0.2, name="SHUT UP! the mob lurches (camera jolt)",
           scene="t04", cue="shut_up")
    for k in range(5):
        mx.add("sfx", G.mud_step(r, 1.2), T + (0.0 if k == 0 else r.uniform(0.12, 0.3)), 0.12 if k == 0 else 0.05, r.uniform(-0.6, 0.6), send=0.3,
               name="mob stomps", scene="t04", cue="shut_up" if k == 0 else None)
    # 87.15 -> 95.72 the FLAGS flood: printing press on the chant grid + typewriters + paper, rising
    T0, T1 = CUES["flags_flood"], CUES["flood_peak"]
    beat = 60.0 / 126.0
    mx.add("sfx", G.press_clatter(r, T1 - T0, 1.0 / (2 * beat), 2.4 / beat), T0, 0.12, send=0.3,
           name="printing press (cycles on the chant grid, accelerating)", scene="t04", cut=97.28)
    mx.add("sfx", G.typewriter(r, T1 - T0, 3.0, 90.0, bell=False), T0, 0.14, send=0.25,
           name="typewriters: 3 -> 90 keys/s", scene="t04", cut=97.28)
    en = np.clip(np.linspace(0.05, 1.0, int((T1 - T0) * 24)) ** 1.5, 0, 1)
    en = en * (1 + 0.3 * B.frame_turb(r, len(en)))
    mx.add("sfx", G.paper_from_energy(np.clip(en, 0, 1.2), 24, 0.0, 1.0, 5), T0, 0.12, name="paper flood (rising)",
           scene="t04", cut=97.28)
    ev.claim_near("t04", T1, 0.3, ["peak", "flood", "fill"])
    mx.add("sfx", L.page_slam(r, 1.4), T1, 0.40, send=0.2, name="FLAGS fill the frame (slam)", scene="t04",
           cue="flood_peak")
    mx.add("sfx", G.typewriter(r, 97.28 - T1 - 0.15, 90, 110, bell=False), T1 + 0.15, 0.14, name="flood at full clatter",
           scene="t04", cut=97.28)
    # 97.30 the Devil's throne room
    T = CUES["devil"]
    ev.claim_near("t04", T, 0.4, ["devil", "throne"])
    mx.add("amb", G.sulfur_bed(r, 105.67 - T), T, 0.06, send=0.4, verb="cavern",
           name="throne room: sulfur vents, furnace rumble, lava gloops", scene="t04", cut=105.67)
    mx.add("sfx", L.sub_boom(r, 44, 28, 2.0, 0.5), T, 0.35, name="the throne room (sub)", scene="t04", cue="devil")
    T = CUES["devil_glitch"]
    ev.claim_near("t04", T, 0.3, ["glitch", "possess"])
    mx.add("sfx", Snd(L.glitch(r, 0.45), 0.0), T, 0.25, name="possession glitch", scene="t04", cue="devil_glitch")
    n = ns(0.5)
    ts = np.sin(L.TAU * np.cumsum(np.interp(L.tt(n), [0, 0.5], [900, 40])) / SR) * L.env_ad(n, 0.002, 0.4)
    mx.add("sfx", L.f32(ts), T, 0.12, name="tape-stop (the Devil's voice overwritten)", scene="t04")
    mx.add("sfx", L.tv_static(r, 2.6) * L.env_pts(ns(2.6), [(0, 1), (2.6, 0.3)])[:, None], T, 0.05,
           name="possessed static", scene="t04", cut=105.67)


def t05(mx, ev):
    r = R(55)
    T = CUES["some_listened"]
    mx.add("sfx", Snd(G.paper_tick(r).x, 0.0), T, 0.15, name="white card (paper)", scene="t05", cue="some_listened")
    T = CUES["found_one"]
    ev.claim_near("t05", T, 0.4, ["pull", "found", "mud"])
    mx.add("sfx", G.mud_suck(r, 0.7), T, 0.30, 0.1, send=0.2, name="a Flag pulled out of the mud (suction -> pop)",
           scene="t05", cue="found_one")
    if not ev.find("t05", ["step"], all_=True, claim=False, t0=110.0, t1=117.0):
        for k in range(12):
            mx.add("sfx", G.mud_step(r, 0.9), 110.2 + 0.36 * k, 0.07, r.uniform(-0.7, 0.7), send=0.3,
                   name="MOVE IT! hurried steps in mud", scene="t05")
    if not ev.find("t05", ["wing"], all_=True, claim=False):
        for k in range(6):
            mx.add("sfx", G.wing_beat(r, 2.0), 113.3 + 0.62 * k, 0.16, 0.2 - 0.05 * k, send=0.4,
                   name="the Seraphlag's wing beats", scene="t05")
    T = CUES["flag_on_effigy"]
    ev.claim_near("t05", T, 0.3, ["effigy", "seat", "pole", "thunk"])
    mx.add("sfx", L.pole_thunk(r), T, 0.40, send=0.3, name="the pole seats atop the effigy (wood)", scene="t05",
           cue="flag_on_effigy")
    mx.add("sfx", Snd(L.impact(r, "wood", 1.6, 1.0, 0.3), 0.0), T + 0.01, 0.2, name="socket knock", scene="t05")
    # 118.92 THE RAIN FREEZE (the Mix hard-cuts everything); bullet time: ringing + a slow heartbeat
    T = CUES["rain_freeze"]
    mx.add("sfx", G.tinnitus(r, 2.2), T, 0.012, name="bullet time: high ringing", scene="t05", cue="rain_freeze")
    for k, tb in enumerate([T + 0.06, T + 1.3]):
        mx.add("sfx", G.heartbeat(r), tb, 0.22 * (1 - 0.3 * k), name="slow heartbeat", scene="t05")
    T = CUES["firmament_reveal"]
    mx.add("sfx", G.drop_cascade(r, 2.2, 800), T, 0.10, send=0.5, verb="sky",
           name="the frozen drops fall away (shimmering cascade)", scene="t05", cue="firmament_reveal")
    mx.add("sfx", G.crystal_resonance(r, 7.0), T + 0.1, 0.10, send=0.5, verb="sky",
           name="the firmament revealed: vast crystalline resonance (D)", scene="t05")
    mx.add("sfx", L.swell(r, 1.2, 200, 6000, 1.5).x[::-1].copy(), T, 0.06, name="air opens", scene="t05")


def t06(mx, ev):
    r = R(66)
    T = CUES["evening"]
    mx.add("amb", G.crickets(r, CUES["deluge"] - T), T, 0.012, name="night: crickets", scene="t06", cut=CUES["deluge"])
    mx.add("amb", L.wind(r, 20.0, 0.2, 0.4, 0.2, 0.05, 0.4), T, 0.10, name="night breeze", scene="t06")
    T = CUES["ignite"]
    ev.claim_near("t06", T, 0.3, ["ignite", "whoomph", "burn"])
    mx.add("sfx", L.whoosh(r, 1.2, 0.25, 90, 1500, 0.7, 0.0, 0.0, dark=0.6), T, 0.35, send=0.3,
           name="IGNITE: whoomph", scene="t06", cue="ignite")
    mx.add("sfx", L.sub_boom(r, 50, 32, 1.4, 0.3), T, 0.25, name="ignition (sub)", scene="t06", cue="ignite")
    T = CUES["flames_reach_flag"]
    ev.claim_near("t06", T, 0.3, ["flag", "ascend", "ember", "lift"])
    mx.add("sfx", L.whoosh(r, 2.2, 0.9, 200, 3000, 0.8, 0.0, 0.0, dark=0.2), T, 0.18, send=0.4,
           name="the Flag lifts free and ascends", scene="t06", cue="flames_reach_flag")
    n = ns(3.5)
    emb = L.hp(L.poisson_clicks(r, n, 60 * np.exp(-L.tt(n) / 1.5)), 2500)
    mx.add("sfx", L.decorrelate(L.norm(emb, 0.8), r), T, 0.06, name="rising embers (spark crackle)", scene="t06")
    T = CUES["firmament_crack"]
    ev.claim_near("t06", T, 0.3, ["crack", "firmament"])
    mx.add("sfx", L.stone_crack(r, 0.12, 1.0, "crystal", 1.2), T, 0.55, send=0.5, verb="sky",
           name="THE FIRMAMENT CRACKS (colossal crystal)", scene="t06", cue="firmament_crack")
    mx.add("sfx", Snd(L.shock(r, 1.5), 0.0), T, 0.4, name="crack shock", scene="t06", cue="firmament_crack")
    mx.add("sfx", L.sub_boom(r, 45, 24, 3.0, 0.5), T, 0.55, name="crack sub", scene="t06", cue="firmament_crack")
    mx.add("sfx", G.crystal_resonance(r, 5.0, (38, 45, 50, 57)), T, 0.08, send=0.4, verb="sky",
           name="the dome rings as it splits", scene="t06")
    n = ns(1.5)
    craz = L.stone_crack(r, 1.5, 1.0, "glass", 1.6, final_bang=False).x
    mx.add("sfx", L.decorrelate(craz, r), T + 0.1, 0.15, send=0.5, name="cracks race across the sky", scene="t06")
    T = CUES["deluge"]
    ev.claim_near("t06", T, 0.3, ["deluge", "water", "pour", "splash", "steam"])
    mx.add("sfx", L.ocean_roar(r, 6.0, 1.0, 0.4) * L.env_pts(ns(6.0), [(0, 0), (0.15, 1), (5, 0.7), (6, 0)])[:, None],
           T, 0.45, name="the waters above pour down (roar)", scene="t06", cue="deluge")
    mx.add("sfx", Snd(L.shock(r, 1.5), 0.0), T, 0.7, name="the waters strike (shock)", scene="t06", cue="deluge")
    mx.add("sfx", G.steam_explosion(r, 3.0), T, 0.40, send=0.3, name="steam explosion: the fire drowned",
           scene="t06", cue="deluge")
    mx.add("sfx", L.sub_boom(r, 42, 22, 4.0, 0.6), T, 0.65, name="deluge sub (<60 Hz)", scene="t06", cue="deluge")
    mx.add("sfx", B.splash(r, 3.0), T + 0.1, 0.45, send=0.3, name="splashes", scene="t06")


def raise_sound(r, s=1.0):
    """one Flag raised: halyard creak then the pole seated (thunk) and the cloth snaps open. sync = thunk."""
    cr = G.halyard_creak(r, 0.5).x
    th = L.pole_thunk(r).x
    sn = FF.snap_crack(r, 0.7)
    x = np.zeros(ns(0.55) + len(th), np.float32)
    x[: len(cr)] += cr * 0.5
    i = ns(0.5)
    x[i:i + len(th)] += th
    x[i + ns(0.08):i + ns(0.08) + len(sn)] += sn * 0.5
    return Snd(L.norm(x, 0.9) * s, 0.5)


def t07(mx, ev):
    r = R(77)
    T = CUES["raise_1"]
    rz = ev.find("t07", ["raise"], all_=True, t0=142.0, t1=159.9)
    if rz:
        pts = [(e["t"], e["pan"], e["strength"]) for e in rz]
    else:    # the flurry: 100 raisings accelerating from raise_1 to raise_100
        t100 = CUES["raise_100"]
        pts = [(T + (t100 - T) * (k / 99) ** 0.5, r.uniform(-0.7, 0.7), 1.0 - 0.5 * k / 99)
               for k in range(100)]
        pts[-1] = (t100, 0.0, 1.0)
    for k, (t, p, s) in enumerate(pts):
        g = 0.30 if k in (0, len(pts) - 1) else 0.30 / (1 + 0.08 * k)
        mx.add("sfx", raise_sound(r, 1.0), t, g * (0.5 + 0.5 * s), p, send=0.3, name="a Flag RAISED (creak, thunk, snap)",
               scene="t07", cue="raise_1" if k == 0 else ("raise_100" if abs(t - CUES["raise_100"]) < 0.02 else None))
    # the Spirit walks on through the seasons
    T = CUES["spirit"]
    n = ns(159.9 - T)
    ta = T + L.tt(n)
    w = L.wind(r, 159.9 - T, whistle=0.3, bright=0.5,
               speed=np.interp(ta, [T, 152, 155.4, 157.5, 159.9], [0.3, 0.45, 0.25, 0.4, 0.3]) *
               (1 + 0.3 * L.smooth_noise(n, r, 0.3, 2)))
    mx.add("amb", w, T, 0.16, name="the years: wind through the seasons", scene="t07")
    # after the smash cut: ~0.3 s of near-silence (one dry drip), then the plain's wind creeps back under "HOWEVER!"
    mx.add("sfx", L.drip(r, 1.1), SMASH + 0.12, 0.05, 0.2, name="a single dry drip in the silence", scene="t07")
    wm = L.wind(r, T - SMASH - 0.3, 0.25, 0.4, 0.2, 0.1, 0.5)
    wm = B.shaped(wm, SMASH + 0.3, [(SMASH + 0.3, 0), (SMASH + 1.8, 1), (T, 1)])
    mx.add("amb", wm, SMASH + 0.3, 0.10, name="after the deluge: wind over the plain (fades in)", scene="t07")
    if not ev.find("t07", ["step"], all_=True, claim=False, t0=T, t1=159.0):
        t = T + 0.2
        while t < CUES["unmask"] - 0.2:
            desc = "leaves" if t < 152.0 else ("snow" if t < 155.4 else "mud")
            mx.add("sfx", G.step_for(r, desc, "mud", 0.8), t, 0.07, 0.1 * np.sin(t * 3), send=0.3,
                   name=f"the lone walker ({desc})", scene="t07")
            t += 0.62
    T = CUES["unmask"]
    mx.add("sfx", L.cloth_rustle(r, 0.4, 0.8, 1.2), T, 0.12, name="he lowers the mask", scene="t07", cue="unmask")


def t08_t09(mx, ev):
    r = R(88)
    T = CUES["blaspheme"]
    ev.claim_near("t08", T, 0.3, ["lightning", "thunder", "blaspheme"])
    mx.add("sfx", L.zap(r, 0.3, 7000, 1500), T, 0.10, name="lightning", scene="t08")
    mx.add("sfx", Snd(L.thunder(r, 1.0).x, 0.0), T, 0.45, send=0.2, name="BLASPHEME: thunderclap", scene="t08",
           cue="blaspheme")
    mx.add("sfx", L.sub_boom(r, 46, 24, 3.0, 0.5), T, 0.45, name="thunder sub", scene="t08", cue="blaspheme")
    T = CUES["kneel"]
    ev.claim_near("t08", T, 0.3, ["kneel", "knee"])
    mx.add("sfx", G.mud_step(r, 1.3), T, 0.40, -0.1, name="knee into mud", scene="t08", cue="kneel")
    mx.add("sfx", G.mud_step(r, 1.1), T + 0.14, 0.2, 0.1, name="second knee", scene="t08")
    am = CUES["amen"]
    t = am + 0.6
    while t < 196.0:
        mx.add("amb", L.bird(r, None, r.uniform(25, 90)), t, r.uniform(0.02, 0.05), r.uniform(-0.9, 0.9), send=0.4,
               name="birds after the rain", scene=scene_at(t))
        t += r.exponential(1.3 if t < 185.4 else 0.9)
    mx.add("sfx", L.swell(r, 1.0, 400, 7000, 1.5), am, 0.05, name="the sunbeam (air)", scene="t08", cue="amen")
    T = CUES["pass_it_on"]
    ev.claim_near("t09", T, 0.3, ["tract", "hand", "pass"])
    mx.add("sfx", L.paper_slide(r, 0.6, 1.0), T, 0.18, name="he hands her the tract (paper)", scene="t09",
           cue="pass_it_on")
    # 185.4 -> 191.6 the Print Gallery Droste: unpitched spectral Shepard rise (Composer: pitched Shepard-Risset)
    T0, T1 = CUES["droste_start"], CUES["droste_end"]
    mx.add("sfx", L.whoosh(r, 1.2, 0.5, 200, 5000, 0.8, 0.0, 0.0), T0, 0.2, name="into the tract", scene="t09",
           cue="droste_start")
    sh = L.decorrelate(L.shepard_rise(r, T1 - T0, 7, 90, 0.5, 0.3), r)
    mx.add("sfx", B.shaped(sh, T0, [(T0, 0), (T0 + 0.8, 1), (T1 - 0.5, 1), (T1, 0)]), T0, 0.07,
           name="Droste spiral: endless unpitched Shepard noise", scene="t09")
    mx.add("sfx", Snd(G.paper_tick(r).x, 0.0), T1, 0.2, name="the spiral resolves (page)", scene="t09", cue="droste_end")
    # next day: drying camp
    n = ns(196.6 - T1)
    br = L.wind(r, 196.6 - T1, 0.18, 0.4, 0.2, 0.0, 0.7)
    mx.add("amb", B.shaped(br, T1, [(T1, 0), (T1 + 0.6, 1), (196.2, 1), (196.6, 0)]), T1, 0.12,
           name="next day: breeze over the drying camp", scene="t09")
    mx.add("amb", G.shade_drips(r, 196.4 - T1, 1.5, 3.0), T1, 0.035, send=0.3, name="last drips off the shades",
           scene="t09")
    T = CUES["tract_toss"]
    ev.claim_near("t09", T, 0.3, ["toss"])
    mx.add("sfx", L.whoosh(r, 1.1, 0.4, 500, 4000, 1.0, 0.3, -0.3), T, 0.15, name="the tract tossed (flutters)",
           scene="t09", cue="tract_toss")
    en = np.concatenate([np.linspace(0.2, 0.8, 6), 0.7 + 0.2 * B.frame_turb(r, 20), np.linspace(0.6, 0.05, 1)])
    mx.add("sfx", G.paper_from_energy(np.clip(en, 0, 1), 24, 0.0, 1.0, 9), T + 0.05, 0.12, name="pages flutter in flight",
           scene="t09")
    T = CUES["tract_lands"]
    ev.claim_near("t09", T, 0.3, ["land", "slap", "mud"])
    mx.add("sfx", G.wet_paper_slap(r), T, 0.60, name="the tract slaps face-down into the mud", scene="t09",
           cue="tract_lands")


def comic_ticks(mx, ev):
    """a faint paper 'tick' each time a speech balloon pops onto the page (comic scenes)."""
    r = R(99)
    bal = ev.find_any(["balloon"])
    if bal:
        return
    for l in TL["lines"]:
        t = l["start"] - 0.03
        if (6.4 <= t < 41.34) or (159.83 <= t < 185.4) or (191.6 <= t < DUR):
            mx.add("sfx", G.paper_tick(r), t, 0.05, 0.0, name="balloon pops (paper tick)", scene=scene_at(t))


# ================================================================================================
# scene-exported events (film-2 specific models first, then the shared generic models)
# ================================================================================================
def place_events(mx, ev):
    r = R(777)
    for e in ev.unused():
        ev.used.add(e["_i"])
        if not (0.0 <= e["t"] < DUR):
            continue
        k = (e["kind"] + " " + e["desc"]).lower()
        st = float(np.clip(e.get("strength", 1.0), 0, 1))
        g = 0.35 + 0.65 * st
        name, src = e["kind"], "event:" + e["file"]
        if "balloon" in e["kind"]:
            mx.add("sfx", G.paper_tick(r), e["t"], 0.05 * g, e["pan"], name="balloon pops (paper tick)",
                   scene=e["scene"], src=src)
        elif re.search(r"\b(glide|panel)", k):
            mx.add("sfx", G.panel_glide(r, 0.6), e["t"], 0.06 * g, e["pan"], name="panel glide", scene=e["scene"], src=src)
        elif re.search(r"\b(step|foot|walk|stomp|run)", e["kind"]):
            mx.add("sfx", G.step_for(r, e["desc"], "mud", 0.6 + 0.4 * st), e["t"], 0.09 * g, e["pan"], send=0.3,
                   name="step (" + e["desc"][:30] + ")", scene=e["scene"], src=src)
        elif e["kind"] == "raise" or "raising" in k:
            mx.add("sfx", raise_sound(r), e["t"], 0.2 * g, e["pan"], send=0.3, name="flag raised", scene=e["scene"],
                   src=src)
        elif re.search(r"\b(pull|suck)", k):
            mx.add("sfx", G.mud_suck(r, 0.6), e["t"], 0.25 * g, e["pan"], send=0.2, name="pulled from mud",
                   scene=e["scene"], src=src)
        elif "wing" in k:
            mx.add("sfx", G.wing_beat(r, 2.0), e["t"], 0.16 * g, e["pan"], send=0.4, name="wing beat", scene=e["scene"],
                   src=src)
        elif e["kind"] == "paper" or re.search(r"\b(page|tract|paper)", e["kind"]):
            mx.add("sfx", L.paper_slide(r, 0.4, 1.0), e["t"], 0.12 * g, e["pan"], name="paper", scene=e["scene"], src=src)
        elif "radio" in k or "squelch" in k:
            mx.add("sfx", G.radio_squelch(r), e["t"], 0.12 * g, e["pan"], name="radio squelch", scene=e["scene"], src=src)
        else:
            out = B.event_sound(e, r)
            if out is None:
                continue
            snd, stem, gb, send = out
            mx.add(stem, snd, e["t"], gb * g, e["pan"], send, name=B.classify(e), scene=e["scene"], src=src)


# ================================================================================================
# physics: flag flutter, paper sim, fire
# ================================================================================================
FALLBACK_FLAGS = {  # scene: (t0, t1, [(t, energy)], pan0, pan1, gain, field)
    "t01": (9.2, 12.8, [(9.2, 0.0), (9.6, 0.3), (12.3, 0.3), (12.8, 0.0)], 0.7, -0.7, 0.35, False),
    "t03": (43.3, 57.3, [(43.3, 0.0), (44.0, 0.12), (57.0, 0.12), (57.3, 0.0)], 0.0, 0.0, 0.3, True),
    "t05": (110.0, 118.9, [(110.0, 0.0), (110.4, 0.55), (116.8, 0.5), (118.9, 0.35)], -0.3, 0.3, 0.45, True),
    "t07": (143.9, 149.5, [(143.9, 0.0), (144.1, 0.4), (149.3, 0.4), (149.5, 0.0)], 0.0, 0.0, 0.3, True),
    "t09": (194.2, 196.3, [(194.2, 0.0), (194.5, 0.3), (196.0, 0.3), (196.3, 0.0)], 0.6, -0.6, 0.35, False),
}
FALLBACK_PAPER = [(0.0, 0.03), (2.5, 0.04), (2.62, 0.95), (3.2, 0.25), (4.8, 0.2), (4.95, 0.8), (5.5, 0.6), (6.0, 0.7),
                  (6.3, 0.3), (6.5, 0.0)]
FALLBACK_FIRE = [(125.3, 0.0), (125.5, 0.6), (127.0, 0.75), (129.7, 1.0), (136.9, 0.95), (137.15, 0.5), (138.6, 0.05),
                 (140.0, 0.0)]


def corr(en, x, fps):
    rms = FF.frame_rms(x, fps, len(en))
    act = np.nonzero(en > 0.02)[0]
    return float(np.corrcoef(en[act], rms[act])[0, 1]) if len(act) > 5 and rms[act].std() > 0 else float("nan")


def physics(mx, report):
    tracks = FF.load_tracks()
    kinds = {}
    for tr in tracks:
        kinds.setdefault((tr["scene"], tr["kind"]), []).append(tr)
        fps, t0 = tr["fps"], tr["f0"] / tr["fps"]
        seed = zlib.crc32(tr["file"].encode()) % 2 ** 31
        if tr["kind"] == "paper":
            x = G.paper_from_energy(tr["energy"], fps, tr["pan"], tr["gain"], seed)
            base, what = 0.25, "PHYSICS paper"
        elif tr["kind"] == "fire":
            x = G.fire_from_energy(tr["energy"], fps, tr["pan"], tr["gain"], seed)
            base, what = 0.24, "PHYSICS fire"
        elif tr["kind"] == "crowd":
            continue
        else:
            x, t0, info = FF.track_audio(tr)
            base, what = (0.3 if info.get("field") else 0.5), "PHYSICS flutter"
        c = corr(tr["energy"], x, fps)
        pk = float(np.abs(x).max()) * base
        base *= min(1.0, 0.3 / max(pk, 1e-9))
        stem = "amb" if tr["kind"] == "fire" else "sfx"
        mx.add(stem, x, t0, base, send=0.2, name=f"{what} <- {tr['file']} (r={c:.3f})", scene=tr["scene"],
               src="physics:" + tr["file"])
        report.append(dict(file=tr["file"], scene=tr["scene"], kind=tr["kind"], frames=len(tr["energy"]), t0=t0,
                           energy_max=float(tr["energy"].max()), corr=c))
    r = R(5)
    for sc, (t0, t1, pts, p0, p1, gain, field) in FALLBACK_FLAGS.items():
        if any(k[0] == sc and k[1] not in ("paper", "fire", "crowd") for k in kinds):
            continue
        nfr = int((t1 - t0) * 24)
        en = np.clip(np.interp(t0 + np.arange(nfr) / 24, *zip(*pts)) * (1 + 0.3 * B.frame_turb(r, nfr)), 0, None)
        pan = np.linspace(p0, p1, nfr)
        x = FF.flutter_from_energy(en, 24, pan, gain, [], seed=zlib.crc32(sc.encode()), t0=t0, field=field)
        mx.add("sfx", x, t0, 0.5 if not field else 0.3, send=0.2, name=f"flutter (fallback energy, awaiting {sc} track)",
               scene=sc, src="fallback-physics")
        report.append(dict(file=f"(fallback {sc} flag)", scene=sc, kind="flag", frames=nfr, t0=t0,
                           energy_max=float(en.max()), corr=corr(en, x, 24)))
    for sc, kind, pts, fn, stem, g in (("t01", "paper", FALLBACK_PAPER, G.paper_from_energy, "sfx", 0.25),
                                       ("t06", "fire", FALLBACK_FIRE, G.fire_from_energy, "amb", 0.24)):
        if (sc, kind) in kinds:
            continue
        t0 = pts[0][0]
        nfr = int((pts[-1][0] - t0) * 24)
        en = np.clip(np.interp(t0 + np.arange(nfr) / 24, *zip(*pts)) * (1 + 0.15 * B.frame_turb(r, nfr, 1.0)), 0, None)
        x = fn(en, 24, 0.0, 1.0, 3)
        mx.add(stem, x, t0, g, send=0.2, name=f"{kind} (fallback energy, awaiting {sc} {kind} track)", scene=sc,
               src="fallback-physics")
        report.append(dict(file=f"(fallback {sc} {kind})", scene=sc, kind=kind, frames=nfr, t0=t0,
                           energy_max=float(en.max()), corr=corr(en, x, 24)))


# ================================================================================================
def finalize(mx):
    lims = {}
    for stem in ("sfx", "amb"):
        b = mx.bus[stem]
        np.nan_to_num(b, copy=False)
        # circular DC/infrasonic high-pass (filter the loop twice around so the seam carries no filter transient)
        ext = np.concatenate([b[-ns(2.0):], b])
        b[:] = L.hp(ext, 18.0, 2)[ns(2.0):]
        mx.bus[stem], lims[stem] = B.limiter(b)
    return lims


def write_cues(mx, report, lims):
    L_ = ["# Foley cue sheet — THE FLAG GAME? (film 2; generated by films/flaggame/tools/fg_foley_build.py)", "",
          "Stems `films/flaggame/audio/stems/sfx.wav` + `ambience.wav`: 48 kHz stereo float32, exactly 199.0 s. "
          "Rebuild: `source films/flaggame/env.sh && python films/flaggame/tools/fg_foley_build.py && "
          "python films/flaggame/tools/fg_foley_check.py --spec` (re-reads films/flaggame/cache/foley).", "",
          "**The loop:** the mixer is circular: anything crossing 199.0 continues at 0.0, and the close rain that "
          "opens the film (tract in the mud) is one continuous signal generated from 196.4 through 199.0 to 7.0, "
          "so 199.0 → 0.0 is sample-continuous (see Verification).", "",
          "**The rain freeze (118.92):** every placement that started before it is hard-cut at 118.92, reverb tails "
          "included; only a high ringing and a slow heartbeat remain until the drops fall away at 119.56.", "",
          "**Smash cut (142.04, with the score):** the deluge and everything else, reverb included, dies at 142.04; "
          "~0.3 s of near-silence (one dry drip) under HOWEVER! (142.33), then the plain's wind fades back in.", "",
          "**Rain = physics per surface:** each drop is an impulse (Poisson rate ∝ storm intensity) exciting its "
          "surface — paper (short bright plate modes), mud (dull splat), camp shade (membrane drum modes), puddles "
          "(Minnaert bubbles) — over a broadband wash.", "",
          "**With Composer:** sub (<60 Hz) on impacts is Foley's; the score's hits are tonal ≥60 Hz. Tonal Foley uses "
          "D/A/E only (safe in every section); the Droste noise is unpitched under Composer's pitched Shepard–Risset; "
          "the FLAGS-flood printing press cycles on the chant's 126 BPM grid.", "",
          f"Limiter min gain: sfx {lims['sfx']:.3f}, ambience {lims['amb']:.3f}.", "",
          "## Physics-derived tracks", "", "| track | kind | scene | frames | start | max energy | r(energy, RMS) |",
          "|---|---|---|---:|---:|---:|---:|"]
    for p in report:
        L_.append(f"| {p['file']} | {p['kind']} | {p['scene']} | {p['frames']} | {p['t0']:.2f} | {p['energy_max']:.2f} | "
                  f"{p['corr']:.3f} |")
    L_ += ["", "## Placements by scene", ""]
    for s in TL["scenes"]:
        evs = sorted([e for e in mx.log if e["scene"] == s["id"]], key=lambda e: e["t"])
        groups = {}
        for e in evs:
            groups.setdefault((e["name"], e["stem"], e["src"]), []).append(e)
        L_ += [f"### {s['id']} — {s['title']} ({s['start']:.2f}–{s['end']:.2f})", "",
               "| t (s) | n | stem | sound | cue | source | peak |", "|---:|---:|---|---|---|---|---:|"]
        for (nm, stem, src), g in sorted(groups.items(), key=lambda kv: kv[1][0]["t"]):
            ts = f"{g[0]['t']:.3f}" if len(g) == 1 else f"{g[0]['t']:.2f}…{g[-1]['t']:.2f}"
            cue = ", ".join(sorted({e['cue'] for e in g if e['cue']}))
            L_.append(f"| {ts} | {len(g)} | {stem} | {nm} | {cue} | {src} | "
                      f"{20 * np.log10(max(max(e['peak'] for e in g), 1e-6)):.1f} dB |")
        L_.append("")
    (OUTD / "CUES.md").write_text("\n".join(L_))


class Events(B.Events):
    def find_any(self, keys):
        return [e for e in self.evs if any(k in e["kind"].lower() for k in keys)]


def main():
    t_start = time.time()
    OUTD.mkdir(parents=True, exist_ok=True)
    STEMS.mkdir(parents=True, exist_ok=True)
    ev = Events(FF.load_events())
    print(f"[fg-foley] {len(ev.evs)} scene events, {len(FF.load_tracks())} physics tracks in {FF.FOLEY_CACHE}")
    mx = Mix()
    for fn in (rain, ):
        fn(mx)
    for fn in (t01_t02, t03, t04, t05, t06, t07, t08_t09, comic_ticks):
        t = time.time()
        fn(mx, ev)
        print(f"[fg-foley] {fn.__name__} ({time.time() - t:.1f}s)")
    place_events(mx, ev)
    report = []
    physics(mx, report)
    mx.render_verbs()
    lims = finalize(mx)
    sf.write(STEMS / "sfx.wav", mx.bus["sfx"], SR, subtype="FLOAT")
    sf.write(STEMS / "ambience.wav", mx.bus["amb"], SR, subtype="FLOAT")
    (OUTD / "placements.json").write_text(json.dumps(mx.log, indent=0))
    (OUTD / "physics.json").write_text(json.dumps(report, indent=1))
    write_cues(mx, report, lims)
    for p in report:
        print(f"[physics] {p['file']:34s} {p['kind']:10s} r={p['corr']:.3f}")
    print(f"[fg-foley] {len(mx.log)} placements; done in {time.time() - t_start:.1f}s; limiter {lims}")


if __name__ == "__main__":
    main()

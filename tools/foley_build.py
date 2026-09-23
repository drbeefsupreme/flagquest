"""THE UNBABELING — Foley build (owner: Foley). Re-runnable; re-reads cache/foley on every run.

    source .venv/bin/activate && python tools/foley_build.py            # full build -> audio/stems/{sfx,ambience}.wav
    python tools/foley_build.py --only s03                               # audition one scene -> audio/foley/preview_s03.wav

Outputs
  audio/stems/sfx.wav, audio/stems/ambience.wav   48 kHz stereo float32, exactly 193.0 s (9,264,000 samples), T=0 aligned
  audio/foley/placements.json                      every placement (time, stem, sound, gain, source)
  audio/foley/physics.json                         per physics track: correlation energy <-> synthesized flutter RMS
  audio/foley/CUES.md                              human-readable cue sheet (generated)
Design: see tools/foley_lib.py (synthesis) and tools/foley_flutter.py (physics-derived flag foley).
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import foley_flutter as FF  # noqa: E402
import foley_lib as L  # noqa: E402
from foley_lib import SR, R, Snd, midi_hz, ns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TL = json.loads((ROOT / "timeline.json").read_text())
DUR = 193.0
NTOT = int(round(DUR * SR))
SIL0, SIL1 = 138.5, 139.6          # sacred silence: digital zero; only a faint wind from SIL1
OUTD = ROOT / "audio" / "foley"
STEMS = ROOT / "audio" / "stems"

CUES = {c["id"]: c["t"] for c in TL["cues"]}
LINES = {l["id"]: l for l in TL["lines"]}
SCENES = {s["id"]: s for s in TL["scenes"]}


def word(lid, w, k=0):
    ws = [x for x in LINES[lid]["words"] if x["w"].strip(",.!?").lower() == w.lower()]
    return ws[k]["s"]


# ------------------------------------------------------------------------------------------------
# spaces (procedural impulse responses)
# ------------------------------------------------------------------------------------------------
IR_SPECS = {
    "space": dict(rt60=7.0, predelay=0.03, band_mul=(1.1, 1.0, 0.95, 0.8), seed=1, er_span=0.12, early=2),
    "plain": dict(rt60=1.1, predelay=0.01, band_mul=(1.2, 1.0, 0.7, 0.4), seed=2, density=0.3,
                  echoes=[(0.23, 0.45, 3000), (0.47, 0.32, 2000), (0.83, 0.22, 1400), (1.35, 0.12, 900)]),
    "room": dict(rt60=0.55, predelay=0.004, band_mul=(1.1, 1.0, 0.8, 0.5), seed=3, er_span=0.025, early=8),
    "cavern": dict(rt60=5.5, predelay=0.045, band_mul=(1.3, 1.0, 0.7, 0.4), seed=4, er_span=0.14, early=10,
                   echoes=[(0.31, 0.3, 2500), (0.62, 0.2, 1500)]),
    "tunnel": dict(rt60=2.3, predelay=0.012, band_mul=(1.25, 1.0, 0.7, 0.4), seed=5, er_span=0.04,
                   flutter=(0.022, 0.72)),
    "map": dict(rt60=1.4, predelay=0.02, band_mul=(1.0, 1.0, 0.9, 0.7), seed=6),
    "temple": dict(rt60=4.2, predelay=0.035, band_mul=(1.2, 1.0, 0.75, 0.45), seed=7, er_span=0.1, early=10),
    "void": dict(rt60=9.0, predelay=0.06, band_mul=(1.0, 1.0, 0.85, 0.6), seed=8, er_span=0.2, early=0),
    "storm": dict(rt60=2.6, predelay=0.02, band_mul=(1.3, 1.0, 0.6, 0.3), seed=9, density=0.5,
                  echoes=[(0.4, 0.3, 1500), (0.9, 0.2, 900)]),
    "dawn": dict(rt60=1.7, predelay=0.02, band_mul=(1.2, 1.0, 0.7, 0.45), seed=10, density=0.3,
                 echoes=[(0.35, 0.25, 2500), (0.95, 0.15, 1200)]),
    "sky": dict(rt60=6.5, predelay=0.04, band_mul=(1.0, 1.0, 0.95, 0.85), seed=11, er_span=0.15, early=0),
    "meadow": dict(rt60=0.8, predelay=0.01, band_mul=(1.1, 1.0, 0.7, 0.4), seed=12, density=0.25,
                   echoes=[(0.28, 0.18, 2500)]),
}
SCENE_VERB = {"s01": "space", "s02": "plain", "s03": "room", "s04": "cavern", "s05a": "tunnel", "s06": "storm",
              "s07a": "dawn", "s07b": "sky", "s08": "meadow"}
SCENE_CUT = {"s04": 90.0, "s06": SIL0}   # hard cuts (no tails across them)
DUCKS = {  # cue: (sfx depth, ambience depth) for the pre-impact breath
    "title_slam": (0.3, 0.6), "slop_crash": (0.15, 0.4), "tv_on": (0.3, 0.8), "hypermeme": (0.5, 0.8),
    "breach": (0.2, 0.5), "nation_fracture": (0.4, 1.0), "faith_fracture": (0.4, 1.0), "philo_shatter": (0.25, 0.6),
    "babel_crack": (0.2, 0.5), "flag_planted": (0.5, 1.0), "glorious": (0.4, 0.8), "end_card": (0.4, 0.8),
    "flag_rotates_in": (0.5, 1.0),
}


def frame_turb(r, nfr, width=2.0):
    """per-frame turbulence (unit std) for synthetic energy curves."""
    from scipy.ndimage import gaussian_filter1d
    x = gaussian_filter1d(r.standard_normal(nfr), width)
    return x / (x.std() + 1e-9)


def verb_for(scene, t):
    if scene == "s05b":
        return "map" if t < 107.8 else ("temple" if t < 114.6 else "void")
    return SCENE_VERB.get(scene, "plain")


def scene_at(t):
    for s in TL["scenes"]:
        if s["start"] <= t < s["end"]:
            return s["id"]
    return "s08"


# ------------------------------------------------------------------------------------------------
# mixer
# ------------------------------------------------------------------------------------------------
class Mix:
    def __init__(self, only=None):
        self.bus = {"sfx": np.zeros((NTOT, 2), np.float32), "amb": np.zeros((NTOT, 2), np.float32)}
        self.sends = {}
        self.log = []
        self.only = only
        # pre-impact 'breath': material already sounding dips just before a big hit and recovers after it,
        # so the hit's transient lands clean on the cue (placements starting at the hit are exempt)
        self.ducks = [(CUES[c], d_sfx, d_amb) for c, (d_sfx, d_amb) in DUCKS.items()]

    def active(self, scene):
        return self.only is None or scene in self.only

    def add(self, stem, snd, t, gain=1.0, pan=None, send=0.0, verb=None, name="", scene=None, cue=None,
            src="design", cut=None, desc=""):
        if isinstance(snd, Snd):
            x, sync = snd.x, snd.sync
        else:
            x, sync = snd, 0.0
        scene = scene or scene_at(t)
        if not self.active(scene) or gain <= 0:
            return
        x = np.asarray(x, np.float32)
        if x.ndim == 1:
            x = L.pan2(x, 0.0 if pan is None else pan)
        elif pan is not None:
            x = L.pan2(x, pan)
        start = t - sync
        i0 = int(round(start * SR))
        # THE HARD CUT: nothing that starts before the silence may sound after it; s04 hard-cuts at 90.0
        if cut is None:
            cut = SCENE_CUT.get(scene)
        end_t = cut if cut is not None else (SIL0 if start < SIL0 else DUR)
        end_t = min(end_t, SIL0) if start < SIL0 else end_t
        i1 = min(NTOT, i0 + len(x), int(round(end_t * SR)))
        a = max(0, i0)
        if i1 <= a:
            return
        seg = x[a - i0:i1 - i0] * gain
        for td, d_sfx, d_amb in self.ducks:
            depth = d_sfx if stem == "sfx" else d_amb
            if depth >= 1.0 or start >= td - 0.005 or i1 <= int((td - 0.12) * SR) or a >= int((td + 0.4) * SR):
                continue
            tg = (np.arange(a, i1) / SR) - td
            g = np.interp(tg, [-0.12, -0.02, 0.0, 0.4], [1.0, depth, depth, 1.0]).astype(np.float32)
            seg = seg * g[:, None]
        if i1 == int(round(end_t * SR)) and i1 - a > 96:  # 2 ms de-click at a forced cut
            seg = seg.copy()
            seg[-96:] *= np.linspace(1, 0, 96)[:, None]
        self.bus[stem][a:i1] += seg
        verb = verb or verb_for(scene, t)
        if send > 0:
            key = (stem, verb, end_t)
            if key not in self.sends:
                self.sends[key] = np.zeros((NTOT, 2), np.float32)
            self.sends[key][a:i1] += seg * send
        self.log.append(dict(t=round(float(t), 4), start=round(float(start), 4), stem=stem, name=name, scene=scene,
                             cue=cue, src=src, gain=round(float(gain), 4), peak=round(float(np.abs(seg).max()), 4),
                             verb=verb if send > 0 else None, send=round(float(send), 3), desc=desc))

    def render_verbs(self):
        irs = {}
        for (stem, verb, end_t), buf in self.sends.items():
            nz = np.nonzero(np.abs(buf).max(1) > 0)[0]
            if len(nz) == 0:
                continue
            if verb not in irs:
                irs[verb] = L.make_ir(**IR_SPECS[verb])
            a, b = nz[0], nz[-1] + 1
            wet = L.convolve(buf[a:b], irs[verb])
            e = min(NTOT, a + len(wet), int(round(end_t * SR)))
            self.bus[stem][a:e] += wet[: e - a]
        self.sends = {}


# ------------------------------------------------------------------------------------------------
# scene events (exported by the scene agents)
# ------------------------------------------------------------------------------------------------
class Events:
    def __init__(self, evs):
        self.evs = evs
        for i, e in enumerate(self.evs):
            e["_i"] = i
        self.used = set()

    def _match(self, e, keys, kind_only=False):
        s = e["kind"].lower() if kind_only else (e["kind"] + " " + e["desc"]).lower()
        return any(k in s for k in keys)

    def find(self, scene, keys, near=None, win=1.5, all_=False, claim=True, t0=None, t1=None, kind_only=False):
        c = [e for e in self.evs if e["scene"] == scene and e["_i"] not in self.used
             and self._match(e, keys, kind_only) and (near is None or abs(e["t"] - near) <= win + 1e-6)
             and (t0 is None or e["t"] >= t0) and (t1 is None or e["t"] < t1)]
        if near is not None and not all_:
            c.sort(key=lambda e: abs(e["t"] - near))
        if not all_:
            c = c[:1]
        if claim:
            self.used.update(e["_i"] for e in c)
        return c if all_ else (c[0] if c else None)

    def t(self, scene, keys, fallback, win=1.5):
        e = self.find(scene, keys, near=fallback, win=win)
        return (e["t"], e["pan"], e["strength"]) if e else (fallback, None, None)

    def claim_near(self, scene, t, win, keys):
        for e in self.evs:
            if e["scene"] == scene and abs(e["t"] - t) <= win and self._match(e, keys):
                self.used.add(e["_i"])

    def unused(self):
        return [e for e in self.evs if e["_i"] not in self.used]


# ------------------------------------------------------------------------------------------------
# generic event -> sound (for every scene hit not already covered by a designed cue sound)
# ------------------------------------------------------------------------------------------------
def surface_for(scene, t):
    return {"s02": "concrete", "s03": "concrete", "s04": "stone", "s05a": "concrete", "s05b": "stone",
            "s06": "rubble", "s07a": "rubble", "s08": "grass"}.get(scene, "concrete")


def material_for(e):
    s = (e["kind"] + " " + e["desc"]).lower()
    for k, m in (("glass", "glass"), ("crystal", "crystal"), ("screen", "screen"), ("phone", "screen"),
                 ("marble", "marble"), ("metal", "metal"), ("steel", "metal"), ("coin", "metal"), ("wood", "wood"),
                 ("pole", "wood"), ("fence", "wood"), ("stone", "stone"), ("rock", "stone"), ("brick", "stone"),
                 ("rubble", "stone"), ("crown", "metal"), ("tin", "tin")):
        if k in s:
            return m
    return "stone"


def nvg_whine(r):
    n = ns(0.9)
    t = L.tt(n)
    f = 3800 + 5200 * (1 - np.exp(-t / 0.25))
    x = np.sin(L.TAU * np.cumsum(f) / SR) * L.env_pts(n, [(0, 0), (0.05, 1), (0.6, 0.5), (0.9, 0)])
    x += L.hp(L.white(n, r), 6000) * 0.05 * L.env_ad(n, 0.01, 0.4)
    return Snd(L.fade(L.f32(x) * 0.5), 0.0)


def tick_line(r, dur=0.12):
    """a fine line drawn in (grid): tiny pen-like swish."""
    n = ns(dur)
    u = L.tt(n) / dur
    x = L.svf(L.white(n, r), 3000 + 5000 * u, 1.5, "bp") * np.sin(np.pi * u) ** 2
    return Snd(L.fade(L.norm(x, 0.6)), 0.0)


def letter_thock(r, k=0):
    x = L.thud(r, 95 * 2 ** (r.integers(-3, 4) / 12), 0.22, 0.9, 1500)
    c = L.impact(r, "wood", 0.8, 0.5, 0.1)
    x[: len(c)] += c * 0.6
    return Snd(L.fade(L.norm(x, 0.8)), 0.0)


def splash(r, size=1.0):
    dur = 0.8 + 0.8 * size
    n = ns(dur)
    t = L.tt(n)
    x = L.svf(L.pink(n, r), 300 + 3000 * np.exp(-t / 0.08), 0.8, "lp") * L.env_ad(n, 0.003, 0.35 * size)
    st = L.as_stereo(x) + L.bubbles_cloud(r, dur, lambda tt_: 400 * np.exp(-tt_ / 0.2) * size + 5,
                                          (1.5, 6.0), 0.5)[:n] * 0.5
    return Snd(L.fade(L.norm(st, 0.95), 0.001, 0.1), 0.0)


def gloop(r):
    return Snd(L.bubble(r, r.uniform(5, 11), r.uniform(0.3, 0.9), 1.0, 0.6), 0.0)


def pat(r):
    x = L.thud(r, 180, 0.12, 1.0, 2500)
    x += L.cloth_rustle(r, 0.12, 2.0).x[: len(x)] * 0.4
    return Snd(L.fade(L.norm(x, 0.7)), 0.0)


def whoosh_short(r, strength=0.6, pan=0.0):
    d = 0.35 + 0.5 * strength
    return L.whoosh(r, d, d * 0.6, 300, 1800 + 2500 * strength, 1.2, pan - 0.3, pan + 0.3)


def flash_swell(r, strength=0.6):
    return L.swell(r, 0.25 + 0.5 * strength, 800, 9000, 2.5, 0.9)


def ember_rise(r, dur=2.0, count=26, pitches=L.D_MAJOR):
    n = ns(dur)
    evs = []
    ms = sorted(r.choice(pitches, count) + 12 * r.integers(1, 3, count))
    for k, m in enumerate(ms):
        x = L.chime(r, float(midi_hz(m + 12)), 1.5, (1.0, 2.76, 5.4), 0.8, 3, 0.8, 0.05)
        evs.append((dur * 0.85 * k / count + r.uniform(0, 0.05), x, r.uniform(0.3, 0.8), r.uniform(-0.8, 0.8)))
    out = L.scatter(n, evs)
    cr = L.hp(L.poisson_clicks(r, n, 40), 3000)
    out += L.decorrelate(cr * 0.25, r)
    return Snd(L.norm(out, 0.9), 0.0)


SPECIFIC = [  # matched first against "kind + desc" (word-prefix match; trailing $ = whole word)
    (("drip",), "drip"), (("velcro",), "velcro"), (("telescop",), "telescope"), (("hammer", "tack$"), "hammer"),
    (("lightning", "thunder"), "thunder"), (("crt thump", "screens power"), "crtcascade"),
    (("peel",), "peel"), (("squelch", "strand", "sticky", "stretch"), "stretch"), (("sniff",), "sniff"),
    (("pen$", "scratch", "flourish"), "pen"), (("coin", "ching", "register"), "coin"),
    (("klaxon", "alarm", "siren"), "klaxon"), (("wink",), "wink"), (("letter",), "letter"),
    (("grid",), "grid"), (("ember",), "embers"), (("candle", "match$"), "candle"), (("salute",), "raise"),
    (("wipe", "swipe"), "wipe"), (("rattle", "boing"), "boing"), (("crown",), "crown"), (("skid",), "skid"),
    (("swings the flag", "jab", "thrust"), "flagswish"), (("bird",), "birds"), (("descend", "lowers"), "descend"),
]
GENERIC = [
    (("explos", "blast", "breach", "boom"), "explosion"), (("shatter",), "shatter"),
    (("crack", "fractur", "split", "subdivi"), "crack"), (("finger", "touch", "pinch"), "touch"),
    (("pat$", "pats$", "slap", "strap"), "pat"), (("glitch", "static", "flicker"), "glitch"),
    (("power on", "power up"), "poweron"), (("rumble", "hum$"), "rumble"), (("sweep",), "sweep"),
    (("swell",), "swell"),
    (("zap", "beam", "energy", "charge"), "zap"),
    (("ping", "notif", "beep", "blip", "readout", "tick$", "ticks$", "hud", "holo", "typing"), "blip"),
    (("chime", "sparkle", "glint", "shimmer", "twinkle", "absorb", "ingest", "materiali"), "chime"),
    (("splash", "slosh", "ripple", "jolt"), "splash"), (("gloop", "bubble"), "gloop"),
    (("dust", "puff"), "dust"), (("creak", "groan"), "creak"),
    (("raise", "unfurl", "flutter"), "raise"),
    (("footstep", "step", "boot", "stomp", "foot$", "feet$", "walk", "run$", "runs$", "climb", "scrape"), "step"),
    (("plant", "thunk", "strike"), "thunk"), (("snap",), "snap"),
    (("slide", "slides$", "grind", "door", "seal", "open"), "slide"),
    (("whoosh", "swish", "swing", "pass$", "passes$", "fly$", "flies$", "swirl", "gust", "descent"), "whoosh"),
    (("flash", "bloom", "glow"), "flash"),
    (("rustle", "cloth", "robe", "turn$", "turns$", "offer", "hood", "grab", "takes$"), "rustle"),
    (("slam", "impact", "lands$", "land$", "hit$", "hits$", "stack", "thud", "drop$", "drops$", "fall"), "impact"),
    (("click", "lock"), "click"),
]
SKIP_KINDS = {"wind", "cut", "black", "ambience", "ambient", "silence"}   # covered by beds / hard cuts


def _rx(keys):
    import re
    return re.compile("|".join(r"\b" + (k[:-1] + r"\b" if k.endswith("$") else k) for k in keys))


_SPEC = [(_rx(k), h) for k, h in SPECIFIC]
_GEN = [(_rx(k), h) for k, h in GENERIC]


def classify(e):
    k = e["kind"].replace("_", " ").lower()
    s = k + " " + e["desc"].lower()
    if k in SKIP_KINDS:
        return "skip"
    if ("nvg" in s or "goggle" in s) and ("whine" in s or "power" in s):
        return "nvg"
    for rx, h in _SPEC:
        if rx.search(s):
            return h
    for rx, h in _GEN:
        if rx.search(k):
            return h
    for rx, h in _GEN:
        if rx.search(s):
            return h
    return "impact"


def hymn_chord(t):
    """pitch classes of the hymn chord sounding at global t (for chord-aware glints in s07b)."""
    hy = TL["hymn"]
    beat = (t - hy["start"]) / (60.0 / hy["bpm"])
    for w, b0, nb, notes in hy["notes"]:
        if b0 <= beat < b0 + nb:
            return sorted({n % 12 for n in notes})
    return [2, 6, 9]   # D major


def chord_glint(r, t, octave=6, dur=2.0, t60=1.4):
    pcs = hymn_chord(t)
    m = 12 * octave + int(r.choice(pcs))
    return Snd(L.chime(r, float(midi_hz(m)), dur, t60=t60, beat=1.2, strike=0.15), 0.0)


def crown_tinks(r, count=12, dur=0.35):
    n = ns(dur + 0.6)
    evs = [(r.uniform(0, dur), L.impact(r, "metal", r.uniform(0.25, 0.45), 1.0, 0.4), r.uniform(0.3, 1.0),
            r.uniform(-0.9, 0.9)) for _ in range(count)]
    return Snd(L.norm(L.scatter(n, evs), 0.9), 0.0)


def boing(r):
    """spring-y rabbit-ear rattle: metal rod modes with a wobbling pitch + rattle clicks."""
    n = ns(0.9)
    t = L.tt(n)
    f = 310 * (1 + 0.08 * np.sin(L.TAU * 7 * t) * np.exp(-t / 0.3))
    x = np.sin(L.TAU * np.cumsum(f) / SR) * L.env_ad(n, 0.002, 0.6)
    x += 0.4 * np.sin(L.TAU * np.cumsum(f * 2.76) / SR) * L.env_ad(n, 0.002, 0.3)
    x += L.hp(L.poisson_clicks(r, n, 60 * np.exp(-t / 0.2)), 2000) * 0.6
    return Snd(L.fade(L.norm(L.f32(x), 0.8)), 0.0)


def sticky_peel(r, dur=0.32):
    """a fingertip peeling off a tacky surface: accelerating adhesive-release crackle (micro-ruptures of the
    film) into a small suction pop as the last contact lets go."""
    n = ns(dur)
    u = L.tt(n) / dur
    exc = L.poisson_clicks(r, n, 300 + 2500 * u ** 2, 0.4 + 0.6 * u)
    x = L.resonators(exc, [1300, 2100, 3400], [0.006, 0.005, 0.004], [1, 0.7, 0.5]) + L.bp(exc, 2500, 8000) * 0.4
    pop = L.bubble(r, 1.6, 1.2, 1.0, 0.5)
    y = np.zeros(n + len(pop), np.float32)
    y[:n] = x
    y[n - ns(0.01):n - ns(0.01) + len(pop)] += pop * 0.8
    return Snd(L.fade(L.norm(y, 0.8), 0.01, 0.02), 0.0)


def skid(r, dur=0.35):
    n = ns(dur)
    u = L.tt(n) / dur
    x = L.bp(L.white(n, r), 1200, 7000) * (1 - u) ** 1.5 * np.clip(u * 20, 0, 1)
    x += L.bp(L.poisson_clicks(r, n, 900 * (1 - u)), 2000, 8000) * 0.4
    return Snd(L.fade(L.norm(x, 0.7)), 0.0)


def flag_swish(r, strength=0.6):
    return Snd(FF.raise_grain(r, 0.0, snap=strength > 0.4), 0.05)


def wing_burst(r, count=40, dur=1.6):
    """a flock bursting up: many wing-beat grains (each bird ~12 Hz flaps decaying) + a few chips."""
    n = ns(dur + 0.5)
    evs = []
    for _ in range(count):
        m = ns(0.5)
        tt_ = L.tt(m)
        rate = r.uniform(9, 15)
        fl = np.exp(-((tt_ * rate) % 1.0) * 6) * np.exp(-tt_ / 0.3)
        g = L.bp(L.white(m, r), 500, 4000) * fl
        evs.append((r.exponential(0.25), g, r.uniform(0.2, 1.0), r.uniform(-1, 1)))
    for _ in range(5):
        evs.append((r.uniform(0.1, 1.0), L.bird(r, "chip", 20).x, 0.4, r.uniform(-1, 1)))
    return Snd(L.norm(L.scatter(n, evs), 0.9), 0.0)


def crt_cascade(r, dur=1.0, count=7):
    n = ns(dur + 1.8)
    evs = [(dur * (k / max(1, count - 1)) ** 1.3, L.crt_on(r).x, r.uniform(0.5, 1.0), r.uniform(-0.7, 0.7))
           for k in range(count)]
    return Snd(L.norm(L.scatter(n, evs), 0.95), 0.0)


def rumble_swell(r, dur=1.4, water=False):
    n = ns(dur)
    x = L.svf(L.brown(n, r), 90 + 120 * np.sin(np.pi * L.tt(n) / dur), 0.8, "lp")
    x *= np.sin(np.pi * L.tt(n) / dur) ** 1.5
    st = L.decorrelate(L.norm(x, 0.8), r)
    if water:
        st += L.lp(L.ocean_roar(r, dur, 0.6, 0.8), 1500) * np.sin(np.pi * L.tt(n) / dur)[:, None] ** 2
    return Snd(st, 0.0)


def power_on(r):
    hm = L.hum(r, 1.5, 55.0, (0.5, 0.4, 0.3, 0.2), 0.3) * L.env_pts(ns(1.5), [(0, 0), (0.05, 1), (1.5, 0)])
    c = L.relay_clunk(r, 1.2)
    hm[: len(c)] += c
    return Snd(L.fade(L.norm(hm, 0.8)), 0.0)


def light_sweep(r, dur=0.8):
    """a line/plane of light sweeping: airy rising band-pass (no pitch)."""
    n = ns(dur)
    u = L.tt(n) / dur
    x = L.svf(L.pink(n, r), 1500 * 6 ** u, 3.0, "bp") * np.sin(np.pi * u) ** 1.2
    return Snd(L.fade(L.decorrelate(L.norm(x, 0.7), r)), 0.0)


def dur_from_desc(e, default):
    """parse '(to 12.0)' / '(0.55 s)' / '20.42->20.9' hints in scene event descriptions."""
    import re
    d = e["desc"]
    m = re.search(r"to ~?(\d+\.\d+)", d)
    if m and float(m.group(1)) > e["t"]:
        return float(m.group(1)) - e["t"]
    m = re.search(r"(\d+\.\d+)\s*s\)", d)
    if m:
        return float(m.group(1))
    m = re.search(r"->\s*(\d+\.\d+)", d)
    if m and float(m.group(1)) > e["t"]:
        return float(m.group(1)) - e["t"]
    return default


TV_WORLD_EXCLUDE = ("slop", "slosh", "armchair", "antenna", "jolt", "floating", "cat ", "drip", "crt", "static",
                    "vertical hold", "collapse")


def event_sound(e, r):
    """-> (Snd, stem, base_gain, send). Level = base_gain * (0.35 + 0.65 * strength). None = skip."""
    h = classify(e)
    sc, st = e["scene"], float(np.clip(e.get("strength", 1.0), 0, 1))
    mat = material_for(e)
    out = _event_sound(h, e, r, sc, st, mat)
    if out is None:
        return None
    snd, stem, g, send = out
    # s03: whatever happens inside the broadcast is heard through the CRT's little speaker
    if sc == "s03" and 48.0 < e["t"] < 61.0 and not any(w in e["desc"].lower() for w in TV_WORLD_EXCLUDE) \
            and h in ("whoosh", "impact", "pat", "pen", "click", "rustle"):
        x = snd.x.mean(1) if snd.x.ndim == 2 else snd.x
        snd = Snd(L.tvify(x), snd.sync)
        send = 0.3
    return snd, stem, g, send


def _event_sound(h, e, r, sc, st, mat):
    if h == "skip":
        return None
    if h == "drip":
        return L.drip(r), "amb", 0.10, 0.5
    if h == "velcro":
        return L.velcro(r, 0.12 + 0.2 * st), "sfx", 0.30, 0.1
    if h == "telescope":
        return Snd(L.impact(r, "metal", 0.6, 1.0, 0.3), 0.0), "sfx", 0.28, 0.12
    if h == "hammer":
        return L.hammer_tap(r, 1), "sfx", 0.40, 0.12
    if h == "nvg":
        return nvg_whine(r), "sfx", 0.12, 0.1
    if h == "thunder":
        s = L.thunder(r, st)
        return Snd(s.x, -(0.05 + (1 - st) * 1.1)), "sfx", 0.30 + 0.35 * st, 0.15
    if h == "crtcascade":
        return crt_cascade(r, dur_from_desc(e, 1.0)), "sfx", 0.25, 0.3
    if h == "explosion":
        return L.explosion(r, 0.6 + 0.6 * st, 4.0, mat, int(20 + 40 * st)), "sfx", 0.55, 0.3
    if h == "shatter":
        return L.shatter(r, mat if mat in ("glass", "marble", "screen", "stone") else "stone", 0.5 + 0.7 * st,
                         int(40 + 120 * st), 3.0, gravity=sc != "s05b"), "sfx", 0.40, 0.3
    if h == "crack":
        return L.stone_crack(r, 0.05 + 0.08 * st, 1.0, "stone", 1.2, final_bang=st > 0.6), "sfx", 0.25, 0.3
    if h == "peel":
        return L.squelch_touch(r), "sfx", 0.22, 0.05
    if h == "stretch":
        return L.squelch_stretch(r, 0.7), "sfx", 0.22, 0.05
    if h == "touch":
        return L.squelch_touch(r), "sfx", 0.18, 0.05
    if h == "sniff":
        return L.sniff(r, 1), "sfx", 0.18, 0.05
    if h == "pen":
        return L.pen_scratch(r, 0.4 + 0.5 * st), "sfx", 0.25, 0.1
    if h == "coin":
        return L.ka_ching(r), "sfx", 0.35, 0.1
    if h == "klaxon":
        return L.klaxon_whoop(r, 1.1), "sfx", 0.16, 0.4
    if h == "candle":
        n = ns(0.35)
        m = L.svf(L.white(n, r), 2500 * np.exp(-L.tt(n) / 0.1) + 600, 1.5, "bp") * L.env_ad(n, 0.004, 0.25)
        return Snd(L.f32(m), 0.0), "sfx", 0.05, 0.4
    if h == "wink":
        return Snd(L.chime(r, float(midi_hz(86)), 1.2, (1.0, 2.76, 5.4), 0.7, 2, 1.0, 0.1), 0.0), "sfx", 0.07, 0.2
    if h == "pat":
        return pat(r), "sfx", 0.20, 0.05
    if h == "letter":
        return letter_thock(r), "sfx", 0.22, 0.2
    if h == "grid":
        return tick_line(r), "sfx", 0.06, 0.2
    if h == "wipe":
        return tick_line(r, dur_from_desc(e, 0.5)), "sfx", 0.07, 0.2
    if h == "boing":
        return boing(r), "sfx", 0.10, 0.2
    if h == "crown":
        return crown_tinks(r, 6 + int(14 * st)), "sfx", 0.07, 0.3
    if h == "skid":
        return skid(r), "sfx", 0.16, 0.1
    if h == "flagswish":
        return flag_swish(r, st), "sfx", 0.22, 0.1
    if h == "descend":   # something colossal lowers out of the storm: vast dark air + electrical static swell
        w = L.whoosh(r, 2.4, 1.6, 50, 500, 0.7, 0.0, 0.0, dark=0.85, shape=1.2)
        n = len(w.x)
        stc = L.tv_static(r, n / SR) * L.env_pts(n, [(0, 0), (1.6, 1), (n / SR, 0)])[:, None]
        return Snd(L.norm(w.x + L.lp(stc, 3000, 2) * 0.3, 0.9), w.sync), "sfx", 0.25, 0.4
    if h == "birds":
        return wing_burst(r, 30 + int(40 * st)), "sfx", 0.12, 0.3
    if h == "rumble":
        return rumble_swell(r, 1.4, water=any(w in e["desc"].lower() for w in ("sea", "flood", "water"))), \
            "sfx", 0.12, 0.3
    if h == "poweron":
        return power_on(r), "sfx", 0.18, 0.4
    if h == "sweep":
        return light_sweep(r, dur_from_desc(e, 0.8)), "sfx", 0.07, 0.4
    if h == "swell":
        return L.swell(r, 0.8, 200, 4000, 1.8), "sfx", 0.10, 0.3
    if h == "glitch":
        return Snd(L.glitch(r, 0.1 + 0.3 * st), 0.0), "sfx", 0.18, 0.1
    if h == "zap":
        return L.zap(r, 0.3 + 0.3 * st, 3000 + 2000 * st, 300), "sfx", 0.14, 0.3
    if h == "blip":
        m = r.choice(L.D_MAJOR) + 24
        return Snd(L.ping(r, float(midi_hz(m)), 0.12, r.integers(0, 3)), 0.0), "sfx", 0.05, 0.2
    if h == "chime":
        if sc == "s07b":
            return chord_glint(r, e["t"]), "sfx", 0.05, 0.5
        m = r.choice(L.D_MAJOR) + 12
        return Snd(L.chime(r, float(midi_hz(m)), 2.5, t60=1.6, beat=1.5), 0.0), "sfx", 0.08, 0.4
    if h == "embers":
        return ember_rise(r, 1.5, 12), "sfx", 0.12, 0.4
    if h == "splash":
        return splash(r, 0.3 + 0.7 * st), "sfx", 0.20, 0.2
    if h == "gloop":
        return gloop(r), "amb", 0.12, 0.3
    if h == "dust":
        return Snd(L.dust_puff(r, 0.5, 0.4), 0.0), "sfx", 0.15, 0.2
    if h == "creak":
        return L.creak(r, 1.2 + st), "sfx", 0.2, 0.3
    if h == "raise":
        return Snd(FF.raise_grain(r, 0.2, True), 0.0), "sfx", 0.20, 0.2
    if h == "step":
        return L.boot_step(r, surface_for(sc, e["t"]), 0.6 + 0.4 * st, sc in ("s05a",)), "sfx", 0.22, 0.2
    if h == "thunk":
        if any(w in e["desc"].lower() for w in ("paper", "page", "card")):
            return L.page_slam(r, 0.8), "sfx", 0.25, 0.1
        return L.pole_thunk(r), "sfx", 0.35, 0.25
    if h == "snap":
        return Snd(FF.snap_crack(r, st), 0.0), "sfx", 0.30, 0.15
    if h == "slide":
        if any(k in e["desc"].lower() for k in ("page", "paper", "card")):
            return L.paper_slide(r, 0.5), "sfx", 0.2, 0.1
        return L.stone_slide(r, 0.8 + st, 1.0), "sfx", 0.3, 0.35
    if h == "whoosh":
        return whoosh_short(r, st), "sfx", 0.22, 0.2
    if h == "flash":
        return flash_swell(r, st), "sfx", 0.10, 0.3
    if h == "rustle":
        return L.cloth_rustle(r, 0.3 + 0.3 * st, 0.6 + 0.8 * st), "sfx", 0.16, 0.08
    if h == "click":
        return Snd(L.impact(r, "tin" if mat == "stone" else mat, 0.5, 1.0, 0.15), 0.0), "sfx", 0.18, 0.1
    # impact
    x = L.thud(r, 70 + 60 * (1 - st), 0.35, 0.9, 1200)
    im = L.impact(r, mat, 1.0 + st, 1.0)
    y = np.zeros(max(len(x), len(im)), np.float32)
    y[: len(x)] += x
    y[: len(im)] += im * 0.8
    return Snd(L.fade(L.norm(y, 0.9)), 0.0), "sfx", 0.30, 0.2


def place_events(mx, ev):
    r = R(777)
    for e in ev.unused():
        ev.used.add(e["_i"])
        if not mx.active(e["scene"]) or not (0.0 <= e["t"] < DUR):
            continue
        out = event_sound(e, r)
        if out is None:
            continue
        snd, stem, g, send = out
        mx.add(stem, snd, e["t"], g * (0.35 + 0.65 * float(np.clip(e["strength"], 0, 1))), e["pan"], send,
               scene=e["scene"], name=classify(e), src="event:" + e["file"], desc=e["desc"])


# ------------------------------------------------------------------------------------------------
# helpers for beds
# ------------------------------------------------------------------------------------------------
def shaped(x, t0, pts):
    """multiply stereo/mono x (starting at global t0) by a piecewise-linear gain envelope in global time."""
    n = len(x)
    g = np.interp(np.arange(n) / SR + t0, [p[0] for p in pts], [p[1] for p in pts]).astype(np.float32)
    return x * (g[:, None] if x.ndim == 2 else g)


def stereo_bed(fn, r, dur):
    """two independent realisations of a mono generator -> wide stereo bed."""
    return L.stereo(fn(r, dur), fn(r, dur))


def rumble(r, dur, fc=80.0, q=0.7):
    n = ns(dur)
    return L.svf(L.brown(n, r), fc * (1 + 0.2 * L.smooth_noise(n, r, 0.3, 2)), q, "lp")


def room_tone(r, dur, fc=600.0):
    n = ns(dur)
    return L.lp(L.pink(n, r), fc, 2)


def slosh(r, dur, rate=0.33, visc=1.0):
    """viscous slop lapping: low band-passed noise with slow wave AM + gloopy bubbles."""
    n = ns(dur)
    t = L.tt(n)
    am = 0.5 + 0.5 * np.sin(L.TAU * rate * t + 2 * L.smooth_noise(n, r, 0.2, 2)) ** 2
    x = L.svf(L.brown(n, r), 220 + 250 * am, 1.2, "bp") * (0.4 + 0.6 * am)
    x += L.svf(L.pink(n, r), 900 + 600 * am, 2.5, "bp") * am ** 3 * 0.35
    return L.f32(x)


def ping_bank(r, pitches, count=48, detune=35):
    bank = []
    for i in range(count):
        m = r.choice(pitches) + 12 * r.integers(1, 3) + r.uniform(-detune, detune) / 100
        bank.append(L.ping(r, float(midi_hz(m)), r.uniform(0.1, 0.3), int(r.integers(0, 3))))
    return bank


def march(r, beats, n_people=160, spread=0.012, dist=0.7):
    """massed lockstep footsteps: each beat = n_people footfalls within +-spread seconds, distant."""
    evs = []
    steps = [L.boot_step(r, "concrete", 1.0, False).x for _ in range(24)]
    t_base = beats[0] - 0.05
    dur = beats[-1] - t_base + 0.6
    for b in beats:
        for p in range(n_people):
            s = steps[r.integers(len(steps))]
            evs.append((b - t_base + r.normal(0, spread), s, r.uniform(0.3, 1.0) / np.sqrt(n_people) * 3,
                        r.uniform(-0.9, 0.9)))
    x = L.scatter(ns(dur), evs)
    x = L.lp(x, 2600 * (1 - dist) + 700, 2)
    return L.norm(x, 0.9), t_base


# ================================================================================================
# SCENES
# ================================================================================================
def s01(mx, ev):
    sc = "s01"
    r = R(101)
    # --- ambience: the pre-cosmic void (no tonal hum: Composer owns the D drone) ---------------
    d = 21.3
    n = ns(d)
    sw = 0.5 + 0.5 * L.smooth_noise(n, r, 0.08, 2)
    air = L.stereo(L.svf(L.brown(n, r), 70 + 90 * sw, 0.7, "lp"), L.svf(L.brown(n, r), 70 + 90 * sw, 0.7, "lp"))
    ether = L.stereo(L.hp(L.pink(n, r), 7000), L.hp(L.pink(n, r), 7000)) * (0.15 + 0.1 * sw[:, None])
    bed = air * 0.5 + ether * 0.25
    bed = shaped(bed, 0.0, [(0, 0), (0.4, 0.0), (3.0, 1.0), (15.95, 1.0), (16.05, 0.0), (21.3, 0.0)])
    mx.add("amb", bed, 0.0, 0.035, name="void air + ether", scene=sc)
    paper = L.stereo(room_tone(r, 5.2, 1800), room_tone(r, 5.2, 1800))
    mx.add("amb", shaped(paper, 16.0, [(16.0, 0), (16.1, 1), (20.3, 1), (21.1, 0)]), 16.0, 0.02,
           name="title-card paper room", scene=sc)
    # golden dust motes: tiny D-major glints
    evs = []
    for k in range(34):
        t = r.uniform(0.8, 15.5)
        m = r.choice(L.D_MAJOR) + 24
        evs.append((t, L.chime(r, float(midi_hz(m)), 1.6, (1.0, 2.76, 5.4), 0.9, 2.5, 0.6, 0.02),
                    r.uniform(0.2, 1.0) * min(1.0, t / 3.0), r.uniform(-0.9, 0.9)))
    mx.add("amb", L.scatter(ns(17.5), evs), 0.0, 0.035, send=0.5, name="golden dust mote glints", scene=sc)
    # --- 2.6 a point of light kindles; 2.95 the Crystal condenses; ~5.55 it is complete ----------------
    t_k, _, _ = ev.t(sc, ["kindle", "point of light"], 2.6, 0.5)
    mx.add("sfx", L.swell(r, 1.2, 800, 12000, 2.5), t_k, 0.08, send=0.6, name="a point of light kindles (air)",
           scene=sc)
    mx.add("sfx", Snd(L.chime(r, float(midi_hz(98)), 4.0, t60=3.0, beat=0.6, strike=0.3), 0.0), t_k, 0.05, send=0.7,
           name="kindle glint (D7)", scene=sc)
    t_c, _, _ = ev.t(sc, ["condens"], 2.95, 1.0)
    t_d, _, _ = ev.t(sc, ["complete", "last edge"], 5.55, 1.0)
    # vertices unfold: accelerating ascending glints (D major) + rising air, arriving at completion
    nsp = 28
    evs = []
    for k in range(nsp):
        u = (k / nsp) ** 0.7
        m = L.D_MAJOR[k % 7] + 12 * (5 + (k * 2) // 7 % 3)
        evs.append((u * (t_d - t_c), L.chime(r, float(midi_hz(m)), 2.0, (1.0, 2.76, 5.4), 1.2, 2, 0.7, 0.08),
                    0.4 + 0.6 * u, r.uniform(-0.8, 0.8)))
    mx.add("sfx", L.scatter(ns(t_d - t_c + 2.2), evs), t_c, 0.05, send=0.6, name="the Crystal condenses (vertices glint)",
           scene=sc)
    mx.add("sfx", L.swell(r, t_d - t_c, 300, 11000, 2.2), t_d, 0.14, send=0.5, name="condensing light (air swell)",
           scene=sc)
    ch = np.zeros(ns(9), np.float32)
    for k, m in enumerate([74, 81, 86, 90]):   # D5 A5 D6 F#6
        c = L.chime(r, float(midi_hz(m)), 8.5, t60=5.0, beat=0.8, bright=0.8, strike=0.2)
        i = ns(0.05 * k)
        ch[i:i + len(c)] += c[: len(ch) - i] * (1 - 0.15 * k)
    mx.add("sfx", L.decorrelate(ch, r, 0.5), t_d, 0.11, send=0.6, name="Crystal complete: glass chord (D-A-D-F#)",
           scene=sc)
    # --- the rubbed-glass voice of the rotating Crystal (t_d .. 11.9) ------------------------------------
    # stick-slip friction locked to the glass mode (the physics of a rubbed wine glass); the standing wave
    # rotates with the 4-D rotation -> amplitude beating at the rotation rate, accelerating at N02 (5.9, peak ~8.6).
    t_acc, _, _ = ev.t(sc, ["rotation", "accelerat"], 5.9, 0.8)
    d = 12.0 - t_d
    n = ns(d)
    t = L.tt(n) + t_d
    rot = np.interp(t, [t_d, t_acc, 8.6, 10.0, 11.3, 12.0], [0.3, 0.4, 3.0, 2.2, 1.4, 0.8])
    press = np.interp(t, [t_d, t_d + 0.8, t_acc, 8.6, 10.8, 11.95], [0.0, 0.45, 0.55, 1.0, 0.7, 0.0])
    out = np.zeros(n, np.float32)
    for f0, lv, dly in ((587.33, 1.0, 0.0), (880.0, 0.55, t_acc - t_d + 0.6)):
        exc = L.stick_slip(r, d, f0, 0.002, press * (t > t_d + dly))
        body = L.resonators(exc, [f0 - 0.7, f0 + 0.7, f0 * 2.32, f0 * 4.25], [3.0, 3.0, 1.5, 0.8],
                            [1.0, 1.0, 0.35, 0.12])
        out += body * lv
    ph = L.TAU * np.cumsum(rot) / SR
    beat = 0.55 + 0.45 * np.cos(ph) ** 2
    out = L.norm(out * beat, 0.9)
    wide = L.pan2(out, 0.35 * np.sin(ph * 0.5))
    mx.add("sfx", L.fade(wide, 0.5, 0.4), t_d, 0.07, send=0.6, name="rubbed-glass Crystal voice (stick-slip, 4-D beating)",
           scene=sc)
    ev.claim_near(sc, t_acc, 0.01, ["rotation"])
    # --- 10.85 the hyperplane sweeps in; 12.0 flag_rotates_in: it slices the Crystal ----------------------
    e = ev.find(sc, ["hyperplane", "plane appears"], near=10.85, win=0.8)
    if e:
        dd = dur_from_desc(e, 1.15)
        mx.add("sfx", light_sweep(r, dd), e["t"], 0.09, e["pan"], send=0.5, name="the hyperplane sweeps in", scene=sc)
    T = CUES["flag_rotates_in"]
    ev.claim_near(sc, T, 0.25, ["slice", "hyperplane", "rotat", "flash"])
    mx.add("sfx", L.whoosh(r, 1.1, 0.35, 900, 9000, 2.0, -0.7, 0.7), T, 0.16, send=0.5,
           name="hyperplane slice 'shhing'", scene=sc, cue="flag_rotates_in")
    mx.add("sfx", L.chime(r, float(midi_hz(98)), 3.0, t60=1.5, beat=2.0, strike=0.6), T, 0.06, send=0.5,
           name="slice glint (D7 glass)", scene=sc, cue="flag_rotates_in")
    mx.add("sfx", Snd(L.impact(r, "glass", 1.1, 1.0, 0.8), 0.0), T, 0.22, 0.1, send=0.5,
           name="the slice: glass 'tink'", scene=sc, cue="flag_rotates_in")
    mx.add("sfx", L.sparkle(r, 1.8, 18, octave=(2, 3)), T + 0.1, 0.035, send=0.6, name="cross-section morph shimmer",
           scene=sc)
    e = ev.find(sc, ["pole draw", "line of light"], near=13.22, win=1.0)
    if e:
        mx.add("sfx", light_sweep(r, max(0.2, dur_from_desc(e, 0.25))), e["t"], 0.08, e["pan"], send=0.5,
               name="the pole draws itself as light", scene=sc)
    t_w, p_w, _ = ev.t(sc, ["becomes wood"], 13.47, 0.8)
    mx.add("sfx", Snd(L.impact(r, "wood", 1.2, 1.0, 0.4), 0.0), t_w, 0.10, p_w, send=0.3, name="light becomes wood 'tok'",
           scene=sc)
    # --- 14.2 flag_snap_open ---------------------------------------------------------------------
    T = CUES["flag_snap_open"]
    ev.claim_near(sc, T, 0.2, ["snap", "gust", "bloom", "flash"])
    ev.claim_near(sc, T + 0.1, 0.15, ["reflection", "shimmer"])
    mx.add("sfx", L.whoosh(r, 1.0, 0.45, 200, 2600, 0.9, -0.6, 0.4, dark=0.3), T, 0.30, send=0.3,
           name="first gust", scene=sc, cue="flag_snap_open")
    mx.add("sfx", Snd(FF.snap_crack(r, 1.0), 0.0), T, 0.55, 0.05, send=0.3, name="FIRST SNAP (cloth whip-crack)",
           scene=sc, cue="flag_snap_open")
    mx.add("sfx", L.sub_boom(r, 50, 30, 1.4, 0.2), T, 0.30, name="snap air-push (sub)", scene=sc, cue="flag_snap_open")
    mx.add("sfx", L.sparkle(r, 2.2, 22, octave=(2, 3)), T + 0.03, 0.05, send=0.6,
           name="bloom: the Flag reflected endlessly in the facets", scene=sc)
    # 15.3 reverse-suck is the Composer's; Foley adds only a faint air draw-in
    e = ev.find(sc, ["reverse-suck", "draws inward"], near=15.3, win=0.3)
    if e:
        mx.add("sfx", L.swell(r, 16.0 - e["t"], 400, 5000, 2.0), 15.95, 0.04, name="air draws in (under music suck)",
               scene=sc)
    # --- 16.0 title_slam: the IVG page ---------------------------------------------------------
    T = CUES["title_slam"]
    ev.claim_near(sc, T, 0.15, ["title_slam", "slam", "page lands"])
    mx.add("sfx", L.page_slam(r, 1.3), T, 0.55, name="IVG page slam", scene=sc, cue="title_slam", send=0.08,
           verb="room")
    mx.add("sfx", L.sub_boom(r, 48, 27, 2.2, 0.3), T, 0.45, name="title sub", scene=sc, cue="title_slam")
    grid = ev.find(sc, ["grid"], all_=True, t0=15.9, t1=20.0)
    if grid:
        gts = []
        for e in grid:
            dd = dur_from_desc(e, 0.0)
            gts += list(e["t"] + np.linspace(0, dd, 11)) if dd > 0.1 else [e["t"]]
    else:
        gts = [16.05 + 0.09 * k for k in range(10)]
    for k, t in enumerate(gts):
        mx.add("sfx", tick_line(r, 0.1), t, 0.035, (k % 5 - 2) * 0.3, name="grid line draws", scene=sc)
    lets = ev.find(sc, ["letter"], all_=True, t0=16.0, t1=20.0)
    lts = [(e["t"], e["pan"]) for e in lets] or [(16.3 + 0.075 * k, -0.6 + 1.2 * k / 12) for k in range(13)]
    for k, (t, p) in enumerate(lts):
        mx.add("sfx", letter_thock(r, k), t, 0.16, p * 0.7, send=0.05, verb="room", name="letter lands", scene=sc)
    # --- page slides UP (hand-off) ----------------------------------------------------------------
    e = ev.find(sc, ["slides up", "slide"], near=20.4, win=0.5)
    t_s = e["t"] if e else 20.3
    dd = dur_from_desc(e, 0.5) if e else 0.7
    ps = L.paper_slide(r, dd + 0.15, 0.8)
    mx.add("sfx", Snd(ps.x, 0.0), t_s, 0.22, name="page slides up", scene=sc)
    mx.add("sfx", L.whoosh(r, dd + 0.4, dd * 0.7, 200, 1400, 0.8, 0.0, 0.0, dark=0.4), t_s + dd * 0.7, 0.12,
           name="page air", scene=sc)


def s02(mx, ev):
    sc = "s02"
    r = R(202)
    # --- ambience: the grey plain ---------------------------------------------------------------
    d = 27.0
    w = L.wind(r, d, level=0.5, gust=0.6, gust_rate=0.12, whistle=0.25, bright=0.35)
    w = shaped(w, 20.6, [(20.6, 0), (21.2, 1), (33.6, 1), (39.2, 0.85), (40.0, 0.35), (47.2, 0.3), (47.6, 0)])
    mx.add("amb", w, 20.6, 0.20, name="open plain wind (gusts + edge-tone whistles)", scene=sc)
    city = L.stereo(rumble(r, 18.6, 70), rumble(r, 18.6, 70)) * 0.8 + L.hum(r, 18.6, 55.0, (0.3, 0.4, 0.2, 0.1), 0.05)[:, None] * 0.05
    mx.add("amb", shaped(city, 21.0, [(21.0, 0), (22.5, 1), (33.6, 1), (39.2, 0.6), (39.6, 0)]), 21.0, 0.06,
           name="distant megastructure machinery", scene=sc)
    # --- 21-24.5 the monoliths rise -------------------------------------------------------------
    for k, (t, p) in enumerate([(21.1, -0.55), (21.8, 0.05), (22.5, 0.6)]):
        g = L.stone_slide(r, 2.8, 0.8)
        mx.add("sfx", g, t, 0.16, p, send=0.35, name="monolith rises (stone grind)", scene=sc)
        mx.add("sfx", L.creak(r, 2.5, 12, 20, (45, 68, 101, 150), 0.4, 0.5), t + 0.4, 0.08, p, send=0.4,
               name="megastructure groan", scene=sc)
    # --- N04: each memeplex lights on its word --------------------------------------------------
    for wname, p in (("Nations", -0.5), ("Faiths", 0.0), ("Philosophies", 0.5)):
        t = word("N04", wname)
        e = ev.find(sc, ["light", wname.lower()[:5]], near=t, win=0.6)
        t, p = (e["t"], e["pan"]) if e else (t, p)
        mx.add("sfx", Snd(L.relay_clunk(r, 1.6), 0.0), t, 0.30, p, send=0.5, name=f"floodlights on: {wname}", scene=sc)
        hm = L.hum(r, 3.0, 55.0, (0.5, 0.35, 0.3, 0.2, 0.15), 0.25) * L.env_pts(ns(3.0), [(0, 0), (0.08, 1), (1.2, 0.5), (3, 0)])
        mx.add("sfx", hm, t + 0.02, 0.05, p, send=0.4, name="floodlight hum swell", scene=sc)
    # --- lockstep march: on the picture's steps (Scene02), else the score's grid (95.238 BPM from 28.56) --
    mev = ev.find(sc, ["march", "lockstep"], all_=True, t0=28.0, t1=33.7)
    beats = [e["t"] for e in mev] or [28.56 + 0.63 * k for k in range(8)]
    x, tb = march(r, beats, 140, 0.010, 0.75)
    mx.add("amb", x, tb, 0.30, send=0.4, name=f"massed lockstep footsteps x{len(beats)} beats", scene=sc)
    # broadcast rings from the three antennas
    pev = ev.find(sc, ["pulse", "broadcast ring"], all_=True, t0=28.0, t1=33.7)
    pts_ = [(e["t"], e["pan"]) for e in pev] or [(b, 0.0) for b in beats[::2]]
    for k, (b, pc) in enumerate(pts_):
        for p in (-0.5, 0.0, 0.5):
            n = ns(0.6)
            tt_ = L.tt(n)
            f = 420 * np.exp(-tt_ / 0.12) + 140
            s = np.sin(L.TAU * np.cumsum(f) / SR) * L.env_ad(n, 0.004, 0.45)
            s += L.hp(L.poisson_clicks(r, n, 300 * np.exp(-tt_ / 0.1)), 2000) * 0.3
            mx.add("sfx", L.f32(s), b + 0.02 * (p + 0.5), 0.03, p, send=0.5, name="broadcast ring pulse", scene=sc)
    # --- 33.6 the slop: distant roar + notification pings multiplying into a roar ----------------
    T0, T1 = CUES["slop_appears"], CUES["slop_crash"]
    roar = L.ocean_roar(r, T1 - T0 + 0.2, 1.0, 0.15)
    tr = np.arange(len(roar)) / SR + T0
    g = (np.exp((tr - T0) / (T1 - T0) * 3.2) - 1) / (np.exp(3.2) - 1)
    roar = L.lp(roar, 900, 2) * (0.25 + 0.75 * g[:, None]) + L.hp(roar, 900) * g[:, None] ** 2
    mx.add("sfx", shaped(roar, T0, [(T0, 0), (T0 + 1.5, 1), (T1, 1), (T1 + 0.02, 0)]), T0, 0.45,
           name="slop wave roar (surf, foam, fizz)", scene=sc, cue="slop_appears")
    bank = ping_bank(r, [62, 65, 67, 69, 72, 74, 77], 60, 40)   # D minor, sickly detuned
    k_rate = np.log(1500.0) / (T1 - 34.0)
    t = 34.0
    evs = []
    while t < T1:
        lam = np.exp(k_rate * (t - 34.0))
        t += r.exponential(1.0 / lam)
        if t >= T1:
            break
        width = min(1.0, 0.3 + (t - 34.0) / 4.0)
        evs.append((t - 34.0, bank[r.integers(len(bank))], lam ** -0.32 * r.uniform(0.5, 1.0), r.uniform(-width, width)))
    pings = L.norm(L.scatter(ns(T1 - 34.0), evs), 0.9)
    mx.add("sfx", pings, 34.0, 0.30, send=0.25, name=f"notification pings x{len(evs)} (1/s -> 1500/s)", scene=sc)
    cards = L.stereo(L.svf(L.pink(ns(4.0), r), 2500, 0.8, "bp"), L.svf(L.pink(ns(4.0), r), 2500, 0.8, "bp"))
    cards *= (0.6 + 0.4 * np.abs(L.smooth_noise(ns(4.0), r, 14, 2)))[:, None]
    mx.add("sfx", shaped(cards, 35.2, [(35.2, 0), (39.2, 1), (39.21, 0)]), 35.2, 0.07,
           name="content-card flutter (thousands of paper cards)", scene=sc)
    sh = L.stereo(rumble(r, 3.0, 55), rumble(r, 3.0, 55))
    mx.add("sfx", shaped(sh, 36.2, [(36.2, 0), (39.2, 1), (39.21, 0)]), 36.2, 0.25, name="rising shake rumble",
           scene=sc)
    gl = ev.find(sc, ["glitch"], all_=True, t0=34.0, t1=38.95)
    gts = sorted([e["t"] for e in gl] + [37.0 + 1.8 * (1 - (1 - k / 7) ** 1.8) for k in range(7)])
    for k, t in enumerate(gts):
        if t > 38.95:
            continue
        mx.add("sfx", Snd(L.glitch(r, 0.05 + 0.02 * (k % 4)), 0.0), t, 0.05 + 0.1 * (t - 36) / 3, r.uniform(-0.8, 0.8),
               name="glitch / RGB-split burst", scene=sc)
    ev.claim_near(sc, 34.85, 0.2, ["roar"])
    e = ev.find(sc, ["lip", "fall"], near=38.8, win=0.3)
    tl0 = e["t"] if e else 38.8
    mx.add("sfx", L.whoosh(r, 0.6 + (T1 - tl0), T1 - tl0 - 0.05, 90, 1100, 0.7, 0.4, -0.2, dark=0.7, shape=1.6),
           T1 - 0.05, 0.30, send=0.3, name="the lip falls (vast air displacement)", scene=sc)
    e = ev.find(sc, ["full height", "swell"], near=38.5, win=0.3)
    if e:
        mx.add("sfx", L.swell(r, 1.2, 150, 2500, 1.5), e["t"], 0.12, name="the wave reaches full height", scene=sc)
    # --- 39.2 SLOP CRASH ---------------------------------------------------------------------------
    T = T1
    ev.claim_near(sc, T, 0.2, ["crash", "wave", "impact", "splash"])
    mx.add("sfx", L.sub_boom(r, 50, 24, 4.5, 0.45), T, 0.75, name="crash sub-boom (<60 Hz, Foley owns)", scene=sc,
           cue="slop_crash")
    big = splash(r, 3.0)
    mx.add("sfx", big, T, 0.70, send=0.3, name="the slop wave breaks (splash + bubble cloud)", scene=sc,
           cue="slop_crash")
    mx.add("sfx", Snd(L.shock(r, 1.5), 0.0), T, 0.4, name="crash shock", scene=sc, cue="slop_crash")
    mx.add("sfx", Snd(L.glitch(r, 0.35), 0.0), T, 0.18, name="pings die in a glitch", scene=sc)
    crk = ev.find(sc, ["crack"], all_=True, t0=T - 0.05, t1=T + 1.5)
    crs = [(e["t"], e["pan"], e["strength"]) for e in crk] or [(T + 0.04, -0.5, 0.9), (T + 0.1, 0.05, 0.8),
                                                               (T + 0.16, 0.55, 0.8)]
    for t, p, s in crs:
        mx.add("sfx", L.stone_crack(r, 0.08, 1.0, "stone", 0.8), t, 0.32 * s, p, send=0.4,
               name="memeplex cracks", scene=sc)
    tpl_ = ev.find(sc, ["topple"], all_=True, t0=T, t1=T + 2.0)
    tps = [(e["t"], e["pan"], e["strength"]) for e in tpl_] or [(T + 0.25, 0.35, 0.8), (T + 0.35, 0.75, 0.7)]
    for t, p, s in tps:
        mx.add("sfx", L.creak(r, 2.6, 8, 16, (38, 57, 85, 128), 0.5, 0.4), t, 0.22 * s, p, send=0.5,
               name="memeplex topples (groan)", scene=sc)
        mx.add("sfx", Snd(L.debris(r, 2.5, 40, "stone", 0.6), 0.0), t + 0.2, 0.22 * s, p, send=0.3,
               name="debris into slop", scene=sc)
        mx.add("sfx", L.whoosh(r, 1.2, 0.7, 80, 600, 0.8, p - 0.2, p + 0.2, dark=0.7), t + 0.7, 0.12, send=0.3,
               name="toppling mass through air", scene=sc)
    fl = ev.find(sc, ["flood"], near=T + 0.3, win=0.6)
    tf = fl["t"] if fl else T
    wash = L.ocean_roar(r, 6.0, 1.0, 0.3)
    mx.add("amb", shaped(wash, T, [(T, 0), (T + 0.1, 0.7), (tf + 0.3, 1), (T + 2.0, 0.6), (T + 6.0, 0)]), T, 0.35,
           name="beige flood rises (wash roar)", scene=sc)
    sk = ev.find(sc, ["sink", "gurgle"], near=39.9, win=0.8)
    if sk:
        mx.add("sfx", L.bubbles_cloud(r, 2.2, lambda u: 60 * np.exp(-u / 0.8), (7.0, 14.0), 1.0, damp_mul=0.6),
               sk["t"], 0.2 * sk["strength"], sk["pan"], send=0.3, name="memeplexes sink (viscous gurgle)", scene=sc)
    rn = ev.find(sc, ["rain", "cards rain"], near=40.0, win=0.8)
    if rn:
        n = ns(3.0)
        pat_ = L.poisson_clicks(r, n, 300 * np.exp(-L.tt(n) / 1.2), 1.0)
        pat_ = L.resonators(pat_, [900, 1700, 2900, 4300], [0.012] * 4, [1, 0.8, 0.5, 0.3]) + L.bp(pat_, 2000, 8000) * 0.3
        mx.add("sfx", L.decorrelate(L.norm(pat_, 0.9), r), rn["t"], 0.07, send=0.3,
               name="cards rain back down onto the flood (paper patter)", scene=sc)
    # --- 40.6 waist-deep in slop -------------------------------------------------------------------
    sl = L.stereo(slosh(r, 7.4, 0.3), slosh(r, 7.4, 0.26))
    mx.add("amb", shaped(sl, 40.2, [(40.2, 0), (41.0, 1), (47.0, 1), (47.5, 0)]), 40.2, 0.16,
           name="waist-deep slop lapping", scene=sc)
    mx.add("amb", L.bubbles_cloud(r, 7.0, 3.0, (5.0, 10.0), 1.0, damp_mul=0.6), 40.4, 0.06,
           name="slop gloops (large viscous bubbles)", scene=sc)
    # every screen its own feed: sparse detuned pings + faint buzz, polytonal clutter
    evs = [(r.uniform(0, 6.2), bank[r.integers(len(bank))], r.uniform(0.3, 1.0), r.uniform(-1, 1)) for _ in range(22)]
    mx.add("amb", L.scatter(ns(6.5), evs), 40.7, 0.05, send=0.2, name="scattered feed pings", scene=sc)
    for k in range(4):
        hb = L.hum(r, 6.3, 60 + 17 * k, (0.2, 0.4, 0.2, 0.3), 0.3) * (0.5 + 0.5 * L.smooth_noise(ns(6.3), r, 1.0, 2))
        mx.add("amb", L.fade(hb, 0.8, 0.3), 40.7, 0.006, -0.8 + 0.55 * k, name="screen buzz", scene=sc)
    # recursive split-screen: 1->4->16->64->256; pitch doubles per recursion level (sonified recursion)
    subs = ev.find(sc, ["subdiv", "split", "tile"], all_=True, t0=40.6, t1=47.2)
    sts = [e["t"] for e in subs] or [word("N07", "different"), word("N07", "face"), word("N07", "until"),
                                     word("N07", "people")]
    for k, t in enumerate(sts):
        lv = min(k, 4)
        mx.add("sfx", Snd(L.glitch(r, 0.04), 0.0), t, 0.07, 0.0, name=f"split x{4 ** (k + 1)}", scene=sc)
        mx.add("sfx", Snd(L.ping(r, float(midi_hz(62 + 12 * lv)), 0.25, 2), 0.0), t, 0.06, 0.0, send=0.3,
               name="recursion ping (octave per level)", scene=sc)
        mx.add("sfx", Snd(L.impact(r, "screen", 1.0 / (1 + lv), 1.0, 0.2), 0.0), t, 0.10, 0.0,
               name="panel snap", scene=sc)
    # --- 47.0-47.5 push into the screen: static (continues into s03, resolves 48.1) --------------
    st = L.tv_static(r, 1.3)
    st = shaped(st, 46.85, [(46.85, 0), (47.45, 1), (47.6, 1), (48.15, 0)])
    mx.add("sfx", st, 46.85, 0.16, name="TV static (push-in -> s03 resolve)", scene="s02")


def s03(mx, ev):
    sc = "s03"
    r = R(303)
    # --- ambience: flooded living room --------------------------------------------------------
    d = 14.2
    sl = L.stereo(slosh(r, d, 0.22), slosh(r, d, 0.19))
    rt = L.stereo(room_tone(r, d, 500), room_tone(r, d, 500))
    bed = sl * 0.8 + rt * 0.25
    mx.add("amb", shaped(bed, 47.4, [(47.4, 0), (47.9, 1), (61.3, 1), (61.5, 0)]), 47.4, 0.12,
           name="flooded room: viscous slop lapping + room tone", scene=sc, cut=61.5)
    mx.add("amb", L.bubbles_cloud(r, 13.8, 1.2, (5.0, 12.0), 1.0, damp_mul=0.7), 47.6, 0.06, send=0.4,
           name="slop gloops", scene=sc, cut=61.5)
    if not ev.find(sc, ["drip"], all_=True, claim=False):
        for k in range(12):
            t = 47.8 + r.uniform(0, 13.4)
            mx.add("amb", L.drip(r, 1.0), t, 0.07, r.uniform(-0.8, 0.8), send=0.5, name="drip from photo frames",
                   scene=sc, cut=61.5)
    # --- 47.5 tv_on ---------------------------------------------------------------------------------
    T = CUES["tv_on"]
    ev.claim_near(sc, T, 0.3, ["tv_on", "power", "click", "whine", "static", "resolve"])
    mx.add("sfx", L.crt_on(r), T, 0.40, send=0.25, name="CRT on: relay tick + degauss thunk + HV crackle",
           scene=sc, cue="tv_on")
    t_off = CUES["tv_on"] + 13.5  # 61.0
    e_off = ev.find(sc, ["collapse", "off", "line"], near=61.0, win=0.4)
    t_off = e_off["t"] if e_off else 61.0
    mx.add("sfx", L.crt_whine(r, t_off - T - 0.3, 1.0), T + 0.3, 0.006, name="15.734 kHz flyback whine + field hum",
           scene=sc, cut=t_off)
    spk = L.tvify(L.pink(ns(t_off - T), r) * 0.2)
    mx.add("sfx", L.fade(spk, 0.4, 0.05), T, 0.025, name="TV speaker hiss", scene=sc, cut=t_off)
    # --- broadcast sounds heard THROUGH the TV speaker -----------------------------------------------
    t_sl, p_sl, _ = ev.t(sc, ["slam", "fist", "banned"], word("J01", "banned"), 0.5)
    thump = L.thud(r, 85, 0.4, 1.0, 1500)
    thump[: ns(0.2)] += L.impact(r, "wood", 1.8, 1.0, 0.2)[: ns(0.2)] * 0.7
    mx.add("sfx", L.tvify(thump), t_sl, 0.35, name="fist slams podium (via TV speaker)", scene=sc)
    j = ev.find(sc, ["jolt", "ripple", "slosh", "splash"], near=t_sl + 0.05, win=0.6)
    mx.add("sfx", splash(r, 0.35), j["t"] if j else t_sl + 0.06, 0.12, send=0.3, name="TV jolts in the slop",
           scene=sc)
    T = CUES["pen_sign"]
    e0 = ev.find(sc, ["pen", "flourish", "sign"], near=T, win=0.6)
    e1 = ev.find(sc, ["pen", "flourish", "sign", "end"], near=T + 0.8, win=0.8)
    t0 = T
    t1 = e1["t"] if e1 and e1["t"] > t0 + 0.3 else T + 0.85
    mx.add("sfx", Snd(L.tvify(L.pen_scratch(r, t1 - t0, 6).x), 0.0), t0, 0.40, name="pen flourish (via TV)",
           scene=sc, cue="pen_sign")
    tw, pw, _ = ev.t(sc, ["wink"], 59.5, 1.0)
    mx.add("sfx", Snd(L.chime(r, float(midi_hz(93)), 1.2, (1.0, 2.76, 5.4), 0.6, 3, 1.0, 0.1), 0.0), tw, 0.05,
           0.3, send=0.2, name="wink 'ting'", scene=sc)
    ts, ps_, _ = ev.t(sc, ["coin slide", "slide", "coin"], 60.4, 0.6)
    n = ns(max(0.2, CUES["ka_ching"] - ts))
    slide = L.svf(L.white(n, r), 3000 + 2000 * L.tt(n) / (n / SR), 3, "bp") * L.env_pts(n, [(0, 0), (0.05, 0.5), (n / SR, 1)])
    mx.add("sfx", L.tvify(slide * 0.5), ts, 0.10, name="coin slides in", scene=sc)
    T = CUES["ka_ching"]
    ev.claim_near(sc, T, 0.25, ["coin", "ching", "pocket", "land"])
    kc = L.ka_ching(r)
    mx.add("sfx", Snd(L.tvify(kc.x, small=False), kc.sync), T, 0.40, name="KA-CHING (register via TV)", scene=sc,
           cue="ka_ching")
    mx.add("sfx", kc, T, 0.10, 0.25, name="KA-CHING dry bell sheen", scene=sc, cue="ka_ching")
    mx.add("sfx", L.sparkle(r, 1.0, 10, octave=(2, 3)), T + 0.02, 0.05, 0.3, send=0.2, name="coin sparkle", scene=sc)
    tp, pp, _ = ev.t(sc, ["pat"], 61.05, 0.5)
    if tp <= t_off + 0.05:
        mx.add("sfx", L.tvify(pat(r).x), min(tp, t_off - 0.06), 0.18, name="pats the pocket (via TV)", scene=sc)
    # --- 61.0 CRT off: collapse to a line -> dot -> black ---------------------------------------------
    ev.claim_near(sc, t_off, 0.5, ["collapse", "dot", "black", "off"])
    mx.add("sfx", L.crt_off(r), t_off, 0.42, send=0.25, name="CRT off: HV collapse zap + cooling ticks", scene=sc,
           cut=61.5)


def s04(mx, ev):
    sc = "s04"
    r = R(404)
    CUT = 90.0
    d = 28.6
    rb = L.stereo(rumble(r, d, 60), rumble(r, d, 60))
    hm = L.hum(r, d, 55.0, (0.5, 0.25, 0.12, 0.05), 0.0)
    ah = L.stereo(room_tone(r, d, 700), room_tone(r, d, 700))
    bed = rb * 0.8 + hm[:, None] * 0.08 + ah * 0.2
    mx.add("amb", shaped(bed, 61.5, [(61.5, 0), (62.6, 1), (72.6, 1), (72.9, 0.6), (76.0, 0.8), (90, 1)]), 61.5, 0.045,
           name="vast underground hum (A1 55 Hz) + air handling", scene=sc, cut=CUT)
    cp = L.cryo_pulse(r, 27.2, 0.72, 1.0)
    cp = shaped(cp, 62.8, [(62.8, 0), (64.0, 1), (72.6, 1), (72.8, 0.35), (76.5, 0.35), (78.5, 0.8), (90, 0.8)])
    mx.add("amb", L.pan2(cp, 0.15), 62.8, 0.10, send=0.3, name="Hypermind cryo pulse-tube breathing", scene=sc,
           cut=CUT)
    # the Hypermind's power draw swells with its own voice (lip-sync envelope from audio/voice/mouth.npz)
    try:
        z = np.load(ROOT / "audio" / "voice" / "mouth.npz")
        msr = float(z["sr"])
        mouth = np.asarray(z["HYPERMIND"], np.float64)
        d2 = 10.5
        n = ns(d2)
        ta = np.arange(n) / SR + 62.7
        env = np.interp(ta, np.arange(len(mouth)) / msr, mouth)
        env = L.lp(L.f32(env), 12, 1)
        ph = L.hum(r, d2, 110.0, (0.6, 0.4, 0.3, 0.2, 0.15, 0.1), 0.35)
        ph = L.lp(ph, 1200, 2) * (0.15 + 0.85 * np.clip(env, 0, 1))
        mx.add("sfx", L.fade(ph, 0.3, 0.4), 62.7, 0.05, send=0.3, name="Hypermind power draw (driven by its voice)",
               scene=sc, cut=CUT)
    except Exception as e:  # mouth file missing
        print("[s04] no mouth envelope:", e)
    fs = L.fsk_chirps(r, 27.5, 0.8)
    fs = shaped(fs, 62.5, [(62.5, 0), (63.5, 1), (72.6, 1), (72.8, 0.2), (77, 0.2), (79, 1), (90, 1)])
    mx.add("amb", L.pan2(fs, -0.4), 62.5, 0.012, send=0.4, name="acolyte console data chatter", scene=sc, cut=CUT)
    for k in range(9):
        mx.add("amb", L.drip(r, 1.3), r.uniform(62.5, 89), 0.05, r.uniform(-0.9, 0.9), send=0.8,
               name="distant cavern drip", scene=sc, cut=CUT)
    # --- 61.5 descent ------------------------------------------------------------------------------
    T = CUES["descent"]
    e = ev.find(sc, ["plunge", "shaft", "descent"], near=T, win=0.4)
    mx.add("sfx", L.whoosh(r, 1.7, 1.0, 120, 900, 0.8, 0.0, 0.0, dark=0.6), (e["t"] if e else T) + 1.0, 0.25, send=0.4,
           name="plunging down the pentagonal shaft (air)", scene=sc, cue=None)
    tl_, pl, _ = ev.t(sc, ["land", "arriv"], 63.0, 0.8)
    mx.add("sfx", Snd(L.thud(r, 45, 1.0, 0.8, 300), 0.0), tl_, 0.20, send=0.5, name="arrive in the Command Center",
           scene=sc)
    # --- H01: scripture spirals up (incense-like paper whisper), six canons absorbed -----------------
    n = ns(7.8)
    tt_ = L.tt(n)
    wh = L.svf(L.pink(n, r), 1500 + 2500 * (tt_ / 7.8), 1.0, "bp") * (0.5 + 0.5 * L.smooth_noise(n, r, 3, 2)) ** 2
    mx.add("sfx", L.fade(L.decorrelate(wh, r), 1.0, 1.0), 63.0, 0.05, send=0.5, name="scripture streams (paper whisper)",
           scene=sc)
    ab = ev.find(sc, ["canon", "scroll", "absorb", "ingest"], all_=True, t0=63.0, t1=72.0)
    abt = [(e["t"], e["pan"]) for e in ab][:6] or [(65.1 + 0.36 * k, (-1) ** k * 0.4) for k in range(6)]
    for k, (t, p) in enumerate(abt):
        mx.add("sfx", L.swell(r, 0.4, 600, 7000, 2.0), t, 0.06, p, send=0.4, name=f"canon {k + 1} inhaled", scene=sc)
        mx.add("sfx", Snd(L.chime(r, float(midi_hz([74, 76, 78, 81, 83, 86][k % 6])), 3.0, t60=2.0, beat=1.2,
                                  strike=0.3), 0.0), t, 0.05, p, send=0.5, name=f"canon {k + 1} absorbed (glass)",
               scene=sc)
    tk = ev.find(sc, ["tick"], all_=True, t0=63.0, t1=72.0, kind_only=True)
    tks = [e["t"] for e in tk] or list(np.arange(69.25, 70.6, 0.075))
    for t in tks:
        mx.add("sfx", Snd(L.impact(r, "tin", 0.35, 0.6, 0.05), 0.0), t, 0.03, -0.3, name="terminal readout tick",
               scene=sc)
    # --- H02: energy descends the cold stages -> beam -> 72.7 hypermeme ---------------------------------
    st = ev.find(sc, ["stage", "tier", "energy", "descend"], all_=True, t0=71.0, t1=72.6)
    stt = [e["t"] for e in st] or list(np.linspace(71.3, 72.25, 6))
    for k, t in enumerate(stt[:8]):
        m = [98, 93, 90, 86, 81, 74, 69, 62][k]
        mx.add("sfx", Snd(L.chime(r, float(midi_hz(m)), 2.0, t60=1.2, beat=2, strike=0.5), 0.0), t, 0.05, 0.0,
               send=0.5, name="cold stage lights (descending D)", scene=sc)
        mx.add("sfx", L.zap(r, 0.18, 2500, 400), t, 0.04, 0.0, send=0.3, name="stage discharge", scene=sc)
    tb, _, _ = ev.t(sc, ["beam"], 72.3, 0.5)
    ev.claim_near(sc, tb, 0.3, ["beam", "pedestal"])
    T = CUES["hypermeme"]
    mx.add("sfx", L.fade(L.plasma_buzz(r, T - tb + 0.1, 110), 0.05, 0.08), tb, 0.06, send=0.4, name="beam hits pedestal",
           scene=sc)
    ev.claim_near(sc, T, 0.3, ["material", "hypermeme", "flash", "appear"])
    mx.add("sfx", L.swell(r, 0.7, 500, 12000, 2.5), T, 0.18, send=0.5, name="hypermeme materializes (air swell)",
           scene=sc, cue="hypermeme")
    ch = np.zeros(ns(6), np.float32)
    for k, m in enumerate([62, 69, 74, 78]):
        c = L.chime(r, float(midi_hz(m + 12)), 6.0, t60=4.0, beat=0.7, strike=0.4)
        ch[: len(c)] += c * (1 - 0.1 * k)
    mx.add("sfx", ch, T, 0.09, send=0.6, name="hypermeme chord (glass D)", scene=sc, cue="hypermeme")
    mx.add("sfx", L.sub_boom(r, 42, 30, 2.0, 0.4), T, 0.35, name="hypermeme sub", scene=sc, cue="hypermeme")
    mx.add("sfx", L.sparkle(r, 2.0, 20), T + 0.02, 0.05, send=0.5, name="materialize sparkle", scene=sc)
    br = L.wind(r, CUT - T, level=0.25, gust=0.5, gust_rate=0.3, whistle=0.0, bright=0.7)
    mx.add("amb", shaped(L.pan2(br, 0.1), T, [(T, 0), (T + 1.5, 1), (CUT, 1)]), T, 0.12,
           name="the impossible breeze (underground)", scene=sc, cut=CUT)
    # --- C02: Crocus lifts the flag; 'Summon the schismmancers': pole strike, seal, klaxons ------------
    tlf, plf, _ = ev.t(sc, ["lift", "raise"], 84.0, 1.5)
    mx.add("sfx", L.cloth_rustle(r, 0.8, 1.2), tlf, 0.12, plf, send=0.2, name="Crocus lifts the hypermeme", scene=sc)
    tps, pps, _ = ev.t(sc, ["pole strike", "strike", "staff", "pole"], 87.85, 0.4)
    mx.add("sfx", L.pole_thunk(r), tps, 0.45, pps, send=0.5, name="pole strikes the floor", scene=sc)
    mx.add("sfx", L.sub_boom(r, 45, 30, 1.6, 0.3), tps, 0.25, name="strike sub", scene=sc)
    tu, pu, _ = ev.t(sc, ["unlock", "seal unlock", "latch"], 88.0, 0.5)
    for k in range(3):
        mx.add("sfx", Snd(L.relay_clunk(r, 1.4), 0.0), tu + 0.11 * k, 0.16, (k - 1) * 0.5, send=0.5,
               name="seal latches release", scene=sc)
    to, po, _ = ev.t(sc, ["seal open", "open", "floor seal"], 88.3, 0.6)
    mx.add("sfx", L.stone_slide(r, CUT - to - 0.1, 1.0), to, 0.30, po, send=0.5, name="pentagonal floor seal grinds open",
           scene=sc, cut=CUT)
    hs = L.swell(r, CUT - to, 1000, 9000, 1.2)
    mx.add("sfx", Snd(hs.x, 0.0), to, 0.10, send=0.3, name="pressure hiss", scene=sc, cut=CUT)
    kl = ev.find(sc, ["klaxon", "sweep", "alarm"], all_=True, t0=86.0, t1=90.0)
    klt = [(e["t"], e["pan"]) for e in kl] or [(88.45 + 0.75 * k, (-1) ** k * 0.5) for k in range(3)]
    for t, p in klt:
        mx.add("sfx", L.klaxon_whoop(r, 0.9), t, 0.12, p, send=0.6, name="gold klaxon sweep (D3 horn)", scene=sc,
               cut=CUT)
    tf, _, _ = ev.t(sc, ["flash", "push"], 89.7, 0.4)
    mx.add("sfx", L.swell(r, CUT - tf + 0.02, 400, 12000, 2.0), CUT, 0.20, name="flash to cut", scene=sc, cut=CUT)


def s05a(mx, ev):
    sc = "s05a"
    r = R(505)
    d = 10.6
    rb = L.stereo(rumble(r, d, 80), rumble(r, d, 80))
    ah = L.stereo(room_tone(r, d, 1400), room_tone(r, d, 1400))
    bed = shaped(rb * 0.8 + ah * 0.2, 90.0, [(90.0, 0), (90.05, 1), (99.2, 1), (99.65, 0.15), (99.7, 0.15),
                                             (100.2, 0.0)])
    mx.add("amb", bed, 90.0, 0.045, name="dim tunnel: rumble + air handling", scene=sc)
    for k in range(8):
        mx.add("amb", L.drip(r, 1.1), r.uniform(91.0, 99.0), 0.06, r.uniform(-0.8, 0.8), send=0.7,
               name="tunnel drip (flutter echo)", scene=sc)
    # --- 90.0 gear up (Scene05a exports every close-up; fallback montage) -------------------------------
    if not ev.find(sc, ["velcro", "hammer", "click", "stomp"], all_=True, claim=False, t0=90.0, t1=91.3):
        mx.add("sfx", L.velcro(r, 0.25), 90.0, 0.30, 0.1, name="velcro rip", scene=sc, cue="gear_up")
        mx.add("sfx", L.hammer_tap(r, 2), 90.3, 0.35, -0.1, name="tack hammer", scene=sc)
        mx.add("sfx", L.telescope(r, 3, 0.09), 90.5, 0.30, 0.0, name="telescoping pole clicks", scene=sc)
        mx.add("sfx", Snd(L.impact(r, "tin", 0.5, 1.0, 0.15), 0.0), 90.83, 0.2, name="NVG locks", scene=sc)
        mx.add("sfx", nvg_whine(r), 90.84, 0.12, name="NVG whine", scene=sc)
        mx.add("sfx", L.boot_step(r, "concrete", 1.3, True), 91.04, 0.35, -0.45, name="boot stomp", scene=sc)
    # telescoping clicks: each 'click' event on a pole gets a slide-in before it (the section travelling)
    for e in ev.find(sc, ["telescop"], all_=True, claim=False, t0=90.0, t1=91.3):
        m = ns(0.07)
        sl = L.svf(L.white(m, r), 2800, 2.5, "bp") * L.env_pts(m, [(0, 0), (0.05, 0.6), (0.07, 1)])
        mx.add("sfx", sl, e["t"] - 0.07, 0.08, e["pan"], name="pole section slides", scene=sc)
    # --- 91.2-98.4 slow-motion hero walk: heavy, slowed boots ----------------------------------------
    if not ev.find(sc, ["step", "foot", "boot"], all_=True, claim=False, t0=91.2, t1=98.4):
        for w_ in range(5):
            p = -0.6 + 0.3 * w_
            t = 91.5 + 0.17 * w_
            while t < 98.2:
                s = L.boot_step(r, "concrete", 1.2, True).x
                s = L.f32(np.interp(np.arange(int(len(s) / 0.72)) * 0.72, np.arange(len(s)), s))  # slow-mo: 0.72x
                mx.add("sfx", s, t, 0.10 + 0.04 * (w_ == 2), p, send=0.4, name="slow-motion boots", scene=sc)
                t += 1.05 + r.uniform(-0.03, 0.03)
    else:  # slow the exported steps too: deeper, heavier
        for e in ev.find(sc, ["step", "foot", "boot"], all_=True, t0=91.2, t1=98.4):
            s = L.boot_step(r, "concrete", 1.2, True).x
            s = L.f32(np.interp(np.arange(int(len(s) / 0.72)) * 0.72, np.arange(len(s)), s))
            mx.add("sfx", s, e["t"], 0.07 + 0.08 * e["strength"], e["pan"], send=0.4, name="slow-motion boot",
                   scene=sc, src="event:" + e["file"])
    # --- 96.1 HUD globe subdivides -------------------------------------------------------------------
    hud = ev.find(sc, ["hud", "holo", "globe", "subdiv", "voronoi"], all_=True, t0=95.5, t1=98.4)
    hts = [(e["t"], e["pan"]) for e in hud] or [(96.3 + 0.22 * k, 0.2) for k in range(5)]
    mx.add("sfx", L.pan2(L.fsk_chirps(r, 1.6, 9.0), 0.2), hts[0][0] - 0.1, 0.02, name="HUD data chatter", scene=sc)
    for k, (t, p) in enumerate(hts):
        mx.add("sfx", Snd(L.ping(r, float(midi_hz(74 + 2 * k)), 0.14, 2), 0.0), t, 0.04, p, send=0.3,
               name="HUD region subdivides", scene=sc)
    # --- K03: stack-up; each 'Breach!' arms a charge (Composer owns the stabs) -------------------------
    for k in range(3):
        t = word("K03", "Breach", k)
        e = ev.find(sc, ["breach", "charge", "stack"], near=t, win=0.3)
        t = e["t"] if e else t
        mx.add("sfx", Snd(L.ping(r, float(midi_hz(98)), 0.09, 1), 0.0), t + 0.12, 0.035, 0.1,
               name="charge arming beep", scene=sc)
        mx.add("sfx", L.cloth_rustle(r, 0.25, 1.4), t, 0.08, -0.2, name="stack-up gear shuffle", scene=sc)
    # --- 99.7 BREACH ---------------------------------------------------------------------------------
    T = CUES["breach"]
    ev.claim_near(sc, T, 0.25, ["door", "breach", "blow", "explos", "blast", "light"])
    mx.add("sfx", L.explosion(r, 1.4, 5.0, "stone", 90), T, 0.70, send=0.35, name="BREACH: door blown in", scene=sc,
           cue="breach")
    mx.add("sfx", L.sub_boom(r, 55, 24, 3.5, 0.35), T, 0.45, name="breach sub", scene=sc, cue="breach")
    mx.add("sfx", Snd(L.impact(r, "metal", 3.5, 1.0), 0.0), T + 0.02, 0.25, send=0.4, name="door slab ring", scene=sc)
    sw = L.swell(r, 0.8, 2000, 14000, 0.6)
    mx.add("sfx", Snd(L.fade(sw.x[::-1].copy(), 0.001, 0.3), 0.0), T + 0.1, 0.10,
           name="blinding flag-light shimmer", scene=sc)
    n = ns(1.4)
    shim = L.svf(L.white(n, r), 7000 + 3000 * L.smooth_noise(n, r, 2, 2), 6, "bp") * L.env_pts(n, [(0, 0), (0.4, 1), (1.4, 0)])
    mx.add("sfx", L.decorrelate(shim, r), T + 0.2, 0.05, name="glow shimmer into s05b", scene=sc)


def map_crack(r, level):
    """recursive border crack: micro-fracture burst whose pitch rises with recursion depth."""
    k = 2 ** min(level, 5)
    dur = 0.18
    n = ns(dur)
    exc = L.poisson_clicks(r, n, 60 * k, 1.0) + 0.0
    exc[0] += 1.0
    fb = 500 * 1.32 ** level
    x = L.resonators(exc, [fb, fb * 1.63, fb * 2.41, fb * 3.7], [0.02, 0.015, 0.01, 0.008], [1, 0.7, 0.5, 0.3])
    x += L.bp(exc, 2000, 9000) * 0.5
    return Snd(L.fade(L.norm(x, 0.8), 0.0005, 0.03), 0.0)


def s05b(mx, ev):
    sc = "s05b"
    r = R(606)
    # --- map ambience ----------------------------------------------------------------------------------
    w = L.wind(r, 7.6, level=0.3, gust=0.4, whistle=0.0, bright=0.5)
    mx.add("amb", shaped(w, 100.3, [(100.3, 0), (100.9, 1), (107.5, 1), (107.9, 0)]), 100.3, 0.18,
           name="high aerial wind over the map", scene=sc)
    sh = L.shepard_rise(r, 2.9, 7, 70, 0.9, 0.3)
    mx.add("amb", shaped(L.decorrelate(sh, r), 100.6, [(100.6, 0), (101.0, 1), (103.0, 1), (103.5, 0)]), 100.6, 0.05,
           send=0.3, name="endless-zoom spectral Shepard rise (noise)", scene=sc)
    for k in range(3):
        mx.add("amb", L.bird(r, "chip", 60), r.uniform(104.0, 107.3), 0.03, r.uniform(-0.8, 0.8), send=0.4,
               name="tiny birds over a doormat nation", scene=sc)
    # 100.6 nation_fracture: the flag is planted at the Nation's centre (shock ring), then recursive cracks
    T = CUES["nation_fracture"]
    tpl, ppl, _ = ev.t(sc, ["plant"], T, 0.3)
    mx.add("sfx", L.pole_thunk(r), tpl, 0.30, ppl, send=0.3, name="flag planted at the Nation's centre", scene=sc,
           cue="nation_fracture" if abs(tpl - T) < 0.02 else None)
    mx.add("sfx", L.sub_boom(r, 46, 28, 1.6, 0.3), tpl, 0.25, name="plant shock (sub)", scene=sc,
           cue="nation_fracture" if abs(tpl - T) < 0.02 else None)
    cr = sorted(ev.find(sc, ["crack", "subdiv", "border"], all_=True, t0=100.55, t1=107.8, kind_only=False),
                key=lambda e: e["t"])
    cts = [(e["t"], e["pan"], e["strength"]) for e in cr] or [(T + 0.17 + 0.3 * k, r.uniform(-0.6, 0.6), 1.0)
                                                              for k in range(8)]
    for k, (t, p, s) in enumerate(cts):
        lv = k
        mx.add("sfx", map_crack(r, lv), t, 0.22 * (0.85 ** min(lv, 6)) * (0.5 + 0.5 * s), p, send=0.3,
               name=f"border generation {lv + 1} cracks (pitch rises with recursion depth)", scene=sc)
        mx.add("sfx", L.stone_crack(r, 0.12, 1.0, "stone", 1.0 + 0.15 * lv, final_bang=False).x, t, 0.08 * 0.85 ** lv,
               p, send=0.3, name="crack front races out", scene=sc)
        mx.add("sfx", L.debris(r, 0.4, 3 + lv, "stone", 0.15, (0.05, 0.3), size=(0.2, 0.5)), t + 0.02,
               0.05, p, name="border grit", scene=sc)
    # --- temple ----------------------------------------------------------------------------------------
    rt = L.stereo(room_tone(r, 7.0, 350), room_tone(r, 7.0, 350))
    cand = L.hp(L.poisson_clicks(r, ns(7.0), 18), 1500)
    mx.add("amb", shaped(rt * 0.9 + L.decorrelate(cand, r) * 0.1, 107.6, [(107.6, 0), (107.8, 1), (114.4, 1), (114.7, 0)]),
           107.6, 0.05, send=0.3, name="temple hush + candle crackle", scene=sc)
    T = CUES["faith_fracture"]
    ev.claim_near(sc, T, 0.2, ["plant", "aisle"])
    mx.add("sfx", L.pole_thunk(r), T, 0.35, 0.0, send=0.6, name="flag planted in the aisle", scene=sc,
           cue="faith_fracture")
    fc = ev.find(sc, ["floor crack", "crack"], near=T + 0.4, win=0.6)
    tfc = fc["t"] if fc else T + 0.35
    mx.add("sfx", L.stone_crack(r, 0.45, 1.0, "stone", 0.9), tfc, 0.40, 0.0, send=0.6, name="temple floor cracks",
           scene=sc)
    sp = ev.find(sc, ["circle", "split", "cabal"], all_=True, t0=T + 0.3, t1=114.5)
    spt = [(e["t"], e["pan"]) for e in sp] or [(108.5, -0.3), (109.0, 0.3), (109.35, -0.5), (109.6, 0.5)]
    for t, p in spt:
        mx.add("sfx", L.stone_crack(r, 0.12, 1.0, "stone", 1.1, final_bang=True), t, 0.18, p, send=0.6,
               name="circle splits", scene=sc)
        mx.add("sfx", L.stone_slide(r, 0.5, 0.4), t + 0.05, 0.08, p, send=0.5, name="stone circle grinds apart",
               scene=sc)
    for k in range(7):
        n = ns(0.35)
        m = L.svf(L.white(n, r), 2500 * np.exp(-L.tt(n) / 0.1) + 600, 1.5, "bp") * L.env_ad(n, 0.004, 0.25)
        mx.add("sfx", m, r.uniform(110.2, 112.6), 0.035, r.uniform(-0.9, 0.9), send=0.5,
               name="each tiny altar's candle strikes alight", scene=sc)
    # --- void --------------------------------------------------------------------------------------------
    T = CUES["philo_shatter"]
    ev.claim_near(sc, T, 0.25, ["shatter", "head", "fracture"])
    pl = ev.find(sc, ["plant", "crown"], near=T - 0.3, win=0.6)
    if pl and pl["t"] < T - 0.05:
        mx.add("sfx", L.pole_thunk(r), pl["t"], 0.3, pl["pan"], send=0.3, name="flag planted on the marble crown",
               scene=sc)
    pre = L.stone_crack(r, 0.3, 1.0, "marble", 1.2, final_bang=False)
    mx.add("sfx", Snd(pre.x, 0.3), T, 0.22, send=0.3, name="marble stresses", scene=sc)
    mx.add("sfx", L.shatter(r, "marble", 1.6, 300, 5.0, gravity=False), T, 0.75, send=0.45,
           name="colossal marble head shatters into the void", scene=sc, cue="philo_shatter")
    mx.add("sfx", L.sub_boom(r, 48, 25, 3.0, 0.4), T, 0.55, name="shatter sub", scene=sc, cue="philo_shatter")
    d = 11.0
    n = ns(d)
    vd = L.stereo(L.svf(L.brown(n, r), 120 + 80 * L.smooth_noise(n, r, 0.1, 2), 0.8, "lp"),
                  L.svf(L.brown(n, r), 120 + 80 * L.smooth_noise(n, r, 0.1, 2), 0.8, "lp"))
    ph = L.stereo(L.svf(L.pink(n, r), 600 * 2 ** (1.5 * L.smooth_noise(n, r, 0.07, 1)), 5, "bp"),
                  L.svf(L.pink(n, r), 600 * 2 ** (1.5 * L.smooth_noise(n, r, 0.07, 1)), 5, "bp"))
    mx.add("amb", shaped(vd * 0.8 + ph * 0.15, 114.6, [(114.6, 0), (115.2, 1), (123.3, 1), (125.5, 1.4), (125.6, 0)]),
           114.6, 0.06, send=0.2, name="the void (dark air, slow resonant drift)", scene=sc)
    for k in range(6):
        t = 116.0 + 1.5 * k + r.uniform(0, 0.8)
        mx.add("sfx", L.whoosh(r, 2.2, 1.1, 90, 700, 0.8, r.uniform(-1, 0), r.uniform(0, 1), dark=0.7), t, 0.07,
               send=0.6, name="fragment drifts past", scene=sc)
    evs = [(r.uniform(0, 9.5), L.impact(r, "marble", 0.5, 1.0), r.uniform(0.2, 1.0), r.uniform(-1, 1)) for _ in range(20)]
    mx.add("sfx", L.scatter(ns(10.0), evs), 115.5, 0.03, send=0.8, name="fragments tick against each other", scene=sc)
    # X07 echo rings
    t0 = LINES["X07"]["end"]
    rings = ev.find(sc, ["ring", "echo"], all_=True, t0=120.5, t1=123.3)
    rts = [(e["t"], e["pan"]) for e in rings] or [(121.2, 0.0), (121.75, 0.0), (122.4, 0.0)]
    for k, (t, p) in enumerate(rts):
        n = ns(0.9)
        tt_ = L.tt(n)
        f = 260 * np.exp(-tt_ / 0.25) + 70
        s = np.sin(L.TAU * np.cumsum(f) / SR) * L.env_ad(n, 0.01, 0.7) * 0.6
        s += L.svf(L.pink(n, r), 400 + 1400 * np.exp(-tt_ / 0.2), 1.0, "bp") * L.env_ad(n, 0.005, 0.5) * 0.5
        mx.add("sfx", L.f32(s), t, 0.05 * 0.8 ** k, p, send=0.7, name="echo ring expands", scene=sc)
    # K07 pull back: the galaxy of fragments swirls -> storm (hand-off)
    n = ns(2.6)
    tt_ = L.tt(n)
    sw = L.svf(L.pink(n, r), 300 + 1600 * (tt_ / 2.6) ** 2, 1.0, "bp") * (tt_ / 2.6) ** 1.5
    mx.add("sfx", L.pan2(sw, 0.7 * np.sin(L.TAU * 0.9 * tt_ ** 1.4)), 123.0, 0.14, send=0.4,
           name="galaxy swirl (fragments spiral)", scene=sc)


def s06(mx, ev):
    sc = "s06"
    r = R(707)
    d = 13.2
    n = ns(d)
    ta = L.tt(n) + 125.3
    speed = np.interp(ta, [125.3, 127, 130, 134, 136, 138.5], [0.7, 0.8, 0.85, 0.9, 1.15, 1.3])
    speed = speed * (1 + 0.25 * L.smooth_noise(n, r, 0.3, 3))
    w = L.wind(r, d, whistle=0.6, bright=0.55, speed=speed)
    mx.add("amb", shaped(w, 125.3, [(125.3, 0), (125.8, 1), (138.5, 1)]), 125.3, 0.22,
           name="storm gale (howl + edge-tone whistles)", scene=sc)
    eg = L.tv_static(r, 11.5, 1.0)
    eg = L.lp(eg, 5000, 2) * (0.5 + 0.5 * np.abs(L.smooth_noise(ns(11.5), r, 5, 2)))[:, None]
    mx.add("amb", shaped(eg, 127.0, [(127.0, 0), (129, 0.4), (136, 0.8), (138.5, 1.0)]), 127.0, 0.05, send=0.2,
           name="Lord Egregore's static", scene=sc)
    rb = L.stereo(rumble(r, d, 55), rumble(r, d, 55))
    mx.add("amb", shaped(rb, 125.3, [(125.3, 0), (126, 0.6), (135.9, 0.7), (136.2, 1.3), (138.5, 1.6)]), 125.3, 0.12,
           name="tower rumble", scene=sc)
    for k in range(4):
        t = 126.5 + 3.0 * k + r.uniform(0, 1.5)
        mx.add("sfx", L.creak(r, 2.5, 10, 22, (40, 61, 92, 137, 205), 0.5, 0.6), t, 0.10, r.uniform(-0.7, 0.7),
               send=0.4, name="tower groans", scene=sc)
    evs = [(r.uniform(0, 12.5), L.impact(r, "stone", 0.4, 1.0), r.uniform(0.2, 1.0), r.uniform(-1, 1)) for _ in range(40)]
    mx.add("amb", L.scatter(ns(13.0), evs), 125.5, 0.05, send=0.3, name="debris trickle", scene=sc)
    # --- 125.5 babel_rise: fragments spiral and stack ---------------------------------------------------
    T = CUES["babel_rise"]
    ev.claim_near(sc, T, 0.3, ["swirl", "helix"])
    mx.add("sfx", L.whoosh(r, 2.4, 0.9, 150, 1600, 0.8, -0.8, 0.8, dark=0.5), T + 0.4, 0.30, send=0.4,
           name="the swirl spirals into the tower", scene=sc, cue="babel_rise")
    stk = ev.find(sc, ["stack", "thunk", "block", "land"], all_=True, t0=125.4, t1=130.0)
    sts = [(e["t"], e["pan"], e["strength"]) for e in stk] or [
        (125.9 + 3.4 * (1 - (1 - k / 10) ** 1.6), r.uniform(-0.6, 0.6), 1.0 - 0.05 * k) for k in range(10)]
    for t, p, s in sts:
        x = L.thud(r, 55, 0.6, 1.2, 600)
        x[: ns(0.3)] += L.impact(r, "stone", 2.5, 1.0, 0.3)[: ns(0.3)] * 0.6
        mx.add("sfx", L.f32(x), t, 0.28 * (0.5 + 0.5 * s), p, send=0.4, name="fragment stacks onto the tower", scene=sc)
        mx.add("sfx", L.debris(r, 1.2, 8, "stone", 0.2), t + 0.03, 0.08, p, name="stack debris", scene=sc)
    # --- lightning (thunder physically synthesized from a tortuous channel) ------------------------------
    lt = ev.find(sc, ["lightning", "thunder", "strike"], all_=True, t0=125.5, t1=138.5)
    lts = [(e["t"], e["pan"], e["strength"]) for e in lt] or [(127.2, -0.6, 0.3), (129.9, 0.5, 0.55),
                                                              (word("C03", "Good") - 0.05, -0.2, 0.9),
                                                              (135.4, 0.6, 0.7), (137.3, -0.3, 1.0)]
    for t, p, c in lts:
        th = L.thunder(r, float(np.clip(c, 0, 1)))
        delay = 0.04 + (1 - c) * 1.0
        mx.add("sfx", Snd(th.x, 0.0), t + delay, 0.15 + 0.3 * c, p, send=0.25, name=f"thunder (closeness {c:.2f})",
               scene=sc)
        if c > 0.5:
            z = L.zap(r, 0.25, 6000, 1500)
            mx.add("sfx", z, t, 0.10 * c, p, name="lightning sizzle", scene=sc)
    # --- Lutie's robes whip in the gale (cloth flutter engine on a synthetic gale energy) ----------------
    e_n = int((133.2 - 129.8) * 24)
    en = 0.55 + 0.2 * np.abs(frame_turb(r, e_n, 1.0))
    x = FF.flutter_from_energy(en, 24, -0.25, 0.6, [], seed=66, size=0.8)
    mx.add("sfx", L.lp(x, 3500, 1), 129.8, 0.18, name="robes whipping (flutter engine, gale energy)", scene=sc)
    # --- 136.0 babel_crack --------------------------------------------------------------------------------
    T = CUES["babel_crack"]
    ev.claim_near(sc, T, 0.2, ["babel_crack", "crack"])
    mx.add("sfx", L.stone_crack(r, 0.08, 1.0, "stone", 0.8), T, 0.65, 0.0, send=0.4, name="THE TOWER CRACKS",
           scene=sc, cue="babel_crack")
    mx.add("sfx", Snd(L.shock(r, 1.2), 0.0), T, 0.45, name="tower crack shock", scene=sc, cue="babel_crack")
    mx.add("sfx", L.sub_boom(r, 50, 24, 3.0, 0.5), T, 0.6, name="crack sub", scene=sc, cue="babel_crack")
    cc = ev.find(sc, ["crack", "cascade"], all_=True, t0=T + 0.05, t1=138.5)
    ccs = [(e["t"], e["pan"], e["strength"]) for e in cc] or [
        (T + 0.15 + 2.2 * (k / 14) ** 0.8, r.uniform(-0.7, 0.7), 0.7) for k in range(14)]
    for k, (t, p, s) in enumerate(ccs):
        mx.add("sfx", L.stone_crack(r, 0.12, 1.0, "stone", 0.9 + 0.08 * k, final_bang=True), t, 0.2 * (0.5 + s), p,
               send=0.4, name="crack races up the tower", scene=sc)
    gl = ev.find(sc, ["glitch"], all_=True, t0=T - 0.5, t1=138.5)
    gts = [(e["t"], e["pan"]) for e in gl] or [(T + 0.1 + 2.35 * (k / 9) ** 0.9, r.uniform(-0.9, 0.9)) for k in range(9)]
    for t, p in gts:
        mx.add("sfx", Snd(L.glitch(r, 0.1 + 0.05 * r.random()), 0.0), t, 0.13, p, name="Egregore screens glitch",
               scene=sc)
        mx.add("sfx", L.tv_static(r, 0.25) * L.env_ad(ns(0.25), 0.002, 0.2)[:, None], t, 0.06, p,
               name="static burst", scene=sc)
    for k in range(5):
        t = T + 0.3 + 0.45 * k
        mx.add("sfx", L.whoosh(r, 0.7, 0.35, 300, 3000, 1.2, r.uniform(-1, 0), r.uniform(0, 1)), t, 0.15, send=0.3,
               name="fragment flies past", scene=sc)
    mx.add("sfx", L.creak(r, 2.5, 8, 30, (36, 55, 82, 124, 186), 0.6, 0.8), T + 0.1, 0.30, 0.0, send=0.4,
           name="the tower's final groan", scene=sc)
    mx.add("sfx", L.debris(r, 2.6, 120, "stone", 1.6), T + 0.1, 0.30, send=0.3, name="max chaos debris", scene=sc)


def s07a(mx, ev):
    sc = "s07a"
    r = R(808)
    d = 15.0
    n = ns(d)
    ta = L.tt(n) + SIL1
    speed = np.interp(ta, [SIL1, 142.0, 142.3, 146, 150, 154.5], [0.18, 0.2, 0.35, 0.3, 0.32, 0.45])
    w = L.wind(r, d, whistle=0.1, bright=0.35, speed=speed * (1 + 0.3 * L.smooth_noise(n, r, 0.15, 3)))
    mx.add("amb", shaped(w, SIL1, [(SIL1, 0), (140.6, 0.55), (142.0, 0.8), (143.0, 1.0), (154.6, 1.0), (155.5, 0)]),
           SIL1, 0.22,
           name="faint dawn wind (the first sound after the silence)", scene=sc)
    t = 143.2
    while t < 154.3:
        mx.add("amb", L.bird(r, None, r.uniform(60, 160)), t, r.uniform(0.02, 0.05), r.uniform(-0.9, 0.9), send=0.5,
               name="distant dawn bird", scene=sc)
        t += r.exponential(1.1)
    # climb
    if not ev.find(sc, ["step", "climb", "scrape", "foot"], all_=True, claim=False, t0=140.0, t1=142.0):
        for k, t in enumerate([140.7, 141.05, 141.42, 141.72]):
            mx.add("sfx", L.boot_step(r, "rubble", 0.7, False), t, 0.12, 0.05, send=0.3, name="Lutie climbs the rubble",
                   scene=sc)
    # 142.0 flag_planted
    T = CUES["flag_planted"]
    ev.claim_near(sc, T, 0.2, ["plant", "thunk"])
    mx.add("sfx", L.pole_thunk(r), T, 0.62, 0.0, send=0.35, name="THUNK: the Flag is planted", scene=sc,
           cue="flag_planted")
    mx.add("sfx", L.sub_boom(r, 44, 28, 2.0, 0.3), T, 0.50, name="plant sub", scene=sc, cue="flag_planted")
    dp, _, _ = ev.t(sc, ["dust"], T + 0.02, 0.3)
    mx.add("sfx", L.decorrelate(L.dust_puff(r, 1.2, 0.5), r), dp, 0.22, send=0.3, name="dust ring", scene=sc)
    mx.add("sfx", Snd(L.debris(r, 1.2, 14, "stone", 0.3, (0.05, 0.5), size=(0.3, 0.8)), 0.0), T + 0.02, 0.12,
           name="pebbles settle", scene=sc)
    tg, pg, _ = ev.t(sc, ["gust", "whoosh"], T + 0.1, 0.4)
    mx.add("sfx", L.whoosh(r, 1.3, 0.5, 200, 2200, 0.9, -0.5, 0.5, dark=0.3), tg, 0.20, send=0.3,
           name="first dawn gust", scene=sc)
    # 148.5 survivors turn
    if not ev.find(sc, ["turn", "rustle"], all_=True, claim=False, t0=148.0, t1=150.0):
        for k in range(18):
            mx.add("sfx", L.cloth_rustle(r, 0.4, 0.7, 0.7), r.uniform(148.5, 149.4), 0.03, r.uniform(-0.9, 0.9),
                   send=0.5, name="survivors turn to look", scene=sc)
    # 150.0 convergence: the phase-transition wave of raised flags
    T = CUES["convergence"]
    crowd = [tr for tr in FF.load_tracks() if tr["scene"] == sc and tr["kind"] == "crowd"]
    evr = ev.find(sc, ["raise"], all_=True, t0=T - 0.2, t1=154.6)
    raises = []
    for e in evr:
        raises.append((e["t"], e["pan"], 0.1, e["strength"]))
    if crowd:
        tr = crowd[0]
        rr = R(42)
        fps = tr["fps"]
        t0 = tr["f0"] / fps
        en = np.clip(tr["energy"], 0, None)
        cum = 0.0
        for i, e_ in enumerate(en):
            k = rr.poisson(e_ * 40)
            cum += k
            for _ in range(k):
                prog = (t0 + i / fps - T) / 4.5
                raises.append((t0 + (i + rr.random()) / fps, float(np.clip(tr["pan"][i] + rr.normal(0, 0.35), -1, 1)),
                               float(np.clip(prog, 0, 1)), 0.5 + 0.5 * rr.random()))
    if len(raises) < 50:
        raises += [tuple(x) for x in FF.procedural_wave(T, 4.3, 2400, seed=150)]
    x, t0 = FF.crowd_wave(T - 0.1, 154.6 - T + 0.1, raises, True, 5, 0.25)
    mx.add("sfx", x, t0, 0.35, send=0.3, name=f"CONVERGENCE: {len(raises)} flags raise (grains + massed flutter bed)",
           scene=sc, cue="convergence")
    mx.add("sfx", L.swell(r, 4.3, 150, 3000, 1.6), 154.4, 0.10, send=0.3, name="crane-up air", scene=sc)


def s07b(mx, ev):
    sc = "s07b"
    r = R(909)
    d = 18.8
    n = ns(d)
    w = L.wind(r, d, level=0.35, gust=0.5, gust_rate=0.2, whistle=0.35, bright=0.85)
    w = shaped(w, 154.3, [(154.3, 0), (155.0, 1), (170.9, 1), (173.0, 1.3), (173.3, 0)])
    mx.add("amb", w, 154.3, 0.14, name="high sky wind (thin, whistling)", scene=sc)
    fb = L.stereo(L.svf(L.pink(ns(4.0), r), 1800, 0.7, "bp"), L.svf(L.pink(ns(4.0), r), 1800, 0.7, "bp"))
    fb *= (0.7 + 0.3 * np.abs(L.smooth_noise(ns(4.0), r, 8, 2)))[:, None]
    mx.add("amb", shaped(fb, 154.5, [(154.5, 1), (158.5, 0)]), 154.5, 0.05,
           name="flag-carpet flutter bed falls away below", scene=sc)
    # every glint is chosen from the hymn chord sounding at that instant (no clashes with the choir)
    evs = []
    for _ in range(40):
        tt_ = r.uniform(0, 18)
        m = 84 + int(r.choice(hymn_chord(154.5 + tt_))) + 12 * r.integers(0, 2)
        evs.append((tt_, L.chime(r, float(midi_hz(m)), 2.0, (1.0, 2.76, 5.4), 1.2, 3, 0.6, 0.02), r.uniform(0.2, 1.0),
                    r.uniform(-1, 1)))
    mx.add("amb", L.scatter(ns(19.5), evs), 154.5, 0.025, send=0.6, name="high shimmer glints (hymn-chord tones)",
           scene=sc)
    # phrase accents (Composer: timpani on these; Foley: air bloom + chord glints + the tiling's flags ripple)
    for k, t in enumerate([154.5, 157.36, 160.21, 163.07]):
        big = k == 3
        ev.claim_near(sc, t, 0.1, ["swell", "bloom", "ring", "ripple", "flags be", "flags are", "flags will"])
        mx.add("sfx", L.whoosh(r, 1.4, 0.18, 100, 900, 0.7, -0.3, 0.3, dark=0.6, shape=1.2), t, 0.16 + 0.12 * big,
               send=0.4, name="phrase accent: air bloom", scene=sc, cue="hymn" if k == 0 else None)
        cnt = 30 if big else 12
        gl = [(r.uniform(0, 1.6 if big else 0.9) ** 1.5, chord_glint(r, t + 0.05, 6 + (i % 2), 2.0, 1.2).x,
               r.uniform(0.3, 1.0), r.uniform(-0.9, 0.9)) for i in range(cnt)]
        mx.add("sfx", L.scatter(ns(3.5), gl), t, 0.035 + 0.02 * big, send=0.6, name="phrase glints (chord tones)",
               scene=sc)
        rg = [(t + r.uniform(0, 0.5), r.uniform(-0.9, 0.9), 0.7, 0.4) for _ in range(10 + 20 * big)]
        x, t0 = FF.crowd_wave(t, 1.5, rg, False, 20 + k)
        mx.add("sfx", x, t0, 0.18, send=0.4, name="tiling flags ripple", scene=sc)
    # 165.9 embers lift off
    te, _, _ = ev.t(sc, ["ember", "lift"], 165.9, 0.6)
    mx.add("sfx", ember_rise(r, 2.6, 30), te, 0.10, send=0.6, name="flags ascend like embers", scene=sc)
    # 166.3 gateless gate: the 10,000-foot Flag
    T = CUES["gateless_gate"]
    ev.claim_near(sc, T, 0.3, ["gate", "colossal", "rise"])
    mx.add("sfx", L.whoosh(r, 3.2, 1.0, 60, 700, 0.6, 0.0, 0.0, dark=0.8, shape=1.5), T, 0.35, send=0.4,
           name="the colossal Flag rises through the clouds", scene=sc, cue="gateless_gate")
    mx.add("sfx", L.sub_boom(r, 38, 26, 3.0, 0.8), T, 0.30, name="colossal presence (sub)", scene=sc,
           cue="gateless_gate")
    mx.add("sfx", flash_swell(r, 0.8), T, 0.08, send=0.5, name="the Gate ignites (light)", scene=sc)
    tg, _, _ = ev.t(sc, ["pass", "through"], 168.6, 1.5)
    mx.add("sfx", L.whoosh(r, 3.0, 1.6, 80, 1400, 0.7, 0.0, 0.0, dark=0.6, shape=1.3), tg, 0.22, send=0.5,
           name="passing through the Gateless Gate", scene=sc)
    n = ns(6.5)
    gr = L.svf(L.white(n, r), 5500 + 2500 * L.smooth_noise(n, r, 0.4, 2), 8, "bp")
    mx.add("sfx", shaped(L.decorrelate(gr, r), T, [(T, 0), (T + 1.5, 1), (T + 6.5, 0)]), T, 0.03, send=0.6,
           name="god-ray shimmer", scene=sc)
    # 170.9 -> full white-gold (Scene07b: full frame at ~172.6) -> hand-off 173.0
    t_w, _, _ = ev.t(sc, ["flash", "full-frame"], 172.6, 0.5)
    ev.claim_near(sc, 170.9, 0.2, ["swell", "bloom"])
    sw = L.swell(r, t_w - 170.9, 300, 14000, 2.2)
    mx.add("sfx", sw, t_w, 0.25, send=0.3, name="bloom to white-gold", scene=sc)
    n = ns(173.4 - t_w)
    hold = L.svf(L.pink(n, r), 9000, 0.7, "hp") * L.env_pts(n, [(0, 1), (173.0 - t_w, 0.8), (n / SR, 0)])
    mx.add("sfx", L.decorrelate(hold, r), t_w, 0.06, send=0.4, name="white-gold hold (air)", scene=sc)
    gl = [(r.uniform(0, t_w - 170.9), chord_glint(r, 171.0, 6 + i % 3, 1.8, 1.0).x, r.uniform(0.3, 1.0),
           r.uniform(-1, 1)) for i in range(40)]
    mx.add("sfx", L.scatter(ns(4.0), gl), 170.9, 0.05, send=0.6, name="white-gold glitter (D chord)", scene=sc)


def s08(mx, ev):
    sc = "s08"
    r = R(1010)
    d = 20.3
    n = ns(d)
    ta = L.tt(n) + 172.8
    speed = 0.22 * (1 + 0.35 * L.smooth_noise(n, r, 0.12, 3))
    w = L.wind(r, d, whistle=0.0, bright=0.8, speed=speed)
    grass = L.stereo(L.hp(L.pink(n, r), 2500), L.hp(L.pink(n, r), 2500)) * (speed[:, None] * 0.25)
    bed = shaped(w + grass, 172.8, [(172.8, 0), (173.5, 1), (190.4, 1), (193.0, 0)])
    mx.add("amb", bed, 172.8, 0.20, name="morning meadow: wind in grass", scene=sc)
    ff = L.stereo(L.svf(L.pink(n, r), 1500, 0.8, "bp"), L.svf(L.pink(n, r), 1500, 0.8, "bp"))
    ff *= (0.6 + 0.4 * np.abs(L.smooth_noise(n, r, 6, 2)))[:, None] * speed[:, None] * 3
    mx.add("amb", shaped(L.lp(ff, 3000, 2), 172.8, [(172.8, 0), (173.8, 1), (190.4, 1), (193.0, 0)]), 172.8, 0.05,
           name="a field of countless distant flags", scene=sc)
    t = 173.6
    while t < 190.5:
        mx.add("amb", L.bird(r, None, r.uniform(25, 90)), t, r.uniform(0.02, 0.06), r.uniform(-0.9, 0.9), send=0.4,
               name="meadow songbird", scene=sc)
        t += r.exponential(0.9)
    # Lutie runs up (fallback)
    if not ev.find(sc, ["step", "run", "foot"], all_=True, claim=False, t0=173.0, t1=174.2):
        for k in range(6):
            mx.add("sfx", L.boot_step(r, "grass", 0.6, False), 173.25 + 0.14 * k, 0.12, -0.5 + 0.08 * k,
                   name="Lutie runs up (grass)", scene=sc)
    if not ev.find(sc, ["offer"], all_=True, claim=False):
        mx.add("sfx", L.cloth_rustle(r, 0.5, 1.0), 174.6, 0.12, -0.1, name="offers the flag", scene=sc)
    # --- the sticky koan: touch -> peel -> strand stretches (to the snap) -> snap & recoil ---------------
    e = ev.find(sc, ["touch"], near=176.5, win=1.0)
    t_to, p_to = (e["t"], e["pan"]) if e else (176.4, 0.05)
    mx.add("sfx", L.squelch_touch(r), t_to, 0.16, p_to, name="fingertip meets the tacky pole", scene=sc)
    e = ev.find(sc, ["peel", "squelch"], near=177.0, win=0.8)
    t_pe, p_pe = (e["t"], e["pan"]) if e else (t_to + 0.55, p_to)
    mx.add("sfx", sticky_peel(r), t_pe, 0.22, p_pe, name="finger peels off the pole (tacky suction)", scene=sc)
    e = ev.find(sc, ["stretch"], near=177.45, win=0.8)
    t_st, p_st = (e["t"], e["pan"]) if e else (t_pe + 0.45, p_pe)
    e = ev.find(sc, ["snap", "recoil"], near=178.9, win=1.2)
    t_sn, p_sn = (e["t"], e["pan"]) if e else (t_st + 0.9, p_st)
    stc = L.squelch_stretch(r, max(0.3, t_sn - t_st + 0.02))
    mx.add("sfx", stc, t_st, 0.20, p_st, name="the glistening strand stretches, thins, beads (rubbery stick-slip)",
           scene=sc)
    mx.add("sfx", Snd(L.bubble(r, 0.8, 0.9, 1.0, 0.35), 0.0), t_sn, 0.14, p_sn, name="strand snaps: wet 'plik'",
           scene=sc)
    mx.add("sfx", L.whoosh(r, 0.25, 0.06, 800, 3000, 2.0, p_sn, p_sn - 0.2), t_sn + 0.01, 0.04,
           name="strand recoils", scene=sc)
    if not ev.find(sc, ["take", "back", "return"], all_=True, claim=False, t0=182.0, t1=185.0):
        mx.add("sfx", L.cloth_rustle(r, 0.5, 1.0), 183.95, 0.12, 0.0, name="hands it back", scene=sc)
    if not ev.find(sc, ["sniff"], all_=True, claim=False):
        mx.add("sfx", L.sniff(r, 2), 184.35, 0.16, -0.15, name="Lutie sniffs her hand", scene=sc)
    if not ev.find(sc, ["walk", "step"], all_=True, claim=False, t0=185.0, t1=188.3):
        for k in range(9):
            g = 0.10 * (1 - k / 11)
            mx.add("sfx", L.lp(L.boot_step(r, "grass", 0.5, False).x, 3000 - 200 * k, 1), 185.3 + 0.33 * k, g,
                   0.1 * (-1) ** k, name="the two walk off into the field", scene=sc)
    # 188.3 GLORIOUS: the salute wave
    T = CUES["glorious"]
    import re
    crowd = [tr for tr in FF.load_tracks() if tr["scene"] == sc and tr["kind"] == "crowd"]
    evr = ev.find(sc, ["salute", "raise"], all_=True, t0=T - 0.3, t1=DUR)
    rr = R(43)
    raises = []
    for e in evr:
        m = re.search(r"(\d+) people", e["desc"])
        cnt = min(int(m.group(1)), 500) if m else 1
        md = re.search(r"(\d+)-(\d+) m", e["desc"])
        mid = (float(md.group(1)) + float(md.group(2))) / 2 if md else 5.0
        dist = float(np.clip(np.log1p(mid) / np.log1p(3000), 0, 1))
        spread = 0.08 + 0.25 * dist
        for _ in range(cnt):
            raises.append((e["t"] + rr.uniform(-spread, spread) * (cnt > 1), float(np.clip(e["pan"] + rr.normal(0, 0.3), -1, 1)),
                           dist, e["strength"] * rr.uniform(0.6, 1.0) / (1 + 0.004 * cnt)))
    if crowd:
        tr = crowd[0]
        fps = tr["fps"]
        t0 = tr["f0"] / fps
        for i, e_ in enumerate(np.clip(tr["energy"], 0, None)):
            for _ in range(rr.poisson(e_ * 40)):
                prog = (t0 + i / fps - T) / 1.2
                raises.append((t0 + (i + rr.random()) / fps, float(np.clip(tr["pan"][i] + rr.normal(0, 0.4), -1, 1)),
                               float(np.clip(prog, 0, 1)), 0.5 + 0.5 * rr.random()))
    if len(raises) < 50:
        raises += [tuple(x) for x in FF.procedural_wave(T, 1.2, 2600, seed=188, jitter=0.05)]
    x, t0 = FF.crowd_wave(T - 0.3, DUR - T + 0.3, raises, True, 9, 0.35)
    mx.add("sfx", x, t0, 0.40, send=0.3, name=f"GLORIOUS salute wave: {len(raises)} flags snap up", scene=sc,
           cue="glorious")
    mx.add("sfx", Snd(FF.snap_crack(r, 1.0), 0.0), T, 0.35, -0.1, name="Lutie's flag snaps up (salute)", scene=sc,
           cue="glorious")
    e = ev.find(sc, ["gust"], near=T + 0.3, win=0.6)
    tgu = e["t"] if e else T + 0.3
    mx.add("sfx", L.whoosh(r, 1.6, 0.35, 150, 2500, 0.8, -0.6, 0.6, dark=0.3), tgu, 0.30, send=0.3,
           name="a gust sweeps the field (ten thousand flags)", scene=sc)
    sn = [(rr.uniform(0, 0.7), FF.snap_crack(rr, rr.uniform(0.3, 0.8)), rr.uniform(0.2, 1.0), rr.uniform(-1, 1))
          for _ in range(40)]
    mx.add("sfx", L.lp(L.scatter(ns(1.0), sn), 5000, 1), tgu - 0.1, 0.12, send=0.4, name="distant flags snap in the gust",
           scene=sc)
    # 189.5 end card
    T = CUES["end_card"]
    ev.claim_near(sc, T, 0.2, ["end card", "card", "slam", "page"])
    mx.add("sfx", L.page_slam(r, 1.2), T, 0.45, name="IVG end-card page", scene=sc, cue="end_card", send=0.1,
           verb="room")
    mx.add("sfx", L.sub_boom(r, 46, 28, 2.0, 0.3), T, 0.30, name="end-card sub", scene=sc, cue="end_card")
    lets = ev.find(sc, ["letter"], all_=True, t0=T, t1=192.0)
    lts = [(e["t"], e["pan"]) for e in lets] or [(190.0 + 0.13 * k, -0.4 + 0.2 * k) for k in range(5)]
    for t, p in lts:
        mx.add("sfx", letter_thock(r), t, 0.20, p * 0.7, send=0.05, verb="room", name="F-L-A-G-S lands", scene=sc)
    grid = ev.find(sc, ["grid", "line"], all_=True, t0=T, t1=192.0)
    for e in grid:
        mx.add("sfx", tick_line(r, 0.1), e["t"], 0.03, e["pan"], name="grid line", scene=sc)


# ------------------------------------------------------------------------------------------------
# physics-derived flutter for every exported flag track (+ fallbacks for scenes without tracks)
# ------------------------------------------------------------------------------------------------
FALLBACK_FLAGS = {
    # scene: (t0, t1, [(t, energy)], pan, gain, snaps[(t, s)], size)
    "s01": (13.6, 16.0, [(13.6, 0.03), (14.1, 0.05), (14.2, 0.95), (14.5, 0.55), (15.2, 0.45), (16.0, 0.4)], 0.05, 0.9,
            [(14.2, 1.0), (14.75, 0.5), (15.4, 0.4)], 1.0),
    "s04": (72.7, 90.0, [(72.7, 0.0), (73.4, 0.18), (80, 0.22), (83.8, 0.2), (84.3, 0.4), (86, 0.3), (90, 0.3)], 0.1,
            0.7, [(84.3, 0.5)], 1.0),
    "s05a": (99.8, 100.5, [(99.8, 0.3), (99.95, 0.9), (100.5, 0.8)], 0.0, 0.8, [(99.95, 0.8), (100.2, 0.6)], 1.0),
    "s06": (130.0, 138.5, [(130, 0.35), (134, 0.4), (136, 0.6), (138.5, 0.7)], 0.3, 0.6, [(136.05, 0.7)], 1.0),
    "s07a": (142.0, 154.5, [(142.0, 0.05), (142.15, 0.85), (143, 0.45), (148, 0.35), (154.5, 0.45)], 0.0, 1.0,
             [(142.15, 1.0), (142.6, 0.6)], 1.0),
    "s07b": (166.3, 173.0, [(166.3, 0.1), (167.3, 0.6), (170, 0.7), (173, 0.8)], 0.0, 1.0, [(167.4, 1.0), (169.8, 0.8)],
             6.0),
    "s08": (173.3, 188.0, [(173.3, 0.4), (174.0, 0.3), (176, 0.15), (183.9, 0.3), (185, 0.2), (188, 0.1)], -0.1, 0.6,
            [], 1.0),
}


def physics(mx, report):
    tracks = [t for t in FF.load_tracks() if t["kind"] != "crowd"]
    have = {t["scene"] for t in tracks}
    for tr in tracks:
        if not mx.active(tr["scene"]):
            continue
        x, t0, info = FF.track_audio(tr)
        corr, rms = FF.correlation(tr, x)
        report.append(dict(file=tr["file"], scene=tr["scene"], name=tr["name"], frames=len(tr["energy"]),
                           t0=t0, energy_max=float(tr["energy"].max()), snaps=info["n_events"],
                           flap_hz_from_sim="flap_hz" in tr, size=info["size"], corr_energy_rms=corr))
        base = 0.3 if info.get("field") else 0.55
        mx.add("sfx", x, t0, base, send=0.25, name=f"PHYSICS flutter <- {tr['file']} (r={corr:.3f})",
               scene=tr["scene"], src="physics:" + tr["file"])
    for sc, (t0, t1, pts, pan, gain, snaps, size) in FALLBACK_FLAGS.items():
        if sc in have or not mx.active(sc):
            continue
        nfr = int((t1 - t0) * 24)
        tf = t0 + np.arange(nfr) / 24
        r = R(zlib.crc32(sc.encode()) % 999)
        en = np.interp(tf, [p[0] for p in pts], [p[1] for p in pts])
        en = en * (1 + 0.35 * frame_turb(r, nfr))   # per-frame turbulence
        en = np.clip(en, 0, None)
        x = FF.flutter_from_energy(en, 24, pan, gain, [(t, s, pan) for t, s in snaps], seed=11, t0=t0, size=size)
        corr = float(np.corrcoef(en, FF.frame_rms(x, 24, nfr))[0, 1])
        report.append(dict(file=f"(fallback {sc})", scene=sc, name="fallback", frames=nfr, t0=t0,
                           energy_max=float(en.max()), snaps=len(snaps), flap_hz_from_sim=False, size=size,
                           corr_energy_rms=corr))
        mx.add("sfx", x, t0, 0.5, send=0.25, name=f"flutter (fallback energy, awaiting {sc} track)", scene=sc,
               src="fallback-physics")


# ------------------------------------------------------------------------------------------------
# master
# ------------------------------------------------------------------------------------------------
def limiter(x, ceiling=0.891):
    """transparent peak limiter (1 ms look-ahead min filter + 80 ms release); only touches peaks > ceiling."""
    from scipy.ndimage import minimum_filter1d
    a = np.abs(x).max(1)
    g = np.minimum(1.0, ceiling / np.maximum(a, 1e-9))
    if g.min() >= 1.0:
        return x, 1.0
    g = minimum_filter1d(g, 97)
    rel = np.exp(-1.0 / (0.08 * SR))
    g2 = _release(g.astype(np.float64), rel)
    g2 = minimum_filter1d(g2, 49)
    return (x * g2[:, None]).astype(np.float32), float(g2.min())


def _release(g, rel):
    from numba import njit

    @njit(cache=False)
    def run(g, rel):
        out = np.empty_like(g)
        s = 1.0
        for i in range(len(g)):
            s = g[i] if g[i] < s else g[i] + (s - g[i]) * rel
            out[i] = s
        return out
    return run(g, rel)


def finalize(mx):
    for stem in ("sfx", "amb"):
        b = mx.bus[stem]
        np.nan_to_num(b, copy=False)
        i0, i1 = int(round(SIL0 * SR)), int(round(SIL1 * SR))
        b[i0:i1] = 0.0                       # sacred silence (placements are already cut at 138.5)
        if stem == "sfx":
            b[i0:int(round(140.0 * SR))] = 0.0
        b[NTOT - ns(0.02):] *= np.linspace(1, 0, ns(0.02))[:, None]
        if stem == "sfx":   # the film fades to black by 193.0: everything with it
            a0 = int(round(191.0 * SR))
            b[a0:] *= np.linspace(1, 0, NTOT - a0)[:, None] ** 2
    lims = {}
    hot = []
    for stem in ("sfx", "amb"):   # diagnostics: the hottest pre-limiter moments and what plays there
        m = np.abs(mx.bus[stem]).max(1)
        blk = SR // 10
        pk = m[: len(m) // blk * blk].reshape(-1, blk).max(1)
        for i in np.argsort(pk)[::-1][:12]:
            if pk[i] < 0.891:
                break
            t = i / 10
            who = sorted([e for e in mx.log if e["stem"] == stem and e["start"] - 0.05 <= t + 0.1 and e["t"] + 3 >= t],
                         key=lambda e: -e["peak"])[:4]
            hot.append(dict(stem=stem, t=t, peak=round(float(pk[i]), 3),
                            who=[f"{e['name'][:40]}@{e['t']:.2f}({e['peak']:.2f})" for e in who]))
    (OUTD / "hot.json").write_text(json.dumps(hot, indent=1))
    for stem in ("sfx", "amb"):
        mx.bus[stem], gmin = limiter(mx.bus[stem])
        lims[stem] = gmin
        i0, i1 = int(round(SIL0 * SR)), int(round(SIL1 * SR))
        mx.bus[stem][i0:i1] = 0.0
    return lims


CATALOGUE = """## How it is made (everything synthesized at 48 kHz — no samples)

| model | physics / method | used for |
|---|---|---|
| cloth flutter (tools/foley_flutter.py) | per-sample parameters interpolated from the scene's cloth simulation: flap pulse train at the sim's `flap_hz` (or Strouhal-rate from energy), band centre ∝ energy, rustle ∝ E^1.35, 'fwap' on every frame-resolved acceleration, whip-crack N-wave on every sim snap, pan from the cloth's screen x | every hero flag, flag fields, the colossal Flag (size-scaled: slower, lower billows) |
| crowd flag waves | one grain per raised flag (swing-up swish + 3-5 decaying flaps + pop) placed at its own time/pan/distance (air absorption), over a bed ∝ √(flags raised) | s07a convergence, s07b tiling ripples, s08 GLORIOUS salute |
| wind | brown+pink turbulence through a per-sample TPT state-variable low-pass tracking wind speed + edge-tone whistles (narrow band-passes, f ∝ U) with gust fbm, decorrelated L/R | every exterior bed, the impossible breeze underground |
| thunder | 3-D tortuous lightning channel (random walk, ~9 m segments); each segment's N-wave arrives at d/343 s with 1/d spreading, broadside radiation, distance-binned air absorption | s06 lightning (closeness = scene strength) |
| bubbles / drips / slop | van den Doel Minnaert bubbles (f0 = 3.26/r, pitch rising as the bubble surfaces), viscous = larger/damped | drips, slop gloops, the crash splash cloud, the sticky 'plik' |
| fracture & debris | accelerating Poisson micro-crack train through stone/marble modal bodies + final split; fragments fall and bounce with restitution (t_k, v_k = e^k v0) and ring with material modes (stone, marble, glass, metal, wood, screen) | memeplex collapse, recursive borders (pitch rises with recursion depth), temple, marble head (void: no gravity, fragments keep ringing), Babel |
| explosion | N-wave shock + fireball roar (brown noise, closing low-pass) + sub + bouncing debris | the breach |
| stick-slip friction | impulse train at a load-dependent slip rate through resonators | rubbed-glass voice of the rotating Crystal (4-D beating), tower/monolith groans, stone doors, pen nib, sticky strand squeak |
| CRT | relay tick, degauss 60 Hz field + 120 Hz rattle, HV crackle, 15.734 kHz flyback whine; collapse zap (falling chirp) + cooling ticks; TV-speaker filter for everything inside the broadcast | s02→s03 static, s03 TV, Egregore's screens |
| modal glass / bells | inharmonic mode sets, detuned pairs (beating), pitched only to the score's key (D major; hymn-chord-aware in s07b) | crystal, glints, canon absorptions, cold stages, ka-ching |
| others | velcro (hook-row release trains), telescoping clicks, tack hammer, boots (heel/crunch/gear), cryo pulse-tube breathing, FSK data chatter, notification pings (1/s → 1500/s, detuned D minor), paper slides/slams, klaxon (D3), squelch/peel/sniff, birds (FM song models), wing bursts, candles, crowns | per scene |

Mixer: procedural impulse responses per space (space, plain+monolith echoes, flooded room, cavern, tunnel flutter echo,
map, temple, void, storm, dawn, sky, meadow); pre-impact 'breath' ducking before big cues so every transient lands clean;
hard cuts at 90.0 (s04→s05a) and 138.5 (silence) apply to reverb tails too; transparent peak limiter at −1 dBFS.
"""


def write_cues_md(mx, report, lims, path):
    lines = ["# Foley cue sheet — THE UNBABELING (generated by tools/foley_build.py)", "",
             "Stems: `audio/stems/sfx.wav` (hard effects + physics-derived flag flutter) and `audio/stems/ambience.wav` "
             "(beds). 48 kHz stereo float32, exactly 193.0 s, T=0 aligned. Rebuild: "
             "`source .venv/bin/activate && python tools/foley_build.py && python tools/foley_check.py --spec` "
             "(re-reads every scene export in cache/foley).", "",
             "**Silence:** both stems are digital zero 138.500–139.600 (every placement that starts before 138.5 is "
             "hard-cut at 138.5, reverb tails included); `sfx` stays zero to 140.0; the only sound from 139.6 is the "
             "faint dawn wind in `ambience`.", "",
             "**Band split with Composer:** Foley owns sub-booms (<60 Hz) and noise/debris transients on big cues; "
             "the score keeps tonal hits ≥ 55 Hz and owns the K03 stabs + 99.2→99.7 reverse-suck and the 15.3→16.0 suck "
             "(Foley adds only faint air). Every tonal Foley element is pitched in the score's key (D major; s02 slop "
             "pings D minor, detuned; s07b glints follow the hymn chord).", "",
             "**Dialogue FX:** radio key-clicks/squelch on COMMANDER lines live in dialogue.wav (Vocalist), not here.", "",
             f"Limiter min gain: sfx {lims.get('sfx', 1):.3f}, ambience {lims.get('amb', 1):.3f} (1.0 = untouched).", "",
             CATALOGUE,
             "## Physics-derived flag flutter (sound computed from the cloth simulation)", "",
             "| track | scene | frames | start s | max energy | snaps | flap_hz from sim | r(energy, flutter RMS) |",
             "|---|---|---:|---:|---:|---:|:-:|---:|"]
    for p in report:
        lines.append(f"| {p['file']} | {p['scene']} | {p['frames']} | {p['t0']:.2f} | {p['energy_max']:.2f} | "
                     f"{p['snaps']} | {'yes' if p['flap_hz_from_sim'] else 'no'} | {p['corr_energy_rms']:.3f} |")
    lines += ["", "Plot: `audio/foley/physics_tracks.png` (sim energy vs synthesized flutter RMS, snap events marked).",
              "", "## Placements by scene", "",
              "Repeated sounds within a scene are collapsed into one row (count, first…last time). `source` = design "
              "(Foley's cue design), event:<file> (a scene-exported hit), physics:<file> (cloth sim).", ""]
    by = {}
    for e in mx.log:
        by.setdefault(e["scene"], []).append(e)
    for s in TL["scenes"]:
        evs = sorted(by.get(s["id"], []), key=lambda e: e["t"])
        groups = {}
        for e in evs:
            key = (e["name"], e["stem"], e["src"].split(":")[0] + (":" + e["src"].split(":")[1] if ":" in e["src"] else ""))
            groups.setdefault(key, []).append(e)
        lines += [f"### {s['id']} — {s['title']} ({s['start']:.1f}–{s['end']:.1f})", "",
                  "| t (s) | n | stem | sound | cue | source | peak |", "|---:|---:|---|---|---|---|---:|"]
        for (name, stem, src), g in sorted(groups.items(), key=lambda kv: kv[1][0]["t"]):
            ts = f"{g[0]['t']:.3f}" if len(g) == 1 else f"{g[0]['t']:.2f}…{g[-1]['t']:.2f}"
            cue = ", ".join(sorted({e["cue"] for e in g if e["cue"]}))
            pk = max(e["peak"] for e in g)
            lines.append(f"| {ts} | {len(g)} | {stem} | {name} | {cue} | {src} | {20 * np.log10(max(pk, 1e-6)):.1f} dB |")
        lines.append("")
    path.write_text("\n".join(lines))


SCENE_FNS = [("s01", s01), ("s02", s02), ("s03", s03), ("s04", s04), ("s05a", s05a), ("s05b", s05b), ("s06", s06),
             ("s07a", s07a), ("s07b", s07b), ("s08", s08)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of scenes to audition (writes audio/foley/preview_*.wav)")
    a = ap.parse_args(argv)
    only = set(a.only.split(",")) if a.only else None
    t_start = time.time()
    OUTD.mkdir(parents=True, exist_ok=True)
    STEMS.mkdir(parents=True, exist_ok=True)
    ev = Events(FF.load_events())
    print(f"[foley] {len(ev.evs)} scene events, {len(FF.load_tracks())} physics tracks in cache/foley")
    mx = Mix(only)
    for sid, fn in SCENE_FNS:
        if mx.active(sid):
            t = time.time()
            fn(mx, ev)
            print(f"[foley] {sid} designed ({time.time() - t:.1f}s)")
    place_events(mx, ev)
    report = []
    physics(mx, report)
    t = time.time()
    mx.render_verbs()
    print(f"[foley] reverbs ({time.time() - t:.1f}s)")
    lims = finalize(mx)
    if only:
        s0 = min(SCENES[s]["start"] for s in only)
        s1 = max(SCENES[s]["end"] for s in only)
        a0, a1 = int(max(0, s0 - 0.5) * SR), int(min(DUR, s1 + 0.5) * SR)
        tag = "_".join(sorted(only))
        sf.write(OUTD / f"preview_{tag}.wav", mx.bus["sfx"][a0:a1] + mx.bus["amb"][a0:a1], SR, subtype="FLOAT")
        sf.write(OUTD / f"preview_{tag}_sfx.wav", mx.bus["sfx"][a0:a1], SR, subtype="FLOAT")
        sf.write(OUTD / f"preview_{tag}_amb.wav", mx.bus["amb"][a0:a1], SR, subtype="FLOAT")
        print(f"[foley] preview -> audio/foley/preview_{tag}.wav ({s0 - 0.5:.1f}s..)")
    else:
        sf.write(STEMS / "sfx.wav", mx.bus["sfx"], SR, subtype="FLOAT")
        sf.write(STEMS / "ambience.wav", mx.bus["amb"], SR, subtype="FLOAT")
        (OUTD / "placements.json").write_text(json.dumps(mx.log, indent=0))
        (OUTD / "physics.json").write_text(json.dumps(report, indent=1))
        write_cues_md(mx, report, lims, OUTD / "CUES.md")
        print(f"[foley] wrote stems + CUES.md ({len(mx.log)} placements)")
    for p in report:
        print(f"[physics] {p['file']:36s} r={p['corr_energy_rms']:.3f} snaps={p['snaps']}")
    print(f"[foley] done in {time.time() - t_start:.1f}s; limiter {lims}")


if __name__ == "__main__":
    main()

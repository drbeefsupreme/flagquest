"""OPVS SCHISMATICVM (film 3) — Foley build (owner: Foley-2). Re-runnable: re-reads films/opus/cache/foley on every run.

    cd <repo> && source .venv/bin/activate && python films/opus/tools/op_foley_build.py      # -> stems + cue sheet
    python films/opus/tools/op_foley_build.py --only m07,m08                                # audition -> audio/foley/preview_*
    python films/opus/tools/op_foley_check.py                                               # verify (appends to CUES.md)

Outputs  films/opus/audio/stems/sfx.wav, ambience.wav   48 kHz stereo float32, exactly 212.0 s (10,176,000 samples), T=0
         films/opus/audio/foley/CUES.md                  cue sheet: every placement + which scene exports were used
         films/opus/audio/foley/{placements,physics,events_used}.json

Every designed hit looks for the scene's exported event first (cache/foley/<scene>_*_events.json, vx.foley.export_events)
and falls back to the plan.json cue / word timing; every exported flag track (FlagTrack.export_foley) becomes
physics-derived flutter; the scenes' tile-physics tracks (kind="tiles": flips / avalanche / rise with per-frame counts)
drive the tesserae engine directly. Unused exported events are voiced by the generic mosaic-aware classifier.
Models: films/opus/tools/op_foley_lib.py (tesserae physics) on tools/foley_lib.py + tools/foley_flutter.py.
"""
import os

os.environ["VX_FILM"] = "opus"
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import argparse  # noqa: E402
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
import foley_build as B  # noqa: E402  (film-1 build: generic event models, Events, limiter)
import foley_flutter as FF  # noqa: E402
import foley_lib as L  # noqa: E402
import op_foley_lib as O  # noqa: E402
from foley_lib import SR, R, Snd, midi_hz, ns  # noqa: E402
from op_foley_lib import db  # noqa: E402

assert FF.FOLEY_CACHE == FROOT / "cache" / "foley", FF.FOLEY_CACHE
CACHE = FF.FOLEY_CACHE
TL = json.loads((FROOT / "timeline.json").read_text())
DUR = float(TL["duration"])
NTOT = int(round(DUR * SR))
assert NTOT == 10_176_000, NTOT
CUES = {c["id"]: c["t"] for c in TL["cues"]}
LINES = {ln["id"]: ln for ln in TL["lines"]}
SCENES = {s["id"]: s for s in TL["scenes"]}
SIL0, SIL_W, SIL1 = 145.0, 146.0, 146.5          # sacred silence; a breath of wind 146.0-146.5 (ambience only)
HARD_CUTS = (95.75, SIL0)                         # m04 -> m05 hard cut; the silence (reverb tails die too)
OUTD = FROOT / "audio" / "foley"
STEMS = FROOT / "audio" / "stems"

VERB_AT = [(0.0, "vault"), (22.0, "nave"), (48.5, "hall"), (62.5, "sanctuary"), (81.2, "hall"), (95.75, "nave"),
           (120.0, "void"), (131.5, "babel"), (145.0, "bare"), (163.0, "dome"), (187.5, "apse")]
DUCKS = {"edict_seal": (0.35, 0.7), "hypermeme": (0.3, 0.6), "breach": (0.25, 0.5), "collapse": (0.45, 0.8),
         "flag_planted": (0.5, 1.0), "convergence": (0.3, 0.8), "glorious": (0.45, 0.8)}


def verb_at(t):
    v = VERB_AT[0][1]
    for t0, name in VERB_AT:
        if t >= t0 - 1e-6:
            v = name
    return v


def scene_at(t):
    for s in TL["scenes"]:
        if s["start"] <= t < s["end"]:
            return s["id"]
    return "m10"


def word(lid, w, k=0):
    ws = [x for x in LINES[lid]["words"] if x["w"].strip(",.!?:").lower() == w.lower()]
    return ws[k]["s"] if len(ws) > k else LINES[lid]["start"]


# ================================================================================================
# scene exports
# ================================================================================================
def load_events():
    """every scene's exported hits (cache/foley/<scene>_<name>_events.json); music cue files are not ours."""
    out, files = [], {}
    for p in sorted(CACHE.glob("*_events.json")):
        if "music" in p.stem or "choir" in p.stem or "vocal" in p.stem:
            continue
        try:
            evs = json.loads(p.read_text())
        except Exception as e:  # half-written by a concurrent export
            print(f"[op-foley] skip {p.name}: {e}")
            continue
        files[p.name] = len(evs)
        scene = p.stem.split("_")[0]
        for e in evs:
            e = dict(e)
            e.update(scene=scene, file=p.name, kind=str(e.get("kind", "hit")).lower(), desc=str(e.get("desc", "")),
                     pan=float(np.clip(float(e.get("pan", 0.0)), -1, 1)), strength=float(e.get("strength", 1.0)),
                     t=float(e["t"]))
            out.append(e)
    return sorted(out, key=lambda e: e["t"]), files


def load_tracks():
    """every exported track, with all extra per-frame arrays (tiles: flips / fall / land / lift / bounce)."""
    out = {}
    for p in sorted(CACHE.glob("*.npz")):
        try:
            z = np.load(p, allow_pickle=True)
            d = {k: z[k] for k in z.files}
        except Exception as e:
            print(f"[op-foley] skip {p.name}: {e}")
            continue
        if "energy" not in d or len(np.atleast_1d(d["energy"])) < 2:
            continue
        scene = p.stem.split("_")[0]
        tr = dict(file=p.name, scene=scene, name=p.stem[len(scene) + 1:], kind=str(d.get("kind", "flag")),
                  f0=int(d.get("f0", 0)), fps=float(d.get("fps", 24)),
                  energy=np.nan_to_num(np.asarray(d["energy"], np.float64)),
                  pan=np.nan_to_num(np.asarray(d.get("pan", np.zeros_like(d["energy"])), np.float64)),
                  gain=float(d.get("gain", 1.0)),
                  events=np.asarray(d.get("events", np.zeros((0, 3))), np.float64).reshape(-1, 3))
        for k, v in d.items():
            if k not in tr and k not in ("kind", "f0", "fps", "gain") and np.ndim(v) == 1:
                tr[k] = np.nan_to_num(np.asarray(v, np.float64))
        out[(scene, tr["name"])] = tr
    return out


class Events(B.Events):
    def get(self, scene, keys, fallback, win=1.0, kind_only=False):
        """-> (t, pan|None, strength|None, src): the exported event nearest `fallback` (claimed) or the fallback."""
        e = self.find(scene, keys, near=fallback, win=win, kind_only=kind_only)
        if e:
            return e["t"], e["pan"], e["strength"], "event:" + e["file"]
        return fallback, None, None, "cue"

    def all(self, scene, keys, t0, t1, kind_only=True, tol=0.008):
        """all matching events in [t0, t1) (claimed); events a scene exported twice (same instant and pan under two
        kinds, e.g. 'tile_set' + 'set') are merged, keeping the stronger."""
        out = []
        for e in sorted(self.find(scene, keys, all_=True, t0=t0, t1=t1, kind_only=kind_only), key=lambda e: e["t"]):
            d = next((o for o in out if abs(o["t"] - e["t"]) <= tol and abs(o["pan"] - e["pan"]) <= 0.05), None)
            if d is None:
                out.append(e)
            elif e["strength"] > d["strength"]:
                out[out.index(d)] = e
        return out


def dur_of(e, default):
    return B.dur_from_desc(e, default) if e else default


def level_of(desc, key="level"):
    m = re.search(key + r"\s*=?\s*(\d+)", desc.lower())
    return int(m.group(1)) if m else None


# ================================================================================================
# mixer
# ================================================================================================
class Mix:
    def __init__(self, only=None):
        self.bus = {"sfx": np.zeros((NTOT, 2), np.float32), "amb": np.zeros((NTOT, 2), np.float32)}
        self.sends = {}
        self.log = []
        self.only = only
        # pre-impact breath: material already sounding dips just before a big hit (placements AT the hit exempt)
        self.ducks = [(CUES[c], a, b) for c, (a, b) in DUCKS.items()]

    def active(self, scene):
        return self.only is None or scene in self.only

    def add(self, stem, snd, t, gain=1.0, pan=None, send=0.0, verb=None, name="", scene=None, cue=None,
            src="design", cut=None, cross=False, noduck=False):
        x, sync = (snd.x, snd.sync) if isinstance(snd, Snd) else (snd, 0.0)
        scene = scene or scene_at(t)
        if gain <= 0 or not self.active(scene):
            return
        x = np.asarray(x, np.float32)
        if x.ndim == 1:
            x = L.pan2(x, 0.0 if pan is None else float(pan))
        elif pan is not None:
            x = L.pan2(x, float(pan))
        start = t - sync
        end_t = cut if cut is not None else DUR
        if not cross:          # hard cuts: nothing that starts before a cut sounds after it (reverb included)
            for hc in HARD_CUTS:
                if start < hc - 1e-6:
                    end_t = min(end_t, hc)
                    break
        i0 = int(round(start * SR))
        i1 = min(NTOT, i0 + len(x), int(round(end_t * SR)))
        a = max(0, i0)
        if i1 <= a:
            return
        seg = x[a - i0:i1 - i0] * gain
        for td, d_sfx, d_amb in self.ducks:
            depth = d_sfx if stem == "sfx" else d_amb
            if noduck or depth >= 1.0 or start >= td - 0.005 or i1 <= int((td - 0.12) * SR) or a >= int((td + 0.4) * SR):
                continue
            tg = np.arange(a, i1) / SR - td
            seg = seg * np.interp(tg, [-0.12, -0.02, 0.0, 0.4], [1.0, depth, depth, 1.0]).astype(np.float32)[:, None]
        if i1 == int(round(end_t * SR)) and i1 - a > 96 and end_t < DUR:
            seg = seg.copy()
            seg[-96:] *= np.linspace(1, 0, 96, dtype=np.float32)[:, None]          # 2 ms de-click at a cut
        self.bus[stem][a:i1] += seg
        if send > 0:
            v = verb or verb_at(max(0.0, t))
            key = (stem, v, end_t)
            if key not in self.sends:
                self.sends[key] = np.zeros((NTOT, 2), np.float32)
            self.sends[key][a:i1] += seg * send
        self.log.append(dict(t=round(float(t), 4), start=round(float(start), 4), stem=stem, name=name, scene=scene,
                             cue=cue, src=src, gain=round(float(gain), 5), peak=round(float(np.abs(seg).max()), 5),
                             send=round(float(send), 3)))

    def render_verbs(self):
        irs = {}
        for (stem, v, end_t), buf in self.sends.items():
            nz = np.nonzero(np.abs(buf).max(1) > 0)[0]
            if len(nz) == 0:
                continue
            if v not in irs:
                irs[v] = L.make_ir(**O.IR_SPECS[v])
            a, b = nz[0], nz[-1] + 1
            wet = L.convolve(buf[a:b], irs[v])
            e = min(NTOT, a + len(wet), int(round(end_t * SR)))
            if e <= a:
                continue
            w = wet[: e - a]
            if e == int(round(end_t * SR)) and e - a > 96 and end_t < DUR:
                w = w.copy()
                w[-96:] *= np.linspace(1, 0, 96, dtype=np.float32)[:, None]
            self.bus[stem][a:e] += w
        self.sends = {}


# ================================================================================================
# helpers
# ================================================================================================
def pan_of(p, default):
    return default if p is None else float(np.clip(p, -1, 1))


def P(x, level_db):
    """scale an array so its peak sits at level_db dBFS."""
    x = np.asarray(x, np.float32)
    return L.f32(x * (db(level_db) / max(1e-9, float(np.abs(x).max()))))


def bed(fn, r, t0, t1, pts, *a, **k):
    """a mono/stereo bed generator over [t0, t1), normalised to unit RMS, then shaped by a global-time gain
    envelope (so the envelope's dB values are the bed's RMS level)."""
    x = fn(r, t1 - t0, *a, **k)
    x = O.rms_norm(L.as_stereo(np.asarray(x, np.float32)), 1.0)
    return O.shaped(x, t0, pts)


def dbpts(pts):
    """[(t, dB)] -> [(t, linear)] (dB <= -120 -> 0)."""
    return [(t, 0.0 if g <= -120 else float(db(g))) for t, g in pts]


def letters_schedule(t0, t1, sylls, pan0=-0.75, pan1=0.75, spread=0.11):
    """syllable-synced letter settings: sylls = letter counts per syllable, spread over [t0, t1)."""
    n_s = len(sylls)
    tot = sum(sylls)
    out, j = [], 0
    for k, cnt in enumerate(sylls):
        ts = t0 + (t1 - t0) * k / n_s
        for q in range(cnt):
            out.append((ts + spread * q / max(1, cnt), pan0 + (pan1 - pan0) * j / max(1, tot - 1)))
            j += 1
    return out


def glints(mx, r, t0, t1, count, lvl=-38.0, pan=(-0.8, 0.8), scene=None, name="gold glints"):
    for _ in range(count):
        mx.add("sfx", O.glint(r), r.uniform(t0, t1), db(lvl + r.uniform(-4, 0)), r.uniform(*pan), send=0.35,
               name=name, scene=scene)


# ================================================================================================
# BEDS (ambience): one continuous basilica, spaces changing with the programme
# ================================================================================================
def beds(mx, ev):
    r = R(9)
    # the basilica's night room tone (continuous; muted by the silence; returns bare and faint after it)
    room = O.basilica_bed(r, DUR)
    pts = dbpts([(0.0, -120), (0.5, -52), (1.6, -41), (21.0, -41), (22.5, -39), (33.5, -39), (34.4, -42),
                 (36.3, -40), (48.5, -40), (49.5, -39), (62.5, -44), (80.8, -44), (81.6, -40), (95.74, -40),
                 (95.76, -38), (106.5, -39), (113.7, -40), (119.8, -40), (120.4, -120), (131.3, -120),
                 (132.5, -40), (141.8, -38), (144.99, -38), (145.0, -120), (146.5, -120), (147.5, -50),
                 (158.9, -48), (159.3, -43), (163.0, -42), (187.5, -41), (199.6, -39), (209.8, -41), (210.6, -48),
                 (211.7, -70), (212.0, -120)])
    mx.add("amb", O.shaped(room, 0.0, pts), 0.0, 1.0, name="the basilica at night (room tone)", scene="m01", cross=True)
    # wind high in the clerestory / dome windows: faint, more in the vault, the bare wall and the dome
    wind = O.high_wind(r, DUR)
    wp = dbpts([(0.0, -120), (1.0, -58), (22.0, -58), (23.0, -64), (95.7, -64), (95.8, -120), (106.5, -120),
                (107.0, -60), (113.7, -60), (114.2, -120), (131.5, -120), (144.9, -120), (145.0, -120),
                (146.0, -120), (146.01, -120), (146.5, -120), (146.6, -56), (163.0, -52), (180.0, -50), (187.5, -58),
                (211.0, -62), (212.0, -120)])
    mx.add("amb", O.shaped(wind, 0.0, wp), 0.0, 1.0, name="wind high in the windows", scene="m01", cross=True)
    # THE BREATH OF WIND at 146.0-146.5: the only sound in the silence (scene may export kind="wind")
    e = ev.find("m08", ["wind"], near=SIL_W, win=0.3, kind_only=True)
    t_w = e["t"] if e and SIL_W - 0.01 <= e["t"] < SIL1 else SIL_W
    br = O.high_wind(R(146), 1.2)
    br = O.shaped(br, t_w, dbpts([(t_w, -120), (t_w + 0.08, -60), (SIL1 - 0.05, -54), (SIL1 + 0.4, -56),
                                  (t_w + 1.2, -120)]))
    mx.add("amb", br, t_w, 1.0, name="a faint breath of wind (the only sound in the silence)", scene="m08",
           src="event:" + e["file"] if e else "cue", cross=True)
    # candle flame close by: m01 from the strike, m08 on the bare wall, m10 until the snuff
    e = ev.find("m01", ["candle_strike", "match"], near=CUES["candle_lit"], win=0.6, claim=False)
    t_c = e["t"] if e else CUES["candle_lit"]
    fl = bed(O.flame_bed, r, t_c, 22.0, dbpts([(t_c, -120), (t_c + 0.4, -46), (21.0, -46), (22.0, -120)]), 0.4)
    mx.add("amb", fl, t_c, 1.0, pan=-0.2, name="the candle flame (plume burble, 11 Hz flicker puffs)", scene="m01")
    fl = bed(O.flame_bed, r, SIL1, 163.0, dbpts([(SIL1, -120), (147.2, -45), (162.0, -46), (163.0, -120)]), 0.5)
    mx.add("amb", fl, SIL1, 1.0, pan=-0.3, name="the one candle before the bare wall", scene="m08")
    t_sn = snuff_time(ev)
    fl = bed(O.flame_bed, r, 199.6, t_sn + 0.05, dbpts([(199.6, -120), (201.0, -50), (t_sn - 0.02, -48),
                                                         (t_sn + 0.05, -120)]), 0.35)
    mx.add("amb", fl, 199.6, 1.0, pan=0.25, name="the altar candle (until the snuff)", scene="m10")
    # m02-m03: the slop's electric hum (rises with the waves; tails into the static)
    lv = lambda tg: np.interp(tg, [34.4, 34.6, 36.2, 36.8, 41.0, 44.0, 47.0, 48.5, 49.6],  # noqa: E731
                              [0.0, 0.08, 0.1, 0.45, 0.6, 0.85, 1.0, 1.0, 0.0])
    n = ns(49.6 - 34.4)
    sb = O.slop_bed(r, 49.6 - 34.4, lv(34.4 + np.arange(n) / SR))
    mx.add("amb", O.rms_norm(sb, db(-33)), 34.4, 1.0, name="the slop's electric hum (screens lit from within)",
           scene="m02")
    # m04: the Hypermind breathing (dilution-refrigerator pulse tube as a polycandelon), wing flames
    cp = O.rms_norm(L.as_stereo(L.cryo_pulse(r, 81.6 - 62.9, 2.4, 1.0)), 1.0)       # a slow breath, not a tempo
    cp = O.shaped(cp, 62.9, dbpts([(62.9, -120), (63.8, -47), (74.1, -45), (74.25, -50), (80.9, -50), (81.6, -120)]))
    mx.add("amb", cp, 62.9, 1.0, name="the Hypermind breathing (pulse-tube cycle)", scene="m04")
    ig = ignition_times(ev)
    n = ns(81.6 - 64.0)
    tg = 64.0 + np.arange(n) / SR
    lvl = np.zeros(n)
    for k, t in enumerate(ig):
        lvl = np.maximum(lvl, (tg > t + 0.2) * (k + 1) / 6.0)
    lvl = L.lp(L.f32(lvl), 2.0, 1).astype(np.float64) * np.interp(tg, [80.9, 81.6], [1.0, 0.0])
    fr = O.flame_roar(r, 81.6 - 64.0, np.clip(lvl, 0, 1))
    mx.add("amb", O.rms_norm(fr, db(-33)), 64.0, 1.0, name="six burning seraph wings (roar grows per canon)",
           scene="m04")
    # m05: the goggles' phosphor tubes (a thin chorus of high whines once they are down)
    n = ns(106.5 - 97.0)
    t = np.arange(n) / SR
    wh = sum(np.sin(2 * np.pi * (7700 + 260 * k) * t + k) * (1 + 0.3 * np.sin(2 * np.pi * 0.3 * t + k)) for k in range(5))
    wh = L.stereo(L.f32(wh), L.f32(np.roll(wh, 97)))
    wh = O.shaped(wh, 97.0, dbpts([(97.0, -120), (98.4, -76), (105.8, -74), (106.2, -120)]))
    mx.add("amb", wh, 97.0, 1.0, name="NVG phosphor tubes whining (chorus)", scene="m05")
    # m06: the congregation's temple hush (candles, robes) and the black void
    th = bed(lambda rr, d: L.hp(L.poisson_clicks(rr, ns(d), 4), 1500) * 0.4 + L.lp(L.pink(ns(d), rr), 500) * 0.2,
             r, 113.7, 120.2, dbpts([(113.7, -120), (114.2, -42), (119.6, -42), (120.2, -120)]))
    mx.add("amb", th, 113.7, 1.0, name="temple hush (candles, robes)", scene="m06")
    vd = bed(void_air, r, 120.0, 131.7, dbpts([(120.0, -120), (120.6, -44), (128.6, -44), (131.0, -48),
                                              (131.7, -120)]))
    mx.add("amb", vd, 120.0, 1.0, name="the void (dark air, slow resonant drift)", scene="m06")
    # m07: Babel's roar bed + Egregore's static (hard stop at 145.0)
    bb = bed(babel_roar, r, 131.5, SIL0, dbpts([(131.5, -120), (132.5, -40), (134.0, -35), (138.0, -32),
                                                (141.8, -28), (143.5, -30), (144.99, -34)]))
    mx.add("amb", bb, 131.5, 1.0, name="Babel's roar (tower mass, screen-brick buzz)", scene="m07")
    e = ev.find("m07", ["egregore"], near=CUES["egregore"], win=1.0, kind_only=True, claim=False)
    t_eg = e["t"] if e else CUES["egregore"]
    st = O.rms_norm(L.lp(L.tv_static(r, 141.9 - t_eg), 6000, 2), 1.0)
    st = O.shaped(st, t_eg, dbpts([(t_eg, -120), (t_eg + 0.6, -40), (141.0, -34), (141.9, -34)]))
    mx.add("amb", st, t_eg, 1.0, name="Lord Egregore's static", scene="m07", cut=141.95)
    # m10: dawn birds outside the apse windows (distant, through glass)
    evs = []
    for _ in range(14):
        b = L.bird(r, None, 60.0).x
        evs.append((r.uniform(0.3, 15.0), L.lp(b, 4500, 2), r.uniform(0.3, 1.0), r.uniform(-0.9, 0.9)))
    bd = L.scatter(ns(16.5), evs)
    mx.add("amb", O.shaped(bd, 187.5, dbpts([(187.5, -120), (188.2, 0), (201.0, 0), (203.5, -120)])), 187.5, db(-40),
           send=0.25, name="dawn birds outside the apse windows", scene="m10")


def void_air(r, dur):
    n = ns(dur)
    a = L.svf(L.brown(n, r), 110 + 70 * L.smooth_noise(n, r, 0.1, 2), 0.8, "lp")
    b = L.svf(L.pink(n, r), 600 * 2 ** (1.4 * L.smooth_noise(n, r, 0.07, 1)), 5, "bp") * 0.18
    return O.rms_norm(L.stereo(a + b, np.roll(a, 4800) + b[::-1].copy()))


def babel_roar(r, dur):
    n = ns(dur)
    u = np.arange(n) / n
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        low = L.svf(L.brown(n, r), 70 + 90 * u, 0.8, "lp")
        mid = L.svf(L.pink(n, r), 300 + 500 * u + 150 * L.smooth_noise(n, r, 0.4, 2), 1.2, "bp") * 0.35
        out[:, ch] = low + mid * (0.7 + 0.3 * L.smooth_noise(n, r, 0.8, 2))
    out += L.as_stereo(L.hum(r, dur, 55.0, (0.3, 0.5, 0.2, 0.2, 0.1, 0.1), 0.35) * 0.08 * u)
    return O.rms_norm(out)


def snuff_time(ev):
    e = ev.find("m10", ["snuff"], near=210.2, win=2.5, claim=False, kind_only=True)
    return e["t"] if e else 210.2


def ignition_times(ev):
    es = ev.find("m04", ["ignite"], all_=True, t0=63.0, t1=74.0, kind_only=True, claim=False)
    ts = sorted(e["t"] for e in es)
    return ts[:6] if len(ts) >= 1 else [word("H01", "Six") + 0.45 * k for k in range(6)]


# ================================================================================================
# m01 · INCIPIT: THE TESSERACT
# ================================================================================================
def m01(mx, ev):
    sc = "m01"
    r = R(101)
    T = CUES["candle_lit"]
    t_m, p, _, src = ev.get(sc, ["candle_strike", "match", "strike"], T, 0.6)
    ev.claim_near(sc, t_m, 0.35, ["candle_strike", "match", "strike"])      # the scrape is part of the strike
    p_m = pan_of(p, -0.2)
    mx.add("sfx", O.match_strike(r), t_m, db(-13), p_m, send=0.3, name="a match is struck (scrape -> flare)", scene=sc,
           cue="candle_lit", src=src)
    t_w, _, _, src = ev.get(sc, ["candle", "wick"], t_m + 0.3, 0.6, kind_only=True)
    mx.add("sfx", O.wick_catch(r), t_w, db(-24), p_m, send=0.35, name="the wick takes the flame", scene=sc, src=src)
    for k in range(4):
        mx.add("sfx", O.glint(r), t_m + 0.4 + 0.3 * k + r.uniform(0, 0.1), db(-40 - 2 * k), p_m + r.uniform(-0.6, 0.6),
               send=0.35, name="the first gold tesserae glint out of the dark", scene=sc)
    # 2.0 INCIPIT · OPVS · SCHISMATICVM: letters set themselves, syllable-synced to the cantor
    lets = ev.all(sc, ["tile_set", "letter", "set"], 1.8, 5.8)
    if lets:
        sched = [(e["t"], e["pan"], e["strength"], "event:" + e["file"]) for e in lets]
    else:
        sched = [(t, pn, 1.0, "cue") for t, pn in cantor_letters("incipit", 2.02, 4.45, [2, 2, 3, 1, 3, 5, 2, 2, 3])]
    for t, pn, s, s_src in sched:
        mx.add("sfx", O.set_flurry(r, int(12 + 10 * np.clip(s, 0, 1)), 0.07, "gold"), t, db(-25) * (0.6 + 0.4 * s),
               pn * 0.85, send=0.45, name="INCIPIT: a letter of gold tesserae sets itself", scene=sc, cue="incipit",
               src=s_src)
        if r.random() < 0.3:
            mx.add("sfx", O.glint(r), t + 0.06, db(-42), pn, send=0.3, name="letter glint", scene=sc)
    # 5.6 the rings of stars turn (the vault rotates): a deep, slow stone grind + star tesserae ticking past
    tes = ev.get(sc, ["tesseract"], CUES["tesseract"], 1.0, kind_only=True)
    fl = ev.get(sc, ["unfurl", "flag"], CUES["flag_rotates_in"], 0.6, kind_only=True)
    t_cut = fl[0] - 0.02
    d = t_cut - 5.6
    n = ns(d)
    grind = L.creak(r, d, 11, 16, (38, 57, 83, 121), 0.35, 0.4).x[:n]
    grind = L.lp(grind, 400, 2) * L.f32(np.interp(np.arange(n) / SR, [0, 1.5, d - 0.3, d], [0, 1, 1, 0]))
    mx.add("sfx", L.decorrelate(O.rms_norm(grind, 1.0), r), 5.6, db(-45), name="the starry vault turns (deep stone grind)",
           scene=sc, cut=t_cut)
    ts = np.sort(r.uniform(5.8, t_cut, 90))
    out = np.zeros((ns(d + 0.2), 2), np.float32)
    O.render_contacts(out, 5.6, O.tile_bank(), r, ts, r.uniform(0.6, 2.2, len(ts)), np.full(len(ts), "wall"), "gold",
                      np.sin(0.7 * ts), 1.0)
    mx.add("sfx", out, 5.6, db(-14), send=0.5, name="star tesserae tick past as the rings turn", scene=sc, cut=t_cut)
    # 8.4 the Tesseract: its edges are courses of gold and silver tesserae; turning in 4-D (XW and YW planes) the
    # projection's edges stretch and fold -> two streams of tile ticks whose density and position follow the two
    # rotations, over a rubbed-glass voice (D5 / A5, stick-slip locked to the glass modes, beating at the rates)
    tesseract(mx, r, tes[0], t_cut, ev, tes[3])
    # 14.8 the Flag rotates in: the slice 'shhing'; all tessellated sound stops; a wind from nowhere
    t_f, src = fl[0], fl[3]
    mx.add("sfx", L.whoosh(r, 1.2, 0.35, 900, 9000, 2.0, -0.6, 0.6), t_f, db(-22), send=0.5,
           name="the 3-D slice passes through our plane ('shhing')", scene=sc, cue="flag_rotates_in", src=src)
    mx.add("sfx", L.swell(r, 0.6, 300, 4000, 2.0).x[::-1].copy(), t_f, db(-30), send=0.4,
           name="the cloth unfolds: a soft seamless breath (no ticks)", scene=sc)
    ev.claim_near(sc, t_f, 0.05, ["emerge"])
    wn = O.rms_norm(L.wind(r, 22.0 - t_f + 0.5, level=0.3, gust=0.5, whistle=0.0, bright=0.3), 1.0)
    wn = O.shaped(wn, t_f - 0.3, dbpts([(t_f - 0.3, -120), (t_f + 0.5, -40), (19.5, -42), (21.8, -46), (22.2, -120)]))
    mx.add("amb", wn, t_f - 0.3, 1.0, name="a wind from nowhere", scene=sc)
    # the Tesseract's tiles scatter outward and circle the Flag - far, soft: nothing touches the cloth
    e = ev.find(sc, ["shatter", "scatter"], near=t_f + 0.05, win=0.6, kind_only=True)
    if e:
        d = dur_of(e, 4.5)
        mx.add("sfx", P(O.drift_clinks(r, min(1.2, d), 22), -30), e["t"] + 0.05, 1.0, send=0.6,
               name="the Tesseract's tiles scatter outward", scene=sc, src="event:" + e["file"])
        n = ns(d)
        lvl = np.interp(np.arange(n) / SR, [0, 0.4, d - 1.0, d], [0, 1, 0.6, 0])
        mx.add("sfx", O.dust_hiss(r, d, lvl, 0.25), e["t"], db(-40), send=0.3,
               name="...and circle the Flag (a faint glittering orbit)", scene=sc, src="event:" + e["file"])
    # 19.6 title: OPVS SCHISMATICVM sets itself in the great gold band
    T = CUES["title"]
    lets = ev.all(sc, ["tile_set", "letter", "title", "set"], T - 0.4, 21.0)
    if lets:
        sched = [(e["t"], e["pan"], e["strength"], "event:" + e["file"]) for e in lets]
    else:
        sched = [(T + 0.5 * (k / 14) ** 0.9, -0.8 + 1.6 * k / 14, 1.0, "cue") for k in range(15)]
    for t, pn, s, s_src in sched:
        mx.add("sfx", O.set_flurry(r, int(18 + 12 * s), 0.09, "gold", vel=0.7), t, db(-23), pn * 0.85, send=0.5,
               name="TITLE: a letter sets itself in the great gold band", scene=sc, cue="title", src=s_src)
    th = L.thud(r, 73.4, 0.5, 0.8, 500)
    mx.add("sfx", th, sched[0][0], db(-24), send=0.4, name="the gold band settles in its bed (low D thud)", scene=sc)
    glints(mx, r, T + 0.3, T + 1.4, 6, -38, scene=sc, name="title glints")
    # 21.0-22.0 tilt down out of the vault
    e = ev.find(sc, ["camera", "tilt"], near=21.0, win=0.8, kind_only=True)
    t_c = e["t"] if e else 21.0
    mx.add("sfx", L.whoosh(r, 2.0, 1.1, 70, 500, 0.7, 0.0, 0.0, dark=0.85, shape=1.5), t_c + 1.1, db(-32), send=0.3,
           name="tilt down out of the vault (air)", scene=sc, src="event:" + e["file"] if e else "cue")


def cantor_letters(key, t0, t1, sylls, pan0=-0.75, pan1=0.75):
    """letter schedule synced to the cantor's sung syllables (cache/foley/cantor_syllables.json when present)."""
    p = CACHE / "cantor_syllables.json"
    try:
        syl = [s["t"] for s in json.loads(p.read_text())[key]]
    except Exception:
        syl = []
    if len(syl) != len(sylls):
        return letters_schedule(t0, t1, sylls, pan0, pan1)
    out, j, tot = [], 0, sum(sylls)
    for ts, cnt in zip(syl, sylls):
        for q in range(cnt):
            out.append((ts + 0.1 * q / max(1, cnt), pan0 + (pan1 - pan0) * j / max(1, tot - 1)))
            j += 1
    return out


def tesseract(mx, r, t0, t1, ev, src):
    sc = "m01"
    d = t1 - t0
    n = ns(d)
    t = np.arange(n) / SR
    tg = t + t0
    # the inner cube passes through the outer at every 'inside_out'; each quarter turn completes 'aligned'
    ios = [(e["t"], "event:" + e["file"]) for e in ev.all(sc, ["inside_out", "inside-out"], t0, t1)] or \
          [(11.2, "design"), (13.2, "design")]
    turns = [(e["t"], "event:" + e["file"]) for e in ev.all(sc, ["turn"], t0, t1)]
    k_io = [x for x, _ in ios]
    # the two rotation planes (XW, YW): their rates rise as the Tesseract gathers speed
    pxw = 2 * np.pi * np.cumsum(np.interp(tg, [t0, t0 + 2, k_io[0], t1], [0.05, 0.16, 0.24, 0.3])) / SR
    pyw = 2 * np.pi * np.cumsum(np.interp(tg, [t0, t0 + 2, k_io[0], t1], [0.03, 0.1, 0.15, 0.2])) / SR
    fade_in = np.clip(t / 1.4, 0, 1) ** 1.5
    # rubbed glass voice (stick-slip locked to a glass mode; the rotation's projection beats its amplitude)
    out = np.zeros(n, np.float32)
    for f0, lv, dly in ((587.33, 1.0, 0.0), (880.0, 0.5, 1.6)):
        press = fade_in * (t > dly) * (0.6 + 0.4 * np.cos(pxw if f0 < 700 else pyw) ** 2)
        exc = L.stick_slip(r, d, f0, 0.002, press)
        body = L.resonators(exc, [f0 - 0.6, f0 + 0.6, f0 * 2.32, f0 * 4.25], [3.0, 3.0, 1.5, 0.8], [1.0, 1.0, 0.3, 0.1])
        out += body * lv
    out = L.norm(out, 1.0) * L.f32(np.clip((t1 - tg) / 0.08, 0, 1))
    mx.add("sfx", L.pan2(out, 0.35 * np.sin(pxw)), t0, db(-33), send=0.55,
           name="the Tesseract's rubbed-glass voice (D5/A5, beating with the 4-D rotation)", scene=sc, cue="tesseract",
           cut=t1, src=src)
    # the courses of tesserae along its edges: tick streams following each rotation (density ~ edge stretch)
    bank = O.tile_bank()
    buf = np.zeros((n + ns(0.2), 2), np.float32)
    for ph, mat, rate0 in ((pxw, "gold", 55.0), (pyw, "smalti", 40.0)):
        dens = rate0 * fade_in * (0.35 + 0.65 * np.abs(np.sin(ph)))
        ts = O.sample_rate_curve(r, 0.0, d, lambda u: np.interp(u, t, dens), int(np.trapezoid(dens, t)))
        pan = 0.75 * np.sin(np.interp(ts, t, ph))
        O.render_contacts(buf, 0.0, bank, r, ts, r.uniform(0.5, 2.0, len(ts)), np.full(len(ts), "wall"), mat, pan)
    mx.add("sfx", buf, t0, db(-15), send=0.5, name="courses of gold/silver tesserae flowing along the turning edges",
           scene=sc, cut=t1)
    # each time the inner cube turns inside-out through the outer: a sweeping flurry + air
    for k, (t_io, s_io) in enumerate(ios):
        fw = O.flip_wave(r, 0.0, 0.7, lambda u: np.sin(np.pi * np.clip(u / 0.7, 0, 1)) ** 2,
                         lambda u, k=k: (-0.7 + 2 * u) * (1 if k % 2 == 0 else -1), "gold", total=90, gain=0.8, spread=0.1)
        mx.add("sfx", fw, t_io - 0.35, db(-21), send=0.5, name="the inner cube turns inside-out through the outer",
               scene=sc, src=s_io, cut=t1)
        mx.add("sfx", L.whoosh(r, 1.2, 0.6, 400, 3500, 1.4, -0.5, 0.5), t_io, db(-35), send=0.5, name="inside-out air",
               scene=sc, cut=t1)
    # each quarter turn completes: every course aligns - a soft collective 'tk' of the whole figure seating
    for t_tu, s_tu in turns:
        mx.add("sfx", O.set_flurry(r, 40, 0.018, "gold", vel=0.5), t_tu, db(-27), 0.0, send=0.5,
               name="a quarter turn completes: the courses align", scene=sc, src=s_tu, cut=t1)
    mx.add("sfx", L.swell(r, 1.6, 600, 11000, 2.2), t0, db(-34), send=0.6, name="the Tesseract appears (air)",
           scene=sc, cue="tesseract", src=src)
    # the stars are revealed to be tiny square shadows: a scatter of silver flips across the vault
    ss = ev.find(sc, ["star_shadow", "shadow"], near=t0, win=4.0, kind_only=True)
    t_ss = ss["t"] if ss else t0 + 0.05
    d_ss = dur_of(ss, 1.0) if ss else 1.0
    fw = O.flip_wave(r, 0.0, d_ss + 0.4, lambda u: np.sin(np.pi * np.clip(u / (d_ss + 0.4), 0, 1)),
                     lambda u: 0.8 * np.sin(4 * u), "smalti", total=160, gain=0.6, spread=0.5)
    mx.add("sfx", fw, t_ss, db(-24), send=0.55, name="the stars flip: tiny square shadows of the Tesseract",
           scene=sc, src="event:" + ss["file"] if ss else "design", cut=t1)


# ================================================================================================
# m02 · THE GREAT MACHINES AND THE SLOP
# ================================================================================================
MACHINES = [("nations", "Nations", -0.55, 3.0, 2.2, 0.8), ("faiths", "Faiths", 0.0, 1.7, 1.5, 0.9),
            ("philosophies", "Philosophies", 0.55, 7.0, 0.7, 1.25)]


def flips_from_track(r, tr, face, cap=40000, t_lo=None, t_hi=None):
    """a scene's per-frame flip counts (kind='tiles', key 'flips' or energy) -> flip times + pans."""
    fps, t0 = tr["fps"], tr["f0"] / tr["fps"]
    c = tr.get("flips")
    if c is None:
        c = tr["energy"] * 60.0
    t, p = O.counts_to_times(r, t0, fps, c, tr["pan"], 1.0, cap)
    k = np.ones(len(t), bool)
    if t_lo is not None:
        k &= t >= t_lo
    if t_hi is not None:
        k &= t < t_hi
    return t[k], np.clip(p[k] + r.normal(0, 0.2, k.sum()), -1, 1)


def render_flips(r, t, p, face, t0, dur):
    """flip grains at times t / pans p; per-flip gain ~ 1/sqrt(local rate) so a torrent stays a torrent, not a wall."""
    bank = O.tile_bank()
    out = np.zeros((ns(dur + 0.3), 2), np.float32)
    g = r.uniform(0.35, 1.0, len(t)) / np.sqrt(1 + O.local_rate(t) / 150.0)
    bank.render(out, t0, t, bank.pick(r, ("flip", face), len(t)), g, p)
    return out


def m02(mx, ev):
    sc = "m02"
    r = R(202)
    t_slop = ev.get(sc, ["tile_flip"], CUES["slop"], 0.8, kind_only=True)
    # 26.4 / 27.4 / 28.1: the three great machines, each revealed on its word; their gilded cogs turn until the slop
    for key, wd, pdef, rate, size, bright in MACHINES:
        tw = word("N04", wd)
        e = ev.find(sc, [key], near=tw, win=1.5) or ev.find(sc, ["machine"], near=tw, win=1.0, kind_only=True)
        t, p = (e["t"], e["pan"]) if e else (tw, pdef)
        src = "event:" + e["file"] if e else "cue"
        mx.add("sfx", O.set_flurry(r, 45, 0.3, "gold", vel=0.6), t, db(-24), p, send=0.45,
               name=f"{wd.upper()}: the machine's panel of tesserae sets", scene=sc, cue="machines", src=src)
        mx.add("sfx", L.thud(r, 73.4 if key == "nations" else 110.0, 0.4, 0.8, 600), t, db(-30), p, send=0.4,
               name=f"{wd.upper()}: mortar thud", scene=sc)
        g = ev.find(sc, ["gears", "gear", "cog"], near=t + 0.3, win=1.5, kind_only=True)
        t_g = g["t"] if g else t + 0.15
        t_end = t_g + dur_of(g, t_slop[0] - 0.12 - t_g) if g else t_slop[0] - 0.12
        mh = re.search(r"halt[^0-9]*(\d+\.\d+)", g["desc"]) if g else None
        t_halt = float(mh.group(1)) if mh else t_end - 0.9
        d = t_end - t_g
        if d <= 0.3:
            continue
        n = ns(d)
        tg = t_g + np.arange(n) / SR
        rt = rate * np.interp(tg, [t_g, t_g + 0.9, t_halt, t_end], [0.15, 1.0, 1.0, 0.0])   # spin up ... halt
        gt = O.gear_train(r, d, rt, size, bright, width=0.2)
        mx.add("sfx", O.rms_norm(gt, db(-40)), t_g, 1.0, p, send=0.35,
               name=f"{wd.upper()}: gilded cogs of tesserae turning", scene=sc,
               src="event:" + g["file"] if g else "design")
        if mh:   # the slop reaches the machine: its cogs grind to a halt
            gr = L.lp(L.stone_slide(r, t_end - t_halt + 0.2, 0.5).x, 2500, 2)
            mx.add("sfx", gr, t_halt, db(-30), p, send=0.35, name=f"{wd.upper()}: the slop jams the cogs (grind)",
                   scene=sc, src="event:" + g["file"])
    # 34.4 THE SLOP: one tessera flips into a glowing phone screen (a cheap little wake 'pip')
    t_s, p_s = t_slop[0], pan_of(t_slop[1], 0.12)
    mx.add("sfx", O.flip_grain(r, "smalti", True), t_s, db(-19), p_s, send=0.45,
           name="THE SLOP: one tessera flips into a glowing screen", scene=sc, cue="slop", src=t_slop[3])
    mx.add("sfx", L.ping(r, 1318.5 * 1.03, 0.2, 2), t_s + 0.06, db(-36), p_s, send=0.3,
           name="the screen wakes (a cheap detuned pip)", scene=sc)
    # 36.3 slop_crash + 'wave upon wave': split-flap waves of screens (the scene's flip physics if exported)
    tr = TRACKS.get((sc, "flips"))
    fw = ev.all(sc, ["flip_wave"], 35.0, 47.5)
    waves = [(e["t"], dur_of(e, 1.6), float(np.clip(e["pan"] * 1.2, -1, 1)), float(np.clip(-e["pan"] * 0.8, -1, 1)),
              900 + 900 * e["strength"]) for e in fw] or \
        [(36.3, 2.3, -1.0, 1.0, 1500.0), (41.05, 1.5, 1.0, -1.0, 1100.0), (41.74, 2.2, -1.0, 1.0, 1800.0)]
    if tr is not None:
        t, p = flips_from_track(r, tr, "screen", 15000, t_s + 0.3, 48.4)
        src = "physics:" + tr["file"]
    else:
        ts, ps = [], []
        for (w0, wd_, pa, pb, peak) in waves:
            rate = lambda u, w0=w0, wd_=wd_, peak=peak: peak * np.sin(np.pi * np.clip((u - w0) / wd_, 0, 1)) ** 1.5  # noqa
            tot = int(peak * wd_ * 0.45)
            tt_ = O.sample_rate_curve(r, w0, w0 + wd_, rate, tot)
            ts.append(tt_)
            ps.append(np.clip(pa + (pb - pa) * (tt_ - w0) / wd_ + r.normal(0, 0.18, tot), -1, 1))
        chaos = O.sample_rate_curve(r, 43.0, 47.3, lambda u: np.interp(u, [43.0, 44.0, 46.6, 47.3], [0, 1, 1, 0]), 1500)
        ts.append(chaos)
        ps.append(r.uniform(-1, 1, len(chaos)))
        t, p = np.concatenate(ts), np.concatenate(ps)
        src = "event:" + fw[0]["file"] if fw else "cue"
    fl = O.win_norm(render_flips(r, t, p, "screen", 36.0, 12.5), -23.0)
    mx.add("sfx", fl, 36.0, 1.0, send=0.35, name=f"split-flap waves: {len(t)} tesserae flip into glowing screens",
           scene=sc, cue="slop_crash", src=src)
    # the mass of each wave: a dark wash + an electrical crackle front
    for (w0, wd_, pa, pb, peak) in waves:
        mx.add("sfx", L.whoosh(r, wd_ + 0.6, wd_ * 0.45, 90, 900, 0.8, pa * 0.8, pb * 0.8, dark=0.7), w0 + wd_ * 0.45,
               db(-27) * (peak / 1800) ** 0.5, send=0.35, name="the slop washes across the wall (dark wash)", scene=sc,
               src="event:" + fw[0]["file"] if fw else "cue")
        n = ns(wd_)
        cr = L.hp(L.poisson_clicks(r, n, 900 * np.sin(np.pi * np.arange(n) / n) ** 2), 2500)
        mx.add("sfx", L.pan2(L.f32(cr), np.linspace(pa, pb, n) * 0.8), w0, db(-30), send=0.2,
               name="electrical crackle along the front", scene=sc)
    # 42.9-47.2: every tessera its own feed: scroll ticks, cheap detuned pings, glitch bursts -> noise
    evs = []
    for _ in range(90):
        tq = r.uniform(0.0, 4.4) ** 0.8 * 4.4 ** 0.2
        m = r.choice([62, 64, 66, 67, 69, 71, 72, 74, 76]) + 12 * r.integers(1, 3)
        evs.append((tq, L.ping(r, float(midi_hz(m)) * r.uniform(0.97, 1.03), 0.12, int(r.integers(0, 3))),
                    r.uniform(0.2, 1.0), r.uniform(-1, 1)))
    for _ in range(40):
        evs.append((r.uniform(0.3, 4.4), L.glitch(r, r.uniform(0.04, 0.16)), r.uniform(0.3, 1.0), r.uniform(-1, 1)))
    fd = L.scatter(ns(5.0), evs)
    e = ev.find(sc, ["noise"], near=43.5, win=2.5, kind_only=True)
    t_n = e["t"] if e else 42.9
    mx.add("sfx", fd, t_n, db(-30), send=0.2, name="every tessera its own feed (pings, glitches) -> noise", scene=sc,
           src="event:" + e["file"] if e else "cue")
    # 47.0-48.5 push into one tessera: its screen's static grows to fill the frame, then resolves in m03
    e = ev.find(sc, ["static"], near=47.0, win=1.5, kind_only=True)
    t_st = e["t"] if e else 47.0
    st = O.rms_norm(L.lp(L.tv_static(r, 49.9 - t_st), 7000, 2), 1.0)
    st = O.shaped(st, t_st, dbpts([(t_st, -120), (48.3, -26), (48.55, -24), (49.2, -34), (49.8, -120)]))
    mx.add("sfx", st, t_st, 1.0, name="push into one tessera: full-frame static (resolves in m03)", scene=sc,
           src="event:" + e["file"] if e else "cue")
    mx.add("sfx", L.swell(r, 48.5 - t_st, 200, 6000, 2.0), 48.5, db(-30), name="push-in air", scene=sc)


# ================================================================================================
# m03 · THE EDICT OF THE JAGUAR
# ================================================================================================
def m03(mx, ev):
    sc = "m03"
    r = R(303)
    # 48.5: the static resolves tile by tile (flipping) into the gold of the Jaguar's panel
    tr = TRACKS.get((sc, "flips"))
    if tr is not None:
        t, p = flips_from_track(r, tr, "gold", 20000, 48.4, 62.5)
        ev.claim_near(sc, 48.6, 1.0, ["flip_wave"])          # the wave is the track itself
        src = "physics:" + tr["file"]
    else:
        e = ev.find(sc, ["flip_wave"], near=48.6, win=1.0, kind_only=True)
        w0 = e["t"] if e else 48.55
        wd_ = dur_of(e, 1.25)
        t = O.sample_rate_curve(r, w0, w0 + wd_, lambda u: np.sin(np.pi * np.clip((u - w0) / wd_, 0, 1)) ** 0.8, 700)
        p = np.clip(r.normal(0, 0.55, len(t)), -1, 1)
        src = "event:" + e["file"] if e else "cue"
    mx.add("sfx", O.win_norm(render_flips(r, t, p, "gold", 48.3, 14.2), -21.0), 48.3, 1.0, send=0.4,
           name=f"the static resolves tile by tile into gold ({len(t)} flips)", scene=sc, src=src)
    ev.claim_near(sc, 49.5, 1.3, ["resolve"])
    # the titulus HIC IMPERATOR IAGVAR LEGEM SANCIT sets letter by letter; the numerals of MMXLII
    for e in ev.all(sc, ["click", "titul", "letter", "numeral", "tile_set"], 48.5, 62.0):
        dsc = e["desc"].lower()
        if "wink" in dsc:
            mx.add("sfx", O.glint(r, 1.3), e["t"], db(-34), e["pan"], send=0.5, name="he winks (a tile glints)",
                   scene=sc, src="event:" + e["file"])
        elif "letter by letter" in dsc:
            titulus_run(mx, r, e, "HICIMPERATORIAGVARLEGEMSANCIT", sc)
        else:
            mx.add("sfx", O.set_flurry(r, int(14 + 16 * e["strength"]), 0.07, "gold", vel=0.55), e["t"], db(-27),
                   e["pan"], send=0.45, name="a numeral / letter sets in gold", scene=sc, src="event:" + e["file"])
    # 52.5 edict_sign: the golden stylus on the codex
    strokes = ev.all(sc, ["stylus", "stroke"], 51.5, 57.5, tol=-1.0)
    span = {id(e): (e["t"], e["t"] + dur_of(e, 1.1)) for e in strokes}
    strokes = [e for e in strokes if sum(1 for o in strokes if o is not e and span[id(o)][0] >= span[id(e)][0] - 1e-3
                                         and span[id(o)][1] <= span[id(e)][1] + 1e-3) < 2]   # per-stroke, not umbrella
    if not strokes:
        strokes = [dict(t=CUES["edict_sign"], desc="(to 53.7)", file=None, pan=0.1)]
    for k, e in enumerate(strokes):
        d = dur_of(e, 1.1)
        mx.add("sfx", O.stylus(r, d, max(2, int(d * 4))), e["t"], db(-24), pan_of(e.get("pan"), 0.1), send=0.3,
               name="the golden stylus signs the codex", scene=sc, cue="edict_sign" if k == 0 else None,
               src="event:" + e["file"] if e.get("file") else "cue")
    # 58.0 edict_seal: the bulla slams; the whole wall's tesserae jump and re-seat (and a few pop out and fall)
    t_b, p, _, src = ev.get(sc, ["seal", "bulla"], CUES["edict_seal"], 0.6, kind_only=True)
    mx.add("sfx", O.bulla_slam(r), t_b, db(-3), pan_of(p, 0.0), send=0.35, name="THE SEAL: the gold bulla slams",
           scene=sc, cue="edict_seal", src=src)
    e = ev.find(sc, ["tiles_jump", "jump", "resettle"], near=t_b + 0.05, win=0.8, kind_only=True)
    t_j = e["t"] if e else t_b + 0.01
    mx.add("sfx", P(O.tiles_jump(r, 3000, dur_of(e, 0.45), 40.0, 7.0, pan_of(p, 0.0)).x, -11), t_j, 1.0, send=0.4,
           name="every tessera on the wall jumps and re-seats", scene=sc, src="event:" + e["file"] if e else "design")
    m = re.search(r"(\d+) pop out", e["desc"]) if e else None
    if m:
        k = int(m.group(1))
        tl_, vl, pl = O.fall_plan(r, t_j + r.uniform(0.05, 0.35, k), r.uniform(1.0, 3.0, k), r.uniform(-0.8, 0.8, k))
        mx.add("sfx", P(O.tile_rain(r, t_j, 2.0, tl_, vl, pl, mats=("gold", 0.6, "smalti", 0.4)), -20), t_j, 1.0,
               send=0.45, name=f"{k} tesserae pop out of the wall and fall", scene=sc, src="event:" + e["file"])
    # J02: the Emperor leans OUT of the wall (his mosaic slab grinds forward; silk chlamys; pendilia swing)
    t_l, p, _, src = ev.get(sc, ["rustle", "lean"], word("J02", "Except") + 0.2, 1.2, kind_only=True)
    mx.add("sfx", L.cloth_rustle(r, 0.6, 0.8, 0.7), t_l, db(-30), pan_of(p, 0.0), send=0.2,
           name="he leans in (silk chlamys)", scene=sc, src=src)
    mx.add("sfx", L.lp(L.stone_slide(r, 0.7, 0.4).x, 2200, 2), t_l, db(-28), pan_of(p, 0.0), send=0.4,
           name="his panel of tesserae grinds out of the wall", scene=sc, src=src)
    for k in range(5):
        mx.add("sfx", O.link_click(r, 0.7, (0.02, 0.08)), t_l + 0.08 + 0.07 * k + r.uniform(0, 0.04), db(-38),
               pan_of(p, 0.0) + r.uniform(-0.15, 0.15), send=0.3, name="the pendilia's jewels swing", scene=sc)
    # the ochre sleeve slips the solidus into his paw: coin drop, then 62.1 KA-CHING
    t_k, p_k, _, src_k = ev.get(sc, ["ka_ching", "ching"], CUES["ka_ching"], 0.5, kind_only=True)
    cs = ev.all(sc, ["coin"], 59.5, t_k - 0.05)
    t_c, p_c, src = (cs[0]["t"], cs[0]["pan"], "event:" + cs[0]["file"]) if cs else (t_k - 0.22, 0.2, "cue")
    sl = ev.all(sc, ["whoosh"], t_c - 1.2, t_c + 0.3)
    for e in sl:   # the sleeve reaching in / withdrawing, the coin raised like the Host
        d = dur_of(e, 0.5)
        mx.add("sfx", L.cloth_rustle(r, max(0.25, d), 0.5 + 0.4 * e["strength"], 0.9), e["t"], db(-33), e["pan"],
               send=0.2, name="the ochre sleeve (" + e["desc"][:30] + ")", scene=sc, src="event:" + e["file"])
    if not sl:
        mx.add("sfx", L.cloth_rustle(r, 0.35, 0.6, 0.9), t_c - 0.45, db(-34), pan_of(p_c, 0.3), send=0.2,
               name="an ochre sleeve slips in", scene=sc)
    mx.add("sfx", O.coin_drop(r), t_c, db(-20), pan_of(p_c, 0.2), send=0.35, name="a gold solidus drops into his paw",
           scene=sc, src=src)
    e = ev.find(sc, ["crack"], near=t_c + 0.05, win=0.6, kind_only=True)
    if e:
        mx.add("sfx", O.bed_crack(r, 0.2, 4, True), e["t"], db(-30), e["pan"], send=0.4,
               name="a hairline crack runs through his halo", scene=sc, src="event:" + e["file"])
    mx.add("sfx", O.ka_ching(r), t_k, db(-12), pan_of(p_k, 0.15), send=0.35, name="KA-CHING (D7/A7 bell)", scene=sc,
           cue="ka_ching", src=src_k)
    # 62.1-62.5 the coin's glint flares to a gold point (hand-off: the first eye opens there)
    ev.claim_near(sc, t_k, 0.05, ["flare"])
    mx.add("sfx", L.swell(r, 0.4, 1500, 12000, 2.5), 62.5, db(-30), send=0.5, name="the glint flares to a gold point",
           scene=sc)
    mx.add("sfx", O.glint(r, 1.5), t_k + 0.12, db(-30), 0.0, send=0.6, name="gold point glint", scene=sc)


def titulus_run(mx, r, e, text, sc):
    """a titulus setting itself letter by letter across its band over the event's duration (left -> right)."""
    d = dur_of(e, 0.06 * len(text))
    k = len(text)
    for q in range(k):
        mx.add("sfx", O.set_flurry(r, int(10 + 8 * e["strength"]), 0.05, "gold", vel=0.5), e["t"] + d * q / k,
               db(-29), float(np.clip(-0.8 + 1.6 * q / max(1, k - 1), -1, 1)), send=0.45,
               name="the titulus sets itself letter by letter", scene=sc, src="event:" + e["file"])


# ================================================================================================
# m04 · FLAGARTHA: THE HYPERMIND AND THE DISPUTATION
# ================================================================================================
def m04(mx, ev):
    sc = "m04"
    r = R(404)
    T_LOCK = ev.get(sc, ["lock"], CUES["hypermeme"], 0.6, kind_only=True)
    # eyes: the first opens out of the gold point in darkness, then the wheels' eyes open
    all_eyes = ev.all(sc, ["eye", "blink"], 62.5, 95.7)
    blinks = [e for e in all_eyes if "blink" in (e["desc"] + e["kind"]).lower()]
    eyes = [e for e in all_eyes if e not in blinks and e["t"] < 74.0]
    if not eyes:
        eyes = [dict(t=CUES["hypermind_wake"], pan=0.0, strength=1.0, file=None)] + \
               [dict(t=63.4 + 0.19 * k + r.uniform(0, 0.08), pan=r.uniform(-0.8, 0.8), strength=0.5, file=None)
                for k in range(9)]
    for k, e in enumerate(eyes):
        big = k == 0
        mx.add("sfx", O.eye_open(r, 2.2 if big else 0.7 + 0.5 * e["strength"]), e["t"], db(-16 if big else -27),
               e["pan"], send=0.5, name="an eye of tesserae opens" + (" (the first, in the dark)" if big else ""),
               scene=sc, cue="hypermind_wake" if big else None, src="event:" + e["file"] if e.get("file") else "cue")
    for e in blinks:     # every eye blinks at once: lids of tiles shut and open
        for dt, g in ((0.0, 1.0), (0.13, 0.8)):
            mx.add("sfx", O.set_flurry(r, 120, 0.05, "gold", vel=0.45), e["t"] + dt, db(-27) * g, e["pan"], send=0.45,
                   name="all the eyes blink at once (lids of tesserae)", scene=sc, src="event:" + e["file"])
    # the wheels within wheels turn (speed from the scene if exported), accelerating into the lock
    t_w0 = eyes[0]["t"]
    wls = ev.all(sc, ["wheels", "wheel"], 62.5, T_LOCK[0])
    t_ws = wls[0]["t"] if wls else t_w0 + 0.3
    d = T_LOCK[0] - t_ws
    n = ns(d)
    tg = t_ws + np.arange(n) / SR
    if wls:   # speed keyframes from the scene: each 'wheels' event's strength = the speed it spins up to (1.2 s)
        kf, prev = [], 0.02
        for e in wls:
            v = 0.15 + 0.95 * e["strength"]
            kf += [(e["t"], prev), (e["t"] + 1.2, v)]
            prev = v
        kf += [(T_LOCK[0] - 0.25, 1.2), (T_LOCK[0], 1.25)]
        kt, kv = zip(*sorted(kf))
        sp = np.interp(tg, kt, kv)
    else:
        sp = np.interp(tg, [t_ws, t_ws + 3.0, 72.4, T_LOCK[0] - 0.25, T_LOCK[0]], [0.05, 0.3, 0.7, 1.15, 1.2])
    mx.add("sfx", O.rms_norm(O.wheel_bed(r, d, sp), db(-30)), t_ws, 1.0, send=0.35,
           name="wheels within wheels of gold tesserae turning (ticks per passing tile)", scene=sc,
           src="event:" + wls[0]["file"] if wls else "design", cut=T_LOCK[0])
    for t in (63.6, 68.1, 71.2):
        mx.add("sfx", O.chain_creak(r, 1.6), t, db(-34), r.uniform(-0.6, 0.6), send=0.5,
               name="the polycandelon's chains creak", scene=sc)
    # H01: the six wings unfold and ignite one by one (six canons)
    unf = ev.all(sc, ["wing_unfold", "unfold"], 63.0, 74.0)
    if not unf:
        unf = [dict(t=64.3 + 0.28 * k, pan=[-0.8, 0.8, -0.5, 0.5, -0.25, 0.25][k], file=None) for k in range(6)]
    for e in unf:
        mx.add("sfx", O.wing_unfold(r, 0.8, 140), e["t"], db(-20), pan_of(e.get("pan"), 0.0), send=0.45,
               name="a seraph's wing of tesserae unfolds", scene=sc, src="event:" + e["file"] if e.get("file") else "cue")
    ig = ignition_times(ev)
    ig_e = {round(e["t"], 3): e for e in ev.all(sc, ["ignite"], 63.0, 74.0)}
    for k, t in enumerate(ig):
        e = ig_e.get(round(t, 3))
        mx.add("sfx", O.ignite(r, 1.0 + 0.1 * k), t, db(-12 - 0.5 * k), pan_of(e["pan"] if e else None,
               [-0.8, 0.8, -0.5, 0.5, -0.25, 0.25][k % 6]), send=0.45,
               name=f"CANON {'I II III IV V VI'.split()[k % 6]}: a wing ignites", scene=sc,
               src="event:" + e["file"] if e else "cue")
    # 74.2 hypermeme: the wheels LOCK
    mx.add("sfx", O.lock_boom(r), T_LOCK[0], db(-2), pan_of(T_LOCK[1], 0.0), send=0.4, name="THE WHEELS LOCK (bronze bolt, D sub)",
           scene=sc, cue="hypermeme", src=T_LOCK[3])
    k = 60
    tl_, vl, pl = O.fall_plan(r, T_LOCK[0] + r.exponential(0.25, k), r.uniform(0.5, 4.0, k), r.uniform(-0.8, 0.8, k))
    mx.add("sfx", P(O.tile_rain(r, T_LOCK[0], 2.0, tl_, vl, pl), -16), T_LOCK[0], 1.0, send=0.45,
           name="shaken tesserae rain from the wheels", scene=sc)
    e = ev.find(sc, ["portal", "pupil"], near=T_LOCK[0] + 0.1, win=0.8, kind_only=True)
    t_p = e["t"] if e else T_LOCK[0] + 0.1
    mx.add("sfx", L.swell(r, 0.7, 800, 12000, 2.0).x[::-1].copy(), t_p, db(-30), send=0.6,
           name="the great pupil opens into light (air)", scene=sc, src="event:" + e["file"] if e else "design")
    mx.add("sfx", O.glint(r, 1.5), t_p + 0.05, db(-34), 0.0, send=0.6, name="light glint", scene=sc)
    # the plain Flag descends from the centre
    e = ev.find(sc, ["descend"], near=T_LOCK[0] + 0.4, win=1.5, kind_only=True)
    t_d = e["t"] if e else T_LOCK[0] + 0.35
    mx.add("sfx", O.descend_chain(r, dur_of(e, 1.8)), t_d, db(-30), 0.0, send=0.45,
           name="the Flag descends on its chain from the centre", scene=sc, src="event:" + e["file"] if e else "cue")
    # 81.2 videtur: the wall re-sets itself into the two-register disputation; tituli set word by word
    T = CUES["videtur"]
    e = ev.find(sc, ["flip_wave"], near=T, win=1.0, kind_only=True)
    w0, wd_ = (e["t"], dur_of(e, 0.9)) if e else (T - 0.05, 0.9)
    t = O.sample_rate_curve(r, w0, w0 + wd_, lambda u: np.sin(np.pi * np.clip((u - w0) / wd_, 0, 1)), 900)
    mx.add("sfx", O.win_norm(render_flips(r, t, np.clip(r.normal(0, 0.6, len(t)), -1, 1), "gold", w0, wd_ + 0.5),
                            -21.0), w0, 1.0,
           send=0.45, name="VIDETVR: the wall re-sets into the disputation (a wave of flips)", scene=sc, cue="videtur",
           src="event:" + e["file"] if e else "cue")
    tit = ev.all(sc, ["tile_set", "titul", "letter"], 80.5, 95.7)
    if not tit:
        tit = [dict(t=w0 + wd_ + 0.1 + 0.22 * k, pan=-0.6 + 0.3 * k, strength=0.8, file=None) for k in range(5)] + \
              [dict(t=CUES["sed_contra"] + 0.15 * k, pan=0.2 + 0.2 * k, strength=0.8, file=None) for k in range(2)] + \
              [dict(t=CUES["respondeo"] + 0.2 * k, pan=-0.3 + 0.3 * k, strength=0.8, file=None) for k in range(3)]
    for e in tit:
        mx.add("sfx", O.set_flurry(r, int(30 + 20 * e["strength"]), 0.22, "gold", vel=0.6), e["t"], db(-25),
               e["pan"], send=0.45, name="a titulus word sets itself (QVAESTIO / VIDETVR / SED CONTRA / RESPONDEO)",
               scene=sc, src="event:" + e["file"] if e.get("file") else "cue")
    # 'the false coherence must fall': the smooth slop-mosaic cracks and falls away; the gold glints behind it
    crs = ev.all(sc, ["crack"], 88.0, 92.0)
    if crs:   # the too-perfect smiling slop-world cracks like a phone screen, tessera by tessera
        for e in crs:
            mx.add("sfx", O.bed_crack(r, 0.12, 3, True), e["t"], db(-22), e["pan"], send=0.4,
                   name="the smiling slop-world cracks like a phone screen", scene=sc, src="event:" + e["file"])
            mx.add("sfx", O.tile_click(r, "screen", "wall", 1.0, 0.8), e["t"], db(-26), e["pan"], send=0.3,
                   name="screen glass snaps", scene=sc)
    else:
        t_cr, p, _, src = ev.get(sc, ["crack"], word("C02", "must"), 1.2, kind_only=True)
        mx.add("sfx", O.bed_crack(r, 0.5, 0), t_cr, db(-12), pan_of(p, 0.0), send=0.45,
               name="the too-perfect slop-mosaic cracks", scene=sc, src=src)
    tls = ev.all(sc, ["tiles", "fall"], 89.0, 93.0)
    counts = [(e, int(m.group(1))) for e in tls for m in [re.search(r"(\d+)\s*tesserae", e["desc"])] if m]
    if counts:   # the scene's per-frame landing counts: every fallen slop tile hits the floor
        t_all = np.concatenate([e["t"] + r.uniform(-0.02, 0.02, c) for e, c in counts])
        p_all = np.concatenate([np.clip(e["pan"] + r.normal(0, 0.5, c), -1, 1) for e, c in counts])
        t_f, d = float(t_all.min()), float(t_all.max() - t_all.min())
        v = np.sqrt(2 * O.G * r.uniform(0.5, 3.0, len(t_all)))
        rain = O.tile_rain(r, t_f, d + 1.0, t_all, v, p_all, mats=("screen", 0.8, "smalti", 0.2), heap_p=0.4)
        src = "event:" + counts[0][0]["file"]
        name = f"the slop-world falls away: {len(t_all)} screen tiles hit the floor"
    else:
        e = ev.find(sc, ["fall"], near=word("C02", "fall"), win=1.2, kind_only=True)
        t_f = e["t"] if e else word("C02", "fall")
        d = dur_of(e, 1.1)
        k = 420
        tl_, vl, pl = O.fall_plan(r, O.sample_rate_curve(r, t_f, t_f + d, lambda u: np.exp(-(u - t_f) / d), k),
                                  r.uniform(0.4, 3.0, k), r.uniform(-0.9, 0.9, k))
        rain = O.tile_rain(r, t_f, d + 1.0, tl_, vl, pl, mats=("screen", 0.8, "smalti", 0.2), heap_p=0.3)
        src = "event:" + e["file"] if e else "cue"
        name = "the slop-mosaic falls away (screen tiles)"
    mx.add("sfx", P(rain, -12), t_f, 1.0, send=0.4, name=name, scene=sc, src=src)
    for q in range(6):
        mx.add("sfx", O.screen_pop(r), t_f + r.uniform(0.1, max(0.2, d)), db(-30), r.uniform(-0.8, 0.8), send=0.3,
               name="slop screens die as they fall", scene=sc)
    e = ev.find(sc, ["glint"], near=word("C02", "rise"), win=1.0, kind_only=True)
    t_g = e["t"] if e else word("C02", "rise")
    glints(mx, r, t_g - 0.1, t_g + 0.8, 5, -36, scene=sc, name="the true gold glints behind ('before the true one can "
           "rise')")
    # RESPONDEO: Crocus raises the Byzantine blessing hand
    t_h, p, _, src = ev.get(sc, ["rustle", "bless", "hand"], word("C03", "summon"), 1.5, kind_only=True)
    mx.add("sfx", L.cloth_rustle(r, 0.55, 0.9, 0.8), t_h, db(-30), pan_of(p, 0.2), send=0.3,
           name="the blessing hand is raised (robes)", scene=sc, src=src)


# ================================================================================================
# m05 · THE SCHISMMANCERS
# ================================================================================================
def src_of(e):
    return "event:" + e["file"] if e.get("file") else "cue"


def m05(mx, ev):
    sc = "m05"
    r = R(505)
    T0 = SCENES[sc]["start"]
    ev.claim_near(sc, T0, 0.05, ["cut"])
    # the frieze: warrior saints shift in their lamellar
    arm = ev.all(sc, ["armour", "armor", "lamellar"], T0, 103.8)
    if not arm:
        arm = [dict(t=T0 + 0.1 + 0.35 * k, pan=[-0.6, 0.5, -0.2][k], strength=0.5, file=None, desc="") for k in range(3)]
    for e in arm:
        d = dur_of(e, 0.4)
        mx.add("sfx", O.lamellar(r, max(0.3, d), 0.4 + 0.6 * e["strength"]), e["t"], db(-24), e["pan"], send=0.35,
               name="lamellar shifts (laced iron plates)", scene=sc, src=src_of(e))
    # K01: the saints lower their goggles in a ripple (servo whine + detent click); then every tube flares
    rip = ev.find(sc, ["goggles_ripple", "ripple"], near=97.0, win=2.0, kind_only=True)
    gg = [e for e in ev.all(sc, ["goggle", "nvg_drop"], T0, 103.8) if "ripple" not in e["kind"]]
    if not gg:
        gg = [dict(t=word("K01", "Actual") + 0.1 + 0.13 * k, pan=-0.85 + 1.7 * k / 8, strength=0.8, file=None)
              for k in range(9)]
    for e in gg:
        mx.add("sfx", O.nvg_drop(r), e["t"], db(-23 + 3 * (e["strength"] - 0.6)), e["pan"], send=0.3,
               name="NVG goggles lowered (servo, detent click)", scene=sc, src=src_of(e) if not rip else src_of(e))
    fl = ev.find(sc, ["nvg_flare", "flare"], near=word("K01", "Target"), win=1.0, kind_only=True)
    t_fl = fl["t"] if fl else word("K01", "Target")
    mx.add("sfx", O.nvg_flare(r, 11), t_fl, db(-28), 0.0, send=0.35, name="every NVG tube flares on 'Target'",
           scene=sc, src="event:" + fl["file"] if fl else "cue")
    # K02: the poles telescope out section by section (clack) and the flags unfurl (cloth: physics + snaps)
    tel = ev.all(sc, ["telescop"], 100.5, 104.0)
    if tel:
        for e in tel:
            m = re.search(r"section (\d+)\s*/\s*(\d+)", e["desc"])
            if m:
                snd, lv = O.telescope_section(r, int(m.group(1)) - 1, int(m.group(2))), -21
            else:
                snd, lv = L.telescope(r, 3, 0.11), -18
            mx.add("sfx", snd, e["t"], db(lv) * (0.6 + 0.4 * e["strength"]), e["pan"], send=0.3,
                   name="a flag-lance telescopes out (section clacks home)", scene=sc, src=src_of(e))
    else:
        for k in range(9):
            mx.add("sfx", L.telescope(r, 3, 0.11), word("K02", "recursive") + 0.09 * k, db(-18), -0.85 + 1.7 * k / 8,
                   send=0.3, name="a flag-lance telescopes out", scene=sc, src="cue")
    unf = ev.all(sc, ["unfurl"], 100.5, 104.5)
    if not unf:
        unf = [dict(t=word("K02", "balkanization") + 0.07 * k, pan=-0.85 + 1.7 * k / 8, strength=0.8, file=None)
               for k in range(9)]
    for e in unf:   # the unrolling swish; the snaps themselves come from each flag's cloth simulation
        mx.add("sfx", Snd(FF.raise_grain(r, 0.15, False), 0.12), e["t"], db(-20), e["pan"], send=0.25,
               name="a flag unrolls over the mosaic (swish)", scene=sc, src=src_of(e))
    # K03 "Breach! Breach! Breach!": three stamps as one, then the charge
    st = ev.all(sc, ["step", "charge", "foot", "stamp"], 103.5, 105.9)
    ev.claim_near(sc, 104.2, 0.8, ["punch_in"])
    t_calm = ev.get(sc, ["calm"], 105.17, 0.8, kind_only=True)
    steps = []
    for e in st:
        dsc = e["desc"].lower()
        if e["kind"] == "charge" or "charge" in dsc:
            d = min(dur_of(e, 105.75 - e["t"]), t_calm[0] - e["t"]) if t_calm[3] != "cue" else dur_of(e, 1.2)
            for g in range(6):
                ph = r.uniform(0.1, 0.3)
                steps += [(e["t"] + ph + 0.26 * q * r.uniform(0.96, 1.04), -0.8 + 1.6 * g / 5, 0.75, None)
                          for q in range(max(1, int(d / 0.26)))]
        elif "as one" in dsc or "stamp" in dsc or "eleven" in dsc:
            mx.add("sfx", O.massed_steps(r, 11, 0.012, (-0.9, 0.9), "boot", True), e["t"], db(-12 + 3 * (e["strength"] - 0.8)),
                   send=0.35, name="eleven armoured saints stamp forward as one", scene=sc, src=src_of(e))
        else:
            steps.append((e["t"], e["pan"], e["strength"], e))
    if not st:
        for q, tw in enumerate([word("K03", "Breach", 0), word("K03", "Breach", 1), word("K03", "Breach", 2)]):
            mx.add("sfx", O.massed_steps(r, 11, 0.012), tw, db(-13 + q), send=0.35,
                   name="the saints stamp forward as one", scene=sc)
        for g in range(6):
            steps += [(104.65 + r.uniform(0.1, 0.3) + 0.26 * q, -0.8 + 1.6 * g / 5, 0.75, None) for q in range(2)]
    for t, p, s, e in steps:
        if t >= min(t_calm[0], 105.78):
            continue
        mx.add("sfx", O.stone_step(r, 0.7 + 0.3 * s, "boot"), t, db(-22), p, send=0.3, name="boots charge on marble",
               scene=sc, src=src_of(e) if e else "design")
        mx.add("sfx", O.lamellar(r, 0.22, 0.8), t + 0.01, db(-29), p, send=0.3, name="lamellar jingles on the stride",
               scene=sc)
    # dead calm: the flags fall limp; light leaks through every joint; the wall creaks and ticks; the air sucks back
    if t_calm[3] != "cue":
        e = ev.find(sc, ["calm"], near=t_calm[0], win=0.05, kind_only=True, claim=False)
        d = dur_of(e, 105.8 - t_calm[0]) if e else 105.8 - t_calm[0]
        mx.add("sfx", O.wall_stress(r, d), t_calm[0], db(-16), send=0.4,
               name="dead calm: the wall creaks and ticks, light leaking through the joints", scene=sc, src=t_calm[3])
    t_in = ev.get(sc, ["inhale", "suck"], CUES["breach"] - 0.25, 0.5, kind_only=True)
    mx.add("sfx", L.swell(r, CUES["breach"] - t_in[0], 150, 5000, 2.5), CUES["breach"], db(-20), send=0.2,
           name="the pre-blast suck-back (air drawn in)", scene=sc, src=t_in[3])
    # 105.8 BREACH: the wall of the mosaic blasts inward; a storm of tesserae at the camera; flag-yellow light
    t_b, p, _, src = ev.get(sc, ["breach", "blast"], CUES["breach"], 0.5, kind_only=True)
    mx.add("sfx", L.explosion(r, 1.3, 4.0, "stone", 40), t_b, db(-3), pan_of(p, 0.0), send=0.3,
           name="BREACH: the mosaic wall blasts inward", scene=sc, cue="breach", src=src)
    e = ev.find(sc, ["tile_storm", "storm"], near=t_b + 0.05, win=0.8, kind_only=True)
    t_s = e["t"] if e else t_b + 0.02
    mx.add("sfx", P(O.tile_storm(r, dur_of(e, 106.5 - t_s), 4000, 0.3), -10), t_s, 1.0, send=0.3,
           name="a storm of tesserae hurled at the camera", scene=sc, src="event:" + e["file"] if e else "design")
    k = 90
    tl_, vl, pl = O.fall_plan(r, t_b + r.exponential(0.2, k), r.uniform(0.5, 6.0, k), r.uniform(-1, 1, k))
    for q in range(5):
        mx.add("sfx", O.chunk_thud(r, r.uniform(0.8, 1.6)), t_b + 0.25 + r.uniform(0, 0.5), db(-16), r.uniform(-0.8, 0.8),
               send=0.35, name="slabs of mosaic land", scene=sc)
    mx.add("sfx", P(O.tile_rain(r, t_b, 2.5, tl_, vl, pl), -14), t_b, 1.0, send=0.35,
           name="tesserae rain down after the blast", scene=sc)
    e = ev.find(sc, ["flags_whoosh", "overhead", "fly over"], near=t_b + 0.05, win=0.8)
    t_o = e["t"] if e else t_b + 0.05
    d = dur_of(e, 0.55)
    for q in range(11):
        mx.add("sfx", Snd(FF.raise_grain(r, 0.0, q % 3 == 0), 0.12), t_o + d * q / 11, db(-17), -0.6 + 1.2 * q / 10,
               send=0.25, name="eleven flags fly over the camera", scene=sc, src="event:" + e["file"] if e else "design")
    mx.add("sfx", L.whoosh(r, d + 0.5, d * 0.6, 200, 3000, 1.0, 0.4, -0.4, dark=0.2), t_o + d * 0.6, db(-18), send=0.25,
           name="cloth over the camera (air)", scene=sc)
    e = ev.find(sc, ["glow"], near=106.15, win=0.6, kind_only=True)
    mx.add("sfx", L.swell(r, 106.5 - t_b, 400, 12000, 1.6), 106.5, db(-20), send=0.3,
           name="flag-yellow light floods the frame (air swell)", scene=sc, src="event:" + e["file"] if e else "cue")


# ================================================================================================
# m06 · RECURSIVE BALKANIZATION
# ================================================================================================
def crack_schedule(ev, sc, t0, t1, fallback):
    cr = ev.all(sc, ["crack"], t0, t1)
    if cr:
        out = []
        for k, e in enumerate(cr):
            lv = level_of(e["desc"])
            out.append((e["t"], e["pan"], e["strength"], k if lv is None else lv, "event:" + e["file"]))
        return out
    return [(t, p, 1.0, lv, "cue") for t, p, lv in fallback]


def m06(mx, ev):
    sc = "m06"
    r = R(606)
    bank = O.tile_bank()
    # 106.5 from the glow onto the mappa mundi (air settles)
    mx.add("sfx", L.swell(r, 0.9, 300, 6000, 1.6).x[::-1].copy(), 106.5, db(-32), send=0.3,
           name="the glow settles onto the mappa mundi", scene=sc)
    # 106.9 NATIONS: the map cracks along its borders, then every piece again and again (pitch rises per generation)
    fb, t, lv = [], word("K04", "nation"), 0
    for g, (tg, cnt) in enumerate([(107.15, 1), (107.72, 2), (108.12, 4), (108.45, 6), (108.72, 8), (108.95, 10),
                                   (109.15, 12)]):
        fb += [(tg + r.uniform(0, 0.12) * (g > 0), r.uniform(-0.8, 0.8) if g else 0.0, g) for _ in range(cnt)]
    for t, p, s, lv, src in crack_schedule(ev, sc, 106.5, 113.0, fb):
        mx.add("sfx", O.bed_crack(r, 0.45, lv, True), t, db(-14 - 1.8 * lv) * (0.6 + 0.4 * s), p, send=0.4,
               name=f"the map cracks along its borders (generation {lv + 1}: shorter, higher)", scene=sc,
               cue="nations" if lv == 0 else None, src=src)
    for e in ev.all(sc, ["drift"], 106.5, 113.7) or [dict(t=108.3, desc="(to 111.5)", file=None)]:
        d = dur_of(e, 3.0)
        mx.add("sfx", L.decorrelate(L.lp(L.stone_slide(r, d, 0.4).x, 1800, 2) * 0.6, r), e["t"], db(-32), send=0.4,
               name="fragments drift apart (slabs scraping over plaster)", scene=sc,
               src="event:" + e["file"] if e.get("file") else "cue")
    sets = ev.all(sc, ["tile_set"], 108.5, 113.6) or [dict(t=r.uniform(109.3, 113.0), pan=r.uniform(-0.9, 0.9),
                                                           strength=0.5, file=None) for _ in range(12)]
    for e in sets:
        mx.add("sfx", O.set_flurry(r, int(10 + 12 * e["strength"]), 0.06, "gold", vel=0.5), e["t"], db(-30), e["pan"],
               send=0.4, name="a tiny kingdom sets itself (one tiny crowned person)", scene=sc,
               src="event:" + e["file"] if e.get("file") else "cue")
    # the camera dives into single tesserae (113.1-113.7, 119.3-120.0)
    for e in ev.all(sc, ["dive"], 106.5, 131.5) or [dict(t=113.1, desc="(to 113.7)", file=None),
                                                     dict(t=119.3, desc="(to 120.0)", file=None)]:
        d = dur_of(e, 0.6)
        mx.add("sfx", L.whoosh(r, d + 0.3, d * 0.8, 200, 5000, 1.2, 0.0, 0.0, dark=0.3), e["t"] + d * 0.8, db(-24),
               send=0.3, name="the camera dives into one tessera", scene=sc,
               src="event:" + e["file"] if e.get("file") else "cue")
    # 113.7 FAITHS: the congregation splits into churches of one; halos split; ANATHEMA SIT flies
    fb = [(word("K05", "faith") + 0.1, 0.0, 0), (word("K05", "cabal"), -0.4, 1), (word("K05", "cabal") + 0.25, 0.4, 1),
          (word("K05", "one") + 0.1, -0.7, 2), (word("K05", "one") + 0.2, 0.7, 2)]
    for t, p, s, lv, src in crack_schedule(ev, sc, 113.6, 119.9, fb):
        mx.add("sfx", O.bed_crack(r, 0.35, lv, True), t, db(-16 - 1.5 * lv), p, send=0.45,
               name="the congregation splits into churches of one", scene=sc, cue="faiths" if lv == 0 else None,
               src=src)
    sets = ev.all(sc, ["tile_set"], 113.6, 119.9) or [dict(t=r.uniform(114.6, 119.5), pan=r.uniform(-0.9, 0.9),
                                                           strength=0.6, file=None) for _ in range(9)]
    for e in sets:
        mx.add("sfx", O.set_flurry(r, int(14 + 14 * e["strength"]), 0.1, "gold", vel=0.55), e["t"], db(-29), e["pan"],
               send=0.45, name="an altar / tiara / dome of one sets itself", scene=sc,
               src="event:" + e["file"] if e.get("file") else "cue")
    hs = ev.all(sc, ["halo"], 113.6, 120.0) or [dict(t=word("X03", "Anathema") + 0.1, pan=-0.5, file=None),
                                                dict(t=word("X04", "anathema") + 0.1, pan=0.5, file=None),
                                                dict(t=word("X05", "pope") + 0.05, pan=0.0, file=None)]
    for e in hs:
        mx.add("sfx", O.halo_split(r), e["t"], db(-18), e["pan"], send=0.5, name="a halo splits in two", scene=sc,
               src="event:" + e["file"] if e.get("file") else "cue")
    fl = ev.all(sc, ["whoosh", "anathema"], 113.6, 120.0) or [dict(t=116.2, pan=-0.4, file=None),
                                                              dict(t=117.35, pan=0.4, file=None)]
    for e in fl:
        p = pan_of(e.get("pan"), 0.0)
        mx.add("sfx", L.whoosh(r, 0.7, 0.35, 300, 3500, 1.2, -p - 0.3 if p > 0 else -0.8, p), e["t"], db(-24),
               send=0.35, name="ANATHEMA SIT flies between them (letter-tiles whoosh)", scene=sc,
               src="event:" + e["file"] if e.get("file") else "cue")
        fw = O.flip_wave(r, 0.0, 0.35, lambda u: np.ones_like(u), lambda u: p + 0 * u, "gold", total=30, spread=0.2)
        mx.add("sfx", fw, e["t"] - 0.1, db(-26), send=0.3, name="letter-tiles rattle in flight", scene=sc)
    # 120.0 PHILOSOPHIES: the floor breaks into floating tesserae drifting into the void
    fb = [(word("K06", "philosophy") + 0.05, 0.0, 0), (word("K06", "crackpot"), -0.5, 1), (121.5, 0.5, 1),
          (122.1, -0.2, 2)]
    for t, p, s, lv, src in crack_schedule(ev, sc, 119.9, 128.5, fb):
        mx.add("sfx", O.bed_crack(r, 0.55, lv, True), t, db(-14 - 1.5 * lv), p, send=0.5, verb="void",
               name="the philosophers' floor breaks apart", scene=sc, cue="philosophies" if lv == 0 else None, src=src)
    fl = ev.all(sc, ["float", "drift"], 119.9, 128.6)
    t_fl = fl[0]["t"] if fl else 120.5
    d = dur_of(fl[0], 7.5) if fl else 7.5
    mx.add("sfx", P(O.drift_clinks(r, d, int(4 * d)), -20), t_fl, 1.0, send=0.6, verb="void",
           name="floating tesserae drift apart (void: they only touch and part)", scene=sc,
           src="event:" + fl[0]["file"] if fl else "cue")
    for q in range(5):
        tq = t_fl + 0.8 + 1.4 * q + r.uniform(0, 0.6)
        mx.add("sfx", L.whoosh(r, 2.2, 1.1, 90, 700, 0.8, r.uniform(-1, 0), r.uniform(0, 1), dark=0.7), tq, db(-34),
               send=0.6, verb="void", name="a floating rock drifts past", scene=sc)
    # tiny_split crackle waves during zooms
    for e in ev.all(sc, ["tiny_split"], 106.5, 128.5):
        d = dur_of(e, 0.6)
        n = ns(d)
        lvl = np.sin(np.pi * np.arange(n) / n) * (0.5 + 0.5 * e["strength"])
        mx.add("sfx", L.pan2(O.split_cascade_bed(r, d, lvl), e["pan"]), e["t"], db(-30), send=0.35,
               name="tesserae subdivide (micro-cleave crackle)", scene=sc, src="event:" + e["file"])
    # 128.6 ENDLESS: every tessera splits into four, recursively (an octave higher per generation) -> dust
    sp = ev.all(sc, ["split"], 128.0, 131.5)
    gens = sorted({(e["t"], level_of(e["desc"]) or k) for k, e in enumerate(sp)}, key=lambda x: x[0])
    if gens:
        gt = [t for t, _ in gens]
        src = "event:" + sp[0]["file"]
    else:
        t0 = word("K07", "schism")
        gt = list(t0 + np.cumsum([0.0, 0.42, 0.34, 0.28, 0.23, 0.19, 0.16, 0.13, 0.11]))
        src = "cue"
    q, counts = O.quadtree(r, gt, 4, 5000, bank=bank)
    mx.add("sfx", q, gt[0], db(-8), send=0.45, name="ENDLESS: every tessera splits into four, and again "
           "(each generation an octave higher, 6 dB softer per piece)", scene=sc, cue="endless", src=src)
    sw = ev.find(sc, ["swirl", "dust"], near=gt[-1], win=2.0, kind_only=True)
    t_sw = sw["t"] if sw else gt[min(5, len(gt) - 1)]
    d = 132.4 - t_sw
    n = ns(d)
    tg = t_sw + np.arange(n) / SR
    lvl = np.interp(tg, [t_sw, t_sw + 0.8, 131.3, 131.8, 132.4], [0.0, 1.0, 1.0, 0.8, 0.0])
    mx.add("sfx", O.dust_hiss(r, d, lvl, 0.45), t_sw, db(-24), send=0.3,
           name="glittering dust swirls (the ultrasonic splits' residue)", scene=sc, src="event:" + sw["file"] if sw else "cue")


# ================================================================================================
# m07 · BABEL
# ================================================================================================
def m07(mx, ev):
    sc = "m07"
    r = R(707)
    T = CUES["babel_rise"]
    mx.add("sfx", L.whoosh(r, 2.6, 1.6, 150, 2200, 0.8, -0.6, 0.6, dark=0.5), T + 1.4, db(-26), send=0.4,
           name="the dust spirals up into the tower", scene=sc, cue="babel_rise")
    tiers = ev.all(sc, ["tier"], T, 141.7) or [dict(t=T + 0.5 + 0.34 * k, pan=r.uniform(-0.4, 0.4), strength=1 - 0.06 * k,
                                                    file=None) for k in range(7)]
    for e in tiers:
        mx.add("sfx", O.set_flurry(r, 70, 0.28, "screen", vel=0.8), e["t"], db(-17), e["pan"], send=0.45,
               name="a tier of Babel sets (bricks of screens and slop tiles)", scene=sc,
               src="event:" + e["file"] if e.get("file") else "cue")
        mx.add("sfx", L.thud(r, 55.0, 0.5, 1.0, 500), e["t"], db(-18), e["pan"], send=0.4, name="tier thud", scene=sc)
        buzz = L.hum(r, 0.5, 55.0, (0.3, 0.5, 0.3, 0.2), 0.3) * L.env_ad(ns(0.5), 0.005, 0.35)
        mx.add("sfx", buzz, e["t"] + 0.03, db(-34), e["pan"], name="its screens buzz alight", scene=sc)
    for q in range(5):
        mx.add("sfx", L.creak(r, 2.4, 8, 20, (40, 61, 92, 137, 205), 0.5, 0.6), 134.6 + 1.4 * q + r.uniform(0, 0.6),
               db(-26), r.uniform(-0.7, 0.7), send=0.45, name="the tower groans", scene=sc)
    t_eg, p, _, src = ev.get(sc, ["egregore"], CUES["egregore"], 1.0, kind_only=True)
    mx.add("sfx", L.swell(r, 1.2, 150, 7000, 1.8), t_eg + 0.3, db(-24), send=0.4,
           name="Lord Egregore fills the dome (static swell)", scene=sc, cue="egregore", src=src)
    mx.add("sfx", L.sub_boom(r, 55, 36.7, 3.0, 0.5), t_eg + 0.3, db(-16), name="Egregore's presence (sub on D)",
           scene=sc)
    e = ev.find(sc, ["egregore_lit", "billboard", "lit"], near=135.4, win=1.5, kind_only=True)
    if e:   # every billboard and the loading-wheel halo light up: a cascade of screens powering on
        k = 16
        for q in range(k):
            tq = e["t"] + 0.5 * (q / k) ** 0.8 + r.uniform(0, 0.03)
            mx.add("sfx", O.flip_grain(r, "smalti", True), tq, db(-26) * r.uniform(0.6, 1.0), r.uniform(-0.9, 0.9),
                   send=0.35, name="the billboards light up (screens flip on)", scene=sc, src="event:" + e["file"])
        hm = L.hum(r, 1.2, 55.0, (0.2, 0.5, 0.3, 0.3, 0.2), 0.4) * L.env_pts(ns(1.2), [(0, 0), (0.5, 1), (1.2, 0)])
        mx.add("sfx", L.decorrelate(hm, r), e["t"], db(-30), send=0.3, name="their electric surge", scene=sc,
               src="event:" + e["file"])
    avalanche(mx, ev, r)


def avalanche(mx, ev, r):
    """141.8 -> 145.0: the tower topples, then the ENTIRE mosaic falls off the wall. Every tessera is simulated:
    detach -> free fall -> first contact (floor, later the growing heap) -> bounces -> rocking settle / skid."""
    sc = "m07"
    T, p, _, src_c = ev.get(sc, ["topple", "collapse"], CUES["collapse"], 0.6, kind_only=True)
    tr = TRACKS.get((sc, "avalanche"))
    slabs = []
    if tr is not None and "land" in tr:
        fps, f0 = tr["fps"], tr["f0"] / tr["fps"]
        tl_, pl = O.counts_to_times(r, f0, fps, tr["land"], tr["pan"], 1.0, 60000)
        keep = tl_ < SIL0 - 0.02
        tl_, pl = tl_[keep], np.clip(pl[keep] + r.normal(0, 0.35, keep.sum()), -1, 1)
        vl = np.sqrt(2 * O.G * r.uniform(1.0, 13.0, len(tl_)))
        det = O.counts_to_times(r, f0, fps, tr.get("fall", tr["land"]), tr["pan"], 1.0, 60000)[0]
        src = "physics:" + tr["file"]
        for q in range(50):
            k = int(r.integers(0, len(tl_))) if len(tl_) else 0
            if len(tl_):
                slabs.append((tl_[k], pl[k], r.uniform(0.6, 2.0)))
    else:
        # the mosaic sheds in sheets: each panel of set tesserae lets go as a burst from one place on the wall;
        # its tiles fall from about the same height, so they land as a wave; the bursts overlap into the avalanche
        N, n_sh = 42000, 34
        sh_t = np.sort(T + 0.25 + 1.75 * r.beta(1.5, 2.4, n_sh))       # most sheets go early; the last land ~144.8
        sh_n = r.dirichlet(np.ones(n_sh) * 1.5) * N
        sh_p, sh_h = r.uniform(-0.95, 0.95, n_sh), r.uniform(3.0, 13.0, n_sh)
        det, h, pn = [], [], []
        for k in range(n_sh):
            m = int(sh_n[k])
            det.append(sh_t[k] + r.gamma(2.0, r.uniform(0.15, 0.6) / 4, m))
            h.append(np.clip(sh_h[k] + r.normal(0, 1.5, m), 0.8, 14.0))
            pn.append(np.clip(sh_p[k] + r.normal(0, 0.12, m), -1, 1))
            if r.random() < 0.65:                       # part of the sheet stays bonded: a slab hits first
                slabs.append((sh_t[k] + np.sqrt(2 * sh_h[k] / O.G), sh_p[k], r.uniform(0.8, 2.2)))
        det, h, pn = np.concatenate(det), np.concatenate(h), np.concatenate(pn)
        k = det < 144.3
        tl_, vl, pl = O.fall_plan(r, det[k], h[k], pn[k])
        keep = tl_ < 144.88
        tl_, vl, pl, det = tl_[keep], vl[keep], pl[keep], det[k][keep]
        src = "design"
    t0 = T - 0.2
    dur = SIL0 - t0
    # the growing heap: the more tiles have landed, the more likely a tile lands on tiles instead of marble
    order = np.argsort(tl_)
    frac = np.empty(len(tl_))
    frac[order] = np.arange(len(tl_)) / max(1, len(tl_))
    heap_p = np.clip(frac * 1.6, 0, 0.85)
    rain = O.tile_rain(r, t0, dur, tl_, vl, pl, heap_p=heap_p, mats=("smalti", 0.4, "gold", 0.3, "stone", 0.15,
                                                                      "screen", 0.15))
    pk = O.rms(rain[ns(143.2 - t0):ns(144.0 - t0)])
    mx.add("sfx", Snd(rain * (db(-16) / max(pk, 1e-9)), T - t0), T, 1.0, send=0.35,
           name=f"THE AVALANCHE: {len(tl_)} tesserae land, bounce, skid, settle (floor, then the heap)", scene=sc,
           cue="collapse", src=src, noduck=True)
    # slabs still bonded by mortar
    for tq, pq, sz in slabs:
        if tq > 144.6:
            continue
        mx.add("sfx", O.chunk_thud(r, sz), tq, db(-17) * r.uniform(0.5, 1.0), pq,
               send=0.4, name="slabs of mosaic (tiles still in their mortar) hit the floor", scene=sc)
    # the pour: the air and the grains cascading down the wall face (~ tiles in flight); the floor's roar (~ landings)
    n = ns(dur)
    tg = t0 + np.arange(n) / SR
    inflight = np.clip(np.searchsorted(np.sort(det), tg) - np.searchsorted(np.sort(tl_), tg), 0, None).astype(float)
    flow = L.lp(L.f32(inflight), 8, 1).astype(np.float64).clip(0, None)
    pb = O.pour_bed(r, dur, flow * 4.0)
    mx.add("sfx", O.rms_norm(pb, db(-27)), t0, 1.0, send=0.3,
           name="the pour: tesserae streaming down the wall face (~ tiles in flight)", scene=sc)
    air = O.swarm_air(r, dur, np.sqrt(flow / max(1.0, flow.max())), None)
    mx.add("sfx", O.rms_norm(air, db(-29)), t0, 1.0, name="the air of the falling mass", scene=sc)
    hist, edges = np.histogram(tl_, bins=int(dur * 40), range=(t0, SIL0))
    rate = np.interp(tg, 0.5 * (edges[1:] + edges[:-1]), hist * 40.0)
    lvl = np.sqrt(np.clip(rate / max(1.0, rate.max()), 0, 1))
    roar = L.stereo(L.svf(L.brown(n, r), 110, 0.7, "lp"), L.svf(L.brown(n, r), 110, 0.7, "lp")) * L.f32(lvl)[:, None]
    mx.add("sfx", O.rms_norm(roar, db(-22)), t0, 1.0, name="the floor's roar under the landing mass", scene=sc)
    # the mortar tearing off the plaster (detachment crackle, rate ~ detachment)
    hist, edges = np.histogram(det, bins=int(dur * 40), range=(t0, SIL0))
    drate = np.interp(tg, 0.5 * (edges[1:] + edges[:-1]), hist * 40.0)
    exc = L.poisson_clicks(r, n, np.clip(drate * 0.08, 0, 4000), 1.0)
    tear = L.resonators(exc, r.uniform(500, 2600, 6), r.uniform(0.003, 0.01, 6), [0.6] * 6) + L.hp(exc, 2500) * 0.5
    mx.add("sfx", L.decorrelate(O.rms_norm(tear, db(-28)), r), t0, 1.0, send=0.3,
           name="the setting bed tears off the plaster (detachment crackle)", scene=sc)
    # the tower: a groan, the crack, its top hinging over, its fall, the impact
    mx.add("sfx", L.creak(r, 1.2, 6, 30, (36, 55, 82, 124, 186), 0.6, 0.8), T - 0.9, db(-18), 0.0, send=0.4,
           name="the tower's last groan", scene=sc)
    mx.add("sfx", O.bed_crack(r, 0.3, 0, True), T, db(-6), pan_of(p, 0.0), send=0.4, name="THE TOWER CRACKS",
           scene=sc, cue="collapse", src=src_c)
    mx.add("sfx", Snd(L.shock(r, 1.2), 0.0), T, db(-8), name="crack shock", scene=sc)
    tp = ev.find(sc, ["topple"], near=T, win=0.01, kind_only=True, claim=False)
    d_top = dur_of(tp, 0.7) if tp else 0.7
    mx.add("sfx", L.creak(r, d_top + 0.3, 14, 5, (30, 46, 69, 103, 155), 0.7, 0.9), T + 0.05, db(-16), pan_of(p, 0.0),
           send=0.45, name="its top hinges over (a slow wrenching groan)", scene=sc, src=src_c)
    ev.claim_near(sc, T, 0.1, ["avalanche"])
    e = ev.find(sc, ["tower_impact", "impact"], near=T + 1.0, win=2.5, kind_only=True)
    t_i, p_i = (e["t"], e["pan"]) if e else (T + 0.55, 0.3)
    mx.add("sfx", L.whoosh(r, t_i - T + 0.4, t_i - T, 60, 900, 0.7, p_i - 0.5, p_i, dark=0.8), t_i, db(-18), send=0.4,
           name="the tower falls (air)", scene=sc)
    for q in range(6):
        mx.add("sfx", O.chunk_thud(r, 2.5), t_i + 0.03 * q, db(-10), float(np.clip(p_i + r.uniform(-0.4, 0.4), -1, 1)),
               send=0.4, name="the tower hits the floor", scene=sc, src="event:" + e["file"] if e else "design")
    for q in range(22):
        mx.add("sfx", O.screen_pop(r), t_i + r.gamma(1.5, 0.12), db(-24), r.uniform(-0.9, 0.9), send=0.3,
               name="its screens die", scene=sc)
    mx.add("sfx", L.sub_boom(r, 50, 36.7, 2.5, 0.4), t_i, db(-6), name="tower impact sub (D)", scene=sc)
    e = ev.find(sc, ["static_drain", "drain"], near=T + 0.2, win=1.0, kind_only=True)
    t_d = e["t"] if e else T + 0.05
    mx.add("sfx", O.static_drain(r, dur_of(e, 0.9)), t_d, db(-18), send=0.2, name="Egregore's static face drains away",
           scene=sc, src="event:" + e["file"] if e else "design")
    # the very last tessera lands, alone - and settles; then the silence
    e = ev.find(sc, ["last_tile", "last"], near=144.55, win=0.4, kind_only=True)
    ev.claim_near(sc, 144.86, 0.2, ["avalanche_end", "end"])
    t_l, p_l = (e["t"], e["pan"]) if e else (144.5, -0.7)
    tt_, vv, ss, _, ids = O.bounce_events(r, np.array([t_l]), np.array([6.5]), np.array(["floor"]), vmin=0.5)
    out = np.zeros((ns(0.8), 2), np.float32)
    O.render_contacts(out, t_l, O.tile_bank(), r, tt_, vv, ss, "gold", p_l)
    mx.add("sfx", P(out, -16), t_l, 1.0, send=0.5, name="the very last tessera lands, alone, and settles", scene=sc,
           src="event:" + e["file"] if e else "design")


# ================================================================================================
# m08 · THE UNBABELING
# ================================================================================================
def m08(mx, ev):
    sc = "m08"
    r = R(808)
    bank = O.tile_bank()
    # one late tessera slides off a heap and ticks across the floor (the first sound after the silence)
    tl_, vl, sfc = np.array([147.25]), np.array([2.2]), np.array(["floor"])
    t, v, s, _, ids = O.bounce_events(r, tl_, vl, sfc)
    out = np.zeros((ns(2.0), 2), np.float32)
    O.render_contacts(out, 147.2, bank, r, t, v, s, "gold", 0.45)
    mx.add("sfx", out, 147.2, db(-6), send=0.6, name="one late tessera slides off a heap and ticks to rest", scene=sc)
    # Crocus crosses the heaps
    steps = ev.all(sc, ["step", "foot"], SIL1, 149.0) or [dict(t=t_, pan=p_, strength=0.7, file=None) for t_, p_ in
                                                            ((147.55, -0.35), (148.05, -0.2), (148.52, -0.08))]
    for e in steps:
        mx.add("sfx", O.stone_step(r, 0.8, "heap"), e["t"], db(-22), e["pan"], send=0.45,
               name="Crocus steps through the heaps of fallen tesserae", scene=sc,
               src="event:" + e["file"] if e.get("file") else "cue")
        mx.add("sfx", L.cloth_rustle(r, 0.4, 0.6, 0.7), e["t"] - 0.05, db(-36), e["pan"], name="robes", scene=sc)
    # 149.0 flag_planted: the Flag is planted in the heap - thud; then tiles scatter down the heap
    t_p, p, _, src = ev.get(sc, ["plant"], CUES["flag_planted"], 0.6, kind_only=True)
    mx.add("sfx", O.plant_in_heap(r), t_p, db(-5), pan_of(p, 0.0), send=0.5,
           name="THE FLAG IS PLANTED in the heap of tesserae (pole thunk, heap crunch, D sub)", scene=sc,
           cue="flag_planted", src=src)
    for e in ev.all(sc, ["scatter"], t_p, 153.0):
        k = 30
        tl2, vl2, pl2 = O.fall_plan(r, e["t"] + r.exponential(0.3, k), r.uniform(0.05, 0.6, k),
                                    np.clip(e["pan"] + r.normal(0, 0.3, k), -1, 1))
        mx.add("sfx", P(O.tile_rain(r, e["t"], 2.0, tl2, vl2, pl2, heap_p=0.5), -18), e["t"], 1.0, send=0.5,
               name="tiles scatter down the heap", scene=sc, src="event:" + e["file"])
    e = ev.find(sc, ["furled", "raises"], near=148.35, win=1.0)
    if e:
        mx.add("sfx", Snd(FF.raise_grain(r, 0.2, False), 0.12), e["t"], db(-26), e["pan"], send=0.4,
               name="Crocus raises the furled Flag overhead (cloth swish)", scene=sc, src="event:" + e["file"])
    e = ev.find(sc, ["unfurl"], near=149.3, win=0.8, kind_only=True)
    if e:
        mx.add("sfx", Snd(FF.raise_grain(r, 0.1, False), 0.12), e["t"], db(-26), e["pan"], send=0.4,
               name="the cloth unrolls from the pole (swish; its flutter is the cloth sim)", scene=sc,
               src="event:" + e["file"])
    rise(mx, ev, r)
    # 162.0-163.0 tilt up into the dome
    e = ev.find(sc, ["camera", "tilt"], near=162.0, win=1.0, kind_only=True)
    t_c = e["t"] if e else 162.0
    mx.add("sfx", L.whoosh(r, 1.8, 1.0, 120, 1500, 0.7, 0.0, 0.0, dark=0.6, shape=1.5), t_c + 1.0, db(-32), send=0.4,
           name="tilt up into the dome (air)", scene=sc, src="event:" + e["file"] if e else "cue")


def rise(mx, ev, r):
    """155.0 -> 159.0: the fallen tesserae rise like a reversed avalanche and set themselves into a new picture."""
    sc = "m08"
    T = ev.get(sc, ["tiles_rise", "rise"], CUES["tiles_rise"], 0.8, kind_only=True)
    TC = ev.get(sc, ["convergence"], CUES["convergence"], 0.6, kind_only=True)
    tr = TRACKS.get((sc, "rise"))
    if tr is not None and "lift" in tr and "land" in tr:
        fps, f0 = tr["fps"], tr["f0"] / tr["fps"]
        lift_t, lift_p = O.counts_to_times(r, f0, fps, tr["lift"], tr["pan"], 1.0, 30000)
        land_t, land_p = O.counts_to_times(r, f0, fps, tr["land"], tr["pan"], 1.0, 30000)
        lift_p = np.clip(lift_p + r.normal(0, 0.4, len(lift_p)), -1, 1)
        land_p = np.clip(land_p + r.normal(0, 0.4, len(land_p)), -1, 1)
        pre = land_t < TC[0] - 0.06                   # the last landings ARE the convergence wave (set as one)
        conv_n = 1500
        land_t, land_p = land_t[pre], land_p[pre]
        src = "physics:" + tr["file"]
    else:
        N = 24000
        lift_t = O.sample_rate_curve(r, T[0], 158.4, lambda u: np.interp(u, [T[0], T[0] + 0.5, 156.3, 157.6, 158.4],
                                                                        [0.1, 0.6, 1.0, 0.8, 0.0]), N)
        lift_p = np.clip(r.normal(0, 0.6, N), -1, 1)
        fly = r.uniform(0.6, 1.9, N)
        land_t = lift_t + fly
        late = land_t >= TC[0] - 0.06
        conv_n = 1500
        land_t, land_p = land_t[~late], np.clip(lift_p[~late] * 0.7 + r.normal(0, 0.3, (~late).sum()), -1, 1)
        src = "design"
    t0 = 152.8
    ir = L.make_ir(**O.IR_SPECS["bare"])
    lay = O.rise_layers(r, t0, TC[0], lift_t, lift_p, land_t, land_p, conv_n, verb_ir=ir)
    rv = lay["rev"]
    mx.add("sfx", Snd(rv * (db(-20) / max(1e-9, O.rms(rv[ns(155.2 - t0):ns(156.8 - t0)]))), T[0] - t0), T[0], 1.0,
           name="the avalanche UN-HAPPENS: every rising tile's bounce train reversed (with its pre-verb)", scene=sc,
           cue="tiles_rise", src=src)
    mx.add("sfx", O.rms_norm(lay["air"], db(-30)), t0, 1.0, send=0.3, name="the swarm's air as the tesserae fly up",
           scene=sc)
    st = lay["set"]
    mx.add("sfx", st * (db(-24) / max(1e-9, O.rms(st[ns(156.5 - t0):ns(158.8 - t0)]))), t0, 1.0, send=0.45,
           name="each tessera sets itself into the wall (tick over wet tup)", scene=sc, src=src)
    cv = lay["conv"]
    mx.add("sfx", Snd(cv * (db(-3) / max(1e-9, float(np.abs(cv).max()))), TC[0] - t0), TC[0], 1.0, send=0.55,
           name=f"CONVERGENCE: the last {conv_n} tesserae set in one wave", scene=sc, cue="convergence", src=TC[3],
           noduck=True)
    mx.add("sfx", L.thud(r, 73.4, 0.9, 0.8, 400), TC[0], db(-12), send=0.4, name="the whole wall seats (low D)",
           scene=sc, cue="convergence")
    glints(mx, r, TC[0] + 0.05, TC[0] + 2.2, 10, -34, scene=sc, name="the completed gold wakes (glints)")
    # each niche completes and its Flag springs open (a little cloth pop per Flag)
    for e in ev.all(sc, ["flag_pop", "pop"], 154.5, 163.0):
        mx.add("sfx", O.flag_pop(r, e["strength"]), e["t"], db(-29 + 6 * (e["strength"] - 0.35)), e["pan"], send=0.4,
               name="a niche completes: its Flag springs open", scene=sc, src="event:" + e["file"])
    # 'All ... together': the gold wave rolls out of the Flag's foot into the convergence
    e = ev.find(sc, ["together"], near=157.85, win=1.0, kind_only=True)
    if e:
        d = dur_of(e, TC[0] - e["t"])
        w = O.flip_wave(r, 0.0, d, lambda u: (u / d) ** 1.5, lambda u: pan_of(e["pan"], 0.2) * (1 - u / d) +
                        np.sin(7 * u) * (u / d), "gold", total=int(500 * d), gain=0.7, spread=0.5)
        mx.add("sfx", O.win_norm(w, -28.0), e["t"], 1.0, send=0.5,
               name="'All ... together': a gold wave rolls out of the Flag's foot", scene=sc, src="event:" + e["file"])
        mx.add("sfx", L.swell(r, d, 300, 9000, 2.0), TC[0], db(-28), send=0.4, name="the gold wave's air", scene=sc)
    ev.claim_near(sc, TC[0], 0.1, ["complete"])
    ev.all(sc, ["land", "alone"], 154.5, 159.5)     # claim: progress markers (the rise track carries every landing)
    # the heaps tremble before they rise
    e = ev.find(sc, ["tremble"], near=153.4, win=1.5)
    t_tr = e["t"] if e else 153.4
    d = dur_of(e, T[0] - t_tr + 0.3)
    k = 900
    tt_ = O.sample_rate_curve(r, t_tr, t_tr + d, lambda u: ((u - t_tr) / d) ** 2, k)
    out = np.zeros((ns(d + 0.5), 2), np.float32)
    O.render_contacts(out, t_tr, O.tile_bank(), r, tt_, r.uniform(0.3, 1.2, k), np.full(k, "heap"),
                      np.where(r.random(k) < 0.5, "gold", "smalti"), np.clip(r.normal(0, 0.6, k), -1, 1))
    mx.add("sfx", out, t_tr, db(-12), send=0.5, name="the heaps tremble", scene=sc,
           src="event:" + e["file"] if e else "design")


# ================================================================================================
# m09 · THE DOME OF FLAGISTAN
# ================================================================================================
def m09(mx, ev):
    sc = "m09"
    r = R(909)
    mx.add("sfx", O.pigeon_flap(r, 8), 163.45, db(-34), 0.55, send=0.7, name="a pigeon leaves a ledge high in the dome",
           scene=sc)
    for e in ev.all(sc, ["rays", "ray"], 162.8, 170.0) or [dict(t=163.2, desc="(to 165.4)", pan=0.0, file=None)]:
        mx.add("sfx", B.light_sweep(r, dur_of(e, 2.0)), e["t"], db(-32), pan_of(e.get("pan"), 0.0), send=0.6,
               name="rays of gold descend from the oculus (air)", scene=sc,
               src="event:" + e["file"] if e.get("file") else "cue")
    for e in ev.all(sc, ["shimmer", "glide", "bloom", "twinkle", "glint", "chime", "ring", "swell"], 163.0, 180.35):
        k = e["kind"]
        if "glide" in k or "ring" in k:
            mx.add("sfx", B.light_sweep(r, dur_of(e, 0.8)), e["t"], db(-38 + 4 * e["strength"]), e["pan"], send=0.6,
                   name="light sweeps through the tiling (air)", scene=sc, src="event:" + e["file"])
        elif "swell" in k:
            d = dur_of(e, 1.2)
            mx.add("sfx", L.swell(r, d, 300, 8000, 2.0), e["t"] + d, db(-34), send=0.5,
                   name="the oculus swells with light (air)", scene=sc, src="event:" + e["file"])
        elif "chime" in k:
            mx.add("sfx", O.glint(r, 1.3), e["t"], db(-38 + 5 * e["strength"]), e["pan"], send=0.65,
                   name="a drum window catches the light (glass glint)", scene=sc, src="event:" + e["file"])
        else:
            mx.add("sfx", O.glint(r), e["t"], db(-40 + 5 * e["strength"]), e["pan"], send=0.65,
                   name="tesserae glint on the hymn", scene=sc, src="event:" + e["file"])
    for e in ev.all(sc, ["raise"], 163.0, 180.35):   # the peoples' poles slide up round the dome: far, soft clacks
        d = dur_of(e, 2.0)
        for q in range(18):
            tq = e["t"] + d * (q / 18) ** 0.8 + r.uniform(0, 0.05)
            mx.add("sfx", O.telescope_section(r, int(r.integers(0, 3))), tq, db(-38) * r.uniform(0.5, 1.0),
                   r.uniform(-0.9, 0.9), send=0.7, name="the peoples' flag-poles slide up round the dome (far)",
                   scene=sc, src="event:" + e["file"])
    # 180.4 the gateless gate: heaven dissolves (tesserae let go and drift down), earth falls inward; the flags remain
    ds = ev.all(sc, ["dissolve"], 179.5, 186.0) or [dict(t=180.5, desc="(to 184.9)", pan=0.0, strength=0.8, file=None)]
    for k, e in enumerate(ds):
        t_d, d = e["t"], dur_of(e, 4.4)
        if "fall" in e["desc"].lower():      # arches, pendentives, drum fall inward: a soft rain of tesserae
            m = int(900 * d)
            tl_, vl, pl = O.fall_plan(r, O.sample_rate_curve(r, t_d, t_d + d, lambda u: np.ones_like(u), m),
                                      r.uniform(0.3, 2.5, m), np.clip(r.normal(0, 0.7, m), -1, 1))
            rain = O.tile_rain(r, t_d, d + 0.5, tl_, vl, pl, heap_p=0.5, mats=("gold", 0.6, "smalti", 0.4))
            mx.add("sfx", O.win_norm(rain, -34.0), t_d, 1.0, send=0.6, name="earth falls inward: a soft rain of tesserae",
                   scene=sc, src=src_of(e))
            continue
        mx.add("sfx", P(O.drift_clinks(r, d, int(14 * d)), -22), t_d, 1.0, send=0.6,
               name="heaven and earth drift apart into tesserae (they only touch and part)", scene=sc,
               cue="gateless_gate" if k == 0 else None, src=src_of(e))
        n = ns(d)
        lvl = np.sin(np.pi * np.arange(n) / n) ** 1.5
        mx.add("sfx", L.decorrelate(O.split_cascade_bed(r, d, lvl * 0.4), r), t_d, db(-34), send=0.5,
               name="the mosaic loosens (micro-cleaves)", scene=sc)
        mx.add("sfx", O.rms_norm(O.swarm_air(r, d, lvl * 0.6, lvl), db(-40)), t_d, 1.0, send=0.4,
               name="drifting tesserae (air)", scene=sc)
    e = ev.find(sc, ["flash"], near=187.0, win=0.8, kind_only=True)
    t_f = e["t"] if e else 187.0
    d = dur_of(e, 0.5)
    mx.add("sfx", L.swell(r, d + 0.35, 400, 12000, 1.8), t_f + d, db(-24), send=0.4,
           name="white-gold flash (air swell)", scene=sc, src="event:" + e["file"] if e else "cue")


# ================================================================================================
# m10 · THE STICKY FLAG: EXPLICIT
# ================================================================================================
STICKY_FALLBACK = [(190.9, "touch"), (191.45, "stretch"), (192.55, "peel"), (193.05, "cling"), (197.85, "touch")]


def sticky_sound(r, desc):
    """stickiness by behaviour only: touch (tacky 'tck'), stretch (a strand), peel, shake (the Flag whipped about -
    it will not let go), cling (cloth clinging to a sleeve)."""
    d = desc.lower()
    if "stretch" in d or "pull" in d:
        return L.squelch_stretch(r, 0.8), -24
    if "peel" in d or "release" in d:
        return B.sticky_peel(r, 0.32), -24
    if "shake" in d:
        x = L.addm(FF.raise_grain(r, 0.2, False) * 0.8, L.squelch_touch(r).x * 0.7)
        return Snd(L.norm(x, 1.0), 0.0), -24
    if "cling" in d or "sleeve" in d:
        x = L.addm(L.squelch_touch(r).x, L.cloth_rustle(r, 0.35, 0.7, 0.8).x * 0.6)
        return Snd(L.norm(x, 1.0), 0.0), -26
    return L.squelch_touch(r), -24


def m10(mx, ev):
    sc = "m10"
    r = R(1010)
    t_o, p, _, src = ev.get(sc, ["offer"], CUES["sticky_offer"], 1.0, kind_only=True)
    mx.add("sfx", L.cloth_rustle(r, 0.5, 0.8, 0.9), t_o, db(-30), pan_of(p, -0.3), send=0.3,
           name="Lutie offers the Flag (robes)", scene=sc, src=src)
    st = ev.all(sc, ["sticky", "peel", "stretch", "cling", "touch", "shake"], 188.0, 199.5)
    items = [(e["t"], e["desc"] + " " + e["kind"], e["pan"], "event:" + e["file"]) for e in st] or \
            [(t, k, -0.1, "cue") for t, k in STICKY_FALLBACK]
    for t, d, p, src in items:
        snd, lv = sticky_sound(r, d)
        mx.add("sfx", snd, t, db(lv), p, send=0.25, name="sticky (behaviour only): " + d.split()[0][:24], scene=sc,
               cue="sticky" if abs(t - CUES["sticky"]) < 1.2 else None, src=src)
    tw = ev.all(sc, ["twinkle", "glint", "shimmer"], 196.5, 199.6)
    if tw:
        for e in tw:
            mx.add("sfx", O.glint(r), e["t"], db(-36), e["pan"], send=0.6, name="tiles twinkle (deadpan)", scene=sc,
                   src="event:" + e["file"])
    else:
        glints(mx, r, word("C07", "sticky") + 0.4, word("C07", "sticky") + 1.1, 5, -37, scene=sc,
               name="tiles twinkle (deadpan)")
    mx.add("sfx", O.pigeon_coo(r), 198.8, db(-38), 0.55, send=0.7, name="a pigeon coos high in the apse", scene=sc)
    # 202.9 GLORIOUS: every figure raises a Flag at once
    T = ev.get(sc, ["glorious"], CUES["glorious"], 0.5, kind_only=True)
    rs = ev.all(sc, ["raise", "salute"], 201.5, 204.5)
    nd = sum(int(m.group(1)) for e in rs for m in [re.search(r"(\d+)\s*flags", e["desc"].lower())] if m)
    if rs and (len(rs) >= 12 or not nd):
        raises = np.array([(e["t"] - 0.12, e["pan"], float(np.clip(1 - e["strength"], 0, 1)),
                            float(np.clip(e["strength"], 0.2, 1))) for e in rs])
        src = "event:" + rs[0]["file"]
    else:   # the whole congregation (the scene may give its count: '~130 Flags'); near ranks first, rear a hair later
        k = max(nd, 130) if rs else 700
        t_r = rs[0]["t"] if rs else T[0]
        d = r.uniform(0, 1, k) ** 0.7
        raises = np.stack([t_r - 0.12 + r.normal(0, 0.045, k) + 0.12 * d, np.clip(r.normal(0, 0.55, k), -1, 1), d,
                           r.uniform(0.6, 1.0, k) / (1 + 3 * d)], -1)
        src = "event:" + rs[0]["file"] if rs else "design"
    for e in ev.all(sc, ["flag_whoosh", "whoosh"], T[0] - 0.4, T[0] + 1.5):   # hero Flags swung past the camera
        mx.add("sfx", Snd(FF.raise_grain(r, 0.0, True), 0.12), e["t"], db(-18 + 4 * (e["strength"] - 0.5)), e["pan"],
               send=0.3, name="a Flag swung up past the camera", scene=sc, src="event:" + e["file"])
    crowd, c0 = FF.crowd_wave(T[0] - 0.4, 3.5, raises, bed=True, seed=33, bed_level=0.45)
    mx.add("sfx", crowd, c0, db(-7), send=0.35, name=f"GLORIOUS: {len(raises)} Flags raised at once (swish, unfurl, snap)",
           scene=sc, cue="glorious", src=src)
    mx.add("sfx", L.whoosh(r, 1.1, 0.42, 140, 2600, 0.9, -0.3, 0.3, dark=0.35), T[0], db(-13), send=0.35,
           name="the massed whoosh of the raise", scene=sc, cue="glorious", src=T[3])
    mx.add("sfx", L.thud(r, 55.0, 0.6, 0.6, 300), T[0] + 0.03, db(-22), name="air pushed by a thousand cloths",
           scene=sc)
    # 204.6 EXPLICIT · OPVS · SCHISMATICVM sets itself along the bottom band; then the donor inscription
    lets = ev.all(sc, ["tile_set", "letter", "titul"], 204.3, 209.8)
    if lets:
        sched = [(e["t"], e["pan"], e["strength"], "event:" + e["file"]) for e in lets]
    else:
        sched = [(t, p, 1.0, "cue") for t, p in cantor_letters("explicit", 204.62, 206.95, [2, 3, 3, 1, 3, 5, 2, 2, 3])]
        donor = "HOCOPVSTESSELLAVITMACHINAANNOSVRVEIIMMXXVI"
        sched += [(207.25 + 1.9 * k / len(donor), -0.8 + 1.6 * k / len(donor), 0.4, "cue") for k in range(len(donor))]
    for t, p, s, src in sched:
        mx.add("sfx", O.set_flurry(r, int(8 + 14 * s), 0.05 + 0.04 * s, "gold", vel=0.45 + 0.2 * s), t,
               db(-25) * (0.45 + 0.55 * s), p * 0.85, send=0.45,
               name="EXPLICIT / donor inscription: letters set themselves", scene=sc,
               cue="explicit" if abs(t - CUES["explicit"]) < 0.05 else None, src=src)
    # 209.5 end card: the candle is snuffed; black by 212.0
    t_sn = snuff_time(ev)
    e = ev.find(sc, ["snuff"], near=t_sn, win=0.05, kind_only=True)
    mx.add("sfx", O.snuff(r), t_sn, db(-15), 0.25, send=0.45, name="the candle is snuffed", scene=sc, cue="end_card",
           src="event:" + e["file"] if e else "cue")
    # the tiles go dark once the flame is gone: the gold's last glints die away, falling
    e = ev.find(sc, ["lights_out", "dark"], near=t_sn + 0.1, win=1.0, kind_only=True)
    if e:
        d = dur_of(e, 1.4)
        for q, m in enumerate([105, 100, 98, 93, 88, 86, 81]):
            mx.add("sfx", Snd(L.chime(r, float(midi_hz(m)), 1.2, (1.0, 2.76), 0.5, 2.0, 0.7, 0.01), 0.0),
                   e["t"] + d * q / 7 + r.uniform(0, 0.05), db(-44 - 1.5 * q), r.uniform(-0.6, 0.6), send=0.5,
                   name="the gold goes dark (last glints falling away)", scene=sc, src="event:" + e["file"])


# ================================================================================================
# scene-exported events not claimed by a design: mosaic-aware generic voicing, then film-1's generic models
# ================================================================================================
GENERIC = [(re.compile(p), h) for p, h in [
    (r"^tiles$", "tilerain"), (r"tinkle|tumbl", "tumble"), (r"^fall$|crumble", "crumble"),
    (r"flip_wave|flips", "flipwave"), (r"tile_flip|\bflip", "flip"), (r"tiles_jump|jump|resettle", "jump"),
    (r"tiny_split", "tinysplit"), (r"\bsplit", "split"), (r"crack|fractur", "crack"), (r"shatter", "shatter"),
    (r"halo", "halo"), (r"drift|float", "drift"), (r"swirl|dust", "swirl"), (r"dissolve", "drift"),
    (r"wink", "glint"), (r"swell", "swell"), (r"\bring\b", "sweep"),
    (r"tile_set|letter|titul|numeral|\bset\b|\bsets\b|setting|kingdom|altar|tiara|click", "set"),
    (r"candle_strike|match", "match"),
    (r"snuff", "snuff"), (r"step|foot|walk|charge|\brun", "step"), (r"armou?r|lamellar", "armour"),
    (r"goggle|nvg", "goggles"), (r"telescop", "telescope"), (r"unfurl|raise|salute", "raise"),
    (r"stylus", "stylus"), (r"seal|bulla", "seal"), (r"coin", "coin"), (r"ching", "kaching"), (r"\beye", "eye"),
    (r"wing", "wing"), (r"ignit", "ignite"), (r"\block", "lock"), (r"descend", "descend"), (r"plant", "plant"),
    (r"sticky|peel|stretch|cling|touch", "sticky"), (r"offer|rustle|robe|sleeve|cloth", "rustle"),
    (r"machine|gear|cog", "gears"), (r"static", "static"), (r"noise|glitch", "glitch"), (r"tier", "tier"),
    (r"storm", "storm"), (r"breach|blast|explo", "breach"), (r"ray|glide|sweep", "sweep"),
    (r"flash|bloom|glow|portal", "flash"), (r"shimmer|twinkle|glint|sparkle", "glint"), (r"dive", "dive"),
    (r"whoosh", "whoosh"),
    (r"camera|tilt|push|wind|silence|tesseract|inside|shadow|egregore|scatter|tremble|rise|avalanche|converg|"
     r"wheel|topple|tower|music|walla", "skip"),
]]


NEVER = re.compile(r"^(radio|voice|vocal|walla|music|cut|punch.*|land|alone|complete|progress|together|inhale|calm|"
                   r"hold|beat|subtitle)$")                         # voices / picture cuts / progress markers


def classify(e):
    k = e["kind"].lower()
    if NEVER.match(k):
        return "skip"
    for rx, h in GENERIC:
        if rx.search(k):
            return h
    for rx, h in GENERIC:
        if rx.search(e["desc"].lower()):
            return h
    return None


def generic_sound(h, e, r):
    """-> (Snd|array, peak dB at strength 1, reverb send) or None."""
    s = float(np.clip(e["strength"], 0, 1))
    lv = level_of(e["desc"]) or 0
    if h == "set":
        return O.set_flurry(r, int(10 + 20 * s), 0.05 + 0.1 * s, "gold"), -26, 0.45
    if h == "tilerain":   # vx.mosaic fx: one event per frame, '(N tesserae)' hitting the floor / bouncing
        m = re.search(r"(\d+)\s*tesserae", e["desc"])
        k = int(min(400, int(m.group(1)) if m else 12))
        t = r.uniform(0.0, 1.0 / 24, k)
        v = np.sqrt(2 * O.G * r.uniform(0.3, 3.0, k))
        rain = O.tile_rain(r, 0.0, 0.1, t, v, np.clip(e["pan"] + r.normal(0, 0.3, k), -1, 1), heap_p=0.3)
        return rain, -30 + 12 * np.sqrt(s), 0.4
    if h == "swell":
        d = max(0.3, dur_of(e, 1.0))
        return Snd(L.swell(r, d, 300, 8000, 2.0).x, 0.0), -32, 0.4
    if h == "flip":
        return O.flip_grain(r, "gold", "screen" in e["desc"].lower() or "slop" in e["desc"].lower()), -24, 0.4
    if h == "flipwave":
        d = dur_of(e, 1.0)
        return O.flip_wave(r, 0.0, d, lambda u: np.sin(np.pi * np.clip(u / d, 0, 1)), lambda u: e["pan"] + 0 * u,
                           "gold", total=int(300 * d * (0.3 + s))), -12, 0.4
    if h == "jump":
        return O.tiles_jump(r, int(500 + 2000 * s), dur_of(e, 0.4)), -14, 0.4
    if h == "tinysplit":
        d = dur_of(e, 0.5)
        n = ns(d)
        return O.split_cascade_bed(r, d, np.sin(np.pi * np.arange(n) / n)), -30, 0.35
    if h == "split":
        return Snd(O.split_grain(r, 650.0 * 2 ** min(lv, 8), lv), 0.0), -20 - 2 * lv, 0.4
    if h == "crack":
        return O.bed_crack(r, 0.4, lv, True), -16 - 1.5 * lv, 0.4
    if h == "shatter":
        return L.shatter(r, "glass", 0.5 + 0.7 * s, int(40 + 100 * s), 3.0), -18, 0.35
    if h in ("tumble", "crumble"):   # torn-out tesserae tumbling / ragged pieces crumbling away: a rain of tiles
        d = max(0.3, dur_of(e, 1.0))
        k = int(25 + 40 * s) if h == "tumble" else int(200 + 300 * s)
        t = O.sample_rate_curve(r, 0.0, d, lambda u: np.exp(-u / (0.5 * d)), k)
        tl_, v, p = O.fall_plan(r, t, r.uniform(0.2, 2.0, k), np.clip(e["pan"] + r.normal(0, 0.5, k), -1, 1))
        return O.tile_rain(r, 0.0, d + 0.5, tl_, v, p, heap_p=0.2), (-24 if h == "tumble" else -16), 0.45
    if h == "halo":
        return O.halo_split(r), -20, 0.5
    if h == "drift":
        return O.drift_clinks(r, dur_of(e, 2.0), int(6 * dur_of(e, 2.0))), -22, 0.6
    if h == "swirl":
        d = dur_of(e, 1.5)
        return O.dust_hiss(r, d, np.sin(np.pi * np.arange(ns(d)) / ns(d))), -28, 0.3
    if h == "match":
        return O.match_strike(r), -14, 0.3
    if h == "snuff":
        return O.snuff(r), -16, 0.4
    if h == "step":
        d = e["desc"].lower()
        shoe = "heap" if ("heap" in d or "tile" in d) else ("boot" if e["scene"] == "m05" else "sandal")
        return O.stone_step(r, 0.6 + 0.4 * s, shoe), -24, 0.4
    if h == "armour":
        return O.lamellar(r, 0.35, 0.4 + 0.6 * s), -26, 0.35
    if h == "goggles":
        return O.nvg_drop(r), -24, 0.3
    if h == "telescope":
        return L.telescope(r, 3, 0.11), -18, 0.3
    if h == "raise":
        return Snd(FF.raise_grain(r, 0.2, s > 0.4), 0.12), -16, 0.3
    if h == "stylus":
        return O.stylus(r, dur_of(e, 0.8)), -24, 0.3
    if h == "seal":
        return O.bulla_slam(r), -6, 0.35
    if h == "coin":
        return O.coin_drop(r), -22, 0.35
    if h == "kaching":
        return O.ka_ching(r), -14, 0.35
    if h == "eye":
        return O.eye_open(r, 0.6 + 0.8 * s), -26, 0.5
    if h == "wing":
        return O.wing_unfold(r), -20, 0.45
    if h == "ignite":
        return O.ignite(r), -14, 0.45
    if h == "lock":
        return O.lock_boom(r), -6, 0.4
    if h == "descend":
        return O.descend_chain(r, dur_of(e, 1.8)), -30, 0.45
    if h == "plant":
        return O.plant_in_heap(r), -8, 0.5
    if h == "sticky":
        snd, lv_ = sticky_sound(r, e["desc"] + " " + e["kind"])
        return snd, lv_, 0.25
    if h == "rustle":
        return L.cloth_rustle(r, 0.3 + 0.3 * s, 0.6 + 0.6 * s, 0.8), -30, 0.25
    if h == "gears":
        d = dur_of(e, 2.0)
        return O.rms_norm(O.gear_train(r, d, 4.0), db(-10)), -28, 0.35
    if h == "static":
        d = dur_of(e, 1.0)
        return L.tv_static(r, d) * L.env_pts(ns(d), [(0, 0), (0.05, 1), (d, 0.3)])[:, None], -28, 0.1
    if h == "glitch":
        return Snd(L.glitch(r, 0.1 + 0.3 * s), 0.0), -26, 0.1
    if h == "tier":
        return O.set_flurry(r, 60, 0.25, "screen", vel=0.8), -18, 0.45
    if h == "storm":
        return O.tile_storm(r, dur_of(e, 0.7), 2000), -14, 0.3
    if h == "breach":
        return L.explosion(r, 0.6 + 0.6 * s, 4.0, "stone", int(20 + 30 * s)), -4, 0.3
    if h == "sweep":
        return B.light_sweep(r, dur_of(e, 0.8)), -34, 0.5
    if h == "flash":   # light growing until '(to T)': the swell rises from the event and peaks at its end
        return Snd(L.swell(r, max(0.25, dur_of(e, 0.5)), 400, 10000, 2.0).x, 0.0), -28, 0.3
    if h == "glint":
        return O.glint(r), -38, 0.6
    if h == "dive":
        d = dur_of(e, 0.6)
        return L.whoosh(r, d + 0.3, d * 0.8, 200, 5000, 1.2, 0.0, 0.0, dark=0.3), -26, 0.3
    if h == "whoosh":
        d = dur_of(e, 0.0)
        if d > 0.9:    # a slow camera move / a long gesture: dark air peaking at 60 %
            return Snd(L.whoosh(r, d + 0.3, d * 0.6, 80, 900, 0.7, -0.2, 0.2, dark=0.7, shape=1.4).x, 0.0), -30, 0.3
        return B.whoosh_short(r, s), -24, 0.3
    return None


def place_events(mx, ev, stats):
    r = R(777)
    for e in ev.unused():
        ev.used.add(e["_i"])
        if not (0.0 <= e["t"] < DUR) or not mx.active(e["scene"]):
            continue
        if SIL0 <= e["t"] < SIL1:
            stats["skipped"].append(e)
            continue
        h = classify(e)
        if h == "skip":
            stats["skipped"].append(e)
            continue
        if h == "set" and "letter by letter" in e["desc"].lower():
            words = re.findall(r"\b[A-Z][A-Z·]{1,}\b", e["desc"])
            titulus_run(mx, r, e, "".join(words).replace("·", "") or "TITVLVS", e["scene"])
            stats["generic"].append(e)
            continue
        out = generic_sound(h, e, r) if h else None
        if out is None:
            impactish = re.search(r"slam|impact|land|hit|stack|thud|drop|fall|crash|bang|boom", e["kind"] + " " + e["desc"],
                                  re.I)
            o = B.event_sound(e, r) if (B.classify(e) != "impact" or impactish) else None   # film-1 generic models
            if o is None:
                stats["skipped"].append(e)
                continue
            snd, stem, g, send = o
            mx.add(stem, snd, e["t"], g * (0.35 + 0.65 * float(np.clip(e["strength"], 0, 1))), e["pan"], send,
                   name=B.classify(e) + " (generic)", scene=e["scene"], src="event:" + e["file"])
            stats["generic"].append(e)
            continue
        snd, lv, send = out
        g = db(lv) * (0.45 + 0.55 * float(np.clip(e["strength"], 0, 1)))
        x = snd.x if isinstance(snd, Snd) else snd
        pk = float(np.abs(x).max()) if len(x) else 1.0
        snd = Snd(np.asarray(x) / max(pk, 1e-9), snd.sync) if isinstance(snd, Snd) else np.asarray(x) / max(pk, 1e-9)
        mx.add("sfx", snd, e["t"], g, e["pan"], send, name=f"{h}: {e['kind']} ({e['desc'][:40]})", scene=e["scene"],
               src="event:" + e["file"])
        stats["generic"].append(e)


# ================================================================================================
# physics: every exported cloth track -> flutter (tools/foley_flutter.py); fallbacks for scenes without one
# ================================================================================================
FLAG_PEAK = {"m01": -19, "m04": -21, "m05": -21, "m06": -20, "m07": -20, "m08": -23, "m09": -23, "m10": -21}
FALLBACK_FLAGS = {  # scene: (t0, t1, [(t, energy)], pan0, pan1, field)
    "m01": (14.8, 22.0, [(14.8, 0.0), (15.2, 0.5), (16.6, 0.3), (18.8, 0.25), (19.6, 0.32), (22.0, 0.25)], 0.1, 0.1,
            False),
    "m04": (74.3, 81.2, [(74.3, 0.0), (74.9, 0.15), (76.0, 0.22), (80.8, 0.2), (81.2, 0.0)], 0.0, 0.0, False),
    "m05": (102.4, 106.5, [(102.4, 0.0), (102.6, 0.7), (103.9, 0.45), (104.5, 0.85), (105.8, 1.0), (106.5, 0.7)],
            -0.2, 0.2, True),
    "m08": (149.0, 163.0, [(149.0, 0.0), (149.3, 0.3), (150.5, 0.18), (155.0, 0.3), (157.0, 0.42), (159.0, 0.3),
                           (163.0, 0.22)], 0.05, 0.05, False),
    "m09": (163.0, 187.5, [(163.0, 0.15), (168.7, 0.25), (180.4, 0.3), (187.0, 0.42), (187.5, 0.2)], 0.0, 0.0, True),
    "m10": (188.5, 202.8, [(188.5, 0.0), (188.8, 0.22), (190.4, 0.12), (193.9, 0.15), (202.8, 0.1)], -0.25, -0.1,
            False),
}
_PREF = {}


def flutter_ref(field):
    """peak of the flutter engine for a steady energy 0.8 (the scale that maps a scene's FLAG_PEAK)."""
    if field not in _PREF:
        x = FF.flutter_from_energy(np.full(96, 0.8), 24, 0.0, 1.0, [], seed=5, field=field)
        _PREF[field] = float(np.abs(x).max())
    return _PREF[field]


def corr(en, x, fps):
    rms = FF.frame_rms(x, fps, len(en))
    act = np.nonzero(en > 0.02)[0]
    return float(np.corrcoef(en[act], rms[act])[0, 1]) if len(act) > 5 and rms[act].std() > 0 else float("nan")


def physics(mx, report):
    have = set()
    for (sc, name), tr in sorted(TRACKS.items()):
        if tr["kind"] == "tiles" or not mx.active(sc):
            continue
        x, t0, info = FF.track_audio(tr)
        field = info.get("field", False)
        g = db(FLAG_PEAK.get(sc, -18) - (5 if field else 0)) / flutter_ref(field)
        c = corr(tr["energy"], x, tr["fps"])
        mx.add("sfx", x, t0, g, send=0.25, name=f"PHYSICS flutter <- {tr['file']} (r={c:.2f})", scene=sc,
               src="physics:" + tr["file"])
        have.add(sc)
        report.append(dict(file=tr["file"], scene=sc, kind=tr["kind"], frames=len(tr["energy"]), t0=round(t0, 3),
                           energy_max=round(float(tr["energy"].max()), 3), snaps=int(len(tr["events"])),
                           flap_hz_from_sim=bool("flap_hz" in tr and np.any(tr["flap_hz"] > 0)), field=field,
                           corr_energy_rms=round(c, 3)))
    r = R(5)
    for sc, (t0, t1, pts, p0, p1, field) in FALLBACK_FLAGS.items():
        if sc in have or not mx.active(sc):
            continue
        nfr = int((t1 - t0) * 24)
        en = np.clip(np.interp(t0 + np.arange(nfr) / 24, *zip(*pts)) * (1 + 0.3 * B.frame_turb(r, nfr)), 0, None)
        x = FF.flutter_from_energy(en, 24, np.linspace(p0, p1, nfr), 1.0, [], seed=zlib.crc32(sc.encode()), t0=t0,
                                   field=field)
        g = db(FLAG_PEAK[sc] - (5 if field else 0)) / flutter_ref(field)
        mx.add("sfx", x, t0, g, send=0.25, name=f"flutter (fallback energy: awaiting the {sc} flag track)", scene=sc,
               src="fallback-physics")
        report.append(dict(file=f"(fallback {sc})", scene=sc, kind="flag", frames=nfr, t0=t0,
                           energy_max=round(float(en.max()), 3), snaps=0, flap_hz_from_sim=False, field=field,
                           corr_energy_rms=round(corr(en, x, 24), 3)))


# ================================================================================================
# master
# ================================================================================================
def finalize(mx):
    lims, hot = {}, []
    for stem in ("sfx", "amb"):
        b = mx.bus[stem]
        np.nan_to_num(b, copy=False)
        b[:] = L.hp(b, 18.0, 2)                                        # no DC / infrasonic waste
        b[ns(SIL0):ns(SIL1 if stem == "sfx" else SIL_W)] = 0.0          # THE SACRED SILENCE
        b[NTOT - ns(0.8):] *= (np.cos(np.linspace(0, np.pi / 2, ns(0.8))) ** 2).astype(np.float32)[:, None]  # black by 212
        m = np.abs(b).max(1)
        blk = SR // 10
        pk = m[: len(m) // blk * blk].reshape(-1, blk).max(1)
        for i in np.argsort(pk)[::-1][:10]:                              # what drives the limiter (diagnostics)
            if pk[i] < 0.891:
                break
            t = i / 10
            who = sorted([e for e in mx.log if e["stem"] == stem and e["start"] - 0.05 <= t + 0.1 and e["t"] + 4 >= t],
                         key=lambda e: -e["peak"])[:4]
            hot.append(dict(stem=stem, t=t, peak_db=round(float(20 * np.log10(pk[i])), 1),
                            who=[f"{e['name'][:48]}@{e['t']:.2f}({20 * np.log10(max(e['peak'], 1e-9)):.1f}dB)"
                                 for e in who]))
        mx.bus[stem], lims[stem] = B.limiter(b)
        mx.bus[stem][ns(SIL0):ns(SIL1 if stem == "sfx" else SIL_W)] = 0.0
    (OUTD / "hot.json").write_text(json.dumps(hot, indent=1))
    return lims


def ns_t(t):
    return int(round(t * SR))


CATALOGUE = """## How it is made (every sound synthesized at 48 kHz from first principles; no samples)

| model | physics / method | used for |
|---|---|---|
| the tessera atom (op_foley_lib.tile_click) | Hertzian contact pulse (tau ~ v^-1/5; wet lime = long soft pulse) + edge/face micro-contacts + chipping noise, exciting the tile's corner/edge chatter modes (smalti / gold / stone / slop-screen) and the struck surface's local modes (marble floor, heap of tiles, mortar bed, set wall); 1,000+ variants in a grain bank placed by a numba kernel | every tile sound in the film |
| rigid-body choreography (bounce_events) | per tile: first contact at free-fall speed, bounces v_k+1 = e v_k with airtime 2v/g, rocking settle (geometric intervals), skids on bare marble, neighbours jostled on the heap; restitution per surface | avalanche, tiles raining after the lock/breach, the plant, the late tile after the silence |
| THE AVALANCHE | ~40,000 tesserae: detachment schedule (or the scene's per-frame land counts) -> free fall from 1-13 m -> floor, then the growing heap (heap probability rises with the landed count) + mortar slabs + the pour (dense grain noise ~ sqrt(flow)) + floor roar + detachment crackle + the tower's groan/crack/impact | m07 141.8-145.0 (hard cut) |
| THE RISE (the avalanche un-happens) | each rising tile's bounce train generated forward and played time-reversed (with its reverb: pre-verb) so it swells into the instant it leaves the floor; the swarm's air; each landing a setting (dry tick over a wet 'tup'); the last wave sets at once (±6 ms) = the convergence | m08 153.4-159.0 |
| split-flap flips | release tick + half-turn whirr + slap onto the other face (+ electrical 'tzt' for a slop screen); Poisson waves with a travelling front | slop waves (m02), the static resolving to gold (m03), the disputation re-set (m04), wings of feather-tiles |
| recursive splitting | modal frequency ~ 1/size: each quadtree generation 4x the pieces, an octave higher, 6 dB softer per piece; beyond ~16 kHz only the cleave clicks and a swirling dust hiss remain | the map's borders (m06), ENDLESS (128.6) |
| candle | match: stick-slip scrape -> flare (falling low-pass) + sulphur sparks + gas whump; flame: plume burble puffing at the ~11 Hz flicker frequency of a diffusion flame; snuff: brass cup, choke, wick sizzle, smoke | 0.8 strike, the one candle on the bare wall, the snuff at the end |
| stone & space | procedural impulse responses per space (vault, nave, hall, sanctuary, void, Babel, bare plaster, dome with flutter echoes, apse); boots / sandals / heap steps; lamellar = Poisson contacts of random laced plates; NVG servo + detent + tube whine; pigeons; wind edge-tones in the high windows | throughout |
| cloth (tools/foley_flutter.py) | per-sample parameters from the scene's cloth simulation: flap pulse train at the sim's flap_hz, band ~ energy, rustle ~ E^1.35, fwap on every acceleration, whip-crack N-wave on every sim snap; crowd raises = grains | every hero flag, flag fields, GLORIOUS |
"""


def write_cues(mx, report, lims, stats, files):
    Ls = ["# Foley cue sheet — OPVS SCHISMATICVM (film 3; generated by films/opus/tools/op_foley_build.py)", "",
          "Stems `films/opus/audio/stems/sfx.wav` (hard effects, tesserae, physics-derived flag flutter) and "
          "`ambience.wav` (the basilica's beds): 48 kHz stereo float32, exactly 212.0 s (10,176,000 samples), T=0. "
          "Rebuild after final renders: `source .venv/bin/activate && python films/opus/tools/op_foley_build.py && "
          "python films/opus/tools/op_foley_check.py` (re-reads every export in films/opus/cache/foley).", "",
          "**Sacred silence:** `sfx` is digital zero 145.000–146.500; `ambience` is digital zero 145.000–146.000 and "
          "carries only a faint breath of wind 146.0–146.5. Everything that starts before 145.0 (the avalanche, "
          "Babel's roar, every reverb tail) is hard-cut at 145.000. The m04→m05 hard cut (95.75) cuts tails too.", "",
          "**With Composer-2:** Foley owns sub (<60 Hz) + noise/debris transients on the big hits; tuned subs sit on "
          "D (36.7/73.4 Hz); tonal Foley (glints, the Tesseract's rubbed glass, ka-ching) uses D/A/E only. "
          "**With Vocalist-2:** radio clicks stay in dialogue; Babel walla and this Babel bed both stop at 145.000.", "",
          f"Limiter min gain: sfx {lims.get('sfx', 1):.3f}, ambience {lims.get('amb', 1):.3f} (1.0 = untouched).", "",
          CATALOGUE,
          "## Scene exports used (films/opus/cache/foley)", ""]
    used_by_file = {}
    for e in ev_all_events:
        d = used_by_file.setdefault(e["file"], dict(total=0, design=0, generic=0, skipped=0))
        d["total"] += 1
    for e in stats["design"]:
        used_by_file[e["file"]]["design"] += 1
    for e in stats["generic"]:
        used_by_file[e["file"]]["generic"] += 1
    for e in stats["skipped"]:
        used_by_file[e["file"]]["skipped"] += 1
    if used_by_file:
        Ls += ["| events file | events | drove designed cues | voiced generically | not voiced |", "|---|---:|---:|---:|---:|"]
        for f, d in sorted(used_by_file.items()):
            Ls.append(f"| {f} | {d['total']} | {d['design']} | {d['generic']} | {d['skipped']} |")
    else:
        Ls.append("_No scene events exported yet: every designed cue sits on plan.json cues / word timings._")
    tiles = [tr for tr in TRACKS.values() if tr["kind"] == "tiles"]
    Ls += ["", "Tile-physics tracks: " + (", ".join(f"`{t['file']}` ({', '.join(k for k in t if k in ('flips', 'fall', 'land', 'lift', 'bounce'))})"
                                                  for t in tiles) if tiles else "none yet (designed fallbacks)."), ""]
    Ls += ["## Physics-derived flag flutter", "",
           "| track | scene | frames | start s | max energy | snaps | flap_hz from sim | field | r(energy, RMS) |",
           "|---|---|---:|---:|---:|---:|:-:|:-:|---:|"]
    for p in report:
        Ls.append(f"| {p['file']} | {p['scene']} | {p['frames']} | {p['t0']:.2f} | {p['energy_max']:.2f} | {p['snaps']} | "
                  f"{'yes' if p['flap_hz_from_sim'] else 'no'} | {'yes' if p['field'] else 'no'} | {p['corr_energy_rms']:.3f} |")
    Ls += ["", "## Placements by scene", "",
           "Repeated sounds within a scene are collapsed into one row (count, first…last). source = design (cue/word "
           "timing), event:<file> (a scene-exported hit), physics:<file> (a scene simulation track).", ""]
    by = {}
    for e in mx.log:
        by.setdefault(e["scene"], []).append(e)
    for s in TL["scenes"]:
        evs = sorted(by.get(s["id"], []), key=lambda e: e["t"])
        groups = {}
        for e in evs:
            groups.setdefault((e["name"], e["stem"], e["src"]), []).append(e)
        Ls += [f"### {s['id']} — {s['title']} ({s['start']:.2f}–{s['end']:.2f})", "",
               "| t (s) | n | stem | sound | cue | source | peak |", "|---:|---:|---|---|---|---|---:|"]
        for (nm, stem, src), g in sorted(groups.items(), key=lambda kv: kv[1][0]["t"]):
            ts = f"{g[0]['t']:.3f}" if len(g) == 1 else f"{g[0]['t']:.2f}…{g[-1]['t']:.2f}"
            cue = ", ".join(sorted({e["cue"] for e in g if e["cue"]}))
            pk = max(e["peak"] for e in g)
            Ls.append(f"| {ts} | {len(g)} | {stem} | {nm} | {cue} | {src} | {20 * np.log10(max(pk, 1e-6)):.1f} dB |")
        Ls.append("")
    (OUTD / "CUES.md").write_text("\n".join(Ls))


SCENE_FNS = [("m01", m01), ("m02", m02), ("m03", m03), ("m04", m04), ("m05", m05), ("m06", m06), ("m07", m07),
             ("m08", m08), ("m09", m09), ("m10", m10)]
TRACKS = {}
ev_all_events = []


def main(argv=None):
    global TRACKS, ev_all_events
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of scenes to audition (-> audio/foley/preview_*.wav)")
    a = ap.parse_args(argv)
    only = set(a.only.split(",")) if a.only else None
    t_start = time.time()
    OUTD.mkdir(parents=True, exist_ok=True)
    STEMS.mkdir(parents=True, exist_ok=True)
    evs, files = load_events()
    ev_all_events = evs
    TRACKS = load_tracks()
    ev = Events(evs)
    print(f"[op-foley] {len(evs)} scene events in {len(files)} files, {len(TRACKS)} tracks in {CACHE}")
    mx = Mix(only)
    t = time.time()
    beds(mx, ev)
    print(f"[op-foley] beds ({time.time() - t:.1f}s)")
    for sid, fn in SCENE_FNS:
        if mx.active(sid):
            t = time.time()
            fn(mx, ev)
            print(f"[op-foley] {sid} designed ({time.time() - t:.1f}s)")
    stats = dict(design=[e for e in evs if e["_i"] in ev.used], generic=[], skipped=[])
    place_events(mx, ev, stats)
    report = []
    t = time.time()
    physics(mx, report)
    print(f"[op-foley] physics ({time.time() - t:.1f}s)")
    t = time.time()
    mx.render_verbs()
    print(f"[op-foley] reverbs ({time.time() - t:.1f}s)")
    lims = finalize(mx)
    if only:
        s0 = min(SCENES[s]["start"] for s in only)
        s1 = max(SCENES[s]["end"] for s in only)
        a0, a1 = ns_t(max(0.0, s0 - 0.5)), ns_t(min(DUR, s1 + 0.5))
        tag = "_".join(sorted(only))
        sf.write(OUTD / f"preview_{tag}.wav", mx.bus["sfx"][a0:a1] + mx.bus["amb"][a0:a1], SR, subtype="FLOAT")
        print(f"[op-foley] preview -> {OUTD / f'preview_{tag}.wav'} (from {a0 / SR:.2f}s)")
    else:
        for stem, fn in (("sfx", "sfx.wav"), ("amb", "ambience.wav")):
            assert mx.bus[stem].shape == (NTOT, 2) and mx.bus[stem].dtype == np.float32
            sf.write(STEMS / fn, mx.bus[stem], SR, subtype="FLOAT")
        (OUTD / "placements.json").write_text(json.dumps(mx.log, indent=0))
        (OUTD / "physics.json").write_text(json.dumps(report, indent=1))
        (OUTD / "events_used.json").write_text(json.dumps(
            {k: [dict(t=e["t"], scene=e["scene"], file=e["file"], kind=e["kind"], desc=e["desc"]) for e in v]
             for k, v in stats.items()}, indent=1))
        write_cues(mx, report, lims, stats, files)
        print(f"[op-foley] wrote {STEMS / 'sfx.wav'}, {STEMS / 'ambience.wav'} + CUES.md ({len(mx.log)} placements)")
    print(f"[op-foley] events: {len(stats['design'])} drove designed cues, {len(stats['generic'])} voiced generically, "
          f"{len(stats['skipped'])} not voiced")
    print(f"[op-foley] done in {time.time() - t_start:.1f}s; limiter {lims}")


if __name__ == "__main__":
    main()

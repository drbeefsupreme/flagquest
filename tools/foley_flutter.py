"""PHYSICS-DERIVED FLAG FOLEY (owner: Foley).

The sound of every simulated hero flag is computed from its cloth simulation, exported by the scenes via
vx.foley.export_track -> cache/foley/<scene>_<name>.npz:
    energy  (n,)  per-frame max-over-substeps mean cloth speed rel. to the pole, 0..~1 (1 = gale)
    pan     (n,)  screen pan -1..1
    gain          closeness / loudness
    events  (k,3) [T_global, snap_strength, pan] trailing-edge whip / tension-release snaps
    flap_hz (n,)  optional: dominant flap frequency measured at substep resolution
    snap    (n,)  optional: per-frame snap strength

Model (per audio sample, all parameters interpolated from the per-frame sim data):
  * FLAP: each flap cycle of the cloth (phase = ∫ flap_hz dt, from the sim, or Strouhal ~ 1.6 + 9·E Hz
    when not exported) pushes a pressure pulse: an asymmetric pulse train with sharpness ∝ E carried by
    band-passed noise whose centre rises with E (a faster cloth sounds brighter).
  * RUSTLE: continuous broadband cloth/air friction noise, level ∝ E^1.35, centre 700 + 3500·E Hz.
  * WHIPS: every local rise of the sim's energy (a real acceleration of the cloth, frame-resolved) adds
    a 'fwap' transient scaled by the rise.
  * SNAPS: every sim snap event -> whip crack (N-wave, high-passed) + body 'thwack' + tiny low thump.
  * Air absorption for distant (low-gain) flags; constant-power pan per sample from the sim's pan track.
Crowd-scale flags (kind='crowd' tracks and raise/salute events) are sonified as grains: every raised flag is a
short swing-up swish + unfurl flutter + pop, placed at its own time/pan/distance, over a bed whose level
grows with sqrt(number of flags raised) (incoherent sum of N flutters).
"""
import json
import zlib
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

import foley_lib as L
from foley_lib import SR, ns, pan2, R

import os as _os

ROOT = Path(__file__).resolve().parents[1]
_FILM = _os.environ.get("VX_FILM", "unbabeling")
# film-aware (vx.config layout): film 1 keeps the legacy root cache, other films use films/<name>/cache/foley
FOLEY_CACHE = (ROOT if _FILM == "unbabeling" else ROOT / "films" / _FILM) / "cache" / "foley"


# ------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------
def load_tracks():
    out = []
    for p in sorted(FOLEY_CACHE.glob("*.npz")):
        try:
            z = np.load(p, allow_pickle=True)
        except Exception as e:  # half-written file from a concurrent export
            print(f"[foley] skip {p.name}: {e}")
            continue
        d = {k: z[k] for k in z.files}
        scene = p.stem.split("_")[0]
        name = p.stem[len(scene) + 1:]
        tr = dict(file=p.name, scene=scene, name=name, kind=str(d.get("kind", "flag")),
                  f0=int(d.get("f0", 0)), fps=float(d.get("fps", 24)),
                  energy=np.nan_to_num(np.asarray(d["energy"], np.float64)),
                  pan=np.nan_to_num(np.asarray(d.get("pan", np.zeros_like(d["energy"])), np.float64)),
                  gain=float(d.get("gain", 1.0)),
                  events=np.asarray(d.get("events", np.zeros((0, 3))), np.float64).reshape(-1, 3))
        for k in ("flap_hz", "snap"):
            if k in d:
                tr[k] = np.nan_to_num(np.asarray(d[k], np.float64))
        if len(tr["energy"]) < 2:
            continue
        out.append(tr)
    return out


def load_events():
    out = []
    for p in sorted(FOLEY_CACHE.glob("*_events.json")):
        try:
            evs = json.loads(p.read_text())
        except Exception as e:
            print(f"[foley] skip {p.name}: {e}")
            continue
        scene = p.stem.split("_")[0]
        for e in evs:
            e = dict(e)
            e["scene"] = scene
            e["file"] = p.name
            e["kind"] = str(e.get("kind", "hit")).lower()
            e["desc"] = str(e.get("desc", ""))
            out.append(e)
    return sorted(out, key=lambda e: e["t"])


# ------------------------------------------------------------------------------------------------
# the flutter engine
# ------------------------------------------------------------------------------------------------
def _upsample(y, fps, n_audio, pchip=True):
    tf = np.arange(len(y)) / fps
    ta = np.arange(n_audio) / SR
    if pchip and len(y) > 2:
        return np.clip(PchipInterpolator(tf, y, extrapolate=True)(np.minimum(ta, tf[-1])), 0, None)
    return np.interp(ta, tf, y)


def snap_crack(r, strength=1.0, bright=1.0):
    """cloth whip-crack: the trailing edge exceeds the speed of the surrounding air wave -> tiny N-wave,
    followed by the cloth's 'thwack' body and a faint low thump of the pushed air."""
    n = ns(0.16)
    x = np.zeros(n, np.float32)
    T = 0.0006 + 0.0006 * (1 - strength)
    nw = L.nwave(T)
    x[: len(nw)] += nw * 1.4 * strength
    x = L.hp(x, 900 * bright, 2)
    m = ns(0.09)
    body = L.svf(L.white(m, r), 380 + 900 * strength * bright, 1.4, "bp") * L.env_ad(m, 0.0008, 0.05)
    x[:m] += body * (0.9 * strength)
    th = L.lp(L.white(ns(0.06), r), 160, 2) * L.env_ad(ns(0.06), 0.001, 0.04)
    x[: len(th)] += th * 0.8 * strength
    return L.fade(x, 0.0002, 0.02)


def flutter_from_energy(energy, fps, pan=0.0, gain=1.0, events=(), flap_hz=None, seed=0, t0=0.0,
                        size=1.0, tail=0.8, field=False):
    """core engine. energy per frame -> stereo audio starting at local time 0 (global t0).
    size: cloth scale (1 = survey flag; >1 = huge: lower flap rate & pitch).
    field: a field of many flags driven by one energy track -> many detuned flap trains, wide & darker."""
    r = R(seed)
    energy = np.clip(np.asarray(energy, np.float64), 0, 2.0)
    nfr = len(energy)
    dur = nfr / fps + tail
    n = ns(dur)
    e = _upsample(np.append(energy, 0.0), fps, n)
    e = L.lp(e.astype(np.float32), 30, 1).astype(np.float64).clip(0, None)
    panA = np.interp(np.arange(n) / SR, np.arange(nfr) / fps, np.broadcast_to(pan, (nfr,)))
    if flap_hz is not None and np.any(np.asarray(flap_hz) > 0):
        fh = np.asarray(flap_hz, np.float64)
        fh = np.where(fh > 0.2, fh, 1.6 + 9.0 * energy ** 0.8)
        fz = np.interp(np.arange(n) / SR, np.arange(nfr) / fps, fh)
    else:
        fz = (1.6 + 9.0 * e ** 0.8)
    fz = np.clip(fz / size, 0.15, 22.0)
    sharp = (5.0 + 22.0 * np.clip(e, 0, 1)) / size ** 0.8   # huge cloth: slow broad billows, not clicks

    def pulses(k_trains):
        acc = np.zeros(n)
        for k in range(k_trains):
            mul = 1.0 if k_trains == 1 else r.uniform(0.7, 1.35)
            ph = np.cumsum(fz * mul * (1 + 0.12 * L.smooth_noise(n, r, 1.5, 2))) / SR + r.random()
            frac = ph % 1.0
            cyc = np.floor(ph).astype(np.int64)
            cyc_gain = r.uniform(0.55, 1.0, cyc.max() + 2)[cyc]
            acc += (1 - np.exp(-frac / 0.02)) * np.exp(-frac * sharp) * cyc_gain
        return acc / np.sqrt(k_trains)

    de = np.diff(energy, prepend=energy[0])
    pk, _ = find_peaks(de, height=0.03, distance=2)

    def mono():
        # --- flap pulse train(s) (phase from the sim's flap rate, turbulence jitter)
        pulse = pulses(6 if field else 1)
        flap = L.svf(L.pink(n, r), (170 + 1100 * np.clip(e, 0, 1.2)) / size ** 0.5, 1.1, "bp") * pulse
        # --- rustle
        rust = L.svf(L.pink(n, r), (700 + 3500 * np.clip(e, 0, 1.2)) / size ** 0.4, 0.6, "bp")
        crinkle = L.bp(L.poisson_clicks(r, n, 60 + 900 * e, np.clip(e, 0, 1)), 2000, 9000) * 0.35
        # --- whips at frame-resolved energy rises (the sim's accelerations)
        whips = np.zeros(n, np.float32)
        for i in pk:
            s = float(np.clip(de[i] * 3.0, 0, 1))
            m = ns(0.07)
            w = L.svf(L.white(m, r), 300 + 1600 * s, 1.3, "bp") * L.env_ad(m, 0.004, 0.05)
            a = ns(i / fps)
            if a + m < n:
                whips[a:a + m] += w * s
        x = flap * (0.9 * np.clip(e, 0, 1.5) ** 1.1) + rust * e ** 1.35 * 0.55 + crinkle + whips * 0.8
        # --- huge cloth: a continuous deep air-roar that follows the energy, billowing slowly with each flap
        if size > 1.5:
            bill = 0.55 + 0.45 * pulse / (pulse.max() + 1e-9)
            lo = L.svf(L.brown(n, r), 60 + 80 * np.clip(e, 0, 1.2), 0.8, "lp") * bill * np.clip(e, 0, 1.2)
            x = x * 0.5 * bill + lo * 0.35
        # --- snaps from the simulation
        for (te, st, sp) in events:
            a = ns(te - t0)
            if 0 <= a < n:
                c = snap_crack(r, float(np.clip(st, 0.05, 1.0)), 1.0 / size ** 0.3)
                b = min(n, a + len(c))
                x[a:b] += c[: b - a] * (0.4 if field else 0.7)
        return L.f32(x)

    if field:
        st = L.stereo(mono(), mono())
        st = L.lp(st, 2500 + 9000 * min(gain, 1.0), 1)
        st = pan2(st, np.clip(panA * 0.5, -1, 1)) * gain
    else:
        x = mono()
        if gain < 0.6:  # distance: air absorption from gain (closeness)
            x = L.lp(x, 2500 + 12000 * gain, 1)
        st = pan2(x, np.clip(panA, -1, 1)) * gain
    return L.f32(st)


def track_audio(tr, seed=0):
    """cache/foley track -> (stereo audio, global start seconds, report dict)."""
    fps = tr["fps"]
    t0 = tr["f0"] / fps
    en = tr["energy"]
    evs = list(tr["events"])
    if not evs and "snap" in tr:  # derive events from per-frame snap
        for i in np.nonzero(tr["snap"] > 0.05)[0]:
            evs.append((t0 + i / fps, float(tr["snap"][i]), float(tr["pan"][i])))
    size = 1.0
    nm = (tr["name"] + tr["kind"]).lower()
    if any(k in nm for k in ("colossal", "giant", "10000", "gate", "huge")):
        size = 6.0
    field = any(k in nm for k in ("field", "crowd", "carpet"))
    x = flutter_from_energy(en, fps, tr["pan"], tr["gain"], evs, tr.get("flap_hz"),
                            seed=zlib.crc32(tr["file"].encode()) % 2 ** 31,
                            t0=t0, size=size, field=field)
    return x, t0, dict(size=size, n_events=len(evs), field=field)


def frame_rms(x, fps, nfr):
    m = x.mean(1) if x.ndim == 2 else x
    hop = SR / fps
    out = np.zeros(nfr)
    for i in range(nfr):
        a, b = int(i * hop), int((i + 1) * hop)
        seg = m[a:b]
        out[i] = np.sqrt((seg ** 2).mean()) if len(seg) else 0
    return out


def correlation(tr, x):
    """Pearson r between the sim's per-frame energy and the synthesized flutter's per-frame RMS."""
    en = tr["energy"]
    rms = frame_rms(x, tr["fps"], len(en))
    if en.std() < 1e-6 or rms.std() < 1e-9:
        return float("nan"), rms
    return float(np.corrcoef(en, rms)[0, 1]), rms


# ------------------------------------------------------------------------------------------------
# crowds
# ------------------------------------------------------------------------------------------------
_GRAIN_CACHE = {}


def raise_grain(r, dist=0.0, snap=True):
    """one flag being swung up and unfurling: swish (arc through the air) + 3-5 decaying flaps + pop.
    dist 0 (near) .. 1 (far): darker and softer."""
    dur = 0.9
    n = ns(dur)
    t = L.tt(n)
    sw_len = 0.22
    bell = np.exp(-((t - sw_len * 0.6) / (sw_len * 0.35)) ** 2)
    sw = L.svf(L.pink(n, r), 500 + 2200 * bell, 1.2, "bp") * bell * 0.8
    fl_rate = r.uniform(6, 10)
    ph = np.clip(t - sw_len * 0.7, 0, None) * fl_rate
    fl = np.exp(-(ph % 1.0) * 9) * (t > sw_len * 0.7) * np.exp(-np.clip(t - sw_len, 0, None) / 0.25)
    fl = L.svf(L.pink(n, r), 900, 1.0, "bp") * fl * 0.7
    x = sw + fl
    if snap:
        c = snap_crack(r, r.uniform(0.3, 0.7))
        a = ns(sw_len * 0.75)
        x[a:a + len(c)] += c[: n - a] * 0.6
    fc = 16000 * (1 - dist) + 1200 * dist
    x = L.lp(L.f32(x), fc, 1) if fc < 15000 else L.f32(x)
    return L.fade(x, 0.003, 0.05)


def grain_bank(seed=7, per=12, levels=5):
    key = (seed, per, levels)
    if key not in _GRAIN_CACHE:
        r = R(seed)
        _GRAIN_CACHE[key] = [[raise_grain(r, lv / max(1, levels - 1), snap=r.random() < 0.7) for _ in range(per)]
                             for lv in range(levels)]
    return _GRAIN_CACHE[key]


def crowd_wave(t0, dur, raises, bed=True, seed=3, bed_level=0.5):
    """raises: array [(t_global, pan, dist01, strength)]. Returns (stereo, start_global)."""
    r = R(seed)
    bank = grain_bank()
    n = ns(dur + 1.5)
    out = np.zeros((n, 2), np.float32)
    raises = np.asarray(raises, np.float64).reshape(-1, 4)
    for (t, p, d, s) in raises:
        lv = int(np.clip(round(d * (len(bank) - 1)), 0, len(bank) - 1))
        g = bank[lv][r.integers(len(bank[lv]))]
        a = ns(t - t0)
        if a < 0 or a >= n:
            continue
        st = pan2(g, p) * (s * (1.0 - 0.75 * d))
        b = min(n, a + len(st))
        out[a:b] += st[: b - a]
    if bed and len(raises):
        # massed flutter bed: level ∝ sqrt(count raised so far); gets darker as the front moves away
        ta = np.arange(n) / SR + t0
        cnt = np.searchsorted(np.sort(raises[:, 0]), ta)
        lvl = np.sqrt(cnt / max(1, len(raises)))
        md = np.interp(ta, np.sort(raises[:, 0]), np.maximum.accumulate(raises[np.argsort(raises[:, 0]), 2]))
        for ch in range(2):
            fl = L.svf(L.pink(n, r), 600 + 2600 * (1 - 0.6 * md), 0.7, "bp")
            am = 0.7 + 0.3 * L.smooth_noise(n, r, 7.0, 2)
            out[:, ch] += fl * lvl * am * bed_level
    return L.norm(out, 0.9), t0


def procedural_wave(t0, dur_front, n_flags=2400, origin=(0.0, 0.0), speed=None, seed=11, width=1.0,
                    near_first=True, jitter=0.12):
    """fallback crowd wave: flags scattered on a plain; each raises when the front (from origin) reaches it."""
    r = R(seed)
    rad = np.sqrt(r.uniform(0.0004, 1.0, n_flags))
    ang = r.uniform(0, 2 * np.pi, n_flags)
    x = rad * np.cos(ang)
    t = t0 + rad * dur_front + r.normal(0, jitter, n_flags) * rad
    t = np.maximum(t, t0)
    pan = np.clip(x * 0.95 * width, -1, 1)
    dist = np.clip(rad, 0, 1)
    strength = 1.0 / (1 + 6 * rad) * r.uniform(0.6, 1.0, n_flags)
    return np.stack([t, pan, dist, strength], -1)

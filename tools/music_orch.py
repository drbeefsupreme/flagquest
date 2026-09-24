"""Orchestration helpers shared by the scores of both films (owned by Composer).

Chord/voicing helpers, sampled-orchestra players (strings, brass, timpani, harp, percussion, tutti hits),
glass/bell voices, cut-offs, and mastering utilities (LUFS, true peak, look-ahead limiter). Film-agnostic:
everything takes global times and writes into a music_lib.Buf; DUR/N come from the active film (VX_FILM).
"""
import os

import numpy as np

from music_lib import (ROOT, SR, SF2, S, Sampler, _resample, db, fm_bell, glass, hz, m, oneshot,  # noqa: F401
                       stereo_place)

CH = {"D": (2, [0, 4, 7]), "Dm": (2, [0, 3, 7]), "G": (7, [0, 4, 7]), "Em": (4, [0, 3, 7]), "A": (9, [0, 4, 7]),
      "Bm": (11, [0, 3, 7]), "Bb": (10, [0, 4, 7]), "C": (0, [0, 4, 7]), "F": (5, [0, 4, 7]), "Gm": (7, [0, 3, 7]),
      "F#m": (6, [0, 3, 7]), "Bbm": (10, [0, 3, 7]), "A7": (9, [0, 4, 7, 10]), "Cm": (0, [0, 3, 7]),
      "B": (11, [0, 4, 7]), "F#": (6, [0, 4, 7]), "Asus4": (9, [0, 5, 7]), "Dadd9": (2, [0, 4, 7, 2])}


def tones(ch, lo, n):
    """first n chord tones >= lo (midi)"""
    root, iv = CH[ch]
    pcs = {(root + i) % 12 for i in iv}
    out, x = [], m(lo)
    while len(out) < n:
        if x % 12 in pcs:
            out.append(x)
        x += 1
    return out


def root(ch, lo):
    r = CH[ch][0]
    x = m(lo)
    while x % 12 != r:
        x += 1
    return x


# ------------------------------------------------------------------------------------------------ players
_INS = {}


def I(key, **kw):
    k = (key, tuple(sorted(kw.items())))
    if k not in _INS:
        _INS[k] = Sampler(key, **kw)
    return _INS[k]


PAN = dict(vln=-0.5, vln2=-0.22, vla=0.12, vc=0.36, cb=0.52, hn=-0.32, tpt=0.24, tbn=0.4, tuba=0.52,
           timp=0.0, harp=-0.58, cel=0.42, glock=0.3, fl=-0.14, ob=0.1, cl=-0.06, bsn=0.18, org=0.0)
STR_KEYS = dict(v1=("vln", "vln"), v2=("vln", "vln2"), va=("vla", "vla"), vc=("vc", "vc"), cb=("cb", "cb"))


def strings(b, t, dur, art="sus", vel=0.6, dyn=None, att=0.08, rel=0.7, bus="str", gain=1.0, bend=None, **parts):
    """parts: v1=[notes], v2=[...], va=[...], vc=[...], cb=[...]  (art: sus|trem|spic|pizz)"""
    for part, notes in parts.items():
        key, pk = STR_KEYS[part]
        from music_lib import index
        name = f"{key}_{art}" if f"{key}_{art}" in index() else f"{key}_sus"
        inst = I(name)
        for j, n in enumerate(notes):
            one = art in ("spic", "pizz")
            inst.note(b, bus, t, n, None if one else dur, vel, PAN[pk] + 0.06 * j, gain=gain * (0.8 if key == "cb" else 1.0),
                      dyn=dyn if not one else None, att=att if not one else 0.0, rel=rel, bend=bend,
                      seed=int(t * 997) + j + sum(map(ord, part)) % 7)  # stable across runs (str hash is salted)
            if key == "cb" and not one:  # solo bass doubled -> small section
                inst.note(b, bus, t + 0.012, n, dur, vel * 0.9, PAN[pk] - 0.1, gain=gain * 0.6, dyn=dyn, att=att,
                          rel=rel, bend=bend, seed=int(t * 131) + 5)


def brass(b, t, dur, art="sus", vel=0.6, dyn=None, att=0.03, rel=0.5, bus="brs", gain=1.0, bend=None, **parts):
    """parts: tpt=[..], hn=[..], tbn=[..], tuba=[..]; art sus|stac (tpt also 'mute')"""
    for part, notes in parts.items():
        key = f"{part}_{art}" if not (part == "tpt" and art == "mute") else "tpt_mute"
        inst = I(key)
        one = art == "stac"
        for j, n in enumerate(notes):
            inst.note(b, bus, t, n, None if one else dur, vel, PAN[part] + 0.07 * j - 0.035 * len(notes), gain=gain,
                      dyn=None if one else dyn, att=0.0 if one else att, rel=rel, bend=bend, seed=int(t * 577) + j)


def timp(b, t, note, vel=0.8, gain=1.0, bus="perc"):
    I("timp").note(b, bus, t, note, None, vel, 0.0, gain=gain * 0.55, width=0.7)


def timp_roll(b, t, dur, note, dyn, gain=1.0, bus="perc"):
    I("timp_roll").note(b, bus, t, note, dur, 0.8, 0.0, gain=gain * 0.5, dyn=dyn, att=0.05, rel=0.25, width=0.7)


def harp_roll(b, t, notes, vel=0.5, spread=0.03, gain=1.0, bus="keys", pan=None):
    for j, n in enumerate(sorted(m(x) for x in notes)):
        I("harp").note(b, bus, t + j * spread, n, None, vel, PAN["harp"] if pan is None else pan, gain=gain * 0.28,
                       seed=j)


def harp_gliss(b, t0, t1, lo, hi, scale=(2, 4, 6, 9, 11), vel=0.5, gain=1.0, down=False, bus="keys"):
    ns = [x for x in range(m(lo), m(hi) + 1) if x % 12 in scale]
    if down:
        ns = ns[::-1]
    for j, n in enumerate(ns):
        tj = t0 + (t1 - t0) * (j / max(1, len(ns) - 1)) ** 1.15
        I("harp").note(b, bus, tj, n, None, vel * (0.8 + 0.2 * j / len(ns)), PAN["harp"] + 0.4 * j / len(ns),
                       gain=gain * 0.22, seed=j)


PERC = "Percussion/"
V1P = "VSCO 1 Percussion/"


def perc(b, t, f, gain=1.0, pan=0.0, width=0.8, bus="perc", **kw):
    return oneshot(b, bus, t, f, gain=gain, pan=pan, width=width, **kw)


def crash(b, t, level="ff", gain=1.0, pan=0.15, dur=None):
    # clash pairs chosen for tight (flam-free) attacks: the main crash lands within ~8 ms of t
    f, g = {"ff": (V1P + "varMetal/Cymbals/clash/crash_hit_fff_loose_2.wav", 0.8),
            "fff": (V1P + "varMetal/Cymbals/clash/crash_hit_fff_loose_2.wav", 1.0),
            "mp": (PERC + "cymbal-crash1_mp_rr1.wav", 1.0),
            "pp": (PERC + "cymbal-crash1_pp_rr1.wav", 1.0)}[level]
    perc(b, t, f, gain * 0.28 * g, pan, dur=dur, align="peak")


DYN_ORDER = ["ppp", "pp", "p", "mp", "mf", "f", "ff", "fff"]


def taiko(b, t, level="ff", gain=1.0, pan=0.0, sticks=False, rr=1):
    """giant ethnic drum (mallet = taiko-like boom, sticks = drier crack). Nearest available dynamic."""
    d = V1P + ("drums/other/ethnic/giant/sticks/" if sticks else "drums/other/ethnic/giant/mallet/")
    pre = "EthnicLargeSticks_hit_" if sticks else "EthnicLargeMallet_hit_"
    have = sorted({f[len(pre):].split("_")[0] for f in os.listdir(os.path.join(ROOT, "assets", "sf2", "vsco2", d))
                   if f.startswith(pre)}, key=DYN_ORDER.index)
    lv = min(have, key=lambda h: abs(DYN_ORDER.index(h) - DYN_ORDER.index(level)))
    fs = sorted(f for f in os.listdir(os.path.join(ROOT, "assets", "sf2", "vsco2", d)) if f.startswith(pre + lv + "_"))
    perc(b, t, d + fs[(rr - 1) % len(fs)], gain * 0.9, pan, 0.6)


def swell_to(b, t_peak, f, gain=1.0, after=0.15, pan=0.0, bus="perc", cut=True):
    """align the loudest point of a crescendo sample (e.g. cymbal roll) to t_peak; cut `after` s later."""
    from music_lib import load_wav
    x, sr = load_wav(os.path.join("assets", "sf2", "vsco2", f))
    env = np.convolve(np.abs(x).max(1), np.ones(441) / 441, "same")
    pk = int(np.argmax(env))
    y = _resample(x, 0.0, np.array([sr / SR]), int(len(x) * SR / sr) - 4)
    pk48 = int(pk * SR / sr)
    if cut:
        k = S(after)
        y = y[:pk48 + k].copy()
        y[pk48:] *= np.linspace(1, 0, len(y) - pk48)[:, None] ** 2
    y = stereo_place(y, pan, 0.9)
    b.add(bus, y, i0=S(t_peak) - pk48, gain=gain)


def cut_after(b, t, fade=0.03):
    """cut-off: everything already written to b fades out over `fade` s ending at t, silence after t."""
    a, z = S(t - fade), S(t)
    for k in b.b:
        b.b[k][a:z] *= (np.cos(np.linspace(0, np.pi / 2, z - a)) ** 2)[:, None]
        b.b[k][z:] = 0


def sf2part(path, bank, preset, gain=1.0):
    return SF2(path, bank, preset, gain=gain)


def glass_note(b, t, n, dur=3.0, gain=1.0, pan=0.0, bus="space", decay=None, att=0.004):
    y = glass(hz(m(n)), dur, att=att, decay=decay or dur * 0.6)
    b.add(bus, stereo_place(np.stack([y, y], 1), pan, 1.0), t=t, gain=gain)


def bell_note(b, t, n, dur=3.5, gain=1.0, pan=0.0, bus="keys", ratio=3.5, index=1.6, decay=1.4):
    y = fm_bell(hz(m(n)), dur, ratio=ratio, index=index, decay=decay)
    b.add(bus, stereo_place(np.stack([y, y], 1), pan, 1.0), t=t, gain=gain)


def tutti(b, t, ch, level=1.0, perc_on=True, sus=0.0, crash_on=True, tpt=True, gong=False, low=True, crash_dur=None):
    """orchestral hit. sus>0 adds sustained strings+brass for sus seconds."""
    v = 0.55 + 0.45 * level
    g = level
    brass(b, t, None, "stac", v, tpt=tones(ch, "D5", 3) if tpt else [], hn=tones(ch, "D4", 4),
          tbn=[root(ch, "C3"), root(ch, "C3") + 7] if low else [], tuba=[root(ch, "C2")] if low else [], gain=g)
    strings(b, t, None, "spic", v, v1=tones(ch, "D5", 2), v2=tones(ch, "A4", 2), va=tones(ch, "D4", 2),
            vc=[root(ch, "C3"), root(ch, "C3") + 7], cb=[root(ch, "C2")] if low else [], gain=g)
    if sus > 0:
        dyn = [(0, 0.9 * level), (0.4, 0.55 * level), (sus, 0.45 * level)]
        strings(b, t, sus, "sus", dyn=dyn, att=0.0, v1=tones(ch, "A5", 2), v2=tones(ch, "D5", 2),
                va=tones(ch, "F4", 2), vc=[root(ch, "C3")], cb=[root(ch, "C2")] if low else [], gain=g * 0.9)
        brass(b, t, sus, "sus", dyn=[(0, 0.8 * level), (0.5, 0.45 * level), (sus, 0.4 * level)], att=0.0,
              hn=tones(ch, "D4", 3), tbn=[root(ch, "C3")], gain=g * 0.8)
    if perc_on:
        r = root(ch, "C2")
        timp(b, t, r if r >= 38 else r + 12, 0.95 * level + 0.05, gain=g)
        taiko(b, t, "ff" if level > 0.7 else "f", gain=g * 0.55)
        if crash_on:
            crash(b, t, "ff" if level > 0.6 else "mp", gain=g, dur=crash_dur)
        if gong:
            perc(b, t, PERC + "gongHit_fff.wav", 0.5 * g, 0.0)


def lufs(x):
    import pyloudnorm as pyln
    return pyln.Meter(SR).integrated_loudness(x.astype(np.float64))


def true_peak(x):
    from scipy.signal import resample_poly
    return float(np.abs(resample_poly(x, 4, 1, axis=0)).max())


def limiter(x, ceiling_db=-1.2, look=0.004, rel=0.12):
    """lookahead peak limiter on 4x-oversampled peak estimate."""
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import resample_poly
    c = db(ceiling_db)
    up = np.abs(resample_poly(x, 4, 1, axis=0)).max(1)
    pk = up.reshape(-1, 4).max(1)[:len(x)]
    need = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    la = int(look * SR)
    need = -maximum_filter1d(-need, size=2 * la + 1)
    g = np.empty_like(need)
    s = 1.0
    ar = np.exp(-1 / (rel * SR))
    for i in range(len(need)):
        s = need[i] if need[i] < s else ar * s + (1 - ar) * need[i]
        g[i] = s
    g = np.convolve(np.pad(g, (la, la), mode="edge"), np.ones(la) / la, "same")[la:-la]
    g = np.minimum(g, need)
    # report where the limiter works hardest (>3 dB), so hits can be re-balanced at the source
    red = np.flatnonzero(g < db(-3))
    if len(red):
        spans, st = [], red[0]
        for p_, q_ in zip(red[:-1], red[1:]):
            if q_ - p_ > SR // 10:
                spans.append((st, p_))
                st = q_
        spans.append((st, red[-1]))
        print("  limiter >3 dB at: " + ", ".join(f"{a_ / SR:.2f}s({20 * np.log10(g[a_:b_ + 1].min()):.1f}dB)"
                                                 for a_, b_ in spans[:20]), flush=True)
    return (x * g[:, None]).astype(np.float32), float(g.min())

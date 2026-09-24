"""THE FLAG GAME? — builds films/flaggame/audio/stems/crowd.wav from Kokoro voices.   Owner: Vocalist.

 0.0-18.4     t01 camp walla in the rain (distant, relaxed chatter)                          ~-38 LUFS
 56.0-87.0    t03/t04 engraved crowd murmur (grumbling hippies; a collective groan on 'WEEKEND' 64.4)
 76.99        X01 "Shut up!" mob support (16 shouted elders, 'Shut' onsets +-35 ms)
 83.0-97.67   THE FLAGS CHANT: ragged scattered 'FLAGS!' under M02, then one 'FLAGS' per beat on
              Composer's grid (films/flaggame/audio/music/grid.json: 126 -> 154.1 BPM, beat 20 = 95.722
              flood_peak), 4 -> 44 voices, jitter 70 -> 12 ms, then thinning to 97.67.  Written to
              films/flaggame/cache/foley/chant_grid.json for T04's kinetic type.
 102.4-104.9  the crowd lands back on the Devil's V02 'FLAGS! FLAGS FLAGS FLAGS!' (gated at 105.67)
 108.3-112.5  "Found one!" era cheers
 118.92       gasp at the rain freeze -> held, rising "ooooh" -> cheer by ~120.5
 122.4-129.73 burn night: "BURN IT!" chant on the burn drum beats, whoops, roar at ignite 125.356;
              cut short after flames_reach_flag 129.729
 137.12-142.0 frenzied screams at the deluge, drowned out by 142.0
Everything dips 6 dB under dialogue lines (not under X01, M03, V02, which the crowd is meant to drown).

usage: source films/flaggame/env.sh && python films/flaggame/tools/fg_vocal_crowd.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pedalboard import Compressor, HighpassFilter, Pedalboard, PeakFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from vocal_crowd import cached, place, say, shouted, whisper
from vocal_lib import (FILM, FILM_ROOT, NS, SR, TL, VOICES, add_at, bandpass, convolve, db, fade, kokoro, lufs,
                       make_ir, rng_for, smoothstep, up48, write_stem)

assert FILM == "flaggame", "source films/flaggame/env.sh first (VX_FILM=flaggame)"
GRID = json.loads((FILM_ROOT / "audio/music/grid.json").read_text())
EN = VOICES["a"] + VOICES["b"]
BAD = {"af_nicole", "bm_lewis"}
CROWD_V = [v for v in EN if v not in BAD]
NO_DIP = {"X01", "M03", "V02"}

WALLA = ["Is it still raining?", "Dude, my tent is a lake.", "Where's the coffee camp?", "I love your tutu!",
         "Have you seen my goggles?", "This mud is so sticky.", "Hugs! Come here!", "Is the art car running?",
         "I lost a boot.", "Who has a towel?", "We should go to the temple later.", "Is that a flag?",
         "Nice fur!", "Does anyone have tape?", "Let's go dance.", "Oh my god, hi!"]
GRUMBLE = ["Ugh, rain.", "Where's my dirt?", "This is the worst burn ever.", "My whole camp is soaked.",
           "Can we just go home?", "It smells so bad.", "Ugh, the mud.", "Who packed the tarps?",
           "I'm so wet.", "This hurricane is our fault.", "Nobody moved anything.", "Ugh."]
CHEERS = ["Found one!", "Woo!", "Yeah!", "Move it!", "Over here!", "Flag!", "Woohoo!", "Yes!"]
SCREAMS = ["Aah!", "No!", "Help!", "Run!", "The sky!", "Oh no!"]


def onset(x, frac=0.12):
    e = np.abs(x)
    k = np.flatnonzero(e > e.max() * frac)
    return (k[0] / SR) if len(k) else 0.0


def yell(text, voice, semis=4.0, expo=1.25, nb=0.6, vow=1.0, speed=1.0):
    """Shouted by prosody injection (raised & widened F0, more vocal effort, optionally drawn-out vowels);
    keeps the model's natural lead-in (onset consonants intact).  48 kHz mono."""
    def fn():
        K = kokoro()
        ps = K.phonemes(text, voice[0])

        def dur_fn(d):
            d = d.copy()
            if vow != 1.0:
                for j, c in enumerate(ps):
                    if c in "æɑʌɔɛIAOWiuɪʊ":
                        d[j + 1] = max(1, int(round(d[j + 1] * vow)))
            return d

        def f0_fn(F0, d):
            vz = F0 > 40
            med = np.median(F0[vz]) if vz.any() else 150.0
            return np.where(vz, med * 2 ** (semis / 12) * (np.maximum(F0, 1) / med) ** expo, F0)
        return K.forward(ps, voice, speed, dur_fn=dur_fn, f0_fn=f0_fn, n_fn=lambda N, d: N + nb * (N > -3))[0]
    a = cached("fg_yell", ("y1", text, voice, semis, expo, nb, vow, speed), fn)
    x = up48(a)
    pk = np.abs(x).max() + 1e-9
    x = 0.7 * x + 0.3 * np.tanh(3 * x / pk) * pk
    return Pedalboard([HighpassFilter(130), PeakFilter(2900, 3.0, 0.9)])(x[None, :].astype(np.float32), SR)[0]


def held(ps, voice, hold, semis0, semis1, nb=0.8, vib=0.3):
    """a held vowel (ps has one vowel) gliding semis0 -> semis1 re: the voice's median, with vibrato."""
    def fn():
        K = kokoro()
        k = next(j for j, c in enumerate(ps) if c in "æɑʌɔɛIAOWiuɪʊ") + 1

        def dur_fn(d):
            d = d.copy()
            d[k] = max(6, int(hold / 0.025))
            return d

        def f0_fn(F0, d):
            vz = F0 > 40
            med = np.median(F0[vz]) if vz.any() else 150.0
            t = np.arange(len(F0)) / 80.0
            s = semis0 + (semis1 - semis0) * smoothstep(t / max(hold, 0.1)) + vib * np.sin(2 * np.pi * 6 * t)
            c = np.concatenate([[0], np.cumsum(d)]) * 2
            w = np.clip(F0 / 60, 0, 1)
            w[int(c[k]) + 2:int(c[k + 1])] = 1
            return w * med * 2 ** (s / 12) + (1 - w) * F0

        def n_fn(N, d):
            c = np.concatenate([[0], np.cumsum(d)]) * 2
            out = N.copy()
            a, b = int(c[k]) + 2, int(c[k + 1])
            out[a:b] = np.maximum(N[a:b], N.max() + nb - 1)
            return out
        return K.forward(ps, voice, dur_fn=dur_fn, f0_fn=f0_fn, n_fn=n_fn)[0]
    return up48(cached("fg_held", ("h1", ps, voice, round(hold, 2), semis0, semis1, nb, vib), fn))


def distant(x, d):
    return bandpass(x, 120, 9000 / d ** 0.6, order=1) * db(-18 * np.log10(d))


# ------------------------------------------------------------------------------------------------ sections
def walla(buf, t0, t1, texts, rate, seed, fn=say):
    r = rng_for("walla", seed)
    t = t0
    while t < t1:
        t += float(r.exponential(1 / rate))
        v = CROWD_V[int(r.integers(0, len(CROWD_V)))]
        x = up48(fn(texts[int(r.integers(0, len(texts)))], v, float(r.uniform(0.95, 1.15))))
        place(buf, distant(x, float(r.uniform(1.5, 5))), t, float(r.uniform(-1, 1)), 0.0)


def shut_up(buf):
    r = rng_for("shutup")
    for i in range(16):
        v = CROWD_V[(i * 5) % len(CROWD_V)]
        x = yell("Shut up!", v, float(r.uniform(2, 5)), 1.3, 0.7)
        place(buf, x * db(float(r.uniform(-4, 0))), 76.99 + float(r.normal(0, 0.02)) - onset(x),
              float(r.uniform(-0.8, 0.8)))
    for i in range(8):                                    # grumbles after
        v = CROWD_V[(i * 7 + 3) % len(CROWD_V)]
        x = up48(say(["Blasphemy.", "Heretic.", "Get out.", "Nonsense."][i % 4], v))
        place(buf, distant(x, 2.5), 77.4 + float(r.uniform(0, 0.8)), float(r.uniform(-1, 1)), 0)


def chant(buf):
    beats = GRID["chant"]["beats"]
    r = rng_for("chant")
    used = []
    # ragged scattered voices under M02
    t = 83.2
    while t < 87.1:
        t += float(r.exponential(1 / (1.0 + 2.5 * smoothstep((t - 83.2) / 3.9))))
        v = CROWD_V[int(r.integers(0, len(CROWD_V)))]
        x = yell("FLAGS!", v, float(r.uniform(1, 4)), 1.2, 0.3)
        place(buf, distant(x, float(r.uniform(1.5, 3.5))), t, float(r.uniform(-1, 1)), -3)
    for b, tb in enumerate(beats):
        u = min(1.0, b / 20)
        n = int(round(4 + 40 * u ** 1.3)) if b <= 20 else int(round(44 * (1 - (b - 20) / 6)))
        jit = 0.07 - 0.058 * u
        g = -14 + 14 * u ** 1.2 if b <= 20 else -3 * (b - 20)
        for i in range(n):
            v = CROWD_V[(b * 11 + i * 3) % len(CROWD_V)]
            x = yell("FLAGS!", v, float(r.uniform(3, 6)), 1.25, 0.8, speed=1.1 + 0.2 * u)
            vo = onset(x) + 0.07                          # vowel lands on the beat
            place(buf, x * db(g + float(r.uniform(-4, 0)) - 10 * np.log10(max(n, 1)) + 12),
                  tb + float(r.normal(0, jit)) - vo, float(r.uniform(-1, 1)), 0)
        used.append(round(tb, 4))
    out = FILM_ROOT / "cache/foley/chant_grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"beats": used, "peak": 95.722, "source": "Vocalist crowd.wav (Composer grid)"}))
    # the crowd lands on the Devil's V02 FLAGS (vowel onsets measured on the dry file)
    for tb in (102.415, 102.865, 103.315, 103.827):
        for i in range(18):
            v = CROWD_V[(i * 5 + int(tb * 10)) % len(CROWD_V)]
            x = yell("FLAGS!", v, float(r.uniform(4, 7)), 1.3, 0.9)
            place(buf, x * db(-6 + float(r.uniform(-4, 0))), tb + float(r.normal(0, 0.02)) - onset(x) - 0.07,
                  float(r.uniform(-1, 1)), 0)
    t = np.arange(NS) / SR
    g = np.where((t > 97.4) & (t < 102.0), np.clip(1 - (t - 97.4) / 0.35, 0, 1), 1.0)
    g = np.where(t >= 105.5, np.clip(1 - (t - 105.5) / 0.17, 0, 1), g)
    buf *= g[:, None].astype(np.float32)


def found_one(buf):
    r = rng_for("found")
    for i in range(18):
        v = CROWD_V[(i * 9 + 1) % len(CROWD_V)]
        x = yell(CHEERS[i % len(CHEERS)], v, float(r.uniform(3, 7)), 1.35, 0.6)
        place(buf, distant(x, float(r.uniform(1.2, 3))), float(r.uniform(108.4, 112.4)), float(r.uniform(-1, 1)), 0)


def awe(buf):
    r = rng_for("awe")
    for i in range(20):                                   # the gasp
        v = CROWD_V[(i * 3) % len(CROWD_V)]
        x = up48(whisper("Hah!", v, 1.2))
        place(buf, x * db(4), 118.92 + float(r.normal(0, 0.03)) - onset(x), float(r.uniform(-1, 1)), 0, fin=0.002)
    for i in range(24):                                   # ooooh, rising
        v = CROWD_V[(i * 5 + 2) % len(CROWD_V)]
        x = held("ˈu", v, float(r.uniform(1.3, 1.7)), -2, 3, nb=0.3)
        n = len(x)
        x = x * np.minimum(1, np.arange(n) / (0.5 * SR)).astype(np.float32)
        place(buf, distant(x, float(r.uniform(1.2, 2.5))), 119.1 + float(r.uniform(0, 0.3)), float(r.uniform(-1, 1)), -2)
    for i in range(24):                                   # cheer
        v = CROWD_V[(i * 7 + 4) % len(CROWD_V)]
        x = yell(["Yeah!", "Woo!", "Wow!", "Yes!"][i % 4], v, float(r.uniform(4, 8)), 1.4, 0.7)
        place(buf, distant(x, float(r.uniform(1.2, 3))), 120.45 + float(r.uniform(0, 0.8)), float(r.uniform(-1, 1)), 0)


def burn(buf):
    beats = GRID["burn"]["beats"]
    r = rng_for("burn")
    for b in range(0, len(beats) - 1, 2):                 # BURN (on beat) IT (next beat)
        n = 10 + b
        for i in range(n):
            v = CROWD_V[(b * 7 + i * 3) % len(CROWD_V)]
            x = yell("BURN IT!", v, float(r.uniform(3, 6)), 1.3, 0.7)
            place(buf, distant(x, float(r.uniform(1.2, 3))) * db(-6 + b * 0.4),
                  beats[b] + float(r.normal(0, 0.025)) - onset(x) - 0.06, float(r.uniform(-1, 1)), 0)
    for i in range(30):                                   # whoops + the roar at ignite
        v = CROWD_V[(i * 5 + 1) % len(CROWD_V)]
        roar = i >= 12
        t = 125.36 + float(r.uniform(-0.05, 0.4)) if roar else float(r.uniform(122.5, 125.2))
        x = yell(["Woo!", "Yeah!", "Burn!", "Whoo!"][i % 4], v, float(r.uniform(5, 9)), 1.4, 0.8, vow=1.6 if roar else 1)
        place(buf, distant(x, float(r.uniform(1.2, 3))), t - (onset(x) if roar else 0), float(r.uniform(-1, 1)), 0)
    t = np.arange(NS) / SR
    g = np.where((t > 122.0) & (t < 131.0), np.clip(1 - (t - 129.75) / 0.25, 0, 1), 1.0)
    buf *= g[:, None].astype(np.float32)


def deluge(buf):
    r = rng_for("deluge")
    for i in range(36):
        v = CROWD_V[(i * 7 + 2) % len(CROWD_V)]
        t0 = 137.12 + float(r.uniform(0, 3.0))
        if i % 2:
            x = held(["ˈɑ", "nˈO", "ˈA"][i % 3], v, float(r.uniform(0.8, 1.6)), float(r.uniform(4, 7)), float(r.uniform(7, 11)))
        else:
            x = yell(SCREAMS[i % len(SCREAMS)], v, float(r.uniform(6, 10)), 1.5, 1.0)
        pk = np.abs(x).max() + 1e-9
        x = 0.5 * x + 0.5 * np.tanh(3 * x / pk) * pk
        place(buf, distant(x, float(r.uniform(1.2, 4))), t0, float(r.uniform(-1, 1)), 0)
    t = np.arange(NS) / SR
    g = np.where((t > 136.0) & (t < 144.0), np.clip((142.0 - t) / 1.2, 0, 1), 1.0)
    buf *= g[:, None].astype(np.float32)


def section(fn, a, b, target, room_wet=-10.0, *args):
    buf = np.zeros((NS, 2), np.float32)
    fn(buf, *args)
    L = lufs(buf[int(a * SR):int(b * SR)])
    if np.isfinite(L):
        buf *= db(target - L)
    i0, i1 = int(max(0, a - 1) * SR), min(NS, int((b + 1) * SR))
    h = make_ir(rt60=1.1, predelay=0.015, lo_mult=0.8, hi_mult=0.45, er_n=6, er_span=0.05, width=1.0, seed=201)
    add_at(buf, convolve(buf[i0:i1], h) * db(room_wet), i0 / SR)
    return buf


def main():
    parts = [
        ("walla", section(lambda b: walla(b, 0.3, 18.2, WALLA, 2.2, "t01"), 0.0, 18.46, -38.0)),
        ("murmur", section(lambda b: walla(b, 56.0, 86.5, GRUMBLE, 1.6, "t03"), 56.0, 87.0, -37.0)),
        ("shutup", section(shut_up, 76.9, 78.2, -21.0)),
        ("chant", section(chant, 87.1, 97.7, -19.0)),
        ("found", section(found_one, 108.3, 112.5, -28.0)),
        ("awe", section(awe, 118.9, 121.5, -25.0)),
        ("burn", section(burn, 122.4, 129.8, -24.0)),
        ("deluge", section(deluge, 137.1, 142.0, -23.0)),
    ]
    mix = sum(b for _, b in parts)
    # 'WEEKEND' groan on F01 is covered by the grumbling murmur; dip under the leads
    t = np.arange(NS) / SR
    g = np.ones(NS, np.float32)
    for ln in TL["lines"]:
        if ln["id"] in NO_DIP:
            continue
        d = np.clip(np.minimum((t - ln["start"] + 0.1) / 0.08, (ln["end"] + 0.15 - t) / 0.25), 0, 1)
        g = np.minimum(g, (1 - d * (1 - db(-6))).astype(np.float32))
    mix *= g[:, None]
    mix[-int(0.05 * SR):] = 0
    pk = np.abs(mix).max()
    if pk > db(-1.5):
        mix *= db(-1.5) / pk
    write_stem("crowd", mix)
    rows = {}
    for name, (a, b) in {"walla": (0, 18.46), "murmur": (56, 87), "shutup": (76.9, 78.2), "chant": (87.1, 97.7),
                         "chant_peak": (95.2, 96.2), "found": (108.3, 112.5), "awe": (118.9, 121.5),
                         "burn": (122.4, 129.8), "deluge": (137.1, 142.0)}.items():
        rows[name] = round(float(lufs(mix[int(a * SR):int(b * SR)])), 1)
        print(f"  {name:10s} {a:6.1f}-{b:6.1f} {rows[name]:6.1f} LUFS", flush=True)
    (FILM_ROOT / "audio/vocal/crowd_levels.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

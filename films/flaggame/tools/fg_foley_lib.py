"""THE FLAG GAME? — film-2 Foley models (owner: Foley). Builds on the shared tools/foley_lib.py.

Rain is modelled per surface from its physics: every drop is an impulse (Poisson process whose rate follows the
storm intensity) exciting the surface it lands on — paper (stiff thin plate: bright short modes), mud (dull
low-passed splat), a camp shade / tarp (stretched membrane: low drum modes, drumming), puddles (Minnaert
bubbles entrained by the impact: the 'plink'), over a broadband wash (the millions of drops you can't resolve).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import foley_lib as L  # noqa: E402
from foley_lib import SR, Snd, ns, R, midi_hz  # noqa: E402

D_MAJOR = L.D_MAJOR


# ------------------------------------------------------------------------------------------------
# rain
# ------------------------------------------------------------------------------------------------
def rain_layer(r, t0, dur, pts, w, level=0.05, bright=1.0):
    """stereo rain over [t0, t0+dur) (global time axis may run past the film end: the caller wraps it).
    pts: [(t_global, intensity 0..1)], w: surface weights {hiss, paper, mud, shade, puddle}.
    level: RMS (linear) at intensity 1."""
    n = ns(dur)
    ta = t0 + L.tt(n)
    inten = np.interp(ta, [p[0] for p in pts], [p[1] for p in pts])
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        x = np.zeros(n, np.float64)
        if w.get("hiss"):
            hs = L.bp(L.pink(n, r), 1500 * bright, 12000) * 0.35 + L.svf(L.pink(n, r), 700 * bright, 0.7, "bp") * 0.2
            x += w["hiss"] * hs * inten ** 1.2
        if w.get("paper"):
            exc = L.poisson_clicks(r, n, 55 * inten, 1.0)
            x += w["paper"] * (L.resonators(exc, [850, 1650, 2900, 4600], [0.012, 0.010, 0.008, 0.006],
                                            [1, 0.7, 0.5, 0.3]) + L.hp(exc, 3000) * 0.5) * 0.25
        if w.get("mud"):
            exc = L.poisson_clicks(r, n, 140 * inten, 1.0)
            x += w["mud"] * (L.lp(exc, 1300 * bright) + L.resonators(exc, [170, 310], [0.02, 0.015], [0.4, 0.3])) * 0.18
        if w.get("shade"):
            exc = L.poisson_clicks(r, n, 240 * inten, 1.0)
            x += w["shade"] * (L.resonators(exc, [205, 335, 555, 820, 1240], [0.05, 0.04, 0.03, 0.025, 0.02],
                                            [1, 0.8, 0.6, 0.4, 0.3]) + L.hp(exc, 2500) * 0.25) * 0.12
        out[:, ch] = x
    if w.get("puddle"):
        rate = lambda u: 9.0 * w["puddle"] * float(np.interp(t0 + u, [p[0] for p in pts], [p[1] for p in pts]))
        out += L.bubbles_cloud(r, dur, rate, (1.2, 3.8), 1.0, (-0.9, 0.9), None, 0.2) * 0.25 * w["puddle"]
    full = inten > 0.8 * inten.max()
    rms = np.sqrt((out[full] ** 2).mean()) if full.any() else 1.0
    return L.f32(out * (level / max(rms, 1e-9)))


def shade_drips(r, dur, rate0=6.0, tau=1.2, level=1.0):
    """after the rain stops: the shade keeps dripping, slower and slower (drops into mud & puddles)."""
    evs, t = [], 0.0
    while t < dur:
        lam = rate0 * np.exp(-t / tau) + 0.25
        t += r.exponential(1.0 / lam)
        evs.append((t, L.drip(r, r.uniform(0.8, 1.4)).x, r.uniform(0.3, 1.0), r.uniform(-0.8, 0.8)))
    return L.norm(L.scatter(ns(dur + 0.5), evs), 0.9) * level


# ------------------------------------------------------------------------------------------------
# paper / comic
# ------------------------------------------------------------------------------------------------
def paper_tick(r):
    """the faint printed-paper 'tick' of a speech balloon popping onto the page."""
    n = ns(0.04)
    exc = np.zeros(n, np.float32)
    exc[0] = 1.0
    x = L.resonators(exc, [1250, 2350, 3900], [0.008, 0.006, 0.004], [1, 0.6, 0.4])
    x += L.hp(L.white(n, r), 2500) * L.env_ad(n, 0.0003, 0.006) * 0.6
    return Snd(L.fade(L.norm(x, 0.8), 0.0002, 0.01), 0.0)


def panel_glide(r, dur=0.6):
    """a comic panel gliding past: soft airy paper whoosh."""
    return L.whoosh(r, dur, dur * 0.5, 600, 4500, 1.2, -0.4, 0.4, dark=0.0)


def paper_from_energy(energy, fps, pan=0.0, gain=1.0, seed=0):
    """PHYSICS: paper-sim energy -> rustle (band rises with speed), page-flap slaps at every energy surge,
    crinkle ticks, and the low air push of the pages."""
    r = R(seed)
    en = np.clip(np.asarray(energy, np.float64), 0, 1.5)
    nfr = len(en)
    n = ns(nfr / fps + 0.6)
    ta = np.arange(n) / SR
    e = np.interp(ta, np.arange(nfr) / fps, en, right=0.0)
    e = L.lp(L.f32(e), 25, 1).astype(np.float64).clip(0, None)
    rust = L.svf(L.pink(n, r), 1200 + 5000 * np.clip(e, 0, 1), 0.8, "bp") * e ** 1.3
    crink = L.bp(L.poisson_clicks(r, n, 20 + 500 * e, np.clip(e, 0, 1)), 2000, 9000) * 0.4
    air = L.svf(L.brown(n, r), 250, 0.7, "lp") * e * 0.35
    x = rust + crink + air
    de = np.diff(en, prepend=en[0])
    pk, _ = find_peaks(de, height=0.04, distance=2)
    for i in pk:
        s = float(np.clip(de[i] * 3, 0, 1))
        m = ns(0.05)
        slap = L.bp(L.white(m, r), 400, 5000) * L.env_ad(m, 0.001, 0.025) * s
        a = ns(i / fps)
        x[a:a + m] += slap[: n - a]
    panA = np.interp(ta, np.arange(nfr) / fps, np.broadcast_to(pan, (nfr,)))
    return L.norm(L.pan2(L.f32(x), np.clip(panA, -1, 1)), 0.9) * gain   # linear: correlation preserved


def typewriter(r, dur, rate0=12.0, rate1=None, bell=True, bank=None):
    """typewriter: type-bar strike (metal) + platen thud + carriage ratchet, Poisson keystrokes."""
    rate1 = rate0 if rate1 is None else rate1
    bank = bank or [L.addm(L.impact(r, "tin", r.uniform(0.3, 0.45), 1.0, 0.08),
                           L.thud(r, r.uniform(140, 200), 0.06, 1.0, 3000) * 0.6) for _ in range(14)]
    evs, t = [], 0.0
    while t < dur:
        lam = rate0 + (rate1 - rate0) * t / max(dur, 1e-6)
        t += r.exponential(1.0 / lam)
        if t < dur:
            evs.append((t, bank[r.integers(len(bank))], r.uniform(0.5, 1.0), r.uniform(-0.25, 0.25)))
    n = ns(dur + 1.5)
    if bell:
        b = L.ring([2280, 2280 * 2.4, 2280 * 3.9], [0.9, 0.5, 0.3], [1, 0.4, 0.2], 1.2, r)
        evs.append((dur + 0.02, b, 0.8, 0.3))
    return L.norm(L.scatter(n, evs), 0.9)


def press_clatter(r, dur, rate0=1.2, rate1=6.0):
    """printing press: each cycle the platen closes (ka) and the frame slams (chunk); paper sheets whoosh out;
    cycle rate accelerates through the flood."""
    n = ns(dur + 0.8)
    evs, t = [], 0.0
    while t < dur:
        u = t / dur
        rate = rate0 * (rate1 / rate0) ** u
        ka = L.impact(r, "metal", 1.8, 1.0, 0.25)
        chunk = L.addm(L.thud(r, 85, 0.25, 1.0, 900), L.impact(r, "wood", 1.6, 1.0, 0.15) * 0.7)
        evs.append((t, ka, 0.6, r.uniform(-0.3, 0.3)))
        evs.append((t + 0.35 / rate, chunk, 1.0, r.uniform(-0.3, 0.3)))
        sheet = L.whoosh(r, 0.35, 0.15, 900, 5000, 1.0, -0.6, 0.6).x
        evs.append((t + 0.5 / rate, sheet, 0.35, 0.0))
        t += 1.0 / rate
    return L.norm(L.scatter(n, evs), 0.9)


def wet_paper_slap(r):
    """a soaked tract slapping face-down into mud: wet splat + thin paper tick + small bubbles."""
    n = ns(0.8)
    x = L.lp(L.white(n, r), 2200, 2) * L.env_ad(n, 0.002, 0.03) * 1.2
    x += L.bp(L.white(n, r), 600, 5000) * L.env_ad(n, 0.0005, 0.012) * 1.5      # the slap itself
    th = L.thud(r, 95, 0.3, 0.8, 800)
    x[: len(th)] += th * 0.8
    x[: ns(0.04)] += paper_tick(r).x[: ns(0.04)] * 0.5
    b = L.bubbles_cloud(r, 0.6, lambda u: 60 * np.exp(-u / 0.12), (1.0, 3.0), 1.0, (-0.3, 0.3)).mean(1)
    x[ns(0.01):ns(0.01) + len(b)] += b[: n - ns(0.01)] * 0.3
    return Snd(L.fade(L.norm(x, 0.9), 0.0003, 0.1), 0.0)


def harp_gliss(r, dur=1.4, lo=62, hi=86):
    """the comic 'flashback ripple' gliss: plucked strings up D major and back down (wavy)."""
    notes = [m for m in range(lo, hi + 1) if m % 12 in (2, 4, 7, 9, 11)]   # D/G pentatonic: safe over D & G major
    seq = notes + notes[::-1][1:]
    n = ns(dur + 1.5)
    evs = []
    for k, m in enumerate(seq):
        f = float(midi_hz(m))
        pl = L.ring([f * h for h in range(1, 6)], [1.2 / h ** 0.5 for h in range(1, 6)],
                    [1 / h ** 1.3 for h in range(1, 6)], 1.4, r)
        t = dur * k / len(seq)
        evs.append((t, pl, 0.6 + 0.4 * np.sin(np.pi * k / len(seq)), np.sin(2 * np.pi * k / len(seq)) * 0.6))
    return L.norm(L.scatter(n, evs), 0.9)


# ------------------------------------------------------------------------------------------------
# mud, bodies, creatures
# ------------------------------------------------------------------------------------------------
def mud_step(r, weight=1.0):
    """boot into wet mud: heel thud + suction squelch (falling band sweep) + a big viscous bubble."""
    n = ns(0.45)
    t = L.tt(n)
    x = L.thud(r, 60 * r.uniform(0.9, 1.1), 0.3, 1.0, 500)
    x = np.pad(x, (0, max(0, n - len(x))))[:n] * weight
    x += L.bp(L.white(n, r), 600, 4000) * L.env_ad(n, 0.0005, 0.012) * 0.8      # sole meets wet mud: the splat
    sq = L.svf(L.white(n, r), 300 + 1300 * np.exp(-np.maximum(t - 0.05, 0) / 0.05), 3, "bp")
    x += sq * L.env_pts(n, [(0, 0), (0.05, 0.2), (0.09, 0.9), (0.2, 0)]) * 0.5
    b = L.bubble(r, r.uniform(6, 11), r.uniform(0.3, 0.8), 1.0, 0.7)
    i = ns(0.12)
    x[i:i + len(b)] += b[: n - i] * 0.35
    return Snd(L.fade(L.norm(x, 0.9), 0.0005, 0.05), 0.0)


def snow_step(r, weight=1.0):
    g = L.gravel(r, 0.2, 3800, 0.55)
    x = L.addm(g * 0.9, L.thud(r, 70, 0.2, 0.6, 400) * 0.4 * weight)
    return Snd(L.fade(L.norm(x, 0.9), 0.001, 0.04), 0.0)


def leaves_step(r, weight=1.0):
    n = ns(0.25)
    exc = L.poisson_clicks(r, n, 2500 * L.env_pts(n, [(0, 0), (0.03, 1), (0.25, 0)]), 1.0)
    x = L.bp(exc, 1800, 9000) + L.resonators(exc, r.uniform(2500, 7000, 5), [0.004] * 5, [0.3] * 5)
    x = L.addm(x, L.thud(r, 75, 0.2, 0.6, 500) * 0.3 * weight)
    return Snd(L.fade(L.norm(x, 0.9), 0.001, 0.04), 0.0)


def step_for(r, desc, default="mud", weight=1.0):
    d = desc.lower()
    if "snow" in d:
        return snow_step(r, weight)
    if "leaf" in d or "leaves" in d:
        return leaves_step(r, weight)
    if "grass" in d:
        return L.boot_step(r, "grass", weight, False)
    if "stone" in d or "wood" in d or "floor" in d:
        return L.boot_step(r, "stone", weight, False)
    return {"mud": mud_step, "snow": snow_step, "leaves": leaves_step}[default](r, weight)


def mud_suck(r, dur=0.7):
    """a flag pole pulled out of deep mud: rising stick-slip suction, then the release pop."""
    n = ns(dur)
    rate = np.interp(L.tt(n), [0, dur], [18, 95])
    ss = L.stick_slip(r, dur, rate, 0.4, L.env_pts(n, [(0, 0), (0.1, 0.6), (dur * 0.9, 1), (dur, 0)]))
    x = L.resonators(ss, [140, 250, 410, 700], [0.04, 0.03, 0.02, 0.015], [1, 0.7, 0.5, 0.3])
    x += L.svf(L.brown(n, r), 300, 1.5, "bp") * L.env_pts(n, [(0, 0), (dur, 1)]) * 0.3
    pop = L.bubble(r, 14, 0.9, 1.0, 0.5)
    y = np.zeros(n + len(pop), np.float32)
    y[:n] = x
    y[:n] *= 0.4
    y[n - ns(0.08):n] *= np.linspace(1.0, 0.35, ns(0.08))       # the mud's grip holds... then lets go:
    y[n - ns(0.01):n - ns(0.01) + len(pop)] += pop * 1.2
    sl = L.bp(L.white(ns(0.03), r), 500, 4000) * L.env_ad(ns(0.03), 0.0005, 0.012)
    y[n:n + len(sl)] += sl * 2.5                                     # the release 'shlupp' (sync point)
    return Snd(L.fade(L.norm(y, 0.9), 0.02, 0.05), dur)


def wing_beat(r, size=1.5):
    """one downstroke of great wings: dark air push + feather flutter (40 Hz vane AM). sync = the push."""
    w = L.whoosh(r, 0.55, 0.22, 70, 700 / size ** 0.3, 0.9, 0.0, 0.0, dark=0.65, shape=2.0)
    n = len(w.x)
    fl = L.bp(L.white(n, r), 1200, 6000) * (0.5 + 0.5 * np.sin(L.TAU * 40 * L.tt(n))) * \
        L.env_pts(n, [(0, 0), (0.18, 1), (0.4, 0)]) * 0.25
    return Snd(L.norm(w.x + fl[:, None], 0.9), w.sync)


def heartbeat(r):
    x = L.addm(L.thud(r, 45, 0.25, 0.2, 150), np.zeros(ns(0.7), np.float32))
    d = L.thud(r, 55, 0.2, 0.2, 150) * 0.7
    x[ns(0.28):ns(0.28) + len(d)] += d
    return Snd(L.lp(x, 180, 2), 0.0)


def tinnitus(r, dur, f=6300.0):
    t = L.tt(ns(dur))
    x = np.sin(L.TAU * f * t) + 0.5 * np.sin(L.TAU * (f + 7) * t)
    return L.f32(x * L.env_pts(len(t), [(0, 0), (0.04, 1), (dur - 0.3, 1), (dur, 0)]))


def drop_cascade(r, dur=2.0, count=700):
    """the frozen drops fall away: a shimmering cascade of tiny water-glass ticks, their pitch sinking and the
    density thinning as they fall out of view."""
    n = ns(dur + 0.3)
    evs = []
    for _ in range(count):
        u = r.random() ** 1.6
        f = 9000 * (0.35 + 0.65 * (1 - u)) * r.uniform(0.8, 1.2)
        tk = L.ring([f, f * 2.3], [r.uniform(0.02, 0.07), 0.02], [1, 0.3], 0.1, r)
        evs.append((u * dur, tk, r.uniform(0.2, 1.0) * (1 - 0.6 * u), r.uniform(-1, 1)))
    return L.norm(L.scatter(n, evs), 0.9)


def crystal_resonance(r, dur=7.0, notes=(38, 45, 50, 54, 57, 62)):
    """the firmament revealed: a vast crystal dome ringing (glass modes, very long decay, slow beating)."""
    n = ns(dur)
    x = np.zeros(n, np.float32)
    for k, m in enumerate(notes):
        c = L.chime(r, float(midi_hz(m)), dur, t60=dur * 0.8, beat=0.35 + 0.1 * k, bright=0.55, strike=0.0)
        x[: len(c)] += c[:n] * (1 - 0.08 * k)
    att = L.env_pts(n, [(0, 0), (0.6, 1), (dur, 1)])
    return L.decorrelate(L.norm(x * att, 0.9), r, 0.7)


def halyard_creak(r, dur=0.7):
    """a flag hoisted: rope over a pulley — rubbery stick-slip squeak rising + rope hiss."""
    n = ns(dur)
    u = L.tt(n) / dur
    ss = L.stick_slip(r, dur, 250 + 450 * u, 0.2, np.sin(np.pi * u) ** 0.7)
    x = L.resonators(ss, [880, 1760, 2640], [0.01, 0.008, 0.006], [1, 0.5, 0.3]) * 0.6
    x += L.bp(L.white(n, r), 1500, 6000) * np.sin(np.pi * u) * 0.25
    return Snd(L.fade(L.norm(x, 0.8), 0.01, 0.05), 0.0)


def flies(r, dur, count=3):
    n = ns(dur)
    out = np.zeros((n, 2), np.float32)
    for k in range(count):
        f = r.uniform(185, 240) * (1 + 0.12 * L.smooth_noise(n, r, 1.5, 2))
        ph = np.cumsum(f) / SR
        saw = 2 * (ph % 1.0) - 1
        am = np.clip(0.5 + 0.6 * L.smooth_noise(n, r, 0.8, 2), 0, 1)
        x = L.bp(L.f32(saw), 200, 3500) * am
        out += L.pan2(x, 0.8 * L.smooth_noise(n, r, 0.4, 2))
    return L.norm(out, 0.9)


def crickets(r, dur, count=5):
    n = ns(dur)
    out = np.zeros((n, 2), np.float32)
    for k in range(count):
        f = r.uniform(4200, 4800)
        car = np.sin(L.TAU * f * L.tt(n))
        gate = np.zeros(n)
        t = r.uniform(0, 0.5)
        per = r.uniform(0.45, 0.8)
        while t < dur:
            for p in range(3):
                a = ns(t + p / 30.0)
                m = ns(0.018)
                gate[a:a + m] = np.hanning(m)[: max(0, min(m, n - a))]
            t += per * r.uniform(0.9, 1.1)
        out += L.pan2(L.f32(car * gate), r.uniform(-0.9, 0.9)) * r.uniform(0.3, 1.0)
    return L.norm(out, 0.9)


def radio_squelch(r, dur=0.22):
    """handheld radio key/unkey: click + band-limited squelch burst."""
    n = ns(dur)
    x = L.bp(L.white(n, r), 400, 3500) * L.env_ad(n, 0.002, dur * 0.6)
    x[: ns(0.004)] += L.impact(r, "tin", 0.3, 1.0, 0.004)[: ns(0.004)]
    return Snd(L.fade(L.norm(L.sat(x * 1.5, 1.5), 0.8), 0.0005, 0.02), 0.0)


def fire_from_energy(energy, fps, pan=0.0, gain=1.0, seed=0):
    """PHYSICS: fire intensity -> combustion roar (turbulent low-passed noise opening with intensity),
    flame-front 'breathing', resin/sap crackle pops (Poisson, rate ∝ intensity^1.5, random small modes),
    hiss of hot gas; decorrelated stereo."""
    r = R(seed)
    en = np.clip(np.asarray(energy, np.float64), 0, 1.5)
    nfr = len(en)
    n = ns(nfr / fps + 0.5)
    ta = np.arange(n) / SR
    e = np.interp(ta, np.arange(nfr) / fps, en, right=0.0)
    e = L.lp(L.f32(e), 12, 1).astype(np.float64).clip(0, None)
    out = np.zeros((n, 2), np.float32)
    breath = 0.75 + 0.25 * L.smooth_noise(n, r, 2.5, 3)
    for ch in range(2):
        roar = L.svf(L.brown(n, r), 110 + 520 * np.clip(e, 0, 1.2), 0.7, "lp") * e ** 1.2 * breath
        body = L.bp(L.pink(n, r), 300, 2500) * e ** 1.6 * 0.35 * breath
        exc = L.poisson_clicks(r, n, 2 + 140 * e ** 1.5, 1.0)
        crack = L.resonators(exc, r.uniform(1500, 6500, 6), [0.004] * 6, [0.5] * 6) + L.hp(exc, 3000) * 0.4
        hiss = L.hp(L.pink(n, r), 5000) * e ** 2 * 0.12
        out[:, ch] = roar + body + crack * 0.35 + hiss
    panA = np.interp(ta, np.arange(nfr) / fps, np.broadcast_to(pan, (nfr,)))
    return L.norm(L.pan2(out, np.clip(panA, -1, 1)), 0.9) * gain   # linear: correlation preserved


def steam_explosion(r, dur=3.0):
    """the deluge hits the burning effigy: a violent flash-boil — hiss burst, sizzle crackle, roaring steam."""
    n = ns(dur)
    t = L.tt(n)
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        hiss = L.hp(L.white(n, r), 1500) * L.env_ad(n, 0.004, 1.4)
        sizz = L.hp(L.poisson_clicks(r, n, 4000 * np.exp(-t / 0.8) + 50), 2000) * 0.6
        roar = L.svf(L.pink(n, r), 600 + 3000 * np.exp(-t / 0.4), 0.8, "lp") * L.env_ad(n, 0.01, 2.0)
        out[:, ch] = hiss * 0.6 + sizz + roar * 0.8
    return Snd(L.fade(L.norm(out, 0.95), 0.001, 0.3), 0.0)


def sulfur_bed(r, dur):
    """the Devil's throne room: sulfurous vent hiss, a low furnace rumble, lava gloops, distant fire."""
    n = ns(dur)
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        hs = L.hp(L.pink(n, r), 3000) * (0.5 + 0.5 * L.smooth_noise(n, r, 0.6, 2)) * 0.35
        rb = L.svf(L.brown(n, r), 65, 0.8, "lp") * 1.2
        out[:, ch] = hs + rb
    out += L.bubbles_cloud(r, dur, 2.2, (8.0, 18.0), 1.0, (-0.8, 0.8), None, 0.6) * 0.5
    out += fire_from_energy(np.full(int(dur * 24), 0.35), 24, 0.0, 0.5, seed=97)[:n]
    return L.norm(out, 0.9)


def candle_bed(r, dur, count=5):
    """candlesticks in the rain: tiny flame flutter (low turbulent burble) + wick sputters."""
    n = ns(dur)
    out = np.zeros((n, 2), np.float32)
    for k in range(count):
        fl = L.svf(L.brown(n, r), 90 + 40 * L.smooth_noise(n, r, 1.0, 2), 1.2, "bp") * \
             (0.6 + 0.4 * L.smooth_noise(n, r, 4.0, 2))
        sp = L.hp(L.poisson_clicks(r, n, 1.5), 1500) * 0.4
        out += L.pan2(L.f32(fl + sp), r.uniform(-0.8, 0.8))
    return L.norm(out, 0.9)


def boombox(r, dur):
    """a little battery boombox's speaker hiss & hum under the forecast, with the button clunks."""
    n = ns(dur)
    bed = L.tvify(L.pink(n, r) * 0.2) + L.hum(r, dur, 60.0, (0.1, 0.3, 0.2, 0.15), 0.1) * 0.05
    x = L.addm(bed * 0.5, np.zeros(n, np.float32))
    for tb in (0.0, dur - 0.08):
        b = L.addm(L.impact(r, "tin", 0.6, 1.0, 0.06), L.thud(r, 160, 0.06, 1.0, 2500) * 0.6)
        i = ns(tb)
        x[i:i + len(b)] += b[: n - i]
    return Snd(L.fade(x, 0.002, 0.02), 0.0)

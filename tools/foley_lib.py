"""THE UNBABELING — procedural Foley synthesis library (owner: Foley).

No samples. Every sound is computed at 48 kHz from first principles:
  * filtered noise (FFT-coloured, Butterworth, per-sample TPT state-variable filters in numba),
  * modal synthesis (damped sinusoids / two-pole resonator banks with material mode ratios),
  * physically-motivated excitation models:
      - Minnaert/van-den-Doel bubbles (drips, slop, splashes: pitch rises as the bubble surfaces),
      - N-wave shocks (explosions, whip cracks, thunder),
      - tortuous lightning channels (thunder = superposed N-waves from a 3-D random-walk channel,
        each arriving at distance/343 s, air-absorption low-passed by distance),
      - bouncing fragments with restitution (debris: impacts at t_k with v_k = e^k v_0),
      - stick-slip friction (creaks, tower groans, pen scratch, rubber squeak),
      - hook-row release trains (velcro), Strouhal-rate flapping (cloth).
Generators return Snd(x, sync): x mono (n,) or stereo (n, 2) float32; `sync` = seconds from the start of
x to the perceptual hit point (the instant that must land on the visual cue).
"""
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy import signal as sps
from scipy.interpolate import CubicSpline

SR = 48000
TAU = 2.0 * np.pi
C_SOUND = 343.0

# D major pitch pool (the score is in D): used for every tonal sparkle/chime so Foley never clashes.
D_MAJOR = [62, 64, 66, 67, 69, 71, 73]


def midi_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


@dataclass
class Snd:
    x: np.ndarray
    sync: float = 0.0

    @property
    def dur(self):
        return len(self.x) / SR


def ns(sec):
    return max(1, int(round(sec * SR)))


def tt(n):
    return np.arange(n, dtype=np.float64) / SR


def addm(*xs):
    """sum arrays of different lengths (zero-padded to the longest), all starting at sample 0."""
    n = max(len(x) for x in xs)
    shp = (n, 2) if any(x.ndim == 2 for x in xs) else (n,)
    out = np.zeros(shp, np.float32)
    for x in xs:
        if x.ndim == 1 and len(shp) == 2:
            out[: len(x)] += x[:, None]
        else:
            out[: len(x)] += x
    return out


def R(seed):
    return np.random.default_rng(seed)


def db(x):
    return 10.0 ** (np.asarray(x, float) / 20.0)


def f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


# ------------------------------------------------------------------------------------------------
# noise
# ------------------------------------------------------------------------------------------------
def white(n, r):
    return r.standard_normal(n).astype(np.float32)


def colored(n, r, alpha=1.0, fmin=12.0):
    """1/f^alpha power spectrum noise, unit std."""
    X = np.fft.rfft(r.standard_normal(n))
    f = np.maximum(np.fft.rfftfreq(n, 1.0 / SR), fmin)
    X *= f ** (-alpha / 2.0)
    X[0] = 0.0
    x = np.fft.irfft(X, n)
    return f32(x / (x.std() + 1e-12))


def pink(n, r):
    return colored(n, r, 1.0)


def brown(n, r):
    return colored(n, r, 2.0, fmin=18.0)


def smooth_noise(n, r, rate_hz, octaves=3):
    """band-limited fbm control signal in ~[-1, 1] (for gusts, wander): cubic-spline value noise."""
    out = np.zeros(n)
    amp, tot = 1.0, 0.0
    ta = np.arange(n, dtype=np.float64)
    for o in range(octaves):
        k = max(4, int(n / SR * rate_hz * 2 ** o) + 4)
        pts = r.standard_normal(k)
        xs = np.linspace(0, n - 1, k)
        out += amp * CubicSpline(xs, pts)(ta)
        tot += amp
        amp *= 0.5
    out = out / tot
    return out / (np.abs(out).max() + 1e-9)


# ------------------------------------------------------------------------------------------------
# filters
# ------------------------------------------------------------------------------------------------
def _wn(f):
    return float(np.clip(f / (SR / 2.0), 1e-5, 0.999))


def lp(x, f, order=2):
    return f32(sps.sosfilt(sps.butter(order, _wn(f), "low", output="sos"), x, axis=0))


def hp(x, f, order=2):
    return f32(sps.sosfilt(sps.butter(order, _wn(f), "high", output="sos"), x, axis=0))


def bp(x, f1, f2, order=2):
    return f32(sps.sosfilt(sps.butter(order, [_wn(f1), _wn(f2)], "band", output="sos"), x, axis=0))


def peq(x, f, gain_db, q=1.0):
    A = 10 ** (gain_db / 40.0)
    w = TAU * f / SR
    al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]
    a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return f32(sps.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0))


@njit(cache=True, fastmath=True)
def _svf(x, fc, q, mode, sr):
    n = x.shape[0]
    y = np.empty(n, np.float32)
    ic1 = 0.0
    ic2 = 0.0
    for i in range(n):
        f = fc[i]
        if f > 0.45 * sr:
            f = 0.45 * sr
        if f < 5.0:
            f = 5.0
        g = np.tan(np.pi * f / sr)
        k = 1.0 / q[i]
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            y[i] = v2
        elif mode == 1:
            y[i] = v1 * k
        else:
            y[i] = x[i] - k * v1 - v2
    return y


def svf(x, fc, q=0.707, mode="lp"):
    """time-varying TPT state-variable filter; fc / q scalars or per-sample arrays."""
    x = np.asarray(x, np.float64)
    fc = np.ascontiguousarray(np.broadcast_to(np.asarray(fc, np.float64), x.shape))
    q = np.ascontiguousarray(np.broadcast_to(np.asarray(q, np.float64), x.shape))
    return _svf(x, fc, q, {"lp": 0, "bp": 1, "hp": 2}[mode], float(SR))


def resonators(exc, freqs, t60s, gains):
    """parallel two-pole resonator bank (modal body excited by `exc`)."""
    y = np.zeros(len(exc), np.float64)
    for f, t60, g in zip(freqs, t60s, gains):
        if f >= 0.45 * SR or f <= 5:
            continue
        r = np.exp(-6.9078 / (t60 * SR))
        w = TAU * f / SR
        y += g * sps.lfilter([np.sin(w)], [1.0, -2 * r * np.cos(w), r * r], exc)
    return f32(y)


def ring(freqs, t60s, amps, dur, r=None, phase=None):
    """sum of exponentially decaying sinusoids (modal impulse response)."""
    n = ns(dur)
    t = tt(n)
    y = np.zeros(n)
    for i, (f, d, a) in enumerate(zip(freqs, t60s, amps)):
        if f >= 0.46 * SR or a == 0:
            continue
        ph = (r.uniform(0, TAU) if r is not None else 0.0) if phase is None else phase
        m = min(n, ns(d * 1.6) + 1)
        y[:m] += a * np.exp(-6.9078 * t[:m] / d) * np.sin(TAU * f * t[:m] + ph)
    return f32(y)


# ------------------------------------------------------------------------------------------------
# envelopes, dynamics, space
# ------------------------------------------------------------------------------------------------
def env_pts(n, pts):
    ts, vs = zip(*pts)
    return f32(np.interp(tt(n), ts, vs))


def env_ad(n, attack, t60, curve=1.0):
    t = tt(n)
    e = np.exp(-6.9078 * np.maximum(t - attack, 0) / t60)
    if attack > 0:
        e = np.where(t < attack, (t / attack) ** curve, e)
    return f32(e)


def fade(x, fin=0.003, fout=0.01):
    x = np.array(x, np.float32, copy=True)
    a, b = min(len(x), ns(fin)), min(len(x), ns(fout))
    w = np.ones(len(x), np.float32)
    if a > 1:
        w[:a] = np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    if b > 1:
        w[-b:] *= np.cos(np.linspace(0, np.pi / 2, b)) ** 2
    return x * (w[:, None] if x.ndim == 2 else w)


def sat(x, drive=1.0):
    return f32(np.tanh(x * drive) / np.tanh(drive))


def norm(x, peak=1.0):
    m = np.abs(x).max()
    return f32(x * (peak / m)) if m > 0 else f32(x)


def pan2(x, pan=0.0):
    """constant-power pan; mono -> stereo, stereo -> balance. pan scalar or per-sample array."""
    p = (np.clip(np.asarray(pan, np.float64), -1, 1) + 1) * np.pi / 4
    gl, gr = np.cos(p) * np.sqrt(2), np.sin(p) * np.sqrt(2)
    if x.ndim == 1:
        return f32(np.stack([x * np.cos(p), x * np.sin(p)], -1))
    return f32(np.stack([x[:, 0] * np.minimum(gl, 1.0), x[:, 1] * np.minimum(gr, 1.0)], -1))


def stereo(l, r_):
    return f32(np.stack([l, r_], -1))


def as_stereo(x):
    return x if x.ndim == 2 else f32(np.stack([x, x], -1))


def decorrelate(x, r, amount=0.6, ms=11.0):
    """mono -> wide stereo with complementary comb/allpass decorrelation (mono-compatible)."""
    d = ns(ms / 1000.0)
    y = np.zeros(len(x) + d, np.float32)
    y[d:] = x
    y = y[: len(x)]
    ap = lp(y, 5000)
    return f32(np.stack([x + amount * ap, x - amount * ap], -1) / (1 + amount * 0.5))


def haas_move(x, pan):
    return pan2(x, pan)


def air(x, dist):
    """air absorption: distance in metres -> gentle high cut."""
    fc = float(np.clip(18000.0 / (1.0 + dist / 150.0), 800.0, 20000.0))
    return lp(x, fc, 1) if fc < 19000 else x


def make_ir(rt60=2.0, predelay=0.012, band_mul=(1.3, 1.0, 0.75, 0.45), seed=0, early=6, er_span=0.06,
            echoes=(), flutter=None, density=1.0, width=1.0, dur=None):
    """procedural stereo room/hall/outdoor impulse response.
    band_mul: RT multipliers for bands <250 | 250-1k | 1k-4k | >4k (air/wall absorption).
    echoes: [(delay_s, gain, lp_hz)] discrete reflections (monoliths, canyon walls).
    flutter: (period_s, decay_per_bounce) tunnel flutter echo."""
    r = R(seed)
    dur = dur or min(12.0, rt60 * 1.25 + predelay + 0.1)
    n = ns(dur)
    t = tt(n)
    out = np.zeros((n, 2), np.float32)
    edges = [(20, 250), (250, 1000), (1000, 4000), (4000, 20000)]
    for ch in range(2):
        acc = np.zeros(n)
        for (lo, hi), m in zip(edges, band_mul):
            nz = r.standard_normal(n)
            if density < 1.0:
                nz *= (r.random(n) < density) / np.sqrt(density)
            b = bp(nz, lo, hi, 2)
            acc += b * np.exp(-6.9078 * t / max(0.05, rt60 * m))
        mix = 1.0 if ch == 0 else width
        acc = acc * mix + (1 - mix) * out[:, 0] if ch == 1 and width < 1 else acc
        out[:, ch] = acc
    # onset shaping (diffuse build-up) and predelay
    build = np.clip(t / max(0.005, er_span * 0.7), 0, 1) ** 1.5
    out *= build[:, None]
    for k in range(early):
        d = r.uniform(0.004, er_span)
        g = 0.9 * np.exp(-d / er_span) * r.uniform(0.5, 1.0)
        i = ns(d)
        if i < n:
            out[i, 0] += g * r.choice([-1, 1]) * 3
            out[ns(d * r.uniform(0.9, 1.1)) % n, 1] += g * r.choice([-1, 1]) * 3
    for d, g, f in echoes:
        i = ns(d)
        if i < n:
            imp = np.zeros(n)
            imp[i] = g * 6
            e = lp(imp, f, 2)
            out[:, 0] += e
            out[:, 1] += np.roll(e, ns(r.uniform(0.001, 0.008)))
    if flutter is not None:
        per, dec = flutter
        k = 1
        while k * per < dur and dec ** k > 0.01:
            i = ns(k * per)
            out[i, k % 2] += 2.0 * dec ** k
            k += 1
    out = np.concatenate([np.zeros((ns(predelay), 2), np.float32), out])
    out /= np.sqrt((out ** 2).sum(0, keepdims=True)) + 1e-9
    return f32(out)


def convolve(x, ir):
    x = as_stereo(x)
    return f32(np.stack([sps.oaconvolve(x[:, c], ir[:, c]) for c in range(2)], -1))


def tvify(x, r=None, small=True):
    """through a CRT set's little speaker: band-limited, resonant, a hint of distortion + cabinet buzz."""
    y = hp(x, 190, 2)
    y = lp(y, 6200 if small else 9000, 2)
    y = peq(y, 1150, 5.0, 1.4)
    y = peq(y, 420, 3.0, 2.0)
    return sat(y * 1.6, 1.4) * 0.7


def crush(x, bits=6, ds=6):
    y = np.repeat(x[::ds], ds, axis=0)[: len(x)]
    q = 2 ** (bits - 1)
    return f32(np.round(y * q) / q)


# ------------------------------------------------------------------------------------------------
# elementary physical models
# ------------------------------------------------------------------------------------------------
def nwave(T):
    """N-wave pressure signature of a shock (duration T seconds)."""
    n = max(3, ns(T))
    x = np.linspace(1, -1, n)
    return f32(x)


def bubble(r, radius_mm=None, xi=None, amp=1.0, damp_mul=0.25):
    """van den Doel bubble: Minnaert f0 = 3.26/r (r in m) rising as f(t)=f0(1+sigma t)."""
    rad = (radius_mm if radius_mm is not None else r.uniform(0.8, 4.0)) / 1000.0
    f0 = 3.26 / rad
    d = (0.13 * f0 + 0.0072 * f0 ** 1.5) * damp_mul
    xi = xi if xi is not None else r.uniform(0.08, 0.4)
    sigma = xi * d
    n = ns(min(1.0, 7.0 / d))
    t = tt(n)
    f = f0 * (1 + sigma * t)
    ph = TAU * np.cumsum(f) / SR
    x = amp * np.sin(ph) * np.exp(-d * t)
    a = min(n, 24)
    x[:a] *= np.linspace(0, 1, a)
    return f32(x)


def scatter(n, events, st=True):
    """events: [(t_s, sig(mono|stereo), gain, pan)] -> stereo buffer."""
    out = np.zeros((n, 2), np.float32)
    for t0, s, g, p in events:
        s = pan2(s, p) if s.ndim == 1 else pan2(s, p)
        i = int(round(t0 * SR))
        if i >= n or i + len(s) <= 0:
            continue
        a, b = max(0, i), min(n, i + len(s))
        out[a:b] += g * s[a - i:b - i]
    return out


MATERIALS = {
    #          mode ratios                           t60 range (s)    base freq range (Hz)   noise click
    "stone": ([1.0, 1.58, 2.31, 3.12, 4.37, 5.9], (0.012, 0.05), (900, 4200), 0.6),
    "marble": ([1.0, 1.71, 2.62, 3.53, 4.61], (0.025, 0.09), (1100, 5200), 0.45),
    "glass": ([1.0, 2.32, 4.25, 6.63, 9.38], (0.25, 0.9), (1800, 6500), 0.15),
    "crystal": ([1.0, 2.32, 4.25, 6.63], (0.8, 2.5), (1100, 4700), 0.05),
    "wood": ([1.0, 2.57, 4.21, 6.1], (0.025, 0.07), (260, 1300), 0.5),
    "metal": ([1.0, 2.76, 5.40, 8.93, 13.3], (0.3, 0.9), (1400, 5200), 0.2),
    "tin": ([1.0, 1.47, 2.09, 2.56, 3.51], (0.05, 0.14), (700, 2600), 0.35),
    "screen": ([1.0, 1.9, 3.3, 5.1], (0.03, 0.08), (600, 2200), 0.5),
}


def impact(r, material="stone", size=1.0, vel=1.0, dur=None, f_base=None):
    """one modal impact (fragment hitting something). size>1 = bigger & lower."""
    ratios, (d0, d1), (f0, f1), click = MATERIALS[material]
    fb = f_base or np.exp(r.uniform(np.log(f0), np.log(f1))) / size
    fr = [fb * q * r.uniform(0.97, 1.03) for q in ratios]
    t60 = [r.uniform(d0, d1) * size ** 0.5 / (1 + 0.35 * k) for k in range(len(ratios))]
    am = [vel * r.uniform(0.4, 1.0) / (1 + 0.6 * k) for k in range(len(ratios))]
    dur = dur or max(t60) * 1.4 + 0.01
    x = ring(fr, t60, am, dur, r)
    nc = ns(0.004)
    x[:nc] += click * vel * r.standard_normal(nc) * np.linspace(1, 0, nc) ** 2
    return fade(x, 0.0005, 0.005)


def debris(r, dur, count, material="stone", t_spread=0.8, height=(0.3, 3.0), rest=(0.25, 0.5), size=(0.4, 1.6),
           pan=(-0.8, 0.8), gravity=9.81, start=0.0):
    """bouncing fragments: each fragment falls from height h, impacts at t_k with v_k=e^k v0."""
    n = ns(dur)
    evs = []
    for _ in range(count):
        t0 = start + r.exponential(t_spread / 3) if t_spread > 0 else start
        h = r.uniform(*height)
        v = np.sqrt(2 * gravity * h)
        t = t0 + v / gravity
        e = r.uniform(*rest)
        s = np.exp(r.uniform(np.log(size[0]), np.log(size[1])))
        p = r.uniform(*pan)
        k = 0
        while v > 0.25 and t < dur and k < 8:
            g = (v / 7.0) ** 1.3 * s ** 0.8
            evs.append((t, impact(r, material, s, 1.0), g, p))
            v *= e
            t += 2 * v / gravity
            k += 1
    return norm(scatter(n, evs), 0.95)


def stick_slip(r, dur, rate, jitter=0.15, amp=None):
    """impulse train at a (time-varying) stick-slip rate in Hz (array or scalar)."""
    n = ns(dur)
    rate = np.broadcast_to(np.asarray(rate, float), (n,))
    amp = np.ones(n) if amp is None else np.broadcast_to(np.asarray(amp, float), (n,))
    ph = np.cumsum(rate * (1 + jitter * r.standard_normal(n) * 0.3)) / SR
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    x = np.zeros(n, np.float32)
    x[idx] = amp[idx] * r.uniform(0.5, 1.0, len(idx))
    return x


def poisson_clicks(r, n, rate, amp=None):
    """Poisson impulse process with per-sample rate (Hz)."""
    rate = np.broadcast_to(np.asarray(rate, float), (n,))
    hits = r.random(n) < rate / SR
    x = np.zeros(n, np.float32)
    a = np.ones(n) if amp is None else np.broadcast_to(np.asarray(amp, float), (n,))
    x[hits] = (a[hits] * r.rayleigh(0.6, hits.sum())).astype(np.float32)
    return x


# ------------------------------------------------------------------------------------------------
# wind, water, weather
# ------------------------------------------------------------------------------------------------
def wind(r, dur, level=0.5, gust=0.5, gust_rate=0.18, whistle=0.2, bright=0.5, speed=None, seed_wander=None):
    """Stereo wind: turbulent body (brown+pink through a moving low-pass that tracks wind speed),
    edge-tone whistles (narrow band-passes, pitch ∝ speed, Strouhal f = 0.2 U / d), independent-but-correlated L/R."""
    n = ns(dur)
    if speed is None:
        g = smooth_noise(n, r, gust_rate, 3)
        speed = np.clip(level * (1 + gust * g), 0.02, 1.5)
    speed = np.broadcast_to(np.asarray(speed, float), (n,))
    out = np.zeros((n, 2), np.float32)
    common = 0.55 * brown(n, r) + 0.45 * pink(n, r)
    for ch in range(2):
        own = 0.55 * brown(n, r) + 0.45 * pink(n, r)
        src = 0.6 * common + 0.8 * own
        sp = speed * (1 + 0.08 * smooth_noise(n, r, 0.5, 2))
        fc = 120 + (500 + 2600 * bright) * sp ** 1.3
        body = svf(src, fc, 0.6, "lp") * sp ** 1.6
        hs = hp(pink(n, r), 2500) * 0.08 * bright * sp ** 2.2
        wh = np.zeros(n, np.float32)
        if whistle > 0:
            for k in range(3):
                wander = smooth_noise(n, r, 0.25 + 0.1 * k, 2)
                fw = (300 + 220 * k) * (0.6 + sp) * (1 + 0.12 * wander)
                wh += svf(white(n, r), fw, 14 + 6 * k, "bp") * (0.5 / (k + 1)) * np.clip(sp - 0.2, 0, None) ** 1.5
        out[:, ch] = body + hs + whistle * wh * 0.9
    return out * 0.5


def ocean_roar(r, dur, level=1.0, swell_rate=0.09):
    n = ns(dur)
    sw = 0.5 + 0.5 * smooth_noise(n, r, swell_rate, 3)
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        b = svf(brown(n, r), 180 + 900 * sw, 0.6, "lp")
        foam = hp(pink(n, r), 1800) * 0.25 * sw ** 2
        fizz = hp(poisson_clicks(r, n, 900 * sw ** 2), 3000) * 0.3
        out[:, ch] = (b * (0.4 + 0.6 * sw) + foam + fizz) * level
    return norm(out, 0.9)


def bubbles_cloud(r, dur, rate, radius=(1.0, 6.0), amp=1.0, pan=(-0.7, 0.7), xi=None, damp_mul=0.25):
    """granular cloud of Minnaert bubbles; rate scalar or callable(t)->Hz."""
    n = ns(dur)
    evs = []
    t = 0.0
    while t < dur:
        rt = rate(t) if callable(rate) else rate
        if rt <= 0:
            t += 0.02
            continue
        t += r.exponential(1.0 / rt)
        rad = np.exp(r.uniform(np.log(radius[0]), np.log(radius[1])))
        evs.append((t, bubble(r, rad, xi, 1.0, damp_mul), amp * r.uniform(0.3, 1.0) * (rad / radius[1]) ** 0.3,
                    r.uniform(*pan)))
    return norm(scatter(n, evs), 0.9)


def drip(r, size=1.0, wet=1.0):
    """a single water drop into a pool: tiny impact tick + rising Minnaert 'plink'."""
    b = bubble(r, r.uniform(1.2, 3.2) * size, r.uniform(0.25, 0.7), 1.0, 0.18)
    tick = hp(white(ns(0.003), r), 3000) * np.linspace(1, 0, ns(0.003)) * 0.25
    x = np.zeros(len(b) + ns(0.006), np.float32)
    x[: len(tick)] += tick
    x[ns(0.004):ns(0.004) + len(b)] += b * wet
    return Snd(fade(x, 0.0003, 0.01), 0.0)


def rain(r, dur, density=800.0, level=1.0):
    n = ns(dur)
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        clicks = poisson_clicks(r, n, density)
        out[:, ch] = bp(clicks, 1500, 9000) * 0.5 + hp(pink(n, r), 3000) * 0.06
    return out * level


def thunder(r, closeness=0.5, dur=None):
    """physically-motivated thunder: 3-D tortuous lightning channel (random walk, ~10 m segments,
    ~16° mean direction change); every segment radiates an N-wave that arrives at d/343 s with 1/d
    spreading and distance-dependent air absorption; dipole-ish radiation strongest broadside.
    sync = first arrival (the crack), time-compressed for cinema."""
    dist = 250 + (1 - closeness) * 4200
    top = 2500 + r.uniform(0, 1500)
    seg = 9.0
    p = np.array([dist * r.uniform(0.7, 1.1), r.uniform(-400, 400), top])
    pts = [p.copy()]
    d = np.array([0.0, 0.0, -1.0])
    while p[2] > 0 and len(pts) < 900:
        d = d + r.normal(0, 0.28, 3)
        d[2] = min(d[2], -0.25)
        d /= np.linalg.norm(d)
        p = p + d * seg
        pts.append(p.copy())
    pts = np.array(pts)
    mid = 0.5 * (pts[1:] + pts[:-1])
    seg_v = pts[1:] - pts[:-1]
    dd = np.linalg.norm(mid, axis=1)
    los = mid / dd[:, None]
    broad = np.sqrt(1 - np.clip(np.abs((seg_v / seg * los).sum(1)), 0, 1) ** 2) + 0.15
    tarr = dd / C_SOUND
    t0 = tarr.min()
    # cinematic compression of the rolling (keep shape, shorten far strikes)
    comp = 0.55 + 0.45 * closeness
    rel = (tarr - t0) * comp
    total = rel.max() + 3.5
    n = ns(dur or total)
    bins = [(0, 600, 16000), (600, 1500, 5000), (1500, 3000, 2200), (3000, 1e9, 900)]
    out = np.zeros(n, np.float64)
    for lo_d, hi_d, fc in bins:
        m = (dd >= lo_d) & (dd < hi_d)
        if not m.any():
            continue
        buf = np.zeros(n)
        for tr, a, dist_i in zip(rel[m], broad[m], dd[m]):
            T = r.uniform(0.004, 0.012) * (1 + dist_i / 2000)
            nw = nwave(T)
            i = ns(tr)
            if i + len(nw) < n:
                buf[i:i + len(nw)] += a * 180.0 / dist_i * nw
        out += lp(buf, fc, 2)
    rumble = svf(brown(n, r), 90 + 60 * closeness, 0.7, "lp") * env_ad(n, 0.25, 3.5 + 2 * (1 - closeness)) * 0.2
    x = out / (np.abs(out).max() + 1e-9) + rumble * (0.5 + 0.5 * (1 - closeness))
    if closeness > 0.6:  # the tearing electrical crackle of a very near strike
        cr = hp(poisson_clicks(r, ns(0.25), 6000), 2500) * env_ad(ns(0.25), 0.002, 0.2)
        x[: len(cr)] += cr * 1.5 * (closeness - 0.6) / 0.4
    x = sat(x * 1.2, 1.3)
    return Snd(fade(norm(x, 0.95), 0.001, 0.5), 0.0)


# ------------------------------------------------------------------------------------------------
# impacts & destruction
# ------------------------------------------------------------------------------------------------
def sub_boom(r, f0=52.0, f1=26.0, dur=3.0, drop=0.35, level=1.0):
    """sub-bass boom (< 60 Hz): pitch-dropping sine with pressure thump."""
    n = ns(dur)
    t = tt(n)
    f = f1 + (f0 - f1) * np.exp(-t / drop)
    ph = TAU * np.cumsum(f) / SR
    x = np.sin(ph) * env_ad(n, 0.006, dur * 0.85)
    th = lp(white(ns(0.12), r), 110, 4) * env_ad(ns(0.12), 0.001, 0.08)
    x[: len(th)] += th * 2.5
    x = sat(x * 1.3, 1.2)
    return Snd(norm(lp(x, 90, 4), 0.95) * level, 0.0)


def shock(r, size=1.0):
    """initial shock crack (N-wave, high-passed so it is felt as a snap, not DC)."""
    T = 0.004 + 0.006 * size
    x = np.zeros(ns(T + 0.06), np.float32)
    nw = nwave(T)
    x[: len(nw)] = nw
    x = hp(x, 300, 2) + hp(white(len(x), r), 1500) * env_ad(len(x), 0.0003, 0.02) * 0.5
    return norm(x, 1.0)


def explosion(r, size=1.0, dur=5.0, material="stone", debris_n=60):
    """breach charge: shock crack + fireball roar (brown noise, closing low-pass) + sub + bouncing debris."""
    n = ns(dur)
    t = tt(n)
    out = np.zeros((n, 2), np.float32)
    sk = shock(r, size)
    for ch in range(2):
        body = brown(n, r)
        fc = 250 + 5200 * np.exp(-t / 0.18) + 600 * np.exp(-t / 1.2)
        body = svf(body, fc, 0.7, "lp") * env_ad(n, 0.004, 1.6 * size)
        roar = svf(pink(n, r), 400 + 1500 * np.exp(-t / 0.5), 0.9, "lp") * env_ad(n, 0.02, 2.8 * size) * 0.5
        out[:, ch] = sat(body * 1.4 + roar, 1.6) * 0.8
        out[: len(sk), ch] += sk * 1.3
    sb = sub_boom(r, 55, 24, min(dur, 4.0), 0.4).x
    out[: len(sb)] += sb[:, None] * 0.9
    if debris_n:
        out += debris(r, dur, debris_n, material, t_spread=1.2, height=(0.5, 4.0), start=0.1) * 0.55
    return Snd(fade(norm(out, 0.95), 0.0005, 0.4), 0.0)


def stone_crack(r, dur=0.8, intensity=1.0, material="stone", bright=1.0, final_bang=True, accel=True):
    """propagating fracture: accelerating train of micro-cracks through a stone body + a final split."""
    n = ns(dur)
    t = tt(n)
    rate = 40 + 900 * (t / dur) ** 2 if accel else np.full(n, 300.0)
    exc = poisson_clicks(r, n, rate, amp=np.clip(0.3 + t / dur, 0, 1))
    exc = hp(exc, 400)
    ratios, (d0, d1), (f0, f1), _ = MATERIALS[material]
    fr = np.exp(r.uniform(np.log(f0 * 0.5), np.log(f1 * bright), 10))
    x = resonators(exc, fr, r.uniform(d0, d1, 10), r.uniform(0.3, 1.0, 10))
    x += hp(exc, 3000) * 0.6
    grind = svf(brown(n, r), 180, 3, "bp") * env_pts(n, [(0, 0), (dur * 0.6, 0.6), (dur, 1)]) * 0.5
    x = x * 0.6 + grind * 0.4
    if final_bang:
        bang = impact(r, material, 2.2, 1.0) * 2.0
        sk = shock(r, 0.5)
        tail = np.zeros(ns(0.6), np.float32)
        tail[: len(bang)] += bang[: len(tail)]
        tail[: len(sk)] += sk
        x = np.concatenate([x, tail])
    x = norm(x, 0.9) * intensity
    return Snd(fade(x, 0.001, 0.05), dur if final_bang else dur * 0.9)


def shatter(r, material="marble", size=1.0, frags=180, dur=4.0, gravity=True):
    """big brittle shatter: crack + shock + fragment cloud (bouncing with gravity, or drifting in a void)."""
    n = ns(dur)
    out = np.zeros((n, 2), np.float32)
    sk = shock(r, 0.8 * size)
    out[: len(sk)] += sk[:, None] * 1.2
    big = impact(r, material, 3.0 * size, 1.0)
    out[: len(big)] += big[:, None] * 1.4
    if gravity:
        out += debris(r, dur, frags, material, t_spread=0.4, height=(0.1, 2.5), size=(0.3, 1.5)) * 0.6
    else:  # void: fragments crack apart from each other and keep ringing as they drift
        evs = []
        for _ in range(frags):
            tt0 = r.exponential(0.25)
            evs.append((tt0, impact(r, material, np.exp(r.uniform(-1.2, 0.5)), 1.0), r.uniform(0.1, 0.6) * np.exp(-tt0),
                        r.uniform(-0.95, 0.95)))
        out += scatter(n, evs) * 0.8
    lo = svf(brown(n, r), 140, 0.7, "lp") * env_ad(n, 0.003, 1.2 * size)
    out += lo[:, None] * 0.8
    return Snd(fade(norm(out, 0.95), 0.0005, 0.3), 0.0)


def creak(r, dur, rate0=18, rate1=45, body=(70, 110, 160, 230, 340), t60=0.25, bright=1.0):
    """stick-slip groan of a huge structure (tower groan / wood creak)."""
    n = ns(dur)
    rate = np.interp(tt(n), [0, dur * 0.5, dur], [rate0, (rate0 + rate1) * 0.6, rate1])
    rate *= 1 + 0.25 * smooth_noise(n, r, 2.0, 2)
    amp = env_pts(n, [(0, 0), (dur * 0.2, 1), (dur * 0.8, 0.8), (dur, 0)])
    exc = stick_slip(r, dur, rate, 0.3, amp)
    fr = np.array(body) * r.uniform(0.9, 1.1, len(body))
    x = resonators(exc, fr, [t60] * len(fr), [1.0 / (1 + 0.3 * k) for k in range(len(fr))])
    x += resonators(exc, fr * 7.3 * bright, [0.03] * len(fr), [0.2] * len(fr))
    return Snd(fade(norm(x, 0.9), 0.02, 0.2), 0.0)


def thud(r, f=70.0, dur=0.35, noise=0.6, lpf=900):
    """dull heavy thud (letters landing, footfalls, body)."""
    n = ns(dur)
    t = tt(n)
    fr = f * (1 + 0.6 * np.exp(-t / 0.015))
    x = np.sin(TAU * np.cumsum(fr) / SR) * env_ad(n, 0.001, dur * 0.6)
    x += lp(white(n, r), lpf, 2) * env_ad(n, 0.0005, 0.05) * noise
    return f32(x)


def gravel(r, dur=0.12, density=1800, bright=1.0):
    """granular crunch: many tiny grain impacts."""
    n = ns(dur)
    amp = env_pts(n, [(0, 0), (dur * 0.15, 1), (dur, 0)])
    exc = poisson_clicks(r, n, density * amp, amp)
    x = bp(exc, 900 * bright, 7000 * bright) + resonators(exc, r.uniform(1500, 5000, 6) * bright,
                                                         [0.008] * 6, [0.4] * 6)
    return f32(x)


def dust_puff(r, dur=0.6, bright=0.5):
    n = ns(dur)
    x = svf(pink(n, r), 300 + 1500 * bright * np.exp(-tt(n) / 0.2), 0.6, "lp")
    return norm(x * env_pts(n, [(0, 0), (0.015, 1), (dur, 0)]) ** 2, 0.9)


# ------------------------------------------------------------------------------------------------
# air & motion
# ------------------------------------------------------------------------------------------------
def whoosh(r, dur=1.2, peak=0.6, f_lo=250, f_hi=2800, q=1.1, pan0=-0.5, pan1=0.5, dark=0.0, shape=2.5):
    """air-displacement pass-by: band-pass swept up to f_hi at the pass then down (Doppler-like),
    bell amplitude, pan sweeps across the pass. sync = the pass (peak)."""
    n = ns(dur)
    t = tt(n)
    u = (t - peak) / np.where(t < peak, peak, max(1e-3, dur - peak))
    bell = np.exp(-shape * u ** 2)
    fc = f_lo + (f_hi - f_lo) * bell * np.where(t < peak, 1.0, 0.85)
    src = (1 - dark) * pink(n, r) + dark * brown(n, r)
    a = svf(src, fc, q, "bp")
    b = svf(src, fc * 0.5, 0.7, "lp") * 0.5
    x = (a + b) * bell ** 1.5
    pan = np.clip(pan0 + (pan1 - pan0) * (1 / (1 + np.exp(-(t - peak) * 8 / max(dur, 0.1)))), -1, 1)
    return Snd(fade(pan2(norm(x, 0.9), pan), 0.005, 0.05), peak)


def swell(r, dur=2.0, f_lo=200, f_hi=6000, curve=2.0, q=0.8):
    """rising noise swell (reverse-cymbal-like air, non-tonal) peaking at the end (sync = end)."""
    n = ns(dur)
    u = tt(n) / dur
    fc = f_lo * (f_hi / f_lo) ** (u ** 1.3)
    x = svf(pink(n, r), fc, q, "lp") * u ** curve
    return Snd(fade(decorrelate(norm(x, 0.9), r), 0.05, 0.004), dur)


def shepard_rise(r, dur, octaves=6, base=80, rate=0.35, width=0.35):
    """endless rising spectral-Shepard noise (for the infinite recursive zoom): octave-spaced noise bands
    each gliding up one octave per 1/rate s, bell-weighted in log-frequency so the whole never arrives."""
    n = ns(dur)
    t = tt(n)
    out = np.zeros(n, np.float32)
    src = pink(n, r)
    for k in range(octaves):
        pos = (k + t * rate) % octaves
        fc = base * 2 ** pos
        w = np.exp(-((pos - octaves / 2) / (octaves * width)) ** 2)
        out += svf(src if k % 2 else -src, fc, 6.0, "bp") * w
    return f32(out / (np.abs(out).max() + 1e-9))


# ------------------------------------------------------------------------------------------------
# tonal/crystalline
# ------------------------------------------------------------------------------------------------
def chime(r, f0, dur=4.0, ratios=(1.0, 2.32, 4.25, 6.63, 9.38), t60=3.0, beat=1.2, bright=1.0, strike=0.3):
    """glass/crystal bowl: inharmonic modes, each a detuned pair (slow beating shimmer)."""
    fr, dc, am = [], [], []
    for k, q in enumerate(ratios):
        for s in (-1, 1):
            fr.append(f0 * q * (1 + s * beat / (2 * f0 * q)))
            dc.append(t60 / (1 + 0.55 * k))
            am.append(bright ** k / (1 + 0.9 * k) * 0.5)
    x = ring(fr, dc, am, dur, r)
    if strike:
        n = ns(0.01)
        x[:n] += strike * hp(white(n, r), 4000) * np.linspace(1, 0, n)
    return fade(x, 0.0008, min(0.5, dur * 0.2))


def sparkle(r, dur=1.5, count=18, pitches=D_MAJOR, octave=(2, 3), pan=(-0.8, 0.8), decay=0.6, start_spread=None):
    """tinkling glints: tiny glass pings on D-major pitches."""
    n = ns(dur)
    evs = []
    spread = start_spread or dur * 0.7
    for _ in range(count):
        m = r.choice(pitches) + 12 * r.integers(octave[0], octave[1] + 1)
        f = float(midi_hz(m))
        x = chime(r, f, decay * 1.5, ratios=(1.0, 2.76, 5.4), t60=decay * r.uniform(0.5, 1.2), beat=3, strike=0.05)
        evs.append((r.uniform(0, spread) ** 1.3 / spread ** 0.3, x, r.uniform(0.2, 1.0), r.uniform(*pan)))
    return Snd(norm(scatter(n, evs), 0.9), 0.0)


def ping(r, f, dur=0.25, kind=0):
    """a notification blip: FM-ish sine with a fast decay (kind picks the timbre)."""
    n = ns(dur)
    t = tt(n)
    if kind == 0:
        x = np.sin(TAU * f * t + 0.8 * np.sin(TAU * f * 2 * t) * np.exp(-t / 0.02))
    elif kind == 1:
        x = np.sin(TAU * f * t) + 0.3 * np.sin(TAU * 2 * f * t)
    else:
        fr = f * (1 + 0.35 * np.exp(-t / 0.03))
        x = np.sin(TAU * np.cumsum(fr) / SR)
    return fade(f32(x * env_ad(n, 0.002, dur * 0.7)), 0.001, 0.02)


def fsk_chirps(r, dur, rate=6.0, lo=1200, hi=2200, level=1.0):
    """data chatter: FSK bit bursts + tiny up/down chirps."""
    n = ns(dur)
    x = np.zeros(n, np.float32)
    t = 0.0
    while t < dur:
        t += r.exponential(1 / rate)
        L = r.uniform(0.04, 0.22)
        m = ns(L)
        baud = r.choice([150, 300, 600])
        bits = r.integers(0, 2, int(L * baud) + 2)
        f = np.where(np.repeat(bits, SR // baud + 1)[:m] > 0, hi, lo) * r.uniform(0.9, 1.6)
        if r.random() < 0.35:
            f = np.linspace(r.uniform(800, 3000), r.uniform(800, 5000), m)
        s = np.sin(TAU * np.cumsum(f) / SR) * env_pts(m, [(0, 0), (0.004, 1), (L - 0.006, 1), (L, 0)])
        i = ns(t)
        if i + m < n:
            x[i:i + m] += s * r.uniform(0.3, 1.0)
    return norm(x, 0.9) * level


# ------------------------------------------------------------------------------------------------
# electrical
# ------------------------------------------------------------------------------------------------
def hum(r, dur, f=60.0, harm=(1.0, 0.5, 0.35, 0.2, 0.12, 0.08), buzz=0.1):
    n = ns(dur)
    t = tt(n)
    x = sum(a * np.sin(TAU * f * (k + 1) * t + r.uniform(0, TAU)) for k, a in enumerate(harm))
    if buzz:
        saw = 2 * ((f * 2 * t) % 1.0) - 1
        x += buzz * hp(f32(saw), 800)
    return f32(x)


def relay_clunk(r, size=1.0):
    """big electrical contactor closing: metal slap + body + sparks."""
    x = addm(impact(r, "tin", 1.2 * size, 1.0) * 0.8, thud(r, 95 / size, 0.25, 0.8, 1500) * 0.8)
    sp = hp(poisson_clicks(r, ns(0.05), 3000), 2000) * env_ad(ns(0.05), 0.0, 0.04) * 0.4
    x[: len(sp)] += sp
    return f32(x)


def crt_on(r, whine_hold=0.0):
    """CRT power-on: relay tick, degauss coil 'thunk-bwommm' (60 Hz field + mechanical 120 Hz rattle
    decaying ~1 s), HV static crackle as the tube charges, then the 15.734 kHz flyback whine fading in.
    sync = relay tick (t=0)."""
    dur = 1.8 + whine_hold
    n = ns(dur)
    t = tt(n)
    x = np.zeros(n, np.float64)
    tick = impact(r, "tin", 0.5, 1.0)
    x[: len(tick)] += tick * 0.9
    clk = hp(white(ns(0.004), r), 1500) * np.linspace(1, 0, ns(0.004)) ** 2
    x[: len(clk)] += clk * 1.2
    dg_env = np.exp(-t / 0.35) * (t > 0.01)
    dg = np.sin(TAU * 60 * t) + 0.6 * np.sin(TAU * 120 * t + 1) + 0.3 * np.sign(np.sin(TAU * 60 * t)) * 0.4
    x += sat(f32(dg * dg_env * 1.4), 2.0) * 0.8
    x += resonators(f32(white(n, r) * dg_env * 0.02), [120, 240, 360, 780], [0.2] * 4, [1, 0.5, 0.3, 0.2]) * 0.9
    crack = hp(poisson_clicks(r, n, 60 * np.exp(-t / 0.6) + 5), 1500) * np.exp(-t / 0.9)
    x += crack * 0.5
    wh = np.sin(TAU * 15734.0 * t) * np.clip((t - 0.15) / 0.4, 0, 1)
    x += wh * 0.05
    return Snd(fade(norm(f32(x), 0.95), 0.0005, 0.05), 0.0)


def crt_whine(r, dur, level=0.05):
    n = ns(dur)
    t = tt(n)
    x = np.sin(TAU * 15734.0 * t + 0.002 * np.cumsum(r.standard_normal(n)) / 30) * level
    x += hum(r, dur, 59.94, (0.0, 0.3, 0.0, 0.15, 0.0, 0.08), 0.02) * level * 0.35
    return f32(x)


def crt_off(r):
    """CRT power-off collapse: flyback whine cut (tick), HV discharge 'tsew' zap (falling chirp +
    crackle), sub pop, then thermal tick-tick of the cooling glass. sync = switch (t=0)."""
    dur = 1.2
    n = ns(dur)
    t = tt(n)
    x = np.zeros(n, np.float64)
    f = 5200 * np.exp(-t / 0.07) + 90
    zap = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / 0.12)
    x += zap * 0.6
    x += hp(white(n, r), 2500) * np.exp(-t / 0.05) * 0.5
    x += hp(poisson_clicks(r, n, 400 * np.exp(-t / 0.1)), 1000) * 0.8
    x += np.sin(TAU * 38 * t) * np.exp(-t / 0.08) * 0.6
    for k in range(3):
        tick = impact(r, "glass", 1.5, 0.2)
        i = ns(0.45 + 0.22 * k + r.uniform(0, 0.08))
        if i + len(tick) < n:
            x[i:i + len(tick)] += tick * 0.25
    return Snd(fade(norm(f32(x), 0.95), 0.0005, 0.1), 0.0)


def tv_static(r, dur, level=1.0):
    """analog snow: white noise, presence-heavy, with 59.94 Hz field buzz and fluttering dropouts."""
    n = ns(dur)
    t = tt(n)
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        nz = peq(white(n, r), 3500, 4, 0.7)
        am = 0.85 + 0.15 * np.sign(np.sin(TAU * 59.94 * t)) + 0.1 * smooth_noise(n, r, 12, 2)
        out[:, ch] = nz * am * 0.25
    out += hum(r, dur, 59.94, (0.2, 0.3, 0.2, 0.2, 0.15, 0.1), 0.3)[:, None] * 0.04
    return norm(out, 0.9) * level


def glitch(r, dur=0.3, level=1.0):
    """digital glitch burst: crushed noise + stuttered slices + tonal square blips."""
    n = ns(dur)
    x = np.zeros(n, np.float32)
    i = 0
    while i < n:
        L = int(SR * r.uniform(0.008, 0.05))
        kind = r.integers(0, 4)
        tt_ = tt(L)
        if kind == 0:
            s = crush(white(L, r), r.integers(2, 6), r.integers(4, 24))
        elif kind == 1:
            s = np.sign(np.sin(TAU * r.uniform(200, 3000) * tt_))
        elif kind == 2:
            s = np.zeros(L)
        else:
            s = crush(np.sin(TAU * r.uniform(60, 400) * tt_), 3, 8)
        rep = r.integers(1, 4)
        s = np.tile(s, rep)[: min(n - i, L * rep)]
        x[i:i + len(s)] = s * r.uniform(0.3, 1.0)
        i += len(s)
    return f32(fade(x, 0.001, 0.01) * level * 0.6)


# ------------------------------------------------------------------------------------------------
# small foley
# ------------------------------------------------------------------------------------------------
def velcro(r, dur=0.3):
    """velcro rip: rows of hooks releasing — Poisson micro-releases with row periodicity (~45 Hz AM)."""
    n = ns(dur)
    t = tt(n)
    pull = env_pts(n, [(0, 0), (0.02, 1), (dur * 0.7, 0.8), (dur, 0)])
    rows = 0.55 + 0.45 * np.abs(np.sin(TAU * 22 * t * (1 + 0.5 * t / dur)))
    exc = poisson_clicks(r, n, 5000 * pull * rows, pull * rows)
    x = bp(exc, 1200, 9000) + resonators(exc, [850, 1900, 3100], [0.01] * 3, [0.3] * 3)
    return Snd(fade(norm(x, 0.9), 0.001, 0.01), 0.02)


def telescope(r, sections=3, gap=0.13):
    """telescoping pole: slide friction then detent click per section, pitch rising as sections thin."""
    evs = []
    t = 0.0
    dur = sections * gap + 0.5
    n = ns(dur)
    for k in range(sections):
        L = gap * 0.75
        m = ns(L)
        sl = svf(white(m, r), 2500 + 900 * k, 2.5, "bp") * env_pts(m, [(0, 0), (L * 0.2, 0.4), (L, 0.8)])
        evs.append((t, sl * 0.3, 1.0, 0.0))
        t += L
        clk = impact(r, "metal", 0.7 - 0.12 * k, 1.0, 0.3)
        evs.append((t, clk, 0.9, 0.0))
        t += gap * 0.25
    return Snd(scatter(n, evs), gap * 0.75)


def hammer_tap(r, n_taps=3, gap=0.16):
    """tack hammer on wooden pole through cloth: steel tick + wood body thunk, taps accelerating."""
    evs = []
    t = 0.0
    for k in range(n_taps):
        s = impact(r, "metal", 0.55, 0.5, 0.2) * 0.5
        w = impact(r, "wood", 1.0, 1.0, 0.25)
        m = max(len(s), len(w))
        x = np.zeros(m, np.float32)
        x[: len(s)] += s
        x[: len(w)] += w
        evs.append((t, x, 0.7 + 0.3 * (k == n_taps - 1), 0.0))
        t += gap * (0.85 ** k)
    return Snd(norm(scatter(ns(t + 0.4), evs), 0.95), 0.0)


def boot_step(r, surface="concrete", weight=1.0, gear=True):
    """heel strike thud + sole scuff/crunch + gear jingle."""
    dur = 0.45
    n = ns(dur)
    x = np.zeros(n, np.float32)
    h = thud(r, 75 * r.uniform(0.9, 1.1), 0.25, 0.9, 700 if surface != "grass" else 400)
    x[: len(h)] += h * weight
    if surface in ("concrete", "rubble", "stone"):
        g = gravel(r, 0.09 if surface == "concrete" else 0.16, 1400 if surface == "concrete" else 2600)
        i = ns(0.01)
        x[i:i + len(g)] += g * (0.5 if surface == "concrete" else 0.9)
    else:  # grass swish
        m = ns(0.18)
        sw = bp(white(m, r), 1500, 7000) * env_pts(m, [(0, 0), (0.03, 1), (0.18, 0)]) * 0.25
        x[:m] += sw
    if gear:
        for k in range(r.integers(1, 3)):
            c = impact(r, "metal", 0.35, 0.25, 0.15)
            i = ns(0.03 + r.uniform(0, 0.08))
            x[i:i + len(c)] += c[: n - i] * 0.35
    return Snd(fade(norm(x, 0.95), 0.0005, 0.05), 0.0)


def cloth_rustle(r, dur=0.5, intensity=1.0, bright=1.0):
    n = ns(dur)
    burst = poisson_clicks(r, n, 90 * intensity, 1.0)
    burst = np.convolve(burst, np.hanning(ns(0.012)), "same")
    env = np.clip(lp(burst, 30, 1) * 8, 0, 1) + 0.15
    x = bp(white(n, r), 700 * bright, 6000 * bright) * env * env_pts(n, [(0, 0), (0.04, 1), (dur * 0.7, 0.7), (dur, 0)])
    return Snd(fade(norm(x, 0.8), 0.005, 0.05), 0.0)


def pen_scratch(r, dur=0.8, strokes=5):
    """felt-tip flourish: friction noise gated by pen speed + stick-slip at a speed-dependent rate."""
    n = ns(dur)
    t = tt(n)
    v = np.zeros(n)
    cs = np.sort(r.uniform(0.05, dur - 0.05, strokes))
    for c in cs:
        v += np.exp(-((t - c) / (dur / strokes * 0.35)) ** 2) * r.uniform(0.6, 1.0)
    v = np.clip(v, 0, 1)
    fr = bp(white(n, r), 2200, 7500) * v ** 1.3
    ss = stick_slip(r, dur, 250 + 900 * v, 0.2, v)
    ss = resonators(ss, [1800, 3300, 5200], [0.004] * 3, [1, 0.6, 0.4])
    x = fr * 0.6 + ss * 0.8
    m = min(n, ns(0.08))
    x[:m] *= np.linspace(0.3, 1.0, m)                    # the stroke gathers speed after touchdown
    nib = impact(r, "wood", 0.25, 1.0, 0.05)          # the nib touches down: the sync point
    x[: len(nib)] += nib * 2.0
    dot = impact(r, "wood", 0.3, 0.6, 0.08)
    i = n - ns(0.07)
    x[i:i + len(dot)] += dot[: n - i] * 0.6
    return Snd(fade(norm(x, 0.8), 0.003, 0.03), 0.0)


def ka_ching(r):
    """cash-register: lever 'ka' (clunk + ratchet) then bell 'ching' (inharmonic small bell) + coin
    jingle. sync = the bell strike."""
    dur = 2.2
    n = ns(dur)
    x = np.zeros(n, np.float32)
    k1 = impact(r, "tin", 1.4, 1.0)
    x[: len(k1)] += k1 * 0.45
    rat = stick_slip(r, 0.07, 180, 0.1)
    rat = resonators(rat, [1400, 2600, 4100], [0.01] * 3, [1, 0.6, 0.4])
    x[ns(0.02):ns(0.02) + len(rat)] += rat * 0.35
    tb = ns(0.12)
    bell = ring([2093 * q for q in (1.0, 2.01, 2.63, 3.95, 5.43)], [1.4, 0.9, 0.7, 0.45, 0.3],
                [1.0, 0.5, 0.45, 0.25, 0.2], dur - 0.12, r)
    bell2 = ring([2793 * q for q in (1.0, 2.63, 3.95)], [1.0, 0.6, 0.4], [0.6, 0.3, 0.2], dur - 0.12, r)
    x[tb:tb + len(bell)] += (bell + bell2) * 0.6
    strike = impact(r, "metal", 0.5, 1.0, 0.1)          # clapper strike: the 'ch' of 'ching' (sync point)
    x[tb:tb + len(strike)] += strike * 1.0
    for k in range(5):
        c = impact(r, "metal", 0.4, 0.5, 0.4)
        i = tb + ns(0.03 + 0.05 * k + r.uniform(0, 0.03))
        x[i:i + len(c)] += c[: n - i] * 0.25 * (0.8 ** k)
    return Snd(fade(norm(x, 0.95), 0.0005, 0.3), 0.12)


def cryo_pulse(r, dur, period=0.72, level=1.0):
    """dilution-refrigerator pulse-tube: rotary-valve gas 'whoosh-chh' every period + compressor
    drone + turbopump whine. The Hypermind's breathing."""
    n = ns(dur)
    t = tt(n)
    ph = (t / period) % 1.0
    inh = np.exp(-((ph - 0.18) / 0.09) ** 2)
    exh = np.exp(-((ph - 0.62) / 0.14) ** 2) * (ph > 0.45)
    hiss_i = svf(white(n, r), 2500 + 3000 * inh, 1.2, "bp") * inh * 0.35
    hiss_e = svf(pink(n, r), 900 + 800 * exh, 0.9, "bp") * exh * 0.5
    thump = svf(brown(n, r), 90, 1.5, "bp") * np.exp(-((ph - 0.5) / 0.03) ** 2) * 0.9
    comp = hum(r, dur, 27.5 * 2, (0.3, 0.6, 0.25, 0.15), 0.0) * 0.12
    turbo = np.sin(TAU * 1210 * t + 3 * smooth_noise(n, r, 0.2, 1)) * 0.012 + np.sin(TAU * 2420 * t) * 0.006
    return f32((hiss_i + hiss_e + thump + comp + turbo) * level)


def squelch_touch(r):
    """sticky: finger meets tacky pole — wet 'tck' (fast resonant sweep) + micro-bubbles."""
    n = ns(0.18)
    t = tt(n)
    x = svf(white(n, r), 3500 * np.exp(-t / 0.02) + 600, 5, "bp") * env_ad(n, 0.001, 0.05)
    b = bubble(r, 1.4, 0.8, 0.5, 0.4)
    x[ns(0.01):ns(0.01) + len(b)] += b[: n - ns(0.01)]
    return Snd(fade(norm(x, 0.8), 0.0005, 0.03), 0.0)


def squelch_stretch(r, dur=0.9):
    """a viscous strand stretching from finger to pole: rubbery stick-slip squeak rising in pitch,
    wet crackle, and a final tiny 'plip' as it snaps (sync = start)."""
    n = ns(dur)
    t = tt(n)
    u = t / dur
    rate = 180 + 700 * u ** 1.5
    ss = stick_slip(r, dur, rate, 0.1, env_pts(n, [(0, 0), (0.1, 0.6), (dur * 0.9, 1), (dur, 0)]))
    sq = svf(ss, 900 + 2200 * u, 9, "bp") * 0.9
    wet = svf(poisson_clicks(r, n, 120 + 300 * u), 1800, 2, "bp") * 0.4
    goo = svf(brown(n, r), 300 + 400 * u, 3, "bp") * env_pts(n, [(0, 0), (0.08, 0.5), (dur, 0.2)]) * 0.5
    x = sq + wet + goo
    pl = bubble(r, 0.9, 0.6, 1.0, 0.3)
    i = n - ns(0.02)
    y = np.zeros(n + len(pl), np.float32)
    y[:n] = x
    y[i:i + len(pl)] += pl * 0.9
    return Snd(fade(norm(y, 0.85), 0.01, 0.03), 0.0)


def sniff(r, count=2):
    """nasal inhalation(s): noise through nasal formants (sync = first sniff onset)."""
    dur = 0.28 * count + 0.2
    n = ns(dur)
    x = np.zeros(n, np.float32)
    for k in range(count):
        m = ns(0.2)
        s = white(m, r)
        s = svf(s, 1300, 3, "bp") * 0.6 + svf(s, 3800, 4, "bp") * 0.5 + hp(s, 6000) * 0.2
        s *= env_pts(m, [(0, 0), (0.03, 1), (0.12, 0.8), (0.2, 0)])
        i = ns(0.26 * k)
        x[i:i + m] += s * (1.0 if k == 0 else 0.8)
    return Snd(fade(norm(x, 0.7), 0.002, 0.02), 0.0)


def paper_slide(r, dur=0.7, bright=1.0):
    n = ns(dur)
    t = tt(n)
    v = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5
    x = svf(white(n, r), 1500 + 3500 * v * bright, 0.9, "bp") * v
    x += bp(poisson_clicks(r, n, 400 * v), 2500, 9000) * 0.3
    return Snd(fade(norm(x, 0.8), 0.01, 0.05), dur * 0.5)


def page_slam(r, size=1.0):
    """a large flat sheet slapping down: air-cushion 'whump' + slap click. sync = impact."""
    pre = 0.18
    n = ns(pre + 0.7)
    t = tt(n)
    x = np.zeros(n, np.float32)
    # the air cushion rushes out, then is squeezed shut just before contact (tiny gap = crisp slap)
    env = np.clip(t / (pre - 0.025), 0, 1) ** 3 * np.clip((pre - t) / 0.02, 0, 1)
    rush = svf(pink(n, r), 300 + 2500 * np.clip(t / pre, 0, 1), 0.8, "lp") * env
    x += rush * 0.3
    i = ns(pre)
    w = thud(r, 62 * size, 0.5, 1.4, 500)
    x[i:i + len(w)] += w[: n - i] * 1.0
    sl = hp(white(ns(0.05), r), 1000) * env_ad(ns(0.05), 0.0002, 0.035)
    x[i:i + len(sl)] += sl * 3.0
    return Snd(fade(norm(x, 0.95), 0.01, 0.1), pre)


def pole_thunk(r):
    """a wooden pole driven into rubble: wood body thunk (pole modes) + ground thud + gravel spray."""
    dur = 1.5
    n = ns(dur)
    x = np.zeros(n, np.float32)
    exc = np.zeros(n, np.float32)
    exc[0] = 1.0
    exc[:ns(0.003)] += hp(white(ns(0.003), r), 1000) * 0.5
    wood = resonators(exc, [165, 430, 830, 1370, 2050], [0.12, 0.08, 0.05, 0.035, 0.02], [1, 0.7, 0.5, 0.35, 0.2])
    x += wood * 3.0
    h = thud(r, 58, 0.6, 1.2, 400)
    x[: len(h)] += h * 1.2
    g = gravel(r, 0.35, 2600, 0.8)
    x[ns(0.005):ns(0.005) + len(g)] += g * 0.7
    return Snd(fade(norm(x, 0.95), 0.0003, 0.2), 0.0)


def bird(r, species=None, dist=30.0):
    """synthetic songbird phrase (sync = start). species: whistle | trill | warble | chip."""
    species = species or r.choice(["whistle", "trill", "warble", "chip"])
    notes = []
    if species == "whistle":
        f0 = r.uniform(2400, 3600)
        for k in range(r.integers(2, 4)):
            L = r.uniform(0.18, 0.4)
            notes.append((k * (L + 0.08), L, np.linspace(f0 * r.uniform(0.9, 1.1), f0 * r.uniform(0.8, 1.25), 64)))
    elif species == "trill":
        f0 = r.uniform(3800, 5600)
        cnt = r.integers(8, 18)
        for k in range(cnt):
            notes.append((k * 0.05, 0.035, np.linspace(f0 * 1.15, f0 * 0.85, 16)))
    elif species == "warble":
        t0 = 0.0
        for k in range(r.integers(4, 9)):
            L = r.uniform(0.05, 0.14)
            a, b_ = r.uniform(2200, 5500, 2)
            notes.append((t0, L, np.linspace(a, b_, 16)))
            t0 += L + r.uniform(0.01, 0.06)
    else:
        f0 = r.uniform(4500, 7000)
        for k in range(r.integers(2, 5)):
            notes.append((k * r.uniform(0.12, 0.2), 0.03, np.linspace(f0, f0 * 0.6, 8)))
    dur = max(s + L for s, L, _ in notes) + 0.05
    n = ns(dur)
    x = np.zeros(n, np.float32)
    for s, L, contour in notes:
        m = ns(L)
        f = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(contour)), contour)
        f *= 1 + 0.02 * np.sin(TAU * r.uniform(20, 60) * tt(m))
        ph = TAU * np.cumsum(f) / SR
        e = np.sin(np.pi * np.linspace(0, 1, m)) ** 1.5
        seg = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * e
        i = ns(s)
        x[i:i + m] += seg[: n - i]
    x = air(x, dist * 12)
    return Snd(fade(x, 0.002, 0.01), 0.0)


def klaxon_whoop(r, dur=1.4, f0=146.8):
    """deep rotating-beacon horn 'whoop' tuned to D3 (pitch sweep up then settle), with motor whirr."""
    n = ns(dur)
    t = tt(n)
    f = f0 * (0.75 + 0.25 * np.clip(t / 0.25, 0, 1)) * (1 + 0.01 * np.sin(TAU * 5 * t))
    ph = TAU * np.cumsum(f) / SR
    x = sum(np.sin(ph * k) / k ** 1.1 for k in range(1, 9))
    x = lp(f32(x), 1400, 2) * env_pts(n, [(0, 0), (0.06, 1), (dur * 0.8, 0.8), (dur, 0)])
    return Snd(fade(norm(x, 0.8), 0.01, 0.1), 0.0)


def stone_slide(r, dur=2.0, weight=1.0):
    """massive stone slab grinding open: low stick-slip + gritty friction + settling clunk."""
    n = ns(dur)
    v = env_pts(n, [(0, 0), (0.2, 0.7), (dur * 0.7, 1.0), (dur, 0)])
    ss = stick_slip(r, dur, 25 + 25 * v, 0.4, v)
    body = resonators(ss, [48, 71, 105, 160, 240], [0.3] * 5, [1, 0.8, 0.6, 0.4, 0.3])
    grit = bp(poisson_clicks(r, n, 1500 * v, v), 600, 5000) * 0.3
    x = body * 0.7 + grit + svf(brown(n, r), 120, 1.0, "lp") * v * 0.4
    end = thud(r, 45, 0.6, 1.0, 300)
    y = np.concatenate([x, np.zeros(len(end), np.float32)])
    y[n - ns(0.05):n - ns(0.05) + len(end)] += end * weight
    return Snd(fade(norm(y, 0.9), 0.02, 0.2), 0.0)


def zap(r, dur=0.4, f0=4000, f1=200):
    n = ns(dur)
    t = tt(n)
    f = f1 + (f0 - f1) * np.exp(-t / (dur * 0.25))
    x = np.sin(TAU * np.cumsum(f) / SR) * env_ad(n, 0.001, dur)
    x += hp(poisson_clicks(r, n, 1500 * np.exp(-t / (dur * 0.3))), 1500) * 0.5
    return Snd(fade(norm(f32(x), 0.9), 0.0005, 0.02), 0.0)


def plasma_buzz(r, dur, f=110.0, level=1.0):
    n = ns(dur)
    t = tt(n)
    saw = 2 * ((f * t + 0.3 * smooth_noise(n, r, 3, 2)) % 1.0) - 1
    x = svf(f32(saw), 1800 + 600 * smooth_noise(n, r, 6, 2), 2, "lp") * 0.5
    x += hp(poisson_clicks(r, n, 300), 2500) * 0.3
    return norm(x, 0.9) * level

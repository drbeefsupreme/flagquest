"""OPVS SCHISMATICVM (film 3) — Foley models: the physics of tesserae (owner: Foley-2).

Builds on the shared tools/foley_lib.py (filters, noise, modal synthesis) and tools/foley_flutter.py (cloth).
No samples: every sound is computed at 48 kHz from first principles.

THE ATOM. A tessera (smalti glass, gold-leaf glass or stone, ~1 cm) is far too small and stiff to ring audibly
by itself (a 10x10x5 mm glass plate's first bending mode is ~170 kHz). What we hear when tesserae move is
  1. the Hertzian contact pulse: duration tau ~ (m^2 / (R E*^2 v))^(1/5) -> faster/harder = shorter = brighter;
     wet lime mortar is soft -> a long pulse -> the dull 'tup' of a tile being set,
  2. the local modes of whatever is struck: the marble floor's 'clack', the glassy chatter of a heap of tiles,
     the neighbours knocked in a set wall,
  3. the rigid-body choreography: edge-then-face double contacts, bounces with restitution
     (v_k+1 = e v_k, t_k+1 = t_k + 2 v_k+1 / g), the rocking settle (contact intervals shrinking geometrically),
     skids across the floor, and on a heap the neighbours jostled.
Every mass process (an avalanche of 50,000 tesserae, a split-flap wave of slop screens, a quadtree of splits,
the fallen tiles rising and setting again) is those three ingredients rendered as grains from a pre-computed
bank and placed by a numba kernel, so the sound of the mosaic literally follows its tile physics.

Splitting: a tile's modal frequencies scale as 1/size, so every quadtree generation (4x the pieces, half the
size) ticks an octave higher and 6 dB softer per piece; after ~5 generations the ticks leave the audible band
and what remains is a hiss of glittering dust (the sound of recursion dissolving into the ultrasonic).
Generators return Snd(x, sync) (x mono or stereo float32, sync = seconds from x[0] to the perceptual hit) or
plain arrays for beds. Tonal elements are pitched only on D / A / E (safe over the whole score, Composer-2).
"""
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import foley_lib as L  # noqa: E402
from foley_lib import SR, TAU, Snd, R, midi_hz, ns  # noqa: E402

G = 9.81
V_REF = 7.0                      # m/s: a tessera falling ~2.5 m; gain 1 at this impact speed
D1_HZ = 36.708                   # tuned subs sit on D (Composer-2)


def db(x):
    return 10.0 ** (np.asarray(x, float) / 20.0)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12)


def rms_norm(x, target=1.0):
    return L.f32(np.asarray(x) * (target / rms(x)))


def at(n, t0, pts):
    """piecewise-linear envelope over an n-sample buffer that starts at global time t0; pts in global seconds."""
    return L.f32(np.interp(np.arange(n) / SR + t0, [p[0] for p in pts], [p[1] for p in pts]))


def shaped(x, t0, pts):
    g = at(len(x), t0, pts)
    return L.f32(x * (g[:, None] if x.ndim == 2 else g))


def wide(fn, r, *a, common=0.5, **k):
    """two realisations of a mono noise generator sharing a common part -> natural stereo width."""
    c = fn(r, *a, **k)
    return L.stereo(common * c + (1 - common) * fn(r, *a, **k), common * c + (1 - common) * fn(r, *a, **k))


def place_mono(out, x, i, gain=1.0, pan=0.0):
    """add mono x into stereo out at sample i (clipped to the buffer), constant-power pan."""
    n = len(out)
    if i >= n or i + len(x) <= 0:
        return
    a, b = max(0, i), min(n, i + len(x))
    p = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    seg = x[a - i:b - i] * gain
    out[a:b, 0] += seg * np.cos(p)
    out[a:b, 1] += seg * np.sin(p)


# ================================================================================================
# the grain engine
# ================================================================================================
@njit(cache=True)
def _place(out, bank, blen, idx, starts, gl, gr):
    n = out.shape[0]
    for e in range(idx.shape[0]):
        k = idx[e]
        s = starts[e]
        m = blen[k]
        if s >= n or s + m <= 0:
            continue
        a = 0 if s >= 0 else -s
        b = m if s + m <= n else n - s
        g1 = gl[e]
        g2 = gr[e]
        for j in range(a, b):
            v = bank[k, j]
            out[s + j, 0] += v * g1
            out[s + j, 1] += v * g2


class Bank:
    """mono grains grouped by key; render() places (time, key-index, gain, pan) events with numba."""

    def __init__(self):
        self.grains, self.groups = [], {}
        self._x = None

    def add(self, key, grains):
        i0 = len(self.grains)
        self.grains += [L.f32(g) for g in grains]
        self.groups.setdefault(key, []).extend(range(i0, len(self.grains)))
        self._x = None

    def pick(self, r, key, count):
        g = np.asarray(self.groups[key], np.int64)
        return g[r.integers(0, len(g), count)]

    def _arrays(self):
        if self._x is None:
            m = max(len(g) for g in self.grains)
            self._x = np.zeros((len(self.grains), m), np.float32)
            self._len = np.array([len(g) for g in self.grains], np.int64)
            for i, g in enumerate(self.grains):
                self._x[i, :len(g)] = g
        return self._x, self._len

    def render(self, out, t0, times, idx, gains, pans):
        if len(times) == 0:
            return out
        x, ln = self._arrays()
        starts = np.round((np.asarray(times, np.float64) - t0) * SR).astype(np.int64)
        th = (np.clip(np.asarray(pans, np.float64), -1, 1) + 1) * np.pi / 4
        g = np.asarray(gains, np.float64)
        _place(out, x, ln, np.asarray(idx, np.int64), starts, (np.cos(th) * g).astype(np.float32),
               (np.sin(th) * g).astype(np.float32))
        return out


# ================================================================================================
# the atom: one tessera contact
# ================================================================================================
TILE = {  # the moving tessera:  ring band (Hz)   ring t60 (s)     ring   click  chip
    "smalti": ((3200, 12500), (0.004, 0.022), 1.0, 0.45, 0.25),
    "gold": ((3800, 14000), (0.007, 0.040), 1.1, 0.40, 0.20),
    "stone": ((1800, 7500), (0.002, 0.009), 0.8, 0.65, 0.30),
    "screen": ((1300, 5200), (0.003, 0.012), 0.8, 0.75, 0.10),
}
SURF = {  # what it strikes: contact tau (s)  body band (Hz)  body t60 (s)    body  micro-contacts  loudness
    "floor": (60e-6, (650, 3200), (0.006, 0.022), 0.9, 1, 1.0),
    "heap": (80e-6, (2400, 9000), (0.003, 0.012), 0.7, 3, 0.75),
    "mortar": (420e-6, (140, 420), (0.012, 0.030), 1.0, 0, 0.55),
    "wall": (70e-6, (1800, 7000), (0.003, 0.010), 0.6, 1, 0.7),
}
VCLASS = (0.3, 0.65, 1.0)        # soft / medium / hard contacts (spectral brightness classes)


def tile_click(r, mat="smalti", surf="floor", vel=1.0, size=1.0):
    """one tessera contact (mono, peak 1): Hertzian pulse (+ edge/face micro-contacts, chipping on hard hits)
    exciting the tile's corner/edge chatter modes and the struck surface's local modes."""
    (lo, hi), (d0, d1), ring_g, click_g, chip_g = TILE[mat]
    tau0, (b0, b1), (e0, e1), body_g, micro, _ = SURF[surf]
    vel = max(float(vel), 0.05)
    tau = tau0 * size ** 0.6 * vel ** -0.2
    n = ns(0.012 + 1.35 * max(d1 * size ** 0.3, e1))
    exc = np.zeros(n)
    m = max(2, int(round(tau * SR)))
    pulse = np.sin(np.pi * (np.arange(m) + 0.5) / m)
    exc[:m] += pulse
    for _ in range(r.integers(0, micro + 1)):          # edge -> face, neighbours knocked
        i = ns(r.uniform(0.0008, 0.007))
        if i + m < n:
            exc[i:i + m] += pulse * r.uniform(0.15, 0.6)
    k = ns(0.0015)
    exc[:k] += chip_g * vel * r.standard_normal(k) * np.exp(-np.arange(k) / (SR * 0.0004))
    hi_v = lo * (hi / lo) ** (0.45 + 0.55 * min(vel, 1.0))     # a harder hit excites higher modes
    fr = np.exp(r.uniform(np.log(lo), np.log(hi_v), 4)) / size ** 0.5
    x = L.resonators(exc, fr, r.uniform(d0, d1, 4) * size ** 0.3, r.uniform(0.35, 1.0, 4)) * ring_g
    fb = np.exp(r.uniform(np.log(b0), np.log(b1), 3)) / size ** 0.25
    x = x + L.resonators(exc, fb, r.uniform(e0, e1, 3), r.uniform(0.4, 1.0, 3)) * body_g * min(1.5, size) ** 0.5
    x = x + L.hp(L.f32(exc), 2000) * click_g
    if surf == "mortar":                                  # wet lime squeezed at the joints
        q = ns(0.02)
        x[:q] += L.lp(L.white(q, r), 1800) * L.env_ad(q, 0.002, 0.015) * 0.25
    return L.fade(L.norm(x, 1.0), 0.0001, 0.004)


def scrape_grain(r, dur=0.08, bright=1.0):
    """a tessera skidding across marble: friction noise with micro stick-slip, decelerating."""
    n = ns(dur)
    u = np.arange(n) / n
    v = (1 - u) ** 1.5
    ss = L.stick_slip(r, dur, 300 + 900 * v, 0.6, v)
    x = L.svf(L.white(n, r), (2200 + 3500 * v) * bright, 1.2, "bp") * v * 0.5
    x = x + L.resonators(ss, r.uniform(2500, 7000, 3) * bright, [0.002] * 3, [1, 0.7, 0.5]) * 0.6
    return L.fade(L.norm(x, 1.0), 0.002, 0.01)


def split_grain(r, f, level=0):
    """a tessera cleaving in two: a brittle snap (sub-ms crack) + the new halves' modes near f (1/size scaling)."""
    n = ns(0.03 if f < 9000 else 0.012)
    exc = np.zeros(n)
    m = ns(0.00025)
    exc[:m] = np.linspace(1, -1, m)
    k = ns(0.003)
    exc[:k] += 0.35 * np.random.default_rng(r.integers(1 << 30)).standard_normal(k) * np.exp(-np.arange(k) / (SR * 0.0008))
    x = L.hp(L.f32(exc), 1500) * 0.8
    if f < 16500:
        fr = f * r.uniform(0.88, 1.12, 3) * np.array([1.0, 1.47, 2.09])
        t60 = 0.014 / (1.35 ** level) * r.uniform(0.7, 1.3, 3)
        x = x + L.resonators(exc, fr, t60, [1.0, 0.5, 0.3])
    return L.fade(L.norm(x, 1.0), 0.0001, 0.003)


def flip_grain(r, face="smalti", screen=False):
    """a tessera turning over on its bed (split-flap): release tick, a half-turn whirr, the slap onto its
    other face (+ for a slop screen: the electrical 'tzt' of the feed lighting). sync = the slap."""
    t_fl = r.uniform(0.016, 0.04)
    n = ns(t_fl + 0.06)
    x = np.zeros(n, np.float32)
    a = tile_click(r, "smalti", "wall", 0.25, 0.7)
    x[:min(n, len(a))] += a[:n] * 0.3
    m = ns(t_fl)
    wh = L.svf(L.white(m, r), r.uniform(3500, 7000), 1.4, "bp") * np.sin(np.pi * np.arange(m) / m) ** 2
    x[ns(0.001):ns(0.001) + m] += wh[: n - ns(0.001)] * 0.12
    i = ns(0.001 + t_fl)
    b = tile_click(r, "screen" if screen else face, "wall", r.uniform(0.6, 1.0), 0.9)
    x[i:i + len(b)] += b[: n - i]
    if screen:
        q = ns(0.012)
        t = np.arange(q) / SR
        tz = (np.sign(np.sin(TAU * r.uniform(90, 140) * t)) * 0.3 + L.hp(L.white(q, r), 4000)) * np.exp(-t / 0.004)
        x[i:i + q] += L.f32(tz[: n - i]) * 0.25
    return Snd(L.fade(L.norm(x, 1.0), 0.0001, 0.005), (i) / SR)


# ------------------------------------------------------------------------------------------------
_BANK = {}


def tile_bank(seed=4242, per=20):
    """the shared grain bank: every (material, surface, velocity class) + skids + splits + flips."""
    if seed in _BANK:
        return _BANK[seed]
    r = R(seed)
    b = Bank()
    for mat in TILE:
        for surf in SURF:
            if mat == "screen" and surf == "mortar":
                continue
            for vc, v in enumerate(VCLASS):
                b.add((mat, surf, vc), [tile_click(r, mat, surf, v * r.uniform(0.85, 1.15), r.uniform(0.7, 1.4))
                                        for _ in range(per)])
    b.add("skid", [scrape_grain(r, r.uniform(0.04, 0.16), r.uniform(0.8, 1.2)) for _ in range(per)])
    for lv in range(9):
        b.add(("split", lv), [split_grain(r, 650.0 * 2 ** lv * r.uniform(0.9, 1.1), lv) for _ in range(12)])
    for face in ("smalti", "gold"):
        b.add(("flip", face), [flip_grain(r, face).x for _ in range(per)])
    b.add(("flip", "screen"), [flip_grain(r, "smalti", True).x for _ in range(per)])
    _BANK[seed] = b
    return b


def vclass(v):
    """impact speed (m/s) -> velocity class index."""
    u = np.asarray(v, float) / V_REF
    return np.where(u > 0.8, 2, np.where(u > 0.3, 1, 0))


# ================================================================================================
# tile physics: bounce trains
# ================================================================================================
REST = {"floor": 0.42, "heap": 0.18, "mortar": 0.0, "wall": 0.3}


def bounce_events(r, t_hit, v0, surf, kmax=7, vmin=0.3, settle=True):
    """vectorised rigid-body choreography of many tesserae.
    t_hit, v0: (N,) first-contact times (s) and speeds (m/s); surf: (N,) surface names.
    Returns arrays (t, v, surf, kind, tile_id): kind 0 = impact/bounce, 2 = settle (rocking to rest)."""
    t_hit = np.asarray(t_hit, np.float64)
    v0 = np.asarray(v0, np.float64)
    surf = np.asarray(surf)
    N = len(t_hit)
    ids = np.arange(N)
    e = np.array([REST[s] for s in surf]) * r.uniform(0.75, 1.2, N)
    fac = np.cumprod(e[:, None] * r.uniform(0.8, 1.1, (N, kmax)), axis=1)
    vk = np.concatenate([v0[:, None], v0[:, None] * fac[:, :-1]], axis=1)          # speed at contact k
    air = 2 * vk[:, 1:] / G * r.uniform(0.65, 1.05, (N, kmax - 1))                   # airtime before contact k
    tk = t_hit[:, None] + np.concatenate([np.zeros((N, 1)), np.cumsum(air, axis=1)], axis=1)
    keep = vk >= vmin
    keep[:, 0] = True
    keep = np.cumprod(keep, axis=1).astype(bool)
    nk = keep.sum(1)
    T, V, S, K, I = [tk[keep]], [vk[keep]], [np.repeat(surf, nk)], [np.zeros(nk.sum(), np.int64)], [np.repeat(ids, nk)]
    if settle:
        last = nk - 1
        t_last = tk[ids, last]
        v_last = vk[ids, last]
        J = 5
        dt = r.uniform(0.010, 0.032, N)[:, None] * r.uniform(0.55, 0.75, N)[:, None] ** np.arange(J)[None, :]
        ts = t_last[:, None] + np.cumsum(dt, axis=1)
        vs = (np.clip(v_last, 0.2, 3.0) * 0.45)[:, None] * 0.7 ** np.arange(1, J + 1)[None, :]
        cnt = r.integers(1, J + 1, N)
        m = (np.arange(J)[None, :] < cnt[:, None]) & (surf[:, None] != "mortar")
        ms = m.sum(1)
        T.append(ts[m])
        V.append(vs[m])
        S.append(np.repeat(surf, ms))
        K.append(np.full(ms.sum(), 2, np.int64))
        I.append(np.repeat(ids, ms))
    return np.concatenate(T), np.concatenate(V), np.concatenate(S), np.concatenate(K), np.concatenate(I)


def render_contacts(out, t0, bank, r, t, v, surf, mat, pan, gain=1.0):
    """place tessera contacts (arrays) into stereo `out` (buffer starting at global t0)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    surf, mat = np.asarray(surf), np.asarray(np.broadcast_to(np.asarray(mat), t.shape))
    pan = np.broadcast_to(np.asarray(pan, float), t.shape)
    vc = vclass(v)
    idx = np.zeros(len(t), np.int64)
    loud = np.zeros(len(t))
    for s in SURF:
        ms = surf == s
        if not ms.any():
            continue
        loud[ms] = SURF[s][5]
        for m_ in TILE:
            mm = ms & (mat == m_)
            if not mm.any():
                continue
            if m_ == "screen" and s == "mortar":
                mm_key = "smalti"
            else:
                mm_key = m_
            for c in range(3):
                sel = np.nonzero(mm & (vc == c))[0]
                if len(sel):
                    idx[sel] = bank.pick(r, (mm_key, s, c), len(sel))
    g = gain * np.clip(v / V_REF, 0, 2.5) ** 1.2 * loud
    bank.render(out, t0, t, idx, g, pan)
    return out


# ================================================================================================
# mass processes
# ================================================================================================
def chunk_thud(r, size=1.0):
    """a slab of set tesserae still bonded by mortar hits the floor: mortar body thud + crumble + stone crack."""
    dur = 0.6
    n = ns(dur)
    x = np.zeros(n, np.float32)
    th = L.thud(r, 62 * r.uniform(0.85, 1.2) / size ** 0.3, 0.45, 1.1, 500)
    x[: len(th)] += th[:n]
    cr = L.gravel(r, 0.22, 2400, 0.7)
    x[ns(0.003):ns(0.003) + len(cr)] += cr[: n - ns(0.003)] * 0.8
    im = L.impact(r, "stone", 2.0 * size, 1.0)
    x[: len(im)] += im[:n] * 0.5
    return L.fade(L.norm(x, 1.0), 0.0002, 0.05)


def tile_rain(r, t0, dur, t_land, v_land, pan, heap_p=None, mats=("smalti", 0.55, "gold", 0.3, "stone", 0.15),
              skid_p=0.25, bank=None, gain=1.0, jostle=0.6):
    """N tesserae landing (first-contact times, speeds m/s, pans) with their full choreography: bounces, settle,
    skids on bare floor, neighbours jostled when they land on the heap. heap_p: (N,) probability of heap.
    -> stereo buffer starting at global t0 (dur + 2.5 s tail)."""
    bank = bank or tile_bank()
    out = np.zeros((ns(dur + 2.5), 2), np.float32)
    N = len(t_land)
    if N == 0:
        return out
    t_land, v_land = np.asarray(t_land, float), np.asarray(v_land, float)
    pan = np.broadcast_to(np.asarray(pan, float), (N,))
    names, ps = np.array(mats[0::2]), np.array(mats[1::2], float)
    mat = names[r.choice(len(names), N, p=ps / ps.sum())]
    hp_ = np.zeros(N) if heap_p is None else np.broadcast_to(np.asarray(heap_p, float), (N,))
    surf = np.where(r.random(N) < hp_, "heap", "floor")
    t, v, s, _, ids = bounce_events(r, t_land, v_land, surf)
    render_contacts(out, t0, bank, r, t, v, s, mat[ids], pan[ids], gain)
    fl = np.nonzero((surf == "floor") & (r.random(N) < skid_p))[0]            # skids across the marble
    if len(fl):
        bank.render(out, t0, t_land[fl] + r.uniform(0.002, 0.03, len(fl)), bank.pick(r, "skid", len(fl)),
                    gain * 0.22 * np.clip(v_land[fl] / V_REF, 0, 2) * r.uniform(0.3, 1.0, len(fl)), pan[fl])
    hj = np.nonzero(surf == "heap")[0]                                          # the heap shifts under the hit
    if len(hj) and jostle > 0:
        k = r.integers(1, 4, len(hj))
        src = np.repeat(hj, k)
        tj = t_land[src] + r.uniform(0.004, 0.08, len(src))
        vj = v_land[src] * r.uniform(0.04, 0.16, len(src))
        render_contacts(out, t0, bank, r, tj, vj, np.full(len(src), "heap"), mat[src],
                        np.clip(pan[src] + r.normal(0, 0.05, len(src)), -1, 1), gain * jostle)
    return out


def fall_plan(r, detach_t, heights, pans):
    """tesserae detaching at detach_t from wall heights (m): free fall -> first contact time and speed."""
    h = np.asarray(heights, float)
    return np.asarray(detach_t, float) + np.sqrt(2 * h / G), np.sqrt(2 * G * h), np.asarray(pans, float)


def sample_rate_curve(r, t0, t1, rate_fn, total):
    """draw `total` event times on [t0, t1) distributed by the (unnormalised) rate curve rate_fn(t)."""
    tg = np.linspace(t0, t1, 4000)
    w = np.clip(np.asarray(rate_fn(tg), float), 0, None)
    c = np.cumsum(w)
    c = c / c[-1]
    return np.sort(np.interp(r.random(int(total)), c, tg))


def local_rate(t, width=0.05):
    """events/s around each event time (sliding window of `width` s)."""
    t = np.asarray(t, float)
    if len(t) == 0:
        return t
    s = np.sort(t)
    return (np.searchsorted(s, t + width / 2) - np.searchsorted(s, t - width / 2)) / width


def counts_to_times(r, t0, fps, counts, pans=None, scale=1.0, cap=None):
    """per-frame event counts (scene physics) -> jittered event times within each frame (+ per-event pans)."""
    c = np.clip(np.nan_to_num(np.asarray(counts, float)) * scale, 0, None)
    k = np.floor(c).astype(np.int64) + (r.random(len(c)) < (c - np.floor(c)))
    if cap is not None and k.sum() > cap:
        k = np.floor(k * cap / k.sum()).astype(np.int64)
    fr = np.repeat(np.arange(len(c)), k)
    t = t0 + (fr + r.random(len(fr))) / fps
    p = np.zeros(len(fr)) if pans is None else np.asarray(pans, float)[fr]
    return t, p


def pour_bed(r, dur, rate, bright=1.0):
    """a flow of glass grains sliding and cascading: dense micro-contact noise, level ~ sqrt(flow rate).
    rate: per-sample flow (tiles/s), array of length ns(dur)."""
    n = ns(dur)
    rate = np.broadcast_to(np.asarray(rate, float), (n,))
    lvl = np.sqrt(np.clip(rate, 0, None) / max(1.0, rate.max()))
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        exc = L.poisson_clicks(r, n, np.clip(rate * 0.6, 0, 20000), 1.0)
        g = L.resonators(exc, r.uniform(2500, 9000, 5) * bright, r.uniform(0.002, 0.006, 5), [0.6] * 5)
        g = g + L.bp(exc, 1500, 11000) * 0.5
        hiss = L.svf(L.pink(n, r), 3500 * bright, 0.7, "bp") * 0.25
        out[:, ch] = (g * 0.4 + hiss) * lvl
    return out


def swarm_air(r, dur, level, up=None):
    """the air of a swarm of flying tesserae: soft broadband rush whose band rises with speed (level)."""
    n = ns(dur)
    level = np.broadcast_to(np.asarray(level, float), (n,))
    up = level if up is None else np.broadcast_to(np.asarray(up, float), (n,))
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        a = L.svf(L.pink(n, r), 500 + 4500 * np.clip(up, 0, 1.2), 0.9, "bp")
        flick = L.hp(L.poisson_clicks(r, n, 1500 * np.clip(level, 0, 1)), 5000) * 0.3
        out[:, ch] = (a + flick) * level
    return out


def flip_wave(r, t0, dur, rate_fn, pan_fn, face="screen", total=None, bank=None, gain=1.0, spread=0.25):
    """a wave of tesserae turning over (split-flap): Poisson flips at rate_fn(t_abs) /s around pan_fn(t_abs).
    face: 'screen' (slop), 'gold' or 'smalti'. -> stereo buffer starting at t0."""
    bank = bank or tile_bank()
    out = np.zeros((ns(dur + 0.3), 2), np.float32)
    tg = np.linspace(t0, t0 + dur, 2000)
    w = np.clip(np.asarray(rate_fn(tg), float), 0, None)
    total = int(total if total is not None else np.trapezoid(w, tg))
    if total <= 0:
        return out
    t = sample_rate_curve(r, t0, t0 + dur, rate_fn, total)
    p = np.clip(np.asarray(pan_fn(t), float) + r.normal(0, spread, total), -1, 1)
    dens = np.interp(t, tg, w)
    g = gain * r.uniform(0.35, 1.0, total) / np.sqrt(1 + dens / 150.0)     # incoherent sum stays sane
    bank.render(out, t0, t, bank.pick(r, ("flip", face), total), g, p)
    return out


def set_flurry(r, count=18, dur=0.08, mat="gold", bank=None, vel=0.55):
    """a letter/figure of tesserae setting itself: `count` settings into the mortar within `dur` (a zipper of
    dry ticks over wet 'tups'). mono Snd, sync = first contact."""
    bank = bank or tile_bank()
    out = np.zeros((ns(dur + 0.08), 2), np.float32)
    t = np.sort(r.uniform(0, dur, count) ** 1.2 / dur ** 0.2)
    t[0] = 0.0
    v = V_REF * vel * r.uniform(0.6, 1.1, count)
    render_contacts(out, 0.0, bank, r, t, v, np.full(count, "mortar"), mat, 0.0)
    render_contacts(out, 0.0, bank, r, t + r.uniform(0.0005, 0.004, count), v * 0.8, np.full(count, "wall"), mat, 0.0,
                    0.9)
    return Snd(L.fade(L.norm(out.mean(1), 1.0), 0.0001, 0.02), 0.0)


def glint(r, bright=1.0, dur=0.9):
    """light catching gold: a tiny, very high glass ping on D / A / E (very soft; 'the sound of light')."""
    m = int(r.choice([86, 93, 88, 98, 100, 105])) + (12 if bright > 1.2 else 0)
    x = L.chime(r, float(midi_hz(m)), dur, (1.0, 2.76), dur * 0.45 * r.uniform(0.7, 1.2), 2.5, 0.8, 0.02)
    return Snd(L.fade(L.norm(x, 1.0), 0.002, 0.2), 0.0)


def tiles_jump(r, n_tiles=2500, dur=0.45, speed=45.0, extent=6.0, pan0=0.0, bank=None):
    """the shock of a blow runs through the wall: every set tessera hops off its bed and re-seats.
    A radial front (speed m/s) from the blow; each tile: a tiny lift tick then its landing on the wall
    (+ one small re-bounce). -> stereo, sync = 0 (the blow)."""
    bank = bank or tile_bank()
    out = np.zeros((ns(dur + 0.5), 2), np.float32)
    d = extent * np.sqrt(r.random(n_tiles))
    ang = r.uniform(0, TAU, n_tiles)
    x = d * np.cos(ang)
    tf = d / speed
    air = r.uniform(0.02, 0.09, n_tiles) * np.exp(-d / (extent * 0.8))
    v = G * air / 2 * 20 + 0.8
    pan = np.clip(pan0 + x / extent * 0.9, -1, 1)
    t_up = tf + r.uniform(0, 0.004, n_tiles)
    render_contacts(out, 0.0, bank, r, t_up, v * 0.3, np.full(n_tiles, "wall"), "gold", pan, 0.5)
    t_dn = t_up + air
    t, vv, s, _, ids = bounce_events(r, t_dn, v, np.full(n_tiles, "wall"), kmax=2, vmin=0.5, settle=False)
    render_contacts(out, 0.0, bank, r, t, vv, s, np.where(r.random(n_tiles) < 0.6, "gold", "smalti")[ids], pan[ids])
    return Snd(out, 0.0)


def quadtree(r, gen_times, n0=4, cap=5000, pans=(-0.9, 0.9), bank=None, spread=None):
    """recursive splitting: generation k (at gen_times[k]) splits 4^k pieces -> 4^(k+1); every split is a cleave
    tick whose modes sit an octave higher per generation (f ~ 1/size) and 6 dB softer per piece.
    -> (stereo buffer starting at gen_times[0], per-generation event counts)."""
    bank = bank or tile_bank()
    t0 = gen_times[0]
    dur = gen_times[-1] - t0 + 1.0
    out = np.zeros((ns(dur), 2), np.float32)
    counts = []
    for k, tk in enumerate(gen_times):
        cnt = int(min(cap, n0 * 4 ** k))
        nxt = gen_times[k + 1] if k + 1 < len(gen_times) else tk + 0.35
        sp = spread if spread is not None else max(0.03, (nxt - tk) * 0.9)
        t = tk + r.random(cnt) ** 1.6 * sp
        g = 0.5 ** k * np.sqrt((n0 * 4 ** k) / cnt) * r.uniform(0.4, 1.0, cnt)
        p = r.uniform(pans[0], pans[1], cnt)
        bank.render(out, t0, t, bank.pick(r, ("split", min(k, 8)), cnt), g, p)
        counts.append(cnt)
    return out, counts


def dust_hiss(r, dur, level, swirl_hz=0.35):
    """glittering dust: the ultrasonic splits' audible residue - a very high hiss (9-16 kHz) with sparkle
    grains, swirling around the listener (pan rotates)."""
    n = ns(dur)
    t = np.arange(n) / SR
    level = np.broadcast_to(np.asarray(level, float), (n,))
    src = L.hp(L.pink(n, r), 9000) * 0.6 + L.hp(L.poisson_clicks(r, n, 2500), 7000) * 0.5
    src = L.lp(src, 16000, 2) * level
    return L.pan2(L.f32(src), 0.8 * np.sin(TAU * swirl_hz * t + TAU * 0.3 * np.sin(TAU * 0.11 * t)))


def split_cascade_bed(r, dur, level):
    """dense subdividing (tiny_split): a crackle of micro-cleaves (rate follows level)."""
    n = ns(dur)
    level = np.broadcast_to(np.asarray(level, float), (n,))
    exc = L.poisson_clicks(r, n, 60 + 1500 * level, 1.0)
    x = L.resonators(exc, r.uniform(4000, 12000, 5), [0.003] * 5, [0.5] * 5) + L.hp(exc, 3000) * 0.6
    return L.f32(x * level)


def rise_layers(r, t0, t_conv, lift_t, lift_pan, land_t, land_pan, conv_n=1500, bank=None, verb_ir=None):
    """THE UNBABELING: the fallen tesserae rise and set themselves again.
    (a) the avalanche UN-HAPPENS: every lifting tile's bounce train is generated forward (floor/heap contacts,
        settle rattle...) and played time-reversed so it swells into the instant the tile leaves the floor
        (with its reverb reversed too: pre-verb),
    (b) the flight: the air of the swarm,
    (c) the setting: each tile set into the wall's mortar (dry tick over a wet tup, forward time),
    (d) the convergence: the last `conv_n` tiles set together in one wave at t_conv (a coherent massed click).
    -> dict of stereo buffers all starting at t0: rev, air, set, conv."""
    bank = bank or tile_bank()
    t_end = max(t_conv + 3.0, (max(lift_t) if len(lift_t) else t0) + 0.5)
    dur = t_end - t0
    n = ns(dur)
    # (a) reversed avalanche: build forward in mirrored time T' = t_end - t
    rev = np.zeros((n, 2), np.float32)
    if len(lift_t):
        lt = np.asarray(lift_t, float)
        m = t_end - lt                                   # mirrored first-contact time (relative to t_end)
        v = r.uniform(3.0, 9.0, len(lt))
        surf = np.where(r.random(len(lt)) < 0.7, "heap", "floor")
        tt_, vv, ss, _, ids = bounce_events(r, m, v, surf)
        fwd = np.zeros((n, 2), np.float32)
        render_contacts(fwd, 0.0, bank, r, tt_, vv, ss, np.where(r.random(len(lt)) < 0.6, "gold", "smalti")[ids],
                        np.asarray(lift_pan, float)[ids], 1.0)
        if verb_ir is not None:
            wet = L.convolve(fwd, verb_ir)[:n] * 0.6
            fwd = fwd + wet
        rev = L.f32(fwd[::-1])
    # (b) flight air
    tg = t0 + np.arange(n) / SR
    if len(lift_t) and len(land_t):
        inflight = (np.searchsorted(np.sort(lift_t), tg) - np.searchsorted(np.sort(land_t), tg)).clip(0, None)
        lv = np.sqrt(inflight / max(1, inflight.max()))
    else:
        lv = np.zeros(n)
    air = swarm_air(r, dur, L.lp(L.f32(lv), 4, 1), None)
    # (c) setting on the wall
    st = np.zeros((n, 2), np.float32)
    if len(land_t):
        lt = np.asarray(land_t, float) - t0
        k = len(lt)
        v = r.uniform(2.0, 6.0, k)
        mat = np.where(r.random(k) < 0.6, "gold", "smalti")
        gk = 1.0 / np.sqrt(1 + local_rate(lt) / 200.0)            # an incoherent crowd of ticks stays sane
        render_contacts(st, 0.0, bank, r, lt, v, np.full(k, "mortar"), mat, land_pan, gk)
        render_contacts(st, 0.0, bank, r, lt + r.uniform(0.0005, 0.004, k), v * 0.8, np.full(k, "wall"), mat, land_pan,
                        0.9 * gk)
    # (d) convergence: the last wave sets at once (±12 ms) - thousands of dry ticks summing into one
    cv = np.zeros((n, 2), np.float32)
    tc = (t_conv - t0) + r.normal(0, 0.006, conv_n)
    pc = r.uniform(-1, 1, conv_n)
    vc = r.uniform(3.0, 7.0, conv_n)
    matc = np.where(r.random(conv_n) < 0.7, "gold", "smalti")
    render_contacts(cv, 0.0, bank, r, tc, vc, np.full(conv_n, "mortar"), matc, pc, 1.0)
    render_contacts(cv, 0.0, bank, r, tc + r.uniform(0.0005, 0.003, conv_n), vc * 0.8, np.full(conv_n, "wall"), matc,
                    pc, 0.9)
    return dict(rev=rev, air=air, set=st, conv=cv)


# ================================================================================================
# designed single events: setting, cracking, splitting
# ================================================================================================
def tile_set(r, mat="gold", vel=0.6, bank=None):
    """ONE tessera pressed into the setting bed: a dry tick against its neighbours over a wet 'tup'. sync 0."""
    bank = bank or tile_bank()
    out = np.zeros((ns(0.08), 2), np.float32)
    render_contacts(out, 0.0, bank, r, [0.0], [V_REF * vel], ["mortar"], mat, 0.0)
    render_contacts(out, 0.0, bank, r, [r.uniform(0.0006, 0.003)], [V_REF * vel * 0.8], ["wall"], mat, 0.0, 1.0)
    return Snd(L.fade(L.norm(out.mean(1), 1.0), 0.0001, 0.01), 0.0)


def bed_crack(r, dur=0.4, level=0, final=True):
    """a crack running through the setting bed between the tesserae: accelerating micro-fractures through lime
    mortar (dull crunch), the tile edges ticking apart as the front passes (glassy), a grinding undertone and a
    final split. level = recursion depth: each generation is shorter, higher, lighter. sync = the final split."""
    k = 1.32 ** level
    dur = max(0.06, dur / np.sqrt(k))
    n = ns(dur)
    t = np.arange(n) / SR
    exc = L.poisson_clicks(r, n, (50 + 1400 * (t / dur) ** 2) * min(k, 4), np.clip(0.3 + t / dur, 0, 1))
    mortar = L.resonators(exc, r.uniform(450, 2400, 6) * k ** 0.6, r.uniform(0.003, 0.01, 6), r.uniform(0.3, 1, 6))
    ticks = L.poisson_clicks(r, n, 25 * min(k, 6) * (0.3 + t / dur), 1.0)
    glass = L.resonators(ticks, r.uniform(3500, 11000, 5) * min(k, 1.6) ** 0.3, r.uniform(0.004, 0.012, 5), [0.6] * 5)
    grind = L.svf(L.brown(n, r), 230 * k ** 0.5, 2.5, "bp") * L.env_pts(n, [(0, 0), (dur * 0.6, 0.5), (dur, 0.9)])
    x = mortar * 0.55 + glass * 0.5 + L.hp(exc, 2500) * 0.6 + grind * 0.35 / k ** 0.5
    if final:
        tail = np.zeros(ns(0.35), np.float32)
        sp = split_grain(r, 650.0 * 2 ** min(level, 5), level)
        tail[: len(sp)] += sp * 1.1
        th = L.thud(r, 90 * k ** 0.5, 0.18, 0.8, 900)
        tail[: len(th)] += th[: len(tail)] * 0.5 / k ** 0.5
        x = np.concatenate([x, tail])
    return Snd(L.fade(L.norm(x, 1.0), 0.001, 0.03), dur if final else dur * 0.9)


def halo_split(r):
    """a gold halo (a ring of gold tesserae) cracks in two: bright crack + the two halves ringing apart as two
    slightly detuned gold rings (on A/E), each drifting in the stereo field. -> stereo Snd, sync 0."""
    n = ns(1.6)
    out = np.zeros((n, 2), np.float32)
    cr = bed_crack(r, 0.08, 3, True).x
    place_mono(out, cr, 0, 0.8, 0.0)
    for side, m in ((-0.5, 93), (0.5, 88)):
        c = L.chime(r, float(midi_hz(m + 12)) * r.uniform(0.997, 1.003), 1.5, (1.0, 2.32, 4.25), 0.9, 3.0, 0.7, 0.05)
        place_mono(out, c, ns(0.004), 0.35, side)
    return Snd(L.fade(out, 0.0001, 0.2), ns(0.08) / SR)


def drift_clinks(r, dur=3.0, count=14):
    """fragments drifting apart in a void (no gravity, no floor): sparse gentle clinks as pieces touch and part,
    a soft grinding as the first ones separate. stereo."""
    bank = tile_bank()
    n = ns(dur + 0.5)
    out = np.zeros((n, 2), np.float32)
    t = np.sort(r.uniform(0, dur, count) ** 1.4 / dur ** 0.4)
    render_contacts(out, 0.0, bank, r, t, r.uniform(0.8, 3.0, count), np.full(count, "wall"),
                    np.where(r.random(count) < 0.5, "gold", "stone"), r.uniform(-0.9, 0.9, count))
    g = L.svf(L.brown(ns(0.8), r), 300, 2.0, "bp") * L.env_pts(ns(0.8), [(0, 0), (0.1, 1), (0.8, 0)])
    place_mono(out, L.f32(g) * 0.4, 0, 1.0, 0.0)
    return L.fade(out, 0.002, 0.2)


# ================================================================================================
# the candle
# ================================================================================================
def match_strike(r):
    """a match struck on the striker: scrape (stick-slip crackle of the head on the grit) -> ignition flare
    (broadband 'pfshh', low-pass falling as the flash dies, sulphur sparks, the small whump of gas) -> the match
    flame's soft roar. sync = the flare."""
    pre = r.uniform(0.075, 0.095)
    dur = pre + 1.8
    n = ns(dur)
    x = np.zeros(n, np.float32)
    m = ns(pre)
    u = np.arange(m) / m
    ss = L.stick_slip(r, pre, 700 + 1900 * u, 0.5, np.clip(u * 3, 0, 1))
    scr = L.resonators(ss, [2300, 3700, 5900, 8200], [0.003] * 4, [1, 0.7, 0.5, 0.3])
    scr = scr + L.bp(L.white(m, r), 1500, 8000) * L.f32(u * 0.35)
    x[:m] += scr * 0.45
    k = n - m
    tf = np.arange(k) / SR
    flare = L.svf(L.white(k, r), 1100 + 8500 * np.exp(-tf / 0.05), 0.8, "lp") * L.env_ad(k, 0.003, 0.32)
    sparks = L.hp(L.poisson_clicks(r, k, 3500 * np.exp(-tf / 0.07) + 30 * np.exp(-tf / 0.5)), 2500) * 0.45
    whump = L.lp(L.white(k, r), 280) * L.env_ad(k, 0.004, 0.18) * 1.1
    burn = L.svf(L.brown(k, r), 160 + 50 * L.smooth_noise(k, r, 3, 2), 0.9, "lp")
    burn = burn * L.env_pts(k, [(0, 0), (0.1, 0.4), (0.6, 0.18), (dur - pre, 0.0)]) * 0.35
    x[m:] += flare + sparks + whump + burn
    return Snd(L.fade(L.norm(x, 1.0), 0.002, 0.3), pre)


def wick_catch(r):
    """the candle wick takes the flame: a soft rising 'fwp' of the flame growing + a little wax sputter. sync 0."""
    n = ns(0.6)
    t = np.arange(n) / SR
    x = L.svf(L.brown(n, r), 120 + 380 * np.exp(-t / 0.12), 0.8, "lp") * L.env_pts(n, [(0, 0), (0.05, 1), (0.6, 0)])
    sp = L.hp(L.poisson_clicks(r, n, 30 * np.exp(-t / 0.2)), 2000) * 0.3
    return Snd(L.fade(L.norm(x + sp, 1.0), 0.003, 0.1), 0.0)


def flame_bed(r, dur, disturb=0.3):
    """a candle flame close by: the buoyant plume's soft burble; when a draught disturbs it the flame puffs at
    the ~11 Hz flicker frequency of a diffusion flame; now and then the wax sputters. mono, unit-ish level."""
    n = ns(dur)
    d = np.clip(0.25 + 0.75 * L.smooth_noise(n, r, 0.22, 2), 0, 1) ** 2 * disturb
    ff = 10.5 + 1.2 * L.smooth_noise(n, r, 0.5, 1)
    puff = 0.5 + 0.5 * np.sin(TAU * np.cumsum(ff) / SR)
    body = L.svf(L.brown(n, r), 55 + 110 * d, 1.1, "lp")
    x = body * (0.35 + 0.65 * d * puff) + L.svf(L.pink(n, r), 700, 0.7, "bp") * d * puff * 0.08
    sput = L.hp(L.poisson_clicks(r, n, 0.3), 1800) * 0.2
    return L.f32(x + sput)


def snuff(r):
    """the brass snuffer comes down over the flame: the cup's small damped 'tnk', the flame choked (a short
    falling 'ffh'), the wick's sizzle, then a thread of smoke (a faint hiss dying away). sync = the cup lands."""
    n = ns(2.4)
    t = np.arange(n) / SR
    x = np.zeros(n, np.float32)
    cup = L.ring([1760.0 * q for q in (1.0, 1.53, 2.37, 3.14)], [0.09, 0.06, 0.04, 0.03], [1, 0.5, 0.4, 0.2], 0.3, r)
    x[: len(cup)] += cup * 0.55
    tk = tile_click(r, "stone", "wall", 0.4, 0.8)
    x[: len(tk)] += tk * 0.35
    ffh = L.svf(L.white(n, r), 2200 * np.exp(-t / 0.05) + 300, 0.8, "lp") * L.env_ad(n, 0.004, 0.16)
    x += ffh * 0.6
    sz = L.hp(L.poisson_clicks(r, n, 400 * np.exp(-t / 0.12)), 3000) * 0.25
    smoke = L.hp(L.pink(n, r), 4500) * L.env_pts(n, [(0, 0), (0.15, 0.0), (0.4, 0.012), (2.4, 0.0)])
    x += sz + smoke
    return Snd(L.fade(L.norm(x, 1.0), 0.0005, 0.3), 0.0)


# ================================================================================================
# stone and space
# ================================================================================================
def stone_step(r, weight=1.0, shoe="sandal"):
    """a footfall on the basilica's marble floor. shoe: 'sandal' (soft leather slap + scuff), 'boot' (hard heel
    click; the lamellar jingle is separate), 'heap' (crunching into a heap of fallen tesserae). sync 0."""
    bank = tile_bank()
    n = ns(0.5)
    x = np.zeros(n, np.float32)
    th = L.thud(r, 80 * r.uniform(0.85, 1.15), 0.2, 0.7, 500)
    x[: len(th)] += th * 0.8 * weight
    if shoe == "boot":
        c = tile_click(r, "stone", "floor", 0.9, 1.6)
        x[: len(c)] += c * 0.8
    elif shoe == "sandal":
        m = ns(0.05)
        sl = L.bp(L.white(m, r), 500, 3500) * L.env_ad(m, 0.001, 0.03)
        x[:m] += sl * 0.6
        c = tile_click(r, "stone", "floor", 0.35, 1.4)
        x[: len(c)] += c * 0.25
    else:  # heap: the foot sinks into loose glass cubes
        out = np.zeros((n, 2), np.float32)
        k = int(r.integers(25, 45))
        tt_ = np.sort(r.exponential(0.035, k))
        render_contacts(out, 0.0, bank, r, tt_, r.uniform(0.5, 3.5, k) * np.exp(-tt_ / 0.08), np.full(k, "heap"),
                        np.where(r.random(k) < 0.5, "gold", "smalti"), 0.0)
        x += out.mean(1) * 1.4
    m = ns(0.09)
    sc = L.bp(L.white(m, r), 1200, 6000) * L.env_pts(m, [(0, 0), (0.02, 1), (0.09, 0)])
    x[ns(0.015):ns(0.015) + m] += sc * 0.15
    return Snd(L.fade(L.norm(x, 1.0), 0.0003, 0.05), 0.0)


def link_click(r, size=1.0, t60=(0.012, 0.045)):
    """one small iron plate / chain link knocking another: a click with a short, damped, inharmonic ring."""
    k = 3
    x = L.ring(r.uniform(2200, 7500, k) / size, r.uniform(t60[0], t60[1], k), r.uniform(0.4, 1.0, k), 0.09, r)
    c = ns(0.001)
    x[:c] += L.hp(L.white(c, r), 2500) * 0.8
    return L.fade(L.norm(x, 1.0), 0.0001, 0.01)


_PLATES = {}


def lamellar(r, dur=0.35, intensity=1.0):
    """lamellar armour: hundreds of small laced iron plates chinking as the body moves - each contact a random
    plate (short, lacing-damped ring), the rate swelling with the motion - plus the leather lacing's creak."""
    if "p" not in _PLATES:
        rr = R(9191)
        _PLATES["p"] = [link_click(rr, rr.uniform(0.8, 1.3)) for _ in range(40)]
    n = ns(dur + 0.1)
    x = np.zeros(n, np.float32)
    rate = lambda u: np.interp(u, [0, 0.04, dur * 0.45, dur], [0, 1, 0.55, 0])  # noqa: E731
    cnt = int((20 + 110 * intensity) * dur)
    ts = sample_rate_curve(r, 0.0, dur, rate, cnt)
    for t in ts:
        p = _PLATES["p"][r.integers(len(_PLATES["p"]))]
        i = ns(t)
        x[i:i + len(p)] += p[: n - i] * r.uniform(0.3, 1.0) * float(rate(t)) ** 0.5
    m = ns(dur)
    x[:m] += L.svf(L.brown(m, r), 380, 1.5, "bp") * L.f32(rate(np.arange(m) / SR)) * 0.25
    return Snd(L.fade(L.norm(x, 1.0), 0.001, 0.05), 0.0)


def nvg_drop(r):
    """night-vision goggles lowered: a small servo (DC motor whine + gear-mesh sidebands) runs, the mount's
    detent clicks home, the phosphor tube's high whine rises. sync = the detent click."""
    run = r.uniform(0.16, 0.26)
    n = ns(run + 0.9)
    t = np.arange(n) / SR
    x = np.zeros(n, np.float64)
    m = ns(run)
    tm = t[:m]
    fm = 1100 + 500 * np.sin(np.pi * tm / run)
    mot = np.sin(TAU * np.cumsum(fm) / SR) * (1 + 0.5 * np.sin(TAU * np.cumsum(fm / 7.0) / SR))
    x[:m] += mot * np.sin(np.pi * tm / run) ** 0.5 * 0.25
    x[:m] += L.bp(L.white(m, r), 2000, 6000) * 0.05
    c = L.impact(r, "tin", 0.45, 1.0, 0.08)
    x[m:m + len(c)] += c[: n - m] * 0.9
    x[m:m + ns(0.003)] += L.hp(L.white(ns(0.003), r), 3000) * 0.6
    tw = t[m:] - t[m]
    wh = np.sin(TAU * np.cumsum(7800 + 1600 * (1 - np.exp(-tw / 0.2))) / SR) * np.clip(tw / 0.1, 0, 1) * np.exp(-tw / 0.5)
    x[m:] += wh * 0.06
    return Snd(L.fade(L.norm(L.f32(x), 1.0), 0.003, 0.05), run)


def telescope_section(r, k=0, of=3):
    """one section of a telescoping flag-lance sliding out and locking: friction slide, then the detent clack
    (thinner sections ring higher). sync = the clack."""
    sl = 0.07
    n = ns(sl + 0.25)
    x = np.zeros(n, np.float32)
    m = ns(sl)
    x[:m] += L.svf(L.white(m, r), 2500 + 900 * k, 2.5, "bp") * L.f32(np.linspace(0.2, 0.9, m)) * 0.25
    c = L.impact(r, "metal", 0.75 - 0.12 * k, 1.0, 0.2)
    x[m:m + len(c)] += c[: n - m]
    t = tile_click(r, "stone", "wall", 0.9, 1.3)
    x[m:m + len(t)] += t[: n - m] * 0.5
    return Snd(L.fade(L.norm(x, 1.0), 0.002, 0.05), sl)


def massed_steps(r, n=11, spread=0.014, pan=(-0.9, 0.9), shoe="boot", armour=True):
    """n soldiers stamp as one: n footfalls within +-spread s across the frieze, + a lamellar rattle burst.
    stereo, sync 0."""
    out = np.zeros((ns(0.8), 2), np.float32)
    for k in range(n):
        s = stone_step(r, r.uniform(0.8, 1.1), shoe).x
        place_mono(out, s, ns(0.02 + r.normal(0, spread)), r.uniform(0.6, 1.0), pan[0] + (pan[1] - pan[0]) * k / max(1, n - 1))
    if armour:
        for k in range(4):
            a = lamellar(r, 0.3, 1.0).x
            place_mono(out, a, ns(0.02 + r.uniform(0, 0.02)), 0.5, r.uniform(*pan))
    return Snd(L.fade(L.norm(out, 1.0), 0.0005, 0.05), 0.02)


def nvg_flare(r, voices=11):
    """every night-vision tube flares at once: a chord of high phosphor whines swelling and settling + a 'fsst'."""
    n = ns(1.0)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in range(voices):
        f = 7300 + 1900 * r.random()
        x += np.sin(TAU * f * (1 + 0.01 * np.exp(-t / 0.1)) * t + r.uniform(0, TAU)) * r.uniform(0.5, 1.0)
    x *= L.env_pts(n, [(0, 0), (0.03, 1), (0.25, 0.5), (1.0, 0)])
    x = x / voices + L.hp(L.white(n, r), 5000) * L.env_ad(n, 0.002, 0.08) * 0.8
    return Snd(L.fade(L.norm(L.f32(x), 1.0), 0.001, 0.05), 0.0)


def wall_stress(r, dur=0.6):
    """a mosaic wall about to blow: light leaking through the joints - the bed creaks and ticks, the ticks
    accelerating toward failure. stereo, sync 0."""
    bank = tile_bank()
    n = ns(dur + 0.3)
    out = np.zeros((n, 2), np.float32)
    k = 70
    ts = sample_rate_curve(r, 0.0, dur, lambda u: (u / dur) ** 2.5 + 0.05, k)
    render_contacts(out, 0.0, bank, r, ts, r.uniform(0.6, 2.5, k), np.full(k, "wall"),
                    np.where(r.random(k) < 0.5, "gold", "smalti"), r.uniform(-1, 1, k))
    cr = L.creak(r, dur, 6, 22, (48, 71, 105, 160), 0.3, 0.7).x
    place_mono(out, L.f32(cr) * 0.5, 0, 1.0, 0.0)
    return Snd(L.fade(out, 0.01, 0.05), 0.0)


def flag_pop(r, strength=0.4):
    """a small Flag springs open in its niche: a short swish and a soft cloth pop (a little snap). mono, sync 0."""
    import foley_flutter as FF
    n = ns(0.3)
    t = np.arange(n) / SR
    bell = np.exp(-((t - 0.05) / 0.025) ** 2)
    x = L.svf(L.pink(n, r), 700 + 2500 * bell, 1.2, "bp") * bell * 0.6
    c = FF.snap_crack(r, float(np.clip(strength, 0.15, 0.8)), 1.0)
    i = ns(0.06)
    x[i:i + len(c)] += c[: n - i] * 0.7
    return Snd(L.fade(L.norm(L.f32(x), 1.0), 0.002, 0.05), 0.06)


def win_norm(x, target_db, win=1.0):
    """scale x so that its loudest `win`-second window has RMS = target_db (dBFS)."""
    m = np.asarray(x, np.float64)
    m = (m ** 2).mean(1) if m.ndim == 2 else m ** 2
    w = max(1, ns(win))
    if len(m) <= w:
        e = m.mean()
    else:
        c = np.concatenate([[0.0], np.cumsum(m)])
        e = ((c[w:] - c[:-w]) / w).max()
    return L.f32(np.asarray(x) * (db(target_db) / np.sqrt(e + 1e-20)))


def door_far(r):
    """a heavy door closing far off in the basilica: iron latch, oak slam, frame rattle; darkened by distance.
    sync = the slam."""
    n = ns(1.2)
    x = np.zeros(n, np.float32)
    pre = ns(0.09)
    lt = L.impact(r, "metal", 0.7, 0.6, 0.2)
    x[: len(lt)] += lt * 0.3
    th = L.thud(r, 58, 0.7, 1.3, 350)
    x[pre:pre + len(th)] += th[: n - pre]
    w = L.impact(r, "wood", 2.5, 1.0, 0.3)
    x[pre:pre + len(w)] += w[: n - pre] * 0.6
    rt = L.impact(r, "metal", 0.9, 0.4, 0.3)
    i = pre + ns(0.03)
    x[i:i + len(rt)] += rt[: n - i] * 0.25
    x = L.lp(x, 1400, 2)
    return Snd(L.fade(L.norm(x, 1.0), 0.002, 0.2), pre / SR)


def pigeon_coo(r):
    """a rock dove high in the vault: 'hoo-rroo-oo' - a low tonal coo (inflated-crop resonance) with the rolled
    'rr' (30-40 Hz amplitude trill) in the middle syllable. sync 0."""
    segs = [(0.0, 0.3, 470, 520, 0.0), (0.38, 0.55, 520, 480, 1.0), (1.0, 0.32, 490, 420, 0.0)]
    n = ns(1.45)
    x = np.zeros(n)
    base = r.uniform(0.93, 1.07)
    for s, d, f0, f1, roll in segs:
        m = ns(d)
        tm = np.arange(m) / SR
        f = np.linspace(f0, f1, m) * base * (1 + 0.01 * np.sin(TAU * 5 * tm))
        ph = TAU * np.cumsum(f) / SR
        v = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
        e = np.sin(np.pi * tm / d) ** 1.2
        if roll:
            e *= 0.55 + 0.45 * np.sin(TAU * 34 * tm)
        i = ns(s)
        x[i:i + m] += v * e
    x = L.lp(L.f32(x), 1500, 2)
    return Snd(L.fade(L.norm(x, 1.0), 0.005, 0.05), 0.0)


def pigeon_flap(r, beats=7):
    """a pigeon leaving a ledge high up: the wing-clap take-off then ~9 Hz beats fading away. sync 0."""
    n = ns(beats / 8.5 + 0.4)
    x = np.zeros(n, np.float32)
    for k in range(beats):
        m = ns(0.06)
        b = L.bp(L.white(m, r), 400, 3500) * L.env_ad(m, 0.002, 0.04)
        if k < 2:
            b = b + L.hp(L.white(m, r), 1500) * L.env_ad(m, 0.0005, 0.01) * 1.5
        i = ns(k / 8.5 + r.uniform(0, 0.01))
        x[i:i + m] += b[: n - i] * (0.9 ** k)
    return Snd(L.fade(L.norm(x, 1.0), 0.001, 0.05), 0.0)


def basilica_bed(r, dur):
    """the basilica at night: the stone mass's sub-sonic breathing (brown < 90 Hz with slow pressure swells from
    the wind outside), the air of the vast volume (very faint pink 150-900 Hz), a thread of high air. stereo,
    unit RMS."""
    n = ns(dur)
    sw = 0.6 + 0.4 * L.smooth_noise(n, r, 0.05, 2)
    cl, cm = L.brown(n, r), L.pink(n, r)
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        low = L.svf(0.6 * cl + 0.8 * L.brown(n, r), 55 + 25 * sw, 0.7, "lp") * sw
        mid = L.bp(0.5 * cm + 0.8 * L.pink(n, r), 150, 900) * 0.16 * (0.8 + 0.2 * sw)
        hi = L.hp(L.pink(n, r), 5000) * 0.015
        out[:, ch] = low + mid + hi
    return rms_norm(out)


def high_wind(r, dur, speed=None):
    """wind high up in the clerestory and dome windows: thin edge-tone whistles and soft gusts, no low end."""
    w = L.wind(r, dur, level=0.35, gust=0.7, gust_rate=0.09, whistle=0.8, bright=0.35, speed=speed)
    return rms_norm(L.lp(L.hp(w, 220, 2), 6000, 1))


# ================================================================================================
# the edict
# ================================================================================================
def stylus(r, dur=0.8, strokes=4):
    """a golden stylus signing on vellum: friction hiss gated by the pen's speed, the nib's stick-slip squeaks,
    the faint ring of the gold stylus, a tap at the first touch. sync 0."""
    n = ns(dur)
    t = np.arange(n) / SR
    v = np.zeros(n)
    for c in np.sort(r.uniform(0.04, dur - 0.04, strokes)):
        v += np.exp(-((t - c) / (dur / strokes * 0.33)) ** 2) * r.uniform(0.6, 1.0)
    v = np.clip(v, 0, 1)
    fr = L.bp(L.white(n, r), 2800, 9500) * v ** 1.3
    grit = L.poisson_clicks(r, n, 150 + 1200 * v, v)                   # the nib catching on the vellum's grain
    sc = L.resonators(grit, r.uniform(2500, 7000, 4), [0.0015] * 4, [1, 0.6, 0.4, 0.3])
    ss = L.stick_slip(r, dur, 300 + 900 * v, 1.5, v * 0.5)              # a little squeak of the gold nib
    ring = L.resonators(ss, [3520.0, 5274.0], [0.04, 0.025], [0.2, 0.12])
    x = fr * 0.7 + sc * 0.45 + ring * 0.2
    tap = L.impact(r, "tin", 0.3, 0.6, 0.04)
    x[: len(tap)] += tap * 1.2
    return Snd(L.fade(L.norm(x, 1.0), 0.002, 0.03), 0.0)


def bulla_slam(r):
    """the imperial seal: the boulloterion's iron jaws slam the gold bulla onto the codex: iron clank, the soft
    metal squashed, the codex and table thud, a sub on D. sync = impact."""
    n = ns(2.5)
    t = np.arange(n) / SR
    x = np.zeros(n, np.float32)
    cl = L.ring([310, 820, 1375, 1940, 2860], [0.5, 0.3, 0.2, 0.14, 0.09], [1, 0.7, 0.5, 0.35, 0.2], 0.8, r)
    x[: len(cl)] += cl * 0.6
    sq = L.lp(L.white(ns(0.03), r), 1100) * L.env_ad(ns(0.03), 0.0005, 0.02)
    x[: len(sq)] += sq * 1.2
    th = L.thud(r, 88, 0.5, 1.4, 700)
    x[: len(th)] += th * 1.0
    sub = np.sin(TAU * np.cumsum(D1_HZ * 2 * (1 + 0.5 * np.exp(-t / 0.03))) / SR) * L.env_ad(n, 0.004, 1.2)
    x += L.f32(sub) * 0.7
    x[:ns(0.002)] += L.hp(L.white(ns(0.002), r), 2000) * 0.8
    return Snd(L.fade(L.norm(x, 1.0), 0.0002, 0.3), 0.0)


def coin_drop(r):
    """a gold solidus dropped into a paw: the rim's bright tick, a short damped ring (gold is soft; the pad
    damps it), the coin rocking flat (two tiny ticks). sync = contact."""
    n = ns(0.5)
    x = np.zeros(n, np.float32)
    rg = L.ring([5870.0, 9390.0, 13300.0], [0.06, 0.04, 0.025], [1, 0.6, 0.35], 0.2, r)
    x[: len(rg)] += rg * 0.5
    c = L.impact(r, "metal", 0.35, 1.0, 0.05)
    x[: len(c)] += c * 0.6
    pad = L.lp(L.white(ns(0.02), r), 900) * L.env_ad(ns(0.02), 0.001, 0.015)
    x[: len(pad)] += pad * 0.5
    for k, dt in enumerate((0.045, 0.072)):
        c2 = L.impact(r, "metal", 0.3, 0.4, 0.04)
        i = ns(dt)
        x[i:i + len(c2)] += c2[: n - i] * (0.25 / (k + 1))
    return Snd(L.fade(L.norm(x, 1.0), 0.0002, 0.05), 0.0)


def ka_ching(r):
    """the joke: 'ka' (a lever clunk + ratchet) then the 'ching' of a small bell tuned D7 / A7 (Composer-2's
    bell+glock sting doubles it) + a jingle of coins. sync = the ching."""
    dur = 2.0
    n = ns(dur)
    x = np.zeros(n, np.float32)
    k1 = L.impact(r, "tin", 1.4, 1.0)
    x[: len(k1)] += k1 * 0.45
    rat = L.resonators(L.stick_slip(r, 0.07, 180, 0.1), [1400, 2600, 4100], [0.01] * 3, [1, 0.6, 0.4])
    x[ns(0.02):ns(0.02) + len(rat)] += rat * 0.35
    tb = ns(0.12)
    bell = L.ring([2349.3 * q for q in (1.0, 2.01, 2.63, 3.95)], [1.1, 0.7, 0.5, 0.3], [1.0, 0.45, 0.4, 0.2],
                  dur - 0.12, r)
    bell2 = L.ring([3520.0 * q for q in (1.0, 2.63)], [0.8, 0.4], [0.5, 0.2], dur - 0.12, r)
    x[tb:tb + len(bell)] += (bell + bell2) * 0.5
    st = L.impact(r, "metal", 0.5, 1.0, 0.1)
    x[tb:tb + len(st)] += st
    for k in range(5):
        c = L.impact(r, "metal", 0.4, 0.5, 0.4)
        i = tb + ns(0.03 + 0.05 * k + r.uniform(0, 0.03))
        x[i:i + len(c)] += c[: n - i] * 0.25 * 0.8 ** k
    return Snd(L.fade(L.norm(x, 1.0), 0.0005, 0.3), 0.12)


# ================================================================================================
# the Hypermind (ophanim-polycandelon)
# ================================================================================================
def eye_open(r, size=1.0):
    """an eye of tesserae opening: the lid tiles slide apart over each other (dense grinding micro-contacts
    accelerating), then the pupil tile settles (a soft glassy tick); a breath of awareness below.
    sync = fully open."""
    d = 0.35 * size ** 0.7
    n = ns(d + 0.6)
    t = np.arange(n) / SR
    x = np.zeros(n, np.float32)
    m = ns(d)
    u = np.arange(m) / m
    exc = L.poisson_clicks(r, m, 200 + 2500 * u ** 1.5, 0.3 + 0.7 * u)
    sl = L.resonators(exc, r.uniform(1800, 7000, 6) / size ** 0.3, [0.004] * 6, [0.5] * 6) + L.hp(exc, 3000) * 0.3
    x[:m] += sl * np.sin(np.pi * u) ** 0.7
    x[:m] += L.svf(L.brown(m, r), 140 / size ** 0.5, 2.0, "bp") * np.sin(np.pi * u) * 0.4
    tk = tile_click(r, "gold", "wall", 0.7, 1.2)
    x[m:m + len(tk)] += tk[: n - m] * 0.8
    br = L.svf(L.brown(n, r), 90, 0.8, "lp") * np.exp(-((t - d) / (0.25 * size)) ** 2) * 0.5 * size
    x += br
    return Snd(L.fade(L.norm(x, 1.0), 0.005, 0.1), d)


def wheel_bed(r, dur, speed, n_wheels=3, tiles_per_rev=(60, 44, 30), rev_s=(9.0, 6.0, 4.0)):
    """ophanim: wheels within wheels of gold tesserae turning. Each rim's tiles pass the pivot -> a tick train
    at speed * tiles_per_rev / rev_s (Hz); the rolling mass -> low rumble swelling once per revolution;
    counter-rotating wheels circle through the stereo field. speed: per-sample 0..1+. stereo."""
    bank = tile_bank()
    n = ns(dur)
    t = np.arange(n) / SR
    speed = np.broadcast_to(np.asarray(speed, float), (n,))
    out = np.zeros((n, 2), np.float32)
    for w in range(n_wheels):
        ph = np.cumsum(speed / rev_s[w]) / SR * (1 if w % 2 == 0 else -1)
        tick_ph = np.abs(ph) * tiles_per_rev[w]
        idx = np.nonzero(np.diff(np.floor(tick_ph)) > 0)[0]
        if len(idx):
            v = 1.2 + 3.5 * np.clip(speed[idx], 0, 1.5)
            pan = 0.7 * np.sin(TAU * ph[idx])
            render_contacts(out, 0.0, bank, r, idx / SR, v, np.full(len(idx), "wall"), "gold", pan,
                            0.5 / (1 + w * 0.3))
        roll = L.svf(L.brown(n, r), 70 + 40 * w, 1.2, "bp") * (0.6 + 0.4 * np.cos(TAU * ph)) * np.clip(speed, 0, 1.5)
        out += L.pan2(L.f32(roll * 0.5), 0.6 * np.sin(TAU * ph + w))
    return out


def wing_unfold(r, dur=0.9, count=160):
    """a seraph's wing of tesserae-feathers unfolding: a fan of feather tiles flipping open in sequence (rate
    swells then eases) + the dark air of a vast wing. stereo, sync 0."""
    out = flip_wave(r, 0.0, dur, lambda tg: np.sin(np.pi * np.clip(tg / dur, 0, 1)) ** 1.5, lambda tg: -0.6 + 1.2 * tg / dur,
                    "gold", total=count, gain=0.8, spread=0.15)
    w = L.whoosh(r, dur + 0.4, dur * 0.6, 120, 1400, 0.8, -0.5, 0.5, dark=0.6, shape=2.0).x
    m = min(len(out), len(w))
    out[:m] += w[:m] * 0.5
    return Snd(L.fade(out, 0.002, 0.1), 0.0)


def ignite(r, size=1.0):
    """a wing ignites: a breath of oil vapour, then 'WHOOMF' - pressure pulse, roaring flash (band swept up then
    down), a burst of sparks; the flame settles into a roar. sync = ignition."""
    pre = 0.15
    n = ns(pre + 1.6)
    x = np.zeros(n, np.float32)
    m = ns(pre)
    x[:m] += L.hp(L.pink(m, r), 2500) * L.f32(np.linspace(0, 1, m) ** 2) * 0.15
    k = n - m
    tf = np.arange(k) / SR
    th = L.lp(L.white(k, r), 240) * L.env_ad(k, 0.004, 0.25) * 1.6
    fc = 300 + 1500 * np.exp(-((tf - 0.08) / 0.12) ** 2) + 500 * np.exp(-tf / 0.6)
    roar = L.svf(L.pink(k, r), fc, 0.9, "lp") * L.env_pts(k, [(0, 0), (0.02, 1), (0.25, 0.7), (1.6, 0.25)])
    sp = L.hp(L.poisson_clicks(r, k, 2500 * np.exp(-tf / 0.12) + 40), 2500) * 0.3
    flick = 0.8 + 0.2 * np.sin(TAU * np.cumsum(9 + 3 * L.smooth_noise(k, r, 1, 1)) / SR)
    x[m:] += (th + roar * flick + sp) * size ** 0.3
    return Snd(L.fade(L.norm(x, 1.0), 0.01, 0.3), pre)


def flame_roar(r, dur, level):
    """burning seraph wings: turbulent combustion roar opening with intensity (level per sample), flicker,
    crackle. stereo."""
    n = ns(dur)
    level = np.broadcast_to(np.asarray(level, float), (n,))
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        roar = L.svf(L.brown(n, r), 140 + 500 * np.clip(level, 0, 1.2), 0.7, "lp")
        body = L.bp(L.pink(n, r), 300, 2200) * 0.3
        fl = 0.75 + 0.25 * L.smooth_noise(n, r, 6, 2)
        cr = L.hp(L.poisson_clicks(r, n, 5 + 60 * level), 2500) * 0.3
        out[:, ch] = (roar + body) * fl * level + cr
    return out


def lock_boom(r):
    """the wheels lock: a ratchet flurry accelerating into a massive bronze bolt CLANK (inharmonic, damped fast),
    the sanctuary shudders (stone thud + D sub), shaken tesserae rain down. sync = the lock."""
    pre = 0.22
    n = ns(pre + 3.5)
    x = np.zeros(n, np.float32)
    ti = pre - 0.2 * 0.7 ** np.arange(10)                     # the wheels' ticks accelerate into the lock
    for k, tk in enumerate(np.sort(ti)):
        c = tile_click(r, "gold", "wall", 0.5 + 0.05 * k, 1.5)
        i = ns(max(0.0, tk))
        x[i:i + len(c)] += c[: n - i] * (0.25 + 0.04 * k)
    i0 = ns(pre)
    cl = L.ring([96, 233, 377, 571, 811, 1183, 1702], [0.7, 0.55, 0.4, 0.3, 0.22, 0.15, 0.1],
                [1, 0.8, 0.7, 0.55, 0.4, 0.3, 0.2], 1.4, r)
    x[i0:i0 + len(cl)] += cl[: n - i0] * 0.8
    x[i0:i0 + ns(0.003)] += L.hp(L.white(ns(0.003), r), 1500) * 1.2
    th = L.thud(r, 55, 1.0, 1.5, 400)
    x[i0:i0 + len(th)] += th[: n - i0] * 1.0
    tt_ = np.arange(n - i0) / SR
    sub = np.sin(TAU * np.cumsum(D1_HZ * (1 + 0.6 * np.exp(-tt_ / 0.05))) / SR) * L.env_ad(n - i0, 0.006, 2.6)
    x[i0:] += L.f32(sub) * 0.9
    return Snd(L.fade(L.norm(x, 1.0), 0.002, 0.4), pre)


def chain_creak(r, dur=1.6):
    """the polycandelon's chains: iron links clinking + a slow creak of the suspension. sync 0."""
    n = ns(dur)
    x = L.creak(r, dur, 12, 26, (190, 310, 470, 690), 0.12, 0.8).x[:n] * 0.6
    for k in range(int(r.integers(3, 7))):
        c = L.impact(r, "metal", r.uniform(0.3, 0.5), 0.6, 0.25)
        i = ns(r.uniform(0, dur - 0.2))
        x[i:i + len(c)] += c[: n - i] * 0.35
    return Snd(L.fade(L.norm(x, 1.0), 0.01, 0.1), 0.0)


def descend_chain(r, dur=2.0):
    """a flag lowered from the centre on a chain: pulley ratchet ticking (slowing to a stop) + chain rattle +
    the cable's creak. sync 0."""
    n = ns(dur + 0.3)
    t = np.arange(n) / SR
    v = np.clip(1 - t / dur, 0, 1) ** 0.6 * np.clip(t / 0.15, 0, 1)
    ph = np.cumsum(14 * v) / SR
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    x = np.zeros(n, np.float32)
    for i in idx:
        c = link_click(r, 1.1, (0.008, 0.03))
        x[i:i + len(c)] += c[: n - i] * 0.4
    rat = L.bp(L.poisson_clicks(r, n, 180 * v, v), 2000, 8000) * 0.3
    cr = L.creak(r, dur, 20, 14, (240, 380, 560), 0.1, 0.8).x
    x[: len(cr)] += cr[:n] * 0.2
    return Snd(L.fade(L.norm(x + rat, 1.0), 0.01, 0.2), 0.0)


# ================================================================================================
# the Flag in the heap, the breach, Babel
# ================================================================================================
def plant_in_heap(r):
    """a flag pole driven into a heap of fallen tesserae: the pole's wooden body thunk, the heap crunching under
    it, the floor's thud and a D sub; then a trickle of tiles sliding down the heap. stereo, sync = impact."""
    bank = tile_bank()
    n = ns(2.8)
    out = np.zeros((n, 2), np.float32)
    exc = np.zeros(ns(1.0), np.float32)
    exc[0] = 1.0
    wood = L.resonators(exc, [165, 430, 830, 1370, 2050], [0.12, 0.08, 0.05, 0.035, 0.02], [1, 0.7, 0.5, 0.35, 0.2])
    place_mono(out, wood * 2.4, 0, 1.0, 0.0)
    th = L.thud(r, 52, 0.7, 1.3, 380)
    place_mono(out, th, 0, 1.0, 0.0)
    tt_ = np.arange(n) / SR
    sub = np.sin(TAU * np.cumsum(D1_HZ * 2 * (1 + 0.4 * np.exp(-tt_ / 0.03))) / SR) * L.env_ad(n, 0.004, 0.9)
    place_mono(out, L.f32(sub), 0, 0.5, 0.0)
    k = 140
    tc = np.sort(r.exponential(0.04, k))
    render_contacts(out, 0.0, bank, r, tc, r.uniform(1.0, 6.0, k) * np.exp(-tc / 0.12), np.full(k, "heap"),
                    np.where(r.random(k) < 0.5, "gold", "smalti"), np.clip(r.normal(0, 0.25, k), -1, 1), 0.9)
    k2 = 45
    ts = 0.25 + np.sort(r.gamma(2.0, 0.25, k2))
    t_land, v_land, pn = ts, r.uniform(0.8, 3.0, k2), np.clip(r.normal(0, 0.35, k2), -1, 1)
    t, v, s, _, ids = bounce_events(r, t_land, v_land, np.where(r.random(k2) < 0.5, "heap", "floor"))
    render_contacts(out, 0.0, bank, r, t, v, s, "smalti", pn[ids], 0.8)
    return Snd(L.fade(L.norm(out, 1.0), 0.0002, 0.3), 0.0)


def tile_storm(r, dur=0.8, count=2500, t_spread=0.35):
    """the breach hurls a storm of tesserae at the camera: each tile screams past (a tiny pass-by: very short
    rising-falling band of air) and strikes something behind us; the mass is a tearing, glittering roar. stereo."""
    bank = tile_bank()
    n = ns(dur + 1.5)
    out = np.zeros((n, 2), np.float32)
    t = np.sort(r.gamma(2.0, t_spread / 2, count))
    t = t[t < dur]
    k = len(t)
    p = np.clip(r.normal(0, 0.55, k), -1, 1)
    render_contacts(out, 0.0, bank, r, t + r.uniform(0.02, 0.2, k), r.uniform(4, 16, k), np.full(k, "floor"),
                    np.where(r.random(k) < 0.5, "gold", "smalti"), -p * 0.8, 0.6)
    tt_ = np.arange(n) / SR
    dens = np.histogram(t, bins=max(2, int((dur + 1.5) * 200)), range=(0, dur + 1.5))[0].astype(float)
    env = np.interp(tt_, np.linspace(0, dur + 1.5, len(dens)), dens)
    env = np.sqrt(env / max(1e-9, env.max()))
    for ch in range(2):
        a = L.svf(L.white(n, r), 2500 + 6000 * env, 1.0, "bp") * env
        out[:, ch] += a * 0.35
    return L.fade(out, 0.0005, 0.2)


def screen_pop(r):
    """a slop screen dying as it shatters: an electrical pop and falling zap + glass. mono, sync 0."""
    z = L.zap(r, r.uniform(0.08, 0.2), r.uniform(2500, 6000), r.uniform(150, 400)).x
    c = tile_click(r, "screen", "floor", 1.0, 1.8)
    x = np.zeros(max(len(z), len(c)), np.float32)
    x[: len(z)] += z * 0.5
    x[: len(c)] += c
    return L.fade(L.norm(x, 1.0), 0.0002, 0.01)


def static_drain(r, dur=1.0):
    """Lord Egregore's static drains away: snow whose band closes and sinks, the field buzz slowing and dying,
    a last crackle. stereo, sync 0."""
    n = ns(dur)
    t = np.arange(n) / SR
    u = np.clip(t / dur, 0, 1)
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        s = L.svf(L.white(n, r), 9000 * (1 - u) ** 2 + 120, 0.8, "lp") * (1 - u) ** 1.5
        hz = 59.94 * (1 - 0.6 * u)
        bz = np.sign(np.sin(TAU * np.cumsum(hz) / SR)) * (1 - u) ** 2 * 0.1
        cr = L.hp(L.poisson_clicks(r, n, 80 * (1 - u)), 2000) * 0.3
        out[:, ch] = s + L.f32(bz) + cr
    return Snd(L.fade(out, 0.005, 0.05), 0.0)


def slop_bed(r, dur, level):
    """the slop's electric hum: thousands of screens lit from within - mains hum on A (55 Hz family, soft
    buzz), coil whine, the hash of countless feeds. level: per-sample 0..1. stereo."""
    n = ns(dur)
    level = np.broadcast_to(np.asarray(level, float), (n,))
    h = L.hum(r, dur, 55.0, (0.5, 0.6, 0.3, 0.25, 0.08, 0.1), 0.25)
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        wh = sum(np.sin(TAU * f * np.arange(n) / SR + r.uniform(0, TAU)) * a
                 for f, a in ((9120.0 * r.uniform(0.99, 1.01), 0.02), (11430.0 * r.uniform(0.99, 1.01), 0.012)))
        hash_ = L.bp(L.poisson_clicks(r, n, 300 + 2500 * level, 1.0), 1500, 7000) * 0.12
        out[:, ch] = (h * 0.3 + L.f32(wh) + hash_) * level
    return out


def gear_train(r, dur, rate=4.0, size=1.0, bright=1.0, width=0.5):
    """a machine of gilded cogs made of tesserae: every tooth engaging ticks (rate teeth/s, scalar or per-sample
    so the machine can wind down), a heavier escapement 'tock' every 4th tooth, the stone wheel's rolling rumble
    (level follows the speed). stereo."""
    bank = tile_bank()
    n = ns(dur)
    out = np.zeros((n, 2), np.float32)
    rate = np.broadcast_to(np.asarray(rate, float), (n,))
    ph = np.cumsum(rate) / SR + r.random()
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    tk = np.clip(idx / SR + r.normal(0, 0.003, len(idx)), 0, dur)
    k = len(tk)
    sp = np.clip(rate / max(1e-9, rate.max()), 0, 1)
    v = (1.2 + 2.2 * sp[idx]) * bright
    render_contacts(out, 0.0, bank, r, tk, v, np.full(k, "wall"), np.where(r.random(k) < 0.6, "gold", "stone"),
                    r.uniform(-width, width, k), 0.8)
    for t in tk[::4]:
        th = L.thud(r, 90 / size, 0.18, 0.6, 700)
        place_mono(out, th, ns(t), 0.25 * float(sp[min(n - 1, ns(t))]), r.uniform(-0.3, 0.3))
    roll = L.svf(L.brown(n, r), 60 / size ** 0.5, 1.5, "bp") * (0.7 + 0.3 * np.sin(TAU * ph / 8)) * sp
    out += L.as_stereo(L.f32(roll * 0.4))
    return out


# ================================================================================================
# spaces: procedural impulse responses of the basilica's rooms
# ================================================================================================
IR_SPECS = {
    "vault": dict(rt60=3.4, predelay=0.018, band_mul=(1.25, 1.0, 0.8, 0.5), seed=31, er_span=0.07, early=8),
    "nave": dict(rt60=5.6, predelay=0.035, band_mul=(1.35, 1.0, 0.72, 0.42), seed=32, er_span=0.12, early=10),
    "hall": dict(rt60=4.6, predelay=0.03, band_mul=(1.3, 1.0, 0.75, 0.45), seed=33, er_span=0.1, early=10),
    "sanctuary": dict(rt60=7.2, predelay=0.05, band_mul=(1.4, 1.0, 0.68, 0.38), seed=34, er_span=0.15, early=6),
    "void": dict(rt60=9.0, predelay=0.07, band_mul=(1.0, 1.0, 0.85, 0.6), seed=35, er_span=0.2, early=0),
    "babel": dict(rt60=6.5, predelay=0.06, band_mul=(1.4, 1.0, 0.7, 0.4), seed=36, er_span=0.14, early=8),
    "bare": dict(rt60=7.5, predelay=0.04, band_mul=(1.3, 1.0, 0.85, 0.55), seed=37, er_span=0.12, early=10),
    "dome": dict(rt60=8.0, predelay=0.045, band_mul=(1.3, 1.0, 0.8, 0.5), seed=38, er_span=0.15, early=6,
                 flutter=(0.095, 0.55)),
    "apse": dict(rt60=5.0, predelay=0.03, band_mul=(1.3, 1.0, 0.78, 0.48), seed=39, er_span=0.1, early=10,
                 echoes=[(0.07, 0.25, 4000)]),
}

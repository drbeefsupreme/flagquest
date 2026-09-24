"""t06 deluge: the waters above pour through the holes of the broken firmament.

Every fallen shard i opens a hole (its Voronoi cell at canopy height). Water pours through it from
tw_i = fall_i + LAG: a column of strands whose leading front falls ballistically (v0, g) with a ragged,
per-strand front; strands are drawn as grey bands with white glints (engraved falling water), the front
breaks into spray. Where the columns reach the playa they splash; the playa floods (level rising, crests
travelling). The same geometry, sliced at z = 0, feeds the fire simulation (quench -> steam explosion).
Analytic: every quantity is a pure function of time (no state), so frames render independently.
"""
import math

import numpy as np

G = 9.81
V0 = 2.5
LAG = 0.06
T_FLOOD = None      # set by the scene (first ground contact)


def drop(tau):
    tau = np.maximum(tau, 0.0)
    return V0 * tau + 0.5 * G * tau * tau


def build(fr, canopy_h, fall_t, rmax=260.0, seed=77):
    """per-hole strand sets. fr: fracture dict (polys, P, rc). fall_t: (n,) shard let-go times"""
    rng = np.random.default_rng(seed)
    holes = []
    for i, poly in enumerate(fr["polys"]):
        if poly is None or fr["rc"][i] > rmax:
            continue
        c = poly.mean(0)
        r = float(np.hypot(c[0], c[1]))
        h = float(canopy_h(r))
        # strands: points on the hole boundary (inset) + interior
        k = len(poly)
        per = np.sum(np.hypot(*(np.roll(poly, -1, 0) - poly).T))
        area = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
        ns = int(np.clip(per / 0.9, 8, 48))
        pts = []
        for s in range(ns):
            u = (s + rng.uniform(0, 1)) / ns * k
            a = int(u) % k
            f = u - int(u)
            p = poly[a] + (poly[(a + 1) % k] - poly[a]) * f
            p = c + (p - c) * rng.uniform(0.55, 0.95)
            pts.append(p)
        pts = np.array(pts)
        holes.append(dict(i=i, c=c, r=r, h=h, poly=poly, area=area, pts=pts,
                          lagp=rng.uniform(0.0, 0.25, len(pts)),       # per-strand ragged front
                          wid=rng.uniform(0.35, 1.0, len(pts)),
                          seed=int(rng.integers(1 << 30)), t0=float(fall_t[i]) + LAG))
    return holes


def strands_at(holes, T, ground=0.0):
    """world segments of all visible strands at time T: (top (n,3), bottom (n,3), width (n,) m, front_spray (n,))"""
    tops, bots, wid, fr_, hole_id = [], [], [], [], []
    for hk, H in enumerate(holes):
        tau = T - H["t0"]
        if tau <= 0:
            continue
        n = len(H["pts"])
        # the column's tail: the hole keeps pouring; the trailing top recedes a little as the sea drains
        taus = tau - H["lagp"]
        ok = taus > 0
        if not ok.any():
            continue
        d = drop(taus[ok])
        top_y = np.full(ok.sum(), H["h"])
        bot_y = np.maximum(ground, H["h"] - d)
        P = H["pts"][ok]
        tops.append(np.stack([P[:, 0], top_y, P[:, 1]], 1))
        bots.append(np.stack([P[:, 0], bot_y, P[:, 1]], 1))
        wid.append(H["wid"][ok] * (0.6 + 0.4 * min(1.0, tau / 0.5)))
        fr_.append((H["h"] - d > ground).astype(np.float32))
        hole_id.append(np.full(ok.sum(), hk))
    if not tops:
        return None
    return (np.concatenate(tops), np.concatenate(bots), np.concatenate(wid), np.concatenate(fr_),
            np.concatenate(hole_id))


def ground_contact(holes, ground=0.0):
    """time each hole's water first reaches the ground"""
    out = []
    for H in holes:
        # solve V0 t + g t^2/2 = h
        h = H["h"] - ground
        t = (-V0 + math.sqrt(V0 * V0 + 2 * G * h)) / G
        out.append(H["t0"] + t)
    return np.array(out)


def water_plane(holes, T, sim, zband=5.0):
    """water density on the fire grid (plane z = 0): columns crossing |z| < zband"""
    wf = None
    for H in holes:
        tau = T - H["t0"]
        if tau <= 0 or abs(H["c"][1]) > zband + 8.0:
            continue
        P = H["poly"]
        # x-extent of the hole where it crosses z in [-zband, zband]
        inb = np.abs(P[:, 1]) < zband + 3.0
        if not inb.any():
            continue
        x0, x1 = P[inb, 0].min(), P[inb, 0].max()
        ytop = H["h"]
        ybot = max(0.0, H["h"] - float(drop(tau)))
        i0, j0 = sim.cell(x0, ytop)
        i1, j1 = sim.cell(x1, ybot)
        i0, i1 = int(max(0, i0)), int(min(sim.nx, i1 + 1))
        j0, j1 = int(max(0, j0)), int(min(sim.ny, j1 + 1))
        if i1 <= i0 or j1 <= j0:
            continue
        if wf is None:
            wf = np.zeros((sim.ny, sim.nx), np.float32)
        wf[j0:j1, i0:i1] += 1.0
    if wf is None:
        return None
    # pooled water on the playa quenches the base of the fire too
    return np.minimum(wf, 2.0)


def flood_level(T, t0):
    if T <= t0:
        return 0.0
    x = T - t0
    return 2.6 * (1.0 - math.exp(-x / 1.35)) + 0.12 * x

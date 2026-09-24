"""t06 fire driver: runs the gas simulation of the effigy burn + steam explosion over the whole scene once
(setup) and caches the fields per frame (uint8) + the embers. Plane z = 0 (the effigy plane), metres."""
import math

import numpy as np

from scenes import t06_fluid as FL

X0, X1 = -46.0, 46.0          # domain (m)
YTOP = 27.0
HC = 0.12                      # cell (m)
NX = int(round((X1 - X0) / HC))
NY = int(round((YTOP + 0.6) / HC))
SUB = 3                        # substeps per frame
NSPARK = 2400
VER = 15
BOOM = 26.0                    # steam explosion: radial blast speed at the source rim (m/s, peak)
PRM = dict(eF=10.0, eT=2.6, eS=0.25,       # emission: flame, temperature, smoke
           bT=80.0, wS=6.0, bM=55.0,       # buoyancy of heat, weight of soot, lift of steam
           cT=0.35, dF=3.2,                # cooling, flame decay (1/s)
           jet=120.0, nf=0.07,             # upward jet at the fuel, flicker noise frequency
           wq=6.0, ws=2.2, wp=90.0,        # deluge: quench rate, steam yield, water drag
           wx=0.0, sg=0.55, vort=4.0,      # wind, soot generation, vorticity confinement
           rad=0.3)                       # emission radius around each beam (m)


def _smooth(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fuel_field(sim, beams):
    """rasterise the effigy lumber into a fuel mask (plane z=0); plinth burns hottest"""
    fuel = np.zeros((sim.ny, sim.nx), np.float32)
    ys, xs = np.mgrid[0:sim.ny, 0:sim.nx]
    wx, wy = sim.world(xs, ys)
    gain = {0: 1.35, 1: 1.0, 2: 1.0, 3: 0.9, 4: 0.9, 5: 0.6}
    for p0, p1, w, part in zip(beams["p0"], beams["p1"], beams["w"], beams["part"]):
        ax, ay, bx, by = p0[0], p0[1], p1[0], p1[1]
        lo_x, hi_x = min(ax, bx) - 0.6, max(ax, bx) + 0.6
        lo_y, hi_y = min(ay, by) - 0.6, max(ay, by) + 0.6
        i0, j1 = sim.cell(lo_x, lo_y)
        i1, j0 = sim.cell(hi_x, hi_y)
        i0, i1 = max(0, int(i0)), min(sim.nx, int(i1) + 2)
        j0, j1 = max(0, int(j0)), min(sim.ny, int(j1) + 2)
        if i1 <= i0 or j1 <= j0:
            continue
        X = wx[j0:j1, i0:i1]
        Y = wy[j0:j1, i0:i1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy + 1e-9
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0, 1)
        d = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
        r = max(float(w) * 0.5, PRM["rad"])
        m = np.clip(1.0 - (d - r) / (HC * 2.5), 0, 1) * gain.get(int(part), 1.0)
        fuel[j0:j1, i0:i1] = np.maximum(fuel[j0:j1, i0:i1], m.astype(np.float32))
    return fuel


def simulate(S, E, T_of, ev, water_fn=None, log=print):
    """S: SceneInfo; E: effigy module; T_of(f) global time of local frame f; ev: dict of event times
    (ignite, reach, crack, open, quench0, quench1). water_fn(T, sim) -> (ny,nx) water density or None.
    Returns dict of uint8 fields (n, NY, NX): flame, smoke, steam, heat + embers."""
    b = E.beams()
    geo = int(abs(float(np.round(b["p0"], 3).sum() * 7.1 + np.round(b["p1"], 3).sum() * 3.3 + len(b["w"]))) * 1000) % 1000003
    path = S.cache / f"fire_v{VER}_{NX}x{NY}_{geo}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    sim = FL.FireSim(NX, NY, HC, X0, YTOP)
    ys, xs = np.mgrid[0:NY, 0:NX]
    wx, wy = sim.world(xs, ys)
    ground = wy < 0.0
    ceil = wy > E.canopy_h(np.abs(wx)) - 0.05
    sim.fuel[:] = fuel_field(sim, E.beams())
    fuel_rows = sim.world(0, np.arange(NY))[1]
    boom_mask = (np.exp(-((wx / 5.5) ** 2 + ((wy - 6.5) / 6.0) ** 2)) * (wy > 0)).astype(np.float32)
    # potential flow of a source of radius R0 at (0, 4 m): radial speed V*r/R0 inside, V*R0/r outside;
    # stored as an acceleration-like field (cells/s per step via env*dt*BOOM)
    bx, by = wx, wy - 4.0
    br = np.hypot(bx, by) + 1e-6
    R0 = 5.0
    prof = np.where(br < R0, br / R0, R0 / br)
    vcell = BOOM / HC
    from scipy.ndimage import gaussian_filter
    nrng = np.random.default_rng(123)
    lump_a = gaussian_filter(nrng.normal(size=(NY, NX)), 9.0)
    lump_b = gaussian_filter(nrng.normal(size=(NY, NX)), 9.0)
    lump_a /= lump_a.std() + 1e-9
    lump_b /= lump_b.std() + 1e-9
    lump_a *= 0.7
    lump_b *= 0.7
    psi = gaussian_filter(nrng.normal(size=(NY, NX)), 11.0)
    psi /= np.abs(psi).max() + 1e-9
    cu = np.gradient(psi, axis=0)
    cv_ = -np.gradient(psi, axis=1)
    cm = np.abs(np.hypot(cu, cv_)).max() + 1e-9
    reach = np.exp(-((wx / 22.0) ** 2 + ((wy - 8.0) / 14.0) ** 2))
    curl_u = (cu / cm * 2600.0 * reach).astype(np.float32)      # accel, cells/s^2
    curl_v = (cv_ / cm * 2600.0 * reach).astype(np.float32)
    boom_u = (vcell * prof * bx / br).astype(np.float32)
    boom_v = (-vcell * prof * by / br).astype(np.float32)     # grid rows grow downward
    n = S.n
    out = {k: np.zeros((n, NY, NX), np.uint8) for k in ("flame", "smoke", "steam", "heat")}
    ex = np.zeros((n, NSPARK, 4), np.float16)     # x, y (m), vx, vy (m/s)
    ea = np.zeros((n, NSPARK), np.uint8)          # brightness 0..255 (0 = dead)
    rng = np.random.default_rng(606)
    px = np.zeros(NSPARK)
    py = np.zeros(NSPARK)
    pvx = np.zeros(NSPARK)
    pvy = np.zeros(NSPARK)
    age = np.full(NSPARK, -1.0)
    life = rng.uniform(1.2, 3.2, NSPARK)
    fuel_idx = np.argwhere(sim.fuel > 0.3)
    nxt = 0
    tI, tR = ev["ignite"], ev["reach"]
    energy = np.zeros(n, np.float32)
    pan = np.zeros(n, np.float32)
    t_start = tI - 0.1
    for f in range(n):
        T1 = T_of(f)
        if T1 < t_start:
            continue
        for s in range(SUB):
            T = T1 - (SUB - 1 - s) / (SUB * S.fps)
            dt = 1.0 / (SUB * S.fps)
            u = (T - tI) / (tR - tI)
            front = E.HEAD_TOP * max(0.0, min(1.0, u)) ** 1.1 if u > 0 else -1.0
            if u > 1:
                front = E.HEAD_TOP + 3.0
            burn = (_smooth(front + 0.2, front - 1.8, fuel_rows) * (T > tI)).astype(np.float32)
            # the flames reach the Flag: a burst of fire engulfs the head
            if T > tR:
                burn = burn + (3.5 * math.exp(-(T - tR) / 0.7) * _smooth(10.5, 13.5, fuel_rows)).astype(np.float32)
            # accelerant burst on the plinth at ignition
            whoosh = math.exp(-max(0.0, T - tI) / 0.35) * (T > tI)
            plinth = (fuel_rows < 2.6).astype(np.float32) * whoosh * 2.5
            burn = burn + plinth
            q = _smooth(ev["quench0"], ev["quench1"], T)
            gain = (0.55 + 0.45 * _smooth(tI, tR + 1.0, T)) * (1.0 - q)
            # the crystal ceiling is solid until the firmament breaks
            sim.solid[:] = ground | (ceil & (T < ev["open"]))
            sim.water[:] = 0.0
            if water_fn is not None:
                wf = water_fn(T, sim)
                if wf is not None:
                    sim.water[:] = wf
            P = PRM
            prm = np.array([P["eF"], P["eT"], P["eS"], P["bT"], P["wS"], P["bM"], P["cT"], P["dF"], P["jet"],
                            P["nf"], P["wq"], P["ws"], P["wp"], P["wx"], P["sg"], gain], np.float32)
            # the steam explosion: flash-boiling at the white-hot pyre when the torrents arrive
            bt = T - ev.get("boom", 1e9)
            if 0.0 <= bt < 1.6:
                env = math.exp(-bt / 0.32) * min(1.0, bt / 0.06)
                sim.boom_u = (boom_u * env).astype(np.float32)
                sim.boom_v = (boom_v * env).astype(np.float32)
                lump = np.clip(0.35 + lump_a * math.cos(bt * 5.0) + lump_b * math.sin(bt * 5.0), 0, 2)
                sim.steam += (dt * 9.0 * env * boom_mask * lump).astype(np.float32)
                sim.temp += (dt * 5.0 * env * boom_mask).astype(np.float32)
                # curl-noise turbulence (divergence-free: survives the projection) -> billows
                ce = min(1.0, bt / 0.1) * math.exp(-bt / 0.9)
                sim.u += (curl_u * ce * dt).astype(np.float32)
                sim.v += (curl_v * ce * dt).astype(np.float32)
            else:
                sim.boom_u = None
            sim.step(T, dt, burn, prm, vort=P["vort"], iters=36)
            # embers: spawn from burning fuel
            rate = 0.0 if u <= 0 else (240.0 + 900.0 * whoosh) * gain
            k = rng.poisson(rate * dt)
            for _ in range(k):
                j, i = fuel_idx[rng.integers(len(fuel_idx))]
                if burn[j] < 0.3:
                    continue
                sidx = nxt % NSPARK
                nxt += 1
                px[sidx] = i + rng.uniform(-0.5, 0.5)
                py[sidx] = j + rng.uniform(-0.5, 0.5)
                pvx[sidx] = rng.normal(0, 18)
                pvy[sidx] = -rng.uniform(20, 80)
                age[sidx] = 0.0
                life[sidx] = rng.uniform(0.9, 3.4)
            FL._sparks_step(px, py, pvx, pvy, age, life, sim.u, sim.v, sim.temp, sim.solid, dt, 3.0, 25.0,
                            240.0, 17, f * SUB + s)
        alive = (age >= 0) & (age <= life)
        ex[f, :, 0] = X0 + (px + 0.5) * HC
        ex[f, :, 1] = YTOP - (py + 0.5) * HC
        ex[f, :, 2] = pvx * HC
        ex[f, :, 3] = -pvy * HC
        br = np.where(alive, np.clip(1.0 - age / np.maximum(life, 1e-3), 0, 1) ** 0.6, 0.0)
        ea[f] = (br * 255).astype(np.uint8)
        out["flame"][f] = np.clip(sim.flame * 90.0, 0, 255).astype(np.uint8)
        out["smoke"][f] = np.clip(sim.smoke * 160.0, 0, 255).astype(np.uint8)
        out["steam"][f] = np.clip(sim.steam * 60.0, 0, 255).astype(np.uint8)
        out["heat"][f] = np.clip(sim.temp * 70.0, 0, 255).astype(np.uint8)
        fl = sim.flame
        tot = float(fl.sum())
        energy[f] = tot
        if tot > 1e-6:
            pan[f] = float((fl.sum(0) * np.arange(NX)).sum() / tot / NX * 2 - 1)
        if f % 48 == 0:
            log(f"  fire f{f} T={T1:.2f} flame={fl.max():.2f} smoke={sim.smoke.max():.2f} steam={sim.steam.max():.2f}")
    res = dict(out, embers=ex, ember_a=ea, energy=energy, pan=pan)
    np.savez(path, **res)
    return res

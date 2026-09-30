"""m09 - THE DOME OF FLAGISTAN (163.0-187.5): the hymn. Owner: M09.

We stand on the floor of the basilica, under the dome, and look straight up. A camera at the "south pole" of the
dome's sphere sees the hemisphere through a stereographic projection - which makes the dome, exactly, a Poincare
disk. So the dome of Flagistan is a hyperbolic {8,3} tiling (m09_hyper): every octagon is a small Pentecost dome
(oculus, rays, eight peoples each alone in an arched niche, each holding a real Flag, windows at the corners).

  163.0   whip-tilt lands (from m08's smear) straight up into the dome: the central octagon fills it - a Pentecost
          dome, the ring of windows (Hagia Sophia's "golden chain"), seraphim on the pendentives, titulus
          HIC OMNES GENTES ... CAELVM ET TERRA TRANSIBVNT ... VEXILLORVM MANEBVNT. The dome wakes with the cantor:
          the gold rays descend from the oculus to the eight peoples (each holds a Flag at its side), the drum
          windows light one by one, the oculus swells.
  168.7   "Flags be." every people raises its Flag (the pole slides up through the fist and telescopes) and the
          dome opens: the Euclidean zoom falls back and the rim turns out to be infinity - rings of domes, peoples
          and crowns of Flags, smaller and smaller, forever (Escher's Circle Limit, in glass and gold).
  171.557 "Flags are." a Moebius glide: one tile over, a new dome comes to the centre (every place is the centre).
  174.414 "Flags will." a loxodromic glide two tiles on, turning.   177.271 "Flags." the tiling spins, rings of
          light race out to the circle at infinity. On every beat (84 BPM, accents on the sung words) a ring of
          light runs out through the tiles and every lamp, oculus and Flag it passes ignites.
  180.4   N09: heaven (the dome, from the oculus outward) and earth (the architecture, from the frame inward)
          dissolve into tesserae falling toward us - the Flags, which were never mosaic, remain, floating in light.
  184.9   N10 "And the Survey was completed." - the flags drift on; 187.0-187.458 white-gold flood to full
          flag_glow (hand-off to m10, which opens on uniform #FFF1B8).

Flags: every one is vx.flag cloth (draw_flag_field), placed by the same Moebius maps that carry the tiles,
composited after the mosaic and the falling tiles. Nothing is drawn over cloth.
"""
import json
import math

import cv2
import numpy as np

from vx import *
from vx import foley
from vx import mosaic as mz
from vx.config import CACHE
from vx.flag import Flag, draw_flag_field

from . import m09_art as ART
from . import m09_hyper as HY

VER_SHEETS, VER_FLAGS, VER_PARTS, VER_HERO = "s18", "f11", "p18", "h9"
F_PX = 455.0                          # design px per w unit (the dome's radius on screen, looking straight up)
CX, CY = 960.0, 540.0
BEAT = 60.0 / 84.0
FLAG_GLOW = C["flag_glow"]

POST = dict(bloom=0.42, bloom_thresh=0.70, bloom_radius=26.0, vignette=0.22, grain=0.0, saturation=1.02)   # no grain: nothing over the cloth


# ================================================================== the plan (all times global seconds)
def _plan(tl):
    cant = [(163.00, 164.19), (164.27, 165.46), (165.54, 167.03), (167.13, 168.67)]
    p = CACHE / "foley" / "cantor_phrases.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            ph = d.get("phrases", []) if isinstance(d, dict) else d
            cc = [(float(x.get("start", x.get("t"))), float(x["end"])) for x in ph][:4]
            if len(cc) == 4:
                cant = cc
        except (ValueError, KeyError, TypeError):
            pass
    hy = tl.hymn
    ph0 = float(hy["choir_start"])
    PH = [ph0 + 4 * k * BEAT for k in range(4)]
    w = {k: tl.word("N09", k) for k in ("Heaven", "earth", "away", "Flags")}
    return dict(
        PH=PH, hymn_end=float(hy["end"]),
        rays=(cant[0][0] + 0.40, cant[1][1] - 0.05),
        windows=(cant[2][0] + 0.06, cant[2][1] - 0.15),
        oculus=(cant[3][0], cant[3][1]),
        reveal=(PH[0], PH[0] + 2.75),
        n09=tl.line("N09")["start"], heaven=w["Heaven"][0], earth=w["earth"][0], away=w["away"][1],
        flags_word=w["Flags"][0],
        n10=tl.line("N10")["start"], completed=tl.word("N10", "completed")[0],
    )


# ================================================================== camera
X0_PITCH = math.radians(24.0)         # the whip lands from 66 degrees pitch
V0_PITCH = -5.0                       # rad/s at the cut (~2300 px/s at the frame centre)
TAU_PITCH = 0.13


def pitch(T):
    t = max(0.0, T - 163.0)
    a = V0_PITCH + X0_PITCH / TAU_PITCH
    x = (X0_PITCH + a * t) * math.exp(-t / TAU_PITCH)
    dx = (a - (X0_PITCH + a * t) / TAU_PITCH) * math.exp(-t / TAU_PITCH)
    return math.pi / 2 - x, -dx


_ROLL = [(163.0, 0.10), (164.6, 0.045), (168.7, 0.05), (171.4, 0.085), (177.27, 0.09), (178.6, 0.15),
         (180.1, 0.075), (187.5, 0.06)]


def roll(T):
    acc = 0.0
    for (a, ra), (b, rb) in zip(_ROLL[:-1], _ROLL[1:]):
        if T <= a:
            break
        u = min(T, b) - a
        acc += u * ra + u * u / (2 * (b - a)) * (rb - ra)
    return acc + 0.35


_ROLL_T = np.linspace(162.9, 187.6, 2000)
_ROLL_V = np.array([roll(t) for t in _ROLL_T])


def zoom(T, P):
    r0, r1 = P["reveal"]
    return HY.XM + (1.0 - HY.XM) * smootherstep(r0, r1, T)


TH_GLIDE1, TH_GLIDE2, ROT3, ROT4 = 0.0, 3 * math.pi / 4, math.pi / 4, -math.pi / 2
TH_DRIFT, V_DRIFT = -2.2, 0.30


def mobius(T, P):
    """view -> world automorphism (a, b) at global time T"""
    PH = P["PH"]
    M = (1 + 0j, 0j)
    s2 = ease_in_out_sine(clamp((T - PH[1]) / (PH[2] - PH[1])))
    M = HY.mob_compose(M, HY.mob_translate(HY.STEP * s2, TH_GLIDE1))
    s3 = ease_in_out_sine(clamp((T - PH[2]) / (PH[3] - PH[2])))
    M = HY.mob_compose(M, HY.mob_compose(HY.mob_translate(2 * HY.STEP * s3, TH_GLIDE2), HY.mob_rotate(ROT3 * s3)))
    s4 = ease_in_out_sine(clamp((T - PH[3]) / (P["hymn_end"] - PH[3])))
    M = HY.mob_compose(M, HY.mob_rotate(ROT4 * s4))
    td = T - P["n09"]
    if td > 0:
        d = V_DRIFT * (td - 0.8 * (1 - math.exp(-td / 0.8)))       # eases in
        M = HY.mob_compose(M, HY.mob_translate(d, TH_DRIFT))
    return M


def cam_basis(T):
    """right, down, forward (world) of the camera at the floor under the dome"""
    p, _ = pitch(T)
    yaw = roll(T)
    fwd = np.array([0.0, math.cos(p), math.sin(p)])
    up = np.array([0.0, -math.sin(p), math.cos(p)])
    right = np.array([1.0, 0.0, 0.0])
    down = -up
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return R @ right, R @ down, R @ fwd


def homography(fc, T):
    """pixel (px + 0.5, py + 0.5) -> w (homogeneous rows d_x, d_y, d_z)"""
    r, d, f = cam_basis(T)
    K = np.array([[1.0 / (fc.s * F_PX), 0, -CX / F_PX], [0, 1.0 / (fc.s * F_PX), -CY / F_PX], [0, 0, 1.0]])
    Rm = np.stack([r, d, f], 1)
    return Rm @ K


def w_to_screen(w, T):
    """w (complex array) -> design screen coords (x, y), and depth sign"""
    r, d, f = cam_basis(T)
    X = np.stack([w.real, w.imag, np.ones_like(w.real)], -1)
    xc = X @ r
    yc = X @ d
    zc = X @ f
    zc = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
    return CX + F_PX * xc / zc, CY + F_PX * yc / zc, zc


# ================================================================== light, emission, rings
RING_ACCENT = (1.0, 0.3, 0.7, 0.3)    # per beat of a phrase: 'Flags' / - / be, are, will / -


def rings(T, P):
    out = []
    PH = P["PH"]
    for k in range(16):
        tb = PH[0] + k * BEAT
        if T < tb:
            break
        tau = T - tb
        base = RING_ACCENT[k % 4]
        if k >= 12:
            base *= 1.35
        s = base * math.exp(-tau / 1.5)
        if s > 0.012:
            out.append((0.2 + 3.1 * tau, s, 0.24 + 0.10 * tau))
    for extra in (PH[3] + 0.13, PH[3] + 0.26):
        if T > extra:
            tau = T - extra
            s = 0.9 * math.exp(-tau / 1.6)
            if s > 0.012:
                out.append((0.2 + 2.6 * tau, s, 0.3 + 0.12 * tau))
    tc = P["completed"]
    if T > tc:
        tau = T - tc
        s = 0.8 * math.exp(-tau / 1.4)
        out.append((0.2 + 2.2 * tau, s, 0.4 + 0.1 * tau))
    if not out:
        out.append((-10.0, 0.0, 1.0))
    return np.array(out[-10:], np.float64)


def emission(T, P):
    E = np.zeros(16, np.float64)
    o0, o1 = P["oculus"]
    E[1] = 0.62 + 0.30 * smoothstep(P["rays"][0], P["rays"][1], T) + 0.55 * smoothstep(o0, o1, T) \
        - 0.40 * smoothstep(P["PH"][0], P["PH"][0] + 1.5, T)
    E[2] = 0.34
    w0, w1 = P["windows"]
    for k in range(8):
        tk = w0 + (w1 - w0) * ((k * 3) % 8) / 8.0
        on = smoothstep(tk, tk + 0.22, T)
        flash = math.exp(-((T - tk - 0.12) / 0.14) ** 2)
        E[3 + k] = 0.14 + 0.46 * on + 0.5 * flash
    return E


def flag_glow(T, P):
    g = 0.0
    r1 = P["rays"][1]
    g += 0.55 * math.exp(-((T - r1) / 0.35) ** 2)                  # the light reaches the Flags
    for k in range(4):
        g += 0.35 * math.exp(-((T - P["PH"][k] - 0.1) / 0.3) ** 2)
    g += 0.16 * smoothstep(P["n09"], P["n09"] + 2.0, T)               # floating in light
    g += 0.45 * math.exp(-((T - P["flags_word"] - 0.15) / 0.35) ** 2)  # "...the ten thousand FLAGS remain"
    g += 0.5 * smoothstep(186.6, 187.3, T)
    return g


def lift(T, P):
    """0 -> 1: every people raises its Flag at 'Flags be.'"""
    return ease_in_out_sine(clamp((T - P["PH"][0] - 0.1) / 1.9))


def pole_world(T, P, sh, fl):
    """world positions (complex) of every copy's pole base and top at time T"""
    anc = sh["anchors"]                     # (K, 3): base, dir, grip
    k = lift(T, P)
    b = anc[:, 0] + ART.LIFT_D * k * anc[:, 1]
    t = b + ART.POLE_L * (1 + (ART.LIFT_L - 1) * k) * anc[:, 1]
    v = fl["var"]
    G = fl["G"]
    zb, zt = b[v], t[v]
    return (G[:, 0] * zb + G[:, 1]) / (G[:, 2] * zb + G[:, 3]), (G[:, 0] * zt + G[:, 1]) / (G[:, 2] * zt + G[:, 3])


def wind(T, P):
    PH = P["PH"]
    w = 1.0
    for a, b, k in ((PH[1], PH[2], 0.35), (PH[2], PH[3], 0.45), (PH[3], P["hymn_end"], 0.3)):
        w += k * math.sin(math.pi * clamp((T - a) / (b - a)))
    return w


# ================================================================== setup
def _bake_sheets(S):
    path = S.cache / f"sheets_{VER_SHEETS}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    fan_s = ((0.0 - ART.S_X0) * ART.S_RES, (0.0 - ART.S_Y0) * ART.S_RES)
    f = ART.sector_fields()
    E = ART.sector_edges()
    sheets = []
    for v in range(HY.NVAR):
        cart = ART.paint_sector(v)
        sh = ART.bake(cart, f["emit"], f["grp"], f["ray"], tile=ART.S_TILE, fine_tile=ART.S_FINE, seed=101 + v,
                      extra_edges=E, grout=1.6, fan_center=fan_s, parts=ART.sector_parts(cart, f))
        sheets.append(sh)
        print(f"[m09] sector sheet {v}: {len(sh['col'])} tesserae", flush=True)
    ca = ART.paint_architecture()
    fa = ART.arch_fields()
    sha = ART.bake(ca, fa["emit"], fa["grp"], fa["ray"], tile=11.0, fine_tile=5.0, seed=77,
                   extra_edges=ART.arch_edges(), grout=1.3, fan_center=(ART.A_N / 2, ART.A_N / 2),
                   parts=ART.arch_titulus_parts(ca))
    print(f"[m09] architecture sheet: {len(sha['col'])} tesserae", flush=True)
    # stack + global tile ids
    out = {}
    off = 0
    labs, rims, us, vs, atls = [], [], [], [], []
    per = {k: [] for k in ("col", "gold", "gt", "tilt", "ray", "emit", "grp", "rnd", "pos", "size", "ang")}
    starts = []
    for sh in sheets + [sha]:
        n = len(sh["col"])
        starts.append(off)
        lab = sh["lab"].copy()
        lab[lab >= 0] += off
        labs.append(lab)
        rims.append(sh["rim"])
        us.append(sh["u"])
        vs.append(sh["v"])
        atl, lvl = ART.mip_atlas(sh["flat"])
        atls.append((atl, lvl))
        for k in per:
            per[k].append(sh[k])
        off += n
    K = HY.NVAR
    out["s_lab"] = np.stack(labs[:K])
    out["s_rim"] = np.stack(rims[:K])
    out["s_u"] = np.stack(us[:K])
    out["s_v"] = np.stack(vs[:K])
    out["s_atl"] = np.stack([a for a, _ in atls[:K]])
    out["s_lvl"] = atls[0][1]
    out["a_lab"] = labs[K][None]
    out["a_rim"] = rims[K][None]
    out["a_u"] = us[K][None]
    out["a_v"] = vs[K][None]
    out["a_atl"] = atls[K][0][None]
    out["a_lvl"] = atls[K][1]
    for k in per:
        out["T" + k] = np.concatenate(per[k])
    out["starts"] = np.array(starts + [off], np.int64)
    # average dome colour (inside the sector) for sub-pixel tiles: rgb + emission
    inside = f["inside"]
    fl = np.mean([sh["flat"][inside] for sh in sheets], axis=(0, 1))
    g = mz.SC["gold"]
    out["avg_dome"] = np.array([fl[0] + fl[3] * g[0], fl[1] + fl[3] * g[1], fl[2] + fl[3] * g[2], fl[4]], np.float64)
    np.savez(path, **out)
    return out


def _path_extent(P):
    ts = np.linspace(163.0, 187.5, 200)
    d = 0.0
    for T in ts:
        a, b = mobius(T, P)
        z = b / np.conj(a)
        d = max(d, 2 * math.atanh(min(abs(z), 0.999999)))
    return d


def _flags_table(S, P, sh):
    path = S.cache / f"flags_{VER_FLAGS}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    rho = _path_extent(P) + 5.4
    Gs = HY.enumerate_sectors(rho_max=rho)
    var, seed = HY.sector_table(Gs)
    out = dict(seed=seed, var=var, rho=np.array([rho]), G=Gs.reshape(-1, 4))
    np.savez(path, **out)
    print(f"[m09] flags: {len(var)} sector copies within rho {rho:.2f}", flush=True)
    return out


def _hash01f(x, y):
    v = np.sin(x * 12.9898 + y * 78.233) * 43758.5453
    return v - np.floor(v)


def _particles(S, P, sh, fl):
    """the tesserae that let go (tile centre, size, angle, colour, detach time), with the kernel's own timing."""
    path = S.cache / f"parts_{VER_PARTS}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    rng = np.random.default_rng(909)
    zref = HY.mob_apply(mobius(P["n09"], P), 0j)
    G = fl["G"]
    A, B, Cc, D = G[:, 0], G[:, 1], G[:, 2], G[:, 3]
    ctr = B / D
    dref = 2 * np.arctanh(np.minimum(np.abs((ctr - zref) / (1 - np.conj(zref) * ctr)), 0.9999999))
    starts = sh["starts"]
    dome = dict(z=[], z2=[], zeta2=[], size=[], col=[], gold=[], gt=[], td=[])
    for ci in np.nonzero(dref < 1.75)[0]:
        v = int(fl["var"][ci])
        i0, i1 = starts[v], starts[v + 1]
        pos = sh["Tpos"][i0:i1]
        zeta = (ART.S_X0 + pos[:, 0] / ART.S_RES) + 1j * (ART.S_Y0 + pos[:, 1] / ART.S_RES)
        ins = (np.abs(np.angle(zeta)) <= HY.AL) & (np.abs(zeta - HY.CC) >= HY.RR)
        if dref[ci] > 0.5:
            ins &= rng.random(len(zeta)) < 0.4          # the first corona: a share of its (tiny) tiles
        idx = np.nonzero(ins)[0]
        zt = zeta[idx]
        ang = sh["Tang"][i0:i1][idx]
        zz = (A[ci] * zt + B[ci]) / (Cc[ci] * zt + D[ci])
        e = zt + 1e-4 * np.exp(1j * ang)
        zz2 = (A[ci] * e + B[ci]) / (Cc[ci] * e + D[ci])
        seed = int(fl["seed"][ci])
        jit = ((seed * 2654435761) % 1000003) / 1000003.0
        rnd = sh["Trnd"][i0:i1][idx]
        rho = 2 * np.arctanh(np.minimum(np.abs((zz - zref) / (1 - np.conj(zref) * zz)), 0.9999999))
        td = P["heaven"] + 0.36 * rho + 0.35 * _hash01f(rnd * 3.1 + jit, jit * 5.7)
        dome["z"].append(zz)
        dome["z2"].append(zz2)
        dome["zeta2"].append(np.abs(zt) ** 2)
        dome["size"].append(sh["Tsize"][i0:i1][idx] / ART.S_RES)
        dome["col"].append(sh["Tcol"][i0:i1][idx])
        dome["gold"].append(sh["Tgold"][i0:i1][idx])
        dome["gt"].append(sh["Tgt"][i0:i1][idx])
        dome["td"].append(td)
    out = {("d_" + k): np.concatenate(v) for k, v in dome.items()}
    # the architecture (earth)
    K = HY.NVAR
    i0, i1 = starts[K], starts[K + 1]
    pos = sh["Tpos"][i0:i1]
    w = ((pos[:, 0] - ART.A_N / 2) + 1j * (pos[:, 1] - ART.A_N / 2)) / ART.A_RES
    keep = (np.abs(w) > 1.0) & (np.abs(w) < 2.4)
    rr = rng.random(len(w))
    keep &= np.where(sh["Tgold"][i0:i1] > 0.5, rr < 0.5, rr < 0.1)
    idx = np.nonzero(keep)[0]
    rnd = sh["Trnd"][i0:i1][idx]
    out["a_w"] = w[idx]
    out["a_ang"] = sh["Tang"][i0:i1][idx]
    out["a_size"] = sh["Tsize"][i0:i1][idx] / ART.A_RES
    out["a_col"] = sh["Tcol"][i0:i1][idx]
    out["a_gold"] = sh["Tgold"][i0:i1][idx]
    out["a_gt"] = sh["Tgt"][i0:i1][idx]
    out["a_td"] = P["earth"] + 1.05 * (2.3 - np.abs(w[idx])) + 0.35 * _hash01f(rnd * 3.1 + 0.5, 0.5 * 5.7)
    for pre in ("d_", "a_"):
        n = len(out[pre + "td"])
        out[pre + "r"] = rng.random((n, 5)).astype(np.float32)
    np.savez(path, **out)
    print(f"[m09] particles: {len(out['d_td'])} dome + {len(out['a_td'])} architecture", flush=True)
    return out


def _roll_v(T):
    return np.interp(T, _ROLL_T, _ROLL_V)


def _drift_minv(T, P, Mend_inv, z):
    """view coordinates at times T (array) of world points z under the dissolve drift M(T) = M_end o T(d, th)"""
    td = np.maximum(T - P["n09"], 0.0)
    d = V_DRIFT * (td - 0.8 * (1 - np.exp(-td / 0.8)))
    a, b = Mend_inv
    zz = (a * z + b) / (np.conj(b) * z + np.conj(a))
    ca, sb = np.cosh(d / 2), -np.sinh(d / 2) * np.exp(1j * TH_DRIFT)
    return (ca * zz + sb) / (np.conj(sb) * zz + ca)


def _tesserae_fall(fc, st, T):
    """premultiplied rgba of the falling tesserae, or None"""
    P = st["P"]
    pt = st["pt"]
    g3 = st["gold3"]
    life_max = 2.7
    xs, ys, hs, an, cols, al = [], [], [], [], [], []
    lxw = math.cos(T * 0.7)
    for pre in ("d_", "a_"):
        td = pt[pre + "td"]
        act = np.nonzero((td <= T) & (td > T - life_max))[0]
        if len(act) == 0:
            continue
        r = pt[pre + "r"][act]
        tdd = td[act]
        tau = T - tdd
        psi = _roll_v(tdd)
        if pre == "d_":
            wv = _drift_minv(tdd, P, st["Mend_inv"], pt["d_z"][act])
            wv2 = _drift_minv(tdd, P, st["Mend_inv"], pt["d_z2"][act])
            k = (1 - np.abs(wv) ** 2) / (1 - pt["d_zeta2"][act])
            spx = pt["d_size"][act] * F_PX * k
            ang0 = np.angle(wv2 - wv)
        else:
            wv = pt["a_w"][act]
            spx = pt["a_size"][act] * F_PX
            ang0 = pt["a_ang"][act]
        rot = np.exp(-1j * psi)
        p = wv * rot * F_PX                          # screen offset from the centre at the detach
        vf = 0.10 + 0.10 * r[:, 0]
        sc = 1.0 / np.maximum(1.0 - vf * tau, 0.08)
        swirl = np.exp(1j * (0.16 * (r[:, 1] - 0.5) * tau))
        wob = 5.0 * np.sin(2.3 * tau + 6.28 * r[:, 2]) * (1 - np.exp(-tau))
        q = p * sc * swirl + wob * np.exp(1j * 6.28 * r[:, 3])
        life = 1.5 + 1.2 * r[:, 4]
        a = np.clip((life - tau) / (0.55 * life), 0, 1) ** 1.5 * np.clip(1.6 - 0.04 * spx * sc, 0, 1)
        tumble = np.abs(np.cos(tau * (2.0 + 5.0 * r[:, 0]) + 6.28 * r[:, 1]))
        glint = np.clip(tumble - 0.9, 0, None) * 8.0
        gold = (pt[pre + "gold"][act] > 0.5)[:, None]
        gcol = g3[1][None, :] * pt[pre + "gt"][act] * (0.7 + 0.6 * tumble[:, None])
        base = np.where(gold, gcol, pt[pre + "col"][act] * (0.6 + 0.45 * tumble[:, None]))
        c = base * (1.0 + 0.25 * np.minimum(tau, 2.0))[:, None] + (glint[:, None] * np.array([1.0, 0.9, 0.7]))
        xs.append((CX + q.real) * fc.s)
        ys.append((CY + q.imag) * fc.s)
        hs.append(0.46 * spx * sc * fc.s)
        an.append(ang0 - psi + (r[:, 3] - 0.5) * 6.0 * tau)
        cols.append(c)
        al.append(a)
    if not xs:
        return None
    xs, ys, hs, an = (np.concatenate(v).astype(np.float64) for v in (xs, ys, hs, an))
    cols = np.concatenate(cols).astype(np.float64)
    al = np.concatenate(al).astype(np.float64)
    vis = (al > 0.003) & (hs > 0.25) & (xs > -60) & (xs < fc.w + 60) & (ys > -60) & (ys < fc.h + 60)
    o = np.nonzero(vis)[0]
    o = o[np.argsort(hs[o])]
    buf = np.zeros((fc.h, fc.w, 4), np.float32)
    HY.splat_tiles(buf, xs[o], ys[o], np.maximum(hs[o], 0.5), an[o], cols[o], al[o])
    return buf


def setup(S):
    tl = S.tl
    P = _plan(tl)
    sh = _bake_sheets(S)
    sh["anchors"] = np.array([ART.person_anchors(v) for v in range(HY.NVAR)], complex)   # pole base, dir, grip
    fl = _flags_table(S, P, sh)
    st = dict(P=P, sh=sh, fl=fl)
    st["pt"] = _particles(S, P, sh, fl)
    st["Mend_inv"] = HY.mob_inverse(mobius(P["n09"], P))
    st["gold3"] = np.stack([mz.SC["gold_d"], mz.SC["gold"], mz.SC["gold_l"]]).astype(np.float64)
    st["mortar3"] = np.asarray(mz.SC["mortar"], np.float64)
    _export_foley(S, P, sh, fl)
    return st


# ================================================================== foley
def _export_foley(S, P, sh, fl):
    PH = P["PH"]
    ev = [dict(t=163.0, kind="whoosh", strength=0.7, pan=0.0,
               desc="whip-tilt lands straight up into the dome (to 163.35), a soft stone settle"),
          dict(t=P["rays"][0], kind="rays", strength=0.6, pan=0.0,
               desc=f"gold rays descend from the oculus to the eight peoples (to {P['rays'][1]:.2f})"),
          dict(t=P["rays"][1], kind="glow", strength=0.6, pan=0.0,
               desc="the light reaches the peoples: the eight Flags beside them brighten")]
    w0, w1 = P["windows"]
    for k in range(8):
        tk = w0 + (w1 - w0) * ((k * 3) % 8) / 8.0
        a = HY.AL + k * 2 * HY.AL + roll(tk)
        ev.append(dict(t=tk, kind="chime", strength=0.35, pan=float(np.clip(math.cos(a), -1, 1)) * 0.8,
                       desc=f"drum window {k} lights up (glass catches the light)"))
    ev.append(dict(t=P["oculus"][0], kind="swell", strength=0.5, pan=0.0,
                   desc=f"the oculus swells with light (to {P['oculus'][1]:.2f})"))
    ev.append(dict(t=PH[0], kind="shimmer", strength=1.0, pan=0.0,
                   desc=f"'Flags be.' the dome opens into the infinite hyperbolic tiling (to {P['reveal'][1]:.2f})"))
    ev.append(dict(t=PH[0] + 0.1, kind="raise", strength=0.8, pan=0.0,
                   desc="every people raises its Flag: poles slide up and telescope (to 170.7)"))
    for k in range(16):
        tb = PH[0] + k * BEAT
        ev.append(dict(t=tb, kind="ring", strength=RING_ACCENT[k % 4] * (1.35 if k >= 12 else 1.0), pan=0.0,
                       desc="a ring of light races out to the circle at infinity; lamps, oculi and Flags ignite"))
    ev.append(dict(t=PH[1], kind="glide", strength=0.7, pan=0.0,
                   desc=f"'Flags are.' Moebius glide one tile over (to {PH[2]:.2f})"))
    ev.append(dict(t=PH[2], kind="glide", strength=0.85, pan=0.0,
                   desc=f"'Flags will.' loxodromic glide two tiles on, turning (to {PH[3]:.2f})"))
    ev.append(dict(t=PH[3], kind="bloom", strength=1.0, pan=0.0,
                   desc=f"'Flags.' the tiling spins, triple rings of light to infinity (to {P['hymn_end']:.2f})"))
    ev.append(dict(t=P["heaven"], kind="dissolve", strength=0.8, pan=0.0,
                   desc=f"heaven dissolves: the dome's tesserae let go from the oculus outward and drift down (to {P['n10']:.1f})"))
    ev.append(dict(t=P["earth"], kind="dissolve", strength=0.7, pan=0.0,
                   desc=f"earth dissolves: arches, pendentives, drum fall inward from the frame edges (to {P['away']:.2f})"))
    ev.append(dict(t=P["flags_word"], kind="glow", strength=0.6, pan=0.0,
                   desc="'...the ten thousand FLAGS remain': the floating flags flare"))
    ev.append(dict(t=P["completed"], kind="ring", strength=0.6, pan=0.0,
                   desc="'completed': a ring of light passes through the floating flags"))
    ev.append(dict(t=187.0, kind="flash", strength=1.0, pan=0.0,
                   desc="white-gold flood to full flag_glow (to 187.458), peak at the cut to m10"))
    foley.export_events(S, "sfx", ev)
    # a hero flag in the scene's wind (energy for the flutter of the field)
    wpx = [700.0 * wind(S.start + i / S.fps, P) for i in range(S.n)]
    path = S.cache / f"hero_{VER_HERO}.npz"
    if path.exists():
        from vx.flag import FlagTrack
        tr = FlagTrack.load(path)
    else:
        tr = Flag(pole=420, seed=9).simulate(S.n, lambda i: (CX, 900.0, 0.0), lambda i: (wpx[i], 0.0))
        tr.save(path)
    tr.export_foley(S, "flags", pan=np.zeros(S.n, np.float32))
    # the field: how much cloth is on screen x how hard it flies
    en = np.zeros(S.n, np.float32)
    for i in range(0, S.n):
        T = S.start + i / S.fps
        zb, zt = pole_world(T, P, sh, fl)
        a, b = HY.mob_inverse(mobius(T, P))
        wb = HY.mob_apply_v((a, b), zb) / zoom(T, P)
        wt = HY.mob_apply_v((a, b), zt) / zoom(T, P)
        L = np.abs(wt - wb) * F_PX
        vis = (np.abs(wb) < 1) & (L > 3)
        area = float(np.sum((0.375 * L[vis]) ** 2)) / (1920 * 1080 * 0.08)
        en[i] = min(1.0, area) * min(1.0, wind(T, P) / 1.0)
    foley.export_track(S, "field", en, 0.0, kind="field")


# ================================================================== render
def _flags(fc, st, T, M, Z, alpha_all=1.0):
    fl = st["fl"]
    P = st["P"]
    a, b = HY.mob_inverse(M)
    zb, zt = pole_world(T, P, st["sh"], fl)
    wb = HY.mob_apply_v((a, b), zb) / Z
    wt = HY.mob_apply_v((a, b), zt) / Z
    inside = (np.abs(wb) < 0.9995) & (np.abs(wt) < 0.9995)
    idx = np.nonzero(inside)[0]
    if len(idx) == 0:
        return None
    xb, yb, zb = w_to_screen(wb[idx], T)
    xt, yt, zt = w_to_screen(wt[idx], T)
    dx, dy = xt - xb, yt - yb
    L = np.hypot(dx, dy)
    cloth = 0.375 * L * fc.s
    keep = (cloth > 0.55) & (zb > 0) & (zt > 0) & (xb > -300) & (xb < 2220) & (yb > -300) & (yb < 1380)
    if not np.any(keep):
        return None
    xb, yb, dx, dy, L, cloth = xb[keep], yb[keep], dx[keep], dy[keep], L[keep], cloth[keep]
    sel = idx[keep]
    order = np.argsort(L)
    ang = np.arctan2(dx, -dy)
    alpha = np.clip((cloth - 0.55) / 1.4, 0, 1) * alpha_all
    cv = Canvas(fc)
    lx, ly, li = mz.candle(T, 0.5, 0.3, 0.2, seed=3)
    light = (-0.35 + 0.3 * (lx - 0.5), -0.75, 0.55)
    # every flag glows as the rings of light pass it (hyperbolic distance from the view centre)
    rho = 2 * np.arctanh(np.minimum(np.abs(wb[idx][keep] * Z), 0.9999999))
    R = rings(T, P)
    ring = np.zeros(len(rho))
    for r0, s0, w0 in R:
        ring += s0 * np.exp(-((rho - r0) / w0) ** 2)
    g = flag_glow(T, P) + (0.55 - 0.3 * smoothstep(P["n10"], P["n10"] + 0.5, T)) * (1 - np.exp(-ring))
    g = np.broadcast_to(g, L.shape)
    draw_flag_field(cv.ctx, xb[order], yb[order], L[order], T, seeds=fl["seed"][sel][order], wind=wind(T, P),
                    side=-1, ang=ang[order], alpha=alpha[order], light=light, glow=g[order], sort=False)
    return cv.rgba()


def render(fc, st):
    T = fc.T
    P = st["P"]
    sh = st["sh"]
    Z = zoom(T, P)
    M = mobius(T, P)
    (ma, mb) = M
    Hm = homography(fc, T)
    lx, ly, li = mz.candle(T, 0.45, -0.2, 0.3, seed=11)
    lxd, lyd = (lx - 0.5) * 2.0, (ly + 0.2) * 1.6
    n = math.hypot(lxd, lyd) + 1e-6
    swell = 0.25 * smoothstep(P["oculus"][0], P["oculus"][1], T) * (1 - smoothstep(P["PH"][0], P["PH"][0] + 1.0, T))
    lit_w = smoothstep(P["windows"][0], P["windows"][1] + 0.3, T)
    wake = 0.5 * smoothstep(P["rays"][0], P["rays"][1] + 0.4, T) + 0.5 * lit_w      # the dome wakes with the cantor
    dis_on = 1.0 if T > P["n09"] else 0.0
    zr = HY.mob_apply(mobius(P["n09"], P), 0j)
    bg_i = 0.80 + 0.12 * smoothstep(P["n10"], P["completed"] + 0.6, T) + 0.2 * smoothstep(186.3, 187.4, T)
    bg_r = 0.55 + 0.55 * smoothstep(P["n09"], P["n10"] + 1.5, T) + 1.8 * smoothstep(186.2, 187.3, T)
    prm = np.array([
        Z, roll(T), li * (0.78 + 0.2 * wake + swell), lxd / n, lyd / n,
        1.0, 0.60 + 0.31 * wake, 0.14 + 0.30 * lit_w, 0.30 + 0.25 * swell * 4,
        1.25 * smoothstep(P["rays"][0], P["rays"][1], T), T,
        P["heaven"], 0.36,
        P["earth"], 1.05, 2.3,
        ART.S_X0, ART.S_Y0, ART.S_RES,
        ART.A_RES, ART.A_N / 2, ART.A_N / 2, bg_i, bg_r,
        1.0, 0.22 * smoothstep(P["PH"][0] + 1.2, P["PH"][0] + 2.8, T), 0.9, dis_on], np.float64)
    out = np.empty((fc.h, fc.w, 3), np.float32)
    fl_ = smoothstep(186.35, 187.35, T)
    core = np.array([0.92, 0.85, 0.66]) * (1 - fl_) + np.array(FLAG_GLOW) * fl_
    rim = np.array([0.30, 0.19, 0.08]) * (1 - fl_) + np.array(FLAG_GLOW) * fl_
    bgcol = np.stack([core, rim]).astype(np.float64)
    HY.render_frame(out, Hm, prm, ma.real, ma.imag, mb.real, mb.imag, zr.real, zr.imag,
                    sh["s_lab"], sh["s_rim"], sh["s_u"], sh["s_v"], sh["s_atl"], sh["s_lvl"],
                    sh["a_lab"], sh["a_rim"], sh["a_u"], sh["a_v"], sh["a_atl"], sh["a_lvl"],
                    sh["Tcol"], sh["Tgold"], sh["Tgt"], sh["Ttilt"], sh["Tray"], sh["Temit"], sh["Trnd"], sh["Tgrp"],
                    emission(T, P), rings(T, P), st["gold3"], st["mortar3"], bgcol, sh["avg_dome"])
    img = out
    if T > P["heaven"]:
        fall = _tesserae_fall(fc, st, T)
        if fall is not None:
            img = over(img, fall)
    fla = _flags(fc, st, T, M, Z)
    if fla is not None:
        img = over(img, fla)
    # the whip lands: vertical motion blur from the pitch rate
    _, rate = pitch(T)
    blur = abs(rate) * F_PX / 24.0 * 1.3 * fc.s
    if blur > 1.5:
        k = int(blur) | 1
        img = cv2.blur(img, (1, k))
    return np.clip(img, 0, None)


def post(fc, st):
    T = fc.T
    f = smoothstep(187.0, 187.41, T) ** 1.3
    if T >= 187.41:
        f = 1.0
    # the eye adapts: from m08's lit wall (the whip) into the candlelit dome
    d = dict(fade=f, fade_color=FLAG_GLOW, exposure=1.0 + 0.9 * (1.0 - smoothstep(163.0, 163.85, T)))
    if f > 0:
        d["grain"] = POST["grain"] * (1 - f)
    return d

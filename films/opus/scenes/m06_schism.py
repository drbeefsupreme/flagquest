"""m06 - RECURSIVE BALKANIZATION (106.5-131.5). Owner: M06.

One continuous plunge down through the recursion. The Orbis Terrarum (a T-O mappa mundi in tesserae) cracks
along the T, then every border, then again and again; each last piece becomes a kingdom of one (X01, X02).
The camera then dives into a single gold tessera of a crown, which is itself a mosaic: the congregation of
saints, which splits into churches of one (X03-X05, ANATHEMA SIT); dive again into one tessera of a tiara:
the philosophers under the tree, whose floor breaks into rocks drifting in the void (X06, X07). K07: every
tessera splits into four, and those into four... glittering dust swirling on the dark (hand-off to m07).

Modules: m06_tess (moving-tile engine, quadtree LOD, shatter), m06_map (nations), m06_faith (faiths);
figures by vx.icon.
"""
import math

import cv2
import numpy as np

from vx import *
from vx import mosaic as mz
from vx.canvas import col

from scenes import m06_tess as tx
from scenes import m06_map as mp
from scenes import m06_faith as fa
from scenes import m06_philo as ph
from scenes import m06_endless as en
from scenes import m06_clipeus as cl

POST = dict(bloom=0.32, bloom_thresh=0.74, grain=0.018, vignette=0.32)

T0, T1 = 106.5, 131.5
GLOW = np.array(col("flag_glow"), np.float32)
DIVE1 = (113.08, 113.7)          # into the gold tessera of CITIZEN2's crown -> the congregation
DIVE2 = (119.28, 120.0)          # into the knob of H3's tiara -> the philosophers
REV_A = (128.3, 128.68)          # "May it schism endlessly": back OUT of the tiara (philosophers -> churches)
REV_B = (128.68, 129.12)         # ... and out of the crown (churches -> the archipelago of kingdoms)
END_ZOOM = 1.55                  # the whole world of kingdoms, where everything turns to dust
K_IN = 1920.0                    # inner-panel units across one portal tessera (the frame width)


def setup(S):
    st = {"nat": mp.NationPanel(S), "faith": fa.FaithPanel(S), "philo": ph.PhiloPanel(S)}
    st["dust"] = en.Dust(st["nat"], cam_end(st, en.T_DUST))
    st["clip"] = cl.Clipeus(S.tl)
    st["dust"].export_handoff(S.cache / "handoff_dust.npz")
    export_sound(S, st)
    return st


# ============================================================ sound: every split, crack, dive and titulus
def export_sound(S, st):
    from vx.foley import export_events
    nat, faith, philo = st["nat"], st["faith"], st["philo"]
    sfx, mus = [], []

    def ev(lst, t, kind, strength, pan=0.0, desc=""):
        lst.append(dict(t=float(t), kind=kind, strength=float(np.clip(strength, 0, 1)), pan=float(np.clip(pan, -1, 1)),
                        desc=desc))
    # NATIONS: the orbis cracks along the T, then along every border, again and again
    names = {1: "the T of the world (three continents)", 2: "every national border", 3: "provinces",
             4: "counties", 5: "kingdoms of one"}
    for g in range(1, nat.G + 1):
        t0 = float(np.min(nat.sh.t0[g]))
        ev(sfx, t0 - 0.07, "crack", 1.0 - 0.12 * (g - 1), 0.0, f"level={g} nations: {names[g]} crack (light in the joints)")
        ev(sfx, t0, "drift", 0.8 - 0.1 * g, 0.0, f"level={g} the pieces jolt apart and drift like islands (to {t0 + 2.6:.2f})")
        ev(mus, t0, "split", g, 0.0, f"section=nations level={g}")
    # torn-out tesserae tumbling at each crack wave
    for g in range(1, 3):
        ev(sfx, float(np.min(nat.sh.t0[g])) + 0.08, "tinkle", 0.5, 0.0, f"level={g} loose tesserae torn out, tumbling (to {float(np.min(nat.sh.t0[g])) + 1.1:.2f})")
    # every kingdom of one sets itself (crowned figures springing up), clustered
    pops = np.sort(nat.k_pop[nat.k_leaf])
    for q in np.array_split(pops, 8):
        if len(q):
            ev(sfx, q[0], "tile_set", 0.45, 0.0, f"{len(q)} kingdoms of one set themselves (crowns glint) (to {q[-1]:.2f})")
    ev(sfx, *DIVE1[:1], "dive", 0.9, 0.0, f"the camera plunges into one gold tessera of CITIZEN2's crown (to {DIVE1[1]})")
    ev(mus, DIVE1[0], "dive", 1.0, 0.0, f"into the crown tessera (to {DIVE1[1]})")
    # FAITHS
    for g, tg in fa.T_SPLIT.items():
        ev(sfx, tg - 0.05, "crack", 1.0 - 0.15 * (g - 1), 0.0, f"level={g} faiths: the one church splits ({['East/West', 'cabals', 'ones'][g - 1]})")
        ev(sfx, tg, "drift", 0.6, 0.0, f"level={g} faiths: halves / cabals / ones slide apart (to {tg + 0.7:.2f})")
        ev(mus, tg, "split", g, 0.0, f"section=faiths level={g}")
    ev(sfx, fa.T_ICON + 0.05, "fall", 0.7, 0.0, f"the ragged pieces of the one mosaic crumble away (to {fa.T_ICON + 1.3:.2f})")
    for i in range(len(fa.SAINTS)):
        x = fa.GRID[i][0]
        ev(sfx, faith.icon_t[i], "tile_set", 0.4, (x - 960) / 1300.0, f"an icon of one sets itself (dome, columns, tiny altar), saint {i}")
    for (t0, t1), lr in ((fa.T_ANA1, 1.0), (fa.T_ANA2, -1.0)):
        ev(sfx, t0, "whoosh", 0.8, -0.3 * lr, f"ANATHEMA SIT: a titulus of tesserae flies {'left->right' if lr > 0 else 'right->left'} (to {t1:.2f})")
        ev(mus, t0, "anathema", 1.0, -0.3 * lr, f"ANATHEMA SIT launched (to {t1:.2f})")
        ev(sfx, t1, "shatter", 0.7, 0.3 * lr, "the letters shatter against the halo")
    ev(sfx, fa.T_ANA1[1] + 0.02, "halo_split", 0.9, 0.2, "HERETIC2's halo cracks in two")
    ev(sfx, fa.T_ANA2[1] + 0.02, "halo_split", 0.9, -0.2, "HERETIC1's halo cracks in two")
    ev(mus, fa.T_ANA1[1], "split", 1, 0.2, "section=faiths halo")
    ev(mus, fa.T_ANA2[1], "split", 1, -0.2, "section=faiths halo")
    ev(sfx, fa.T_CROWN[1], "tile_set", 0.8, 0.25, "HERETIC3 crowns himself pope: the tiara lands")
    for i in range(len(fa.SAINTS)):
        if i not in fa.HEROES:
            ev(sfx, faith.tiara_t[i], "tile_set", 0.3, (fa.GRID[i][0] - 1380) / 1300.0, f"a tiara pops onto saint {i}")
    ev(sfx, DIVE2[0], "dive", 0.9, 0.0, f"the camera plunges into the knob of the pope's tiara (to {DIVE2[1]})")
    ev(mus, DIVE2[0], "dive", 1.0, 0.0, f"into the tiara tessera (to {DIVE2[1]})")
    # PHILOSOPHIES
    ev(sfx, ph.T_BREAK - 0.05, "crack", 1.0, 0.0, "level=1 philosophies: the floor of the Academy breaks into rocks")
    ev(sfx, ph.T_BREAK, "float", 0.7, 0.0, f"the rocks float apart into the void (to {ph.T_DRIFT[1]:.2f})")
    ev(sfx, ph.T_CRUMBLE[0], "float", 0.5, 0.0, f"the gold dissolves into drifting tesserae (to {ph.T_CRUMBLE[1] + 2.0:.2f})")
    ev(mus, ph.T_BREAK, "split", 1, 0.0, "section=philo level=1 floor")
    ev(sfx, ph.T_POETS, "crack", 0.6, -0.2, "level=2 philosophies: CRACKPOT1's rock splits again at 'poets'")
    ev(mus, ph.T_POETS, "split", 2, -0.2, "section=philo level=2")
    for w_, tw in ph.SVM:
        ev(sfx, tw - 0.05, "tile_set", 0.5, 0.35, f"{w_}: letters of tesserae set themselves in the void")
    # ENDLESS: back out through the recursion, then every tessera splits into four, and those into four...
    ev(sfx, REV_A[0], "dive", 0.8, 0.0, f"reverse: out of the tiara tessera (to {REV_A[1]})")
    ev(sfx, REV_B[0], "dive", 0.8, 0.0, f"reverse: out of the crown tessera (to {REV_B[1]})")
    ev(mus, REV_A[0], "dive", 1.0, 0.0, f"reverse flight out through the recursion (to {REV_B[1]})")
    for g, tg in enumerate(en.T_GEN, start=1):
        ev(sfx, tg, "split", 1.0 - 0.1 * (g - 1), 0.0, f"level={g} endless: every tessera in frame splits into four")
        ev(mus, tg, "split", g, 0.0, f"section=endless level={g}")
    ev(sfx, en.T_SWIRL[0], "swirl", 0.8, 0.0, "glittering dust swirling counter-clockwise, rising (to 131.5)")
    export_events(S, "sfx", sfx)
    export_events(S, "splits", mus)


# ============================================================ cameras (global time)
def _lz(T, pts):
    """piecewise log-zoom with smooth easing between keys [(T, zoom), ...]"""
    if T <= pts[0][0]:
        return pts[0][1]
    for (ta, za), (tb, zb) in zip(pts, pts[1:]):
        if T <= tb:
            u = smootherstep(ta, tb, T)
            return math.exp(math.log(za) + (math.log(zb) - math.log(za)) * u)
    return pts[-1][1]


def _lp(T, pts):
    """piecewise smooth point path [(T, (x, y)), ...]"""
    if T <= pts[0][0]:
        return np.array(pts[0][1], np.float64)
    for (ta, a), (tb, b) in zip(pts, pts[1:]):
        if T <= tb:
            u = smootherstep(ta, tb, T)
            return np.array(a) * (1 - u) + np.array(b) * u
    return np.array(pts[-1][1], np.float64)


def cam_nations(st, T):
    nat = st["nat"]
    z = _lz(T, [(106.5, 1.10), (106.95, 1.0), (107.25, 1.04), (108.2, 2.0), (108.6, 2.8), (109.08, 3.3),
                (109.42, 14.0), (111.5, 15.8), (111.7, 10.0), (111.95, 15.0), (113.0, 16.0)])
    # frame the hero's figure (feet on his island), then whip to his neighbour for X02
    f1 = nat.k_feet[nat.hero] - np.array([0.0, 0.5 * mp.KING_H])
    f2 = nat.k_feet[nat.hero2] - np.array([0.0, 0.5 * mp.KING_H])
    h1, _ = nat.leaf_point(nat.hero, T, f1)
    h2, _ = nat.leaf_point(nat.hero2, T, f2)
    k = smootherstep(111.55, 111.9, T)
    hp = h1 * (1 - k) + h2 * k
    w = smootherstep(107.2, 108.95, T)
    c = mp.C0 * (1 - w) + hp * w
    return tx.view_matrix(c[0], c[1], z, 0.0)


def cam_faith(st, T):
    fp = st["faith"]
    h1, _, _ = fp.saint_pos(8, T)
    h2, _, _ = fp.saint_pos(9, T)
    h3, _, _ = fp.saint_pos(10, T)
    two = (h1 + h2) / 2 - np.array([0.0, 285.0])
    c = _lp(T, [(113.7, (960, 540)), (114.75, (960, 548)), (115.4, (960, 760)), (115.5, (960, 760)),
                (115.78, tuple(two)), (117.5, tuple(two)), (117.82, tuple(h3 - np.array([60.0, 285.0]))),
                (118.5, tuple(h3 - np.array([60.0, 285.0]))), (118.95, tuple(h3 - np.array([30.0, 320.0])))])
    z = _lz(T, [(113.7, 1.0), (114.75, 1.05), (115.4, 0.62), (115.5, 0.62), (115.78, 1.45), (117.5, 1.5),
                (117.82, 1.75), (118.5, 1.8), (118.95, 1.15)])
    return tx.view_matrix(c[0], c[1], z, 0.0)


def cam_philo(st, T):
    pp = st["philo"]
    p1, h1 = pp.phil_pos(2, T)
    p2, h2 = pp.phil_pos(5, T)
    f1 = tuple(p1 - np.array([-20.0, 0.52 * h1]))
    f2 = tuple(p2 - np.array([-120.0, 0.55 * h2]))
    c = _lp(T, [(120.0, (960, 540)), (121.05, (960, 548)), (122.7, (960, 560)), (123.05, (960, 560)),
                (123.35, f1), (126.2, f1), (126.55, f2), (128.1, f2), (128.3, (960, 540))])
    z = _lz(T, [(120.0, 1.0), (121.05, 1.04), (122.7, 0.6), (123.05, 0.6), (123.35, 1.75), (126.2, 1.9),
                (126.55, 1.8), (128.1, 1.9), (128.3, 1.0)])
    return tx.view_matrix(c[0], c[1], z, 0.0)


def cam_end(st, T):
    """the archipelago of kingdoms around the queen's island, where everything turns to dust (static)."""
    nat = st["nat"]
    c, _ = nat.leaf_point(nat.hero2, REV_B[1], nat.k_feet[nat.hero2] - np.array([0.0, 0.5 * mp.KING_H]))
    return tx.view_matrix(c[0], c[1], END_ZOOM, 0.0)


# ============================================================ the dive into one tessera
def _square_angle(th):
    """a square tile looks the same every 90 degrees: the smallest rotation that stands it upright."""
    return (th + math.pi / 4) % (math.pi / 2) - math.pi / 4


def dive_views(T, t0, t1, V0, portal):
    """outer view V1 plunging into the portal tessera, and the inner panel's view V2 (inner units -> screen)."""
    u = clamp((T - t0) / (t1 - t0))
    c, th, side = portal
    th = _square_angle(th)
    side_in = side * 0.84
    z0 = math.hypot(V0[0, 0], V0[1, 0])
    z1 = K_IN / side_in
    q = u * u * u * (u * (u * 6 - 15) + 10)
    z = math.exp(math.log(z0) + (math.log(z1) - math.log(z0)) * q)
    inv = np.linalg.inv(np.vstack([V0, [0, 0, 1]]))
    c0 = (inv @ np.array([960.0, 540.0, 1.0]))[:2]
    w = smootherstep(0.0, 0.42, u)
    cc = c0 * (1 - w) + np.asarray(c) * w
    rot = -th * smootherstep(0.12, 0.95, u)
    V1 = tx.view_matrix(cc[0], cc[1], z, rot)
    a = side_in / K_IN
    ca, sa = math.cos(th) * a, math.sin(th) * a
    A = np.array([[ca, -sa, c[0] - (ca * 960 - sa * 540)], [sa, ca, c[1] - (sa * 960 + ca * 540)]])
    V2 = tx.compose(V1[None], A[None])[0]
    return V1, V2, u, side_in


def render_dive(fc, st, T, t0, t1, outer, V0, inner, light, tint, reverse=False, lam=0.0):
    img = np.zeros((fc.h, fc.w, 3), np.float32)
    portal = outer.portal(T)
    if reverse:          # out of the tessera: the same flight, played backwards
        T_ = t1 - (T - t0)
    else:
        T_ = T
    V1, V2, u, side_in = dive_views(T_, t0, t1, V0, portal)
    outer.render(fc, img, V1, T, light, portal=True, lam_extra=lam)
    z2 = math.hypot(V2[0, 0], V2[1, 0])
    # the window: the portal tessera's glass, on screen
    c, th, side = portal
    th = _square_angle(th)
    hs = side_in / 2
    corners = np.array([[-hs, -hs], [hs, -hs], [hs, hs], [-hs, hs]])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    wp = corners @ R.T + np.asarray(c)
    sp = wp @ V1[:, :2].T + V1[:, 2]
    size_px = side_in * math.hypot(V1[0, 0], V1[1, 0])
    a = smoothstep(26.0, 380.0, size_px)
    if a <= 0.001:
        return img
    # the inner picture: rendered straight through V2 when it is at least full size, else offscreen + warped
    inner_img = np.zeros_like(img)
    if z2 >= 0.5:
        inner.render(fc, inner_img, V2, T, light, lam_extra=lam)
    else:
        # the whole square inner wall at zoom 0.5 (960 px across), then shrunk into the tessera
        full = np.zeros_like(img)
        Voff = tx.view_matrix(960.0, 540.0, 0.5)
        inner.render(fc, full, Voff, T, light, lam_extra=lam)
        Mp = tx.compose(V2[None], np.linalg.inv(np.vstack([Voff, [0, 0, 1]]))[None, :2])[0]
        f = min(1.0, max(z2 * 2.0 * 1.4, 1.0 / 64))
        if f < 1.0:
            full = cv2.resize(full, (max(2, int(fc.w * f)), max(2, int(fc.h * f))), interpolation=cv2.INTER_AREA)
            Mp[:, :2] = Mp[:, :2] / f
        Mp[:, 2] *= fc.s
        inner_img = cv2.warpAffine(full, Mp, (fc.w, fc.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=tuple(float(v) for v in tint))
    # seen through coloured glass at first
    k = a * a
    inner_img = inner_img * (np.asarray(tint, np.float32) * (1 - k) * 1.25 + k)
    m = np.zeros((fc.h, fc.w), np.uint8)
    cv2.fillConvexPoly(m, np.round(sp * fc.s * 16).astype(np.int32), 255, lineType=cv2.LINE_AA, shift=4)
    m = (m.astype(np.float32) / 255.0 * a)[..., None]
    return img * (1 - m) + inner_img * m


# ============================================================ render
def render(fc, st):
    T = fc.T
    light = mz.candle(T, seed=6)
    nat, faith, philo = st["nat"], st["faith"], st["philo"]
    if T < DIVE1[0]:
        img = np.zeros((fc.h, fc.w, 3), np.float32)
        nat.render(fc, img, cam_nations(st, T), T, light)
    elif T < DIVE1[1]:
        rgb, g = nat.portal_colour()
        tint = mz.SC["gold"] if g > 0.5 else rgb
        img = render_dive(fc, st, T, DIVE1[0], DIVE1[1], nat, cam_nations(st, DIVE1[0]), faith, light, tint)
    elif T < DIVE2[0]:
        img = np.zeros((fc.h, fc.w, 3), np.float32)
        faith.render(fc, img, cam_faith(st, T), T, light)
    elif T < DIVE2[1]:
        img = render_dive(fc, st, T, DIVE2[0], DIVE2[1], faith, cam_faith(st, DIVE2[0]), philo, light,
                          mz.SC["gold"])
    elif T < REV_A[0]:
        img = np.zeros((fc.h, fc.w, 3), np.float32)
        philo.render(fc, img, cam_philo(st, T), T, light)
    elif T < REV_A[1]:
        img = render_dive(fc, st, T, REV_A[0], REV_A[1], faith, tx.view_matrix(960.0, 540.0, 1.0), philo, light,
                          mz.SC["gold"], reverse=True)
    elif T < REV_B[1]:
        rgb, g = nat.portal_colour()
        img = render_dive(fc, st, T, REV_B[0], REV_B[1], nat, cam_end(st, T), faith, light,
                          mz.SC["gold"] if g > 0.5 else rgb, reverse=True, lam=en.lam_extra(T))
    elif T < en.T_DUST:
        img = np.zeros((fc.h, fc.w, 3), np.float32)
        nat.render(fc, img, cam_end(st, T), T, light, lam_extra=en.lam_extra(T))
    else:
        dust = st["dust"]
        img = np.empty((fc.h, fc.w, 3), np.float32)
        img[...] = dust.background(T, mz.SC["mortar"] * 0.5)
        dust.render(fc, img, T)
    st["clip"].render(fc, img, T, light)
    g = 1.0 - ease_out(clamp((T - 106.5) / 0.6), 2.5)
    if g > 0:          # the breach's light drains away (screen blend keeps it luminous, not grey)
        img = 1.0 - (1.0 - np.clip(img, 0, 1)) * (1.0 - GLOW * g)
    return img


def post(fc, st):
    return {}

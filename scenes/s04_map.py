"""s04 — the Geomantic map-table texture: a polar azimuthal-equidistant map of the real Earth
(global_land_mask) inside a basalt pentagon, gold ley lines (great circles between geomantic nodes, forming
overlapping pentagrams, as in the real Geomantic Command Center), and a beige slop-crack crazing layer.

Textures are square, cover the table's circumcircle, and map to world (X, Z) on the table top:
  u = N/2 + X * N/(2 Rt),   v = N/2 - Z * N/(2 Rt)       (+Z = far side = top of texture)
"""
import math

import cairo
import cv2
import numpy as np

from vx.canvas import col

N = 1600                  # texture size (px)
MAP_R = 0.735             # map disc radius / table circumradius (pentagon inradius is 0.809)
LAT_EDGE = -58.0          # latitude at the disc rim

# geomantic nodes (lat, lon)
NODES = {
    "atlanta": (33.75, -84.39), "blackrock": (40.79, -119.2), "teotihuacan": (19.69, -98.84),
    "machu": (-13.16, -72.55), "reykjavik": (64.1, -21.9), "giza": (29.98, 31.13),
    "stonehenge": (51.18, -1.83), "delphi": (38.48, 22.50), "zimbabwe": (-20.27, 30.93),
    "kailash": (31.07, 81.31), "uluru": (-25.34, 131.04), "angkor": (13.41, 103.87),
    "fuji": (35.36, 138.73), "tasmania": (-42.0, 147.0), "easter": (-27.11, -109.35),
    "baikal": (53.5, 108.0), "capetown": (-33.9, 18.4), "hawaii": (19.8, -155.5),
    "svalbard": (78.2, 15.6), "fuego": (-54.8, -68.3), "madagascar": (-18.9, 47.5),
}
STARS = [
    ["atlanta", "blackrock", "teotihuacan", "machu", "reykjavik"],
    ["giza", "stonehenge", "delphi", "zimbabwe", "kailash"],
    ["uluru", "angkor", "fuji", "tasmania", "baikal"],
    ["atlanta", "giza", "kailash", "uluru", "easter"],
    ["svalbard", "hawaii", "fuego", "capetown", "madagascar"],
]


def ll_to_uv(lat, lon, n=N):
    """(lat, lon) degrees -> texture pixel (u, v)"""
    R = n / 2 * MAP_R
    r = (90.0 - lat) / (90.0 - LAT_EDGE) * R
    a = math.radians(lon)
    return n / 2 + r * math.sin(a), n / 2 + r * math.cos(a)


def uv_to_world(u, v, Rt, n=N):
    """texture pixel -> world (X, Z) on a table of circumradius Rt"""
    k = 2 * Rt / n
    return (u - n / 2) * k, -(v - n / 2) * k


def ley_segments():
    """list of polylines [(u,v),...] for every pentagram edge (straight chords on the map, as in the
    Geomantic Command Center game: a virtual ley line is drawn between every two Flags)"""
    segs = []
    for grp in STARS:
        pts = [NODES[g] for g in grp]
        uv = [ll_to_uv(*p) for p in pts]
        cu = sum(p[0] for p in uv) / 5
        cv_ = sum(p[1] for p in uv) / 5
        order = sorted(range(5), key=lambda i: math.atan2(uv[i][0] - cu, uv[i][1] - cv_))
        for j in range(5):
            segs.append([uv[order[j]], uv[order[(j + 2) % 5]]])
    return segs


def _pent_uv(scale=1.0, n=N):
    a = math.pi + np.arange(5) * 2 * math.pi / 5
    r = n / 2 * scale
    return [(n / 2 + r * math.sin(x), n / 2 - r * math.cos(x)) for x in a]


def _land_mask(n=N, ss=2):
    from global_land_mask import globe
    m = n * ss
    R = m / 2 * MAP_R
    yy, xx = np.mgrid[0:m, 0:m].astype(np.float64)
    dx, dy = xx + 0.5 - m / 2, yy + 0.5 - m / 2
    r = np.hypot(dx, dy)
    lat = 90.0 - r / R * (90.0 - LAT_EDGE)
    lon = np.degrees(np.arctan2(dx, dy))
    ok = lat >= -89.9
    lat_c = np.clip(lat, -89.9, 89.9)
    land = np.zeros((m, m), np.float32)
    land[ok] = globe.is_land(lat_c[ok], lon[ok]).astype(np.float32)
    land[r > R] = 0
    land = cv2.resize(land, (n, n), interpolation=cv2.INTER_AREA)
    return land


def _to_rgba(surf, n):
    surf.flush()
    buf = np.ndarray((n, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :n]
    out = np.empty((n, n, 4), np.float32)
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = buf[..., 2], buf[..., 1], buf[..., 0], buf[..., 3]
    return out / 255.0


def build(cache_dir):
    """-> dict(base=(N,N,4) premult, ley=(N,N,3) additive, cracks=(N,N,4) premult, crack_glow=(N,N,3))"""
    path = cache_dir / "map_tex_v4.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k].astype(np.float32) / 255.0 for k in z.files}
    n = N
    land = _land_mask(n)
    R = n / 2 * MAP_R
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(xx + 0.5 - n / 2, yy + 0.5 - n / 2) / R
    disc = np.clip((1.0 - r) * R, 0, 1)[..., None]

    # ---------------- base: basalt pentagon frame (cairo)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, n, n)
    ctx = cairo.Context(surf)
    P = _pent_uv(1.0)
    ctx.move_to(*P[0])
    for p in P[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    g = cairo.RadialGradient(n / 2, n / 2, n * 0.2, n / 2, n / 2, n * 0.55)
    g.add_color_stop_rgb(0, *col("#1A1E2C"))
    g.add_color_stop_rgb(1, *col("#0E1019"))
    ctx.set_source(g)
    ctx.fill()
    # inlay: inset pentagon + gold line
    for sc, w, a in ((0.95, 5, 0.9), (0.915, 2, 0.6)):
        Q = _pent_uv(sc)
        ctx.move_to(*Q[0])
        for p in Q[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.set_source_rgba(*col("#B98A45"), a)
        ctx.set_line_width(w)
        ctx.stroke()
    # degree ring
    ctx.set_source_rgba(*col("#B98A45"), 0.85)
    ctx.set_line_width(3)
    ctx.arc(n / 2, n / 2, R + 14, 0, 2 * math.pi)
    ctx.stroke()
    for k in range(360 // 5):
        a = math.radians(k * 5)
        l0 = R + 14
        l1 = R + (34 if k % 6 == 0 else 24)
        ctx.move_to(n / 2 + l0 * math.sin(a), n / 2 + l0 * math.cos(a))
        ctx.line_to(n / 2 + l1 * math.sin(a), n / 2 + l1 * math.cos(a))
        ctx.set_line_width(2.5 if k % 6 == 0 else 1.4)
        ctx.stroke()
    # corner sockets
    for p in _pent_uv(0.84):
        ctx.arc(p[0], p[1], 16, 0, 2 * math.pi)
        ctx.set_source_rgba(*col("#B98A45"), 0.8)
        ctx.set_line_width(3)
        ctx.stroke()
        ctx.arc(p[0], p[1], 6, 0, 2 * math.pi)
        ctx.fill()
    frame = _to_rgba(surf, n)

    # ---------------- ocean + land (numpy)
    rr = np.clip(r, 0, 1)[..., None]
    ocean = (np.array(col("#123047"), np.float32) * (1 - rr) + np.array(col("#0A1826"), np.float32) * rr)
    # graticule
    lat = 90.0 - r * (90.0 - LAT_EDGE)
    lon = np.degrees(np.arctan2(xx + 0.5 - n / 2, yy + 0.5 - n / 2))
    gl = np.minimum(np.abs(((lat + 7.5) % 15) - 7.5) / (15 * 0.004 * (1 + r)), 1)
    gm = np.minimum(np.abs(((lon + 7.5) % 15) - 7.5) / (15 * 0.003 * (1 + 2 * np.maximum(r, 0.05)) / np.maximum(r, 0.05) * 0.2), 1)
    grat = (1 - np.minimum(gl, gm))[..., None] * 0.45 * (r < 1)[..., None]
    ocean = ocean + grat * np.array(col("#2B5670"), np.float32) * 0.8
    land_s = cv2.GaussianBlur(land, (0, 0), 0.8)
    coast = np.clip(cv2.GaussianBlur(land, (0, 0), 1.6) * 2.4 - land_s * 2.2, 0, 1)
    rng = np.random.default_rng(7)
    tex = cv2.GaussianBlur(rng.random((n // 8, n // 8)).astype(np.float32), (0, 0), 1.5)
    tex = cv2.resize(tex, (n, n), interpolation=cv2.INTER_CUBIC)[..., None]
    land_c = np.array(col("#3D5E6E"), np.float32) * (0.85 + 0.3 * tex) + np.array(col("#6F97A6"), np.float32) * 0.0
    L = land[..., None]
    mapc = ocean * (1 - L) + land_c * L + coast[..., None] * np.array(col("#8DC1CF"), np.float32) * 0.7
    base = frame.copy()
    base[..., :3] = base[..., :3] * (1 - disc) + mapc * disc
    base[..., 3:] = np.maximum(base[..., 3:], disc)

    # ---------------- ley lines (additive)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, n, n)
    ctx = cairo.Context(surf)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for seg in ley_segments():
        ctx.move_to(*seg[0])
        for p in seg[1:]:
            ctx.line_to(*p)
        ctx.set_source_rgba(*col("#FFB24A"), 1.0)
        ctx.set_line_width(3.2)
        ctx.stroke()
    for nm, (la, lo) in NODES.items():
        u, v = ll_to_uv(la, lo)
        ctx.arc(u, v, 7, 0, 2 * math.pi)
        ctx.set_source_rgba(*col("#FFD9A0"), 1)
        ctx.fill()
    ley = _to_rgba(surf, n)[..., :3]
    ley = ley * 0.9 + cv2.GaussianBlur(ley, (0, 0), 9) * 1.6

    # ---------------- slop crazing (Voronoi cracks, jagged)
    from scipy.spatial import Voronoi
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, n, n)
    ctx = cairo.Context(surf)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.arc(n / 2, n / 2, R, 0, 2 * math.pi)
    ctx.clip()
    for level, (count, wid, alpha) in enumerate(((38, 3.6, 1.0), (170, 1.3, 0.7))):
        pts = rng.random((count, 2)) * n * 1.1 - n * 0.05
        vor = Voronoi(pts)
        for (a, b) in vor.ridge_vertices:
            if a < 0 or b < 0:
                continue
            if level == 1 and rng.random() < 0.5:
                continue
            p0, p1 = vor.vertices[a], vor.vertices[b]
            d = p1 - p0
            ln = float(np.hypot(*d))
            if ln < 1:
                continue
            k = max(2, int(ln / 14))
            nrm = np.array([-d[1], d[0]]) / ln
            ctx.move_to(*p0)
            for i in range(1, k):
                q = p0 + d * (i / k) + nrm * rng.normal(0, 3.2 + level * 0.5)
                ctx.line_to(*q)
            ctx.line_to(*p1)
            ctx.set_source_rgba(*col("#C9C0A8"), alpha)
            ctx.set_line_width(wid * (0.6 + 0.8 * rng.random()))
            ctx.stroke()
    cracks = _to_rgba(surf, n)
    crack_glow = cv2.GaussianBlur(cracks[..., :3], (0, 0), 7) * np.array(col("#E8C8D0"), np.float32)

    out = dict(base=base, ley=np.clip(ley, 0, 1), cracks=cracks, crack_glow=np.clip(crack_glow, 0, 1))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: (np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8) for k, v in out.items()})
    return out

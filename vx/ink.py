"""THE PRINT SHOP — the printed look of THE FLAG GAME? (film 2).  Owner: Printer.

The whole world of the film is BLACK INK ON WHITE PAPER. Scenes paint their world in GRAYSCALE with
cairo (any gray value; pure black strokes stay solid ink; white stays paper) and hand the image to a
print style. FLAGS ARE NEVER PRINTED: flags are drawn on their own layer with vx.flag and laid on top
as the "spot yellow plate" with spot() — plain yellow, unscreened, never halftoned, never hatched.

Images: float32 numpy, (h, w, 3) RGB or (h, w) tone in [0, 1]; luminance is used as tone
(0 = solid ink, 1 = bare paper). Layers with coverage are premultiplied (h, w, 4) RGBA.
Line art is separated from tone automatically: pure-black shapes and their antialiased edges are kept
as crisp solid ink (with a touch of ink spread and edge roughness); only the tone around them is screened.
So draw contours / lettering-like strokes in pure black, tones in grey, highlights in pure white.

Coordinates. Every screen and texture lives on the PAGE. space="screen" (default): the page is the
frame (screen-locked, static images never shimmer). space="page": the page moves under the camera:
page point (design px) = screen design px / zoom + origin, i.e. `origin` = page coords of the frame's
top-left corner and `zoom` = magnification (2 = page seen twice as big: dots and lines grow too).
Glue the print to a camera move by passing the camera's origin/zoom every frame, or, for any camera
(rotation, Droste spirals...), space="page", matrix=M where M maps PAGE design px -> SCREEN design px
(a cairo.Matrix — e.g. the one you ctx.transform() the page with — or a 2x3 array). Every print
function and wet() accept matrix=.

API
---
tract(fc, img, *, lpi=None, angle=45.0, paper=True, space="screen", origin=(0, 0), zoom=1.0,
      gain=0.35, rough=0.45, boil=0) -> rgb
    Chick-tract comic print: AM halftone of round dots at `angle` degrees (dots join into a checkerboard
    at 50%, invert into round holes in the shadows), real dot gain, rough newsprint edges, solid black
    kept solid, highlights bare paper. lpi = dot rows per 1080 page px (default 70 -> big comic dots).
engrave(fc, img, *, angle=None, spacing=None, cross=True, paper=True, space="screen", origin=(0, 0),
        zoom=1.0, follow=1.0, flow_len=110.0, phase=None, phase_mask=None, wiggle=0.12, boil=0) -> rgb
    Gustave-Dore wood engraving: tone becomes parallel line screens whose width encodes darkness;
    light tones taper into flicked dashes, bare-white highlights, two levels of crosshatch for deep
    shadow (cross=True -> 3 layers, False -> 1, or an int 1..3), solid black kept.
    Direction: by default the hatching follows form — the image's own contours/isophotes (smoothed
    structure tensor) bend a low-frequency phase field (a screened-Poisson least-squares fit on an 8 px
    grid, pinned to the plain screen wherever the image is flat), so it is smooth, local and temporally
    stable: no swimming in flat areas, no halo around moving objects. follow=0 -> straight lines;
    flow_len = how far (design px) a bend may carry across form (smaller = more local).
    Moving heroes: print them with print_layer(..., "engrave", space="page", origin=(-x, -y)) so their
    hatching rides with them instead of sliding under a fixed screen.
    angle: scalar radians = base LINE direction (default -0.31 rad, rising gently to the right, y down),
           or an (h, w) map of LINE directions in radians (overrides the automatic flow).
    spacing: design px between lines on the page (default 6.0).
    phase: (h, w) map in units of line spacing (engraved lines are its level sets; overrides everything
           for layer 1), or a list [p1, p2, p3] (None entries = automatic) for the crosshatch layers too.
           phase_mask: optional (h, w) 0..1 where the phase map applies (else everywhere).
manga(fc, img, *, lpi=None, **kw) -> rgb
    Fine screentone (default 125 lpi, crisp, low gain) for the anime gag; same kwargs as tract.
    Speed / focus lines: vx.comic.speed_lines (Letterer) drawn into the GREY world, then print.
reprint(fc, img, from_style, to_style, progress, **kw) -> rgb
    Re-screen one style into another: the two screens' threshold functions are interpolated, so dots
    melt into engraved lines (or back) continuously. progress: scalar 0..1 or (h, w) map (wipes/ripples).
    kw are shared by both styles (lpi/angle go to a dot style, spacing/follow/... to engrave;
    tract `angle` is in degrees, engrave's in radians: pass dot_angle= / line_angle= to disambiguate).
print_layer(fc, rgba, style="tract", **kw) -> rgba   print a premultiplied layer, keeping its coverage,
    so scenes can stack: paper <- back print <- spot(flags) <- front print (hands/foreground over a pole).
    style: "tract" | "engrave" | "manga" (kw forwarded).
paper(fc, *, space="screen", origin=(0, 0), zoom=1.0) -> rgb   the paper stock (neutral white fibres)
spot(dst_rgb, flag_rgba) -> rgb        lay the flag layer (premultiplied RGBA of a Canvas that contains
    ONLY vx.flag drawings) over the print. The only place yellow enters the film. Nothing is ever
    applied to the cloth: print/wet/letter everything else BEFORE or around it.
wet(fc, printed_rgb, mask, amount=1.0, *, space="screen", origin=(0, 0), zoom=1.0, buckle=1.0, bleed=1.0,
    darken=1.0, tide=1.0, show_through=0.05) -> rgb
    The deluge soaking the page, applied to a PRINT (before spot()): mask (h, w) 0..1 wetness (animate
    the flood front). Wet paper darkens and turns translucent (mirrored show-through of the other side),
    ink softens and feathers along the fibres, solids wash out, a tide line forms at the wet front, and
    the sheet cockles (page-anchored wrinkles: displacement + shading). Each effect has its own gain.
PAPER, INK                             the two print colours (RGB tuples)

Common kwargs: paper=False -> flat PAPER colour (no stock texture); boil=N -> edge roughness re-rolls
every N frames (a living print; 0 = static).
Performance (1080p, 4 numba threads, VX_INK_THREADS to change): tract/manga ~24 ms, engrave ~55 ms,
reprint ~80-150 ms, wet ~70 ms. Half-res previews are 4x cheaper.
"""
import math
import os

import cv2
import numpy as np

from . import ink_k as K
from .config import W, H

PAPER = (0.945, 0.945, 0.942)   # neutral white stock (keep it neutral: yellow belongs to the Flags)
INK = (0.062, 0.062, 0.068)     # neutral, very slightly cool black

_THREADS = int(os.environ.get("VX_INK_THREADS", "4"))
_T = {}


# ================================================================== textures (built once per process)
def _gauss_noise(n, sigma, seed):
    """periodic n x n gaussian-filtered noise, unit std"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n))
    f = np.fft.fftfreq(n)
    r2 = f[:, None] ** 2 + f[None, :] ** 2
    y = np.real(np.fft.ifft2(np.fft.fft2(x) * np.exp(-2.0 * math.pi ** 2 * sigma ** 2 * r2)))
    return (y / (y.std() + 1e-12)).astype(np.float32)


def _fibres(n, count, seed, lw=(0.35, 0.9), length=(6.0, 60.0)):
    import cairo
    rng = np.random.default_rng(seed)
    surf = cairo.ImageSurface(cairo.FORMAT_A8, n, n)
    ctx = cairo.Context(surf)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for _ in range(count):
        x, y = rng.uniform(0, n, 2)
        a = rng.uniform(0, math.pi)
        L = math.exp(rng.uniform(math.log(length[0]), math.log(length[1])))
        bend = rng.normal(0, 0.35) * L
        dx, dy = math.cos(a) * L, math.sin(a) * L
        mx, my = x + dx * 0.5 - math.sin(a) * bend * 0.5, y + dy * 0.5 + math.cos(a) * bend * 0.5
        ctx.set_line_width(rng.uniform(*lw))
        ctx.set_source_rgba(0, 0, 0, rng.uniform(0.25, 0.8))
        for ox in (-n, 0, n):
            for oy in (-n, 0, n):
                if (min(x, x + dx) + ox > n + 4 or max(x, x + dx) + ox < -4 or
                        min(y, y + dy) + oy > n + 4 or max(y, y + dy) + oy < -4):
                    continue
                ctx.move_to(x + ox, y + oy)
                ctx.curve_to(mx + ox, my + oy, mx + ox, my + oy, x + dx + ox, y + dy + oy)
                ctx.stroke()
    surf.flush()
    buf = np.ndarray((n, surf.get_stride()), np.uint8, surf.get_data())[:, :n]
    return buf.astype(np.float32) / 255.0


def _tex():
    if _T:
        return _T
    # fine noise: ink edge roughness (1 texel = 1 page px); mid noise: burin wiggle, taper, ink mottle
    _T["nzf"] = np.clip(_gauss_noise(512, 0.9, 11) * 0.45, -1, 1)
    _T["nzm"] = np.clip(_gauss_noise(256, 1.1, 12) * 0.45, -1, 1)
    # paper stock tile (1 texel = 1 page px), neutral multiplier around 1.0
    n = 1024
    tex = (1.0 + 0.010 * _gauss_noise(n, 40.0, 21) + 0.006 * _gauss_noise(n, 9.0, 22)
           + 0.007 * _gauss_noise(n, 0.6, 23)
           - 0.040 * _fibres(n, 2600, 24) + 0.022 * _fibres(n, 900, 25, lw=(0.5, 1.2)))
    tex = tex / tex.mean()
    pyr = [tex.astype(np.float32)]
    while pyr[-1].shape[0] > 64:
        pyr.append(cv2.resize(pyr[-1], (pyr[-1].shape[1] // 2, pyr[-1].shape[0] // 2),
                              interpolation=cv2.INTER_AREA))
    _T["pyr"] = pyr
    # calibrated round-dot screen: T(f) = fraction of the cell where the spot function exceeds f
    N = 1024
    u = (np.arange(N) + 0.5) / N
    f = 0.5 * (np.cos(2 * np.pi * u)[:, None] + np.cos(2 * np.pi * u)[None, :])
    fs = np.sort(f.ravel())
    NL = 4097
    F = np.linspace(-1.0, 1.0, NL)
    A = 1.0 - np.searchsorted(fs, F) / fs.size
    G = np.abs(np.gradient(A, F))
    G = np.convolve(np.pad(G, 8, mode="edge"), np.ones(9) / 9.0, mode="same")[8:-8]
    _T["lutA"] = A.astype(np.float64)
    _T["lutG"] = G.astype(np.float64)
    _T["dummy2"] = np.zeros((1, 1), np.float32)
    _T["dummy3"] = np.zeros((1, 1, 1), np.float32)
    return _T


# ================================================================== helpers
_LUMA = np.array([[0.2126, 0.7152, 0.0722]], np.float32)


def _tone(img):
    if img.ndim == 2:
        return np.ascontiguousarray(img, dtype=np.float32)
    x = np.ascontiguousarray(img[..., :3], dtype=np.float32)
    return cv2.transform(x, _LUMA)


def _prep(img):
    """luminance L (clipped to [0, 1]) and its 3x3 local min / max (line-art separation)"""
    Lum = np.clip(_tone(img), 0.0, 1.0)
    k = np.ones((3, 3), np.uint8)
    return Lum, cv2.erode(Lum, k), cv2.dilate(Lum, k)


def _page_map(fc, space, origin, zoom, matrix=None):
    """page coords of an output pixel: page = A @ (x_pix, y_pix) + t (pixel centres at j + 0.5).
    Returns (A (2, 2), t (2,), cx = page px per output pixel)."""
    if space == "page" and matrix is not None:
        if hasattr(matrix, "xx"):                       # cairo.Matrix (page -> screen design px)
            M = np.array([[matrix.xx, matrix.xy, matrix.x0], [matrix.yx, matrix.yy, matrix.y0]], np.float64)
        else:
            M = np.asarray(matrix, np.float64).reshape(2, 3)
        Li = np.linalg.inv(M[:, :2])
        A = Li / fc.s
        t = -Li @ M[:, 2]
    elif space == "page":
        A = np.eye(2) / (fc.s * float(zoom))
        t = np.array([float(origin[0]), float(origin[1])])
    else:
        A = np.eye(2) / fc.s
        t = np.zeros(2)
    return A, t, math.sqrt(abs(np.linalg.det(A)))


def _tile_M(A, t, q, texel, off):
    """cv2.warpAffine (inverse-map) matrix sampling a periodic page tile onto a 1/q-res pixel grid"""
    Aq = A * q
    c = Aq @ np.array([0.5, 0.5]) + t
    return np.array([[Aq[0, 0] / texel, Aq[0, 1] / texel, c[0] / texel + off[0] - 0.5],
                     [Aq[1, 0] / texel, Aq[1, 1] / texel, c[1] / texel + off[1] - 0.5]], np.float32)


def _base_params(fc, space, origin, zoom, paper, boil, gain, rough, matrix=None):
    T = _tex()
    P = np.zeros(K.NP, np.float64)
    A, t, cx = _page_map(fc, space, origin, zoom, matrix)
    P[K.CX], P[K.OX], P[K.OY], P[K.INVS] = cx, t[0], t[1], 1.0 / fc.s
    P[K.MA00], P[K.MA01], P[K.MA10], P[K.MA11] = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    pxs = 1.0 / cx                                   # output px per page px
    P[K.GAIN] = gain * pxs
    P[K.ROUGH] = rough * min(pxs, 1.0)
    P[K.SPREAD] = min(1.0, 0.30 * pxs) * (1.0 if gain > 0 else 0.0)
    P[K.LROUGH] = min(0.8, 0.35 * pxs) * (rough / 0.45 if rough > 0 else 0.0)
    P[K.PR], P[K.PG], P[K.PB] = PAPER
    P[K.IR], P[K.IG], P[K.IB] = INK
    P[K.PAPER_ON] = 1.0 if paper else 0.0
    P[K.MOTT] = 0.22
    P[K.PIN] = 0.07
    lev = max(0.0, min(len(T["pyr"]) - 1.001, math.log2(max(cx, 1e-6))))
    l0 = int(lev)
    P[K.R0], P[K.R1], P[K.PM] = 1.0 / 2 ** l0, 1.0 / 2 ** (l0 + 1), lev - l0
    if boil:
        r = np.random.default_rng(int(fc.F // max(1, int(boil))) * 7919 + 5)
        P[K.NOX], P[K.NOY] = r.uniform(0, 512, 2)
    # neutral defaults for the other style's slots
    P[K.CELL], P[K.CA], P[K.SA] = 15.0, math.cos(math.pi / 4), math.sin(math.pi / 4)
    P[K.LO_D], P[K.HI_D], P[K.LO_L], P[K.HI_L] = 0.04, 0.93, 0.03, 0.96
    P[K.K], P[K.GCELL] = 1.0 / 6.0, 8.0
    P[K.A1], P[K.A2] = 0.55, 0.5
    P[K.NLAY] = 3
    return P, (T["pyr"][l0], T["pyr"][l0 + 1])


def _set_dots(P, lpi, angle_deg, default_lpi):
    P[K.CELL] = 1080.0 / float(lpi or default_lpi)
    a = math.radians(angle_deg)
    P[K.CA], P[K.SA] = math.cos(a), math.sin(a)


_BASE_ANGLE = -0.31          # line direction of the first hatch layer (radians, y down)
_CROSS = (0.0, 1.0, -1.22)   # layer offsets from the first layer's direction


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _rhs(gx, gy, h):
    """D^T g for a target gradient field g sampled at cell centres (edges get the neighbour average)"""
    gh, gw = gx.shape
    ex = 0.5 * (gx[:, 1:] + gx[:, :-1])
    ey = 0.5 * (gy[1:, :] + gy[:-1, :])
    rhs = np.zeros((gh, gw))
    rhs[:, :-1] -= ex
    rhs[:, 1:] += ex
    rhs[:-1, :] -= ey
    rhs[1:, :] += ey
    return rhs / h


def _solve(rhs, lam, h):
    """argmin |D psi - g|^2 + sum lam psi^2 (free borders): two-level Jacobi-PCG (coarse solve, fine polish)"""
    gh, gw = rhs.shape
    cw, ch = (gw + 1) // 2, (gh + 1) // 2
    rc = cv2.resize(rhs, (cw, ch), interpolation=cv2.INTER_AREA)
    lc = cv2.resize(lam, (cw, ch), interpolation=cv2.INTER_AREA)
    xc = K.cg_solve(rc, lc, np.zeros((ch, cw)), 2.0 * h, 160, 2e-3)
    x0 = cv2.resize(xc, (gw, gh), interpolation=cv2.INTER_LINEAR)
    return K.cg_solve(rhs, lam, np.ascontiguousarray(x0), h, 30, 1e-3)


def _flow(fc, S, P, base, angle_map, follow, flow_len, zoom):
    """bent phase offsets psi (3, gh, gw) on an 8-design-px grid (screen space, shifted by P[GOX/GOY]
    design px so that under a page-locked pan the analysis grid rides with the page)"""
    gc = P[K.GCELL]
    gw, gh = int(round(W / gc)), int(round(H / gc))
    dx, dy = P[K.GOX], P[K.GOY]

    def shifted(a):
        if dx == 0.0 and dy == 0.0:
            return a
        M = np.array([[1, 0, dx * fc.s], [0, 1, dy * fc.s]], np.float32)
        return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    if angle_map is not None:
        a2 = 2.0 * np.asarray(angle_map, np.float32)
        c2 = cv2.resize(shifted(np.cos(a2)), (gw, gh), interpolation=cv2.INTER_AREA)
        s2 = cv2.resize(shifted(np.sin(a2)), (gw, gh), interpolation=cv2.INTER_AREA)
        theta_n = 0.5 * np.arctan2(s2, c2) + math.pi / 2
        conf = np.clip(np.hypot(c2, s2), 0, 1)
        hdelta = None
    else:
        aw, ah = gw * 2, gh * 2                                      # 4-design-px analysis grid
        # the flow follows FORM (broad tone), not thin marks: rain streaks, hairlines and hatching drawn
        # into the world (< ~6-10 px) are removed first, so animated streaks never steer the engraving
        k3 = np.ones((3, 3), np.uint8)
        hw2, hh2 = W // 2, H // 2
        S = shifted(S)
        sm = S if S.shape[1] <= hw2 else cv2.resize(S, (hw2, hh2), interpolation=cv2.INTER_AREA)
        sm = cv2.morphologyEx(cv2.morphologyEx(sm, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k3)
        sm = cv2.resize(sm, (aw, ah), interpolation=cv2.INTER_AREA)
        sm = cv2.morphologyEx(cv2.morphologyEx(sm, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k3)
        sm = cv2.GaussianBlur(sm, (0, 0), 1.3)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / 32.0          # darkness per design px
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / 32.0
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.8)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.8)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.8)
        jxx, jxy, jyy = (cv2.resize(a, (gw, gh), interpolation=cv2.INTER_AREA) for a in (jxx, jxy, jyy))
        theta_n = 0.5 * np.arctan2(2 * jxy, jxx - jyy)               # normal of the isophotes
        disc = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
        l1 = 0.5 * (jxx + jyy + disc)
        l2 = 0.5 * (jxx + jyy - disc)
        coh = (l1 - l2) / (l1 + l2 + 1e-9)
        conf = follow * coh * _smoothstep(0.0012, 0.006, np.sqrt(np.maximum(l1, 0)))
        hdelta = True
    an1 = base + math.pi / 2
    dl = (theta_n - an1 + math.pi / 2) % math.pi - math.pi / 2          # (-pi/2, pi/2]
    if hdelta:  # follow up to 50 deg, then ease back to the base direction (continuous across +-90)
        lim = math.radians(50)
        ad = np.abs(dl)
        dl = np.sign(dl) * np.where(ad <= lim, ad, lim * (math.pi / 2 - ad) / (math.pi / 2 - lim))
    dl = cv2.GaussianBlur((dl * conf).astype(np.float32), (0, 0), 0.7).astype(np.float64)
    ks = P[K.K] / zoom                                                  # cycles per screen design px
    # screening: weak where the image has form to follow, strong (pinned to the plain screen) where it is
    # flat, so bent phase never leaks into empty backgrounds (no halo swimming around moving objects)
    m = cv2.GaussianBlur(conf.astype(np.float32), (0, 0), 0.8)
    m = np.clip(m / 0.5, 0, 1).astype(np.float64)
    lam = 1.0 / float(flow_len) ** 2 + (1.0 - m) ** 2 / 16.0 ** 2
    ux, uy = np.cos(dl) - 1.0, np.sin(dl)
    p1 = _solve(_rhs(ux, uy, gc), lam, gc)
    p2 = _solve(_rhs(-uy, ux, gc), lam, gc)
    psi = np.zeros((3, gh, gw), np.float32)
    for L, off in enumerate(_CROSS[: int(P[K.NLAY])]):
        a = an1 + off                                   # target of layer L = ks * Rot(a) (u)
        psi[L] = ks * (math.cos(a) * p1 + math.sin(a) * p2)
    return psi


def _run(S, MN, MX, out, P, psi, ph, phm, mixmap, paps):
    import numba
    T = _tex()
    n0 = numba.get_num_threads()
    nt = max(1, min(_THREADS, numba.config.NUMBA_NUM_THREADS))
    if nt != n0:
        numba.set_num_threads(nt)
    try:
        K.k_print(S, MN, MX, out, P, T["lutA"], T["lutG"], psi, ph[0], ph[1], ph[2], phm, mixmap,
                  T["nzf"], T["nzm"], paps[0], paps[1])
    finally:
        if nt != n0:
            numba.set_num_threads(n0)
    return out


def _lines_setup(fc, S, P, *, angle=None, spacing=None, cross=True, follow=1.0, flow_len=110.0,
                 phase=None, phase_mask=None, wiggle=0.12, zoom=1.0, space="screen", origin=(0.0, 0.0),
                 matrix=None):
    T = _tex()
    P[K.K] = 1.0 / float(spacing or 6.0)
    P[K.NLAY] = 3 if cross is True else (1 if cross is False else int(np.clip(cross, 1, 3)))
    P[K.WIG] = wiggle
    P[K.TAPER] = 0.35
    P[K.DASH] = 18.0
    P[K.RMIN] = 0.5
    P[K.LGAIN] = 0.10 / P[K.CX]
    amap = None
    if angle is not None and np.ndim(angle) == 2:
        amap = angle
        a2 = 2.0 * np.asarray(angle, np.float64)
        base = 0.5 * math.atan2(np.sin(a2).mean(), np.cos(a2).mean())
    else:
        base = float(angle) if angle is not None else _BASE_ANGLE
    for L, off in enumerate(_CROSS):
        a = base + math.pi / 2 + off
        P[K.N1X + 2 * L], P[K.N1Y + 2 * L] = math.cos(a), math.sin(a)
    z = float(zoom) if space == "page" else 1.0
    rot = 0.0
    if space == "page":   # analysis grid rides with the page (screen offset of the page origin, mod cell)
        if matrix is not None:
            A, t, _ = _page_map(fc, space, origin, zoom, matrix)
            Ml = np.linalg.inv(A * fc.s)                   # page -> screen design px (linear part)
            z = math.sqrt(abs(np.linalg.det(Ml)))
            rot = math.atan2(Ml[1, 0], Ml[0, 0])
            so = -Ml @ t                                   # screen position of the page origin
        else:
            so = (-float(origin[0]) * z, -float(origin[1]) * z)
        P[K.GOX] = so[0] % P[K.GCELL]
        P[K.GOY] = so[1] % P[K.GCELL]
    if amap is not None or follow > 0:
        psi = _flow(fc, S, P, base + rot, amap, follow, flow_len, z)
        P[K.USE_PSI] = 1.0
    else:
        psi = T["dummy3"]
    ph = [T["dummy2"]] * 3
    if phase is not None:
        plist = list(phase) if isinstance(phase, (list, tuple)) else [phase]
        for L, p in enumerate(plist[:3]):
            if p is not None:
                ph[L] = np.ascontiguousarray(p, dtype=np.float32)
                P[K.USE_PH1 + L] = 1.0
    phm = T["dummy2"]
    if phase_mask is not None:
        phm = np.ascontiguousarray(phase_mask, dtype=np.float32)
        P[K.USE_PHM] = 1.0
    return psi, ph, phm


def _print(fc, img, modeA, kwA, modeB=None, kwB=None, progress=None, common=None):
    common = common or {}
    space = common.get("space", "screen")
    origin = common.get("origin", (0.0, 0.0))
    zoom = common.get("zoom", 1.0)
    matrix = common.get("matrix")
    S, MN, MX = _prep(img)
    P, paps = _base_params(fc, space, origin, zoom, common.get("paper", True), common.get("boil", 0),
                           common.get("gain", 0.35), common.get("rough", 0.45), matrix)
    T = _tex()
    psi, ph, phm = T["dummy3"], [T["dummy2"]] * 3, T["dummy2"]
    mixmap = T["dummy2"]
    P[K.MODE_A] = modeA
    P[K.MODE_B] = modeA if modeB is None else modeB
    for mode, kw in ((modeA, kwA), (modeB, kwB)):
        if mode is None:
            continue
        if mode == K.DOTS:
            _set_dots(P, kw.get("lpi"), kw.get("angle", 45.0), kw.get("default_lpi", 70.0))
            if "lo" in kw:
                P[K.LO_D] = kw["lo"]
            if "hi" in kw:
                P[K.HI_D] = kw["hi"]
        else:
            psi, ph, phm = _lines_setup(fc, S, P, zoom=zoom, space=space, origin=origin, matrix=matrix, **kw)
    if "gain_px" in common:
        P[K.GAIN] = common["gain_px"]
    if progress is not None:
        if np.ndim(progress) == 2:
            mixmap = np.ascontiguousarray(np.clip(progress, 0, 1), dtype=np.float32)
            P[K.USE_MIXMAP] = 1.0
        else:
            P[K.MIX] = float(np.clip(progress, 0, 1))
            if P[K.MIX] >= 1.0:           # fully arrived: print the target style natively
                P[K.MODE_A], P[K.MIX] = P[K.MODE_B], 0.0
    out = np.empty((fc.h, fc.w, 3), np.float32)
    return _run(S, MN, MX, out, P, psi, ph, phm, mixmap, paps)


# ================================================================== public API
def paper(fc, *, space="screen", origin=(0.0, 0.0), zoom=1.0, matrix=None, **kw):
    """the paper stock: neutral white with subtle fibres and mottling (never yellowish)"""
    img = np.ones((fc.h, fc.w), np.float32)
    return _print(fc, img, K.DOTS, {}, common=dict(space=space, origin=origin, zoom=zoom, matrix=matrix))


def tract(fc, img, *, lpi=None, angle=45.0, paper=True, space="screen", origin=(0.0, 0.0), zoom=1.0,
          gain=0.35, rough=0.45, boil=0, matrix=None):
    return _print(fc, img, K.DOTS, dict(lpi=lpi, angle=angle),
                  common=dict(space=space, origin=origin, zoom=zoom, paper=paper, boil=boil,
                              gain=gain, rough=rough, matrix=matrix))


def manga(fc, img, *, lpi=None, angle=45.0, paper=True, space="screen", origin=(0.0, 0.0), zoom=1.0,
          gain=0.15, rough=0.2, boil=0, matrix=None):
    return _print(fc, img, K.DOTS, dict(lpi=lpi, angle=angle, default_lpi=125.0, lo=0.03, hi=0.95),
                  common=dict(space=space, origin=origin, zoom=zoom, paper=paper, boil=boil,
                              gain=gain, rough=rough, matrix=matrix))


def engrave(fc, img, *, angle=None, spacing=None, cross=True, paper=True, space="screen",
            origin=(0.0, 0.0), zoom=1.0, follow=1.0, flow_len=110.0, phase=None, phase_mask=None,
            wiggle=0.12, boil=0, rough=0.35, matrix=None):
    kw = dict(angle=angle, spacing=spacing, cross=cross, follow=follow, flow_len=flow_len,
              phase=phase, phase_mask=phase_mask, wiggle=wiggle)
    return _print(fc, img, K.LINES, kw,
                  common=dict(space=space, origin=origin, zoom=zoom, paper=paper, boil=boil,
                              gain=0.35, rough=rough, matrix=matrix))


_LINE_KEYS = ("spacing", "cross", "follow", "flow_len", "phase", "phase_mask", "wiggle", "line_angle")


def reprint(fc, img, from_style, to_style, progress, **kw):
    """re-screen from_style into to_style ("tract" | "manga" | "engrave"); progress scalar or (h, w)"""
    common = {k: kw[k] for k in ("space", "origin", "zoom", "paper", "boil", "gain", "rough", "matrix")
              if k in kw}

    def style_kw(style):
        if style in ("tract", "manga"):
            d = dict(lpi=kw.get("lpi"), angle=kw.get("dot_angle", 45.0))
            if style == "manga":
                d.update(default_lpi=125.0, lo=0.03, hi=0.95)
            return K.DOTS, d
        d = {k: kw[k] for k in _LINE_KEYS if k in kw and k != "line_angle"}
        if "line_angle" in kw:
            d["angle"] = kw["line_angle"]
        return K.LINES, d

    mA, kA = style_kw(from_style)
    mB, kB = style_kw(to_style)
    if np.ndim(progress) == 0 and float(progress) <= 0.0:
        return _print(fc, img, mA, kA, common=common)
    if mA == mB == K.DOTS:   # tract <-> manga: the screens differ in pitch; cross-fade by threshold too
        pass
    return _print(fc, img, mA, kA, mB, kB, progress=progress, common=common)


_STYLES = {"tract": tract, "engrave": engrave, "manga": manga}


def print_layer(fc, rgba, style="tract", **kw):
    a = np.clip(rgba[..., 3:4], 0, 1)
    rgb = np.where(a > 1e-4, rgba[..., :3] / np.maximum(a, 1e-4), 1.0)   # un-premultiply over paper-white
    out = _STYLES[style](fc, rgb, **kw)
    return np.concatenate([out * a, a], axis=-1).astype(np.float32)


def spot(dst_rgb, flag_rgba):
    return flag_rgba[..., :3] + dst_rgb * (1.0 - flag_rgba[..., 3:4])


# ================================================================== the deluge: a soaked page
_GRID = {}


def _pixgrid(h, w):
    g = _GRID.get((h, w))
    if g is None:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        g = _GRID[(h, w)] = (xs, ys)
    return g


def _page_tile(fc, tile, texel, pm, off=(0.0, 0.0), q=1):
    """sample a periodic tile on the page (texel = page px per texel; pm = _page_map(...)) -> 1/q-res"""
    A, t, _ = pm
    return cv2.warpAffine(tile, _tile_M(A, t, q, texel, off), ((fc.w + q - 1) // q, (fc.h + q - 1) // q),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_WRAP)


def wet(fc, img, mask, amount=1.0, *, space="screen", origin=(0.0, 0.0), zoom=1.0, buckle=1.0,
        bleed=1.0, darken=1.0, tide=1.0, show_through=0.05, matrix=None):
    """Soak a PRINTED page (output of tract/engrave/manga, BEFORE spot(): never apply to the flag plate).
    mask: (h, w) wetness 0..1 (animate it: the flood front), amount: global multiplier.
    Wet paper darkens and turns translucent (a mirrored ghost of the other side shows through), ink
    wicks into the fibres (feathered bleed halo, washed-out solids), tide lines form at the wet front,
    and the sheet buckles (displacement + shading of slow page-anchored wrinkles).  -> rgb"""
    T = _tex()
    h, w = fc.h, fc.w
    m = np.clip(np.asarray(mask, np.float32) * float(amount), 0.0, 1.0)
    if m.ndim == 3:
        m = m[..., 0]
    if not np.any(m > 1e-3):
        return img
    pm = _page_map(fc, space, origin, zoom, matrix)
    pxs = 1.0 / pm[2]                                        # output px per page px
    Lum = _tone(img)
    pl = 0.2126 * PAPER[0] + 0.7152 * PAPER[1] + 0.0722 * PAPER[2]
    il = 0.2126 * INK[0] + 0.7152 * INK[1] + 0.0722 * INK[2]
    cov = np.clip((pl - Lum) * (1.0 / (pl - il)), 0.0, 1.0)
    # un-inked paper brightness (keeps the stock texture of the print)
    tex = np.clip((Lum - il * cov) / np.maximum(1.0 - cov, 1e-3) * (1.0 / pl), 0.8, 1.1)
    tex[cov > 0.98] = 1.0
    # ---- ink bleed: lines go soft and fat, capillary feathers creep along the fibres, a faint haze
    #      gathers around dense ink, and solids wash out a little (mottled)
    fib = _page_tile(fc, T["nzf"], 1.4, pm, (31.0, 7.0))
    b1 = cv2.GaussianBlur(cov, (0, 0), max(0.5, 0.9 * pxs))
    b2 = cv2.GaussianBlur(cov, (0, 0), max(1.0, 3.0 * pxs))
    mb = m * bleed
    feather = np.clip((b1 - 0.12) * 3.0, 0.0, 1.0) * np.clip((fib - 0.15) * 3.0, 0.0, 1.0) * 0.8
    spread = np.maximum(np.minimum(b1 * 1.3, 1.0), feather) + 0.08 * b2
    cov_w = np.maximum(cov, np.minimum(spread, 1.0) * mb)
    cov_w *= 1.0 - 0.14 * mb * cov_w * (0.65 + 0.35 * fib)
    # ---- wet paper: darker, translucent (show-through of the reversed other side), tide lines
    paper_k = 1.0 - 0.10 * darken * m
    if tide > 0:
        gm = cv2.GaussianBlur(m, (0, 0), max(1.0, 3.0 * pxs))
        gx = cv2.Sobel(gm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gm, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy) * (pxs / 8.0)                  # wetness change per page px
        paper_k -= tide * np.clip(edge * 30.0, 0.0, 1.0) * np.exp(-((gm - 0.5) / 0.22) ** 2) * 0.09
    if show_through > 0:
        paper_k *= 1.0 - cv2.GaussianBlur(np.ascontiguousarray(cov[:, ::-1]), (0, 0),
                                          max(1.5, 4.0 * pxs)) * (show_through * m)
    a = tex * paper_k * (1.0 - cov_w)                                   # paper weight
    b = cov_w                                                           # ink weight
    # ---- buckling (cockle): slow wrinkles anchored to the page, grown where wetter; computed at 1/4 res
    if buckle > 0:
        q = 4
        hq, wq = (h + q - 1) // q, (w + q - 1) // q
        Hq = _page_tile(fc, T["nzm"], 30.0, pm, (3.0, 91.0), q)
        Hq = Hq * cv2.resize(m, (wq, hq), interpolation=cv2.INTER_AREA)
        Hq = cv2.GaussianBlur(Hq, (0, 0), max(0.8, 6.0 * pxs / q))
        # slope per page px, in units of wrinkle height
        hx = cv2.resize(cv2.Sobel(Hq, cv2.CV_32F, 1, 0, ksize=3) * (pxs / (8.0 * q)), (w, h))
        hy = cv2.resize(cv2.Sobel(Hq, cv2.CV_32F, 0, 1, ksize=3) * (pxs / (8.0 * q)), (w, h))
        amp = 160.0 * pxs * buckle                                         # displacement (output px)
        xs, ys = _pixgrid(h, w)
        mx_, my_ = xs + hx * amp, ys + hy * amp
        a = cv2.remap(a, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        b = cv2.remap(b, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        shade = 1.0 + np.clip((-hx - 0.6 * hy) * 3.2 * buckle, -0.08, 0.08)   # slope std ~0.008 -> ~3%
        a *= shade
        b *= shade
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        out[..., c] = a * PAPER[c] + b * INK[c]
    return out


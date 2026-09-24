"""THE PRINT GALLERY ENGINE (t09).  Owner: T09PassItOn.

An exact de Smit–Lenstra conformal Droste ("Escher's Print Gallery") of a self-similar tract page.

Base image B (page units, PU): the tract page P = [0, PAGE_W] x [0, PAGE_H] whose panel 2 shows Summer
holding the open tract; the right page of that tract is the WINDOW P' = P / S + (x', y').  The similarity
z -> p + (z - p) / S maps P onto P' and fixes the point p, so the plane (minus p) is tiled by the levels
    level k = the page scaled by S**k about p,     B(z) = B(p + S (z - p))           (exact self-similarity)
Every level is PRINTED IN ITS OWN PAGE SPACE (halftone dots, keylines and lettering scale with the level).

Escher/Lenstra map.  Work in logarithmic coordinates u = log(z - p): B is doubly periodic in u with the
lattice  L = Z ln S + Z 2 pi i.  The output (screen) image is
    D(x) = B(p + exp(beta * log(x - q) + C)),       beta = (2 pi i + m ln S) / (2 pi i)
(q = screen position of the singularity, C = complex camera offset, m = strands (m ln S per turn)).
beta * 2 pi i lies in L, so D is single-valued: the straight Droste picture twisted into the Print Gallery.
beta = 1 is the plain (straight) Droste zoom.  Any other beta is only seamless while q is OFF screen (the
branch cut of the log is kept pointing away from the frame) — that is how the picture bends in and out.

Rendering.  B is baked once (setup) into a LOG-POLAR TEXTURE T[theta, u_re] over one lattice cell
(u_re in [u0, u0 + ln S), theta in [0, 2 pi)) — uniform detail at every scale.  It is built in radial bands:
each band is a cartesian raster of levels 0 and +1 printed at the band's own resolution, remapped (cv2.remap)
into the texture.  A mip pyramid (2x2 box, wrap-periodic) + per-pixel trilinear LOD from the exact
conformal footprint |beta| / |x - q| gives an anti-aliased spiral all the way into the singularity.

The Flags are laid on a separate spot plate: the texture keeps (print, flag rgba, print-over-flag rgba), so
flag cloth is resampled only as plain yellow (never screened) and can be re-baked per frame (alive flags).
"""
import hashlib
import math
import time
from pathlib import Path

import cairo
import cv2
import numpy as np

from vx import W, H, Canvas
from vx import ink
from vx.config import ROOT, FILM_ROOT

from scenes import t01_page as P

# ------------------------------------------------------------------ geometry of the self-similar page
S_DROSTE = 10.0                       # one level = x10
LN_S = math.log(S_DROSTE)
WIN_X, WIN_Y = 420.0, 612.0           # top-left of the window P' (page 8 = LEFT page of the held tract), PU
WIN_W, WIN_H = P.PAGE_W / S_DROSTE, P.PAGE_H / S_DROSTE
FIX = (WIN_X * S_DROSTE / (S_DROSTE - 1.0), WIN_Y * S_DROSTE / (S_DROSTE - 1.0))   # fixed point p (PU)
TWO_PI = 2.0 * math.pi


def window_rect():
    return (WIN_X, WIN_Y, WIN_W, WIN_H)


def beta_for(m=1.0, kappa=1.0):
    """Lenstra exponent (2 pi i + m ln S) / (2 pi i) with the twist scaled by kappa (1 = exact)."""
    return complex(1.0, -kappa * m * LN_S / TWO_PI)


def r0():
    """circumradius of the window about the fixed point: the texture's inner radius (levels 0/+1 only)"""
    px, py = FIX
    xs = (WIN_X, WIN_X + WIN_W)
    ys = (WIN_Y, WIN_Y + WIN_H)
    return max(math.hypot(x - px, y - py) for x in xs for y in ys) * 1.0005


# ------------------------------------------------------------------ fake frame context for off-screen rasters
class RasterFC:
    """just enough of vx.render.FrameCtx for vx.ink / vx.canvas on an arbitrary raster (1 px = 1 design unit)"""

    def __init__(self, w, h, T=0.0, tl=None):
        self.w, self.h, self.s = int(w), int(h), 1.0
        self.W, self.H = W, H
        self.T = T
        self.t = 0.0
        self.f = 0
        self.F = 0
        self.fps = 24
        self.tl = tl

    def px(self, v):
        return v


def raster_canvas(w, h, bg=None):
    return Canvas(s=1.0, w=int(w), h=int(h), bg=bg)


def _apply(ctx, sigma, ox, oy):
    ctx.scale(sigma, sigma)
    ctx.translate(-ox, -oy)


def render_level(w, h, sigma, ox, oy, level_art, tl=None):
    """Print ONE level of the self-similar page into a (h, w) raster, px = (PU - o) * sigma.

    level_art: object with world(ctx), flags(ctx), front(ctx), letters(ctx, fc), holders(ctx) — all in PU,
    static (the printed page).  Returns premultiplied float32 layers:
        base  (h, w, 4): the print below the flags (panels printed with page-locked dots)
        flag  (h, w, 4): the spot plate (ONLY vx.flag drawings; never printed over)
        top   (h, w, 4): everything printed over the flags (front objects, page stock, keylines,
                         page number, lettering, the thumbs over the window)
    The page rect is opaque except the window P', which is transparent (the next level shows through).
    """
    fc = RasterFC(w, h, tl=tl)
    kw = dict(space="page", origin=(ox, oy), zoom=sigma, lpi=P.TRACT_LPI, angle=P.TRACT_ANGLE)
    panels = [P.PANELS["p1"], P.PANELS["p2"], P.PANELS["p3"]]

    world = raster_canvas(w, h, bg=(1, 1, 1))
    c = world.ctx
    c.save()
    _apply(c, sigma, ox, oy)
    level_art.world(c)
    c.restore()
    base = ink.tract(fc, world.rgb(), **kw)

    fl = raster_canvas(w, h)
    c = fl.ctx
    c.save()
    _apply(c, sigma, ox, oy)
    level_art.flags(c)
    c.restore()
    flag = fl.rgba()

    fr = raster_canvas(w, h)
    c = fr.ctx
    c.save()
    _apply(c, sigma, ox, oy)
    level_art.front(c)
    c.restore()
    front = fr.rgba()
    top = np.zeros((h, w, 4), np.float32)
    if front[..., 3].max() > 0:
        top = ink.print_layer(fc, front, "tract", **kw)

    # page stock + keylines + number + letters (unscreened) -> over top
    tp = raster_canvas(w, h)
    c = tp.ctx
    c.save()
    _apply(c, sigma, ox, oy)
    P.draw_page_frame(c, level_art.page_no, panels)
    level_art.letters(c, fc)
    c.restore()
    fr2 = tp.rgba()
    top = fr2 + top * (1.0 - fr2[..., 3:4])

    # page mask minus the window; everything outside the page is transparent
    mk = raster_canvas(w, h)
    c = mk.ctx
    c.save()
    _apply(c, sigma, ox, oy)
    c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    c.rectangle(0, 0, P.PAGE_W, P.PAGE_H)
    c.rectangle(*window_rect())
    c.set_source_rgba(1, 1, 1, 1)
    c.fill()
    c.restore()
    m = mk.rgba()[..., 3:4]
    base4 = np.concatenate([base * m, m], axis=-1)
    flag = flag * m
    top = top * m

    # the hands that hold the window (thumbs over the page edge) are printed AFTER the window punch
    hd = raster_canvas(w, h)
    c = hd.ctx
    c.save()
    _apply(c, sigma, ox, oy)
    level_art.holders(c)
    c.restore()
    hl = hd.rgba()
    if hl[..., 3].max() > 0:
        hp = ink.print_layer(fc, hl, "tract", **kw)
        top = hp + top * (1.0 - hp[..., 3:4])
    return base4, flag, top


def flatten(base, flag, top):
    """premultiplied layers -> premultiplied rgba (base, spot flags, top)"""
    out = flag + base * (1.0 - flag[..., 3:4])
    return top + out * (1.0 - top[..., 3:4])


# ------------------------------------------------------------------ the log-polar texture
class Texture:
    """Log-polar texture of the self-similar page over one lattice cell + mip pyramid.
    tex[j, i]: theta = j * dth, u_re = u0 + i * du.   Layers stored as uint8 premultiplied RGBA
    (print 'lo' = base under flags, 'hi' = print over flags, 'fl' = flag plate) and 'rgb' = the flattened
    composite on paper-black-free background (every texel is covered)."""

    def __init__(self, nu, u0, levels):
        self.nu, self.u0 = nu, u0
        self.levels = levels              # list of dicts {"rgb": uint8 (NT, NR, 3)}
        self.NT, self.NR = levels[0]["rgb"].shape[:2]
        self.du = LN_S / self.NR
        self.dth = TWO_PI / self.NT
        self.mean = levels[-1]["rgb"].reshape(-1, 3).mean(0).astype(np.float32) / 255.0


def texture_dims(nu):
    nr = int(round(nu * LN_S / 64.0)) * 64
    du = LN_S / nr
    nt = int(round(TWO_PI / du / 64.0)) * 64
    return nr, nt


def _mips(img):
    """wrap-periodic box pyramid (sizes halve while even, then INTER_AREA)"""
    out = [img]
    cur = img
    while min(cur.shape[:2]) > 4:
        h, w = cur.shape[:2]
        if h % 2 == 0 and w % 2 == 0:
            f = cur.astype(np.float32)
            nxt = (f[0::2, 0::2] + f[1::2, 0::2] + f[0::2, 1::2] + f[1::2, 1::2]) * 0.25
        else:
            nxt = cv2.resize(cur.astype(np.float32), (max(1, (w + 1) // 2), max(1, (h + 1) // 2)),
                             interpolation=cv2.INTER_AREA)
        cur = np.clip(nxt + 0.5, 0, 255).astype(np.uint8) if img.dtype == np.uint8 else nxt
        out.append(cur)
    return out


def _src_hash(extra=""):
    h = hashlib.sha1(extra.encode())
    files = sorted(list((ROOT / "vx").glob("ink*.py")) + list((ROOT / "vx").glob("comic*.py")) +
                   list((ROOT / "vx").glob("chars*.py")) + list((ROOT / "vx").glob("flag*.py")) +
                   list((FILM_ROOT / "scenes").glob("t09_*.py")) + [FILM_ROOT / "scenes" / "t01_page.py"] +
                   list((FILM_ROOT / "scenes").glob("t08_camp*.py")) + list((FILM_ROOT / "scenes").glob("t08_art*.py")))
    for f in files:
        if f.exists():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def build_texture(level_art, nu, cache_dir=None, nb=8, tag="", log=print):
    """Bake the log-polar texture of the self-similar page (levels 0 and +1 cover one lattice cell)."""
    nr, nt = texture_dims(nu)
    nb = max(1, nb)
    while nr % nb:
        nb -= 1
    u0 = math.log(r0())
    key = f"droste_{_src_hash(tag + repr((nu, nr, nt, nb, S_DROSTE, WIN_X, WIN_Y)))}.npz"
    if cache_dir is not None:
        path = Path(cache_dir) / key
        if path.exists():
            z = np.load(path)
            return Texture(nu, u0, [{"rgb": z["rgb"]}]), z
    t0 = time.time()
    du = LN_S / nr
    dth = TWO_PI / nt
    px, py = FIX
    rgb = np.zeros((nt, nr, 3), np.uint8)
    per = nr // nb
    th = (np.arange(nt, dtype=np.float64) * dth)[:, None]
    cos_t, sin_t = np.cos(th), np.sin(th)
    for b in range(nb):
        i0, i1 = b * per, (b + 1) * per
        ra = math.exp(u0 + i0 * du)
        rb = math.exp(u0 + (i1 + 1) * du) * 1.004
        dens = 1.0 / (ra * du)                     # px per PU: one texel per pixel at the band's inner edge
        half = int(math.ceil(rb * dens)) + 3
        size = 2 * half
        ox, oy = px - half / dens, py - half / dens
        acc = None
        for k in (0, 1):
            sk = S_DROSTE ** k
            sig = dens * sk
            oxk, oyk = px + (ox - px) / sk, py + (oy - py) / sk
            base, flag, top = render_level(size, size, sig, oxk, oyk, level_art)
            lay = flatten(base, flag, top)
            acc = lay if acc is None else lay + acc * (1.0 - lay[..., 3:4])
        a = np.clip(acc[..., 3:4], 1e-6, 1.0)
        col = np.where(acc[..., 3:4] > 1e-4, acc[..., :3] / a, 0.0)
        r = np.exp(u0 + np.arange(i0, i1, dtype=np.float64) * du)[None, :]
        mx = ((px + r * cos_t) - ox) * dens - 0.5
        my = ((py + r * sin_t) - oy) * dens - 0.5
        band = cv2.remap(np.ascontiguousarray(col.astype(np.float32)), mx.astype(np.float32),
                         my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rgb[:, i0:i1] = np.clip(band * 255.0 + 0.5, 0, 255).astype(np.uint8)
        log(f"[droste] band {b + 1}/{nb}  r {ra:.1f}-{rb:.1f} PU  raster {size}px  {time.time() - t0:.1f}s")
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(cache_dir) / key, rgb=rgb)
    return Texture(nu, u0, [{"rgb": rgb}]), None


def make_pyramid(tex):
    lv = _mips(tex.levels[0]["rgb"])
    tex.levels = [{"rgb": x} for x in lv]
    tex.mean = lv[-1].reshape(-1, 3).mean(0).astype(np.float32) / 255.0
    return tex


# ------------------------------------------------------------------ alive flags inside the Print Gallery
def _halve(src):
    f = src.astype(np.float32)
    return np.clip((f[0::2, 0::2] + f[1::2, 0::2] + f[0::2, 1::2] + f[1::2, 1::2]) * 0.25 + 0.5, 0, 255).astype(np.uint8)


class FlagBaker:
    """Re-lays the spot plate inside the log-polar texture every frame, so every nested Flag of the infinite
    page flutters (all in sync — it is the same Flag).  The texture itself is baked WITHOUT flags; flags are
    rasterised per frame at exactly the resolution the texture needs where they sit (1 texel = r * du PU),
    remapped into their log-polar tile and composited as unprinted yellow (no screen, nothing on top).
    flags = [(bbox_fn(T) -> (x0, y0, x1, y1) PU, draw_fn(ctx, T) drawing vx.flag in PU)]"""

    def __init__(self, tex, flags):
        self.tex = tex
        self.flags = flags
        self.base = [lv["rgb"] for lv in tex.levels]
        # halving levels are updated by region; the small odd tail is recomputed whole
        self.nhalf = 0
        for k in range(1, len(self.base)):
            h, w = self.base[k - 1].shape[:2]
            if h % 2 or w % 2 or self.base[k].shape[:2] != (h // 2, w // 2):
                break
            self.nhalf = k

    def _tile(self, bbox):
        """texel index ranges (j0, j1, i0, i1) (may exceed the texture: wrap) covering a page bbox"""
        tex = self.tex
        x0, y0, x1, y1 = bbox
        xs = np.linspace(x0, x1, 9)
        ys = np.linspace(y0, y1, 9)
        X, Y = np.meshgrid(xs, ys)
        dx, dy = X - FIX[0], Y - FIX[1]
        r = np.hypot(dx, dy)
        cx, cy = (x0 + x1) / 2 - FIX[0], (y0 + y1) / 2 - FIX[1]
        a0 = math.atan2(cy, cx)
        ang = a0 + np.arctan2(dy * math.cos(a0) - dx * math.sin(a0), dx * math.cos(a0) + dy * math.sin(a0))
        ur = np.log(np.maximum(r, 1e-6)) - tex.u0
        k = np.floor(ur.mean() / LN_S)
        ur = ur - k * LN_S                              # the flag's own lattice copy (levels 0/+1)
        i0, i1 = int(math.floor(ur.min() / tex.du)) - 2, int(math.ceil(ur.max() / tex.du)) + 3
        j0, j1 = int(math.floor(ang.min() / tex.dth)) - 2, int(math.ceil(ang.max() / tex.dth)) + 3
        return j0, j1, i0, i1, k, max(float(r.min()), 1.0)

    def frame(self, T):
        tex = self.tex
        NT, NR = self.base[0].shape[:2]
        lv0 = self.base[0].copy()
        dirty = []
        for bbox_fn, draw_fn in self.flags:
            bb = bbox_fn(T)
            if bb is None:
                continue
            j0, j1, i0, i1, k, rmin = self._tile(bb)
            dens = min(24.0, 1.0 / (rmin * tex.du))                # raster px per PU
            x0, y0, x1, y1 = bb
            wpx = int(math.ceil((x1 - x0) * dens)) + 4
            hpx = int(math.ceil((y1 - y0) * dens)) + 4
            cv = Canvas(s=1.0, w=wpx, h=hpx)
            c = cv.ctx
            c.scale(dens, dens)
            c.translate(-x0 + 2 / dens, -y0 + 2 / dens)
            draw_fn(c, T)
            fl = cv.rgba()
            if fl[..., 3].max() <= 0:
                continue
            jj = np.arange(j0, j1)
            ii = np.arange(i0, i1)
            th = jj[:, None] * tex.dth
            rr = np.exp(tex.u0 + (ii[None, :]) * tex.du + k * LN_S)
            zx = FIX[0] + rr * np.cos(th)
            zy = FIX[1] + rr * np.sin(th)
            mx = ((zx - x0) * dens + 2 - 0.5).astype(np.float32)
            my = ((zy - y0) * dens + 2 - 0.5).astype(np.float32)
            t = cv2.remap(fl, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            a = t[..., 3:4]
            if a.max() <= 0:
                continue
            rows = np.mod(jj, NT)
            cols = np.mod(ii, NR)
            dst = lv0[np.ix_(rows, cols)].astype(np.float32) / 255.0
            out = t[..., :3] + dst * (1.0 - a)
            lv0[np.ix_(rows, cols)] = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
            dirty.append((rows, cols))
        levels = [lv0]
        # propagate through the halving mips (dirty rows/cols only), recompute the small tail
        cur_dirty = dirty
        for L in range(1, len(self.base)):
            if L <= self.nhalf:
                arr = self.base[L].copy()
                nd = []
                for rows, cols in cur_dirty:
                    r2 = np.unique(rows // 2)
                    c2 = np.unique(cols // 2)
                    prev = levels[L - 1]
                    blk = prev[np.ix_(np.concatenate([2 * r2, 2 * r2 + 1]), np.concatenate([2 * c2, 2 * c2 + 1]))]
                    nr_, nc_ = len(r2), len(c2)
                    f = blk.astype(np.float32)
                    avg = (f[:nr_, :nc_] + f[nr_:, :nc_] + f[:nr_, nc_:] + f[nr_:, nc_:]) * 0.25
                    arr[np.ix_(r2, c2)] = np.clip(avg + 0.5, 0, 255).astype(np.uint8)
                    nd.append((r2, c2))
                cur_dirty = nd
                levels.append(arr)
            else:
                prev = levels[L - 1]
                h, w = self.base[L].shape[:2]
                levels.append(np.clip(cv2.resize(prev.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) + 0.5,
                                      0, 255).astype(np.uint8))
        out = Texture.__new__(Texture)
        out.__dict__.update(tex.__dict__)
        out.levels = [{"rgb": x} for x in levels]
        return out


# ------------------------------------------------------------------ sampling (the Escher map)
_GRID = {}


def _grid(h, w, s):
    k = (h, w, s)
    if k not in _GRID:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        _GRID[k] = ((xs + 0.5) / s, (ys + 0.5) / s)
    return _GRID[k]


def log_coords(fc, q, beta, C, psi=None, xy=None):
    """per pixel u = beta * log(x - q) + C and footprint |du| per pixel.  xy = optional (xs, ys) design-unit
    arrays (default: the whole frame).  psi = direction (radians) from q toward the frame: the branch cut of
    the log points the other way (cut at psi + pi)."""
    xs, ys = xy if xy is not None else _grid(fc.h, fc.w, fc.s)
    q = q if isinstance(q, complex) else complex(q[0], q[1])
    zx = xs - q.real
    zy = ys - q.imag
    if psi is None:
        psi = math.atan2(H / 2 - q.imag, W / 2 - q.real)
    cp, sp = math.cos(psi), math.sin(psi)
    rho = np.hypot(zx, zy)
    rho = np.maximum(rho, 1e-9)
    ang = np.arctan2(zy * cp - zx * sp, zx * cp + zy * sp) + psi
    lr = np.log(rho)
    br, bi = beta.real, beta.imag
    ur = br * lr - bi * ang + C.real
    ui = bi * lr + br * ang + C.imag
    foot = abs(beta) / (rho * fc.s)          # |du| per output pixel
    return ur, ui, foot, rho


def sample(tex, ur, ui, foot, rho=None, bias=0.0):
    """trilinear mip sample of the texture at log coords; returns float32 rgb (h, w, 3) in [0, 1]"""
    h, w = ur.shape
    period = LN_S
    fu = np.mod(ur - tex.u0, period) / period         # 0..1
    fv = np.mod(ui, TWO_PI) / TWO_PI
    lam = np.log2(np.maximum(foot / tex.du, 1e-6)) + bias
    nl = len(tex.levels)
    lam = np.clip(lam, 0.0, nl - 1 + 0.999)
    out = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w), np.float32)
    lo = np.floor(lam).astype(np.int32)
    fr = (lam - lo).astype(np.float32)
    for l in range(nl):
        wl = np.where(lo == l, 1.0 - fr, 0.0) + np.where(lo == l - 1, fr, 0.0)
        wl = wl.astype(np.float32)
        sel = wl > 1e-5
        if not sel.any():
            continue
        yy, xx = np.nonzero(sel)
        y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        img = tex.levels[l]["rgb"]
        th, tw = img.shape[:2]
        mx = (fu[y0:y1, x0:x1] * tw - 0.5).astype(np.float32)
        my = (fv[y0:y1, x0:x1] * th - 0.5).astype(np.float32)
        smp = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        ww = wl[y0:y1, x0:x1, None]
        out[y0:y1, x0:x1] += smp.astype(np.float32) * (ww / 255.0)
        wsum[y0:y1, x0:x1] += ww[..., 0]
    miss = 1.0 - np.clip(wsum, 0, 1)
    out += miss[..., None] * tex.mean
    return out


def droste(fc, tex, q, beta, C, psi=None, bias=0.0):
    ur, ui, foot, rho = log_coords(fc, q, beta, C, psi)
    return sample(tex, ur, ui, foot, rho, bias)


def droste_box(fc, tex, q, beta, C, psi, box, xmap=None, bias=0.0):
    """Droste sample for the pixel box (ix0, iy0, ix1, iy1) only.  xmap = (cam, wx0, wx1, full0): the page
    region [wx0, wx1] (PU, foreshortened by a turning page) shows the full page [full0, wx1] stretched."""
    ix0, iy0, ix1, iy1 = box
    ys, xs = np.mgrid[iy0:iy1, ix0:ix1].astype(np.float64)
    xs = (xs + 0.5) / fc.s
    ys = (ys + 0.5) / fc.s
    if xmap is not None:
        cam, wx0, wx1, full0 = xmap
        px = cam.cx + (xs - W / 2) / cam.zoom
        k = (wx1 - full0) / max(wx1 - wx0, 1e-6)
        px = wx1 - (wx1 - px) * k
        xs = W / 2 + (px - cam.cx) * cam.zoom
    ur, ui, foot, rho = log_coords(fc, q, beta, C, psi, xy=(xs, ys))
    if xmap is not None:
        foot = foot * k
    return sample(tex, ur, ui, foot, rho, bias)


def page_to_u(z):
    """base point (PU, complex) -> u = log(z - p)"""
    return np.log(complex(z) - complex(*FIX))

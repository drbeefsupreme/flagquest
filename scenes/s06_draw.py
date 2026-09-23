"""s06 helper: renders the 3D Tower (blocks, terraces, crown merlons, shouting people, babble bubbles)
through a Cam3 into cairo contexts. `ctx` = main layer, `gctx` = emissive layer (blurred + added later)."""
import math

import cairo
import numpy as np

from vx import W, H, C, col, clamp, hash01, smoothstep, ease_out_back
from vx.chars import draw_crowd

from . import s06_glyphs as G
from .s06_tower import STONE, MARBLE, MAP, SLOP, PHONE, ARCH, smooth_arr

AMB = np.array([0.25, 0.26, 0.34])
KEY_DIR = np.array([-0.55, 0.62, -0.56])
KEY_DIR = KEY_DIR / np.linalg.norm(KEY_DIR)
KEY_COL = np.array([0.60, 0.60, 0.74])
EG_COL = np.array([0.62, 0.86, 1.0])
FLASH_COL = np.array([0.88, 0.90, 1.0])
MORTAR = np.array([0.075, 0.07, 0.11])
PAPER = (0.95, 0.94, 0.90)
INK = col("ink")


class Light:
    """per-frame lighting environment"""

    def __init__(self, fog_col, eg_pos=None, eg_k=0.0, flash_pos=None, flash_k=0.0, fog_near=1800.0,
                 fog_far=16000.0, sea_fog=0.85, amb_k=1.0):
        self.fog_col = np.asarray(fog_col, np.float64)
        self.eg_pos = None if eg_pos is None else np.asarray(eg_pos, np.float64)
        self.eg_k, self.flash_k, self.fog_near, self.fog_far = eg_k, flash_k, fog_near, fog_far
        self.flash_pos = None if flash_pos is None else np.asarray(flash_pos, np.float64)
        self.sea_fog, self.amb_k = sea_fog, amb_k

    def shade(self, base, Cc, N, z, emissive=None):
        """base (n,3) albedo, Cc (n,3) centres, N (n,3) normals facing camera, z depth -> rgb (n,3), fog k"""
        lum = AMB[None, :] * self.amb_k + KEY_COL[None, :] * np.clip(N @ KEY_DIR, 0, 1)[:, None] * 0.75
        if self.eg_pos is not None and self.eg_k > 0:
            d = self.eg_pos[None, :] - Cc
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            L = d / dist
            fall = np.clip(1.0 - (dist[:, 0] - 800) / 5200, 0, 1) ** 1.5
            lum = lum + EG_COL[None, :] * (np.clip((N * L).sum(1), 0, 1) * 0.8 + 0.25)[:, None] * \
                (fall * self.eg_k)[:, None]
        if self.flash_pos is not None and self.flash_k > 0:
            d = self.flash_pos[None, :] - Cc
            L = d / np.linalg.norm(d, axis=1, keepdims=True)
            lum = lum + FLASH_COL[None, :] * (np.clip((N * L).sum(1), 0, 1) * 1.1 + 0.2)[:, None] * self.flash_k
        c = base * lum
        if emissive is not None:
            c = c + emissive
        fk = 1 - np.exp(-np.maximum(0, z - self.fog_near) / self.fog_far)
        fk = np.clip(fk + smooth_arr(-360, -640, Cc[:, 1]) * self.sea_fog, 0, 0.97)
        c = c * (1 - fk[:, None]) + self.fog_col[None, :] * fk[:, None]
        return np.clip(c, 0, 1.5), fk


# ------------------------------------------------------------------------------------------ helpers


def _quad(ctx, xs, ys):
    ctx.move_to(xs[0], ys[0])
    ctx.line_to(xs[1], ys[1])
    ctx.line_to(xs[2], ys[2])
    ctx.line_to(xs[3], ys[3])
    ctx.close_path()


def _unit_matrix(xs, ys):
    """affine taking the unit square (u right, v down) onto the quad TL,TR,BR,BL (approx)"""
    xx = ((xs[1] - xs[0]) + (xs[2] - xs[3])) / 2
    yx = ((ys[1] - ys[0]) + (ys[2] - ys[3])) / 2
    xy = ((xs[3] - xs[0]) + (xs[2] - xs[1])) / 2
    yy = ((ys[3] - ys[0]) + (ys[2] - ys[1])) / 2
    det = xx * yy - xy * yx
    if abs(det) < 1.0:
        return None
    return cairo.Matrix(xx, yx, xy, yy, xs[0], ys[0])


def _rgb(ctx, c, a=1.0, k=1.0):
    ctx.set_source_rgba(min(1, c[0] * k), min(1, c[1] * k), min(1, c[2] * k), a)


def _details(ctx, gctx, mat, c, seed, size, t, emis_k, scr):
    """material detail in unit-square space; ctx already transformed"""
    v = int(seed * 1000) % 3
    if mat == STONE:
        _rgb(ctx, c, 1, 1.22)
        ctx.rectangle(0, 0, 1, 0.07)
        ctx.fill()
        _rgb(ctx, c, 1, 0.74)
        ctx.rectangle(0, 0.9, 1, 0.1)
        ctx.fill()
        if size > 22:
            ctx.set_line_width(0.045)
            _rgb(ctx, c, 1, 0.68)
            if v == 0:
                ctx.rectangle(0.16, 0.22, 0.68, 0.52)
                ctx.stroke()
            elif v == 1:
                ctx.arc(0.5, 0.78, 0.3, math.pi, 2 * math.pi)
                ctx.stroke()
                ctx.arc(0.5, 0.78, 0.18, math.pi, 2 * math.pi)
                ctx.stroke()
            else:
                for yy in (0.34, 0.6):
                    ctx.move_to(0.08, yy)
                    ctx.line_to(0.92, yy)
                ctx.stroke()
    elif mat == MARBLE:
        n = 4 if v != 2 else 3
        ctx.set_line_width(0.05)
        for i in range(n):
            x = (i + 0.5) / n
            _rgb(ctx, c, 1, 0.8)
            ctx.move_to(x - 0.03, 0.06)
            ctx.line_to(x - 0.03, 0.94)
            ctx.stroke()
            _rgb(ctx, c, 1, 1.12)
            ctx.move_to(x + 0.03, 0.06)
            ctx.line_to(x + 0.03, 0.94)
            ctx.stroke()
        if v == 2 and size > 20:
            _rgb(ctx, c, 1, 0.72)
            ctx.set_line_width(0.04)
            for sx in (0.22, 0.78):
                for rr in (0.16, 0.09):
                    ctx.new_sub_path()
                    ctx.arc(sx, 0.2, rr, 0, 2 * math.pi)
                ctx.stroke()
        if v == 1:
            _rgb(ctx, MORTAR, 1)
            ctx.move_to(1, 0)
            ctx.line_to(0.72, 0)
            ctx.line_to(0.86, 0.22)
            ctx.line_to(1, 0.3)
            ctx.close_path()
            ctx.fill()
    elif mat == MAP:
        sea = (c[0] * 0.62, c[1] * 0.74, c[2] * 0.95)
        _rgb(ctx, sea, 1)
        ctx.move_to(0, 0.55 + 0.3 * (v - 1) * 0.5)
        ctx.curve_to(0.3, 0.3 + 0.2 * v, 0.6, 0.9, 1, 0.62 - 0.1 * v)
        ctx.line_to(1, 1)
        ctx.line_to(0, 1)
        ctx.close_path()
        ctx.fill()
        if size > 18:
            ctx.set_line_width(0.035)
            ctx.set_source_rgba(0.7, 0.83, 0.92, 0.9)
            ctx.move_to(0.1 + 0.2 * v, 0.0)
            ctx.curve_to(0.3, 0.3, 0.7, 0.2, 0.55, 0.62)
            ctx.stroke()
            ctx.set_source_rgba(0.08, 0.08, 0.12, 0.65)
            ctx.set_dash([0.07, 0.05])
            ctx.move_to(0.0, 0.18 + 0.1 * v)
            ctx.line_to(0.45, 0.35)
            ctx.line_to(0.62, 0.0)
            ctx.stroke()
            ctx.set_dash([])
            ctx.set_source_rgba(0.93, 0.92, 0.9, 0.9)
            for (x, y) in ((0.3, 0.25), (0.75, 0.3), (0.2, 0.42)):
                ctx.new_sub_path()
                ctx.arc(x, y, 0.035, 0, 2 * math.pi)
            ctx.fill()
    elif mat == SLOP:
        ctx.set_source_rgba(1, 1, 1, 0.22)
        ctx.rectangle(0.1, 0.1, 0.8, 0.5)
        ctx.fill()
        c2 = (c[2], c[0], c[1])
        _rgb(ctx, c2, 0.85)
        ctx.new_sub_path()
        ctx.arc(0.5, 0.36, 0.17, 0, 2 * math.pi)
        ctx.fill()
        if size > 16:
            _rgb(ctx, c, 0.9, 0.62)
            ctx.rectangle(0.12, 0.7, 0.74, 0.07)
            ctx.rectangle(0.12, 0.82, 0.5, 0.07)
            ctx.fill()
    elif mat == PHONE:
        fl = emis_k
        sc = (scr[0] * fl, scr[1] * fl, scr[2] * fl)
        ctx.rectangle(0.12, 0.1, 0.76, 0.8)
        _rgb(ctx, sc, 1)
        ctx.fill()
        if size > 14:
            ctx.set_source_rgba(1, 1, 1, 0.35)
            for i in range(3):
                ctx.rectangle(0.2, 0.2 + i * 0.2, 0.6 - 0.15 * ((i + v) % 2), 0.08)
            ctx.fill()
        if gctx is not None:
            gctx.set_matrix(ctx.get_matrix())
            gctx.rectangle(0.12, 0.1, 0.76, 0.8)
            _rgb(gctx, sc, 0.9)
            gctx.fill()
    elif mat == ARCH:
        _rgb(ctx, c, 1, 1.18)
        ctx.rectangle(0, 0, 1, 0.07)
        ctx.fill()

        def arch(cc):
            cc.move_to(0.22, 1.0)
            cc.line_to(0.22, 0.5)
            cc.arc(0.5, 0.5, 0.28, math.pi, 2 * math.pi)
            cc.line_to(0.78, 1.0)
            cc.close_path()
        arch(ctx)
        ctx.set_source_rgb(0.025, 0.025, 0.05)
        ctx.fill()
        lk = 0.35 + 0.65 * emis_k
        ctx.rectangle(0.36 + 0.1 * (v - 1), 0.74, 0.12, 0.16)
        ctx.set_source_rgba(scr[0], scr[1], scr[2], 0.9 * lk)
        ctx.fill()
        if gctx is not None:
            gctx.set_matrix(ctx.get_matrix())
            gctx.rectangle(0.3 + 0.1 * (v - 1), 0.68, 0.24, 0.28)
            gctx.set_source_rgba(scr[0] * lk, scr[1] * lk, scr[2] * lk, 0.6)
            gctx.fill()


def _crown(ctx, x, y, w, typ, c):
    """tiny crown silhouette standing on (x, y), width w (screen px). Never a flag, never a cross."""
    h = w * 0.72
    ctx.new_path()
    if typ == 0:
        pts = [(-0.5, 0), (-0.5, -0.55), (-0.3, -0.25), (-0.15, -0.9), (0, -0.35), (0.15, -0.9), (0.3, -0.25),
               (0.5, -0.55), (0.5, 0)]
    elif typ == 1:
        pts = [(-0.5, 0), (-0.55, -0.8), (-0.28, -0.45), (0, -1.0), (0.28, -0.45), (0.55, -0.8), (0.5, 0)]
    else:
        pts = [(-0.45, 0), (-0.5, -0.6), (-0.25, -0.75), (0, -0.62), (0.25, -0.75), (0.5, -0.6), (0.45, 0)]
    ctx.move_to(x + pts[0][0] * w, y + pts[0][1] * h)
    for px, py in pts[1:]:
        ctx.line_to(x + px * w, y + py * h)
    ctx.close_path()
    _rgb(ctx, c, 1)
    ctx.fill()
    _rgb(ctx, c, 1, 0.6)
    ctx.rectangle(x - 0.5 * w, y - 0.22 * h, w, 0.22 * h)
    ctx.fill()
    if w > 6:
        _rgb(ctx, c, 1, 1.35)
        for px in ((-0.15, 0.15) if typ == 0 else ((-0.55, 0, 0.55) if typ == 1 else (0,))):
            ctx.new_sub_path()
            ctx.arc(x + px * w, y - (1.0 if typ == 1 and px == 0 else 0.9 if typ == 0 else 0.8) * h, w * 0.07,
                    0, 2 * math.pi)
        ctx.fill()


CROWN_COLS = [np.array([0.78, 0.80, 0.86]), np.array([0.56, 0.58, 0.65]), np.array([0.55, 0.38, 0.33])]


# ------------------------------------------------------------------------------------------ main draw


def draw_tower(ctx, gctx, tw, cam, t, light, atlas=None, collapse=None, people_on=True, bubbles_on=True,
               crowd_t=None, lod=1.0, glitch=0.0, eg_draw=None, pre_tiers=None, bub_min=11.0, bub_alpha=1.0,
               bub_frac=1.0, orb=None, orb_k=1.0):
    """draw everything tower-related. Returns info dict (screen positions useful for fx)."""
    st = tw.blocks_at(t, collapse)
    P, N, landed, two, morph = st["P"], st["N"], st["landed"], st["two"], st["morph"]
    n = tw.nb
    sx, sy, z = cam.proj(P.reshape(-1, 3))
    sx, sy, z = sx.reshape(n, 4), sy.reshape(n, 4), z.reshape(n, 4)
    Cc = P.mean(1)
    tocam = cam.pos[None, :] - Cc
    facing = (N * tocam).sum(1)
    cosv = np.abs(facing) / (np.linalg.norm(tocam, axis=1) + 1e-6)
    Nf = np.where(facing[:, None] < 0, -N, N)
    vis = (z.min(1) > 60) & ((facing > 0) | two)
    vis &= (sx.max(1) > -80) & (sx.min(1) < W + 80) & (sy.max(1) > -80) & (sy.min(1) < H + 80)
    zc = z.mean(1)
    lit, fk = light.shade(tw.b_col, Cc, Nf, zc)
    # emissive flicker for phones / arches
    fl = 0.55 + 0.45 * np.sin(t * (7 + 9 * tw.b_seed) + tw.b_seed * 40) ** 2
    if glitch > 0:
        fl = fl * (1 - glitch * (np.sin(t * 60 + tw.b_seed * 90) > 0.3))
    size = np.hypot(sx[:, 1] - sx[:, 0], sy[:, 1] - sy[:, 0]) + np.hypot(sx[:, 3] - sx[:, 0], sy[:, 3] - sy[:, 0])
    size *= 0.5
    mort, _ = light.shade(np.tile(MORTAR, (n, 1)), Cc, Nf, zc)

    # axis depth for splitting free-flying blocks
    axd = cam.proj(np.stack([tw.axis_y(Cc[:, 1])[0], Cc[:, 1], tw.axis_y(Cc[:, 1])[1]], -1))[2]
    free = vis & (~landed | two)
    back_free = np.nonzero(free & (zc > axd))[0]
    front_free = np.nonzero(free & (zc <= axd))[0]
    tier_vis = [np.nonzero(vis & landed & ~two & (tw.b_tier == k))[0] for k in range(tw.NT)]

    def side_faces(idx):
        """projected + shaded side/top/bottom faces of protruding landed blocks -> list per block"""
        k = tw.b_tier[idx]
        ax, az = tw.ax[k], tw.az[k]
        rb, rw = tw.b_r[idx], tw.r[k] - 2.0
        th0, th1, y0, y1 = tw.b_th0[idx], tw.b_th1[idx], tw.b_y0[idx], tw.b_y1[idx]

        def pt(th, r, y):
            return np.stack([ax + r * np.cos(th), y, az + r * np.sin(th)], -1)
        faces = [
            (np.stack([pt(th0, rw, y1), pt(th0, rb, y1), pt(th0, rb, y0), pt(th0, rw, y0)], 1),
             np.stack([np.sin(th0), 0 * th0, -np.cos(th0)], -1), 0.78),
            (np.stack([pt(th1, rb, y1), pt(th1, rw, y1), pt(th1, rw, y0), pt(th1, rb, y0)], 1),
             np.stack([-np.sin(th1), 0 * th1, np.cos(th1)], -1), 0.78),
            (np.stack([pt(th0, rw, y1), pt(th1, rw, y1), pt(th1, rb, y1), pt(th0, rb, y1)], 1),
             np.tile([0.0, 1.0, 0.0], (len(idx), 1)), 1.05),
            (np.stack([pt(th0, rb, y0), pt(th1, rb, y0), pt(th1, rw, y0), pt(th0, rw, y0)], 1),
             np.tile([0.0, -1.0, 0.0], (len(idx), 1)), 0.55),
        ]
        out = []
        for Fp, Fn, kk in faces:
            Fc = Fp.mean(1)
            v = (Fn * (cam.pos[None, :] - Fc)).sum(1) > 0
            fx, fy, fz = cam.proj(Fp.reshape(-1, 3))
            col_, _ = light.shade(tw.b_col[idx] * kk, Fc, Fn, fz.reshape(-1, 4).mean(1))
            out.append((v, fx.reshape(-1, 4), fy.reshape(-1, 4), col_))
        return out

    def draw_blocks(idx, sort=True, tier=False):
        if len(idx) == 0:
            return
        if tier:
            idx = idx[np.argsort(tw.b_r[idx] - tw.r[tw.b_tier[idx]])]
            sf = side_faces(idx)
        elif sort:
            idx = idx[np.argsort(-zc[idx])]
        for j, i in enumerate(idx):
            xs, ys = sx[i], sy[i]
            sz = size[i]
            if tier and sz > 5 and tw.b_r[i] - tw.r[tw.b_tier[i]] > 3:
                for (v, fx, fy, fcol) in sf:
                    if v[j]:
                        _quad(ctx, fx[j], fy[j])
                        _rgb(ctx, fcol[j], 1)
                        ctx.fill()
            if sz < 3.0 * lod:
                _quad(ctx, xs, ys)
                _rgb(ctx, lit[i], 1)
                ctx.fill()
                continue
            mo = morph[i]
            if not tier and mo < 0.999:
                # a shard/card still in flight: its own outline, squared into a brick on arrival
                M = _unit_matrix(xs, ys)
                if M is None:
                    continue
                sq = tw.b_shard[i]
                d = sq - 0.5
                lim = np.maximum(np.abs(d).max(1, keepdims=True), 1e-3)
                pts = sq * (1 - mo) + (0.5 + d / lim * 0.5) * mo
                ctx.save()
                ctx.transform(M)
                ctx.move_to(*pts[0])
                for p in pts[1:]:
                    ctx.line_to(*p)
                ctx.close_path()
                _rgb(ctx, lit[i], 1)
                ctx.fill_preserve()
                ctx.save()
                ctx.clip_preserve()
                if gctx is not None:
                    gctx.save()
                if sz > 9 * lod:
                    mm = tw.b_mat[i] if tw.b_mat[i] != ARCH else STONE
                    edge = min(1.0, 1.6 * cosv[i]) ** 0.7     # phones seen edge-on barely glow
                    _details(ctx, gctx, mm, lit[i], tw.b_seed[i], sz, t, fl[i] * (1 - fk[i] * 0.6) * edge,
                             tw.b_scr[i] * (1 - fk[i] * 0.5))
                if gctx is not None:
                    gctx.restore()
                ctx.restore()
                ctx.set_line_width(max(0.004, 1.3 / max(sz, 1)))
                _rgb(ctx, lit[i], 0.9, 1.45)
                ctx.stroke()
                ctx.restore()
                continue
            # mortar (slightly larger) then inset face
            _quad(ctx, xs, ys)
            _rgb(ctx, mort[i], 1)
            ctx.set_line_width(max(0.8, sz * 0.05))
            ctx.fill_preserve()
            ctx.stroke()
            mcx, mcy = xs.mean(), ys.mean()
            g = 0.9
            ix, iy = mcx + (xs - mcx) * g, mcy + (ys - mcy) * g
            mat = tw.b_mat[i]
            if sz > 9 * lod:
                M = _unit_matrix(ix, iy)
            else:
                M = None
            if M is None:
                _quad(ctx, ix, iy)
                c = lit[i] if mat != PHONE else lit[i] * 0.6 + tw.b_scr[i] * fl[i] * 0.3
                _rgb(ctx, c, 1)
                ctx.fill()
                continue
            ctx.save()
            ctx.transform(M)
            if mat == SLOP:
                _rrect_unit(ctx, 0.14)
            elif mat == PHONE:
                _rrect_unit(ctx, 0.12)
            else:
                ctx.rectangle(0, 0, 1, 1)
            _rgb(ctx, lit[i], 1)
            ctx.fill()
            if gctx is not None:
                gctx.save()
            _details(ctx, gctx, mat, lit[i], tw.b_seed[i], sz, t, fl[i] * (1 - fk[i] * 0.6),
                     tw.b_scr[i] * (1 - fk[i] * 0.5))
            if gctx is not None:
                gctx.restore()
            ctx.restore()

    # ---- people / crowns / floors per terrace
    pp = tw.people_pos()
    psx, psy, pz = cam.proj(pp)
    ph = cam.scale_at(pz) * tw.PERSON_H * tw.p_hs
    pn = np.stack([np.cos(tw.p_th), np.zeros(tw.np), np.sin(tw.p_th)], -1)
    pface = (pn * (cam.pos[None, :] - pp)).sum(1)
    app = tw.tier_done[tw.p_tier] + 0.12 + tw.p_rnd * 0.5
    pvis = (pface > -0.15 * np.linalg.norm(cam.pos[None, :] - pp, axis=1)) & (pz > 80) & (t > app) & \
        (psx > -60) & (psx < W + 60) & (psy > -60) & (psy < H + 160) & (ph > 1.2)
    _, pfk = light.shade(np.ones((tw.np, 3)), pp, np.tile([0, 1.0, 0], (tw.np, 1)), pz)
    bubble_list = []
    ct = t if crowd_t is None else crowd_t

    def draw_people(k):
        if not people_on:
            return
        idx = np.nonzero(pvis & (tw.p_tier == k))[0]
        if len(idx) == 0:
            return
        idx = idx[np.argsort(-pz[idx])]
        big = []
        for i in idx:
            h = ph[i]
            pop = ease_out_back(clamp((t - app[i]) / 0.25))
            if pop <= 0.01:
                continue
            h *= max(0.05, pop)
            fogk = pfk[i]
            x, y = psx[i], psy[i]
            if h < 11 * lod:
                _tiny_person(ctx, gctx, x, y, h, tw.p_seed[i], ct, fogk, light.fog_col, tw.p_prop[i])
            else:
                kind = tw.p_kind[i]
                prop = "phone" if tw.p_prop[i] < 0.62 else ("megaphone" if tw.p_prop[i] < 0.86 else None)
                pose = "shout" if prop != "megaphone" else "cup_shout"
                if prop is None:
                    pose = ["point", "declare", "cup_shout"][tw.p_seed[i] % 3]
                d = dict(x=x, y=y, h=h, kind=kind, seed=int(tw.p_seed[i] % 997), pose=pose,
                         facing=int(tw.p_face[i]), mouth=0.55 + 0.45 * math.sin(ct * 13 + tw.p_seed[i]),
                         expr="furious" if tw.p_seed[i] % 2 else "angry", crown=bool(tw.p_seed[i] % 5 == 0),
                         tint=(tuple(light.fog_col), float(min(0.95, fogk + 0.08))), t_off=float(tw.p_rnd[i] * 9))
                if prop:
                    d["prop"] = prop
                    d["phone"] = prop == "phone"
                big.append(d)
            if bubbles_on and h >= 2.0 and atlas is not None and hash01(int(tw.p_seed[i]), 12) < bub_frac:
                bubble_list.append((pz[i], i, x, y, h, fogk))
        if big:
            anchors = draw_crowd(ctx, big, ct)
            if gctx is not None and anchors:
                for a, d in zip(anchors, big):
                    if d.get("prop") == "phone" and a:
                        px, py = a.get("prop", a.get("hand_r", (d["x"], d["y"] - d["h"] * 0.6)))
                        gctx.set_source_rgba(0.62, 0.85, 1.0, 0.55 * (1 - d["tint"][1]))
                        gctx.new_sub_path()
                        gctx.arc(px, py, max(1.5, d["h"] * 0.06), 0, 2 * math.pi)
                        gctx.fill()

    cr_pos = np.stack([tw.ax[tw.c_tier] + tw.r[tw.c_tier] * np.cos(tw.c_th), tw.y1[tw.c_tier],
                       tw.az[tw.c_tier] + tw.r[tw.c_tier] * np.sin(tw.c_th)], -1)
    csx, csy, cz = cam.proj(cr_pos)
    cn = np.stack([np.cos(tw.c_th), np.zeros(len(tw.c_th)), np.sin(tw.c_th)], -1)
    cvis = ((cn * (cam.pos[None, :] - cr_pos)).sum(1) > 0) & (cz > 80) & (csx > -40) & (csx < W + 40) & \
        (csy > -40) & (csy < H + 60)
    cw = cam.scale_at(cz) * 44 * tw.c_s
    ccol_lit = [light.shade(np.tile(cc, (len(cz), 1)), cr_pos, cn, cz)[0] for cc in CROWN_COLS]

    def draw_crowns(k):
        idx = np.nonzero(cvis & (tw.c_tier == k))[0]
        for i in idx:
            pop = ease_out_back(clamp((t - tw.tier_done[k] - 0.05 - tw.c_rnd[i] * 0.3) / 0.22))
            if pop <= 0.02 or cw[i] < 1.2:
                continue
            _crown(ctx, csx[i], csy[i] + 1, cw[i] * pop, tw.c_type[i], ccol_lit[int(tw.c_rnd[i] * 3) % 3][i])

    def draw_floor(k):
        if cam.pos[1] <= tw.y1[k] or t < tw.tier_done[k]:
            return
        th = np.linspace(0, 2 * math.pi, 56, endpoint=False)
        pts = np.stack([tw.ax[k] + tw.r[k] * np.cos(th), np.full_like(th, tw.y1[k]),
                        tw.az[k] + tw.r[k] * np.sin(th)], -1)
        fx, fy, fz = cam.proj(pts)
        if fz.min() < 60:
            return
        base = np.array([[0.30, 0.30, 0.36]])
        c, _ = light.shade(base, np.array([[tw.ax[k], tw.y1[k], tw.az[k]]]), np.array([[0, 1.0, 0]]),
                           np.array([fz.mean()]))
        ctx.move_to(fx[0], fy[0])
        for a, b in zip(fx[1:], fy[1:]):
            ctx.line_to(a, b)
        ctx.close_path()
        _rgb(ctx, c[0], 1)
        ctx.fill()

    # ---- loose orbiting fragments (vortex around the tower)
    orb_back, orb_front = None, None
    if orb is not None and orb_k > 0:
        oP, oN = orb.at(t, tw)
        ox, oy, oz = cam.proj(oP.reshape(-1, 3))
        ox, oy, oz = ox.reshape(-1, 4), oy.reshape(-1, 4), oz.reshape(-1, 4)
        oC = oP.mean(1)
        ofc = (oN * (cam.pos[None, :] - oC)).sum(1)
        oNf = np.where(ofc[:, None] < 0, -oN, oN)
        ozc = oz.mean(1)
        olit, ofk = light.shade(orb.col, oC, oNf, ozc)
        ovis = (oz.min(1) > 90) & (ox.max(1) > -200) & (ox.min(1) < W + 200) & (oy.max(1) > -200) & \
            (oy.min(1) < H + 200) & (t > orb.appear) & (oC[:, 1] < tw.top + 250)
        oaxd = cam.proj(np.stack([tw.axis_y(oC[:, 1])[0], oC[:, 1], tw.axis_y(oC[:, 1])[1]], -1))[2]
        ob = np.nonzero(ovis & (ozc > oaxd))[0]
        of = np.nonzero(ovis & (ozc <= oaxd))[0]
        orb_back = ob[np.argsort(-ozc[ob])]
        orb_front = of[np.argsort(-ozc[of])]

        def draw_orb(idx):
            for i in idx:
                xs, ys = ox[i], oy[i]
                a = clamp((t - orb.appear[i]) / 0.3) * orb_k
                _quad(ctx, xs, ys)
                _rgb(ctx, MORTAR * 0.8, a)
                ctx.set_line_width(2.0)
                ctx.fill_preserve()
                ctx.stroke()
                mcx, mcy = xs.mean(), ys.mean()
                ix, iy = mcx + (xs - mcx) * 0.9, mcy + (ys - mcy) * 0.9
                M = _unit_matrix(ix, iy)
                if M is None:
                    continue
                ctx.save()
                ctx.transform(M)
                ctx.rectangle(0, 0, 1, 1)
                _rgb(ctx, olit[i], a)
                ctx.fill()
                if gctx is not None:
                    gctx.save()
                _details(ctx, gctx, int(orb.mat[i]), olit[i], orb.seed[i], 40.0, t, 0.9 * (1 - ofk[i] * 0.6),
                         orb.scr[i])
                if gctx is not None:
                    gctx.restore()
                ctx.restore()

    # ---- draw order
    if orb_back is not None:
        draw_orb(orb_back)
    draw_blocks(back_free)
    if pre_tiers is not None:
        pre_tiers()
    order = np.argsort(-np.abs(tw.ym - cam.pos[1]))
    for k in order:
        draw_floor(k)
        draw_blocks(tier_vis[k], tier=True)
        draw_crowns(k)
        if k >= 1:
            draw_people(k - 1)
    if eg_draw is not None:
        eg_draw()
    draw_blocks(front_free)
    if orb_front is not None:
        draw_orb(orb_front)
    if bubbles_on and bubble_list:
        bubble_list.sort(key=lambda b: -b[0])
        for zb, i, x, y, h, fogk in bubble_list:
            _bubble(ctx, gctx, atlas, tw, i, x, y, h, fogk, t, app[i], glitch, bub_min, bub_alpha)
    return dict(P=P, sx=sx, sy=sy, z=z, vis=vis, psx=psx, psy=psy, ph=ph, pvis=pvis)


def _rrect_unit(ctx, r):
    ctx.new_sub_path()
    ctx.arc(1 - r, r, r, -math.pi / 2, 0)
    ctx.arc(1 - r, 1 - r, r, 0, math.pi / 2)
    ctx.arc(r, 1 - r, r, math.pi / 2, math.pi)
    ctx.arc(r, r, r, math.pi, 1.5 * math.pi)
    ctx.close_path()


def _tiny_person(ctx, gctx, x, y, h, seed, t, fogk, fogc, prop):
    """cheap LOD figure for very small people: body + head, jittering shout, phone glint"""
    s = int(seed)
    cols = [(0.49, 0.52, 0.58), (0.27, 0.35, 0.48), (0.25, 0.43, 0.43), (0.42, 0.29, 0.42), (0.54, 0.29, 0.23),
            (0.36, 0.37, 0.42)]
    c = np.array(cols[s % len(cols)]) * 0.8
    c = c * (1 - fogk) + np.asarray(fogc) * fogk
    ctx.set_source_rgb(*c)
    w = h * 0.32
    ctx.move_to(x - w / 2, y)
    ctx.line_to(x - w * 0.3, y - h * 0.72)
    ctx.line_to(x + w * 0.3, y - h * 0.72)
    ctx.line_to(x + w / 2, y)
    ctx.close_path()
    ctx.fill()
    jit = math.sin(t * 17 + s) * h * 0.02
    ctx.new_sub_path()
    ctx.arc(x + jit, y - h * 0.84, h * 0.13, 0, 2 * math.pi)
    ctx.fill()
    # arm up (shouting / holding phone)
    ctx.set_line_width(max(0.6, h * 0.07))
    side = 1 if s % 2 else -1
    ctx.move_to(x + side * w * 0.25, y - h * 0.68)
    ctx.line_to(x + side * w * 0.55, y - h * 0.86 + jit)
    ctx.stroke()
    if prop < 0.62 and gctx is not None:
        k = 1 - fogk
        gctx.set_source_rgba(0.62 * k, 0.85 * k, 1.0 * k, 0.9)
        gctx.new_sub_path()
        gctx.arc(x + side * w * 0.55, y - h * 0.86 + jit, max(0.8, h * 0.07), 0, 2 * math.pi)
        gctx.fill()


def _bubble(ctx, gctx, atlas, tw, i, x, y, h, fogk, t, app, glitch, bub_min=11.0, bub_alpha=1.0):
    seed = int(tw.p_seed[i])
    per = 0.75 + 0.9 * hash01(seed, 3)
    ph = hash01(seed, 4) * per
    tt = t - app - 0.15 + ph
    if tt < ph:
        return
    cyc = int(tt // per)
    tau = (tt - cyc * per) / per
    on = 0.8
    if tau > on:
        return
    grow = ease_out_back(clamp(tau * per / 0.13))
    shrink = clamp((on - tau) * per / 0.07)
    s = grow * shrink
    if s < 0.05:
        return
    nstr = len(atlas["masks"])
    gi = int(hash01(seed * 31 + cyc, 5) * nstr) % nstr
    m = atlas["masks"][gi]
    mh, mw = m.shape
    hb = max(bub_min, h * 0.62) * (0.9 + 0.35 * hash01(seed + cyc, 6))
    th = hb * 0.5
    tw_ = th * mw / mh
    maxw = hb * 6.5
    if tw_ > maxw:
        th *= maxw / tw_
        tw_ = maxw
    bw = tw_ + hb * 0.55
    side = 1 if int(tw.p_face[i]) > 0 else -1
    if hash01(seed + cyc, 7) < 0.3:
        side = -side
    bx = x + side * (bw * 0.35 + h * 0.1)
    by = y - h * 1.12 - hb * 0.55 - hash01(seed + cyc, 8) * h * 0.35
    if glitch > 0:
        bx += (hash01(seed + int(t * 24), 9) - 0.5) * glitch * 30
    style = hash01(seed * 7 + cyc, 10)
    alpha = (1 - fogk * 0.75) * bub_alpha
    ctx.save()
    ctx.translate(bx, by)
    ctx.scale(s, s)
    # tail toward the mouth
    mx, my = x - bx, (y - h * 0.86) - by
    if style < 0.2:
        fill, tcol = (0.10, 0.11, 0.18), (0.62, 0.86, 1.0)
    elif style < 0.3:
        pastel = [(0.91, 0.78, 0.82), (0.72, 0.88, 0.91), (0.79, 0.75, 0.66)]
        fill, tcol = pastel[seed % 3], INK
    else:
        fill, tcol = PAPER, INK
    burst = 0.62 < style < 0.8
    ctx.move_to(-bw * 0.08 * side, hb * 0.3)
    ctx.line_to(mx * 0.75, my * 0.75)
    ctx.line_to(bw * 0.12 * side, hb * 0.3)
    ctx.close_path()
    ctx.set_source_rgba(*fill, alpha)
    ctx.fill()
    if burst:
        nsp = 14
        ctx.new_path()
        for k in range(nsp * 2):
            a = k / (nsp * 2) * 2 * math.pi
            rr = 1.0 if k % 2 == 0 else 0.78 + 0.1 * hash01(seed + k, 11)
            px, py = math.cos(a) * bw * 0.62 * rr, math.sin(a) * hb * 0.78 * rr
            ctx.line_to(px, py)
        ctx.close_path()
    else:
        r = hb / 2
        ctx.new_sub_path()
        ctx.arc(bw / 2 - r, 0, r, -math.pi / 2, math.pi / 2)
        ctx.arc(-bw / 2 + r, 0, r, math.pi / 2, 1.5 * math.pi)
        ctx.close_path()
    ctx.set_source_rgba(*fill, alpha)
    ctx.fill_preserve()
    ctx.set_line_width(max(0.7, hb * 0.05))
    ctx.set_source_rgba(0.08, 0.08, 0.12, alpha * 0.8)
    ctx.stroke()
    G.draw_strip(ctx, atlas, gi, 0, 0, th, tcol, alpha)
    ctx.restore()
    if style < 0.2 and gctx is not None:
        gctx.save()
        gctx.translate(bx, by)
        gctx.scale(s, s)
        G.draw_strip(gctx, atlas, gi, 0, 0, th, (0.3, 0.45, 0.6), alpha * 0.8)
        gctx.restore()

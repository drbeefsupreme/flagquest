"""Painter for vx.chars (owned by RigMaster). Draws a solved Rig in LOCAL units (h/100, y down, x = facing dir).

Style: luminous flat vector, two-tone cel shading (shadow crescent cut by a light-shifted copy of each
shape), optional rim light, no dark outlines. Faces are projected from a sphere so 3/4, profile and back
views stay consistent; every feature is culled when it turns away.
"""
import math

import cairo

from .canvas import col
from .ease import noise1, hash01, clamp

TAU = math.tau
BIG = 4000.0


def mixc(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


SHADOW_TINT = (0.20, 0.16, 0.34)


def shadow_of(c, k=0.72):
    c = col(c)
    return mixc((c[0] * k, c[1] * k, c[2] * k), SHADOW_TINT, 0.16)


def light_of(c, k=0.14):
    c = col(c)
    return mixc(c, (1.0, 0.98, 0.94), k)


# ============================================================ path primitives
def p_circle(ctx, x, y, r):
    ctx.move_to(x + r, y)
    ctx.arc(x, y, r, 0, TAU)
    ctx.close_path()


def p_ellipse(ctx, x, y, rx, ry, rot=0.0):
    if rx <= 1e-6 or ry <= 1e-6:
        return
    m = ctx.get_matrix()
    ctx.translate(x, y)
    if rot:
        ctx.rotate(rot)
    ctx.scale(rx, ry)
    ctx.move_to(1, 0)
    ctx.arc(0, 0, 1, 0, TAU)
    ctx.close_path()
    ctx.set_matrix(m)


def p_capsule(ctx, p1, r1, p2, r2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy)
    if L <= abs(r1 - r2) + 1e-6:
        if r1 >= r2:
            p_circle(ctx, p1[0], p1[1], r1)
        else:
            p_circle(ctx, p2[0], p2[1], r2)
        return
    th = math.atan2(dy, dx)
    al = math.acos(max(-1.0, min(1.0, (r1 - r2) / L)))
    ctx.new_sub_path()
    ctx.arc(p1[0], p1[1], r1, th + al, th - al + TAU)
    ctx.arc(p2[0], p2[1], r2, th - al, th + al)
    ctx.close_path()


def p_chain(ctx, pts, rs):
    for i in range(len(pts) - 1):
        p_capsule(ctx, pts[i], rs[i], pts[i + 1], rs[i + 1])


def p_poly(ctx, pts):
    if len(pts) < 3:
        return
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()


def p_smooth(ctx, pts, close=True, k=0.33):
    n = len(pts)
    if n < 3:
        p_poly(ctx, pts)
        return
    ctx.move_to(*pts[0])
    rng = range(n) if close else range(n - 1)
    for i in rng:
        p0 = pts[(i - 1) % n] if close else pts[max(i - 1, 0)]
        p1 = pts[i % n]
        p2 = pts[(i + 1) % n] if close else pts[min(i + 1, n - 1)]
        p3 = pts[(i + 2) % n] if close else pts[min(i + 2, n - 1)]
        ctx.curve_to(p1[0] + (p2[0] - p0[0]) * k * 0.5, p1[1] + (p2[1] - p0[1]) * k * 0.5,
                     p2[0] - (p3[0] - p1[0]) * k * 0.5, p2[1] - (p3[1] - p1[1]) * k * 0.5, p2[0], p2[1])
    if close:
        ctx.close_path()


def p_leaf(ctx, x, y, L, w, ang):
    """almond/leaf shape from (x,y) along ang"""
    c, s = math.cos(ang), math.sin(ang)
    tx, ty = x + c * L, y + s * L
    nx, ny = -s * w, c * w
    mx, my = x + c * L * 0.5, y + s * L * 0.5
    ctx.move_to(x, y)
    ctx.curve_to(mx + nx, my + ny, mx + nx, my + ny, tx, ty)
    ctx.curve_to(mx - nx, my - ny, mx - nx, my - ny, x, y)
    ctx.close_path()


# ============================================================ head sphere projection
class HP:
    def __init__(self, r, yaw, pitch):
        self.r = r
        self.cy, self.sy = math.cos(yaw), math.sin(yaw)
        self.cp, self.sp = math.cos(pitch), math.sin(pitch)
        self.yaw = yaw

    def rot(self, X, Y, Z):
        Y1 = Y * self.cp + Z * self.sp
        Z1 = Z * self.cp - Y * self.sp
        return X * self.cy + Z1 * self.sy, Y1, Z1 * self.cy - X * self.sy

    def pt(self, lon, lat, rad=1.0):
        cl = math.cos(lat)
        X, Y, Z = self.rot(math.sin(lon) * cl, math.sin(lat), math.cos(lon) * cl)
        return (X * self.r * rad, -Y * self.r * rad, Z)

    def vec(self, v, rad=1.0):
        X, Y, Z = self.rot(*v)
        return (X * self.r * rad, -Y * self.r * rad, Z)

    def cap(self, lon, lat, gam, rad=1.0, push=1.0, n=30, jag=None):
        cl = math.cos(lat)
        A = (math.sin(lon) * cl, math.sin(lat), math.cos(lon) * cl)
        up = (0.0, 1.0, 0.0) if abs(A[1]) < 0.95 else (0.0, 0.0, 1.0)
        d = A[0] * up[0] + A[1] * up[1] + A[2] * up[2]
        E2 = (up[0] - A[0] * d, up[1] - A[1] * d, up[2] - A[2] * d)
        l2 = math.sqrt(E2[0] ** 2 + E2[1] ** 2 + E2[2] ** 2) or 1.0
        E2 = (E2[0] / l2, E2[1] / l2, E2[2] / l2)
        E1 = (E2[1] * A[2] - E2[2] * A[1], E2[2] * A[0] - E2[0] * A[2], E2[0] * A[1] - E2[1] * A[0])
        R = self.r * rad
        cosg = math.cos(gam)
        pts, zs = [], []
        for i in range(n):
            s = TAU * i / n
            g = gam - (jag(s) if jag else 0.0)
            cg, sg = math.cos(g), math.sin(g)
            cs, ss = math.cos(s), math.sin(s)
            v = (A[0] * cg + (E1[0] * cs + E2[0] * ss) * sg, A[1] * cg + (E1[1] * cs + E2[1] * ss) * sg,
                 A[2] * cg + (E1[2] * cs + E2[2] * ss) * sg)
            X, Y, Z = self.rot(*v)
            pts.append((X * R, -Y * R))
            zs.append(Z)
        if min(zs) >= 0:
            return pts
        # rotated axis, to test which rim arc lies inside the cap
        Ax, Ay, Az = self.rot(*A)
        Rp = R * push
        if max(zs) < 0:
            if Az >= cosg:  # the whole visible disc is inside the cap
                return [(math.cos(TAU * i / n) * Rp, math.sin(TAU * i / n) * Rp) for i in range(n)]
            return []
        # one contiguous visible run; start right after a hidden sample
        k0 = next(i for i in range(n) if zs[i] >= 0 and zs[i - 1] < 0)
        run = []
        i = k0
        # entry crossing
        a, b = (i - 1) % n, i
        f = zs[a] / (zs[a] - zs[b])
        run.append((pts[a][0] + (pts[b][0] - pts[a][0]) * f, pts[a][1] + (pts[b][1] - pts[a][1]) * f))
        while zs[i % n] >= 0 and len(run) <= n:
            run.append(pts[i % n])
            i += 1
        a, b = (i - 1) % n, i % n
        f = zs[a] / (zs[a] - zs[b])
        run.append((pts[a][0] + (pts[b][0] - pts[a][0]) * f, pts[a][1] + (pts[b][1] - pts[a][1]) * f))
        # close along the silhouette rim through the side that is inside the cap
        a_end = math.atan2(run[-1][1], run[-1][0])
        a_start = math.atan2(run[0][1], run[0][0])
        d_ccw = (a_start - a_end) % TAU
        mid = a_end + d_ccw * 0.5
        # screen (x, y) -> sphere (X, -Y, 0)
        inside = (math.cos(mid) * Ax - math.sin(mid) * Ay) >= cosg
        sweep = d_ccw if inside else d_ccw - TAU
        m = max(4, int(abs(sweep) / TAU * n) + 2)
        out = list(run)
        if push != 1.0:
            out.append((math.cos(a_end) * Rp, math.sin(a_end) * Rp))
        for j in range(1, m):
            ang = a_end + sweep * j / m
            out.append((math.cos(ang) * Rp, math.sin(ang) * Rp))
        if push != 1.0:
            out.append((math.cos(a_start) * Rp, math.sin(a_start) * Rp))
        return out


# ============================================================ painter
class Painter:
    def __init__(self, ctx, ch, rig, lod, o):
        self.ctx = ctx
        self.ch = ch
        self.rig = rig
        self.sp = rig.sp
        self.lod = lod
        self.o = o
        self.t = o["t"]
        f = o["facing_sign"]
        lx, ly = o["light"]
        ln = math.hypot(lx, ly) or 1.0
        self.lv = (lx * f / ln, ly / ln)
        self.sil = o.get("silhouette")
        self.sil = col(self.sil) if self.sil is not None else None
        tint = o.get("tint")
        self.tint = (col(tint[0]), float(tint[1])) if tint else None
        rim = o.get("rim")
        self.rim = None
        if rim:
            rc = col(rim[0])
            rs = float(rim[1]) if len(rim) > 1 else 0.8
            rd = rim[2] if len(rim) > 2 else (-o["light"][0] or 0.8, -0.35)
            rl = math.hypot(*rd) or 1.0
            self.rim = (rc, rs, (rd[0] * f / rl, rd[1] / rl))
        self.shade_k = o.get("shade", 1.0)
        self.pal = ch.pal

    # ---------------------------------------------------------------- colour
    def C(self, c):
        if self.sil is not None:
            return self.sil
        c = col(self.pal.get(c, c)) if isinstance(c, str) else c
        if self.tint:
            c = mixc(c, self.tint[0], self.tint[1])
        return c

    def src(self, c, a=1.0):
        r, g, b = self.C(c)
        self.ctx.set_source_rgba(r, g, b, a)

    def raw(self, c, a=1.0):
        """colour that ignores silhouette (glows) but honours tint"""
        c = col(self.pal.get(c, c)) if isinstance(c, str) else c
        if self.tint:
            c = mixc(c, self.tint[0], self.tint[1] * 0.5)
        self.ctx.set_source_rgba(c[0], c[1], c[2], a)

    # ---------------------------------------------------------------- shaded part
    def part(self, build, base, sh=None, d=1.6, rim=1.0, a=1.0, hi=None):
        ctx = self.ctx
        ctx.new_path()
        build(ctx)
        path = ctx.copy_path()
        bc = self.C(base)
        if self.sil is not None:
            ctx.set_source_rgba(bc[0], bc[1], bc[2], a)
            ctx.fill()
            self._rim(path, d, rim, a)
            return path
        if self.lod >= 1 and d > 0 and self.shade_k > 0:
            scol = self.C(sh) if sh is not None else self.C(shadow_of(self.pal.get(base, base) if isinstance(base, str) else base))
            if self.shade_k < 1:
                scol = mixc(bc, scol, self.shade_k)
            ctx.set_source_rgba(scol[0], scol[1], scol[2], a)
            ctx.fill()
            ctx.save()
            ctx.append_path(path)
            ctx.clip()
            ctx.translate(self.lv[0] * d, self.lv[1] * d)
            ctx.append_path(path)
            ctx.set_source_rgba(bc[0], bc[1], bc[2], a)
            ctx.fill()
            ctx.restore()
            if hi and self.lod >= 2:
                # soft lit band on the edge facing the key light (single-subpath shapes only)
                hc = self.C(light_of(self.pal.get(base, base) if isinstance(base, str) else base, 0.2))
                w = d * (hi if isinstance(hi, float) else 0.45)
                ctx.save()
                ctx.append_path(path)
                ctx.clip()
                ctx.new_path()
                ctx.rectangle(-BIG, -BIG, 2 * BIG, 2 * BIG)
                ctx.translate(-self.lv[0] * w, -self.lv[1] * w)
                ctx.append_path(path)
                ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
                ctx.set_source_rgba(hc[0], hc[1], hc[2], a * 0.55)
                ctx.fill()
                ctx.restore()
        else:
            ctx.set_source_rgba(bc[0], bc[1], bc[2], a)
            ctx.fill()
        self._rim(path, d, rim, a)
        return path

    def _rim(self, path, d, k, a):
        if self.rim is None or k <= 0:
            return
        rc, rs, rd = self.rim
        w = max(0.9, d * 0.8)
        ctx = self.ctx
        ctx.save()
        ctx.append_path(path)
        ctx.clip()
        ctx.push_group()
        ctx.append_path(path)
        ctx.set_source_rgba(rc[0], rc[1], rc[2], 1.0)
        ctx.fill()
        ctx.translate(-rd[0] * w, -rd[1] * w)
        ctx.append_path(path)
        ctx.set_operator(cairo.OPERATOR_DEST_OUT)
        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.fill()
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(min(1.0, rs * k * a))
        ctx.restore()

    def flat(self, build, c, a=1.0):
        ctx = self.ctx
        ctx.new_path()
        build(ctx)
        self.src(c, a)
        ctx.fill()

    def stroke(self, build, c, w, a=1.0):
        ctx = self.ctx
        ctx.new_path()
        build(ctx)
        self.src(c, a)
        ctx.set_line_width(w)
        ctx.stroke()

    # ================================================================ body pieces
    def ring_pts(self, center, w, d):
        """silhouette half-width of an elliptic body ring under yaw"""
        c, s = self.rig.cphi, self.rig.sphi
        return math.sqrt((w * c) ** 2 + (d * s) ** 2)

    def pj(self, p):
        c, s = self.rig.cphi, self.rig.sphi
        return (p[0] * s - p[2] * c, -p[1])

    def zof(self, p):
        return p[0] * self.rig.cphi + p[2] * self.rig.sphi

    def surf(self, center, w, d, theta, du=0.0):
        """point on a body ring surface; theta 0 = front, +pi/2 = near side"""
        return (center[0] + d * math.cos(theta), center[1] + du, center[2] + w * math.sin(theta))

    def loft(self, rings, bottom_sag=0.0, top_round=0.0):
        """rings: list of (center3d, w, d, dir2d-perp or None) top->bottom. returns closed point list"""
        L, R = [], []
        for (cen, w, d, perp) in rings:
            x, y = self.pj(cen)
            X = self.ring_pts(cen, w, d)
            px, py = perp if perp else (1.0, 0.0)
            L.append((x - px * X, y - py * X))
            R.append((x + px * X, y + py * X))
        pts = L + R[::-1]
        return pts

    def torso_frame(self):
        r = self.rig
        P = r.P
        hx, hy = P["hip"]
        nx, ny = P["neck"]
        dx, dy = nx - hx, ny - hy
        ln = math.hypot(dx, dy) or 1.0
        perp = (-dy / ln, dx / ln)
        if perp[0] < 0:
            perp = (-perp[0], -perp[1])
        return perp

    # ---------------------------------------------------------------- arms
    def arm(self, near, sleeve, part="all"):
        r = self.rig
        sp = self.sp
        P = r.P
        tag = "n" if near else "f"
        S, E, Hh = P["sh_" + tag], P["el_" + tag], P["hand_" + tag]
        # forearm direction
        dx, dy = Hh[0] - E[0], Hh[1] - E[1]
        ln = math.hypot(dx, dy) or 1.0
        ux, uy = dx / ln, dy / ln
        hs = sp["hand"]
        W = (Hh[0] - ux * hs * 0.62, Hh[1] - uy * hs * 0.62)
        ar = sp["arm_r"]
        kind = sleeve
        wind = self.o["wind_x"] * 0.8
        cloth = self.pal["sleeve"]
        if kind == "bell":
            rs = (ar * 0.82, ar * 0.86, ar * 1.38)
        elif kind == "fit":
            rs = (ar * 0.86, ar * 0.8, ar * 0.72)
        elif kind == "bare":
            rs = (ar * 0.78, ar * 0.66, ar * 0.56)
            cloth = "skin"
        else:
            rs = (ar, ar * 0.9, ar * 1.15)
        if part in ("all", "upper"):
            self.part(lambda c: p_capsule(c, S, rs[0], E, rs[1]), cloth, d=ar * 0.5)
        if part in ("all", "fore"):
            if part == "fore":
                self.part(lambda c: p_circle(c, E[0], E[1], rs[1]), cloth, d=ar * 0.5)
            if kind == "bell":
                # sleeve bell flutters with wind / motion
                aw = abs(wind)
                fl = 0.4 * noise1(self.t * 2.3 + (1 if near else 4), 3) + aw * 0.9 * math.sin(self.t * (9 + 3 * aw) + (0 if near else 2))
                W2 = (W[0] + wind * 2.6 + fl * 0.6, W[1] - aw * 0.9 + fl * 0.3)
                rb = rs[2] * (1 + 0.22 * aw)
                rs = (rs[0], rs[1], rb)
                self.part(lambda c: (p_capsule(c, E, rs[1], W2, rs[2])), cloth, d=ar * 0.55)
                # cuff opening
                if self.lod >= 1:
                    vx_, vy_ = W2[0] - E[0], W2[1] - E[1]
                    vl = math.hypot(vx_, vy_) or 1.0
                    ang = math.atan2(vy_ / vl, vx_ / vl)
                    self.part(lambda c: p_ellipse(c, W2[0] + vx_ / vl * rs[2] * 0.3, W2[1] + vy_ / vl * rs[2] * 0.3,
                                                  rs[2] * 0.26, rs[2] * 0.66, ang), "cuffin", d=0)
            else:
                self.part(lambda c: p_capsule(c, E, rs[1], W, rs[2]), cloth, d=ar * 0.5)
                if kind == "fit" and self.lod >= 2 and self.pal.get("cuff"):
                    ang = math.atan2(uy, ux)
                    self.part(lambda c: p_ellipse(c, W[0], W[1], rs[2] * 0.45, rs[2] * 1.02, ang), "cuff", d=0.4)
            self.hand(Hh, (ux, uy), r.shape_f if near else r.shape_b, near)

    def hand(self, Hh, u, shape, near):
        sp = self.sp
        hs = sp["hand"]
        ux, uy = u
        ang = math.atan2(uy, ux)
        skin = "hand"
        # thumb side: perpendicular that points "forward" (+x local) in the hanging pose
        nx, ny = -uy, ux
        if nx < 0:
            nx, ny = -nx, -ny
        if shape == "grip" and self.rig.pole is not None:
            pdx, pdy = self.rig.pole[3]
            pl = math.hypot(pdx, pdy) or 1.0
            pa = math.atan2(pdy / pl, pdx / pl)
            x, y = Hh

            def b(c):
                m = c.get_matrix()
                c.translate(x, y)
                c.rotate(pa)
                rr = hs * 0.62
                c.new_sub_path()
                c.arc(-hs * 0.55 + rr, -hs * 0.72 + rr, rr, math.pi, 1.5 * math.pi)
                c.arc(hs * 0.55 - rr, -hs * 0.72 + rr, rr, 1.5 * math.pi, 0)
                c.arc(hs * 0.55 - rr, hs * 0.72 - rr, rr, 0, 0.5 * math.pi)
                c.arc(-hs * 0.55 + rr, hs * 0.72 - rr, rr, 0.5 * math.pi, math.pi)
                c.close_path()
                c.set_matrix(m)
            self.part(b, skin, d=hs * 0.35)
            if self.lod >= 2:
                # knuckle creases across the pole
                ca, sa = math.cos(pa), math.sin(pa)
                for k in (-0.22, 0.22):
                    self.stroke(lambda c: (c.move_to(x + ca * hs * k - sa * hs * 0.1, y + sa * hs * k + ca * hs * 0.1),
                                           c.line_to(x + ca * hs * k - sa * hs * 0.62, y + sa * hs * k + ca * hs * 0.62)),
                                shadow_of(self.pal["hand"], 0.8), hs * 0.09, 0.8)
            return
        x, y = Hh
        if shape in ("fist", "pinch", "point", "finger", "grip"):
            rr = hs * 0.8
            self.part(lambda c: p_ellipse(c, x, y, rr * 1.02, rr * 0.9, ang), skin, d=hs * 0.35)
            if shape == "point":
                fx, fy = x + ux * hs * 1.5, y + uy * hs * 1.5
                self.part(lambda c: p_capsule(c, (x + ux * hs * 0.3, y + uy * hs * 0.3), hs * 0.3, (fx, fy), hs * 0.24),
                          skin, d=hs * 0.2)
            elif shape == "finger":
                self.part(lambda c: p_capsule(c, (x, y - hs * 0.2), hs * 0.3, (x + hs * 0.1, y - hs * 1.7), hs * 0.24),
                          skin, d=hs * 0.2)
            # thumb
            tx, ty = x + nx * rr * 0.55 + ux * rr * 0.1, y + ny * rr * 0.55 + uy * rr * 0.1
            self.part(lambda c: p_ellipse(c, tx, ty, rr * 0.5, rr * 0.32, ang + 0.4), skin, d=hs * 0.15)
            return
        if shape == "open":
            self.part(lambda c: p_ellipse(c, x + ux * hs * 0.15, y + uy * hs * 0.15, hs * 1.08, hs * 0.86, ang),
                      skin, d=hs * 0.35)
            tx, ty = x + nx * hs * 0.75 - ux * hs * 0.05, y + ny * hs * 0.75 - uy * hs * 0.05
            self.part(lambda c: p_ellipse(c, tx, ty, hs * 0.55, hs * 0.3, ang + 0.9 * (1 if ny * ux - nx * uy > 0 else -1)),
                      skin, d=hs * 0.15)
            if self.lod >= 2:
                for k in (-0.28, 0.12):
                    px, py = x + ux * hs * 0.55 + nx * hs * k, y + uy * hs * 0.55 + ny * hs * k
                    self.stroke(lambda c: (c.move_to(px, py), c.line_to(px + ux * hs * 0.55, py + uy * hs * 0.55)),
                                shadow_of(self.pal["hand"], 0.8), hs * 0.08, 0.7)
            return
        if shape == "cup":
            self.part(lambda c: p_ellipse(c, x, y, hs * 0.72, hs * 0.95, ang), skin, d=hs * 0.3)
            self.flat(lambda c: p_ellipse(c, x + ux * hs * 0.28, y + uy * hs * 0.28, hs * 0.3, hs * 0.62, ang),
                      shadow_of(self.pal["hand"], 0.72), 0.85)
            return
        # relax: mitten
        self.part(lambda c: p_ellipse(c, x + ux * hs * 0.1, y + uy * hs * 0.1, hs * 1.0, hs * 0.76, ang), skin, d=hs * 0.35)
        tx, ty = x + nx * hs * 0.6 - ux * hs * 0.12, y + ny * hs * 0.6 - uy * hs * 0.12
        self.part(lambda c: p_ellipse(c, tx, ty, hs * 0.48, hs * 0.3, ang + 0.5 * (1 if ny * ux - nx * uy > 0 else -1)),
                  skin, d=hs * 0.15)

    # ---------------------------------------------------------------- legs & feet
    def leg(self, near, trouser, shoe, bare_below=None, pad=None):
        r = self.rig
        sp = self.sp
        P = r.P
        tag = "n" if near else "f"
        Hh, K, A = P["hip_" + tag], P["knee_" + tag], P["ank_" + tag]
        lr = sp["leg_r"]
        if r.sit > 0.5:
            return
        if trouser is not None:
            self.part(lambda c: p_chain(c, [Hh, K, A], [lr * 1.1, lr * 0.85, lr * 0.62]), trouser, d=lr * 0.45)
        if bare_below is not None:
            self.part(lambda c: p_capsule(c, K, lr * 0.62, A, lr * 0.45), bare_below, d=lr * 0.35)
        if pad is not None:
            kx, ky = K
            ux, uy = A[0] - K[0], A[1] - K[1]
            ln = math.hypot(ux, uy) or 1.0
            ux, uy = ux / ln, uy / ln
            self.part(lambda c: p_capsule(c, (kx - ux * lr * 0.15, ky - uy * lr * 0.15), lr * 0.78,
                                          (kx + ux * lr * 0.75, ky + uy * lr * 0.75), lr * 0.62), pad, d=lr * 0.35)
            if self.lod >= 2:
                self.stroke(lambda c: (c.move_to(kx + ux * lr * 0.2 - uy * lr * 0.6, ky + uy * lr * 0.2 + ux * lr * 0.6),
                                       c.line_to(kx + ux * lr * 0.2 + uy * lr * 0.6, ky + uy * lr * 0.2 - ux * lr * 0.6)),
                            light_of(self.pal[pad] if isinstance(pad, str) else pad, 0.15), lr * 0.14, 0.8)
        self.foot(near, shoe)

    def foot(self, near, shoe):
        r = self.rig
        sp = self.sp
        tag = "n" if near else "f"
        A = r.P["ank_" + tag]
        f = r.feet[0] if near else r.feet[1]
        toe = f[2]
        L = sp["foot"]
        c, s = r.cphi, r.sphi
        # forward direction on screen (pseudo-perspective: toward camera goes down)
        fx, fy = s, c * 0.22
        ca, sa = math.cos(-toe), math.sin(-toe)
        fx, fy = fx * ca - fy * sa, fx * sa + fy * ca
        heel = (A[0] - fx * L * 0.22, A[1] + sp["ank"] * 0.55 - fy * L * 0.22)
        tip = (A[0] + fx * L * 0.72, A[1] + sp["ank"] * 0.75 + fy * L * 0.72)
        rr = sp["ank"] * 1.05 + 0.4
        wide = rr * (1.0 + 0.5 * c)
        self.part(lambda cc: (p_capsule(cc, heel, wide * 0.95, tip, wide * 0.85)), shoe, d=rr * 0.5)
        if self.lod >= 2:
            sole = shadow_of(self.pal[shoe] if isinstance(shoe, str) else shoe, 0.55)
            self.stroke(lambda cc: (cc.move_to(heel[0] - wide * 0.3 * fx, heel[1] + wide * 0.82),
                                    cc.line_to(tip[0] + wide * 0.4 * fx, tip[1] + wide * 0.72)), sole, wide * 0.35)

    # ---------------------------------------------------------------- robe / garments
    def robe(self, hem_y=None, hem_w=None, top="robe", skirt=True, waist_only=False, flare=1.0, colour="robe"):
        """long garment loft from shoulders to hem. returns point list (for clipping details)"""
        r = self.rig
        sp = self.sp
        B = r.B
        perp = self.torso_frame()
        N, Hh = B["neck"], B["hip"]
        lean = r.lean
        dep = sp["dep"]
        up = (math.sin(lean), math.cos(lean), 0.0)

        def along(t):
            return (Hh[0] + (N[0] - Hh[0]) * t, Hh[1] + (N[1] - Hh[1]) * t, 0.0)

        rings = []
        shc = (N[0] - up[0] * 1.5, N[1] - up[1] * 1.5, 0.0)
        rings.append(((N[0] + up[0] * 1.6, N[1] + up[1] * 1.6, 0.0), sp["head_r"] * 0.4, dep * 0.5, perp))
        rings.append((shc, sp["sh_w"] + sp["arm_r"] * 0.55, dep * 0.95, perp))
        rings.append((along(0.62), sp["chest_w"], dep * 1.05, perp))
        rings.append((along(0.18), sp["waist_w"], dep * 1.0, perp))
        hem_u = -(hem_y if hem_y is not None else sp["hem_y"]) + r.hem_lift
        hw = hem_w if hem_w is not None else sp["hem_w"]
        if skirt:
            sway = r.sway + 0.7 * noise1(self.t * 0.9 + self.ch.seed, 21)
            ff, fb = r.feet
            fa = max(ff[0], fb[0])
            ba = min(ff[0], fb[0])
            # hem follows the stride (push the robe with the legs)
            hem_c = (0.5 * (fa + ba) * 0.6 + sway * 0.4, hem_u, 0.0)
            ext = max(0.0, (fa - ba) * 0.5 - hw * 0.5)
            ku = hem_u + (Hh[1] - hem_u) * 0.45
            rings.append(((Hh[0] * 0.5 + sway * 0.15, ku, 0.0), (sp["waist_w"] + hw) * 0.5 * flare + ext * 0.3,
                          dep * 1.05 + ext * 0.3, None))
            rings.append((hem_c, hw * flare + ext * 0.8, hw * 0.72 + ext * 0.6, None))
        pts = self.loft(rings)
        # bottom hem curve: add sag points between hem corners
        if skirt:
            n = len(rings)
            Lh, Rh = pts[n - 1], pts[n]
            cx, cy = 0.5 * (Lh[0] + Rh[0]), 0.5 * (Lh[1] + Rh[1])
            hw2 = 0.5 * (Rh[0] - Lh[0])
            sag = hw * 0.16 * r.cphi + 0.6
            wav = [0.5 * noise1(self.t * 1.3 + k * 1.7 + self.ch.seed, 23) for k in range(3)]
            mid = [(cx - hw2 * 0.55, cy + sag * 0.8 + wav[0]), (cx, cy + sag + wav[1]), (cx + hw2 * 0.55, cy + sag * 0.8 + wav[2])]
            pts = pts[:n] + mid + pts[n:]
            pts = self.wind_warp(pts, -Hh[1], -hem_u)
        return pts

    def wind_warp(self, pts, y0, y1, amt=1.0):
        """billow + flap cloth points below y0 (local screen y) downwind; strength grows toward y1"""
        w = self.o["wind_x"] * amt
        if abs(w) < 1e-3:
            return pts
        t = self.t
        aw = abs(w)
        sg = 1.0 if w > 0 else -1.0
        span = max(y1 - y0, 1e-3)
        out = []
        for (x, y) in pts:
            f = (y - y0) / span
            if f <= 0:
                out.append((x, y))
                continue
            f = min(f, 1.2)
            ph = t * (7.0 + 3.0 * aw) - x * 0.35 * sg + y * 0.08
            flap = math.sin(ph) * 0.6 + 0.4 * math.sin(ph * 1.7 + 1.3)
            down = 1.0 if x * sg > 0 else 0.35  # downwind edge billows, upwind edge presses to the legs
            dx = w * f * (3.2 + 2.2 * down) + sg * aw * f * f * 1.6 * flap * down
            dy = -aw * f * f * down * (1.8 + 1.4 * (0.5 + 0.5 * flap))
            out.append((x + dx, y + dy))
        return out

    def fill_pts(self, pts, colour, d=3.0, smooth_k=0.33):
        return self.part(lambda c: p_smooth(c, pts, True, smooth_k), colour, d=d, hi=0.5)

    def clip_path(self, path):
        self.ctx.save()
        self.ctx.new_path()
        self.ctx.append_path(path)
        self.ctx.clip()

    def unclip(self):
        self.ctx.restore()

    def folds(self, top_t=0.2, n=3, colour="robe", alpha=0.45, hem_u=None):
        """soft vertical fold shadows from waist to hem on the visible front of the skirt"""
        if self.lod < 2:
            return
        r = self.rig
        sp = self.sp
        B = r.B
        Hh = B["hip"]
        hem_u = -sp["hem_y"] if hem_u is None else hem_u
        sway = r.sway + self.o["wind_x"] * 3.2
        for k in range(n):
            th = (k - (n - 1) / 2) * 0.62 + 0.25 * noise1(self.t * 0.6 + k * 3.1, 31)
            top = self.surf((Hh[0], Hh[1] - 3, 0.0), sp["waist_w"], sp["dep"], th)
            bot = self.surf((sway * 0.4, hem_u + 1.5, 0.0), sp["hem_w"] * 0.95, sp["hem_w"] * 0.7, th)
            if self.zof(top) < -1 or self.zof(bot) < 0:
                continue
            a = self.pj(top)
            b = self.pj(bot)
            w = 1.1 + 0.8 * (k % 2)
            self.flat(lambda c: (c.move_to(a[0] - 0.3, a[1]),
                                 c.curve_to(a[0] + (b[0] - a[0]) * 0.3 - w, a[1] + (b[1] - a[1]) * 0.5,
                                            b[0] - w * 1.4, b[1] - 6, b[0] - w, b[1]),
                                 c.line_to(b[0] + w, b[1]),
                                 c.curve_to(b[0] + w * 0.3, b[1] - 6, a[0] + (b[0] - a[0]) * 0.3 + w * 0.2,
                                            a[1] + (b[1] - a[1]) * 0.5, a[0] + 0.3, a[1]),
                                 c.close_path()), shadow_of(self.pal[colour], 0.78), alpha)

    def belt(self, colour="belt", knot=True, width=2.2, u_off=1.5):
        r = self.rig
        sp = self.sp
        Hh = r.B["hip"]
        cen = (Hh[0] + math.sin(r.lean) * u_off, Hh[1] + u_off, 0.0)
        pts = []
        for i in range(17):
            th = -math.pi + TAU * i / 16
            p = self.surf(cen, sp["waist_w"] * 1.02, sp["dep"] * 1.05, th)
            if self.zof(p) > -0.5:
                pts.append(self.pj(p))
        if len(pts) >= 2:
            self.stroke(lambda c: (c.move_to(*pts[0]), [c.line_to(*q) for q in pts[1:]]), colour, width)
        if knot and self.lod >= 1:
            kp = self.surf(cen, sp["waist_w"], sp["dep"] * 1.05, 0.55)
            if self.zof(kp) > 0:
                x, y = self.pj(kp)
                self.part(lambda c: p_circle(c, x, y, width * 0.95), colour, d=0.6)
                sw = self.o["wind_x"] * 2.5 + 0.8 * noise1(self.t * 1.5 + 2, 41)
                for k, L in ((0.0, 9.0), (1.4, 7.0)):
                    self.stroke(lambda c: (c.move_to(x, y), c.curve_to(x + sw * 0.3 + k, y + L * 0.4, x + sw * 0.8 + k,
                                                                       y + L * 0.7, x + sw + k * 1.2, y + L)),
                                colour, width * 0.7)

    def pouch(self):
        r = self.rig
        sp = self.sp
        Hh = r.B["hip"]
        p = self.surf((Hh[0], Hh[1] - 2.5, 0.0), sp["waist_w"] * 1.05, sp["dep"] * 1.1, 1.2)
        if self.zof(p) < -2:
            return
        x, y = self.pj(p)
        sw = 0.6 * math.sin(self.t * 2.1) + self.o["wind_x"]
        w, hh = 3.4, 4.2
        if self.lod >= 1:
            # sprig of herb peeking out (green, never yellow)
            for k, (dx, L, a) in enumerate(((-0.8, 4.5, -0.5), (0.6, 5.5, -0.15), (1.5, 3.8, 0.35))):
                bx, by = x + dx + sw * 0.2, y - hh * 0.4
                self.part(lambda c: p_leaf(c, bx, by, L, 1.1, -math.pi / 2 + a + sw * 0.08), "herb", d=0.5, rim=0.5)
        self.part(lambda c: (c.move_to(x - w, y - hh * 0.5),
                             c.curve_to(x - w * 1.1, y + hh * 0.9, x + w * 1.1, y + hh * 0.9, x + w, y - hh * 0.5),
                             c.close_path()), "leather", d=1.0)
        self.part(lambda c: (c.move_to(x - w * 1.05, y - hh * 0.55), c.line_to(x + w * 1.05, y - hh * 0.55),
                             c.curve_to(x + w * 0.6, y + hh * 0.15, x - w * 0.6, y + hh * 0.15, x - w * 1.05, y - hh * 0.55),
                             c.close_path()), shadow_of(self.pal["leather"], 0.8), d=0)

    def neck(self):
        r = self.rig
        sp = self.sp
        N = r.P["neck"]
        Hc = r.P["head"]
        nw = sp["head_r"] * 0.34
        self.part(lambda c: p_capsule(c, (N[0], N[1] + 1.5), nw, (Hc[0] * 0.3 + N[0] * 0.7, Hc[1] * 0.4 + N[1] * 0.6), nw),
                  "skin", d=1.0)

    def collar_hood_down(self, colour="robe"):
        """hood lying folded behind the neck on the shoulders"""
        r = self.rig
        sp = self.sp
        N = r.B["neck"]
        pts = []
        for i in range(13):
            th = math.pi * 0.5 + math.pi * i / 12   # near side -> back -> far side
            p = self.surf((N[0] - 1.0, N[1] - 1.2, 0.0), sp["head_r"] * 0.95, sp["head_r"] * 0.85, th)
            pts.append(self.pj(p))
        inner = []
        for i in range(13):
            th = math.pi * 1.5 - math.pi * i / 12
            p = self.surf((N[0] - 0.3, N[1] + 0.8, 0.0), sp["head_r"] * 0.5, sp["head_r"] * 0.45, th)
            inner.append(self.pj(p))
        pts = pts + inner
        self.part(lambda c: p_smooth(c, pts, True, 0.3), shadow_of(self.pal[colour], 0.86), d=1.5)

    # ================================================================ heads
    def head_frame(self):
        r = self.rig
        x, y = r.P["head"]
        ctx = self.ctx
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(r.head_rot)

    def head_base(self, hp, jaw=None, nose=True, colour="skin"):
        sp = self.sp
        R = sp["head_r"]
        jx, jy = jaw or sp["jaw"]
        jy = jy + 0.12 * self.rig.mouth * (0.5 + getattr(self.rig, "yell", 0.0))
        yaw = hp.yaw
        cx = R * 0.36 * math.sin(yaw) * math.cos(max(0, yaw - math.pi / 2))
        cz = math.cos(yaw)

        def b(c):
            p_circle(c, 0, 0, R)
            if cz > -0.6:
                p_ellipse(c, cx * (1 if yaw <= math.pi / 2 else max(0, 1 - (yaw - math.pi / 2))), R * 0.32,
                          R * jx * (0.78 + 0.22 * abs(cz)), R * jy)
            chin = hp.pt(0.0, -0.86, 0.97)
            if chin[2] > 0.0 and 0.3 < yaw < 1.45:
                p_circle(c, chin[0], chin[1], R * 0.24)
            if nose and self.lod >= 1:
                tip = hp.pt(0, -0.14, 1.0 + 0.1 * sp["nose"])
                if tip[2] > 0.05:
                    p_circle(c, tip[0], tip[1], R * 0.1 * (0.6 + 0.5 * sp["nose"]))
        return self.part(b, colour, d=R * 0.2)

    def ear(self, hp, near, colour="skin", size=1.0):
        R = self.sp["head_r"]
        lon = -math.pi / 2 if near else math.pi / 2
        x, y, z = hp.pt(lon, -0.02, 0.98)
        if z < -0.2:
            return
        ang = 0.15 * (1 if x < 0 else -1)
        self.part(lambda c: p_ellipse(c, x, y, R * 0.17 * size * (0.55 + 0.45 * abs(z)), R * 0.25 * size, ang), colour,
                  d=R * 0.08)
        if self.lod >= 2 and z > 0.2:
            self.flat(lambda c: p_ellipse(c, x, y, R * 0.07 * size, R * 0.13 * size, ang),
                      shadow_of(self.pal[colour], 0.8), 0.8)

    def face(self, hp, E, blink, look, mouth, eyes_style="std", lash="lash", brow="brow", mouth_y=-0.5,
             brow_w=0.075, brow_len=1.0, show_mouth=True, blush=True, pupil_c="iris", brow_in=0.35):
        sp = self.sp
        R = sp["head_r"]
        lod = self.lod
        yaw = hp.yaw
        lxl, lyl = look
        pitch_f = 0.0
        # ------------------------------------------------ eyes
        wink_far = E["wn"] > 0.5
        for near in (True, False):
            lon = -sp["eye_sep"] if near else sp["eye_sep"]
            x, y, z = hp.pt(lon, sp["eye_y"] + pitch_f, 1.0)
            if z < 0.12:
                continue
            fore = z ** 0.55
            ew = sp["eye_w"] * R * E["es"] * fore
            eh = sp["eye_h"] * R * E["es"]
            close = clamp(sp["lid0"] + E["lt"] + blink)
            if not near and wink_far or (near and self.o.get("wink") == "near") or (not near and self.o.get("wink") == "far"):
                close = 1.0
            lower = clamp(E["lb"] * 0.55 + E["sq"] * 0.35)
            if lod == 0:
                continue
            if lod == 1:
                if close > 0.8:
                    self.stroke(lambda c: (c.move_to(x - ew, y), c.line_to(x + ew, y)), lash, eh * 0.35)
                else:
                    self.flat(lambda c: p_ellipse(c, x + lxl * ew * 0.25, y + lyl * eh * 0.2, ew * 0.62,
                                                  eh * 0.75 * (1 - close * 0.8)), lash)
                continue
            if close > 0.9:
                # closed: happy arc (smile) or relaxed down-arc
                up = E["sm"] > 0.5 and blink < 0.5
                yy = y + eh * (0.25 if not up else 0.05)
                self.stroke(lambda c: (c.move_to(x - ew * 1.05, yy),
                                       c.curve_to(x - ew * 0.4, yy + (-eh * 0.5 if up else eh * 0.45),
                                                  x + ew * 0.4, yy + (-eh * 0.5 if up else eh * 0.45), x + ew * 1.05,
                                                  yy - eh * 0.05)),
                            lash, R * 0.055)
                continue
            # sclera
            ctx = self.ctx
            ctx.save()
            ctx.new_path()
            p_ellipse(ctx, x, y, ew, eh)
            self.src("sclera")
            ctx.fill_preserve()
            ctx.clip()
            # iris + pupil follow look
            ir = min(ew / max(fore, 0.3), eh) * 0.68
            ix = x + (lxl * 0.5 + 0.18 * math.sin(yaw) * 0) * ew * 0.9
            iy = y + lyl * eh * 0.35 + eh * 0.04
            self.src(pupil_c)
            p_ellipse(ctx, ix, iy, ir * fore ** 0.3 * 0.92, ir)
            ctx.fill()
            self.src("pupil")
            pr = ir * 0.55 * E["ps"]
            p_ellipse(ctx, ix, iy, pr * fore ** 0.3, pr)
            ctx.fill()
            self.raw("sclera", 0.95)
            p_circle(ctx, ix - ir * 0.32, iy - ir * 0.38, ir * 0.24)
            ctx.fill()
            # upper lid
            ly = y - eh + 2 * eh * close
            lidc = self.pal.get("lid", shadow_of(self.pal["skin"], 0.9))
            self.src(lidc)
            ctx.move_to(x - ew * 1.3, y - eh * 1.3)
            ctx.line_to(x + ew * 1.3, y - eh * 1.3)
            ctx.line_to(x + ew * 1.3, ly)
            ctx.curve_to(x + ew * 0.4, ly + eh * 0.25 * (1 - close), x - ew * 0.4, ly + eh * 0.25 * (1 - close),
                         x - ew * 1.3, ly)
            ctx.close_path()
            ctx.fill()
            if lower > 0.01:
                by = y + eh - 2 * eh * lower
                self.src(self.pal["skin"])
                ctx.move_to(x - ew * 1.3, y + eh * 1.3)
                ctx.line_to(x + ew * 1.3, y + eh * 1.3)
                ctx.line_to(x + ew * 1.3, by)
                ctx.curve_to(x + ew * 0.4, by - eh * 0.3, x - ew * 0.4, by - eh * 0.3, x - ew * 1.3, by)
                ctx.close_path()
                ctx.fill()
            ctx.restore()
            # lash line along the upper lid
            ly = y - eh + 2 * eh * close
            lw = R * (0.06 if eyes_style == "big" else 0.045)
            flick = 1.0 if eyes_style == "big" else 0.0
            outer = -1 if near else 1
            self.stroke(lambda c: (c.move_to(x - ew * 1.02, ly + eh * 0.1 * (1 - close)),
                                   c.curve_to(x - ew * 0.4, ly - eh * 0.12 * (1 - close) + eh * 0.25 * (1 - close) * 0.5,
                                              x + ew * 0.4, ly - eh * 0.12 * (1 - close) + eh * 0.25 * (1 - close) * 0.5,
                                              x + ew * 1.02, ly + eh * 0.1 * (1 - close)),
                                   c.line_to(x + outer * ew * (1.02 + 0.25 * flick) if outer > 0 else x + ew * 1.02,
                                             ly - eh * 0.12 * flick if outer > 0 else ly)),
                        lash, lw)
            if flick and outer < 0:
                self.stroke(lambda c: (c.move_to(x - ew * 1.02, ly + eh * 0.1 * (1 - close)),
                                       c.line_to(x - ew * 1.25, ly - eh * 0.1)), lash, lw * 0.9)
        # ------------------------------------------------ brows
        if lod >= 1:
            for near in (True, False):
                lon = -sp["eye_sep"] if near else sp["eye_sep"]
                raise_ = E["by"] + (E["ba"] if not near else -E["ba"] * 0.3)
                lat = sp["eye_y"] + 0.36 + raise_ * 0.12 - E["sq"] * 0.05
                inner_lon = lon * brow_in
                outer_lon = lon * 1.45 * brow_len
                bi = E["bi"]
                p_in = hp.pt(inner_lon, lat + bi * 0.1 - (0.02 if bi < 0 else 0), 1.02)
                p_mid = hp.pt(lon * 0.95, lat + 0.06 + raise_ * 0.03, 1.02)
                p_out = hp.pt(outer_lon, lat - 0.03 - bi * 0.03, 1.0)
                if p_mid[2] < 0.15:
                    continue
                bw = R * brow_w * (1.0 if lod >= 2 else 1.3)
                self.stroke(lambda c: (c.move_to(p_in[0], p_in[1]), c.curve_to(p_mid[0], p_mid[1], p_mid[0], p_mid[1],
                                                                                p_out[0], p_out[1])),
                            brow, bw)
        # ------------------------------------------------ nose
        if lod >= 2:
            tip = hp.pt(0, -0.14, 1.0 + 0.1 * sp["nose"])
            if tip[2] > 0.1:
                s = math.sin(yaw)
                nr = R * 0.07 * (0.6 + 0.5 * sp["nose"])
                self.flat(lambda c: p_ellipse(c, tip[0] - s * nr * 0.5, tip[1] + nr * 0.8, nr * 1.1, nr * 0.55),
                          shadow_of(self.pal["skin"], 0.78), 0.85)
        # ------------------------------------------------ blush
        if blush and lod >= 2 and E["bl"] + self.pal.get("blush_base", 0.0) > 0.01:
            a = clamp(E["bl"] * 0.5 + self.pal.get("blush_base", 0.0))
            for near in (True, False):
                lon = -sp["eye_sep"] * 1.15 if near else sp["eye_sep"] * 1.15
                x, y, z = hp.pt(lon, -0.28, 1.0)
                if z > 0.2:
                    self.flat(lambda c: p_ellipse(c, x, y, R * 0.16 * z ** 0.6, R * 0.09), "blush", a * 0.55)
        # ------------------------------------------------ mouth
        if show_mouth and lod >= 1:
            self.mouth(hp, E, mouth, mouth_y)

    def mouth(self, hp, E, m, lat=-0.5, lip="lip", scale=1.0):
        sp = self.sp
        R = sp["head_r"] * scale
        x, y, z = hp.pt(0.0, lat, 1.0)
        if z < 0.1:
            return
        o = clamp(m + E["mo"])
        vw = 1.0 + 0.16 * noise1(self.t * 8.0 + 3.3, 51) * clamp(m * 3)
        smile = E["sm"]
        sk = E["sk"]
        yell = getattr(self.rig, "yell", 0.0) * clamp(o * 1.6)
        wl = 0.3 * E["mw"] * (1 + 0.18 * smile) * vw * (1 + 0.12 * o) * (1 + 0.35 * yell)
        c0 = sk * 0.06
        lat_l = lat + smile * 0.09 + sk * 0.05
        lat_r = lat + smile * 0.09 - sk * 0.05
        L = hp.pt(c0 - wl, lat_l, 1.0)
        Rr = hp.pt(c0 + wl, lat_r, 1.0)
        Cc = hp.pt(c0, lat, 1.0)

        def vis(p):
            if p[2] >= 0.05:
                return p[0], p[1]
            # corner rolled behind the cheek: pin it to the silhouette
            ln = math.hypot(p[0], p[1]) or 1.0
            return p[0] / ln * R * 0.97, p[1]
        (xl, cyl), (xr, cyr) = vis(L), vis(Rr)
        x = Cc[0]
        W = 0.5 * (xr - xl)
        xm = 0.5 * (xl + xr)
        Hm = R * 0.36 * o / vw * (1 + 0.9 * yell)
        if self.lod == 1 or o < 0.06:
            mid = y + smile * R * 0.07 + Hm * 0.5
            self.stroke(lambda c: (c.move_to(xl, cyl), c.curve_to(xl + (x - xl) * 0.6, mid, xr + (x - xr) * 0.6, mid,
                                                                   xr, cyr)),
                        lip, R * 0.05 + Hm * 0.8)
            return
        top_mid = y + smile * R * 0.02 - Hm * 0.12
        bot_mid = y + Hm + smile * R * 0.04

        def b(c):
            c.move_to(xl, cyl)
            c.curve_to(xl + (x - xl) * 0.5, top_mid, xr + (x - xr) * 0.5, top_mid, xr, cyr)
            c.curve_to(xr + (x - xr) * 0.2, bot_mid, xl + (x - xl) * 0.2, bot_mid, xl, cyl)
            c.close_path()
        x = xm
        ctx = self.ctx
        ctx.save()
        ctx.new_path()
        b(ctx)
        self.src("mouth_in")
        ctx.fill_preserve()
        ctx.clip()
        if self.lod >= 2:
            if E["th"] > 0.2 or o > 0.3:
                self.src("teeth")
                ctx.rectangle(x - W, top_mid - R, 2 * W, R + Hm * 0.26 * (0.6 + E["th"] * 0.6))
                ctx.fill()
            if o > 0.25:
                self.src("tongue")
                p_ellipse(ctx, x + W * 0.1, bot_mid + Hm * 0.05, W * 0.6, Hm * 0.42)
                ctx.fill()
        ctx.restore()

    # ---------------------------------------------------------------- hair
    def hair_cap(self, hp, style, seed, colour="hair"):
        sp = self.sp
        R = sp["head_r"]
        ctx = self.ctx
        t = self.t
        wind = self.o["wind_x"]
        if style in ("bald", None, "none"):
            if style == "bald":
                # fringe around the back
                pass
            return
        big = {"messy": 1.08, "short": 1.04, "bob": 1.1, "long": 1.08, "bun": 1.05, "curly": 1.32, "pony": 1.05,
               "wild": 1.22, "buzz": 1.01, "side": 1.05}.get(style, 1.06)
        gam = {"messy": 1.16, "short": 1.2, "bob": 1.1, "long": 1.1, "bun": 1.22, "curly": 1.06, "pony": 1.2,
               "wild": 1.12, "buzz": 1.26, "side": 1.18}.get(style, 1.16)
        bang = {"messy": 0.3, "short": 0.16, "bob": 0.24, "long": 0.12, "bun": 0.05, "curly": 0.12, "pony": 0.06,
                "wild": 0.3, "buzz": 0.0, "side": 0.3}.get(style, 0.12)
        nb = 13 if style != "side" else 5
        ph0 = hash01(seed, 55)

        def jag(s):
            w = max(0.0, math.sin(s)) ** 0.8
            k = (s * nb / math.pi + ph0) % 2.0
            tri = (1 - abs(k - 1)) ** 1.6
            if style == "side":
                w *= 0.5 + 0.5 * math.cos(s)
            side = 0.2 * math.cos(s) ** 2 * (1 if math.sin(s) > -0.3 else 0.5)
            return bang * tri * w - side

        face = hp.cap(0.0, -0.3, gam, 1.0, 1.45, 120, jag)
        cut = {"short": (-1.4, 1.12), "messy": (-1.4, 1.08), "bun": (-1.4, 1.14), "pony": (-1.4, 1.14),
               "buzz": (-1.35, 1.2), "side": (-1.4, 1.12), "curly": (-1.5, 1.0), "wild": (-1.5, 1.02),
               "bob": (-1.6, 0.82), "long": (-1.7, 0.62)}.get(style, (-1.4, 1.12))
        jawcap = hp.cap(0.0, cut[0], cut[1], 1.0, 1.6, 30)

        def outer(c):
            n = 40
            pts = []
            for i in range(n):
                a = TAU * i / n
                wob = 0.0
                if style in ("messy", "wild", "curly"):
                    wob = 0.035 * math.sin(a * 7 + seed) + 0.025 * math.sin(a * 13 + seed * 2.1)
                    if style == "curly":
                        wob = 0.05 * abs(math.sin(a * 9 + seed))
                rr = R * (big + wob)
                pts.append((math.cos(a) * rr, math.sin(a) * rr))
            if style == "bob":
                pts = [(x, y * (1.0 if y < 0 else 1.18)) for x, y in pts]
            p_smooth(c, pts, True, 0.4)

        ctx.save()
        for ex in (jawcap, face):
            ctx.new_path()
            ctx.rectangle(-BIG, -BIG, 2 * BIG, 2 * BIG)
            p_poly(ctx, ex)
            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            ctx.clip()
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
        hpath = self.part(outer, colour, d=R * 0.18, hi=0.4)
        # sheen
        sx, sy, sz = hp.pt(-0.5, 0.75, 1.0)
        if self.lod >= 2 and self.sil is None and sz > 0.1:
            ctx.new_path()
            ctx.append_path(hpath)
            ctx.clip()
            self.flat(lambda c: p_ellipse(c, sx + R * 0.1, sy, R * 0.4, R * 0.14, -0.4), light_of(self.pal[colour], 0.12),
                      0.55)
        ctx.restore()
        if style == "messy" and self.lod >= 1:
            # a lock in front of the near ear
            b0 = hp.pt(-1.18, 0.42, 1.04)
            b1 = hp.pt(-1.1, -0.28, 1.03)
            if b0[2] > 0.0:
                ang = math.atan2(b1[1] - b0[1], b1[0] - b0[0]) + wind * 0.15
                ln = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
                self.part(lambda c: p_leaf(c, b0[0], b0[1], ln, R * 0.14, ang), colour, d=R * 0.05)
        if style == "messy":
            sw = wind * 0.35 + 0.08 * math.sin(t * 2.4 + seed)
            for k, (lon, L, a0) in enumerate(((-0.55, 0.42, -0.5), (0.05, 0.5, 0.05), (0.6, 0.36, 0.55))):
                bx, by, bz = hp.pt(lon, 1.12, big * 0.97)
                ang = -math.pi / 2 + a0 + sw * (1 + 0.3 * k) + hp.yaw * 0.25
                self.part(lambda c: p_leaf(c, bx, by, R * L, R * 0.12, ang), colour, d=R * 0.06)
        if style == "wild":
            sw = wind * 0.4 + 0.12 * math.sin(t * 2.0 + seed)
            for k in range(6):
                lon = -1.3 + k * 0.5
                bx, by, bz = hp.pt(lon, 0.85, 1.12)
                ang = math.atan2(by, bx) + sw + 0.2 * math.sin(k * 2.3)
                self.part(lambda c: p_leaf(c, bx * 0.9, by * 0.9, R * 0.55, R * 0.16, ang), colour, d=R * 0.06)
        if style == "bun":
            bx, by, bz = hp.pt(math.pi * 0.85, 0.95, 1.15)
            self.part(lambda c: p_circle(c, bx, by, R * 0.38), colour, d=R * 0.1)

    def hair_back(self, hp, style, seed, colour="hair"):
        """long hair falling behind the head (drawn before the head)"""
        R = self.sp["head_r"]
        sw = self.o["wind_x"] * 1.2 + 0.15 * math.sin(self.t * 1.6 + seed)
        if style == "long":
            self.part(lambda c: p_smooth(c, [(-R * 1.05, -R * 0.2), (-R * 1.12 + sw * R * 0.3, R * 1.5),
                                             (R * 0.2 + sw * R * 0.5, R * 1.75), (R * 1.0, R * 0.9), (R * 0.9, -R * 0.4),
                                             (0, -R * 1.05)], True, 0.35), colour, d=R * 0.15)
        elif style == "pony":
            bx, by, bz = hp.pt(math.pi, 0.25, 1.0)
            self.part(lambda c: p_leaf(c, bx, by, R * 1.5, R * 0.3, math.pi / 2 + 0.3 - sw * 0.8 +
                                       (0.6 if bx < 0 else -0.6)), colour, d=R * 0.12)
        elif style == "messy":
            for k, (lon, lat, L) in enumerate(((math.pi - 0.2, -0.35, 0.42), (math.pi + 0.35, -0.3, 0.36),
                                               (math.pi - 0.75, -0.2, 0.34), (math.pi + 0.95, -0.05, 0.3))):
                x, y, z = hp.pt(lon, lat, 1.05)
                if z > 0.25:
                    continue
                ang = math.atan2(y, x) + 0.35 * (1 if x > 0 else -1) * -1 + sw * 0.3
                self.part(lambda c: p_leaf(c, x * 0.92, y * 0.92, R * L, R * 0.13, ang), colour, d=R * 0.06)

"""s04 — tiny 3D engine for Flagartha: perspective camera, near-plane clipping, lit & fogged polygons,
3D polylines and billboards, all drawn with cairo in design units (so --scale works).

World units: centimetres. Y up, floor at Y = 0, the Hypermind hangs on the X = Z = 0 axis.
"""
import math

import cairo
import numpy as np

from vx.config import W, H


class Cam3:
    """pinhole camera. yaw: 0 looks along +Z, + turns toward +X. pitch: + looks up. roll: + turns clockwise.
    f: focal length in design px."""

    def __init__(self, pos, yaw=0.0, pitch=0.0, roll=0.0, f=1000.0, near=4.0, shift=(0.0, 0.0)):
        self.pos = np.asarray(pos, np.float64)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        fwd = np.array([sy * cp, sp, cy * cp])
        right = np.array([cy, 0.0, -sy])
        up = np.cross(fwd, right)
        cr, sr = math.cos(roll), math.sin(roll)
        right, up = right * cr - up * sr, up * cr + right * sr
        self.R = np.stack([right, up, fwd])
        self.right, self.up, self.fwd = right, up, fwd
        self.f, self.near = float(f), near
        self.cx, self.cy = W / 2 + shift[0], H / 2 + shift[1]
        self.yaw, self.pitch = yaw, pitch

    @staticmethod
    def look_at(pos, target, f=1000.0, roll=0.0, **kw):
        d = np.asarray(target, float) - np.asarray(pos, float)
        yaw = math.atan2(d[0], d[2])
        pitch = math.atan2(d[1], math.hypot(d[0], d[2]))
        return Cam3(pos, yaw, pitch, roll, f, **kw)

    def to_cam(self, P):
        return (np.asarray(P, np.float64) - self.pos) @ self.R.T

    def proj(self, P):
        """(...,3) world -> (sx, sy, z) arrays; z = depth along view axis"""
        c = self.to_cam(P)
        z = c[..., 2]
        zs = np.maximum(z, 1e-3)
        return self.cx + self.f * c[..., 0] / zs, self.cy - self.f * c[..., 1] / zs, z

    def p2(self, P):
        sx, sy, z = self.proj(P)
        return float(sx), float(sy), float(z)

    def scale_at(self, P):
        z = float(self.to_cam(P)[2])
        return self.f / max(z, 1e-3)

    def billboard(self, P, up_len=100.0):
        """affine (cairo.Matrix) mapping a local 2D frame (x right, y DOWN, in world cm) anchored at world P
        onto the screen: x follows the camera's horizontal right, y follows projected world-down (a cylindrical
        billboard: vertical things stay vertical in the world)."""
        P = np.asarray(P, float)
        h = np.array([self.right[0], 0.0, self.right[2]])
        n = np.linalg.norm(h)
        h = h / n if n > 1e-6 else np.array([1.0, 0, 0])
        x0, y0, z0 = self.p2(P)
        if z0 < self.near:
            return None, z0
        s = self.f / z0
        # horizontal axis: pure scale (no foreshortening for a camera-facing card)
        ax, ay = s, 0.0
        # vertical axis: projected world -Y
        x1, y1, z1 = self.p2(P - np.array([0.0, up_len, 0.0]))
        if z1 < self.near:
            bx, by = 0.0, s
        else:
            bx, by = (x1 - x0) / up_len, (y1 - y0) / up_len
        # keep the card upright-ish if the camera rolls: rotate x-axis by projected up direction
        L = math.hypot(bx, by) + 1e-9
        ux, uy = by / L, -bx / L          # perpendicular to vertical, pointing screen-right
        ax, ay = ux * s, uy * s
        return cairo.Matrix(ax, ay, bx, by, x0, y0), z0


# ------------------------------------------------------------------ clipping + drawing
def clip_near(C, near):
    """Sutherland-Hodgman of a polygon in camera coords (N,3) against z >= near"""
    out = []
    n = len(C)
    for i in range(n):
        a = C[i]
        b = C[(i + 1) % n]
        ain, bin_ = a[2] >= near, b[2] >= near
        if ain:
            out.append(a)
        if ain != bin_:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * t)
    return np.array(out) if len(out) >= 3 else None


def poly_screen(cam, P):
    """world polygon -> list of screen points (clipped) or None"""
    Cc = cam.to_cam(P)
    if (Cc[:, 2] < cam.near).all():
        return None
    if (Cc[:, 2] < cam.near).any():
        Cc = clip_near(Cc, cam.near)
        if Cc is None:
            return None
    z = Cc[:, 2]
    return np.stack([cam.cx + cam.f * Cc[:, 0] / z, cam.cy - cam.f * Cc[:, 1] / z], 1)


def path_pts(ctx, pts, close=True):
    ctx.move_to(float(pts[0, 0]), float(pts[0, 1]))
    for p in pts[1:]:
        ctx.line_to(float(p[0]), float(p[1]))
    if close:
        ctx.close_path()


def facing(cam, P, n):
    """True if polygon with normal n (world) faces the camera"""
    return float(np.dot(cam.pos - P[0], n)) > 0


def on_screen(pts, margin=200):
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return x1 > -margin and x0 < W + margin and y1 > -margin and y0 < H + margin


def fill_poly3(ctx, cam, P, rgb, alpha=1.0):
    pts = poly_screen(cam, P)
    if pts is None or not on_screen(pts):
        return None
    path_pts(ctx, pts)
    ctx.set_source_rgba(*rgb, alpha)
    ctx.fill()
    return pts


def grad_poly3(ctx, cam, P, cols, alpha=1.0, seam=0.0):
    """polygon with per-vertex colours cols (N,3): approximated by a linear gradient between the darkest
    and brightest vertex (screen space). seam>0 strokes the edge in the fill to hide AA cracks."""
    Cc = cam.to_cam(P)
    if (Cc[:, 2] < cam.near).all():
        return None
    pts = poly_screen(cam, P)
    if pts is None or not on_screen(pts):
        return None
    lum = cols @ np.array([0.3, 0.55, 0.15])
    i0, i1 = int(np.argmin(lum)), int(np.argmax(lum))
    a = cam.proj(P[i0])
    b = cam.proj(P[i1])
    path_pts(ctx, pts)
    if a[2] > cam.near and b[2] > cam.near and abs(lum[i1] - lum[i0]) > 0.01:
        g = cairo.LinearGradient(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
        g.add_color_stop_rgba(0, *cols[i0], alpha)
        g.add_color_stop_rgba(1, *cols[i1], alpha)
        ctx.set_source(g)
    else:
        ctx.set_source_rgba(*cols.mean(0), alpha)
    if seam > 0:
        ctx.fill_preserve()
        ctx.set_line_width(seam)
        ctx.stroke()
    else:
        ctx.fill()
    return pts


def polyline3(ctx, cam, P, rgb, width_world, alpha=1.0, min_w=0.35, max_w=60.0):
    """3D polyline with perspective width (uses mean depth). Clips segments behind the near plane."""
    Cc = cam.to_cam(P)
    z = Cc[:, 2]
    ok = z > cam.near
    if not ok.any():
        return
    zs = np.maximum(z, cam.near)
    sx = cam.cx + cam.f * Cc[:, 0] / zs
    sy = cam.cy - cam.f * Cc[:, 1] / zs
    wz = float(np.median(zs[ok]))
    w = min(max_w, max(min_w, width_world * cam.f / wz))
    ctx.set_line_width(w)
    ctx.set_source_rgba(*rgb, alpha)
    started = False
    for i in range(len(P)):
        if ok[i]:
            if not started:
                ctx.move_to(float(sx[i]), float(sy[i]))
                started = True
            else:
                ctx.line_to(float(sx[i]), float(sy[i]))
        else:
            started = False
    ctx.stroke()
    return w


# ------------------------------------------------------------------ shapes
def ngon(n, r, y, rot=0.0, cx=0.0, cz=0.0):
    a = rot + np.arange(n) * 2 * math.pi / n
    return np.stack([cx + r * np.sin(a), np.full(n, float(y)), cz + r * np.cos(a)], 1)


def pent(r, y, rot=math.pi, cx=0.0, cz=0.0):
    """pentagon in the XZ plane; default rotation puts a vertex at -Z (towards the usual camera) so the far
    side is a flat wall"""
    return ngon(5, r, y, rot, cx, cz)

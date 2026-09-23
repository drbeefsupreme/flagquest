"""s08 helper: numba rasteriser for thousands of tiny meadow people (NOT flags - flags go through vx.flag).

Everything here works in PIXEL coordinates on a premultiplied float32 RGBA buffer (h, w, 4).
Primitives are analytically anti-aliased (signed-distance coverage), so tiny far figures stay smooth.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, inline="always")
def _put(buf, x, y, r, g, b, a):
    ia = 1.0 - a
    buf[y, x, 0] = r * a + buf[y, x, 0] * ia
    buf[y, x, 1] = g * a + buf[y, x, 1] * ia
    buf[y, x, 2] = b * a + buf[y, x, 2] * ia
    buf[y, x, 3] = a + buf[y, x, 3] * ia


@njit(cache=True, fastmath=True)
def poly(buf, px, py, n, r, g, b, a):
    """convex polygon (any winding), AA by min signed edge distance"""
    H, W = buf.shape[0], buf.shape[1]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += px[i] * py[j] - px[j] * py[i]
    sg = 1.0 if area > 0 else -1.0
    if abs(area) < 0.08:          # degenerate -> dot
        cx = 0.0
        cy = 0.0
        for i in range(n):
            cx += px[i]
            cy += py[i]
        ix, iy = int(cx / n), int(cy / n)
        if 0 <= ix < W and 0 <= iy < H:
            _put(buf, ix, iy, r, g, b, min(1.0, abs(area) * 0.5 + 0.05) * a)
        return
    nx = np.empty(8)
    ny = np.empty(8)
    nc = np.empty(8)
    for i in range(n):
        j = (i + 1) % n
        ex, ey = px[j] - px[i], py[j] - py[i]
        L = math.sqrt(ex * ex + ey * ey) + 1e-9
        # inside distance d = sg * (ex*(y-py) - ey*(x-px)) / L
        nx[i] = -sg * ey / L
        ny[i] = sg * ex / L
        nc[i] = -(nx[i] * px[i] + ny[i] * py[i])
    x0 = px[0]
    x1 = px[0]
    y0 = py[0]
    y1 = py[0]
    for i in range(1, n):
        x0 = min(x0, px[i])
        x1 = max(x1, px[i])
        y0 = min(y0, py[i])
        y1 = max(y1, py[i])
    ix0 = max(0, int(math.floor(x0 - 1)))
    ix1 = min(W - 1, int(math.ceil(x1 + 1)))
    iy0 = max(0, int(math.floor(y0 - 1)))
    iy1 = min(H - 1, int(math.ceil(y1 + 1)))
    for y in range(iy0, iy1 + 1):
        fy = y + 0.5
        for x in range(ix0, ix1 + 1):
            fx = x + 0.5
            dm = 1e9
            for i in range(n):
                d = nx[i] * fx + ny[i] * fy + nc[i]
                if d < dm:
                    dm = d
            cov = dm + 0.5
            if cov <= 0.0:
                continue
            if cov > 1.0:
                cov = 1.0
            _put(buf, x, y, r, g, b, cov * a)


@njit(cache=True, fastmath=True)
def disk(buf, cx, cy, rad, r, g, b, a):
    H, W = buf.shape[0], buf.shape[1]
    if rad < 0.6:
        a = a * (rad / 0.6) ** 2
        rad = 0.6
    ix0 = max(0, int(cx - rad - 1))
    ix1 = min(W - 1, int(cx + rad + 1))
    iy0 = max(0, int(cy - rad - 1))
    iy1 = min(H - 1, int(cy + rad + 1))
    for y in range(iy0, iy1 + 1):
        dy = y + 0.5 - cy
        for x in range(ix0, ix1 + 1):
            dx = x + 0.5 - cx
            cov = rad - math.sqrt(dx * dx + dy * dy) + 0.5
            if cov <= 0.0:
                continue
            if cov > 1.0:
                cov = 1.0
            _put(buf, x, y, r, g, b, cov * a)


@njit(cache=True, fastmath=True)
def seg(buf, x0, y0, x1, y1, wd, r, g, b, a):
    """thick AA line segment (butt ends)"""
    if wd < 1.0:
        a = a * wd
        wd = 1.0
    dx, dy = x1 - x0, y1 - y0
    L = math.sqrt(dx * dx + dy * dy) + 1e-9
    nx, ny = -dy / L * wd * 0.5, dx / L * wd * 0.5
    px = np.empty(4)
    py = np.empty(4)
    px[0], py[0] = x0 + nx, y0 + ny
    px[1], py[1] = x1 + nx, y1 + ny
    px[2], py[2] = x1 - nx, y1 - ny
    px[3], py[3] = x0 - nx, y0 - ny
    poly(buf, px, py, 4, r, g, b, a)


@njit(cache=True, fastmath=True)
def draw_people(buf, fx, fy, k, hgt, lean, body, skin, hair, hood, hand, rim, rimk, order, s):
    """fx, fy: feet (design px); k: design px per metre; hgt: metres; lean: radians;
    body/skin/hair: (n,3) colours; hood: (n,) bool; hand: (n,4) shoulder_x, shoulder_y, hand_x, hand_y (design px);
    rim (3,) back-light colour, rimk (n,) rim strength; order: draw order; s = pixel scale."""
    px = np.empty(8)
    py = np.empty(8)
    for q in range(order.shape[0]):
        i = order[q]
        h = hgt[i] * k[i] * s
        if h < 0.8:
            continue
        x = fx[i] * s
        y = fy[i] * s
        sl = math.sin(lean[i])
        br, bg, bb = body[i, 0] * 0.9, body[i, 1] * 0.9, body[i, 2] * 0.9
        ra = min(1.0, rimk[i])
        rw = max(0.7, 0.014 * h)
        top = y - 0.80 * h
        lx = sl * 0.8 * h
        # robe: hexagon with rounded-ish shoulders
        px[0], py[0] = x - 0.21 * h, y
        px[1], py[1] = x + 0.21 * h, y
        px[2], py[2] = x + 0.15 * h + lx, top + 0.06 * h
        px[3], py[3] = x + 0.10 * h + lx, top
        px[4], py[4] = x - 0.10 * h + lx, top
        px[5], py[5] = x - 0.15 * h + lx, top + 0.06 * h
        n = 6 if h > 4.0 else 4
        if n == 4:
            px[2], py[2] = x + 0.12 * h + lx, top
            px[3], py[3] = x - 0.12 * h + lx, top
        if ra > 0.01 and h > 3.0:        # rim: the same silhouette nudged up, in light
            for m in range(n):
                py[m] -= rw
            poly(buf, px, py, n, rim[0], rim[1], rim[2], 0.8 * ra)
            for m in range(n):
                py[m] += rw
        poly(buf, px, py, n, br, bg, bb, 1.0)
        if h > 6.0:                      # shadow side for volume (sun is behind: the right half falls off)
            px[0], py[0] = x + 0.02 * h, y
            px[1], py[1] = x + 0.21 * h, y
            px[2], py[2] = x + 0.15 * h + lx, top + 0.06 * h
            px[3], py[3] = x + 0.10 * h + lx, top
            px[4], py[4] = x + 0.01 * h + lx, top
            poly(buf, px, py, 5, br * 0.78, bg * 0.78, bb * 0.8, 0.8)
        hx = x + sl * 0.9 * h
        hy = y - 0.895 * h
        hr = 0.085 * h
        # arm from shoulder to hand (holding the pole)
        if h > 5.0:
            seg(buf, hand[i, 0] * s, hand[i, 1] * s, hand[i, 2] * s, hand[i, 3] * s, 0.06 * h,
                br * 0.86, bg * 0.86, bb * 0.86, 1.0)
            disk(buf, hand[i, 2] * s, hand[i, 3] * s, 0.032 * h, skin[i, 0], skin[i, 1], skin[i, 2], 1.0)
        if hood[i]:
            if ra > 0.01 and h > 3.0:
                disk(buf, hx - 0.08 * hr, hy - 0.12 * hr - rw, hr * 1.2, rim[0], rim[1], rim[2], 0.85 * ra)
            disk(buf, hx - 0.08 * hr, hy - 0.12 * hr, hr * 1.2, br * 0.82, bg * 0.82, bb * 0.82, 1.0)
            if h > 5.0:
                disk(buf, hx + 0.12 * hr, hy + 0.12 * hr, hr * 0.72, skin[i, 0] * 0.85, skin[i, 1] * 0.85,
                     skin[i, 2] * 0.85, 1.0)
        else:
            if ra > 0.01 and h > 3.0:
                disk(buf, hx, hy - rw, hr, rim[0], rim[1], rim[2], 0.85 * ra)
            disk(buf, hx, hy, hr, skin[i, 0] * 0.9, skin[i, 1] * 0.9, skin[i, 2] * 0.9, 1.0)
            if h > 7.0:
                for m in range(6):
                    ang = math.pi + math.pi * m / 5.0
                    px[m] = hx + math.cos(ang) * hr * 1.05
                    py[m] = hy + math.sin(ang) * hr * 1.05 + 0.2 * hr
                poly(buf, px, py, 6, hair[i, 0], hair[i, 1], hair[i, 2], 1.0)

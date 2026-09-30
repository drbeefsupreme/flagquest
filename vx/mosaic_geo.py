"""vx.mosaic_geo - numba kernels for tesserae as small convex polygons (owner: Tessellator).

Placement with cut-to-fit: a new tile that overlaps already-set tiles is trimmed along the facing edge of each
neighbour (leaving a mortar joint), exactly as a mosaicist nips a tessera to fit; if too little is left it is
rejected and the gap-filler comes back later with a smaller piece.

Polygons: (KMAX, 2) float64 vertex arrays + a vertex count, positive signed area (algebraic), so the outward
normal of edge i->i+1 is (ey, -ex)/|e|.
"""
import math

import numpy as np
from numba import njit

KMAX = 8          # max vertices per tessera
KBUF = KMAX + 4   # working buffer while clipping


@njit(cache=True)
def area(P, n):
    a = 0.0
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        a += P[i, 0] * P[j, 1] - P[j, 0] * P[i, 1]
    return 0.5 * a


@njit(cache=True)
def centroid(P, n):
    a = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        cr = P[i, 0] * P[j, 1] - P[j, 0] * P[i, 1]
        a += cr
        cx += (P[i, 0] + P[j, 0]) * cr
        cy += (P[i, 1] + P[j, 1]) * cr
    if abs(a) < 1e-12:
        sx = 0.0
        sy = 0.0
        for i in range(n):
            sx += P[i, 0]
            sy += P[i, 1]
        return sx / max(n, 1), sy / max(n, 1)
    return cx / (3.0 * a), cy / (3.0 * a)


@njit(cache=True)
def radius(P, n, cx, cy):
    r = 0.0
    for i in range(n):
        d = (P[i, 0] - cx) ** 2 + (P[i, 1] - cy) ** 2
        if d > r:
            r = d
    return math.sqrt(r)


@njit(cache=True)
def edge(P, n, e):
    """outward unit normal (nx, ny) and offset c of edge e: the polygon lies in {nx*x + ny*y <= c}."""
    j = e + 1 if e + 1 < n else 0
    ex = P[j, 0] - P[e, 0]
    ey = P[j, 1] - P[e, 1]
    L = math.sqrt(ex * ex + ey * ey)
    if L < 1e-12:
        return 0.0, 0.0, -1e30
    nx = ey / L
    ny = -ex / L
    return nx, ny, nx * P[e, 0] + ny * P[e, 1]


@njit(cache=True)
def clip(P, n, nx, ny, c, out):
    """keep the part of convex polygon P with nx*x + ny*y >= c. Returns vertex count written to out."""
    m = 0
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        di = P[i, 0] * nx + P[i, 1] * ny - c
        dj = P[j, 0] * nx + P[j, 1] * ny - c
        if di >= 0.0:
            if m < out.shape[0]:
                out[m, 0] = P[i, 0]
                out[m, 1] = P[i, 1]
                m += 1
        if (di >= 0.0) != (dj >= 0.0):
            t = di / (di - dj)
            if m < out.shape[0]:
                out[m, 0] = P[i, 0] + t * (P[j, 0] - P[i, 0])
                out[m, 1] = P[i, 1] + t * (P[j, 1] - P[i, 1])
                m += 1
    return m


@njit(cache=True)
def min_width(P, n):
    """minimal width of a convex polygon (min over edges of the farthest vertex from that edge line)."""
    best = 1e30
    for e in range(n):
        nx, ny, c = edge(P, n, e)
        if c < -1e29:
            continue
        far = 0.0
        for i in range(n):
            d = c - (P[i, 0] * nx + P[i, 1] * ny)
            if d > far:
                far = d
        if far < best:
            best = far
    return best


@njit(cache=True)
def simplify(P, n, kmax):
    """drop the vertices with the smallest turn until n <= kmax (keeps convexity)."""
    while n > kmax:
        bi = 0
        bs = 1e30
        for i in range(n):
            a = i - 1 if i > 0 else n - 1
            b = i + 1 if i + 1 < n else 0
            ax = P[i, 0] - P[a, 0]
            ay = P[i, 1] - P[a, 1]
            bx = P[b, 0] - P[i, 0]
            by = P[b, 1] - P[i, 1]
            s = abs(ax * by - ay * bx)
            if s < bs:
                bs = s
                bi = i
        for i in range(bi, n - 1):
            P[i, 0] = P[i + 1, 0]
            P[i, 1] = P[i + 1, 1]
        n -= 1
    return n


@njit(cache=True)
def dedupe(P, n, eps):
    m = 0
    for i in range(n):
        if m > 0 and abs(P[i, 0] - P[m - 1, 0]) < eps and abs(P[i, 1] - P[m - 1, 1]) < eps:
            continue
        P[m, 0] = P[i, 0]
        P[m, 1] = P[i, 1]
        m += 1
    while m > 1 and abs(P[0, 0] - P[m - 1, 0]) < eps and abs(P[0, 1] - P[m - 1, 1]) < eps:
        m -= 1
    return m


@njit(cache=True)
def separate(P, n, Q, m, g, tmp):
    """Trim P so it keeps a joint g from convex Q. Returns the new count (P modified in place), n unchanged
    if already separated."""
    best = 1e30
    bnx = 0.0
    bny = 0.0
    bc = 0.0
    for e in range(m):
        nx, ny, c = edge(Q, m, e)
        if c < -1e29:
            continue
        pmin = 1e30
        for i in range(n):
            d = P[i, 0] * nx + P[i, 1] * ny
            if d < pmin:
                pmin = d
        depth = c + g - pmin
        if depth <= 0.0:
            return n
        if depth < best:
            best = depth
            bnx = nx
            bny = ny
            bc = c
    for f in range(n):
        nx, ny, c = edge(P, n, f)
        if c < -1e29:
            continue
        qmin = 1e30
        for i in range(m):
            d = Q[i, 0] * nx + Q[i, 1] * ny
            if d < qmin:
                qmin = d
        if qmin - c >= g:
            return n
    k = clip(P, n, bnx, bny, bc + g, tmp)
    for i in range(k):
        P[i, 0] = tmp[i, 0]
        P[i, 1] = tmp[i, 1]
    return k


@njit(cache=True)
def place_batch(C, cn, cinfo, minfrac, minw, joint,
                polys, nvs, cen, rad, info, head, nxt, cnt, grid, rmax):
    """Place candidate polygons C[k,:cn[k]] one after another with cut-to-fit.
    cinfo[k] = (T, angle, level, chain, joint_scale, kind). grid = (x0, y0, cell, ncx, ncy). cnt, rmax: (1,) arrays.
    minw: min width as a fraction of T. Returns number placed."""
    P = np.empty((KBUF, 2))
    tmp = np.empty((KBUF, 2))
    x0 = grid[0]
    y0 = grid[1]
    cell = grid[2]
    ncx = int(grid[3])
    ncy = int(grid[4])
    cap = polys.shape[0]
    placed = 0
    for k in range(C.shape[0]):
        if cnt[0] >= cap:
            break
        n = cn[k]
        if n < 3:
            continue
        for i in range(n):
            P[i, 0] = C[k, i, 0]
            P[i, 1] = C[k, i, 1]
        a0 = area(P, n)
        if a0 < 0.0:  # enforce positive orientation
            for i in range(n // 2):
                tx = P[i, 0]
                ty = P[i, 1]
                P[i, 0] = P[n - 1 - i, 0]
                P[i, 1] = P[n - 1 - i, 1]
                P[n - 1 - i, 0] = tx
                P[n - 1 - i, 1] = ty
            a0 = -a0
        if a0 < 1e-6:
            continue
        T = cinfo[k, 0]
        g = joint * cinfo[k, 4] if cinfo[k, 4] > 0 else joint
        cx, cy = centroid(P, n)
        r = radius(P, n, cx, cy)
        reach = r + rmax[0] + g
        i0 = int(math.floor((cx - reach - x0) / cell))
        i1 = int(math.floor((cx + reach - x0) / cell))
        j0 = int(math.floor((cy - reach - y0) / cell))
        j1 = int(math.floor((cy + reach - y0) / cell))
        ok = True
        for cj in range(max(j0, 0), min(j1, ncy - 1) + 1):
            if not ok:
                break
            for ci in range(max(i0, 0), min(i1, ncx - 1) + 1):
                if not ok:
                    break
                q = head[cj * ncx + ci]
                while q >= 0:
                    dx = cen[q, 0] - cx
                    dy = cen[q, 1] - cy
                    lim = r + rad[q] + g
                    if dx * dx + dy * dy < lim * lim:
                        n = separate(P, n, polys[q], nvs[q], g, tmp)
                        if n < 3:
                            ok = False
                            break
                        n = dedupe(P, n, 1e-7)
                        if n < 3 or area(P, n) < minfrac * a0:
                            ok = False
                            break
                    q = nxt[q]
        if not ok:
            continue
        if n > KMAX:
            n = simplify(P, n, KMAX)
        if min_width(P, n) < minw * T:
            continue
        idx = cnt[0]
        for i in range(n):
            polys[idx, i, 0] = P[i, 0]
            polys[idx, i, 1] = P[i, 1]
        for i in range(n, KMAX):
            polys[idx, i, 0] = P[n - 1, 0]
            polys[idx, i, 1] = P[n - 1, 1]
        nvs[idx] = n
        ncx_, ncy_ = centroid(P, n)
        cen[idx, 0] = ncx_
        cen[idx, 1] = ncy_
        rr = radius(P, n, ncx_, ncy_)
        rad[idx] = rr
        if rr > rmax[0]:
            rmax[0] = rr
        for j in range(info.shape[1]):
            info[idx, j] = cinfo[k, j]
        ci = min(max(int(math.floor((ncx_ - x0) / cell)), 0), ncx - 1)
        cj = min(max(int(math.floor((ncy_ - y0) / cell)), 0), ncy - 1)
        c = cj * ncx + ci
        nxt[idx] = head[c]
        head[c] = idx
        cnt[0] = idx + 1
        placed += 1
    return placed


@njit(cache=True)
def chain_quads(pts, nrm, T, Hh, closed, jscale, lenvar, irr, seed, out, oinfo, level, chain, kind):
    """Walk a resampled chain (uniform spacing) and cut it into quads (tiles along a course).
    pts/nrm (M,2) design units, T/Hh (M,) tile size and row height at each point. Writes quads to out
    (C,4,2) and (T, ang, level, chain, jscale, kind) to oinfo. Returns count."""
    M = pts.shape[0]
    if M < 3:
        return 0
    np.random.seed(seed)
    ds = 0.0
    for i in range(1, M):
        ds += math.sqrt((pts[i, 0] - pts[i - 1, 0]) ** 2 + (pts[i, 1] - pts[i - 1, 1]) ** 2)
    if closed:
        ds += math.sqrt((pts[0, 0] - pts[M - 1, 0]) ** 2 + (pts[0, 1] - pts[M - 1, 1]) ** 2)
        step = ds / M
    else:
        step = ds / (M - 1)
    total = ds
    if step <= 0:
        return 0
    s0 = np.random.random() * T[0] if closed else np.random.random() * 0.45 * T[0]
    cnt = 0
    guard = 0
    while cnt < out.shape[0] and guard < 100000:
        guard += 1
        i0 = int(s0 / step)
        if i0 >= M:
            if closed:
                i0 = i0 % M
            else:
                break
        t = T[i0]
        g = 0.09 * t * jscale
        if g < 0.45:
            g = 0.45
        if g > 2.2:
            g = 2.2
        L = t * (1.0 + lenvar * (2.0 * np.random.random() - 1.0)) - g
        if L < 0.45 * t:
            L = 0.45 * t
        s1 = s0 + L
        if closed:
            if s0 >= total:
                break
        else:
            if s1 > total:
                rem = total - s0
                if rem < 0.5 * t:
                    break
                s1 = total
                L = rem
        # interpolate position and normal at s0 and s1
        p0x, p0y, n0x, n0y = _interp(pts, nrm, s0, step, M, closed)
        p1x, p1y, n1x, n1y = _interp(pts, nrm, s1, step, M, closed)
        h = Hh[i0] * 0.5
        # corners
        ax = p0x + n0x * h
        ay = p0y + n0y * h
        bx = p1x + n1x * h
        by = p1y + n1y * h
        cx = p1x - n1x * h
        cy = p1y - n1y * h
        dx = p0x - n0x * h
        dy = p0y - n0y * h
        # validity: both 'rails' must run the same way as the chord, else build a chord rectangle
        chx = p1x - p0x
        chy = p1y - p0y
        cl = math.sqrt(chx * chx + chy * chy)
        if cl < 1e-6:
            s0 = s1 + g
            continue
        ux = chx / cl
        uy = chy / cl
        top = (bx - ax) * ux + (by - ay) * uy
        bot = (cx - dx) * ux + (cy - dy) * uy
        if top < 0.25 * cl or bot < 0.25 * cl:
            vx = -uy
            vy = ux
            if vx * (n0x + n1x) + vy * (n0y + n1y) < 0:
                vx = -vx
                vy = -vy
            ax = p0x + vx * h
            ay = p0y + vy * h
            bx = p1x + vx * h
            by = p1y + vy * h
            cx = p1x - vx * h
            cy = p1y - vy * h
            dx = p0x - vx * h
            dy = p0y - vy * h
        j = irr * t
        out[cnt, 0, 0] = ax + j * np.random.randn()
        out[cnt, 0, 1] = ay + j * np.random.randn()
        out[cnt, 1, 0] = bx + j * np.random.randn()
        out[cnt, 1, 1] = by + j * np.random.randn()
        out[cnt, 2, 0] = cx + j * np.random.randn()
        out[cnt, 2, 1] = cy + j * np.random.randn()
        out[cnt, 3, 0] = dx + j * np.random.randn()
        out[cnt, 3, 1] = dy + j * np.random.randn()
        oinfo[cnt, 0] = t
        oinfo[cnt, 1] = math.atan2(chy, chx)
        oinfo[cnt, 2] = level
        oinfo[cnt, 3] = chain
        oinfo[cnt, 4] = jscale
        oinfo[cnt, 5] = kind
        cnt += 1
        s0 = s1 + g
        if closed and s0 >= total:
            break
    return cnt


@njit(cache=True)
def _interp(pts, nrm, s, step, M, closed):
    f = s / step
    i = int(math.floor(f))
    u = f - i
    if closed:
        i0 = i % M
        i1 = (i + 1) % M
    else:
        if i >= M - 1:
            i0 = M - 1
            i1 = M - 1
            u = 0.0
        else:
            i0 = i
            i1 = i + 1
    px = pts[i0, 0] + u * (pts[i1, 0] - pts[i0, 0])
    py = pts[i0, 1] + u * (pts[i1, 1] - pts[i0, 1])
    nx = nrm[i0, 0] + u * (nrm[i1, 0] - nrm[i0, 0])
    ny = nrm[i0, 1] + u * (nrm[i1, 1] - nrm[i0, 1])
    L = math.sqrt(nx * nx + ny * ny)
    if L > 1e-9:
        nx /= L
        ny /= L
    return px, py, nx, ny


@njit(cache=True)
def raster_cover(mask, polys, nvs, count, k, grow):
    """mark pixels (pixel centres at (i+0.5)/k design units) within `grow` of any tile polygon."""
    h, w = mask.shape
    for t in range(count):
        n = nvs[t]
        xmin = 1e30
        xmax = -1e30
        ymin = 1e30
        ymax = -1e30
        for i in range(n):
            x = polys[t, i, 0]
            y = polys[t, i, 1]
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
        a = max(int((xmin - grow) * k), 0)
        b = min(int((xmax + grow) * k) + 1, w - 1)
        c = max(int((ymin - grow) * k), 0)
        d = min(int((ymax + grow) * k) + 1, h - 1)
        for py in range(c, d + 1):
            yy = (py + 0.5) / k
            for px in range(a, b + 1):
                xx = (px + 0.5) / k
                dmax = -1e30
                for e in range(n):
                    nx, ny, cc = edge(polys[t], n, e)
                    if cc < -1e29:
                        continue
                    dd = nx * xx + ny * yy - cc
                    if dd > dmax:
                        dmax = dd
                if dmax <= grow:
                    mask[py, px] = 1


@njit(cache=True)
def split4(polys, nvs, cen, ang, joint, out, onv):
    """split each convex polygon into 4 along its two axes through the centroid, leaving joints.
    out (4N, KMAX, 2), onv (4N,). Children of tile i are 4i..4i+3."""
    N = polys.shape[0]
    P = np.empty((KBUF, 2))
    A = np.empty((KBUF, 2))
    B = np.empty((KBUF, 2))
    for i in range(N):
        n = nvs[i]
        ca = math.cos(ang[i])
        sa = math.sin(ang[i])
        cx = cen[i, 0]
        cy = cen[i, 1]
        g = joint[i] * 0.5
        for q in range(4):
            sx = 1.0 if (q == 0 or q == 3) else -1.0
            sy = 1.0 if q < 2 else -1.0
            for k in range(n):
                P[k, 0] = polys[i, k, 0]
                P[k, 1] = polys[i, k, 1]
            m = n
            # half-plane along axis u: sx * ((x-c).u) >= g
            nx = ca * sx
            ny = sa * sx
            m = clip(P, m, nx, ny, nx * cx + ny * cy + g, A)
            # axis v = (-sa, ca)
            nx = -sa * sy
            ny = ca * sy
            m = clip(A, m, nx, ny, nx * cx + ny * cy + g, B)
            o = 4 * i + q
            if m > KMAX:
                m = simplify(B, m, KMAX)
            onv[o] = m
            for k in range(m):
                out[o, k, 0] = B[k, 0]
                out[o, k, 1] = B[k, 1]
            for k in range(m, KMAX):
                out[o, k, 0] = B[max(m - 1, 0), 0]
                out[o, k, 1] = B[max(m - 1, 0), 1]

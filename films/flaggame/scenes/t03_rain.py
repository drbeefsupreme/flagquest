"""Engraved rain for the flashback (owner T03Genesis; imported by T04FlagMaker for continuity).

Paint these on the GREY world BEFORE vx.ink.engrave: the streaks are pure paper white (>= 0.97), so they
print as bare-paper gaps cutting through the hatching (Dore's rain), never as grey. Deterministic: a pure
function of the global time T. NEVER draw rain on the flag layer (flags are spotted in afterwards).

draw_rain(ctx, T, *, rect=(0, 0, 1920, 1080), density=1.0, slant=0.18, seed=0, alpha=1.0, speed=1.0,
          width=1.0, length=1.0)
    downpour falling top -> bottom, slanting to the RIGHT by `slant` (dx/dy; wind from the left).
    Three depth layers (far: fine, dense, slower on screen; near: long, bold, fast). density 1.0 = heavy
    downpour (~1600 streaks on a full frame); 0 = none. width/length scale the streaks.
splashes(ctx, T, y0, y1, *, x0=0, x1=1920, density=1.0, seed=0, alpha=1.0, size=1.0)
    rain crowns bursting on the mud between screen rows y0 (far, small) .. y1 (near, big).
"""
import math

import numpy as np

_CACHE = {}

LAYERS = (  # (streaks per 1920x1080 at density 1, speed px/s, length px, width px)
    (900, 1500.0, 38.0, 1.1),
    (520, 2300.0, 70.0, 1.7),
    (180, 3300.0, 125.0, 2.6),
)


def _table(seed, n, k):
    key = (seed, n, k)
    t = _CACHE.get(key)
    if t is None:
        rng = np.random.default_rng(seed * 7919 + k * 131 + 17)
        t = (rng.random(n), rng.random(n), rng.uniform(0.75, 1.25, n), rng.uniform(0.8, 1.2, n))
        _CACHE[key] = t
    return t


def draw_rain(ctx, T, *, rect=(0, 0, 1920, 1080), density=1.0, slant=0.18, seed=0, alpha=1.0, speed=1.0,
              width=1.0, length=1.0):
    if density <= 0 or alpha <= 0:
        return
    x0, y0, w, h = rect
    area = (w * h) / (1920.0 * 1080.0)
    ctx.save()
    ctx.rectangle(x0, y0, w, h)
    ctx.clip()
    ctx.set_line_cap(1)
    ctx.set_source_rgba(1.0, 1.0, 1.0, alpha)
    for k, (n0, sp, L, wd) in enumerate(LAYERS):
        n = int(n0 * density * area)
        if n <= 0:
            continue
        xs, ph, ls, vs = _table(seed, max(n, 1), k)
        xs, ph, ls, vs = xs[:n], ph[:n], ls[:n], vs[:n]
        Lk = L * length
        span = h + Lk * 1.5
        yy = ((ph * span + T * sp * speed * vs) % span) + y0 - Lk
        # streak bases spread over the slanted band so the rain never thins at an edge
        xx = x0 - h * max(slant, 0.0) + xs * (w + h * abs(slant)) + slant * (yy - y0)
        ctx.set_line_width(wd * width)
        lx = Lk * ls * slant / math.sqrt(1 + slant * slant)
        ly = Lk * ls / math.sqrt(1 + slant * slant)
        for i in range(n):
            ctx.move_to(xx[i], yy[i])
            ctx.line_to(xx[i] + lx[i], yy[i] + ly[i])
        ctx.stroke()
    ctx.restore()


def splashes(ctx, T, y0, y1, *, x0=0.0, x1=1920.0, density=1.0, seed=0, alpha=1.0, size=1.0):
    """white crowns + droplets on the mud; each lives ~0.2 s"""
    if density <= 0:
        return
    n = int(260 * density * (x1 - x0) / 1920.0)
    life = 0.22
    slot = math.floor(T / life)
    ctx.save()
    ctx.set_line_cap(1)
    ctx.set_source_rgba(1.0, 1.0, 1.0, alpha)
    for s in (slot - 1, slot):
        rng = np.random.default_rng((seed * 100003 + int(s) * 7 + 5) & 0x7FFFFFFF)
        xs = rng.uniform(x0, x1, n)
        v = rng.random(n) ** 0.8
        off = rng.random(n) * life
        for i in range(n):
            age = T - (s * life + off[i])
            if age < 0 or age > life:
                continue
            u = age / life
            y = y0 + (y1 - y0) * v[i]
            k = (0.35 + 1.4 * v[i]) * size
            r = k * (5 + 14 * u)
            ctx.set_line_width(max(0.8, k * 1.6 * (1 - u)))
            # crown: an open ellipse arc
            ctx.save()
            ctx.translate(xs[i], y)
            ctx.scale(1.0, 0.32)
            ctx.arc(0, 0, r, math.pi * 1.05, math.pi * 1.95)
            ctx.restore()
            ctx.stroke()
            # droplets thrown up
            for j in (-1, 0, 1):
                hx = xs[i] + j * r * 0.8
                hy = y - k * (18 * u - 16 * u * u) * (1.3 - 0.3 * abs(j)) - k * 2
                ctx.arc(hx, hy, max(0.7, k * 1.3 * (1 - u)), 0, 2 * math.pi)
                ctx.fill()
    ctx.restore()

"""t01 — the churned playa mud under the tract (procedural heightfields, cm units).  Owner: T01Tract.

World frame (cm): x right, y forward (away from camera), z up.  Two nested fields:
  near: NEAR_SIZE cm square around NEAR_C at NEAR_N^2 texels (fine detail around the tract)
  far : FAR_SIZE cm square around FAR_C at FAR_N^2 texels (boot prints, ruts, puddles to the horizon)
Water table WATER_Z: everything below it is a rain puddle.
height(x, y) samples the combined field on the CPU (paper sim ground contact).
"""
import math

import numpy as np
import cv2

NEAR_C = (-8.0, 12.0)
NEAR_SIZE = 110.0
NEAR_N = 2048
FAR_C = (0.0, 1250.0)
FAR_SIZE = 2900.0
FAR_N = 2048
WATER_Z = -0.55

# the tract rests here (loop pose); the gust carries it to the left (see t01_paper)
TRACT_XY = (4.0, 8.0)


def _fbm(n, size_cm, feature_cm, octaves, seed, gain=0.5):
    """periodic-free fbm via upsampled white noise (cv2), returns (n, n) float32 ~[-1,1]"""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n), np.float32)
    amp, norm = 1.0, 0.0
    f = feature_cm
    for o in range(octaves):
        cells = max(2, int(round(size_cm / f)))
        g = rng.standard_normal((cells + 3, cells + 3)).astype(np.float32)
        up = cv2.resize(g, (n + int(3 * n / cells), n + int(3 * n / cells)), interpolation=cv2.INTER_CUBIC)
        off = int(1.5 * n / cells)
        out += amp * up[off:off + n, off:off + n]
        norm += amp
        amp *= gain
        f *= 0.5
    return out / norm


def _grid(n, center, size):
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    x = center[0] - size / 2 + (xs + 0.5) * size / n
    y = center[1] - size / 2 + (ys + 0.5) * size / n
    return x, y


def _boot(x, y, px, py, ang, left, depth=1.7, seed=0):
    """one boot print (sole + heel with lug tread), carved; returns (carve, rim) fields"""
    c, s = math.cos(ang), math.sin(ang)
    lx = (x - px) * c + (y - py) * s          # across the foot
    ly = -(x - px) * s + (y - py) * c         # along the foot (toe +)
    lx = lx * (1 if left else -1)
    # sole outline: toe ellipse + waist + heel
    toe = (lx / 5.4) ** 2 + ((ly - 6.0) / 9.5) ** 2
    heel = ((lx + 0.3) / 4.4) ** 2 + ((ly + 9.0) / 5.2) ** 2
    waist = np.where(np.abs(ly + 2.0) < 4.0, (lx / 4.2) ** 2, 9.0)
    d = np.minimum(np.minimum(toe, heel), waist)
    inside = np.clip((1.0 - d) * 6.0, 0, 1)
    # chevron lugs
    lug = 0.5 + 0.5 * np.sign(np.sin((ly + np.abs(lx) * 0.55) * 1.35))
    heel_lug = 0.5 + 0.5 * np.sign(np.sin(ly * 1.6))
    lugs = np.where(ly < -4.0, heel_lug, lug)
    carve = inside * depth * (0.78 + 0.22 * lugs)
    rim = np.exp(-((np.sqrt(np.maximum(d, 0)) - 1.12) / 0.22) ** 2) * 0.30 * depth
    return carve, rim


def _tire(x, y, pts, width=4.8, depth=1.1):
    """bike tire rut along a polyline: carve + raised lips + tread chevrons"""
    best = np.full(x.shape, 1e9, np.float32)
    along = np.zeros(x.shape, np.float32)
    acc = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / (L * L), 0, 1)
        qx, qy = x0 + t * dx - x, y0 + t * dy - y
        dd = np.sqrt(qx * qx + qy * qy)
        m = dd < best
        best = np.where(m, dd, best)
        along = np.where(m, acc + t * L, along)
        acc += L
    u = best / (width / 2)
    prof = np.clip(1 - u * u, 0, 1) ** 0.6
    tread = 0.5 + 0.5 * np.sign(np.sin(along * 2.2 + np.minimum(best, width) * 1.1))
    carve = prof * depth * (0.85 + 0.15 * tread)
    lip = np.exp(-((u - 1.25) / 0.35) ** 2) * depth * 0.45
    return carve, lip


def build(seed=69):
    """-> dict(near=(n,n) float32 heights cm, far=(n,n), near_rect, far_rect, water)"""
    out = {}
    for name, n, c, size in (("near", NEAR_N, NEAR_C, NEAR_SIZE), ("far", FAR_N, FAR_C, FAR_SIZE)):
        x, y = _grid(n, c, size)
        h = np.zeros((n, n), np.float32)
        # broad undulation + churn
        h += 1.4 * _fbm(n, size, 60.0, 3, seed + 1)
        churn = _fbm(n, size, 16.0, 3, seed + 2, 0.45)
        h += 0.8 * churn
        # sticky clumps: ridged noise (soft crests of pushed mud)
        rid = 1.0 - np.abs(_fbm(n, size, 9.0, 2, seed + 3, 0.5)) * 2.2
        h += 0.30 * np.clip(rid, 0, 1) ** 2
        # boot-print trail(s): a walker crossing behind the tract, another near the camera
        rng = np.random.default_rng(seed + 7)
        carve = np.zeros_like(h)
        rim = np.zeros_like(h)
        trails = [((-160.0, 70.0), (150.0, 38.0)), ((-120.0, -40.0), (160.0, -25.0)), ((40.0, 330.0), (-60.0, 90.0)),
                  ((-200.0, 160.0), (200.0, 210.0))]
        for tk, ((ax, ay), (bx, by)) in enumerate(trails):
            L = math.hypot(bx - ax, by - ay)
            ang = math.atan2(bx - ax, by - ay)      # foot points along travel
            steps = int(L / 36.0)
            for k in range(steps):
                t = (k + 0.5) / steps
                side = 1 if k % 2 == 0 else -1
                px = ax + (bx - ax) * t + math.cos(ang) * 9.0 * side
                py = ay + (by - ay) * t - math.sin(ang) * 9.0 * side
                px += rng.normal(0, 1.2)
                py += rng.normal(0, 1.2)
                if abs(px - c[0]) > size / 2 + 20 or abs(py - c[1]) > size / 2 + 20:
                    continue
                # keep the tract's rest mound clean
                if math.hypot(px - TRACT_XY[0], py - TRACT_XY[1]) < 20 or math.hypot(px + 14, py - TRACT_XY[1]) < 18:
                    continue
                cv_, rm = _boot(x, y, px, py, -ang + rng.normal(0, 0.12), side > 0, 1.5 + rng.random() * 0.9)
                carve = np.maximum(carve, cv_)
                rim = np.maximum(rim, rm)
        # churn: many overlapping random boot prints all over (a Burn walks through here all day)
        rng2 = np.random.default_rng(seed + 99 + (0 if name == "near" else 1))
        m = 70 if name == "near" else 2500
        for k in range(m):
            px = c[0] + (rng2.random() - 0.5) * size
            py = c[1] + (rng2.random() - 0.5) * size
            if math.hypot(px - TRACT_XY[0], py - TRACT_XY[1] - 2) < 24 or math.hypot(px + 12, py - TRACT_XY[1]) < 22:
                continue
            a1, l1, d1, k1 = rng2.random() * 6.28, rng2.random() < 0.5, 0.8 + rng2.random() * 1.2, 0.6 + 0.4 * rng2.random()
            px0 = c[0] - size / 2
            py0 = c[1] - size / 2
            i0 = max(0, int((px - 22 - px0) / size * n))
            i1 = min(n, int((px + 22 - px0) / size * n) + 1)
            j0 = max(0, int((py - 22 - py0) / size * n))
            j1 = min(n, int((py + 22 - py0) / size * n) + 1)
            if i1 <= i0 or j1 <= j0:
                continue
            cv_, rm = _boot(x[j0:j1, i0:i1], y[j0:j1, i0:i1], px, py, a1, l1, d1)
            carve[j0:j1, i0:i1] = np.maximum(carve[j0:j1, i0:i1], cv_ * k1)
            rim[j0:j1, i0:i1] = np.maximum(rim[j0:j1, i0:i1], rm)
        # bike rut arcing across the foreground and one behind
        for pts in ([(-220, -18), (-80, -6), (40, -12), (220, -34)], [(-220, 120), (-60, 96), (80, 104), (240, 140)]):
            cv_, lp = _tire(x, y, pts)
            carve = np.maximum(carve, cv_)
            rim = np.maximum(rim, lp)
        h += rim - carve
        # the tract's rest: a low smooth mound (pushed-up bank of the rut) tilted toward the camera
        for (mx, my, rx, ry, a, tilt) in ((TRACT_XY[0], TRACT_XY[1] + 2, 19.0, 16.0, 1.8, 0.20),
                                          (-12.0, TRACT_XY[1] + 1, 20.0, 18.0, 1.3, 0.06)):
            dm = ((x - mx) / rx) ** 2 + ((y - my) / ry) ** 2
            bump = np.exp(-dm * 1.1)
            h = h * (1 - 0.85 * bump) + (a + tilt * (y - my)) * bump
        # fine micro-roughness only in the near field (the far field is blurred anyway)
        if name == "near":
            h += 0.02 * _fbm(n, size, 1.2, 2, seed + 11)
        out[name] = h.astype(np.float32)
    out["near_rect"] = (NEAR_C[0] - NEAR_SIZE / 2, NEAR_C[1] - NEAR_SIZE / 2, NEAR_SIZE, NEAR_SIZE)
    out["far_rect"] = (FAR_C[0] - FAR_SIZE / 2, FAR_C[1] - FAR_SIZE / 2, FAR_SIZE, FAR_SIZE)
    out["water"] = WATER_Z
    return out


def load(cache_dir):
    p = cache_dir / "mud_v1.npz"
    if p.exists():
        z = np.load(p)
        return {"near": z["near"], "far": z["far"], "near_rect": tuple(z["near_rect"]), "far_rect": tuple(z["far_rect"]),
                "water": float(z["water"])}
    d = build()
    np.savez(p, near=d["near"], far=d["far"], near_rect=np.array(d["near_rect"]), far_rect=np.array(d["far_rect"]),
             water=d["water"])
    return d


class Ground:
    """CPU sampler of the combined field for the paper sim: smooth (the paper bridges micro-lumps)."""

    def __init__(self, mud, smooth_cm=0.6):
        self.mud = mud
        n = mud["near"].shape[0]
        k = smooth_cm / (mud["near_rect"][2] / n)
        self.near = cv2.GaussianBlur(mud["near"], (0, 0), k) if k > 0.3 else mud["near"]
        self.far = cv2.GaussianBlur(mud["far"], (0, 0), max(0.5, smooth_cm / (mud["far_rect"][2] / mud["far"].shape[0])))

    @staticmethod
    def _bil(a, rect, x, y):
        n = a.shape[0]
        u = (x - rect[0]) / rect[2] * n - 0.5
        v = (y - rect[1]) / rect[3] * n - 0.5
        u = np.clip(u, 0, n - 1.001)
        v = np.clip(v, 0, n - 1.001)
        i, j = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - i, v - j
        return (a[j, i] * (1 - fu) * (1 - fv) + a[j, i + 1] * fu * (1 - fv) + a[j + 1, i] * (1 - fu) * fv
                + a[j + 1, i + 1] * fu * fv)

    def height(self, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        r = self.mud["near_rect"]
        hn = self._bil(self.near, r, x, y)
        hf = self._bil(self.far, self.mud["far_rect"], x, y)
        # blend near -> far over the outer 10% of the near field
        ex = np.maximum(np.abs(x - (r[0] + r[2] / 2)), np.abs(y - (r[1] + r[3] / 2))) / (r[2] / 2)
        w = np.clip((ex - 0.88) / 0.1, 0, 1)
        return hn * (1 - w) + hf * w

    def grid(self, x0, y0, x1, y1, n):
        """sampled height grid for the numba sim: (n, n) array + rect"""
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        X, Y = np.meshgrid(xs, ys)
        return self.height(X, Y).astype(np.float64), (x0, y0, x1 - x0, y1 - y0)

"""m10 - the Basilica of the Unbabeling seen from the nave: geometry, charts, camera path, projector (owner M10).

World: design units (1 unit = 0.25 cm, 400 per metre); X right, Y DOWN, Z toward the viewer (vx.mosaic v1).
The apse faces the camera at the far end (-Z). Charts follow the vx.mosaic v1 conventions:
  conch      quarter sphere, centre (AX, SPRING_Y, 0), radius R; front-view stereographic chart = world (x, y) at z=0
             (conch = the upper half-disc v <= SPRING_Y); chart scale k = 2/(1+(rho/R)^2) world units per chart unit
  hemicycle  half cylinder, axis (AX, *, 0), radius R, from the floor up to the springing; chart u = HEMI_U0 + R*phi
             (phi = azimuth, 0 at the back, negative to the left; u in 0..pi R), v = world y
  planes     arch front (z = ARCH_Z), side walls (x = XL / XR), floor (y = FLOOR_Y), ceiling (y = CEIL_Y);
             chart = origin + u*ex + v*ey (see PLANES)
  soffit     the tunnel through the arch wall (z 0..ARCH_Z): half cylinder about the horizontal axis (AX, SPRING_Y)
             above the springing (vx.mosaic Vault chart: u = arc length from the left springing, v = ARCH_Z - z)
  jambs      its two vertical sides (S_JAMBL: u = ARCH_Z - z, S_JAMBR: u = z; v = y - SPRING_Y)
"""
import math

import numpy as np
from numba import njit

U = 400.0                      # world units per metre
R = 6.0 * U                    # apse radius
AX = R                         # apse axis x (conch chart origin at the top-left of the half-disc)
SPRING_Y = R                   # springing line (conch / hemicycle seam), y
FLOOR_Y = SPRING_Y + 8.0 * U   # floor: the springing is 8 m up
CEIL_Y = FLOOR_Y - 18.0 * U    # nave ceiling 18 m
ARCH_Z = 1.0 * U               # front face of the triumphal-arch wall (1 m thick)
XL, XR = AX - 8.0 * U, AX + 8.0 * U   # nave side walls (16 m nave: 2 m piers beside the 12 m apse)
Z_FAR = ARCH_Z + 34.0 * U      # west end of the modelled nave

# surface ids
S_NONE, S_CONCH, S_HEMI, S_ARCH, S_SOFFIT, S_LEFT, S_RIGHT, S_FLOOR, S_CEIL, S_JAMBL, S_JAMBR = range(11)
SURF_NAMES = ["none", "conch", "hemi", "arch", "soffit", "left", "right", "floor", "ceil", "jambL", "jambR"]

# plane charts: origin, ex, ey  (chart (u, v) -> origin + u ex + v ey)
PLANES = {
    S_ARCH: ((XL, CEIL_Y, ARCH_Z), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    S_LEFT: ((XL, CEIL_Y, Z_FAR), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    S_RIGHT: ((XR, CEIL_Y, ARCH_Z), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    S_FLOOR: ((XL, FLOOR_Y, -R), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    S_CEIL: ((XL, CEIL_Y, ARCH_Z), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}
# chart extents (u, v) per surface
EXTENT = {
    S_CONCH: (2 * R, R),
    S_HEMI: (math.pi * R, FLOOR_Y),                  # u = HEMI_U0 + R*phi, v = world y (tiles only below SPRING_Y)
    S_ARCH: (XR - XL, FLOOR_Y - CEIL_Y),
    S_SOFFIT: (math.pi * R, ARCH_Z),
    S_JAMBL: (ARCH_Z, FLOOR_Y - SPRING_Y),
    S_JAMBR: (ARCH_Z, FLOOR_Y - SPRING_Y),
    S_LEFT: (Z_FAR - ARCH_Z, FLOOR_Y - CEIL_Y),
    S_RIGHT: (Z_FAR - ARCH_Z, FLOOR_Y - CEIL_Y),
    S_FLOOR: (XR - XL, Z_FAR + R),
    S_CEIL: (XR - XL, Z_FAR - ARCH_Z),
}
HEMI_U0 = math.pi * R / 2       # vx.mosaic v1 Hemicycle(R, cx, cz, u0) default: u runs 0..pi R
HEMI_H = FLOOR_Y - SPRING_Y     # hemicycle wall height


# ============================================================ conch / hemicycle mappings
def conch_to_world(u, v):
    """conch chart -> world (x, y, z) (numpy broadcast)"""
    a = np.asarray(u, np.float64) - AX
    b = np.asarray(v, np.float64) - SPRING_Y
    t = 2 * R * R / (a * a + b * b + R * R)
    return AX + t * a, SPRING_Y + t * b, R * (1 - t)


def conch_k(u, v):
    a = np.asarray(u, np.float64) - AX
    b = np.asarray(v, np.float64) - SPRING_Y
    return 2.0 / (1.0 + (a * a + b * b) / (R * R))


def world_to_conch(x, y, z):
    """sphere point -> chart (stereographic from the front pole (AX, SPRING_Y, R))"""
    dx, dy, dz = np.asarray(x) - AX, np.asarray(y) - SPRING_Y, np.asarray(z)
    s = R / (R - dz)
    return AX + dx * s, SPRING_Y + dy * s


def conch_dir(phi, theta):
    """chart point of azimuth phi (0 = back, <0 left) and elevation theta (radians) on the conch"""
    x = AX + R * math.cos(theta) * math.sin(phi)
    y = SPRING_Y - R * math.sin(theta)
    z = -R * math.cos(theta) * math.cos(phi)
    u, v = world_to_conch(x, y, z)
    return float(u), float(v)


# ============================================================ camera
class Cam:
    """pinhole: eye, target, up = -Y, fov = HORIZONTAL degrees (vx.mosaic.Camera convention), screen 1920x1080
    design units with the optical axis at `at`."""

    def __init__(self, eye, target, fov=50.0, roll=0.0, at=(960.0, 540.0)):
        self.eye = np.asarray(eye, np.float64)
        self.target = np.asarray(target, np.float64)
        self.fov = float(fov)
        self.roll = float(roll)
        self.at = (float(at[0]), float(at[1]))
        f = self.target - self.eye
        f /= np.linalg.norm(f)
        up = np.array([math.sin(roll), -math.cos(roll), 0.0])
        r = np.cross(up, f)            # screen right (X right, Y down, Z toward viewer: right = up x fwd)
        r /= np.linalg.norm(r)
        d = np.cross(r, f)             # screen down = right x fwd
        self.f, self.r, self.d = f, r, d
        self.focal = 960.0 / math.tan(math.radians(self.fov) / 2)   # design px

    def project(self, P):
        """world (..., 3) -> screen design (..., 2), depth (...)"""
        P = np.asarray(P, np.float64) - self.eye
        z = P @ self.f
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        x = self.at[0] + self.focal * (P @ self.r) / zs
        y = self.at[1] + self.focal * (P @ self.d) / zs
        return np.stack([x, y], -1), z

    def scale_at(self, P):
        """design px per world unit at world point P (frontal)"""
        z = (np.asarray(P, np.float64) - self.eye) @ self.f
        return self.focal / z

    def mosaic(self):
        """the same camera as a vx.mosaic v1 Camera"""
        from vx import mosaic as mz
        return mz.Camera(eye=tuple(self.eye), target=tuple(self.target), up=(math.sin(self.roll), -math.cos(self.roll), 0.0),
                         fov=self.fov, at=self.at)


# ============================================================ ray caster (numba): per pixel surface id + chart uv + footprint
@njit(cache=True, fastmath=True)
def _cast(w, h, s, eye, fx, rx, dx_, focal, atx, aty, R_, AX_, SY, FY, CY, AZ, XL_, XR_, ZF,
          sid, uu, vv, fp):
    for py in range(h):
        for px in range(w):
            sx = (px + 0.5) / s - atx
            sy = (py + 0.5) / s - aty
            Dx = fx[0] * focal + rx[0] * sx + dx_[0] * sy
            Dy = fx[1] * focal + rx[1] * sx + dx_[1] * sy
            Dz = fx[2] * focal + rx[2] * sx + dx_[2] * sy
            Ln = math.sqrt(Dx * Dx + Dy * Dy + Dz * Dz)
            Dx /= Ln
            Dy /= Ln
            Dz /= Ln
            cosc = focal / math.sqrt(focal * focal + sx * sx + sy * sy)   # cos of angle to the optical axis
            best = 1e30
            bs = 0
            bu = 0.0
            bv = 0.0
            bcos = 1.0
            ox = eye[0]
            oy = eye[1]
            oz = eye[2]
            # --- conch: sphere centre (AX, SY, 0), keep z <= 0 and y <= SY
            cx = ox - AX_
            cy = oy - SY
            cz = oz
            B = cx * Dx + cy * Dy + cz * Dz
            Cq = cx * cx + cy * cy + cz * cz - R_ * R_
            disc = B * B - Cq
            if disc > 0.0:
                sq = math.sqrt(disc)
                for k in range(2):
                    t = -B - sq if k == 0 else -B + sq
                    if t > 1e-3 and t < best:
                        X = ox + t * Dx
                        Y = oy + t * Dy
                        Z = oz + t * Dz
                        if Z <= 0.0 and Y <= SY:
                            best = t
                            bs = 1
                            sc = R_ / (R_ - Z)
                            bu = AX_ + (X - AX_) * sc
                            bv = SY + (Y - SY) * sc
                            nx = (X - AX_) / R_
                            ny = (Y - SY) / R_
                            nz = Z / R_
                            ki = 2.0 / (1.0 + ((bu - AX_) ** 2 + (bv - SY) ** 2) / (R_ * R_))
                            bcos = abs(nx * Dx + ny * Dy + nz * Dz) / ki
            # --- hemicycle: cylinder axis (AX, *, 0) radius R, z <= 0, SY <= y <= FY
            a2 = Dx * Dx + Dz * Dz
            if a2 > 1e-12:
                b2 = (ox - AX_) * Dx + oz * Dz
                c2 = (ox - AX_) ** 2 + oz * oz - R_ * R_
                disc = b2 * b2 - a2 * c2
                if disc > 0.0:
                    sq = math.sqrt(disc)
                    for k in range(2):
                        t = (-b2 - sq) / a2 if k == 0 else (-b2 + sq) / a2
                        if t > 1e-3 and t < best:
                            X = ox + t * Dx
                            Y = oy + t * Dy
                            Z = oz + t * Dz
                            if Z <= 0.0 and Y >= SY and Y <= FY:
                                best = t
                                bs = 2
                                phi = math.atan2(X - AX_, -Z)
                                bu = 0.5 * math.pi * R_ + R_ * phi
                                bv = Y
                                bcos = abs(((X - AX_) * Dx + Z * Dz) / R_)
            # --- arch front plane z = AZ (outside the opening)
            if abs(Dz) > 1e-12:
                t = (AZ - oz) / Dz
                if t > 1e-3 and t < best:
                    X = ox + t * Dx
                    Y = oy + t * Dy
                    if X >= XL_ and X <= XR_ and Y >= CY and Y <= FY:
                        ddx = X - AX_
                        ddy = Y - SY
                        inside = (abs(ddx) <= R_ and ddy >= 0.0) or (ddx * ddx + ddy * ddy <= R_ * R_)
                        if not inside:
                            best = t
                            bs = 3
                            bu = X - XL_
                            bv = Y - CY
                            bcos = abs(Dz)
            # --- soffit: half cylinder about the horizontal axis (AX, SY) for 0 <= z <= AZ, above the springing
            a3 = Dx * Dx + Dy * Dy
            if a3 > 1e-12:
                b3 = (ox - AX_) * Dx + (oy - SY) * Dy
                c3 = (ox - AX_) ** 2 + (oy - SY) ** 2 - R_ * R_
                disc = b3 * b3 - a3 * c3
                if disc > 0.0:
                    sq = math.sqrt(disc)
                    for k in range(2):
                        t = (-b3 - sq) / a3 if k == 0 else (-b3 + sq) / a3
                        if t > 1e-3 and t < best:
                            X = ox + t * Dx
                            Y = oy + t * Dy
                            Z = oz + t * Dz
                            if Z >= 0.0 and Z <= AZ and Y <= SY:
                                best = t
                                bs = 4
                                ang = math.atan2(-(Y - SY), -(X - AX_))   # 0 at the left springing, pi at the right
                                bu = R_ * ang                              # = vx.mosaic Vault(u0=pi R/2) chart
                                bv = AZ - Z
                                bcos = abs(((X - AX_) * Dx + (Y - SY) * Dy) / R_)
            # --- jambs x = AX -/+ R, 0 <= z <= AZ, SY <= y <= FY
            if abs(Dx) > 1e-12:
                for k in range(2):
                    xj = AX_ - R_ if k == 0 else AX_ + R_
                    t = (xj - ox) / Dx
                    if t > 1e-3 and t < best:
                        Y = oy + t * Dy
                        Z = oz + t * Dz
                        if Z >= 0.0 and Z <= AZ and Y >= SY and Y <= FY:
                            best = t
                            bs = 9 + k                                 # Plane charts of the two jambs
                            bu = AZ - Z if k == 0 else Z
                            bv = Y - SY
                            bcos = abs(Dx)
            # --- side walls
            if abs(Dx) > 1e-12:
                for k in range(2):
                    xw = XL_ if k == 0 else XR_
                    t = (xw - ox) / Dx
                    if t > 1e-3 and t < best:
                        Y = oy + t * Dy
                        Z = oz + t * Dz
                        if Z >= AZ and Z <= ZF and Y >= CY and Y <= FY:
                            best = t
                            bs = 5 + k
                            bu = (ZF - Z) if k == 0 else (Z - AZ)
                            bv = Y - CY
                            bcos = abs(Dx)
            # --- floor / ceiling
            if abs(Dy) > 1e-12:
                for k in range(2):
                    yp = FY if k == 0 else CY
                    t = (yp - oy) / Dy
                    if t > 1e-3 and t < best:
                        X = ox + t * Dx
                        Z = oz + t * Dz
                        if X >= XL_ and X <= XR_ and Z <= ZF:
                            ok = False
                            if k == 0:
                                if Z >= AZ:
                                    ok = True
                                elif Z >= 0.0:
                                    ok = abs(X - AX_) <= R_
                                else:
                                    ok = (X - AX_) ** 2 + Z * Z <= R_ * R_
                            else:
                                ok = Z >= AZ
                            if ok:
                                best = t
                                bs = 7 + k
                                bu = X - XL_ if k == 0 else XR_ - X
                                bv = Z + R_ if k == 0 else Z - AZ
                                bcos = abs(Dy)
            sid[py, px] = bs
            uu[py, px] = bu
            vv[py, px] = bv
            # chart units per screen pixel: distance * pixel angle / cos(incidence)
            if bs > 0:
                fp[py, px] = best / (focal * s) / cosc / max(bcos, 0.05)
            else:
                fp[py, px] = 0.0


def cast(cam, w, h, s):
    """per pixel (sid int8, u, v float32 chart coords, footprint float32 chart units per pixel)"""
    sid = np.zeros((h, w), np.int8)
    uu = np.zeros((h, w), np.float32)
    vv = np.zeros((h, w), np.float32)
    fp = np.zeros((h, w), np.float32)
    _cast(w, h, float(s), cam.eye, cam.f, cam.r, cam.d, cam.focal, cam.at[0], cam.at[1], R, AX, SPRING_Y, FLOOR_Y,
          CEIL_Y, ARCH_Z, XL, XR, Z_FAR, sid, uu, vv, fp)
    return sid, uu, vv, fp


# ============================================================ texture sampling with a mip pyramid
class Tex:
    """a chart image (float32 rgb or rgba) + mip pyramid; texel = chart units per pixel at level 0."""

    def __init__(self, img, texel, u0=0.0, v0=0.0):
        import cv2
        self.lv = [np.ascontiguousarray(img)]           # float32 or uint8 (uint8 samples come back / 255)
        while min(self.lv[-1].shape[:2]) > 8:
            self.lv.append(cv2.pyrDown(self.lv[-1]))
        self.texel = float(texel)
        self.u0, self.v0 = float(u0), float(v0)

    def sample(self, u, v, fp, lod_bias=0.0):
        """u, v, fp: 1-D chart coords / footprints -> (M, C) trilinear"""
        import cv2
        lod = np.log2(np.maximum(fp, 1e-6) / self.texel) + lod_bias
        lod = np.clip(lod, 0, len(self.lv) - 1)
        l0 = np.floor(lod).astype(np.int32)
        fr = (lod - l0).astype(np.float32)
        ch = self.lv[0].shape[2] if self.lv[0].ndim == 3 else 1
        out = np.zeros((len(u), ch), np.float32)
        u = np.asarray(u, np.float32)
        v = np.asarray(v, np.float32)
        for L in np.unique(l0):
            m = l0 == L
            for k, wt in ((0, 1 - fr[m]), (1, fr[m])):
                LL = min(L + k, len(self.lv) - 1)
                sc = 1.0 / (self.texel * 2 ** LL)
                smp = remap_points(self.lv[LL], (u[m] - self.u0) * sc - 0.5, (v[m] - self.v0) * sc - 0.5)
                out[m] += smp.reshape(-1, ch) * wt[:, None]
        if self.lv[0].dtype == np.uint8:
            out *= 1.0 / 255.0
        return out


def remap_points(img, mx, my, interp=None):
    """bilinear lookup of arbitrary pixel coords (1-D) in img -> (M, C); rows of 4096 (cv2 size limits)"""
    import cv2
    M = len(mx)
    ch = img.shape[2] if img.ndim == 3 else 1
    if M == 0:
        return np.zeros((0, ch), np.float32)
    RW = 4096
    rows = (M + RW - 1) // RW
    px = np.zeros(rows * RW, np.float32)
    py = np.zeros(rows * RW, np.float32)
    px[:M] = mx
    py[:M] = my
    smp = cv2.remap(img, px.reshape(rows, RW), py.reshape(rows, RW), interp if interp is not None else cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return smp.reshape(rows * RW, ch)[:M]

"""t07: a small silhouette-figure system for the montage styles (Iwo Jima marines, Seurat strollers, woodcut
pilgrims, astronauts...). Anatomical width profiles along a 2D skeleton, smooth outlines, optional rim/shade.
Owner: T07Raised.  All coordinates in cell content units."""
import math

from scenes.t07_cells import ik2, curve, grey

PROF = {
    "thigh": [(0.0, 0.11), (0.35, 0.1), (0.8, 0.07), (1.0, 0.062)],
    "shin": [(0.0, 0.062), (0.25, 0.07), (0.7, 0.048), (1.0, 0.04)],
    "uarm": [(0.0, 0.07), (0.3, 0.066), (1.0, 0.05)],
    "farm": [(0.0, 0.052), (0.3, 0.054), (1.0, 0.038)],
    "neck": [(0.0, 0.062), (1.0, 0.056)],
}
# the calf/biceps bulge sits on one side: (+1 = left of the bone direction)
BULGE = {"shin": -1, "farm": 1, "uarm": 1, "thigh": 1}


def _w(prof, t):
    for (t0, w0), (t1, w1) in zip(prof, prof[1:]):
        if t <= t1:
            u = (t - t0) / max(t1 - t0, 1e-6)
            return w0 + (w1 - w0) * u
    return prof[-1][1]


def bone(a, b, prof, H, k=1.0, bulge=0, n=6):
    """closed outline of a limb (list of points)"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1e-6
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    left, right = [], []
    for i in range(n + 1):
        t = i / n
        w = _w(prof, t) * H * k * 0.5
        bl = 1.0 + (0.18 * math.sin(math.pi * min(1, t * 1.6)) if bulge else 0.0)
        wl = w * (bl if bulge > 0 else 1.0)
        wr = w * (bl if bulge < 0 else 1.0)
        px, py = a[0] + dx * t, a[1] + dy * t
        left.append((px + nx * wl, py + ny * wl))
        right.append((px - nx * wr, py - ny * wr))
    # round caps
    cap_b = [(b[0] + ux * _w(prof, 1) * H * k * 0.45, b[1] + uy * _w(prof, 1) * H * k * 0.45)]
    cap_a = [(a[0] - ux * _w(prof, 0) * H * k * 0.45, a[1] - uy * _w(prof, 0) * H * k * 0.45)]
    return left + cap_b + right[::-1] + cap_a


def fill_pts(ctx, pts):
    curve(ctx, pts, close=True, tension=0.8)
    ctx.fill()


def pose(hip, H, lean=0.0, facing=1, hand_f=None, hand_b=None, foot_f=None, foot_b=None, tilt=0.0, arm_bend=1,
         knee_bend=1):
    f = facing
    ts = 0.29 * H
    ua, fa, th, sh = 0.175 * H, 0.16 * H, 0.245 * H, 0.245 * H
    hr = 0.07 * H
    neck = (hip[0] + f * math.sin(lean) * ts, hip[1] - math.cos(lean) * ts)
    headc = (neck[0] + f * math.sin(lean + tilt) * hr * 1.2, neck[1] - math.cos(lean + tilt) * hr * 1.2)
    shf = (neck[0] + f * 0.028 * H, neck[1] + 0.035 * H)
    shb = (neck[0] - f * 0.028 * H, neck[1] + 0.035 * H)
    hpf = (hip[0] + f * 0.02 * H, hip[1])
    hpb = (hip[0] - f * 0.02 * H, hip[1])
    hand_f = hand_f or (shf[0] + f * 0.03 * H, shf[1] + ua + fa - 0.01 * H)
    hand_b = hand_b or (shb[0] - f * 0.02 * H, shb[1] + ua + fa - 0.01 * H)
    foot_f = foot_f or (hpf[0] + f * 0.05 * H, hip[1] + th + sh)
    foot_b = foot_b or (hpb[0] - f * 0.05 * H, hip[1] + th + sh)
    elf = ik2(shf, hand_f, ua, fa, bend=-f * arm_bend)
    elb = ik2(shb, hand_b, ua, fa, bend=-f * arm_bend)
    knf = ik2(hpf, foot_f, th, sh, bend=f * knee_bend)
    knb = ik2(hpb, foot_b, th, sh, bend=f * knee_bend)
    return dict(hip=hip, neck=neck, head=headc, hr=hr, shf=shf, shb=shb, elf=elf, elb=elb, haf=hand_f, hab=hand_b,
                hpf=hpf, hpb=hpb, knf=knf, knb=knb, ftf=foot_f, ftb=foot_b, H=H, f=f, lean=lean, tilt=tilt)


def torso(J, k=1.0, chest=1.0, belly=0.0):
    """side/3-4 view torso outline: back line, chest forward (facing), pelvis"""
    H, f = J["H"] * k, J["f"]
    hip, neck = J["hip"], J["neck"]
    dx, dy = neck[0] - hip[0], neck[1] - hip[1]
    L = math.hypot(dx, dy) or 1e-6
    ux, uy = dx / L, dy / L          # up the spine
    fx, fy = f * -uy, f * ux         # toward the chest

    def at(t, off):
        return (hip[0] + dx * t + fx * off * H, hip[1] + dy * t + fy * off * H)
    front = [at(-0.08, 0.075), at(0.2, 0.066 + belly * 0.03), at(0.5, 0.07 * chest), at(0.78, 0.085 * chest),
             at(0.98, 0.05)]
    back = [at(1.02, -0.05), at(0.8, -0.075), at(0.5, -0.06), at(0.2, -0.07), at(-0.08, -0.08)]
    return front + back


def seg(ctx, a, b, prof, H, k=1.0, bulge=0, n=8):
    """fill a limb: width profile + muscle bulge on one side, round ends (no knobs at the joints)"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1e-6
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    left, right = [], []
    for i in range(n + 1):
        t = i / n
        w = _w(prof, t) * H * k * 0.5
        bl = 1.0 + 0.2 * math.sin(math.pi * min(1.0, t * 1.5)) if bulge else 1.0
        wl = w * (bl if bulge > 0 else 1.0)
        wr = w * (bl if bulge < 0 else 1.0)
        px, py = a[0] + dx * t, a[1] + dy * t
        left.append((px + nx * wl, py + ny * wl))
        right.append((px - nx * wr, py - ny * wr))
    ang = math.atan2(dy, dx)
    ctx.new_sub_path()
    ctx.move_to(*left[0])
    for p in left[1:]:
        ctx.line_to(*p)
    ctx.arc_negative(b[0], b[1], _w(prof, 1) * H * k * 0.5, ang + math.pi / 2, ang - math.pi / 2)
    for p in right[::-1]:
        ctx.line_to(*p)
    ctx.arc_negative(a[0], a[1], _w(prof, 0) * H * k * 0.5, ang - math.pi / 2, ang - 3 * math.pi / 2)
    ctx.close_path()
    ctx.fill()


def draw(ctx, J, g_back=0.2, g_front=0.25, k=1.0, head=True, chest=1.0, belly=0.0, skip=()):
    """draw the figure as a silhouette (back limbs in g_back, the rest g_front)"""
    H = J["H"]
    f = J["f"]
    grey(ctx, g_back)
    if "legb" not in skip:
        seg(ctx, J["hpb"], J["knb"], PROF["thigh"], H, k, f)
        seg(ctx, J["knb"], J["ftb"], PROF["shin"], H, k, -f)
        foot(ctx, J["ftb"], J["knb"], H * k, f)
    if "armb" not in skip:
        seg(ctx, J["shb"], J["elb"], PROF["uarm"], H, k, 1)
        seg(ctx, J["elb"], J["hab"], PROF["farm"], H, k, 1)
        hand(ctx, J["hab"], J["elb"], H * k)
    grey(ctx, g_front)
    fill_pts(ctx, torso(J, k, chest, belly))
    for s in ("shf", "shb"):
        ctx.new_sub_path(); ctx.arc(J[s][0], J[s][1], 0.046 * H * k, 0, 2 * math.pi); ctx.fill()
    if head:
        seg(ctx, J["neck"], J["head"], PROF["neck"], H, k * 1.15)
        ctx.save()
        ctx.translate(*J["head"])
        ctx.rotate(J["lean"] + J["tilt"])
        ctx.scale(J["f"], 1)
        r = J["hr"] * k
        curve(ctx, [(-0.95 * r, -0.1 * r), (-0.75 * r, -0.9 * r), (0.15 * r, -1.1 * r), (0.85 * r, -0.6 * r),
                    (1.0 * r, 0.1 * r), (1.1 * r, 0.35 * r), (0.8 * r, 0.95 * r), (0.2 * r, 1.08 * r),
                    (-0.55 * r, 0.75 * r)], close=True, tension=0.9)
        ctx.fill()
        ctx.restore()
    if "legf" not in skip:
        seg(ctx, J["hpf"], J["knf"], PROF["thigh"], H, k, f)
        seg(ctx, J["knf"], J["ftf"], PROF["shin"], H, k, -f)
        foot(ctx, J["ftf"], J["knf"], H * k, f)
    if "armf" not in skip:
        seg(ctx, J["shf"], J["elf"], PROF["uarm"], H, k, 1)
        seg(ctx, J["elf"], J["haf"], PROF["farm"], H, k, 1)
        hand(ctx, J["haf"], J["elf"], H * k)


def foot(ctx, ft, kn, H, f):
    x, y = ft
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(f, 1)
    curve(ctx, [(-0.045 * H, -0.06 * H), (0.03 * H, -0.055 * H), (0.1 * H, -0.02 * H), (0.105 * H, 0.008 * H),
                (-0.05 * H, 0.008 * H)], close=True, tension=0.5)
    ctx.fill()
    ctx.restore()


def hand(ctx, ha, el, H):
    dx, dy = ha[0] - el[0], ha[1] - el[1]
    ctx.save()
    ctx.translate(*ha)
    ctx.rotate(math.atan2(dy, dx))
    ctx.scale(1.0, 0.82)
    ctx.new_sub_path()
    ctx.arc(0.012 * H, 0, 0.034 * H, 0, 2 * math.pi)
    ctx.restore()
    ctx.fill()


def outline(ctx, J, lw, g=0.0, k=1.0):
    """line-art version (woodcut, blueprint, stitch): stroke every part outline"""
    H = J["H"]
    grey(ctx, g)
    ctx.set_line_width(lw)
    for a, b, p, bl in (("hpb", "knb", "thigh", 1), ("knb", "ftb", "shin", -1), ("shb", "elb", "uarm", 1),
                        ("elb", "hab", "farm", 1), ("hpf", "knf", "thigh", 1), ("knf", "ftf", "shin", -1),
                        ("shf", "elf", "uarm", 1), ("elf", "haf", "farm", 1)):
        curve(ctx, bone(J[a], J[b], PROF[p], H, k, bl * J["f"]), close=True, tension=0.8)
        ctx.stroke()
    curve(ctx, torso(J, k), close=True, tension=0.8)
    ctx.stroke()
    ctx.arc(J["head"][0], J["head"][1], J["hr"] * k, 0, 2 * math.pi)
    ctx.stroke()

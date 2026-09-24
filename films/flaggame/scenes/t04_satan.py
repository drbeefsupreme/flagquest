"""t04 — the Devil of page 25 panel 3, drawn for the engraving: a Chick-tract Satan (horns, bat ears, furry
cheeks, slit pupils, a Cheshire grin full of fangs) in a black high-collared robe, a lyre in his claws.

draw_satan(ctx, x, y, s, p, t) -> anchors      (x, y) = head centre, s = scale (1: head ~330 px wide)
p (all optional): brow -1 (worried) .. 1 (evil), lid 0 closed .. 1 open, grin 0 (grimace) .. 1 (Cheshire),
    jaw 0..1 (lip-sync), gaze (gx, gy), possessed 0..1 (eyes roll white, veins, the scream), sweat 0..1,
    tilt (rad), pluck 0..1 (the plucking finger), snap None|seconds since the strings snapped
Everything is grey tone + pure-black contours (vx.ink.engrave hatches the tone; black stays solid ink).
"""
import math

import cairo

from vx import lin_grad, rad_grad, set_color, smooth, ellipse, clamp, lerp, hash01

K = (0.0, 0.0, 0.0)


def G(v):
    return (v, v, v)


def _mirror(pts):
    return [(-x, y) for x, y in reversed(pts)]


def _stroke(ctx, w, c=K):
    set_color(ctx, c)
    ctx.set_line_width(w)
    ctx.stroke()


def _robe(ctx, p, t):
    # black robe, shoulders and a tall flared collar (lighter lining) behind the head
    ctx.save()
    ctx.move_to(-360, 1100)
    ctx.curve_to(-380, 520, -330, 330, -170, 250)
    ctx.line_to(170, 250)
    ctx.curve_to(330, 330, 380, 520, 360, 1100)
    ctx.close_path()
    set_color(ctx, G(0.02))
    ctx.fill()
    for side in (-1, 1):
        ctx.move_to(side * 90, 250)
        ctx.curve_to(side * 180, 120, side * 250, -40, side * 300, -150)
        ctx.curve_to(side * 330, -40, side * 330, 120, side * 250, 330)
        ctx.close_path()
        ctx.set_source(lin_grad(side * 90, 0, side * 320, 0, [(0, G(0.55 if side < 0 else 0.28)),
                                                            (1, G(0.12))]))
        ctx.fill_preserve()
        _stroke(ctx, 5)
    ctx.arc(0, 330, 26, 0, 2 * math.pi)
    ctx.set_source(rad_grad(-8, 322, 2, 26, [(0, G(0.98)), (1, G(0.35))]))
    ctx.fill_preserve()
    _stroke(ctx, 4)
    ctx.restore()


def _horns(ctx, p, t):
    """thick ram/goat horns sweeping up and out, ringed"""
    for side in (-1, 1):
        ctx.save()
        ctx.scale(side, 1)
        ctx.move_to(40, -150)
        ctx.curve_to(90, -210, 190, -250, 235, -320)
        ctx.curve_to(250, -345, 250, -375, 238, -395)
        ctx.curve_to(230, -360, 215, -335, 190, -318)
        ctx.curve_to(150, -290, 110, -270, 100, -230)
        ctx.curve_to(100, -200, 110, -170, 125, -140)
        ctx.close_path()
        lit = side < 0
        ctx.set_source(lin_grad(40, -140, 240, -390, [(0, G(0.85 if lit else 0.5)), (0.6, G(0.6 if lit else 0.3)),
                                                       (1, G(0.3 if lit else 0.12))]))
        ctx.fill_preserve()
        _stroke(ctx, 6)
        ctx.set_line_width(3.5)
        for k in range(8):
            u = (k + 0.5) / 8
            ax, ay = lerp(60, 225, u), lerp(-160, -335, u) + 20 * math.sin(u * 3)
            bx, by = lerp(118, 205, u), lerp(-150, -300, u)
            ctx.move_to(ax, ay)
            ctx.curve_to(ax + 12, ay + 14, bx - 10, by - 6, bx, by)
            set_color(ctx, K)
            ctx.stroke()
        ctx.restore()


def _ears(ctx, p, t):
    for side in (-1, 1):
        ctx.save()
        ctx.scale(side, 1)
        ctx.move_to(140, -80)
        ctx.curve_to(185, -120, 220, -150, 250, -170)
        ctx.curve_to(235, -115, 205, -50, 148, -5)
        ctx.close_path()
        ctx.set_source(lin_grad(140, 0, 250, 0, [(0, G(0.55 if side < 0 else 0.35)), (1, G(0.2))]))
        ctx.fill_preserve()
        _stroke(ctx, 5)
        ctx.move_to(160, -70)
        ctx.curve_to(190, -100, 212, -122, 230, -140)
        ctx.curve_to(218, -100, 196, -58, 160, -28)
        ctx.close_path()
        set_color(ctx, G(0.08))
        ctx.fill()
        ctx.restore()


HEAD_R = [(0, -168), (70, -160), (120, -125), (145, -60), (150, -5), (135, 60), (105, 115), (58, 158), (0, 178)]


def _head_path(ctx):
    pts = HEAD_R + _mirror(HEAD_R[1:-1])
    smooth(ctx, pts, close=True, tension=0.5)


def _fur(ctx, p, t):
    """furry cheek tufts: jagged spikes along both cheeks"""
    for side in (-1, 1):
        ctx.save()
        ctx.scale(side, 1)
        ctx.move_to(145, -40)
        for k in range(8):
            u = k / 7
            bx = 150 - 40 * u
            by = -30 + 150 * u
            ctx.line_to(bx + 48 + 10 * math.sin(k * 2.1), by + 6)
            ctx.line_to(bx + 4, by + 18)
        ctx.line_to(60, 150)
        ctx.line_to(100, -30)
        ctx.close_path()
        ctx.set_source(lin_grad(90, 0, 200, 0, [(0, G(0.55 if side < 0 else 0.3)), (1, G(0.25 if side < 0 else 0.1))]))
        ctx.fill_preserve()
        _stroke(ctx, 4)
        ctx.restore()


def _eyes(ctx, p, t, anc):
    lid = clamp(p.get("lid", 0.7))
    brow = p.get("brow", 1.0)
    gx, gy = p.get("gaze", (0.0, 0.0))
    poss = p.get("possessed", 0.0)
    for side in (-1, 1):
        cx, cy = side * 64, -38
        ctx.save()
        ctx.translate(cx, cy)
        ctx.rotate(side * (-0.28 + 0.1 * (1 - brow)))
        # socket shadow
        ellipse(ctx, 0, -4, 58, 40)
        ctx.set_source(rad_grad(0, -4, 10, 60, [(0, G(0.25)), (1, G(0.25), 0.0)]))
        ctx.fill()
        hh = 26 * (0.25 + 0.75 * lid) * (1 + 0.5 * poss)
        # almond
        ctx.move_to(-44, 0)
        ctx.curve_to(-20, -hh * 1.3, 20, -hh * 1.3, 46, -4)
        ctx.curve_to(20, hh * 0.9, -20, hh * 0.9, -44, 0)
        ctx.close_path()
        set_color(ctx, G(1.0))
        ctx.fill_preserve()
        ctx.save()
        ctx.clip_preserve()
        if poss < 0.5:
            ix, iy = gx * 14 + side * 2, gy * 8
            ctx.new_path()
            ctx.arc(ix, iy, 19, 0, 2 * math.pi)
            ctx.set_source(rad_grad(ix - 5, iy - 5, 2, 20, [(0, G(0.75)), (1, G(0.35))]))
            ctx.fill()
            ellipse(ctx, ix, iy, 4.2 * (1 - 2 * poss), 17)
            set_color(ctx, K)
            ctx.fill()
            ctx.arc(ix - 7, iy - 7, 4, 0, 2 * math.pi)
            set_color(ctx, G(1.0))
            ctx.fill()
        else:                                         # rolled back: bloodshot veins on white
            ctx.new_path()
            ctx.set_line_width(1.6)
            for k in range(6):
                a = k / 6 * 2 * math.pi + 0.3
                ctx.move_to(math.cos(a) * 44, math.sin(a) * hh)
                ctx.line_to(math.cos(a) * 20 + 5 * math.sin(t * 40 + k), math.sin(a) * hh * 0.4)
            set_color(ctx, G(0.35))
            ctx.stroke()
        ctx.restore()
        ctx.new_path()
        ctx.move_to(-44, 0)
        ctx.curve_to(-20, -hh * 1.3, 20, -hh * 1.3, 46, -4)
        _stroke(ctx, 7)
        ctx.move_to(-44, 0)
        ctx.curve_to(-20, hh * 0.9, 20, hh * 0.9, 46, -4)
        _stroke(ctx, 3)
        ctx.restore()
        anc["eye_l" if side < 0 else "eye_r"] = (cx, cy)


def _brows(ctx, p, t):
    b = clamp(p.get("brow", 1.0), -1, 1)
    for side in (-1, 1):
        inner_y = lerp(-82, -58, (b + 1) / 2)
        outer_y = lerp(-80, -104, (b + 1) / 2)
        ctx.move_to(side * 18, inner_y + 4)
        ctx.curve_to(side * 50, inner_y - 16, side * 90, outer_y - 6, side * 128, outer_y - 10)
        ctx.curve_to(side * 95, outer_y + 12, side * 55, inner_y + 12, side * 18, inner_y + 4)
        ctx.close_path()
        set_color(ctx, K)
        ctx.fill()
    # frown furrows between the brows
    ctx.set_line_width(3.5)
    for side in (-1, 1):
        ctx.move_to(side * 10, -70)
        ctx.curve_to(side * 16, -95, side * 12, -115, side * 6, -130)
        set_color(ctx, K)
        ctx.stroke()


def _nose(ctx, p, t):
    ctx.move_to(-26, 8)
    ctx.curve_to(-18, -20, 18, -20, 26, 8)
    ctx.curve_to(12, 24, -12, 24, -26, 8)
    ctx.close_path()
    ctx.set_source(lin_grad(-26, 0, 26, 0, [(0, G(0.7)), (1, G(0.2))]))
    ctx.fill_preserve()
    _stroke(ctx, 4)
    for side in (-1, 1):
        ellipse(ctx, side * 11, 12, 6, 4, rot=side * 0.4)
        set_color(ctx, K)
        ctx.fill()
    # snarl lines from the nose to the mouth corners
    ctx.set_line_width(4)
    g = p.get("grin", 1.0)
    for side in (-1, 1):
        ctx.move_to(side * 30, 12)
        ctx.curve_to(side * 70, 20, side * (100 + 20 * g), 30, side * (112 + 18 * g), 30 - 10 * g)
        set_color(ctx, K)
        ctx.stroke()


def _mouth(ctx, p, t, anc):
    g = clamp(p.get("grin", 1.0))
    jaw = clamp(p.get("jaw", 0.0) + 0.9 * p.get("possessed", 0.0) * (0.7 + 0.3 * math.sin(t * 25)))
    wdt = lerp(78, 132, g)
    cy_corner = lerp(92, 28, g)                   # the Cheshire corners ride up the cheeks
    cy_mid = lerp(84, 88, g)
    open_ = 10 + jaw * 95
    # outer lips
    up = [(-wdt, cy_corner), (-wdt * 0.5, cy_mid - 10 + 8 * g), (0, cy_mid), (wdt * 0.5, cy_mid - 10 + 8 * g),
          (wdt, cy_corner)]
    lo = [(wdt, cy_corner), (wdt * 0.5, cy_mid + open_ * 0.8), (0, cy_mid + open_), (-wdt * 0.5, cy_mid + open_ * 0.8),
          (-wdt, cy_corner)]
    ctx.save()
    smooth(ctx, up + lo[1:-1], close=True)
    set_color(ctx, G(0.02))
    ctx.fill_preserve()
    ctx.clip()
    # teeth: fangs along both lips
    nt = 13
    for row, pts, dirn in ((0, up, 1), (1, lo, -1)):
        for k in range(nt):
            u = (k + 0.5) / nt
            x = lerp(-wdt, wdt, u)
            # y on the lip curve (piecewise quadratic through the 5 points)
            yy = _lip_y(pts, x)
            fang = 1.8 if k in (2, nt - 3) else 1.0
            L = (16 + 10 * math.sin(u * math.pi)) * fang
            hw = wdt / nt * 0.95
            ctx.move_to(x - hw, yy - 6 * dirn)
            ctx.line_to(x, yy + L * dirn)
            ctx.line_to(x + hw, yy - 6 * dirn)
            ctx.close_path()
            set_color(ctx, G(1.0))
            ctx.fill_preserve()
            set_color(ctx, K)
            ctx.set_line_width(2.2)
            ctx.stroke()
    # forked tongue when the jaw drops
    if jaw > 0.35:
        ty_ = cy_mid + open_ * 0.75
        ctx.move_to(-22, ty_)
        ctx.curve_to(-18, ty_ - 30 * jaw, 18, ty_ - 30 * jaw, 22, ty_)
        ctx.close_path()
        set_color(ctx, G(0.45))
        ctx.fill()
    ctx.restore()
    smooth(ctx, up + lo[1:-1], close=True)
    _stroke(ctx, 6)
    anc["mouth"] = (0.0, cy_mid + open_ * 0.5)


def _lip_y(pts, x):
    xs = [q[0] for q in pts]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        lo_, hi_ = min(a[0], b[0]), max(a[0], b[0])
        if lo_ <= x <= hi_:
            u = (x - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0
            u = u * u * (3 - 2 * u)
            return a[1] + (b[1] - a[1]) * u
    return pts[0][1]


def _goatee(ctx, p, t):
    jaw = clamp(p.get("jaw", 0.0) + 0.9 * p.get("possessed", 0.0))
    dy = jaw * 60
    ctx.move_to(-40, 160 + dy)
    ctx.curve_to(-20, 200 + dy, -8, 250 + dy, 4, 290 + dy)
    ctx.curve_to(10, 250 + dy, 28, 200 + dy, 40, 160 + dy)
    ctx.close_path()
    set_color(ctx, G(0.03))
    ctx.fill()


def _lyre(ctx, p, t, anc):
    """a classical lyre: soundbox, two curving arms, a crossbar, seven strings"""
    ctx.save()
    ctx.translate(-170, 480)
    ctx.rotate(-0.18)
    snap = p.get("snap")
    for side in (-1, 1):
        ctx.move_to(side * 60, 120)
        ctx.curve_to(side * 150, 40, side * 170, -120, side * 110, -250)
        ctx.curve_to(side * 95, -290, side * 120, -330, side * 150, -330)
        ctx.line_to(side * 140, -305)
        ctx.curve_to(side * 125, -300, side * 118, -275, side * 128, -250)
        ctx.curve_to(side * 190, -120, side * 170, 60, side * 80, 150)
        ctx.close_path()
        ctx.set_source(lin_grad(-150, 0, 150, 0, [(0, G(0.95)), (0.5, G(0.6)), (1, G(0.3))]))
        ctx.fill_preserve()
        _stroke(ctx, 4)
    ctx.rectangle(-125, -262, 250, 22)
    ctx.set_source(lin_grad(0, -262, 0, -240, [(0, G(0.9)), (1, G(0.35))]))
    ctx.fill_preserve()
    _stroke(ctx, 4)
    ctx.move_to(-95, 100)
    ctx.curve_to(-100, 190, 100, 190, 95, 100)
    ctx.close_path()
    ctx.set_source(lin_grad(-95, 0, 95, 0, [(0, G(0.85)), (1, G(0.25))]))
    ctx.fill_preserve()
    _stroke(ctx, 4)
    pl = p.get("pluck", 0.0)
    for k in range(7):
        x = -60 + k * 20
        ctx.set_line_width(2.0)
        set_color(ctx, K)
        if snap is not None and snap >= 0:
            # snapped: each string curls away from the crossbar, whipping
            s_ = min(1.0, snap * 5)
            ctx.move_to(x, 110)
            cx_ = x + (k - 3) * 30 * s_ + 25 * math.sin(snap * 30 + k) * math.exp(-snap * 4)
            ctx.curve_to(x, 20, cx_, -40 - 60 * (1 - s_), cx_ + (k - 3) * 10, -100 + 120 * s_)
            ctx.stroke()
            ctx.move_to(x, -240)
            ctx.curve_to(x, -220, x - (k - 3) * 25 * s_, -200 + 30 * s_, x - (k - 3) * 30 * s_, -170 + 60 * s_)
            ctx.stroke()
        else:
            vib = 5 * math.sin(t * 90 + k) * math.exp(-((t * 9 + k * 0.7) % 3.0)) if pl > 0 else 0.0
            ctx.move_to(x, 110)
            ctx.curve_to(x + vib, 30, x - vib, -150, x, -240)
            ctx.stroke()
    ctx.restore()
    anc["lyre"] = (-170, 380)


def _hands(ctx, p, t):
    """clawed hands on the lyre: one grips the arm, the other plucks"""
    pl = p.get("pluck", 0.0)
    for (hx, hy, rot, curl) in ((-300, 330, 0.5, 0.2), (-140 + 30 * pl, 470 - 20 * pl, -0.3, 0.5 + 0.4 * pl)):
        ctx.save()
        ctx.translate(hx, hy)
        ctx.rotate(rot)
        ctx.scale(1.45, 1.45)
        ellipse(ctx, 0, 0, 46, 36)
        ctx.set_source(lin_grad(-46, 0, 46, 0, [(0, G(0.8)), (1, G(0.35))]))
        ctx.fill_preserve()
        _stroke(ctx, 4)
        for f in range(4):
            a = -0.9 + f * 0.45
            L = 60 * (1 - 0.35 * curl)
            x0, y0 = math.cos(a) * 36, math.sin(a) * 30
            x1, y1 = x0 + math.cos(a + curl) * L, y0 + math.sin(a + curl) * L
            ctx.move_to(x0, y0)
            ctx.line_to(x1, y1)
            ctx.set_line_width(16)
            set_color(ctx, K)
            ctx.stroke()
            ctx.move_to(x0, y0)
            ctx.line_to(x1, y1)
            ctx.set_line_width(10)
            set_color(ctx, G(0.7))
            ctx.stroke()
            # claw
            ctx.move_to(x1 - 6, y1)
            ctx.line_to(x1 + math.cos(a + curl * 1.6) * 22, y1 + math.sin(a + curl * 1.6) * 22)
            ctx.line_to(x1 + 6, y1)
            ctx.close_path()
            set_color(ctx, K)
            ctx.fill()
        ctx.restore()


def _sweat(ctx, p, t):
    sw = p.get("sweat", 0.0)
    if sw <= 0.05:
        return
    for k in range(3):
        ph = (t * 0.9 + k * 0.37) % 1.0
        x = 150 - 40 * k
        y = -120 + 40 * k + ph * 90
        a = sw * (1 - ph)
        ctx.move_to(x, y - 22)
        ctx.curve_to(x + 12, y - 2, x + 12, y + 12, x, y + 12)
        ctx.curve_to(x - 12, y + 12, x - 12, y - 2, x, y - 22)
        set_color(ctx, G(1.0), a)
        ctx.fill_preserve()
        set_color(ctx, K, a)
        ctx.set_line_width(3)
        ctx.stroke()


def draw_satan(ctx, x, y, s, p, t):
    """the whole figure; returns anchors in the caller's coordinates"""
    anc = {}
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    _robe(ctx, p, t)
    tilt = p.get("tilt", 0.0)
    ctx.save()
    ctx.rotate(tilt)
    _horns(ctx, p, t)
    _ears(ctx, p, t)
    _fur(ctx, p, t)
    _head_path(ctx)
    ctx.set_source(lin_grad(-160, -60, 160, 60, [(0, G(0.86)), (0.45, G(0.62)), (1, G(0.28))]))
    ctx.fill_preserve()
    _stroke(ctx, 6)
    # modelling: brow ridge light, cheek hollows
    ctx.save()
    _head_path(ctx)
    ctx.clip()
    ellipse(ctx, 70, 60, 70, 50)
    ctx.set_source(rad_grad(70, 60, 5, 70, [(0, G(0.18), 0.55), (1, G(0.18), 0.0)]))
    ctx.fill()
    ellipse(ctx, -60, -120, 80, 30)
    ctx.set_source(rad_grad(-60, -120, 5, 80, [(0, G(1.0), 0.5), (1, G(1.0), 0.0)]))
    ctx.fill()
    ctx.restore()
    _eyes(ctx, p, t, anc)
    _brows(ctx, p, t)
    _nose(ctx, p, t)
    _mouth(ctx, p, t, anc)
    _goatee(ctx, p, t)
    _sweat(ctx, p, t)
    ctx.restore()
    _lyre(ctx, p, t, anc)
    _hands(ctx, p, t)
    ctx.restore()
    ct, st_ = math.cos(tilt), math.sin(tilt)
    out = {}
    for k, (ax, ay) in anc.items():
        if k == "lyre":
            out[k] = (x + ax * s, y + ay * s)
        else:
            out[k] = (x + (ax * ct - ay * st_) * s, y + (ax * st_ + ay * ct) * s)
    out["head"] = (x, y)
    return out

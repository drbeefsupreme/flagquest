"""t07: the Soviet-poster worker (hero of the opener, ref page 26 panel 3). Three-value poster: paper white,
flat grey tint, solid black; light from the upper right (from the Flag).  Owner: T07Raised.
Content units (cell height 1000)."""
import math

from scenes.t07_cells import grey, P, curve, limb, ik2

HI, LI, MID, DK, BLK = 0.985, 0.8, 0.6, 0.42, 0.06


def fill(ctx, pts, g, smooth=True, t=0.6):
    (curve if smooth else P)(ctx, pts, close=True, **({"tension": t} if smooth else {}))
    grey(ctx, g)
    ctx.fill()


def _profile(ctx):
    """closed outline of the head (skull + profile face), head-local units, facing right"""
    ctx.move_to(-4, -48)
    ctx.curve_to(18, -50, 30, -44, 32, -36)          # crown -> hairline
    ctx.curve_to(35, -28, 36, -18, 36.5, -12)         # forehead
    ctx.curve_to(37.5, -9, 37.5, -7, 35.5, -4.5)      # brow ridge -> nose root
    ctx.curve_to(38, 0, 43, 8, 47, 13)                # nose bridge
    ctx.curve_to(48.5, 15, 46, 17.5, 43, 17)          # tip
    ctx.curve_to(41, 16.8, 39.5, 17.5, 39, 18.5)      # under the nose
    ctx.curve_to(39.5, 20.5, 41.2, 21.8, 41, 23)      # upper lip
    ctx.curve_to(40, 23.8, 39, 24.2, 38.8, 24.6)      # mouth
    ctx.curve_to(40, 25.6, 40.2, 27.2, 38.8, 28.4)    # lower lip
    ctx.curve_to(37.6, 29.2, 37.4, 30.2, 38.2, 31.5)  # under-lip crease
    ctx.curve_to(40.8, 34, 41, 38.5, 37.5, 40.5)      # square chin
    ctx.curve_to(31, 43, 20, 42, 10, 38.5)            # under the chin
    ctx.curve_to(2, 36, -2, 33, -4, 24)               # strong jaw -> angle
    ctx.curve_to(-8, 18, -20, 16, -28, 10)            # nape
    ctx.curve_to(-38, -2, -38, -28, -28, -40)         # back of the skull
    ctx.curve_to(-22, -46, -14, -48, -4, -48)
    ctx.close_path()


def head(ctx, x, y, s, tilt, mouth=0.0):
    """worker's head in profile facing right. (x, y) = centre of the cranium, s = scale (head ~100*s tall)"""
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(tilt)
    ctx.scale(s, s)
    _profile(ctx)
    grey(ctx, LI)
    ctx.fill_preserve()
    ctx.save()
    ctx.clip()
    # the lit front of the face (light from the upper right, from the Flag)
    fill(ctx, [(20, -44), (40, -40), (52, 0), (52, 30), (44, 50), (26, 46), (30, 36), (27, 28), (24, 20), (20, 12),
               (26, 4), (24, -12), (18, -26)], HI, t=0.45)
    # cheekbone plane + the hollow under it
    fill(ctx, [(4, 2), (20, 0), (26, 8), (22, 16), (8, 16)], HI, t=0.5)
    fill(ctx, [(0, 12), (12, 16), (22, 22), (26, 32), (20, 38), (6, 34), (-2, 24)], MID, t=0.5)
    # under-jaw + back of the jaw in shadow
    fill(ctx, [(-6, 22), (2, 33), (12, 39), (28, 42), (40, 40), (40, 60), (-20, 60), (-30, 24)], BLK, smooth=False)
    ctx.restore()
    # the outline (brush weight)
    _profile(ctx)
    grey(ctx, BLK)
    ctx.set_line_width(1.6)
    ctx.stroke()
    # eye in profile: open wedge, iris looking up/forward; heavy upper lid; brow
    fill(ctx, [(24, -10), (30, -12.5), (34, -9.5), (31, -6), (25, -6.5)], BLK, t=0.4)
    fill(ctx, [(28.5, -9.8), (32.8, -10.2), (31, -7.4), (28.6, -7.6)], HI, smooth=False)
    fill(ctx, [(30.6, -10.3), (32.6, -10.3), (31.4, -8.2), (30.4, -8.4)], BLK, smooth=False)
    fill(ctx, [(18, -15), (28, -19.5), (37.5, -15), (36, -12), (28, -15.5), (19, -11.5)], BLK)
    # socket shadow behind the eye
    fill(ctx, [(16, -12), (24, -10), (25, -5), (19, -2)], MID)
    # nostril wing + nasolabial fold
    fill(ctx, [(37, 12), (42, 13.5), (42.5, 16.2), (38.5, 16.4), (36.5, 14.5)], BLK, t=0.5)
    ctx.set_line_width(1.4); grey(ctx, BLK)
    ctx.move_to(35.5, 15.5); ctx.curve_to(33.5, 20, 33.8, 25, 36, 28.5); ctx.stroke()
    # mouth line (opens with `mouth`)
    m = max(0.0, min(1.0, mouth))
    fill(ctx, [(33, 24.2), (39.2, 24.4), (38.6, 25.4 + 3.2 * m), (33.4, 25.4 + 2 * m)], BLK, smooth=False)
    # ear
    fill(ctx, [(-8, -6), (-1, -9), (3, -2), (2, 8), (-2, 12), (-8, 10), (-7, 2)], HI, t=0.5)
    ctx.set_line_width(1.5); grey(ctx, BLK)
    curve(ctx, [(-8, -6), (-1, -9), (3, -2), (2, 8), (-2, 12), (-8, 10), (-7, 2)], close=True, tension=0.5)
    ctx.stroke()
    fill(ctx, [(-5, -4), (-1, -5), (0.5, 1), (-2, 7), (-4.5, 4)], MID, t=0.5)
    # hair under the cap: short, black, sideburn
    fill(ctx, [(-36, -24), (-10, -28), (20, -28), (24, -24), (12, -22), (6, -16), (4, -8), (-2, -8), (-12, -14),
               (-28, -6), (-36, -10)], BLK, t=0.5)
    # the cap: soft crown, band, short peak (tilted back on the head)
    fill(ctx, [(-40, -16), (-42, -36), (-26, -54), (4, -58), (28, -52), (38, -38), (37, -26), (8, -24), (-28, -18)],
         BLK, t=0.55)
    fill(ctx, [(-4, -55), (14, -55), (27, -48), (30, -40), (16, -45), (0, -49)], MID, t=0.6)
    fill(ctx, [(30, -30), (56, -28), (60, -23), (53, -21), (33, -23)], BLK, t=0.4)
    fill(ctx, [(36, -28.5), (55, -27), (56, -25.6), (37, -27)], LI, smooth=False)
    ctx.restore()


def worker(ctx, fl, fu, tilt, mouth=0.0, breathe=0.0):
    """bust of the worker (lower left), fists at fl (lower) and fu (upper). content units, full-frame layout."""
    hx, hy = -170.0, 400.0 + breathe
    # back / far shoulder mass
    fill(ctx, [(-640, 1100), (-610, 800), (-520, 620), (-360, 548), (-220, 552), (-100, 610), (20, 720),
               (140, 900), (180, 1100)], MID, t=0.5)
    # the chest turned to the light
    fill(ctx, [(-230, 585), (-110, 612), (10, 710), (90, 820), (120, 1100), (-170, 1100), (-220, 860)], LI, t=0.5)
    # shadow side of the torso (solid) + fold lines
    fill(ctx, [(-640, 1100), (-610, 800), (-540, 650), (-470, 610), (-480, 800), (-430, 1100)], BLK, t=0.5)
    grey(ctx, BLK); ctx.set_line_width(9)
    for (a, b, c, d) in (((-70, 780), (-20, 840), (10, 920), (16, 1030)), ((-190, 820), (-150, 900), (-140, 980),
                                                                           (-150, 1090))):
        ctx.move_to(*a); ctx.curve_to(*b, *c, *d)
    ctx.stroke()
    # neck
    fill(ctx, [(-245, 460), (-115, 460), (-80, 560), (-100, 640), (-190, 680), (-262, 600)], LI, t=0.5)
    fill(ctx, [(-165, 470), (-112, 470), (-82, 560), (-100, 630), (-140, 600)], HI, t=0.5)
    fill(ctx, [(-248, 470), (-210, 480), (-212, 640), (-262, 600)], BLK, t=0.5)
    ctx.set_line_width(4); grey(ctx, BLK)
    ctx.move_to(-190, 500); ctx.curve_to(-170, 560, -140, 600, -120, 640); ctx.stroke()
    # open collar: white points, black V
    fill(ctx, [(-275, 555), (-205, 598), (-150, 700), (-238, 655)], HI, t=0.4)
    fill(ctx, [(-120, 588), (-58, 600), (-40, 652), (-100, 662), (-150, 700)], HI, t=0.4)
    fill(ctx, [(-205, 606), (-112, 606), (-150, 700)], BLK, smooth=False)
    ctx.set_line_width(4)
    for pts in (((-275, 555), (-205, 598), (-150, 700), (-238, 655)), ((-120, 588), (-58, 600), (-40, 652),
                                                                     (-100, 662), (-150, 700))):
        P(ctx, pts); ctx.stroke()
    # head
    head(ctx, hx, hy, 3.35, tilt, mouth=mouth)
    # far arm to the lower fist (mostly behind the chest)
    shb = (-420.0, 700.0)
    el_b = ik2(shb, fl, 320, 300, bend=-1)
    grey(ctx, MID)
    limb(ctx, shb, el_b, 180, 140); ctx.fill()
    limb(ctx, el_b, fl, 130, 110); ctx.fill()
    # near arm: sleeve rolled to the elbow, bare forearm up to the upper fist
    shf = (-40.0, 690.0)
    el_f = ik2(shf, fu, 250, 260, bend=1)
    grey(ctx, LI)
    limb(ctx, shf, el_f, 175, 150); ctx.fill()
    grey(ctx, BLK); ctx.set_line_width(7)
    limb(ctx, shf, el_f, 175, 150); ctx.stroke()
    ex, ey = el_f
    dx, dy = fu[0] - ex, fu[1] - ey
    L = math.hypot(dx, dy) or 1
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    c0 = (ex + ux * 12, ey + uy * 12)
    grey(ctx, HI)
    limb(ctx, c0, fu, 124, 96); ctx.fill()
    fill(ctx, [(c0[0] + nx * 62 + ux * 10, c0[1] + ny * 62 + uy * 10), (fu[0] + nx * 48 - ux * 30, fu[1] + ny * 48 - uy * 30),
               (fu[0] + nx * 20 - ux * 60, fu[1] + ny * 20 - uy * 60), (c0[0] + nx * 26 + ux * 60, c0[1] + ny * 26 + uy * 60)],
         MID, t=0.5)
    grey(ctx, BLK); ctx.set_line_width(6)
    limb(ctx, c0, fu, 124, 96); ctx.stroke()
    # the rolled cuff
    grey(ctx, LI)
    limb(ctx, (ex - ux * 16, ey - uy * 16), c0, 150, 140); ctx.fill()
    grey(ctx, BLK); ctx.set_line_width(6)
    limb(ctx, (ex - ux * 16, ey - uy * 16), c0, 150, 140); ctx.stroke()


def fist(ctx, x, y, ang, s=1.0, g=HI):
    """a fist gripping the pole (FRONT layer, over the pole): knuckles toward the viewer"""
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    ctx.scale(s, s)
    outline = [(-60, -40), (-14, -58), (40, -52), (60, -26), (60, 30), (34, 54), (-36, 52), (-64, 12)]
    fill(ctx, outline, g, t=0.5)
    # fingers wrapped around the pole: four rounded segments, shadow between
    for i, yy in enumerate((-34, -10, 14, 38)):
        fill(ctx, [(2, yy - 11), (50, yy - 13), (58, yy - 2), (52, yy + 9), (4, yy + 10)], HI, t=0.5)
        grey(ctx, BLK); ctx.set_line_width(4)
        ctx.move_to(4, yy + 11); ctx.curve_to(24, yy + 13, 44, yy + 12, 55, yy + 8); ctx.stroke()
    # thumb and the heel of the hand in shadow
    fill(ctx, [(-50, -36), (10, -52), (34, -42), (6, -28), (-34, -22)], LI, t=0.5)
    fill(ctx, [(-62, 8), (-40, -20), (-20, -10), (-16, 44), (-38, 50)], MID, t=0.5)
    grey(ctx, BLK); ctx.set_line_width(6)
    curve(ctx, outline, close=True, tension=0.5); ctx.stroke()
    ctx.restore()

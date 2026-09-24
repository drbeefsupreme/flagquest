"""t09 art: page 8's panels p2 (the reading shot: the page she holds) and p3 (THE NEXT DAY), the held
tract, the hands, the level adapter for the Print-Gallery texture.  All drawing in PAGE UNITS (t01_page).
Owner: T09PassItOn.

Drawer contract (t01_page.render_page + two t09 extras):
    world(ctx, T, rect) · flags(ctx, T, rect) · front(ctx, T, rect) · letters(ctx, fc, T, rect)
    holders(ctx, T, rect)   printed over the window (thumbs, the cover flap, Crow's hand) — p2 only
"""
import math

import cairo
import numpy as np

from vx import W, H, set_color, lin_grad, rad_grad
from vx import comic
from vx.ease import clamp, smoothstep, smootherstep, ease_in_out, ease_out, ease_in, ease_out_back
from vx.flag import Flag, draw_pole
from vx.chars import Character

from scenes import t01_page as P
from scenes import t09_droste as D
from scenes import t08_art as TA
from scenes.t09_ink import g, stroke, shape, poly, hatch, path, lerp, lerp2

FPS = 24
T_START = 181.1666667
T_PASS = 184.427
T_HAND_IN, T_HAND_OUT = 184.02, 184.95
T_OPEN0, T_OPEN1 = 184.74, 185.16
T_P1_PRINT, T_P2_PRINT, T_P3_PRINT = 184.20, 185.40, 191.60
PAGE_NO = 8

SPINE = D.WIN_X + D.WIN_W + 1.0                  # 521: between page 8 (window, left) and page 9 (right)
BOOK_Y0, BOOK_Y1 = D.WIN_Y, D.WIN_Y + D.WIN_H
PG9 = (SPINE + 1.0, D.WIN_Y, D.WIN_W, D.WIN_H)   # page 9 (right page)

P2 = P.PANELS["p2"]
P3 = P.PANELS["p3"]

# p2: the planted flag (Crow's), behind her, in the sunbeam
P2_FLAG_BASE = (396.0, 626.0)
P2_FLAG_POLE = 138.0
# p3: THE NEXT DAY
SUMMER_WALK_Y = 1062.0
SUMMER_H = 150.0
SUMMER_POLE = 168.0


def flap_angle(T):
    """cover flap angle phi: 0 = closed (cover on the right), pi = open (page 8 on the left)"""
    u = clamp((T - T_OPEN0) / (T_OPEN1 - T_OPEN0), 0, 1)
    return math.pi * smootherstep(0.0, 1.0, u)


def booklet_state(T):
    """'none' (still in Crow's hand / absent), 'closed', 'opening', 'open'"""
    if T < T_PASS:
        return "none"
    if T < T_OPEN0:
        return "closed"
    if T < T_OPEN1:
        return "opening"
    return "open"


def window_extent(T):
    """visible part of page 8 (the window) as x-range in PU; None when hidden"""
    st = booklet_state(T)
    if st in ("none", "closed"):
        return None
    phi = flap_angle(T)
    if phi <= math.pi / 2:
        return None
    w = D.WIN_W * abs(math.cos(phi))
    return (SPINE - 1.0 - w, SPINE - 1.0)


# ============================================================ small drawing pieces
def _thumb(ctx, x, y, ang, s=1.0, skin=0.99):
    """a thumb seen from the front over a page edge; (x, y) = thumb tip"""
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    ctx.scale(s, s)
    pts = [(-5.2, -2), (-4.8, 8), (-3.6, 17), (0, 22), (3.8, 17), (5.2, 8), (4.8, -3), (2.5, -8), (-2.8, -8)]
    shape(ctx, pts, g(skin), line=1.5)
    stroke(ctx, [(-2.6, 3.0), (0, 4.2), (2.6, 3.0)], 0.9, taper=(0.3, 0.3))       # nail
    stroke(ctx, [(-3.8, 13), (0.5, 14.5), (3.5, 13)], 0.8, taper=(0.3, 0.3))       # knuckle crease
    ctx.restore()


def _fingers_back(ctx, x, y, side, s=1.0, skin=0.97):
    """fingertips curling round the page edge from behind (seen at the page's outer edge)"""
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(side * s, s)
    for k, dy in enumerate((-12, -3, 6, 14)):
        shape(ctx, [(0, dy - 4), (5.5, dy - 3.5), (7.2, dy), (5.5, dy + 3.6), (0, dy + 4)], g(skin), line=1.3)
    ctx.restore()


def _sleeve_arm(ctx, pts, w, fill=0.38, cuff=True):
    """jacket sleeve along pts (shoulder -> wrist), faux-fur cuff at the end"""
    stroke(ctx, pts, w + 3.0, taper=(0.0, 0.0), color=(0, 0, 0))
    stroke(ctx, pts, w, taper=(0.0, 0.0), color=g(fill))
    if cuff:
        x, y = pts[-1]
        px, py = pts[-2]
        a = math.atan2(y - py, x - px)
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(a)
        fur = [(-6, -w * 0.62), (-1, -w * 0.72), (3, -w * 0.55), (5, -w * 0.2), (6, w * 0.2), (3, w * 0.58),
               (-1, w * 0.7), (-6, w * 0.6), (-8, 0)]
        shape(ctx, fur, g(1.0), line=1.6)
        for k in range(5):
            yy = -w * 0.5 + k * w * 0.25
            stroke(ctx, [(-5, yy), (-1, yy + 1.5), (3, yy)], 0.9)
        ctx.restore()


def draw_cover(ctx, x, y, w, h):
    """front cover of THE FLAG GAME? tract (ink only: the cover drawing's flag is omitted at this size)"""
    ctx.save()
    ctx.rectangle(x, y, w, h)
    set_color(ctx, g(0.0))                 # tiny page: its dark stock prints as solid ink
    ctx.fill()
    # title burst
    cx, cy = x + w * 0.5, y + h * 0.24
    pts = []
    for k in range(28):
        a = k / 28 * 2 * math.pi
        r = (0.42 if k % 2 == 0 else 0.30) * w
        pts.append((cx + math.cos(a) * r * 1.05, cy + math.sin(a) * r * 0.62))
    poly(ctx, pts, g(1.0), line=1.2)
    ctx.select_font_face(comic.FONT_LETTER, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    for txt, dy, sz in (("THE FLAG", -3.0, w * 0.13), ("GAME?", w * 0.12, w * 0.16)):
        ctx.set_font_size(sz)
        e = ctx.text_extents(txt)
        ctx.move_to(cx - e.x_advance / 2, cy + dy)
        ctx.set_source_rgb(0, 0, 0)
        ctx.show_text(txt)
    # cover picture panel
    px0, py0, pw, ph = x + w * 0.08, y + h * 0.46, w * 0.84, h * 0.42
    ctx.rectangle(px0, py0, pw, ph)
    set_color(ctx, g(0.9))
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(0.9)
    ctx.stroke()
    # a laughing girl (Summer) and a far walker
    shape(ctx, [(px0 + pw * 0.2, py0 + ph), (px0 + pw * 0.25, py0 + ph * 0.55), (px0 + pw * 0.42, py0 + ph * 0.5),
                (px0 + pw * 0.5, py0 + ph)], g(0.55), line=0.8)
    shape(ctx, [(px0 + pw * 0.26, py0 + ph * 0.42), (px0 + pw * 0.33, py0 + ph * 0.2),
                (px0 + pw * 0.42, py0 + ph * 0.4), (px0 + pw * 0.34, py0 + ph * 0.55)], g(0.95), line=0.8)
    stroke(ctx, [(px0 + pw * 0.75, py0 + ph * 0.25), (px0 + pw * 0.75, py0 + ph * 0.8)], 0.9)
    ctx.set_font_size(w * 0.075)
    ctx.move_to(x + w * 0.08, y + h * 0.965)
    ctx.set_source_rgb(0.86, 0.86, 0.86)
    ctx.show_text("IVG")
    e = ctx.text_extents("FREE")
    ctx.move_to(x + w * 0.92 - e.x_advance, y + h * 0.965)
    ctx.show_text("FREE")
    ctx.restore()


def draw_page9(ctx, x, y, w, h):
    """page 9 of the held tract: the tract's last page ('To Be Continued...', like the scan's page 27)"""
    ctx.save()
    ctx.rectangle(x, y, w, h)
    set_color(ctx, g(0.0))                 # tiny page: its dark stock prints as solid ink
    ctx.fill()
    bx, by, bw, bh = x + w * 0.07, y + h * 0.05, w * 0.86, h * 0.27
    ctx.rectangle(bx, by, bw, bh)
    set_color(ctx, g(1.0))
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(0.5)
    ctx.stroke()
    ctx.set_line_width(0.55)
    for k in range(6):                                         # lorem ipsum | FLAGS columns (unreadably small)
        yy = by + bh * (0.3 + k * 0.11)
        ctx.move_to(bx + bw * 0.06, yy)
        ctx.line_to(bx + bw * (0.44 - 0.03 * (k % 2)), yy)
        ctx.move_to(bx + bw * 0.56, yy)
        ctx.line_to(bx + bw * 0.94, yy)
    ctx.stroke()
    ctx.select_font_face("Nimbus Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(w * 0.105)
    t = "To Be Continued..."
    e = ctx.text_extents(t)
    ctx.move_to(x + (w - e.x_advance) / 2, y + h * 0.53)
    ctx.set_source_rgb(1, 1, 1)
    ctx.show_text(t)
    ctx.set_font_size(w * 0.05)
    ctx.set_source_rgb(0.86, 0.86, 0.86)
    ctx.move_to(x + w * 0.9, y + h * 0.965)
    ctx.show_text("9")
    ctx.restore()


# ============================================================ p2 — THE READING SHOT (over her shoulder)
class Reading:
    """page 8 panel 2: over Summer's shoulder, the open tract in her hands (page 8 = the window)"""

    def __init__(self, flag_track, f0_global):
        self.track = flag_track        # FlagTrack in PU, indexed by (global frame - f0_global)
        self.f0 = f0_global

    # ------------------------------------------------ timing helpers
    def _fi(self, T):
        return clamp(T * FPS - self.f0, 0, self.track.n - 1)

    def summer_motion(self, T):
        """(head nod, body breathing) small secondary motion"""
        b = 1.2 * math.sin(T * 2.1)
        look = smoothstep(T_PASS - 0.1, T_OPEN1, T)          # she bows her head to read
        jolt = 2.5 * math.exp(-max(0.0, T - (T_OPEN1 + 0.02)) * 9.0) * (T > T_OPEN1)
        return b, look, jolt

    def hand_pose(self, T):
        """(left thumb tip, right thumb tip, left fingers side) in PU; before the pass the hands wait, cupped"""
        wait_l, wait_r = (505.0, 752.0), (618.0, 758.0)
        if T < T_PASS - 0.05:
            u = smoothstep(183.9, T_PASS - 0.05, T)
            lift = 8.0 * u
            return (wait_l[0], wait_l[1] - lift), (wait_r[0], wait_r[1] - lift), "wait"
        if T < T_OPEN0:
            u = smoothstep(T_PASS - 0.05, T_PASS + 0.12, T)
            l = lerp2((wait_l[0], wait_l[1] - 8), (SPINE + 97.0, 668.0), u)
            r = lerp2((wait_r[0], wait_r[1] - 8), (SPINE + 101.0, 722.0), u)
            return l, r, "closed"
        phi = flap_angle(T)
        fx = SPINE + (D.WIN_W + 0.5) * math.cos(phi)
        return (fx, 668.0 + 8.0 * math.sin(phi)), (SPINE + 101.0, 722.0), "open"

    # ------------------------------------------------ layers
    def world(self, ctx, T, rect):
        x0, y0, w, h = rect
        hor = 92.0                                           # horizon (panel-local): we look down at the mud
        ctx.save()
        ctx.translate(x0, y0)
        TA.sky(ctx, w, h, T, top=0.95, bottom=0.985, horizon=hor)
        TA.glory_lines(ctx, -120.0, -260.0, 1.0, n=34, r0=40.0, r1=1100.0, a0=0.05, a1=math.pi * 0.5, lw=0.7)
        TA.tents(ctx, w, hor, 811, tone=0.72, n=7, scale=0.5)
        TA.ground(ctx, w, h, hor, 812, T, near=0.93, far=0.975, ruts=22)
        TA.puddles(ctx, [(600, 150, 70, 7, 3), (790, 215, 95, 9, 4), (700, 312, 80, 9, 5)], T, sky_tone=0.99)
        ctx.restore()
        # --- the held tract: page 9 (right) + its shadow on the mud
        st = booklet_state(T)
        if st != "none":
            ctx.save()
            sx0 = D.WIN_X + 4 if st == "open" else SPINE + 4
            ctx.rectangle(sx0, BOOK_Y0 + 7, SPINE + D.WIN_W + 6 - sx0, D.WIN_H + 1)
            set_color(ctx, g(0.72))
            ctx.fill()
            ctx.restore()
            draw_page9(ctx, *PG9)
            if st == "open":
                stroke(ctx, [(SPINE, BOOK_Y0 - 1), (SPINE, BOOK_Y1 + 1)], 1.6, taper=(0.05, 0.05))
        # --- Summer, over the shoulder (back of the head in the foreground)
        self._summer_back(ctx, T)

    def _summer_back(self, ctx, T):
        b, look, jolt = self.summer_motion(T)
        hx, hy = 228.0, 596.0 + b * 0.4 + look * 8.0 - jolt
        tilt = 0.06 + 0.05 * look
        lt, rt, mode = self.hand_pose(T)
        sway = 3.0 * math.sin(T * 2.3) + jolt * 1.5
        # ---- arms (behind the collar/jacket): black sleeves with shine, white faux-fur cuffs
        if mode == "wait":
            _sleeve_arm(ctx, [(330, 800), (400, 782), (lt[0] - 24, lt[1] + 10)], 34, fill=0.0)
            _sleeve_arm(ctx, [(700, 800), (660, 792), (rt[0] + 14, rt[1] + 12)], 30, fill=0.0)
            for (x, y), sd in ((lt, 1), (rt, -1)):
                shape(ctx, [(x - 18 * sd, y + 8), (x - 6 * sd, y - 7), (x + 10 * sd, y - 9), (x + 19 * sd, y + 1),
                            (x + 8 * sd, y + 12)], g(0.97), line=1.8)
                stroke(ctx, [(x - 6 * sd, y - 1), (x + 8 * sd, y - 2)], 1.0)
        else:
            _sleeve_arm(ctx, [(330, 800), (392, 772), (lt[0] - 22, lt[1] + 30)], 34, fill=0.0)
            _sleeve_arm(ctx, [(720, 800), (682, 788), (rt[0] + 12, rt[1] + 28)], 30, fill=0.0)
            _fingers_back(ctx, lt[0] - 2, lt[1] + 10, -1)
            _fingers_back(ctx, rt[0] + 2, rt[1] + 2, 1)
        # ---- jacket: solid black shoulders with vinyl shine
        shape(ctx, [(52, 802), (58, 738), (110, 700), (228, 684), (346, 700), (432, 742), (452, 802)], g(0.0),
              line=2.0)
        for a, b2 in (((96, 790), (128, 740)), ((356, 800), (382, 752)), ((402, 800), (416, 772))):
            stroke(ctx, [a, ((a[0] + b2[0]) / 2 + 5, (a[1] + b2[1]) / 2), b2], 3.2, color=g(1.0), taper=(0.5, 0.5))
        # ---- faux-fur collar: a tufted white ring round the neck
        fur = []
        for k in range(25):
            a = math.pi * (1.0 + k / 24)
            r = 1.0 + (0.07 if k % 2 else 0.0)
            fur.append((hx + 6 + math.cos(a) * 128 * r, hy + 132 + math.sin(a) * 34 * r))
        for k in range(12):
            a = k / 11 * math.pi
            r = 1.0 + (0.08 if k % 2 else 0.0)
            fur.append((hx + 6 + math.cos(a) * 134 * r, hy + 136 + math.sin(a) * 30 * r))
        shape(ctx, fur, g(1.0), line=2.4)
        ctx.save()
        ctx.new_path()
        path(ctx, fur, True)
        ctx.clip()
        shape(ctx, [(hx - 150, hy + 146), (hx + 160, hy + 146), (hx + 160, hy + 190), (hx - 150, hy + 190)], g(0.84))
        rng = np.random.default_rng(3)
        for k in range(34):
            a = math.pi * (1.02 + 0.96 * rng.uniform())
            rr = rng.uniform(0.5, 1.0)
            x, y = hx + 6 + math.cos(a) * 128 * rr, hy + 138 + math.sin(a) * 30 * rr + 8
            stroke(ctx, [(x, y), (x + 3, y + 5), (x + 2, y + 10)], 1.2, taper=(0.3, 0.6))
        ctx.restore()
        # ---- head (turned a touch toward the tract, bowed to read)
        ctx.save()
        ctx.translate(hx, hy)
        ctx.rotate(tilt)
        # neck + goggle strap at the nape
        shape(ctx, [(-40, 72), (40, 72), (46, 128), (-46, 128)], g(0.99), line=2.0)
        # lost profile: cheek, an eyelash flick, freckles
        shape(ctx, [(84, -26), (108, -6), (118, 30), (110, 66), (88, 92), (70, 60)], g(0.99), line=2.0)
        stroke(ctx, [(110, -8), (120, -14), (126, -12)], 1.6, taper=(0.1, 0.7))
        stroke(ctx, [(112, -2), (123, -4)], 1.2, taper=(0.1, 0.8))
        for fx, fy in ((106, 30), (111, 38), (101, 40), (108, 46)):
            ctx.arc(fx, fy, 1.3, 0, 2 * math.pi)
            ctx.set_source_rgb(0, 0, 0)
            ctx.fill()
        # the hair mass: paper-white blonde volume with a clumpy edge; only the shadow side is screened
        head = []
        for k in range(40):
            a = -math.pi / 2 + (k / 40) * 2 * math.pi
            up = max(0.0, -math.sin(a))                           # clumps along the top, smooth at the nape
            bump = (5.0 if k % 3 == 0 else (-2.0 if k % 3 == 1 else 1.5)) * up
            head.append((math.cos(a) * (111 + bump), -12 + math.sin(a) * (120 + bump) + (6 if math.sin(a) > 0.6 else 0)))
        shape(ctx, head, g(1.0), line=0.0, n=4)
        ctx.save()
        ctx.new_path()
        path(ctx, head, True, n=4)
        ctx.clip()
        shape(ctx, [(50, -150), (150, -60), (150, 130), (10, 130), (66, 70), (90, 8), (70, -70)], g(0.88))
        ctx.restore()
        shape(ctx, head, None, line=3.0, n=4)
        # strands: few, confident, uneven — from the parting, combed out and down into the two low ties
        clumps = ((-126, 1.00, 2.6, 1.00, 0), (-116, 0.28, 1.0, 0.98, 6), (-104, 0.46, 1.2, 0.95, -4),
                  (-78, 1.00, 1.7, 0.88, 3), (-60, 0.30, 0.9, 0.84, 0), (-34, 0.55, 1.1, 0.74, 5),
                  (0, 1.00, 1.5, 0.60, -3), (40, 0.45, 0.9, 0.40, 2))
        for sd in (-1, 1):
            pe = (sd * 90.0, 64.0)
            for k, (y0, reach, wdt, bul, jit) in enumerate(clumps):
                if sd > 0 and k in (1, 4):
                    continue                                      # asymmetry: hair is never mirrored
                p0 = (sd * (3.0 + 2 * math.sin(k * 1.7)), y0 + (jit if sd > 0 else -jit * 0.5))
                c = (sd * (54 + 68 * bul + jit), (y0 + pe[1]) * 0.5 - 24 * bul)
                ts = np.linspace(0.0, reach, 10)
                pts = [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * pe[0],
                        (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * pe[1]) for t in ts]
                stroke(ctx, pts, wdt, taper=(0.1, 0.8 if reach < 1 else 0.3))
            for k in range(3):                                   # gathered into the tie
                a = -0.8 + 0.4 * k
                stroke(ctx, [(pe[0] - sd * 24 * math.cos(a), pe[1] + 24 * math.sin(a)), (pe[0] - sd * 5, pe[1] + 3)],
                       1.0, taper=(0.5, 0.2))
            for k in range(3):                                   # flyaways along the silhouette
                a = -math.pi / 2 + sd * (0.5 + 0.35 * k)
                x, y = math.cos(a) * 112, -12 + math.sin(a) * 121
                stroke(ctx, [(x, y), (x + sd * 7, y - 5 + 3 * k), (x + sd * 11, y - 2 + 5 * k)], 1.0, taper=(0.2, 0.8))
        stroke(ctx, [(1, -131), (5, -86), (2, -40)], 2.4, taper=(0.05, 0.5))                    # the parting
        stroke(ctx, [(3, -20), (1, 30), (2, 98)], 1.6, taper=(0.4, 0.3))
        # pigtails + scrunchies (in front of the collar)
        for sd in (-1, 1):
            tx, ty = sd * 94.0, 70.0
            sw = sway * (1.0 if sd > 0 else 0.8)
            tail = [(tx - 8 * sd, ty), (tx + 26 * sd, ty + 42), (tx + 30 * sd + sw, ty + 104),
                    (tx + 16 * sd + sw * 1.3, ty + 154), (tx + 6 * sd + sw * 1.5, ty + 184),
                    (tx - 6 * sd + sw, ty + 150), (tx - 10 * sd, ty + 96), (tx - 14 * sd, ty + 40)]
            shape(ctx, tail, g(1.0), line=2.4)
            for k in range(6):
                u = 0.12 + 0.13 * k
                stroke(ctx, [(tx + sd * (6 - 8 * u), ty + 14 + 150 * u - 12),
                             (tx + sd * (18 - 6 * u) + sw * u, ty + 30 + 150 * u),
                             (tx + sd * (8 - 4 * u) + sw * u * 1.2, ty + 50 + 150 * u)], 1.2, taper=(0.3, 0.6))
            ctx.save()
            ctx.translate(tx, ty)
            shape(ctx, [(-13, -4), (-6, -12), (8, -11), (14, -2), (9, 9), (-8, 10)], g(0.0), line=1.0)
            stroke(ctx, [(-6, -6), (4, -8)], 2.0, color=g(1.0), taper=(0.3, 0.3))
            ctx.restore()
        ctx.restore()

    def flags(self, ctx, T, rect):
        x, y, w, h = P.inset(rect, P.BORDER / 2 + 0.5)
        ctx.save()
        ctx.rectangle(x, y, w, h)
        ctx.clip()
        self.track.draw(ctx, self._fi(T))
        ctx.restore()

    def flag_bbox(self, T):
        """cloth bbox (PU) for lettering avoidance"""
        v = self.track.verts[int(round(self._fi(T)))]
        return (float(v[..., 0].min()), float(v[..., 1].min()), float(v[..., 0].max()), float(v[..., 1].max()))

    def front(self, ctx, T, rect):
        pass

    def cover_quads(self, T, live=True):
        """where the closed tract's FRONT COVER is: [cairo.Matrix mapping the cover page (1000 x 1298 PU of the
        tract's own page) -> level PU].  The cover itself is T01's printed cover texture (its flag alive)."""
        st = booklet_state(T)
        s = D.WIN_W / P.PAGE_W
        out = []
        if live and T_HAND_IN <= T < T_PASS:
            hx, hy = self.crow_hand_xy(T)
            M = cairo.Matrix(s, 0, 0, s, hx - 95.0, hy - D.WIN_H * 0.6)
            out.append(M)
        elif st == "closed":
            out.append(cairo.Matrix(s, 0, 0, s, PG9[0], PG9[1]))
        elif st == "opening":
            c = math.cos(flap_angle(T))
            if c > 0:                                    # front cover still facing us, hinged on the spine
                sx = SPINE + 1.0
                out.append(cairo.Matrix(s * max(c, 1e-3), 0, 0, s, sx + (PG9[0] - sx) * c, PG9[1]))
        return out

    def crow_hand_xy(self, T):
        k_in = smootherstep(0, 1, (T - T_HAND_IN) / (T_PASS - T_HAND_IN))
        k_out = smootherstep(0, 1, (T - (T_PASS + 0.16)) / (T_HAND_OUT - T_PASS - 0.16))
        press = 5.0 * math.sin(math.pi * clamp((T - T_PASS + 0.04) / 0.16, 0, 1))
        rest = (SPINE + 96.0, 690.0)
        start = (1010.0, 760.0)
        away = (1015.0, 790.0)
        hx, hy = lerp2(start, rest, k_in) if k_out <= 0 else lerp2(rest, away, k_out)
        return hx, hy + press

    fallback_covers = True              # set False when T01's printed cover texture is composited instead

    def holders(self, ctx, T, rect, live=True, covers=None):
        """over the window: Crow's hand bringing the tract, the flap's shadow, Summer's thumbs
        (covers=True also inks a stand-in cover, when T01's printed cover texture is unavailable)"""
        st = booklet_state(T)
        lt, rt, mode = self.hand_pose(T)
        if covers is None:
            covers = self.fallback_covers
        if covers:
            for M in self.cover_quads(T, live):
                ctx.save()
                ctx.transform(M)
                draw_cover(ctx, 0, 0, P.PAGE_W, P.PAGE_H)
                ctx.restore()
        if live and T_HAND_IN <= T <= T_HAND_OUT:
            self._crow_hand(ctx, T)
        if st == "opening":
            c = math.cos(flap_angle(T))
            if c > 0:                                    # the flap's shadow falling on page 9
                ctx.rectangle(SPINE + 1 + D.WIN_W * c, BOOK_Y0, 6 + 10 * (1 - c), D.WIN_H)
                ctx.set_source_rgba(0, 0, 0, 0.25 * (1 - c))
                ctx.fill()
        if mode != "wait":
            _thumb(ctx, lt[0] + 7.0, lt[1] - 4.0, -0.25, 1.0)
            _thumb(ctx, rt[0] - 7.0, rt[1] - 2.0, 0.3, 1.0)

    def _crow_hand(self, ctx, T):
        """Crow's arm from the right edge: presses the closed tract into her hands on PASS"""
        hx, hy = self.crow_hand_xy(T)
        # coat sleeve with fur cuff
        _sleeve_arm(ctx, [(hx + 260, hy + 70), (hx + 120, hy + 38), (hx + 26, hy + 8)], 44, fill=0.0)
        # the hand: back of the hand + fingers laid over the tract
        shape(ctx, [(hx + 30, hy - 22), (hx - 4, hy - 30), (hx - 30, hy - 26), (hx - 44, hy - 12),
                    (hx - 40, hy + 4), (hx - 8, hy + 10), (hx + 24, hy + 22), (hx + 36, hy + 2)], g(0.98), line=2.2)
        for k in range(3):
            yy = hy - 22 + k * 9
            stroke(ctx, [(hx - 8, yy), (hx - 30, yy + 2), (hx - 42, yy + 6)], 1.3)
        stroke(ctx, [(hx + 10, hy - 14), (hx + 22, hy - 6)], 1.2)

    def letters(self, ctx, fc, T, rect, hold=30.0):
        x0, y0, w, h = rect
        av = [self.flag_bbox(T)]
        comic.say(ctx, fc, "P30", x0 + 690, y0 + 58, text="Welcome to the Survey, sister. Now go... ...and PASS IT ON.",
                  tokens=(7, None), tail=(x0 + w + 30, y0 + 230), kind="offpanel", size=P.LETTER["balloon"],
                  w=190, hold=hold, T=T, avoid=av, bounds=(x0 + 8, y0 + 8, x0 + w - 8, y0 + h - 8))


# ============================================================ p3 — THE NEXT DAY
P3_HOR = 172.0                        # panel-local horizon
TOSS_T = (195.95, 196.15, 196.30, 196.52)   # wind-up, back, release (tract_toss), follow-through


class NextDay:
    """page 8 panel 3: the camp the next day. Sun, mud drying, Raven in her chair under the (dry) shade with the
    tract Summer passed her; Summer walks by carrying the Flag — and waves."""

    def __init__(self, walk, flag_track, f0_global, crack_seed=7):
        self.walk = walk                  # T -> Summer state dict | None (t09_passiton.summer_walk)
        self.track = flag_track
        self.f0 = f0_global
        self.raven = Character("raven", 3, render="ink")
        self.summer = Character("summer", 1, render="ink")
        self.rain = TA.Rain(P3[2], P3[3], 380, 831, slant=0.22, length=(10, 24))
        self.tl = None
        rng = np.random.default_rng(crack_seed)
        self.cracks = self._make_cracks(rng)

    def _fi(self, T):
        return clamp(T * FPS - self.f0, 0, self.track.n - 1)

    def _make_cracks(self, rng):
        """drying-mud crack network (panel-local): a Voronoi tiling of the GROUND PLANE seen in perspective,
        so the plates shrink toward the horizon; returns [(p0, p1, width)]"""
        from scipy.spatial import Voronoi
        w, h = P3[2], P3[3]
        f = h - P3_HOR                                      # ground depth z = 1 at the panel's bottom edge
        cx = w * 0.5
        pts = []
        while len(pts) < 900:
            X, z = rng.uniform(-60, 60), rng.uniform(0.85, 24.0)
            if abs(X) < (cx + 60) / f * z:
                pts.append((X, z))
        pts = np.array(pts)
        vor = Voronoi(pts)
        segs = []
        for (a, b) in vor.ridge_vertices:
            if a < 0 or b < 0:
                continue
            (X0, z0), (X1, z1) = vor.vertices[a], vor.vertices[b]
            if min(z0, z1) < 0.6:
                continue
            p0 = (cx + X0 * f / z0, P3_HOR + f / z0)
            p1 = (cx + X1 * f / z1, P3_HOR + f / z1)
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1.2 or max(p0[0], p1[0]) < -20 or min(p0[0], p1[0]) > w + 20:
                continue
            zm = 0.5 * (z0 + z1)
            segs.append((p0, p1, max(0.35, min(2.6, 2.3 / zm))))
        return segs

    # ------------------------------------------------ beats
    def rain_amount(self, T):
        return smoothstep(196.30, 196.75, T)

    def toss_progress(self, T):
        a, b, c, d = TOSS_T
        if T < a:
            return None
        if T < b:
            return 0.35 * smoothstep(a, b, T)
        if T < c:
            return 0.35 + 0.35 * (T - b) / (c - b)
        return 0.7 + 0.3 * smoothstep(c, d, T)

    def raven_args(self, T):
        """Raven in her chair (panel-local)"""
        x, y, h = 196.0, 332.0, 300.0
        wk = self.walk_state(T)
        lookx = 1.0 if wk is None else clamp((wk["x"] - P3[0] - x) / 260.0, -1, 1)
        mouth = self.tl.mouth("RAVEN", T) if self.tl is not None else 0.0
        glance = smoothstep(195.65, 195.9, T)                  # a look at the tract before tossing it
        pr = self.toss_progress(T)
        pose = {"sit": 1.0}
        kw = dict(chair="camp", legs_crossed=True, turn=0.55)
        if pr is not None:
            pose["toss"] = 1.0
            kw["progress"] = pr
        if pr is None or pr < 0.7:
            kw["prop"] = "tract"
        callback = smoothstep(194.70, 194.95, T)
        expr = {"sardonic": 1.0 - 0.6 * callback, "disdain": 0.6 * callback}
        look = (lookx * (1 - glance) + 0.3 * glance, -0.15 + 0.4 * glance)
        return dict(x=x, y=y, h=h, pose=pose, t=T, facing=1, mouth=mouth, look=look, expr=expr, **kw)

    def walk_state(self, T):
        return self.walk(T) if callable(self.walk) else None

    def _local_anchors(self, T):
        a = dict(self.raven_args(T))
        x, y, h, pose = a.pop("x"), a.pop("y"), a.pop("h"), a.pop("pose")
        a.pop("mouth", None)
        a.pop("expr", None)
        a.pop("look", None)
        t = a.pop("t")
        return self.raven.anchors(x, y, h, pose, t, **a)

    def raven_mouth_xy(self, T):
        m = self._local_anchors(T)["mouth"]
        return (P3[0] + m[0], P3[1] + m[1])

    def raven_hand_xy(self, T):
        m = self._local_anchors(T)["hand_r"]
        return (P3[0] + m[0], P3[1] + m[1])

    # ------------------------------------------------ layers
    def world(self, ctx, T, rect):
        x0, y0, w, h = rect
        rain = self.rain_amount(T)
        ctx.save()
        ctx.translate(x0, y0)
        TA.sky(ctx, w, h, T, top=0.955 - 0.25 * rain, bottom=0.985 - 0.15 * rain, horizon=P3_HOR)
        # the sun: a Chick-tract sun, rays turning slowly
        sx, sy = 690.0, 64.0
        for k in range(22):
            a = k / 22 * 2 * math.pi + 0.08 * T
            r0, r1 = 40, 66 + (12 if k % 2 else 0)
            stroke(ctx, [(sx + math.cos(a) * r0, sy + math.sin(a) * r0),
                         (sx + math.cos(a) * r1, sy + math.sin(a) * r1)], 3.0, taper=(0.05, 0.85))
        ctx.arc(sx, sy, 31, 0, 2 * math.pi)
        set_color(ctx, g(1.0))
        ctx.fill_preserve()
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(2.2)
        ctx.stroke()
        for cx, cy, s in ((150, 42, 0.8), (430, 30, 1.0)):
            pts = [(cx - 60 * s, cy + 12), (cx - 44 * s, cy - 6), (cx - 14 * s, cy - 12), (cx + 8 * s, cy - 22 * s),
                   (cx + 36 * s, cy - 12), (cx + 60 * s, cy + 2), (cx + 62 * s, cy + 14)]
            shape(ctx, pts, g(0.97), line=1.5)
        if rain > 0:                                        # the storm comes back over the sun
            cx = sx + 140 - 150 * rain
            cl = [(cx - 190, 118), (cx - 170, 50), (cx - 90, 14), (cx + 10, -18), (cx + 140, 6), (cx + 210, 70),
                  (cx + 120, 124), (cx - 40, 132)]
            ctx.save()
            ctx.translate(0, -90 * (1 - rain))
            shape(ctx, cl, g(0.46), line=2.4)
            ctx.restore()
        TA.tents(ctx, w, P3_HOR, 833, tone=0.68, n=8, scale=0.62)
        TA.ground(ctx, w, h, P3_HOR, 834, T, near=0.92 - 0.12 * rain, far=0.965 - 0.1 * rain, ruts=16)
        # drying mud: cracks
        ctx.save()
        ctx.set_source_rgb(0, 0, 0)
        for p0, p1, lw in self.cracks:
            ctx.set_line_width(lw)
            ctx.move_to(*p0)
            ctx.line_to(*p1)
            ctx.stroke()
        ctx.restore()
        TA.puddles(ctx, [(640, 214, 64, 6, 11), (790, 300, 52, 6, 12)], T, sky_tone=0.99, stop=0.0)
        # Summer's EMPTY chair next to Raven's
        TA.camp_chair(ctx, 396, 318, 60, facing=-1)
        # Summer walking by (midground) — body; the pole hand goes to the front layer
        wk = self.walk_state(T)
        if wk is not None:
            self._summer(ctx, T, wk, "back")
        self._draw_raven(ctx, T)
        if rain > 0:
            self.rain.draw(ctx, T, amount=rain)
        ctx.restore()

    def _summer(self, ctx, T, wk, layer):
        kw = dict(wk["kw"])
        for k in ("hand_l", "hand_r"):                   # IK targets are absolute PU; we draw panel-local
            if k in kw:
                kw[k] = (kw[k][0] - P3[0], kw[k][1] - P3[1])
        self.summer.draw(ctx, wk["x"] - P3[0], wk["y"] - P3[1], wk["h"], wk["pose"], T, facing=wk["facing"],
                         expr=wk["expr"], look=wk["look"], layer=layer, **kw)

    def _draw_raven(self, ctx, T):
        a = dict(self.raven_args(T))
        x, y, h, pose = a.pop("x"), a.pop("y"), a.pop("h"), a.pop("pose")
        self.raven.draw(ctx, x, y, h, pose, **a)

    def flags(self, ctx, T, rect):
        if self.walk_state(T) is None:
            return
        x, y, w, h = P.inset(rect, P.BORDER / 2 + 0.5)
        ctx.save()
        ctx.rectangle(x, y, w, h)
        ctx.clip()
        self.track.draw(ctx, self._fi(T))
        ctx.restore()

    def front(self, ctx, T, rect):
        wk = self.walk_state(T)
        if wk is not None:
            ctx.save()
            ctx.translate(rect[0], rect[1])
            self._summer(ctx, T, wk, "front")
            ctx.restore()

    def avoid(self, T):
        """the cloth's path (PU) for the R02 balloon"""
        if T < 192.0:
            return None
        return lambda TT: comic.track_avoid(self.track, self._fi(TT)) if TT * FPS >= self.f0 else []

    def letters(self, ctx, fc, T, rect, caption=True):
        x0, y0, w, h = rect
        if caption:
            comic.caption(ctx, "THE NEXT DAY...", x0 + 8, y0 + 8, 150, size=P.LETTER["caption"])
        mx, my = self.raven_mouth_xy(T)
        comic.say(ctx, fc, "R02", x0 + 300, y0 + 74, tail=(mx + 4, my - 3), size=P.LETTER["balloon"], w=230, T=T,
                  avoid=self.avoid(T), bounds=(x0 + 8, y0 + 40, x0 + w - 8, y0 + h - 8))


# ============================================================ the printed page (static) for the Droste texture
class StaticLevel:
    """page 8 as printed: every panel frozen at its print time; window punched by t09_droste.render_level"""
    page_no = PAGE_NO

    def __init__(self, p1, p2, p3, fc_like):
        self.p = {"p1": (p1, T_P1_PRINT), "p2": (p2, T_P2_PRINT), "p3": (p3, T_P3_PRINT)}
        self.fc = fc_like

    def _each(self, ctx, meth):
        for k, (d, T) in self.p.items():
            fn = getattr(d, meth, None)
            if fn is None:
                continue
            r = P.PANELS[k]
            ctx.save()
            ctx.rectangle(*r)
            ctx.clip()
            fn(ctx, T, r)
            ctx.restore()

    def world(self, ctx):
        self._each(ctx, "world")

    with_flags = False                  # the Print-Gallery texture is baked without flags (FlagBaker lays them live)

    def flags(self, ctx):
        if self.with_flags:
            self._each(ctx, "flags")

    def front(self, ctx):
        self._each(ctx, "front")

    def letters(self, ctx, fc):
        fc = self.fc
        for k, (d, T) in self.p.items():
            fn = getattr(d, "letters", None)
            if fn is None:
                continue
            fc.T = T
            fn(ctx, fc, T, P.PANELS[k])

    def holders(self, ctx):
        d, T = self.p["p2"]
        ctx.save()
        ctx.rectangle(*P.PANELS["p2"])
        ctx.clip()
        d.holders(ctx, T, P.PANELS["p2"], live=False)
        ctx.restore()


# ============================================================ fallback p1 (until T08's drawer lands)
class P1Fallback:
    def __init__(self):
        self.crow = Character("crow", render="ink")
        self.summer = Character("summer", render="ink")

    def world(self, ctx, T, rect):
        x0, y0, w, h = rect
        ctx.set_source(lin_grad(0, y0, 0, y0 + h, [(0, g(0.82)), (0.55, g(0.93)), (0.56, g(0.62)), (1, g(0.7))]))
        ctx.rectangle(x0, y0, w, h)
        ctx.fill()
        self.summer.draw(ctx, x0 + 300, y0 + h - 20, 300, {"kneel": 1, "pray": 1}, T, facing=1, render="ink",
                         eyes_closed=True)
        self.crow.draw(ctx, x0 + 560, y0 + h - 20, 330, "stand", T, facing=-1, render="ink",
                       mouth=0.0)

    def letters(self, ctx, fc, T, rect, hold=30.0):
        x0, y0, w, h = rect
        comic.say(ctx, fc, "P30", x0 + 560, y0 + 70, tokens=(0, 7), tail=(x0 + 560, y0 + 120), size=22, T=T,
                  hold=hold)


# ============================================================ page 8 p1 continued (T08's prayer panel, Crow speaks P30)
def make_p1(T8):
    """T08's P8P1 drawer, alive after 181.17: Crow opens his eyes, welcomes her, pulls a tract from his coat
    and holds it out on 'Now go...'; Summer comes down from the sunbeam to look at him."""
    base = T8.P8P1
    T1 = T8.T1

    class P8P1Cont(base):
        hold = 1e9

        def crow(self, T):
            d = base.crow(self, min(T, T1))
            if T <= T1:
                return d
            d["t"] = T
            d["mouth"] = T8.mouth("PREACHER", T)
            wake = smoothstep(T1 + 0.15, T1 + 0.55, T)
            unclasp = smoothstep(181.45, 182.05, T)
            offer = smootherstep(0.0, 1.0, (T - 183.05) / 0.75)
            d["eyes_closed"] = wake < 0.5
            d["nod"] = -0.12 + 0.14 * wake - 0.05 * offer
            d["look"] = (-1.0, 0.15 - 0.1 * offer)
            d["expr"] = {"serene": 1.0 - 0.6 * wake, "fervent": 0.35 * wake, "beatific": 0.25 * wake}
            pose = {"kneel": 1.0, "pray": 1.0 - unclasp}
            if offer > 0:
                pose["offer_tract"] = offer
                d["prop"] = "tract"
                # the tract travels from his coat to just before her clasped hands
                sx, sy = d["x"], d["y"]
                chest = (sx - 18.0, sy - d["h"] * 0.52)
                reach = (sx - 150.0, sy - d["h"] * 0.50)
                d["hand_r"] = lerp2(chest, reach, offer)
                d["shape_r"] = "pinch"
            elif unclasp > 0.5 and T > 182.75:
                d["prop"] = "tract"             # the tract comes out of his coat on 'sister'
                pose["offer_tract"] = 0.25 * smoothstep(182.75, 183.05, T)
            d["pose"] = pose
            return d

        def summer(self, T):
            d = base.summer(self, min(T, T1))
            if T <= T1:
                return d
            d["t"] = T
            down = smoothstep(181.6, 182.3, T)
            d["eyes_closed"] = False
            d["nod"] = 0.35 * (1 - down) + 0.02 * down
            d["look"] = (-0.6 + 1.6 * down, -0.7 + 0.8 * down)
            d["expr"] = {"beatific": 1.0 - 0.5 * down, "rapt": 0.5 * down}
            reach = smoothstep(183.55, 184.0, T)
            d["pose"] = {"kneel": 1.0, "pray": 1.0 - 0.7 * reach}
            return d

        def raven(self, T):
            return base.raven(self, min(T, T1))

    return P8P1Cont()

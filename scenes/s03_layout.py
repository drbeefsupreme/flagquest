"""s03 layout: world geometry, TV float physics, camera path. Pure functions of global time T.

World units = design units of the WIDE shot (camera zoom 1 shows the whole flooded living room).
"""
import math
from functools import lru_cache

from vx.ease import clamp, lerp, seg, ease_in_out, ease_out, ease_in, smoothstep, fbm1, noise1

T0, T1 = 47.5, 61.5

# ---------------------------------------------------------------- the room
EYE_Y = 360.0          # eye level (vanishing line) in world
WALL_WL = 606.0        # waterline on the back wall

# the TV: local frame origin = centre of the tube opening; units = world
TV_X, TV_Y = 1010.0, 452.0
TUBE_HW, TUBE_HH = 208.0, 156.0          # half size of the glass opening (4:3)
TV_WL_LOCAL = 212.0                        # waterline in TV-local y (rest)
CAB_L, CAB_R, CAB_T, CAB_B = -262.0, 372.0, -214.0, 470.0   # cabinet box (local)

# ---------------------------------------------------------------- key times
def _tl():
    from vx.timeline import get_timeline
    return get_timeline()


@lru_cache(maxsize=1)
def times():
    tl = _tl()
    ban = tl.word("J01", "banned")
    return dict(
        tv_on=tl.cue("tv_on"),                 # 47.5
        lock=48.02,                            # vertical hold locks
        pull0=48.05, pull1=50.55,              # pull back to the wide
        push1=56.9,                            # slow push to the medium ends
        slam=ban[0],                           # fist hits the podium on "BANNED" (music stab 57.00)
        pen=tl.cue("pen_sign"),                # 57.6 flourish begins
        j02=tl.line("J02")["start"],
        push2a=57.9, push2b=60.35,             # intimate push-in for the aside
        coin=tl.cue("ka_ching"),               # 60.9 coin lands in pocket
        off=61.0,                              # CRT switch-off
    )


# ---------------------------------------------------------------- TV float physics
def slam_kick(T, ts):
    """damped response of the floating TV to the slam impulse -> (dy, drot)"""
    dt = T - ts
    if dt < 0:
        return 0.0, 0.0
    e = math.exp(-dt * 2.3)
    dy = 21.0 * math.sin(dt * 2 * math.pi * 1.35) * e
    dr = -0.045 * math.sin(dt * 2 * math.pi * 1.1 + 0.4) * math.exp(-dt * 1.9)
    return dy, dr


def tv_pose(T):
    """(x, y, rot) of the tube centre in world + waterline offset. Slow buoyant bob, slam jolt."""
    tm = times()
    bob = 6.5 * math.sin(T * 1.13 + 0.7) + 2.6 * math.sin(T * 2.07 + 1.9)
    roll = 0.026 * math.sin(T * 0.83 + 0.2) + 0.009 * math.sin(T * 1.71 + 2.2)
    dy, dr = slam_kick(T, tm["slam"])
    x = TV_X + 3.0 * math.sin(T * 0.37)
    y = TV_Y + bob + dy
    return x, y, roll + dr


# ---------------------------------------------------------------- camera
def _logz(a, b, t):
    return math.exp(lerp(math.log(a), math.log(b), t))


Z_OPEN = 4.95


def camera(T):
    """(cx, cy, zoom) world point at screen centre. Log-space zoom so moves feel uniform."""
    tm = times()
    tx, ty, tr = tv_pose(T)
    # key framings
    wide = (968.0, 520.0, 1.0)
    med = (TV_X + 28.0, TV_Y + 18.0, 1.62)
    close = (TV_X + 6.0, TV_Y + 6.0, 2.62)
    if T < tm["pull0"]:
        # locked on the tube (follow its bob so the static stays full frame)
        return tx, ty, Z_OPEN
    if T < tm["pull1"] + 0.6:
        k = seg(T, tm["pull0"], tm["pull1"], lambda u: ease_in_out(u, 2.6))
        # the lock-on to the bobbing tube releases as we pull back
        cx = lerp(tx, wide[0], k)
        cy = lerp(ty, wide[1], k)
        z = _logz(Z_OPEN, wide[2], k)
        # settle drift after the pull
        d = seg(T, tm["pull1"], tm["pull1"] + 0.6, ease_out)
        z *= 1 + 0.004 * d
        return cx, cy, z
    if T < tm["push2a"]:
        k = seg(T, tm["pull1"] + 0.6, tm["push1"], lambda u: ease_in_out(u, 2.0))
        z = _logz(1.004, med[2], k)
        cx = lerp(wide[0], med[0], k)
        cy = lerp(wide[1], med[1], k)
        # after the push: hold medium with tiny drift until the aside
        d = seg(T, tm["push1"], tm["push2a"], ease_in_out)
        z *= 1 + 0.01 * d
        return cx, cy, z
    k = seg(T, tm["push2a"], tm["push2b"], lambda u: ease_in_out(u, 2.2))
    z = _logz(med[2] * 1.01, close[2], k)
    cx = lerp(med[0], close[0], k)
    cy = lerp(med[1], close[1], k)
    # creep during coin + switch off
    d = seg(T, tm["push2b"], T1, ease_in_out)
    z *= 1 + 0.025 * d
    return cx, cy, z


def cam_shake(T):
    tm = times()
    dt = T - tm["slam"]
    a = 0.0
    if dt >= 0:
        a = 15.0 * math.exp(-dt * 6.0)
    return a * fbm1(T * 23.0, 5, 2), a * fbm1(T * 23.0, 77, 2)


def world_to_screen(x, y, cam):
    cx, cy, z = cam
    return 960.0 + (x - cx) * z, 540.0 + (y - cy) * z


def tube_geom(T, s):
    """tube centre (px), half sizes (px), rotation for a frame at scale s."""
    cam = camera(T)
    sh = cam_shake(T)
    x, y, r = tv_pose(T)
    sx, sy = world_to_screen(x, y, cam)
    z = cam[2]
    return (sx + sh[0]) * s, (sy + sh[1]) * s, TUBE_HW * z * s, TUBE_HH * z * s, r

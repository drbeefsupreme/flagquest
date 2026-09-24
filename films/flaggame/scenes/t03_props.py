"""t03 props for the forecast (owner T03Genesis): the boombox in the mud, the Ranger's handheld radio.
Drawn in grey tones + black line art for vx.ink.engrave. Local coordinates: boombox body centre at (0, 0),
900 x 460 units; radio: 100 x 260 units, centre at (0, 0)."""
import math

import cairo
import numpy as np

from vx import *

BB_W, BB_H = 900.0, 460.0
SPK_L = (-262.0, 40.0)
SPK_R = (262.0, 40.0)
SPK_R0 = 150.0


def _g(ctx, v, a=1.0):
    ctx.set_source_rgba(v, v, v, a)


def _ink(ctx, w):
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(w)


def speaker(ctx, cx, cy, R, pulse=0.0, t=0.0):
    """grille ring + cone rings + dust cap; pulse 0..1 pushes the cone"""
    # chrome surround
    ctx.arc(cx, cy, R * 1.08, 0, 2 * math.pi)
    ctx.set_source(rad_grad(cx - R * 0.4, cy - R * 0.5, R * 0.1, R * 1.3,
                            [(0, (0.95, 0.95, 0.95)), (0.6, (0.6, 0.6, 0.6)), (1, (0.3, 0.3, 0.3))]))
    ctx.fill()
    _ink(ctx, 3.5)
    ctx.arc(cx, cy, R * 1.08, 0, 2 * math.pi)
    ctx.stroke()
    # the cone: dark, concentric (the hurricane's eye becomes this)
    k = 1.0 + 0.035 * pulse
    ctx.arc(cx, cy, R * 0.97, 0, 2 * math.pi)
    ctx.set_source(rad_grad(cx + R * 0.2, cy + R * 0.25, R * 0.1, R, [(0, (0.55, 0.55, 0.55)), (1, (0.12, 0.12, 0.12))]))
    ctx.fill()
    for i, rr in enumerate((0.9, 0.78, 0.64, 0.5)):
        _ink(ctx, 2.2 if i % 2 == 0 else 1.2)
        ctx.arc(cx, cy, R * rr * (k if i > 1 else 1.0), 0, 2 * math.pi)
        ctx.stroke()
    # dust cap, lit
    ctx.arc(cx, cy, R * 0.26 * k, 0, 2 * math.pi)
    ctx.set_source(rad_grad(cx - R * 0.1, cy - R * 0.1, 1, R * 0.3, [(0, (0.97, 0.97, 0.97)), (1, (0.35, 0.35, 0.35))]))
    ctx.fill_preserve()
    _ink(ctx, 2.5)
    ctx.stroke()
    # perforated grille: a lattice of holes
    ctx.save()
    ctx.arc(cx, cy, R * 0.97, 0, 2 * math.pi)
    ctx.clip()
    _g(ctx, 0.03)
    step = R * 0.085
    for iy in range(-12, 13):
        for ix in range(-12, 13):
            x = cx + (ix + 0.5 * (iy % 2)) * step
            y = cy + iy * step * 0.87
            if (x - cx) ** 2 + (y - cy) ** 2 < (R * 0.95) ** 2:
                ctx.arc(x, y, step * 0.2, 0, 2 * math.pi)
                ctx.close_path()
    ctx.fill()
    ctx.restore()


def boombox(ctx, t, level=0.0, wet=0.0, mud=0.6, seed=3):
    """the ghetto blaster; level = voice level 0..1 (cones pulse, VU needles jump)"""
    rng = np.random.default_rng(seed)
    x0, y0 = -BB_W / 2, -BB_H / 2
    # handle
    _ink(ctx, 26)
    ctx.move_to(-300, y0 + 10)
    ctx.line_to(-300, y0 - 70)
    ctx.curve_to(-300, y0 - 110, 300, y0 - 110, 300, y0 - 70)
    ctx.line_to(300, y0 + 10)
    ctx.stroke()
    _g(ctx, 0.7)
    ctx.set_line_width(12)
    ctx.move_to(-300, y0 - 70)
    ctx.curve_to(-300, y0 - 102, 300, y0 - 102, 300, y0 - 70)
    ctx.stroke()
    # telescopic antenna, bent
    _ink(ctx, 7)
    ctx.move_to(360, y0 + 5)
    ctx.line_to(520, y0 - 330)
    ctx.stroke()
    _g(ctx, 0.9)
    ctx.set_line_width(2.5)
    ctx.move_to(362, y0 + 2)
    ctx.line_to(518, y0 - 326)
    ctx.stroke()
    _ink(ctx, 4)
    ctx.move_to(520, y0 - 330)
    ctx.line_to(610, y0 - 520)
    ctx.stroke()
    ctx.arc(612, y0 - 524, 9, 0, 2 * math.pi)
    ctx.fill()
    # body
    rrect(ctx, x0, y0, BB_W, BB_H, 40)
    ctx.set_source(lin_grad(0, y0, 0, y0 + BB_H, [(0, (0.82, 0.82, 0.82)), (0.12, (0.62, 0.62, 0.62)),
                                                  (1, (0.34, 0.34, 0.34))]))
    ctx.fill_preserve()
    _ink(ctx, 6)
    ctx.stroke()
    # chrome top strip + piano keys
    rrect(ctx, x0 + 30, y0 + 16, BB_W - 60, 44, 12)
    _g(ctx, 0.9)
    ctx.fill_preserve()
    _ink(ctx, 2.5)
    ctx.stroke()
    for i in range(7):
        kx = -170 + i * 50
        rrect(ctx, kx, y0 - 12, 42, 30, 5)
        _g(ctx, 0.25 if i != 2 else 0.75)
        ctx.fill_preserve()
        _ink(ctx, 2)
        ctx.stroke()
    # speakers
    speaker(ctx, *SPK_L, SPK_R0, pulse=level, t=t)
    speaker(ctx, *SPK_R, SPK_R0, pulse=level, t=t)
    # centre panel: tuning scale, cassette deck, VU meters
    rrect(ctx, -95, y0 + 80, 190, 330, 10)
    _g(ctx, 0.3)
    ctx.fill_preserve()
    _ink(ctx, 3)
    ctx.stroke()
    rrect(ctx, -80, y0 + 94, 160, 60, 5)
    _g(ctx, 0.9)
    ctx.fill_preserve()
    _ink(ctx, 2)
    ctx.stroke()
    ctx.select_font_face("Nimbus Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(12)
    for i, s in enumerate(("88", "92", "96", "100", "104", "108")):
        ctx.move_to(-72 + i * 27, y0 + 112)
        ctx.show_text(s)
        ctx.move_to(-66 + i * 27, y0 + 118)
        ctx.line_to(-66 + i * 27, y0 + 128)
    ctx.stroke()
    _ink(ctx, 3)
    ctx.move_to(-10, y0 + 100)          # the tuning needle
    ctx.line_to(-10, y0 + 148)
    ctx.stroke()
    ctx.set_font_size(11)
    ctx.move_to(-70, y0 + 146)
    ctx.show_text("FM  STEREO")
    # VU meters with needles jumping to the forecast
    for i, vx_ in enumerate((-42, 42)):
        cx, cy = vx_, y0 + 205
        rrect(ctx, cx - 36, cy - 28, 72, 50, 6)
        _g(ctx, 0.92)
        ctx.fill_preserve()
        _ink(ctx, 2)
        ctx.stroke()
        ctx.arc(cx, cy + 16, 34, math.radians(-150), math.radians(-30))
        ctx.stroke()
        a = math.radians(-140 + 110 * min(1.0, level * (0.85 + 0.3 * i) + 0.05 * math.sin(t * 31 + i)))
        _ink(ctx, 2.2)
        ctx.move_to(cx, cy + 16)
        ctx.line_to(cx + 36 * math.cos(a), cy + 16 + 36 * math.sin(a))
        ctx.stroke()
    # cassette window with two reels
    rrect(ctx, -78, y0 + 250, 156, 110, 8)
    _g(ctx, 0.12)
    ctx.fill_preserve()
    _ink(ctx, 2.5)
    ctx.stroke()
    rrect(ctx, -66, y0 + 262, 132, 70, 6)
    _g(ctx, 0.75)
    ctx.fill()
    for rx in (-34, 34):
        ctx.arc(rx, y0 + 297, 18, 0, 2 * math.pi)
        _g(ctx, 0.2)
        ctx.fill()
        for k in range(6):
            a = t * 3.0 + k * math.pi / 3
            _g(ctx, 0.85)
            ctx.set_line_width(3)
            ctx.move_to(rx + 5 * math.cos(a), y0 + 297 + 5 * math.sin(a))
            ctx.line_to(rx + 14 * math.cos(a), y0 + 297 + 14 * math.sin(a))
            ctx.stroke()
    # brand plate
    ctx.set_font_size(22)
    _g(ctx, 0.95)
    ctx.move_to(-62, y0 + 395)
    ctx.show_text("SOUNDKING")
    # mud: blotches spattered on the lower body (the comic's spotted look)
    ctx.save()
    rrect(ctx, x0, y0, BB_W, BB_H, 40)
    ctx.clip()
    for i in range(int(46 * mud)):
        px = rng.uniform(x0, x0 + BB_W)
        py = y0 + BB_H * (0.35 + 0.75 * rng.random() ** 0.6)
        rr = rng.uniform(6, 26)
        _g(ctx, 0.08)
        ctx.save()
        ctx.translate(px, py)
        ctx.rotate(rng.uniform(0, 3))
        ctx.scale(1.0, rng.uniform(0.5, 1.0))
        ctx.arc(0, 0, rr, 0, 2 * math.pi)
        ctx.restore()
        ctx.fill()
        # drips
        if rng.random() < 0.35:
            ctx.set_line_width(rr * 0.35)
            ctx.move_to(px, py)
            ctx.line_to(px + rng.uniform(-3, 3), py + rr * rng.uniform(1.5, 4))
            ctx.stroke()
    # wet sheen: pure-white highlight streaks along the top edge
    if wet > 0:
        _g(ctx, 1.0, min(1.0, wet))
        ctx.set_line_width(5)
        for i in range(9):
            hx = x0 + 60 + i * 95 + 20 * math.sin(i * 3.1)
            ctx.move_to(hx, y0 + 70)
            ctx.line_to(hx + 4, y0 + 70 + 30 + 25 * ((i * 7) % 3))
        ctx.stroke()
    ctx.restore()


def radio(ctx, t, level=0.0, led=True):
    """a Ranger's handheld radio (walkie-talkie), centre (0,0), 100 x 260"""
    # antenna
    _ink(ctx, 14)
    ctx.move_to(-22, -130)
    ctx.line_to(-22, -250)
    ctx.stroke()
    _g(ctx, 0.5)
    ctx.set_line_width(5)
    ctx.move_to(-24, -135)
    ctx.line_to(-24, -245)
    ctx.stroke()
    # knobs
    for kx in (8, 34):
        rrect(ctx, kx - 9, -150, 18, 26, 4)
        _g(ctx, 0.15)
        ctx.fill()
    # body
    rrect(ctx, -50, -130, 100, 260, 18)
    ctx.set_source(lin_grad(-50, 0, 50, 0, [(0, (0.55, 0.55, 0.55)), (0.35, (0.35, 0.35, 0.35)), (1, (0.12, 0.12, 0.12))]))
    ctx.fill_preserve()
    _ink(ctx, 4)
    ctx.stroke()
    # speaker grille slots
    _ink(ctx, 5)
    for i in range(6):
        ctx.move_to(-32, -86 + i * 16)
        ctx.line_to(32, -86 + i * 16)
    ctx.stroke()
    # PTT button on the side, LED (pure white when transmitting)
    rrect(ctx, -60, -40, 12, 60, 4)
    _g(ctx, 0.1)
    ctx.fill()
    if led:
        ctx.arc(30, -112, 6, 0, 2 * math.pi)
        _g(ctx, 1.0 if level > 0.1 else 0.3)
        ctx.fill()
    # little screen
    rrect(ctx, -30, 20, 60, 34, 4)
    _g(ctx, 0.85)
    ctx.fill_preserve()
    _ink(ctx, 2)
    ctx.stroke()
    ctx.select_font_face("Nimbus Mono PS", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(15)
    ctx.move_to(-24, 43)
    ctx.show_text("CH 1")

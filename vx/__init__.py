"""vx - the Vexillomantic animation engine for THE UNBABELING.
Scenes typically do:  from vx import *
"""
import os as _os

# 15 agents share 32 threads: keep per-process thread pools small (renders parallelise by process).
for _k, _v in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
               ("NUMBA_NUM_THREADS", "4")):
    _os.environ.setdefault(_k, _v)

import math

import cairo
import numpy as np

from .config import W, H, FPS, SR, PAL, C, ROOT, OUT, CACHE, AUDIO, ASSETS, FONT_SANS, FONT_SERIF, FONT_MONO, hex2rgb
from .ease import (clamp, lerp, lerp_c, remap, smoothstep, smootherstep, ease_in, ease_out, ease_in_out,
                   ease_in_out_sine, ease_out_back, ease_out_elastic, seg, spring, pulse, hash01, noise1, fbm1,
                   shake, value_noise2, fbm2)
from .canvas import (Canvas, col, set_color, mix, shade, surface_from_array, paint_array, over, over_alpha, screen,
                     add, multiply, solid, blur, glow, gradient_v, radial_mask, warp_affine, rect, rrect, circle,
                     ellipse, poly, smooth, lin_grad, rad_grad, text, text_width)
from .timeline import get_timeline
from .camera import Camera

"""2D camera: world point (x, y) appears at screen centre, scaled by zoom, rotated by rot (radians)."""
import math
from dataclasses import dataclass

from .config import W, H


@dataclass
class Camera:
    x: float = W / 2
    y: float = H / 2
    zoom: float = 1.0
    rot: float = 0.0

    def apply(self, ctx, parallax=1.0):
        """push camera transform on a (design-scaled) cairo ctx. parallax<1 = distant layer moves less.
        Use ctx.save() before / ctx.restore() after."""
        z = 1 + (self.zoom - 1) * parallax
        cx = W / 2 + (self.x - W / 2) * parallax
        cy = H / 2 + (self.y - H / 2) * parallax
        ctx.translate(W / 2, H / 2)
        ctx.rotate(self.rot * parallax)
        ctx.scale(z, z)
        ctx.translate(-cx, -cy)

    def to_screen(self, px, py, parallax=1.0):
        z = 1 + (self.zoom - 1) * parallax
        cx = W / 2 + (self.x - W / 2) * parallax
        cy = H / 2 + (self.y - H / 2) * parallax
        dx, dy = (px - cx) * z, (py - cy) * z
        c, s = math.cos(self.rot * parallax), math.sin(self.rot * parallax)
        return W / 2 + dx * c - dy * s, H / 2 + dx * s + dy * c

    def to_world(self, sx, sy):
        dx, dy = sx - W / 2, sy - H / 2
        c, s = math.cos(-self.rot), math.sin(-self.rot)
        dx, dy = dx * c - dy * s, dx * s + dy * c
        return self.x + dx / self.zoom, self.y + dy / self.zoom

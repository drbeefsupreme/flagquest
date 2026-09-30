"""vx.mosaic - OPVS SCHISMATICVM (film 3): every image is set in tesserae. Owner: Tessellator.

THE ONE RULE: the Flag is the only thing that is NOT made of tesserae. Never pass flag cloth through this module;
composite Flags (vx.flag) LAST, over the mosaic.

======================================================================================================== PIPELINE
    cart = Canvas(fc, bg=mz.SMALTI["lapis"])        # 1. the CARTOON: the mosaicist's full-size design (cairo),
    ...figures, tituli, gold ground...                #    flat smalti colours, dark contour strokes ~1.1 tile wide
    lay = mz.Layout.flow(guide, tile=12, fine=face_mask, gold=gold_mask, ground=gold_mask)   # 2. FIXED layout
                                                      #    (setup(); cached on disk automatically)
    img = mz.render(fc, cart.rgb(), lay, light=mz.candles(fc.T, [(x, y, z)]))                # 3. set the tiles
    img = flag_track.draw(...) over img               # 4. Flags LAST, whole, never tessellated

Coordinates: design units (1920x1080 frame). A layout lives in a 2D CHART = its wall's own design units (a full
frame by default, or a wall bigger than the frame, or a dome/conch chart). World 3D: X right, Y DOWN, Z toward
the viewer; the default flat wall is the plane z = 0 with chart (x, y). Everything is deterministic and a pure
function of (fc, state): the same layout gives the same tiles every frame, so the wall never boils.
Colours: sRGB 0..1 in and out; lighting is computed in linear light inside.
Render cost (1080p, 1 core, measured on the shared box): a 20k-tile wall 0.08-0.15 s; zoom 4 on it 0.12-0.25 s;
160k tiles all visible 0.19 s; 666k tiles through a Camera 0.6 s; dome/conch views +30%; 30-50k loose tumbling
tiles 0.3-0.6 s. --scale 0.5 renders the same tiles (they scale with the frame) at ~1/3 the cost.

======================================================================================================== PALETTE
SMALTI {name: '#hex'}, SC {name: np.float32 rgb}: lapis lapis_d lapis_l night_vault star porphyry porphyry_l tyrian
  malachite verdigris terracotta ochre_red marble marble_d bone flesh flesh_d flesh_l flesh_s ink slate silver gold
  gold_d gold_l mortar mortar_l plaster sinopia turquoise sea emerald olive leaf rose coral wine violet amethyst
  white grey umber sienna ochre hair bed bed_d.
Gold is browner/deeper than flag yellow (#FFC41A) so plain Flags always read cleaner and brighter.
Materials (Layout.mat / Tiles.mat / render(mat=)): GLASS 0 (smalti), GOLD 1 (leaf under glass), SILVER 2,
  STONE 3 (matte marble/limestone - flesh, whites), PEARL 4 (mother-of-pearl, iridescent), SCREEN 5 (a glowing
  phone screen with its own scrolling feed - the slop). Gold/silver/pearl take only the VALUE of the cartoon
  (paint SMALTI gold = normal leaf, gold_d = shadowed leaf, gold_l = bright leaf); glass/stone take its colour.

======================================================================================================== LAYOUTS
Layout.flow(guide, tile=12, seed=0, extent=(W,H), edges=None, fine=None, fine_tile=0.55*tile, *, gold=, silver=,
            pearl=, stone=, ground=None, ground_style='rows'|'fan'|'echo', echo=2, ground_angle=0, fan_center=None,
            density=None, alpha=None, lenvar=0.22, irregular=0.05, joint=1.0, groups=None, cache=True)
    ANDAMENTO. guide = the cartoon (float rgb/rgba, any resolution covering `extent` design units), an int label
    image, or an icon.cartoon() dict (rgb/fine/gold/silver/pearl/edges/ground/extent are taken from it; its alpha
    is NOT - pass alpha=d['alpha'] to tile only the figures, e.g. for render_layer).
    Colour boundaries (+ explicit `edges`, + the borders of fine/ground/gold/alpha masks) are contours; courses of
    tesserae run parallel to them: course 0 hugs every contour on both sides (a figure's dark outline course, the
    halo course around it in the ground), courses 1, 2, ... echo outward. Strokes thinner than a tile get one
    course of narrower tiles down their middle. Tiles are hand-cut convex polygons (irregular corners, varying
    length); where courses collide they are nipped to fit; gaps get small filler pieces.
    fine: mask where opus vermiculatum (smaller tiles, fine_tile) is laid - faces, hands, feet, eyes, letters.
    ground: the background region: `echo` halo courses around figures, then straight rows (ground_style 'rows',
        at ground_angle) / concentric rings about fan_center ('fan') / keep echoing ('echo', opus musivum).
    gold/silver/pearl/stone: masks -> material per tile (can also be given at render time).
    density: fn(x, y) -> physical units per chart unit (e.g. conch.k, dome.k) so tiles keep a constant physical
        size on curved surfaces.  alpha: only lay tiles where alpha >= 0.5 (a layer/figure).
    groups: int label image -> lay.grp per tile (e.g. Titulus.draw_labels for letter-by-letter animation).
    Masks: bool or float (>= 0.5 = on), any resolution covering the same extent. Setup-time: 0.3-3 s per frame of
    extent (cached in <film>/cache/mosaic/ keyed by the inputs; cache=False to rebuild).
    Cartoon conventions: contour strokes 1.0-1.3 x tile wide become ONE dark outline course; fold/feature lines
    0.8-1.0 x tile; in fine regions >= 1 fine tile. Tituli: fine_tile ~ 0.085 * cap height.
Layout.grid(tile=14, rect=None, ang=0, jitter=0.12, seed=0, extent=(W,H), brick=True)   straight courses
Layout.rings(cx, cy, r0, r1, tile=12, seed=0, extent=(W,H))                             concentric courses
Layout(xy, ang, size, seed=0, extent=(W,H), poly=None, nv=None, **arrays)   from centres (square tiles 0.9*size)
Layout.from_polys(poly (N,K,2), nv (N,), extent, seed)                    from your own polygons
lay.xy (N,2) centroid, .ang, .size, .poly (N,8,2) + .nv, .tilt (N,2), .jit (N,3), .rnd (N,), .mat (N,),
    .lev (N,) course index (0 = outline course; -1 ground rows; -2 fillers), .chain (N,) course id,
    .parent (N,) (split), .grp (N,), .area (N,), .extent
lay.keep(mask | fn(x,y)) / lay.subset(idx) / lay + lay2 / lay.copy() / len(lay) / lay.bounds()
lay.set_material(GOLD, mask=| fn=| idx=) ; lay.save(path) / Layout.load(path)
lay.split(levels=1, mask=None) -> Layout: each selected tile splits into 4 along its own axes ("may it schism
    endlessly"); children keep colour/material/course; .parent maps each child to its tile in `lay`.
lay.tiles(rgb, **overrides) -> Tiles (for render_tiles).

======================================================================================================== RENDERING
render(fc, cartoon, lay, gold=None, light=None, grout=1.3, mortar="mortar", var=0.06, sparkle=1.0, bevel=1.0,
       alpha=None, s=None, return_alpha=False, *, view=None, surface=None, rgb=None, mat=None, silver=None,
       pearl=None, stone=None, screen=None, present=None, gain=None, emit=None, offset=None, lift=None, rot=None,
       joint_emit=None, joint_img=None, bg=None, holes="bed", crect=None, melts=1.0, ambient=None, exposure=1.0,
       env=1.0, hdr=False, t=None, sample_mode="medoid", shadows=True, screen_gain=1.0, painted=0.5)
       -> float32 (h, w, 3) sRGB
  cartoon: image covering lay.extent at ANY scale (tiles are single-coloured: ~4 px per tile is plenty; s=0.5 for
      tile >= 10 regardless of zoom) - or crect=(x0,y0,x1,y1): the image covers only that chart rectangle (paint
      just the visible window). Each tessera takes the most typical of 5 samples inside it (sample_mode 'medoid')
      -> one real colour per tile, quantised to a limited set of MELTS (melts=1; 0 = off; >1 coarser) with dithered
      transitions, + per-tile batch variation (var).
  view: None -> the layout extent at scale s (=fc.s): the v0 behaviour (output (extent*s) px);
        2x3 affine M (chart -> screen design units) | 3x3 homography | View(cx, cy, zoom, rot) | Camera(...)
        (+ surface=). Tiles are rasterised per frame as polygons in screen px: crisp at any zoom, off-screen
        tiles culled, the cartoon is still sampled in chart space.
  light: None (default basilica rig) | v0 tuple from candle() | Light | [Light, ...]; ambient=float|rgb.
  gold/silver/pearl/stone/screen: chart masks (any res) or per-tile (N,) fractions -> materials (else lay.mat).
  rgb=(N,3): per-tile sRGB colours (skip sampling; e.g. from sample() once, recoloured per frame).
  present=(N,) codes: >= 0.5 (True) tile | 0 (False) SOCKET: the setting bed with the tile's imprint (lighter lime
      ridges between the pits) | -1 bare: no bed either -> `bg` shows (plaster + sinopia). holes='bg': every
      absent tile shows bg. gain=(N,) multiplies the light on a tile (landing flashes, snuff waves, twinkles).
  emit: (N,3) linear emissive per tile, or an image in cartoon space ((h,w) mask -> white, (h,w,3) colour),
      unaffected by lights (glowing eyes, flames). joint_emit=(N,3): the mortar around those tiles glows (light
      leaking through the joints); joint_img=(h,w,3) screen-space joint glow.
  offset=(N,2) chart units: tiles shift but stay bedded (trembling, cracks opening).
  lift=(N,) toward the viewer / rot=(N,3,3) or axis-angle (N,3) about the tile centre in its local frame
      (x = chart +u, y = chart +v, z = surface normal): lifted/rotated tiles become LOOSE tiles (depth-sorted,
      soft shadow on the wall, back faces show the back of the tessera: its glass dirtied with mortar) and their
      socket shows.
      rot_x(a) / rot_y(a) / rot_z(a) / axis_angle(v) build (N,3,3) arrays.
  bg: sRGB image (fc.h, fc.w, 3) or colour shown where there is no tile and no bed.
  mortar: joint colour; the mortar is lit only by its neighbouring tiles' light (unlit mortar stays dark) and is
      darker in narrow joints. painted=0.5: how much the setting bed under the joints was painted in the design's
      colours (each joint takes a dark version of its neighbour; ochre-red bole under gold); 0 = plain mortar.
      grout: joint width (design units; 1.3 = the layout's natural joints).
  sparkle: glint strength; bevel: rim relief; env: brightness of the environment reflected by gold/silver;
  exposure: linear multiplier; hdr: keep values > 1; return_alpha: premultiplied rgba (tiles + bed coverage) for
      painter's-order compositing of several surfaces.
render_tiles(fc, tiles, view=None, surface=None, light=None, loose=False, bg=None, return_alpha=False, t=None,
             **same look kwargs) -> sRGB
  Arbitrary per-frame tesserae with the same material look. view=None: tile xy are SCREEN design units.
  loose=False: bedded wall tiles (mortar joints between them; posed tiles pop out as in render());
  loose=True: free tiles composited over bg, depth-sorted, with shadows on the wall plane (flat views).
Tiles(xy, ang=0, size=12, rgb, gold=None, mat=None, z=None|lift=, rot=None, alpha=None, gain=None, emit=None,
      present=None, poly=None, nv=None, tilt=None, jit=None, rnd=None, ids=None, stretch=None, offset=None, seed=0)
  per-tile arrays (broadcast). Square tiles (side = size - natural joint) unless poly/nv given. z: height toward
  the camera (world z for flat views). stretch=(N,2) screen design units: motion smear. ids: stable identities
  for the random material properties when N/order change between frames.
render_layer(fc, cartoon_rgba, lay, M=None, **kw) -> premultiplied rgba (fc.h, fc.w, 4): a figure carrying its
  own tiles, local cartoon + local layout, placed by the 2x3 affine M (rasterised through M, crisp).
sample(cartoon, lay, crect=None, mode='medoid'|'mean'|'centre') -> (N,3) sRGB per tile
sample_mask(mask, lay, crect=None) -> (N,) fraction
tiles(fc, cartoon, lay) -> (xy, ang, size, rgb)  (v0)      draw_tiles(ctx, xy, ang, size, rgb, alpha, shade) (v0 cairo)
raster(lay, s) -> (label, edge_px, u, v) (v0 helper)       tile_colours(img, lab, n, alpha) (v0 helper)
smalti_colours(rgb, jit, rnd, melts, var) -> linear albedo;  srgb_to_lin / lin_to_srgb

======================================================================================================== LIGHT
Light(pos=(x,y,z) | dir=(x,y,z), color=(1,.86,.66), power=1, radius=700): point light (irradiance =
    power / (1 + (d/radius)^2)) or directional (dir points TO the light). World design units, z in front of the wall.
candles(t, [(x,y,z), ...], color, power=1.3, radius=520, flicker_amt=0.28, sway=6, seed=0) -> [Light]: flames that
    flicker and sway (the gold twinkles even with a locked camera).
candle(t, x=0.5, y=-0.35, flicker=0.35, seed=0) -> (lx, ly, intensity): v0 tuple = a flame over the frame at
    normalised screen coords (lx, ly), in front of the wall.   flicker(t, seed, amount) -> multiplier.
basilica(t, key=0.8, ambient=0.34) -> (lights, ambient): the default rig (soft warm clerestory key + ambient).
glow(fc, lights, view=None, radius=90, strength=0.5) -> additive (h,w,3) halo around point lights (the flames' glow
    in the air); img = np.clip(img + mz.glow(fc, L, view), 0, 1).
Gold/silver: each tessera is set at its own small tilt (+ the undulating setting bed), so it mirrors a different
  part of a dim basilica environment and the candles: the gold ground is mottled dark-brown..pale gold and it
  shimmers when the light or the camera moves (view-dependent: the eye comes from the view).

======================================================================================================== VIEWS
View(cx=960, cy=540, zoom=1, rot=0, at=(960,540), eye=None): chart point (cx,cy) at screen point `at`; zoom =
    screen units per chart unit; virtual eye at distance 1.3*W/zoom for gold and lifted-tile parallax.
View.affine(M) / View.homography(Hm); view.plane_to_screen(uv), view.screen_to_plane(xy),
    view.warp(fc, wall_img, s_img) -> screen image of a chart-space image (sinopia, plaster, masks).
Camera(eye, target, up=(0,-1,0), fov=50 (HORIZONTAL degrees), at=(960,540) lens shift, near=1)
Camera.frontal(cx, cy, zoom=1, dist=1500): perspective twin of View.   camera.ray(xy), camera.project3(P)
Surfaces (chart -> world; .embed(uv), .frame(uv) -> (P, T1, T2, N), .k(u, v) physical per chart unit,
          .intersect(o, D), .to_chart(P)):
  Plane(origin=(0,0,0), ex=(1,0,0), ey=(0,1,0))  flat wall anywhere (nave side walls: ex = (0,0,-1)).
  Conch(R, cx, cy, cz=0): apse semi-dome; chart = the front view (stereographic from (cx,cy,+R)), springing
      midpoint at chart (cx,cy), conch = upper half-disc of radius R; k = 2/(1+(rho/R)^2);
      Camera(eye=(cx,cy,R), target=(cx,cy,0), fov=90) sees the chart 1:1.
  Hemicycle(R, cx, cz=0, u0=pi*R/2): the half-cylinder wall below a conch; chart u = u0 + arc length, v = world y.
  Dome(R, center=(0,0,0), chart_center=(2R,2R)): seen from below; stereographic chart (scale 1 at the crown, the
      dome fills the disc of radius 2R), chart +u = world +X, +v = world +Z; k = 1/(1+(rho/2R)^2).
      dome_up(dome, height=R, fov=None, rot=0) -> Camera under the dome looking up (height=R: chart 1:1).
  Vault(R, cx, cy, z0=0, u0=pi*R/2): barrel vault along -Z; chart u = u0 + arc length across, v = depth.
  Several surfaces in one frame: render each with return_alpha=True and composite back to front.

======================================================================================================== TITULI
Titulus(text, x, y, size, color=lapis_d, font='roman', align='center', tracking=0.14, latin=True, sep=None,
        leading=1.45, dot_color=None, over_color=None)          size = CAP HEIGHT (design units), y = baseline
  fonts: 'roman' (VX Cinzel, Roman square capitals), 'roman_black' (heavier; best at 40-70 px), 'roman_light',
         'byzantine' (Marcellus), 'initial' (Cinzel Decorative).  latin=True: upper-case, U->V, J->I, W->VV.
  markup: [VXL] overline (abbreviations, e.g. VXL = VEXILLVM), · interpunct (sep='·' turns every space into
          one), + rosette, * square tessera, ~ hedera leaf, \\n new line. No crosses / Christian nomina sacra: the
          theology of the film is Vexillomantic.
  .draw(ctx, alpha=1, upto=None)   into the cartoon (upto = first n glyphs: letter-by-letter reveals)
  .draw_fine(ctx, margin=0.38)     white mask for Layout.flow(fine=...) (letters + their halo course)
  .draw_mask(ctx, margin=0)        exact letter mask;  .draw_labels(ctx) + labels_from(rgb) -> glyph index image
                                   for Layout.flow(groups=...) -> lay.grp = glyph index (-1 = none)
  .boxes [(x0,y0,x1,y1)] per glyph, .n glyph count, .width
titulus(ctx, text, x, y, size, ...) -> Titulus (drawn).  latinize(text).  Fonts: python -m vx.mosaic_type.

======================================================================================================== EVENTS (mz.fx)
Deterministic, pure functions of t (build in setup, query per frame); poses are dicts for render(**pose):
    fall = fx.Fall(lay, t_detach, floor=1150)              # setup
    img = mz.render(fc, cart, lay, present=codes, bg=sinopia_bg, **fall.pose(fc.t))
fx.Fall(lay | xy (N,2), t_detach (N,) (np.inf = stays), floor=<world y>, g=2400, bounce=0.34, friction=0.55, v0=None (N,3),
        spin=None (N,3), turns=2, heap=True, depth=(30, 320), kick=70, spread=60, bounces=4, seed=0)
    tiles pop off, tumble in 3D, hit the floor, bounce, rattle, rest on a growing heap (in a frontal View the floor
    is the line y=floor; look down at it with a Camera to see the heap). .pose(t) -> offset/lift/rot,
    .state(t) -> (pos (N,3), rot (N,3,3), phase), .fallen(t) -> bool, .rest() -> final (pos, rot),
    .events(T0) floor impacts (aggregated per frame), .detach_events(T0), .energy(n_frames, t_start).
fx.Fly(slots (N,2), src_pos (N,3), t0 (N,), dur=1.2, src_rot=None, arc=160, rise=0.25, turns=1, settle=0.18,
       glint=1.8, seed=0): the Unbabeling - tiles fly from e.g. Fall.rest() back to their slots, spin home, click in
       (a tiny overshoot into the wall) with a glint. .pose(t) -> offset/lift/rot/gain; .events(T0) arrival clicks.
fx.flip(t, t0 (N,), dur=0.3, axis='x', hop=0.18, size=12) -> dict(rot, lift, new): split-flap re-colour;
    render(..., rgb=np.where(f['new'][:, None], B, A), rot=f['rot'], lift=f['lift']).
fx.pop(t, t0, dur=0.5, height=14, wobble=0.25) -> dict(lift, rot)   tiles jump out and settle (a seal slam)
fx.tremble(t, n, amp=1.2, freq=17) -> dict(offset)                   loosening tiles, still bedded
fx.schism(child_lay, t, t0, dur=0.6, spread=0.35, twist=0.12) -> dict(offset, rot): a lay.split() layout's children
    drift apart from their parent's centre (t0 per parent or per child) - "may it schism endlessly".
fx.Crack(lay, start, direction=0, length=900, t0=0, speed=900, open=3.5, band=40, branch=0.18, wander=0.35, seed=0)
    a jagged branching crack revealed at `speed`; .pose(t) pushes tiles apart along the joints, .near(t, width) ->
    (N,) weight for joint_emit glow, .paths(t) polylines, .events(T0); fx.draw_cracks(ctx, paths) (cairo).
fx.wave_times(lay, origin (x,y) | ('x', x0) | ('y', y0), speed, t0=0, jitter=0) -> per-tile start times of a front.
fx.combine(pose_a, pose_b, ...) merges poses (offsets/lifts add, rotations compose, gains multiply).
fx.aggregate(times, strengths, xs, kind, desc, T0) -> foley events; fx.export(S, name, events) ->
    vx.foley.export_events (cache/foley/<scene>_<name>_events.json for the Foley department).
Slop screens: mat=SCREEN tiles show their own scrolling feed (render(t=fc.T) drives it; screen_gain brightness).
    A tessera turning into a screen: f = fx.flip(t, t0); render(..., mat=np.where(f['new'], mz.SCREEN, lay.mat),
    rot=f['rot'], lift=f['lift']) - or ramp gain/emit for a slow glow-up.
Foley event kinds from fx: 'tiles' (floor impacts, one event per frame, strength = sqrt of summed impact energy),
    'crack' (detaches, crack fronts), 'click' (tiles re-setting). Each event: t (GLOBAL s), strength 0..1, pan, desc.

======================================================================================================== SINOPIA
Sinopia(extent=(W,H), cartoon=None, edges=None, s=0.5, seed=0, grid=150, wash=0.35, lines=1.0,
        plaster='#D3CCBE', ink='#A0442C', cracks=5, line_w=3.2)
    bare lime plaster (mottled, trowel marks, pits, hairline cracks) with the red-ochre preparatory drawing derived
    from the cartoon's contours (one wobbly dry-brush line per contour, fainter pentimenti, shadow washes, a
    snapped-cord squaring grid). .image (h,w,3) chart-space sRGB at scale s.
    .view(fc, view=None, surface=None, light=None, ambient=None, lit=True) -> the plaster as seen through the view,
    lit by the same rig as the tiles -> pass as render(bg=...) with present codes -1 where the bed has gone.
irradiance(fc, view, surface, light, ambient) -> (h,w,3) linear light on a matte wall (for painted backgrounds).
remap_chart(fc, img, s_img, camera, surface) -> any chart-space image seen on a curved surface.
"""
import itertools
import math

import cv2
import numpy as np
from numba import njit as _njit_early

from .config import W, H
from .canvas import col
from . import mosaic_geo as _geo
from . import mosaic_ras as _ras
from .mosaic_view import View, Camera, Plane, Conch, Hemicycle, Dome, Vault, dome_up

KMAX = _geo.KMAX

# ============================================================ palette
# smalti + gold (non-flag). Gold is browner/deeper than flag yellow so the Flags always read cleaner and brighter.
SMALTI = {
    "lapis": "#1D2C6B", "lapis_d": "#101A45", "lapis_l": "#3D56A6", "night_vault": "#0E1633",
    "star": "#E9D9A6", "porphyry": "#6E1F2A", "porphyry_l": "#9B3440", "tyrian": "#5A1E4E",
    "malachite": "#1F5B45", "verdigris": "#3E8C7A", "terracotta": "#A4553A", "ochre_red": "#8E3B22",
    "marble": "#E8E1D3", "marble_d": "#B9AF9C", "bone": "#DCCFB4", "flesh": "#D9A98A", "flesh_d": "#A86F55",
    "ink": "#17131A", "slate": "#4B5160", "silver": "#C9CED6",
    "gold": "#B8862B", "gold_d": "#6E4A16", "gold_l": "#F3D27A",
    "mortar": "#4A4238", "mortar_l": "#7D7263", "plaster": "#D8CDB8", "sinopia": "#A0442C",
    # v1 additions
    "turquoise": "#2F7F86", "sea": "#224E6B", "emerald": "#1E6B4A", "olive": "#5E6B2E", "leaf": "#3E7A3A",
    "rose": "#C7747A", "coral": "#B8503E", "wine": "#4A1424", "violet": "#4B3470", "amethyst": "#6D4B8C",
    "white": "#F2EEE4", "grey": "#8A8680", "umber": "#4A3222", "sienna": "#8A4E2A", "ochre": "#B07A2E",
    "flesh_l": "#EBC9AE", "flesh_s": "#7E4E3A", "hair": "#3A2418", "bed": "#7E776B", "bed_d": "#5E574D",
}
SC = {k: np.array(col(v), np.float32) for k, v in SMALTI.items()}

# material codes (Layout.mat / Tiles.mat)
GLASS, GOLD, SILVER, STONE, PEARL, SCREEN = 0, 1, 2, 3, 4, 5
BACK, BED = _ras.M_BACK, _ras.M_BED

_TILT = np.array([0.045, 0.075, 0.07, 0.03, 0.06, 0.02, 0.05, 0.0], np.float32)   # normal spread per material (rad)
_KEYS = itertools.count(1)


def joint_of(t):
    """natural mortar joint (design units) for tesserae of size t."""
    return np.clip(0.09 * np.asarray(t, np.float32), 0.45, 2.2)


# ============================================================ colour science
def srgb_to_lin(c):
    return _ras.srgb_to_lin(c)


def lin_to_srgb(c):
    return _ras.lin_to_srgb(c)


def _oklab(lin):
    lin = np.maximum(np.asarray(lin, np.float32), 0)
    l = 0.4122214708 * lin[..., 0] + 0.5363325363 * lin[..., 1] + 0.0514459929 * lin[..., 2]
    m = 0.2119034982 * lin[..., 0] + 0.6806995451 * lin[..., 1] + 0.1073969566 * lin[..., 2]
    s = 0.0883024619 * lin[..., 0] + 0.2817188376 * lin[..., 1] + 0.6299787005 * lin[..., 2]
    l, m, s = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], -1).astype(np.float32)


def _oklab_inv(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return np.maximum(np.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                                -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                                -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], -1), 0).astype(np.float32)


def smalti_colours(rgb, jit=None, rnd=None, melts=1.0, var=0.06):
    """sRGB tile colours (N,3) -> linear smalti albedo: quantised to a limited set of melts (dithered in
    Oklab lightness so gradients break into mixed rows of neighbouring tones) + per-tile batch variation."""
    rgb = np.asarray(rgb, np.float32).reshape(-1, 3)
    lab = _oklab(srgb_to_lin(rgb))
    n = len(rgb)
    if melts and melts > 0:
        qL, qc = 0.05 * melts, 0.02 * melts
        d = (rnd - 0.5) * 0.85 if rnd is not None else 0.0
        lab[:, 0] = qL * np.floor(lab[:, 0] / qL + 0.5 + d)
        lab[:, 1:] = qc * np.round(lab[:, 1:] / qc)
    if var and jit is not None:
        lab[:, 0] += var * 0.42 * jit[:, 0]
        lab[:, 1] += var * 0.10 * jit[:, 1]
        lab[:, 2] += var * 0.10 * jit[:, 2]
    lab[:, 0] = np.clip(lab[:, 0], 0.0, 1.0)
    return _oklab_inv(lab)


_GOLD_LIN = np.array([0.86, 0.64, 0.25], np.float32)       # gold leaf under glass (linear reflectance)
_SILVER_LIN = np.array([0.88, 0.90, 0.93], np.float32)
_PEARL_LIN = np.array([0.92, 0.90, 0.87], np.float32)
_BACK_LIN = srgb_to_lin(np.array([0.52, 0.50, 0.47], np.float32))
_LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
_GOLD_REF = float(srgb_to_lin(SC["gold"]) @ _LUM)
_SILVER_REF = float(srgb_to_lin(SC["silver"]) @ _LUM)


# ============================================================ lights
class Light:
    """Point light (pos=(x,y,z) world design units, z in front of the wall) or directional light
    (dir=(x,y,z) pointing TO the light). color: sRGB tint; power: intensity; radius: distance at which a point
    light has fallen to half (irradiance = power / (1 + (d/radius)^2))."""

    def __init__(self, pos=None, dir=None, color=(1.0, 0.86, 0.66), power=1.0, radius=700.0):
        if pos is None and dir is None:
            dir = (-0.35, -0.6, 0.72)
        self.directional = dir is not None
        self.p = np.asarray(dir if self.directional else pos, np.float64)
        self.color = np.asarray(col(color), np.float32)
        self.power = float(power)
        self.radius = float(radius)

    def lin(self):
        return srgb_to_lin(self.color) * self.power

    def __repr__(self):
        return f"Light({'dir' if self.directional else 'pos'}={self.p.round(2).tolist()}, power={self.power:.2f})"


def candle(t, x=0.5, y=-0.35, flicker=0.35, seed=0):
    """v0 light tuple (lx, ly, intensity): a candle held over the wall, slowly wandering + flickering.
    lx, ly are normalised SCREEN coordinates of the flame (0..1 across/down the frame, negative y = above it)."""
    a = math.sin(t * 0.37 + seed) * 0.6 + math.sin(t * 0.13 + 2 * seed) * 0.4
    fl = 1.0 + flicker * (0.5 * math.sin(t * 11.3 + seed) + 0.3 * math.sin(t * 17.9) + 0.2 * math.sin(t * 7.1))
    return (x + 0.35 * a, y + 0.2 * math.cos(t * 0.29 + seed), max(0.3, fl))


def flicker(t, seed=0, amount=0.3):
    """deterministic candle flicker multiplier around 1."""
    s = seed * 1.7
    f = (0.45 * math.sin(t * 11.3 + s) + 0.25 * math.sin(t * 17.9 + 2 * s) + 0.2 * math.sin(t * 7.1 + 3 * s)
         + 0.1 * math.sin(t * 29.3 + s))
    return max(0.2, 1.0 + amount * f)


def candles(t, positions, color=(1.0, 0.9, 0.76), power=1.3, radius=520.0, flicker_amt=0.28, sway=6.0, seed=0):
    """point lights for candle flames at world positions [(x, y, z), ...]; each flickers and sways
    (sway = flame wander in design units -> the gold glints twinkle)."""
    out = []
    for i, p in enumerate(positions):
        sd = seed * 31 + i * 7
        dx = sway * (math.sin(t * 2.3 + sd) * 0.6 + math.sin(t * 5.7 + 2 * sd) * 0.4)
        dy = sway * 0.6 * math.sin(t * 3.1 + 3 * sd)
        out.append(Light(pos=(p[0] + dx, p[1] + dy, p[2]), color=color,
                         power=power * flicker(t, sd, flicker_amt), radius=radius))
    return out


def glow(fc, lights, view=None, radius=90.0, strength=0.5, core=0.18):
    """additive halo image (fc.h, fc.w, 3) around point lights seen through `view` (the candle flames' glow in the
    air, not on the wall): img = np.clip(img + mz.glow(fc, L, view), 0, 1). radius in design units at zoom 1."""
    h, w = fc.h, fc.w
    out = np.zeros((h, w, 3), np.float32)
    v = _as_view(view) or View()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for L in lights if isinstance(lights, (list, tuple)) else [lights]:
        if not isinstance(L, Light) or L.directional:
            continue
        p, dep = v.project3(L.p[None, :])
        if getattr(v, "kind", "") == "camera" and dep[0] <= 0:
            continue
        sx, sy = p[0] * fc.s
        z = getattr(v, "zoom", 1.0) if v.kind != "camera" else v.F / max(dep[0], 1e-3)
        r = radius * z * fc.s
        if sx < -4 * r or sy < -4 * r or sx > w + 4 * r or sy > h + 4 * r:
            continue
        d2 = ((xx - sx) ** 2 + (yy - sy) ** 2) / (r * r)
        f = strength * L.power * (np.exp(-d2) * 0.6 + np.exp(-d2 * 12.0) * core * 4.0 + 0.12 / (1.0 + d2 * 2.0))
        out += f[..., None] * L.color[None, None, :]
    return out


def basilica(t=0.0, key=0.8, ambient=0.34, warm=True):
    """the default light rig: a soft warm key from high front-left (the clerestory) + ambient. Returns
    (lights, ambient_rgb_linear)."""
    c = (1.0, 0.88, 0.70) if warm else (1.0, 1.0, 1.0)
    return [Light(dir=(-0.38, -0.62, 0.69), color=c, power=key)], np.float32(ambient) * srgb_to_lin(
        np.array((1.0, 0.9, 0.78) if warm else (1, 1, 1), np.float32))


def _lights(light, view, ambient):
    """normalise the `light` argument -> (list of Light, ambient linear rgb)."""
    if light is None:
        L, amb = basilica()
    elif isinstance(light, Light):
        L, amb = [light], None
    elif isinstance(light, (list, tuple)) and len(light) and isinstance(light[0], Light):
        L, amb = list(light), None
    elif isinstance(light, (list, tuple)) and len(light) == 3 and not isinstance(light[0], (list, tuple)):
        lx, ly, li = light      # v0 candle tuple: flame over the frame, in front of the wall
        sx, sy = lx * W, ly * H
        if view is not None and getattr(view, "kind", "") in ("affine", "homog"):
            p = view.screen_to_plane(np.array([sx, sy]))
            z = 0.85 * W / max(view.zoom, 1e-6)
        else:
            p, z = np.array([sx, sy]), 0.85 * W
        L = [Light(pos=(p[0], p[1], z), color=(1.0, 0.8, 0.55), power=1.25 * li, radius=1.1 * z),
             Light(dir=(-0.3, -0.5, 0.8), color=(1.0, 0.9, 0.75), power=0.25)]
        amb = None
    else:
        raise TypeError(f"light: expected None, v0 tuple, Light or [Light]; got {type(light)}")
    if ambient is not None:
        a = np.asarray(ambient, np.float32)
        amb = (a * srgb_to_lin(np.array((1.0, 0.9, 0.78), np.float32))) if a.ndim == 0 else srgb_to_lin(a)
    elif amb is None:
        amb = np.float32(0.30) * srgb_to_lin(np.array((1.0, 0.9, 0.78), np.float32))
    return L, np.asarray(amb, np.float32)


def _env(level):
    """environment reflected by gold/silver: a dim basilica - the lit nave in front of the wall (broad), a few
    lamps / windows / lit gold beyond (small lobes). R ~ +Z for a wall facing the viewer."""
    rng = np.random.default_rng(1234)
    J = 12
    d = np.zeros((J, 3))
    d[0] = (0.0, 0.18, 1.0)                         # the lit nave / floor in front of the wall
    d[1:, :2] = rng.normal(0, 0.42, (J - 1, 2))
    d[1:, 1] -= 0.08
    d[1:, 2] = 1.0
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    p = np.concatenate([[2.5], rng.uniform(35, 130, J - 1)])
    w = np.concatenate([[0.30], rng.uniform(0.12, 0.55, J - 1)])
    c = np.stack([np.ones(J), np.concatenate([[0.80], rng.uniform(0.74, 0.88, J - 1)]),
                  np.concatenate([[0.55], rng.uniform(0.45, 0.62, J - 1)])], 1) * w[:, None]
    base = np.array([0.062, 0.050, 0.034])
    return (d.astype(np.float32), (c * level).astype(np.float32), p.astype(np.float32),
            (base * level).astype(np.float32))


_ENV = {}


def _env_cached(level):
    k = round(float(level), 3)
    if k not in _ENV:
        _ENV[k] = _env(k)
    return _ENV[k]


# ============================================================ layouts
_ARRS = ("xy", "ang", "size", "poly", "nv", "tilt", "jit", "rnd", "mat", "lev", "chain", "parent", "grp", "area")


def _square_polys(xy, ang, side):
    n = len(xy)
    c, s = np.cos(ang), np.sin(ang)
    h = side * 0.5
    loc = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32)
    P = np.empty((n, KMAX, 2), np.float32)
    for k in range(4):
        lx, ly = loc[k, 0] * h, loc[k, 1] * h
        P[:, k, 0] = xy[:, 0] + c * lx - s * ly
        P[:, k, 1] = xy[:, 1] + s * lx + c * ly
    P[:, 4:] = P[:, 3:4]
    return P, np.full(n, 4, np.int8)


def _poly_centroid(P, nv):
    """vectorised centroids of padded convex polygons (N,K,2) with nv (N,) vertices."""
    P = np.asarray(P, np.float64)
    n, K = P.shape[:2]
    k = np.arange(K)[None, :]
    nxt = np.where(k + 1 < nv[:, None], k + 1, 0)
    valid = k < nv[:, None]
    x0, y0 = P[..., 0], P[..., 1]
    x1 = np.take_along_axis(x0, nxt, 1)
    y1 = np.take_along_axis(y0, nxt, 1)
    cr = np.where(valid, x0 * y1 - x1 * y0, 0.0)
    a = cr.sum(1)
    cx = ((x0 + x1) * cr).sum(1)
    cy = ((y0 + y1) * cr).sum(1)
    ok = np.abs(a) > 1e-12
    mx = np.where(valid, x0, 0).sum(1) / np.maximum(nv, 1)
    my = np.where(valid, y0, 0).sum(1) / np.maximum(nv, 1)
    safe = np.where(ok, a, 1.0)
    return np.stack([np.where(ok, cx / (3 * safe), mx), np.where(ok, cy / (3 * safe), my)], 1).astype(np.float32)


def _poly_area(P, nv):
    n = len(P)
    a = np.zeros(n, np.float64)
    for k in range(KMAX):
        j = (k + 1)
        valid = k < nv
        nxt = np.where(j < nv, j, 0)
        x0, y0 = P[:, k, 0], P[:, k, 1]
        x1 = P[np.arange(n), np.minimum(nxt, KMAX - 1), 0]
        y1 = P[np.arange(n), np.minimum(nxt, KMAX - 1), 1]
        a += np.where(valid, x0 * y1 - x1 * y0, 0.0)
    return np.abs(0.5 * a).astype(np.float32)


class Layout:
    """A FIXED set of tesserae in a 2D chart (design units; a full frame by default). Per tile:
      xy (N,2) centroid, ang (N,) course direction, size (N,) nominal edge, poly (N,8,2) + nv (N,) the hand-cut
      convex polygon, tilt (N,2) glass tilt, jit (N,3) colour jitter, rnd (N,) uniform, mat (N,) material code,
      lev (N,) course index (0 = outline course hugging a contour, 1,2.. echo courses, -1 ground rows, -2 fillers),
      chain (N,) course id, parent (N,) parent tile (split), grp (N,) group label (e.g. letter index), area (N,).
    The same layout gives the same tiles every frame, so the wall never boils."""

    def __init__(self, xy, ang, size, seed=0, extent=(W, H), poly=None, nv=None, **arrays):
        xy = np.ascontiguousarray(xy, np.float32).reshape(-1, 2)
        n = len(xy)
        self.xy = xy
        self.ang = np.ascontiguousarray(np.broadcast_to(np.asarray(ang, np.float32), (n,)))
        self.size = np.ascontiguousarray(np.broadcast_to(np.asarray(size, np.float32), (n,)))
        if poly is None:
            poly, nv = _square_polys(self.xy, self.ang, self.size * 0.9)
        self.poly = np.ascontiguousarray(poly, np.float32).reshape(n, KMAX, 2)
        self.nv = np.ascontiguousarray(nv, np.int8).reshape(n)
        self.extent = (int(round(extent[0])), int(round(extent[1])))
        self.seed = int(seed)
        rng = np.random.default_rng(int(seed) + 7919)
        self.tilt = rng.normal(0, 1, (n, 2)).astype(np.float32)
        self.jit = rng.normal(0, 1, (n, 3)).astype(np.float32)
        self.rnd = rng.random(n).astype(np.float32)
        self.mat = np.zeros(n, np.uint8)
        self.lev = np.full(n, -1, np.int16)
        self.chain = np.full(n, -1, np.int32)
        self.parent = np.full(n, -1, np.int32)
        self.grp = np.zeros(n, np.int32)
        self.area = None
        for k, v in arrays.items():
            if k not in _ARRS:
                raise TypeError(f"Layout: unknown per-tile array {k!r}")
            setattr(self, k, np.ascontiguousarray(v, getattr(self, k).dtype if getattr(self, k) is not None else None))
        if self.area is None:
            self.area = _poly_area(self.poly, self.nv)
        self.key = (next(_KEYS), n)

    # ---------------------------------------------------------------- basics
    def __len__(self):
        return len(self.xy)

    def __repr__(self):
        return f"Layout(n={len(self)}, extent={self.extent}, size~{np.median(self.size) if len(self) else 0:.1f})"

    def _new(self, idx=None, arrays=None):
        out = Layout.__new__(Layout)
        for k in _ARRS:
            v = getattr(self, k) if arrays is None else arrays[k]
            setattr(out, k, np.ascontiguousarray(v if idx is None else v[idx]))
        out.extent, out.seed = self.extent, self.seed
        out.key = (next(_KEYS), len(out.xy))
        return out

    def subset(self, idx):
        """tiles by index or boolean mask (keeps all per-tile arrays)."""
        return self._new(np.asarray(idx))

    def keep(self, fn_or_mask):
        """subset: fn(x, y) -> bool array, or a boolean image mask covering the extent (any resolution)."""
        if callable(fn_or_mask):
            k = np.asarray(fn_or_mask(self.xy[:, 0], self.xy[:, 1]), bool)
        else:
            k = sample_mask(fn_or_mask, self, mode="centre") >= 0.5
        return self.subset(k)

    def __add__(self, o):
        assert self.extent == o.extent, "Layout +: extents differ"
        arr = {k: np.concatenate([getattr(self, k), getattr(o, k)]) for k in _ARRS}
        arr["parent"] = np.concatenate([self.parent, np.where(o.parent >= 0, o.parent + len(self), -1)])
        return self._new(arrays=arr)

    def copy(self):
        return self._new(np.arange(len(self)))

    def bounds(self):
        v = self.poly
        return float(v[..., 0].min()), float(v[..., 1].min()), float(v[..., 0].max()), float(v[..., 1].max())

    def set_material(self, code, mask=None, fn=None, idx=None):
        """mark tiles as a material (GOLD, SILVER, STONE, PEARL, SCREEN) by image mask (>=0.5 at the tile),
        fn(x, y) -> bool, or index/boolean array. Returns self."""
        if idx is not None:
            sel = np.zeros(len(self), bool)
            sel[np.asarray(idx)] = True
        elif fn is not None:
            sel = np.asarray(fn(self.xy[:, 0], self.xy[:, 1]), bool)
        else:
            sel = sample_mask(mask, self) >= 0.5
        self.mat[sel] = code
        return self

    def tiles(self, rgb=None, **kw):
        """a Tiles bundle for render_tiles() from this layout (per-tile arrays can be overridden in kw)."""
        d = dict(xy=self.xy, ang=self.ang, size=self.size, poly=self.poly, nv=self.nv, tilt=self.tilt, jit=self.jit,
                 rnd=self.rnd, mat=self.mat, rgb=rgb if rgb is not None else (0.5, 0.5, 0.5))
        d.update(kw)
        return Tiles(**d)

    def save(self, path):
        np.savez_compressed(path, extent=np.array(self.extent), seed=self.seed,
                            **{k: getattr(self, k) for k in _ARRS})

    @staticmethod
    def load(path):
        z = np.load(path)
        out = Layout.__new__(Layout)
        for k in _ARRS:
            setattr(out, k, np.ascontiguousarray(z[k]))
        out.extent = tuple(int(v) for v in z["extent"])
        out.seed = int(z["seed"])
        out.key = (next(_KEYS), len(out.xy))
        return out

    # ---------------------------------------------------------------- transforms
    def split(self, levels=1, mask=None, joint=None, seed=None):
        """Recursive schism: every selected tile splits into 4 smaller tesserae along its own axes (children
        keep the parent's colour, material and course; `parent` maps each child to its tile in THIS layout).
        mask: image / fn(x,y) / bool array selecting which tiles split (others are kept whole)."""
        lay = self
        base_parent = np.arange(len(self), dtype=np.int32)
        for lv in range(int(levels)):
            if mask is None:
                sel = np.ones(len(lay), bool)
            elif callable(mask):
                sel = np.asarray(mask(lay.xy[:, 0], lay.xy[:, 1]), bool)
            elif np.ndim(mask) == 1:
                m = np.asarray(mask, bool)
                sel = m[base_parent] if len(m) == len(self) else m
            else:
                sel = sample_mask(mask, lay) >= 0.5
            idx = np.nonzero(sel)[0]
            if len(idx) == 0:
                break
            j = joint_of(lay.size[idx] * 0.5) if joint is None else np.full(len(idx), joint, np.float32)
            out = np.zeros((4 * len(idx), KMAX, 2), np.float64)
            onv = np.zeros(4 * len(idx), np.int32)
            _geo.split4(lay.poly[idx].astype(np.float64), lay.nv[idx].astype(np.int32),
                        lay.xy[idx].astype(np.float64), lay.ang[idx].astype(np.float64), j.astype(np.float64),
                        out, onv)
            ok = onv >= 3
            pidx = np.repeat(idx, 4)[ok]
            P = out[ok].astype(np.float32)
            nv = onv[ok].astype(np.int8)
            cxy = _poly_centroid(P, nv.astype(np.int64)) if len(P) else np.zeros((0, 2), np.float32)
            rng = np.random.default_rng((seed if seed is not None else lay.seed) * 31 + lv + 101)
            kids = {
                "xy": cxy, "ang": lay.ang[pidx], "size": lay.size[pidx] * 0.5, "poly": P, "nv": nv,
                "tilt": rng.normal(0, 1, (len(P), 2)).astype(np.float32),
                "jit": (0.6 * lay.jit[pidx] + 0.8 * rng.normal(0, 1, (len(P), 3))).astype(np.float32),
                "rnd": rng.random(len(P)).astype(np.float32), "mat": lay.mat[pidx], "lev": lay.lev[pidx],
                "chain": lay.chain[pidx], "parent": base_parent[pidx], "grp": lay.grp[pidx],
                "area": _poly_area(P, nv),
            }
            rest = np.nonzero(~sel)[0]
            arr = {k: np.concatenate([getattr(lay, k)[rest], kids[k]]) for k in _ARRS}
            arr["parent"] = np.concatenate([base_parent[rest], kids["parent"]])
            base_parent = arr["parent"].astype(np.int32)
            lay = lay._new(arrays=arr)
        return lay

    # ---------------------------------------------------------------- builders
    @staticmethod
    def grid(tile=14.0, rect=None, ang=0.0, jitter=0.12, seed=0, extent=(W, H), brick=True):
        """opus tessellatum: straight courses (brick-offset rows) of hand-cut tiles. rect=(x,y,w,h)."""
        from .mosaic_lay import rows_layout
        return rows_layout(tile, rect, ang, jitter, seed, extent, brick)

    @staticmethod
    def rings(cx, cy, r0, r1, tile=12.0, seed=0, extent=(W, H)):
        """concentric courses (halos, domes, rotae, the Tesseract's vault)."""
        from .mosaic_lay import rings_layout
        return rings_layout(cx, cy, r0, r1, tile, seed, extent)

    @staticmethod
    def flow(guide, tile=12.0, seed=0, extent=None, edges=None, fine=None, fine_tile=None, rows=True, *,
             gold=None, silver=None, pearl=None, stone=None, ground=None, ground_style="rows", echo=2,
             ground_angle=0.0, fan_center=None, density=None, alpha=None, lenvar=0.22, irregular=0.05,
             joint=1.0, groups=None, cache=True, **_ignored):
        """ANDAMENTO: courses of tesserae that follow the contours of the design (see MANUAL)."""
        from .mosaic_lay import flow_layout
        return flow_layout(guide, tile=tile, seed=seed, extent=extent, edges=edges, fine=fine, fine_tile=fine_tile,
                           rows=rows, gold=gold, silver=silver, pearl=pearl, stone=stone, ground=ground,
                           ground_style=ground_style, echo=echo, ground_angle=ground_angle, fan_center=fan_center,
                           density=density, alpha=alpha, lenvar=lenvar, irregular=irregular, joint=joint,
                           groups=groups, cache=cache)

    @staticmethod
    def from_polys(poly, nv, extent=(W, H), seed=0, size=None, ang=None, **arrays):
        poly = np.asarray(poly, np.float32)
        nv = np.asarray(nv, np.int8)
        cxy = _poly_centroid(poly, nv.astype(np.int64))
        area = _poly_area(poly, nv)
        if size is None:
            size = np.sqrt(area)
        if ang is None:
            e = poly[:, 1] - poly[:, 0]
            ang = np.arctan2(e[:, 1], e[:, 0])
        return Layout(cxy, ang, size, seed, extent, poly=poly, nv=nv, area=area, **arrays)


# ============================================================ Tiles (per-frame bundles for render_tiles)
class Tiles:
    """Per-frame tesserae for render_tiles(). All per-tile arguments broadcast to N.
    xy (N,2): centre (chart/world units of the view; SCREEN design units if view=None)
    ang, size: square tiles (side = size - natural joint) unless poly (N,K,2) + nv (N,) polygons are given
    rgb (N,3) sRGB smalti colour; gold (N,) 0..1 (>=0.5 -> gold leaf) or mat (N,) material codes
    z (alias lift) (N,): height toward the camera (along the surface normal); rot: (N,3,3) rotation or
        axis-angle (N,3) about the tile centre in its local frame (x = +u, y = +v, z = normal)
    alpha (N,) opacity; gain (N,) light multiplier; emit (N,3) linear emissive (unlit); present (N,) bool
    tilt (N,2), jit (N,3), rnd (N,): material randomness (default: hashed from ids = arange(N) -> stable per id)
    stretch (N,2): motion smear in screen design units (loose tiles)."""

    def __init__(self, xy, ang=0.0, size=12.0, rgb=(0.5, 0.5, 0.5), gold=None, mat=None, z=None, lift=None,
                 rot=None, alpha=None, gain=None, emit=None, present=None, poly=None, nv=None, tilt=None, jit=None,
                 rnd=None, ids=None, stretch=None, offset=None, seed=0):
        self.xy = np.ascontiguousarray(xy, np.float32).reshape(-1, 2)
        n = len(self.xy)
        bc = lambda v, dt, shp=(): np.ascontiguousarray(np.broadcast_to(np.asarray(v, dt), (n,) + shp))
        self.n = n
        self.ang = bc(ang, np.float32)
        self.size = bc(size, np.float32)
        self.rgb = bc(rgb, np.float32, (3,))
        if mat is not None:
            self.mat = bc(mat, np.uint8)
        else:
            self.mat = np.zeros(n, np.uint8)
            if gold is not None:
                self.mat[bc(gold, np.float32) >= 0.5] = GOLD
        if offset is not None:
            self.xy = self.xy + bc(offset, np.float32, (2,))
        if poly is None:
            side = self.size - joint_of(self.size)
            self.poly, self.nv = _square_polys(self.xy, self.ang, side)
        else:
            self.poly = np.ascontiguousarray(poly, np.float32).reshape(n, -1, 2)
            if self.poly.shape[1] < KMAX:
                pad = np.repeat(self.poly[:, -1:], KMAX - self.poly.shape[1], axis=1)
                self.poly = np.concatenate([self.poly, pad], 1)
            self.nv = bc(nv if nv is not None else self.poly.shape[1], np.int8)
            if offset is not None:
                self.poly = self.poly + bc(offset, np.float32, (2,))[:, None, :]
        zz = z if z is not None else lift
        self.z = None if zz is None else bc(zz, np.float32)
        self.rot = _rotations(rot, n)
        self.alpha = None if alpha is None else bc(alpha, np.float32)
        self.gain = None if gain is None else bc(gain, np.float32)
        self.emit = None if emit is None else bc(emit, np.float32, (3,))
        self.present = None if present is None else bc(present, np.float32)
        self.stretch = None if stretch is None else bc(stretch, np.float32, (2,))
        ids = np.arange(n) if ids is None else np.asarray(ids).astype(np.int64)
        if tilt is None or jit is None or rnd is None:
            h = _hash_normals(ids, seed)
            self.tilt = bc(tilt, np.float32, (2,)) if tilt is not None else h[:, :2]
            self.jit = bc(jit, np.float32, (3,)) if jit is not None else h[:, 2:5]
            self.rnd = bc(rnd, np.float32) if rnd is not None else _hash_uniform(ids, seed)
        else:
            self.tilt, self.jit, self.rnd = bc(tilt, np.float32, (2,)), bc(jit, np.float32, (3,)), bc(rnd, np.float32)


def _hash_uniform(ids, seed=0):
    x = (np.asarray(ids, np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(seed * 7919 + 1)) & np.uint64(
        0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(31)
    x = (x * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(29)
    return ((x >> np.uint64(11)).astype(np.float64) / float(1 << 53)).astype(np.float32)


def _hash_normals(ids, seed=0):
    ids = np.asarray(ids, np.int64)
    out = np.empty((len(ids), 6), np.float32)
    for k in range(3):
        u1 = np.clip(_hash_uniform(ids * 6 + 2 * k, seed), 1e-7, 1)
        u2 = _hash_uniform(ids * 6 + 2 * k + 1, seed)
        r = np.sqrt(-2 * np.log(u1))
        out[:, 2 * k] = r * np.cos(2 * np.pi * u2)
        out[:, 2 * k + 1] = r * np.sin(2 * np.pi * u2)
    return out


def _rotations(rot, n):
    if rot is None:
        return None
    r = np.asarray(rot, np.float32)
    if r.shape == (3, 3):
        return np.ascontiguousarray(np.broadcast_to(r, (n, 3, 3)))
    if r.ndim == 2 and r.shape[1] == 3:
        return axis_angle(r)
    return np.ascontiguousarray(r.reshape(n, 3, 3))


def axis_angle(v):
    """(N,3) axis-angle vectors -> (N,3,3) rotation matrices (Rodrigues)."""
    v = np.asarray(v, np.float64).reshape(-1, 3)
    th = np.linalg.norm(v, axis=1)
    k = v / np.maximum(th, 1e-12)[:, None]
    c, s = np.cos(th), np.sin(th)
    C = 1 - c
    x, y, z = k[:, 0], k[:, 1], k[:, 2]
    R = np.stack([np.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
                  np.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
                  np.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], 1)
    return R.astype(np.float32)


def rot_x(a):
    """(N,) angles -> (N,3,3) rotations about the tile's local x axis (split-flap flips, tilting forward)."""
    a = np.asarray(a, np.float64).reshape(-1)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], 1).astype(np.float32)


def rot_y(a):
    a = np.asarray(a, np.float64).reshape(-1)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], 1).astype(np.float32)


def rot_z(a):
    a = np.asarray(a, np.float64).reshape(-1)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], 1).astype(np.float32)


# ============================================================ sampling the cartoon
def _sample_points(lay_or_xy, ang=None, size=None, k=0.28):
    if isinstance(lay_or_xy, (Layout, Tiles)):
        xy, ang, size = lay_or_xy.xy, lay_or_xy.ang, lay_or_xy.size
    else:
        xy = lay_or_xy
    c, s = np.cos(ang), np.sin(ang)
    d = (k * size)[:, None]
    offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    pts = np.empty((len(xy), 5, 2), np.float32)
    for j, (a, b) in enumerate(offs):
        pts[:, j, 0] = xy[:, 0] + d[:, 0] * (a * c - b * s)
        pts[:, j, 1] = xy[:, 1] + d[:, 0] * (a * s + b * c)
    return pts


def _img_coords(pts, shape, extent, crect):
    gh, gw = shape[:2]
    x0, y0, x1, y1 = crect if crect is not None else (0, 0, extent[0], extent[1])
    ix = np.clip(((pts[..., 0] - x0) * (gw / (x1 - x0))).astype(np.int32), 0, gw - 1)
    iy = np.clip(((pts[..., 1] - y0) * (gh / (y1 - y0))).astype(np.int32), 0, gh - 1)
    return iy, ix


def sample(cartoon, lay, crect=None, mode="medoid", extent=None):
    """per-tile colour (N,3) from a cartoon image covering the layout extent (or the wall rectangle crect=(x0,y0,
    x1,y1)) at any resolution. mode 'medoid': the most typical of 5 samples inside the tile (crisp, one real
    colour per tessera, like a mosaicist choosing a smalto) | 'mean' (smoother under animation) | 'centre'."""
    img = np.asarray(cartoon, np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3:4]
        img = img[..., :3] / np.maximum(a, 1e-4)
    img = img[..., :3]
    ext = extent or lay.extent
    if mode == "centre":
        iy, ix = _img_coords(lay.xy, img.shape, ext, crect)
        return img[iy, ix].astype(np.float32)
    pts = _sample_points(lay)
    iy, ix = _img_coords(pts, img.shape, ext, crect)
    c = img[iy, ix]                                      # (N,5,3)
    if mode == "mean":
        return c.mean(1).astype(np.float32)
    d = np.abs(c[:, :, None, :] - c[:, None, :, :]).sum(-1).sum(-1)   # (N,5)
    j = np.argmin(d + np.array([0, 1e-4, 1e-4, 1e-4, 1e-4], np.float32), axis=1)
    return c[np.arange(len(c)), j].astype(np.float32)


def sample_mask(mask, lay, crect=None, mode="mean", extent=None):
    """per-tile fraction (N,) of a 2D mask (bool/float, any resolution covering the extent or crect)."""
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., -1] if m.shape[2] in (2, 4) else m[..., 0]
    m = m.astype(np.float32)
    ext = extent or lay.extent
    if mode == "centre":
        iy, ix = _img_coords(lay.xy, m.shape, ext, crect)
        return m[iy, ix]
    pts = _sample_points(lay)
    iy, ix = _img_coords(pts, m.shape, ext, crect)
    return m[iy, ix].mean(1).astype(np.float32)


# ============================================================ projection + shading core
def _is_plane_view(view):
    return view is None or getattr(view, "kind", "") in ("affine", "homog")


def _as_view(view, lay_extent=None):
    if view is None or isinstance(view, (View, Camera)):
        return view
    v = np.asarray(view, np.float64)
    if v.shape == (2, 3):
        return View.affine(v)
    if v.shape == (3, 3):
        return View.homography(v)
    raise TypeError("view: expected None, 2x3 affine, 3x3 homography, mz.View or mz.Camera")


class _Geo:
    """projected tiles: V (M,K,2) px, per-tile centre px, jacobian (M,2,2) px per chart unit, world P/N/T1/T2,
    depth, index into the source arrays."""


def _frame_wall(view, surface, s, poly, nv, cen, size, area, want):
    """project bedded (unposed) tiles. Returns _Geo restricted to tiles whose bbox touches the frame `want`
    (w, h px, margin)."""
    w, h, margin = want
    g = _Geo()
    n = len(cen)
    if view is None or view.kind == "affine":
        if view is None:
            A = np.eye(2) * s
            b = np.zeros(2)
        else:
            A = view.M[:, :2] * s
            b = view.M[:, 2] * s
        c = cen @ A.T + b
        # cull by centre + radius first (cheap)
        r = 0.75 * size * math.sqrt(abs(np.linalg.det(A))) + margin
        vis = (c[:, 0] > -r) & (c[:, 0] < w + r) & (c[:, 1] > -r) & (c[:, 1] < h + r)
        idx = np.nonzero(vis)[0]
        V = poly[idx] @ A.T.astype(np.float32) + b.astype(np.float32)
        g.idx, g.V, g.c = idx, V.astype(np.float32), c[idx].astype(np.float32)
        g.J = np.broadcast_to(A.astype(np.float32), (len(idx), 2, 2))
        g.P = np.concatenate([cen[idx], np.zeros((len(idx), 1), np.float32)], 1)
        g.N = np.broadcast_to(np.array([0, 0, 1], np.float32), (len(idx), 3))
        g.T1 = np.broadcast_to(np.array([1, 0, 0], np.float32), (len(idx), 3))
        g.T2 = np.broadcast_to(np.array([0, 1, 0], np.float32), (len(idx), 3))
        g.kscale = np.ones(len(idx), np.float32)
        return g
    if view.kind == "homog" or (view.kind == "camera" and (surface is None or isinstance(surface, Plane))):
        if view.kind == "homog":
            Hm = view.Hm
            surf = None
        else:
            surf = surface or Plane()
            A = np.stack([view.F * view.r + view.at[0] * view.f, view.F * view.d + view.at[1] * view.f, view.f])
            Hm = np.stack([A @ surf.ex, A @ surf.ey, A @ (surf.o - view.eye)], 1)
        S = np.diag([s, s, 1.0]) @ Hm
        u = cen[:, 0].astype(np.float64)
        v = cen[:, 1].astype(np.float64)
        X = S[0, 0] * u + S[0, 1] * v + S[0, 2]
        Y = S[1, 0] * u + S[1, 1] * v + S[1, 2]
        Wd = S[2, 0] * u + S[2, 1] * v + S[2, 2]
        if view.kind == "homog" and np.median(Wd) < 0:
            Wd, X, Y, S = -Wd, -X, -Y, -S
        near = getattr(view, "near", 1e-6) if view.kind == "camera" else 1e-9
        ok = Wd > near
        iw = 1.0 / np.where(ok, Wd, 1.0)
        cx, cy = X * iw, Y * iw
        J00 = (S[0, 0] - cx * S[2, 0]) * iw
        J01 = (S[0, 1] - cx * S[2, 1]) * iw
        J10 = (S[1, 0] - cy * S[2, 0]) * iw
        J11 = (S[1, 1] - cy * S[2, 1]) * iw
        sc = np.sqrt(np.abs(J00 * J11 - J01 * J10))
        r = 0.75 * size * sc + margin
        vis = ok & (cx > -r) & (cx < w + r) & (cy > -r) & (cy < h + r)
        if surf is not None:
            nrm = surf.nrm
            vis &= ((view.eye - surf.o) @ nrm - (u * (surf.ex @ nrm) + v * (surf.ey @ nrm))) > 0
        idx = np.nonzero(vis)[0]
        c = np.stack([cx[idx], cy[idx]], 1)
        J = np.stack([np.stack([J00[idx], J01[idx]], -1), np.stack([J10[idx], J11[idx]], -1)], 1)
        g.idx, g.c, g.J = idx, c.astype(np.float32), J.astype(np.float32)
        g.V = _verts_J(c, J, poly, cen, idx)
        m = len(idx)
        if surf is None:
            g.P = np.concatenate([cen[idx], np.zeros((m, 1), np.float32)], 1)
            g.N = np.broadcast_to(np.array([0, 0, 1], np.float32), (m, 3))
            g.T1 = np.broadcast_to(np.array([1, 0, 0], np.float32), (m, 3))
            g.T2 = np.broadcast_to(np.array([0, 1, 0], np.float32), (m, 3))
        else:
            g.P = (surf.o + u[idx, None] * surf.ex + v[idx, None] * surf.ey).astype(np.float32)
            g.N = np.broadcast_to(nrm.astype(np.float32), (m, 3))
            g.T1 = np.broadcast_to((surf.ex / np.linalg.norm(surf.ex)).astype(np.float32), (m, 3))
            g.T2 = np.broadcast_to((surf.ey / np.linalg.norm(surf.ey)).astype(np.float32), (m, 3))
            g.depth = Wd[idx]
        g.kscale = np.ones(m, np.float32)
        return g
    # camera + curved surface
    surf = surface
    e = 0.5
    cen64 = cen.astype(np.float64)
    P, T1, T2, N = surf.frame(cen64)
    # visible side + in front of the camera
    toeye = view.eye - P
    front = (toeye * N).sum(1) > 0
    sc0, d0 = view.project3(P)
    su, du = view.project3(surf.embed(cen64 + [e, 0.0]))
    sv, dv = view.project3(surf.embed(cen64 + [0.0, e]))
    near = getattr(view, "near", 1.0)
    ok = front & (d0 > near) & (du > near) & (dv > near)
    c = sc0 * s
    J00 = (su[:, 0] - sc0[:, 0]) * (s / e)
    J10 = (su[:, 1] - sc0[:, 1]) * (s / e)
    J01 = (sv[:, 0] - sc0[:, 0]) * (s / e)
    J11 = (sv[:, 1] - sc0[:, 1]) * (s / e)
    sc = np.sqrt(np.abs(J00 * J11 - J01 * J10))
    r = 0.75 * size * sc + margin
    vis = ok & (c[:, 0] > -r) & (c[:, 0] < w + r) & (c[:, 1] > -r) & (c[:, 1] < h + r)
    idx = np.nonzero(vis)[0]
    J = np.stack([np.stack([J00[idx], J01[idx]], -1), np.stack([J10[idx], J11[idx]], -1)], 1)
    g.idx, g.c, g.J = idx, c[idx].astype(np.float32), J.astype(np.float32)
    g.V = _verts_J(c[idx], J, poly, cen, idx)
    g.P, g.N = P[idx].astype(np.float32), N[idx].astype(np.float32)
    g.T1, g.T2 = T1[idx].astype(np.float32), T2[idx].astype(np.float32)
    g.kscale = surf.k(cen[idx, 0], cen[idx, 1]).astype(np.float32) if hasattr(surf, "k") else np.ones(len(idx))
    g.depth = d0[idx]
    return g


@_njit_early(cache=True)
def _verts_J_k(c, J, poly, cen, idx, out):
    for m in range(idx.shape[0]):
        i = idx[m]
        for k in range(poly.shape[1]):
            lx = poly[i, k, 0] - cen[i, 0]
            ly = poly[i, k, 1] - cen[i, 1]
            out[m, k, 0] = c[m, 0] + J[m, 0, 0] * lx + J[m, 0, 1] * ly
            out[m, k, 1] = c[m, 1] + J[m, 1, 0] * lx + J[m, 1, 1] * ly


def _verts_J(c, J, poly, cen, idx):
    out = np.empty((len(idx), poly.shape[1], 2), np.float32)
    _verts_J_k(np.ascontiguousarray(c, np.float64), np.ascontiguousarray(J, np.float64),
               np.ascontiguousarray(poly, np.float32), np.ascontiguousarray(cen, np.float32),
               np.ascontiguousarray(idx, np.int64), out)
    return out


def _frame_loose(view, surface, s, poly, nv, cen, lift, rot, world=False):
    """project posed tiles (3D). Tile local frame = surface frame at its centre (x=T1, y=T2, z=N).
    world=True: cen are world (x, y) with z=lift for flat views (render_tiles with free tiles)."""
    g = _Geo()
    n = len(cen)
    loc = (poly - cen[:, None, :]).astype(np.float64)                      # chart units
    if view is None or view.kind in ("affine", "homog") or (surface is None and world):
        P0 = np.concatenate([cen.astype(np.float64), np.zeros((n, 1))], 1)
        T1 = np.broadcast_to([1.0, 0, 0], (n, 3))
        T2 = np.broadcast_to([0, 1.0, 0], (n, 3))
        N = np.broadcast_to([0, 0, 1.0], (n, 3))
        k = np.ones(n)
    else:
        surf = surface or Plane()
        P0, T1, T2, N = surf.frame(cen.astype(np.float64))
        k = surf.k(cen[:, 0], cen[:, 1]) if hasattr(surf, "k") else np.ones(n)
    L3 = np.concatenate([loc * k[:, None, None], np.zeros(loc.shape[:2] + (1,))], -1)   # (n,K,3) local frame
    if rot is not None:
        L3 = np.einsum("nij,nkj->nki", rot.astype(np.float64), L3)
        nl = rot[:, :, 2].astype(np.float64)                                           # R @ (0,0,1)
    else:
        nl = np.broadcast_to([0, 0, 1.0], (n, 3))
    z = (lift if lift is not None else np.zeros(n)).astype(np.float64)
    Wv = (P0[:, None, :] + L3[..., 0:1] * T1[:, None, :] + L3[..., 1:2] * T2[:, None, :]
          + (L3[..., 2:3] + z[:, None, None]) * N[:, None, :])
    Pc = P0 + z[:, None] * N
    Nw = nl[:, 0:1] * T1 + nl[:, 1:2] * T2 + nl[:, 2:3] * N
    T1w = (rot[:, :, 0:1].astype(np.float64) * 0 + 0) if False else None
    if view is None:
        view_ = View()
        # screen design units: identity; eye above the frame centre
        scr, dep = view_.project3(Wv.reshape(-1, 3))
        scr_c, dep_c = view_.project3(Pc)
        eye = view_.eye
    else:
        scr, dep = view.project3(Wv.reshape(-1, 3))
        scr_c, dep_c = view.project3(Pc)
        eye = view.eye
    g.V = (scr.reshape(n, -1, 2) * s).astype(np.float32)
    g.c = (scr_c * s).astype(np.float32)
    g.depth = dep_c
    g.vdepth = dep.reshape(n, -1)
    g.P = Pc.astype(np.float32)
    g.N = Nw.astype(np.float32)
    # rotated tangents for tilt
    if rot is not None:
        t1l = rot[:, :, 0].astype(np.float64)
        t2l = rot[:, :, 1].astype(np.float64)
        g.T1 = (t1l[:, 0:1] * T1 + t1l[:, 1:2] * T2 + t1l[:, 2:3] * N).astype(np.float32)
        g.T2 = (t2l[:, 0:1] * T1 + t2l[:, 1:2] * T2 + t2l[:, 2:3] * N).astype(np.float32)
    else:
        g.T1, g.T2 = np.asarray(T1, np.float32), np.asarray(T2, np.float32)
    g.eye = eye
    g.idx = np.arange(n)
    return g


def _screen_dirs(view, surface, g, s, ltan, loose):
    """screen-space (px) unit direction of the tangential key light + u axis per tile (for bevels / screens)."""
    # express ltan in the tile's (T1, T2) basis and push through the jacobian (or re-project for loose tiles)
    a = np.einsum("ij,ij->i", ltan, g.T1)
    b = np.einsum("ij,ij->i", ltan, g.T2)
    if not loose and hasattr(g, "J"):
        d = np.einsum("mij,mj->mi", g.J, np.stack([a, b], -1))
        u = np.einsum("mij,mj->mi", g.J, np.stack([np.ones_like(a), np.zeros_like(a)], -1))
    else:
        # finite difference through the projection
        P = g.P.astype(np.float64)
        vw = view if view is not None else View()
        eps = 0.5
        s0, _ = vw.project3(P)
        s1, _ = vw.project3(P + eps * ltan.astype(np.float64))
        s2, _ = vw.project3(P + eps * g.T1.astype(np.float64))
        d = (s1 - s0) * (s / eps)
        u = (s2 - s0)
    mag = np.sqrt(a * a + b * b)
    dn = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(dn, 1e-9) * np.clip(mag * 1.25, 0.28, 1.0)[:, None]
    # frontal light: default relief from the upper left
    weak = dn[:, 0] < 1e-6
    d[weak] = np.array([-0.2, -0.25])
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    return d.astype(np.float32), u.astype(np.float32)


_UND = None


def _undulation(cen, amp=0.032):
    """slopes (N,2) of the uneven setting bed: the wall is never flat, so broad areas of gold tilt together and
    the ground shows soft bands of brighter / darker leaf. A fixed function of the chart position."""
    global _UND
    if _UND is None:
        r = np.random.default_rng(99)
        k = r.normal(0, 1, (7, 2))
        k /= np.linalg.norm(k, axis=1, keepdims=True)
        lam = r.uniform(220, 900, 7)
        _UND = (k * (2 * np.pi / lam)[:, None], r.uniform(0, 2 * np.pi, 7), r.uniform(0.5, 1.0, 7))
    kv, ph, a = _UND
    arg = cen.astype(np.float64) @ kv.T + ph                  # (N,7)
    d = np.cos(arg) * a                                       # derivative of sin
    sl = (d[:, :, None] * kv[None, :, :] / np.linalg.norm(kv, axis=1)[None, :, None]).sum(1)
    return (amp * sl / np.sqrt((a * a).sum() / 2)).astype(np.float32)


_UNDC = {}


def _lay_undulation(lay):
    """per-tile bed undulation of a layout (cached per layout)."""
    u = _UNDC.get(lay.key)
    if u is None:
        u = _undulation(lay.xy)
        if len(_UNDC) > 16:
            _UNDC.pop(next(iter(_UNDC)))
        _UNDC[lay.key] = u
    return u


def _und_cached(wall, ii):
    u = wall.get("und")
    if u is None:
        return _undulation(wall["cen"][ii])
    return u[ii]


def _tile_normals(g, tilt, mat, und=None):
    sig = _TILT[np.minimum(mat, len(_TILT) - 1)][:, None]
    tl = tilt if und is None else tilt * 1.0
    n = g.N + sig * (tl[:, 0:1] * g.T1 + tl[:, 1:2] * g.T2)
    if und is not None:
        n = n + und[:, 0:1] * g.T1 + und[:, 1:2] * g.T2
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)).astype(np.float32)


def _light_arrays(L):
    K = len(L)
    Lp = np.zeros((max(K, 1), 3), np.float32)
    Lc = np.zeros((max(K, 1), 3), np.float32)
    Lr = np.ones(max(K, 1), np.float32)
    Ld = np.zeros(max(K, 1), np.float32)
    for k, l in enumerate(L):
        Lp[k] = l.p
        Lc[k] = l.lin()
        Lr[k] = l.radius
        Ld[k] = 1.0 if l.directional else 0.0
    return Lp, Lc, Lr, Ld


def _shade(g, eye, alb, mat, gain, emit, rnd, tilt, lights, amb, sparkle, env, und=None):
    M = len(g.idx)
    Nt = _tile_normals(g, tilt, mat, und)
    Lp, Lc, Lr, Ld = _light_arrays(lights)
    ambl = float(np.mean(amb)) / 0.26
    envd, envc, envp, envb = _env_cached(env * max(ambl, 0.05))
    base = np.empty((M, 3), np.float32)
    spec = np.empty((M, 3), np.float32)
    rim = np.empty((M, 3), np.float32)
    irr = np.empty((M, 3), np.float32)
    ltan = np.empty((M, 3), np.float32)
    _ras.shade_tiles(np.ascontiguousarray(g.P, np.float32), Nt, np.asarray(eye, np.float32),
                     np.ascontiguousarray(alb, np.float32), np.ascontiguousarray(mat, np.int32),
                     np.ascontiguousarray(gain, np.float32), np.ascontiguousarray(emit, np.float32),
                     np.ascontiguousarray(rnd, np.float32), Lp, Lc, Lr, Ld, amb.astype(np.float32),
                     envd, envc, envp, envb, float(sparkle), base, spec, rim, irr, ltan)
    return base, spec, rim, irr, ltan


def _pixel_params(g, s, mat, size, area, bevel, kind):
    M = len(g.idx)
    if hasattr(g, "J") and not getattr(g, "loose", False):
        J = g.J
        apx = area * np.abs(J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])
    else:
        apx = _poly_area_px(g.V)
    hs = (0.5 * np.sqrt(np.maximum(apx, 1e-6))).astype(np.float32)
    tpx = 2 * hs
    bevw = np.clip(0.2 * tpx, 0.45, 14.0).astype(np.float32)
    bk = np.where(mat == STONE, 0.28, np.where((mat == GOLD) | (mat == SILVER), 0.42, 0.5)) * bevel
    bk = np.where(tpx < 3.0, bk * np.clip(tpx / 3.0, 0, 1), bk).astype(np.float32)
    dk = np.where(mat == STONE, 0.06, np.where((mat == GOLD) | (mat == SILVER), 0.10, 0.16)).astype(np.float32)
    bedx = (0.6 + 0.26 * tpx).astype(np.float32)
    return hs, bevw, bk, dk, bedx


def _poly_area_px(V, signed=False):
    x, y = V[..., 0].astype(np.float64), V[..., 1].astype(np.float64)
    a = 0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, 1)
    return a if signed else np.abs(a)


def _orient(V, nv):
    """reverse the vertex order of mirrored (negative-area) projected polygons (views looking up into a dome,
    reflected affines, tiles seen from behind) so the raster kernels always see positive orientation."""
    a = _poly_area_px(V, signed=True)
    neg = a < 0
    if not neg.any():
        return V
    V = V.copy()
    ii = np.nonzero(neg)[0]
    n = nv[ii].astype(np.int64)[:, None]
    k = np.arange(V.shape[1])[None, :]
    src = np.where(k < n, n - 1 - k, 0)
    V[ii] = np.take_along_axis(V[ii], np.repeat(src[:, :, None], 2, 2), 1)
    return V


def _bg_linear(bg, h, w):
    if bg is None:
        return None
    if isinstance(bg, np.ndarray) and bg.ndim == 3:
        img = np.ascontiguousarray(bg[..., :3], np.float32)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        out = np.empty_like(img)
        _ras.decode(out, np.clip(img, 0, 1))
        return out
    c = srgb_to_lin(np.asarray(col(bg), np.float32))
    return np.ascontiguousarray(np.broadcast_to(c, (h, w, 3)), np.float32)


_BUF = {}


def _buffers(h, w):
    k = (h, w)
    b = _BUF.get(k)
    if b is None:
        b = dict(acc=np.zeros((h, w, 3), np.float32), cov=np.zeros((h, w), np.float32),
                 bedd=np.full((h, w), _ras.BIG, np.float32), bedi=np.zeros((h, w), np.int32),
                 beda=np.zeros((h, w), np.float32), frame=np.zeros((h, w, 3), np.float32))
        _BUF.clear()
        _BUF[k] = b
    b["acc"].fill(0)
    b["cov"].fill(0)
    b["bedd"].fill(_ras.BIG)
    b["beda"].fill(0)
    return b


_DUMMY_LAB = np.zeros((1, 1), np.int32)


def _draw(fc_or_size, s, view, surface, *, wall=None, loose=None, lights=None, amb=None, bg=None, mortar="mortar",
          sparkle=1.0, bevel=1.0, env=1.0, exposure=1.0, hdr=False, return_alpha=False, t=0.0, joint_img=None,
          shadows=True, out_lin=False, lab=None, undulate=True, painted=0.5):
    """core: wall = dict of bedded tile arrays (chart units), loose = dict of posed tile arrays."""
    w, h = fc_or_size
    buf = _buffers(h, w)
    acc, cov, bedd, bedi, beda, frame = buf["acc"], buf["cov"], buf["bedd"], buf["bedi"], buf["beda"], buf["frame"]
    bgl = _bg_linear(bg, h, w)
    mort = srgb_to_lin(np.asarray(SC[mortar] if isinstance(mortar, str) and mortar in SC else col(mortar), np.float32))
    mirr = np.zeros((1, 3), np.float32)
    memit = np.zeros((1, 3), np.float32)
    aolo = np.full(1, 0.42, np.float32)
    eye = view.eye if view is not None else View().eye
    if wall is not None and len(wall["cen"]):
        g = _frame_wall(view, surface, s, wall["poly"], wall["nv"], wall["cen"], wall["size"], wall["area"],
                        (w, h, 4.0))
        if len(g.idx):
            ii = g.idx
            mat = wall["mat"][ii]
            alb = _alb(wall, ii)
            base, spec, rim, irr, ltan = _shade(g, eye, alb, mat, wall["gain"][ii], wall["emit"][ii],
                                                wall["rnd"][ii], wall["tilt"][ii], lights, amb, sparkle, env,
                                                _und_cached(wall, ii) if undulate else None)
            ldir, axu = _screen_dirs(view, surface, g, s, ltan, False)
            hs, bevw, bk, dk, bedx = _pixel_params(g, s, mat, wall["size"][ii], wall["area"][ii], bevel, None)
            kind = wall["kind"][ii].astype(np.int32)
            scr = kind == _ras.K_SCREEN
            if scr.any():
                base[scr] = wall["emit_screen"][ii][scr]
            # the setting bed was painted in the colours of the design before the tesserae were pressed in, so each
            # joint takes a dark version of its neighbour's colour (warm ochre-red under the gold)
            pb = float(painted)
            bedc = alb * 0.5
            gm = (mat == GOLD)
            if gm.any():
                bedc[gm] = _BOLE_LIN
            mfac = (1.0 - pb) + pb * bedc / np.maximum(mort, 1e-4)
            mirr = (irr * 0.9 * mfac * (0.86 + 0.28 * wall["rnd"][ii, None])).astype(np.float32)
            imp = kind == _ras.K_IMPRINT
            aolo = np.full(len(ii), 0.42, np.float32)
            if imp.any():      # around empty sockets the joints are raised lime ridges, lighter than the bed
                mirr[imp] = irr[imp] * (_BED_LIN * 1.12 / np.maximum(mort, 1e-4))
                aolo[imp] = 0.95
            memit = wall["jemit"][ii] + 0.12 * wall["emit"][ii]
            inset = wall["inset"][ii]
            alpha = wall["alpha"][ii]
            seed = wall["rnd"][ii]
            wl = lab is not None
            dj = g.J[:, 0, 0] * g.J[:, 1, 1] - g.J[:, 0, 1] * g.J[:, 1, 0]
            if (dj < 0).any():                                   # mirrored views (looking up into a dome, ...)
                g.V = _orient(g.V, wall["nv"][ii])
            _ras.wall_pass(acc, cov, bedd, bedi, beda, lab if wl else _DUMMY_LAB, g.V,
                           wall["nv"][ii].astype(np.int32), kind, base, spec, rim, ldir, bevw, bk, dk, g.c, hs, axu,
                           alpha, inset, bedx, seed.astype(np.float32), float(t), wl)
            if lab is not None:
                m = lab >= 0
                lab[m] = ii[lab[m]]
    has_bg = bgl is not None
    _ras.composite(frame, acc, cov, bedd, bedi, beda, bgl if has_bg else frame, mort, mirr, memit,
                   1.6 * max(s, 0.3), aolo, has_bg)
    if joint_img is not None:
        ji = np.asarray(joint_img, np.float32)
        jm = (bedd < 1e8) & (cov < 1)
        frame[jm] += ji[jm] * (1 - np.minimum(cov[jm], 1))[:, None]
    fa = None
    if return_alpha:
        fa = np.empty((h, w), np.float32)
        _ras.alpha_of(fa, cov, beda)
    if loose is not None and len(loose["cen"]):
        g = _frame_loose(view, surface, s, loose["poly"], loose["nv"], loose["cen"], loose["lift"], loose["rot"],
                         world=loose.get("world", False))
        g.loose = True
        # cull: behind camera / off screen
        V = g.V
        vis = (V[..., 0].max(1) > -2) & (V[..., 0].min(1) < w + 2) & (V[..., 1].max(1) > -2) & (V[..., 1].min(1) < h + 2)
        if getattr(view, "kind", "") == "camera":
            vis &= g.vdepth.min(1) > getattr(view, "near", 1.0)
        if vis.any():
            ii = np.nonzero(vis)[0]
            gg = _Geo()
            for k in ("V", "c", "depth", "P", "N", "T1", "T2"):
                setattr(gg, k, getattr(g, k)[ii])
            gg.idx = ii
            gg.loose = True
            mat = loose["mat"][ii].copy()
            # back faces: the back of the tessera (its own glass dirtied with mortar; gold shows the leaf dimly)
            vv = eye[None, :] - gg.P
            back = (vv * gg.N).sum(1) < 0
            metal_back = None
            if back.any():
                gg.N = gg.N.copy()
                gg.N[back] *= -1
                mb = mat[back]
                metal = (mb == GOLD) | (mb == SILVER)
                metal_back = np.zeros(len(ii), bool)
                metal_back[np.nonzero(back)[0][metal]] = True
                mat[back] = np.where(metal, mb, BACK)
            alb = _alb(loose, ii, mat)
            if metal_back is not None:
                alb[metal_back] *= 0.55
            base, spec, rim, irr, ltan = _shade(gg, eye, alb, mat, loose["gain"][ii], loose["emit"][ii],
                                                loose["rnd"][ii], loose["tilt"][ii], lights, amb, sparkle, env)
            ldir, axu = _screen_dirs(view, surface, gg, s, ltan, True)
            hs, bevw, bk, dk, _ = _pixel_params(gg, s, mat, loose["size"][ii], loose["area"][ii], bevel, None)
            kind = loose["kind"][ii].astype(np.int32)
            kind[back] = _ras.K_TILE
            scr = kind == _ras.K_SCREEN
            if scr.any():
                base[scr] = loose["emit_screen"][ii][scr]
            alpha = loose["alpha"][ii].copy()
            Vd = gg.V
            st = loose.get("stretch")
            if st is not None:
                sv = st[ii] * s
                d = np.einsum("mkj,mj->mk", Vd - gg.c[:, None, :], sv)
                Vd = Vd + np.where(d[..., None] >= 0, 0.5, -0.5) * sv[:, None, :]
                a0 = np.maximum(_poly_area_px(gg.V), 1e-3)
                alpha = alpha * np.clip(a0 / np.maximum(_poly_area_px(Vd), 1e-3), 0.25, 1.0)
                Vd = Vd.astype(np.float32)
            Vd = _orient(np.ascontiguousarray(Vd, np.float32), loose["nv"][ii])
            if shadows and _is_plane_view(view) and surface is None:
                z = loose["lift"][ii] if loose["lift"] is not None else np.zeros(len(ii))
                key = lights[0] if lights else None
                if key is not None and (z > 0.5).any():
                    if key.directional:
                        lvec = key.p / np.linalg.norm(key.p)
                        off = -z[:, None] * (lvec[:2] / max(lvec[2], 0.15))[None, :]
                    else:
                        lz = np.maximum(key.p[2] - z, 30.0)
                        off = -(z / lz)[:, None] * (key.p[None, :2] - gg.P[:, :2])
                    offs = _dir_to_screen(view, off, s)
                    soft = np.minimum(0.6 + 0.05 * z * s, 8.0 * max(s, 0.25)).astype(np.float32)
                    dark = (0.5 * np.exp(-z / 45.0) * np.clip(z / 3.0, 0, 1) * alpha * (z < 150)).astype(np.float32)
                    # shadow shape = the tile's own footprint on the wall
                    _ras.shadow_pass(frame, Vd, loose["nv"][ii].astype(np.int32), offs.astype(np.float32), soft, dark)
            order = np.argsort(-gg.depth, kind="stable").astype(np.int64)
            _ras.loose_pass(frame, fa if fa is not None else _DUMMY_LAB.astype(np.float32), Vd,
                            loose["nv"][ii].astype(np.int32), order, kind, base, spec, rim, ldir, bevw, bk, dk, gg.c,
                            hs, axu, alpha, loose["rnd"][ii].astype(np.float32), float(t), fa is not None)
    if out_lin:
        return frame
    out = np.empty((h, w, 3), np.float32)
    if return_alpha and bg is None:
        # frame is premultiplied in LINEAR light: un-premultiply, encode, premultiply in sRGB
        a = fa[..., None]
        frame = np.where(a > 1e-4, frame / np.maximum(a, 1e-4), 0.0).astype(np.float32)
        _ras.encode(out, frame, float(exposure), bool(hdr))
        return np.dstack([out * a, fa])
    _ras.encode(out, frame, float(exposure), bool(hdr))
    if return_alpha:
        return np.dstack([out, np.ones((h, w), np.float32)])
    return out


def _dir_to_screen(view, off, s):
    if view is None:
        return off * s
    if view.kind == "affine":
        return off @ view.M[:, :2].T * s
    if view.kind == "homog":
        c = np.array([W / 2, H / 2])
        p0 = view.screen_to_plane(c)
        J = np.stack([view.plane_to_screen(p0 + [1, 0]) - c, view.plane_to_screen(p0 + [0, 1]) - c], -1)
        return off @ J.T * s
    return off * s


def _bundle(n, poly, nv, cen, size, area, rgb, mat, jit, rnd, tilt, melts, var, gain=None, emit=None, alpha=None,
            present=None, jemit=None, inset=None, holes="bed", screen_gain=1.0):
    mat = np.asarray(mat, np.uint8)
    kind = np.zeros(n, np.uint8)
    scr = mat == SCREEN
    kind[scr] = _ras.K_SCREEN
    emit_screen = np.zeros((n, 3), np.float32)
    if scr.any():
        emit_screen[scr] = screen_gain * (0.85 + 0.3 * rnd[scr, None])
    d = dict(poly=poly, nv=nv, cen=cen, size=size, area=area, rgb=np.ascontiguousarray(rgb, np.float32),
             mat=mat.copy(), jit=jit, rnd=rnd, tilt=tilt, kind=kind, emit_screen=emit_screen, melts=float(melts or 0.0),
             var=float(var or 0.0),
             gain=np.ones(n, np.float32) if gain is None else np.asarray(gain, np.float32),
             emit=np.zeros((n, 3), np.float32) if emit is None else np.asarray(emit, np.float32),
             alpha=np.ones(n, np.float32) if alpha is None else np.asarray(alpha, np.float32),
             jemit=np.zeros((n, 3), np.float32) if jemit is None else np.asarray(jemit, np.float32),
             inset=np.zeros(n, np.float32) if inset is None else np.asarray(inset, np.float32))
    d["bare"] = None
    d["sock"] = None
    if present is not None:
        pv = np.asarray(present, np.float32)
        gone = pv < 0.5
        bare = pv < -0.5 if holes == "bed" else gone
        sock = gone & ~bare
        if sock.any():
            d["sock"] = sock
        if bare.any():
            d["bare"] = bare
    return d


_BED_LIN = srgb_to_lin(SC["bed"])
_BOLE_LIN = srgb_to_lin(np.array(col("#6E3A22"), np.float32))     # ochre-red bole painted under the gold


def _socket(d, idx):
    """turn tiles idx of a bundle into empty sockets: the setting bed with the tile's imprint (in place)."""
    for k in ("kind", "mat", "emit", "jemit"):
        d[k] = d[k].copy()
    d["kind"][idx] = _ras.K_IMPRINT
    d["mat"][idx] = BED
    d["emit"][idx] = 0


_GOLD_LIN_ = np.ascontiguousarray(_GOLD_LIN, np.float32)


def _alb(d, ii, mat=None):
    """linear albedo of the visible tiles ii (numba)."""
    out = np.empty((len(ii), 3), np.float32)
    m = d["mat"] if mat is None else mat
    idx = np.ascontiguousarray(ii, np.int64)
    if mat is not None:        # mat already restricted to ii
        full = np.zeros(len(d["mat"]), np.uint8)
        full[idx] = mat
        m = full
    _ras.albedo(d["rgb"], np.ascontiguousarray(m, np.uint8), np.ascontiguousarray(d["jit"], np.float32),
                np.ascontiguousarray(d["rnd"], np.float32), idx, d["melts"], d["var"], _GOLD_LIN_,
                _SILVER_LIN, _PEARL_LIN, _BACK_LIN, _BED_LIN, _GOLD_REF, _SILVER_REF, out)
    return out


def _select(d, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d["cen"]) else v)
            for k, v in d.items()}


def _pose_split(n, lift, rot, thresh_lift=0.25, thresh_rot=0.01):
    posed = np.zeros(n, bool)
    if lift is not None:
        posed |= np.abs(lift) > thresh_lift
    if rot is not None:
        posed |= (np.abs(rot - np.eye(3, dtype=np.float32)[None]).reshape(n, -1).max(1)) > thresh_rot
    return posed


# ============================================================ public rendering
def render_tiles(fc, tiles, view=None, surface=None, light=None, loose=False, bg=None, return_alpha=False, t=None,
                 mortar="mortar", ambient=None, exposure=1.0, sparkle=1.0, bevel=1.0, melts=1.0, var=0.06, env=1.0,
                 holes="bed", hdr=False, joint_emit=None, joint_img=None, shadows=True, s=None, size_px=None,
                 screen_gain=1.0, painted=0.5):
    """Render a Tiles bundle with the film's material look. See MANUAL."""
    view = _as_view(view)
    s = fc.s if s is None else s
    wh = size_px or (fc.w, fc.h)
    T = tiles
    n = T.n
    Ls, amb = _lights(light, view, ambient)
    d = _bundle(n, T.poly, T.nv, T.xy, T.size, _poly_area(T.poly, T.nv), T.rgb, T.mat, T.jit, T.rnd, T.tilt,
                melts, var, T.gain, T.emit, T.alpha, T.present, joint_emit, None, holes, screen_gain)
    tt = fc.T if t is None and hasattr(fc, "T") else (t or 0.0)
    bare, sock = d["bare"], d["sock"]
    d["bare"] = d["sock"] = None
    if loose:
        d["lift"] = T.z
        d["rot"] = T.rot
        d["stretch"] = T.stretch
        d["world"] = True
        if T.present is not None:
            d["alpha"] = d["alpha"] * (np.asarray(T.present) >= 0.5)
        return _draw(wh, s, view, surface, loose=d, lights=Ls, amb=amb, bg=bg, mortar=mortar, sparkle=sparkle,
                     bevel=bevel, env=env, exposure=exposure, hdr=hdr, return_alpha=return_alpha, t=tt,
                     joint_img=joint_img, shadows=shadows, painted=painted)
    posed = _pose_split(n, T.z, T.rot)
    lo = None
    sel = np.ones(n, bool) if bare is None else ~bare
    if posed.any():
        lo = _select(d, posed)
        lo["lift"] = T.z[posed] if T.z is not None else None
        lo["rot"] = T.rot[posed] if T.rot is not None else None
        lo["stretch"] = T.stretch[posed] if T.stretch is not None else None
        if holes != "bed":
            sel &= ~posed
    sk = posed | (sock if sock is not None else False)
    if bare is not None:
        sk &= ~bare
    if holes == "bed" and np.any(sk):
        _socket(d, np.nonzero(sk)[0])
    wall = d if sel.all() else _select(d, np.nonzero(sel)[0])
    return _draw(wh, s, view, surface, wall=wall, loose=lo, lights=Ls, amb=amb, bg=bg, mortar=mortar,
                 sparkle=sparkle, bevel=bevel, env=env, exposure=exposure, hdr=hdr, return_alpha=return_alpha, t=tt,
                 joint_img=joint_img, shadows=shadows, painted=painted)


def render(fc, cartoon, lay, gold=None, light=None, grout=1.3, mortar="mortar", var=0.06, sparkle=1.0,
           bevel=1.0, alpha=None, s=None, return_alpha=False, *, view=None, surface=None, rgb=None, mat=None,
           silver=None, pearl=None, stone=None, screen=None, present=None, gain=None, emit=None, offset=None,
           lift=None, rot=None, joint_emit=None, joint_img=None, bg=None, holes="bed", crect=None, melts=1.0,
           ambient=None, exposure=1.0, env=1.0, hdr=False, t=None, sample_mode="medoid", shadows=True,
           screen_gain=1.0, painted=0.5):
    """Set the cartoon in tesserae. See MANUAL. v0-compatible: render(fc, cartoon, lay, gold=, light=) with the
    cartoon covering lay.extent at pixel scale s (= fc.s) returns the (h, w, 3) image of that extent."""
    view = _as_view(view)
    n = len(lay)
    if rgb is None:
        if cartoon is None:
            raise ValueError("render: need a cartoon or rgb=(N,3)")
        rgb = sample(cartoon, lay, crect=crect, mode=sample_mode)
    rgb = np.ascontiguousarray(np.broadcast_to(np.asarray(rgb, np.float32), (n, 3)))
    m = lay.mat.copy() if mat is None else np.ascontiguousarray(np.broadcast_to(np.asarray(mat, np.uint8), (n,)))
    for code, mk in ((GOLD, gold), (SILVER, silver), (PEARL, pearl), (STONE, stone), (SCREEN, screen)):
        if mk is None:
            continue
        mk = np.asarray(mk)
        frac = mk.astype(np.float32) if mk.ndim == 1 and len(mk) == n else sample_mask(mk, lay, crect=crect)
        m[frac >= 0.5] = code
    if alpha is not None:
        a = np.asarray(alpha)
        ta = a.astype(np.float32) if a.ndim == 1 and len(a) == n else sample_mask(a, lay, crect=crect)
        keep = ta >= 0.5
    else:
        keep = None
    if s is None:
        s = fc.s
    if view is None and surface is None:
        wh = (int(round(lay.extent[0] * s)), int(round(lay.extent[1] * s)))
    else:
        wh = (fc.w, fc.h)
    Ls, amb = _lights(light, view, ambient)
    inset = np.full(n, (grout - 1.3) * 0.5 * s, np.float32) if grout != 1.3 else None
    em = None
    if emit is not None:
        e = np.asarray(emit, np.float32)
        if e.ndim >= 2 and e.shape[0] != n:     # an image in cartoon space
            if e.ndim == 2:
                em = sample_mask(e, lay, crect=crect)[:, None] * np.ones(3, np.float32)
            else:
                em = srgb_to_lin(sample(e, lay, crect=crect, mode="mean"))
        else:
            em = np.ascontiguousarray(np.broadcast_to(e if e.ndim == 2 else e[:, None], (n, 3)), np.float32)
    # per-tile poses: offsets move bedded tiles; lifted / rotated tiles become loose and leave their socket
    off = None if offset is None else np.ascontiguousarray(np.broadcast_to(np.asarray(offset, np.float32), (n, 2)))
    lf = None if lift is None else np.ascontiguousarray(np.broadcast_to(np.asarray(lift, np.float32), (n,)))
    rotm = _rotations(rot, n) if rot is not None else None
    posed = _pose_split(n, lf, rotm) if (lf is not None or rotm is not None) else np.zeros(n, bool)
    cen, poly = lay.xy, lay.poly
    if off is not None:
        ob = np.where(posed[:, None], 0.0, off).astype(np.float32)
        cen = cen + ob
        poly = poly + ob[:, None, :]
    d = _bundle(n, poly, lay.nv, cen, lay.size, lay.area, rgb, m, lay.jit, lay.rnd, lay.tilt, melts, var,
                None if gain is None else np.broadcast_to(np.asarray(gain, np.float32), (n,)),
                em, None, present, None if joint_emit is None else np.broadcast_to(
                    np.asarray(joint_emit, np.float32), (n, 3)), inset, holes, screen_gain)
    d["key"] = None
    d["und"] = _lay_undulation(lay)
    sel = np.ones(n, bool)
    if keep is not None:
        sel &= keep
    if d.get("bare") is not None:
        sel &= ~d["bare"]
    bare = d["bare"] if d["bare"] is not None else np.zeros(n, bool)
    sock = d["sock"]
    d["bare"] = d["sock"] = None
    lo = None
    pi = posed & (keep if keep is not None else True)
    if np.any(pi):
        idx = np.nonzero(pi)[0]
        lo = _select(d, idx)
        lo["cen"] = lay.xy[idx] + (off[idx] if off is not None else 0)
        lo["poly"] = lay.poly[idx] + (off[idx][:, None, :] if off is not None else 0)
        lo["lift"] = lf[idx] if lf is not None else None
        lo["rot"] = rotm[idx] if rotm is not None else None
        lo["stretch"] = None
        lo["key"] = None
        if holes != "bed":
            sel &= ~pi
    sk = (pi | (sock if sock is not None else False)) & ~bare
    if holes == "bed" and np.any(sk):
        _socket(d, np.nonzero(sk)[0])
    wall = d
    if not sel.all():
        wall = _select(d, np.nonzero(sel)[0])
        wall["key"] = None
    tt = (fc.T if hasattr(fc, "T") else 0.0) if t is None else t
    return _draw(wh, s, view, surface, wall=wall, loose=lo, lights=Ls, amb=amb, bg=bg, mortar=mortar,
                 sparkle=sparkle, bevel=bevel, env=env, exposure=exposure, hdr=hdr, return_alpha=return_alpha, t=tt,
                 joint_img=joint_img, shadows=shadows, painted=painted)


def render_layer(fc, cartoon_rgba, lay, M=None, **kw):
    """A moving figure/object carrying its own tiles: its LOCAL premultiplied rgba cartoon (covering lay.extent,
    any resolution) is set with a fixed local layout (tiles with alpha < 0.5 dropped) and placed on screen with the
    2x3 affine M (local design units -> screen design units), rasterised directly (crisp at any scale).
    Returns premultiplied rgba (fc.h, fc.w, 4)."""
    rgba = np.asarray(cartoon_rgba, np.float32)
    a = rgba[..., 3]
    rgb = rgba[..., :3] / np.maximum(a[..., None], 1e-4)
    if M is None:
        kw.setdefault("s", fc.s)
        out = render(fc, rgb, lay, alpha=a, return_alpha=True, **kw)
        if out.shape[:2] != (fc.h, fc.w):
            full = np.zeros((fc.h, fc.w, 4), np.float32)
            hh, ww = min(fc.h, out.shape[0]), min(fc.w, out.shape[1])
            full[:hh, :ww] = out[:hh, :ww]
            return full
        return out
    return render(fc, rgb, lay, alpha=a, return_alpha=True, view=View.affine(np.asarray(M, np.float64)), **kw)


def tiles(fc, cartoon, lay, gold=None, s=None):
    """per-tile colours for tile physics (falling, flying, re-setting): returns (xy, ang, size, rgb)."""
    return lay.xy.copy(), lay.ang.copy(), lay.size.copy(), sample(cartoon, lay)


def tile_colours(img, lab, n, alpha=None):
    """mean colour (and alpha) of an image under each tile of a label raster (v0 helper)."""
    L = lab.ravel()
    m = L >= 0
    L = L[m]
    cnt = np.bincount(L, minlength=n).astype(np.float32) + 1e-6
    out = np.empty((n, 3), np.float32)
    flat = img.reshape(-1, img.shape[-1])[m]
    for c in range(3):
        out[:, c] = np.bincount(L, flat[:, c], minlength=n) / cnt
    a = None
    if alpha is not None:
        a = np.bincount(L, alpha.ravel()[m], minlength=n) / cnt
    return out, a, cnt


_RCACHE = {}


def raster(lay, s, grout=1.3):
    """(label, edge_px, u, v) for a layout at pixel scale s (label -1 = mortar). Cached per layout+scale.
    v0 helper; v1 renders without it."""
    key = (lay.key, round(s, 4), grout)
    r = _RCACHE.get(key)
    if r is None:
        w, h = int(round(lay.extent[0] * s)), int(round(lay.extent[1] * s))
        lab = -np.ones((h, w), np.int32)
        edge = np.zeros((h, w), np.float32)
        uu = np.zeros((h, w), np.float32)
        vv = np.zeros((h, w), np.float32)
        inset = (grout - 1.3) * 0.5 * s
        _lab_pass(lab, edge, uu, vv, (lay.poly * s).astype(np.float32), lay.nv.astype(np.int32),
                  (lay.xy * s).astype(np.float32), lay.ang.astype(np.float32), (lay.size * s * 0.5).astype(np.float32),
                  float(inset))
        r = (lab, edge, uu, vv)
        if len(_RCACHE) > 8:
            _RCACHE.pop(next(iter(_RCACHE)))
        _RCACHE[key] = r
    return r


@_njit_early(cache=True)
def _lab_pass(lab, edge, uu, vv, V, nv, c, ang, hs, inset):
    h, w = lab.shape
    for m in range(V.shape[0]):
        n = nv[m]
        xmin = V[m, :n, 0].min()
        xmax = V[m, :n, 0].max()
        ymin = V[m, :n, 1].min()
        ymax = V[m, :n, 1].max()
        ca = math.cos(ang[m])
        sa = math.sin(ang[m])
        for py in range(max(int(ymin), 0), min(int(ymax) + 1, h - 1) + 1):
            for px in range(max(int(xmin), 0), min(int(xmax) + 1, w - 1) + 1):
                fx = px + 0.5
                fy = py + 0.5
                dmax = -1e30
                for e in range(n):
                    j = e + 1 if e + 1 < n else 0
                    ex = V[m, j, 0] - V[m, e, 0]
                    ey = V[m, j, 1] - V[m, e, 1]
                    L = math.sqrt(ex * ex + ey * ey)
                    if L < 1e-6:
                        continue
                    nx = ey / L
                    ny = -ex / L
                    d = nx * (fx - V[m, e, 0]) + ny * (fy - V[m, e, 1])
                    if d > dmax:
                        dmax = d
                din = -dmax - inset
                if din > 0:
                    lab[py, px] = m
                    edge[py, px] = din
                    dx = fx - c[m, 0]
                    dy = fy - c[m, 1]
                    uu[py, px] = (dx * ca + dy * sa) / hs[m]
                    vv[py, px] = (-dx * sa + dy * ca) / hs[m]


def draw_tiles(ctx, xy, ang, size, rgb, alpha=1.0, shade=None):
    """draw loose tesserae with cairo (design units): little bevelled squares. shade (N,) optional 0..1.5.
    (For thousands of tiles use render_tiles(..., loose=True) - numba, same look as the wall.)"""
    for i in range(len(xy)):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        hs = float(size[i]) * 0.46
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(float(ang[i]))
        r, g, b = (float(c) for c in rgb[i])
        k = 1.0 if shade is None else float(shade[i])
        ctx.rectangle(-hs, -hs, 2 * hs, 2 * hs)
        ctx.set_source_rgba(min(1, r * k), min(1, g * k), min(1, b * k), alpha)
        ctx.fill()
        # bevel: light upper-left rim, dark lower-right rim
        ctx.set_line_width(max(0.6, hs * 0.18))
        ctx.move_to(-hs, hs)
        ctx.line_to(-hs, -hs)
        ctx.line_to(hs, -hs)
        ctx.set_source_rgba(min(1, r * k * 1.25 + 0.06), min(1, g * k * 1.25 + 0.06), min(1, b * k * 1.25 + 0.06),
                            alpha * 0.8)
        ctx.stroke()
        ctx.move_to(hs, -hs)
        ctx.line_to(hs, hs)
        ctx.line_to(-hs, hs)
        ctx.set_source_rgba(r * k * 0.6, g * k * 0.6, b * k * 0.6, alpha * 0.8)
        ctx.stroke()
        ctx.restore()


from .mosaic_type import Titulus, titulus, latinize, labels_from, face as font_face  # noqa: E402
from . import mosaic_fx as fx  # noqa: E402
from .mosaic_sinopia import Sinopia, irradiance, remap_chart  # noqa: E402

MANUAL = __doc__

"""vx.icon - OPVS SCHISMATICVM (film 3): every character as a BYZANTINE MOSAIC FIGURE CARTOON. Owner: Iconographer.

The mosaicist's full-size design (the "cartoon") of every figure in the basilica - Ravenna / Hosios Loukas / Monreale /
Hagia Sophia: frontal, elongated figures, almond eyes under heavy lids, long straight noses, small mouths, flesh in flat
tone bands (olive proplasmos under warm sarkoma, pink cheek spots, white highlight strokes), heavy dark contours,
drapery as flat fields crossed by dark fold lines and light highlight lines, gold chrysography, jewels, nimbi.
vx.mosaic then sets the cartoon in tesserae. THE ONE RULE: icon.py never draws a Flag (no cloth, no pole); it only
returns where the pole goes (anchors['pole']) and draws the fists that grip it (part='front').

QUICK START
    from vx import icon, mosaic as mz
    lutie = icon.Figure("lutie")
    def draw(ctx, layer):                                  # draw the whole picture once per layer
        lutie.draw(ctx, 600, 1000, 820, pose="bless", t=fc.t, mouth=tl.mouth("LUTIE", fc.T), look=(0.4, 0),
                   layer=layer, tile=11)
    c = icon.cartoon(draw, s=fc.s)                         # rgba + rgb + fine/gold/silver/pearl/glow/edges masks
    lay = mz.Layout.flow(c["rgb"], tile=11, fine=c["fine"] > 0.5, fine_tile=5.5, gold=c["gold"], silver=c["silver"],
                         pearl=c["pearl"], ground=c["ground"])       # ONCE, in setup (cache it)
    img = mz.render(fc, c["rgb"], lay, gold=c["gold"], silver=c["silver"], pearl=c["pearl"])

COORDINATES
    (x, y) = the ground point between the feet (design units, any ctx transform is honoured); h = height from the
    soles to the crown of the HEAD (halo, hair tuft, helmet, crown, tiara, raised hands stick out above; use
    anchors['bbox'] or Figure.box(h)). Internally a figure is 100 units tall (u = h/100 design px per unit).
    Proportions ~7 heads (heads are a little large on purpose: faces must read in tiles).

Figure(kind, seed=0, tile=None, **style)
    kinds (KINDS):
      'lutie'         novice vexiherbologist: ochre habit, hood down on the shoulders, messy dark hair + cowlick,
                      holds a sprig of weld (Reseda luteola, the dyer's herb) in her left fist; gold nimbus
      'crocus'        Master Crocus as a Church Father: black inner robe, ochre phelonion with gold chrysography,
                      plain ochre omophorion with gold edges (no crosses), ochre veil-hood, long white beard,
                      jewelled codex; pearled nimbus
      'schismmancer'  warrior saint (St George / St Demetrius): lamellar klibanion (silver), porphyry chlamys + gold
                      fibula, pteruges, ochre tunic, leggings, knee pads, laced boots, helmet with QUAD-tube NVG on a
                      flip mount (goggles=0 up .. 1 down), sword at the hip; 'stand' = lance (pole) in the right fist
      'schism_actual' the commander: same + radio headset with boom mic; 'stand' = pole + orb in the left hand
      'jaguar'        President Jaguar as Justinian: jaguar head (rosettes), jewelled stemma + pendilia (pearl
                      strings), navy chlamys with gold tablion + jewelled fibula, white tunic with a red 'tie' stripe,
                      red shoes, spotted tail tip under the hem, edict in the left paw
      'courtier' / 'guard' / 'deacon'   San Vitale court: white chlamys + porphyry tablion / short tunic, torque,
                      spear (drawn, silver blade) + plain oval shield / white dalmatic + open codex
      'citizen'       ordinary people in cool desaturated tunics, crowned (nations of one: tiny gold crown)
      'heretic'       haloed saint of the congregation (m/f by seed); tiara=True -> papal tiara
      'pope'          a pope of one: triregnum tiara (no cross), cope with gold orphreys, plain pallium, tiny church
      'philosopher'   Pompeii Academy: tunic + himation over the left shoulder, beard, scroll, sandals
      'crackpot'      a philosopher with wild radiating hair (+ pose 'rant')
      'people'        generic crowd (m/f, veils, cloaks, skins, robes vary by seed)
      'hypermind'     the ophanim-polycandelon (see HYPERMIND); 'egregore' the faceless anti-Pantocrator (EGREGORE)
    seed: variety (skin, hair, beard, robe colours, sex, age) for the crowd kinds and warriors.
    tile: default tile size (design px) the figure will be tessellated with -> minimum line widths (a contour is
          one dark course: >= 1.15 tile; folds >= 0.9 tile; face features >= 1 fine tile = 0.5 tile). None = no clamp.
    style overrides: robe=, over=, sleeve=, cuff=, shoes= (IP colour keys: 'ochre','black','navy','white',
          'porphyry','lapis','grey_c','blue_c','teal_c','plum_c','rust_c','green_c','sand_c', ...), skin=
          ('light'|'warm'|'olive'|'brown'|'dark'), hair= ('lutie','short','cap','long','long_f','bun','veil','bald',
          'wild','none'), hair_color='#rrggbb', beard= ('short','pointed','full','long','patriarch','wild','stubble'|
          None), sex='m'|'f', halo=True|False|'gold'|'gold_pearl'|'square'|'square_gold' (square = pale-blue
          nimbus of the living), prop=, crown=, tiara=, shield=, shield_c=, tail=, headset=, build=0.8..1.2,
          head_scale=1.0 (the jaguar's head is already 1.25x), veil_c=, cloak= (warriors' chlamys family:
          'porphyry' default, 'tyrian', 'lapis', 'navy', ...), tablion= (courtier/emperor panel colour)

  .draw(ctx, x, y, h, pose='stand', t=0.0, mouth=0.0, look=(0, 0), expr='neutral', facing=0, layer='color',
        part='all', tile=None, blink=None, alpha=1.0, tint=None, silhouette=None, lod=None, **kw) -> anchors
      t       seconds (blinks, march cycle, 'rant', tail sway); frames are pure functions of t
      mouth   0..1 lip-sync openness: pass tl.mouth(SPEAKER, fc.T) (dark mouth, teeth; jaguar: fangs)
      look    (-1..1, -1..1) gaze in WORLD screen space (+x right, +y down): the irises slide in the almonds
      expr    'neutral','worried','serene','stern','joyful','shouting','awe','wry','sly','proud','sad','angry'
              or a blend {name: weight}
      facing  0 frontal (hieratic) .. +-1 three-quarter turned to screen right/left (features shift, figure mirrors
              for negative values: the rig's right hand is then on screen right)
      blink   None = automatic blinking from t, True/False, 0..1 lid closure, or (screen_left, screen_right) per eye;
              wink=0..1 closes the screen-right eye (wink_side='l' the other); eyes_closed=True
      layer   'color' (the cartoon) | mask layers 'fine','gold','silver','pearl','glow' (opaque WHITE where the
              material is, ERASING under everything else, so occlusion is exact - draw every figure on one canvas per
              layer) | 'edges' (thin white centre-lines of every contour/fold for Layout.flow(edges=...))
      part    'all' | 'back' (everything but the fists gripping a pole) | 'front' (only those fists). A held Flag:
              draw part='back' -> tessellate -> composite the vx.flag pole+cloth -> tessellate+composite part='front'.
      tile    tile size (design px) for line-width clamping and the level of detail (overrides Figure(tile=))
      alpha, tint=(colour, amount), silhouette=colour   (colour layer only; distance haze, shadows)
      lod     None auto (from head size in fine tiles, or device px when tile=None) | 0 full | 1 simple | 2 tiny
    kw (all optional):
      arms='pose'           take BOTH arms from another pose (e.g. pose='march', arms='hold_pole')
      arm_r=/arm_l='pose'   take one arm from another pose (e.g. pose='hold_pole1', arm_l='bless'); swap=True mirrors
      hand_r=/hand_l=(x, y) world targets for the wrists (elbow re-placed); ang_r/ang_l= finger angle (deg, 0 up,
                            + outward); shape_r/shape_l= hand shape (HANDS below)
      prop=                 'weld','book','codex_open','scroll','edict','church','orb', None/'none'
      pole_ang=rad          world angle of a held pole (0 = up, + leans right); pole_mode= override (see anchors)
      ground_y=             world y where a pole base rests / a planted pole lands (default: the figure's ground)
      progress=0..1         'plant' (impact at PLANT_IMPACT = 0.6), 'crown_self' (tiara reaches the head at 1)
      fend=0..1, shake=(dx, dy)   'fend': the right hand rises beside the head, palm out (refusing the sticky Flag)
      speed=, phase=        'march' gait (phase 0..1 overrides t)
      lean=, bow=, nod=, tilt=   whole-body lean (rad, + toward the facing direction), torso bow, head nod/tilt
      wind=0..2             cloaks fly ('charge' sets 1)
      goggles=0..1          NVG flip mount (warriors): 0 up on the helmet .. 1 down over the eyes
      halo=False            skip the nimbus (anchors['halo'] is still returned so a scene can paint its own)
      crown=, tiara=        per-draw headgear toggles for citizens / heretics
      pend=(a_left, a_right)  the jaguar's pendilia (pearl strings): world angles (rad), 0 = hanging straight down

  .anchors(x, y, h, pose, t, facing, **kw) -> the same dict, pure (no drawing)
  .box(h, pose=None) -> (x0, y0, x1, y1) offsets from the ground point; pose=None = conservative for every pose
  .bust(ctx, cx, cy, size, pose='stand', **draw kw) -> anchors: head + shoulders fitted in a roundel of diameter size

POSES (POSE_NAMES): stand, orans (raised hands), bless (Byzantine blessing at the chest), bless_high, speak (speech
    gesture), object (raised index: 'videtur quod non'), one_finger, declare (arm raised high, palm out: adlocutio),
    point, point_self, offer (both hands forward, cupped; with pole=True or a flag -> 'present'), present (both fists
    on a pole held forward; pole_ang=), let_go (hands open, the pole anchor stays at the right palm), explain (pole in
    the right fist, left hand swept out), look_up (donor: hands raised, head + eyes up), alarm (hands flung up to the
    head), recoil, clasp (praying hands), kneel (stance + praying hands), sit_throne (jewelled throne + footstool),
    sit (plain bench), march (procession gait; arms from arms= or the kind), hold_pole (two fists, pole on the ground),
    hold_pole1 (one fist), lance (warrior: pole + hand on the sword), command (pole + orb), rest_pole (hand high on a
    planted pole, other hand speaking), plant (raise overhead, drive down: progress), brace (weight forward, lance
    raised), charge (lunge, lance couched, cloak flying), salute / raise (pole raised high), sign (stylus on an edict),
    radio (hand to the headset), candle (a candle in the right fist; anchors['flame']), shout (fists up), rant
    (flailing, animated by t), crown_self (both hands lower a papal tiara onto the head: progress), fend (the right
    hand rises beside the head, palm out: 'no, thank you'; fend=0..1, shake=; anchors['cuff_r']).
    Turned figures (|facing| > 0.3) gesture with the arm on the side they face (point, bless, speak, object ...);
    swap=False keeps the canonical right-hand gesture.
    pose may also be a blend {name: weight} (arm targets interpolate; the heaviest pose gives stance and hand shapes).
HANDS: relax, back, palm, bless, speak, point, point_up, hold, fist, grip (a fist around a pole), cup/open (palm up),
    pray, stylus (anchors['stylus_tip']), pinch (anchors['pinch']), cup_ear, veil.

ANCHORS (world design coordinates; angles in radians)
    head, head_r, face, eyes [(x,y) left, right on screen], mouth, top (crown of hair/headgear), neck, chest, hip,
    halo (cx, cy, r), feet (the ground point), foot_r, foot_l, shoulder_r/l, elbow_r/l, wrist_r/l,
    hand_r / hand_l (palm or fist centre; _r = the rig's RIGHT hand = screen left when facing >= 0), hand_active
    (the hand doing the pose's gesture: hand_r, or hand_l when a turned figure swapped arms), hand_r_ang /
    hand_l_ang (finger direction, atan2 convention), cuff_r / cuff_l (x, y, forearm direction) on the front of the
    forearm near the cuff, pole (base_x, base_y, ang) ready for vx.flag Flag.simulate(pole_fn) (base of the pole,
    ang 0 = straight up, + clockwise), grip (upper fist on the pole), grips, flame (candle), candle, orb (x, y, r),
    lenses [(x, y, r) x4] (NVG tubes, for glow), prop (held attribute), stylus_tip, pinch, bbox (x0, y0, x1, y1),
    scale (= h/100), h, kind.

HELPERS
    cartoon(draw_fn, s=1.0, w=None, h=None, extent=None, bg=None, layers=...) -> dict(rgba (premultiplied), rgb,
        alpha, fine, gold, silver, pearl, glow, edges, ground (= 1 - alpha), s, extent): calls draw_fn(ctx, layer)
        once per layer on (extent x s) canvases. Build Layout.flow once (setup) from these; per frame only 'color'
        (and 'gold' if metal moves) is needed: cartoon(draw_fn, s, layers=('color', 'gold')).
    crowd(ctx, people, t=0.0, layer='color', tile=None, sort=True) -> [anchors]: many figures back-to-front;
        people = [dict(kind, x, y, h, seed, pose, facing, expr, mouth, look, t_off, alpha, tint, style={...}, ...)].
        ~0.6 ms per small figure; figures are cached per (kind, seed, style).
    people_variants(n, seed=0, kinds=('people',)) -> [(kind, seed)]
    draw_eye(ctx, x, y, w, open=1.0, look=(0, 0), rot=0.0, lid='gold'|'ink'|colour, layer='color', tile=None,
        iris=None, glow=False): one Byzantine eye, w = width in design px (gold lids go into the gold mask).
    draw_altar(ctx, x, y, h, layer='color', tile=None) -> dict(top, table): a tiny altar with a ciborium.
    Painter(ctx, layer, tile, ...): the layer-routing drawing helper (poly/line/ellipse/circle/dots/clip/push/pop)
        if you want your own props to obey the same mask conventions.
    IP: the smalti palette (hex); rgb(key) -> (r, g, b).

HYPERMIND  Figure('hypermind') or Hypermind(seed).draw(ctx, x, y, h, t=0.0, state='idle'|'training'|'compiling'|
    'done', wake=None, canons=None, spin=None, look=(0, 0), flag=0.0, layer='color', tile=None, part='all')
    -> anchors. See vx/icon_hyper.py.
EGREGORE   Figure('egregore') or Egregore(seed).draw(ctx, x, y, h, t=0.0, layer='color', tile=None, glitch=0.0)
    -> anchors. See vx/icon_hyper.py.

TILES + LOD. Design for the tile you will use: a 700 px figure at tile 11-12 (fine 5.5-6) keeps eyes, brows, lids,
    nostrils and lips each one course of fine tiles. Head length in fine tiles: < 10.5 -> 'tiny' face (dark almonds,
    brow-nose line, mouth dot), 10.5-28 -> 'simple' (the classic mosaic eye: white | dark iris | white, brows lifted
    one course above the lids, no highlight strokes), >= 28 -> 'full' (pupils, catch-lights, lid creases, eye sockets,
    white highlight strokes). Without tile= the LOD follows the head size in device pixels (<30 / 30-90 / >=90).
PERFORMANCE (1 core, 1080p, h=900, measured): colour pass 6-16 ms per hero figure (people 6, lutie 7, crocus 9,
    warriors 11-13, jaguar 14-16), each mask pass 7-18 ms (interior strokes are skipped in masks); the first call of a
    process adds ~20 ms (shapely hand library). crowd(): 200 figures at 90-130 px in ~125 ms (0.6 ms each).
    Hypermind ~20 ms, Egregore ~10 ms. cartoon() of 5 hero figures at s=1: all 7 layers ~0.7 s (setup only),
    ('color', 'gold') ~0.12 s (per frame). Build Layout.flow in setup; per frame re-draw only what moves.
"""
import math

import cairo
import numpy as np

from .canvas import Canvas, col
from .ease import hash01, clamp, smoothstep

TAU = math.tau

# ================================================================ palette (smalti; hex)
# Flags own the pure yellows. Robes are ochre (duller), gold is brown-metallic, everything else is non-yellow.
IP = {
    # lines
    "ink": "#17121A", "contour": "#4E2419", "brown_line": "#3A2416",
    # flesh (sarkoma), proplasmos (olive-green shadow), white highlights, pink cheeks, lips
    "flesh": "#DFAA8B", "flesh_l": "#EFCBAF", "flesh_hl": "#FBF1E3", "flesh_m": "#C28A6C",
    "olive": "#8E8564", "olive_d": "#5E5A42", "cheek": "#D9766B", "lip_u": "#8E2C27", "lip_l": "#C4544A",
    "mouth_in": "#35100E", "tooth": "#EFE7D6", "eye_w": "#F1EBDF", "iris": "#3A2617", "pupil": "#0F0908",
    # hair
    "hair_k": "#1C1411", "hair": "#3B271C", "hair_l": "#6E4E37", "grey": "#8E8E8A", "white_h": "#EEEBE4",
    "white_s": "#AAB2BC", "beard_d": "#6C7682",
    # cloth families: base / dark (shadow field) / light (highlight) / line (fold lines)
    "ochre": "#B77916", "ochre_d": "#7A4C0A", "ochre_l": "#D9A140", "ochre_k": "#3A2404",
    "black": "#24232C", "black_d": "#111117", "black_l": "#4E5467", "black_k": "#060609",
    "navy": "#1C2A5F", "navy_d": "#0F1638", "navy_l": "#4660A8", "navy_k": "#070A1C",
    "white": "#E9E3D5", "white_d": "#A3ABB3", "white_l": "#FCFAF3", "white_k": "#5E6064",
    "porphyry": "#76222F", "porphyry_d": "#4A111B", "porphyry_l": "#A8434F", "porphyry_k": "#26070C",
    "lapis": "#27398A", "lapis_d": "#16215A", "lapis_l": "#5670C0", "lapis_k": "#0A0F2E",
    "tyrian": "#5C1F52", "tyrian_d": "#3A1034", "tyrian_l": "#8A4580", "tyrian_k": "#1C0619",
    "grey_c": "#7D8494", "grey_c_d": "#555B69", "grey_c_l": "#A6ACB9", "grey_c_k": "#2B2F38",
    "blue_c": "#46597A", "blue_c_d": "#2E3C55", "blue_c_l": "#7187AA", "blue_c_k": "#161E2C",
    "teal_c": "#3F6E6E", "teal_c_d": "#2A4B4C", "teal_c_l": "#6F9C99", "teal_c_k": "#132424",
    "plum_c": "#6B4A6B", "plum_c_d": "#48304A", "plum_c_l": "#957298", "plum_c_k": "#241526",
    "rust_c": "#8A4B3A", "rust_c_d": "#5E3024", "rust_c_l": "#B37462", "rust_c_k": "#2C130D",
    "green_c": "#4C6A4A", "green_c_d": "#324832", "green_c_l": "#7A9876", "green_c_k": "#172417",
    "sand_c": "#A99A82", "sand_c_d": "#7D705D", "sand_c_l": "#CFC3AE", "sand_c_k": "#3B3328",
    "leather": "#6A3A22", "leather_d": "#40220F", "leather_l": "#955C3A", "leather_k": "#1E0E05",
    "steel": "#8E96A4", "steel_d": "#59606E", "steel_l": "#CBD1DA", "steel_k": "#22262E",
    "helmet": "#737A66", "helmet_d": "#4B503F", "helmet_l": "#9AA288", "helmet_k": "#1B1E16",
    "drab": "#5C6250", "drab_d": "#3B3F33", "drab_l": "#838B71", "drab_k": "#1A1C16",
    "red": "#A0262E", "red_d": "#621319", "red_l": "#CC4A50", "red_k": "#2E070A",
    # metals + jewels (gold mask decides the leaf; the cartoon value modulates it)
    "gold": "#B8862B", "gold_d": "#6E4A16", "gold_l": "#F3D27A", "silver": "#C9CED6", "silver_d": "#80868F",
    "pearl": "#F3EFE4", "ruby": "#B01E2E", "emerald": "#1E8A5A", "sapphire": "#2B55B5", "amethyst": "#6B3A8A",
    # fur (jaguar), ground, wood, wax, glass
    "fur": "#C27A36", "fur_d": "#8A4A1C", "fur_l": "#E3B98A", "fur_cream": "#EFE0C4", "rosette": "#1E1410",
    "nose_pk": "#9A5A55", "ground": "#3F6B45", "ground_d": "#284830", "wood": "#7A5230", "wood_d": "#4A3018",
    "wax": "#EDE4CF", "wax_d": "#BFB294", "glass_g": "#5DE08A", "glass_d": "#0E2A1C", "screen": "#9FD8FF",
    "weld": "#7C8A3A", "weld_d": "#4F5A22", "weld_l": "#A9B45A", "leaf": "#4E6B34", "leaf_d": "#2F4520",
    "plaster": "#D8CDB8", "mortar": "#4A4238", "slate": "#4B5160",
}
_RGB = {}


def rgb(c):
    """palette key (IP, then vx.config PAL) / '#hex' / tuple -> (r, g, b) floats"""
    if isinstance(c, str):
        v = _RGB.get(c)
        if v is None:
            v = col(IP[c]) if c in IP else col(c)
            _RGB[c] = v
        return v
    return tuple(c[:3])


def mixc(a, b, t):
    a, b = rgb(a), rgb(b)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


# ================================================================ painter (one drawing, many layers)
LAYERS = ("color", "fine", "gold", "silver", "pearl", "glow", "edges")
MASK_LAYERS = ("fine", "gold", "silver", "pearl", "glow")
# minimum line widths as a fraction of the tile size (agreed with the Tessellator)
LINE_MIN = {"contour": 1.15, "fold": 0.9, "light": 0.8, "gold": 0.85, "fine": 1.0, "fine2": 0.7, "thin": 0.0}
_OVER = cairo.OPERATOR_OVER
_ERASE = cairo.OPERATOR_DEST_OUT
# interior detail strokes (folds, highlights, face features) lie inside a fill that already erased the masks under it:
# mask layers skip them unless they carry the layer's material (halves the cost of a mask pass)
_INTERIOR = frozenset(("fold", "light", "thin", "fine2"))


class Painter:
    """Routes every shape of a drawing to the active layer.

    layer 'color' paints the cartoon.  Mask layers ('fine', 'gold', 'silver', 'pearl', 'glow') paint opaque white where
    a shape's material `mat` contains that word and ERASE (dest-out) under every other shape, so occlusion is exact
    (a hand in front of a gold halo removes the gold there).  'edges' paints thin white centre-lines for every stroke
    and erases under fills.  Line widths are clamped from below by the tile size (LINE_MIN x tile, design px)."""

    def __init__(self, ctx, layer="color", tile=None, alpha=1.0, tint=None, silhouette=None, fine_tile=None):
        if layer not in LAYERS:
            raise ValueError(f"icon: unknown layer {layer!r}; use one of {LAYERS}")
        self.ctx = ctx
        self.layer = layer
        self.tile = float(tile) if tile else 0.0
        self.fine_tile = float(fine_tile) if fine_tile else 0.5 * self.tile
        self.alpha = float(alpha)
        self.tint = (rgb(tint[0]), float(tint[1])) if tint else None
        self.sil = rgb(silhouette) if silhouette else None
        self.k = 1.0          # user units of the caller per current drawing unit
        self._ks = []
        self.lod = 0
        m = ctx.get_matrix()
        self.ds0 = math.sqrt(abs(m.xx * m.yy - m.xy * m.yx)) or 1.0    # device px per caller unit

    # ------------------------------------------------------------ transforms
    def push(self, x=0.0, y=0.0, rot=0.0, sx=1.0, sy=None):
        c = self.ctx
        c.save()
        c.translate(x, y)
        if rot:
            c.rotate(rot)
        sy = sx if sy is None else sy
        c.scale(sx, sy)
        self._ks.append(self.k)
        self.k *= math.sqrt(abs(sx * sy))

    def pop(self):
        self.ctx.restore()
        self.k = self._ks.pop()

    # ------------------------------------------------------------ sources
    def col(self, c):
        if self.sil is not None:
            return self.sil
        r, g, b = rgb(c)
        if self.tint is not None:
            (tr, tg, tb), a = self.tint
            r, g, b = r + (tr - r) * a, g + (tg - g) * a, b + (tb - b) * a
        return r, g, b

    def _src(self, c, mat, line):
        ctx, L = self.ctx, self.layer
        if L == "color":
            ctx.set_operator(_OVER)
            r, g, b = self.col(c)
            ctx.set_source_rgba(r, g, b, self.alpha)
        elif L == "edges":
            if line:
                ctx.set_operator(_OVER)
                ctx.set_source_rgba(1, 1, 1, 1)
            else:
                ctx.set_operator(_ERASE)
                ctx.set_source_rgba(0, 0, 0, 1)
        elif mat and L in mat:
            ctx.set_operator(_OVER)
            ctx.set_source_rgba(1, 1, 1, 1)
        else:
            ctx.set_operator(_ERASE)
            ctx.set_source_rgba(0, 0, 0, 1)

    def width(self, w, cls="contour"):
        """line width in current units, clamped by the tile size (design px)"""
        if self.layer == "edges":
            return 1.3 / (self.ds0 * self.k)          # a hairline in DEVICE pixels, whatever the ctx transform
        m = LINE_MIN.get(cls, 0.0)
        if m <= 0.0 or self.tile <= 0.0:
            return w
        base = self.fine_tile if cls.startswith("fine") else self.tile
        return max(w, m * base / self.k)

    # ------------------------------------------------------------ primitives
    def _finish(self, fill, line, lw, cls, mat, lmat):
        ctx = self.ctx
        if fill is not None:
            self._src(fill, mat, False)
            if line is not None:
                ctx.fill_preserve()
            else:
                ctx.fill()
        if line is not None:
            lm = lmat if lmat is not None else _line_mat(mat)
            if self.layer in MASK_LAYERS and cls in _INTERIOR and not (lm and self.layer in lm):
                ctx.new_path()
                return
            ctx.set_line_width(self.width(lw, cls))
            self._src(line, lm, True)
            ctx.stroke()
        if fill is None and line is None:
            ctx.new_path()

    def poly(self, pts, fill=None, line=None, lw=0.0, cls="contour", mat=None, lmat=None, close=True, smooth=False,
             tension=0.5):
        ctx = self.ctx
        if smooth:
            _smooth(ctx, pts, close, tension)
        else:
            ctx.move_to(*pts[0])
            for p in pts[1:]:
                ctx.line_to(*p)
            if close:
                ctx.close_path()
        self._finish(fill, line, lw, cls, mat, lmat)

    def line(self, pts, c, lw, cls="fold", mat=None, smooth=True, tension=0.5):
        """open stroke through pts"""
        if len(pts) < 2:
            return
        if self.layer in MASK_LAYERS and cls in _INTERIOR and not (mat and self.layer in mat):
            return
        ctx = self.ctx
        if smooth and len(pts) > 2:
            _smooth(ctx, pts, False, tension)
        else:
            ctx.move_to(*pts[0])
            for p in pts[1:]:
                ctx.line_to(*p)
        ctx.set_line_width(self.width(lw, cls))
        self._src(c, mat, True)
        ctx.stroke()

    def ellipse(self, cx, cy, rx, ry, rot=0.0, fill=None, line=None, lw=0.0, cls="contour", mat=None, lmat=None):
        ctx = self.ctx
        rx, ry = max(rx, 1e-4), max(ry, 1e-4)
        ctx.save()
        ctx.translate(cx, cy)
        if rot:
            ctx.rotate(rot)
        ctx.scale(rx, ry)
        ctx.new_sub_path()
        ctx.arc(0, 0, 1, 0, TAU)
        ctx.close_path()
        ctx.restore()
        self._finish(fill, line, lw, cls, mat, lmat)

    def circle(self, cx, cy, r, fill=None, line=None, lw=0.0, cls="contour", mat=None, lmat=None):
        ctx = self.ctx
        ctx.new_sub_path()
        ctx.arc(cx, cy, max(r, 1e-4), 0, TAU)
        ctx.close_path()
        self._finish(fill, line, lw, cls, mat, lmat)

    def ring(self, cx, cy, r0, r1, fill, mat=None):
        """annulus r0 < r < r1 (even-odd)"""
        ctx = self.ctx
        ctx.new_sub_path()
        ctx.arc(cx, cy, r1, 0, TAU)
        ctx.close_path()
        ctx.new_sub_path()
        ctx.arc_negative(cx, cy, r0, TAU, 0)
        ctx.close_path()
        self._src(fill, mat, False)
        ctx.fill()

    def dots(self, pts, r, fill, mat=None):
        """many small discs in one fill (pearls, rosettes, tile-sized dots)"""
        if not len(pts):
            return
        ctx = self.ctx
        for x, y in pts:
            ctx.new_sub_path()
            ctx.arc(x, y, r, 0, TAU)
            ctx.close_path()
        self._src(fill, mat, False)
        ctx.fill()

    def clip(self, pts, smooth=True):
        """clip following drawing to a polygon until unclip()"""
        ctx = self.ctx
        ctx.save()
        if smooth:
            _smooth(ctx, pts, True)
        else:
            ctx.move_to(*pts[0])
            for p in pts[1:]:
                ctx.line_to(*p)
            ctx.close_path()
        ctx.clip()
        self._ks.append(self.k)

    def unclip(self):
        self.ctx.restore()
        self.k = self._ks.pop()

    def paint_image(self, img, x, y, w, h, mat=None, nearest=True):
        """paint a float rgb (n,m,3) image stretched over the rect (static, screens); masks get a solid rect"""
        ctx = self.ctx
        if self.layer == "color":
            from .canvas import surface_from_array
            a = np.asarray(img, np.float32)
            if self.tint is not None:
                (tr, tg, tb), k = self.tint
                a = a + (np.array([tr, tg, tb], np.float32) - a) * k
            surf = surface_from_array(np.clip(a, 0, 1))
            ctx.save()
            ctx.rectangle(x, y, w, h)
            ctx.clip()
            ctx.translate(x, y)
            ctx.scale(w / a.shape[1], h / a.shape[0])
            ctx.set_source_surface(surf, 0, 0)
            ctx.get_source().set_filter(cairo.FILTER_NEAREST if nearest else cairo.FILTER_GOOD)
            ctx.set_operator(_OVER)
            ctx.paint_with_alpha(self.alpha)
            ctx.restore()
        else:
            ctx.rectangle(x, y, w, h)
            self._src("ink", mat, False)
            ctx.fill()


def _line_mat(mat):
    """outline strokes keep 'fine' but are never metal/glow (a halo's dark ring is not gold)"""
    if not mat:
        return None
    return "fine" if "fine" in mat else None


def _smooth(ctx, pts, close=False, tension=0.5):
    n = len(pts)
    if n < 3:
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        if close:
            ctx.close_path()
        return
    P = pts
    k = tension / 3.0
    ctx.move_to(*P[0])
    rng = range(n) if close else range(n - 1)
    for i in rng:
        if close:
            p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        else:
            p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, n - 1)]
        ctx.curve_to(p1[0] + (p2[0] - p0[0]) * k, p1[1] + (p2[1] - p0[1]) * k,
                     p2[0] - (p3[0] - p1[0]) * k, p2[1] - (p3[1] - p1[1]) * k, p2[0], p2[1])
    if close:
        ctx.close_path()


# ================================================================ small geometry
def lerp2(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def rot2(p, a, o=(0.0, 0.0)):
    c, s = math.cos(a), math.sin(a)
    x, y = p[0] - o[0], p[1] - o[1]
    return (o[0] + c * x - s * y, o[1] + s * x + c * y)


def auto_blink(t, seed):
    """deterministic blinking: 0 open .. 1 closed (a blink every ~3-4 s, sometimes double)"""
    per = 3.6
    k = math.floor((t + seed * 0.731) / per)
    b = 0.0
    for kk in (k - 1, k, k + 1):
        t0 = kk * per - seed * 0.731 + 0.4 + hash01(kk, 900 + seed) * 2.4
        x = (t - t0) / 0.07
        b = max(b, math.exp(-x * x))
        if hash01(kk, 901 + seed) < 0.18:
            x = (t - t0 - 0.23) / 0.07
            b = max(b, math.exp(-x * x))
    return b


# ================================================================ expressions
# lid: upper-lid closure 0..1, sq: lower lid up (smile squint), bin/bout: inner/outer brow raise (+ up, head units),
# frown: inner brow ends down/in, smile: mouth corners (+ up), open: extra mouth opening, wide: eye whites around iris,
# blush: cheek spot strength, brow1: extra raise of the figure's LEFT brow only (wry/sly), smirk: one-sided smile
EXPR = {
    "neutral": dict(lid=0.16, sq=0.0, bin=0.0, bout=0.0, frown=0.0, smile=0.0, open=0.0, wide=0.0, blush=1.0),
    "worried": dict(lid=0.0, sq=0.0, bin=0.06, bout=-0.02, frown=-1.5, smile=-0.9, open=0.08, wide=0.55, blush=0.7),
    "serene": dict(lid=0.52, sq=0.15, bin=0.01, bout=0.004, frown=-0.1, smile=0.45, open=0.0, wide=0.0, blush=1.0),
    "stern": dict(lid=0.25, sq=0.1, bin=-0.04, bout=0.02, frown=1.4, smile=-0.5, open=0.0, wide=0.0, blush=0.8),
    "joyful": dict(lid=0.3, sq=0.8, bin=0.03, bout=0.015, frown=-0.3, smile=1.6, open=0.25, wide=0.0, blush=1.6),
    "shouting": dict(lid=0.0, sq=0.0, bin=-0.02, bout=0.02, frown=0.9, smile=-0.2, open=0.85, wide=0.5, blush=1.3),
    "awe": dict(lid=0.0, sq=0.0, bin=0.05, bout=0.03, frown=-0.6, smile=0.1, open=0.45, wide=0.7, blush=1.3),
    "wry": dict(lid=0.36, sq=0.12, bin=-0.01, bout=0.0, frown=0.35, smile=0.2, open=0.0, wide=0.0, blush=1.0, brow1=0.06,
                smirk=0.9),
    "sly": dict(lid=0.55, sq=0.2, bin=-0.01, bout=0.0, frown=0.4, smile=0.45, open=0.0, wide=0.0, blush=1.0, brow1=0.07,
                smirk=1.0),
    "proud": dict(lid=0.36, sq=0.0, bin=0.01, bout=0.015, frown=0.15, smile=0.35, open=0.0, wide=0.0, blush=1.0),
    "sad": dict(lid=0.3, sq=0.0, bin=0.03, bout=-0.015, frown=-0.8, smile=-0.7, open=0.0, wide=0.0, blush=0.7),
    "angry": dict(lid=0.1, sq=0.1, bin=-0.035, bout=0.02, frown=1.3, smile=-0.6, open=0.25, wide=0.3, blush=1.5),
}
EXPRS = tuple(EXPR)
_EXPR_KEYS = ("lid", "sq", "bin", "bout", "frown", "smile", "open", "wide", "blush", "brow1", "smirk")


def blend_expr(expr):
    """'name' or {name: weight} -> parameter dict"""
    if isinstance(expr, str):
        e = EXPR.get(expr, EXPR["neutral"])
        return {k: e.get(k, 0.0) for k in _EXPR_KEYS}
    tot = sum(expr.values()) or 1.0
    out = {k: 0.0 for k in _EXPR_KEYS}
    for name, w in expr.items():
        e = EXPR.get(name, EXPR["neutral"])
        for k in _EXPR_KEYS:
            out[k] += e.get(k, 0.0) * w / tot
    return out


# ================================================================ poses
# Arm entries: (elbow, wrist, hand shape, hand angle deg). Coordinates (dx_out, dy): dx_out = distance from the body
# midline, POSITIVE = outward on that arm's own side (negative crosses the body), dy = below the shoulder line; units
# are figure-height/100 for a reference figure with shoulder half-width 9.5. Hand angle: direction of the fingers,
# 0 = up, +90 = outward, 180 = down, -90 = inward. Grip hands ('grip') orient themselves along the pole.
PLANT_IMPACT = 0.6


def _A(el, wr, shape, ang=0.0):
    return (el, wr, shape, ang)


_STAND_R = _A((13.0, 17.5), (6.6, 11.0), "speak", -26)
_STAND_L = _A((10.8, 15.0), (3.0, 19.5), "hold", -92)
_POLE_R = _A((14.2, 13.0), (11.6, 6.5), "grip")
POSES = {
    "stand": dict(R=_STAND_R, L=_STAND_L),
    "orans": dict(R=_A((20.0, 8.5), (18.6, -8.5), "palm", -6), L=_A((20.0, 8.5), (18.6, -8.5), "palm", -6), nod=0.02),
    "bless": dict(R=_A((13.2, 17.5), (5.4, 9.5), "bless", -14), L=_STAND_L),
    "bless_high": dict(R=_A((17.0, 5.0), (13.2, -12.0), "bless", -8), L=_STAND_L, nod=0.03),
    "speak": dict(R=_A((13.0, 17.5), (6.8, 11.5), "speak", -24), L=_STAND_L),
    "object": dict(R=_A((15.6, 9.5), (13.6, -4.5), "point_up", -4), L=_STAND_L, tilt=0.03),
    "point": dict(R=_A((19.5, 4.5), (28.0, -0.5), "point", 80), L=_STAND_L),
    "offer": dict(R=_A((12.8, 15.5), (4.8, 13.0), "cup", -84), L=_A((12.8, 15.5), (4.8, 13.0), "cup", -84)),
    "look_up": dict(R=_A((16.2, 12.0), (14.0, 0.5), "palm", 8), L=_A((16.2, 12.0), (14.0, 0.5), "palm", 8), nod=0.26,
                    gaze=(0.0, -1.0)),
    "hold_pole": dict(R=_POLE_R, L=_A((5.5, 23.5), (-11.6, 21.0), "grip"), pole="ground"),
    "hold_pole1": dict(R=_POLE_R, L=_STAND_L, pole="ground"),
    "rest_pole": dict(R=_A((15.5, 7.5), (13.4, -2.5), "grip"), L=_A((13.0, 17.5), (6.8, 11.5), "speak", -24),
                      pole="ground"),
    "plant": dict(R=_POLE_R, L=_A((9.0, 6.0), (-3.5, -2.0), "grip"), pole="plant"),
    "brace": dict(R=_A((15.5, 7.0), (12.2, -3.5), "grip"), L=_A((9.5, 17.5), (-10.0, 13.5), "grip"), pole="raised",
                  lean=0.06, tilt=-0.03),
    "charge": dict(R=_A((15.0, 10.0), (9.0, 3.0), "grip"), L=_A((7.5, 18.5), (-4.0, 15.5), "grip"), pole="couched",
                   lean=0.13, wind=1.0),
    "recoil": dict(R=_A((16.0, 10.0), (14.6, -3.0), "palm", 20), L=_A((16.0, 10.0), (14.6, -3.0), "palm", 20),
                   lean=-0.075, nod=0.08),
    "kneel": dict(R=_A((8.4, 16.5), (1.3, 10.5), "pray", 2), L=_A((8.4, 16.5), (1.3, 10.5), "pray", 2),
                  stance="kneel"),
    "march": dict(stance="march"),
    "sit_throne": dict(R=_A((13.2, 17.5), (5.6, 9.5), "bless", -14), L=_A((12.6, 16.5), (6.4, 23.5), "hold", -84),
                       stance="sit"),
    "sign": dict(R=_A((13.0, 18.5), (3.8, 16.5), "stylus", -125), L=_A((12.5, 17.0), (-1.2, 21.5), "hold", -95),
                 nod=-0.1, gaze=(0.0, 0.8)),
    "radio": dict(R=_POLE_R, L=_A((16.5, 4.5), (10.4, -8.0), "cup_ear", -170), pole="ground", tilt=0.05),
    "salute": dict(R=_A((13.2, -11.0), (11.2, -25.5), "grip"), L=_STAND_L, pole="raised_high", nod=0.1),
    "clasp": dict(R=_A((8.4, 16.5), (1.3, 10.5), "pray", 2), L=_A((8.4, 16.5), (1.3, 10.5), "pray", 2)),
    "shout": dict(R=_A((18.0, 3.5), (15.2, -12.5), "fist"), L=_A((18.0, 3.5), (15.2, -12.5), "fist"), nod=0.1),
    "candle": dict(R=_A((12.0, 16.5), (3.4, 12.0), "grip"), L=_A((12.0, 16.5), (3.6, 15.5), "cup", -84), pole="candle"),
    "declare": dict(R=_A((15.5, -2.5), (15.0, -18.5), "palm", -6), L=_STAND_L, nod=0.06),
    "one_finger": dict(R=_A((15.0, 9.5), (12.6, -6.5), "point_up", -4), L=_STAND_L, tilt=0.03),
    "alarm": dict(R=_A((17.0, 6.0), (10.8, -9.5), "palm", -18), L=_A((17.0, 6.0), (10.8, -9.5), "palm", -18),
                  nod=0.05),
    "present": dict(R=_A((12.0, 15.0), (-3.0, 9.0), "grip"), L=_A((11.5, 18.0), (3.0, 17.5), "grip"), pole="free",
                    lean=0.03),
    "let_go": dict(R=_A((12.0, 15.5), (-2.5, 9.5), "palm", -70), L=_A((11.5, 18.0), (3.5, 17.0), "palm", -80),
                   pole="palm", lean=0.01),
    "explain": dict(R=_POLE_R, L=_A((17.5, 10.5), (24.5, 5.5), "speak", 62), pole="ground"),
    "fend": dict(R=_STAND_R, L=_STAND_L),          # animated: see _pose_fend (kw fend=0..1, shake=(dx,dy))
    "rant": dict(R=_STAND_R, L=_STAND_L, nod=0.1),  # animated: see _pose_rant
    "point_self": dict(R=_A((13.0, 17.0), (4.2, 12.5), "point", -62), L=_STAND_L, tilt=-0.03),
    "crown_self": dict(R=_A((15.5, -3.0), (7.2, -17.0), "hold", -40), L=_A((15.5, -3.0), (7.2, -17.0), "hold", -40),
                       nod=0.12),
    "sit": dict(R=_A((13.0, 17.5), (6.8, 11.5), "speak", -24), L=_A((12.6, 16.5), (6.4, 23.5), "hold", -84),
                stance="sit"),
    "command": dict(R=_POLE_R, L=_A((12.5, 16.5), (6.8, 13.5), "cup", -80), pole="ground"),
    "lance": dict(R=_POLE_R, L=_A((14.0, 13.5), (10.4, 22.5), "fist"), pole="ground"),
}
POSES["raise"] = POSES["salute"]
POSE_NAMES = tuple(POSES)
_PALM_VIEW = {"palm", "bless", "speak", "point_up", "cup", "open", "pray", "pinch_up"}


_GESTURE_SWAP = {"point", "object", "one_finger", "declare", "bless_high", "speak", "bless", "rest_pole", "explain",
                 "point_self", "sign"}
# hand shapes that keep a canonical orientation when the wrist is placed by a world target (hand_r=/hand_l=)
_TARGET_ANG = {"open": -82.0, "cup": -82.0, "pinch": -12.0, "palm": -6.0, "bless": -12.0, "point_up": -4.0,
               "speak": -24.0, "pray": 2.0, "cup_ear": -170.0}


def _pose_fend(d, t, kw):
    """Crocus pushes the Flag away: forearm raised across the body (fend 0 rest .. 1 shoulder height)"""
    k = clamp(float(kw.get("fend", 1.0)), 0.0, 1.0)
    e0, w0 = (13.0, 17.5), (6.6, 11.0)
    e1, w1 = (6.0, 17.5), (14.0, 0.0)
    el = lerp2(e0, e1, k)
    wr = lerp2(w0, w1, k)
    d = dict(d)
    d["R"] = (el, wr, "palm" if k > 0.4 else "speak", -26 + 44 * k)
    return d


def _pose_rant(d, t, kw):
    """arms flailing (crackpots screaming into the void)"""
    d = dict(d)
    a1, a2 = math.sin(t * 5.3), math.sin(t * 4.1 + 1.3)
    b1, b2 = math.sin(t * 3.7 + 0.7), math.sin(t * 6.1 + 2.1)
    d["R"] = ((18.5 + 2.0 * a1, 3.0 + 3.0 * b1), (17.0 + 5.0 * a1, -12.0 + 7.0 * a2), "fist" if a2 > 0 else "palm",
              10 * a1)
    d["L"] = ((18.5 + 2.0 * b2, 4.0 + 3.0 * a2), (16.0 + 5.0 * b2, -10.0 + 7.0 * b1), "palm" if b1 > 0 else "fist",
              10 * b2)
    d["tilt"] = 0.06 * math.sin(t * 2.3)
    return d


_POSE_FN = {"fend": _pose_fend, "rant": _pose_rant}


def _pose_dict(name, t, kw):
    d = POSES[name]
    fn = _POSE_FN.get(name)
    return fn(d, t, kw) if fn else d


def _arm_of(pose, side, t=0.0, kw=None):
    p = _pose_dict(pose, t, kw or {}) if pose in POSES else POSES["stand"]
    return p.get(side) or POSES["stand"][side]


# ================================================================ the rig
class Rig:
    """a solved figure, all points in LOCAL units (figure height 100, origin at the ground point, y down, x toward the
    viewer's right in the un-mirrored frame). world(p) -> design coordinates."""
    __slots__ = ("x", "y", "u", "m", "lean", "c", "s", "HL", "SW", "head", "head_rot", "turn", "nod", "tilt",
                 "neck", "sh", "el", "wr", "shape", "hang", "hand_ang", "hip_y", "hem_y", "stance", "feet", "knees",
                 "drop", "bow", "pole", "grip", "gaze", "phase", "wind", "prop", "progress", "t", "sit", "kneel",
                 "shoulder_y", "chest", "waist_y", "extra")

    def world(self, p):
        px, py = p
        return (self.x + self.u * self.m * (self.c * px - self.s * py), self.y + self.u * (self.s * px + self.c * py))

    def local(self, q):
        """world design point -> local units"""
        dx, dy = (q[0] - self.x) / self.u, (q[1] - self.y) / self.u
        dx *= self.m
        return (self.c * dx + self.s * dy, -self.s * dx + self.c * dy)

    def wang(self, a):
        """local direction angle (radians, atan2 convention) -> world"""
        v = (math.cos(a), math.sin(a))
        w = (self.m * (self.c * v[0] - self.s * v[1]), self.s * v[0] + self.c * v[1])
        return math.atan2(w[1], w[0])


def _norm_pose(pose):
    if isinstance(pose, str):
        return {pose if pose in POSES else "stand": 1.0}
    out = {k: float(w) for k, w in dict(pose).items() if k in POSES and w > 0}
    return out or {"stand": 1.0}


def _mix_arm(entries):
    """weighted blend of arm entries [(entry, w)]; shape of the heaviest"""
    tot = sum(w for _, w in entries) or 1.0
    ex = sum(e[0][0] * w for e, w in entries) / tot
    ey = sum(e[0][1] * w for e, w in entries) / tot
    wx = sum(e[1][0] * w for e, w in entries) / tot
    wy = sum(e[1][1] * w for e, w in entries) / tot
    ang = sum(e[3] * w for e, w in entries) / tot
    shape = max(entries, key=lambda ew: ew[1])[0][2]
    return (ex, ey), (wx, wy), shape, ang


def solve(sp, pose="stand", t=0.0, facing=0.0, x=0.0, y=0.0, h=100.0, kw=None):
    """pose + kind spec -> Rig (pure)"""
    kw = kw or {}
    R = Rig()
    R.t = t
    R.extra = {}
    poses = _norm_pose(pose)
    pmap = sp.get("pose_map", {})
    # kind-specific pose substitutions (e.g. a warrior's 'stand' carries the lance)
    poses = {(pmap.get(k) if pmap.get(k) in POSES else k): w for k, w in poses.items()}
    if kw.get("pole") and "offer" in poses:
        poses["present"] = poses.pop("offer")
    main = max(poses, key=poses.get)
    tot = sum(poses.values())
    P = {k: _pose_dict(k, t, kw) for k in poses}
    arms = kw.get("arms")
    if arms is not None and arms not in POSES:
        arms = None

    def pget(key, default=0.0):
        return sum(P[k].get(key, default) * w for k, w in poses.items()) / tot

    stance = P[main].get("stance", "stand")
    R.stance = stance
    R.progress = float(kw.get("progress", 1.0))
    R.HL = sp["HL"]
    R.SW = sp["SW"]
    R.u = h / 100.0
    R.x, R.y = x, y
    fac = float(facing)
    R.m = -1.0 if fac < 0 else 1.0
    R.turn = min(1.0, abs(fac))
    R.wind = float(kw.get("wind", pget("wind")))
    lean = pget("lean") + float(kw.get("lean", 0.0))
    R.bow = float(kw.get("bow", 0.0))
    R.nod = pget("nod") + float(kw.get("nod", 0.0))
    R.tilt = pget("tilt") + float(kw.get("tilt", 0.0))
    gz = [0.0, 0.0]
    for k, w in poses.items():
        g = P[k].get("gaze")
        if g:
            gz[0] += g[0] * w / tot
            gz[1] += g[1] * w / tot
    R.gaze = tuple(gz)

    # ---------------------------------------------------------- stance: vertical drop of the upper body
    HL = R.HL
    crown = -100.0
    drop = 0.0
    R.phase = 0.0
    R.sit = stance == "sit"
    R.kneel = stance == "kneel"
    if stance == "sit":
        drop = 21.0
    elif stance == "kneel":
        drop = 24.0
        R.bow += 0.06 * R.turn
    elif stance == "march":
        spd = float(kw.get("speed", 1.0))
        ph = kw.get("phase")
        R.phase = (t * 0.9 * spd) % 1.0 if ph is None else float(ph) % 1.0
        drop = 0.9 * abs(math.sin(TAU * R.phase)) - 0.3
        lean += 0.035 * R.turn
    if main == "plant":
        pr = R.progress
        # wind-up (0..0.42): rise on the toes; thrust (0.42..0.6): crouch forward; then hold
        upk = smoothstep(0.0, 0.4, pr) * (1 - smoothstep(0.42, PLANT_IMPACT, pr))
        thrust = smoothstep(0.45, PLANT_IMPACT, pr)
        settle = smoothstep(PLANT_IMPACT, 1.0, pr)
        drop += -2.0 * upk + 5.5 * thrust * (1 - 0.6 * settle)
        lean += -0.03 * upk + 0.07 * thrust * (1 - 0.5 * settle)
    R.drop = drop
    R.lean = lean
    R.c, R.s = math.cos(lean), math.sin(lean)
    R.hip_y = -50.0 + drop
    if stance == "sit":
        R.hip_y = -30.0
    elif stance == "kneel":
        R.hip_y = -26.0

    def up(p):
        """reference standing upper-body point -> posed local point (drop + bow about the hip)"""
        q = (p[0], p[1] + drop)
        if R.bow:
            q = rot2(q, R.bow, (0.0, R.hip_y))
        return q

    R.extra["up"] = up
    chin = crown + HL
    R.shoulder_y = chin + 4.9
    shy = R.shoulder_y
    # three-quarter view: the body narrows (the far side recedes), the head leads a little toward the facing side
    tq = R.turn
    R.extra["squash"] = 1.0 - 0.13 * tq
    R.head = up((1.4 * tq, crown + HL * 0.5))
    R.head_rot = R.bow + R.tilt
    R.neck = up((0.0, chin + 2.2))
    R.chest = up((0.0, shy + 8.5))
    R.waist_y = up((0.0, -60.0))[1]
    SW = R.SW
    R.sh = {"r": up((-SW * (1.0 - 0.06 * tq), shy)), "l": up((SW * (1.0 - 0.2 * tq), shy))}
    sx = SW / 9.5

    # ---------------------------------------------------------- arms
    R.el, R.wr, R.shape, R.hand_ang = {}, {}, {}, {}
    arm_over = {"R": kw.get("arm_r") or arms, "L": kw.get("arm_l") or arms}
    swap = kw.get("swap")
    if swap is None:
        # a figure turned 3/4 gestures with the arm on the side it faces
        swap = R.turn > 0.3 and main in _GESTURE_SWAP and not arms
    swap = bool(swap)
    R.extra["active"] = "l" if swap else "r"
    for side, sg in (("R", -1.0), ("L", 1.0)):
        src = ("L" if side == "R" else "R") if swap else side
        if arm_over[side] in POSES:
            ent = _arm_of(arm_over[side], src, t, kw)
        elif stance == "march" and main == "march":
            ent = _arm_of(sp.get("march_arms", "offer"), src, t, kw)
        else:
            ents = []
            for k, w in poses.items():
                e_ = P[k].get(src) or _arm_of("stand", src)
                if e_ is _STAND_L and sp.get("l_hold"):
                    e_ = sp["l_hold"]     # the kind's own way of holding its attribute
                ents.append((e_, w))
            ent = _mix_arm(ents)
        (ex, ey), (wx, wy), shape, ang = ent
        fwd = 5.5 * tq if (main in ("march", "offer", "present", "look_up", "clasp", "kneel", "candle") or
                           arm_over[side] in ("offer", "present")) else 0.0
        sq_ = 1.0 - (0.06 if sg < 0 else 0.2) * tq
        el = up((sg * ex * sx * sq_ + fwd * 0.5, shy + ey))
        wr = up((sg * wx * sx * sq_ + fwd, shy + wy))
        key = side.lower()
        R.el[key], R.wr[key], R.shape[key], R.hand_ang[key] = el, wr, shape, ang
    shake = kw.get("shake")
    if shake is not None and R.u:
        R.wr["r"] = (R.wr["r"][0] + shake[0] / R.u * R.m, R.wr["r"][1] + shake[1] / R.u)
        R.el["r"] = (R.el["r"][0] + 0.4 * shake[0] / R.u * R.m, R.el["r"][1] + 0.4 * shake[1] / R.u)
    # per-hand world targets (IK: the elbow is re-placed)
    for key in ("r", "l"):
        tgt = kw.get("hand_" + key)
        if tgt is not None:
            q = R.local(tgt) if R.u else tgt
            R.wr[key] = q
            s = R.sh[key]
            sg = -1.0 if key == "r" else 1.0
            mx, my = (s[0] + q[0]) / 2, (s[1] + q[1]) / 2
            dx, dy = q[0] - s[0], q[1] - s[1]
            d = math.hypot(dx, dy) + 1e-6
            bend = max(0.0, 27.5 * sx - d) * 0.5
            nx, ny = -dy / d, dx / d
            if nx * sg < 0:
                nx, ny = -nx, -ny
            R.el[key] = (mx + nx * bend * 0.8, my + abs(ny) * bend * 0.4 + 1.0)
            shp = kw.get("shape_" + key) or R.shape[key]
            if "ang_" + key in kw:
                R.hand_ang[key] = float(kw["ang_" + key])
            elif shp in _TARGET_ANG:
                R.hand_ang[key] = _TARGET_ANG[shp]
            else:
                fx_, fy_ = q[0] - R.el[key][0], q[1] - R.el[key][1]
                a = math.degrees(math.atan2(fx_, -fy_))
                R.hand_ang[key] = a * (-1 if key == "r" else 1)
        if kw.get("shape_" + key):
            R.shape[key] = kw["shape_" + key]

    # ---------------------------------------------------------- the pole (flags, lances, candles)
    R.pole = None
    R.grip = {}
    if arms in POSES and POSES[arms].get("pole"):
        mode = POSES[arms]["pole"]
    else:
        mode = next((P[k].get("pole") for k in sorted(poses, key=poses.get, reverse=True) if P[k].get("pole")), None)
    mode = kw.get("pole_mode", mode)
    gl = float(kw.get("ground_local", 0.0))
    if kw.get("ground_y") is not None and R.u:
        gl = R.local((x, float(kw["ground_y"])))[1]
    grips = [k for k in ("r", "l") if R.shape[k] == "grip"]
    if mode == "palm":
        grips = ["r"]
    if mode and grips:
        if main == "plant":
            pr = R.progress
            # hands (and pole) rise overhead, then drive down: impact at PLANT_IMPACT
            rise = smoothstep(0.0, 0.42, pr)
            thrust = smoothstep(0.42, PLANT_IMPACT, pr)
            hy_top = shy - 13.0 + drop
            hy_bot = shy + 7.0 + drop
            gy = hy_bot + (hy_top - hy_bot) * rise
            gy = gy + (shy + 9.0 + drop - gy) * thrust
            gx = -8.2 * sx                       # beside the head, never across the face
            k1 = rise * (1 - thrust)
            R.wr["r"] = (gx, gy)
            R.wr["l"] = (gx, gy + 10.5)
            R.el["r"] = (-(15.0 + 3.0 * k1) * sx, gy + 10.0 - 4.0 * k1)
            R.el["l"] = ((6.5 + 3.0 * k1) * sx, gy + 17.0 - 3.0 * k1)
            R.shape["r"] = R.shape["l"] = "grip"
            grips = ["r", "l"]
        a = R.wr[grips[0]]
        b = R.wr[grips[1]] if len(grips) > 1 else None
        if "pole_ang" in kw or mode in ("ground", "plant", "candle", "palm") or b is None:
            ang = float(kw.get("pole_ang", 0.0)) * R.m
            if mode in ("raised",) and "pole_ang" not in kw:
                ang = 0.12
            elif mode == "couched" and "pole_ang" not in kw:
                ang = 0.62
            ang -= R.lean
            dirx, diry = math.sin(ang), -math.cos(ang)
        else:
            dx, dy = a[0] - b[0], a[1] - b[1]
            if dy > 0:
                dx, dy = -dx, -dy
            d = math.hypot(dx, dy) + 1e-9
            dirx, diry = dx / d, dy / d
        # the pole passes through the upper grip; the other fist slides onto the same line
        top = a if (b is None or a[1] <= b[1]) else b
        if b is not None:
            for k in grips:
                g = R.wr[k]
                tt = (g[0] - top[0]) * dirx + (g[1] - top[1]) * diry
                R.wr[k] = (top[0] + dirx * tt, top[1] + diry * tt)
        low = top
        for k in grips:
            if R.wr[k][1] > low[1]:
                low = R.wr[k]
        if mode == "ground":
            tt = (gl - top[1]) / (diry if abs(diry) > 1e-6 else -1e-6)
            base = (top[0] + dirx * tt, gl)
        elif mode == "plant":
            pr = R.progress
            lift = 9.0 + 21.0 * smoothstep(0.0, 0.42, pr)
            lift *= 1 - smoothstep(0.42, PLANT_IMPACT, pr)
            base = (top[0], gl - lift)
            if base[1] - low[1] < 3.0:
                base = (top[0], low[1] + 3.0)
        elif mode == "candle":
            base = (top[0] - dirx * 3.4, top[1] - diry * 3.4)
        elif mode == "raised_high":
            base = (top[0] - dirx * 16.0, top[1] - diry * 16.0)
        elif mode == "palm":
            base = (top[0] - dirx * 13.0, top[1] - diry * 13.0)
        else:  # raised / couched / free: the base a little below the lowest fist
            base = (low[0] - dirx * 9.0, low[1] - diry * 9.0)
        R.pole = (base, (dirx, diry), mode)
        for k in grips:
            R.grip[k] = R.wr[k]
            if R.shape[k] == "grip":
                R.hand_ang[k] = math.degrees(math.atan2(dirx, -diry))   # along the pole
    R.prop = kw.get("prop", None if main in ("candle", "crown_self", "present", "let_go") else sp.get("prop"))
    if R.prop == "none":
        R.prop = None
    R.extra["main"] = main

    # ---------------------------------------------------------- legs / feet (local; drawing decides visibility)
    ph = R.phase
    if stance == "march":
        sw = math.sin(TAU * ph)
        st = 6.5 * sw
        lift_f = max(0.0, math.cos(TAU * ph)) * 2.6
        lift_b = max(0.0, -math.cos(TAU * ph)) * 2.6
        R.feet = {"r": (-2.2 + st, -lift_f), "l": (2.2 - st, -lift_b)}
        R.knees = {"r": (-3.2 + st * 0.6, -26.0), "l": (3.2 - st * 0.6, -26.0)}
    elif stance == "kneel":
        R.feet = {"r": (-5.0, 0.0), "l": (5.0, 0.0)}
        R.knees = {"r": (-6.5, -2.0), "l": (6.5, -2.0)}
    elif stance == "sit":
        R.feet = {"r": (-4.6, -4.2), "l": (4.6, -4.2)}
        R.knees = {"r": (-6.6, -31.0), "l": (6.6, -31.0)}
    else:
        spread = 1.0 + (0.35 if main in ("brace", "charge") else 0.0)
        fwd = 1.0 if main == "charge" else 0.0
        R.feet = {"r": (-3.4 * spread - 1.2 * fwd, 0.0), "l": (3.4 * spread + 1.0 * fwd, -0.5 * fwd)}
        R.knees = {"r": (-4.0 * spread, -27.0), "l": (4.0 * spread, -27.0)}
    R.hem_y = sp.get("hem_y", -4.0)
    return R


# ================================================================ kinds
KINDS = ("lutie", "crocus", "schismmancer", "schism_actual", "jaguar", "courtier", "guard", "deacon", "citizen",
         "heretic", "pope", "crackpot", "philosopher", "people", "hypermind", "egregore")
HUMAN_KINDS = KINDS[:-2]


class Figure:
    """A Byzantine mosaic figure (see the manual). Figure(kind, seed=0, tile=None, **style)."""

    def __init__(self, kind="citizen", seed=0, tile=None, **style):
        from . import icon_body as B
        if kind not in KINDS:
            raise ValueError(f"icon.Figure: unknown kind {kind!r}; kinds: {KINDS}")
        self.kind = kind
        self.seed = int(seed)
        self.tile = tile
        self.style = dict(style)
        self._special = None
        if kind == "hypermind":
            from .icon_hyper import Hypermind
            self._special = Hypermind(seed=seed, **style)
        elif kind == "egregore":
            from .icon_hyper import Egregore
            self._special = Egregore(seed=seed, **style)
        else:
            self.sp = B.make_spec(kind, self.seed, self.style)

    # ------------------------------------------------------------ public
    def anchors(self, x, y, h, pose="stand", t=0.0, facing=0, **kw):
        """pure: the anchors dict draw() would return (no drawing)"""
        if self._special is not None:
            return self._special.anchors(x, y, h, t=t, **kw)
        from . import icon_body as B
        R = solve(self.sp, pose, t, facing, x, y, h, kw)
        return B.anchors(self, R, kw)

    def box(self, h, pose=None, facing=0, **kw):
        """bounding box (x0, y0, x1, y1) as offsets from the ground point for a figure of height h.
        pose=None: conservative, valid for every pose (halo, raised arms, tiara, flailing); else that pose's bbox."""
        if self._special is not None:
            return self._special.box(h)
        if pose is None:
            return (-0.42 * h, -1.3 * h, 0.42 * h, 0.04 * h)
        a = self.anchors(0.0, 0.0, h, pose=pose, facing=facing, **kw)
        return a["bbox"]

    def bust(self, ctx, cx, cy, size, pose="stand", **kw):
        """head + shoulders (a clipeus / roundel portrait) fitting a disc of diameter `size` centred at (cx, cy).
        Same keywords as draw(); returns the anchors. The figure is cut just below the chest."""
        if self._special is not None:
            raise ValueError("bust() is for human kinds")
        HL = self.sp["HL"]
        top = -100.0 - 0.3 * HL
        bot = -100.0 + HL + 4.9 + 16.0
        h = size * 0.9 / (bot - top) * 100.0
        u = h / 100.0
        y = cy - (top + bot) / 2.0 * u
        clip = [(-70.0, top - 60.0), (70.0, top - 60.0), (70.0, bot), (-70.0, bot)]
        return self.draw(ctx, cx, y, h, pose=pose, _bust_clip=clip, **kw)

    def draw(self, ctx, x, y, h, pose="stand", t=0.0, mouth=0.0, look=(0.0, 0.0), expr="neutral", facing=0,
             layer="color", part="all", tile=None, blink=None, alpha=1.0, tint=None, silhouette=None, lod=None,
             **kw):
        """draw into ctx (design units, any transform); returns anchors (world/design coordinates)"""
        if self._special is not None:
            return self._special.draw(ctx, x, y, h, t=t, layer=layer, tile=tile if tile is not None else self.tile,
                                      alpha=alpha, tint=tint, look=look, mouth=mouth, part=part, **kw)
        from . import icon_body as B
        R = solve(self.sp, pose, t, facing, x, y, h, kw)
        tile = self.tile if tile is None else tile
        P = Painter(ctx, layer, tile, alpha, tint, silhouette)
        o = dict(kw)
        o.update(mouth=float(mouth), look=look, expr=blend_expr(expr), part=part, blink=blink, lod=lod,
                 expr_name=expr if isinstance(expr, str) else max(expr, key=expr.get))
        B.draw_figure(P, self, R, o)
        return B.anchors(self, R, kw)


# ================================================================ helpers for scenes
def draw_eye(ctx, x, y, w, open=1.0, look=(0.0, 0.0), rot=0.0, lid="gold", layer="color", tile=None, iris=None,
             alpha=1.0, glow=False):
    """a single Byzantine eye (almond, white, dark iris + pupil, heavy lid line); w = eye width in design px.
    open 0 = shut .. 1 = open; lid 'gold' (gold leaf lid line, in the gold mask) | 'ink' | any colour key."""
    from . import icon_body as B
    P = Painter(ctx, layer, tile, alpha)
    P.push(x, y, rot, w)
    B.single_eye(P, open, look, lid, iris, glow)
    P.pop()


def draw_altar(ctx, x, y, h, layer="color", tile=None, alpha=1.0, tint=None):
    """a tiny altar of one (table, white cloth, porphyry antependium, gold ciborium on four columns).
    (x, y) = ground centre, h = total height (dome top). Returns dict(top=(x, y), table=(x, y))."""
    from .icon_body import draw_altar as _da
    return _da(ctx, x, y, h, layer, tile, alpha, tint)


def cartoon(draw_fn, s=1.0, w=None, h=None, extent=None, bg=None, layers=("color", "fine", "gold", "silver", "pearl",
                                                                        "glow", "edges")):
    """Render one drawing on every layer at pixel scale s.

    draw_fn(ctx, layer) must draw the whole picture (figures, crowds ...) passing layer=layer to every icon call.
    extent = (w, h) design units of the canvas (default 1920x1080); w/h = pixel size override.
    Returns dict(rgba=premultiplied (H,W,4), rgb=(H,W,3) over bg (or black), alpha=(H,W), fine, gold, silver, pearl,
    glow, edges=(H,W) float 0..1 masks, ground=1-alpha, s, extent)."""
    ex = extent or (1920, 1080)
    W_ = w or int(round(ex[0] * s))
    H_ = h or int(round(ex[1] * s))
    out = {"s": s, "extent": ex}
    for L in layers:
        cv = Canvas(s=s, w=W_, h=H_)
        draw_fn(cv.ctx, L)
        a = cv.rgba()
        if L == "color":
            out["rgba"] = a
            out["alpha"] = a[..., 3].copy()
            if bg is not None:
                b = np.array(rgb(bg), np.float32)
                out["rgb"] = a[..., :3] + b * (1 - a[..., 3:4])
            else:
                out["rgb"] = a[..., :3].copy()
            out["ground"] = 1.0 - out["alpha"]
        else:
            out[L] = a[..., 3].copy()
    return out


def crowd(ctx, people, t=0.0, layer="color", tile=None, sort=True):
    """Draw many figures (back to front by y). people: [dict(kind, x, y, h, seed=0, pose='stand', facing=0, expr,
    mouth, look, t_off, alpha, tint, ...any draw kw)]. Figures are cached per (kind, seed, style); small figures
    automatically use the simplified LOD. Returns the list of anchors."""
    items = sorted(people, key=lambda d: d["y"]) if sort else list(people)
    out = []
    for d in items:
        d = dict(d)
        kind = d.pop("kind", "people")
        seed = int(d.pop("seed", 0))
        style = d.pop("style", None) or {}
        fig = _crowd_fig(kind, seed, tuple(sorted(style.items())))
        x, y, hh = d.pop("x"), d.pop("y"), d.pop("h")
        tt = t + d.pop("t_off", 0.0)
        out.append(fig.draw(ctx, x, y, hh, t=tt, layer=layer, tile=d.pop("tile", tile), **d))
    return out


_CROWD = {}


def _crowd_fig(kind, seed, style):
    key = (kind, seed, style)
    f = _CROWD.get(key)
    if f is None:
        if len(_CROWD) > 4000:
            _CROWD.clear()
        f = _CROWD[key] = Figure(kind, seed, **dict(style))
    return f


def people_variants(n, seed=0, kinds=("people",)):
    """n deterministic (kind, seed) pairs for crowds"""
    return [(kinds[int(hash01(i, seed + 77) * len(kinds)) % len(kinds)], seed * 1000 + i) for i in range(n)]

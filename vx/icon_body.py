"""Human figure drawing for vx.icon: kinds (specs), faces, hair, beards, headgear, hands, garments, props.

Everything here draws through an icon.Painter in LOCAL units (figure height 100, origin at the ground point, y down).
Heads are drawn in HEAD units (crown -0.5 .. chin +0.5), hands in HAND units (wrist 0 .. fingertip 1).
"""
import math

from .ease import hash01, clamp
from .icon import IP, TAU, mixc, lerp2, rot2, auto_blink

FINE = "fine"


def _hs(seed, k):
    return hash01(k, 1000 + seed)


def _pick(seed, k, seq):
    return seq[int(_hs(seed, k) * len(seq)) % len(seq)]


# ================================================================ skins
SKINS = {
    "light": dict(flesh="#E3B196", flesh_l="#F2D2B8", flesh_m="#C98F72", olive="#958B69", olive_d="#666148",
                  contour="#5A2A1E", cheek="#DC7B6F", lip_u="#8E2C27", lip_l="#C4574C", hl="#FCF3E6"),
    "warm": dict(flesh="#D69C79", flesh_l="#E8BD9B", flesh_m="#B27B5D", olive="#887C59", olive_d="#58533B",
                 contour="#4E2419", cheek="#D16C60", lip_u="#86281F", lip_l="#BB4F43", hl="#F8EBDA"),
    "olive": dict(flesh="#C08B63", flesh_l="#D8A77E", flesh_m="#9E6D4B", olive="#776C4B", olive_d="#4C4531",
                  contour="#40200F", cheek="#BA604F", lip_u="#7A2419", lip_l="#A9483A", hl="#F1DFC6"),
    "brown": dict(flesh="#94613F", flesh_l="#B07C56", flesh_m="#754B30", olive="#5F5337", olive_d="#3B3322",
                  contour="#2A140A", cheek="#9D4C3B", lip_u="#5A2016", lip_l="#83382C", hl="#D9B696"),
    "dark": dict(flesh="#6C4531", flesh_l="#8A5B40", flesh_m="#533423", olive="#4B4131", olive_d="#2D261A",
                 contour="#1C0E07", cheek="#7C3B2F", lip_u="#4A1C16", lip_l="#6E302A", hl="#C4A084"),
}
SKIN_ORDER = ("light", "warm", "olive", "brown", "dark")

# cloth families (base, dark, light, line)
FAMS = ("ochre", "black", "navy", "white", "porphyry", "lapis", "grey_c", "blue_c", "teal_c", "plum_c", "rust_c",
        "green_c", "sand_c", "leather", "steel", "drab", "red")
PEOPLE_FAMS = ("grey_c", "blue_c", "teal_c", "plum_c", "rust_c", "green_c", "sand_c", "white")
HAIR_COLS = (("#1C1411", "#0E0A08", "#4A3526"), ("#3B271C", "#1E130D", "#6E4E37"), ("#5A3A22", "#2E1C10", "#8C6440"),
             ("#2A2624", "#141210", "#58514B"), ("#7E7B76", "#4E4C49", "#B4B1AA"), ("#E6E2DA", "#9AA1AA", "#FFFFFF"))


def fam(name):
    return name, name + "_d", name + "_l", name + "_k"


# ================================================================ kind specs
def make_spec(kind, seed, style):
    """kind + seed + style overrides -> spec dict (proportions, colours, costume)"""
    s = seed
    sp = dict(kind=kind, seed=seed, HL=13.4, SW=9.5, hem_y=-4.0, hem_w=12.2, waist_w=9.0, hip_w=10.0, face_w=0.35,
              jaw=1.0, nose=1.0, eye=1.0, mouth_w=1.0, sex="m", age="adult", skin=SKINS["warm"],
              hair=dict(style="short", c=HAIR_COLS[1]), beard=None, head=None, halo=False, halo_style="gold",
              dress="tunic", robe="grey_c", over=None, clavi=None, sleeve=None, cuff=None, shoes="leather",
              prop=None, pose_map={}, march_arms="offer", goggles=0.0, headset=False, tail=False, shield=None,
              brows=None, wrinkles=0, tiara=False, crown=False, altar=False, border=None, belt=None)
    if kind == "lutie":
        sp.update(HL=14.3, SW=8.7, hem_w=11.2, waist_w=8.2, hip_w=9.2, face_w=0.355, jaw=1.06, nose=0.84, eye=1.08,
                  mouth_w=0.92, sex="f", age="young", skin=SKINS["light"], halo=True,
                  hair=dict(style="lutie", c=("#2E1E16", "#150D09", "#5E4230")), dress="novice", robe="ochre",
                  sleeve="ochre", cuff="ochre_d", shoes="sandal", prop="weld", head="hood_down", belt="ochre_k",
                  march_arms="offer", pose_map={"raise": "salute"}, l_hold=((13.2, 13.5), (11.4, 21.5), "fist", -92))
    elif kind == "crocus":
        sp.update(HL=13.3, SW=9.9, hem_w=12.8, waist_w=9.8, hip_w=10.8, face_w=0.335, jaw=0.95, nose=1.12, eye=1.0,
                  sex="m", age="old", skin=SKINS["light"], halo=True, halo_style="gold_pearl",
                  hair=dict(style="bald", c=HAIR_COLS[5]), beard=dict(style="patriarch", c=HAIR_COLS[5]),
                  brows=HAIR_COLS[4], dress="hierarch", robe="black", over="ochre", sleeve="black", cuff="gold",
                  shoes="black", prop="book", head="veil", wrinkles=2, march_arms="bless")
    elif kind in ("schismmancer", "schism_actual"):
        sk = SKINS[_pick(s, 1, SKIN_ORDER)]
        beard = _pick(s, 2, (None, None, "short", "pointed", "stubble"))
        hc = HAIR_COLS[int(_hs(s, 3) * 4)]
        sp.update(HL=13.0, SW=10.3, hem_y=-31.0, hem_w=12.8, waist_w=9.4, hip_w=10.6, face_w=0.345, jaw=1.04,
                  skin=sk, halo=True, hair=dict(style="short", c=hc),
                  beard=dict(style=beard, c=hc) if beard else None, dress="warrior", robe="ochre", sleeve="ochre",
                  cuff="ochre_d", shoes="boots", head="helmet", prop=None, over="ochre_d",
                  goggles=float(style.get("goggles", 0.0)), headset=kind == "schism_actual",
                  pose_map={"stand": "lance", "raise": "salute"}, march_arms="lance")
        if kind == "schism_actual":
            sp.update(prop="orb", beard=dict(style="short", c=hc))
            sp["pose_map"] = {"stand": "command", "raise": "salute"}
    elif kind == "jaguar":
        sp.update(HL=19.0, SW=10.6, hem_w=12.6, waist_w=10.0, hip_w=10.8, face_w=0.44, sex="m", halo=True,
                  halo_style="gold_pearl", dress="emperor", robe="white", over="navy", sleeve="white", cuff="gold",
                  shoes="red", prop="edict", head="stemma", tail=True, hair=dict(style="none", c=HAIR_COLS[0]),
                  march_arms="offer")
    elif kind in ("courtier", "guard", "deacon"):
        sk = SKINS[_pick(s, 1, SKIN_ORDER[:4])]
        hc = HAIR_COLS[int(_hs(s, 3) * 4)]
        beard = _pick(s, 2, (None, "short", "short", "full", None))
        sp.update(skin=sk, hair=dict(style=_pick(s, 4, ("short", "short", "cap")), c=hc),
                  beard=dict(style=beard, c=hc) if beard else None)
        if kind == "courtier":
            sp.update(dress="courtier", robe="white", over="white", clavi="porphyry", tablion="porphyry",
                      sleeve="white", shoes="black", prop=None, march_arms="offer")
        elif kind == "guard":
            sp.update(dress="guard", robe=_pick(s, 5, ("white", "sand_c", "grey_c")), over=None, hem_y=-30.0,
                      sleeve=None, shoes="boots", shield="oval", torque=True, prop=None, SW=10.2, spear=True,
                      pose_map={"stand": "lance", "raise": "salute"}, march_arms="lance")
            sp["sleeve"] = sp["robe"]
        else:
            sp.update(dress="deacon", robe="white", clavi="porphyry", sleeve="white", shoes="black", prop="codex_open",
                      beard=None, age="young", pose_map={"stand": "offer"}, march_arms="offer")
    elif kind in ("citizen", "people", "heretic", "pope"):
        sex = style.get("sex") or ("f" if _hs(s, 11) < 0.45 else "m")
        sk = SKINS[_pick(s, 1, SKIN_ORDER)]
        hc = HAIR_COLS[min(5, int(_hs(s, 3) * (5 if sex == "f" else 6)))]
        age = _pick(s, 12, ("young", "adult", "adult", "old"))
        if age == "old" and sex == "m":
            hc = HAIR_COLS[4 + int(_hs(s, 13) * 2)]
        robe = _pick(s, 5, PEOPLE_FAMS)
        over = _pick(s, 6, PEOPLE_FAMS + (None, None))
        if over == robe:
            over = None
        if sex == "f":
            hair = dict(style=_pick(s, 4, ("veil", "veil", "bun", "long_f")), c=hc)
            beard = None
        else:
            hair = dict(style=_pick(s, 4, ("short", "short", "cap", "long", "bald")), c=hc)
            beard = _pick(s, 2, (None, None, "short", "full", "stubble", "pointed"))
            beard = dict(style=beard, c=hc) if beard else None
        sp.update(sex=sex, skin=sk, hair=hair, beard=beard, age=age, dress="tunic", robe=robe, over=over,
                  clavi=_pick(s, 7, (None, "porphyry", "navy", "black", None)), sleeve=robe, shoes="leather",
                  veil_c=_pick(s, 8, ("lapis", "porphyry", "teal_c", "plum_c", "white", "navy")),
                  HL=13.6 if sex == "f" else 13.4, SW=8.9 if sex == "f" else 9.6, hem_w=11.6 if sex == "f" else 12.0,
                  face_w=0.345 if sex == "f" else 0.35, jaw=1.02 if sex == "f" else 1.0, nose=0.92 if sex == "f" else 1.0,
                  eye=1.04 if sex == "f" else 1.0)
        if age == "old":
            sp["wrinkles"] = 1
        if kind == "citizen":
            sp.update(crown=bool(style.get("crown", True)), halo=False)
        elif kind == "heretic":
            sp.update(halo=True, robe=_pick(s, 5, ("white", "white", "grey_c", "blue_c", "porphyry", "teal_c", "sand_c")),
                      tiara=bool(style.get("tiara", False)))
            sp["sleeve"] = sp["robe"]
        elif kind == "pope":
            sp.update(dress="papal", tiara=True, halo=False, robe="white",
                      over=_pick(s, 6, ("porphyry", "red", "lapis", "green_c", "plum_c")), sleeve="white",
                      shoes="red", prop="church", sex="m", beard=beard if sex == "m" else None,
                      hair=dict(style="short", c=hc) if hair["style"] in ("veil", "bun", "long_f") else hair)
    elif kind in ("philosopher", "crackpot"):
        sk = SKINS[_pick(s, 1, SKIN_ORDER[:4])]
        hc = HAIR_COLS[_pick(s, 3, (0, 1, 2, 3, 4, 4, 5))]
        beard = _pick(s, 2, ("full", "full", "long", "short", "pointed"))
        hair = _pick(s, 4, ("short", "bald", "long", "cap"))
        pose_map = {}
        if kind == "crackpot":
            hair, beard = "wild", _pick(s, 2, ("long", "full", "wild"))
            hc = HAIR_COLS[_pick(s, 3, (4, 4, 5, 5))]
        sp.update(skin=sk, hair=dict(style=hair, c=hc), beard=dict(style=beard, c=hc), age="old", wrinkles=1 + (kind == "crackpot"),
                  dress="himation", robe=_pick(s, 5, ("white", "sand_c", "grey_c", "white")),
                  over=_pick(s, 6, ("sand_c", "grey_c", "rust_c", "white", "blue_c", "plum_c")), sleeve=None,
                  shoes="sandal", prop="scroll", pose_map=pose_map, march_arms="stand")
        if sp["over"] == sp["robe"]:
            sp["over"] = "rust_c"
    # style overrides (colour keys, props, toggles)
    for k in ("robe", "over", "sleeve", "cuff", "shoes", "prop", "halo", "halo_style", "head", "clavi", "sex",
              "shield", "tail", "tiara", "crown", "hem_w", "headset", "altar", "veil_c", "cloak", "tablion", "shield_c"):
        if k in style:
            sp[k] = style[k]
    if "skin" in style:
        sp["skin"] = SKINS[style["skin"]] if isinstance(style["skin"], str) else style["skin"]
    if "hair" in style:
        sp["hair"] = dict(sp["hair"], style=style["hair"])
    if "hair_color" in style:
        c = style["hair_color"]
        sp["hair"] = dict(sp["hair"], c=(c, mixc(c, "#000000", 0.5), mixc(c, "#FFFFFF", 0.3)))
    if "beard" in style:
        sp["beard"] = dict(style=style["beard"], c=sp["hair"]["c"]) if style["beard"] else None
    if "head_scale" in style:
        sp["HL"] *= float(style["head_scale"])
    if "build" in style:
        b = float(style["build"])
        for k in ("SW", "hem_w", "waist_w", "hip_w"):
            sp[k] *= b
    if not sp.get("prop") and not sp.get("l_hold"):
        sp["l_hold"] = ((11.6, 15.0), (9.4, 24.5), "relax", 184)
    if sp.get("tiara") and sp["kind"] != "pope":
        sp["head"] = "tiara"
    elif sp["kind"] == "pope":
        sp["head"] = "tiara"
    elif sp.get("crown") and sp["kind"] == "citizen":
        sp["head"] = "tiny_crown"
    return sp


# ================================================================ drawing entry
def _lod(P, R):
    """0 full .. 2 tiny, from the head size in (fine) tiles or device pixels"""
    m = P.ctx.get_matrix()
    dev = math.sqrt(abs(m.xx * m.yy - m.xy * m.yx))      # device px per local unit
    head_px = R.HL * dev
    if P.tile > 0:
        head_design = R.HL * R.u
        n = head_design / max(P.fine_tile, 1e-6)
        return 0 if n >= 28.0 else (1 if n >= 10.5 else 2)
    return 0 if head_px >= 90 else (1 if head_px >= 30 else 2)


def draw_figure(P, F, R, o):
    sp = F.sp
    ctx = P.ctx
    P.push(R.x, R.y, 0.0, R.u * R.m, R.u)
    if R.lean:
        ctx.rotate(R.lean)
    lod = o.get("lod")
    P.lod = _lod(P, R) if lod is None else int(lod)
    part = o.get("part", "all")
    halo = o.get("halo", sp.get("halo"))
    bust = o.get("_bust_clip")
    if bust is not None:
        P.clip(bust, smooth=False)
    if part in ("all", "back"):
        if halo:
            draw_halo(P, sp, R, halo if isinstance(halo, str) else sp.get("halo_style", "gold"))
        _costume(P, F, R, o, "back")
        draw_head(P, F, R, o, "back")
        _costume(P, F, R, o, "body")
        draw_head(P, F, R, o, "neck")
        _costume(P, F, R, o, "collar")
        draw_head(P, F, R, o, "front")
        _costume(P, F, R, o, "arms")
    if part in ("all", "front"):
        for side in ("r", "l"):
            if R.shape[side] == "grip" and R.pole is not None:
                draw_hand(P, F, R, side, front=True)
    if bust is not None:
        P.unclip()
    P.pop()


# ================================================================ halo
def halo_geom(sp, R):
    """(cx, cy, r) in local units"""
    HL = R.HL
    k = 0.8 if sp.get("head") in ("helmet", "tiara", "stemma", "veil") else 0.76
    if sp["kind"] == "jaguar":
        k = 0.84
    c = rot2((0.0, -0.1 * HL), R.head_rot)
    return R.head[0] + c[0], R.head[1] + c[1], k * HL


def draw_halo(P, sp, R, style="gold"):
    cx, cy, r = halo_geom(sp, R)
    if style in ("square", "square_gold"):
        s = r * 0.98
        fillc = "gold" if style == "square_gold" else "#B9C7D8"
        m = "gold" if style == "square_gold" else None
        pts = [(cx - s, cy - s * 1.02), (cx + s, cy - s * 1.02), (cx + s, cy + s * 0.9), (cx - s, cy + s * 0.9)]
        P.poly(pts, fill=fillc, mat=m)
        w = P.width(0.075 * R.HL, "contour")
        inset = [(cx - s + w / 2, cy - s * 1.02 + w / 2), (cx + s - w / 2, cy - s * 1.02 + w / 2),
                 (cx + s - w / 2, cy + s * 0.9 - w / 2), (cx - s + w / 2, cy + s * 0.9 - w / 2)]
        P.poly(inset, line="#5A2418", lw=w, cls="contour")
        return
    P.circle(cx, cy, r, fill="gold", mat="gold")
    if P.lod == 0 and style == "gold_pearl":
        n = 26
        rr = r * 0.84
        P.dots([(cx + rr * math.cos(i * TAU / n), cy + rr * math.sin(i * TAU / n)) for i in range(n)],
               r * 0.045, "pearl", mat="pearl")
    w = P.width(0.075 * R.HL, "contour")
    P.circle(cx, cy, r - w / 2, line="#5A2418", lw=w, cls="thin")


# ================================================================ heads
_FACE_HALF = ((0.0, 0.505), (0.125, 0.483), (0.228, 0.42), (0.297, 0.31), (0.336, 0.17), (0.35, 0.02),
              (0.344, -0.13), (0.318, -0.26), (0.25, -0.36), (0.13, -0.418), (0.0, -0.435))


def face_outline(sp):
    fw = sp["face_w"] / 0.35
    jw = sp["jaw"]
    half = [(x * fw * (jw if y > 0.15 else 1.0), y) for x, y in _FACE_HALF]
    right = half[1:-1]
    left = [(-x, y) for x, y in reversed(right)]
    return [half[0]] + right + [half[-1]] + left


def _fx_fn(R):
    al = 0.55 * R.turn
    if al <= 1e-4:
        return lambda x: x
    TR = 0.38

    def fx(x):
        return TR * math.sin(math.asin(clamp(x / TR, -1.0, 1.0)) + al)
    return fx


def draw_head(P, F, R, o, stage):
    sp = F.sp
    P.push(R.head[0], R.head[1], R.head_rot, R.HL)
    if sp["kind"] == "jaguar" and stage != "neck":
        jaguar_head(P, sp, R, o, stage)
    elif stage == "back":
        hair_back(P, sp, R, o)
        headgear(P, sp, R, o, "back")
    elif stage == "neck":
        draw_neck(P, sp, R)
    else:
        draw_face(P, sp, R, o)
        hair_front(P, sp, R, o)
        draw_beard(P, sp, R, o)
        headgear(P, sp, R, o, "front")
    P.pop()


def _neck_bottom(R):
    """neck base in head units (just below the posed shoulder line)"""
    shy = 0.5 * (R.sh["r"][1] + R.sh["l"][1])
    return max(0.55, (shy + 1.6 - R.head[1]) / R.HL)


def draw_neck(P, sp, R):
    S = sp["skin"]
    nb = _neck_bottom(R)
    nw = 0.185 * (sp["face_w"] / 0.35) * (1.12 if sp["sex"] == "m" else 0.94)
    if sp["kind"] == "jaguar":
        pts = [(-0.24, 0.3), (-0.23, 0.55), (-0.27, nb), (0.27, nb), (0.23, 0.55), (0.24, 0.3)]
        P.poly(pts, fill="fur_d", line="rosette", lw=0.03, cls="fine2", mat=FINE, smooth=True)
        return
    pts = [(-nw * 0.9, 0.2), (-nw, 0.62), (-nw * 1.14, nb), (nw * 1.14, nb), (nw, 0.62), (nw * 0.9, 0.2)]
    P.poly(pts, fill=S["flesh"], mat=FINE)
    if P.lod < 2:
        P.clip(pts, smooth=False)
        P.poly([(-nw * 1.2, 0.2), (-nw * 1.2, 0.5), (-nw * 0.4, 0.6), (nw * 0.2, 0.64), (nw * 1.2, 0.56), (nw * 1.2, 0.2)],
               fill=S["olive"], mat=FINE, smooth=True)
        P.poly([(nw * 0.55, 0.5), (nw * 1.3, 0.45), (nw * 1.3, nb + 0.1), (nw * 0.8, nb + 0.1)], fill=S["flesh_m"], mat=FINE)
        if P.lod == 0:
            P.line([(-nw * 0.45, 0.7), (-nw * 0.3, 0.86)], S["olive_d"], 0.022, "fine2", FINE)
            P.line([(nw * 0.5, 0.68), (nw * 0.32, 0.86)], S["olive_d"], 0.022, "fine2", FINE)
        P.unclip()
    P.line([(-nw * 0.9, 0.2), (-nw, 0.62), (-nw * 1.14, nb)], S["contour"], 0.03, "fine2", FINE, smooth=False)
    P.line([(nw * 0.9, 0.2), (nw, 0.62), (nw * 1.14, nb)], S["contour"], 0.03, "fine2", FINE, smooth=False)


def _ears(P, sp, R, S, fx):
    if sp["hair"]["style"] in ("lutie", "long", "long_f", "veil", "wild") or sp.get("head") in ("veil", "helmet"):
        return
    al = 0.55 * R.turn
    for sg in (-1, 1):
        if sg > 0 and al > 0.3:
            continue
        sh = -al * 0.06 if sg < 0 else 0.0
        fw = sp["face_w"] / 0.35
        pts = [(sg * (0.325 * fw) + sh, -0.03), (sg * 0.372 * fw + sh, -0.055), (sg * 0.408 * fw + sh, 0.0),
               (sg * 0.412 * fw + sh, 0.1), (sg * 0.386 * fw + sh, 0.18), (sg * 0.345 * fw + sh, 0.215),
               (sg * 0.322 * fw + sh, 0.19)]
        P.poly(pts, fill=S["flesh_m"], line=S["contour"], lw=0.026, cls="fine2", mat=FINE, smooth=True)
        if P.lod == 0:
            P.line([(sg * 0.35 * fw + sh, 0.0), (sg * 0.385 * fw + sh, 0.04), (sg * 0.378 * fw + sh, 0.13),
                    (sg * 0.352 * fw + sh, 0.16)], S["olive_d"], 0.02, "fine2", FINE)


def eye_closures(sp, R, o, lid0, seed_extra=0):
    """-> {local eye side sg (-1/+1): lid closure} from expr lid, auto blinks, blink=(screen_left, screen_right),
    wink=0..1 (closes the screen-right eye; wink_side='l' for the other)"""
    blink = o.get("blink")
    out = {}
    auto = auto_blink(R.t, sp["seed"] + seed_extra)
    for sg in (-1, 1):
        scr = 0 if sg * R.m < 0 else 1          # 0 = eye on screen left
        if blink is None:
            bk = auto
        elif blink is True:
            bk = 1.0
        elif blink is False:
            bk = 0.0
        elif isinstance(blink, (tuple, list)):
            bk = float(blink[scr])
        else:
            bk = float(blink)
        w = float(o.get("wink", 0.0) or 0.0)
        if w and scr == (0 if o.get("wink_side") == "l" else 1):
            bk = max(bk, w)
        if o.get("eyes_closed"):
            bk = 1.0
        out[sg] = clamp(lid0 + (1.0 - lid0) * bk, 0.0, 1.0)
    return out


def draw_face(P, sp, R, o):
    S = sp["skin"]
    lod = P.lod
    e = o["expr"]
    fx = _fx_fn(R)
    al = 0.55 * R.turn
    fw = sp["face_w"] / 0.35
    out = face_outline(sp)
    _ears(P, sp, R, S, fx)
    P.poly(out, fill=S["flesh"], mat=FINE, smooth=True)
    ex = 0.157 * fw
    ey = 0.02
    ew = 0.122 * sp["eye"] * (1.08 if lod == 1 else 1.0)
    nose = sp["nose"]
    my = 0.352 + (nose - 1.0) * 0.06
    old = sp.get("wrinkles", 0)
    if lod < 2:
        # ---- banded modelling (the Byzantine way): concentric tone bands just inside the contour
        P.clip(out)
        P.push(-0.028 - al * 0.05, -0.014)
        P.poly(out, line=S["flesh_m"], lw=0.26, cls="thin", mat=FINE, smooth=True)
        P.poly(out, line=S["olive"], lw=0.09, cls="fine2", mat=FINE, smooth=True)
        P.pop()
        # lit planes: forehead, cheekbones, chin
        P.ellipse(fx(-0.02), -0.2, 0.2 * fw, 0.1, fill=S["flesh_l"], mat=FINE)
        for sg in (-1, 1):
            P.ellipse(fx(sg * 0.16 * fw), 0.15, 0.1 * fw, 0.075, fill=S["flesh_l"], mat=FINE)
        P.ellipse(fx(0.0), 0.44, 0.07, 0.035, fill=S["flesh_l"], mat=FINE)
        # eye sockets (olive under the brows, around the eye) - full detail only: in coarse tiles they merge the
        # brow into the eye
        if lod == 0:
            for sg in (-1, 1):
                cx = fx(sg * ex)
                P.ellipse(cx + sg * 0.01, ey - 0.02, ew * 1.32, ew * 0.82, fill=S["flesh_m"], mat=FINE)
                P.ellipse(cx + sg * 0.012, ey + 0.012, ew * 1.1, ew * 0.6, fill=S["olive"], mat=FINE)
        # cheeks (pink), the Byzantine blush spots
        bl = clamp(e["blush"], 0.0, 1.6)
        ck = mixc(S["flesh"], S["cheek"], min(1.0, 0.75 * bl))
        for sg in (-1, 1):
            P.ellipse(fx(sg * 0.205 * fw), 0.205, 0.066 * fw, 0.05, fill=ck, mat=FINE)
        # nose: shadow side strip
        P.poly([(fx(0.016), -0.03), (fx(0.05), -0.06), (fx(0.062), 0.08 * nose), (fx(0.066), 0.2 * nose),
                (fx(0.08), 0.265 * nose), (fx(0.02), 0.27 * nose), (fx(0.012), 0.12 * nose)],
               fill=S["olive"], mat=FINE, smooth=False)
        # under the lower lip / chin
        P.ellipse(fx(0.0), my + 0.066, 0.05, 0.016, fill=S["olive"], mat=FINE)
        if old:
            for k in range(min(old, 2)):
                yy = -0.225 - 0.055 * k
                P.line([(fx(-0.13 + 0.03 * k), yy + 0.01), (fx(0.0), yy - 0.004), (fx(0.13 - 0.03 * k), yy + 0.01)],
                       S["flesh_m"], 0.018, "fine2", FINE)
            for sg in (-1, 1):
                P.line([(fx(sg * 0.1), 0.2), (fx(sg * 0.14), 0.3), (fx(sg * 0.15), 0.38)], S["flesh_m"], 0.02, "fine2", FINE)
        P.unclip()
    # ---- eyes, brows, nose, mouth
    lids = eye_closures(sp, R, o, e["lid"], 3 * (sp["kind"] == "crocus"))
    lx, ly = o.get("look", (0.0, 0.0)) or (0.0, 0.0)
    gx, gy = R.gaze
    lx = clamp((lx + gx) * R.m, -1.0, 1.0)
    ly = clamp(ly + gy, -1.0, 1.0)
    nod = clamp(R.nod, -0.5, 0.5)
    P.push(0.0, -0.1 * nod, 0.0, 1.0, 1.0 - 0.14 * nod)      # head tilted back: features ride up, nose shortens
    for sg in (-1, 1):
        lid = lids[sg]
        if ly < -0.3:
            lid = max(0.0, lid - 0.15 * (-ly)) if lid < 0.9 else lid
        cx = fx(sg * ex)
        wk = 1.0 - (0.22 * al if sg > 0 else -0.06 * al)
        eye(P, cx, ey, ew * wk, sg, lid, e["sq"], e["wide"], (lx, ly), S, lod, sp)
    brows(P, sp, R, e, fx, ex, ey, ew, S, lod)
    nose_lines(P, sp, S, fx, nose, lod)
    draw_mouth(P, sp, S, e, o.get("mouth", 0.0), fx, my, lod)
    P.pop()
    if lod == 0:
        hl = S["hl"]
        P.line([(fx(-0.006), 0.04 * nose), (fx(-0.01), 0.12 * nose), (fx(-0.012), 0.2 * nose)], hl, 0.018, "fine2", FINE)
        for sg in (-1, 1):
            P.line([(fx(sg * 0.16 * fw), 0.12), (fx(sg * 0.22 * fw), 0.118)], S["flesh_l"], 0.018, "fine2", FINE)
        P.ellipse(fx(0.0), 0.455, 0.03, 0.014, fill=hl, mat=FINE)
    # ---- the contour (last: one crisp dark course around the face)
    P.poly(out, line=S["contour"], lw=0.036, cls="fine", mat=FINE, smooth=True)


def _eye_pts(cx, cy, w, s, wide, sq):
    hu = w * (0.58 + 0.2 * wide)
    hl = w * (0.37 + 0.12 * wide) * (1.0 - 0.5 * sq)
    inner = (cx - s * 0.97 * w, cy + 0.1 * w)
    outer = (cx + s * w, cy - 0.03 * w)
    up = [inner, (cx - s * 0.6 * w, cy - 0.64 * hu), (cx - s * 0.06 * w, cy - hu), (cx + s * 0.5 * w, cy - 0.8 * hu), outer]
    lo = [outer, (cx + s * 0.5 * w, cy + 0.74 * hl), (cx - s * 0.1 * w, cy + hl), (cx - s * 0.62 * w, cy + 0.64 * hl), inner]
    return up, lo, hu, hl


def eye(P, cx, cy, w, s, lid, sq, wide, look, S, lod, sp=None, lidcol=None, lidmat=FINE, iris=None, glow=False,
        lidfill=None, lidfill_mat=FINE):
    """one almond eye in the current units; s=+1 for the eye on the image's right (inner corner toward -x)"""
    up, lo, hu, hl = _eye_pts(cx, cy, w, s, wide, sq)
    white = up + lo[1:-1]
    irc = iris or "iris"
    lc = lidcol or "ink"
    wm = FINE + (" glow" if glow else "")
    if lod >= 2:
        # tiny: a dark almond with a light corner
        if lid > 0.7:
            P.line([up[0], lerp2(up[2], lo[2], 0.6), up[-1]], lc, 0.3 * w, "fine2", lidmat)
            return
        P.poly(white, fill=irc, mat=wm, smooth=True)
        P.line(up, lc, 0.18 * w, "fine2", lidmat)
        return
    P.poly(white, fill="eye_w", mat=wm, smooth=True)
    ix = cx + s * 0.05 * w + look[0] * 0.4 * w
    iy = cy + 0.12 * w + look[1] * 0.3 * w
    ir = 0.5 * w if lod == 0 else 0.47 * w
    P.clip(white)
    P.circle(ix, iy, ir, fill=irc, mat=wm)
    if lod == 0:
        P.circle(ix, iy, ir * 0.52, fill="pupil", mat=wm)
        P.circle(ix - 0.18 * w, iy - 0.2 * w, ir * 0.2, fill="eye_w", mat=wm)
    # the upper lid comes down
    closed = [lerp2(up[0], up[-1], i / 4.0) for i in range(5)]
    closed = [(p[0], p[1] + 0.45 * hl * math.sin(math.pi * i / 4.0)) for i, p in enumerate(closed)]
    edge = [lerp2(up[i], closed[i], lid) for i in range(5)]
    if lid > 0.02:
        P.poly(up + list(reversed(edge))[1:-1], fill=lidfill or (S["flesh_m"] if S else "flesh_m"), mat=lidfill_mat,
               smooth=False)
    P.unclip()
    # lid line + tail, crease, lower lid
    tail = (up[-1][0] + s * 0.16 * w, up[-1][1] + 0.1 * w)
    P.line(edge + [tail], lc, 0.2 * w if lod == 0 else 0.3 * w, "fine", lidmat)
    if lod == 0 and lid < 0.8:
        cr = [(p[0], p[1] - 0.3 * w) for p in up[1:]]
        P.line(cr, (S["contour"] if S else "contour"), 0.085 * w, "fine2", FINE)
    if lid < 0.9 and lod == 0:
        P.line(lo[:4], (S["contour"] if S else "contour"), 0.09 * w, "fine2", FINE)
    if lod == 0 and S:
        P.line([(p[0], p[1] + 0.3 * w) for p in lo[1:4]], S["olive"], 0.1 * w, "fine2", FINE)


def single_eye(P, open_, look, lid, iris, glow):
    """a free eye in unit coordinates (width 1) for draw_eye(): gold lids are dark gold leaf (gold mask) over a
    bright-gold eyelid, so the eye reads on a gold ground"""
    if lid == "gold":
        lidcol, lmat, lfill, lfmat = "gold_d", "fine gold", "gold_l", "fine gold"
    else:
        lidcol, lmat, lfill, lfmat = ("ink" if lid == "ink" else lid), FINE, None, FINE
    eye(P, 0.0, 0.0, 0.5, 1, 1.0 - clamp(open_, 0.0, 1.0), 0.0, 0.0, (clamp(look[0], -1, 1), clamp(look[1], -1, 1)),
        None, 0, None, lidcol=lidcol, lidmat=lmat, iris=iris, glow=glow, lidfill=lfill, lidfill_mat=lfmat)


def brows(P, sp, R, e, fx, ex, ey, ew, S, lod):
    bc = sp.get("brows") or sp["hair"]["c"]
    col_b = bc[1] if sp["hair"]["style"] != "bald" or not sp.get("brows") else bc[0]
    if sp.get("brows"):
        col_b = bc[0]
    for sg in (-1, 1):
        raise1 = e.get("brow1", 0.0) if sg > 0 else 0.0
        bi = e["bin"] + raise1
        bo = e["bout"] + raise1
        fr = e["frown"]
        base = [(0.05 - 0.014 * fr, -0.086 + 0.028 * fr - bi), (0.105, -0.127 - 0.4 * bi - 0.3 * bo + 0.01 * fr),
                (0.185, -0.142 - 0.6 * bo - 0.2 * bi), (0.265, -0.118 - bo), (0.315, -0.075 - 0.8 * bo)]
        lift = 0.035 if lod >= 1 else 0.0
        pts = [(fx(sg * x * (sp["face_w"] / 0.35)), y + ey - 0.02 - lift) for x, y in base]
        th0, th1 = (0.038, 0.02) if lod == 0 else (0.045, 0.03)
        top, bot = [], []
        n = len(pts)
        for i, p in enumerate(pts):
            a = pts[min(i + 1, n - 1)]
            b = pts[max(i - 1, 0)]
            dx, dy = a[0] - b[0], a[1] - b[1]
            d = math.hypot(dx, dy) + 1e-9
            nx, ny = -dy / d, dx / d
            if ny > 0:
                nx, ny = -nx, -ny
            th = (th0 + (th1 - th0) * i / (n - 1)) * 0.5
            top.append((p[0] + nx * th, p[1] + ny * th))
            bot.append((p[0] - nx * th, p[1] - ny * th))
        w_min = P.width(0.0, "fine")
        if w_min > th0:
            P.line(pts, col_b, th0, "fine", FINE)
        else:
            P.poly(top + list(reversed(bot)), fill=col_b, mat=FINE, smooth=True)


def nose_lines(P, sp, S, fx, nose, lod):
    n = nose
    c = S["contour"]
    # the long Byzantine nose line from the brow down the shadow side into the nostril
    pts = [(fx(0.052), -0.075), (fx(0.056), 0.06 * n), (fx(0.062), 0.18 * n), (fx(0.082), 0.236 * n),
           (fx(0.088), 0.262 * n), (fx(0.066), 0.284 * n), (fx(0.028), 0.282 * n)]
    if lod >= 2:
        P.line([pts[0], pts[2], pts[4], pts[-1]], c, 0.04, "fine", FINE)
        return
    P.line(pts, c, 0.028, "fine", FINE)
    P.line([(fx(-0.026), 0.284 * n), (fx(-0.06), 0.28 * n), (fx(-0.076), 0.258 * n)], c, 0.024, "fine2", FINE)
    if lod == 0:
        P.ellipse(fx(0.0), 0.262 * n, 0.03, 0.016, fill=S["flesh_l"], mat=FINE)


def draw_mouth(P, sp, S, e, mouth, fx, my, lod):
    mw = 0.074 * sp["mouth_w"]
    sm = e["smile"]
    sk = e.get("smirk", 0.0)
    op = clamp(max(mouth, 0.0) * 0.95 + e["open"], 0.0, 1.3) * 0.085
    cl = my - 0.02 * sm - 0.02 * sk
    cr = my - 0.02 * sm + 0.004 * sk
    L = (fx(-mw), cl)
    Rr = (fx(mw), cr)
    if lod >= 2:
        if op > 0.02:
            P.ellipse(fx(0.0), my + op * 0.4, mw * 0.7, op * 0.55 + 0.01, fill="mouth_in", mat=FINE)
        else:
            P.line([L, (fx(0.0), my + 0.006), Rr], S["lip_u"], 0.03, "fine2", FINE)
        return
    if op < 0.008:
        P.poly([L, (fx(-0.036), my - 0.018), (fx(0.0), my - 0.009), (fx(0.036), my - 0.018), Rr,
                (fx(0.032), my + 0.004), (fx(0.0), my + 0.006), (fx(-0.032), my + 0.004)], fill=S["lip_u"], mat=FINE,
               smooth=True)
        P.poly([(fx(-0.056), my + 0.004), (fx(0.0), my + 0.006), (fx(0.056), my + 0.004), (fx(0.036), my + 0.034),
                (fx(0.0), my + 0.042), (fx(-0.036), my + 0.034)], fill=S["lip_l"], mat=FINE, smooth=True)
        P.line([L, (fx(-0.03), my + 0.002 - 0.004 * sm), (fx(0.0), my + 0.006), (fx(0.03), my + 0.002 - 0.004 * sm), Rr],
               S["contour"], 0.016, "fine2", FINE)
        return
    # open mouth: dark interior, teeth, lips around
    bot = my + op
    inner = [L, (fx(-0.034), my - 0.008), (fx(0.0), my - 0.004), (fx(0.034), my - 0.008), Rr,
             (fx(0.046), bot - 0.3 * op), (fx(0.0), bot), (fx(-0.046), bot - 0.3 * op)]
    P.poly(inner, fill="mouth_in", mat=FINE, smooth=True)
    if op > 0.035 and lod == 0:
        P.clip(inner)
        P.poly([(fx(-0.06), my - 0.02), (fx(0.06), my - 0.02), (fx(0.06), my + 0.014), (fx(-0.06), my + 0.014)],
               fill="tooth", mat=FINE, smooth=False)
        P.unclip()
    P.line([L, (fx(-0.034), my - 0.012), (fx(0.0), my - 0.008), (fx(0.034), my - 0.012), Rr], S["lip_u"], 0.022,
           "fine2", FINE)
    P.line([(fx(-0.05), bot - 0.25 * op + 0.008), (fx(0.0), bot + 0.012), (fx(0.05), bot - 0.25 * op + 0.008)],
           S["lip_l"], 0.026, "fine2", FINE)


# ================================================================ hair
def _lock(P, base, tip, w, fill, line, lw, mat=FINE, bend=0.0):
    """a tapered lock of hair from base to tip (width w at the base)"""
    dx, dy = tip[0] - base[0], tip[1] - base[1]
    d = math.hypot(dx, dy) + 1e-9
    nx, ny = -dy / d, dx / d
    mid = (base[0] + dx * 0.55 + nx * bend * d, base[1] + dy * 0.55 + ny * bend * d)
    pts = [(base[0] + nx * w / 2, base[1] + ny * w / 2), (mid[0] + nx * w * 0.3, mid[1] + ny * w * 0.3), tip,
           (mid[0] - nx * w * 0.3, mid[1] - ny * w * 0.3), (base[0] - nx * w / 2, base[1] - ny * w / 2)]
    P.poly(pts, fill=fill, line=line, lw=lw, cls="fine2", mat=mat, smooth=True, tension=0.4)


def hair_back(P, sp, R, o):
    st = sp["hair"]["style"]
    c, cd, cl = sp["hair"]["c"]
    if sp.get("head") in ("veil",):
        return
    if st == "lutie":
        P.poly([(-0.43, 0.36), (-0.475, 0.12), (-0.485, -0.12), (-0.43, -0.35), (-0.3, -0.51), (-0.1, -0.58),
                (0.1, -0.585), (0.3, -0.525), (0.435, -0.37), (0.49, -0.12), (0.48, 0.12), (0.45, 0.36), (0.37, 0.43),
                (0.33, 0.3), (0.25, 0.36), (-0.24, 0.36), (-0.32, 0.3), (-0.37, 0.43)], fill=c, line=cd, lw=0.03,
               cls="fine2", mat=FINE, smooth=True)
        if P.lod == 0:
            for sg in (-1, 1):
                P.line([(sg * 0.43, -0.02), (sg * 0.45, 0.18), (sg * 0.4, 0.34)], cl, 0.022, "fine2", FINE)
    elif st in ("long", "long_f"):
        drop = 0.95 if st == "long" else 1.2
        P.poly([(-0.44, drop), (-0.5, 0.35), (-0.49, -0.15), (-0.37, -0.46), (0.0, -0.575), (0.37, -0.46), (0.49, -0.15),
                (0.5, 0.35), (0.44, drop), (0.3, drop + 0.06), (0.26, 0.45), (-0.26, 0.45), (-0.3, drop + 0.06)],
               fill=c, line=cd, lw=0.03, cls="fine2", mat=FINE, smooth=True)
        if P.lod == 0:
            for sg in (-1, 1):
                for k in range(2):
                    xx = sg * (0.38 + 0.05 * k)
                    P.line([(xx, 0.1), (xx + sg * 0.03, 0.45), (xx - sg * 0.01, drop - 0.05)], cl, 0.02, "fine2", FINE)
    elif st == "bun":
        P.circle(0.0, -0.55, 0.16, fill=c, line=cd, lw=0.03, cls="fine2", mat=FINE)
    elif st == "veil":
        vc = sp.get("veil_c", "lapis")
        P.poly([(-0.56, 1.25), (-0.58, 0.4), (-0.52, -0.22), (-0.37, -0.52), (0.0, -0.64), (0.37, -0.52), (0.52, -0.22),
                (0.58, 0.4), (0.56, 1.25)], fill=vc, line=vc + "_k" if vc + "_k" in IP else "ink", lw=0.03,
               cls="contour", smooth=True)
    elif st == "wild":
        # a storm cloud of curls around the head, a few flyaway locks
        sd = sp["seed"]
        blobs = []
        for i in range(13):
            a = math.pi * (0.92 + 1.16 * i / 12)
            rr = 0.5 + 0.06 * _hs(sd, i)
            br = 0.15 + 0.05 * _hs(sd, i + 30)
            wob = 0.02 * math.sin(R.t * 4.0 + i)
            blobs.append((rr * math.cos(a) * 1.08, -0.08 + rr * math.sin(a) + wob, br))
        for x, y, r in blobs:
            P.circle(x, y, r + 0.02, fill=cd, mat=FINE)
        for x, y, r in blobs:
            P.circle(x, y, r, fill=c, mat=FINE)
        if P.lod < 2:
            for i, (x, y, r) in enumerate(blobs):
                if i % 2 == 0:
                    P.line([(x - r * 0.5, y + r * 0.1), (x, y - r * 0.4), (x + r * 0.5, y + r * 0.1)], cl, 0.025, "fine2", FINE)
            for i in range(3):
                a = math.pi * (1.2 + 0.3 * i)
                x0, y0 = 0.62 * math.cos(a), -0.08 + 0.6 * math.sin(a)
                _lock(P, (x0, y0), (x0 * 1.35, y0 * 1.25 - 0.05), 0.07, c, cd, 0.02, bend=0.3)


def hair_front(P, sp, R, o):
    st = sp["hair"]["style"]
    c, cd, cl = sp["hair"]["c"]
    lod = P.lod
    if sp.get("head") in ("veil", "helmet"):
        if sp.get("head") == "helmet":
            # sideburns under the helmet rim
            for sg in (-1, 1):
                P.poly([(sg * 0.33, -0.2), (sg * 0.37, -0.02), (sg * 0.35, 0.12), (sg * 0.31, 0.05), (sg * 0.3, -0.12)],
                       fill=c, mat=FINE, smooth=True)
        return
    if st == "lutie":
        top = [(0.44, 0.08), (0.475, -0.12), (0.435, -0.36), (0.3, -0.52), (0.1, -0.585), (-0.1, -0.58), (-0.3, -0.51),
               (-0.43, -0.35), (-0.475, -0.12), (-0.44, 0.08)]
        hl = [(-0.37, 0.1), (-0.345, -0.08), (-0.3, -0.2), (-0.22, -0.19), (-0.18, -0.29), (-0.1, -0.19),
              (-0.03, -0.3), (0.05, -0.18), (0.11, -0.3), (0.19, -0.21), (0.25, -0.29), (0.31, -0.19), (0.345, -0.07),
              (0.37, 0.1)]
        P.poly(top + hl, fill=c, line=cd, lw=0.03, cls="fine2", mat=FINE, smooth=True, tension=0.35)
        # the messy tuft (cowlick) + a forehead curl
        for base, tip, bend, wl in (((-0.06, -0.55), (0.1, -0.76), -0.3, 0.12), ((0.06, -0.56), (0.24, -0.7), -0.28, 0.1),
                                    ((-0.14, -0.53), (-0.08, -0.7), -0.35, 0.09), ((0.0, -0.3), (0.06, -0.13), 0.25, 0.1)):
            _lock(P, base, tip, wl, c, cd, 0.028, bend=bend)
        if lod < 2:
            strands = ([(0.0, -0.55), (0.2, -0.47), (0.35, -0.3), (0.41, -0.08)],
                       [(-0.02, -0.55), (-0.22, -0.47), (-0.36, -0.28), (-0.42, -0.05)],
                       [(0.07, -0.56), (0.15, -0.42), (0.19, -0.3)], [(-0.07, -0.56), (-0.14, -0.42), (-0.19, -0.3)])
            for pts in (strands if lod == 0 else strands[:2]):
                P.line(pts, cl, 0.022, "fine2", FINE)
    elif st in ("short", "cap"):
        n = 11
        top = [(0.35, -0.02), (0.395, -0.2), (0.36, -0.38), (0.24, -0.5), (0.0, -0.548), (-0.24, -0.5), (-0.36, -0.38),
               (-0.395, -0.2), (-0.35, -0.02)]
        hl = []
        for i in range(n + 1):
            a = i / n
            x = -0.34 + 0.68 * a
            y = -0.05 - 0.24 * math.sin(math.pi * a) ** 0.6
            if st == "short" and 0 < i < n:
                y += 0.035 * (i % 2)
            hl.append((x, y))
        P.poly(top + hl, fill=c, line=cd, lw=0.03, cls="fine2", mat=FINE, smooth=True, tension=0.35)
        if lod == 0 and st == "short":
            for i in range(7):
                a = math.pi * (1.12 + 0.76 * i / 6)
                P.circle(0.27 * math.cos(a), -0.12 + 0.3 * math.sin(a), 0.05, line=cl, lw=0.018, cls="fine2", mat=FINE)
    elif st in ("long", "long_f"):
        top = [(0.4, 0.22), (0.44, -0.1), (0.38, -0.36), (0.2, -0.515), (0.0, -0.548), (-0.2, -0.515), (-0.38, -0.36),
               (-0.44, -0.1), (-0.4, 0.22)]
        hl = [(-0.34, 0.12), (-0.33, -0.12), (-0.25, -0.26), (-0.11, -0.31), (0.0, -0.27), (0.11, -0.31), (0.25, -0.26),
              (0.33, -0.12), (0.34, 0.12)]
        P.poly(top + hl, fill=c, line=cd, lw=0.03, cls="fine2", mat=FINE, smooth=True, tension=0.4)
        if lod == 0:
            for sg in (-1, 1):
                P.line([(sg * 0.02, -0.52), (sg * 0.2, -0.44), (sg * 0.33, -0.25), (sg * 0.37, 0.05)], cl, 0.022, "fine2", FINE)
    elif st == "bun":
        top = [(0.35, 0.0), (0.39, -0.2), (0.35, -0.39), (0.2, -0.51), (0.0, -0.545), (-0.2, -0.51), (-0.35, -0.39),
               (-0.39, -0.2), (-0.35, 0.0)]
        hl = [(-0.33, -0.04), (-0.31, -0.18), (-0.22, -0.28), (-0.08, -0.31), (0.0, -0.28), (0.08, -0.31), (0.22, -0.28),
              (0.31, -0.18), (0.33, -0.04)]
        P.poly(top + hl, fill=c, line=cd, lw=0.03, cls="fine2", mat=FINE, smooth=True, tension=0.4)
    elif st == "veil":
        vc = sp.get("veil_c", "lapis")
        # a strip of hair under the veil edge, then the veil framing the face
        P.poly([(-0.33, -0.1), (-0.27, -0.27), (0.0, -0.33), (0.27, -0.27), (0.33, -0.1), (0.3, -0.18), (0.0, -0.27),
                (-0.3, -0.18)], fill=c, mat=FINE, smooth=True)
        outer = [(-0.5, 0.5), (-0.52, 0.1), (-0.48, -0.28), (-0.35, -0.53), (0.0, -0.63), (0.35, -0.53), (0.48, -0.28),
                 (0.52, 0.1), (0.5, 0.5)]
        inner = [(0.36, 0.5), (0.37, 0.12), (0.33, -0.12), (0.24, -0.26), (0.0, -0.31), (-0.24, -0.26), (-0.33, -0.12),
                 (-0.37, 0.12), (-0.36, 0.5)]
        P.poly(outer + inner, fill=vc, line=vc + "_k" if vc + "_k" in IP else "ink", lw=0.03, cls="contour", smooth=True)
        if lod == 0:
            P.line(inner[1:-1], vc + "_l" if vc + "_l" in IP else "white", 0.03, "light")
    elif st == "bald":
        for sg in (-1, 1):
            P.poly([(sg * 0.33, -0.2), (sg * 0.39, -0.14), (sg * 0.4, 0.02), (sg * 0.36, 0.08), (sg * 0.33, -0.02)],
                   fill=c, line=cd, lw=0.02, cls="fine2", mat=FINE, smooth=True)
    elif st == "wild":
        hl = [(-0.36, 0.1), (-0.35, -0.12), (-0.28, -0.25), (-0.18, -0.2), (-0.1, -0.3), (0.0, -0.22), (0.1, -0.31),
              (0.19, -0.21), (0.27, -0.27), (0.33, -0.15), (0.36, 0.1)]
        top = [(0.46, 0.08), (0.49, -0.2), (0.4, -0.44), (0.0, -0.56), (-0.4, -0.44), (-0.49, -0.2), (-0.46, 0.08)]
        P.poly(hl + top, fill=c, line=cd, lw=0.03, cls="fine2", mat=FINE, smooth=True, tension=0.3)


def draw_beard(P, sp, R, o):
    b = sp.get("beard")
    if not b or not b.get("style"):
        return
    st = b["style"]
    c, cd, cl = b["c"]
    S = sp["skin"]
    lod = P.lod
    mouth = clamp(o.get("mouth", 0.0) + o["expr"]["open"], 0.0, 1.3)
    op = mouth * 0.085
    fw = sp["face_w"] / 0.35
    if st == "stubble":
        tone = mixc(S["flesh_m"], c, 0.35)
        P.poly([(-0.34 * fw, 0.1), (-0.3 * fw, 0.34), (-0.18, 0.47), (0.0, 0.51), (0.18, 0.47), (0.3 * fw, 0.34),
                (0.34 * fw, 0.1), (0.26 * fw, 0.26), (0.1, 0.33), (0.0, 0.31), (-0.1, 0.33), (-0.26 * fw, 0.26)],
               fill=tone, mat=FINE, smooth=True)
        return
    length = {"short": 0.58, "pointed": 0.76, "full": 0.98, "long": 1.22, "patriarch": 1.5, "wild": 1.15}.get(st, 0.6)
    wide = {"short": 0.345, "pointed": 0.35, "full": 0.37, "long": 0.37, "patriarch": 0.385, "wild": 0.4}.get(st, 0.35) * fw
    tip = 0.1 if st in ("pointed",) else 0.16
    outer = [(-wide * 0.95, 0.02), (-wide * 1.02, 0.26), (-wide * 0.98, 0.5), (-wide * 0.82, 0.5 + (length - 0.5) * 0.45),
             (-wide * 0.55, 0.5 + (length - 0.5) * 0.8), (-tip * wide, length - 0.02), (0.0, length + op * 0.5),
             (tip * wide, length - 0.02), (wide * 0.55, 0.5 + (length - 0.5) * 0.8), (wide * 0.82, 0.5 + (length - 0.5) * 0.45),
             (wide * 0.98, 0.5), (wide * 1.02, 0.26), (wide * 0.95, 0.02)]
    if length < 0.62:
        outer = [(-wide * 0.97, 0.04), (-wide * 0.95, 0.28), (-wide * 0.7, 0.46), (-0.16, 0.555), (0.0, 0.585 + op * 0.5),
                 (0.16, 0.555), (wide * 0.7, 0.46), (wide * 0.95, 0.28), (wide * 0.97, 0.04)]
    my = 0.352 + (sp["nose"] - 1.0) * 0.06
    inner = [(0.29 * fw, 0.1), (0.25 * fw, 0.27), (0.11, my + 0.012), (0.055, my + 0.05 + op), (0.0, my + 0.062 + op),
             (-0.055, my + 0.05 + op), (-0.11, my + 0.012), (-0.25 * fw, 0.27), (-0.29 * fw, 0.1)]
    shape = outer + inner
    P.poly(shape, fill=c, mat=FINE, smooth=True, tension=0.45)
    if lod < 2:
        P.clip(shape)
        P.push(-0.05, 0.0)
        P.poly(shape, line=cd if st != "patriarch" else "white_s", lw=0.16, cls="thin", mat=FINE, smooth=True, tension=0.45)
        P.pop()
        strand = cl if st != "patriarch" else "beard_d"
        if st in ("patriarch", "long", "full", "wild") and lod == 0:
            n = 7 if st == "patriarch" else 5
            for i in range(n):
                a = (i + 0.5) / n * 2 - 1
                x0 = a * wide * 0.85
                pts = [(x0, 0.3 + 0.1 * (1 - abs(a))), (x0 * 0.92 + 0.03 * math.sin(i * 1.7), 0.6),
                       (x0 * 0.7 - 0.03 * math.sin(i), 0.5 + (length - 0.5) * 0.7), (x0 * 0.35, length - 0.08)]
                P.line(pts, strand, 0.022, "fine2", FINE)
        elif lod == 0:
            for i in range(4):
                a = (i + 0.5) / 4 * 2 - 1
                P.line([(a * wide * 0.8, 0.3), (a * wide * 0.6, 0.5)], strand, 0.02, "fine2", FINE)
        P.unclip()
    P.poly(shape, line=cd if st != "patriarch" else "beard_d", lw=0.028, cls="fine2", mat=FINE, smooth=True, tension=0.45)
    # moustache
    for sg in (-1, 1):
        m = [(sg * 0.012, 0.302), (sg * 0.075, 0.298), (sg * 0.14, 0.322), (sg * 0.19, 0.372),
             (sg * (0.2 if st == "patriarch" else 0.16), 0.45 if st == "patriarch" else 0.4), (sg * 0.15, 0.405),
             (sg * 0.1, 0.36), (sg * 0.03, 0.34)]
        P.poly(m, fill=c, line=cd if st != "patriarch" else "beard_d", lw=0.022, cls="fine2", mat=FINE, smooth=True)


# ================================================================ headgear
def headgear(P, sp, R, o, stage):
    hd = sp.get("head")
    lod = P.lod
    if hd == "veil":
        oc = sp.get("over", "ochre")
        if stage == "back":
            back = [(-0.66, 1.62), (-0.64, 0.75), (-0.58, 0.12), (-0.54, -0.3), (-0.39, -0.57), (-0.14, -0.67),
                    (0.14, -0.67), (0.39, -0.57), (0.54, -0.3), (0.58, 0.12), (0.64, 0.75), (0.66, 1.62)]
            P.poly(back, fill=oc, line=oc + "_k", lw=0.035, cls="contour", smooth=True)
            if lod < 2:
                P.clip(back)
                for sg in (-1, 1):
                    P.poly([(sg * 0.45, 0.2), (sg * 0.7, 0.2), (sg * 0.72, 1.7), (sg * 0.5, 1.7)], fill=oc + "_d")
                P.unclip()
                for sg in (-1, 1):
                    P.line([(sg * 0.5, 0.35), (sg * 0.56, 0.9), (sg * 0.58, 1.5)], oc + "_k", 0.03, "fold")
        else:
            outer = [(-0.49, 0.62), (-0.52, 0.1), (-0.48, -0.3), (-0.35, -0.55), (0.0, -0.64), (0.35, -0.55), (0.48, -0.3),
                     (0.52, 0.1), (0.49, 0.62)]
            inner = [(0.345, 0.55), (0.36, 0.1), (0.335, -0.16), (0.25, -0.3), (0.1, -0.37), (0.0, -0.38), (-0.1, -0.37),
                     (-0.25, -0.3), (-0.335, -0.16), (-0.36, 0.1), (-0.345, 0.55)]
            P.poly(outer + inner, fill=oc, line=oc + "_k", lw=0.035, cls="contour", smooth=True)
            if lod < 2:
                P.clip(outer + inner)
                P.poly([(0.2, -0.7), (0.6, -0.7), (0.6, 0.7), (0.42, 0.7), (0.42, -0.2)], fill=oc + "_d")
                P.unclip()
                P.line(inner[1:-1], "gold", 0.04, "gold", "gold")
                P.line([(-0.2, -0.6), (-0.4, -0.42), (-0.47, -0.1)], oc + "_l", 0.03, "light")
                P.line([(0.12, -0.62), (0.3, -0.55), (0.42, -0.36)], oc + "_k", 0.03, "fold")
    elif hd == "helmet" and stage == "front":
        _helmet(P, sp, R, o)
    elif hd == "tiny_crown" and stage == "front":
        c0 = sp.get("crown_c", "gold")
        y0 = -0.5 if sp["hair"]["style"] not in ("veil",) else -0.6
        w = 0.19
        pts = [(-w, y0 + 0.02), (-w * 1.05, y0 - 0.1), (-w * 0.6, y0 - 0.05), (-w * 0.3, y0 - 0.16), (0.0, y0 - 0.07),
               (w * 0.3, y0 - 0.16), (w * 0.6, y0 - 0.05), (w * 1.05, y0 - 0.1), (w, y0 + 0.02)]
        P.poly(pts, fill=c0, line="gold_d", lw=0.025, cls="fine2", mat="fine gold", smooth=False)
        if lod < 2:
            P.dots([(-w * 0.55, y0 - 0.015), (w * 0.55, y0 - 0.015)], 0.028, "ruby", mat=FINE)
            P.dots([(0.0, y0 - 0.02)], 0.03, "emerald", mat=FINE)
            P.dots([(-w * 1.05, y0 - 0.11), (-w * 0.3, y0 - 0.17), (w * 0.3, y0 - 0.17), (w * 1.05, y0 - 0.11)], 0.024,
                   "pearl", mat="fine pearl")
    elif hd == "tiara":
        if stage == "back":
            for sg in (-1, 1):
                P.poly([(sg * 0.26, -0.1), (sg * 0.36, -0.1), (sg * 0.4, 0.95), (sg * 0.3, 0.95)], fill="red",
                       line="red_k", lw=0.025, cls="contour")
        else:
            prog = o.get("_tiara_on", 1.0)
            if prog >= 0.999:
                tiara(P, 0.0, 0.0, 1.0, lod)


def tiara(P, x, y, k, lod):
    """papal triregnum (no cross: a plain gold knob), base centre at (x, y - 0.27k) in head units"""
    P.push(x, y, 0.0, k)
    pts = [(-0.35, -0.27), (-0.375, -0.5), (-0.33, -0.76), (-0.2, -0.96), (0.0, -1.05), (0.2, -0.96), (0.33, -0.76),
           (0.375, -0.5), (0.35, -0.27)]
    P.poly(pts, fill="white", line="white_k", lw=0.03, cls="contour", smooth=True)
    if lod < 2:
        P.clip(pts)
        P.poly([(0.12, -1.1), (0.5, -1.1), (0.5, 0.0), (0.2, 0.0)], fill="white_d")
        P.unclip()
    for yy, ww in ((-0.33, 0.36), (-0.58, 0.36), (-0.83, 0.28)):
        band = [(-ww, yy + 0.045), (-ww, yy - 0.045), (ww, yy - 0.045), (ww, yy + 0.045)]
        P.poly(band, fill="gold", mat="gold", smooth=False)
        if lod < 2:
            P.dots([(-ww * 0.6 + ww * 0.4 * i, yy - 0.075) for i in range(4)], 0.035, "gold", mat="gold")
            P.dots([(-ww * 0.5, yy), (ww * 0.5, yy)], 0.022, "ruby", mat=FINE)
            P.dots([(0.0, yy)], 0.024, "emerald", mat=FINE)
    P.circle(0.0, -1.1, 0.06, fill="gold", line="gold_d", lw=0.02, cls="fine2", mat="gold")
    P.pop()


def _helmet(P, sp, R, o):
    """modern high-cut helmet (matte drab), side rails, and a quad-tube NVG housing on a flip mount"""
    lod = P.lod
    g = clamp(float(o.get("goggles", sp.get("goggles", 0.0))), 0.0, 1.0)
    shell = [(-0.46, -0.02), (-0.5, -0.24), (-0.45, -0.46), (-0.28, -0.62), (0.0, -0.675), (0.28, -0.62), (0.45, -0.46),
             (0.5, -0.24), (0.46, -0.02), (0.39, -0.03), (0.37, -0.17), (0.31, -0.25), (0.0, -0.285), (-0.31, -0.25),
             (-0.37, -0.17), (-0.39, -0.03)]
    P.poly(shell, fill="helmet", line="helmet_k", lw=0.035, cls="contour", smooth=True, tension=0.35)
    if lod < 2:
        P.clip(shell)
        P.poly([(0.16, -0.72), (0.62, -0.72), (0.62, 0.1), (0.36, 0.1), (0.28, -0.3)], fill="helmet_d")
        P.unclip()
        P.line([(-0.37, -0.44), (-0.2, -0.58), (0.0, -0.62)], "helmet_l", 0.035, "light")
        P.line([(-0.44, -0.21), (-0.3, -0.27), (0.0, -0.31), (0.3, -0.27), (0.44, -0.21)], "helmet_k", 0.03, "fold")
    for sg in (-1, 1):
        P.poly([(sg * 0.36, -0.43), (sg * 0.46, -0.37), (sg * 0.47, -0.22), (sg * 0.38, -0.25)], fill="steel_d",
               line="steel_k", lw=0.02, cls="fine2", mat="silver")
    if sp.get("headset"):
        P.ellipse(0.43, 0.06, 0.1, 0.13, fill="black", line="black_k", lw=0.03, cls="contour")
        P.line([(0.46, 0.14), (0.4, 0.3), (0.26, 0.38), (0.12, 0.385)], "black", 0.035, "fine2")
        P.ellipse(0.1, 0.385, 0.035, 0.028, fill="black_l", line="black_k", lw=0.015, cls="fine2")
    # mount shroud on the brow of the helmet
    P.poly([(-0.09, -0.47), (0.09, -0.47), (0.075, -0.33), (-0.075, -0.33)], fill="black", line="black_k", lw=0.02,
           cls="fine2")
    hx, hy, hw, hh, ryf = _nvg_housing(g)
    if g > 0.35:
        # the flip arm: a trapezoid widening from the shroud into the housing (never a stalk + bar = a cross)
        P.poly([(-0.09, -0.36), (0.09, -0.36), (0.3, hy - hh * 0.45), (-0.3, hy - hh * 0.45)], fill="black",
               line="black_k", lw=0.02, cls="fine2", smooth=False)
    box = [(hx - hw / 2, hy - hh / 2), (hx + hw / 2, hy - hh / 2), (hx + hw / 2, hy + hh / 2), (hx - hw / 2, hy + hh / 2)]
    P.poly(box, fill="black", line="black_k", lw=0.025, cls="fine2", smooth=False)
    for (x, y, r) in nvg_lenses(g):
        P.ellipse(x, y, r * 1.2, r * 1.2 * ryf, fill="black_l", line="black_k", lw=0.02, cls="fine2")
        P.ellipse(x, y, r, r * ryf, fill="glass_d", mat="glow")
        if lod == 0:
            P.ellipse(x - r * 0.3, y - r * 0.3 * ryf, r * 0.3, r * 0.22 * ryf, fill="#1F5A3A", mat="glow")


def _nvg_housing(g):
    """(cx, cy, w, h, lens foreshortening) of the NVG housing in head units for flip position g (0 up .. 1 down)"""
    g = clamp(g, 0.0, 1.0)
    phi = (1.0 - g) * 0.72 * math.pi
    hy = -0.36 + 0.39 * math.cos(phi)
    ryf = max(0.28, abs(math.cos(phi)))
    return 0.0, hy, 0.86, 0.2 * (0.45 + 0.55 * ryf), ryf


def nvg_lenses(g):
    """4 lens centres (x, y, r) in head units for goggles position g (0 up .. 1 down)"""
    hx, hy, hw, hh, ryf = _nvg_housing(g)
    return [(x, hy, r) for x, r in ((-0.3, 0.07), (-0.1, 0.082), (0.1, 0.082), (0.3, 0.07))]


# ================================================================ hands
# HAND units: wrist at (0,0), fingers toward +y, thumb side +x, seen from the palm ('palm' view) unless noted.
_HANDS = {}


def _hand_lib():
    if _HANDS:
        return _HANDS
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    def cap(pts, r):
        return LineString(pts).buffer(r, quad_segs=6)

    palm = Polygon([(-0.18, 0.0), (0.16, 0.0), (0.2, 0.2), (0.205, 0.46), (-0.205, 0.47), (-0.215, 0.22)]).buffer(0.035)

    def build(name, fingers, thumb, view, extra=None, lines=None, palm_poly=palm):
        parts = [palm_poly] + [cap(f, r) for f, r in fingers]
        if thumb:
            parts.append(cap(thumb[0], thumb[1]))
        if extra is not None:
            parts.append(extra)
        g = unary_union(parts).simplify(0.006)
        if g.geom_type != "Polygon":
            g = max(g.geoms, key=lambda q: q.area)
        ln = list(lines or [])
        tips = []
        for f, r in fingers + ([(tuple(thumb[0][-2:]), thumb[1])] if thumb else []):
            a, b_ = f[-2], f[-1]
            dx, dy = b_[0] - a[0], b_[1] - a[1]
            d = math.hypot(dx, dy) + 1e-9
            if d > 0.12:
                tips.append((b_[0] + dx / d * r, b_[1] + dy / d * r, dx / d, dy / d, r))
        # valleys between neighbouring extended fingers
        for (f1, r1), (f2, r2) in zip(fingers[:-1], fingers[1:]):
            a, b = f1[0], f2[0]
            ta, tb = f1[-1], f2[-1]
            la = math.hypot(ta[0] - a[0], ta[1] - a[1])
            lb = math.hypot(tb[0] - b[0], tb[1] - b[1])
            if min(la, lb) < 0.2:
                continue
            m0 = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.02)
            k = min(la, lb) * 0.55
            dx = (ta[0] + tb[0] - a[0] - b[0]) / 2
            dy = (ta[1] + tb[1] - a[1] - b[1]) / 2
            d = math.hypot(dx, dy) + 1e-9
            ln.append([m0, (m0[0] + dx / d * k, m0[1] + dy / d * k)])
        _HANDS[name] = dict(out=list(g.exterior.coords), lines=ln, view=view, tips=tips)

    F = lambda x0, y0, x1, y1, r: (((x0, y0), (x1, y1)), r)
    # open palm, fingers together-ish, thumb out (orans, recoil, declare)
    build("palm", [F(0.14, 0.45, 0.165, 0.88, 0.05), F(0.05, 0.47, 0.055, 0.97, 0.052), F(-0.045, 0.47, -0.055, 0.92, 0.05),
                   F(-0.135, 0.44, -0.165, 0.79, 0.045)], ([(0.15, 0.13), (0.29, 0.3), (0.34, 0.47)], 0.055), "palm",
          lines=[[(0.17, 0.12), (0.02, 0.2), (-0.1, 0.36)]])
    # Byzantine blessing: index + middle up, ring finger bent to meet the thumb, little finger up
    build("bless", [F(0.125, 0.45, 0.13, 0.92, 0.05), F(0.03, 0.47, 0.035, 0.97, 0.052), F(-0.14, 0.44, -0.18, 0.77, 0.045)],
          ([(0.15, 0.13), (0.1, 0.34), (0.0, 0.5)], 0.055), "palm",
          extra=cap([(-0.055, 0.46), (-0.05, 0.58), (0.0, 0.56)], 0.05),
          lines=[[(-0.04, 0.52), (0.0, 0.54)], [(-0.1, 0.47), (-0.12, 0.6)]])
    build("speak", [F(0.14, 0.45, 0.2, 0.84, 0.05), F(0.05, 0.47, 0.095, 0.93, 0.052), F(-0.045, 0.47, -0.02, 0.87, 0.05),
                    F(-0.135, 0.44, -0.13, 0.75, 0.045)], ([(0.15, 0.13), (0.31, 0.29), (0.39, 0.43)], 0.055), "palm")
    build("point_up", [F(0.12, 0.45, 0.13, 0.97, 0.052)], ([(0.15, 0.13), (0.1, 0.34), (-0.02, 0.44)], 0.055), "palm",
          extra=cap([(0.03, 0.5), (-0.13, 0.5)], 0.07),
          lines=[[(0.06, 0.44), (0.06, 0.56)], [(-0.04, 0.45), (-0.04, 0.56)], [(-0.12, 0.44), (-0.12, 0.54)]])
    build("point", [F(0.12, 0.45, 0.13, 0.97, 0.052)], ([(0.17, 0.16), (0.21, 0.36), (0.18, 0.48)], 0.055), "back",
          extra=cap([(0.03, 0.52), (-0.13, 0.5)], 0.075),
          lines=[[(0.06, 0.44), (0.06, 0.58)], [(-0.04, 0.45), (-0.04, 0.58)], [(-0.12, 0.44), (-0.12, 0.56)]])
    build("relax", [F(0.13, 0.45, 0.13, 0.84, 0.052), F(0.045, 0.47, 0.04, 0.92, 0.053), F(-0.045, 0.47, -0.05, 0.88, 0.052),
                    F(-0.13, 0.44, -0.14, 0.76, 0.047)], ([(0.16, 0.12), (0.24, 0.3), (0.24, 0.46)], 0.055), "back")
    build("back", [F(0.13, 0.45, 0.14, 0.88, 0.052), F(0.045, 0.47, 0.045, 0.96, 0.053), F(-0.045, 0.47, -0.05, 0.91, 0.052),
                   F(-0.13, 0.44, -0.15, 0.79, 0.047)], ([(0.16, 0.12), (0.26, 0.3), (0.28, 0.47)], 0.055), "back")
    build("hold", [F(0.14, 0.45, 0.16, 0.66, 0.055), F(0.05, 0.47, 0.055, 0.7, 0.056), F(-0.045, 0.47, -0.05, 0.68, 0.055),
                   F(-0.135, 0.44, -0.15, 0.62, 0.05)], ([(0.16, 0.12), (0.3, 0.3), (0.34, 0.46)], 0.058), "back",
          lines=[[(0.1, 0.5), (0.1, 0.62)], [(0.0, 0.5), (0.0, 0.64)], [(-0.09, 0.49), (-0.09, 0.6)]])
    build("fist", [], ([(0.2, 0.2), (0.12, 0.42), (-0.05, 0.46)], 0.06), "back",
          extra=Polygon([(-0.2, 0.08), (0.2, 0.08), (0.22, 0.56), (-0.21, 0.56)]).buffer(0.04),
          lines=[[(0.08, 0.5), (0.1, 0.6)], [(-0.02, 0.5), (-0.02, 0.6)], [(-0.12, 0.5), (-0.13, 0.59)]])
    build("cup", [F(0.0, 0.5, 0.0, 0.96, 0.06)], ([(0.1, 0.25), (0.2, 0.46), (0.2, 0.6)], 0.06), "palm",
          palm_poly=Polygon([(-0.11, 0.0), (0.1, 0.0), (0.13, 0.5), (0.06, 0.8), (-0.12, 0.8), (-0.13, 0.4)]).buffer(0.03),
          lines=[[(-0.04, 0.62), (-0.03, 0.9)], [(0.02, 0.6), (0.03, 0.88)]])
    build("pray", [F(0.04, 0.45, 0.02, 0.95, 0.055)], ([(0.08, 0.14), (0.13, 0.32), (0.1, 0.46)], 0.05), "palm",
          palm_poly=Polygon([(0.0, 0.0), (0.12, 0.02), (0.13, 0.45), (0.0, 0.5)]).buffer(0.02),
          lines=[[(0.06, 0.5), (0.05, 0.85)]])
    build("stylus", [F(0.13, 0.45, 0.1, 0.72, 0.05)], ([(0.17, 0.14), (0.24, 0.36), (0.14, 0.6)], 0.055), "back",
          extra=cap([(0.04, 0.5), (-0.13, 0.48)], 0.075),
          lines=[[(0.0, 0.46), (0.0, 0.58)], [(-0.09, 0.46), (-0.09, 0.57)]])
    build("pinch", [F(0.12, 0.45, 0.18, 0.78, 0.05)], ([(0.17, 0.14), (0.3, 0.38), (0.22, 0.72)], 0.055), "back",
          extra=cap([(0.03, 0.52), (-0.13, 0.5)], 0.075),
          lines=[[(-0.02, 0.46), (-0.02, 0.58)], [(-0.1, 0.46), (-0.1, 0.57)]])
    # grip: a fist around a pole; POLE frame: pole along y through (0,0), knuckles toward the viewer, wrist at -x
    grip = Polygon([(-0.22, -0.2), (0.18, -0.22), (0.22, -0.1), (0.22, 0.12), (0.18, 0.22), (-0.22, 0.21)]).buffer(0.05)
    build("grip", [], ([(-0.2, -0.22), (0.02, -0.27), (0.15, -0.22)], 0.06), "back", palm_poly=grip,
          lines=[[(0.0, -0.105), (0.25, -0.105)], [(0.0, 0.0), (0.26, 0.0)], [(0.0, 0.105), (0.25, 0.105)]])
    for alias, src in (("open", "cup"), ("cup_ear", "back"), ("veil", "fist")):
        _HANDS[alias] = _HANDS[src]
    return _HANDS


def _hand_len(sp, R):
    return R.HL * (0.74 if sp["sex"] == "m" else 0.7) * (1.08 if sp["kind"] == "jaguar" else 1.0)


def _hand_frame(R, side, shape):
    """(x, y, rot, flip) of the hand in local units"""
    wr = R.wr[side]
    sg = -1.0 if side == "r" else 1.0
    if shape == "grip" and R.pole is not None:
        dirx, diry = R.pole[1]
        rot = math.atan2(dirx, -diry)              # hand +y runs DOWN the pole (thumb on top)
        # +x must point away from the elbow (the wrist is at -x)
        cx, cy = math.cos(rot), math.sin(rot)
        el = R.el[side]
        ex, ey = el[0] - wr[0], el[1] - wr[1]
        flip = -1.0 if (ex * cx + ey * cy) > 0 else 1.0
        return wr[0], wr[1], rot, flip
    a = math.radians(R.hand_ang[side])
    dx, dy = sg * math.sin(a), -math.cos(a)
    rot = math.atan2(-dx, dy)
    view = _hand_lib()[shape]["view"] if shape in _hand_lib() else "back"
    flip = -1.0 if ((side == "l") != (view == "back")) else 1.0
    return wr[0], wr[1], rot, flip


def draw_hand(P, F, R, side, front=False):
    sp = F.sp
    shape = R.shape[side]
    lib = _hand_lib()
    H = lib.get(shape) or lib["relax"]
    S = sp["skin"]
    paw = sp["kind"] == "jaguar"
    fill, line = (S["flesh"], S["contour"]) if not paw else ("fur", "rosette")
    x, y, rot, flip = _hand_frame(R, side, shape)
    HN = _hand_len(sp, R)
    if shape == "veil":
        fam_ = sp.get("over") or sp["robe"]
        P.push(x, y, rot, HN * flip, HN)
        P.poly([(-0.26, -0.05), (0.26, -0.05), (0.32, 0.5), (0.0, 0.72), (-0.32, 0.5)], fill=fam_,
               line=fam_ + "_k" if fam_ + "_k" in IP else "ink", lw=0.05, cls="contour", smooth=True)
        P.line([(-0.15, 0.1), (-0.05, 0.5)], fam_ + "_d" if fam_ + "_d" in IP else "ink", 0.04, "fold")
        P.pop()
        return
    P.push(x, y, rot, HN * flip, HN)
    if shape == "grip":
        P.poly(H["out"], fill=fill, mat=FINE, smooth=False)
    else:
        P.poly(H["out"], fill=fill, mat=FINE, smooth=False)
    if P.lod < 2:
        if not paw:
            # a little shading on the palm heel / back of the hand
            P.clip(H["out"], smooth=False)
            P.push(0.06, 0.0)
            P.poly(H["out"], line=S["flesh_m"], lw=0.1, cls="thin", mat=FINE, smooth=False)
            P.pop()
            P.unclip()
        for ln in H["lines"]:
            P.line(ln, line, 0.028, "fine2", FINE, smooth=False)
        if paw:
            _paw_claws(P, H)
    P.poly(H["out"], line=line, lw=0.04, cls="fine", mat=FINE, smooth=False)
    P.pop()


def _paw_claws(P, H):
    """jaguar paws: dark claws at every extended digit, a pad on the palm side, rosettes on the back"""
    for x, y, dx, dy, r in H.get("tips", ()):
        nx, ny = -dy, dx
        P.poly([(x - dx * 0.03 + nx * r * 0.55, y - dy * 0.03 + ny * r * 0.55),
                (x + dx * 0.1, y + dy * 0.1),
                (x - dx * 0.03 - nx * r * 0.55, y - dy * 0.03 - ny * r * 0.55)], fill="rosette", mat=FINE)
    if H["view"] == "palm":
        P.ellipse(0.0, 0.3, 0.12, 0.09, fill="#5A3A34", mat=FINE)
    else:
        P.dots([(0.06, 0.22), (-0.08, 0.3), (0.0, 0.12)], 0.04, "rosette", mat=FINE)


# ================================================================ cloth helpers
def _c(f, suf):
    k = f + suf
    return k if k in IP else f


def cloth(P, pts, f, smooth=True, model=True, mat=None, lw=0.55, shade=1.0, light=1.0):
    """a flat cloth field: base colour, a tubular shadow band on the right / light band on the left, dark contour"""
    P.poly(pts, fill=f, mat=mat, smooth=smooth)
    if model and P.lod < 2:
        P.clip(pts, smooth)
        if shade:
            P.push(-1.6 * shade, 0.0)
            P.poly(pts, line=_c(f, "_d"), lw=3.2 * shade, cls="thin", mat=mat, smooth=smooth)
            P.pop()
        if light and P.lod == 0:
            P.push(1.3 * light, 0.0)
            P.poly(pts, line=_c(f, "_l"), lw=1.2 * light, cls="thin", mat=mat, smooth=smooth)
            P.pop()
        P.unclip()
    if lw:
        P.poly(pts, line=_c(f, "_k"), lw=lw, cls="contour", smooth=smooth)


def offset_line(pts, d):
    """polyline offset by d (left normal, y down)"""
    out = []
    n = len(pts)
    for i, p in enumerate(pts):
        a = pts[min(i + 1, n - 1)]
        b = pts[max(i - 1, 0)]
        dx, dy = a[0] - b[0], a[1] - b[1]
        l = math.hypot(dx, dy) + 1e-9
        out.append((p[0] - dy / l * d, p[1] + dx / l * d))
    return out


def band(P, pts, w, f, edge=None, edge_w=0.5, mat=None, edge_mat=None, lw_out=0.45, line_c=None):
    """a stole/border band of width w along a polyline, optional coloured edges (e.g. gold)"""
    L = offset_line(pts, w / 2)
    Rr = offset_line(pts, -w / 2)
    poly = L + list(reversed(Rr))
    P.poly(poly, fill=f, mat=mat, smooth=True)
    if edge:
        P.line(offset_line(pts, w / 2 - edge_w * 0.6), edge, edge_w, "gold" if edge == "gold" else "light",
               edge_mat or ("gold" if edge == "gold" else None))
        P.line(offset_line(pts, -w / 2 + edge_w * 0.6), edge, edge_w, "gold" if edge == "gold" else "light",
               edge_mat or ("gold" if edge == "gold" else None))
    if lw_out:
        P.poly(poly, line=line_c or _c(f, "_k"), lw=lw_out, cls="contour", smooth=True)
    return poly


def robe_outline(R, sp, hem_y=None, hem_w=None, sh_ext=1.5, top=True):
    """closed outline of a garment hanging from the shoulders (local units), following the stance"""
    U = R.extra["up"]
    SW, shy = R.SW, R.shoulder_y
    sq = R.extra.get("squash", 1.0)
    ww, pw = sp["waist_w"] * sq, sp["hip_w"] * sq
    hy = sp["hem_y"] if hem_y is None else hem_y
    hw = (sp["hem_w"] if hem_w is None else hem_w) * sq
    st = R.stance
    long_ = hy > -22
    SWr, SWl = SW * (1.0 - 0.06 * R.turn), SW * (1.0 - 0.2 * R.turn)
    profile_kneel = st == "kneel" and R.turn > 0.4

    def side(sg):
        SW = SWr if sg < 0 else SWl
        pts = []
        if top:
            pts += [U((sg * SW * 0.5, shy - 2.4)), U((sg * (SW + sh_ext * 0.55), shy - 0.8))]
        pts += [U((sg * (SW + sh_ext), shy + 2.6)), U((sg * (SW + sh_ext * 0.6), shy + 11.0)), U((sg * ww, -61.0))]
        if st == "sit":
            pts += [(sg * (pw + 0.6), R.hip_y - 3.0), (sg * (pw + 2.2), -29.0)]
            if long_:
                pts += [(sg * (pw + 1.4), -19.0), (sg * (hw + 0.2), -4.4)]
        elif profile_kneel:
            # kneeling seen three-quarter: thighs forward to the knee, the back of the robe down to the heels
            if sg > 0:
                pts += [(pw + 1.0, R.hip_y - 1.0), (pw + 7.5, R.hip_y + 4.0), (pw + 11.0, -6.0), (pw + 11.5, 0.0)]
            else:
                pts += [(-pw - 0.5, R.hip_y - 1.0), (-pw - 2.5, -9.0), (-pw - 4.0, 0.0)]
        elif st == "kneel":
            pts += [(sg * (pw + 1.2), R.hip_y - 1.0)]
            if long_:
                pts += [(sg * (hw + 2.0), -7.0), (sg * (hw + 3.2), 0.0)]
        else:
            pts += [(sg * pw, R.hip_y)]
            if long_:
                ext = 0.0
                if st == "march":
                    fr, fl = R.feet["r"], R.feet["l"]
                    lo = min(fr[0], fl[0]) - 3.2
                    hi = max(fr[0], fl[0]) + 3.2
                    ext = max(0.0, -hw - lo) if sg < 0 else max(0.0, hi - hw)
                pts += [(sg * (pw + 0.5 + ext * 0.35), -28.0), (sg * (hw + ext), hy)]
            else:
                pts += [(sg * (hw), hy)]
        return pts

    L = side(-1.0)
    Rt = side(1.0)
    by = L[-1][1]
    hem = [(L[-1][0] * 0.5, by + 0.7), (0.0, by + 1.0), (Rt[-1][0] * 0.5, by + 0.7)]
    return L + hem + list(reversed(Rt))


def lower_folds(P, R, sp, f, y0, y1, w0, w1, n=5, seed=0, chrys=False, light=True, x0=0.0, fields=True):
    """vertical tubular folds of a hanging skirt between y0 (top) and y1 (hem) - the Byzantine column of cloth"""
    if P.lod >= 2:
        return
    k, d, l = _c(f, "_k"), _c(f, "_d"), _c(f, "_l")
    sd = sp["seed"] * 7 + seed
    sq = R.extra.get("squash", 1.0)
    w0, w1 = w0 * sq, w1 * sq
    if R.stance == "march" and y1 > -22:
        # the leading leg presses the cloth: a long contour line hip -> knee -> foot with a highlight beside it
        fr, fl = R.feet["r"], R.feet["l"]
        lead = fr if fr[0] > fl[0] else fl
        hip = (lead[0] * 0.35 + 1.0, R.hip_y + 3.0)
        knee = (lead[0] * 0.7 + 2.0, -27.0)
        foot = (lead[0] + 1.2, y1 - 1.2)
        P.line([hip, knee, foot], k, 0.5, "fold")
        P.line([(hip[0] - 1.4, hip[1] + 3.0), (knee[0] - 1.4, knee[1]), (foot[0] - 1.6, foot[1] - 3.0)], l, 0.45, "light")
    for i in range(n):
        s = -0.8 + 1.6 * (i + 0.5) / n + 0.12 * (_hs(sd, i) - 0.5)
        v0 = 0.08 + 0.3 * _hs(sd, i + 20)
        ph = _hs(sd, i + 40) * 6.0
        pts = []
        for j in range(6):
            v = v0 + (1.0 - v0) * j / 5.0
            yy = y0 + (y1 - y0) * v
            ww = w0 + (w1 - w0) * v
            pts.append((x0 + s * ww + 0.35 * math.sin(v * 4.0 + ph), yy - (0.5 if j == 5 else 0.0)))
        if fields and i % 2 == 0:
            P.poly(pts + [(p[0] + 1.5, p[1]) for p in reversed(pts)], fill=d, smooth=True)
        P.line(pts, k, 0.5, "fold")
        if light:
            P.line([(p[0] - 1.1, p[1]) for p in pts[1:]], "gold" if chrys else l, 0.42 if not chrys else 0.4,
                   "gold" if chrys else "light", "gold" if chrys else None)


def chest_folds(P, R, f, n=2, depth=4.0, w=0.75, chrys=False, y_off=0.0):
    if P.lod >= 2:
        return
    U = R.extra["up"]
    SW, shy = R.SW, R.shoulder_y
    k, l = _c(f, "_k"), _c(f, "_l")
    for i in range(n):
        yy = shy + 3.0 + 3.2 * i + y_off
        pts = [U((-SW * w, yy)), U((-SW * w * 0.5, yy + depth * 0.75)), U((0.0, yy + depth)), U((SW * w * 0.5, yy + depth * 0.75)),
               U((SW * w, yy))]
        P.line(pts, k, 0.45, "fold")
        if P.lod == 0:
            P.line([(p[0], p[1] - 0.9) for p in pts[1:-1]], "gold" if chrys else l, 0.38, "gold" if chrys else "light",
                   "gold" if chrys else None)


def belt(P, R, sp, c, w=2.2, y=-61.0, ends=True, mat=None):
    U = R.extra["up"]
    ww = sp["waist_w"] + 0.5
    pts = [U((-ww, y - w / 2)), U((ww, y - w / 2)), U((ww, y + w / 2)), U((-ww, y + w / 2))]
    P.poly(pts, fill=c, line="ink", lw=0.4, cls="contour", mat=mat, smooth=False)
    if ends:
        a = U((1.5, y + w / 2))
        P.line([a, (a[0] + 0.4, a[1] + 7.0), (a[0] - 0.3, a[1] + 12.0)], c, w * 0.55, "fold", mat)
        P.line([(a[0] + 1.6, a[1]), (a[0] + 2.3, a[1] + 6.0)], c, w * 0.5, "fold", mat)


def draw_feet(P, sp, R, kind="shoe"):
    """feet below the hem; kind: shoe colour key, 'sandal', 'boots'"""
    S = sp["skin"]
    st = R.stance
    if st == "kneel" and R.turn < 0.4:
        return
    for side in ("r", "l"):
        fx, fy = R.feet[side]
        sg = -1.0 if side == "r" else 1.0
        if st == "march" or R.turn > 0.45:
            # 3/4: both feet point to +x
            pts = [(fx - 1.6, fy - 1.9), (fx + 1.2, fy - 2.0), (fx + 4.6, fy - 0.6), (fx + 4.4, fy + 0.2), (fx - 1.8, fy + 0.2)]
        else:
            ang = sg * 0.55
            c, s_ = math.cos(ang), math.sin(ang)
            base = [(-1.3, -1.9), (1.3, -1.9), (1.2, 2.6), (0.0, 3.5), (-1.2, 2.6)]
            pts = []
            for px, py in base:
                # foot seen from the front/above: length down-outward
                pts.append((fx + c * px - s_ * py * 0.8, fy - 0.2 + (s_ * px + c * py) * 0.55))
        if kind == "sandal":
            P.poly(pts, fill=S["flesh"], line=S["contour"], lw=0.35, cls="fine2", mat=FINE, smooth=True)
            if P.lod < 2:
                P.line([lerp2(pts[0], pts[1], 0.1), lerp2(pts[-1], pts[-2], 0.2)], "leather", 0.45, "fine2", FINE)
                P.line([lerp2(pts[1], pts[2], 0.3), lerp2(pts[-1], pts[-2], 0.7)], "leather", 0.45, "fine2", FINE)
        elif kind == "boots":
            P.poly(pts, fill="black", line="black_k", lw=0.4, cls="contour", smooth=True)
        else:
            c = kind if kind in IP else "leather"
            P.poly(pts, fill=c, line=_c(c, "_k") if _c(c, "_k") != c else "ink", lw=0.4, cls="contour", smooth=True)
            if P.lod == 0:
                P.line([lerp2(pts[0], pts[1], 0.3), lerp2(pts[1], pts[2], 0.5)], _c(c, "_l"), 0.3, "light")


def warrior_legs(P, sp, R, legc="leather_d", pads=True):
    """legs below a knee-length tunic: leggings, knee pads, laced boots"""
    hy = sp["hem_y"]
    for side in ("r", "l"):
        fx, fy = R.feet[side]
        kx, ky = R.knees[side]
        top = (kx * 0.95, hy - 3.0)
        pts = [(top[0] - 2.3, top[1]), (top[0] + 2.3, top[1]), (kx + 2.2, ky + 4.0), (fx + 1.5, fy - 7.0),
               (fx + 1.4, fy - 1.5), (fx - 1.4, fy - 1.5), (fx - 1.6, fy - 7.0), (kx - 2.3, ky + 4.0)]
        P.poly(pts, fill=legc, line="ink", lw=0.45, cls="contour", smooth=False)
        # boot shaft
        bt = [(kx + 2.05, ky + 12.0), (fx + 1.55, fy - 1.0), (fx - 1.55, fy - 1.0), (kx - 2.05, ky + 12.0)]
        P.poly(bt, fill="black", line="black_k", lw=0.4, cls="contour", smooth=False)
        if P.lod < 2:
            for j in range(3):
                yy = ky + 14.0 + j * 4.0
                P.line([(kx - 1.3 + (fx - kx) * (j / 4), yy), (kx + 1.3 + (fx - kx) * (j / 4), yy + 1.0)], "black_l", 0.35, "light")
        if pads:
            P.poly([(kx - 2.5, ky - 1.0), (kx + 2.5, ky - 1.0), (kx + 2.4, ky + 4.2), (kx - 2.4, ky + 4.2)], fill="black",
                   line="black_k", lw=0.4, cls="contour", smooth=True)
            if P.lod == 0:
                P.line([(kx - 1.4, ky + 0.2), (kx + 1.2, ky + 0.2)], "black_l", 0.35, "light")
    draw_feet(P, sp, R, "boots")


# ================================================================ sleeves / arms
def sleeve_poly(sh, el, wr, r0, r1, r2):
    def nrm(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy) + 1e-9
        return (-dy / d, dx / d)
    n0 = nrm(sh, el)
    n2 = nrm(el, wr)
    nx, ny = n0[0] + n2[0], n0[1] + n2[1]
    d = math.hypot(nx, ny)
    if d < 1e-6:
        nx, ny = n0
        k = 1.0
    else:
        nx, ny = nx / d, ny / d
        k = min(1.6, 1.0 / max(0.4, nx * n0[0] + ny * n0[1]))
    Lp = [(sh[0] + n0[0] * r0, sh[1] + n0[1] * r0), (el[0] + nx * r1 * k, el[1] + ny * r1 * k),
          (wr[0] + n2[0] * r2, wr[1] + n2[1] * r2)]
    Rp = [(sh[0] - n0[0] * r0, sh[1] - n0[1] * r0), (el[0] - nx * r1 * k, el[1] - ny * r1 * k),
          (wr[0] - n2[0] * r2, wr[1] - n2[1] * r2)]
    return Lp, Rp


def draw_sleeve(P, sp, R, side, f, cuff=None, radii=(3.1, 2.7, 2.3), bell=0.0, folds=True, upper=True, short=None):
    sh, el, wr = R.sh[side], R.el[side], R.wr[side]
    r0, r1, r2 = radii
    r2 = r2 + bell
    # stop the sleeve a little before the hand so the cuff sits at the wrist
    dx, dy = wr[0] - el[0], wr[1] - el[1]
    d = math.hypot(dx, dy) + 1e-9
    HN = _hand_len(sp, R)
    back = 0.12 * HN if R.shape[side] != "grip" else 0.3 * HN
    end = (wr[0] - dx / d * back, wr[1] - dy / d * back)
    if short is not None:
        end = lerp2(el, end, short)
    Lp, Rp = sleeve_poly(sh, el, end, r0, r1, r2)
    pts = Lp + list(reversed(Rp))
    k, dk, lt = _c(f, "_k"), _c(f, "_d"), _c(f, "_l")
    P.poly(pts, fill=f, smooth=False)
    if upper:
        P.circle(sh[0], sh[1], r0, fill=f)
    if folds and P.lod < 2:
        P.clip(pts, smooth=False)
        P.line([lerp2(Lp[0], Lp[1], 0.1), Lp[1], lerp2(Lp[1], Lp[2], 0.9)], dk, 1.6, "fold")
        P.unclip()
        if P.lod == 0:
            P.line([lerp2(Rp[0], Rp[1], 0.2), lerp2(Rp[0], Rp[1], 0.85)], lt, 0.42, "light")
        for t_ in (0.25, 0.55):
            a = lerp2(Lp[1], Lp[2], t_)
            b = lerp2(Rp[1], Rp[2], t_ + 0.12)
            P.line([a, lerp2(a, b, 0.5), b], k, 0.42, "fold")
    P.line(Lp, k, 0.5, "contour", smooth=False)
    P.line(Rp, k, 0.5, "contour", smooth=False)
    if cuff:
        cp = [lerp2(Lp[2], Lp[1], 0.16), Lp[2], Rp[2], lerp2(Rp[2], Rp[1], 0.16)]
        P.poly(cp, fill=cuff, line="ink", lw=0.35, cls="contour", mat="gold" if cuff == "gold" else None, smooth=False)
    return pts


def drape_from_arm(P, R, side, f, length=16.0, width=1.0, chrys=False):
    """a cascade of cloth hanging from the forearm (the mantle's zig-zag fall)"""
    el, wr = R.el[side], R.wr[side]
    sg = -1.0 if side == "r" else 1.0
    a = lerp2(el, wr, 0.15)
    b = lerp2(el, wr, 0.85)
    lo = max(a[1], b[1]) + length
    x0, x1 = (a[0], b[0]) if a[0] < b[0] else (b[0], a[0])
    xm = (x0 + x1) / 2
    wd = max(3.0, abs(x1 - x0)) * width
    pts = [a, b, (xm + wd * 0.55 * sg, lo - 3.0), (xm + wd * 0.25 * sg, lo - 1.0), (xm, lo), (xm - wd * 0.2 * sg, lo - 2.5),
           (xm - wd * 0.5 * sg, lo + 0.6)]
    cloth(P, pts, f, smooth=False)
    if P.lod < 2:
        for j in range(3):
            t_ = 0.25 + 0.25 * j
            p0 = lerp2(a, b, t_)
            P.line([p0, (p0[0] + (xm - p0[0]) * 0.4, p0[1] + length * 0.6)], _c(f, "_k"), 0.42, "fold")
            if chrys:
                P.line([(p0[0] - 0.8, p0[1] + 1.0), (p0[0] + (xm - p0[0]) * 0.4 - 0.8, p0[1] + length * 0.5)], "gold",
                       0.35, "gold", "gold")


# ================================================================ props
def prop_frame(R, side, sp):
    """(x, y, rot) where a held prop sits: above/in front of the hand"""
    x, y, rot, flip = _hand_frame(R, side, R.shape[side])
    HN = _hand_len(sp, R)
    return x, y, rot, flip, HN


def draw_prop(P, sp, R, side, prop, o):
    x, y, rot, flip, HN = prop_frame(R, side, sp)
    wr = R.wr[side]
    lod = P.lod
    if prop == "book":
        c = (wr[0] - 0.4 * (1 if side == "l" else -1), wr[1] - 4.8)
        P.push(c[0], c[1], -0.08 if side == "l" else 0.08, 1.0)
        w, h = 3.6, 4.8
        P.poly([(-w + 0.6, -h + 0.5), (w + 0.6, -h + 0.5), (w + 0.6, h + 0.5), (-w + 0.6, h + 0.5)], fill="white_d",
               line="ink", lw=0.4, cls="contour", smooth=False)
        P.poly([(-w, -h), (w, -h), (w, h), (-w, h)], fill="gold", mat="gold", smooth=False)
        if lod < 2:
            P.poly([(-w * 0.55, -h * 0.62), (w * 0.55, -h * 0.62), (w * 0.55, h * 0.62), (-w * 0.55, h * 0.62)],
                   line="gold_d", lw=0.35, cls="thin", smooth=False)
            P.dots([(-w * 0.72, -h * 0.8), (w * 0.72, -h * 0.8), (-w * 0.72, h * 0.8), (w * 0.72, h * 0.8)], 0.55, "pearl",
                   mat="pearl")
            P.dots([(0.0, 0.0)], 1.1, "ruby")
            P.dots([(0.0, -h * 0.55), (0.0, h * 0.55)], 0.6, "emerald")
            P.dots([(-w * 0.55, 0.0), (w * 0.55, 0.0)], 0.6, "sapphire")
        P.poly([(-w, -h), (w, -h), (w, h), (-w, h)], line="gold_d", lw=0.5, cls="contour", smooth=False)
        P.pop()
    elif prop == "codex_open":
        c = (0.0, wr[1] - 3.0)
        pts_l = [(c[0] - 8.0, c[1] - 4.5), (c[0], c[1] - 3.6), (c[0], c[1] + 4.6), (c[0] - 8.0, c[1] + 3.8)]
        pts_r = [(c[0] + 8.0, c[1] - 4.5), (c[0], c[1] - 3.6), (c[0], c[1] + 4.6), (c[0] + 8.0, c[1] + 3.8)]
        for pts in (pts_l, pts_r):
            P.poly(pts, fill="white_l", line="ink", lw=0.4, cls="contour", smooth=False)
        if lod < 2:
            for j in range(4):
                yy = c[1] - 2.2 + 1.5 * j
                P.line([(c[0] - 6.5, yy - 0.4), (c[0] - 1.2, yy)], "red_d" if j == 0 else "ink", 0.35, "fold")
                P.line([(c[0] + 1.2, yy), (c[0] + 6.5, yy - 0.4)], "ink", 0.35, "fold")
    elif prop == "scroll":
        P.push(x, y, rot, 1.0)
        sgx = flip
        P.poly([(-1.1 * sgx, -2.0), (1.1 * sgx, -2.0), (1.1 * sgx, 9.0), (-1.1 * sgx, 9.0)], fill="white", line="white_k",
               lw=0.4, cls="contour", smooth=False)
        P.ellipse(0.0, 9.0, 1.1, 0.45, fill="white_d", line="white_k", lw=0.3, cls="thin")
        if lod < 2:
            P.line([(0.0, -1.5), (0.0, 8.5)], "white_d", 0.35, "fold", smooth=False)
        P.pop()
    elif prop == "edict":
        c = (wr[0] + (1.5 if side == "l" else -1.5), wr[1] + 2.0)
        P.push(c[0], c[1], 0.0, 1.0)
        sheet = [(-3.4, -7.0), (3.4, -7.0), (3.2, 9.0), (-3.2, 9.0)]
        P.poly(sheet, fill="white_l", line="white_k", lw=0.4, cls="contour", smooth=False)
        for yy in (-7.0, 9.0):
            P.ellipse(0.0, yy, 3.8, 0.9, fill="white", line="white_k", lw=0.35, cls="contour")
        if lod < 2:
            for j in range(6):
                yy = -4.8 + 2.0 * j
                P.line([(-2.4, yy), (-0.4, yy + 0.25 * math.sin(j)), (2.3, yy - 0.2)], "ink", 0.3, "fold")
        # the imperial seal (bulla) on a red cord
        P.line([(0.0, 9.5), (0.2, 12.0)], "red", 0.4, "fold", smooth=False)
        P.circle(0.2, 13.0, 1.3, fill="gold", line="gold_d", lw=0.3, cls="thin", mat="gold")
        P.pop()
    elif prop == "weld":
        # Reseda luteola: a tall spike of tiny greenish flowers (the dyer's herb), held upright in the fist
        fd = rot + math.pi / 2
        cx_, cy_ = x + math.cos(fd) * 0.33 * HN, y + math.sin(fd) * 0.33 * HN
        base = (cx_ - 0.3, cy_ + 5.0)
        top = (cx_ + (1.6 if side == "l" else -1.6), cy_ - 31.0)
        P.line([base, lerp2(base, top, 0.5), top], "leaf_d", 0.55, "contour")
        for j, t_ in enumerate((0.08, 0.2, 0.33)):
            p = lerp2(base, top, t_)
            s_ = 1 if j % 2 else -1
            _lock(P, p, (p[0] + s_ * 3.2, p[1] - 3.0), 1.1, "leaf", "leaf_d", 0.25, mat=None, bend=0.2 * s_)
        spike0 = lerp2(base, top, 0.42)
        dx, dy = top[0] - spike0[0], top[1] - spike0[1]
        L_ = math.hypot(dx, dy)
        nx, ny = -dy / L_, dx / L_
        pts = [(spike0[0] + nx * 1.4, spike0[1] + ny * 1.4), (top[0] + nx * 0.2, top[1] + ny * 0.2),
               (top[0] - nx * 0.2, top[1] - ny * 0.2), (spike0[0] - nx * 1.4, spike0[1] - ny * 1.4)]
        P.poly(pts, fill="weld", line="weld_d", lw=0.35, cls="fold", smooth=False)
        if lod < 2:
            pts2 = []
            for j in range(12):
                t_ = j / 12.0
                p = lerp2(spike0, top, t_)
                w_ = 1.5 * (1 - t_) + 0.3
                s_ = 1 if j % 2 else -1
                pts2.append((p[0] + nx * w_ * s_, p[1] + ny * w_ * s_))
            P.dots(pts2, 0.55, "weld_l")
    elif prop == "church":
        c = (wr[0], wr[1] - 5.2)
        P.push(c[0], c[1], 0.0, 1.0)
        P.poly([(-4.2, -1.0), (4.2, -1.0), (4.2, 4.2), (-4.2, 4.2)], fill="#C9A98A", line="ink", lw=0.4, cls="contour",
               smooth=False)
        P.poly([(-4.8, -1.0), (0.0, -3.4), (4.8, -1.0)], fill="porphyry", line="ink", lw=0.4, cls="contour", smooth=False)
        P.poly([(-2.2, -2.6), (-2.2, -4.2), (2.2, -4.2), (2.2, -2.6)], fill="#C9A98A", line="ink", lw=0.35, cls="contour",
               smooth=False)
        P.ellipse(0.0, -4.4, 2.3, 2.1, fill="gold", line="gold_d", lw=0.3, cls="thin", mat="gold")
        P.poly([(-2.4, -4.4), (2.4, -4.4), (2.4, -4.0), (-2.4, -4.0)], fill="#C9A98A", smooth=False)
        if lod < 2:
            for xx in (-2.6, 0.0, 2.6):
                P.poly([(xx - 0.6, 0.6), (xx + 0.6, 0.6), (xx + 0.6, 3.4), (xx - 0.6, 3.4)], fill="lapis_d", smooth=False)
        P.pop()
    elif prop == "orb":
        c, r = orb_geom(R, side, sp)
        P.circle(c[0], c[1], r, fill="lapis", line="lapis_k", lw=0.4, cls="contour")
        if lod < 2:
            P.line([(c[0] - r, c[1] + 0.2), (c[0] + r, c[1] + 0.2)], "gold", 0.5, "gold", "gold", smooth=False)
            P.ellipse(c[0] - r * 0.35, c[1] - r * 0.4, r * 0.28, r * 0.2, fill="lapis_l")
    elif prop == "candle":
        pass
    elif prop == "sword":
        pass


def orb_geom(R, side, sp):
    x, y, rot, flip = _hand_frame(R, side, R.shape[side])
    HN = _hand_len(sp, R)
    r = 0.42 * HN
    return (x + math.cos(rot - math.pi / 2) * 0.0, y - 0.55 * HN - r * 0.4), r


def candle_geom(R, sp):
    """(bottom, top) of the candle held in the 'candle' pose"""
    if R.pole is None:
        return None
    base, (dx, dy), mode = R.pole
    top = R.wr["r"]
    L = 14.0
    return (top[0] - dx * 4.0, top[1] - dy * 4.0), (top[0] + dx * (L - 4.0), top[1] + dy * (L - 4.0))


def draw_candle(P, R, sp):
    g = candle_geom(R, sp)
    if g is None:
        return
    (x0, y0), (x1, y1) = g
    w = 1.2
    P.poly([(x0 - w, y0), (x0 + w, y0), (x1 + w * 0.85, y1), (x1 - w * 0.85, y1)], fill="wax", line="wax_d", lw=0.35,
           cls="contour", smooth=False)
    if P.lod < 2:
        P.line([(x0 + 0.5, y0 - 1.0), (x1 + 0.4, y1 + 1.0)], "wax_d", 0.35, "fold", smooth=False)
    P.line([(x1, y1), (x1, y1 - 1.1)], "ink", 0.3, "fine2", smooth=False)


def sword_at_hip(P, R, sp):
    """a sheathed spatha at the left hip (viewer's right), hilt near the belt"""
    U = R.extra["up"]
    hilt = U((sp["waist_w"] + 0.8, -58.0))
    tip = (hilt[0] + 4.0, -26.0)
    P.line([hilt, tip], "leather_d", 1.8, "contour", smooth=False)
    P.line([hilt, tip], "leather", 1.1, "fold", smooth=False)
    P.line([(hilt[0] - 2.2, hilt[1] - 0.2), (hilt[0] + 2.2, hilt[1] + 0.4)], "gold", 0.8, "gold", "gold", smooth=False)
    P.line([hilt, (hilt[0] - 0.4, hilt[1] - 3.2)], "leather_d", 0.8, "fold", smooth=False)
    P.circle(hilt[0] - 0.5, hilt[1] - 3.8, 0.7, fill="gold", line="gold_d", lw=0.2, cls="thin", mat="gold")


def draw_altar(ctx, x, y, h, layer="color", tile=None, alpha=1.0, tint=None):
    """a tiny altar (table + white cloth + ciborium dome on four columns). (x, y) = ground centre, h = total height.
    Returns dict(top=(x, y_top), table=(x, y_table))."""
    from .icon import Painter
    P = Painter(ctx, layer, tile, alpha, tint)
    u = h / 123.2
    P.push(x, y, 0.0, u)
    P.lod = 0 if h > 90 else 1
    # table
    P.poly([(-30, -46), (30, -46), (27, 0), (-27, 0)], fill="#C9A98A", line="ink", lw=2.0, cls="contour", smooth=False)
    P.poly([(-34, -52), (34, -52), (34, -44), (-34, -44)], fill="white", line="white_k", lw=1.6, cls="contour", smooth=False)
    P.poly([(-12, -44), (12, -44), (12, -18), (-12, -18)], fill="porphyry", line="ink", lw=1.4, cls="contour",
           smooth=False)
    if P.lod == 0:
        P.poly([(-10, -40), (10, -40), (10, -22), (-10, -22)], line="gold", lw=1.4, cls="gold", mat="gold", smooth=False)
    # ciborium: four columns + dome
    for xx in (-26, -9, 9, 26):
        P.poly([(xx - 2.2, -86), (xx + 2.2, -86), (xx + 2.2, -52), (xx - 2.2, -52)], fill="white", line="white_k", lw=1.2,
               cls="contour", smooth=False)
    P.poly([(-32, -86), (32, -86), (32, -92), (-32, -92)], fill="gold", line="gold_d", lw=1.2, cls="contour", mat="gold",
           smooth=False)
    P.poly([(-26, -92), (-24, -104), (-12, -114), (0, -117), (12, -114), (24, -104), (26, -92)], fill="gold",
           line="gold_d", lw=1.2, cls="contour", mat="gold", smooth=True)
    P.circle(0, -120, 3.2, fill="gold", line="gold_d", lw=1.0, cls="thin", mat="gold")
    P.pop()
    return dict(top=(x, y - h), table=(x, y - 52.0 * u))


# ================================================================ costumes
def _throne(P, sp, R):
    main = R.extra.get("main")
    U = R.extra["up"]
    shy = R.shoulder_y
    if main == "sit":
        # a plain marble bench
        P.poly([(-15.0, -31.5), (15.0, -31.5), (15.0, -27.5), (-15.0, -27.5)], fill="white", line="white_k", lw=0.5,
               cls="contour", smooth=False)
        P.poly([(-13.5, -27.5), (13.5, -27.5), (13.0, 0.0), (-13.0, 0.0)], fill="white_d", line="white_k", lw=0.5,
               cls="contour", smooth=False)
        return
    top = U((0.0, shy - 9.0))
    back = [(-14.0, -30.0), (-14.5, top[1] + 8.0), (-12.0, top[1] + 1.5), (0.0, top[1] - 1.5), (12.0, top[1] + 1.5),
            (14.5, top[1] + 8.0), (14.0, -30.0)]
    P.poly(back, fill="gold", mat="gold", smooth=True)
    if P.lod < 2:
        inner = [(-11.5, -31.0), (-11.8, top[1] + 8.5), (-9.8, top[1] + 3.8), (0.0, top[1] + 1.4), (9.8, top[1] + 3.8),
                 (11.8, top[1] + 8.5), (11.5, -31.0)]
        P.poly(inner, fill="lapis", line="gold_d", lw=0.4, cls="contour", smooth=True)
        pts = []
        for i in range(15):
            t_ = i / 14.0
            a = math.pi * (1.0 - t_)
            pts.append((13.0 * math.cos(a), top[1] + 7.0 - 7.0 * math.sin(a)))
        P.dots(pts, 0.55, "pearl", mat="pearl")
    P.poly(back, line="gold_d", lw=0.5, cls="contour", smooth=True)
    # cushion bolster
    P.poly([(-17.5, -34.0), (17.5, -34.0), (18.5, -31.0), (17.5, -28.0), (-17.5, -28.0), (-18.5, -31.0)], fill="porphyry",
           line="porphyry_k", lw=0.5, cls="contour", smooth=True)
    for sg in (-1, 1):
        P.ellipse(sg * 17.2, -31.0, 1.6, 3.0, fill="gold", line="gold_d", lw=0.3, cls="thin", mat="gold")
    # seat front + footstool
    P.poly([(-15.5, -28.0), (15.5, -28.0), (15.0, -5.0), (-15.0, -5.0)], fill="gold", line="gold_d", lw=0.5,
           cls="contour", mat="gold", smooth=False)
    if P.lod < 2:
        for xx in (-10.0, 0.0, 10.0):
            P.poly([(xx - 3.2, -24.0), (xx + 3.2, -24.0), (xx + 3.2, -9.0), (xx - 3.2, -9.0)], fill="lapis_d",
                   line="gold_d", lw=0.35, cls="thin", smooth=False)
            P.dots([(xx, -16.5)], 1.0, "ruby")
    P.poly([(-10.5, -5.0), (10.5, -5.0), (11.0, 0.0), (-11.0, 0.0)], fill="gold", line="gold_d", lw=0.5, cls="contour",
           mat="gold", smooth=False)
    if P.lod < 2:
        P.dots([(-7.0, -2.5), (0.0, -2.5), (7.0, -2.5)], 0.8, "emerald")


def _hood_roll(P, R, f):
    U = R.extra["up"]
    SW, shy = R.SW, R.shoulder_y
    pts = [U((-SW + 1.2, shy + 0.6)), U((-SW + 1.6, shy - 2.0)), U((-4.0, shy - 4.2)), U((0.0, shy - 4.9)),
           U((4.0, shy - 4.2)), U((SW - 1.6, shy - 2.0)), U((SW - 1.2, shy + 0.6)), U((SW - 3.0, shy + 1.2)),
           U((3.0, shy + 0.6)), U((0.0, shy + 2.9)), U((-3.0, shy + 0.6)), U((-SW + 3.0, shy + 1.2))]
    cloth(P, pts, f, smooth=True, shade=0.5, light=0.5)
    if P.lod < 2:
        P.line([U((-SW + 2.6, shy - 1.0)), U((-3.4, shy - 2.8)), U((0.0, shy - 3.3)), U((3.4, shy - 2.8)),
                U((SW - 2.6, shy - 1.0))], _c(f, "_k"), 0.45, "fold")


def _clavi(P, R, sp, c, top=None, bot=None, x=4.2, w=1.4):
    U = R.extra["up"]
    shy = R.shoulder_y
    bot = sp["hem_y"] if bot is None else bot
    for sg in (-1, 1):
        a = U((sg * x, shy - 1.0))
        b = (sg * (x + 1.2), bot - 0.5)
        P.line([a, lerp2(a, b, 0.5), b], c, w, "fold", smooth=False)


def _arms_generic(P, F, R, o, sleeve_f, cuff=None, radii=(3.1, 2.7, 2.3), bell=0.0, cover=(), drape=None, bare=(),
                  after=None):
    sp = F.sp
    order = ("l", "r") if R.m * (1 if R.turn > 0 else 0) >= 0 else ("r", "l")
    for side in order:
        if side in cover:
            continue
        if side in bare:
            draw_sleeve(P, sp, R, side, sleeve_f, None, radii, 0.0, short=0.1)
            S = sp["skin"]
            el, wr = R.el[side], R.wr[side]
            Lp, Rp = sleeve_poly(lerp2(R.sh[side], el, 0.65), el, wr, 2.2, 2.1, 1.6)
            P.poly(Lp + list(reversed(Rp)), fill=S["flesh"], line=S["contour"], lw=0.4, cls="fine2", mat=FINE,
                   smooth=False)
            continue
        draw_sleeve(P, sp, R, side, sleeve_f, cuff, radii, bell)
    if drape:
        drape()
    _hands_and_props(P, F, R, o)
    if after:
        after()


def _hands_and_props(P, F, R, o):
    sp = F.sp
    prop = R.prop
    main = R.extra.get("main")
    hold_side = None
    for side in ("l", "r"):
        if prop and (R.shape[side] in ("hold", "cup", "open") or (prop == "weld" and R.shape[side] == "fist")):
            hold_side = side
            break
    if main == "candle":
        draw_candle(P, R, sp)
    if prop == "codex_open" and main in ("offer", "stand", "present", "march"):
        draw_prop(P, sp, R, "l", prop, o)
        hold_side = None
    if hold_side and prop not in ("codex_open",):
        draw_prop(P, sp, R, hold_side, prop, o)
    if main == "crown_self":
        pr = clamp(float(o.get("progress", 1.0)), 0.0, 1.0)
        if pr < 0.999:
            lift = (1.0 - pr) * 0.62
            P.push(R.head[0], R.head[1], R.head_rot, R.HL)
            tiara(P, 0.0, -lift, 1.0, P.lod)
            P.pop()
    for side in ("r", "l"):
        if R.shape[side] == "grip" and R.pole is not None:
            continue
        draw_hand(P, F, R, side)
        if R.shape[side] == "stylus":
            x, y, rot, flip = _hand_frame(R, side, "stylus")
            HN = _hand_len(sp, R)
            P.push(x, y, rot, HN * flip, HN)
            P.line([(0.12, 0.2), (0.08, 1.15)], "gold", 0.07, "gold", "gold", smooth=False)
            P.pop()
    if prop == "orb" and hold_side is None:
        pass


def dress_novice(P, F, R, o, stage):
    sp = F.sp
    f = sp["robe"]
    if stage == "back":
        if R.sit:
            _throne(P, sp, R)
    elif stage == "body":
        draw_feet(P, sp, R, sp["shoes"])
        pts = robe_outline(R, sp)
        cloth(P, pts, f)
        lower_folds(P, R, sp, f, R.waist_y + 3.0, sp["hem_y"] - 0.5, sp["waist_w"] * 0.8, sp["hem_w"] * 0.95, n=5)
        if P.lod < 2 and not R.sit and not R.kneel:
            hem_pts = sorted([p for p in pts if p[1] > sp["hem_y"] - 3.0], key=lambda p: p[0])
            if len(hem_pts) >= 2:
                P.line([(p[0], p[1] - 1.1) for p in hem_pts], sp.get("cuff") or _c(f, "_d"), 1.4, "fold")
        belt(P, R, sp, sp.get("belt") or _c(f, "_k"), w=1.6, y=-60.0)
    elif stage == "collar":
        if sp.get("head") == "hood_down":
            _hood_roll(P, R, f)
    elif stage == "arms":
        _arms_generic(P, F, R, o, sp["sleeve"] or f, sp.get("cuff"), radii=(3.0, 2.6, 2.3), bell=1.3)


def dress_hierarch(P, F, R, o, stage):
    sp = F.sp
    U = R.extra["up"]
    SW, shy = R.SW, R.shoulder_y
    oc = sp.get("over") or "ochre"
    if stage == "back":
        if R.sit:
            _throne(P, sp, R)
    elif stage == "body":
        draw_feet(P, sp, R, sp["shoes"])
        pts = robe_outline(R, sp)
        cloth(P, pts, "black")
        lower_folds(P, R, sp, "black", -30.0 + (R.drop if not R.sit else 0), sp["hem_y"] - 0.5, sp["hip_w"], sp["hem_w"],
                    n=4, seed=3, fields=False)
        # the phelonion: a bell-shaped ochre cape to the knees, with gold chrysography
        hw = sp["hem_w"] + 3.2
        low = -27.0 if not R.sit else -26.0
        if R.kneel:
            low = -6.0
        ph = [U((-SW - 2.0, shy + 1.2)), U((-SW - 1.2, shy - 2.4)), U((-3.8, shy - 4.2)), U((3.8, shy - 4.2)),
              U((SW + 1.2, shy - 2.4)), U((SW + 2.0, shy + 1.2)), U((SW + 4.8, shy + 19.0)), (hw, low - 3.5),
              (hw - 1.2, low), (6.0, low + 1.8), (0.0, low + 2.2), (-6.0, low + 1.8), (-(hw - 1.2), low), (-hw, low - 3.5),
              U((-SW - 4.8, shy + 19.0))]
        cloth(P, ph, oc, shade=1.2)
        if P.lod < 2:
            for i, sg in enumerate((-1, -1, 1, 1)):
                k = (i % 2)
                a = U((sg * (3.0 + 3.2 * k), shy + 3.0 + 3.0 * k))
                b = (sg * (5.0 + 5.8 * k), low + 1.0)
                m = lerp2(a, b, 0.5)
                m = (m[0] + sg * 1.2, m[1])
                P.line([a, m, b], _c(oc, "_k"), 0.5, "fold")
                P.line([(a[0] - 1.0, a[1] + 1.5), (m[0] - 1.0, m[1]), (b[0] - 1.0, b[1] - 1.5)], "gold", 0.42, "gold", "gold")
        # the omophorion (a stole, plain: no crosses) - Y over the shoulders + a long hanging band
        wst = 3.2
        y_path = [U((-SW + 0.6, shy - 1.8)), U((-4.0, shy + 5.0)), U((0.0, shy + 9.0)), U((4.0, shy + 5.0)),
                  U((SW - 0.6, shy - 1.8))]
        band(P, y_path, wst, "ochre_l", edge="gold", edge_w=0.45)
        hang = [U((0.0, shy + 8.0)), U((0.0, shy + 22.0)), (0.0, low + 5.5)]
        band(P, hang, wst, "ochre_l", edge="gold", edge_w=0.45)
        if P.lod < 2:
            P.poly([(-wst / 2, low + 5.5), (wst / 2, low + 5.5), (wst / 2 + 0.2, low + 7.3), (-wst / 2 - 0.2, low + 7.3)],
                   fill="gold", mat="gold", smooth=False)
    elif stage == "arms":
        def cape_over_arms():
            # the phelonion drapes over the upper arms; the black forearms emerge from under its edge
            for side in ("r", "l"):
                sh, el = R.sh[side], R.el[side]
                end = lerp2(sh, el, 0.9)
                Lp, Rp = sleeve_poly(sh, lerp2(sh, el, 0.5), end, 3.8, 3.8, 3.6)
                region = Lp + list(reversed(Rp))
                cloth(P, region, oc, smooth=False, shade=0.6, light=0.6, lw=0.0)
                P.circle(sh[0], sh[1], 3.8, fill=oc)
                P.line(Lp, _c(oc, "_k"), 0.5, "contour", smooth=False)
                P.line(Rp, _c(oc, "_k"), 0.5, "contour", smooth=False)
                m_ = lerp2(Lp[2], Rp[2], 0.5)
                P.line([Lp[2], (m_[0], m_[1] + 0.9), Rp[2]], _c(oc, "_k"), 0.55, "contour")
                if P.lod < 2:
                    P.line([lerp2(sh, el, 0.12), lerp2(sh, el, 0.8)], "gold", 0.4, "gold", "gold", smooth=False)
        _arms_generic(P, F, R, o, "black", "gold", radii=(2.9, 2.6, 2.1), drape=cape_over_arms)


def dress_warrior(P, F, R, o, stage):
    sp = F.sp
    U = R.extra["up"]
    SW, shy = R.SW, R.shoulder_y
    cl = sp.get("cloak") or "porphyry"
    wind = R.wind
    if stage == "back":
        wx = 7.0 * wind
        cape = [U((-SW - 1.0, shy - 1.5)), U((SW + 1.0, shy - 1.5)), U((SW + 5.0, shy + 20.0)), (SW + 8.0 + wx, -22.0),
                (SW + 6.0 + wx * 1.4, -14.0), (4.0 + wx, -12.5), (-4.0 + wx * 0.8, -12.5), (-SW - 6.0 + wx * 0.6, -14.0),
                (-SW - 7.5 + wx * 0.5, -22.0), U((-SW - 5.0, shy + 20.0))]
        cloth(P, cape, cl, shade=1.3)
        if P.lod < 2:
            for sg in (-1, 1):
                P.line([U((sg * (SW + 2.0), shy + 6.0)), (sg * (SW + 4.5) + wx * 0.8, -30.0), (sg * (SW + 5.0) + wx, -16.0)],
                       _c(cl, "_k"), 0.5, "fold")
    elif stage == "body":
        warrior_legs(P, sp, R, legc="leather_d", pads=True)
        f = sp["robe"]
        pts = robe_outline(R, sp, hem_w=sp["hem_w"] + 0.5)
        cloth(P, pts, f)
        lower_folds(P, R, sp, f, -50.0 + R.drop, sp["hem_y"] - 0.5, sp["hip_w"], sp["hem_w"] + 0.3, n=4, seed=5)
        hem_pts = sorted([p for p in pts if p[1] > sp["hem_y"] - 2.5], key=lambda p: p[0])
        if P.lod < 2 and len(hem_pts) >= 2:
            P.line([(p[0], p[1] - 1.0) for p in hem_pts], "gold", 1.2, "gold", "gold")
        # lamellar klibanion (silver plates in rows)
        ww, pw = sp["waist_w"], sp["hip_w"]
        cu = [U((-SW + 0.2, shy - 1.2)), U((-5.0, shy - 2.6)), U((5.0, shy - 2.6)), U((SW - 0.2, shy - 1.2)),
              U((SW + 0.9, shy + 9.0)), U((ww + 0.8, -60.0)), U((pw + 0.4, -52.0)), U((-pw - 0.4, -52.0)),
              U((-ww - 0.8, -60.0)), U((-SW - 0.9, shy + 9.0))]
        P.poly(cu, fill="steel_l", mat="silver", smooth=False)
        if P.lod < 2:
            ctx = P.ctx
            y0 = shy + 1.0
            tl = P.tile / P.k if P.tile else 0.0
            rh = max(2.6, 2.0 * tl)
            pw_ = max(2.2, 1.7 * tl)
            rows = int((-52.0 - y0) / rh)
            P.clip(cu, smooth=False)
            ctx.new_path()
            for r_ in range(rows + 1):
                yy = y0 + r_ * rh
                ctx.move_to(*U((-pw - 2.0, yy)))
                ctx.line_to(*U((pw + 2.0, yy)))
            ctx.set_line_width(P.width(0.36, "fold"))
            P._src("steel_k", None, True)
            ctx.stroke()
            if P.lod == 0:
                ctx.new_path()
                for r_ in range(rows + 1):
                    yy = y0 + r_ * rh
                    off = pw_ * 0.5 if r_ % 2 else 0.0
                    for c_ in range(-8, 9):
                        xx = c_ * pw_ + off
                        ctx.move_to(*U((xx, yy)))
                        ctx.line_to(*U((xx, yy + rh)))
                ctx.set_line_width(P.width(0.22, "thin"))
                P._src("steel_d", None, True)
                ctx.stroke()
            P.push(-1.0, 0.0)
            P.poly(cu, line="steel_d", lw=2.4, cls="thin", mat="silver", smooth=False)
            P.pop()
            P.unclip()
        P.poly(cu, line="steel_k", lw=0.5, cls="contour", smooth=False)
        # pteruges (leather strips) below the cuirass
        n = 8
        for i in range(n):
            x0 = -pw - 0.2 + (2 * pw + 0.4) * (i + 0.5) / n
            a = U((x0, -52.2))
            b = (x0 * 1.06, -43.0 + R.drop)
            P.poly([(a[0] - 1.0, a[1]), (a[0] + 1.0, a[1]), (b[0] + 1.0, b[1]), (b[0] - 1.0, b[1])], fill="porphyry",
                   line="porphyry_k", lw=0.35, cls="fold", smooth=False)
            if P.lod < 2:
                P.poly([(b[0] - 1.0, b[1] - 1.2), (b[0] + 1.0, b[1] - 1.2), (b[0] + 1.0, b[1]), (b[0] - 1.0, b[1])],
                       fill="gold", mat="gold", smooth=False)
        belt(P, R, sp, "gold", w=1.8, y=-60.0, ends=False, mat="gold")
        if R.extra.get("main") in ("lance", "hold_pole1", "stand", "march"):
            sword_at_hip(P, R, sp)
        # cloak edge + fibula on the right shoulder
        fb = U((-SW + 1.0, shy - 0.8))
        P.poly([U((-SW - 1.2, shy - 1.6)), fb, U((-SW + 0.5, shy + 5.0)), U((-SW - 2.2, shy + 12.0))], fill=cl,
               line=_c(cl, "_k"), lw=0.45, cls="contour", smooth=True)
        P.circle(fb[0], fb[1], 1.35, fill="gold", line="gold_d", lw=0.3, cls="thin", mat="gold")
        P.circle(fb[0], fb[1], 0.6, fill="ruby")
    elif stage == "collar":
        pass
    elif stage == "arms":
        def shoulder_guards():
            for side in ("r", "l"):
                sh, el = R.sh[side], R.el[side]
                for j in range(3):
                    a = lerp2(sh, el, 0.05 + 0.1 * j)
                    b = lerp2(sh, el, 0.38 + 0.08 * j)
                    off = (j - 1) * 1.6
                    dx, dy = el[0] - sh[0], el[1] - sh[1]
                    d = math.hypot(dx, dy) + 1e-9
                    nx, ny = -dy / d * off, dx / d * off
                    P.poly([(a[0] + nx - 0.9, a[1] + ny), (a[0] + nx + 0.9, a[1] + ny), (b[0] + nx + 0.9, b[1] + ny),
                            (b[0] + nx - 0.9, b[1] + ny)], fill="porphyry", line="porphyry_k", lw=0.3, cls="fold",
                           smooth=False)
        _arms_generic(P, F, R, o, sp["sleeve"], "steel", radii=(3.2, 2.7, 2.3), drape=shoulder_guards)


def dress_emperor(P, F, R, o, stage):
    sp = F.sp
    U = R.extra["up"]
    SW, shy = R.SW, R.shoulder_y
    tun = sp["robe"]
    ch = sp.get("over") or "navy"
    kind = sp["kind"]
    if stage == "back":
        if R.sit:
            _throne(P, sp, R)
    elif stage == "body":
        draw_feet(P, sp, R, sp["shoes"])
        pts = robe_outline(R, sp)
        cloth(P, pts, tun, shade=0.8)
        lower_folds(P, R, sp, tun, -58.0 + R.drop, sp["hem_y"] - 0.5, sp["waist_w"], sp["hem_w"], n=4, seed=2)
        hem_pts = sorted([p for p in pts if p[1] > sp["hem_y"] - 2.5], key=lambda p: p[0])
        if len(hem_pts) >= 2:
            P.line([(p[0], p[1] - 1.1) for p in hem_pts], "gold", 1.3, "gold", "gold")
        if sp.get("clavi"):
            _clavi(P, R, sp, sp["clavi"], x=-5.5, w=1.2)
        if kind == "jaguar":
            # the red tie (an echo of the President's suit)
            P.poly([U((-0.9, shy - 2.6)), U((0.9, shy - 2.6)), U((1.5, shy + 13.0)), U((0.0, shy + 15.5)),
                    U((-1.5, shy + 13.0))], fill="red", line="red_k", lw=0.35, cls="contour", smooth=False)
        # the chlamys: fastened on the right shoulder, covering the left side and the left arm
        hw = sp["hem_w"] + 1.2
        low = sp["hem_y"] - 2.5 if not R.sit else -6.0
        if R.kneel:
            low = -1.0
        chl = [U((-SW + 0.8, shy - 1.6)), U((-2.0, shy - 3.4)), U((2.5, shy - 3.4)), U((SW + 1.2, shy - 1.8)),
               U((SW + 2.4, shy + 3.0)), U((SW + 3.4, shy + 15.0)), (hw + 0.8, -30.0), (hw, low), (2.0, low + 0.8),
               (-4.0, low + 0.6), (-4.5, -30.0), U((-3.2, shy + 18.0)), U((-3.5, shy + 6.0))]
        if kind == "jaguar":
            chl[1] = U((2.2, shy + 1.0))
            chl[2] = U((3.2, shy - 3.0))
        cloth(P, chl, ch, shade=1.2)
        if P.lod < 2:
            for j in range(4):
                a = U((-1.0 + 3.2 * j, shy + 2.0 + 1.2 * j))
                b = (-2.0 + 4.0 * j, low - 1.0)
                P.line([a, lerp2(a, b, 0.5), b], _c(ch, "_k"), 0.45, "fold")
                if P.lod == 0:
                    P.line([(a[0] - 1.0, a[1] + 3.0), (b[0] - 1.0, b[1] - 3.0)], _c(ch, "_l"), 0.4, "light")
        # the tablion: an embroidered panel on the front edge
        tb = sp.get("tablion", "gold")
        tpts = [(-4.6, -64.0 + R.drop), (3.6, -64.8 + R.drop), (3.8, -45.0 + R.drop), (-4.4, -44.2 + R.drop)]
        if R.sit:
            tpts = [(p[0], p[1] + 8.0) for p in tpts]
        P.poly(tpts, fill=tb, line="ink", lw=0.4, cls="contour", mat="gold" if tb == "gold" else None, smooth=False)
        if P.lod < 2:
            cx_ = sum(p[0] for p in tpts) / 4
            cy_ = sum(p[1] for p in tpts) / 4
            for i in range(2):
                for j in range(3):
                    x_ = cx_ - 2.0 + 4.0 * i
                    y_ = cy_ - 6.0 + 6.0 * j
                    P.circle(x_, y_, 1.3, fill="green_c" if tb == "gold" else "gold", line="red" if tb == "gold" else None,
                             lw=0.4, cls="thin", mat=None if tb == "gold" else "gold")
        # the fibula (a big jewelled brooch with three pendants) on the right shoulder
        fb = U((-SW + 0.6, shy - 0.4))
        P.circle(fb[0], fb[1], 1.9, fill="gold", line="gold_d", lw=0.35, cls="thin", mat="gold")
        P.circle(fb[0], fb[1], 0.9, fill="ruby" if kind == "jaguar" else "sapphire")
        if P.lod < 2:
            for k_ in (-1, 0, 1):
                a = (fb[0] + 0.9 * k_, fb[1] + 1.8)
                b = (fb[0] + 1.1 * k_, fb[1] + 4.2)
                P.line([a, b], "gold", 0.3, "gold", "gold", smooth=False)
                P.circle(b[0], b[1] + 0.5, 0.55, fill="pearl", mat="pearl")
        if sp.get("tail"):
            _tail(P, sp, R)
    elif stage == "collar":
        if kind == "jaguar":
            a = U((0.0, shy - 1.0))
            pts = [(a[0] - 5.2, a[1] - 1.2), (a[0] - 2.6, a[1] + 1.8), (a[0], a[1] + 2.6), (a[0] + 2.6, a[1] + 1.8),
                   (a[0] + 5.2, a[1] - 1.2)]
            band(P, pts, 2.0, "gold", mat="gold", lw_out=0.35, line_c="gold_d")
            if P.lod < 2:
                P.dots([lerp2(pts[i], pts[i + 1], f) for i in range(4) for f in (0.25, 0.75)], 0.45, "pearl", mat="pearl")
                P.dots([pts[2]], 0.75, "ruby")
    elif stage == "arms":
        def cover_l():
            # the chlamys drapes over the left forearm
            drape_from_arm(P, R, "l", ch, length=9.0, width=0.8)
        _arms_generic(P, F, R, o, sp["sleeve"], sp.get("cuff") or "gold", radii=(3.0, 2.6, 2.2), cover=("l",),
                      drape=cover_l)


def _tail(P, sp, R):
    """the jaguar's spotted tail tip, curling out from under the hem (a small heresy)"""
    hw = sp["hem_w"]
    y0 = sp["hem_y"] - 1.5 if not R.sit else -6.0
    t = R.t
    sw = 1.2 * math.sin(t * 1.7)
    pts = [(hw - 1.0, y0 - 0.5), (hw + 2.5, y0 - 1.0), (hw + 5.0 + sw * 0.3, y0 - 3.5), (hw + 5.5 + sw, y0 - 7.5),
           (hw + 3.8 + sw, y0 - 10.0)]
    P.line(pts, "fur_d", 2.5, "contour")
    P.line(pts, "fur", 1.6, "fold")
    if P.lod < 2:
        P.dots([pts[1], (pts[2][0] + 0.2, pts[2][1] + 0.4)], 0.55, "rosette")
    P.circle(pts[-1][0], pts[-1][1], 1.1, fill="rosette")


def dress_tunic(P, F, R, o, stage):
    sp = F.sp
    U = R.extra["up"]
    SW, shy = R.SW, R.shoulder_y
    f = sp["robe"]
    ov = sp.get("over")
    if stage == "back":
        if R.sit:
            _throne(P, sp, R)
        if ov:
            cape = [U((-SW - 1.0, shy - 1.8)), U((SW + 1.0, shy - 1.8)), U((SW + 3.5, shy + 16.0)), (SW + 4.2, -30.0),
                    (SW + 3.0, -20.0), (-SW - 3.0, -20.0), (-SW - 4.2, -30.0), U((-SW - 3.5, shy + 16.0))]
            if R.sit or R.kneel:
                cape = cape[:3] + [(SW + 4.0, -8.0 if R.sit else -2.0), (-SW - 4.0, -8.0 if R.sit else -2.0)] + cape[-1:]
            cloth(P, cape, ov, shade=1.2)
    elif stage == "body":
        draw_feet(P, sp, R, sp.get("shoes", "leather"))
        pts = robe_outline(R, sp)
        cloth(P, pts, f)
        if sp.get("clavi") and P.lod < 2:
            _clavi(P, R, sp, sp["clavi"])
        lower_folds(P, R, sp, f, R.waist_y + 3.0, sp["hem_y"] - 0.5, sp["waist_w"] * 0.8, sp["hem_w"] * 0.95, n=4)
        if not ov:
            chest_folds(P, R, f, n=2, depth=3.0, w=0.62, y_off=2.0)
            belt(P, R, sp, _c(f, "_k"), w=1.3, y=-60.0, ends=False)
        else:
            for sg in (-1, 1):
                fr = [U((sg * (SW - 2.0), shy - 2.0)), U((sg * (SW + 1.6), shy + 0.5)), U((sg * (SW + 2.4), shy + 16.0)),
                      (sg * (SW + 2.8), -33.0 if not R.sit else -26.0), (sg * (SW - 1.2), -34.5 if not R.sit else -27.0),
                      U((sg * (SW - 3.2), shy + 14.0))]
                cloth(P, fr, ov, shade=0.8, smooth=True)
    elif stage == "arms":
        _arms_generic(P, F, R, o, sp.get("sleeve") or f, None, radii=(3.0, 2.6, 2.2))


def dress_courtier(P, F, R, o, stage):
    dress_emperor(P, F, R, o, stage)


def dress_guard(P, F, R, o, stage):
    sp = F.sp
    U = R.extra["up"]
    shy = R.shoulder_y
    f = sp["robe"]
    if stage == "body":
        warrior_legs(P, sp, R, legc=_c(f, "_d"), pads=False)
        pts = robe_outline(R, sp)
        cloth(P, pts, f)
        lower_folds(P, R, sp, f, -52.0 + R.drop, sp["hem_y"] - 0.5, sp["hip_w"], sp["hem_w"], n=4, seed=4)
        chest_folds(P, R, f, n=3, depth=3.2, w=0.66, y_off=1.5)
        belt(P, R, sp, "leather", w=1.5, y=-60.0, ends=False)
        if P.lod < 2:
            _clavi(P, R, sp, "porphyry", bot=sp["hem_y"], x=3.8, w=1.0)
    elif stage == "collar":
        if sp.get("torque"):
            a = U((0.0, shy - 1.2))
            P.ellipse(a[0], a[1], 3.6, 1.6, line="gold", lw=0.9, cls="gold", mat="gold")
    elif stage == "arms":
        def spear():
            if sp.get("spear") and R.pole is not None:
                base, (dx, dy), mode = R.pole
                L = 130.0
                top = (base[0] + dx * L, base[1] + dy * L)
                P.line([base, top], "wood_d", 1.4, "contour", smooth=False)
                P.line([base, top], "wood", 0.8, "fold", smooth=False)
                nx, ny = -dy, dx
                P.poly([(top[0] + nx * 1.6, top[1] + ny * 1.6), (top[0] + dx * 7.0, top[1] + dy * 7.0),
                        (top[0] - nx * 1.6, top[1] - ny * 1.6), (top[0] - dx * 1.5, top[1] - dy * 1.5)], fill="steel",
                       line="steel_k", lw=0.4, cls="contour", mat="silver", smooth=False)

        def shield():
            if sp.get("shield"):
                el, wr = R.el["l"], R.wr["l"]
                c = lerp2(el, wr, 0.55)
                c = (c[0] + 1.5, c[1] + 1.0)
                sc = sp.get("shield_c", _pick(sp["seed"], 9, ("lapis", "porphyry", "green_c", "white")))
                P.ellipse(c[0], c[1], 8.2, 12.5, fill=sc, line="gold_d", lw=1.2, cls="contour")
                P.ellipse(c[0], c[1], 7.0, 11.3, line="gold", lw=0.6, cls="gold", mat="gold")
                P.circle(c[0], c[1], 1.8, fill="gold", line="gold_d", lw=0.3, cls="thin", mat="gold")
        _arms_generic(P, F, R, o, sp.get("sleeve") or f, None, radii=(3.1, 2.6, 2.3), drape=spear, after=shield)


def dress_deacon(P, F, R, o, stage):
    sp = F.sp
    f = sp["robe"]
    if stage == "back":
        if R.sit:
            _throne(P, sp, R)
    elif stage == "body":
        draw_feet(P, sp, R, sp.get("shoes", "black"))
        pts = robe_outline(R, sp)
        cloth(P, pts, f, shade=0.8)
        _clavi(P, R, sp, sp.get("clavi") or "porphyry", x=4.4, w=1.5)
        lower_folds(P, R, sp, f, R.waist_y + 3.0, sp["hem_y"] - 0.5, sp["waist_w"] * 0.8, sp["hem_w"] * 0.95, n=4)
        chest_folds(P, R, f, n=2, depth=3.0, w=0.6, y_off=2.0)
    elif stage == "arms":
        _arms_generic(P, F, R, o, f, None, radii=(3.1, 2.8, 2.6), bell=0.9)


def dress_himation(P, F, R, o, stage):
    sp = F.sp
    U = R.extra["up"]
    SW, shy = R.SW, R.shoulder_y
    f = sp["robe"]
    ov = sp.get("over") or "sand_c"
    if stage == "back":
        if R.sit:
            _throne(P, sp, R)
    elif stage == "body":
        draw_feet(P, sp, R, "sandal")
        pts = robe_outline(R, sp)
        cloth(P, pts, f, shade=0.8)
        chest_folds(P, R, f, n=2, depth=2.5, w=0.5, y_off=1.0)
        ww, pw, hw = sp["waist_w"], sp["hip_w"], sp["hem_w"]
        low = -15.0 if not R.sit else -8.0
        if R.kneel:
            low = -1.0
        him = [U((SW + 2.0, shy + 1.5)), U((SW + 1.0, shy - 2.2)), U((SW - 3.4, shy - 2.8)), U((1.5, shy + 7.0)),
               U((-3.8, shy + 14.0)), U((-ww - 0.8, -58.0)), (-pw - 1.2, R.hip_y), (-hw - 0.2, low - 1.5), (-5.0, low + 1.0),
               (4.0, low + 1.4), (hw + 0.4, low - 1.0), (pw + 1.4, R.hip_y), U((SW + 2.6, shy + 14.0))]
        cloth(P, him, ov, shade=1.2)
        if P.lod < 2:
            # the rolled upper border + diagonal folds radiating from the left shoulder
            P.line([U((SW - 3.0, shy - 2.0)), U((1.5, shy + 7.5)), U((-3.8, shy + 14.5)), U((-ww - 0.5, -57.5))],
                   _c(ov, "_d"), 1.6, "fold")
            for j in range(5):
                a = U((SW - 1.0 - 0.8 * j, shy + 1.0 + 1.5 * j))
                b = (-hw + 2.5 + 3.6 * j, low - 1.5)
                P.line([a, lerp2(a, b, 0.55), b], _c(ov, "_k"), 0.45, "fold")
                if P.lod == 0:
                    P.line([(a[0] - 1.3, a[1] + 2.0), (b[0] - 1.2, b[1] - 2.0)], _c(ov, "_l"), 0.4, "light")
    elif stage == "arms":
        def cover():
            drape_from_arm(P, R, "l", ov, length=12.0, width=0.9)
        _arms_generic(P, F, R, o, f, None, radii=(3.0, 2.5, 2.2), cover=("l",), drape=cover, bare=("r",))


def dress_papal(P, F, R, o, stage):
    sp = F.sp
    U = R.extra["up"]
    SW, shy = R.SW, R.shoulder_y
    ov = sp.get("over") or "porphyry"
    if stage == "back":
        if R.sit:
            _throne(P, sp, R)
        cope = [U((-SW - 1.0, shy - 1.8)), U((SW + 1.0, shy - 1.8)), U((SW + 4.0, shy + 16.0)), (sp["hem_w"] + 3.0, -12.0),
                (sp["hem_w"] + 2.2, -6.0), (-sp["hem_w"] - 2.2, -6.0), (-sp["hem_w"] - 3.0, -12.0), U((-SW - 4.0, shy + 16.0))]
        cloth(P, cope, ov, shade=1.2)
    elif stage == "body":
        draw_feet(P, sp, R, sp.get("shoes", "red"))
        pts = robe_outline(R, sp)
        cloth(P, pts, "white", shade=0.7)
        lower_folds(P, R, sp, "white", R.waist_y + 3.0, sp["hem_y"] - 0.5, sp["waist_w"] * 0.8, sp["hem_w"] * 0.95, n=4)
        for sg in (-1, 1):
            fr = [U((sg * (SW - 2.2), shy - 2.2)), U((sg * (SW + 1.8), shy + 0.3)), U((sg * (SW + 3.0), shy + 16.0)),
                  (sg * (sp["hem_w"] + 1.6), -9.0 if not R.sit else -8.0), (sg * 5.2, -7.5 if not R.sit else -7.0),
                  U((sg * 4.0, shy + 12.0))]
            cloth(P, fr, ov, shade=0.8, smooth=False)
            P.line([U((sg * 4.4, shy + 1.0)), U((sg * 4.4, shy + 12.0)), (sg * 5.8, -8.0)], "gold", 1.8, "gold", "gold")
        a = U((0.0, shy + 3.0))
        P.poly([(a[0] - 2.6, a[1] - 1.6), (a[0] + 2.6, a[1] - 1.6), (a[0] + 2.6, a[1] + 1.6), (a[0] - 2.6, a[1] + 1.6)],
               fill="gold", line="gold_d", lw=0.35, cls="thin", mat="gold", smooth=False)
        # pallium: a plain white Y band with dark tips
        y_path = [U((-SW + 0.5, shy - 1.8)), U((-3.0, shy + 5.0)), U((0.0, shy + 8.0)), U((3.0, shy + 5.0)),
                  U((SW - 0.5, shy - 1.8))]
        band(P, y_path, 2.2, "white", lw_out=0.35, line_c="white_k")
        band(P, [U((0.0, shy + 7.0)), U((0.0, shy + 24.0))], 2.2, "white", lw_out=0.35, line_c="white_k")
        e = U((0.0, shy + 24.0))
        P.poly([(e[0] - 1.1, e[1] - 1.8), (e[0] + 1.1, e[1] - 1.8), (e[0] + 1.1, e[1] + 0.2), (e[0] - 1.1, e[1] + 0.2)],
               fill="black", smooth=False)
    elif stage == "arms":
        _arms_generic(P, F, R, o, "white", "gold", radii=(3.0, 2.7, 2.4), bell=0.6)


_DRESS = {"novice": dress_novice, "hierarch": dress_hierarch, "warrior": dress_warrior, "emperor": dress_emperor,
          "courtier": dress_courtier, "guard": dress_guard, "deacon": dress_deacon, "tunic": dress_tunic,
          "himation": dress_himation, "papal": dress_papal}


def _costume(P, F, R, o, stage):
    fn = _DRESS.get(F.sp["dress"], dress_tunic)
    fn(P, F, R, o, stage)


# ================================================================ President Jaguar's head (Justinian's crown)
_ROSETTES = ((-0.3, -0.3), (-0.14, -0.38), (0.06, -0.41), (0.25, -0.33), (0.38, -0.18), (-0.4, -0.12), (0.42, 0.03),
             (-0.43, 0.06), (-0.32, 0.26), (0.33, 0.25), (0.0, -0.26), (-0.18, -0.2), (0.19, -0.21))


def jaguar_head(P, sp, R, o, stage):
    if stage != "front":
        return
    lod = P.lod
    e = o["expr"]
    fx = _fx_fn(R)
    # ears (behind the crown)
    for sg in (-1, 1):
        ear = [(sg * 0.2, -0.42), (sg * 0.27, -0.6), (sg * 0.4, -0.66), (sg * 0.5, -0.55), (sg * 0.48, -0.34)]
        P.poly(ear, fill="fur", line="rosette", lw=0.035, cls="fine", mat=FINE, smooth=True)
        P.poly([(sg * 0.27, -0.45), (sg * 0.32, -0.58), (sg * 0.41, -0.6), (sg * 0.45, -0.5), (sg * 0.42, -0.4)],
               fill="rosette" if lod >= 1 else "fur_d", mat=FINE, smooth=True)
    out = [(0.0, 0.49), (0.17, 0.46), (0.31, 0.37), (0.43, 0.22), (0.49, 0.04), (0.5, -0.12), (0.46, -0.28), (0.37, -0.4),
           (0.2, -0.48), (0.0, -0.5), (-0.2, -0.48), (-0.37, -0.4), (-0.46, -0.28), (-0.5, -0.12), (-0.49, 0.04),
           (-0.43, 0.22), (-0.31, 0.37), (-0.17, 0.46)]
    P.poly(out, fill="fur", mat=FINE, smooth=True)
    if lod < 2:
        P.clip(out)
        P.push(-0.035, -0.02)
        P.poly(out, line="fur_d", lw=0.16, cls="thin", mat=FINE, smooth=True)
        P.pop()
        # pale mask around the eyes, cheeks and muzzle
        for sg in (-1, 1):
            P.ellipse(fx(sg * 0.18), 0.04, 0.14, 0.1, fill="fur_l", mat=FINE)
            P.ellipse(fx(sg * 0.26), 0.25, 0.16, 0.12, fill="fur_cream", mat=FINE)
        P.poly([(fx(0.0), 0.5), (fx(0.16), 0.46), (fx(0.23), 0.34), (fx(0.2), 0.22), (fx(0.08), 0.14), (fx(0.0), 0.15),
                (fx(-0.08), 0.14), (fx(-0.2), 0.22), (fx(-0.23), 0.34), (fx(-0.16), 0.46)], fill="fur_cream", mat=FINE,
               smooth=True)
        # nose bridge
        P.poly([(fx(-0.05), -0.1), (fx(0.05), -0.1), (fx(0.07), 0.14), (fx(-0.07), 0.14)], fill="fur_l", mat=FINE)
        # rosettes (rings with a warm centre) + small spots
        rr = 0.055
        for i, (x, y) in enumerate(_ROSETTES):
            if lod == 0:
                P.circle(fx(x), y, rr, fill="fur_d", line="rosette", lw=0.024, cls="fine2", mat=FINE)
            else:
                P.circle(fx(x), y, rr * 0.7, fill="rosette", mat=FINE)
        P.dots([(fx(x), y) for x, y in ((-0.07, -0.33), (0.08, -0.3), (-0.02, -0.18), (0.12, -0.12), (-0.12, -0.1),
                                        (0.28, 0.1), (-0.28, 0.1))], 0.022, "rosette", mat=FINE)
        P.unclip()
    # eyes: big almonds with green-gold irises
    lids = eye_closures(sp, R, o, e["lid"], 11)
    lx, ly = o.get("look", (0.0, 0.0)) or (0.0, 0.0)
    lx = clamp((lx + R.gaze[0]) * R.m, -1, 1)
    ly = clamp(ly + R.gaze[1], -1, 1)
    for sg in (-1, 1):
        eye(P, fx(sg * 0.175), 0.03, 0.11, sg, lids[sg], e["sq"], e["wide"], (lx, ly), None, lod, sp, lidcol="rosette",
            iris="#6F8A3A")
        # the dark tear line
        if lod < 2:
            P.line([(fx(sg * 0.08), 0.08), (fx(sg * 0.1), 0.16), (fx(sg * 0.14), 0.22)], "rosette", 0.025, "fine2", FINE)
    # brows (dark fur marks) - expression
    for sg in (-1, 1):
        r1 = e.get("brow1", 0.0) if sg > 0 else 0.0
        bi = e["bin"] + r1
        fr = e["frown"]
        pts = [(fx(sg * 0.07), -0.1 + 0.03 * fr - bi), (fx(sg * 0.17), -0.14 - 0.5 * bi), (fx(sg * 0.28), -0.1 - e["bout"] - r1)]
        P.line(pts, "rosette", 0.045, "fine", FINE)
    # nose
    P.poly([(fx(-0.1), 0.14), (fx(0.1), 0.14), (fx(0.06), 0.22), (fx(0.0), 0.255), (fx(-0.06), 0.22)], fill="nose_pk",
           line="rosette", lw=0.025, cls="fine2", mat=FINE, smooth=True)
    # mouth: Y line + smug curve; open = dark jaw with fangs
    mouth = clamp(o.get("mouth", 0.0) + e["open"], 0.0, 1.3)
    op = mouth * 0.13
    sm = e["smile"]
    sk = e.get("smirk", 0.0)
    cy = 0.33
    lc = (fx(-0.15), cy - 0.02 * sm - 0.03 * sk)
    rc = (fx(0.15), cy - 0.02 * sm + 0.005 * sk)
    if op > 0.01:
        inner = [lc, (fx(-0.08), cy + 0.02), (fx(0.0), cy - 0.01), (fx(0.08), cy + 0.02), rc, (fx(0.1), cy + 0.02 + op),
                 (fx(0.0), cy + 0.04 + op * 1.15), (fx(-0.1), cy + 0.02 + op)]
        P.poly(inner, fill="mouth_in", line="rosette", lw=0.022, cls="fine2", mat=FINE, smooth=True)
        if lod < 2:
            P.ellipse(fx(0.0), cy + 0.02 + op * 0.8, 0.05, 0.02 + op * 0.15, fill="lip_l", mat=FINE)
            for sg in (-1, 1):
                P.poly([(fx(sg * 0.055), cy + 0.012), (fx(sg * 0.1), cy + 0.012), (fx(sg * 0.075), cy + 0.012 + 0.055)],
                       fill="tooth", mat=FINE)
                if op > 0.05:
                    yb = cy + 0.035 + op * 1.05
                    P.poly([(fx(sg * 0.05), yb), (fx(sg * 0.09), yb), (fx(sg * 0.07), yb - 0.045)], fill="tooth", mat=FINE)
    P.line([(fx(0.0), 0.255), (fx(0.0), cy - 0.008)], "rosette", 0.025, "fine2", FINE, smooth=False)
    P.line([lc, (fx(-0.08), cy + 0.022), (fx(0.0), cy - 0.008), (fx(0.08), cy + 0.022), rc], "rosette", 0.028, "fine",
           FINE)
    if lod == 0:
        for sg in (-1, 1):
            P.dots([(fx(sg * 0.1), 0.28), (fx(sg * 0.14), 0.3), (fx(sg * 0.12), 0.25)], 0.012, "rosette", mat=FINE)
            for k in range(3):
                a = (fx(sg * 0.17), 0.28 + 0.025 * k)
                P.line([a, (fx(sg * (0.42 + 0.02 * k)), 0.24 + 0.06 * k)], "white_l", 0.012, "thin", FINE, smooth=False)
    P.poly(out, line="rosette", lw=0.035, cls="fine", mat=FINE, smooth=True)
    stemma(P, sp, R, lod, o.get("pend"))


def stemma(P, sp, R, lod, pend=None):
    """the imperial crown: a jewelled gold band, pearl rows, trefoil crests and the pendilia (pearl strings)"""
    y0, y1 = -0.44, -0.3
    band_pts = [(-0.4, y1 + 0.02), (-0.42, y0), (0.0, y0 - 0.03), (0.42, y0), (0.4, y1 + 0.02), (0.0, y1 - 0.01)]
    # pendilia: strings of pearls hanging past the cheeks
    for sg in (-1, 1):
        sw = 0.0
        if pend is not None:
            scr = 0 if sg * R.m < 0 else 1
            sw = float(pend[scr]) * R.m - R.head_rot
        o_ = (sg * 0.41, y1)
        for k in range(2):
            x = sg * (0.43 + 0.05 * k)
            pts = [(x + sg * 0.01 * j, y1 + 0.06 + 0.075 * j) for j in range(9 - k)]
            if sw:
                pts = [rot2(p, sw, o_) for p in pts]
            P.line([o_, pts[-1]], "gold_d", 0.012, "thin", smooth=False)
            P.dots(pts, 0.03, "pearl", mat="fine pearl")
            P.circle(pts[-1][0], pts[-1][1] + 0.05, 0.035, fill="ruby" if k == 0 else "emerald", mat=FINE)
    for i in range(5):
        x = -0.32 + 0.16 * i
        P.poly([(x - 0.06, y0 + 0.01), (x, y0 - 0.14), (x + 0.06, y0 + 0.01)], fill="gold", line="gold_d", lw=0.018,
               cls="fine2", mat="fine gold", smooth=True)
        P.circle(x, y0 - 0.15, 0.03, fill="pearl", mat="fine pearl")
    P.poly(band_pts, fill="gold", line="gold_d", lw=0.022, cls="fine2", mat="fine gold", smooth=False)
    if lod < 2:
        P.dots([(-0.36 + 0.06 * i, y0 + 0.012) for i in range(13)], 0.018, "pearl", mat="fine pearl")
        P.dots([(-0.36 + 0.06 * i, y1 - 0.006) for i in range(13)], 0.018, "pearl", mat="fine pearl")
        gems = ("ruby", "emerald", "sapphire", "emerald", "ruby")
        for i, g in enumerate(gems):
            x = -0.26 + 0.13 * i
            P.poly([(x - 0.035, (y0 + y1) / 2 - 0.035), (x + 0.035, (y0 + y1) / 2 - 0.035),
                    (x + 0.035, (y0 + y1) / 2 + 0.035), (x - 0.035, (y0 + y1) / 2 + 0.035)], fill=g, mat=FINE, smooth=False)


# ================================================================ anchors
def anchors(F, R, kw):
    sp = F.sp
    W = R.world
    HL = R.HL
    hx, hy = R.head
    fx = _fx_fn(R)

    def hp(p):
        q = rot2((p[0] * HL, p[1] * HL), R.head_rot)
        return (hx + q[0], hy + q[1])

    a = {"scale": R.u, "h": 100.0 * R.u, "kind": sp["kind"]}
    a["head"] = W(R.head)
    a["head_r"] = 0.5 * HL * R.u
    a["face"] = W(hp((fx(0.0), 0.08)))
    fw = sp["face_w"] / 0.35
    ex = 0.157 * fw if sp["kind"] != "jaguar" else 0.175
    eyes = [W(hp((fx(-ex), 0.02))), W(hp((fx(ex), 0.02)))]
    a["eyes"] = sorted(eyes)
    a["mouth"] = W(hp((fx(0.0), 0.352 + (sp["nose"] - 1.0) * 0.06 if sp["kind"] != "jaguar" else 0.34)))
    cx, cy, r = halo_geom(sp, R)
    hcw = W((cx, cy))
    a["halo"] = (hcw[0], hcw[1], r * R.u)
    top = {"tiara": -1.15, "helmet": -0.68, "tiny_crown": -0.67, "veil": -0.67, "stemma": -0.64}.get(sp.get("head"), -0.5)
    if sp["hair"]["style"] == "lutie":
        top = -0.83
    elif sp["hair"]["style"] == "wild":
        top = -0.75
    elif sp["kind"] == "jaguar":
        top = -0.66
    a["top"] = W(hp((0.0, top)))
    a["neck"] = W(R.neck)
    a["chest"] = W(R.chest)
    a["hip"] = W((0.0, R.hip_y))
    a["feet"] = (R.x, R.y)
    HN = _hand_len(sp, R)
    for side in ("r", "l"):
        a["shoulder_" + side] = W(R.sh[side])
        a["elbow_" + side] = W(R.el[side])
        a["wrist_" + side] = W(R.wr[side])
        shape = R.shape[side]
        x, y, rot, flip = _hand_frame(R, side, shape)
        if shape == "grip" and R.pole is not None:
            pc = (x, y)
            fd = rot + math.pi / 2
        else:
            fd = rot + math.pi / 2          # +y of the hand frame
            pc = (x + math.cos(fd) * 0.42 * HN, y + math.sin(fd) * 0.42 * HN)
        a["hand_" + side] = W(pc)
        a["hand_" + side + "_ang"] = R.wang(fd)
        el, wr = R.el[side], R.wr[side]
        fa = math.atan2(wr[1] - el[1], wr[0] - el[0])
        cpt = lerp2(el, wr, 0.78)
        a["cuff_" + side] = (*W(cpt), R.wang(fa))
        a["foot_" + side] = W(R.feet[side])
    a["hand_active"] = a["hand_" + R.extra.get("active", "r")]
    if R.pole is not None:
        base, (dx, dy), mode = R.pole
        b = W(base)
        tp = W((base[0] + dx * 10.0, base[1] + dy * 10.0))
        ang = math.atan2(tp[0] - b[0], b[1] - tp[1])
        a["pole"] = (b[0], b[1], ang)
        gs = sorted(R.grip.values(), key=lambda p: p[1])
        if gs:
            a["grip"] = W(gs[0])
            a["grips"] = [W(g) for g in gs]
        if mode == "candle":
            g = candle_geom(R, sp)
            if g:
                a["flame"] = W(g[1])
                a["candle"] = (W(g[0]), W(g[1]))
    if sp.get("head") == "helmet":
        g = clamp(float(kw.get("goggles", sp.get("goggles", 0.0))), 0.0, 1.0)
        a["lenses"] = [(*W(hp((x, y))), rr * HL * R.u) for x, y, rr in nvg_lenses(g)]
    if R.prop == "orb":
        side = "l" if R.shape["l"] in ("cup", "open", "hold") else "r"
        c, r_ = orb_geom(R, side, sp)
        a["orb"] = (*W(c), r_ * R.u)
    # held prop centre
    for side in ("l", "r"):
        if R.shape[side] in ("hold", "cup", "open") and R.prop:
            a["prop"] = W((R.wr[side][0], R.wr[side][1] - 4.5))
            break
    if R.shape["r"] == "stylus" or R.shape["l"] == "stylus":
        side = "r" if R.shape["r"] == "stylus" else "l"
        x, y, rot, flip = _hand_frame(R, side, "stylus")
        tip = (x + (-math.sin(rot)) * 1.15 * HN + math.cos(rot) * 0.08 * HN * flip,
               y + math.cos(rot) * 1.15 * HN + math.sin(rot) * 0.08 * HN * flip)
        a["stylus_tip"] = W(tip)
    if R.extra.get("main") in ("pinch",) or R.shape["r"] == "pinch" or R.shape["l"] == "pinch":
        side = "r" if R.shape["r"] == "pinch" else "l"
        x, y, rot, flip = _hand_frame(R, side, "pinch")
        a["pinch"] = W((x - math.sin(rot) * 0.78 * HN, y + math.cos(rot) * 0.78 * HN))
    # bounding box
    pts = [a["top"], (hcw[0] - r * R.u, hcw[1] - r * R.u), (hcw[0] + r * R.u, hcw[1] + r * R.u), a["feet"]]
    for side in ("r", "l"):
        pts += [a["hand_" + side], a["elbow_" + side]]
    hw = max(sp["hem_w"] + 3.0, R.SW + 5.0)
    pts += [W((-hw, -2.0)), W((hw, -2.0)), W((-R.SW - 4.0, R.shoulder_y)), W((R.SW + 4.0, R.shoulder_y))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    m = 0.06 * HL * R.u
    a["bbox"] = (min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)
    return a

"""Reference sheet for vx.icon: every kind in several poses/expressions, as the raw cartoon (left) and set in
tesserae through vx.mosaic (right), with real vx.flag Flags composited between the back and front layers.

    VX_FILM=opus python -m vx.icon_sheet            -> films/opus/out/icon_sheet.png (+ icon_sheet/<panel>.png)
    VX_FILM=opus python -m vx.icon_sheet faces      -> only the named panel(s)
"""
import sys
import time
from types import SimpleNamespace

import cairo
import cv2
import numpy as np

from .config import OUT
from .canvas import Canvas
from . import icon
from . import flag as vflag

W, H = 1920, 1080


def F(kind, x, y, h, seed=0, label=None, flag=None, style=None, **kw):
    if flag is not None and not isinstance(flag, dict):
        kw["flag"] = flag          # a draw keyword (e.g. the Hypermind's flag descent), not a held Flag
        flag = None
    return dict(kind=kind, x=x, y=y, h=h, seed=seed, label=label, flag=flag, style=style or {}, kw=kw)


PANELS = [
    dict(name="vexillians", title="THE VEXILLIANS - Lutie + Master Crocus", bg="gold", tile=10.0, terra=True, figs=[
        F("lutie", 250, 1010, 730, label="lutie hold_pole + Flag", pose="hold_pole", t=0.4,
          flag=dict(pole=900, wind=0.7, side=-1)),
        F("lutie", 640, 1010, 730, label="lutie look_up / awe", pose="look_up", expr="awe", t=0.9),
        F("lutie", 960, 1010, 730, label="lutie alarm / worried", pose="alarm", expr="worried", mouth=0.6, t=2.0),
        F("crocus", 1290, 1010, 760, label="crocus bless / serene", pose="bless", expr="serene", t=0.3),
        F("crocus", 1680, 1010, 760, label="crocus fend / wry", pose="fend", fend=1.0, expr="wry", t=1.2),
    ]),
    dict(name="warriors", title="THE SCHISMMANCERS - warrior saints with Flags for lances", bg="gold", tile=10.0,
         terra=True, figs=[
             F("schismmancer", 230, 1010, 760, seed=1, label="schismmancer lance, goggles up", pose="stand", t=0.2,
               goggles=0.0, flag=dict(pole=920, wind=0.8, side=-1)),
             F("schism_actual", 640, 1010, 760, seed=4, label="schism_actual command, goggles down", pose="stand",
               goggles=1.0, mouth=0.5, t=0.7),
             F("schismmancer", 1030, 1010, 760, seed=7, label="brace", pose="brace", goggles=0.5, expr="stern",
               t=1.1),
             F("schismmancer", 1480, 1010, 760, seed=12, label="charge (cloak flying) + Flag", pose="charge",
               goggles=1.0, expr="shouting", mouth=0.8, t=1.4, flag=dict(pole=560, wind=1.2, side=1)),
         ]),
    dict(name="court", title="THE EDICT OF THE JAGUAR - San Vitale", bg="gold", tile=10.0, terra=True, figs=[
        F("guard", 170, 1010, 700, seed=3, label="guard (spear + shield)", pose="stand", t=0.5),
        F("courtier", 480, 1010, 700, seed=5, label="courtier", pose="stand", t=0.9),
        F("jaguar", 850, 1010, 760, label="jaguar sly", pose="stand", expr="sly", t=0.2),
        F("deacon", 1200, 1010, 700, seed=2, label="deacon (open codex)", pose="stand", t=0.4),
        F("jaguar", 1610, 1010, 760, label="jaguar sit_throne / proud / wink", pose="sit_throne", expr="proud",
          mouth=0.5, wink=1.0, t=0.2),
    ]),
    dict(name="schism", title="NATIONS OF ONE, CABALS OF ONE, CRACKPOTS", bg="gold", tile=9.0, terra=True, figs=[
        F("citizen", 150, 1010, 640, seed=3, label="citizen declare (crowned)", pose="declare", expr="shouting",
          mouth=0.7, style=dict(sex="m")),
        F("citizen", 420, 1010, 640, seed=8, label="citizen one_finger", pose="one_finger", expr="joyful",
          style=dict(sex="f")),
        F("heretic", 690, 1010, 640, seed=11, label="heretic point", pose="point", facing=0.8, expr="angry",
          mouth=0.8, style=dict(sex="m")),
        F("heretic", 950, 1010, 640, seed=4, label="heretic point", pose="point", facing=-0.8, expr="shouting",
          mouth=0.5, style=dict(sex="f")),
        F("pope", 1210, 1010, 640, seed=6, label="pope of one", pose="bless", expr="proud"),
        F("crackpot", 1480, 1010, 640, seed=2, label="crackpot rant", pose="rant", expr="shouting", mouth=0.9,
          t=0.3),
        F("philosopher", 1760, 1010, 640, seed=5, label="philosopher sit", pose="sit", expr="serene"),
    ]),
    dict(name="faces", title="FACES - expressions, gaze, lip-sync (fine tiles)", bg="gold", tile=8.0, figs=[
        F("lutie", 250, 280, 500, label="lutie neutral", bust=True, expr="neutral", look=(0.4, 0.0), t=0.3),
        F("lutie", 730, 280, 500, label="lutie worried", bust=True, expr="worried", look=(-0.5, -0.3), t=0.3),
        F("lutie", 1210, 280, 500, label="lutie joyful, mouth .8", bust=True, expr="joyful", mouth=0.8, t=0.3),
        F("crocus", 1690, 280, 500, label="crocus serene", bust=True, expr="serene", t=0.3),
        F("crocus", 250, 800, 500, label="crocus wry, speaking", bust=True, expr="wry", mouth=0.5, t=0.3),
        F("jaguar", 730, 800, 500, label="jaguar sly", bust=True, expr="sly", t=0.3),
        F("schism_actual", 1210, 800, 500, seed=4, label="schism_actual, goggles down", bust=True, goggles=1.0,
          mouth=0.6, t=0.3),
        F("heretic", 1690, 800, 500, seed=4, label="heretic (f) awe", bust=True, expr="awe", style=dict(sex="f"),
          t=0.3),
    ]),
    dict(name="peoples", title="PEOPLES - procession, kneel, orans, crowds (LOD)", bg="gold", tile=6.0, terra=True,
         figs=[
             F("people", 130, 600, 420, seed=21, label="march", pose="march", facing=0.8, t=0.1),
             F("people", 300, 600, 420, seed=22, label="march", pose="march", facing=0.8, t=0.45),
             F("philosopher", 470, 600, 420, seed=23, label="march", pose="march", facing=0.8, t=0.8),
             F("people", 700, 600, 420, seed=24, label="kneel", pose="kneel", facing=0.0),
             F("people", 900, 600, 420, seed=25, label="orans", pose="orans"),
             F("citizen", 1110, 600, 420, seed=26, label="clasp", pose="clasp"),
             F("heretic", 1300, 600, 420, seed=27, label="raise (pole)", pose="raise"),
             F("pope", 1490, 600, 420, seed=28, label="crown_self .4", pose="crown_self", progress=0.4),
             F("lutie", 1700, 600, 420, seed=0, label="lutie candle", pose="candle", expr="serene"),
         ], crowd=dict(n=150, y0=720, y1=1060, h0=80, h1=150)),
    dict(name="machines", title="HYPERMIND (ophanim-polycandelon) + LORD EGREGORE", bg="lapis", tile=8.0, figs=[
        F("hypermind", 380, 560, 380, label="hypermind training (3 canons)", state="training", canons=3.0, t=2.0),
        F("hypermind", 1000, 560, 380, label="hypermind done (flag descending)", state="done", flag=0.25, t=3.0),
        F("egregore", 1590, 1060, 560, label="egregore", t=0.6),
    ]),
]


def _fc(s, t=0.0):
    return SimpleNamespace(s=s, w=int(round(W * s)), h=int(round(H * s)), t=t, T=t, W=W, H=H, f=0, F=0)


def _draw_figs(ctx, layer, panel, figs, part_for_flag="back", only_flagged=False):
    tile = panel["tile"]
    out = []
    if panel.get("terra") and not only_flagged:
        # the green ground line the saints stand on
        P = icon.Painter(ctx, layer, tile)
        P.poly([(0, 1000), (W, 1000), (W, H), (0, H)], fill="#2F5A3A", smooth=False)
        P.line([(0, 1000), (W, 1000)], "#1E3A26", 3.0, "contour", smooth=False)
        if layer == "color":
            rng = np.random.default_rng(7)
            for _ in range(40):
                x = rng.uniform(20, W - 20)
                y = rng.uniform(1015, 1070)
                P.circle(x, y, 4.0, fill=("#B84A3A", "#E8E1D3", "#3F7A4A")[int(rng.integers(0, 3))])
    for d, fig in figs:
        if only_flagged and not d.get("flag"):
            continue
        kw = dict(d["kw"])
        part = (part_for_flag if d.get("flag") else "all")
        if only_flagged:
            part = "front"
        if kw.pop("bust", False):
            a = fig.bust(ctx, d["x"], d["y"], d["h"], layer=layer, tile=tile, **kw)
        else:
            a = fig.draw(ctx, d["x"], d["y"], d["h"], layer=layer, tile=tile, part=part, **kw)
        out.append(a)
    return out


def _flags(ctx, figs, anchors, t):
    for (d, fig), a in zip(figs, anchors):
        fl = d.get("flag")
        if not fl or "pole" not in a:
            continue
        bx, by, ang = a["pole"]
        vflag.draw_flag(ctx, bx, by, pole=fl.get("pole", 800), t=t + d["kw"].get("t", 0.0), wind=fl.get("wind", 0.8),
                        side=fl.get("side", 1), ang=ang, seed=d.get("seed", 0), light=(-0.45, -0.8, 0.4))


def _crowd(panel):
    c = panel.get("crowd")
    if not c:
        return []
    kinds = ("people", "citizen", "heretic", "people", "philosopher", "people", "schismmancer", "pope", "crackpot",
             "courtier", "people", "deacon")
    rng = np.random.default_rng(11)
    out = []
    for i in range(c["n"]):
        y = rng.uniform(c["y0"] + 60, c["y1"])
        k = (y - c["y0"]) / (c["y1"] - c["y0"])
        h = c["h0"] + (c["h1"] - c["h0"]) * k
        out.append(dict(kind=kinds[i % len(kinds)], seed=100 + i, x=rng.uniform(40, W - 40), y=y, h=h,
                        pose=("stand", "orans", "raise", "stand", "clasp")[i % 5], facing=(0, 0.8, -0.8)[i % 3],
                        t_off=rng.uniform(0, 5)))
    return out


def render_panel(panel, s=1.0, log=print):
    from . import mosaic as mz
    t0 = time.time()
    tile = panel["tile"]
    figs = [(d, icon.Figure(d["kind"], d["seed"], **d["style"])) for d in panel["figs"]]
    crowd = _crowd(panel)
    anchors = []

    def draw(ctx, layer):
        if crowd:
            icon.crowd(ctx, crowd, t=0.5, layer=layer, tile=tile * 0.8)
        a = _draw_figs(ctx, layer, panel, figs)
        if layer == "color":
            anchors[:] = a

    timing = {}
    tc = time.time()
    c = icon.cartoon(draw, s=s, bg=panel["bg"])
    timing["cartoon"] = time.time() - tc
    bg_is_gold = panel["bg"] == "gold"
    gold = np.maximum(c["gold"], c["ground"]) if bg_is_gold else c["gold"]
    tf = time.time()
    lay = mz.Layout.flow(c["rgb"], tile=tile, seed=3, fine=c["fine"] > 0.5, fine_tile=tile * 0.5, gold=gold,
                         silver=c["silver"], pearl=c["pearl"], ground=c["ground"], edges=None)
    timing["flow"] = time.time() - tf
    fc = _fc(s)
    tr = time.time()
    mos = mz.render(fc, c["rgb"], lay, gold=gold, silver=c["silver"], pearl=c["pearl"])
    timing["render"] = time.time() - tr
    raw = c["rgb"].copy()
    # Flags: whole cloth over the tiles, then the gripping fists (tessellated as their own layer) over the poles
    if any(d.get("flag") for d, _ in figs):
        fl = Canvas(s=s)
        _flags(fl.ctx, figs, anchors, 0.0)
        fr = fl.rgba()
        mos = fr[..., :3] + mos * (1 - fr[..., 3:4])
        raw = fr[..., :3] + raw * (1 - fr[..., 3:4])

        def draw_front(ctx, layer):
            _draw_figs(ctx, layer, panel, figs, only_flagged=True)
        cf = icon.cartoon(draw_front, s=s, layers=("color", "fine"))
        if cf["alpha"].max() > 0.01:
            lay_f = mz.Layout.flow(cf["rgb"], tile=tile, seed=5, fine=cf["fine"] > 0.5, fine_tile=tile * 0.5,
                                   alpha=cf["alpha"])
            front = mz.render_layer(fc, cf["rgba"], lay_f)
            mos = front[..., :3] + mos * (1 - front[..., 3:4])
            raw = cf["rgba"][..., :3] + raw * (1 - cf["alpha"][..., None])
    log(f"[icon_sheet] {panel['name']}: {len(lay)} tiles  cartoon {timing['cartoon']:.2f}s  flow {timing['flow']:.2f}s"
        f"  render {timing['render']:.2f}s  total {time.time() - t0:.1f}s")
    return raw, np.clip(mos, 0, 1), anchors


def _label(img, panel, anchors, s):
    """small labels under the figures (sheet only, not part of the mosaic)"""
    h, w = img.shape[:2]
    cv = Canvas(s=s, w=w, h=h)
    ctx = cv.ctx
    ctx.select_font_face("Nimbus Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(24)

    def txt(x, y, t_, size=24):
        ctx.set_font_size(size)
        e = ctx.text_extents(t_)
        ctx.move_to(x - e.width / 2, y)
        ctx.text_path(t_)
        ctx.set_source_rgba(0, 0, 0, 0.85)
        ctx.set_line_width(5)
        ctx.stroke_preserve()
        ctx.set_source_rgba(1, 1, 1, 1)
        ctx.fill()

    for d, a in zip(panel["figs"], anchors):
        if not d.get("label"):
            continue
        if d["kw"].get("bust"):
            txt(d["x"], d["y"] + d["h"] * 0.5 + 14, d["label"], 22)
        elif d["kind"] == "hypermind":
            txt(d["x"], min(1068, d["y"] + d["h"] * 0.62), d["label"], 22)
        else:
            txt(d["x"], min(1068, d["y"] + 40), d["label"], 22)
    txt(W / 2, 42, panel["title"], 34)
    lab = cv.rgba()
    return lab[..., :3] + img * (1 - lab[..., 3:4])


def build(names=None, s=1.0):
    outdir = OUT / "icon_sheet"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in PANELS:
        if names and p["name"] not in names:
            continue
        raw, mos, anchors = render_panel(p, s)
        raw = _label(raw, p, anchors, s)
        mos = _label(mos, p, anchors, s)
        for tag, im in (("raw", raw), ("mosaic", mos)):
            cv2.imwrite(str(outdir / f"{p['name']}_{tag}.png"),
                        cv2.cvtColor((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        half = lambda im: cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA)
        rows.append(np.concatenate([half(raw), half(mos)], 1))
    if not rows:
        return None
    sheet = np.concatenate(rows, 0)
    head = np.full((70, sheet.shape[1], 3), 0.08, np.float32)
    cvh = Canvas(s=1.0, w=sheet.shape[1], h=70)
    ctx = cvh.ctx
    ctx.select_font_face("Nimbus Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(30)
    ctx.move_to(24, 46)
    ctx.set_source_rgb(0.95, 0.9, 0.8)
    ctx.show_text("vx.icon - Byzantine mosaic figure cartoons (left) and set in tesserae by vx.mosaic (right)")
    lab = cvh.rgba()
    head = lab[..., :3] + head * (1 - lab[..., 3:4])
    sheet = np.concatenate([head, sheet], 0)
    path = OUT / "icon_sheet.png"
    cv2.imwrite(str(path), cv2.cvtColor((np.clip(sheet, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"[icon_sheet] -> {path}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return path


if __name__ == "__main__":
    build(set(sys.argv[1:]) or None)

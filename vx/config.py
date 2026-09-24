"""Global constants. Design space is always 1920x1080 @ 24 fps.

Films: the engine (vx/) is shared; each film has its own timeline, scenes, audio, cache and renders.
Select the film with the VX_FILM environment variable (see films/<name>/env.sh):
  VX_FILM=unbabeling (default) -> legacy layout at the repo root (THE UNBABELING, film 1)
  VX_FILM=<name>               -> films/<name>/{timeline.json, script/, scenes/, audio/, cache/, out/}
"""
import os
from pathlib import Path

W, H = 1920, 1080          # design resolution (scenes always draw in these units)
FPS = 24
SR = 48000                 # audio sample rate for every stem

ROOT = Path(__file__).resolve().parents[1]
FILM = os.environ.get("VX_FILM", "unbabeling")
FILM_ROOT = ROOT if FILM == "unbabeling" else ROOT / "films" / FILM
if not FILM_ROOT.is_dir():
    raise RuntimeError(f"VX_FILM={FILM!r}: no film directory at {FILM_ROOT}")
OUT = FILM_ROOT / "out"
CACHE = FILM_ROOT / "cache"
AUDIO = FILM_ROOT / "audio"
SCRIPT = FILM_ROOT / "script"
TIMELINE = FILM_ROOT / "timeline.json"
ASSETS = ROOT / "assets"   # shared between films (fonts, soundfonts, samples)


def hex2rgb(h: str):
    h = h.lstrip("#")
    if len(h) in (3, 4):
        h = "".join(ch * 2 for ch in h[:3])
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# PALETTE. Yellow belongs ONLY to Flags (cloth), Vexillian robes (duller ochre),
# the Crystal's light and the Hypermind's gold. Everything else stays non-yellow.
# ---------------------------------------------------------------------------
PAL = {
    # the Flag (plain, unmarked). Cloth shading may only interpolate between these.
    "flag":        "#FFC41A",
    "flag_shadow": "#D99200",
    "flag_deep":   "#B87400",
    "flag_light":  "#FFE07A",
    "flag_glow":   "#FFF1B8",
    # untreated furring-lumber pole
    "pole":        "#D9B98A",
    "pole_shadow": "#A88456",
    # Vexillian robes: muted ochre so the Flags always read brighter
    "robe":        "#C9921C",
    "robe_shadow": "#8F6410",
    "robe_light":  "#E0B04A",
    # ink / void / paper
    "ink":         "#15131A",
    "night":       "#0B0E1F",
    "night2":      "#161A33",
    "dusk1":       "#1B1F3B",
    "dusk2":       "#3A2F5B",
    "dusk3":       "#8C4F63",
    "ember":       "#C86B4F",
    "paper":       "#F3EBD9",
    "bone":        "#E6DFD0",
    # the grey world of full-stack memeplexes
    "concrete":    "#6E7280",
    "concrete_d":  "#4A4E5C",
    "concrete_l":  "#9DA3B0",
    "fog":         "#A7ABB8",
    # slop
    "slop_flesh":  "#E8C8D0",
    "slop_cyan":   "#B8E0E8",
    "slop_beige":  "#C9C0A8",
    "slop_mauve":  "#8A7F8F",
    "glitch_m":    "#FF2E88",
    "glitch_c":    "#00E5FF",
    # people (flagless civilians): cool, desaturated
    "skin1":       "#E9C3A6",
    "skin2":       "#C98E6B",
    "skin3":       "#8D5A3F",
    "skin4":       "#5C3A2A",
    "cloth_grey":  "#7D8494",
    "cloth_blue":  "#46597A",
    "cloth_teal":  "#3F6E6E",
    "cloth_plum":  "#6B4A6B",
    "cloth_rust":  "#8A4B3A",
    "screen_glow": "#9FD8FF",
}
C = {k: hex2rgb(v) for k, v in PAL.items()}

FONT_SANS = "Nimbus Sans"          # Helvetica clone -> IVG / Swiss style
FONT_SERIF = "C059"                # Century Schoolbook clone -> scripture
FONT_MONO = "Nimbus Mono PS"       # Hypermind / terminals

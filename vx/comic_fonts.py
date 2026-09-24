"""Lettering faces for vx.comic (film 2).  Owner: Letterer.

`python -m vx.comic_fonts` downloads the OFL sources (Google Fonts repo), cuts static instances out of the
variable families with fontTools, renames them to unique families, installs everything into
~/.local/share/fonts (fc-cache) and keeps copies + licenses in assets/fonts/.

Families (what cairo's select_font_face sees):
  Archivo Black         Omnibus-Type, OFL — balloons/captions (the tract's Arial Black look)
  VX Archivo Italic     Archivo Italic  wght 900 wdth 100  — footnotes, italic emphasis
  VX Archivo Condensed  Archivo         wght 800 wdth 62   — the 'HOWEVER...' condensed caption, anime balloon
  VX Archivo Expanded   Archivo         wght 900 wdth 125  — the white title card
  VX Archivo Bold       Archivo         wght 700 wdth 100  — lighter body lettering (long captions)
  Anton                 OFL — impact face (signs, the Devil's FLAGS)
  VX Oswald             Oswald          wght 600           — condensed alternative
  Bangers               OFL — onomatopoeia
"""
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .config import ROOT

ASSETS = ROOT / "assets" / "fonts"
USER = Path.home() / ".local/share/fonts/vx_letterer"
GF = "https://github.com/google/fonts/raw/main/ofl/"
SRC = {
    "ArchivoBlack-Regular.ttf": "archivoblack/ArchivoBlack-Regular.ttf",
    "Anton-Regular.ttf": "anton/Anton-Regular.ttf",
    "Bangers-Regular.ttf": "bangers/Bangers-Regular.ttf",
    "Archivo[wdth,wght].ttf": "archivo/Archivo%5Bwdth,wght%5D.ttf",
    "Archivo-Italic[wdth,wght].ttf": "archivo/Archivo-Italic%5Bwdth,wght%5D.ttf",
    "Oswald[wght].ttf": "oswald/Oswald%5Bwght%5D.ttf",
}
LICENSES = {"archivoblack": "OFL-ArchivoBlack.txt", "archivo": "OFL-Archivo.txt", "anton": "OFL-Anton.txt",
            "bangers": "OFL-Bangers.txt", "oswald": "OFL-Oswald.txt"}
INSTANCES = [  # (source, family, axes)
    ("Archivo-Italic[wdth,wght].ttf", "VX Archivo Italic", {"wght": 900, "wdth": 100}),
    ("Archivo[wdth,wght].ttf", "VX Archivo Condensed", {"wght": 800, "wdth": 62}),
    ("Archivo[wdth,wght].ttf", "VX Archivo Expanded", {"wght": 900, "wdth": 125}),
    ("Archivo[wdth,wght].ttf", "VX Archivo Bold", {"wght": 700, "wdth": 100}),
    ("Oswald[wght].ttf", "VX Oswald", {"wght": 600}),
]
STATIC = ["ArchivoBlack-Regular.ttf", "Anton-Regular.ttf", "Bangers-Regular.ttf"]


def _fetch(dst, rel):
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GF + rel, dst)
    return dst


def _rename(font, family):
    name = font["name"]
    ps = family.replace(" ", "") + "-Regular"
    for rec in list(name.names):
        if rec.nameID in (16, 17, 21, 22, 25):
            name.removeNames(nameID=rec.nameID)
    for nid, val in ((1, family), (2, "Regular"), (3, ps), (4, family), (6, ps)):
        name.setName(val, nid, 3, 1, 0x409)
        name.setName(val, nid, 1, 0, 0)
    font["OS/2"].usWeightClass = 400
    font["OS/2"].fsSelection = (font["OS/2"].fsSelection & ~0b1100001) | 0b1000000   # REGULAR, not italic/bold
    font["head"].macStyle = 0
    if "STAT" in font:
        del font["STAT"]


def build():
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    src = ASSETS / "src"
    ASSETS.mkdir(parents=True, exist_ok=True)
    USER.mkdir(parents=True, exist_ok=True)
    for fn, rel in SRC.items():
        _fetch(src / fn, rel)
    for d, fn in LICENSES.items():
        _fetch(ASSETS / fn, f"{d}/OFL.txt")
    out = []
    for fn in STATIC:
        shutil.copy2(src / fn, ASSETS / fn)
        out.append(ASSETS / fn)
    for fn, family, axes in INSTANCES:
        f = instantiateVariableFont(TTFont(src / fn), axes)
        _rename(f, family)
        p = ASSETS / (family.replace(" ", "") + ".ttf")
        f.save(p)
        out.append(p)
    for p in out:
        shutil.copy2(p, USER / p.name)
    subprocess.run(["fc-cache", "-f", str(USER)], check=False)
    return out


if __name__ == "__main__":
    for p in build():
        print(p)

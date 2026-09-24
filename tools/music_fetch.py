"""Fetch the free instrument libraries used by the score (idempotent).

  python tools/music_fetch.py

- assets/sf2/GeneralUser-GS.sf2      (S. Christian Collins, free license)
- assets/sf2/MuseScore_General.sf2   (MIT)
- assets/sf2/vsco2/<subset>           Versilian Studios Chamber Orchestra 2 CE (CC0), real multisampled
                                      strings / brass / winds / percussion / organ, mapped by tools/music_samples.py
"""
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF2 = os.path.join(ROOT, "assets", "sf2")
VS = os.path.join(SF2, "vsco2")

SOUNDFONTS = {
    "GeneralUser-GS.sf2": "https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2",
    "MuseScore_General.sf2": "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2",
}

VSCO_DIRS = [
    "Strings/Violin Section/susVib", "Strings/Violin Section/Trem", "Strings/Violin Section/Pizz",
    "Strings/Violin Section/Spic",
    "Strings/Viola Section/susvib", "Strings/Viola Section/trem", "Strings/Viola Section/pizz",
    "Strings/Viola Section/spic",
    "Strings/Cello Section/susvib", "Strings/Cello Section/trem", "Strings/Cello Section/pizzT",
    "Strings/Cello Section/spic",
    "Strings/Solo Contrabass/SusVib", "Strings/Solo Contrabass/Pizz", "Strings/Solo Contrabass/Spic",
    "Strings/Solo Violin/Arco Vib", "Strings/Harp",
    "Brass/F Horn/sus", "Brass/F Horn/stac", "Brass/Trumpet/sus", "Brass/Trumpet/stac",
    "Brass/Trumpet/straightM-sus", "Brass/Tenor Trombone/sus", "Brass/Tenor Trombone/stac",
    "Brass/Tuba/sus", "Brass/Tuba/stac",
    "Woodwinds/Flute/susNV", "Woodwinds/Flute/stac", "Woodwinds/Piccolo/Stac", "Woodwinds/Piccolo/Sus",
    "Woodwinds/Oboe/Sus", "Woodwinds/Clarinet/susLong", "Woodwinds/Clarinet/stac",
    "Woodwinds/Bassoon/stac", "Woodwinds/Bassoon/sus",
    "Percussion/Timpani", "Percussion/Timpani/Rolls", "Percussion/Glock", "Percussion/Marimba",
    "Percussion/Xylo", "Keys/Organ/Loud", "Keys/Organ/Quiet", "Keys/Organ",
    "Keys/Upright Nr1",
    "Miscellania Raw/Misc 2/NepaleseBells", "Miscellania Raw/Misc 2/glock_glisses",
    "VSCO 1 Percussion/drums/other/ethnic/giant", "VSCO 1 Percussion/drums/other/ethnic/giant/mallet",
    "VSCO 1 Percussion/drums/other/ethnic/giant/sticks", "VSCO 1 Percussion/drums/other/ethnic/giant/hand",
    "VSCO 1 Percussion/drums/bass", "VSCO 1 Percussion/drums/tenor/tenor_lower",
    "VSCO 1 Percussion/drums/snare/drum3_marching", "VSCO 1 Percussion/varMetal/various",
    "VSCO 1 Percussion/varMetal/Cymbals/clash", "VSCO 1 Percussion/varMetal/Cymbals/susp",
]
# loose files directly inside Percussion/ (prefix match)
VSCO_PERC_PREFIX = ("BDrumNewhit", "cymbal-crash", "susCymb1", "gongHit", "Snare2-", "Triangle3-Hit", "TB_hit",
                    "Tamb1-", "vibraring", "BellTree", "timpsnare", "Anvil")


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    subprocess.run(["curl", "-sfL", "--retry", "3", "-o", tmp, url], check=True)
    os.replace(tmp, dst)


def main():
    for name, url in SOUNDFONTS.items():
        fetch(url, os.path.join(SF2, name))
    with urllib.request.urlopen("https://api.github.com/repos/sgossner/VSCO-2-CE/git/trees/master?recursive=1") as r:
        tree = json.load(r)["tree"]
    want = []
    for t in tree:
        if t["type"] != "blob":
            continue
        p = t["path"]
        d, f = p.rsplit("/", 1) if "/" in p else ("", p)
        if not f.lower().endswith((".wav", ".txt")):
            continue
        if d in VSCO_DIRS or (d == "Percussion" and f.startswith(VSCO_PERC_PREFIX)):
            want.append(p)
    base = "https://raw.githubusercontent.com/sgossner/VSCO-2-CE/master/"
    jobs = [(base + urllib.parse.quote(p), os.path.join(VS, p)) for p in want]
    print(f"VSCO2: {len(jobs)} files")
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(lambda a: fetch(*a), jobs))
    print("done")


if __name__ == "__main__":
    sys.exit(main())

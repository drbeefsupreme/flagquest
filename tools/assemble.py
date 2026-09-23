"""Final assembly: verify scene renders, concatenate, burn subtitles, mux the final mix.
usage: python tools/assemble.py [--scenes-dir out/scenes] [--audio audio/final_mix.wav] [--nosubs] [--fast]"""
import argparse
import json
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TL = json.loads((ROOT / "timeline.json").read_text())


def ts(t):
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def split2(s, width=58):
    if len(s) <= width:
        return s
    best, bi = 1e9, None
    for i, ch in enumerate(s):
        if ch == " ":
            score = abs(i - len(s) / 2) - (16 if s[i - 1] in ",.!?:" else 0)
            if score < best:
                best, bi = score, i
    return s[:bi] + r"\N" + s[bi + 1:] if bi else "\\N".join(textwrap.wrap(s, width))


def write_ass(path):
    head = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Nimbus Sans,44,&H00EEF2F4,&H000000FF,&H00141214,&H96000000,0,0,0,0,100,100,0.4,0,1,2.4,1.4,2,200,200,62,1
Style: Lyric,C059,50,&H00D8F4FF,&H000000FF,&H00141214,&H96000000,0,1,0,0,100,100,1.0,0,1,2.2,1.4,2,200,200,62,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    for l in TL["lines"]:
        text = l["text"].replace("twenty forty-two", "2042")
        words = l["words"]
        if len(text) <= 64 or not words:
            ev.append((l["start"], l["end"] + 0.25, "Default", split2(text)))
            continue
        # break long lines into sentence chunks timed from the TTS word timestamps
        chunks, cur, t0, t1 = [], "", None, None
        for w in words:
            tok = w["w"].replace("twenty", "20").replace("forty-two", "42")
            if any(ch.isalnum() for ch in tok):
                cur = (cur + " " + tok) if cur and not cur.endswith("20") else (cur + tok)
                t0 = w["s"] if t0 is None else t0
                t1 = w["e"]
            else:
                cur += tok
                if tok in ".!?" and len(cur) > 24:
                    chunks.append([cur.strip(), t0, t1])
                    cur, t0 = "", None
        if cur.strip():
            chunks.append([cur.strip(), t0, t1])
        for k, (s, a0, a1) in enumerate(chunks):
            end = chunks[k + 1][1] - 0.02 if k + 1 < len(chunks) else l["end"] + 0.25
            ev.append((a0, end, "Default", split2(s)))
    h = TL["hymn"]
    beat = 60.0 / h["bpm"]
    lyr = [("Flags be.", 0, 4), ("Flags are.", 4, 8), ("Flags will.", 8, 12), ("Flags.", 12, 16)]
    for s, b0, b1 in lyr:
        ev.append((h["start"] + b0 * beat, h["start"] + b1 * beat + (0.6 if s == "Flags." else 0.0), "Lyric", "~ " + s + " ~"))
    g = next(c["t"] for c in TL["cues"] if c["id"] == "glorious")
    ev.append((g, g + 1.1, "Default", "GLORIOUS!"))
    ev.sort()
    body = "".join(f"Dialogue: 0,{ts(a)},{ts(b)},{st},,0,0,0,,{{\\fad(90,160)}}{tx}\n" for a, b, st, tx in ev)
    path.write_text(head + body)


def nframes(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets,width,height,r_frame_rate", "-of", "json", str(p)],
                       capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    return int(s["nb_read_packets"]), s["width"], s["height"], s["r_frame_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-dir", default=str(ROOT / "out/scenes"))
    ap.add_argument("--audio", default=str(ROOT / "audio/final_mix.wav"))
    ap.add_argument("--out", default=str(ROOT / "out/THE_UNBABELING.mp4"))
    ap.add_argument("--nosubs", action="store_true")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    sd = Path(a.scenes_dir)
    parts, ok = [], True
    for sc in TL["scenes"]:
        p = sd / f"{sc['id']}.mp4"
        want = sc["f1"] - sc["f0"]
        if not p.exists():
            print(f"MISSING {p}")
            ok = False
            continue
        n, w, h, r = nframes(p)
        flag = "" if (n == want and (w, h) == (1920, 1080) and r == "24/1") else "  <-- MISMATCH"
        ok &= not flag
        print(f"{sc['id']:5s} {n:5d}/{want:5d} frames {w}x{h} @{r}{flag}")
        parts.append(p)
    if not ok:
        raise SystemExit("fix scene renders first")
    lst = ROOT / "out/.concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    video = ROOT / "out/film_video.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(video)], check=True)
    ass = ROOT / "out/subs.ass"
    write_ass(ass)
    (ROOT / "out/subs_path.txt").write_text(str(ass))
    preset = ["-preset", "veryfast", "-crf", "20"] if a.fast else ["-preset", "slow", "-crf", "16", "-tune", "animation"]
    outs = [(Path(a.out), True)]
    if a.nosubs:
        outs.append((Path(a.out).with_name(Path(a.out).stem + "_nosubs.mp4"), False))
    for out, subs in outs:
        vf = ["-vf", f"ass={ass}"] if subs else []
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats", "-i", str(video), "-i", a.audio, "-map", "0:v:0", "-map",
               "1:a:0", *vf, "-c:v", "libx264", *preset, "-pix_fmt", "yuv420p", "-profile:v", "high",
               "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-movflags", "+faststart", "-shortest", str(out)]
        subprocess.run(cmd, check=True)
        print("wrote", out)


if __name__ == "__main__":
    main()

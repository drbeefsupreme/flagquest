# flagquest

Two short animated films about the Vexillians and their Survey Flags, made entirely by code:

| Film | Runtime | Style |
|---|---|---|
| **THE UNBABELING** | 3:13 (193 s) | luminous flat-vector animation |
| **THE FLAG GAME?** | 3:19 (199 s), loops seamlessly | a Chick-tract parody in black ink on white paper |

Both are 1920×1080 at 24 fps. No image or video generator was used: every frame is drawn by Python, every flag is a physically simulated cloth, and the voices, singing, score and sound effects are produced by the scripts in this repo.

Source lore: [wiki.vexillomancy.org](https://wiki.vexillomancy.org/wiki/Vexillians) (Vexillians, Survey Flags, Vexillomantic holy texts) and [Flagazine #69](https://vexillomancy.org/archives/flagazine/69/flagazine.pdf).

## The One Rule

**Flags are plain yellow with no other markings upon them.** Every flag is drawn only through `vx/flag.py`, which can output nothing but the palette yellows. Nothing may be drawn over the cloth. In film 2 the flags are the only colour in the film, laid over the print as an unscreened "spot yellow plate". `tools/qa_onerule.py` scans rendered scenes for anything enclosed inside flag cloth.

## The films

**THE UNBABELING.** A 4D crystal rotates the first Flag into existence. A tsunami of AI slop splinters humanity into one feed per face. President Jaguar exempts the yellow memes from the Noospheric Munitions Act. The Hypermind of Flagartha compiles a hypermeme, and it turns out to be a plain Flag. The schismmancers recursively balkanize every nation, faith and philosophy, until a new Babel collapses into silence. At dawn a single Flag is planted, and thousands rise across the plain and sing "Flags be. Flags are. Flags will. Flags." over a hyperbolic Flagistan. It ends with the sticky-flag koan and "Glorious!".

**THE FLAG GAME?** A tract lies face-down in the mud at a rainy Burn. The wind flips it open and the comic comes alive: Summer confesses she hates "that FLAG GAME", and Dr. Beelzebub Crow tells her every word was WRONG!! His flashback plays as an animated Doré engraving:
- the six primordial Flags;
- the hippies' stench becoming a hurricane;
- the Flag Maker, and the Devil who can only say "FLAGS";
- every raindrop freezing as the Flag props up the firmament;
- the effigy burn and the deluge;
- 100 re-raisings of the Flag in 2.8 seconds.

A lorem-ipsum altar call converts Summer. She is handed the tract, which recursively contains itself in an Escher Print-Gallery spiral. The tract lands face-down in the mud: the last frame is the first frame.

## Repository layout

```
vx/                  shared animation engine (both films)
  config.py          design space, palette, fonts, per-film paths (VX_FILM)
  canvas.py camera.py ease.py timeline.py post.py render.py foley.py
  flag*.py           cloth physics (numba XPBD + aerodynamics) and the plain-yellow renderer
  chars*.py          2D character rig: poses, expressions, lip-sync, ink mode for film 2
  ink*.py            print looks: halftone, Doré engraving, manga screentone, wet paper
  comic*.py          voice-synced lettering: balloons, captions, SFX, fonts
scenes/ script/ docs/ timeline.json    film 1 (repo root, the default VX_FILM=unbabeling)
films/flaggame/      film 2: scenes/ script/ docs/ tools/ ref/ timeline.json env.sh
tools/               shared build tools + film 1's audio department scripts
assets/              lettering fonts (OFL), land mask; assets/sf2/ is downloaded, not tracked
ref/                 Flagazine #69 reference text and contact sheet (the PDF is not tracked)
```

Each film has a production bible and shot list (`docs/BIBLE.md`, `docs/SHOTS.md`, and the same under `films/flaggame/docs/`). The locked master timeline is `timeline.json`, built from `script/plan.json` and `script/lines.json` by `tools/build_timeline.py`. It holds scenes, dialogue with per-word timestamps, and named sync cues.

## Setup

- **Linux** with `ffmpeg` (libx264 and libass), cairo development headers, and the `fonts-urw-base35` and Noto fonts.
- **EGL** for film 2's first scene, which uses moderngl in a headless context.
- **Python 3.12** in a `uv` venv at `.venv`.

```sh
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install numpy scipy numba pycairo opencv-python-headless pillow soundfile shapely opensimplex \
    librosa pedalboard pyloudnorm praat-parselmouth mido pretty_midi moderngl faster-whisper \
    "kokoro>=0.9.4" "misaki[en]"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --no-deps tinysoundfont
```

A few extra packages were added during production (e.g. `uharfbuzz`, `pymunk`, `fonttools`, `pyworld`). Install them if a script reports an ImportError.

One-time downloads:

```sh
python -m vx.comic_fonts                              # lettering fonts -> ~/.local/share/fonts
python tools/music_fetch.py && python tools/music_samples.py   # score samples (~1.9 GB) -> assets/sf2/
```

The Kokoro-82M voice model and the Whisper `small.en` model download automatically on first use.

## Building a film

Voice WAVs, caches and renders are not tracked in git; the scripts regenerate them.

**Film 1: THE UNBABELING** (run from the repo root):

```sh
source .venv/bin/activate
python tools/tts.py && python tools/build_timeline.py        # voices, lip-sync envelopes, dry dialogue
python -m vx.render s01 --workers 4                           # repeat for s02 s03 s04 s05a s05b s06 s07a s07b s08
python tools/vocal_dialogue.py && python tools/vocal_crowd.py && python tools/vocal_choir.py
python tools/music_score.py && python tools/foley_build.py
python tools/mix.py && python tools/assemble.py --nosubs --share
```

Output goes to `out/THE_UNBABELING{,_nosubs,_web}.mp4`. The main version has burned-in subtitles.

**Film 2: THE FLAG GAME?**

```sh
source films/flaggame/env.sh                                  # activates .venv, sets VX_FILM=flaggame
python -m vx.render t01 --workers 4                           # repeat for t02 ... t09, in order
python films/flaggame/tools/fg_vocal_dialogue.py && python films/flaggame/tools/fg_vocal_crowd.py
python films/flaggame/tools/fg_vocal_choir.py && python films/flaggame/tools/fg_music_score.py
python films/flaggame/tools/fg_foley_build.py
python tools/mix.py && python tools/assemble.py --share
python films/flaggame/tools/print_tract.py                    # bonus: the film as a 24-page printable tract
```

Output goes to `films/flaggame/out/THE_FLAG_GAME{,_web}.mp4` and `films/flaggame/out/print/`. The print output is a reader PDF, a saddle-stitch imposed PDF, and black and yellow separation plates. Every line is lettered on screen, so there are no burned-in subtitles.

**Render order and previews.**
- Render scenes before building audio: the music and foley scripts read hit events and flag-motion tracks that each scene's setup exports.
- Render scenes in order, because some read caches written by earlier scenes.
- For quick looks, `python -m vx.render <scene> --at 1.0,4.5` writes stills and a contact sheet, and `--scale 0.5 --step 2` writes a half-resolution preview.

**QA tools.**
- `tools/qa_dialogue.py` transcribes the final mix with Whisper and reports word error rate per line.
- `tools/qa_onerule.py` scans renders for markings on flags.
- `tools/review.py` builds contact sheets.
- `tools/check_cuts.py` compares every scene hand-off.

## How it was made

Claude, running in the omp agent harness, acted as director. For each film it wrote the script, recorded and checked the voices, locked the timeline, and ran 15 parallel sub-agents that coordinated over a message bus: one per scene, the shared libraries (cloth, rig, print, lettering), and the voice, music and foley departments.
- **Picture:** pycairo, NumPy/numba, OpenCV and moderngl, encoded with ffmpeg.
- **Voices:** Kokoro text-to-speech with audio-driven lip-sync. The choir is the same model with its pitch and note lengths overridden, finished with Praat and WORLD vocoder tools.
- **Score:** VSCO2 Community Edition orchestral samples (CC0), plus the GeneralUser GS and MuseScore General SoundFonts.
- **Sound effects:** procedural NumPy synthesis. Each flag's flutter is computed from its own cloth simulation.
- **Mix:** pedalboard and pyloudnorm, mastered to −14 LUFS.

Each film took about two hours on one workstation.

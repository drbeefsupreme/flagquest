# flagquest

Three short animated films about the Vexillians and their Survey Flags, made entirely by code:

| Film | Runtime | Style |
|---|---|---|
| **THE UNBABELING** | 3:13 (193 s) | luminous flat-vector animation |
| **THE FLAG GAME?** | 3:19 (199 s), loops seamlessly | a Chick-tract parody in black ink on white paper |
| **OPVS SCHISMATICVM** | 3:32 (212 s) | film 1 remade as living Byzantine glass-and-gold mosaics |

All are 1920×1080 at 24 fps. No image or video generator was used: every frame is drawn by Python, every flag is a physically simulated cloth, and the voices, singing, score and sound effects are produced by the scripts in this repo.

Source lore: [wiki.vexillomancy.org](https://wiki.vexillomancy.org/wiki/Vexillians) (Vexillians, Survey Flags, Vexillomantic holy texts) and [Flagazine #69](https://vexillomancy.org/archives/flagazine/69/flagazine.pdf).

## The One Rule

**Flags are plain yellow with no other markings upon them.** Every flag is drawn only through `vx/flag.py`, which can output nothing but the palette yellows, always fully opaque. Nothing may be drawn over the cloth, film grain included. In film 2 the flags are the only colour in the film, laid over the print as an unscreened "spot yellow plate". In film 3 the flags are the only things that are not made of tesserae. `tools/qa_onerule.py` scans rendered scenes for anything enclosed inside flag cloth (`--smooth` for film 3, where gold tesserae are yellow too).

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

**OPVS SCHISMATICVM.** Film 1 retold as the mosaic programme of a candlelit basilica, in the manner of Ravenna, San Vitale, Monreale and San Marco. Everything is set in tesserae except the Flag, the one seamless thing. A cantor intones INCIPIT, and a Tesseract (every tessera is one of its shadows) turns in a starry vault until the Flag rotates out of it. HVMANITAS, a face made of 2,600 tiny people, dissolves into slop phones. President Jaguar is Justinian in the San Vitale panel. The Hypermind is an ophanim of gold wheels full of eyes, and the Master and novice argue in the form of the *Summa*. The schismmancers are warrior saints under IN HOC SIGNO SCINDES, and each nation, church and philosophy splits into tiles of one. Babel is the whole mosaic falling off the wall into silence and bare plaster with its red sinopia. The fallen tiles rise and set themselves again (SINGVLI ET VNIVERSI), and the hymn is an antiphon, Latin plainchant answered in English, under a hyperbolic {8,3} Pentecost dome. It ends with the sticky-flag koan, "Glorious!", EXPLICIT and the sung *Vexillo gratias*.

## Repository layout

```
vx/                  shared animation engine (all films)
  config.py          design space, palette, fonts, per-film paths (VX_FILM)
  canvas.py camera.py ease.py timeline.py post.py render.py foley.py
  flag*.py           cloth physics (numba XPBD + aerodynamics) and the plain-yellow renderer
  chars*.py          2D character rig: poses, expressions, lip-sync, ink mode for film 2
  ink*.py            print looks: halftone, Doré engraving, manga screentone, wet paper
  comic*.py          voice-synced lettering: balloons, captions, SFX, fonts
  mosaic*.py         film 3's mosaic engine: andamento layouts, smalti/gold shading, walls/domes, tituli, tile physics, sinopia
  icon*.py           film 3's Byzantine figure cartoons (every character, lip-sync, fine/gold masks for the tesserae)
scenes/ script/ docs/ timeline.json    film 1 (repo root, the default VX_FILM=unbabeling)
films/flaggame/      film 2: scenes/ script/ docs/ tools/ ref/ timeline.json
films/opus/          film 3: scenes/ script/ docs/ tools/ timeline.json
tools/               shared build tools + film 1's audio department scripts
assets/              lettering fonts (OFL), land mask; assets/sf2/ is downloaded, not tracked
ref/                 Flagazine #69 reference text and contact sheet (the PDF is not tracked)
```

Each film has a production bible and shot list (`docs/BIBLE.md`, `docs/SHOTS.md`, and the same under `films/flaggame/docs/` and `films/opus/docs/`). The locked master timeline is `timeline.json`, built from `script/plan.json` and `script/lines.json` by `tools/build_timeline.py`. It holds scenes, dialogue with per-word timestamps, and named sync cues.

## Setup

- **Linux** with `ffmpeg` (libx264 and libass), cairo development headers, and the `fonts-urw-base35` and Noto fonts.
- **EGL** for film 2's first scene, which uses moderngl in a headless context.
- **Python 3.12** in a `uv` venv at `.venv`.

```sh
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install numpy scipy numba pycairo opencv-python-headless pillow soundfile shapely opensimplex \
    librosa pedalboard pyloudnorm praat-parselmouth mido pretty_midi moderngl faster-whisper \
    "kokoro>=0.9.4" "misaki[en,ja,zh]"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --no-deps tinysoundfont
```

A few extra packages were added during production (e.g. `uharfbuzz`, `pymunk`, `fonttools`, `pyworld`). Install them if a script reports an ImportError. Film 3's Latin cantor and multilingual Babel use Kokoro's non-English voices; their espeak-ng phonemizer ships inside the venv (`espeakng-loader`).

One-time downloads:

```sh
python -m vx.comic_fonts                              # lettering fonts -> ~/.local/share/fonts
python tools/music_fetch.py && python tools/music_samples.py   # score samples (~1.9 GB) -> assets/sf2/
python -m vx.mosaic_type                              # film 3 titulus fonts (cut from OFL Cinzel) -> assets/fonts/
```

The Kokoro-82M voice model and the Whisper `small.en` model download automatically on first use.

## Building a film

Voice WAVs, caches and renders are not tracked in git. Run everything from the repo root. These are the commands the films were built with on the original machine; they have not been tested from a fresh clone.

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
source .venv/bin/activate
export VX_FILM=flaggame                                       # selects films/flaggame (the default is film 1)
python -m vx.render t01 --workers 4                           # repeat for t02 ... t09, in order
python films/flaggame/tools/fg_vocal_dialogue.py && python films/flaggame/tools/fg_vocal_crowd.py
python films/flaggame/tools/fg_vocal_choir.py && python films/flaggame/tools/fg_music_score.py
python films/flaggame/tools/fg_foley_build.py
python tools/mix.py && python tools/assemble.py --share
python films/flaggame/tools/print_tract.py                    # bonus: the film as a 24-page printable tract
```

Output goes to `films/flaggame/out/THE_FLAG_GAME{,_web}.mp4` and `films/flaggame/out/print/`. The print output is a reader PDF, a saddle-stitch imposed PDF, and black and yellow separation plates. Every line is lettered on screen, so there are no burned-in subtitles.

**Film 3: OPVS SCHISMATICVM**

```sh
source .venv/bin/activate
export VX_FILM=opus                                           # selects films/opus
python tools/tts.py && python tools/build_timeline.py
python films/opus/tools/op_vocal_dialogue.py && python films/opus/tools/op_vocal_crowd.py
python films/opus/tools/op_vocal_choir.py                     # also writes the cantor syllable times the scenes sync to
python -m vx.render m01 --workers 4                           # repeat for m02 ... m10, in order
python films/opus/tools/op_music_score.py && python films/opus/tools/op_foley_build.py
python tools/mix.py && python tools/assemble.py --share
```

Output goes to `films/opus/out/OPVS_SCHISMATICVM{,_web}.mp4`. The on-screen text is Latin tituli set in the mosaic, so there are no burned-in subtitles (`--subs` adds English ones). Reference sheets for the two mosaic libraries: `python -m vx.mosaic_sheet` and `python -m vx.icon_sheet`.

**Render order and previews.**
- Render scenes before building the score and foley: those scripts read hit events and flag-motion tracks that each scene's setup exports. Film 3 builds its voices first, because its scenes sync letters and light to the sung syllables.
- Render scenes in order, because some read caches written by earlier scenes (film 3: m07 continues m06's dust, m08 starts on m07's fallen tiles).
- For quick looks, `python -m vx.render <scene> --at 1.0,4.5` writes stills and a contact sheet, and `--scale 0.5 --step 2` writes a half-resolution preview.

**QA tools.**
- `tools/qa_dialogue.py` transcribes the final mix with Whisper and reports word error rate per line.
- `tools/qa_onerule.py` scans renders for markings on flags (`--smooth` for film 3).
- `tools/review.py` builds contact sheets.
- `tools/check_cuts.py` compares every scene hand-off.

## How it was made

Claude, running in the omp agent harness, acted as director. For each film it wrote the script, recorded and checked the voices, locked the timeline, and ran 15 parallel sub-agents that coordinated over a message bus: one per scene, the shared libraries (cloth, rig, print, lettering), and the voice, music and foley departments.
- **Picture:** pycairo, NumPy/numba, OpenCV and moderngl, encoded with ffmpeg.
- **Voices:** Kokoro text-to-speech with audio-driven lip-sync. The choir is the same model with its pitch and note lengths overridden, finished with Praat and WORLD vocoder tools.
- **Score:** VSCO2 Community Edition orchestral samples (CC0), plus the GeneralUser GS and MuseScore General SoundFonts.
- **Sound effects:** procedural NumPy synthesis. Each flag's flutter is computed from its own cloth simulation.
- **Mix:** pedalboard and pyloudnorm, mastered to −14 LUFS.

Each film took about two to three hours on one workstation.

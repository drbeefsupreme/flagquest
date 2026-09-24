# THE FLAG GAME? — Production Bible (film 2)

A 3:19 (199.0 s) animated Chick-tract parody about the Vexillians and their Survey Flags, adapted from
the tract in Flagazine #69 (pages 23–27; scans in `films/flaggame/ref/page_23..27.png` — LOOK AT THEM).
Everything is made by code in this repo, like film 1 (THE UNBABELING, which the client loved). The brief:
**"BLOW ME AWAY — demonstrate capabilities not even thought of as possible."** Coherent story, real acting,
dialogue, sound. Festival-grade art direction; not a tech demo.

---------------------------------------------------------------------------------------------------
## 0. THE ONE RULE (absolute, zero exceptions)

**FLAGS ARE PLAIN YELLOW WITH NO OTHER MARKINGS UPON THEM.**

- Every flag is a plain yellow rectangle of cloth on a plain untreated wooden pole, drawn ONLY with
  `vx.flag` (`FlagTrack.draw`, `draw_flag`, `draw_flag_field`, `draw_furled`, `draw_cloth`).
- The only variation allowed on the cloth is smooth lighting between the palette yellows
  (`flag_deep` → `flag_shadow` → `flag` → `flag_light` → `flag_glow`). No outline, border, text, symbol,
  stripe, pattern, emblem, number, stain, char, mud, rain streak, shadow of an object, halftone dot,
  hatch line, paper texture, grain, speech balloon, particle — NOTHING may ever be drawn over the cloth.
- **This film is black ink on white paper. The Flags are the ONLY colour in the whole film.** They are
  never printed: they are laid on top of the print as the "spot yellow plate" (`vx.ink.spot`) — real,
  smooth, alive, physically simulated. That is the thesis: *the Flags are THE LIVING ESSENCE*.
- Only a clearly separate solid foreground object may pass in front of a flag (a hand gripping the pole,
  a person walking in front). Rain, sparks, embers, fire, smoke, water, text, balloons, emanata, dust:
  never over the cloth (mask them out of the cloth silhouette).
- A flag never burns, chars, gets muddy or tears. When fire reaches it, it lifts free and ascends.
- The source comic says "each was a mix of Red, Blue and Yellow". We never show that: the six primordial
  flags are plain yellow, and the script turns it into a joke ("They look yellow to me." / "Red and Blue
  are merely the ABSENCE of Yellow. II VEXILLIANS 5").
- No other object may *become* a flag with markings (no national/religious banners, no emoji flags;
  Iwo-Jima/Liberty-style compositions in the montage use plain yellow flags).
- Yellow is sacred: nothing else in the film is yellow or warm-coloured. Paper is neutral white, ink is
  neutral black. The only allowed non-grey besides flag yellow is a flag's own soft yellow glow/bloom.

---------------------------------------------------------------------------------------------------
## 1. Story

A Chick tract comes to life. The film is a LOOP: its last frame is its first frame.

**Present day, a rainy regional Burn.** Summer, a sunny blonde hippie, and her friend Raven watch a flag-
bearer pass. Summer cheerfully confesses (Chick-style) that she hates "that FLAG GAME" because she is a
"LAZY and ENTITLED HIPPY". The flag-bearer — Dr. Beelzebub Crow, playa evangelist — is suddenly beside
them: "Amazing, every word of what you just said... WAS WRONG!!" The Flags are not a game; they are the
LIVING ESSENCE OF THE BURN ITSELF (footnote read at pharma-ad speed: "Survey FLAGS is strictly an
interactive art installation"). He tells the story FROM THE BEGINNING — back at Alchemy IX...

**The flashback (Gustave-Doré engraving).** Six Flags on an op-art hexagon field ("They look yellow to
me."). The STENCH of the hippies rises and twists into a hurricane; the forecast says RAIN. ALL. WEEKEND.
(a Ranger on the radio: "does anyone know where my dirt went?"). The FLAG MAKER walks amongst them: by
FINDING and MOVING the FLAGS the rain can be stopped. Most rejected Him ("Shut up!"). He explains that each
Flag signifies its own referent — but all they hear is FLAGS FLAGS FLAGS FLAGS; even the Devil, trying to
call them "just a g—", can only say "FLAGS!". BUT THERE WERE SOME THAT LISTENED... They find and move the
Flags. AND IT WORKED: Saturday afternoon the ALL YELLOW FLAG is set atop the effigy by a masked Seraphlag —
every raindrop freezes midair — and the Flag is revealed to be propping up the FIRMAMENT itself, a
crystal sky-dome holding back the waters above. That evening the effigy burns FOR ALL TO SEE; the very
second the flames reach the Flag, a miracle: the Flag lifts free like an ember — and without it to hold it
back, THE FIRMAMENT OPENS and THE RAIN RETURNS (a Doré deluge). HOWEVER the Flag is RAISED AGAIN... 50, then
100 time! — across every age and art style. The Spirit of the Flags follows the Burn through a Burn with
no fire and the years without Burns: a lone masked flag-bearer walking through the seasons... who lowers
his mask: it is Crow. We are back in the present.

**Present.** "Never forget: FLAGS are the HEART and SOUL of the Burn itself, and speaking against them is
BLASPHEME." Thunder. Summer, on her knees: "What must I do?" — "Repeat after me." They pray, in perfect
sincerity, the lorem-ipsum prayer while an altar-call choir sings "Just as I am, without one Flag". Amen.
The rain stops. Crow hands her the tract — "Now go... and PASS IT ON." We dive into the tract she holds:
its page shows her holding the tract, which shows her holding the tract... an Escher Print-Gallery spiral.
It resolves on the next day: Summer walks past carrying a Flag. Raven: "Ugh, there goes another one of
those YELLOW FLAGS." She tosses the tract into the mud, face down — which is the first frame of the film.

### Characters (all original designs; do NOT copy the likeness of any real person in the reference photo-collage)
- **Summer** — early 20s, sunny, open face, freckles, blonde hair in two pigtails with bangs (fine
  romance-comic strands), faux-fur-collared jacket, goggles round her neck. Cheerful even while confessing.
  Converts. Voice `SUMMER`.
- **Raven** — Summer's friend: long glossy black hair (solid ink with white highlight streaks), big hoop
  earrings, dark lipstick, half-lidded disdain, black clothes; lounges in a camp chair. Voice `RAVEN`.
- **Dr. Beelzebub Crow** — playa evangelist, 50s: trucker cap covered in pins and patches, tinted aviator
  glasses, grizzled goatee, fur-collared long coat, a sheaf of tracts, carries a Flag. Pointing finger,
  fervent, Chick-tract witness. Tells the whole flashback (V.O.). Voice `PREACHER`.
- **The Flag Maker** — tall, robed, long hair and beard, radiant (engraved light rays), patient; a
  carpenter's apron with the DIY flag tools (hammer, carpet tacks, sandpaper, scissors, pencil — Flagazine's
  "FLAG DIY ASSEMBLY"); walks among seven tall candlesticks. Voice `FLAGMAKER`.
- **Hippies** — muddy burners in the rain: dreadlocks, fur coats, tutus, face paint, goggles; clothes
  blotched with mud spots (the comic's spotted crowd); stink lines rising. Voices `HIPPIE1/2` + crowd.
- **The Pharisees** — hooded elders in rain ponchos, stern, bearded (the comic's "Shut up!" panel).
  Voice `PHARISEES` (a mob).
- **The Devil** — Chick-tract Satan: horns, cat-grin, slit pupils, on a throne, holding a lyre.
  Voice `DEVIL`.
- **The Seraphlag** — a vast feathered-winged angel with a halo and a surgical face mask (the comic's masked
  angel) who carries the ALL YELLOW FLAG to the top of the effigy.
- **The Effigy** — a 40-foot wooden figure of lattice lumber, arms raised; burned Saturday night.
- **The Firmament** — the ancient sky-dome (raqia): a colossal crystal vault over the Burn with the
  "waters above" pressing on it; the Flag's pole props it up like a tent pole.

### Dialogue (LOCKED — `films/flaggame/timeline.json`; audio `films/flaggame/audio/voice/raw/<ID>.wav`)
Display text (what is LETTERED, typos and CAPS emphasis are the tract's own — keep them) → speaker, start (s).

| ID | T | Speaker | Lettered text |
|---|---:|---|---|
| R01 | 7.40 | RAVEN | Ugh, there goes another one of those YELLOW FLAGS. |
| S01 | 11.60 | SUMMER | I hate that FLAG GAME because I am a LAZY and ENTITLED HIPPY |
| S02 | 15.58 | SUMMER | It just doesn't FIT IN with how I think a Burn works |
| P01 | 18.74 | PREACHER | Amazing, every word of what you just said... |
| P02 | 21.91 | PREACHER | ...WAS WRONG!!  (footnote *1ST VEXILLIANS 11s) |
| S03 | 23.22 | SUMMER | WHAT? HOW CAN THIS BE??? |
| P03 | 25.20 | PREACHER | For ONE THING, the FLAGS are not a GAME. |
| P04 | 27.83 | PREACHER | The FLAGS are THE LIVING ESSENCE OF THE BURN ITSELF!!! |
| D01 | 31.06 | DISCLAIMER | Survey FLAGS is strictly an interactive art installation (the ** footnote) |
| S04 | 33.75 | SUMMER | Huh? What you are saying is... its so weird.... (anime) |
| P05 | 36.75 | PREACHER | Here, why don't I tell you the story FROM THE BEGINNING |
| P06 | 40.03 | PREACHER | It all started back at Alchemy IX... |
| P07 | 44.29 | PREACHER | In the beginning there were 6 FLAGS, and each was a mix of Red, Blue and Yellow. |
| S05 | 49.42 | SUMMER | They look yellow to me. (off-panel, from the present) |
| P08 | 50.77 | PREACHER | Red and Blue are merely the ABSENCE of Yellow. II VEXILLIANS 5 |
| P09 | 56.00 | PREACHER | That year, the terrible STENCH of the hippies caused a great hurricane to decend on the burn. |
| F01 | 61.65 | FORECAST | ...and the forcast for Alchemy: RAIN. ALL. WEEKEND. (radio) |
| G01 | 64.63 | RANGER | Hey, over, uhmmm... does anyone know where my dirt went? (radio) |
| P10 | 67.75 | PREACHER | The FLAG MAKER walked amongst them... |
| M01 | 70.10 | FLAGMAKER | By FINDING and MOVING the FLAGS, the rain can be stopped. |
| P11 | 74.73 | PREACHER | Most people rejected His message. |
| X01 | 76.98 | PHARISEES | Shut up! |
| P12 | 78.02 | PREACHER | They hated FLAGS because He told them the truth. FLAGS 4:20 |
| M02 | 82.74 | FLAGMAKER | Each FLAG is a symbol that signifies its own referent. Therefore sign and signifier become are made one in the FLAGS. |
| M03 | 92.09 | FLAGMAKER | Each FLAG is itself a FLAG... and each FLAG is every FLAG. |
| V01 | 97.30 | DEVIL | Listen, my children. The Flags are nothing but a... a... a... |
| V02 | 102.36 | DEVIL | FLAGS! FLAGS FLAGS FLAGS! |
| P13 | 106.43 | PREACHER | BUT THERE WERE SOME THAT LISTENED... |
| H01 | 108.79 | HIPPIE1 | Found one! |
| H02 | 110.11 | HIPPIE2 | Move it! MOVE IT! |
| P14 | 111.86 | PREACHER | AND IT WORKED. |
| P15 | 113.15 | PREACHER | Saturday afternoon the ALL YELLOW FLAG was placed on top of the effigy... |
| P16 | 118.35 | PREACHER | ...and the rain stopped entirely. |
| P17 | 122.71 | PREACHER | The FLAG was still there in the evening, when the effigy was burned FOR ALL TO SEE |
| P18 | 127.82 | PREACHER | But the very second the FLAMES reached the FLAG, a miracle happened. |
| P19 | 132.37 | PREACHER | Without the YELLOW FLAG to hold it back, the FIRMAMENT OPENED |
| P20 | 136.50 | PREACHER | and THE RAIN RETURNED. |
| P21 | 142.24 | PREACHER | HOWEVER, the YELLOW FLAG would be RAISED AGAIN... 50, then 100 time! |
| P22 | 149.58 | PREACHER | The Spirit of the FLAGS would follow the Burn as it changed location |
| P23 | 154.17 | PREACHER | Through a Burn without any fire And the years without any Burns |
| P24 | 160.15 | PREACHER | So never forget, FLAGS are the HEART and SOUL of the Burn itself, |
| P25 | 164.43 | PREACHER | and speaking against them is BLASPHEME. |
| S06 | 167.81 | SUMMER | What must I do? |
| P26 | 169.09 | PREACHER | Repeat after me. |
| P27 | 170.67 | PREACHER | Lorem ipsum dolor sit amet... |
| S07 | 172.96 | SUMMER | Lorem ipsum dolor sit amet... |
| P28 | 174.97 | PREACHER | consectetur adipiscing elit. |
| S08 | 177.16 | SUMMER | consectetur adipiscing elit. |
| P29 | 179.14 | PREACHER | Amen. |
| S09 | 180.13 | SUMMER | Amen. |
| P30 | 181.56 | PREACHER | Welcome to the Survey, sister. Now go... and PASS IT ON. |
| R02 | 192.07 | RAVEN | Ugh, there goes another one of those YELLOW FLAGS. (the loop) |

Exact times: `tl.line(id)`; per-word: `tl.words(id)` (each word has `w` spoken token, `d` display word,
`s`/`e` global seconds); `tl.word(id, "WRONG")`. Lip-sync: `tl.mouth(SPEAKER, T)`. Every line has
`meta.letter` (balloon | shout | caption | radio | offpanel | footnote | title) telling how to letter it.

### Scenes (global seconds; frame ranges in timeline.json)

| Scene | Module (films/flaggame/scenes/) | Start | End | Title | Style |
|---|---|---:|---:|---|---|
| t01 | t01_tract.py | 0.00 | 18.46 | The Tract | physical tract in the rain → tract comic |
| t02 | t02_wrong.py | 18.46 | 43.29 | Every Word Was Wrong | tract comic (+icon, +anime), flashback ripple |
| t03 | t03_genesis.py | 43.29 | 67.46 | In the Beginning | op-art genesis → engraving |
| t04 | t04_flagmaker.py | 67.46 | 105.67 | The Flag Maker | engraving, kinetic FLAGS type, the Devil |
| t05 | t05_worked.py | 105.67 | 122.29 | And It Worked | white card → engraving; rain freeze; firmament |
| t06 | t06_firmament.py | 122.29 | 142.04 | The Firmament Opened | engraving: fire, crystal fracture, deluge |
| t07 | t07_raised.py | 142.04 | 159.83 | Raised Again | 100-raisings art-history montage → engraving walker → tract |
| t08 | t08_blaspheme.py | 159.83 | 181.17 | Blaspheme | tract comic, altar call |
| t09 | t09_passiton.py | 181.17 | 199.00 | Pass It On | tract comic → Print-Gallery Droste → physical tract (loop) |

Named cues (`tl.cue(id)`; full list with descriptions in timeline.json "cues"), e.g. tract_flip 2.6,
dive_in 6.4, wrong_burst 22.12, anime 33.75, flashback 41.34, stench 57.34, hurricane 59.34,
rain_begins 62.95, shut_up 76.99, flags_flood 87.15, flood_peak 95.72, devil_glitch 102.42,
flag_on_effigy 116.90, rain_freeze 118.92, firmament_reveal 119.56, ignite 125.36,
flames_reach_flag 129.73, firmament_crack 135.58, deluge 137.12, raise_1 143.94, raise_100 146.77,
unmask 158.8, blaspheme 166.22, amen 180.13, pass_it_on 184.43, droste_start 185.4, droste_end 191.6,
callback 194.76, tract_lands 197.4. Visual hits MUST land on these (sound is built on them).
Extra timing data: `timeline.json["chant"]` (the FLAGS FLAGS chant 87.15→peak 95.72→97.7) and
`timeline.json["hymn"]` (altar-call hymn 169.07→181.0, "Just as I am, without one Flag", amen 180.13).

---------------------------------------------------------------------------------------------------
## 2. Art direction

**Two print styles** (both provided by `vx.ink`, owned by Printer):
1. **TRACT** (present day: t01 inside the tract, t02, t08, t09): a Chick tract. Confident black brush line
   art with variable width, solid black masses (Raven's hair, deep shadows, night), mid-tones as big visible
   halftone dots, bare white paper highlights. Chick-tract acting: huge expressions, sweat drops, shock
   emanata, pointing fingers, radial "burst" backgrounds on the big lines.
2. **ENGRAVING** (the flashback, t03 after the op-art genesis → t07's walker): a Gustave Doré Bible
   engraving — dense parallel hatching that follows form, crosshatch in deep shadow, dramatic chiaroscuro,
   radiant light, rain as engraved white streaks, clouds like Doré's Deluge. Captions: white rectangular
   boxes, black border, heavy caps — exactly like the tract's caption boxes.
**Special styles:** op-art (t03 genesis: black/white hexagon op-art that moves), Byzantine icon (t02
"not a GAME"), anime/manga screentone (t02 gag, no ahegao — bewildered, sparkly, sweat drop, blush),
the art-history montage (t07), and the physical world around the tract (t01/t09: smooth greyscale
"photographic" rain and mud with depth of field — the only halftone is on the tract's pages).
**Colour:** black, white, greys — and the Flags. `vx.ink.PAPER` (neutral white) and `vx.ink.INK`.
Outside panels, the tract page is `#2B2B2D` like the reference scans.
**Presentation:** the present-day story is a real tract being read: portrait pages with three wide white
panels (black borders on the dark page, page numbers in the corner, like the scans). The camera frames one
panel at a time and glides panel-to-panel in reading order; inside the framed panel the drawing is alive
(characters act, lip-sync, balloons pop in synced to the voices). Spectacle breaks the panel borders
(hurricane over the gutters, the FLAGS text flood covering the page, the deluge flooding and soaking the
paper — ink bleeds, paper buckles — while the Flags stay pristine). t01 establishes the page design;
t02/t08/t09 follow it (agree via hub).
**Lettering** (`vx.comic`, owned by Letterer): every spoken line is lettered — there are NO burned-in
subtitles. Words appear exactly as spoken (`vx.comic.say`). Captions for the V.O. narration, balloons with
tails to the speaker's mouth, jagged radio balloons, burst balloons for shouts, off-panel tails for
interruptions from the present, footnote boxes ("*1ST VEXILLIANS 11s", "FLAGS 4:20"). Lettering is laid
out after the flag plate and must never overlap flag cloth. Onomatopoeia (KRAKOOM, FWOOSH, SPLOOSH, THUNK)
welcome.
**Motion:** everything eased, anticipation/overshoot, secondary motion; comic "on twos" (12 fps holds) is
allowed for tract panels if it looks more drawn; the flag cloth always simulates at 24 fps.
**Flags:** canonical 36"×30" cloth on a 96" pole (vx.flag defaults); heroes `Flag.simulate`, crowds
`draw_flag`/`draw_flag_field`. In the rain they hang heavy and wet (lower wind, calmer) but stay plain.

**Compositing order (every frame):**
`world (grey, cairo) → vx.ink print → vx.ink.spot(flag layer) → front objects printed with print_layer
(hands on poles, foreground people) → lettering (vx.comic) → film look (POST)`. The flag layer is a Canvas
that contains ONLY vx.flag drawings. POST: `grain=0` (the paper is the texture), mild vignette, bloom only
where light sources are.

**Transitions (the neighbouring owners agree the exact hand-off frame via hub):**
- 0.0: the loop frame — the tract face-down in the mud, back page up, rain. t09's last frame == t01's first.
- t01→t02 @18.46: continuous reading (same page), the Preacher is simply there in the next panel.
- t02→t03 @43.29: the flashback ripple (wavy dissolve on "Alchemy" 41.34) into the op-art void.
- t03→t04 @67.46: engraving continuous (rain, mud, the hippies) — the Flag Maker enters.
- t04→t05 @105.67: hard cut to the white card BUT THERE WERE SOME THAT LISTENED...
- t05→t06 @122.29: day → dusk dissolve on the same effigy composition.
- t06→t07 @142.04: from the deluge to the first raising (smash cut on "HOWEVER").
- t07→t08 @159.83: the walker unmasks (158.8) and the engraving "reprints" into the tract style; we are in the camp.
- t08→t09 @181.17: continuous (same panel/camp).

---------------------------------------------------------------------------------------------------
## 3. Engine

Shell: **`source films/flaggame/env.sh`** (activates `.venv`, sets `VX_FILM=flaggame`, cds to the repo).
Every command below assumes it. Film paths: `vx.config.FILM_ROOT = films/flaggame` (`OUT`, `CACHE`,
`AUDIO`, `SCRIPT`, `TIMELINE`); shared assets in `assets/` (fonts, soundfonts). Film 1 lives at the repo root
(default VX_FILM=unbabeling) — never break it: shared-library changes must be backward compatible.

Python 3.12 venv: numpy, scipy, numba, pycairo, opencv, shapely, pillow, soundfile, librosa, pedalboard,
pyloudnorm, praat-parselmouth, pyworld (via film 1's vocal tools), kokoro (TTS), tinysoundfont, mido,
pretty_midi, moderngl, faster-whisper, uharfbuzz. `uv pip install <pkg>` for more (no sudo). Fonts go in
`~/.local/share/fonts` (then `fc-cache -f`).

- `vx/canvas.py`, `vx/camera.py`, `vx/ease.py`, `vx/post.py`, `vx/timeline.py`, `vx/foley.py`,
  `vx/render.py` — unchanged from film 1 (read their docstrings). `tl.words(id)` words carry `d`.
- `vx/ink.py` — **Printer**: `tract`, `engrave`, `manga`, `print_layer`, `paper`, `spot`, `PAPER`, `INK`.
- `vx/comic.py` — **Letterer**: `say`, `balloon`, `caption`, `footnote`, `sfx`, `burst_bg`, `emanata`,
  `panel`, `flags_flood`, `reveal_count`.
- `vx/chars.py` (+ chars_rig/paint/kinds) — **RigMaster**: adds `render="ink"` (grey fills + black brush
  contours for the print styles) and the tract cast kinds `summer`, `raven`, `crow`, `flagmaker`, `hippie`,
  `pharisee`, `devil`, `seraph` (+ poses kneel, pray, read, toss, sit…). Stubs already accept these.
- `vx/flag.py` — unchanged (film 1's FlagSmith): cloth physics + plain-yellow renderer.
- Runner: `python -m vx.render t03 --at 1,4.5,9` (stills + contact sheet → films/flaggame/out/stills/),
  `--scale 0.5 --step 2 --workers 4` (preview → out/preview/), `--workers 4` (final → out/scenes/t03.mp4).

**Scene module contract** (`films/flaggame/scenes/tXX_name.py`; helpers `scenes/tXX_*.py`; shared helpers
agreed between owners e.g. `scenes/effigy.py` owned by t05):
```python
from vx import *
from vx import ink, comic
from vx.flag import Flag, draw_flag
from vx.chars import Character, draw_crowd
POST = dict(bloom=0.15, grain=0.0, vignette=0.12)
def setup(S): ...            # deterministic precompute; cache heavy sims in S.cache
def render(fc, state):       # pure function of the frame -> float32 (fc.h, fc.w, 3)
    world = Canvas(fc, bg=(1, 1, 1)); ...grey drawing...
    img = ink.tract(fc, world.rgb())            # or ink.engrave(...)
    flags = Canvas(fc); track.draw(flags.ctx, fc.f)
    img = ink.spot(img, flags.rgba())
    top = Canvas(fc); comic.say(top.ctx, fc, "R01", x, y, tail=mouth_xy)
    return top.over(img)
```
Scenes import each other's modules as `scenes.<module>` (the film's own package). Draw in design units
(1920×1080) so `--scale` works. Budget ≲ 1.5 s/frame at 1080p. CPU: 16 cores shared by 15 agents —
`--workers 4` max, one render at a time, prefer stills and half-res previews.

---------------------------------------------------------------------------------------------------
## 4. Audio contract

All stems: 48 kHz stereo float32 WAV, exactly 199.0 s (9,552,000 samples), aligned to T=0, in
`films/flaggame/audio/stems/`. Dry dialogue reference: `audio/stems/dialogue_dry.wav`; lip-sync:
`audio/voice/mouth.npz`. Scene hit exports: `films/flaggame/cache/foley/` (`vx.foley.export_events`,
`track.export_foley`).

| Stem | Owner | Content |
|---|---|---|
| dialogue.wav | Vocalist | every line placed exactly, processed per cast fx and scene acoustics |
| crowd.wav | Vocalist | hippie walla, "SHUT UP!" mob, the FLAGS FLAGS chant, burn crowd, deluge screams |
| choir.wav | Vocalist | TTS voices that SING: the altar-call hymn, "FLAGS" amens, montage "HUP!"s |
| music.wav | Composer | the score |
| sfx.wav + ambience.wav | Foley | hard effects, physics-derived flag flutter, rain, the Burn |

Main owns the final mix (`tools/mix.py`, settings in `films/flaggame/script/plan.json` "mix") and assembly
(`tools/assemble.py`). Film 1's audio tools in `tools/` are film-1 specific: write film-2 tools in
`films/flaggame/tools/` and import/adapt the shared DSP libraries (`tools/vocal_lib.py`, `tools/music_lib.py`,
`tools/foley_lib.py`, `tools/foley_flutter.py`); if a lib hard-codes film-1 paths or duration, make it
film-aware through `vx.config` (FILM_ROOT/TIMELINE/AUDIO/CACHE) without changing film-1 results.

---------------------------------------------------------------------------------------------------
## 5. Coordination

- Own only your files; message the owner via `hub` (op:"send") for changes elsewhere.
- Library owners keep public signatures stable and broadcast (to:"all") when versions land.
  Scene owners code against the documented APIs immediately (the stubs work).
- Scene owners export sound hits (`vx.foley.export_events`) and hero-flag tracks (`track.export_foley`).
- No project-wide tests/linters/formatters; no git commits (Main commits). Verify visually: render stills
  and LOOK at them with the read tool; iterate until genuinely beautiful and funny.
- Agents: Printer, Letterer, RigMaster, Vocalist, Composer, Foley, T01Tract, T02Wrong, T03Genesis,
  T04FlagMaker, T05Worked, T06Firmament, T07Raised, T08Blaspheme, T09PassItOn. Main = director.

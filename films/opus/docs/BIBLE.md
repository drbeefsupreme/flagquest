# OPVS SCHISMATICVM — Production Bible (film 3)

*The Unbabeling, set in glass and gold.* A 3:32 (212.0 s) remake of film 1, THE UNBABELING, in which **every scene is a
mosaic** and the telling is old-school theology: Byzantine and Ravenna glass-and-gold mosaics, Latin tituli, a scholastic
disputation, councils and anathemas, plainchant. 1920×1080, 24 fps, 5088 frames. Every frame and sound is made by code in
this repo. Brief: *"completely original and mind-blowing"*. Festival-grade art, not a tech demo.

Film directory: `films/opus/` — select it with `export VX_FILM=opus` (then `source .venv/bin/activate`).

---------------------------------------------------------------------------------------------------
## 0. THE ONE RULE (absolute, zero exceptions)

**FLAGS ARE PLAIN YELLOW WITH NO OTHER MARKINGS UPON THEM.**

- Every flag is a plain yellow rectangle of cloth on a plain untreated wooden pole, drawn ONLY through `vx/flag.py`
  (`Flag(...).simulate(...)` → `FlagTrack.draw(ctx, i)`, `draw_flag`, `draw_flag_field`).
- **The Flag is the only thing in the film that is NOT made of tesserae.** Never tessellate, tile, texture, outline,
  grout, crack, glitter or overlay flag cloth. Composite flags AFTER the mosaic, on top, and draw nothing over them.
  (Mosaic tiles may fly *behind* or *around* a flag, never across the cloth.)
- No banners, labara, pennants or standards with symbols; nations, faiths and philosophies are shown with crowns, towers,
  domes, books, halos, maps, tiaras, scrolls — never with flags. No crosses, chi-rhos or religious symbols on anything
  that could read as a flag. Tituli (inscriptions) live on the wall, never on cloth.
- Gold is the wall's divine light, but it must stay browner, deeper and more metallic than flag yellow (#FFC41A), so the
  seamless flat Flags always read as the cleanest, brightest yellow in any frame.

---------------------------------------------------------------------------------------------------
## 1. Concept

The whole film is the mosaic programme of an imaginary basilica — the Basilica of the Unbabeling — seen by a pilgrim's
candlelit eye travelling narthex → nave → apse → dome. Every image is set in tesserae (smalti glass, gold leaf, stone in
mortar) and the mosaics come alive: figures move by their tiles re-colouring, whole panels of tiles travel, single tiles
fly, split, fall and re-set themselves.

- **Seamless vs. shattered.** The world is made of pieces; the Flag is whole. (Cyprian, *De unitate ecclesiae*: the
  seamless robe that cannot be divided. N03: "A sign of itself, woven without seam.")
- **The one and the many.** A mosaic is "Each of them alone. All of them together." Every tessera is separate, held
  apart and held together by the same mortar. Schism is tiles splitting; the Unbabeling is not a melt into one colour,
  it is every tile re-set so that together they make one picture.
- **Tessera / Tesseract.** Film 1's Crystal is now the Tesseract (from *tessera*, "four-cornered piece"). Every
  tessera in creation is one of its shadows.
- **Babel = the mosaic falls off the wall.** When everyone is a cabal of one, the picture cannot hold: the entire
  mosaic sheds off the wall in an avalanche of tesserae, leaving bare lime plaster with the red **sinopia** (the
  mosaicist's underdrawing). Silence. Then the one whole Flag is planted, and the fallen tesserae rise and set
  themselves again.
- **Old-school theology.** Latin tituli name each panel like the Bayeux Tapestry ("HIC…"); INCIPIT opens the film and
  EXPLICIT closes it; the Hypermind is a biblically-accurate ophanim; the Master and novice argue in the form of the
  *Summa* (VIDETVR QVOD NON / SED CONTRA / RESPONDEO DICENDVM); the schismmancers are warrior saints under the motto
  IN HOC SIGNO SCINDES ("in this sign you shall split"); the heretics anathematise each other; the hymn is an antiphon,
  cantor in Latin, choir in English; the film ends with the cantor's Explicit and the sung response *Vexillo gratias*.

Characters (drawn by `vx/icon.py`, owner Iconographer): **Lutie** (novice vexiherbologist, ochre robe), **Master Crocus**
(ancient Hierophant, black robe + ochre over-robe and veil, white beard), **the Schismmancers** as Byzantine warrior saints
(lamellar over ochre robes, NVG helmets, telescoping flag-lances; **Schism Actual** with a radio headset), **President
Jaguar** as Emperor Justinian, **the Hypermind** as an ophanim-polycandelon (wheels within wheels full of eyes, six wings
= six canons), **Lord Egregore** (faceless anti-Pantocrator of static), citizens / heretics / popes / crackpots /
philosophers / peoples.

---------------------------------------------------------------------------------------------------
## 2. Dialogue (LOCKED — animate to it)

Source `script/lines.json`, placement `script/plan.json`, built by `tools/build_timeline.py` into `timeline.json`.
Voice WAVs: `audio/voice/raw/<ID>.wav` (24 kHz mono, dry). Word timestamps: `tl.word("N03", "rotated")` → (start, end).
Lip-sync: `tl.mouth(SPEAKER, T)` (0..1), `tl.speaking(SPEAKER, T)`.

| ID | start | end | Speaker | Line |
|----|------:|----:|---------|------|
| A01 | 2.00 | 4.55 | CANTOR | Incipit opus schismaticum. |
| N01 | 5.60 | 7.67 | LECTOR | In the beginning, there was no beginning. |
| N02 | 8.40 | 12.00 | LECTOR | Only the Tesseract, turning in a direction you cannot point to. |
| N03 | 13.00 | 19.27 | LECTOR | And out of that turning, the Flag rotated into existence. A sign of itself, woven without seam. |
| N04 | 23.00 | 29.08 | LECTOR | Once, humanity dwelt inside great machines of meaning. Nations. Faiths. Philosophies. |
| N05 | 29.80 | 32.96 | LECTOR | Each one a mosaic, promising to be the whole picture. |
| N06 | 34.40 | 35.56 | LECTOR | Then came the slop. |
| N07 | 41.00 | 47.44 | LECTOR | Wave upon wave of it. A different feed for every face, until no two tesserae showed the same world. |
| J01 | 49.30 | 58.53 | JAGUAR | My fellow Americans, and everybody else! Today I sign the Noospheric Munitions Act of twenty forty-two. Weaponized memes are hereby banned! |
| J02 | 59.00 | 61.82 | JAGUAR | Except the yellow ones. Those are... a sacrament. |
| H01 | 64.00 | 71.62 | HYPERMIND | Training complete. Six canons ingested. Every statement about Flags is true. |
| H02 | 72.40 | 73.83 | HYPERMIND | Hypermeme compiled. |
| L01 | 76.20 | 77.17 | LUTIE | It's... a Flag. |
| C01 | 77.90 | 80.24 | CROCUS | It was always going to be a Flag. |
| L02 | 81.20 | 86.42 | LUTIE | Master, it seems the world should not be broken further. The slop has already shattered it. |
| C02 | 87.00 | 92.22 | CROCUS | On the contrary. The false coherence must fall, before the true one can rise. |
| C03 | 92.80 | 95.53 | CROCUS | I answer that: summon the schismmancers. |
| K01 | 96.60 | 101.13 | COMMANDER | Schism Actual to all teams. Target: every full-stack memeplex on Earth. |
| K02 | 101.60 | 103.37 | COMMANDER | Method: recursive balkanization. |
| K03 | 103.90 | 105.06 | COMMANDER | Breach! Breach! Breach! |
| K04 | 106.90 | 108.49 | COMMANDER | Every nation, a nation of one. |
| X01 | 109.20 | 111.54 | CITIZEN1 | I hereby declare the Sovereign Republic of Me! |
| X02 | 111.80 | 113.03 | CITIZEN2 | Population: one! |
| K05 | 113.70 | 115.14 | COMMANDER | Every faith, a cabal of one. |
| X03 | 115.60 | 116.38 | HERETIC1 | Anathema! |
| X04 | 116.50 | 117.45 | HERETIC2 | No, you're anathema! |
| X05 | 117.70 | 119.21 | HERETIC3 | I am my own pope! |
| K06 | 120.00 | 122.82 | COMMANDER | Every philosophy, a crackpot, screaming into the void. |
| X06 | 123.20 | 126.12 | CRACKPOT1 | Numbers were made up by guys who were angry at poets! |
| X07 | 126.50 | 127.95 | CRACKPOT2 | I am, therefore I am! |
| K07 | 128.60 | 129.88 | COMMANDER | May it schism endlessly. |
| L03 | 136.00 | 138.78 | LUTIE | Master! Nobody understands anybody anymore! |
| C04 | 139.40 | 141.32 | CROCUS | Good. Now they are ready. |
| C05 | 150.60 | 154.87 | CROCUS | A Flag needs no translation. It means only... Flag. |
| N08 | 156.50 | 158.88 | LECTOR | Each of them alone. All of them together. |
| (cantor) | 163.00 | 168.70 | CANTOR (sung) | *Vexilla sint. Vexilla sunt. Vexilla erunt. Vexilla.* (solo plainchant, mode I) |
| (hymn) | 168.70 | 180.10 | CHOIR (sung) | "Flags be. Flags are. Flags will. Flags." (84 BPM, 16 beats, organum) |
| N09 | 180.40 | 184.45 | LECTOR | Heaven and earth will pass away, but the ten thousand Flags remain. |
| N10 | 184.90 | 186.55 | LECTOR | And the Survey was completed. |
| L04 | 188.50 | 189.65 | LUTIE | Master! For you. |
| C06 | 190.40 | 193.24 | CROCUS | I do not want this Flag. It is sticky. |
| L05 | 193.90 | 196.23 | LUTIE | But Master, all Flags are one Flag! |
| C07 | 197.00 | 198.57 | CROCUS | This one is sticky. |
| N11 | 199.60 | 202.38 | LECTOR | Go forth, O signifier, and become glorious. |
| (crowd) | 202.90 | 204.00 | ALL | "Glorious!" (the Signifier's salute) |
| A02 | 204.60 | 207.15 | CANTOR | Explicit opus schismaticum. |
| (response) | 207.60 | 209.60 | CHOIR (sung) | *Vexillo gratias.* |

Scenes (global seconds; frame ranges in timeline.json):

| Scene | Module | Start | End | Title |
|---|---|---:|---:|---|
| m01 | scenes/m01_incipit.py | 0.0 | 22.0 | Incipit: the Tesseract |
| m02 | scenes/m02_slop.py | 22.0 | 48.5 | The Great Machines and the Slop |
| m03 | scenes/m03_edict.py | 48.5 | 62.5 | The Edict of the Jaguar |
| m04 | scenes/m04_hypermind.py | 62.5 | 95.75 | Flagartha: the Hypermind and the Disputation |
| m05 | scenes/m05_breach.py | 95.75 | 106.5 | The Schismmancers |
| m06 | scenes/m06_schism.py | 106.5 | 131.5 | Recursive Balkanization |
| m07 | scenes/m07_babel.py | 131.5 | 146.0 | Babel |
| m08 | scenes/m08_unbabel.py | 146.0 | 163.0 | The Unbabeling |
| m09 | scenes/m09_dome.py | 163.0 | 187.5 | The Dome of Flagistan |
| m10 | scenes/m10_explicit.py | 187.5 | 212.0 | The Sticky Flag: Explicit |

Named sync cues: `timeline.json["cues"]` / `tl.cue(id)` (full list with descriptions in `script/plan.json`). Visual hits
MUST land on them (the sound is built on them): `candle_lit` 0.8, `incipit` 2.0, `tesseract` 8.4, `flag_rotates_in`
14.8, `title` 19.6, `slop` 34.4, `slop_crash` 36.3, `edict_sign` 52.5, `edict_seal` 58.0, `ka_ching` 62.1,
`hypermind_wake` 62.9, `hypermeme` 74.2, `videtur` 81.2, `sed_contra` 87.0, `respondeo` 92.8, `schismmancers` 95.75,
`breach` 105.8, `nations` 106.9, `faiths` 113.7, `philosophies` 120.0, `endless` 128.6, `babel_rise` 131.5,
`egregore` 134.0, `collapse` 141.8, `silence` 145.0, `flag_planted` 149.0, `tiles_rise` 155.0, `convergence` 159.0,
`hymn_cantor` 163.0, `hymn` 168.7, `hymn_end` 180.1, `gateless_gate` 180.4, `survey_done` 184.9, `sticky` 190.4,
`glorious` 202.9, `explicit` 204.6, `gratias` 207.6, `end_card` 209.5, `film_end` 212.0.

---------------------------------------------------------------------------------------------------
## 2b. Art direction

**Look:** photographed Byzantine mosaic, alive. Ravenna (Galla Placidia's starry vault, San Vitale's Justinian panel,
Sant'Apollinare Nuovo's processions), Hagia Sophia, Hosios Loukas, Monreale, the San Marco Genesis cupola and Pentecost
dome, the Pompeii Alexander and Academy mosaics, Cosmati pavements. Tesserae ~8–14 px in grounds, ~4–6 px in faces and
letters; courses (andamento) hug every contour; gold grounds shimmer as the candlelight moves. Never a pixel filter:
tiles are square-ish, set in rows, with mortar joints and bevels. Composition is iconic: frontal figures, hierarchic
scale, registers and bands, borders (guilloche, meander, rainbow bands, jewelled frames), architecture framing each panel.

**Light:** a basilica at night: warm candle pools, flickering; gold wakes up where the light falls; deep lapis and black
elsewhere. Let the camera move slowly over walls, vaults and domes (the tiles glint as it moves). Big moments may flood
the frame with gold light or flag-yellow glow.

**Palette:** `vx.mosaic.SMALTI` / `SC` — lapis, night_vault, porphyry, tyrian, malachite, verdigris, terracotta, marble,
bone, flesh, ink, slate, silver, gold/gold_d/gold_l, mortar, plaster, sinopia. Slop may use the old `slop_*`, `glitch_m`,
`glitch_c`, `screen_glow` from `vx.config.C` (sickly pastel glass lit from within). Vexillian robes: `robe` family ochre.
Flags: vx.flag's palette yellows only.

**Lettering (tituli):** Latin in Roman square capitals set in tesserae (V for U; interpuncts ·; abbreviation overlines),
dark on gold or white/gold on lapis, in bands above/below panels. English appears only small, as a secondary gloss when
needed. No subtitles are burned into scenes. Nothing is ever written on flags.

**Motion:** everything eased; the Byzantine world is stately (slow camera, holds), except when it breaks (schism, breach,
collapse: fast, violent, physical). Secondary motion: candle flicker, gold glint, cloth-simulated flags, blinks,
breathing (tiles re-colour subtly), wheels turning.

**Transitions (joint responsibility of the two scene owners):**
- m01→m02 @22.0: m01 tilts DOWN from the starry lapis vault; m02 opens on the lapis upper border of the nave wall and
  continues the tilt down onto the great mosaic of the machines.
- m02→m03 @48.5: m02 pushes into ONE slop tessera until its glowing screen is full-frame static (grey-pastel noise);
  m03 opens on that static, which resolves (tile by tile, flipping) into the gold of the Jaguar's panel.
- m03→m04 @62.5: the solidus glint in the Jaguar's paw flares to a gold point; m04 opens from that point in darkness
  (the first eye of the Hypermind opens where the point was).
- m04→m05 @95.75: hard cut on "schismmancers" (last 0.3 s of m04 may push in / flash).
- m05→m06 @106.5: the breach's blinding flag-yellow light (`flag_glow` #FFF1B8) fills the frame by 106.5; m06 opens
  from that glow fading onto the mappa mundi.
- m06→m07 @131.5: m06 ends with the recursively splitting tesserae as glittering dust swirling on a dark ground; m07
  opens on that dust already spiralling up into the Tower of Babel.
- m07→m08 @146.0: m07 goes silent and near-black at 145.0 (bare plaster, sinopia barely visible, one candle); m08 opens
  on the same bare wall and candle.
- m08→m09 @163.0: m08 tilts UP from the re-set wall into the dome; m09 opens looking up into the dome.
- m09→m10 @187.5: white-gold flash (`flag_glow` full frame) at 187.0–187.5; m10 opens from it into the apse.
- m10 ends on the donor inscription / EXPLICIT; a candle is snuffed; black by 212.0.

---------------------------------------------------------------------------------------------------
## 3. Engine

Read the source — it is short. `vx/config.py` (W, H, FPS, SR, palette `C`, per-film paths), `vx/canvas.py` (Canvas in
design units, compositing helpers `over/screen/add/blur/glow`, path helpers, `text`), `vx/camera.py`, `vx/ease.py`,
`vx/timeline.py`, `vx/post.py` (global look: bloom, vignette, grain, grade — set per scene with `POST = {...}`),
`vx/flag.py` (THE FLAG — read its docstring), `vx/foley.py` (`export_track`, `export_events` for the sound team),
`vx/render.py` (runner).

- **`vx/mosaic.py` (owner Tessellator)** — the mosaic engine. Its top docstring is the API manual. Core:
  `Layout.grid/rings/flow(...)` fixed tile layouts (build in setup, never per frame), `render(fc, cartoon, lay, gold=,
  light=candle(t))`, `render_layer(fc, rgba, lay, M)` for moving panels/figures, `tiles()` + `draw_tiles()` for loose
  tiles, plus (as they land) walls/domes/apse projection, tituli, tile physics (fall, fly, subdivide, crack, flip,
  screen-tiles), sinopia.
- **`vx/icon.py` (owner Iconographer)** — Byzantine figure cartoons: `Figure(kind).draw(ctx, x, y, h, pose=, t=,
  mouth=, look=, expr=)` → anchors (hands, `pole` for a held flag, head, halo), with fine/gold masks for the mosaic.

**Scene pipeline:** paint the cartoon with cairo (figures via vx.icon, ground, borders, tituli) → mosaic it (vx.mosaic,
with fine and gold masks) → composite loose tiles/light → composite FLAGS LAST (vx.flag, whole cloth) → return rgb.

**Scene module contract** (`films/opus/scenes/mXX_name.py`):
```python
from vx import *
from vx import mosaic as mz
from vx.flag import Flag, draw_flag
from vx.icon import Figure
POST = dict(bloom=0.35, grain=0.0, vignette=0.3)     # optional
def setup(S):          # S.id, S.start, S.end, S.dur, S.n, S.f0, S.fps, S.s, S.cache, S.tl
    ...                # deterministic precompute (layouts, cloth sims). Cache heavy results under S.cache.
    return state
def render(fc, state): # fc.f/t local frame/seconds, fc.F/T global, fc.p progress, fc.s scale, fc.w/h pixels
    ...; return img    # float32 (fc.h, fc.w, 3) sRGB
def post(fc, state): return dict(fade=...)          # optional per-frame look overrides
```
Rules: render is a pure function of (fc, state) — frames render in parallel, out of order; derive every animation from
fc.t / fc.T (tile physics included). Draw in design units so `--scale 0.5` works. Budget ≤ 1.5 s/frame at 1080p.
CPU etiquette: 32 cores shared by 15 agents — `--workers 4` max, numba single-threaded, prefer stills and `--scale 0.5`.

- `python -m vx.render m03 --at 1,4.5,9` → stills + contact sheet in `films/opus/out/stills/` (LOOK at them)
- `python -m vx.render m03 --scale 0.5 --step 2 --workers 4` → preview video in `films/opus/out/preview/`
- `python -m vx.render m03 --workers 4` → final `films/opus/out/scenes/m03.mp4`

---------------------------------------------------------------------------------------------------
## 4. Audio contract

All stems: 48 kHz, stereo, float32 WAV, exactly 212.0 s (10,176,000 samples), aligned to T=0, in `films/opus/audio/stems/`.

| Stem | Owner | Content |
|---|---|---|
| `dialogue.wav` | Vocalist | every timeline line, processed per cast fx + the basilica acoustic, placed exactly |
| `crowd.wav` | Vocalist | walla: the procession, slop whispers, the heretics' council, Babel's multilingual cacophony (hard stop 145.0), "Glorious!" |
| `choir.wav` | Vocalist | the convergence (babble → one chord, 155–163), the antiphon (cantor 163.0–168.7, choir 168.7–180.1), *Vexillo gratias* 207.6–209.6, choir pads |
| `music.wav` | Composer | score: organ, ison drone, bells, strings, the imperial fanfare, the schism canon, Babel's cluster |
| `sfx.wav` + `ambience.wav` | Foley | tesserae (setting clicks, glass tinkle, avalanche), candle, stone, armour, breach, coins, seal, physics-derived flag flutter; the basilica's room tone |

The director owns the final mix (`tools/mix.py`: ducking, bus compression, −14 LUFS, −1 dBTP) and assembly.
Dry reference: `audio/stems/dialogue_dry.wav`. Lip-sync envelopes: `audio/voice/mouth.npz`.
**SILENCE 145.0–146.5 is sacred:** every stem (near) silent there, except a faint breath of wind in the last 0.5 s.
Scene owners export sound hits with `vx.foley.export_events` (global seconds) and flag-motion tracks with
`FlagTrack.export_foley`; the sound team reads them from `films/opus/cache/foley/`.

---------------------------------------------------------------------------------------------------
## 5. Coordination

- Own only your files. Never edit another agent's files — message the owner with `write agent://<Name>` (Tessellator,
  Iconographer, M01..M10, Vocalist, Composer, Foley; the director is your parent).
- Library owners keep public signatures stable and tell the director when a new version lands.
- Scene owners code against the documented APIs immediately (v0 already works) and ask library owners for missing
  capabilities early; export foley events and keep them current.
- Hand-offs: agree the exact last/first frame with your neighbour scene owner(s) directly.
- Do not spawn sub-agents. No project-wide test suites, linters or formatters. Verify your own work visually.
- When done, report: files written, how to render, render time per frame, known issues.

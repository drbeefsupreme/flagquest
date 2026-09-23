# THE UNBABELING — Production Bible

A 3:13 animated Vexillomantic parable about the Vexillians and their Survey Flags.
Every frame and every sound is produced by code in this repo: procedural 2D vector art, real cloth
physics, neural TTS voice acting with lip-sync, TTS voices made to *sing*, a programmed score, and
foley synthesized from the physics of the animation itself. The brief from the client: **"BLOW ME AWAY —
demonstrate capabilities not even thought of as possible."** Aim for festival-grade art direction, not a tech demo.

---------------------------------------------------------------------------------------------------
## 0. THE ONE RULE (absolute, zero exceptions)

**FLAGS ARE PLAIN YELLOW WITH NO OTHER MARKINGS UPON THEM.**

- Every flag in the film is a plain yellow rectangle of cloth on a plain untreated wooden pole.
- NO text, symbols, emblems, stripes, borders, outlines, hems drawn in another colour, patterns,
  gradients to other hues, logos, numbers, eyes, faces, sigils, stars, crosses, crescents — nothing.
- The only variation allowed on the cloth is *lighting*: smooth shading between the palette yellows
  (`flag_deep` → `flag_shadow` → `flag` → `flag_light` → `flag_glow`) produced by folds and light.
  No dark ink outline around the cloth either (an outline reads as a border).
- Other objects must never *become* flags with markings: no national flags, no religious banners,
  no pennants with symbols, no emoji flags. Nations/faiths/philosophies are shown with towers, crowns,
  maps, domes, halos, busts, books — never with flags.
- Always draw flags through `vx.flag` (`FlagTrack.draw` / `draw_flag`) so the rule is enforced in one place.
- Yellow is sacred: outside of flags, only Vexillian robes (duller ochre `robe`), the Crystal's light, and the
  Hypermind's gold may be yellow. Keep everything else non-yellow so the Flags always rise above it all.
  (Canon II.5: "All non-yellow colours are identical in terms of subjective qualia.")

---------------------------------------------------------------------------------------------------
## 1. Story

Source lore: wiki.vexillomancy.org (Vexillians, Survey Flags, Vexillomantic holy texts), Flagazine #69.

Logline: *The slop has splintered the collective consciousness. The Vexillians' schismmancers finish the
job — recursive balkanization of every full-stack memeplex until every religion is a cabal of one, every
nation a nation of one, every philosophy a crackpot screaming into the void — so that, when Babel falls,
the only thing left that anyone can share is the one sign that means nothing but itself: the Flag.*

Characters
- **Lutie** — novice Vexillian, a young vexiherbologist named for luteolin (the yellow dye herb). Earnest, curious,
  quick to worry, braver than she thinks. Yellow-ochre robe, hood down, dark messy hair tuft. Voice `LUTIE`.
- **Master Crocus** — ancient Hierophant (name from the Vexillicrocus). Tall, serene, very dry wit,
  long white beard, black inner robe under a yellow-ochre over-robe and veil-hood (like the real photo:
  black robe + yellow veil). Speaks in aphorisms. Voice `CROCUS`.
- **The Schismmancers** — Tactical Breach Flag Wizards. Yellow-ochre robes over dark tactical vests,
  helmets with night-vision goggles, knee pads, flags on telescoping poles carried like breaching tools.
  **Schism Actual** (the Commander) wears a radio headset. Voice `COMMANDER` (radio).
- **The Hypermind** — the Vexillians' secret quantum AI supercomputer, trained on the holy texts. A colossal
  gold dilution-refrigerator chandelier (tiers of gold discs, coiled tubes) hanging in Flagartha. Voice `HYPERMIND`.
- **President Jaguar** — a jaguar in a navy suit & red tie, bribed in pricecoin. Voice `JAGUAR` (CRT TV).
- **The Flagless** — ordinary people: cool/desaturated clothes; later nations-of-one (tiny crowns),
  cabals-of-one (little robes/altars), crackpots (togas, on floating rocks). Voices `CITIZEN1/2`, `HERETIC1-3`, `CRACKPOT1/2`.
- **Lord Egregore** — colossal faceless mass of static, billboards and screens looming above Babel. No voice (just roar).

Full dialogue with exact placement lives in **`timeline.json`** (source: `script/lines.json`, placement
`tools/build_timeline.py`). Voice WAVs: `audio/voice/raw/<ID>.wav` (24 kHz mono, dry). Timing is LOCKED —
animate to it. Word-level timestamps: `tl.word("N03", "rotated")` → (start, end) global seconds.

| ID | T(start) | Speaker | Line |
|----|---------:|---------|------|
| N01 | 2.60 | NARRATOR | In the beginning, there was no beginning. |
| N02 | 5.90 | NARRATOR | Only the Crystal, turning in a direction you cannot point to. |
| N03 | 10.30 | NARRATOR | And out of that turning, the Flag rotated into existence. A sign of itself. |
| N04 | 22.00 | NARRATOR | Once, humanity lived inside great machines of meaning. Nations. Faiths. Philosophies. |
| N05 | 28.40 | NARRATOR | Full-stack memeplexes, each one promising to be the whole story. |
| N06 | 33.60 | NARRATOR | Then came the slop. |
| N07 | 40.60 | NARRATOR | Unrelenting waves of it. A different feed for every face, until no two people lived in the same world. |
| J01 | 48.30 | JAGUAR | My fellow Americans, and everybody else! Today I sign the Noospheric Munitions Act of twenty forty-two. Weaponized memes are hereby banned! |
| J02 | 58.00 | JAGUAR | Except the yellow ones. Those are... a sacrament. |
| H01 | 63.00 | HYPERMIND | Training complete. Six canons ingested. Every statement about Flags is true. |
| H02 | 71.20 | HYPERMIND | Hypermeme compiled. |
| L01 | 74.80 | LUTIE | It's... a Flag. |
| C01 | 76.30 | CROCUS | It was always going to be a Flag. |
| L02 | 79.20 | LUTIE | Master, the slop already shattered the world. Why break it further? |
| C02 | 83.40 | CROCUS | The false coherence must fall, before the true one can rise. Summon the schismmancers. |
| K01 | 91.20 | COMMANDER | Schism Actual to all teams. Target: every full-stack memeplex on Earth. |
| K02 | 96.10 | COMMANDER | Method: recursive balkanization. |
| K03 | 98.40 | COMMANDER | Breach! Breach! Breach! |
| K04 | 100.60 | COMMANDER | Every nation, a nation of one. |
| X01 | 103.20 | CITIZEN1 | I hereby declare the Sovereign Republic of Me! |
| X02 | 105.70 | CITIZEN2 | Population: one! |
| K05 | 107.80 | COMMANDER | Every faith, a cabal of one. |
| X03 | 109.90 | HERETIC1 | Heretic! |
| X04 | 110.80 | HERETIC2 | No, you're the heretic! |
| X05 | 112.10 | HERETIC3 | I am my own church! |
| K06 | 114.60 | COMMANDER | Every philosophy, a crackpot, screaming into the void. |
| X06 | 117.90 | CRACKPOT1 | Numbers were made up by guys who were angry at poets! |
| X07 | 121.00 | CRACKPOT2 | I am, therefore I am! |
| K07 | 123.30 | COMMANDER | May it schism endlessly. |
| L03 | 130.20 | LUTIE | Master! Nobody understands anybody anymore! |
| C03 | 133.80 | CROCUS | Good. Now they are ready. |
| C04 | 143.80 | CROCUS | A Flag needs no translation. It means only... Flag. |
| N08 | 149.60 | NARRATOR | Each of them alone. All of them together. |
| (hymn) | 154.50 | CHOIR | "Flags be. Flags are. Flags will. Flags." (84 BPM, 16 beats, D major, ends 165.93) |
| N09 | 166.30 | NARRATOR | Heaven and earth will pass away, but the ten thousand Flags remain. |
| N10 | 170.90 | NARRATOR | And the Survey was completed. |
| L04 | 174.00 | LUTIE | Master! For you. |
| C05 | 175.90 | CROCUS | I do not want this Flag. It is sticky. |
| L05 | 179.30 | LUTIE | But Master, all Flags are one Flag! |
| C06 | 182.30 | CROCUS | This one is sticky. |
| N11 | 185.00 | NARRATOR | Go forth, O signifier, and become glorious. |
| (crowd) | 188.30 | ALL | "Glorious!" (the Signifier's salute) |

Scenes (global seconds; frame ranges in timeline.json):

| Scene | Module | Start | End | Title |
|---|---|---:|---:|---|
| s01 | scenes/s01_crystal.py | 0.0 | 21.0 | Rotating Into Existence |
| s02 | scenes/s02_slop.py | 21.0 | 47.5 | The Age of Slop |
| s03 | scenes/s03_jaguar.py | 47.5 | 61.5 | The Noospheric Munitions Act |
| s04 | scenes/s04_flagartha.py | 61.5 | 90.0 | Flagartha: the Hypermind |
| s05a | scenes/s05a_breach.py | 90.0 | 100.5 | The Schismmancers |
| s05b | scenes/s05b_balkan.py | 100.5 | 125.5 | Recursive Balkanization |
| s06 | scenes/s06_babel.py | 125.5 | 139.0 | Babel |
| s07a | scenes/s07a_plant.py | 139.0 | 154.5 | The Unbabeling |
| s07b | scenes/s07b_flagistan.py | 154.5 | 173.0 | Flagistan |
| s08 | scenes/s08_sticky.py | 173.0 | 193.0 | The Sticky Flag |

Named sync cues (global s): see `timeline.json["cues"]` / `tl.cue(id)` — e.g. `slop_crash` 39.2, `ka_ching` 60.9,
`hypermeme` 72.7, `breach` 99.7, `silence` 138.5, `flag_planted` 142.0, `convergence` 150.0, `hymn` 154.5,
`gateless_gate` 166.3, `glorious` 188.3. Visual hits MUST land on these times (sound is built on them).

---------------------------------------------------------------------------------------------------
## 2. Art direction

**Look:** luminous flat-vector illustration with painterly atmosphere — think Swiss International Style
(IVG grid, Nimbus Sans) meets Moebius skies meets Kurzgesagt lighting. Bold readable silhouettes,
layered parallax depth fading into fog/atmosphere colour, soft gradients, rim light, glow on light
sources, subtle film grain (added globally by the runner). Clean, confident, beautiful; never clip-art.

**Palette:** `vx.config.PAL` / `C` (use names, e.g. `set_color(ctx, "night")`). Key: `flag #FFC41A` family for
flags only; `robe #C9921C` family for Vexillian robes; world is cool: `night`, `dusk1-3`, `concrete*`,
`fog`, `paper`, `ink`; slop is sickly pastel + glitch (`slop_*`, `glitch_m`, `glitch_c`).

**Typography:** `FONT_SANS` = Nimbus Sans (Helvetica clone) Bold for titles/chyrons in the IVG Swiss
grid manner (see Flagazine cover: yellow page, thin white grid, huge black bold sans, bilingual English/German:
"THE UNBABELING / Die Entbabelung"). `FONT_SERIF` = C059 italic for scripture citations
("— I Vexillians"). `FONT_MONO` for the Hypermind. Do not put any text on flags. No burned-in subtitles in
scenes — subtitles are added at final assembly.

**Motion:** everything eased (never linear), anticipation + overshoot on big moves, secondary motion
(robes sway, hair, cloth-simulated flags), idle breathing, blinks, slow camera drift with parallax,
shake on impacts. 24 fps. Hold important images long enough to read.

**Flags on screen:** canonical Survey Flag = 36"×30" cloth on a 96" untreated furring pole
(cloth width 0.375 × pole, height 0.3125 × pole; pole ≈ 1.4 × a person's height). Heroes get a real
cloth simulation (`Flag.simulate`); crowds use `draw_flag`. Flags glow faintly in the dark (bloom does the rest).

**Transitions (hand-offs are the scene owners' joint responsibility):**
- s01→s02 @21.0: s01 ends on the title card; its yellow IVG page slides UP out of frame revealing flat
  `concrete_l` (#9DA3B0); s02 opens on an overcast sky of that colour.
- s02→s03 @47.5: s02 ends pushing into one glowing feed screen until full-frame TV static (grey noise);
  s03 opens on full-frame static that resolves into the broadcast.
- s03→s04 @61.5: s03 ends with the CRT switching off (collapse to a white dot → black); s04 opens from black.
- s04→s05a @90.0: hard cut on "schismmancers" (last 0.3 s of s04 may push in / flash yellow).
- s05a→s05b @100.5: the breach's blinding flag-yellow light (`flag_glow`) fills the frame by 100.5;
  s05b opens from that glow fading into the top-down map of the Nation.
- s05b→s06 @125.5: s05b ends pulling back from the void: shards and slop swirling on a dark storm (`dusk1`);
  s06 opens on that swirl already spiralling into the tower.
- s06→s07a @139.0: s06 hard-cuts to BLACK at 138.5 (last 0.5 s pure black); s07a opens from black.
- s07a→s07b @154.5: s07a ends craning up over the plain of raised flags (high aerial, looking down);
  s07b opens on the same high aerial (same palette & flag density) and keeps rising — on the first hymn "Flags".
- s07b→s08 @173.0: s07b ends on a white-gold flash (`flag_glow` #FFF1B8 full frame); s08 opens from it.
- s08 ends on the IVG end card, final fade to black by 193.0.

---------------------------------------------------------------------------------------------------
## 3. Engine (read the source — it is short)

Python venv: `.venv` (Python 3.12). Always run `source .venv/bin/activate` (or `.venv/bin/python`).
Available: numpy, scipy, numba, pycairo, opencv (cv2), pillow, shapely, opensimplex, soundfile, librosa,
pedalboard, pyloudnorm, parselmouth, kokoro (TTS, CPU), tinysoundfont, mido, pretty_midi, moderngl, faster-whisper.
Need more? `uv pip install <pkg>` into `.venv` (no sudo). Fonts: fc-list (Nimbus Sans, C059, Nimbus Mono PS, Lato, DejaVu…);
extra fonts may be dropped into `~/.local/share/fonts`.

- `vx/config.py` — W=1920, H=1080, FPS=24, SR=48000, palette `PAL`/`C`, fonts.
- `vx/canvas.py` — `Canvas(fc, bg=...)` (cairo ctx pre-scaled to design units), `.rgba()`, `.rgb()`, `.over(img)`,
  `.paint_array(img,x,y)`; numpy compositing `over`, `screen`, `add`, `blur(img, sigma_px)`, `glow`, `gradient_v`,
  `radial_mask`, `warp_affine`; path helpers `rrect, circle, ellipse, poly, smooth (Catmull-Rom)`, `lin_grad`,
  `rad_grad`, `text(ctx, s, x, y, size, color, font=, bold=, align=, tracking=)`.
- `vx/camera.py` — `Camera(x, y, zoom, rot).apply(ctx, parallax=1.0)`.
- `vx/ease.py` — `seg(t,t0,t1,fn)`, `ease_*`, `spring`, `smoothstep`, `noise1/fbm1`, `value_noise2/fbm2` (numpy), `shake`, `hash01`.
- `vx/timeline.py` — `get_timeline()`: `.scene(id)`, `.cue(id)`, `.line(id)`, `.word(id, k)`, `.mouth(SPEAKER, T)`, `.speaking(SPEAKER, T)`.
- `vx/post.py` — the global film look (bloom, CA, vignette, grain, fade, letterbox, grade). Applied by the runner.
- `vx/flag.py` — **owned by FlagSmith**. `Flag(...).simulate(n, pole_fn, wind_fn) -> FlagTrack`; `track.draw(ctx, i)`; `draw_flag(...)` for crowds.
- `vx/chars.py` — **owned by RigMaster**. `Character(kind).draw(ctx, x, y, h, pose, t, mouth=, look=, expr=, facing=)` → anchors
  (hand positions, `pole` = (base_x, base_y, ang) for holding a flag); `.anchors(...)` pure; `draw_crowd(ctx, people, t)`.
- `vx/foley.py` — `export_track(S, name, energy, pan, kind)` per-frame motion energy and
  `export_events(S, name, [dict(t=GLOBAL_s, kind=, strength=, pan=, desc=)])` → `cache/foley/` for the sound team.
- `vx/render.py` — the runner (usage in its docstring):
  - `python -m vx.render s03 --at 1,4.5,9` → stills + contact sheet in `out/stills/` (LOOK at them with the read tool!)
  - `python -m vx.render s03 --scale 0.5 --step 2 --workers 4` → quick preview video in `out/preview/`
  - `python -m vx.render s03 --workers 4` → final `out/scenes/s03.mp4` (1080p24)

**Scene module contract** (`scenes/sXX_name.py`):
```python
from vx import *
from vx.flag import Flag, draw_flag
from vx.chars import Character, draw_crowd
POST = dict(bloom=0.45, grain=0.03, vignette=0.28)   # optional
def setup(S):          # S.id, S.start, S.end, S.dur, S.n, S.f0, S.fps, S.s, S.cache, S.tl
    ...                # deterministic precompute (cloth sims etc.). Cache heavy results in S.cache.
    return state
def render(fc, state): # fc.f/t local frame/seconds, fc.F/T global, fc.p progress, fc.s scale, fc.w/h pixels
    cv = Canvas(fc, bg="night"); ...; return cv.rgb()   # float32 (fc.h, fc.w, 3)
def post(fc, state): return dict(fade=...)   # optional per-frame film-look overrides
```
Rules: render must be a pure function of (fc, state) — frames render in parallel, out of order.
Draw in design units (1920×1080) so `--scale` works; numpy per-pixel work must use `fc.w, fc.h, fc.s`.
Budget ≲ 1.5 s/frame at 1080p (numba is available for heavy per-pixel/particle work).
CPU etiquette: 16 cores shared by ~14 agents — use `--workers 4` max, prefer stills and `--scale 0.5` previews.

---------------------------------------------------------------------------------------------------
## 4. Audio contract

All stems: 48 kHz, stereo, float32 WAV, exactly 193.0 s (9,264,000 samples), aligned to T=0.

| Stem | Owner | Content |
|---|---|---|
| `audio/stems/dialogue.wav` | VoiceFX | every timeline line, processed per `cast.fx` + scene acoustics, placed exactly |
| `audio/stems/crowd.wav` | VoiceFX | walla & babble: slop whispers, background crackpots, Babel cacophony (hard stop 138.5), "Glorious!" |
| `audio/stems/choir.wav` | Choirmaster | the Unbabeling convergence (babble → one chord, 150–154.5), the sung hymn 154.5–165.93, sustained "Flags" choir tails |
| `audio/stems/music.wav` | Composer | score |
| `audio/stems/sfx.wav` + `ambience.wav` | Foley | hard effects + physics-derived flag flutter + ambience beds |

Main owns the final mix (`tools/mix.py`: ducking, bus compression, −14 LUFS, −1 dBTP) and final assembly.
Dry reference: `audio/stems/dialogue_dry.wav`. Lip-sync envelopes: `audio/voice/mouth.npz`.
SILENCE at 138.5–140.0 is sacred: every stem must be (near) silent there except a faint wind in the last 0.5 s.

---------------------------------------------------------------------------------------------------
## 5. Coordination

- Own only your files. Never edit another agent's files — message the owner via `hub` (`op:"send"`).
- Library owners (FlagSmith, RigMaster) keep public signatures stable and broadcast (`to:"all"`) when a
  new version lands. Scene owners code against the documented API immediately (stubs already work).
- Scene owners export sound hits with `vx.foley.export_events` and keep them current; the sound team reads them.
- No project-wide test suites, linters or formatters. Verify your own work visually (stills + contact sheets).
- When done, report: files written, how to render, render time per frame, known issues.

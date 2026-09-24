# THE FLAG GAME? — Score cue sheet (film 2, Composer)

Stem: `films/flaggame/audio/stems/music.wav`: 48 kHz, stereo, float32, exactly 199.0 s (9,552,000 samples), aligned to T=0.
Rebuild (after `source films/flaggame/env.sh`):
- `python films/flaggame/tools/fg_music_score.py` renders all sections plus the master (about 1 minute).
- `--only t04,t05` re-renders only those sections; `--master` masters only.
- Check with `python tools/music_verify.py [--spec all]`.

Shared engine (both films):
- `tools/music_lib.py`: DSP and sampler. It is film-aware: DUR and N come from the active film's timeline via `vx.config`.
- `tools/music_orch.py`: orchestration and mastering helpers, split out of film 1's score. Film-1 output is unchanged: a bit-exact A/B was run with a fixed hash seed.
- Samples: VSCO2-CE, plus the new VS Upright Nr. 1 piano. Soundfonts: MuseScore General and GeneralUser GS (via `tools/music_fetch.py` and `tools/music_samples.py`).

Every time below is global seconds. Rows marked ◆ are named timeline cues. Rows marked ◇ follow scene exports in `films/flaggame/cache/foley/`; the score re-reads them on every render. Every sync point the score places is logged in `films/flaggame/audio/music/cues.json`.

## Palette

An evangelical-revival palette, played absolutely straight:
- harmonium (reed organ)
- a real pipe organ (VSCO2 Rode "loud" and NT5 "quiet" manuals and pedals)
- a Hammond-style tonewheel organ for gospel moments
- a VS upright piano
- choir "aah"/"ooh" (from the SoundFonts)

For the Doré flashback it becomes a Cecil-B.-DeMille biblical orchestra: strings, horns, trumpets, trombones, tuba, timpani, taiko, gong, harp and celesta. There are also three novelty colours:
- **pharma-ad bed:** nylon guitar and a warm pad
- **J-pop kira-kira:** glock gliss, crystal and tinker-bell presets
- **sitcom sting:** slap bass and rimshot

## Tonal plan

The tonal centre is **D** for the whole film:
- **D major:** the tract world, the Flag Maker and the miracles.
- **D minor:** the storm, the FLAGS chant, the Devil, the burn and BLASPHEME.
- **E♭ major:** the altar-call hymn, a semitone "lift" in the Vocalist's arrangement.
- The ending returns to D, so the last chord continues into the first chord of the loop.

## Leitmotif: "the Altar Call"

The incipit of **WOODWORTH** (W. B. Bradbury, 1849, public domain), the tune of *Just as I Am*: **D–E | F♯ – F♯ | A – G F♯ | E – F♯ G | F♯** (a pickup, then 3/4). It is harmonised D–D–D–A–A–D and appears as:

| where | form |
|---|---|
| t01 0.62 | a lone harmonium in the rain, cut off mid-phrase by the tract's gospel-radio sting |
| t04 67.46 | the Flag Maker's organ voluntary (flute stop over a soft organ and choir "ooh") |
| t05 108.79 | a hopeful pizzicato march (clarinet, then flute) for finding and moving the flags |
| t05 113.4 / 119.9 | horns under the Seraphlag's ascent, then trumpets over the firmament glory |
| t06 129.73 | the fragile ascending line of the freed Flag (solo violin and celesta, climbing out of sight) |
| t07 143.94 | a fast-forward Sousa-style march: the motif on trumpet and piccolo, advancing one note per two raisings |
| t07 150.08 | a harmonica on the lonely road, with a fiddle drone |
| t08 169.09 | the hymn itself (the Vocalist's E♭ arrangement, doubled by the organ) |
| t09 196.62–197.40 | the gospel "A-men" that lands the tract in the mud and loops the film |

Also quoted: J. S. Bach, **Toccata in D minor, BWV 565** (public domain), for the Devil. The fast-forward march in t07 is an original Sousa-style march built on the leitmotif; no Sousa tune is quoted.

## The loop

The score's buffer is circular: any note or reverb tail that rings past 199.0 continues at 0.0. The final gospel D-major chord (197.40) therefore carries on under the harmonium's D-major chord at the start of the film. At the seam, the last 50 ms measure −27.7 dB RMS and the first 50 ms −28.0 dB RMS. The waveform is continuous across the join.

## Cue list

### t01 — The Tract (0.00–18.46), D major
| t | cue |
|---:|---|
| 0.00 | harmonium D chord in the rain (the previous loop's Amen still ringing) |
| 0.62 | **leitmotif** on the lone harmonium (D–E–F♯…) |
| ◆ 2.60 | tract_flip: **gospel-radio sting**, heard through a small speaker: a Hammond gliss into I–IV–I (D6/9 → G/D → D), with choir, piano and tambourine |
| ◆ 4.90 | pages_riffle: a flutter of harp notes |
| ◆ 6.40 | dive_in: a string swell and a harp gliss that lands as the comic comes alive; then Sunday-school piano in 3/4, very soft |
| ◆ 10.09 | flag_passes: a celesta and glass shimmer as the only colour passes |
| 15.4 | a cheeky bassoon hiccup after "LAZY and ENTITLED HIPPY" |

### t02 — Every Word Was Wrong (18.46–43.29)
| t | cue |
|---:|---|
| ◆ 18.46 | preacher_appears: an organ "dun!" and a bass pizz |
| 19.0–21.6 | church-social stride piano (G major, 2/4 at 120 BPM) that stops for the punchline |
| ◆ 22.12 | wrong_burst: **dramatic organ stab**. Full D minor chord with 32′ pedal, plus timpani, low brass and a crash |
| 23.1–25.2 | silent-film diminished tremolo under "WHAT? HOW CAN THIS BE???" |
| ◆ 25.20 | icon: a Byzantine ison (low male "ooh" drone on D), a tubular-bell toll and soft organ |
| ◆ 29.29 | essence: **heavenly choir-pad radiance** in D major, with harp gliss, glass and high strings |
| ◆ 31.06–33.32 | disclaimer: a bland **pharma-commercial bed** (nylon-guitar arpeggios and a warm pad) that cuts off with the footnote |
| ◆ 33.75 | anime: **J-pop kira-kira** (glock gliss, crystal Dmaj7 chime, celesta sparkle, tinker bell, a tiny brass stab); a falling pizz for the sweat drop |
| 36.75–40.0 | conspiratorial tiptoe in G minor: cello pizz and a low clarinet |
| 40.0 | a warm string swell into… |
| ◆ 41.34 | flashback: the **harp-gliss ripple** (up, down, up), a celesta chime and a wavy, detuned pad |

### t03 — In the Beginning (43.29–67.46), D minor
| t | cue |
|---:|---|
| ◆ 43.29 | genesis: a **Philip Glass–style arpeggio machine** (tonewheel organ, eighths at 150/min, bars of 1.6 s, Dm–B♭–Gm–A with rotating additive patterns); cello pedal, then marimba (47.0), flute (51.0) and bass (53.0) |
| ~55.3 | the machine darkens into a storm ostinato (spiccato D minor) whose grid lands on 57.339 and 59.339 |
| ◆ 57.34 | stench: the **stench motif**. A tuba and bassoon stab, then a lurching chromatic sneer (E♭–D–C♯ with pitch sags), a muted-trumpet "wah" and a trombone "pew" slide |
| 57.9–59.34 | the stink twists into the sky: tremolo strings, horns, a timpani roll, a Shepard whirl and a cymbal swell |
| ◆ 59.34 | hurricane: D minor tutti with gong, then whirling chromatic violin runs and tremolo |
| ◆ 62.95 | rain_begins: **downpour hit on "RAIN"** (tutti, full organ, a descending harp gliss) |
| 63.4–67.46 | a storm bed receding under the Ranger's radio; a bassoon "blup" after "where my dirt went?" |

### t04 — The Flag Maker (67.46–105.67)
| t | cue |
|---:|---|
| ◆ 67.46 | flagmaker: **sacred D major**, soft organ and choir "ooh", with the leitmotif on a flute stop |
| 74.73 | "Most people rejected His message": B minor, low strings |
| ◆ 76.98 | shut_up: **stab** (C minor full organ, brass, timpani, giant drum and rimshot) |
| 78.0–87.1 | solemn organ Em–A–D–G under P12 and M02 |
| ◆ 87.15 → 95.72 | flags_flood: the **chant machine**, locked to the crowd's grid (`grid.json`: 126 → 154.1 BPM, 20 beats, downbeats 87.153 / 89.013 / 90.790 / 92.496 / 94.138 / 95.722). An organ pedal pulse on every beat, spiccato eighths, taiko and timpani on downbeats, a snare from bar 3 and horns from bar 4. Harmony Dm–Dm–B♭–A |
| ◆ 95.72 | flood_peak: tutti, full organ and gong; the frenzy continues at 154 BPM until the Devil |
| ◆ 97.30 | devil: **Bach, Toccata in D minor**. Opening mordent phrase on full organ in octaves, with timpani; second statement 8vb at 98.95 |
| 99.80 / 100.59 / 101.34 | the organ tries the third statement on each of the Devil's "a…" and chokes, each time lower, weaker and flatter |
| ◆ 102.42 | devil_glitch: **possession hit**. A diminished-7th full organ over D, a choir shriek, brass, timpani and crash, with a buffer-stutter glitch and a riser into it |
| 102.87 / 103.32 / 103.83 | a D minor organ stab plus giant drum on each "FLAGS!" |
| 104.1–105.67 | a held, glitching D minor into a **hard stop at 105.67** (reverb cut too) |

### t05 — And It Worked (105.67–122.29), D major
| t | cue |
|---:|---|
| ◆ 105.67 | some_listened: hush, then a **single celesta A5** (with a glass double) on the white card |
| ◆ 108.79 | found_one: a **hopeful pizzicato march** (2/4 at 112 BPM: bass pizz, viola off-beats, triangle, the leitmotif on clarinet then flute); horns on each "Move it!" |
| ◆ 111.86 | it_worked: **triumphant fanfare** (D major tutti, trumpets, horns, trombones) |
| 113.15–116.9 | the Seraphlag's ascent: rising harp arpeggios, a string crescendo, choir "aah", the leitmotif in horns |
| ◆ 116.90 | flag_on_effigy: arrival chord (horns, strings, timpani, glock) |
| ◆ 118.92 | rain_freeze: **everything stops** (dry sound and reverb tails) except a suspended high chord (glass and string harmonics A5–D6–E6–F♯6–A6) |
| ◆ 119.56 | firmament_reveal: **full choir and organ glory**: organ with pedal, choir "aah", strings, a tutti, cymbal swell and harp gliss; the leitmotif in trumpets |

### t06 — The Firmament Opened (122.29–142.04), D minor
| t | cue |
|---:|---|
| 122.44 | **burn-night drums** at 123.49 BPM (`grid.json`, shared with the crowd's whoops): taiko, big hand drum, tenor drum |
| 124.0–125.36 | fire swell (tremolo, horns, riser, cymbal) |
| ◆ 125.36 | ignite: **fire hit** (D minor tutti); the drums get louder, with a low-brass fire ostinato D–D–F–E |
| ◆ 129.73 | flames_reach_flag: **the drums drop dead**. The Flag's ascent is a **fragile ascending line** (the leitmotif on solo violin and celesta, climbing on to D6 and out of sight, with rising glass) |
| 132.3–135.5 | dread: a low tremolo cluster rising, low brass |
| ◆ 135.58 | firmament_crack: **apocalyptic crack** (brass cluster, dissonant full organ, tutti, a cascade of ~90 glass shards across the stereo field) |
| ◆ 137.12 | deluge: **catastrophic tutti** (full organ, choir, trombones, tuba, trumpets, strings sinking in pitch, a timpani roll, a falling Shepard tone, gong), held to… |
| 142.04 | …the **smash cut** (reverb cut too) |

### t07 — Raised Again (142.04–159.83)
| t | cue |
|---:|---|
| 142.33 | a "HOWEVER!" brass stinger and timpani; then a snare roll |
| ◆ 143.94 → 145.14 | raise_1 → 50: the **fast-forward Sousa-style march**, locked to T07Raised's 100 exported raisings. A **snare on every raising**, which becomes a roll as they crowd to 5 ms apart. A **cymbal on every full-frame hero cut** (#1, 2, 3, 5, 8, 13, 21, 34). The band runs on its own clock that follows the raisings but never plays faster than a real band: tuba and bass drum, horn chords, the leitmotif on trumpet and piccolo |
| ◇ 145.14 | "50!" lands (brass stab and timpani), then the band holds its breath (tremolo) over the 50-panel page; a snare roll leads back into… |
| ◇ 145.96 → 146.77 | …the second run (#51 on "then"), accelerating again to the 100th |
| ◆ 146.77 | raise_100: **tutti peak** with full brass |
| ◆ 149.58 | spirit: a **lonely Americana walk**: harmonica (the leitmotif), a fiddle double-stop drone, sparse guitar chords |
| ◆ 155.45 / 157.50 | no_fire: B minor. no_burns: G minor, a plaintive harmonica |
| ◆ 158.80 | unmask: a **warm resolve** (G minor → D major, the minor plagal), strings, harp and celesta |

### t08 — Blaspheme (159.83–181.17)
| t | cue |
|---:|---|
| 159.9 | a grave organ pedal on D in the rain; tremolo building from 163.0 |
| ◆ 166.22 | blaspheme: **thunder-organ** (full D minor organ with a 32′ sub, tutti, gong) |
| 167.7 | B♭7 under "What must I do?" (the dominant of E♭) |
| ◆ 169.09 | hymn: **the organ doubles the Vocalist's altar-call hymn exactly** (E♭ major, 3/4, beat 0.525714 s, `audio/vocal/hymn_grid.json`): S/A/T/B on the quiet manual, the bass 8vb on the pedal, soft strings on soprano and bass, a harp on the pickup |
| ◆ 180.13 | amen: the plagal A♭ → **E♭ "A-men"** lands on Summer's "Amen." The sunbeam is a harp roll and glass |

### t09 — Pass It On (181.17–199.00)
| t | cue |
|---:|---|
| 183.44 | "Now go…": an A7 pad (leading back to D) |
| ◆ 184.43 | pass_it_on: **tender resolve** (D major strings, piano and harp) |
| ◆ 185.40 → 191.60 | droste: an **endless Shepard–Risset rise** (a synth Shepard tone, plus a chromatic tremolo Shepard scale across the string sections and a rotating celesta spiral), which resolves on… |
| ◆ 191.60 | droste_end: …a bright D-major morning chord (strings, piano, glock) |
| ◆ 194.76 | callback: **sitcom sting** (slap-bass riff D–D–A–C–D, rimshots, a horn button) |
| 196.30 | tract_toss: a falling harp gliss |
| ◆ 197.40 | tract_lands: **gospel "A-men"**. G ("A-", 196.62) → D ("men", 197.40) on Hammond, pipe organ, piano and choir. It rings through 199.0 into the loop, together with the Vocalist's "FLAGS" amen in D (vowel at 197.40) |

## Coordination
- **Vocalist (crowd and choir):**
  - the chant and burn grids are mine (`films/flaggame/audio/music/grid.json`) and the crowd sings on them
  - the hymn arrangement is the Vocalist's (E♭), and the organ follows `audio/vocal/hymn_grid.json`
  - the final "FLAGS" amen is in D, with its vowel at 197.40
- **Foley:** Foley owns the sub (< 60 Hz) and the noise impacts; my hits are tonal (the percussion bus is high-passed at 55 Hz). I sent Foley the keys per section. Foley does an unpitched Shepard noise under my pitched Droste rise.

## Verification (`python tools/music_verify.py`, final render)
- **Format:** 48 000 Hz, 2 ch, 32-bit float, **9 552 000 frames** (199.000 s).
- **Peaks:** sample peak −1.21 dBFS, **true peak −1.20 dBTP**, no clipping. A 3:1 "glue" compressor (15 ms attack) and a look-ahead limiter tame the hits.
- **Integrated loudness:** **−20.1 LUFS** (stem reference level; Main's mix sets the final level).

  | t01 | t02 | t03 | t04 | t05 | t06 | t07 | t08 | t09 |
  |---|---|---|---|---|---|---|---|---|
  | −27.5 | −23.5 | −18.9 | −22.3 | −17.2 | −15.2 | −17.1 | −23.8 | −27.3 LUFS |

  Music sits 5 dB lower in 250 Hz–4 kHz while a line is spoken; Main side-chain ducks as well.
- **Hit sync** is measured as the spectral-flux onset on the final stem (librosa: n_fft 1024, hop 1.33 ms). **All 41 named cues** checked are within **±13.3 ms**.

  | cue | error | cue | error | cue | error |
  |---|---:|---|---:|---|---:|
  | tract_flip 2.60 | +9.3 | stench 57.339 | +5.3 | flag_on_effigy 116.901 | +5.3 |
  | dive_in 6.40 | +2.7 | hurricane 59.339 | +5.3 | rain_freeze 118.917 | +8.0 |
  | flag_passes 10.093 | +5.3 | rain_begins 62.951 | +5.3 | firmament_reveal 119.555 | +5.3 |
  | preacher_appears 18.46 | +5.3 | flagmaker 67.46 | +13.3 | ignite 125.356 | +6.7 |
  | wrong_burst 22.117 | +2.7 | shut_up 76.98 | +8.0 | flames_reach_flag 129.729 | +8.0 |
  | icon 25.20 | +12.0 | flags_flood 87.153 | +9.3 | firmament_crack 135.575 | +4.0 |
  | essence 29.288 | +6.7 | flood_peak 95.722 | +8.0 | deluge 137.118 | +5.3 |
  | anime 33.75 | +2.7 | devil 97.30 | +9.3 | raise_1 / raise_100 | +6.6 / +8.0 |
  | flashback 41.335 | +6.7 | devil_glitch 102.415 | +4.0 | spirit / unmask | +10.7 / +6.7 |
  | genesis 43.29 | +9.3 | some_listened 105.67 | +6.7 | blaspheme / hymn / amen | +2.7 / +8.0 / +6.7 |
  | found_one 108.79 | +5.3 | it_worked 111.86 | +6.7 | pass_it_on, droste_start/end, callback, tract_lands | +9.3, +5.3/+5.3, +8.0, +8.0 |

# THE UNBABELING — Score cue sheet (Composer)

Stem: `audio/stems/music.wav` — 48 kHz, stereo, float32, exactly 193.0 s (9,264,000 samples), aligned to T=0.
Rebuild: `python tools/music_fetch.py` (one-time: sample libraries) → `python tools/music_samples.py` (one-time:
sample index) → `python tools/music_score.py` (≈2–4 min) → `python tools/music_verify.py [--spec all]`.
All times are GLOBAL seconds. Sync points marked ◆ are named timeline cues; ◇ are picture events taken live from the
scene owners' `cache/foley/<scene>_hits_events.json` (the score re-reads them on every render, so it follows the cut).
Every sync point written by the score is logged with its time in `audio/music/cues.json`.

## The leitmotif — "the Flag"

The hymn's soprano line, **A – F♯ – B – G – C♯ – A | D** (scale degrees 5-3-6-4-7-5-1 in D major): falling
thirds that climb by step and land on the tonic, harmonised **D | D | G | Em | A | A | D** (exactly the hymn's
chords). The whole film is the theme searching for the hymn it came from:

| Scene | Transformation |
|---|---|
| s01 | innocent: glass + celesta, 6/8 lilt at the hymn's own pulse (dotted quarter ≈ 82 BPM); final D lands on the snap |
| s01 title | the letters THE UNBABELING drop as plucks: T-H-E = D major triad, U-N-B-A-B-E-L = the theme, I-N-G = triad 8va |
| s02 | corrupted: D minor (A–F–B♭–G–C–A–D), seasick FM synth + sick solo violin, detuning, six-voice canon drowned by the surge |
| s03 | parody: presidential dotted-rhythm pomp on muted trumpet, all of it heard through the CRT |
| s04 | mystical: the Hypermind's arpeggiator is built from theme cells; on the hush the theme returns on glass, pp |
| s05a | martial: D Dorian (A–F–B♮–G–C–A–D) in horns + trombones over a 3+3+2 spiccato/taiko ostinato |
| s06 | Babel: stretto canon — the theme enters in all twelve keys, each voice looping at its own tempo |
| s07a | solo violin ("Each of them alone") → violin section ("All of them together") → the final D lands exactly on the hymn downbeat |
| s07b | the hymn itself (sung by the Vocalist's choir), then the theme augmented in trumpets/horns through the Gateless Gate |
| s08 | quoted on celesta after "all Flags are one Flag!", a warm horn statement under N11, innocent glass once more under the end card |

## Instrumentation

- **Real orchestra** — Versilian Studios VSCO-2 CE (CC0) multisamples, indexed by pitch analysis
  (`tools/music_samples.py`: every sample's f0 is measured and cents-corrected to exact A=440 ET; attack onsets detected):
  violin/viola/cello sections (sustain vib, tremolo, spiccato, pizzicato), contrabass, solo violin, harp, F horns,
  trumpets (open + straight mute), tenor trombones, tuba (sustain + staccato), flute, piccolo, oboe, clarinet,
  bassoon, timpani (hits + rolls, principal-mode tuned), glockenspiel, marimba, xylophone, pipe organ (Rode loud +
  NT5 quiet manuals and pedals), cymbals, suspended cymbal rolls, gong, giant ethnic drum (taiko role), marching
  snare, tenor drum, triangle, flexatone.
- **SoundFont** (MuseScore General, via tinysoundfont, sample-accurate): celesta, tubular bells, choir "aah"/"ooh",
  reed organ (harmonium).
- **DSP synthesis** (`tools/music_lib.py`): D1 sub drones and 32' organ pedal, FM glass/bells, glass harmonica stars,
  supersaw pads, Shepard–Risset glissandi (rising, falling, accelerating), noise risers, reverse-reverb swells,
  tape stop, stutter/glitch, a synthetic violin natural harmonic with bow noise, lo-fi "feed" jingles.
- **Space**: synthetic multiband convolution reverbs (hall 2.8 s, cathedral 6.5 s for organ/choir, void 9 s for
  glass/stars, plate, room), orchestral seating pans, per-bus EQ; reverb tails are *cut* at the dramatic cuts
  (72.70 hush, 90.00, 100.55, 138.50, 154.50 space) so silences are real.
- **Dialogue space**: every note is written around the locked lines, plus a dynamic 250 Hz–4 kHz dip (−5 dB) driven
  by the timeline's line spans. Main additionally side-chain ducks.
- Tonal hits only: Foley owns the sub-boom (< 60 Hz) and noise debris on the big cues.

## Cue list

### s01 — Rotating Into Existence (0.0–21.0), D major
| t | cue |
|---:|---|
| 0.00 | D1 sub hum + dark open-fifth supersaw pad fade in; glass-harmonica "stars" (D pentatonic, random pans) |
| 0.4–2.60 | reverse glass shimmer condenses into N01 (the Crystal forms) |
| 3.00–11.75 | rotating crystal arpeggio (4 notes = tesseract), circling the stereo field, accelerating from 5.9 (W-plane turn) |
| 5.90–12.00 | Shepard–Risset glissando ("a direction you cannot point to"), accelerating, resolves onto D partials at 12.0 |
| ◆ 12.00 | **Flag theme, innocent**: glass + celesta A5 12.00, F♯ 12.49, B 12.73, G 13.22, C♯ 13.47, A 13.96 — harp chords on D/G/Em/A |
| ◆ 14.20 | final D ON the snap: bloom (strings, harp gliss to D7, glock), "a sign of itself" = the last figure reflected 4× smaller & higher |
| 15.30–16.00 | reverse-reverb chord + suspended cymbal suck |
| ◆ 16.00 | **TITLE HIT**: D major tutti (staccato brass & strings, timpani, taiko, crash, glock) → sustained D major |
| ◇ 16.46–17.20 | one pluck (celesta + marimba + pizz) per title letter (13 letters, times from Scene01) |
| ◇ 17.02 | harp glissando with the "Die Entbabelung" wipe |
| 18.00 | theme 2 octaves up on celesta/glock/pizz |
| 20.30–21.00 | page slides up: harmony darkens D → Dm, low brass + timpani roll into s02 |

### s02 — The Age of Slop (21.0–47.5), D minor, 95.24 BPM (beat 0.63 s; bars 21.00/23.52/26.04/28.56/31.08)
| t | cue |
|---:|---|
| 21.00 | the machines of meaning: 3+3+2 spiccato ostinato (cellos/basses/violas), low brass chords Dm–B♭ |
| ◇ 25.20 / 26.20 / 26.87 | monolith stabs on "Nations" (Dm) / "Faiths" (B♭) / "Philosophies" (Gm) |
| 28.56–33.60 | lockstep: march snare backbeats; ◇ radio pings on Scene02's broadcast-ring pulses; tremolo + horn crescendo… |
| ◆ 33.60 | …CUT on the downbeat ("Then came the slop."): low cello D + eerie high B♭ |
| 34.80–39.20 | **Flag theme corrupted** (D minor, detuned FM + seasick solo violin) in a 6-voice canon (transposed −1, +6, +5, −3, +1), glitch stutters, drowned by rising tremolo cluster, accelerating Shepard, noise riser, timpani roll, suspended cymbal |
| 39.155 | the surge stops dead 45 ms before the impact (micro-silence so the crash punches) |
| ◆ 39.20 | **SLOP CRASH**: D-minor braam with E♭ cluster (trombones, tuba, horns, strings), timpani, giant drum, stick-cymbal edge (Foley owns the crash noise + sub); the whole chord sinks −3 semitones as the memeplexes go under |
| 40.30–47.30 | slow low string "waves" under N07 |
| ◇ 42.45–47.45 | feeds multiply with the split-screens (1 → 4 → 16 → 48 → 96 lo-fi jingles in random keys/tempi, phone-band), then smear into static |

### s03 — The Noospheric Munitions Act (47.5–61.5), everything THROUGH the CRT
| t | cue |
|---:|---|
| ◆ 47.50 | presidential fanfare tuning in through the static (triplet pickups, pitch warble, narrow band) |
| ◇ 48.02 | fanfare chord lands as the vertical hold locks |
| 48.30–57.00 | patriotic bed: snare roll, horn/string chords D–G–D–A–Bm–G–A, muted trumpet plays the theme in dotted pomp |
| ◆/◇ 57.00 | tutti stab on "banned!" (fist slam) |
| ◆ 57.60 | pen flourish: harp glissando + glock |
| 60.19 | "…a sacrament": harmonium plagal Amen (G → D) |
| ◆ 60.90 | **KA-CHING** brass "DA!" button + glock sparkle + triangle |
| 61.00–61.45 | the CRT dies: tape stop to silence |

### s04 — Flagartha (61.5–90.0), 85.7 BPM (beat 0.70; bars 61.5/64.3/67.1/69.9), D pedal
| t | cue |
|---:|---|
| 61.50 | 32' + 16' organ D pedal, quiet organ chords D–Bm–G–A over it, low "ooh" choir |
| 62.20–71.20 | the Hypermind's arpeggiator (16ths, theme cells), filter opening as it wakes, dotted-8th echo |
| 63.00 | reveal: harp glissando + glock |
| ◇ 65.50–67.25 | six canon scrolls absorbed: tubular bells + glass rising D–E–F♯–A–B–D |
| 70.11 | "true": glass D major chord |
| ◇ 71.30–72.15 | six descending bells on the six plates; riser to the beam |
| ◆ 72.70 | **HUSH**: everything (including reverb) cuts away; the Flag theme on glass + celesta, pp, over high string harmonics |
| 79.20 | "Why break it further?": B minor → G → Em strings |
| 83.40 | C02: organ + strings; bass falls on "fall", rises on "rise" |
| ◇ 87.85 | the pole-butt strike on "Summon": D minor, timpani; tremolo crescendo, taiko accelerating, ◇ horn klaxons on the gold sweeps 88.40/88.85/89.30 |
| ◆ 89.40 | **BRAAM** on "schismmancers" (D minor, brass + strings + timpani + giant drum); suspended cymbal suck to the cut |
| ◆ 90.00 | hard cut (all s04 sound incl. reverb ends) |

### s05a — The Schismmancers (90.0–100.5), D Dorian, 114.29 BPM (beat 0.525; bars 90.0/92.1/94.2/96.3)
| t | cue |
|---:|---|
| ◆ 90.00 | gear-up downbeat: taiko + rimshot + timpani + low brass; 3+3+2 16th spiccato ostinato, taiko accents, tenor drum ghosts, rimshots on 2 & 4 |
| 90.00–98.40 | **Flag theme, martial** (Dorian) in horns + trombones, 2 beats per note; snare roll into the stack-up |
| 98.40 / 98.70 / 99.02 | "Breach! Breach! Breach!": tutti stabs Gm / A / B♭ (rising bass G–A–B♭) |
| 99.02–99.70 | C major riser: string glissando up, horns, suspended cymbal, noise |
| ◆ 99.70 | **BREACH**: D major blaze (Picardy after all that minor), brass + strings + choir "aah" + harp + glock |
| 100.55 | clean hand-off into the map (cut incl. reverb) |

### s05b — Recursive Balkanization (100.5–125.5): stingers divide the octave in three (D → F♯ → B♭)
| t | cue |
|---:|---|
| ◆ 100.60 | **NATION** stinger, D minor |
| ◇ 100.77–102.64 | fractal pizzicato/marimba/xylo/harp: generation g of Scene05b's border cracks plays 2^g notes |
| 103.20–105.03 | tiny oom-pah anthem for the Republic of Me (tuba + pizz), snare roll into… |
| 105.03 | …"Me!": piccolo + trumpets "ta-DA" + crash |
| 106.96 | after "Population: one!": a lonely triangle + one pizz |
| ◆ 107.80 | **FAITH** stinger, F♯ minor + organ; ◇ at each floor-schism generation (107.90/108.25/108.73) every organ voice splits into a fractal cluster of detuned voices (1 → 2 → 4 → 8) |
| 110.68 / 111.95 | the heretics' organ duel: C minor stab, then F♯ minor (a tritone away) |
| 113.80 | "I am my own church!": harmonium Amen (B → F♯) |
| ◇ 114.30 | the flag is planted on the marble crown: soft timpani + cello pizz B♭ |
| ◆ 114.60 | **PHILOSOPHY** stinger, B♭ minor — ◇ the head bursts: ~170 glass fragments scatter across the stereo field into the void |
| 114.6–123.3 | the void: dark B♭-minor pad, a slowly *falling* Shepard, star glints |
| 120.85 | flexatone after "…angry at poets!" |
| 122.50 | "I am, therefore I am!" → celesta "I–am" echoing away |

### s06 — Babel (123.3–138.5)
| t | cue |
|---:|---|
| 123.30–138.50 | endless, accelerating Shepard–Risset rise (synth) + a sampled chromatic tremolo Shepard scale from 125.5 |
| ◇ 125.95–129.24 | **the Flag theme in stretto canon**: a new voice in a new key enters with each tier of the tower (horn, muted trumpet, oboe, clarinet, solo violin, organ, xylophone, trombone, flute, marimba, bassoon, glock — all 12 keys) |
| 123.3–138.5 | two grinding pulses: 114.3 BPM taiko/spiccato vs 95.2 BPM machine bass |
| ◇ 129.00 | lightning wakes Lord Egregore: low brass cluster + timpani + taiko |
| 130.2–135.8 | the chaos recedes under L03 (−4 dB: Lutie shouts over it) and C03 (−7 dB: Crocus is calm) |
| 133.6–136.1 | Crocus' flag: the one steady thing — a pure D (D5+D6) and organ D pedal |
| ◆ 136.00 | **BABEL CRACK**: cluster tutti; everything bends upward, riser, gong |
| 137.20–138.50 | the whole mix stutters (slices shrinking 110 → 25 ms) |
| ◆ 138.50 | **HARD STOP** — digital zero 138.500–140.000 |

### s07a — The Unbabeling (140.0–154.5), D major
| t | cue |
|---:|---|
| 140.00 | from digital silence: a single high string harmonic, **A6** — the theme's first note — slowly swelling |
| 141.00 | contrabass D pedal enters |
| ◆ 142.00 | the planted Flag: soft low D (harp, timpani, pizz) under Foley's thunk; ◇ harp unfurls to the snap-open 142.25 (celesta + glock catch the sun); dawn chord |
| 147.41 | celesta D on "…Flag." |
| 148.50 | distant horns as the survivors look up |
| 149.60 | **theme**: solo violin A ("Each"), F♯ ("alone") … violin section B ("All"), G ("together"), C♯, A … |
| ◆ 150.00 | convergence: plucks (harp, celesta, pizz, glass) one by one, then a flood (rate grows e^3.9), stereo spreading outward; D major swell, organ, timpani roll |
| 154.47 | a breath (everything lifts, incl. the harmonic) … |
| ◆ 154.50 | … and the theme's D lands **with the choir** on the hymn downbeat |

### s07b — Flagistan (154.5–173.0): the hymn, 84 BPM, D major
| t | cue |
|---:|---|
| ◆ 154.50 | hymn accompaniment: organ doubles the SATB exactly (+ pedal 8vb), violins I/II double soprano/alto 8va, cellos/basses the bass; strict ET, strict tempo; mp → mf → f → ff; tiny breaths before each phrase |
| 154.50 / 157.36 / 160.21 / 163.07 | timpani on each phrase ("Flags be / Flags are / Flags will / Flags."), harp chords on every chord |
| 160.21 | horns join (f) |
| 163.07 | final "Flags.": full brass, cymbal swell + crash, glock; timpani roll to the climax |
| 165.93 | embers: harp glissando + celesta sparkle as the flags lift off |
| ◆ 166.30 | **GATELESS GATE**: D major tutti + gong + choir "aah"; the theme augmented in trumpets/horns (violins/flute 8va): A 166.30, F♯ 167.10, B 167.55, G 168.35, C♯ 168.80, A 169.60, D 170.05 |
| 170.90 / 171.35 / 171.81 | the lift ♭VI–♭VII–I (B♭ → C → D; melody D–E–F♯) landing on "completed" |
| 171.81–173.00 | radiant fade: D major, tremolo violins, celesta shimmer, into the white-gold flash |

### s08 — The Sticky Flag (173.0–193.0)
| t | cue |
|---:|---|
| 173.00 | morning: flute, harp, soft strings |
| 173.48 | pizzicato run up the scale as Lutie runs in |
| 175.33 | after "For you.": celesta ta-da |
| 175.45–178.75 | light tiptoe bed under the first exchange: marimba on the beat, pizzicato off-beats, cello pizz (100 BPM, pp) |
| ◇ 177.45–178.90 | the sticky strand: a solo violin glissando stretches slowly up… |
| ◇ 178.90 | …snaps: pizz plink, then a droopy pizz + bassoon "blup" |
| 179.35–181.60 | the tiptoe bed returns under Lutie's protest… |
| 181.68 | …and stops: after "all Flags are one Flag!" the theme is quoted on celesta (+ pizz); then silence for the deadpan "This one is sticky." |
| ◇ 184.22 / 184.47 | Lutie sniffs her hand: two questioning clarinet staccati |
| 185.00 | N11: warm horn statement of the theme, strings + harp swell, timpani roll |
| ◆ 188.30 | **GLORIOUS!** final tutti; trumpet fanfare triplets + harp glissando ripple with the salute wave |
| ◆ 189.50 | **IVG end-card sting**: staccato brass, pizzicato tutti, timpani, triangle, glock |
| ◇ 190.10–190.41 | a pluck on each end-card letter F-L-A-G-S |
| 190.55–192.35 | the innocent theme on celesta + glass one last time over soft D major (final D 192.35 + harp) |
| 191.80–193.00 | the whole score (incl. reverb) fades with the picture; digital zero at 193.0 |

## Verification (`python tools/music_verify.py`, final render)

- Format: 48 000 Hz, 2 ch, 32-bit float, **9 264 000 frames** (193.000 s), T=0 aligned.
- **Silence 138.500–140.000: digital zero** (max |x| = 0). Reverb tails are cut at 138.5 too, so nothing leaks back at 140.0.
- No clipping: sample peak −1.21 dBFS, **true peak −1.20 dBTP** (4× oversampled look-ahead limiter; it works > 3 dB only on
  six hits, max 5.1 dB at 166.37).
- **Integrated loudness −19.9 LUFS** (stem reference level; Main's mix sets the final −14 LUFS). Per scene: s01 −21.3,
  s02 −24.3, s03 −26.8 (through the TV), s04 −24.9, s05a −16.7, s05b −23.5, s06 −18.9, s07a −24.2, s07b −16.9, s08 −18.0 LUFS.
  Momentary (400 ms) curve: `audio/music/cache/momentary_loudness.txt`; spectrograms: `audio/music/spec_<scene>.png`.
- Sub energy (< 60 Hz) in the 0.5 s after every Foley impact cue is 23–46 dB below the music's full-band level (hits are tonal; Foley owns the sub).
- **Hit sync** — spectral-flux onset (librosa onset strength, n_fft 1024, hop 1.33 ms; the flux peak = perceptual attack, it
  reads ≈ +5–8 ms even for a perfectly placed glass note) measured on the final stem, error vs cue:

| cue | t | error | | cue | t | error |
|---|---:|---:|---|---|---:|---:|
| flag_rotates_in | 12.000 | +8.0 ms | | nation_fracture | 100.600 | +4.0 ms |
| flag_snap_open | 14.200 | +8.0 ms | | faith_fracture | 107.800 | +1.3 ms |
| title_slam | 16.000 | +6.7 ms | | philo_shatter | 114.600 | +4.0 ms |
| slop_crash | 39.200 | +9.3 ms | | babel_crack | 136.000 | +6.7 ms |
| tv_on | 47.500 | +5.3 ms | | silence (cut) | 138.500 | +0.5 ms |
| ka_ching | 60.900 | +4.0 ms | | flag_planted | 142.000 | +5.3 ms |
| hypermeme (cut) | 72.700 | −11.5 ms | | convergence | 150.000 | +6.7 ms |
| schismmancers | 89.400 | +6.7 ms | | hymn | 154.500 | +9.3 ms |
| gear_up | 90.000 | +4.0 ms | | gateless_gate | 166.300 | +6.7 ms |
| breach | 99.700 | +9.3 ms | | glorious | 188.300 | +6.7 ms |
| | | | | end_card | 189.500 | +6.7 ms |

  Worst named-cue error: **11.5 ms** (all within ±15 ms). Hymn phrase timpani 154.50/157.36/160.21/163.07: +9.3/+5.3/+6.7/+6.7 ms.
  Every note is placed sample-accurately; slow-speaking samples are pre-rolled so their attack (min of their own flux
  peak and −20 dB point) lands on the beat; cymbal clashes are aligned by their bloom.

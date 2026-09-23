# Vocal stems — what is where (owner: Vocalist)

All stems: `audio/stems/*.wav`, 48 kHz, stereo, float32, exactly 193.0 s (9,264,000 samples), T=0 aligned.
Rebuild (deterministic, cached in `audio/vocal/cache/`):

```
python tools/vocal_dialogue.py   # ~10 s   -> dialogue.wav (+ audio/vocal/dialogue_lines/<ID>.wav, index.json)
python tools/vocal_choir.py      # ~5 min cold, seconds warm -> choir.wav (+ audio/vocal/choir_parts/, choir_notes.json)
python tools/vocal_crowd.py      # ~5 min cold, ~15 s warm -> crowd.wav
python tools/vocal_check.py dialogue choir crowd levels view   # verification -> audio/vocal/check_*.json, spectra.png
```

Code: `tools/vocal_lib.py` (Kokoro prosody injection, WORLD, PSOLA, procedural IRs), `tools/vocal_corpus.py`
(lore / slop / translations), `tools/vocal_dialogue.py`, `tools/vocal_choir.py`, `tools/vocal_crowd.py`,
`tools/vocal_check.py`.

## The trick: Kokoro with prosody injection
`vocal_lib.Kokoro.forward()` re-implements Kokoro-82M's StyleTTS2 forward pass so the duration predictor's
per-phoneme frames, the F0 curve (Hz, 80 frames/s) and the energy curve can be replaced before the
iSTFTNet decoder runs. Holding a vowel = giving its phoneme 100+ alignment frames and pinning energy;
singing = forcing F0 to a MIDI pitch with scoop + delayed vibrato; shouting = raising/expanding F0 and energy.
The model's natural leading silence must be kept (cutting `<bos>` frames destroys onset consonants).
Kokoro sings cleanly up to ~D4 (female) / ~A3 (male); above that it produces sub-harmonics, so higher
notes are rendered in the singer's comfortable register and WORLD-shifted up (formants preserved).

## dialogue.wav
Every line placed at `int(start*48000)` — the same sample as `dialogue_dry.wav`, so lip-sync is untouched
(timing verified: envelope cross-correlation lag vs dry = 0 ms on every line, see `check_dialogue.json`).
Processing (per `cast.fx` + scene):

| lines | treatment |
|---|---|
| NARRATOR s01 N01-N03, s07b N09-N10 | warm close-mic (HPF, +2.5 dB @190, air shelf, 2.5:1) + ethereal hall (RT 4.2 s) with an octave-up PSOLA "shimmer" ghost only inside the reverb |
| NARRATOR s02 / s07a / s08 | same voicing, dry plate (RT 1.0 @ -24 dB) / dawn plain (RT 1.6 @ -21) |
| LUTIE / CROCUS s04 | Flagartha stone chamber (RT 2.6, dense early reflections, -14 dB) |
| LUTIE / CROCUS s06 | rubble ledge in a gale: short outdoor verb, far slap off the tower (170 ms), slow wind-buffet gain wobble; L03 is shouted (PSOLA +1.5 st, drive, presence) |
| CROCUS C04 s07a, s08 L04-C06 | open air (dawn / meadow) |
| COMMANDER K01-K07 | tactical radio tuned for intelligibility (Main QA): 220-5200 Hz band, light drive, presence 1.8/2.6 kHz, hint of codec, faint carrier hiss; **PTT key-up click 0.155 s before each line, an inhale, and a squelch-tail burst + release click ~70 ms after each line** (Foley agreed not to double these) |
| JAGUAR J01-J02 | inside the broadcast: podium room -> broadcast comp -> boxy CRT speaker (200-6000 Hz, 430/1150/2700 Hz resonances, mild drive) -> flooded living room (RT 0.85, wobbling water reflection) |
| HYPERMIND H01-H02 | PSOLA-flattened dead monotone on D3 + octave-down double (D2) + ring-mod sidebands (D5) + feedback comb tuned to D3 + cavern (RT 6 s) inside the stone chamber; digital buffer-stutter on "…is **true**-tru-tru-t" (tail ends ~71.0, before H02) |
| CITIZEN1/2, HERETIC1-3 | shouted: PSOLA +2.5 st & contour x1.35, parallel drive, presence, 105 ms slap; X01/X02 open-air islands, X03-X05 vast temple (RT 3.4) |
| CRACKPOT1/2 | screamed (+2 st, contour x1.5, drive) and hurled into the void: 12 ping-pong repeats that darken and sag in pitch + RT 9 s void reverb (tails run ~12 s, under K07/s06) |

Pans (scene owners' on-screen x, applied x0.6): J01 .08, L01 -.3, C01 .3, L02 -.35, C02 .15, K02 -.35, K03 .25,
K07 -.4, X01 -.25, X02 .4, X03 -.35, X04 .35, X05 .05, X06 -.2, X07 .25, L03 -.35, C03 .2, C04 -.3,
L04 -.35→-.25, C05 .3, L05 -.3, C06 .3; everything else centre.
Level: every line normalized so its processed dry voice sits at -20 LUFS (radio -20.5, TV -21, shouts -18.5).

## crowd.wav
| time | content |
|---|---|
| 34.8-39.2 | slop whisper-chorus creeping in with the wave: slop phrases turned into TRUE whispers (WORLD resynthesis, all-aperiodic excitation), formant/time-smeared (up to 2.6x), a few backwards; density 1.6/s → 15/s |
| ~37.6-39.2 | reverse-reverb swell sucked into the crash |
| 39.2 | `slop_crash`: 40 whispers at once + slowed, pitched-down "as an AI language model" |
| 39.5-47.15 | swirling whispers circling the stereo field under N07 (lower), fade out by 47.15 for the TV static |
| 100.9-107.4 | nations-of-one: distant declarations across little seas (all languages), slap |
| 108.0-114.3 | cabals-of-one: whispered heresies and murmured prayers in the temple |
| 114.8-125.3 | crackpots screaming lore into the void, far, with echoes; densifying from 122.5 into the swirl |
| 125.5-136.0 | THE BABEL CACOPHONY: ~380 voice streams, all 54 Kokoro voices, 9 languages (EN US/UK, ES, FR, IT, PT, HI, JA, ZH) shouting Vexillomantic lore; density and shouting rise, speech speeds/pitches up ~11% into the crack; -3.5 dB dips under L03 (130.2-133.0) and C03 (133.8-135.7) |
| 135.45-138.5 | + 56 sustained screams ("No!/Why?/Me!/Hey!/Stop!/Mine!") rising 3-5.5 st — the roar at 136.0 |
| **138.5** | **razor HARD STOP**: 1.5 ms raised-cosine ending exactly at 138.5000 s, digital zero through 140.0 (asserted in code) |
| 188.3 | "GLORIOUS!": core = every voice once, ±25 ms around 188.30, centre ±0.6; ripple = second take of every voice rolling outward to the edges 188.4-189.0 (quieter, darker, wetter) + echo off the hills; fades by 193.0 |

## choir.wav
| time | content |
|---|---|
| 150.0-154.5 | THE UNBABELING CONVERGENCE: 32 voices (8 languages) speak lore; phoneme durations stretch (speech slows into chant) and F0 glides in the log domain from each voice's own speech contour onto a D-major chord spread D2-A5 (D2 A2 D3 F#3 A3 D4 F#4 A4 D5 F#5 A5), each voice ending on a held vowel; voices start scattered in the stereo field and gather to their register's side; murmur under N08 → full chord landing at 154.5; released by ~155.1 under the first hymn chord |
| 154.1-154.5 | the whole choir inhales |
| 154.5-165.93 | HYMN "Flags be. Flags are. Flags will. Flags." 84 BPM, SATB exactly as timeline.json, A=440 ET, 6 singers per part (S af_heart af_sky bf_emma bf_isabella af_aoede af_kore · A af_sarah bf_alice bf_lily af_jessica af_nova af_river · T am_michael am_eric am_puck am_liam bm_daniel am_echo · B am_fenrir bm_george am_onyx am_adam em_santa bm_fable), choral diction "Flahgs / bee / ah / will", vowel onsets on the beat (±15 ms jitter, ±8 cent detune per singer, vibrato 5.1-5.9 Hz after 300 ms), catch-breaths between phrases |
| 163.07-165.93 | final "Flags." swells (+5 dB, extra vocal effort) and cuts off with the z on 165.93 |
| 165.93-~168.5 | cathedral tail (RT 3.3 s), faded to nothing by 168.8 (Composer's bVI-bVII-I lift starts ~167) |
Dynamics: phrase 1 mp, 2 mf, 3 f, final ff. Staging: S left, A left-centre, T right-centre, B right.
`audio/vocal/choir_parts/{convergence,S,A,T,B}.wav` = dry section buses (sample 0 = 149.0 s) if the mix
wants to rebalance.

## Levels and notes for the mix
| stem | integrated | peak | section levels |
|---|---|---|---|
| dialogue.wav | -20.0 LUFS | -1.8 dBFS | every line's processed dry voice normalized to ~-20 LUFS |
| crowd.wav | -24.5 LUFS | -4.8 dBFS | slop 34.8-47: -25 · murmurs 100.5-125.5: -35 · Babel 125.5-138.5: -21 (roar 136-138.5: -18.5) · Glorious 188.2-190.5: -18 |
| choir.wav | -20.1 LUFS | -3.0 dBFS | peak moment = final "Flags." 163-166 |

- **Silence 138.5-140.0 is digital zero in all three stems** (crowd's last non-zero sample 138.49996 s; the
  Babel bus is razor-cut with a 1.5 ms raised cosine ending exactly at 138.5).
- Ducking suggestions: crowd under N07 (40.6-47.0) and under L03/C03 in s06 (already -3.5 dB dips baked in);
  murmurs in s05b are already low. Choir: the convergence starts as a murmur under N08 (149.6-151.9).
- Choir tail: -9 dB by 167.5, <= -25 dB by 168.3 (Composer: Em at 168.35 must not rub).
- Radio clicks/squelch are in dialogue.wav (Foley does not double them); CRT hum/whine are Foley's.
- Reverb/echo tails of any line are ducked 12 dB under every later line they overlap (e.g. the X07 void under K07); s05b murmurs dip 7 dB under every s05b line.

## Verification (`python tools/vocal_check.py ...`, results in `audio/vocal/check_*.json`)
- Format: all three stems 48 kHz / 2 ch / float32 / 9,264,000 frames.
- Dialogue timing: every line is placed on the same sample as `dialogue_dry.wav`; envelope cross-correlation
  processed-vs-dry = 0.0 ms on 39 lines, 2.0 ms on X05 (whole line and first-450-ms window).
- Dialogue intelligibility: faster-whisper small.en transcripts of the processed lines match the script
  (WER 0 on 30 lines; the rest are proper nouns, e.g. "Noah Spheric", "Skism actual", and are the same on the dry file).
- Hymn pitch: 168 sung notes (24 singers x 7), sustained-vowel median error mean 3.9 / max 10.5 cents vs A=440 ET
  (includes each singer's deliberate +-8 cent detune); section buses S/A/T/B all within -6..-1.6 cents on every note.
- Convergence chord peaks at 153.9-154.45: all 11 chord tones present, within +-8 cents (F#5 -18.6).
- Crowd: hard stop verified; "GLORIOUS!" onset 188.29 s; faster-whisper hears the 108-voice shout as "GORIOUS!".
- `audio/vocal/spectra.png`: spectrogram contact sheet of every event.

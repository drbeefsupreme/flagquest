# THE FLAG GAME? — vocal stems (owner: Vocalist)

Stems in `films/flaggame/audio/stems/`: 48 kHz stereo float32, exactly 9,552,000 samples (199.0 s).
Rebuild: `source films/flaggame/env.sh` then
`python films/flaggame/tools/fg_vocal_dialogue.py` (~10 s), `fg_vocal_crowd.py` (~5 min cold),
`fg_vocal_choir.py` (~5-10 min cold). Verify dialogue: `python tools/vocal_check.py dialogue levels`.
`tools/vocal_lib.py` is film-aware through vx/config.py (VX_FILM). Film 1's outputs are unchanged.

## dialogue.wav (-20.6 LUFS, peak -1.0 dBFS)
Each line is placed on the same sample as dialogue_dry.wav. All processing preserves timing: EQ, dynamics, PSOLA pitch shaping, reverbs and doubles after the dry onset, and pre-roll before it. Verified: 0 of 52 lines fail. Processed-vs-dry envelope lag is within ±10 ms on every line. Whisper word error rate is within 0.05 of the dry file's.
- PREACHER, present (t01/t02/t08/t09): fervent, open air under a camp shade (short canopy room). P02 and P25 are shouted. P06 'Alchemy IX…' has a wobbling echo tail for the flashback ripple.
- PREACHER, flashback P07-P23: a warm vintage church-filmstrip record. Close, low-mid warmth, soft top, tape grit, faint crackle and hiss.
- CAPS emphasis: every ALL-CAPS display word (from the lettering's `d` field) is lifted +1.3 semitones by PSOLA and +2.5 dB, with 40 ms ramps. Applies to PREACHER; SUMMER gets +1.0 st / +2 dB.
- t08 (from T08): P24-P26 outdoors in heavy rain under the shade; the altar call P27-S09 is intimate and close-mic (proximity warmth, shade reverb 7 dB lower); P30, after the rain stops, is dry sunny open air.
- SUMMER and RAVEN: camp shade in the rain. S05 and P08 (off-panel, from the present while the flashback freezes 49.40-52.3) are dry and close (shade room 8 dB lower), panned -0.55 (T03).
- FLAGMAKER: divine. Octave-down PSOLA double at -11 dB, cathedral (RT 4.8 s), and an octave-up 'halo' only inside the reverb.
- DEVIL: -5 st PSOLA, sub double at -17 st, grit, and a reverse-reverb swell in the 0.7 s before each line. V02 is possessed: adds a +7 st double, 73 Hz ring mod and heavier drive.
- FORECAST: AM boombox radio (250-4800 Hz, light drive) in the mud.
- RANGER: handheld radio (280-4400 Hz), key-up click 0.135 s before the line and a squelch tail after it.
- DISCLAIMER: pharma-ad legal read. Dry, double-compressed, bright.
- PHARISEES X01: the lead plus 7 PSOLA-shifted elders, each delayed 0-25 ms (never early), mild drive. The crowd stem adds 16 more.
- HIPPIE1/2: shouted in the rain.
- Pans (scene owners' on-screen x, applied ×0.6) are in `audio/vocal/pans.json`: R01 -.3, S01/S02 .35, P01 .35, P02 .3, S03 -.4, P03/P04 -.25, S04 -.1, P05 .3, P06 .25, M01 .1, M02 -.35, X01 .15, S05/P08 -.55, G01 .35, P24 .3, P25 .2, S06 -.15, P26 .25, P27/P28/P29 .3, S07/S08/S09 -.2, P30 .35, R02 -.45. Everything else is centre.
- Any line's tail ducks 12 dB under later lines it overlaps.

## crowd.wav (-22.1 LUFS, peak -3.1 dBFS)
| time | content | LUFS |
|---|---|---|
| 0.3-18.2 | camp walla in the rain: distant chatter | -39.4 |
| 56-87 | engraved crowd grumbling ("Ugh, rain", "Where's my dirt?") | ~-37 (-31 incl. mob) |
| 76.99 | X01 "SHUT UP!" mob: 16 shouted elders, onsets ±20 ms, then grumbles | -20.3 |
| 83.2-87.1 | ragged, scattered single "FLAGS!" under M02 | |
| 87.153-97.67 | THE FLAGS CHANT on Composer's grid (audio/music/grid.json): one "FLAGS" per beat, 126→154.1 BPM, 4→44 voices, jitter 70→12 ms, peak at 95.722 (-13.5 LUFS short-term), thinning after, gone by 97.75 (-60 LUFS) | -17.5 |
| 102.415 / 102.865 / 103.315 / 103.827 | the crowd lands on the Devil's V02 FLAGS (18 voices each); gated by 105.67 | |
| 108.4-112.4 | "Found one!" cheers | -29.5 |
| 118.92 | whispered gasp (20 voices), then a held rising "ooooh", then a cheer at 120.45 | -26.2 |
| 122.44-129.73 | "BURN IT!" on the burn drum beats (grid.json burn), whoops, roar at ignite 125.36; cut after 129.75 | -28.6 |
| 137.12-142.0 | deluge screams (held rising screams + "No!/Help!/The sky!"), faded out by 142.0 | -23.0 |
- The crowd dips 6 dB under every dialogue line, except X01, M03 and V02, which it is meant to swamp.
- The chant's beat times are written to `films/flaggame/cache/foley/chant_grid.json` for T04's kinetic type.
- Whisper (crowd stem only) hears: "SHUT UP!", "Found one. Found one.", "Burn it, burn it, burn it, burn it.", and "Let's go! The sky!" at the deluge. The dense chant peak and the V02 burst read to whisper as noise, not "flags".

## choir.wav (see `audio/vocal/choir_notes.json` for per-note pitch, the whisper transcript, HUP timing and levels)
- 169.09-181.0 hymn, WOODWORTH (Bradbury 1849, public domain) abridged: "Just as I am, without one Flag … I come, I come … A-men". E♭ major, 3/4, beat 0.525714 s (114.13 BPM) from the pickup 'Just' at 169.09; 'men' on beat 21 = 180.13 (Summer's Amen). Plagal A♭→E♭ ending. SATB is in `audio/vocal/hymn_grid.json`; Composer's organ doubles it exactly.
- Hymn performance: 3 singers per part (S af_heart af_sky bf_emma · A bf_alice af_jessica af_nova · T am_puck am_liam bm_daniel · B am_fenrir bm_george am_onyx). Vowel onsets on the beats, onset consonants 1.5× longer, and every note pitch-checked with a lower-register retry. Warm small hall (RT 1.5 s), about -25 LUFS, dipping 4 dB under the prayer lines.
- 143.94-146.77: one "HUP!" per raising, locked to T07Raised's 100 'raise' events (`films/flaggame/cache/foley/t07_hits_events.json`: #1 143.944 … #50 145.140, hold, #51 145.958 … #100 146.769). Placement error 0.01 ms. Full-frame hero cuts (strength 1.0) are 4 dB louder than in-betweens. Pitch rises up to +30% with the acceleration; -24.2 LUFS.
- 197.40: a radiant "FLAGS" on D major (69/66/62/50, the film's tonic, looping into the opening chord at 0.0), vowel on cue tract_lands. It swells and is released to silence by 198.95.

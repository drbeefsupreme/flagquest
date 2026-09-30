"""OPVS SCHISMATICVM — the sung score: single source of truth for every note the Vocalist's stems sing, and the grid
the Composer's organ follows (written to films/opus/audio/vocal/hymn_grid.json).   Owner: Vocalist-2.

Tuning A4 = 440 Hz, 12-TET.  Final D everywhere (mode I / D dorian -> D major).  MIDI: D2 38, A2 45, D3 50, F3 53,
G3 55, A3 57, Bb3 58, C4 60, D4 62, F#4 66, A4 69, D5 74.

 A01 incipit  (2.02)    cantor, recto tono: intonation D-F-G on "In-ci-pit", recite A3 (the accent "MA" long and
                        stressed), cadence G, F-E-D on "-cum", hummed m on D3
 antiphon     (163.00)  cantor solo, mode I, free rhythm: "Vexilla sint. Vexilla sunt. Vexilla erunt. Vexilla."
 hymn         (168.70)  choir, 84 BPM, 16 beats: organum that blossoms into triads
                        D5ths - C5ths (strict parallel organum) | Dm7 - D/G/C quartal over a D drone (oblique organum)
                        | G - A (discant: thirds appear) | D major (the Picardy third, full choir)
 A02 explicit (204.62)  cantor, as the incipit
 gratias      (207.60)  choir response "Vexillo gratias.": I I I | IV IV | I   (plagal)
 convergence  (159.00)  the babble lands on D major D2 A2 D3 A3 D4 F#4 A4 D5; thins to the vocal ison D2+D3
                        (161.5-163.0), held under the cantor until 168.6
"""
from __future__ import annotations

# ------------------------------------------------------------------------------------------------ cantor
# (syllable, vowel onset s re: first vowel, [(midi, dur s), ...])
INCIPIT_PS = "inʧˈipit ˈɔpus skizmˈatikum."
INCIPIT_T0 = 2.02
INCIPIT = [("In", 0.00, [(50, 0.18)]), ("ci", 0.25, [(53, 0.17)]), ("pit", 0.50, [(55, 0.19)]),
           ("o", 0.77, [(57, 0.18)]), ("pus", 1.01, [(57, 0.16)]), ("schis", 1.27, [(57, 0.18)]),
           ("ma", 1.53, [(57, 0.36)], {"level": 1.5}), ("ti", 1.94, [(55, 0.18)]),
           ("cum", 2.18, [(53, 0.14), (52, 0.14), (50, 0.26)], {"coda": 0.32})]

EXPLICIT_PS = "ˈɛkspliʧit ˈɔpus skizmˈatikum."
EXPLICIT_T0 = 204.62
EXPLICIT = [("Ex", 0.00, [(50, 0.16)]), ("pli", 0.34, [(53, 0.14)]), ("cit", 0.58, [(55, 0.19)]),
            ("o", 0.85, [(57, 0.18)]), ("pus", 1.09, [(57, 0.16)]), ("schis", 1.35, [(57, 0.18)]),
            ("ma", 1.61, [(57, 0.36)], {"level": 1.5}), ("ti", 2.02, [(55, 0.18)]),
            ("cum", 2.26, [(53, 0.14), (52, 0.14), (50, 0.28)], {"coda": 0.32})]

ANTIPHON_PS = "veksˈilla sˈint. veksˈilla sˈunt. veksˈilla ˈɛrunt. veksˈilla."
ANTIPHON_T0 = 163.00
ANTIPHON = [
    # 1  Vexilla sint.   (rise to the tenor A)
    ("Ve", 0.00, [(50, 0.16)]), ("xil", 0.24, [(53, 0.12), (55, 0.14)]), ("la", 0.58, [(57, 0.18)]),
    ("sint", 0.84, [(57, 0.34)]),
    # 2  Vexilla sunt.   (round the tenor with Bb, settle on the mediant F)
    ("Ve", 1.36, [(57, 0.16)]), ("xil", 1.60, [(58, 0.12), (57, 0.14)]), ("la", 1.94, [(55, 0.18)]),
    ("sunt", 2.20, [(53, 0.34)]),
    # 3  Vexilla erunt.  (the climb: C4, the future)
    ("Ve", 2.72, [(53, 0.16)]), ("xil", 2.96, [(57, 0.16)]), ("la", 3.22, [(60, 0.16)]),
    ("e", 3.46, [(60, 0.12), (58, 0.14)]), ("runt", 3.80, [(57, 0.32)]),
    # 4  Vexilla.        (the cadence: melisma on the accent, long final D)
    ("Ve", 4.30, [(57, 0.16)]), ("xil", 4.54, [(53, 0.11), (55, 0.11), (53, 0.11), (52, 0.13)]),
    ("la", 5.08, [(50, 0.62)]),
]
ANTIPHON_PHRASES = [(0, 4), (4, 8), (8, 13), (13, 16)]      # syllable index ranges of the four phrases

# ------------------------------------------------------------------------------------------------ hymn
HYMN_T0 = 168.70
BPM = 84.0
SPB = 60.0 / BPM
BEATS = 16
HYMN_END = HYMN_T0 + BEATS * SPB                            # 180.1286
# (word, beat, beats, [S, A, T, B], divisi extra notes)
HYMN = [
    ("Flags", 0, 2, [69, 62, 57, 50], []),        # D open fifth
    ("be", 2, 2, [67, 60, 55, 48], []),           # C open fifth: strict parallel organum (all voices down a step)
    ("Flags", 4, 2, [69, 65, 60, 50], []),        # Dm7 over the D drone (oblique organum)
    ("are", 6, 2, [72, 67, 62, 50], []),          # D-G-C quartal over the drone
    ("Flags", 8, 2, [71, 67, 62, 43], []),        # G major (dorian IV): the first full triad
    ("will", 10, 2, [73, 64, 57, 45], []),        # A major (C# leading tone)
    ("Flags", 12, 4, [74, 66, 57, 50], [69, 38]),  # D major, Picardy third, divisi S2 A4 + B2 D2
]
HYMN_PS = {"Flags": "flˈaɡz", "be": "bˈiː", "are": "ˈɑː", "will": "wˈɪl"}   # tall choral vowels

# ------------------------------------------------------------------------------------------------ gratias
GRATIAS_PS = "veksˈillo ɡrˈadzias."             # ti -> dz: Kokoro swallows the a before ʦ
GRATIAS_T0 = 207.60
# (syllable, onset s re: 207.60, dur s, [S, A, T, B])
_I = [69, 66, 62, 50]
_IV = [71, 67, 62, 43]
GRATIAS = [("Ve", 0.00, 0.26, _I), ("xil", 0.32, 0.22, _I), ("lo", 0.62, 0.30, _I),
           ("gra", 1.00, 0.28, _IV), ("ti", 1.34, 0.20, _IV), ("as", 1.58, 0.62, _I)]
GRATIAS_DIV = [38, 57]                                   # divisi on the final: B2 D2, T2 A3

# ------------------------------------------------------------------------------------------------ convergence
CONV_T = 159.0
CONV_CHORD = [38, 45, 50, 57, 62, 66, 69, 74]            # D2 A2 D3 A3 D4 F#4 A4 D5
ISON = [38, 50]                                          # D2 + D3, held under the cantor
ISON_SPAN = (161.5, 168.6)


def sched(syls):
    """score syllables -> op_vocal_lib.chant() schedule (one entry per vowel)"""
    out = []
    for s in syls:
        e = dict(t=s[1], notes=[tuple(n) for n in s[2]])
        if len(s) > 3:
            e.update(s[3])
        out.append(e)
    return out


def grid():
    """the JSON the Composer reads (global seconds)."""
    def cantor(t0, syl):
        out = []
        for s, t, notes, *_ in syl:
            tt = t0 + t
            for m, d in notes:
                out.append(dict(t=round(tt, 4), dur=round(d, 3), midi=m, syl=s))
                tt += d
        return out
    return {
        "tuning": "A4=440 Hz, 12-TET", "final": "D", "mode": "I (D dorian) -> D major",
        "note": "t = vowel onset (the beat); consonants start ~40-90 ms earlier. Written by op_vocal_score.py.",
        "cantor_incipit": cantor(INCIPIT_T0, INCIPIT),
        "cantor": cantor(ANTIPHON_T0, ANTIPHON),
        "cantor_phrases": [dict(t=round(ANTIPHON_T0 + ANTIPHON[a][1], 3),
                                end=round(ANTIPHON_T0 + ANTIPHON[b - 1][1] + sum(n[1] for n in ANTIPHON[b - 1][2]), 3),
                                text=" ".join(s[0] for s in ANTIPHON[a:b]))
                           for a, b in ANTIPHON_PHRASES],
        "explicit": cantor(EXPLICIT_T0, EXPLICIT),
        "choir": [dict(t=round(HYMN_T0 + b * SPB, 4), beat=b, beats=n, word=w, satb=satb, divisi=dv)
                  for w, b, n, satb, dv in HYMN],
        "hymn": dict(start=HYMN_T0, bpm=BPM, beat_s=round(SPB, 6), beats=BEATS, end=round(HYMN_END, 4),
                     phrase_starts=[round(HYMN_T0 + k * 4 * SPB, 4) for k in range(4)]),
        "convergence": dict(t=CONV_T, midis=CONV_CHORD, ison=ISON, ison_span=list(ISON_SPAN),
                            desc="babble lands on the D major chord at 159.0, thins to the ison D2+D3 161.5-163.0, "
                                 "ison held under the cantor until 168.6"),
        "gratias": [dict(t=round(GRATIAS_T0 + t, 4), dur=d, syl=s, satb=satb) for s, t, d, satb in GRATIAS],
        "gratias_divisi_final": GRATIAS_DIV,
    }

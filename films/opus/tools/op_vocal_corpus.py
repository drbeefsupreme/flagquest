"""OPVS SCHISMATICVM — text material for the crowd and choir stems.   Owner: Vocalist-2.
Film 1's corpus (tools/vocal_corpus.py: Vexillomantic lore, slop phrases, Babel translations) is re-exported and
extended with this film's theology: a processional litany, the heretics' council, Latin and Greek for Babel."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from vocal_corpus import BABEL_TX, CRACKPOTS, FAITHS, LORE, NATIONS, SLOP  # noqa: E402,F401

# m02 procession: a leader intones, the procession murmurs the response (Italianate Latin, Kokoro 'i' phonemes)
LITANY_LEAD = ["sàncte Augustìne", "sàncte Thòma", "sàncte Benedìcte", "sàncta Mònica", "sàncte Hierònyme",
               "sàncte Gregòri", "sàncte Ambròsi", "sàncte Basìli"]
LITANY_RESP = "òra pro nòbis"
# the machines' own processions, murmured underneath (vernacular + Latin tags of nations, faiths, philosophies)
MACHINE_MURMUR = {
    "a": ["Long live the king.", "We the people.", "In the beginning was the word.", "Know thyself.",
          "All things flow.", "Lord, have mercy.", "One nation, under heaven.", "The unexamined life.",
          "Glory to the fatherland.", "Blessed are the meek.", "I think, therefore I am.", "Peace be with you."],
    "i": ["Kýrie eléison.", "Te rogàmus, àudi nos.", "Pànta rèi.", "Cògito, èrgo sum.", "Pro pàtria.",
          "Nòsce te ìpsum.", "Vìvat rex.", "Gràtia plèna.", "Dòminus vobìscum.", "Òmnia mutàntur."],
}

# m06 the heretics' council: anathemas flying between the churches of one
COUNCIL_LATIN = ["Anàthema sit!", "Hǽreticus!", "Anàthema!", "Schìsma!", "Excommunicàtus es!", "Àbsit!",
                 "Ego sum pàpa!", "Anàthema sit, anàthema!", "Contra fìdem!", "Àpage!"]
COUNCIL_EN = ["Heretic!", "Anathema!", "Blasphemy!", "Apostate!", "I alone am orthodox!", "Excommunicated!",
              "The council is mine!", "Schismatic!", "I am the true church!", "Depart from me!"]

# Babel additions: Latin (Italian voices) and Church Greek (Portuguese/Spanish voices read the transliteration)
BABEL_LATIN = ["Nemo intèllegit nèminem!", "Ego sum ecclèsia mea!", "Nùmeri non sunt veri!", "Àudi me! Àudi me!",
               "Ùnus sum!", "Anàthema sit!", "Tùrris ùsque ad cælum!", "Confùsio linguàrum!"]
BABEL_GREEK = ["Kýrie eléison!", "Ánathema ésto!", "Oudeís akoúei!", "Egó eimí!", "Pánta rhei!"]

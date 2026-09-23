"""Read-only access to the locked master timeline (timeline.json) + lip-sync envelopes."""
import json
from functools import lru_cache

import numpy as np

from .config import TIMELINE, AUDIO


class Timeline:
    def __init__(self, path=TIMELINE):
        d = json.loads(path.read_text())
        self.d = d
        self.fps = d["fps"]
        self.duration = d["duration"]
        self.scenes = d["scenes"]
        self.lines = d["lines"]
        self.cues = {c["id"]: c["t"] for c in d["cues"]}
        self.hymn = d["hymn"]
        self._mouth = None

    # ---------------------------------------------------------------- scenes
    def scene(self, sid):
        return next(s for s in self.scenes if s["id"] == sid)

    def cue(self, cid):
        """global time (s) of a named cue"""
        return self.cues[cid]

    # ---------------------------------------------------------------- dialogue
    def line(self, lid):
        return next(l for l in self.lines if l["id"] == lid)

    def lines_for(self, scene=None, speaker=None):
        return [l for l in self.lines if (scene is None or l["scene"] == scene)
                and (speaker is None or l["speaker"] == speaker)]

    def speaking(self, speaker, T, pad=0.0):
        """the line `speaker` is saying at global time T (or None)"""
        for l in self.lines:
            if l["speaker"] == speaker and l["start"] - pad <= T <= l["end"] + pad:
                return l
        return None

    def words(self, lid):
        """spoken words of line lid (punctuation tokens removed): [{w, s, e}] in global seconds"""
        return [w for w in self.line(lid)["words"] if any(ch.isalnum() for ch in w["w"])]

    def word(self, lid, k, n=0):
        """(start, end) global seconds of word k of line lid. k = index among spoken words, or the word
        text (case-insensitive); n = which occurrence (0-based) for repeated words, e.g. word("K03","Breach",2)."""
        ws = self.words(lid)
        if isinstance(k, str):
            hits = [w for w in ws if w["w"].lower() == k.lower()]
            if len(hits) <= n:
                raise KeyError(f"{lid}: word {k!r} occurrence {n} not in {[w['w'] for w in ws]}")
            return hits[n]["s"], hits[n]["e"]
        return ws[k]["s"], ws[k]["e"]

    def mouth(self, speaker, T):
        """lip-sync mouth openness 0..1 for `speaker` at global time T"""
        if self._mouth is None:
            z = np.load(AUDIO / "voice/mouth.npz")
            self._mouth = {k: z[k] for k in z.files if k != "sr"}
            self._msr = int(z["sr"])
        a = self._mouth.get(speaker)
        if a is None:
            return 0.0
        x = T * self._msr
        i = int(x)
        if i < 0 or i + 1 >= len(a):
            return 0.0
        f = x - i
        return float(a[i] * (1 - f) + a[i + 1] * f)

    def subtitle(self, T):
        for l in self.lines:
            if l["start"] - 0.1 <= T <= l["end"] + 0.35:
                return l
        return None


@lru_cache(maxsize=1)
def get_timeline():
    return Timeline()

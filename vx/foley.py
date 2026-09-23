"""Physics -> sound bridge. Scenes export per-frame motion energy of simulated objects (flags, debris,
crowds) so the Foley department can synthesize sound that is literally derived from the animation.

Files land in cache/foley/<scene>_<name>.npz with keys:
  kind (str), f0 (global frame of local frame 0), fps, energy (float32[n]), pan (float32[n] in -1..1),
  gain (float), events (float32[k,3] -> [global_seconds, strength, pan]), plus any **extra per-frame
  arrays passed by the caller (e.g. flap_hz, snap) saved as float32 under their own keys.
"""
import numpy as np

from .config import CACHE, W


def export_track(S, name, energy, pan=0.0, gain=1.0, kind="flag", events=None, **extra):
    d = CACHE / "foley"
    d.mkdir(parents=True, exist_ok=True)
    energy = np.asarray(energy, np.float32)
    pan = np.broadcast_to(np.asarray(pan, np.float32), energy.shape).astype(np.float32)
    ev = np.asarray(events if events is not None else np.zeros((0, 3)), np.float32).reshape(-1, 3)
    reserved = {"kind", "f0", "fps", "energy", "pan", "gain", "events"}
    bad = reserved & set(extra)
    if bad:
        raise ValueError(f"export_track: extra keys clash with reserved names {bad}")
    np.savez(d / f"{S.id}_{name}.npz", kind=kind, f0=S.f0, fps=S.fps, energy=energy, pan=pan,
             gain=float(gain), events=ev, **{k: np.asarray(v, np.float32) for k, v in extra.items()})


def screen_pan(x):
    """design-space x -> stereo pan -1..1"""
    return float(np.clip((x - W / 2) / (W / 2), -1, 1))



def export_events(S, name, events):
    """Label visual hits for the sound team. events = [dict(t=GLOBAL_seconds, kind="shatter"|"impact"|
    "whoosh"|"click"|"thunk"|"snap"|"crack"|..., strength=0..1, pan=-1..1, desc="what happens")].
    Written to cache/foley/<scene>_<name>_events.json (overwrite freely; the sound team re-reads)."""
    import json
    d = CACHE / "foley"
    d.mkdir(parents=True, exist_ok=True)
    evs = sorted(({"t": float(e["t"]), "kind": e.get("kind", "hit"), "strength": float(e.get("strength", 1.0)),
                   "pan": float(e.get("pan", 0.0)), "desc": e.get("desc", "")} for e in events), key=lambda e: e["t"])
    (d / f"{S.id}_{name}_events.json").write_text(json.dumps(evs, indent=1))
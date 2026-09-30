"""Verify the film-3 Foley stems (owner: Foley-2).

    cd <repo> && source .venv/bin/activate && python films/opus/tools/op_foley_check.py [--spec]

Checks: exact format (48 kHz stereo float32, 10,176,000 samples); the sacred silence (sfx digital zero 145.0-146.5,
ambience digital zero 145.0-146.0 + only a faint breath of wind 146.0-146.5); cue/event-aligned transients (the
onset of every designed hit vs the time it was placed on, ±15 ms); ceiling; per-scene levels; physics-flutter
correlations. Writes audio/foley/check.json and appends a Verification section to CUES.md.
--spec also writes per-scene spectrograms audio/foley/spec_<scene>.png and levels.png.
"""
import os

os.environ["VX_FILM"] = "opus"
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

HERE = Path(__file__).resolve().parent
FROOT = HERE.parent
sys.path.insert(0, str(FROOT.parents[1] / "tools"))
import foley_check as C  # noqa: E402  (film-1 onset detector)

TL = json.loads((FROOT / "timeline.json").read_text())
SR = 48000
NTOT = int(round(TL["duration"] * SR))
OUTD = FROOT / "audio" / "foley"
CUES = {c["id"]: c["t"] for c in TL["cues"]}
TRANSIENT = ["candle_lit", "incipit", "edict_seal", "ka_ching", "hypermeme", "breach", "collapse", "flag_planted",
             "convergence", "end_card"]


def rms_db(x):
    x = np.asarray(x, np.float64)
    return round(float(10 * np.log10((x ** 2).mean() + 1e-20)), 1) if len(x) else -200.0


def main():
    rep, stems = {}, {}
    for nm in ("sfx", "ambience"):
        p = FROOT / "audio" / "stems" / f"{nm}.wav"
        info = sf.info(str(p))
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        stems[nm] = x
        s = lambda a, b: x[int(round(a * SR)):int(round(b * SR))]  # noqa: E731
        rep[nm] = dict(sr=sr, channels=info.channels, subtype=info.subtype, frames=info.frames,
                       exact_length=info.frames == NTOT, peak_db=round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 2),
                       rms_db=rms_db(x), max_abs_silence_145_146=float(np.abs(s(145.0, 146.0)).max()),
                       max_abs_146_146_5=float(np.abs(s(146.0, 146.5)).max()), rms_146_146_5=rms_db(s(146.0, 146.5)),
                       rms_144_145=rms_db(s(144.0, 145.0)), rms_146_5_147_5=rms_db(s(146.5, 147.5)),
                       nan=bool(not np.all(np.isfinite(x))),
                       scenes={sc["id"]: rms_db(s(sc["start"], sc["end"])) for sc in TL["scenes"]})
    ok = all(rep[n]["exact_length"] and rep[n]["sr"] == SR and rep[n]["channels"] == 2 and rep[n]["subtype"] == "FLOAT"
             for n in rep)
    rep["format_ok"] = ok
    rep["silence_ok"] = (rep["sfx"]["max_abs_silence_145_146"] == 0.0 and rep["sfx"]["max_abs_146_146_5"] == 0.0 and
                         rep["ambience"]["max_abs_silence_145_146"] == 0.0 and rep["ambience"]["rms_146_146_5"] < -45)
    # transients: onset of each designed hit vs the time it was placed on (cue, or the scene's exported event)
    log = json.loads((OUTD / "placements.json").read_text())
    al = []
    for cid in TRANSIENT:
        pl = sorted([e for e in log if e.get("cue") == cid and e["stem"] == "sfx"], key=lambda e: -e["peak"])
        if not pl:
            continue
        t = pl[0]["t"]
        o = C.onset_near(stems["sfx"], t, 0.06)
        al.append(dict(cue=cid, cue_t=CUES[cid], placed_t=t, placed_by=pl[0]["src"], sound=pl[0]["name"][:60],
                       onset=round(o, 4), err_ms=round((o - t) * 1000, 1)))
    rep["transients"] = al
    rep["transients_max_abs_err_ms"] = max(abs(a["err_ms"]) for a in al) if al else None
    ph = json.loads((OUTD / "physics.json").read_text())
    rep["physics"] = ph
    used = json.loads((OUTD / "events_used.json").read_text())
    rep["events"] = {k: len(v) for k, v in used.items()}
    (OUTD / "check.json").write_text(json.dumps(rep, indent=1, default=float))
    for nm in ("sfx", "ambience"):
        s = rep[nm]
        print(f"{nm:8s} {s['frames']} frames exact={s['exact_length']} {s['subtype']} peak {s['peak_db']} dB "
              f"rms {s['rms_db']} dB | 145-146 max {s['max_abs_silence_145_146']} | 146-146.5 max "
              f"{s['max_abs_146_146_5']:.2e} rms {s['rms_146_146_5']} dB | 144-145 {s['rms_144_145']} dB")
        print("         scenes: " + "  ".join(f"{k} {v}" for k, v in s["scenes"].items()))
    print(f"format_ok={rep['format_ok']} silence_ok={rep['silence_ok']}")
    for a in al:
        print(f"  {a['cue']:13s} cue {a['cue_t']:7.2f} placed {a['placed_t']:8.3f} ({a['placed_by'][:28]:28s}) onset "
              f"{a['onset']:8.3f} err {a['err_ms']:+6.1f} ms  {a['sound']}")
    for p in ph:
        print(f"  physics {p['file']:24s} r={p['corr_energy_rms']}")
    # CUES.md verification section
    cues_md = OUTD / "CUES.md"
    txt = cues_md.read_text().split("\n## Verification")[0].rstrip()
    V = ["", "", "## Verification (films/opus/tools/op_foley_check.py)", ""]
    for nm in ("sfx", "ambience"):
        s = rep[nm]
        V.append(f"- `{nm}.wav`: {s['frames']} frames ({'exact' if s['exact_length'] else 'WRONG LENGTH'}), {s['sr']} Hz, "
                 f"{s['channels']} ch, {s['subtype']}; peak {s['peak_db']} dBFS, RMS {s['rms_db']} dBFS; 145.0–146.0 max "
                 f"|x| = {s['max_abs_silence_145_146']}; 146.0–146.5 max |x| = {s['max_abs_146_146_5']:.2e} "
                 f"(RMS {s['rms_146_146_5']} dBFS); 144–145 RMS {s['rms_144_145']} dBFS")
    V.append(f"- format ok: {rep['format_ok']}; sacred silence ok: {rep['silence_ok']}")
    V.append(f"- scene events: {rep['events']}")
    V += ["", f"Transient hits (max |error| {rep['transients_max_abs_err_ms']} ms):", "",
          "| cue | cue t | placed at | placed by | onset | err ms | sound |", "|---|---:|---:|---|---:|---:|---|"]
    V += [f"| {a['cue']} | {a['cue_t']:.2f} | {a['placed_t']:.3f} | {a['placed_by']} | {a['onset']:.3f} | "
          f"{a['err_ms']:+.1f} | {a['sound']} |" for a in al]
    V += ["", "Per-scene RMS (dBFS): " + "; ".join(f"{k}: sfx {rep['sfx']['scenes'][k]} / amb {rep['ambience']['scenes'][k]}"
                                                   for k in rep["sfx"]["scenes"])]
    cues_md.write_text(txt + "\n".join(V) + "\n")
    if "--spec" in sys.argv:
        spec(stems)


def spec(stems):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal as sps
    for sc in TL["scenes"]:
        a, b = int(sc["start"] * SR), int(sc["end"] * SR)
        fig, axs = plt.subplots(3, 1, figsize=(18, 9), gridspec_kw=dict(height_ratios=[3, 3, 1.3]))
        for ax, nm in zip(axs[:2], ("sfx", "ambience")):
            m = stems[nm][a:b].mean(1)
            f, t, S = sps.spectrogram(m, SR, nperseg=2048, noverlap=1536)
            ax.pcolormesh(t + sc["start"], f, 10 * np.log10(S + 1e-15), vmin=-135, vmax=-45, shading="auto",
                          cmap="magma")
            ax.set_yscale("symlog", linthresh=200)
            ax.set_ylim(20, 22000)
            ax.set_ylabel(nm)
            for c in TL["cues"]:
                if sc["start"] <= c["t"] < sc["end"]:
                    ax.axvline(c["t"], color="cyan", lw=0.6, alpha=0.6)
                    ax.text(c["t"], 15000, c["id"], color="cyan", fontsize=7, rotation=90, va="top")
        for nm, col in (("sfx", "C1"), ("ambience", "C0")):
            m = stems[nm][a:b].mean(1).astype(np.float64)
            hop = int(0.02 * SR)
            e = np.sqrt(np.convolve(m ** 2, np.ones(hop) / hop, "same")[::hop])
            axs[2].plot(np.arange(len(e)) * 0.02 + sc["start"], 20 * np.log10(e + 1e-9), col, lw=0.7, label=nm)
        for ln in TL["lines"]:
            if sc["start"] <= ln["start"] < sc["end"]:
                axs[2].axvspan(ln["start"], ln["end"], color="0.85", zorder=0)
        axs[2].set_ylim(-80, 0)
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(fontsize=7)
        axs[2].set_xlim(sc["start"], sc["end"])
        fig.suptitle(f"{sc['id']} — {sc['title']}")
        fig.tight_layout()
        fig.savefig(OUTD / f"spec_{sc['id']}.png", dpi=62)
        plt.close(fig)


if __name__ == "__main__":
    main()

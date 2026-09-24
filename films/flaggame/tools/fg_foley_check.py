"""Verify film-2 Foley stems (owner: Foley).   source films/flaggame/env.sh && python films/flaggame/tools/fg_foley_check.py [--spec]

Checks: exact format; cue onsets (±15 ms); THE LOOP (199.0 -> 0.0 continuity); the rain freeze (118.92: nothing but
ringing/heartbeat); physics correlations; peaks. Appends a Verification section to CUES.md; --spec writes plots.
"""
import os

os.environ.setdefault("VX_FILM", "flaggame")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal as sps  # noqa: E402

HERE = Path(__file__).resolve().parent
FROOT = HERE.parent
sys.path.insert(0, str(FROOT.parents[1] / "tools"))
import foley_check as C  # noqa: E402  (film-1 onset detector)

TL = json.loads((FROOT / "timeline.json").read_text())
SR, NTOT = 48000, int(round(TL["duration"] * 48000))
OUTD = FROOT / "audio" / "foley"
CUES = {c["id"]: c["t"] for c in TL["cues"]}
TRANSIENT = ["wrong_burst", "shut_up", "flood_peak", "devil_glitch", "found_one", "flag_on_effigy", "firmament_crack",
             "deluge", "raise_1", "blaspheme", "kneel", "tract_lands"]


def rms_db(x):
    return round(float(10 * np.log10((x.astype(np.float64) ** 2).mean() + 1e-20)), 1)


def main():
    rep, stems = {}, {}
    for nm in ("sfx", "ambience"):
        p = FROOT / "audio" / "stems" / f"{nm}.wav"
        info = sf.info(str(p))
        x, _ = sf.read(str(p), dtype="float32", always_2d=True)
        stems[nm] = x
        d = np.abs(np.diff(x, axis=0)).max(1)
        seam = float(np.abs(x[0] - x[-1]).max())
        rep[nm] = dict(frames=info.frames, sr=info.samplerate, ch=info.channels, subtype=info.subtype,
                       exact=info.frames == NTOT and info.samplerate == SR and info.channels == 2 and info.subtype == "FLOAT",
                       peak_dbfs=round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 2), rms_dbfs=rms_db(x),
                       clipped=int((np.abs(x) >= 0.999).sum()), nan=bool(np.isnan(x).any()),
                       loop_seam_jump=round(seam, 5), loop_seam_vs_p99_step=round(seam / (np.percentile(d, 99) + 1e-12), 3),
                       loop_rms_last_1s=rms_db(x[-SR:]), loop_rms_first_1s=rms_db(x[:SR]),
                       freeze_118_95_119_5_rms=rms_db(x[int(118.95 * SR):int(119.5 * SR)]),
                       pre_freeze_118_4_118_9_rms=rms_db(x[int(118.4 * SR):int(118.9 * SR)]))
    mix = stems["sfx"] + stems["ambience"]
    # spectral match across the seam (the loop must sound identical)
    f, a = sps.welch(mix[-2 * SR:].mean(1), SR, nperseg=4096)
    _, b = sps.welch(mix[:2 * SR].mean(1), SR, nperseg=4096)
    band = (f > 60) & (f < 16000)
    rep["loop_spectral_diff_db"] = round(float(np.abs(10 * np.log10(a[band] / b[band])).mean()), 2)
    al = []
    for cid in TRANSIENT:
        o = C.onset_near(stems["sfx"], CUES[cid])
        al.append(dict(cue=cid, t=CUES[cid], onset=round(o, 4), err_ms=round((o - CUES[cid]) * 1000, 1)))
    rep["cue_alignment"] = al
    rep["cue_alignment_max_abs_err_ms"] = max(abs(x["err_ms"]) for x in al)
    rep["physics"] = json.loads((OUTD / "physics.json").read_text())
    (OUTD / "check.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("cue_alignment", "physics")}, indent=1, default=float))
    for x in al:
        print(f"  cue {x['cue']:16s} {x['t']:7.2f} onset {x['onset']:8.3f} err {x['err_ms']:+6.1f} ms")
    cues_md = OUTD / "CUES.md"
    txt = cues_md.read_text().split("\n## Verification")[0].rstrip()
    V = ["", "", "## Verification (films/flaggame/tools/fg_foley_check.py)", ""]
    for nm in ("sfx", "ambience"):
        s = rep[nm]
        V.append(f"- `{nm}.wav`: {s['sr']} Hz, {s['ch']} ch, {s['frames']} frames, {s['subtype']} (exact {s['exact']}); "
                 f"peak {s['peak_dbfs']} dBFS, RMS {s['rms_dbfs']} dBFS, clipped {s['clipped']}; loop seam |x[0]-x[-1]| "
                 f"{s['loop_seam_jump']} ({s['loop_seam_vs_p99_step']}× the 99th-pct sample step), RMS last/first second "
                 f"{s['loop_rms_last_1s']}/{s['loop_rms_first_1s']} dBFS; freeze window 118.95–119.5 RMS "
                 f"{s['freeze_118_95_119_5_rms']} dBFS (vs {s['pre_freeze_118_4_118_9_rms']} just before)")
    V.append(f"- loop spectral difference (last 2 s vs first 2 s, 60 Hz–16 kHz): {rep['loop_spectral_diff_db']} dB mean")
    V += ["", f"Cue onsets (max |error| {rep['cue_alignment_max_abs_err_ms']} ms):", "", "| cue | t | onset | err ms |",
          "|---|---:|---:|---:|"] + [f"| {x['cue']} | {x['t']:.2f} | {x['onset']:.3f} | {x['err_ms']:+.1f} |" for x in al]
    cues_md.write_text(txt + "\n".join(V) + "\n")
    if "--spec" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hop = 2400
        fig, ax = plt.subplots(2, 1, figsize=(26, 7), sharex=True)
        for k, (nm, x) in enumerate(stems.items()):
            n = len(x) // hop
            rm = 10 * np.log10((x[: n * hop] ** 2).mean(1).reshape(n, hop).mean(1) + 1e-12)
            pk = 20 * np.log10(np.abs(x[: n * hop]).max(1).reshape(n, hop).max(1) + 1e-9)
            tt = np.arange(n) * hop / SR
            ax[k].plot(tt, pk, lw=0.6)
            ax[k].plot(tt, rm, lw=0.8)
            ax[k].set_ylim(-80, 0)
            ax[k].set_ylabel(nm)
            for s in TL["scenes"]:
                ax[k].axvline(s["start"], color="k", lw=0.6)
                ax[k].text(s["start"] + 0.3, -5, s["id"], fontsize=8)
            for t in CUES.values():
                ax[k].axvline(t, color="r", lw=0.4, alpha=0.4)
        ax[1].set_xlim(0, TL["duration"])
        fig.tight_layout()
        fig.savefig(OUTD / "levels.png", dpi=80)
        plt.close(fig)


if __name__ == "__main__":
    main()

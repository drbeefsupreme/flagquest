"""Verify the Foley stems (owner: Foley).

    python tools/foley_check.py [--spec]      # numeric report -> audio/foley/check.json (+ PNGs with --spec)

Checks: exact format/length, cue-aligned transients (±15 ms), physics-flutter correlation, sacred silence
138.5–139.6, clipping/peaks. With --spec: audio/foley/levels.png and per-scene spectrograms spec_<scene>.png.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import argparse  # noqa: E402
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import foley_flutter as FF  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TL = json.loads((ROOT / "timeline.json").read_text())
SR = 48000
NTOT = 9264000
OUTD = ROOT / "audio" / "foley"
# cues whose designed sound has a sharp transient ON the cue (the rest are swells/whooshes/wave onsets)
TRANSIENT = ["flag_rotates_in", "flag_snap_open", "title_slam", "slop_crash", "tv_on", "pen_sign", "ka_ching",
             "hypermeme", "breach", "nation_fracture", "faith_fracture", "philo_shatter", "babel_crack",
             "flag_planted", "glorious", "end_card"]

def onset_near(x, t, win=0.08):
    """strongest onset within ±win of t: 4 ms RMS (1 ms hop) of the >300 Hz band, onset score = level-weighted
    3 ms dB rise. Returns the time where the rise begins (seconds)."""
    a, b = int((t - win - 0.03) * SR), int((t + win + 0.01) * SR)
    seg = x[max(0, a):b].mean(1).astype(np.float64)
    seg = sps.sosfilt(sps.butter(2, 300 / (SR / 2), "high", output="sos"), seg)
    hop, w = 48, 192
    p = np.convolve(seg ** 2, np.ones(w) / w, "full")[w - 1:][::hop]      # causal window ending at frame
    le = 10 * np.log10(p + 1e-14)
    flux = np.zeros_like(le)
    flux[3:] = le[3:] - le[:-3]
    score = flux * np.clip((le - le.max() + 30) / 30, 0, 1)
    i0 = int(0.03 * SR / hop)
    i1 = len(score) - int(0.01 * SR / hop)
    i = int(np.argmax(score[i0:i1])) + i0
    # the rise began ~2 frames before the causal-window peak of the flux; refine: first frame of the rise
    j = i
    while j > i0 and le[j - 1] < le[j] - 0.5:
        j -= 1
    return max(0, a) / SR + j * hop / SR


def physics_plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    trs = [t for t in FF.load_tracks() if t["kind"] != "crowd"]
    fig, axs = plt.subplots(len(trs), 1, figsize=(18, 2.1 * len(trs)))
    axs = np.atleast_1d(axs)
    for ax, tr in zip(axs, trs):
        x, t0, info = FF.track_audio(tr)
        rms = FF.frame_rms(x, tr["fps"], len(tr["energy"]))
        act = np.nonzero(tr["energy"] > 0.01)[0]
        if len(act) == 0:
            continue
        a, b = max(0, act[0] - 12), min(len(rms), act[-1] + 12)
        tt = t0 + np.arange(a, b) / tr["fps"]
        r = np.corrcoef(tr["energy"], rms)[0, 1]
        ax.plot(tt, tr["energy"][a:b] / (tr["energy"].max() + 1e-9), lw=1.2, label="sim energy (norm.)")
        ax.plot(tt, rms[a:b] / (rms.max() + 1e-12), lw=0.9, alpha=0.8, label="synthesized flutter RMS (norm.)")
        for ev in tr["events"]:
            if tt[0] <= ev[0] <= tt[-1]:
                ax.axvline(ev[0], color="r", lw=0.6, alpha=0.6)
        ax.set_title(f"{tr['file']}  r = {r:.3f}  (red: sim snap events)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="store_true")
    a = ap.parse_args(argv)
    rep = {}
    stems = {}
    for nm in ("sfx", "ambience"):
        p = ROOT / "audio" / "stems" / f"{nm}.wav"
        info = sf.info(str(p))
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        stems[nm] = x
        rep[nm] = dict(samplerate=info.samplerate, channels=info.channels, frames=info.frames, subtype=info.subtype,
                       exact=(info.samplerate == SR and info.channels == 2 and info.frames == NTOT and
                              info.subtype == "FLOAT"),
                       peak_dbfs=round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 2),
                       rms_dbfs=round(float(10 * np.log10((x ** 2).mean() + 1e-20)), 2),
                       clipped_samples=int((np.abs(x) >= 0.999).sum()),
                       nan=bool(np.isnan(x).any()))
        i0, i1 = int(138.5 * SR), int(139.6 * SR)
        rep[nm]["silence_138.5_139.6_maxabs"] = float(np.abs(x[i0:i1]).max())
        rep[nm]["last_10ms_maxabs"] = float(np.abs(x[-480:]).max())
    rep["sfx"]["silence_138.5_140.0_maxabs"] = float(np.abs(stems["sfx"][int(138.5 * SR):int(140.0 * SR)]).max())
    amb = stems["ambience"]
    w = amb[int(139.6 * SR):int(141.5 * SR)]
    rep["ambience"]["wind_139.6_141.5_rms_dbfs"] = round(float(10 * np.log10((w ** 2).mean() + 1e-20)), 1)
    pre = amb[int(138.0 * SR):int(138.5 * SR)]
    rep["ambience"]["pre_cut_138.0_138.5_rms_dbfs"] = round(float(10 * np.log10((pre ** 2).mean() + 1e-20)), 1)
    # --- cue alignment
    cues = {c["id"]: c["t"] for c in TL["cues"]}
    sfx = stems["sfx"]
    al = []
    for cid in TRANSIENT:
        t = cues[cid]
        o = onset_near(sfx, t)
        al.append(dict(cue=cid, t=t, onset=round(o, 4), err_ms=round((o - t) * 1000, 1)))
    rep["cue_alignment"] = al
    rep["cue_alignment_max_abs_err_ms"] = max(abs(x["err_ms"]) for x in al)
    # --- physics correlation (isolated renders from the build + stem-level)
    phys = json.loads((OUTD / "physics.json").read_text()) if (OUTD / "physics.json").exists() else []
    tracks = {t["file"]: t for t in FF.load_tracks()}
    for p in phys:
        tr = tracks.get(p["file"])
        if tr is None:
            continue
        en = tr["energy"]
        f0 = tr["f0"]
        act = np.nonzero(en > 0.02)[0]
        if len(act) < 10:
            continue
        seg = sfx[int(f0 / 24 * SR):int((f0 + len(en)) / 24 * SR)].mean(1)
        rms = np.array([np.sqrt((seg[int(i * 2000):int((i + 1) * 2000)] ** 2).mean() + 1e-20) for i in range(len(en))])
        p["corr_in_stem_mix"] = round(float(np.corrcoef(en[act], rms[act])[0, 1]), 3)
    rep["physics"] = phys
    (OUTD / "check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("cue_alignment", "physics")}, indent=1))
    for x in al:
        print(f"  cue {x['cue']:18s} {x['t']:7.2f}  onset {x['onset']:8.3f}  err {x['err_ms']:+6.1f} ms")
    for p in phys:
        print(f"  physics {p['file']:30s} r(isolated)={p['corr_energy_rms']:.3f}  r(in stem)={p.get('corr_in_stem_mix')}")
    write_verification(rep)
    if a.spec:
        plots(stems, cues)
        physics_plot(OUTD / "physics_tracks.png")


def write_verification(rep):
    p = OUTD / "CUES.md"
    if not p.exists():
        return
    txt = p.read_text().split("\n## Verification")[0].rstrip()
    L = ["", "", "## Verification (tools/foley_check.py)", ""]
    for nm in ("sfx", "ambience"):
        s = rep[nm]
        L.append(f"- `{nm}.wav`: {s['samplerate']} Hz, {s['channels']} ch, {s['frames']} frames, {s['subtype']} "
                 f"(exact: {s['exact']}); peak {s['peak_dbfs']} dBFS, RMS {s['rms_dbfs']} dBFS, "
                 f"clipped samples {s['clipped_samples']}; max |x| in 138.5–139.6 = {s['silence_138.5_139.6_maxabs']}")
    L.append(f"- sfx max |x| 138.5–140.0 = {rep['sfx']['silence_138.5_140.0_maxabs']}; dawn wind 139.6–141.5 RMS "
             f"{rep['ambience']['wind_139.6_141.5_rms_dbfs']} dBFS")
    L += ["", f"Cue alignment (strongest >300 Hz onset within ±80 ms; max |error| "
              f"{rep['cue_alignment_max_abs_err_ms']} ms):", "", "| cue | t | measured onset | error ms |", "|---|---:|---:|---:|"]
    for x in rep["cue_alignment"]:
        L.append(f"| {x['cue']} | {x['t']:.2f} | {x['onset']:.3f} | {x['err_ms']:+.1f} |")
    L += ["", "Physics flutter correlation (Pearson r between the sim's per-frame energy and the synthesized flutter's "
              "per-frame RMS; 'in stem' = same against the whole sfx stem incl. every other sound):", "",
          "| track | r isolated | r in stem |", "|---|---:|---:|"]
    for x in rep["physics"]:
        L.append(f"| {x['file']} | {x['corr_energy_rms']:.3f} | {x.get('corr_in_stem_mix', '')} |")
    p.write_text(txt + "\n".join(L) + "\n")


def plots(stems, cues):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hop = 2400
    fig, ax = plt.subplots(2, 1, figsize=(26, 7), sharex=True)
    for k, (nm, x) in enumerate(stems.items()):
        m = np.abs(x).max(1)
        n = len(m) // hop
        pk = 20 * np.log10(m[: n * hop].reshape(n, hop).max(1) + 1e-9)
        rm = 10 * np.log10((x[: n * hop] ** 2).mean(1).reshape(n, hop).mean(1) + 1e-12)
        tt = np.arange(n) * hop / SR
        ax[k].plot(tt, pk, lw=0.6, label="peak")
        ax[k].plot(tt, rm, lw=0.8, label="rms")
        ax[k].set_ylim(-80, 0)
        ax[k].set_ylabel(f"{nm} dBFS")
        for s in TL["scenes"]:
            ax[k].axvline(s["start"], color="k", lw=0.6)
            ax[k].text(s["start"] + 0.3, -5, s["id"], fontsize=8)
        for cid, t in cues.items():
            ax[k].axvline(t, color="r", lw=0.4, alpha=0.5)
        ax[k].grid(alpha=0.3)
    ax[1].set_xlim(0, 193)
    fig.tight_layout()
    fig.savefig(OUTD / "levels.png", dpi=80)
    plt.close(fig)
    mix = stems["sfx"] + stems["ambience"]
    for s in TL["scenes"]:
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        seg = mix[a:b].mean(1)
        f, t, S = sps.spectrogram(seg, SR, nperseg=2048, noverlap=1536)
        S = 10 * np.log10(S + 1e-14)
        fig, ax = plt.subplots(figsize=(22, 5))
        ax.pcolormesh(t + s["start"], f, S, vmin=S.max() - 90, vmax=S.max(), shading="auto", cmap="magma")
        ax.set_yscale("symlog", linthresh=200)
        ax.set_ylim(20, 24000)
        for cid, tc in cues.items():
            if s["start"] <= tc <= s["end"]:
                ax.axvline(tc, color="c", lw=0.8)
                ax.text(tc, 18000, cid, color="c", fontsize=8)
        ax.set_title(f"{s['id']} sfx+ambience")
        fig.tight_layout()
        fig.savefig(OUTD / f"spec_{s['id']}.png", dpi=70)
        plt.close(fig)


if __name__ == "__main__":
    main()

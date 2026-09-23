"""Render the Foley synthesis models one by one for auditioning (owner: Foley).

    python tools/foley_audition.py      -> audio/foley/audition/<name>.wav + audio/foley/audition_sheet.png
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys  # noqa: E402
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import foley_build as B  # noqa: E402
import foley_flutter as FF  # noqa: E402
import foley_lib as L  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "audio" / "foley" / "audition"


def items():
    r = L.R(5)
    en = np.clip(0.25 + 0.2 * np.sin(np.arange(96) / 5) + 0.08 * B.frame_turb(r, 96), 0, 1)
    en[40:44] += 0.5
    yield "flag_flutter_from_energy", FF.flutter_from_energy(en, 24, 0.0, 1.0, [(1.7, 0.9, 0.0)], seed=3)
    yield "colossal_flag", FF.flutter_from_energy(np.clip(en + 0.3, 0, 1), 24, 0.0, 1.0, [], seed=4, size=6.0)
    yield "wind_gusts", L.wind(r, 4.0, 0.5, 0.7, 0.3, 0.4, 0.4)
    yield "storm_gale", L.wind(r, 4.0, 1.0, 0.4, 0.3, 0.7, 0.55)
    yield "thunder_near", L.thunder(r, 0.95).x
    yield "thunder_far", L.thunder(r, 0.2).x
    yield "explosion_breach", L.explosion(r, 1.4, 4.0, "stone", 90).x
    yield "stone_crack", L.stone_crack(r, 0.5).x
    yield "marble_shatter_void", L.shatter(r, "marble", 1.4, 200, 4.0, gravity=False).x
    yield "slop_splash", B.splash(r, 3.0).x
    yield "drips", L.scatter(L.ns(3.0), [(0.2 + 0.6 * k, L.drip(r).x, 1.0, 0.0) for k in range(4)])
    yield "crt_on", L.crt_on(r).x
    yield "crt_off", L.crt_off(r).x
    yield "tv_static", L.tv_static(r, 1.5)
    yield "pen_scratch", L.pen_scratch(r, 0.8).x
    yield "ka_ching", L.ka_ching(r).x
    yield "cryo_pulse", L.cryo_pulse(r, 4.0)
    yield "data_chirps", L.fsk_chirps(r, 3.0, 6)
    yield "velcro", L.velcro(r, 0.3).x
    yield "telescope_clicks", L.telescope(r, 3, 0.12).x
    yield "hammer_taps", L.hammer_tap(r, 3).x
    yield "boots_concrete", L.scatter(L.ns(2.4), [(0.1 + 0.55 * k, L.boot_step(r).x, 1.0, 0.0) for k in range(4)])
    yield "tower_groan", L.creak(r, 3.0, 10, 22, (40, 61, 92, 137, 205), 0.5, 0.6).x
    yield "stone_door", L.stone_slide(r, 2.0).x
    yield "crystal_chime", L.chime(r, 587.33, 5.0, t60=4.0)
    yield "sparkle", L.sparkle(r, 2.0, 24).x
    yield "whoosh", L.whoosh(r, 1.2).x
    yield "pole_thunk", L.pole_thunk(r).x
    yield "page_slam", L.page_slam(r, 1.3).x
    yield "sticky_touch_peel_stretch", L.scatter(L.ns(2.6), [(0.1, L.squelch_touch(r).x, 1.0, 0.0),
                                                               (0.6, B.sticky_peel(r).x, 1.0, 0.0),
                                                               (1.0, L.squelch_stretch(r, 1.4).x, 1.0, 0.0)])
    yield "sniff", L.sniff(r, 2).x
    yield "birds", L.scatter(L.ns(4.0), [(0.4 * k, L.bird(r, sp, 20).x, 1.0, 0.0)
                                        for k, sp in enumerate(["whistle", "trill", "warble", "chip"] * 2)])
    yield "klaxon", L.klaxon_whoop(r).x
    yield "crowd_wave", FF.crowd_wave(0.0, 3.0, FF.procedural_wave(0.2, 2.0, 800, seed=2), True, 3, 0.35)[0]
    yield "rubbed_glass", None


def rubbed_glass(r):
    d = 4.0
    n = L.ns(d)
    t = L.tt(n)
    press = np.interp(t, [0, 0.8, 3.5, 4.0], [0, 0.8, 1.0, 0])
    exc = L.stick_slip(r, d, 587.33, 0.002, press)
    body = L.resonators(exc, [586.6, 588.0, 587.33 * 2.32, 587.33 * 4.25], [3, 3, 1.5, 0.8], [1, 1, 0.35, 0.12])
    beat = 0.55 + 0.45 * np.cos(L.TAU * np.cumsum(np.interp(t, [0, 4], [0.3, 3.0])) / L.SR) ** 2
    return L.norm(body * beat, 0.9)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    its = [(k, v if v is not None else rubbed_glass(L.R(9))) for k, v in items()]
    cols = 5
    rows = (len(its) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows))
    for ax, (name, x) in zip(axs.flat, its):
        x = L.norm(np.asarray(x, np.float32), 0.9)
        sf.write(OUT / f"{name}.wav", x, L.SR, subtype="FLOAT")
        m = x.mean(1) if x.ndim == 2 else x
        f, t, S = sps.spectrogram(m, L.SR, nperseg=1024, noverlap=768)
        S = 10 * np.log10(S + 1e-14)
        ax.pcolormesh(t, f, S, vmin=S.max() - 80, vmax=S.max(), shading="auto", cmap="magma")
        ax.set_yscale("symlog", linthresh=300)
        ax.set_ylim(30, 22000)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=6)
    for ax in list(axs.flat)[len(its):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT.parent / "audition_sheet.png", dpi=60)
    print(f"{len(its)} sounds -> {OUT}")


if __name__ == "__main__":
    main()

"""Plot logistic calibration beside the original manuscript's Fig. 6(b).

Reads the 1,400 saved held-out AMH/AN selection cases, the saved calibration
artifact, and the original density-curve arrays. It does not refit or simulate.
Run:
    python plot_calibration_with_kde.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
OLD_PANEL = (
    RESULTS
    / "Test_700-20261006T065051Z-1-001"
    / "Test_700"
    / "data_plot"
    / "figure_model_selection"
    / "plot_data_npy"
)
EPS = 1e-12  # Same value as calibrate_selection.py.
AMH_COLOR = "#1d9e75"
AN_COLOR = "#d85a30"

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 22,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 400,
    }
)


def main() -> None:
    amh = pd.read_pickle(RESULTS / "amh.pkl")
    an = pd.read_pickle(RESULTS / "an.pkl")
    if len(amh) != 700 or len(an) != 700:
        raise ValueError("Expected 700 held-out AMH and 700 held-out AN cases.")
    if not (amh["true_model"].astype(str) == "AMH").all():
        raise ValueError("AMH results have unexpected truth labels.")
    if not (an["true_model"].astype(str) == "AN").all():
        raise ValueError("AN results have unexpected truth labels.")
    combined = pd.concat([amh, an], ignore_index=True)
    ell = np.log(
        (combined["d_AN"].to_numpy(float) + EPS)
        / (combined["d_AMH"].to_numpy(float) + EPS)
    )
    log10_ratio = ell / np.log(10.0)
    labels = combined["true_model"].astype(str).to_numpy()

    artifact = joblib.load(HERE / "model_selection_calibration_128_500.joblib")
    a = float(artifact["logistic"]["slope"])
    if int(artifact["logistic"]["n"]) != len(combined):
        raise ValueError("Calibration artifact and source cases differ in size.")

    kde_x = np.load(OLD_PANEL / "kde_x.npy")
    kde_amh = np.load(OLD_PANEL / "kde_amh.npy")
    kde_an = np.load(OLD_PANEL / "kde_an.npy")
    if not (len(kde_x) == len(kde_amh) == len(kde_an)):
        raise ValueError("Original Fig. 6(b) KDE arrays have different lengths.")
    if not np.isfinite(np.c_[kde_x, kde_amh, kde_an]).all():
        raise ValueError("Original Fig. 6(b) KDE arrays contain nonfinite values.")

    fig, (ax_reg, ax_old) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )

    # Logistic regression: each dot is an observed binary class label.
    # It is not a probability computed from the fitted curve.
    x = np.linspace(float(log10_ratio.min()), float(log10_ratio.max()), 700)
    for family, value, color, marker in (
            ("AMH", 1, AMH_COLOR, "o"),
            ("AN", 0, AN_COLOR, "s"),
        ):
            mask = labels == family
            ax_reg.scatter(
                log10_ratio[mask],
                np.full(mask.sum(), value),
                s=110,
                alpha=0.28,
                color=color,
                marker=marker,
                edgecolors="none",
                label=rf"Observed {family} ($n={mask.sum()}$)",
                rasterized=True,
                zorder=3,
            )


    ax_reg.plot(
        x,
        # The stored slope was fitted against the natural logarithm. Convert
        # it so the same logistic curve is shown against log10(d_AN/d_AMH).
        expit((a * np.log(10.0)) * x),
        color="#252525",
        linewidth=2.5,
        label=rf"Logistic fit, $a={a * np.log(10.0):.3f}$",
        zorder=4,
    )
    ax_reg.axvline(0, color="0.55", linestyle="--", linewidth=1.2, zorder=1)
    
    ax_reg.set_xlim(log10_ratio.min() - 0.2, log10_ratio.max() + 0.2)
    ax_reg.set_ylim(-0.08, 1.08)
    ax_reg.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax_reg.set_xlabel(r"$\log_{10}(d_{\rm AN}/d_{\rm AMH})$")
    ax_reg.set_ylabel(r"$P_{\mathrm{AMH}}$")
    # ax_reg.set_title("(a) Calibrated logistic regression", loc="left")
    ax_reg.legend(loc="upper left", frameon=False, markerscale=0.85)

    # Reproduce the original Fig. 6(b) density curves from saved plot data.
    ax_old.plot(kde_x, kde_amh, color=AMH_COLOR, linewidth=2.2, label="True AMH")
    ax_old.fill_between(kde_x, kde_amh, color=AMH_COLOR, alpha=0.22)
    ax_old.plot(kde_x, kde_an, color=AN_COLOR, linewidth=2.2, label="True AN")
    ax_old.fill_between(kde_x, kde_an, color=AN_COLOR, alpha=0.22)
    ax_old.axvline(0, color="0.3", linestyle="--", linewidth=0.9, label="Decision boundary")
    ax_old.set_xlim(float(kde_x.min()), float(kde_x.max()))
    ax_old.set_ylim(0, max(float(kde_amh.max()), float(kde_an.max())) * 1.1)
    ax_old.set_xlabel(r"$\log_{10}(d_{\rm AN}/d_{\rm AMH})$")
    ax_old.set_ylabel("Density")
    # ax_old.set_title("(b) Original Fig. 6(b)", loc="left")
    ax_old.legend(loc="upper right", frameon=False)

    output_dir = HERE / "fig"
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = output_dir / f"calibration_with_original_fig6b.{extension}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)
    print(
        f"Used {len(combined)} matched cases; saved natural-log slope a={a:.9f}; "
        f"equivalent base-10 slope a={a * np.log(10.0):.9f}"
    )


if __name__ == "__main__":
    main()

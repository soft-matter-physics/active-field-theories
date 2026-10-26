"""Four-panel modified-AMH robustness plot from saved selection results.

For each nonzero lambda, x is the relative AMH-versus-AN log discrepancy
ratio and y is the AMH discrepancy divided by its calibrated absolute-fit
threshold. Dots are the 100 already evaluated observations. No simulations,
model training, or calibration are performed.

Run from any directory:
    python /absolute/path/to/plot_lambda_selection_quadrants.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
AMH_COLOR = "#1d9e75"
AN_COLOR = "#d85a30"
NEITHER_COLOR = "#4c4c4c"
BOUNDARY_COLOR = "#555555"

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 400,
    }
)


def main() -> None:
    output_dir = HERE / "fig"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_cases = pd.read_csv(HERE / "cases.csv")
    required = {
        "dataset",
        "status",
        "lam",
        "log_discrepancy_ratio",
        "normalized_d_AMH",
        "final_decision",
    }
    missing = required.difference(all_cases.columns)
    if missing:
        raise ValueError(f"cases.csv lacks columns: {sorted(missing)}")
    cases = all_cases[
        (all_cases["status"] == "ok")
        & all_cases["dataset"].str.startswith("AMH_lambda_")
    ].copy()
    cases["log10_ratio"] = cases["log_discrepancy_ratio"] / np.log(10.0)

    lambdas = [-1.0, -0.5, 0.5, 1.0]
    for lam in lambdas:
        group = cases[np.isclose(cases["lam"], lam)]
        if len(group) != 100:
            raise ValueError(f"Expected 100 observations at lambda={lam}, found {len(group)}.")
        if not group["final_decision"].isin(["AMH", "NEITHER"]).all():
            raise ValueError(f"Unexpected final decision at lambda={lam}.")
        if not np.isfinite(group[["log10_ratio", "normalized_d_AMH"]].to_numpy(float)).all():
            raise ValueError(f"Nonfinite plot coordinates at lambda={lam}.")
        if (group["normalized_d_AMH"] <= 0).any():
            raise ValueError("Normalized AMH discrepancies must be positive.")

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 10), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, lam in zip(axes.flat, lambdas):
        group = cases[np.isclose(cases["lam"], lam)]
        chosen = group[group["final_decision"] == "AMH"]
        rejected = group[group["final_decision"] == "NEITHER"]

        # Separate criteria: x=0 is a relative AMH/AN tie; y=1 is the
        # calibrated absolute AMH discrepancy threshold.
        ax.axvline(0, color=BOUNDARY_COLOR, linestyle=":", linewidth=1.0, zorder=1)
        ax.axhline(1, color=BOUNDARY_COLOR, linestyle="--", linewidth=1.0, zorder=1)
        ax.scatter(
            chosen["log10_ratio"],
            chosen["normalized_d_AMH"],
            color=AMH_COLOR,
            marker="o",
            s=68,
            alpha=0.42,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        ax.scatter(
            rejected["log10_ratio"],
            rejected["normalized_d_AMH"],
            color=NEITHER_COLOR,
            marker="x",
            s=100,
            linewidths=2.0,
            zorder=3,
        )
        ax.set_title(
            rf"$\lambda={lam:g}$: ({len(chosen)} AMH, {len(rejected)} NOTT)",
            loc="left",
            fontsize=18,  # Adjust this value down (e.g., 8 or 9) to make it smaller
        )
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlim(-0.20, 2.78)
        ax.set_ylim(0.003, 3.0)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0.01, 0.1, 1])
        ax.grid(axis="y", which="major", color="0.88", linewidth=0.5)
        ax.set_axisbelow(True)

    fig.supxlabel(r"$\log_{10}(d_{\rm AN}/d_{\rm AMH})$")
    fig.supylabel(r"$G_{\rm AMH}=d_{\rm AMH}/T_{\rm AMH}$")
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=AMH_COLOR,
               markersize=8, label="Decision: AMH"),
        Line2D([], [], marker="x", linestyle="none", color=NEITHER_COLOR,
               markersize=9, label="Decision: NOTO"),
        Line2D([], [], linestyle=":", color=BOUNDARY_COLOR,
               linewidth=1, label="Equal discrepancies"),
        Line2D([], [], linestyle="--", color=BOUNDARY_COLOR,
               linewidth=1, label="AMH-fit limit"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=4,
        frameon=False,
        fontsize=16,
        columnspacing=1.0,
        handlelength=1.7,
    )

    for suffix in ("pdf", "png"):
        output = output_dir / f"lambda_relative_and_absolute_fit.{suffix}"
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved {output}")
    plt.close(fig)
    print(
        "Modified AMH outcomes:",
        ", ".join(
            f"lambda={lam:g}: "
            f"{sum((np.isclose(cases['lam'], lam)) & (cases['final_decision'] == 'AMH'))} AMH, "
            f"{sum((np.isclose(cases['lam'], lam)) & (cases['final_decision'] == 'NEITHER'))} NOTT"
            for lam in lambdas
        ),
    )
    plot_standard_cases(all_cases, output_dir)


def plot_standard_cases(all_cases: pd.DataFrame, output_dir: Path) -> None:
    """Compare new standard AMH and AN data using the correct vertical scale."""
    standard = all_cases[
        (all_cases["status"] == "ok")
        & all_cases["dataset"].isin(["AMH", "AN"])
    ].copy()
    if len(standard) != 200:
        raise ValueError(f"Expected 200 standard cases, found {len(standard)}.")
    standard["log10_ratio"] = standard["log_discrepancy_ratio"] / np.log(10.0)
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), sharex=True, sharey=True,
        constrained_layout=True,
    )
    style = {
        "AMH": (AMH_COLOR, "o", 68),
        "AN": (AN_COLOR, "s", 68),
        "NEITHER": (NEITHER_COLOR, "x", 100),
    }
    for ax, family in zip(axes, ("AMH", "AN")):
        group = standard[standard["dataset"] == family]
        if len(group) != 100:
            raise ValueError(f"Expected 100 standard {family} cases.")
        g_column = f"normalized_d_{family}"
        if g_column not in group.columns:
            raise ValueError(f"Missing {g_column} for standard {family}.")
        if not np.isfinite(group[["log10_ratio", g_column]].to_numpy(float)).all():
            raise ValueError(f"Nonfinite coordinates for standard {family}.")
        if (group[g_column] <= 0).any():
            raise ValueError(f"{g_column} must be positive.")

        ax.axvline(0, color=BOUNDARY_COLOR, linestyle=":", linewidth=1, zorder=1)
        ax.axhline(1, color=BOUNDARY_COLOR, linestyle="--", linewidth=1, zorder=1)
        for decision in ("AMH", "AN", "NEITHER"):
            subset = group[group["final_decision"] == decision]
            if subset.empty:
                continue
            color, marker, size = style[decision]
            ax.scatter(
                subset["log10_ratio"],
                subset[g_column],
                color=color,
                marker=marker,
                s=size,
                alpha=0.42 if decision != "NEITHER" else 1.0,
                edgecolors="none" if marker != "x" else None,
                linewidths=2.0 if marker == "x" else 0,
                rasterized=True,
                zorder=3 if decision == "NEITHER" else 2,
            )
        count_text = ", ".join(
            f"{sum(group['final_decision'] == outcome)} {'NOTT' if outcome == 'NEITHER' else outcome}"
            for outcome in ("AMH", "AN", "NEITHER")
            if (group["final_decision"] == outcome).any()
        )
        ax.set_title(f"{family}: ({count_text})", loc="left", fontsize=16)
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(0.001, 2.5)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_yticks([0.01, 0.1, 1])
        ax.set_ylabel(rf"$G_{{\rm {family}}}=d_{{\rm {family}}}/T_{{\rm {family}}}$")
        ax.grid(axis="y", which="major", color="0.88", linewidth=0.5)
        ax.set_axisbelow(True)

    fig.supxlabel(r"$\log_{10}(d_{\rm AN}/d_{\rm AMH})$")
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=AMH_COLOR,
               markersize=8, label="AMH"),
        Line2D([], [], marker="s", linestyle="none", color=AN_COLOR,
               markersize=8, label="AN"),
        Line2D([], [], marker="x", linestyle="none", color=NEITHER_COLOR,
               markersize=9, label="NOTO"),
        Line2D([], [], linestyle=":", color=BOUNDARY_COLOR,
               linewidth=1, label="Equal discrepancies"),
        Line2D([], [], linestyle="--", color=BOUNDARY_COLOR,
               linewidth=1, label="Model-fit limit"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=5,
        frameon=False,
        fontsize=16,
        columnspacing=1.1,
        handlelength=1.7,
    )
    for suffix in ("pdf", "png"):
        output = output_dir / f"standard_amh_an_relative_and_absolute_fit.{suffix}"
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved {output}")
    plt.close(fig)
    print(
        "Standard outcomes:",
        ", ".join(
            f"{family}: "
            + ", ".join(
                f"{sum((standard['dataset'] == family) & (standard['final_decision'] == decision))} {'NOTT' if decision == 'NEITHER' else decision}"
                for decision in ("AMH", "AN", "NEITHER")
                if ((standard["dataset"] == family) & (standard["final_decision"] == decision)).any()
            )
            for family in ("AMH", "AN")
        ),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the ABIDE ablation figure from the adjacent CSV data file."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ablation_data.csv"
FIGURE_DIR = ROOT / "figures"
EXPECTED_GROUPS = ["(a)", "(b)", "(c)", "(d)"]
ACC_COLOR = "#91BEF5"
BA_COLOR = "#F4A7D2"
Y_MIN, Y_MAX = 80.0, 100.0
Y_TICKS = np.arange(80.0, 100.1, 5.0)


def read_data() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    if [row["group"] for row in rows] != EXPECTED_GROUPS:
        raise ValueError(f"Expected groups in order {EXPECTED_GROUPS}")

    numeric_fields = (
        "acc_mean_pct",
        "acc_sd_pct",
        "ba_mean_pct",
        "ba_sd_pct",
    )
    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])

    return rows


def main() -> None:
    rows = read_data()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    centers = np.arange(len(rows), dtype=float)
    bar_width = 0.27
    separation = 0.21
    acc_positions = centers - separation
    ba_positions = centers + separation
    acc_means = np.asarray([row["acc_mean_pct"] for row in rows], dtype=float)
    acc_sds = np.asarray([row["acc_sd_pct"] for row in rows], dtype=float)
    ba_means = np.asarray([row["ba_mean_pct"] for row in rows], dtype=float)
    ba_sds = np.asarray([row["ba_sd_pct"] for row in rows], dtype=float)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig, ax_acc = plt.subplots(figsize=(3.5, 2.5))
    ax_ba = ax_acc.twinx()
    ax_ba.patch.set_visible(False)

    edge_color = "#34383D"
    error_color = "#34383D"
    bar_style = {"edgecolor": edge_color, "linewidth": 0.45, "zorder": 2}
    ax_acc.bar(
        acc_positions,
        acc_means - Y_MIN,
        bottom=Y_MIN,
        width=bar_width,
        color=ACC_COLOR,
        **bar_style,
    )
    ax_acc.bar(
        ba_positions,
        ba_means - Y_MIN,
        bottom=Y_MIN,
        width=bar_width,
        color=BA_COLOR,
        **bar_style,
    )

    error_style = {
        "fmt": "none",
        "ecolor": error_color,
        "elinewidth": 0.7,
        "capsize": 2.0,
        "capthick": 0.7,
        "zorder": 4,
    }
    ax_acc.errorbar(acc_positions, acc_means, yerr=acc_sds, **error_style)
    ax_acc.errorbar(ba_positions, ba_means, yerr=ba_sds, **error_style)

    for positions, means, sds in (
        (acc_positions, acc_means, acc_sds),
        (ba_positions, ba_means, ba_sds),
    ):
        for x_value, mean, sd in zip(positions, means, sds):
            ax_acc.annotate(
                f"{mean:.2f}",
                xy=(x_value, mean + sd),
                xytext=(0, 2.0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5.7,
                color="#24282C",
                clip_on=False,
            )

    ax_acc.set_xlim(-0.56, len(rows) - 0.44)
    ax_acc.set_ylim(Y_MIN, Y_MAX)
    ax_ba.set_ylim(Y_MIN, Y_MAX)
    ax_acc.set_yticks(Y_TICKS)
    ax_ba.set_yticks(Y_TICKS)
    ax_acc.set_xticks(centers, [row["group"] for row in rows])
    ax_acc.set_ylabel("ACC (%)", labelpad=4)
    ax_ba.set_ylabel("BA (%)", labelpad=4)
    ax_acc.spines["top"].set_visible(False)
    ax_ba.spines["top"].set_visible(False)
    ax_ba.spines["left"].set_visible(False)
    ax_ba.spines["bottom"].set_visible(False)
    ax_acc.tick_params(axis="both", length=2.5, width=0.6, color=edge_color, pad=2.5)
    ax_ba.tick_params(axis="y", length=2.5, width=0.6, color=edge_color, pad=2.5)
    ax_acc.grid(False)
    ax_ba.grid(False)

    fig.text(0.5, 0.965, "ABIDE", ha="center", va="top", fontsize=9.5)
    legend_handles = [
        Patch(facecolor=ACC_COLOR, edgecolor=edge_color, linewidth=0.45, label="ACC"),
        Patch(facecolor=BA_COLOR, edgecolor=edge_color, linewidth=0.45, label="BA"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        handleheight=0.75,
        handletextpad=0.4,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.16, right=0.84, bottom=0.19, top=0.79)

    fig.savefig(FIGURE_DIR / "abide_ablation.png", dpi=600)
    fig.savefig(FIGURE_DIR / "abide_ablation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

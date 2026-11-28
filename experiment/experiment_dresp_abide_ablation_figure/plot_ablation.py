#!/usr/bin/env python3
"""Render the ABIDE ablation preview from the adjacent CSV data file."""

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
Y_TICKS = np.arange(80.0, 100.1, 4.0)
PREVIEW_PNG = FIGURE_DIR / "abide_ablation_preview.png"
PREVIEW_PDF = FIGURE_DIR / "abide_ablation_preview.pdf"

# The requested Baseline means are substitute preview values. No reliable
# ten-fold SD for these new means was found, so its preview error bars are
# intentionally omitted. The formal CSV remains the source of the original
# values and is never rewritten by this script.
PREVIEW_BASELINE = {
    "acc_mean_pct": 86.64,
    "acc_sd_pct": None,
    "ba_mean_pct": 86.40,
    "ba_sd_pct": None,
}


def read_data() -> list[dict[str, object]]:
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


def prepare_preview_data(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Apply the requested Baseline substitute values to an in-memory copy."""

    preview_rows = [dict(row) for row in rows]
    preview_rows[0].update(PREVIEW_BASELINE)
    return preview_rows


def metric_array(
    rows: list[dict[str, object]], field: str
) -> np.ndarray:
    """Convert numeric fields while retaining missing SD values as NaN."""

    return np.asarray(
        [np.nan if row[field] is None else float(row[field]) for row in rows],
        dtype=float,
    )


def draw_errorbars(
    ax: plt.Axes,
    positions: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    error_style: dict[str, object],
) -> None:
    """Draw error bars only for metrics with verified SD values."""

    valid = np.isfinite(sds)
    if valid.any():
        ax.errorbar(positions[valid], means[valid], yerr=sds[valid], **error_style)


def annotate_values(
    ax: plt.Axes,
    positions: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    x_offset_points: float,
) -> None:
    """Place mean labels above an error-bar cap or the bar when SD is absent."""

    for x_value, mean, sd in zip(positions, means, sds):
        has_sd = np.isfinite(sd)
        anchor_y = mean + sd if has_sd else mean
        offset_points = 2.0 if has_sd else 2.8
        ax.annotate(
            f"{mean:.2f}",
            xy=(x_value, anchor_y),
            xytext=(x_offset_points, offset_points),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.7,
            color="#24282C",
            clip_on=False,
        )


def main() -> None:
    rows = prepare_preview_data(read_data())
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    centers = np.arange(len(rows), dtype=float)
    bar_width = 0.30
    acc_positions = centers - bar_width / 2.0
    ba_positions = centers + bar_width / 2.0
    acc_means = metric_array(rows, "acc_mean_pct")
    acc_sds = metric_array(rows, "acc_sd_pct")
    ba_means = metric_array(rows, "ba_mean_pct")
    ba_sds = metric_array(rows, "ba_sd_pct")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.6,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
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
    bar_style = {"edgecolor": edge_color, "linewidth": 0.40, "zorder": 2}
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
    draw_errorbars(ax_acc, acc_positions, acc_means, acc_sds, error_style)
    draw_errorbars(ax_acc, ba_positions, ba_means, ba_sds, error_style)
    annotate_values(ax_acc, acc_positions, acc_means, acc_sds, x_offset_points=-2.2)
    annotate_values(ax_acc, ba_positions, ba_means, ba_sds, x_offset_points=2.2)

    ax_acc.set_xlim(-0.56, len(rows) - 0.44)
    ax_acc.set_ylim(Y_MIN, Y_MAX)
    ax_ba.set_ylim(Y_MIN, Y_MAX)
    ax_acc.set_yticks(Y_TICKS)
    ax_ba.set_yticks(Y_TICKS)
    ax_acc.set_xticks(centers, [row["group"] for row in rows])
    ax_acc.set_ylabel("ACC (%)", labelpad=3)
    ax_ba.set_ylabel("BA (%)", labelpad=3)
    ax_acc.spines["top"].set_visible(False)
    ax_ba.spines["top"].set_visible(False)
    ax_ba.spines["left"].set_visible(False)
    ax_ba.spines["bottom"].set_visible(False)
    ax_acc.tick_params(axis="both", length=2.1, width=0.55, color=edge_color, pad=2.0)
    ax_ba.tick_params(axis="y", length=2.1, width=0.55, color=edge_color, pad=2.0)
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

    fig.savefig(PREVIEW_PNG, dpi=600)
    fig.savefig(PREVIEW_PDF)
    plt.close(fig)


if __name__ == "__main__":
    main()

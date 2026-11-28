#!/usr/bin/env python3
"""Create the four-modality DARG preview figure.

The values in this script are supplied preview values.  They are kept
separate from the formal DARG result files and formal figure outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
PDF_PATH = FIGURES / "darg_relation_quality_4mod_preview.pdf"
PNG_PATH = FIGURES / "darg_relation_quality_4mod_preview.png"

MODALITIES = ("PHENO", "ANAT", "FUNC", "CORR")
COLORS = {
    "PHENO": "#4C91D9",
    "ANAT": "#9575CD",
    "FUNC": "#F0AD4E",
    "CORR": "#23B6A8",
}
MARKERS = {
    "PHENO": "o",
    "ANAT": "s",
    "FUNC": "^",
    "CORR": "D",
}

RELATION_GROUPS = ("ASD–ASD", "ASD–TD", "TD–TD")
RELATION_VALUES = np.array(
    [
        [4.64, -3.75, 3.93],
        [7.90, -6.38, 6.69],
        [7.41, -5.98, 6.27],
        [47.65, -38.47, 40.33],
    ],
    dtype=float,
)

CONDITIONS = ("Original", "DARG", "Permuted")
PROPAGATION_VALUES = np.array(
    [
        [49.8816, 47.9126, 49.8816],
        [49.4690, 46.1254, 49.4690],
        [49.2332, 46.0965, 49.2332],
        [48.1945, 25.4142, 48.2344],
    ],
    dtype=float,
)


def configure_matplotlib() -> None:
    """Use an embedded TrueType font and a compact publication style."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "text.color": "#2D333B",
            "axes.labelcolor": "#2D333B",
            "axes.titlecolor": "#2D333B",
            "axes.edgecolor": "#56616D",
            "xtick.color": "#56616D",
            "ytick.color": "#56616D",
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def validate_values() -> None:
    """Guard the figure against accidental changes to the supplied values."""
    expected_relation = np.array(
        [
            [4.64, -3.75, 3.93],
            [7.90, -6.38, 6.69],
            [7.41, -5.98, 6.27],
            [47.65, -38.47, 40.33],
        ],
        dtype=float,
    )
    expected_propagation = np.array(
        [
            [49.8816, 47.9126, 49.8816],
            [49.4690, 46.1254, 49.4690],
            [49.2332, 46.0965, 49.2332],
            [48.1945, 25.4142, 48.2344],
        ],
        dtype=float,
    )
    np.testing.assert_array_equal(RELATION_VALUES, expected_relation)
    np.testing.assert_array_equal(PROPAGATION_VALUES, expected_propagation)
    assert RELATION_VALUES.shape == (4, 3)
    assert PROPAGATION_VALUES.shape == (4, 3)


def style_axes(ax) -> None:
    """Apply the shared axis, grid, and tick treatment."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6E9ED", linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(axis="both", which="major", length=3.0, width=0.65, pad=2.0)


def draw_relation_panel(ax) -> None:
    """Draw grouped four-modality relation-adjustment bars."""
    group_positions = np.arange(len(RELATION_GROUPS), dtype=float)
    bar_width = 0.145
    offsets = (np.arange(len(MODALITIES), dtype=float) - 1.5) * 0.17

    for modality_index, modality in enumerate(MODALITIES):
        positions = group_positions + offsets[modality_index]
        bars = ax.bar(
            positions,
            RELATION_VALUES[modality_index],
            width=bar_width,
            color=COLORS[modality],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=modality,
        )
        assert len(bars) == len(RELATION_GROUPS)

        if modality == "CORR":
            for bar, value in zip(bars, RELATION_VALUES[modality_index]):
                label_y = value + 1.45 if value >= 0 else value - 1.45
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    f"{value:+.2f}%",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=5.6,
                    fontweight="semibold",
                    color="#2D333B",
                    clip_on=True,
                    zorder=4,
                )

    ax.axhline(0.0, color="#56616D", linewidth=0.9, zorder=2)
    ax.set_xlim(-0.53, 2.53)
    ax.set_ylim(-50.0, 60.0)
    ax.set_yticks(np.arange(-40.0, 61.0, 20.0))
    ax.set_xticks(group_positions, RELATION_GROUPS)
    ax.set_ylabel("Relative Edge Weight Adjustment (%)", labelpad=6.0)
    ax.set_title("(a) Disease Relation Reweighting", loc="left", pad=5.0, fontweight="semibold")
    style_axes(ax)


def draw_propagation_panel(ax) -> None:
    """Draw the four-modality propagation lines with small x offsets."""
    condition_positions = np.arange(len(CONDITIONS), dtype=float)
    x_offsets = np.array([-0.070, -0.023, 0.023, 0.070], dtype=float)

    for modality_index, modality in enumerate(MODALITIES):
        x_positions = condition_positions + x_offsets[modality_index]
        ax.plot(
            x_positions,
            PROPAGATION_VALUES[modality_index],
            color=COLORS[modality],
            linewidth=1.25,
            marker=MARKERS[modality],
            markersize=4.7,
            markerfacecolor=COLORS[modality],
            markeredgecolor="white",
            markeredgewidth=0.65,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
            label=modality,
        )

    ax.set_xlim(-0.23, 2.23)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0.0, 101.0, 20.0))
    ax.set_xticks(condition_positions, CONDITIONS)
    ax.set_ylabel("Cross-Disease Weight Ratio (%)", labelpad=6.0)
    ax.set_title("(b) Cross-Disease Propagation", loc="left", pad=5.0, fontweight="semibold")
    style_axes(ax)


def make_figure() -> None:
    configure_matplotlib()
    validate_values()
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 3.15), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.06, 1.00),
        wspace=0.34,
        left=0.095,
        right=0.985,
        bottom=0.22,
        top=0.79,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    draw_relation_panel(ax_a)
    draw_propagation_panel(ax_b)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[modality],
            linewidth=1.25,
            marker=MARKERS[modality],
            markersize=4.7,
            markerfacecolor=COLORS[modality],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=modality,
        )
        for modality in MODALITIES
    ]
    fig.legend(
        handles=legend_handles,
        labels=list(MODALITIES),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=4,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.40,
        columnspacing=1.35,
        borderaxespad=0.0,
    )

    fig.savefig(PDF_PATH, format="pdf", facecolor="white")
    fig.savefig(PNG_PATH, format="png", dpi=600, facecolor="white")
    plt.close(fig)

    print(f"wrote {PDF_PATH}")
    print(f"wrote {PNG_PATH}")


if __name__ == "__main__":
    make_figure()

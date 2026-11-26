from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[1]
COORD = EXP / 'results/tsne_coordinates.csv'
MODELS = ('SPromptGL', 'DARG', 'DARG_JSL', 'DReSP')
TITLES = {'SPromptGL': '(a) SPromptGL', 'DARG': '(b) DARG', 'DARG_JSL': '(c) DARG + JSL', 'DReSP': '(d) DReSP'}
COLORS = {0: '#D98282', 1: '#4C9A91'}
LABELS = {0: 'ASD', 1: 'TD'}


def read_rows():
    with COORD.open(newline='') as f:
        return list(csv.DictReader(f))


def scatter(ax, rows, title):
    for label in (0, 1):
        q = [r for r in rows if int(r['true_label']) == label]
        ax.scatter([float(r['tsne_x']) for r in q], [float(r['tsne_y']) for r in q],
                   s=7, alpha=0.72, c=COLORS[label], label=LABELS[label], linewidths=0)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    rows = read_rows()
    individual = EXP / 'results/figures/individual'
    comparison = EXP / 'results/figures/comparison'
    for model in MODELS: (individual / model).mkdir(parents=True, exist_ok=True)
    comparison.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 11):
        fold_rows = {m: [r for r in rows if r['model'] == m and int(r['fold']) == fold] for m in MODELS}
        for model in MODELS:
            fig, ax = plt.subplots(figsize=(4.2, 3.5), facecolor='white')
            scatter(ax, fold_rows[model], TITLES[model])
            ax.legend(frameon=False, loc='upper right', fontsize=8, markerscale=1.4, handletextpad=0.3)
            fig.tight_layout(pad=0.8)
            fig.savefig(individual / model / f'fold{fold:02d}.png', dpi=args.dpi, facecolor='white')
            plt.close(fig)
        fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.5), facecolor='white', constrained_layout=True)
        for ax, model in zip(axes, MODELS):
            scatter(ax, fold_rows[model], TITLES[model])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc='lower center', ncol=2, fontsize=9,
                   bbox_to_anchor=(0.5, -0.01), markerscale=1.4, handletextpad=0.3)
        fig.savefig(comparison / f'fold{fold:02d}_comparison.png', dpi=args.dpi, facecolor='white', bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()

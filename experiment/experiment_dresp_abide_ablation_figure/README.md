# DReSP ABIDE Ablation Figure

## Scope

This standalone experiment directory renders the ABIDE ablation figure only. It does not train models or write to model code, checkpoints, or historical result directories. The four supplied ten-fold mean and standard deviation values are preserved at four decimal places in `ablation_data.csv`; plotted labels show each mean rounded to two decimals. Error bars use the unrounded SD values from that CSV.

## Groups and model definitions

| Panel | Figure label | Historical identity | Definition |
|---|---|---|---|
| (a) | Baseline | E43.4-E0 | CAM-only. PEN and GFR were physically removed. |
| (b) | Baseline + DARG | E74.4-G2 | DARG includes GFR and its patient-graph learning path, including GraphLearn and GCN. |
| (c) | Baseline + DARG + JSL | `experiment_darg_jsl` | DARG/GFR and CAM144 with the JSL module. This is the no-DSE JSL ablation. |
| (d) | Baseline + DARG + DPSL (DSE + JSL) | `DPSL_full_retrain_from_warmup` | Full DReSP group. DPSL includes DSE and JSL. |

The E43.4-E0 identity and values were checked against the Experiment 43.4 audited entry in `/model/spromptgl/experiment/handoff.md`. The recorded source directory `/data/adodas/SPromptGL-main/experiment/experiment43_4_physical_cam_only_validation` is not present in the current mounts, so its raw run files could not be independently re-read during figure production. The other three identities and values were checked against their corresponding handoff entries and reports under `/model/spromptgl/experiment/`, including `experiment_darg_jsl/results/report.md` and `DPSL_full_retrain_from_warmup/results/report.md`.

## Metric and data provenance

The plotted second metric is BA, based on the historical hard-prediction Hard-AUC/BA field. It is not probability ROC-AUC. `Prob-AUC` is not included. The data rows are:

| Panel | ACC (%) | BA (%) |
|---|---:|---:|
| (a) | 91.7294 ± 3.0819 | 91.4880 ± 3.0586 |
| (b) | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 |
| (c) | 92.7638 ± 2.6289 | 92.5968 ± 2.6616 |
| (d) | 93.1087 ± 2.6257 | 93.0348 ± 2.6450 |

Values were supplied for this figure and cross-checked against historical summaries where available. They were not adjusted, smoothed, or estimated. The E43.4 values use the recorded handoff summary as the available historical evidence because the original result directory is unavailable in the current mount.

## Figure method

`plot_ablation.py` reads `ablation_data.csv` and uses NumPy and Matplotlib to produce a grouped chart with four equally spaced groups and two bars per group. ACC is light blue (`#91BEF5`) and BA is light pink (`#F4A7D2`). The left and right axes both span 80–100% with identical 5-point ticks. Error bars show the exact ten-fold SD, with end caps. Mean labels display two decimal places above the upper error-bar caps. The chart uses a white background without grid lines, significance markers, gain labels, or probability AUC. PDF fonts are embedded as TrueType using `pdf.fonttype=42`.

The style reference is Fig. 2 of [SPromptGL: Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease Prediction](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf). Its ABIDE panel uses four `(a)`–`(d)` groups, paired metric bars, error bars, and value labels. The current [SPromptGL GitHub repository](https://github.com/tangaii/SPromptGL) and repository-local Python files were checked. No reusable Python/Matplotlib plotting script was located in the current repository tree.

## Figure caption

Fig. X. Ablation study on the ABIDE dataset.
(a) Baseline (CAM-only);
(b) Baseline + DARG;
(c) Baseline + DARG + JSL;
(d) Baseline + DARG + DPSL (DSE + JSL).
Bars represent the ten-fold mean values of accuracy (ACC)
and balanced accuracy (BA), with error bars indicating
standard deviations.

## Reproduction

Run from this directory:

```bash
python plot_ablation.py
```

The script writes `figures/abide_ablation.png` at 600 DPI and vector `figures/abide_ablation.pdf`. It requires Python, NumPy, and Matplotlib. No training or checkpoint access is needed to reproduce the figure.

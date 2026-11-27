# DARG Disease-Aware Patient Relation Quality Analysis

This directory contains the final frozen-checkpoint mechanism analysis and the single two-panel MICCAI figure. The experiment uses the formal `E82.4.6.10-G1` task, checkpoint-internal `E82.4.6-G2 SCMR-DA-SRPE`, and `E74.4-G2 Positive-DCR` DARG path. It restores the ten audited Best checkpoints and does not train, fine-tune, select a new checkpoint, or modify model code, checkpoints, historical data, or historical results.

The authoritative directory is `/model/spromptgl/experiment/experiment_darg_relation_quality`. The preferred OSSFS directory failed with a real `Input/output error`, so this is the only execution directory. The server source files are checked against GitHub commit `85826c2` before any fold is evaluated.

Run the complete frozen analysis with:

```bash
python3.12 -I -B /model/spromptgl/experiment/experiment_darg_relation_quality/run_analysis.py
```

The script restores each formal Best checkpoint through the audited fold-aware runtime, verifies archived and independent-evaluator predictions, extracts the formal GraphLearn and DARG adjacency diagnostics, computes `normalize_adj(A + I)`, and performs 20 local score permutations per fold. The score-permuted intervention keeps the original graph support, gates, and q-value multiset and only shuffles q-to-edge assignments.

## Final figure

`figures/darg_relation_quality.pdf` and `figures/darg_relation_quality.png` contain exactly two panels.

Panel (a), `Disease Relation Reweighting`, uses CORR and strict `clean_test_test` support. For every fold it computes `(A_DARG - A_Original) / A_Original` on existing non-self-loop upper-triangle edges and separates ASD-ASD, ASD-TD, and TD-TD connections. The displayed bars are the equal-weight means of the ten fold-level means. Error bars are population SD with `np.std(..., ddof=0)`. Bar colors follow the measured sign.

Panel (b), `Cross-Disease Propagation`, shows all ten paired `Original` and `DARG` CORR ratios in the `at_least_one_clean_test` scope. Both conditions come from the same frozen DReSP checkpoint, with Original denoting its GraphLearn graph before DARG reweighting. The orange points are the 20 permutation-index means, where each point is first averaged across ten folds. No permutation interval is interpreted as a confidence interval.

The propagation ratio uses the formal symmetric normalization of `A + I` and excludes self-loops from evaluated edges. It measures the fraction of normalized propagation weight on cross-disease edges and does not establish a causal classification-accuracy change. The historical `corr_fold1_heatmap.npz` is retained as an audit artifact only and is not used in the final figure.

## Results

The final CORR Panel (a) values are:

| Relation | Mean relative adjustment (%) | SD (%) |
|---|---:|---:|
| ASD-ASD | 47.6497 | 12.4734 |
| ASD-TD | -38.4682 | 6.8587 |
| TD-TD | 40.3348 | 12.5428 |

The historical merged same-disease and cross-disease summaries are reproduced. The same-disease edge-count merge differs from the direct tensor-order calculation by `6.87e-7` percentage points due to float32 summation order.

For Panel (b), the ten-fold means are `48.1945%` Original and `25.4142%` DARG, a paired reduction of `22.7802` percentage points in all ten folds. The strict test-test sensitivity is `51.6350%` to `32.4521%`, a reduction of `19.1829` percentage points in all ten folds. The 20 permutation means have mean `48.2344%` and range `47.8079%` to `48.7299%`.

The independent audit is recorded in `results/audit_status.json` and passes all source, checkpoint, prediction, graph-support, normalization, three-group partition, historical-reproduction, and permutation checks. The complete Chinese report and caption are in `results/report.md`. Compact reproducible figure data are in `results/figure_data.npz`.

The scientific conclusion is `PARTIAL_RELATION_QUALITY_SUPPORT`. The frozen CORR analysis supports same-disease edge enhancement, cross-disease edge suppression, and lower cross-disease propagation-weight fraction. It does not support a claim that every modality improves or that graph-quality changes causally improve classification accuracy.

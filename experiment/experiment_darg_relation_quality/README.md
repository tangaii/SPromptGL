# DARG Disease-Aware Patient Relation Quality Analysis

This frozen mechanism analysis uses the audited `E82.4.6.10-G1` ten-fold Best checkpoints. The checkpoint-internal model is `E82.4.6-G2 SCMR-DA-SRPE`, with `E74.4-G2 Positive-DCR` as the DARG graph path. No model parameter, checkpoint, historical input, or historical result is modified.

The authoritative directory is `/model/spromptgl/experiment/experiment_darg_relation_quality`. The preferred OSSFS directory failed with `Input/output error`, so no duplicate was created there.

Run with `python3.12 -I -B /model/spromptgl/experiment/experiment_darg_relation_quality/run_analysis.py`. The script restores each formal Best checkpoint through `real_model_runtime.py`, calls the verified DARG `rudag_forward` interface, calculates `normalize_adj(A + I)` with the formal utility, performs 20 local score permutations, and writes the report inputs and one two-panel figure.

The primary evaluation set contains edges with at least one endpoint in the current fold's test set after excluding the fold's historical DARG-supervised overlap. The strict sensitivity set contains only clean test-test edges. The figure shows no patient names or identifiable labels. The compact controlled-server artifact stores only the global row indices needed to reproduce the selected heatmap.

## Final MICCAI Figure redesign

The current `figures/darg_relation_quality.pdf` and `.png` are the final two-panel figure. No training or model selection is performed. The script checks the server source files against GitHub commit `9b0bc6b`, restores each fold's audited Best checkpoint through the formal fold adapter, and verifies archived and independent-evaluator predictions.

Panel (a) uses all ten folds on strict clean test-test CORR support. It plots fold-level same-disease and cross-disease means of `(A_DARG - A_Original) / A_Original`, with all fold points shown. Its auxiliary heatmap selects the fold whose original CORR clean test-test edge density is closest to the ten-fold median, breaking ties by lower fold number. It stores ordered global row indices to make the displayed matrix reproducible. `figure_data.npz` remains a controlled-server artifact because those indices can be linked back to the cohort through the source data.

Panel (b) shows the ten-fold paired CORR propagation ratios for the original GraphLearn graph and the DARG-updated graph from the same frozen DReSP checkpoints. The Score-Permuted condition has 20 independent deterministic local-RNG permutations per fold. For each permutation index, results are averaged across all ten folds. The displayed orange interval is the minimum-to-maximum range of these 20 permutation means, not a confidence interval. The y-axis spans the full 0-100% ratio range.

The graph metric uses the formal symmetric normalization `normalize_adj(A + I)` and excludes self-loops from evaluated edges. Reported conclusions concern propagation coefficients and do not imply a causal classification-accuracy gain. Existing `graph_quality_by_fold.csv` and `corr_fold1_heatmap.npz` remain unchanged for historical audit.

Reproduce with:

```bash
python3.12 -I -B /model/spromptgl/experiment/experiment_darg_relation_quality/run_analysis.py
```

The final report, audit, summary, run manifest and compact figure data are in `results/`.

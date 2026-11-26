# Experiment 53.3 — Frozen Design

## Scope

One formal group: `E1_CADR_WD0001_GRAPH_FALLBACK_FIX`, seed 0, ten fixed StratifiedKFold folds. `WD0001` means CADR AdamW `weight_decay=0.001`. The sole model change from Experiment 53.2 `CADR_WD_0p001` is the classification propagation used when that modality's fold-local OOF disease-relation reliability is exactly zero.

## Audited baseline and sources

The 53.2 integrated model produced modality-specific original GraphLearn adjacencies, then multiplied disease positive/negative edge weights by OOF reliability. For reliability zero, both disease graphs became zero and the original code still called the modality GCN with only self-loops. This removed the original patient graph from that branch. The zero-reliability folds are selected by the existing patient-disjoint outer-train OOF rule; the decision uses no outer-test labels.

The Experiment 51 repaired `LegacyFixedGFR` is the base: four original GraphLearn modules, four two-layer full-320D GCNs, four original fusion weights, and the PEN-free original CAM. Existing original optimizer settings remain GFR Adam `lr=0.0038, wd=0.11` and CAM Adam `lr=0.0038, wd=0.11`. Experiment 53.2's WD sweep explicitly overwrote every added AdamW parameter group's weight decay to 0.001, including bias and normalization groups. This run preserves that actual setting: CADR AdamW `lr=0.001`, both added groups `wd=0.001`.

The 53.2 source files, WD=0.001 fold metrics and best checkpoints, 43.1 anchors, ABIDE data, and relation cache were audited read-only. Experiment 53.2 source SHA values and per-fold cache/checkpoint hashes are recorded in `manifest.json`; cache JSON and NPZ bytes are copied and SHA-verified before each fold is prepared.

## Frozen forward definition

For modality `m` with reliability `r_m == 0`, keep its positive and negative disease adjacency and direct disease-difference message at zero, but compute:

`M_m = GCN_m(X_320, normalize_adj(A_original_m + I))`

and fuse exactly as the repaired Experiment 51 GFR:

`H = X_320 + sum_m sigmoid(fusion_weight_m) * M_m`.

There is no reliability gate on this fallback message. For `r_m > 0`, retain the complete Experiment 53.2 branch unchanged:

`M_m = H_pos_m - 0.25 * H_neg_m + 0.02 * D_m`

`H += sigmoid(fusion_weight_m) * sigmoid(reliability_gate_logits_m) * M_m`.

The original CAM is the only classifier and receives the single fused `871 x 320` representation. `use_cadr=False` retains the original E1 behavior. The forward diagnostics label branches `ORIGINAL_FALLBACK` and `CADR_DISEASE`; fallback messages are not counted as disease positive/negative messages.

## Unchanged protocol

ABIDE inputs are 871 x 320 with dimensions 48/6/10/256 and labels ASD=0, TD=1. We retain seed 0, 10-fold `StratifiedKFold(shuffle=True, random_state=10)`, transductive 871-node features, historical CV-before-fold preprocessing, `val=test`, strict test-fold ACC checkpoint selection, ten-epoch warmup, maximum 450 epochs, and stopping when `wait > 200`. The relation cache supplies the same training pairs, OOF reliability and calibration as 53.2. No test label enters relation training or fallback selection.

The comparison is single-seed and uses shared folds. Because checkpoint selection uses `val=test`, preprocessing precedes CV, and graphs are transductive, the metrics are exploratory and potentially optimistic rather than unbiased generalization estimates.

## Minimal source difference

In `cadr_gfr.py`, the reliability-zero path now calls the existing modality GCN on its own original GraphLearn adjacency and uses the original fusion coefficient. The reliable CADR calculation is unchanged. Additional output fields only expose the branch identity and fallback message for audit. No historical source or result is modified.

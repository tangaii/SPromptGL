# Experiment 53.3 — CADR-GFR Original Graph Fallback Fix

## Design and audit

This run changes one CADR-GFR forward branch relative to Experiment 53.2 CADR_WD_0p001. A modality with fold-local OOF reliability exactly zero keeps its disease positive/negative adjacencies and direct disease message at zero, but now propagates through its own original GraphLearn adjacency with the already initialized two-layer GCN and the original sigmoid(fusion_weight) coefficient. Reliable modalities retain the original 53.2 CADR branch. `use_cadr=False` remains the historical E1 baseline path.

The 53.2 relation cache, pair samples, OOF reliability, Platt calibration, untrained anchors, and WD=0.001 fold metrics/checkpoints were read-only sources. Cache JSON/NPZ SHA, split identity, reference checkpoint SHA, and initialization hashes were checked before training. All CADR AdamW parameter groups retain WD=0.001, including bias and LayerNorm groups; base GFR and CAM remain Adam WD=0.11.

Preflight verified exact fallback-message equality with `GCN_m(X, normalize_adj(A_original + I))`, zero disease adjacencies/direct messages for fallback modalities, unchanged CORR CADR outputs against the 53.2 implementation, fallback persistence under `positive_off`, and nonzero classification gradients to the fallback GCN and GraphLearn. The zero-reliability decision is fold-local and derived only from inner patient-disjoint outer-train OOF data.

## Classification results

| Metric | 53.2 WD=0.001 reference | 53.3 fallback fix | Delta (pp) | W/T/L |
|---|---:|---:|---:|---:|
| ACC | 91.9592 ± 2.8219% | 91.7294 ± 2.9950% | -0.2299 | 2/4/4 |
| Hard-AUC | 91.6845 ± 2.8990% | 91.6046 ± 2.9574% | -0.0799 | 5/1/4 |
| Probability-AUC | 93.1977 ± 3.1799% | 93.5681 ± 3.0886% | +0.3705 | 5/0/5 |
| SPE-TD | 95.2775 ± 3.7181% | 93.1360 ± 4.3174% | -2.1415 | 1/1/8 |
| SEN-ASD | 88.0915 ± 5.2075% | 90.0732 ± 3.5288% | +1.9817 | 4/6/0 |
| BA | 91.6845 ± 2.8990% | 91.6046 ± 2.9574% | -0.0799 | 5/1/4 |
| ASD-F1 | 90.9944 ± 3.1878% | 91.0065 ± 3.1369% | +0.0121 | 5/1/4 |

## Modality fallback and gradient diagnostics

| Modality | Zero-reliability folds | Propagation mode | Original edges | Fallback message/input | GCN cls grad | GraphLearn cls grad |
|---|---:|---|---:|---:|---:|---:|
| PHENO | 10/10 | ORIGINAL_FALLBACK | 76240.6 | 7.393e-05 | 1.451e-02 | 2.652e-05 |
| ANAT | 10/10 | ORIGINAL_FALLBACK | 143741.4 | 4.340e-05 | 1.085e-02 | 2.130e-05 |
| FUNC | 10/10 | ORIGINAL_FALLBACK | 69009.9 | 1.698e-05 | 4.669e-03 | 3.470e-06 |
| CORR | 0/10 | CADR_DISEASE | 59863.9 | 0.000e+00 | 6.985e-04 | 2.794e-04 |

## Paired differences

| Metric | Mean paired difference | Wins / ties / losses |
|---|---:|---:|
| ACC | -0.2299 pp | 2/4/4 |
| Hard-AUC | -0.0799 pp | 5/1/4 |
| Probability-AUC | +0.3705 pp | 5/0/5 |
| SPE-TD | -2.1415 pp | 1/1/8 |
| SEN-ASD | +1.9817 pp | 4/6/0 |

## Interpretation and limitations

The table compares the new run fold-by-fold with the independently audited 53.2 WD=0.001 group. A restored original-graph message establishes that the base patient graph reaches the unreliable modality's GCN; it does not by itself establish a diagnostic gain. The `val=test` checkpoint-selection rule, CV-before-fold preprocessing, and transductive 871-node graphs can make estimates optimistic. This single-seed, shared-cohort comparison is exploratory, not an unbiased external generalization estimate.

## Per-modality fused contribution and fixed-checkpoint interventions

The contribution norm is the modality message multiplied by the fusion coefficient; fallback modalities use only `sigmoid(fusion_weight)`, while disease-reliable modalities also use their learned reliability gate.

| Modality | Propagation | Mean fused contribution / input norm |
|---|---|---:|
| PHENO | ORIGINAL_FALLBACK | 4.097e-05 |
| ANAT | ORIGINAL_FALLBACK | 2.396e-05 |
| FUNC | ORIGINAL_FALLBACK | 9.514e-06 |
| CORR | CADR_DISEASE | 2.508e-03 |

At the same best checkpoint, `positive_off_disease_only` leaves PHENO's fallback message bitwise unchanged; `modality_off_PHENO` removes its full fallback contribution. These are evaluation-only interventions, not additional training groups.

| Intervention | Mean ACC delta (pp) | Prediction flips / test patients | Mean representation delta norm |
|---|---:|---:|---:|
| positive_off_disease_only | -0.2299 | 2/871 | 0.695163 |
| modality_off_PHENO | -0.1149 | 1/871 | 0.00968176 |

# Experiment 81.2: End-to-End Reliability-Gated Relational Correction

Status: PASS; formal training 20/20; independent best checkpoint audits 20/20; latest state audits 20/20.
Formal directory: `/model/spromptgl/experiment/experiment81_2_e2e_relational_correction`. Training wall time: 1643.4 s.
All six metrics are percentages, reported as ten-fold mean ± population SD. Hard-AUC is historical balanced accuracy from hard predictions; Prob-AUC is probability ROC-AUC.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E81.2-G1-Self-Local-E2E | 91.6144 ± 2.9170 | 91.3569 ± 2.9528 | 93.8158 ± 2.7896 | 88.0793 ± 5.1166 | 94.6346 ± 4.5612 | 90.6595 ± 3.2530 |
| E81.2-G2-Cross-Local-E2E | 91.9592 ± 2.9591 | 91.7559 ± 2.9767 | 93.5520 ± 2.7364 | 89.0854 ± 4.0402 | 94.4265 ± 3.6455 | 91.1175 ± 3.2202 |
| E74.4-G2-Positive-DCR | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 | 93.4946 ± 2.9195 | 90.0671 ± 3.8862 | 93.9917 ± 4.7320 | 91.4590 ± 2.9013 |
| E81.1-G1-Visual-PRCL-v2 | 92.0742 ± 2.9789 | 91.9051 ± 2.8926 | 93.4621 ± 3.0770 | 89.8232 ± 3.2483 | 93.9870 ± 5.1280 | 91.3509 ± 3.0094 |

## Paired comparisons

Differences are left minus right in percentage points. W/T/L and t-based 95% intervals are descriptive across folds.

| Comparison | Scope | Metric | Mean Δ (95% interval) | W/T/L |
|---|---|---|---:|---:|
| G1_vs_DARG | all10 | ACC | -0.5747 (-1.1561, +0.0067) | 0/6/4 |
| G1_vs_DARG | all10 | Hard-AUC | -0.6724 (-1.3647, +0.0198) | 0/5/5 |
| G1_vs_DARG | all10 | Prob-AUC | +0.3212 (-0.1069, +0.7494) | 4/0/6 |
| G1_vs_DARG | all10 | SEN-ASD | -1.9878 (-4.3409, +0.3653) | 0/6/4 |
| G1_vs_DARG | all10 | SPE-TD | +0.6429 (-0.6133, +1.8991) | 3/6/1 |
| G1_vs_DARG | all10 | ASD-F1 | -0.7996 (-1.6430, +0.0439) | 0/5/5 |
| G1_vs_DARG | without_fold2 | ACC | -0.5109 (-1.1527, +0.1310) | 0/6/3 |
| G1_vs_DARG | without_fold2 | Hard-AUC | -0.6117 (-1.3849, +0.1616) | 0/5/4 |
| G1_vs_DARG | without_fold2 | Prob-AUC | +0.2686 (-0.2001, +0.7372) | 3/0/6 |
| G1_vs_DARG | without_fold2 | SEN-ASD | -1.9377 (-4.6164, +0.7411) | 0/6/3 |
| G1_vs_DARG | without_fold2 | SPE-TD | +0.7144 (-0.7055, +2.1342) | 3/5/1 |
| G1_vs_DARG | without_fold2 | ASD-F1 | -0.7375 (-1.6854, +0.2104) | 0/5/4 |
| G2_vs_DARG | all10 | ACC | -0.2299 (-0.7499, +0.2902) | 1/6/3 |
| G2_vs_DARG | all10 | Hard-AUC | -0.2735 (-0.9108, +0.3639) | 1/6/3 |
| G2_vs_DARG | all10 | Prob-AUC | +0.0574 (-0.5100, +0.6248) | 7/0/3 |
| G2_vs_DARG | all10 | SEN-ASD | -0.9817 (-3.3680, +1.4046) | 1/6/3 |
| G2_vs_DARG | all10 | SPE-TD | +0.4348 (-0.7827, +1.6522) | 2/7/1 |
| G2_vs_DARG | all10 | ASD-F1 | -0.3415 (-1.1236, +0.4405) | 1/6/3 |
| G2_vs_DARG | without_fold2 | ACC | -0.2554 (-0.8444, +0.3336) | 1/5/3 |
| G2_vs_DARG | without_fold2 | Hard-AUC | -0.3038 (-1.0260, +0.4183) | 1/5/3 |
| G2_vs_DARG | without_fold2 | Prob-AUC | -0.0423 (-0.6357, +0.5512) | 6/0/3 |
| G2_vs_DARG | without_fold2 | SEN-ASD | -1.0908 (-3.7959, +1.6144) | 1/5/3 |
| G2_vs_DARG | without_fold2 | SPE-TD | +0.4831 (-0.8988, +1.8650) | 2/6/1 |
| G2_vs_DARG | without_fold2 | ASD-F1 | -0.3795 (-1.2654, +0.5065) | 1/5/3 |
| G2_vs_G1 | all10 | ACC | +0.3448 (-0.3321, +1.0218) | 5/3/2 |
| G2_vs_G1 | all10 | Hard-AUC | +0.3990 (-0.3776, +1.1755) | 5/3/2 |
| G2_vs_G1 | all10 | Prob-AUC | -0.2639 (-0.9155, +0.3878) | 5/0/5 |
| G2_vs_G1 | all10 | SEN-ASD | +1.0061 (-1.3888, +3.4010) | 4/4/2 |
| G2_vs_G1 | all10 | SPE-TD | -0.2081 (-1.3367, +0.9204) | 2/5/3 |
| G2_vs_G1 | all10 | ASD-F1 | +0.4580 (-0.4618, +1.3778) | 5/3/2 |
| G2_vs_G1 | without_fold2 | ACC | +0.2554 (-0.4808, +0.9917) | 4/3/2 |
| G2_vs_G1 | without_fold2 | Hard-AUC | +0.3078 (-0.5454, +1.1611) | 4/3/2 |
| G2_vs_G1 | without_fold2 | Prob-AUC | -0.3109 (-1.0436, +0.4219) | 4/0/5 |
| G2_vs_G1 | without_fold2 | SEN-ASD | +0.8469 (-1.8515, +3.5453) | 3/4/2 |
| G2_vs_G1 | without_fold2 | SPE-TD | -0.2313 (-1.5161, +1.0536) | 2/4/3 |
| G2_vs_G1 | without_fold2 | ASD-F1 | +0.3580 (-0.6581, +1.3741) | 4/3/2 |
| G2_vs_E81.1_G1_context | all10 | ACC | -0.1149 (-0.7217, +0.4918) | 2/5/3 |
| G2_vs_E81.1_G1_context | all10 | Hard-AUC | -0.1492 (-0.8000, +0.5016) | 2/3/5 |
| G2_vs_E81.1_G1_context | all10 | Prob-AUC | +0.0899 (-0.6806, +0.8604) | 5/0/5 |
| G2_vs_E81.1_G1_context | all10 | SEN-ASD | -0.7378 (-3.2421, +1.7665) | 2/5/3 |
| G2_vs_E81.1_G1_context | all10 | SPE-TD | +0.4394 (-1.5880, +2.4668) | 2/5/3 |
| G2_vs_E81.1_G1_context | all10 | ASD-F1 | -0.2334 (-0.9780, +0.5111) | 2/3/5 |
| G2_vs_E81.1_G1_context | without_fold2 | ACC | -0.2554 (-0.8444, +0.3336) | 1/5/3 |
| G2_vs_E81.1_G1_context | without_fold2 | Hard-AUC | -0.3013 (-0.9310, +0.3284) | 1/3/5 |
| G2_vs_E81.1_G1_context | without_fold2 | Prob-AUC | +0.0056 (-0.8452, +0.8564) | 4/0/5 |
| G2_vs_E81.1_G1_context | without_fold2 | SEN-ASD | -1.0908 (-3.7959, +1.6144) | 1/5/3 |
| G2_vs_E81.1_G1_context | without_fold2 | SPE-TD | +0.4882 (-1.8190, +2.7955) | 2/4/3 |
| G2_vs_E81.1_G1_context | without_fold2 | ASD-F1 | -0.4102 (-1.1259, +0.3055) | 1/3/5 |

## Mechanism

Frozen best-checkpoint audits distinguish the current trained base OutputLayer branch, its corrected branch, and the historical E74.4-G2 checkpoint.

| Group | mean |delta| | Nonzero correction | Probability MAE vs current base | Flips vs current base | Corrected/harmed vs DARG | Null identity | Mean Null | Mean entropy |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| E81.2-G1-Self-Local-E2E | 0.002271 | 0.9966 | 0.000508 | 0.20 | 4/9 | True | 0.1095 | 0.9524 |
| E81.2-G2-Cross-Local-E2E | 0.019538 | 0.9977 | 0.004252 | 0.50 | 5/7 | True | 0.0438 | 0.7108 |

The complete per-fold mechanisms are in `mechanism.csv`. `corrected/harmed vs DARG` counts patient decisions changed from the historical best checkpoint. Current-base comparisons identify how much change occurs after the new end-to-end training trajectory versus after the bounded correction itself. The G2-to-G1 replacement is a frozen inference intervention, not an independently trained ablation.

## Excluding Fold 2

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E81.2-G1-Self-Local-E2E | 91.7044 ± 3.0615 | 91.4364 ± 3.1024 | 93.8828 ± 2.9329 | 88.1098 ± 5.3925 | 94.7631 ± 4.7907 | 90.7328 ± 3.4211 |
| E81.2-G2-Cross-Local-E2E | 91.9598 ± 3.1191 | 91.7442 ± 3.1375 | 93.5720 ± 2.8837 | 88.9566 ± 4.2392 | 94.5318 ± 3.8282 | 91.0908 ± 3.3934 |
| E74.4-G2-Positive-DCR | 92.2153 ± 2.9131 | 92.0481 ± 2.8723 | 93.6142 ± 3.0541 | 90.0474 ± 4.0959 | 94.0487 ± 4.9847 | 91.4703 ± 3.0580 |
| E81.1-G1-Visual-PRCL-v2 | 92.2153 ± 3.1082 | 92.0455 ± 3.0166 | 93.5664 ± 3.2267 | 90.0474 ± 3.3497 | 94.0436 ± 5.4024 | 91.5010 ± 3.1365 |

## Decision

- Gate A performance floor: G2 ACC 91.9592% vs DARG 92.1891%; Hard-AUC 91.7559% vs 92.0294%.
- Gate B cross-local value: G2−G1 ACC +0.3448 pp, W/T/L 5/3/2; paired rows are in `paired_comparisons.csv`.
- Gate C net historical correction: G2−DARG ACC -0.2299 pp, W/T/L 1/6/3; patient-level corrected and harmed counts are reported per fold.
- Gate D end-to-end: all 10 folds pass base-off parity; public RNG triplet mismatch count is 0.
- Interpretation is preliminary and protocol-matched, not external clinical validation.

## Audit limits

Historical checkpoint selection uses test-fold ACC (validation equals test); preprocessing and RFE precede CV; the 871-patient graph is transductive; Fold 2 rows 377 and 407 overlap historical train/test and remain in DARG training while their new correction is forced to zero during training. These choices preserve historical comparability and bias generalization estimates. Fold intervals are descriptive because the training folds overlap. CAM144's 16 fixed coordinate slices are not anatomical regions. G1 repeats each local as its 15 candidates, changing the Null competition prior.

## Artifacts

Training code SHA: `554b7f2a21a9b0e13bb41959c39fe5260319302cc07319c1041c1fb3b4982c1a`. Fold metrics: `results/fold_metrics.csv`; paired comparisons: `results/paired_comparisons.csv`; mechanism: `results/mechanism.csv`; audit: `results/audit_status.json`.

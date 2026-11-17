# Experiment 81.3 — End-to-End Gradient-Protected Pairwise Relational Learning

Status: PASS; formal training 20/20; independent best-checkpoint audits 20/20; latest-state audits 20/20.
Formal directory: `/model/spromptgl/experiment/experiment81_3_e2e_pairwise_relational_learning`. Training wall time: 751.7 s.
The paper and official/project repositories were checked before implementation. The original GFR/CAM/PEN description and 10-fold ACC/AUC/SEN/SPE reporting were retained for provenance; the formal model here remains the historical PEN-free E74.4 CAM/DARG base.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E81.3-G1-UnaryAttention-E2E | 91.7294 ± 2.8130 | 91.5648 ± 2.7767 | 93.8486 ± 2.8233 | 89.3415 ± 3.1147 | 93.7882 ± 3.7905 | 90.9350 ± 2.9815 |
| E81.3-G2-PairwiseRelation-E2E | 91.7294 ± 2.9505 | 91.5401 ± 2.9690 | 93.4772 ± 2.8732 | 89.0793 ± 4.0349 | 94.0009 ± 3.4622 | 90.8841 ± 3.1806 |
| E74.4-G2-Positive-DCR | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 | 93.4946 ± 2.9195 | 90.0671 ± 3.8862 | 93.9917 ± 4.7320 | 91.4590 ± 2.9013 |
| E81.2-G2-Cross-Local-E2E | 91.9592 ± 2.9591 | 91.7559 ± 2.9767 | 93.5520 ± 2.7364 | 89.0854 ± 4.0402 | 94.4265 ± 3.6455 | 91.1175 ± 3.2202 |

## Paired comparisons

Differences are left minus right in percentage points. W/T/L and t-based 95% intervals are descriptive across overlapping folds.

| Comparison | Scope | Metric | Mean Δ (95% interval) | W/T/L |
|---|---|---|---:|---:|
| G1_vs_DARG | all10 | ACC | -0.4598 (-1.0347, +0.1152) | 1/4/5 |
| G1_vs_DARG | all10 | Hard-AUC | -0.4646 (-1.1837, +0.2546) | 1/3/6 |
| G1_vs_DARG | all10 | Prob-AUC | +0.3540 (-0.0277, +0.7357) | 8/0/2 |
| G1_vs_DARG | all10 | SEN-ASD | -0.7256 (-3.8617, +2.4105) | 1/4/5 |
| G1_vs_DARG | all10 | SPE-TD | -0.2035 (-2.1740, +1.7669) | 2/6/2 |
| G1_vs_DARG | all10 | ASD-F1 | -0.5240 (-1.3780, +0.3300) | 1/3/6 |
| G1_vs_DARG | without_fold2 | ACC | -0.3831 (-1.0079, +0.2416) | 1/4/4 |
| G1_vs_DARG | without_fold2 | Hard-AUC | -0.3807 (-1.1712, +0.4099) | 1/3/5 |
| G1_vs_DARG | without_fold2 | Prob-AUC | +0.3462 (-0.0883, +0.7808) | 7/0/2 |
| G1_vs_DARG | without_fold2 | SEN-ASD | -0.5352 (-4.0756, +3.0051) | 1/4/4 |
| G1_vs_DARG | without_fold2 | SPE-TD | -0.2261 (-2.4711, +2.0189) | 2/5/2 |
| G1_vs_DARG | without_fold2 | ASD-F1 | -0.4313 (-1.3748, +0.5122) | 1/3/5 |
| G2_vs_DARG | all10 | ACC | -0.4598 (-1.0347, +0.1152) | 1/4/5 |
| G2_vs_DARG | all10 | Hard-AUC | -0.4893 (-1.1515, +0.1730) | 2/3/5 |
| G2_vs_DARG | all10 | Prob-AUC | -0.0174 (-0.3939, +0.3590) | 4/0/6 |
| G2_vs_DARG | all10 | SEN-ASD | -0.9878 (-3.5004, +1.5247) | 2/4/4 |
| G2_vs_DARG | all10 | SPE-TD | +0.0093 (-1.6091, +1.6276) | 2/6/2 |
| G2_vs_DARG | all10 | ASD-F1 | -0.5749 (-1.3487, +0.1989) | 2/3/5 |
| G2_vs_DARG | without_fold2 | ACC | -0.3831 (-1.0079, +0.2416) | 1/4/4 |
| G2_vs_DARG | without_fold2 | Hard-AUC | -0.4081 (-1.1334, +0.3171) | 2/3/4 |
| G2_vs_DARG | without_fold2 | Prob-AUC | -0.0076 (-0.4358, +0.4207) | 4/0/5 |
| G2_vs_DARG | without_fold2 | SEN-ASD | -0.8266 (-3.6598, +2.0067) | 2/4/3 |
| G2_vs_DARG | without_fold2 | SPE-TD | +0.0103 (-1.8342, +1.8548) | 2/5/2 |
| G2_vs_DARG | without_fold2 | ASD-F1 | -0.4879 (-1.3408, +0.3650) | 2/3/4 |
| G2_vs_G1 | all10 | ACC | +0.0000 (-0.3876, +0.3876) | 1/8/1 |
| G2_vs_G1 | all10 | Hard-AUC | -0.0247 (-0.4726, +0.4232) | 2/6/2 |
| G2_vs_G1 | all10 | Prob-AUC | -0.3714 (-0.7914, +0.0485) | 1/0/9 |
| G2_vs_G1 | all10 | SEN-ASD | -0.2622 (-2.5476, +2.0232) | 2/6/2 |
| G2_vs_G1 | all10 | SPE-TD | +0.2128 (-1.6094, +2.0350) | 1/8/1 |
| G2_vs_G1 | all10 | ASD-F1 | -0.0509 (-0.5420, +0.4402) | 2/6/2 |
| G2_vs_G1 | without_fold2 | ACC | +0.0000 (-0.4418, +0.4418) | 1/7/1 |
| G2_vs_G1 | without_fold2 | Hard-AUC | -0.0275 (-0.5379, +0.4829) | 2/5/2 |
| G2_vs_G1 | without_fold2 | Prob-AUC | -0.3538 (-0.8303, +0.1227) | 1/0/8 |
| G2_vs_G1 | without_fold2 | SEN-ASD | -0.2913 (-2.8949, +2.3123) | 2/5/2 |
| G2_vs_G1 | without_fold2 | SPE-TD | +0.2364 (-1.8395, +2.3123) | 1/7/1 |
| G2_vs_G1 | without_fold2 | ASD-F1 | -0.0566 (-0.6161, +0.5030) | 2/5/2 |
| G2_vs_E81.2_G2_context | all10 | ACC | -0.2299 (-0.5766, +0.1168) | 0/8/2 |
| G2_vs_E81.2_G2_context | all10 | Hard-AUC | -0.2158 (-0.6525, +0.2209) | 2/5/3 |
| G2_vs_E81.2_G2_context | all10 | Prob-AUC | -0.0748 (-0.7534, +0.6038) | 5/1/4 |
| G2_vs_E81.2_G2_context | all10 | SEN-ASD | -0.0061 (-2.2173, +2.2051) | 2/5/3 |
| G2_vs_E81.2_G2_context | all10 | SPE-TD | -0.4255 (-1.9975, +1.1464) | 2/6/2 |
| G2_vs_E81.2_G2_context | all10 | ASD-F1 | -0.2334 (-0.7290, +0.2623) | 2/5/3 |
| G2_vs_E81.2_G2_context | without_fold2 | ACC | -0.1277 (-0.4222, +0.1668) | 0/8/1 |
| G2_vs_E81.2_G2_context | without_fold2 | Hard-AUC | -0.1043 (-0.5106, +0.3020) | 2/5/2 |
| G2_vs_E81.2_G2_context | without_fold2 | Prob-AUC | +0.0347 (-0.6852, +0.7547) | 5/1/3 |
| G2_vs_E81.2_G2_context | without_fold2 | SEN-ASD | +0.2642 (-2.1576, +2.6861) | 2/5/2 |
| G2_vs_E81.2_G2_context | without_fold2 | SPE-TD | -0.4728 (-2.2602, +1.3146) | 2/5/2 |
| G2_vs_E81.2_G2_context | without_fold2 | ASD-F1 | -0.1084 (-0.5725, +0.3557) | 2/5/2 |

## Mechanism and gradient protection

The independent audit separates the trained E81.3 base OutputLayer branch, its final corrected branch, and the historical E74.4-G2 checkpoint. The correction branch uses eta=0.1; the preflight eta switch is diagnostic only.

| Group | mean |delta| | P95 | Nonzero | Prob MAE vs base | Flips vs base | Corrected/harmed vs DARG | Null identity | Readout entropy |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| E81.3-G1-UnaryAttention-E2E | 0.363184 | 0.613468 | 1.0000 | 0.026207 | 1.50 | 9/13 | True | 0.9512 |
| E81.3-G2-PairwiseRelation-E2E | 0.345305 | 0.706486 | 1.0000 | 0.024292 | 0.90 | 8/12 | True | 0.9343 |

### Gate and patient-conditioned readout diagnostics

These are descriptive diagnostics; the gate is a learned correction-control signal and the local weights are not anatomical ROI importance.

| Group | Gate mean (P05–P95) | Null weight mean | Attn entropy | Top real weight | Readout entropy | Readout max | High-confidence errors corrected |
|---|---:|---:|---:|---:|---:|---:|---:|
| E81.3-G1-UnaryAttention-E2E | 0.6998 (0.5113–0.9011) | 0.0530 | 0.5622 | 0.4879 | 0.9512 | 0.1306 | 0/23 |
| E81.3-G2-PairwiseRelation-E2E | 0.6487 (0.4021–0.9314) | 0.0391 | 0.7089 | 0.3532 | 0.9343 | 0.1487 | 0/24 |

G2 frozen unary-input intervention (using G2 projection weights): probability MAE 0.019740, hard flips 1.10 per fold, ACC change -1.0319 pp, Hard-AUC change -1.1135 pp, ASD-F1 change -1.2880 pp, relation-summary L2 change 3.9060.

The complete patient-level diagnostics are in `mechanism.csv`. G2 unary replacement is a frozen inference intervention using G2's own projection weights and is not an independently trained control.

## Excluding Fold 2

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E81.3-G1-UnaryAttention-E2E | 91.8321 ± 2.9473 | 91.6674 ± 2.9089 | 93.9605 ± 2.9549 | 89.5122 ± 3.2385 | 93.8226 ± 3.9941 | 91.0389 ± 3.1256 |
| E81.3-G2-PairwiseRelation-E2E | 91.8321 ± 3.0931 | 91.6399 ± 3.1136 | 93.6067 ± 3.0008 | 89.2209 ± 4.2295 | 94.0590 ± 3.6448 | 90.9824 ± 3.3382 |
| E74.4-G2-Positive-DCR | 92.2153 ± 2.9131 | 92.0481 ± 2.8723 | 93.6142 ± 3.0541 | 90.0474 ± 4.0959 | 94.0487 ± 4.9847 | 91.4703 ± 3.0580 |
| E81.2-G2-Cross-Local-E2E | 91.9598 ± 3.1191 | 91.7442 ± 3.1375 | 93.5720 ± 2.8837 | 88.9566 ± 4.2392 | 94.5318 ± 3.8282 | 91.0908 ± 3.3934 |

## Decision

- G2 performance floor: ACC 91.7294% (threshold 92.1891%), Hard-AUC 91.5401% (threshold 92.0294%).
- G2−G1 ACC: +0.0000 pp, W/T/L 1/8/1; G2−DARG ACC: -0.4598 pp; G2−DARG Hard-AUC: -0.4893 pp.
- Scientific classification: **NO_GO_E81_3**.
- This experiment contains no text Prompt, disease prototype, or new modality and therefore does not establish medical Prompt effectiveness.

## Audit limits

Historical selection uses test-fold ACC (validation equals test); preprocessing/RFE precede CV; the patient graph is transductive; Fold 2 retains the historical overlap while suppressing only the new correction during training. Fold intervals are descriptive because training sets overlap. Latent CAM coordinate slices are not anatomical regions.

## Artifacts

Formal training source SHA: `03cc136d62bce081fe5043600dbd1c17dd15bddabf60b2d8ec39ed224ba3825d`; corrected audit/source SHA: `238195c2de37f39e606acb057aeec814380588b32ea4efebbf7fbd268026130b`. Code: `relation_learning.py`, `run.py`, `audit.py`, `path_adapter.py`; results: `results/preflight.json`, `results/fold_metrics.csv`, `results/paired_comparisons.csv`, `results/mechanism.csv`, `results/audit_status.json`, `results/report.md`.

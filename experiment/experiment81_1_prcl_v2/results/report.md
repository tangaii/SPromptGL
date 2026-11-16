# Experiment 81.1: PRCL-v2

Status: COMPLETE_AUDITED; formal folds 40/40; independent best Checkpoint audits 40/40; latest state audits 40/40.
Directory: /model/spromptgl/experiment/experiment81_1_prcl_v2. Training wall time: 1430.0 s; sum of task wall times: 9655.7 s.

All metrics are percentages, reported as ten-fold mean ± population SD. Hard-AUC is the historical hard-prediction balanced-accuracy metric; Prob-AUC is ROC-AUC from class probabilities.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E81.1-G1-Visual-PRCL-v2 | 92.0742 ± 2.9789 | 91.9051 ± 2.8926 | 93.4621 ± 3.0770 | 89.8232 ± 3.2483 | 93.9870 ± 5.1280 | 91.3509 ± 3.0094 |
| E81.1-G2-Medical-Add-PRCL-v2 | 91.7294 ± 2.7174 | 91.4554 ± 2.7470 | 93.2772 ± 3.0712 | 87.8415 ± 3.7557 | 95.0694 ± 2.9098 | 90.7596 ± 3.0289 |
| E81.1-G3-Medical-Interaction-PRCL-v2 | 91.5008 ± 2.8776 | 91.3041 ± 2.8380 | 93.3715 ± 2.9787 | 88.8293 ± 3.9209 | 93.7789 ± 4.6961 | 90.6582 ± 3.0047 |
| E81.1-G4-Gram-Matched-Control-PRCL-v2 | 91.3845 ± 3.0589 | 91.1526 ± 3.0428 | 93.6141 ± 2.8685 | 88.0915 ± 3.9834 | 94.2137 ± 4.2027 | 90.4598 ± 3.2891 |
| E74.4-G2 DARG | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 | 93.4946 ± 2.9195 | 90.0671 ± 3.8862 | 93.9917 ± 4.7320 | 91.4590 ± 2.9013 |

## Historical E81

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E81-G1-Visual-PRCL | 91.8443 ± 2.7486 | 91.6760 ± 2.7403 | 93.5254 ± 2.9076 | 89.5732 ± 4.1636 | 93.7789 ± 4.3793 | 91.0565 ± 2.9186 |
| E81-G2-Medical-PRCL | 91.7294 ± 2.8596 | 91.5090 ± 2.8489 | 93.4620 ± 2.8721 | 88.5915 ± 3.7117 | 94.4265 ± 3.7570 | 90.8523 ± 3.0955 |
| E81-G3-Anonymous-PRCL | 91.8443 ± 2.9346 | 91.6069 ± 2.9758 | 93.5316 ± 3.0353 | 88.5793 ± 4.6346 | 94.6346 ± 3.7826 | 90.9401 ± 3.2784 |

## Excluding Fold 2

Absolute nine-fold mean ± population SD; the paired_comparisons.csv also contains nine-fold paired differences.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E81.1-G1-Visual-PRCL-v2 | 92.2153 ± 3.1082 | 92.0455 ± 3.0166 | 93.5664 ± 3.2267 | 90.0474 ± 3.3497 | 94.0436 ± 5.4024 | 91.5010 ± 3.1365 |
| E81.1-G2-Medical-Add-PRCL-v2 | 91.8321 ± 2.8459 | 91.5311 ± 2.8857 | 93.3373 ± 3.2317 | 87.5745 ± 3.8679 | 95.4877 ± 2.7672 | 90.8169 ± 3.1876 |
| E81.1-G3-Medical-Interaction-PRCL-v2 | 91.5781 ± 3.0234 | 91.3777 ± 2.9825 | 93.3832 ± 3.1396 | 88.9431 ± 4.1172 | 93.8123 ± 4.9490 | 90.7313 ± 3.1588 |
| E81.1-G4-Gram-Matched-Control-PRCL-v2 | 91.4490 ± 3.2179 | 91.2094 ± 3.2024 | 93.6468 ± 3.0218 | 88.1233 ± 4.1977 | 94.2954 ± 4.4225 | 90.5108 ± 3.4633 |
| E74.4-G2 DARG | 92.2153 ± 2.9131 | 92.0481 ± 2.8723 | 93.6142 ± 3.0541 | 90.0474 ± 4.0959 | 94.0487 ± 4.9847 | 91.4703 ± 3.0580 |
| E81-G1-Visual-PRCL | 91.9598 ± 2.8742 | 91.7763 ± 2.8711 | 93.5483 ± 3.0640 | 89.4986 ± 4.3825 | 94.0539 ± 4.5335 | 91.1468 ± 3.0631 |
| E81-G2-Medical-PRCL | 91.9598 ± 2.9248 | 91.7261 ± 2.9235 | 93.6428 ± 2.9730 | 88.6789 ± 3.9027 | 94.7734 ± 3.8053 | 91.0704 ± 3.1892 |
| E81-G3-Anonymous-PRCL | 91.9598 ± 3.0717 | 91.6847 ± 3.1271 | 93.6023 ± 3.1916 | 88.1233 ± 4.6677 | 95.2462 ± 3.4867 | 90.9917 ± 3.4519 |

## Paired comparisons

Differences are left minus right in percentage points. W/T/L and t-based 95% intervals are descriptive across folds; the folds have overlapping training subjects.

| Comparison | Scope | ACC Δ (W/T/L) | Hard-AUC Δ | Prob-AUC Δ | SEN Δ | SPE Δ | ASD-F1 Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| G1_vs_DARG | all10 | -0.1149 (3/3/4) | -0.1243 | -0.0325 | -0.2439 | -0.0046 | -0.1081 |
| G1_vs_DARG | without_fold2 | +0.0000 (3/3/3) | -0.0026 | -0.0479 | +0.0000 | -0.0051 | +0.0307 |
| G2_vs_E81_G2 | all10 | -0.0000 (1/8/1) | -0.0535 | -0.1848 | -0.7500 | +0.6429 | -0.0926 |
| G2_vs_E81_G2 | without_fold2 | -0.1277 (0/8/1) | -0.1950 | -0.3055 | -1.1043 | +0.7144 | -0.2535 |
| G3_vs_G2 | all10 | -0.2286 (1/6/3) | -0.1513 | +0.0943 | +0.9878 | -1.2905 | -0.1014 |
| G3_vs_G2 | without_fold2 | -0.2540 (1/5/3) | -0.1534 | +0.0459 | +1.3686 | -1.6754 | -0.0856 |
| G3_vs_G4 | all10 | +0.1162 (2/7/1) | +0.1515 | -0.2426 | +0.7378 | -0.4348 | +0.1985 |
| G3_vs_G4 | without_fold2 | +0.1292 (2/6/1) | +0.1683 | -0.2636 | +0.8198 | -0.4831 | +0.2205 |
| G3_vs_G1 | all10 | -0.5734 (0/6/4) | -0.6010 | -0.0906 | -0.9939 | -0.2081 | -0.6927 |
| G3_vs_G1 | without_fold2 | -0.6371 (0/5/4) | -0.6678 | -0.1832 | -1.1043 | -0.2313 | -0.7697 |
| G3_vs_DARG | all10 | -0.6883 (0/4/6) | -0.7253 | -0.1231 | -1.2378 | -0.2128 | -0.8008 |
| G3_vs_DARG | without_fold2 | -0.6371 (0/4/5) | -0.6704 | -0.2310 | -1.1043 | -0.2364 | -0.7389 |

## Mechanism

Across frozen held-out fold patients: mean relation entropy 2.1376, Null weight 0.0253, Null-dominant rate 0.0066, same-prompt patient similarity 0.5513, between-prompt similarity 0.9919.
Within-slot candidate attention mass 3.3758 (candidate base rate 0.20; enrichment 1.0825); cross-slot mass 12.2188 (base rate 0.80; enrichment 0.9794).
Alpha mean 0.04353; residual ratio mean/P50/P95 0.05185/0.05374/0.06692; fraction near 10% cap 0.00000; mean unbounded-to-bound ratio 0.65199.
Nonlinear interaction mean norm 0.13770; classification gradient norms for alpha gate and interaction input 0.0032462 and 2.92252e-05.

| Group | Null weight | Null-dominant | Prompt similarity | Alpha | Residual mean / P95 | Near 10% cap | Interaction grad |
|---|---:|---:|---:|---:|---:|---:|---:|
| E81.1-G1-Visual-PRCL-v2 | 0.0268 | 0.0046 | 0.9917 | 0.04112 | 0.05315 / 0.06776 | 0.00000 | 2.8e-05 |
| E81.1-G2-Medical-Add-PRCL-v2 | 0.0199 | 0.0017 | 0.9907 | 0.04478 | 0.05117 / 0.06634 | 0.00000 | 2.94e-05 |
| E81.1-G3-Medical-Interaction-PRCL-v2 | 0.0312 | 0.0128 | 0.9931 | 0.04645 | 0.05494 / 0.07094 | 0.00000 | 3.08e-05 |
| E81.1-G4-Gram-Matched-Control-PRCL-v2 | 0.0234 | 0.0073 | 0.9919 | 0.04179 | 0.04813 / 0.06265 | 0.00000 | 2.87e-05 |

Frozen G3 interventions are post-training probes. Probability MAE is relative to the intact G3 Checkpoint; the number in parentheses is total hard-prediction flips across folds.

| G3 intervention | Probability MAE (flips) | Mean ACC delta (pp) | Mean Prob-AUC delta (pp) |
|---|---:|---:|---:|
| gamma_zero | 0.000019 (0) | +0.0000 | +0.0000 |
| medical_to_same_gram_control | 0.000029 (0) | +0.0000 | +0.0000 |
| multiplicative_to_additive_query | 0.000100 (0) | +0.0000 | +0.0053 |
| nonlinear_interaction_off | 0.000010 (0) | +0.0000 | +0.0000 |
| normal_abnormal_swap | 0.000031 (0) | +0.0000 | -0.0053 |
| null_only | 0.005300 (0) | +0.0000 | +0.0159 |
| prcl_off | 0.005300 (0) | +0.0000 | +0.0159 |
| text_off | 0.000019 (0) | +0.0000 | +0.0000 |
| uniform_relations | 0.001071 (0) | +0.0000 | +0.0053 |
| writeback_off | 0.005300 (0) | +0.0000 | +0.0159 |

The mechanism CSV includes per-fold frozen interventions for PRCL-off, uniform relation attention, Null-only, text-off, gamma-zero, normal/abnormal group swap, nonlinear-interaction-off, medical-to-same-Gram-control replacement, and multiplicative-to-additive Query replacement in G3. These interventions are not independently retrained ablations.

## Audit and scientific gates

- RNG triplet mismatches against E74.4 common epochs: 0; between E81.1 groups: 0.
- Base-off parity: 10/10; two-step restore replay: PASS.
- Prompt cache SHA: b22e50ed6191c3c8e0f3a98b0c85c33ccc30267b8aed9bbcf05f794ab0c8d6fd; G4 orthogonal Gram max error: 4.441e-16.
- Gate A local relation repair vs DARG: NOT_SUPPORTED.
- Gate B shared repair vs E81 medical-add: NO_ACC_IMPROVEMENT.
- Gate C multiplicative vs additive prompt Query: NOT_ESTABLISHED.
- Gate D medical anchors vs same-Gram rotated control: NOT_ESTABLISHED.
- Gate E complete method vs DARG: NOT_SUPPORTED.
- Final research decision: NO_GO_PRCL.

Limitations retained from the historical comparison protocol: validation equals test and test ACC selects checkpoints; preprocessing/RFE occurs before CV; the patient graph is transductive; Fold 2 rows 377 and 407 occur in both historical train and test sets and remain in the original DARG path, while they are excluded from the new residual penalty. Ten-fold intervals are descriptive because training sets overlap. The 16 local CAM tokens are latent coordinates, not anatomical ROIs.

## Artifacts

Source, configuration, and cache identities are frozen in manifest.json. Per-fold results are in fold_metrics.csv, paired differences in paired_comparisons.csv, and frozen mechanism checks in mechanism.csv.

Audit execution record: 5 attempts, final attempt PASS; details are in `audit_attempts.json`.

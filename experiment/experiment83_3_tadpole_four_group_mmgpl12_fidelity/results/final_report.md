# Experiment 83.3 formal report

The locked TADPOLE protocol uses 598 patients, six modalities, three classes, seed 0, and ten fixed outer folds.
ACC and AUC_original are recomputed from fold-out hard predictions. AUC_probability is a supplementary macro OVR ROC-AUC from the stored probabilities.

| Group | Source | ACC | AUC_original | AUC_probability |
|---|---|---:|---:|---:|
| E28-E0 Original SPromptGL | REUSED_VERIFIED_FORMAL from E83.1 | 87.13 ± 5.11% | 90.16 ± 3.35% | 94.53 ± 1.92% |
| E43.1-E00 No-PEN Historical Baseline | REUSED_VERIFIED_FORMAL from E83.1 | 88.78 ± 3.96% | 90.73 ± 2.88% | 95.40 ± 2.23% |
| E74.4-G2 DARG Positive-DCR | REUSED_VERIFIED_FORMAL from E83.1 | 92.47 ± 3.19% | 93.98 ± 2.74% | 95.55 ± 3.01% |
| E82.4.6.10-G1 DARG+DPSL+MMGPL12 | NEW_FORMAL_TRAINING in E83.3 | 92.48 ± 3.00% | 93.42 ± 2.64% | 94.56 ± 2.89% |

## Paper comparison

SPromptGL Table 2 reports ACC 94.81 ± 2.95% and AUC 95.59 ± 2.57%.
The reported AUC comparison uses AUC_original. AUC_probability is retained as a diagnostic.

| Contrast | Mean ΔACC | Mean ΔAUC_original | Improved / degraded / unchanged folds | Net correct patients |
|---|---:|---:|---:|---:|
| G2-G1 | +1.65 pp | +0.58 pp | 4 / 4 / 2 | +10 |
| G3-G2 | +3.69 pp | +3.25 pp | 9 / 0 / 1 | +22 |
| G4-G3 | +0.00 pp | -0.56 pp | 3 / 2 / 5 | +0 |
| G4-G1 | +5.35 pp | +3.27 pp | 9 / 1 / 0 | +32 |

## E83.1 comparison

G1, G2, and G3 inherit the E83.1 formal run after independent replay. E83.1 G4 is retained as a historical eight-prompt reference and was not modified into the MMGPL-12 run.

| Group | E83.1 ACC | E83.3 ACC | ΔACC | E83.1 AUC_original | E83.3 AUC_original | ΔAUC_original |
|---|---:|---:|---:|---:|---:|---:|
| E28-E0 Original SPromptGL | 87.13% | 87.13% | +0.00 pp | 90.16% | 90.16% | +0.00 pp |
| E43.1-E00 No-PEN Historical Baseline | 88.78% | 88.78% | +0.00 pp | 90.73% | 90.73% | +0.00 pp |
| E74.4-G2 DARG Positive-DCR | 92.47% | 92.47% | +0.00 pp | 93.98% | 93.98% | +0.00 pp |
| E82.4.6.10-G1 DARG+DPSL+MMGPL12 | 92.31% | 92.48% | +0.17 pp | 93.29% | 93.42% | +0.13 pp |

## Method and audit status

G1 preserves Original SPromptGL with the eight historical prompts and physical PEN. G2 physically removes PEN. G3 preserves the signed DARG graph and Positive-DCR. G4 uses the same DARG warmup state per fold and the SCMR-DA-SRPE core with twelve MMGPL queries.
The MMGPL MCI group is a semantic proxy for MMGPL's LMCI descriptions. It does not redefine TADPOLE labels.
The twelve-query SCMR bridge maps frozen RN50 text vectors through the historical P_text projection. The historical eight-query binary branch passed exact tensor, objective, and EMA regression before the three-class extension.
DARG and DPSL audits report mechanism presence, intervention output changes, and patient-level correction or damage counts. A positive floating-point difference alone is not treated as a classification gain.
G3−G2 includes its historical GFR, relation learning, warmup, optimizer, and weight-decay differences. G4−G3 measures the combined DPSL and MMGPL-12 adaptation. Independent prompt-only causality is not identifiable from these four groups.
The frozen historical protocol performs RFE, mean imputation, and StandardScaler before outer CV and uses the test fold for Best selection through val_idx == test_idx. These limitations apply to every group.

## Mechanism interventions

G3 replay observed 56855 positive and 14145 negative signed edges across folds. Direct-Off changed 0 test predictions and Disease-Neutral changed 0 test predictions across the inherited folds, while log-probability differences are retained in `audits/checkpoint_replay.json`.
G4 replay observed 52559 positive and 15384 negative signed edges. Direct-Off changed 0 test predictions. DPSL Full versus Residual-Off corrected 2 patients, damaged 0, and changed net correct count by 2. The routed Top-2 counts were NC-MCI 4019, NC-AD 75, and MCI-AD 1886.
Residual-Off exactly restored the native output in every G4 replay. Nonzero residuals and log-probability changes therefore establish participation, while the patient-level counts establish the observed classification effect.

Combined checkpoint replay status: `PASS` for 40 of 40 folds.

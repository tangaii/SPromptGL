# Experiment 83.4 Final Report

Status `COMPLETE_AUDITED`. Method fidelity `GO_METHOD_FIDELITY`. Performance `NO_GO_PERFORMANCE`. Prompt causality `PROMPT_CAUSALITY_INCONCLUSIVE`.

TADPOLE contains 598 patients with NC/MCI/AD counts 209/315/74, six modalities 145/148/3/36/24/9, and 365 features. The fixed ten-fold split, seed 0, historical full-data RFE, imputation and pre-CV scaling, transductive patient graph, and test-fold Best selection are retained.

## Formal Results

Means and population standard deviations are calculated across ten folds from stored fold-out predictions. AUC_original uses the historical one-hot hard-prediction macro OVR definition. AUC_probability is a probability-ranking diagnostic.

| Group | Formal source | ACC | AUC_original | AUC_probability | Correct / 598 |
|---|---|---:|---:|---:|---:|
| E28-E0 Original SPromptGL | REUSED_VERIFIED_FORMAL | 87.13 ± 5.11% | 90.16 ± 3.35% | 94.53 ± 1.92% | 521 / 598 |
| E43.1-E00 No-PEN | REUSED_VERIFIED_FORMAL | 88.78 ± 3.96% | 90.73 ± 2.88% | 95.40 ± 2.23% | 531 / 598 |
| E74.4-G2 DARG Positive-DCR | REUSED_VERIFIED_FORMAL | 92.47 ± 3.19% | 93.98 ± 2.74% | 95.55 ± 3.01% | 553 / 598 |
| E82.4.6.10-G1 DARG+DPSL+MMGPL12 all-pair | NEW_FORMAL_TRAINING | 91.64 ± 3.68% | 93.18 ± 3.34% | 96.15 ± 2.03% | 548 / 598 |

## Comparisons

G4 Full minus G3 is -0.84 percentage points ACC and -0.80 points AUC_original, with -5 net correct test-fold predictions.
G4 Native was correct on 547 of 598 test patients. G4 Full was correct on 548. The direct DPSL inference correction changed 1 errors to correct and 0 correct predictions to errors, for a net +1.
Compared with G3, the G4 Native path had 11 newly correct and 17 newly incorrect patients, a net -6. This isolates the jointly trained native backbone difference from the direct Full-minus-Native inference effect.

E83.3 historical G4 recomputed from its archived predictions was ACC 92.48 ± 3.00% and AUC_original 93.42 ± 2.64%. E83.4 G4 changed by -0.84 ACC points and -0.25 AUC_original points.
The paper reports ACC 94.81 ± 2.95% and AUC 95.59 ± 2.57%. E83.4 G4 is -3.17 ACC points and -2.41 AUC_original points from those references. The paper's ablation rows are not one-to-one equivalents of E83.4 groups.

## Class and AD Diagnostics

G3 classified AD correctly for 65/74 patients. G4 Native and Full each classified 64/74 correctly, while G4 Full AD probability OVR AUC was 0.9578. G4 Full class correct counts were NC 201/209, MCI 283/315, and AD 64/74.
DPSL pair evidence includes NC-AD and MCI-AD, and every test row receives all three pair candidates. The count of AD true-class test rows with a structurally excluded pair is zero. G4 NC-AD and MCI-AD conflict counts and directed native margins are recorded per fold. AD patients corrected from native AD to Full MCI total 0.

## Routing and Mechanisms

Across 5980 transductive node decisions, maximum-weight route counts were NC-MCI 3990, NC-AD 2, and MCI-AD 1988. Their mean weights were 0.4258, 0.2538, and 0.3204. Across 598 held-out test-patient decisions, maximum-weight counts were 390, 0, and 208. These are node/fold decisions, not independent patient counts across folds. All three residual candidates were computed and contributed to the weighted aggregate. Per-class and native-correctness weight distributions are in the mechanism audit.
G3 DARG had 56855 positive and 14145 negative signed edges. Direct-Off changed 0 test argmax predictions, with maximum log-probability change 1.67e-06. Disease-Neutral changed 0 predictions, with maximum GCN message difference 0.000552.
G4 DARG had 53917 positive and 15107 negative signed edges. Direct-Off changed 0 test argmax predictions, with maximum log-probability change 1.67e-06. Disease-Neutral changed 0 predictions, with maximum GCN message difference 0.000732.
DPSL residual-off restored native logits exactly in all ten G4 folds. The maximum CAM residual ratio was 0.07999586, within the 0.08 limit. The maximum observed shared-gradient increment fraction was 0.000999888, below the 0.001 budget.

## Fidelity and Limits

The original SPromptGL TADPOLE GFR, CAM, RES, eight-prompt PEN, loss, and optimizer behaviors were retained. Historical second-modality GFR overwrite and ALoss observation-only behavior were verified against the archive. No E83.3 G1/G2/G3 computation defect was established. G1-G3 were reused after independent replay, and 129/129 selected E83.3 artifacts were byte-identical to their E83.4 copies. E83.3 G4 Best and Latest replays each passed 10/10, but those checkpoints were used only as a historical audit and prediction comparison, never to initialize E83.4. G4 used a shared DPSL parameter set across all pairs and a deterministic per-fold G3 warmup replay. The binary 8-query SCMR regression and all three-class math checks passed before training.

MMGPL prompt vectors and projected queries passed finiteness and rank checks. Class query centroids remain highly similar, especially MCI and AD. The historical 8-pair Procrustes alignment has rank at most eight in 32 dimensions. Full inference uses q_medical. The anonymous query remains diagnostic, and no prompt variant was selected by test performance.
The q_medical class-centroid cosine similarities were NC-MCI 0.9700, NC-AD 0.9735, and MCI-AD 0.9983. These semantic similarities are retained as a limitation, not addressed through test-driven prompt edits.

The fixed historical protocol includes full-data feature selection and scaling before outer CV, transductive patient graphs, and Best selection on the test fold. Results are not an unbiased estimate of generalization and do not establish full reproduction of the paper. Prompt-only causality remains inconclusive because no registered prompt-only group exists.
The formal-start preflight passed before G4 training, and the ten G4 checkpoints share the current training-module fingerprint `22966e7bcac168afcd48693c65dca78f65d3264c269ca94b5536ab2289d088d2`. Post-training changes were limited to replay and reporting audits. The original start-time preflight file was replaced by a refreshed hash-bound audit, so its exact SHA-256 is not retained.

Best replay `PASS` for 40/40 folds. Latest restore and deterministic next-step replay `PASS` for 40/40 folds. Per-fold G3/G4 warmup parity passed for 10/10 folds. G1-G3 account for 30 verified inherited folds and G4 for ten newly trained folds.

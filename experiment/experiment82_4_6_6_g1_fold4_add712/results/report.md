# Experiment 82.4.6.6-G1

## Status

COMPLETE_AUDITED. Fold 4 was the only retrained fold. The experiment is an intentional training-test label-overlap diagnostic.

## Frozen protocol and membership

- Base state was E74.4-G2 through the verified E61 epoch 10 state. Training resumed at epoch 11 with seed 0 and no per-epoch reseeding.
- Fold 4 DARG native training rows were 784 before overlap expansion, 787 in E82.4.6.5-G1, and 788 after adding patient 712. SCMR legal supervision rows were 784. Gaussian and EMA legal rows were 784. Test rows were 87.
- Patients 712, 713, 727, and 773 were retained in the test set and trained by DARG classification and relation objectives. Their new spectral and prompt residuals were exactly zero and they were excluded from SCMR, Gaussian, and EMA legal membership.
- Read-only reused Fold 2 had 792 DARG rows and 784 SCMR legal rows. Read-only reused Fold 3 had 789 DARG rows and 784 SCMR legal rows. The remaining reused folds retain their E82.4.6.5-G1 memberships.
- Patient labels were independently checked from the original label array. Patients 712, 713, 727, and 773 were ASD with encoding 0. The 1-based to 0-based conversion was checked for every patient row.
- Relation training used 4096 pairs with 2048 same-class and 2048 different-class pairs. The pair sampler did not advance the historical global RNG.

## Ten-fold metrics

Values are mean ± population SD in percent. Hard-AUC is BA from hard predictions. Prob-AUC uses continuous ASD probability.

| Method | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 DARG | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 | 93.4946 ± 2.9195 | 90.0671 ± 3.8862 | 93.9917 ± 4.7320 | 91.4590 ± 2.9013 |
| E82.4.6-G2 | 92.1891 ± 2.6675 | 92.0503 ± 2.6446 | 93.6688 ± 2.9012 | 90.3171 ± 3.4469 | 93.7835 ± 4.0472 | 91.4729 ± 2.8570 |
| E82.4.6.5-G1 | 92.9937 ± 2.7941 | 92.8540 ± 2.8212 | 94.7937 ± 2.8263 | 91.0549 ± 3.7565 | 94.6531 ± 3.3634 | 92.3136 ± 3.1081 |
| E82.4.6.6-G1 | 92.9937 ± 2.7941 | 92.8912 ± 2.8540 | 94.8575 ± 2.9025 | 91.5549 ± 4.2198 | 94.2276 ± 3.1941 | 92.3386 ± 3.1295 |

## Fold 4 and paired analysis

- E82.4.6.5-G1 to E82.4.6.6-G1 on all 871 patients. Corrections 3. Harms 3. Net 0. ΔACC 0.0000 pp. ΔBA 0.0345 pp. ΔProb-AUC -0.0366 pp.
- Fold 4. Corrections 4:712;4:713;4:727. Harms 4:301;4:517;4:806. Net 0. ΔACC 0.0000 pp. ΔBA 0.3723 pp. ΔProb-AUC 0.6383 pp.
- Fold 4 patient 712 changed from hard prediction 1 to the correct prediction. Its previous residual norm was 0.135066, while the new spectral-in, spectral-out, and joint residual norms were exactly zero. Its CAM144 feature and classification probability were identical in Residual-off and Full.
- Patients 713 and 727 changed to correct predictions. Patient 773 remained correct. The other 83 Fold-4 test patients had corrections 0, harms 3, and net -3.
- The fixed union of 17 current or previous overlap patients had corrections 3, harms 0, and net 3. The explicitly constructed outside-union set had 854 patients with corrections 0, harms 3, and net -3.
- The 854-patient set is a fixed descriptive paired-analysis cohort and is not an independent validation set.

### Fold results

| Fold | Result | Corrections | Harms | Net |
|---:|---|---:|---:|---:|
| 1 | tie | 0 | 0 | 0 |
| 2 | tie | 0 | 0 | 0 |
| 3 | tie | 0 | 0 | 0 |
| 4 | tie | 3 | 3 | 0 |
| 5 | tie | 0 | 0 | 0 |
| 6 | tie | 0 | 0 | 0 |
| 7 | tie | 0 | 0 | 0 |
| 8 | tie | 0 | 0 | 0 |
| 9 | tie | 0 | 0 | 0 |
| 10 | tie | 0 | 0 | 0 |

## Fixed-weight mechanism audit

- Residual-off to Full across ten folds. Corrections 2. Harms 0. Net 2. ΔACC 0.2296 pp. ΔBA 0.2309 pp. Mean true-margin change 0.0594. Fold 4 had no hard-prediction change.
- Anonymous Query to Medical Query across ten folds. CAM144 changed for 854 patients. ASD probability changed for 854 patients. Hard prediction flipped for 1 patients. Corrections 1. Harms 0. Net 1. ΔACC 0.1148 pp. ΔBA 0.1068 pp. Mean true-margin change 0.0087. Fold 4 CAM144 changes 83, probability changes 83, and hard flips 0.
- The four Fold-4 overlap patients had exact zero spectral-in, spectral-out, and joint residual. The other 83 test patients retained an active residual path.
- The maximum Fold-4 relative joint CAM residual was 0.013334, below the 0.08 limit. The shared differential-gradient audit passed the 0.001 limit.
- Jacobian spectral-in, Jacobian spectral-out, medical Prompt, CAM144, end-to-end classification, DARG classification, DARG relation, Gaussian, and EMA checks passed.

## Checkpoint and RNG audit

- Best epoch 182 and Latest epoch 383 were reloaded. Best SHA 51c92445fd8c97905eb8d7d05ba48f6a47b7a5566302fb81de0285acb4eea1ff. Latest SHA 55a7a4bc179983ce96f2c0f8c582b702a1ea4047253ef12bc69998bf59205463. Journal SHA eabd17ac5c712dc89ae8055a8ab4d4bc60e6face641bc9202690518c75e2951a.
- Latest next-step training and evaluation recovery were exact. RNG was replayed epoch by epoch. Historical six-dropout order was preserved. The E82.4.6.5 source manifest and every reused result and checkpoint identity matched its frozen SHA.
- The historical E82.4.6.5 Fold-4 Best was epoch 267 with 83/87. It was used only for comparison. The new run started from E61 epoch 10 and selected epoch 182 independently.
- The checkpoint payload retains the historical canonical parent field G2 used by the E82.4.6.5 G1 loader. The task files, manifest, and experiment identity are G1.
- A preflight audit-entry repair registered the initial EMA summary key. A separate audit-entry repair registered list-based masking for tuple-index handling. Neither changed the frozen training implementation or any checkpoint.

## Scientific interpretation

The Fold-4 change corrected patients 712, 713, and 727 while introducing three other Fold-4 harms, leaving ACC unchanged. The effect is attributable to the intentional training-test label exposure and cannot be treated as independent generalization. This result does not by itself justify a GO decision for the MICCAI 2027 second innovation module.

Authoritative directory. `/model/spromptgl/experiment/experiment82_4_6_6_g1_fold4_add712`

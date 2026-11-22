# Experiment 82.4.6.7-G1

## A. STATUS

COMPLETE_AUDITED. Only Fold 2 was retrained. This is an intentional training-test label-overlap diagnostic.

## B. Ten-fold performance

Values are mean ± population SD in percent. Hard-AUC is BA from hard predictions. Prob-AUC uses continuous ASD probability.

| Method | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 DARG | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 | 93.4946 ± 2.9195 | 90.0671 ± 3.8862 | 93.9917 ± 4.7320 | 91.4590 ± 2.9013 |
| E82.4.6-G2 | 92.1891 ± 2.6675 | 92.0503 ± 2.6446 | 93.6688 ± 2.9012 | 90.3171 ± 3.4469 | 93.7835 ± 4.0472 | 91.4729 ± 2.8570 |
| E82.4.6.4-G1 | 93.2236 ± 2.6480 | 93.0683 ± 2.7049 | 94.9308 ± 2.6814 | 91.0488 ± 3.9247 | 95.0879 ± 2.8655 | 92.5271 ± 3.0068 |
| E82.4.6.5-G1 | 92.9937 ± 2.7941 | 92.8540 ± 2.8212 | 94.7937 ± 2.8263 | 91.0549 ± 3.7565 | 94.6531 ± 3.3634 | 92.3136 ± 3.1081 |
| E82.4.6.6-G1 | 92.9937 ± 2.7941 | 92.8912 ± 2.8540 | 94.8575 ± 2.9025 | 91.5549 ± 4.2198 | 94.2276 ± 3.1941 | 92.3386 ± 3.1295 |
| E82.4.6.7-G1 | 92.9937 ± 2.7941 | 92.8912 ± 2.8540 | 94.9211 ± 2.9695 | 91.5549 ± 4.2198 | 94.2276 ± 3.1941 | 92.3386 ± 3.1295 |

## C. Fold 2 paired comparison

E82.4.6.6-G1 to E82.4.6.7-G1. Corrections 1 (2:540), harms 1 (2:18), net 0. ΔACC 0.0000 pp, ΔBA 0.0000 pp, ΔProb-AUC 0.0700 pp.
Ten-fold paired ΔSEN-ASD 0.0000 pp, ΔSPE-TD 0.0000 pp, and ΔASD-F1 0.0000 pp.
Fold 2 corrections 2:540, harms 2:18, net 0. ΔACC 0.0000 pp, ΔBA 0.0000 pp, ΔProb-AUC 0.6363 pp.

## D. Patient 540

True label ASD=0. E82.4.6.6-G1 prediction 1 with ASD probability 0.10522760 and true-class margin -2.140444. E82.4.6.7-G1 Residual-off prediction 0 with ASD probability 0.85563940. Full prediction 0 with ASD probability 0.85563940 and margin 1.779534. ASD probability change versus the direct reference was 0.75041179. Corrected versus the reference is True. Patient 540 participated in DARG classification and relation training and was excluded from new SCMR label supervision, Gaussian, EMA, and spectral/Prompt residual updates. Its new spectral-in, spectral-out, and joint residual norms were 0, 0, and 0.

## E. Other Fold 2 patients
The eight retained overlap patients had corrections 0 (none), harms 0 (none), net 0. The other 78 test patients had corrections 0 (none), harms 1 (2:18), net -1.

## F. Residual-off, spectral/Prompt, and medical Query mechanisms
Residual-off to Full across ten folds. Corrections 2, harms 0, net 2, ΔACC 0.2296 pp, ΔBA 0.2309 pp, mean margin change 0.0734.
Anonymous Query to Medical Query across ten folds. CAM144 changed for 853 patients, ASD probability changed for 853, and hard prediction flipped for 1. Corrections 1, harms 0, net 1. Fold 2 CAM144 changes 78, probability changes 78, hard flips 0.
All nine Fold 2 overlap patients had exact zero spectral-in, spectral-out, and joint residual, while the remaining 78 Fold 2 test patients retained an active residual path. The maximum relative joint CAM residual was 0.077723. The 8% residual limit and 0.1% shared-gradient budget passed.

## G. Checkpoint, RNG, gradient, masking, and reuse audit
Best epoch 251, Latest epoch 450. Best SHA 956cb5309aefe5f864e4831e6c29b1da46dfc253d91c5d4515af600b1399e0d3. Latest SHA 4b3dcde14ff0d9b996df7ba80611a26ffdcda1f2b0f2e6f039e5fbb3bb8d6f0e. Journal SHA 74f61cb79380cf9609e60ac65c1510cbca89208fc08b7f88e4a26fc9dd5b1e01.
Latest next-step training and evaluation recovery were exact. RNG was replayed epoch by epoch, the six-dropout order was preserved, DARG classification and relation gradients were present, SCMR residual gradients were zero on masked rows and nonzero on legal rows, and Gaussian/EMA legal membership remained 784.
The nine reused folds were read-only E82.4.6.6-G1. Their patient-level probabilities, hard predictions, margins, and checkpoint identities were verified exactly. The E82.4.6.6-G1 Fold 4 result was not used as a starting weight.

## H. Files and scientific conclusion
Authoritative directory `/model/spromptgl/experiment/experiment82_4_6_7_g1_fold2_add540`. Required result files, Manifest, runtime replay, Fold 2 Best/Latest, and handoff are recorded with SHA values. The fixed current-or-reference exposure union contains 18 patient-fold entries and the explicitly constructed outside set contains 853 patients. The Fold 2 unexposed analysis contains 78 patients.
The observed changes are attributable to intentional training-test label exposure under the historical val=test, single-seed, and transductive-graph protocol. They are not independent-generalization evidence and do not by themselves justify a GO decision for the MICCAI 2027 second innovation module.

### Per-fold outcome

| Fold | Result | Corrections | Harms | Net |
|---:|---|---:|---:|---:|
| 1 | tie | 0 | 0 | 0 |
| 2 | tie | 1 | 1 | 0 |
| 3 | tie | 0 | 0 | 0 |
| 4 | tie | 0 | 0 | 0 |
| 5 | tie | 0 | 0 | 0 |
| 6 | tie | 0 | 0 | 0 |
| 7 | tie | 0 | 0 | 0 |
| 8 | tie | 0 | 0 | 0 |
| 9 | tie | 0 | 0 | 0 |
| 10 | tie | 0 | 0 | 0 |

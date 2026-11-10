# Experiment 76.3 — Nonlinear Prototype Incremental Utility Probe

Status: PASS after independent deterministic audit; conclusion: **INCONCLUSIVE**.

## Design and provenance

E74.4-G2 best checkpoints were frozen. Each outer fold had one CAM144 extraction forward; OutputLayer ran once and no model, CAM, classifier, optimizer, or checkpoint state was updated. B4 uses the exact E76.2-B1 reliable-only 12-slot evidence features; B5 uses E76.2-B2 reliable+hard-pair 12-slot evidence. The only readout change is the classifier: both are StandardScaler + RBF-SVC (C=1, gamma=scale, probability=False).

The historical B0-NL OOF patient IDs, predictions, scores, and fold metrics are reused from E76.2's audited cache and independently re-scored; B0-NL is not refit. ASD=0, TD=1; `classes_=[0,1]` is checked and the ROC score is `-decision_function`, so larger values indicate ASD. It is a ranking score, not a calibrated probability; log-loss/Brier are not calculated for SVC.

Five outer-training inner folds use the E76.2 fixed split. Training-side prototype evidence is cross-fitted over three disjoint support/query folds using seed 76300+fold; each validation patient is excluded from its own prototype bank. StandardScaler fits only the inner training support. Fold 2 rows 377 and 407 are excluded from added support.

E76.2's historical CAM encoder was trained using outer-training subjects, and its base checkpoint was selected with val=test. Therefore these are frozen-encoder inner-OOF readout diagnostics, not independent end-to-end generalization estimates. Pre-CV preprocessing, transductive graph construction, and Fold 2's historical overlap remain. Adding evidence also changes standardized feature distribution and the effective `gamma=scale`, so any difference is a feature-configuration result, not proof of novel disease information.

## Results

Mean ± population SD across ten outer folds, percent. Hard-AUC is balanced accuracy from hard predictions; Prob-AUC uses the continuous SVC decision score.

| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| B0-NL CAM-only RBF-SVM | 97.2571±0.5750 | 97.2156±0.5566 | 99.1698±0.2213 | 96.6634±0.7365 | 97.7678±1.0150 | 97.0268±0.6128 |
| B4 CAM+Reliable RBF-SVM | 97.2954±0.5594 | 97.2512±0.5399 | 99.1782±0.2175 | 96.6634±0.6605 | 97.8391±0.9767 | 97.0671±0.5965 |
| B5 CAM+Reliable/Hard Paired RBF-SVM | 97.3975±0.5972 | 97.3521±0.5796 | 99.1903±0.2114 | 96.7463±0.5779 | 97.9579±0.9216 | 97.1771±0.6388 |

## Paired contrasts

Delta is left method minus right method in percentage points. Confidence intervals are descriptive paired t intervals across outer folds; n=10 and the nine-fold sensitivity excludes Fold 2.

| Comparison | Metric | Δ pp | 95% CI | W/T/L | Δ pp without Fold 2 |
|---|---|---:|---:|---:|---:|
| B4 CAM+Reliable RBF-SVM − B0-NL CAM-only RBF-SVM | ACC | +0.038 | [-0.048, +0.125] | 2/7/1 | +0.043 |
| B4 CAM+Reliable RBF-SVM − B0-NL CAM-only RBF-SVM | Hard-AUC | +0.036 | [-0.050, +0.121] | 2/7/1 | +0.040 |
| B4 CAM+Reliable RBF-SVM − B0-NL CAM-only RBF-SVM | Prob-AUC | +0.008 | [-0.002, +0.019] | 6/1/3 | +0.009 |
| B4 CAM+Reliable RBF-SVM − B0-NL CAM-only RBF-SVM | SEN | +0.000 | [-0.093, +0.093] | 1/8/1 | +0.000 |
| B4 CAM+Reliable RBF-SVM − B0-NL CAM-only RBF-SVM | SPE | +0.071 | [-0.043, +0.186] | 2/8/0 | +0.079 |
| B4 CAM+Reliable RBF-SVM − B0-NL CAM-only RBF-SVM | ASD-F1 | +0.040 | [-0.052, +0.133] | 2/7/1 | +0.045 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B0-NL CAM-only RBF-SVM | ACC | +0.140 | [-0.005, +0.286] | 5/4/1 | +0.156 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B0-NL CAM-only RBF-SVM | Hard-AUC | +0.136 | [-0.016, +0.289] | 5/3/2 | +0.152 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B0-NL CAM-only RBF-SVM | Prob-AUC | +0.020 | [-0.014, +0.055] | 7/0/3 | +0.024 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B0-NL CAM-only RBF-SVM | SEN | +0.083 | [-0.253, +0.419] | 3/4/3 | +0.092 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B0-NL CAM-only RBF-SVM | SPE | +0.190 | [-0.034, +0.414] | 4/6/0 | +0.211 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B0-NL CAM-only RBF-SVM | ASD-F1 | +0.150 | [-0.011, +0.311] | 5/3/2 | +0.167 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B4 CAM+Reliable RBF-SVM | ACC | +0.102 | [-0.018, +0.222] | 4/6/0 | +0.113 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B4 CAM+Reliable RBF-SVM | Hard-AUC | +0.101 | [-0.032, +0.233] | 4/4/2 | +0.112 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B4 CAM+Reliable RBF-SVM | Prob-AUC | +0.012 | [-0.014, +0.038] | 6/1/3 | +0.015 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B4 CAM+Reliable RBF-SVM | SEN | +0.083 | [-0.240, +0.406] | 2/6/2 | +0.092 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B4 CAM+Reliable RBF-SVM | SPE | +0.119 | [-0.001, +0.239] | 4/6/0 | +0.132 |
| B5 CAM+Reliable/Hard Paired RBF-SVM − B4 CAM+Reliable RBF-SVM | ASD-F1 | +0.110 | [-0.026, +0.246] | 4/4/2 | +0.122 |

## Fold 2 sensitivity

Nine-fold mean ± population SD (%):

| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| B0-NL CAM-only RBF-SVM | 97.1791±0.5537 | 97.1431±0.5400 | 99.1456±0.2203 | 96.6610±0.7763 | 97.6251±0.9700 | 96.9447±0.5915 |
| B4 CAM+Reliable RBF-SVM | 97.2217±0.5416 | 97.1827±0.5261 | 99.1549±0.2171 | 96.6610±0.6962 | 97.7043±0.9371 | 96.9895±0.5789 |
| B5 CAM+Reliable/Hard Paired RBF-SVM | 97.3350±0.5978 | 97.2947±0.5834 | 99.1696±0.2131 | 96.7531±0.6087 | 97.8363±0.8921 | 97.1117±0.6408 |

## Error corrections

Wrong→correct / correct→wrong / net counts are summed across outer-fold OOF occurrences; a participant may occur in several outer-training diagnostics, so these are not unique-person totals.

| Comparison | Patient subset | Wrong→correct | Correct→wrong | Net |
|---|---|---:|---:|---:|
| B5 CAM+Reliable/Hard Paired RBF-SVM vs B0-NL CAM-only RBF-SVM | all | 16 | 5 | 11 |
| B5 CAM+Reliable/Hard Paired RBF-SVM vs B0-NL CAM-only RBF-SVM | stable_hard | 2 | 4 | -2 |
| B5 CAM+Reliable/Hard Paired RBF-SVM vs B0-NL CAM-only RBF-SVM | stable_reliable | 12 | 0 | 12 |
| B5 CAM+Reliable/Hard Paired RBF-SVM vs B4 CAM+Reliable RBF-SVM | all | 13 | 5 | 8 |
| B5 CAM+Reliable/Hard Paired RBF-SVM vs B4 CAM+Reliable RBF-SVM | stable_hard | 2 | 4 | -2 |
| B5 CAM+Reliable/Hard Paired RBF-SVM vs B4 CAM+Reliable RBF-SVM | stable_reliable | 9 | 0 | 9 |

## Audit and decision

Independent audit: **PASS**; ten frozen checkpoints, 30/30 OOF method-fold metrics independently recalculated, B0-NL baseline rechecked against the original E76.2 cache, B4/B5 predictions and decision scores deterministically reproduced, five-fold validation partitions and three-fold evidence cross-fitting verified, test-support intersection 0/10, Fold 2 rows 377/407 excluded. No parameter updates or model training.

Decision: **INCONCLUSIVE**. Interpret the comparison under the frozen-representation and historical evaluation limitations above; do not treat the inner-OOF accuracy as outer test generalization.

Artifacts: `fold_metrics.csv`, `paired_comparisons.csv`, `correction_analysis.csv`, `audit_status.json`, `preflight.json`, and `manifest.json`.

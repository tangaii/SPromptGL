# Experiment 58 — FAVE-PEN

## Execution and base

Formal training was limited to E58-FULL and E58-NUM, seed 0, 10 folds each. The 53.3 group `CADR_WD_0p01_GFR_WD_0p11` is a read-only reference; each new fold was rebuilt from the 43.1 untrained anchor. The `/data/adodas/SPromptGL-main` data root and `/model/spromptgl` historical source root are distinct filesystems; cross-root identity was established by SHA256, not path aliasing.

The baseline audit verifies all ten reference checkpoint SHA values, fold splits, patient-pair caches, base initialization hashes, and optimizer weight decays. GFR and CAM Adam retain WD 0.11; CADR AdamW retains the actual sweep WD 0.01 in both groups. The new FAVE AdamW uses matrix WD 1e-4 and bias/normalization WD 0. The 10-epoch warmup is reproduced exactly; the two-step inactive-path replay matched the historical forward, loss, updates, optimizer state, and RNG state bitwise.

Audit correction before delivery: an initial report-generation path paired the selected WD=0.01/GFR-WD=0.11 checkpoint identities with metrics from a different read-only reference group. The source now binds selected-group fold metrics to the same-fold checkpoint SHA, epoch, and PASS audit record. The corrected 53.3 baseline is 92.0742% ACC, 91.8653% Hard-AUC, 93.6684% Probability-AUC, 94.6392% SPE, and 89.0915% SEN. No model training or checkpoint was changed; all 20 Experiment 58 checkpoints were independently read again after the correction.

## Method and evidence scope

FAVE-PEN fits a fold-local Ridge conditional mean for each of the 256 standardized CORR features using only outer-train TD subjects and categorical PHENO age-bin/site/sex indicators. Five-fold patient-level cross-fitting supplies residual scales for training TD rows. The variance estimate uses `sqrt(0.9 * feature_variance + 0.1 * median_variance)` with a 0.05 floor. Outer-train ASD and test features use the final model fit on outer-train TD; test labels are not used. ANAT/FUNC inputs produce a bounded quality gate, not a clinical reliability probability.

Only feature 0 is eligible for semantic matching: verified left thalamus–left anterior superior temporal gyrus. The ROI audit verifies 252/256 named connections; four unknown-extra-ROI edges (195, 225, 232, 236) are excluded from Prompt matching while remaining in base CORR. The sole directional literature hypothesis is increased thalamocortical connectivity from a small pediatric study. The finding is anatomically compatible, but its region/estimand is broader than this exact selected edge. Coverage is therefore 1/252, and the model does not treat support as an ASD diagnosis. A reversed-direction text is a counterfactual only; the match objective permits only the supported positive hypothesis or a reject slot. Contradiction and unknown states are reported separately.

The correct text vectors use OpenAI CLIP RN50; the downloaded checkpoint SHA matched the official RN50 SHA. E58-NUM has the same FAVE trainable initialization and dimensions but a fixed seeded anonymous vector bank; it retains the same anatomical/direction mask and is not a no-prior control. The prompt mapper, evidence encoder, reject score, quality gate, and bounded residual affect only the CAM CORR position after the final CAM encoder; the original OutputLayer and other three positions remain unchanged. The initial residual coefficient is zero.

The losses are the unchanged 53.3 objective plus `0.05 * L_match + 0.01 * L_verify`. Both evidence losses are restricted to outer-train patient indices. `L_match` assigns strongly increased deviations to the single supported prompt and contradictory/unknown rows to the learned reject slot. `L_verify` checks the signed interaction between the supported and reversed text using training-fold TD-reference substitution. This audits model faithfulness, not biological or clinical causality.

## Results

All values are percentages, mean ± population SD across the ten fixed folds. Hard-AUC uses hard predictions; Probability-AUC uses continuous probabilities.

| Group | ACC | Hard-AUC | Probability-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 历史只读参考 | 92.0742 ± 2.8427 | 91.8653 ± 2.8348 | 93.6684 ± 3.2619 | 89.0915 ± 3.5263 | 94.6392 ± 3.5168 | 91.8653 ± 2.8348 | 91.2470 ± 3.0658 |
| 等容量匿名向量对照 | 91.6144 ± 3.1353 | 91.3980 ± 3.1128 | 93.5309 ± 2.9064 | 88.5915 ± 4.0194 | 94.2044 ± 4.3427 | 91.3980 ± 3.1128 | 90.7465 ± 3.3504 |
| FAVE-PEN医学文本组 | 91.3845 ± 3.3868 | 91.1533 ± 3.4558 | 93.1971 ± 2.9079 | 88.0976 ± 5.5228 | 94.2091 ± 3.8720 | 91.1533 ± 3.4558 | 90.4152 ± 3.7959 |

Paired comparisons use the exact fold metrics and report percentage-point differences. Fold counts are not treated as independent clinical cohorts.

| Comparison | Metric | Mean Δ (pp) | Fold Δ (pp), folds 1–10 | Win/Tie/Loss |
|---|---|---:|---|---:|
| E58-FULL - E58-BASE-53.3 | acc | -0.6897 | `[0.0, -2.29885057, -1.14942529, 0.0, -1.14942529, 0.0, 0.0, 0.0, -2.29885057, 0.0]` | 0/6/4 |
| E58-FULL - E58-BASE-53.3 | hard_auc | -0.7120 | `[0.15568241, -2.30646872, -1.2195122, 0.0, -1.06382979, 0.0, 0.0, 0.0, -2.87234043, 0.18617021]` | 2/4/4 |
| E58-FULL - E58-BASE-53.3 | prob_auc | -0.4714 | `[-0.83030618, -1.06044539, 1.2195122, -1.54255319, 0.10638298, 0.0, -2.07446809, -0.37234043, 0.21276596, -0.37234043]` | 3/1/6 |
| E58-FULL - E58-BASE-53.3 | spe | -0.4302 | `[-2.12765957, -2.17391304, 0.0, 0.0, -2.12765957, 0.0, 0.0, 0.0, 4.25531915, -2.12765957]` | 1/5/4 |
| E58-FULL - E58-BASE-53.3 | sen | -0.9939 | `[2.43902439, -2.43902439, -2.43902439, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0, 2.5]` | 2/5/3 |
| E58-FULL - E58-BASE-53.3 | ba | -0.7120 | `[0.15568241, -2.30646872, -1.2195122, 0.0, -1.06382979, 0.0, 0.0, 0.0, -2.87234043, 0.18617021]` | 2/4/4 |
| E58-FULL - E58-BASE-53.3 | asd_f1 | -0.8318 | `[0.12820513, -2.4691358, -1.41975309, 0.0, -1.11666177, 0.0, 0.0, 0.0, -3.63247863, 0.19230769]` | 2/4/4 |
| E58-FULL - E58-NUM | acc | -0.2299 | `[0.0, 0.0, -1.14942529, 0.0, -1.14942529, 0.0, 0.0, 0.0, -1.14942529, 1.14942529]` | 1/6/3 |
| E58-FULL - E58-NUM | hard_auc | -0.2446 | `[0.15568241, 0.13255567, -1.48462354, 0.0, -1.06382979, 0.0, 0.0, 0.0, -1.62234043, 1.43617021]` | 3/4/3 |
| E58-FULL - E58-NUM | prob_auc | -0.3338 | `[-0.41515309, -0.31813362, -0.42417815, -0.85106383, 0.69148936, -0.74468085, -1.38297872, -0.53191489, -0.15957447, 0.79787234]` | 2/0/8 |
| E58-FULL - E58-NUM | spe | +0.0046 | `[-2.12765957, -2.17391304, 4.34782609, 0.0, -2.12765957, 0.0, 0.0, 0.0, 4.25531915, -2.12765957]` | 2/4/4 |
| E58-FULL - E58-NUM | sen | -0.4939 | `[2.43902439, 2.43902439, -7.31707317, 0.0, 0.0, 0.0, 0.0, 0.0, -7.5, 5.0]` | 3/5/2 |
| E58-FULL - E58-NUM | ba | -0.2446 | `[0.15568241, 0.13255567, -1.48462354, 0.0, -1.06382979, 0.0, 0.0, 0.0, -1.62234043, 1.43617021]` | 3/4/3 |
| E58-FULL - E58-NUM | asd_f1 | -0.3313 | `[0.12820513, 0.34380372, -2.05882353, 0.0, -1.11666177, 0.0, 0.0, 0.0, -2.2005772, 1.59090909]` | 3/4/3 |
| E58-NUM - E58-BASE-53.3 | acc | -0.4598 | `[0.0, -2.29885057, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.14942529, -1.14942529]` | 0/7/3 |
| E58-NUM - E58-BASE-53.3 | hard_auc | -0.4674 | `[0.0, -2.43902439, 0.26511135, 0.0, 0.0, 0.0, 0.0, 0.0, -1.25, -1.25]` | 1/6/3 |
| E58-NUM - E58-BASE-53.3 | prob_auc | -0.1375 | `[-0.41515309, -0.74231177, 1.64369035, -0.69148936, -0.58510638, 0.74468085, -0.69148936, 0.15957447, 0.37234043, -1.17021277]` | 4/0/6 |
| E58-NUM - E58-BASE-53.3 | spe | -0.4348 | `[0.0, 0.0, -4.34782609, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | 0/9/1 |
| E58-NUM - E58-BASE-53.3 | sen | -0.5000 | `[0.0, -4.87804878, 4.87804878, 0.0, 0.0, 0.0, 0.0, 0.0, -2.5, -2.5]` | 1/6/3 |
| E58-NUM - E58-BASE-53.3 | ba | -0.4674 | `[0.0, -2.43902439, 0.26511135, 0.0, 0.0, 0.0, 0.0, 0.0, -1.25, -1.25]` | 1/6/3 |
| E58-NUM - E58-BASE-53.3 | asd_f1 | -0.5004 | `[0.0, -2.81293952, 0.63907044, 0.0, 0.0, 0.0, 0.0, 0.0, -1.43190143, -1.3986014]` | 1/6/3 |

## Mechanism audit

| Group | Mean test reject rate | Mean test residual/CORR-position ratio | Mean quality-gate SD | Mean classification-to-FAVE gradient norm |
|---|---:|---:|---:|---:|
| E58-NUM | 0.9471 | 0.000446846 | 0.0896 | 0.00010601 |
| E58-FULL | 0.9414 | 0.000516544 | 0.1087 | 0.000171393 |

Fixed-checkpoint interventions freeze CADR-GFR output and CAM encoder features; only the FAVE evidence assignment/residual and unchanged OutputLayer are recomputed. They are not additional training groups.

| Intervention | Group | Mean ACC Δ (pp) | Mean Probability-AUC Δ (pp) | Prediction flips / test patients | Mean probability L1 |
|---|---|---:|---:|---:|---:|
| anatomy_mismatch_prompt | E58-FULL | +0.0000 | +0.0000 | 0/871 | 2.26026e-06 |
| correct_prompt | E58-FULL | +0.0000 | +0.0000 | 0/871 | 0 |
| correct_prompt | E58-NUM | +0.0000 | +0.0000 | 0/871 | 0 |
| direction_reversed_prompt | E58-FULL | +0.0000 | +0.0000 | 0/871 | 3.71664e-06 |
| test_connection_to_TD_reference | E58-FULL | +0.0000 | -0.0053 | 0/871 | 1.99518e-05 |
| test_connection_to_TD_reference | E58-NUM | +0.0000 | +0.0000 | 0/871 | 2.12648e-05 |
| test_connection_to_unmapped_hypothesis_edge | E58-FULL | +0.0000 | -0.0053 | 0/871 | 2.68843e-05 |
| test_connection_to_unmapped_hypothesis_edge | E58-NUM | +0.0000 | +0.0000 | 0/871 | 2.59854e-05 |
| zero_fave_residual | E58-FULL | +0.0000 | -0.0053 | 0/871 | 1.99518e-05 |
| zero_fave_residual | E58-NUM | +0.0000 | +0.0000 | 0/871 | 2.12648e-05 |

## Conclusion

Engineering status: `GO` if all 20 checkpoints independently read back and the preflight passed; see checkpoint audit CSV. Semantic value versus the matched anonymous control: `NOT_ESTABLISHED` by the pre-registered descriptive criterion (positive mean ACC difference and at least 7/10 ACC wins); this is not a significance test. Diagnostic increment versus the historical baseline: `NOT_ESTABLISHED` under the same descriptive ACC criterion. Overall exploratory decision: **NO-GO**.

The evidence base has only one directionally supported connection out of 252 named edges, so the study is a narrow proof-of-mechanism test, not broad anatomical Prompt validation. A nonzero residual, gradient, or attention assignment alone does not establish semantic value. Report support, contradiction, and unknown as model evidence states only. The retained protocol uses val=test for checkpoint selection, CV-before-fold feature selection/standardization, an 871-node transductive graph, and a single seed; these results are protocol-matched exploratory estimates and do not establish unbiased external generalization.

## Files and provenance

See `baseline_audit.json` for fold-by-fold source, anchor, split, pair-cache, reference-checkpoint, and initialization SHA values; `manifest.json` records final training/audit status. Runtime checkpoints and caches are kept under `.runtime/`.

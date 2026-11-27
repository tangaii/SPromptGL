# DReSP Dual Label-Guided t-SNE Exploration

## A. STATUS

COMPLETE_AUDITED

All coordinates using true or shuffled labels are marked LABEL-GUIDED / SUPERVISED / EXPLORATORY in the result rows and PNG titles. This is a visualization study and reports no new ACC, AUC, SEN, SPE, or F1 claim.

## B. Input provenance and SHA

The 40 historical feature archives and 40 historical C5 arrays were copied byte-for-byte from the compact-separation experiment. CAM-only E43.4-E0 features were regenerated from the physical CAM-only network and are stored as ten complete 871-patient NPZ archives.
CAM-only checkpoint status: `REBUILT_FROM_VERIFIED_E43.1_ANCHORS_AND_MATCHED_E43.4_E0_HISTORY`. The original E43.4-E0 Best binaries were not present in the available archive. The ten E43.1 anchors matched the E43.4 manifest hashes, so the exact E0 protocol was rebuilt and matched historical per-fold predictions and metrics.

The SPromptGL source is the isolated E28-E0 seed-0 retrain from the prior feature audit. Its per-fold best epoch and historical metrics match, but original historical Best checkpoint byte identity was unavailable. DARG, DARG + JSL, and DReSP have per-fold source Best checkpoint paths and SHA-256 recorded in `inputs/model_checkpoint_provenance.json`. The prior DARG + JSL top-level manifest remains `BLOCKED_PREFLIGHT`, while its saved checkpoint files and result audit are independently hashed here.

The SPromptGL source is the isolated E28-E0 seed-0 retrain from the prior feature audit. Its per-fold best epoch and historical metrics match, but original historical Best checkpoint byte identity was unavailable. DARG, DARG + JSL, and DReSP have per-fold source Best checkpoint paths and SHA-256 recorded in `inputs/model_checkpoint_provenance.json`. The prior DARG + JSL top-level manifest remains `BLOCKED_PREFLIGHT`, while its saved checkpoint files and result audit are independently hashed here.

## C. Method definitions

Method A applies z_new = mu_class + alpha (z_original - mu_class) + beta (mu_class - mu_global) to the historical or replayed C5 coordinates. It is a post-hoc label-guided transformation.
Method B computes StandardScaler then PCA50, forms Euclidean D, scales same-class distances by s and adds gamma times the median positive PCA50 pair distance to different-class distances, then runs TSNE(metric=precomputed, init=random, random_state=0, n_iter=3000). It is supervised t-SNE with labels in the distance input.

## D. Formal groups

A uses P0-P9, B uses S0-S9, and shuffled-label controls use N1 and N2. Every group was applied to CAM-only, SPromptGL, DARG, DARG + JSL, and DReSP across all ten folds. C5 is retained as the original unsupervised reference.

## E. Parameter scan

The scan table contains 500 rows covering 25 Method A configurations and 25 Method B configurations on Fold 7 and Fold 8 for five models. No group was removed based on its score.

## F. Five-model ten-fold statistics

`formal_summary.csv` reports mean and population SD across ten folds. `formal_metrics.csv` retains every model, group, and fold. The tightest observed formal geometry row by raw mean within-class distance was `A/P9/DReSP`. The largest between/within ratio was `A/P9/DARG`. These extrema are mathematical consequences of the label constraints and are not model-learning gains.

## G. Paired comparisons

`paired_comparisons.csv` includes C5 versus every formal group, S0 versus S1-S9, shuffled controls versus their matched true-label settings, and CAM-only versus each other model. C5 versus S0 includes the random-init difference. S0 versus S1-S9 is the cleaner comparison for the distance modification itself.

## H. Method A versus Method B

Method A contracts or separates an existing two-dimensional layout by construction. Method B changes the high-dimensional neighborhood relation before t-SNE and has a separate random-init reference S0. Their coordinates are not interchangeable and should not be interpreted on a shared absolute axis across independent model fits.

## I. Compactness and separation

Within-class distances, covariance scales, center distance, normalized within distance, between/within ratio, Silhouette, and cross-class neighbor rates are reported together. Lower within distance and higher center distance can arise solely from the prescribed transform, so these numbers describe the visualization geometry.

## J. Trustworthiness and high-dimensional structure

Trustworthiness@15/@30 and PCA50-to-2D neighborhood overlap always use the unmodified PCA50 as the reference. The lowest observed formal Trustworthiness@15 row was `A/P7/DARG`. This metric is the relevant check for whether a stronger visual constraint sacrifices local structure.

## K. Label permutation controls

N1 uses alpha=0.65 and beta=0.50 with a single patient-level permutation seeded 2027. N2 uses s=0.75 and gamma=0.50 with the same permutation. Metrics for N1 and N2 are calculated with true labels. They test how much separation can be created by an external label assignment.

## L. Ten-fold figures

PNG files are under `results/figures/`. Scan previews use 300 DPI. Formal and paired figures use 600 DPI. Each five-model grid uses the requested 2x3 layout. All plots contain 871 points and explicit method and label-use text.

## M. DReSP fold-level performance

DReSP rows remain available in `formal_metrics.csv` and `paired_comparisons.csv` by fold. Any relative position after label guidance is a visualization comparison. It does not establish a DReSP diagnostic advantage.

## N. Label-guided bias

True test-patient labels were used in the visualization. P9 and the strongest B groups are intentionally extreme constraints. Their separation is expected mathematically and cannot validate independent disease representation.

## O. Scientific interpretation

The report separates natural C5 geometry, post-hoc label-guided geometry, supervised distance-driven geometry, and neighborhood preservation. A tighter or more separated two-dimensional plot is not evidence of improved ACC, AUC, generalization, or learned diagnostic representation.

## P. Paper-use limits

This study uses the real labels of the same 871 patients used to form the input archives. It provides no independent test-set generalization evidence. Any figure used in a paper must retain its method label and the exploratory/supervised qualification.

## Q. Independent audit

The independent audit is written to `results/audit_status.json`. The final status is COMPLETE_AUDITED only if the CAM-only identity, source SHA, all coordinates, all rows, label permutation, PNG inventory, and population-SD recomputation pass.

## R. Files and handoff

The authoritative directory is `/model/spromptgl/experiment/experiment_dresp_tsne_five_models_label_guided`. `SHA256SUMS` is generated after all outputs and the single append to `/model/spromptgl/experiment/handoff.md`. The manifest does not contain a self-SHA reference.

## Selected descriptive values

C5 model summaries are in `raw_c5_comparison.csv`. DReSP C5 two-dimensional Silhouette mean is 0.46258575048746575; CAM-only C5 two-dimensional Silhouette mean is 0.423329205141464.

## S. Pooled quantitative findings across all five models

The following pooled values use all five models and ten folds for each formal setting. They describe geometry, not classification performance.

| Method/group | Silhouette 2D | ASD within | TD within | Mean within | Normalized within | Center distance | Mean cross-class distance | Between/within | Trust@15 | Trust@30 | Fit KL | Original-space KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A/P0 | 0.438 ± 0.133 | 8.905 ± 1.160 | 9.298 ± 1.009 | 9.101 ± 1.057 | 0.718 ± 0.098 | 14.973 ± 4.180 | 17.642 ± 2.737 | 1.703 ± 0.582 | 0.954 ± 0.011 | 0.937 ± 0.014 | n/a | 0.707 ± 0.089 |
| A/P1 | 0.494 ± 0.139 | 7.569 ± 0.986 | 7.903 ± 0.857 | 7.736 ± 0.898 | 0.669 ± 0.106 | 14.973 ± 4.180 | 16.912 ± 3.001 | 2.003 ± 0.685 | 0.946 ± 0.012 | 0.931 ± 0.015 | n/a | 0.745 ± 0.091 |
| A/P2 | 0.583 ± 0.144 | 5.788 ± 0.754 | 6.044 ± 0.656 | 5.916 ± 0.687 | 0.587 ± 0.116 | 14.973 ± 4.180 | 16.098 ± 3.379 | 2.620 ± 0.896 | 0.936 ± 0.015 | 0.920 ± 0.018 | n/a | 0.829 ± 0.099 |
| A/P3 | 0.690 ± 0.143 | 4.007 ± 0.522 | 4.184 ± 0.454 | 4.096 ± 0.476 | 0.476 ± 0.124 | 14.973 ± 4.180 | 15.505 ± 3.750 | 3.784 ± 1.294 | 0.938 ± 0.016 | 0.922 ± 0.019 | n/a | 0.951 ± 0.106 |
| A/P4 | 0.575 ± 0.144 | 8.905 ± 1.160 | 9.298 ± 1.009 | 9.101 ± 1.057 | 0.594 ± 0.115 | 22.459 ± 6.271 | 24.237 ± 5.020 | 2.554 ± 0.874 | 0.936 ± 0.015 | 0.921 ± 0.018 | n/a | 0.807 ± 0.101 |
| A/P5 | 0.662 ± 0.144 | 8.905 ± 1.160 | 9.298 ± 1.009 | 9.101 ± 1.057 | 0.507 ± 0.122 | 29.945 ± 8.361 | 31.264 ± 7.323 | 3.406 ± 1.165 | 0.937 ± 0.016 | 0.921 ± 0.019 | n/a | 0.873 ± 0.115 |
| A/P6 | 0.625 ± 0.145 | 7.569 ± 0.986 | 7.903 ± 0.857 | 7.736 ± 0.898 | 0.544 ± 0.120 | 22.459 ± 6.271 | 23.735 ± 5.306 | 3.005 ± 1.028 | 0.936 ± 0.016 | 0.920 ± 0.019 | n/a | 0.844 ± 0.108 |
| A/P7 | 0.700 ± 0.142 | 5.788 ± 0.754 | 6.044 ± 0.656 | 5.916 ± 0.687 | 0.465 ± 0.125 | 22.459 ± 6.271 | 23.198 ± 5.668 | 3.930 ± 1.344 | 0.938 ± 0.016 | 0.922 ± 0.019 | n/a | 0.916 ± 0.115 |
| A/P8 | 0.836 ± 0.111 | 4.007 ± 0.522 | 4.184 ± 0.454 | 4.096 ± 0.476 | 0.351 ± 0.103 | 29.945 ± 8.361 | 30.202 ± 8.144 | 7.568 ± 2.589 | 0.941 ± 0.012 | 0.925 ± 0.015 | n/a | 1.110 ± 0.138 |
| A/P9 | 0.939 ± 0.046 | 2.226 ± 0.290 | 2.324 ± 0.252 | 2.275 ± 0.264 | 0.332 ± 0.055 | 44.918 ± 12.541 | 44.968 ± 12.500 | 20.434 ± 6.990 | 0.943 ± 0.011 | 0.926 ± 0.014 | n/a | 1.454 ± 0.182 |
| A/N1 | 0.436 ± 0.132 | 5.836 ± 0.749 | 6.085 ± 0.653 | 5.960 ± 0.683 | 0.720 ± 0.098 | 9.766 ± 2.724 | 11.510 ± 1.784 | 1.694 ± 0.576 | 0.943 ± 0.013 | 0.929 ± 0.015 | n/a | 0.819 ± 0.090 |
| B/S0 | 0.424 ± 0.150 | 8.983 ± 1.253 | 9.421 ± 1.272 | 9.202 ± 1.137 | 0.729 ± 0.110 | 14.557 ± 4.790 | 17.484 ± 3.017 | 1.651 ± 0.648 | 0.953 ± 0.012 | 0.937 ± 0.014 | 0.730 ± 0.097 | 0.709 ± 0.095 |
| B/S1 | 0.563 ± 0.122 | 8.167 ± 0.782 | 8.598 ± 0.683 | 8.382 ± 0.681 | 0.623 ± 0.091 | 19.818 ± 5.231 | 21.431 ± 4.200 | 2.411 ± 0.730 | 0.956 ± 0.010 | 0.940 ± 0.012 | 0.668 ± 0.081 | 0.722 ± 0.085 |
| B/S2 | 0.729 ± 0.063 | 7.280 ± 0.508 | 7.866 ± 0.525 | 7.573 ± 0.499 | 0.490 ± 0.051 | 29.878 ± 5.646 | 30.560 ± 5.455 | 3.971 ± 0.833 | 0.955 ± 0.010 | 0.940 ± 0.012 | 0.597 ± 0.066 | 0.851 ± 0.142 |
| B/S3 | 0.772 ± 0.034 | 7.153 ± 0.447 | 7.754 ± 0.444 | 7.454 ± 0.423 | 0.459 ± 0.024 | 34.072 ± 4.489 | 34.631 ± 4.393 | 4.587 ± 0.658 | 0.954 ± 0.010 | 0.939 ± 0.012 | 0.576 ± 0.054 | 0.914 ± 0.186 |
| B/S4 | 0.821 ± 0.022 | 6.726 ± 0.418 | 7.396 ± 0.399 | 7.061 ± 0.392 | 0.436 ± 0.017 | 40.531 ± 3.621 | 40.958 ± 3.577 | 5.762 ± 0.634 | 0.953 ± 0.009 | 0.937 ± 0.011 | 0.554 ± 0.046 | 0.998 ± 0.218 |
| B/S5 | 0.862 ± 0.014 | 6.406 ± 0.383 | 7.147 ± 0.455 | 6.777 ± 0.408 | 0.426 ± 0.014 | 49.761 ± 4.161 | 50.059 ± 4.110 | 7.363 ± 0.681 | 0.952 ± 0.009 | 0.936 ± 0.011 | 0.534 ± 0.050 | 1.080 ± 0.245 |
| B/S6 | 0.875 ± 0.009 | 6.328 ± 0.368 | 7.097 ± 0.442 | 6.713 ± 0.397 | 0.428 ± 0.016 | 54.208 ± 4.152 | 54.484 ± 4.133 | 8.086 ± 0.577 | 0.952 ± 0.010 | 0.936 ± 0.011 | 0.532 ± 0.052 | 1.113 ± 0.252 |
| B/S7 | 0.874 ± 0.011 | 6.368 ± 0.359 | 7.125 ± 0.430 | 6.747 ± 0.389 | 0.428 ± 0.015 | 54.134 ± 4.299 | 54.410 ± 4.258 | 8.045 ± 0.720 | 0.952 ± 0.010 | 0.936 ± 0.012 | 0.526 ± 0.053 | 1.109 ± 0.256 |
| B/S8 | 0.875 ± 0.009 | 6.330 ± 0.357 | 7.085 ± 0.432 | 6.708 ± 0.386 | 0.427 ± 0.015 | 54.033 ± 3.682 | 54.297 ± 3.654 | 8.072 ± 0.596 | 0.952 ± 0.010 | 0.936 ± 0.011 | 0.529 ± 0.053 | 1.110 ± 0.253 |
| B/S9 | 0.875 ± 0.009 | 6.379 ± 0.350 | 7.120 ± 0.422 | 6.749 ± 0.378 | 0.426 ± 0.015 | 54.291 ± 4.312 | 54.566 ± 4.289 | 8.055 ± 0.617 | 0.952 ± 0.009 | 0.935 ± 0.010 | 0.525 ± 0.052 | 1.107 ± 0.250 |
| B/N2 | 0.069 ± 0.026 | 28.610 ± 2.224 | 28.631 ± 3.051 | 28.620 ± 2.011 | 1.760 ± 0.176 | 5.285 ± 2.799 | 30.846 ± 2.024 | 0.185 ± 0.099 | 0.936 ± 0.012 | 0.914 ± 0.014 | 0.481 ± 0.061 | 3.310 ± 0.065 |

Across pooled formal rows, the smallest mean within-class distance occurred at `A/P9`. The largest center-distance/within-distance ratio occurred at `A/P9`. The lowest Trustworthiness@15 occurred at `A/P2`.

### C5 raw and two-dimensional reference by model

| Model | Raw 144D Silhouette | PCA50 Silhouette | C5 2D Silhouette | C5 Trust@15 | C5 Trust@30 | C5 KL |
|---|---:|---:|---:|---:|---:|---:|
| CAM-only | 0.2720 | 0.1765 | 0.4233 | 0.9565 | 0.9390 | 0.7348 |
| SPromptGL | 0.3645 | 0.1489 | 0.4084 | 0.9497 | 0.9311 | 0.7729 |
| DARG | 0.3867 | 0.1698 | 0.4565 | 0.9537 | 0.9362 | 0.7289 |
| DARG + JSL | 0.3359 | 0.1703 | 0.4405 | 0.9529 | 0.9369 | 0.7124 |
| DReSP | 0.3422 | 0.1888 | 0.4626 | 0.9586 | 0.9442 | 0.6877 |

### Method-specific paired deltas

A is paired against C5. B is paired against S0, which holds the unmodified distance matrix with random initialization. Values are pooled over five models and ten folds.

| Setting | Δ Silhouette | Δ normalized within | Δ center distance | Δ mean cross-class distance | Δ between/within | Δ Trust@15 | Δ Trust@30 | Δ original-space KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A/P0 vs C5 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| A/P1 vs C5 | 0.056 ± 0.011 | -0.049 ± 0.010 | -0.000 ± 0.000 | -0.730 ± 0.327 | 0.301 ± 0.103 | -0.008 ± 0.003 | -0.006 ± 0.002 | 0.038 ± 0.007 |
| A/P2 vs C5 | 0.145 ± 0.026 | -0.131 ± 0.025 | 0.000 ± 0.000 | -1.544 ± 0.776 | 0.917 ± 0.314 | -0.019 ± 0.007 | -0.017 ± 0.006 | 0.122 ± 0.019 |
| A/P3 vs C5 | 0.252 ± 0.043 | -0.242 ± 0.041 | -0.000 ± 0.000 | -2.137 ± 1.213 | 2.081 ± 0.712 | -0.016 ± 0.010 | -0.016 ± 0.010 | 0.244 ± 0.028 |
| A/P4 vs C5 | 0.136 ± 0.025 | -0.123 ± 0.024 | 7.486 ± 2.090 | 6.595 ± 2.339 | 0.851 ± 0.291 | -0.018 ± 0.006 | -0.017 ± 0.006 | 0.099 ± 0.021 |
| A/P5 vs C5 | 0.223 ± 0.039 | -0.211 ± 0.037 | 14.973 ± 4.180 | 13.622 ± 4.667 | 1.703 ± 0.582 | -0.017 ± 0.009 | -0.016 ± 0.009 | 0.166 ± 0.036 |
| A/P6 vs C5 | 0.187 ± 0.033 | -0.174 ± 0.032 | 7.486 ± 2.090 | 6.093 ± 2.652 | 1.302 ± 0.445 | -0.019 ± 0.009 | -0.018 ± 0.008 | 0.137 ± 0.028 |
| A/P7 vs C5 | 0.262 ± 0.045 | -0.253 ± 0.042 | 7.486 ± 2.090 | 5.556 ± 3.049 | 2.227 ± 0.762 | -0.016 ± 0.010 | -0.016 ± 0.010 | 0.208 ± 0.037 |
| A/P8 vs C5 | 0.397 ± 0.064 | -0.367 ± 0.052 | 14.973 ± 4.180 | 12.560 ± 5.545 | 5.865 ± 2.006 | -0.013 ± 0.007 | -0.013 ± 0.007 | 0.403 ± 0.067 |
| A/P9 vs C5 | 0.500 ± 0.097 | -0.386 ± 0.060 | 29.945 ± 8.361 | 27.326 ± 9.894 | 18.731 ± 6.407 | -0.012 ± 0.003 | -0.012 ± 0.003 | 0.747 ± 0.117 |
| B/S1 vs S0 | 0.139 ± 0.065 | -0.105 ± 0.049 | 5.261 ± 2.234 | 3.947 ± 1.650 | 0.760 ± 0.279 | 0.002 ± 0.005 | 0.003 ± 0.006 | 0.013 ± 0.036 |
| B/S2 vs S0 | 0.306 ± 0.103 | -0.239 ± 0.081 | 15.321 ± 3.177 | 13.076 ± 3.278 | 2.320 ± 0.445 | 0.002 ± 0.006 | 0.003 ± 0.007 | 0.142 ± 0.062 |
| B/S3 vs S0 | 0.348 ± 0.125 | -0.269 ± 0.101 | 19.515 ± 3.007 | 17.147 ± 2.547 | 2.936 ± 0.405 | 0.001 ± 0.007 | 0.001 ± 0.007 | 0.205 ± 0.110 |
| B/S4 vs S0 | 0.397 ± 0.141 | -0.293 ± 0.113 | 25.974 ± 4.432 | 23.473 ± 3.459 | 4.111 ± 0.687 | -0.000 ± 0.006 | 0.000 ± 0.007 | 0.289 ± 0.144 |
| B/S5 vs S0 | 0.438 ± 0.147 | -0.302 ± 0.105 | 35.204 ± 6.101 | 32.575 ± 4.996 | 5.712 ± 0.855 | -0.001 ± 0.007 | -0.001 ± 0.008 | 0.371 ± 0.174 |
| B/S6 vs S0 | 0.452 ± 0.148 | -0.300 ± 0.104 | 39.651 ± 6.123 | 37.000 ± 4.912 | 6.435 ± 0.761 | -0.002 ± 0.007 | -0.002 ± 0.008 | 0.404 ± 0.183 |
| B/S7 vs S0 | 0.450 ± 0.147 | -0.301 ± 0.107 | 39.577 ± 5.977 | 36.926 ± 4.977 | 6.394 ± 0.842 | -0.001 ± 0.007 | -0.001 ± 0.008 | 0.400 ± 0.187 |
| B/S8 vs S0 | 0.451 ± 0.147 | -0.302 ± 0.108 | 39.475 ± 5.719 | 36.813 ± 4.322 | 6.421 ± 0.744 | -0.002 ± 0.007 | -0.002 ± 0.008 | 0.401 ± 0.186 |
| B/S9 vs S0 | 0.451 ± 0.147 | -0.303 ± 0.106 | 39.734 ± 6.015 | 37.081 ± 4.828 | 6.403 ± 0.754 | -0.002 ± 0.007 | -0.002 ± 0.008 | 0.398 ± 0.182 |

### Scan results

`kl_divergence` is the fitted t-SNE objective from the C5 fit or the modified-distance Method B fit. Method A does not refit t-SNE, so its fit KL is not applicable and is left blank. `original_space_kl_divergence` is computed for every coordinate set from a full-pairwise symmetric P derived once from unmodified PCA50 and a Student-t Q from the resulting 2D coordinates. This common reference-space KL supports comparison across C5, A, and B.

`parameter_scan_summary.csv` aggregates each of the 25 Method A and 25 Method B settings over all five models and Fold 7/8. It retains the pooled population SD and every underlying row remains in `parameter_scan.csv`.

### Model ordering changes and DReSP comparison

C5 two-dimensional Silhouette ordering from highest to lowest was `DReSP, DARG, DARG + JSL, CAM-only, SPromptGL`. 17 of the twenty true-label formal parameter settings changed that ordering. The detailed model and fold pairings are in `paired_comparisons.csv` and `paired_group_deltas.csv`.
DReSP does not have a single coordinate-wise visual advantage because each model's t-SNE is fitted independently. The pooled geometry metrics can change rank under label constraints. Such reversals measure the sensitivity of each model's own feature geometry and the explicit label intervention.

### DReSP ten-fold detail for predeclared examples

| Method/group | Fold | Silhouette 2D | ASD within | TD within | Center distance | B/W ratio | Trust@15 | Trust@30 | t-SNE fit KL | Original-space KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C5/C5 | 01 | 0.1037 | 13.6803 | 12.5685 | 5.8023 | 0.4421 | 0.9521 | 0.9265 | 0.8305 | 0.7815 |
| C5/C5 | 02 | 0.5707 | 7.9633 | 8.6075 | 19.1849 | 2.3155 | 0.9603 | 0.9479 | 0.5870 | 0.5732 |
| C5/C5 | 03 | 0.4558 | 9.4739 | 9.1403 | 16.4648 | 1.7691 | 0.9517 | 0.9358 | 0.7431 | 0.7181 |
| C5/C5 | 04 | 0.5215 | 7.9926 | 9.0101 | 17.1879 | 2.0218 | 0.9610 | 0.9494 | 0.7194 | 0.7003 |
| C5/C5 | 05 | 0.3029 | 10.2434 | 10.6915 | 11.1689 | 1.0670 | 0.9844 | 0.9769 | 0.5678 | 0.5503 |
| C5/C5 | 06 | 0.4596 | 9.0860 | 9.2254 | 15.7219 | 1.7172 | 0.9608 | 0.9466 | 0.7206 | 0.6994 |
| C5/C5 | 07 | 0.5595 | 8.0943 | 8.5609 | 18.6208 | 2.2360 | 0.9448 | 0.9302 | 0.6359 | 0.6189 |
| C5/C5 | 08 | 0.6279 | 7.3679 | 7.4854 | 20.6434 | 2.7796 | 0.9440 | 0.9276 | 0.7343 | 0.7263 |
| C5/C5 | 09 | 0.4445 | 8.6051 | 9.0328 | 14.5931 | 1.6547 | 0.9604 | 0.9470 | 0.6803 | 0.6664 |
| C5/C5 | 10 | 0.5797 | 8.3760 | 8.8426 | 20.1065 | 2.3354 | 0.9665 | 0.9539 | 0.6580 | 0.6433 |
| A/P0 | 01 | 0.1037 | 13.6803 | 12.5685 | 5.8023 | 0.4421 | 0.9521 | 0.9265 | n/a | 0.7815 |
| A/P0 | 02 | 0.5707 | 7.9633 | 8.6075 | 19.1849 | 2.3155 | 0.9603 | 0.9479 | n/a | 0.5732 |
| A/P0 | 03 | 0.4558 | 9.4739 | 9.1403 | 16.4648 | 1.7691 | 0.9517 | 0.9358 | n/a | 0.7181 |
| A/P0 | 04 | 0.5215 | 7.9926 | 9.0101 | 17.1879 | 2.0218 | 0.9610 | 0.9494 | n/a | 0.7003 |
| A/P0 | 05 | 0.3029 | 10.2434 | 10.6915 | 11.1689 | 1.0670 | 0.9844 | 0.9769 | n/a | 0.5503 |
| A/P0 | 06 | 0.4596 | 9.0860 | 9.2254 | 15.7219 | 1.7172 | 0.9608 | 0.9466 | n/a | 0.6994 |
| A/P0 | 07 | 0.5595 | 8.0943 | 8.5609 | 18.6208 | 2.2360 | 0.9448 | 0.9302 | n/a | 0.6189 |
| A/P0 | 08 | 0.6279 | 7.3679 | 7.4854 | 20.6434 | 2.7796 | 0.9440 | 0.9276 | n/a | 0.7263 |
| A/P0 | 09 | 0.4445 | 8.6051 | 9.0328 | 14.5931 | 1.6547 | 0.9604 | 0.9470 | n/a | 0.6664 |
| A/P0 | 10 | 0.5797 | 8.3760 | 8.8426 | 20.1065 | 2.3354 | 0.9665 | 0.9539 | n/a | 0.6433 |
| A/P6 | 01 | 0.1840 | 11.6283 | 10.6832 | 8.7034 | 0.7802 | 0.9095 | 0.8842 | n/a | 0.9330 |
| A/P6 | 02 | 0.7467 | 6.7688 | 7.3164 | 28.7773 | 4.0862 | 0.9470 | 0.9368 | n/a | 0.6705 |
| A/P6 | 03 | 0.6611 | 8.0528 | 7.7693 | 24.6972 | 3.1219 | 0.9273 | 0.9141 | n/a | 0.8718 |
| A/P6 | 04 | 0.7076 | 6.7937 | 7.6586 | 25.7818 | 3.5678 | 0.9492 | 0.9376 | n/a | 0.8181 |
| A/P6 | 05 | 0.5162 | 8.7069 | 9.0878 | 16.7534 | 1.8830 | 0.9741 | 0.9672 | n/a | 0.6860 |
| A/P6 | 06 | 0.6654 | 7.7231 | 7.8416 | 23.5828 | 3.0303 | 0.9470 | 0.9350 | n/a | 0.8486 |
| A/P6 | 07 | 0.7373 | 6.8801 | 7.2767 | 27.9312 | 3.9460 | 0.9333 | 0.9205 | n/a | 0.7221 |
| A/P6 | 08 | 0.7822 | 6.2627 | 6.3626 | 30.9651 | 4.9052 | 0.9314 | 0.9151 | n/a | 0.8351 |
| A/P6 | 09 | 0.6512 | 7.3144 | 7.6779 | 21.8897 | 2.9201 | 0.9438 | 0.9316 | n/a | 0.8089 |
| A/P6 | 10 | 0.7428 | 7.1196 | 7.5162 | 30.1597 | 4.1214 | 0.9517 | 0.9396 | n/a | 0.7798 |
| A/P9 | 01 | 0.8079 | 3.4201 | 3.1421 | 17.4068 | 5.3052 | 0.9478 | 0.9269 | n/a | 1.4061 |
| A/P9 | 02 | 0.9638 | 1.9908 | 2.1519 | 57.5546 | 27.7860 | 0.9474 | 0.9366 | n/a | 1.1615 |
| A/P9 | 03 | 0.9526 | 2.3685 | 2.2851 | 49.3945 | 21.2287 | 0.9380 | 0.9215 | n/a | 1.5265 |
| A/P9 | 04 | 0.9584 | 1.9982 | 2.2525 | 51.5636 | 24.2614 | 0.9503 | 0.9387 | n/a | 1.3677 |
| A/P9 | 05 | 0.9218 | 2.5608 | 2.6729 | 33.5068 | 12.8042 | 0.9778 | 0.9694 | n/a | 1.4012 |
| A/P9 | 06 | 0.9513 | 2.2715 | 2.3064 | 47.1656 | 20.6060 | 0.9490 | 0.9354 | n/a | 1.4681 |
| A/P9 | 07 | 0.9625 | 2.0236 | 2.1402 | 55.8624 | 26.8325 | 0.9360 | 0.9228 | n/a | 1.2316 |
| A/P9 | 08 | 0.9698 | 1.8420 | 1.8713 | 61.9301 | 33.3556 | 0.9329 | 0.9151 | n/a | 1.3182 |
| A/P9 | 09 | 0.9493 | 2.1513 | 2.2582 | 43.7793 | 19.8568 | 0.9484 | 0.9349 | n/a | 1.4063 |
| A/P9 | 10 | 0.9640 | 2.0940 | 2.2107 | 60.3194 | 28.0252 | 0.9525 | 0.9382 | n/a | 1.3154 |
| A/N1 | 01 | 0.1030 | 8.9594 | 8.2275 | 3.8051 | 0.4428 | 0.9323 | 0.9158 | n/a | 0.9320 |
| A/N1 | 02 | 0.5631 | 5.2864 | 5.7071 | 12.5252 | 2.2786 | 0.9376 | 0.9311 | n/a | 0.7555 |
| A/N1 | 03 | 0.4529 | 6.2017 | 5.9862 | 10.7432 | 1.7629 | 0.9357 | 0.9211 | n/a | 0.8422 |
| A/N1 | 04 | 0.5192 | 5.2503 | 5.8879 | 11.2116 | 2.0132 | 0.9474 | 0.9390 | n/a | 0.8152 |
| A/N1 | 05 | 0.3023 | 6.6901 | 6.9557 | 7.2847 | 1.0677 | 0.9806 | 0.9744 | n/a | 0.6312 |
| A/N1 | 06 | 0.4563 | 5.9682 | 6.0522 | 10.2516 | 1.7057 | 0.9485 | 0.9382 | n/a | 0.8212 |
| A/N1 | 07 | 0.5582 | 5.2989 | 5.5796 | 12.1334 | 2.2307 | 0.9357 | 0.9243 | n/a | 0.7123 |
| A/N1 | 08 | 0.6240 | 4.8242 | 4.9689 | 13.4671 | 2.7503 | 0.9242 | 0.9113 | n/a | 0.8778 |
| A/N1 | 09 | 0.4450 | 5.6048 | 5.8676 | 9.4998 | 1.6561 | 0.9585 | 0.9466 | n/a | 0.7201 |
| A/N1 | 10 | 0.5759 | 5.5194 | 5.8126 | 13.1123 | 2.3142 | 0.9527 | 0.9417 | n/a | 0.7868 |
| B/S0 | 01 | 0.1074 | 9.9006 | 14.8428 | 5.6537 | 0.4570 | 0.9625 | 0.9442 | 0.8210 | 0.7698 |
| B/S0 | 02 | 0.5541 | 8.2720 | 8.4726 | 18.6428 | 2.2267 | 0.9569 | 0.9437 | 0.5857 | 0.5701 |
| B/S0 | 03 | 0.4796 | 8.9929 | 8.6094 | 16.7046 | 1.8980 | 0.9441 | 0.9280 | 0.7367 | 0.7133 |
| B/S0 | 04 | 0.5039 | 8.0029 | 9.0821 | 16.6390 | 1.9478 | 0.9586 | 0.9443 | 0.7082 | 0.6900 |
| B/S0 | 05 | 0.1457 | 14.0587 | 9.4242 | 5.5483 | 0.4725 | 0.9789 | 0.9726 | 0.6303 | 0.6128 |
| B/S0 | 06 | 0.4197 | 9.2320 | 9.9624 | 14.7579 | 1.5377 | 0.9600 | 0.9456 | 0.7220 | 0.7003 |
| B/S0 | 07 | 0.5615 | 8.4912 | 8.7242 | 18.9547 | 2.2021 | 0.9487 | 0.9327 | 0.6373 | 0.6191 |
| B/S0 | 08 | 0.6283 | 7.3057 | 7.5612 | 20.8476 | 2.8046 | 0.9484 | 0.9355 | 0.6885 | 0.6809 |
| B/S0 | 09 | 0.4502 | 8.1381 | 9.0198 | 14.4661 | 1.6862 | 0.9609 | 0.9515 | 0.6563 | 0.6446 |
| B/S0 | 10 | 0.5856 | 7.9795 | 8.6129 | 20.3615 | 2.4543 | 0.9613 | 0.9492 | 0.6466 | 0.6326 |
| B/S6 | 01 | 0.8673 | 7.2409 | 8.2476 | 58.6330 | 7.5712 | 0.9597 | 0.9376 | 0.5034 | 1.3241 |
| B/S6 | 02 | 0.8761 | 6.4965 | 7.2603 | 55.7696 | 8.1080 | 0.9521 | 0.9406 | 0.4502 | 0.7700 |
| B/S6 | 03 | 0.8617 | 6.1503 | 6.7011 | 46.3758 | 7.2173 | 0.9470 | 0.9317 | 0.5025 | 1.0694 |
| B/S6 | 04 | 0.8794 | 6.5734 | 7.0354 | 56.6926 | 8.3318 | 0.9528 | 0.9397 | 0.5623 | 0.9980 |
| B/S6 | 05 | 0.8651 | 6.9874 | 7.6179 | 54.2399 | 7.4274 | 0.9805 | 0.9744 | 0.3643 | 1.0311 |
| B/S6 | 06 | 0.8613 | 6.7134 | 7.3780 | 50.7926 | 7.2090 | 0.9598 | 0.9432 | 0.5419 | 1.0757 |
| B/S6 | 07 | 0.8781 | 6.1299 | 7.0123 | 54.2205 | 8.2514 | 0.9346 | 0.9188 | 0.5479 | 0.8863 |
| B/S6 | 08 | 0.8872 | 5.8450 | 6.6525 | 55.9953 | 8.9610 | 0.9425 | 0.9229 | 0.5771 | 0.9022 |
| B/S6 | 09 | 0.8892 | 6.0216 | 6.7722 | 58.2630 | 9.1080 | 0.9547 | 0.9427 | 0.5450 | 1.1054 |
| B/S6 | 10 | 0.8800 | 6.3438 | 7.3232 | 57.3413 | 8.3912 | 0.9545 | 0.9415 | 0.5315 | 0.9137 |
| B/S8 | 01 | 0.8742 | 7.4242 | 8.0980 | 62.0279 | 7.9921 | 0.9634 | 0.9391 | 0.4982 | 1.3331 |
| B/S8 | 02 | 0.8695 | 6.3954 | 7.1366 | 51.9076 | 7.6719 | 0.9557 | 0.9420 | 0.4619 | 0.7729 |
| B/S8 | 03 | 0.8675 | 6.2027 | 6.7974 | 49.0735 | 7.5497 | 0.9483 | 0.9328 | 0.5071 | 1.0842 |
| B/S8 | 04 | 0.8742 | 6.4025 | 6.9558 | 53.3230 | 7.9835 | 0.9560 | 0.9423 | 0.5453 | 0.9814 |
| B/S8 | 05 | 0.8658 | 6.9943 | 7.5547 | 54.3551 | 7.4720 | 0.9810 | 0.9743 | 0.3472 | 1.0299 |
| B/S8 | 06 | 0.8717 | 6.6808 | 7.4629 | 55.3984 | 7.8337 | 0.9607 | 0.9429 | 0.5345 | 1.0734 |
| B/S8 | 07 | 0.8802 | 6.3404 | 6.9810 | 55.7404 | 8.3686 | 0.9367 | 0.9253 | 0.5140 | 0.8676 |
| B/S8 | 08 | 0.8870 | 5.7833 | 6.6048 | 55.0295 | 8.8842 | 0.9404 | 0.9212 | 0.5664 | 0.8972 |
| B/S8 | 09 | 0.8845 | 6.1401 | 6.9143 | 56.7147 | 8.6890 | 0.9562 | 0.9460 | 0.5020 | 1.0501 |
| B/S8 | 10 | 0.8734 | 6.4148 | 7.2039 | 53.9799 | 7.9273 | 0.9500 | 0.9373 | 0.5695 | 0.9335 |
| B/N2 | 01 | 0.0619 | 27.1789 | 26.7821 | 8.1615 | 0.3025 | 0.9391 | 0.9083 | 0.5959 | 3.1963 |
| B/N2 | 02 | 0.0914 | 25.1442 | 28.1338 | 6.5258 | 0.2450 | 0.9415 | 0.9240 | 0.3796 | 3.2240 |
| B/N2 | 03 | 0.0450 | 28.9034 | 33.8294 | 4.8722 | 0.1553 | 0.9302 | 0.9000 | 0.5094 | 3.3721 |
| B/N2 | 04 | 0.0979 | 23.0678 | 29.1536 | 8.0345 | 0.3077 | 0.9393 | 0.9167 | 0.4765 | 3.2712 |
| B/N2 | 05 | 0.0433 | 26.9546 | 29.0393 | 4.6624 | 0.1665 | 0.9739 | 0.9533 | 0.4532 | 3.2095 |
| B/N2 | 06 | 0.0676 | 25.2459 | 25.9690 | 2.9318 | 0.1145 | 0.9424 | 0.9256 | 0.4814 | 3.2286 |
| B/N2 | 07 | 0.0877 | 28.0935 | 30.9304 | 10.4412 | 0.3538 | 0.9319 | 0.9116 | 0.4344 | 3.3221 |
| B/N2 | 08 | 0.1157 | 32.5931 | 23.8631 | 3.2670 | 0.1157 | 0.9349 | 0.9170 | 0.4039 | 3.3399 |
| B/N2 | 09 | 0.0683 | 26.0337 | 27.7995 | 2.6882 | 0.0999 | 0.9456 | 0.9305 | 0.4546 | 3.2946 |
| B/N2 | 10 | 0.1138 | 29.1117 | 24.8928 | 9.8379 | 0.3643 | 0.9499 | 0.9334 | 0.4098 | 3.2834 |

## T. Constraint strength and candidate visualization sets

P9 applies alpha=0.25 and beta=2.00, the most aggressive post-hoc transformation in the registered Method A grid. S8 applies s=0.40 and gamma=1.00, the strongest combined distance constraint in Method B. These are over-constrained for claims about naturally learned structure when their separation increases together with a Trustworthiness decline relative to their proper baseline.
P6 and S6 are retained as moderate-constraint figure examples alongside C5, S0, P9/S8, and both shuffled-label controls. They are disclosure-complete exploration examples rather than selected model winners. The paper-facing candidate files are indexed by `selected_visualization_examples.json`.

## U. Required interpretation boundary

Original model feature separability is represented by raw144/PCA50/C5 values. Method A is a post-hoc mathematical rearrangement of C5. Method B uses labels before optimization. Neither label-guided Silhouette nor a more separated plot is an independent classification measure. Test-patient labels participated in this visualization, so no independent test generalization evidence is available.

# DReSP ABIDE t-SNE Compactness and Separation

## A. STATUS

COMPLETE_AUDITED_TSNE_COMPACT_SEPARATION. The computed audit result is recorded in `results/audit_status.json` and `manifest.json`.

## B. Historical Inputs and Parameter Identity

- Frozen feature source: `/model/spromptgl/experiment/experiment_dresp_tsne_visual_optimization/inputs/`, 40 archives across four models and ten folds.
- Every archive contains 871 ordered patient IDs, binary labels, and finite 144D features. The copy is checked byte-for-byte against the historical input manifest.
- Feature layers remain SPromptGL CAM144, DARG CAM144, DARG + JSL enhanced CAM144, and DReSP enhanced CAM144.
- Repository comparison: [SPromptGL on GitHub](https://github.com/tangaii/SPromptGL), checked at commit `4ba363c68bceeedc5721e197975270204fa313a4`. The historical formal workflow uses StandardScaler, PCA50, and t-SNE. No model, feature extractor, training code, weights, or historical experiment artifacts were changed.
- Runtime: Python `3.12.3`, scikit-learn `1.3.2`, TSNE iteration keyword `n_iter`.
- Shared preprocessing: StandardScaler → PCA(50, random_state=0) → t-SNE. The PCA50 representation is the reference space for Trustworthiness.

## C. Parameter Scan Results

The targeted scan covered only Fold 7 and Fold 8, all four models, and the requested 13 candidates. It contains 104 metric rows, of which 16 B5/B6 runs reuse historical coordinates after SHA and configuration checks. The other 88 runs were fitted independently. All candidates used Euclidean distance, PCA initialization, random state 0, and 3000 iterations. No run produced non-finite coordinates.

Values show mean ± population SD across the eight model-fold combinations for each candidate.

| Candidate | Perplexity | Early exaggeration | Learning rate | Silhouette 2D | Normalized Within | Separation Ratio | Cross-class @15 | Trustworthiness @15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 100 | 16 | 200 | 0.440335 ± 0.103919 | 0.623225 ± 0.095028 | 2.357887 ± 0.751191 | 0.086778 ± 0.030172 | 0.956016 ± 0.005533 |
| C1 | 150 | 24 | 500 | 0.440493 ± 0.090098 | 0.622297 ± 0.080240 | 2.325557 ± 0.635906 | 0.084797 ± 0.027892 | 0.952475 ± 0.006398 |
| C2 | 80 | 24 | 200 | 0.443960 ± 0.086890 | 0.617475 ± 0.079685 | 2.362878 ± 0.627650 | 0.090356 ± 0.027268 | 0.953153 ± 0.004933 |
| C3 | 100 | 24 | 200 | 0.441696 ± 0.092482 | 0.621854 ± 0.081844 | 2.341748 ± 0.635657 | 0.087113 ± 0.025434 | 0.955886 ± 0.002826 |
| C4 | 100 | 32 | 200 | 0.436635 ± 0.097407 | 0.624587 ± 0.087074 | 2.313172 ± 0.684902 | 0.089562 ± 0.028150 | 0.952078 ± 0.004512 |
| C5 | 100 | 48 | 200 | 0.456651 ± 0.087244 | 0.605278 ± 0.080140 | 2.462361 ± 0.670171 | 0.085390 ± 0.025094 | 0.954165 ± 0.006668 |
| C6 | 100 | 32 | 500 | 0.439852 ± 0.094011 | 0.622057 ± 0.087280 | 2.348111 ± 0.706045 | 0.088777 ± 0.026972 | 0.955874 ± 0.004755 |
| C7 | 150 | 32 | 200 | 0.443747 ± 0.089965 | 0.615749 ± 0.078617 | 2.361659 ± 0.629282 | 0.086816 ± 0.030070 | 0.952712 ± 0.005912 |
| C8 | 150 | 32 | 500 | 0.450389 ± 0.089919 | 0.611591 ± 0.084433 | 2.412749 ± 0.663047 | 0.084137 ± 0.025787 | 0.951400 ± 0.006422 |
| C9 | 150 | 48 | 500 | 0.449806 ± 0.088417 | 0.614888 ± 0.086333 | 2.406327 ± 0.698674 | 0.085199 ± 0.027713 | 0.951235 ± 0.006760 |
| C10 | 150 | 64 | 500 | 0.451435 ± 0.093611 | 0.608658 ± 0.084998 | 2.420135 ± 0.672343 | 0.085400 ± 0.028280 | 0.950340 ± 0.004211 |
| C11 | 200 | 24 | 500 | 0.453063 ± 0.081034 | 0.612983 ± 0.074762 | 2.401999 ± 0.593719 | 0.082090 ± 0.025742 | 0.946952 ± 0.005577 |
| C12 | 200 | 32 | 500 | 0.449460 ± 0.085813 | 0.612628 ± 0.079706 | 2.389775 ± 0.648969 | 0.081822 ± 0.025942 | 0.948116 ± 0.006001 |

KL divergence, runtime, bounds, input and coordinate hashes, and per-run metrics are in `results/parameter_comparison.csv`. KL is diagnostic only and was not used to rank candidates across differing perplexities or feature distributions.

## D. Final Parameter Choice

The selection rule was frozen before formal ten-fold generation. It compares the four-model mean against B5, discards candidates whose Trustworthiness@15 falls more than 0.015 below B5, then ranks by mean Silhouette 2D, lower Normalized Within, higher Separation Ratio, lower cross-class neighbor rates, and higher Trustworthiness@15. It selected C5 uniformly for all models and folds.

Final shared parameters are perplexity 100, early exaggeration 48, learning rate 200, Euclidean metric, PCA initialization, random state 0, and 3000 iterations. C5 raises the scan mean Silhouette 2D from 0.440335 for B5 to 0.456651, lowers Normalized Within from 0.623225 to 0.605278, and changes mean Trustworthiness@15 by -0.001850. This is an exploratory visualization choice based on label-derived metrics.

## E. Four-Model Results

All values are mean ± population SD (`ddof=0`) across ten folds. Trustworthiness is measured against PCA50, the representation passed into t-SNE.

| Model | Silhouette 2D | Silhouette raw 144D | Silhouette standardized 144D | Silhouette PCA50 | Trustworthiness @15 | Trustworthiness @30 | Normalized Within | Separation Ratio | Cross-class rate @15 | Cross-class rate @30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPromptGL | 0.408409 ± 0.088650 | 0.364482 ± 0.109351 | 0.148436 ± 0.030176 | 0.148904 ± 0.030281 | 0.949665 ± 0.012076 | 0.931053 ± 0.016942 | 0.651284 ± 0.081367 | 2.126870 ± 0.565533 | 0.088680 ± 0.031782 | 0.096445 ± 0.035493 |
| DARG | 0.456493 ± 0.128751 | 0.386738 ± 0.093446 | 0.169197 ± 0.046946 | 0.169755 ± 0.047142 | 0.953713 ± 0.007466 | 0.936240 ± 0.009170 | 0.604832 ± 0.105452 | 2.484676 ± 0.802658 | 0.090027 ± 0.054559 | 0.095254 ± 0.062783 |
| DARG + JSL | 0.440536 ± 0.090006 | 0.335923 ± 0.127086 | 0.169743 ± 0.032066 | 0.170332 ± 0.032278 | 0.952934 ± 0.012640 | 0.936916 ± 0.014717 | 0.623838 ± 0.080235 | 2.320672 ± 0.586087 | 0.084998 ± 0.034828 | 0.091209 ± 0.039259 |
| DReSP | 0.462586 ± 0.148410 | 0.342247 ± 0.114899 | 0.188117 ± 0.036243 | 0.188807 ± 0.036341 | 0.958581 ± 0.011122 | 0.944176 ± 0.014395 | 0.602345 ± 0.130112 | 2.542786 ± 0.922525 | 0.072997 ± 0.021766 | 0.079005 ± 0.031022 |

The all-model mean ± population SD across 40 runs is Silhouette 2D 0.442006 ± 0.118666; Silhouette raw 144D 0.357348 ± 0.113627; Silhouette standardized 144D 0.168873 ± 0.039513; Silhouette PCA50 0.169449 ± 0.039685; Trustworthiness @15 0.953723 ± 0.011464; Trustworthiness @30 0.937096 ± 0.014852; Normalized Within 0.620575 ± 0.103249; Separation Ratio 2.368751 ± 0.752194; Cross-class rate @15 0.084175 ± 0.038255; Cross-class rate @30 0.090478 ± 0.044429.

Fold 7 and Fold 8 informed parameter selection. The remaining eight folds are descriptive because prior rounds of parameter exploration had already inspected other folds. They are not a strictly independent validation set.

| Model | Silhouette 2D | Silhouette raw 144D | Silhouette standardized 144D | Silhouette PCA50 | Trustworthiness @15 | Trustworthiness @30 | Normalized Within | Separation Ratio | Cross-class rate @15 | Cross-class rate @30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPromptGL | 0.409210 ± 0.098510 | 0.370168 ± 0.121087 | 0.151853 ± 0.032381 | 0.152309 ± 0.032525 | 0.948503 ± 0.013192 | 0.929791 ± 0.018635 | 0.651268 ± 0.090634 | 2.143656 ± 0.629004 | 0.090011 ± 0.035377 | 0.097704 ± 0.039456 |
| DARG | 0.456823 ± 0.143728 | 0.395686 ± 0.087811 | 0.173730 ± 0.051383 | 0.174314 ± 0.051587 | 0.952248 ± 0.007643 | 0.933866 ± 0.008767 | 0.602591 ± 0.117508 | 2.512593 ± 0.892934 | 0.091705 ± 0.060818 | 0.097225 ± 0.070036 |
| DARG + JSL | 0.457541 ± 0.093105 | 0.369809 ± 0.117790 | 0.171313 ± 0.032815 | 0.171933 ± 0.033017 | 0.951569 ± 0.013553 | 0.936690 ± 0.016298 | 0.610675 ± 0.084724 | 2.424678 ± 0.612588 | 0.075708 ± 0.032777 | 0.080468 ± 0.036722 |
| DReSP | 0.429806 ± 0.147873 | 0.322698 ± 0.119314 | 0.183111 ± 0.037453 | 0.183732 ± 0.037548 | 0.962129 ± 0.009573 | 0.947993 ± 0.013628 | 0.633062 ± 0.126197 | 2.300465 ± 0.850703 | 0.078062 ± 0.021479 | 0.085142 ± 0.031815 |

## F. B5/B6 Comparison

The B5 comparison below uses the historical ten-fold metrics and C5's formal ten-fold results. For B6, only the verified historical Fold 7/8 coordinates were reused, so full-ten-fold B6 comparison is unavailable. The tables below give C5 minus each baseline on the eight selection runs.

| Model | B5 → C5 Silhouette 2D (Δ) | B5 → C5 Trustworthiness @15 (Δ) | B5 → C5 Trustworthiness @30 (Δ) | B5 → C5 Separation Ratio (Δ) |
| --- | --- | --- | --- | --- |
| SPromptGL | 0.393875 ± 0.091283 → 0.408409 ± 0.088650 (+0.014534) | 0.953071 ± 0.012474 → 0.949665 ± 0.012076 (-0.003406) | 0.938198 ± 0.015172 → 0.931053 ± 0.016942 (-0.007144) | 1.990356 ± 0.568963 → 2.126870 ± 0.565533 (+0.136514) |
| DARG | 0.453943 ± 0.127672 → 0.456493 ± 0.128751 (+0.002550) | 0.956345 ± 0.007589 → 0.953713 ± 0.007466 (-0.002632) | 0.939535 ± 0.009934 → 0.936240 ± 0.009170 (-0.003294) | 2.462693 ± 0.768072 → 2.484676 ± 0.802658 (+0.021982) |
| DARG + JSL | 0.437051 ± 0.103277 → 0.440536 ± 0.090006 (+0.003486) | 0.957253 ± 0.010473 → 0.952934 ± 0.012640 (-0.004319) | 0.941724 ± 0.012258 → 0.936916 ± 0.014717 (-0.004808) | 2.311288 ± 0.650610 → 2.320672 ± 0.586087 (+0.009384) |
| DReSP | 0.456910 ± 0.154395 → 0.462586 ± 0.148410 (+0.005676) | 0.961736 ± 0.009724 → 0.958581 ± 0.011122 (-0.003155) | 0.947811 ± 0.012438 → 0.944176 ± 0.014395 (-0.003635) | 2.515956 ± 0.977077 → 2.542786 ± 0.922525 (+0.026830) |

| Reference | Model | Silhouette 2D delta | Normalized Within delta | Separation Ratio delta | Cross-class rate @15 delta | Cross-class rate @30 delta | Trustworthiness @15 delta | Trustworthiness @30 delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B5 | SPromptGL | +0.018583 | -0.031140 | +0.168487 | +0.005473 | +0.006257 | -0.000541 | -0.003996 |
| B5 | DARG | +0.022186 | -0.021688 | +0.159743 | -0.000268 | -0.000096 | -0.000277 | +0.000319 |
| B5 | DARG + JSL | +0.033821 | -0.027899 | +0.163196 | -0.011902 | -0.009166 | -0.002156 | -0.003889 |
| B5 | DReSP | -0.009323 | +0.008940 | -0.073528 | +0.001148 | +0.001703 | -0.004428 | -0.002755 |
| B6 | SPromptGL | +0.022511 | -0.027336 | +0.163636 | +0.005013 | +0.007348 | -0.000773 | -0.004673 |
| B6 | DARG | +0.030414 | -0.030609 | +0.240730 | +0.002028 | +0.003004 | +0.005962 | +0.001826 |
| B6 | DARG + JSL | -0.003302 | +0.009581 | -0.055845 | -0.005741 | +0.000115 | +0.001626 | -0.004011 |
| B6 | DReSP | +0.015011 | -0.019713 | +0.198695 | +0.001072 | +0.001894 | -0.000053 | -0.004454 |

| Reference | Fold | Silhouette 2D delta | Normalized Within delta | Separation Ratio delta | Cross-class rate @15 delta | Cross-class rate @30 delta | Trustworthiness @15 delta | Trustworthiness @30 delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B5 | 7 | +0.004489 | -0.005982 | +0.007545 | -0.000593 | +0.001626 | -0.000910 | -0.000873 |
| B5 | 8 | +0.028144 | -0.029912 | +0.201404 | -0.002181 | -0.002277 | -0.002791 | -0.004288 |
| B6 | 7 | +0.008395 | -0.012524 | +0.081118 | +0.002602 | +0.004449 | +0.000955 | -0.003204 |
| B6 | 8 | +0.023923 | -0.021514 | +0.192490 | -0.001416 | +0.001732 | +0.002425 | -0.002452 |

The complete reference-by-model-by-fold changes are also recorded in `results/selected_parameters.json`.

## G. Within-Class Compactness

`Within` is the equal-weight mean of the ASD and TD average distances to their own centroids. It is stored with `Global_RMS_Scale` in `results/final_metrics.csv`. Since t-SNE's scale is arbitrary, the cross-configuration comparison uses `Normalized_Within = Within / Global_RMS_Scale`. Lower values indicate tighter class distributions relative to the embedding's overall scale. The ten-fold Normalized Within values are reported above for each model.

## H. Between-Class Separation

`Separation_Ratio = Between / Within` and the @15/@30 cross-class neighbor rates describe complementary aspects of class layout. Lower cross-class rates indicate less local mixing. The centroid ratio can underdescribe non-convex or fragmented clusters, so it is interpreted together with Silhouette and neighbor mixing. Trustworthiness measures structure retention independently of the label-based separation measures.

## I. Figures

- Candidate previews: `results/preview_figures/`, 26 PNGs covering C0-C12 on Fold 7 and Fold 8. Each figure uses a 2×2 grid in SPromptGL, DARG, DARG + JSL, DReSP order and includes all 871 points per panel.
- Formal individual plots: `results/formal_figures/individual/`, 40 PNGs.
- Formal same-fold four-model comparisons: `results/formal_figures/comparison/`, ten PNGs.
- All formal plots are 600 DPI, white background, fixed ASD/TD colors, shared equal axis scale within each comparison, and no grid. No PDF, SVG, or EPS files were produced.

Comparison images: [`Fold 01`](formal_figures/comparison/fold01_comparison.png) · [`Fold 02`](formal_figures/comparison/fold02_comparison.png) · [`Fold 03`](formal_figures/comparison/fold03_comparison.png) · [`Fold 04`](formal_figures/comparison/fold04_comparison.png) · [`Fold 05`](formal_figures/comparison/fold05_comparison.png) · [`Fold 06`](formal_figures/comparison/fold06_comparison.png) · [`Fold 07`](formal_figures/comparison/fold07_comparison.png) · [`Fold 08`](formal_figures/comparison/fold08_comparison.png) · [`Fold 09`](formal_figures/comparison/fold09_comparison.png) · [`Fold 10`](formal_figures/comparison/fold10_comparison.png).

## J. Audit and Invariance

The audit script independently checks the 40 historical and copied input hashes, archive arrays, patient IDs, labels, shapes and finite values. It replays B5/B6 historical preview coordinates, freshly recomputes all 40 final C5 embeddings, checks the CSV against the saved coordinates and input row order, recomputes the reported geometry and Trustworthiness metrics, validates the 50 PNGs and color pixels, and checks the fitting code for label-free t-SNE fitting. Its computed results are in `results/audit_status.json`.

The formal run manifest records 40 fresh fits from frozen features, a single C5 parameter set, and `labels_passed_to_tsne: false`. Labels are used only for post-fit assessment and exploratory candidate selection. No point was deleted or manually repositioned. All 871 transductive feature rows are present in each fold visualization. They are not all independent test patients for that fold.

## K. Scientific Limitations and Paper Use

Because true labels informed parameter selection through Silhouette and neighbor-mixing metrics, this is an exploratory, label-aware visualization selection. The final two-dimensional Silhouette is not an independent validation metric. The eight non-selection folds are descriptive, not strictly independent, given earlier parameter exploration. t-SNE coordinates depend on the chosen projection and are not evidence of classification accuracy, ACC, AUC, or independent generalization. Paper figures should be described as exploratory feature-space visualizations, with the label-informed selection and transductive inclusion of all 871 patients disclosed.

## L. Files and Handoff

- Coordinates: `results/tsne_coordinates.csv` (34,840 patient rows) and `results/formal_coordinates/` (40 NumPy coordinate arrays).
- Core outputs: `results/parameter_comparison.csv`, `results/selected_parameters.json`, `results/final_metrics.csv`, `results/audit_status.json`, and `manifest.json`.
- Reproducible code: `code/`.
- Handoff: the experiment summary is appended to `/model/spromptgl/experiment/handoff.md`. The 1,505,928-byte prior prefix retains SHA256 `03cdd64d6a967103af701b155d61269cedb248171b29d4f6c610e8bf4eead8a6`, and the 3,804-byte appended section hashes to `9a27182efa4d8e1e924c209d86ce7eef6444d54cff4ae93a6e06a9918b489cfe`.

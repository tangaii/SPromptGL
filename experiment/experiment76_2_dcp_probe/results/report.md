# Experiment 76.2 — DCP-Probe

Status: COMPLETE_AUDITED; independent metric/isolation audit PASS. ten frozen E74.4-G2 folds; no end-to-end training.

## Source audit

E74.4-G2 audit: 30/30; historical outer-test-selected reference ACC 92.1891±2.7648%, Hard-AUC 92.0294±2.7255%, Prob-AUC 93.4946±2.9195%. E76 audit: 40/40, NO-GO. E76.1 audit: 30/30, NO-GO. SPromptGL motivates using its real CAM144 classifier input; CoW motivates prediction/error-driven hard examples; PGP-SAM motivates inter/intra-class prototype diagnostics. PGP-SAM uses learnable embedding prototypes; the normalized means and medoids here are patient prototypes, not its prototype implementation. Historical checkpoint metrics are not directly comparable to the inner-OOF readouts below.

Nested validation is conditional on a frozen encoder already trained on the outer-training cohort, not end-to-end OOF. Each outer-train set is split into five stratified folds; only each 4/5 support set builds the prototype bank. Readout-training evidence uses an additional three-fold cross-fit to prevent self-matching. StandardScaler is fitted inside each readout training support. Outer test labels are not used.

Stable roles use frozen best-model predictions plus preregistered confidence and >=9/10 predictions under ten deterministic CAM144-RMS perturbations. These perturbations measure local classifier stability, not epoch persistence. Same-fold E61 warmup, E74.4 best, and final latest model/CAM weights are SHA-deduplicated and compared as saved snapshots; snapshot agreement is not called persistent training-time error. Fold 2 rows 377/407 remain in the historical base but are removed from all added support pools.

Sources: [SPromptGL](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [CoW](https://ijcai.org/proceedings/2025/139), [PGP-SAM](https://arxiv.org/abs/2501.06692).

## Nested OOF results

Mean ± population SD across ten outer folds (%); the paired unit is the outer fold (n=10), not 50 inner folds. B0-NL ROC-AUC uses uncalibrated decision scores; its log-loss and Brier score are not applicable.

| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| B0_CAM_only_Linear | 96.2494 ± 0.7561 | 96.2112 ± 0.7361 | 98.4720 ± 0.2909 | 95.6986 ± 0.6684 | 96.7237 ± 1.0968 | 95.9394 ± 0.8063 |
| B0_NL_CAM_only_RBF_SVM | 97.2571 ± 0.5750 | 97.2156 ± 0.5566 | 99.1698 ± 0.2213 | 96.6634 ± 0.7365 | 97.7678 ± 1.0150 | 97.0268 ± 0.6128 |
| B1_CAM_Reliable | 96.2494 ± 0.7689 | 96.2111 ± 0.7516 | 98.4826 ± 0.2877 | 95.6986 ± 0.7017 | 96.7237 ± 1.0763 | 95.9391 ± 0.8214 |
| B2_CAM_Reliable_Hard_Paired | 96.4408 ± 0.6111 | 96.3931 ± 0.6072 | 98.5296 ± 0.2717 | 95.7538 ± 0.6904 | 97.0323 ± 0.7676 | 96.1388 ± 0.6595 |
| B3_Shuffled_Pair_Control | 96.2979 ± 0.6723 | 96.2555 ± 0.6597 | 98.4892 ± 0.3049 | 95.6876 ± 0.6475 | 96.8235 ± 0.9212 | 95.9883 ± 0.7212 |

## Excluding Fold 2

Descriptive mean ± population SD across the remaining nine outer folds (%); omission does not remove the other historical evaluation biases.

| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| B0_CAM_only_Linear | 96.2153 ± 0.7896 | 96.1849 ± 0.7715 | 98.4797 ± 0.3057 | 95.7732 ± 0.6640 | 96.5967 ± 1.0841 | 95.9078 ± 0.8440 |
| B0_NL_CAM_only_RBF_SVM | 97.1791 ± 0.5537 | 97.1431 ± 0.5400 | 99.1456 ± 0.2203 | 96.6610 ± 0.7763 | 97.6251 ± 0.9700 | 96.9447 ± 0.5915 |
| B1_CAM_Reliable | 96.2153 ± 0.8033 | 96.1849 ± 0.7879 | 98.4916 ± 0.3019 | 95.7732 ± 0.7011 | 96.5966 ± 1.0610 | 95.9075 ± 0.8600 |
| B2_CAM_Reliable_Hard_Paired | 96.4705 ± 0.6373 | 96.4265 ± 0.6312 | 98.5513 ± 0.2780 | 95.8345 ± 0.6816 | 97.0186 ± 0.8079 | 96.1740 ± 0.6862 |
| B3_Shuffled_Pair_Control | 96.3145 ± 0.7067 | 96.2764 ± 0.6922 | 98.5186 ± 0.3078 | 95.7609 ± 0.6419 | 96.7918 ± 0.9658 | 96.0098 ± 0.7572 |

## Paired contrasts

Positive is in favor of B2; 95% intervals are descriptive paired t intervals over outer folds.

| B2 minus | Metric | Δ pp | 95% CI pp | W/T/L | Δ pp without Fold 2 |
|---|---|---:|---:|---:|---:|
| B0_CAM_only_Linear | acc | +0.191 | [-0.053, 0.436] | 6/2/2 | +0.255 |
| B0_CAM_only_Linear | hard_auc | +0.182 | [-0.051, 0.415] | 6/1/3 | +0.242 |
| B0_CAM_only_Linear | prob_auc | +0.058 | [-0.004, 0.119] | 7/0/3 | +0.072 |
| B0_CAM_only_Linear | sen | +0.055 | [-0.126, 0.236] | 3/5/2 | +0.061 |
| B0_CAM_only_Linear | spe | +0.309 | [-0.099, 0.717] | 6/3/1 | +0.422 |
| B0_CAM_only_Linear | asd_f1 | +0.199 | [-0.057, 0.456] | 6/1/3 | +0.266 |
| B0_NL_CAM_only_RBF_SVM | acc | -0.816 | [-1.184, -0.448] | 0/0/10 | -0.709 |
| B0_NL_CAM_only_RBF_SVM | hard_auc | -0.823 | [-1.195, -0.451] | 0/0/10 | -0.717 |
| B0_NL_CAM_only_RBF_SVM | prob_auc | -0.640 | [-0.859, -0.422] | 0/0/10 | -0.594 |
| B0_NL_CAM_only_RBF_SVM | sen | -0.910 | [-1.554, -0.265] | 2/0/8 | -0.827 |
| B0_NL_CAM_only_RBF_SVM | spe | -0.735 | [-1.286, -0.185] | 1/1/8 | -0.607 |
| B0_NL_CAM_only_RBF_SVM | asd_f1 | -0.888 | [-1.289, -0.487] | 0/0/10 | -0.771 |
| B1_CAM_Reliable | acc | +0.191 | [-0.033, 0.416] | 7/1/2 | +0.255 |
| B1_CAM_Reliable | hard_auc | +0.182 | [-0.033, 0.396] | 7/0/3 | +0.242 |
| B1_CAM_Reliable | prob_auc | +0.047 | [-0.007, 0.101] | 7/0/3 | +0.060 |
| B1_CAM_Reliable | sen | +0.055 | [-0.100, 0.211] | 4/4/2 | +0.061 |
| B1_CAM_Reliable | spe | +0.309 | [-0.058, 0.676] | 7/2/1 | +0.422 |
| B1_CAM_Reliable | asd_f1 | +0.200 | [-0.037, 0.436] | 7/0/3 | +0.267 |
| B3_Shuffled_Pair_Control | acc | +0.143 | [0.025, 0.261] | 9/0/1 | +0.156 |
| B3_Shuffled_Pair_Control | hard_auc | +0.138 | [0.024, 0.251] | 9/0/1 | +0.150 |
| B3_Shuffled_Pair_Control | prob_auc | +0.040 | [0.004, 0.077] | 8/0/2 | +0.033 |
| B3_Shuffled_Pair_Control | sen | +0.066 | [-0.042, 0.174] | 6/1/3 | +0.074 |
| B3_Shuffled_Pair_Control | spe | +0.209 | [0.015, 0.403] | 8/1/1 | +0.227 |
| B3_Shuffled_Pair_Control | asd_f1 | +0.150 | [0.025, 0.275] | 9/0/1 | +0.164 |

## H1 prototype evidence

Verdict: NOT_SUPPORTED; independent audit PASS. Stable role totals and per-fold counts: R_ASD: 3334 total, [338, 346, 349, 340, 354, 325, 320, 321, 347, 294]; R_TD: 4052 total, [392, 413, 387, 413, 401, 398, 407, 419, 414, 408]; H_ASD_to_TD: 128 total, [12, 9, 5, 13, 9, 9, 15, 25, 9, 22]; H_TD_to_ASD: 50 total, [9, 3, 10, 3, 9, 4, 4, 1, 3, 4].
Reliable held-out same-class minus other-class cosine was positive in 10/10 folds, average 0.8616. Reliable class prototypes and at least one hard-candidate role had ≥3 supports in 10/10 folds, so the result is not a shortage-only failure. The joint class-aligned hard-prototype rule also requires a positive hard-to-own-class versus hard-to-other-class contrast; 0/10 folds met it. Across the 19 available directional contrasts, the hard prototypes were closer to the opposite-class reliable prototype in every case. Same-fold history provided 3–3 distinct snapshots per fold; selected-best epochs 123–339, final-latest epochs 324–450; only 79/178 stable-error candidates retained the same wrong direction across every saved snapshot, so agreement is sparse checkpoint consistency, not continuous persistence. Site-matched nearest-neighbor same-minus-other similarity averaged 0.5080 across folds, suggesting class structure is not explained solely by requiring a different site, but site confounding remains possible. Pair-direction bootstrap cosine-to-full averaged 0.9742; vector stability did not correct the class-direction issue.

## H2 incremental utility

Pre-registered diagnostic conclusion: NO-GO: STOP_CURRENT_PROTOTYPE_MEMORY_DIRECTION. See the paired contrasts and foldwise predictions; B2−B0 tests added evidence over CAM144, B2−B1 tests the hard-pair increment, B2−B3 tests actual versus shuffled hard-pair roles, and B2 versus B0-NL tests whether the result exceeds an RBF readout. Five private pair shuffles are summarized per outer fold; their repeats are not independent folds.

## OOF Prediction Changes

Counts sum fold-level inner-OOF patient occurrences, not unique people, because patients can appear in multiple outer-training diagnostics. B3 totals average five private shuffles per fold.

| B2 versus | Subset | Wrong→correct | Correct→wrong | Net | ASD→TD errors before→after | TD→ASD errors before→after |
|---|---|---:|---:|---:|---:|---:|
| B0_CAM_only_Linear | all | 35.0 | 20.0 | 15.0 | 156.0→154.0 | 138.0→125.0 |
| B0_CAM_only_Linear | stable_hard | 1.0 | 4.0 | -3.0 | 99.0→101.0 | 42.0→43.0 |
| B0_CAM_only_Linear | stable_reliable | 21.0 | 8.0 | 13.0 | 20.0→19.0 | 55.0→43.0 |
| B1_CAM_Reliable | all | 33.0 | 18.0 | 15.0 | 156.0→154.0 | 138.0→125.0 |
| B1_CAM_Reliable | stable_hard | 1.0 | 4.0 | -3.0 | 99.0→101.0 | 42.0→43.0 |
| B1_CAM_Reliable | stable_reliable | 19.0 | 8.0 | 11.0 | 19.0→19.0 | 54.0→43.0 |
| B3_Shuffled_Pair_Control | all | 22.8 | 11.6 | 11.2 | 156.4→154.0 | 133.8→125.0 |
| B3_Shuffled_Pair_Control | stable_hard | 1.4 | 2.0 | -0.6 | 100.8→101.0 | 42.6→43.0 |
| B3_Shuffled_Pair_Control | stable_reliable | 14.2 | 4.8 | 9.4 | 21.0→19.0 | 50.4→43.0 |
| B0_NL_CAM_only_RBF_SVM | all | 46.0 | 110.0 | -64.0 | 121.0→154.0 | 94.0→125.0 |
| B0_NL_CAM_only_RBF_SVM | stable_hard | 6.0 | 20.0 | -14.0 | 91.0→101.0 | 39.0→43.0 |
| B0_NL_CAM_only_RBF_SVM | stable_reliable | 22.0 | 50.0 | -28.0 | 11.0→19.0 | 23.0→43.0 |

## Limitations

Historical checkpoint selection used the test fold, preprocessing was fit before CV, graph construction is transductive, and Fold 2 has two base train/test labels in common. No adjustment here removes those biases. Even favorable results are diagnostic only, conditional on the frozen representation, and do not validate a clinical mechanism or an end-to-end DCMP module.

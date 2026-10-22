# Experiment 34 — SeRCA: Semantic-guided Relational Complementary Alignment

Status: **COMPLETE**. Completed 100/100 folds; GPU0 50/50, GPU1 50/50. Total elapsed wall time: 1377.8s (0.38h); summed worker time: 4764.6s. Full worker timing and RNG evidence are in `manifest.json` and `.runtime/`.

## Research Question and Audit

This experiment tests whether frozen GPT-4 disease-functional-network concepts, constrained by the previously verified HO FC feature identities and medical sources, can condition the four already learned GFR subject-relation matrices. SeRCA operates on between-subject relations; it does not treat those matrices as ROI graphs, does not force modality feature equality, and does not use labels to create patient similarities.

The MICCAI 2025 SPromptGL paper reports 92.30% ABIDE accuracy for its full method. The local code is the implementation authority here: its four GraphLearn outputs are patient-to-patient matrices thresholded at cosine >0.9, GFR propagates the full 320D concatenated feature through each matrix normalized with the repository `normalize_adj(A+I)`, and the `i==1` branch overwrites the `i==0` `combined_feat`. PLN is constructed in its original order, but the B0-A forward bypasses PEN and passes CAM output directly to the unchanged OutputLayer. The original `simple-2` objective is `ClsLoss + mean(four GraphConstructLoss)`, while computed ALoss is not in backward. CAM `[871,4,36]` contains mixed latent positions; four 9D slices are not four anatomical ROIs.

Paper Eq. (1)-(7) and Figure 2 describe the broader SPromptGL components but do not resolve these local implementation details. The paper's 92.30% result is not the local re-trained B0-A result. Current Experiment 32/33 B0-A (92.0742% ACC) is the baseline reference. Experiments 11 and 19 supply the verified numeric FC identity map and the frozen disease-concept support sets. Experiments 12-18, 20-31 show repeated failures or weak evidence for free Prompt keys, forced feature alignment, semantic graph calibration, highly similar CLIP category vectors, or QC evidence; Experiment 32 found incorrect Prompt controls were not reliably worse, and Experiment 33 found wrong pairing and FC-only controls could beat its proposed QC condition.

Related work already includes semantic-conditioned brain interactions (KD-Brain), concept weighting/graphs (MMGPL), multimodal patient graph and shared/private learning (MGDR), conditional dependency modeling (InfoBridge), and evidence that forced alignment can help or hurt by task (To Align or Not to Align). SeRCA's narrow test is whether a frozen, anatomy-supported FC concept changes existing GFR edges and contributes a difference-message through the existing CAM path; semantic guidance, concepts, graph alignment, and complementarity are not claimed as novel by themselves.

## Frozen Concepts and Operator

| Concept | FC support |
|---|---:|
| D1 — Social/temporal-temporoparietal connectivity | 72 |
| D2 — Frontoparietal/executive connectivity | 56 |
| D3 — Thalamocortical/sensorimotor connectivity | 45 |
| D4 — Cingulate-insular/salience connectivity | 32 |

D1-D4 overlaps (shared FC features) are D1_D2=14, D1_D3=8, D1_D4=7, D2_D3=4, D2_D4=5, D3_D4=2. Features 195, 225, 232 and 236 remain in the original B0-A CORR input but are excluded from semantic concepts because verified anatomical endpoint names are unavailable. Concept membership is candidate measurement support, not evidence that every ASD participant has a lesion or a common FC direction. Random concepts use one local PCG64 bijection on the 252 verified feature identities, preserving each support size and all pairwise overlaps, but not anatomical topology.

For each concept, signed FC columns are standardized with outer-train means/scales. The concept similarity is `S^k_ij=exp(-||Z_i^k-Z_j^k||²/tau_k)`, where `tau_k` is the median positive outer-train pair distance and `center_k` is its outer-train similarity mean. These matrices are fixed by fold, label-free, and transductive over all 871 participants; held-out FC values may affect pair similarities, but not fitted scales.

For existing raw GFR adjacency `A_m`, SeRCA forms `W_mk=diag(A_m)+offdiag(A_m)⊙exp(S^k-center_k)`, preserving the diagonal and existing off-diagonal support, then `B_mk=normalize_adj(W_mk+I)` using the root utility. The ordinary `B_m=normalize_adj(A_m+I)` is captured from the same GFR forward. With CAM input `H` immediately after the existing InputLayer, SeRCA pools `R[t,m,k]=(B_mk-B_m)@H[:,t,:]` using a low-capacity softmax over relation/concept coefficients. Its bounded residual is `Delta[t]=0.10*tanh(alpha_t)*sum(pi[t,m,k]R[t,m,k])`, capped per subject/token at 10% of `||H||`; `alpha` starts at zero. No graph is learned separately, no GFR graph loss changed, and no CLIP text encoding was added.

## Frozen Groups and Protocol

| Group | Registered contrast | GPU |
|---|---|---:|
| S0_B0A | B0-A, no SeRCA intervention | 0 |
| S1_NUMERIC_GLOBAL | one 256D numeric FC kernel copied into four concept slots | 1 |
| S2_RANDOM_CONCEPTS | fixed size/overlap matched random FC feature identities | 0 |
| S3_SINGLE_CONCEPT | D1 only | 1 |
| S4_CORR_REL_ONLY | four concepts, CORR relation only | 0 |
| S5_AUX_REL_ONLY | four concepts, PHENO/ANAT/FUNC relation sources only | 1 |
| S6_UNIFORM_RELATION | fixed uniform relation/concept weights; theta frozen | 0 |
| S7_DIRECT_MESSAGE | conditioned neighborhood message without subtracting base message | 1 |
| S8_PATIENT_PERMUTED | within-site FC concept patient bijection; original input/GFR unchanged | 0 |
| S9_SERCA_FULL | registered full SeRCA primary model | 1 |

All ten groups ran one seed-0, ten-fold stratified CV over the existing standardized/RFE-selected 871×320 data, labels ASD=0 (403), TD=1 (468), and all-subject transductive GFR. Splits use `StratifiedKFold(10, shuffle=True, random_state=10)`. The protocol is `nepoch=450`, `early=200`, `lr=.0038`, `weight_decay=.11`, `dropout=.35`, `n_hidden=18`, `n_head=2`, `nlayer=1`, `weighted-cosine`, `GCN`, `concat`, `simple-2`. `val_idx=test_idx`; strict test/validation ACC improvement replaces the checkpoint and ties preserve the first. Hard-AUC uses the original hard one-hot predictions. The four formal metrics are ACC, Hard-AUC, SEN-ASD (ASD positive), SPE-TD.

S0 ran first as a continuous seed-0 fold stream and recorded Python, NumPy, PyTorch CPU and CUDA entry states before each model construction. S1-S9 each ran an independent continuous ten-fold worker but restored the matching S0 fold-entry RNG before each construction. This is a paired construction control; it is not natural continuous cross-fold RNG for those nine groups. Both original Adam optimizers and losses were preserved; new parameters use one additional PLN optimizer group at lr=.0038 and weight decay 0. S6 freezes theta and has four trainable alpha values rather than 68 trainable alpha/theta elements. Checkpoints include model, best model, both Adams, wait/epoch and RNG states and are written atomically.

## Four-Metric Results

Mean ± population SD across ten folds, in percent:

| Group | ACC | Hard-AUC | SEN-ASD | SPE-TD |
|---|---:|---:|---:|---:|
| S0_B0A | 92.0742 ± 2.6000% | 91.8546 ± 2.5815% | 89.0793 ± 3.7283% | 94.6300 ± 4.0457% |
| S1_NUMERIC_GLOBAL | 92.0742 ± 2.6503% | 91.8972 ± 2.6233% | 89.5854 ± 3.4710% | 94.2091 ± 3.8720% |
| S2_RANDOM_CONCEPTS | 91.6157 ± 2.8704% | 91.4377 ± 2.7959% | 89.3232 ± 3.5531% | 93.5523 ± 4.9698% |
| S3_SINGLE_CONCEPT | 91.7294 ± 2.9505% | 91.5066 ± 2.9408% | 88.5915 ± 4.3332% | 94.4218 ± 4.3370% |
| S4_CORR_REL_ONLY | 91.9592 ± 2.5772% | 91.7529 ± 2.5794% | 89.0793 ± 3.5652% | 94.4265 ± 3.3763% |
| S5_AUX_REL_ONLY | 91.5008 ± 3.0122% | 91.3421 ± 2.9315% | 89.3354 ± 3.3396% | 93.3488 ± 4.8764% |
| S6_UNIFORM_RELATION | 91.3845 ± 3.0154% | 91.2433 ± 3.0129% | 89.3415 ± 3.2911% | 93.1452 ± 3.1907% |
| S7_DIRECT_MESSAGE | 91.2696 ± 2.8788% | 91.0230 ± 2.8346% | 87.8415 ± 4.1970% | 94.2044 ± 4.7581% |
| S8_PATIENT_PERMUTED | 92.0729 ± 3.2341% | 91.8870 ± 3.2237% | 89.3476 ± 3.7841% | 94.4265 ± 3.6345% |
| S9_SERCA_FULL | 91.9592 ± 2.4726% | 91.7722 ± 2.4317% | 89.3354 ± 2.7208% | 94.2091 ± 3.5038% |

The complete 100 fold rows, best epoch, hard-confusion counts, initialization/RNG hashes and all same-fold deltas are in `fold_metrics.csv`.

## Paired Comparisons

Every delta is S9 minus its same-fold control. Mean differences are percentage points; W/T/L counts are shown for ACC and Hard-AUC. Each metric row includes all ten fold deltas in fold order 1→10. The counts are descriptive, not inferential statistics.

| Contrast | Metric | Mean Δ (pp) | W/T/L | Fold deltas (pp, folds 1→10) |
|---|---|---:|---:|---|
| S9 − S0_B0A | ACC | -0.1149 | 1/7/2 | +0.0000, +0.0000, +0.0000, +0.0000, -1.1494, +0.0000, +1.1494, +0.0000, +0.0000, -1.1494 |
| S9 − S0_B0A | Hard-AUC | -0.0824 | 2/4/4 | +0.0000, +0.1326, -0.2651, +0.0000, -0.8777, -0.1862, +1.6223, +0.0000, +0.0000, -1.2500 |
| S9 − S0_B0A | SEN-ASD | +0.2561 | — | +0.0000, +2.4390, -4.8780, +0.0000, +2.5000, -2.5000, +7.5000, +0.0000, +0.0000, -2.5000 |
| S9 − S0_B0A | SPE-TD | -0.4209 | — | +0.0000, -2.1739, +4.3478, +0.0000, -4.2553, +2.1277, -4.2553, +0.0000, +0.0000, +0.0000 |
| S9 − S1_NUMERIC_GLOBAL | ACC | -0.1149 | 1/7/2 | +0.0000, +0.0000, +0.0000, +0.0000, +1.1494, -1.1494, +0.0000, +0.0000, +0.0000, -1.1494 |
| S9 − S1_NUMERIC_GLOBAL | Hard-AUC | -0.1250 | 2/6/2 | +0.0000, +0.0000, +0.0000, +0.0000, +1.0638, -1.2500, +0.3723, +0.0000, +0.0000, -1.4362 |
| S9 − S1_NUMERIC_GLOBAL | SEN-ASD | -0.2500 | — | +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, -2.5000, +5.0000, +0.0000, +0.0000, -5.0000 |
| S9 − S1_NUMERIC_GLOBAL | SPE-TD | +0.0000 | — | +0.0000, +0.0000, +0.0000, +0.0000, +2.1277, +0.0000, -4.2553, +0.0000, +0.0000, +2.1277 |
| S9 − S2_RANDOM_CONCEPTS | ACC | +0.3435 | 5/3/2 | +1.1364, +1.1494, +1.1494, +0.0000, +1.1494, -1.1494, +1.1494, +0.0000, +0.0000, -1.1494 |
| S9 − S2_RANDOM_CONCEPTS | Hard-AUC | +0.3345 | 5/3/2 | +1.0638, +1.0870, +0.8218, +0.0000, +1.2500, -1.2500, +1.6223, +0.0000, +0.0000, -1.2500 |
| S9 − S2_RANDOM_CONCEPTS | SEN-ASD | +0.0122 | — | +0.0000, +0.0000, -4.8780, +0.0000, +2.5000, -2.5000, +7.5000, +0.0000, +0.0000, -2.5000 |
| S9 − S2_RANDOM_CONCEPTS | SPE-TD | +0.6568 | — | +2.1277, +2.1739, +6.5217, +0.0000, +0.0000, +0.0000, -4.2553, +0.0000, +0.0000, +0.0000 |
| S9 − S3_SINGLE_CONCEPT | ACC | +0.2299 | 4/4/2 | +0.0000, +0.0000, +1.1494, +0.0000, +1.1494, -1.1494, +1.1494, +1.1494, +0.0000, -1.1494 |
| S9 − S3_SINGLE_CONCEPT | Hard-AUC | +0.2656 | 5/3/2 | +0.0000, +0.1326, +1.0870, +0.0000, +1.0638, -1.2500, +1.8085, +1.0638, +0.0000, -1.2500 |
| S9 − S3_SINGLE_CONCEPT | SEN-ASD | +0.7439 | — | +0.0000, +2.4390, +0.0000, +0.0000, +0.0000, -2.5000, +10.0000, +0.0000, +0.0000, -2.5000 |
| S9 − S3_SINGLE_CONCEPT | SPE-TD | -0.2128 | — | +0.0000, -2.1739, +2.1739, +0.0000, +2.1277, +0.0000, -6.3830, +2.1277, +0.0000, +0.0000 |
| S9 − S4_CORR_REL_ONLY | ACC | +0.0000 | 3/4/3 | +0.0000, -1.1494, +0.0000, +1.1494, +1.1494, -1.1494, +0.0000, +1.1494, +0.0000, -1.1494 |
| S9 − S4_CORR_REL_ONLY | Hard-AUC | +0.0194 | 4/2/4 | -0.1557, -1.0870, +0.0000, +1.0638, +1.2500, -1.2500, +0.5585, +1.0638, +0.0000, -1.2500 |
| S9 − S4_CORR_REL_ONLY | SEN-ASD | +0.2561 | — | -2.4390, +0.0000, +0.0000, +0.0000, +2.5000, -2.5000, +7.5000, +0.0000, +0.0000, -2.5000 |
| S9 − S4_CORR_REL_ONLY | SPE-TD | -0.2174 | — | +2.1277, -2.1739, +0.0000, +2.1277, +0.0000, +0.0000, -6.3830, +2.1277, +0.0000, +0.0000 |
| S9 − S5_AUX_REL_ONLY | ACC | +0.4585 | 5/3/2 | +1.1364, +0.0000, +1.1494, +0.0000, +1.1494, -1.1494, +0.0000, +2.2989, +1.1494, -1.1494 |
| S9 − S5_AUX_REL_ONLY | Hard-AUC | +0.4302 | 6/2/2 | +1.2195, +0.0000, +0.9544, +0.0000, +1.0638, -1.2500, +0.3723, +2.1277, +1.0638, -1.2500 |
| S9 − S5_AUX_REL_ONLY | SEN-ASD | +0.0000 | — | +2.4390, +0.0000, -2.4390, +0.0000, +0.0000, -2.5000, +5.0000, +0.0000, +0.0000, -2.5000 |
| S9 − S5_AUX_REL_ONLY | SPE-TD | +0.8603 | — | +0.0000, +0.0000, +4.3478, +0.0000, +2.1277, +0.0000, -4.2553, +4.2553, +2.1277, +0.0000 |
| S9 − S6_UNIFORM_RELATION | ACC | +0.5747 | 5/4/1 | +0.0000, +0.0000, +2.2989, +1.1494, +1.1494, -1.1494, +1.1494, +1.1494, +0.0000, +0.0000 |
| S9 − S6_UNIFORM_RELATION | Hard-AUC | +0.5289 | 6/1/3 | -0.1557, +0.1326, +2.3065, +1.0638, +1.2500, -1.2500, +1.0638, +1.0638, +0.0000, -0.1862 |
| S9 − S6_UNIFORM_RELATION | SEN-ASD | -0.0061 | — | -2.4390, +2.4390, +2.4390, +0.0000, +2.5000, -2.5000, +0.0000, +0.0000, +0.0000, -2.5000 |
| S9 − S6_UNIFORM_RELATION | SPE-TD | +1.0638 | — | +2.1277, -2.1739, +2.1739, +2.1277, +0.0000, +0.0000, +2.1277, +2.1277, +0.0000, +2.1277 |
| S9 − S7_DIRECT_MESSAGE | ACC | +0.6897 | 7/2/1 | +0.0000, +1.1494, +1.1494, +1.1494, +1.1494, +0.0000, +1.1494, +1.1494, +1.1494, -1.1494 |
| S9 − S7_DIRECT_MESSAGE | Hard-AUC | +0.7493 | 7/2/1 | +0.0000, +1.4846, +0.8218, +1.2500, +1.2500, +0.0000, +1.6223, +1.0638, +1.2500, -1.2500 |
| S9 − S7_DIRECT_MESSAGE | SEN-ASD | +1.4939 | — | +0.0000, +7.3171, -4.8780, +2.5000, +2.5000, +0.0000, +7.5000, +0.0000, +2.5000, -2.5000 |
| S9 − S7_DIRECT_MESSAGE | SPE-TD | +0.0046 | — | +0.0000, -4.3478, +6.5217, +0.0000, +0.0000, +0.0000, -4.2553, +2.1277, +0.0000, +0.0000 |
| S9 − S8_PATIENT_PERMUTED | ACC | -0.1136 | 3/3/4 | -1.1364, +1.1494, +1.1494, -1.1494, +1.1494, -1.1494, +0.0000, +0.0000, +0.0000, -1.1494 |
| S9 − S8_PATIENT_PERMUTED | Hard-AUC | -0.1148 | 3/2/5 | -1.2195, +1.3521, +1.2195, -1.0638, +1.2500, -1.2500, -0.1862, +0.0000, +0.0000, -1.2500 |
| S9 − S8_PATIENT_PERMUTED | SEN-ASD | -0.0122 | — | -2.4390, +4.8780, +2.4390, +0.0000, +2.5000, -2.5000, -2.5000, +0.0000, +0.0000, -2.5000 |
| S9 − S8_PATIENT_PERMUTED | SPE-TD | -0.2174 | — | +0.0000, -2.1739, +0.0000, -2.1277, +0.0000, +0.0000, +2.1277, +0.0000, +0.0000, +0.0000 |

## B0-A Reproduction

The new S0 versus Experiment 32 and Experiment 33 matches initial trunk hashes on 10/10 folds, pre-construction RNG hashes on 10/10, best epochs on 10/10, and all four metrics on 10/10. No prior checkpoint was read or loaded. Historical Experiment 32 B0-A has ACC=92.0742%, Hard-AUC=91.8546%, SEN-ASD=89.0793%, SPE-TD=94.6300%. The MICCAI paper full-model result is 92.30% and is not the same comparison.

| Fold | E32 ACC | S0 ACC | Δ pp | E32 Hard-AUC | S0 Hard-AUC | Δ pp | E32 SEN | S0 SEN | E32 SPE | S0 SPE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 95.4545 | 95.4545 | +0.0000 | 95.1220 | 95.1220 | +0.0000 | 90.2439 | 90.2439 | 100.0000 | 100.0000 |
| 2 | 88.5057 | 88.5057 | +0.0000 | 88.3351 | 88.3351 | +0.0000 | 85.3659 | 85.3659 | 91.3043 | 91.3043 |
| 3 | 88.5057 | 88.5057 | +0.0000 | 88.7328 | 88.7328 | +0.0000 | 92.6829 | 92.6829 | 84.7826 | 84.7826 |
| 4 | 94.2529 | 94.2529 | +0.0000 | 93.9362 | 93.9362 | +0.0000 | 90.0000 | 90.0000 | 97.8723 | 97.8723 |
| 5 | 94.2529 | 94.2529 | +0.0000 | 94.1223 | 94.1223 | +0.0000 | 92.5000 | 92.5000 | 95.7447 | 95.7447 |
| 6 | 95.4023 | 95.4023 | +0.0000 | 95.3723 | 95.3723 | +0.0000 | 95.0000 | 95.0000 | 95.7447 | 95.7447 |
| 7 | 90.8046 | 90.8046 | +0.0000 | 90.1862 | 90.1862 | +0.0000 | 82.5000 | 82.5000 | 97.8723 | 97.8723 |
| 8 | 89.6552 | 89.6552 | +0.0000 | 89.3085 | 89.3085 | +0.0000 | 85.0000 | 85.0000 | 93.6170 | 93.6170 |
| 9 | 90.8046 | 90.8046 | +0.0000 | 90.5585 | 90.5585 | +0.0000 | 87.5000 | 87.5000 | 93.6170 | 93.6170 |
| 10 | 93.1034 | 93.1034 | +0.0000 | 92.8723 | 92.8723 | +0.0000 | 90.0000 | 90.0000 | 95.7447 | 95.7447 |

## Mechanism Audit

The following fold-averaged diagnostics describe the learned graphs and messages; they are not additional classification metrics. In S0 the four raw subject graphs average 166,448 / 287,937 / 135,361 / 81,194 off-diagonal directed edges, with densities 21.97% / 38.00% / 17.86% / 10.71% in PHENO / ANAT / FUNC / CORR order. These are relations among 871 participants, not brain-region edges. In S9, every supported off-diagonal edge changed weight and no unsupported edge was added (the latter was also checked directly in preflight). The mean relative change in normalized off-diagonal weight mass was 10.02% / 9.99% / 9.56% / 6.54% across the four relations. Thus the CORR-derived semantic similarity also reweights the three auxiliary relation matrices, but this is a numeric intervention, not evidence that those covariates are disease-specific measurements.

S9's concept-similarity standard deviations were 0.1518 / 0.1552 / 0.1641 / 0.1583 for D1-D4, so none collapsed to a constant matrix. S2's matched random supports also remained non-degenerate (0.1397 / 0.1459 / 0.1447 / 0.1520). Mean normalized off-diagonal graph changes for S2 were 9.09% / 8.97% / 8.57% / 6.21%, comparable in scale to S9; the S9-vs-S2 result is not explained by random concepts failing to perturb the graph. The S1 global numeric control also changed normalized graph weight mass (7.98% / 7.90% / 7.45% / 4.89%).

Across S9 folds, mean alpha by CAM target position was 0.3250 / 0.1730 / 0.0218 / -0.2438. Mean learned relation weights in PHENO / ANAT / FUNC / CORR order were 0.2624 / 0.2475 / 0.2545 / 0.2356; weights were close to uniform and all four sources had nonzero pre-alpha, weight-aggregated message norms (0.01268 / 0.01104 / 0.01226 / 0.00651). These coefficients establish that the implementation used all relation branches, but not that each branch added useful predictive information. The actual mean residual-to-input norm by target was only 0.0561% / 0.0268% / 0.0250% / 0.0338%, with a largest observed per-patient/token ratio of 0.7099%, below the 10% cap. The residual was nonzero but small. S4 used only CORR (mean relation weights 0 / 0 / 0 / 1); S5 used only PHENO / ANAT / FUNC (0.3420 / 0.3264 / 0.3316 / 0). S9's ACC ties S4 and Hard-AUC is only +0.0194 pp, so cross-relation complementarity is not demonstrated. S9 beats S7 by +0.6897 pp ACC and +0.7493 pp Hard-AUC, consistent with the registered difference message helping relative to direct conditioned messages in this run.

For S8, the within-site patient mapping was bijective in all ten folds, there were no singleton sites, and 97.78% of patient identities changed on average. Original classifier inputs and GFR inputs stayed fixed. Yet S9 trails S8 by 0.1136 pp ACC and 0.1148 pp Hard-AUC. The correct concept mapping beats the random mapping (S9−S2: +0.3435 pp ACC, +0.3345 pp Hard-AUC), but does not beat the full 256D numeric control S1 (−0.1149 pp ACC, −0.1250 pp Hard-AUC) or the patient-mismatch control S8. This does not establish an independent medical-semantic contribution. Candidate concept supports are literature-grounded measurement ranges, not evidence of patient-specific pathology.

Each worker loaded the CLIP model once and accessed the original cached prompt-feature path once per fold (10 calls); SeRCA added no text-encoder call. The original PEN modules remained constructed, while PEN forward remained bypassed as in B0-A. Across all ten folds, all ten configurations had exactly matching fold-specific pre-construction RNG hashes and original trunk initialization hashes. S0 retained its continuous ten-fold seed-0 stream; S1-S9 restored S0's corresponding fold-entry Python, NumPy, PyTorch CPU and CUDA RNG states before model construction. This pairing does not imply natural cross-fold RNG continuity for S1-S9.

Diagnostic summaries must be interpreted alongside paired classification results. S9 is not better than S0, S1, or S8, and its near-tie with S4 does not support four-relation complementarity. Although S9 beats the random concept and direct-message controls, that partial pattern is insufficient to claim that correct disease semantics are necessary or that cross-modal relations provide complementary predictive information.

## Decision: NO-GO

The frozen method was evaluated as registered. The results do not satisfy every internal criterion required to claim that correct medical concepts guide cross-modal complementary relations.

The internal screen is single-seed and not a significance test. This inherited protocol uses `val=test` for checkpoint selection, preprocessing/RFE before cross-validation and transductive graphs over all 871 participants; its CV results are not unbiased independent-test generalization evidence. A positive screen remains provisional and requires multiseed and independent-site validation.

## Reproducibility and Files

`manifest.json` records protocol, group/GPU assignment, original/upstream source hashes, data and history hashes, concept/prior generation hashes, S0 parity, preflight evidence, worker timing, completion counts and decisions. `integration.patch` shows the isolated runtime-only source edits; `.runtime/source/` holds their exact source copies and a symlink to the original CLIP package. RNG entry snapshots are retained. Epoch checkpoints and derived similarity caches are removed after successful verification to keep the final artifact compact; the preparation code regenerates caches deterministically. The complete artifact is currently staged at `/tmp/SPromptGL-E34-SeRCA-stage-20261022/experiment/experiment34_serca_semantic_relational_alignment`. The `/data` OSSFS mount returned `Input/output error` for both target-directory creation/copy and the guarded handoff append; the original `handoff.md` remained unchanged and its prefix was reverified. The exact append text is retained as `handoff_append.md` in the staged experiment. No root model, data or historical experiment was modified and no Git action was performed.

## Fold Metrics

All 100 folds with ACC, Hard-AUC, SEN-ASD, SPE-TD, confusion counts, best epoch, both hashes and fold-paired deltas are in `fold_metrics.csv`.

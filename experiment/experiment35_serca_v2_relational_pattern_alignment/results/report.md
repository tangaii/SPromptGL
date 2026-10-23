# Experiment 35 — SeRCA-v2: Disease-Semantic Relational Pattern Alignment

Status: **COMPLETE**; 120/120 folds completed. GPU0 60/60 and GPU1 60/60. Wall time 1339.2s (0.37 h); summed worker time 5196.2s.

## Question and Audit

The experiment asks whether frozen disease-domain concepts can define useful patient-level joint-deviation relations beyond ordinary FC similarity, and whether those relations improve messages from multiple GFR sources. Pair rules are a conservative human synthesis from the frozen Experiment 19 prompt wording and cited studies. The retained Experiment 19 package does not include a raw GPT-4 response transcript or exact model version, so no new pair rule is attributed verbatim to GPT-4.

Source audit preserved the local code behavior: four GraphLearn outputs are participant-to-participant relations, thresholded at cosine >0.9 and returned raw; GFR uses `normalize_adj(A+I)` and full 320D concatenated features, with the `i==1` overwrite of `i==0`; original PLN construction and CLIP cache lifecycle stay intact; B0-A bypasses PEN forward. The `simple-2` backward objective remains the classification loss plus mean raw-graph losses; ALoss is not optimized. The CAM `[871,4,36]` positions are mixed latent features, not four original modalities or ROIs. PHENO contains age/site/sex encodings; ANAT6 and FUNC10 are scan-quality features.

The MICCAI 2025 paper's full-model 92.30% is distinct from the local fresh B0-A 92.0742%. Related work already covers concept weighting/token graphs (MMGPL), multimodal graph disentanglement (MGDR), semantic-conditioned brain interactions/pathology constraints (KD-Brain), conditional dependency modeling (InfoBridge), and task-dependent alignment (To Align or Not to Align). This run tests a narrow co-deviation rule and message intervention; those ingredients alone are not claimed as novel.

## Frozen Medical Prior

The two retained undirected rules are D2–D4 and D3–D4. D2–D4 is motivated by ASD coactivation-pattern work finding less frequent midcingulo-insular with central-executive/lateral-frontoparietal configurations. D3–D4 is motivated by thalamocortical/sensory, insular and salience associations with sensory responsivity. These studies motivate hypotheses, not exact feature identities or universal ASD effects. The observed mapped set overlaps are D2–D4=5 features and D3–D4=2; none is interpreted as a direct cross-network FC edge. The feature counts are D1=72, D2=56, D3=45, D4=32; unknown feature identities 195, 225, 232, 236 remain in the B0-A input and are excluded from the semantic branch. Full source URLs, prompt text, limitations, and all feature indices are in `medical_prior.json`.

## Method

For each outer fold, a Ridge(alpha=10) FC reference is fitted only on outer-train TD using PHENO age/site/sex one-hot values. Train TD predictions are 5-fold OOF; train ASD and held-out patients use the full outer-train-TD fit. OOF-TD residual scales set signed z values, clipped to [-3,3]. Each domain has positive and negative deviations above |z|=1. Four pairwise sign products are residualized against the eight first-order scores using outer-train rows only and then train-standardized. FULL uses the eight channels for D2–D4 and D3–D4. E4 uses all six pairs; E5 uses fixed wrong pairs with the same pair count and degree multiset.

The semantic kernel uses a train-only median squared-distance scale. Directed Top-8 recipients connect only to outer-train sources; the added branch row-normalizes independently of the original symmetric GFR normalization. The same GFR raw matrices are reused from the same forward. Each message source uses a lightweight modality-specific Linear+LayerNorm to 36 dimensions. A low-capacity router chooses null or PHENO/ANAT/FUNC/CORR. The full difference is `T_sem@V - T_base@V`. Detached outer-train RMS scaling is capped at 10x when messages are tiny; `0.05*tanh(alpha)` starts at zero and each token residual is capped at 5% of its input norm. E10 direct-message calibration uses the same rule; a same-checkpoint direct counterfactual is also saved for E11.

## Groups and Protocol

All groups use one seed, ten StratifiedKFold splits (`shuffle=True`, `random_state=10`), 871 subjects (ASD=403 label 0, TD=468 label 1), the existing globally preprocessed/RFE-selected 320 inputs, all-node transductive original GFR, `nepoch=450`, `early=200`, `lr=.0038`, `weight_decay=.11`, `dropout=.35`, `n_hidden=18`, `n_head=2`, and the historical `simple-2` loss. `val_idx=test_idx`; strict ACC improvement selects a checkpoint and ties keep the first. Hard-AUC uses original hard one-hot predictions. The four formal metrics are ACC, Hard-AUC, SEN-ASD, and SPE-TD.

| Group | Frozen contrast | GPU |
|---|---|---:|
| E0_B0A | Fresh B0-A baseline | 0 |
| E1_SERCA34_CAL | Experiment34 four single-concept kernels with calibrated E35 messages | 1 |
| E2_NUMERIC_GLOBAL | All-256-FC numeric relation, fixed projection only for router inputs | 0 |
| E3_FIRST_ORDER | D1-D4 signed first-order deviations, no pair products | 1 |
| E4_ALL_PAIR | All six pair interactions without semantic rule selection | 0 |
| E5_RANDOM_RULE | Degree-multiset-matched wrong pair rules | 1 |
| E6_PATIENT_SHUFFLED | Correct rules, train/test/site-constrained patient shuffle | 0 |
| E7_ORIGINAL_EDGES | Correct rules; no new semantic Top-k neighbors | 1 |
| E8_CORR_ONLY | Correct rules; CORR relation source only | 0 |
| E9_AUX_ONLY | Correct rules; PHENO/ANAT/FUNC sources only | 1 |
| E10_DIRECT_MATCHED | Correct rules; direct messages with same train-RMS calibration and cap | 0 |
| E11_SERCA_V2_FULL | Registered SeRCA-v2 full model | 1 |

E0 preserves the natural continuous seed-0 fold stream and stores each pre-construction Python/NumPy/torch CPU/CUDA state. E1-E11 restore the corresponding E0 fold-entry state before constructing each fold. This pairs initialization, not later training trajectories. Every worker has atomic epoch/fold-boundary checkpoints containing both original Adams, model parameters, best state, early-stop state and RNG; completed folds are committed once.

## Four Metrics

Mean ± population SD across folds, percent:

| Group | ACC | Hard-AUC | SEN-ASD | SPE-TD |
|---|---:|---:|---:|---:|
| E0_B0A | 92.0742 ± 2.6000% | 91.8546 ± 2.5815% | 89.0793 ± 3.7283% | 94.6300 ± 4.0457% |
| E1_SERCA34_CAL | 91.5008 ± 2.7843% | 91.2969 ± 2.7630% | 88.5976 ± 3.6938% | 93.9963 ± 3.9613% |
| E2_NUMERIC_GLOBAL | 91.6144 ± 3.0498% | 91.3793 ± 3.0553% | 88.3415 ± 4.3028% | 94.4172 ± 4.0275% |
| E3_FIRST_ORDER | 91.8443 ± 3.0233% | 91.6691 ± 2.9546% | 89.5732 ± 3.8439% | 93.7650 ± 5.1908% |
| E4_ALL_PAIR | 91.6144 ± 2.9170% | 91.5091 ± 2.8368% | 90.3171 ± 3.2605% | 92.7012 ± 4.9769% |
| E5_RANDOM_RULE | 91.7307 ± 2.8586% | 91.5492 ± 2.8742% | 89.0976 ± 3.4981% | 94.0009 ± 2.8784% |
| E6_PATIENT_SHUFFLED | 91.6144 ± 3.0062% | 91.4484 ± 2.9532% | 89.3354 ± 3.3396% | 93.5615 ± 4.4468% |
| E7_ORIGINAL_EDGES | 91.7294 ± 3.1245% | 91.5362 ± 3.0900% | 89.0854 ± 3.7260% | 93.9870 ± 4.4272% |
| E8_CORR_ONLY | 91.4982 ± 3.4617% | 91.2783 ± 3.4655% | 88.3476 ± 4.4039% | 94.2091 ± 3.9922% |
| E9_AUX_ONLY | 91.6131 ± 3.0927% | 91.4026 ± 3.1096% | 88.5915 ± 4.1720% | 94.2137 ± 3.6134% |
| E10_DIRECT_MATCHED | 92.0742 ± 2.8427% | 91.8763 ± 2.7870% | 89.3354 ± 3.3306% | 94.4172 ± 4.3608% |
| E11_SERCA_V2_FULL | 91.2696 ± 2.9692% | 91.0235 ± 3.0337% | 87.6159 ± 4.8988% | 94.4311 ± 3.0759% |

`fold_metrics.csv` contains all 120 folds, confusion counts, best epochs, initialization/RNG hashes and E0 deltas.

## Paired Comparisons

Each result is E11 minus its same-fold control; deltas are percentage points. W/T/L are fold wins/ties/losses for ACC and Hard-AUC. These ten overlapping CV fits are descriptive, not independent samples.

| Control | Metric | Mean Δ pp | W/T/L | Fold deltas 1–10 (pp) |
|---|---|---:|---:|---|
| E11 − E0_B0A | ACC | -0.8046 | 0/5/5 | +0.0000, -2.2989, -1.1494, -1.1494, -2.2989, +0.0000, +0.0000, +0.0000, -1.1494, +0.0000 |
| E11 − E0_B0A | Hard-AUC | -0.8312 | 0/5/5 | +0.0000, -2.4390, -1.6172, -1.0638, -1.9415, +0.0000, +0.0000, +0.0000, -1.2500, +0.0000 |
| E11 − E0_B0A | SEN-ASD | -1.4634 | — | +0.0000, -4.8780, -9.7561, +0.0000, +2.5000, +0.0000, +0.0000, +0.0000, -2.5000, +0.0000 |
| E11 − E0_B0A | SPE-TD | -0.1989 | — | +0.0000, +0.0000, +6.5217, -2.1277, -6.3830, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000 |
| E11 − E1_SERCA34_CAL | ACC | -0.2312 | 2/5/3 | +1.1364, -2.2989, +0.0000, +0.0000, +0.0000, -1.1494, -1.1494, +1.1494, +0.0000, +0.0000 |
| E11 − E1_SERCA34_CAL | Hard-AUC | -0.2735 | 2/4/4 | +1.2195, -2.4390, -0.2651, +0.0000, +0.0000, -1.0638, -1.2500, +1.0638, +0.0000, +0.0000 |
| E11 − E1_SERCA34_CAL | SEN-ASD | -0.9817 | — | +2.4390, -4.8780, -4.8780, +0.0000, +0.0000, +0.0000, -2.5000, +0.0000, +0.0000, +0.0000 |
| E11 − E1_SERCA34_CAL | SPE-TD | +0.4348 | — | +0.0000, +0.0000, +4.3478, +0.0000, +0.0000, -2.1277, +0.0000, +2.1277, +0.0000, +0.0000 |
| E11 − E2_NUMERIC_GLOBAL | ACC | -0.3448 | 1/6/3 | +0.0000, -2.2989, +1.1494, +0.0000, -1.1494, -1.1494, +0.0000, +0.0000, +0.0000, +0.0000 |
| E11 − E2_NUMERIC_GLOBAL | Hard-AUC | -0.3559 | 2/5/3 | +0.0000, -2.7041, +1.0870, +0.1862, -1.0638, -1.0638, +0.0000, +0.0000, +0.0000, +0.0000 |
| E11 − E2_NUMERIC_GLOBAL | SEN-ASD | -0.7256 | — | +0.0000, -9.7561, +0.0000, +2.5000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000 |
| E11 − E2_NUMERIC_GLOBAL | SPE-TD | +0.0139 | — | +0.0000, +4.3478, +2.1739, -2.1277, -2.1277, -2.1277, +0.0000, +0.0000, +0.0000, +0.0000 |
| E11 − E3_FIRST_ORDER | ACC | -0.5747 | 0/5/5 | +0.0000, -1.1494, +0.0000, -1.1494, -1.1494, -1.1494, +0.0000, +0.0000, -1.1494, +0.0000 |
| E11 − E3_FIRST_ORDER | Hard-AUC | -0.6456 | 0/4/6 | +0.0000, -1.4846, -0.5302, -1.0638, -1.0638, -1.0638, +0.0000, +0.0000, -1.2500, +0.0000 |
| E11 − E3_FIRST_ORDER | SEN-ASD | -1.9573 | — | +0.0000, -7.3171, -9.7561, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, -2.5000, +0.0000 |
| E11 − E3_FIRST_ORDER | SPE-TD | +0.6660 | — | +0.0000, +4.3478, +8.6957, -2.1277, -2.1277, -2.1277, +0.0000, +0.0000, +0.0000, +0.0000 |
| E11 − E4_ALL_PAIR | ACC | -0.3448 | 1/6/3 | +0.0000, -2.2989, +0.0000, +0.0000, +0.0000, -1.1494, -1.1494, +1.1494, +0.0000, +0.0000 |
| E11 − E4_ALL_PAIR | Hard-AUC | -0.4857 | 1/5/4 | +0.0000, -2.7041, -0.5302, +0.0000, +0.0000, -1.0638, -1.6223, +1.0638, +0.0000, +0.0000 |
| E11 − E4_ALL_PAIR | SEN-ASD | -2.7012 | — | +0.0000, -9.7561, -9.7561, +0.0000, +0.0000, +0.0000, -7.5000, +0.0000, +0.0000, +0.0000 |
| E11 − E4_ALL_PAIR | SPE-TD | +1.7299 | — | +0.0000, +4.3478, +8.6957, +0.0000, +0.0000, -2.1277, +4.2553, +2.1277, +0.0000, +0.0000 |
| E11 − E5_RANDOM_RULE | ACC | -0.4611 | 2/4/4 | +1.1364, -2.2989, +0.0000, -1.1494, -2.2989, -1.1494, +0.0000, +0.0000, +0.0000, +1.1494 |
| E11 − E5_RANDOM_RULE | Hard-AUC | -0.5258 | 2/1/7 | +1.0638, -2.4390, -0.1326, -1.0638, -2.1277, -1.0638, -0.3723, +0.0000, -0.1862, +1.0638 |
| E11 − E5_RANDOM_RULE | SEN-ASD | -1.4817 | — | +0.0000, -4.8780, -2.4390, +0.0000, +0.0000, +0.0000, -5.0000, +0.0000, -2.5000, +0.0000 |
| E11 − E5_RANDOM_RULE | SPE-TD | +0.4302 | — | +2.1277, +0.0000, +2.1739, -2.1277, -4.2553, -2.1277, +4.2553, +0.0000, +2.1277, +2.1277 |
| E11 − E6_PATIENT_SHUFFLED | ACC | -0.3448 | 2/4/4 | +0.0000, -2.2989, +1.1494, -1.1494, +0.0000, -1.1494, +0.0000, +0.0000, -1.1494, +1.1494 |
| E11 − E6_PATIENT_SHUFFLED | Hard-AUC | -0.4250 | 2/3/5 | +0.0000, -2.5716, +0.8218, -1.0638, +0.0000, -1.0638, -0.1862, +0.0000, -1.2500, +1.0638 |
| E11 − E6_PATIENT_SHUFFLED | SEN-ASD | -1.7195 | — | +0.0000, -7.3171, -4.8780, +0.0000, +0.0000, +0.0000, -2.5000, +0.0000, -2.5000, +0.0000 |
| E11 − E6_PATIENT_SHUFFLED | SPE-TD | +0.8696 | — | +0.0000, +2.1739, +6.5217, -2.1277, +0.0000, -2.1277, +2.1277, +0.0000, +0.0000, +2.1277 |
| E11 − E7_ORIGINAL_EDGES | ACC | -0.4598 | 1/4/5 | +0.0000, -1.1494, +0.0000, -1.1494, -1.1494, -1.1494, +0.0000, +1.1494, -1.1494, +0.0000 |
| E11 − E7_ORIGINAL_EDGES | Hard-AUC | -0.5127 | 1/3/6 | +0.0000, -1.4846, -0.2651, -1.0638, -1.0638, -1.0638, +0.0000, +1.0638, -1.2500, +0.0000 |
| E11 − E7_ORIGINAL_EDGES | SEN-ASD | -1.4695 | — | +0.0000, -7.3171, -4.8780, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, -2.5000, +0.0000 |
| E11 − E7_ORIGINAL_EDGES | SPE-TD | +0.4440 | — | +0.0000, +4.3478, +4.3478, -2.1277, -2.1277, -2.1277, +0.0000, +2.1277, +0.0000, +0.0000 |
| E11 − E8_CORR_ONLY | ACC | -0.2286 | 1/6/3 | -1.1364, +0.0000, +0.0000, -1.1494, +0.0000, -1.1494, +0.0000, +1.1494, +0.0000, +0.0000 |
| E11 − E8_CORR_ONLY | Hard-AUC | -0.2548 | 2/3/5 | -1.2195, +0.0000, -0.2651, -1.0638, +0.1862, -1.0638, +0.0000, +1.0638, -0.1862, +0.0000 |
| E11 − E8_CORR_ONLY | SEN-ASD | -0.7317 | — | -2.4390, +0.0000, -4.8780, +0.0000, +2.5000, +0.0000, +0.0000, +0.0000, -2.5000, +0.0000 |
| E11 − E8_CORR_ONLY | SPE-TD | +0.2220 | — | +0.0000, +0.0000, +4.3478, -2.1277, -2.1277, -2.1277, +0.0000, +2.1277, +2.1277, +0.0000 |
| E11 − E9_AUX_ONLY | ACC | -0.3435 | 2/4/4 | -1.1364, -2.2989, +0.0000, +0.0000, -1.1494, -1.1494, +0.0000, +1.1494, +0.0000, +1.1494 |
| E11 − E9_AUX_ONLY | Hard-AUC | -0.3791 | 3/1/6 | -1.2195, -2.4390, -0.1326, +0.1862, -1.0638, -1.0638, +0.0000, +1.0638, -0.1862, +1.0638 |
| E11 − E9_AUX_ONLY | SEN-ASD | -0.9756 | — | -2.4390, -4.8780, -2.4390, +2.5000, +0.0000, +0.0000, +0.0000, +0.0000, -2.5000, +0.0000 |
| E11 − E9_AUX_ONLY | SPE-TD | +0.2174 | — | +0.0000, +0.0000, +2.1739, -2.1277, -2.1277, -2.1277, +0.0000, +2.1277, +2.1277, +2.1277 |
| E11 − E10_DIRECT_MATCHED | ACC | -0.8046 | 0/3/7 | +0.0000, -1.1494, -1.1494, -1.1494, -1.1494, -1.1494, -1.1494, +0.0000, -1.1494, +0.0000 |
| E11 − E10_DIRECT_MATCHED | Hard-AUC | -0.8528 | 0/3/7 | +0.0000, -1.6172, -1.2195, -0.8777, -1.0638, -1.0638, -1.4362, +0.0000, -1.2500, +0.0000 |
| E11 − E10_DIRECT_MATCHED | SEN-ASD | -1.7195 | — | +0.0000, -9.7561, -2.4390, +2.5000, +0.0000, +0.0000, -5.0000, +0.0000, -2.5000, +0.0000 |
| E11 − E10_DIRECT_MATCHED | SPE-TD | +0.0139 | — | +0.0000, +6.5217, +0.0000, -4.2553, -2.1277, -2.1277, +2.1277, +0.0000, +0.0000, +0.0000 |

## B0-A and Mechanisms

E0 matches Experiments 32, 33 and staged Experiment 34 on initial trunk hash 10/10, pre-construction RNG 10/10, best epoch 10/10, and all four metrics 10/10. It reads no historical checkpoint.
Across 871 held-out predictions, E0 has 802 correct (ASD FN=44, TD FP=25); E11 has 795 correct (ASD FN=50, TD FP=26), a net change of -7 correct, +6 ASD FN and +1 TD FP. E10 has 802 correct (ASD FN=43, TD FP=26).

For E11, mean alpha by CAM position is `[-0.07911633849143981, -0.01849124561995268, -0.03343123942613602, -0.25269088298082354]`; mean route weights (null, PHENO, ANAT, FUNC, CORR) are `[0.09587747287005186, 0.16155363097786904, 0.13389363512396812, 0.1848699316382408, 0.42380531430244445]`; mean source contribution RMS is `[0.0030378301627933977, 0.011054859333671629, 0.013535364530980587, 0.05198486156295985]`. The per-token vector `[0.9954071044921875, 0.9980442225933075, 0.9978757083415986, 0.9866139650344848]` contains mean residual-cap multipliers, not residual/input ratios. The mean across folds of the per-fold maximum actual residual/input norm ratio is 4.6862%, and the worst observed value is 5.0000012% (floating-point tolerance at the 5% cap). Same-checkpoint direct-message predictions changed on 0.57% of test subjects on average after matching train RMS and using the same 5% cap; calibrated direct RMS was 0.450437. Per-source same-checkpoint ablations are summarized here: `{"PHENO": {"changed_fraction": 0.0, "acc": 0.9126959247648904, "hard_auc": 0.910234679948557}, "ANAT": {"changed_fraction": 0.0, "acc": 0.9126959247648904, "hard_auc": 0.910234679948557}, "FUNC": {"changed_fraction": 0.0, "acc": 0.9126959247648904, "hard_auc": 0.910234679948557}, "CORR": {"changed_fraction": 0.0, "acc": 0.9126959247648904, "hard_auc": 0.910234679948557}}`. Full graph overlap, per-fold scales, station similarity, message contributions, and route details are in `diagnostics.csv`.

A nonzero gate or residual is not proof of predictive complementarity. The decisive evidence is E11 versus E2/E4/E5/E6 for semantic necessity, E11 versus E8/E9 for source complementarity, E11 versus E10 for difference messaging, and the fixed-checkpoint source ablations. PHENO is demographics, ANAT/FUNC are QC; neither is an independent disease imaging modality.

## Decision: NO-GO

The frozen method was evaluated as registered. The results do not satisfy all internal criteria for the claimed semantic and cross-source mechanism.

This is a single-seed internal screen, not a significance test or SOTA claim. The inherited protocol selects on `val=test`, performs RFE/preprocessing before outer CV, and uses transductive graphs; it cannot establish unbiased independent-site generalization.

## Reproducibility and Delivery

`manifest.json` records source/data/history hashes, frozen configuration, priors, caches, RNG pairing, per-worker timing, completion and decision. `integration.patch` and `.runtime/source/` record the isolated source edits. Fold prior caches and epoch checkpoints are removed only after successful aggregation; deterministic code and retained RNG-entry snapshots regenerate the inputs. The intended final `/data` path and root `handoff.md` append status are recorded in the manifest; the working artifact is staged under `/tmp` because the OSSFS experiment directory returned I/O error.

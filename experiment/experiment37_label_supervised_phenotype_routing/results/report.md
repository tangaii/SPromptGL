# Experiment 37 — LSPCR

Status: **COMPLETE**; 140/140 folds. GPU0=70, GPU1=70; dispatch wall time=2045.3s.

## Audit and Design

SPromptGL already defines a label-supervised adjacency in the paper, so LSPCR does not claim first use of labels in graph learning. In the repository's `simple-2` training path, `ALoss` is computed but excluded; the optimized historical objective is classification plus the mean four GraphConstruct losses. `GraphLearn` uses projected cosine with a detached `>0.9` threshold. The original GFR PHENO branch is overwritten at its second branch and was not changed.

Experiment 36's T0/B1 four-graph trunk is not the original B0: it uses shared two-layer GCN parameters, four separately propagated full-320D messages, learned softmax message fusion, and row-stochastic normalization. E36 T0 changed some supported edges but did not beat the original B0 on ACC, fixed-checkpoint text-off predictions did not change class, and fusion weights remained close to 0.25. E36's single graph became too dense and performed poorly, so E37 keeps the four-graph symmetric path. Its prior text audit found PHENO48 mainly age-bin/sex/site; the text CSV additionally has real age, FIQ/VIQ/PIQ and handedness. ANAT/FUNC are QC/motion fields; CORR descriptions are feature-slot blocks, not verified ROI names. Dense BioLORD versus zero-padded numeric conditions are not a capacity-matched language comparison.

PhenoBrain conditions ROI-level communication within each participant. LSPCR instead supervises patient-to-patient Q/K scores. The model never uses a label to construct the inference graph: labels form soft train-train targets during training only; held-out labels are excluded from every pair sampler and auxiliary loss. G12 directly supervises only existing thresholded GraphLearn train-train edges, exposing the detached hard-threshold limitation.

## Method

For receiver i and sender j, the score used by PCPR is `e(i<-j)=q(z_i)^T k(z_j)/sqrt(64)`. Supported non-self scores are row-centered and symmetrized, then existing edge weights are multiplied by `exp(beta*tanh(e_centered))`; `beta` is bounded by 0.25 and initialized at 0.10. Missing edges remain zero, the diagonal stays unchanged, and row-stochastic propagation carries the full 320D sender feature. Q/K/beta use a separate GFR-Adam group with zero weight decay; historical GFR/PLN parameters retain 0.11.

SUPPORT-BCE uses supported train-train pairs; ALL-PAIR-BCE supervises all train-train non-self pairs without adding edges. Each class has a 0.9/0.1 soft target; positive and negative BCE means are averaged equally. Ranking uses up to four lowest-scoring same-label and four highest-scoring different-label train peers per anchor, with margin 0.5. G13 averages class-balanced losses within sites that contain both pair types. G10 shuffles only auxiliary train labels within site; classification retains true training labels. G12 supervises raw GraphLearn probabilities only where the existing thresholded graph has support. All objectives add `lambda_relation * Lrelation` to unchanged classification and graph losses; original ALoss remains excluded.

HYBRID embeddings are composed exactly as E36 `modal_full[:,0]` for PHENO and `modal_only[:,1:4]` for ANAT/FUNC/CORR. The frozen encoder revision, pooling, and L2 normalization are unchanged. Inference is phenotype-conditioned and label-free. The inherited `val=test` selection, pre-CV feature selection/preprocessing, and transductive graph protocol remain optimistic and do not establish independent generalization.

## E36 Replay

G00 replay: **MATCH**; checks `{"acc": true, "asd_fn": true, "asd_tp": true, "best_epoch": true, "hard_auc": true, "initial_parameter_sha256": true, "initial_rng_sha256": true, "sen_asd": true, "spe_td": true, "td_fp": true, "td_tn": true}`.

## Formal Metrics

Mean ± population SD over ten folds, percent. AUC is the original hard-label protocol, not probability ROC-AUC.

|Group|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|G00_T0_REPLAY|92.0742 ± 2.4963|91.9423 ± 2.4248|93.5615 ± 4.2383|90.3232 ± 2.8246|
|G01_HYBRID_BASE|91.4995 ± 3.0994|91.3474 ± 3.0600|93.3534 ± 4.5766|89.3415 ± 3.9788|
|G02_FULL_SUPPORT_BCE|91.6144 ± 2.9170|91.4033 ± 2.8817|94.2091 ± 4.1041|88.5976 ± 3.6775|
|G03_HYBRID_SUPPORT_BCE|91.6144 ± 2.9170|91.3980 ± 2.9011|94.2044 ± 4.1193|88.5915 ± 4.0194|
|G04_FULL_ALL_BCE|91.7294 ± 2.9950|91.5120 ± 2.9877|94.4265 ± 3.7570|88.5976 ± 3.8515|
|G05_HYBRID_ALL_BCE|91.2696 ± 3.0569|91.0446 ± 3.0914|93.9917 ± 4.3141|88.0976 ± 5.0618|
|G06_HYBRID_RANK|91.7294 ± 2.9505|91.5199 ± 2.9177|94.2044 ± 4.3427|88.8354 ± 3.7320|
|G07_HYBRID_ALL_BCE_LOW|91.6144 ± 2.9170|91.4826 ± 2.8498|93.1360 ± 4.7309|89.8293 ± 3.4050|
|G08_HYBRID_ALL_BCE_HIGH|91.3845 ± 3.2270|91.1191 ± 3.2356|94.6346 ± 4.2390|87.6037 ± 4.6946|
|G09_HYBRID_TEXT_SHUFFLE|91.4995 ± 3.1835|91.3048 ± 3.1424|93.7743 ± 4.2842|88.8354 ± 3.3805|
|G10_HYBRID_AUX_LABEL_SHUFFLE|91.6144 ± 3.0498|91.4219 ± 3.0711|93.9963 ± 3.7257|88.8476 ± 4.3125|
|G11_HYBRID_NUMERIC_BCE|91.6144 ± 2.9619|91.4987 ± 2.9836|93.1499 ± 3.8147|89.8476 ± 4.4613|
|G12_RAW_GRAPH_LABEL_SUP|91.4995 ± 2.5884|91.3094 ± 2.5605|93.7835 ± 3.6749|88.8354 ± 3.3716|
|G13_HYBRID_SITE_BALANCED|91.4995 ± 3.0994|91.2459 ± 3.0572|94.4126 ± 4.4777|88.0793 ± 3.5256|

## Paired Results

Delta is left minus right in percentage points. W/T/L use the ten matched folds for ACC and Hard-AUC; folds overlap in training data and are not independent datasets.

|Left − right|ΔACC|ACC W/T/L|ΔHard-AUC|Hard-AUC W/T/L|ΔSPE|ΔSEN|
|---|---:|---:|---:|---:|---:|---:|
|G01_HYBRID_BASE − G00_T0_REPLAY|-0.5747|1/3/6|-0.5949|1/2/7|-0.2081|-0.9817|
|G02_FULL_SUPPORT_BCE − G00_T0_REPLAY|-0.4598|1/4/5|-0.5390|1/3/6|+0.6475|-1.7256|
|G03_HYBRID_SUPPORT_BCE − G01_HYBRID_BASE|+0.1149|3/4/3|+0.0505|4/2/4|+0.8511|-0.7500|
|G04_FULL_ALL_BCE − G02_FULL_SUPPORT_BCE|+0.1149|3/5/2|+0.1087|4/4/2|+0.2174|-0.0000|
|G05_HYBRID_ALL_BCE − G03_HYBRID_SUPPORT_BCE|-0.3448|0/7/3|-0.3533|2/4/4|-0.2128|-0.4939|
|G05_HYBRID_ALL_BCE − G04_FULL_ALL_BCE|-0.4598|1/4/5|-0.4674|1/4/5|-0.4348|-0.5000|
|G05_HYBRID_ALL_BCE − G01_HYBRID_BASE|-0.2299|2/4/4|-0.3028|2/4/4|+0.6383|-1.2439|
|G06_HYBRID_RANK − G05_HYBRID_ALL_BCE|+0.4598|5/4/1|+0.4753|5/4/1|+0.2128|+0.7378|
|G07_HYBRID_ALL_BCE_LOW − G05_HYBRID_ALL_BCE|+0.3448|5/2/3|+0.4380|5/2/3|-0.8557|+1.7317|
|G08_HYBRID_ALL_BCE_HIGH − G05_HYBRID_ALL_BCE|+0.1149|2/7/1|+0.0745|2/6/2|+0.6429|-0.4939|
|G05_HYBRID_ALL_BCE − G09_HYBRID_TEXT_SHUFFLE|-0.2299|2/4/4|-0.2602|4/2/4|+0.2174|-0.7378|
|G05_HYBRID_ALL_BCE − G10_HYBRID_AUX_LABEL_SHUFFLE|-0.3448|1/5/4|-0.3773|3/1/6|-0.0046|-0.7500|
|G05_HYBRID_ALL_BCE − G11_HYBRID_NUMERIC_BCE|-0.3448|1/5/4|-0.4541|2/2/6|+0.8418|-1.7500|
|G05_HYBRID_ALL_BCE − G12_RAW_GRAPH_LABEL_SUP|-0.2299|1/6/3|-0.2648|2/4/4|+0.2081|-0.7378|
|G05_HYBRID_ALL_BCE − G13_HYBRID_SITE_BALANCED|-0.2299|2/4/4|-0.2013|2/3/5|-0.4209|+0.0183|
|G00_T0_REPLAY − E36_B0_ORIGINAL_B0A|+0.0000|1/8/1|+0.0877|3/5/2|-1.0685|+1.2439|
|G00_T0_REPLAY − E36_B1_FOUR_GRAPH_BASE|+0.8046|7/2/1|+0.8605|8/1/1|-0.0046|+1.7256|
|G00_T0_REPLAY − E36_T0_FOUR_SYM_TEXT|+0.0000|0/10/0|-0.0000|0/10/0|+0.0000|-0.0000|
|G06_HYBRID_RANK − E36_B0_ORIGINAL_B0A|-0.3448|2/3/5|-0.3347|2/3/5|-0.4255|-0.2439|
|G06_HYBRID_RANK − E36_B1_FOUR_GRAPH_BASE|+0.4598|4/6/0|+0.4381|4/4/2|+0.6383|+0.2378|
|G06_HYBRID_RANK − E36_T0_FOUR_SYM_TEXT|-0.3448|1/5/4|-0.4224|1/2/7|+0.6429|-1.4878|
|G05_HYBRID_ALL_BCE − E36_B0_ORIGINAL_B0A|-0.8046|1/4/5|-0.8100|2/3/5|-0.6383|-0.9817|
|G05_HYBRID_ALL_BCE − E36_B1_FOUR_GRAPH_BASE|+0.0000|2/6/2|-0.0372|2/5/3|+0.4255|-0.5000|
|G05_HYBRID_ALL_BCE − E36_T0_FOUR_SYM_TEXT|-0.8046|0/5/5|-0.8977|1/2/7|+0.4302|-2.2256|

## Mechanism and Decision

A separate reproducible one-step replay using the saved fold-entry RNG audited all 120 supervised group/fold combinations. G05 had positive first-step auxiliary Q/K gradient norms in 10/10 folds (mean query=1.29474e-05, key=1.28346e-05); training also updated Q/K. This audit used disposable model instances and did not change the formal checkpoints or metrics. G12's raw GraphLearn projection gradients were nonzero on existing edges (mean PHENO/ANAT/FUNC/CORR norms: `[0.004808812006376684, 0.20991893410682677, 0.18615943491458892, 0.3541252315044403]`); the detached `>0.9` mask still prevents absent patient edges from receiving direct auxiliary supervision.

For G05, fixed-checkpoint replacement of real text by the deterministic within-pool mismatch changed no held-out class (0.0000 flip fraction) and changed logits by only 5.58e-07 on average. Turning PCPR off also changed no held-out predicted class. The relation score was trained, but its held-out prediction effect was negligible. Mean fusion weights remained `[0.24995601773262024, 0.2500689193606377, 0.24998040944337846, 0.24999464452266693]`, close to equal routing.

|G05 source|beta|same / different edge multiplier|difference|QK-score vs text-cosine Spearman|BioLORD cosine mean ± pair SD|mean absolute raw-edge change|
|---|---:|---:|---:|---:|---:|---:|
|PHENO|0.1030|1.006498 / 0.994109|0.012389|0.0657|0.7398 ± 0.1265|0.017870|
|ANAT|0.1030|1.000124 / 0.999844|0.000280|0.0701|0.9768 ± 0.0073|0.004078|
|FUNC|0.1030|1.000075 / 0.999924|0.000150|0.1391|0.9944 ± 0.0028|0.001809|
|CORR|0.1030|1.000010 / 1.000000|0.000010|0.0297|0.9980 ± 0.0010|0.000939|

These values are measured on outer-train supported edges. The main G05 auxiliary loss remained near its uninformative BCE level and same/different multiplier separation was concentrated in PHENO; the ANAT/FUNC/CORR changes were very small. No new graph edges were added.

The held-out-label invariance check ran for all 140 formal folds and requires bitwise-identical inference logits after replacing held-out labels. Auxiliary pair construction asserts zero test-pair participation. G09 shuffled an average of 828.6/871 text assignments within train/test pools; G10 permuted 47.9% of train auxiliary-label positions within site while preserving each site's label counts.

Execution recovery: the first SUPPORT-BCE attempt encountered a thresholded modality with no available same/different supported pair; the absent-class term was set to zero and the fold resumed from its atomic checkpoint. A later G09 startup assertion incorrectly assumed unshuffled text identity; it was corrected to validate the fixed within-pool permutation and G09 then completed all ten folds. Neither failed startup produced a duplicate result row; the final table is keyed uniquely by group/seed/fold.

Decision: **NO-GO**. Pre-registered internal checks: `{"G05_beats_HYBRID_base_and_both_mismatch_controls": false, "G05_vs_G00_at_least_0.35pp_acc_and_hard_auc": false, "QK_auxiliary_gradient_reaches_router_10_of_10": true, "QK_parameters_updated_during_training": true, "formal_significance_claim": false, "real_text_changes_heldout_predictions_or_logits_materially": false}`. This is a single-seed exploratory decision, not evidence of statistical significance, clinical validity, or SOTA.

## Reproducibility

`manifest.json` records source/data/cache/config/code/result hashes, fold assignment, RNG pairing, group/GPU assignment, task completion and delivery state. `fold_metrics.csv` contains 140 unique group/seed/fold rows. `initial_aux_gradient_audit.csv` stores the separate 120-row gradient replay, `text_cosine_audit.csv` records real BioLORD cosine distributions over outer-train pairs, and `diagnostics.csv` joins these audits to per-fold training and inference diagnostics. `.runtime/` contains the validated immutable E36 text caches, fold-entry RNG states, atomic task checkpoints and logs. No root model, data, or historical experiment was edited.

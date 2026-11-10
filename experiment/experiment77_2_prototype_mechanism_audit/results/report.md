# Experiment 77.2: PIM-Audit

**Execution: COMPLETE_READ_ONLY. Decision: NO_GO_PROTOTYPE.**
No E74.4/E77/E77.1 model, optimizer, checkpoint, prompt or memory was trained, updated or reinitialized.

## Provenance and exact computation

Frozen E77.1-G2 source: `/data/adodas/SPromptGL-main/experiment/experiment77_1_sc_perm`; E74.4-G2 source: `/model/spromptgl/experiment/experiment74_4_soft_positive_dcr`. Ten E77.1 and ten E74.4 best checkpoints were SHA-locked before analysis. See `manifest.json`.
E77.1-G2 is `dugsa.PLNNoPEN`, with no active PEN. Raw CORR is `state.parts[3]` `[871,256]`; CAM's corresponding CORR latent block is captured from the CAM-to-OutputLayer `[871,144]` input slice `108:144` `[871,36]`. This is a shared patient/modality provenance, not an invertible or anatomically aligned representation.
The exact hard evidence path is `sim = Top2Softmax(feature cosine + semantic relation)`, `contrast = sim - competing_typical/1.35 - 0.05`, `gate = sigmoid(contrast/0.2)*valid`, `value = 0.5*tanh(2*relu(contrast))*gate`; weighted modality evidence is summed by target class, clamped to `[0,0.5]`, and enters the bounded ASD-vs-TD log-probability correction. The read-only audit restored class/hard usage counters after forward and verified RNG, model/CAM state and memory state hashes were unchanged.
Checkpoint hard state stores current patient IDs and first/last stable admission epoch, update frequency, current-window votes, confidence, novelty and prototype assignments. These times are first/last **stable-pool admissions**, not first/last single mistakes. `hard_diagnostics` preserves aggregate counts per epoch, but not epoch-by-epoch patient membership or the complete prediction sequence. Duration and recurrence below are therefore bounded summaries of stored state, not reconstructed trajectories.

## Q1: Hard-memory stability

FP is true TD (label 1) predicted ASD (0); FN is true ASD (0) predicted TD (1). Across fold-specific G2 banks, the current pool contains 577 FP and 639 FN records (fold sums, not globally unique people), representing 169 unique global FP IDs and 198 unique global FN IDs. Mean per-fold pool sizes are 57.7 FP and 63.9 FN. Current best-checkpoint model predictions still err on 70/577 FP and 161/639 FN pool records; the rest are currently corrected.
Stored stable-admission spans have median 41.0 epochs (FP) and 5.0 epochs (FN); update-frequency medians are 18.0 and 4.0. These are sparse admissions, not proof of continuous error. 114/169 FP IDs and 126/198 FN IDs recur in multiple folds. Fold-level same-direction Jaccard is 0.3128 for FP and 0.3329 for FN, versus matched-label/training-overlap random means 0.0549 and 0.0726; observed/random ratios are 5.69x/4.58x. 45/45 FP and 45/45 FN fold pairs exceed the matched random upper-tail threshold. Repeated subjects across folds are not independent external replications.
With hard evidence disabled at the same checkpoints, test-fold wrong-to-correct versus correct-to-wrong changes sum to 1 versus 0; on stored FP/FN training-pool records the directionally corrected count is 2 versus 4. Mean true-class test-margin change (full minus hard-off) is 0.04396.

## Q2: Conditional increment

Zero-fit frozen scores are reported as ROC-AUC and class-conditional score distributions; prototype similarity is not presented as a calibrated probability. The CPU probes are the single predeclared L2 logistic fit per fold, C=1, with train-only scaling and test-overlap IDs removed. Prototype feature scores for training probes use the implementation's leave-one-subject-out centers.
P1−P0 (CORR256 + evidence vs CORR256) mean paired ΔBA/ΔAUC/log-loss is +3.248/+0.019 pp/+1.67120; fold wins for BA/AUC are 8/10 and 5/10. P3−P2 (CAM144 + evidence vs CAM144) is +0.959/-0.665 pp/+0.00629; fold wins are 6/10 and 2/10. P4 margin augmentation mean ΔAUC/BA is -0.389/+0.471 pp.

| Frozen probe | ACC | BA | ROC-AUC | SEN | SPE | Log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 CORR256 | 85.874 | 85.748 | 91.645 | 84.128 | 87.368 | 4.6224 | 0.1173 |
| P1 CORR256 + evidence | 89.091 | 88.996 | 91.664 | 87.835 | 90.157 | 6.2936 | 0.1003 |
| P2 CAM144 | 88.408 | 88.309 | 92.625 | 87.104 | 89.514 | 6.0771 | 0.1025 |
| P3 CAM144 + evidence | 89.327 | 89.268 | 91.961 | 88.591 | 89.944 | 6.0834 | 0.0976 |
| P4 raw margin | 90.121 | 90.012 | 93.776 | 88.585 | 91.438 | 3.7566 | 0.0855 |
| P4 margin + evidence | 90.580 | 90.483 | 93.387 | 89.323 | 91.642 | 4.0503 | 0.0847 |

| Frozen zero-fit score | ROC-AUC mean ± SD (%) |
|---|---:|
| E74.4_frozen_margin | 93.495 ± 2.919 |
| E77.1_pre_correction_margin | 93.776 ± 2.760 |
| E77.1_full_final_margin | 93.866 ± 2.668 |
| typical_feature_only_difference | 91.697 ± 2.232 |
| typical_semantic_conditioned_difference | 93.083 ± 3.490 |
| hard_FP_FN_evidence_difference | 89.610 ± 3.719 |
| typical_plus_hard_evidence_difference | 91.956 ± 3.522 |
Frozen model test metrics, mean±population SD (%) in ACC/Hard-AUC/Prob-AUC/SEN/SPE order: E74.4-G2 92.1891±2.7648/92.0294±2.7255/93.4946±2.9195/90.0671±3.8862/93.9917±4.7320; E77.1-G2 91.9592±2.7267/91.7957±2.6486/93.8660±2.6681/89.8171±3.0738/93.7743±4.7048.
The probe is a new supervised statistic fitted to frozen representations; it does not imply information theoretically independent of CORR/CAM or unbiased clinical generalization. The upstream E77.1 checkpoint itself remains test-ACC-selected.
After excluding Fold 2 (descriptive only), checkpoint means ± population SD (%) are: E74 ACC/Hard-AUC/Prob-AUC/SEN/SPE = 92.2153±2.9131/92.0481±2.8723/93.6142±3.0541/90.0474±4.0959/94.0487±4.9847%; E771 ACC/Hard-AUC/Prob-AUC/SEN/SPE = 92.0875±2.8454/91.9092±2.7687/93.9503±2.7997/89.7696±3.2366/94.0487±4.8828%. Probe paired ΔBA/ΔAUC excluding Fold 2 (pp): P1−P0 +2.449/-0.497; P3−P2 +0.809/-0.568.

## Q3: ReLU and gradient

For the test folds, FP has a valid prototype for 871/871 patient queries and 3484 valid patient-modality queries; FN has 871/871 and 3484. Among valid patient-modality queries, contrast<=0 / exact-zero value are FP 54.7%/54.7% and FN 63.6%/63.6%; these are query-level fractions, not patient-level fractions. Near-zero-positive contributions occur in 0.0% of FP positive values and 0.0% of FN positive values. Target hard-score clipping is 0.0%/3.7% of all test patient queries.
On allowed training queries the FP/FN valid-query zero-value fractions are 56.9%/66.3%; on test queries they are 54.7%/63.6%. Test modality zero-value fractions (PHENO/ANAT/FUNC/CORR) are FP 33.2%/58.4%/49.9%/77.4% and FN 42.0%/60.7%/57.2%/94.6%.
Direction-specific local derivative zero fractions on valid test queries are FP/FN 54.7%/63.6%; small-positive-gradient fractions are 0.0%/0.0%. On stable training memory records, frozen ranking loss is 0.10453; among positive per-patient ranking losses, 13.6% have zero contrast-path gradient.
A fixed-weight Softplus(beta=5) counterfactual, with the same gate and 0.5 saturation cap, changes hard-evidence coverage from 98.4% to 100.0%, test true-class margin by +0.00822 on average, and changes predictions wrong-to-right/right-to-wrong 0/0. This is an operator diagnosis only, not trained-model performance.

## Hypothesis decisions and limits

| Core hypothesis | Result | Continue? |
|---|---|---|
| H1: historical errors are reproducible | PASS | YES |
| H2: prototypes add information beyond CAM | FAIL | NO |
| H3: ReLU truncation is the main failure | FAIL | NO |

Final decision: **NO_GO_PROTOTYPE**. H1=PASS supports descriptive recurrence only, while H2=FAIL found no stable conditional increment beyond CAM; stop adding prototype/Prompt modules under this representation. H3=FAIL does not change that conclusion.
Limitations preserved: Fold-2 train/test overlap IDs 377 and 407; `val=test` checkpoint choice; pre-CV RFE; transductive full-graph features. The Fold-2-excluded nine-fold summaries above are descriptive only. No claim of independent generalization or cross-seed stability is made.
All source/checkpoint/data hashes were identical before and after the audit. No model or checkpoint was saved or modified.

Main artifacts: `summary.csv`, `probe.csv`, `details.json`, `manifest.json`.

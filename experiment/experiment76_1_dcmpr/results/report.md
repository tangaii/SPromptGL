# Experiment 76.1 — DCMP-R

Status: independently audited after reconstructing all best checkpoints from disk.

Audit-only repair: Removed a duplicate test-index selection from the role-attention diagnostic; the full attention tensor had already been restricted to test rows. Audit script SHA changed from `31034b490ec81cc9fab17f1ff3daa9cef326113bb7c88dd2a6b11fe519dd2419` to `0d97b921eeee01f7b4411e7179b2655e6595834c65d578fca46ff1192c6e2f8d`. Training code/configuration and all locked historical inputs retained their preflight SHA; no model was retrained.

## Literature and source audit

The implementation was checked against SPromptGL's GFR/CAM/PEN and classifier path, and the locked E74.4-G2-Positive-DCR runner. It keeps one CAM call, captures the existing 144-dimensional classifier input, and applies the refined module immediately before the unchanged OutputLayer. BrainPrompt's official ABIDE prompt source lists TD/TC before ASD; the reused frozen CLIP RN50 cache is mapped to ASD=0, TD=1. This is not a reproduction of BrainPrompt's original language encoder. CoW motivates support self-prediction and error-driven hard prototypes; PGP-SAM motivates inter-/intra-class prototypes and prompt-conditioned features. HP²L is directly related at the broad prototype-prompt level; this experiment's distinct hypothesis is bidirectional historical confusion memory and selective patient correction, not hierarchical disorder classification.

Sources: [SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [BrainPrompt paper](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [BrainPrompt official ABIDE prompts](https://github.com/AngusMonroe/BrainPrompt/blob/main/data/prompts/label_prompts/abide_label.txt), [PGP-SAM paper](https://arxiv.org/abs/2501.06692), [PGP-SAM code](https://github.com/PRIS-CV/PGP-SAM), [CoW paper](https://ijcai.org/proceedings/2025/139), [CoW code](https://github.com/jcjiang99/CoW), [HP²L DOI](https://doi.org/10.1016/j.media.2026.104063).

## Method and protocol

All three groups share identical DCMP-R parameter shapes and per-fold initialization. G1 retrieves class-balanced typical ASD/TD memories; G2 adds hard prototypes only from actual 5-fold OOF LogisticRegression probe errors, capped at 20% of each true class; G3 matches G2 but replaces disease text with norm- and class-cosine-matched anonymous vectors. The probe is cross-fitted on the frozen warmup CAM144 features, so this is OOF only for the probe and not strict OOF for the shared deep encoder. Typical and hard memories update from detached same-step CAM144 features using EMA=0.9 and deterministically re-cluster every 20 epochs inside fixed legal pools. Query routing does not read query labels; training queries are removed from their own cluster sums and counts. The added memory excludes all outer test IDs and, in Fold 2, rows 377/407 despite their inherited base train/test overlap.
A role-balanced two-stage retrieval uses cosine logits divided by fixed tau=0.10, with role-local prototype attention and a learned role-confidence composer. Explicit ASD/TD evidence differences and typical/hard conflicts feed a shared 144-D feature correction. Correction is capped at 5% of the CAM144 norm; hard-margin and easy-preservation terms use fixed training-only probe masks.

Checkpoint selection retains historical val=test strict-ACC selection, pre-CV RFE/standardization, and the transductive 871-node graph. Metrics are therefore paired mechanism results under a known biased protocol, not unbiased generalization estimates.

## Results

All values are mean ± population SD in percent. Hard-AUC is the historical hard-prediction AUC/BA; Prob-AUC is continuous-score ROC-AUC.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 91.9592 ± 2.8219 | 91.7901 ± 2.8294 | 93.6198 ± 3.1466 | 89.5793 ± 4.5762 | 94.0009 ± 4.3852 | 91.1605 ± 3.0837 |
| G2 | 91.9606 ± 2.5243 | 91.7955 ± 2.5262 | 93.6858 ± 2.5728 | 89.5854 ± 3.4710 | 94.0056 ± 3.3042 | 91.1683 ± 2.7454 |
| G3 | 91.9592 ± 2.8219 | 91.7715 ± 2.8154 | 93.5736 ± 2.9566 | 89.3293 ± 3.7140 | 94.2137 ± 3.8661 | 91.1484 ± 3.0461 |
| E74.4-G2-Positive-DCR | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 | 93.4946 ± 2.9195 | 90.0671 ± 3.8862 | 93.9917 ± 4.7320 | 91.4590 ± 2.9013 |

## Paired comparisons

Comparisons use within-fold differences; confidence intervals are descriptive paired t intervals.

| Comparison | Metric | Δ pp | 95% CI pp | W/T/L |
|---|---|---:|---:|---:|
| G1 vs E74.4-G2 | acc | -0.2299 | [-0.5766, 0.1168] | 0/8/2 |
| G1 vs E74.4-G2 | hard_auc | -0.2393 | [-0.7756, 0.2970] | 1/6/3 |
| G1 vs E74.4-G2 | prob_auc | 0.1252 | [-0.3670, 0.6174] | 6/0/4 |
| G1 vs E74.4-G2 | sen | -0.4878 | [-4.3247, 3.3491] | 1/6/3 |
| G1 vs E74.4-G2 | spe | 0.0093 | [-2.9567, 2.9752] | 3/6/1 |
| G2 vs E74.4-G2 | acc | -0.2286 | [-0.8761, 0.4190] | 2/4/4 |
| G2 vs E74.4-G2 | hard_auc | -0.2339 | [-0.9129, 0.4451] | 2/3/5 |
| G2 vs E74.4-G2 | prob_auc | 0.1912 | [-0.3637, 0.7461] | 6/1/3 |
| G2 vs E74.4-G2 | sen | -0.4817 | [-2.8202, 1.8568] | 1/5/4 |
| G2 vs E74.4-G2 | spe | 0.0139 | [-2.0404, 2.0682] | 2/5/3 |
| G3 vs E74.4-G2 | acc | -0.2299 | [-0.5766, 0.1168] | 0/8/2 |
| G3 vs E74.4-G2 | hard_auc | -0.2579 | [-0.6034, 0.0876] | 0/7/3 |
| G3 vs E74.4-G2 | prob_auc | 0.0790 | [-0.3065, 0.4644] | 7/0/3 |
| G3 vs E74.4-G2 | sen | -0.7378 | [-1.9205, 0.4449] | 0/8/2 |
| G3 vs E74.4-G2 | spe | 0.2220 | [-0.9200, 1.3640] | 1/8/1 |
| G2 vs G1 | acc | 0.0013 | [-0.5453, 0.5479] | 2/6/2 |
| G2 vs G1 | hard_auc | 0.0054 | [-0.5055, 0.5162] | 4/3/3 |
| G2 vs G1 | prob_auc | 0.0660 | [-0.7509, 0.8829] | 4/1/5 |
| G2 vs G1 | sen | 0.0061 | [-2.2206, 2.2328] | 2/5/3 |
| G2 vs G1 | spe | 0.0046 | [-2.2668, 2.2760] | 3/4/3 |
| G2 vs G3 | acc | 0.0013 | [-0.5453, 0.5479] | 2/6/2 |
| G2 vs G3 | hard_auc | 0.0240 | [-0.5854, 0.6333] | 3/3/4 |
| G2 vs G3 | prob_auc | 0.1122 | [-0.2573, 0.4818] | 5/0/5 |
| G2 vs G3 | sen | 0.2561 | [-1.8806, 2.3928] | 2/5/3 |
| G2 vs G3 | spe | -0.2081 | [-1.7258, 1.3095] | 3/4/3 |

## Fold-2-Excluded Sensitivity

These are descriptive nine-fold summaries after omitting Fold 2; they do not remove the remaining historical evaluation biases.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 91.9598 ± 2.9746 | 91.7822 ± 2.9823 | 93.7239 ± 3.3004 | 89.5054 ± 4.8181 | 94.0590 ± 4.6187 | 91.1385 ± 3.2497 |
| G2 | 92.0890 ± 2.6297 | 91.9237 ± 2.6318 | 93.7972 ± 2.6890 | 89.7832 ± 3.6049 | 94.0641 ± 3.4780 | 91.2981 ± 2.8647 |
| G3 | 91.9598 ± 2.9746 | 91.7615 ± 2.9675 | 93.6666 ± 3.1026 | 89.2276 ± 3.9017 | 94.2954 ± 4.0670 | 91.1251 ± 3.2100 |
| E74.4-G2-Positive-DCR | 92.2153 ± 2.9131 | 92.0481 ± 2.8723 | 93.6142 ± 3.0541 | 90.0474 ± 4.0959 | 94.0487 ± 4.9847 | 91.4703 ± 3.0580 |

Paired differences below are G2 minus the named comparator, in percentage points; W/T/L follows the same order.

| Comparison | Metric | Δ pp | W/T/L |
|---|---|---:|---:|
| G2 vs E74.4-G2 | ACC | -0.1263 | 2/4/3 |
| G2 vs E74.4-G2 | Hard-AUC | -0.1244 | 2/3/4 |
| G2 vs E74.4-G2 | Prob-AUC | +0.1830 | 5/1/3 |
| G2 vs E74.4-G2 | SEN-ASD | -0.2642 | 1/5/3 |
| G2 vs E74.4-G2 | SPE-TD | +0.0154 | 2/4/3 |
| G2 vs E74.4-G2 | ASD-F1 | -0.1721 | 2/3/4 |
| G2 vs G1 | ACC | +0.1292 | 2/6/1 |
| G2 vs G1 | Hard-AUC | +0.1415 | 4/3/2 |
| G2 vs G1 | Prob-AUC | +0.0733 | 4/0/5 |
| G2 vs G1 | SEN-ASD | +0.2778 | 2/5/2 |
| G2 vs G1 | SPE-TD | +0.0051 | 3/3/3 |
| G2 vs G1 | ASD-F1 | +0.1596 | 4/3/2 |
| G2 vs G3 | ACC | +0.1292 | 2/6/1 |
| G2 vs G3 | Hard-AUC | +0.1621 | 3/3/3 |
| G2 vs G3 | Prob-AUC | +0.1306 | 5/0/4 |
| G2 vs G3 | SEN-ASD | +0.5556 | 2/5/2 |
| G2 vs G3 | SPE-TD | -0.2313 | 3/3/3 |
| G2 vs G3 | ASD-F1 | +0.1730 | 3/3/3 |

## Mechanism findings

Independent checkpoint audits: 30/30. Maximum test residual fraction: 0.042452; mean test residual fraction: 0.021438. These fractions are direct measured ‖delta‖/‖h‖, not a nominal cap.
Correction-off hard flips across all folds/groups: 1. Fold 2 exposed rows were checked in every intervention; affected counts are reported in `interventions.csv`.
Shared-epoch base RNG comparisons: 10950; exact triplets: 10950; mismatches: 0.

G2−G1 tests the added bidirectional hard-memory contribution. G2−G3 tests whether the real BrainPrompt disease vectors outperform the geometry-matched anonymous control. Fixed-checkpoint correction-off, confusion-memory-off, and text-condition-off interventions are diagnostic and do not replace the trained group comparisons.

## Decision and limitations

Scientific decision: **NO-GO**. G2 did not recover the E74.4-G2 ACC or Hard-AUC, G2−G1 and G2−G3 changes were negligible and inconsistent, and disabling G2 correction or confusion memory produced no hard-prediction flips. The decision in `audit_status.json` follows the pre-registered targets: base-level recovery, G2 gain over G1 with net hard-case corrections, and G2 gain over anonymous G3. This single-seed protocol retains val=test checkpoint selection, pre-CV preprocessing, a transductive graph, and Fold 2 overlap, so it cannot support an unbiased generalization claim. The warmup encoder has seen outer-training features when probe OOF errors are generated. Follow-up should prioritize leakage-controlled independent validation rather than expand prototype searches.
